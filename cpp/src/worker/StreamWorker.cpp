/***************************************************************************************************
 *
 * Copyright (C) 2025-2026 Genome4me Incorporated - All Rights Reserved.
 *
 * This software, including its source code, embedded concepts, and associated
 * documentation, is proprietary to Genome4me Incorporated and is protected
 * under trade secret and copyright law. Unauthorized use, copying, modification,
 * distribution, or disclosure to third parties, in whole or in part, is
 * strictly prohibited unless prior written permission is granted by Genome4me
 * Incorporated. Any such unauthorized actions constitute an infringement of
 * the intellectual property rights of Genome4me Incorporated. For licensing
 * inquiries or permissions, please contact Genome4me Incorporated.
 *
 **************************************************************************************************/

#include "worker/StreamWorker.h"

#include <algorithm>
#include <iostream>
#include <utility>

#include "bam/BamRecordExtractor.h"
#include "merger/RecordMerger.h"
#include "npz/NpzWriter.h"
#include "sam/SamDispatcher.h"
#include "utils/Utils.h"

namespace deeprm {
  StreamWorker::StreamWorker(int id, const Arguments& args, const vector<string>& pod5_files,
                             const vector<Pod5RecordMeta>& pod5_meta, SamDispatcher& dispatcher,
                             atomic<bool>& failed)
    : worker_id(id), args(args), pod5_files(pod5_files), pod5_meta_records(pod5_meta),
      dispatcher(dispatcher), failed(failed), is_running(false)
  {
  }

  StreamWorker::~StreamWorker()
  {
    stop();
  }

  void StreamWorker::start()
  {
    is_running = true;
    log_info() << "stream worker " << worker_id
        << " starting with " << pod5_meta_records.size()
        << " POD5 records" << endl;
    worker_thread = thread(&StreamWorker::process_loop, this);
  }

  void StreamWorker::stop()
  {
    is_running = false;
    queue_cv.notify_all();
    if (worker_thread.joinable()) {
      worker_thread.join();
    }
  }

  void StreamWorker::signal_stop()
  {
    // Under the queue lock: process_loop checks is_running and then waits while holding it, so a stop
    // signalled between its check and its wait would otherwise be lost and the run would hang.
    {
      lock_guard<mutex> lock(queue_mutex);
      is_running = false;
    }
    queue_cv.notify_all();
  }

  void StreamWorker::wait_for_completion()
  {
    if (worker_thread.joinable()) {
      worker_thread.join();
    }
  }

  void StreamWorker::add_bam_data(bam1_t* read)
  {
    unique_lock<mutex> lock(queue_mutex);
    bam_queue.push(read);
    queue_cv.notify_one();
  }

  void StreamWorker::process_loop()
  {
    // Row strides come from the token options once.
    NpzWriter writer(args.output_path, args.chunk_size, worker_id,
                     NpzWriter::strides_from(args.cb_len, args.kmer_len, args.max_token_len,
                                             args.sampling, args.sig_window));
    NormalizationFactors norm_factors;
    RecordMerger merger(norm_factors, args.cb_len, args.kmer_len,
                        args.max_token_len, args.sampling, args.dwell_shift,
                        args.sig_window, args.label_div, args.padding_mode,
                        args.use_trimming);

    // One POD5 reader at a time. Block round-robin hands this worker long runs
    // of reads from the same file, so the reader only has to be swapped when
    // the file changes.
    uint32_t held_file = 0;
    Pod5FileReader_t* held_reader = nullptr;
    long reader_opens = 0;
    // Read-table batch reused across reads of the same file and batch; it
    // refers into held_reader, so it is released before that reader closes.
    Pod5BatchCache batch_cache;

    unordered_map<Uuid128, Pod5RecordMeta> pod5_index;
    // Build POD5 index for this worker
    pod5_index.reserve(pod5_meta_records.size());
    for (const auto& record : pod5_meta_records) {
      pod5_index[record.read_id] = record;
    }
    log_info() << "stream worker " << worker_id
        << " indexed " << pod5_meta_records.size()
        << " POD5 records" << endl;

    // Reads processed so far, kept only for periodic progress logs; file
    // naming no longer carries a unit field, so nothing else consumes this.
    long reads_processed = 0;

    while (is_running || !bam_queue.empty()) {
      unique_lock<mutex> lock(queue_mutex);

      if (bam_queue.empty()) {
        if (!is_running) break;
        queue_cv.wait(lock);
        continue;
      }

      // Get BAM read from queue
      bam1_t* read = bam_queue.front();
      bam_queue.pop();
      lock.unlock();

      // Keep the dispatcher's global in-flight count in step with this pop.
      dispatcher.on_worker_pop();

      // Once any worker has failed the run's output is discarded, so stop
      // doing expensive work -- but keep draining: breaking out would leave
      // the in-flight count high and park the dispatcher forever.
      if (failed.load(memory_order_relaxed)) {
        bam_destroy1(read);
        continue;
      }

      // Extract parent_id from bam1_t for POD5 lookup
      uint8_t* pi_tag = bam_aux_get(read, "pi");
      const char* pid = pi_tag ? bam_aux2Z(pi_tag) : bam_get_qname(read);

      // Find matching POD5 record
      auto pod5_it = pod5_index.find(uuid128_from_str(pid));
      if (pod5_it != pod5_index.end()) {
        // Reuse the held reader while the file is unchanged; swap on change.
        auto& meta = pod5_it->second;
        if (!held_reader || meta.file_index != held_file) {
          if (held_reader) {
            batch_cache.release();
            pod5_close_and_free_reader(held_reader);
            held_reader = nullptr;
          }
          held_reader = pod5_open_file(pod5_files[meta.file_index].c_str());
          held_file = meta.file_index;
          ++reader_opens;
          if (!held_reader) {
            log_err() << "stream worker " << worker_id
                      << " failed to open POD5: " << pod5_files[meta.file_index] << endl;
            failed.store(true, memory_order_relaxed);
          }
        }
        if (!held_reader) {
          bam_destroy1(read);
          continue;
        }
        meta.reader = held_reader;

        BamRecord bam_record;
        if (BamRecordExtractor::extract_record(read, args.filter_flag, args.qcut,
                                               args.base_of_interest, args.use_trimming,
                                               args.trim_threshold, args.trim_penalty,
                                               args.sampling, bam_record) &&
            !merger.process_merged_record_with_meta(bam_record, meta, batch_cache,
                                                    writer)) {
          log_err() << "stream worker " << worker_id
                    << " failed to read POD5 signal for " << uuid128_to_hex(meta.read_id)
                    << " in " << pod5_files[meta.file_index] << endl;
          failed.store(true, memory_order_relaxed);
        }
        bam_destroy1(read);

        if (++reads_processed % 1000 == 0) {
          if (worker_id == 0) {
            log_info() << "[MEM] worker 0 reads " << reads_processed
                << " (queue=" << bam_queue.size() << ", opens=" << reader_opens << "): "
                << get_rss_mb() << " MB" << endl;
          }
          log_info() << "stream worker " << worker_id
              << " processed " << reads_processed << " reads" << endl;
        }
      } else {
        bam_destroy1(read);
      }
    }

    log_info() << "stream worker " << worker_id << " flushing remaining records" << endl;
    writer.flush();
    if (writer.write_failed()) {
      // counted in NpzWriter::files_not_written, which fails the run with its own cause
      log_err() << "stream worker " << worker_id << " could not write an output chunk" << endl;
    }

    batch_cache.release();
    if (held_reader)
      pod5_close_and_free_reader(held_reader);

    log_info() << "stream worker " << worker_id
        << " completed processing. Total reads: " << reads_processed
        << ", POD5 reader opens: " << reader_opens << endl;
  }
}
