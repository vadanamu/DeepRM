/**
 * Copyright (C) 2026 Genome4me Incorporated - All Rights Reserved.
 * This software, including its source code, embedded concepts, and associated
 * documentation, is proprietary to Genome4me Incorporated and is protected
 * under trade secret and copyright law. Unauthorized use, copying, modification,
 * distribution, or disclosure to third parties, in whole or in part, is
 * strictly prohibited unless prior written permission is granted by Genome4me
 * Incorporated. Any such unauthorized actions constitute an infringement of
 * the intellectual property rights of Genome4me Incorporated. For licensing
 * inquiries or permissions, please contact Genome4me Incorporated.
 */

#include "worker/FileWorker.h"

#include <iostream>
#include <utility>

#include <htslib/bgzf.h>

#include "bam/BamRecordExtractor.h"
#include "merger/RecordMerger.h"
#include "utils/Utils.h"

namespace deeprm {
  FileWorker::FileWorker(int id, const Arguments& args, const vector<string>& pod5_files,
                         const vector<Pod5RecordMeta>& pod5_meta_records,
                         unordered_map<Uuid128, vector<uint64_t>>& bam_index,
                         atomic<bool>& failed)
    : worker_id(id), args(args), pod5_files(pod5_files), pod5_meta_records(pod5_meta_records),
      bam_index(bam_index), failed(failed),
      readers(pod5_files.size(), nullptr), reader_tried(pod5_files.size(), false),
      bam_file(sam_open(args.bam_path.c_str(), "r")),
      bam_header(bam_file ? sam_hdr_read(bam_file) : nullptr),
      bam_read(bam_init1()),
      writer(args.output_path, args.chunk_size, id,
             NpzWriter::strides_from(args.cb_len, args.kmer_len, args.max_token_len,
                                     args.sampling, args.sig_window))
  {
    if (!bam_file || !bam_header) {
      // Every matched read this worker owns would otherwise be dropped silently
      // at the guard in run() and the run would still exit 0. Flag it so the
      // runner reports failure instead of shipping an incomplete "success".
      log_err() << "file worker " << worker_id
                << " cannot open BAM for lazy fetch: " << args.bam_path << endl;
      failed.store(true, memory_order_relaxed);
    }
  }

  FileWorker::~FileWorker()
  {
    // The cached batch refers into one of the readers: release it first.
    batch_cache.release();
    for (Pod5FileReader_t* reader : readers)
      if (reader)
        pod5_close_and_free_reader(reader);
    bam_destroy1(bam_read);
    if (bam_header)
      sam_hdr_destroy(bam_header);
    if (bam_file)
      sam_close(bam_file);
  }

  Pod5FileReader_t* FileWorker::reader_for(uint32_t file_index)
  {
    if (!reader_tried[file_index]) {
      reader_tried[file_index] = true;
      ++reader_opens;
      readers[file_index] = pod5_open_file(pod5_files[file_index].c_str());
      if (!readers[file_index]) {
        log_err() << "file worker " << worker_id << " failed to open POD5: "
                  << pod5_files[file_index] << endl;
        failed.store(true, memory_order_relaxed);
      }
    }
    return readers[file_index];
  }

  bool FileWorker::process_read(Pod5RecordMeta meta, const vector<uint64_t>& voffsets,
                                RecordMerger& merger, BamRecord& bam_rec)
  {
    // bam_index is keyed by parent_id, so every voffset here belongs to
    // meta.read_id by construction. Re-read the record, run the shared
    // extractor (the same predicate as the streaming dispatcher + worker), and
    // merge. The BamRecord is recycled across fetches: extract_record clears
    // its push_back-filled fields on entry.
    for (uint64_t voffset : voffsets) {
      if (bgzf_seek(bam_file->fp.bgzf, static_cast<int64_t>(voffset), SEEK_SET) < 0 ||
          sam_read1(bam_file, bam_header, bam_read) < 0) {
        // A record that harvest saw but phase 2 cannot re-read means the output
        // would be missing it -- fatal for file mode's output equivalence.
        log_err() << "file worker " << worker_id
                  << " failed to re-read BAM record at voffset " << voffset << endl;
        return false;
      }
      if (BamRecordExtractor::extract_record(bam_read, args.filter_flag, args.qcut,
                                             args.base_of_interest, args.use_trimming,
                                             args.trim_threshold, args.trim_penalty,
                                             args.sampling, bam_rec) &&
          !merger.process_merged_record_with_meta(bam_rec, meta, batch_cache, writer)) {
        // Signal that cannot be read is a missing read in the output.
        log_err() << "file worker " << worker_id << " failed to read POD5 signal for "
                  << uuid128_to_hex(meta.read_id) << " in " << pod5_files[meta.file_index]
                  << endl;
        return false;
      }
    }
    return true;
  }

  void FileWorker::run()
  {
    log_info() << "Starting file worker " << worker_id << " with "
               << pod5_meta_records.size() << " POD5 records" << endl;

    if (pod5_meta_records.empty()) {
      writer.flush();
      log_info() << "file worker " << worker_id << " completed (no records)" << endl;
      return;
    }

    // Set normalization factors same as Python code
    NormalizationFactors norm_factors;
    RecordMerger merger(norm_factors, args.cb_len, args.kmer_len, args.max_token_len,
                        args.sampling, args.dwell_shift, args.sig_window,
                        args.label_div, args.padding_mode, args.use_trimming);

    // Each read is processed and written the moment it is matched, so nothing
    // is accumulated here; the writer cuts files purely on -k boundaries.
    BamRecord bam_rec;
    long reads_processed = 0;
    for (const auto& pod5_meta : pod5_meta_records) {
      // Cooperative early stop: once any worker has failed, this run's output
      // will be discarded, so stop doing expensive work. A relaxed load of an
      // otherwise-unwritten line on the per-read loop costs nothing measurable.
      if (failed.load(memory_order_relaxed))
        break;

      // The map is this worker's alone, so the entry can be read in place and
      // erased when done. A read_id may carry several BAM records: a split read
      // shares one parent via the pi tag.
      auto bam_it = bam_index.find(pod5_meta.read_id);
      if (bam_it == bam_index.end())
        continue;

      Pod5FileReader_t* reader = reader_for(pod5_meta.file_index);
      if (reader && bam_file && bam_header) {
        Pod5RecordMeta meta = pod5_meta;
        meta.reader = reader;
        if (!process_read(move(meta), bam_it->second, merger, bam_rec)) {
          failed.store(true, memory_order_relaxed);
          break;
        }
        ++reads_processed;
      }

      // Consumed either way. A POD5 file that failed to open will not open on a
      // later read, so keeping its voffsets back would only hold them for the
      // rest of the run.
      bam_index.erase(bam_it);
    }

    // Flush remaining records (same as Python code)
    writer.flush();
    if (writer.write_failed()) {
      // counted in NpzWriter::files_not_written, which fails the run with its own cause
      log_err() << "file worker " << worker_id << " could not write an output chunk" << endl;
    }
    // Same shape as the stream worker's line so run summaries can parse both.
    log_info() << "file worker " << worker_id
               << " completed processing. Total reads: " << reads_processed
               << ", POD5 reader opens: " << reader_opens << endl;
  }
}  // namespace deeprm
