/***************************************************************************************************
 *
 * Copyright (C) 2025 Genome4me Incorporated - All Rights Reserved.
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

#include <algorithm>
#include <iostream>
#include <thread>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <shared_mutex>
#include <filesystem>
#include <glob.h>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <sys/resource.h>

#include "args/ArgumentParser.h"
#include "bam/BamReader.h"
#include "pod5/Pod5Reader.h"
#include "merger/RecordMerger.h"
#include "merger/MergedDataWorker.h"
#include "sam/SamDispatcher.h"
#include "npz/NpzWriter.h"
#include "utils/Utils.h"

using namespace std;
using namespace deeprm;

vector<string> get_pod5_files(const string& directory)
{
  vector<string> files;
  string pattern = directory + "/*.pod5";
  glob_t glob_result;

  if (glob(pattern.c_str(), GLOB_TILDE, nullptr, &glob_result) == 0) {
    for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
      files.emplace_back(glob_result.gl_pathv[i]);
    }
  }
  sort(files.begin(), files.end());

  globfree(&glob_result);
  return files;
}

void process_pod5_meta_worker(int worker_id, const vector<string>& pod5_files,
                              vector<vector<Pod5RecordMeta>>& pod5_meta_records)
{
  log_info() << "Starting POD5 metadata worker " << worker_id << endl;

  // Reserve space for expected number of files
  pod5_meta_records.reserve(pod5_files.size());

  // Reuse this vector to avoid repeated allocations
  vector<Pod5RecordMeta> file_meta_records;

  // Process each POD5 file separately to maintain file-level structure
  int file_index = 0;
  for (const string& pod5_file : pod5_files) {
    file_index++;
    log_info() << "POD5 metadata worker " << worker_id << " processing " << file_index
        << "/" << pod5_files.size() << " files" << endl;
    Pod5Reader pod5_reader(pod5_file);
    file_meta_records.clear(); // Clear for reuse

    if (!pod5_reader.parse_single_pod5_meta(pod5_file, file_meta_records)) {
      log_err() << "Corrupted POD5 file: " << pod5_file << endl;
      continue;
    }

    // Add this file's metadata records as a separate vector
    pod5_meta_records.push_back(move(file_meta_records));
  }

  int total_records = 0;
  for (const auto& file_records : pod5_meta_records) {
    total_records += file_records.size();
  }
}

void process_bam_worker(int worker_id, int num_workers, const Arguments& args,
                        vector<BamRecord>& bam_records,
                        unordered_map<string, int>& ref_index_dict)
{
  log_info() << "Starting BAM worker " << worker_id << endl;

  BamReader reader(args.bam_path, args.qcut, args.base_of_interest,
                   args.filter_flag, ref_index_dict);
  auto records = reader.parse_bam(worker_id, num_workers, args.bam_threads);

  size_t record_count = records.size(); // Save size before move
  bam_records = move(records);
  log_info() << "BAM worker " << worker_id << " processed " << record_count << " records" << endl;
}

// Process POD5 records in batches, looking up and erasing bam_index entries
// incrementally to free memory as processing progresses.
// Thread safety: shared_mutex guards bam_index - shared lock for find,
// exclusive lock for erase.
void process_merged_data_meta_worker(int worker_id, const Arguments& args,
                                     const vector<Pod5RecordMeta>& pod5_meta_records,
                                     unordered_map<string, vector<BamRecord>>& bam_index,
                                     shared_mutex& bam_index_mutex)
{
  log_info() << "Starting record merger worker " << worker_id
      << " with " << pod5_meta_records.size() << " POD5 records" << endl;

  // Per-worker POD5 readers, opened lazily for thread-safe access
  unordered_map<string, Pod5FileReader_t*> worker_readers;

  // Set normalization factors same as Python code
  NormalizationFactors norm_factors;

  // Create NPZ writer
  NpzWriter writer(args.output_path, args.chunk_size, worker_id);

  if (pod5_meta_records.empty()) {
    writer.flush();
    log_info() << "Record merger worker " << worker_id
        << " completed (no records)" << endl;
    return;
  }

  // Process pod5_meta_records in batches of process_once, looking up
  // bam_index directly instead of building a full merged_records vector.
  int output_index = 0;
  size_t current_idx = 0;

  while (current_idx < pod5_meta_records.size()) {
    size_t batch_end = min(current_idx + static_cast<size_t>(args.process_once),
                           pod5_meta_records.size());

    // Build matched pairs for this batch only
    vector<Pod5RecordMeta> batch_pod5_meta;
    vector<BamRecord> batch_bam;
    vector<string> keys_to_erase;

    // Create merged records for this batch (same as Python's pd.merge).
    // Move all BAM records with matching read_id.
    // Safe: each read_id belongs to exactly one worker via pod5_index,
    // so moved-from entries are never accessed by another worker.
    {
      shared_lock<shared_mutex> rlock(bam_index_mutex);
      for (size_t i = current_idx; i < batch_end; ++i) {
        const auto& pod5_meta = pod5_meta_records[i];
        auto bam_it = bam_index.find(pod5_meta.read_id);
        if (bam_it != bam_index.end()) {
          for (auto& bam_rec : bam_it->second) {
            // Lazy-open per-worker reader for thread-safe POD5 access
            Pod5RecordMeta meta_copy = pod5_meta;
            auto [rit, inserted] = worker_readers.try_emplace(meta_copy.file_path, nullptr);
            if (inserted) {
              rit->second = pod5_open_file(meta_copy.file_path.c_str());
              if (!rit->second) {
                log_err() << "Record merger worker " << worker_id
                          << " failed to open POD5: "
                          << meta_copy.file_path << endl;
              }
            }
            if (!rit->second)
              continue;
            meta_copy.reader = rit->second;
            batch_pod5_meta.push_back(move(meta_copy));
            batch_bam.push_back(move(bam_rec));
          }
          keys_to_erase.push_back(pod5_meta.read_id);
        }
      }
    }

    current_idx = batch_end;

    if (!batch_pod5_meta.empty()) {
      output_index++;

      // Create merger and process
      RecordMerger merger(norm_factors, args.cb_len, args.kmer_len, args.max_token_len,
                          args.sampling, args.dwell_shift, args.sig_window, args.label_div);

      merger.add_bam_records(move(batch_bam));
      merger.add_pod5_meta_records(move(batch_pod5_meta));

      vector<ProcessedRecord> processed_records = merger.merge_and_process_with_meta();

      if (!processed_records.empty()) {
        // Write to NPZ
        writer.add_records(move(processed_records));
      }

      writer.increment_processing_unit();
    }

    // Erase consumed bam_index entries to free memory progressively
    if (!keys_to_erase.empty()) {
      unique_lock<shared_mutex> wlock(bam_index_mutex);
      for (const auto& key : keys_to_erase) {
        bam_index.erase(key);
      }
    }
  }

  // Flush remaining records (same as Python code)
  writer.flush();

  // Close per-worker POD5 readers
  for (auto& [path, reader] : worker_readers)
    if (reader)
      pod5_close_and_free_reader(reader);

  log_info() << "Record merger worker " << worker_id << " completed" << endl;
}


int main(int argc, char* argv[])
{
  auto start_time = chrono::high_resolution_clock::now();
  log_start_time();

  // Parse arguments
  ArgumentParser parser;
  Arguments args = parser.parse(argc, argv);
  parser.validate_arguments();

  struct rlimit rlim;
  if (getrlimit(RLIMIT_NOFILE, &rlim) == 0) {
    rlim.rlim_cur = rlim.rlim_max;
    if (setrlimit(RLIMIT_NOFILE, &rlim) == 0) {
      log_info() << "File descriptor limit set to: " << rlim.rlim_cur << endl;
    } else {
      log_err() << "Warning: Failed to set file descriptor limit" << endl;
    }
  }

  log_info() << "Started DeepRM Preprocessing" << endl;

  // Get POD5 files
  vector<string> pod5_files = get_pod5_files(args.pod5_path);
  if (pod5_files.empty()) {
    log_err() << "No POD5 files found in directory: " << args.pod5_path << endl;
    return 1;
  }

  log_info() << "Found " << pod5_files.size() << " POD5 files" << endl;

  // Step 1: Start SAM dispatcher and POD5 metadata workers concurrently

  // Start SAM dispatcher
  SamDispatcher sam_dispatcher(args);

  // Check if consistency mode is enabled
  if (!args.consistency) {
    // Normal mode: Use streaming approach with SAM dispatcher
    log_info() << "Running in normal mode - streaming BAM records" << endl;
    sam_dispatcher.start();
  }

  // Process POD5 files with multiple workers (metadata only)
  // Use min(num_files, cpu_count) threads for reading, then redistribute
  int num_pod5_readers = min(static_cast<int>(pod5_files.size()), args.cpu_count);
  vector<thread> pod5_threads;
  vector<vector<vector<Pod5RecordMeta>>> pod5_reader_records(num_pod5_readers);

  log_info() << "Starting POD5 metadata reading with "
      << num_pod5_readers << " reader threads" << endl;

  // Calculate base size and remainder like np.array_split for reading
  size_t base_size = pod5_files.size() / num_pod5_readers;
  size_t remainder = pod5_files.size() % num_pod5_readers;

  size_t current_idx = 0;
  for (int i = 0; i < num_pod5_readers; ++i) {
    // First 'remainder' readers get one extra file
    size_t reader_file_count = base_size + (i < static_cast<int>(remainder) ? 1 : 0);

    if (current_idx < pod5_files.size()) {
      vector<string> reader_files(
        pod5_files.begin() + current_idx,
        pod5_files.begin() + current_idx + reader_file_count
      );
      pod5_threads.emplace_back(process_pod5_meta_worker, i, reader_files,
                                ref(pod5_reader_records[i]));
      current_idx += reader_file_count;
    }
  }

  // Wait for POD5 readers to complete
  for (auto& t : pod5_threads) {
    t.join();
  }

  // Count total POD5 records across all readers
  unsigned long count_pod5_records = 0;
  for (const auto& reader_files : pod5_reader_records) {
    for (const auto& file_records : reader_files) {
      count_pod5_records += file_records.size();
    }
  }

  log_info() << "POD5 metadata reading completed. Total records: "
      << count_pod5_records << endl;

  // Step 2: Assign POD5 records to workers
  // Strategy depends on whether pod5 files >= cpu_count
  int num_workers;
  vector<vector<Pod5RecordMeta>> worker_pod5_records;

  if (static_cast<int>(pod5_files.size()) >= args.cpu_count) {
    // pod5 >= CPU: use reader-level assignment directly (skip flatten/redistribute)
    num_workers = num_pod5_readers;
    worker_pod5_records.resize(num_workers);

    log_info() << "POD5 files (" << pod5_files.size()
        << ") >= CPUs (" << args.cpu_count
        << "): direct reader assignment with "
        << num_workers << " workers" << endl;

    for (int i = 0; i < num_pod5_readers; ++i) {
      size_t count = 0;
      for (const auto& file_records : pod5_reader_records[i]) {
        count += file_records.size();
      }
      worker_pod5_records[i].reserve(count);
      for (auto& file_records : pod5_reader_records[i]) {
        worker_pod5_records[i].insert(
            worker_pod5_records[i].end(),
            make_move_iterator(file_records.begin()),
            make_move_iterator(file_records.end()));
      }
    }
  } else {
    // pod5 < CPU: flatten all records, then redistribute among cpu_count workers
    num_workers = args.cpu_count;

    log_info() << "POD5 files (" << pod5_files.size()
        << ") < CPUs (" << args.cpu_count
        << "): redistributing records among "
        << num_workers << " workers" << endl;

    vector<Pod5RecordMeta> all_pod5_records;
    all_pod5_records.reserve(count_pod5_records);

    for (auto& reader_files : pod5_reader_records) {
      for (auto& file_records : reader_files) {
        all_pod5_records.insert(
            all_pod5_records.end(),
            make_move_iterator(file_records.begin()),
            make_move_iterator(file_records.end()));
      }
    }

    worker_pod5_records.resize(num_workers);

    size_t records_base_size = all_pod5_records.size() / num_workers;
    size_t records_remainder = all_pod5_records.size() % num_workers;

    current_idx = 0;
    for (int i = 0; i < num_workers; ++i) {
      size_t worker_record_count =
          records_base_size + (i < static_cast<int>(records_remainder) ? 1 : 0);

      worker_pod5_records[i].reserve(worker_record_count);
      for (size_t j = 0;
           j < worker_record_count && current_idx < all_pod5_records.size();
           ++j, ++current_idx) {
        worker_pod5_records[i].push_back(move(all_pod5_records[current_idx]));
      }
    }
  }

  pod5_reader_records.clear();
  pod5_reader_records.shrink_to_fit();

  log_info() << "[MEM] POD5 redistributed: " << get_rss_mb() << " MB" << endl;

  // Create POD5 index (read_id -> worker_id mapping)
  log_info() << "Creating POD5 index for merged data workers" << endl;
  unordered_map<string, int> pod5_index;
  pod5_index.reserve(count_pod5_records);
  for (int worker_id = 0; worker_id < num_workers; ++worker_id) {
    for (const auto& record : worker_pod5_records[worker_id]) {
      pod5_index[record.read_id] = worker_id;
    }
  }

  log_info() << "[MEM] pod5_index created: " << get_rss_mb() << " MB" << endl;

  // Check if consistency mode is enabled
  if (args.consistency) {
    // Consistency mode: Read all BAM records first with multiple workers, and distribute to workers
    log_info() << "Running in consistency mode - reading all BAM records first" << endl;

    // Parse BAM file with multiple workers
    int num_bam_workers = args.cpu_count / args.bam_threads;
    vector<thread> bam_threads;
    vector<vector<BamRecord>> bam_results(num_bam_workers);

    log_info() << "Starting BAM parsing with " << num_bam_workers << " workers" << endl;

    samFile* bam_file = sam_open(args.bam_path.c_str(), "r");
    if (!bam_file) {
      log_err() << "Failed to open BAM file: " << args.bam_path << endl;
      return 1;
    }

    sam_hdr_t* header = sam_hdr_read(bam_file);
    if (!header) {
      log_err() << "Failed to read BAM header" << endl;
      sam_close(bam_file);
      return 1;
    }

    // Build reference index dictionary
    unordered_map<string, int> ref_index_dict;
    for (int i = 0; i < header->n_targets; ++i) {
      ref_index_dict[string(header->target_name[i])] = i;
    }

    sam_hdr_destroy(header);
    sam_close(bam_file);

    for (int i = 0; i < num_bam_workers; ++i) {
      bam_threads.emplace_back(process_bam_worker, i, num_bam_workers, ref(args),
                               ref(bam_results[i]), ref(ref_index_dict));
    }

    // Wait for BAM workers to complete
    for (auto& t : bam_threads) {
      t.join();
    }

    // Merge BAM results
    vector<BamRecord> entire_bam_records;
    // Pre-calculate total size for reserve
    size_t total_bam_records = 0;
    for (const auto& result : bam_results) {
      total_bam_records += result.size();
    }
    entire_bam_records.reserve(total_bam_records);

    for (const auto& result : bam_results) {
      entire_bam_records.insert(entire_bam_records.end(), make_move_iterator(result.begin()),
                                make_move_iterator(result.end()));
    }

    log_info() << "BAM parsing completed. Total records: " << entire_bam_records.size() << endl;

    // Step 3: Process merged data - one thread per worker for full CPU utilization
    vector<thread> merge_threads;

    log_info() << "Starting merged data processing with "
        << num_workers << " workers" << endl;

    // Index BAM records by parent_id - multiple BAM records can have same read_id
    unordered_map<string, vector<BamRecord>> bam_index;
    for (auto& bam_rec : entire_bam_records) {
      bam_index[bam_rec.parent_id].push_back(move(bam_rec));
    }
    entire_bam_records.clear();
    entire_bam_records.shrink_to_fit();

    // Mutex for thread-safe bam_index erase during incremental processing
    shared_mutex bam_index_mutex;

    for (int i = 0; i < num_workers; ++i) {
      // Each merge worker processes its assigned POD5 records
      merge_threads.emplace_back(process_merged_data_meta_worker, i, ref(args),
                                 cref(worker_pod5_records[i]), ref(bam_index),
                                 ref(bam_index_mutex));
    }

    // Wait for merge workers to complete
    for (auto& t : merge_threads) {
      t.join();
    }

    // Stop SAM dispatcher (not used in consistency mode, but need to clean up)
    sam_dispatcher.stop();
  } else {
    // Create merged data workers - use all CPUs
    vector<MergedDataWorker*> merged_workers;
    for (int i = 0; i < num_workers; ++i) {
      auto worker = new MergedDataWorker(i, args, worker_pod5_records[i]);
      worker->start();
      merged_workers.push_back(worker);
    }

    log_info() << "Started " << num_workers << " merged data workers" << endl;
    log_info() << "[MEM] workers created: " << get_rss_mb() << " MB" << endl;

    // Set workers for SAM dispatcher
    sam_dispatcher.set_workers(&merged_workers, &pod5_index);

    log_info() << "[MEM] dispatch started: " << get_rss_mb() << " MB" << endl;

    // Wait for SAM dispatcher to finish
    sam_dispatcher.stop();
    log_info() << "[MEM] dispatch completed: " << get_rss_mb() << " MB" << endl;

    // Signal all workers to stop (notify them to process final batches)
    for (auto worker : merged_workers) {
      worker->signal_stop();
    }

    // Wait for all workers to finish processing final batches
    for (auto worker : merged_workers) {
      worker->wait_for_completion();
      delete worker;
    }
  }

  auto end_time = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
  double runtime_minutes = duration.count() / 60000.0;

  log_info() << "Finished DeepRM Preprocessing" << endl;
  log_info() << fixed << setprecision(2)
      << "Total runtime: " << runtime_minutes << " minutes" << endl;
  return 0;
}
