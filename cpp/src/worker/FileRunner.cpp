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

#include "worker/FileRunner.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <unordered_map>

#include "bam/BaiIndex.h"
#include "bam/BamReader.h"
#include "utils/Utils.h"
#include "worker/FileWorker.h"

namespace deeprm {
  void FileRunner::harvest_worker(int worker_id, int num_workers, const Arguments& args,
                                  vector<pair<Uuid128, uint64_t>>& out, bool ranged,
                                  uint64_t v_lo, uint64_t v_hi, atomic<bool>& failed)
  {
    log_info() << "Starting BAM worker " << worker_id << endl;
    BamReader reader(args.bam_path, args.filter_flag);
    if (ranged)
      reader.set_voffset_range(v_lo, v_hi);
    out = reader.harvest_offsets(worker_id, num_workers, args.bam_threads);
    if (reader.failed())
      failed.store(true, memory_order_relaxed);
    log_info() << "BAM worker " << worker_id << " harvested " << out.size() << " records"
               << endl;
  }

  vector<pair<uint64_t, uint64_t>> FileRunner::plan_harvest(int num_bam_workers) const
  {
    // Fast path: partition the BAM into contiguous voffset ranges from the
    // .bai linear index so each worker decompresses only its slice. The Python
    // preprocessing's conditions and fallbacks apply: index present and not older than the
    // BAM, header coordinate-sorted, and enough split points to actually
    // parallelise -- anything else falls back to the modulo scan, which is
    // correct and merely slower.
    auto [fast_ok, why] = BaiIndex::fast_path_ok(args.bam_path);
    if (!fast_ok) {
      log_info() << "Phase 1 voffset fast path off (" << why << "); using the modulo scan"
                 << endl;
      return {};
    }
    vector<pair<uint64_t, uint64_t>> ranges =
        BaiIndex::plan_ranges(BaiIndex::linear_offsets(args.bam_path), num_bam_workers);
    // A short reference set gives too few split points to fill the workers;
    // the voffset path then runs on ~1 worker and is SLOWER than modulo (3.16x
    // measured in the Python preprocessing). Use whichever parallelises.
    if (!ranges.empty() && static_cast<int>(ranges.size()) < max(2, num_bam_workers / 2)) {
      log_info() << "Phase 1: only " << ranges.size() << " usable ranges for "
                 << num_bam_workers << " workers (short reference set); using the modulo "
                 << "scan instead, which parallelises by record" << endl;
      ranges.clear();
    }
    return ranges;
  }

  // Spawns the voffset harvest; routing against the assignment happens in
  // run(), after both the harvest and the POD5 scan finish.
  void FileRunner::start_bam()
  {
    log_info() << "Running in file mode - harvesting all BAM records first, "
               << "then processing in POD5 order" << endl;
    // At least one harvest worker, as the Python preprocessing: with -t below -a the division is 0, and
    // a run with no harvest worker would emit nothing and still exit 0.
    int num_bam_workers = max(1, args.cpu_count / args.bam_threads);
    vector<pair<uint64_t, uint64_t>> ranges = plan_harvest(num_bam_workers);
    bool ranged = !ranges.empty();
    int spawn_workers = ranged ? static_cast<int>(ranges.size()) : num_bam_workers;
    harvest_results.resize(spawn_workers);
    if (ranged) {
      log_info() << "Phase 1: voffset partition, " << ranges.size() << " ranges over "
                 << spawn_workers << " workers, overlapped with the POD5 scan" << endl;
    } else {
      log_info() << "Starting BAM harvest with " << spawn_workers
                 << " workers (modulo scan), overlapped with the POD5 scan" << endl;
    }
    for (int i = 0; i < spawn_workers; ++i) {
      uint64_t lo = ranged ? ranges[i].first : 0;
      uint64_t hi = ranged ? ranges[i].second : 0;
      harvest_threads.emplace_back(harvest_worker, i, spawn_workers, cref(args),
                                   ref(harvest_results[i]), ranged, lo, hi, ref(harvest_failed));
    }
  }

  // Contiguous partition: by reader when there are at least as many files as
  // workers, by record otherwise.
  WorkerAssignment FileRunner::assign(Pod5Catalog& catalog)
  {
    if (static_cast<int>(catalog.files().size()) >= args.cpu_count)
      return WorkerAssignment::by_reader(catalog);
    return WorkerAssignment::flatten_split(catalog, args.cpu_count);
  }

  bool FileRunner::run(WorkerAssignment& assignment)
  {
    // Phase 1 has been running since start_bam(); routing needs the index.
    for (auto& t : harvest_threads)
      t.join();
    if (harvest_failed.load()) {
      log_err() << "BAM harvest failed: part of the BAM could not be read" << endl;
      return false;
    }
    size_t total_bam_records = 0;
    for (const auto& result : harvest_results)
      total_bam_records += result.size();
    log_info() << "BAM harvest completed. Total records: " << total_bam_records << endl;

    const int num_workers = assignment.num_workers();
    log_info() << "Starting file-mode processing with " << num_workers << " workers" << endl;

    // Index the harvested voffsets by parent_id, one map per worker, routed
    // by the assignment. Bucket order equals the old record order (worker id
    // order == file order under the voffset partition), so phase-2 output is
    // byte-identical to the extract-up-front implementation this replaced.
    // Unmatched records are dropped here, exactly as before.
    const auto& pod5_index = assignment.index();
    vector<unordered_map<Uuid128, vector<uint64_t>>> bam_index(num_workers);
    size_t unmatched_bam = 0;
    for (auto& result : harvest_results) {
      for (auto& [parent_id, voffset] : result) {
        auto owner = pod5_index.find(parent_id);
        if (owner == pod5_index.end()) {
          ++unmatched_bam;
          continue;
        }
        bam_index[owner->second][move(parent_id)].push_back(voffset);
      }
      result.clear();
      result.shrink_to_fit();
    }
    log_info() << "BAM voffsets indexed per worker, " << unmatched_bam
               << " dropped with no POD5 match" << endl;
    log_info() << "[MEM] bam_index partitioned: " << get_rss_mb() << " MB" << endl;

    // Set by any worker that could not open its BAM or a POD5 file, re-read
    // a record it was supposed to process, or read a read's signal; turns a
    // silently-incomplete run into a non-zero exit.
    atomic<bool> worker_failed{false};
    auto& worker_records = assignment.worker_records();
    vector<thread> threads;
    for (int i = 0; i < num_workers; ++i) {
      // Constructed on its own thread so the writer's buffers are allocated
      // where they are used.
      threads.emplace_back([&, i] {
        FileWorker worker(i, args, assignment.files(), worker_records[i], bam_index[i],
                          worker_failed);
        worker.run();
      });
    }
    for (auto& t : threads)
      t.join();
    return !worker_failed.load(memory_order_relaxed);
  }
}  // namespace deeprm
