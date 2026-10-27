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

#pragma once

#include <atomic>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

#include "utils/Uuid128.h"
#include "worker/ModeRunner.h"

using namespace std;

namespace deeprm {
  // File mode (-b FILE). Phase 1 harvests (parent_id, voffset) for every BAM
  // record that passes the cheap pre-filter (non-empty SEQ, -g flag mask)
  // while the POD5 scan runs; phase 2 routes those by read_id to the
  // worker that owns the read and runs one FileWorker per worker, which
  // re-reads each record by voffset and processes in POD5 order.
  class FileRunner : public ModeRunner {
  public:
    explicit FileRunner(const Arguments& args) : args(args) {}
    void start_bam() override;
    WorkerAssignment assign(Pod5Catalog& catalog) override;
    bool run(WorkerAssignment& assignment) override;

  private:
    // One harvest thread's slice of the BAM: a voffset range from the .bai
    // fast path, or (ranged == false) the modulo scan.
    static void harvest_worker(int worker_id, int num_workers, const Arguments& args,
                               vector<pair<Uuid128, uint64_t>>& out, bool ranged,
                               uint64_t v_lo, uint64_t v_hi, atomic<bool>& failed);
    // Decides between the .bai voffset partition and the modulo scan.
    vector<pair<uint64_t, uint64_t>> plan_harvest(int num_bam_workers) const;

    const Arguments& args;
    vector<thread> harvest_threads;
    vector<vector<pair<Uuid128, uint64_t>>> harvest_results;
    // A harvest thread could not read its share of the BAM.
    atomic<bool> harvest_failed{false};
  };
}  // namespace deeprm

