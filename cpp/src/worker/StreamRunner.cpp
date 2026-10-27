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

#include "worker/StreamRunner.h"

#include <atomic>
#include <memory>
#include <vector>

#include "utils/Utils.h"
#include "worker/StreamWorker.h"

namespace deeprm {
  void StreamRunner::start_bam()
  {
    log_info() << "Running in streaming mode - processing BAM records in BAM "
               << "order as they arrive" << endl;
    dispatcher.start();
  }

  WorkerAssignment StreamRunner::assign(Pod5Catalog& catalog)
  {
    return WorkerAssignment::block_round_robin(catalog, args.cpu_count,
                                               kRankBlock);
  }

  bool StreamRunner::run(WorkerAssignment& assignment)
  {
    const int num_workers = assignment.num_workers();
    auto& worker_records = assignment.worker_records();
    // Any worker that cannot open a POD5 file or read a signal sets
    // worker_failed; checked after the join.
    atomic<bool> worker_failed{false};
    vector<unique_ptr<StreamWorker>> workers;
    workers.reserve(num_workers);
    for (int i = 0; i < num_workers; ++i) {
      workers.push_back(make_unique<StreamWorker>(i, args, assignment.files(),
                                                  worker_records[i], dispatcher, worker_failed));
      workers.back()->start();
    }
    log_info() << "Started " << num_workers << " stream workers" << endl;
    log_info() << "[MEM] workers created: " << get_rss_mb() << " MB" << endl;

    dispatcher.set_workers(workers, assignment.index());
    log_info() << "[MEM] dispatch started: " << get_rss_mb() << " MB" << endl;
    dispatcher.stop();
    log_info() << "[MEM] dispatch completed: " << get_rss_mb() << " MB" << endl;

    // A --bam-cmd producer that did not exit 0 means the stream may have
    // been truncated: let the workers drain and flush (the files are
    // already on disk either way), but fail the run so nothing downstream
    // mistakes partial output for a complete one.
    bool ok = !dispatcher.producer_failed();
    for (auto& worker : workers)
      worker->signal_stop();
    for (auto& worker : workers)
      worker->wait_for_completion();
    return ok && !worker_failed.load(memory_order_relaxed);
  }
}  // namespace deeprm
