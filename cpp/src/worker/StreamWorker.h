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

#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <vector>
#include <htslib/sam.h>
#include "args/ArgumentParser.h"
#include "pod5/Pod5Reader.h"

using namespace std;

namespace deeprm {
  class SamDispatcher; // Forward declaration

  class StreamWorker {
  private:
    int worker_id;
    const Arguments& args;
    // Catalog file list that the records' file_index refers to.
    const vector<string>& pod5_files;
    const vector<Pod5RecordMeta>& pod5_meta_records;

    // Reports queue pops to the dispatcher's global in-flight counter.
    SamDispatcher& dispatcher;

    // Shared with every worker and read by StreamRunner::run() after the join: set when this
    // worker could not open a POD5 file or read a read's signal. The run then
    // exits non-zero instead of shipping incomplete output as a success.
    atomic<bool>& failed;

    queue<bam1_t*> bam_queue;
    mutex queue_mutex;
    condition_variable queue_cv;
    atomic<bool> is_running;
    thread worker_thread;

    // Runs on worker_thread; owns the NpzWriter so its buffers are allocated
    // on the thread that uses them.
    void process_loop();

  public:
    StreamWorker(int id, const Arguments& args, const vector<string>& pod5_files,
                 const vector<Pod5RecordMeta>& pod5_meta, SamDispatcher& dispatcher,
                 atomic<bool>& failed);
    ~StreamWorker();

    void start();
    void stop();
    void signal_stop();
    void wait_for_completion();
    void add_bam_data(bam1_t* read);
  };
}
