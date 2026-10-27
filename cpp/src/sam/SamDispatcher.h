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

#include <deque>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <sys/types.h>
#include <memory>
#include <thread>
#include <unordered_map>
#include <string>
#include <vector>

#include <htslib/hts.h>
#include <htslib/sam.h>

#include "args/ArgumentParser.h"
#include "utils/Uuid128.h"

using namespace std;

namespace deeprm {
  class StreamWorker; // Forward declaration

  class SamDispatcher {
  private:
    string bam_path;
    // Streaming producer command (--bam-cmd); its stdout is the BAM stream.
    string bam_cmd;
    // pid of the spawned producer; -1 when reading from a path/stdin.
    pid_t producer_pid;
    bool producer_error;
    int bq_cutoff;
    char base_of_interest;
    int filter_flag;
    int bam_threads;
    static constexpr size_t read_chunk_size = 1024;
    // -Q, reads per worker. The bound is per worker rather than absolute so it
    // tracks -t; set_workers recomputes it once the real worker count is known.
    size_t max_queue_per_worker;
    size_t max_queue_size;

    deque<bam1_t*> internal_queue;
    mutex queue_mutex;
    condition_variable queue_cv;

    // Total bam1_t currently sitting in ALL worker queues. The dispatcher
    // stops draining internal_queue while this is at or above max_queue_size,
    // which lets internal_queue fill and the existing bound stall the reader.
    // Bounding the TOTAL rather than each worker's queue avoids head-of-line
    // blocking: one hot worker may hold most of the budget while the others
    // keep being fed; the dispatcher only pauses when the whole pipeline is
    // genuinely over budget. Measured need: at -t 512 -Q 1000 a ~3% worker
    // slowdown grew the unbounded queues to ~165 GB; a per-worker bound fixed
    // the memory but cost +75% runtime to head-of-line stalls.
    //
    // Checked once per 1024-read dispatch chunk, not per read, so the bound is
    // soft by up to one chunk (~12 MB) -- irrelevant against a multi-GB budget.
    // Workers decrement on pop and wake the dispatcher on the cap-crossing
    // decrement only.
    atomic<long> worker_inflight;
    mutex inflight_mutex;
    condition_variable inflight_cv;

    atomic<bool> is_running;
    atomic<bool> workers_ready;
    atomic<bool> reading_complete;
    thread read_thread;
    thread dispatch_thread;

    samFile* bam_file;
    sam_hdr_t* header;

    // Owned by the StreamRunner; both outlive the dispatcher's threads. Set
    // by set_workers() once the POD5 scan has produced the assignment.
    const vector<unique_ptr<StreamWorker>>* stream_workers;
    const unordered_map<Uuid128, int>* pod5_index;

    void read_loop();
    void stop_after_error(int pipe_fd);
    void reap_producer();
    void dispatch_loop();

  public:
    SamDispatcher(const Arguments& args);
    ~SamDispatcher();

    void start();
    void stop();

    // True when a --bam-cmd producer exited non-zero (or abnormally): the
    // stream may be truncated, so the caller must fail the run.
    bool producer_failed() const { return producer_error; }
    // Starts the dispatch thread; the workers must already be running.
    void set_workers(const vector<unique_ptr<StreamWorker>>& workers,
                     const unordered_map<Uuid128, int>& index);

    // Called by a worker after popping one read from its queue. Decrements the
    // global in-flight count and wakes the dispatcher exactly on the decrement
    // that crosses back under the cap.
    void on_worker_pop()
    {
      long prev = worker_inflight.fetch_sub(1, memory_order_relaxed);
      if (prev == static_cast<long>(max_queue_size)) {
        // Lock/unlock pairs the notify with the dispatcher's predicate check,
        // closing the lost-wakeup window.
        { lock_guard<mutex> g(inflight_mutex); }
        inflight_cv.notify_one();
      }
    }
  };
}
