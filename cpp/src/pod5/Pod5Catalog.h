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
#include <cstddef>
#include <string>
#include <thread>
#include <vector>

#include "pod5/Pod5Reader.h"

using namespace std;

namespace deeprm {
  // The POD5 directory as the run sees it: the sorted file list and, after
  // scan_async()/join(), every read's metadata (read_id, batch/row, sample count; never
  // signal), grouped by reader thread and then by file in scan order. That
  // grouping is what WorkerAssignment consumes.
  class Pod5Catalog {
  public:
    explicit Pod5Catalog(const string& directory);

    const vector<string>& files() const { return pod5_files; }

    // Starts min(files, max_threads) reader threads, each scanning a
    // contiguous slice of files (np.array_split sizing). Returns at once so
    // the caller can overlap other work; join() waits for them.
    void scan_async(int max_threads);
    void join();
    int reader_count() const { return num_readers; }
    size_t record_count() const { return total_records; }
    // POD5 files that could not be read in full (open, batch, row or sample count); their reads are
    // missing, so the run fails when this is not 0.
    size_t failed_file_count() const { return failed_files.load(); }

    // The scan result, to be moved into a WorkerAssignment.
    vector<vector<vector<Pod5RecordMeta>>>& reader_records() { return scanned; }

  private:
    static vector<string> list_files(const string& directory);
    // files are pod5_files[first_index ...]; every record gets its catalog
    // index so workers can look the path up without carrying it.
    static void scan_worker(int worker_id, const vector<string>& files, size_t first_index,
                            vector<vector<Pod5RecordMeta>>& out, atomic<size_t>& failed);

    vector<string> pod5_files;
    int num_readers = 0;
    vector<thread> scan_threads;
    vector<vector<vector<Pod5RecordMeta>>> scanned;
    size_t total_records = 0;
    atomic<size_t> failed_files{0};
  };
}  // namespace deeprm

