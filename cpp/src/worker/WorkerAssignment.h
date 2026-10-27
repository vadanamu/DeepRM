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

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "pod5/Pod5Catalog.h"
#include "pod5/Pod5Reader.h"
#include "utils/Uuid128.h"

using namespace std;

namespace deeprm {
  // POD5 records as the metadata scan produces them: one vector per reader
  // thread, one inner vector per file, in scan order.
  using ScannedRecords = vector<vector<vector<Pod5RecordMeta>>>;

  // The partition of every POD5 record into per-worker vectors, plus the
  // read_id -> worker index built from it and the file list the records'
  // file_index refers to. Records are moved out of the catalog; the three
  // strategies below are the only way to build one.
  class WorkerAssignment {
  public:
    // Streaming: rank = position in the file walk (files in scan order, rows
    // in batch order); worker = (rank / block) % num_workers. dorado emits
    // reads in roughly the order it walked the POD5 directory, so under a
    // contiguous partition the live window of a raw BAM lands on one or two
    // workers (~20 of 512 cores busy on the 2,527-file set). Round-robin over
    // rank blocks spreads any window wider than block * num_workers across
    // every worker, while a block keeps consecutive reads (almost always one
    // file) on one worker so its single reader is reused.
    static WorkerAssignment block_round_robin(Pod5Catalog& catalog, int num_workers,
                                              size_t block);

    // File mode, files >= workers: each reader thread's files become one
    // worker's slice, unchanged. A worker then owns whole files and reads
    // them sequentially.
    static WorkerAssignment by_reader(Pod5Catalog& catalog);

    // File mode, files < workers: flatten every record and split the sequence
    // into num_workers contiguous slices (np.array_split sizing), so a file
    // may be shared by neighbouring workers.
    static WorkerAssignment flatten_split(Pod5Catalog& catalog, int num_workers);

    int num_workers() const { return static_cast<int>(records.size()); }
    const vector<string>& files() const { return pod5_files; }
    size_t record_count() const { return total_records; }
    vector<vector<Pod5RecordMeta>>& worker_records() { return records; }
    const unordered_map<Uuid128, int>& index() const { return read_index; }
    unordered_map<Uuid128, int>& index() { return read_index; }

  private:
    WorkerAssignment() = default;
    static size_t count_records(const ScannedRecords& scanned);
    // Builds read_index from records; called by every factory last.
    void build_index();

    vector<string> pod5_files;
    vector<vector<Pod5RecordMeta>> records;
    unordered_map<Uuid128, int> read_index;
    size_t total_records = 0;
  };
}  // namespace deeprm

