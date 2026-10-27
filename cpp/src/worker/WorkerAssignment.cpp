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

#include "worker/WorkerAssignment.h"

#include <iterator>
#include <utility>

#include "utils/Utils.h"

namespace deeprm {
  size_t WorkerAssignment::count_records(const ScannedRecords& scanned)
  {
    size_t n = 0;
    for (const auto& reader_files : scanned)
      for (const auto& file_records : reader_files)
        n += file_records.size();
    return n;
  }

  void WorkerAssignment::build_index()
  {
    read_index.reserve(total_records);
    for (int worker_id = 0; worker_id < num_workers(); ++worker_id)
      for (const auto& record : records[worker_id])
        read_index[record.read_id] = worker_id;
  }

  WorkerAssignment WorkerAssignment::block_round_robin(Pod5Catalog& catalog,
                                                       int num_workers, size_t block)
  {
    WorkerAssignment a;
    a.pod5_files = catalog.files();
    ScannedRecords scanned = move(catalog.reader_records());
    a.total_records = count_records(scanned);
    a.records.resize(num_workers);

    log_info() << "POD5 records (" << a.total_records
        << "): streaming block round-robin, " << block
        << " reads per block over " << num_workers << " workers" << endl;

    // Records move straight into their worker's vector and each source file's
    // vector is released as it is drained, so this costs no flatten copy.
    size_t reserve_each = a.total_records / num_workers + block;
    for (auto& worker_records : a.records)
      worker_records.reserve(reserve_each);

    size_t rank = 0;
    for (auto& reader_files : scanned) {
      for (auto& file_records : reader_files) {
        for (auto& record : file_records) {
          a.records[(rank / block) % num_workers].push_back(move(record));
          ++rank;
        }
        file_records.clear();
        file_records.shrink_to_fit();
      }
    }
    a.build_index();
    return a;
  }

  WorkerAssignment WorkerAssignment::by_reader(Pod5Catalog& catalog)
  {
    WorkerAssignment a;
    a.pod5_files = catalog.files();
    ScannedRecords scanned = move(catalog.reader_records());
    a.total_records = count_records(scanned);
    a.records.resize(scanned.size());

    log_info() << "POD5 records (" << a.total_records
        << "): direct reader assignment with " << scanned.size() << " workers" << endl;

    for (size_t i = 0; i < scanned.size(); ++i) {
      size_t count = 0;
      for (const auto& file_records : scanned[i])
        count += file_records.size();
      a.records[i].reserve(count);
      for (auto& file_records : scanned[i])
        a.records[i].insert(a.records[i].end(),
                            make_move_iterator(file_records.begin()),
                            make_move_iterator(file_records.end()));
    }
    a.build_index();
    return a;
  }

  WorkerAssignment WorkerAssignment::flatten_split(Pod5Catalog& catalog,
                                                   int num_workers)
  {
    WorkerAssignment a;
    a.pod5_files = catalog.files();
    ScannedRecords scanned = move(catalog.reader_records());
    a.total_records = count_records(scanned);

    log_info() << "POD5 records (" << a.total_records
        << "): redistributing records among " << num_workers << " workers" << endl;

    vector<Pod5RecordMeta> all_records;
    all_records.reserve(a.total_records);
    for (auto& reader_files : scanned)
      for (auto& file_records : reader_files)
        all_records.insert(all_records.end(),
                           make_move_iterator(file_records.begin()),
                           make_move_iterator(file_records.end()));

    a.records.resize(num_workers);
    vector<size_t> counts = Utils::split_sizes(all_records.size(), num_workers);
    size_t idx = 0;
    for (int i = 0; i < num_workers; ++i) {
      size_t count = counts[i];
      a.records[i].reserve(count);
      for (size_t j = 0; j < count; ++j, ++idx)
        a.records[i].push_back(move(all_records[idx]));
    }
    a.build_index();
    return a;
  }
}  // namespace deeprm
