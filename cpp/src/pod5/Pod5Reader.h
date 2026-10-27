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

#include <string>
#include <vector>

// POD5 C API includes
extern "C" {
#include <pod5_format/c_api.h>
}

#include "utils/Uuid128.h"

using namespace std;

namespace deeprm {
  struct Pod5RecordMeta {
    Uuid128 read_id;
    // Position of the file in Pod5Catalog::files(); the path itself is kept
    // once in the catalog, so a record costs no string and no heap block.
    uint32_t file_index = 0;
    // Per-use scratch slot: null after the metadata scan; each worker
    // sets it to its OWN lazily-opened reader right before fetching
    // signal (readers are not thread-safe, see 2a7f1c5).
    Pod5FileReader_t* reader = nullptr;
    size_t batch_index;
    size_t row_index;
    size_t sample_count;
  };

  // Holds one read-table batch across consecutive reads that share a reader
  // and batch_index. Fetching the batch per read re-decodes and re-validates
  // it every time and funnels through Arrow's global shared_ptr mutex pool,
  // so the batch is kept until a read needs a different one. One cache per
  // worker thread, never shared. release() must run before the reader it
  // refers to is closed.
  class Pod5BatchCache {
  private:
    Pod5FileReader_t* reader_ = nullptr;
    size_t batch_index_ = 0;
    Pod5ReadRecordBatch_t* batch_ = nullptr;

  public:
    Pod5BatchCache() = default;
    Pod5BatchCache(const Pod5BatchCache&) = delete;
    Pod5BatchCache& operator=(const Pod5BatchCache&) = delete;
    ~Pod5BatchCache();

    // The batch for (reader, batch_index), reusing the held one when it
    // matches. Null when the fetch fails.
    Pod5ReadRecordBatch_t* get(Pod5FileReader_t* reader, size_t batch_index);
    void release();
  };

  class Pod5Reader {
  private:
    string pod5_path;

  public:
    Pod5Reader(const string& path);
    ~Pod5Reader();

    bool parse_single_pod5_meta(const string& file_path,
                                vector<Pod5RecordMeta>& meta_records);

    static bool read_pod5_record(const Pod5RecordMeta& meta, Pod5BatchCache& cache,
                                 vector<int16_t>& signal);
  };
} // namespace deeprm
