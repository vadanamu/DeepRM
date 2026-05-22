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

#include "../merger/RecordMerger.h"
#include <string>
#include <vector>
#include <cnpy.h>

using namespace std;
using namespace deeprm;

class NpzWriter {
private:
  string output_path;
  int chunk_size;
  int worker_id;
  int processing_unit_id;
  int chunk_id;

  vector<ProcessedRecord> buffer;

  // Reusable flat arrays for save_chunk (capacity preserved across calls)
  vector<uint16_t> flat_segment_len;
  vector<float> flat_signal;
  vector<uint8_t> flat_kmer;
  vector<float> flat_dwell_motor;
  vector<float> flat_dwell_pore;
  vector<uint8_t> flat_bq;
  vector<int64_t> label_ids;
  vector<int64_t> read_ids;

  // Reusable NPZ scratch buffers — reused across all chunks to avoid per-chunk
  // fresh allocations. With 256 workers, fresh allocations of ~80 MB per chunk
  // caused page-fault storms and mmap_lock contention (>40% CPU). Reusing the
  // buffer drops that to <8%.
  vector<uint8_t> npz_uncomp_buf;
  vector<uint8_t> npz_comp_buf;

  void save_chunk(const vector<ProcessedRecord>& records,
                  size_t offset, size_t count, const string& filename);
  string generate_filename(bool is_last_processing_unit, bool is_last_chunk);

public:
  NpzWriter(const string& output_path, int chunk_size, int worker_id);

  void add_records(vector<ProcessedRecord>&& records);
  void flush();
  void increment_processing_unit();
  void reset_chunk_id();
};
