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

#include "pod5/Pod5Reader.h"

#include <glob.h>

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace deeprm {
  Pod5Reader::Pod5Reader(const string& path) : pod5_path(path)
  {
  }

  Pod5Reader::~Pod5Reader()
  {
  }

  bool Pod5Reader::parse_single_pod5_meta(const string& file_path,
                                          vector<Pod5RecordMeta>& meta_records)
  {
    try {
      // Check if file exists
      if (!filesystem::exists(file_path)) {
        cerr << "POD5 file does not exist: " << file_path << endl;
        return false;
      }

      // Initialize POD5 library
      pod5_error_t error = pod5_init();
      if (error != POD5_OK) {
        cerr << "Failed to initialize POD5: " << pod5_get_error_string() << endl;
        return false;
      }

      // Open POD5 file
      Pod5FileReader_t* reader = pod5_open_file(file_path.c_str());
      if (!reader) {
        cerr << "Failed to open POD5 file: " << file_path
            << " - " << pod5_get_error_string() << endl;
        pod5_terminate();
        return false;
      }

      // Get batch count
      size_t batch_count = 0;
      error = pod5_get_read_batch_count(&batch_count, reader);
      if (error != POD5_OK) {
        cerr << "Failed to get batch count: " << pod5_get_error_string() << endl;
        pod5_close_and_free_reader(reader);
        pod5_terminate();
        return false;
      }

      // Process each batch
      for (size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
        Pod5ReadRecordBatch_t* batch = nullptr;
        error = pod5_get_read_batch(&batch, reader, batch_index);
        if (error != POD5_OK) {
          // A batch that cannot be read makes the whole file unreadable: its reads would be lost.
          cerr << "Failed to get batch " << batch_index
              << ": " << pod5_get_error_string() << endl;
          pod5_close_and_free_reader(reader);
          return false;
        }

        // Get batch row count
        size_t batch_row_count = 0;
        error = pod5_get_read_batch_row_count(&batch_row_count, batch);
        if (error != POD5_OK) {
          cerr << "Failed to get batch row count: " << pod5_get_error_string() <<
              endl;
          pod5_free_read_batch(batch);
          pod5_close_and_free_reader(reader);
          return false;
        }

        // Process each row (read) in batch - metadata only
        for (size_t row = 0; row < batch_row_count; ++row) {
          Pod5RecordMeta meta;

          // Get read information
          ReadBatchRowInfo_t row_info;
          uint16_t read_table_version = 0;
          error = pod5_get_read_batch_row_info_data(
            batch, row, READ_BATCH_ROW_INFO_VERSION, &row_info, &read_table_version);
          if (error != POD5_OK) {
            cerr << "Failed to get row info for row " << row
                << ": " << pod5_get_error_string() << endl;
            pod5_free_read_batch(batch);
            pod5_close_and_free_reader(reader);
            return false;
          }

          // Convert read ID to string
          meta.read_id = uuid128_from_bytes(row_info.read_id);

          // `reader` keeps its null default and file_index is filled by the
          // catalog. Every consumer (streaming worker and file-mode worker
          // alike) opens its own reader per file and sets the field right
          // before use, so keeping the scan reader alive here only pinned
          // Arrow memory (~24 GB measured at 99 files) for the whole run.
          meta.batch_index = batch_index;
          meta.row_index = row;

          // Get sample count
          size_t sample_count = 0;
          error = pod5_get_read_complete_sample_count(reader, batch, row, &sample_count);
          if (error != POD5_OK) {
            cerr << "Failed to get sample count for read " << uuid128_to_hex(meta.read_id)
                << ": " << pod5_get_error_string() << endl;
            pod5_free_read_batch(batch);
            pod5_close_and_free_reader(reader);
            return false;
          }
          meta.sample_count = sample_count;

          meta_records.push_back(move(meta));
        }

        // Free batch memory
        pod5_free_read_batch(batch);
      }

      // The scan reader is not referenced by the metadata (workers open
      // their own); close it now so Arrow releases the file's footer and
      // batch state instead of holding it until process exit.
      pod5_close_and_free_reader(reader);

      return true;
    } catch (const exception& e) {
      cerr << "Exception reading POD5 file " << file_path << ": " << e.what() <<
          endl;
      return false;
    }
  }

  Pod5BatchCache::~Pod5BatchCache()
  {
    release();
  }

  Pod5ReadRecordBatch_t* Pod5BatchCache::get(Pod5FileReader_t* reader, size_t batch_index)
  {
    if (batch_ && reader_ == reader && batch_index_ == batch_index)
      return batch_;
    release();
    Pod5ReadRecordBatch_t* batch = nullptr;
    if (pod5_get_read_batch(&batch, reader, batch_index) != POD5_OK) {
      cerr << "Failed to get batch " << batch_index
          << ": " << pod5_get_error_string() << endl;
      return nullptr;
    }
    reader_ = reader;
    batch_index_ = batch_index;
    batch_ = batch;
    return batch_;
  }

  void Pod5BatchCache::release()
  {
    if (batch_) {
      pod5_free_read_batch(batch_);
      batch_ = nullptr;
    }
    reader_ = nullptr;
  }

  bool Pod5Reader::read_pod5_record(const Pod5RecordMeta& meta, Pod5BatchCache& cache,
                                    vector<int16_t>& signal)
  {
    try {
      if (!meta.reader) {
        cerr << "Invalid reader pointer in metadata" << endl;
        return false;
      }

      // Only the batch is re-fetched per read; no row-info call is needed on this path.
      Pod5ReadRecordBatch_t* batch = cache.get(meta.reader, meta.batch_index);
      if (!batch)
        return false;

      signal.resize(meta.sample_count);
      pod5_error_t error = pod5_get_read_complete_signal(
        meta.reader, batch, meta.row_index, meta.sample_count, signal.data());
      if (error != POD5_OK) {
        cerr << "Failed to get signal for read " << uuid128_to_hex(meta.read_id)
            << ": " << pod5_get_error_string() << endl;
        return false;
      }
      return true;
    } catch (const exception& e) {
      cerr << "Exception reading POD5 record: " << e.what() << endl;
      return false;
    }
  }
} // namespace deeprm
