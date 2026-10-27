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

#include <atomic>
#include <cstdint>
#include <algorithm>
#include <string>
#include <vector>
#include <cnpy.h>
#include <libdeflate.h>

using namespace std;

// Chunked NPZ writer over preallocated fixed-stride row arrays.
//
// Producers write each record's tokens straight into the writer's own arrays
// (acquire_row / commit_row); no intermediate per-record object exists. The
// arrays hold chunk_size rows at strides derived from the token options, so
// the whole buffer is allocated once per worker (~82 MB at defaults) and
// reused for every chunk. This replaced a vector<ProcessedRecord> stage that
// cost ~10 KB and 7 heap allocations per record (~163 MB and ~5e9 allocations
// per worker per run).
//
// Rows are written to disk at the per-chunk OBSERVED maxima, not the strides:
// commit_row tracks the max actual length per array, and save_chunk uses those
// as the npy widths. The array contents and widths therefore match the writer
// it replaced, which measured the same maxima over the buffered records.
class NpzWriter {
public:
  // Fixed per-row strides, derived once from the token options. sig is the
  // fixed width (max_token_len + kmer_len - 1) * sampling; blocks that
  // exceed it -- only possible from a pathological move table, since
  // validate_geometry rejects sig_window > kmer_len -- are dropped by the
  // producer's stride guard, matching the Python preprocessing's dropped_wide rule.
  struct TokenStrides {
    size_t seg;    // segment_len_arr row: cb_len - kmer_len + 1
    size_t sig;    // signal_token row
    size_t kmer;   // kmer_token row: cb_len
    size_t tok;    // dwell_motor / dwell_pore / bq row: cb_len - kmer_len + 1
  };

  static TokenStrides strides_from(int cb_len, int kmer_len, int max_token_len,
                                   int sampling, int sig_window)
  {
    TokenStrides s;
    s.seg = static_cast<size_t>(cb_len - kmer_len + 1);
    s.sig = static_cast<size_t>(max_token_len + kmer_len - 1) *
            static_cast<size_t>(sampling);
    (void)sig_window;  // no longer widens the row; over-wide blocks are dropped
    s.kmer = static_cast<size_t>(cb_len);
    s.tok = s.seg;
    return s;
  }

  // Zeroed destinations for one record's tokens. Valid until the next
  // acquire_row / commit_row call on the same writer.
  struct RowRef {
    uint16_t* seg;
    float* sig;
    uint8_t* kmer;
    float* motor;
    float* pore;
    uint8_t* bq;
  };

  NpzWriter(const string& output_path, int chunk_size, int worker_id,
            const TokenStrides& strides);
  ~NpzWriter();
  // NpzWriter owns npz_compressor, a raw libdeflate handle freed in the destructor.
  // The compiler-generated copy duplicates only the pointer, leaving two writers
  // that free the same handle. This turns an accidental copy into a compile error.
  NpzWriter(const NpzWriter&) = delete;
  NpzWriter& operator=(const NpzWriter&) = delete;

  // Returns the next row, zero-filled. Calling acquire_row again without an
  // intervening commit_row returns the same row re-zeroed, so a producer that
  // abandons a row mid-fill (filter rejection) just moves on.
  RowRef acquire_row();

  // Producers must not write past this many elements into row.sig. Blocks fit
  // by construction (see strides_from), but a bounds check beats memory
  // corruption if that analysis is ever wrong.
  size_t sig_stride() const { return strides.sig; }

  // Finalizes the acquired row: records the actual lengths (they become the
  // chunk's npy widths via max) and advances the row count. The chunk file
  // itself is cut lazily by the next acquire_row (see the note there); the
  // trailing rows go out via flush(). rid0/rid1 are the read UUID halves,
  // parsed once by the producer at extraction time.
  void commit_row(int64_t rid0, int64_t rid1, int64_t label_id,
                  size_t seg_len, size_t sig_len, size_t kmer_len,
                  size_t motor_len, size_t pore_len, size_t bq_len);

  void flush();

  // True once any chunk could not be written (the chunk is removed). The worker then fails the run, so a run
  // that lost output never exits 0.
  bool write_failed() const { return failed_write; }
  // Output files that could not be written in full, over all workers of the run (main reports them as their own
  // cause of failure).
  static inline atomic<long> files_not_written{0};

private:
  bool failed_write = false;
  string output_path;
  int chunk_size;
  int worker_id;
  int chunk_id;

  TokenStrides strides;

  // Preallocated row storage: chunk_size rows per array, fixed stride.
  vector<uint16_t> rows_seg;
  vector<float> rows_sig;
  vector<uint8_t> rows_kmer;
  vector<float> rows_motor;
  vector<float> rows_pore;
  vector<uint8_t> rows_bq;
  vector<int64_t> rows_label;
  vector<int64_t> rows_rid;   // 2 per row
  size_t row_count;

  // Observed maxima within the current chunk; the widths actually written.
  size_t max_seg, max_sig, max_kmer, max_motor, max_pore, max_bq;

  vector<uint8_t> npz_zbuf;
  // libdeflate has no streaming API, so each array is first laid out contiguously in rawbuf and
  // then compressed in one call into zbuf.
  vector<uint8_t> npz_rawbuf;
  libdeflate_compressor* npz_compressor;

  void save_chunk(size_t count, const string& filename);
  void reset_maxima();
  string generate_filename(bool is_last_chunk);
};
