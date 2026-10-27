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

#include "merger/RecordMerger.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <unordered_map>
#include <utility>

namespace deeprm {
  namespace {
    constexpr int kPadNone = 0;
    constexpr int kPadZero = 1;
    constexpr int kPadEdge = 2;

    int parse_padding_mode(const string& mode)
    {
      if (mode == "zero")
        return kPadZero;
      if (mode == "edge")
        return kPadEdge;
      return kPadNone;
    }

    // np.median over one segment: sorted copy, even length averages the two
    // middle values.
    double median_of(const vector<double>& v)
    {
      vector<double> tmp = v;
      sort(tmp.begin(), tmp.end());
      size_t n = tmp.size();
      if (n == 0)
        return 0.0;
      if (n % 2 == 1)
        return tmp[n / 2];
      return (tmp[n / 2 - 1] + tmp[n / 2]) / 2.0;
    }

    // The Python preprocessing's slice_scalar_token + pad_scalar. Returns false only for the
    // drop cases (out of bounds in none mode, empty clip); python's silent
    // numpy slice clamping and the min_len return-unpadded shortcut are
    // reproduced, so the caller's length checks do the remaining dropping.
    template <typename T>
    bool slice_scalar(const vector<T>& arr, long start, long end,
                      long lb, long rb, int mode, long min_len, T zero,
                      vector<T>& out)
    {
      out.clear();
      long n = static_cast<long>(arr.size());
      if (mode == kPadNone) {
        if (start < lb || end > rb)
          return false;
        long cs = min(max(start, 0L), n);
        long ce = min(max(end, 0L), n);
        if (cs < ce)
          out.assign(arr.begin() + cs, arr.begin() + ce);
        return true;
      }
      long cs = max(start, lb);
      long ce = min(end, rb);
      if (cs >= ce)
        return false;
      long pad_left = max(0L, lb - start);
      long pad_right = max(0L, end - rb);
      long bs = min(max(cs, 0L), n);
      long be = min(max(ce, 0L), n);
      out.assign(arr.begin() + bs, arr.begin() + be);
      if (static_cast<long>(out.size()) < min_len)
        return true;  // pad_scalar: too short to pad, returned as-is
      T left = (mode == kPadEdge) ? out.front() : zero;
      T right = (mode == kPadEdge) ? out.back() : zero;
      out.insert(out.begin(), pad_left, left);
      out.insert(out.end(), pad_right, right);
      return true;
    }

    // The Python preprocessing's slice_signal_segments + pad_signal: per-base signal segments,
    // padded with sampling-length constant segments (zeros, or the median of
    // the first/last kept segment for edge mode).
    bool slice_segments(const vector<vector<double>>& segs, long start, long end,
                        long lb, long rb, int sampling, int mode, long min_len,
                        vector<vector<double>>& out)
    {
      out.clear();
      long n = static_cast<long>(segs.size());
      long cs = max(start, lb);
      long ce = min(end, rb);
      if (cs >= ce)
        return false;
      long pad_left = max(0L, lb - start);
      long pad_right = max(0L, end - rb);
      long bs = min(max(cs, 0L), n);
      long be = min(max(ce, 0L), n);
      out.assign(segs.begin() + bs, segs.begin() + be);
      if (pad_left == 0 && pad_right == 0)
        return true;
      if (out.empty() || static_cast<long>(out.size()) < min_len)
        return true;  // pad_signal: too short to pad, returned as-is
      vector<double> left_pad(sampling,
                              mode == kPadEdge ? median_of(out.front()) : 0.0);
      vector<double> right_pad(sampling,
                               mode == kPadEdge ? median_of(out.back()) : 0.0);
      out.insert(out.begin(), pad_left, left_pad);
      out.insert(out.end(), pad_right, right_pad);
      return true;
    }
  } // namespace

  RecordMerger::RecordMerger(const NormalizationFactors& nf, int cb_len, int kmer_len,
                             int max_token_len, int sampling, int dwell_shift, int sig_window,
                             uint64_t label_div, const string& padding_mode, bool use_trimming)
    : norm_factors(nf), cb_len(cb_len), kmer_len(kmer_len), max_token_len(max_token_len),
      sampling(sampling), dwell_shift(dwell_shift), sig_window(sig_window), label_div(label_div),
      pad_mode(parse_padding_mode(padding_mode)), use_trimming(use_trimming)
  {
  }

  bool RecordMerger::process_merged_record_with_meta(const BamRecord& bam_rec,
                                                     const Pod5RecordMeta& pod5_meta,
                                                     Pod5BatchCache& batch_cache,
                                                     NpzWriter& writer) const
  {
    // Read pod5 signal data from metadata
    vector<int16_t> signal_data;

    if (!Pod5Reader::read_pod5_record(pod5_meta, batch_cache, signal_data)) {
      cerr << "Failed to read POD5 signal for read_id: " << pod5_meta.read_id <<
          endl;
      return false;
    }

    // Convert int16_t signal to double using transform for better performance
    vector<double> signal(signal_data.size());
    transform(signal_data.begin(), signal_data.end(), signal.begin(),
              [](int16_t val) { return static_cast<double>(val); });

    // Generate dwell tokens
    auto dwell_token = Utils::move_to_dwell(bam_rec.mv, 0.2, 0.8, 0.5, 1.5, sampling);

    // Normalize and segment signal
    auto signal_segments = Utils::normalize_trim_segment_signal(
      signal, bam_rec.mv, bam_rec.sp, bam_rec.ts, bam_rec.ns,
      norm_factors.quantile_a, norm_factors.quantile_b,
      norm_factors.shift_mult, norm_factors.scale_mult, sampling);

    // No usable segments is a legitimate empty result.
    if (signal_segments.empty()) {
      return true;
    }

    // Python semantics (shared by file mode and streaming): every token
    // slice is taken against [left_bound, right_bound) with the configured
    // padding mode; in none mode an out-of-bounds slice drops the site, and
    // motor/pore/bq are required (no zero-length conditional fields).
    int cb_half_len = cb_len / 2;
    int trim = kmer_len / 2;
    long min_len = cb_half_len + 2;
    long q_len = static_cast<long>(bam_rec.seq.size());
    long left_bound = use_trimming ? bam_rec.trim_left : 1;
    long right_bound = use_trimming ? bam_rec.trim_right : q_len;
    long tok_width = cb_len - kmer_len + 1;

    // Reused per-site buffers; sizes are tiny except sig_padded (pad modes only).
    vector<float> motor_buf, pore_buf;
    vector<uint8_t> bq_buf, motif_buf;
    vector<vector<double>> sig_padded;
    vector<uint16_t> segment_len_arr;

    for (const auto& ap : bam_rec.ap) {
      long q_pos = ap.first;
      long r_pos = ap.second;
      long start_pos = q_pos - cb_half_len;
      long end_pos = q_pos + cb_half_len + 1;

      if (!slice_scalar(dwell_token, start_pos + dwell_shift + trim,
                        end_pos + dwell_shift - trim, left_bound, right_bound,
                        pad_mode, 1, 0.0f, motor_buf))
        continue;
      if (!slice_scalar(dwell_token, start_pos + trim, end_pos - trim,
                        left_bound, right_bound, pad_mode, min_len - trim, 0.0f,
                        pore_buf))
        continue;
      if (!slice_scalar(bam_rec.bq, start_pos + trim, end_pos - trim,
                        left_bound, right_bound, pad_mode, min_len - trim,
                        static_cast<uint8_t>(0), bq_buf))
        continue;
      if (!slice_scalar(bam_rec.seq, start_pos, end_pos, left_bound, right_bound,
                        pad_mode, min_len, static_cast<uint8_t>(0), motif_buf))
        continue;

      // Signal segments: none mode keeps the no-copy range path; pad modes
      // materialize the padded per-base segment list.
      long seg_lo, seg_hi;
      const vector<vector<double>>* seg_src;
      if (pad_mode == kPadNone) {
        if (start_pos < left_bound || end_pos > right_bound)
          continue;
        // A clamped (shorter) slice could never pass the cb_len check below.
        if (end_pos > static_cast<long>(signal_segments.size()))
          continue;
        seg_src = &signal_segments;
        seg_lo = start_pos;
        seg_hi = end_pos;
      } else {
        if (!slice_segments(signal_segments, start_pos, end_pos, left_bound,
                            right_bound, sampling, pad_mode, min_len, sig_padded))
          continue;
        seg_src = &sig_padded;
        seg_lo = 0;
        seg_hi = static_cast<long>(sig_padded.size());
      }

      // Length/shape gate: everything must have its exact width.
      segment_len_arr.resize(seg_hi - seg_lo);
      transform(seg_src->begin() + seg_lo, seg_src->begin() + seg_hi,
                segment_len_arr.begin(),
                [this](const auto& seg) {
                  return static_cast<uint16_t>(seg.size() / this->sampling);
                });
      if (cmp_not_equal(segment_len_arr.size(), cb_len))
        continue;

      long token_len = 0;
      for (long i = trim; i < static_cast<long>(segment_len_arr.size()) - trim; ++i)
        token_len += segment_len_arr[i];
      if (token_len <= 0 || token_len > max_token_len)
        continue;

      if (static_cast<long>(motor_buf.size()) != tok_width ||
          static_cast<long>(pore_buf.size()) != tok_width ||
          static_cast<long>(bq_buf.size()) != tok_width ||
          cmp_not_equal(motif_buf.size(), cb_len))
        continue;

      auto signal_block = Utils::segmented_signal_to_block_range(
        *seg_src, seg_lo, seg_hi, segment_len_arr, kmer_len, sampling,
        sig_window, max_token_len);
      if (signal_block.empty())
        continue;

      // The sig stride is the fixed width (max_token_len + kmer_len - 1)
      // * sampling. Blocks can exceed it only on a pathological move table
      // (validate_geometry rejects sig_window > kmer_len); drop the site,
      // exactly like the Python preprocessing's dropped_wide rule.
      if (signal_block.size() > writer.sig_stride()) {
        cerr << "RecordMerger: signal block (" << signal_block.size()
             << ") exceeds row stride (" << writer.sig_stride()
             << "), dropping site" << endl;
        continue;
      }

      // All filters passed: write this site straight into the writer's next
      // row. The row is zeroed by acquire_row, so unfilled tails stay zero.
      NpzWriter::RowRef row = writer.acquire_row();

      size_t seg_len = segment_len_arr.size() - 2 * trim;
      copy(segment_len_arr.begin() + trim, segment_len_arr.end() - trim, row.seg);

      size_t sig_len = signal_block.size();
      for (size_t j = 0; j < sig_len; ++j)
        row.sig[j] = static_cast<float>(signal_block[j]);

      size_t kmer_token_len = motif_buf.size();
      copy(motif_buf.begin(), motif_buf.end(), row.kmer);

      size_t motor_len = motor_buf.size();
      copy(motor_buf.begin(), motor_buf.end(), row.motor);

      size_t pore_len = pore_buf.size();
      copy(pore_buf.begin(), pore_buf.end(), row.pore);

      size_t bq_len = bq_buf.size();
      transform(bq_buf.begin(), bq_buf.end(), row.bq,
                [](uint8_t q) { return min(q, static_cast<uint8_t>(60)); });

      // Label ID: (reference * label_div + position + 1) * strand
      int64_t label_id = static_cast<int64_t>(bam_rec.ref * label_div + r_pos + 1) *
                         bam_rec.strand;

      writer.commit_row(bam_rec.rid0, bam_rec.rid1, label_id, seg_len, sig_len,
                        kmer_token_len, motor_len, pore_len, bq_len);
    }

    return true;
  }

} // namespace deeprm
