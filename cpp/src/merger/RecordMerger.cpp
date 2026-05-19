/***************************************************************************************************
 *
 * Copyright (C) 2025 Genome4me Incorporated - All Rights Reserved.
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

#include "RecordMerger.h"
#include <algorithm>
#include <iostream>
#include <functional>
#include <utility>
#include <unordered_map>

namespace deeprm {
  namespace {
    // Watson-Crick complement for reverse-complementing the query_sequence of
    // reverse-strand reads back into RNA-sense orientation. Non-ACGT (e.g.
    // 'N') is passed through unchanged.
    uint8_t complement_base(uint8_t b)
    {
      switch (b) {
        case 'A': return 'T';
        case 'T': return 'A';
        case 'C': return 'G';
        case 'G': return 'C';
        default:  return b;
      }
    }
  } // namespace

  RecordMerger::RecordMerger(const NormalizationFactors& nf, int cb_len, int kmer_len,
                             int max_token_len, int sampling, int dwell_shift, int sig_window,
                             uint64_t label_div)
    : norm_factors(nf), cb_len(cb_len), kmer_len(kmer_len), max_token_len(max_token_len),
      sampling(sampling), dwell_shift(dwell_shift), sig_window(sig_window), label_div(label_div)
  {
  }

  void RecordMerger::add_bam_records(vector<BamRecord>&& records)
  {
    bam_records = move(records);
  }

  void RecordMerger::add_pod5_meta_records(vector<Pod5RecordMeta>&& meta_records)
  {
    pod5_meta_records = move(meta_records);
  }

  bool RecordMerger::process_merged_record_with_meta(const BamRecord& bam_rec,
                                                     const Pod5RecordMeta& pod5_meta,
                                                     vector<ProcessedRecord>& output) const
  {
    // Read pod5 signal data from metadata
    vector<int16_t> signal_data;

    if (!Pod5Reader::read_pod5_record(pod5_meta, signal_data)) {
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

    if (signal_segments.empty()) {
      return false;
    }

    // Present reverse-strand reads in RNA-sense orientation. The model is
    // trained only on RNA-sense, base-of-interest-centred context.
    // signal_segments / dwell_token are already in basecalled (RNA-sense
    // 5'->3') order for both strands. The BAM stores SEQ/QUAL of reverse-
    // strand reads reverse-complemented into forward-reference order, and
    // ap.first (q_pos) is forward-reference. So for reverse-strand reads
    // reverse-complement seq, reverse bq, and mirror q_pos; signal/dwell are
    // left untouched.
    const int q_len = static_cast<int>(bam_rec.seq.size());
    const bool is_rev = (bam_rec.strand == -1);

    vector<uint8_t> sense_seq;
    vector<uint8_t> sense_bq;
    const vector<uint8_t>* useq = &bam_rec.seq;
    const vector<uint8_t>* ubq = &bam_rec.bq;
    if (is_rev) {
      sense_seq.resize(q_len);
      for (int i = 0; i < q_len; ++i) {
        sense_seq[i] = complement_base(bam_rec.seq[q_len - 1 - i]);
      }
      sense_bq.assign(bam_rec.bq.rbegin(), bam_rec.bq.rend());
      useq = &sense_seq;
      ubq = &sense_bq;
    }

    // Process each aligned pair
    int cb_half_len = cb_len / 2;
    int trim = kmer_len / 2;

    // Pre-allocate memory for better performance
    output.reserve(bam_rec.ap.size());

    for (const auto& ap : bam_rec.ap) {
      int q_pos = ap.first;
      int r_pos = ap.second;

      // Mirror q_pos into the RNA-sense (basecalled) coordinate system for
      // reverse-strand reads, matching sense seq/bq and the already
      // sense-ordered signal/dwell arrays.
      if (is_rev) {
        q_pos = q_len - 1 - q_pos;
      }

      int start_pos = q_pos - cb_half_len;
      int end_pos = q_pos + cb_half_len + 1;

      // Filter by context (strand-uniform: all arrays are RNA-sense)
      if (start_pos < 0 || end_pos + dwell_shift - trim >= q_len) {
        continue;
      }

      // Extract signal segments for this context
      if (start_pos >= static_cast<int>(signal_segments.size()) ||
        end_pos > static_cast<int>(signal_segments.size())) {
        continue;
      }

      // Create segment length array using transform for better performance
      vector<uint16_t> segment_len_arr(end_pos - start_pos);
      transform(signal_segments.begin() + start_pos,
                     signal_segments.begin() + end_pos,
                     segment_len_arr.begin(),
                     [this](const auto& seg) {
                       return static_cast<uint16_t>(seg.size() / this->sampling);
                     });

      if (cmp_not_equal(segment_len_arr.size(), cb_len)) {
        continue;
      }

      // Calculate token length
      int token_len = 0;
      for (int i = trim; i < static_cast<int>(segment_len_arr.size()) - trim; ++i) {
        token_len += segment_len_arr[i];
      }

      if (token_len <= 0 || token_len > max_token_len) {
        continue;
      }

      // Convert to signal block without copying segments
      auto signal_block = Utils::segmented_signal_to_block_range(
        signal_segments, start_pos, end_pos, segment_len_arr, kmer_len, sampling,
        sig_window, max_token_len);

      if (signal_block.empty()) {
        continue;
      }

      // Create processed record
      ProcessedRecord proc_rec;
      proc_rec.read_id = bam_rec.read_id;

      // Trim segment length array
      proc_rec.segment_len_arr = vector<uint16_t>(
        segment_len_arr.begin() + trim,
        segment_len_arr.end() - trim);

      proc_rec.signal_token = signal_block;

      // Extract k-mer tokens (RNA-sense seq for reverse-strand reads)
      proc_rec.kmer_token = vector<uint8_t>(
        useq->begin() + start_pos,
        useq->begin() + end_pos);

      // Extract dwell tokens. Motor leads the pore by dwell_shift bases;
      // dwell_token is in basecalled (RNA-sense) order for both strands, so
      // the offset is +dwell_shift uniformly.
      if (start_pos + dwell_shift + trim < static_cast<int>(dwell_token.size()) &&
        end_pos + dwell_shift - trim <= static_cast<int>(dwell_token.size())) {
        proc_rec.dwell_motor_token = vector<float>(
          dwell_token.begin() + start_pos + dwell_shift + trim,
          dwell_token.begin() + end_pos + dwell_shift - trim);
      }

      if (start_pos + trim < static_cast<int>(dwell_token.size()) &&
        end_pos - trim <= static_cast<int>(dwell_token.size())) {
        proc_rec.dwell_pore_token = vector<float>(
          dwell_token.begin() + start_pos + trim,
          dwell_token.begin() + end_pos - trim);
      }

      // Extract base quality (RNA-sense bq for reverse-strand reads)
      if (start_pos + trim < static_cast<int>(ubq->size()) &&
        end_pos - trim <= static_cast<int>(ubq->size())) {
        proc_rec.bq_token = vector<uint8_t>(
          ubq->begin() + start_pos + trim,
          ubq->begin() + end_pos - trim);

        // Clip quality values
        for (uint8_t& q : proc_rec.bq_token) {
          q = min(q, static_cast<uint8_t>(60));
        }
      }

      // Create label ID: (reference * label_div + position + 1) * strand
      proc_rec.label_id = static_cast<int64_t>(bam_rec.ref * label_div + r_pos + 1) *
                          bam_rec.strand;

      output.push_back(move(proc_rec));
    }

    return true;
  }

  vector<ProcessedRecord> RecordMerger::merge_and_process_with_meta()
  {
    vector<ProcessedRecord> results;

    // Process records with same index (already matched in main.cpp)
    if (bam_records.size() != pod5_meta_records.size()) {
      cerr << "Error: BAM and POD5 meta record counts don't match in RecordMerger" <<
          endl;
      return results;
    }

    for (size_t i = 0; i < bam_records.size(); ++i) {
      const BamRecord& bam_rec = bam_records[i];
      const Pod5RecordMeta& pod5_rec_meta = pod5_meta_records[i];

      // Verify read_id match for safety
      if (bam_rec.parent_id != pod5_rec_meta.read_id) {
        cerr << "Error: parent_id mismatch at index " << i
            << ": BAM=" << bam_rec.parent_id
            << ", POD5_META=" << pod5_rec_meta.read_id << endl;
        return results;
      }

      process_merged_record_with_meta(bam_rec, pod5_rec_meta, results);
    }

    return results;
  }

  void RecordMerger::clear()
  {
    bam_records.clear();
    pod5_meta_records.clear();
  }
} // namespace deeprm
