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

#include "utils/Utils.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#ifdef USE_MPFR
#include <mpfr.h>
#endif

double Utils::mean_phred(const vector<uint8_t>& phred)
{
  if (phred.empty()) return 0.0;

  double sum = 0.0;
  for (uint8_t score : phred) {
    sum += pow(10.0, -score / 10.0);
  }
  return -10.0 * log10(sum / phred.size());
}

vector<float> Utils::move_to_dwell(const vector<bool>& move,
                                        double quantile_a, double quantile_b,
                                        double shift_mult, double scale_mult,
                                        int sampling)
{
  // Implementation of same logic as Python code
  // move = np.arange(1, len(move)+1, dtype = np.int32)[np.flip(move)]
  vector<int32_t> indices;
  for (size_t i = 0; i < move.size(); ++i) {
    if (move[move.size() - 1 - i]) {
      // np.flip(move)
      indices.push_back(static_cast<int32_t>(i + 1));
    }
  }

  // move = np.concatenate([np.zeros(1, dtype=np.int32), move])
  vector<int32_t> move_positions;
  move_positions.push_back(0);
  move_positions.insert(move_positions.end(), indices.begin(), indices.end());

  // move = move[1:] - move[:-1]
  vector<float> dwell_times;

#ifdef USE_MPFR
  mpfr_t mpfr_input, mpfr_result;
  mpfr_init(mpfr_input);
  mpfr_init(mpfr_result);
#endif
  for (size_t i = 1; i < move_positions.size(); ++i) {
    int32_t dwell = move_positions[i] - move_positions[i - 1];
#ifdef USE_MPFR
    mpfr_set_si(mpfr_input, dwell * sampling, MPFR_RNDN);
    mpfr_log10(mpfr_result, mpfr_input, MPFR_RNDN);
    dwell_times.push_back((mpfr_get_flt(mpfr_result, MPFR_RNDN)));
#else
    dwell_times.push_back(std::log10(static_cast<float>(dwell * sampling)));
#endif
  }

#ifdef USE_MPFR
  mpfr_clear(mpfr_input);
  mpfr_clear(mpfr_result);
  mpfr_free_cache();
#endif

  if (dwell_times.empty()) {
    return dwell_times;
  }

  float quantile_a_value = quantile(dwell_times, quantile_a);
  float quantile_b_value = quantile(dwell_times, quantile_b);

  // Single precision throughout, as numpy 2 does with a Python float times a
  // float32 (NEP 50): the multipliers are rounded to float first.
  float q_shift = max(0.1f, static_cast<float>(shift_mult) * (quantile_a_value + quantile_b_value));
  float q_scale = max(0.1f, static_cast<float>(scale_mult) * (quantile_b_value - quantile_a_value));

  // Vectorize dwell time normalization
#pragma GCC ivdep
  for (size_t i = 0; i < dwell_times.size(); ++i) {
    dwell_times[i] = (dwell_times[i] - q_shift) / q_scale;
  }

  return dwell_times;
}

vector<vector<double>> Utils::normalize_trim_segment_signal(
  const vector<double>& signal, const vector<bool>& move,
  int sp, int ts, int ns, double quantile_a, double quantile_b,
  double shift_mult, double scale_mult, int sampling)
{
  if (sp >= static_cast<int>(signal.size())) {
    return {};
  }

  vector<double> trimmed_signal(signal.begin() + sp, signal.end());
  int signal_len = static_cast<int>(trimmed_signal.size());

  // signal[sp:][ts:ns] as numpy slices it: an ns past the end stops at the end, and an empty slice
  // (ts at or after ns) drops the read.
  if (ns == 0) ns = signal_len;
  if (ns > signal_len) ns = signal_len;
  if (ts >= ns) {
    return {};
  }

  vector<double> processed_signal(trimmed_signal.begin() + ts, trimmed_signal.begin() + ns);

  if (processed_signal.empty()) {
    return {};
  }

  reverse(processed_signal.begin(), processed_signal.end());

  double quantile_a_value = quantile(processed_signal, quantile_a);
  double quantile_b_value = quantile(processed_signal, quantile_b);

  double q_shift = max(10.0, shift_mult * (quantile_a_value + quantile_b_value));
  double q_scale = max(1.0, scale_mult * (quantile_b_value - quantile_a_value));

  // Vectorize signal normalization
#pragma GCC ivdep
  for (size_t i = 0; i < processed_signal.size(); ++i) {
    processed_signal[i] = (processed_signal[i] - q_shift) / q_scale;
  }

  // np.where(move)[0][1:] * sampling: every move but the FIRST TRUE one (not index 0).
  vector<int> move_indices;
  bool first_move_seen = false;
  for (size_t i = 0; i < move.size(); ++i) {
    if (move[i]) {
      if (!first_move_seen) {
        first_move_seen = true;
        continue;
      }
      move_indices.push_back(static_cast<int>(i) * sampling);
    }
  }

  for (int& idx : move_indices) {
    idx = static_cast<int>(processed_signal.size()) - idx;
  }
  // A move table longer than the signal (the BAM and the POD5 do not fit each other) would put some
  // split points at or before the start; the read is dropped, as in the Python preprocessing.
  if (!move_indices.empty() && move_indices.back() <= 0) {
    move_past_signal.fetch_add(1, memory_order_relaxed);
    return {};
  }
  reverse(move_indices.begin(), move_indices.end());

  vector<vector<double>> segments;
  int start = 0;
  for (int idx : move_indices) {
    if (idx > start && idx <= static_cast<int>(processed_signal.size())) {
      segments.emplace_back(processed_signal.begin() + start,
                            processed_signal.begin() + idx);
      start = idx;
    }
  }

  if (start < static_cast<int>(processed_signal.size())) {
    segments.emplace_back(processed_signal.begin() + start,
                          processed_signal.end());
  }

  return segments;
}

vector<size_t> Utils::split_sizes(size_t n, int parts)
{
  vector<size_t> sizes(parts, n / parts);
  for (size_t i = 0; i < n % parts; ++i)
    ++sizes[i];
  return sizes;
}

vector<double> Utils::segmented_signal_to_block_range(
  const vector<vector<double>>& signal_segmented,
  int start_idx, int end_idx,
  const vector<uint16_t>& segment_len_arr,
  int kmer, int sampling, int sig_window, int pad_to)
{
  try {
    int kmer_pad = (kmer - 1) / 2;
    int lr_pad = (sig_window - 1) / 2;

    int l_skip_sum = 0;
    for (int i = 0; i < kmer_pad && i < static_cast<int>(segment_len_arr.size()); ++i) {
      l_skip_sum += segment_len_arr[i];
    }
    int l_skip = (l_skip_sum - lr_pad) * sampling;

    int r_skip_sum = 0;
    int arr_start_idx = max(0, static_cast<int>(segment_len_arr.size()) - kmer_pad);
    for (int i = arr_start_idx; i < static_cast<int>(segment_len_arr.size()); ++i) {
      r_skip_sum += segment_len_arr[i];
    }
    int r_skip = (r_skip_sum - lr_pad) * sampling;

    // Concatenate only [start_idx, end_idx) straight into one reserved buffer;
    // segments outside the range are never touched and no sub-list is built.
    vector<double> concatenated;
    size_t total_size = 0;
    for (int i = start_idx; i < end_idx; ++i) {
      total_size += signal_segmented[i].size();
    }
    concatenated.reserve(total_size);

    for (int i = start_idx; i < end_idx; ++i) {
      concatenated.insert(concatenated.end(),
                          signal_segmented[i].begin(),
                          signal_segmented[i].end());
    }

    if (concatenated.size() % sampling != 0) {
      return {};
    }

    if (r_skip < 0) r_skip = 0;
    vector<double> trimmed = vector<double>(concatenated.begin() + l_skip,
                                            concatenated.end() - r_skip);

    int target_len = (pad_to + kmer - 1) * sampling;
    int padding = target_len - static_cast<int>(trimmed.size());

    if (padding > 0) {
      trimmed.resize(trimmed.size() + padding, 0.0f);
    }

    return trimmed;
  } catch (...) {
    return {};
  }
}
