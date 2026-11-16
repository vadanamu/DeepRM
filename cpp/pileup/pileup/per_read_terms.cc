/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include "pileup/per_read_terms.h"

#include <algorithm>
#include <cstddef>

#include "pileup/numpy_math.h"
#include "pileup/svml_port.h"

namespace deeprm {
namespace {

// Per-read cut of the stoichiometry rule (_RULE_CUT of the Python pileup).
constexpr float kRuleCut = static_cast<float>(0.98);
// The largest prediction whose calibrated term is +0.0 (see
// compute_calibrated): 1.4135108e-10.
constexpr float kLastPlusZero = 0x1.36d596p-33F;

// np.clip(x, epsilon, 1.0) in float32.
float clip_to_unit(float x, float epsilon) {
  return std::min(std::max(x, epsilon), 1.0F);
}

}  // namespace

// NumPy computes these terms in float32: a Python float beside a float32
// array is converted to float32.
PerReadTermsCalculator::PerReadTermsCalculator(const TermOptions& options)
    : threshold_(static_cast<float>(options.threshold)),
      epsilon_(static_cast<float>(options.epsilon)),
      calibrated_(options.calibrated) {}

const PerReadTerms& PerReadTermsCalculator::compute(
    std::span<const float> pred) {
  const std::size_t size = pred.size();
  terms_.count_pos.resize(size);
  terms_.logsum_1_p_pos.resize(size);
  terms_.kl_div_neg.resize(size);
  terms_.kl_div_pos.resize(size);
  terms_.count_pos_rule.resize(size);
  terms_.logsum_1_p_pos_rule.resize(size);
  terms_.logsum_1_p_pos_calib.resize(size);
  buffer_a_.resize(size);
  buffer_b_.resize(size);
  compute_kl_div(pred);
  compute_logsums(pred);
  compute_calibrated(pred);
  return terms_;
}

// kl_div = p * log2(2 * p + eps) + (1 - p) * log2(2 * (1 - p) + eps)
void PerReadTermsCalculator::compute_kl_div(std::span<const float> pred) {
  const std::size_t size = pred.size();
  for (std::size_t i = 0; i < size; ++i) {
    buffer_a_[i] = 2.0F * pred[i] + epsilon_;
    buffer_b_[i] = 2.0F * (1.0F - pred[i]) + epsilon_;
  }
  svml_log2(buffer_a_, buffer_a_);
  svml_log2(buffer_b_, buffer_b_);
  for (std::size_t i = 0; i < size; ++i) {
    const float left = pred[i] * buffer_a_[i];
    const float right = (1.0F - pred[i]) * buffer_b_[i];
    const float kl_div = left + right;
    // Multiplied by the mask, as NumPy does (keeps the sign of a zero).
    terms_.kl_div_neg[i] = kl_div * (pred[i] <= 0.5F ? 1.0F : 0.0F);
    terms_.kl_div_pos[i] = kl_div * (pred[i] > 0.5F ? 1.0F : 0.0F);
  }
}

// log10(clip(1 - p)) times the 0/1 count; float32 * int64 is float64.
void PerReadTermsCalculator::compute_logsums(std::span<const float> pred) {
  const std::size_t size = pred.size();
  for (std::size_t i = 0; i < size; ++i) {
    buffer_a_[i] = clip_to_unit(1.0F - pred[i], epsilon_);
  }
  svml_log10(buffer_a_, buffer_a_);
  for (std::size_t i = 0; i < size; ++i) {
    const double log_1_p = static_cast<double>(buffer_a_[i]);
    terms_.count_pos[i] = pred[i] >= threshold_ ? 1 : 0;
    terms_.count_pos_rule[i] = pred[i] >= kRuleCut ? 1 : 0;
    terms_.logsum_1_p_pos[i] =
        log_1_p * static_cast<double>(terms_.count_pos[i]);
    terms_.logsum_1_p_pos_rule[i] =
        log_1_p * static_cast<double>(terms_.count_pos_rule[i]);
  }
}

// The same term as logsum_1_p_pos_rule, on the calibrated probability.
//
// Most predictions need no calibration: below kCalibratedFrom the calibrated
// probability is below the rule's cut, so the term is the logarithm of
// 1 - probability times 0. That is +0.0 up to kLastPlusZero, where the
// calibrated probability is so small that 1 - probability is 1, and -0.0
// above. tests/check_terms.py checks this for every float32.
void PerReadTermsCalculator::compute_calibrated(std::span<const float> pred) {
  rows_.clear();
  for (std::size_t i = 0; i < pred.size(); ++i) {
    if (calibrated_ && pred[i] < kCalibratedFrom) {
      terms_.logsum_1_p_pos_calib[i] = pred[i] <= kLastPlusZero ? 0.0 : -0.0;
      continue;
    }
    const std::size_t at = rows_.size();
    rows_.push_back(static_cast<std::uint32_t>(i));
    buffer_b_[at] = calibrated_ ? calibrated_probability(pred[i]) : pred[i];
    buffer_a_[at] = clip_to_unit(1.0F - buffer_b_[at], epsilon_);
  }
  const std::span<float> logs(buffer_a_.data(), rows_.size());
  svml_log10(logs, logs);
  for (std::size_t at = 0; at < rows_.size(); ++at) {
    const double confident = buffer_b_[at] >= kRuleCut ? 1.0 : 0.0;
    terms_.logsum_1_p_pos_calib[rows_[at]] =
        static_cast<double>(buffer_a_[at]) * confident;
  }
}

}  // namespace deeprm
