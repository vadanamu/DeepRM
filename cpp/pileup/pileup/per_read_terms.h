/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_PER_READ_TERMS_H_
#define DEEPRM_PILEUP_PER_READ_TERMS_H_

#include <cstdint>
#include <span>
#include <vector>

namespace deeprm {

// The settings of the per-read terms (--threshold, --epsilon and whether the
// predictions get the v1.2 model's calibration).
struct TermOptions {
  double threshold;
  double epsilon;
  bool calibrated;
};

// The term each row adds to every per-site sum except count_all, in the
// dtypes of the Python pileup (_per_read_terms): counts are 0 or 1.
struct PerReadTerms {
  std::vector<std::uint8_t> count_pos;
  std::vector<double> logsum_1_p_pos;
  std::vector<float> kl_div_neg;
  std::vector<float> kl_div_pos;
  std::vector<std::uint8_t> count_pos_rule;
  std::vector<double> logsum_1_p_pos_rule;
  std::vector<double> logsum_1_p_pos_calib;
};

// Computes the terms of `pred` (one probability per row), bit for bit as
// NumPy 2.4.6 does on AVX512. Not thread-safe per object; one per thread.
class PerReadTermsCalculator {
 public:
  explicit PerReadTermsCalculator(const TermOptions& options);

  // The result is valid until the next call.
  const PerReadTerms& compute(std::span<const float> pred);

 private:
  void compute_kl_div(std::span<const float> pred);
  void compute_logsums(std::span<const float> pred);
  void compute_calibrated(std::span<const float> pred);

  float threshold_;
  float epsilon_;
  bool calibrated_;
  PerReadTerms terms_;
  std::vector<float> buffer_a_;
  std::vector<float> buffer_b_;
  std::vector<std::uint32_t> rows_;  // the rows that need the calibration
};

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_PER_READ_TERMS_H_
