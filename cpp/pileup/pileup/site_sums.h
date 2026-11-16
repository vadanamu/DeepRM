/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_SITE_SUMS_H_
#define DEEPRM_PILEUP_SITE_SUMS_H_

#include <cstdint>

namespace deeprm {

// The per-site sums of the pileup (SITE_SUMS of the Python pileup).
struct SiteSums {
  std::int64_t count_all = 0;
  std::int64_t count_pos = 0;
  std::int64_t count_pos_rule = 0;
  double logsum_1_p_pos = 0.0;
  double kl_div_neg = 0.0;
  double kl_div_pos = 0.0;
  double logsum_1_p_pos_rule = 0.0;
  double logsum_1_p_pos_calib = 0.0;
};

// What the rows of one per-read file add to one site. The kl_div sums are
// float32 because NumPy adds a file's float32 terms in float32.
struct FileSite {
  std::int64_t label_id;
  std::uint32_t count_all;
  std::uint32_t count_pos;
  std::uint32_t count_pos_rule;
  float kl_div_neg;
  float kl_div_pos;
  double logsum_1_p_pos;
  double logsum_1_p_pos_rule;
  double logsum_1_p_pos_calib;
};

// A site and its sums.
struct LabeledSums {
  std::int64_t label_id;
  SiteSums sums;
};

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_SITE_SUMS_H_
