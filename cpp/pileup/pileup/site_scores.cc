/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include "pileup/site_scores.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace deeprm {
namespace {

// Constants of the stoichiometry rule (see the Python pileup).
constexpr double kRuleCut = 0.98;
constexpr double kHalf = 0.5;
constexpr double kBackgroundRate = 504.0 / 100000.0;
// scipy.stats.norm.ppf(0.99)
constexpr double kNormalQuantile = 0x1.29c5c4630ff0ep+1;
// np.log10(1 - 0.98) as NumPy 2.4.6 gives it on AVX512 (SVML); the C
// library's log10 gives the neighbouring double.
constexpr double kLog10OfOneMinusRuleCut = -0x1.b2efb2bd8217fp+0;
constexpr int kSteps = 1000;
// Distance to a step within which the two pow implementations may disagree.
constexpr double kStepMargin = 1e-14;

double step_value(int step) {
  return step == kSteps ? 1.0 : static_cast<double>(step) * (1.0 / kSteps);
}

// One-sided Wilson upper bound of the proportion p on n reads.
double wilson_upper(double p, double n) {
  const double z = kNormalQuantile;
  p = std::min(std::max(p, 0.0), 1.0);
  n = std::max(n, 1.0);
  return (p + z * z / (2 * n) +
          z * std::sqrt(p * (1 - p) / n + z * z / (4 * n * n))) /
         (1 + z * z / n);
}

}  // namespace

int modscore_step(double x) {
  if (std::isnan(x)) return kSteps + 1;
  int low = 0;
  int high = kSteps + 1;
  while (low < high) {
    const int middle = (low + high) / 2;
    if (step_value(middle) < x) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low;
}

SiteScorer::SiteScorer(double threshold, double epsilon, bool mask)
    : epsilon_(epsilon),
      log_1_threshold_(threshold == kRuleCut ? kLog10OfOneMinusRuleCut
                                             : std::log10(1 - threshold)),
      threshold_is_recorded_(threshold == kRuleCut),
      mask_(mask) {}

SiteScores SiteScorer::score(const SiteSums& sums) const {
  const double n = static_cast<double>(sums.count_all);
  const double k = static_cast<double>(sums.count_pos_rule);
  const double eq12 =
      sums.kl_div_pos / (sums.kl_div_neg + sums.kl_div_pos + epsilon_);
  const double exponent =
      sums.logsum_1_p_pos / n * (1 + std::pow(10.0, 2 * (eq12 - 1)));
  const double raw = 1 - std::pow(10.0, exponent);
  const int step = modscore_step(raw);
  const bool near_low = step > 0 && raw - step_value(step - 1) <= kStepMargin;
  const bool near_high =
      step <= kSteps && step_value(step) - raw <= kStepMargin;
  // Eq. 14 at --threshold (the v1.1 value) and at the rule's cut.
  const double eq14 =
      eq12 * (log_1_threshold_ * eq12 > sums.logsum_1_p_pos / n ? 1.0 : 0.0);
  const double s =
      eq12 * (kLog10OfOneMinusRuleCut * eq12 > sums.logsum_1_p_pos_rule / n
                  ? 1.0
                  : 0.0);
  const double evidence = sums.logsum_1_p_pos_calib / kLog10OfOneMinusRuleCut;
  const double n1 = std::max(n, 1.0);
  const bool has_s = s > 0;
  const bool h_holds = has_s && evidence > n * std::min(s, kHalf);
  const bool l_holds = sums.count_pos_rule >= 1 && n > 0 &&
                       wilson_upper(k / n1, n) < kBackgroundRate;
  const bool reported = h_holds || l_holds;
  const double value = has_s ? s : k / n1;
  const bool written = reported || !mask_;
  return SiteScores{
      .modscore = static_cast<double>(step) / kSteps,
      .stoichiometry = written ? value : 0.0,
      .stoich_state = static_cast<std::int8_t>(reported ? 1 : -1),
      .stoichiometry_release =
          written ? value : std::numeric_limits<double>::quiet_NaN(),
      .stoichiometry_eq14 = eq14,
      .modscore_near_step = exponent != 0 && (near_low || near_high)};
}

}  // namespace deeprm
