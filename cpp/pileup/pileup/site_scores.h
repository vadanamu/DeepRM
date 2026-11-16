/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_SITE_SCORES_H_
#define DEEPRM_PILEUP_SITE_SCORES_H_

#include <cstdint>

#include "pileup/site_sums.h"

namespace deeprm {

// modscore and the stoichiometry outputs of one site (_site_scores of the
// Python pileup).
struct SiteScores {
  double modscore;
  double stoichiometry;
  std::int8_t stoich_state;
  double stoichiometry_release;
  double stoichiometry_eq14;
  // The value under the modscore lies so close to one of its 1/1000 steps
  // that NumPy's pow (SVML) could give the neighbouring step.
  bool modscore_near_step;
};

// np.digitize(x, np.linspace(0, 1, 1001), right=True): the number of steps
// i / 1000 below x (1001 for NaN).
int modscore_step(double x);

class SiteScorer {
 public:
  // mask: the stoichiometry of a site the rule does not report is written as
  // 0 (NaN in stoichiometry_release); false (--report-all-stoichiometry)
  // writes its value as for the other sites. Its state stays -1.
  SiteScorer(double threshold, double epsilon, bool mask = true);

  SiteScores score(const SiteSums& sums) const;

  // False when log10(1 - threshold) comes from the C library instead of a
  // value recorded from NumPy (only the default threshold is recorded).
  bool threshold_is_recorded() const { return threshold_is_recorded_; }

 private:
  double epsilon_;
  double log_1_threshold_;
  bool threshold_is_recorded_;
  bool mask_;
};

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_SITE_SCORES_H_
