/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include <cmath>
#include <vector>

#include "pileup/site_scores.h"
#include "pileup/site_table.h"
#include "tests/unit/unit_test.h"

namespace deeprm::test {
namespace {

FileSite file_site(std::int64_t label_id, double logsum, float kl_div_pos) {
  return FileSite{.label_id = label_id,
                  .count_all = 2,
                  .count_pos = 1,
                  .count_pos_rule = 1,
                  .kl_div_neg = 0.0F,
                  .kl_div_pos = kl_div_pos,
                  .logsum_1_p_pos = logsum,
                  .logsum_1_p_pos_rule = 0.0,
                  .logsum_1_p_pos_calib = 0.0};
}

void table_sorts_and_counts(Checker& t) {
  SiteTable table;
  const std::vector<FileSite> first = {file_site(-7, 0.0, 0.0F),
                                       file_site(5, 0.0, 0.0F)};
  const std::vector<FileSite> second = {file_site(-9, 0.0, 0.0F),
                                        file_site(5, 0.0, 0.0F)};
  table.add(first, 0);
  table.add(second, 0);
  const std::vector<LabeledSums> sites = table.finish();
  t.equal(sites.size(), std::size_t{3});
  if (sites.size() != 3) return;
  t.equal(sites[0].label_id, std::int64_t{-9});
  t.equal(sites[1].label_id, std::int64_t{-7});
  t.equal(sites[2].label_id, std::int64_t{5});
  t.equal(sites[2].sums.count_all, std::int64_t{4});
  t.equal(sites[2].sums.count_pos, std::int64_t{2});
}

// (1e16 + 1) - 1e16 is 0 in one sum, but 1 when the group that holds 1e16
// and -1e16 is added up on its own.
double sum_in_groups(const std::int32_t groups[3]) {
  SiteTable table;
  const double terms[3] = {1.0, 1e16, -1e16};
  for (int i = 0; i < 3; ++i) {
    const std::vector<FileSite> file = {file_site(1, terms[i], 0.0F)};
    table.add(file, groups[i]);
  }
  return table.finish()[0].sums.logsum_1_p_pos;
}

void table_adds_groups_separately(Checker& t) {
  const std::int32_t one_group[3] = {0, 0, 0};
  const std::int32_t two_groups[3] = {0, 1, 1};
  t.equal(sum_in_groups(one_group), 0.0);
  t.equal(sum_in_groups(two_groups), 1.0);
}

void table_keeps_many_sites(Checker& t) {
  SiteTable table;
  std::vector<FileSite> file;
  for (std::int64_t label = 1; label <= 100000; ++label) {
    file.push_back(file_site(label * 1000003 % 999983 + 1, 0.0, 0.5F));
  }
  table.add(file, 0);
  table.add(file, 1);
  const std::vector<LabeledSums> sites = table.finish();
  bool ok = true;
  for (std::size_t i = 0; i < sites.size(); ++i) {
    ok = ok && sites[i].sums.kl_div_pos == 1.0 &&
         (i == 0 || sites[i - 1].label_id < sites[i].label_id);
  }
  t.check(ok, "every site has both files, in label order");
}

void modscore_steps_match_digitize(Checker& t) {
  // np.digitize(x, np.linspace(0, 1, 1001), right=True)
  t.equal(modscore_step(0.0), 0);
  t.equal(modscore_step(1e-300), 1);
  t.equal(modscore_step(0.001), 1);
  t.equal(modscore_step(0.0010000000000000002), 2);
  t.equal(modscore_step(0.5), 500);
  t.equal(modscore_step(1.0), 1000);
  t.equal(modscore_step(1.5), 1001);
  t.equal(modscore_step(-1.0), 0);
  t.equal(modscore_step(std::nan("")), 1001);
}

// The expected values were recorded from _site_scores(sums, 0.98, 1e-30).
void scores_match_the_python_pileup(Checker& t) {
  const SiteSums sums[] = {
      {5865, 172, 172, -382.824960231781, 3996.0291093066335,
       428.23496298200916, -382.824960231781, -120.5},
      {10, 0, 0, 0.0, 7.5, 0.0, 0.0, 0.0},
      {100, 60, 60, -150.5, 20.25, 55.5, -150.5, -140.25},
      {3, 3, 3, -9.25, 0.0, 2.75, -9.25, -8.5},
      {2000, 4, 4, -7.5, 1500.5, 3.25, -7.5, -1.5}};
  const double modscore[] = {0x1.22d0e56041893p-3, 0.0, 0x1.fa5e353f7ced9p-1,
                             1.0, 0x1.26e978d4fdf3bp-7};
  const double stoichiometry[] = {0.0, 0.0, 0x1.7720f353a4c0ap-1, 1.0, 0.0};
  const int state[] = {-1, -1, 1, 1, -1};
  const double eq14[] = {0.0, 0.0, 0x1.7720f353a4c0ap-1, 1.0,
                         0x1.1b47f82b3dc14p-9};
  const SiteScorer scorer(0.98, 1e-30);
  for (int i = 0; i < 5; ++i) {
    const SiteScores scores = scorer.score(sums[i]);
    t.equal(bits(scores.modscore), bits(modscore[i]));
    t.equal(bits(scores.stoichiometry), bits(stoichiometry[i]));
    t.equal(static_cast<int>(scores.stoich_state), state[i]);
    t.equal(bits(scores.stoichiometry_eq14), bits(eq14[i]));
    if (state[i] == 1) {
      t.equal(bits(scores.stoichiometry_release), bits(stoichiometry[i]));
    } else {
      t.equal(bits(scores.stoichiometry_release), 0x7ff8000000000000ULL);
    }
    t.check(!scores.modscore_near_step, "far from a modscore step");
  }
}

// --report-all-stoichiometry: the sites the rule does not report get the value
// the Python pileup writes with mask=False; the states and the other sites do
// not change.
void unmasked_scores_match_the_python_pileup(Checker& t) {
  const SiteSums sums[] = {
      {5865, 172, 172, -382.824960231781, 3996.0291093066335,
       428.23496298200916, -382.824960231781, -120.5},
      {10, 0, 0, 0.0, 7.5, 0.0, 0.0, 0.0},
      {100, 60, 60, -150.5, 20.25, 55.5, -150.5, -140.25},
      {3, 3, 3, -9.25, 0.0, 2.75, -9.25, -8.5},
      {2000, 4, 4, -7.5, 1500.5, 3.25, -7.5, -1.5}};
  const double stoichiometry[] = {0x1.e07c4fca39877p-6, 0.0,
                                  0x1.7720f353a4c0ap-1, 1.0,
                                  0x1.1b47f82b3dc14p-9};
  const int state[] = {-1, -1, 1, 1, -1};
  const SiteScorer masked(0.98, 1e-30);
  const SiteScorer scorer(0.98, 1e-30, false);
  for (int i = 0; i < 5; ++i) {
    const SiteScores scores = scorer.score(sums[i]);
    const SiteScores as_masked = masked.score(sums[i]);
    t.equal(bits(scores.stoichiometry), bits(stoichiometry[i]));
    t.equal(bits(scores.stoichiometry_release), bits(stoichiometry[i]));
    t.equal(static_cast<int>(scores.stoich_state), state[i]);
    t.equal(bits(scores.modscore), bits(as_masked.modscore));
    t.equal(bits(scores.stoichiometry_eq14), bits(as_masked.stoichiometry_eq14));
  }
}

}  // namespace

std::vector<TestCase> site_tests() {
  return {{"site table sorts and counts", table_sorts_and_counts},
          {"site table adds worker groups separately",
           table_adds_groups_separately},
          {"site table keeps many sites", table_keeps_many_sites},
          {"modscore steps match np.digitize", modscore_steps_match_digitize},
          {"site scores match the Python pileup",
           scores_match_the_python_pileup},
          {"unmasked site scores match the Python pileup",
           unmasked_scores_match_the_python_pileup}};
}

}  // namespace deeprm::test
