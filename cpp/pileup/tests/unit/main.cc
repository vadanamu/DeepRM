/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include <cstdlib>
#include <print>
#include <vector>

#include "tests/unit/unit_test.h"

int main() {
  using deeprm::test::TestCase;
  std::vector<TestCase> cases;
  for (const std::vector<TestCase>& group :
       {deeprm::test::math_tests(), deeprm::test::npz_tests(),
        deeprm::test::site_tests(), deeprm::test::table_tests(),
        deeprm::test::setup_tests(), deeprm::test::genome_tests(),
        deeprm::test::modbam_tests()}) {
    cases.insert(cases.end(), group.begin(), group.end());
  }
  int failed = 0;
  for (const TestCase& test : cases) {
    deeprm::test::Checker checker;
    test.run(checker);
    std::println("{} {}", checker.failures() == 0 ? "PASS" : "FAIL", test.name);
    failed += checker.failures() == 0 ? 0 : 1;
  }
  std::println("{} tests, {} failed", cases.size(), failed);
  return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
