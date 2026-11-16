/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_TESTS_UNIT_UNIT_TEST_H_
#define DEEPRM_TESTS_UNIT_UNIT_TEST_H_

#include <bit>
#include <cstdint>
#include <format>
#include <print>
#include <source_location>
#include <string_view>
#include <vector>

namespace deeprm::test {

// Collects the failed checks of one test case.
class Checker {
 public:
  void check(bool ok, std::string_view what,
             std::source_location where = std::source_location::current()) {
    if (ok) return;
    ++failures_;
    std::println("    {}:{}: {}", where.file_name(), where.line(), what);
  }

  template <typename Left, typename Right>
  void equal(const Left& left, const Right& right,
             std::source_location where = std::source_location::current()) {
    if (left == right) return;
    check(false, std::format("{} != {}", left, right), where);
  }

  int failures() const { return failures_; }

 private:
  int failures_ = 0;
};

struct TestCase {
  const char* name;
  void (*run)(Checker&);
};

inline std::uint32_t bits(float value) {
  return std::bit_cast<std::uint32_t>(value);
}

inline std::uint64_t bits(double value) {
  return std::bit_cast<std::uint64_t>(value);
}

// The test cases of each test file.
std::vector<TestCase> math_tests();
std::vector<TestCase> npz_tests();
std::vector<TestCase> site_tests();
std::vector<TestCase> table_tests();
std::vector<TestCase> setup_tests();
std::vector<TestCase> genome_tests();
std::vector<TestCase> modbam_tests();

}  // namespace deeprm::test

#endif  // DEEPRM_TESTS_UNIT_UNIT_TEST_H_
