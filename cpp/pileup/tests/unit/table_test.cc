/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include <string>
#include <vector>

#include "pileup/table_writer.h"
#include "tests/unit/unit_test.h"

namespace deeprm::test {
namespace {

// The rows and the expected lines were recorded from bed_formatter() of the
// Python pileup on the sums of site_test.cc.
void bed_lines_match_the_python_pileup(Checker& t) {
  const std::vector<std::string> names = {"chr1", "chrM", "a\"b", "chr2"};
  const std::vector<TableRow> rows = {
      {0, 10, false,
       {5865, 172, 172, -382.824960231781, 3996.0291093066335,
        428.23496298200916, -382.824960231781, -120.5}},
      {1, 16567, true, {10, 0, 0, 0.0, 7.5, 0.0, 0.0, 0.0}},
      {2, 5, false, {100, 60, 60, -150.5, 20.25, 55.5, -150.5, -140.25}},
      {3, 7, true, {3, 3, 3, -9.25, 0.0, 2.75, -9.25, -8.5}},
      {3, 8, false, {2000, 4, 4, -7.5, 1500.5, 3.25, -7.5, -1.5}}};
  const std::string expected =
      "chr1\t10\t11\ta\t142\t+\t10\t11\t255,0,0\t5865\t0.00\t0\t5865\t0\t0\t0\t0\t0\n"
      "chrM\t16567\t16568\ta\t0\t-\t16567\t16568\t255,0,0\t10\t0.00\t0\t10\t0\t0\t0\t0\t0\n"
      "\"a\"\"b\"\t5\t6\ta\t989\t+\t5\t6\t255,0,0\t100\t73.27\t73\t27\t0\t0\t0\t0\t0\n"
      "chr2\t7\t8\ta\t1000\t-\t7\t8\t255,0,0\t3\t100.00\t3\t0\t0\t0\t0\t0\t0\n"
      "chr2\t8\t9\ta\t9\t+\t8\t9\t255,0,0\t2000\t0.00\t0\t2000\t0\t0\t0\t0\t0\n";
  const SiteScorer scorer(0.98, 1e-30);
  const TableFormat format(names, scorer);
  const TablePart part = format.make_part(rows);
  t.equal(part.bed, expected);
  t.equal(part.rows, std::size_t{5});
  t.equal(part.runs.size(), std::size_t{13});
}

void npz_arrays_have_the_python_dtypes(Checker& t) {
  const std::vector<std::string> names = {"chr1", "chr10_alt"};
  const SiteScorer scorer(0.98, 1e-30);
  const TableFormat format(names, scorer);
  const std::vector<NpzArray> arrays = format.empty_arrays();
  t.equal(arrays.size(), std::size_t{13});
  t.equal(arrays[0].name, std::string("ref_names"));
  t.equal(arrays[0].descr, std::string("<U9"));  // the longest name
  t.equal(arrays[2].descr, std::string("<U1"));
  t.equal(arrays[7].name, std::string("stoich_state"));
  t.equal(arrays[7].descr, std::string("|i1"));
  t.equal(arrays[12].name, std::string("kl_div_pos"));
  // 2 rows of a 9-character name: 2 * 9 * 4 bytes
  const std::vector<TableRow> rows = {{0, 1, false, {1, 0, 0, 0, 1, 0, 0, 0}},
                                      {1, 2, true, {1, 0, 0, 0, 1, 0, 0, 0}}};
  t.equal(format.make_part(rows).runs[0].size, std::uint64_t{72});
}

}  // namespace

std::vector<TestCase> table_tests() {
  return {{"BED lines match the Python pileup",
           bed_lines_match_the_python_pileup},
          {"npz arrays have the Python dtypes",
           npz_arrays_have_the_python_dtypes}};
}

}  // namespace deeprm::test
