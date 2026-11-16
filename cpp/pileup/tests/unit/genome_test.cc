/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include <cmath>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "pileup/annotation.h"
#include "pileup/genome_map.h"
#include "pileup/genomic_table.h"
#include "tests/unit/unit_test.h"

namespace deeprm::test {
namespace {

// T3 has one exon end too few and T4 unsorted exons: both are dropped, as is
// the second T2.
constexpr std::string_view kAnnotation =
    "g\tT1.2\tchrB\t+\t100\t200\t100\t200\t2\t100,150,\t110,160,\n"
    "g\tT2\tchrA\t-\t100\t200\t100\t200\t2\t100,150,\t110,160,\n"
    "g\tT2\tchrC\t-\t100\t200\t100\t200\t1\t100,\t110,\n"
    "g\tT3\tchrA\t+\t100\t200\t100\t200\t2\t100,150,\t110,\n"
    "g\tT4\tchrA\t+\t100\t200\t100\t200\t2\t150,100,\t160,110,\n";

void annotation_keeps_valid_transcripts(Checker& t) {
  const auto transcripts = parse_annotation(kAnnotation);
  t.check(transcripts.has_value() && transcripts->size() == 2, "T1.2 and T2");
  if (!transcripts || transcripts->size() != 2) return;
  t.equal((*transcripts)[0].name, std::string("T1.2"));
  t.equal((*transcripts)[1].chrom, std::string("chrA"));
  t.check(!(*transcripts)[1].plus, "T2 is on '-'");
  t.equal((*transcripts)[1].exon_ends[1], std::int64_t{160});
}

void annotation_refuses_bad_tables(Checker& t) {
  t.check(!parse_annotation("a\tb\tc\n").has_value(), "3 columns");
  t.check(!parse_annotation("T\tchr1\t.\t1\t2\t1\t2\t1\t1,\t2,\n").has_value(),
          "a strand that is not + or -");
  t.check(!parse_annotation("T\tchr1\t+\t5\t2\t1\t2\t1\t1,\t2,\n").has_value(),
          "no valid transcript");
  t.check(parse_annotation("T\tchr1\t+\t1\t2\t1\t2\t1\t1\t2\n").has_value(),
          "GenePred columns, lists without the trailing comma");
}

void versions_are_removed(Checker& t) {
  t.equal(without_version("ENST00000123456.7"), std::string("ENST00000123456"));
  t.equal(without_version("NM_005517.14"), std::string("NM_005517"));
  t.equal(without_version("a.b.3"), std::string("a.b"));
  t.equal(without_version("T1"), std::string("T1"));
  t.equal(without_version("T1."), std::string("T1."));
  t.equal(without_version("T1.x"), std::string("T1.x"));
  t.equal(without_version(".5"), std::string(".5"));
}

// The sites, and what GenomeMap.map of the Python pileup returns for them.
void genome_map_matches_the_python_pileup(Checker& t) {
  const auto transcripts = parse_annotation(kAnnotation);
  if (!transcripts) return t.check(false, "the annotation parses");
  const GenomeMap genome({"T1.2", "T2.5", "T9", "T3"}, *transcripts);
  t.equal(genome.chroms().size(), std::size_t{2});
  t.equal(genome.chroms()[0], std::string("chrA"));
  t.equal(genome.matched_exactly(), std::size_t{1});
  t.equal(genome.matched_without_version(), std::size_t{1});
  struct Case {
    std::uint32_t reference;
    std::int64_t position;
    bool opposite;
    std::uint32_t chrom;
    bool minus;
    std::int64_t genomic;
  };
  const Case cases[] = {
      {0, 0, false, 1, false, 100}, {0, 9, false, 1, false, 109},
      {0, 10, true, 1, true, 150},  {0, 19, false, 1, false, 159},
      {1, 0, false, 0, true, 159},  {1, 9, true, 0, false, 150},
      {1, 10, false, 0, true, 109}, {1, 19, false, 0, true, 100}};
  for (const Case& c : cases) {
    const auto site = genome.map(c.reference, c.position, c.opposite);
    t.check(site.has_value(), "the site maps");
    if (!site) continue;
    t.equal(site->chrom, c.chrom);
    t.equal(site->minus, c.minus);
    t.equal(site->position, c.genomic);
  }
  t.check(!genome.map(2, 0, false), "a reference not in the annotation");
  t.check(!genome.map(3, 0, false), "a reference whose row was dropped");
  t.check(!genome.map(0, 20, false), "past the last exon");
  t.check(!genome.map(1, -1, false), "before the first exon");
}

GenomicRow row(std::int64_t position, bool minus, double value) {
  GenomicRow out{.position = position, .minus = minus, .sums = {}};
  out.sums.count_all = 1;
  out.sums.kl_div_neg = value;
  return out;
}

// Sums recorded from pandas 2.2.3: df.groupby("k").v.sum().
void genomic_rows_add_up_as_pandas_does(Checker& t) {
  std::vector<GenomicRow> rows = {row(7, true, 5.0), row(9, false, 1.0)};
  for (int i = 0; i < 10; ++i) rows.push_back(row(9, false, 1e-16));
  rows.push_back(row(3, false, -0.0));
  rows.push_back(row(9, false, 0.1));
  rows.push_back(row(9, false, 0.7));
  rows.push_back(row(9, false, std::numeric_limits<double>::quiet_NaN()));
  // in two parts, as the rows of two site tables come
  const std::span<const GenomicRow> all(rows);
  const std::vector<std::span<const GenomicRow>> parts = {all.first(5),
                                                          all.subspan(5)};
  const std::vector<TableRow> table = add_up_genomic_rows(parts, 4);
  t.equal(table.size(), std::size_t{3});
  if (table.size() != 3) return;
  t.check(!table[0].minus && table[0].position == 3, "'+' first, by position");
  t.check(!table[1].minus && table[1].position == 9, "then position 9");
  t.check(table[2].minus && table[2].position == 7, "'-' last");
  t.equal(table[1].reference, std::uint32_t{4});
  t.equal(table[1].sums.count_all, std::int64_t{14});
  t.equal(table[1].sums.kl_div_neg, 0x1.cccccccccccd1p+0);
  t.check(!std::signbit(table[0].sums.kl_div_neg), "-0.0 alone sums to 0.0");
}

}  // namespace

std::vector<TestCase> genome_tests() {
  return {{"annotation keeps valid transcripts",
           annotation_keeps_valid_transcripts},
          {"annotation refuses bad tables", annotation_refuses_bad_tables},
          {"transcript versions are removed", versions_are_removed},
          {"genome map matches the Python pileup",
           genome_map_matches_the_python_pileup},
          {"genomic rows add up as pandas does",
           genomic_rows_add_up_as_pandas_does}};
}

}  // namespace deeprm::test
