/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_GENOMIC_TABLE_H_
#define DEEPRM_PILEUP_GENOMIC_TABLE_H_

#include <cstdint>
#include <span>
#include <vector>

#include "pileup/site_sums.h"
#include "pileup/table_writer.h"

namespace deeprm {

// A transcript site on one chromosome, with its sums.
struct GenomicRow {
  std::int64_t position;
  bool minus;
  SiteSums sums;
};

// The rows of one chromosome added up per (strand, position), '+' first, then
// by position; `reference` goes into every row. The rows come in parts
// that, one after another, are in the order of the transcript table: the
// float sums are those of pandas' groupby().sum(), which adds the rows of a
// group in order with Kahan's compensation.
std::vector<TableRow> add_up_genomic_rows(
    std::span<const std::span<const GenomicRow>> parts,
    std::uint32_t reference);

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_GENOMIC_TABLE_H_
