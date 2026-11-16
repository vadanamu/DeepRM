/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_ANNOTATION_H_
#define DEEPRM_PILEUP_ANNOTATION_H_

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

namespace deeprm {

// One transcript of an annotation. The exons are 0-based, half-open and
// sorted by genomic start.
struct Transcript {
  std::string name;
  std::string chrom;
  bool plus = true;
  std::vector<std::int64_t> exon_starts;
  std::vector<std::int64_t> exon_ends;
};

// The valid transcripts of a RefFlat (11 columns), GenePred (10) or RefGene
// (15) table, as parse_refflat of the Python pileup keeps them: in file
// order, the first of each name.
std::expected<std::vector<Transcript>, std::string> parse_annotation(
    std::string_view text);

std::expected<std::vector<Transcript>, std::string> read_annotation(
    const std::string& path);

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_ANNOTATION_H_
