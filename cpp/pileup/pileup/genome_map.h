/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_GENOME_MAP_H_
#define DEEPRM_PILEUP_GENOME_MAP_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "pileup/annotation.h"

namespace deeprm {

// A site in genomic coordinates.
struct GenomicSite {
  std::uint32_t chrom;  // index into GenomeMap::chroms()
  bool minus;
  std::int64_t position;  // 0-based
};

// Where the references (transcripts) of a BAM header lie on the genome
// (GenomeMap of the Python pileup). A reference is found in the annotation by
// its exact name, or else by the name without its version (".N").
class GenomeMap {
 public:
  GenomeMap(const std::vector<std::string>& references,
            const std::vector<Transcript>& transcripts);

  // The chromosomes of the annotated references, sorted.
  const std::vector<std::string>& chroms() const { return chroms_; }

  // References found by their exact name, and only without the version.
  std::size_t matched_exactly() const { return matched_exactly_; }
  std::size_t matched_without_version() const { return matched_unversioned_; }

  // The genomic site of position `position` (0-based) of a reference;
  // `opposite`: the site is on the strand opposite the transcript. Nothing
  // for a reference that is not annotated or a position outside its exons.
  std::optional<GenomicSite> map(std::uint32_t reference, std::int64_t position,
                                 bool opposite) const;

 private:
  struct Reference {
    bool annotated = false;
    bool plus = true;
    std::uint32_t chrom = 0;
    std::size_t first_exon = 0;  // in the exon arrays below
    std::size_t exons = 0;
    std::int64_t length = 0;
  };

  void add_exons(const Transcript& transcript, Reference& reference);

  std::vector<Reference> references_;
  std::vector<std::string> chroms_;
  // The exons of each reference in transcript order (reversed for '-').
  std::vector<std::int64_t> exon_offset_;  // start within the transcript
  std::vector<std::int64_t> exon_start_;
  std::vector<std::int64_t> exon_end_;
  std::size_t matched_exactly_ = 0;
  std::size_t matched_unversioned_ = 0;
};

// The name without its version: "ENST00000123456.7" -> "ENST00000123456".
std::string without_version(const std::string& name);

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_GENOME_MAP_H_
