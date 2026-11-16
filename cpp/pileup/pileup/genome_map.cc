/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include "pileup/genome_map.h"

#include <algorithm>
#include <unordered_map>

namespace deeprm {

std::string without_version(const std::string& name) {
  const std::size_t dot = name.rfind('.');
  if (dot == std::string::npos || dot == 0 || dot + 1 == name.size()) {
    return name;
  }
  const bool digits = std::all_of(name.begin() + dot + 1, name.end(),
                                  [](char c) { return c >= '0' && c <= '9'; });
  return digits ? name.substr(0, dot) : name;
}

GenomeMap::GenomeMap(const std::vector<std::string>& references,
                     const std::vector<Transcript>& transcripts)
    : references_(references.size()) {
  std::unordered_map<std::string, std::size_t> by_name;
  for (std::size_t i = 0; i < transcripts.size(); ++i) {
    by_name.emplace(transcripts[i].name, i);
  }
  std::vector<const Transcript*> found(references.size(), nullptr);
  for (std::size_t i = 0; i < references.size(); ++i) {
    auto at = by_name.find(references[i]);
    if (at != by_name.end()) {
      ++matched_exactly_;
    } else {
      at = by_name.find(without_version(references[i]));
      if (at != by_name.end()) ++matched_unversioned_;
    }
    if (at == by_name.end()) continue;
    found[i] = &transcripts[at->second];
    chroms_.push_back(found[i]->chrom);
  }
  std::ranges::sort(chroms_);
  chroms_.erase(std::unique(chroms_.begin(), chroms_.end()), chroms_.end());
  for (std::size_t i = 0; i < references.size(); ++i) {
    if (found[i] == nullptr) continue;
    references_[i].chrom = static_cast<std::uint32_t>(
        std::ranges::lower_bound(chroms_, found[i]->chrom) - chroms_.begin());
    add_exons(*found[i], references_[i]);
  }
}

void GenomeMap::add_exons(const Transcript& transcript, Reference& reference) {
  reference.annotated = true;
  reference.plus = transcript.plus;
  reference.first_exon = exon_start_.size();
  reference.exons = transcript.exon_starts.size();
  for (std::size_t i = 0; i < reference.exons; ++i) {
    const std::size_t exon = transcript.plus ? i : reference.exons - 1 - i;
    exon_offset_.push_back(reference.length);
    exon_start_.push_back(transcript.exon_starts[exon]);
    exon_end_.push_back(transcript.exon_ends[exon]);
    reference.length +=
        transcript.exon_ends[exon] - transcript.exon_starts[exon];
  }
}

std::optional<GenomicSite> GenomeMap::map(std::uint32_t reference,
                                          std::int64_t position,
                                          bool opposite) const {
  const Reference& ref = references_[reference];
  if (!ref.annotated || position < 0 || position >= ref.length) {
    return std::nullopt;
  }
  const auto first = exon_offset_.begin() +
                     static_cast<std::ptrdiff_t>(ref.first_exon);
  const auto last = first + static_cast<std::ptrdiff_t>(ref.exons);
  // the last exon that starts at or before the position
  const std::size_t exon = static_cast<std::size_t>(
      std::upper_bound(first, last, position) - exon_offset_.begin() - 1);
  const std::int64_t offset = position - exon_offset_[exon];
  return GenomicSite{
      .chrom = ref.chrom,
      // a site on the opposite strand is the same base on the other strand
      .minus = ref.plus == opposite,
      .position = ref.plus ? exon_start_[exon] + offset
                           : exon_end_[exon] - offset - 1};
}

}  // namespace deeprm
