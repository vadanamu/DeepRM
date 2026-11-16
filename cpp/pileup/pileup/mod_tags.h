/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_MOD_TAGS_H_
#define DEEPRM_PILEUP_MOD_TAGS_H_

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pileup/read_rows.h"

namespace deeprm {

// The (high, low) int64 halves of a read name that is a UUID, as the per-read
// files hold them (the UUID's 16 bytes read as two native int64). Accepts the
// spellings Python's uuid.UUID does: hyphens, braces, "urn:uuid:".
std::optional<std::pair<std::int64_t, std::int64_t>> parse_read_id(
    std::string_view name);

// An alignment as the BAM record holds it.
struct Alignment {
  std::span<const std::uint32_t> cigar;  // length << 4 | operation
  std::int64_t reference_start;
  bool reverse;
  // Per base of the stored sequence: 1 where it is the modified base in the
  // read's original orientation (A; T on a reverse-strand read).
  std::span<const std::uint8_t> is_base;
};

// Makes the MM and ML tags of a read from its rows, as get_mm_tag and
// _mm_ml_batch of the Python pileup do. One per thread.
class ModTagMaker {
 public:
  // The tags of `alignment` from `rows`; ml() is empty when no row lies on
  // a modified base, and the read then gets no ML tag.
  void make(const Alignment& alignment, std::span<const ReadRow> rows);

  const std::string& mm() const { return mm_; }
  const std::vector<std::uint8_t>& ml() const { return ml_; }

 private:
  // A run of bases aligned to the reference without a gap.
  struct Block {
    std::int64_t reference_start;
    std::int64_t query_start;
    std::int64_t length;
  };
  // A row on the read: where it lies in the original orientation and in the
  // stored sequence.
  struct Call {
    std::int64_t original;
    std::int64_t stored;
    std::uint8_t probability;
  };

  void find_blocks(const Alignment& alignment);
  void find_calls(const Alignment& alignment, std::span<const ReadRow> rows);
  void write_tags(const Alignment& alignment);

  std::vector<Block> blocks_;
  std::vector<Call> calls_;
  std::vector<std::uint32_t> bases_before_;  // modified bases before each base
  std::string mm_;
  std::vector<std::uint8_t> ml_;
};

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_MOD_TAGS_H_
