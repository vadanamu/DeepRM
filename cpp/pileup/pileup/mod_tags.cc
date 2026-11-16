/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include "pileup/mod_tags.h"

#include <algorithm>
#include <charconv>
#include <cstring>

namespace deeprm {
namespace {

constexpr std::string_view kNoCalls = "A+a?;";

// CIGAR operations M I D N S H P = X as numbers 0-8. P is counted as read
// bases, as pysam's get_aligned_pairs() counts it.
bool consumes_query(std::uint32_t op) {
  return op == 0 || op == 1 || op == 4 || op == 6 || op == 7 || op == 8;
}
bool consumes_reference(std::uint32_t op) {
  return op == 0 || op == 2 || op == 3 || op == 7 || op == 8;
}
bool aligns(std::uint32_t op) { return op == 0 || op == 7 || op == 8; }

int hex_value(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

void remove_all(std::string& text, std::string_view part) {
  for (std::size_t at = text.find(part); at != std::string::npos;
       at = text.find(part, at)) {
    text.erase(at, part.size());
  }
}

}  // namespace

std::optional<std::pair<std::int64_t, std::int64_t>> parse_read_id(
    std::string_view name) {
  std::string hex(name);
  if (hex.size() != 36) {  // not the usual spelling: what uuid.UUID removes
    remove_all(hex, "urn:");
    remove_all(hex, "uuid:");
    while (!hex.empty() && (hex.front() == '{' || hex.front() == '}')) {
      hex.erase(0, 1);
    }
    while (!hex.empty() && (hex.back() == '{' || hex.back() == '}')) {
      hex.pop_back();
    }
  }
  remove_all(hex, "-");
  if (hex.size() != 32) return std::nullopt;
  std::uint8_t bytes[16];
  for (std::size_t i = 0; i < 16; ++i) {
    const int high = hex_value(hex[2 * i]);
    const int low = hex_value(hex[2 * i + 1]);
    if (high < 0 || low < 0) return std::nullopt;
    bytes[i] = static_cast<std::uint8_t>(high * 16 + low);
  }
  std::pair<std::int64_t, std::int64_t> id;
  std::memcpy(&id.first, bytes, 8);
  std::memcpy(&id.second, bytes + 8, 8);
  return id;
}

void ModTagMaker::find_blocks(const Alignment& alignment) {
  blocks_.clear();
  std::int64_t query = 0;
  std::int64_t reference = alignment.reference_start;
  for (const std::uint32_t item : alignment.cigar) {
    const std::uint32_t op = item & 0xF;
    const std::int64_t length = item >> 4;
    if (aligns(op)) blocks_.push_back(Block{reference, query, length});
    if (consumes_query(op)) query += length;
    if (consumes_reference(op)) reference += length;
  }
}

// The rows that lie on an aligned base, sorted by where they lie on the read.
void ModTagMaker::find_calls(const Alignment& alignment,
                             std::span<const ReadRow> rows) {
  calls_.clear();
  const std::int64_t bases =
      static_cast<std::int64_t>(alignment.is_base.size());
  for (const ReadRow& row : rows) {
    // the last block that starts at or before the row's position
    const auto after = std::ranges::upper_bound(blocks_, row.position, {},
                                                &Block::reference_start);
    if (after == blocks_.begin()) continue;
    const Block& block = *(after - 1);
    if (row.position >= block.reference_start + block.length) continue;
    const std::int64_t stored =
        block.query_start + (row.position - block.reference_start);
    calls_.push_back(Call{
        .original = alignment.reverse ? bases - 1 - stored : stored,
        .stored = stored,
        .probability = row.probability});
  }
  std::ranges::sort(calls_, {}, &Call::original);
}

void ModTagMaker::write_tags(const Alignment& alignment) {
  const std::span<const std::uint8_t> is_base = alignment.is_base;
  bases_before_.resize(is_base.size() + 1);
  bases_before_[0] = 0;
  for (std::size_t i = 0; i < is_base.size(); ++i) {
    bases_before_[i + 1] = bases_before_[i] + is_base[i];
  }
  mm_.assign("A+a?");
  ml_.clear();
  std::int64_t last_rank = -1;
  for (std::size_t first = 0; first < calls_.size();) {
    std::size_t end = first + 1;
    bool agree = true;
    while (end < calls_.size() && calls_[end].original == calls_[first].original) {
      agree = agree && calls_[end].probability == calls_[first].probability;
      ++end;
    }
    const Call& call = calls_[first];
    first = end;
    // a base listed twice counts only when its rows agree
    if (!agree || call.stored < 0 ||
        call.stored >= static_cast<std::int64_t>(is_base.size()) ||
        is_base[static_cast<std::size_t>(call.stored)] == 0) {
      continue;
    }
    // the modified bases before this one in the original orientation
    const std::size_t stored = static_cast<std::size_t>(call.stored);
    const std::int64_t rank =
        alignment.reverse ? bases_before_.back() - bases_before_[stored + 1]
                          : bases_before_[stored];
    char text[24];
    const std::to_chars_result number =
        std::to_chars(text, text + sizeof(text), rank - last_rank - 1);
    mm_.push_back(',');
    mm_.append(text, number.ptr);
    ml_.push_back(call.probability);
    last_rank = rank;
  }
  mm_.push_back(';');
}

void ModTagMaker::make(const Alignment& alignment,
                       std::span<const ReadRow> rows) {
  find_blocks(alignment);
  find_calls(alignment, rows);
  write_tags(alignment);
  if (ml_.empty()) mm_.assign(kNoCalls);
}

}  // namespace deeprm
