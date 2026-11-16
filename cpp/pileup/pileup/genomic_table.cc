/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
// KahanSum adds as the group_sum of pandas does (Copyright (c) 2008-2011, AQR
// Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team;
// (c) 2011-2023, Open source contributors; BSD-3-Clause: see
// THIRD_PARTY_NOTICES).
#include "pileup/genomic_table.h"

#include <algorithm>
#include <cmath>

namespace deeprm {
namespace {

// A float64 sum as pandas' group_sum makes it.
class KahanSum {
 public:
  void add(double value) {
    if (std::isnan(value)) return;  // pandas skips missing values
    const double y = value - compensation_;
    const double t = sum_ + y;
    compensation_ = t - sum_ - y;
    if (std::isnan(compensation_)) compensation_ = 0;  // after an infinity
    sum_ = t;
  }
  double sum() const { return sum_; }

 private:
  double sum_ = 0;
  double compensation_ = 0;
};

// Where a row sorts, and which row it is.
struct Key {
  bool minus;
  std::int64_t position;
  std::uint32_t part;
  std::uint32_t row;
};

bool same_site(const Key& a, const Key& b) {
  return a.minus == b.minus && a.position == b.position;
}

// The sums of the rows keys[first, end), in that order.
SiteSums add_up(std::span<const std::span<const GenomicRow>> parts,
                const std::vector<Key>& keys, std::size_t first,
                std::size_t end) {
  SiteSums total;
  KahanSum floats[5];
  for (std::size_t i = first; i < end; ++i) {
    const SiteSums& sums = parts[keys[i].part][keys[i].row].sums;
    total.count_all += sums.count_all;
    total.count_pos += sums.count_pos;
    total.count_pos_rule += sums.count_pos_rule;
    floats[0].add(sums.logsum_1_p_pos);
    floats[1].add(sums.kl_div_neg);
    floats[2].add(sums.kl_div_pos);
    floats[3].add(sums.logsum_1_p_pos_rule);
    floats[4].add(sums.logsum_1_p_pos_calib);
  }
  total.logsum_1_p_pos = floats[0].sum();
  total.kl_div_neg = floats[1].sum();
  total.kl_div_pos = floats[2].sum();
  total.logsum_1_p_pos_rule = floats[3].sum();
  total.logsum_1_p_pos_calib = floats[4].sum();
  return total;
}

}  // namespace

std::vector<TableRow> add_up_genomic_rows(
    std::span<const std::span<const GenomicRow>> parts,
    std::uint32_t reference) {
  std::size_t rows = 0;
  for (const std::span<const GenomicRow> part : parts) rows += part.size();
  std::vector<Key> keys;
  keys.reserve(rows);
  for (std::size_t part = 0; part < parts.size(); ++part) {
    for (std::size_t row = 0; row < parts[part].size(); ++row) {
      keys.push_back(Key{parts[part][row].minus, parts[part][row].position,
                         static_cast<std::uint32_t>(part),
                         static_cast<std::uint32_t>(row)});
    }
  }
  // part and row as the last keys keep the rows of a site in their order
  std::ranges::sort(keys, [](const Key& a, const Key& b) {
    if (a.minus != b.minus) return b.minus;
    if (a.position != b.position) return a.position < b.position;
    if (a.part != b.part) return a.part < b.part;
    return a.row < b.row;
  });
  std::vector<TableRow> table;
  for (std::size_t first = 0; first < keys.size();) {
    std::size_t end = first + 1;
    while (end < keys.size() && same_site(keys[first], keys[end])) ++end;
    table.push_back(TableRow{.reference = reference,
                             .position = keys[first].position,
                             .minus = keys[first].minus,
                             .sums = add_up(parts, keys, first, end)});
    first = end;
  }
  return table;
}

}  // namespace deeprm
