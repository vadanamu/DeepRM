/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include "pileup/site_table.h"

#include <algorithm>
#include <cstddef>

namespace deeprm {
namespace {

constexpr unsigned kInitialBits = 12;
// Sites looked up together in SiteTable::add.
constexpr std::size_t kLookups = 16;
constexpr std::uint64_t kHashMultiplier = 0x9E3779B97F4A7C15ULL;

std::size_t hash(std::int64_t label_id, unsigned shift) {
  return static_cast<std::size_t>(
      (static_cast<std::uint64_t>(label_id) * kHashMultiplier) >> shift);
}

}  // namespace

SiteTable::SiteTable()
    : index_(std::size_t{1} << kInitialBits, 0), shift_(64 - kInitialBits) {}

void SiteTable::grow() {
  --shift_;
  index_.assign(index_.size() * 2, 0);
  const std::size_t mask = index_.size() - 1;
  for (std::size_t number = 0; number < size_; ++number) {
    std::size_t at = hash(slot(number).label_id, shift_);
    while (index_[at] != 0) at = (at + 1) & mask;
    index_[at] = static_cast<std::uint32_t>(number + 1);
  }
}

SiteTable::Slot& SiteTable::find_or_insert(std::int64_t label_id) {
  if ((size_ + 1) * 2 > index_.size()) grow();
  const std::size_t mask = index_.size() - 1;
  std::size_t at = hash(label_id, shift_);
  while (index_[at] != 0) {
    Slot& found = slot(index_[at] - 1);
    if (found.label_id == label_id) return found;
    at = (at + 1) & mask;
  }
  if (size_ == blocks_.size() * kBlockSlots) {
    blocks_.push_back(std::make_unique_for_overwrite<Slot[]>(kBlockSlots));
  }
  Slot& added = slot(size_);
  added = Slot{.label_id = label_id,
               .count_all = 0,
               .count_pos = 0,
               .count_pos_rule = 0,
               .group = -1,
               .group_sums = {0.0, 0.0, 0.0, 0.0, 0.0},
               .total = {0.0, 0.0, 0.0, 0.0, 0.0}};
  index_[at] = static_cast<std::uint32_t>(++size_);
  return added;
}

// Adds the group's float sums to the total, as 1.2 adds a worker's table.
void SiteTable::close_group(Slot& slot) {
  for (std::size_t i = 0; i < 5; ++i) {
    slot.total[i] += slot.group_sums[i];
    slot.group_sums[i] = 0.0;
  }
}

void SiteTable::add_to(Slot& slot, const FileSite& site, std::int32_t group) {
  if (slot.group != group) {
    close_group(slot);
    slot.group = group;
  }
  slot.count_all += site.count_all;
  slot.count_pos += site.count_pos;
  slot.count_pos_rule += site.count_pos_rule;
  slot.group_sums[0] += site.logsum_1_p_pos;
  slot.group_sums[1] += static_cast<double>(site.kl_div_neg);
  slot.group_sums[2] += static_cast<double>(site.kl_div_pos);
  slot.group_sums[3] += site.logsum_1_p_pos_rule;
  slot.group_sums[4] += site.logsum_1_p_pos_calib;
}

// The table is far larger than a core's cache, so finding a site's slot
// waits for memory twice (the index, then the slot). The sites are therefore
// looked up a few at a time: the waits of one group overlap, as none of its
// lookups depends on another.
void SiteTable::add(std::span<const FileSite> sites, std::int32_t group) {
  std::uint32_t number[kLookups];
  std::int64_t found[kLookups];
  for (std::size_t first = 0; first < sites.size(); first += kLookups) {
    const std::size_t count = std::min(kLookups, sites.size() - first);
    for (std::size_t i = 0; i < count; ++i) {
      number[i] = index_[hash(sites[first + i].label_id, shift_)];
    }
    for (std::size_t i = 0; i < count; ++i) {
      found[i] = number[i] != 0 ? slot(number[i] - 1).label_id : 0;
    }
    for (std::size_t i = 0; i < count; ++i) {
      const FileSite& site = sites[first + i];
      // a slot numbered before stays the site's; any other case (an empty
      // or another site's place in the index) is looked up in full
      const bool hit = number[i] != 0 && found[i] == site.label_id;
      add_to(hit ? slot(number[i] - 1) : find_or_insert(site.label_id), site,
             group);
    }
  }
}

std::vector<LabeledSums> SiteTable::finish() {
  std::vector<LabeledSums> sites;
  sites.reserve(size_);
  index_ = std::vector<std::uint32_t>();  // "= {}" would keep the memory
  for (std::size_t block = 0; block < blocks_.size(); ++block) {
    const std::size_t slots = std::min(kBlockSlots, size_ - block * kBlockSlots);
    for (std::size_t i = 0; i < slots; ++i) {
      Slot& slot = blocks_[block][i];
      close_group(slot);
      sites.push_back(LabeledSums{
          .label_id = slot.label_id,
          .sums = {.count_all = slot.count_all,
                   .count_pos = slot.count_pos,
                   .count_pos_rule = slot.count_pos_rule,
                   .logsum_1_p_pos = slot.total[0],
                   .kl_div_neg = slot.total[1],
                   .kl_div_pos = slot.total[2],
                   .logsum_1_p_pos_rule = slot.total[3],
                   .logsum_1_p_pos_calib = slot.total[4]}});
    }
    blocks_[block].reset();  // freed as it is copied
  }
  blocks_.clear();
  blocks_.shrink_to_fit();
  size_ = 0;
  std::ranges::sort(sites, {}, &LabeledSums::label_id);
  return sites;
}

}  // namespace deeprm
