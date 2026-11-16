/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_SITE_TABLE_H_
#define DEEPRM_PILEUP_SITE_TABLE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "pileup/site_sums.h"

namespace deeprm {

// The per-site sums of one range of sites, added up file by file.
//
// The float sums are those of DeepRM 1.2 to the bit: 1.2 adds the files of a
// worker, in file order, into a sum that starts at 0, and then adds the
// workers in order into another. add() must therefore get the files in file
// order, each with the worker group it belongs to.
//
// Aligned to a cache line: the tables lie side by side and each is filled
// by another thread.
class alignas(64) SiteTable {
 public:
  SiteTable();

  // Adds the sites of one file, which belongs to worker group `group`.
  void add(std::span<const FileSite> sites, std::int32_t group);

  // All sites sorted by label_id; the table is empty afterwards.
  std::vector<LabeledSums> finish();

 private:
  // The float sums of the worker group that last touched the site are kept
  // apart from the total until another group touches it.
  // The counts are 32-bit to keep a slot small: a site has far fewer rows
  // than 2^32 (a run has fewer reads than that).
  struct Slot {
    std::int64_t label_id;
    std::uint32_t count_all;
    std::uint32_t count_pos;
    std::uint32_t count_pos_rule;
    std::int32_t group;
    double group_sums[5];  // logsum_1_p_pos, kl_div_neg, kl_div_pos,
                           // logsum_1_p_pos_rule, logsum_1_p_pos_calib
    double total[5];       // the float sums, in the order of group_sums
  };

  Slot& find_or_insert(std::int64_t label_id);
  void add_to(Slot& slot, const FileSite& site, std::int32_t group);
  void grow();
  static void close_group(Slot& slot);

  Slot& slot(std::size_t number) {
    return blocks_[number >> kBlockBits][number & (kBlockSlots - 1)];
  }

  // The slots lie in blocks of a fixed size: a growing table neither copies
  // them nor keeps spare room for as many again.
  static constexpr unsigned kBlockBits = 14;
  static constexpr std::size_t kBlockSlots = std::size_t{1} << kBlockBits;
  std::vector<std::unique_ptr<Slot[]>> blocks_;
  std::size_t size_ = 0;  // slots in use
  std::vector<std::uint32_t> index_;  // open addressing; 0 = empty, else slot + 1
  unsigned shift_;
};

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_SITE_TABLE_H_
