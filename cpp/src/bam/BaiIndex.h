/***************************************************************************************************
 *
 * Copyright (C) 2026 Genome4me Incorporated - All Rights Reserved.
 *
 * This software, including its source code, embedded concepts, and associated
 * documentation, is proprietary to Genome4me Incorporated and is protected
 * under trade secret and copyright law. Unauthorized use, copying, modification,
 * distribution, or disclosure to third parties, in whole or in part, is
 * strictly prohibited unless prior written permission is granted by Genome4me
 * Incorporated. Any such unauthorized actions constitute an infringement of
 * the intellectual property rights of Genome4me Incorporated. For licensing
 * inquiries or permissions, please contact Genome4me Incorporated.
 *
 **************************************************************************************************/

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace deeprm {
  // BGZF virtual-offset partitioning of a BAM from its .bai linear index,
  // as the Python preprocessing does (_fast_path_ok / _bai_linear_offsets /
  // _plan_voffset_ranges): file mode splits the BAM into contiguous
  // voffset ranges so each phase-1 worker decompresses only its slice,
  // instead of every worker scanning the whole file modulo-style.
  class BaiIndex {
  public:
    // Whether the voffset fast path is safe. Three conditions, all required
    // (as in the Python preprocessing): the .bai exists, it is NOT OLDER than the BAM (the mtime check is
    // what actually catches a stale index), and the header says
    // SO:coordinate. Returns (ok, reason).
    static pair<bool, string> fast_path_ok(const string& bam_path);

    // Sorted, deduped, nonzero linear-index virtual offsets from the .bai.
    // These are guaranteed record boundaries, emitted per 16 kb of reference
    // coordinate. Empty on any anomaly (missing, truncated, trailing bytes):
    // a wrong partition is far worse than falling back to the modulo scan.
    static vector<uint64_t> linear_offsets(const string& bam_path);

    // Partition split points into up to n_parts contiguous [lo, hi) voffset
    // ranges balanced by COMPRESSED BYTES (voffset >> 16). The final range's
    // hi is 0, meaning "to EOF". Empty when there are too few points.
    static vector<pair<uint64_t, uint64_t>> plan_ranges(
        const vector<uint64_t>& offsets, int n_parts);
  };
} // namespace deeprm
