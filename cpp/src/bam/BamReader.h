/***************************************************************************************************
 *
 * Copyright (C) 2025-2026 Genome4me Incorporated - All Rights Reserved.
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
#include <vector>
#include <memory>
#include <htslib/hts.h>
#include <htslib/sam.h>

#include "bam/BamRecord.h"
#include "utils/Uuid128.h"

using namespace std;

namespace deeprm {
  class BamReader {
  private:
    string bam_path;
    // SAM flag bits to exclude at harvest time. Record extraction itself lives
    // in BamRecordExtractor, which phase 2 runs on the re-read record.
    int filter_flag;
    // Voffset range mode (as in the Python preprocessing): when set, harvest_offsets seeks
    // to v_lo and consumes records until the first whose START voffset is
    // >= v_hi (v_hi == 0 means "to EOF"), instead of the modulo scan.
    bool ranged;
    uint64_t v_lo;
    uint64_t v_hi;
    // Set when this reader's share could not be read in full (open, header, seek or a read error other
    // than end of file); the run then fails instead of missing part of the BAM.
    bool read_failed = false;

  public:
    BamReader(const string& path, int filter_flag);
    ~BamReader();

    // Lazy file-mode phase 1: walk this reader's share of the BAM and return
    // (parent_id, record-start voffset) pairs for records passing the cheap
    // pre-filter (empty SEQ, -g flag mask). No extraction happens here --
    // phase 2 re-reads each record via bgzf_seek and runs the shared
    // extractor, so the resident index shrinks from whole records (~14 KB
    // each) to ~170 B per parent.
    vector<pair<Uuid128, uint64_t>> harvest_offsets(int process_id,
                                                   int num_processes,
                                                   int num_threads);

    bool failed() const { return read_failed; }

    // Enable voffset range mode for this reader (the fast path).
    void set_voffset_range(uint64_t lo, uint64_t hi)
    {
      ranged = true;
      v_lo = lo;
      v_hi = hi;
    }
  };
} // namespace deeprm
