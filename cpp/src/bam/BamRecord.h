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
  // Fields ordered by alignment (8-byte members first) so the compiler
  // inserts no interior padding; only the tail after `strand` pads.
  struct BamRecord {
    vector<uint8_t> seq;
    vector<uint8_t> bq;
    vector<pair<int32_t, int32_t>> ap;
    vector<bool> mv;
    // Read UUID as two int64 halves, parsed ONCE at extraction. The writer
    // stores them into the npz read_id column as-is; keeping the 36-char
    // string here cost ~80 B/record and one UUID parse per SITE (~300x per
    // read) in commit_row.
    int64_t rid0;
    int64_t rid1;
    int32_t ts;
    int32_t ns;
    int32_t sp;
    int32_t ref;
    // Quality-trim bounds [trim_left, trim_right) over the query, as in the
    // Python preprocessing: (0, seq_len) when trimming is off or failed.
    int32_t trim_left;
    int32_t trim_right;
    int8_t strand; // -1 for reversed, 1 for not reversed
  };
} // namespace deeprm
