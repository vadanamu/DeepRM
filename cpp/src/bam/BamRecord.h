/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of DeepRM. Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
 * International License; see LICENSE.md at the top of the DeepRM repository.
 */

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
