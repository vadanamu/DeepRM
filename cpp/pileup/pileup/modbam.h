/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_MODBAM_H_
#define DEEPRM_PILEUP_MODBAM_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "pileup/read_rows.h"

struct bam1_t;  // htslib's alignment record

namespace deeprm {

// What to write: the input BAM (sorted by coordinate; with an index it is
// read by all threads, without one by a single thread), the modBAM, the
// folder for temporary files and the number of threads.
struct ModbamJob {
  std::string input;
  std::string output;
  std::string tmp_root;
  std::size_t threads;
};

// Writes `output` and its .bai: every record of `input` whose read has rows
// on the record's reference, in the order of `input`, with MM/ML tags from
// those rows. False when no record was written (no file is left then).
std::expected<bool, std::string> write_modbam(const ModbamJob& job,
                                              const ReadRows& rows);

// The header text with its lines in the order the Python pileup writes them:
// @HD, then every @SQ, @RG, @PG and @CO line; other lines are dropped.
std::string merge_order_header(std::string_view text);

// The IDs of the header's lines of `kind` ("@RG" or "@PG").
std::vector<std::string> header_ids(std::string_view text,
                                    std::string_view kind);

// The record as the bytes of a BAM file, as htslib's bam_write1 writes them.
// False for a record the BAM format cannot hold.
bool encode_bam_record(const bam1_t* record, std::vector<std::uint8_t>& out);

// A part of the input that one thread reads and writes: the references
// first..last whole, or (first == last) the reads of one reference that
// start in [begin, end).
struct ModbamTask {
  std::int32_t first;
  std::int32_t last;
  std::int64_t begin;
  std::int64_t end;
  // A window that holds several tasks' worth of reads and cannot be cut.
  bool dense = false;
};

// How much of the input the reads that overlap [begin, end) of a reference
// take (in any unit, such as compressed bytes from the index).
using ModbamWeight = std::function<std::uint64_t(
    std::int32_t reference, std::int64_t begin, std::int64_t end)>;

// Tasks of about `target` reads each, in the order of the input, from the
// reads and the length of every reference (0 reads: the reference is left
// out). A reference with more reads is cut into windows that `weight` gives
// about the same share of the reference: reads lie very unevenly along a
// reference, and one large task would be written long after the others.
std::vector<ModbamTask> plan_modbam_tasks(
    const std::vector<std::uint64_t>& reads,
    const std::vector<std::int64_t>& lengths, std::uint64_t target,
    const ModbamWeight& weight);

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_MODBAM_H_
