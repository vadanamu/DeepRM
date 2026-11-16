/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_NPZ_WRITER_H_
#define DEEPRM_PILEUP_NPZ_WRITER_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "pileup/parallel.h"

struct libdeflate_compressor;
struct libdeflate_decompressor;

namespace deeprm {

// A run of an array's bytes as a raw deflate stream that ends at a byte
// boundary without a final block, so that the next run's stream can follow.
struct DeflatedRun {
  std::vector<std::uint8_t> bytes;
  std::uint32_t crc32 = 0;
  std::uint64_t size = 0;  // before compression
};

// Makes runs with libdeflate. libdeflate only makes whole streams, so the run
// is compressed in blocks (as BGZF does) and each block's stream is turned into
// an open one: its final-block flag is cleared and an empty stored block
// brings it to a byte boundary. Not thread-safe; use one per thread.
class RunDeflater {
 public:
  RunDeflater();
  ~RunDeflater();
  RunDeflater(const RunDeflater&) = delete;
  RunDeflater& operator=(const RunDeflater&) = delete;

  DeflatedRun deflate(std::span<const std::uint8_t> raw);

 private:
  void append(std::span<const std::uint8_t> raw,
              std::vector<std::uint8_t>& out);
  bool append_compressed(std::span<const std::uint8_t> raw,
                         std::vector<std::uint8_t>& out);
  bool inflates(std::size_t stream_size, std::size_t raw_size);

  libdeflate_compressor* compressor_;
  libdeflate_decompressor* decompressor_;
  std::vector<std::uint8_t> stream_;  // one block's stream, being closed
  std::vector<std::uint8_t> check_;   // what it inflates to
};

// The CRC-32 of two byte strings joined, from the CRC-32 of each.
std::uint32_t crc32_combine(std::uint32_t first, std::uint32_t second,
                            std::uint64_t second_size);

// One 1-D array of an npz file, given as consecutive runs.
struct NpzArray {
  std::string name;
  std::string descr;  // NumPy dtype string, e.g. "<f8"
  std::size_t rows = 0;
  std::vector<DeflatedRun> runs;
};

// Writes the arrays as an npz file (zip64, one deflated .npy per array).
Status write_npz(const std::string& path, const std::vector<NpzArray>& arrays,
                 std::size_t threads);

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_NPZ_WRITER_H_
