/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_NPZ_READER_H_
#define DEEPRM_PILEUP_NPZ_READER_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

struct libdeflate_decompressor;

namespace deeprm {

// One array of an npz file: its NumPy dtype string, shape and raw bytes.
struct NpyArray {
  std::string name;   // without the ".npy" suffix
  std::string descr;  // e.g. "<f4", "<i8"
  std::vector<std::size_t> shape;
  std::vector<std::uint8_t> data;
};

// Reads the arrays of npz files (zip of .npy members, stored or deflated).
// Not thread-safe; use one reader per thread.
class NpzReader {
 public:
  NpzReader();
  ~NpzReader();
  NpzReader(const NpzReader&) = delete;
  NpzReader& operator=(const NpzReader&) = delete;

  // All arrays of the npz held in `file`, in archive order.
  std::expected<std::vector<NpyArray>, std::string> read_all(
      std::span<const std::uint8_t> file);

  // The array `name` of the npz held in `file`.
  std::expected<NpyArray, std::string> read(std::span<const std::uint8_t> file,
                                            std::string_view name);

 private:
  libdeflate_decompressor* decompressor_;
};

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_NPZ_READER_H_
