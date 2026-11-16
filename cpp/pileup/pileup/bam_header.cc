/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include "pileup/bam_header.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <optional>
#include <span>

#include "libdeflate/libdeflate.h"

namespace deeprm {
namespace {

constexpr std::size_t kBlockHeaderSize = 12;  // up to and including XLEN
constexpr std::size_t kBlockFooterSize = 8;   // CRC32 and ISIZE

std::uint32_t load_le(std::span<const std::uint8_t> bytes, std::size_t offset,
                      std::size_t width) {
  std::uint32_t value = 0;
  for (std::size_t i = 0; i < width; ++i) {
    value |= static_cast<std::uint32_t>(bytes[offset + i]) << (8 * i);
  }
  return value;
}

bool read_exact(std::ifstream& in, std::vector<std::uint8_t>& bytes,
                std::size_t size) {
  bytes.resize(size);
  in.read(reinterpret_cast<char*>(bytes.data()),
          static_cast<std::streamsize>(size));
  return static_cast<std::size_t>(in.gcount()) == size;
}

// The size of the BGZF block from the BC subfield of its extra field.
std::optional<std::size_t> block_size(std::span<const std::uint8_t> extra) {
  for (std::size_t at = 0; at + 4 <= extra.size();) {
    const std::size_t length = load_le(extra, at + 2, 2);
    if (extra[at] == 'B' && extra[at + 1] == 'C' && length == 2 &&
        at + 6 <= extra.size()) {
      return load_le(extra, at + 4, 2) + 1;
    }
    at += 4 + length;
  }
  return std::nullopt;
}

// Appends the data of the next BGZF block to `out`; false at the end of
// the file or on a block that is not BGZF.
bool read_block(std::ifstream& in, libdeflate_decompressor* decompressor,
                std::vector<std::uint8_t>& out) {
  std::vector<std::uint8_t> header;
  if (!read_exact(in, header, kBlockHeaderSize)) return false;
  if (header[0] != 31 || header[1] != 139 || (header[3] & 4) == 0) return false;
  std::vector<std::uint8_t> extra;
  if (!read_exact(in, extra, load_le(header, 10, 2))) return false;
  const std::optional<std::size_t> size = block_size(extra);
  const std::size_t before = kBlockHeaderSize + extra.size();
  if (!size || *size < before + kBlockFooterSize) return false;
  std::vector<std::uint8_t> rest;
  if (!read_exact(in, rest, *size - before)) return false;
  const std::size_t compressed = rest.size() - kBlockFooterSize;
  const std::size_t data_size = load_le(rest, compressed + 4, 4);
  const std::size_t old_size = out.size();
  out.resize(old_size + data_size);
  return libdeflate_deflate_decompress(decompressor, rest.data(), compressed,
                                       out.data() + old_size, data_size,
                                       nullptr) == LIBDEFLATE_SUCCESS;
}

// The names from the uncompressed start of a BAM file, or nothing when
// `data` does not hold the whole header yet.
std::optional<std::vector<std::string>> parse_names(
    std::span<const std::uint8_t> data) {
  if (data.size() < 12) return std::nullopt;
  std::size_t at = 8 + load_le(data, 4, 4);
  if (data.size() < at + 4) return std::nullopt;
  const std::size_t count = load_le(data, at, 4);
  at += 4;
  std::vector<std::string> names;
  names.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    if (data.size() < at + 4) return std::nullopt;
    const std::size_t length = load_le(data, at, 4);
    if (length == 0 || data.size() < at + 4 + length + 4) return std::nullopt;
    names.emplace_back(reinterpret_cast<const char*>(data.data()) + at + 4,
                       length - 1);  // without the terminating NUL
    at += 4 + length + 4;
  }
  return names;
}

}  // namespace

std::expected<std::vector<std::string>, std::string> read_reference_names(
    const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::unexpected("cannot open " + path);
  libdeflate_decompressor* decompressor = libdeflate_alloc_decompressor();
  std::vector<std::uint8_t> data;
  std::optional<std::vector<std::string>> names;
  while (!names && read_block(in, decompressor, data)) {
    if (data.size() >= 4 && (data[0] != 'B' || data[1] != 'A' ||
                             data[2] != 'M' || data[3] != 1)) {
      break;
    }
    names = parse_names(data);
  }
  libdeflate_free_decompressor(decompressor);
  if (!names) return std::unexpected("cannot read the BAM header of " + path);
  return std::move(*names);
}

}  // namespace deeprm
