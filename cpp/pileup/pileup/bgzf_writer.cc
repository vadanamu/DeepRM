/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include "pileup/bgzf_writer.h"

#include <algorithm>

#include "libdeflate/libdeflate.h"

namespace deeprm {
namespace {

// Bytes of data per block, as htslib writes them.
constexpr std::size_t kBlockData = 0xff00;
constexpr std::size_t kMaxBlock = 0x10000;
constexpr std::size_t kHeaderSize = 18;
constexpr std::size_t kFooterSize = 8;
// A gzip member with the BGZF extra field "BC"; the block size follows.
constexpr std::uint8_t kHeader[16] = {0x1f, 0x8b, 8, 4, 0,   0,   0, 0,
                                      0,    0xff, 6, 0, 'B', 'C', 2, 0};
constexpr std::uint8_t kEndBlock[28] = {
    0x1f, 0x8b, 0x08, 0x04, 0, 0, 0, 0, 0, 0xff, 0x06, 0, 0x42, 0x43,
    0x02, 0,    0x1b, 0,    3, 0, 0, 0, 0, 0, 0,    0, 0,    0};

void put_le(std::uint8_t* out, std::uint32_t value, std::size_t width) {
  for (std::size_t i = 0; i < width; ++i) {
    out[i] = static_cast<std::uint8_t>(value >> (8 * i));
  }
}

}  // namespace

BgzfWriter::BgzfWriter(int level)
    : compressor_(libdeflate_alloc_compressor(level)) {
  pending_.reserve(kBlockData);
}

BgzfWriter::~BgzfWriter() { libdeflate_free_compressor(compressor_); }

void BgzfWriter::keep_together(std::size_t size) {
  if (pending_.size() + size > kBlockData) flush();
}

void BgzfWriter::write(std::span<const std::uint8_t> bytes) {
  while (!bytes.empty()) {
    const std::size_t size =
        std::min(bytes.size(), kBlockData - pending_.size());
    pending_.insert(pending_.end(), bytes.begin(), bytes.begin() + size);
    bytes = bytes.subspan(size);
    if (pending_.size() == kBlockData) flush();
  }
}

void BgzfWriter::flush() {
  if (pending_.empty()) return;
  const std::size_t at = blocks_.size();
  blocks_.resize(at + kMaxBlock);
  std::uint8_t* block = blocks_.data() + at;
  // kBlockData bytes always fit: libdeflate adds a few bytes at most
  const std::size_t deflated = libdeflate_deflate_compress(
      compressor_, pending_.data(), pending_.size(), block + kHeaderSize,
      kMaxBlock - kHeaderSize - kFooterSize);
  const std::size_t size = kHeaderSize + deflated + kFooterSize;
  std::copy(std::begin(kHeader), std::end(kHeader), block);
  put_le(block + 16, static_cast<std::uint32_t>(size - 1), 2);
  put_le(block + kHeaderSize + deflated,
         libdeflate_crc32(0, pending_.data(), pending_.size()), 4);
  put_le(block + kHeaderSize + deflated + 4,
         static_cast<std::uint32_t>(pending_.size()), 4);
  blocks_.resize(at + size);
  pending_.clear();
}

std::vector<std::uint8_t> BgzfWriter::take_blocks() {
  std::vector<std::uint8_t> blocks = std::move(blocks_);
  blocks_ = std::vector<std::uint8_t>();
  taken_ += blocks.size();
  return blocks;
}

std::span<const std::uint8_t> bgzf_end_block() { return kEndBlock; }

}  // namespace deeprm
