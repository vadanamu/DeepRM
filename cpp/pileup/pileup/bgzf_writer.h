/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_BGZF_WRITER_H_
#define DEEPRM_PILEUP_BGZF_WRITER_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

struct libdeflate_compressor;

namespace deeprm {

// Makes BGZF blocks in memory with libdeflate (the block format of BAM
// files). One per thread.
class BgzfWriter {
 public:
  explicit BgzfWriter(int level);
  ~BgzfWriter();
  BgzfWriter(const BgzfWriter&) = delete;
  BgzfWriter& operator=(const BgzfWriter&) = delete;

  // Ends the current block if `size` more bytes do not fit in it, so that a
  // record of that size is not split (as htslib's bgzf_flush_try).
  void keep_together(std::size_t size);
  void write(std::span<const std::uint8_t> bytes);
  // Ends the current block.
  void flush();

  // Where the next byte goes: the offset of its block among the blocks so
  // far, shifted left by 16, plus its offset in the block.
  std::uint64_t tell() const {
    return ((taken_ + blocks_.size()) << 16) | pending_.size();
  }

  // Bytes of the finished blocks that are still in the writer.
  std::size_t held() const { return blocks_.size(); }

  // The finished blocks; takes them out of the writer (tell() goes on
  // counting them).
  std::vector<std::uint8_t> take_blocks();
  // Makes tell() count from 0 again, for a writer whose blocks were taken.
  void restart() { taken_ = 0; }

 private:
  libdeflate_compressor* compressor_;
  std::vector<std::uint8_t> pending_;  // of the current block
  std::vector<std::uint8_t> blocks_;
  std::uint64_t taken_ = 0;  // bytes of the blocks taken out
};

// The empty block that ends a BGZF file.
std::span<const std::uint8_t> bgzf_end_block();

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_BGZF_WRITER_H_
