/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include "pileup/npz_writer.h"

#include <algorithm>
#include <cstring>

#include "libdeflate/libdeflate.h"

#include "pileup/output_file.h"

namespace deeprm {
namespace {

constexpr int kDeflateLevel = 1;
constexpr std::uint16_t kZipVersion = 45;  // zip64
constexpr std::uint16_t kMethodDeflate = 8;
constexpr std::uint16_t kDosDate = 0x0021;  // 1980-01-01
constexpr std::uint32_t kUse64 = 0xFFFFFFFF;
constexpr std::size_t kNpyAlignment = 64;
// An empty final block: ends the deflate stream of a member.
constexpr std::uint8_t kStreamEnd[] = {0x03, 0x00};
// An empty stored block, for a stream that ends 3 to 7 bits before a byte
// boundary: the block's 3 header bits are the zero bits that pad the stream.
constexpr std::uint8_t kJoinInPadding[] = {0x00, 0x00, 0xFF, 0xFF};
// The same for a stream that ends 0 to 2 bits before a byte boundary: the
// header bits run into one more zero byte.
constexpr std::uint8_t kJoinPastPadding[] = {0x00, 0x00, 0x00, 0xFF, 0xFF};
// Bytes compressed as one stream. Small enough that libdeflate makes a single
// deflate block of it at level 1 (checked per block, see append_compressed).
constexpr std::size_t kBlockSize = 32768;
// A block that libdeflate split is halved down to this size, then stored.
constexpr std::size_t kSmallestSplit = 4096;
constexpr std::uint32_t kCrcPolynomial = 0xEDB88320;

using Bytes = std::vector<std::uint8_t>;

void put(Bytes& out, std::uint64_t value, std::size_t width) {
  for (std::size_t i = 0; i < width; ++i) {
    out.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
  }
}

void put(Bytes& out, const std::string& text) {
  out.insert(out.end(), text.begin(), text.end());
}

// The .npy header (format 1.0) of a 1-D array.
Bytes npy_header(const std::string& descr, std::size_t rows) {
  std::string dict = "{'descr': '" + descr +
                     "', 'fortran_order': False, 'shape': (" +
                     std::to_string(rows) + ",), }";
  const std::size_t fixed = 10;  // magic, version and the header length
  const std::size_t padding =
      kNpyAlignment - (fixed + dict.size() + 1) % kNpyAlignment;
  dict.append(padding % kNpyAlignment, ' ');
  dict.push_back('\n');
  Bytes out = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  put(out, dict.size(), 2);
  put(out, dict);
  return out;
}

// Where one member lies in the file, and what its directory entry holds.
struct Member {
  Bytes local_header;
  DeflatedRun npy_header;
  std::uint32_t crc32 = 0;
  std::uint64_t size = 0;
  std::uint64_t compressed_size = 0;
  std::uint64_t offset = 0;
};

Bytes local_header(const std::string& name, const Member& member) {
  Bytes out;
  put(out, 0x04034b50, 4);
  put(out, kZipVersion, 2);
  put(out, 0, 2);
  put(out, kMethodDeflate, 2);
  put(out, 0, 2);
  put(out, kDosDate, 2);
  put(out, member.crc32, 4);
  put(out, kUse64, 4);
  put(out, kUse64, 4);
  put(out, name.size(), 2);
  put(out, 20, 2);
  put(out, name);
  put(out, 0x0001, 2);
  put(out, 16, 2);
  put(out, member.size, 8);
  put(out, member.compressed_size, 8);
  return out;
}

void put_directory_entry(Bytes& out, const std::string& name,
                         const Member& member) {
  put(out, 0x02014b50, 4);
  put(out, (3 << 8) | kZipVersion, 2);
  put(out, kZipVersion, 2);
  put(out, 0, 2);
  put(out, kMethodDeflate, 2);
  put(out, 0, 2);
  put(out, kDosDate, 2);
  put(out, member.crc32, 4);
  put(out, kUse64, 4);
  put(out, kUse64, 4);
  put(out, name.size(), 2);
  put(out, 28, 2);
  put(out, 0, 2);
  put(out, 0, 2);
  put(out, 0, 2);
  put(out, std::uint64_t{0600} << 16, 4);
  put(out, kUse64, 4);
  put(out, name);
  put(out, 0x0001, 2);
  put(out, 24, 2);
  put(out, member.size, 8);
  put(out, member.compressed_size, 8);
  put(out, member.offset, 8);
}

// The central directory with its zip64 and classic end records.
Bytes directory(const std::vector<NpzArray>& arrays,
                const std::vector<Member>& members, std::uint64_t offset) {
  Bytes out;
  for (std::size_t i = 0; i < arrays.size(); ++i) {
    put_directory_entry(out, arrays[i].name + ".npy", members[i]);
  }
  const std::uint64_t size = out.size();
  put(out, 0x06064b50, 4);
  put(out, 44, 8);
  put(out, kZipVersion, 2);
  put(out, kZipVersion, 2);
  put(out, 0, 4);
  put(out, 0, 4);
  put(out, arrays.size(), 8);
  put(out, arrays.size(), 8);
  put(out, size, 8);
  put(out, offset, 8);
  put(out, 0x07064b50, 4);
  put(out, 0, 4);
  put(out, offset + size, 8);
  put(out, 1, 4);
  put(out, 0x06054b50, 4);
  put(out, 0, 2);
  put(out, 0, 2);
  put(out, std::min<std::uint64_t>(arrays.size(), 0xFFFF), 2);
  put(out, std::min<std::uint64_t>(arrays.size(), 0xFFFF), 2);
  put(out, kUse64, 4);
  put(out, kUse64, 4);
  put(out, 0, 2);
  return out;
}

Member plan_member(RunDeflater& deflater, const NpzArray& array,
                   std::uint64_t offset) {
  Member member;
  member.npy_header = deflater.deflate(npy_header(array.descr, array.rows));
  member.crc32 = member.npy_header.crc32;
  member.size = member.npy_header.size;
  member.compressed_size = member.npy_header.bytes.size() + sizeof(kStreamEnd);
  for (const DeflatedRun& run : array.runs) {
    member.crc32 = crc32_combine(member.crc32, run.crc32, run.size);
    member.size += run.size;
    member.compressed_size += run.bytes.size();
  }
  member.offset = offset;
  member.local_header = local_header(array.name + ".npy", member);
  return member;
}

// A piece of the file: where it goes and its bytes.
struct Piece {
  std::uint64_t offset;
  std::span<const std::uint8_t> bytes;
};

void add_pieces(const NpzArray& array, const Member& member,
                std::vector<Piece>& pieces) {
  std::uint64_t at = member.offset;
  const auto add = [&](std::span<const std::uint8_t> bytes) {
    pieces.push_back(Piece{at, bytes});
    at += bytes.size();
  };
  add(member.local_header);
  add(member.npy_header.bytes);
  for (const DeflatedRun& run : array.runs) add(run.bytes);
  add(kStreamEnd);
}

}  // namespace

RunDeflater::RunDeflater()
    : compressor_(libdeflate_alloc_compressor(kDeflateLevel)),
      decompressor_(libdeflate_alloc_decompressor()),
      stream_(libdeflate_deflate_compress_bound(compressor_, kBlockSize) +
              sizeof(kJoinPastPadding) + sizeof(kStreamEnd)),
      check_(kBlockSize) {}

RunDeflater::~RunDeflater() {
  libdeflate_free_compressor(compressor_);
  libdeflate_free_decompressor(decompressor_);
}

DeflatedRun RunDeflater::deflate(std::span<const std::uint8_t> raw) {
  DeflatedRun run;
  run.size = raw.size();
  run.crc32 = libdeflate_crc32(0, raw.data(), raw.size());
  for (std::size_t at = 0; at < raw.size(); at += kBlockSize) {
    append(raw.subspan(at, std::min(kBlockSize, raw.size() - at)), run.bytes);
  }
  run.bytes.shrink_to_fit();
  return run;
}

void RunDeflater::append(std::span<const std::uint8_t> raw, Bytes& out) {
  if (append_compressed(raw, out)) return;
  if (raw.size() > kSmallestSplit) {
    append(raw.first(raw.size() / 2), out);
    append(raw.subspan(raw.size() / 2), out);
    return;
  }
  out.push_back(0);  // a stored block that is not the final one
  put(out, raw.size(), 2);
  put(out, ~raw.size() & 0xFFFF, 2);
  out.insert(out.end(), raw.begin(), raw.end());
}

// Appends the block as an open stream. False when libdeflate made more than
// one deflate block of it: the final-block flag to clear is then out of reach.
bool RunDeflater::append_compressed(std::span<const std::uint8_t> raw,
                                    Bytes& out) {
  const std::size_t size = libdeflate_deflate_compress(
      compressor_, raw.data(), raw.size(), stream_.data(),
      stream_.size() - sizeof(kJoinPastPadding) - sizeof(kStreamEnd));
  // the flag is the stream's first bit; set there, the first block is the last
  if (size == 0 || (stream_[0] & 1) == 0) return false;
  stream_[0] &= 0xFE;
  // Where the stream ends inside its last byte is not known, so try both
  // joins: only the right one inflates.
  for (const std::span<const std::uint8_t> join :
       {std::span<const std::uint8_t>(kJoinInPadding),
        std::span<const std::uint8_t>(kJoinPastPadding)}) {
    std::copy(join.begin(), join.end(), stream_.begin() + size);
    if (inflates(size + join.size(), raw.size())) {
      out.insert(out.end(), stream_.begin(),
                 stream_.begin() + size + join.size());
      return true;
    }
  }
  return false;
}

// True if the open stream in `stream_`, once ended, inflates to `raw_size`
// bytes.
bool RunDeflater::inflates(std::size_t stream_size, std::size_t raw_size) {
  std::copy(std::begin(kStreamEnd), std::end(kStreamEnd),
            stream_.begin() + stream_size);
  return libdeflate_deflate_decompress(
             decompressor_, stream_.data(), stream_size + sizeof(kStreamEnd),
             check_.data(), raw_size, nullptr) == LIBDEFLATE_SUCCESS;
}

namespace {

// Polynomials over GF(2) modulo the CRC polynomial, bit-reversed: bit 31 is
// the coefficient of x^0.
std::uint32_t multiply_polynomials(std::uint32_t a, std::uint32_t b) {
  std::uint32_t product = 0;
  for (std::uint32_t bit = 0x80000000; bit != 0; bit >>= 1) {
    if (a & bit) product ^= b;
    b = (b & 1) ? (b >> 1) ^ kCrcPolynomial : b >> 1;
  }
  return product;
}

}  // namespace

std::uint32_t crc32_combine(std::uint32_t first, std::uint32_t second,
                            std::uint64_t second_size) {
  // x^(8 * second_size), by repeated squaring of x^8
  std::uint32_t shift = 0x80000000;
  std::uint32_t power = 0x00800000;
  for (std::uint64_t n = second_size; n != 0; n >>= 1) {
    if (n & 1) shift = multiply_polynomials(shift, power);
    power = multiply_polynomials(power, power);
  }
  return multiply_polynomials(shift, first) ^ second;
}

Status write_npz(const std::string& path, const std::vector<NpzArray>& arrays,
                 std::size_t threads) {
  RunDeflater deflater;
  std::vector<Member> members;
  std::uint64_t offset = 0;
  for (const NpzArray& array : arrays) {
    members.push_back(plan_member(deflater, array, offset));
    offset += members.back().local_header.size() +
              members.back().compressed_size;
  }
  const Bytes end = directory(arrays, members, offset);
  std::vector<Piece> pieces;
  for (std::size_t i = 0; i < arrays.size(); ++i) {
    add_pieces(arrays[i], members[i], pieces);
  }
  pieces.push_back(Piece{offset, end});
  OutputFile file;
  if (Status status = file.open(path); !status) return status;
  const Status written =
      parallel_for(threads, pieces.size(), [&](std::size_t, std::size_t i) {
        return file.write_at(pieces[i].offset, pieces[i].bytes);
      });
  if (!written) return written;
  return file.close();
}

}  // namespace deeprm
