/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include "pileup/npz_reader.h"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <optional>

#include "libdeflate/libdeflate.h"

namespace deeprm {
namespace {

using Bytes = std::span<const std::uint8_t>;

constexpr std::uint32_t kEndOfDirectorySignature = 0x06054b50;
constexpr std::uint32_t kDirectoryEntrySignature = 0x02014b50;
constexpr std::uint32_t kLocalHeaderSignature = 0x04034b50;
constexpr std::size_t kEndOfDirectorySize = 22;
constexpr std::size_t kDirectoryEntrySize = 46;
constexpr std::size_t kLocalHeaderSize = 30;
constexpr std::uint16_t kMethodStored = 0;
constexpr std::uint16_t kMethodDeflate = 8;
constexpr std::string_view kNpySuffix = ".npy";
constexpr std::string_view kNpyMagic = "\x93NUMPY";

// One member of the zip, from its central directory entry.
struct ZipMember {
  std::string name;
  std::uint16_t method;
  std::uint32_t crc32;
  std::size_t compressed_size;
  std::size_t size;
  std::size_t local_header_offset;
};

std::uint32_t load_le(Bytes bytes, std::size_t offset, std::size_t width) {
  std::uint32_t value = 0;
  for (std::size_t i = 0; i < width; ++i) {
    value |= static_cast<std::uint32_t>(bytes[offset + i]) << (8 * i);
  }
  return value;
}

std::optional<std::size_t> find_end_of_directory(Bytes file) {
  if (file.size() < kEndOfDirectorySize) return std::nullopt;
  // The record is followed only by the archive comment (at most 65,535 bytes).
  const std::size_t last = file.size() - kEndOfDirectorySize;
  const std::size_t first = last > 0xFFFF ? last - 0xFFFF : 0;
  for (std::size_t offset = last + 1; offset-- > first;) {
    if (load_le(file, offset, 4) == kEndOfDirectorySignature) return offset;
  }
  return std::nullopt;
}

std::expected<std::vector<ZipMember>, std::string> read_directory(Bytes file) {
  const std::optional<std::size_t> end = find_end_of_directory(file);
  if (!end) return std::unexpected("not a zip file");
  const std::size_t count = load_le(file, *end + 10, 2);
  std::size_t offset = load_le(file, *end + 16, 4);
  if (count == 0xFFFF || offset == 0xFFFFFFFF) {
    return std::unexpected("zip64 directory is not supported");
  }
  std::vector<ZipMember> members;
  members.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    if (offset + kDirectoryEntrySize > file.size() ||
        load_le(file, offset, 4) != kDirectoryEntrySignature) {
      return std::unexpected("bad zip directory entry");
    }
    const std::size_t name_size = load_le(file, offset + 28, 2);
    const std::size_t skip =
        load_le(file, offset + 30, 2) + load_le(file, offset + 32, 2);
    if (offset + kDirectoryEntrySize + name_size + skip > file.size()) {
      return std::unexpected("truncated zip directory");
    }
    ZipMember member{
        .name = std::string(reinterpret_cast<const char*>(file.data()) +
                                offset + kDirectoryEntrySize,
                            name_size),
        .method = static_cast<std::uint16_t>(load_le(file, offset + 10, 2)),
        .crc32 = load_le(file, offset + 16, 4),
        .compressed_size = load_le(file, offset + 20, 4),
        .size = load_le(file, offset + 24, 4),
        .local_header_offset = load_le(file, offset + 42, 4),
    };
    if (member.compressed_size == 0xFFFFFFFF || member.size == 0xFFFFFFFF ||
        member.local_header_offset == 0xFFFFFFFF) {
      return std::unexpected("zip64 member is not supported: " + member.name);
    }
    members.push_back(std::move(member));
    offset += kDirectoryEntrySize + name_size + skip;
  }
  return members;
}

// The member's bytes as stored in the archive (still compressed).
std::expected<Bytes, std::string> stored_bytes(Bytes file,
                                               const ZipMember& member) {
  const std::size_t offset = member.local_header_offset;
  if (offset + kLocalHeaderSize > file.size() ||
      load_le(file, offset, 4) != kLocalHeaderSignature) {
    return std::unexpected("bad zip local header: " + member.name);
  }
  // The local extra field can differ from the directory's (NumPy adds zip64).
  const std::size_t begin = offset + kLocalHeaderSize +
                            load_le(file, offset + 26, 2) +
                            load_le(file, offset + 28, 2);
  if (begin + member.compressed_size > file.size()) {
    return std::unexpected("truncated zip member: " + member.name);
  }
  return file.subspan(begin, member.compressed_size);
}

std::expected<std::vector<std::uint8_t>, std::string> extract(
    libdeflate_decompressor* decompressor, Bytes file,
    const ZipMember& member) {
  const std::expected<Bytes, std::string> stored = stored_bytes(file, member);
  if (!stored) return std::unexpected(stored.error());
  std::vector<std::uint8_t> out(member.size);
  if (member.method == kMethodStored) {
    if (stored->size() != out.size()) {
      return std::unexpected("bad stored size: " + member.name);
    }
    std::copy(stored->begin(), stored->end(), out.begin());
  } else if (member.method == kMethodDeflate) {
    const libdeflate_result result = libdeflate_deflate_decompress(
        decompressor, stored->data(), stored->size(), out.data(), out.size(),
        nullptr);
    if (result != LIBDEFLATE_SUCCESS) {
      return std::unexpected("inflate failed: " + member.name);
    }
  } else {
    return std::unexpected("unsupported zip method: " + member.name);
  }
  if (libdeflate_crc32(0, out.data(), out.size()) != member.crc32) {
    return std::unexpected("CRC mismatch: " + member.name);
  }
  return out;
}

// The text between `open` and `close` after `key` in the .npy header dict.
std::optional<std::string_view> header_value(std::string_view header,
                                             std::string_view key, char open,
                                             char close) {
  const std::size_t at = header.find(key);
  if (at == std::string_view::npos) return std::nullopt;
  const std::size_t begin = header.find(open, at + key.size());
  if (begin == std::string_view::npos) return std::nullopt;
  const std::size_t end = header.find(close, begin + 1);
  if (end == std::string_view::npos) return std::nullopt;
  return header.substr(begin + 1, end - begin - 1);
}

std::optional<std::vector<std::size_t>> parse_shape(std::string_view text) {
  std::vector<std::size_t> shape;
  while (!text.empty()) {
    const std::size_t comma = std::min(text.find(','), text.size());
    std::string_view item = text.substr(0, comma);
    text.remove_prefix(std::min(comma + 1, text.size()));
    while (!item.empty() && item.front() == ' ') item.remove_prefix(1);
    while (!item.empty() && item.back() == ' ') item.remove_suffix(1);
    if (item.empty()) continue;
    std::size_t dim = 0;
    const std::from_chars_result parsed =
        std::from_chars(item.data(), item.data() + item.size(), dim);
    if (parsed.ec != std::errc() || parsed.ptr != item.data() + item.size()) {
      return std::nullopt;
    }
    shape.push_back(dim);
  }
  return shape;
}

// Fills descr and shape from the .npy header and drops it from `array.data`.
std::expected<void, std::string> strip_npy_header(NpyArray& array) {
  const std::vector<std::uint8_t>& data = array.data;
  const std::string bad = "bad npy header: " + array.name;
  if (data.size() < 10 ||
      std::memcmp(data.data(), kNpyMagic.data(), kNpyMagic.size()) != 0) {
    return std::unexpected(bad);
  }
  const std::size_t length_width = data[6] == 1 ? 2 : 4;
  const std::size_t header_begin = 8 + length_width;
  if (data.size() < header_begin) return std::unexpected(bad);
  const std::size_t data_begin =
      header_begin + load_le(data, 8, length_width);
  if (data_begin > data.size()) return std::unexpected(bad);
  const std::string_view header(
      reinterpret_cast<const char*>(data.data()) + header_begin,
      data_begin - header_begin);
  const std::optional<std::string_view> descr =
      header_value(header, "'descr':", '\'', '\'');
  const std::optional<std::string_view> order =
      header_value(header, "'fortran_order':", ' ', ',');
  const std::optional<std::string_view> shape_text =
      header_value(header, "'shape':", '(', ')');
  if (!descr || !order || !shape_text) return std::unexpected(bad);
  if (*order != "False") {
    return std::unexpected("Fortran order is not supported: " + array.name);
  }
  std::optional<std::vector<std::size_t>> shape = parse_shape(*shape_text);
  if (!shape) return std::unexpected(bad);
  array.descr = std::string(*descr);
  array.shape = std::move(*shape);
  array.data.erase(array.data.begin(),
                   array.data.begin() + static_cast<std::ptrdiff_t>(data_begin));
  return {};
}

std::string array_name(std::string_view member_name) {
  if (member_name.ends_with(kNpySuffix)) {
    member_name.remove_suffix(kNpySuffix.size());
  }
  return std::string(member_name);
}

std::expected<NpyArray, std::string> read_member(
    libdeflate_decompressor* decompressor, Bytes file,
    const ZipMember& member) {
  std::expected<std::vector<std::uint8_t>, std::string> data =
      extract(decompressor, file, member);
  if (!data) return std::unexpected(data.error());
  NpyArray array{.name = array_name(member.name),
                 .descr = {},
                 .shape = {},
                 .data = std::move(*data)};
  const std::expected<void, std::string> stripped = strip_npy_header(array);
  if (!stripped) return std::unexpected(stripped.error());
  return array;
}

}  // namespace

NpzReader::NpzReader() : decompressor_(libdeflate_alloc_decompressor()) {}

NpzReader::~NpzReader() { libdeflate_free_decompressor(decompressor_); }

std::expected<std::vector<NpyArray>, std::string> NpzReader::read_all(
    std::span<const std::uint8_t> file) {
  const std::expected<std::vector<ZipMember>, std::string> members =
      read_directory(file);
  if (!members) return std::unexpected(members.error());
  std::vector<NpyArray> arrays;
  arrays.reserve(members->size());
  for (const ZipMember& member : *members) {
    std::expected<NpyArray, std::string> array =
        read_member(decompressor_, file, member);
    if (!array) return std::unexpected(array.error());
    arrays.push_back(std::move(*array));
  }
  return arrays;
}

std::expected<NpyArray, std::string> NpzReader::read(
    std::span<const std::uint8_t> file, std::string_view name) {
  const std::expected<std::vector<ZipMember>, std::string> members =
      read_directory(file);
  if (!members) return std::unexpected(members.error());
  for (const ZipMember& member : *members) {
    if (array_name(member.name) == name) {
      return read_member(decompressor_, file, member);
    }
  }
  return std::unexpected("no array named " + std::string(name));
}

}  // namespace deeprm
