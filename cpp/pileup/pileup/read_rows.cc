/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include "pileup/read_rows.h"

#include <algorithm>
#include <cmath>

#include "pileup/parallel.h"

namespace deeprm {
namespace {

// Small numbers of either sign as small unsigned numbers.
std::uint64_t zigzag(std::int64_t value) {
  return (static_cast<std::uint64_t>(value) << 1) ^
         static_cast<std::uint64_t>(value >> 63);
}

std::int64_t unzigzag(std::uint64_t value) {
  return static_cast<std::int64_t>(value >> 1) ^
         -static_cast<std::int64_t>(value & 1);
}

// 7 bits per byte, the lowest first; the top bit tells that more follow.
void put_varint(std::vector<std::uint8_t>& out, std::uint64_t value) {
  while (value >= 0x80) {
    out.push_back(static_cast<std::uint8_t>(value) | 0x80);
    value >>= 7;
  }
  out.push_back(static_cast<std::uint8_t>(value));
}

std::uint64_t get_varint(const std::vector<std::uint8_t>& bytes,
                         std::size_t& at) {
  std::uint64_t value = 0;
  for (unsigned shift = 0;; shift += 7) {
    const std::uint8_t byte = bytes[at++];
    value |= static_cast<std::uint64_t>(byte & 0x7F) << shift;
    if ((byte & 0x80) == 0) return value;
  }
}

}  // namespace

std::uint8_t probability_byte(float prediction) {
  // np.clip(np.rint(pred * 255), 0, 255).astype(np.uint8), pred float32
  const float scaled = std::nearbyint(prediction * 255.0F);
  return static_cast<std::uint8_t>(std::clamp(scaled, 0.0F, 255.0F));
}

void PackedRows::add(const ReadKey& key, const ReadRow& row) {
  if (runs_.empty() || runs_.back().key != key) {
    runs_.push_back(Run{.key = key,
                        .offset = static_cast<std::uint32_t>(bytes_.size()),
                        .rows = 0});
    last_position_ = 0;
  }
  ++runs_.back().rows;
  put_varint(bytes_, zigzag(row.position - last_position_));
  bytes_.push_back(row.probability);
  last_position_ = row.position;
}

void PackedRows::finish() {
  runs_.shrink_to_fit();
  bytes_.shrink_to_fit();
}

void PackedRows::unpack(const Run& run, std::vector<ReadRow>& rows) const {
  std::size_t at = run.offset;
  std::int64_t position = 0;
  for (std::uint32_t i = 0; i < run.rows; ++i) {
    position += unzigzag(get_varint(bytes_, at));
    rows.push_back(ReadRow{.position = position, .probability = bytes_[at++]});
  }
}

std::size_t ReadRows::bucket_of(const ReadKey& key) {
  const std::uint64_t mixed =
      (static_cast<std::uint64_t>(key.high) ^
       static_cast<std::uint64_t>(key.low) ^
       static_cast<std::uint64_t>(key.reference)) *
      0x9E3779B97F4A7C15ULL;
  return static_cast<std::size_t>(mixed >> 54);  // the top 10 bits
}

ReadRows::ReadRows(std::vector<PackedRows> files, std::size_t threads)
    : files_(std::move(files)), buckets_(kBuckets) {
  // Every thread takes a share of the buckets and collects their reads from
  // all files: no two threads write to the same bucket.
  const std::size_t shares = std::clamp<std::size_t>(threads, 1, kBuckets);
  std::vector<std::vector<std::int32_t>> references(shares);
  parallel_for(shares, shares, [&](std::size_t, std::size_t share) {
    for (std::size_t file = 0; file < files_.size(); ++file) {
      const std::vector<PackedRows::Run>& runs = files_[file].runs();
      for (std::size_t run = 0; run < runs.size(); ++run) {
        const std::size_t bucket = bucket_of(runs[run].key);
        if (bucket % shares != share) continue;
        buckets_[bucket].push_back(
            Entry{.key = runs[run].key,
                  .file = static_cast<std::uint32_t>(file),
                  .run = static_cast<std::uint32_t>(run)});
        std::vector<std::int32_t>& seen = references[share];
        if (seen.empty() || seen.back() != runs[run].key.reference) {
          seen.push_back(runs[run].key.reference);
        }
      }
    }
    for (std::size_t bucket = share; bucket < kBuckets; bucket += shares) {
      std::ranges::sort(buckets_[bucket]);
    }
    std::ranges::sort(references[share]);
    const auto repeated = std::ranges::unique(references[share]);
    references[share].erase(repeated.begin(), repeated.end());
    return Status{};
  });
  for (const std::vector<std::int32_t>& part : references) {
    references_.insert(references_.end(), part.begin(), part.end());
  }
  std::ranges::sort(references_);
  const auto repeated = std::ranges::unique(references_);
  references_.erase(repeated.begin(), repeated.end());
}

bool ReadRows::has_reference(std::int32_t reference) const {
  return std::ranges::binary_search(references_, reference);
}

bool ReadRows::contains(const ReadKey& key) const {
  return std::ranges::binary_search(buckets_[bucket_of(key)], key, {},
                                    &Entry::key);
}

bool ReadRows::find(const ReadKey& key, std::vector<ReadRow>& rows) const {
  rows.clear();
  const auto range = std::ranges::equal_range(buckets_[bucket_of(key)], key, {},
                                              &Entry::key);
  for (const Entry& entry : range) {
    files_[entry.file].unpack(files_[entry.file].runs()[entry.run], rows);
  }
  return !range.empty();
}

}  // namespace deeprm
