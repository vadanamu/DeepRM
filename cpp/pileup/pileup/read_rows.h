/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_READ_ROWS_H_
#define DEEPRM_PILEUP_READ_ROWS_H_

#include <compare>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace deeprm {

// A read on one reference: the reference index and the two int64 halves of
// the read's UUID, as the per-read files hold them.
struct ReadKey {
  std::int32_t reference;
  std::int64_t high;
  std::int64_t low;

  friend auto operator<=>(const ReadKey&, const ReadKey&) = default;
};

// One prediction of a read: the reference position and the probability as
// the modBAM holds it (0-255).
struct ReadRow {
  std::int64_t position;
  std::uint8_t probability;
};

// round(p * 255) as the Python pileup makes it, in float32.
std::uint8_t probability_byte(float prediction);

// The per-read rows of one per-read file, packed: a file holds the rows of a
// read together, with positions close to each other.
class PackedRows {
 public:
  // Rows of one read, consecutive in the file.
  struct Run {
    ReadKey key;
    std::uint32_t offset;  // of its first row in the packed bytes
    std::uint32_t rows;
  };

  void add(const ReadKey& key, const ReadRow& row);
  // Frees the memory that add() kept spare.
  void finish();

  const std::vector<Run>& runs() const { return runs_; }
  // Appends the rows of `run` to `rows`.
  void unpack(const Run& run, std::vector<ReadRow>& rows) const;

 private:
  std::vector<Run> runs_;
  std::vector<std::uint8_t> bytes_;
  std::int64_t last_position_ = 0;
};

// The per-read rows of all files, found by read.
class ReadRows {
 public:
  // Sorts the reads of the files on up to `threads` threads.
  ReadRows(std::vector<PackedRows> files, std::size_t threads);

  bool empty() const { return references_.empty(); }
  bool has_reference(std::int32_t reference) const;
  // Whether the read has rows.
  bool contains(const ReadKey& key) const;
  // The rows of the read into `rows`; false when it has none.
  bool find(const ReadKey& key, std::vector<ReadRow>& rows) const;

 private:
  struct Entry {
    ReadKey key;
    std::uint32_t file;
    std::uint32_t run;

    friend auto operator<=>(const Entry&, const Entry&) = default;
  };

  // The reads are shared out by a hash of their key, so that the buckets
  // can be sorted side by side.
  static constexpr std::size_t kBuckets = 1024;
  static std::size_t bucket_of(const ReadKey& key);

  std::vector<PackedRows> files_;
  std::vector<std::vector<Entry>> buckets_;  // each sorted
  std::vector<std::int32_t> references_;     // those with rows, sorted
};

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_READ_ROWS_H_
