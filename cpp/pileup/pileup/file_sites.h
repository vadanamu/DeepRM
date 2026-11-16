/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_FILE_SITES_H_
#define DEEPRM_PILEUP_FILE_SITES_H_

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "pileup/npz_reader.h"
#include "pileup/parallel.h"
#include "pileup/per_read_terms.h"
#include "pileup/read_rows.h"
#include "pileup/site_sums.h"

namespace deeprm {

// How the rows of a per-read file are read (--slice, --flip).
struct ReadOptions {
  std::optional<std::size_t> slice;
  bool flip;
  std::int64_t label_div;  // separates reference and position in label_id
};

// The bytes of the file at `path`.
Status read_file_bytes(const std::string& path,
                       std::vector<std::uint8_t>& bytes);

// Reads per-read files and adds up their rows per site. One per thread.
class FileSiteReader {
 public:
  FileSiteReader(const ReadOptions& read_options,
                 const TermOptions& term_options);

  // The sites of the per-read file whose bytes are `file` (`path` names it
  // in errors), sorted by label_id, into `sites`; with `rows`, also its
  // per-read rows (for the modBAM) into `rows`.
  Status read(std::span<const std::uint8_t> file, const std::string& path,
              std::vector<FileSite>& sites, PackedRows* rows = nullptr);

  // The label_id of every row of the file (valid or not), appended.
  Status read_labels(const std::string& path,
                     std::vector<std::int64_t>& labels);

 private:
  Status load_rows(std::span<const std::uint8_t> file, const std::string& path,
                   bool with_read_ids);
  void pack_rows(PackedRows& rows) const;
  void add_up(std::vector<FileSite>& sites);

  ReadOptions read_options_;
  NpzReader npz_;
  PerReadTermsCalculator calculator_;
  std::vector<std::uint8_t> file_;
  std::vector<float> pred_;
  std::vector<std::int64_t> label_id_;
  std::vector<std::int64_t> read_id_;  // two per row
  std::vector<std::pair<std::int64_t, std::uint32_t>> order_;
};

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_FILE_SITES_H_
