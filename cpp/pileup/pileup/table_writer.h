/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_TABLE_WRITER_H_
#define DEEPRM_PILEUP_TABLE_WRITER_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "pileup/npz_writer.h"
#include "pileup/output_file.h"
#include "pileup/parallel.h"
#include "pileup/site_scores.h"
#include "pileup/site_sums.h"

namespace deeprm {

// One site of a site table (transcript or genomic).
struct TableRow {
  std::uint32_t reference;  // index into the table's reference names
  std::int64_t position;    // 0-based
  bool minus;
  SiteSums sums;
};

// Consecutive rows of a site table, formatted: their BED lines and the runs
// of their npz arrays.
struct TablePart {
  std::size_t rows = 0;
  std::size_t near_steps = 0;  // rows with SiteScores::modscore_near_step
  std::string bed;
  std::vector<DeflatedRun> runs;  // in the order of the npz arrays
};

// Formats the rows of a site table as the Python pileup writes them
// (bed_formatter and _table_arrays). Thread-safe.
class TableFormat {
 public:
  TableFormat(const std::vector<std::string>& reference_names,
              const SiteScorer& scorer);

  TablePart make_part(std::span<const TableRow> rows) const;

  // Name and dtype of each npz array, in file order.
  std::vector<NpzArray> empty_arrays() const;

 private:
  void put_columns(const TableRow& row, const SiteScores& scores,
                   std::vector<std::uint8_t>* columns) const;

  const std::vector<std::string>& names_;
  const SiteScorer& scorer_;
  std::vector<std::string> bed_names_;  // quoted where a CSV writer would
  std::size_t name_width_;              // characters of the '<U' dtype
};

// Writes <stem>.bed and <stem>.npz from parts given in row order.
class TableWriter {
 public:
  TableWriter(const TableFormat& format, std::size_t threads);

  Status open(const std::string& stem);
  // Writes the BED lines of the parts and keeps their npz runs.
  Status append(std::vector<TablePart>& parts);
  Status finish();

  std::size_t rows() const { return rows_; }
  std::size_t near_steps() const { return near_steps_; }

 private:
  std::size_t threads_;
  std::string stem_;
  OutputFile bed_;
  std::uint64_t bed_size_ = 0;
  std::vector<NpzArray> arrays_;
  std::size_t rows_ = 0;
  std::size_t near_steps_ = 0;
};

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_TABLE_WRITER_H_
