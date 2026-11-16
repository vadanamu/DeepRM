/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include "pileup/table_writer.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>

namespace deeprm {
namespace {

using Bytes = std::vector<std::uint8_t>;

constexpr std::size_t kArrays = 13;
// Bytes kept per BED line before the first is written (a line has about 70).
constexpr std::size_t kBedLineRoom = 96;
// Rows whose array bytes are compressed together.
constexpr std::size_t kRowsPerRun = 8192;

void append_integer(std::string& out, std::int64_t value) {
  char text[24];
  const std::to_chars_result result =
      std::to_chars(text, text + sizeof(text), value);
  out.append(text, result.ptr);
}

// '%.2f' % value, and '' for NaN (as pandas writes a missing value).
void append_two_decimals(std::string& out, double value) {
  if (std::isnan(value)) return;
  char text[512];
  const std::to_chars_result result = std::to_chars(
      text, text + sizeof(text), value, std::chars_format::fixed, 2);
  out.append(text, result.ptr);
}

// np.rint(x).astype(np.int64), then np.clip(., low, high).
std::int64_t rint_clip(double x, std::int64_t low, std::int64_t high) {
  const double rounded = std::nearbyint(x);
  if (std::isnan(rounded)) return low;  // NumPy casts NaN to the minimum
  if (rounded <= static_cast<double>(low)) return low;
  if (rounded >= static_cast<double>(high)) return high;
  return static_cast<std::int64_t>(rounded);
}

// The field as a CSV writer with minimal quoting writes it.
std::string csv_field(const std::string& name) {
  if (name.find_first_of("\t\"\n\r") == std::string::npos) return name;
  std::string quoted = "\"";
  for (const char c : name) {
    if (c == '"') quoted.push_back('"');
    quoted.push_back(c);
  }
  quoted.push_back('"');
  return quoted;
}

void append_bed_line(std::string& out, const std::string& name,
                     const TableRow& row, const SiteScores& scores) {
  const std::int64_t count_all = row.sums.count_all;
  const std::int64_t modified = rint_clip(
      static_cast<double>(count_all) * scores.stoichiometry, 0, count_all);
  out.append(name).push_back('\t');
  append_integer(out, row.position);
  out.push_back('\t');
  append_integer(out, row.position + 1);
  out.append("\ta\t");
  append_integer(out, rint_clip(scores.modscore * 1000, 0, 1000));
  out.push_back('\t');
  out.push_back(row.minus ? '-' : '+');
  out.push_back('\t');
  append_integer(out, row.position);
  out.push_back('\t');
  append_integer(out, row.position + 1);
  out.append("\t255,0,0\t");
  append_integer(out, count_all);
  out.push_back('\t');
  append_two_decimals(out, scores.stoichiometry * 100);
  out.push_back('\t');
  append_integer(out, modified);
  out.push_back('\t');
  append_integer(out, count_all - modified);
  out.append("\t0\t0\t0\t0\t0\n");
}

template <typename Value>
void put(Bytes& out, Value value) {
  const std::size_t at = out.size();
  out.resize(at + sizeof(value));
  std::memcpy(out.data() + at, &value, sizeof(value));
}

// The text as UTF-32 padded with zeros to `width` characters ('<U' dtype).
void put_text(Bytes& out, const std::string& text, std::size_t width) {
  for (std::size_t i = 0; i < width; ++i) {
    put(out, static_cast<std::uint32_t>(
                 i < text.size() ? static_cast<unsigned char>(text[i]) : 0));
  }
}

}  // namespace

TableFormat::TableFormat(const std::vector<std::string>& reference_names,
                         const SiteScorer& scorer)
    : names_(reference_names), scorer_(scorer), name_width_(1) {
  for (const std::string& name : names_) {
    bed_names_.push_back(csv_field(name));
    name_width_ = std::max(name_width_, name.size());
  }
}

std::vector<NpzArray> TableFormat::empty_arrays() const {
  const std::string text = "<U" + std::to_string(name_width_);
  const std::pair<const char*, std::string> columns[kArrays] = {
      {"ref_names", text},
      {"ref_pos", "<i8"},
      {"ref_strand", "<U1"},
      {"modscore", "<f8"},
      {"stoichiometry", "<f8"},
      {"count_all", "<i8"},
      {"count_pos", "<i8"},
      {"stoich_state", "|i1"},
      {"stoichiometry_release", "<f8"},
      {"stoichiometry_eq14", "<f8"},
      {"logsum_1_p_pos", "<f8"},
      {"kl_div_neg", "<f8"},
      {"kl_div_pos", "<f8"}};
  std::vector<NpzArray> arrays;
  for (const auto& [name, descr] : columns) {
    arrays.push_back(
        NpzArray{.name = name, .descr = descr, .rows = 0, .runs = {}});
  }
  return arrays;
}

// The npz arrays' bytes for some rows, one buffer per array.
void TableFormat::put_columns(const TableRow& row, const SiteScores& scores,
                              std::vector<std::uint8_t>* columns) const {
  put_text(columns[0], names_[row.reference], name_width_);
  put(columns[1], row.position);
  put(columns[2], static_cast<std::uint32_t>(row.minus ? '-' : '+'));
  put(columns[3], scores.modscore);
  put(columns[4], scores.stoichiometry);
  put(columns[5], row.sums.count_all);
  put(columns[6], row.sums.count_pos);
  put(columns[7], scores.stoich_state);
  put(columns[8], scores.stoichiometry_release);
  put(columns[9], scores.stoichiometry_eq14);
  put(columns[10], row.sums.logsum_1_p_pos);
  put(columns[11], row.sums.kl_div_neg);
  put(columns[12], row.sums.kl_div_pos);
}

// The arrays are compressed a few thousand rows at a time: their bytes
// before compression (some 100 to 250 per row) are then never held for a
// whole part, of which as many are made at once as there are threads.
TablePart TableFormat::make_part(std::span<const TableRow> rows) const {
  TablePart part;
  part.rows = rows.size();
  part.runs.resize(kArrays);
  part.bed.reserve(rows.size() * kBedLineRoom);
  Bytes columns[kArrays];
  RunDeflater deflater;
  for (std::size_t first = 0; first < rows.size(); first += kRowsPerRun) {
    for (Bytes& column : columns) column.clear();
    for (const TableRow& row :
         rows.subspan(first, std::min(kRowsPerRun, rows.size() - first))) {
      const SiteScores scores = scorer_.score(row.sums);
      part.near_steps += scores.modscore_near_step ? 1 : 0;
      append_bed_line(part.bed, bed_names_[row.reference], row, scores);
      put_columns(row, scores, columns);
    }
    for (std::size_t i = 0; i < kArrays; ++i) {
      const DeflatedRun run = deflater.deflate(columns[i]);
      DeflatedRun& all = part.runs[i];
      all.crc32 = crc32_combine(all.crc32, run.crc32, run.size);
      all.size += run.size;
      all.bytes.insert(all.bytes.end(), run.bytes.begin(), run.bytes.end());
    }
  }
  for (DeflatedRun& run : part.runs) run.bytes.shrink_to_fit();
  return part;
}

TableWriter::TableWriter(const TableFormat& format, std::size_t threads)
    : threads_(threads), arrays_(format.empty_arrays()) {}

Status TableWriter::open(const std::string& stem) {
  stem_ = stem;
  return bed_.open(stem + ".bed");
}

Status TableWriter::append(std::vector<TablePart>& parts) {
  std::vector<std::uint64_t> offsets;
  for (TablePart& part : parts) {
    offsets.push_back(bed_size_);
    bed_size_ += part.bed.size();
    rows_ += part.rows;
    near_steps_ += part.near_steps;
    if (part.rows == 0) continue;
    for (std::size_t i = 0; i < arrays_.size(); ++i) {
      arrays_[i].rows += part.rows;
      arrays_[i].runs.push_back(std::move(part.runs[i]));
    }
  }
  const Status written =
      parallel_for(threads_, parts.size(), [&](std::size_t, std::size_t i) {
        const std::string& bed = parts[i].bed;
        return bed_.write_at(
            offsets[i], {reinterpret_cast<const std::uint8_t*>(bed.data()),
                         bed.size()});
      });
  parts.clear();
  return written;
}

Status TableWriter::finish() {
  if (Status status = bed_.close(); !status) return status;
  const Status written = write_npz(stem_ + ".npz", arrays_, threads_);
  arrays_ = std::vector<NpzArray>();  // the compressed arrays are not kept
  return written;
}

}  // namespace deeprm
