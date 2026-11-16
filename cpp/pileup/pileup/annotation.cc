/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include "pileup/annotation.h"

#include <charconv>
#include <fstream>
#include <iterator>
#include <optional>
#include <span>
#include <unordered_set>

namespace deeprm {
namespace {

constexpr std::size_t kColumns = 10;

std::vector<std::string_view> split(std::string_view text, char separator) {
  std::vector<std::string_view> fields;
  while (true) {
    const std::size_t at = text.find(separator);
    fields.push_back(text.substr(0, at));
    if (at == std::string_view::npos) return fields;
    text.remove_prefix(at + 1);
  }
}

std::optional<std::int64_t> parse_integer(std::string_view text) {
  std::int64_t value = 0;
  const std::from_chars_result parsed =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || parsed.ec != std::errc() ||
      parsed.ptr != text.data() + text.size()) {
    return std::nullopt;
  }
  return value;
}

std::string_view strip(std::string_view text) {
  while (!text.empty() && text.front() == ' ') text.remove_prefix(1);
  while (!text.empty() && text.back() == ' ') text.remove_suffix(1);
  return text;
}

// Non-negative integers separated by commas, with at most one trailing comma
// (UCSC style). Nothing for any other text.
std::optional<std::vector<std::int64_t>> parse_exon_list(std::string_view text) {
  text = strip(text);
  std::vector<std::int64_t> values;
  if (text.empty()) return values;
  if (text.back() == ',') text.remove_suffix(1);
  for (const std::string_view item : split(text, ',')) {
    const std::optional<std::int64_t> value = parse_integer(item);
    if (!value || item.front() == '-') return std::nullopt;
    values.push_back(*value);
  }
  return values;
}

// True if every exon is longer than 0 and neither starts nor ends decrease.
bool exons_are_valid(const Transcript& transcript) {
  const std::vector<std::int64_t>& starts = transcript.exon_starts;
  const std::vector<std::int64_t>& ends = transcript.exon_ends;
  for (std::size_t i = 0; i < starts.size(); ++i) {
    if (ends[i] <= starts[i]) return false;
    if (i > 0 && (starts[i] < starts[i - 1] || ends[i] < ends[i - 1])) {
      return false;
    }
  }
  return true;
}

// The 10 GenePred columns of a row with `columns` columns.
std::span<const std::string_view> genepred_columns(
    const std::vector<std::string_view>& fields) {
  const std::span<const std::string_view> all(fields);
  return fields.size() == 11 ? all.subspan(1) : all.first(kColumns);
}

// The transcript of one row; nothing for a row that the Python pileup drops.
std::expected<std::optional<Transcript>, std::string> parse_row(
    std::span<const std::string_view> row) {
  if (row[2] != "+" && row[2] != "-") {
    return std::unexpected("Invalid strand '" + std::string(row[2]) +
                           "' for transcript " + std::string(row[0]));
  }
  std::int64_t numbers[5];
  for (std::size_t i = 0; i < 5; ++i) {
    const std::optional<std::int64_t> value = parse_integer(row[3 + i]);
    if (!value) {
      return std::unexpected("not an integer in the row of transcript " +
                             std::string(row[0]));
    }
    numbers[i] = *value;
  }
  std::optional<std::vector<std::int64_t>> starts = parse_exon_list(row[8]);
  std::optional<std::vector<std::int64_t>> ends = parse_exon_list(row[9]);
  if (!starts || !ends) {
    return std::unexpected("bad exon list in the row of transcript " +
                           std::string(row[0]));
  }
  // txEnd > txStart, cdsEnd >= cdsStart, exonCount > 0 and as many exons
  const std::int64_t exons = numbers[4];
  if (numbers[1] <= numbers[0] || numbers[3] < numbers[2] || exons <= 0 ||
      static_cast<std::int64_t>(starts->size()) != exons ||
      static_cast<std::int64_t>(ends->size()) != exons) {
    return std::nullopt;
  }
  Transcript transcript{.name = std::string(row[0]),
                        .chrom = std::string(row[1]),
                        .plus = row[2] == "+",
                        .exon_starts = std::move(*starts),
                        .exon_ends = std::move(*ends)};
  if (!exons_are_valid(transcript)) return std::nullopt;
  return transcript;
}

}  // namespace

std::expected<std::vector<Transcript>, std::string> parse_annotation(
    std::string_view text) {
  std::vector<Transcript> transcripts;
  std::unordered_set<std::string> seen;
  std::size_t columns = 0;
  for (std::string_view line : split(text, '\n')) {
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    if (line.empty()) continue;
    const std::vector<std::string_view> fields = split(line, '\t');
    if (columns == 0) {
      columns = fields.size();
      if (columns != 10 && columns != 11 && columns != 15) {
        return std::unexpected(
            "Invalid annotation file format. Expected 10 (GenePred), 11 "
            "(RefFlat), or 15 (RefGene) columns.");
      }
    }
    if (fields.size() != columns) {
      return std::unexpected("annotation rows differ in their column count");
    }
    std::expected<std::optional<Transcript>, std::string> transcript =
        parse_row(genepred_columns(fields));
    if (!transcript) return std::unexpected(transcript.error());
    if (!*transcript || !seen.insert((*transcript)->name).second) continue;
    transcripts.push_back(std::move(**transcript));
  }
  if (transcripts.empty()) {
    return std::unexpected("No valid transcripts found in the annotation file.");
  }
  return transcripts;
}

std::expected<std::vector<Transcript>, std::string> read_annotation(
    const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::unexpected("--annot: annotation file not found: " + path);
  const std::string text((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  return parse_annotation(text);
}

}  // namespace deeprm
