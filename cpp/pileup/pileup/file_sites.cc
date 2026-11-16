/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include "pileup/file_sites.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>

namespace deeprm {

Status read_file_bytes(const std::string& path,
                       std::vector<std::uint8_t>& bytes) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) return std::unexpected("cannot open " + path);
  bytes.resize(static_cast<std::size_t>(in.tellg()));
  in.seekg(0);
  in.read(reinterpret_cast<char*>(bytes.data()),
          static_cast<std::streamsize>(bytes.size()));
  if (!in) return std::unexpected("cannot read " + path);
  return {};
}

namespace {

template <typename Value>
Value load(const std::vector<std::uint8_t>& bytes, std::size_t index) {
  Value value;
  std::memcpy(&value, bytes.data() + index * sizeof(value), sizeof(value));
  return value;
}

// Whether the predictions of row `row` (`width` values) are all finite.
bool row_is_finite(const NpyArray& pred, std::size_t row, std::size_t width) {
  for (std::size_t column = 0; column < width; ++column) {
    if (!std::isfinite(load<float>(pred.data, row * width + column))) {
      return false;
    }
  }
  return true;
}

Status check_arrays(const NpyArray& pred, const NpyArray& label_id,
                    const std::optional<std::size_t>& slice,
                    const std::string& path) {
  if (pred.descr != "<f4") {
    return std::unexpected("Expected pred to be float32 in " + path);
  }
  if (label_id.descr != "<i8") {
    return std::unexpected("Expected label_id to be int64 in " + path);
  }
  if (pred.shape.empty() || label_id.shape.empty() ||
      pred.shape[0] != label_id.shape[0]) {
    return std::unexpected("Length of pred and label_id do not match in " +
                           path);
  }
  if (pred.shape.size() != (slice ? 2U : 1U)) {
    return std::unexpected(
        std::string("Expected pred to be ") + (slice ? "2D" : "1D") + " in " +
        path);
  }
  if (slice && *slice >= pred.shape[1]) {
    return std::unexpected("Slice index is out of bounds for pred in " + path);
  }
  return {};
}

}  // namespace

FileSiteReader::FileSiteReader(const ReadOptions& read_options,
                               const TermOptions& term_options)
    : read_options_(read_options), calculator_(term_options) {}

// The valid rows of the file into pred_ and label_id_, as _read_file() of
// the Python pileup returns them.
Status FileSiteReader::load_rows(std::span<const std::uint8_t> file,
                                 const std::string& path,
                                 bool with_read_ids) {
  const std::expected<NpyArray, std::string> pred = npz_.read(file, "pred");
  if (!pred) return std::unexpected(path + ": " + pred.error());
  const std::expected<NpyArray, std::string> label_id =
      npz_.read(file, "label_id");
  if (!label_id) return std::unexpected(path + ": " + label_id.error());
  const std::optional<std::size_t>& slice = read_options_.slice;
  if (Status status = check_arrays(*pred, *label_id, slice, path); !status) {
    return status;
  }
  std::expected<NpyArray, std::string> read_id = NpyArray{};
  if (with_read_ids) {
    read_id = npz_.read(file, "read_id");
    if (!read_id) return std::unexpected(path + ": " + read_id.error());
    if (read_id->descr != "<i8" || read_id->shape.size() != 2 ||
        read_id->shape[0] != label_id->shape[0] || read_id->shape[1] != 2) {
      return std::unexpected("Expected read_id to be int64 pairs in " + path);
    }
  }
  const std::size_t width = slice ? pred->shape[1] : 1;
  pred_.clear();
  label_id_.clear();
  read_id_.clear();
  for (std::size_t row = 0; row < pred->shape[0]; ++row) {
    if (!row_is_finite(*pred, row, width)) continue;
    const float value =
        load<float>(pred->data, row * width + (slice ? *slice : 0));
    const float used = read_options_.flip ? 1.0F - value : value;
    if (used < 0.0F || used > 1.0F) {
      return std::unexpected("pred is outside [0, 1] in " + path);
    }
    pred_.push_back(used);
    label_id_.push_back(load<std::int64_t>(label_id->data, row));
    if (with_read_ids) {
      read_id_.push_back(load<std::int64_t>(read_id->data, 2 * row));
      read_id_.push_back(load<std::int64_t>(read_id->data, 2 * row + 1));
    }
  }
  return {};
}

// The rows as the modBAM needs them: by read, with the reference position
// and the probability as a byte.
void FileSiteReader::pack_rows(PackedRows& file_rows) const {
  PackedRows rows;  // of this function's own, as the sites in add_up
  for (std::size_t i = 0; i < label_id_.size(); ++i) {
    const std::int64_t label = label_id_[i];
    const std::int64_t packed = (label < 0 ? -label : label) - 1;
    rows.add(ReadKey{.reference = static_cast<std::int32_t>(
                         packed / read_options_.label_div),
                     .high = read_id_[2 * i],
                     .low = read_id_[2 * i + 1]},
             ReadRow{.position = packed % read_options_.label_div,
                     .probability = probability_byte(pred_[i])});
  }
  rows.finish();
  file_rows = std::move(rows);
}

// Adds the rows up per site in row order (np.add.at in each term's dtype).
// The sites are collected in a vector of this function's own: the vectors
// of the files of a batch lie side by side, and threads that each append to
// their own would keep taking the same cache line from each other.
void FileSiteReader::add_up(std::vector<FileSite>& file_sites) {
  std::vector<FileSite> sites = std::move(file_sites);
  const PerReadTerms& terms = calculator_.compute(pred_);
  order_.resize(label_id_.size());
  for (std::size_t i = 0; i < label_id_.size(); ++i) {
    order_[i] = {label_id_[i], static_cast<std::uint32_t>(i)};
  }
  std::ranges::sort(order_);
  sites.clear();
  sites.reserve(order_.size());
  for (std::size_t first = 0; first < order_.size();) {
    const std::int64_t label = order_[first].first;
    std::uint32_t count_all = 0;
    std::uint32_t count_pos = 0;
    std::uint32_t count_pos_rule = 0;
    float kl_div_neg = 0.0F;
    float kl_div_pos = 0.0F;
    double logsum_1_p_pos = 0.0;
    double logsum_1_p_pos_rule = 0.0;
    double logsum_1_p_pos_calib = 0.0;
    for (; first < order_.size() && order_[first].first == label; ++first) {
      const std::uint32_t row = order_[first].second;
      ++count_all;
      count_pos += terms.count_pos[row];
      count_pos_rule += terms.count_pos_rule[row];
      kl_div_neg += terms.kl_div_neg[row];
      kl_div_pos += terms.kl_div_pos[row];
      logsum_1_p_pos += terms.logsum_1_p_pos[row];
      logsum_1_p_pos_rule += terms.logsum_1_p_pos_rule[row];
      logsum_1_p_pos_calib += terms.logsum_1_p_pos_calib[row];
    }
    sites.push_back(FileSite{.label_id = label,
                             .count_all = count_all,
                             .count_pos = count_pos,
                             .count_pos_rule = count_pos_rule,
                             .kl_div_neg = kl_div_neg,
                             .kl_div_pos = kl_div_pos,
                             .logsum_1_p_pos = logsum_1_p_pos,
                             .logsum_1_p_pos_rule = logsum_1_p_pos_rule,
                             .logsum_1_p_pos_calib = logsum_1_p_pos_calib});
  }
  file_sites = std::move(sites);
}

Status FileSiteReader::read(std::span<const std::uint8_t> file,
                            const std::string& path,
                            std::vector<FileSite>& sites, PackedRows* rows) {
  if (Status status = load_rows(file, path, rows != nullptr); !status) {
    return status;
  }
  if (rows != nullptr) pack_rows(*rows);
  add_up(sites);
  return {};
}

Status FileSiteReader::read_labels(const std::string& path,
                                   std::vector<std::int64_t>& labels) {
  if (Status status = read_file_bytes(path, file_); !status) return status;
  const std::expected<NpyArray, std::string> label_id =
      npz_.read(file_, "label_id");
  if (!label_id) return std::unexpected(path + ": " + label_id.error());
  if (label_id->descr != "<i8") {
    return std::unexpected("Expected label_id to be int64 in " + path);
  }
  const std::size_t count = label_id->data.size() / sizeof(std::int64_t);
  for (std::size_t i = 0; i < count; ++i) {
    labels.push_back(load<std::int64_t>(label_id->data, i));
  }
  return {};
}

}  // namespace deeprm
