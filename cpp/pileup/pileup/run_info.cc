/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include "pileup/run_info.h"

#include <fstream>
#include <iterator>
#include <optional>
#include <print>

namespace deeprm {
namespace {

constexpr std::string_view kRunInfoName = "deeprm_run.json";
// md5 of the bundled weights file (BUNDLED_WEIGHTS_MD5 of the Python code).
constexpr std::string_view kBundledWeightsMd5 =
    "920a59eb2e895f48fc226a014246a1f7";

// The text of the value of the top-level key `key`, up to the next comma,
// brace or line end. The file is written by json.dump with one key per line.
std::optional<std::string_view> value_of(std::string_view json,
                                         std::string_view key) {
  const std::string quoted = "\"" + std::string(key) + "\"";
  const std::size_t at = json.find(quoted);
  if (at == std::string_view::npos) return std::nullopt;
  const std::size_t colon = json.find(':', at + quoted.size());
  if (colon == std::string_view::npos) return std::nullopt;
  std::string_view value = json.substr(colon + 1);
  value = value.substr(0, value.find_first_of(",}\n"));
  while (!value.empty() && value.front() == ' ') value.remove_prefix(1);
  while (!value.empty() && value.back() == ' ') value.remove_suffix(1);
  return value;
}

}  // namespace

bool run_info_is_calibrated(std::string_view json) {
  if (const std::optional<std::string_view> flag =
          value_of(json, "site_rule_calibration")) {
    return *flag == "true";
  }
  const std::optional<std::string_view> md5 = value_of(json, "model_md5");
  return md5 && *md5 == "\"" + std::string(kBundledWeightsMd5) + "\"";
}

bool inputs_calibrated(const std::string& input_dir) {
  const std::string path = input_dir + "/" + std::string(kRunInfoName);
  std::ifstream in(path);
  if (!in) {
    std::println(stderr,
                 "warning: no {} in {}; the stoichiometry rule uses the "
                 "predictions without the v1.2 model's calibration",
                 kRunInfoName, input_dir);
    return false;
  }
  const std::string json{std::istreambuf_iterator<char>(in),
                         std::istreambuf_iterator<char>()};
  const bool calibrated = run_info_is_calibrated(json);
  if (!calibrated) {
    std::println(stderr,
                 "warning: {}: not the bundled v1.2 weights; the stoichiometry "
                 "rule uses the predictions without calibration",
                 path);
  }
  return calibrated;
}

}  // namespace deeprm
