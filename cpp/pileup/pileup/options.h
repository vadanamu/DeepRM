/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_OPTIONS_H_
#define DEEPRM_PILEUP_OPTIONS_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace deeprm {

// The command-line options of `deeprm call pileup`.
struct Options {
  std::string input;
  std::string output;
  std::string bam;
  std::string annot;
  std::string postfix;
  std::string tmp_dir;
  std::size_t threads = 0;
  double threshold = 0.98;
  double epsilon = 1e-30;
  std::optional<std::size_t> slice;
  bool flip = false;
  bool skip_modbam = false;
  bool report_all_stoichiometry = false;
  // --calibration on|off; unset: from deeprm_run.json (inputs_calibrated)
  std::optional<bool> calibrated;
  std::int64_t label_div = 1000000000;
};

// The help text: the usage line and one line per option.
std::string usage();

// What the arguments ask to print in place of a run.
enum class InfoRequest { kHelp, kVersion, kFullVersion };

// The first of -h, --help, -v, --version and -vv among the arguments, if any.
std::optional<InfoRequest> info_request(
    std::span<const std::string_view> arguments);

std::expected<Options, std::string> parse_options(
    std::span<const std::string_view> arguments);

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_OPTIONS_H_
