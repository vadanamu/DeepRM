/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include "pileup/options.h"

#include <algorithm>
#include <charconv>
#include <thread>

namespace deeprm {
namespace {

template <typename Number>
bool parse(std::string_view text, Number& value) {
  const std::from_chars_result result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  return result.ec == std::errc() && result.ptr == text.data() + text.size();
}

bool is(std::string_view argument, std::string_view long_name,
        std::string_view short_name) {
  return argument == long_name || (!short_name.empty() && argument == short_name);
}

// Sets the option `name` names from `value`; false for a bad value.
bool set_value(std::string_view name, std::string_view value,
               Options& options) {
  if (is(name, "--input", "-i")) return (options.input = value, true);
  if (is(name, "--output", "-o")) return (options.output = value, true);
  if (is(name, "--bam", "-b")) return (options.bam = value, true);
  if (is(name, "--annot", "-a")) return (options.annot = value, true);
  if (is(name, "--postfix", "-x")) return (options.postfix = value, true);
  if (is(name, "--tmp-dir", "")) return (options.tmp_dir = value, true);
  if (is(name, "--calibration", "")) {
    if (value != "on" && value != "off") return false;
    options.calibrated = value == "on";
    return true;
  }
  if (is(name, "--thread", "-t")) {
    return parse(value, options.threads) && options.threads > 0;
  }
  if (is(name, "--threshold", "-th")) return parse(value, options.threshold);
  if (is(name, "--epsilon", "-e")) return parse(value, options.epsilon);
  if (is(name, "--label_div", "-d")) {
    return parse(value, options.label_div) && options.label_div > 0;
  }
  if (is(name, "--slice", "-s")) {
    std::size_t slice = 0;
    if (!parse(value, slice)) return false;
    options.slice = slice;
    return true;
  }
  return false;
}

bool takes_value(std::string_view name) {
  constexpr std::string_view kNames[] = {
      "--input", "-i", "--output", "-o", "--bam", "-b", "--annot", "-a",
      "--postfix", "-x", "--tmp-dir", "--calibration", "--thread", "-t",
      "--threshold", "-th",
      "--epsilon", "-e", "--label_div", "-d", "--slice", "-s"};
  return std::ranges::find(kNames, name) != std::end(kNames);
}

}  // namespace

std::string usage() {
  return "Usage: deeprm_pileup -i DIR -o DIR -b BAM [options]\n"
         "\n"
         "Site-level pileup of DeepRM per-read predictions; the outputs are\n"
         "those of `deeprm call pileup`.\n"
         "\n"
         "Required:\n"
         "  -i,  --input DIR        folder of per-read predictions\n"
         "  -o,  --output DIR       folder for the site-level files\n"
         "  -b,  --bam FILE         BAM of the reads given to `deeprm call run`\n"
         "\n"
         "Options:\n"
         "  -t,  --thread INT       threads [95% of the CPUs]\n"
         "  -a,  --annot FILE       annotation (RefFlat, GenePred or RefGene):\n"
         "                          also write the genomic site tables\n"
         "  -sm, --skip-modbam      do not write the modBAM\n"
         "  -th, --threshold FLOAT  probability from which a read counts as\n"
         "                          modified [0.98]\n"
         "  -e,  --epsilon FLOAT    constant that keeps the scores finite [1e-30]\n"
         "  -x,  --postfix STR      suffix of the genomic pileup file names\n"
         "  -s,  --slice INT        model output to use, numbered from 0\n"
         "  -f,  --flip             use 1 - p for each probability p\n"
         "  -d,  --label_div INT    label ID factor of `deeprm call prep`\n"
         "                          [1000000000]\n"
         "       --tmp-dir DIR      folder for temporary files [output folder]\n"
         "       --calibration on|off\n"
         "                          apply the v1.2 model's calibration, or not\n"
         "                          [from deeprm_run.json in the input folder]\n"
         "       --report-all-stoichiometry\n"
         "                          also write the stoichiometry of the sites the\n"
         "                          rule does not report (their state stays -1)\n"
         "  -h,  --help             print this help\n"
         "  -v,  --version          print the version";
}

std::optional<InfoRequest> info_request(
    std::span<const std::string_view> arguments) {
  for (std::size_t i = 0; i < arguments.size(); ++i) {
    const std::string_view argument = arguments[i];
    if (is(argument, "--help", "-h")) return InfoRequest::kHelp;
    if (is(argument, "--version", "-v")) return InfoRequest::kVersion;
    if (argument == "-vv") return InfoRequest::kFullVersion;
    if (takes_value(argument)) ++i;  // its value, which may be "-v" (a postfix)
  }
  return std::nullopt;
}

std::expected<Options, std::string> parse_options(
    std::span<const std::string_view> arguments) {
  Options options;
  for (std::size_t i = 0; i < arguments.size(); ++i) {
    const std::string_view name = arguments[i];
    if (is(name, "--flip", "-f")) {
      options.flip = true;
    } else if (is(name, "--skip-modbam", "-sm")) {
      options.skip_modbam = true;
    } else if (name == "--report-all-stoichiometry") {
      options.report_all_stoichiometry = true;
    } else if (takes_value(name) && i + 1 < arguments.size()) {
      if (!set_value(name, arguments[++i], options)) {
        return std::unexpected("bad value for " + std::string(name));
      }
    } else {
      return std::unexpected("bad argument: " + std::string(name));
    }
  }
  if (options.input.empty() || options.output.empty() || options.bam.empty()) {
    return std::unexpected("--input, --output and --bam are required");
  }
  if (options.threads == 0) {
    // the default of the Python pileup: 95% of the CPUs
    options.threads = std::max<std::size_t>(
        1, std::thread::hardware_concurrency() * 95 / 100);
  }
  return options;
}

}  // namespace deeprm
