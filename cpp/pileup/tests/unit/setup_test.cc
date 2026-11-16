/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include <algorithm>
#include <cstddef>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

#include "pileup/options.h"
#include "pileup/pileup.h"
#include "pileup/run_info.h"
#include "pileup/version.h"
#include "tests/unit/unit_test.h"

namespace deeprm::test {
namespace {

std::vector<std::size_t> group_sizes(std::size_t files, std::size_t groups) {
  std::vector<std::size_t> sizes(groups, 0);
  for (const std::int32_t group : worker_groups(files, groups)) {
    ++sizes[static_cast<std::size_t>(group)];
  }
  return sizes;
}

void worker_groups_match_array_split(Checker& t) {
  // [len(a) for a in np.array_split(np.arange(10), 4)] == [3, 3, 2, 2]
  t.check(group_sizes(10, 4) == std::vector<std::size_t>{3, 3, 2, 2}, "10 / 4");
  t.check(group_sizes(7, 7) == std::vector<std::size_t>(7, 1), "7 / 7");
  const std::vector<std::int32_t> groups = worker_groups(10, 4);
  t.check(std::ranges::is_sorted(groups), "the files of a group are consecutive");
}

void options_are_parsed(Checker& t) {
  const std::vector<std::string_view> arguments = {
      "-i", "in", "--output", "out", "-b", "x.bam", "-t", "45", "-sm",
      "-th", "0.9", "--slice", "1", "-f", "-d", "1000"};
  const auto options = parse_options(arguments);
  t.check(options.has_value(), "the arguments are accepted");
  if (!options) return;
  t.equal(options->input, std::string("in"));
  t.equal(options->output, std::string("out"));
  t.equal(options->threads, std::size_t{45});
  t.equal(options->threshold, 0.9);
  t.equal(options->epsilon, 1e-30);
  t.check(options->slice.has_value() && *options->slice == 1, "--slice 1");
  t.check(options->flip && options->skip_modbam, "the flags are set");
  t.equal(options->label_div, std::int64_t{1000});
}

void bad_options_are_refused(Checker& t) {
  const std::vector<std::string_view> no_bam = {"-i", "in", "-o", "out"};
  t.check(!parse_options(no_bam).has_value(), "--bam is required");
  const std::vector<std::string_view> bad_thread = {"-i", "in", "-o", "out",
                                                    "-b", "x", "-t", "0"};
  t.check(!parse_options(bad_thread).has_value(), "--thread 0");
  const std::vector<std::string_view> unknown = {"-i", "in", "-o", "out",
                                                 "-b", "x", "--what"};
  t.check(!parse_options(unknown).has_value(), "an unknown option");
}

void help_names_every_option(Checker& t) {
  const std::string help = usage();
  for (const std::string_view name :
       {"-i,  --input", "-o,  --output", "-b,  --bam", "-t,  --thread",
        "-a,  --annot", "-sm, --skip-modbam", "-th, --threshold",
        "-e,  --epsilon", "-x,  --postfix", "-s,  --slice", "-f,  --flip",
        "-d,  --label_div", "--tmp-dir", "-h,  --help", "-v,  --version"}) {
    t.check(help.contains(name), name);
  }
  t.check(!help.contains("-vv"), "-vv is not in the help");
  for (const auto part : std::views::split(help, '\n')) {
    t.check(std::ranges::distance(part) <= 80, "a help line fits 80 columns");
  }
}

void help_and_version_are_asked_for(Checker& t) {
  using Arguments = std::vector<std::string_view>;
  t.check(info_request(Arguments{"-h"}) == InfoRequest::kHelp, "-h");
  t.check(info_request(Arguments{"-i", "in", "--help"}) == InfoRequest::kHelp,
          "--help after other options");
  t.check(info_request(Arguments{"-v"}) == InfoRequest::kVersion, "-v");
  t.check(info_request(Arguments{"--version"}) == InfoRequest::kVersion,
          "--version");
  t.check(info_request(Arguments{"-vv"}) == InfoRequest::kFullVersion, "-vv");
  t.check(!info_request(Arguments{"-i", "in", "-o", "out", "-b", "x"}),
          "a run");
  t.check(!info_request(Arguments{}), "no arguments");
}

void version_text_has_three_lines(Checker& t) {
  // the line DeepRM shows: the program, the version, the build time and commit
  t.check(version().starts_with("DeepRM Pileup v1.2.1 20"), "the version line");
  const std::string text = version_text();
  t.check(text.starts_with("Program: deeprm_pileup (DeepRM pileup)\n"),
          "the program line");
  t.check(text.contains("\nVersion: 1.2.1 built 20"), "the version line");
  t.check(text.ends_with("\nLicense: CC BY-NC-SA 4.0"),
          "the license line");
  t.equal(std::ranges::count(text, '\n'), std::ptrdiff_t{2});
}

void run_info_tells_the_bundled_weights(Checker& t) {
  t.check(run_info_is_calibrated(
              "{\n \"model_md5\": \"x\",\n \"site_rule_calibration\": true,\n}"),
          "the flag is true");
  t.check(!run_info_is_calibrated("{\n \"site_rule_calibration\": false\n}"),
          "the flag is false");
  t.check(run_info_is_calibrated(
              "{\n \"model_md5\": \"920a59eb2e895f48fc226a014246a1f7\"\n}"),
          "an earlier run file with the bundled md5");
  t.check(!run_info_is_calibrated("{\n \"model_md5\": \"abc\"\n}"),
          "another model file");
  t.check(!run_info_is_calibrated("{}"), "no information");
}

}  // namespace

std::vector<TestCase> setup_tests() {
  return {{"worker groups match np.array_split", worker_groups_match_array_split},
          {"options are parsed", options_are_parsed},
          {"bad options are refused", bad_options_are_refused},
          {"help names every option", help_names_every_option},
          {"help and version are asked for", help_and_version_are_asked_for},
          {"version text has three lines", version_text_has_three_lines},
          {"run info tells the bundled weights",
           run_info_tells_the_bundled_weights}};
}

}  // namespace deeprm::test
