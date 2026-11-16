/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include "pileup/math_tools.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <print>
#include <vector>

#include "pileup/cpu_features.h"
#include "pileup/numpy_math.h"
#include "pileup/parallel.h"
#include "pileup/per_read_terms.h"
#include "pileup/svml_call.h"
#include "pileup/svml_port.h"

namespace deeprm {
namespace {

constexpr std::size_t kBlock = std::size_t{1} << 16;

template <typename Number>
std::optional<Number> parse(std::string_view text) {
  Number value{};
  const std::from_chars_result result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc() || result.ptr != text.data() + text.size()) {
    return std::nullopt;
  }
  return value;
}

template <typename Value>
void write_array(const std::vector<Value>& values) {
  std::fwrite(values.data(), sizeof(values[0]), values.size(), stdout);
}

// The floats whose bit patterns are first, first + 1, ...
std::vector<float> floats_from_bits(std::uint64_t first, std::size_t count) {
  std::vector<float> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    values[i] = std::bit_cast<float>(static_cast<std::uint32_t>(first + i));
  }
  return values;
}

// Writes float32 calibration_logit_libm() of every input.
int logit_range(std::uint64_t first, std::uint64_t count) {
  std::vector<float> logit;
  for (std::uint64_t done = 0; done < count; done += kBlock) {
    const std::size_t size = std::min<std::uint64_t>(kBlock, count - done);
    const std::vector<float> pred = floats_from_bits(first + done, size);
    logit.resize(size);
    std::ranges::transform(pred, logit.begin(), calibration_logit_libm);
    write_array(logit);
  }
  return EXIT_SUCCESS;
}

// Writes, per block of 65,536 inputs, the arrays of PerReadTerms in order.
int terms_range(std::uint64_t first, std::uint64_t count,
                const TermOptions& options) {
  PerReadTermsCalculator calculator(options);
  for (std::uint64_t done = 0; done < count; done += kBlock) {
    const std::size_t size = std::min<std::uint64_t>(kBlock, count - done);
    const PerReadTerms& terms =
        calculator.compute(floats_from_bits(first + done, size));
    write_array(terms.count_pos);
    write_array(terms.logsum_1_p_pos);
    write_array(terms.kl_div_neg);
    write_array(terms.kl_div_pos);
    write_array(terms.count_pos_rule);
    write_array(terms.logsum_1_p_pos_rule);
    write_array(terms.logsum_1_p_pos_calib);
  }
  return EXIT_SUCCESS;
}

// Compares the ported functions with SVML for the 65,536 inputs from `first`;
// returns how many log2 and log10 results differ.
std::array<std::uint64_t, 2> port_differences(std::uint64_t first) {
  const std::vector<float> in = floats_from_bits(first, kBlock);
  std::vector<float> log2(kBlock);
  std::vector<float> log10(kBlock);
  svml_log2_avx512(in, log2);
  svml_log10_avx512(in, log10);
  std::array<std::uint64_t, 2> differences{};
  for (std::size_t i = 0; i < kBlock; ++i) {
    const auto bits = [](float value) {
      return std::bit_cast<std::uint32_t>(value);
    };
    differences[0] += bits(ported_log2(in[i])) != bits(log2[i]);
    differences[1] += bits(ported_log10(in[i])) != bits(log10[i]);
  }
  return differences;
}

int svml_port_check(std::size_t threads) {
  if (!cpu_has_avx512()) {
    std::println(stderr, "svml-port-check needs a CPU with AVX512");
    return EXIT_FAILURE;
  }
  const std::size_t blocks = (std::uint64_t{1} << 32) / kBlock;
  std::atomic<std::uint64_t> log2_differences{0};
  std::atomic<std::uint64_t> log10_differences{0};
  const Status status =
      parallel_for(threads, blocks, [&](std::size_t, std::size_t block) {
        const std::array<std::uint64_t, 2> differences =
            port_differences(block * kBlock);
        log2_differences += differences[0];
        log10_differences += differences[1];
        return Status{};
      });
  std::println("float32 bit patterns: {}", blocks * kBlock);
  std::println("log2 differences: {}", log2_differences.load());
  std::println("log10 differences: {}", log10_differences.load());
  const bool same = status && log2_differences == 0 && log10_differences == 0;
  std::println("{}", same ? "RESULT IDENTICAL" : "RESULT DIFFERENT");
  return same ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace

int run_math_tool(std::string_view command,
                  std::span<const std::string_view> arguments) {
  if (command == "calibration-checksum") {
    std::println("{:#018x}", calibration_checksum());
    return EXIT_SUCCESS;
  }
  if (command == "svml-port-check") {
    const std::optional<std::size_t> threads =
        arguments.empty() ? std::optional<std::size_t>(1)
                          : parse<std::size_t>(arguments[0]);
    return threads ? svml_port_check(*threads) : EXIT_FAILURE;
  }
  if (command != "logit-range" && command != "terms-range") return -1;
  if (arguments.size() < 2) return EXIT_FAILURE;
  const std::optional<std::uint64_t> first =
      parse<std::uint64_t>(arguments[0]);
  const std::optional<std::uint64_t> count =
      parse<std::uint64_t>(arguments[1]);
  if (!first || !count) return EXIT_FAILURE;
  if (command == "logit-range") return logit_range(*first, *count);
  if (arguments.size() != 5) return EXIT_FAILURE;
  const std::optional<double> threshold = parse<double>(arguments[2]);
  const std::optional<double> epsilon = parse<double>(arguments[3]);
  if (!threshold || !epsilon) return EXIT_FAILURE;
  return terms_range(*first, *count,
                     TermOptions{.threshold = *threshold,
                                 .epsilon = *epsilon,
                                 .calibrated = arguments[4] == "1"});
}

}  // namespace deeprm
