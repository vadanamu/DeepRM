/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <fstream>
#include <iterator>
#include <optional>
#include <print>
#include <string>
#include <string_view>
#include <vector>

#include "libdeflate/libdeflate.h"

#include "pileup/allocator_setup.h"
#include "pileup/cpu_features.h"
#include "pileup/math_tools.h"
#include "pileup/npz_reader.h"
#include "pileup/options.h"
#include "pileup/pileup.h"
#include "pileup/svml_port.h"
#include "pileup/version.h"

namespace {

std::expected<std::vector<std::uint8_t>, std::string> read_file(
    const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::unexpected("cannot open " + path);
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in),
                                   std::istreambuf_iterator<char>());
}

// Prints "<file> <array> <dtype> <shape> <crc32 of the array bytes>" per array.
int npz_info(int argc, char** argv) {
  deeprm::NpzReader reader;
  for (int i = 2; i < argc; ++i) {
    const std::expected<std::vector<std::uint8_t>, std::string> file =
        read_file(argv[i]);
    if (!file) {
      std::println(stderr, "{}", file.error());
      return EXIT_FAILURE;
    }
    const std::expected<std::vector<deeprm::NpyArray>, std::string> arrays =
        reader.read_all(*file);
    if (!arrays) {
      std::println(stderr, "{}: {}", argv[i], arrays.error());
      return EXIT_FAILURE;
    }
    for (const deeprm::NpyArray& array : *arrays) {
      std::string shape;
      for (const std::size_t dim : array.shape) {
        shape += std::to_string(dim) + ",";
      }
      std::println("{}\t{}\t{}\t({})\t{:08x}", argv[i], array.name,
                   array.descr, shape,
                   libdeflate_crc32(0, array.data.data(), array.data.size()));
    }
  }
  return EXIT_SUCCESS;
}

int pileup(const std::vector<std::string_view>& arguments) {
  const std::expected<deeprm::Options, std::string> options =
      deeprm::parse_options(arguments);
  if (!options) {
    std::println(stderr, "error: {}\nTry `deeprm_pileup --help`.",
                 options.error());
    return EXIT_FAILURE;
  }
  if (!deeprm::svml_uses_avx512()) {
    std::println(stderr, "note: log2 and log10 run without AVX512 (the "
                         "results are the same)");
  }
  const deeprm::Status status = deeprm::run_pileup(*options);
  if (!status) {
    std::println(stderr, "error: {}", status.error());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  deeprm::set_up_allocator();
  const std::vector<std::string_view> all_arguments(argv + 1, argv + argc);
  if (const std::optional<deeprm::InfoRequest> request =
          deeprm::info_request(all_arguments)) {
    // DeepRM runs `--version` to learn whether the program can run here, and
    // uses the Python pileup when it cannot: so the version is printed only on
    // a CPU the pileup runs on.
    if (*request != deeprm::InfoRequest::kHelp &&
        !deeprm::cpu_has_avx2_fma()) {
      std::println(stderr, "deeprm_pileup needs a CPU with AVX2 and FMA");
      return EXIT_FAILURE;
    }
    switch (*request) {
      case deeprm::InfoRequest::kHelp:
        std::println("{}", deeprm::usage());
        break;
      case deeprm::InfoRequest::kVersion:
        std::println("{}", deeprm::version());
        break;
      case deeprm::InfoRequest::kFullVersion:
        std::println("{}", deeprm::version_text());
        break;
    }
    return EXIT_SUCCESS;
  }
  if (!deeprm::cpu_has_avx2_fma()) {
    std::println(stderr, "deeprm_pileup needs a CPU with AVX2 and FMA");
    return EXIT_FAILURE;
  }
  const std::vector<std::string_view> arguments(argv + std::min(argc, 2),
                                                argv + argc);
  if (argc >= 2) {
    const int result = deeprm::run_math_tool(argv[1], arguments);
    if (result >= 0) return result;
  }
  if (argc >= 3 && std::string_view(argv[1]) == "npz-info") {
    return npz_info(argc, argv);
  }
  return pileup(all_arguments);
}
