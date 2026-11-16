/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include "pileup/version.h"

#include <format>
#include <string_view>

#include "pileup/build_time.h"

namespace deeprm {
namespace {

constexpr std::string_view kVersion = "1.2.1";
constexpr std::string_view kLicense = "CC BY-NC-SA 4.0";

}  // namespace

// One line, as `deeprm_preprocess --version` prints it; DeepRM shows it.
std::string version() {
  return std::format("DeepRM Pileup v{} {}-{}", kVersion, DEEPRM_BUILD_TIME,
                     DEEPRM_GIT_HASH);
}

std::string version_text() {
  return std::format(
      "Program: deeprm_pileup (DeepRM pileup)\n"
      "Version: {} built {}-{}\n"
      "License: {}",
      kVersion, DEEPRM_BUILD_TIME, DEEPRM_GIT_HASH, kLicense);
}

}  // namespace deeprm
