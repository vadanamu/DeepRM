/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_MATH_TOOLS_H_
#define DEEPRM_PILEUP_MATH_TOOLS_H_

#include <span>
#include <string_view>

namespace deeprm {

// Developer commands that write raw arrays to stdout for the checks against
// NumPy in tests/. Each takes a range of float32 bit patterns as its input.
//   logit-range <first_bits> <count>
//   terms-range <first_bits> <count> <threshold> <epsilon> <calibrated 0|1>
// And one that needs AVX512 and prints its result:
//   svml-port-check [threads]: ported_log2 and ported_log10 against the SVML
//   routines for every float32 bit pattern; fails if any result differs.
// Returns the exit code, or -1 when `command` is not one of them.
int run_math_tool(std::string_view command,
                  std::span<const std::string_view> arguments);

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_MATH_TOOLS_H_
