/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include "pileup/cpu_features.h"

namespace deeprm {

// The compiler's CPU query is the one nonstandard call of this file. It is
// not in svml_call.cc: that file is compiled for AVX512 and must not run on
// a CPU without it.
bool cpu_has_avx512() {
  return __builtin_cpu_supports("avx512f") &&
         __builtin_cpu_supports("avx512dq") &&
         __builtin_cpu_supports("avx512vl") &&
         __builtin_cpu_supports("avx512bw");
}

bool cpu_has_avx2_fma() {
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}

}  // namespace deeprm
