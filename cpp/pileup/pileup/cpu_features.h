/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_CPU_FEATURES_H_
#define DEEPRM_PILEUP_CPU_FEATURES_H_

namespace deeprm {

// Whether this CPU has AVX512 F, DQ, VL and BW, which the SVML routines use.
bool cpu_has_avx512();

// Whether this CPU has AVX2 and FMA, which the program is compiled for.
bool cpu_has_avx2_fma();

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_CPU_FEATURES_H_
