/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_SVML_PORT_H_
#define DEEPRM_PILEUP_SVML_PORT_H_

#include <span>

namespace deeprm {

// log2 and log10 of each float as NumPy computes them on AVX512 (SVML).
// `out` has the size of `in` and may be the same memory. On a CPU with AVX512
// the SVML routines run; elsewhere their port below, which gives the same bits.
void svml_log2(std::span<const float> in, std::span<float> out);
void svml_log10(std::span<const float> in, std::span<float> out);

// Whether svml_log2 and svml_log10 run the SVML routines. False without
// AVX512, and when DEEPRM_PILEUP_NO_AVX512 is set to anything but 0 (to test
// the port on a CPU that has AVX512).
bool svml_uses_avx512();

// The SVML routines in standard C++: the same operations in the same order,
// one value at a time. tests/ compares them with SVML for every float32.
float ported_log2(float x);
float ported_log10(float x);

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_SVML_PORT_H_
