/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_SVML_CALL_H_
#define DEEPRM_PILEUP_SVML_CALL_H_

#include <span>

namespace deeprm {

// log2 and log10 of each float by the SVML routines NumPy uses on AVX512.
// `out` has the size of `in` and may be the same memory. Call them only on a
// CPU with AVX512 (cpu_has_avx512()); svml_port.h has the functions to use.
void svml_log2_avx512(std::span<const float> in, std::span<float> out);
void svml_log10_avx512(std::span<const float> in, std::span<float> out);

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_SVML_CALL_H_
