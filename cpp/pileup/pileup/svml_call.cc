/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include "pileup/svml_call.h"

#include <immintrin.h>

#include <cstddef>

extern "C" {
// Defined in svml/*.s; the names and the register interface are SVML's.
__m512 __svml_log2f16(__m512 x);
__m512 __svml_log10f16(__m512 x);
}

namespace deeprm {
namespace {

constexpr std::size_t kLanes = 16;

template <typename Function>
void apply(Function function, std::span<const float> in, std::span<float> out) {
  const std::size_t size = in.size();
  std::size_t i = 0;
  for (; i + kLanes <= size; i += kLanes) {
    _mm512_storeu_ps(out.data() + i, function(_mm512_loadu_ps(in.data() + i)));
  }
  if (i < size) {
    // The unused lanes hold 1, as in NumPy's loop.
    const __mmask16 mask = static_cast<__mmask16>((1U << (size - i)) - 1U);
    const __m512 x =
        _mm512_mask_loadu_ps(_mm512_set1_ps(1.0F), mask, in.data() + i);
    _mm512_mask_storeu_ps(out.data() + i, mask, function(x));
  }
}

}  // namespace

void svml_log2_avx512(std::span<const float> in, std::span<float> out) {
  apply([](__m512 x) { return __svml_log2f16(x); }, in, out);
}

void svml_log10_avx512(std::span<const float> in, std::span<float> out) {
  apply([](__m512 x) { return __svml_log10f16(x); }, in, out);
}

}  // namespace deeprm
