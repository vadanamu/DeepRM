/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_NUMPY_MATH_H_
#define DEEPRM_PILEUP_NUMPY_MATH_H_

#include <cstdint>

namespace deeprm {

// exp(x) in float32, bit for bit as NumPy's AVX512 loop computes it.
float numpy_exp(float x);

// log(p) - log1p(-p) of the clipped probability, rounded to float32, from
// the C library alone. It differs from NumPy's for the few inputs listed in
// logit_exceptions.h.
float calibration_logit_libm(float pred);

// The same value as NumPy computes it.
float calibration_logit(float pred);

// The per-read probability after the v1.2 model's fixed calibration, as
// _calibrated_probability() of the Python pileup returns it.
float calibrated_probability(float pred);

// The smallest prediction that the per-read terms calibrate (see
// PerReadTermsCalculator::compute_calibrated): only these go through the C
// library's log and log1p.
inline constexpr float kCalibratedFrom = 0.99F;

// A checksum of calibrated_probability() over every float32 from
// kCalibratedFrom to 1.
std::uint64_t calibration_checksum();

// Whether that checksum is the one recorded from a build whose calibrated
// probabilities were all compared with NumPy's (tests/check_terms.py). False
// tells that this C library computes log or log1p differently, so that
// outputs may differ from DeepRM's in a last bit.
bool calibration_is_verified();

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_NUMPY_MATH_H_
