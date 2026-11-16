/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
// numpy_exp() follows NumPy's float32 exp for AVX512, with its constants
// (Copyright (c) 2005-2025, NumPy Developers; BSD-3-Clause: see
// THIRD_PARTY_NOTICES).
#include "pileup/numpy_math.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include "pileup/logit_exceptions.h"

namespace deeprm {
namespace {

// Constants of NumPy's float32 exp (npy_simd_data.h).
constexpr float kExpMax = 88.72283935546875F;
constexpr float kExpMin = -103.97208404541015625F;
constexpr float kLog2E = 1.442695040888963407359924681001892137F;
constexpr float kRintMagic = 0x1.800000p+23F;
constexpr float kLogE2High = -6.93145752e-1F;
constexpr float kLogE2Low = -1.42860677e-6F;
constexpr float kExpP0 = 9.999999999980870924916e-01F;
constexpr float kExpP1 = 7.257664613233124478488e-01F;
constexpr float kExpP2 = 2.473615434895520810817e-01F;
constexpr float kExpP3 = 5.114512081637298353406e-02F;
constexpr float kExpP4 = 6.757896990527504603057e-03F;
constexpr float kExpP5 = 5.082762527590693718096e-04F;
constexpr float kExpQ0 = 1.000000000000000000000e+00F;
constexpr float kExpQ1 = -2.742335390411667452936e-01F;
constexpr float kExpQ2 = 2.159509375685829852307e-02F;

// Fixed part of the v1.2 model (_CALIBRATION of the Python pileup).
constexpr float kCalibration = static_cast<float>(1.3088);
constexpr double kProbabilityMin = 1e-12;
constexpr double kProbabilityMax = 1 - 1e-12;
// calibration_checksum() of the build compared with NumPy 2.4.6 (glibc 2.28).
constexpr std::uint64_t kCalibrationChecksum =
    0x449c25fc4d5c7982ULL;

}  // namespace

float numpy_exp(float x) {
  if (std::isnan(x)) return std::numeric_limits<float>::quiet_NaN();
  if (x >= kExpMax) return std::numeric_limits<float>::infinity();
  if (x <= kExpMin) return 0.0F;
  // Round x / ln(2) to the nearest integer by adding and removing 1.5 * 2^23.
  float quadrant = x * kLog2E;
  quadrant = quadrant + kRintMagic;
  quadrant = quadrant - kRintMagic;
  // Cody-Waite range reduction; the third step is NumPy's (it adds zero).
  float reduced = std::fma(quadrant, kLogE2High, x);
  reduced = std::fma(quadrant, kLogE2Low, reduced);
  reduced = std::fma(quadrant, 0.0F, reduced);
  float numerator = std::fma(kExpP5, reduced, kExpP4);
  numerator = std::fma(numerator, reduced, kExpP3);
  numerator = std::fma(numerator, reduced, kExpP2);
  numerator = std::fma(numerator, reduced, kExpP1);
  numerator = std::fma(numerator, reduced, kExpP0);
  float denominator = std::fma(kExpQ2, reduced, kExpQ1);
  denominator = std::fma(denominator, reduced, kExpQ0);
  return std::scalbn(numerator / denominator, static_cast<int>(quadrant));
}

float calibration_logit_libm(float pred) {
  const double p = std::min(
      std::max(static_cast<double>(pred), kProbabilityMin), kProbabilityMax);
  return static_cast<float>(std::log(p) - std::log1p(-p));
}

float calibration_logit(float pred) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(pred);
  const auto found = std::ranges::lower_bound(
      kLogitExceptions, bits, {}, &LogitException::pred_bits);
  if (found != kLogitExceptions.end() && found->pred_bits == bits) {
    return std::bit_cast<float>(found->logit_bits);
  }
  return calibration_logit_libm(pred);
}

float calibrated_probability(float pred) {
  const float z = calibration_logit(pred) / kCalibration;
  if (z >= 0.0F) return 1.0F / (1.0F + numpy_exp(-z));
  const float e = numpy_exp(z);
  return e / (1.0F + e);
}

std::uint64_t calibration_checksum() {
  std::uint64_t checksum = 0;
  const std::uint32_t last = std::bit_cast<std::uint32_t>(1.0F);
  for (std::uint32_t bits = std::bit_cast<std::uint32_t>(kCalibratedFrom);
       bits <= last; ++bits) {
    const std::uint32_t result = std::bit_cast<std::uint32_t>(
        calibrated_probability(std::bit_cast<float>(bits)));
    checksum = (checksum ^ result) * 0x100000001B3ULL;  // FNV-1a, per value
  }
  return checksum;
}

bool calibration_is_verified() {
  return calibration_checksum() == kCalibrationChecksum;
}

}  // namespace deeprm
