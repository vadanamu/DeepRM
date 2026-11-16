/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
// The algorithm and the tables below are those of svml/svml_z0_log2_s_la.s and
// svml/svml_z0_log10_s_la.s (Copyright (C) 2023 Intel Corporation;
// BSD-3-Clause: see svml/LICENSE and THIRD_PARTY_NOTICES).
#include "pileup/svml_port.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string_view>

#include "pileup/cpu_features.h"
#include "pileup/svml_call.h"

namespace deeprm {
namespace {

using Table = std::array<std::uint32_t, 16>;

// Coefficients of the polynomial per interval of the mantissa, highest degree
// first; the index is the top four bits of the mantissa.
constexpr std::array<Table, 4> kLog2 = {{
    {0xbea77e4aU, 0xbe8aae3dU, 0xbe67fe32U, 0xbe43d1b6U, 0xbe26a589U,
     0xbe0ee09bU, 0xbdf6a8a1U, 0xbdd63b49U, 0xbf584e51U, 0xbf3e80a1U,
     0xbf2892f0U, 0xbf15d377U, 0xbf05b525U, 0xbeef8e30U, 0xbed75c8fU,
     0xbec24184U},
    {0x3ef5910cU, 0x3ef045a1U, 0x3ee7d87eU, 0x3eddbb84U, 0x3ed2d6dfU,
     0x3ec7bbd2U, 0x3ebcc42fU, 0x3eb22616U, 0x3e8f3399U, 0x3eb1223eU,
     0x3ec9db4aU, 0x3edb7a09U, 0x3ee79a1aU, 0x3eef77cbU, 0x3ef407a4U,
     0x3ef607b4U},
    {0xbf38a934U, 0xbf387de6U, 0xbf37f6f0U, 0xbf37048bU, 0xbf35a88aU,
     0xbf33ed04U, 0xbf31df56U, 0xbf2f8d82U, 0xbf416814U, 0xbf3daf58U,
     0xbf3b5c08U, 0xbf39fa2aU, 0xbf393713U, 0xbf38d7e1U, 0xbf38b2cdU,
     0xbf38aa62U},
    {0x3fb8aa3bU, 0x3fb8a9c0U, 0x3fb8a6e8U, 0x3fb89f4eU, 0x3fb890cbU,
     0x3fb879b1U, 0x3fb858d8U, 0x3fb82d90U, 0x3fb8655eU, 0x3fb8883aU,
     0x3fb89aeaU, 0x3fb8a42fU, 0x3fb8a848U, 0x3fb8a9c9U, 0x3fb8aa2fU,
     0x3fb8aa3bU},
}};

constexpr std::array<Table, 4> kLog10 = {{
    {0xbdc9ae9bU, 0xbda6fcf4U, 0xbd8bac76U, 0xbd6bca30U, 0xbd48a99bU,
     0xbd2c0a9fU, 0xbd1480dbU, 0xbd00faf2U, 0xbe823aa9U, 0xbe656348U,
     0xbe4afbb9U, 0xbe346895U, 0xbe20ffffU, 0xbe103a0bU, 0xbe01a91cU,
     0xbde9e84eU},
    {0x3e13d888U, 0x3e10a87cU, 0x3e0b95c3U, 0x3e057f0bU, 0x3dfde038U,
     0x3df080d9U, 0x3de34c1eU, 0x3dd68333U, 0x3dac6e8eU, 0x3dd54a51U,
     0x3df30f40U, 0x3e04235dU, 0x3e0b7033U, 0x3e102c90U, 0x3e12ebadU,
     0x3e141ff8U},
    {0xbe5e5a9bU, 0xbe5e2677U, 0xbe5d83f5U, 0xbe5c6016U, 0xbe5abd0bU,
     0xbe58a6fdU, 0xbe562e02U, 0xbe5362f8U, 0xbe68e27cU, 0xbe646747U,
     0xbe619a73U, 0xbe5ff05aU, 0xbe5f0570U, 0xbe5e92d0U, 0xbe5e662bU,
     0xbe5e5c08U},
    {0x3ede5bd8U, 0x3ede5b45U, 0x3ede57d8U, 0x3ede4eb1U, 0x3ede3d37U,
     0x3ede2166U, 0x3eddf9d9U, 0x3eddc5bbU, 0x3ede08edU, 0x3ede32e7U,
     0x3ede4967U, 0x3ede5490U, 0x3ede597fU, 0x3ede5b50U, 0x3ede5bcaU,
     0x3ede5bd9U},
}};

// log10(2), by which log10 multiplies the exponent.
constexpr std::uint32_t kLog10Of2 = 0x3e9a209bU;

constexpr std::uint32_t kFractionMask = 0x007fffffU;
constexpr std::uint32_t kQuietBit = 0x00400000U;
constexpr std::uint32_t kDefaultNan = 0xffc00000U;

float as_float(std::uint32_t bits) { return std::bit_cast<float>(bits); }

// A positive finite x as SVML splits it (VGETMANT to [0.75, 1.5), VGETEXP).
struct Split {
  float reduced;       // the mantissa minus 1
  float exponent;      // x = mantissa * 2^exponent
  std::uint32_t index;  // the top four bits of the mantissa
};

Split split(std::uint32_t bits) {
  std::uint32_t fraction = bits & kFractionMask;
  std::int32_t exponent = static_cast<std::int32_t>(bits >> 23) - 127;
  if ((bits >> 23) == 0) {
    // a denormal: both instructions normalize it first
    const int shift = std::countl_zero(fraction) - 8;
    fraction = (fraction << shift) & kFractionMask;
    exponent = -126 - shift;
  }
  // a mantissa of 1.5 or more is halved, and the exponent takes the 1
  const std::uint32_t upper = fraction >> 22;
  const float mantissa = as_float(((127U - upper) << 23) | fraction);
  return Split{.reduced = mantissa - 1.0F,
               .exponent = static_cast<float>(
                   exponent + static_cast<std::int32_t>(upper)),
               .index = fraction >> 19};
}

// The result for a NaN, a zero, an infinity or a negative x; false for the
// other inputs.
bool special(std::uint32_t bits, float& result) {
  const std::uint32_t magnitude = bits & 0x7fffffffU;
  if (magnitude > 0x7f800000U) {
    result = as_float(bits | kQuietBit);  // the NaN itself, made quiet
  } else if (magnitude == 0) {
    result = -std::numeric_limits<float>::infinity();
  } else if ((bits >> 31) != 0) {
    result = as_float(kDefaultNan);  // what 0 / 0 gives on x86
  } else if (magnitude == 0x7f800000U) {
    result = std::numeric_limits<float>::infinity();
  } else {
    return false;
  }
  return true;
}

// The polynomial in `r` without its constant term, which the callers add in
// the last fused step.
float polynomial(const std::array<Table, 4>& table, std::uint32_t index,
                 float r) {
  float p = std::fma(as_float(table[0][index]), r, as_float(table[1][index]));
  p = std::fma(p, r, as_float(table[2][index]));
  return std::fma(p, r, as_float(table[3][index]));
}

bool decide_avx512() {
  const char* off = std::getenv("DEEPRM_PILEUP_NO_AVX512");
  if (off != nullptr && *off != '\0' && std::string_view(off) != "0") {
    return false;
  }
  return cpu_has_avx512();
}

}  // namespace

float ported_log2(float x) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(x);
  if (float result = 0.0F; special(bits, result)) return result;
  const Split s = split(bits);
  return std::fma(polynomial(kLog2, s.index, s.reduced), s.reduced,
                  s.exponent);
}

float ported_log10(float x) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(x);
  if (float result = 0.0F; special(bits, result)) return result;
  const Split s = split(bits);
  const float exponent_part = s.exponent * as_float(kLog10Of2);
  return std::fma(polynomial(kLog10, s.index, s.reduced), s.reduced,
                  exponent_part);
}

bool svml_uses_avx512() {
  static const bool use = decide_avx512();
  return use;
}

void svml_log2(std::span<const float> in, std::span<float> out) {
  if (svml_uses_avx512()) return svml_log2_avx512(in, out);
  for (std::size_t i = 0; i < in.size(); ++i) out[i] = ported_log2(in[i]);
}

void svml_log10(std::span<const float> in, std::span<float> out) {
  if (svml_uses_avx512()) return svml_log10_avx512(in, out);
  for (std::size_t i = 0; i < in.size(); ++i) out[i] = ported_log10(in[i]);
}

}  // namespace deeprm
