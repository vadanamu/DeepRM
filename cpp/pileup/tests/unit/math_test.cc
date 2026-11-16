/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "pileup/cpu_features.h"
#include "pileup/numpy_math.h"
#include "pileup/per_read_terms.h"
#include "pileup/svml_call.h"
#include "pileup/svml_port.h"
#include "tests/unit/unit_test.h"

// The expected bit patterns were recorded from NumPy 2.4.6 (AVX512) and the
// Python pileup's _per_read_terms(pred, 0.98, 1e-30, True).
namespace deeprm::test {
namespace {

const std::vector<float> kPred = {0.0F,   0.25F, 0.5F,  0.75F, 0.979F,
                                  0.98F,  0.99F, 0.999F, 1.0F};

void exp_matches_numpy(Checker& t) {
  t.equal(bits(numpy_exp(-1.5F)), 0x3e647c3bU);
  t.equal(bits(numpy_exp(3.25F)), 0x41ce529eU);
  t.equal(bits(numpy_exp(-20.0F)), 0x310da433U);
  t.equal(bits(numpy_exp(0.5F)), 0x3fd3094cU);
}

void exp_handles_the_ends(Checker& t) {
  t.equal(numpy_exp(0.0F), 1.0F);
  t.equal(numpy_exp(89.0F), std::numeric_limits<float>::infinity());
  t.equal(numpy_exp(-104.0F), 0.0F);
  t.check(std::isnan(numpy_exp(std::numeric_limits<float>::quiet_NaN())),
          "exp(NaN) is NaN");
}

void svml_handles_any_length(Checker& t) {
  // 37 values: two full registers and a tail; every lane gives log2(8) = 3
  const std::vector<float> in(37, 8.0F);
  std::vector<float> out(37, 0.0F);
  svml_log2(in, out);
  for (const float value : out) t.equal(value, 3.0F);
  // a tail alone; the values after it are not written
  svml_log10(std::vector<float>(5, 1.0F), std::span<float>(out).first(5));
  t.equal(out[4], 0.0F);
  t.equal(out[5], 3.0F);
}

// {input, np.log2, np.log10}: bit patterns from NumPy 2.4.6 on AVX512.
struct LogCase {
  std::uint32_t in;
  std::uint32_t log2;
  std::uint32_t log10;
};

constexpr LogCase kLogCases[] = {
    {0x3f7ae148U, 0xbceec442U, 0xbc0fc079U},  // 0.98
    {0x3f000000U, 0xbf800000U, 0xbe9a209bU},  // 0.5
    {0x3e800000U, 0xc0000000U, 0xbf1a209bU},  // 0.25
    {0x3f400000U, 0xbed47fccU, 0xbdffdfe2U},  // 0.75
    {0x3f7fffffU, 0xb3b8aa3bU, 0xb2de5bd9U},  // the float below 1
    {0x3f800000U, 0x00000000U, 0x00000000U},  // 1
    {0x00000001U, 0xc3150000U, 0xc23369f4U},  // the smallest denormal
    {0x007fffffU, 0xc2fc0000U, 0xc217b819U},  // the largest denormal
    {0x00800000U, 0xc2fc0000U, 0xc217b819U},  // the smallest normal float
    {0x0da24260U, 0xc2c750d1U, 0xc1f00001U},  // 1e-30
    {0x3ca3d70aU, 0xc0b49a78U, 0xbfd977daU},  // 0.02
    {0x3fc00000U, 0x3f15c01aU, 0x3e345145U},  // 1.5
    {0x3fbfffffU, 0x3f15c019U, 0x3e345142U},  // the float below 1.5
    {0x41200000U, 0x40549a78U, 0x3f800000U},  // 10
    {0x7f7fffffU, 0x43000000U, 0x421a209bU},  // the largest float
    {0x00000000U, 0xff800000U, 0xff800000U},  // 0
    {0x80000000U, 0xff800000U, 0xff800000U},  // -0
    {0x7f800000U, 0x7f800000U, 0x7f800000U},  // inf
    {0xbf800000U, 0xffc00000U, 0xffc00000U},  // -1
    {0xff800000U, 0xffc00000U, 0xffc00000U},  // -inf
    {0x7fc00000U, 0x7fc00000U, 0x7fc00000U},  // a quiet NaN
    {0x7f800001U, 0x7fc00001U, 0x7fc00001U},  // a signaling NaN
    {0xffc12345U, 0xffc12345U, 0xffc12345U},  // a negative NaN
};

void ported_logs_match_numpy(Checker& t) {
  for (const LogCase& c : kLogCases) {
    const float x = std::bit_cast<float>(c.in);
    t.equal(bits(ported_log2(x)), c.log2);
    t.equal(bits(ported_log10(x)), c.log10);
  }
}

void svml_logs_match_numpy(Checker& t) {
  // through the functions the pileup calls, whichever path this CPU takes;
  // 23 values: one full register and a tail on the AVX512 path
  std::vector<float> in;
  for (const LogCase& c : kLogCases) in.push_back(std::bit_cast<float>(c.in));
  std::vector<float> log2(in.size());
  std::vector<float> log10(in.size());
  svml_log2(in, log2);
  svml_log10(in, log10);
  for (std::size_t i = 0; i < in.size(); ++i) {
    t.equal(bits(log2[i]), kLogCases[i].log2);
    t.equal(bits(log10[i]), kLogCases[i].log10);
  }
}

void ported_logs_match_svml(Checker& t) {
  if (!cpu_has_avx512()) return;  // nothing to compare with on this CPU
  // every 4,099th bit pattern: about a million floats over the whole range
  std::vector<float> in;
  for (std::uint64_t pattern = 0; pattern < (std::uint64_t{1} << 32);
       pattern += 4099) {
    in.push_back(std::bit_cast<float>(static_cast<std::uint32_t>(pattern)));
  }
  std::vector<float> log2(in.size());
  std::vector<float> log10(in.size());
  svml_log2_avx512(in, log2);
  svml_log10_avx512(in, log10);
  std::size_t differences = 0;
  for (std::size_t i = 0; i < in.size(); ++i) {
    differences += bits(ported_log2(in[i])) != bits(log2[i]);
    differences += bits(ported_log10(in[i])) != bits(log10[i]);
  }
  t.equal(differences, std::size_t{0});
}

void calibration_matches_numpy(Checker& t) {
  const std::uint32_t expected[] = {0x303a6535, 0x3e9a7338, 0x3f000000,
                                    0x3f32c665, 0x3f731753, 0x3f738cbd,
                                    0x3f78934c, 0x3f7eb305, 0x3f800000};
  for (std::size_t i = 0; i < kPred.size(); ++i) {
    t.equal(bits(calibrated_probability(kPred[i])), expected[i]);
  }
}

void terms_match_the_python_pileup(Checker& t) {
  PerReadTermsCalculator calculator(
      TermOptions{.threshold = 0.98, .epsilon = 1e-30, .calibrated = true});
  const PerReadTerms& terms = calculator.compute(kPred);
  const std::uint8_t count[] = {0, 0, 0, 0, 0, 1, 1, 1, 1};
  const double logsum[] = {0.0, -0.0, -0.0, -0.0, -0.0, -0x1.b2efbap+0,
                           -0x1.000006p+1, -0x1.80003p+1, -0x1.e00002p+4};
  const std::uint32_t kl_neg[] = {0x3f800000, 0x3e41404e, 0, 0, 0, 0, 0, 0, 0};
  const std::uint32_t kl_pos[] = {0, 0, 0, 0x3e41404e, 0x3f5a5cf4, 0x3f5bca90,
                                  0x3f6b5126, 0x3f7d1464, 0x3f800000};
  const double calib[] = {0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
                          -0x1.25a3ccp+1, -0x1.e00002p+4};
  for (std::size_t i = 0; i < kPred.size(); ++i) {
    t.equal(terms.count_pos[i], count[i]);
    t.equal(terms.count_pos_rule[i], count[i]);
    t.equal(bits(terms.logsum_1_p_pos[i]), bits(logsum[i]));
    t.equal(bits(terms.logsum_1_p_pos_rule[i]), bits(logsum[i]));
    t.equal(bits(terms.kl_div_neg[i]), kl_neg[i]);
    t.equal(bits(terms.kl_div_pos[i]), kl_pos[i]);
    t.equal(bits(terms.logsum_1_p_pos_calib[i]), bits(calib[i]));
  }
}

void uncalibrated_terms_use_pred(Checker& t) {
  PerReadTermsCalculator calculator(
      TermOptions{.threshold = 0.98, .epsilon = 1e-30, .calibrated = false});
  const PerReadTerms& terms = calculator.compute(kPred);
  for (std::size_t i = 0; i < kPred.size(); ++i) {
    t.equal(bits(terms.logsum_1_p_pos_calib[i]),
            bits(terms.logsum_1_p_pos_rule[i]));
  }
}

// Fails where the C library's log or log1p differs from glibc 2.28's, with
// which every calibrated probability was compared with NumPy's.
void calibration_is_the_verified_one(Checker& t) {
  t.check(calibration_is_verified(), "calibration checksum");
}

}  // namespace

std::vector<TestCase> math_tests() {
  return {{"calibration is the one compared with NumPy",
           calibration_is_the_verified_one},
          {"exp matches NumPy", exp_matches_numpy},
          {"exp handles the ends of its range", exp_handles_the_ends},
          {"SVML calls handle any length", svml_handles_any_length},
          {"ported log2 and log10 match NumPy", ported_logs_match_numpy},
          {"log2 and log10 match NumPy on this CPU", svml_logs_match_numpy},
          {"ported log2 and log10 match SVML", ported_logs_match_svml},
          {"calibration matches NumPy", calibration_matches_numpy},
          {"per-read terms match the Python pileup",
           terms_match_the_python_pileup},
          {"uncalibrated terms use pred", uncalibrated_terms_use_pred}};
}

}  // namespace deeprm::test
