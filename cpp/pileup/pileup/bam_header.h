/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_BAM_HEADER_H_
#define DEEPRM_PILEUP_BAM_HEADER_H_

#include <expected>
#include <string>
#include <vector>

namespace deeprm {

// The reference names of the BAM file at `path`, in header order.
std::expected<std::vector<std::string>, std::string> read_reference_names(
    const std::string& path);

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_BAM_HEADER_H_
