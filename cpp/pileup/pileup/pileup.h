/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_PILEUP_H_
#define DEEPRM_PILEUP_PILEUP_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "pileup/options.h"
#include "pileup/parallel.h"

namespace deeprm {

// The worker group of each of `files` per-read files: the groups are the
// files of the `groups` workers of DeepRM 1.2 (np.array_split).
std::vector<std::int32_t> worker_groups(std::size_t files, std::size_t groups);

// The per-read files of `input_dir` (*.npz, as glob lists them), sorted.
std::vector<std::string> list_input_files(const std::string& input_dir);

// Runs the pileup: writes the site-level files into options.output.
Status run_pileup(const Options& options);

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_PILEUP_H_
