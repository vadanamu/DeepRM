/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_RUN_INFO_H_
#define DEEPRM_PILEUP_RUN_INFO_H_

#include <string>
#include <string_view>

namespace deeprm {

// Whether deeprm_run.json (given as its text) says that the per-read files
// come from the bundled v1.2 weights (inputs_calibrated of the Python
// pileup).
bool run_info_is_calibrated(std::string_view json);

// The same for the per-read folder `input_dir`; false, with a warning on
// stderr, when the folder has no deeprm_run.json.
bool inputs_calibrated(const std::string& input_dir);

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_RUN_INFO_H_
