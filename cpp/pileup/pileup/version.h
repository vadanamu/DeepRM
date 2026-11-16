/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_VERSION_H_
#define DEEPRM_PILEUP_VERSION_H_

#include <string>

namespace deeprm {

// The DeepRM release whose pileup this program reproduces: what -v prints.
std::string version();

// What -vv prints: the program, its version with the build time and the git
// commit ("*" after it: built from changed sources), and the contact.
std::string version_text();

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_VERSION_H_
