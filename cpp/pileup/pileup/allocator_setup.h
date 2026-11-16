/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_ALLOCATOR_SETUP_H_
#define DEEPRM_PILEUP_ALLOCATOR_SETUP_H_

namespace deeprm {

// Makes the C library give the memory of large freed blocks back to the
// system at once. Call before the first large allocation.
void set_up_allocator();

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_ALLOCATOR_SETUP_H_
