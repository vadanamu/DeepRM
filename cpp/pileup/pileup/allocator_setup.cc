/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include "pileup/allocator_setup.h"

#include <malloc.h>

namespace deeprm {
namespace {

// Blocks of this size or more get a mapping of their own.
constexpr int kMapThreshold = 1 << 20;
// An arena that runs out grows by this much.
constexpr int kHeapStep = 8 << 20;

}  // namespace

// glibc raises its mmap threshold (to as much as 32 MB) whenever a mapped
// block is freed; smaller blocks then come from the heaps of the threads'
// arenas, which keep freed memory. The site tables, their output and the
// genomic rows are allocated and freed by many threads in blocks of 1-100
// MB, and the process then held about twice what it used. A fixed threshold
// keeps those blocks in mappings, which are returned when freed.
//
// The per-read rows of the modBAM are some 13 GB in small blocks that the
// threads allocate file by file. Their arenas grew 128 KB at a time, and
// every step holds up all the threads of the process (it changes the memory
// map); with the larger step the files are read twice as fast.
void set_up_allocator() {
  mallopt(M_MMAP_THRESHOLD, kMapThreshold);
  mallopt(M_TOP_PAD, kHeapStep);
}

}  // namespace deeprm
