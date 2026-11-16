/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_OUTPUT_FILE_H_
#define DEEPRM_PILEUP_OUTPUT_FILE_H_

#include <cstdint>
#include <span>
#include <string>

#include "pileup/parallel.h"

namespace deeprm {

// A file written at given offsets; write_at() may be called from several
// threads at once.
class OutputFile {
 public:
  OutputFile() = default;
  ~OutputFile();
  OutputFile(const OutputFile&) = delete;
  OutputFile& operator=(const OutputFile&) = delete;

  Status open(const std::string& path);
  Status write_at(std::uint64_t offset, std::span<const std::uint8_t> bytes);
  Status close();

 private:
  int descriptor_ = -1;
  std::string path_;
};

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_OUTPUT_FILE_H_
