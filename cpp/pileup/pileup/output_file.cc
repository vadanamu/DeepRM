/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include "pileup/output_file.h"

#include <fcntl.h>
#include <unistd.h>

#include <cstddef>

namespace deeprm {

OutputFile::~OutputFile() {
  if (descriptor_ >= 0) ::close(descriptor_);
}

Status OutputFile::open(const std::string& path) {
  path_ = path;
  descriptor_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (descriptor_ < 0) return std::unexpected("cannot create " + path);
  return {};
}

Status OutputFile::write_at(std::uint64_t offset,
                            std::span<const std::uint8_t> bytes) {
  std::size_t done = 0;
  while (done < bytes.size()) {
    const ssize_t written =
        ::pwrite(descriptor_, bytes.data() + done, bytes.size() - done,
                 static_cast<off_t>(offset + done));
    if (written <= 0) return std::unexpected("cannot write " + path_);
    done += static_cast<std::size_t>(written);
  }
  return {};
}

Status OutputFile::close() {
  const int result = ::close(descriptor_);
  descriptor_ = -1;
  if (result != 0) return std::unexpected("cannot close " + path_);
  return {};
}

}  // namespace deeprm
