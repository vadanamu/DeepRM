/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of DeepRM. Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
 * International License; see LICENSE.md at the top of the DeepRM repository.
 */

#include "worker/ModeRunner.h"

#include "worker/FileRunner.h"
#include "worker/StreamRunner.h"

namespace deeprm {
  std::unique_ptr<ModeRunner> ModeRunner::create(const Arguments& args)
  {
    if (args.streaming)
      return std::make_unique<StreamRunner>(args);
    return std::make_unique<FileRunner>(args);
  }
}  // namespace deeprm
