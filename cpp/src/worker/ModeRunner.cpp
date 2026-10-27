/**
 * Copyright (C) 2026 Genome4me Incorporated - All Rights Reserved.
 * This software, including its source code, embedded concepts, and associated
 * documentation, is proprietary to Genome4me Incorporated and is protected
 * under trade secret and copyright law. Unauthorized use, copying, modification,
 * distribution, or disclosure to third parties, in whole or in part, is
 * strictly prohibited unless prior written permission is granted by Genome4me
 * Incorporated. Any such unauthorized actions constitute an infringement of
 * the intellectual property rights of Genome4me Incorporated. For licensing
 * inquiries or permissions, please contact Genome4me Incorporated.
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
