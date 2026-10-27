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

#pragma once

#include "sam/SamDispatcher.h"
#include "worker/ModeRunner.h"

namespace deeprm {
  // Streaming mode (-b - or --bam-cmd). The dispatcher starts reading the
  // stream during the POD5 scan; once reads are assigned, one StreamWorker
  // per worker consumes its queue until the stream ends.
  class StreamRunner : public ModeRunner {
  public:
    explicit StreamRunner(const Arguments& args) : args(args), dispatcher(args) {}
    void start_bam() override;
    WorkerAssignment assign(Pod5Catalog& catalog) override;
    bool run(WorkerAssignment& assignment) override;

  private:
    // Reads are handed to workers in blocks of this many consecutive POD5
    // ranks, round-robin. Big enough that a block almost always stays inside
    // one file, so the worker's single reader serves the block; small enough that
    // any live window wider than kRankBlock * workers reaches every worker.
    static constexpr size_t kRankBlock = 32;

    const Arguments& args;
    SamDispatcher dispatcher;
  };
}  // namespace deeprm

