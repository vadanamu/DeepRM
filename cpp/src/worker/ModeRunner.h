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

#include <memory>

#include "args/ArgumentParser.h"
#include "pod5/Pod5Catalog.h"
#include "worker/WorkerAssignment.h"

namespace deeprm {
  // One processing mode, end to end. main() drives every mode the same way:
  //   runner->start_bam()        begin reading the BAM side; it needs nothing
  //                              from POD5, so it overlaps the catalog scan
  //   catalog.scan_async / join
  //   runner->assign(catalog)    the partition this mode wants
  //   runner->run(assignment)    the workers; false = output not trustworthy
  class ModeRunner {
  public:
    virtual ~ModeRunner() = default;
    virtual void start_bam() = 0;
    virtual WorkerAssignment assign(Pod5Catalog& catalog) = 0;
    virtual bool run(WorkerAssignment& assignment) = 0;

    // Picks the runner from the BAM input form (see Arguments::streaming).
    static std::unique_ptr<ModeRunner> create(const Arguments& args);
  };
}  // namespace deeprm

