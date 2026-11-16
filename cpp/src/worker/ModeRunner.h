/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of DeepRM. Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
 * International License; see LICENSE.md at the top of the DeepRM repository.
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

