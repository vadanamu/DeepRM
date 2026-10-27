/***************************************************************************************************
 *
 * Copyright (C) 2025-2026 Genome4me Incorporated - All Rights Reserved.
 *
 * This software, including its source code, embedded concepts, and associated
 * documentation, is proprietary to Genome4me Incorporated and is protected
 * under trade secret and copyright law. Unauthorized use, copying, modification,
 * distribution, or disclosure to third parties, in whole or in part, is
 * strictly prohibited unless prior written permission is granted by Genome4me
 * Incorporated. Any such unauthorized actions constitute an infringement of
 * the intellectual property rights of Genome4me Incorporated. For licensing
 * inquiries or permissions, please contact Genome4me Incorporated.
 *
 **************************************************************************************************/

#pragma once

#include <memory>
#include <vector>

#include "bam/BamReader.h"
#include "npz/NpzWriter.h"
#include "pod5/Pod5Reader.h"
#include "utils/Utils.h"

using namespace std;

namespace deeprm {
  class RecordMerger {
  private:
    NormalizationFactors norm_factors;
    int cb_len;
    int kmer_len;
    int max_token_len;
    int sampling;
    int dwell_shift;
    int sig_window;
    uint64_t label_div;
    // Site handling, as in the Python preprocessing: 0 = drop out-of-bounds contexts (none, default),
    // 1 = zero padding, 2 = edge padding. Trim bounds come from BamRecord.
    int pad_mode;
    bool use_trimming;

  public:
    RecordMerger(const NormalizationFactors& nf, int cb_len, int kmer_len,
                 int max_token_len, int sampling, int dwell_shift, int sig_window,
                 uint64_t label_div, const string& padding_mode, bool use_trimming);

    // Stateless past construction: it holds tokenization settings, never
    // records. Each accepted site is written straight into the writer's
    // preallocated row (acquire_row / commit_row) -- no per-record object is
    // created, and nothing accumulates outside the writer's fixed chunk
    // arrays (the per-site signal block is a transient). Both execution modes call this,
    // which is what keeps their memory behaviour identical for the same
    // options.
    //
    // Returns false when the read's POD5 signal could not be read. That
    // is a run failure for the caller to report.
    bool process_merged_record_with_meta(const BamRecord& bam_rec,
                                         const Pod5RecordMeta& pod5_meta,
                                         Pod5BatchCache& batch_cache,
                                         NpzWriter& writer) const;
  };
} // namespace deeprm
