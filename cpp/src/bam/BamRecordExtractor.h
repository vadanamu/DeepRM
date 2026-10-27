/***************************************************************************************************
 *
 * Copyright (C) 2026 Genome4me Incorporated - All Rights Reserved.
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

#include <atomic>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <htslib/sam.h>

#include "bam/BamRecord.h"

using namespace std;

namespace deeprm {
  // Single implementation of bam1_t -> BamRecord extraction, shared by both
  // execution modes: FileWorker (file mode, -b FILE, on the record re-read by
  // voffset) and StreamWorker (streaming mode). Keeping one implementation is
  // what guarantees the two modes select the same record set and produce
  // identical output.
  class BamRecordExtractor {
  public:
    // The cheap first cut, shared by the file-mode harvest, the streaming
    // dispatcher and extract_record: a read with no SEQ or any bit of the -g
    // mask set is dropped before anything else is looked at.
    static bool passes_prefilter(const bam1_t* read, int filter_flag)
    {
      return read->core.l_qseq != 0 && (read->core.flag & filter_flag) == 0;
    }

    // The POD5 read this record belongs to: the pi tag for a split read,
    // otherwise the query name.
    static const char* parent_id(const bam1_t* read)
    {
      uint8_t* pi_tag = bam_aux_get(read, "pi");
      return pi_tag ? bam_aux2Z(pi_tag) : bam_get_qname(read);
    }

    // Applies the filter/quality checks and fills `record` on success.
    // Returns false when the read must be skipped, in which case `record`
    // holds partial data that the caller must not use.
    //
    // Two release rules, the same as in the Python preprocessing:
    //  * a read whose move table stride (mv[0]) is not `sampling` is skipped and counted in
    //    stride_mismatch, because the signal is cut at `sampling` samples per move;
    //  * a reverse-strand read is turned into RNA-sense order: the base of interest is
    //    complemented when the aligned pairs are selected, and the sequence is
    //    reverse-complemented, the base qualities reversed and the query positions mirrored.
    //    The signal and the move table are already in RNA-sense order.
    static bool extract_record(bam1_t* read, int filter_flag, int bq_cutoff,
                               char base_of_interest, bool use_trimming,
                               int trim_threshold, int trim_penalty, int sampling,
                               BamRecord& record);

    // Reads skipped by the stride rule, over all threads of the run.
    static inline atomic<long> stride_mismatch{0};
    // Records that passed the other checks but have no MD tag (the run fails when this is not 0).
    static inline atomic<long> missing_md{0};
    // Records whose MD tag does not fit their CIGAR or is not a string (the run fails when this is not 0,
    // as the Python preprocessing stops on such a record).
    static inline atomic<long> invalid_md{0};

    // Quality trimming, ported from the Python preprocessing (quality_trimming): returns the final
    // [trim_left, trim_right) interval, or (0, 0) when trimming failed --
    // the caller then keeps the read untrimmed, as the Python preprocessing does.
    static pair<int, int> quality_trimming(
        const vector<uint8_t>& seq, const vector<uint8_t>& bq,
        int threshold, int reward, int penalty,
        int len_threshold, double artifact_cutoff);

    static vector<pair<int32_t, int32_t>> get_aligned_pairs(bam1_t* read, char boi);
    static string build_alignment_sequence(bam1_t* read);
    static string build_reference_sequence(bam1_t* read);
    static uint32_t get_md_reference_length(const char* md_tag);
  };
} // namespace deeprm
