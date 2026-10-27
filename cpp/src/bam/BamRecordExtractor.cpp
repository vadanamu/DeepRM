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

#include "bam/BamRecordExtractor.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <utility>

#include "utils/Utils.h"
#include "utils/Uuid128.h"

namespace deeprm {
  // Return length of the reference span an MD tag describes: match run lengths, plus one per
  // mismatch base, plus the bases of each ^-deletion.
  uint32_t BamRecordExtractor::get_md_reference_length(const char* md_tag)
  {
    uint32_t length = 0;
    int idx = 0;

    while (md_tag[idx] != '\0') {
      if (md_tag[idx] >= '0' && md_tag[idx] <= '9') {
        // Number of matches
        int num = 0;
        while (md_tag[idx] >= '0' && md_tag[idx] <= '9') {
          num = num * 10 + (md_tag[idx] - '0');
          idx++;
        }
        length += num;
      } else if (md_tag[idx] == '^') {
        // Deletion
        idx++;
        while (md_tag[idx] >= 'A' && md_tag[idx] <= 'Z') {
          length++;
          idx++;
        }
      } else {
        // Mismatch
        length++;
        idx++;
      }
    }

    return length;
  }

  // Build soft-clip-stripped query sequence with every MD-tag edit applied (mismatches substituted,
  // deletions inserted), mirroring pysam's build_alignment_sequence. Empty string when the MD tag
  // is absent.
  string BamRecordExtractor::build_alignment_sequence(bam1_t* read)
  {
    if (!read) return "";

    uint8_t* md_tag_ptr = bam_aux_get(read, "MD");
    if (!md_tag_ptr) return "";

    // Get query start and end considering soft clipping
    uint32_t start = 0;
    uint32_t end = read->core.l_qseq;
    uint32_t* cigar = bam_get_cigar(read);

    // Soft clips at each end, after any hard clip there (for example 4H3S...): a hard clip is
    // outermost and its bases are not in SEQ, as pysam's build_alignment_sequence assumes.
    int64_t n_cigar = read->core.n_cigar;
    int64_t first = 0;
    while (first < n_cigar && bam_cigar_op(cigar[first]) == BAM_CHARD_CLIP) ++first;
    if (first < n_cigar && bam_cigar_op(cigar[first]) == BAM_CSOFT_CLIP) {
      start = bam_cigar_oplen(cigar[first]);
    }
    int64_t last = n_cigar - 1;
    while (last > first && bam_cigar_op(cigar[last]) == BAM_CHARD_CLIP) --last;
    if (last > first && bam_cigar_op(cigar[last]) == BAM_CSOFT_CLIP) {
      end -= bam_cigar_oplen(cigar[last]);
    }

    // Get read sequence
    uint8_t* seq = bam_get_seq(read);
    string read_sequence;
    read_sequence.reserve(end - start);
    for (uint32_t i = start; i < end; ++i) {
      read_sequence += seq_nt16_str[bam_seqi(seq, i)];
    }

    uint32_t r_idx = 0;
    int s_idx = 0;
    uint32_t max_len = 0;

    // Calculate alignment length
    for (uint32_t k = 0; k < read->core.n_cigar; ++k) {
      int op = bam_cigar_op(cigar[k]);
      int len = bam_cigar_oplen(cigar[k]);

      switch (op) {
        case BAM_CMATCH:
        case BAM_CEQUAL:
        case BAM_CDIFF:
        case BAM_CDEL:
        case BAM_CINS:
        case BAM_CPAD:
          max_len += len;
          break;
      }
    }

    if (max_len == 0) return "";

    string s(max_len + 1, '\0');

    // Build alignment sequence from CIGAR
    for (uint32_t k = 0; k < read->core.n_cigar; ++k) {
      int op = bam_cigar_op(cigar[k]);
      int len = bam_cigar_oplen(cigar[k]);

      switch (op) {
        case BAM_CMATCH:
        case BAM_CEQUAL:
        case BAM_CDIFF:
          for (int i = 0; i < len; ++i) {
            s[s_idx] = read_sequence[r_idx];
            r_idx++;
            s_idx++;
          }
          break;
        case BAM_CDEL:
          for (int i = 0; i < len; ++i) {
            s[s_idx] = '-';
            s_idx++;
          }
          break;
        case BAM_CREF_SKIP:
          break;
        case BAM_CINS:
        case BAM_CPAD:
          for (int i = 0; i < len; ++i) {
            // Encode insertions as lowercase
            s[s_idx] = tolower(read_sequence[r_idx]);
            r_idx++;
            s_idx++;
          }
          break;
        case BAM_CSOFT_CLIP:
        case BAM_CHARD_CLIP:
          break;
      }
    }

    // Get MD tag
    char* md_tag;
    uint8_t md_typecode = md_tag_ptr[0];
    if (md_typecode == 'Z') {
      md_tag = bam_aux2Z(md_tag_ptr);
    } else if (md_typecode == 'A') {
      // Work around HTSeq bug (one buffer per thread: workers extract records in parallel)
      static thread_local char md_buffer[2];
      md_buffer[0] = bam_aux2A(md_tag_ptr);
      md_buffer[1] = '\0';
      md_tag = md_buffer;
    } else {
      invalid_md.fetch_add(1, memory_order_relaxed);
      return "";
    }

    // Apply MD tag
    int md_idx = 0;
    s_idx = 0;
    int nmatches = 0;

    // Count insertions
    int insertions = 0;
    for (int i = 0; cmp_less(i, max_len); ++i) {
      if (s[i] >= 'a' && s[i] <= 'z') {
        insertions++;
      }
    }

    // Validate MD tag length
    uint32_t md_len = get_md_reference_length(md_tag);
    if (md_len + insertions > max_len) {
      invalid_md.fetch_add(1, memory_order_relaxed);
      return "";
    }

    s_idx = 0;
    r_idx = 0;

    while (md_tag[md_idx] != '\0') {
      if (md_tag[md_idx] >= '0' && md_tag[md_idx] <= '9') {
        // Number of matches
        nmatches = nmatches * 10 + (md_tag[md_idx] - '0');
        md_idx++;
        continue;
      } else {
        // Process matches, skipping insertions
        for (int x = 0; x < nmatches; ++x) {
          while (cmp_less(s_idx, max_len) && s[s_idx] >= 'a' && s[s_idx] <= 'z') {
            s_idx++;
          }
          s_idx++;
        }
        while (cmp_less(s_idx, max_len) && s[s_idx] >= 'a' && s[s_idx] <= 'z') {
          s_idx++;
        }

        r_idx += nmatches;
        nmatches = 0;

        if (md_tag[md_idx] == '^') {
          // Deletion
          md_idx++;
          while (md_tag[md_idx] >= 'A' && md_tag[md_idx] <= 'Z') {
            if (cmp_less(s_idx, max_len)) {
              s[s_idx] = md_tag[md_idx];
            }
            s_idx++;
            md_idx++;
          }
        } else {
          // Mismatch - enforce lowercase
          char c = md_tag[md_idx];
          if (c <= 'Z') {
            c += 32;
          }
          if (cmp_less(s_idx, max_len)) {
            s[s_idx] = c;
          }
          s_idx++;
          r_idx++;
          md_idx++;
        }
      }
    }

    // Process remaining matches
    for (int x = 0; x < nmatches; ++x) {
      while (cmp_less(s_idx, max_len) && s[s_idx] >= 'a' && s[s_idx] <= 'z') {
        s_idx++;
      }
      s_idx++;
    }
    while (cmp_less(s_idx, max_len) && s[s_idx] >= 'a' && s[s_idx] <= 'z') {
      s_idx++;
    }

    return s.substr(0, s_idx);
  }

  // Build reference sequence covered by the alignment, rebuilt by walking the CIGAR over
  // build_alignment_sequence() (insertions dropped, deletions kept), mirroring pysam's
  // build_reference_sequence.
  string BamRecordExtractor::build_reference_sequence(bam1_t* read)
  {
    string ref_seq = build_alignment_sequence(read);
    if (ref_seq.empty()) return "";

    string s;
    s.reserve(ref_seq.length());
    uint32_t* cigar = bam_get_cigar(read);
    uint32_t r_idx = 0;

    for (uint32_t k = 0; k < read->core.n_cigar; ++k) {
      int op = bam_cigar_op(cigar[k]);
      int len = bam_cigar_oplen(cigar[k]);

      switch (op) {
        case BAM_CMATCH:
        case BAM_CEQUAL:
        case BAM_CDIFF:
          for (int i = 0; i < len; ++i) {
            if (r_idx < ref_seq.length()) {
              s += ref_seq[r_idx];
              r_idx++;
            }
          }
          break;
        case BAM_CDEL:
          for (int i = 0; i < len; ++i) {
            if (r_idx < ref_seq.length()) {
              s += ref_seq[r_idx];
              r_idx++;
            }
          }
          break;
        case BAM_CREF_SKIP:
          break;
        case BAM_CINS:
        case BAM_CPAD:
          r_idx += len;
          break;
        case BAM_CSOFT_CLIP:
        case BAM_CHARD_CLIP:
          break;
      }
    }

    return s;
  }

  // Return (query_pos, ref_pos) pairs for every aligned site whose REFERENCE base is `boi`, from
  // the MD-reconstructed reference. Empty when MD is absent.
  vector<pair<int32_t, int32_t>> BamRecordExtractor::get_aligned_pairs(bam1_t* read, char boi)
  {
    vector<pair<int32_t, int32_t>> pairs;

    // Get MD tag to reconstruct reference sequence
    uint8_t* md_tag = bam_aux_get(read, "MD");
    if (!md_tag) {
      return pairs;
    }

    // Build reference sequence using pysam logic
    string ref_seq = build_reference_sequence(read);
    if (ref_seq.empty()) {
      return pairs;
    }

    // Exact-count reserve: every emitted site is a `boi` position of ref_seq,
    // so this is a tight upper bound (only boi under deletions don't emit).
    // Doubling growth left ~1.5x capacity resident in the file-mode index for the
    // whole run, and shrink_to_fit() traded that for per-record allocator
    // churn that grew streaming RSS by ~19 GB at t512; counting first has
    // neither cost.
    pairs.reserve(count(ref_seq.begin(), ref_seq.end(), boi));

    uint32_t* cigar = bam_get_cigar(read);
    int32_t ref_pos = read->core.pos;
    int32_t query_pos = 0;
    int32_t ref_seq_idx = 0;

    for (uint32_t i = 0; i < read->core.n_cigar; ++i) {
      int op = bam_cigar_op(cigar[i]);
      int len = bam_cigar_oplen(cigar[i]);

      switch (op) {
        case BAM_CMATCH:
        case BAM_CEQUAL:
        case BAM_CDIFF:
          for (int j = 0; j < len; ++j) {
            // Check reference base
            if (cmp_less(ref_seq_idx, ref_seq.length()) && ref_seq[ref_seq_idx] == boi) {
              pairs.emplace_back(query_pos, ref_pos + j);
            }
            query_pos++;
            ref_seq_idx++;
          }
          ref_pos += len;
          break;
        case BAM_CINS:
        case BAM_CSOFT_CLIP:
        case BAM_CPAD:
          query_pos += len;
          break;
        case BAM_CDEL:
          ref_seq_idx += len;
          ref_pos += len;
          break;
        case BAM_CREF_SKIP:
          ref_pos += len;
          break;
        case BAM_CHARD_CLIP:
        case BAM_CBACK:
        default:
          break;
      }
    }

    return pairs;
  }

  // Used as single record-selection path shared by file mode and streaming. Checks in order:
  // empty SEQ, flag mask, mv tag present, mean-phred cutoff (bq is filled first), MD/aligned
  // pairs; the remaining fields (seq/mv, trim bounds) are filled after the last check.
  // False = skip this read; the partially-filled record must then be discarded, not reused.
  namespace {
    char complement_base(char b)
    {
      switch (b) {
        case 'A': return 'T';
        case 'T': return 'A';
        case 'C': return 'G';
        case 'G': return 'C';
        case 'a': return 't';
        case 't': return 'a';
        case 'c': return 'g';
        case 'g': return 'c';
        default: return b;
      }
    }
  }  // namespace

  bool BamRecordExtractor::extract_record(bam1_t* read, int filter_flag, int bq_cutoff,
                                          char base_of_interest, bool use_trimming,
                                          int trim_threshold, int trim_penalty, int sampling,
                                          BamRecord& record)
  {
    // bq, seq and mv are filled with push_back while every other field is
    // overwritten, so stale contents would accumulate instead of being
    // replaced.
    record.bq.clear();
    record.seq.clear();
    record.mv.clear();

    if (!passes_prefilter(read, filter_flag))
      return false;

    // Check for mv tag
    uint8_t* mv_tag = bam_aux_get(read, "mv");
    if (!mv_tag)
      return false;

    // Get base qualities. A record stored without them (QUAL '*', first byte 0xff) is skipped, as the
    // Python preprocessing does.
    uint8_t* qual = bam_get_qual(read);
    if (read->core.l_qseq > 0 && qual[0] == 0xff)
      return false;
    record.bq.reserve(read->core.l_qseq);
    for (int i = 0; i < read->core.l_qseq; ++i) {
      record.bq.push_back(qual[i]);
    }

    // Check quality threshold
    if (Utils::mean_phred(record.bq) < bq_cutoff)
      return false;

    // Release rule: the move table must hold a stride and at least one move, and the stride must
    // be --sampling, the step the signal is cut at.
    if (bam_auxB_len(mv_tag) < 2)
      return false;
    if (bam_auxB2i(mv_tag, 0) != sampling) {
      stride_mismatch.fetch_add(1, memory_order_relaxed);
      return false;
    }

    // The aligned pairs need the reference bases from the MD tag. A record without one cannot be read
    // correctly; it is counted and the run fails at the end, as the Python preprocessing stops on it.
    if (!bam_aux_get(read, "MD")) {
      missing_md.fetch_add(1, memory_order_relaxed);
      return false;
    }

    // Get aligned pairs for base of interest. Release rule: on a reverse-strand read the base of
    // interest is its complement on the forward reference.
    const bool is_reverse = (read->core.flag & BAM_FREVERSE) != 0;
    record.ap = get_aligned_pairs(read, is_reverse ? complement_base(base_of_interest) : base_of_interest);
    if (record.ap.empty())
      return false;

    // Read id: parse the UUID once here instead of once per SITE in
    // commit_row. The 16 bytes have byte alignment, so memcpy is the defined
    // way to take the halves.
    const char* qname = bam_get_qname(read);
    uint8_t uuid[16];
    if (parse_uuid36(qname, uuid)) {
      memcpy(&record.rid0, uuid, sizeof(record.rid0));
      memcpy(&record.rid1, uuid + sizeof(record.rid0), sizeof(record.rid1));
    } else {
      record.rid0 = 0;
      record.rid1 = 0;
      // Log so silent corruption (every row showing the same null UUID)
      // doesn't go unnoticed downstream.
      cerr << "BamRecordExtractor: invalid UUID '" << qname
           << "', writing zero read_id" << endl;
    }
    // Get tags
    uint8_t* ts_tag = bam_aux_get(read, "ts");
    record.ts = ts_tag ? bam_aux2i(ts_tag) : 0;

    uint8_t* ns_tag = bam_aux_get(read, "ns");
    record.ns = ns_tag ? bam_aux2i(ns_tag) : 0;

    uint8_t* sp_tag = bam_aux_get(read, "sp");
    record.sp = sp_tag ? bam_aux2i(sp_tag) : 0;

    // Get move array
    int mv_len = bam_auxB_len(mv_tag);
    if (mv_len > 0)
      record.mv.reserve(mv_len - 1);
    for (int i = 1; i < mv_len; ++i) {
      int64_t mv_val = bam_auxB2i(mv_tag, i);
      record.mv.push_back(mv_val != 0);
    }

    // Get sequence
    uint8_t* seq = bam_get_seq(read);
    record.seq.reserve(read->core.l_qseq);
    for (int i = 0; i < read->core.l_qseq; ++i) {
      record.seq.push_back(seq_nt16_str[bam_seqi(seq, i)]);
    }

    // Get reference id
    record.ref = read->core.tid;

    // Get strand information
    record.strand = (read->core.flag & BAM_FREVERSE) ? -1 : 1;

    if (is_reverse) {
      // Release rule: present a reverse-strand read in RNA-sense order. The BAM stores its SEQ and
      // QUAL reverse-complemented into forward-reference order, and the aligned pairs give
      // forward-reference query positions, while the signal and the move table are in basecalled
      // (RNA-sense 5'->3') order. So reverse-complement seq, reverse bq and mirror the query
      // positions. Trimming below then runs in RNA-sense coordinates too.
      reverse(record.seq.begin(), record.seq.end());
      for (auto& b : record.seq)
        b = static_cast<uint8_t>(complement_base(static_cast<char>(b)));
      reverse(record.bq.begin(), record.bq.end());
      const int32_t last = static_cast<int32_t>(record.seq.size()) - 1;
      for (auto& p : record.ap)
        p.first = last - p.first;
    }

    // Quality-trim bounds (as in the Python preprocessing): a failed trim keeps the read
    // untrimmed as (0, seq_len).
    record.trim_left = 0;
    record.trim_right = static_cast<int32_t>(record.seq.size());
    if (use_trimming) {
      auto [tl, tr] = quality_trimming(record.seq, record.bq, trim_threshold, 1,
                                       trim_penalty, 10, 0.5);
      if (!(tl == 0 && tr == 0)) {
        record.trim_left = tl;
        record.trim_right = tr;
      }
    }

    return true;
  }

  namespace {
    // np.argmin: index of the FIRST minimum.
    size_t argmin_first(const vector<long>& v)
    {
      size_t best = 0;
      for (size_t i = 1; i < v.size(); ++i)
        if (v[i] < v[best])
          best = i;
      return best;
    }

    // The Python preprocessing's quality_trimming_func, mirrored expression by expression:
    // candidate trim points from cumulative reward/penalty scores evaluated
    // from both ends, np.argmin first-minimum tie-breaking, and the
    // scores[:-t or None] slice idiom.
    pair<size_t, size_t> quality_trimming_func(const vector<uint8_t>& bq,
                                               int threshold, int reward,
                                               int penalty)
    {
      size_t n = bq.size();
      vector<long> scores(n);
      for (size_t i = 0; i < n; ++i)
        scores[i] = (static_cast<int>(bq[i]) >= threshold) ? reward : penalty;

      vector<long> acc;
      acc.reserve(n + 1);
      long run = 0;

      // trim_3p_3p = argmin([scores[-1]] ++ cumsum(flip(scores)))
      acc.push_back(scores[n - 1]);
      for (size_t i = n; i-- > 0;) {
        run += scores[i];
        acc.push_back(run);
      }
      size_t t33 = argmin_first(acc);

      // trim_3p_5p = argmin([scores[0]] ++ cumsum(scores[:-t33 or None]))
      size_t lim = (t33 == 0) ? n : n - t33;
      acc.clear();
      acc.push_back(scores[0]);
      run = 0;
      for (size_t i = 0; i < lim; ++i) {
        run += scores[i];
        acc.push_back(run);
      }
      size_t t35 = argmin_first(acc);

      // trim_5p_5p = argmin([scores[0]] ++ cumsum(scores))
      acc.clear();
      acc.push_back(scores[0]);
      run = 0;
      for (size_t i = 0; i < n; ++i) {
        run += scores[i];
        acc.push_back(run);
      }
      size_t t55 = argmin_first(acc);

      // trim_5p_3p = argmin([scores[-1]] ++ cumsum(flip(scores[t55:])))
      acc.clear();
      acc.push_back(scores[n - 1]);
      run = 0;
      for (size_t i = n; i-- > t55;) {
        run += scores[i];
        acc.push_back(run);
      }
      size_t t53 = argmin_first(acc);

      size_t t5, t3;
      if (t33 + t35 > t53 + t55) {
        t5 = t55;
        t3 = t53;
      } else {
        t5 = t35;
        t3 = t33;
      }
      if (t5 == n || t3 == n)
        return {0, 0};
      return {t5, n - t3};
    }
  } // namespace

  // Return the kept [left, right) bounds after score-based end trimming plus the GAA-repeat
  // artifact re-trim loop. Return {0, 0} when trimming fails, i.e. the kept span is
  // no longer than len_threshold bases.
  namespace {
    // Total length of the matches of (GAA){5,} in s, as Python's re.finditer finds them: leftmost first,
    // greedy, not overlapping. A plain scan, because std::regex matches recursively and a long GAA run
    // can use up a worker thread's stack.
    long gaa_repeat_length(const string& s)
    {
      long total = 0;
      size_t i = 0;
      const size_t n = s.size();
      while (i + 15 <= n) {
        size_t k = 0;
        while (i + 3 * (k + 1) <= n && s.compare(i + 3 * k, 3, "GAA") == 0)
          ++k;
        if (k >= 5) {
          total += static_cast<long>(3 * k);
          i += 3 * k;
        } else {
          ++i;
        }
      }
      return total;
    }
  }  // namespace

  pair<int, int> BamRecordExtractor::quality_trimming(
      const vector<uint8_t>& seq_in, const vector<uint8_t>& bq_in,
      int threshold, int reward, int penalty,
      int len_threshold, double artifact_cutoff)
  {
    // Iteratively re-trim while a GAA-repeat artifact dominates the kept span,
    // consuming ends exactly like the python loop (which mutates seq/bq).
    string seq(seq_in.begin(), seq_in.end());
    vector<uint8_t> bq = bq_in;
    bool failed = false;
    long f5 = 0, f3 = 0;

    while (true) {
      auto [t5, t3] = quality_trimming_func(bq, threshold, reward, penalty);
      if (static_cast<long>(t3) - static_cast<long>(t5) <= len_threshold) {
        failed = true;
        break;
      }
      string kept = seq.substr(t5, t3 - t5);
      double artifact_fraction = 0.0;
      if (!kept.empty()) {
        long artifact = gaa_repeat_length(kept);
        artifact_fraction = static_cast<double>(artifact) / kept.size();
      }
      if (artifact_fraction < artifact_cutoff) {
        // Order matters: f3 uses the pre-update f5, as in python.
        f3 = f5 + static_cast<long>(t3);
        f5 = f5 + static_cast<long>(t5);
        break;
      }
      size_t t5_len = t5;
      size_t t3_len = seq.size() - t3;
      if (t3_len >= t5_len) {
        f5 += static_cast<long>(t3);
        seq.erase(0, t3);
        bq.erase(bq.begin(), bq.begin() + t3);
      } else {
        seq.resize(t5);
        bq.resize(t5);
      }
      if (seq.empty()) {
        failed = true;
        break;
      }
    }

    if (failed)
      return {0, 0};
    return {static_cast<int>(f5), static_cast<int>(f3)};
  }
} // namespace deeprm
