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

#include "BamReader.h"
#include "../utils/Utils.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <cctype>
#include <utility>

namespace deeprm {
  BamReader::BamReader(const string& path, int bq_threshold, char boi, int filter_flag,
                       unordered_map<string, int>& ref_index_dict)
    : bam_path(path), bq_cutoff(bq_threshold), base_of_interest(boi),
      filter_flag(filter_flag), ref_index_dict(ref_index_dict)
  {
  }

  BamReader::~BamReader()
  {
  }

  uint32_t BamReader::get_md_reference_length(const char* md_tag)
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

  string BamReader::build_alignment_sequence(bam1_t* read)
  {
    if (!read) return "";

    uint8_t* md_tag_ptr = bam_aux_get(read, "MD");
    if (!md_tag_ptr) return "";

    // Get query start and end considering soft clipping
    uint32_t start = 0;
    uint32_t end = read->core.l_qseq;
    uint32_t* cigar = bam_get_cigar(read);

    // Check for soft clipping at the beginning
    if (read->core.n_cigar > 0) {
      int op = bam_cigar_op(cigar[0]);
      if (op == BAM_CSOFT_CLIP) {
        start = bam_cigar_oplen(cigar[0]);
      }
      // Check for soft clipping at the end
      if (read->core.n_cigar > 1) {
        op = bam_cigar_op(cigar[read->core.n_cigar - 1]);
        if (op == BAM_CSOFT_CLIP) {
          end -= bam_cigar_oplen(cigar[read->core.n_cigar - 1]);
        }
      }
    }

    // Get read sequence
    uint8_t* seq = bam_get_seq(read);
    string read_sequence;
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
      // Work around HTSeq bug
      static char md_buffer[2];
      md_buffer[0] = bam_aux2A(md_tag_ptr);
      md_buffer[1] = '\0';
      md_tag = md_buffer;
    } else {
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

  string BamReader::build_reference_sequence(bam1_t* read)
  {
    string ref_seq = build_alignment_sequence(read);
    if (ref_seq.empty()) return "";

    string s;
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

  vector<pair<int32_t, int32_t>> BamReader::get_aligned_pairs(bam1_t* read, char boi)
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
            if (cmp_less(ref_seq_idx, ref_seq.length()) && ref_seq[ref_seq_idx] ==
              boi) {
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

  void BamReader::process_read(bam1_t* read, vector<BamRecord>& records)
  {
    // BAM_FUNMAP and l_qseq==0 are unconditional safety checks: downstream
    // code dereferences reference and sequence, so an unmapped record (tid
    // == -1, no CIGAR) or a sequence-less secondary alignment (the BAM
    // standard stores secondaries with seq='*', i.e. l_qseq==0) would
    // crash. The default `-g 276` happens to mask both via its 4 (UNMAP)
    // and 256 (SECONDARY) bits, but any user-supplied -g value missing
    // them — e.g. `-g 0`, `-g 4`, `-g 16` — would let the unsafe records
    // through. These two checks make the path safe for any -g value.
    //
    // filter_flag is then the user-tunable SAM-flag mask, kept identical to
    // MergedDataWorker's no-C path so that -C and no-C produce the same
    // record set for any -g value.
    if (read->core.flag & BAM_FUNMAP) return;
    if (read->core.l_qseq == 0) return;
    if (read->core.flag & filter_flag) return;

    // Check for mv tag
    uint8_t* mv_tag = bam_aux_get(read, "mv");
    if (!mv_tag) return;

    // Get base qualities
    vector<uint8_t> bq;
    uint8_t* qual = bam_get_qual(read);
    for (int i = 0; i < read->core.l_qseq; ++i) {
      bq.push_back(qual[i]);
    }

    // Check quality threshold
    if (Utils::mean_phred(bq) < bq_cutoff) return;

    // Get aligned pairs for base of interest
    auto ap = get_aligned_pairs(read, base_of_interest);
    if (ap.empty()) return;

    BamRecord record;

    // Get read ID and parent ID
    record.read_id = string(bam_get_qname(read));
    uint8_t* pi_tag = bam_aux_get(read, "pi");
    record.parent_id = pi_tag ? string(bam_aux2Z(pi_tag)) : record.read_id;

    // Get tags
    uint8_t* ts_tag = bam_aux_get(read, "ts");
    record.ts = ts_tag ? bam_aux2i(ts_tag) : 0;

    uint8_t* ns_tag = bam_aux_get(read, "ns");
    record.ns = ns_tag ? bam_aux2i(ns_tag) : 0;

    uint8_t* sp_tag = bam_aux_get(read, "sp");
    record.sp = sp_tag ? bam_aux2i(sp_tag) : 0;

    // Get move array
    int mv_len = bam_auxB_len(mv_tag);
    for (int i = 1; i < mv_len; ++i) {
      int64_t mv_val = bam_auxB2i(mv_tag, i);
      record.mv.push_back(mv_val != 0);
    }

    // Get sequence
    uint8_t* seq = bam_get_seq(read);
    for (int i = 0; i < read->core.l_qseq; ++i) {
      record.seq.push_back(seq_nt16_str[bam_seqi(seq, i)]);
    }

    record.bq = bq;

    // Get reference id
    record.ref = read->core.tid;

    record.ap = ap;

    // Get strand information
    record.strand = (read->core.flag & BAM_FREVERSE) ? -1 : 1;

    records.push_back(record);
  }

  vector<BamRecord> BamReader::parse_bam(int process_id, int num_processes, int num_threads)
  {
    vector<BamRecord> records;

    samFile* bam_file = sam_open(bam_path.c_str(), "r");
    if (!bam_file) {
      cerr << "Failed to open BAM file: " << bam_path << endl;
      return records;
    }

    // Set threads
    if (num_threads > 1) {
      hts_set_threads(bam_file, num_threads);
    }

    sam_hdr_t* header = sam_hdr_read(bam_file);
    if (!header) {
      cerr << "Failed to read BAM header" << endl;
      sam_close(bam_file);
      return records;
    }

    bam1_t* read = bam_init1();
    int read_idx = 0;

    while (sam_read1(bam_file, header, read) >= 0) {
      if (read_idx % num_processes == process_id) {
        process_read(read, records);
      }
      read_idx++;
    }

    bam_destroy1(read);
    sam_hdr_destroy(header);
    sam_close(bam_file);

    return records;
  }
} // namespace deeprm
