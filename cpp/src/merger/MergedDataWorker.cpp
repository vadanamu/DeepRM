/***************************************************************************************************
 *
 * Copyright (C) 2025 Genome4me Incorporated - All Rights Reserved.
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

#include "MergedDataWorker.h"
#include "../utils/Utils.h"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cctype>
#include <utility>

namespace deeprm {
  namespace {
    // Watson-Crick complement; returns the input unchanged for anything that
    // is not a canonical base so non-ACGT bases-of-interest still work.
    char complement_base(char b)
    {
      switch (b) {
        case 'A': return 'T';
        case 'T': return 'A';
        case 'C': return 'G';
        case 'G': return 'C';
        default:  return b;
      }
    }
  } // namespace

  MergedDataWorker::MergedDataWorker(int id, const Arguments& args,
                                     const vector<Pod5RecordMeta>& pod5_meta)
    : worker_id(id), args(args), pod5_meta_records(pod5_meta),
      bam_header(nullptr), is_running(false), writer(nullptr),
      output_index(0), current_file_index(0)
  {
  }

  MergedDataWorker::~MergedDataWorker()
  {
    stop();
    if (writer) {
      delete writer;
    }
  }

  void MergedDataWorker::start()
  {
    is_running = true;

    // Create NPZ writer
    writer = new NpzWriter(args.output_path, args.chunk_size, worker_id);

    log_info() << "merged data worker " << worker_id
        << " starting with " << pod5_meta_records.size()
        << " POD5 records" << endl;
    worker_thread = thread(&MergedDataWorker::process_loop, this);
  }

  void MergedDataWorker::stop()
  {
    is_running = false;
    queue_cv.notify_all();
    if (worker_thread.joinable()) {
      worker_thread.join();
    }
  }

  void MergedDataWorker::signal_stop()
  {
    is_running = false;
    queue_cv.notify_all();
  }

  void MergedDataWorker::wait_for_completion()
  {
    if (worker_thread.joinable()) {
      worker_thread.join();
    }
  }

  void MergedDataWorker::add_bam_data(bam1_t* read)
  {
    unique_lock<mutex> lock(queue_mutex);
    bam_queue.push(read);
    queue_cv.notify_one();
  }

  void MergedDataWorker::process_loop()
  {
    NormalizationFactors norm_factors;

    unordered_map<string, Pod5RecordMeta> pod5_index;
    // Build POD5 index for this worker
    pod5_index.reserve(pod5_meta_records.size());
    for (const auto& record : pod5_meta_records) {
      pod5_index[record.read_id] = record;
    }
    log_info() << "merged data worker " << worker_id
        << " indexed " << pod5_meta_records.size()
        << " POD5 records" << endl;

    vector<pair<Pod5RecordMeta, BamRecord>> batch_merged;

    while (is_running || !bam_queue.empty()) {
      unique_lock<mutex> lock(queue_mutex);

      if (bam_queue.empty()) {
        if (!is_running) break;
        queue_cv.wait(lock);
        continue;
      }

      // Get BAM read from queue
      bam1_t* read = bam_queue.front();
      bam_queue.pop();
      lock.unlock();

      // Convert bam1_t to BamRecord
      BamRecord bam_record;
      if (!convert_bam1_to_record(read, bam_header, bam_record))
        continue;

      // Find matching POD5 record
      auto pod5_it = pod5_index.find(bam_record.parent_id);
      if (pod5_it != pod5_index.end()) {
        batch_merged.emplace_back(pod5_it->second, bam_record);

        // Process batch when it reaches process_once size
        if (batch_merged.size() >= static_cast<size_t>(args.process_once)) {
          output_index++;
          log_info() << "merged data worker " << worker_id
              << " processing batch " << output_index
              << " with " << batch_merged.size() << " records" << endl;

          // Create merger and process
          RecordMerger merger(norm_factors, args.cb_len, args.kmer_len, args.max_token_len,
                              args.sampling, args.dwell_shift, args.sig_window, args.label_div);

          vector<Pod5RecordMeta> batch_pod5_meta;
          vector<BamRecord> batch_bam;
          batch_pod5_meta.reserve(batch_merged.size());
          batch_bam.reserve(batch_merged.size());

          for (auto& pair : batch_merged) {
            batch_pod5_meta.push_back(move(pair.first));
            batch_bam.push_back(move(pair.second));
          }

          merger.add_bam_records(move(batch_bam));
          merger.add_pod5_meta_records(move(batch_pod5_meta));

          vector<ProcessedRecord> processed_records = merger.merge_and_process_with_meta();

          if (!processed_records.empty()) {
            writer->add_records(processed_records);
          }

          writer->increment_processing_unit();
          batch_merged.clear();
        }
      }

      // Clean up
      bam_destroy1(read);
    }

    // Process remaining records
    if (!batch_merged.empty()) {
      output_index++;
      log_info() << "merged data worker " << worker_id
          << " processing final batch " << output_index
          << " with " << batch_merged.size() << " records" << endl;

      RecordMerger merger(norm_factors, args.cb_len, args.kmer_len, args.max_token_len,
                          args.sampling, args.dwell_shift, args.sig_window, args.label_div);

      vector<Pod5RecordMeta> batch_pod5_meta;
      vector<BamRecord> batch_bam;
      batch_pod5_meta.reserve(batch_merged.size());
      batch_bam.reserve(batch_merged.size());

      for (auto& pair : batch_merged) {
        batch_pod5_meta.push_back(move(pair.first));
        batch_bam.push_back(move(pair.second));
      }

      merger.add_bam_records(move(batch_bam));
      merger.add_pod5_meta_records(move(batch_pod5_meta));

      vector<ProcessedRecord> processed_records = merger.merge_and_process_with_meta();

      if (!processed_records.empty()) {
        writer->add_records(processed_records);
      }

      writer->increment_processing_unit();
    }

    // Flush remaining records in the worker thread
    if (writer) {
      log_info() << "merged data worker " << worker_id << " flushing remaining records" << endl;
      writer->flush();
    }

    log_info() << "merged data worker " << worker_id
        << " completed processing. Total batches: "
        << output_index << endl;
  }

  bool MergedDataWorker::convert_bam1_to_record(bam1_t* read, sam_hdr_t* header,
                                                BamRecord& record) const
  {
    if (read->core.flag & args.filter_flag)
      return false;

    // Check for mv tag
    uint8_t* mv_tag = bam_aux_get(read, "mv");
    if (!mv_tag)
      return false;

    // Get base qualities
    uint8_t* qual = bam_get_qual(read);
    for (int i = 0; i < read->core.l_qseq; ++i) {
      record.bq.push_back(qual[i]);
    }

    // Check quality threshold
    if (Utils::mean_phred(record.bq) < args.qcut)
      return false;

    // Get aligned pairs
    record.ap = get_aligned_pairs(read, args.base_of_interest, header);
    if (record.ap.empty())
      return false;

    // Get read ID and parent ID
    record.read_id = bam_get_qname(read);
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

    // Get reference id
    record.ref = read->core.tid;

    // Get strand information
    record.strand = (read->core.flag & BAM_FREVERSE) ? -1 : 1;

    return true;
  }

  vector<pair<int32_t, int32_t>> MergedDataWorker::get_aligned_pairs(
    bam1_t* read, char boi, sam_hdr_t* /*header*/)
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

    // Make the base-of-interest strand-aware: for a reverse-strand read the
    // forward-reference base at a base-of-interest site is the complement of
    // `boi` (e.g. an m6A 'A' appears as 'T' on the forward reference).
    char effective_boi = (read->core.flag & BAM_FREVERSE)
                           ? complement_base(boi) : boi;

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
            if (cmp_less(ref_seq_idx, ref_seq.length()) && ref_seq[ref_seq_idx] == effective_boi) {
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

  string MergedDataWorker::build_alignment_sequence(bam1_t* read)
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

  string MergedDataWorker::build_reference_sequence(bam1_t* read)
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

  uint32_t MergedDataWorker::get_md_reference_length(const char* md_tag)
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
}
