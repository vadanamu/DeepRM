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

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <htslib/hts.h>
#include <htslib/sam.h>

using namespace std;

namespace deeprm {
  struct BamRecord {
    string read_id; // 32 bytes
    string parent_id; // 32 bytes
    int32_t ts; // 4 bytes
    int32_t ns; // 4 bytes
    int32_t sp; // 4 bytes
    int32_t ref; // 4 bytes
    vector<uint8_t> seq; // 24 bytes
    vector<uint8_t> bq; // 24 bytes
    vector<pair<int32_t, int32_t>> ap; // 24 bytes
    vector<bool> mv; // 40 bytes
    int8_t strand; // 1 byte: -1 for reversed, 1 for not reversed
  };

  class BamReader {
  private:
    string bam_path;
    int bq_cutoff;
    char base_of_interest;
    int filter_flag;  // SAM flag bits to exclude (matches MergedDataWorker no-C path)
    unordered_map<string, int> ref_index_dict;

    vector<pair<int32_t, int32_t>> get_aligned_pairs(bam1_t* read, char boi);
    string build_alignment_sequence(bam1_t* read);
    string build_reference_sequence(bam1_t* read);
    uint32_t get_md_reference_length(const char* md_tag);

  public:
    BamReader(const string& path, int bq_threshold, char boi, int filter_flag,
              unordered_map<string, int>& ref_index_dict);
    ~BamReader();

    vector<BamRecord> parse_bam(int process_id, int num_processes, int num_threads);
    void process_read(bam1_t* read, vector<BamRecord>& records);
  };
} // namespace deeprm
