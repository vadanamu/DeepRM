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

#pragma once

#include <string>
#include <thread>

using namespace std;

struct Arguments {
  string pod5_path;
  string bam_path;
  string output_path;
  int cpu_count;
  int qcut;
  int chunk_size;
  int max_token_len;
  int sampling;
  char base_of_interest;
  int kmer_len;
  int cb_len;
  int bam_threads;
  int process_once;
  int dwell_shift;
  int sig_window;
  int filter_flag;
  uint64_t label_div;
  bool consistency;

  // Default values
  Arguments()
  {
    cpu_count = max(1, static_cast<int>(thread::hardware_concurrency()));
    qcut = 0;
    chunk_size = 16000;
    max_token_len = 200;
    sampling = 6;
    base_of_interest = 'A';
    kmer_len = 5;
    cb_len = 21;
    bam_threads = 16;
    process_once = 1000;
    dwell_shift = 10;
    sig_window = 5;
    // Default 276 = unmapped 4 + reversed 16 + secondary 256 (transcriptome-
    // mapped BAMs: drop antisense). Use 260 (unmapped 4 + secondary 256) for
    // genome-mapped BAMs so minus-strand-gene reads are kept and processed in
    // RNA-sense.
    filter_flag = 276;
    label_div = 1000000000ULL;
    consistency = false;
  }
};

class ArgumentParser {
private:
  Arguments args;

  void print_help(const char* program_name);
  void print_version();

public:
  ArgumentParser();
  ~ArgumentParser();

  Arguments parse(int argc, char* argv[]);
  void validate_arguments();
};
