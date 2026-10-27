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
  int dwell_shift;
  int sig_window;
  int filter_flag;
  // Site handling, as in the Python preprocessing. padding_mode: "none" drops out-of-bounds contexts
  // (default), "zero"/"edge" keep them padded. Trimming is the same as there.
  string padding_mode;
  // Streaming producer command: its stdout is the BAM input, and a non-zero exit
  // fails the whole run so a truncated stream can never pass as a complete one.
  // Mutually exclusive with -b.
  string bam_cmd;
  bool use_trimming;
  int trim_threshold;
  int trim_penalty;
  int max_queue;
  uint64_t label_div;
  // Derived, not an option: true when the BAM arrives as a stream (-b - or
  // --bam-cmd) and is processed in BAM order as it flows in; false when -b
  // names a file, which is harvested up front and processed in POD5 order.
  // The input form decides because the file mode needs a seekable, re-readable
  // BAM and a stream is forward-only.
  bool streaming;

  // Default values
  Arguments()
  {
    cpu_count = max(1, static_cast<int>(thread::hardware_concurrency()));
    qcut = 0;  // release default: keep all reads (deeprm call prep passes its own value anyway)
    chunk_size = 16000;
    max_token_len = 200;
    sampling = 6;
    base_of_interest = 'A';
    kmer_len = 5;
    cb_len = 21;
    // Lowered from 16: streaming is insensitive to -a (22:30 vs 22:09 at
    // t512), while file mode runs cpu_count / bam_threads phase-1 parsers, so a
    // smaller -a means more parsers (t112: 7 -> 28, 1:06:47 -> 1:03:02).
    bam_threads = 4;
    dwell_shift = 10;
    sig_window = 5;
    filter_flag = 276;
    padding_mode = "none";
    bam_cmd = "";
    use_trimming = false;
    trim_threshold = 5;
    trim_penalty = -15;
    // Reads per worker. Bounds BOTH the reader's staging queue and the TOTAL
    // sitting in all worker queues at -Q * workers (SamDispatcher's
    // worker_inflight), so the budget tracks -t. 1000 reads/worker is ~12 MB
    // of bam1_t. Small budgets starve the workers: an absolute -Q 4000 once
    // collapsed throughput 10x without lowering peak memory.
    max_queue = 1000;
    label_div = 1000000000ULL;
    streaming = false;
  }
};

class ArgumentParser {
private:
  Arguments args;

  void print_help(const char* program_name);
  // Reject parameter geometry that cannot emit a row (silent empty success)
  // or corrupts output.
  void validate_geometry();
  void print_version();

public:
  ArgumentParser();
  ~ArgumentParser();

  Arguments parse(int argc, char* argv[]);
  void validate_arguments();
};
