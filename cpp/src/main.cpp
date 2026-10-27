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

#include <fcntl.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <mimalloc.h>

#include "args/ArgumentParser.h"
#include <htslib/hts.h>

#include "bam/BamRecordExtractor.h"
#include "npz/NpzWriter.h"
#include "pod5/Pod5Catalog.h"
#include "utils/Utils.h"
#include "worker/ModeRunner.h"
#include "worker/WorkerAssignment.h"

using namespace std;
using namespace deeprm;

// mimalloc purge delay in milliseconds. The default of 10 ms hands freed pages back to the OS
// (madvise) before the workers reuse them; at 512 workers that madvise/page-fault pairing
// serialises on mmap_lock and cost 40% runtime.
// At 1.2 s nearly every freed page is reused first, so purges rarely happen.
constexpr long kMimallocPurgeDelayMs = 1200;

// True for a BGZF-compressed BAM, and for a file htslib cannot open (the file reader then reports why).
static bool is_bgzf_bam(const string& path)
{
  htsFile* f = hts_open(path.c_str(), "r");
  if (!f)
    return true;
  const htsFormat* format = hts_get_format(f);
  const bool bgzf_bam = format->format == htsExactFormat::bam && format->compression == htsCompression::bgzf;
  hts_close(f);
  return bgzf_bam;
}

// True when `path` is a terminal (for example /dev/stdin with nothing piped in). Opening a terminal reads nothing.
static bool is_terminal(const string& path)
{
  int fd = open(path.c_str(), O_RDONLY | O_NOCTTY | O_NONBLOCK);
  if (fd < 0)
    return false;  // the stream reader opens it again and says why it cannot
  const bool terminal = isatty(fd);
  close(fd);
  return terminal;
}

int main(int argc, char* argv[])
{
  // A parent that ignores SIGCHLD passes that on through exec, and then waitpid() cannot read the exit status of the
  // --bam-cmd command, so a command that failed would look as if it had exited cleanly.
  signal(SIGCHLD, SIG_DFL);
  auto start_time = chrono::high_resolution_clock::now();
  log_start_time();

  ArgumentParser parser;
  Arguments args = parser.parse(argc, argv);
  parser.validate_arguments();
  // File mode seeks by BGZF virtual offset, so it needs a BGZF-compressed BAM in a regular file. A path that is not
  // a regular file (/dev/stdin, <(command), a named pipe) is read as a stream without being opened first, which
  // would use up its first bytes; any other file htslib reads (a SAM file, as 1.1.3 accepted) is read as a stream
  // too.
  struct stat st;
  if (!args.streaming && stat(args.bam_path.c_str(), &st) == 0 && !S_ISREG(st.st_mode)) {
    // A terminal would be waited on with no message, as -b - would (refused in ArgumentParser). Only a character
    // device can be one; opening a pipe to look would wait for its writer.
    if (S_ISCHR(st.st_mode) && is_terminal(args.bam_path)) {
      cerr << "Error: -b " << args.bam_path << " is a terminal, not a BAM; pipe a BAM in (e.g. dorado ... | ...) or "
              "use --bam-cmd\n";
      return 1;
    }
    log_info() << args.bam_path << " is not a regular file: reading it as a stream" << endl;
    args.streaming = true;
  } else if (!args.streaming && !is_bgzf_bam(args.bam_path)) {
    log_info() << args.bam_path << " is not a BGZF-compressed BAM: reading it as a stream" << endl;
    args.streaming = true;
  }

  // Set before any allocation-heavy work so it covers the whole run. mi_option_set takes precedence
  // over the MIMALLOC_PURGE_DELAY environment variable.
  mi_option_set(mi_option_purge_delay, kMimallocPurgeDelayMs);

  struct rlimit rlim;
  if (getrlimit(RLIMIT_NOFILE, &rlim) == 0) {
    rlim.rlim_cur = rlim.rlim_max;
    if (setrlimit(RLIMIT_NOFILE, &rlim) == 0) {
      log_info() << "File descriptor limit set to: " << rlim.rlim_cur << endl;
    } else {
      log_err() << "Warning: Failed to set file descriptor limit" << endl;
    }
  }

  log_info() << "Started DeepRM Preprocessing" << endl;

  Pod5Catalog catalog(args.pod5_path);
  if (catalog.files().empty()) {
    log_err() << "No POD5 files found in directory: " << args.pod5_path << endl;
    return 1;
  }
  log_info() << "Found " << catalog.files().size() << " POD5 files" << endl;

  // The BAM side (dispatcher read loop, or the file-mode voffset harvest)
  // runs concurrently with the POD5 metadata scan; both are needed before
  // reads can be routed to workers.
  unique_ptr<ModeRunner> runner = ModeRunner::create(args);
  runner->start_bam();
  catalog.scan_async(args.cpu_count);
  catalog.join();

  WorkerAssignment assignment = runner->assign(catalog);
  log_info() << "[MEM] POD5 assigned and indexed: " << get_rss_mb() << " MB" << endl;

  // Every reason this run's output must not be trusted; the run exits non-zero when there is one, so
  // nothing downstream mistakes partial output for a complete one.
  vector<string> causes;
  if (size_t bad_pod5 = catalog.failed_file_count(); bad_pod5 > 0) {
    log_err() << bad_pod5 << " POD5 files could not be read in full, so their reads are missing" << endl;
    causes.push_back(to_string(bad_pod5) + " POD5 files could not be read");
  }
  // False when a worker could not read its input (POD5 open, signal read, the BAM, or a
  // file-mode BAM re-read) or the --bam-cmd producer failed.
  if (!runner->run(assignment))
    causes.push_back("the BAM or a read's signal could not be read in full, or the BAM producer failed");
  if (long unwritten = NpzWriter::files_not_written.load(); unwritten > 0) {
    log_err() << unwritten << " output files could not be written in full" << endl;
    causes.push_back("output files could not be written");
  }

  if (long no_md = BamRecordExtractor::missing_md.load(); no_md > 0) {
    log_err() << no_md << " BAM records have no MD tag, so their reference bases are unknown; align with MD "
              << "tags (for example samtools calmd)" << endl;
    causes.push_back("BAM records without an MD tag");
  }
  if (long bad_md = BamRecordExtractor::invalid_md.load(); bad_md > 0) {
    log_err() << bad_md << " BAM records have an MD tag that is not a string or does not fit their CIGAR" << endl;
    causes.push_back("BAM records with an invalid MD tag");
  }
  if (long past = Utils::move_past_signal.load(); past > 0) {
    log_err() << "Skipped " << past << " reads whose move table is longer than their signal (do the BAM and the POD5 "
              << "files belong together?)" << endl;
  }
  if (long skipped = BamRecordExtractor::stride_mismatch.load(); skipped > 0) {
    log_err() << "Skipped " << skipped << " reads due to mv-tag stride mismatch with --sampling="
              << args.sampling << endl;
  }

  auto end_time = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
  double runtime_minutes = duration.count() / 60000.0;

  if (!causes.empty()) {
    string joined;
    for (const auto& c : causes)
      joined += (joined.empty() ? "" : "; ") + c;
    log_err() << "DeepRM Preprocessing FAILED (" << joined << "); discard this run's output" << endl;
    return 1;
  }

  log_info() << "Finished DeepRM Preprocessing" << endl;
  log_info() << fixed << setprecision(2)
      << "Total runtime: " << runtime_minutes << " minutes" << endl;
  return 0;
}
