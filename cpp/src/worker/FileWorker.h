/**
 * Copyright (C) 2026 Genome4me Incorporated - All Rights Reserved.
 * This software, including its source code, embedded concepts, and associated
 * documentation, is proprietary to Genome4me Incorporated and is protected
 * under trade secret and copyright law. Unauthorized use, copying, modification,
 * distribution, or disclosure to third parties, in whole or in part, is
 * strictly prohibited unless prior written permission is granted by Genome4me
 * Incorporated. Any such unauthorized actions constitute an infringement of
 * the intellectual property rights of Genome4me Incorporated. For licensing
 * inquiries or permissions, please contact Genome4me Incorporated.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <htslib/sam.h>

#include "args/ArgumentParser.h"
#include "npz/NpzWriter.h"
#include "pod5/Pod5Reader.h"
#include "utils/Uuid128.h"

using namespace std;

namespace deeprm {
  class RecordMerger;
  struct BamRecord;

  // File-mode phase 2 for one worker: walks its POD5 records in file order and,
  // for each one that the harvest matched, re-reads the BAM record(s) by
  // voffset, runs the shared extractor and merges. The counterpart of
  // StreamWorker, which takes the same records from a dispatcher queue
  // instead; both share RecordMerger, NpzWriter and Pod5BatchCache.
  //
  // One instance per worker thread, constructed and run on that thread so the
  // writer and reader buffers are allocated where they are used. The
  // destructor closes everything in dependency order (batch cache before the
  // readers it refers into), so every exit path is leak-free.
  class FileWorker {
  public:
    // bam_index maps parent read_id -> voffsets of its BAM records; it is this
    // worker's alone and entries are erased as they are consumed. failed is
    // shared with every worker and read by the runner after the join.
    FileWorker(int id, const Arguments& args, const vector<string>& pod5_files,
               const vector<Pod5RecordMeta>& pod5_meta_records,
               unordered_map<Uuid128, vector<uint64_t>>& bam_index,
               atomic<bool>& failed);
    ~FileWorker();
    FileWorker(const FileWorker&) = delete;
    FileWorker& operator=(const FileWorker&) = delete;

    void run();

  private:
    int worker_id;
    const Arguments& args;
    // Catalog file list that the records' file_index refers to.
    const vector<string>& pod5_files;
    const vector<Pod5RecordMeta>& pod5_meta_records;
    unordered_map<Uuid128, vector<uint64_t>>& bam_index;
    atomic<bool>& failed;

    // Per-worker POD5 readers by file_index, opened lazily on first use; a
    // failed open stays null and is not retried. A worker owns a contiguous
    // slice of files, so only a few entries are ever non-null.
    vector<Pod5FileReader_t*> readers;
    vector<bool> reader_tried;
    long reader_opens = 0;
    // Read-table batch reused across consecutive reads of one file and batch;
    // in file mode the records arrive in POD5 order, so this is nearly every
    // read. It refers into a reader above and is released before they close.
    Pod5BatchCache batch_cache;

    // Own BAM handle for voffset fetches; readers are not shareable across
    // threads any more than POD5 readers are. No hts thread pool: each fetch
    // inflates a single BGZF block, which threads cannot help.
    samFile* bam_file;
    sam_hdr_t* bam_header;
    bam1_t* bam_read;

    NpzWriter writer;

    // Opens (or returns) the reader for a catalog file index; null when it
    // cannot be opened, which has already been reported and flagged.
    Pod5FileReader_t* reader_for(uint32_t file_index);
    // Re-reads every BAM record of one matched read and merges it. False
    // means the run is lost (BAM re-read or signal read failed).
    bool process_read(Pod5RecordMeta meta, const vector<uint64_t>& voffsets,
                      RecordMerger& merger, BamRecord& bam_rec);
  };
}  // namespace deeprm

