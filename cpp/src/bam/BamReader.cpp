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

#include "bam/BamReader.h"

#include <iostream>
#include <utility>

#include <htslib/bgzf.h>

#include "bam/BamRecordExtractor.h"
#include "utils/Utils.h"

namespace deeprm {
  BamReader::BamReader(const string& path, int filter_flag)
    : bam_path(path), filter_flag(filter_flag), ranged(false), v_lo(0), v_hi(0)
  {
  }

  BamReader::~BamReader()
  {
  }

  vector<pair<Uuid128, uint64_t>> BamReader::harvest_offsets(int process_id,
                                                            int num_processes,
                                                            int num_threads)
  {
    vector<pair<Uuid128, uint64_t>> out;

    samFile* bam_file = sam_open(bam_path.c_str(), "r");
    if (!bam_file) {
      cerr << "Failed to open BAM file: " << bam_path << endl;
      read_failed = true;
      return out;
    }
    // File mode seeks by BGZF virtual offset, so it needs a BGZF-compressed BAM; a SAM or CRAM file
    // would crash it.
    const htsFormat* format = hts_get_format(bam_file);
    if (format->format != htsExactFormat::bam || format->compression != htsCompression::bgzf) {
      if (process_id == 0)
        cerr << "Not a BAM file: " << bam_path << " (file mode needs a BGZF-compressed BAM; convert it with "
             << "samtools view -b, or stream it with -b -)" << endl;
      read_failed = true;
      sam_close(bam_file);
      return out;
    }
    if (num_threads > 1)
      hts_set_threads(bam_file, num_threads);

    sam_hdr_t* header = sam_hdr_read(bam_file);
    if (!header) {
      cerr << "Failed to read BAM header" << endl;
      read_failed = true;
      sam_close(bam_file);
      return out;
    }
    // Reading the header checks the end of a seekable BAM for the BGZF end-of-file block. Without
    // it the file was cut off (a writer that stopped between two blocks leaves it so) and its last
    // records are missing, although every block that is there reads cleanly.
    if (bam_file->fp.bgzf->no_eof_block) {
      if (process_id == 0)
        cerr << "The BAM file has no BGZF end-of-file block, so it is cut off (or still being written): "
             << bam_path << endl;
      read_failed = true;
      sam_hdr_destroy(header);
      sam_close(bam_file);
      return out;
    }

    bam1_t* read = bam_init1();
    BGZF* bgzf = bam_file->fp.bgzf;

    // The voffset must be captured BEFORE sam_read1 consumes the record --
    // afterwards bgzf_tell points at the NEXT record. Only the checks that
    // read bam1_core_t directly are applied here (empty SEQ, -g flag mask);
    // qcut / mv / MD checks need the decoded record and stay in
    // extract_record, which phase 2 runs on the
    // re-read copy, so the final record set is decided by exactly the same
    // predicate as before. Streaming's dispatcher pre-filter tests the same
    // pair, so both modes discard a record at the earliest point it can be
    // recognised.
    auto harvest_one = [&](uint64_t voffset) {
      if (!BamRecordExtractor::passes_prefilter(read, filter_flag))
        return;
      out.emplace_back(uuid128_from_str(BamRecordExtractor::parent_id(read)), voffset);
    };

    if (ranged) {
      // Fast path: seek to this reader's range and consume records until
      // the first whose START voffset is >= v_hi. The bound is tested BEFORE
      // reading, so a record straddling the boundary belongs to exactly one
      // range -- the union then covers every record once.
      if (bgzf_seek(bgzf, static_cast<int64_t>(v_lo), SEEK_SET) < 0) {
        cerr << "Failed to seek BAM to voffset " << v_lo << endl;
        read_failed = true;
      } else {
        while (true) {
          uint64_t v = static_cast<uint64_t>(bgzf_tell(bgzf));
          if (v_hi != 0 && v >= v_hi)
            break;
          int r = sam_read1(bam_file, header, read);
          if (r < -1) {
            cerr << "Failed to read a BAM record (error " << r << ")" << endl;
            read_failed = true;
          }
          if (r < 0)
            break;
          harvest_one(v);
        }
      }
    } else {
      int read_idx = 0;
      while (true) {
        uint64_t v = static_cast<uint64_t>(bgzf_tell(bgzf));
        int r = sam_read1(bam_file, header, read);
        if (r < -1) {
          cerr << "Failed to read a BAM record (error " << r << ")" << endl;
          read_failed = true;
        }
        if (r < 0)
          break;
        if (read_idx % num_processes == process_id)
          harvest_one(v);
        read_idx++;
      }
    }

    bam_destroy1(read);
    sam_hdr_destroy(header);
    sam_close(bam_file);

    return out;
  }
} // namespace deeprm
