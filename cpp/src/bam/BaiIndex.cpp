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

#include "bam/BaiIndex.h"

#include <sys/stat.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>

#include <htslib/kstring.h>
#include <htslib/sam.h>

#include "utils/Utils.h"

namespace deeprm {
  namespace {
    // Read exactly n bytes or throw. A short read must not pass silently: on
    // a truncated .bai the partition would then be built from an INCOMPLETE
    // offset list -- a wrong split, not a refusal.
    void read_exact(FILE* fh, void* buf, size_t n, const char* what)
    {
      if (fread(buf, 1, n, fh) != n)
        throw runtime_error(string("truncated .bai: short read at ") + what);
    }

    void skip_exact(FILE* fh, long n, const char* what)
    {
      if (fseek(fh, n, SEEK_CUR) != 0)
        throw runtime_error(string("truncated .bai: cannot skip ") + what);
    }
  } // namespace

  pair<bool, string> BaiIndex::fast_path_ok(const string& bam_path)
  {
    string bai = bam_path + ".bai";
    struct stat st_bai;
    if (stat(bai.c_str(), &st_bai) != 0) {
      struct stat st_csi;
      if (stat((bam_path + ".csi").c_str(), &st_csi) == 0)
        return {false, "only a .csi index (no BAI linear index)"};
      return {false, "no .bai index"};
    }
    struct stat st_bam;
    if (stat(bam_path.c_str(), &st_bam) != 0)
      return {false, "cannot stat BAM"};
    // The mtime check is the one that actually catches a stale index; SO:
    // describes the header, so it passes unchanged for an index left over
    // from an earlier version of a still-coordinate-sorted file.
    if (st_bai.st_mtim.tv_sec < st_bam.st_mtim.tv_sec ||
        (st_bai.st_mtim.tv_sec == st_bam.st_mtim.tv_sec &&
         st_bai.st_mtim.tv_nsec < st_bam.st_mtim.tv_nsec)) {
      return {false, "the .bai is OLDER than the BAM (stale index); re-run "
                     "samtools index, or accept the modulo scan"};
    }

    samFile* fp = sam_open(bam_path.c_str(), "r");
    if (!fp)
      return {false, "BAM unreadable"};
    sam_hdr_t* hdr = sam_hdr_read(fp);
    if (!hdr) {
      sam_close(fp);
      return {false, "header unreadable"};
    }
    kstring_t ks = KS_INITIALIZE;
    bool sorted = sam_hdr_find_tag_hd(hdr, "SO", &ks) == 0 && ks.s &&
                  string(ks.s) == "coordinate";
    string so = ks.s ? ks.s : "";
    ks_free(&ks);
    sam_hdr_destroy(hdr);
    sam_close(fp);
    if (!sorted)
      return {false, "header SO='" + so + "', not coordinate"};
    return {true, "ok"};
  }

  vector<uint64_t> BaiIndex::linear_offsets(const string& bam_path)
  {
    string bai = bam_path + ".bai";
    vector<uint64_t> offs;
    FILE* fh = fopen(bai.c_str(), "rb");
    if (!fh)
      return offs;
    try {
      char magic[4];
      read_exact(fh, magic, 4, "magic");
      if (string(magic, 4) != string("BAI\x01", 4))
        throw runtime_error("bad magic");
      int32_t n_ref = 0;
      read_exact(fh, &n_ref, 4, "n_ref");
      if (n_ref < 0)
        throw runtime_error("negative n_ref");
      for (int32_t r = 0; r < n_ref; ++r) {
        int32_t n_bin = 0;
        read_exact(fh, &n_bin, 4, "n_bin");
        for (int32_t b = 0; b < n_bin; ++b) {
          skip_exact(fh, 4, "bin id");
          int32_t n_chunk = 0;
          read_exact(fh, &n_chunk, 4, "n_chunk");
          skip_exact(fh, 16L * n_chunk, "chunks");
        }
        int32_t n_intv = 0;
        read_exact(fh, &n_intv, 4, "n_intv");
        if (n_intv > 0) {
          size_t base = offs.size();
          offs.resize(base + n_intv);
          read_exact(fh, offs.data() + base, 8UL * n_intv, "linear index");
        }
      }
      // The only thing allowed after the last reference is the optional
      // 8-byte n_no_coor.
      char tail[9];
      size_t extra = fread(tail, 1, sizeof(tail), fh);
      if (extra != 0 && extra != 8)
        throw runtime_error("unexpected trailing bytes");
    } catch (const exception& e) {
      log_err() << "Rejected " << bai << " (" << e.what()
                << "); falling back to the modulo scan, which is correct and "
                << "merely slower" << endl;
      fclose(fh);
      return {};
    }
    fclose(fh);
    sort(offs.begin(), offs.end());
    offs.erase(unique(offs.begin(), offs.end()), offs.end());
    // Drop the zero entries (windows before the first alignment).
    offs.erase(offs.begin(),
               find_if(offs.begin(), offs.end(),
                       [](uint64_t v) { return v > 0; }));
    return offs;
  }

  vector<pair<uint64_t, uint64_t>> BaiIndex::plan_ranges(
      const vector<uint64_t>& offsets, int n_parts)
  {
    vector<pair<uint64_t, uint64_t>> ranges;
    if (offsets.size() < 2)
      return ranges;
    // A BGZF virtual offset is (block_offset << 16) | within_block, so
    // v >> 16 is the compressed file position -- the phase-1 work metric
    // (balancing by read count left a 1.90x worker-time spread in the Python
    // preprocessing).
    vector<uint64_t> pos(offsets.size());
    for (size_t i = 0; i < offsets.size(); ++i)
      pos[i] = offsets[i] >> 16;
    if (pos.back() <= pos.front())
      return ranges;
    uint64_t total = pos.back() - pos.front();
    size_t n = offsets.size();
    int parts = max(1, min(n_parts, static_cast<int>(n) - 1));
    vector<size_t> cuts;
    for (int k = 1; k < parts; ++k) {
      uint64_t target = pos.front() +
                        static_cast<uint64_t>(k) * total / parts;
      size_t idx = lower_bound(pos.begin(), pos.end(), target) - pos.begin();
      cuts.push_back(min(max(idx, size_t(1)), n - 1));
    }
    sort(cuts.begin(), cuts.end());
    cuts.erase(unique(cuts.begin(), cuts.end()), cuts.end());
    vector<size_t> bounds;
    bounds.push_back(0);
    bounds.insert(bounds.end(), cuts.begin(), cuts.end());
    bounds.push_back(n - 1);
    for (size_t i = 0; i + 1 < bounds.size(); ++i) {
      uint64_t lo = offsets[bounds[i]];
      uint64_t hi = offsets[bounds[i + 1]];
      if (hi > lo)  // drop empty blocks (a huge reference can straddle a target)
        ranges.push_back({lo, hi});
    }
    // The final range runs to EOF: hi == 0 means "no upper bound".
    if (!ranges.empty())
      ranges.back().second = 0;
    return ranges;
  }
} // namespace deeprm
