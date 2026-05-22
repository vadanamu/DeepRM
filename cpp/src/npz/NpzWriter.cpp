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

#include "NpzWriter.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <uuid/uuid.h>
#include <zlib.h>

// =====================================================================================
// NpzFileSession: single-pass NPZ (= ZIP container of .npy files) writer.
//
// Replaces the cnpy::npz_save("a", ...) usage pattern, which opens/closes the file
// and rewrites the ZIP central directory once per array (so 8 arrays = 8 file opens
// and 8 cumulative central-directory rewrites). Beyond the syscall overhead, cnpy
// also allocates a fresh ~80 MB buffer per call; with 256 workers, this triggered
// a page-fault storm that serialized on the per-process mmap_lock (rwsem).
//
// This implementation:
//   - opens the file once, appends each array's local header + compressed payload,
//     then writes the central directory + EOCD record at the end.
//   - reuses caller-provided scratch buffers across chunks, so page faults occur
//     only once per worker (first chunk) instead of every chunk × every array.
//
// Output compatibility (matters because deeprm has multiple NPZ producers
// — this C++ writer plus several Python paths that use numpy.savez_compressed
// in inference_preprocess_python.py / pileup_deeprm.py / train_compile.py
// — and a single set of consumers (inference_dataloader.py, train.py, etc.)
// that must handle every producer's output identically via np.load):
//   - cnpy::npz_save:           byte-identical (uses the same cnpy NPY header).
//   - numpy.savez_compressed:   read-compatible — np.load returns arrays that
//                               compare element-wise equal, so downstream
//                               readers cannot distinguish C++ output from
//                               Python output. The .npz file itself differs
//                               by ~48 B per entry because cnpy pads the
//                               NPY header to a 16-byte boundary while numpy
//                               pads to 64 bytes (both are valid per the
//                               .npy format spec).
// =====================================================================================

namespace {

template <typename T>
inline void append_le(vector<char>& v, T value)
{
  for (size_t i = 0; i < sizeof(T); ++i)
    v.push_back(static_cast<char>((value >> (i * 8)) & 0xFF));
}

inline void append_str(vector<char>& v, const string& s)
{
  v.insert(v.end(), s.begin(), s.end());
}

// DOS time/date encoding for the ZIP "last modified" fields.
// time:  bits [15:11]=hour, [10:5]=min, [4:0]=sec/2  (all zero is valid 00:00:00)
// date:  bits [15:9]=year-1980, [8:5]=month(1-12), [4:0]=day(1-31)
//        date=0x0000 would mean month=0/day=0, which is invalid; use the proper
//        epoch encoding 1980-01-01 = (0<<9)|(1<<5)|1 = 0x0021 to match the
//        default Python zipfile / numpy.savez_compressed output and stay valid.
constexpr uint16_t kDosTimeZero  = 0x0000;  // 00:00:00
constexpr uint16_t kDosDateEpoch = 0x0021;  // 1980-01-01

// One ZIP entry's worth of metadata, accumulated in memory until close().
struct NpzEntry {
  string name;
  uint32_t crc;
  uint32_t comp_size;
  uint32_t uncomp_size;
  uint32_t local_header_offset;
  uint16_t method;  // 8 = deflate, 0 = stored
};

class NpzFileSession {
public:
  // uncomp_buf / comp_buf are caller-owned and reused across chunks.
  NpzFileSession(const string& filename,
                 vector<uint8_t>& uncomp_buf, vector<uint8_t>& comp_buf)
    : fp(fopen(filename.c_str(), "wb")),
      uncomp_buf(uncomp_buf), comp_buf(comp_buf), offset(0)
  {
    if (!fp) throw runtime_error("NpzFileSession: cannot open " + filename);
  }

  // close() may throw via checked_write(); swallow here so the destructor
  // never propagates an exception (UB during stack unwinding).
  ~NpzFileSession()
  {
    if (fp) {
      try { close(); }
      catch (...) {}
    }
  }

  template <typename T>
  void add_array(const string& base_name, const T* data, const vector<size_t>& shape)
  {
    // 1. Build the .npy header + raw data into a contiguous buffer.
    string fname = base_name + ".npy";
    vector<char> npy_header = cnpy::create_npy_header<T>(shape);
    size_t nels = accumulate(shape.begin(), shape.end(), size_t(1),
                             multiplies<size_t>());
    size_t uncomp_size = nels * sizeof(T) + npy_header.size();

    // ZIP (without ZIP64) constrains every entry size to 32 bits, and zlib's
    // uInt (avail_in / avail_out) is typically 32-bit as well. Refuse > 4 GiB
    // up front instead of silently truncating into a corrupt ZIP entry.
    if (uncomp_size > numeric_limits<uint32_t>::max())
      throw runtime_error("NpzFileSession: array exceeds 4 GiB "
                          "(ZIP64 not supported)");

    if (uncomp_buf.size() < uncomp_size) uncomp_buf.resize(uncomp_size);
    memcpy(uncomp_buf.data(), npy_header.data(), npy_header.size());
    memcpy(uncomp_buf.data() + npy_header.size(), data, nels * sizeof(T));

    // 2. CRC over the uncompressed payload (npy header + data).
    uint32_t crc = (uint32_t)crc32(0L, uncomp_buf.data(), uncomp_size);

    // 3. Compress with raw deflate (window=-15) so the output is a valid ZIP
    //    deflate stream (no zlib wrapper). compressBound() returns the
    //    worst-case bound for zlib-wrapped output; raw deflate output is at
    //    most that minus the 6-byte wrapper, so this is sufficient.
    size_t bound = compressBound(uncomp_size);
    if (comp_buf.size() < bound) comp_buf.resize(bound);
    size_t comp_size = 0;
    z_stream zs{};
    int init_rc = deflateInit2(&zs, Z_DEFAULT_COMPRESSION, Z_DEFLATED,
                               -15, 8, Z_DEFAULT_STRATEGY);
    if (init_rc == Z_OK) {
      zs.next_in = uncomp_buf.data();
      zs.avail_in = (uInt)uncomp_size;
      zs.next_out = comp_buf.data();
      zs.avail_out = (uInt)bound;
      int deflate_rc = deflate(&zs, Z_FINISH);
      if (deflate_rc == Z_STREAM_END) {
        comp_size = zs.total_out;
      } else {
        // compressBound() guarantees the output buffer is large enough, so
        // Z_BUF_ERROR / Z_STREAM_ERROR here indicate a real problem (memory
        // pressure, corrupt z_stream state). Mirror the deflateInit2 path
        // and log so the stored-mode fallback isn't silent.
        cerr << "NpzFileSession: deflate(Z_FINISH) failed (rc=" << deflate_rc
             << ") for '" << base_name << "', falling back to stored mode" << endl;
      }
      deflateEnd(&zs);
    } else {
      // deflateInit2 only fails on real resource problems (Z_MEM_ERROR etc.).
      // Falling back to stored mode silently would mask OOM in 256-worker runs,
      // so log it here and let the stored-mode path below take over.
      cerr << "NpzFileSession: deflateInit2 failed (rc=" << init_rc
           << ") for '" << base_name << "', falling back to stored mode" << endl;
    }
    bool use_compressed = (comp_size > 0 && comp_size < uncomp_size);
    uint32_t out_size = use_compressed ? (uint32_t)comp_size : (uint32_t)uncomp_size;
    uint16_t method = use_compressed ? 8 : 0;

    // 4. Remember this entry's placement; central directory writes happen in close().
    //    The ZIP filename length field is uint16; guard the cast even though our
    //    fixed array names are short (extension-safety).
    if (fname.size() > numeric_limits<uint16_t>::max())
      throw runtime_error("NpzFileSession: array name exceeds 64 KiB");

    // Without ZIP64, the local_header_offset and the EOCD's cd_offset are both
    // 32-bit fields. Verify the cumulative file position (this entry's start +
    // local header + payload) still fits, otherwise close() would emit a ZIP
    // with wrap-around offsets that readers would mis-locate. The 30 below is
    // the fixed local-header size.
    size_t entry_end = offset + 30 + fname.size() + out_size;
    if (entry_end > numeric_limits<uint32_t>::max())
      throw runtime_error("NpzFileSession: cumulative file size would exceed "
                          "4 GiB (ZIP64 not supported)");

    entries.push_back({fname, crc, out_size, (uint32_t)uncomp_size,
                       (uint32_t)offset, method});

    // 5. ZIP local file header (PK\x03\x04, 30 bytes + filename).
    vector<char> hdr;
    hdr.reserve(30 + fname.size());
    append_le<uint32_t>(hdr, 0x04034B50);     // local file header signature
    append_le<uint16_t>(hdr, 20);              // version needed to extract
    append_le<uint16_t>(hdr, 0);               // general purpose bit flag
    append_le<uint16_t>(hdr, method);
    append_le<uint16_t>(hdr, kDosTimeZero);    // last mod time (00:00:00)
    append_le<uint16_t>(hdr, kDosDateEpoch);   // last mod date (1980-01-01)
    append_le<uint32_t>(hdr, crc);
    append_le<uint32_t>(hdr, out_size);        // compressed size
    append_le<uint32_t>(hdr, (uint32_t)uncomp_size);
    append_le<uint16_t>(hdr, (uint16_t)fname.size());
    append_le<uint16_t>(hdr, 0);               // extra field length
    append_str(hdr, fname);
    checked_write(hdr.data(), hdr.size());
    offset += hdr.size();

    // 6. Payload (compressed or stored, depending on which is smaller).
    const void* payload = use_compressed ? (const void*)comp_buf.data()
                                         : (const void*)uncomp_buf.data();
    checked_write(payload, out_size);
    offset += out_size;
  }

  void close()
  {
    if (!fp) return;

    try {
      // ZIP (without ZIP64) caps the EOCD's entry count at uint16_t; guard the
      // cast in case future code adds many entries.
      if (entries.size() > numeric_limits<uint16_t>::max())
        throw runtime_error("NpzFileSession: too many entries for non-ZIP64 EOCD");

      // Central directory: one record per entry (PK\x01\x02, 46 bytes + filename).
      size_t cd_offset = offset;
      vector<char> cd;
      for (const auto& e : entries) {
        append_le<uint32_t>(cd, 0x02014B50);    // central dir signature
        append_le<uint16_t>(cd, 20);             // version made by
        append_le<uint16_t>(cd, 20);             // version needed
        append_le<uint16_t>(cd, 0);              // flags
        append_le<uint16_t>(cd, e.method);
        append_le<uint16_t>(cd, kDosTimeZero);   // mod time (00:00:00)
        append_le<uint16_t>(cd, kDosDateEpoch);  // mod date (1980-01-01)
        append_le<uint32_t>(cd, e.crc);
        append_le<uint32_t>(cd, e.comp_size);
        append_le<uint32_t>(cd, e.uncomp_size);
        append_le<uint16_t>(cd, (uint16_t)e.name.size());
        append_le<uint16_t>(cd, 0);              // extra length
        append_le<uint16_t>(cd, 0);              // comment length
        append_le<uint16_t>(cd, 0);              // disk number
        append_le<uint16_t>(cd, 0);              // internal file attrs
        append_le<uint32_t>(cd, 0);              // external file attrs
        append_le<uint32_t>(cd, e.local_header_offset);
        append_str(cd, e.name);
      }
      checked_write(cd.data(), cd.size());

      // End of Central Directory record (PK\x05\x06, 22 bytes).
      vector<char> footer;
      append_le<uint32_t>(footer, 0x06054B50);  // EOCD signature
      append_le<uint16_t>(footer, 0);            // disk number
      append_le<uint16_t>(footer, 0);            // disk where CD starts
      append_le<uint16_t>(footer, (uint16_t)entries.size());
      append_le<uint16_t>(footer, (uint16_t)entries.size());
      append_le<uint32_t>(footer, (uint32_t)cd.size());
      append_le<uint32_t>(footer, (uint32_t)cd_offset);
      append_le<uint16_t>(footer, 0);            // comment length
      checked_write(footer.data(), footer.size());
    } catch (...) {
      // Ensure FD is released even on write failure (e.g., disk full); without
      // this, repeated I/O errors in 256-worker runs would leak descriptors.
      fclose(fp);
      fp = nullptr;
      throw;
    }

    fclose(fp);
    fp = nullptr;
  }

private:
  FILE* fp;
  vector<uint8_t>& uncomp_buf;  // reused across chunks (caller-owned)
  vector<uint8_t>& comp_buf;    // reused across chunks (caller-owned)
  size_t offset;                // current write position within file
  vector<NpzEntry> entries;

  void checked_write(const void* data, size_t size)
  {
    if (fwrite(data, 1, size, fp) != size)
      throw runtime_error("NpzFileSession: write failed");
  }
};

}  // namespace

NpzWriter::NpzWriter(const string& output_path, int chunk_size, int worker_id)
  : output_path(output_path), chunk_size(chunk_size), worker_id(worker_id),
    processing_unit_id(1), chunk_id(0)
{
  filesystem::create_directories(output_path);
}

string NpzWriter::generate_filename(bool is_last_processing_unit, bool is_last_chunk)
{
  string filename = output_path + "/" + to_string(worker_id) + "-";

  if (is_last_processing_unit) {
    filename += "last-";
  } else {
    filename += to_string(processing_unit_id) + "-";
  }

  if (is_last_chunk) {
    filename += "last.npz";
  } else {
    filename += to_string(chunk_id) + ".npz";
  }

  return filename;
}

void NpzWriter::save_chunk(const vector<ProcessedRecord>& records,
                           size_t offset, size_t count, const string& filename)
{
  if (count == 0) return;

  // Determine max dimensions across this chunk's records (rows are padded to max).
  size_t max_segment_len = 0, max_signal_len = 0, max_kmer_len = 0;
  size_t max_dwell_motor_len = 0, max_dwell_pore_len = 0, max_bq_len = 0;

  for (size_t i = offset; i < offset + count; ++i) {
    const auto& r = records[i];
    max_segment_len = max(max_segment_len, r.segment_len_arr.size());
    max_signal_len = max(max_signal_len, r.signal_token.size());
    max_kmer_len = max(max_kmer_len, r.kmer_token.size());
    max_dwell_motor_len = max(max_dwell_motor_len, r.dwell_motor_token.size());
    max_dwell_pore_len = max(max_dwell_pore_len, r.dwell_pore_token.size());
    max_bq_len = max(max_bq_len, r.bq_token.size());
  }

  // Reuse member flat arrays (capacity preserved across calls).
  flat_segment_len.assign(count * max_segment_len, 0);
  flat_signal.assign(count * max_signal_len, 0.0f);
  flat_kmer.assign(count * max_kmer_len, 0);
  flat_dwell_motor.assign(count * max_dwell_motor_len, 0.0f);
  flat_dwell_pore.assign(count * max_dwell_pore_len, 0.0f);
  flat_bq.assign(count * max_bq_len, 0);
  label_ids.assign(count, 0);
  read_ids.assign(count * 2, 0);

  for (size_t i = 0; i < count; ++i) {
    const auto& r = records[offset + i];
    size_t seg_off = i * max_segment_len;
    size_t sig_off = i * max_signal_len;
    size_t kmer_off = i * max_kmer_len;
    size_t dm_off = i * max_dwell_motor_len;
    size_t dp_off = i * max_dwell_pore_len;
    size_t bq_off = i * max_bq_len;

    copy(r.segment_len_arr.begin(), r.segment_len_arr.end(), flat_segment_len.begin() + seg_off);
    // signal_token is double, flat_signal is float — convert
    for (size_t j = 0; j < r.signal_token.size(); ++j)
      flat_signal[sig_off + j] = static_cast<float>(r.signal_token[j]);
    copy(r.kmer_token.begin(), r.kmer_token.end(), flat_kmer.begin() + kmer_off);
    copy(r.dwell_motor_token.begin(), r.dwell_motor_token.end(), flat_dwell_motor.begin() + dm_off);
    copy(r.dwell_pore_token.begin(), r.dwell_pore_token.end(), flat_dwell_pore.begin() + dp_off);
    copy(r.bq_token.begin(), r.bq_token.end(), flat_bq.begin() + bq_off);
    label_ids[i] = r.label_id;

    // Convert UUID string to two int64 halves. uuid_t is unsigned char[16] with
    // only 1-byte alignment, so reinterpret_cast<int64_t*> would be UB on
    // strict-aliasing/aligned-access platforms; use memcpy and let the
    // compiler turn it into two MOVs.
    uuid_t uuid;
    if (uuid_parse(r.read_id.c_str(), uuid) == 0) {
      memcpy(&read_ids[i * 2], uuid, sizeof(uuid_t));
    } else {
      // Leave (0, 0) but log so silent corruption (every row showing the same
      // null UUID) doesn't go unnoticed downstream.
      cerr << "NpzWriter: invalid UUID '" << r.read_id
           << "' (worker " << worker_id << "), writing zero read_id" << endl;
    }
  }

  // Write all 8 arrays in a single NPZ pass — one fopen, one fclose, one
  // central-directory write (vs cnpy's 8 of each).
  try {
    NpzFileSession sess(filename, npz_uncomp_buf, npz_comp_buf);
    sess.add_array("segment_len_arr",   flat_segment_len.data(), {count, max_segment_len});
    sess.add_array("signal_token",      flat_signal.data(),      {count, max_signal_len});
    sess.add_array("kmer_token",        flat_kmer.data(),        {count, max_kmer_len});
    sess.add_array("dwell_motor_token", flat_dwell_motor.data(), {count, max_dwell_motor_len});
    sess.add_array("dwell_pore_token",  flat_dwell_pore.data(),  {count, max_dwell_pore_len});
    sess.add_array("bq_token",          flat_bq.data(),          {count, max_bq_len});
    sess.add_array("label_id",          label_ids.data(),        {count});
    sess.add_array("read_id",           read_ids.data(),         {count, (size_t)2});
    sess.close();
  } catch (const exception& e) {
    cerr << "Error saving NPZ file " << filename << ": " << e.what() << endl;
    // Avoid leaving a partial NPZ on disk: a downstream NpzFileSession
    // destructor on the throw path may have written a (possibly truncated)
    // file via fopen("wb") + close(), which would otherwise be picked up
    // by readers as a real chunk.
    error_code ec;
    filesystem::remove(filename, ec);
  }
}

void NpzWriter::add_records(vector<ProcessedRecord>&& records)
{
  buffer.insert(buffer.end(), make_move_iterator(records.begin()),
                make_move_iterator(records.end()));

  // Save complete chunks directly from buffer using offset
  size_t offset = 0;
  while (buffer.size() - offset >= static_cast<size_t>(chunk_size)) {
    string filename = generate_filename(false, false);
    save_chunk(buffer, offset, chunk_size, filename);
    chunk_id++;
    offset += chunk_size;
  }

  // Keep only remaining records
  if (offset > 0) {
    buffer.erase(buffer.begin(), buffer.begin() + offset);
  }
}

void NpzWriter::flush()
{
  if (!buffer.empty()) {
    size_t offset = 0;
    while (buffer.size() - offset >= static_cast<size_t>(chunk_size)) {
      string filename = generate_filename(true, false);
      save_chunk(buffer, offset, chunk_size, filename);
      chunk_id++;
      offset += chunk_size;
    }

    // Save remaining records
    if (offset < buffer.size()) {
      string filename = generate_filename(true, true);
      save_chunk(buffer, offset, buffer.size() - offset, filename);
    }
    buffer.clear();
  }
}

void NpzWriter::increment_processing_unit()
{
  processing_unit_id++;
  chunk_id = 0;
}

void NpzWriter::reset_chunk_id()
{
  chunk_id = 0;
}
