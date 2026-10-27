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

#include "npz/NpzWriter.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <libdeflate.h>

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
//   - lays each array out contiguously in a caller-owned staging buffer (npy
//     header, then the emitted prefix of every row) and compresses it in one
//     libdeflate call into a second caller-owned buffer; the two transients
//     are the uncompressed array and its compressed payload.
//   - opens the file once and only appends: complete local header (CRC and
//     sizes are known once compression finishes), then payload, then the
//     central directory + EOCD record at the end. No seeks, strictly
//     sequential.
//   - reuses both staging buffers and the compressor across chunks, so page
//     faults occur once per worker (first chunk) instead of every chunk ×
//     every array.
//
// Output compatibility (matters because deeprm has multiple NPZ producers
// — this C++ writer plus several Python paths that use numpy.savez_compressed
// in inference_preprocess_python.py / pileup_deeprm.py / train_compile.py
// — and a single set of consumers (inference_dataloader.py, train.py, etc.)
// that must handle every producer's output identically via np.load):
//   - cnpy::npz_save:           read-compatible; the NPY headers are built by
//                               the same cnpy helper, but the deflate stream
//                               (libdeflate vs zlib) and the ZIP DOS date
//                               field differ, so the files are not
//                               byte-identical.
//   - numpy.savez_compressed:   read-compatible — np.load returns arrays that
//                               compare element-wise equal, so downstream
//                               readers cannot distinguish C++ output from
//                               Python output. The .npz bytes differ: the
//                               deflate streams are from different
//                               implementations, and cnpy pads the NPY header
//                               to a 16-byte boundary while numpy pads to 64
//                               (both are valid per the .npy format spec).
// =====================================================================================

namespace {

// libdeflate level 6 is libdeflate's default and matches zlib's default (Z_DEFAULT_COMPRESSION),
// which the python reference (np.savez_compressed) uses, so the output size stays comparable.
constexpr int kNpzDeflateLevel = 6;

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
  // rawbuf, zbuf and the compressor are caller-owned and reused across
  // chunks: rawbuf stages one array uncompressed, zbuf its compressed form.
  NpzFileSession(const string& filename, vector<uint8_t>& rawbuf,
                 vector<uint8_t>& zbuf, libdeflate_compressor* compressor)
    : fp(fopen(filename.c_str(), "wb")), rawbuf(rawbuf), zbuf(zbuf),
      compressor(compressor), offset(0)
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

  // Rows are produced by `emit`, laid out contiguously in rawbuf (npy header
  // first), then compressed in one libdeflate call into zbuf. Once the array
  // is done its CRC and sizes are known, so the local header is written
  // complete, followed by the payload — the file is written strictly front to
  // back.
  //
  // `emit(sink)` must call sink(ptr, bytes) for each row in order. If
  // compression fails (output bound exceeded) or does not shrink the data,
  // rawbuf itself is written as the entry (stored mode); no replay is needed.
  //
  // Output is a raw deflate stream (ZIP method 8) exactly as with zlib, so
  // np.load reads it unchanged; only the byte sequence differs from zlib's.
  template <typename T, typename Emit>
  void add_array_stream(const string& base_name, const vector<size_t>& shape,
                        Emit&& emit)
  {
    string fname = base_name + ".npy";
    vector<char> npy_header = cnpy::create_npy_header<T>(shape);
    size_t nels = accumulate(shape.begin(), shape.end(), size_t(1),
                             multiplies<size_t>());
    size_t uncomp_size = nels * sizeof(T) + npy_header.size();

    // ZIP (without ZIP64) constrains every entry size to 32 bits. Refuse
    // > 4 GiB up front instead of silently truncating into a corrupt ZIP entry.
    if (uncomp_size > numeric_limits<uint32_t>::max())
      throw runtime_error("NpzFileSession: array exceeds 4 GiB "
                          "(ZIP64 not supported)");

    // The ZIP filename length field is uint16; guard the cast even though our
    // fixed array names are short (extension-safety).
    if (fname.size() > numeric_limits<uint16_t>::max())
      throw runtime_error("NpzFileSession: array name exceeds 64 KiB");

    // 1. Lay the entry out contiguously: npy header, then each emitted row.
    if (rawbuf.size() < uncomp_size) rawbuf.resize(uncomp_size);
    size_t used = 0;
    auto sink = [&](const void* p, size_t n) {
      if (n == 0) return;
      memcpy(rawbuf.data() + used, p, n);
      used += n;
    };
    sink(npy_header.data(), npy_header.size());
    emit(sink);
    if (used != uncomp_size)
      throw runtime_error("NpzFileSession: emitted size mismatch");
    uint32_t crc = libdeflate_crc32(0, rawbuf.data(), uncomp_size);

    // 2. Compress in one call. A zero return means the output did not fit
    // the bound, which libdeflate documents as "store it instead"; the same
    // applies when compression does not shrink the data.
    size_t comp_size = 0;
    if (compressor) {
      size_t bound = libdeflate_deflate_compress_bound(compressor, uncomp_size);
      if (zbuf.size() < bound) zbuf.resize(bound);
      comp_size = libdeflate_deflate_compress(compressor, rawbuf.data(),
                                              uncomp_size, zbuf.data(), bound);
    }
    bool use_compressed = comp_size > 0 && comp_size < uncomp_size;
    const uint8_t* payload = use_compressed ? zbuf.data() : rawbuf.data();
    used = use_compressed ? comp_size : uncomp_size;
    uint32_t out_size = (uint32_t)used;
    uint16_t method = use_compressed ? 8 : 0;

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

    // 3. ZIP local file header (PK\x03\x04, 30 bytes + filename), complete —
    //    CRC and sizes are known by now, so nothing needs patching later.
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

    // 4. Payload.
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

    // The central directory and the end record are the last bytes written, and they may still be in
    // the stdio buffer: they reach the file system only here, which is also where NFS and Lustre
    // report delayed errors (disk full, quota). Without these checks a file that cannot be opened
    // would count as written.
    const bool flushed = fflush(fp) == 0;
    const bool closed = fclose(fp) == 0;
    fp = nullptr;
    if (!flushed || !closed)
      throw runtime_error(string("could not finish writing the file: ") + strerror(errno));
  }

private:
  FILE* fp;
  vector<uint8_t>& rawbuf;      // uncompressed staging (caller-owned, reused)
  vector<uint8_t>& zbuf;        // compressed staging (caller-owned, reused)
  libdeflate_compressor* compressor;  // caller-owned; null = stored mode
  size_t offset;                // current write position within file
  vector<NpzEntry> entries;

  void checked_write(const void* data, size_t size)
  {
    if (fwrite(data, 1, size, fp) != size)
      throw runtime_error("NpzFileSession: write failed");
  }
};

}  // namespace

NpzWriter::NpzWriter(const string& output_path, int chunk_size, int worker_id,
                     const TokenStrides& strides)
  : output_path(output_path), chunk_size(chunk_size), worker_id(worker_id),
    chunk_id(0), strides(strides),
    rows_seg(static_cast<size_t>(chunk_size) * strides.seg, 0),
    rows_sig(static_cast<size_t>(chunk_size) * strides.sig, 0.0f),
    rows_kmer(static_cast<size_t>(chunk_size) * strides.kmer, 0),
    rows_motor(static_cast<size_t>(chunk_size) * strides.tok, 0.0f),
    rows_pore(static_cast<size_t>(chunk_size) * strides.tok, 0.0f),
    rows_bq(static_cast<size_t>(chunk_size) * strides.tok, 0),
    rows_label(static_cast<size_t>(chunk_size), 0),
    rows_rid(static_cast<size_t>(chunk_size) * 2, 0),
    row_count(0),
    npz_compressor(libdeflate_alloc_compressor(kNpzDeflateLevel))
{
  reset_maxima();
  filesystem::create_directories(output_path);
  // Allocation failure only; every array would then be stored uncompressed,
  // which is valid output but not what anyone wants silently.
  if (!npz_compressor)
    cerr << "NpzWriter: libdeflate_alloc_compressor failed for worker "
         << worker_id << "; arrays will be stored uncompressed" << endl;
}

NpzWriter::~NpzWriter()
{
  if (npz_compressor)
    libdeflate_free_compressor(npz_compressor);
}

void NpzWriter::reset_maxima()
{
  max_seg = max_sig = max_kmer = max_motor = max_pore = max_bq = 0;
}

NpzWriter::RowRef NpzWriter::acquire_row()
{
  // Lazy chunk cut (as in the Python preprocessing): a full buffer is saved as a numbered
  // chunk only when the next row arrives. The final chunk therefore always
  // goes through flush() as {worker}-last.npz, even when the worker's record
  // count is an exact multiple of chunk_size.
  if (row_count >= static_cast<size_t>(chunk_size)) {
    save_chunk(row_count, generate_filename(false));
    chunk_id++;
    row_count = 0;
    reset_maxima();
  }

  // Zero the whole row at its strides. An abandoned row (producer filtered the
  // record out mid-fill) is re-zeroed on the next acquire, so partial writes
  // cannot leak into the next committed record.
  size_t i = row_count;
  RowRef row;
  row.seg = rows_seg.data() + i * strides.seg;
  row.sig = rows_sig.data() + i * strides.sig;
  row.kmer = rows_kmer.data() + i * strides.kmer;
  row.motor = rows_motor.data() + i * strides.tok;
  row.pore = rows_pore.data() + i * strides.tok;
  row.bq = rows_bq.data() + i * strides.tok;
  fill_n(row.seg, strides.seg, static_cast<uint16_t>(0));
  fill_n(row.sig, strides.sig, 0.0f);
  fill_n(row.kmer, strides.kmer, static_cast<uint8_t>(0));
  fill_n(row.motor, strides.tok, 0.0f);
  fill_n(row.pore, strides.tok, 0.0f);
  fill_n(row.bq, strides.tok, static_cast<uint8_t>(0));
  return row;
}

void NpzWriter::commit_row(int64_t rid0, int64_t rid1, int64_t label_id,
                           size_t seg_len, size_t sig_len, size_t kmer_len,
                           size_t motor_len, size_t pore_len, size_t bq_len)
{
  size_t i = row_count;
  rows_label[i] = label_id;

  // The UUID was parsed once at extraction; these are its two halves.
  rows_rid[i * 2] = rid0;
  rows_rid[i * 2 + 1] = rid1;

  // The observed maxima become the chunk's npy widths, matching what the
  // previous writer measured over its buffered records.
  max_seg = max(max_seg, seg_len);
  max_sig = max(max_sig, sig_len);
  max_kmer = max(max_kmer, kmer_len);
  max_motor = max(max_motor, motor_len);
  max_pore = max(max_pore, pore_len);
  max_bq = max(max_bq, bq_len);

  ++row_count;
}

string NpzWriter::generate_filename(bool is_last_chunk)
{
  string filename = output_path + "/" + to_string(worker_id) + "-";

  if (is_last_chunk) {
    filename += "last.npz";
  } else {
    filename += to_string(chunk_id) + ".npz";
  }

  return filename;
}

void NpzWriter::save_chunk(size_t count, const string& filename)
{
  if (count == 0) return;

  // Emit each array straight from the fixed-stride row storage. Only the
  // observed-max prefix of each row is emitted, so the on-disk widths equal
  // the measured maxima exactly as the record-buffer writer produced.
  try {
    NpzFileSession sess(filename, npz_rawbuf, npz_zbuf, npz_compressor);

    // Emit `width` leading elements of each row from `data` laid out at
    // `stride` elements per row. Rows were zeroed at acquire_row, so the
    // region between the record's actual length and `width` is zero padding.
    auto emit_rows = [count](const auto* data, size_t stride, size_t width) {
      return [=](auto&& sink) {
        using T = remove_cvref_t<decltype(*data)>;
        for (size_t i = 0; i < count; ++i)
          sink(data + i * stride, width * sizeof(T));
      };
    };

    sess.add_array_stream<uint16_t>("segment_len_arr", {count, max_seg},
      emit_rows(rows_seg.data(), strides.seg, max_seg));
    sess.add_array_stream<float>("signal_token", {count, max_sig},
      emit_rows(rows_sig.data(), strides.sig, max_sig));
    sess.add_array_stream<uint8_t>("kmer_token", {count, max_kmer},
      emit_rows(rows_kmer.data(), strides.kmer, max_kmer));
    sess.add_array_stream<float>("dwell_motor_token", {count, max_motor},
      emit_rows(rows_motor.data(), strides.tok, max_motor));
    sess.add_array_stream<float>("dwell_pore_token", {count, max_pore},
      emit_rows(rows_pore.data(), strides.tok, max_pore));
    sess.add_array_stream<uint8_t>("bq_token", {count, max_bq},
      emit_rows(rows_bq.data(), strides.tok, max_bq));
    sess.add_array_stream<int64_t>("label_id", {count},
      emit_rows(rows_label.data(), 1, 1));
    sess.add_array_stream<int64_t>("read_id", {count, (size_t)2},
      emit_rows(rows_rid.data(), 2, 2));

    sess.close();
  } catch (const exception& e) {
    cerr << "Error saving NPZ file " << filename << ": " << e.what() << endl;
    failed_write = true;
    files_not_written.fetch_add(1);
    // Avoid leaving a partial NPZ on disk: a downstream NpzFileSession
    // destructor on the throw path may have written a (possibly truncated)
    // file via fopen("wb") + close(), which would otherwise be picked up
    // by readers as a real chunk.
    error_code ec;
    filesystem::remove(filename, ec);
  }
}

void NpzWriter::flush()
{
  // The lazy cut in acquire_row leaves between 1 and chunk_size rows here for
  // any worker that produced records; they become its single {worker}-last.npz.
  if (row_count > 0) {
    string filename = generate_filename(true);
    save_chunk(row_count, filename);
    row_count = 0;
    reset_maxima();
  }
}
