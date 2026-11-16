/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include <cstring>
#include <string>
#include <vector>

#include "libdeflate/libdeflate.h"

#include "pileup/npz_reader.h"
#include "pileup/npz_writer.h"
#include "tests/unit/unit_test.h"

namespace deeprm::test {
namespace {

using Bytes = std::vector<std::uint8_t>;

void put(Bytes& out, std::uint64_t value, std::size_t width) {
  for (std::size_t i = 0; i < width; ++i) {
    out.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
  }
}

void put(Bytes& out, const std::string& text) {
  out.insert(out.end(), text.begin(), text.end());
}

// A zip with one stored member.
Bytes stored_zip(const std::string& name, const Bytes& data) {
  const std::uint32_t crc = libdeflate_crc32(0, data.data(), data.size());
  Bytes zip;
  for (const std::uint32_t signature : {0x04034b50U, 0x02014b50U}) {
    const std::size_t directory = zip.size();
    put(zip, signature, 4);
    put(zip, 20, signature == 0x02014b50U ? 4 : 2);
    put(zip, 0, 8);
    put(zip, crc, 4);
    put(zip, data.size(), 4);
    put(zip, data.size(), 4);
    put(zip, name.size(), 2);
    put(zip, 0, signature == 0x02014b50U ? 16 : 2);
    put(zip, name);
    if (signature == 0x04034b50U) {
      zip.insert(zip.end(), data.begin(), data.end());
      continue;
    }
    const std::size_t size = zip.size() - directory;
    put(zip, 0x06054b50, 4);
    put(zip, 0, 4);
    put(zip, 1, 2);
    put(zip, 1, 2);
    put(zip, size, 4);
    put(zip, directory, 4);
    put(zip, 0, 2);
  }
  return zip;
}

Bytes npy(const std::string& dict, const Bytes& data) {
  Bytes out = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  put(out, dict.size(), 2);
  put(out, dict);
  out.insert(out.end(), data.begin(), data.end());
  return out;
}

void reads_a_stored_array(Checker& t) {
  const Bytes data = {1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0};
  const Bytes zip = stored_zip(
      "label_id.npy",
      npy("{'descr': '<i8', 'fortran_order': False, 'shape': (2,), }\n", data));
  NpzReader reader;
  const auto array = reader.read(zip, "label_id");
  t.check(array.has_value(), "the array is read");
  if (!array) return;
  t.equal(array->descr, std::string("<i8"));
  t.equal(array->shape.size(), std::size_t{1});
  t.equal(array->shape[0], std::size_t{2});
  t.check(array->data == data, "the bytes are the array's");
  t.check(!reader.read(zip, "pred").has_value(), "a missing array is an error");
}

void reads_a_two_dimensional_shape(Checker& t) {
  const Bytes zip = stored_zip(
      "pred.npy",
      npy("{'descr': '<f4', 'fortran_order': False, 'shape': (1, 2), }\n",
          Bytes(8, 0)));
  NpzReader reader;
  const auto arrays = reader.read_all(zip);
  t.check(arrays.has_value() && arrays->size() == 1, "one array");
  if (!arrays) return;
  t.equal((*arrays)[0].shape.size(), std::size_t{2});
  t.equal((*arrays)[0].shape[1], std::size_t{2});
}

void refuses_damaged_files(Checker& t) {
  NpzReader reader;
  t.check(!reader.read_all(Bytes(100, 0)).has_value(), "not a zip");
  Bytes zip = stored_zip(
      "x.npy", npy("{'descr': '<i8', 'fortran_order': False, 'shape': (1,), }\n",
                   Bytes(8, 7)));
  zip[zip.size() / 3] ^= 0xFF;  // inside the member's data or header
  t.check(!reader.read_all(zip).has_value(), "a changed byte is an error");
}

Bytes inflate_joined(const std::vector<DeflatedRun>& runs, std::size_t size,
                     bool& ok) {
  Bytes stream;
  for (const DeflatedRun& run : runs) {
    stream.insert(stream.end(), run.bytes.begin(), run.bytes.end());
  }
  stream.insert(stream.end(), {0x03, 0x00});
  Bytes out(size, 0xFF);
  libdeflate_decompressor* decompressor = libdeflate_alloc_decompressor();
  ok = libdeflate_deflate_decompress(decompressor, stream.data(), stream.size(),
                                     out.data(), out.size(),
                                     nullptr) == LIBDEFLATE_SUCCESS;
  libdeflate_free_decompressor(decompressor);
  return out;
}

// The runs of an array, joined and ended, inflate to the array's bytes.
void deflated_runs_join(Checker& t) {
  Bytes first(1000, 0);
  Bytes second(500, 0);
  for (std::size_t i = 0; i < first.size(); ++i) first[i] = i % 7;
  for (std::size_t i = 0; i < second.size(); ++i) second[i] = i % 5;
  RunDeflater deflater;
  const DeflatedRun a = deflater.deflate(first);
  const DeflatedRun b = deflater.deflate(second);
  bool ok = false;
  const Bytes out = inflate_joined({a, b}, 1500, ok);
  t.check(ok, "the joined stream inflates");
  t.check(std::memcmp(out.data(), first.data(), 1000) == 0, "first run");
  t.check(std::memcmp(out.data() + 1000, second.data(), 500) == 0, "second run");
  t.equal(a.size, std::uint64_t{1000});
  t.equal(a.crc32, libdeflate_crc32(0, first.data(), first.size()));
}

// Runs of every kind of content and size: several blocks, text that packs
// well, noise that libdeflate stores, and sizes around the block size.
void deflated_runs_of_any_content_join(Checker& t) {
  RunDeflater deflater;
  std::uint32_t state = 12345;
  const auto next = [&state] {
    state = state * 1664525 + 1013904223;
    return static_cast<std::uint8_t>(state >> 24);
  };
  Bytes all;
  std::vector<DeflatedRun> runs;
  for (const std::size_t size :
       {0, 1, 2, 3, 100, 4096, 32767, 32768, 32769, 200000}) {
    for (int kind = 0; kind < 4; ++kind) {
      Bytes raw(size);
      for (std::size_t i = 0; i < size; ++i) {
        const std::uint8_t noise = next();
        raw[i] = kind == 0   ? noise
                 : kind == 1 ? static_cast<std::uint8_t>(i % 251)
                 : kind == 2 ? static_cast<std::uint8_t>(noise & 3)
                             : static_cast<std::uint8_t>("chr1"[i % 4]);
      }
      runs.push_back(deflater.deflate(raw));
      all.insert(all.end(), raw.begin(), raw.end());
    }
  }
  bool ok = false;
  const Bytes out = inflate_joined(runs, all.size(), ok);
  t.check(ok, "the joined stream inflates");
  t.check(out == all, "to the bytes that went in");
}

void crc32_of_joined_bytes(Checker& t) {
  Bytes first(1234, 0);
  Bytes second(77777, 0);
  for (std::size_t i = 0; i < first.size(); ++i) first[i] = i * 31 % 256;
  for (std::size_t i = 0; i < second.size(); ++i) second[i] = i * 17 % 255;
  Bytes joined = first;
  joined.insert(joined.end(), second.begin(), second.end());
  const std::uint32_t a = libdeflate_crc32(0, first.data(), first.size());
  const std::uint32_t b = libdeflate_crc32(0, second.data(), second.size());
  t.equal(crc32_combine(a, b, second.size()),
          libdeflate_crc32(0, joined.data(), joined.size()));
  t.equal(crc32_combine(a, 0, 0), a);
}

}  // namespace

std::vector<TestCase> npz_tests() {
  return {{"npz reader reads a stored array", reads_a_stored_array},
          {"npz reader reads a 2-D shape", reads_a_two_dimensional_shape},
          {"npz reader refuses damaged files", refuses_damaged_files},
          {"deflated runs join into one stream", deflated_runs_join},
          {"deflated runs of any content join",
           deflated_runs_of_any_content_join},
          {"crc32 of joined bytes", crc32_of_joined_bytes}};
}

}  // namespace deeprm::test
