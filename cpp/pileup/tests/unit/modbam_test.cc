/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "htslib/bgzf.h"
#include "htslib/sam.h"
#include "libdeflate/libdeflate.h"

#include "pileup/bgzf_writer.h"

#include "pileup/mod_tags.h"
#include "pileup/modbam.h"
#include "pileup/read_rows.h"
#include "tests/unit/unit_test.h"

namespace deeprm::test {
namespace {

// Bytes recorded from NumPy: np.clip(np.rint(pred * 255), 0, 255) in float32.
void probability_bytes_match_numpy(Checker& t) {
  const float preds[] = {0.0F, 0.5F, 0.0019607844F, 0.998F,
                         1.0F, 0.0980392F, 0.9941176F};
  const int bytes[] = {0, 128, 0, 254, 255, 25, 254};
  for (std::size_t i = 0; i < std::size(preds); ++i) {
    t.equal(static_cast<int>(probability_byte(preds[i])), bytes[i]);
  }
}

void packed_rows_come_back(Checker& t) {
  const ReadKey a{3, 11, -12};
  const ReadKey b{3, 11, 13};
  const ReadKey c{-0 + 7, 1, 1};
  std::vector<PackedRows> files(2);
  files[0].add(a, {1000000, 7});
  files[0].add(a, {1000003, 8});
  files[0].add(b, {5, 9});
  files[0].add(b, {2, 10});  // positions may go down
  files[1].add(a, {999999999, 255});
  files[1].add(c, {0, 0});
  for (PackedRows& file : files) file.finish();
  t.equal(files[0].runs().size(), std::size_t{2});
  const ReadRows rows(std::move(files), 3);
  std::vector<ReadRow> found;
  t.check(rows.find(a, found) && found.size() == 3, "read a: both files");
  if (found.size() == 3) {
    t.equal(found[1].position, std::int64_t{1000003});
    t.equal(found[2].position, std::int64_t{999999999});
    t.equal(static_cast<int>(found[2].probability), 255);
  }
  t.check(rows.find(b, found) && found.size() == 2, "read b");
  if (found.size() == 2) t.equal(found[1].position, std::int64_t{2});
  t.check(!rows.find(ReadKey{3, 11, 14}, found) && found.empty(), "no rows");
  t.check(rows.has_reference(3) && rows.has_reference(7), "references");
  t.check(!rows.has_reference(4), "a reference without rows");
}

// The halves recorded from Python: np.frombuffer(uuid.UUID(name).bytes, int64).
void read_ids_match_python(Checker& t) {
  for (const std::string_view name :
       {"2ef40a41-9cbe-4116-9974-87afee5d488e",
        "2EF40A419CBE4116997487AFEE5D488E",
        "{urn:uuid:2ef40a41-9cbe-4116-9974-87afee5d488e}"}) {
    const auto id = parse_read_id(name);
    t.check(id.has_value(), "a UUID");
    if (!id) continue;
    t.equal(id->first, std::int64_t{1603772520636019758});
    t.equal(id->second, std::int64_t{-8194196242272127847});
  }
  t.check(!parse_read_id("read_17"), "not a UUID");
  t.check(!parse_read_id("2ef40a41-9cbe-4116-9974-87afee5d488g"), "not hex");
}

// Tags recorded from _mm_ml_batch of the Python pileup for the read
// TTAACAGATTACAGATA, CIGAR 3S5M2I4M1D3M at reference position 100.
void mod_tags_match_the_python_pileup(Checker& t) {
  const std::string sequence = "TTAACAGATTACAGATA";
  // length << 4 | operation: 3S 5M 2I 4M 1D 3M
  const std::vector<std::uint32_t> cigar = {3 << 4 | 4, 5 << 4 | 0, 2 << 4 | 1,
                                            4 << 4 | 0, 1 << 4 | 2, 3 << 4 | 0};
  const std::vector<ReadRow> rows = {
      {100, 10}, {101, 20}, {102, 30}, {104, 40},  {105, 50}, {106, 60},
      {108, 70}, {109, 80}, {110, 90}, {111, 100}, {112, 110}, {99, 1},
      {113, 2},  {101, 20}, {104, 41}, {107, 5}};
  ModTagMaker maker;
  for (const bool reverse : {false, true}) {
    std::vector<std::uint8_t> is_base;
    for (const char base : sequence) {
      is_base.push_back(base == (reverse ? 'T' : 'A') ? 1 : 0);
    }
    const Alignment alignment{cigar, 100, reverse, is_base};
    maker.make(alignment, rows);
    if (reverse) {
      t.equal(maker.mm(), std::string("A+a?,0;"));
      t.check(maker.ml() == std::vector<std::uint8_t>{100}, "ML, reverse");
    } else {
      t.equal(maker.mm(), std::string("A+a?,1,0,1,0,0,0;"));
      t.check(maker.ml() == std::vector<std::uint8_t>{10, 30, 50, 5, 90, 110},
              "ML, forward");
    }
    maker.make(alignment, std::vector<ReadRow>{{500, 1}});
    t.equal(maker.mm(), std::string("A+a?;"));
    t.check(maker.ml().empty(), "no ML without calls");
  }
}

void header_lines_are_reordered(Checker& t) {
  const std::string_view text =
      "@HD\tVN:1.6\tSO:coordinate\n@PG\tID:a\tPN:x\n@SQ\tSN:c\tLN:9\n"
      "@XX\tzz\n@RG\tPU:u\tID:g1\n@CO\tnote\n@PG\tPN:y\tID:b\n";
  t.equal(merge_order_header(text),
          std::string("@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:c\tLN:9\n"
                      "@RG\tPU:u\tID:g1\n@PG\tID:a\tPN:x\n@PG\tPN:y\tID:b\n"
                      "@CO\tnote\n"));
  t.check(header_ids(text, "@RG") == std::vector<std::string>{"g1"}, "RG IDs");
  t.check(header_ids(text, "@PG") == std::vector<std::string>{"a", "b"},
          "PG IDs");
}

void modbam_tasks_cover_the_references(Checker& t) {
  // reads per reference; reference 1 has none. Reference 3 is large and has
  // all its reads in the first quarter of its 1,000,000 bases, evenly.
  const ModbamWeight weight = [](std::int32_t, std::int64_t begin,
                                 std::int64_t end) {
    const std::int64_t covered = std::min<std::int64_t>(end, 250000) -
                                 std::min<std::int64_t>(begin, 250000);
    return static_cast<std::uint64_t>(covered);
  };
  const std::vector<ModbamTask> tasks = plan_modbam_tasks(
      {40, 0, 70, 250, 10, 5}, {9, 9, 9, 1000000, 9, 9}, 100, weight);
  t.equal(tasks.size(), std::size_t{8});
  if (tasks.size() != 8) return;
  t.check(tasks[0].first == 0 && tasks[0].last == 2, "references 0-2");
  // 100 of the 250 reads are 100,000 of the weight: windows are halved
  // until they hold less
  t.check(tasks[1].first == 3 && tasks[1].begin == 0 && tasks[1].end == 62500,
          "first window of reference 3");
  t.check(tasks[4].begin == 187500 && tasks[4].end == 250000, "fourth window");
  t.check(tasks[5].begin == 250000 && tasks[5].end == 500000,
          "a window without reads is not cut");
  t.check(tasks[6].begin == 500000 && tasks[6].end > 1000000,
          "the last is open");
  t.check(tasks[7].first == 4 && tasks[7].last == 5, "references 4-5");
}

// The data of all BGZF blocks in `blocks`, inflated block by block.
std::vector<std::uint8_t> inflate_blocks(const std::vector<std::uint8_t>& blocks,
                                         std::size_t& count) {
  std::vector<std::uint8_t> data;
  libdeflate_decompressor* decompressor = libdeflate_alloc_decompressor();
  count = 0;
  for (std::size_t at = 0; at + 18 <= blocks.size(); ++count) {
    const std::size_t size = (blocks[at + 16] | blocks[at + 17] << 8) + 1;
    std::vector<std::uint8_t> block(0x10000);
    std::size_t inflated = 0;
    if (libdeflate_gzip_decompress(decompressor, blocks.data() + at, size,
                                   block.data(), block.size(),
                                   &inflated) != LIBDEFLATE_SUCCESS) {
      break;
    }
    data.insert(data.end(), block.begin(), block.begin() + inflated);
    at += size;
  }
  libdeflate_free_decompressor(decompressor);
  return data;
}

void bgzf_blocks_hold_the_data(Checker& t) {
  std::vector<std::uint8_t> data(200000);
  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<std::uint8_t>(i * i >> 7);
  }
  BgzfWriter writer(1);
  writer.write(std::span(data).first(100));
  t.equal(writer.tell(), std::uint64_t{100});
  writer.keep_together(70000);  // more than a block: nothing to keep whole
  writer.keep_together(65200);  // does not fit after 100 bytes
  t.equal(writer.tell() & 0xFFFF, std::uint64_t{0});
  t.check(writer.tell() >> 16 != 0, "the first block was ended");
  writer.write(std::span(data).subspan(100));
  writer.flush();
  const std::uint64_t end = writer.tell();
  const std::vector<std::uint8_t> blocks = writer.take_blocks();
  t.equal(end, static_cast<std::uint64_t>(blocks.size()) << 16);
  std::size_t count = 0;
  t.check(inflate_blocks(blocks, count) == data, "the blocks inflate to the data");
  t.equal(count, std::size_t{5});  // 100, then 3 full blocks and the rest
  t.equal(bgzf_end_block().size(), std::size_t{28});
}

// The bytes htslib's bam_write1 writes for the record.
std::vector<std::uint8_t> written_by_htslib(const bam1_t* record) {
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("deeprm_unit_" + std::to_string(std::rand()) + ".bgzf")).string();
  BGZF* out = bgzf_open(path.c_str(), "wu");  // not compressed
  const bool written = bam_write1(out, record) >= 0;
  bgzf_close(out);
  if (!written) return {};
  std::vector<std::uint8_t> bytes(std::filesystem::file_size(path));
  std::FILE* in = std::fopen(path.c_str(), "rb");
  const std::size_t size = std::fread(bytes.data(), 1, bytes.size(), in);
  std::fclose(in);
  std::filesystem::remove(path);
  bytes.resize(size);
  return bytes;
}

// A record with `operations` CIGAR operations of 1M each.
bam1_t* make_record(std::size_t operations) {
  const std::vector<std::uint32_t> cigar(operations, 1 << 4 | BAM_CMATCH);
  const std::string sequence(operations, 'A');
  const std::string quality(operations, 30);
  bam1_t* record = bam_init1();
  bam_set1(record, 4, "read", 16, 3, 1000, 60, cigar.size(), cigar.data(), -1,
           -1, 0, sequence.size(), sequence.data(), quality.data(), 64);
  const std::int32_t value = 77;
  bam_aux_append(record, "NM", 'i', 4,
                 reinterpret_cast<const std::uint8_t*>(&value));
  return record;
}

// 70,000 operations do not fit the record's 16-bit count: the real CIGAR
// goes into a CG tag.
void records_are_encoded_as_htslib_does(Checker& t) {
  for (const std::size_t operations : {std::size_t{5}, std::size_t{70000}}) {
    bam1_t* record = make_record(operations);
    std::vector<std::uint8_t> ours;
    t.check(encode_bam_record(record, ours), "the record fits the format");
    t.check(ours == written_by_htslib(record), "the same bytes as bam_write1");
    bam_destroy1(record);
  }
}

}  // namespace

std::vector<TestCase> modbam_tests() {
  return {{"probability bytes match NumPy", probability_bytes_match_numpy},
          {"packed per-read rows come back", packed_rows_come_back},
          {"read IDs match Python's uuid", read_ids_match_python},
          {"MM/ML tags match the Python pileup",
           mod_tags_match_the_python_pileup},
          {"header lines are reordered", header_lines_are_reordered},
          {"modBAM tasks cover the references",
           modbam_tasks_cover_the_references},
          {"BGZF blocks hold the data", bgzf_blocks_hold_the_data},
          {"BAM records are encoded as htslib does",
           records_are_encoded_as_htslib_does}};
}

}  // namespace deeprm::test
