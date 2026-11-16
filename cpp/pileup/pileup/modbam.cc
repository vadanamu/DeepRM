/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include "pileup/modbam.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <atomic>
#include <climits>
#include <condition_variable>
#include <deque>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <print>
#include <thread>

#include "htslib/bgzf.h"
#include "htslib/hts.h"
#include "htslib/sam.h"

#include "pileup/bgzf_writer.h"
#include "pileup/mod_tags.h"
#include "pileup/output_file.h"
#include "pileup/parallel.h"

namespace deeprm {
namespace {

constexpr std::int64_t kWholeReference = std::numeric_limits<std::int64_t>::max();
constexpr std::uint64_t kTasksPerThread = 8;
constexpr std::uint64_t kSmallestTask = 20000;  // reads
constexpr std::size_t kCopyBytes = std::size_t{8} << 20;
// Bytes of a shard that are kept in memory while it is made.
constexpr std::size_t kShardMemory = std::size_t{64} << 20;
// Records that are tagged and compressed together.
constexpr std::size_t kBatchRecords = 512;
// Batches that can wait for a thread, per thread.
constexpr std::size_t kWaitingBatchesPerThread = 2;
// Threads that inflate the input for the thread that reads a dense window.
constexpr int kDenseReadThreads = 8;
constexpr int kLevel = 1;  // BGZF level, as the Python pileup
constexpr int kBaiMinShift = 14;
constexpr int kBaiLevels = 5;

struct CloseSam { void operator()(samFile* file) const { sam_close(file); } };
struct FreeHeader { void operator()(sam_hdr_t* h) const { sam_hdr_destroy(h); } };
struct FreeIndex { void operator()(hts_idx_t* i) const { hts_idx_destroy(i); } };
struct FreeRecord { void operator()(bam1_t* b) const { bam_destroy1(b); } };
struct FreeIterator { void operator()(hts_itr_t* i) const { hts_itr_destroy(i); } };

using SamFile = std::unique_ptr<samFile, CloseSam>;
using Header = std::unique_ptr<sam_hdr_t, FreeHeader>;
using Index = std::unique_ptr<hts_idx_t, FreeIndex>;
using Record = std::unique_ptr<bam1_t, FreeRecord>;

// Where a written record lies, for the index.
struct IndexEntry {
  std::int32_t reference;
  std::int64_t begin;
  std::int64_t end;
  std::uint64_t offset;  // virtual offset of the record's end in its shard
  bool mapped;
};

// The records of one task as BGZF blocks.
struct Shard {
  std::vector<std::uint8_t> blocks;  // until they are written
  std::string path;                  // of the file they can wait in
  bool in_file = false;              // whether (all of) them wait there
  std::uint64_t size = 0;
  std::uint64_t offset = 0;  // in the modBAM
  std::vector<IndexEntry> entries;
};

// Moves the shard's blocks from memory to the end of its file. `more`: the
// shard is not finished, and keeps the memory for its next blocks (fresh
// memory for every part of a large shard cost 10 s of a run).
Status move_to_file(Shard& shard, bool more) {
  std::filesystem::create_directories(
      std::filesystem::path(shard.path).parent_path());
  std::ofstream out(shard.path, std::ios::binary | std::ios::app);
  out.write(reinterpret_cast<const char*>(shard.blocks.data()),
            static_cast<std::streamsize>(shard.blocks.size()));
  out.close();
  if (!out) return std::unexpected("cannot write " + shard.path);
  shard.size += shard.blocks.size();
  if (more) {
    shard.blocks.clear();
  } else {
    // frees the memory; "= {}" would only empty the vector
    shard.blocks = std::vector<std::uint8_t>();
  }
  shard.in_file = true;
  return {};
}

// Copies the file at `path` to `offset` of `file`.
Status copy_file(const std::string& path, std::uint64_t size,
                 std::uint64_t offset, OutputFile& file) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::unexpected("cannot open " + path);
  std::vector<std::uint8_t> bytes(std::min<std::uint64_t>(kCopyBytes, size));
  for (std::uint64_t done = 0; done < size;) {
    const std::size_t part = std::min<std::uint64_t>(bytes.size(), size - done);
    in.read(reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(part));
    if (!in) return std::unexpected("cannot read " + path);
    if (Status status = file.write_at(offset + done, {bytes.data(), part});
        !status) {
      return status;
    }
    done += part;
  }
  return {};
}

// Writes the shards to the modBAM in order, each as soon as it and all
// those before it are finished, so that most are written once and straight
// from memory. A shard that is finished long before its turn would wait in
// memory with many others (reads lie unevenly: one shard can take as long
// as hundreds of others); past kWaitingBytes such shards wait in their files
// instead.
class ShardSink {
 public:
  ShardSink(OutputFile& file, std::vector<Shard>& shards)
      : file_(file), shards_(shards), done_(shards.size(), false) {}

  // Called by the thread that finished shard `i`.
  Status finish(std::size_t i) {
    Shard& shard = shards_[i];
    bool to_file = shard.in_file;  // a large shard, partly there already
    if (!to_file) {
      const std::lock_guard<std::mutex> lock(mutex_);
      to_file = i != next_ && waiting_ + shard.blocks.size() > kWaitingBytes;
      if (!to_file) waiting_ += shard.blocks.size();
    }
    if (to_file) {
      if (Status status = move_to_file(shard, false); !status) return status;
    } else {
      shard.size = shard.blocks.size();
    }
    std::vector<std::size_t> ready;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      done_[i] = true;
      for (; next_ < shards_.size() && done_[next_]; ++next_) {
        shards_[next_].offset = offset_;
        offset_ += shards_[next_].size;
        ready.push_back(next_);
      }
    }
    for (const std::size_t number : ready) {
      if (Status status = write(shards_[number]); !status) return status;
    }
    return {};
  }

  // The size of all shards, once all are finished.
  std::uint64_t size() const { return offset_; }

 private:
  static constexpr std::uint64_t kWaitingBytes = std::uint64_t{2} << 30;

  Status write(Shard& shard) {
    if (shard.in_file) {
      const Status copied =
          copy_file(shard.path, shard.size, shard.offset, file_);
      std::filesystem::remove(shard.path);
      return copied;
    }
    const std::vector<std::uint8_t> blocks = std::move(shard.blocks);
    shard.blocks = std::vector<std::uint8_t>();
    const Status written = file_.write_at(shard.offset, blocks);
    const std::lock_guard<std::mutex> lock(mutex_);
    waiting_ -= shard.size;
    return written;
  }

  OutputFile& file_;
  std::vector<Shard>& shards_;
  std::mutex mutex_;
  std::vector<bool> done_;
  std::size_t next_ = 0;
  std::uint64_t offset_ = 0;
  std::uint64_t waiting_ = 0;  // bytes of finished shards in memory
};

// The IDs a record's RG and PG tags may name.
struct TagIds {
  std::vector<std::string> read_groups;
  std::vector<std::string> programs;
};

// What one thread keeps between its tasks.
struct Worker {
  SamFile input;
  Record record{bam_init1()};
  ModTagMaker maker;
  std::vector<ReadRow> rows;
  std::vector<std::uint8_t> is_base;
  std::vector<std::uint8_t> ml;
  std::vector<std::uint8_t> raw;  // the record as BAM bytes
  BgzfWriter blocks{kLevel};
  bool inflates_in_threads = false;  // whether `input` has threads of its own
};

// The input, shared by the threads (each reads through its own handle).
struct Source {
  const ModbamJob& job;
  const ReadRows& rows;
  Header header;
  Index index;
  TagIds ids;
};

void replace_tag(bam1_t* record, const char tag[2], char type, int size,
                 const std::uint8_t* data) {
  if (std::uint8_t* old = bam_aux_get(record, tag)) bam_aux_del(record, old);
  if (data != nullptr) bam_aux_append(record, tag, type, size, data);
}

// What samtools merge -c -p did to every record up to DeepRM 1.2: the RG and
// then the PG tag move to the end, or are dropped when the header does not
// have their ID.
void move_id_tag(bam1_t* record, const char tag[2],
                 const std::vector<std::string>& ids) {
  const std::uint8_t* old = bam_aux_get(record, tag);
  if (old == nullptr || (*old != 'Z' && *old != 'H' && *old != 'A')) return;
  const std::string value =
      *old == 'A' ? std::string(1, static_cast<char>(old[1]))
                  : std::string(reinterpret_cast<const char*>(old + 1));
  const bool known = std::ranges::find(ids, value) != ids.end();
  replace_tag(record, tag, 'Z', static_cast<int>(value.size() + 1),
              known ? reinterpret_cast<const std::uint8_t*>(value.c_str())
                    : nullptr);
}

// Gives the record the MM and ML tags of `worker.maker` (and the MN tag, if
// it has one, the length they refer to).
void set_tags(Worker& worker, const TagIds& ids) {
  bam1_t* record = worker.record.get();
  const std::string& mm = worker.maker.mm();
  replace_tag(record, "MM", 'Z', static_cast<int>(mm.size() + 1),
              reinterpret_cast<const std::uint8_t*>(mm.c_str()));
  if (bam_aux_get(record, "MN") != nullptr) {
    const std::int32_t bases = record->core.l_qseq;
    replace_tag(record, "MN", 'i', sizeof(bases),
                reinterpret_cast<const std::uint8_t*>(&bases));
  }
  const std::vector<std::uint8_t>& ml = worker.maker.ml();
  worker.ml.assign({'C', 0, 0, 0, 0});  // B array: subtype and count
  for (std::size_t i = 0; i < 4; ++i) {
    worker.ml[1 + i] = static_cast<std::uint8_t>(ml.size() >> (8 * i));
  }
  worker.ml.insert(worker.ml.end(), ml.begin(), ml.end());
  replace_tag(record, "ML", 'B', static_cast<int>(worker.ml.size()),
              ml.empty() ? nullptr : worker.ml.data());
  move_id_tag(record, "RG", ids.read_groups);
  move_id_tag(record, "PG", ids.programs);
}

// Tags the record in `worker` from its read's rows; false for a record that
// is not written (no sequence, or no rows on its reference).
bool tag_record(Worker& worker, const Source& source) {
  bam1_t* record = worker.record.get();
  if (record->core.l_qseq == 0 || record->core.tid < 0) return false;
  const auto id = parse_read_id(bam_get_qname(record));
  if (!id) return false;
  const ReadKey key{record->core.tid, id->first, id->second};
  if (!source.rows.find(key, worker.rows)) return false;
  const bool reverse = bam_is_rev(record);
  const std::uint8_t* sequence = bam_get_seq(record);
  const int target = reverse ? 8 : 1;  // T or A in the 4-bit code
  worker.is_base.resize(static_cast<std::size_t>(record->core.l_qseq));
  for (std::size_t i = 0; i < worker.is_base.size(); ++i) {
    worker.is_base[i] = bam_seqi(sequence, i) == target ? 1 : 0;
  }
  worker.maker.make(
      Alignment{.cigar = {bam_get_cigar(record), record->core.n_cigar},
                .reference_start = record->core.pos,
                .reverse = reverse,
                .is_base = worker.is_base},
      worker.rows);
  set_tags(worker, source.ids);
  return true;
}

void put_le(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    out.push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

void put(std::vector<std::uint8_t>& out, const std::uint8_t* data,
         std::size_t size) {
  out.insert(out.end(), data, data + size);
}

// Records of a task as they were read, to be tagged and compressed by any
// thread; then also the result of that.
struct Batch {
  struct Item {
    bam1_core_t core;
    std::size_t offset;  // of the record's data in `data`
    int size;
  };

  void add(const bam1_t* record) {
    items.push_back(Item{record->core, data.size(), record->l_data});
    data.insert(data.end(), record->data, record->data + record->l_data);
  }

  std::vector<Item> items;
  std::vector<std::uint8_t> data;
  std::vector<std::uint8_t> blocks;  // the records as BGZF blocks
  std::vector<IndexEntry> entries;   // with offsets from the start of `blocks`
  Status status;
  bool done = false;  // guarded by the Helpers
};

// Tags the record in `worker` and writes it to the worker's blocks, if its
// read has rows.
Status write_record(Worker& worker, const Source& source,
                    std::vector<IndexEntry>& entries) {
  const bam1_t* record = worker.record.get();
  if (!tag_record(worker, source)) return {};
  if (!encode_bam_record(record, worker.raw)) {
    return std::unexpected(std::string("the BAM format cannot hold read ") +
                           bam_get_qname(record));
  }
  worker.blocks.keep_together(worker.raw.size());
  worker.blocks.write(worker.raw);
  entries.push_back(IndexEntry{.reference = record->core.tid,
                               .begin = record->core.pos,
                               .end = bam_endpos(record),
                               .offset = worker.blocks.tell(),
                               .mapped = (record->core.flag & BAM_FUNMAP) == 0});
  return {};
}

// Ends the blocks that the worker made for `batch` and gives them to it.
void take_blocks(Worker& worker, Batch& batch) {
  worker.blocks.flush();
  // the last record ends where the next batch's first block will start
  if (!batch.entries.empty()) batch.entries.back().offset = worker.blocks.tell();
  batch.blocks = worker.blocks.take_blocks();
  worker.blocks.restart();
}

// Tags the batch's records and makes BGZF blocks of them.
Status process_batch(Worker& worker, const Source& source, Batch& batch) {
  bam1_t* record = worker.record.get();
  for (const Batch::Item& item : batch.items) {
    bam1_t view = {};  // the record in the batch, for htslib to copy
    view.core = item.core;
    view.data = batch.data.data() + item.offset;
    view.l_data = item.size;
    view.m_data = static_cast<std::uint32_t>(item.size);
    if (bam_copy1(record, &view) == nullptr) {
      return std::unexpected("out of memory");
    }
    if (Status status = write_record(worker, source, batch.entries); !status) {
      return status;
    }
  }
  take_blocks(worker, batch);
  batch.items = std::vector<Batch::Item>();
  batch.data = std::vector<std::uint8_t>();
  return {};
}

// Batches that threads take from the threads that read a task. Reads lie
// unevenly: a dense window (the mitochondrial genome, say) holds as many
// reads as a hundred other tasks and cannot be cut. Reading it stays with
// one thread; tagging and compressing its records, most of the work, is
// shared: every thread takes waiting batches before it starts a task of its
// own, and the threads without a task take them until all tasks are done.
class Helpers {
 public:
  explicit Helpers(std::size_t room) : room_(room) {}

  // Whether a thread waits for a batch that nobody has offered yet.
  bool someone_waits() {
    const std::lock_guard<std::mutex> lock(mutex_);
    return waiting_ > queue_.size();
  }

  // Queues the batch for another thread; false when the queue is full.
  bool offer(const std::shared_ptr<Batch>& batch) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.size() >= room_) return false;
    queue_.push_back(batch);
    work_.notify_one();
    return true;
  }

  // A queued batch, if there is one.
  std::shared_ptr<Batch> take_if_any() {
    const std::lock_guard<std::mutex> lock(mutex_);
    return pop();
  }

  // The next batch, waiting for it; nothing when no task is left to give any.
  std::shared_ptr<Batch> take() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++waiting_;
    work_.wait(lock, [this] { return !queue_.empty() || closed_; });
    --waiting_;
    return pop();
  }

  void finish(Batch& batch, Status status) {
    const std::lock_guard<std::mutex> lock(mutex_);
    batch.status = std::move(status);
    batch.done = true;
    done_.notify_all();
  }

  bool is_done(const Batch& batch) {
    const std::lock_guard<std::mutex> lock(mutex_);
    return batch.done;
  }

  void wait(const Batch& batch) {
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [&batch] { return batch.done; });
  }

  // Every task is finished: the waiting threads can go.
  void close() {
    const std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    work_.notify_all();
  }

 private:
  std::shared_ptr<Batch> pop() {
    if (queue_.empty()) return nullptr;
    std::shared_ptr<Batch> batch = std::move(queue_.front());
    queue_.pop_front();
    return batch;
  }

  std::size_t room_;  // batches that can wait
  std::mutex mutex_;
  std::condition_variable work_;
  std::condition_variable done_;
  std::deque<std::shared_ptr<Batch>> queue_;
  std::size_t waiting_ = 0;
  bool closed_ = false;
};

// Makes a shard of the records of one task, batch by batch.
class ShardBuilder {
 public:
  // `dense`: the task is a dense window, whose batches always go to other
  // threads; those of another task do when threads wait for work.
  ShardBuilder(Worker& worker, const Source& source, Helpers& helpers,
               Shard& shard, bool dense)
      : worker_(worker),
        source_(source),
        helpers_(helpers),
        shard_(shard),
        dense_(dense),
        shared_(dense) {}

  // Adds the record that is in the worker. While no thread waits for work
  // the record is tagged and compressed here, at once; otherwise it is copied
  // into a batch for another thread.
  Status add() {
    const bam1_t* record = worker_.record.get();
    if (shared_) {
      if (!has_rows(record)) return {};  // not worth copying
      current_->add(record);
    } else if (Status status = write_record(worker_, source_, current_->entries);
               !status) {
      return status;
    }
    if (++added_ < kBatchRecords) return {};
    return dispatch();
  }

  // The rest of the records, and the batches that other threads process.
  Status finish() {
    if (Status status = dispatch(); !status) return status;
    return collect(true);
  }

 private:
  bool has_rows(const bam1_t* record) const {
    if (record->core.l_qseq == 0 || record->core.tid < 0) return false;
    const auto id = parse_read_id(bam_get_qname(record));
    return id && source_.rows.contains(
                     ReadKey{record->core.tid, id->first, id->second});
  }

  // Ends the current batch: one made here is finished; one of copied
  // records goes to a thread that waits (or is processed here, if none
  // waits any more). Then decides how the next batch is made.
  Status dispatch() {
    const std::shared_ptr<Batch> batch = std::move(current_);
    current_ = std::make_shared<Batch>();
    added_ = 0;
    if (!shared_) {
      take_blocks(worker_, *batch);
      helpers_.finish(*batch, Status{});
    } else if (!batch->items.empty() && !helpers_.offer(batch)) {
      helpers_.finish(*batch, process_batch(worker_, source_, *batch));
    } else if (batch->items.empty()) {
      helpers_.finish(*batch, Status{});
    }
    batches_.push_back(batch);
    shared_ = dense_ || helpers_.someone_waits();
    return collect(false);
  }

  // Appends the finished batches, in order, to the shard; `all`: waits for
  // those that are not finished (and meanwhile processes queued ones).
  Status collect(bool all) {
    while (!batches_.empty()) {
      Batch& batch = *batches_.front();
      if (!helpers_.is_done(batch)) {
        if (!all) break;
        if (const std::shared_ptr<Batch> queued = helpers_.take_if_any()) {
          helpers_.finish(*queued, process_batch(worker_, source_, *queued));
          continue;
        }
        helpers_.wait(batch);
      }
      if (!batch.status) return batch.status;
      const std::uint64_t base = shard_.size + shard_.blocks.size();
      for (IndexEntry& entry : batch.entries) {
        entry.offset += base << 16;
        shard_.entries.push_back(entry);
      }
      shard_.blocks.insert(shard_.blocks.end(), batch.blocks.begin(),
                           batch.blocks.end());
      batches_.pop_front();
      if (shard_.blocks.size() > kShardMemory) {
        // a shard of very many reads: not all of it is kept in memory
        if (Status status = move_to_file(shard_, true); !status) return status;
      }
    }
    return {};
  }

  Worker& worker_;
  const Source& source_;
  Helpers& helpers_;
  Shard& shard_;
  std::shared_ptr<Batch> current_ = std::make_shared<Batch>();
  std::deque<std::shared_ptr<Batch>> batches_;  // dispatched, in order
  std::size_t added_ = 0;  // records seen for the current batch
  bool dense_;
  bool shared_;  // whether the current batch is for another thread
};

// Adds the records of `reference` that start in [begin, end) to the shard.
Status write_reference(Worker& worker, const Source& source,
                       std::int32_t reference, const ModbamTask& task,
                       ShardBuilder& shard) {
  const std::unique_ptr<hts_itr_t, FreeIterator> reads(sam_itr_queryi(
      source.index.get(), reference, task.begin,
      task.end == kWholeReference ? HTS_POS_MAX : task.end));
  if (!reads) return std::unexpected("cannot read " + source.job.input);
  if (task.dense && !worker.inflates_in_threads) {
    // One thread reads such a window and the others tag and compress its
    // records, so the reading sets the pace: more threads inflate for it.
    hts_set_threads(worker.input.get(), kDenseReadThreads);
    worker.inflates_in_threads = true;
  }
  bam1_t* record = worker.record.get();
  int result = 0;
  while ((result = sam_itr_next(worker.input.get(), reads.get(), record)) >= 0) {
    // a read that starts before the window belongs to the window before
    if (record->core.pos < task.begin || record->core.pos >= task.end) continue;
    if (Status status = shard.add(); !status) return status;
  }
  if (result < -1) return std::unexpected("cannot read " + source.job.input);
  return {};
}

// Adds the records of the whole input, read from its start: for a BAM
// without an index.
Status write_all(Worker& worker, const Source& source, ShardBuilder& shard) {
  // more threads than this one inflate the input's blocks
  hts_set_threads(worker.input.get(), static_cast<int>(source.job.threads));
  const Header header(sam_hdr_read(worker.input.get()));
  if (!header) return std::unexpected("cannot read " + source.job.input);
  int result = 0;
  while ((result = sam_read1(worker.input.get(), header.get(),
                             worker.record.get())) >= 0) {
    if (Status status = shard.add(); !status) return status;
  }
  if (result < -1) return std::unexpected("cannot read " + source.job.input);
  return {};
}

Status write_shard(Worker& worker, const Source& source, const ModbamTask& task,
                   Helpers& helpers, Shard& shard) {
  if (!worker.input) {
    worker.input.reset(sam_open(source.job.input.c_str(), "rb"));
    if (!worker.input) return std::unexpected("cannot open " + source.job.input);
  }
  ShardBuilder builder(worker, source, helpers, shard, task.dense);
  if (!source.index) {
    if (Status status = write_all(worker, source, builder); !status) {
      return status;
    }
  }
  for (std::int32_t reference = task.first;
       source.index && reference <= task.last; ++reference) {
    if (!source.rows.has_reference(reference)) continue;
    if (Status status =
            write_reference(worker, source, reference, task, builder);
        !status) {
      return status;
    }
  }
  return builder.finish();
}

// Writes every task's shard. The dense tasks are started first, so that
// their reading goes on beside the other tasks; every thread processes the
// batches that wait before it takes a task, and those without a task until
// all tasks are done.
Status write_shards(const Source& source, const std::vector<ModbamTask>& tasks,
                    std::vector<Shard>& shards, ShardSink& sink) {
  const std::size_t threads = std::max<std::size_t>(source.job.threads, 1);
  std::vector<std::size_t> order(tasks.size());
  std::ranges::iota(order, std::size_t{0});
  std::ranges::stable_partition(
      order, [&tasks](std::size_t task) { return tasks[task].dense; });
  std::vector<Worker> workers(threads);
  Helpers helpers(kWaitingBatchesPerThread * threads);
  std::atomic<std::size_t> next{0};
  std::atomic<std::size_t> left{tasks.size()};
  std::atomic<bool> failed{false};
  std::mutex mutex;
  Status result;
  const auto work = [&](std::size_t worker) {
    const auto help = [&](const std::shared_ptr<Batch>& batch) {
      helpers.finish(*batch, process_batch(workers[worker], source, *batch));
    };
    for (std::size_t at = next.fetch_add(1); at < order.size();
         at = next.fetch_add(1)) {
      while (const std::shared_ptr<Batch> batch = helpers.take_if_any()) {
        help(batch);
      }
      const std::size_t task = order[at];
      if (!failed.load()) {
        Status status = write_shard(workers[worker], source, tasks[task],
                                    helpers, shards[task + 1]);
        if (status) status = sink.finish(task + 1);
        if (!status) {
          const std::lock_guard<std::mutex> lock(mutex);
          if (!failed.exchange(true)) result = std::move(status);
        }
      }
      if (left.fetch_sub(1) == 1) helpers.close();
    }
    while (const std::shared_ptr<Batch> batch = helpers.take()) help(batch);
  };
  if (tasks.empty()) helpers.close();
  {
    std::vector<std::jthread> pool;
    for (std::size_t t = 1; t < threads; ++t) pool.emplace_back(work, t);
    work(0);
  }
  return result;
}

// The header of the modBAM as BGZF blocks: the input's references, and its
// text with the lines in the Python pileup's order.
std::vector<std::uint8_t> header_blocks(const Source& source) {
  const sam_hdr_t* header = source.header.get();
  const std::string text = merge_order_header(
      std::string_view(sam_hdr_str(source.header.get()), sam_hdr_length(source.header.get())));
  std::vector<std::uint8_t> raw = {'B', 'A', 'M', 1};
  put_le(raw, static_cast<std::uint32_t>(text.size()));
  put(raw, reinterpret_cast<const std::uint8_t*>(text.data()), text.size());
  const int references = sam_hdr_nref(header);
  put_le(raw, static_cast<std::uint32_t>(references));
  for (int i = 0; i < references; ++i) {
    const std::string_view name = sam_hdr_tid2name(header, i);
    put_le(raw, static_cast<std::uint32_t>(name.size() + 1));
    put(raw, reinterpret_cast<const std::uint8_t*>(name.data()), name.size());
    raw.push_back(0);
    put_le(raw, static_cast<std::uint32_t>(sam_hdr_tid2len(header, i)));
  }
  BgzfWriter out(kLevel);
  out.write(raw);
  out.flush();
  return out.take_blocks();
}

// Writes the .bai from the records' places in the shards (the first shard
// is the header).
Status write_index(const Source& source, const std::vector<Shard>& shards,
                   std::uint64_t size) {
  // the records start where the header ends (all of the file without a shard)
  const std::uint64_t first = shards.size() > 1 ? shards[1].offset : size;
  const Index index(hts_idx_init(sam_hdr_nref(source.header.get()), HTS_FMT_BAI,
                                 first << 16, kBaiMinShift, kBaiLevels));
  if (!index) return std::unexpected("cannot index " + source.job.output);
  for (std::size_t i = 1; i < shards.size(); ++i) {
    for (const IndexEntry& entry : shards[i].entries) {
      if (hts_idx_push(index.get(), entry.reference, entry.begin, entry.end,
                       (shards[i].offset << 16) + entry.offset,
                       entry.mapped) < 0) {
        return std::unexpected("cannot index " + source.job.output);
      }
    }
  }
  hts_idx_finish(index.get(), size << 16);
  const std::string path = source.job.output + ".bai";
  if (hts_idx_save_as(index.get(), source.job.output.c_str(), path.c_str(),
                      HTS_FMT_BAI) < 0) {
    return std::unexpected("cannot write " + path);
  }
  return {};
}

std::expected<Source, std::string> open_source(const ModbamJob& job,
                                               const ReadRows& rows) {
  const SamFile input(sam_open(job.input.c_str(), "rb"));
  if (!input) return std::unexpected("cannot open " + job.input);
  Header header(sam_hdr_read(input.get()));
  if (!header) return std::unexpected("cannot read the header of " + job.input);
  const std::string_view text(sam_hdr_str(header.get()),
                              sam_hdr_length(header.get()));
  // htslib reports a missing index as an error; here it is not one
  const htsLogLevel log_level = hts_get_log_level();
  hts_set_log_level(HTS_LOG_OFF);
  Index index(sam_index_load(input.get(), job.input.c_str()));
  hts_set_log_level(log_level);
  if (!index) {
    // Without an index the records are written in the order of the input,
    // which is the order DeepRM writes only for a sorted input (it sorts
    // the modBAM of another).
    const std::string_view first_line = text.substr(0, text.find('\n'));
    if (!first_line.starts_with("@HD") ||
        first_line.find("\tSO:coordinate") == std::string_view::npos) {
      return std::unexpected(
          "the modBAM needs a BAM sorted by coordinate (" + job.input +
          " has no index and its header does not say SO:coordinate); pass "
          "--skip-modbam, or sort and index it");
    }
    std::println(stderr,
                 "warning: no index for {}; the modBAM is made by one thread "
                 "(the others only read the BAM)",
                 job.input);
  }
  TagIds ids{header_ids(text, "@RG"), header_ids(text, "@PG")};
  return Source{job, rows, std::move(header), std::move(index), std::move(ids)};
}

std::vector<ModbamTask> plan_tasks(const Source& source) {
  const int references = sam_hdr_nref(source.header.get());
  if (!source.index) {  // one task: the whole input, read from its start
    return {ModbamTask{0, references - 1, 0, kWholeReference}};
  }
  std::vector<std::uint64_t> reads(static_cast<std::size_t>(references), 0);
  std::vector<std::int64_t> lengths(reads.size(), 0);
  std::uint64_t total = 0;
  for (int i = 0; i < references; ++i) {
    if (!source.rows.has_reference(i)) continue;
    std::uint64_t mapped = 0;
    std::uint64_t unmapped = 0;
    const bool counted =
        hts_idx_get_stat(source.index.get(), i, &mapped, &unmapped) == 0;
    reads[static_cast<std::size_t>(i)] = counted ? mapped + unmapped : 1;
    lengths[static_cast<std::size_t>(i)] = sam_hdr_tid2len(source.header.get(), i);
    total += reads[static_cast<std::size_t>(i)];
  }
  // the compressed bytes the index lists for the reads of a window
  const ModbamWeight weight = [&source](std::int32_t reference,
                                        std::int64_t begin, std::int64_t end) {
    const std::unique_ptr<hts_itr_t, FreeIterator> reads(
        sam_itr_queryi(source.index.get(), reference, begin, end));
    std::uint64_t bytes = 0;
    for (int i = 0; reads && i < reads->n_off; ++i) {
      bytes += (reads->off[i].v >> 16) - (reads->off[i].u >> 16) + 1;
    }
    return bytes;
  };
  return plan_modbam_tasks(
      reads, lengths,
      std::max(kSmallestTask, total / (kTasksPerThread * source.job.threads)),
      weight);
}

}  // namespace

bool encode_bam_record(const bam1_t* record, std::vector<std::uint8_t>& out) {
  const bam1_core_t& core = record->core;
  const std::uint32_t name_size = core.l_qname - core.l_extranul;
  const bool long_cigar = core.n_cigar > 0xffff;
  const hts_pos_t span =
      long_cigar ? bam_cigar2rlen(static_cast<int>(core.n_cigar),
                                  bam_get_cigar(record))
                 : 0;
  if (name_size > 255 || core.pos > INT32_MAX || core.mpos > INT32_MAX ||
      core.isize < INT32_MIN || core.isize > INT32_MAX || span >= (1 << 28)) {
    return false;
  }
  out.clear();
  // the fake CIGAR and the "CGBI" tag with its length take 16 bytes more
  put_le(out, static_cast<std::uint32_t>(record->l_data) - core.l_extranul +
                  32 + (long_cigar ? 16 : 0));
  put_le(out, static_cast<std::uint32_t>(core.tid));
  put_le(out, static_cast<std::uint32_t>(core.pos));
  put_le(out, static_cast<std::uint32_t>(core.bin) << 16 |
                  static_cast<std::uint32_t>(core.qual) << 8 | name_size);
  put_le(out, static_cast<std::uint32_t>(core.flag) << 16 |
                  (long_cigar ? 2 : core.n_cigar));
  put_le(out, static_cast<std::uint32_t>(core.l_qseq));
  put_le(out, static_cast<std::uint32_t>(core.mtid));
  put_le(out, static_cast<std::uint32_t>(core.mpos));
  put_le(out, static_cast<std::uint32_t>(core.isize));
  put(out, record->data, name_size);
  const std::size_t data_size = static_cast<std::size_t>(record->l_data);
  if (!long_cigar) {
    put(out, record->data + core.l_qname, data_size - core.l_qname);
    return true;
  }
  // More operations than the record's 16-bit count holds: the CIGAR becomes
  // <read length>S<reference length>N and the real one goes into a CG tag.
  const std::size_t cigar_begin = core.l_qname;
  const std::size_t cigar_end = cigar_begin + std::size_t{core.n_cigar} * 4;
  put_le(out, static_cast<std::uint32_t>(core.l_qseq) << 4 | BAM_CSOFT_CLIP);
  put_le(out, static_cast<std::uint32_t>(span) << 4 | BAM_CREF_SKIP);
  put(out, record->data + cigar_end, data_size - cigar_end);
  put(out, reinterpret_cast<const std::uint8_t*>("CGBI"), 4);
  put_le(out, core.n_cigar);
  put(out, record->data + cigar_begin, cigar_end - cigar_begin);
  return true;
}

std::string merge_order_header(std::string_view text) {
  static constexpr std::string_view kKinds[] = {"@HD", "@SQ", "@RG", "@PG", "@CO"};
  std::string ordered;
  for (const std::string_view kind : kKinds) {
    for (std::string_view rest = text; !rest.empty();) {
      const std::size_t end = std::min(rest.find('\n'), rest.size());
      const std::string_view line = rest.substr(0, end);
      rest.remove_prefix(std::min(end + 1, rest.size()));
      if (line.substr(0, 3) == kind) ordered.append(line).push_back('\n');
    }
  }
  return ordered;
}

std::vector<std::string> header_ids(std::string_view text,
                                    std::string_view kind) {
  std::vector<std::string> ids;
  for (std::string_view rest = text; !rest.empty();) {
    const std::size_t end = std::min(rest.find('\n'), rest.size());
    std::string_view line = rest.substr(0, end);
    rest.remove_prefix(std::min(end + 1, rest.size()));
    if (line.substr(0, 3) != kind || line.substr(3, 1) != "\t") continue;
    // the first ID field of the line
    for (std::size_t tab = line.find('\t'); tab != std::string_view::npos;
         tab = line.find('\t')) {
      line.remove_prefix(tab + 1);
      if (line.starts_with("ID:")) {
        ids.emplace_back(line.substr(3, line.find('\t') - 3));
        break;
      }
    }
  }
  return ids;
}

namespace {

// The windows of one reference for plan_modbam_tasks.
class WindowPlanner {
 public:
  WindowPlanner(std::int32_t reference, std::uint64_t limit,
                const ModbamWeight& weight, std::vector<ModbamTask>& tasks)
      : reference_(reference), limit_(limit), weight_(weight), tasks_(tasks) {}

  // Cuts [begin, end) into windows of at most the limit each; `open`: the
  // last window also takes the reads that start past `end`.
  void cut(std::int64_t begin, std::int64_t end, bool open) {
    const std::uint64_t whole = weight_(reference_, begin, end);
    if (whole <= limit_ || end - begin < 2 * kSmallestWindow) {
      return add(begin, end, open, whole > kDenseTasks * limit_);
    }
    const std::int64_t middle = begin + (end - begin) / 2;
    const std::uint64_t left = weight_(reference_, begin, middle);
    const std::uint64_t right = weight_(reference_, middle, end);
    if (left + right > whole + whole / 2) {
      // The halves weigh as much as the whole: the index does not tell
      // such short windows apart. Equal lengths, then.
      const std::int64_t pieces = std::min<std::int64_t>(
          static_cast<std::int64_t>((whole + limit_ - 1) / limit_),
          (end - begin) / kSmallestWindow);
      const std::int64_t step = (end - begin + pieces - 1) / pieces;
      for (std::int64_t piece = 0; piece < pieces; ++piece) {
        const std::int64_t piece_end = std::min(begin + (piece + 1) * step, end);
        add(begin + piece * step, piece_end, open && piece + 1 == pieces,
            whole > kDenseTasks * limit_ * static_cast<std::uint64_t>(pieces));
      }
      return;
    }
    cut(begin, middle, false);
    cut(middle, end, open);
  }

 private:
  // A shorter window would read mostly reads that start before it.
  // The index tells windows of 16 kb apart; a shorter one would mostly read
  // reads that start before it.
  static constexpr std::int64_t kSmallestWindow = 16384;
  // A window with this many times a task's reads is a dense one.
  static constexpr std::uint64_t kDenseTasks = 3;

  void add(std::int64_t begin, std::int64_t end, bool open, bool dense) {
    tasks_.push_back(ModbamTask{reference_, reference_, begin,
                                open ? kWholeReference : end, dense});
  }

  std::int32_t reference_;
  std::uint64_t limit_;
  const ModbamWeight& weight_;
  std::vector<ModbamTask>& tasks_;
};

}  // namespace

std::vector<ModbamTask> plan_modbam_tasks(
    const std::vector<std::uint64_t>& reads,
    const std::vector<std::int64_t>& lengths, std::uint64_t target,
    const ModbamWeight& weight) {
  std::vector<ModbamTask> tasks;
  std::uint64_t held = 0;  // reads of the task being filled; 0: none open
  for (std::size_t i = 0; i < reads.size(); ++i) {
    if (reads[i] == 0) continue;
    const std::int32_t reference = static_cast<std::int32_t>(i);
    if (reads[i] > target && lengths[i] > 0) {
      // the share of the reference that `target` of its reads take
      const std::uint64_t whole = weight(reference, 0, lengths[i]);
      const std::uint64_t limit = std::max<std::uint64_t>(
          1, static_cast<std::uint64_t>(static_cast<double>(whole) *
                                        static_cast<double>(target) /
                                        static_cast<double>(reads[i])));
      WindowPlanner(reference, limit, weight, tasks).cut(0, lengths[i], true);
      held = 0;
      continue;
    }
    if (held == 0) {
      tasks.push_back(ModbamTask{reference, reference, 0, kWholeReference});
    }
    tasks.back().last = reference;
    held += reads[i];
    if (held >= target) held = 0;
  }
  return tasks;
}

std::expected<bool, std::string> write_modbam(const ModbamJob& job,
                                              const ReadRows& rows) {
  std::expected<Source, std::string> source = open_source(job, rows);
  if (!source) return std::unexpected(source.error());
  const std::vector<ModbamTask> tasks = plan_tasks(*source);
  // a folder of its own, as the Python pileup's (tempfile.mkdtemp): runs that
  // share --tmp-dir or an output folder do not write into each other's
  std::error_code made;
  std::filesystem::create_directories(job.tmp_root, made);
  std::string folder = job.tmp_root + "/.deeprm_modbam_XXXXXX";
  if (::mkdtemp(folder.data()) == nullptr) {
    return std::unexpected("cannot make a temporary folder in " + job.tmp_root);
  }
  const std::string part = job.output + ".part";
  OutputFile file;
  if (Status status = file.open(part); !status) {
    std::filesystem::remove_all(folder);
    return std::unexpected(status.error());
  }
  std::vector<Shard> shards(tasks.size() + 1);  // the header first
  for (std::size_t i = 0; i < shards.size(); ++i) {
    shards[i].path = folder + "/shard_" + std::to_string(i) + ".bgzf";
  }
  ShardSink sink(file, shards);
  shards[0].blocks = header_blocks(*source);
  Status status = sink.finish(0);
  const auto seconds = [start = std::chrono::steady_clock::now()] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start).count();
  };
  if (status) status = write_shards(*source, tasks, shards, sink);
  std::size_t records = 0;
  for (const Shard& shard : shards) records += shard.entries.size();
  if (status) status = file.write_at(sink.size(), bgzf_end_block());
  if (status) status = file.close();
  const double shards_seconds = seconds();
  std::filesystem::remove_all(folder);
  if (!status) {
    std::filesystem::remove(part);
    return std::unexpected(status.error());
  }
  // a modBAM with no record (no read of the BAM has predictions) is written
  // with its header and index, as the Python pileup writes it
  std::filesystem::rename(part, job.output);
  status = write_index(*source, shards, sink.size());
  if (!status) return std::unexpected(status.error());
  std::println(stderr,
               "modBAM: {} records in {} shards; shards {:.1f} s, index "
               "{:.1f} s",
               records, tasks.size(), shards_seconds,
               seconds() - shards_seconds);
  return true;
}

}  // namespace deeprm
