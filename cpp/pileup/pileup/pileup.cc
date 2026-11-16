/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#include "pileup/pileup.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <print>
#include <span>
#include <thread>

#include "pileup/bam_header.h"
#include "pileup/file_sites.h"
#include "pileup/genome_map.h"
#include "pileup/genomic_table.h"
#include "pileup/modbam.h"
#include "pileup/numpy_math.h"
#include "pileup/read_rows.h"
#include "pileup/run_info.h"
#include "pileup/site_scores.h"
#include "pileup/site_table.h"
#include "pileup/table_writer.h"

namespace deeprm {
namespace {

// Files whose label_ids are sampled to share the sites between the tables.
constexpr std::size_t kSampleFiles = 256;
// Site tables per thread: more tables than threads evens out their sizes.
constexpr std::size_t kTablesPerThread = 4;
constexpr std::size_t kMaxTables = 1024;
// Files read, per thread, before their sites are added to the tables.
constexpr std::size_t kBatchFilesPerThread = 16;

using Clock = std::chrono::steady_clock;

// The memory the process holds now, in GB (0 where /proc does not tell).
double memory_gb() {
  std::ifstream status("/proc/self/statm");
  std::size_t total = 0;
  std::size_t resident = 0;
  status >> total >> resident;
  return static_cast<double>(resident) * 4096.0 / 1e9;
}

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

// The state of one pileup run.
struct Run {
  const Options& options;
  std::vector<std::string> files;
  std::vector<std::int32_t> groups;
  std::vector<std::string> references;
  std::vector<std::int64_t> bounds;  // table i holds bounds[i-1] <= id < bounds[i]
  std::vector<SiteTable> tables;
  std::vector<std::unique_ptr<FileSiteReader>> readers;  // one per thread
  std::vector<PackedRows> read_rows;  // per file; without --skip-modbam
  std::optional<GenomeMap> genome;  // with --annot
  // genomic[table][chromosome]: the table's sites that map to the genome
  std::vector<std::vector<std::vector<GenomicRow>>> genomic;
  double read_seconds = 0;  // reading the files and adding up their rows
  double add_seconds = 0;   // adding the files' sites to the tables
};

// At most `tables` - 1 label_ids that split the sites into ranges with
// about the same number of rows each, from a sample of the files. They
// change only how the work is shared, not the sums.
Status sample_bounds(Run& run, std::size_t tables) {
  const std::size_t step = std::max<std::size_t>(1, run.files.size() / kSampleFiles);
  std::vector<std::string> sample;
  for (std::size_t i = 0; i < run.files.size() && sample.size() < kSampleFiles;
       i += step) {
    sample.push_back(run.files[i]);
  }
  std::vector<std::vector<std::int64_t>> labels(sample.size());
  const Status read = parallel_for(
      run.readers.size(), sample.size(), [&](std::size_t worker, std::size_t i) {
        return run.readers[worker]->read_labels(sample[i], labels[i]);
      });
  if (!read) return read;
  std::vector<std::int64_t> all;
  for (const std::vector<std::int64_t>& part : labels) {
    all.insert(all.end(), part.begin(), part.end());
  }
  std::ranges::sort(all);
  for (std::size_t i = 1; i < tables && !all.empty(); ++i) {
    const std::int64_t bound = all[i * all.size() / tables];
    if (run.bounds.empty() || run.bounds.back() != bound) {
      run.bounds.push_back(bound);
    }
  }
  return {};
}

// The sites of `sites` (sorted by label_id) that belong to table `table`.
std::span<const FileSite> sites_of_table(const Run& run,
                                         const std::vector<FileSite>& sites,
                                         std::size_t table) {
  const auto first_at_or_above = [&](std::int64_t bound) {
    return std::ranges::lower_bound(sites, bound, {}, &FileSite::label_id);
  };
  const auto begin =
      table == 0 ? sites.begin() : first_at_or_above(run.bounds[table - 1]);
  const auto end = table == run.bounds.size()
                       ? sites.end()
                       : first_at_or_above(run.bounds[table]);
  return {begin, end};
}

using FileBytes = std::vector<std::vector<std::uint8_t>>;

// The bytes of the files [first, end). As many threads read as compute:
// more of them did not read files that are not in the system's cache any
// faster (on Lustre the client sends the servers only a few requests at a
// time, whatever the number of threads), and cost system time when they are.
Status read_files(const Run& run, std::size_t first, std::size_t end,
                  FileBytes& bytes) {
  return parallel_for(run.readers.size(), end - first,
                      [&](std::size_t, std::size_t i) {
                        return read_file_bytes(run.files[first + i], bytes[i]);
                      });
}

// Adds the sites of the files [first, end), whose bytes are `bytes`, to the
// tables: the files in parallel, then the tables in parallel, each in file
// order.
Status add_batch(Run& run, std::size_t first, std::size_t end,
                 const FileBytes& bytes,
                 std::vector<std::vector<FileSite>>& batch) {
  const std::size_t threads = run.readers.size();
  const Clock::time_point start = Clock::now();
  const Status read = parallel_for(
      threads, end - first, [&](std::size_t worker, std::size_t i) {
        return run.readers[worker]->read(
            bytes[i], run.files[first + i], batch[i],
            run.read_rows.empty() ? nullptr : &run.read_rows[first + i]);
      });
  if (!read) return read;
  run.read_seconds += seconds_since(start);
  const Clock::time_point read_end = Clock::now();
  const Status added = parallel_for(
      threads, run.tables.size(), [&](std::size_t, std::size_t table) {
        for (std::size_t i = 0; i < end - first; ++i) {
          run.tables[table].add(sites_of_table(run, batch[i], table),
                                run.groups[first + i]);
        }
        return Status{};
      });
  run.add_seconds += seconds_since(read_end);
  return added;
}

Status add_all_files(Run& run) {
  const std::size_t batch_files = kBatchFilesPerThread * run.readers.size();
  const std::size_t files = run.files.size();
  std::vector<std::vector<FileSite>> batch(batch_files);
  FileBytes bytes(batch_files);
  FileBytes ahead(batch_files);
  // the next batch is read while the sites of this one are added up
  Status read = read_files(run, 0, std::min(batch_files, files), ahead);
  for (std::size_t first = 0; first < files; first += batch_files) {
    if (!read) return read;
    const std::size_t end = std::min(first + batch_files, files);
    bytes.swap(ahead);
    Status added;
    {
      const std::jthread reader([&] {
        read = read_files(run, end, std::min(end + batch_files, files), ahead);
      });
      added = add_batch(run, first, end, bytes, batch);
    }
    if (!added) return added;
  }
  return {};
}

// The table rows of sites sorted by label_id (_decode_label_id).
std::expected<std::vector<TableRow>, std::string> decode_rows(
    const Run& run, const std::vector<LabeledSums>& sites) {
  std::vector<TableRow> rows;
  rows.reserve(sites.size());
  for (const LabeledSums& site : sites) {
    const bool minus = site.label_id < 0;
    const std::int64_t packed = (minus ? -site.label_id : site.label_id) - 1;
    const std::int64_t reference = packed / run.options.label_div;
    if (reference >= static_cast<std::int64_t>(run.references.size())) {
      return std::unexpected(
          "label_id contains transcript IDs outside the BAM reference range.");
    }
    rows.push_back(TableRow{.reference = static_cast<std::uint32_t>(reference),
                            .position = packed % run.options.label_div,
                            .minus = minus,
                            .sums = site.sums});
  }
  return rows;
}

// Keeps the rows that map to the genome, per chromosome, in their order.
void map_to_genome(const GenomeMap& genome, const std::vector<TableRow>& rows,
                   std::vector<std::vector<GenomicRow>>& by_chrom) {
  // counted first, so that each chromosome's rows take no spare room
  std::vector<std::size_t> counts(genome.chroms().size(), 0);
  for (const TableRow& row : rows) {
    const std::optional<GenomicSite> site =
        genome.map(row.reference, row.position, row.minus);
    if (site) ++counts[site->chrom];
  }
  by_chrom.resize(counts.size());
  for (std::size_t chrom = 0; chrom < counts.size(); ++chrom) {
    by_chrom[chrom].reserve(counts[chrom]);
  }
  for (const TableRow& row : rows) {
    const std::optional<GenomicSite> site =
        genome.map(row.reference, row.position, row.minus);
    if (!site) continue;
    by_chrom[site->chrom].push_back(GenomicRow{
        .position = site->position, .minus = site->minus, .sums = row.sums});
  }
}

// Scores the sites of every table and writes pileup.bed and pileup.npz.
Status write_site_table(Run& run, const TableFormat& format,
                        TableWriter& writer) {
  const std::size_t threads = run.readers.size();
  const std::size_t step = threads;
  for (std::size_t first = 0; first < run.tables.size(); first += step) {
    const std::size_t count = std::min(step, run.tables.size() - first);
    std::vector<TablePart> parts(count);
    const Status made =
        parallel_for(threads, count, [&](std::size_t, std::size_t i) -> Status {
          const std::expected<std::vector<TableRow>, std::string> rows =
              decode_rows(run, run.tables[first + i].finish());
          if (!rows) return std::unexpected(rows.error());
          parts[i] = format.make_part(*rows);
          if (run.genome) map_to_genome(*run.genome, *rows, run.genomic[first + i]);
          return {};
        });
    if (!made) return made;
    if (Status status = writer.append(parts); !status) return status;
  }
  return writer.finish();
}

// The table rows of one chromosome from the rows the site tables mapped to
// it, in table order; frees those rows.
std::vector<TableRow> chrom_rows(Run& run, std::size_t chrom,
                                 std::uint32_t reference) {
  std::vector<std::span<const GenomicRow>> parts;
  for (const std::vector<std::vector<GenomicRow>>& table : run.genomic) {
    if (!table.empty()) parts.emplace_back(table[chrom]);
  }
  std::vector<TableRow> rows = add_up_genomic_rows(parts, reference);
  for (std::vector<std::vector<GenomicRow>>& table : run.genomic) {
    if (!table.empty()) std::vector<GenomicRow>().swap(table[chrom]);
  }
  return rows;
}

// The site tables are written as .part files and given their names once every
// output is written, as the Python pileup moves its tables in place last: a run
// that fails leaves none of them, and this guard removes their .part files.
class PartFiles {
 public:
  PartFiles() = default;
  PartFiles(const PartFiles&) = delete;
  PartFiles& operator=(const PartFiles&) = delete;
  ~PartFiles() {
    std::error_code error;
    for (const std::string& stem : stems_) {
      std::filesystem::remove(stem + ".part.bed", error);
      std::filesystem::remove(stem + ".part.npz", error);
    }
  }
  void add(std::string stem) { stems_.push_back(std::move(stem)); }
  void commit() {
    for (const std::string& stem : stems_) {
      std::filesystem::rename(stem + ".part.bed", stem + ".bed");
      std::filesystem::rename(stem + ".part.npz", stem + ".npz");
    }
    stems_.clear();
  }

 private:
  std::vector<std::string> stems_;
};

// Adds up the mapped sites per genomic position, a chromosome per thread,
// and writes genomic_pileup<postfix>.part.bed and .part.npz (named by
// PartFiles once every output is written).
Status write_genomic_table(Run& run, const SiteScorer& scorer,
                           std::size_t& near_steps) {
  std::vector<std::size_t> chroms;  // those with sites
  std::vector<std::string> names;
  for (std::size_t chrom = 0; chrom < run.genome->chroms().size(); ++chrom) {
    const bool used = std::ranges::any_of(run.genomic, [&](const auto& table) {
      return !table.empty() && !table[chrom].empty();
    });
    if (!used) continue;
    chroms.push_back(chrom);
    names.push_back(run.genome->chroms()[chrom]);
  }
  if (chroms.empty()) {
    return std::unexpected(
        "No valid genomic position found. Please verify the annotation file.");
  }
  const TableFormat format(names, scorer);
  TableWriter writer(format, run.options.threads);
  const std::string stem =
      run.options.output + "/genomic_pileup" + run.options.postfix;
  if (Status status = writer.open(stem + ".part"); !status) return status;
  std::vector<TablePart> parts(chroms.size());
  const Status made = parallel_for(
      run.options.threads, chroms.size(), [&](std::size_t, std::size_t i) {
        parts[i] = format.make_part(
            chrom_rows(run, chroms[i], static_cast<std::uint32_t>(i)));
        return Status{};
      });
  if (!made) return made;
  if (Status status = writer.append(parts); !status) return status;
  if (Status status = writer.finish(); !status) return status;
  near_steps += writer.near_steps();
  std::println(stderr, "genomic pileup: {} sites on {} chromosomes",
               writer.rows(), chroms.size());
  return {};
}

// Reads the annotation and tells how many references it has.
Status prepare_genome(Run& run) {
  const std::expected<std::vector<Transcript>, std::string> transcripts =
      read_annotation(run.options.annot);
  if (!transcripts) return std::unexpected(transcripts.error());
  run.genome.emplace(run.references, *transcripts);
  std::println(stderr,
               "genomic pileup: {} of {} references matched the annotation by "
               "name, {} after removing the version",
               run.genome->matched_exactly(), run.references.size(),
               run.genome->matched_without_version());
  return {};
}

// Writes modbam_<input BAM name> from the per-read rows of all files.
Status write_modbam_output(Run& run) {
  const Options& options = run.options;
  run.tables.clear();
  const ReadRows rows(std::move(run.read_rows), options.threads);
  const std::string name =
      "modbam_" + std::filesystem::path(options.bam).filename().string();
  const std::expected<bool, std::string> written = write_modbam(
      ModbamJob{.input = options.bam,
                .output = options.output + "/" + name,
                .tmp_root = options.tmp_dir.empty() ? options.output
                                                    : options.tmp_dir,
                .threads = options.threads},
      rows);
  if (!written) return std::unexpected(written.error());
  if (!*written) {
    std::println(stderr, "warning: no read of the BAM has predictions; {} was "
                         "not written", name);
  }
  return {};
}

Status prepare(Run& run, bool calibrated) {
  const Options& options = run.options;
  run.files = list_input_files(options.input);
  if (run.files.empty()) {
    return std::unexpected("No inference .npz files found in: " + options.input);
  }
  run.groups = worker_groups(run.files.size(),
                             std::min(options.threads, run.files.size()));
  std::expected<std::vector<std::string>, std::string> references =
      read_reference_names(options.bam);
  if (!references) return std::unexpected(references.error());
  run.references = std::move(*references);
  for (std::size_t i = 0; i < options.threads; ++i) {
    run.readers.push_back(std::make_unique<FileSiteReader>(
        ReadOptions{.slice = options.slice,
                    .flip = options.flip,
                    .label_div = options.label_div},
        TermOptions{.threshold = options.threshold,
                    .epsilon = options.epsilon,
                    .calibrated = calibrated}));
  }
  const std::size_t tables =
      std::min(kTablesPerThread * options.threads, kMaxTables);
  if (Status status = sample_bounds(run, tables); !status) return status;
  run.tables.resize(run.bounds.size() + 1);
  run.genomic.resize(run.tables.size());
  if (!options.skip_modbam) run.read_rows.resize(run.files.size());
  if (!options.annot.empty()) return prepare_genome(run);
  return {};
}

void report(const Options& options, const SiteScorer& scorer,
            std::size_t near_steps) {
  if (!scorer.threshold_is_recorded()) {
    std::println(stderr,
                 "warning: --threshold {} is not the default; log10(1 - "
                 "threshold) comes from the C library and may differ from "
                 "NumPy's in the last bit",
                 options.threshold);
  }
  if (near_steps > 0) {
    std::println(stderr,
                 "warning: {} sites lie within 1e-14 of a modscore step; "
                 "their modscore may differ from the Python pileup's by 0.001",
                 near_steps);
  }
}

}  // namespace

std::vector<std::int32_t> worker_groups(std::size_t files, std::size_t groups) {
  std::vector<std::int32_t> group_of;
  group_of.reserve(files);
  for (std::size_t group = 0; group < groups; ++group) {
    // np.array_split: the first files % groups groups get one file more
    const std::size_t size = files / groups + (group < files % groups ? 1 : 0);
    group_of.insert(group_of.end(), size, static_cast<std::int32_t>(group));
  }
  return group_of;
}

std::vector<std::string> list_input_files(const std::string& input_dir) {
  std::vector<std::string> files;
  std::error_code error;
  for (const std::filesystem::directory_entry& entry :
       std::filesystem::directory_iterator(input_dir, error)) {
    const std::string name = entry.path().filename().string();
    // glob's "*.npz" does not match hidden files
    if (name.ends_with(".npz") && !name.starts_with(".")) {
      files.push_back(entry.path().string());
    }
  }
  std::ranges::sort(files);
  return files;
}

Status run_pileup(const Options& options) {
  const Clock::time_point start = Clock::now();
  std::filesystem::create_directories(options.output);
  Run run{.options = options, .files = {}, .groups = {}, .references = {},
          .bounds = {}, .tables = {}, .readers = {},
          .read_rows = {}, .genome = {}, .genomic = {}, .read_seconds = 0, .add_seconds = 0};
  // DeepRM decides it with its own reading of deeprm_run.json and passes it
  const bool calibrated = options.calibrated ? *options.calibrated
                                             : inputs_calibrated(options.input);
  if (calibrated && !calibration_is_verified()) {
    std::println(stderr,
                 "warning: this C library computes log or log1p differently "
                 "from the one deeprm_pileup was checked with; the "
                 "stoichiometry of a site may differ from DeepRM's in a last "
                 "bit");
  }
  if (Status status = prepare(run, calibrated); !status) return status;
  std::println(stderr, "[{:8.1f}s] {} files, {} references, {} site tables",
               seconds_since(start), run.files.size(), run.references.size(),
               run.tables.size());
  if (Status status = add_all_files(run); !status) return status;
  std::println(stderr,
               "[{:8.1f}s] sites added up (files {:.1f} s, tables {:.1f} s); "
               "{:.1f} GB held",
               seconds_since(start), run.read_seconds, run.add_seconds,
               memory_gb());
  const SiteScorer scorer(options.threshold, options.epsilon,
                          !options.report_all_stoichiometry);
  const TableFormat format(run.references, scorer);
  PartFiles parts;  // before the writer: the writer closes its files first
  TableWriter writer(format, options.threads);
  const std::string stem = options.output + "/pileup";
  parts.add(stem);
  if (Status status = writer.open(stem + ".part"); !status) return status;
  if (Status status = write_site_table(run, format, writer); !status) {
    return status;
  }
  if (writer.rows() == 0) {
    return std::unexpected(
        "No valid pileup entries were produced from the inference outputs.");
  }
  std::size_t near_steps = writer.near_steps();
  if (run.genome) {
    std::println(stderr, "[{:8.1f}s] site table written; {:.1f} GB held",
                 seconds_since(start), memory_gb());
    parts.add(options.output + "/genomic_pileup" + options.postfix);
    if (Status status = write_genomic_table(run, scorer, near_steps); !status) {
      return status;
    }
  }
  report(options, scorer, near_steps);
  if (!options.skip_modbam) {
    std::println(stderr, "[{:8.1f}s] site tables written; {:.1f} GB held",
                 seconds_since(start), memory_gb());
    if (Status status = write_modbam_output(run); !status) return status;
  }
  parts.commit();
  std::println(stderr, "[{:8.1f}s] Pileup finished: {} sites",
               seconds_since(start), writer.rows());
  return {};
}

}  // namespace deeprm
