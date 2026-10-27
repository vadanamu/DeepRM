# 📜 Changelog

## [1.2] - 2026-10-27

Changes since 1.1.3.

### Highlights
- A new m6A model with higher recall, most of all for clustered m6A: 95.7 %, against 85.8 % with DeepRM v1.1.
- Inference 1.9 to 3.0 times as fast as DeepRM v1.1 on A100, H100 and B200 GPUs.
- A new C++ preprocessing: 8 to 13 times less peak memory than v1.1's, and 2 to 2.5 times as fast with input on a
  local SSD.
- One-command install: `pip install deeprm` also installs PyTorch 2.10.0 and, on Linux x86_64, FlashAttention-4.

### Benchmarks
#### m6A detection
| Benchmark | DeepRM v1.2 | DeepRM v1.1 | Dorado (v6.0.0 SUP) |
|---|---|---|---|
| Recall, clustered m6A (nearest other m6A 1–4 nt away) | **95.7 %** | 85.8 % | 78.9 % |
| Recall, isolated m6A (nearest other m6A 20 nt or more away) | **98.9 %** | 98.0 % | 94.0 % |
| ONT all-5-mer synthetic RNA dataset, AU-ROC | **0.977** | 0.969 | 0.971 |
| ONT all-5-mer synthetic RNA dataset, AU-PR | **0.933** | 0.883 | 0.923 |

- Recall on fully m6A-modified in vitro transcribed (IVT) RNA, at a false positive rate of 0.0054 % on unmodified IVT
  RNA. All-5-mer dataset: [Oxford Nanopore Technologies open data](https://epi2me.nanoporetech.com/rna-mod-validation-all5mer-2026.07/), CC BY-NC 4.0.
- HEK293T benchmark of the DeepRM paper (sites called m6A by at least three of four methods, depth ≥ 20; all tools on
  the same sites): AU-PR at DRACH / non-DRACH sites 0.964 / 0.716 (DeepRM v1.2), 0.961 / 0.704 (DeepRM v1.1) and
  0.959 / 0.161 (Dorado); AU-ROC within 0.01 of both.
- Full benchmark results and methods are reserved for a future publication.

#### Inference speed compared with DeepRM v1.1
| GPU | Model only, one GPU | Whole library | Time, DeepRM v1.2 | Time, DeepRM v1.1 |
|---|---|---|---|---|
| NVIDIA B200 | 2.98× | 2.66× (8 GPUs) | 1.51 h | 4.02 h |
| NVIDIA H100 | 1.88× | 1.93× (8 GPUs) | 2.72 h | 5.25 h |
| NVIDIA A100 | 2.08× | 2.21× (4 GPUs) | 9.16 h | 20.22 h |

- Model only: the model alone, without reading files. Whole library: inference on one full sequencing library, without
  the pileup.

#### Preprocessing (`deeprm call prep`)
| Setting | DeepRM v1.1 C++ binary | DeepRM v1.2 C++ binary | Python preprocessing |
|---|---|---|---|
| Sorted BAM file, 224 threads, 2 TB memory | 79 min, 785 GB | **32 min, 103 GB** | 109 min, 250 GB |
| BAM streamed from Dorado, 224 threads, 2 TB memory | 63 min, 1,211 GB | **31 min, 95 GB** | — |
| Sorted BAM file, 112 threads, 256 GB memory | out of memory | **49 min, 61 GB** | 167 min, 119 GB |

- Wall-clock time and peak memory, with input on a local SSD. On network storage the time gain is smaller. Run on
  Dorado's output while Dorado basecalled, the v1.2 preprocessing finished with Dorado: 142 min for both together, and
  142 min for Dorado alone.
- Python preprocessing, DeepRM v1.1 against v1.2 on the call example (peak memory of all processes): 70 GiB against
  3.1 GiB with 8 workers, and 87 GiB against 4.0 GiB with 16 workers. DeepRM v1.1's Python preprocessing gives every
  worker its own copy of the BAM data, so its memory grows with the BAM size times the number of workers.

### Added
- Inference engines chosen automatically for each GPU, with FlashAttention-4 on A100, H100, H200 and B200 GPUs.
- `deeprm check` reports the inference engine, whether the C++ binary starts, and PyTorch or driver problems with
  their fixes.
- `molecule-level/deeprm_run.json`: the model and settings of a run. `--resume` refuses to continue if the model
  file, the input files, the device type, the number of GPU processes, the engine or `--output-id` changed.
- `deeprm call prep --bam-cmd`: runs a basecaller or aligner command and reads its BAM output.
- `deeprm call prep --zero-pad` and `--edge-pad` (keep sites near read ends) and `--trim` (trim low-quality read ends).
- `deeprm train prep --method lev|pw`, `--di` (designs with two A per block), `--edge-blocks` and `--reuse-blocks-dir`.
- The `train` extra for training: `pip install "deeprm[train]"` (torchmetrics 1.0 or newer).

### Changed
- New default model (`deeprm.model.deeprm_v12`) with one output, the m6A probability. It expects Dorado 2.0.0
  `rna004_sup@v6.0.0` basecalls made with `--chunksize 12000`.
- `deeprm call run` runs only v1.2 model files. For v1.1 model files, use `pip install deeprm==1.1.3`.
- Stoichiometry (BED column 11) is reported only at sites where the reads decide it; other sites get 0.
- Each GPU process uses its own GPU as the default device (before, every process also used memory on GPU 0).
- A batch that does not fit in GPU memory runs in smaller parts, with one warning.
- `deeprm call prep`: a new C++ binary and a new Python preprocessing, which give the same rows.
- The C++ binary needs no system library other than glibc (the v1.1 binary needed libssl 1.1).
- When the binary cannot start, `deeprm call prep` says why and uses the Python preprocessing.
- `deeprm call prep`, `deeprm call run`, `deeprm train prep` and `deeprm train compile` refuse to write over the
  results of an earlier run (`deeprm call run --resume` continues a stopped run).
- `deeprm train prep`: a new context-block extractor; `--min`/`--max` default to 200/2,000 (were 0/1,000).
- `deeprm train run` trains the v1.2 architecture (deeprm_v12) and needs a CUDA GPU; `--loss` takes only `BCEWL`.
- Genomic coordinates (`--annot`) are computed faster, and transcript names with a version also match the annotation
  without it.
- An unsorted input BAM is sorted into `<name>.sorted.bam` beside the input for the pileup, and reused.
- modBAM: keeps the input read group and `@PG` lines, and always writes valid `MM`, `ML` and `MN` tags.
- Install: PyTorch 2.10.0 and FlashAttention-4 are dependencies; Python 3.10 or newer and glibc 2.28 or newer are needed.
- A refused command prints `deeprm <group>: error: ...` and exits with status 2 (`DEEPRM_TRACEBACK=1` shows the
  traceback).
- The licence in the package metadata is CC BY-NC-SA 4.0, as in `LICENSE.md` (it said MIT).
- The call example BAM holds Dorado `rna004_sup@v6.0.0` basecalls.

### Fixed
- A NaN or infinite input value no longer turns the other predictions of its batch into NaN.
- An unreadable input shard is skipped with a warning, and a stopped run leaves no cut-off per-read file.
- Preprocessing fails the run, instead of exiting 0, on an unreadable POD5 file, a cut-off BAM or missing MD tags.
- The binary places the sites of reads with a soft clip next to a hard clip on the right bases (1.1.3 did not).
- Reads with a damaged move table or signal are skipped instead of stopping the run.
- Pileup: sites on the strand opposite the transcript get their own genomic rows.
- The input BAM is never overwritten by sorting, and many-core machines no longer run out of open files.
- `deeprm train run` no longer fails with `AttributeError`.

### Removed
- The v1.1 model code (`deeprm.model.deeprm_model`), `deeprm.utils.activations`, `deeprm.utils.memory`, and several
  v1.1 functions of the preprocessing, pileup and training-data modules.
- `deeprm train prep` options `--rbq` (use `--qcut`), `--cbq`, `--fi`, `--chunk` (`-n` is now `--process-once`),
  `--max-size` (`-m`), `--min-size` (`-i`), `--keep` and `--resume`.
- `deeprm train run` options `--no-bq` and `--t-act`, and `--loss BCE`.

### Known limits
- `deeprm train` does not reproduce the v1.2 model, whose training data and methods are reserved for a future
  publication.
- Tested on B200, H100 and A100 GPUs and on the CPU. Other NVIDIA GPUs are supported but were not tested. AMD GPUs are
  not supported.
- Repeated runs on the same machine with `fa4` (the default on A100, H100 and B200) or `eager` give bit-identical
  per-read output. Per-read probabilities differ slightly between GPU types, between engines, and between runs of the
  `fast` engine (troubleshooting page).
- PyTorch 2.10.0 and FlashAttention-4 4.0.0b35 (a beta release) are pinned, so DeepRM needs its own environment.
- An unreadable POD5 file stops `deeprm call prep` and `deeprm train prep`; move it out of the folder to go on.
- A basecaller that fails behind `-b -` is not detected; `--bam-cmd` detects it.
- Supplementary alignments with a hard clip, kept by the default `--filter-flag 276`, get the signal features of other
  bases; `--filter-flag 2324` skips them.

## [1.1.3] - 2026-04-14

### Changed
- The modBAM sort in `deeprm call pileup` uses at most about half of the free memory (it used 4 GB per thread).

### Fixed
- `deeprm call pileup` used far more memory than needed, most of all when writing the modBAM file.
- `deeprm call run --resume` skipped input files that were never processed when more than one data-loading worker
  was used (`--worker`, 4 per GPU by default).
- The Python preprocessing and `deeprm call pileup` no longer fail on a BAM file without an index: they index it, or
  first sort it into `<name>.sorted.bam` if it is not sorted by coordinate.
- The usage line of `deeprm --help` lists the command groups as call, train, qc and check (it said inference, train
  and qc).

### Known limits
- modBAM: a read whose sites are in more than one per-read file can be written more than once, each copy with only
  part of its sites (1.1.2 wrote such a read once).

## [1.1.2] - 2026-04-09

### Changed
- BED column 5 is modscore × 1000, from 0 to 1000 (it was modscore × 100).
- `deeprm call run` writes one per-read file per input file in `molecule-level/`; `--flush` no longer sets how often
  files are written.
- `--prefetch` also sets how many input files each data-loading worker reads at once (it was fixed at 16).
- With `--skip-modbam`, the pileup no longer collects per-read data for the modBAM file, so it uses less memory and
  time.
- `deeprm call run`, `deeprm call pileup` and the Python preprocessing check option values, including `--gpu` and
  `--gpu-pool`, and stop with a clear message on a bad value, missing POD5 or prediction files, or a failed worker.
- C++ preprocessing source in the GitHub repository (`cpp/`): lower peak memory, a `--max-queue` (`-Q`) option, and a
  fix for a crash when there are fewer POD5 files than threads. The binary in the package is still the one from
  1.1.1, without these changes.

### Fixed
- `deeprm call run --resume` continues after the last finished input file (before, it always started over). Per-read
  files after the first missing one are removed.
- `deeprm call run` scores the last, part-filled batch on each GPU (from 1.0.3 to 1.1.1, up to one batch per GPU was
  left out), and no longer fails when a GPU gets less than one batch of data.
- `deeprm call run --gpu-pool` with GPU IDs other than the first ones (for example `--gpu-pool 2 3` or
  `--gpu-pool 0 2`) no longer leaves out some input files or reads some of them twice.
- `deeprm call run` runs on the CPU when no GPU is found or with `--gpu 0`, and `--worker 0` works (both stopped with
  an error).
- `pileup.bed` and `pileup.npz` give the strand as `+`/`-` (was `1`/`-1`), and `genomic_pileup.bed` gives positions
  as whole numbers (they had two decimals).
- BED column 12 (modified reads) is rounded instead of cut off, and column 13 (unmodified reads) is the read count
  minus column 12 (it was the read count minus the stoichiometry).
- The genomic pileup reads annotation files whose exon lists have no trailing comma (such transcripts were dropped).
- modBAM: a probability of 1.0 was written as 0 in the `ML` tag (scores are now the probability × 255, rounded).
- modBAM: `MM` tags are valid for reads whose sites are spread over several per-read files, sites whose read base is
  not an A are left out, and reads with a name that is not a UUID, or with no sequence, are skipped instead of
  stopping the pileup.
- `--slice` (pick one output of a multi-output model) no longer fails in the pileup.
- Python preprocessing: a failed worker stops the run (before, its reads were silently left out); reads with no move
  table or unusable signal are skipped, and so are reads whose move table stride is not `--sampling` (with a warning
  that counts them); a `--thread` smaller than `--bam-thread` no longer fails.

## [1.1.1] - 2026-03-25

Released on GitHub as v1.1.1.r3 (2026-03-20) and v1.1.1.r6 (2026-03-25), and on PyPI as `1.1.1.post5` and
`1.1.1.post6` (2026-03-25). There was no 1.1.0 release.

### Added
- `--skip-modbam` (`-sm`) for `deeprm call run` and `deeprm call pileup`: write the site-level BED file without the
  modBAM file. It came with the 2026-03-25 releases; in `1.1.1.post5`, `deeprm call pileup` spelled it
  `--skip_modbam` (`-k`).

### Changed
- The model weights are the same as in 1.0.x; what changed is how the site-level results are computed.
- modscore is a score from 0 to 1 in steps of 0.001 (it was in arbitrary units), in the transcript and the genomic
  (`--annot`) pileup. BED column 5 (modscore × 100) now runs from 0 to 100.
- Stoichiometry is reported as 0 at sites where the reads called modified (probability ≥ `--threshold`, default 0.98)
  give too little support for it.
- The C++ preprocessing in `deeprm call prep` spreads the reads over all threads when there are fewer POD5 files than
  threads (it used one thread per POD5 file), and reads the BAM file with `--bam-thread` threads (default 4; it used
  one). Its log lines start with a timestamp.

### Fixed
- Reads that Dorado split from one parent read and that aligned to the same reference were treated as one read. Each
  read now keeps its own ID, also in the modBAM file.
- Writing the modBAM file no longer fails on reads with no scored A, or with scored positions that are not an A
  (failures since 1.0.9).

### Known limits
- From 1.1.1 to 1.1.3, the C++ binary in the package can crash when there are fewer POD5 files than threads (fixed in
  the 1.1.2 source, not in the binary). `deeprm call prep` then falls back to the slower Python preprocessing, which
  cannot read a streamed BAM (`-b -`).

## [1.0.9] - 2026-02-26

### Changed
- Requires pandas below 3.0 and numpy 2.0 or newer.
- Docs: the example that streams Dorado output into `deeprm call prep` also saves a sorted, indexed BAM file with
  samtools.
- Docs: the citation is the published paper in Nature Communications (doi:10.1038/s41467-025-67417-w).

### Fixed
- modBAM: the `MM` and `ML` tags have one entry per scored site each, as the SAM specification requires (`ML` now
  also holds scores of 0).
- modBAM: sites at reference positions with no read base (deletions) are left out, so they no longer shift the other
  sites.
- modBAM: `MM` tags use `A+a?` and end with `;`, so adenosines that were not scored count as unknown, not unmodified.

## [1.0.8] - 2025-12-03

### Changed
- Install docs: PyTorch 2.3 or newer is needed (was 2.0 or newer).

### Fixed
- The Python preprocessing (used when the C++ binary is missing or fails) finds the POD5 files when `--pod5` is a
  folder. From 1.0.2 to 1.0.7 it found none.

## [1.0.7] - 2025-11-18

### Added
- modBAM output: `deeprm call run` and `deeprm call pileup` write `modbam_<input BAM name>`, sorted and indexed, with
  m6A calls in the `MM` and `ML` tags.
- Docs: a Zenodo DOI badge for the DeepRM software archive.

### Fixed
- BED column 4 (the modification code) is `a`, the code for m6A. Version 1.0.6 wrote `m`, the code for 5mC.

## [1.0.6] - 2025-11-11

### Changed
- The site-level BED file follows the bedMethyl format (18 columns) instead of DeepRM's own 9 columns. Columns 14 to
  18 are zeros.
- BED column 5 is modscore × 100 and column 11 is the percentage of modified reads. Values have 2 decimals (were 6).

### Fixed
- Docs: the `deeprm call prep` examples in the README and on the usage page use the right options
  (`-p <POD5 dir> -b <BAM> -o <output dir>`) and can be copied as written, and the bundled model file is shown at its
  real path, `weight/deeprm_weights.pt`.

## [1.0.5] - 2025-09-26

### Fixed
- Pileup: sites on the minus strand were reported 2 bases from their true position. This mostly affected reads
  aligned to a genome. The label ID decoding example in the docs has the same fix.

## [1.0.4] - 2025-09-13

### Added
- Docs: the format of the per-read `.npz` output, with Python code to turn read IDs back into UUIDs and to decode
  label IDs into reference, position and strand.

### Fixed
- `deeprm call run` could overwrite a per-read output file with the last file of the same GPU process, losing those
  predictions (since 1.0.3).

## [1.0.3] - 2025-09-12

### Changed
- `deeprm call run --batch` (`-s`) defaults to 16,000 (was 10,000).

### Fixed
- `deeprm call run` predicted the rows left over at the end of each chunk of input data (fewer than one batch) twice,
  so they were repeated in the per-read output and counted twice in the BED file.

## [1.0.2] - 2025-09-04

### Changed
- `deeprm call pileup` reads only the `.npz` files in its input folder (it also read `.pkl` and `.tsv` files).

### Fixed
- `deeprm call run` runs the model. In 1.0.1 it skipped inference and ran only the pileup.
- `deeprm call prep` no longer stops with a permission error when the bundled C++ binary was installed without
  execute permission.
- `deeprm call run` and `deeprm train run` skip an unreadable `.npz` input file with a warning instead of stopping.
- The help of `deeprm call` shows `deeprm call` instead of `deeprm inference`.

## [1.0.1] - 2025-09-01

### Added
- First public release: m6A detection from Oxford Nanopore direct RNA sequencing (POD5 files and a Dorado BAM file
  with move tables), with a trained model in the package.
- `pip install deeprm` on Linux x86_64 with Python 3.9 to 3.12; install PyTorch 2.0 or newer first.
- `deeprm call prep`, `call run` and `call pileup`: per-read predictions and a site-level BED file with a modification
  score, stoichiometry and read counts. With `--annot` (a RefFlat, GenePred or RefGene file), sites from a
  transcriptome alignment are also given in genome coordinates.
- `deeprm call prep` uses a bundled C++ preprocessing binary, falls back to the Python preprocessing, and can read a
  BAM file streamed from Dorado (`-b -`). The C++ source is in the GitHub repository.
- `deeprm train prep`, `train compile` and `train run`: train your own model from unmodified and modified synthetic RNA
  reads, on one or more GPUs, with TensorBoard logs. Training reports accuracy, AU-ROC and other metrics only when
  torchmetrics is installed.
- `deeprm qc run`, `qc alignment` and `qc block`: summaries and plots of reads, alignment error rates and training
  blocks.
- `deeprm check`: checks the PyTorch, CUDA and GPU driver set-up and suggests fixes.
