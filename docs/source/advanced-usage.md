# 🧩 Advanced Usage
More detail on the commands of {doc}`usage`. Errors and their fixes are on [Troubleshooting](troubleshooting.md).

(prep-streaming)=
## Streaming from Dorado
* With `-b -`, `deeprm call prep` reads the BAM from standard input. A path that is not a regular file
  (`<(command)`, a named pipe) is read the same way.
* A stream that ends without the BAM end-of-file block, as a killed basecaller leaves it, fails the run.
* A basecaller that exits with an error but closes its output normally is not seen through a pipe. Run the pipeline
  after `set -o pipefail`, or give the basecaller command to `--bam-cmd`, which fails the run if the command fails.

(inference-engines)=
## Inference engines
DeepRM picks the engine by itself; `deeprm check` prints which one. All engines run the same model; their per-read
probabilities differ slightly from each other. With `fa4` or `eager`, repeated runs on the same machine give
bit-identical results ({ref}`run-to-run`).

| Engine | Used on |
|---|---|
| `fa4` (fastest) | NVIDIA GPUs of compute capability 8.0, 9.0 and 10.0 (for example A100, H100, H200 and B200), with a PyTorch build for CUDA 12.8 or newer |
| `fast` | Other GPUs of compute capability 7.0 or newer, and the CPU |
| `eager` (slowest, most memory) | GPUs where `torch.compile` cannot run: compute capability below 7.0, or no C compiler or Python C headers |

* On a GPU, the first batch compiles the model (`fa4`, `fast`). This takes up to about half a minute.
* If `fa4` could run but is not used, `deeprm check` and `deeprm call run` print a framed warning with the steps that
  fix it.
* `--engine fa4|fast|eager` forces an engine and turns that warning off. Use it only when needed, for example
  `--engine eager` when the model cannot compile on an older GPU. A forced `fa4` also runs, untested, on other
  NVIDIA GPUs of compute capability 8.0 or newer.
* A model file that does not name its architecture needs `--model-type deeprm_v12`.
* `--engine` and `--model-type` are not shown by `deeprm call run -h`.

(batch-memory)=
## Batch size and GPU memory
* `-s` (`--batch`, default 16000) sets the rows per batch. If a batch does not fit in GPU memory, DeepRM halves it
  until it fits, keeps that size for the rest of the run and logs it once.
* Set `-s` lower yourself to skip these retries. With the default batch, the model needs about 13 to 15 GiB of GPU
  memory with `fa4` or `fast`, and about 70 GiB with `eager`. Memory scales with `-s`.
* On a 16 GB GPU, start with `-s 8000` (`fast`) or `-s 2000` (`eager`).

(genomic-coordinates)=
## Genomic coordinates (`--annot`)
* For reads aligned to transcripts, `--annot <annotation_file>` also writes `site-level/genomic_pileup<postfix>.bed`
  and `.npz` in genomic coordinates.
* The file can be RefFlat (11 columns), GenePred (10) or RefGene (15). Remove the first column (`bin`) of a UCSC
  `refGene.txt` first: `cut -f 2- refGene.txt > refGene.15col.txt`.
* Transcripts are matched by name. A BAM name with a version (`ENST00000123456.7`) also matches an annotation name
  without it (`ENST00000123456`), not the other way round.
* The log counts the transcripts not found. Their sites are left out of the genomic tables.
* Sites from reads aligned in reverse (kept with `-g 260`) get their own rows on the opposite genomic strand.

(output-formats)=
## Output formats
### Site-level BED file
The BED file is [bedMethyl](https://genome.ucsc.edu/goldenpath/help/bedMethyl.html), one row per site:
* Columns 1 to 3: reference name, start (0-based) and end.
* Column 5: modscore × 1000 (0 to 1000).
* Column 6: strand.
* Column 10: number of reads at the site.
* Column 11: stoichiometry (%). See {ref}`site-stoichiometry`.
* Columns 12 and 13: estimated numbers of modified and unmodified reads.
* Columns 14 to 18: 0 (kept for a future update).

(site-stoichiometry)=
### Stoichiometry
* The stoichiometry of a site is the fraction of the As at that site that are m6A.
* Some sites are **not reported**: their reads are not enough to decide the stoichiometry. Their stoichiometry is
  written as 0 (BED column 11 and `stoichiometry`) and as `NaN` in `stoichiometry_release`.
* `--threshold` (default 0.98) changes the modscore and `count_pos`. It does not change which sites are reported
  or their stoichiometry.
* A reported stoichiometry below about 1 % is close to background.

### Site-level NPZ file
`pileup.npz` and `genomic_pileup<postfix>.npz` load with `numpy.load`. They hold one value per site in these arrays:
* `ref_names`, `ref_pos` (0-based), `ref_strand`
* `modscore`, `stoichiometry` (both from 0 to 1)
* `count_all` (reads), `count_pos` (reads at or above `--threshold`)
* `stoich_state`: `1` reported, `-1` not reported
* `stoichiometry_release`: the stoichiometry, `NaN` where not reported
* `stoichiometry_eq14`: the stoichiometry as DeepRM v1.1 computes it
* `logsum_1_p_pos`, `kl_div_pos`, `kl_div_neg`: the per-site sums the scores come from

### Molecule-level NPZ file
Each `molecule-level/inference_*.npz` file holds:
* `read_id`: the read UUID as two 64-bit integers per row.
* `label_id`: the reference, position and strand of the site.
* `pred`: the per-read modification probability (0 to 1). `NaN` marks a row without usable signal; the pileup
  skips it.

Decode them with:
```python
import uuid
import numpy as np

def read_uuid(read_id_row):  # one row of read_id
    return uuid.UUID(bytes=read_id_row.tobytes())

def decode_label_id(label_id, label_div=10**9):  # label_div: --label-div of deeprm call prep
    strand = np.sign(label_id)
    label_id_abs = np.abs(label_id) - 1
    return label_id_abs // label_div, label_id_abs % label_div, strand  # reference index, position, strand
```
The reference index counts the references in the BAM header, from 0.

(resume-run)=
## Resuming a run (`--resume`)
* `--resume` continues a stopped `deeprm call run` in the same output folder. Unfinished input files run again.
* It checks that the model file, input files, engine, device type and number of GPUs match the first run (as
  recorded in `molecule-level/deeprm_run.json`).

(training-details)=
## Training data preparation
* `deeprm train prep` finds the context blocks of each read and writes their features. `-h` lists all options.
* `--method lev` (default): designs with 6-nt spacers, as in the example data. `--method pw`: designs with 15-nt
  spacers; add `--di` for two As per block. Each method has its own defaults.
* Skipped reads: shorter than `--min` (200) or longer than `--max` (2,000) bases, or mean base quality below
  `--qcut` (7). Low-quality read ends are trimmed unless you add `--no-trim`.
* Output: `data/*.npz` (the features, the input of `deeprm train compile`), `blocks/*.pkl` (blocks per read), and
  the logs and settings (`preprocess.log`, `command.log`, `args.json`).
* `deeprm train compile` keeps blocks with a score of at least `--score` (default 1.0) and splits them into `train/`
  and `val/` (`--val-frac`, default 0.05). Its `--output` folder must not exist yet.
