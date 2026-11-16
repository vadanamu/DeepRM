# DeepRM Pileup (C++): how DeepRM runs it

`deeprm call pileup`, and the pileup at the end of `deeprm call run`, run the C++ pileup program `deeprm_pileup` when it
can give the Python pileup's results here, and the Python pileup (`src/deeprm/inference/pileup_deeprm.py`) otherwise.
The two give the same output files. The Python side is `src/deeprm/inference/pileup.py`; it has the structure of the
preprocessing binary (`deeprm_preprocess`, `src/deeprm/inference/inference_preprocess.py`).

The program's source is in [`cpp/pileup/`](pileup/), with its build steps in
[`cpp/pileup/BUILD.md`](pileup/BUILD.md). The binary shipped in the wheel is
`src/deeprm/inference/deeprm_pileup`.

## Which program runs

1. `--pileup-bin <path>`, if given; else `DEEPRM_PILEUP_BIN`, if set; else the bundled `deeprm_pileup`.
2. The Python pileup runs instead when:
   - the installation has no bundled binary, or it cannot start here (`deeprm_pileup --version` does not exit 0: not
     Linux x86_64, glibc older than 2.28, a CPU without AVX2 and FMA, a noexec folder, ...; a warning says why);
   - `--threshold` is not 0.98: the program has log10(1 - threshold) as NumPy computes it for the default only;
   - NumPy here does not compute the per-read terms as the program does (checked on sample probabilities with
     `deeprm_pileup terms-range`, and log10(1 - 0.98)): the program computes them as NumPy does with AVX512, and NumPy
     computes them otherwise on a CPU without AVX512, or in a NumPy version that changed them;
   - the bundled program fails on an input (an npz form it does not read, for example): its work is removed and the
     Python pileup runs, so a bad input fails with the Python pileup's message.
3. A program named with `--pileup-bin` or `DEEPRM_PILEUP_BIN` must start (`deeprm call run` checks it before the
   inference) and must not fail; NumPy differing gives a warning only.

## The command line it gets

The Python side checks the settings with the Python pileup's own checks, resolves `--thread`, sorts and indexes the BAM
when the modBAM is written, reads `deeprm_run.json` (`inputs_calibrated`), and passes the result. Each option goes
under the Python pileup's own name (as `deeprm call pileup` takes it):

| Option | When | Meaning |
|---|---|---|
| `--input DIR` | always | Folder of per-read files: every `*.npz` in it, in sorted file-name order |
| `--output DIR` | always | Output folder (already made) |
| `--bam FILE` | always | The BAM of the reads (sorted and indexed when the modBAM is written) |
| `--thread N` | always | Threads; the results follow it as the Python pileup's follow its processes |
| `--threshold X` | always (0.98) | Per-read probability at or above which a read counts as modified |
| `--epsilon X` | always | The small constant of the site scores |
| `--postfix S` | when not empty | Suffix of the genomic tables: `genomic_pileup<S>.bed` and `.npz` |
| `--slice K` | when set | For 2-D predictions: the column to use |
| `--flip` | when set | Use 1 - p for each per-read probability p |
| `--label_div N` | always | The label_id packing factor |
| `--annot FILE` | when set | Annotation: also write the genomic tables |
| `--skip-modbam` | when set | Do not write the modBAM |
| `--tmp-dir DIR` | when set | Folder for the modBAM's temporary files (default: the output folder) |
| `--report-all-stoichiometry` | when set | Write the stoichiometry of the sites the rule does not report too |
| `--calibration on\|off` | always | Whether the per-read files get the v1.2 model's fixed calibration |

## What it must write

The Python pileup's outputs for the same input and `--thread`: `pileup.bed` (the same bytes), `pileup.npz` (the same
arrays: names, dtypes, values to the bit), with `--annot` the genomic tables, and without `--skip-modbam`
`modbam_<input BAM file name>` with the same header lines and records and its `.bai` (compressed bytes may differ). A
modBAM with no record (no read of the BAM has predictions) is still written, with its header and index.

Known limit: a site whose modscore lies within 1e-14 of a 0.001 step may get a modscore 0.001 apart (the program
computes 10^x with the C library, NumPy with its own code). The program counts such sites and warns; none occurred in
the checks of the release.

## Exit status and clean-up

- Exit 0 on success; non-zero, with the reason on standard error, on a failure.
- The site tables are written as `.part` files and get their names once every output is written. A failure removes
  them and the modBAM's temporary folder (`.deeprm_modbam_XXXXXX`, its own per run).
- Ctrl-C ends the program at once; the Python side then removes what it was writing (its `.part` files, the modBAM and
  its index, its temporary folder), and leaves files that were there before. As with the Python pileup, nothing is
  cleaned up after SIGTERM.

## Building and checking the shipped binary

- The steps are in [`cpp/pileup/BUILD.md`](pileup/BUILD.md), section "The binary shipped with DeepRM": a conda-forge
  GCC 14 on a glibc 2.28 sysroot, libstdc++, libgcc and zlib linked statically, the debug information stripped, the
  unit tests (`make test`). The shipped binary loads only glibc's own libraries, has no RPATH, needs no GLIBC symbol
  above 2.28 and holds no build path.
- `pytest tests/test_pileup_parity.py` (with `DEEPRM_PILEUP_BIN` for a binary that is not bundled) runs it against the
  Python pileup on small inputs; the release checks also compare full runs of both on the test data.
