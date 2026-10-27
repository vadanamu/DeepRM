# 💻 Usage
Details: {doc}`advanced-usage`. Errors and fixes: [Troubleshooting](troubleshooting.md).

## Inference usage
![deeprm_inference_pipeline.png](../images/deeprm_inference_pipeline.png)

### Prepare Data
#### Accelerated preparation (recommended, default)
Basecall and preprocess in one pipeline; the BAM is saved for `deeprm call run`.
```bash
dorado basecaller --reference <ref_fasta> --chunksize 12000 --min-qscore 0 --emit-moves --estimate-poly-a rna004_sup@v6.0.0 <pod5_dir> \
| tee >(samtools sort -@ <threads> -O BAM -o <bam_path> - && samtools index -@ <threads> <bam_path>) \
| deeprm call prep -p <pod5_dir> -b - -o <prep_dir>
```
* Keep `rna004_sup@v6.0.0` and `--chunksize 12000`: the model expects reads basecalled this way.
* Run it after `set -o pipefail`, so that a basecaller error fails the pipeline ({ref}`prep-streaming`).
* "Falling back to Python preprocessing" in the log: see {ref}`Troubleshooting <prep-troubleshooting>`.

#### Sequential preparation
For a BAM already basecalled with move tags, or basecalling on another machine.
```bash
# Basecall (skip if already done)
dorado basecaller --reference <ref_fasta> --chunksize 12000 --min-qscore 0 --emit-moves --estimate-poly-a rna004_sup@v6.0.0 <pod5_dir> > <raw_bam_path>
# Filter, sort and index
samtools view -@ <threads> -bh -F 276 <raw_bam_path> | samtools sort -@ <threads> -o <bam_path> -
samtools index -@ <threads> <bam_path>
# Preprocess
deeprm call prep -p <pod5_dir> -b <bam_path> -o <prep_dir>
```
* For a genome alignment, use `-F 260` here and `-g 260` in `deeprm call prep`.

#### Main options of `deeprm call prep`
* `-g` (`--filter-flag`, default 276): SAM flag bits of the reads to skip (260 for a genome alignment).
* `-q` (`--qcut`, default 0): skip reads with a lower mean base quality.
* `--zero-pad` or `--edge-pad`: pad sites near a read end instead of skipping them. `--trim`: trim low-quality
  read ends.
* `deeprm call prep -h` lists all options.

### Run Inference
```bash
deeprm call run -i <prep_dir> -b <bam_path> -o <prediction_dir>
```
This runs the model, then the pileup. The results go to `<prediction_dir>/<prep_dir name>/`, or to
`<prediction_dir>/<prep_dir name>-<postfix>/` with `--postfix <postfix>` ({ref}`output-files`).
* `--gpu-pool 0 1`: the GPUs to use (default: all). `--gpu 0` runs on the CPU, which is much slower.
* `-s` (`--batch`, default 16000): batch size ({ref}`batch-memory`).
* `--annot <annotation_file>`: also write genomic coordinates ({ref}`genomic-coordinates`).
* `--threshold` (default 0.98): the per-read probability at which a read counts as modified.
* `--skip-modbam`: do not write the modBAM file.
* `--resume`: continue a stopped run ({ref}`resume-run`).
* `--model <model_file>`: another v1.2 model file, such as one from `deeprm train run` (default: the bundled model).
  DeepRM v1.1 model files need DeepRM v1.1 (`pip install deeprm==1.1.3`).
* Engines: {ref}`inference-engines`.

### Run the pileup again (optional)
To remake the site-level files, for example with another `--threshold` or `--annot`:
```bash
deeprm call pileup -i <prediction_dir>/<prep_dir name>/molecule-level -b <bam_path> -o <site_dir>
```

(output-files)=
### Output files
`deeprm call run` writes these files under `<prediction_dir>/<prep_dir name>/` (`<prep_dir name>-<postfix>` with
`--postfix`). `<postfix>` is the text given to `--postfix`, empty by default.

| File | What it holds |
|---|---|
| `site-level/pileup.bed` | One row per site ([bedMethyl](https://genome.ucsc.edu/goldenpath/help/bedMethyl.html)) |
| `site-level/pileup.npz` | The same sites with all site statistics |
| `site-level/genomic_pileup<postfix>.bed`, `.npz` | The same in genomic coordinates (with `--annot`) |
| `site-level/modbam_<input BAM name>` | Per-read calls in [MM and ML tags](https://samtools.github.io/hts-specs/SAMtags.pdf) |
| `molecule-level/inference_*.npz` | Per-read predictions |
| `molecule-level/deeprm_run.json` | The model and settings of the run |

Columns, arrays and stoichiometry: {ref}`output-formats`.

## Training usage
* `deeprm train` trains a v1.2 model (deeprm_v12) on synthetic-oligonucleotide data. Run it with
  `deeprm call run --model <model_file>`.
* It does not reproduce the released v1.2 model, whose training data and methods are reserved for a future
  publication.
* The example training data are older and were basecalled with `rna004_130bps_sup@v5.0.0`; basecall new data with
  `rna004_sup@v6.0.0`, as below.

![deeprm_train_pipeline.png](../images/deeprm_train_pipeline.png)
### Prepare Data
```bash
# Basecall (skip if already done)
dorado basecaller --chunksize 12000 --min-qscore 0 --emit-moves --estimate-poly-a rna004_sup@v6.0.0 <pod5_dir> > <bam_path>
samtools index -@ <threads> <bam_path>
# Preprocess the unmodified and the modified data
deeprm train prep -p <unmodified_pod5_dir> -b <unmodified_bam_path> -o <prep_dir>/a
deeprm train prep -p <m6a_pod5_dir> -b <m6a_bam_path> -o <prep_dir>/m6a
```
* `--method` (`lev` or `pw`) must match the oligonucleotide design: {ref}`training-details`.

### Compile Training Data
```bash
deeprm train compile -n <prep_dir>/a/data -p <prep_dir>/m6a/data -o <compiled_dir>
```
* `-c`: rows per output file (default 4000).

### Run Training
```bash
deeprm train run --data <compiled_dir> --output <output_dir> --gpu-pool <gpu_pool>
```
* `--batch` (default 1024): lower it if the GPU runs out of memory.
* To start from the released weights, add `--load-checkpoint <weights_file>`. Their path:
  ```bash
  python -c "import deeprm, pathlib; print(pathlib.Path(deeprm.__file__).parent / 'weight' / 'deeprm_weights.pt')"
  ```
* The trained model files (.pt) go to a subfolder of `<output_dir>`.
