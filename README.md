# DeepRM
#### Deep learning for RNA Modification
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-red?logo=github)](https://github.com/vadanamu/DeepRM)
[![CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
![GitHub Repo stars](https://img.shields.io/github/stars/vadanamu/DeepRM?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/vadanamu/DeepRM)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/vadanamu/DeepRM)
![GitHub contributors](https://img.shields.io/github/contributors/vadanamu/DeepRM)
![GitHub language count](https://img.shields.io/github/languages/count/vadanamu/DeepRM)
![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17636697.svg)

![deeprm.png](https://raw.githubusercontent.com/vadanamu/DeepRM/public/docs/images/deeprm.png)

## ✨ Introduction
DeepRM is a deep learning-based framework for RNA modification detection using Nanopore direct RNA sequencing.
This repository contains the source code for training and running DeepRM.

## 🆕 What's new in v1.2
DeepRM v1.2 uses substantially less memory, runs faster and detects RNA modifications more accurately.

**1. Substantially lower memory usage and higher speed**
* The preprocessing uses 8 to 13 times less peak memory than DeepRM v1.1's.
* With input on a local SSD, the preprocessing runs 2 to 2.5 times as fast as DeepRM v1.1's. Run on Dorado's output
  while Dorado basecalls, it finishes when basecalling does, so it adds no time to the run.
* Inference is 1.9 to 3.0 times as fast as DeepRM v1.1 on A100, H100 and B200 GPUs.

**2. Improved RNA modification detection accuracy**
* Higher m6A recall, most of all for clustered m6A (the nearest other m6A 1–4 nt away): 95.7 %, against 85.8 % with
  DeepRM v1.1 and 78.9 % with Dorado (v6.0.0 SUP), at the same false positive rate.
* Oxford Nanopore Technologies' all-5-mer synthetic RNA dataset: AU-PR 0.933, against 0.883 with DeepRM v1.1 and 0.923
  with Dorado.
* HEK293T benchmark of the DeepRM paper: AU-PR 0.964 at DRACH and 0.716 at non-DRACH sites, against 0.961 and 0.704 with
  DeepRM v1.1, and 0.959 and 0.161 with Dorado.

Benchmark details and the full list of changes: [📜 Changelog](https://github.com/vadanamu/DeepRM/blob/public/CHANGELOG.md).

## ⚡ Quick installation
In a new Python environment (Python 3.10 or newer) on Linux x86_64:
```bash
pip install deeprm
deeprm check
```
* `deeprm check` reports any problem it finds, with the command that fixes it.
* With uv: `uv pip install deeprm "flash-attn-4[cu13]==4.0.0b35"`. DeepRM v1.2 needs FlashAttention-4 version
  4.0.0b35, a pre-release (test) version, and uv installs it only when the command names this version; without it,
  uv installs DeepRM v1.1.3.
* Other setups: see [Installation](#-installation).

## Table of Contents
* [✨ Introduction](#-introduction)
* [🆕 What's new in v1.2](#-whats-new-in-v12)
* [⚡ Quick installation](#-quick-installation)
* [🎯 Key Features](#-key-features)
* [📦 Installation](#-installation)
* [🚀 Quickstart](#-quickstart)
* [💻 Usage](#-usage)
  * [Inference](#inference-usage)
  * [Training](#training-usage)
* [🔧 Troubleshooting](#-troubleshooting)
* [📝 Citation](#-citation)
* [📝 License](#-license)
* [🏛 Contributors](#-contributors)
* [🏛 Acknowledgements](#-acknowledgements)

📚 Full documentation: https://deeprm.readthedocs.io/

## 🎯 Key Features
* **High accuracy**: Achieves state-of-the-art accuracy in RNA modification detection and stoichiometry measurement.
* **Single-molecule resolution**: Provides single-molecule level predictions for RNA modifications.
* **End-to-end pipeline**: Easy-to-use pipeline from raw reads to molecule-level and site-level predictions.
* **Fast**: Compiled inference and a fast C++ preprocessing.
* **Customizable**: Supports training of custom models.

## 📦 Installation
### Prerequisites
* Linux x86_64 with glibc 2.28 or newer (Ubuntu 20.04, Debian 10, RHEL, Rocky or Alma 8, or newer)
* Python 3.10+
* An NVIDIA GPU: recommended for inference (which also runs on the CPU, more slowly), required for training.
* Optional: [Dorado](https://github.com/nanoporetech/dorado) 2.0.0 (basecalling) and
  [SAMtools](http://www.htslib.org/) 1.16.1+ (BAM files).

### Installation options
* Estimated time: ~10 minutes. Use a new environment: DeepRM installs PyTorch 2.10.0 and replaces any other version.

1. With pip: see [Quick installation](#-quick-installation).

2. With uv, using uv's own Python (it has the C headers needed to compile the model):
```bash
uv venv --python 3.12 --python-preference only-managed deeprm-env && source deeprm-env/bin/activate
uv pip install deeprm "flash-attn-4[cu13]==4.0.0b35"
```
* Keep the FlashAttention-4 version in the command (see [Quick installation](#-quick-installation)). If deeprm-env
  already holds an older DeepRM, add `-U` (upgrade).

3. From source:
```bash
git clone https://github.com/vadanamu/deeprm
cd deeprm
pip install -e .
```

* For training: `pip install "deeprm[train]"`.
* Older GPUs, CPU-only and other setups:
  [Advanced installation](https://deeprm.readthedocs.io/en/latest/advanced-installation.html).

### Verify Installation
```bash
deeprm --version
deeprm check
```
* With no problem found, `deeprm check` prints "Environment checks passed successfully."

## 🚀 Quickstart
* Example POD5 and BAM files are in the repository's `example` folder (pip does not install them). Run the commands
  below from the repository folder.

### RNA Modification Detection
* Estimated time: about 3 minutes on one B200 GPU.

1️⃣ **Prepare data**
```bash
deeprm call prep -p example/call-example/call_example_pod5 -b example/call-example/call_example.bam -o <prep_dir>
```
* (Alternative) Basecall your own POD5 files and prepare them in one pass:
  ```bash
  dorado basecaller --reference <ref_fasta> --chunksize 12000 --min-qscore 0 --emit-moves --estimate-poly-a rna004_sup@v6.0.0 <pod5_dir> \
  | tee >(samtools sort -@ <threads> -O BAM -o <bam_path> - && samtools index -@ <threads> <bam_path>) \
  | deeprm call prep -p <pod5_dir> -b - -o <prep_dir>
  ```
    * Keep `rna004_sup@v6.0.0` and `--chunksize 12000`: the model expects reads basecalled this way.

2️⃣ **Run inference**
```bash
deeprm call run -b example/call-example/call_example.bam -i <prep_dir> -o <pred_dir>
```
* Results: `<pred_dir>/<prep_dir name>/` (see [Output files](#output-files)).

### Model Training
* Estimated time: ~1 hour

1️⃣ **Prepare unmodified & modified training data**
```bash
deeprm train prep -p example/train-example/A/train_example_A_pod5 -b example/train-example/A/train_example_A.bam -o <prep_dir>/a
deeprm train prep -p example/train-example/m6A/train_example_m6A_pod5 -b example/train-example/m6A/train_example_m6A.bam -o <prep_dir>/m6a
```

2️⃣ **Compile training data**
```bash
deeprm train compile -n <prep_dir>/a/data -p <prep_dir>/m6a/data -o <prep_dir>/compiled -c 1000
```
* `-c 1000` suits this small example (default: 4000).

3️⃣ **Run training**
```bash
deeprm train run --data <prep_dir>/compiled --output <output_dir> --batch 64
```
* The output is a model file (.pt).
* This trains the v1.2 architecture (deeprm_v12) from scratch. The training data and methods of the released weights
  are reserved for a future publication.


## 💻 Usage
### Inference usage
![deeprm_inference_pipeline.png](https://raw.githubusercontent.com/vadanamu/DeepRM/public/docs/images/deeprm_inference_pipeline.png)

* Prepare data with `deeprm call prep` as in the [Quickstart](#-quickstart). For reads aligned to a genome, add `-g 260`.
  Other ways: [Usage](https://deeprm.readthedocs.io/en/latest/usage.html).
* Run inference with the bundled v1.2 model:
```bash
deeprm call run --input <prep_dir> --bam <bam_path> --output <prediction_dir> --gpu-pool <gpu_pool>
```
* `--annot <annotation_file>` (RefFlat, GenePred or RefGene) also writes genomic coordinates, for reads aligned to
  transcripts.
* `--model <model_file>` runs a model made with `deeprm train run`. DeepRM v1.1 model files need DeepRM v1.1
  (`pip install deeprm==1.1.3`).

#### Output files
Under `<prediction_dir>/<prep_dir name>/`:
* `site-level/pileup.bed`: one row per site ([bedMethyl](https://genome.ucsc.edu/goldenpath/help/bedMethyl.html)).
  Column 5 is the modscore × 1000; column 11 is the stoichiometry (%), 0 where it is not reported.
* `site-level/modbam_<input BAM name>`: per-read calls in MM/ML tags
  ([SAMtags](https://samtools.github.io/hts-specs/SAMtags.pdf)).
* `molecule-level/inference_*.npz`: per-read predictions.
* All output files: [Usage](https://deeprm.readthedocs.io/en/latest/usage.html#output-files).

### Training usage
![deeprm_train_pipeline.png](https://raw.githubusercontent.com/vadanamu/DeepRM/public/docs/images/deeprm_train_pipeline.png)
* Three steps (`prep`, `compile`, `run`), as in the [Quickstart](#model-training). Options:
  [Usage](https://deeprm.readthedocs.io/en/latest/usage.html).


## 🔧 Troubleshooting
* GPU not used, or "no kernel image is available": run `deeprm check` and run the command it prints.
* "Falling back to Python preprocessing" in the log:
  [Troubleshooting](https://deeprm.readthedocs.io/en/latest/troubleshooting.html#prep-troubleshooting).
* Out of GPU memory: lower `-s` for `deeprm call run` (default: 16000) or `--batch` for `deeprm train run` (default: 1024).
* Other problems: [Troubleshooting](https://deeprm.readthedocs.io/en/latest/troubleshooting.html).

## 📝 Citation
If you use DeepRM in your research, please cite the following paper:
```text
@article{kang2026deeprm,
  title={Comprehensive discovery of m6A sites in the human transcriptome at single-molecule resolution},
  author={Kang, Gihyeon and Hwang, Hyeonseo and Jeon, Hyeonseong and Choi, Heejin and Chang, Hee Ryung and Yeo, Nagyeong and Park, Junehee and Son, Narae and Jeon, Eunkyeong and Lim, Jungmin and Yun, Jaeung and Choi, Wook and Jo, Jae-Yoon and Kim, Jong-Seo and Park, Sangho and Kim, Yoon Ki and Baek, Daehyun},
  journal={Nature Communications},
  volume={17},
  pages={664},
  year={2026},
  publisher={Springer Nature},
  doi={10.1038/s41467-025-67417-w}
}
```
The article is fully open access and available at <https://doi.org/10.1038/s41467-025-67417-w>.

## 📝 License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>
<br />DeepRM is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>
by Seoul National University R&DB Foundation and Genome4me Inc.

See the [LICENSE](https://github.com/vadanamu/DeepRM/blob/public/LICENSE.md) file for details.

## 🏛 Contributors
This repository is developed and maintained by the following organization:
* **Laboratory of Computational Biology, School of Biological Sciences, Seoul National University**
    * Principal Investigator: Prof. Daehyun Baek
* **Genome4me, Inc., Seoul, Republic of Korea**


## 🏛 Acknowledgements
This study was supported by the National Research Foundation of Korea (NRF) funded by the Ministry of Science and ICT, Republic of Korea (MSIT) (RS-2019-NR037866, RS-2020-NR049252, RS-2020-NR049538, and RS-2022-NR067483), by a grant of Korean ARPA-H Project through the Korea Health Industry Development Institute (KHIDI), funded by the Ministry of Health & Welfare, Republic of Korea (RS-2025-25422732), by Artificial Intelligence Industrial Convergence Cluster Development Project funded by MSIT and Gwangju Metropolitan City, by National IT Industry Promotion Agency (NIPA) funded by MSIT, and by Korea Research Environment Open Network (KREONET) managed and operated by Korea Institute of Science and Technology Information (KISTI).
