# 📦 Installation

(quick-installation)=
## ⚡ Quick installation
In a new Python environment (Python 3.10 or newer) on Linux x86_64:
```bash
pip install deeprm
deeprm check
```
* This installs DeepRM v1.2 with PyTorch 2.10.0 and FlashAttention-4. What `deeprm check` prints: see
  Verify Installation below.
* With uv 0.12 or newer: `uv pip install deeprm` ({ref}`older uv <uv-prerelease>`).
* For an older GPU, a machine without a GPU or another PyTorch build: {ref}`pytorch-builds`.

## Prerequisites
* Linux x86_64 with glibc 2.28 or newer ({ref}`supported-systems`).
* Python 3.10 or newer.
* An NVIDIA GPU is recommended. DeepRM also runs on the CPU, more slowly. Training needs a GPU.
* Optional: Dorado 2.0.0, for basecalling ({ref}`dorado-basecalling`).
* Optional: [SAMtools](http://www.htslib.org/) 1.16.1 or newer, for BAM files.

## Installation options
Install DeepRM in its own environment. It needs PyTorch 2.10.0 and replaces any other PyTorch version there.

1. With pip, in a venv or conda environment: the commands in {ref}`quick-installation`.
2. With uv, in a new environment with uv's own Python:
   ```bash
   uv venv --python 3.12 --python-preference only-managed deeprm-env && source deeprm-env/bin/activate
   uv pip install deeprm
   ```
   This needs uv 0.12 or newer ({ref}`older uv <uv-prerelease>`). To install uv, see
   [uv's installation page](https://docs.astral.sh/uv/getting-started/installation/).
3. From source:
   ```bash
   git clone https://github.com/vadanamu/deeprm
   cd deeprm
   pip install -e .    # with uv: uv pip install -e .
   ```

For training, also install the `train` extra:
```bash
pip install "deeprm[train]"    # with uv: uv pip install "deeprm[train]"
```

## Verify Installation
```bash
deeprm --version
deeprm check
```
* `deeprm check` prints the engine `deeprm call run` will use and whether the C++ preprocessing binary starts.
* For each problem it finds, it prints the command that fixes it. It exits with status 1 if it finds an error.
* If something fails: {doc}`troubleshooting`. More options: {doc}`advanced-installation`.
