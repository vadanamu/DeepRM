# 📦 Advanced Installation
Details that most users do not need. The basic install is on the {doc}`installation` page.

(supported-systems)=
## Supported systems
* Linux x86_64 with glibc 2.28 or newer (Ubuntu 20.04, Debian 10, RHEL, Rocky or Alma 8, or newer).
  `ldd --version` shows your glibc.
* Other systems are not supported or tested.

(pytorch-builds)=
## PyTorch builds
`pip install deeprm` installs PyTorch 2.10.0 built for CUDA 12.8. It runs on NVIDIA GPUs from V100 to B200 and RTX 50
(driver 525 or newer; 570 or newer for B200 and RTX 50), and on the CPU. In these cases, install another build:

* CUDA 13.0 build: optional, for newer GPUs (for example H100 or B200) with driver 580 or newer. It does not run on
  V100 or older GPUs.
  ```bash
  pip install deeprm --extra-index-url https://download.pytorch.org/whl/cu130
  uv pip install deeprm --torch-backend=cu130
  ```
* CUDA 12.6 build: for older GPUs (GTX 900, GTX 10, P100, P40).
  ```bash
  pip install deeprm --extra-index-url https://download.pytorch.org/whl/cu126
  uv pip install deeprm --torch-backend=cu126
  ```
* CPU-only build: for a machine without a GPU (a much smaller download).
  ```bash
  pip install deeprm --extra-index-url https://download.pytorch.org/whl/cpu
  uv pip install deeprm --torch-backend=cpu
  ```

`nvidia-smi` shows your GPU and driver version. If the installed build does not fit your GPU, `deeprm check` says so
and prints the command that installs the right build.

## uv
* uv's own Python (`--python-preference only-managed`) has the C headers that the `fa4` and `fast` engines need. Some
  system Pythons do not.
* `--torch-backend` works only with `uv pip install`. In a uv project (`uv add deeprm torch`), choose the PyTorch
  build in `pyproject.toml` as in [uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

(fa4-engine)=
## FlashAttention-4
* DeepRM installs FlashAttention-4 on Linux x86_64. It gives the fastest engine, `fa4`.
* Where DeepRM uses `fa4`, and the warning when it does not: {ref}`inference-engines`.

(dorado-basecalling)=
## Dorado
* DeepRM v1.2 expects reads basecalled with Dorado 2.0.0 and the `rna004_sup@v6.0.0` model, with `--emit-moves`
  and `--chunksize 12000`. Dorado 1.4.0 and older do not have this model.
  * https://github.com/nanoporetech/dorado
* For `deeprm call`, align the reads with `--reference`: the preprocessing needs the `MD` tag. The commands are in
  {doc}`usage`.

(prep-binary)=
## The C++ preprocessing binary
`deeprm call prep` runs `deeprm_preprocess`, a C++ program installed with DeepRM. It gives the same rows as the
Python preprocessing, faster.
* It needs Linux x86_64 with glibc 2.27 or newer. It needs no other system library.
* If it cannot start, `deeprm call prep` uses the Python preprocessing instead:
  {ref}`Troubleshooting <prep-troubleshooting>`.
* `deeprm call prep` also uses the Python preprocessing when `--pod5` is a single file, or when you set an option
  that only the Python preprocessing has. The log says which.
* When the binary works, `deeprm check` prints `C++ preprocessing binary: starts (...)`.

(prep-binary-other-build)=
### Use another build
```bash
export DEEPRM_PREPROCESS_BIN=/path/to/deeprm_preprocess
# or: deeprm call prep --preprocess-bin /path/to/deeprm_preprocess ...
```
* If this binary cannot start, `deeprm call prep` stops with an error. It does not use the Python preprocessing.
* Unset the variable, or drop the option, to go back to the bundled binary.

(prep-binary-build)=
### Build the binary
The build steps are in [`cpp/README.md`](https://github.com/vadanamu/DeepRM/blob/public/cpp/README.md) in the source
code.
