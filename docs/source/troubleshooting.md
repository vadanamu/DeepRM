# 🔧 Troubleshooting
Run `deeprm check` first. It names most problems and prints the command that fixes them.

## Installation and GPU
* **DeepRM runs on the CPU, or stops with "no kernel image is available".** Run `deeprm check` and run the command
  it prints: it installs a PyTorch build for your GPU and driver. If `nvidia-smi` fails after a driver update,
  restart the machine.
* **`fa4` is not used.** If FlashAttention-4 is missing, `deeprm check` prints the command that installs it. On a GPU
  that `fa4` was not tested on, DeepRM runs `fast` on purpose; add `--engine fa4` to try `fa4` there.
* **DeepRM runs the slower `eager` engine because no C compiler or no `Python.h` was found.** Install `gcc` and your
  Python's headers (for example `python3.12-dev` on Ubuntu), or make the environment with uv or conda. On GPUs of
  compute capability 8.0 or newer (for example A100, H100 and B200), the warning prints the exact command.

## Basecalling
* **Which Dorado options to use.** See {ref}`dorado-basecalling`. `--chunksize 12000` also avoids "illegal memory
  access" errors on some GPUs.

(prep-troubleshooting)=
## Preprocessing (`deeprm call prep`)
* **The log says "Falling back to Python preprocessing".** The C++ binary cannot start here. The run still works,
  2 to 5 times more slowly. Fixes: {ref}`binary-not-starting`.
* **Prep refuses an output folder that already holds `.npz` files.** Use a new or empty output folder.
* **"The preprocessing wrote no rows".** Check that `-p` names the POD5 files the BAM was basecalled from.
* **"BAM records have no MD tag".** Align the reads when you basecall (`--reference`).
* **"Corrupted POD5 file" or "failed to read POD5 signal".** The POD5 file is damaged (for example, cut off). Copy
  it again, or move it out of the POD5 folder.
* **"The SAM input does not end with a newline", or the binary stops on a truncated BAM.** The file was cut off.
  Make it again. To stream from a basecaller, use `--bam-cmd`: it stops the run when the basecaller fails.
* **"-b - reads the BAM from stdin, but stdin is a terminal".** Pipe the BAM in, or use `--bam-cmd`.
* **After prep stops with an error,** run it again in a new or empty output folder.
* **Prep runs out of memory.** Use fewer threads (`-t`) or a smaller output chunk (`-k`). When the BAM is streamed,
  also lower `-Q`. The Python preprocessing also takes a smaller `-n`, or `--auto-size`.
* **The disk fills up during the Python preprocessing.** Set `DEEPRM_PREP_CACHE_DIR` to put its BAM cache on a
  larger disk.

(binary-not-starting)=
### The binary does not start
The log line and the `PREP_BINARY_CANNOT_START` warning of `deeprm check` give the reason.
* **On a file system where programs cannot run (noexec):** copy the binary to a folder where programs can run and
  set `DEEPRM_PREPROCESS_BIN` to the copy.
* **No execute permission:** run `chmod +x` on the file.
* **glibc is older than the binary needs:** use a system or container with a newer glibc, or build the binary on this
  system ({ref}`prep-binary-build`).
* **The system has no glibc (Alpine and other musl systems):** use a glibc-based system or container image.
* **A system library cannot be loaded:** install the library the message names, or rebuild the binary with the static
  flags ({ref}`prep-binary-build`).
* **The file is missing or incomplete:** reinstall DeepRM.
* **Another CPU type or operating system, or "Illegal instruction":** nothing to fix; the Python preprocessing runs.

## Inference (`deeprm call run`)
* **The GPU runs out of memory.** DeepRM splits the batch and says so. To avoid this, lower `--batch` (`-s`, default
  16000).
* **The `fast` engine cannot compile the model (for example on T4, RTX 20 or V100).** Run with `--engine eager`.
* **"The compile cache folder ... does not allow running files".** Set `TORCHINDUCTOR_CACHE_DIR` to another folder,
  or run with `--engine eager`.
* **The run fails inside FlashAttention-4.** Run with `--engine fast`.
* **"Skipping input shard ...".** The shard is unreadable or malformed, and its reads are missing from the results.
  Re-create it under the same name, then run again with `--resume`. Do not delete it.
* **"The input shards do not fit this model".** Run `deeprm call prep` again with the default `--cb-len`,
  `--kmer-len` and `--max-token-len`.
* **A warning says rows hold NaN or infinite values.** The shard is damaged or was not made by `deeprm call prep`. Those rows get
  NaN and the pileup skips them. Make the input again with `deeprm call prep`.
* **A GPU error is hard to trace to a batch.** Run again with `DEEPRM_SYNC_LOOP=1` to run one batch at a time.

(run-to-run)=
### Differences between runs, engines and GPU types
* Repeated runs on the same machine with the `fa4` engine (the default on A100, H100 and B200) or the `eager` engine
  give bit-identical per-read output, also with more GPUs and after `--resume`.
* The `fast` engine can differ in the last digits between runs, because its compiled code depends on what the compile
  cache holds. A run whose batches were split to fit in GPU memory can differ in the same way.
* Different GPU types differ in the last digits. Different engines differ by up to about 0.02 in per-read
  probability, so a few reads can move across the 0.98 cut. This is not an error.

## Outputs
* **`deeprm call run` refuses an output folder that already holds per-read files.** Add `--resume` to continue that
  run, or use a new or empty output folder.
* **"... is older than ...", about the BAM index.** DeepRM could not rebuild the index, so the modBAM may miss reads.
  Run `samtools index` on the BAM where you can write, or add `--skip-modbam` if you do not need the modBAM.

## Training (`deeprm train`)
* **No training metrics, or "torchmetrics is not importable".** Install the training extra:
  `pip install "deeprm[train]"` (with uv: `uv pip install "deeprm[train]"`).
* **Training runs out of memory.** Lower `--batch` (default 1024).
* **`deeprm train compile` says "No files were written for ...".** The input is small for the chunk size. Remove the
  output folder and run again with a smaller `-c` (for example `-c 1000`).
