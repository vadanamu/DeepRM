# `deeprm call`

Detect m6A in two steps: `deeprm call prep` prepares the POD5 and BAM files, then `deeprm call run` runs the model
and also makes the site-level results. `deeprm call pileup` remakes the site-level results from the per-read files
of a run.

## Group help

```{argparse}
:module: deeprm.inference.cli
:func: parser
:prog: deeprm call
```

## `deeprm call prep`

```{eval-rst}
.. argparse::
   :module: deeprm.inference.inference_preprocess
   :func: add_arguments
   :passparser:
   :prog: deeprm call prep

   --thread -t : @replace
      Number of threads to use (default: 95% of the CPUs)
```

## `deeprm call run`

```{argparse}
:module: deeprm.inference.inference
:func: add_arguments
:passparser:
:prog: deeprm call run
```

## `deeprm call pileup`

```{argparse}
:module: deeprm.inference.pileup_deeprm
:func: add_arguments
:passparser:
:prog: deeprm call pileup
```

## Examples

```bash
# 1) Preprocess the POD5 and BAM files (with the C++ binary, or the Python preprocessing if the binary cannot start)
deeprm call prep -p pod5/ -b reads.bam -o prep/ --thread 8

# 2) Run the bundled v1.2 model, then the pileup
deeprm call run -i prep/ -b reads.bam -o pred/

# Optional: remake only the site-level results, from the per-read files
deeprm call pileup -i pred/prep/molecule-level -o pileup/ -b reads.bam
```
