# `deeprm qc`

Quality-control plots for basecalled reads (`deeprm qc run`), alignments (`deeprm qc alignment`) and training data
(`deeprm qc block`).

## Group help

```{argparse}
:module: deeprm.qc.cli
:func: parser
:prog: deeprm qc
```

## `deeprm qc run`

```{eval-rst}
.. argparse::
   :module: deeprm.qc.inspect_run
   :func: add_arguments
   :passparser:
   :prog: deeprm qc run

   --process -p : @replace
      Number of worker processes (default: about a quarter of the CPUs)
```

## `deeprm qc alignment`

```{eval-rst}
.. argparse::
   :module: deeprm.qc.inspect_alignment
   :func: add_arguments
   :passparser:
   :prog: deeprm qc alignment

   --process -p : @replace
      Number of worker processes (default: about a quarter of the CPUs)
```

## `deeprm qc block`

```{argparse}
:module: deeprm.qc.inspect_block
:func: add_arguments
:passparser:
:prog: deeprm qc block
```

## Examples

```bash
# Inspect a basecalled run
deeprm qc run -i <bam_file> -o <output_dir>

# Inspect training data: the data folders that deeprm train prep writes
deeprm qc block -i <prep_dir>/a/data <prep_dir>/m6a/data -t A m6A -o <qc_dir>

# Inspect an alignment
deeprm qc alignment -i <bam_file> -o <output_dir>
```
