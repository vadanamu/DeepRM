# `deeprm train`

Prepare training data and train a model of the v1.2 architecture (deeprm_v12). Training needs a CUDA GPU and the
training extra: `python -m pip install "deeprm[train]"`.

## Group help

```{argparse}
:module: deeprm.train.cli
:func: parser
:prog: deeprm train
```

## `deeprm train prep`

```{eval-rst}
.. argparse::
   :module: deeprm.train.train_preprocess
   :func: add_arguments
   :passparser:
   :prog: deeprm train prep

   --threads -t --cpu -c : @replace
      Number of worker processes (default: 95% of the CPUs)
```

## `deeprm train compile`

```{eval-rst}
.. argparse::
   :module: deeprm.train.train_compile
   :func: add_arguments
   :passparser:
   :prog: deeprm train compile

   --thread -t : @replace
      Number of worker processes (default: 90% of the CPUs)
```

## `deeprm train run`

```{argparse}
:module: deeprm.train.train
:func: add_arguments
:passparser:
:prog: deeprm train run
```

## Examples

```bash
# Prepare the unmodified and the modified training data
deeprm train prep -p a_pod5/ -b a.bam -o train_prep/a
deeprm train prep -p m6a_pod5/ -b m6a.bam -o train_prep/m6a

# Compile the training dataset
deeprm train compile -n train_prep/a/data -p train_prep/m6a/data -o ds/

# Train a model
deeprm train run --data ds/ --output runs/exp1
```
