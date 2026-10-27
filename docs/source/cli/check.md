# `deeprm check`

Checks the DeepRM installation: PyTorch, torchmetrics, the GPU and the C++ preprocessing binary. It prints the
problems it finds, then the engine `deeprm call run` would use on this machine and whether the preprocessing binary
starts. It exits with status 1 if it found an error.

## Usage

```{argparse}
:module: deeprm.utils.check_installation
:func: parser
:prog: deeprm check
```

## Examples

```bash
# Show the problems found and the engine deeprm call run would use
deeprm check

# Also print all environment details
deeprm check --verbose

# Before training: a missing torchmetrics is an error, not a warning
deeprm check --require-train
```
