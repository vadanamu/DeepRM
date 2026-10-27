"""
Run the model on preprocessed files, save per-read predictions, then make the site-level results.

Shapes use the letters of deeprm.model.deeprm_v12. Here B is the rows of one model call: up to ``--batch`` on a GPU,
and at most 1,000 on the CPU.
"""

# CPU_ROWS sets the CPU limit. U_pad is the engines' centre-token buffer (centre_slot_block(--batch) on a GPU).

import argparse
import collections
import glob
import hashlib
import os
import pathlib
import pickle
import platform
import queue
import threading
from contextlib import nullcontext

import numpy as np
import torch
import torch.multiprocessing as mp
import tqdm
from torch.amp import autocast

from deeprm.inference.inference_dataloader import NONFINITE_KEY, load_dataset
from deeprm.model import deeprm_v12_engines as engines
from deeprm.utils import check_deps
from deeprm.utils.logging import get_logger
from deeprm.utils.utils import BUNDLED_WEIGHTS_MD5, RUN_INFO_NAME, read_run_info, save_run_info

log = get_logger(__name__)

# Tensor-name prefixes that only DeepRM v1.1 weights have. They identify a v1.1 file that names no architecture.
V11_KEY_PREFIXES = ("cnn_encoder.", "upscale_layer.")


def add_arguments(parser: argparse.ArgumentParser):
    """Adds the command-line arguments to `parser`."""
    parser.add_argument(
        "--input", "-i", dest="data", type=str, required=True, help="Folder written by deeprm call prep"
    )
    parser.add_argument("--bam", "-b", type=str, required=True, help="BAM file of the reads given to deeprm call prep")
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output folder; the results go to <output>/<input folder name>"
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None, help="Model file (.pt; default: the bundled v1.2 model)"
    )
    # Hidden from --help. deeprm_v12 (the only accepted value) runs a model file that names another module, or none, as
    # deeprm_v12.
    parser.add_argument("--model-type", "-y", type=str, default=None, help=argparse.SUPPRESS)
    # Hidden from --help. Overrides the automatic engine choice.
    parser.add_argument("--engine", type=str, choices=engines.ENGINES, default="auto", help=argparse.SUPPRESS)
    parser.add_argument(
        "--batch", "-s", type=int, default=16000, help="Batch size (default 16000; lower it on GPUs with less memory)"
    )
    parser.add_argument(
        "--gpu",
        "-g",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all; 0 runs on the CPU). --gpu-pool overrides it.",
        dest="num_gpu",
    )
    parser.add_argument(
        "--prefetch", "-p", type=int, default=4, help="Input files each data-loading worker reads ahead"
    )
    parser.add_argument("--worker", "-w", type=int, default=4, help="Data-loading worker processes per GPU")
    parser.add_argument(
        "--postfix",
        "-x",
        type=str,
        default="",
        help="Suffix for the output folder (<output>/<input folder name>-<postfix>) and for the genomic pileup file "
        "names",
    )
    parser.add_argument(
        "--flush",
        "-f",
        type=int,
        default=100,
        help="Batches of predictions collected before they are joined into one array (on the CPU, and on a GPU with "
        "DEEPRM_SYNC_LOOP=1)",
    )
    parser.add_argument("--resume", action="store_true", help="Continue a stopped run in the same output folder")
    parser.add_argument(
        "--gpu-pool", "-gp", type=int, nargs="+", help="IDs of the GPUs to use, for example 0 1 (overrides --gpu)"
    )
    parser.add_argument(
        "--output-id",
        "-id",
        type=int,
        default=None,
        help="For a model with several outputs: the output to use, numbered from 0 (only this output is saved)",
    )
    parser.add_argument(
        "--thread", "-t", type=int, default=None, help="Number of threads for the pileup (default: 95%% of the CPUs)"
    )
    parser.add_argument(
        "--threshold",
        "-th",
        type=float,
        default=0.98,
        help="Per-read probability at or above which a read counts as modified. It changes the modscore and "
        "count_pos; which sites report a stoichiometry, and its value, do not change.",
    )
    parser.add_argument(
        "--epsilon",
        "-ep",
        type=float,
        default=1e-30,
        help="Small constant that keeps the logarithms and divisions of the site scores finite; keep the default",
    )
    parser.add_argument(
        "--slice",
        "-sl",
        type=int,
        default=None,
        help="For a model with several outputs: save all outputs, and make the pileup from this one (numbered from 0)",
    )
    parser.add_argument(
        "--flip", "-fl", action="store_true", help="Use 1 - p in place of each per-read probability p in the pileup"
    )
    parser.add_argument(
        "--skip-modbam", "-sm", action="store_true", help="Do not write the modBAM file (per-read MM and ML tags)"
    )
    parser.add_argument(
        "--label_div",
        "-d",
        type=int,
        default=10**9,
        help="Factor that packs the reference index and the position into one label ID; must be the --label-div of "
        "deeprm call prep",
    )
    parser.add_argument(
        "--annot",
        "-a",
        type=str,
        default=None,
        help="Annotation file (RefFlat, GenePred or RefGene) to also write the site tables in genomic coordinates",
    )
    return None


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch <= 0:
        raise ValueError("--batch must be a positive integer.")
    if args.flush <= 0:
        raise ValueError("--flush must be a positive integer.")
    if args.prefetch <= 0:
        raise ValueError("--prefetch must be a positive integer.")
    if args.worker < 0:
        raise ValueError("--worker must be non-negative.")
    if not (0.0 <= args.threshold < 1.0):
        raise ValueError("--threshold must satisfy 0 <= threshold < 1.")
    if args.epsilon <= 0:
        raise ValueError("--epsilon must be positive.")
    _check_model_type(getattr(args, "model_type", None))


def _no_gpu_advice():
    """Why PyTorch sees no GPU and how to fix it (the advice of deeprm check). None without an NVIDIA driver or GPU.

    The driver is read as deeprm check reads it (check_deps.cuda_unavailable_cause): from libcuda first, then from the
    nvidia-smi banner.
    """
    cause = check_deps.cuda_unavailable_cause(torch.version.cuda)  # None for a CPU-only build: never driver_too_old
    if torch.version.cuda is not None:
        return check_deps.cuda_unavailable_advice(cause)
    # A CPU-only build
    if cause in ("no_driver", "no_gpu"):  # the CPU is expected
        return None
    if platform.machine().lower() in ("aarch64", "arm64"):  # as deeprm check: the CPU only on ARM
        return (
            "On Linux on ARM (aarch64), DeepRM v1.2 runs on the CPU only: it was not tested with the CUDA builds of "
            "PyTorch for ARM."
        )
    install = check_deps.torch_install_hint()
    if cause == "nvidia_smi_fails":  # as deeprm check: fix the driver first
        advice = check_deps.cuda_unavailable_advice(cause)
        if check_deps.driver_cuda():
            advice += f" Then install a CUDA build of PyTorch: {install}"
        return advice
    return f"This PyTorch is a CPU-only build. To use the GPU: {install}"


def _normalize_gpu_config(args: argparse.Namespace) -> None:
    available_gpu = torch.cuda.device_count()
    if args.gpu_pool is not None:
        if len(args.gpu_pool) == 0:
            raise ValueError("--gpu-pool cannot be empty.")
        if len(set(args.gpu_pool)) != len(args.gpu_pool):
            raise ValueError("--gpu-pool contains duplicate GPU IDs.")
        invalid = [gpu for gpu in args.gpu_pool if gpu < 0 or gpu >= available_gpu]
        if invalid:
            raise ValueError(f"Invalid GPU IDs in --gpu-pool: {invalid}. Available GPUs: 0..{available_gpu - 1}.")
        args.num_gpu = len(args.gpu_pool)
    elif args.num_gpu is None:
        args.num_gpu = available_gpu
        args.gpu_pool = list(range(args.num_gpu))
        advice = _no_gpu_advice() if available_gpu == 0 else None
        if advice:  # no warning without an NVIDIA driver
            log.warning(
                "PyTorch %s (CUDA build: %s) sees no GPU, although an NVIDIA driver is present. Running on the CPU, "
                "which is much slower. %s",
                torch.__version__,
                torch.version.cuda or "none, a CPU-only build",
                advice,
            )
    elif args.num_gpu > 0:
        if args.num_gpu > available_gpu:
            hint = ""
            if available_gpu == 0:
                hint = f" PyTorch sees no GPU. {_no_gpu_advice() or 'No NVIDIA driver or GPU was found.'}"
            raise ValueError(f"Requested {args.num_gpu} GPUs, but only {available_gpu} are available.{hint}")
        args.gpu_pool = list(range(args.num_gpu))
    else:
        args.num_gpu = 0
        args.gpu_pool = []

    if args.num_gpu == 0:
        args.gpu_pool = []
    missing = [gpu for gpu in check_deps.gpus_without_code(torch) if gpu[0] in args.gpu_pool]
    if missing:
        index, name, cc = missing[0]
        visible = [tuple(torch.cuda.get_device_capability(gpu)) for gpu in args.gpu_pool]
        raise ValueError(
            f"PyTorch {torch.__version__} has no code for GPU {index} ({name}, compute capability {cc[0]}.{cc[1]}). "
            f"Install a build for it: {check_deps.torch_install_hint(gpu_ccs=visible)}"
        )


def _bundled_model_path():
    """The bundled v1.2 weights (the default --model)."""
    return os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "weight", "deeprm_weights.pt")


def main(args: argparse.Namespace):
    """Runs per-read inference, then the site-level pileup."""
    if args.model is None:
        args.model = _bundled_model_path()
    if not args.model.endswith(".pt"):
        raise ValueError("Invalid model path. It should be a .pt file.")
    if not os.path.isfile(args.model):
        raise ValueError(f"--model: model file not found: {args.model}")
    if args.data.endswith("/"):
        args.data = args.data[:-1]
    if not os.path.isdir(args.data):
        raise ValueError("Invalid data path. It should be a directory containing data files.")

    _validate_args(args)
    # Check the pileup settings too, so that a bad --annot stops the run before the inference.
    from deeprm.inference.pileup_deeprm import _validate_args as _validate_pileup_args

    _validate_pileup_args(args)
    _normalize_gpu_config(args)

    output = f"{args.output}/{os.path.basename(args.data)}"
    if len(args.postfix) > 0:
        output = f"{output}-{args.postfix}"
    args.output = output

    inference_output = os.path.join(args.output, "molecule-level")
    pileup_output = os.path.join(args.output, "site-level")
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(inference_output, exist_ok=True)
    os.makedirs(pileup_output, exist_ok=True)

    args.output = inference_output
    write_run_info(args)
    fa4_warning = getattr(args, "fa4_warning", None)
    if fa4_warning:
        log.warning("%s", fa4_warning)
    try:
        run_inference(args)
        log.info("Inference Program Finished.")

        from deeprm.inference.pileup_deeprm import main as pileup_main

        args.input = inference_output
        args.output = pileup_output
        pileup_main(args)
        log.info("Pileup Program Finished.")
    finally:
        if fa4_warning:
            log.warning("%s", fa4_warning)  # repeated at the end, where the progress bars cannot hide it
    return None


def _check_model_type(model_type):
    """Refuses a --model-type other than deeprm_v12, the only architecture deeprm call run builds."""
    if model_type is None or model_type == engines.ARCHITECTURE:
        return
    if model_type == "deeprm_model":
        raise ValueError(
            "--model-type deeprm_model: DeepRM v1.2 no longer includes the v1.1 model (architecture deeprm_model). "
            "Run v1.1 model files with DeepRM v1.1 (pip install deeprm==1.1.3); for a v1.2 model file, leave "
            "--model-type out."
        )
    raise ValueError(
        f"--model-type {model_type!r}: deeprm call run builds only {engines.ARCHITECTURE} models. Leave --model-type "
        f"out, or use --model-type {engines.ARCHITECTURE}."
    )


def _check_output_choice(n_label, output_id, slice_idx):
    """Refuses --output-id / --slice that do not fit a model with `n_label` outputs, before the inference.

    The per-read predictions are 1-D for a one-output model or with --output-id, and 2-D otherwise; the pileup's
    --slice picks a column of 2-D predictions.
    """
    if output_id is not None and not 0 <= output_id < n_label:
        raise ValueError(f"--output-id {output_id}: this model has {n_label} output(s), numbered 0 to {n_label - 1}.")
    if slice_idx is not None:
        if n_label == 1:
            raise ValueError(f"--slice {slice_idx}: this model has one output; leave --slice out.")
        if output_id is not None:
            raise ValueError("--slice and --output-id both choose an output of the model; give only --output-id.")
        if not 0 <= slice_idx < n_label:
            raise ValueError(f"--slice {slice_idx}: this model has {n_label} outputs, numbered 0 to {n_label - 1}.")
    elif n_label > 1 and output_id is None:
        raise ValueError(f"This model has {n_label} outputs; choose one with --output-id (0 to {n_label - 1}).")


def load_model_file(path, map_location="cpu"):
    """Loads a model file with torch.load(weights_only=True): tensors and plain values only, so that loading a file
    runs no code from it. Files written by deeprm train, and the bundled weights, are in this form."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as exc:
        reason = (str(exc).splitlines() or [type(exc).__name__])[0]
        raise ValueError(
            f"{path} holds Python objects other than tensors and plain values. DeepRM loads model files with "
            f"torch.load(weights_only=True), so that loading a file runs no code from it; save only the model's "
            f"state_dict, its model_config and plain values. ({reason})"
        ) from exc


def _model_name(model_config, model_type=None, state_dict=None, option="--model"):
    """Returns the model architecture (always deeprm_v12) or raises ValueError.

    A v1.1 file is refused even with --model-type deeprm_v12 (found by its name or, if unnamed, its tensor names). A
    file that names another module, or none, needs --model-type deeprm_v12; the strict weight load then checks every
    tensor.
    `option` is the option that gave the file (--model of call run, --load-checkpoint of train run), for the advice.
    """
    _check_model_type(model_type)
    named = model_config.get("model_type") or model_config.get("model")
    v11_tensors = state_dict is not None and any(str(key).startswith(V11_KEY_PREFIXES) for key in state_dict)
    if named == "deeprm_model" or (not named and v11_tensors):
        if option == "--model":
            advice = "or leave --model out to use the bundled v1.2 model"
        else:
            advice = f"or give {option} a DeepRM v1.2 model file, such as the bundled one"
        raise ValueError(
            "This model file holds a DeepRM v1.1 model (architecture deeprm_model), which DeepRM v1.2 no longer "
            f"includes. Run it with DeepRM v1.1 (pip install deeprm==1.1.3), {advice}."
        )
    if model_type or named == engines.ARCHITECTURE:
        return engines.ARCHITECTURE
    what = f"names the model module {named!r}" if named else "does not name its model architecture"
    raise ValueError(
        f"The model file {what}; deeprm call run runs only DeepRM v1.2 ({engines.ARCHITECTURE}) model files. If this "
        f"file holds an {engines.ARCHITECTURE} model, add --model-type {engines.ARCHITECTURE}."
    )


def _choose_engine(model_config, model_type_arg, requested, device):
    """Returns (engine, architecture, reason) for `device`. A model with dwell/base-quality input runs only on eager."""
    model_name = _model_name(model_config, model_type_arg)
    if _dwell_bq_dim(model_config) > 0 and requested in ("auto", "fa4", "fast"):
        if requested != "auto":
            raise ValueError("This model takes dwell/base-quality input, which only --engine eager supports.")
        requested = "eager"
    engine, reason = engines.resolve_engine(requested, device)
    return engine, model_name, reason


def _recorded_form(info):
    """The form recorded in an existing deeprm_run.json (None if it recorded none)."""
    used = info.get("engine_used") or {}
    if "form" in used:
        return used["form"]
    if used.get("module"):
        return f"module {used['module']}"  # read for compatibility; this code always writes "form"
    if "form" in info:
        return info["form"]  # from write_run_info, before the worker records its engine
    return info.get("model_name")


def _resume_form_error(output_dir, before, now):
    """The ValueError for a resumed run whose engine form differs from the one that wrote the outputs."""
    advice = (
        f"Resume with --engine {before.split('/')[0]}, or use a new output folder."
        if "/" in before
        else "Use a new output folder."
    )
    return ValueError(
        f"--resume: the outputs in {output_dir} were made with {before}; this run would use {now}. {advice}"
    )


def _dwell_bq_dim(model_config):
    """Number of dwell/base-quality input channels the model was trained with (v1.2 model: 0)."""
    dim = 3
    if model_config.get("no_bq", False):
        dim -= 1
    if model_config.get("no_dwell", False):
        dim -= 2
    return dim


def _dwell_bq_channels(model_config):
    """Indices of the (dwell motor, dwell pore, base quality) channels the model takes; None if it takes none."""
    no_bq, no_dwell = model_config.get("no_bq", False), model_config.get("no_dwell", False)
    if no_bq and no_dwell:
        return None
    if no_bq:
        return [0, 1]
    if no_dwell:
        return [2]
    return [0, 1, 2]


def _file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _input_list(data_dir):
    """The input shards of a run as {count, md5 of their sorted file names}, for the --resume check."""
    names = sorted(os.path.basename(path) for path in glob.glob(os.path.join(data_dir, "*.npz")))
    return {"count": len(names), "md5": hashlib.md5("\n".join(names).encode()).hexdigest()}


def write_run_info(args):
    """Writes deeprm_run.json (the model and run settings) to the per-read output folder; the pileup reads it.

    Raises ValueError for a non-deeprm_v12 model file, for --output-id / --slice that do not fit the model's outputs,
    for GPUs that cannot share one engine form, without ``--resume`` for a folder that already holds per-read outputs
    and, with ``--resume``, for a model file, input files, device type, GPU count or engine form that differ from the
    outputs'.
    """
    path = os.path.join(args.output, RUN_INFO_NAME)
    has_outputs = bool(glob.glob(os.path.join(args.output, "inference_*.npz")))
    if has_outputs and not getattr(args, "resume", False):
        # this run would rewrite only its own files, and the pileup would add up the old ones too
        raise ValueError(
            f"{args.output} holds per-read outputs of an earlier run; use --resume, or a new or empty output folder."
        )
    model_md5 = _file_md5(args.model)
    n_ranks = max(1, getattr(args, "num_gpu", 0) or 0)
    device_type = "cuda" if (getattr(args, "num_gpu", 0) or 0) > 0 else "cpu"
    inputs = _input_list(args.data)
    previous = {}
    if getattr(args, "resume", False):
        if os.path.isfile(path):
            previous = read_run_info(path)
            if previous.get("model_md5") != model_md5:
                raise ValueError(
                    f"--resume: the outputs in {args.output} were made with another model file "
                    f"({previous.get('model_path')}, md5 {previous.get('model_md5')}). "
                    "Use that model, or a new output folder."
                )
            if previous.get("n_ranks", n_ranks) != n_ranks:
                raise ValueError(
                    f"--resume: the outputs in {args.output} were made with {previous.get('n_ranks')} GPU processes; "
                    f"this run has {n_ranks}. Resume with the same number of GPUs, or use a new output folder."
                )
            if previous.get("device_type", device_type) != device_type:
                raise ValueError(
                    f"--resume: the outputs in {args.output} were made on the {previous['device_type'].upper()}; "
                    f"this run would use the {device_type.upper()}. Resume on the same kind of device, or use a new "
                    "output folder."
                )
            if previous.get("inputs", inputs) != inputs:
                raise ValueError(
                    f"--resume: the input folder {args.data} holds other files than when the outputs in {args.output} "
                    f"were made ({previous['inputs']['count']} files then, {inputs['count']} now). Resuming after the "
                    "input files changed would skip or repeat shards. Restore them (re-create a removed shard under "
                    "the same name), or use a new output folder."
                )
            output_id = getattr(args, "output_id", None)
            if "output_id" in previous and previous["output_id"] != output_id:
                raise ValueError(
                    f"--resume: the outputs in {args.output} were made with --output-id {previous['output_id']}; this "
                    f"run has --output-id {output_id}. Resume with the same --output-id, or use a new output folder."
                )
        elif has_outputs:
            raise ValueError(
                f"--resume: {args.output} holds per-read outputs without {RUN_INFO_NAME} (made by DeepRM before v1.2). "
                "Resume them with the DeepRM version and model that made them, or use a new output folder."
            )
    save_dict = load_model_file(args.model)
    model_config = save_dict.get("model_config", {})
    _model_name(model_config, args.model_type, save_dict.get("model_state_dict"))  # refuses a v1.1 or unknown file
    meta = save_dict.get("deeprm_meta", {})
    n_label = int(model_config.get("n_label", 1))
    _check_output_choice(n_label, getattr(args, "output_id", None), getattr(args, "slice", None))
    pool = list(getattr(args, "gpu_pool", None) or [])
    devices = [torch.device("cuda", gpu) for gpu in pool] if pool else [torch.device("cpu")]
    if pool and engines.is_rocm():
        log.warning("DeepRM v1.2 was not tested on AMD GPUs (ROCm). FlashAttention-4 (fa4) does not run on them.")
    choices = [_choose_engine(model_config, args.model_type, args.engine, device) for device in devices]
    forms = sorted({engines.engine_form(engine) for engine, _, _ in choices})
    if len(forms) > 1 and args.engine == "auto":
        # GPUs of different types: use one engine that all of them can run
        common = "eager" if any(engine == "eager" for engine, _, _ in choices) else "fast"
        log.warning(
            "The GPUs of this run (%s) would run different engines (%s); all of them use %s.",
            pool,
            ", ".join(forms),
            common,
        )
        choices = [_choose_engine(model_config, args.model_type, common, device) for device in devices]
        why = f"GPUs of different types: {common} on all of them"
        choices = [(engine, module, why) for engine, module, _ in choices]
        forms = sorted({engines.engine_form(engine) for engine, _, _ in choices})
    if len(forms) > 1:
        raise ValueError(
            f"The GPUs of this run ({pool}) would run different engines ({', '.join(forms)}). "
            "Use GPUs of one type (--gpu-pool), or choose an engine that runs on all of them (--engine fast)."
        )
    before = _recorded_form(previous)
    if before is not None and before != forms[0]:
        raise _resume_form_error(args.output, before, forms[0])
    info = {
        "deeprm_version": _package_version(),
        "model_path": os.path.abspath(args.model),
        "model_md5": model_md5,
        # true only for the bundled v1.2 weights; deeprm call pileup reads this
        "site_rule_calibration": model_md5 == BUNDLED_WEIGHTS_MD5,
        "output_id": getattr(args, "output_id", None),
        "model_name": choices[0][1],
        "form": forms[0],
        "engine_requested": args.engine,
        "model_description": meta.get("description"),
        "batch": args.batch,
        "device_type": device_type,
        "n_ranks": n_ranks,
        "inputs": inputs,
    }
    args.engine_resolved = list(choices[0][::2])  # (engine, reason)
    args.fa4_warning = _fa4_warning(args, model_config, pool, devices, choices)  # main prints it at start and end
    if previous.get("engine_used"):
        info["engine_used"] = previous["engine_used"]  # checked above, and again by rank 0 in _record_engine
    del save_dict
    save_run_info(path, info)
    return info


def _fa4_warning(args, model_config, pool, devices, choices):
    """The warning for a GPU run that misses fa4 for a reason the user can fix, or because fa4 was not tested on that
    GPU; else None. Built once, in the parent."""
    if args.engine != "auto" or not pool or _dwell_bq_dim(model_config) > 0:
        return None  # on purpose: an engine was asked for, a CPU run, or an eager-only model
    try:
        found = engines.fa4_warning([(gpu, device, choice[0]) for gpu, device, choice in zip(pool, devices, choices)])
    except Exception as exc:  # noqa: BLE001 -- a warning must never stop a run
        log.debug("Could not check why fa4 is not used: %s", exc)
        return None
    return found[1] if found else None


def _record_engine(output_dir, engine, reason, device):
    """Records rank 0's engine in deeprm_run.json; refuses a resumed run that would change the recorded form."""
    path = os.path.join(output_dir, RUN_INFO_NAME)
    if not os.path.isfile(path):
        return
    info = read_run_info(path)
    form = engines.engine_form(engine)
    before = _recorded_form({"engine_used": info.get("engine_used")}) if info.get("engine_used") else None
    if before is not None and before != form:
        raise _resume_form_error(output_dir, before, form)
    info["engine_used"] = {"engine": engine, "form": form, "reason": reason, "device": str(device), "rank": 0}
    save_run_info(path, info)


def _package_version():
    try:
        from deeprm import __version__

        return __version__
    except ImportError:
        return "unknown"


def sigmoid_np(x):
    """Numerically stable sigmoid in float32."""
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    np.negative(x, out=out, where=pos)
    np.exp(out, out=out, where=pos)
    out[pos] = 1.0 / (1.0 + out[pos])
    np.exp(x, out=out, where=~pos)
    out[~pos] = out[~pos] / (1.0 + out[~pos])
    return out


def run_inference(args):
    """Runs per-read inference: one spawned process per GPU, or this process on the CPU."""
    torch.multiprocessing.set_sharing_strategy("file_system")
    log.info("Inference Program Started.")
    if args.num_gpu > 0:
        log.info(f"Using {args.num_gpu} GPUs: {args.gpu_pool}.")
    else:
        log.info("Using CPU.")

    log.info(f"Model path: {args.model}")
    log.info(f"Output directory: {args.output}")

    if args.num_gpu > 0:
        mp.spawn(inference_worker, nprocs=args.num_gpu, args=(vars(args),), join=True)
    else:
        inference_worker(0, vars(args))
    return None


def _discover_resume_point(output_dir: str, rank: int) -> int:
    # A .part file is a per-read file whose write was cut off (inference_loop renames it when complete); its shard
    # runs again.
    for part in glob.glob(os.path.join(output_dir, f"inference_{rank}_*.part")):
        try:
            os.remove(part)
        except OSError:
            pass
    pattern = os.path.join(output_dir, f"inference_{rank}_*.npz")
    paths = glob.glob(pattern)
    if not paths:
        return 0

    indexed_paths = {}
    for path in paths:
        stem = os.path.basename(path).rsplit(".", 1)[0]
        try:
            idx = int(stem.split("_")[-1])
        except ValueError:
            continue
        indexed_paths[idx] = path

    saved = 0
    while saved in indexed_paths:
        saved += 1

    stale_indices = sorted(idx for idx in indexed_paths if idx >= saved)
    if stale_indices:
        for idx in stale_indices:
            try:
                os.remove(indexed_paths[idx])
            except OSError:
                pass
        log.warning(
            "Found non-contiguous or stale inference outputs for rank %s. Removed shard indices >= %s: %s",
            rank,
            saved,
            stale_indices,
        )

    return saved


def inference_worker(rank, args_dict):
    """Runs inference in one process, on GPU gpu_pool[rank] or on the CPU. `args_dict` is vars(args)."""
    if args_dict["num_gpu"] > 0:
        # A spawned rank. torch's spawn treats a KeyboardInterrupt in a rank as a clean exit, expecting the parent to
        # have been interrupted too. A SIGINT that reaches only this rank would then pass as a success, and the pileup
        # would run on incomplete outputs, so it is turned into an error.
        try:
            return _inference_worker(rank, args_dict)
        except KeyboardInterrupt as interrupt:
            message = f"Rank {rank} was interrupted; its outputs are incomplete. Run again with --resume."
            # A Ctrl-C while the rank was stopping on an error (writing its queued files first) would hide that error;
            # the pipelined loop records it as stopped_on. Not __context__: that is any error being handled when the
            # Ctrl-C came, such as an out-of-memory error the loop was recovering from.
            first = getattr(interrupt, "stopped_on", None)
            if first is not None:
                raise RuntimeError(f"{message} It had stopped on this error first: {first!r}") from first
            raise RuntimeError(message) from None
    return _inference_worker(rank, args_dict)


def _inference_worker(rank, args_dict):
    use_gpu = args_dict["num_gpu"] > 0
    device = torch.device(f"cuda:{args_dict['gpu_pool'][rank]}") if use_gpu else torch.device("cpu")
    if use_gpu:
        # This process's default GPU. Without it, CUDA work that names no device (the DataLoader's pin-memory thread,
        # stream switches) runs on cuda:0 and opens a second CUDA context there for every rank.
        torch.cuda.set_device(device)

    # Every tensor loads onto this process's device, whatever device it was saved from (a file that deeprm train run
    # saved with --gpu-pool 2 3 has its tensors on cuda:2).
    save_dict = load_model_file(args_dict["model"], map_location=device)
    model_config = save_dict["model_config"]
    # Checked again here for callers that start the worker without write_run_info.
    _model_name(model_config, args_dict.get("model_type"), save_dict["model_state_dict"])

    resolved = args_dict.get("engine_resolved")  # the engine write_run_info chose for this run, and why
    requested = resolved[0] if resolved else args_dict.get("engine", "auto")
    engine, _, reason = _choose_engine(model_config, args_dict.get("model_type"), requested, device)
    if resolved and engine == resolved[0]:
        reason = resolved[1]
    log.info("Rank %d: engine %s (%s), form %s.", rank, engine, reason, engines.engine_form(engine))
    if rank == 0:
        _record_engine(args_dict["output"], engine, reason, device)
    args_dict["dwell_bq_channels"] = _dwell_bq_channels(model_config)
    if model_config.get("drop_kmer_input", False):
        raise ValueError("This model was trained with drop_kmer_input=True, which deeprm call run does not support.")

    args_dict["output_logits"] = bool(model_config.get("output_logits", False))
    model = _build_model(engines.ENGINE_CLASSES[engine], model_config)

    if rank == 0:
        total_params = sum(parameter.numel() for parameter in model.parameters())
        log.info("Model parameters: %s", f"{total_params:,}")

    model.load_state_dict(state_dict=save_dict["model_state_dict"], strict=True)  # every tensor must match
    save_dict.clear()  # load_state_dict copied the tensors; free them
    # The weights stay fp32; on a GPU, the autocast in _inference_loop_sync / _inference_loop_pipelined sets the
    # compute precision.
    model.to(device)
    # Required: the fast and fa4 engines fold the head's Linear -> BatchNorm pairs into one Linear, which is exact
    # only in eval mode.
    model.eval()
    # Fix U_pad = centre_slot_block(--batch) before the first call (fast and fa4 only). U_pad is a static shape of the
    # compiled graph, so without this a later, larger batch would compile a new graph. The smaller parts that
    # _predict and _launch use after an out-of-memory error keep this U_pad.
    if device.type == "cuda" and hasattr(model, "set_batch_size"):
        model.set_batch_size(args_dict["batch"])  # GPU only: the CPU runs uncompiled

    resume_from = _discover_resume_point(args_dict["output"], rank) if args_dict["resume"] else 0
    if resume_from > 0:
        log.info("Rank %d resuming from input shard index %d.", rank, resume_from)

    data_loader = load_dataset(
        args_dict["data"],
        rank,
        max(1, args_dict["num_gpu"]),
        num_files_read_once=args_dict["prefetch"],
        prefetch_factor=args_dict["prefetch"],
        worker=args_dict["worker"],
        cb_len=model_config["block_len"] + model_config["kmer_size"] - 1,
        kmer_len=model_config["kmer_size"],
        sampling=int(model_config["signal_size"] / model_config["kmer_size"]),
        sig_window=model_config["kmer_size"],
        resume_from=resume_from,
        pin_memory=use_gpu,
        max_seq_len=model.max_seq_len,  # a shard with more signal windows is skipped, not run
    )

    inference_loop(args_dict, rank, device, model, data_loader, start_index=resume_from)
    return None


def _build_model(model_class, model_config):
    """An untrained `model_class` model with the sizes in `model_config`."""
    return model_class(
        d_model=model_config["enc_dim"],
        n_heads=model_config["head"],
        d_ff=model_config["lin_dim"],
        n_layers=model_config["enc_layer"],
        lin_depth=model_config["lin_layer"],
        t_act=model_config["t_act"],
        lin_act=model_config["lin_act"],
        encoder_dropout=model_config["enc_dropout"],
        lin_dropout=model_config["lin_dropout"],
        kmer_size=model_config["kmer_size"],
        signal_size=model_config["signal_size"],
        block_len=model_config["block_len"],
        seq_len=model_config["seq_len"],
        signal_stride=model_config["signal_stride"],
        dwell_bq_dim=_dwell_bq_dim(model_config),
        kmer_mode=model_config.get("kmer_mode", "5mer"),
        n_class=model_config.get("n_class", 1),
        n_label=model_config.get("n_label", 1),
        output_logits=bool(model_config.get("output_logits", False)),
    )


def to_device(data, device, dwell_bq_channels=(0, 1, 2)):
    """Moves the model inputs of one batch part (B rows) to `device`.

    Args:
        dwell_bq_channels: channels of dwell_bq_token the model takes; None (the v1.2 model) passes None instead.

    Returns:
        tuple: src_kmer [B, M] int32 (bases as ASCII codes), src_signal [B, R] float32, src_seg_len [B, L] int32
            (number of 6-sample steps, = signal windows, of each base), src_dwell_bq [B, L, E] float32 or None.
    """
    # Non-blocking copies let the host queue the model's work while they run. They are asynchronous only from pinned
    # memory, which the data loader gives on a GPU run with workers. On the CPU, .to() returns the same tensor.
    non_blocking = device.type == "cuda"
    dwell_bq = None
    if dwell_bq_channels is not None:
        dwell_bq = data["dwell_bq_token"]  # [B, L, 3] float32: dwell motor, dwell pore, base quality
        if list(dwell_bq_channels) != [0, 1, 2]:
            dwell_bq = dwell_bq[..., list(dwell_bq_channels)]  # [B, L, 3] -> [B, L, E]; a list index makes a copy
        dwell_bq = dwell_bq.to(device, non_blocking=non_blocking)
    return (
        data["kmer_token"].to(device, non_blocking=non_blocking),
        data["signal_token"].to(device, non_blocking=non_blocking),
        data["segment_len"].to(device, non_blocking=non_blocking),
        dwell_bq,
    )


def _flush_cpu_buffers(pred_buffer, label_id_buffer, read_id_buffer, pred_parts, label_parts, read_parts):
    if not pred_buffer:
        return
    pred_parts.append(np.concatenate(pred_buffer, axis=0))
    label_parts.append(np.concatenate(label_id_buffer, axis=0))
    read_parts.append(np.concatenate(read_id_buffer, axis=0))
    pred_buffer.clear()
    label_id_buffer.clear()
    read_id_buffer.clear()


def _pick_output(pred, output_id):
    """With --output-id, [B, C] model output -> [B], that output's column; otherwise unchanged. Both loops use it."""
    if output_id is not None and pred.ndim == 2:
        pred = pred[:, output_id]  # [B, C] -> [B]
    return pred


def _drop_single_column(pred):
    """[B, 1] (a one-output model) -> [B]; other shapes unchanged. Both loops use it."""
    if pred.ndim == 2 and pred.shape[1] == 1:
        pred = pred[:, 0]  # [B, 1] -> [B]
    return pred


def _to_numpy_pred(pred, output_id=None):
    """Model output -> float32 numpy array: [B] for one output, [B, C] for several outputs without --output-id."""
    pred = _pick_output(pred, output_id)
    # fast and fa4 return fp16 under autocast; .float() makes float32. .cpu() waits for the GPU to finish this call.
    return _drop_single_column(pred.detach().float().cpu().numpy())


CPU_ROWS = 1000  # rows per model call on the CPU (16,000 rows at once would need about 51 GB of memory)


def _halve_rows(fit, rows):
    """After a GPU out-of-memory error at `rows` rows per model call, halves B (fit["rows"]). Both loops use it.

    `fit` lives for the whole loop, so later batches keep the smaller B; "halved" keeps the first B that failed, for
    the warning of _report_halving.
    """
    fit.setdefault("halved", rows)
    fit["rows"] = max(1, rows // 2)


def _report_halving(fit, rows):
    """Warns once, after the first batch that ran at the halved size `rows`."""
    if fit.get("halved") and not fit.get("reported"):
        fit["reported"] = True
        log.warning(
            "The GPU has too little memory for %d rows at a time; this run uses %d rows at a time. The results stay "
            "the same apart from small float differences. (--batch %d skips this step.)",
            fit["halved"],
            rows,
            rows,
        )


def _predict(model, batch_cpu, device, dwell_bq_channels, output_id, fit):
    """Runs the model on one batch in parts of fit["rows"] rows; returns [n_rows] float32 predictions.

    After a GPU out-of-memory error the part size is halved and kept for the rest of the run. A row's prediction does
    not depend on the other rows, so the results stay the same apart from small float differences.
    """
    n_rows = int(batch_cpu["label_id"].shape[0])  # rows of this batch (at most --batch)
    while True:
        rows = fit["rows"]  # B: rows per model call
        try:
            parts = []
            for start in range(0, n_rows, rows):
                # rows [start, start + B) of every field, as views; the last part may be shorter
                part = {key: value[start : start + rows] for key, value in batch_cpu.items()}
                # [B, C] logits -> [B] float32 numpy on the host (C = 1)
                parts.append(_to_numpy_pred(model(*to_device(part, device, dwell_bq_channels)), output_id))
            _report_halving(fit, rows)
            # parts in row order -> [n_rows] float32 ([n_rows, C] for several outputs and no --output-id)
            return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
        except torch.OutOfMemoryError:
            if device.type != "cuda" or rows <= 1:
                raise
            # Halve B and rerun the whole batch. The earlier parts have finished: .cpu() waited for each of them.
            torch.cuda.empty_cache()
            _halve_rows(fit, rows)


def _inference_loop_sync(args_dict, rank, device, model, data_loader, start_index=0):
    """The synchronous loop: H2D copy, model and D2H copy in series, one batch at a time. Used on the CPU, or on a
    GPU with DEEPRM_SYNC_LOOP=1; inference_loop otherwise runs the pipelined loop below.

    Args:
        start_index: index of this rank's first shard (the resume point); it numbers the output files.
    """
    # GPU: fp16 autocast for the whole loop (fa4 takes only fp16 or bf16). cache_enabled keeps each weight's fp16
    # copy for the whole loop, so uncompiled code casts it once. CPU: fp32.
    amp_ctx = autocast(enabled=device.type == "cuda", cache_enabled=device.type == "cuda", device_type="cuda")
    if device.type != "cuda":
        amp_ctx = nullcontext()

    dwell_bq_channels = args_dict.get("dwell_bq_channels", [0, 1, 2])  # None for the v1.2 model
    output_logits = args_dict.get("output_logits", False)  # True for the v1.2 model: it returns logits
    # B: --batch on a GPU (halved on out-of-memory), at most CPU_ROWS on the CPU
    fit = {"rows": args_dict["batch"] if device.type == "cuda" else min(args_dict["batch"], CPU_ROWS)}
    with amp_ctx:
        with torch.inference_mode():
            processed_any = False
            unreadable = 0
            for chunk_idx, chunk_data_cpu in enumerate(
                tqdm.tqdm(data_loader, total=len(data_loader), smoothing=0), start=start_index
            ):
                if chunk_data_cpu is None:  # a shard that could not be read or is malformed (warned by the loader)
                    unreadable += 1
                    continue
                # One input shard (one prep .npz file) as CPU tensors: kmer_token [n_rows, M] int32 (ASCII bases),
                #   signal_token [n_rows, R] float32, segment_len [n_rows, L] int32, dwell_bq_token [n_rows, L, 3],
                #   label_id [n_rows] int64 (reference, position and strand of the A in one integer),
                #   read_id [n_rows, 2] int64 (the 16 bytes of the read UUID).
                n_rows = int(chunk_data_cpu["label_id"].shape[0])
                if n_rows == 0:
                    log.warning("Skipping empty input shard at rank=%d chunk=%d.", rank, chunk_idx)
                    continue

                processed_any = True
                # [n_rows, ...] -> views of --batch rows (no copy); the last one may be shorter
                batch_splits = {k: torch.split(v, args_dict["batch"]) for k, v in chunk_data_cpu.items()}
                n_batches = len(batch_splits["label_id"])

                pred_buffer = []
                label_id_buffer = []
                read_id_buffer = []
                pred_parts = []
                label_parts = []
                read_parts = []

                for bidx in range(n_batches):
                    batch_data_cpu = {k: v[bidx] for k, v in batch_splits.items()}
                    pred = _predict(model, batch_data_cpu, device, dwell_bq_channels, args_dict["output_id"], fit)
                    if NONFINITE_KEY in batch_data_cpu:
                        # rows whose inputs held NaN or Inf; the data loader zeroed them so the batch stays finite
                        pred[batch_data_cpu[NONFINITE_KEY].numpy()] = np.nan
                    pred_buffer.append(pred)
                    # the ids of the same rows (numpy views of the CPU tensors)
                    label_id_buffer.append(batch_data_cpu["label_id"].numpy())
                    read_id_buffer.append(batch_data_cpu["read_id"].numpy())

                    # every --flush batches, join the buffers into one part (row order kept)
                    if len(pred_buffer) >= args_dict["flush"]:
                        _flush_cpu_buffers(
                            pred_buffer,
                            label_id_buffer,
                            read_id_buffer,
                            pred_parts,
                            label_parts,
                            read_parts,
                        )

                _flush_cpu_buffers(
                    pred_buffer,
                    label_id_buffer,
                    read_id_buffer,
                    pred_parts,
                    label_parts,
                    read_parts,
                )

                if not pred_parts:
                    log.warning("No predictions were produced for rank=%d chunk=%d.", rank, chunk_idx)
                    continue

                # One array per field in shard row order, so row i is the same read and A in all three:
                # preds [n_rows] float32, label_ids [n_rows] int64, read_ids [n_rows, 2] int64.
                preds = pred_parts[0] if len(pred_parts) == 1 else np.concatenate(pred_parts, axis=0)
                label_ids = label_parts[0] if len(label_parts) == 1 else np.concatenate(label_parts, axis=0)
                read_ids = read_parts[0] if len(read_parts) == 1 else np.concatenate(read_parts, axis=0)

                out_path = f"{args_dict['output']}/inference_{rank}_{chunk_idx}.npz"
                _write_shard(out_path, preds, label_ids, read_ids, output_logits)

            _report_rank_end(rank, processed_any, unreadable)

    return None


def _write_shard(out_path, preds, label_ids, read_ids, output_logits):
    """Writes one shard's per-read file (both loops use it): preds [n_rows] (logits if output_logits), label_ids
    [n_rows] int64, read_ids [n_rows, 2] int64."""
    # Logits -> float32 probabilities. A row whose centre base has no signal windows has a NaN logit (0 / 0 in the
    # centre-token mean) and stays NaN.
    if output_logits:
        preds = sigmoid_np(preds)  # pred = probability that the centre A is m6A
    # Written under a name that does not end in .npz, then renamed: a run stopped during the write leaves no cut-off
    # .npz file (the pileup reads every .npz file; --resume counts each one as done).
    part_path = out_path + ".part"
    with open(part_path, "wb") as fh:
        np.savez_compressed(fh, label_id=label_ids, read_id=read_ids, pred=preds)
    os.replace(part_path, out_path)


def _report_rank_end(rank, processed_any, unreadable):
    """The end-of-rank messages of both loops; raises RuntimeError if the rank could use none of its shards."""
    if unreadable:
        log.warning(
            "Rank %d skipped %d input shard(s) that could not be read or were malformed; their reads are missing from "
            "the results.",
            rank,
            unreadable,
        )
    if not processed_any and unreadable:
        raise RuntimeError(
            f"Rank {rank} could use none of its {unreadable} input shard(s); see the warnings above. "
            "Check that --input is the folder that deeprm call prep wrote."
        )
    if not processed_any:
        log.warning("No input shards were processed for rank=%d. No inference outputs were written.", rank)


# ================================================================================================ pipelined GPU loop
# The GPU does not wait for the host: the next batch is copied in on a separate CUDA stream while the current one
# computes, the token counts the fa4 engine needs are summed on the CPU before the copy (no device-to-host read in
# the engine), predictions return by asynchronous copies into pinned memory that are read two batches later, and
# the per-shard files are written by a background thread. The same batches run through the same kernels in the
# same order, so the outputs are those of the synchronous loop.

SYNC_LOOP_ENV = "DEEPRM_SYNC_LOOP"  # =1 runs the synchronous loop on a GPU too (for comparison)
MAX_IN_FLIGHT = 2  # batches whose predictions are still on their way to the host while the next ones are queued
WRITER_QUEUE = 8  # shards waiting to be written before the loop pauses


class _ShardWriter:
    """Writes each shard's npz in a background thread, so the loop waits for the disk only when WRITER_QUEUE shards
    are already waiting to be written."""

    def __init__(self, output_logits):
        self._queue = queue.Queue(maxsize=WRITER_QUEUE)
        self._error = None
        self._reported = False
        self._output_logits = output_logits
        self._handed = 0  # shards put in the queue (counted by the loop's thread)
        self._written = 0  # shards written (counted by the writer thread)
        self._done = threading.Event()  # set when the writer thread ends, after its last write or on an error
        self._thread = threading.Thread(target=self._run, name="deeprm-writer", daemon=True)
        self._thread.start()

    def put(self, out_path, preds, label_ids, read_ids):
        while True:
            self.check()
            if not self._thread.is_alive():
                raise RuntimeError("The output writer thread stopped without writing all files.")
            try:
                self._queue.put((out_path, preds, label_ids, read_ids), timeout=1.0)
                self._handed += 1
                return
            except queue.Full:
                continue

    def close(self):
        """Writes the shards still queued, then stops the thread. A Ctrl-C while it waits (for room in the queue, or
        for the writes) still lets them be written (a second one stops at once; --resume then redoes them) and is
        raised again afterwards."""
        try:
            self._queue_end()
            left = self._handed - self._written  # the queued shards and the one being written
            if left > 0 and self._thread.is_alive():
                log.info("Writing the last %d output file(s).", left)
            self._wait()
        except KeyboardInterrupt:
            log.warning("Interrupted while writing the last output files; the files already queued are written first.")
            self._queue_end()  # again if the Ctrl-C came before the end marker was queued (a second marker is unread)
            self._wait()
            raise
        self.check()

    def stopped(self):
        """True if a write failed or the writer thread has ended (before close(), it ends only on a failed write): it
        writes no more files."""
        return self._error is not None or self._done.is_set()

    def _queue_end(self):
        """Queues the end marker, after the shards already queued; waits for room in the queue."""
        # A writer that stopped on an error no longer takes items: never wait for room in its queue then.
        while self._thread.is_alive():
            try:
                self._queue.put(None, timeout=1.0)
                return
            except queue.Full:
                continue

    def _wait(self):
        # Not Thread.join(): on Python 3.10 to 3.12 a join() that a Ctrl-C cuts short marks the thread as stopped
        # although it is still writing (CPython bpo-45274), so a second join() would return at once and the rank would
        # exit in the middle of a file. An Event.wait() that is cut short can simply be called again.
        while not self._done.wait(0.5):
            pass

    def check(self):
        """Raises the writer's error, once (a failed write stops the loop at its next batch)."""
        if self._error is not None and not self._reported:
            self._reported = True
            raise RuntimeError(f"Writing an output file failed: {self._error!r}") from self._error

    def _run(self):
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    return
                try:
                    _write_shard(*item, self._output_logits)
                    self._written += 1
                except BaseException as exc:  # noqa: BLE001 -- reported to the loop by the next put / close / check
                    self._error = exc
                    return
        finally:
            self._done.set()


# One batch whose predictions are on their way to the host: host_pred is the pinned buffer they are copied into, and
# event fires when that copy is done. last: the shard's last batch. nonfinite_rows: [B] bool or None (see
# _queue_result).
_Pending = collections.namedtuple("_Pending", "chunk_idx last host_pred event label_ids read_ids nonfinite_rows")


def batch_counts(seg_len_cpu, centre):
    """(real tokens, centre tokens, longest row, any negative length) of a CPU batch, summed with numpy.

    The fa4 engine needs the first two to size its graph, and the last two for its input checks. Summed on the host,
    they spare the engine a device-to-host read, which would make the host wait for the GPU. Each value is computed
    as the engine computes it on the device, so both give the same output.
    """
    # The engine's own computation of these values is in deeprm_v12_engines.Fa4Engine._forward_batch.
    seg = seg_len_cpu.numpy()  # [B, L] int32, a view of the (pinned) loader memory
    row_len = seg.sum(axis=1, dtype=np.int64)  # [B]
    return (
        int(row_len.sum()),
        # clamped at 0 like the engine's count; a negative length is refused anyway, before the counts are used
        int(np.maximum(seg[:, centre].astype(np.int64), 0).sum()),
        int(row_len.max()) if row_len.size else 0,
        bool((seg < 0).any()),
    )


def _launch(model, batch_cpu, device, dwell_bq_channels, output_id, fit, streams, centre):
    """Queues one batch on the GPU: H2D on the copy stream, the model on the compute stream. Returns [B, C] on
    the device (or [B] with --output-id), not yet computed. Halves the part size after an out-of-memory error, as
    _predict does."""
    copy_stream, compute_stream = streams
    n_rows = int(batch_cpu["label_id"].shape[0])
    accepts_counts = getattr(model, "ACCEPTS_COUNTS", False)
    while True:
        rows = fit["rows"]
        try:
            parts = []
            for start in range(0, n_rows, rows):
                part = {key: value[start : start + rows] for key, value in batch_cpu.items()}
                with torch.cuda.stream(copy_stream):
                    inputs = to_device(part, device, dwell_bq_channels)  # asynchronous from pinned memory
                # The model's kernels, queued on the compute stream, must not start before its inputs have arrived.
                compute_stream.wait_stream(copy_stream)
                # The inputs were allocated on the copy stream. record_stream keeps the allocator from giving their
                # memory to a later copy while the compute stream still reads them (that would change predictions
                # silently, without an error).
                for tensor in inputs:
                    if tensor is not None:
                        tensor.record_stream(compute_stream)
                # A centre index outside the rows would make batch_counts fail with an IndexError; without host counts,
                # the engine refuses such input with its own message.
                use_counts = accepts_counts and 0 <= centre < part["segment_len"].shape[1]
                kwargs = {"counts": batch_counts(part["segment_len"], centre)} if use_counts else {}
                parts.append(_pick_output(model(*inputs, **kwargs), output_id))
            _report_halving(fit, rows)
            return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        except torch.OutOfMemoryError:
            if rows <= 1:
                raise
            # Halve B and rerun the whole batch, as _predict does. Unlike there, earlier batches may still be running
            # on the GPU (nothing has waited for them): wait for them, so that the retry starts on an idle GPU.
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            _halve_rows(fit, rows)


def _queue_result(in_flight, pending_batch, compute_stream, nonfinite_rows=None):
    """Starts the asynchronous copy of one batch's predictions to pinned memory and remembers it.

    nonfinite_rows ([B] bool numpy, or None): rows whose predictions _collect_result sets to NaN.
    """
    chunk_idx, last, pred, label_ids, read_ids = pending_batch
    # The prediction's own dtype (no GPU temporary), or float32 if numpy cannot hold it (bfloat16); pinned on a GPU
    dtype = pred.dtype if pred.dtype in (torch.float16, torch.float32, torch.float64) else torch.float32
    host = torch.empty(pred.shape, dtype=dtype, pin_memory=pred.is_cuda)
    host.copy_(pred, non_blocking=True)  # queued after the model's kernels
    event = torch.cuda.Event()
    event.record(compute_stream)
    in_flight.append(_Pending(chunk_idx, last, host, event, label_ids, read_ids, nonfinite_rows))


def _collect_result(pending, shard_parts, writer, output_dir, rank):
    """Waits for one batch's predictions; when its shard is complete, hands the shard to the writer."""
    pending.event.synchronize()
    writer.check()  # a failed write stops the loop within one batch
    # astype always copies, so the pinned buffer is freed once `pending` is dropped (astype(copy=False) would keep a
    # view of it for float32 output); fp16 -> float32 is exact, as .float() on the GPU is.
    pred = _drop_single_column(pending.host_pred.numpy().astype(np.float32))
    if pending.nonfinite_rows is not None:
        pred[pending.nonfinite_rows] = np.nan  # rows whose inputs held NaN or Inf (zeroed by the data loader)
    parts = shard_parts.setdefault(pending.chunk_idx, ([], [], []))
    parts[0].append(pred)
    parts[1].append(pending.label_ids)
    parts[2].append(pending.read_ids)
    if pending.last:
        preds, labels, reads = (p[0] if len(p) == 1 else np.concatenate(p, axis=0) for p in parts)
        del shard_parts[pending.chunk_idx]
        writer.put(f"{output_dir}/inference_{rank}_{pending.chunk_idx}.npz", preds, labels, reads)


def inference_loop(args_dict, rank, device, model, data_loader, start_index=0):
    """Runs the model on this rank's shards; writes inference_{rank}_{shard}.npz for each non-empty, usable shard.

    On a GPU the loop is pipelined: the next batch is copied in while the current one computes, and a background
    thread writes the files. On the CPU, or with DEEPRM_SYNC_LOOP=1, the loop is synchronous. Both write the same
    files.

    Args:
        start_index: index of this rank's first shard (the resume point); it numbers the output files.
    """
    if device.type != "cuda" or os.environ.get(SYNC_LOOP_ENV, "") == "1":
        return _inference_loop_sync(args_dict, rank, device, model, data_loader, start_index=start_index)
    return _inference_loop_pipelined(args_dict, rank, device, model, data_loader, start_index=start_index)


def _inference_loop_pipelined(args_dict, rank, device, model, data_loader, start_index=0):
    """The pipelined GPU loop (see the comment above _ShardWriter); writes the same files as _inference_loop_sync."""
    amp_ctx = autocast(enabled=True, cache_enabled=True, device_type="cuda")
    dwell_bq_channels = args_dict.get("dwell_bq_channels", [0, 1, 2])  # None for the v1.2 model
    output_logits = args_dict.get("output_logits", False)
    output_id = args_dict["output_id"]
    batch_rows = args_dict["batch"]
    fit = {"rows": batch_rows}
    centre = model.block_len // 2
    copy_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.current_stream(device)
    streams = (copy_stream, compute_stream)
    writer = _ShardWriter(output_logits)
    in_flight = collections.deque()
    shard_parts = {}
    processed_any = False
    unreadable = 0
    try:
        with amp_ctx, torch.inference_mode():
            for chunk_idx, chunk_data_cpu in enumerate(
                tqdm.tqdm(data_loader, total=len(data_loader), smoothing=0), start=start_index
            ):
                if chunk_data_cpu is None:  # a shard that could not be read or is malformed (warned by the loader)
                    unreadable += 1
                    continue
                # One input shard as CPU tensors (fields and shapes as in _inference_loop_sync)
                n_rows = int(chunk_data_cpu["label_id"].shape[0])
                if n_rows == 0:
                    log.warning("Skipping empty input shard at rank=%d chunk=%d.", rank, chunk_idx)
                    continue
                processed_any = True
                # [n_rows, ...] -> views of --batch rows (no copy); the last one may be shorter
                batch_splits = {k: torch.split(v, batch_rows) for k, v in chunk_data_cpu.items()}
                n_batches = len(batch_splits["label_id"])
                for bidx in range(n_batches):
                    batch_data_cpu = {k: v[bidx] for k, v in batch_splits.items()}
                    pred = _launch(model, batch_data_cpu, device, dwell_bq_channels, output_id, fit, streams, centre)
                    # the ids of the same rows (numpy views of the CPU tensors)
                    label_ids = batch_data_cpu["label_id"].numpy()
                    read_ids = batch_data_cpu["read_id"].numpy()
                    last = bidx == n_batches - 1
                    nonfinite_rows = batch_data_cpu[NONFINITE_KEY].numpy() if NONFINITE_KEY in batch_data_cpu else None
                    _queue_result(
                        in_flight, (chunk_idx, last, pred, label_ids, read_ids), compute_stream, nonfinite_rows
                    )
                    while len(in_flight) > MAX_IN_FLIGHT:
                        _collect_result(in_flight.popleft(), shard_parts, writer, args_dict["output"], rank)
            while in_flight:
                _collect_result(in_flight.popleft(), shard_parts, writer, args_dict["output"], rank)
    except BaseException as exc:
        # After an error, the shard being collected is not written: neither its finished batches (in shard_parts)
        # nor the batches still in flight. The shards already handed to the writer are written by close() below,
        # unless the writer itself has failed; --resume redoes the rest. The loop's own error is the one raised; a
        # writer failure at the same time is only logged. The error is logged first: a Ctrl-C during close() would
        # replace it, and after a Ctrl-C at the terminal the parent process never shows this rank's error.
        if writer.stopped():  # a write failed (usually the error raised here): no file still queued is written
            log.error(
                "Rank %d stopped on %r; the output writer has stopped, so no more output files are written.",
                rank,
                exc,
            )
        else:
            log.error("Rank %d stopped on %r; the output files already queued are written first.", rank, exc)
        try:
            writer.close()
        except KeyboardInterrupt as interrupt:
            if isinstance(exc, Exception):
                interrupt.stopped_on = exc  # inference_worker names it in its error
            raise
        except Exception as close_exc:  # noqa: BLE001
            log.error("Writing the queued output files also failed: %r", close_exc)
        raise
    writer.close()
    _report_rank_end(rank, processed_any, unreadable)
    return None
