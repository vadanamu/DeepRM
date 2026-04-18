"""
DeepRM Inference Module

This program handles the inference process for DeepRM models, including loading the
model, processing input data, and saving the output predictions.
"""

import argparse
import glob
import importlib
import os
import pathlib
from contextlib import nullcontext

import numpy as np
import tqdm

from deeprm.inference import inference_preprocess_python as stream_prep
from deeprm.inference.inference_dataloader import load_dataset
from deeprm.inference.pileup_deeprm import main as pileup_main
from deeprm.utils import check_deps
from deeprm.utils.logging import get_logger

log = get_logger(__name__)
check_deps.check_torch_available()

import torch
import torch.multiprocessing as mp
from torch.amp import autocast


def add_arguments(parser: argparse.ArgumentParser):
    """Adds command-line arguments.

    Args:
        parser (argparse.ArgumentParser): Argument parser to which arguments will be added.

    Returns:
        None
    """
    parser.add_argument("--input", "-i", dest="data", type=str, default=None, help="Preprocessed data path")
    parser.add_argument(
        "--preprocess-mode",
        choices=["disk", "stream"],
        default="disk",
        help="disk: read from --input directory, stream: preprocess on-the-fly from --pod5.",
    )
    parser.add_argument("--pod5", type=str, default=None, help="POD5 path for --preprocess-mode stream")
    parser.add_argument(
        "--stream-save-dir",
        type=str,
        default=None,
        help="Optional output directory to also persist streamed preprocessed chunks as .npz",
    )
    # Streaming preprocess knobs (mirrors preprocess CLI).
    parser.add_argument(
        "--prep-thread",
        type=int,
        default=max(1, int((os.cpu_count() or 1) * 0.95)),
        help="Prep threads",
    )
    parser.add_argument("--prep-qcut", type=int, default=0, help="Prep BQ cutoff")
    parser.add_argument("--prep-chunk", type=int, default=16000, help="Prep chunk size")
    parser.add_argument("--prep-max-token-len", type=int, default=200, help="Prep max token length")
    parser.add_argument("--prep-sampling", type=int, default=6, help="Prep sampling rate")
    parser.add_argument("--prep-boi", type=str, default="A", help="Prep base of interest")
    parser.add_argument("--prep-kmer-len", type=int, default=5, help="Prep k-mer length")
    parser.add_argument("--prep-cb-len", type=int, default=21, help="Prep context block length")
    parser.add_argument("--prep-bam-thread", type=int, default=4, help="Prep BAM thread per process")
    parser.add_argument("--prep-process-once", type=int, default=1000, help="Prep reads per processing batch")
    parser.add_argument("--prep-dwell-shift", type=int, default=10, help="Prep dwell shift")
    parser.add_argument("--prep-sig-window", type=int, default=5, help="Prep signal window")
    parser.add_argument("--prep-label-div", type=int, default=10**9, help="Prep label divisor")
    parser.add_argument("--bam", "-b", type=str, required=True, help="BAM file path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output path")
    parser.add_argument("--model", "-m", type=str, default=None, help="Model path")
    parser.add_argument("--model-type", "-y", type=str, default="deeprm_model", help="Model type")
    parser.add_argument("--batch", "-s", type=int, default=16000, help="Batch size")
    parser.add_argument("--gpu", "-g", type=int, default=None, help="Num. of GPU devices", dest="num_gpu")
    parser.add_argument("--prefetch", "-p", type=int, default=4, help="Number of files to prefetch")
    parser.add_argument("--worker", "-w", type=int, default=4, help="Number of workers per GPU")
    parser.add_argument("--postfix", "-x", type=str, default="", help="Postfix for output directory")
    parser.add_argument(
        "--flush", "-f", type=int, default=100, help="Number of minibatches to accumulate before CPU-side flush."
    )
    parser.add_argument("--resume", action="store_true", help="Resume terminated inference.")
    parser.add_argument("--gpu-pool", "-gp", type=int, nargs="+", help="GPU pool")
    parser.add_argument("--output-id", "-id", type=int, default=None, help="Output ID for multi-output models.")
    parser.add_argument("--thread", "-t", type=int, default=None, help="Number of threads to use for pileup")
    parser.add_argument("--threshold", "-th", type=float, default=0.98, help="Positive threshold")
    parser.add_argument("--epsilon", "-ep", type=float, default=1e-30, help="Epsilon value")
    parser.add_argument("--slice", "-sl", type=int, default=None, help="Slice index (for 2D predictions)")
    parser.add_argument("--flip", "-fl", action="store_true", help="Flip label")
    parser.add_argument("--skip-modbam", "-sm", action="store_true", help="Skip modBAM writing and only output BED")
    parser.add_argument(
        "--label_div", "-d", type=int, default=10**9, help="Divisor for label_id to separate transcript and position"
    )
    parser.add_argument("--annot", "-a", type=str, default=None, help="Annotation file (e.g., refFlat.txt)")
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
    if args.preprocess_mode == "disk":
        if not args.data:
            raise ValueError("--input is required when --preprocess-mode=disk.")
        if args.data.endswith("/"):
            args.data = args.data[:-1]
        if not os.path.isdir(args.data):
            raise ValueError("Invalid data path. It should be a directory containing data files.")
    else:
        if not args.pod5:
            raise ValueError("--pod5 is required when --preprocess-mode=stream.")
        if args.resume:
            raise ValueError("--resume is only supported when --preprocess-mode=disk.")


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
    elif args.num_gpu > 0:
        if args.num_gpu > available_gpu:
            raise ValueError(f"Requested {args.num_gpu} GPUs, but only {available_gpu} are available.")
        args.gpu_pool = list(range(args.num_gpu))
    else:
        args.num_gpu = 0
        args.gpu_pool = []

    if args.num_gpu == 0:
        args.gpu_pool = []


def main(args: argparse.Namespace):
    """Main function to run the evaluation pipeline.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None

    Notes:
        1. Parse command-line arguments.
        2. Create necessary directories.
        3. Run inference.
    """
    if args.model is None:
        ## Get directory of the current file
        deeprm_root = pathlib.Path(__file__).parent.parent.resolve()
        args.model = os.path.join(deeprm_root, "weight", "deeprm_weights.pt")
    if not args.model.endswith(".pt"):
        raise ValueError("Invalid model path. It should be a .pt file.")
    _validate_args(args)
    _normalize_gpu_config(args)

    dataset_name = os.path.basename(args.data) if args.preprocess_mode == "disk" else "streamed-preprocess"
    output = f"{args.output}/{dataset_name}"
    if len(args.postfix) > 0:
        output = f"{output}-{args.postfix}"
    args.output = output

    inference_output = os.path.join(args.output, "molecule-level")
    pileup_output = os.path.join(args.output, "site-level")
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(inference_output, exist_ok=True)
    os.makedirs(pileup_output, exist_ok=True)

    args.output = inference_output
    run_inference(args)
    log.info("Inference Program Finished.")

    args.input = inference_output
    args.output = pileup_output
    pileup_main(args)
    log.info("Pileup Program Finished.")
    return None


def run_inference(args):
    """Runs the inference process.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    torch.multiprocessing.set_sharing_strategy("file_system")
    log.info("Inference Program Started.")
    if args.num_gpu > 0:
        log.info(f"Using {args.num_gpu} GPUs: {args.gpu_pool}.")
    else:
        log.info("Using CPU.")

    ## make tensorboard directory
    log.info(f"Model path: {args.model}")
    log.info(f"Output directory: {args.output}")

    if args.preprocess_mode == "stream":
        run_inference_stream(args)
        return None

    if args.num_gpu > 0:
        mp.spawn(inference_worker, nprocs=args.num_gpu, args=(vars(args),), join=True)
    else:
        inference_worker(0, vars(args))
    return None


def _run_chunk_inference(args_dict, rank, device, model, chunk_idx, chunk_data_cpu):
    n_rows = int(chunk_data_cpu["label_id"].shape[0])
    if n_rows == 0:
        log.warning("Skipping empty input shard at rank=%d chunk=%d.", rank, chunk_idx)
        return False

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
        batch_data_gpu = to_device(batch_data_cpu, device)
        pred = model(*batch_data_gpu)
        if args_dict["output_id"] is not None:
            pred = pred[args_dict["output_id"]]
        pred_buffer.append(pred.detach().cpu().numpy())
        label_id_buffer.append(batch_data_cpu["label_id"].detach().cpu().numpy())
        read_id_buffer.append(batch_data_cpu["read_id"].detach().cpu().numpy())

        if len(pred_buffer) >= args_dict["flush"]:
            _flush_cpu_buffers(pred_buffer, label_id_buffer, read_id_buffer, pred_parts, label_parts, read_parts)

    _flush_cpu_buffers(pred_buffer, label_id_buffer, read_id_buffer, pred_parts, label_parts, read_parts)
    if not pred_parts:
        log.warning("No predictions were produced for rank=%d chunk=%d.", rank, chunk_idx)
        return False

    preds = pred_parts[0] if len(pred_parts) == 1 else np.concatenate(pred_parts, axis=0)
    label_ids = label_parts[0] if len(label_parts) == 1 else np.concatenate(label_parts, axis=0)
    read_ids = read_parts[0] if len(read_parts) == 1 else np.concatenate(read_parts, axis=0)

    out_path = f"{args_dict['output']}/inference_{rank}_{chunk_idx}.npz"
    np.savez_compressed(out_path, label_id=label_ids, read_id=read_ids, pred=preds)
    return True


def _init_model_for_device(args_dict, rank):
    use_gpu = args_dict["num_gpu"] > 0
    device = torch.device(f"cuda:{args_dict['gpu_pool'][rank]}") if use_gpu else torch.device("cpu")
    if use_gpu:
        map_location = {"cuda:0": str(device)}
    else:
        map_location = "cpu"
    save_dict = torch.load(args_dict["model"], map_location=map_location, weights_only=False)
    model_config = save_dict["model_config"]
    if args_dict["model_type"] is not None:
        model_config["model"] = args_dict["model_type"]

    dwell_bq_dim = 3
    TransformerModel = importlib.import_module(f"deeprm.model.{model_config['model']}").TransformerModel
    model = TransformerModel(
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
        dwell_bq_dim=dwell_bq_dim,
    )
    model.load_state_dict(state_dict=save_dict["model_state_dict"], strict=False)
    save_dict.clear()
    model.to(device)
    model.eval()
    return model, model_config, device


def run_inference_stream(args):
    """Run on-the-fly preprocessing + inference without intermediate disk writes."""
    if args.num_gpu > 1:
        raise ValueError("Stream mode currently supports CPU or a single GPU. Use --gpu 1 or --gpu-pool <one id>.")

    args_dict = vars(args)
    model, _, device = _init_model_for_device(args_dict, rank=0)

    prep_ns = argparse.Namespace(
        pod5=args.pod5,
        bam=args.bam,
        output=args.stream_save_dir or args.output,
        thread=args.prep_thread,
        qcut=args.prep_qcut,
        chunk=args.prep_chunk,
        max_token_len=args.prep_max_token_len,
        sampling=args.prep_sampling,
        boi=args.prep_boi,
        kmer_len=args.prep_kmer_len,
        cb_len=args.prep_cb_len,
        bam_thread=args.prep_bam_thread,
        process_once=args.prep_process_once,
        dwell_shift=args.prep_dwell_shift,
        sig_window=args.prep_sig_window,
        label_div=args.prep_label_div,
        filter_flag=276,
    )

    chunk_idx = 0
    processed_any = False

    def _emit_chunk(chunk_arrays):
        nonlocal chunk_idx, processed_any
        chunk_data_cpu = {
            "read_id": torch.as_tensor(chunk_arrays["read_id"]),
            "label_id": torch.as_tensor(chunk_arrays["label_id"]),
            "segment_len": torch.as_tensor(chunk_arrays["segment_len_arr"], dtype=torch.int32),
            "signal_token": torch.as_tensor(chunk_arrays["signal_token"], dtype=torch.float32),
            "kmer_token": torch.as_tensor(chunk_arrays["kmer_token"], dtype=torch.int32),
            "dwell_bq_token": torch.as_tensor(
                np.stack(
                    (chunk_arrays["dwell_motor_token"], chunk_arrays["dwell_pore_token"], chunk_arrays["bq_token"]),
                    axis=-1,
                ),
                dtype=torch.float32,
            ),
        }
        processed_any = _run_chunk_inference(args_dict, 0, device, model, chunk_idx, chunk_data_cpu) or processed_any
        chunk_idx += 1

    amp_ctx = autocast(enabled=device.type == "cuda", cache_enabled=device.type == "cuda", device_type="cuda")
    if device.type != "cuda":
        amp_ctx = nullcontext()

    with amp_ctx:
        with torch.no_grad():
            stream_prep.stream(
                prep_ns,
                emit_chunk=_emit_chunk,
                write_to_disk=args.stream_save_dir is not None,
                disk_output_path=args.stream_save_dir,
            )

    if not processed_any:
        log.warning("No streamed chunks were processed. No inference outputs were written.")
    return None


def _discover_resume_point(output_dir: str, rank: int) -> int:
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
    """Worker function for running inference on a single GPU.

    Args:
        rank (int): Rank of the current process.
        args_dict (dict): Dictionary of command-line arguments.

    Returns:
        None
    """
    model, model_config, device = _init_model_for_device(args_dict, rank)

    if rank == 0:
        total_params = sum(parameter.numel() for parameter in model.parameters())
        log.info("Model parameters: %s", f"{total_params:,}")

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
    )

    inference_loop(args_dict, rank, device, model, data_loader, start_index=resume_from)
    return None


def to_device(data, device):
    """Transfers data to the specified GPU device using a non-blocking stream.

    Args:
        data (dict): Dictionary containing the data to be transferred.
        device (torch.device): The target GPU device.

    Returns:
        tuple: A tuple containing the transferred data tensors (src_kmer, src_signal, src_seg_len, src_dwell_bq).
    """
    non_blocking = device.type == "cuda"
    return (
        data["kmer_token"].to(device, non_blocking=non_blocking),
        data["signal_token"].to(device, non_blocking=non_blocking),
        data["segment_len"].to(device, non_blocking=non_blocking),
        data["dwell_bq_token"].to(device, non_blocking=non_blocking),
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


def inference_loop(args_dict, rank, device, model, data_loader, start_index=0):
    """Runs the inference loop for the given model and data loader.

    Args:
        args_dict (dict): Dictionary of command-line arguments.
        rank (int): Rank of the current process.
        gpu_id (int): ID of the GPU to use.
        model (torch.nn.Module): The model to run inference on.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.

    Returns:
        None
    """
    amp_ctx = autocast(enabled=device.type == "cuda", cache_enabled=device.type == "cuda", device_type="cuda")
    if device.type != "cuda":
        amp_ctx = nullcontext()

    with amp_ctx:
        with torch.no_grad():
            processed_any = False
            for chunk_idx, chunk_data_cpu in enumerate(
                tqdm.tqdm(data_loader, total=len(data_loader), smoothing=0), start=start_index
            ):
                if chunk_data_cpu is None:
                    continue
                n_rows = int(chunk_data_cpu["label_id"].shape[0])
                if n_rows == 0:
                    log.warning("Skipping empty input shard at rank=%d chunk=%d.", rank, chunk_idx)
                    continue

                processed_any = _run_chunk_inference(args_dict, rank, device, model, chunk_idx, chunk_data_cpu) or processed_any

            if not processed_any:
                log.warning("No input shards were processed for rank=%d. No inference outputs were written.", rank)

    return None
