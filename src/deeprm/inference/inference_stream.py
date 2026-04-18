"""
DeepRM Streaming Inference Module

This module orchestrates preprocessing and inference in a single streaming
pipeline. Instead of writing preprocessed data to disk and reading it back,
data flows from preprocessing workers to inference workers via an in-memory
multiprocessing Queue.

Usage:
    deeprm call stream --pod5 <pod5_dir> --bam <bam_file> --output <output_dir>

Optional ``--tee`` flag writes preprocessed data to disk alongside streaming,
useful for debugging or for later re-use with ``deeprm call run``.
"""

from __future__ import annotations

import argparse
import gc
import glob
import multiprocessing as mp
import os
import pathlib
import traceback

import numpy as np
import pandas as pd

from deeprm.inference.inference import _normalize_gpu_config, run_inference_stream
from deeprm.inference.inference_preprocess_python import (
    df_to_arrays,
    get_norm_factor,
    parse_bam,
    segment_normalize_signal,
    wait_for_processes,
)
from deeprm.inference.pileup_deeprm import main as pileup_main
from deeprm.utils import check_deps
from deeprm.utils.logging import get_logger
from deeprm.utils.utils import maybe_index_bam

log = get_logger(__name__)
check_deps.check_torch_available()

import torch  # noqa: E402


def add_arguments(parser: argparse.ArgumentParser):
    """Add CLI arguments for the streaming inference pipeline.

    Combines preprocessing and inference arguments, plus streaming-specific flags.
    """
    num_cpu = os.cpu_count()

    # -- Preprocessing arguments --
    parser.add_argument("--pod5", "-p", type=str, required=True, help="POD5 input directory or file")
    parser.add_argument("--bam", "-b", type=str, required=True, help="Dorado BAM file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--thread", "-t", type=int, default=max(1, int(num_cpu * 0.95)), help="Number of threads")
    parser.add_argument("--qcut", "-q", type=int, default=0, help="BQ cutoff")
    parser.add_argument("--chunk", "-k", type=int, default=16000, help="Preprocessing chunk size")
    parser.add_argument("--max-token-len", "-z", type=int, default=200, help="Maximum token length")
    parser.add_argument("--sampling", type=int, default=6, help="Sampling rate")
    parser.add_argument("--boi", "-y", type=str, default="A", help="Base of interest")
    parser.add_argument("--kmer-len", "-e", type=int, default=5, help="k-mer length")
    parser.add_argument("--cb-len", "-l", type=int, default=21, help="Context block length")
    parser.add_argument("--bam-thread", "-a", type=int, default=4, help="BAM decompression threads per process")
    parser.add_argument("--process-once", "-n", type=int, default=1000, help="Reads per processing batch")
    parser.add_argument("--dwell-shift", type=int, default=10, help="Distance between motor and pore")
    parser.add_argument("--sig-window", type=int, default=5, help="Signal window size")
    parser.add_argument("--label-div", "-d", type=int, default=10**9, help="Label division factor")
    parser.add_argument("--filter-flag", "-g", type=int, default=276, help="(Not used, for compatibility)")

    # -- Inference arguments --
    parser.add_argument("--model", "-m", type=str, default=None, help="Model path (.pt)")
    parser.add_argument("--model-type", type=str, default="deeprm_model", help="Model type")
    parser.add_argument("--batch", "-s", type=int, default=16000, help="Inference batch size")
    parser.add_argument("--gpu", type=int, default=None, help="Number of GPU devices", dest="num_gpu")
    parser.add_argument("--prefetch", type=int, default=4, help="Number of files to prefetch")
    parser.add_argument("--worker", "-w", type=int, default=4, help="Number of workers per GPU")
    parser.add_argument("--postfix", "-x", type=str, default="", help="Postfix for output directory")
    parser.add_argument(
        "--flush", "-f", type=int, default=100, help="Minibatches to accumulate before CPU-side flush"
    )
    parser.add_argument("--gpu-pool", "-gp", type=int, nargs="+", help="GPU pool")
    parser.add_argument("--output-id", "-id", type=int, default=None, help="Output ID for multi-output models")
    parser.add_argument("--threshold", "-th", type=float, default=0.98, help="Positive threshold")
    parser.add_argument("--epsilon", "-ep", type=float, default=1e-30, help="Epsilon value")
    parser.add_argument("--slice", "-sl", type=int, default=None, help="Slice index (for 2D predictions)")
    parser.add_argument("--flip", "-fl", action="store_true", help="Flip label")
    parser.add_argument("--skip-modbam", "-sm", action="store_true", help="Skip modBAM writing")
    parser.add_argument("--annot", type=str, default=None, help="Annotation file (e.g., refFlat.txt)")

    # -- Streaming-specific arguments --
    parser.add_argument("--tee", action="store_true", help="Also write preprocessed data to disk")
    parser.add_argument("--queue-size", type=int, default=8, help="Max chunks buffered in the streaming queue")

    return None


def _make_stream_sink(data_queue, tee_output_path=None):
    """Create a chunk sink that puts data on the queue and optionally writes to disk.

    Args:
        data_queue (multiprocessing.Queue): Queue for streaming chunks to inference.
        tee_output_path (str or None): If set, also write .npz files to this directory.

    Returns:
        callable: A function with signature ``sink(save_path, df)`` that can replace
            ``save_npz`` in ``segment_normalize_signal``.
    """

    def sink(save_path, df):
        arrays = df_to_arrays(df)
        if tee_output_path is not None:
            disk_path = os.path.join(tee_output_path, os.path.basename(save_path))
            np.savez_compressed(disk_path, **arrays)
        # collate_fn expects "segment_len"; on-disk format uses "segment_len_arr"
        arrays["segment_len"] = arrays.pop("segment_len_arr")
        data_queue.put(arrays)

    return sink


def _streaming_preprocess_worker(
    pid,
    bam_df,
    pod5_paths,
    norm_factor,
    token_output_path,
    data_queue,
    tee_output_path,
    cb_len,
    kmer_len,
    chunk_size,
    max_token_len,
    sampling,
    dwell_shift,
    sig_window,
    process_once,
    label_div,
):
    """Wrapper that runs segment_normalize_signal with a streaming chunk_sink.

    On error, puts an error marker on the queue before re-raising so that the
    inference consumer can detect the failure.
    """
    sink = _make_stream_sink(data_queue, tee_output_path)
    try:
        segment_normalize_signal(
            bam_df,
            pod5_paths,
            norm_factor,
            pid,
            token_output_path,
            cb_len=cb_len,
            kmer_len=kmer_len,
            chunk_size=chunk_size,
            max_token_len=max_token_len,
            sampling=sampling,
            dwell_shift=dwell_shift,
            sig_window=sig_window,
            process_once=process_once,
            label_div=label_div,
            chunk_sink=sink,
        )
    except Exception:
        data_queue.put(("ERROR", pid, traceback.format_exc()))
        raise
    finally:
        data_queue.put(None)


def main(args: argparse.Namespace):
    """Orchestrate streaming preprocessing + inference + pileup.

    Steps:
        1. Validate arguments and resolve defaults.
        2. Parse BAM data (multiprocessing).
        3. Create streaming queue and spawn preprocessing workers.
        4. Spawn inference workers that consume from the queue.
        5. Wait for all workers, then run pileup on predictions.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    # -- Validate inputs --
    if not os.path.exists(args.pod5):
        raise FileNotFoundError(f"Input directory {args.pod5} does not exist")
    if not os.path.exists(args.bam):
        raise FileNotFoundError(f"BAM file {args.bam} does not exist")
    if args.thread < 1:
        raise ValueError("--thread must be >= 1")
    if args.bam_thread < 1:
        raise ValueError("--bam-thread must be >= 1")
    if args.sampling < 1:
        raise ValueError("--sampling must be >= 1")
    if args.process_once < 1:
        raise ValueError("--process-once must be >= 1")
    if args.batch <= 0:
        raise ValueError("--batch must be a positive integer.")
    if args.flush <= 0:
        raise ValueError("--flush must be a positive integer.")
    if args.queue_size < 1:
        raise ValueError("--queue-size must be >= 1")

    # -- Resolve model path --
    if args.model is None:
        deeprm_root = pathlib.Path(__file__).parent.parent.resolve()
        args.model = os.path.join(deeprm_root, "weight", "deeprm_weights.pt")
    if not args.model.endswith(".pt"):
        raise ValueError("Invalid model path. It should be a .pt file.")

    # -- Resolve GPU config --
    _normalize_gpu_config(args)

    # -- Resolve BAM index --
    args.bam = maybe_index_bam(args.bam, args.thread)

    # -- Set up output directories --
    os.makedirs(args.output, exist_ok=True)
    inference_output = os.path.join(args.output, "molecule-level")
    pileup_output = os.path.join(args.output, "site-level")
    os.makedirs(inference_output, exist_ok=True)
    os.makedirs(pileup_output, exist_ok=True)

    tee_output_path = None
    if args.tee:
        tee_output_path = os.path.join(args.output, "preprocessed")
        os.makedirs(tee_output_path, exist_ok=True)
        log.info(f"Tee mode: preprocessed data will also be written to {tee_output_path}")

    # -- BAM parsing (same as inference_preprocess_python.main) --
    log.info("Started DeepRM Streaming Pipeline")
    log.info("Phase 1/3: Parsing BAM file")
    norm_factor = get_norm_factor()

    manager = mp.Manager()
    bam_df_list = manager.list()
    n_bam_procs = max(1, args.thread // args.bam_thread)
    proc_list = []
    for pid in range(n_bam_procs):
        proc = mp.Process(
            target=parse_bam,
            args=(pid, n_bam_procs, args.bam_thread, bam_df_list, args.bam, args.qcut, args.boi, args.sampling),
        )
        proc_list.append(proc)
        proc.start()
    wait_for_processes(proc_list, "BAM parsing")

    bam_frames = list(bam_df_list)
    if len(bam_frames) == 0:
        raise RuntimeError("BAM parsing produced no worker outputs")
    bam_df = pd.concat(bam_frames, ignore_index=True)
    bam_df.set_index("parent_id", inplace=True)
    manager.shutdown()
    del bam_frames
    gc.collect()

    # -- Discover POD5 files --
    mp.set_start_method("fork", force=True)

    if os.path.isfile(args.pod5):
        pod5_paths_split = [[args.pod5]]
    else:
        pod5_file_list = glob.glob(os.path.join(args.pod5, "*.pod5"))
        if len(pod5_file_list) == 0:
            raise FileNotFoundError(f"No POD5 files found in directory: {args.pod5}")
        pod5_paths_split = np.array_split(pod5_file_list, min(args.thread, len(pod5_file_list)))

    num_producers = len(pod5_paths_split)

    # -- Create streaming queue --
    queue_size = max(4, args.queue_size)
    data_queue = mp.Queue(maxsize=queue_size)
    log.info(f"Streaming queue created (maxsize={queue_size}, producers={num_producers})")

    # -- Phase 2: Spawn preprocessing workers --
    log.info("Phase 2/3: Starting preprocessing + inference")
    # A dummy output path for naming purposes (used in outpath generation inside
    # segment_normalize_signal even though disk writes only happen if tee is set).
    token_output_path = tee_output_path if tee_output_path else os.path.join(args.output, "_stream_tmp")

    preproc_procs = []
    for pid, pod5_paths in enumerate(pod5_paths_split):
        proc = mp.Process(
            target=_streaming_preprocess_worker,
            args=(
                pid,
                bam_df,
                pod5_paths,
                norm_factor,
                token_output_path,
                data_queue,
                tee_output_path,
                args.cb_len,
                args.kmer_len,
                args.chunk,
                args.max_token_len,
                args.sampling,
                args.dwell_shift,
                args.sig_window,
                args.process_once,
                args.label_div,
            ),
        )
        preproc_procs.append(proc)
        proc.start()

    gc.collect()

    # -- Phase 2 (cont.): Run inference concurrently --
    # Set the output to the molecule-level directory for inference predictions.
    args.output = inference_output
    # Set resume=False for streaming (no resume support).
    args.resume = False

    try:
        run_inference_stream(args, data_queue, num_producers)
    except Exception:
        log.error("Inference failed, terminating preprocessing workers.")
        for proc in preproc_procs:
            if proc.is_alive():
                proc.terminate()
        raise
    finally:
        # Wait for all preprocessing workers to finish.
        for proc in preproc_procs:
            proc.join(timeout=30)

    # Check for preprocessing failures.
    failed = [(proc.pid, proc.exitcode) for proc in preproc_procs if proc.exitcode != 0]
    if failed:
        failed_str = ", ".join([f"pid={pid}, exitcode={ec}" for pid, ec in failed])
        raise RuntimeError(f"Preprocessing failed in subprocess(es): {failed_str}")

    log.info("Inference complete.")

    # -- Phase 3: Pileup --
    log.info("Phase 3/3: Running pileup")
    args.input = inference_output
    args.output = pileup_output
    pileup_main(args)

    log.info("DeepRM Streaming Pipeline Finished.")
    return None
