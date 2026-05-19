"""
DeepRM Inference Preprocessing Module

This script segments and normalizes raw signal data from POD5 files and corresponding
BAM alignments. It extracts dwell times, context blocks, and signal windows,
then tokenizes and saves them as compressed .npz chunks for downstream modeling.

Key steps:
1. Parse POD5 and BAM to collect signal and alignment metadata.
2. Compute dwell-time and normalize signal windows.
3. Segment signals based on move, and then format into fixed-length blocks.
4. Save processed tokens in chunks for model input.
"""

import argparse
import gc
import glob
import multiprocessing as mp
import os
import uuid

import numpy as np
import pandas as pd
import pod5
import pysam
import tqdm

from deeprm.utils.logging import get_logger
from deeprm.utils.utils import maybe_index_bam

log = get_logger(__name__)

# Complement used to make the base-of-interest filter strand-aware. For a
# reverse-strand read the forward-reference base at a base-of-interest site is
# the complement of `boi` (e.g. an m6A 'A' shows up as 'T' on the forward ref).
_BASE_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}

# ASCII complement table for reverse-complementing the (uint8) query_sequence
# of reverse-strand reads back into RNA-sense orientation. Non-ACGT bytes
# (e.g. 'N') map to themselves.
_ASCII_COMPLEMENT = np.arange(256, dtype=np.uint8)
for _a, _b in [(65, 84), (84, 65), (67, 71), (71, 67), (97, 116), (116, 97), (99, 103), (103, 99)]:
    _ASCII_COMPLEMENT[_a] = _b


def add_arguments(parser: argparse.ArgumentParser):
    """
    Adds command-line arguments.
    Args:
        parser (argparse.ArgumentParser): Argument parser to which arguments will be added.
    Returns:
        None
    """
    num_cpu = os.cpu_count()
    parser.add_argument("--pod5", "-p", type=str, required=True, help="POD5 Input directory")
    parser.add_argument("--bam", "-b", type=str, required=True, help="Dorado BAM file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--thread", "-t", type=int, default=max(1, int(num_cpu * 0.95)), help="Number of thread to use")
    parser.add_argument("--qcut", "-q", type=int, default=0, help="BQ cutoff")
    parser.add_argument("--chunk", "-k", type=int, default=16000, help="Chunk size")
    parser.add_argument("--max-token-len", "-z", type=int, default=200, help="Maximum token length")
    parser.add_argument("--sampling", "-s", type=int, default=6, help="Sampling rate")
    parser.add_argument("--boi", "-y", type=str, default="A", help="Base of interest")
    parser.add_argument("--kmer-len", "-e", type=int, default=5, help="k-mer length")
    parser.add_argument("--cb-len", "-l", type=int, default=21, help="Context block length")
    parser.add_argument("--bam-thread", "-a", type=int, default=4, help="BAM decompression thread per process")
    parser.add_argument("--process-once", "-n", type=int, default=1000, help="Reads per processing batch")
    parser.add_argument("--dwell-shift", "-f", type=int, default=10, help="Distance between motor and pore")
    parser.add_argument("--sig-window", "-w", type=int, default=5, help="Signal window size")
    parser.add_argument("--label-div", "-d", type=int, default=10**9, help="Label division factor")
    parser.add_argument(
        "--filter-flag",
        "-g",
        type=int,
        default=276,
        help="BAM flag bits to exclude (read skipped if read.flag & filter-flag). "
        "Use 276 (unmapped+reverse+secondary) for transcriptome-mapped BAMs; "
        "260 (unmapped+secondary) for genome-mapped BAMs so minus-strand-gene "
        "reads are kept and processed in RNA-sense. Honored by both paths.",
    )

    return None


def wait_for_processes(proc_list, stage_name):
    """
    Join a list of processes and raise if any subprocess failed.

    Args:
        proc_list (list[multiprocessing.Process]): child processes to join.
        stage_name (str): human-readable pipeline stage name.

    Returns:
        None
    """
    failed = []
    for proc in proc_list:
        proc.join()
        if proc.exitcode != 0:
            failed.append((proc.pid, proc.exitcode))

    if failed:
        failed_str = ", ".join([f"pid={pid}, exitcode={exitcode}" for pid, exitcode in failed])
        raise RuntimeError(f"{stage_name} failed in subprocess(es): {failed_str}")

    return None


def main(args: argparse.Namespace):
    """
    Run the full preprocessing pipeline with multiprocessing.

    Steps:
    1. Parse arguments and prepare output.
    2. Spawn processes to parse BAM data.
    3. Consolidate BAM DataFrame.
    4. Spawn processes to segment and normalize POD5 signals.
    5. Finalize and exit.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """

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

    args.bam = maybe_index_bam(args.bam, args.thread)

    os.makedirs(args.output, exist_ok=True)

    log.info("Started DeepRM Preprocessing")
    os.makedirs(args.output, exist_ok=True)
    norm_factor = get_norm_factor()

    manager = mp.Manager()
    bam_df = manager.list()
    n_bam_procs = max(1, args.thread // args.bam_thread)
    proc_list = []
    for pid in range(n_bam_procs):
        proc = mp.Process(
            target=parse_bam,
            args=(
                pid,
                n_bam_procs,
                args.bam_thread,
                bam_df,
                args.bam,
                args.qcut,
                args.boi,
                args.sampling,
                args.filter_flag,
            ),
        )
        proc_list.append(proc)
        proc.start()
    wait_for_processes(proc_list, "BAM parsing")

    bam_frames = list(bam_df)
    if len(bam_frames) == 0:
        raise RuntimeError("BAM parsing produced no worker outputs")
    bam_df = pd.concat(bam_frames, ignore_index=True)
    bam_df.set_index("parent_id", inplace=True)
    manager.shutdown()
    gc.collect()

    mp.set_start_method("fork", force=True)

    if os.path.isfile(args.pod5):
        pod5_paths_split = [[args.pod5]]

    else:
        pod5_file_list = glob.glob(os.path.join(args.pod5, "*.pod5"))
        if len(pod5_file_list) == 0:
            raise FileNotFoundError(f"No POD5 files found in directory: {args.pod5}")
        pod5_paths_split = np.array_split(pod5_file_list, min(args.thread, len(pod5_file_list)))

    proc_list = []
    for pid, pod5_paths in enumerate(pod5_paths_split):
        proc = mp.Process(
            target=segment_normalize_signal,
            args=(
                bam_df,
                pod5_paths,
                norm_factor,
                pid,
                args.output,
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
        proc_list.append(proc)
        proc.start()

    gc.collect()
    wait_for_processes(proc_list, "Signal segmentation/normalization")

    log.info("Finished DeepRM Preprocessing")
    return None


def get_norm_factor():
    """
    Return default normalization factors for signal and dwell scaling.

    Returns:
        dict: Keys 'quantile_a','quantile_b','shift_mult','scale_mult'.
    """
    norm_factor_default = {}
    norm_factor_default["quantile_a"] = 0.2
    norm_factor_default["quantile_b"] = 0.8
    norm_factor_default["shift_mult"] = 0.48
    norm_factor_default["scale_mult"] = 0.59
    return norm_factor_default


def mean_phred(phred):
    """
    Calculate the mean Phred quality score from an array of scores.
    Args:
        phred (numpy.ndarray): Array of Phred scores.
    Returns:
        float: Mean Phred quality score.
    """
    return -10 * np.log10(np.mean(10 ** (-phred / 10)))


def segmented_signal_to_block(signal_segmented, segment_len_arr, kmer, sampling, sig_window, pad_to):
    """
    Convert segmented signal into a fixed-length block for given k-mer context.

    Args:
        signal_segmented (list): list of signal segments around each base. (list of numpy.ndarray)
        segment_len_arr (numpy.ndarray): lengths of each segment in sampling units.
        kmer (int): length of k-mer context.
        sampling (int): samples per signal unit.
        sig_window (int): size of local signal window for padding calculation.
        pad_to (int): desired total length of output block in signal units.

    Returns:
        numpy.ndarray or None: concatenated, trimmed, and padded signal block, or None on failure.
    """
    try:
        kmer_pad = (kmer - 1) // 2
        lr_pad = (sig_window - 1) // 2

        l_skip_units = np.sum(segment_len_arr[:kmer_pad]) - lr_pad
        r_skip_units = np.sum(segment_len_arr[-kmer_pad:]) - lr_pad
        if l_skip_units < 0 or r_skip_units < 0:
            return None

        l_skip = int(l_skip_units * sampling)
        r_skip = int(r_skip_units * sampling)

        signal_segmented = np.concatenate(signal_segmented)
        if len(signal_segmented) % sampling != 0:
            return None
        if l_skip > len(signal_segmented):
            return None
        if r_skip > 0 and (l_skip + r_skip) > len(signal_segmented):
            return None

        if r_skip > 0:
            signal_segmented = signal_segmented[l_skip:-r_skip]
        else:
            signal_segmented = signal_segmented[l_skip:]

        if len(signal_segmented) == 0:
            return None

        padding = (pad_to + kmer - 1) * sampling - len(signal_segmented)
        if padding > 0:
            signal_segmented = np.pad(signal_segmented, (0, padding), mode="constant", constant_values=0)
    except Exception:
        return None
    return signal_segmented


def create_segment_len_arr(segment_arr, sampling):
    """
    Compute segment-length array in sampling units for each sub-segment.

    Args:
        segment_arr (list): raw signal segments (list of numpy.ndarray).
        sampling (int): samples per signal unit.

    Returns:
        numpy.ndarray: integer length array per segment after downsampling.
    """
    return np.array([len(x) for x in segment_arr], dtype=int) // sampling


def move_to_dwell(move, quantile_a, quantile_b, shift_mult, scale_mult, sampling=6):
    """
    Transform raw move array into scaled dwell-time tokens.

    Args:
        move (numpy.ndarray): boolean array indicating move events per sample.
        quantile_a (float): lower quantile for scaling.
        quantile_b (float): upper quantile for scaling.
        shift_mult (float): shift multiplier.
        scale_mult (float): scale multiplier.
        sampling (int): samples per signal unit.

    Returns:
        numpy.ndarray or None: standardized dwell-time values, or None for invalid input.

    Notes:
        1. Convert boolean moves into positions and compute deltas.
        2. Log-transform dwell durations.
        3. Scale and shift based on quantiles and multipliers.
    """
    move = np.asarray(move, dtype=bool)
    if move.ndim != 1 or move.size == 0:
        return None

    dwell = np.arange(1, len(move) + 1, dtype=np.int32)[np.flip(move)]
    dwell = np.concatenate([np.zeros(1, dtype=np.int32), dwell])
    dwell = dwell[1:] - dwell[:-1]
    if dwell.size == 0 or np.any(dwell <= 0):
        return None

    dwell = np.log10((dwell * sampling).astype(np.float32))
    if dwell.size == 0 or not np.all(np.isfinite(dwell)):
        return None

    quantile_a_value = np.quantile(dwell, quantile_a)
    quantile_b_value = np.quantile(dwell, quantile_b)
    q_shift = max(0.1, shift_mult * (quantile_a_value + quantile_b_value))
    q_scale = max(0.1, scale_mult * (quantile_b_value - quantile_a_value))
    dwell = (dwell - q_shift) / q_scale
    if not np.all(np.isfinite(dwell)):
        return None
    return dwell


def normalise_trim_segment_signal(signal, move, sp, ts, ns, quantile_a, quantile_b, shift_mult, scale_mult, sampling=6):
    """
    Normalize and segment raw signal based on trimming and dwell indices.

    Args:
        signal (numpy.ndarray): raw signal trace.
        move (numpy.ndarray): dwell-time tokens.
        sp (int): samples to skip at start.
        ts (int): trim start index.
        ns (int): trim end index (0 means till end).
        quantile_a (float): lower quantile threshold.
        quantile_b (float): upper quantile threshold.
        shift_mult (float): shift multiplier.
        scale_mult (float): scale multiplier.
        sampling (int): samples per signal unit.

    Returns:
        list or None: list of per-base signal segments or None on error.

    Notes:
        1. Trim start (sp) and segment window (ts:ns).
        2. Flip signal for reverse processing.
        3. Shift and scale signal by quantile multipliers.
        4. Split by dwell move indices to segment per-base.
    """
    try:
        signal = np.asarray(signal)
        move = np.asarray(move, dtype=bool)

        signal = signal[sp:]
        signal_len = len(signal)
        if ns == 0:
            ns = signal_len
        signal = signal[ts:ns]
        if len(signal) == 0:
            return None
        signal = np.flip(signal, axis=0)

        quantile_a_value = np.quantile(signal, quantile_a)
        quantile_b_value = np.quantile(signal, quantile_b)

        q_shift = max(10.0, shift_mult * (quantile_a_value + quantile_b_value))
        q_scale = max(1.0, scale_mult * (quantile_b_value - quantile_a_value))
        signal = (signal - q_shift) / q_scale
        if not np.all(np.isfinite(signal)):
            return None

        move_idx = np.where(move)[0][1:] * sampling
        move_idx = len(signal) - move_idx
        move_idx = np.flip(move_idx, axis=0)
        signal = np.array_split(signal, move_idx)
        if len(signal) == 0:
            return None
    except Exception:
        return None
    return signal


def parse_pod5(pod5_path, read_ids):
    """
    Read signals and calibration from a POD5 file and return as DataFrame.

    Args:
        pod5_path (str): file path to the POD5 file.
        read_ids (list): list of read IDs to extract signals for. (list of UUIDs)

    Returns:
        pandas.DataFrame: with columns ['signal', 'offset', 'scale'], indexed by 'read_id'.
    """
    signal_list = []
    offset_list = []
    scale_list = []
    id_list = []

    with pod5.Reader(pod5_path) as reader:
        for i, record in enumerate(reader.reads(read_ids)):
            try:
                signal_arr = record.signal
                offset = record.calibration.offset
                scale = record.calibration.scale
                rid = record.read_id
            except Exception:
                continue

            signal_list.append(signal_arr)
            offset_list.append(offset)
            scale_list.append(scale)
            id_list.append(rid)

    signal_df = pd.DataFrame(
        {"signal": signal_list, "parent_id": id_list, "offset": offset_list, "scale": scale_list}
    ).set_index("parent_id")
    del signal_list, offset_list, scale_list, id_list
    gc.collect()
    return signal_df


def parse_bam(pid, n_procs, n_thread, bam_data, bam_path, bq_cutoff, boi, expected_sampling, filter_flag=276):
    """
    Extract move tags and alignment information from a BAM file in parallel.

    Args:
        pid (int): process ID for sharding.
        n_procs (int): total number of processes.
        n_thread (int): thread for BAM reading.
        bam_data (list): multiprocessing.Manager list to collect DataFrames.
        bam_path (str): path to the BAM file.
        bq_cutoff (int): minimum average base quality threshold.
        boi (str): base-of-interest for alignment extraction.
        expected_sampling (int): expected move-tag stride in samples.
        filter_flag (int): BAM flag bits to exclude (read skipped if
            ``read.flag & filter_flag``). Same semantics as the C++ path.
            Use 276 (unmapped+reverse+secondary) for transcriptome-mapped
            BAMs, 260 (unmapped+secondary) for genome-mapped BAMs so minus-
            strand-gene reads are kept and processed in RNA-sense.

    Returns:
        None (appends DataFrame to bam_data).
    """
    bam_df = {k: [] for k in ["read_id", "parent_id", "ts", "ns", "sp", "bq", "mv", "seq", "ref", "ap", "strand"]}
    input_bam = pysam.AlignmentFile(bam_path, "rb", check_sq=False, threads=n_thread)
    ref_index_dict = {ref: i for i, ref in enumerate(input_bam.references)}
    mv_stride_mismatch = 0

    for read_idx, read in tqdm.tqdm(enumerate(input_bam), total=input_bam.mapped + input_bam.unmapped):
        if read_idx % n_procs != pid:
            continue
        if (read.flag & filter_flag) or (not read.has_tag("mv")):
            continue
        try:
            bq = np.array(read.query_qualities, dtype=np.int8)
            if mean_phred(bq) < bq_cutoff:
                continue
        except Exception:
            continue

        try:
            mv_tag = read.get_tag("mv")
        except Exception:
            continue
        if len(mv_tag) < 2:
            continue

        mv_sampling = int(mv_tag[0])
        if mv_sampling != expected_sampling:
            mv_stride_mismatch += 1
            continue

        boi_fwd = boi.upper()
        boi_strand = _BASE_COMPLEMENT.get(boi_fwd, boi_fwd) if read.is_reverse else boi_fwd
        ap = np.array(read.get_aligned_pairs(matches_only=True, with_seq=True), dtype=object)
        ap = ap[ap[:, 2] == boi_strand][:, :2].astype(np.int32)
        if len(ap) == 0:
            continue
        read_id = uuid.UUID(str(read.query_name))
        parent_id = uuid.UUID(str(read.get_tag("pi"))) if read.has_tag("pi") else read_id

        ts = read.get_tag("ts") if read.has_tag("ts") else 0
        ns = read.get_tag("ns") if read.has_tag("ns") else 0
        sp = read.get_tag("sp") if read.has_tag("sp") else 0
        mv = np.array(mv_tag[1:], dtype=bool)
        seq = np.array(list(read.query_sequence)).view(np.int32).astype(np.uint8)
        strand = -1 if read.is_reverse else 1
        bam_df["read_id"].append(read_id)
        bam_df["parent_id"].append(parent_id)
        bam_df["ts"].append(ts)
        bam_df["ns"].append(ns)
        bam_df["sp"].append(sp)
        bam_df["bq"].append(bq)
        bam_df["mv"].append(mv)
        bam_df["seq"].append(seq)
        bam_df["ref"].append(read.reference_name)
        bam_df["ap"].append(ap)
        bam_df["strand"].append(strand)

    input_bam.close()
    if mv_stride_mismatch > 0:
        log.warning(
            "Skipped %d reads in BAM worker %d due to mv-tag stride mismatch with --sampling=%d",
            mv_stride_mismatch,
            pid,
            expected_sampling,
        )
    bam_df = pd.DataFrame.from_dict(bam_df, orient="columns")
    bam_df["ref"] = bam_df["ref"].map(ref_index_dict)
    bam_df = bam_df.dropna()
    bam_df[["ts", "ns", "sp"]] = bam_df[["ts", "ns", "sp"]].astype(np.int32)
    bam_data.append(bam_df)
    gc.collect()
    return None


def segment_normalize_signal(
    bam_df,
    pod5_paths,
    norm_factor,
    pid,
    token_output_path,
    cb_len=21,
    kmer_len=5,
    chunk_size=10000,
    max_token_len=200,
    sampling=6,
    dwell_shift=10,
    sig_window=5,
    process_once=1000,
    label_div=10**9,
):
    """
    Segment and normalize signals per read, and save token chunks.

    Args:
        bam_df (pandas.DataFrame): alignment metadata indexed by parent_id.
        pod5_paths (list): list of POD5 file paths.
        norm_factor (dict): normalization parameters.
        pid (int): process ID for naming outputs.
        token_output_path (str): directory to write .npz chunks.
        cb_len (int): context block length.
        kmer_len (int): k-mer length for segmentation.
        chunk_size (int): number of tokens per output file.
        max_token_len (int): maximum length of token in signal units.
        sampling (int): samples per signal unit.
        dwell_shift (int): shift for dwell token alignment.
        sig_window (int): local signal window size.
        process_once (int): reads to process per batch.
        label_div (int): label division factor for unique ID generation.

    Returns:
        None (writes .npz files).
    """
    trim = kmer_len // 2
    shift_mult = norm_factor["shift_mult"]
    scale_mult = norm_factor["scale_mult"]
    quantile_a = norm_factor["quantile_a"]
    quantile_b = norm_factor["quantile_b"]
    cb_half_len = cb_len // 2
    buffer = []
    output_index = 0

    for pod5_path in tqdm.tqdm(pod5_paths):
        with pod5.Reader(pod5_path) as reader:
            read_ids = reader.read_ids

        if len(read_ids) == 0:
            continue

        read_id_split = np.array_split(np.array(read_ids), np.ceil(len(read_ids) / process_once))

        for read_ids in read_id_split:
            try:
                signal_df = parse_pod5(pod5_path, read_ids)
            except Exception:
                print(f"Corrupted POD5 file: {pod5_path}")
                continue
            if len(signal_df) == 0:
                continue
            valid_index = signal_df.index.intersection(bam_df.index)
            if len(valid_index) == 0:
                continue
            signal_df = pd.merge(signal_df, bam_df.loc[valid_index], left_index=True, right_index=True, how="inner")
            if len(signal_df) == 0:
                continue

            output_index += 1
            out_prefix = f"{token_output_path}/{pid}-{output_index}"
            signal_df["dwell_token"] = signal_df["mv"].apply(lambda x: move_to_dwell(x, 0.2, 0.8, 0.5, 1.5, sampling))
            ## Normalize and segment signal
            signal_df["signal"] = signal_df.apply(
                lambda x: normalise_trim_segment_signal(
                    x["signal"],
                    x["mv"],
                    x["sp"],
                    x["ts"],
                    x["ns"],
                    quantile_a,
                    quantile_b,
                    shift_mult,
                    scale_mult,
                    sampling,
                ),
                axis=1,
            )
            signal_df = signal_df[signal_df["signal"].notnull() & signal_df["dwell_token"].notnull()].copy()
            if len(signal_df) == 0:
                continue

            ## Present reverse-strand reads in RNA-sense orientation.
            #
            # The model is trained only on RNA-sense, base-of-interest-centred
            # context. `signal`/`dwell_token` are already in basecalled
            # (RNA-sense 5'->3') order for both strands. The BAM, however,
            # stores SEQ/QUAL of reverse-strand reads reverse-complemented into
            # forward-reference order, and get_aligned_pairs returns forward-
            # reference q_pos. So for reverse-strand reads bring seq/bq/q_pos
            # into RNA-sense: reverse-complement seq, reverse bq, and mirror
            # q_pos (q_pos is mirrored below, once q_len is known). signal and
            # dwell_token are left untouched (already RNA-sense).
            rev_mask = signal_df["strand"] == -1
            if rev_mask.any():
                signal_df.loc[rev_mask, "seq"] = signal_df.loc[rev_mask, "seq"].apply(
                    lambda s: _ASCII_COMPLEMENT[s][::-1]
                )
                signal_df.loc[rev_mask, "bq"] = signal_df.loc[rev_mask, "bq"].apply(lambda q: q[::-1])

            ## Explode read-level data to base-level data
            signal_df = (
                signal_df[["read_id", "strand", "bq", "seq", "signal", "dwell_token", "ref", "ap"]]
                .explode("ap")
                .reset_index(drop=True)
            )
            if len(signal_df) == 0:
                continue
            gc.collect()

            # derive positions and lengths
            signal_df["q_pos"] = signal_df["ap"].apply(lambda x: x[0])
            signal_df["r_pos"] = signal_df["ap"].apply(lambda x: x[1])
            signal_df["q_len"] = signal_df["seq"].apply(len)
            # Mirror q_pos for reverse-strand reads so it indexes the RNA-sense
            # (basecalled) coordinate system, matching the now sense-oriented
            # seq/bq and the already sense-ordered signal/dwell arrays.
            signal_df["q_pos"] = np.where(
                signal_df["strand"] == -1,
                signal_df["q_len"] - 1 - signal_df["q_pos"],
                signal_df["q_pos"],
            )
            signal_df["start_pos"] = signal_df["q_pos"] - cb_half_len
            signal_df["end_pos"] = signal_df["q_pos"] + cb_half_len + 1
            signal_df["label_id"] = (signal_df["ref"] * label_div + signal_df["r_pos"] + 1) * signal_df["strand"]

            # filter by context (strand-uniform: all arrays are RNA-sense)
            signal_df = signal_df[
                (signal_df["start_pos"] >= 0) & (signal_df["end_pos"] + dwell_shift - trim < signal_df["q_len"])
            ]
            if len(signal_df) == 0:
                continue
            # slice tokens
            signal_df["signal"] = signal_df.apply(lambda x: x["signal"][x["start_pos"] : x["end_pos"]], axis=1)
            # Motor leads the pore by `dwell_shift` bases; dwell_token is in
            # basecalled (RNA-sense) order for both strands, so the offset is
            # +dwell_shift uniformly.
            signal_df["dwell_motor_token"] = signal_df.apply(
                lambda x: x["dwell_token"][(x["start_pos"] + dwell_shift + trim) : (x["end_pos"] + dwell_shift - trim)],
                axis=1,
            )
            signal_df["dwell_pore_token"] = signal_df.apply(
                lambda x: x["dwell_token"][(x["start_pos"] + trim) : (x["end_pos"] - trim)], axis=1
            )
            signal_df["bq"] = signal_df.apply(lambda x: x["bq"][x["start_pos"] + trim : x["end_pos"] - trim], axis=1)
            signal_df["motif"] = signal_df.apply(lambda x: x["seq"][x["start_pos"] : x["end_pos"]], axis=1)
            signal_df["segment_len_arr"] = signal_df["signal"].apply(lambda x: create_segment_len_arr(x, sampling))
            signal_df["token_len"] = signal_df["segment_len_arr"].apply(lambda x: np.sum(x[trim:-trim]))
            signal_df = signal_df[
                (signal_df["segment_len_arr"].apply(lambda x: len(x) == cb_len))
                & (signal_df["token_len"] <= max_token_len)
                & (signal_df["token_len"] > 0)
            ]
            try:
                signal_df["signal"] = signal_df.apply(
                    lambda x: segmented_signal_to_block(
                        x["signal"], x["segment_len_arr"], kmer_len, sampling, sig_window, max_token_len
                    ),
                    axis=1,
                )
            except Exception as e:
                log.warning(f"{e}")
                log.warning("Signal Tokenization Error: - Skipping")
                continue
            signal_df = signal_df[signal_df["signal"].notnull()]
            if len(signal_df) == 0:
                continue
            signal_df["segment_len_arr"] = signal_df["segment_len_arr"].apply(lambda x: x[trim:-trim])
            signal_df = signal_df[
                [
                    "read_id",
                    "segment_len_arr",
                    "signal",
                    "motif",
                    "dwell_motor_token",
                    "dwell_pore_token",
                    "bq",
                    "label_id",
                ]
            ].copy()
            signal_df.rename(columns={"motif": "kmer_token", "signal": "signal_token", "bq": "bq_token"}, inplace=True)
            signal_df["segment_len_arr"] = signal_df["segment_len_arr"].apply(lambda x: x.astype(np.uint16))
            signal_df["signal_token"] = signal_df["signal_token"].apply(lambda x: x.astype(np.float32))
            signal_df["bq_token"] = signal_df["bq_token"].apply(lambda x: np.clip(x, 0, 60).astype(np.uint8))

            if len(buffer) > 0:
                signal_df = pd.concat([buffer, signal_df], ignore_index=True)
                buffer = []

            if len(signal_df) < chunk_size:
                buffer = signal_df

            else:
                for chunk_idx in range(0, len(signal_df) // chunk_size):
                    chunk = signal_df.iloc[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
                    outpath = f"{out_prefix}-{chunk_idx}.npz"
                    save_npz(outpath, chunk)

                chunk = signal_df.iloc[(len(signal_df) // chunk_size) * chunk_size :].copy()
                if len(chunk) > 0:
                    buffer = chunk
                del signal_df

            gc.collect()

    out_prefix = f"{token_output_path}/{pid}-last"
    if len(buffer) > 0:
        for chunk_idx in range(0, len(buffer) // chunk_size):
            chunk = buffer.iloc[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
            outpath = f"{out_prefix}-{chunk_idx}.npz"
            save_npz(outpath, chunk)
        chunk = buffer.iloc[(len(buffer) // chunk_size) * chunk_size :].copy()
        if len(chunk) > 0:
            outpath = f"{out_prefix}-last.npz"
            save_npz(outpath, chunk)
    return None


def save_npz(save_path, df):
    """
    Serialize token DataFrame to a compressed .npz file.

    Args:
        save_path (str): Output .npz file path.
        df (pandas.DataFrame): DataFrame with token columns.

    Returns:
        None
    """
    segment_len_arr = np.stack(df["segment_len_arr"].values)
    signal_token = np.stack(df["signal_token"].values)
    kmer_token = np.stack(df["kmer_token"].values)
    dwell_motor_token = np.stack(df["dwell_motor_token"].values)
    dwell_pore_token = np.stack(df["dwell_pore_token"].values)
    bq_token = np.stack(df["bq_token"].values)
    label_id = df["label_id"].values
    read_id = np.frombuffer(b"".join(u.bytes for u in df["read_id"].values), dtype=np.int64).reshape(-1, 2)
    np.savez_compressed(
        save_path,
        segment_len_arr=segment_len_arr,
        signal_token=signal_token,
        kmer_token=kmer_token,
        dwell_motor_token=dwell_motor_token,
        dwell_pore_token=dwell_pore_token,
        bq_token=bq_token,
        label_id=label_id,
        read_id=read_id,
    )
    return None
