"""
DeepRM training-data preparation (`deeprm train prep`).

Reads POD5 files and the matching BAM file, finds the context blocks of each read (deeprm.train.extract_block, chosen
with ``--method lev|pw``), normalises and segments the signal by the move table, and writes the tokens of each block
(signal, sequence, dwell time and base quality) as compressed .npz files to ``<output>/<postfix>`` (default "data"),
with the blocks themselves in ``<output>/blocks``. An output folder that already holds a run is refused.
"""

import argparse, gc, os, pod5, pysam, tqdm, glob, psutil, uuid, re, json, time, sys
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import pandas as pd
import shutil
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.compute as pc
import pyarrow.ipc as ipc
from deeprm.utils.utils import mean_phred
from deeprm.utils.logging import get_logger, add_log_path
from deeprm.train.extract_block import (
    DAGPairwiseBlockExtractor, DAGLevenshteinBlockExtractor,
    DEFAULT_DI_GAP_LIST, DEFAULT_DI_FLANK_LIST, DEFAULT_DI_SPACER_LIST, MONO_GAP,
)
from pathlib import Path

LOGGER = get_logger(__name__)


def create_segment_len_arr(segment_arr, sampling):
    """
    Compute segment-length array in sampling units for each sub-segment.

    Args:
        segment_arr (list of np.ndarray): raw signal segments.
        sampling (int): samples per signal unit.

    Returns:
        np.ndarray: integer length array per segment after downsampling.
    """
    return np.array([len(x) for x in segment_arr], dtype=int) // sampling


def move_to_dwell(move, quantile_a, quantile_b, shift_mult, scale_mult, sampling=6):
    """
    Transform raw move array into scaled dwell-time tokens.

    Steps:
    1. Convert boolean moves into positions and compute deltas.
    2. Log-transform dwell durations.
    3. Scale and shift based on quantiles and multipliers.

    Args:
        move (array-like): boolean array indicating move events per sample.
        quantile_a (float): lower quantile for scaling.
        quantile_b (float): upper quantile for scaling.
        shift_mult (float): shift multiplier.
        scale_mult (float): scale multiplier.
        sampling (int): samples per signal unit.

    Returns:
        np.ndarray: standardized dwell-time values.
    """
    move = np.asarray(move, dtype=np.bool_)
    if move.size == 0:
        return np.empty(0, dtype=np.float32)

    move_pos = np.arange(1, len(move) + 1, dtype=np.int32)[np.flip(move)]
    if move_pos.size == 0:
        return np.empty(0, dtype=np.float32)

    delta = np.empty(move_pos.size + 1, dtype=np.int32)
    delta[0] = 0
    delta[1:] = move_pos
    delta = delta[1:] - delta[:-1]
    dwell = np.log10((delta * sampling).astype(np.float32))
    # q as float32, so the quantile is placed in single precision with every numpy version (see move_to_dwell in
    # deeprm.inference.inference_preprocess_python)
    quantile_a_value = np.quantile(dwell, np.float32(quantile_a))
    quantile_b_value = np.quantile(dwell, np.float32(quantile_b))
    q_shift = max(0.1, shift_mult * (quantile_a_value + quantile_b_value))
    q_scale = max(0.1, scale_mult * (quantile_b_value - quantile_a_value))
    dwell = (dwell - q_shift) / q_scale
    return dwell.astype(np.float32, copy=False)


def trim_scale_segment_signal(signal, move, sp, ts, ns, quantile_a, quantile_b, shift_mult, scale_mult, sampling):
    move = np.asarray(move, dtype=np.bool_)
    signal = np.asarray(signal)
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
    signal = ((signal - q_shift) / q_scale).astype(np.float32, copy=False)
    move_idx = np.flatnonzero(move)[1:] * sampling
    move_idx = len(signal) - move_idx
    move_idx = np.flip(move_idx, axis=0)
    signal = np.array_split(signal, move_idx)
    if len(signal) == 0:
        return None
    return signal


def parse_pod5(pod5_path, read_ids):
    """
    Read signals and calibration from a POD5 file and return as Dict.

    Args:
        pod5_path (str): file path to the POD5 file.

    Returns:
        dict: Dictionary mapping read IDs to their corresponding signal arrays.

    """
    signal_dict = {}

    with pod5.Reader(pod5_path) as reader:
        for i, record in enumerate(reader.reads(read_ids)):
            try:
                signal_arr = record.signal
                rid = record.read_id.bytes
            except Exception as e:
                LOGGER.warning(f"Failed to read POD5 record from {pod5_path}: {e}")
                continue

            signal_dict[rid] = signal_arr

    gc.collect()
    return signal_dict


def quality_trimming_func(bq:np.array, threshold:int=25, reward:int=1, penalty:int=-2) -> tuple[int, int]:

    scores = np.where(bq >= threshold, reward, penalty)

    trim_3p_3p = np.argmin(np.concatenate((scores[-1:],np.cumsum(np.flip(scores)))))
    trim_3p_5p = np.argmin(np.concatenate([scores[0:1],np.cumsum(scores[:-trim_3p_3p or None])]))
    trim_5p_5p = np.argmin(np.concatenate([scores[0:1],np.cumsum(scores)]))
    trim_5p_3p = np.argmin(np.concatenate((scores[-1:],np.cumsum(np.flip(scores[trim_5p_5p:])))))

    if (trim_3p_3p + trim_3p_5p) > (trim_5p_3p + trim_5p_5p):
        trim_5p = trim_5p_5p
        trim_3p = trim_5p_3p

    else:
        trim_5p = trim_3p_5p
        trim_3p = trim_3p_3p

    if trim_5p == len(bq) or trim_3p == len(bq):
        return 0, 0

    return trim_5p, len(bq) - trim_3p


def quality_trimming(seq: str, bq:np.array, threshold:int=25, reward:int=1, penalty:int=-2, len_threshold = 10, artifact_cutoff = 0.5) -> tuple[int, int]:

    pat = "(GAA){5,}"
    pat = re.compile(pat)

    trim_finish = False
    trim_failed = False
    final_trim_5p = 0
    final_trim_3p = 0

    while not trim_finish:
        trim_5p, trim_3p = quality_trimming_func(bq=bq, threshold = threshold, reward = reward, penalty = penalty)
        if trim_3p - trim_5p <= len_threshold:
            trim_finish = True
            trim_failed = True
        else:
            seq_trimmed = seq[trim_5p:trim_3p]
            if np.sum([x.end()-x.start() for x in re.finditer(pat,seq_trimmed)])/len(seq_trimmed) < artifact_cutoff:
                trim_finish = True
                trim_failed = False
                final_trim_3p = final_trim_5p + trim_3p
                final_trim_5p = final_trim_5p + trim_5p
            else:
                trim_5p_len = trim_5p
                trim_3p_len = len(seq) - trim_3p
                if trim_3p_len >= trim_5p_len:
                    final_trim_5p = final_trim_5p + trim_3p
                    seq = seq[trim_3p:]
                    bq = bq[trim_3p:]
                else:
                    seq = seq[:trim_5p]
                    bq = bq[:trim_5p]

        if len(seq) == 0:
            trim_finish = True
            trim_failed = True

    if trim_failed:
        return 0, 0

    else:
        return final_trim_5p, final_trim_3p



def pad_signal(signal_list, pad_left, pad_right, sampling, mode="zero", min_len=1):
    """Pad a list of per-base signal segments on either side."""
    if mode == "none" or (pad_left == 0 and pad_right == 0):
        return signal_list
    if len(signal_list) == 0 or len(signal_list) < min_len:
        return signal_list

    dtype = signal_list[0].dtype
    if mode == "edge":
        left_value = np.median(signal_list[0])
        right_value = np.median(signal_list[-1])
        left_pad = [np.full(sampling, left_value, dtype=dtype) for _ in range(pad_left)]
        right_pad = [np.full(sampling, right_value, dtype=dtype) for _ in range(pad_right)]
    elif mode == "zero":
        left_pad = [np.zeros(sampling, dtype=dtype) for _ in range(pad_left)]
        right_pad = [np.zeros(sampling, dtype=dtype) for _ in range(pad_right)]
    else:
        raise ValueError(f"Invalid mode: {mode}, select between: (none/zero/edge)")
    return left_pad + list(signal_list) + right_pad


def pad_scalar(arr, pad_left, pad_right, mode="zero", min_len=1, constant_values = 0):
    """Pad a 1D scalar token array on either side."""
    if mode == "none" or (pad_left == 0 and pad_right == 0):
        return arr
    elif len(arr) == 0 or len(arr) < min_len:
        return arr
    elif mode == "edge":
        return np.pad(arr, (pad_left, pad_right), mode="edge")
    elif mode == "zero":
        return np.pad(arr, (pad_left, pad_right), mode="constant", constant_values=constant_values)
    else:
        raise ValueError(f"Invalid mode: {mode}, select between: (none/zero/edge)")


def slice_signal_segments(signal_list, start, end, left_bound, right_bound, sampling, padding_mode, min_len):
    """Slice per-base signal segments with optional padding against effective bounds."""
    if padding_mode == "none":
        if start < left_bound or end > right_bound:
            return None
        return signal_list[start:end]

    clipped_start = max(start, left_bound)
    clipped_end = min(end, right_bound)
    if clipped_start >= clipped_end:
        return None

    pad_left = max(0, left_bound - start)
    pad_right = max(0, end - right_bound)
    return pad_signal(
        signal_list[clipped_start:clipped_end],
        pad_left,
        pad_right,
        sampling,
        mode=padding_mode,
        min_len=min_len,
    )


def slice_scalar_token(arr, start, end, left_bound, right_bound, padding_mode, min_len, constant_values=0):
    """Slice a scalar token array with optional padding against effective bounds."""
    if padding_mode == "none":
        if start < left_bound or end > right_bound:
            return None
        return arr[start:end]

    clipped_start = max(start, left_bound)
    clipped_end = min(end, right_bound)
    if clipped_start >= clipped_end:
        return None

    pad_left = max(0, left_bound - start)
    pad_right = max(0, end - right_bound)
    return pad_scalar(
        arr[clipped_start:clipped_end],
        pad_left,
        pad_right,
        mode=padding_mode,
        min_len=min_len,
        constant_values=constant_values,
    )


def parse_bam(pid, n_procs, n_threads, out_path, bam_path, bq_cutoff, max_len, min_len,
              trim_penalty, trim_threshold, no_trim, sampling):
    """
    Extract move tags and alignment information from a BAM file in parallel.

    The first value of the Dorado `mv` tag is the move stride. It must match
    `sampling`; otherwise all downstream dwell/signal coordinates are wrong.
    This function therefore fails hard on stride mismatch instead of silently
    producing coordinate-corrupted tokens.
    """
    bam_df = {k: [] for k in ["read_id", "ts", "ns", "sp", "bq", "mv", "seq", "tl", "tr"]}
    skipped = {
        "missing_mv": 0,
        "bad_bq": 0,
        "bad_seq": 0,
        "length_filter": 0,
        "bad_uuid": 0,
        "bad_numeric": 0,
    }

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False, threads=n_threads) as input_bam:
        total = input_bam.mapped + input_bam.unmapped
        for i, read in tqdm.tqdm(
            enumerate(input_bam),
            total=total,
            desc=f"Parsing BAM P{pid + 1}/{n_procs}",
            leave=False,
        ):
            if i % n_procs != pid:
                continue
            if not read.has_tag("mv"):
                skipped["missing_mv"] += 1
                continue

            try:
                bq_raw = read.query_qualities
                if bq_raw is None:
                    skipped["bad_bq"] += 1
                    continue
                bq = np.asarray(bq_raw, dtype=np.float32)
                if bq_cutoff > 0 and mean_phred(bq) < bq_cutoff:
                    continue
                bq = np.clip(bq, 0, 60).astype(np.uint8)
            except Exception as e:
                skipped["bad_bq"] += 1
                LOGGER.debug(f"Skipping read {read.query_name}: failed BQ parsing: {e}")
                continue

            seq_str = read.query_sequence
            if seq_str is None:
                skipped["bad_seq"] += 1
                continue
            seqlen = len(seq_str)
            if seqlen == 0 or seqlen > max_len or seqlen < min_len:
                skipped["length_filter"] += 1
                continue
            if len(bq) != seqlen:
                skipped["bad_seq"] += 1
                LOGGER.debug(
                    f"Skipping read {read.query_name}: sequence/BQ length mismatch "
                    f"({seqlen} vs {len(bq)})"
                )
                continue
            seq = np.frombuffer(seq_str.encode("ascii"), dtype=np.int8)

            if not no_trim:
                trim_5p, trim_3p = quality_trimming(
                    seq_str,
                    bq.astype(np.int32),
                    penalty=trim_penalty,
                    threshold=trim_threshold,
                )
            else:
                trim_5p, trim_3p = 0, seqlen

            try:
                read_id_raw = str(read.get_tag("pi")) if read.has_tag("pi") else str(read.query_name)
                read_id = uuid.UUID(read_id_raw).bytes
            except Exception as e:
                skipped["bad_uuid"] += 1
                LOGGER.debug(f"Skipping read {read.query_name}: invalid UUID: {e}")
                continue

            ts = read.get_tag("ts") if read.has_tag("ts") else 0
            ns = read.get_tag("ns") if read.has_tag("ns") else 0
            sp = read.get_tag("sp") if read.has_tag("sp") else 0

            mv_tag = read.get_tag("mv")
            if len(mv_tag) == 0:
                skipped["missing_mv"] += 1
                continue
            mv_stride = int(mv_tag[0])
            if mv_stride != sampling:
                raise ValueError(
                    f"BAM mv stride mismatch in read {read.query_name}: "
                    f"mv[0]={mv_stride}, but --sampling={sampling}. "
                    "Use the correct --sampling or regenerate BAM with the expected move stride."
                )
            mv = np.asarray(mv_tag[1:], dtype=np.bool_)

            if np.any(np.isnan(bq.astype(np.float32))) or np.any(np.isnan([ts, ns, sp])):
                skipped["bad_numeric"] += 1
                continue

            bam_df["read_id"].append(read_id)
            bam_df["ts"].append(ts)
            bam_df["ns"].append(ns)
            bam_df["sp"].append(sp)
            bam_df["bq"].append(bq)
            bam_df["mv"].append(mv)
            bam_df["seq"].append(seq)
            bam_df["tl"].append(trim_5p)
            bam_df["tr"].append(trim_3p)

    save_bam_as_feather(bam_df, out_path)
    LOGGER.info(
        f"Parsed BAM shard {pid + 1}/{n_procs}: kept={len(bam_df['read_id'])}, "
        f"skipped={skipped}"
    )
    del bam_df
    gc.collect()
    return None


def segmented_signal_to_block(signal_segmented, segment_len_arr, kmer, sampling, sig_window, pad_to):
    """Return one fixed-length signal token or None when the geometry is invalid."""
    if len(signal_segmented) == 0:
        return None

    kmer_pad = (kmer - 1) // 2
    lr_pad = (sig_window - 1) // 2
    l_skip = int((np.sum(segment_len_arr[:kmer_pad]) - lr_pad) * sampling)
    r_skip = int((np.sum(segment_len_arr[-kmer_pad:]) - lr_pad) * sampling)
    if l_skip < 0 or r_skip < 0:
        return None

    try:
        signal_token = np.concatenate(signal_segmented)
    except ValueError:
        return None

    if len(signal_token) % sampling != 0:
        return None

    signal_token = signal_token[l_skip:] if r_skip == 0 else signal_token[l_skip:-r_skip]

    expected_len = int((pad_to + kmer - 1) * sampling)
    if len(signal_token) > expected_len:
        return None
    if len(signal_token) < expected_len:
        signal_token = np.pad(
            signal_token,
            (0, expected_len - len(signal_token)),
            mode="constant",
            constant_values=0,
        )
    if len(signal_token) != expected_len:
        return None
    return signal_token.astype(np.float32, copy=False)


def _atomic_pickle(df: pd.DataFrame, save_path: str) -> None:
    """Write a pickle atomically so interrupted jobs do not leave valid-looking partial files."""
    tmp_path = f"{save_path}.tmp.{os.getpid()}"
    try:
        df.to_pickle(tmp_path)
        os.replace(tmp_path, save_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def save_npz(save_path, data):

    if not data or len(data["score"]) == 0:
        return None

    tmp_path = f"{save_path}.tmp.{os.getpid()}.npz"
    try:
        arrays = dict(
            segment_len_arr=np.stack(data["segment_len_arr"]),
            signal_token=np.stack(data["signal_token"]),
            kmer_token=np.stack(data["kmer_token"]),
            dwell_motor_token=np.stack(data["dwell_motor_token"]),
            dwell_pore_token=np.stack(data["dwell_pore_token"]),
            bq_token=np.stack(data["bq_token"]),
            block_score=np.asarray(data["score"], dtype=np.float32),
        )
        if "read_id" in data:
            arrays.update(
                read_id=np.asarray(data["read_id"], dtype="S16"),
                pos_RM=np.asarray(data["pos_RM"], dtype=np.int64),
                cb_idx=np.asarray(data["cb_idx"], dtype=np.int32),
                tl=np.asarray(data["tl"], dtype=np.int32),
                tr=np.asarray(data["tr"], dtype=np.int32),
            )
        # v4: which anchor of the context block this token is (0 for mono
        # blocks, 0/1 for di-m6A) and which designed type the block resolved to
        # (-1 = mono). Both are written whenever present so downstream code can
        # pair the two m6A sites of a di block and stratify by their spacing.
        if "anchor_idx" in data:
            arrays.update(anchor_idx=np.asarray(data["anchor_idx"], dtype=np.int32))
        if "di_gap" in data:
            arrays.update(di_gap=np.asarray(data["di_gap"], dtype=np.int32))
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, save_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return None


def _build_block_metadata(bam_df, from_block_df, min_score, cb_pad, padding_mode="none"):
    read_ids = bam_df["read_id"].tolist()
    tl_arr = bam_df["tl"].to_numpy(dtype=np.int64, copy=False)
    tr_arr = bam_df["tr"].to_numpy(dtype=np.int64, copy=False)

    read_row = []
    out_read_id = []
    pos_rm = []
    score = []
    cb_idx = []
    anchor_idx = []
    di_gap = []
    tl_keep = []
    tr_keep = []

    def _column_or_default(df, name, default):
        """v4 columns are optional so older block pickles keep loading."""
        if name in df.columns:
            return df[name]
        return [default] * len(df)

    if from_block_df is None:
        for row_idx, (rid, tl, tr, pos_list, score_list, idx_list, anc_list, gap_list) in enumerate(
                zip(
                    read_ids,
                    tl_arr,
                    tr_arr,
                    bam_df["pos_RM"],
                    bam_df["score"],
                    bam_df["cb_idx"],
                    bam_df["anchor_idx"],
                    bam_df["di_gap"],
                )
        ):
            if not pos_list:
                continue
            for pos, sc, idx, anc, gap in zip(pos_list, score_list, idx_list, anc_list, gap_list):
                if sc < min_score:
                    continue
                if padding_mode == "none":
                    if pos - cb_pad < 0:
                        continue
                    if pos + cb_pad + 1 > (tr - tl):
                        continue
                else:
                    if pos < 0 or pos >= (tr - tl):
                        continue
                read_row.append(row_idx)
                out_read_id.append(rid)
                pos_rm.append(int(pos) + int(tl))
                score.append(sc)
                cb_idx.append(int(idx))
                anchor_idx.append(int(anc))
                di_gap.append(int(gap))
                tl_keep.append(int(tl))
                tr_keep.append(int(tr))
    else:
        read_index = {rid: idx for idx, rid in enumerate(read_ids)}
        iterator = zip(
            from_block_df["read_id"],
            from_block_df["pos_RM"],
            from_block_df["score"],
            _column_or_default(from_block_df, "cb_idx", -1),
            _column_or_default(from_block_df, "anchor_idx", 0),
            _column_or_default(from_block_df, "di_gap", MONO_GAP),
        )
        for rid, pos, sc, idx, anc, gap in iterator:
            row_idx = read_index.get(rid)
            if row_idx is None or sc < min_score:
                continue
            tl = tl_arr[row_idx]
            tr = tr_arr[row_idx]
            if padding_mode == "none":
                if pos - cb_pad < tl:
                    continue
                if pos + cb_pad + 1 > tr:
                    continue
            else:
                if pos < tl or pos >= tr:
                    continue
            read_row.append(row_idx)
            out_read_id.append(rid)
            pos_rm.append(int(pos))
            score.append(sc)
            cb_idx.append(int(idx))
            anchor_idx.append(int(anc))
            di_gap.append(int(gap))
            tl_keep.append(int(tl))
            tr_keep.append(int(tr))

    if not pos_rm:
        return pd.DataFrame(columns=["read_row", "read_id", "pos_RM", "score", "cb_idx",
                                     "anchor_idx", "di_gap", "tl", "tr"])

    return pd.DataFrame(
        {
            "read_row": np.asarray(read_row, dtype=np.int32),
            "read_id": out_read_id,
            "pos_RM": np.asarray(pos_rm, dtype=np.int64),
            "score": np.asarray(score),
            "cb_idx": np.asarray(cb_idx, dtype=np.int32),
            "anchor_idx": np.asarray(anchor_idx, dtype=np.int32),
            "di_gap": np.asarray(di_gap, dtype=np.int32),
            "tl": np.asarray(tl_keep, dtype=np.int32),
            "tr": np.asarray(tr_keep, dtype=np.int32),
        }
    )


def _prepare_read_cache(bam_df, block_df, signal_dict, quantile_a, quantile_b, shift_mult, scale_mult, sampling):
    if block_df.empty:
        return {}

    used_rows = np.unique(block_df["read_row"].to_numpy(dtype=np.int64, copy=False))
    read_ids = bam_df["read_id"].tolist()
    seq_list = bam_df["seq"].tolist()
    bq_list = bam_df["bq"].tolist()
    mv_list = bam_df["mv"].tolist()
    ts_arr = bam_df["ts"].to_numpy(dtype=np.int64, copy=False)
    ns_arr = bam_df["ns"].to_numpy(dtype=np.int64, copy=False)
    sp_arr = bam_df["sp"].to_numpy(dtype=np.int64, copy=False)
    tr_arr = bam_df["tr"].to_numpy(dtype=np.int64, copy=False)

    cache = {}
    for row_idx in used_rows:
        rid = read_ids[row_idx]
        signal = signal_dict.get(rid)
        if signal is None:
            continue
        seq = seq_list[row_idx]
        bq = bq_list[row_idx]
        mv = mv_list[row_idx]
        if len(seq) != len(bq):
            LOGGER.debug(f"Skipping read cache row {row_idx}: seq/BQ length mismatch")
            continue
        dwell_token = move_to_dwell(mv, 0.2, 0.8, 0.5, 1.5, sampling)
        dwell_length = len(dwell_token)
        if dwell_length == 0:
            continue
        signal_segmented = trim_scale_segment_signal(
            signal,
            mv,
            int(sp_arr[row_idx]),
            int(ts_arr[row_idx]),
            int(ns_arr[row_idx]),
            quantile_a,
            quantile_b,
            shift_mult,
            scale_mult,
            sampling,
        )
        if signal_segmented is None:
            continue
        if len(signal_segmented) < int(tr_arr[row_idx]):
            LOGGER.debug(
                f"Skipping read cache row {row_idx}: signal segments shorter than trim end "
                f"({len(signal_segmented)} < {int(tr_arr[row_idx])})"
            )
            continue
        cache[int(row_idx)] = {
            "seq": seq,
            "bq": bq,
            "dwell_token": dwell_token,
            "dwell_length": dwell_length,
            "signal": signal_segmented,
        }
    return cache


def segment_normalize_signal(
        shm_name,
        buf_len,
        pod5_paths,
        norm_factor,
        block_output_path,
        token_output_path,
        dag_config,
        process_once=1000,
        kmer=5,
        target_cb_len=21,
        sampling=6,
        sig_window=5,
        min_score=0.0,
        max_token_len=200,
        dwell_shift=10,
        reuse_blocks_dir=None,
        pkl_allowlist=None,
        dag_method ="pw",
        padding_mode="none"
):
    """
    Segment and normalize signals per read, and save token chunks.

    If `reuse_blocks_dir` is provided, this function will load precomputed block
    anchors from `{reuse_blocks_dir}/{pod5_name}-{chunk_idx}.pkl` and skip the
    expensive DAG/aligner-based block extraction (`extract_blocks_from_reads`).
    It will then re-tokenize signals using the current `target_cb_len` and
    write tokens to `{token_output_path}/{pod5_name}-{chunk_idx}.npz`.

    Args:
        shm_name (str): name of shared memory containing Arrow IPC bytes.
        buf_len (int): length of valid bytes in shared memory.
        pod5_paths (list): list of POD5 file paths.
        norm_factor (dict): normalization parameters.
        block_output_path (str): directory to write (or mirror) .pkl block files.
        token_output_path (str): directory to write .npz chunks.
        dag_config (dict): DAG extraction config (used only when reuse_blocks_dir is None).
        process_once (int): reads to process per batch when extracting blocks.
        kmer (int): k-mer size.
        target_cb_len (int): context block length for tokenization.
        sampling (int): samples per signal unit.
        sig_window (int): signal window size.
        min_score (float): minimum score to keep a block. PW scores are normalized to [0, 1].
        max_token_len (int): maximum token length to keep.
        dwell_shift (int): dwell shift used for motor/pore token slicing.
        reuse_blocks_dir (str|None): directory containing precomputed `blocks/*.pkl`.
        pkl_allowlist (list[str]|None): if provided, only process block pickles whose
            basename (without extension) is in this allowlist.
        padding_mode (str): One of "none", "zero", or "edge" for boundary handling
            during token slicing.

    Returns:
        None (writes .npz files).
    """
    # Attach to shared memory Arrow table
    shm = shared_memory.SharedMemory(name=shm_name)
    buf = pa.py_buffer(memoryview(shm.buf[:buf_len]))
    tab = ipc.open_stream(pa.BufferReader(buf)).read_all()

    p = psutil.Process(os.getpid())
    p.ionice(psutil.IOPRIO_CLASS_BE, 0)

    shift_mult = norm_factor["shift_mult"]
    scale_mult = norm_factor["scale_mult"]
    quantile_a = norm_factor["quantile_a"]
    quantile_b = norm_factor["quantile_b"]

    trim = kmer // 2
    cb_pad = target_cb_len // 2
    allowset = set(pkl_allowlist) if pkl_allowlist is not None else None
    extractor = None
    if reuse_blocks_dir is None:
        if dag_method == "pw":
            extractor = DAGPairwiseBlockExtractor(**dag_config)
        elif dag_method == "lev":
            extractor = DAGLevenshteinBlockExtractor(**dag_config)
        else:
            raise ValueError(f"Unknown dag_method: {dag_method}")

    for pod5_path in tqdm.tqdm(pod5_paths):
        pod5_name = os.path.basename(pod5_path).replace(".pod5", "")

        if reuse_blocks_dir is not None:
            # Directly iterate over existing block pickles for this POD5
            pkl_paths = sorted(glob.glob(os.path.join(reuse_blocks_dir, f"{pod5_name}-*.pkl")))
            if allowset is not None:
                pkl_paths = [p for p in pkl_paths if os.path.splitext(os.path.basename(p))[0] in allowset]
            if len(pkl_paths) == 0:
                continue

            for pkl_path in pkl_paths:
                base = os.path.splitext(os.path.basename(pkl_path))[0]
                block_path = os.path.join(block_output_path, f"{base}.pkl")
                token_path = os.path.join(token_output_path, f"{base}.npz")

                try:
                    block_df = pd.read_pickle(pkl_path)
                except Exception as e:
                    LOGGER.warning(f"Failed to read block pickle {pkl_path}: {e}")
                    continue
                if block_df is None or len(block_df) == 0:
                    continue

                # Convert UUID bytes -> UUID string for pod5.Reader.reads()
                try:
                    read_ids_bytes = block_df["read_id"].unique().tolist()
                    read_ids_pod5 = [str(uuid.UUID(bytes=rid)) for rid in read_ids_bytes]
                except Exception as e:
                    LOGGER.warning(f"Invalid read_id column in block pickle {pkl_path}: {e}")
                    continue

                try:
                    signal_dict = parse_pod5(pod5_path, read_ids_pod5)
                except Exception:
                    LOGGER.warning(f"Corrupted POD5 file: {pod5_path}")
                    continue
                if len(signal_dict) == 0:
                    continue

                bam_df = get_bam_rows_for_batch(signal_dict.keys(), tab)
                if bam_df.empty:
                    continue

                tokenize(bam_df=bam_df,
                         signal_dict=signal_dict,
                         pod5_path=pod5_path,
                         token_path=token_path,
                         block_path = block_path,
                         min_score=min_score,
                         quantile_a=quantile_a,
                         quantile_b=quantile_b,
                         shift_mult=shift_mult,
                         scale_mult=scale_mult,
                         sampling=sampling,
                         kmer=kmer,
                         target_cb_len=target_cb_len,
                         sig_window=sig_window,
                         max_token_len=max_token_len,
                         dwell_shift=dwell_shift,
                         cb_pad=cb_pad,
                         trim=trim,
                         from_block_df=block_df,
                         padding_mode=padding_mode,
                         )

            # Done with this POD5 in reuse mode

        else:
            # ---------------------------------------------------------------------
            # Original path: compute blocks via DAG extraction per chunk, then tokenize
            # ---------------------------------------------------------------------
            with pod5.Reader(pod5_path) as reader:
                read_ids = reader.read_ids

            read_id_split = np.array_split(np.array(read_ids), np.ceil(len(read_ids) / process_once))

            for chunk_idx, read_ids in enumerate(read_id_split):
                base = f"{pod5_name}-{chunk_idx}"
                if allowset is not None and base not in allowset:
                    LOGGER.warning(f"POD5 not in ALLOWSET: {pod5_path}")
                    continue

                block_path = os.path.join(block_output_path, f"{pod5_name}-{chunk_idx}.pkl")
                token_path = os.path.join(token_output_path, f"{pod5_name}-{chunk_idx}.npz")

                try:
                    signal_dict = parse_pod5(pod5_path, read_ids)
                except Exception:
                    LOGGER.warning(f"Corrupted POD5 file: {pod5_path}")
                    continue
                if len(signal_dict) == 0:
                    LOGGER.warning(f"No BAM-matched reads in POD5 file: {pod5_path}")
                    continue

                # Pull only the BAM rows for this batch from the memory-mapped Feather table
                bam_df = get_bam_rows_for_batch(signal_dict.keys(), tab)
                if bam_df.empty:
                    LOGGER.warning(f"No BAM-matched reads in POD5 file: {pod5_path}")
                    continue

                extraction_dict = extractor.extract_blocks(bam_df=bam_df)

                bam_df["pos_RM"] = extraction_dict["pos"]
                bam_df["score"] = extraction_dict["score"]
                bam_df["cb_idx"] = extraction_dict["cb_idx"]
                bam_df["global_idx"] = extraction_dict["global_idx"]
                bam_df["anchor_idx"] = extraction_dict["anchor_idx"]
                bam_df["di_gap"] = extraction_dict["di_gap"]

                bam_df = bam_df[bam_df["pos_RM"].apply(len) > 0]
                if len(bam_df) == 0:
                    LOGGER.warning(f"No valid positions extracted from: {pod5_path}")
                    continue

                tokenize(bam_df=bam_df,
                         signal_dict=signal_dict,
                         pod5_path=pod5_path,
                         token_path=token_path,
                         block_path = block_path,
                         min_score=min_score,
                         quantile_a=quantile_a,
                         quantile_b=quantile_b,
                         shift_mult=shift_mult,
                         scale_mult=scale_mult,
                         sampling=sampling,
                         kmer=kmer,
                         target_cb_len=target_cb_len,
                         sig_window=sig_window,
                         max_token_len=max_token_len,
                         dwell_shift=dwell_shift,
                         cb_pad=cb_pad,
                         trim=trim,
                         from_block_df=None,
                         padding_mode=padding_mode,
                         )

    # Tear down shared memory
    tab = None
    buf = None
    gc.collect()
    shm.close()
    return None


def tokenize(bam_df, signal_dict, pod5_path, token_path, block_path, min_score,
             quantile_a, quantile_b, shift_mult, scale_mult, sampling,
             kmer, target_cb_len, sig_window, max_token_len, dwell_shift, cb_pad, trim,
             from_block_df=None, padding_mode="none"):

    block_df = _build_block_metadata(
        bam_df=bam_df,
        from_block_df=from_block_df,
        min_score=min_score,
        cb_pad=cb_pad,
        padding_mode=padding_mode,
    )
    if block_df.empty:
        return None

    read_cache = _prepare_read_cache(
        bam_df=bam_df,
        block_df=block_df,
        signal_dict=signal_dict,
        quantile_a=quantile_a,
        quantile_b=quantile_b,
        shift_mult=shift_mult,
        scale_mult=scale_mult,
        sampling=sampling,
    )
    if not read_cache:
        return None

    valid_read_rows = set(read_cache.keys())
    block_df = block_df[block_df["read_row"].isin(valid_read_rows)].reset_index(drop=True)
    if block_df.empty:
        return None

    token_store = {
        "score": [],
        "segment_len_arr": [],
        "signal_token": [],
        "kmer_token": [],
        "dwell_motor_token": [],
        "dwell_pore_token": [],
        "bq_token": [],
        "read_id": [],
        "pos_RM": [],
        "cb_idx": [],
        "anchor_idx": [],
        "di_gap": [],
        "tl": [],
        "tr": [],
    }

    min_len = cb_pad + 2

    for row in block_df.itertuples(index=False):
        cache = read_cache.get(int(row.read_row))
        if cache is None:
            continue

        left_bound = int(row.tl)
        right_bound = int(row.tr)
        start_pos = int(row.pos_RM) - cb_pad
        end_pos = int(row.pos_RM) + cb_pad + 1

        signal_slice = slice_signal_segments(
            cache["signal"],
            start_pos,
            end_pos,
            left_bound,
            right_bound,
            sampling,
            padding_mode,
            min_len,
        )
        if signal_slice is None:
            continue

        segment_len_arr = create_segment_len_arr(signal_slice, sampling)
        if len(segment_len_arr) != target_cb_len:
            continue

        dwell_motor_token = slice_scalar_token(
            cache["dwell_token"],
            start_pos + dwell_shift + trim,
            end_pos + dwell_shift - trim,
            left_bound,
            right_bound,
            padding_mode,
            1,
            constant_values = 0
        )
        dwell_pore_token = slice_scalar_token(
            cache["dwell_token"],
            start_pos + trim,
            end_pos - trim,
            left_bound,
            right_bound,
            padding_mode,
            min_len - trim,
            constant_values = 0
        )
        bq_token = slice_scalar_token(
            cache["bq"],
            start_pos + trim,
            end_pos - trim,
            left_bound,
            right_bound,
            padding_mode,
            min_len - trim,
            constant_values = 0
        )
        kmer_token = slice_scalar_token(
            cache["seq"],
            start_pos,
            end_pos,
            left_bound,
            right_bound,
            padding_mode,
            min_len,
            constant_values = 0
        )

        if any(x is None for x in (dwell_motor_token, dwell_pore_token, bq_token, kmer_token)):
            continue

        if len(dwell_motor_token) != (target_cb_len - 2 * trim):
            continue
        if len(dwell_pore_token) != (target_cb_len - 2 * trim):
            continue
        if len(bq_token) != (target_cb_len - 2 * trim):
            continue
        if len(kmer_token) != target_cb_len:
            continue

        token_len = int(np.sum(segment_len_arr[trim:-trim]))
        if token_len <= 0 or token_len > max_token_len:
            continue

        signal_token = segmented_signal_to_block(signal_slice, segment_len_arr, kmer, sampling, sig_window, max_token_len)
        if signal_token is None:
            continue

        token_store["score"].append(float(row.score))
        token_store["segment_len_arr"].append(segment_len_arr[trim:-trim].astype(np.uint16, copy=False))
        token_store["signal_token"].append(signal_token)
        token_store["kmer_token"].append(np.asarray(kmer_token))
        token_store["dwell_motor_token"].append(np.asarray(dwell_motor_token))
        token_store["dwell_pore_token"].append(np.asarray(dwell_pore_token))
        token_store["bq_token"].append(np.asarray(bq_token, dtype=np.int8))
        token_store["read_id"].append(row.read_id)
        token_store["pos_RM"].append(int(row.pos_RM))
        token_store["cb_idx"].append(int(row.cb_idx))
        token_store["anchor_idx"].append(int(row.anchor_idx))
        token_store["di_gap"].append(int(row.di_gap))
        token_store["tl"].append(int(row.tl))
        token_store["tr"].append(int(row.tr))

    if not token_store["score"]:
        return None

    accepted_block_df = pd.DataFrame({
        "read_id": token_store["read_id"],
        "pos_RM": np.asarray(token_store["pos_RM"], dtype=np.int64),
        "score": np.asarray(token_store["score"], dtype=np.float32),
        "cb_idx": np.asarray(token_store["cb_idx"], dtype=np.int32),
        "anchor_idx": np.asarray(token_store["anchor_idx"], dtype=np.int32),
        "di_gap": np.asarray(token_store["di_gap"], dtype=np.int32),
        "tr": np.asarray(token_store["tr"], dtype=np.int32),
        "tl": np.asarray(token_store["tl"], dtype=np.int32),
    })
    _atomic_pickle(accepted_block_df, block_path)
    save_npz(token_path, token_store)
    gc.collect()
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


def save_bam_as_feather(bam_df, path):
    tbl = pa.table({
        # Normal columns: ensure fixed-size types for compact storage
        "read_id": pa.array(bam_df["read_id"], type=pa.binary(16)),
        "ts": pa.array(np.array(bam_df["ts"], dtype=np.int32)),
        "ns": pa.array(np.array(bam_df["ns"], dtype=np.int32)),
        "sp": pa.array(np.array(bam_df["sp"], dtype=np.int32)),
        "tl": pa.array(np.array(bam_df["tl"], dtype=np.int32)),
        "tr": pa.array(np.array(bam_df["tr"], dtype=np.int32)),
        # list columns: store variable-length arrays compactly
        "bq": pa.array(bam_df["bq"], type=pa.large_list(pa.int8())),
        "mv": pa.array(bam_df["mv"], type=pa.large_list(pa.bool_())),
        "seq": pa.array(bam_df["seq"], type=pa.large_list(pa.int8())),
    })
    feather.write_feather(tbl, path)
    return None

def get_bam_rows_for_batch(read_ids, tab) -> pd.DataFrame:
    """
    From the memory-mapped Feather table, return only rows whose read_id is in `read_ids`.
    Keep list-like columns in Python form until they are actually needed to reduce
    per-batch materialization overhead. `seq` is eagerly converted because DAG extraction
    requires NumPy arrays with `.tobytes()` support.
    """
    key = pa.array(read_ids, type=pa.binary(16))
    mask = pc.is_in(tab["read_id"], value_set=key)
    sub = tab.filter(mask)
    pdf = sub.to_pandas()
    if pdf.empty:
        return pdf
    pdf["seq"] = [np.asarray(x, dtype=np.int8) for x in pdf["seq"]]
    return pdf


def _list_lengths_from_chunk(arr: pa.LargeListArray) -> memoryview:
    # Returns per-row lengths (offsets[i+1] - offsets[i]) as a NumPy-style view
    # Works without copying; arrow offsets are int64 for LargeList
    # Note: for pa.list_(), use arr.offsets.cast('int64') similarly.
    offs = arr.offsets  # int64 buffer
    # Slice to [1:] and [:-1] without copies
    b = offs.buffers()[1]
    # Convert to memoryview of int64 (Arrow ensures 8-byte aligned)
    mv = memoryview(b).cast('q')  # signed long long
    # mv[0:len(arr)+1] are offsets; lengths are diff of adjacent pairs.
    # We can't diff a memoryview in place; we’ll scan as we accumulate.
    return mv


def _iter_weighted_slices(table: pa.Table, list_cols: list[str], max_values: int):
    """
    Yield (start, length) row windows where the sum of list lengths across list_cols
    in the window is <= max_values. Works chunk-by-chunk to avoid materializing big arrays.
    """
    # Build chunk iterators for columns we care about
    list_col_chunks = []
    for name in list_cols:
        carr = table[name]
        # Ensure LargeList; if it's list(), it still works (offsets are int32) but safer to cast earlier.
        chunks = []
        for ch in carr.chunks if hasattr(carr, "chunks") else [carr]:
            if not pa.types.is_large_list(ch.type) and not pa.types.is_list(ch.type):
                raise TypeError(f"Column {name} must be (Large)List; got {ch.type}")
            chunks.append(ch)
        list_col_chunks.append(chunks)

    # We will walk rows using the chunk boundaries of the first list column.
    # To keep logic simple, we first align all columns to the same chunking:
    # combine_chunks() is safe here per-part IF parts are modest; otherwise we can re-chunk cheaply.
    t = table  # assume per-part table not enormous; if huge, skip combine and handle per-chunk.

    n = t.num_rows
    start = 0
    while start < n:
        # Grow the slice until we would exceed max_values
        length = 0
        # We compute lengths using offsets of each list col without materializing arrays
        # For simplicity, we use to_pandas().apply(len) on a tiny probe window when rows are small,
        # but here we do a low-overhead scan via offsets.
        step = 1
        # Try to grow geometrically, then back off (binary-ish growth) to reduce slicing overhead
        while start + length < n:
            trial_len = min(n - start, length + step)
            # Estimate total values for list columns in [start, start+trial_len)
            # We compute by slicing the column (cheap; zero-copy) and reading child length summations.
            sub_total = 0
            for name in list_cols:
                sub = t[name][start:start+trial_len]  # zero-copy slice
                # Turn to a single chunk for offsets access; this is small slice
                if isinstance(sub, pa.ChunkedArray):
                    sub = pa.chunked_array(sub.chunks).combine_chunks()
                arr = sub  # LargeListArray
                offs = arr.offsets
                obuf = offs.buffers()[1]
                omv = memoryview(obuf).cast('q')  # int64 offsets
                # offsets length == trial_len+1
                sub_total += (omv[trial_len] - omv[0])

            if sub_total <= max_values:
                length = trial_len
                step = max(1, step * 2)  # try to grow faster
            else:
                # too big; shrink growth
                if step == 1:
                    break
                step = max(1, step // 2)

        if length == 0:
            # Pathological single row with gigantic list(s); just skip it
            length = 1
            skip = True
        else:
            skip = False

        yield (start, length, skip)
        start += length


def table_to_shared_memory(paths: list,
                           list_cols=("bq","mv","seq"),
                           max_values=50_000_000):
    """
    Stream multiple small record batches into a single IPC stream, ensuring that
    for each batch, the sum of list child-element counts across `list_cols`
    stays <= max_values. This avoids Arrow's 2^31-1 primitive array limit.
    """
    if not paths:
        raise ValueError("No tables to serialize")
    total_len = 0
    # Unify schema to avoid drift (nulls, empty inferred types, etc.)
    schema = feather.read_table(paths[0]).schema
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, schema) as writer:
        for idx, path in tqdm.tqdm(enumerate(paths), desc="Serializing BAM Table", total=len(paths)):
            tbl = feather.read_table(path)
            total_len += tbl.num_rows
            if tbl.schema != schema:
                tbl = tbl.cast(schema, safe=False)
            # Produce slices that respect value-volume cap
            for start, length, skip in _iter_weighted_slices(tbl, list(list_cols), max_values):
                if not skip:
                    writer.write_table(tbl.slice(start, length))
    buf = sink.getvalue().to_pybytes()
    shm = shared_memory.SharedMemory(create=True, size=len(buf))
    shm.buf[:len(buf)] = buf
    LOGGER.info(f"Serialized {total_len} BAM rows")
    return shm, len(buf)


def distribute_records(idxstat, n_procs):
    """
    Distribute records from idxstat output to n_procs with greedy algorithm.
    Args:
        idxstat (list of tuples): output from pysam.get_index_statistics().
        n_procs (int): number of processes to distribute to.
    Returns:
        list of list: distributed records.
    """
    proc_records = [[] for _ in range(n_procs)]
    proc_sizes = [0 for _ in range(n_procs)]
    sorted_idxstat = sorted(idxstat, key=lambda x: x.mapped, reverse=True)
    for record in sorted_idxstat:
        min_proc = np.argmin(proc_sizes)
        proc_records[min_proc].append(record)
        proc_sizes[min_proc] += record.mapped
    proc_ref_names = [[r.contig for r in recs] for recs in proc_records]
    proc_n_reads = [sum(r.mapped for r in recs) for recs in proc_records]
    LOGGER.info(f"Distributed {len(idxstat)} references to {n_procs} processes")
    return proc_ref_names, proc_n_reads



def _opt(*names):
    """Option strings with both spellings of each long option (--max-token-len and --max_token_len)."""
    out = []
    for name in names:
        out.append(name)
        if name.startswith("--") and ("_" in name[2:] or "-" in name[2:]):
            alt = "--" + (name[2:].replace("_", "-") if "_" in name[2:] else name[2:].replace("-", "_"))
            if alt not in out:
                out.append(alt)
    return out


# Defaults of the two extraction methods (research parse_args: the "lev" and "pw" subcommands). Every method option
# is parsed with default None, then filled from the table of the chosen method (_resolve_method_defaults).
LEV_DEFAULTS = {
    "min_score": 0.0, "indel_tolerance": 3, "indel_penalty": 3, "cb_size_tolerance": 3, "skip_size_tolerance": 4,
    "anchor_mismatch_penalty": 6, "anchor_uncertainty_penalty": 2, "spacer_mismatch_tolerance": 3,
    "spacer_mismatch_penalty": 2, "spacer_size_tolerance": 1, "anchor_list": ["A", "A", "A"], "anchor_shift": None,
    "spacer_list": ["CGACAU", "CCAUUG", "AAGCGU", "GUAGUC"], "spacer_size": 6, "cb_pad": 10, "cb_per_bb": 3,
    "skip_anchor": False,
}
PW_DEFAULTS = {
    "min_score": 0.0, "indel_tolerance": 25, "indel_penalty": 26, "cb_size_tolerance": 19,
    "anchor_mismatch_penalty": 150, "anchor_uncertainty_penalty": 2, "spacer_size": None, "cb_pad": 20,
    "cb_per_bb": 1, "spacer_match_score": 3, "spacer_mismatch_penalty": 5, "spacer_gap_open_penalty": 3,
    "spacer_gap_extend_penalty": 1, "spacer_edit_tolerance": 6, "anchor_list": ["A"], "anchor_shift": None,
    "spacer_list": None, "max_iter": 10, "max_alignments_per_iter": 2000, "skip_anchor": False,
    "di_anchor": False, "di_gap_list": None, "di_flank_list": None, "di_include_mono": False,
    "di_allow_partial": False, "di_type": None, "di_consensus": "batch", "di_consistent_type": True,
    "true_bb_period": None, "edge_blocks": False, "edge_anchor_window": 1,
}
METHOD_DEFAULTS = {"lev": LEV_DEFAULTS, "pw": PW_DEFAULTS}


def _defaults_text(key):
    """The help-text note of an extraction option: its default under each method that has it."""
    parts = []
    for method, table in METHOD_DEFAULTS.items():
        if key in table and table[key] is not None:
            value = table[key]
            parts.append(f"{method}: {' '.join(map(str, value)) if isinstance(value, list) else value}")
    return f"(default {'; '.join(parts)})" if parts else "(default: set from the design)"


def add_arguments(parser: argparse.ArgumentParser):
    """Adds the command-line arguments of `deeprm train prep`.

    Args:
        parser (argparse.ArgumentParser): Argument parser to which arguments will be added.

    Returns:
        None
    """
    num_cpu = os.cpu_count() or 1
    parser.add_argument("--pod5", "-p", type=str, default=None, help="POD5 input directory")
    parser.add_argument("--bam", "-b", type=str, default=None, help="Dorado BAM file (with move tables)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--method", dest="dag_method", choices=tuple(METHOD_DEFAULTS), default="lev",
        help="Context-block extraction: lev (Levenshtein, for designs with 6-nt spacers; default) or pw (pairwise "
             "alignment, for designs with 15-nt spacers, also with two As per block (--di))",
    )
    parser.add_argument("--postfix", "-x", type=str, default="data", help="Name of the output folder for the features (default data)")
    parser.add_argument(
        "--threads", "-t", "--cpu", "-c", dest="cpu", type=int, default=max(1, int(num_cpu * 0.95)),
        help="Number of worker processes (default: 95%% of the CPUs)",
    )
    parser.add_argument(
        "--cfg", dest="config", type=str, default=None,
        help="JSON file of settings, keyed as in the args.json a run writes; -p, -b and -o given here win over it",
    )
    parser.add_argument(
        *_opt("--reuse_blocks_dir"), dest="reuse_blocks_dir", type=str, default=None,
        help="The blocks folder of an earlier run to reuse: its context blocks are not searched for again, and only "
             "their features are written",
    )
    parser.add_argument(
        "--sample", type=int, default=None, help="With --reuse-blocks-dir only: process just this many block files."
    )
    parser.add_argument("--intermediate", type=str, default=None, help="Folder of BAM tables (.feather files) to read instead of the BAM; by default they are made in the output "
             "folder and removed at the end")
    parser.add_argument("--max", dest="max_read_len", type=int, default=2000, help="Longest read kept (default 2000)")
    parser.add_argument("--min", dest="min_read_len", type=int, default=200, help="Shortest read kept (default 200)")
    parser.add_argument("--qcut", "-q", type=int, default=7, help="Skip reads whose mean base quality is below this (default 7)")
    parser.add_argument(
        *_opt("--bam_threads"), dest="bam_threads", type=int, default=4,
        help="BAM decompression threads per BAM parser process (default 4)",
    )
    parser.add_argument(
        *_opt("--process_once"), "-n", dest="process_once", type=int, default=400,
        help="Reads per POD5 processing batch (default 400)",
    )
    parser.add_argument(
        *_opt("--max_token_len"), "-mtl", dest="max_token_len", type=int, default=200, help="Most signal windows in a context block; blocks with more are skipped (default 200)"
    )
    parser.add_argument("--kmer", "-km", type=int, default=5, help="k-mer length; must be odd (default 5)")
    parser.add_argument(
        *_opt("--target_cb_len"), "-tcl", dest="target_cb_len", type=int, default=21,
        help="Context-block length in bases for the features, centred on the site; must be odd (default 21)",
    )
    parser.add_argument(
        "--sampling", "-sp", type=int, default=6, help="Signal samples per move-table step (the stride in the BAM mv tag; default 6)"
    )
    parser.add_argument(
        *_opt("--sig_window"), "-sw", dest="sig_window", type=int, default=5, help="Signal window size; must be odd (default 5)"
    )
    parser.add_argument(*_opt("--dwell_shift"), "-ds", dest="dwell_shift", type=int, default=10, help="Distance in bases between the motor and the pore, for the dwell times (default 10)")
    parser.add_argument(
        *_opt("--padding_mode"), dest="padding_mode", type=str, choices=["none", "zero", "edge"], default="none",
        help="Context blocks that reach past a (trimmed) read end: none skips them, zero pads them with zeros, edge "
             "pads them with the edge value (default none)",
    )
    parser.add_argument(
        *_opt("--no_trim"), "-nt", dest="no_trim", action="store_true", help="Do not trim low-quality read ends"
    )
    parser.add_argument(*_opt("--trim_threshold"), "-tt", dest="trim_threshold", type=int, default=5, help="Base quality that counts as good when trimming (default 5)")
    parser.add_argument(*_opt("--trim_penalty"), "-tp", dest="trim_penalty", type=int, default=-15, help="Score of a base below --trim-threshold when trimming (default -15)")

    g = parser.add_argument_group("extraction (defaults depend on --method)")
    g.add_argument(
        *_opt("--min_score"), "-ms", dest="min_score", type=float, default=None,
        help="Minimum block score " + _defaults_text("min_score"),
    )
    g.add_argument("--it", dest="indel_tolerance", type=int, default=None, help="Most inserted and deleted bases in a context block, in all "
               + _defaults_text("indel_tolerance"))
    g.add_argument("--ip", dest="indel_penalty", type=int, default=None, help="Score penalty per inserted or deleted base " + _defaults_text("indel_penalty"))
    g.add_argument("--cst", dest="cb_size_tolerance", type=int, default=None, help="Most bases by which a context block may be longer or shorter than designed "
               + _defaults_text("cb_size_tolerance"))
    g.add_argument("--amp", dest="anchor_mismatch_penalty", type=int, default=None, help="Score penalty when the anchor base is not found "
               + _defaults_text("anchor_mismatch_penalty"))
    g.add_argument(
        "--aup", dest="anchor_uncertainty_penalty", type=int, default=None,
        help="Score penalty for each extra place the anchor base could be; it changes the block score, not which "
        "blocks are found (0: no penalty) "
        + _defaults_text("anchor_uncertainty_penalty"),
    )
    g.add_argument("--ac", "--boi", dest="anchor_list", type=str, nargs="+", default=None, help="Anchor sequence of each context block in a building block "
               + _defaults_text("anchor_list"))
    g.add_argument("--as", dest="anchor_shift", type=int, nargs="+", default=None,
                   help="Position of the site in each anchor, counted from 0 (default: the last base of the anchor)")
    g.add_argument(
        "--sp", dest="spacer_list", type=str, nargs="+", default=None,
        help="Spacer sequences " + _defaults_text("spacer_list")[:-1] + "; pw: the 15-nt spacer pair, or the pair for two As per block with --di)",
    )
    g.add_argument("--ss", dest="spacer_size", type=int, default=None, help="Spacer length (default: the spacers' length)")
    g.add_argument("--cp", dest="cb_pad", type=int, default=None, help="Bases on each side of the anchor in a context block " + _defaults_text("cb_pad"))
    g.add_argument("--cb", dest="cb_per_bb", type=int, default=None, help="Context blocks per building block " + _defaults_text("cb_per_bb"))
    g.add_argument(
        "--smp", "--smm", dest="spacer_mismatch_penalty", type=int, default=None,
        help="Score penalty per spacer mismatch " + _defaults_text("spacer_mismatch_penalty"),
    )
    g.add_argument("--sa", dest="skip_anchor", action="store_true", default=None, help="Do not check the anchor bases")
    lev = parser.add_argument_group("--method lev only")
    lev.add_argument("--kst", dest="skip_size_tolerance", type=int, default=None, help="Most bases, per context block between them, by which the distance between two spacer matches may "
                     "differ from the design " + _defaults_text("skip_size_tolerance"))
    lev.add_argument("--smt", dest="spacer_mismatch_tolerance", type=int, default=None, help="Most edits in a pair of spacers, in all "
                     + _defaults_text("spacer_mismatch_tolerance"))
    lev.add_argument("--sst", dest="spacer_size_tolerance", type=int, default=None, help="Spacer-length tolerance; no effect in practice, because the --kst check allows no difference between "
                     "adjacent spacers "
                     + _defaults_text("spacer_size_tolerance"))
    pw = parser.add_argument_group("--method pw only")
    pw.add_argument("--sms", dest="spacer_match_score", type=int, default=None, help="Score per matching base in a spacer alignment " + _defaults_text("spacer_match_score"))
    pw.add_argument("--sgo", dest="spacer_gap_open_penalty", type=int, default=None, help="Gap-open penalty in a spacer alignment " + _defaults_text("spacer_gap_open_penalty"))
    pw.add_argument("--sge", dest="spacer_gap_extend_penalty", type=int, default=None, help="Gap-extend penalty in a spacer alignment " + _defaults_text("spacer_gap_extend_penalty"))
    pw.add_argument("--set", dest="spacer_edit_tolerance", type=int, default=None, help="Most edits in a spacer alignment " + _defaults_text("spacer_edit_tolerance"))
    pw.add_argument("--mi", dest="max_iter", type=int, default=None, help="Most search rounds for each spacer in a read " + _defaults_text("max_iter"))
    pw.add_argument(
        "--mai", dest="max_alignments_per_iter", type=int, default=None,
        help="Most spacer alignments used in each search round " + _defaults_text("max_alignments_per_iter"),
    )
    pw.add_argument(
        "--di", dest="di_anchor", action="store_true", default=None,
        help="Two As per context block: each block carries two anchors separated by a designed gap",
    )
    pw.add_argument(
        *_opt("--di_gaps"), dest="di_gap_list", type=int, nargs="+", default=None,
        help=f"Random nt between the two anchors, one per designed type "
             f"(default: {' '.join(map(str, DEFAULT_DI_GAP_LIST))})",
    )
    pw.add_argument(
        *_opt("--di_flanks"), dest="di_flank_list", type=int, nargs="+", default=None,
        help=f"Random nt between spacer and nearest anchor, same order as --di-gaps "
             f"(default: {' '.join(map(str, DEFAULT_DI_FLANK_LIST))})",
    )
    pw.add_argument(
        *_opt("--di_mixed"), dest="di_include_mono", action="store_true", default=None,
        help="Also admit single-anchor (mono-m6A) blocks",
    )
    pw.add_argument(
        *_opt("--di_partial"), dest="di_allow_partial", action="store_true", default=None,
        help="Accept a di block when only one of its two anchors survived basecalling",
    )
    pw.add_argument(
        *_opt("--di_type"), dest="di_type", type=int, default=None, help="The designed type of this run, given as its gap"
    )
    pw.add_argument(
        *_opt("--di_consensus"), dest="di_consensus", default=None, choices=("batch", "read", "none"),
        help="How to resolve the designed type when --di-type is not given (default batch)",
    )
    pw.add_argument(
        *_opt("--di_no_consensus"), dest="di_consistent_type", action="store_false", default=None,
        help="Deprecated alias for --di-consensus none",
    )
    pw.add_argument(
        *_opt("--legacy_bb_period"), dest="true_bb_period", action="store_false", default=None,
        help="With --di: count one spacer, not all of them, in the building-block length, as runs without --di do",
    )
    pw.add_argument(
        *_opt("--edge_blocks"), dest="edge_blocks", action="store_true", default=None,
        help="Also reconstruct the terminal context blocks whose outer spacer was clipped off the read",
    )
    pw.add_argument(
        *_opt("--edge_window"), dest="edge_anchor_window", type=int, default=None,
        help="How far the anchor base may sit from its designed offset in an edge block (default 1)",
    )
    return None


def _resolve_method_defaults(args):
    """Fills every extraction option left unset with the default of the chosen method; refuses options of the other.

    Raises:
        ValueError: An option that only the other method has was given.
    """
    if args.dag_method not in METHOD_DEFAULTS:
        raise ValueError(f"Unknown DAG method: {args.dag_method}, choose from 'pw' and 'lev'.")
    mine = METHOD_DEFAULTS[args.dag_method]
    other = METHOD_DEFAULTS["pw" if args.dag_method == "lev" else "lev"]
    foreign = sorted(k for k in other if k not in mine and getattr(args, k, None) is not None)
    if foreign:
        raise ValueError(f"--method {args.dag_method} does not take: {', '.join(foreign)}")
    for key, value in mine.items():
        if getattr(args, key, None) is None:
            setattr(args, key, list(value) if isinstance(value, list) else value)
    return args


def _merge_config(args):
    """Applies the --cfg file: its keys override options, except -p, -b and -o given on the command line.

    Raises:
        ValueError: A key that is not a setting of this command (keys are the names args.json uses).
    """
    if args.config is None:
        return args
    with open(args.config) as config_file:
        config_dict = json.load(config_file)
    unknown = sorted(key for key in config_dict if not hasattr(args, key))
    if unknown:
        raise ValueError(
            f"--cfg {args.config}: unknown settings {', '.join(unknown)}. Use the names that args.json uses "
            "(for example dag_method, min_score, cpu)."
        )
    for key, value in config_dict.items():
        if key in ("pod5", "bam", "output", "config") and getattr(args, key) is not None:
            continue
        setattr(args, key, value)
    return args


def _check_new_output(args):
    """Refuses an output folder that already holds prep output, and --sample without --reuse-blocks-dir.

    A second run into the same folder would leave the first run's files next to its own, and deeprm train compile
    would read both.
    """
    if args.sample is not None and args.reuse_blocks_dir is None:
        raise ValueError("--sample works only with --reuse-blocks-dir (it limits the number of block files reused).")
    token_dir = os.path.join(args.output, args.postfix)
    old = glob.glob(os.path.join(token_dir, "*.npz"))
    if args.reuse_blocks_dir is None:
        old += glob.glob(os.path.join(args.output, "blocks", "*.pkl"))
    if old:
        raise ValueError(
            f"{args.output} already holds the output of a run ({len(old)} files in data or blocks). Use a new --output "
            "folder, or remove that one first."
        )
    return args


def _join_or_raise(proc_list, label: str) -> None:
    """Join child processes and fail the parent if any child failed."""
    failures = []
    for proc in proc_list:
        proc.join()
        if proc.exitcode != 0:
            failures.append((proc.pid, proc.exitcode))

    if failures:
        for proc in proc_list:
            if proc.is_alive():
                proc.terminate()
        for proc in proc_list:
            if proc.is_alive():
                proc.join(timeout=5)
        raise RuntimeError(f"{label} worker failure(s): {failures}")


def _validate_runtime_args(args) -> None:
    if args.bam_threads <= 0:
        raise ValueError("--bam-threads must be a positive integer")
    if args.cpu <= 0:
        raise ValueError("--cpu/-c must be a positive integer")
    if args.sampling <= 0:
        raise ValueError("--sampling must be a positive integer")
    if args.process_once <= 0:
        raise ValueError("--process_once must be a positive integer")
    if args.kmer <= 0 or args.kmer % 2 == 0:
        raise ValueError("--kmer must be a positive odd integer")
    if args.sig_window <= 0 or args.sig_window % 2 == 0:
        raise ValueError("--sig_window must be a positive odd integer")
    if args.target_cb_len <= 0 or args.target_cb_len % 2 == 0:
        raise ValueError("--target_cb_len must be a positive odd integer")
    if args.max_token_len <= 0:
        raise ValueError("--max_token_len must be a positive integer")
    if not hasattr(args, "max_alignments_per_iter"):
        args.max_alignments_per_iter = 2000
    if args.dag_method == "pw" and not (0.0 <= float(args.min_score) <= 1.0):
        raise ValueError("PW mode emits normalized scores; --min_score must be between 0 and 1")

    _apply_di_defaults(args)

    args.anchor_list = [str(x).upper().replace("U", "T") for x in args.anchor_list]
    args.spacer_list = [str(x).upper().replace("U", "T") for x in args.spacer_list]

    anchor_size = len(args.anchor_list[0])
    if any(len(x) != anchor_size for x in args.anchor_list):
        raise ValueError("All anchors must have the same length")
    if len(args.spacer_list) != args.cb_per_bb + 1:
        raise ValueError(
            f"Expected len(spacer_list) == cb_per_bb + 1; got "
            f"{len(args.spacer_list)} spacers and cb_per_bb={args.cb_per_bb}"
        )
    if len(args.anchor_list) < args.cb_per_bb:
        raise ValueError(
            f"Expected at least cb_per_bb anchors; got {len(args.anchor_list)} anchors "
            f"and cb_per_bb={args.cb_per_bb}"
        )
    if any(len(x) != args.spacer_size for x in args.spacer_list):
        raise ValueError(
            f"All spacers must have length == --ss/--spacer_size ({args.spacer_size}); "
            f"observed lengths={[len(x) for x in args.spacer_list]}"
        )
    if args.indel_tolerance < args.cb_size_tolerance:
        raise ValueError("indel_tolerance must be >= cb_size_tolerance")
    if args.anchor_shift is None:
        args.anchor_shift = [anchor_size - 1 for _ in args.anchor_list]
    elif len(args.anchor_shift) != len(args.anchor_list):
        raise ValueError("anchor_shift length must match anchor_list length")

    if getattr(args, "di_anchor", False):
        if args.dag_method != "pw":
            raise ValueError(
                "--di requires the pairwise extractor (dag_method 'pw'). The "
                "di-m6A library uses 19-nt spacers, which is the pairwise "
                "algorithm family; 'lev' is for designs with 6-nt spacers."
            )
        if args.skip_anchor:
            raise ValueError("--di and --sa are mutually exclusive: di mode "
                             "localises a block by its anchors")
        if len(args.di_gap_list) != len(args.di_flank_list):
            raise ValueError(
                f"--di_gaps and --di_flanks must have the same length; got "
                f"{len(args.di_gap_list)} gaps and {len(args.di_flank_list)} flanks"
            )
        cb_sizes = sorted({2 * f + g + 2 * anchor_size
                           for g, f in zip(args.di_gap_list, args.di_flank_list)})
        bb_sizes = sorted({cb + args.spacer_size * len(args.spacer_list)
                           for cb in cb_sizes})
        if args.di_type is not None and args.di_type not in args.di_gap_list:
            raise ValueError(
                f"--di_type {args.di_type} is not one of the designed gaps "
                f"{args.di_gap_list}"
            )
        scope = ("given (--di_type %s)" % args.di_type if args.di_type is not None
                 else args.di_consensus)
        LOGGER.info(
            f"di-m6A enabled: {len(args.di_gap_list)} types, gaps="
            f"{args.di_gap_list}, flanks={args.di_flank_list}, "
            f"CB sizes={cb_sizes}, BB sizes={bb_sizes}, "
            f"type resolution={scope}, "
            f"include_mono={args.di_include_mono}, partial={args.di_allow_partial}"
        )


## Design constants selected when --ss / --sp are not supplied.
PW_DEFAULT_SPACERS = ["CGACAGAUGAGUUCC", "CCUUGAUAGACAGUC"]   # 15 nt
PW_DEFAULT_SPACER_SIZE = 15


def _apply_di_defaults(args):
    """Resolve the design constants the user did not supply.

    --ss/--sp default to None so "not supplied" is distinguishable from "typed
    the default value"; whatever the user did pass always wins. Without --di
    this restores the defaults for 15-nt spacers and one A per block.
    """
    di = bool(getattr(args, "di_anchor", False))

    # v3 building-block period unless di asks for the corrected one
    if getattr(args, "true_bb_period", None) is None:
        args.true_bb_period = di

    if di:
        if args.di_gap_list is None:
            args.di_gap_list = list(DEFAULT_DI_GAP_LIST)
        if args.di_flank_list is None:
            args.di_flank_list = list(DEFAULT_DI_FLANK_LIST)

    if getattr(args, "spacer_list", None) is None:
        args.spacer_list = (list(DEFAULT_DI_SPACER_LIST) if di
                            else list(PW_DEFAULT_SPACERS))
        LOGGER.info(f"spacer_list not supplied -> {args.spacer_list}")

    if getattr(args, "spacer_size", None) is None:
        args.spacer_size = (len(args.spacer_list[0]) if di
                            else PW_DEFAULT_SPACER_SIZE)
        LOGGER.info(f"spacer_size not supplied -> {args.spacer_size}")


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
        args (argparse.Namespace): Parsed command-line arguments (see add_arguments).

    Returns:
        None
    """

    _merge_config(args)

    missing = []
    if getattr(args, "pod5", None) is None:
        missing.append("--pod5/-p")
    if getattr(args, "bam", None) is None:
        missing.append("--bam/-b")
    if getattr(args, "output", None) is None:
        missing.append("--output/-o")
    if missing:
        raise ValueError(f"Missing required arguments after config merge: {', '.join(missing)}")
    _resolve_method_defaults(args)
    _check_new_output(args)

    Path(args.output).mkdir(parents=True, exist_ok=True)

    ## Make log file ##
    log_path = os.path.join(args.output, "preprocess.log")
    add_log_path(LOGGER, log_path)
    LOGGER.info("Started DeepRM Preprocessing")

    if args.config is not None:
        LOGGER.info(f"Loaded configuration from: {args.config}")
    if not os.path.exists(args.pod5):
        raise FileNotFoundError(f"Input directory {args.pod5} does not exist")
    if not os.path.exists(args.bam):
        raise FileNotFoundError(f"BAM file {args.bam} does not exist")
    _validate_runtime_args(args)
    if args.cb_pad != args.target_cb_len // 2:
        LOGGER.warning(
            f"Extraction cb_pad ({args.cb_pad}) differs from token cb_pad "
            f"({args.target_cb_len // 2}). This is allowed, but make sure it is intentional."
        )

    ## Save command log and arguments ##
    cmd_log_path = os.path.join(args.output, "command.log")
    arg_log_path = os.path.join(args.output, "args.json")
    with open(cmd_log_path, "w") as cmd_log_file:
        cmd_log_file.write(" ".join(sys.argv) + "\n")
    with open(arg_log_path, "w") as arg_log_file:
        json.dump(vars(args), arg_log_file, indent=4)

    dag_config = {
        "indel_tolerance": args.indel_tolerance,
        "indel_penalty": args.indel_penalty,
        "cb_size_tolerance": args.cb_size_tolerance,
        "anchor_mismatch_penalty": args.anchor_mismatch_penalty,
        "anchor_uncertainty_penalty": args.anchor_uncertainty_penalty,   # v3
        "spacer_size": args.spacer_size,
        "cb_pad": args.cb_pad,
        "cb_per_bb": args.cb_per_bb,
        "spacer_mismatch_penalty": args.spacer_mismatch_penalty,
        "anchor_list": args.anchor_list,
        "anchor_shift": args.anchor_shift,
        "spacer_list": args.spacer_list,
        "skip_anchor": args.skip_anchor,
    }

    if args.dag_method == "pw":
        dag_config.update({
            "spacer_edit_tolerance": args.spacer_edit_tolerance,
            "spacer_match_score": args.spacer_match_score,
            "spacer_gap_open_penalty": args.spacer_gap_open_penalty,
            "spacer_gap_extend_penalty": args.spacer_gap_extend_penalty,
            "max_iter": args.max_iter,
            "max_alignments_per_iter": args.max_alignments_per_iter,
            # v4: di-m6A (two anchors per context block)
            "di_anchor": args.di_anchor,
            "di_gap_list": args.di_gap_list,
            "di_flank_list": args.di_flank_list,
            "di_allow_partial": args.di_allow_partial,
            "di_include_mono": args.di_include_mono,
            "di_type": args.di_type,
            "di_consensus": args.di_consensus,
            "di_consistent_type": args.di_consistent_type,
            "true_bb_period": args.true_bb_period,
            # v4: terminal blocks whose outer spacer was clipped off the read
            "edge_blocks": args.edge_blocks,
            "edge_anchor_window": args.edge_anchor_window,
        })
    elif args.dag_method == "lev":
        dag_config.update({
            "skip_size_tolerance": args.skip_size_tolerance,
            "spacer_mismatch_tolerance": args.spacer_mismatch_tolerance,
            "spacer_size_tolerance": args.spacer_size_tolerance,
        })
    else:
        raise ValueError(f"Unknown DAG method: {args.dag_method}, choose from 'pw' and 'lev'.")

    block_output_path = f"{args.output}/blocks"
    token_output_path = f"{args.output}/{args.postfix}"
    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(block_output_path).mkdir(parents=True, exist_ok=True)
    Path(token_output_path).mkdir(parents=True, exist_ok=True)

    norm_factor = get_norm_factor()


    if args.intermediate is None:
        intermediate_dir = f"{args.output}/intermediate_{time.strftime('%Y%m%d%H%M%S')}"
        Path(intermediate_dir).mkdir(parents=True, exist_ok=True)

        n_bam_procs = max(1, args.cpu // args.bam_threads)
        LOGGER.info(f"Using {n_bam_procs} processes to parse BAM")

        proc_list = []
        for pid in range(n_bam_procs):
            out_path = f"{intermediate_dir}/bam_part_{pid}.feather"
            proc = mp.Process(target=parse_bam, args=(pid, n_bam_procs, args.bam_threads, out_path,
                                                      args.bam, args.qcut, args.max_read_len, args.min_read_len,
                                                      args.trim_penalty, args.trim_threshold, args.no_trim, args.sampling))
            proc_list.append(proc)
            proc.start()
        _join_or_raise(proc_list, "BAM parsing")
        gc.collect()

        paths = [f"{intermediate_dir}/bam_part_{pid}.feather" for pid in range(n_bam_procs)]
        missing_paths = [path for path in paths if not os.path.exists(path)]
        if missing_paths:
            raise RuntimeError(f"BAM parser did not create expected Feather file(s): {missing_paths}")

    else:
        intermediate_dir = args.intermediate
        paths = glob.glob(f"{intermediate_dir}/*.feather")

    ## read bam feather table and put it in shared memory
    shm, buf_len = table_to_shared_memory(paths)
    LOGGER.info(f"Saved BAM DataFrame to shared memory {shm.name}, size {buf_len:,} bytes")

    gc.collect()

    mp.set_start_method("spawn", force=True)
    pod5_paths = glob.glob(f"{args.pod5}/*.pod5")


    # Optional: reuse existing block pickles to skip DAG extraction.
    # If --sample is set, only the first N (sorted) pickle files are processed (globally).
    pkl_allowlist = None
    if args.reuse_blocks_dir is not None:
        if not os.path.exists(args.reuse_blocks_dir):
            raise FileNotFoundError(f"reuse_blocks_dir does not exist: {args.reuse_blocks_dir}")
        all_pkls = sorted(glob.glob(os.path.join(args.reuse_blocks_dir, "*.pkl")))
        if args.sample is not None:
            if args.sample <= 0:
                raise ValueError("--sample must be a positive integer")
            all_pkls = all_pkls[: min(args.sample, len(all_pkls))]
        pkl_allowlist = [os.path.splitext(os.path.basename(p))[0] for p in all_pkls]
        LOGGER.info(f"Reusing {len(pkl_allowlist)} block pickles from: {args.reuse_blocks_dir}")


    pod5_sizes = [os.path.getsize(p) for p in pod5_paths]
    pod5_paths_split = [[] for _ in range(args.cpu)]
    sorted_indices = np.argsort(pod5_sizes)[::-1]
    proc_sizes = [0 for _ in range(args.cpu)]
    for idx in sorted_indices:
        min_proc = np.argmin(proc_sizes)
        pod5_paths_split[min_proc].append(pod5_paths[idx])
        proc_sizes[min_proc] += pod5_sizes[idx]
    LOGGER.info(f"Using {args.cpu} processes to parse POD5")


    proc_list = []
    for pid, pod5_paths in enumerate(pod5_paths_split):
        proc = mp.Process(target=segment_normalize_signal,
                          args=(shm.name, buf_len, pod5_paths, norm_factor, block_output_path, token_output_path, dag_config,
                                args.process_once, args.kmer, args.target_cb_len, args.sampling,
                                args.sig_window, args.min_score, args.max_token_len, args.dwell_shift,
                                args.reuse_blocks_dir, pkl_allowlist, args.dag_method, args.padding_mode
                                ))
        proc_list.append(proc)
        proc.start()

    gc.collect()
    try:
        _join_or_raise(proc_list, "POD5 tokenization")
    finally:
        shm.close()
        shm.unlink()

    if args.intermediate is None:
        ## remove flush files
        shutil.rmtree(intermediate_dir)


    LOGGER.info("Finished DeepRM Preprocessing")
    return None
