"""
DeepRM Inference Preprocessing Module (Python)

Segments and normalizes the raw signal of each read (POD5) with its basecalled alignment (BAM),
and writes context-block tokens as compressed .npz chunks for the model. `deeprm call prep` runs
the bundled C++ binary when it can start, and this module when it cannot, when an option that only
this module has is set, or when --pod5 names a single POD5 file. Once the binary has started, its
failure stops the run.

The pipeline has two phases:

1. Phase 1 reads the BAM into an on-disk cache. Reads are filtered (flag mask, mean base
   quality, move table, base of interest) and each read is routed into the cache stream of the
   POD5 file that holds its signal, so phase 2 reads one contiguous span per file. When the BAM
   has a current .bai index and is coordinate-sorted, the workers split the BAM by BGZF virtual
   offset ranges taken from the index, balanced by compressed bytes. Otherwise each worker scans
   the whole BAM and keeps every n-th record.
2. Phase 2 reads the signals from the POD5 files. The POD5 reads are cut into contiguous
   read-range work units ("atoms"), balanced by read count, and the workers take them from a
   shared queue. Each worker joins the signal with the cached alignment, cuts one context block
   per base of interest and writes the rows in chunks (deflate level 4).

Two work metrics, on purpose: phase 1 balances by compressed bytes, phase 2 by reads.

The per-site arithmetic (normalization, segmentation, token slicing), and so the emitted rows,
are those of the preprocessing the DeepRM model was developed with, except for input that
preprocessing mishandled:

* reads mapped to the reverse strand are turned into RNA-sense order;
* reads whose move table stride is not --sampling are skipped;
* a degenerate move table or signal drops the read, and a context block whose signal segments
  do not fit its move table drops the site, instead of failing the run or writing a block cut
  from the wrong place;
* the main process never opens a POD5 file before it starts the workers, so no POD5 file is
  lost to a failed open in a worker.

A POD5 file, or a read in it, that cannot be read fails the run, as in the binary, instead of
being skipped.
"""

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import gc
import glob
import hashlib  # sidecar signatures
import multiprocessing as mp
import os
import queue as _queue
import re
import shutil
import stat
import struct
import threading
import time
import uuid
import zipfile

import numpy as np
import pod5
import pysam

from deeprm.utils.logging import get_logger

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

# ════════════════════════════════════════════════════════════════════════════════════════
# Per-site arithmetic.
# ════════════════════════════════════════════════════════════════════════════════════════

# Per-read record of the on-disk BAM cache: offsets/lengths + small scalars only.
# This layout is the pipeline's memory budget:
#   - 93 B/read, the only RESIDENT per-read state (~2.2 GiB at 23.8 M reads)
#   - inherited by every forked worker
#   - keep it fixed-width and object-free; variable-length data goes in the blobs,
#     addressed by an (offset, length) pair like qoff/qlen
_META_FIELDS = (
    ("pkey0", np.uint64),   # parent read uuid, bytes[0:8]  (searchsorted key)
    ("pkey1", np.uint64),   # parent read uuid, bytes[8:16] (collision check)
    ("rid0", np.int64),     # own read uuid, as written to the npz `read_id` field
    ("rid1", np.int64),
    ("qoff", np.int64),     # offset into the seq/bq blobs (both are qlen long)
    ("qlen", np.int32),
    ("mvoff", np.int64),    # offset into the packed move blob (bytes)
    ("mvlen", np.int32),    # *unpacked* move length, in bits
    ("apoff", np.int64),    # offset into the aligned-pair blob, in pairs
    ("apn", np.int32),
    ("ts", np.int32),
    ("ns", np.int32),
    ("sp", np.int32),
    ("ref", np.int32),
    ("strand", np.int8),
    ("tl", np.int32),       # quality-trim left bound
    ("tr", np.int32),       # quality-trim right bound
)

_BLOBS = ("seq", "bq", "mvp", "ap")

_BLOB_DTYPE = {"seq": np.uint8, "bq": np.int8, "mvp": np.uint8, "ap": np.int32}

# POD5 batches to process per reader instance; see segment_normalize_signal.
_POD5_REOPEN_BATCHES = 32

def validate_geometry(cb_len, kmer_len, sig_window, chunk=None):
    """Reject parameter geometry that cannot emit a row. Raises ValueError; never returns False.

    An EVEN --cb-len or an EVEN --kmer-len would produce zero rows and exit 0: a silent empty
    success that a pipeline reads as a completed run. That is worse than a crash, so both are
    refused up front.

    Why they cannot work:
      * cb_len must be ODD: the context block is centred on the site, so it needs a middle position.
        `cb_half_len = cb_len // 2` and the block spans [c-half, c+half], which is cb_len only when
        cb_len is odd.
      * kmer_len must be ODD: `trim = kmer_len // 2` is removed from BOTH ends, so an even k-mer
        trims asymmetrically and the emitted token arrays no longer match `tok_width =
        cb_len - kmer_len + 1`. Every read then fails the per-read length checks and is dropped.
    """
    if not isinstance(cb_len, int) or isinstance(cb_len, bool):
        raise ValueError(f"--cb-len must be an int (got {type(cb_len).__name__}: {cb_len!r})")
    if cb_len < 3 or cb_len % 2 == 0:
        raise ValueError(
            f"--cb-len must be an ODD integer >= 3 (got {cb_len}). The context block is centred "
            f"on the site, so an even length has no middle position and every read is dropped, "
            f"which would otherwise look like a successful run that emitted nothing.")
    if not isinstance(kmer_len, int) or isinstance(kmer_len, bool):
        raise ValueError(f"--kmer-len must be an int (got {type(kmer_len).__name__}: {kmer_len!r})")
    if kmer_len < 3 or kmer_len % 2 == 0:
        raise ValueError(
            f"--kmer-len must be an ODD integer >= 3 (got {kmer_len}). Even k trims asymmetrically "
            f"(`trim = kmer_len // 2` is removed from BOTH ends) so no read matches tok_width = "
            f"cb_len - kmer_len + 1. k=1 is refused for a different reason: trim becomes 0 and "
            f"`segment_len_arr[trim:-trim]` is `arr[0:-0]` == `arr[0:0]`, an EMPTY slice, so "
            f"token_len is 0 at every site and the run emits ZERO rows while exiting 0.")
    if chunk is not None and (not isinstance(chunk, int) or chunk < 1):
        raise ValueError(
            f"--chunk must be an integer >= 1 (got {chunk!r}). A non-positive chunk makes the "
            f"accumulator never flush, so the run hangs instead of failing.")
    if kmer_len > cb_len:
        raise ValueError(f"--kmer-len ({kmer_len}) cannot exceed --cb-len ({cb_len})")
    if not isinstance(sig_window, int) or isinstance(sig_window, bool):
        raise ValueError(
            f"--sig-window must be an int (got {type(sig_window).__name__}: {sig_window!r})")
    if sig_window < 1 or sig_window % 2 == 0:
        raise ValueError(
            f"--sig-window must be an ODD integer >= 1 (got {sig_window}). The local signal window "
            f"is centred on each base, as in the C++ binary, which refuses an even window. A zero or "
            f"negative window makes the per-position local-signal slice empty, so the emitted blocks "
            f"keep their correct SHAPE while their contents degrade to zeros.")
    if sig_window > kmer_len:
        raise ValueError(
            f"--sig-window ({sig_window}) > --kmer-len ({kmer_len}) produces over-wide signal "
            f"blocks that overflow the fixed sig_width buffer")

def wait_for_processes(proc_list, stage_name):
    """
    Join a list of processes and raise if any subprocess failed.

    The original pipeline only ``join()``-ed the workers, so a worker killed by the OOM
    killer (exitcode ``-9``) or any crash was silently ignored and the run finished with
    truncated output that looked complete. Fail loudly instead.

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
        detail = ", ".join(f"pid={pid} exitcode={code}" for pid, code in failed)
        hint = ""
        if any(code == -9 for _, code in failed):
            hint = (" (exitcode -9 = SIGKILL, typically the OOM killer; lower --thread for fewer workers,"
                    " or raise --bam-thread for fewer BAM workers)")
        raise RuntimeError(f"{stage_name} failed in subprocess(es): {detail}{hint}")

    return None

def quality_trimming_func(bq: np.ndarray, threshold: int = 25, reward: int = 1, penalty: int = -2) -> tuple[int, int]:
    """
    Find a high-quality interior interval using cumulative quality rewards/penalties.

    Args:
        bq (numpy.ndarray): Base-quality scores.
        threshold (int): Quality threshold to count a position as a reward.
        reward (int): Score added for bases meeting the threshold.
        penalty (int): Score added for bases below the threshold.

    Returns:
        tuple[int, int]: Inclusive 5' trim start and exclusive 3' trim end.
    """
    scores = np.where(bq >= threshold, reward, penalty)

    trim_3p_3p = np.argmin(np.concatenate((scores[-1:], np.cumsum(np.flip(scores)))))
    trim_3p_5p = np.argmin(np.concatenate([scores[0:1], np.cumsum(scores[:-trim_3p_3p or None])]))
    trim_5p_5p = np.argmin(np.concatenate([scores[0:1], np.cumsum(scores)]))
    trim_5p_3p = np.argmin(np.concatenate((scores[-1:], np.cumsum(np.flip(scores[trim_5p_5p:])))))

    if (trim_3p_3p + trim_3p_5p) > (trim_5p_3p + trim_5p_5p):
        trim_5p = trim_5p_5p
        trim_3p = trim_5p_3p
    else:
        trim_5p = trim_3p_5p
        trim_3p = trim_3p_3p

    if trim_5p == len(bq) or trim_3p == len(bq):
        return 0, 0

    return trim_5p, len(bq) - trim_3p

def quality_trimming(
        seq: str,
        bq: np.ndarray,
        threshold: int = 25,
        reward: int = 1,
        penalty: int = -2,
        len_threshold: int = 10,
        artifact_cutoff: float = 0.5,
) -> tuple[int, int]:
    """
    Iteratively trim low-quality and artifact-rich ends from a read.

    Args:
        seq (str): Query sequence.
        bq (numpy.ndarray): Base-quality scores.
        threshold (int): Quality threshold for rewarding bases.
        reward (int): Reward for a high-quality base.
        penalty (int): Penalty for a low-quality base.
        len_threshold (int): Minimum acceptable retained length.
        artifact_cutoff (float): Maximum tolerated fraction of GAA-repeat artifact.

    Returns:
        tuple[int, int]: Inclusive 5' trim start and exclusive 3' trim end.
    """
    pat = re.compile(r"(GAA){5,}")

    trim_finish = False
    trim_failed = False
    final_trim_5p = 0
    final_trim_3p = 0

    while not trim_finish:
        trim_5p, trim_3p = quality_trimming_func(bq=bq, threshold=threshold, reward=reward, penalty=penalty)
        if trim_3p - trim_5p <= len_threshold:
            trim_finish = True
            trim_failed = True
        else:
            seq_trimmed = seq[trim_5p:trim_3p]
            artifact_fraction = 0.0
            if len(seq_trimmed) > 0:
                artifact_fraction = sum(x.end() - x.start() for x in re.finditer(pat, seq_trimmed)) / len(seq_trimmed)

            if artifact_fraction < artifact_cutoff:
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
    if padding_mode == 'none':
        if start < left_bound or end > right_bound:
            return None
        return signal_list[start:end]

    clipped_start = max(start, left_bound)
    clipped_end = min(end, right_bound)
    if clipped_start >= clipped_end:
        return None

    pad_left = max(0, left_bound - start)
    pad_right = max(0, end - right_bound)
    return pad_signal(signal_list[clipped_start:clipped_end], pad_left, pad_right, sampling,
                      mode=padding_mode, min_len = min_len)

def slice_scalar_token(arr, start, end, left_bound, right_bound, padding_mode, min_len, constant_values = 0):
    """Slice a scalar token array with optional padding against effective bounds."""
    if padding_mode == 'none':
        if start < left_bound or end > right_bound:
            return None
        return arr[start:end]

    clipped_start = max(start, left_bound)
    clipped_end = min(end, right_bound)
    if clipped_start >= clipped_end:
        return None

    pad_left = max(0, left_bound - start)
    pad_right = max(0, end - right_bound)
    return pad_scalar(arr[clipped_start:clipped_end], pad_left, pad_right, mode=padding_mode, min_len = min_len)

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

    Release guard: a block whose skip lengths are negative or longer than the signal returns None
    (the row is dropped) instead of being sliced from the wrong end. Valid blocks are unchanged.
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

    Release guard: an empty or malformed move table, or a non-finite result, returns None (the
    read is dropped) instead of failing the worker. Valid move tables give the same values.
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

    # q as float32: from numpy 2.4, a Python-float q makes np.quantile place a float32 array's quantile in double
    # precision; older numpy (which the research code ran with) and the binary use single precision. A float32 q gives
    # the single-precision result with every numpy version (on long reads the two differ by up to ~1e-5).
    quantile_a_value = np.quantile(dwell, np.float32(quantile_a))
    quantile_b_value = np.quantile(dwell, np.float32(quantile_b))
    q_shift = max(0.1, shift_mult * (quantile_a_value + quantile_b_value))
    q_scale = max(0.1, scale_mult * (quantile_b_value - quantile_a_value))
    dwell = (dwell - q_shift) / q_scale
    if not np.all(np.isfinite(dwell)):
        return None
    return dwell

# Reads this process dropped because their move table is longer than their signal (each phase-2 worker logs its count).
_MOVE_PAST_SIGNAL = [0]


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

    Release guard: any error, or a non-finite normalized signal, returns None (the read is
    dropped) instead of failing the worker. Valid reads give the same segments.
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
        # A move table longer than the signal (the BAM and the POD5 do not fit each other) puts split
        # points at or before the start, so the segments would belong to other bases: drop the read,
        # as the C++ binary does.
        if move_idx.size and move_idx[-1] <= 0:
            _MOVE_PAST_SIGNAL[0] += 1
            return None
        move_idx = np.flip(move_idx, axis=0)
        signal = np.array_split(signal, move_idx)
        if len(signal) == 0:
            return None
    except Exception:
        return None
    return signal

def _read_pod5_batch(reader, read_ids):
    """
    Read one batch of signals from an already-open POD5 reader.

    Returns the raw signals and the read uuids split into two uint64 key columns, so a
    batch can be resolved against the cache with ``np.searchsorted`` and no per-read
    Python objects.

    Args:
        reader (pod5.Reader): open reader for the file the ids belong to.
        read_ids (list): read ids to fetch.

    Returns:
        tuple(list[numpy.ndarray or Exception], numpy.ndarray, numpy.ndarray): signals, key0, key1.
        A read whose signal alone cannot be read has the error in place of its signal; it fails the
        run only if the BAM uses it (see segment_normalize_v3).

    Raises:
        RuntimeError: a read's id or sample count cannot be read (the caller adds the file's path).

    Release fix: the research code skipped a read it could not read, and the run exited 0 with that
    read missing. This follows the binary instead, which reads the id and sample count of every read
    in every file and fails the run when one cannot be read, and reads the signal only of the reads
    the BAM uses.
    """
    signals = []
    raw = []
    for record in reader.reads(read_ids):
        ## Read both fields before appending either: they are index-aligned downstream,
        ## so appending the signal then failing on the id would desync the whole batch.
        try:
            signal = record.signal
            read_id = record.read_id.bytes
        except Exception as exc:
            ## The binary's POD5 scan reads the id and the sample count of every read, so a read
            ## whose id or sample count cannot be read fails the run there too, used or not.
            name = "(id unreadable)"
            try:
                name = str(record.read_id)
                read_id = record.read_id.bytes
                sum(row.sample_count for row in record.signal_rows)
            except Exception:
                raise RuntimeError(f"cannot read read {name}: {type(exc).__name__}: {exc}") from exc
            ## Without its traceback: that holds this frame and so this list, which would keep the
            ## whole batch in memory until the garbage collector runs.
            signal = exc.with_traceback(None)
        signals.append(signal)
        raw.append(read_id)
    if not signals:
        return [], np.empty(0, np.uint64), np.empty(0, np.uint64)
    keys = np.frombuffer(b"".join(raw), dtype=np.uint64).reshape(-1, 2)
    return signals, np.ascontiguousarray(keys[:, 0]), np.ascontiguousarray(keys[:, 1])

def _uuid_keys(u):
    """Split a uuid into its two uint64 halves and the two int64s stored as `read_id`."""
    raw = np.frombuffer(u.bytes, dtype=np.uint64)
    return (int(raw[0]), int(raw[1])), raw.view(np.int64)

def _emit_read(row, cache, signal, norm_factor, acc,
               cb_len, kmer_len, max_token_len, sampling, dwell_shift, sig_window,
               label_div, padding_mode, use_trimming,
               trim, cb_half_len, min_len,
               quantile_a, quantile_b, shift_mult, scale_mult):
    """
    Turn one (POD5 read, BAM alignment) pair into context-block rows.

    Rows are computed one site at a time and streamed into the chunk accumulator, so memory
    holds one read's segments and one site's slices; nothing scales with batch size or site
    count.

    Per-read work (unpack move, move_to_dwell, normalise_trim_segment_signal) is done once,
    above the site loop. A read whose dwell or signal normalization returns None yields no
    sites.

    Reverse-strand reads arrive here already in RNA-sense order (see parse_bam_v3), so this
    function is the same for both strands.
    """
    seq, bq, mvp, mvlen, ap = cache.arrays(row)
    move = np.unpackbits(mvp, count=mvlen).astype(bool)

    dwell_token = move_to_dwell(move, 0.2, 0.8, 0.5, 1.5, sampling)
    segments = normalise_trim_segment_signal(
        signal, move, int(cache.sp[row]), int(cache.ts[row]), int(cache.ns[row]),
        quantile_a, quantile_b, shift_mult, scale_mult, sampling,
    )
    if segments is None or dwell_token is None:
        return

    strand = int(cache.strand[row])
    ref = int(cache.ref[row])
    q_len = len(seq)
    left_bound = int(cache.tl[row]) if use_trimming else 1
    right_bound = int(cache.tr[row]) if use_trimming else q_len
    rid = (int(cache.rid0[row]), int(cache.rid1[row]))

    for pair in ap:
        q_pos = int(pair[0])
        r_pos = int(pair[1])
        start_pos = q_pos - cb_half_len
        end_pos = q_pos + cb_half_len + 1
        ## Compute in numpy int64 so an extreme --label-div wraps like an int64 column would,
        ## rather than raising OverflowError from a Python int.
        with np.errstate(over="ignore"):
            label_id = (np.int64(ref) * np.int64(label_div) + np.int64(r_pos) + np.int64(1)) * np.int64(strand)

        sig = slice_signal_segments(segments, start_pos, end_pos, left_bound, right_bound,
                                    sampling, padding_mode, min_len)
        motor = slice_scalar_token(dwell_token, start_pos + dwell_shift + trim, end_pos + dwell_shift - trim,
                                   left_bound, right_bound, padding_mode, 1, constant_values=0)
        pore = slice_scalar_token(dwell_token, start_pos + trim, end_pos - trim,
                                  left_bound, right_bound, padding_mode, min_len - trim, constant_values=0)
        bq_tok = slice_scalar_token(bq, start_pos + trim, end_pos - trim,
                                    left_bound, right_bound, padding_mode, min_len - trim, constant_values=0)
        motif = slice_scalar_token(seq, start_pos, end_pos,
                                   left_bound, right_bound, padding_mode, min_len, constant_values=0)

        if sig is None or motor is None or pore is None or bq_tok is None or motif is None:
            continue

        segment_len_arr = create_segment_len_arr(sig, sampling)
        token_len = np.sum(segment_len_arr[trim:-trim])
        if not (
                len(segment_len_arr) == cb_len
                and token_len <= max_token_len
                and token_len > 0
                and len(motor) == cb_len - kmer_len + 1
                and len(pore) == cb_len - kmer_len + 1
                and len(bq_tok) == cb_len - kmer_len + 1
                and len(motif) == cb_len
        ):
            continue

        block = segmented_signal_to_block(sig, segment_len_arr, kmer_len, sampling, sig_window, max_token_len)
        if block is None:
            continue

        segment_len_arr = segment_len_arr[trim:-trim].astype(np.uint16)
        block = block.astype(np.float32)
        bq_tok = np.clip(bq_tok, 0, 60).astype(np.uint8)
        acc.add_row(block, motif, segment_len_arr, motor, pore, bq_tok, label_id, rid)


log = get_logger(__name__)

_CACHE_DIR_ENV = "DEEPRM_PREP_CACHE_DIR"
_KEEP_CACHE_ENV = "DEEPRM_PREP_KEEP_CACHE"
_SUMMARY_ENV = "DEEPRM_PREP_SUMMARY_INDEX"   # dir holding pkey0/pkey1/file_idx/files.txt

# Compression level. np.savez_compressed uses zlib's default (6); 4 gives 99.6% of the ratio for
# 67% of the time because zlib switches to lazy matching at 4. Levels 5-9 buy ~nothing.
_DEFLATE_LEVEL = 4

# Phase-2 atom sizing: total_reads / (n_workers * this). Small enough that the queue self-
# balances despite rows-per-read varying ~2.3x across files; large enough that per-atom
# overhead stays negligible.
_ATOMS_PER_WORKER = 8

# Never split below this many reads per atom: the per-atom pod5 open must be amortised.
_MIN_ATOM_READS = 5000

# ★Route-in-place opens 4 streams (seq/bq/mvp/ap) per GROUP per phase-1 worker. At 99 pod5
# files that is 396 handles and ~394 MiB of buffers per worker; at 508 files it is
# 2,032 handles -- OVER the usual 1024 RLIMIT_NOFILE soft limit -- and ~2 GiB of buffers.
# Three defences, applied together:
#   1. raise RLIMIT_NOFILE to the hard limit at start-up
#   2. small per-stream buffers (the OS page cache does the real buffering)
#   3. cap the group count, bucketing several pod5 files per group if necessary
_STREAM_BUF = 1 << 16          # 64 KiB per stream (was 256 KiB): 4*508*64K = 130 MiB/worker
_MAX_GROUPS = 256              # bucket beyond this; locality is per-bucket, still contiguous


def _max_map_count():
    """The kernel's limit on memory maps per process (vm.max_map_count; 65530 when it cannot be read)."""
    try:
        with open("/proc/sys/vm/max_map_count") as f:
            return int(f.read())
    except (OSError, ValueError):
        return 65530


def _raise_nofile(want):
    """Raise the open-file soft limit toward ``want`` (bounded by the hard limit)."""
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(hard, max(soft, want))
    if target > soft:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            log.info("Raised RLIMIT_NOFILE %d -> %d (hard %s)", soft, target, hard)
        except Exception as exc:
            log.warning("Could not raise RLIMIT_NOFILE from %d (%s)", soft, exc)
    return min(target, hard)


# ════════════════════════════════════════════════════════════════════════════════════════
# Summary index — read_id -> pod5 file
# ════════════════════════════════════════════════════════════════════════════════════════
class _SummaryIndex:
    """
    Sorted uint64 key -> pod5-file-index map, built once from the sequencing summary.

    The summary is ~8.8 GiB of text; parsing it per run is untenable, so it is pre-parsed into
    ~578 MiB of .npy (69 s for 35.7 M rows) and memory-mapped here. Lookup is
    `np.searchsorted`, measured at 0.095 us/read => ~2 s of CPU for a 23.8 M read run.
    """

    def __init__(self, idx_dir):
        self.dir = idx_dir
        self.pkey0 = np.load(os.path.join(idx_dir, "pkey0.npy"), mmap_mode="r")
        self.pkey1 = np.load(os.path.join(idx_dir, "pkey1.npy"), mmap_mode="r")
        self.file_idx = np.load(os.path.join(idx_dir, "file_idx.npy"), mmap_mode="r")
        self.files = [line.strip() for line in open(os.path.join(idx_dir, "files.txt")) if line.strip()]
        self.n_files = len(self.files)

    def route(self, key0, key1):
        """Vectorised: (uint64 halves) -> file index per read, or -1 when unknown."""
        out = np.full(len(key0), -1, dtype=np.int32)
        if len(self) == 0 or len(key0) == 0:
            return out
        # ★Scan the WHOLE equal-pkey0 run, not just the first candidate: a pkey0 duplicate
        # otherwise makes a perfectly valid read unroutable.
        lo = np.searchsorted(self.pkey0, key0, side="left")
        hi = np.searchsorted(self.pkey0, key0, side="right")
        p1 = np.asarray(self.pkey1)
        fi = np.asarray(self.file_idx)
        for j in range(len(key0)):
            a, b = int(lo[j]), int(hi[j])
            if b <= a:
                continue
            if b - a == 1:                       # overwhelmingly the common case
                if p1[a] == key1[j]:
                    out[j] = fi[a]
                continue
            m = np.nonzero(p1[a:b] == key1[j])[0]
            if m.size:
                out[j] = fi[a + int(m[0])]
        return out

    def __len__(self):
        return len(self.pkey0)



def _pod5_set_signature(pod5_paths):
    """
    Cheap identity for a POD5 set: (basename, size, mtime_ns) for every file.

    Used to invalidate a cached index. Same lesson as the record-boundary sidecar and the
    stale-`.bai` hazard: an index silently describing a DIFFERENT set of files is worse than no
    index.
    """
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    for p in sorted(pod5_paths):
        st = os.stat(p)
        h.update(os.path.basename(p).encode())
        h.update(struct.pack("<qq", st.st_size, st.st_mtime_ns))
    return h.hexdigest()


def _fail_on_unreadable_pod5(bad):
    """
    Fail the run on POD5 files that cannot be read, as the binary does: their reads would be missing.

    Args:
        bad (list[tuple[str, str]]): (path, error text) of each such file.
    """
    for path, err in bad:
        log.error("Corrupted POD5 file: %s (%s)", path, err)
    raise RuntimeError(f"{len(bad)} POD5 files could not be read, so their reads would be missing "
                       f"(first: {bad[0][0]}: {bad[0][1]})")


def _index_one_pod5(args_tuple):
    """Worker: write one POD5 file's read-id keys as a shard. Metadata only, no signal.
    Returns (file index, read count, error text or None)."""
    i, path, out_dir = args_tuple
    try:
        with pod5.Reader(path) as r:
            # pod5 raises on a valid file with no reads; the binary reads it as 0 reads.
            ids = list(r.read_ids) if r.batch_count else []
    except Exception as exc:
        return i, 0, f"{type(exc).__name__}: {exc}"
    if ids:
        raw = b"".join(uuid.UUID(str(x)).bytes for x in ids)
        k = np.frombuffer(raw, dtype=np.uint64).reshape(-1, 2)
    else:
        k = np.empty((0, 2), dtype=np.uint64)
    np.save(os.path.join(out_dir, f"k{i}.npy"), k)
    return i, len(ids), None


def _build_index_from_pod5(pod5_paths, out_dir, n_workers):
    """
    Build the read_id -> pod5-file index from POD5 metadata alone.

    ★A sequencing summary is not required: each POD5 knows its own read ids, and `read_ids`
    reads only the footer. This is the default source when no summary/index is supplied, and
    it makes the grouped cache layout usable by anyone with just the POD5 set.
    Parallel across files because a large file's id list costs ~14 s to read.
    """
    tmp = out_dir + ".building"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    t0 = time.time()
    jobs = [(i, p, tmp) for i, p in enumerate(pod5_paths)]
    with mp.Pool(processes=max(1, min(n_workers, len(jobs)))) as pool:
        results = pool.map(_index_one_pod5, jobs)
    bad = [(pod5_paths[i], err) for i, _, err in results if err is not None]
    if bad:
        shutil.rmtree(tmp, ignore_errors=True)
        _fail_on_unreadable_pod5(bad)
    counts = [0] * len(pod5_paths)
    for i, n, _ in results:
        counts[i] = n
    k0l, k1l, fil = [], [], []
    for i in range(len(pod5_paths)):
        k = np.load(os.path.join(tmp, f"k{i}.npy"))
        if len(k) == 0:
            continue
        k0l.append(np.ascontiguousarray(k[:, 0]))
        k1l.append(np.ascontiguousarray(k[:, 1]))
        fil.append(np.full(len(k), i, dtype=np.uint16))
    if not k0l:
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    k0 = np.concatenate(k0l)
    k1 = np.concatenate(k1l)
    fi = np.concatenate(fil)
    o = np.argsort(k0, kind="stable")
    os.makedirs(tmp, exist_ok=True)
    np.save(os.path.join(tmp, "pkey0.npy"), k0[o])
    np.save(os.path.join(tmp, "pkey1.npy"), k1[o])
    np.save(os.path.join(tmp, "file_idx.npy"),
            fi[o].astype(np.uint8) if len(pod5_paths) < 256 else fi[o])
    np.save(os.path.join(tmp, "reads_per_file.npy"), np.array(counts, dtype=np.int64))
    with open(os.path.join(tmp, "files.txt"), "w") as fh:
        fh.write("\n".join(os.path.basename(p) for p in pod5_paths))
    with open(os.path.join(tmp, "pod5.sig"), "w") as fh:
        fh.write(_pod5_set_signature(pod5_paths))
    for i in range(len(pod5_paths)):
        os.remove(os.path.join(tmp, f"k{i}.npy"))
    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(tmp, out_dir)            # atomic: a reader never sees a partial index
    log.info("Built POD5-derived index: %d reads over %d files in %.0f s",
             len(k0), len(pod5_paths), time.time() - t0)
    return out_dir


def _find_sequencing_summary(pod5_paths, explicit=None):
    """Locate the sequencing summary MinKNOW writes during the run.

    ★It is always produced by sequencing, so it is the natural default source for the
    read_id -> pod5-file map: nothing has to be built or shipped alongside the data.
    Layout is `<run>/raw/sequencing_summary_*.txt` beside `<run>/raw/pod5/`, so the pod5
    directory itself and its parent are the two places worth looking.
    """
    if explicit:
        if os.path.exists(explicit):
            return explicit
        log.warning("--summary %s does not exist; falling back.", explicit)
        return None
    if not pod5_paths:
        return None
    d = os.path.dirname(os.path.abspath(pod5_paths[0]))
    for cand_dir in (d, os.path.dirname(d)):
        hits = sorted(glob.glob(os.path.join(cand_dir, "sequencing_summary*.txt")))
        if hits:
            # Largest wins: a resumed run can leave a short truncated summary next to the real one.
            hits.sort(key=lambda f: os.path.getsize(f), reverse=True)
            return hits[0]
    return None


def _uuid_hex_to_keys(hex36):
    """(n, 36) uint8 of dashed uuid TEXT -> (n, 2) uint64, byte-for-byte what `_uuid_keys` makes.

    Vectorised because the alternative is 35.7 M `uuid.UUID(s).bytes` calls: a nibble lookup over
    the whole block is one pass of numpy instead of a Python loop per read.
    """
    lut = np.zeros(256, dtype=np.uint8)
    lut[np.frombuffer(b"0123456789", np.uint8)] = np.arange(10, dtype=np.uint8)
    lut[np.frombuffer(b"abcdef", np.uint8)] = np.arange(10, 16, dtype=np.uint8)
    lut[np.frombuffer(b"ABCDEF", np.uint8)] = np.arange(10, 16, dtype=np.uint8)
    keep = np.ones(36, dtype=bool)
    keep[[8, 13, 18, 23]] = False                      # the four dashes
    nib = lut[hex36[:, keep]]                          # (n, 32) nibbles
    raw = ((nib[:, 0::2] << 4) | nib[:, 1::2])         # (n, 16) bytes
    return np.ascontiguousarray(raw).view(np.uint64)   # native order, as np.frombuffer gives


def _summary_batches(summary_path, block_mb=256):
    """Stream (read_id, filename_pod5) column pairs from the summary.

    ★A cold run on new data is the primary case and the summary is ~26 GB of text, so this uses
    pyarrow's threaded CSV reader over a projected pair of columns rather than a row-wise parse.
    Falls back to pandas where pyarrow is unavailable, which is correct but much slower.
    """
    try:
        import pyarrow as pa
        import pyarrow.csv as pacsv
    except ImportError:
        import pandas as pd
        for ch in pd.read_csv(summary_path, sep="\t", usecols=["read_id", "filename_pod5"],
                              dtype=str, chunksize=4_000_000):
            yield ch["read_id"].to_numpy(), ch["filename_pod5"].to_numpy(), len(ch)
        return
    ro = pacsv.ReadOptions(use_threads=True, block_size=block_mb << 20)
    po = pacsv.ParseOptions(delimiter="\t")
    co = pacsv.ConvertOptions(include_columns=["read_id", "filename_pod5"],
                              column_types={"read_id": pa.string(),
                                            "filename_pod5": pa.string()})
    with pacsv.open_csv(summary_path, read_options=ro, parse_options=po,
                        convert_options=co) as reader:
        for batch in reader:
            yield batch.column("read_id"), batch.column("filename_pod5"), batch.num_rows


def _build_index_from_summary(summary_path, pod5_paths, out_dir, block_mb=256):
    """
    Build the read_id -> pod5-file index from the sequencing summary.

    ★`files.txt` lists the POD5 SET BEING PROCESSED, not the summary's file universe, because
    `file_idx` is an index into that list. A run given a subset of the pod5 files (the identity
    gate uses three) would otherwise route reads to the wrong file entirely. Rows naming a pod5
    outside the set are dropped, which is what makes a subset run correct rather than merely
    smaller.

    Returns out_dir, or None if the summary yielded nothing usable (caller then falls back).
    """
    want = {os.path.basename(p): i for i, p in enumerate(pod5_paths)}
    t0 = time.time()
    tmp = out_dir + ".building"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    k0l, k1l, fil = [], [], []
    counts = [0] * len(pod5_paths)
    n_rows = 0
    try:
        for rid, fn, n in _summary_batches(summary_path, block_mb):
            n_rows += n
            if n == 0:
                continue
            if hasattr(fn, "dictionary_encode"):                  # arrow path
                d = fn.dictionary_encode()
                # Map each distinct filename ONCE, not once per row.
                lut = np.array([want.get(v.as_py(), -1) for v in d.dictionary], dtype=np.int64)
                codes = np.asarray(d.indices.to_numpy(zero_copy_only=False))
                fi_all = np.where(np.asarray(d.indices.is_valid()), lut[codes], -1) \
                    if hasattr(d.indices, "is_valid") else lut[codes]
                rid_np = None
            else:                                                 # pandas fallback
                fi_all = np.array([want.get(v, -1) for v in fn], dtype=np.int64)
                rid_np = rid
            keep = fi_all >= 0
            if not keep.any():
                continue
            if rid_np is None:
                sel = np.flatnonzero(keep)
                import pyarrow as pa
                ids = rid.take(pa.array(sel))
                import pyarrow.compute as pc
                ids = ids.combine_chunks() if hasattr(ids, "combine_chunks") else ids
                widths = np.asarray(pc.binary_length(ids))
                if len(widths) and (widths != 36).any():
                    raise ValueError("read_id column is not 36-char dashed uuid text")
                # `take` above returns a fresh array, so the value buffer starts at offsets[0].
                offs = np.frombuffer(ids.buffers()[1], dtype=np.int32, count=len(ids) + 1)
                start, stop = int(offs[0]), int(offs[-1])
                hex36 = np.frombuffer(ids.buffers()[2], dtype=np.uint8,
                                      count=stop - start, offset=start).reshape(-1, 36)
            else:
                sel = np.flatnonzero(keep)
                hex36 = np.frombuffer("".join(rid_np[sel]).encode(), np.uint8).reshape(-1, 36)
            k = _uuid_hex_to_keys(hex36)
            fi = fi_all[sel].astype(np.uint16)
            k0l.append(np.ascontiguousarray(k[:, 0]))
            k1l.append(np.ascontiguousarray(k[:, 1]))
            fil.append(fi)
            for i, c in zip(*np.unique(fi, return_counts=True)):
                counts[int(i)] += int(c)
    except Exception as exc:                                            # noqa: BLE001
        # A summary that cannot be parsed is not a reason to fail the run: the POD5 files carry
        # the same mapping. Name the file and the reason, then fall back.
        log.warning("Sequencing summary %s unusable (%s: %s); deriving the index from POD5 "
                    "instead.", summary_path, type(exc).__name__, exc)
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    if not k0l:
        log.warning("Sequencing summary %s named none of the %d POD5 files given; deriving the "
                    "index from POD5 instead.", summary_path, len(pod5_paths))
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    k0 = np.concatenate(k0l)
    k1 = np.concatenate(k1l)
    fi = np.concatenate(fil)
    o = np.argsort(k0, kind="stable")
    np.save(os.path.join(tmp, "pkey0.npy"), k0[o])
    np.save(os.path.join(tmp, "pkey1.npy"), k1[o])
    np.save(os.path.join(tmp, "file_idx.npy"),
            fi[o].astype(np.uint8) if len(pod5_paths) < 256 else fi[o])
    np.save(os.path.join(tmp, "reads_per_file.npy"), np.array(counts, dtype=np.int64))
    with open(os.path.join(tmp, "files.txt"), "w") as fh:
        fh.write("\n".join(os.path.basename(p) for p in pod5_paths))
    with open(os.path.join(tmp, "pod5.sig"), "w") as fh:
        fh.write(_pod5_set_signature(pod5_paths))
    # A summary can disagree with the POD5 files: on one 97-file run it listed 76,908,220 rows while
    # the POD5s held 76,446,622 reads, and 22,536 reads present in POD5 were MISSING from it. A read
    # the index cannot route is SKIPPED by phase 1 (see the `fidx < 0` branch), so an incomplete
    # summary silently drops rows that the POD5-derived index would have kept. Verify coverage
    # against POD5 read counts -- footer-only, ~1-3 ms/file -- and fall back rather than lose data.
    # Release fix: counted in child processes, never in this one (see _pod5_read_counts).
    mismatch = []
    p5_counts = _pod5_read_counts(pod5_paths, min(8, os.cpu_count() or 1))
    for i, (pth, (n_p5, err)) in enumerate(zip(pod5_paths, p5_counts)):
        if err is not None:
            log.warning("Cannot read POD5 footer for %s (%s); cannot verify summary coverage, "
                        "so deriving the index from POD5 instead.", os.path.basename(pth), err)
            shutil.rmtree(tmp, ignore_errors=True)
            return None
        if n_p5 != counts[i]:
            mismatch.append((os.path.basename(pth), counts[i], n_p5))
    if mismatch:
        shown = "; ".join(f"{f}: summary {a:,} vs pod5 {b:,}" for f, a, b in mismatch[:3])
        log.warning("Sequencing summary does NOT match the POD5 set for %d of %d files (%s%s). "
                    "Deriving the index from POD5 metadata instead, because an incomplete summary "
                    "silently drops reads.", len(mismatch), len(pod5_paths), shown,
                    ", ..." if len(mismatch) > 3 else "")
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(tmp, out_dir)            # atomic: a reader never sees a partial index
    dt = time.time() - t0
    log.info("Built index from sequencing summary %s: %d of %d rows matched the POD5 set, "
             "%d files, in %.0f s (%.0f MB/s)", os.path.basename(summary_path), len(k0), n_rows,
             len(pod5_paths), dt, os.path.getsize(summary_path) / max(dt, 1e-9) / 1e6)
    return out_dir


def _resolve_index(args, pod5_paths):
    """
    Pick an index source, in precedence order:
      1. --summary-index DIR   (prebuilt; fastest, nothing to do)
      2. a cached index whose pod5.sig still matches
      3. --summary [FILE]      (opt-in; bare --summary auto-discovers beside the POD5 dir)
      4. ★POD5 metadata        (the DEFAULT)
      5. --no-group, or no read in any POD5 file  -> None (ungrouped layout)

    A POD5 file that cannot be read while the index is built fails the run (as in the binary).

    ★POD5 metadata is the default because it is both faster and more trustworthy than the
    sequencing summary, measured on one 97-file run: 48.6 s against 70.8 s for the 26.3 GB
    summary, and the summary disagreed with the POD5s on 96 of 97 files while MISSING 22,536
    reads they contain. The POD5 files ARE the signal being processed, so an index derived from
    them cannot disagree with the data by construction. `--summary` remains available for a run
    whose summary is known good, and is still coverage-checked before it is used.
    """
    if args.no_group:
        log.info("--no-group: using the ungrouped cache layout.")
        return None
    explicit = args.summary_index or os.environ.get(_SUMMARY_ENV)
    sig = _pod5_set_signature(pod5_paths)
    if explicit:
        need = ["pkey0.npy", "pkey1.npy", "file_idx.npy", "files.txt"]
        explicit_sig = os.path.join(explicit, "pod5.sig")
        if not all(os.path.exists(os.path.join(explicit, f)) for f in need):
            log.warning("Summary index at %s is incomplete; will derive one from POD5.", explicit)
        elif os.path.exists(explicit_sig) and open(explicit_sig).read().strip() != sig:
            log.warning("Summary index at %s was made from other POD5 files; will derive one from POD5.", explicit)
        else:
            return explicit
    cache = args.index_cache or os.path.join(args.output, ".deeprm_pod5index")
    sigfile = os.path.join(cache, "pod5.sig")
    if os.path.exists(sigfile) and open(sigfile).read().strip() == sig:
        log.info("Reusing cached POD5-derived index at %s", cache)
        return cache
    if os.path.exists(sigfile):
        log.info("Cached index is stale (POD5 set changed); rebuilding.")
    # ★Only when asked. A summary is never read by default -- see the docstring for why.
    want = getattr(args, "summary", None)
    if want:
        summary = _find_sequencing_summary(pod5_paths, None if want == "auto" else want)
        if summary:
            log.info("Using sequencing summary %s (--summary)", summary)
            built = _build_index_from_summary(summary, pod5_paths, cache)
            if built:
                return built
            # _build_index_from_summary already said why; fall through to POD5.
        else:
            log.warning("--summary given but no summary found; deriving the index from POD5.")
    return _build_index_from_pod5(pod5_paths, cache, args.cpu)


def _load_summary_index(args):
    """Locate and load the summary index, or return None to fall back to the ungrouped layout."""
    idx = os.environ.get(_SUMMARY_ENV) or getattr(args, "summary_index", None)
    if not idx:
        log.info("No summary index given; using the ungrouped cache layout.")
        return None
    need = ["pkey0.npy", "pkey1.npy", "file_idx.npy", "files.txt"]
    if not all(os.path.exists(os.path.join(idx, f)) for f in need):
        log.warning("Summary index at %s is incomplete; using the ungrouped layout.", idx)
        return None
    si = _SummaryIndex(idx)
    log.info("Summary index: %d reads over %d pod5 files", len(si), si.n_files)
    return si


# ════════════════════════════════════════════════════════════════════════════════════════
# Phase-1 partitioning — BGZF virtual-offset ranges from the BAI linear index
# ════════════════════════════════════════════════════════════════════════════════════════
def _bai_linear_offsets(bam_path):
    """
    Parse the `.bai` and return sorted, deduped linear-index virtual offsets.

    These are guaranteed record boundaries. Crucially they are emitted per 16 kb of REFERENCE
    COORDINATE, so their count scales with reference length, not reference count -- which is
    why a 25-reference genome BAM still yields ~50k split points. pysam exposes no accessor
    for the linear index, hence the manual parse.
    """
    bai = bam_path + ".bai"
    if not os.path.exists(bai):
        return None

    def _exact(fh, n, what):
        """Read exactly n bytes or raise. ★A short read used to pass silently: `fh.read(8*n_intv)`
        on a truncated file returns fewer bytes, `np.frombuffer` accepts whatever it gets, and the
        partition is then built from an INCOMPLETE offset list -- a wrong split, not a refusal.
        Truncation that happens to leave an 8-byte-aligned tail was completely invisible."""
        b = fh.read(n)
        if len(b) != n:
            raise ValueError(f"truncated .bai: wanted {n} bytes for {what}, got {len(b)}")
        return b

    try:
        with open(bai, "rb") as fh:
            if _exact(fh, 4, "magic") != b"BAI\x01":
                return None
            (n_ref,) = struct.unpack("<i", _exact(fh, 4, "n_ref"))
            if n_ref < 0:
                raise ValueError(f"negative n_ref {n_ref}")
            offs = []
            for r in range(n_ref):
                (n_bin,) = struct.unpack("<i", _exact(fh, 4, f"n_bin[{r}]"))
                for _ in range(n_bin):
                    _exact(fh, 4, "bin id")
                    (n_chunk,) = struct.unpack("<i", _exact(fh, 4, "n_chunk"))
                    _exact(fh, 16 * n_chunk, "chunks")
                (n_intv,) = struct.unpack("<i", _exact(fh, 4, f"n_intv[{r}]"))
                if n_intv:
                    offs.append(np.frombuffer(_exact(fh, 8 * n_intv, "linear index"), dtype="<u8"))
            # The only thing allowed after the last reference is the optional 8-byte n_no_coor.
            tail = fh.read()
            if len(tail) not in (0, 8):
                raise ValueError(f"{len(tail)} unexpected trailing bytes")
    except Exception as exc:
        log.warning("Rejected %s (%s); falling back to the modulo scan, which is correct and "
                    "merely slower.", bai, exc)
        return None
    if not offs:
        return None
    a = np.unique(np.concatenate(offs))
    return a[a > 0]


def _fast_path_ok(bam_path):
    """
    Decide whether the voffset fast path is safe. Returns (ok, reason).

    Three conditions, all required: the index exists, it is NOT OLDER than the BAM, and the header
    says coordinate-sorted.

    ★The mtime check is the one that actually catches a stale index. The earlier docstring claimed
    the `SO` check did -- it does not: `SO:coordinate` describes the BAM header, so an index left
    over from an EARLIER version of a still-coordinate-sorted file passes it unchanged and then
    mis-partitions silently. Comparing mtimes is what samtools itself warns on, and a wrong
    partition is far worse than falling back to the modulo scan.
    """
    bai = bam_path + ".bai"
    if not os.path.exists(bai):
        if os.path.exists(bam_path + ".csi"):
            return False, "only a .csi index (no BAI linear index)"
        return False, "no .bai index"
    try:
        if os.stat(bai).st_mtime_ns < os.stat(bam_path).st_mtime_ns:
            return False, ("the .bai is OLDER than the BAM (stale index); re-run samtools index, "
                           "or accept the modulo scan")
    except OSError as exc:
        return False, f"cannot stat BAM/.bai ({exc})"
    try:
        with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as b:
            so = (b.header.to_dict().get("HD", {}) or {}).get("SO", "")
    except Exception as exc:
        return False, f"header unreadable ({exc})"
    if so != "coordinate":
        return False, f"header SO={so!r}, not coordinate"
    return True, "ok"


# ────────────────────────────────────────────────────────────────────────────────────────────
# AUTO-SIZING (--auto-size). The constants and the MEASURED table are the calibration runs of
# the model below.
# ────────────────────────────────────────────────────────────────────────────────────────────
# Runtime auto-sizing — choose the phase-2 worker count from the machine's ACTUAL free RAM.
#
# Why
# ---
# Defaults are pinned for reproducible benchmarking, which is right there and wrong on a shared node.
# An older pipeline showed the cost of the wrong answer: it drove a 503 GiB node to 503.2 GiB and
# the OOM killer fired.
#
# The model
# ---------
#     peak_sysused_gib  ≈  BASE + PER_WORKER x n_workers
#
# measured directly, by running the SAME data at two worker counts and changing nothing else
# (9,351,788 cached reads, 99 pod5 files, 2,471,634,869 rows out of BOTH arms):
#
#     workers   peak sysused   peak anon
#        96       117.52         94.66
#        24        44.54         23.03
#     => anon:    0.995 GiB/worker, intercept ~0   (process memory is purely per-worker)
#     => sysused: 1.014 GiB/worker, intercept 20.2 GiB (page cache, slab, kernel)
#
# ★The earlier model's per-worker term rose with reads/worker as `42.9*ln(rpw) - 137.1`. That is
# REFUTED at production scale: 96 workers ran at 97,414 reads/worker for 0.986 GiB each, and 24 workers
# at 389,658 reads/worker for 0.960 GiB each. **4x the reads per worker moved per-worker memory by
# -3 %.** The log term was fitted over 1,711-15,398 reads/worker and does not extend; extrapolating it
# is what made the old rule under-predict by up to 3.1x.
#
# Safety
# ------
# SAFETY is empirical, not theoretical: the largest ratio of measured peak to this model's prediction
# across every run so far needs 2.88x on the per-worker term (a genome run with the record-boundary
# sidecar), so SAFETY=2.9 covers it. Genome
# runs carry a larger dataset-dependent term than the transcriptome calibration exposes, and rather than
# model a term measured on one dataset, the factor absorbs it. ★Re-check `validate_against_measured()`
# whenever a new full run lands; if a ratio exceeds SAFETY, raise SAFETY rather than explaining it away.
#
# This costs nothing where it matters: on a 488 GiB node the rule still permits more workers than
# the 112 cores, so it would not have throttled any production run — while it refuses the older
# pipeline, whose workers needed ~122 GiB each.

GIB = 2 ** 30


KIB = 1024


BASE_GIB = 20.2


PER_WORKER_GIB = 1.014


SAFETY = 2.9


MIN_HEADROOM = 0.10


IDX_B_PER_READ = 97.2


SUMIDX_B_PER_READ = 17.0


META_B_PER_READ = 93 * 1.5


def mem_available_bytes() -> int:
    """MemAvailable — the only figure worth sizing against.

    NOT MemTotal: on a shared node most of MemTotal is page cache belonging to someone else's job,
    and planning against it is planning to evict them.
    """
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * KIB
    raise RuntimeError("MemAvailable not present in /proc/meminfo")


def predict_peak_gib(n_workers: int, n_reads: int = 0, safety: float = SAFETY) -> float:
    """Predicted peak `MemTotal-MemAvailable` for a run of `n_workers` phase-2 workers."""
    # ★Safety multiplies the PER-WORKER term only. The BASE intercept is page cache, slab and
    # kernel -- largely reclaimable, and smaller on a smaller machine -- so inflating it made the
    # rule refuse a 64 GiB node outright instead of simply planning fewer workers. The excess seen
    # on the genome runs scales with workers, not with the baseline, so that is where it belongs.
    fixed = n_reads * (IDX_B_PER_READ + SUMIDX_B_PER_READ) / GIB
    return BASE_GIB + PER_WORKER_GIB * safety * n_workers + fixed


def plan(n_reads: int, n_pod5_files: int, avail_ram: int, n_cores: int,
         safety: float = SAFETY) -> dict:
    """Size a run for the machine it is actually on.

    Returns the plan and the reasoning. `feasible=False` means even one worker does not fit, which
    is a refusal, not a suggestion -- running anyway is how a node ends up at 503.2 of 503 GiB.
    """
    notes, guards = [], []
    reserve = min(max(0.10 * avail_ram, 4 * GIB), 32 * GIB)
    fixed = n_reads * (IDX_B_PER_READ + SUMIDX_B_PER_READ)
    meta_p1 = n_reads * META_B_PER_READ

    budget = avail_ram * (1 - MIN_HEADROOM) - reserve - fixed - BASE_GIB * GIB
    per_worker = PER_WORKER_GIB * GIB * safety
    n_workers = int(budget // per_worker) if budget > 0 else 0

    if n_workers < 1:
        guards.append(
            f"does not fit: {avail_ram/GIB:.0f} GiB available leaves {budget/GIB:.1f} GiB after "
            f"reserve ({reserve/GIB:.0f}) + fixed index ({fixed/GIB:.1f}) + base "
            f"({BASE_GIB:.0f}), and one worker needs {per_worker/GIB:.2f} GiB")
    if n_workers > n_cores:
        notes.append(f"memory allows {n_workers} workers; capped at {n_cores} cores")
    n_workers = max(1, min(n_workers, n_cores))

    # Phase 1 is a different shape: its cost is the retained per-read meta plus a modest per-worker
    # term, and it has never been the binding constraint on these nodes. Report it, do not size on it.
    p1_note = (f"phase-1 retained meta {meta_p1/GIB:.1f} GiB at {n_reads/1e6:.1f} M reads")
    if meta_p1 > avail_ram * 0.5:
        guards.append(f"phase-1 meta alone is {meta_p1/GIB:.0f} GiB of {avail_ram/GIB:.0f} available")

    need_fd = 4 * n_pod5_files + 64
    if need_fd > 1024:
        notes.append(f"needs {need_fd} fds/worker for route-in-place; the run raises RLIMIT_NOFILE itself")

    peak = predict_peak_gib(n_workers, n_reads, safety)
    return dict(
        feasible=not guards, n_workers=n_workers, safety=safety,
        predicted_peak_gib=peak, avail_gib=avail_ram / GIB, reserve_gib=reserve / GIB,
        fixed_index_gib=fixed / GIB, per_worker_gib=per_worker / GIB,
        headroom_gib=avail_ram / GIB - peak, phase1_note=p1_note,
        notes=notes, guards=guards)


MEASURED = [
    ("calib 96w (fit)",            96, 9351788, 117.52),
    ("calib 24w (fit)",            24, 9351788, 44.54),
    ("transcriptome (oos)",        99, 23372838, 129.32),
    ("genome BAI-only (oos)",      97, 43242685, 186.69),
    ("genome +sidecar (oos)",      97, 43242685, 303.43),
]


def validate_against_measured(safety: float = SAFETY) -> list:
    out = []
    for label, w, n_reads, measured in MEASURED:
        pred = predict_peak_gib(w, n_reads, safety)
        raw = predict_peak_gib(w, n_reads, 1.0)
        out.append(dict(label=label, workers=w, predicted_gib=pred, raw_gib=raw,
                        measured_gib=measured, ratio=pred / measured, raw_ratio=raw / measured))
    return out


def _autosize_phase2(args, n_ph2, n_reads, n_pod5_files):
    """Cap the phase-2 worker count at what this machine's FREE memory supports. Opt-in.

    Off by default on purpose: a benchmark pins `--thread` so the comparison isolates the code
    change rather than the machine's mood. On a shared node the pinned value is the wrong default,
    and an older pipeline showed what the wrong answer costs -- 503.2 GiB used of 503, OOM killer.

    Only ever REDUCES the worker count. Sizing up would mean trusting the model to find spare
    capacity, and the model is an empirical envelope, not a mechanism.
    """
    if not getattr(args, "auto_size", False):
        return n_ph2
    avail = mem_available_bytes()
    p = plan(n_reads, n_pod5_files, avail, n_ph2)
    for g in p["guards"]:
        log.warning("Auto-size GUARD: %s", g)
    for n in p["notes"]:
        log.info("Auto-size: %s", n)
    if not p["feasible"]:
        # ★The plan said it does not fit. Running anyway is how a node reaches 503.2 of 503 GiB.
        raise RuntimeError(
            "Auto-size refuses this run: " + "; ".join(p["guards"]) +
            f". Free memory is {avail / 2 ** 30:.0f} GiB. Reduce --thread, use a larger node, or drop "
            f"--auto-size to override (at the risk of the OOM killer).")
    chosen = min(n_ph2, p["n_workers"])
    log.info("Auto-size: MemAvailable %.0f GiB, %d reads -> %d phase-2 workers "
             "(requested %d, predicted peak %.0f GiB, safety %.1f)",
             avail / 2 ** 30, n_reads, chosen, n_ph2, p["predicted_peak_gib"], p["safety"])
    if chosen < n_ph2:
        log.warning("Auto-size REDUCED phase-2 workers %d -> %d to fit %.0f GiB available.",
                    n_ph2, chosen, avail / 2 ** 30)
    return max(1, chosen)


def _max_gap_bytes(offsets):
    """Largest compressed-byte gap between consecutive split points — what floors one worker."""
    cb = (np.asarray(offsets, dtype=np.uint64) >> np.uint64(16)).astype(np.int64)
    return int(np.diff(np.sort(cb)).max()) if len(cb) > 1 else 0


# ────────────────────────────────────────────────────────────────────────────────────────────
# RECORD-BOUNDARY SIDECAR (.rbsc). The reader and the builder both live here, so no
# separate command is needed to produce a sidecar (`--build-sidecar`).
# ────────────────────────────────────────────────────────────────────────────────────────────
# Record-boundary sidecar (`.rbsc`) — validated READ path for the phase-1 partitioner.
#
# Why this exists
# ---------------
# Phase 1 is partitioned by BGZF virtual offsets taken from the BAI linear index. The linear index
# emits one offset per 16 kb of REFERENCE COORDINATE, so a short reference carrying a huge pileup
# gets almost no split points. Measured on one human genome BAM: **chrM is 16,569 bp but holds
# 15,057,580 reads (34 % of mapped) inside about two linear-index windows**, so one worker inherits
# a third of the BAM and phase 1 runs 1.85x SLOWER than a whole-file modulo partition.
#
# The sidecar stores extra, verified record boundaries inside such hotspots. Feeding them to
# `_plan_voffset_ranges(extra=...)` takes the max inter-point gap from 10,649,707,303 B to
# 66,503,829 B and the static byte balance from 6.398 to 1.004 at 26 workers.
#
#     hdr: 8s MAGIC | u32 hdr_len | u16 version | u16 flags
#          u64 bam_size | u64 bam_mtime_ns | u64 bai_size | u64 bai_mtime_ns
#          u32 cap_bytes | u32 stride_bytes
#          u64 n_bai_points | u64 mapped_end_voff
#          16B bam_sig | 16B bai_sig | u64 n_points
#     body: u64[n] voffsets (strictly ascending) | i32[n] witness ref_id | i32[n] witness ref_start
#     trailer: 16B blake2b-128 over everything above
#
# ★A sidecar is an OPTIMISATION, never a source of truth. Every rejection path returns None and
# falls back to BAI-only, which is correct and merely slower. A voffset that is not a real record
# boundary makes htslib fail *inside* the reader and poisons the handle, so a wrong sidecar is far
# worse than no sidecar — which is why validation is layered and why `verify` defaults to checking
# EVERY point (measured 1.85 s for 3,663 points; the cheap signature check samples only 16 MiB and
# cannot catch a single-byte edit).



MAGIC = b"BAMRBSC1"


FMT_VERSION = 1


HDR = struct.Struct("<8sIHHQQQQIIQQ16s16sQ")


_SAMPLE_N = 16


_SAMPLE_LEN = 1 << 20


_PSEUDO_BIN = 37450


class SidecarReject(Exception):
    """The sidecar cannot be trusted for this BAM. Always caught; never propagated to a run."""


def sidecar_path(bam: str) -> str:
    return bam + ".rbsc"


def bam_signature(path: str, size: int = None) -> bytes:
    """blake2b-128 over 16 x 1 MiB samples spread across the file, salted with its size."""
    size = size if size is not None else os.path.getsize(path)
    h = hashlib.blake2b(digest_size=16)
    h.update(struct.pack("<Q", size))
    with open(path, "rb", buffering=0) as fh:
        for k in range(_SAMPLE_N):
            off = 0 if k == 0 else max(0, min(size - _SAMPLE_LEN,
                                              (size - _SAMPLE_LEN) * k // (_SAMPLE_N - 1)))
            fh.seek(off)
            h.update(fh.read(_SAMPLE_LEN))
    return h.digest()


def file_digest(path: str) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb", buffering=0) as fh:
        while True:
            b = fh.read(1 << 22)
            if not b:
                break
            h.update(b)
    return h.digest()


def bai_mapped_end(bai_path: str) -> int:
    """Largest per-reference end voffset from the BAI pseudo-bin — the end of mapped data."""
    with open(bai_path, "rb") as fh:
        buf = fh.read()
    if buf[:4] != b"BAI\x01":
        raise SidecarReject("not a BAI index")
    pos = 4
    (n_ref,) = struct.unpack_from("<i", buf, pos)
    pos += 4
    u_i = struct.Struct("<i").unpack_from
    u_bin = struct.Struct("<Ii").unpack_from
    u_chunk = struct.Struct("<QQ").unpack_from
    end = 0
    for _ in range(n_ref):
        (n_bin,) = u_i(buf, pos)
        pos += 4
        for _ in range(n_bin):
            b, n_chunk = u_bin(buf, pos)
            pos += 8
            if b == _PSEUDO_BIN and n_chunk == 2:
                end = max(end, u_chunk(buf, pos)[1])
            pos += 16 * n_chunk
        (n_intv,) = u_i(buf, pos)
        pos += 4
        pos += 8 * n_intv
    return int(end)


def verify_points(bam: str, pts, rid, pos) -> list:
    """Seek to each point and confirm a record starts exactly there, with that identity.

    A bad offset raises *inside* htslib and leaves the handle unusable — even ``close()`` then
    throws. Rejecting a sidecar must never take the caller down with it, so the handle is
    reopened after any failure and htslib's stderr chatter is silenced for the duration.
    """
    import pysam
    bad = []
    verb = pysam.set_verbosity(0)
    b = None
    try:
        for v, r, p in zip(pts, rid, pos):
            if b is None:
                b = pysam.AlignmentFile(bam, "rb", check_sq=False, threads=1)
            try:
                b.seek(int(v))
                if b.tell() != int(v):
                    bad.append((int(v), "tell mismatch", b.tell()))
                    continue
                rd = next(b)
                if b.tell() == int(v):
                    bad.append((int(v), "no progress", None))
                elif rd.reference_id != int(r) or rd.reference_start != int(p):
                    bad.append((int(v), "witness",
                                (rd.reference_id, rd.reference_start, int(r), int(p))))
            except Exception as exc:
                bad.append((int(v), type(exc).__name__, str(exc)[:80]))
                try:
                    b.close()
                except Exception:
                    pass
                b = None                      # poisoned handle; reopen for the next point
    finally:
        if b is not None:
            try:
                b.close()
            except Exception:
                pass
        pysam.set_verbosity(verb)
    return bad


def read_sidecar(bam: str, linear_offsets, path: str = None, level: int = 3,
                 n_verify: int = 8, seed: int = 0):
    """Validated load. Returns ``(points, meta)`` or raises :class:`SidecarReject`.

    ``linear_offsets`` is the BAI linear-offset array the CALLER will actually partition with.
    Checking the sidecar against that exact array — rather than reparsing the index here — is
    what guarantees the sidecar describes the same split-point universe the partitioner sees.

    Validation layers, cheapest first; every one must pass:
      0. magic, version, declared length, self-integrity digest, strictly ascending points
      1. BAM and BAI size + mtime_ns unchanged; BAI point count and mapped_end unchanged
      2. BAM and BAI content signatures  (samples 16 MiB — will NOT catch a single-byte edit)
      3. seek to points and confirm a record begins there with the recorded identity
         (``level>=3`` checks every point; ``level==2`` checks ``n_verify`` at random)
    """
    path = path or sidecar_path(bam)
    if not os.path.exists(path):
        raise SidecarReject(f"no sidecar at {path}")
    with open(path, "rb") as fh:
        raw = fh.read()
    if len(raw) < HDR.size + 16 or raw[:8] != MAGIC:
        raise SidecarReject("bad magic / too short")
    if hashlib.blake2b(raw[:-16], digest_size=16).digest() != raw[-16:]:
        raise SidecarReject("self-integrity digest mismatch (truncated or corrupt)")
    (_, hlen, ver, _flags, bsz, bmt, isz, imt, cap, stride,
     n_bai, mapped_end, bsig, isig, n) = HDR.unpack_from(raw, 0)
    if ver != FMT_VERSION:
        raise SidecarReject(f"format version {ver} != {FMT_VERSION}")
    if len(raw) != hlen + n * 16 + 16:
        raise SidecarReject(f"length {len(raw)} != expected {hlen + n * 16 + 16}")

    bai = bam + ".bai"
    st, stb = os.stat(bam), os.stat(bai)
    if (st.st_size, st.st_mtime_ns) != (bsz, bmt):
        raise SidecarReject(f"BAM size/mtime changed ({st.st_size},{st.st_mtime_ns}) "
                            f"!= ({bsz},{bmt})")
    if (stb.st_size, stb.st_mtime_ns) != (isz, imt):
        raise SidecarReject("BAI size/mtime changed")
    if linear_offsets is None:
        raise SidecarReject("caller has no BAI linear offsets to check against")
    if len(linear_offsets) != n_bai:
        raise SidecarReject(f"BAI linear point count {len(linear_offsets)} != recorded {n_bai}")
    if bai_mapped_end(bai) != mapped_end:
        raise SidecarReject("BAI mapped_end changed")

    pts = np.frombuffer(raw, np.uint64, n, hlen).copy()
    rid = np.frombuffer(raw, np.int32, n, hlen + 8 * n).copy()
    pos = np.frombuffer(raw, np.int32, n, hlen + 12 * n).copy()
    if n and not np.all(np.diff(pts) > 0):
        raise SidecarReject("points not strictly ascending")

    if bam_signature(bam, st.st_size) != bsig:
        raise SidecarReject("BAM content signature mismatch")
    if file_digest(bai) != isig:
        raise SidecarReject("BAI content signature mismatch")

    if level >= 2 and n:
        idx = (np.arange(n) if level >= 3
               else np.unique(np.random.default_rng(seed).integers(0, n, size=min(n_verify, n))))
        bad = verify_points(bam, pts[idx], rid[idx], pos[idx])
        if bad:
            raise SidecarReject(f"witness check failed on {len(bad)}/{len(idx)} points: {bad[:3]}")

    meta = dict(n_points=int(n), cap=int(cap), stride=int(stride), n_bai_points=int(n_bai),
                mapped_end=int(mapped_end), level=int(level), path=path, bytes=len(raw),
                verified=int(len(idx)) if (level >= 2 and n) else 0)
    return pts, meta


def load_or_none(bam: str, linear_offsets, path: str = None, level: int = 3):
    """`read_sidecar` that never raises: returns ``(points, meta)`` or ``(None, reason)``.

    This is the entry point a pipeline should use. A sidecar is a speed optimisation; failing
    to load one must degrade to the BAI-only partition, which is correct, never abort the run.
    """
    try:
        pts, meta = read_sidecar(bam, linear_offsets, path=path, level=level)
    except SidecarReject as exc:
        return None, str(exc)
    except Exception as exc:                                        # noqa: BLE001
        return None, f"unexpected {type(exc).__name__}: {exc}"
    return pts, meta


DEFAULT_CAP = 32 << 20


DEFAULT_STRIDE = 4 << 20


class Prefetcher:
    """Pull a compressed byte range into the page cache ahead of a serial walker.

    Measured on an NFS4 file system (rsize 1 MiB): ONE sequential reader gets 40 MiB/s, but
    8-32 concurrent readers aggregate to 253-379 MiB/s.  The record chain is inherently
    serial, so the walk cannot itself be split -- but its INPUT can be fetched in
    parallel.  Threads stay within `window` bytes of the walker so page-cache growth is
    bounded regardless of interval size.
    """
    __slots__ = ("path", "c0", "c1", "chunk", "window", "pos", "_next", "_lk", "_cv",
                 "_stop", "_ths", "bytes_read")

    def __init__(self, path, c0, c1, threads=8, chunk=1 << 25, window=4 << 30):
        import threading
        self.path, self.c0, self.c1 = path, c0, c1
        self.chunk, self.window = chunk, window
        self.pos = c0
        self._next = c0
        self._lk = threading.Lock()
        self._cv = threading.Condition(self._lk)
        self._stop = False
        self.bytes_read = 0
        self._ths = [threading.Thread(target=self._run, daemon=True) for _ in range(threads)]

    def _run(self):
        buf = bytearray(self.chunk)
        mv = memoryview(buf)
        with open(self.path, "rb", buffering=0) as fh:
            while True:
                with self._cv:
                    while not self._stop and self._next >= self.pos + self.window:
                        self._cv.wait(0.5)
                    if self._stop or self._next >= self.c1:
                        return
                    off = self._next
                    self._next = min(off + self.chunk, self.c1)
                    n = self._next - off
                fh.seek(off)
                got = fh.readinto(mv[:n])
                with self._lk:
                    self.bytes_read += got or 0

    def start(self):
        for t in self._ths:
            t.start()
        return self

    def advance(self, coffset):
        with self._cv:
            if coffset > self.pos:
                self.pos = coffset
                self._cv.notify_all()

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        for t in self._ths:
            t.join(timeout=10)


def walk_pysam(path, lo_voff, hi_voff, stride, bam_threads=1, want_witness=True,
               max_records=None, prefetch_threads=0, prefetch_window=4 << 30,
               prefetch_chunk=1 << 25):
    """The straightforward walker: `tell()` + `next()` on a pysam AlignmentFile.

    With prefetch_threads > 0, a thread pool pulls the interval into the page cache ahead
    of the cursor.  It only ever READS; it cannot change what the walk sees.
    """
    import pysam
    pf = None
    if prefetch_threads > 0 and hi_voff is not None:
        pf = Prefetcher(path, lo_voff >> 16, (hi_voff >> 16) + 1, threads=prefetch_threads,
                        chunk=prefetch_chunk, window=prefetch_window).start()
    bam = pysam.AlignmentFile(path, "rb", check_sq=False, threads=bam_threads)
    bam.seek(lo_voff)
    points, ref_ids, poss = [], [], []
    n_rec = 0
    last_c = lo_voff >> 16
    v = lo_voff
    while True:
        v = bam.tell()
        if hi_voff is not None and v >= hi_voff:
            break
        if max_records is not None and n_rec >= max_records:
            break
        try:
            rd = next(bam)
        except StopIteration:
            break
        vc = v >> 16
        if vc - last_c >= stride:
            points.append(v)
            last_c = vc
            if want_witness:
                ref_ids.append(rd.reference_id)
                poss.append(rd.reference_start)
        n_rec += 1
        if pf is not None and (n_rec & 0x3FFFF) == 0:
            pf.advance(vc)
    bam.close()
    pf_bytes = 0
    if pf is not None:
        pf.advance(1 << 62)
        pf_bytes = pf.bytes_read
        pf.close()
    return dict(points=points, ref_ids=ref_ids, positions=poss, n_records=n_rec,
                end_voff=v, cbytes=(v >> 16) - (lo_voff >> 16), prefetch_bytes=pf_bytes)


def bai_facts(bam):
    """(split points, mapped_end_voff, gap array) -- the thing the sidecar extends."""
    # `_bai_linear_offsets` refuses a truncated .bai instead of silently partitioning on a short
    # offset list.
    S = _bai_linear_offsets(bam)
    if S is None:
        raise ValueError(f"{bam}: no usable .bai; a sidecar extends the BAI, it cannot replace it")
    mapped_end = int(bai_mapped_end(bam + ".bai"))
    cb = (S >> np.uint64(16)).astype(np.int64)
    gaps = np.diff(np.append(cb, mapped_end >> 16))
    return S, mapped_end, gaps


def write_sidecar(bam, points, ref_ids, positions, cap, stride, n_bai_points, mapped_end,
                  out=None):
    out = out or sidecar_path(bam)
    st, stb = os.stat(bam), os.stat(bam + ".bai")
    pts = np.asarray(points, dtype=np.uint64)
    order = np.argsort(pts, kind="stable")
    pts = pts[order]
    rid = np.asarray(ref_ids, dtype=np.int32)[order]
    pos = np.asarray(positions, dtype=np.int32)[order]
    if len(pts) and not np.all(np.diff(pts) > 0):
        raise ValueError("sidecar points are not strictly ascending after sort")
    head = HDR.pack(MAGIC, HDR.size, FMT_VERSION, 0,
                    st.st_size, st.st_mtime_ns, stb.st_size, stb.st_mtime_ns,
                    cap, stride, n_bai_points, mapped_end,
                    bam_signature(bam, st.st_size), file_digest(bam + ".bai"), len(pts))
    body = head + pts.tobytes() + rid.tobytes() + pos.tobytes()
    tmp = out + f".tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(body)
        fh.write(hashlib.blake2b(body, digest_size=16).digest())
    os.replace(tmp, out)                      # atomic: readers never see a partial file
    return out


def _sidecar_walk_one(args):
    (bam, lo, hi, stride, bam_threads, pf_threads, pf_window) = args
    t = time.time()
    r = walk_pysam(bam, lo, hi, stride, bam_threads=bam_threads,
                           prefetch_threads=pf_threads, prefetch_window=pf_window)
    return dict(lo=lo, hi=hi, wall=time.time() - t, n_records=r["n_records"],
                cbytes=(hi >> 16) - (lo >> 16), points=r["points"],
                ref_ids=r["ref_ids"], positions=r["positions"])


def build_sidecar(bam, cap=DEFAULT_CAP, stride=None, procs=8, bam_threads=8, pf_threads=8,
          pf_window=4 << 30, out=None, verbose=True):
    """Walk every over-cap interval and write the sidecar.  Returns a report dict."""
    import multiprocessing as mp
    stride = stride if stride is not None else min(cap, DEFAULT_STRIDE)
    S, mapped_end, gaps = bai_facts(bam)
    over = np.flatnonzero(gaps > cap)
    tasks = []
    for i in over:
        lo = int(S[i])
        hi = int(S[i + 1]) if i + 1 < len(S) else mapped_end
        tasks.append((bam, lo, hi, stride, bam_threads, pf_threads, pf_window))
    tasks.sort(key=lambda t: -(t[2] >> 16) + (t[1] >> 16))       # longest first (LPT)
    walk_bytes = int(gaps[over].sum())
    if verbose:
        print(f"[build] cap={cap>>20} MiB stride={stride>>20} MiB -> {len(tasks)} intervals, "
              f"{walk_bytes:,} B to walk ({100*walk_bytes/gaps.sum():.1f}% of "
              f"{gaps.sum():,} mapped B); procs={procs} bam_threads={bam_threads} "
              f"prefetch={pf_threads}", flush=True)
    t0 = time.time()
    if procs <= 1 or len(tasks) <= 1:
        res = [_sidecar_walk_one(t) for t in tasks]
    else:
        mp.set_start_method("fork", force=True)
        with mp.Pool(min(procs, len(tasks))) as pool:
            res = pool.map(_sidecar_walk_one, tasks, chunksize=1)
    wall = time.time() - t0

    pts, rid, pos = [], [], []
    for r in res:
        pts += r["points"]
        rid += r["ref_ids"]
        pos += r["positions"]
    n_rec = sum(r["n_records"] for r in res)
    p = write_sidecar(bam, pts, rid, pos, cap, stride, len(S), mapped_end, out=out)
    rep = dict(bam=bam, sidecar=p, cap=cap, stride=stride, procs=procs,
               bam_threads=bam_threads, prefetch_threads=pf_threads,
               n_intervals=len(tasks), walk_bytes=walk_bytes,
               mapped_bytes=int(gaps.sum()), n_bai_points=int(len(S)),
               n_new_points=len(pts), records_walked=n_rec, wall_s=wall,
               sidecar_bytes=os.path.getsize(p),
               per_interval=[{k: v for k, v in r.items()
                              if k in ("lo", "hi", "wall", "n_records", "cbytes")}
                             for r in sorted(res, key=lambda r: -r["cbytes"])[:12]])
    if verbose:
        print(f"[build] {len(pts):,} new points from {n_rec:,} records in {wall:.1f}s "
              f"({walk_bytes/wall/2**20:.0f} MiB/s); sidecar {os.path.getsize(p):,} B", flush=True)
    return rep


# ★A sidecar is worth building when the BAI alone would leave ONE phase-1 worker holding a slice
# far larger than its fair share. That is a MEASURED property of (BAM, worker count), not a
# property of the reference: on one genome BAM the largest BAI gap is 3.9x one worker's share at
# 16 workers and 24.4x at 99; on one transcriptome BAM it is 1.13x at 16 -- nothing to fix -- but
# 7.0x at 99. So "genome vs transcriptome" is the wrong question to ask, and the ratio is the right
# one; a transcriptome BAM run wide enough has the same straggler.
_SIDECAR_MIN_RATIO = 2.0        # build when max gap exceeds this multiple of a worker's fair share
_SIDECAR_MIN_CAP = 8 << 20      # never walk intervals smaller than this, whatever the share is


def _autobuild_sidecar(args, offsets, n_parts):
    """Build the record-boundary sidecar automatically when it would remove a straggler.

    ★No flag needed on a first, cold run. Returns the sidecar path to use, or None to stay on the
    BAI-only partition. Never raises: failing to build one costs speed, never correctness.
    """
    if getattr(args, "no_sidecar", False) or offsets is None or len(offsets) < 2:
        return None
    path = args.sidecar or sidecar_path(args.bam)
    if os.path.exists(path):
        return path                       # already there; _load_sidecar_points validates it
    try:
        mapped_end = int(bai_mapped_end(args.bam + ".bai"))
        cb = (np.asarray(offsets) >> np.uint64(16)).astype(np.int64)
        gaps = np.diff(np.append(cb, mapped_end >> 16))
        total = int(gaps.sum())
        if total <= 0 or n_parts < 1:
            return None
        share = total / n_parts
        ratio = float(gaps.max()) / share
        if ratio < _SIDECAR_MIN_RATIO:
            log.info("No sidecar needed: largest BAI gap is %.2fx one worker's share over %d "
                     "workers (threshold %.1fx).", ratio, n_parts, _SIDECAR_MIN_RATIO)
            return None

        # ★The run's own peak, not the sidecar's: better phase-1 balance keeps more workers busy
        # at once, which is exactly why it is faster AND why it costs memory (one genome run: 186.7
        # -> 303.4 GiB). SAFETY in the sizing model was calibrated ON that with-sidecar run, so a
        # prediction that fits covers this case too. Do not spend memory the machine does not have
        # just to save wall-clock.
        avail = mem_available_bytes() / GIB
        need = predict_peak_gib(n_parts)
        if need > avail:
            log.warning("Sidecar would help (largest gap %.2fx a worker's share) but the run is "
                        "predicted to peak at %.0f GiB with only %.0f GiB available, and better "
                        "balance RAISES peak memory. Staying on the BAI-only partition; pass "
                        "--build-sidecar to override.", ratio, need, avail)
            return None

        # Only intervals bigger than a worker's fair share are worth walking.
        cap = max(int(share), _SIDECAR_MIN_CAP)
        walk = int(gaps[gaps > cap].sum())
        log.info("Building a sidecar: largest BAI gap is %.2fx one worker's share over %d workers; "
                 "walking %.2f GB of %.2f GB mapped (cap %d MiB).",
                 ratio, n_parts, walk / 1e9, total / 1e9, cap >> 20)
        t0 = time.time()
        if not os.access(os.path.dirname(os.path.abspath(args.bam)) or ".", os.W_OK):
            # BAM may sit on read-only storage; keep the artifact with the run instead.
            path = os.path.join(args.output, os.path.basename(args.bam) + ".rbsc")
            log.info("BAM directory is not writable; writing the sidecar to %s", path)
        rep = build_sidecar(args.bam, cap=cap, stride=min(cap, DEFAULT_STRIDE),
                            procs=max(1, min(args.cpu, 8)), out=path, verbose=False)
        log.info("Sidecar built: %s points from %s records in %.1f s -> %s",
                 f"{rep['n_new_points']:,}", f"{rep['records_walked']:,}",
                 time.time() - t0, path)
        args.sidecar = path
        return path
    except Exception as exc:                                            # noqa: BLE001
        log.warning("Automatic sidecar build failed (%s: %s); using the BAI-only partition, "
                    "which is correct and merely slower.", type(exc).__name__, exc)
        return None


def _load_sidecar_points(args, offsets):
    """Load the record-boundary sidecar, or return None. NEVER fatal.

    The sidecar supplies verified record boundaries inside a coordinate hotspot -- the only way to
    subdivide a pileup like chrM, which on one genome BAM packed 34 % of mapped reads into ~2
    linear-index windows and left ONE phase-1 worker holding a third of the BAM.

    Failing to load one costs speed, never correctness: the BAI-only partition is still correct.
    A sidecar that is WRONG, on the other hand, hands the reader a voffset that is not a record
    boundary, which fails inside htslib and poisons the handle -- so every rejection path here
    degrades to BAI-only rather than raising.
    """
    if getattr(args, "no_sidecar", False) or offsets is None:
        return None
    # Every rejection path below degrades to the BAI-only partition, which is correct and merely
    # slower.
    path = args.sidecar or sidecar_path(args.bam)
    if not os.path.exists(path):
        if args.sidecar:
            log.warning("No sidecar at %s; using the BAI-only partition.", path)
        return None
    t0 = time.time()
    pts, meta = load_or_none(args.bam, offsets, path=path, level=args.sidecar_level)
    if pts is None:
        log.warning("Sidecar REJECTED (%s); using the BAI-only partition, which is correct "
                    "and merely slower.", meta)
        return None
    before = _max_gap_bytes(offsets)
    after = _max_gap_bytes(np.unique(np.concatenate([offsets, pts])))
    log.info("Sidecar accepted: %s points from %s (level %d, %d verified, %.1f s); "
             "split points %d -> %d, max gap %.2f -> %.2f GB",
             f"{meta['n_points']:,}", os.path.basename(path), meta["level"], meta["verified"],
             time.time() - t0, len(offsets), len(offsets) + meta["n_points"],
             before / 1e9, after / 1e9)
    return pts


def _plan_voffset_ranges(offsets, n_parts, extra=None):
    """
    Partition split points into ``n_parts`` contiguous [lo, hi) voffset ranges, balanced by
    COMPRESSED BYTES (the phase-1 work metric -- balancing by read count instead left a 1.90x
    worker-time spread).

    ``extra`` optionally supplies additional record-boundary offsets from the hotspot sidecar,
    which is the only way to subdivide a pileup like chrM (16.6 kb of coordinate holding
    10.65 GB, hence only 2 linear-index points).

    A BGZF virtual offset is (block_offset << 16) | within_block, so ``v >> 16`` is the
    compressed file position and is what byte balance must be computed on.
    """
    if offsets is None or len(offsets) < 2:
        return None
    if extra is not None and len(extra):
        offsets = np.unique(np.concatenate([offsets, np.asarray(extra, dtype="<u8")]))
    pos = (offsets >> 16).astype(np.int64)              # compressed byte position
    total = pos[-1] - pos[0]
    if total <= 0:
        return None
    n_parts = max(1, min(int(n_parts), len(offsets) - 1))
    targets = pos[0] + (np.arange(1, n_parts) * total) // n_parts
    cuts = np.searchsorted(pos, targets)
    cuts = np.unique(np.clip(cuts, 1, len(offsets) - 1))
    bounds = [0, *cuts.tolist(), len(offsets) - 1]
    ranges = []
    for i in range(len(bounds) - 1):
        lo, hi = int(offsets[bounds[i]]), int(offsets[bounds[i + 1]])
        if hi > lo:                                      # drop empty blocks (a huge reference
            ranges.append((lo, hi))                      # can straddle a target)
    # the final range runs to EOF: hi=0 means "no upper bound"
    if ranges:
        ranges[-1] = (ranges[-1][0], 0)
    return ranges


# ════════════════════════════════════════════════════════════════════════════════════════
# Phase-1 cache writer — one stream per pod5 file (the locality change)
# ════════════════════════════════════════════════════════════════════════════════════════
class _GroupedShardWriter:
    """
    Streaming writer that routes each read into the stream of ITS pod5 file.

    One stream per worker, sorted later by UUID, would leave a phase-2 worker's rows scattered
    across the whole payload (measured: 387,026 rows for one pod5 file spread over 121.46 GiB,
    4.25 distinct 4 KiB pages per row). Routing here costs 0.095 us/read and no extra I/O, and
    makes each worker's payload contiguous (1.003x read amplification).

    Memory: payload is buffered per group and flushed; only the fixed-width meta row
    (93 B/read) is retained.
    """

    def __init__(self, cache_dir, shard, n_groups, flush_reads=4096):
        self.shard, self.n_groups = shard, n_groups
        self._flush_reads = flush_reads
        self._dir = cache_dir
        self._fh, self._buf, self._pos = {}, {}, {}
        for g in range(n_groups):
            for b in _BLOBS:
                self._fh[(g, b)] = open(os.path.join(cache_dir, f"g{g}.s{shard}.{b}"), "wb",
                                        buffering=_STREAM_BUF)
                self._buf[(g, b)] = []
                self._pos[(g, b)] = 0
        self._meta = {g: np.empty(1 << 12, dtype=np.dtype(list(_META_FIELDS))) for g in range(n_groups)}
        self._n = {g: 0 for g in range(n_groups)}
        self._since = 0

    def add(self, group, pkey, rid, seq, bq, mvp, mvlen, ap, ts, ns, sp, ref, strand, tl, tr):
        g = int(group)
        m, n = self._meta[g], self._n[g]
        if n == len(m):
            grown = np.empty(len(m) * 2, dtype=m.dtype)
            grown[:n] = m
            self._meta[g] = m = grown
        m[n] = (pkey[0], pkey[1], rid[0], rid[1],
                self._pos[(g, "seq")], len(seq),
                self._pos[(g, "mvp")], mvlen,
                self._pos[(g, "ap")] // 2, len(ap),
                ts, ns, sp, ref, strand, tl, tr)
        self._n[g] = n + 1
        for name, arr in (("seq", seq), ("bq", bq), ("mvp", mvp), ("ap", ap.reshape(-1))):
            self._buf[(g, name)].append(arr.tobytes())
            self._pos[(g, name)] += arr.size
        self._since += 1
        if self._since >= self._flush_reads:
            self._flush()

    def _flush(self):
        for k, fh in self._fh.items():
            if self._buf[k]:
                fh.write(b"".join(self._buf[k]))
                self._buf[k].clear()
        self._since = 0

    def close(self):
        self._flush()
        for fh in self._fh.values():
            fh.close()
        for g in range(self.n_groups):
            np.save(os.path.join(self._dir, f"g{g}.s{self.shard}.meta.npy"), self._meta[g][: self._n[g]])
        self._meta = None


class _GroupedBamCache:
    """
    Read-only, fork-safe cache view with a per-pod5-file directory.

    Exposes the interface `_emit_read` reads (`arrays(row)`, `.sp/.ts/.ns/.ref/.strand/
    .tl/.tr/.rid0/.rid1`), plus `rows_for_file(i)` and `lookup_within`, so a phase-2 worker
    reads only its own contiguous span instead of searching the whole index.
    """

    def __init__(self, cache_dir, n_shards, n_groups):
        metas, shard_ids, grp_ids = [], [], []
        for g in range(n_groups):
            for s in range(n_shards):
                p = os.path.join(cache_dir, f"g{g}.s{s}.meta.npy")
                if not os.path.exists(p):
                    continue
                m = np.load(p)
                if len(m) == 0:
                    continue
                metas.append(m)
                shard_ids.append(np.full(len(m), s, dtype=np.int32))
                grp_ids.append(np.full(len(m), g, dtype=np.int32))
        if not metas:
            self._empty(n_groups)
            return
        meta = np.concatenate(metas)
        shard = np.concatenate(shard_ids)
        grp = np.concatenate(grp_ids)
        del metas, shard_ids, grp_ids
        # group-major, then key -- so each pod5 file's rows are contiguous AND searchable
        order = np.lexsort((meta["pkey0"], grp))
        for name, _ in _META_FIELDS:
            setattr(self, name, np.ascontiguousarray(meta[name][order]))
        self.shard = np.ascontiguousarray(shard[order])
        self.group = np.ascontiguousarray(grp[order])
        del meta, shard, grp, order
        self._starts = np.searchsorted(self.group, np.arange(n_groups), side="left")
        self._ends = np.searchsorted(self.group, np.arange(n_groups), side="right")
        self.n_groups = n_groups
        self.blob = {
            b: {(g, s): (np.memmap(p, dtype=_BLOB_DTYPE[b], mode="r")
                         if os.path.getsize(p) else np.empty(0, _BLOB_DTYPE[b]))
                for g in range(n_groups) for s in range(n_shards)
                for p in [os.path.join(cache_dir, f"g{g}.s{s}.{b}")] if os.path.exists(p)}
            for b in _BLOBS
        }
        self.nbytes_on_disk = sum(os.path.getsize(os.path.join(cache_dir, f))
                                  for f in os.listdir(cache_dir) if not f.endswith(".npy"))

    def _empty(self, n_groups):
        for name, dtype in _META_FIELDS:
            setattr(self, name, np.empty(0, dtype))
        self.shard = np.empty(0, np.int32)
        self.group = np.empty(0, np.int32)
        self._starts = np.zeros(n_groups, np.int64)
        self._ends = np.zeros(n_groups, np.int64)
        self.n_groups = n_groups
        self.blob = {b: {} for b in _BLOBS}
        self.nbytes_on_disk = 0

    def __len__(self):
        return len(self.pkey0)

    def rows_for_file(self, gi):
        """Contiguous [start, end) row span belonging to pod5 file ``gi``."""
        return int(self._starts[gi]), int(self._ends[gi])

    def lookup_within(self, gi, key0, key1):
        """Resolve keys inside one file's span. One-to-many, like an index join."""
        lo, hi = self.rows_for_file(gi)
        if hi <= lo or len(key0) == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        sub0 = self.pkey0[lo:hi]
        a = np.searchsorted(sub0, key0, side="left")
        b = np.searchsorted(sub0, key0, side="right")
        cnt = (b - a).astype(np.int64)
        tot = int(cnt.sum())
        if tot == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        src = np.repeat(np.arange(len(key0), dtype=np.int64), cnt)
        st = np.repeat(a.astype(np.int64), cnt)
        within = np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        rows = lo + st + within
        keep = self.pkey1[rows] == key1[src]
        return src[keep], rows[keep]

    def arrays(self, row):
        """(seq, bq, packed_move, move_len, aligned_pairs) for one cached read."""
        k = (int(self.group[row]), int(self.shard[row]))
        qo, ql = int(self.qoff[row]), int(self.qlen[row])
        mo, ml = int(self.mvoff[row]), int(self.mvlen[row])
        ao, an = int(self.apoff[row]), int(self.apn[row])
        nb = (ml + 7) // 8
        return (self.blob["seq"][k][qo: qo + ql],
                self.blob["bq"][k][qo: qo + ql],
                self.blob["mvp"][k][mo: mo + nb],
                ml,
                self.blob["ap"][k][2 * ao: 2 * (ao + an)].reshape(-1, 2))

    def close(self):
        self.blob = {b: {} for b in _BLOBS}

    def __getstate__(self):
        raise TypeError("_GroupedBamCache is fork-shared and must not be pickled")


# ════════════════════════════════════════════════════════════════════════════════════════
# Phase-1 worker
# ════════════════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════════════════
# Streaming input — basecaller output straight into prep, no intermediate BAM
# ════════════════════════════════════════════════════════════════════════════════════════
def _start_bam_stream(args):
    """Start reading the BAM as a forward-only STREAM. Returns (producer or None, _StreamTail); the caller opens
    the AlignmentFile on ``tail.read_end``, inside the try whose finally calls _finish_bam_stream.

    Three forms, and the difference matters for correctness rather than convenience:

      --bam -            read stdin. Whatever is upstream is not ours, so if it dies half way we
                         see a clean EOF and CANNOT tell a complete run from a truncated one.
      --bam <path>       a path that is not a regular file (/dev/stdin, <(command), a named pipe),
                         read as a stream without being opened to look inside first (that would use
                         up its first bytes). As with --bam -, a producer that fails CANNOT be seen.
                         This form opens the path itself, so its copy closes it at the end.
      --bam-cmd '<cmd>'  we SPAWN the producer, so we can wait on it and read its EXIT CODE. A
                         producer that crashes mid-stream fails the run instead of yielding a
                         short cache that phase 2 would turn into plausible partial output.

    Every form also checks the end of the stream for the BGZF end-of-file block (see _StreamTail):
    a stream cut off between two blocks reads cleanly to its end, but lacks that block. A SAM text
    stream has no such block; it must end with a newline (see _is_sam_text).
    """
    import subprocess
    if args.bam_cmd:
        log.info("Streaming: spawning producer -> %s", args.bam_cmd)
        # The command stays in the caller's process group and session, as in the binary: Ctrl-C and a hang-up
        # reach it, and a command that prompts on the terminal (ssh, sudo) can read it.
        proc = subprocess.Popen(args.bam_cmd, shell=True, stdout=subprocess.PIPE, stderr=None, bufsize=0)
        return proc, _StreamTail(proc.stdout.fileno())
    if args.bam not in (None, "", "-"):
        # a path that is not a regular file (/dev/stdin, <(command), a named pipe): opening it to look inside would
        # use up its first bytes
        log.info("Streaming: reading %s (not a regular file) as a stream (an upstream command that fails CANNOT be "
                 "seen; use --bam-cmd if you want a failed producer to fail the run)", args.bam)
        return None, _StreamTail(os.open(args.bam, os.O_RDONLY), own_source=True)
    log.info("Streaming: reading BAM from stdin (an upstream command that fails CANNOT be "
             "seen; use --bam-cmd if you want a failed producer to fail the run)")
    return None, _StreamTail(0)


# The 28-byte empty BGZF block that ends every complete BGZF file (SAM/BAM specification, 4.1.2).
_BGZF_EOF = bytes.fromhex("1f8b08040000000000ff0600424302001b0003000000000000000000")

_SAM_CUT_OFF = (
    "The SAM input does not end with a newline, so it was cut off inside its last record (that record would be "
    "read with a shortened move table or other missing fields, and give wrong rows). Refusing to continue. If the "
    "SAM is complete and only its last newline is missing, add the newline.")


def _is_sam_text(bam):
    """Whether pysam reads this input as SAM text that is not compressed.

    SAM text has no end-of-file marker, and htslib reads a last line without its newline as a whole record,
    so a SAM cut off inside a record (in its move table, for example) reads cleanly to its end. Every SAM
    writer ends each record with a newline, so a last byte that is not a newline means the input was cut off.
    A SAM cut exactly at a line end holds only whole records and cannot be told from a shorter SAM.
    """
    return bam.format == "SAM" and bam.compression == "NONE"


def _refuse_cut_off_sam_file(path):
    """Fail before any work when a SAM text file given with --bam does not end with a newline (see _is_sam_text)."""
    with pysam.AlignmentFile(path, "rb", check_sq=False) as bam:
        if not _is_sam_text(bam):
            return
    with open(path, "rb") as fh:
        fh.seek(-1, os.SEEK_END)
        if fh.read(1) != b"\n":
            raise RuntimeError(_SAM_CUT_OFF)


class _StreamTail:
    """Copies a BAM stream into a pipe that pysam reads, keeping the stream's last 28 bytes.

    pysam does not report whether a stream ended with the BGZF end-of-file block, and a stream cut
    off between two blocks (as a writer that is killed leaves it) reads cleanly to its end. The
    copy lets the run check that block after the last record, and the newline that ends a SAM text.
    """

    def __init__(self, src_fd, own_source=False):
        self.src_fd = src_fd
        self.own_source = own_source
        r, w = os.pipe()
        self.read_end = os.fdopen(r, "rb", buffering=0)
        os.set_blocking(w, False)  # so the copy can stop on the flag even when pysam does not read
        self._w = w
        self.head = b""
        self.tail = b""
        self.error = None
        self.sam_text = False  # set by the caller once pysam has opened the stream (see _is_sam_text)
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._copy, name="bam-stream-copy", daemon=True)
        self.thread.start()

    def _copy(self):
        import select
        try:
            # poll(), not select(): select() refuses a descriptor of 1024 or more (in a process that holds many open
            # files), so the copy would end at once with nothing copied.
            source, pipe = select.poll(), select.poll()
            source.register(self.src_fd, select.POLLIN)
            pipe.register(self._w, select.POLLOUT)
            while not self._stop.is_set():
                if not source.poll(200):
                    continue
                chunk = os.read(self.src_fd, 1 << 20)
                if not chunk:
                    break
                view = memoryview(chunk)
                while view and not self._stop.is_set():
                    try:
                        view = view[os.write(self._w, view):]
                    except BlockingIOError:
                        pipe.poll(200)
                if view:
                    break  # stopped before pysam had the whole chunk
                # only bytes pysam was given count for the checks of the stream's start and end
                if len(self.head) < 4:
                    self.head = (self.head + chunk)[:4]
                self.tail = (self.tail + chunk[-28:])[-28:]
        except BaseException as exc:  # pysam closed its end (a failed run), the source failed, or anything else:
            self.error = exc          # whatever ends the copy early, the stream did not end cleanly
        finally:
            os.close(self._w)
            if self.own_source:
                os.close(self.src_fd)

    def stop(self):
        """Stop copying (after a failure): the pipe pysam reads then ends, so its handle can close."""
        self._stop.set()
        self.thread.join(timeout=5)

    def ended_cleanly(self):
        """No copy error, and a BGZF stream ends with the end-of-file block (a SAM text stream has none)."""
        self.thread.join()
        if self.error is not None:
            return False
        return not self.head.startswith(_BGZF_EOF[:4]) or self.tail == _BGZF_EOF


def _finish_bam_stream(handle, proc, tail, failed):
    """Close the stream and FAIL if it is incomplete: the producer we spawned did not exit cleanly,
    the stream lacks the BGZF end-of-file block, or a SAM text stream does not end with a newline
    (see _is_sam_text). After a failure (``failed``), the producer is
    stopped instead of waited for (SIGTERM to the shell that runs it, then SIGKILL after 30 s), so a
    producer that is still running cannot hang the run; the original error then propagates.

    The order matters after a failure. The copy stops first, then the handle and the command's pipe
    close, and only then is the command stopped: a part of the command that still writes, also one
    that writes when it gets SIGTERM, gets SIGPIPE at once instead of blocking on a full pipe until
    the SIGKILL.

    Two measures keep a close of the handle from waiting on the command, and both are kept because
    each covers a close the other does not. (1) The handle has no decompression threads (the caller
    opens it with threads=1). When the header cannot be read, pysam closes the handle itself, before
    this function runs and while the copy still runs; with a reader thread, that close would wait on
    the pipe for as long as the command keeps it open. (2) Here the copy stops before the handle
    closes, which ends the pipe the handle reads, so this close would not wait even with a reader
    thread."""
    import subprocess
    rc = 0
    stopped = False
    if failed:
        tail.stop()
        stopped = proc is not None and proc.poll() is None  # still running when the run failed
    for f in (handle, tail.read_end):
        if f is None:
            continue
        try:
            f.close()
        except Exception:
            pass
    if proc is not None:
        try:
            proc.stdout.close()
        except Exception:
            pass
        if not failed:
            rc = proc.wait()
        else:
            if stopped:
                proc.terminate()
            try:
                rc = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = proc.wait()
            if stopped:
                log.error("Stopped the BAM producer after the error")
            elif rc != 0:
                log.error("The BAM producer exited %d", rc)
    if failed:
        return
    if rc != 0:
        raise RuntimeError(
            f"Streaming producer exited {rc}. The BAM stream is INCOMPLETE, so the cache holds "
            f"only the records that arrived before it died. Refusing to continue -- a partial "
            f"cache would otherwise produce output that looks like a successful short run.")
    if not tail.ended_cleanly():
        raise RuntimeError(
            "The BAM stream ended without the BGZF end-of-file block, so it was cut off and its last "
            "records are missing. Refusing to continue.")
    if tail.sam_text and not tail.tail.endswith(b"\n"):
        raise RuntimeError(_SAM_CUT_OFF)
    if proc is not None:
        log.info("Streaming producer exited 0.")


def parse_bam_v3(pid, vrange, n_procs, n_thread, cache_dir, bam_path, bq_cutoff, boi,
                 use_trimming, trim_penalty, trim_threshold, summary_dir, n_groups, idx2local=None,
                 bam_handle=None, filter_flag=276, expected_sampling=None):
    """
    Parse this worker's slice of the BAM into the grouped cache.

    ``vrange`` is a (lo, hi) BGZF virtual-offset range, or None to use the modulo scan (every
    worker reads the whole BAM and keeps every ``n_procs``-th record).
    ``bam_handle`` is an already-open AlignmentFile -- the STREAMING mode, where the source is a
    pipe that can only be read forward once, so neither a voffset range nor a modulo scan applies
    (both require seeking or re-reading). One worker consumes the whole stream.
    The per-read body -- filters, base-of-interest selection, quality trimming -- is the same
    in every mode; only which records this worker sees, and which stream they land in, differ.

    Two release fixes are applied here, before a read is cached:

    * ``expected_sampling``: a read whose move table stride differs from it is skipped, because
      the signal is cut at that many samples per move. ``None`` disables the check.
    * Reverse-strand reads (kept when ``filter_flag`` has no 16 bit) are turned into RNA-sense
      order: the base of interest is complemented when the aligned pairs are selected, and the
      sequence is reverse-complemented, the base qualities reversed and the query positions
      mirrored. The signal and the move table are already in RNA-sense order, so the cached read
      is then the same for both strands. Forward reads are unchanged.
    """
    si = _SummaryIndex(summary_dir) if summary_dir else None
    writer = _GroupedShardWriter(cache_dir, pid, n_groups)
    bam = bam_handle if bam_handle is not None else pysam.AlignmentFile(
        bam_path, "rb", check_sq=False, threads=n_thread)
    ref_index = {r: i for i, r in enumerate(bam.references)}
    n_seen = 0
    n_unroutable = 0
    n_stride_mismatch = 0

    if bam_handle is not None:
        it = enumerate(bam)                      # streaming: one forward pass, this worker sees all
    elif vrange is not None:
        lo, hi = vrange
        bam.seek(lo)
        it = _ranged_iter(bam, hi)
    else:
        it = ((i, r) for i, r in enumerate(bam) if i % n_procs == pid)

    for _, read in it:
        n_seen += 1
        # Skip when (flag & filter_flag), the same rule as the C++ binary. Default 276 =
        # unmapped+reverse+secondary.
        if (read.flag & filter_flag) or (not read.has_tag("mv")):
            continue
        try:
            bq = np.array(read.query_qualities, dtype=np.int8)
            if mean_phred(bq) < bq_cutoff:
                continue
        except Exception:
            continue
        # Release fix: the move table must hold a stride and at least one move, and the stride
        # must be --sampling, the step the signal is cut at.
        mv_tag = read.get_tag("mv")
        if len(mv_tag) < 2:
            continue
        if expected_sampling is not None and int(mv_tag[0]) != expected_sampling:
            n_stride_mismatch += 1
            continue
        # Release fix: on a reverse-strand read the base of interest is its complement on the
        # forward reference.
        boi_strand = _BASE_COMPLEMENT.get(boi, boi) if read.is_reverse else boi
        ap = np.array(read.get_aligned_pairs(matches_only=True, with_seq=True), dtype=object)
        ap = ap[ap[:, 2] == boi_strand][:, :2].astype(np.int32)
        if len(ap) == 0:
            continue
        ref = ref_index.get(read.reference_name, -1)
        if ref < 0:
            continue

        read_id = uuid.UUID(str(read.query_name))
        parent_id = uuid.UUID(str(read.get_tag("pi"))) if read.has_tag("pi") else read_id
        ts = read.get_tag("ts") if read.has_tag("ts") else 0
        ns = read.get_tag("ns") if read.has_tag("ns") else 0
        sp = read.get_tag("sp") if read.has_tag("sp") else 0
        mv = np.array(mv_tag[1:], dtype=bool)
        seq_str = read.query_sequence
        seq = np.array(list(seq_str)).view(np.int32).astype(np.uint8)
        strand = -1 if read.is_reverse else 1

        if strand == -1:
            # Release fix: present reverse-strand reads in RNA-sense orientation. The BAM stores
            # SEQ/QUAL of a reverse-strand read reverse-complemented into forward-reference order,
            # and get_aligned_pairs returns forward-reference query positions, while the signal
            # and the move table are in basecalled (RNA-sense 5'->3') order for both strands.
            # So reverse-complement seq, reverse bq and mirror the query positions.
            seq = np.ascontiguousarray(_ASCII_COMPLEMENT[seq][::-1])
            seq_str = seq.tobytes().decode("ascii")
            bq = np.ascontiguousarray(bq[::-1])
            ap[:, 0] = len(seq) - 1 - ap[:, 0]

        if use_trimming:
            tl, tr = quality_trimming(seq_str, bq.astype(np.int32),
                                      penalty=trim_penalty, threshold=trim_threshold)
            if tl == 0 and tr == 0:
                tr = len(seq)
        else:
            tl, tr = 0, len(seq)

        pkey, _ = _uuid_keys(parent_id)
        _, rid = _uuid_keys(read_id)
        if si is not None:
            k0 = np.array([pkey[0]], dtype=np.uint64)
            k1 = np.array([pkey[1]], dtype=np.uint64)
            fidx = int(si.route(k0, k1)[0])
            if fidx < 0:
                # Not in any POD5 supplied to this run => no signal will ever join it. Caching it
                # would only store rows phase 2 never matches; skipping is equivalent and avoids
                # storing rows that are unreachable by construction.
                n_unroutable += 1
                continue
            g = idx2local[fidx]
            if g >= n_groups - 1:
                # routed to the catch-all => the index names a pod5 this run did not supply
                n_unroutable += 1
                continue
        else:
            g = 0
        writer.add(g, pkey, rid, seq, bq, np.packbits(mv), mv.size, ap,
                   int(ts), int(ns), int(sp), int(ref), strand, int(tl), int(tr))

    bam.close()
    writer.close()
    if n_stride_mismatch > 0:
        log.warning(
            "Skipped %d reads in BAM worker %d due to mv-tag stride mismatch with --sampling=%d",
            n_stride_mismatch,
            pid,
            expected_sampling,
        )
    np.save(os.path.join(cache_dir, f"nseen.s{pid}.npy"),
            np.array([n_seen, n_unroutable], dtype=np.int64))
    gc.collect()
    return None


def _ranged_iter(bam, hi):
    """
    Yield records until the first whose START voffset is >= hi.

    The bound is tested with `bam.tell()` BEFORE reading, so a record straddling the boundary
    belongs to exactly one range -- which is what makes the union cover every record once.
    hi == 0 means "to EOF".
    """
    i = 0
    while True:
        if hi and bam.tell() >= hi:
            return
        try:
            read = next(bam)
        except StopIteration:
            return
        yield i, read
        i += 1


# ════════════════════════════════════════════════════════════════════════════════════════
# Phase-2 output
# ════════════════════════════════════════════════════════════════════════════════════════
def _write_npz(path, payload, level=_DEFLATE_LEVEL):
    """
    `np.savez_compressed` at an explicit deflate level.

    numpy hardcodes no level, so it inherits zlib's default (6). Level 4 gets 99.6% of the
    ratio for 67% of the time because zlib switches to lazy matching there. The archive layout
    is unchanged, so `np.load(path)[field]` is unaffected -- only the compressed stream differs.
    """
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=level) as zf:
        for name, arr in payload.items():
            with zf.open(name + ".npy", "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.asanyarray(arr), allow_pickle=False)


class _ChunkAccumulatorV3:
    """
    Preallocated chunk buffers, written at deflate level 4, optionally on a writer thread.

    ★Buffer dtypes ARE the on-disk dtypes: assignment casts silently, so a wrong dtype here
    changes the output file with no error. The dwell tokens are float32 (`move_to_dwell` stays
    float32; `np.quantile` preserves it) -- a float64 buffer shipped once and corrupted them.

    The writer thread only pays off when workers are I/O-blocked (the cold regime); warm and
    core-saturated it collapses to ~2% because it moves CPU rather than reducing it. Queue
    depth is 1 by design: depth 2 costs a third payload buffer for nothing.
    """

    def __init__(self, chunk_size, sig_width, tok_width, kmer_width, out_prefix, threaded=True):
        self.chunk_size, self.n, self.chunk_idx = chunk_size, 0, 0
        self.out_prefix, self.sig_width, self.dropped_wide = out_prefix, sig_width, 0
        self.buf = {
            "signal_token":      np.zeros((chunk_size, sig_width), np.float32),
            "kmer_token":        np.zeros((chunk_size, kmer_width), np.uint8),
            "segment_len_arr":   np.zeros((chunk_size, tok_width), np.uint16),
            "dwell_motor_token": np.zeros((chunk_size, tok_width), np.float32),
            "dwell_pore_token":  np.zeros((chunk_size, tok_width), np.float32),
            "bq_token":          np.zeros((chunk_size, tok_width), np.uint8),
            "label_id":          np.zeros(chunk_size, np.int64),
            "read_id":           np.zeros((chunk_size, 2), np.int64),
        }
        self._q = _queue.Queue(maxsize=1) if threaded else None
        self._err = None
        self._closed = False
        if self._q is not None:
            self._t = threading.Thread(target=self._writer_loop, daemon=True)
            self._t.start()

    def _writer_loop(self):
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            path, payload = item
            try:
                _write_npz(path, payload)
            except Exception as exc:          # surface, never swallow
                self._err = exc
            finally:
                self._q.task_done()

    def add_row(self, signal_token, kmer_token, segment_len_arr, motor, pore, bq_token, label_id, rid):
        if len(signal_token) != self.sig_width:
            self.dropped_wide += 1            # only reachable when sig_window > kmer_len
            return
        if self.n == self.chunk_size:
            self.flush()
        i = self.n
        b = self.buf
        b["signal_token"][i] = signal_token
        b["kmer_token"][i] = kmer_token
        b["segment_len_arr"][i] = segment_len_arr
        b["dwell_motor_token"][i] = motor
        b["dwell_pore_token"][i] = pore
        b["bq_token"][i] = bq_token
        b["label_id"][i] = label_id
        b["read_id"][i, 0], b["read_id"][i, 1] = rid[0], rid[1]
        self.n += 1

    def flush(self, final=False):
        if self._closed:
            raise RuntimeError("flush() after close() -- the chunk would be discarded")
        if self._err:
            raise RuntimeError(f"npz writer thread failed: {self._err}")
        if self.n == 0:
            return
        payload = {k: v[: self.n].copy() for k, v in self.buf.items()}
        name = f"{self.out_prefix}-last.npz" if final else f"{self.out_prefix}-{self.chunk_idx}.npz"
        if self._q is not None:
            self._q.put((name, payload))
        else:
            _write_npz(name, payload)
        self.chunk_idx += 1
        self.n = 0

    def close(self):
        if self._closed:
            return                       # a second close() must not deadlock on the sentinel
        self.flush(final=True)           # ★flush BEFORE marking closed, or this rejects itself
        self._closed = True
        if self._q is not None:
            self._q.put(None)
            # ★Bounded join: if the writer thread died, an unbounded _q.join() hangs forever
            # with no timeout anywhere above it.
            self._t.join(timeout=600)
            if self._t.is_alive():
                raise RuntimeError("npz writer thread did not finish within 600 s")
            if self._err:
                raise RuntimeError(f"npz writer thread failed: {self._err}")


# ════════════════════════════════════════════════════════════════════════════════════════
# Phase-2 worker — pulls contiguous read-range atoms off a shared queue
# ════════════════════════════════════════════════════════════════════════════════════════
def segment_normalize_v3(cache, atom_q, norm_factor, pid, out_dir, cb_len, kmer_len, chunk_size,
                         max_token_len, sampling, dwell_shift, sig_window, process_once,
                         label_div, padding_mode, use_trimming, pod5_paths, threaded_writer,
                         file2group):
    """
    Consume (pod5_file_index, read_lo, read_hi) atoms until the queue is empty.

    One pod5 file per worker would pin the makespan to the largest file. Atoms are CONTIGUOUS
    read ranges -- stride-interleaved slices measured 2.3x slower -- and concurrent readers on
    one pod5 file cost nothing.
    """
    trim = kmer_len // 2
    if 2 * (cb_len // 2) + 1 != cb_len or cb_len - 2 * trim <= 0:
        log.warning("Geometry --cb-len=%d/--kmer-len=%d yields no window; no tokens written.",
                    cb_len, kmer_len)
        return
    cb_half_len = cb_len // 2
    min_len = cb_half_len + 2
    sig_width = (max_token_len + kmer_len - 1) * sampling
    qa, qb = norm_factor["quantile_a"], norm_factor["quantile_b"]
    sm, sc = norm_factor["shift_mult"], norm_factor["scale_mult"]
    acc = _ChunkAccumulatorV3(chunk_size, sig_width, cb_len - 2 * trim, cb_len,
                              f"{out_dir}/{pid}", threaded=threaded_writer)
    ids_cache = {}          # file index -> read_ids, so repeated atoms do not re-list
    n_atoms = 0
    while True:
        try:
            gi, r_lo, r_hi = atom_q.get_nowait()
        except _queue.Empty:
            break               # ★ONLY Empty ends the loop. A bare `except Exception` would
                                # also swallow a manager/connection error and silently truncate
                                # the output while still exiting 0.
        n_atoms += 1
        path = pod5_paths[gi]
        grp = file2group[gi]        # ★file index != group index (ungrouped => every file -> 0)
        # A POD5 file, or a read in it, that cannot be read stops the run here, as in the binary:
        # skipping it would leave its reads out of a run that still exits 0.
        if gi not in ids_cache:
            try:
                with pod5.Reader(path) as reader:
                    ids_cache[gi] = np.array(reader.read_ids)
            except Exception as exc:
                raise RuntimeError(f"Corrupted POD5 file: {path} ({exc})") from exc
        ids = ids_cache[gi][r_lo:r_hi]
        if len(ids) == 0:
            continue
        nb = int(np.ceil(len(ids) / process_once))
        batches = np.array_split(ids, nb)
        # ★ONE open per atom. Reopening per batch-group cost more in NFS open-state locking
        # than the page-cache it was meant to release; the atom is already a bounded slice.
        try:
            reader = pod5.Reader(path)
        except Exception as exc:
            raise RuntimeError(f"Corrupted POD5 file: {path} ({exc})") from exc
        with reader:
            for batch in batches:
                    try:
                        signals, k0, k1 = _read_pod5_batch(reader, list(batch))
                    except Exception as exc:
                        raise RuntimeError(f"Corrupted POD5 file: {path} ({exc})") from exc
                    if not signals:
                        continue
                    src, rows = cache.lookup_within(grp, k0, k1)
                    if len(src) == 0:
                        del signals
                        continue
                    for s_i, row in zip(src, rows):
                        signal = signals[int(s_i)]
                        if isinstance(signal, Exception):
                            # The BAM uses this read, so its rows would be missing. (The binary reads
                            # the signal of these reads only, so an unused read does not stop it.)
                            rid = uuid.UUID(bytes=np.array([k0[s_i], k1[s_i]], dtype=np.uint64).tobytes())
                            raise RuntimeError(f"Corrupted POD5 file: {path} (cannot read the signal of read "
                                               f"{rid}: {signal})") from signal
                        _emit_read(int(row), cache, signal, norm_factor, acc,
                                   cb_len, kmer_len, max_token_len, sampling, dwell_shift,
                                   sig_window, label_div, padding_mode, use_trimming,
                                   trim, cb_half_len, min_len, qa, qb, sm, sc)
                    del signals
    acc.close()
    # Completeness receipt: the parent asserts these sum to the number of atoms dispatched, so a
    # worker that died or stopped early cannot pass as success.
    np.save(os.path.join(out_dir, f".atoms.{pid}.npy"), np.array([n_atoms], dtype=np.int64))
    if acc.dropped_wide:
        log.warning("Worker %d dropped %d over-wide windows (sig_window > kmer_len)",
                    pid, acc.dropped_wide)
    if _MOVE_PAST_SIGNAL[0]:
        log.warning("Worker %d skipped %d reads whose move table is longer than their signal (do the BAM and the "
                    "POD5 files belong together?)", pid, _MOVE_PAST_SIGNAL[0])


def _count_reads_one(path):
    """Worker: (read count, None) of one POD5 file (footer only), or (-1, error text)."""
    try:
        with pod5.Reader(path) as r:
            # pod5 raises on a valid file with no reads; the binary reads it as 0 reads.
            return (int(r.num_reads) if r.batch_count else 0), None
    except Exception as exc:                                            # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}"


def _pod5_read_counts(pod5_paths, n_workers):
    """
    Read counts of POD5 files, each opened in a child process.

    Release fix: the main process must not open a POD5 file before it forks the phase-2 workers.
    pod5's bundled Arrow picks the names of its temporary folders from a random generator that is
    seeded once per process; workers forked after the main process has used it all draw the same
    names, and every open that loses the race fails with "Failed to make temporary directory".
    The research code then logged "Corrupted POD5 file", skipped the file, and still ended with
    exit 0 and part of the rows missing (on the 11-file example, 6 files were lost this way);
    the release fails the run on a file it cannot open, so the race would stop the run instead.
    Pool workers forked before any POD5 open in the main process seed their own generators.
    """
    if not pod5_paths:
        return []
    with mp.Pool(processes=max(1, min(n_workers, len(pod5_paths)))) as pool:
        return pool.map(_count_reads_one, pod5_paths)


def _build_atoms(pod5_paths, n_workers, reads_per_file=None):
    """
    Split each pod5 file into contiguous read-range atoms, largest-first.

    Work metric is READS (R^2 0.9896) not bytes/samples (0.6827) -- phase-2 cost tracks
    aligned A-sites, not signal length. `pod5.Reader.num_reads` costs 1-3 ms/file (footer
    only) and matched the summary's per-file counts on 99/99 files.

    Release fix: the counts are read in child processes (_pod5_read_counts), never in this one.
    A file whose count cannot be read fails the run, as in the binary.
    """
    counts = [int(reads_per_file[i]) if reads_per_file is not None and i < len(reads_per_file) else None
              for i in range(len(pod5_paths))]
    todo = [i for i, c in enumerate(counts) if c is None]
    bad = []
    for i, (n, err) in zip(todo, _pod5_read_counts([pod5_paths[i] for i in todo], n_workers)):
        if n < 0:
            bad.append((pod5_paths[i], err))
        counts[i] = n if n >= 0 else 0
    if bad:
        _fail_on_unreadable_pod5(bad)
    total = sum(counts)
    if total == 0:
        return [], counts
    # ★Floor the atom size. Without it a small run yields ~tens of reads per atom and the
    # per-atom pod5 open dominates -- observed as every worker stuck in D state on
    # nfs_set_open_stateid_locked. An atom must be worth opening a file for.
    atom = max(_MIN_ATOM_READS, total // max(1, n_workers * _ATOMS_PER_WORKER))
    atoms = []
    for i, c in enumerate(counts):
        for lo in range(0, c, atom):
            atoms.append((i, lo, min(lo + atom, c)))
    atoms.sort(key=lambda a: a[2] - a[1], reverse=True)
    return atoms, counts


# ════════════════════════════════════════════════════════════════════════════════════════
def main(args: argparse.Namespace):
    """
    Run the full preprocessing pipeline on parsed arguments (see add_arguments).

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        int: 0 on success. With --validate-sizing, 1 when a measured peak exceeds the model.
    """
    print("Started DeepRM Preprocessing")
    early = _finish_args(args)
    if early is not None:
        return early
    args.boi = args.boi.upper()
    if args.boi == "U":
        args.boi = "T"
    if args.boi not in ["A", "C", "G", "T"]:
        raise ValueError(f"Base of interest {args.boi} not supported, must be one of A,C,G,T")
    if args.chunk < 1:
        raise ValueError("--chunk must be >= 1")
    if args.cb_len < 1 or args.kmer_len < 1:
        raise ValueError("--cb-len and --kmer-len must be >= 1")
    os.makedirs(args.output, exist_ok=True)
    log.info("Started DeepRM Preprocessing")
    norm_factor = get_norm_factor()

    pod5_paths = ([args.pod5] if os.path.isfile(args.pod5)
                  else sorted(glob.glob(os.path.join(args.pod5, "*.pod5"))))
    if not pod5_paths:
        raise FileNotFoundError(f"No POD5 files found in directory: {args.pod5}")

    idx_dir = _resolve_index(args, pod5_paths)
    si = _SummaryIndex(idx_dir) if idx_dir else None
    if si is not None:
        log.info("Index: %d reads over %d pod5 files", len(si), si.n_files)
    # Groups are pod5 files. Order must match pod5_paths, so map by basename.
    idx2local = None
    if si is not None:
        base2g = {os.path.basename(f): i for i, f in enumerate(si.files)}
        missing = [p for p in pod5_paths if os.path.basename(p) not in base2g]
        if missing:
            log.warning("%d pod5 files absent from the index; ungrouped layout.", len(missing))
            si = None
    # One group per pod5 file gives the best locality, but the writer holds 4 streams per
    # group per worker. Bucket if that would exhaust file handles or RAM.
    if si is not None:
        n_files = len(pod5_paths)
        # Budget BOTH sides: each phase-1 worker holds 4 streams per group, and the PARENT
        # later mmaps 4 blobs per (group, shard) for the whole cache -- the larger of the two.
        n_bam_est = max(1, args.cpu // max(1, args.bam_threads))
        # The parent also keeps 2 pipe descriptors per phase-2 worker (at most --thread of them).
        pipes = 2 * args.cpu
        want = max(4 * (n_files + 1), 4 * (n_files + 1) * n_bam_est) + pipes + 128
        avail = _raise_nofile(want)
        # The most groups that fit: the parent keeps 4 memory maps (each with a descriptor) per
        # (group, shard) and the worker pipes, and each phase-1 worker 4 streams per group.
        fit = min(_MAX_GROUPS, (avail - 128 - pipes) // (4 * n_bam_est),
                  (_max_map_count() - 4096) // (4 * n_bam_est), (avail - 64) // 4)
        if fit < 2:
            log.warning("The open-file limit (%d) leaves no room to group %d pod5 files; ungrouped layout.",
                        avail, n_files)
            si = None
    if si is not None:
        bucket = 1 if n_files + 1 <= fit else int(np.ceil(n_files / (fit - 1)))
        if bucket > 1:
            log.warning("Bucketing %d pod5 files %d-per-group (open-file/RAM limit); "
                        "locality is per-bucket.", n_files, bucket)
        n_groups = int(np.ceil(n_files / bucket)) + 1        # +1 = catch-all for unroutable
        # ★The index numbers files its own way and may list files this run does not use. Map
        # index-file -> LOCAL dense group; anything not in this run maps to the catch-all, which
        # the parent then treats as a coverage failure rather than silently dropping.
        local_of_base = {os.path.basename(p): i for i, p in enumerate(pod5_paths)}
        idx2local = np.full(si.n_files, n_groups - 1, dtype=np.int32)
        for j, f in enumerate(si.files):
            b = os.path.basename(f)
            if b in local_of_base:
                idx2local[j] = local_of_base[b] // bucket
    else:
        bucket = 1
        n_groups = 1

    cache_base = os.environ.get(_CACHE_DIR_ENV) or os.path.join(args.output, ".deeprm_bamcache")
    os.makedirs(cache_base, exist_ok=True)
    cache_dir = os.path.join(cache_base, f"run-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    os.makedirs(cache_dir, exist_ok=False)

    mp.set_start_method("fork", force=True)

    # ── phase 1 ──────────────────────────────────────────────────────────────────────────
    n_bam = max(1, args.cpu // max(1, args.bam_threads))
    # a path that is not a regular file (/dev/stdin, <(command), a named pipe) is a stream too
    streaming = (args.bam == "-") or bool(args.bam_cmd) or not stat.S_ISREG(os.stat(args.bam).st_mode)
    ok, why = (False, "streaming input (no index, single forward pass)") if streaming \
        else _fast_path_ok(args.bam)
    offsets = _bai_linear_offsets(args.bam) if ok else None
    _autobuild_sidecar(args, offsets, n_bam)      # ★no flag needed on a cold run
    extra = _load_sidecar_points(args, offsets)
    ranges = _plan_voffset_ranges(offsets, n_bam, extra=extra) if ok else None
    # ★A short reference set gives too few split points to fill the workers; the voffset path then
    # runs on ~1 worker and is SLOWER than modulo (measured 3.16x). Use whichever actually
    # parallelises.
    if n_bam == 1:
        ranges = None     # one worker reads the whole BAM: there is nothing to partition
    if ranges and len(ranges) < max(2, n_bam // 2):
        log.warning("Phase 1: only %d usable ranges for %d workers (short reference set); "
                    "using the modulo scan instead, which parallelises by record.",
                    len(ranges), n_bam)
        ranges = None
    if ranges:
        log.info("Phase 1: voffset partition, %d ranges over %d workers", len(ranges), n_bam)
        n_bam = len(ranges)
    else:
        if streaming:
            pass          # the streaming branch logs its own, accurate line below
        elif n_bam == 1:
            log.info("Phase 1: one BAM worker reads the whole BAM")
        else:
            log.warning("Phase 1: falling back to modulo scan (%s)",
                        why if not ok else "too few split points")

    if streaming:
        # ★One worker, in THIS process. A pipe is forward-only and single-consumer, so there is no
        # partition to hand out -- forking N workers would just have N-1 of them read nothing.
        # It writes the SAME grouped shard layout phase 2 already consumes, so phase 2 is untouched
        # and the emitted rows are identical by construction rather than by re-verification.
        log.info("Phase 1: STREAMING (1 worker, no partition); pipe is forward-only")
        producer, tail = _start_bam_stream(args)
        handle, failed = None, True
        try:
            # No decompression threads on a stream, so no close can wait on the command (see _finish_bam_stream).
            # Parsing the records in Python is the slow step here, so one decompression thread costs little.
            handle = pysam.AlignmentFile(tail.read_end, "rb", check_sq=False, threads=1)
            tail.sam_text = _is_sam_text(handle)
            # n_thread (1) is not used: the handle is already open
            parse_bam_v3(0, None, 1, 1, cache_dir, args.bam, args.qcut,
                         args.boi, args.trim, args.trim_penalty, args.trim_threshold,
                         si.dir if si is not None else None, n_groups, idx2local,
                         bam_handle=handle, filter_flag=args.filter_flag,
                         expected_sampling=args.sampling)
            failed = False
        finally:
            _finish_bam_stream(handle, producer, tail, failed)
        n_bam = 1
        procs = []
    else:
        procs = []
        for pid in range(n_bam):
            vr = ranges[pid] if ranges else None
            p = mp.Process(target=parse_bam_v3,
                           args=(pid, vr, n_bam, args.bam_threads, cache_dir, args.bam, args.qcut,
                                 args.boi, args.trim, args.trim_penalty, args.trim_threshold,
                                 si.dir if si is not None else None, n_groups, idx2local,
                                 None, args.filter_flag, args.sampling))
            procs.append(p)
            p.start()
        wait_for_processes(procs, "BAM parsing")

    _ns = [np.load(os.path.join(cache_dir, f)) for f in os.listdir(cache_dir) if f.startswith("nseen.")]
    n_seen = int(sum(int(a[0]) for a in _ns))
    n_unroutable = int(sum(int(a[1]) for a in _ns if a.size > 1))
    if n_unroutable:
        # Expected whenever the BAM spans more pod5 files than were supplied. Reported, never
        # silent -- the original defect was that these were cached and then dropped unnoticed.
        # Half or more points at a wrong --pod5 folder or another POD5 set, not at a partial one.
        (log.warning if 2 * n_unroutable >= n_seen else log.info)(
            "Skipped %d of %d BAM records with no POD5 in this run (%.1f%%); "
            "they have no signal to join.", n_unroutable, n_seen, 100.0 * n_unroutable / max(1, n_seen))
    cache = _GroupedBamCache(cache_dir, n_shards=n_bam, n_groups=n_groups)
    log.info("Indexed %d BAM records (n_seen=%d, %.1f GiB cache)",
             len(cache), n_seen, cache.nbytes_on_disk / 2 ** 30)
    gc.collect()

    # ── phase 2 ──────────────────────────────────────────────────────────────────────────
    n_ph2 = min(args.cpu, max(1, len(pod5_paths)) * _ATOMS_PER_WORKER)
    n_ph2 = _autosize_phase2(args, n_ph2, len(cache), len(pod5_paths))
    # ★Atoms are indexed by POD5 FILE; the cache is indexed by GROUP. Keep them separate and
    # carry an explicit map. Ungrouped runs have one group, so every file maps to 0.
    if si is not None:
        file2group = [i // bucket for i in range(len(pod5_paths))]   # local numbering, matches idx2local
    else:
        file2group = [0] * len(pod5_paths)
    atoms, counts = _build_atoms(pod5_paths, n_ph2)
    log.info("Phase 2: %d atoms over %d workers", len(atoms), n_ph2)

    for f in os.listdir(args.output):        # clear receipts from any previous/failed run
        if f.startswith(".atoms."):
            os.remove(os.path.join(args.output, f))
    mgr = mp.Manager()
    q = mgr.Queue()
    for a in atoms:
        q.put(a)

    procs = []
    for pid in range(n_ph2):
        p = mp.Process(target=segment_normalize_v3,
                       args=(cache, q, norm_factor, pid, args.output, args.cb_len, args.kmer_len,
                             args.chunk, args.max_token_len, args.sampling, args.dwell_shift,
                             args.sig_window, args.process_once, args.label_div,
                             args.padding_mode, args.trim, pod5_paths,
                             not args.no_writer_thread, file2group))
        procs.append(p)
        p.start()
    gc.collect()
    try:
        wait_for_processes(procs, "Signal segmentation/normalization")
        # ★Prove every dispatched atom was consumed. Phase 1 has the same discipline via n_seen;
        # without it a silently-truncated phase 2 is indistinguishable from success.
        done = sum(int(np.load(os.path.join(args.output, f))[0])
                   for f in os.listdir(args.output) if f.startswith(".atoms."))
        for f in os.listdir(args.output):
            if f.startswith(".atoms."):
                os.remove(os.path.join(args.output, f))
        if done != len(atoms):
            raise RuntimeError(
                f"Phase 2 consumed {done} of {len(atoms)} atoms -- output is incomplete. "
                f"(A worker exited early; check for an OOM kill or a queue failure.)")
        log.info("Phase 2 consumed all %d atoms", done)
    finally:
        cache.close()
        mgr.shutdown()

    if not os.environ.get(_KEEP_CACHE_ENV):
        shutil.rmtree(cache_dir, ignore_errors=True)
    log.info("Finished DeepRM Preprocessing")
    return 0


def add_arguments(parser: argparse.ArgumentParser):
    """
    Adds command-line arguments.

    The options before --summary-index are shared with the C++ binary, which `deeprm call prep`
    runs when it can start; they keep the binary's names (--process-once is used only by this
    module, and the binary ignores it). The options from --summary-index on exist only in this
    module: setting one of them makes `deeprm call prep` use this module directly.

    Args:
        parser (argparse.ArgumentParser): Argument parser to which arguments will be added.
    Returns:
        None
    """
    num_cpu = os.cpu_count()
    parser.add_argument("--pod5", "-p", type=str, required=True,
                        help="POD5 input directory, or a single POD5 file")
    parser.add_argument("--bam", "-b", type=str, default=None,
                        help="Dorado BAM file, or - to read the BAM from standard input")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--thread", "-t", type=int, default=max(1, int(num_cpu * 0.95)),
                        help="Number of threads to use (default: 95%% of the CPUs)")
    parser.add_argument(
        "--qcut",
        "-q",
        type=int,
        default=0,
        help="Skip reads whose mean base quality is below this value (default 0: keep all reads)",
    )
    parser.add_argument("--chunk", "-k", type=int, default=16000, help="Rows per output file")
    parser.add_argument("--max-token-len", "-z", type=int, default=200,
                        help="Most signal windows in a context block; blocks with more are skipped")
    parser.add_argument("--sampling", "-s", type=int, default=6,
                        help="Signal samples per move-table step (the stride in the BAM mv tag; 6 for "
                             "RNA004 basecalls). Reads with another stride are skipped.")
    parser.add_argument("--boi", "-y", type=str, default="A", help="Base of interest (A for m6A)")
    parser.add_argument("--kmer-len", "-e", type=int, default=5, help="k-mer length; must be odd")
    parser.add_argument("--cb-len", "-l", type=int, default=21,
                        help="Context-block length in bases, centred on the site; must be odd")
    parser.add_argument("--bam-thread", "-a", type=int, default=4, help="BAM decompression threads per process")
    parser.add_argument("--process-once", "-n", type=int, default=1000,
                        help="Reads per processing batch (Python preprocessing only; the binary ignores it)")
    parser.add_argument("--dwell-shift", "-f", type=int, default=10,
                        help="Distance in bases between the motor and the pore, for the dwell times")
    parser.add_argument("--sig-window", "-w", type=int, default=5, help="Signal window size")
    parser.add_argument("--label-div", "-d", type=int, default=10**9,
                        help="Factor that packs the reference index and the position into one label ID; "
                             "deeprm call run and deeprm call pileup must use the same value (--label_div)")
    parser.add_argument(
        "--filter-flag",
        "-g",
        type=int,
        default=276,
        help="SAM flag bits of the reads to skip (a read is skipped if its flag has any of these bits). "
        "Use 276 (unmapped, reverse, secondary) for a transcriptome alignment, and 260 (unmapped, secondary) "
        "for a genome alignment, so that reads of minus-strand genes are kept and processed in the RNA's direction.",
    )

    g = parser.add_mutually_exclusive_group()
    g.add_argument("--zero-pad", action="store_true",
                   help="Pad context blocks that reach past a read end with zeros, instead of "
                        "skipping those sites")
    g.add_argument("--edge-pad", action="store_true",
                   help="Pad context blocks that reach past a read end with the edge value, "
                        "instead of skipping those sites")
    parser.add_argument("--trim", action="store_true",
                        help="Trim low-quality read ends before cutting context blocks")
    parser.add_argument("--trim-threshold", type=int, default=5,
                        help="Base quality that counts as good when trimming (--trim)")
    parser.add_argument("--trim-penalty", type=int, default=-15,
                        help="Score of a base below --trim-threshold when trimming (--trim)")
    parser.add_argument("--bam-cmd", type=str, default=None,
                        help="Run this command and read the BAM it writes to its standard output, "
                             "for example a basecaller command. Unlike '--bam -', the run fails if "
                             "the command fails, so a cut-off stream cannot pass as a short run.")
    # Options of the Python preprocessing only (the C++ binary has none of them): setting one makes
    # `deeprm call prep` use this module.
    parser.add_argument("--summary-index", type=str, default=None,
                        help="Folder of a read-to-POD5-file index made by an earlier run "
                             "(<output>/.deeprm_pod5index). By default the index is made from the "
                             "POD5 files.")
    parser.add_argument("--index-cache", type=str, default=None,
                        help="Folder for the read-to-POD5-file index (default "
                             "<output>/.deeprm_pod5index); an index there that matches the POD5 "
                             "files is reused")
    parser.add_argument("--no-group", action="store_true",
                        help="Keep the BAM cache in one part instead of one part per POD5 file "
                             "(slower to read; the same rows)")
    parser.add_argument("--summary", nargs="?", const="auto", default=None,
                        help="Make the read-to-POD5-file index from a sequencing summary instead of "
                             "from the POD5 files. --summary alone looks for "
                             "sequencing_summary*.txt beside the POD5 folder; --summary FILE names "
                             "one. A summary that does not match the POD5 files is not used.")
    parser.add_argument("--sidecar", type=str, default=None,
                        help="File of extra BAM split points (.rbsc; default <bam>.rbsc) that lets "
                             "the workers share a sorted, indexed BAM evenly when one region holds "
                             "very many reads. Made automatically when useful; a missing or invalid "
                             "file is ignored.")
    parser.add_argument("--no-sidecar", action="store_true",
                        help="Do not use a .rbsc file, and do not make one automatically")
    parser.add_argument("--sidecar-level", type=int, default=3, choices=(0, 1, 2, 3),
                        help="How thoroughly a .rbsc file is checked before use: 3 (default) checks "
                             "every split point against the BAM, 2 a sample of them; 0 and 1 check "
                             "only the file itself, the size and time of the BAM and its index, and "
                             "a 16 MiB sample of the BAM")
    parser.add_argument("--build-sidecar", action="store_true",
                        help="Make the .rbsc file for --bam (next to the BAM, or at --sidecar PATH) "
                             "and exit. Not needed normally: one is made automatically when a worker "
                             "would otherwise get more than twice its share of the BAM.")
    parser.add_argument("--sidecar-cap-mb", type=int, default=DEFAULT_CAP >> 20,
                        help="For --build-sidecar: only gaps between index points larger than this "
                             "many MiB of BAM get extra split points")
    parser.add_argument("--sidecar-stride-mb", type=int, default=None,
                        help="For --build-sidecar: one split point per this many MiB of BAM")
    parser.add_argument("--sidecar-procs", type=int, default=8,
                        help="For --build-sidecar: number of worker processes")
    # A self-test of the --auto-size memory model; hidden from --help.
    parser.add_argument("--validate-sizing", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--auto-size", action="store_true",
                        help="Lower the number of signal workers to what the free memory supports, "
                             "by a fixed memory model. The run stops when not even one worker fits "
                             "(the model needs about 30 GiB of free memory). Off by default.")
    parser.add_argument("--no-writer-thread", action="store_true",
                        help="Compress output files in the worker instead of on a separate thread")

    return None


def _refuse_a_terminal(bam):
    """Fail at once when the BAM would be read from a terminal (-b - or -b /dev/stdin with nothing piped in): the
    run would wait for keyboard input with no message. The binary refuses both in the same way."""
    if bam == "-":
        if os.isatty(0):
            raise ValueError("-b - reads the BAM from stdin, but stdin is a terminal; pipe a BAM in "
                             "(e.g. dorado ... | ...) or use --bam-cmd")
        return
    # Only a character device can be a terminal; opening a pipe to look would wait for its writer.
    if not stat.S_ISCHR(os.stat(bam).st_mode):
        return
    try:
        fd = os.open(bam, os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)  # opening a terminal reads nothing
    except OSError:
        return  # the stream reader opens it again and says why it cannot
    try:
        terminal = os.isatty(fd)
    finally:
        os.close(fd)
    if terminal:
        raise ValueError(f"-b {bam} is a terminal, not a BAM; pipe a BAM in (e.g. dorado ... | ...) or use --bam-cmd")


def _finish_args(args):
    """
    Check and complete parsed arguments before a run.

    Runs the two modes that exit without preprocessing (--build-sidecar, --validate-sizing) and
    returns their exit code; returns None for a normal run, after the checks below.
    """
    if args.thread < 1:
        raise ValueError("--thread must be >= 1")
    if args.bam_thread < 1:
        raise ValueError("--bam-thread must be >= 1")
    if args.sampling < 1:
        raise ValueError("--sampling must be >= 1")
    if args.process_once < 1:
        raise ValueError("--process-once must be >= 1")
    # The pipeline code reads the CPU and BAM thread counts under these names.
    args.cpu = args.thread
    args.bam_threads = args.bam_thread
    if getattr(args, "build_sidecar", False):
        if args.bam in (None, "", "-"):
            raise ValueError("--build-sidecar needs --bam with a BAM file")
        out = build_sidecar(args.bam, cap=args.sidecar_cap_mb << 20,
                            stride=(args.sidecar_stride_mb << 20) if args.sidecar_stride_mb
                            else None, procs=args.sidecar_procs, out=args.sidecar)
        print(f"sidecar: {out['sidecar']}  {out['n_new_points']:,} points from "
              f"{out['records_walked']:,} records in {out['wall_s']:.1f} s "
              f"({out['sidecar_bytes']:,} B)")
        return 0
    if getattr(args, "validate_sizing", False):
        print(f"{'run':<28}{'W':>4}{'raw':>9}{'safe_pred':>11}{'measured':>10}{'safe?':>8}")
        ok = True
        for r in validate_against_measured():
            safe = r["predicted_gib"] >= r["measured_gib"]
            ok &= safe
            print(f"{r['label']:<28}{r['workers']:>4}{r['raw_gib']:>8.1f}G"
                  f"{r['predicted_gib']:>8.1f}G{r['measured_gib']:>9.1f}G"
                  f"{'  OK' if safe else '  UNDER':>8}")
        print("SIZING_SAFE" if ok else "SIZING_UNSAFE -- raise SAFETY")
        return 0 if ok else 1
    args.padding_mode = "edge" if args.edge_pad else ("zero" if args.zero_pad else "none")
    validate_geometry(args.cb_len, args.kmer_len, args.sig_window, args.chunk)
    if args.filter_flag < 0 or args.filter_flag > 0xFFF:
        raise ValueError(f"--filter-flag must be a SAM flag mask in [0, 4095] (got "
                         f"{args.filter_flag}); it is ANDed with read.flag, so a negative or "
                         f"over-wide mask silently drops everything.")
    if not os.path.exists(args.pod5):
        raise FileNotFoundError(f"Input directory {args.pod5} does not exist")
    if args.bam_cmd and args.bam not in (None, "", "-"):
        raise ValueError("pass either --bam or --bam-cmd, not both")
    if args.bam_cmd:
        args.bam = "-"
    if args.bam is None:
        raise ValueError("--bam is required (a BAM file, or - for standard input), unless --bam-cmd is given")
    if args.bam != "-" and not os.path.exists(args.bam):
        raise FileNotFoundError(f"BAM file {args.bam} does not exist")
    if not args.bam_cmd:
        _refuse_a_terminal(args.bam)
    if args.bam != "-" and stat.S_ISREG(os.stat(args.bam).st_mode):
        _refuse_cut_off_sam_file(args.bam)  # a stream is checked at its end (_finish_bam_stream)
    os.makedirs(args.output, exist_ok=True)
    return None
