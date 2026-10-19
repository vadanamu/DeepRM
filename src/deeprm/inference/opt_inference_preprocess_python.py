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

Memory design
-------------
The per-read BAM payload (move table, sequence, qualities, aligned pairs) is far too
large to keep as Python objects: for a 24 M read direct-RNA run it is ~325 GiB as a
pandas frame of object columns, and holding it in a ``multiprocessing.Manager`` list
*and* in the parent doubles that again.

So the BAM is parsed once into a **ragged flat-array cache on disk** -- a few contiguous
binary blobs plus per-read offsets -- which the POD5 workers ``mmap`` read-only:

* no Python object exists per read, so fork copy-on-write dirties nothing;
* read ids are ``uint64`` pairs found with ``np.searchsorted`` instead of a 24 M entry
  ``PyObjectHashTable`` rebuilt inside every worker;
* the move table is stored bit-packed (8x smaller -- it is ~60% of the payload);
* what stays resident is page cache (shared, reclaimable), not anonymous heap.

Signal handling changes shape rather than precision: reads are normalised one at a time
and windows are written straight into preallocated chunk buffers, instead of holding a
whole POD5 batch of segment views plus one 1224-sample block object per (read, site).
The arithmetic is unchanged, so the emitted tokens are bit-for-bit identical to the
previous implementation.

``DEEPRM_PREP_CACHE_DIR`` relocates the cache (e.g. to local NVMe scratch); it defaults
to ``<output>/.deeprm_bamcache`` and is deleted on success unless
``DEEPRM_PREP_KEEP_CACHE`` is set.
"""

import argparse
import gc
import glob
import multiprocessing as mp
import os
import shutil
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

# Environment overrides (deliberately NOT argparse options: inference_preprocess.py
# forwards every CLI argument to the accelerated C++ binary, so a new flag there would
# make that binary fail and silently divert every run to this Python fallback).
_CACHE_DIR_ENV = "DEEPRM_PREP_CACHE_DIR"
_KEEP_CACHE_ENV = "DEEPRM_PREP_KEEP_CACHE"

# Per-read record of the on-disk BAM cache. Payload lives in the shard blobs; this is
# only offsets/lengths and small scalars, so it stays a contiguous, object-free array.
_META_FIELDS = (
    ("pkey0", np.uint64),  # parent read uuid, bytes[0:8]  (searchsorted key)
    ("pkey1", np.uint64),  # parent read uuid, bytes[8:16] (collision check)
    ("rid0", np.int64),  # own read uuid, as written to the npz `read_id` field
    ("rid1", np.int64),
    ("qoff", np.int64),  # offset into the seq/bq blobs (shared: both are qlen long)
    ("qlen", np.int32),
    ("mvoff", np.int64),  # offset into the packed move blob
    ("mvlen", np.int32),  # *unpacked* move length, in bits
    ("apoff", np.int64),  # offset into the aligned-pair blob, in pairs
    ("apn", np.int32),
    ("ts", np.int32),
    ("ns", np.int32),
    ("sp", np.int32),
    ("ref", np.int32),
    ("strand", np.int8),
)

_BLOBS = ("seq", "bq", "mvp", "ap")
_BLOB_DTYPE = {"seq": np.uint8, "bq": np.int8, "mvp": np.uint8, "ap": np.int32}

# POD5 batches to process per reader instance; see segment_normalize_signal.
_POD5_REOPEN_BATCHES = 32


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
    2. Spawn processes to parse the BAM into the on-disk flat cache.
    3. Build the sorted read-id index over that cache.
    4. Spawn processes to segment and normalize POD5 signals.
    5. Drop the cache and exit.

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
    ## The original divided by these and died with ZeroDivisionError / built a negative
    ## width array; check them up front instead of hanging or crashing mid-run.
    if args.chunk < 1:
        raise ValueError("--chunk must be >= 1")
    if args.cb_len < 1 or args.kmer_len < 1:
        raise ValueError("--cb-len and --kmer-len must be >= 1")

    args.bam = maybe_index_bam(args.bam, args.thread)

    os.makedirs(args.output, exist_ok=True)

    log.info("Started DeepRM Preprocessing")
    norm_factor = get_norm_factor()

    ## Resolve POD5 inputs up front: the number of files caps the phase-2 fan-out, and
    ## failing here is cheaper than failing after the whole BAM has been parsed.
    if os.path.isfile(args.pod5):
        pod5_paths_split = [[args.pod5]]
    else:
        pod5_file_list = sorted(glob.glob(os.path.join(args.pod5, "*.pod5")))
        if len(pod5_file_list) == 0:
            raise FileNotFoundError(f"No POD5 files found in directory: {args.pod5}")
        pod5_paths_split = np.array_split(pod5_file_list, min(args.thread, len(pod5_file_list)))

    cache_base = os.environ.get(_CACHE_DIR_ENV) or os.path.join(args.output, ".deeprm_bamcache")
    os.makedirs(cache_base, exist_ok=True)
    cache_dir = _make_cache_dir(cache_base)
    _check_cache_space(cache_dir, args.bam)

    mp.set_start_method("fork", force=True)

    ## ---- Phase 1: BAM -> flat on-disk cache (one shard per worker) ----
    n_bam_procs = max(1, args.thread // args.bam_thread)
    proc_list = []
    for pid in range(n_bam_procs):
        proc = mp.Process(
            target=parse_bam,
            args=(
                pid,
                n_bam_procs,
                args.bam_thread,
                cache_dir,
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

    ## ---- Build the sorted read-id index once, in the parent ----
    bam_cache = _BamCache(cache_dir, n_shards=n_bam_procs)
    if len(bam_cache) == 0:
        raise RuntimeError("BAM parsing produced no usable reads")
    log.info("Indexed %d BAM records (%.1f GiB cache)", len(bam_cache), bam_cache.nbytes_on_disk / 2**30)
    gc.collect()

    ## ---- Phase 2: POD5 -> tokens. Children inherit the cache read-only. ----
    proc_list = []
    for pid, pod5_paths in enumerate(pod5_paths_split):
        proc = mp.Process(
            target=segment_normalize_signal,
            args=(
                bam_cache,
                list(pod5_paths),
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

    try:
        wait_for_processes(proc_list, "Signal segmentation/normalization")
    finally:
        bam_cache.close()

    if not os.environ.get(_KEEP_CACHE_ENV):
        shutil.rmtree(cache_dir, ignore_errors=True)

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

    Kept for API compatibility; the pipeline itself uses the vectorised equivalent in
    :func:`_emit_read`, which slices the contiguous normalised signal directly instead
    of concatenating per-base segment views.

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

    Kept for API compatibility. The pipeline uses :func:`_normalise_signal_bounds`, which
    returns the same normalised signal plus the segment *boundaries* rather than
    materialising one ndarray view per base.

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
    """
    out = _normalise_signal_bounds(signal, move, sp, ts, ns, quantile_a, quantile_b, shift_mult, scale_mult, sampling)
    if out is None:
        return None
    norm, bounds = out
    return [norm[bounds[i] : bounds[i + 1]] for i in range(len(bounds) - 1)]


def _normalise_signal_bounds(signal, move, sp, ts, ns, quantile_a, quantile_b, shift_mult, scale_mult, sampling=6):
    """
    Normalise the trimmed signal and return it with its per-base segment boundaries.

    The segments produced by ``np.array_split`` are consecutive slices of one array, so
    the concatenation of segments ``[i:j]`` is exactly ``norm[bounds[i]:bounds[j]]``.
    Carrying the boundaries instead of the split views removes one ndarray object per
    base (~1 M objects per POD5 batch) and turns block assembly into a memcpy.

    Args:
        signal (numpy.ndarray): raw signal trace.
        move (numpy.ndarray): boolean move array.
        sp (int): samples to skip at start.
        ts (int): trim start index.
        ns (int): trim end index (0 means till end).
        quantile_a, quantile_b (float): quantile thresholds.
        shift_mult, scale_mult (float): shift/scale multipliers.
        sampling (int): samples per signal unit.

    Returns:
        tuple(numpy.ndarray, numpy.ndarray) or None: (float32 normalised signal,
        int64 segment boundaries of length n_segments+1), or None on error.
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

        ## Kept in float64, exactly as before. Only one read is normalised at a time here
        ## (the original held a whole 1000-read batch), so the wider dtype costs ~0.5 MB
        ## rather than gigabytes -- and it keeps the emitted signal_token bit-for-bit
        ## identical to the previous implementation, so no downstream revalidation.
        quantile_a_value = np.quantile(signal, quantile_a)
        quantile_b_value = np.quantile(signal, quantile_b)

        q_shift = max(10.0, shift_mult * (quantile_a_value + quantile_b_value))
        q_scale = max(1.0, scale_mult * (quantile_b_value - quantile_a_value))
        norm = (signal - q_shift) / q_scale
        if not np.all(np.isfinite(norm)):
            return None

        move_idx = np.where(move)[0][1:] * sampling
        move_idx = len(norm) - move_idx
        move_idx = np.flip(move_idx, axis=0)

        bounds = np.empty(len(move_idx) + 2, dtype=np.int64)
        bounds[0] = 0
        bounds[1:-1] = move_idx
        bounds[-1] = len(norm)
        ## A malformed read (move table longer than the signal it claims to describe)
        ## would give negative or non-monotonic bounds; np.array_split silently produced
        ## nonsense there, so drop the read instead.
        if bounds.size < 2 or bounds[1] < 0 or np.any(np.diff(bounds) < 0):
            return None
    except Exception:
        return None
    return norm, bounds


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
        for record in reader.reads(read_ids):
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


def _read_pod5_batch(reader, read_ids):
    """
    Read one batch of signals from an already-open POD5 reader.

    Unlike :func:`parse_pod5` this reuses the reader (the original reopened the file for
    every 1000-read batch) and skips the calibration fields, which the pipeline never
    consumes.

    Args:
        reader (pod5.Reader): open reader for the file the ids belong to.
        read_ids (list): read ids to fetch.

    Returns:
        tuple(list[numpy.ndarray], numpy.ndarray, numpy.ndarray): signals, and the read
        uuids split into two uint64 key columns.
    """
    signals = []
    raw = []
    for record in reader.reads(read_ids):
        ## Both fields must be read BEFORE either list is appended: `signals` and `raw`
        ## are index-aligned downstream, so appending the signal and then failing on the
        ## read id would shift every later read in the batch onto the wrong id and the
        ## wrong trace -- silently, with no error.
        try:
            signal = record.signal
            read_id = record.read_id.bytes
        except Exception:
            continue
        signals.append(signal)
        raw.append(read_id)
    if not signals:
        return [], np.empty(0, np.uint64), np.empty(0, np.uint64)
    keys = np.frombuffer(b"".join(raw), dtype=np.uint64).reshape(-1, 2)
    return signals, np.ascontiguousarray(keys[:, 0]), np.ascontiguousarray(keys[:, 1])


class _ShardWriter:
    """
    Streaming writer for one BAM-cache shard.

    Payload is appended to four flat blobs; per-read offsets accumulate in a contiguous
    structured array. Nothing is retained per read, so a worker's resident set stays at
    the flush buffer size regardless of how many reads it parses.
    """

    def __init__(self, cache_dir, shard, flush_reads=4096):
        self.shard = shard
        self._flush_reads = flush_reads
        self._fh = {b: open(os.path.join(cache_dir, f"shard{shard}.{b}"), "wb", buffering=1 << 20) for b in _BLOBS}
        self._buf = {b: [] for b in _BLOBS}
        self._pos = {b: 0 for b in _BLOBS}
        self._meta = np.empty(1 << 16, dtype=np.dtype(list(_META_FIELDS)))
        self._n = 0
        self._since_flush = 0
        self._meta_path = os.path.join(cache_dir, f"shard{shard}.meta.npy")

    def add(self, pkey, rid, seq, bq, mvp, mvlen, ap, ts, ns, sp, ref, strand):
        """Append one read. Arrays must already be in their stored dtypes."""
        if self._n == len(self._meta):
            grown = np.empty(len(self._meta) * 2, dtype=self._meta.dtype)
            grown[: self._n] = self._meta
            self._meta = grown
        self._meta[self._n] = (
            pkey[0],
            pkey[1],
            rid[0],
            rid[1],
            self._pos["seq"],
            len(seq),
            self._pos["mvp"],
            mvlen,
            self._pos["ap"] // 2,
            len(ap),
            ts,
            ns,
            sp,
            ref,
            strand,
        )
        self._n += 1
        for name, arr in (("seq", seq), ("bq", bq), ("mvp", mvp), ("ap", ap.reshape(-1))):
            self._buf[name].append(arr.tobytes())
            self._pos[name] += arr.size
        self._since_flush += 1
        if self._since_flush >= self._flush_reads:
            self._flush()

    def _flush(self):
        for name, fh in self._fh.items():
            if self._buf[name]:
                fh.write(b"".join(self._buf[name]))
                self._buf[name].clear()
        self._since_flush = 0

    def close(self):
        self._flush()
        for fh in self._fh.values():
            fh.close()
        np.save(self._meta_path, self._meta[: self._n])
        self._meta = None


class _BamCache:
    """
    Read-only, fork-safe view over the on-disk BAM cache.

    Built once in the parent: shard metadata is concatenated and sorted by read-id key
    so lookups are a vectorised ``np.searchsorted``. Every attribute is either a
    contiguous numpy array or an ``mmap`` -- there are no Python objects per read, so
    forked workers share all of it and dirty none of it.
    """

    def __init__(self, cache_dir, n_shards):
        self.cache_dir = cache_dir
        metas = []
        shard_ids = []
        for shard in range(n_shards):
            path = os.path.join(cache_dir, f"shard{shard}.meta.npy")
            if not os.path.exists(path):
                continue
            meta = np.load(path)
            if len(meta) == 0:
                continue
            metas.append(meta)
            shard_ids.append(np.full(len(meta), shard, dtype=np.int32))

        if not metas:
            self._empty()
            return

        meta = np.concatenate(metas)
        shard_id = np.concatenate(shard_ids)
        del metas, shard_ids

        ## Sort by the primary key so batches resolve with two searchsorted calls.
        order = np.argsort(meta["pkey0"], kind="stable")
        for name, _ in _META_FIELDS:
            setattr(self, name, np.ascontiguousarray(meta[name][order]))
        self.shard = np.ascontiguousarray(shard_id[order])
        del meta, shard_id, order

        self._blob_paths = {
            b: [os.path.join(cache_dir, f"shard{s}.{b}") for s in range(n_shards)] for b in _BLOBS
        }
        self.blob = {
            b: [
                np.memmap(p, dtype=_BLOB_DTYPE[b], mode="r") if os.path.getsize(p) else np.empty(0, _BLOB_DTYPE[b])
                for p in paths
            ]
            for b, paths in self._blob_paths.items()
        }
        self.nbytes_on_disk = sum(os.path.getsize(p) for paths in self._blob_paths.values() for p in paths)

    def _empty(self):
        for name, dtype in _META_FIELDS:
            setattr(self, name, np.empty(0, dtype))
        self.shard = np.empty(0, np.int32)
        self.blob = {b: [] for b in _BLOBS}
        self.nbytes_on_disk = 0

    def __len__(self):
        return len(self.pkey0)

    def lookup(self, key0, key1):
        """
        Resolve POD5 read-id keys to cache rows.

        Args:
            key0, key1 (numpy.ndarray): uint64 halves of each POD5 read uuid.

        Returns:
            tuple(numpy.ndarray, numpy.ndarray): parallel arrays of POD5 index and cache
            row. A read split by the basecaller yields one entry per child alignment,
            reproducing the one-to-many index join of the original merge.
        """
        if len(self) == 0 or len(key0) == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        lo = np.searchsorted(self.pkey0, key0, side="left")
        hi = np.searchsorted(self.pkey0, key0, side="right")
        cnt = (hi - lo).astype(np.int64)
        total = int(cnt.sum())
        if total == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        src = np.repeat(np.arange(len(key0), dtype=np.int64), cnt)
        starts = np.repeat(lo.astype(np.int64), cnt)
        within = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        rows = starts + within
        ## The 64-bit primary key can collide; verify the other half before accepting.
        keep = self.pkey1[rows] == key1[src]
        return src[keep], rows[keep]

    def arrays(self, row):
        """Return (seq, bq, packed_move, move_len, aligned_pairs) views for one cache row."""
        s = int(self.shard[row])
        qo, ql = int(self.qoff[row]), int(self.qlen[row])
        mo, ml = int(self.mvoff[row]), int(self.mvlen[row])
        ao, an = int(self.apoff[row]), int(self.apn[row])
        nbytes = (ml + 7) // 8
        return (
            self.blob["seq"][s][qo : qo + ql],
            self.blob["bq"][s][qo : qo + ql],
            self.blob["mvp"][s][mo : mo + nbytes],
            ml,
            self.blob["ap"][s][2 * ao : 2 * (ao + an)].reshape(-1, 2),
        )

    def close(self):
        self.blob = {b: [] for b in _BLOBS}

    def __getstate__(self):
        raise TypeError("_BamCache is fork-shared and must not be pickled")


def _make_cache_dir(base):
    """
    Create a private per-run cache directory under ``base``.

    Deliberately NOT ``rmtree(base)``: ``DEEPRM_PREP_CACHE_DIR`` invites pointing this at
    an existing scratch directory, and wiping it would destroy unrelated files and make
    two concurrent runs delete each other's shards mid-flight. Only the private
    subdirectory this run creates is ever removed.

    Args:
        base (str): directory to create the per-run cache under.

    Returns:
        str: path to the new, empty, run-private cache directory.
    """
    run_dir = os.path.join(base, f"run-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


# Measured on ON0197 (direct-RNA, 24 M reads): seq+bq+packed move+aligned pairs.
_CACHE_BYTES_PER_READ = 6500


def _check_cache_space(cache_dir, bam_path):
    """
    Refuse to start if the BAM cache clearly will not fit.

    Running out of space partway through parsing wastes the whole phase, so spend a
    couple of milliseconds on the index counters up front.
    """
    try:
        with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
            n_reads = bam.mapped + bam.unmapped
    except Exception:
        return  # no index counters available; nothing useful to check
    if n_reads <= 0:
        return
    need = n_reads * _CACHE_BYTES_PER_READ
    free = shutil.disk_usage(cache_dir).free
    log.info(
        "BAM cache estimate %.1f GiB for %d reads; %.1f GiB free at %s",
        need / 2**30,
        n_reads,
        free / 2**30,
        cache_dir,
    )
    if free < need:
        ## Only a warning: the estimate counts records that the flag/quality filters will
        ## drop, so it over-estimates, and refusing a run the previous version would have
        ## completed is a worse failure than running out of space and saying so.
        log.warning(
            "BAM cache may not fit at %s: estimated ~%.1f GiB, %.1f GiB free. "
            "Point %s at a larger filesystem if parsing fails.",
            cache_dir, need / 2**30, free / 2**30, _CACHE_DIR_ENV,
        )


def _uuid_keys(u):
    """Split a uuid into its two uint64 halves and the two int64s stored as `read_id`."""
    raw = np.frombuffer(u.bytes, dtype=np.uint64)
    return (int(raw[0]), int(raw[1])), raw.view(np.int64)


def parse_bam(pid, n_procs, n_thread, bam_data, bam_path, bq_cutoff, boi, expected_sampling, filter_flag=276):
    """
    Extract move tags and alignment information from a BAM file in parallel.

    Args:
        pid (int): process ID for sharding.
        n_procs (int): total number of processes.
        n_thread (int): thread for BAM reading.
        bam_data (str | list): cache directory to stream this shard into, or a
            ``multiprocessing`` list to append a DataFrame to (legacy callers).
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
        None (writes a cache shard, or appends a DataFrame to ``bam_data``).
    """
    legacy = not isinstance(bam_data, (str, os.PathLike))
    writer = None if legacy else _ShardWriter(bam_data, pid)
    rows = {k: [] for k in ["read_id", "parent_id", "ts", "ns", "sp", "bq", "mv", "seq", "ref", "ap", "strand"]}

    input_bam = pysam.AlignmentFile(bam_path, "rb", check_sq=False, threads=n_thread)
    ref_index_dict = {ref: i for i, ref in enumerate(input_bam.references)}
    mv_stride_mismatch = 0
    boi_fwd = boi.upper()
    boi_rev = _BASE_COMPLEMENT.get(boi_fwd, boi_fwd)

    for read_idx, read in tqdm.tqdm(enumerate(input_bam), total=input_bam.mapped + input_bam.unmapped, disable=pid != 0):
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

        boi_strand = boi_rev if read.is_reverse else boi_fwd
        ap = np.array(read.get_aligned_pairs(matches_only=True, with_seq=True), dtype=object)
        ap = ap[ap[:, 2] == boi_strand][:, :2].astype(np.int32)
        if len(ap) == 0:
            continue

        ref = ref_index_dict.get(read.reference_name, -1)
        if ref < 0:
            continue

        mv = np.array(mv_tag[1:], dtype=bool)
        seq = np.array(list(read.query_sequence)).view(np.int32).astype(np.uint8)
        strand = -1 if read.is_reverse else 1
        read_id = uuid.UUID(str(read.query_name))
        parent_id = uuid.UUID(str(read.get_tag("pi"))) if read.has_tag("pi") else read_id
        ts = read.get_tag("ts") if read.has_tag("ts") else 0
        ns = read.get_tag("ns") if read.has_tag("ns") else 0
        sp = read.get_tag("sp") if read.has_tag("sp") else 0

        if legacy:
            rows["read_id"].append(read_id)
            rows["parent_id"].append(parent_id)
            rows["ts"].append(ts)
            rows["ns"].append(ns)
            rows["sp"].append(sp)
            rows["bq"].append(bq)
            rows["mv"].append(mv)
            rows["seq"].append(seq)
            rows["ref"].append(read.reference_name)
            rows["ap"].append(ap)
            rows["strand"].append(strand)
        else:
            pkey, _ = _uuid_keys(parent_id)
            _, rid = _uuid_keys(read_id)
            writer.add(pkey, rid, seq, bq, np.packbits(mv), mv.size, ap, ts, ns, sp, ref, strand)

    input_bam.close()
    if mv_stride_mismatch > 0:
        log.warning(
            "Skipped %d reads in BAM worker %d due to mv-tag stride mismatch with --sampling=%d",
            mv_stride_mismatch,
            pid,
            expected_sampling,
        )

    if legacy:
        bam_df = pd.DataFrame.from_dict(rows, orient="columns")
        bam_df["ref"] = bam_df["ref"].map(ref_index_dict)
        bam_df = bam_df.dropna()
        bam_df[["ts", "ns", "sp"]] = bam_df[["ts", "ns", "sp"]].astype(np.int32)
        bam_data.append(bam_df)
    else:
        writer.close()
    gc.collect()
    return None


class _ChunkAccumulator:
    """
    Preallocated output buffers, flushed as soon as ``chunk_size`` rows are ready.

    The original built one Python object per (read, site) -- a 1224-sample float64 block
    among them -- inside an object-dtype column, then stacked and chunked. Writing rows
    straight into a fixed buffer removes ~10 kB of transient heap per site.
    """

    def __init__(self, chunk_size, sig_width, tok_width, kmer_width, out_prefix, chunk_sink=None):
        self.chunk_size = chunk_size
        self.n = 0
        self.out_prefix = out_prefix
        self.chunk_idx = 0
        self.chunk_sink = chunk_sink
        self.buf = {
            "signal_token": np.zeros((chunk_size, sig_width), np.float32),
            "kmer_token": np.zeros((chunk_size, kmer_width), np.uint8),
            "segment_len_arr": np.zeros((chunk_size, tok_width), np.uint16),
            "dwell_motor_token": np.zeros((chunk_size, tok_width), np.float32),
            "dwell_pore_token": np.zeros((chunk_size, tok_width), np.float32),
            "bq_token": np.zeros((chunk_size, tok_width), np.uint8),
            "label_id": np.zeros(chunk_size, np.int64),
            "read_id": np.zeros((chunk_size, 2), np.int64),
        }

    def room(self):
        return self.chunk_size - self.n

    def flush(self, final=False):
        if self.n == 0:
            return
        payload = {k: v[: self.n].copy() for k, v in self.buf.items()}
        name = f"{self.out_prefix}-last.npz" if final else f"{self.out_prefix}-{self.chunk_idx}.npz"
        if self.chunk_sink is not None:
            self.chunk_sink(payload, name)
        else:
            np.savez_compressed(name, **payload)
        self.chunk_idx += 1
        self.n = 0
        ## The signal buffer is the only one that is not fully overwritten (short blocks
        ## are zero-padded), so it must be cleared before reuse.
        self.buf["signal_token"][:] = 0


def _emit_read(cache_row, cache, signal, norm_factor, cfg, acc):
    """
    Turn one (POD5 read, BAM alignment) pair into context-block rows.

    Args:
        cache_row (int): row in the BAM cache.
        cache (_BamCache): the cache.
        signal (numpy.ndarray): raw int16 signal for the POD5 read.
        norm_factor (dict): normalisation parameters.
        cfg (dict): geometry (cb_len, kmer_len, sampling, ...).
        acc (_ChunkAccumulator): output buffer.

    Returns:
        int: number of rows emitted.
    """
    sampling = cfg["sampling"]
    cb_len, trim, cb_half = cfg["cb_len"], cfg["trim"], cfg["cb_half"]
    dwell_shift, kmer_pad, lr_pad = cfg["dwell_shift"], cfg["kmer_pad"], cfg["lr_pad"]
    sig_width, max_token_len = cfg["sig_width"], cfg["max_token_len"]

    seq, bq, mvp, mvlen, ap = cache.arrays(cache_row)
    move = np.unpackbits(mvp, count=mvlen).astype(bool)

    dwell = move_to_dwell(move, 0.2, 0.8, 0.5, 1.5, sampling)
    if dwell is None:
        return 0
    out = _normalise_signal_bounds(
        signal,
        move,
        int(cache.sp[cache_row]),
        int(cache.ts[cache_row]),
        int(cache.ns[cache_row]),
        norm_factor["quantile_a"],
        norm_factor["quantile_b"],
        norm_factor["shift_mult"],
        norm_factor["scale_mult"],
        sampling,
    )
    if out is None:
        return 0
    norm, bounds = out
    nseg = len(bounds) - 1

    ## Present reverse-strand reads in RNA-sense orientation: signal/dwell are already
    ## basecalled order, but the BAM stores SEQ/QUAL reverse-complemented into forward-
    ## reference order and reports forward-reference q_pos.
    strand = int(cache.strand[cache_row])
    q_len = len(seq)
    if strand == -1:
        seq = _ASCII_COMPLEMENT[seq][::-1]
        bq = bq[::-1]
        q_pos = q_len - 1 - ap[:, 0].astype(np.int64)
    else:
        q_pos = ap[:, 0].astype(np.int64)
    r_pos = ap[:, 1].astype(np.int64)

    start = q_pos - cb_half
    end = q_pos + cb_half + 1

    keep = (start >= 0) & (end + dwell_shift - trim < q_len)
    ## The dwell/segment axes are indexed independently of q_len, so bound them too.
    ## They only disagree if the move table and SEQ disagree about the read length, in
    ## which case the original produced a short (ragged) window and died in np.stack.
    ## `end <= nseg` / `<= len(dwell)` bound the top of the dwell and segment axes, which
    ## are indexed independently of q_len. The lower bound matters too: a negative
    ## --dwell-shift would make the motor window index negatively, and numpy would
    ## silently wrap it to the end of the array instead of erroring as the original did.
    inbound = (end <= nseg) & (end + dwell_shift - trim <= len(dwell)) & (start + trim + dwell_shift >= 0)
    cfg["dropped_ragged"] += int((keep & ~inbound).sum())
    keep &= inbound
    if not keep.any():
        return 0
    start, end, r_pos = start[keep], end[keep], r_pos[keep]

    seglen = np.diff(bounds) // sampling
    win = start[:, None] + np.arange(cb_len)
    segwin = seglen[win]

    token_len = segwin[:, trim : cb_len - trim].sum(axis=1)
    l_units = segwin[:, :kmer_pad].sum(axis=1) - lr_pad
    ## Bug-for-bug with the original's `segment_len_arr[-kmer_pad:]`: when kmer_pad is 0
    ## (--kmer-len 1 or 2) that slice is `[-0:]`, i.e. the WHOLE array, not an empty one.
    ## Only reachable for non-default k-mer lengths, but a drop-in replacement should not
    ## quietly change which rows survive; fixing it is a separate decision.
    r_span = segwin if kmer_pad == 0 else segwin[:, cb_len - kmer_pad :]
    r_units = r_span.sum(axis=1) - lr_pad

    b0 = bounds[start]
    b1 = bounds[end]
    span = b1 - b0
    l_skip = l_units * sampling
    r_skip = r_units * sampling
    body = span - l_skip - np.maximum(r_skip, 0)

    ok = (
        (token_len > 0)
        & (token_len <= max_token_len)
        & (l_units >= 0)
        & (r_units >= 0)
        & (span % sampling == 0)
        & (l_skip <= span)
        & ((r_skip <= 0) | (l_skip + r_skip <= span))
        & (body > 0)
    )
    ## token_len <= max_token_len bounds `body` whenever every segment length is a whole
    ## number of sampling units. If one is not, the block would overrun the buffer; the
    ## original built an over-long row and died later in np.stack.
    fits = body <= sig_width
    cfg["dropped_oversized"] += int((ok & ~fits).sum())
    ok &= fits
    if not ok.any():
        return 0

    start, end, r_pos = start[ok], end[ok], r_pos[ok]
    b0, l_skip, r_skip, body = b0[ok], l_skip[ok], r_skip[ok], body[ok]
    segwin = segwin[ok]

    label_id = (int(cache.ref[cache_row]) * cfg["label_div"] + r_pos + 1) * strand
    rid = (int(cache.rid0[cache_row]), int(cache.rid1[cache_row]))

    tok_idx = (start + trim)[:, None] + np.arange(cb_len - 2 * trim)
    kmer_idx = start[:, None] + np.arange(cb_len)

    kmer_all = seq[kmer_idx]
    bq_all = np.clip(bq[tok_idx], 0, 60).astype(np.uint8)
    segl_all = segwin[:, trim : cb_len - trim].astype(np.uint16)
    pore_all = dwell[tok_idx]
    motor_all = dwell[tok_idx + dwell_shift]

    written = 0
    m = len(start)
    i = 0
    while i < m:
        if acc.room() == 0:
            acc.flush()
        take = min(acc.room(), m - i)
        sl = slice(i, i + take)
        dst = slice(acc.n, acc.n + take)
        acc.buf["kmer_token"][dst] = kmer_all[sl]
        acc.buf["bq_token"][dst] = bq_all[sl]
        acc.buf["segment_len_arr"][dst] = segl_all[sl]
        acc.buf["dwell_pore_token"][dst] = pore_all[sl]
        acc.buf["dwell_motor_token"][dst] = motor_all[sl]
        acc.buf["label_id"][dst] = label_id[sl]
        acc.buf["read_id"][dst, 0] = rid[0]
        acc.buf["read_id"][dst, 1] = rid[1]
        sig_buf = acc.buf["signal_token"]
        for j in range(take):
            k = i + j
            lo = b0[k] + l_skip[k]
            n_body = body[k]
            sig_buf[acc.n + j, :n_body] = norm[lo : lo + n_body]
            if n_body < sig_width:
                sig_buf[acc.n + j, n_body:] = 0
        acc.n += take
        written += take
        i += take
    return written


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
    chunk_sink=None,
):
    """
    Segment and normalize signals per read, and save token chunks.

    Args:
        bam_df (_BamCache | pandas.DataFrame): alignment metadata. A DataFrame (legacy
            callers) is converted to the flat representation first.
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
        chunk_sink (callable): optional ``fn(payload_dict, name)`` replacing the .npz write.

    Returns:
        None (writes .npz files).
    """
    cache = bam_df if isinstance(bam_df, _BamCache) else _cache_from_dataframe(bam_df)

    trim = kmer_len // 2
    ## The context window spans 2*(cb_len//2)+1 segments, so for an EVEN --cb-len it does
    ## not equal cb_len and the geometry is self-inconsistent; the original's
    ## `len(segment_len_arr) == cb_len` filter rejected every row. Likewise cb_len <=
    ## 2*trim leaves no token axis at all. Reproduce "no output", loudly, rather than
    ## emitting rows whose signal_token and kmer_token describe different windows.
    if 2 * (cb_len // 2) + 1 != cb_len or cb_len - 2 * trim <= 0:
        log.warning(
            "Geometry --cb-len=%d with --kmer-len=%d yields no valid context window "
            "(cb-len must be odd and > 2*(kmer-len//2)); no tokens will be written.",
            cb_len,
            kmer_len,
        )
        return None
    cfg = {
        "sampling": sampling,
        "cb_len": cb_len,
        "trim": trim,
        "cb_half": cb_len // 2,
        "dwell_shift": dwell_shift,
        "kmer_pad": (kmer_len - 1) // 2,
        "lr_pad": (sig_window - 1) // 2,
        "sig_width": (max_token_len + kmer_len - 1) * sampling,
        "max_token_len": max_token_len,
        "label_div": label_div,
        ## Rows dropped by guards the original lacked; both indicate a BAM whose move
        ## table disagrees with its SEQ, which used to surface as a crash in np.stack.
        "dropped_ragged": 0,
        "dropped_oversized": 0,
    }
    acc = _ChunkAccumulator(
        chunk_size,
        cfg["sig_width"],
        cb_len - 2 * trim,
        cb_len,
        f"{token_output_path}/{pid}",
        chunk_sink=chunk_sink,
    )

    for pod5_path in tqdm.tqdm(pod5_paths, disable=pid != 0):
        try:
            with pod5.Reader(pod5_path) as reader:
                read_ids = np.array(reader.read_ids)
        except Exception:
            log.warning("Corrupted POD5 file: %s", pod5_path)
            continue
        if len(read_ids) == 0:
            continue
        n_batches = int(np.ceil(len(read_ids) / process_once))
        batches = np.array_split(read_ids, n_batches)

        ## Reopening the reader releases the file pages it has mapped. The original paid
        ## an open per 1000-read batch; holding it for a whole 14 GB file instead lets
        ## clean page cache pile up in RSS, so recycle it periodically -- a compromise
        ## that keeps the mapped window bounded at ~1/18 of the per-file reopen cost.
        for group_start in range(0, n_batches, _POD5_REOPEN_BATCHES):
            group = batches[group_start : group_start + _POD5_REOPEN_BATCHES]
            try:
                reader = pod5.Reader(pod5_path)
            except Exception:
                log.warning("Corrupted POD5 file: %s", pod5_path)
                break
            with reader:
                for batch in group:
                    try:
                        signals, key0, key1 = _read_pod5_batch(reader, list(batch))
                    except Exception:
                        log.warning("Corrupted POD5 file: %s", pod5_path)
                        continue
                    if not signals:
                        continue
                    src, rows = cache.lookup(key0, key1)
                    if len(src) == 0:
                        del signals
                        continue
                    for s, row in zip(src, rows):
                        _emit_read(int(row), cache, signals[int(s)], norm_factor, cfg, acc)
                    del signals

    acc.flush(final=True)
    if cfg["dropped_ragged"] or cfg["dropped_oversized"]:
        log.warning(
            "Worker %d dropped %d ragged and %d oversized windows (move table inconsistent with SEQ)",
            pid,
            cfg["dropped_ragged"],
            cfg["dropped_oversized"],
        )
    return None


def _cache_from_dataframe(bam_df):
    """
    Build an in-memory :class:`_BamCache` from a legacy ``parse_bam`` DataFrame.

    Provided so external callers that still hand over a DataFrame keep working; the
    pipeline itself never takes this path.

    Args:
        bam_df (pandas.DataFrame): frame indexed by parent_id with the legacy columns.

    Returns:
        _BamCache: equivalent flat cache backed by in-RAM arrays.
    """
    cache = _BamCache.__new__(_BamCache)
    cache._empty()
    n = len(bam_df)
    if n == 0:
        return cache

    parent = bam_df.index.to_numpy()
    read_ids = bam_df["read_id"].to_numpy()
    pk = np.frombuffer(b"".join(u.bytes for u in parent), dtype=np.uint64).reshape(-1, 2)
    rd = np.frombuffer(b"".join(u.bytes for u in read_ids), dtype=np.int64).reshape(-1, 2)

    seq = np.concatenate([np.asarray(x, np.uint8) for x in bam_df["seq"]])
    bq = np.concatenate([np.asarray(x, np.int8) for x in bam_df["bq"]])
    mv_list = [np.asarray(x, bool) for x in bam_df["mv"]]
    mvp = np.concatenate([np.packbits(m) for m in mv_list])
    ap_list = [np.asarray(x, np.int32).reshape(-1, 2) for x in bam_df["ap"]]
    ap = np.concatenate(ap_list).reshape(-1)

    qlen = np.array([len(x) for x in bam_df["seq"]], np.int32)
    mvlen = np.array([m.size for m in mv_list], np.int32)
    apn = np.array([len(a) for a in ap_list], np.int32)

    order = np.argsort(pk[:, 0], kind="stable")
    cache.pkey0 = np.ascontiguousarray(pk[:, 0][order])
    cache.pkey1 = np.ascontiguousarray(pk[:, 1][order])
    cache.rid0 = np.ascontiguousarray(rd[:, 0][order])
    cache.rid1 = np.ascontiguousarray(rd[:, 1][order])
    cache.qoff = np.ascontiguousarray((np.cumsum(qlen) - qlen).astype(np.int64)[order])
    cache.qlen = np.ascontiguousarray(qlen[order])
    cache.mvoff = np.ascontiguousarray((np.cumsum((mvlen + 7) // 8) - (mvlen + 7) // 8).astype(np.int64)[order])
    cache.mvlen = np.ascontiguousarray(mvlen[order])
    cache.apoff = np.ascontiguousarray((np.cumsum(apn) - apn).astype(np.int64)[order])
    cache.apn = np.ascontiguousarray(apn[order])
    for name in ("ts", "ns", "sp", "ref"):
        cache.__dict__[name] = np.ascontiguousarray(bam_df[name].to_numpy().astype(np.int32)[order])
    cache.strand = np.ascontiguousarray(bam_df["strand"].to_numpy().astype(np.int8)[order])
    cache.shard = np.zeros(n, np.int32)
    cache.blob = {"seq": [seq], "bq": [bq], "mvp": [mvp], "ap": [ap]}
    cache.nbytes_on_disk = seq.nbytes + bq.nbytes + mvp.nbytes + ap.nbytes
    return cache


def df_to_arrays(df):
    """
    Convert a token DataFrame into the dict of stacked arrays stored in a .npz.

    Args:
        df (pandas.DataFrame): DataFrame with token columns.

    Returns:
        dict: arrays keyed by the .npz field names.
    """
    return {
        "segment_len_arr": np.stack(df["segment_len_arr"].values),
        "signal_token": np.stack(df["signal_token"].values),
        "kmer_token": np.stack(df["kmer_token"].values),
        "dwell_motor_token": np.stack(df["dwell_motor_token"].values),
        "dwell_pore_token": np.stack(df["dwell_pore_token"].values),
        "bq_token": np.stack(df["bq_token"].values),
        "label_id": df["label_id"].values,
        "read_id": np.frombuffer(b"".join(u.bytes for u in df["read_id"].values), dtype=np.int64).reshape(-1, 2),
    }


def save_npz(save_path, df):
    """
    Serialize token DataFrame to a compressed .npz file.

    Args:
        save_path (str): Output .npz file path.
        df (pandas.DataFrame): DataFrame with token columns.

    Returns:
        None
    """
    np.savez_compressed(save_path, **df_to_arrays(df))
    return None
