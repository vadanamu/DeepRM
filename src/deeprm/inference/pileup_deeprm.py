"""
Aggregate per-read predictions into site-level results (pileup).

This script performs post-processing on DeepRM prediction files to generate a pileup.
It reads .npz prediction arrays, groups statistics by label IDs, and computes metrics.

The two metrics calculated are:

1. modscore: A score reflecting the site-level modification probability. (0-1 range)
2. stoichiometry: Estimated modification stoichiometry of the site. (0-1 range)

From v1.2, a reporting rule decides which stoichiometries are reported; see "Stoichiometry" in the advanced usage
guide. The same scores are computed for the transcript table and, with ``--annot``, for the genomic table.

Finally, it writes the site tables (pileup.bed and pileup.npz; with ``--annot`` also genomic_pileup<postfix>.bed and
.npz) and, unless ``--skip-modbam`` is given, the modBAM file.
"""

import argparse
import gc
import glob
import itertools
import multiprocessing as mp
import os
import queue
import shutil
import tempfile
import time
import traceback
import uuid
from multiprocessing import resource_tracker
from multiprocessing import shared_memory as shm

import numpy as np
import pandas as pd
import psutil
import pysam
import tqdm
from scipy import stats

from deeprm.inference.pileup_genomic import pileup_genomic
from deeprm.utils.logging import get_logger
from deeprm.utils.utils import BUNDLED_WEIGHTS_MD5, RUN_INFO_NAME, maybe_index_bam, read_run_info

log = get_logger(__name__)

# Workers are forked, so they share the parent's resource tracker (see shm_view). A context, not
# mp.set_start_method(): importing this module does not change the start method of the program that imports it.
_MP = mp.get_context("fork")


def add_arguments(parser: argparse.ArgumentParser):
    """
    Adds command-line arguments.

    Args:
        parser (argparse.ArgumentParser): Argument parser to which arguments will be added.
    Returns:
        None
    """
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Folder of per-read predictions (the molecule-level folder that deeprm call run writes)",
    )
    parser.add_argument("--output", "-o", type=str, required=True, help="Output folder for the site-level files")
    parser.add_argument(
        "--thread", "-t", type=int, default=None, help="Number of threads to use (default: 95%% of the CPUs)"
    )
    parser.add_argument("--bam", "-b", type=str, required=True, help="BAM file of the reads given to deeprm call run")
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
        "-e",
        type=float,
        default=1e-30,
        help="Small constant that keeps the logarithms and divisions of the site scores finite; keep the default",
    )
    parser.add_argument(
        "--postfix",
        "-x",
        type=str,
        default="",
        help="Suffix for the genomic pileup file names (genomic_pileup<postfix>.bed and .npz)",
    )
    parser.add_argument(
        "--slice",
        "-s",
        type=int,
        default=None,
        help="For predictions of a model with several outputs: the output to use, numbered from 0",
    )
    parser.add_argument("--flip", "-f", action="store_true", help="Use 1 - p in place of each per-read probability p")
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
    parser.add_argument(
        "--skip-modbam", "-sm", action="store_true", help="Do not write the modBAM file (per-read MM and ML tags)"
    )
    return None


# ---- BEGIN SITE STOICHIOMETRY RULE: replace this whole block; keep the two signatures ----
# Site-level stoichiometry reporting rule of DeepRM v1.2. Everything in this block is private to it.
#
# The pileup estimates a site's stoichiometry `s` from the per-read probabilities (Eq. 12) and sets it to 0 when the
# site fails the Eq. 14 test at the per-read cut 0.98. The rule always uses this cut (_RULE_CUT), whatever --threshold
# is, because its constants were measured at it; it adds up its own per-site sums at the cut (_RULE_SUMS). With
# n = count_all and k = count_pos_rule (reads at or above the cut), the rule reports a site if and only if
# max(h, l) > 0, where
#   h = E_cal / n - min(s, 1/2)   (only where s > 0): the calibrated confident reads pay for the stoichiometry, capped
#       at one half. E_cal = logsum_1_p_pos_calib / log10(1 - cut), the Eq. 14 evidence on calibrated probabilities.
#   l = F - w(k / n, n)           (only where k >= 1): w is the one-sided 99% Wilson upper bound of the confident-read
#       rate k / n on n reads, and F = 0.00504 is the rate of confident reads (p >= 0.98) on unmodified control
#       reads (measured).
# A term whose condition does not hold does not count. The reported value is s where s > 0, and the confident-read
# fraction k / n otherwise; values of about 0.001-0.006 at depth cannot be told from background. Each site gets one of
# two states:
#   1 (value): the stoichiometry is reported.
#  -1 (not reported): the stoichiometry is written as 0 (NaN in stoichiometry_release).
# Predictions of the bundled v1.2 weights get a fixed calibration in E_cal; predictions of other model files are used
# as they are (see inputs_calibrated). The rule reads no label and changes no detection score (modscore). It has no
# settings: --threshold changes modscore, count_pos, logsum_1_p_pos and stoichiometry_eq14 (the v1.1 value), not the
# rule.
#
# The two functions the rest of the pileup calls, with signatures a replacement rule must keep:
#   _per_read_rule_terms(pred, threshold, epsilon, calibrated) -> {name: per-read array} for every name in _RULE_SUMS
#   _site_stoichiometry(sums, stoich_eq12, threshold) -> (stoichiometry, stoich_state, stoichiometry_release,
#       stoichiometry_eq14); `sums` maps every SITE_SUMS name to its per-site array.
# A new per-site sum for the rule (one int64 or float64 value per site) is one row in _RULE_SUMS plus its entry in
# _per_read_rule_terms; the workers, the genomic pileup and the shared-memory packing then carry it without other
# changes. Rule sums are not written out.
_RULE_CUT = 0.98  # per-read cut of the rule; the constants below were measured at it
_CONFIDENCE = 0.99  # one-sided level of the Wilson bound
_HALF = 0.5  # h: the stoichiometry the confident reads must pay for is capped at one half
_BACKGROUND_RATE = 504 / 100000  # F: confident reads (p >= 0.98) per read on unmodified control reads

_STATE_VALUE = 1  # stoichiometry reported
_STATE_NOT_REPORTED = -1  # written as 0 (NaN in stoichiometry_release)

_CALIBRATION = 1.3088  # fixed part of the v1.2 model; not a setting

# Per-site sums only the rule uses, and their dtypes, all at _RULE_CUT: the confident reads (k), the Eq. 14 evidence,
# and the Eq. 14 evidence on calibrated probabilities.
_RULE_SUMS = (
    ("count_pos_rule", np.int64),
    ("logsum_1_p_pos_rule", np.float64),
    ("logsum_1_p_pos_calib", np.float64),
)


def _sigmoid_np(x):
    """Numerically stable sigmoid in float32."""
    x = np.asarray(x)
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    np.negative(x, out=out, where=pos)
    np.exp(out, out=out, where=pos)
    out[pos] = 1.0 / (1.0 + out[pos])
    np.exp(x, out=out, where=~pos)
    out[~pos] = out[~pos] / (1.0 + out[~pos])
    return out


def _calibrated_probability(pred, calibrated=True):
    """Per-read probabilities as the rule's calibrated Eq. 14 test uses them.

    Args:
        pred (numpy.ndarray): Per-read probabilities.
        calibrated (bool): True for predictions of the bundled v1.2 weights (see inputs_calibrated), which get the
            model's fixed calibration; False returns `pred` unchanged.

    Returns:
        numpy.ndarray: float32 probabilities.
    """
    if not calibrated:
        return np.asarray(pred, dtype=np.float32)
    p = np.clip(np.asarray(pred, dtype=np.float64), 1e-12, 1 - 1e-12)
    logit = (np.log(p) - np.log1p(-p)).astype(np.float32)
    return _sigmoid_np(logit / np.float32(_CALIBRATION))


def _eq14_pass(stoich_eq12, logsum_1_p_pos, count_all, threshold):
    """Eq. 14 test: True where the site's evidence supports its stoichiometry at the per-read cut."""
    return (np.log10(1 - threshold) * stoich_eq12) > (logsum_1_p_pos / count_all)


def _wilson_upper(p, n, z):
    """One-sided Wilson upper bound of the proportion p on n reads, for the normal quantile z."""
    p = np.clip(np.asarray(p, dtype=np.float64), 0, 1)
    n = np.maximum(np.asarray(n, dtype=np.float64), 1.0)
    return (p + z * z / (2 * n) + z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def _apply_b3v(s, count_all, count_pos, e_cal):
    """Applies the reporting rule to a site table.

    Args:
        s (numpy.ndarray): Stoichiometry after the Eq. 14 test (0 where the site failed it).
        count_all (numpy.ndarray): Reads per site (n).
        count_pos (numpy.ndarray): Reads at or above the per-read cut per site (k).
        e_cal (numpy.ndarray): Calibrated evidence E_cal per site; only read where s > 0.

    Returns:
        tuple: (state (int8), release stoichiometry (float64, NaN where not reported)).
    """
    s = np.asarray(s, dtype=np.float64)
    n = np.asarray(count_all, dtype=np.float64)
    k = np.asarray(count_pos, dtype=np.int64)
    e_cal = np.asarray(e_cal, dtype=np.float64)
    n1 = np.maximum(n, 1.0)
    has_s = s > 0
    h_holds = has_s & (e_cal > n * np.minimum(s, _HALF))  # h > 0, multiplied by n
    upper = _wilson_upper(k / n1, n, stats.norm.ppf(_CONFIDENCE))
    l_holds = (k >= 1) & (n > 0) & (upper < _BACKGROUND_RATE)  # l > 0
    reported = h_holds | l_holds
    value = np.where(has_s, s, k / n1)
    state = np.where(reported, _STATE_VALUE, _STATE_NOT_REPORTED).astype(np.int8)
    return state, np.where(reported, value, np.nan)


def _per_read_rule_terms(pred, threshold, epsilon, calibrated):
    """The per-read term of every _RULE_SUMS sum, at the rule's cut _RULE_CUT.

    Args:
        pred (numpy.ndarray): Per-read probabilities (float32), one per row.
        threshold (float): Per-read cut of the other sums; not used, the rule's sums are at _RULE_CUT.
        epsilon (float): Small constant for log safety.
        calibrated (bool): Whether the predictions get the v1.2 model's fixed calibration.

    Returns:
        dict: {name: numpy.ndarray of one value per row} for every name in _RULE_SUMS.
    """
    # the same terms as count_pos and logsum_1_p_pos of _per_read_terms, at _RULE_CUT
    confident = (pred >= _RULE_CUT).astype(np.int64)
    pred_calib = _calibrated_probability(pred, calibrated)
    confident_calib = (pred_calib >= _RULE_CUT).astype(np.int64)
    return {
        "count_pos_rule": confident,
        "logsum_1_p_pos_rule": np.log10(np.clip(1 - pred, epsilon, 1.0)) * confident,
        "logsum_1_p_pos_calib": np.log10(np.clip(1 - pred_calib, epsilon, 1.0)) * confident_calib,
    }


def _site_stoichiometry(sums, stoich_eq12, threshold):
    """Site stoichiometry as written to the BED and NPZ outputs (the reporting rule).

    Args:
        sums (Mapping[str, numpy.ndarray]): Per-site sums, one array per SITE_SUMS name.
        stoich_eq12 (numpy.ndarray): Eq. 12 stoichiometry per site.
        threshold (float): Per-read cut (--threshold). Only the Eq. 14 stoichiometry output uses it; the rule uses
            _RULE_CUT.

    Returns:
        tuple: (reported stoichiometry: the value where the state is 1, else 0; state (int8); release stoichiometry
        with NaN where not reported; Eq. 14 stoichiometry at `threshold`, before the rule, as in v1.1).
    """
    stoich_eq12 = np.asarray(stoich_eq12, dtype=np.float64)
    n = np.asarray(sums["count_all"], dtype=np.float64)
    eq14 = stoich_eq12 * _eq14_pass(stoich_eq12, sums["logsum_1_p_pos"], n, threshold)
    s = stoich_eq12 * _eq14_pass(stoich_eq12, sums["logsum_1_p_pos_rule"], n, _RULE_CUT)
    e_cal = np.asarray(sums["logsum_1_p_pos_calib"], dtype=np.float64) / np.log10(1 - _RULE_CUT)
    state, release = _apply_b3v(s, sums["count_all"], sums["count_pos_rule"], e_cal)
    reported = np.where(state == _STATE_VALUE, release, 0.0)
    return reported, state, release, eq14


# ---- END SITE STOICHIOMETRY RULE ----


# The per-site sums every pileup worker adds up per label_id, and their dtypes: the base sums the scores and the
# outputs use, then the rule's own sums. A new base sum is one row here plus its term in _per_read_terms().
_BASE_SUMS = (
    ("count_all", np.int64),
    ("count_pos", np.int64),
    ("logsum_1_p_pos", np.float64),
    ("kl_div_neg", np.float64),
    ("kl_div_pos", np.float64),
)
SITE_SUMS = _BASE_SUMS + _RULE_SUMS
# The sums written to pileup.npz and genomic_pileup.npz besides count_all and count_pos (rule sums are not written).
_WRITTEN_SUMS = tuple(key for key, _ in _BASE_SUMS if key not in ("count_all", "count_pos"))


def _sums_by_dtype(site_sums):
    """(int64 names, float64 names) of a sum registry. A worker sends its sums in one block per dtype (plus label_id:
    3 shared-memory segments per worker), so every sum must be one int64 or float64 value per site."""
    ints = tuple(key for key, dtype in site_sums if dtype is np.int64)
    floats = tuple(key for key, dtype in site_sums if dtype is np.float64)
    other = [key for key, _ in site_sums if key not in ints + floats]
    if other:
        raise TypeError(f"Site sums must have dtype np.int64 or np.float64: {other}")
    return ints, floats


INT_SUMS, FLOAT_SUMS = _sums_by_dtype(SITE_SUMS)
# modscore is reported in steps of 1 / _DIGITIZATION
_DIGITIZATION = 1000


def _per_read_terms(pred, threshold, epsilon, calibrated):
    """The per-read term of every SITE_SUMS sum except count_all (which counts the rows).

    Args:
        pred (numpy.ndarray): Per-read probabilities (float32), one per row.
        threshold (float): Per-read cut that counts a read as positive.
        epsilon (float): Small constant for log safety.
        calibrated (bool): Whether the predictions get the v1.2 model's fixed calibration in the stoichiometry rule.

    Returns:
        dict: {name: numpy.ndarray of one value per row}.
    """
    count_pos = (pred >= threshold).astype(np.int64)
    kl_div = pred * np.log2(2 * pred + epsilon) + (1 - pred) * np.log2(2 * (1 - pred) + epsilon)
    terms = {
        "count_pos": count_pos,
        "logsum_1_p_pos": np.log10(np.clip(1 - pred, epsilon, 1.0)) * count_pos,
        "kl_div_neg": kl_div * (pred <= 0.5),
        "kl_div_pos": kl_div * (pred > 0.5),
    }
    terms.update(_per_read_rule_terms(pred, threshold, epsilon, calibrated))
    return terms


def _eq12_stoichiometry(kl_div_pos, kl_div_neg, epsilon):
    """Eq. 12: stoichiometry before the Eq. 14 test. modscore uses it too, so it is not part of the rule block."""
    return kl_div_pos / (kl_div_neg + kl_div_pos + epsilon)


def _site_scores(sums, threshold, epsilon):
    """modscore and the stoichiometry outputs of one site table (transcript or genomic) from its per-site sums.

    Args:
        sums (Mapping[str, numpy.ndarray]): Per-site sums, one array per SITE_SUMS name.
        threshold (float): Per-read cut.
        epsilon (float): Small constant for division safety.

    Returns:
        dict: modscore, stoichiometry, stoich_state, stoichiometry_release and stoichiometry_eq14, one value per site.
    """
    count_all = sums["count_all"]
    logsum_1_p_pos = sums["logsum_1_p_pos"]
    stoich_eq12 = _eq12_stoichiometry(sums["kl_div_pos"], sums["kl_div_neg"], epsilon)
    modscore = 1 - np.power(10, logsum_1_p_pos / count_all * (1 + np.power(10, 2 * (stoich_eq12 - 1))))
    modscore = np.digitize(modscore, np.linspace(0, 1, _DIGITIZATION + 1), right=True) / _DIGITIZATION
    stoichiometry, state, release, eq14 = _site_stoichiometry(sums, stoich_eq12, threshold)
    return {
        "modscore": modscore,
        "stoichiometry": stoichiometry,
        "stoich_state": state,
        "stoichiometry_release": release,
        "stoichiometry_eq14": eq14,
    }


def _decode_label_id(label_id, label_div):
    """label_id -> (reference index, 0-based position), as int64 arrays. The strand is the sign of label_id; 1 was
    added to |label_id| during preprocessing so that no label_id is 0."""
    label_id_abs = np.abs(label_id) - 1
    return label_id_abs // label_div, label_id_abs % label_div


def _sorted_unique(x):
    """np.unique(x) of a 1-D integer array, by sort and compare. Same result; NumPy 2.3's np.unique is much slower on
    tens of millions of int64 values."""
    x = np.sort(x)
    if x.size == 0:
        return x
    return x[np.concatenate(([True], x[1:] != x[:-1]))]


def _merge_site_sums(ids_list, sums_list):
    """Adds up per-chunk site sums into one table, in list order.

    Args:
        ids_list (list): label_id array of each chunk; the label_ids of one chunk are unique.
        sums_list (list): For each chunk, its per-site sums ({SITE_SUMS name: array}, same order as its label_ids).

    Returns:
        tuple: (sorted unique label_ids of all chunks, {SITE_SUMS name: summed array}).
    """
    global_ids = _sorted_unique(np.concatenate(ids_list))
    merged = {key: np.zeros(len(global_ids), dtype=dtype) for key, dtype in SITE_SUMS}
    for ids, sums in zip(ids_list, sums_list):
        idx = np.searchsorted(global_ids, ids)
        for key, _ in SITE_SUMS:
            merged[key][idx] += sums[key]
    return global_ids, merged


def _pack_sums(label_id, sums):
    n = len(label_id)
    ints = np.stack([sums[key] for key in INT_SUMS], axis=1) if n else np.empty((0, len(INT_SUMS)), np.int64)
    floats = np.stack([sums[key] for key in FLOAT_SUMS], axis=1) if n else np.empty((0, len(FLOAT_SUMS)), np.float64)
    return {
        "label_id": np.ascontiguousarray(label_id, dtype=np.int64),
        "ints": np.ascontiguousarray(ints, dtype=np.int64),
        "floats": np.ascontiguousarray(floats, dtype=np.float64),
    }


def _unpack_sums(ints, floats):
    sums = {key: ints[:, i] for i, key in enumerate(INT_SUMS)}
    sums.update({key: floats[:, i] for i, key in enumerate(FLOAT_SUMS)})
    return sums


def inputs_calibrated(input_dir):
    """True when the per-read files in `input_dir` come from the bundled v1.2 weights.

    The stoichiometry step gives the predictions of the bundled weights a fixed adjustment, and uses the predictions
    of other model files as they are. The answer is read from deeprm_run.json, which deeprm call run writes beside the
    per-read files; without that file it is False.
    """
    info_path = os.path.join(input_dir, RUN_INFO_NAME)
    if not os.path.isfile(info_path):
        log.warning(
            f"No {RUN_INFO_NAME} in {input_dir}, so the model that made these predictions is unknown; the "
            "stoichiometry rule uses them without the v1.2 model's calibration."
        )
        return False
    info = read_run_info(info_path)
    if "site_rule_calibration" in info:
        bundled = info["site_rule_calibration"] is True
    else:  # a run file of an earlier v1.2 build
        bundled = info.get("model_md5") == BUNDLED_WEIGHTS_MD5
    if bundled:
        return True
    log.warning(
        f"{info_path}: the per-read files come from a model file other than the bundled v1.2 weights, so the "
        "stoichiometry rule uses their predictions without the v1.2 model's calibration."
    )
    return False


def _validate_args(args: argparse.Namespace) -> None:
    """Refuses bad pileup settings before any work starts. deeprm call run calls it too, before the inference."""
    if args.thread is not None and args.thread <= 0:
        raise ValueError("--thread must be a positive integer.")
    if not (0.0 <= args.threshold < 1.0):
        raise ValueError("--threshold must satisfy 0 <= threshold < 1.")
    if args.epsilon <= 0:
        raise ValueError("--epsilon must be positive.")
    if args.label_div <= 0:
        raise ValueError("--label_div must be positive.")
    if args.slice is not None and args.slice < 0:
        raise ValueError("--slice must be >= 0.")
    if args.annot is not None and not os.path.isfile(args.annot):
        raise ValueError(f"--annot: annotation file not found: {args.annot}")
    bam = getattr(args, "bam", None)  # required on the command line; a caller's own Namespace may leave it out
    if bam is not None and not os.path.isfile(bam):
        raise ValueError(f"--bam: BAM file not found: {bam}")
    if bam is not None and not str(bam).lower().endswith(".bam"):
        raise ValueError(f"--bam: expected a .bam file, got: {os.path.basename(bam)}")


def shm_put_array(name: str, arr: np.ndarray):
    """Create a SharedMemory block named `name` and copy `arr` into it."""
    block = shm.SharedMemory(name=name, create=True, size=max(1, arr.nbytes))  # size 0 is not allowed
    view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=block.buf)
    view[...] = arr
    return block, {"name": name, "shape": arr.shape, "dtype": str(arr.dtype)}


def shm_view(meta):
    """
    Open a SharedMemory block and create a numpy view from its buffer.

    The parent process only borrows these segments; workers remain the sole
    owners responsible for unlinking them after the parent acknowledges it is
    done reading.
    """
    block = shm.SharedMemory(name=meta["name"], create=False)
    # The parent starts the resource tracker before it forks the workers (main), so the parent and the workers share
    # one tracker: the worker's unlink() clears the name for both, and nothing is left to warn about at exit.
    arr = np.ndarray(tuple(meta["shape"]), dtype=np.dtype(meta["dtype"]), buffer=block.buf)
    return block, arr


def _savez(path, **arrays):
    """np.savez_compressed, with zlib level 1 instead of 6. The file has the same members (one .npy per array), so
    np.load reads the same arrays; it is written about 3 times faster and is about 10-20% larger. No array is pickled,
    so the file loads without allow_pickle."""
    import zipfile

    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as zf:
        for key, val in arrays.items():
            with zf.open(key + ".npy", "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(val), allow_pickle=False)


def _write_site_table(stem, ref_names, ref_pos, ref_strand, scores, sums):
    """Writes one site table as <stem>.bed (bedMethyl) and <stem>.npz. The transcript and the genomic table both use
    it, so their files have the same format.

    Args:
        stem (str): Output path without the extension.
        ref_names (numpy.ndarray): Reference (transcript or chromosome) name per site.
        ref_pos (numpy.ndarray): 0-based position per site.
        ref_strand (numpy.ndarray): '+' or '-' per site.
        scores (dict): The output of _site_scores for these sites.
        sums (dict): The per-site sums of these sites, one array per SITE_SUMS name.
    """
    ref_names = np.asarray(ref_names, dtype=str)  # str, not object: the NPZ loads without allow_pickle
    ref_strand = np.asarray(ref_strand, dtype=str)
    bed_formatter(
        ref_names=ref_names,
        ref_pos=ref_pos,
        ref_strand=ref_strand,
        modscore=scores["modscore"],
        stoichiometry=scores["stoichiometry"],
        count_all=sums["count_all"],
        count_pos=sums["count_pos"],
        output_path=stem + ".bed",
    )
    _savez(
        stem + ".npz",
        ref_names=ref_names,
        ref_pos=ref_pos,
        ref_strand=ref_strand,
        modscore=scores["modscore"],
        stoichiometry=scores["stoichiometry"],
        count_all=sums["count_all"],
        count_pos=sums["count_pos"],
        stoich_state=scores["stoich_state"],
        stoichiometry_release=scores["stoichiometry_release"],
        stoichiometry_eq14=scores["stoichiometry_eq14"],
        **{key: sums[key] for key in _WRITTEN_SUMS},
    )


def main(args: argparse.Namespace):
    """
    Runs the pileup. Worker processes sum the per-read predictions of their files per site (and save their per-read
    rows for the modBAM); the parent adds the workers' sums up, scores the sites and writes pileup.bed and
    pileup.npz, and with ``--annot`` the genomic tables. Unless ``--skip-modbam`` is given, modBAM writers join the
    rows of each read from all workers and write modbam_<input BAM name> with MM/ML tags.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None: Results are saved in the output directory.
    """
    start = time.time()
    _validate_args(args)
    if args.thread is None:
        args.thread = max(1, int(0.95 * mp.cpu_count()))

    if not args.skip_modbam:  # the modBAM writers read the BAM through its index; without them only the header is read
        args.bam = maybe_index_bam(args.bam, args.thread)

    os.makedirs(args.output, exist_ok=True)
    calibrated = inputs_calibrated(args.input)

    file_paths = sorted(glob.glob(os.path.join(args.input, "*.npz")))
    if len(file_paths) == 0:
        raise ValueError(f"No inference .npz files found in: {args.input}")
    file_paths_split = np.array_split(file_paths, min(args.thread, len(file_paths)))
    # the modBAM step's temporary files (the workers' per-read rows, the writers' shards) go to a folder of their own,
    # so that none of them can have the name of an output (such as modbam_<input BAM name>)
    tmp_dir = None if args.skip_modbam else tempfile.mkdtemp(prefix=".deeprm_modbam_", dir=args.output)

    # start the resource tracker here, so the forked workers share it (see shm_view)
    resource_tracker.ensure_running()
    meta_q = _MP.Queue()
    ack_q = _MP.Queue()
    rows_q = _MP.Queue()
    proc_list = []
    for pid, file_subset in enumerate(file_paths_split):
        proc = _MP.Process(
            target=worker,
            args=(
                pid,
                list(file_subset),
                meta_q,
                ack_q,
                rows_q,
                tmp_dir,
                args.label_div,
                args.slice,
                args.threshold,
                args.epsilon,
                args.flip,
                not args.skip_modbam,
                calibrated,
            ),
        )
        proc.start()
        proc_list.append(proc)

    try:
        metas = _collect_worker_payloads(meta_q, proc_list)
    except RuntimeError:
        _remove_modbam_temp(tmp_dir)
        raise
    rows_stems = []
    if not args.skip_modbam:
        rows_stems = [rows_q.get() for _ in range(len(proc_list))]  # (pid, stem or None), in completion order
        # in worker order: the rows of a read stay in file order when the modBAM writers join them
        rows_stems = [stem for _, stem in sorted(rows_stems, key=lambda item: item[0]) if stem]
    errors = [md for md in metas if md.get("error")]
    if errors:
        _release_workers(metas, ack_q, proc_list)
        _remove_modbam_temp(tmp_dir)
        raise RuntimeError(f"Pileup worker {errors[0]['pid']} failed:\n{errors[0]['error']}")
    # add the workers' sums in worker order, not in the order they finished: a float sum can depend on the order
    metas.sort(key=lambda md: md["pid"])

    ## copy each worker's arrays out of shared memory and close its blocks at once (few open files at a time)
    ids_list, sums_list = [], []
    try:
        for md in tqdm.tqdm(metas, desc="Loading shared memory", leave=False):
            arrays = md["arrays"]
            blocks = []
            try:
                b, ids = shm_view(arrays["label_id"])
                blocks.append(b)
                ids_list.append(np.array(ids))
                b, ints = shm_view(arrays["ints"])
                blocks.append(b)
                b, floats = shm_view(arrays["floats"])
                blocks.append(b)
                sums_list.append(_unpack_sums(np.array(ints), np.array(floats)))
                del ids, ints, floats
            finally:
                for b in blocks:
                    b.close()
    except BaseException:
        # let every worker unlink its blocks and exit, then report the error
        _release_workers(metas, ack_q, proc_list)
        _remove_modbam_temp(tmp_dir)
        raise
    for md in metas:
        ack_q.put(md["pid"])

    if not any(len(ids) for ids in ids_list):
        for proc in proc_list:
            proc.join()
        _remove_modbam_temp(tmp_dir)
        raise ValueError("No valid pileup entries were produced from the inference outputs.")
    label_id, sums = _merge_site_sums(ids_list, sums_list)
    del ids_list, sums_list

    for proc in proc_list:
        proc.join()
        if proc.exitcode != 0:
            _remove_modbam_temp(tmp_dir)
            raise RuntimeError(f"Pileup worker exited with code {proc.exitcode}.")
    gc.collect()

    ## Reference names from the BAM header; label_id -> reference, position and strand
    with pysam.AlignmentFile(args.bam, "rb", check_sq=False) as input_bam:
        ref_arr = np.array(input_bam.references)
    transcript_id, ref_pos = _decode_label_id(label_id, args.label_div)
    if transcript_id.size and np.max(transcript_id) >= len(ref_arr):
        _remove_modbam_temp(tmp_dir)
        raise ValueError("label_id contains transcript IDs outside the BAM reference range.")
    ref_names = ref_arr[transcript_id]
    ref_strand = np.where(label_id > 0, "+", "-")

    if not args.skip_modbam:
        modbam_out_path = os.path.join(args.output, "modbam_" + os.path.basename(args.bam))
        try:
            written = rows_stems and _write_modbam(
                args.bam, rows_stems, _sorted_unique(transcript_id), modbam_out_path, args.thread, tmp_dir
            )
        finally:
            _remove_modbam_temp(tmp_dir)
        if not written:
            log.warning("No valid modBAM entries were produced; skipping modBAM writing.")

    ## Calculate modscore and stoichiometry metrics, and write the transcript table
    scores = _site_scores(sums, args.threshold, args.epsilon)
    _write_site_table(os.path.join(args.output, "pileup"), ref_names, ref_pos, ref_strand, scores, sums)

    if args.annot:
        ## Genomic pileup: the same sums added up per genomic position, scored and written the same way
        input_df = pd.DataFrame({"ref_names": ref_names, "ref_pos": ref_pos, "ref_strand": ref_strand, **sums})
        genomic_df = pileup_genomic(input_df, args.annot, [key for key, _ in SITE_SUMS])
        genomic_sums = {key: genomic_df[key].values for key, _ in SITE_SUMS}
        _write_site_table(
            os.path.join(args.output, f"genomic_pileup{args.postfix}"),
            genomic_df["chrom"].values,
            genomic_df["pos"].values,
            genomic_df["strand"].values,
            _site_scores(genomic_sums, args.threshold, args.epsilon),
            genomic_sums,
        )

    elapsed = time.time() - start
    log.info(f"Pileup finished in {elapsed:.2f} seconds.")
    return None


def _collect_worker_payloads(meta_q, proc_list, poll_s=5.0, name="Pileup worker"):
    """One payload per worker (its `pid` is its index in proc_list); raises if a worker exited without sending one,
    instead of waiting forever. `name` names the workers in the error."""
    metas = []
    while len(metas) < len(proc_list):
        try:
            metas.append(meta_q.get(timeout=poll_s))
        except queue.Empty:
            delivered = {md["pid"] for md in metas}
            lost = [(i, proc) for i, proc in enumerate(proc_list) if proc.exitcode is not None and i not in delivered]
            if lost:
                for proc in proc_list:
                    if proc.is_alive():
                        proc.terminate()
                errors = [md for md in metas if md.get("error")]
                if errors:
                    raise RuntimeError(f"{name} {errors[0]['pid']} failed:\n{errors[0]['error']}")
                i, proc = lost[0]
                raise RuntimeError(f"{name} {i} exited with code {proc.exitcode} without sending its results.")
    return metas


def _release_workers(metas, ack_q, proc_list, timeout_s=30.0):
    """After a failure in the parent: acknowledge every payload (so each worker unlinks its blocks), then stop them."""
    for md in metas:
        if not md.get("error"):
            ack_q.put(md["pid"])
    for proc in proc_list:
        proc.join(timeout=timeout_s)
        if proc.is_alive():
            proc.terminate()
            proc.join()


def _remove_modbam_temp(tmp_dir):
    """Removes the folder of the modBAM step's temporary files (after the step, or after a failure); None: no folder."""
    if tmp_dir is not None:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def grouped_sum(n_unique, idx, vals):
    """
    Sum values in 'vals' according to group indices 'idx'.

    Args:
        n_unique (int): Number of unique groups.
        idx (numpy.ndarray): Integer indices mapping each element in 'vals' to a group.
        vals (numpy.ndarray): Values to sum per group.

    Returns:
        numpy.ndarray: Array of summed values of length n_unique.
    """
    group_sums = np.zeros((n_unique,), dtype=vals.dtype)
    np.add.at(group_sums, idx, vals)
    return group_sums


def _publish(pid, arrays, meta_q, ack_q):
    """Puts a worker's arrays into shared memory, sends their description and waits for the parent to read them."""
    base = f"deeprm_pileup_{pid}_{uuid.uuid4().hex}"
    blocks = []
    metas = {}
    try:
        for key, arr in arrays.items():
            block, meta = shm_put_array(f"{base}_{key}", arr)
            blocks.append(block)
            metas[key] = meta
    except BaseException:
        for b in blocks:
            b.close()
            b.unlink()
        raise  # nothing was sent yet: worker() reports the error
    parent = os.getppid()
    meta_q.put({"pid": pid, "arrays": metas})
    try:
        while True:  # wait for the parent's acknowledgement; stop waiting if the parent is gone
            try:
                ack_q.get(timeout=5.0)
                break
            except queue.Empty:
                if os.getppid() != parent:
                    break
        for b in blocks:
            b.close()
            b.unlink()
    except Exception as exc:
        raise _PublishedError(str(exc)) from exc


def worker(
    pid,
    file_paths,
    meta_q,
    ack_q,
    rows_q,
    rows_dir,
    label_div,
    slice_idx,
    threshold_pos,
    epsilon,
    flip,
    make_modbam,
    calibrated,
):
    """Runs one pileup worker: sums the per-read predictions of `file_paths` per site, saves its per-read rows for the
    modBAM when asked, and sends the site sums to the parent. Any error is sent to the parent instead of leaving it
    waiting."""
    try:
        _worker(
            pid,
            file_paths,
            meta_q,
            ack_q,
            rows_q,
            rows_dir,
            label_div,
            slice_idx,
            threshold_pos,
            epsilon,
            flip,
            make_modbam,
            calibrated,
        )
    except _PublishedError:
        raise
    except BaseException:  # noqa: BLE001 -- reported to the parent, then re-raised
        if make_modbam:
            rows_q.put((pid, None))
        meta_q.put({"pid": pid, "error": traceback.format_exc()})
        raise SystemExit(1)


class _PublishedError(Exception):
    """An error after the worker already sent its payload (the parent has it; do not send a second one)."""


def _worker(
    pid,
    file_paths,
    meta_q,
    ack_q,
    rows_q,
    rows_dir,
    label_div,
    slice_idx,
    threshold_pos,
    epsilon,
    flip,
    make_modbam,
    calibrated,
):
    """
    Sums the per-read predictions of `file_paths` per site, saves this worker's per-read rows for the modBAM when
    asked (the modBAM writers read them; see _write_modbam), and sends the site sums to the parent through shared
    memory. Worker 0 shows the progress bars.

    Args:
        pid (int): Worker index (its index in the parent's process list).
        file_paths (list): The .npz per-read files of this worker.
        meta_q (multiprocessing.Queue): Sends the description of the shared-memory blocks (or the error) to the parent.
        ack_q (multiprocessing.Queue): The parent's acknowledgement that it has read the blocks.
        rows_q (multiprocessing.Queue): (pid, stem of this worker's saved per-read rows, or None when it has none).
        rows_dir (str): Folder for the saved per-read rows (None when make_modbam is False).
        label_div (int): Divisor that separates the reference index and the position in label_id.
        slice_idx (int): Column of 2-D predictions, or None for 1-D predictions.
        threshold_pos (float): Per-read cut that counts a read as positive.
        epsilon (float): Small constant for log and division safety.
        flip (bool): Whether to invert probabilities (1 - p).
        make_modbam (bool): Whether to save the per-read rows for the modBAM.
        calibrated (bool): Whether the predictions get the v1.2 model's fixed calibration in the stoichiometry rule.
    """
    ids_list, sums_list = [], []
    modbam_rows = _ReadRows() if make_modbam else None

    for path in tqdm.tqdm(file_paths, desc="Reading input files", leave=False, disable=pid != 0):
        with np.load(path) as data:
            pred = np.asarray(data["pred"])
            label_id = np.asarray(data["label_id"])
            read_id = np.asarray(data["read_id"]) if make_modbam else None  # only the modBAM needs it

        assert pred.dtype == np.float32, f"Expected pred to be float32, but got {pred.dtype} in {path}"
        # int64 only: the sign of label_id is the strand
        assert label_id.dtype == np.int64, f"Expected label_id to be int64, but got {label_id.dtype} in {path}"
        assert pred.shape[0] == label_id.shape[0], f"Length of pred and label_id do not match in {path}"
        if make_modbam:
            assert read_id.shape[0] == label_id.shape[0], f"Length of read_id and label_id do not match in {path}"

        if slice_idx is not None:
            assert pred.ndim == 2, f"Expected pred to be 2D as --slice was given, but got {pred.ndim} in {path}"
            if slice_idx >= pred.shape[1]:
                raise IndexError(f"Slice index {slice_idx} is out of bounds for pred with shape {pred.shape} in {path}")
            valid_idx = np.all(np.isfinite(pred), axis=1)
            pred = pred[valid_idx, slice_idx]
        else:
            assert pred.ndim == 1, f"Expected pred to be 1D as --slice was not given, but got {pred.ndim} in {path}"
            valid_idx = np.isfinite(pred)
            pred = pred[valid_idx]

        label_id = label_id[valid_idx]
        if make_modbam:
            read_id = read_id[valid_idx]

        if flip:
            pred = 1 - pred

        if pred.size == 0:
            continue

        pred_min = float(np.min(pred))
        pred_max = float(np.max(pred))
        assert pred_min >= 0.0, f"Minimum value of pred is {pred_min} in {path}"
        assert pred_max <= 1.0, f"Maximum value of pred is {pred_max} in {path}"

        if make_modbam:
            ref_ids, poss = _decode_label_id(label_id, label_div)
            modbam_rows.add(ref_ids, read_id, poss, np.clip(np.rint(pred * 255), 0, 255).astype(np.uint8))

        terms = _per_read_terms(pred, threshold_pos, epsilon, calibrated)
        unique_id, id_idx, count_all = np.unique(label_id, return_inverse=True, return_counts=True)
        n_unique = len(unique_id)
        sums = {"count_all": count_all.astype(np.int64, copy=False)}
        for key, _ in SITE_SUMS:
            if key != "count_all":
                sums[key] = grouped_sum(n_unique, id_idx, terms[key])
        ids_list.append(unique_id)
        sums_list.append(sums)

    if ids_list:
        payload = _pack_sums(*_merge_site_sums(ids_list, sums_list))
    else:  # no valid row in these files: an empty payload
        payload = _pack_sums(np.empty(0, np.int64), {key: np.empty(0, dtype) for key, dtype in SITE_SUMS})

    if make_modbam:
        stem = os.path.join(rows_dir, f"modbam_rows_{pid}")
        rows_q.put((pid, stem if modbam_rows.save_by_reference(stem) else None))  # filled only from valid rows

    del modbam_rows, ids_list, sums_list
    gc.collect()

    _publish(pid, payload, meta_q, ack_q)
    return None


def _modbam_buckets(bam_path, ref_ids, n):
    """Splits the references that have rows and aligned reads into at most `n` runs of consecutive references with
    about the same number of aligned reads (from the BAM index). Returns a list of reference-index arrays."""
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        mapped = np.zeros(bam.nreferences, dtype=np.int64)
        for stat in bam.get_index_statistics():
            mapped[bam.get_tid(stat.contig)] = stat.mapped
    ref_ids = np.asarray(ref_ids, dtype=np.int64)
    ref_ids = ref_ids[(ref_ids >= 0) & (ref_ids < len(mapped))]
    ref_ids = ref_ids[mapped[ref_ids] > 0]
    if ref_ids.size == 0:
        return []
    weight = np.cumsum(mapped[ref_ids])
    n = min(n, len(ref_ids))
    cuts = np.searchsorted(weight, weight[-1] * np.arange(1, n) / n, side="left") + 1
    cuts = np.unique(np.clip(cuts, 1, len(ref_ids) - 1)) if len(ref_ids) > 1 else []
    return [refs for refs in np.split(ref_ids, cuts) if refs.size]


def _modbam_writer(bucket, refs, rows_stems, bam_path, shard_path, result_q):
    """One modBAM writer: joins the rows of references `refs` from every worker and writes them as one shard."""
    try:
        rows = _ReadRows.load(rows_stems, int(refs[0]), int(refs[-1]) + 1)
        path = None
        if len(rows):
            write_modbam_worker_from_map(bam_path, shard_path, rows, progress=bucket == 0, references=refs)
            path = shard_path
        result_q.put({"pid": bucket, "path": path})
    except BaseException:  # noqa: BLE001 -- reported to the parent, then re-raised
        result_q.put({"pid": bucket, "error": traceback.format_exc()})
        raise SystemExit(1)


def _write_modbam(bam_path, rows_stems, ref_ids, out_path, threads, tmp_dir):
    """Writes the modBAM from the workers' saved rows, with the shards in `tmp_dir`. Returns False when no read got a
    record.

    The references with rows are split into runs of consecutive references (_modbam_buckets); one writer per run joins
    the rows of its references from every worker (in worker order, so the rows of a read stay in file order), fetches
    only those references from the indexed BAM and writes a shard. A read whose rows came from several workers gets
    one record per alignment. The shards are then merged in reference order.
    """
    buckets = _modbam_buckets(bam_path, ref_ids, threads)
    if not buckets:
        return False
    result_q = _MP.Queue()
    procs = []
    for b, refs in enumerate(buckets):
        shard_path = os.path.join(tmp_dir, f"modbam_shard_{b}.bam")
        proc = _MP.Process(target=_modbam_writer, args=(b, refs, rows_stems, bam_path, shard_path, result_q))
        proc.start()
        procs.append(proc)
    try:
        results = _collect_worker_payloads(result_q, procs, name="modBAM writer")
    finally:
        for proc in procs:
            proc.join()
    errors = [r for r in results if r.get("error")]
    if errors:
        raise RuntimeError(f"modBAM writer {errors[0]['pid']} failed:\n{errors[0]['error']}")
    shards = [r["path"] for r in sorted(results, key=lambda r: r["pid"]) if r["path"]]
    if not shards:
        return False
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        coordinate_sorted = bam.header.to_dict().get("HD", {}).get("SO") == "coordinate"
    # the shards hold consecutive references in order, each fetched in coordinate order from the indexed BAM, so
    # their merge is sorted already; with a header that says so, the sort is skipped
    merge_modbam_shards(shards, out_path, threads, presorted=coordinate_sorted)
    return True


def merge_modbam_shards(shard_paths, out_path, threads, presorted=False):
    """Merges modBAM shards (in the order given) into one coordinate-sorted, indexed modBAM, and removes the shards.

    Every shard has the input BAM's header, so -c / -p keep its @RG and @PG IDs (and the reads' RG tags) instead of
    giving each shard its own renamed copy. With `presorted` (the shards hold consecutive references in order, each
    coordinate-sorted, under a header that says SO:coordinate) the merge is the output; it is sorted only if indexing
    it fails.
    """
    shard_paths = [p for p in shard_paths if p and os.path.exists(p)]
    if not shard_paths:
        return None

    unsorted_path = out_path + ".unsorted.bam"
    pysam_thread_memory = max(1, int(0.5 * psutil.virtual_memory().available / (1024**2) / threads))
    if presorted:
        pysam.merge("-@", str(threads), "-c", "-p", "-f", out_path, *shard_paths)
        try:
            pysam.index("-@", str(threads), out_path)
        except pysam.SamtoolsError:  # not sorted after all: sort it as below
            os.replace(out_path, unsorted_path)
            presorted = False
    else:
        pysam.merge("-@", str(threads), "-c", "-p", "-f", unsorted_path, *shard_paths)
    if not presorted:
        pysam.sort("-@", str(threads), f"-m {pysam_thread_memory}M", "-o", out_path, unsorted_path)
        pysam.index("-@", str(threads), out_path)
        os.remove(unsorted_path)

    for path in shard_paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    return None


def bed_formatter(ref_names, ref_pos, ref_strand, modscore, stoichiometry, count_all, count_pos, output_path):
    """
    Writes one site table as a bedMethyl file.

    The lines are the ones pandas writes (DataFrame.to_csv with tabs and float_format "%.2f"); pyarrow's CSV writer
    writes them several times faster. Names that would need CSV quoting (a tab, a quote or a line break), and a
    pyarrow that cannot write the table, use pandas.

    Args:
        ref_names (numpy.ndarray): Array of reference names.
        ref_pos (numpy.ndarray): Array of reference positions.
        ref_strand (numpy.ndarray): Array of reference strands ('+' or '-').
        modscore (numpy.ndarray): modscore scores.
        stoichiometry (numpy.ndarray): stoichiometry scores.
        count_all (numpy.ndarray): Total counts.
        count_pos (numpy.ndarray): Positive counts.
        output_path (str): The BED file to write.

    Returns:
        None
    """
    ref_names = np.asarray(ref_names)
    ref_pos = np.asarray(ref_pos, dtype=np.int64)
    count_all = np.asarray(count_all, dtype=np.int64)
    count_pos = np.asarray(count_pos, dtype=np.int64)
    stoichiometry = np.asarray(stoichiometry, dtype=np.float64)
    modscore = np.asarray(modscore, dtype=np.float64)
    ref_strand = np.asarray(ref_strand).astype(str)

    estimated_mod_count = np.clip(np.rint(count_all * stoichiometry).astype(np.int64), 0, count_all)
    columns = {
        "col1": ref_names,
        "col2": ref_pos,
        "col3": ref_pos + 1,
        "col4": "a",
        "col5": np.clip(np.rint(modscore * 1000).astype(np.int64), 0, 1000),
        "col6": ref_strand,
        "col7": ref_pos,
        "col8": ref_pos + 1,
        "col9": "255,0,0",
        "col10": count_all,
        "col11": stoichiometry * 100,
        "col12": estimated_mod_count,
        "col13": count_all - estimated_mod_count,
        "col14": 0,
        "col15": 0,
        "col16": 0,
        "col17": 0,
        "col18": 0,
    }
    if not _write_bed_pyarrow(columns, len(ref_pos), output_path):
        _write_bed_pandas(columns, len(ref_pos), output_path)
    return None


# Characters that make pandas quote a text field (tab = the separator, the quote character, line breaks)
_CSV_QUOTED = ("\t", '"', "\n", "\r")


def _two_decimals(x):
    """'%.2f' % v for every value of `x` (NaN as '', as pandas writes it), as a pyarrow dictionary array.

    A value in [0, 100] whose hundredths are not within 1e-6 of a rounding tie has one of 10,001 texts, found by
    rounding. Every other value (a near-tie, a negative value or -0.0, anything above 100, inf) is formatted by
    Python's '%.2f', which is what pandas calls.
    """
    import pyarrow as pa

    x = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        scaled = x * 100.0
        k = np.floor(scaled + 0.5)
        near_tie = np.abs(scaled - np.floor(scaled) - 0.5) < 1e-6
        in_table = np.isfinite(scaled) & ~near_tie & (k >= 0) & (k <= 10000) & ~np.signbit(x)
    words = ["%.2f" % (i / 100) for i in range(10001)]
    codes = np.zeros(len(x), dtype=np.int32)
    codes[in_table] = k[in_table].astype(np.int32)
    rest = np.flatnonzero(~in_table)
    if rest.size:
        # the %-formatting pandas applies with float_format="%.2f"; NaN is written as ""
        texts = np.array(["" if v != v else "%.2f" % v for v in x[rest].tolist()], dtype=object)  # noqa: UP031
        uniq, inv = np.unique(texts, return_inverse=True)
        codes[rest] = len(words) + inv.astype(np.int32)
        words += uniq.tolist()
    return pa.DictionaryArray.from_arrays(pa.array(codes), pa.array(words, type=pa.string()))


def _text_column(values, n):
    """A text column as a pyarrow dictionary array, and its distinct texts."""
    import pyarrow as pa

    if isinstance(values, str):
        words = [values]
        codes = np.zeros(n, dtype=np.int32)
    else:
        codes, uniq = pd.factorize(np.asarray(values), sort=False)
        words = [str(w) for w in uniq]
        codes = codes.astype(np.int32)
    return pa.DictionaryArray.from_arrays(pa.array(codes), pa.array(words, type=pa.string())), words


def _write_bed_pyarrow(columns, n, output_path):
    """Writes the BED lines with pyarrow. Returns False, having written nothing, when pandas must write them."""
    try:
        import pyarrow as pa
        import pyarrow.csv as pacsv
    except ImportError:  # e.g. a pyarrow built for NumPy 1.x; _two_decimals and _text_column run only after this
        return False

    table = {}
    for name, values in columns.items():
        if name == "col11":
            table[name] = _two_decimals(values)
        elif isinstance(values, str) or np.asarray(values).dtype.kind in "OUS":
            table[name], words = _text_column(values, n)
            if any(c in w for w in words for c in _CSV_QUOTED):
                return False
        elif isinstance(values, int):
            table[name] = pa.array(np.full(n, values, dtype=np.int64))
        else:
            table[name] = pa.array(np.asarray(values, dtype=np.int64))
    options = pacsv.WriteOptions(include_header=False, delimiter="\t", quoting_style="none")
    try:
        pacsv.write_csv(pa.table(table), output_path, write_options=options)
    except (pa.ArrowException, TypeError, ValueError):
        return False  # a pyarrow that cannot write this table: pandas writes the file again
    return True


def _write_bed_pandas(columns, n, output_path):
    df = pd.DataFrame({name: [v] * n if isinstance(v, str) else v for name, v in columns.items()})
    for name in ("col14", "col15", "col16", "col17", "col18"):
        df[name] = np.zeros(n, dtype=np.int64)
    df.to_csv(output_path, sep="\t", header=False, index=False, float_format="%.2f")


_RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    """Reverse-complement a DNA string (non-ACGTN chars are passed through)."""
    return seq.translate(_RC_TABLE)[::-1]


def get_mm_tag(q_pos, preds, seq, base="A", mod="a"):
    q_pos = np.asarray(q_pos, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.uint8)

    if q_pos.size == 0:
        return f"{base}+{mod}?;", []

    order = np.argsort(q_pos, kind="stable")
    q_pos = q_pos[order]
    preds = preds[order]
    # A read base can come twice, when two records of one read align read bases to the same reference base: rows are
    # matched to a record by (reference, read) only, so which row is this record's own is unknown. MM lists each base
    # once: with the call its rows agree on, and not at all when they disagree.
    first = np.ones(q_pos.size, dtype=bool)
    first[1:] = q_pos[1:] != q_pos[:-1]
    starts = np.flatnonzero(first)
    agree = np.minimum.reduceat(preds, starts) == np.maximum.reduceat(preds, starts)
    q_pos, preds = q_pos[first][agree], preds[first][agree]

    base_positions = np.flatnonzero(np.frombuffer(seq.encode("latin-1", "replace"), dtype=np.uint8) == ord(base))
    if base_positions.size == 0:
        return f"{base}+{mod}?;", []

    keep = np.isin(q_pos, base_positions)
    q_pos = q_pos[keep]
    preds = preds[keep]
    if q_pos.size == 0:
        return f"{base}+{mod}?;", []

    idx = np.searchsorted(base_positions, q_pos)
    run_lengths = np.empty_like(idx)
    run_lengths[0] = idx[0]
    if len(idx) > 1:
        run_lengths[1:] = np.diff(idx) - 1
    mm_tag = f"{base}+{mod}?,{','.join(map(str, run_lengths.tolist()))};"
    ml_tag = preds.tolist()
    return mm_tag, ml_tag


# A saved per-read row (the reference index is saved apart, sorted, so a run of references is found by bisection)
_ROW_DTYPE = np.dtype([("hi", "<i8"), ("lo", "<i8"), ("pos", "<i8"), ("pred", "u1")])


class _ReadRows:
    """Per-read rows for the modBAM, as numpy arrays.

    Rows are grouped by (reference index, read id high, read id low); within a read they stay in the order they were
    added (file order, then row order), as the earlier dict of lists kept them.
    """

    def __init__(self):
        self._parts = []
        self._index = {}

    def add(self, ref_ids, read_id, positions, preds_u8):
        """Adds one file's rows: reference index, read id (n x 2 int64), reference position and uint8 probability."""
        self._parts.append(
            (
                np.asarray(ref_ids, dtype=np.int64),
                np.asarray(read_id[:, 0], dtype=np.int64),
                np.asarray(read_id[:, 1], dtype=np.int64),
                np.asarray(positions, dtype=np.int64),
                np.asarray(preds_u8, dtype=np.uint8),
            )
        )

    def save_by_reference(self, stem):
        """Saves the added rows as <stem>.ref.npy (reference index) and <stem>.rows.npy, sorted by reference index
        (stably: the rows of a read keep their order). Returns False, saving nothing, when there are no rows."""
        if not self._parts:
            return False
        ref, hi, lo, pos, pred = (np.concatenate([part[i] for part in self._parts]) for i in range(5))
        self._parts = []
        order = np.argsort(ref, kind="stable")
        rows = np.empty(len(ref), dtype=_ROW_DTYPE)
        rows["hi"], rows["lo"], rows["pos"], rows["pred"] = hi[order], lo[order], pos[order], pred[order]
        np.save(stem + ".ref.npy", ref[order])
        np.save(stem + ".rows.npy", rows)
        return True

    @classmethod
    def load(cls, stems, ref_lo, ref_hi):
        """The rows with reference index in [ref_lo, ref_hi) of the saved tables `stems`, added in the order of
        `stems`, grouped by read."""
        table = cls()
        for stem in stems:
            ref = np.load(stem + ".ref.npy", mmap_mode="r")
            a, b = (int(i) for i in np.searchsorted(ref, [ref_lo, ref_hi], side="left"))
            if b > a:
                rows = np.load(stem + ".rows.npy", mmap_mode="r")[a:b]
                read_id = np.stack([rows["hi"], rows["lo"]], axis=1)
                table.add(np.array(ref[a:b]), read_id, np.array(rows["pos"]), np.array(rows["pred"]))
        return table.finish()

    @classmethod
    def from_map(cls, data_map):
        """A table from a dict {(reference index, read id high, read id low): (positions, uint8 probabilities)}."""
        rows = cls()
        for (ref_id, hi, lo), (positions, preds) in data_map.items():
            n = len(positions)
            read_id = np.empty((n, 2), dtype=np.int64)
            read_id[:, 0], read_id[:, 1] = hi, lo
            rows.add(np.full(n, ref_id, dtype=np.int64), read_id, positions, preds)
        return rows.finish()

    def finish(self):
        """Groups the rows by read; call once, after the last add()."""
        if not self._parts:
            self._pos = np.empty(0, np.int64)
            self._pred = np.empty(0, np.uint8)
            return self
        ref, hi, lo, pos, pred = (np.concatenate([part[i] for part in self._parts]) for i in range(5))
        self._parts = []
        order = np.lexsort((lo, hi, ref))  # lexsort is stable: the rows of a read keep their order
        ref, hi, lo, self._pos, self._pred = ref[order], hi[order], lo[order], pos[order], pred[order]
        new = np.concatenate(([True], (ref[1:] != ref[:-1]) | (hi[1:] != hi[:-1]) | (lo[1:] != lo[:-1])))
        start = np.flatnonzero(new)
        end = np.append(start[1:], len(ref))
        self._index = {
            (a, b, c): (i, j)
            for a, b, c, i, j in zip(
                ref[start].tolist(), hi[start].tolist(), lo[start].tolist(), start.tolist(), end.tolist()
            )
        }
        return self

    def __len__(self):
        return len(self._index)

    def get(self, key):
        """(positions, uint8 probabilities) of one (reference index, read id high, read id low), or None."""
        span = self._index.get(key)
        if span is None:
            return None
        return self._pos[span[0] : span[1]], self._pred[span[0] : span[1]]

    def read_names(self):
        """{read name as written by the basecaller (canonical UUID text): (read id high, read id low)}."""
        return {
            str(uuid.UUID(bytes=np.array([hi, lo], dtype=np.int64).tobytes())): (hi, lo) for _, hi, lo in self._index
        }


def _is_canonical_uuid(name):
    """True for lower-case 8-4-4-4-12 text: such a name not in _ReadRows.read_names() cannot be one of its reads."""
    return len(name) == 36 and name[8] == name[13] == name[18] == name[23] == "-" and name == name.lower()


# CIGAR operations (pysam codes 0-8: M I D N S H P = X) that consume the query, the reference, or align a base to one.
# P (6) never reaches these tables: _query_positions leaves reads with padding to get_aligned_pairs().
_CONSUMES_QUERY = np.zeros(9, dtype=bool)
_CONSUMES_QUERY[[0, 1, 4, 7, 8]] = True
_CONSUMES_REF = np.zeros(9, dtype=bool)
_CONSUMES_REF[[0, 2, 3, 7, 8]] = True
_ALIGNED = np.zeros(9, dtype=bool)
_ALIGNED[[0, 7, 8]] = True


def _query_positions(read, ref_pos):
    """Query positions of the reference positions `ref_pos` of an aligned read, from its CIGAR blocks.

    Returns:
        tuple: (query positions, mask of those aligned to a base), the pairs read.get_aligned_pairs() gives (matches
        and mismatches; not insertions, deletions or skipped regions). None for a read without CIGAR, with padding (P:
        pysam's get_aligned_pairs() counts padded bases as query bases) or with an operation outside M I D N S H = X;
        the caller then uses get_aligned_pairs() itself.
    """
    cigar = read.cigartuples
    if not cigar:
        return None
    ops = np.asarray(cigar, dtype=np.int64)
    op, length = ops[:, 0], ops[:, 1]
    if np.any((op < 0) | (op > 8) | (op == 6)):
        return None
    q_len = length * _CONSUMES_QUERY[op]
    r_len = length * _CONSUMES_REF[op]
    q_start = np.cumsum(q_len) - q_len
    r_start = read.reference_start + np.cumsum(r_len) - r_len
    aligned = _ALIGNED[op]
    a_r, a_q, a_len = r_start[aligned], q_start[aligned], length[aligned]
    ref_pos = np.asarray(ref_pos, dtype=np.int64)
    if a_r.size == 0:
        return np.zeros(len(ref_pos), dtype=np.int64), np.zeros(len(ref_pos), dtype=bool)
    block = np.searchsorted(a_r, ref_pos, side="right") - 1
    ok = block >= 0
    block = np.where(ok, block, 0)
    ok &= ref_pos < a_r[block] + a_len[block]
    return a_q[block] + (ref_pos - a_r[block]), ok


def write_modbam_worker_from_map(in_path, out_path, data_map, progress=True, references=None):
    """Writes a modBAM shard: every read of `in_path` that has rows in `data_map`, with MM/ML tags from those rows.

    Args:
        in_path (str): Input BAM.
        out_path (str): The shard to write (same header as the input).
        data_map: The rows, as a dict {(reference index, read id high, read id low): (reference positions, uint8
            probabilities)}, or as the table of per-read rows that the pileup workers save.
        progress (bool): Show a progress bar.
        references (list): Only the reads aligned to these reference indices, in this order (fetched from the
            index when the BAM has one). None: the whole BAM, in file order.
    """
    if data_map is None or len(data_map) == 0:
        return None
    rows = data_map if isinstance(data_map, _ReadRows) else _ReadRows.from_map(data_map)
    names = rows.read_names()

    in_bam = pysam.AlignmentFile(in_path, "rb")
    out_bam = pysam.AlignmentFile(out_path, "wb", template=in_bam)
    # mapped/unmapped come from the index; without one the bar shows no total
    total = (in_bam.mapped or 0) + (in_bam.unmapped or 0) if in_bam.has_index() else None
    reads = in_bam
    if references is not None:
        references = [int(tid) for tid in references]
        if in_bam.has_index():
            mapped = {in_bam.get_tid(stat.contig): stat.mapped for stat in in_bam.get_index_statistics()}
            total = sum(mapped.get(tid, 0) for tid in references)
            reads = itertools.chain.from_iterable(in_bam.fetch(tid=tid) for tid in references)
        else:  # no index: scan the whole BAM for these references
            wanted = set(references)
            reads = (read for read in in_bam if read.reference_id in wanted)
    for read in tqdm.tqdm(reads, total=total, leave=False, disable=not progress):
        read_id = read.query_name
        if read_id is None or read.query_sequence is None:
            continue
        hi_lo = names.get(read_id)
        if hi_lo is None:
            if _is_canonical_uuid(read_id):
                continue  # a read with no rows in this table
            try:  # other spellings of a UUID (upper case, no hyphens, ...)
                read_id_high, read_id_low = np.frombuffer(uuid.UUID(read_id).bytes, dtype=np.int64)
            except (ValueError, AttributeError):
                continue
            hi_lo = (int(read_id_high), int(read_id_low))
        ref_id = read.reference_id
        if ref_id < 0:
            continue

        payload = rows.get((int(ref_id), hi_lo[0], hi_lo[1]))
        if payload is None:
            continue

        pos_arr, pred_arr = payload
        mapped = _query_positions(read, pos_arr)
        if mapped is None:
            mapping_rpos_to_qpos = {r: q for q, r in read.get_aligned_pairs() if r is not None and q is not None}
            qpos = [mapping_rpos_to_qpos.get(int(r)) for r in pos_arr.tolist()]
            keep = np.array([q is not None for q in qpos], dtype=bool)
            qpos = np.array([q for q in qpos if q is not None], dtype=np.int64)
            pred = pred_arr[keep]
        else:
            qpos, ok = mapped
            qpos, pred = qpos[ok], pred_arr[ok]

        # MM/ML are defined relative to the ORIGINAL (basecalled) read
        # orientation. For reverse-strand alignments the BAM stores SEQ
        # reverse-complemented and get_aligned_pairs' query positions index
        # that forward-reference SEQ, so recover the original orientation:
        # reverse-complement SEQ and mirror the query positions. The modified
        # base is the RNA adenosine ('A') in that original orientation.
        seq = str(read.query_sequence)
        if read.is_reverse:
            qpos = len(seq) - 1 - qpos
            seq = reverse_complement(seq)

        mm_tag, ml_tag = get_mm_tag(qpos, pred, seq)
        read.set_tag("MM", mm_tag, "Z")
        if read.has_tag("MN"):
            # MN is the SEQ length MM and ML refer to; an MN the basecaller wrote before hard clipping no longer fits
            read.set_tag("MN", len(seq), "i")
        if len(ml_tag) > 0:
            read.set_tag("ML", ml_tag)
        else:
            read.set_tag("ML", None)  # drop any ML tag the basecaller wrote, so MM and ML always agree
        out_bam.write(read)
    in_bam.close()
    out_bam.close()
    return None
