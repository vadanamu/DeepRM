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
import array
import collections
import concurrent.futures
import contextlib
import ctypes
import errno
import glob
import io
import itertools
import math
import multiprocessing as mp
import os
import queue
import resource
import shutil
import signal
import struct
import sys
import tempfile
import threading
import time
import traceback
import uuid
import zlib
from multiprocessing import shared_memory as shm

import numpy as np
import pandas as pd
import pysam
import tqdm
from scipy import stats

from deeprm.inference.pileup_genomic import GenomeMap, parse_refflat
from deeprm.utils.logging import get_logger
from deeprm.utils.utils import (
    BUNDLED_WEIGHTS_MD5,
    RUN_INFO_NAME,
    default_threads,
    maybe_index_bam,
    read_run_info,
    sort_memory_per_thread,
)

log = get_logger(__name__)

# Workers are forked. A context, not mp.set_start_method(): importing this module does not change the start method of
# the program that imports it.
_MP = mp.get_context("fork")

# Memory bounds of the pileup. None of them changes an output: they set only how much is held in memory at a time.
# A worker adds its files' site tables into its running table once they hold this many sites (or as many sites as the
# running table, whichever is more), so it holds about its own sites, not its rows.
_MERGE_MIN_SITES = 1 << 22
# A worker writes its per-read rows for the modBAM to disk, sorted by reference, each time it holds this many.
_MODBAM_RUN_ROWS = 1 << 22
# A modBAM writer loads the rows of its references a slice at a time, a slice holding about this many rows (a slice is
# at least one reference; a reference with more rows is read in windows of positions). The writers run while the main
# process writes the site tables, so this also keeps the two together below the peak of the workers.
_MODBAM_SLICE_ROWS = 1 << 21
# The modBAM is written as about this many tasks per writer process, which the writers take from a queue: a writer that
# ends a task early takes the next one instead of waiting for the slowest.
_MODBAM_TASKS_PER_WRITER = 4
# BGZF compression level of the modBAM: 1, the fastest (about 4 times as fast as the default 6, and the file is about
# a fifth larger). The records do not depend on it; `samtools view -b` compresses the file again at another level.
_MODBAM_LEVEL = 1
# Reads whose MM/ML tags are made together (_mm_ml_batch).
_MODBAM_BATCH = 1024
# Most shards copied into the modBAM at a time (_join_by_copy).
_JOIN_COPIES = 16
# The site sums are added up by site adders, each over its own range of label_ids, from the rows that the readers of
# the per-read files send them. None of the numbers below changes an output.
# Per-read files a reader sends on at a time (a chunk never holds files of two worker groups, see _plan_chunks).
_CHUNK_FILES = 64
# Most readers; the other processes of --thread are site adders.
_MAX_READERS = 32
# Chunks a site adder receives ahead of the one it is adding up (about 100 MB of rows at most), so that the readers
# and the other site adders do not wait for one that has more to add.
_CHUNKS_AHEAD = 128
# Threads of a reader that read the files of its next chunk.
_READ_THREADS = 8
# Interpreter switch interval of a site adder, in seconds (the default is 0.005).
_SWITCH_INTERVAL = 0.0005
# Per-read files read before the pileup to give the site adders about the same number of rows each.
_SAMPLE_FILES = 256
# A site adder merges the sites it has seen since the last merge into its sorted index once they are this many.
_INDEX_NEW_SITES = 1 << 16
# Sites a part of a site table scores and writes at a time.
_PART_SITES = 1 << 18


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
        "--thread",
        "-t",
        type=int,
        default=None,
        help="Number of threads to use (default: 95%% of the CPUs this process can use)",
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
    parser.add_argument(
        "--tmp-dir",
        type=str,
        default=None,
        help="Folder for the pileup's largest temporary files, the per-read rows and the modBAM parts (default: the "
        "output folder, which still needs room for the outputs). A local disk is faster than a network file system.",
    )
    parser.add_argument(
        "--report-all-stoichiometry",
        action="store_true",
        help="Also write the stoichiometry of the sites whose reads are not enough to decide it, instead of 0 (NaN in "
        "stoichiometry_release); their stoich_state is still -1",
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
#  -1 (not reported): the stoichiometry is written as 0 (NaN in stoichiometry_release), unless
#      --report-all-stoichiometry is given: then it is written as the value above, and the state still says -1.
# Predictions of the bundled v1.2 weights get a fixed calibration in E_cal; predictions of other model files are used
# as they are (see inputs_calibrated). The rule reads no label and changes no detection score (modscore). It has no
# settings: --threshold changes modscore, count_pos, logsum_1_p_pos and stoichiometry_eq14 (the v1.1 value), not the
# rule.
#
# The two functions the rest of the pileup calls, with signatures a replacement rule must keep:
#   _per_read_rule_terms(pred, threshold, epsilon, calibrated) -> {name: per-read array} for every name in _RULE_SUMS
#   _site_stoichiometry(sums, stoich_eq12, threshold, mask=True) -> (stoichiometry, stoich_state,
#       stoichiometry_release, stoichiometry_eq14); `sums` maps every SITE_SUMS name to its per-site array; with
#       mask=False (--report-all-stoichiometry) the sites that are not reported get a value too.
# A new per-site sum for the rule (one int64 or float64 value per site) is one row in _RULE_SUMS plus its entry in
# _per_read_rule_terms; the workers, their saved site tables and the genomic pileup then carry it without other
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


def _site_stoichiometry(sums, stoich_eq12, threshold, mask=True):
    """Site stoichiometry as written to the BED and NPZ outputs (the reporting rule).

    Args:
        sums (Mapping[str, numpy.ndarray]): Per-site sums, one array per SITE_SUMS name.
        stoich_eq12 (numpy.ndarray): Eq. 12 stoichiometry per site.
        threshold (float): Per-read cut (--threshold). Only the Eq. 14 stoichiometry output uses it; the rule uses
            _RULE_CUT.
        mask (bool): Hide the value of the sites that are not reported (False: --report-all-stoichiometry).

    Returns:
        tuple: (reported stoichiometry: the value where the state is 1, else 0; state (int8); release stoichiometry
        with NaN where not reported; Eq. 14 stoichiometry at `threshold`, before the rule, as in v1.1). With
        mask=False, the value of every site in both, the state unchanged.
    """
    stoich_eq12 = np.asarray(stoich_eq12, dtype=np.float64)
    n = np.asarray(sums["count_all"], dtype=np.float64)
    eq14 = stoich_eq12 * _eq14_pass(stoich_eq12, sums["logsum_1_p_pos"], n, threshold)
    s = stoich_eq12 * _eq14_pass(stoich_eq12, sums["logsum_1_p_pos_rule"], n, _RULE_CUT)
    e_cal = np.asarray(sums["logsum_1_p_pos_calib"], dtype=np.float64) / np.log10(1 - _RULE_CUT)
    state, release = _apply_b3v(s, sums["count_all"], sums["count_pos_rule"], e_cal)
    if not mask:  # the value _apply_b3v reports, for every site
        k = np.asarray(sums["count_pos_rule"], dtype=np.int64)
        release = np.where(s > 0, s, k / np.maximum(n, 1.0))
        return release, state, release, eq14
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
    """(int64 names, float64 names) of a sum registry. The workers' site tables (_SiteSums, _save_table) and the merge
    of them hold every sum as one int64 or float64 value per site, so any other dtype is refused."""
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


def _site_scores(sums, threshold, epsilon, mask=True):
    """modscore and the stoichiometry outputs of one site table (transcript or genomic) from its per-site sums.

    Args:
        sums (Mapping[str, numpy.ndarray]): Per-site sums, one array per SITE_SUMS name.
        threshold (float): Per-read cut.
        epsilon (float): Small constant for division safety.
        mask (bool): Hide the stoichiometry of the sites that are not reported (see _site_stoichiometry).

    Returns:
        dict: modscore, stoichiometry, stoich_state, stoichiometry_release and stoichiometry_eq14, one value per site.
    """
    count_all = sums["count_all"]
    logsum_1_p_pos = sums["logsum_1_p_pos"]
    stoich_eq12 = _eq12_stoichiometry(sums["kl_div_pos"], sums["kl_div_neg"], epsilon)
    modscore = 1 - np.power(10, logsum_1_p_pos / count_all * (1 + np.power(10, 2 * (stoich_eq12 - 1))))
    modscore = np.digitize(modscore, np.linspace(0, 1, _DIGITIZATION + 1), right=True) / _DIGITIZATION
    stoichiometry, state, release, eq14 = _site_stoichiometry(sums, stoich_eq12, threshold, mask)
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


class _SiteSums:
    """A worker's per-site sums, added up file by file.

    Each file's table (its sorted unique label_ids and their sums) is kept only until the next merge into the running
    table, which runs once the kept tables hold _MERGE_MIN_SITES sites or as many sites as the running table. So the
    worker holds about its own sites, not one table per file. Every site's sum starts at 0 and adds the files in their
    order, as _merge_site_sums over all the files does, so the sums are the same to the bit.
    """

    def __init__(self, min_sites=None):
        self.min_sites = _MERGE_MIN_SITES if min_sites is None else min_sites
        self.label_id = np.empty(0, np.int64)
        self.sums = {key: np.empty(0, dtype) for key, dtype in SITE_SUMS}
        self._kept_ids, self._kept_sums, self._kept = [], [], 0

    def add(self, unique_id, sums):
        """Adds one file's table: its sorted unique label_ids, and their sums ({SITE_SUMS name: array})."""
        self._kept_ids.append(unique_id)
        self._kept_sums.append(sums)
        self._kept += len(unique_id)
        if self._kept >= max(self.min_sites, len(self.label_id)):
            self._merge()

    def _merge(self):
        if not self._kept_ids:
            return
        label_id = _sorted_unique(np.concatenate([self.label_id, *self._kept_ids]))
        merged = {key: np.zeros(len(label_id), dtype=dtype) for key, dtype in SITE_SUMS}
        idx = np.searchsorted(label_id, self.label_id)
        for key, _ in SITE_SUMS:
            merged[key][idx] = self.sums[key]  # the sums so far, copied as they are
        self.label_id, self.sums = label_id, None
        for ids, sums in zip(self._kept_ids, self._kept_sums):
            idx = np.searchsorted(label_id, ids)
            for key, _ in SITE_SUMS:
                merged[key][idx] += sums[key]
        self.sums = merged
        self._kept_ids, self._kept_sums, self._kept = [], [], 0

    def result(self):
        """(sorted unique label_ids, {SITE_SUMS name: array}) of all the files added."""
        self._merge()
        return self.label_id, self.sums


def _save_table(stem, label_id, sums):
    """Writes a worker's site table as <stem>.label_id.npy and one <stem>.<sum name>.npy per sum."""
    np.save(f"{stem}.label_id.npy", np.ascontiguousarray(label_id, dtype=np.int64))
    for key, dtype in SITE_SUMS:
        np.save(f"{stem}.{key}.npy", np.ascontiguousarray(sums[key], dtype=dtype))


def _union_sorted(a, b):
    """The sorted unique values of two sorted int64 arrays."""
    c = np.concatenate((a, b))
    c.sort(kind="stable")  # two sorted runs, which the stable sort (timsort for int64) merges in linear time
    if c.size == 0:
        return c
    return c[np.concatenate(([True], c[1:] != c[:-1]))]


def _merge_worker_tables(stems):
    """Adds up the workers' site tables (written by _save_table) in the order of `stems`, one table at a time: the same
    sums as _merge_site_sums over all of them, while holding only the result and one worker's table.

    Returns:
        tuple: (sorted unique label_ids of all tables, {SITE_SUMS name: summed array}).
    """
    label_id = np.empty(0, np.int64)
    for stem in stems:
        label_id = _union_sorted(label_id, np.load(f"{stem}.label_id.npy"))
    merged = {key: np.zeros(len(label_id), dtype=dtype) for key, dtype in SITE_SUMS}
    for stem in stems:
        idx = np.searchsorted(label_id, np.load(f"{stem}.label_id.npy"))
        for key, _ in SITE_SUMS:
            merged[key][idx] += np.load(f"{stem}.{key}.npy")
        del idx
    return label_id, merged


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
    tmp_dir = getattr(args, "tmp_dir", None)
    if tmp_dir:
        if os.path.exists(tmp_dir) and not os.path.isdir(tmp_dir):
            raise ValueError(f"--tmp-dir: not a folder: {tmp_dir}")
        parent = os.path.abspath(tmp_dir)
        while not os.path.exists(parent):
            parent = os.path.dirname(parent)
        if not os.path.isdir(parent) or not os.access(parent, os.W_OK | os.X_OK):
            raise ValueError(f"--tmp-dir: cannot write in {parent}")


def shm_put_array(name: str, arr: np.ndarray):
    """Create a SharedMemory block named `name` and copy `arr` into it. (Kept from 1.2; the pileup no longer uses
    it.)"""
    block = shm.SharedMemory(name=name, create=True, size=max(1, arr.nbytes))  # size 0 is not allowed
    view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=block.buf)
    view[...] = arr
    return block, {"name": name, "shape": arr.shape, "dtype": str(arr.dtype)}


def shm_view(meta):
    """Open a SharedMemory block and create a numpy view from its buffer. (Kept from 1.2; the pileup no longer uses
    it.)"""
    block = shm.SharedMemory(name=meta["name"], create=False)
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


def _table_arrays(ref_names, ref_pos, ref_strand, scores, sums):
    """The arrays of a site table's NPZ, in the order they are written."""
    return {
        "ref_names": ref_names,
        "ref_pos": ref_pos,
        "ref_strand": ref_strand,
        "modscore": scores["modscore"],
        "stoichiometry": scores["stoichiometry"],
        "count_all": sums["count_all"],
        "count_pos": sums["count_pos"],
        "stoich_state": scores["stoich_state"],
        "stoichiometry_release": scores["stoichiometry_release"],
        "stoichiometry_eq14": scores["stoichiometry_eq14"],
        **{key: sums[key] for key in _WRITTEN_SUMS},
    }


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
    _savez(stem + ".npz", **_table_arrays(ref_names, ref_pos, ref_strand, scores, sums))


class _TablePart:
    """One part of a site table, written by the process that made its sites: the BED lines (<stem>.bed) and each array
    of the NPZ as a raw deflate stream (<stem>.<array>.z). _join_table puts the parts of a table together, so the
    table is never whole in one process."""

    def __init__(self, stem):
        self.stem = stem
        self.rows = 0
        self._bed = open(stem + ".bed", "wb")
        self._arrays = {}  # name -> [dtype, CRC-32, bytes, compressor, file]

    def add(self, ref_names, ref_pos, ref_strand, scores, sums):
        """Adds sites to the part; the arguments of _write_site_table."""
        ref_names = np.asarray(ref_names, dtype=str)
        ref_strand = np.asarray(ref_strand, dtype=str)
        lines = self.stem + ".lines.bed"
        bed_formatter(
            ref_names=ref_names,
            ref_pos=ref_pos,
            ref_strand=ref_strand,
            modscore=scores["modscore"],
            stoichiometry=scores["stoichiometry"],
            count_all=sums["count_all"],
            count_pos=sums["count_pos"],
            output_path=lines,
        )
        with open(lines, "rb") as fh:
            shutil.copyfileobj(fh, self._bed, 1 << 24)
        os.remove(lines)
        for name, arr in _table_arrays(ref_names, ref_pos, ref_strand, scores, sums).items():
            arr = np.ascontiguousarray(arr)
            if name not in self._arrays:
                # the stream zipfile writes at level 1 (see _savez)
                self._arrays[name] = [arr.dtype, 0, 0, zlib.compressobj(1, zlib.DEFLATED, -15), None]
                self._arrays[name][4] = open(f"{self.stem}.{name}.z", "wb")
            entry = self._arrays[name]
            if arr.dtype != entry[0]:
                raise TypeError(f"Site table array {name}: {arr.dtype} after {entry[0]}.")
            data = arr.view(np.uint8)
            entry[1] = zlib.crc32(data, entry[1])
            entry[2] += data.nbytes
            entry[4].write(entry[3].compress(data))
        self.rows += len(ref_pos)

    def close(self):
        """Ends the part's files. Returns what _join_table takes: {"stem", "rows", "arrays": {name: (dtype, CRC-32,
        bytes)}}."""
        self._bed.close()
        arrays = {}
        for name, (dtype, crc, size, compressor, fh) in self._arrays.items():
            fh.write(compressor.flush(zlib.Z_SYNC_FLUSH))  # ends at a byte, so that the next part's stream can follow
            fh.close()
            arrays[name] = (dtype.str, crc, size)
        return {"stem": self.stem, "rows": self.rows, "arrays": arrays}


def _crc32_combine(crc1, crc2, len2):
    """The CRC-32 of two runs of bytes, one after the other, from the CRC-32 of each and the length of the second
    (zlib's crc32_combine, which Python's zlib does not have)."""

    def times(mat, vec):
        total, i = 0, 0
        while vec:
            if vec & 1:
                total ^= mat[i]
            vec >>= 1
            i += 1
        return total

    def square(mat):
        return [times(mat, row) for row in mat]

    if len2 <= 0:
        return crc1
    odd = [0xEDB88320] + [1 << n for n in range(31)]  # the operator of one zero bit
    even = square(odd)
    odd = square(even)
    while True:  # one zero byte, two, four, ...: applied for each bit of len2
        even = square(odd)
        if len2 & 1:
            crc1 = times(even, crc1)
        len2 >>= 1
        if not len2:
            break
        odd = square(even)
        if len2 & 1:
            crc1 = times(odd, crc1)
        len2 >>= 1
        if not len2:
            break
    return crc1 ^ crc2


def _append_file(src_path, out):
    """Appends a file to the open binary file `out`: by copy_file_range (see _copy_bytes), or else read and written."""
    out.flush()
    at = out.seek(0, os.SEEK_END)
    with open(src_path, "rb") as fh:
        try:
            _copy_bytes(fh.fileno(), out.fileno(), os.fstat(fh.fileno()).st_size, 0, at)
        except OSError:
            out.truncate(at)  # what a copy that failed part-way wrote
            out.seek(at)
            shutil.copyfileobj(fh, out, 1 << 24)
    out.seek(0, os.SEEK_END)


def _join_table(stem, parts):
    """Writes the site table <stem>.bed and <stem>.npz from its parts (the output of _TablePart.close, in the order of
    their sites) and removes the parts' files. The files hold what _write_site_table writes for the whole table: the
    same BED lines, and an NPZ that loads as the same arrays (each array's parts are joined into one deflate stream,
    without being compressed again)."""
    import zipfile

    parts = [part for part in parts if part["rows"]]
    with open(stem + ".bed", "wb") as out:
        for part in parts:
            _append_file(part["stem"] + ".bed", out)
    rows = sum(part["rows"] for part in parts)
    with zipfile.ZipFile(stem + ".npz", mode="w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for name, (dtype, _, _) in parts[0]["arrays"].items():
            if any(part["arrays"][name][0] != dtype for part in parts):
                raise TypeError(f"Site table array {name}: its parts have different dtypes.")
            header = io.BytesIO()
            np.lib.format.write_array_header_1_0(
                header,
                {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)), "fortran_order": False, "shape": (rows,)},
            )
            header = header.getvalue()
            compressor = zlib.compressobj(1, zlib.DEFLATED, -15)
            head = compressor.compress(header) + compressor.flush(zlib.Z_SYNC_FLUSH)
            crc, size = zlib.crc32(header), len(header)
            for part in parts:
                crc = _crc32_combine(crc, part["arrays"][name][1], part["arrays"][name][2])
                size += part["arrays"][name][2]
            paths = [f"{part['stem']}.{name}.z" for part in parts]
            tail = b"\x03\x00"  # an empty last block: ends the deflate stream
            # the member is written as zipfile writes it, with streams that are compressed already
            info = zipfile.ZipInfo(name + ".npy", date_time=time.localtime(time.time())[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            info.CRC, info.file_size = crc, size
            info.compress_size = len(head) + sum(os.path.getsize(path) for path in paths) + len(tail)
            info.header_offset = zf.fp.tell()
            zf.fp.write(info.FileHeader(zip64=True))
            zf.fp.write(head)
            for path in paths:
                _append_file(path, zf.fp)
            zf.fp.write(tail)
            zf.filelist.append(info)
            zf.NameToInfo[info.filename] = info
            zf.start_dir = zf.fp.tell()
            zf._didModify = True
    for part in parts:
        for path in [part["stem"] + ".bed"] + [f"{part['stem']}.{name}.z" for name in part["arrays"]]:
            os.remove(path)


def main(args: argparse.Namespace):
    """
    Runs the pileup. Reader processes read the per-read files (and save their per-read rows for the modBAM); site
    adders sum the predictions per site, each over its own range of sites, score them and write their parts of
    pileup.bed and pileup.npz, and with ``--annot`` of the genomic tables; the parent joins the parts. Unless
    ``--skip-modbam`` is given, modBAM writers join the rows of each read from all readers and write
    modbam_<input BAM name> with MM/ML tags.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None: Results are saved in the output directory.
    """
    start = time.time()
    _validate_args(args)
    if args.thread is None:
        args.thread = default_threads()

    if not args.skip_modbam:  # the modBAM writers read the BAM through its index; without them only the header is read
        args.bam = maybe_index_bam(args.bam, args.thread)

    os.makedirs(args.output, exist_ok=True)
    calibrated = inputs_calibrated(args.input)
    _pileup(args, calibrated)

    elapsed = time.time() - start
    log.info(f"Pileup finished in {elapsed:.2f} seconds.")
    return None


def _pileup(args, calibrated):
    """The body of main()."""
    file_paths = sorted(glob.glob(os.path.join(args.input, "*.npz")))
    if len(file_paths) == 0:
        raise ValueError(f"No inference .npz files found in: {args.input}")
    # a caller's own Namespace may leave --tmp-dir out
    tmp_root = getattr(args, "tmp_dir", None) or args.output
    os.makedirs(tmp_root, exist_ok=True)
    # The site adders write their parts of the site tables into a folder of their own; it is removed when the pileup
    # ends.
    sums_dir = tempfile.mkdtemp(prefix=".deeprm_sums_", dir=tmp_root)
    try:
        _pileup_with_tables(args, calibrated, file_paths, sums_dir, tmp_root)
    finally:
        with _sigint_held():
            _remove_tree(sums_dir)


def _plan_chunks(n_files, n_groups):
    """(first file, end, worker group) of each chunk of at most _CHUNK_FILES per-read files, in file order.

    The worker groups are the files of the `n_groups` workers of DeepRM 1.2 (np.array_split). A float sum depends on
    the order of its terms, and 1.2 adds a site's rows up per worker and then the workers: the site adders add them
    in the same order (_SiteTable), so the sums are those of 1.2 for the same --thread."""
    chunks = []
    for group, files in enumerate(np.array_split(np.arange(n_files), n_groups)):
        first, end = int(files[0]), int(files[-1]) + 1
        chunks += [(a, min(a + _CHUNK_FILES, end), group) for a in range(first, end, _CHUNK_FILES)]
    return chunks


def _site_bounds(file_paths, n):
    """At most n - 1 label_ids that split the sites into ranges with about the same number of rows each, from a
    sample of the per-read files. Site adder i takes the label_ids from bound i - 1 up to, not including, bound i.
    The bounds change only how the work is shared, not the sums."""
    step = max(1, len(file_paths) // _SAMPLE_FILES)
    ids = []
    for path in file_paths[::step][:_SAMPLE_FILES]:
        try:
            with np.load(path) as data:
                ids.append(np.asarray(data["label_id"], dtype=np.int64).ravel())
        except Exception:  # noqa: BLE001 -- a file that cannot be read is reported by its reader
            continue
    ids = np.concatenate(ids) if ids else np.empty(0, np.int64)
    if n <= 1 or ids.size == 0:
        return np.empty(0, np.int64)
    return _sorted_unique(np.quantile(ids, np.arange(1, n) / n, method="lower").astype(np.int64))


def _next_payload(meta_q, proc_list, metas, poll_s=5.0, name="Pileup worker"):
    """The next payload of the processes in `proc_list` (its `pid` is its index in the list; `metas`: the payloads
    received so far, by pid). Raises for a process that failed or exited without sending one, since the others would
    wait for it forever."""
    while True:
        try:
            md = meta_q.get(timeout=poll_s)
        except queue.Empty:
            lost = [(i, proc) for i, proc in enumerate(proc_list) if proc.exitcode is not None and i not in metas]
            if lost:
                raise RuntimeError(
                    f"{name} {lost[0][0]} exited with code {lost[0][1].exitcode} without sending its results."
                ) from None
            continue
        if md.get("error"):
            raise RuntimeError(f"{name} {md['pid']} failed:\n{md['error']}")
        return md


def _pileup_with_tables(args, calibrated, file_paths, sums_dir, tmp_root):
    """The body of _pileup(), inside the lifetime of the folder of the site tables' parts.

    Readers read the per-read files and send each row to the site adder of its label_id (and save the per-read rows
    for the modBAM); each site adder adds up its range of sites and writes its part of the site tables. So a site is
    held once, by one process, and no table is merged or held whole."""
    proc_list, writers, tables_dir, tmp_dir = [], None, None, None
    try:  # any failure (Ctrl-C too): no worker or writer left, no temporary file left
        # the modBAM step's temporary files (the readers' per-read rows, the writers' shards) go to a folder of their
        # own, so that none of them can have the name of an output (such as modbam_<input BAM name>)
        tmp_dir = None if args.skip_modbam else tempfile.mkdtemp(prefix=".deeprm_modbam_", dir=tmp_root)

        ## Reference names from the BAM header, and with --annot where each reference lies on the genome
        with pysam.AlignmentFile(args.bam, "rb", check_sq=False) as input_bam:
            ref_arr = np.array(input_bam.references)
        genome = GenomeMap(ref_arr, parse_refflat(args.annot)) if args.annot else None

        chunks = _plan_chunks(len(file_paths), min(args.thread, len(file_paths)))
        # about a quarter of the processes read: adding the rows up takes about three times as long as reading them
        n_readers = max(1, min(args.thread // 4 + 1, _MAX_READERS, len(chunks)))
        n_adders = max(1, args.thread - n_readers)
        _raise_open_files(2 * n_readers * n_adders + 1024)  # every process inherits both ends of every pipe
        # with an open-file limit that cannot be raised: fewer processes, so that their pipes fit
        soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        if soft != resource.RLIM_INFINITY:
            # besides the pipes: the descriptor the parent keeps for each process it starts, and the queue's
            n_pipes = max(1, (soft - _open_files() - 64 - 2 * args.thread - 16) // 2)
            n_readers = min(n_readers, max(1, math.isqrt(n_pipes // 2)))
            n_adders = min(n_adders, max(1, n_pipes // n_readers))
        bounds = _site_bounds(file_paths, n_adders)
        n_adders = len(bounds) + 1
        pipes = [[_MP.Pipe(duplex=False) for _ in range(n_adders)] for _ in range(n_readers)]  # [reader][adder]

        meta_q = _MP.Queue()
        for pid in range(n_adders):
            adder_args = (
                pipes,
                chunks,
                os.path.join(sums_dir, f"part_{pid}"),
                ref_arr,
                genome,
                args.label_div,
                args.threshold,
                args.epsilon,
                calibrated,
                not getattr(args, "report_all_stoichiometry", False),  # a caller's own Namespace may leave it out
            )
            proc = _MP.Process(target=worker, args=(_site_adder, pid, meta_q) + adder_args)
            proc.start()
            proc_list.append(proc)
        for pid in range(n_adders, n_adders + n_readers):
            reader_args = (
                n_adders,
                file_paths,
                chunks,
                bounds,
                pipes,
                tmp_dir,
                args.label_div,
                args.slice,
                args.flip,
                not args.skip_modbam,
            )
            proc = _MP.Process(target=worker, args=(_reader, pid, meta_q) + reader_args)
            proc.start()
            proc_list.append(proc)
        for conn in itertools.chain.from_iterable(itertools.chain.from_iterable(pipes)):
            conn.close()  # the readers and the site adders hold them now

        metas = {}
        readers = range(n_adders, n_adders + n_readers)
        while not all(pid in metas for pid in readers):
            md = _next_payload(meta_q, proc_list, metas)
            metas[md["pid"]] = md
        # the saved rows in reader order, and each reader's runs in the order it wrote them, so the writers join the
        # rows of a read in the same order on every run
        rows_stems = [stem for pid in readers for stem in metas[pid]["rows"]]

        # The modBAM writers start now: they need only the saved rows and the references with rows (each reader
        # listed its own). The site adders write their parts of the site tables meanwhile; the tables are put
        # together in a folder of their own and moved beside the modBAM once it is written, so that a run that fails
        # leaves neither (as when the modBAM came first).
        tables_dir = tempfile.mkdtemp(prefix=".deeprm_tables_", dir=args.output)
        if not args.skip_modbam and rows_stems:
            # the writers inherit it: up to 8 files of each reader, and the pipes of the writers started before them
            _raise_open_files(10 * n_readers + 1024)
            ref_ids = _sorted_unique(np.concatenate([metas[pid]["refs"] for pid in readers]))
            writers = _start_modbam_writers(args.bam, rows_stems, ref_ids, args.thread, tmp_dir)
        while len(metas) < len(proc_list):
            md = _next_payload(meta_q, proc_list, metas)
            metas[md["pid"]] = md
        for proc in proc_list:
            proc.join()  # each process exits once it has sent its payload
        codes = [proc.exitcode for proc in proc_list]
        _stop_processes(proc_list)
        for code in codes:
            if code != 0:
                raise RuntimeError(f"Pileup worker exited with code {code}.")

        _site_outputs(args, [metas[pid] for pid in range(n_adders)], genome, sums_dir, tables_dir)
        if not args.skip_modbam:
            modbam_out_path = os.path.join(args.output, "modbam_" + os.path.basename(args.bam))
            written = writers is not None and _finish_modbam(writers, args.bam, modbam_out_path, args.thread)
            if not written:
                log.warning("No valid modBAM entries were produced; skipping modBAM writing.")
        for name in sorted(os.listdir(tables_dir)):
            os.replace(os.path.join(tables_dir, name), os.path.join(args.output, name))
    finally:
        with _sigint_held():
            _stop_processes(proc_list)
            if writers is not None:
                _stop_modbam_writers(writers)
            _remove_modbam_temp(tmp_dir)
            if tables_dir is not None:
                _remove_tree(tables_dir)


def _site_outputs(args, parts, genome, sums_dir, out_dir):
    """Writes the site-level files into `out_dir` from the site adders' payloads `parts` (in the order of their
    sites): pileup.bed and pileup.npz, and with --annot the genomic tables."""
    # a site adder stops at a reference the BAM does not have, so its table can be empty: that error first
    if any(md["bad_refs"] for md in parts):
        raise ValueError("label_id contains transcript IDs outside the BAM reference range.")
    if not sum(md["table"]["rows"] for md in parts):
        raise ValueError("No valid pileup entries were produced from the inference outputs.")
    _join_table(os.path.join(out_dir, "pileup"), [md["table"] for md in parts])

    if genome is not None:
        ## Genomic pileup: the same sums added up per genomic position, scored and written the same way, a
        ## chromosome at a time
        genome.log_matches(pd.unique(np.concatenate([md["refs"] for md in parts])))
        counts = np.stack([md["genomic"] for md in parts])  # [part][chromosome]: mapped sites
        chroms = np.flatnonzero(counts.sum(axis=0))
        if chroms.size == 0:
            raise ValueError("No valid genomic position found. Please verify the annotation file.")
        names = np.asarray(genome.chroms[chroms], dtype=str)  # one dtype for the names of all the chromosomes
        starts = np.cumsum(counts, axis=1) - counts
        tasks = []
        mask = not getattr(args, "report_all_stoichiometry", False)  # a caller's own Namespace may leave it out
        for i, chrom in enumerate(chroms.tolist()):
            rows = [
                (f"{md['table']['stem']}.genomic.npy", int(starts[p, chrom]), int(starts[p, chrom] + counts[p, chrom]))
                for p, md in enumerate(parts)
                if counts[p, chrom]
            ]
            stem = os.path.join(sums_dir, f"genomic_{i}")
            tasks.append((rows, names[i : i + 1], stem, args.threshold, args.epsilon, mask))
        tables = _run_tasks(_genomic_part, tasks, args.thread, name="Genomic pileup worker")
        _join_table(os.path.join(out_dir, f"genomic_pileup{args.postfix}"), tables)


def _run_tasks(fn, tasks, threads, name):
    """fn(*task) for every task, in up to `threads` processes that take the tasks from a queue, largest first (a
    task's first argument lists its rows as (file, first row, end)). Returns the results in task order."""
    task_q, result_q = _MP.Queue(), _MP.Queue()

    def run():
        try:
            while True:
                i = task_q.get()
                if i is None:
                    break
                result_q.put({"task": i, "result": fn(*tasks[i])})
        except BaseException:  # noqa: BLE001 -- reported to the parent, then re-raised
            result_q.put({"task": None, "error": traceback.format_exc()})
            raise SystemExit(1)

    procs = []
    results = {}
    try:  # the processes are forked before the first put starts the queue's feeder thread
        for _ in range(min(threads, len(tasks))):
            proc = _MP.Process(target=run)
            proc.start()
            procs.append(proc)
        for i in sorted(range(len(tasks)), key=lambda i: -sum(b - a for _, a, b in tasks[i][0])):
            task_q.put(i)
        for _ in procs:
            task_q.put(None)
        while len(results) < len(tasks):
            try:
                result = result_q.get(timeout=5.0)
            except queue.Empty:
                if any(proc.exitcode for proc in procs):
                    raise RuntimeError(f"{name} ended before its tasks were done.") from None
                continue
            if result.get("error"):
                raise RuntimeError(f"{name} failed:\n{result['error']}")
            results[result["task"]] = result["result"]
    finally:
        _stop_processes(procs)
    return [results[i] for i in range(len(tasks))]


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
                i, code = lost[0][0], lost[0][1].exitcode  # read before the processes are closed
                _stop_processes(proc_list)
                errors = [md for md in metas if md.get("error")]
                if errors:
                    raise RuntimeError(f"{name} {errors[0]['pid']} failed:\n{errors[0]['error']}")
                raise RuntimeError(f"{name} {i} exited with code {code} without sending its results.")
    return metas


def _stop_processes(procs, wait_s=30.0):
    """Stops the processes in `procs` that are still running (and kills those not ended `wait_s` seconds later), waits
    for them and closes them, since each holds a pipe; empties `procs`, so that a second call does nothing."""
    for proc in procs:
        if proc.is_alive():
            proc.terminate()
    _join_all(procs, wait_s)
    for proc in procs:
        if proc.exitcode is None:
            proc.kill()
    _join_all(procs, wait_s)
    for proc in list(procs):
        procs.remove(proc)  # before close(): a list never holds a closed process, even if this loop is interrupted
        if proc.exitcode is None:  # in uninterruptible I/O even after SIGKILL: left to end on its own
            log.warning(f"Process {proc.pid} has not ended; it is left running.")
        else:
            proc.close()


def _join_all(procs, wait_s):
    """Waits up to `wait_s` seconds in all for the processes to end."""
    deadline = time.monotonic() + wait_s
    for proc in procs:
        proc.join(max(0.0, deadline - time.monotonic()))


@contextlib.contextmanager
def _sigint_held():
    """Holds Ctrl-C (SIGINT) back until the block ends and then delivers it, so that it does not cut short the cleanup
    in the block (stopping the processes, removing the temporary files); a second Ctrl-C stops at once. Only the main
    thread receives signals, and a process started with Ctrl-C ignored keeps ignoring it: there the block runs as it
    is."""
    if threading.current_thread() is not threading.main_thread() or signal.getsignal(signal.SIGINT) in (
        None,
        signal.SIG_IGN,
    ):
        yield
        return
    held = []

    def hold(signum, frame):
        if held:
            raise KeyboardInterrupt
        held.append(signum)

    previous = signal.signal(signal.SIGINT, hold)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
        if held:
            signal.raise_signal(signal.SIGINT)


def _remove_tree(path):
    """Removes a temporary folder; one that cannot be removed (in full) is named in a warning."""
    shutil.rmtree(path, ignore_errors=True)
    if os.path.lexists(path):
        log.warning(f"Could not remove the temporary folder {path}; it can be deleted.")


def _remove_modbam_temp(tmp_dir):
    """Removes the folder of the modBAM step's temporary files (after the step, or after a failure); None: no folder."""
    if tmp_dir is not None:
        for path in [p for p in _SAVED_MAPS if p.startswith(os.path.join(tmp_dir, ""))]:
            del _SAVED_MAPS[path]  # unmapped once no array uses it, so the files' space is freed
        _remove_tree(tmp_dir)


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


def worker(target, pid, meta_q, *args):
    """Runs one pileup process (a reader or a site adder): ``target(pid, meta_q, *args)``. Any error is sent to the
    parent instead of leaving it waiting."""
    try:
        target(pid, meta_q, *args)
    except BaseException:  # noqa: BLE001 -- reported to the parent, then re-raised
        meta_q.put({"pid": pid, "error": traceback.format_exc()})
        raise SystemExit(1)


def _read_bytes(path):
    """The bytes of one per-read file."""
    with open(path, "rb") as fh:  # one read of the whole file: the zip reader's small reads seek on a disk
        return fh.read()


def _read_file(path, content, slice_idx, flip, make_modbam):
    """The valid rows of one per-read file, from its bytes `content`: (label_id, pred, read_id). `read_id` is None
    unless `make_modbam` (only the modBAM needs it)."""
    with np.load(io.BytesIO(content)) as data:
        pred = np.asarray(data["pred"])
        label_id = np.asarray(data["label_id"])
        read_id = np.asarray(data["read_id"]) if make_modbam else None

    assert pred.dtype == np.float32, f"Expected pred to be float32, but got {pred.dtype} in {path}"
    # int64 only: the sign of label_id is the strand
    assert label_id.dtype == np.int64, f"Expected label_id to be int64, but got {label_id.dtype} in {path}"
    assert pred.shape[0] == label_id.shape[0], f"Length of pred and label_id do not match in {path}"
    if read_id is not None:
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
    if read_id is not None:
        read_id = read_id[valid_idx]

    if flip:
        pred = 1 - pred

    if pred.size:
        pred_min = float(np.min(pred))
        pred_max = float(np.max(pred))
        assert pred_min >= 0.0, f"Minimum value of pred is {pred_min} in {path}"
        assert pred_max <= 1.0, f"Maximum value of pred is {pred_max} in {path}"
    return label_id, pred, read_id


def _reader(
    pid,
    meta_q,
    n_adders,
    file_paths,
    chunks,
    bounds,
    pipes,
    rows_dir,
    label_div,
    slice_idx,
    flip,
    make_modbam,
):
    """
    Reads every len(pipes)-th chunk of per-read files, sends each row to the site adder of its label_id, and saves
    the per-read rows for the modBAM when asked (the modBAM writers read them; see _start_modbam_writers). The first
    reader shows the progress bar.

    A site adder takes the chunks in order, each from the reader that has it, so the rows of a site reach it in file
    order. The memory does not follow the rows: a full pipe makes the reader wait for its site adders, and the
    per-read rows for the modBAM are written to disk, sorted by reference, every _MODBAM_RUN_ROWS rows.

    Args:
        pid (int): Index in the parent's process list; the readers follow the `n_adders` site adders.
        meta_q (multiprocessing.Queue): Sends {"pid", "rows": names of the saved runs of per-read rows, in the order
            written, "refs": the references with rows} (or the error) to the parent.
        n_adders (int): Number of site adders.
        file_paths (list): The .npz per-read files, sorted.
        chunks (list): The output of _plan_chunks.
        bounds (numpy.ndarray): The output of _site_bounds.
        pipes (list): pipes[reader][site adder]: (receiving end, sending end).
        rows_dir (str): Folder for the saved per-read rows (None when make_modbam is False).
        label_div (int): Divisor that separates the reference index and the position in label_id.
        slice_idx (int): Column of 2-D predictions, or None for 1-D predictions.
        flip (bool): Whether to invert probabilities (1 - p).
        make_modbam (bool): Whether to save the per-read rows for the modBAM.
    """
    reader = pid - n_adders
    conns = [send for _, send in pipes[reader]]
    for conn in itertools.chain.from_iterable(itertools.chain.from_iterable(pipes)):
        if not any(conn is mine for mine in conns):
            conn.close()
    own = chunks[reader :: len(pipes)]
    modbam_rows = _ReadRows() if make_modbam else None
    rows_stems, refs = [], np.empty(0, np.int64)

    def save_rows():
        stem = modbam_rows.save_by_reference(os.path.join(rows_dir, f"modbam_rows_{reader}"))
        if stem is not None:  # filled only from valid rows
            rows_stems.append(stem)

    # The files are read by threads, a chunk ahead of the rows: on a network file system most of a read is waiting.
    pool = concurrent.futures.ThreadPoolExecutor(_READ_THREADS)

    def read_ahead(k):
        if k >= len(own):
            return None
        return [pool.submit(_read_bytes, path) for path in file_paths[own[k][0] : own[k][1]]]

    bar = tqdm.tqdm(
        total=sum(end - first for first, end, _ in own), desc="Reading input files", leave=False, disable=reader != 0
    )
    contents = read_ahead(0)
    for k, (first, end, _) in enumerate(own):
        contents, ahead = read_ahead(k + 1), contents
        labels, preds, counts = [], [], []
        for path, content in zip(file_paths[first:end], ahead):
            label_id, pred, read_id = _read_file(path, content.result(), slice_idx, flip, make_modbam)
            labels.append(label_id)
            preds.append(pred)
            counts.append(len(label_id))
            if make_modbam and pred.size:
                ref_ids, poss = _decode_label_id(label_id, label_div)
                modbam_rows.add(ref_ids, read_id, poss, np.clip(np.rint(pred * 255), 0, 255).astype(np.uint8))
                if modbam_rows.rows_held() >= _MODBAM_RUN_ROWS:
                    save_rows()
        label_id, pred = np.concatenate(labels), np.concatenate(preds)
        file_no = np.repeat(np.arange(end - first, dtype=np.uint16), counts)
        if make_modbam:  # the references with rows: the modBAM writers are planned from them
            refs = _union_sorted(refs, _sorted_unique(_decode_label_id(label_id, label_div)[0]))
        # the rows of each site adder, in file order (the sort by adder is stable)
        adder = np.searchsorted(bounds, label_id, side="right").astype(np.uint16)  # 16 bits: a radix sort
        order = np.argsort(adder, kind="stable")
        edges = np.searchsorted(adder[order], np.arange(len(conns) + 1))
        label_id, pred, file_no = label_id[order], pred[order], file_no[order]
        for i, conn in enumerate(conns):
            a, b = edges[i], edges[i + 1]
            conn.send_bytes(label_id[a:b])
            conn.send_bytes(pred[a:b])
            conn.send_bytes(file_no[a:b])
        bar.update(end - first)
    bar.close()
    pool.shutdown()

    if make_modbam:
        save_rows()  # the rows not saved yet; afterwards the table holds none
    # one message, after the last step that can fail
    meta_q.put({"pid": pid, "rows": rows_stems, "refs": refs})
    return None


class _SiteTable:
    """The per-site sums of one site adder, added up chunk by chunk.

    The sums are those of DeepRM 1.2 to the bit. 1.2 sums the rows of a site within a file (in the terms' own dtype),
    adds the files of a worker in file order into a sum that starts at 0, and then adds the workers in order into
    another; add() and end_group() add in that order: `_group` holds the float sums of the current worker group, and
    `_total` those of the groups before (and the integer sums, whose order does not matter).

    A site keeps the slot it got when it was first seen (the arrays grow at their end), and is found through a sorted
    index: `_ids`/`_slots`, and the sites seen since the last merge in `_new_ids`/`_new_slots`.
    """

    def __init__(self, threshold, epsilon, calibrated):
        self._terms = (threshold, epsilon, calibrated)
        self._n = 0
        self._label_id = np.empty(1 << 16, np.int64)  # by slot
        self._total = {key: np.zeros(1 << 16, dtype) for key, dtype in SITE_SUMS}
        self._group = {key: np.zeros(1 << 16, np.float64) for key in FLOAT_SUMS}
        self._ids, self._slots = np.empty(0, np.int64), np.empty(0, np.int64)
        self._new_ids, self._new_slots = np.empty(0, np.int64), np.empty(0, np.int64)

    def _find(self, label_id):
        """The slot of each of the sorted unique `label_id`; a site not seen before gets the next slot."""
        slots = np.full(len(label_id), -1, dtype=np.int64)
        for ids, known in ((self._ids, self._slots), (self._new_ids, self._new_slots)):
            if len(ids):
                at = np.minimum(np.searchsorted(ids, label_id), len(ids) - 1)
                found = ids[at] == label_id
                slots[found] = known[at[found]]
        new = np.flatnonzero(slots < 0)
        if new.size:
            end = self._n + new.size
            if end > len(self._label_id):
                size = max(end, len(self._label_id) * 5 // 4)
                self._label_id = np.resize(self._label_id, size)
                for sums in (self._total, self._group):
                    for key, arr in sums.items():
                        sums[key] = np.concatenate([arr, np.zeros(size - len(arr), arr.dtype)])
            slots[new] = np.arange(self._n, end)
            self._label_id[self._n : end] = label_id[new]
            self._n = end
            merged = self._merge_index(self._new_ids, self._new_slots, label_id[new], slots[new])
            self._new_ids, self._new_slots = merged
            if len(self._new_ids) >= _INDEX_NEW_SITES:
                self._ids, self._slots = self._merge_index(self._ids, self._slots, self._new_ids, self._new_slots)
                self._new_ids, self._new_slots = np.empty(0, np.int64), np.empty(0, np.int64)
        return slots

    @staticmethod
    def _merge_index(ids, slots, more_ids, more_slots):
        ids, slots = np.concatenate((ids, more_ids)), np.concatenate((slots, more_slots))
        order = np.argsort(ids, kind="stable")  # two sorted runs, which the stable sort merges in linear time
        return ids[order], slots[order]

    def add(self, label_id, pred, file_no):
        """Adds the rows of one chunk of files (`file_no`: the file of each row within the chunk), in file order."""
        if len(label_id) == 0:
            return
        terms = _per_read_terms(pred, *self._terms)
        order = np.argsort(label_id, kind="stable")  # by site; the rows of a site stay in file order
        label_id, file_no = label_id[order], file_no[order]
        new_site = np.ones(len(label_id), dtype=bool)
        new_site[1:] = label_id[1:] != label_id[:-1]
        new_file = new_site.copy()  # the first row of each (site, file)
        new_file[1:] |= file_no[1:] != file_no[:-1]
        in_file = np.cumsum(new_file) - 1
        first = np.flatnonzero(new_file)
        slots = self._find(label_id[new_site])[np.cumsum(new_site[first]) - 1]  # of each (site, file)
        for key, dtype in SITE_SUMS:
            if key == "count_all":
                per_file = np.diff(np.append(first, len(label_id)))
            else:
                per_file = grouped_sum(len(first), in_file, terms[key][order])  # as 1.2 sums a file's rows
            # np.add.at adds one by one, in order: the files of a site in file order
            np.add.at(self._total[key] if dtype is np.int64 else self._group[key], slots, per_file)

    def end_group(self):
        """Ends a worker group: its float sums are added to the total."""
        for key in FLOAT_SUMS:
            self._total[key][: self._n] += self._group[key][: self._n]
            self._group[key][: self._n] = 0

    def result(self):
        """(sorted unique label_ids, {SITE_SUMS name: array}) of all the rows added; call after the last
        end_group()."""
        order = np.argsort(self._label_id[: self._n])
        return self._label_id[: self._n][order], {key: arr[: self._n][order] for key, arr in self._total.items()}


# A site in genomic coordinates, as a site adder saves it for _genomic_part (the chromosome is where it is saved).
_GENOMIC_ROW = np.dtype([("pos", "<i8"), ("minus", "?")] + [(key, np.dtype(dtype).str) for key, dtype in SITE_SUMS])


def _site_adder(pid, meta_q, pipes, chunks, stem, ref_arr, genome, label_div, threshold, epsilon, calibrated, mask):
    """
    Adds up the per-site sums of one range of label_ids from the rows the readers send, and writes its part of the
    transcript table; with `genome`, also saves its sites in genomic coordinates for the genomic table.

    Args:
        pid (int): Index in the parent's process list, and of this adder's range of sites.
        meta_q (multiprocessing.Queue): Sends {"pid", "table": the output of _TablePart.close, "bad_refs": whether a
            label_id names a reference the BAM does not have, "refs": the references of its sites in site order,
            "genomic": saved sites per chromosome of `genome`} (or the error) to the parent.
        pipes (list): pipes[reader][site adder]: (receiving end, sending end).
        chunks (list): The output of _plan_chunks.
        stem (str): Path of this adder's files, without the extension.
        ref_arr (numpy.ndarray): Reference names of the BAM header.
        genome (GenomeMap): Where each reference lies on the genome; None without --annot.
        label_div (int): Divisor that separates the reference index and the position in label_id.
        threshold (float): Per-read cut that counts a read as positive.
        epsilon (float): Small constant for log and division safety.
        calibrated (bool): Whether the predictions get the v1.2 model's fixed calibration in the stoichiometry rule.
        mask (bool): Hide the stoichiometry of the sites that are not reported (False: --report-all-stoichiometry).
    """
    conns = [row[pid][0] for row in pipes]
    for conn in itertools.chain.from_iterable(itertools.chain.from_iterable(pipes)):
        if not any(conn is mine for mine in conns):
            conn.close()
    # A thread receives the chunks, up to _CHUNKS_AHEAD ahead of the sums: a reader sends its chunk on to the next
    # site adder, instead of waiting until this one has added up the chunks before.
    received = queue.Queue(maxsize=_CHUNKS_AHEAD)

    def receive():
        try:
            for i in range(len(chunks)):
                conn = conns[i % len(conns)]  # the reader of chunk i
                dtypes = (np.int64, np.float32, np.uint16)  # label_id, pred, file within the chunk
                received.put([np.frombuffer(conn.recv_bytes(), dtype=dtype) for dtype in dtypes])
        except BaseException as exc:  # noqa: BLE001 -- raised below, by the adder
            received.put(exc)

    # the receiving thread gets the interpreter lock soon after each read, so that the readers do not wait for it
    sys.setswitchinterval(_SWITCH_INTERVAL)
    threading.Thread(target=receive, daemon=True).start()
    table = _SiteTable(threshold, epsilon, calibrated)
    group = 0
    for _, _, chunk_group in chunks:
        if chunk_group != group:
            table.end_group()
            group = chunk_group
        rows = received.get()
        if isinstance(rows, BaseException):
            raise rows
        table.add(*rows)
    table.end_group()
    label_id, sums = table.result()
    del table

    part = _TablePart(stem)
    bad_refs, refs, genomic = False, [], []
    for a in range(0, len(label_id), _PART_SITES):
        ids = label_id[a : a + _PART_SITES]
        part_sums = {key: arr[a : a + _PART_SITES] for key, arr in sums.items()}
        transcript_id, ref_pos = _decode_label_id(ids, label_div)
        if np.max(transcript_id) >= len(ref_arr):
            bad_refs = True  # the parent stops the run
            break
        ## Calculate modscore and stoichiometry metrics, and write the sites
        scores = _site_scores(part_sums, threshold, epsilon, mask)
        part.add(ref_arr[transcript_id], ref_pos, np.where(ids > 0, "+", "-"), scores, part_sums)
        if genome is not None:
            # a '-' site is an A on the strand opposite the transcript: the same base, on the other genomic strand
            sites, chrom, minus, pos = genome.map(transcript_id, ref_pos, ids < 0)
            rows = np.empty(len(sites), dtype=_GENOMIC_ROW)
            rows["pos"], rows["minus"] = pos, minus
            for key in part_sums:
                rows[key] = part_sums[key][sites]
            refs.append(pd.unique(transcript_id))
            genomic.append((chrom, rows))
    payload = {"pid": pid, "table": part.close(), "bad_refs": bad_refs}
    if genome is not None:
        chrom = np.concatenate([c for c, _ in genomic]) if genomic else np.empty(0, np.int64)
        rows = np.concatenate([r for _, r in genomic]) if genomic else np.empty(0, _GENOMIC_ROW)
        # by chromosome, in site order within one (the order pileup_genomic adds them up in)
        np.save(stem + ".genomic.npy", rows[np.argsort(chrom, kind="stable")])
        payload["genomic"] = np.bincount(chrom, minlength=len(genome.chroms))
        payload["refs"] = pd.unique(np.concatenate(refs)) if refs else np.empty(0, np.int64)
    meta_q.put(payload)
    return None


def _genomic_part(rows, name, stem, threshold, epsilon, mask=True):
    """Writes one chromosome's part of the genomic table: the sites that the site adders saved for it, added up per
    (strand, position) as pileup_genomic adds them up, then scored and written as the transcript table.

    Args:
        rows (list): (file of saved sites, first row, end) of each site adder with sites on the chromosome, in the
            order of the site adders.
        name (numpy.ndarray): The chromosome's name, as an array of one string.
        stem (str): Path of the part's files, without the extension.
        threshold (float): Per-read cut.
        epsilon (float): Small constant for division safety.
        mask (bool): Hide the stoichiometry of the sites that are not reported (see _site_stoichiometry).

    Returns:
        dict: The output of _TablePart.close.
    """
    sites = np.concatenate([np.load(path, mmap_mode="r")[a:b] for path, a, b in rows])
    df = pd.DataFrame({"strand": np.where(sites["minus"], "-", "+").astype(object), "pos": sites["pos"]})
    for key, _ in SITE_SUMS:
        df[key] = sites[key]
    del sites
    df = df.groupby(["strand", "pos"], as_index=False).agg({key: "sum" for key, _ in SITE_SUMS})
    part = _TablePart(stem)
    for a in range(0, len(df), _PART_SITES):
        piece = df.iloc[a : a + _PART_SITES]
        sums = {key: piece[key].to_numpy() for key, _ in SITE_SUMS}
        part.add(
            np.repeat(name, len(piece)),
            piece["pos"].to_numpy(dtype=np.int64),
            piece["strand"].to_numpy(),
            _site_scores(sums, threshold, epsilon, mask),
            sums,
        )
    return part.close()


# The files of the saved runs (see _ReadRows.save_by_reference): name -> (dtype, values per row).
_SAVED_FILES = {
    "rows": (None, 1),
    "part_key": (np.int64, 4),
    "part_at": (np.int64, 1),
    "keys_ref": (np.int64, 1),
    "keys": (np.int64, 2),
    "keys_span": (np.int64, 3),
    "refidx": (np.int64, 2),
    "keyidx": (np.int64, 2),
}
# path -> the whole file, memory-mapped once per process (until _remove_modbam_temp), the most recently used last. A map
# holds a file descriptor, so at most _saved_cap() maps are kept.
_SAVED_MAPS = collections.OrderedDict()


def _saved_cap():
    """Maps _saved may keep: those it holds plus the file descriptors still free, less 256 left for everything else
    (the BAM, the shard being written, pipes, samtools). With a low open-file limit that cannot be raised, the least
    recently used maps are dropped."""
    soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    if soft == resource.RLIM_INFINITY:
        return 1 << 20
    return max(16, len(_SAVED_MAPS) + soft - _open_files() - 256)


def _open_files():
    """The file descriptors this process has open (listed on Linux and macOS; elsewhere estimated from the maps)."""
    for fd_dir in ("/proc/self/fd", "/dev/fd"):
        try:
            return len(os.listdir(fd_dir))
        except OSError:
            continue
    return len(_SAVED_MAPS) + 64


def _raise_open_files(want):
    """Raises the open-file soft limit toward `want`, as far as the hard limit allows (prep does the same)."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = want if hard == resource.RLIM_INFINITY else min(hard, want)
    if soft != resource.RLIM_INFINITY and target > soft:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        except (ValueError, OSError) as exc:
            log.warning(f"Could not raise the open-file limit from {soft} ({exc}).")


def _saved_dtype(name):
    dtype, width = _SAVED_FILES[name]
    return _ROW_DTYPE if dtype is None else np.dtype(dtype), width


def _saved_len(prefix, name):
    """The number of rows already in the saved-run file <prefix>.<name>."""
    dtype, width = _saved_dtype(name)
    path = f"{prefix}.{name}"
    return os.path.getsize(path) // (dtype.itemsize * width) if os.path.exists(path) else 0


def _save_append(prefix, name, arr):
    """Appends `arr` (C order, of the file's dtype) to the saved-run file <prefix>.<name>; returns its rows."""
    dtype, _ = _saved_dtype(name)
    with open(f"{prefix}.{name}", "ab") as fh:
        np.ascontiguousarray(arr, dtype=dtype).tofile(fh)
    return len(arr)


def _starts_index(ref):
    """(reference, first row) of each reference in the sorted reference column `ref`, as an (n, 2) array."""
    start = np.flatnonzero(np.concatenate(([True], ref[1:] != ref[:-1]))) if len(ref) else np.empty(0, np.int64)
    return np.stack([ref[start], start], axis=1)


def _saved(stem, name):
    """One saved run's array `name` (see _SAVED_FILES), as a view of the worker's file, memory-mapped once per process.
    A run's name is <prefix>#<first row>:<rows>:<first part>:<parts>:<first ...> (the same for its two indexes), as
    _ReadRows.save_by_reference returns it."""
    prefix, _, where = stem.rpartition("#")
    row_at, n_rows, key_at, n_keys, ridx_at, n_ridx, kidx_at, n_kidx = (int(x) for x in where.split(":"))
    at, n = {"refidx": (ridx_at, n_ridx), "keyidx": (kidx_at, n_kidx)}.get(
        name, (key_at, n_keys) if name.startswith(("keys", "part")) else (row_at, n_rows)
    )
    path = f"{prefix}.{name}"
    whole = _SAVED_MAPS.get(path)
    if whole is None or at + n > len(whole):  # not mapped yet, or mapped before this run was appended
        dtype, width = _saved_dtype(name)
        whole = np.memmap(path, dtype=dtype, mode="r") if os.path.getsize(path) else np.empty(0, dtype)
        if width > 1:
            whole = whole.reshape(-1, width)
        _SAVED_MAPS[path] = whole
        cap = _saved_cap()
        while len(_SAVED_MAPS) > cap:
            _SAVED_MAPS.popitem(last=False)  # closed once no array uses it
    else:
        _SAVED_MAPS.move_to_end(path)
    return whole[at : at + n]


def _bounds(stem, lo, hi, keys=False):
    """[a, b): the rows (with `keys`, the parts) of one saved run whose reference index lies in [lo, hi), from the
    run's index of where each reference starts."""
    idx = _saved(stem, "keyidx" if keys else "refidx")
    n = len(_saved(stem, "keys_ref" if keys else "rows"))
    i, j = (int(x) for x in np.searchsorted(idx[:, 0], [lo, hi], side="left"))
    return (int(idx[i, 1]) if i < len(idx) else n), (int(idx[j, 1]) if j < len(idx) else n)


def _ref_slices(rows_stems, refs, max_rows):
    """Splits `refs` (sorted reference indices) into runs of consecutive references whose rows in the saved tables
    `rows_stems` add up to about `max_rows` each; a run holds at least one reference, so one reference with more rows is
    a run of its own (the writer then reads it in windows of positions). The row counts come from the saved reference
    columns, read in place.

    Returns:
        list: (reference indices of the run, rows of the run) pairs.
    """
    refs = np.asarray(refs, dtype=np.int64)
    counts = np.zeros(len(refs), dtype=np.int64)
    for stem in rows_stems:
        idx = _saved(stem, "refidx")
        ends = np.append(idx[1:, 1], len(_saved(stem, "rows")))
        i = np.searchsorted(idx[:, 0], refs, side="left")
        found = i < len(idx)
        found[found] = idx[i[found], 0] == refs[found]
        counts[found] += ends[i[found]] - idx[i[found], 1]
    slices, start, held = [], 0, 0
    for i, n in enumerate(counts.tolist()):
        if held and held + n > max_rows:
            slices.append((refs[start:i], held))
            start, held = i, 0
        held += n
    slices.append((refs[start:], held))
    return [(s, n) for s, n in slices if s.size]


def _read_spans(rows_stems, tid):
    """(first position, last position, rows) of each part (see _ReadRows.save_by_reference) on reference `tid`, from
    the saved runs' part files, as an (n, 3) array."""
    spans = []
    for stem in rows_stems:
        a, b = _bounds(stem, tid, tid + 1, keys=True)
        if b > a:
            spans.append(np.array(_saved(stem, "keys_span")[a:b]))
    return np.concatenate(spans) if spans else np.empty((0, 3), np.int64)


def _pos_windows(rows_stems, tid, n_rows, max_rows):
    """Start positions of windows of at most about `max_rows` rows of reference `tid` (`n_rows`: its rows, not needed
    any more); the first window starts at 0. Any starts give the same modBAM; these make the windows cheap to write.

    A window starts where no read's rows cross, in a gap between the reads' spans (for reads aligned to a genome:
    between genes), and takes the following runs of overlapping reads while they fit, so that it walks and loads the
    rows of its own reads only. A run of overlapping reads with more than `max_rows` rows is cut inside, at every
    `max_rows` rows. The rows are counted from the parts (their first position decides the window that loads them)."""
    spans = _read_spans(rows_stems, tid)
    if not len(spans):
        return np.zeros(1, dtype=np.int64)
    spans = spans[np.argsort(spans[:, 0], kind="stable")]
    first = spans[:, 0]
    before = np.concatenate(([0], np.cumsum(spans[:, 2])))  # rows of the parts before each part; then of all
    # the gaps: a run of overlapping spans starts at a part whose first row lies after the last row of every part
    # before; the window starts just after that last row (a read starts at or before its first row)
    reach = np.maximum.accumulate(spans[:, 1])
    gaps = reach[:-1][first[1:] > reach[:-1]] + 1
    starts, s = [0], 0  # s: the rows before the current window's start
    last_gap, last_r = 0, 0
    ends = [*zip(gaps.tolist(), before[np.searchsorted(first, gaps, side="left")].tolist()), (None, int(before[-1]))]
    for gap, r in ends:  # each gap, then the end of the reference
        if r - s > max_rows:  # the current window would pass `max_rows` here
            if last_r > s:  # end it at the gap before
                starts.append(last_gap)
                s = last_r
            while r - s > max_rows:  # a run of overlapping reads too large for one window: cut at a part's start
                k = int(np.searchsorted(before, s + max_rows, side="right")) - 1
                if before[k] <= s:  # one part larger than a window: cut after it
                    k += 1
                if k < len(first) and first[k] <= starts[-1]:
                    # more than `max_rows` rows of parts that start at one position (reads of one end of a deep
                    # transcript): they are one window, and the next starts at the position after
                    k = int(np.searchsorted(first, starts[-1], side="right"))
                if k >= len(first):
                    break
                starts.append(int(first[k]))
                s = int(before[k])
        last_gap, last_r = gap, r
    return np.unique(np.array(starts, dtype=np.int64))


class _RefRuns:
    """The saved rows of one reference in each of the saved runs `rows_stems` that has some (in their order): where
    they lie, found once."""

    def __init__(self, rows_stems, tid):
        self.tid = tid
        self.tables = []  # (run, first row of the reference, end)
        for stem in rows_stems:
            a, b = _bounds(stem, tid, tid + 1)
            if b > a:
                self.tables.append((stem, a, b))

    def __len__(self):
        return len(self.tables)

    def arrays(self, k):
        """The reference's rows in run k: (reference, first position and read id of each part; its first row; the
        rows)."""
        stem, a, b = self.tables[k]
        at = _saved(stem, "part_at")
        i, j = (int(x) for x in np.searchsorted(at, [a, b], side="left"))
        return np.array(_saved(stem, "part_key")[i:j]), at[i:j] - a, _saved(stem, "rows")[a:b]


_KEY_MIX = np.uint64(0x9E3779B97F4A7C15)


def _key_hash(hi, lo):
    """A 64-bit hash of read ids (high, low int64 halves; arrays or scalars) that mixes both halves, so that read ids
    which differ only in one half (not random UUIDs) still spread out."""
    with np.errstate(over="ignore"):
        return np.asarray(hi).astype(np.uint64) * _KEY_MIX + np.asarray(lo).astype(np.uint64)


def _ref_parts(runs):
    """The parts (see _ReadRows.save_by_reference) of the reference of `runs` (a _RefRuns), from the runs' part
    files, as arrays sorted by the hash of their read id: "hash", "hi", "lo" (read id), "first", "last" (positions) and
    "table" (index in `runs`)."""
    cols = []
    for k, (stem, _, _) in enumerate(runs.tables):
        a, b = _bounds(stem, runs.tid, runs.tid + 1, keys=True)
        keys = np.array(_saved(stem, "keys")[a:b])
        span = np.array(_saved(stem, "keys_span")[a:b])
        cols.append((keys[:, 0], keys[:, 1], span[:, 0], span[:, 1], np.full(b - a, k, dtype=np.int64)))
    names = ("hi", "lo", "first", "last", "table")
    if not cols:
        parts = {name: np.empty(0, np.int64) for name in names}
        parts["hash"] = np.empty(0, np.uint64)
        return parts
    parts = {name: np.concatenate([c[i] for c in cols]) for i, name in enumerate(names)}
    parts["hash"] = _key_hash(parts["hi"], parts["lo"])
    order = np.argsort(parts["hash"], kind="stable")
    return {name: arr[order] for name, arr in parts.items()}


def _read_parts(read, parts):
    """(read id high, read id low, indices of its parts in `parts`) of a read with rows on the reference of `parts`,
    else None. The read id comes from the name, however it spells the UUID (canonical, upper case, no hyphens, ...)."""
    if read.query_name is None or read.query_sequence is None:
        return None
    try:
        hi, lo = (int(x) for x in np.frombuffer(uuid.UUID(read.query_name).bytes, dtype=np.int64))
    except (ValueError, AttributeError):
        return None
    h = _key_hash(np.int64(hi), np.int64(lo))
    a = int(np.searchsorted(parts["hash"], h, side="left"))
    b = int(np.searchsorted(parts["hash"], h, side="right"))
    idx = [m for m in range(a, b) if parts["hi"][m] == hi and parts["lo"][m] == lo]
    return (hi, lo, idx) if idx else None


def _write_modbam_windows(in_bam, out_bam, rows_stems, tid, starts, stop, bar, tag_ids=None):
    """Writes the reads of reference `tid` of the indexed `in_bam` that start from starts[0] to `stop` (None: to the
    end), a window of positions at a time (window i from starts[i] to starts[i + 1]), so that a writer never holds all
    the rows of a large reference (a chromosome, for reads aligned to a genome; a transcript with deep coverage).

    The reads come in one walk over the BAM, in its (coordinate) order, as in one pass over the reference; each window
    takes those that start in it and have rows on the reference (the others are not written, as in one pass). Its rows
    are those of the parts that start in the window (the rows are saved sorted by the first position of their part),
    and the parts of its reads that start elsewhere but overlap the read's alignment: every row inside a read's
    alignment, which are the only rows its tags can use. So a window loads about the rows of its own reads, however far
    they reach. A read with rows on the reference, but none in its alignment (they come from another alignment of the
    same read), gets the tags of no row, as in one pass.
    """
    runs = _RefRuns(rows_stems, tid)
    parts = _ref_parts(runs)
    tag_ids = _merge_tag_ids(in_bam.header) if tag_ids is None else tag_ids
    window, reads = 0, []
    for read in in_bam.fetch(tid=tid, start=starts[0], stop=stop):
        if read.reference_start < starts[0]:
            continue  # written with the task before
        while window + 1 < len(starts) and read.reference_start >= starts[window + 1]:
            _write_window(out_bam, reads, runs, parts, starts[window], starts[window + 1], tag_ids, bar)
            window, reads = window + 1, []
        bar.update(1)
        found = _read_parts(read, parts)
        if found is not None:
            reads.append((read, *found))
    _write_window(out_bam, reads, runs, parts, starts[window], stop, tag_ids, bar)


def _write_window(out_bam, reads, runs, parts, start, stop, tag_ids, bar):
    """Writes the reads of one window from `start` to `stop` (None: to the end of the reference) of
    _write_modbam_windows; `reads`: (read, read id high, read id low, indices of its parts) of each."""
    if not reads:
        return
    # the parts that start outside the window but overlap the alignment of their read, loaded by (run, first position)
    extra = set()
    for read, hi, lo, idx in reads:
        r_start, r_end = read.reference_start, read.reference_end or read.reference_start + 1
        for m in idx:
            first = int(parts["first"][m])
            outside = first < start or (stop is not None and first >= stop)
            if outside and parts["last"][m] >= r_start and first < r_end:
                extra.add((int(parts["table"][m]), first, hi, lo))
    rows = _ReadRows.load_window(
        None, runs.tid, start, np.iinfo(np.int64).max if stop is None else stop, runs, sorted(extra)
    )
    empty = (np.empty(0, np.int64), np.empty(0, np.uint8))
    batch = []
    for read, hi, lo, _ in reads:
        payload = rows.get((runs.tid, hi, lo))
        batch.append((read, empty if payload is None else payload))
        if len(batch) >= _MODBAM_BATCH:
            _write_batch(batch, out_bam, tag_ids)
            batch = []
    _write_batch(batch, out_bam, tag_ids)


def _mapped_per_reference(bam):
    """{reference index: mapped reads} of an indexed BAM, from its index."""
    return {bam.get_tid(stat.contig): stat.mapped for stat in bam.get_index_statistics()}


class _NoBar:
    """A progress bar that shows nothing (the modBAM writers run while the main process writes the site tables)."""

    def update(self, n=1):
        pass


def _plan_modbam_tasks(bam_path, rows_stems, ref_ids, threads):
    """The modBAM tasks, in reference order, about _MODBAM_TASKS_PER_WRITER per writer with about the same number of
    aligned reads each: ("refs", reference indices) for a run of consecutive references, or ("windows", reference
    index, window starts, stop) for a part of one large reference (its windows of positions, as in
    _write_modbam_windows). Every read of the BAM's references with rows falls in one task, so the shards of the tasks,
    joined in order, hold the records of one pass over the references.

    The references are taken in order: consecutive ones go into a task until it holds about its share of the reads,
    and a reference with more than two shares is split into parts of about one share each, whatever its neighbours."""
    n_tasks = max(1, threads * _MODBAM_TASKS_PER_WRITER)
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        mapped = _mapped_per_reference(bam)
        n_refs = bam.nreferences
    ref_ids = [int(tid) for tid in np.asarray(ref_ids, dtype=np.int64) if 0 <= tid < n_refs and mapped.get(int(tid))]
    if not ref_ids:
        return []
    target = max(1, sum(mapped[tid] for tid in ref_ids) // n_tasks)
    tasks, run, held = [], [], 0

    def end_run():
        nonlocal run, held
        if run:
            tasks.append(("refs", np.array(run, dtype=np.int64)))
        run, held = [], 0

    for tid in ref_ids:
        if mapped[tid] > 2 * target:
            # a reference with the reads of several tasks: its windows, in parts of about `target` reads
            end_run()
            starts = _pos_windows(rows_stems, tid, None, _MODBAM_SLICE_ROWS).tolist()
            k = max(1, min(len(starts), -(-mapped[tid] // target)))
            cuts = [len(starts) * j // k for j in range(k + 1)]
            for j in range(k):
                stop = starts[cuts[j + 1]] if j + 1 < k else None
                tasks.append(("windows", tid, starts[cuts[j] : cuts[j + 1]], stop))
            continue
        if held and held + mapped[tid] > target:
            end_run()
        run.append(tid)
        held += mapped[tid]
    end_run()
    return tasks


def _merge_order_header(header, comments=True):
    """The BAM header with its lines in the order samtools merge -c -p wrote them into the modBAM up to DeepRM 1.2:
    @HD, then every @SQ, @RG, @PG and @CO line, each kind in its own order (lines of other kinds dropped, as merge
    did). Without `comments`, no @CO line: the writers give the shards that header, and the join puts the full one on
    the modBAM, so each comment is there once (merge copied it once per shard). Lines end at newlines only (a field
    can hold a carriage return)."""
    kinds = {"@HD": [], "@SQ": [], "@RG": [], "@PG": [], "@CO": []}
    for line in str(header).split("\n"):
        if line[:3] in kinds and (comments or line[:3] != "@CO"):
            kinds[line[:3]].append(line)
    return pysam.AlignmentHeader.from_text("".join(line + "\n" for lines in kinds.values() for line in lines))


def _write_modbam_task(in_bam, task, rows_stems, shard_path, tag_ids=None, header=None):
    """Writes one task's shard (BGZF level _MODBAM_LEVEL) from the rows of every worker; returns its path, or None
    when the task has no row. The rows are loaded a slice of references (_ref_slices), or for a large reference a
    window of positions, at a time, and the slices are written one after another into the shard. `header`: the
    shard's header, _merge_order_header of the BAM's (made here when None)."""
    out_bam = None

    def out():
        nonlocal out_bam
        if out_bam is None:
            out_bam = pysam.AlignmentFile(
                shard_path,
                "wb",
                header=_merge_order_header(in_bam.header) if header is None else header,
                format_options=[f"level={_MODBAM_LEVEL}".encode()],
            )
        return out_bam

    bar = _NoBar()
    try:
        if task[0] == "windows":
            _, tid, starts, stop = task
            _write_modbam_windows(in_bam, out(), rows_stems, tid, starts, stop, bar, tag_ids)
        else:
            for refs_slice, n_rows in _ref_slices(rows_stems, task[1], _MODBAM_SLICE_ROWS):
                if len(refs_slice) == 1 and n_rows > _MODBAM_SLICE_ROWS and in_bam.has_index():
                    tid = int(refs_slice[0])
                    starts = _pos_windows(rows_stems, tid, n_rows, _MODBAM_SLICE_ROWS).tolist()
                    _write_modbam_windows(in_bam, out(), rows_stems, tid, starts, None, bar, tag_ids)
                    continue
                rows = _ReadRows.load(rows_stems, int(refs_slice[0]), int(refs_slice[-1]) + 1)
                if len(rows):
                    _write_modbam_reads(in_bam, out(), rows, refs_slice, bar, tag_ids)
                del rows
    finally:
        if out_bam is not None:
            out_bam.close()
    return shard_path if out_bam is not None else None


def _open_shared_bam(bam_path):
    """Opens the BAM once for all the modBAM writers. Forked after this, they share its header and index in memory
    (over 1 GB for the hundreds of thousands of references of a transcriptome) instead of each loading its own.
    Returns (the open BAM, its file descriptor), or (None, None) where the descriptor cannot be told (no /proc):
    each writer then opens the BAM itself."""
    try:
        before = set(os.listdir("/proc/self/fd"))
        bam = pysam.AlignmentFile(bam_path, "rb")
        real = os.path.realpath(bam_path)
        opened = set(os.listdir("/proc/self/fd")) - before
        fds = [int(fd) for fd in opened if os.path.realpath(f"/proc/self/fd/{fd}") == real]
    except OSError:
        return None, None
    if len(fds) != 1:
        bam.close()
        return None, None
    return bam, fds[0]


def _unshare_file(fd, path):
    """Gives the descriptor `fd` of the file `path`, inherited from the parent process, a file position of its own
    (forked processes share one), at the place it is at."""
    own = os.open(path, os.O_RDONLY)
    try:
        os.lseek(own, os.lseek(fd, 0, os.SEEK_CUR), os.SEEK_SET)
        os.dup2(own, fd)
    finally:
        os.close(own)


def _trim_memory():
    """Returns the memory freed so far to the system; the C library otherwise keeps it for the process (glibc only)."""
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except (AttributeError, OSError):
        pass


def _modbam_writer(task_q, result_q, tasks, rows_stems, bam_path, tmp_dir, shared=(None, None)):
    """One modBAM writer process: takes task numbers from `task_q` until None, and writes each task's shard as
    <tmp_dir>/modbam_shard_<task>.bam. `shared`: the output of _open_shared_bam in the parent."""
    try:
        in_bam, fd = shared
        if in_bam is None:
            in_bam = pysam.AlignmentFile(bam_path, "rb")
        else:
            _unshare_file(fd, bam_path)
        try:
            tag_ids, header = _merge_tag_ids(in_bam.header), _merge_order_header(in_bam.header, comments=False)
            while True:
                i = task_q.get()
                if i is None:
                    break
                shard = os.path.join(tmp_dir, f"modbam_shard_{i}.bam")
                path = _write_modbam_task(in_bam, tasks[i], rows_stems, shard, tag_ids, header)
                if path:
                    _index_shard(path)
                result_q.put({"task": i, "path": path})
                _trim_memory()  # a task of a deep reference holds much more than the next one needs
        finally:
            in_bam.close()
    except BaseException:  # noqa: BLE001 -- reported to the parent, then re-raised
        result_q.put({"task": None, "error": traceback.format_exc()})
        raise SystemExit(1)


def _start_modbam_writers(bam_path, rows_stems, ref_ids, threads, tmp_dir):
    """Starts up to `threads` modBAM writer processes, which take the tasks of _plan_modbam_tasks from a queue (so a
    writer that ends a task early takes the next one) and write their shards into `tmp_dir`. Returns what
    _finish_modbam and _stop_modbam_writers take, or None when no reference has reads."""
    tasks = _plan_modbam_tasks(bam_path, rows_stems, ref_ids, threads)
    _SAVED_MAPS.clear()  # the planner's maps: each writer maps what it needs, and the parent needs none of them
    if not tasks:
        return None
    task_q, result_q = _MP.Queue(), _MP.Queue()
    n_procs = min(threads, len(tasks))
    procs = []
    shared = _open_shared_bam(bam_path)
    try:  # the writers are forked before the first put starts the queue's feeder thread
        for _ in range(n_procs):
            writer_args = (task_q, result_q, tasks, rows_stems, bam_path, tmp_dir, shared)
            proc = _MP.Process(target=_modbam_writer, args=writer_args)
            proc.start()
            procs.append(proc)
        for i in range(len(tasks)):
            task_q.put(i)
        for _ in range(n_procs):
            task_q.put(None)
    except BaseException:  # a writer that cannot start: stop those that did
        _stop_modbam_writers({"procs": procs, "bam": shared[0]})
        raise
    return {"procs": procs, "result_q": result_q, "tasks": tasks, "tmp_dir": tmp_dir, "bam": shared[0]}


def _finish_modbam(writers, bam_path, out_path, threads):
    """Waits for the modBAM writers and joins their shards, in task order, into `out_path` (indexed). Returns False
    when no read got a record."""
    procs, result_q, n_tasks = writers["procs"], writers["result_q"], len(writers["tasks"])
    paths = {}
    try:
        while len(paths) < n_tasks:
            try:
                result = result_q.get(timeout=5.0)
            except queue.Empty:
                if all(proc.exitcode is not None for proc in procs) or any(proc.exitcode for proc in procs):
                    try:  # a result sent just before its writer ended
                        result = result_q.get(timeout=1.0)
                    except queue.Empty:
                        codes = [proc.exitcode for proc in procs]
                        raise RuntimeError(
                            f"modBAM writers ended with {n_tasks - len(paths)} of {n_tasks} tasks not done (exit codes "
                            f"{codes})."
                        ) from None
                else:
                    continue
            if result.get("error"):
                raise RuntimeError(f"modBAM writer failed:\n{result['error']}")
            paths[result["task"]] = result["path"]
    finally:
        _stop_modbam_writers(writers)
    written = [i for i in range(n_tasks) if paths[i]]
    if not written:
        return False
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        coordinate_sorted = bam.header.to_dict().get("HD", {}).get("SO") == "coordinate"
        header = _merge_order_header(bam.header)  # with the comments, which the shards do not carry
    # the shards hold consecutive references in order, each fetched in coordinate order from the indexed BAM, so
    # joined in order they are sorted already; with a header that says so, the sort is skipped
    merge_modbam_shards(
        [paths[i] for i in written],
        out_path,
        threads,
        presorted=coordinate_sorted,
        header=header,
        shard_refs=[_task_refs(writers["tasks"][i]) for i in written],
    )
    return True


def _task_refs(task):
    """The references a modBAM task writes, sorted."""
    return np.array([task[1]], dtype=np.int64) if task[0] == "windows" else np.asarray(task[1], dtype=np.int64)


def _index_shard(path):
    """Indexes a shard as soon as it is written (it is still in the page cache), so that the join can put the
    modBAM's index together from the shards' (_merge_shard_indexes) instead of reading the whole modBAM again. A shard
    that cannot be indexed is left without an index, and the shards are then joined by samtools."""
    try:
        pysam.index(path)
    except pysam.SamtoolsError as exc:
        log.warning(f"Could not index the modBAM part {os.path.basename(path)} ({exc}); samtools will join the parts.")
        if os.path.exists(path + ".bai"):  # a partial index (a full disk, for example) is not left for the join
            os.remove(path + ".bai")


def _stop_modbam_writers(writers):
    """Stops the modBAM writers that are still running (after a failure elsewhere) and closes them; a no-op the second
    time."""
    _stop_processes(writers["procs"])
    bam = writers.pop("bam", None)  # the BAM the writers shared
    if bam is not None:
        bam.close()


def merge_modbam_shards(shard_paths, out_path, threads, presorted=False, header=None, shard_refs=None):
    """Merges modBAM shards (in the order given) into one coordinate-sorted, indexed modBAM, and removes the shards.

    Every shard has the input BAM's header. With `presorted` (the shards hold consecutive references in order, each
    coordinate-sorted, under a header that says SO:coordinate) the shards are joined as they are (samtools cat copies
    their compressed blocks, so nothing is decompressed or compressed again): the records are those a merge writes,
    in the same order, under the first shard's header. The result is sorted only if indexing it fails. Otherwise they
    are merged (-c / -p keep the input's @RG and @PG IDs and the reads' RG tags instead of giving each shard its own
    renamed copy) and sorted. `header` (a pysam header), when given, is the modBAM's header (samtools -h), instead of
    the first shard's (cat) or the shards' merged (merge). With `presorted`, `header` and `shard_refs` (the sorted
    references of each shard) and an index beside every shard, the modBAM is first put together without reading the
    shards (_join_by_copy, _merge_shard_indexes); if that cannot be done, samtools joins them as above.
    """
    keep = [i for i, p in enumerate(shard_paths) if p and os.path.exists(p)]
    shard_paths = [shard_paths[i] for i in keep]
    shard_refs = None if shard_refs is None else [shard_refs[i] for i in keep]
    if not shard_paths:
        return None

    # written under temporary names and moved into place when complete, so that a failure leaves no partial modBAM
    part_path, unsorted_path = out_path + ".part.bam", out_path + ".unsorted.bam"
    header_path = os.path.join(os.path.dirname(shard_paths[0]), "modbam_header.sam")
    header_args = []
    pysam_thread_memory = sort_memory_per_thread(threads)
    moved = False
    try:
        joined = presorted and header is not None and shard_refs is not None
        joined = joined and _join_without_reading(shard_paths, shard_refs, part_path, header, threads)
        if not joined:
            if header is not None:
                with open(header_path, "w") as fh:
                    fh.write(str(header))
                header_args = ["-h", header_path]
            if presorted:
                pysam.cat(*header_args, "-o", part_path, *shard_paths)
                try:
                    pysam.index("-@", str(threads), part_path)
                except pysam.SamtoolsError:  # not sorted after all: sort it as below
                    os.replace(part_path, unsorted_path)
                    presorted = False
            else:
                pysam.merge("-@", str(threads), "-c", "-p", "-f", *header_args, unsorted_path, *shard_paths)
            if not presorted:
                pysam.sort("-@", str(threads), f"-m {pysam_thread_memory}M", "-o", part_path, unsorted_path)
                pysam.index("-@", str(threads), part_path)
                os.remove(unsorted_path)
        # an index left beside an earlier modBAM goes first, then the modBAM, then its index: stopped anywhere, the
        # folder never holds a modBAM with another modBAM's index
        if os.path.lexists(out_path + ".bai"):
            os.remove(out_path + ".bai")
        os.replace(part_path, out_path)
        moved = True
        os.replace(part_path + ".bai", out_path + ".bai")
    except BaseException:
        for path in (part_path, part_path + ".bai", unsorted_path) + ((out_path,) if moved else ()):
            if os.path.exists(path):
                os.remove(path)
        raise
    finally:
        if header_args:
            os.remove(header_path)

    for path in shard_paths:
        for name in (path, path + ".bai"):
            try:
                os.remove(name)
            except FileNotFoundError:
                pass
    return None


# The 28-byte end-of-file block of a BGZF file (SAM specification, 4.1.2), and the BAI pseudo-bin that holds a
# reference's span in the file and its numbers of mapped and unmapped reads (5.2.1).
_BGZF_EOF = bytes.fromhex("1f8b08040000000000ff0600424302001b0003000000000000000000")
_BAI_META_BIN = 37450


def _join_without_reading(shard_paths, shard_refs, part_path, header, threads=1):
    """Puts the modBAM `part_path` and its index together from the shards and their indexes without reading the
    shards' records (_join_by_copy, _merge_shard_indexes). Returns False, leaving no file, when that cannot be done
    (no copy_file_range, a shard without an index or of an unexpected layout); the caller then uses samtools."""
    missing = [os.path.basename(p) for p in shard_paths if not os.path.exists(p + ".bai")]
    if missing:
        log.warning(f"{len(missing)} modBAM parts have no index (first: {missing[0]}); samtools joins the parts.")
        return False
    try:
        spans = _join_by_copy(shard_paths, part_path, header, threads)
        _merge_shard_indexes(shard_paths, shard_refs, spans, part_path + ".bai")
    except (OSError, ValueError, struct.error) as exc:
        log.warning(f"The modBAM parts could not be joined by copying ({exc}); samtools joins them instead.")
        for path in (part_path, part_path + ".bai"):
            if os.path.exists(path):
                os.remove(path)
        return False
    log.info(f"Joined {len(shard_paths)} modBAM parts by copying them, with their indexes merged.")
    return True


def _copy_file_range():
    """os.copy_file_range, or else the C library's through ctypes (a Python built against an old C library lacks the
    first); None when there is neither."""
    if hasattr(os, "copy_file_range"):
        return os.copy_file_range
    try:
        fn = ctypes.CDLL(None, use_errno=True).copy_file_range
    except (AttributeError, OSError):
        return None
    offset = ctypes.POINTER(ctypes.c_longlong)
    fn.argtypes = [ctypes.c_int, offset, ctypes.c_int, offset, ctypes.c_size_t, ctypes.c_uint]
    fn.restype = ctypes.c_ssize_t

    def copy_file_range(src, dst, count, offset_src, offset_dst):
        a, b = ctypes.c_longlong(offset_src), ctypes.c_longlong(offset_dst)
        n = fn(src, ctypes.byref(a), dst, ctypes.byref(b), count, 0)
        if n < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        return n

    return copy_file_range


def _copy_bytes(src_fd, dst_fd, count, offset_src, offset_dst):
    """Copies `count` bytes between two open files with copy_file_range: on NFS 4.2 the file server copies them, so
    they do not travel to this machine and back; on a local file system the kernel copies them. Between two file
    systems, which copy_file_range refuses, the bytes are read and written."""
    copy = _copy_file_range()
    if copy is None:
        raise OSError(errno.ENOSYS, "copy_file_range is not available")
    across = False  # the two files are on different file systems (a --tmp-dir elsewhere): copy through this machine
    while count:
        if not across:
            try:
                n = copy(src_fd, dst_fd, min(count, 1 << 30), offset_src, offset_dst)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                across = True
        if across:
            n = os.pwrite(dst_fd, os.pread(src_fd, min(count, 1 << 23), offset_src), offset_dst)
        if n <= 0:
            raise OSError(errno.EIO, f"copy_file_range copied {n} bytes")
        count, offset_src, offset_dst = count - n, offset_src + n, offset_dst + n


def _join_by_copy(shard_paths, part_path, header, threads=1):
    """Writes the modBAM `part_path`: `header`, then each shard's compressed blocks after its own header, copied as
    they are (samtools cat copies the same blocks), then the end-of-file block. Up to _JOIN_COPIES shards (at most
    `threads`) are copied at a time, each to its own place in the modBAM. Returns, per shard, (where its records
    start in it, where they start in the modBAM, their length in bytes). Raises ValueError for a shard whose records
    do not start a block or that does not end with the end-of-file block."""
    with pysam.AlignmentFile(part_path, "wb", header=header):
        pass
    at = os.path.getsize(part_path) - len(_BGZF_EOF)
    out = os.open(part_path, os.O_RDWR)
    try:
        if os.pread(out, len(_BGZF_EOF), at) != _BGZF_EOF:
            raise ValueError("the header does not end with the BGZF end-of-file block")
        spans = []
        head = None  # the compressed header of the shard before
        for path in shard_paths:
            size = os.path.getsize(path)
            with open(path, "rb") as fh:
                # The writers give every shard the same header, so its blocks are the same bytes in each: a shard
                # that starts with the bytes of the one before has its first record where that one has it. Its own
                # header is parsed only otherwise (a header can have hundreds of thousands of lines).
                if head is None or os.pread(fh.fileno(), len(head), 0) != head:
                    with pysam.AlignmentFile(path, "rb", check_sq=False) as shard:
                        first = shard.tell()  # the virtual offset of the first record
                    head = os.pread(fh.fileno(), first >> 16, 0)
                if first & 0xFFFF or os.pread(fh.fileno(), len(_BGZF_EOF), size - len(_BGZF_EOF)) != _BGZF_EOF:
                    raise ValueError(f"{path}: its records do not start a block, or it has no end-of-file block")
            start = first >> 16
            n = size - len(_BGZF_EOF) - start
            spans.append((start, at, n))
            at += n
        os.ftruncate(out, at + len(_BGZF_EOF))
        os.pwrite(out, _BGZF_EOF, at)

        def copy(job):
            path, (start, to, n) = job
            fin = os.open(path, os.O_RDONLY)
            try:
                _copy_bytes(fin, out, n, start, to)
            finally:
                os.close(fin)

        # threads: copy_file_range runs without the interpreter lock
        with concurrent.futures.ThreadPoolExecutor(max(1, min(threads, _JOIN_COPIES))) as pool:
            list(pool.map(copy, zip(shard_paths, spans)))
    finally:
        os.close(out)
    return spans


def _read_bai(path, refs):
    """The index (BAI) of a shard with reads of the references `refs` (sorted) only: (number of references,
    {reference: ({bin: (n, 2) array of chunk virtual offsets}, linear-index offsets)}, reads without a coordinate).
    Raises ValueError when any other reference has an entry."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:4] != b"BAI\1":
        raise ValueError(f"{path}: not a BAI index")
    n_ref = struct.unpack_from("<i", data, 4)[0]
    pos, cur, entries = 8, 0, {}
    for r in [int(r) for r in refs] + [n_ref]:
        gap = 8 * (r - cur)  # the references in between: no bin and no linear offset each
        if r < cur or data[pos : pos + gap] != bytes(gap):
            raise ValueError(f"{path}: a reference outside the shard's has an index entry")
        pos += gap
        if r == n_ref:
            break
        (n_bin,) = struct.unpack_from("<i", data, pos)
        pos += 4
        bins = {}
        for _ in range(n_bin):
            b, n_chunk = struct.unpack_from("<Ii", data, pos)
            bins[b] = np.frombuffer(data, "<u8", 2 * n_chunk, pos + 8).reshape(-1, 2)
            pos += 8 + 16 * n_chunk
        (n_intv,) = struct.unpack_from("<i", data, pos)
        entries[r] = (bins, np.frombuffer(data, "<u8", n_intv, pos + 4))
        pos += 4 + 8 * n_intv
        cur = r + 1
    n_no_coor = struct.unpack_from("<Q", data, pos)[0] if pos + 8 <= len(data) else 0
    return n_ref, entries, n_no_coor


def _merge_shard_indexes(shard_paths, shard_refs, spans, out_path):
    """Writes the index of the modBAM of _join_by_copy from the shards' own indexes. A shard's blocks are copied as
    they are, so each virtual offset moves by where its block went and keeps its place inside the block. The chunks of
    a reference that is in several shards (a large reference written in windows) are put together in shard order;
    each 16 kb window of the linear index takes the smallest offset a shard gives it (each points before that shard's
    first read in the window, and the shards are in file order); the read counts add up."""
    n_refs, merged, n_no_coor = set(), {}, 0
    for path, refs, (start, at, n) in zip(shard_paths, shard_refs, spans):
        n_ref, entries, no_coor = _read_bai(path + ".bai", refs)
        n_refs.add(n_ref)
        n_no_coor += no_coor

        # every virtual offset of the shard (chunk ends, linear index, the meta bin's span; not its read counts),
        # moved in one call
        pieces = []
        for bins, lin in entries.values():
            pieces += [(chunks[:1] if b == _BAI_META_BIN else chunks).ravel() for b, chunks in bins.items()]
            pieces.append(lin)
        flat = np.concatenate(pieces) if pieces else np.empty(0, np.uint64)
        block = flat >> np.uint64(16)
        if np.any(block < start) or np.any(block > start + n):
            raise ValueError(f"{path}.bai: a virtual offset outside the shard's records")
        flat = ((block - np.uint64(start) + np.uint64(at)) << np.uint64(16)) | (flat & np.uint64(0xFFFF))
        k = 0
        for r, (bins, lin) in entries.items():
            moved_bins = {}
            for b, chunks in bins.items():
                m = 2 if b == _BAI_META_BIN else chunks.size
                offsets = flat[k : k + m]
                k += m
                # the meta bin: its span in the file moves, its read counts do not
                moved_bins[b] = np.stack([offsets, chunks[1]]) if b == _BAI_META_BIN else offsets.reshape(-1, 2)
            moved_lin = flat[k : k + len(lin)]
            k += len(lin)
            if not bins and not len(lin):
                continue
            m_bins, m_lin = merged.setdefault(r, ({}, []))
            for b, chunks in moved_bins.items():
                m_bins.setdefault(b, []).append(chunks)
            if len(lin):
                m_lin.append(moved_lin)
    if len(n_refs) != 1:
        raise ValueError(f"the shards' indexes have different numbers of references: {sorted(n_refs)}")
    n_ref = n_refs.pop()
    out = [b"BAI\1", struct.pack("<i", n_ref)]
    cur = 0
    for r in sorted(merged):
        bins, lins = merged[r]
        out.append(bytes(8 * (r - cur)))
        out.append(struct.pack("<i", len(bins)))
        for b in sorted(bins):
            parts = bins[b]
            if b == _BAI_META_BIN:
                span = np.array([min(p[0, 0] for p in parts), max(p[0, 1] for p in parts)], dtype="<u8")
                counts = np.array([sum(p[1, 0] for p in parts), sum(p[1, 1] for p in parts)], dtype="<u8")
                chunks = np.stack([span, counts])
            else:
                chunks = np.concatenate(parts).astype("<u8", copy=False)
            out.append(struct.pack("<Ii", b, len(chunks)))
            out.append(np.ascontiguousarray(chunks, dtype="<u8").tobytes())
        lin = np.full(max((len(x) for x in lins), default=0), np.iinfo(np.uint64).max, dtype=np.uint64)
        for x in lins:
            lin[: len(x)] = np.minimum(lin[: len(x)], x)
        out.append(struct.pack("<i", len(lin)))
        out.append(lin.astype("<u8").tobytes())
        cur = r + 1
    out.append(bytes(8 * (n_ref - cur)))
    out.append(struct.pack("<Q", n_no_coor))
    with open(out_path, "wb") as fh:
        fh.write(b"".join(out))


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


# A saved per-read row. What the rows of a part share (reference, first position, read id) is saved once per part,
# apart (part_key), with the part's first row (part_at): 5 bytes per row, where the read id alone took 16.
_ROW_DTYPE = np.dtype([("pos", "<i4"), ("pred", "u1")])


class _ReadRows:
    """Per-read rows for the modBAM, as numpy arrays.

    Rows are grouped by (reference index, read id high, read id low); within a read they stay in the order they were
    added (the order of the saved tables, each sorted by reference and position).
    """

    def __init__(self):
        self._parts = []
        self._keys = []
        self._held = 0
        self._index = {}

    def rows_held(self):
        """Rows added since the last save_by_reference() or finish()."""
        return self._held

    def add(self, ref_ids, read_id, positions, preds_u8):
        """Adds one file's rows: reference index, read id (n x 2 int64), reference position and uint8 probability."""
        self._held += len(ref_ids)
        # the parts of the rows: a part is a run of rows of one (reference, read) (a file holds a read's rows
        # together), kept with its first and last position and its number of rows
        ref, hi, lo = np.asarray(ref_ids, dtype=np.int64), read_id[:, 0], read_id[:, 1]
        pos = np.asarray(positions, dtype=np.int64)
        new = np.ones(len(ref), dtype=bool)
        new[1:] = (ref[1:] != ref[:-1]) | (hi[1:] != hi[:-1]) | (lo[1:] != lo[:-1])
        start = np.flatnonzero(new)
        span = (np.minimum.reduceat(pos, start), np.maximum.reduceat(pos, start)) if len(start) else (pos, pos)
        n = np.diff(np.append(start, len(ref)))
        self._keys.append(
            (ref[new], np.asarray(hi[new], dtype=np.int64), np.asarray(lo[new], dtype=np.int64), *span, n)
        )
        self._parts.append(
            (
                np.asarray(ref_ids, dtype=np.int64),
                np.asarray(read_id[:, 0], dtype=np.int64),
                np.asarray(read_id[:, 1], dtype=np.int64),
                np.asarray(positions, dtype=np.int64),
                np.asarray(preds_u8, dtype=np.uint8),
            )
        )

    def save_by_reference(self, prefix):
        """Appends the added rows, as one run, to the file <prefix>.rows (position and probability), sorted by
        reference and the first position of their part (stably, so the rows of a part stay together and in their
        order), with the reference, first position and read id of each part in that order in <prefix>.part_key and its
        first row in .part_at. The run's parts also go to <prefix>.keys_ref (reference index), .keys (read id high,
        low) and .keys_span (first, last position, rows), sorted by reference, read id and first position, with an
        index of where each reference starts in each (<prefix>.refidx, .keyidx), so that a reader finds a
        reference's rows without searching the large files. Returns the run's name (see _saved), or None, saving
        nothing, when there are no rows.

        A worker appends all its runs to its own few files: a large run has thousands of runs, and a file for each
        would be opened again by every reader (slow on a network file system).

        Sorted by the first position of their parts, the rows of the reads that start in a window of positions lie
        together, so a modBAM writer loads them alone (_write_modbam_windows). A read's tags depend only on its
        (position, probability) pairs (get_mm_tag sorts them, and keeps the order of equal positions), so the order
        of a read's rows does not change the modBAM."""
        if not self._parts:
            return None
        pos, pred = (np.concatenate([part[i] for part in self._parts]) for i in (3, 4))
        k_ref, k_hi, k_lo, k_first, k_last, k_n = (np.concatenate([part[i] for part in self._keys]) for i in range(6))
        row_at, key_at = _saved_len(prefix, "rows"), _saved_len(prefix, "keys_ref")
        ridx_at, kidx_at = _saved_len(prefix, "refidx"), _saved_len(prefix, "keyidx")
        self._parts, self._keys, self._held = [], [], 0
        if len(pos) and (pos.min() < 0 or pos.max() > np.iinfo(np.int32).max):
            raise ValueError("A position does not fit the saved rows (a BAM position is below 2**31).")
        # the parts are runs of rows in the order added: sorting the parts sorts the rows
        order = np.lexsort((k_first, k_ref))  # lexsort is stable
        n = k_n[order]
        at = np.cumsum(n) - n  # the first row of each part, once sorted
        added_at = (np.cumsum(k_n) - k_n)[order]
        row_order = np.repeat(added_at - at, n) + np.arange(len(pos))
        rows = np.empty(len(pos), dtype=_ROW_DTYPE)
        rows["pos"], rows["pred"] = pos[row_order], pred[row_order]
        n_rows = len(rows)
        ref = k_ref[order]
        new_ref = np.flatnonzero(np.concatenate(([True], ref[1:] != ref[:-1])))
        n_ridx = _save_append(prefix, "refidx", np.stack([ref[new_ref], at[new_ref]], axis=1))
        _save_append(prefix, "rows", rows)
        _save_append(prefix, "part_key", np.stack([ref, k_first[order], k_hi[order], k_lo[order]], axis=1))
        _save_append(prefix, "part_at", at)
        del pos, pred, row_order, rows
        order = np.lexsort((k_first, k_lo, k_hi, k_ref))
        n_kidx = _save_append(prefix, "keyidx", _starts_index(k_ref[order]))
        _save_append(prefix, "keys_ref", k_ref[order])
        _save_append(prefix, "keys", np.stack([k_hi[order], k_lo[order]], axis=1))
        _save_append(prefix, "keys_span", np.stack([k_first[order], k_last[order], k_n[order]], axis=1))
        return f"{prefix}#{row_at}:{n_rows}:{key_at}:{len(order)}:{ridx_at}:{n_ridx}:{kidx_at}:{n_kidx}"

    @classmethod
    def load(cls, stems, ref_lo, ref_hi):
        """The rows with reference index in [ref_lo, ref_hi) of the saved tables `stems`, added in the order of
        `stems`, grouped by read."""
        table = cls()
        for stem in stems:
            a, b = _bounds(stem, ref_lo, ref_hi)
            if b > a:
                at = _saved(stem, "part_at")
                i, j = (int(x) for x in np.searchsorted(at, [a, b], side="left"))
                key = np.array(_saved(stem, "part_key")[i:j])
                n = np.diff(np.append(at[i:j], b))  # rows of each part
                rows = _saved(stem, "rows")[a:b]
                table.add(
                    np.repeat(key[:, 0], n),
                    np.repeat(key[:, 2:], n, axis=0),
                    np.array(rows["pos"]),
                    np.array(rows["pred"]),
                )
        return table.finish()

    @classmethod
    def load_window(cls, stems, tid, pos_lo, pos_hi, runs=None, parts=()):
        """The rows of reference `tid` in the saved runs `stems`: those of every part (see save_by_reference) whose
        first position lies in [pos_lo, pos_hi), and those of `parts` [(run, first position, read id high, low)],
        parts that start elsewhere; grouped by read. The parts asked for are read by (run, first position): each such
        block of rows once. `runs`: _RefRuns(stems, tid), which a writer opens once for all the windows of a
        reference."""
        runs = _RefRuns(stems, tid) if runs is None else runs
        extra = {}
        for k, first, hi, lo in parts:
            extra.setdefault(k, {}).setdefault(first, set()).add((hi, lo))
        table = cls()
        for k in range(len(runs)):
            key, at, rows = runs.arrays(k)
            first, n = key[:, 1], np.diff(np.append(at, len(rows)))
            c, d = (int(i) for i in np.searchsorted(first, [pos_lo, pos_hi], side="left"))
            chunks = [np.arange(c, d)]  # parts
            for f, want in extra.get(k, {}).items():
                c, d = (int(i) for i in np.searchsorted(first, [f, f + 1], side="left"))
                keep = [(h, lo) in want for h, lo in zip(key[c:d, 2].tolist(), key[c:d, 3].tolist())]
                chunks.append(c + np.flatnonzero(keep))
            for parts_k in chunks:
                if len(parts_k):
                    # the rows of these parts, one part after the other
                    n_k = n[parts_k]
                    where = np.repeat(at[parts_k] - (np.cumsum(n_k) - n_k), n_k) + np.arange(n_k.sum())
                    part = rows[where[0] : where[-1] + 1] if where[-1] - where[0] + 1 == len(where) else rows[where]
                    table.add(
                        np.full(len(part), tid, dtype=np.int64),
                        np.repeat(key[parts_k, 2:], n_k, axis=0),
                        np.array(part["pos"]),
                        np.array(part["pred"]),
                    )
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
        self._parts, self._keys, self._held = [], [], 0
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

    in_bam = pysam.AlignmentFile(in_path, "rb")
    out_bam = pysam.AlignmentFile(out_path, "wb", template=in_bam)
    # mapped/unmapped come from the index; without one the bar shows no total
    total = (in_bam.mapped or 0) + (in_bam.unmapped or 0) if in_bam.has_index() else None
    if references is not None and in_bam.has_index():
        mapped = _mapped_per_reference(in_bam)
        total = sum(mapped.get(int(tid), 0) for tid in references)
    with tqdm.tqdm(total=total, leave=False, disable=not progress) as bar:  # tags kept where they are, as in 1.2
        _write_modbam_reads(in_bam, out_bam, rows, references, bar, tag_ids=False)
    in_bam.close()
    out_bam.close()
    return None


def _write_modbam_reads(in_bam, out_bam, rows, references, bar, tag_ids=None):
    """Writes to the open `out_bam` every read of the open `in_bam` that has rows in `rows` (a _ReadRows table), with
    MM/ML tags from those rows; `references` as in write_modbam_worker_from_map. `bar` counts the reads read.
    `tag_ids`: _merge_tag_ids of the BAM's header (found here when None)."""
    names = rows.read_names()
    tag_ids = _merge_tag_ids(in_bam.header) if tag_ids is None else tag_ids  # False: the tags as they are
    reads = in_bam
    if references is not None:
        references = [int(tid) for tid in references]
        if in_bam.has_index():
            reads = itertools.chain.from_iterable(in_bam.fetch(tid=tid) for tid in references)
        else:  # no index: scan the whole BAM for these references
            wanted = set(references)
            reads = (read for read in in_bam if read.reference_id in wanted)
    batch = []
    for read in reads:
        bar.update(1)
        payload = _read_payload(read, rows, names)
        if payload is not None:
            batch.append((read, payload))
            if len(batch) >= _MODBAM_BATCH:
                _write_batch(batch, out_bam, tag_ids)
                batch = []
    _write_batch(batch, out_bam, tag_ids)


def _read_payload(read, rows, names, ref_keys=None):
    """(positions, uint8 probabilities) of one read's rows in `rows` (`names`: rows.read_names()), or None for a read
    that is not written (no rows). `ref_keys`, for a table of one window of a reference (_write_modbam_windows): the
    (read id high, read id low) of every read with rows on that reference; such a read without rows in the window gets
    no rows, so it is written with the tags of no row."""
    read_id = read.query_name
    if read_id is None or read.query_sequence is None:
        return None
    hi_lo = names.get(read_id)
    if hi_lo is None:
        if ref_keys is None and _is_canonical_uuid(read_id):
            return None  # a read with no rows in this table
        try:  # other spellings of a UUID (upper case, no hyphens, ...); and, in a window, every read not in it
            read_id_high, read_id_low = np.frombuffer(uuid.UUID(read_id).bytes, dtype=np.int64)
        except (ValueError, AttributeError):
            return None
        hi_lo = (int(read_id_high), int(read_id_low))
    ref_id = read.reference_id
    if ref_id < 0:
        return None

    payload = rows.get((int(ref_id), hi_lo[0], hi_lo[1]))
    if payload is None:
        if ref_keys is None or hi_lo not in ref_keys:
            return None
        payload = (np.empty(0, np.int64), np.empty(0, np.uint8))  # its rows on this reference lie outside the window
    return payload


def _write_modbam_read(read, rows, names, out_bam, ref_keys=None, tag_ids=None):
    """Writes one read to `out_bam` with MM/ML tags from its rows in `rows` (see _read_payload); a read with no rows
    is not written."""
    payload = _read_payload(read, rows, names, ref_keys)
    if payload is not None:
        _write_tagged(read, payload, out_bam, tag_ids)


def _merge_tag_ids(header):
    """{"RG": IDs of the @RG lines, "PG": IDs of the @PG lines} of a BAM header, for _as_merged."""
    ids = {"RG": set(), "PG": set()}
    for line in str(header).splitlines():
        if line.startswith(("@RG\t", "@PG\t")):
            for field in line.split("\t")[1:]:
                if field.startswith("ID:"):
                    ids[line[1:3]].add(field[3:])
                    break
    return ids


def _as_merged(read, tag_ids):
    """Gives a record the tags samtools merge -c -p gave every record of the modBAM up to DeepRM 1.2, when it merged
    the writers' shards (its bam_translate): the RG tag and then the PG tag move to the end of the tags, or are dropped
    when their ID is not in the header. The shards are now joined as they are (samtools cat), so the writers do this,
    and the records stay the same to the byte."""
    for tag in ("RG", "PG"):
        if read.has_tag(tag):
            value = read.get_tag(tag)
            if not isinstance(value, str):
                continue  # not a Z tag: samtools cannot read it as an ID either
            read.set_tag(tag, None)
            if value in tag_ids[tag]:
                read.set_tag(tag, value, "Z")


def _write_tagged(read, payload, out_bam, tag_ids=None):
    """Writes one read with the MM/ML tags of its rows `payload` (positions, uint8 probabilities), one read at a time
    (_mm_ml_batch makes the same tags for many reads at once). `tag_ids`: see _write_modbam_reads."""
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
        read.set_tag("ML", array.array("B", ml_tag))  # uint8 values: B:C, as for the list, set faster
    else:
        read.set_tag("ML", None)  # drop any ML tag the basecaller wrote, so MM and ML always agree
    if tag_ids is not False:
        _as_merged(read, _merge_tag_ids(out_bam.header) if tag_ids is None else tag_ids)
    out_bam.write(read)


def _mm_ml_batch(reads, payloads):
    """The MM text and ML values of each read from its rows, for many reads at once: the tags _write_tagged makes one
    read at a time (_query_positions, the original orientation of a reverse-strand read, get_mm_tag), as numpy
    operations over all the reads and rows together.

    Returns:
        list: per read, (MM text, ML values as a uint8 array), or None for a read _write_tagged must do itself (no
        CIGAR, or an operation _query_positions leaves to get_aligned_pairs).
    """
    out = [None] * len(reads)
    idx, cigars, seqs, starts, rev, pos_list, pred_list = [], [], [], [], [], [], []
    for k, (read, (pos, pred)) in enumerate(zip(reads, payloads)):
        cigar = read.cigartuples
        if not cigar or any(op < 0 or op > 8 or op == 6 for op, _ in cigar):
            continue
        idx.append(k)
        cigars.append(cigar)
        seqs.append(read.query_sequence.encode("latin-1", "replace"))
        starts.append(read.reference_start)
        rev.append(read.is_reverse)
        pos_list.append(pos)
        pred_list.append(pred)
    m = len(idx)
    if m == 0:
        return out
    # CIGAR blocks of all reads: query and reference start of each operation within its read (as _query_positions)
    n_ops = np.fromiter((len(c) for c in cigars), dtype=np.int64, count=m)
    ops = np.fromiter(itertools.chain.from_iterable(itertools.chain.from_iterable(cigars)), dtype=np.int64)
    op, length = ops[0::2], ops[1::2]
    ro = np.repeat(np.arange(m), n_ops)
    first_op = np.cumsum(n_ops) - n_ops
    q_len, r_len = length * _CONSUMES_QUERY[op], length * _CONSUMES_REF[op]
    q_excl, r_excl = np.cumsum(q_len) - q_len, np.cumsum(r_len) - r_len
    q_start = q_excl - q_excl[first_op][ro]
    r_start = np.asarray(starts, dtype=np.int64)[ro] + r_excl - r_excl[first_op][ro]
    aligned = _ALIGNED[op]
    a_read, a_r, a_q, a_len = ro[aligned], r_start[aligned], q_start[aligned], length[aligned]
    # each row's block: the last aligned block of its read that starts at or before the row's position
    n_rows = np.fromiter((len(p) for p in pos_list), dtype=np.int64, count=m)
    pos = np.concatenate(pos_list).astype(np.int64, copy=False)
    pred = np.concatenate(pred_list).astype(np.uint8, copy=False)
    pr = np.repeat(np.arange(m), n_rows)
    lens = np.fromiter((len(x) for x in seqs), dtype=np.int64, count=m)
    is_rev = np.asarray(rev, dtype=bool)
    if a_read.size and pos.size:
        shift = np.int64(1) << 34
        blk = np.searchsorted(a_read * shift + a_r, pr * shift + pos, side="right") - 1
        valid = blk >= 0
        blk = np.where(valid, blk, 0)
        valid &= (a_read[blk] == pr) & (pos < a_r[blk] + a_len[blk])
        q_stored = a_q[blk] + (pos - a_r[blk])  # query position in the stored (reference) orientation
    else:
        valid = np.zeros(len(pos), dtype=bool)
        q_stored = np.zeros(len(pos), dtype=np.int64)
    r, qs, pp = pr[valid], q_stored[valid], pred[valid]
    q_orig = np.where(is_rev[r], lens[r] - 1 - qs, qs)  # in the original (basecalled) orientation
    # get_mm_tag: sorted by position (stably), a position listed twice only when its rows agree, A bases only
    order = np.lexsort((q_orig, r))
    r, qs, pp, q_orig = r[order], qs[order], pp[order], q_orig[order]
    if r.size:
        first = np.ones(r.size, dtype=bool)
        first[1:] = (r[1:] != r[:-1]) | (q_orig[1:] != q_orig[:-1])
        group = np.flatnonzero(first)
        agree = np.minimum.reduceat(pp, group) == np.maximum.reduceat(pp, group)
        keep = np.flatnonzero(first)[agree]
        r, qs, pp, q_orig = r[keep], qs[keep], pp[keep], q_orig[keep]
    # the base in the original orientation: A, read as T on the stored sequence of a reverse-strand read
    seq_all = np.frombuffer(b"".join(seqs), dtype=np.uint8)
    off = np.cumsum(lens) - lens
    target = np.where(is_rev, ord("T"), ord("A")).astype(np.uint8)
    is_base = (seq_all == np.repeat(target, lens)).astype(np.int64)
    inside = (qs >= 0) & (qs < lens[r])
    on_base = np.zeros(r.size, dtype=bool)
    on_base[inside] = is_base[off[r[inside]] + qs[inside]] == 1
    r, qs, pp = r[on_base], qs[on_base], pp[on_base]
    # rank of each kept base among the read's bases (in the original orientation), then the MM skip counts
    before = np.concatenate(([0], np.cumsum(is_base)))
    rank = np.where(is_rev[r], before[off[r] + lens[r]] - before[off[r] + qs + 1], before[off[r] + qs] - before[off[r]])
    new_read = np.ones(r.size, dtype=bool)
    new_read[1:] = r[1:] != r[:-1]
    skip = np.where(new_read, rank, rank - np.concatenate(([0], rank[:-1])) - 1)
    words = list(map(str, skip.tolist()))
    counts = np.bincount(r, minlength=m).tolist()
    at = 0
    for j, n in enumerate(counts):
        if n:
            out[idx[j]] = ("A+a?," + ",".join(words[at : at + n]) + ";", pp[at : at + n])
        else:
            out[idx[j]] = ("A+a?;", pp[:0])
        at += n
    return out


def _write_batch(batch, out_bam, tag_ids=None):
    """Writes reads with the MM/ML tags of their rows, in order; `batch`: (read, payload) pairs (see _read_payload).
    The tags are made for all of them at once (_mm_ml_batch); a read it leaves out is done by _write_tagged.
    `tag_ids`: see _write_modbam_reads."""
    if not batch:
        return
    tag_ids = _merge_tag_ids(out_bam.header) if tag_ids is None else tag_ids  # False: the tags as they are
    reads = [read for read, _ in batch]
    tags = _mm_ml_batch(reads, [payload for _, payload in batch])
    for (read, payload), tag in zip(batch, tags):
        if tag is None:
            _write_tagged(read, payload, out_bam, tag_ids)
            continue
        mm_tag, ml = tag
        read.set_tag("MM", mm_tag, "Z")
        if read.has_tag("MN"):
            # MN is the SEQ length MM and ML refer to; an MN the basecaller wrote before hard clipping no longer fits
            read.set_tag("MN", read.query_length, "i")
        if len(ml) > 0:
            read.set_tag("ML", array.array("B", ml.tobytes()))
        else:
            read.set_tag("ML", None)  # drop any ML tag the basecaller wrote, so MM and ML always agree
        if tag_ids is not False:
            _as_merged(read, tag_ids)
        out_bam.write(read)
