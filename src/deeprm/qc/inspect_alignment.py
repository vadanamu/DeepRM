"""
DeepRM QC Module: Inspect Alignment

Inspect alignment quality by extracting CIGAR string and calculating error rates.
This module reads a BAM file, extracts the CIGAR strings, and computes the error rates for each read.
"""

import argparse
import multiprocessing as mp
import os
import re
from collections import deque

import numpy as np
import pandas as pd
import pysam
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from deeprm.utils.logging import get_logger
from deeprm.utils.utils import mean_phred

log = get_logger(__name__)

plt.style.use("default")
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {"font.size": 22, "legend.facecolor": "white", "legend.framealpha": 1, "legend.frameon": 1, "lines.linewidth": 2}
)


def add_arguments(parser: argparse.ArgumentParser):
    """
    Adds command-line arguments.

    Args:
        parser (argparse.ArgumentParser): Argument parser to which arguments will be added.

    Returns:
        None
    """
    parser.add_argument("--input", "-i", type=str, dest="input", help="BAM file of aligned reads", required=True)
    parser.add_argument("--output", "-o", type=str, dest="output", help="Output folder", required=True)
    parser.add_argument(
        "--process",
        "-p",
        type=int,
        dest="process",
        help="Number of worker processes (default: about a quarter of the CPUs)",
        default=int(mp.cpu_count() * 0.95 // 4),
    )
    parser.add_argument(
        "--thread", "-t", type=int, dest="thread", help="BAM decompression threads per process", default=4
    )
    parser.add_argument(
        "--mapq", "-m", type=int, dest="mapq", help="Skip reads with a mapping quality below this", default=30
    )
    parser.add_argument(
        "--bq", "-b", type=int, dest="bq", help="Skip reads whose mean base quality is below this", default=7
    )
    parser.add_argument(
        "--min-len",
        "-l",
        type=int,
        dest="len_cutoff",
        help="Skip reads shorter than this (0: keep all reads)",
        default=0,
    )

    return None


def main(args: argparse.Namespace):
    """
    Main function to run the alignment inspection pipeline.
    This function parses command line arguments, checks for existing output,
    and runs the CIGAR extraction and error rate calculation.
    It also plots the error rates using KDE and boxplot.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """

    run_flag = True

    if os.path.exists(args.output):
        log.info("Output directory already exists. Attempting to load pickle")
        try:
            error_df = pd.read_pickle(f"{args.output}/error_rate.pkl")
            run_flag = False
        except Exception:
            log.warning("Error rate pickle does not exist. Re-running alignment and error rate calculation")
            run_flag = True

    if run_flag:
        os.makedirs(args.output, exist_ok=True)

        log.info("Extracting CIGAR string")
        error_df = extract_cigar_master(args)
        error_df.to_pickle(f"{args.output}/error_rate.pkl")

    assert len(error_df) > 0, "Error rate dataframe is empty. Check input BAM file"

    log.info("Plotting error rate")
    plot_kde(error_df, args)
    plot_boxplot(error_df, args)

    return None


def extract_cigar_worker(pid, args, error_dict):
    """
    Worker function to extract CIGAR strings and calculate error rates for a given process ID.

    Args:
        pid (int): Process ID for multiprocessing.
        args (argparse.Namespace): Parsed command line arguments.
        error_dict (dict): Shared dictionary to store error rates.

    Returns:
        None
    """
    bamfile = pysam.AlignmentFile(args.input, "rb", check_sq=False, threads=args.thread)
    total = bamfile.mapped + bamfile.unmapped
    error_rates = deque(maxlen=total // args.process + 1)
    for i, read in tqdm(enumerate(bamfile), total=total):
        if i % args.process == pid:
            if args.len_cutoff > 0:
                if read.query_length < args.len_cutoff:
                    continue
            if read.is_unmapped or read.is_secondary:
                continue
            if mean_phred(np.array(read.query_qualities, dtype=int)) < args.bq:
                continue
            if read.mapping_quality < args.mapq:
                continue
            if not read.has_tag("MD"):
                continue
            error_rates.append(get_error_rate_func(read.cigarstring, read.get_tag("MD"), use_md=True))
    bamfile.close()
    error_dict[pid] = np.stack(error_rates, axis=0)
    return None


def extract_cigar_master(args):
    """
    Master function to extract CIGAR strings and calculate error rates using multiprocessing.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
    Returns:
        pandas.DataFrame: DataFrame containing error rates for each read.
    """
    manager = mp.Manager()
    error_dict = manager.dict()
    proc_list = []
    for pid in range(args.process):
        proc = mp.Process(target=extract_cigar_worker, args=(pid, args, error_dict))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()
    error_df = np.concatenate(list(error_dict.values()), axis=0)
    error_df = pd.DataFrame(error_df, columns=["MIS_RATE", "INS_RATE", "DEL_RATE"])
    error_df["ERROR_RATE"] = error_df["MIS_RATE"] + error_df["INS_RATE"] + error_df["DEL_RATE"]
    return error_df


def md_to_mismatch_arr(md):
    """
    Convert MD tag to mismatch array.
    1 = mismatch, 0 = match. Deletions are ignored (filled as matches).

    Args:
        md (str): MD tag string from the BAM file.

    Returns:
        numpy.ndarray: Array of mismatches (1s) and matches (0s).
    """
    pattern = re.compile(r"(\d+)|(\^[A-Z]+)|([A-Z])")
    result = []
    for match in pattern.finditer(md):
        if match.group(1):  # Match run
            result.append(np.zeros(int(match.group(1)), dtype=int))
        elif match.group(2):  # Deletion
            # Deletions are not counted as mismatches, fill with 0s for length of deletion
            result.append(np.zeros(len(match.group(2)[1:]), dtype=int))
        elif match.group(3):  # Mismatch
            result.append(np.ones(1, dtype=int))
    return np.concatenate(result, axis=0)


def get_error_rate_func(cigar, md, use_md=True):
    """
    Calculate error rates from CIGAR string and MD tag.

    Args:
        cigar (str): CIGAR string from the BAM file.
        md (str): MD tag string from the BAM file.
        use_md (bool): Whether to use MD tag for mismatch calculation. Default is True.

    Returns:
        numpy.ndarray: Array containing mismatch rate, insertion rate, and deletion rate.
    """
    cigar_list = re.findall(r"(\d+)([A-Z,=])", cigar)
    mismatch = 0
    insertion = 0
    deletion = 0
    ref_length = 0

    for length, match in cigar_list:
        if match in ["=", "M"]:
            ref_length += int(length)
        elif match == "X":
            mismatch += int(length)
            ref_length += int(length)
        elif match == "I":
            insertion += int(length)
        elif match == "D":
            deletion += int(length)
            ref_length += int(length)

    mis_rate = mismatch / ref_length
    ins_rate = insertion / ref_length
    del_rate = deletion / ref_length

    if use_md:
        mis_arr = md_to_mismatch_arr(md)
        mis_rate = np.sum(mis_arr) / ref_length

    error_rates = np.array((mis_rate, ins_rate, del_rate))
    return error_rates


def plot_kde(df_error, args):
    """
    Plot the distribution of read alignment accuracy using KDE.

    Args:
        df_error (pandas.DataFrame): DataFrame containing error rates for each read.
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        None
    """
    plt.rcParams.update({"font.size": 26})
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_xlabel("Read Alignment accuracy")
    ax.set_ylabel("Density")
    ax.set_xlim(0.8, 1.0)
    sns.histplot(
        1 - df_error["ERROR_RATE"],
        ax=ax,
        color="royalblue",
        label=f"Pass (n={len(df_error):,})",
        binwidth=0.001,
        binrange=(0.7, 1.0),
        kde=False,
        stat="density",
    )
    ## vline at median
    median = np.median(1 - df_error["ERROR_RATE"])
    ax.axvline(median, color="royalblue", linestyle="--", linewidth=3)
    ax.text(median, 0.9 * ax.get_ylim()[1], f"{median:.3f}", color="royalblue")
    ax.legend()
    plt.savefig(f"{args.output}/per_read_phred_error_kde.png", dpi=300)
    plt.close()
    return None


def plot_boxplot(df_error, args):
    """
    Plot a boxplot of the error rates for each read.

    Args:
        df_error (pandas.DataFrame): DataFrame containing error rates for each read.
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        None
    """
    ## Plot mis, ins, del rate
    plt.rcParams.update({"font.size": 26})
    fig, ax = plt.subplots(figsize=(20, 20))
    ## Use whiskers, no fliers
    sns.boxplot(data=df_error[["MIS_RATE", "INS_RATE", "DEL_RATE"]], ax=ax, palette="Set2", linewidth=3, fliersize=0)
    ax.set_xlabel("Error type")
    ax.set_ylabel("Error rate")
    ax.set_ylim(0, 0.03)
    ax.legend()
    plt.savefig(f"{args.output}/per_read_error_boxplot.png", dpi=300)
    plt.close()
    return None
