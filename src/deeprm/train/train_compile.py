"""
DeepRM Training Dataset Compilation Module

This module compiles training data from positive and negative token files into a structured format.
This script reads NPZ files containing tokenized data, samples it based on specified criteria,
and saves it in a structured directory format.
"""

import argparse
import gc
import glob
import multiprocessing as mp
import os

import numpy as np
import tqdm

from deeprm.utils.logging import get_logger

log = get_logger(__name__)


def add_arguments(parser: argparse.ArgumentParser):
    """
    Adds command-line arguments.

    Args:
        parser (argparse.ArgumentParser): Argument parser to which arguments will be added.
    Returns:
        None
    """

    parser.add_argument(
        "--pos",
        "-p",
        dest="pos_path",
        type=str,
        default=None,
        nargs="+",
        help="Folders of modified (positive) training data: the data folders that deeprm train prep writes",
    )
    parser.add_argument(
        "--neg",
        "-n",
        dest="neg_path",
        type=str,
        default=None,
        nargs="+",
        help="Folders of unmodified (negative) training data: the data folders that deeprm train prep writes",
    )
    parser.add_argument(
        "--output", "-o", dest="out_path", type=str, required=True, help="Output folder; it must not exist yet"
    )
    parser.add_argument(
        "--thread",
        "-t",
        dest="cpu",
        type=int,
        default=int(os.cpu_count() * 0.9),
        help="Number of worker processes (default: 90%% of the CPUs)",
    )
    parser.add_argument("--chunk", "-c", dest="chunk", type=int, default=4000, help="Rows per output file")
    parser.add_argument(
        "--score",
        "-s",
        dest="score",
        type=float,
        default=1.0,
        help="Keep the blocks with a block score of at least this",
    )
    parser.add_argument(
        "--val-frac", "-v", type=float, default=0.05, help="Fraction of the blocks put in val/; the rest go to train/"
    )
    return None


def main(args: argparse.Namespace):
    """
    Main function to run the data compilation process.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """

    if os.path.exists(args.out_path):
        raise ValueError(f"{args.out_path} already exists. Use a new --output folder, or remove that one first.")
    for in_path in (args.pos_path or []) + (args.neg_path or []):
        if not _token_files(in_path):
            raise ValueError(
                f"{in_path} holds no token files (*.npz). Give the token folder of deeprm train prep, "
                "<prep output>/data."
            )
    os.makedirs(args.out_path)

    if args.pos_path is None:
        log.warning("No positive token files specified")
    else:
        for pos_path in args.pos_path:
            if not os.path.exists(pos_path):
                log.error(f"Positive token file {pos_path} does not exist")
                raise FileNotFoundError(f"Positive token file {pos_path} does not exist")

    if args.neg_path is None:
        log.warning("No negative token files specified")
    else:
        for neg_path in args.neg_path:
            if not os.path.exists(neg_path):
                log.error(f"Negative token file {neg_path} does not exist")
                raise FileNotFoundError(f"Negative token file {neg_path} does not exist")

    os.makedirs(args.out_path, exist_ok=True)

    for set_name in ["train", "val"]:
        for label in ["pos", "neg"]:
            os.makedirs(f"{args.out_path}/{set_name}/{label}", exist_ok=True)

    assert 0.0 <= args.val_frac < 1.0, "Validation fraction must be in [0.0, 1.0)"
    set_split_dict = {"train": 1.0 - args.val_frac, "val": args.val_frac}

    if args.pos_path is not None:
        sample_and_save(
            args.pos_path,
            args.out_path,
            args.cpu,
            label=1,
            chunk=args.chunk,
            set_split_dict=set_split_dict,
            score=args.score,
        )

    if args.neg_path is not None:
        sample_and_save(
            args.neg_path,
            args.out_path,
            args.cpu,
            label=0,
            chunk=args.chunk,
            set_split_dict=set_split_dict,
            score=args.score,
        )

    # deeprm train run needs every split it reads to hold files; small inputs can leave one empty
    labels = [name for name, given in (("pos", args.pos_path), ("neg", args.neg_path)) if given is not None]
    empty = [
        f"{set_name}/{label}"
        for set_name, fraction in set_split_dict.items()
        if fraction > 0
        for label in labels
        if not glob.glob(f"{args.out_path}/{set_name}/{label}/*.npz")
    ]
    if empty:
        raise ValueError(
            f"No files were written for {', '.join(empty)}: the input has too few rows of block score >= {args.score} "
            f"for chunks of {args.chunk} rows. Remove {args.out_path}, then run again with a smaller --chunk (for "
            "example -c 1000), or with more input data."
        )
    return None


def _token_files(in_path):
    """The token files of a prep output folder; a temporary file left by an interrupted write is not one."""
    return sorted(p for p in glob.glob(os.path.join(in_path, "*.npz")) if ".tmp." not in os.path.basename(p))


def sample_and_save(
    in_path_list,
    out_path,
    ncpu,
    label,
    chunk,
    label_dict={0: "neg", 1: "pos"},
    set_split_dict={"train": 0.95, "val": 0.05},
    score=1.0,
    id_digit=9,
    shuffle=True,
    read_once=100,
):
    """
    Samples data from input files and saves it to the output directory.

    Args:
        in_path_list (list): List of input file paths.
        out_path (str): Output directory path.
        ncpu (int): Number of CPUs to use.
        label (int): Label for the data (0 for negative, 1 for positive).
        chunk (int): Chunk size for saving data.
        label_dict (dict): Dictionary mapping labels to strings.
        set_split_dict (dict): Dictionary defining the split ratios for train and validation sets.
        score (float): Score threshold.
        id_digit (int): Number of digits for file IDs.
        shuffle (bool): Whether to shuffle the data.
        read_once (int): Number of files to read at once.

    Returns:
        None
    """
    in_file_list = [x for in_path in in_path_list for x in _token_files(in_path)]
    column_keys = [
        "segment_len_arr",
        "signal_token",
        "kmer_token",
        "dwell_motor_token",
        "dwell_pore_token",
        "bq_token",
        "block_score",
    ]
    if shuffle:
        in_file_list = np.random.permutation(in_file_list)
    in_file_list = np.array_split(in_file_list, ncpu)
    proc_list = []
    man = mp.Manager()
    remainder_dict = man.dict()
    label_str = label_dict[label]

    for set_name in set_split_dict:
        remainder_dict[set_name] = man.list()

    for pid in range(ncpu):
        proc = mp.Process(
            target=sample_and_save_worker,
            args=(
                ncpu,
                pid,
                in_file_list[pid],
                out_path,
                label_str,
                set_split_dict,
                score,
                chunk,
                label,
                remainder_dict,
                id_digit,
                shuffle,
                read_once,
                column_keys,
            ),
        )
        proc_list.append(proc)
        proc.start()

    for proc in proc_list:
        proc.join()
    failed = [proc.exitcode for proc in proc_list if proc.exitcode != 0]
    if failed:
        raise RuntimeError(f"{len(failed)} compile worker(s) failed (exit codes {failed}); the output is incomplete.")

    pid = ncpu
    file_id = [-1]

    for key in remainder_dict:
        set_name = key
        remainder_data_list = remainder_dict[key]
        if len(remainder_data_list) > 0:
            remainder_data = {}
            for column_key in column_keys:
                remainder_data[column_key] = np.concatenate([x[column_key] for x in remainder_data_list])
            buffer_dict = None
            chunk_save_data(
                ncpu,
                pid,
                file_id,
                remainder_data,
                column_keys,
                out_path,
                label_str,
                set_name,
                buffer_dict,
                chunk,
                id_digit,
            )
            del remainder_data
            gc.collect()
    del remainder_dict
    gc.collect()

    return None


def pad_signal(signal, max_len):
    """
    Pads the signal to the maximum length with zeros.

    Args:
        signal (numpy.ndarray): Input signal array.
        max_len (int): Maximum length to pad to.

    Returns:
        numpy.ndarray: Padded signal array.
    """
    return np.concatenate([signal, np.zeros(max_len - len(signal), dtype=np.float32)])


def sample_and_save_worker(
    ncpu,
    pid,
    in_file_list,
    out_path,
    label_str,
    set_split_dict,
    score,
    chunk,
    label,
    remainder_dict,
    id_digit,
    shuffle,
    read_once,
    column_keys,
):
    """
    Worker function to sample and save data.

    Args:
        ncpu (int): Number of CPUs to use.
        pid (int): Process ID.
        in_file_list (list): List of input file paths.
        out_path (str): Output directory path.
        label_str (str): Label string for the data.
        set_split_dict (dict): Dictionary defining the split ratios for train and validation sets.
        score (float): Score threshold.
        chunk (int): Chunk size for saving data.
        label (int): Label for the data (0 for negative, 1 for positive).
        remainder_dict (dict): Dictionary to store remainder data.
        id_digit (int): Number of digits for file IDs.
        shuffle (bool): Whether to shuffle the data.
        read_once (int): Number of files to read at once.
        column_keys (list): List of column keys for the data.

    Returns:
        None
    """
    file_id = [-1]
    data_buffer = {x: [] for x in column_keys}
    buffer_dict = {key: None for key in remainder_dict.keys()}

    for df_idx, df_path in tqdm.tqdm(enumerate(in_file_list), desc=f"Saving {label_str} data", total=len(in_file_list)):
        with np.load(df_path) as data:
            for key in column_keys:
                data_buffer[key].append(data[key])

        if len(data_buffer[column_keys[0]]) < read_once and df_idx < len(in_file_list) - 1:
            continue

        else:
            data = {key: np.concatenate(data_buffer[key]) for key in column_keys}

            if shuffle:
                idx = np.random.permutation(len(data[column_keys[0]]))
                for key in column_keys:
                    data[key] = data[key][idx]

            data_buffer = {x: [] for x in column_keys}
            gc.collect()

            bool_idx = data["block_score"] >= score
            score_data = {key: data[key][bool_idx] for key in column_keys}
            save_split_data(
                ncpu,
                pid,
                file_id,
                score_data,
                column_keys,
                out_path,
                label_str,
                set_split_dict,
                chunk,
                id_digit,
                buffer_dict,
            )

            del data
            gc.collect()

    for key in buffer_dict:
        if buffer_dict[key] is not None:
            remainder_dict[key].append(buffer_dict[key])
    del buffer_dict
    gc.collect()

    return None


def save_split_data(
    ncpu,
    pid,
    file_id,
    data,
    column_keys,
    out_path,
    label_str,
    set_split_dict,
    chunk,
    id_digit,
    buffer_dict,
):
    """
    Saves split data to the output directory.

    Args:
        ncpu (int): Number of CPUs to use.
        pid (int): Process ID.
        file_id (list): List containing the file ID.
        data (dict): Dictionary containing the data to save.
        column_keys (list): List of column keys for the data.
        out_path (str): Output directory path.
        label_str (str): Label string for the data.
        set_split_dict (dict): Dictionary defining the split ratios for train and validation sets.
        chunk (int): Chunk size for saving data.
        id_digit (int): Number of digits for file IDs.
        buffer_dict (dict): Dictionary to store buffer data.

    Returns:
        None
    """
    set_idx = np.cumsum(
        [0] + [int(np.floor(len(data[column_keys[0]]) * split_ratio)) for split_ratio in set_split_dict.values()]
    )

    for idx, set_name in enumerate(set_split_dict):
        set_idx_start = set_idx[idx]
        set_idx_end = set_idx[idx + 1]
        set_data = {key: data[key][set_idx_start:set_idx_end] for key in column_keys}
        buffer = buffer_dict[set_name]

        if buffer is not None:
            set_data = {key: np.concatenate([buffer[key], set_data[key]]) for key in column_keys}
            buffer_dict[set_name] = None
            gc.collect()

        chunk_save_data(
            ncpu,
            pid,
            file_id,
            set_data,
            column_keys,
            out_path,
            label_str,
            set_name,
            buffer_dict,
            chunk,
            id_digit,
        )

        del set_data
        gc.collect()

    return None


def chunk_save_data(
    ncpu,
    pid,
    file_id,
    set_data,
    column_keys,
    out_path,
    label_str,
    set_name,
    buffer_dict,
    chunk,
    id_digit,
):
    """
    Saves data in chunks to the output directory.

    Args:
        ncpu (int): Number of CPUs to use.
        pid (int): Process ID.
        file_id (list): List containing the file ID.
        set_data (dict): Dictionary containing the data to save.
        column_keys (list): List of column keys for the data.
        out_path (str): Output directory path.
        label_str (str): Label string for the data.
        set_name (str): Set name (train or val).
        buffer_dict (dict): Dictionary to store buffer data.
        chunk (int): Chunk size for saving data.
        id_digit (int): Number of digits for file IDs.

    Returns:
        None
    """
    len_set_data = len(set_data[column_keys[0]])
    for chunk_idx in range(0, len_set_data // chunk + 1):
        file_id[0] += 1
        out_data_id = (ncpu + 1) * file_id[0] + pid
        chunk_data = {
            key: val[chunk_idx * chunk : min((chunk_idx + 1) * chunk, len_set_data)] for key, val in set_data.items()
        }

        if len(chunk_data[column_keys[0]]) == chunk:
            save_path = f"{out_path}/{set_name}/{label_str}/{str(out_data_id).zfill(id_digit)}.npz"
            if os.path.exists(save_path):
                log.warning(f"File {save_path} already exists - overwriting.")
            chunk_data.pop("block_score")
            np.savez_compressed(save_path, **chunk_data)
            del chunk_data
            gc.collect()

        elif buffer_dict is not None:
            buffer = buffer_dict[set_name]
            if buffer is not None:
                buffer_dict[set_name] = {key: np.concatenate([buffer[key], chunk_data[key]]) for key in column_keys}
            else:
                buffer_dict[set_name] = chunk_data.copy()

    return None
