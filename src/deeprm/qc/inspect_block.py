"""
DeepRM QC Module: Inspect Block Files

Inspect training data for quality control: the data folder that deeprm train prep writes, or block files of DeepRM
v1.1's training preparation.
Plot distribution of base quality, motif composition, nucleotide composition, and block score distribution.
"""

import argparse
import glob
import itertools as it
import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from deeprm.utils.logging import get_logger

log = get_logger(__name__)

plt.style.use("default")
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {"font.size": 22, "legend.facecolor": "white", "legend.framealpha": 1, "legend.frameon": 1, "lines.linewidth": 2}
)

_SEED = 0  # seed of every random choice, so that a run can be repeated
_NPZ_KEYS = ("kmer_token", "bq_token", "block_score")  # arrays of a data file that the plots use


def prep_defaults():
    """
    Defaults of deeprm train prep that fit its data.

    Returns:
        tuple: Minimum block score of the default --method, and the context-block length in bases.
    """
    from deeprm.train import train_preprocess

    parser = argparse.ArgumentParser(add_help=False)
    train_preprocess.add_arguments(parser)
    method = parser.get_default("dag_method")
    return train_preprocess.METHOD_DEFAULTS[method]["min_score"], parser.get_default("target_cb_len")


def add_arguments(parser: argparse.ArgumentParser):
    """
    Adds command-line arguments.

    Args:
        parser (argparse.ArgumentParser): Argument parser to which arguments will be added.

    Returns:
        None
    """
    min_score, cb_len = prep_defaults()
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        nargs="+",
        help="Training data, one or more: the data folder that deeprm train prep writes (<output>/data), or a block "
        "file of DeepRM v1.1's training preparation",
    )
    parser.add_argument("--output", "-o", type=str, required=True, help="Output folder; it is made if missing")
    parser.add_argument(
        "--intermediate",
        "-m",
        type=str,
        nargs="+",
        default=None,
        help="Saved block tables (block_df_dict.pkl) to read instead of the inputs, one per --input (default: "
        "qc/block_df_dict.pkl in the folder above a data folder, or two folders above a block file)",
    )
    parser.add_argument(
        "--score",
        "-p",
        type=float,
        default=min_score,
        help="Smallest block score of the blocks in the plots; the block score plot shows all blocks (default "
        f"{min_score}, the default minimum score of deeprm train prep)",
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        nargs="+",
        help="Name of each input in the plots, one per --input (default: for a data folder, the name of the folder "
        "above it, which is the output folder of deeprm train prep; for a block file, its file name)",
    )
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        required=True,
        nargs="+",
        help="Modification of each input, one per --input, for example m6A (for the plot labels and colours)",
    )
    parser.add_argument(
        "--sample",
        "-s",
        type=int,
        default=int(1e6),
        help="Most blocks used from each input (default 1000000); the files of a data folder are read in a random "
        "order with a fixed seed, so a run can be repeated",
    )
    parser.add_argument(
        "--cb-len",
        "-c",
        type=int,
        default=cb_len,
        help=f"Context-block length in bases, to place the base qualities on their positions in the block (default "
        f"{cb_len}, as in deeprm train prep)",
    )
    return None


def main(args: argparse.Namespace):
    """
    Main function to inspect block files.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        None
    """

    if len(args.input) != len(args.type):
        raise ValueError("Give one --type per --input")
    for path in args.input:
        if not os.path.exists(path):
            raise ValueError(f"Input not found: {path}")
    if args.intermediate is not None:
        if len(args.intermediate) != len(args.input):
            raise ValueError("Give one --intermediate per --input")
        for path in args.intermediate:
            if not os.path.exists(path):
                raise ValueError(f"Saved block table not found: {path}")
    else:
        args.intermediate = [default_intermediate(path) for path in args.input]
    if args.name is not None:
        if len(args.name) != len(args.input):
            raise ValueError("Give one --name per --input")
    else:
        args.name = [default_name(path) for path in args.input]
        log.info(f"Names of the inputs: {', '.join(args.name)}")

    warm_color_list = it.cycle(["tomato", "coral", "orange", "gold", "goldenrod", "chocolate"])
    cool_color_list = it.cycle(["royalblue", "dodgerblue", "deepskyblue", "skyblue", "lightblue", "powderblue"])
    modified_name_list = ["m6A", "m1A", "Am", "I", "m5C", "hm5C", "Cm", "m7G", "m1G", "Gm", "m5U", "Um", "pseU"]

    block_names = [f"{name} ({block_type})" for name, block_type in zip(args.name, args.type)]
    if len(set(block_names)) < len(block_names):
        raise ValueError("Two inputs have the same name and --type; give each one a name with --name")
    color_dict = {}
    for block_name, block_type in zip(block_names, args.type):
        if block_type in modified_name_list:
            color = next(warm_color_list)
        else:
            color = next(cool_color_list)
        color_dict[block_name] = color

    os.makedirs(args.output, exist_ok=True)
    saved_path = os.path.join(args.output, "block_df_dict.pkl")
    tables = load_saved_tables([saved_path], block_names) if os.path.exists(saved_path) else None
    output_file_exists = tables is not None
    if tables is None and all(os.path.exists(path) for path in args.intermediate):
        tables = load_saved_tables(args.intermediate, block_names)

    if tables is None:
        block_df_dict = {}
        perfect_block_df_dict = {}
        for path, block_name in zip(args.input, block_names):
            block_df = read_input(path, args.sample)
            perfect_block_df = block_df[block_df["score"] >= args.score]
            n_kept = len(perfect_block_df)
            log.info(f"{block_name}: {len(block_df):,} blocks, {n_kept:,} with a score of {args.score} or more")
            perfect_block_df_dict[block_name] = perfect_block_df.sample(
                min(args.sample, len(perfect_block_df)), random_state=_SEED
            )
            block_df_dict[block_name] = block_df.sample(min(args.sample, len(block_df)), random_state=_SEED)
    else:
        block_df_dict, perfect_block_df_dict = tables

    for block_name, perfect_block_df in perfect_block_df_dict.items():
        if len(perfect_block_df) == 0:
            raise ValueError(f"No block of {block_name} has a score of {args.score} or more; lower --score")
        motif_len = len(perfect_block_df["motif"].iloc[0])
        if motif_len != args.cb_len:
            log.warning(f"The motifs of {block_name} are {motif_len} bases long, but --cb-len is {args.cb_len}")

    if not output_file_exists:
        with open(f"{args.output}/block_df_dict.pkl", "wb") as f:
            pickle.dump(block_df_dict, f)

        with open(f"{args.output}/perfect_block_df_dict.pkl", "wb") as f:
            pickle.dump(perfect_block_df_dict, f)

    nucleotide_composition(perfect_block_df_dict, args.output)
    motif_composition(perfect_block_df_dict, args.output)
    bq_plot(perfect_block_df_dict, color_dict, args.output, cb_len=args.cb_len)
    plot_violin(perfect_block_df_dict, color_dict, args.cb_len, args.output)
    motif_cdf(perfect_block_df_dict, color_dict, args.output)
    block_score_distribution(block_df_dict, color_dict, args.output)
    log.info(f"Plots written to {args.output}")

    return None


def default_name(path):
    """
    Name of an input in the plots.

    Args:
        path (str): Data folder of deeprm train prep, one of its .npz files, or a block file of DeepRM v1.1.

    Returns:
        str: For a data folder or a data file, the name of the output folder of deeprm train prep; for a block file,
        its file name without the extension.
    """
    path = os.path.abspath(path)
    if os.path.isdir(path):
        return os.path.basename(os.path.dirname(path)) or os.path.basename(path)
    if path.endswith(".npz"):
        return os.path.basename(os.path.dirname(os.path.dirname(path))) or os.path.basename(path)
    return os.path.splitext(os.path.basename(path))[0]


def default_intermediate(path):
    """
    Default saved block table of an input.

    Args:
        path (str): Data folder of deeprm train prep, one of its .npz files, or a block file of DeepRM v1.1.

    Returns:
        str: qc/block_df_dict.pkl in the folder above a data folder, or two folders above a file.
    """
    path = os.path.abspath(path)
    if os.path.isdir(path):
        base = os.path.dirname(path)
    else:
        base = os.path.dirname(os.path.dirname(path))
    return os.path.join(base, "qc", "block_df_dict.pkl")


def load_saved_tables(paths, block_names):
    """
    Reads block tables saved by an earlier run: each block_df_dict.pkl and the perfect_block_df_dict.pkl beside it.

    Args:
        paths (list): Paths of the block_df_dict.pkl files.
        block_names (list): Names of the inputs, as "<name> (<type>)".

    Returns:
        tuple: (block_df_dict, perfect_block_df_dict) with one table per name, or None when a file cannot be read or
        does not hold every name.
    """
    block_df_dict = {}
    perfect_block_df_dict = {}
    try:
        for path in paths:
            perfect_path = os.path.join(os.path.dirname(path), f"perfect_{os.path.basename(path)}")
            with open(path, "rb") as f:
                block_df_dict.update(pickle.load(f))
            with open(perfect_path, "rb") as f:
                perfect_block_df_dict.update(pickle.load(f))
    except Exception as exc:
        log.info(f"Could not read the saved block tables in {', '.join(paths)} ({exc}); reading the inputs.")
        return None
    missing = [name for name in block_names if name not in block_df_dict or name not in perfect_block_df_dict]
    if missing:
        log.info(f"The saved block tables in {', '.join(paths)} do not hold {', '.join(missing)}; reading the inputs.")
        return None
    log.info(
        f"Using the block tables saved in {', '.join(paths)} by an earlier run; delete them to read the inputs again."
    )
    block_df_dict = {name: block_df_dict[name] for name in block_names}
    perfect_block_df_dict = {name: perfect_block_df_dict[name] for name in block_names}
    return block_df_dict, perfect_block_df_dict


def read_input(path, sample):
    """
    Reads one input as a table of blocks.

    Args:
        path (str): Data folder of deeprm train prep, one of its .npz files, or a block file of DeepRM v1.1.
        sample (int): Most blocks read from a data folder; a block file is read whole.

    Returns:
        pandas.DataFrame: Blocks, with the columns motif (bases), bq (base quality per position) and score.
    """
    if os.path.isdir(path) or path.endswith(".npz"):
        return read_data_folder(path, sample)
    try:
        block_df = pd.read_pickle(path)
    except Exception as exc:
        raise ValueError(f"Cannot read {path} as a block file: {exc}") from exc
    if not isinstance(block_df, pd.DataFrame):
        raise ValueError(f"{path} does not hold a table of blocks")
    if "score" not in block_df.columns and "block_score" in block_df.columns:
        block_df = block_df.rename(columns={"block_score": "score"})
    missing = [column for column in ("motif", "bq", "score") if column not in block_df.columns]
    if missing:
        raise ValueError(
            f"{path} has no {', '.join(missing)} column. For the output of deeprm train prep, give its data folder "
            "(-i <output>/data)"
        )
    return block_df


def read_data_folder(path, sample):
    """
    Reads the training data that deeprm train prep writes: a folder of .npz files, or one of them.

    The files are read in a random order with a fixed seed, until `sample` blocks are read.

    Args:
        path (str): Data folder, or one .npz file.
        sample (int): Most blocks to read; the files are read whole, so a few more can be read.

    Returns:
        pandas.DataFrame: Blocks, with the columns motif (bases), bq (base quality per position) and score.
    """
    files = sorted(glob.glob(os.path.join(path, "*.npz"))) if os.path.isdir(path) else [path]
    if not files:
        raise ValueError(f"No .npz file in {path}; give the data folder that deeprm train prep writes")
    order = np.random.default_rng(_SEED).permutation(len(files))
    kmer_list, bq_list, score_list = [], [], []
    n_blocks = 0
    for idx in order:
        with np.load(files[idx]) as data:
            missing = [key for key in _NPZ_KEYS if key not in data.files]
            if missing:
                raise ValueError(
                    f"{files[idx]} has no {', '.join(missing)}; it is not a data file of deeprm train prep"
                )
            kmer_list.append(data["kmer_token"])
            bq_list.append(data["bq_token"])
            score_list.append(data["block_score"])
        n_blocks += len(score_list[-1])
        if n_blocks >= sample:
            break
    log.info(f"{path}: {n_blocks:,} blocks read from {len(score_list)} of {len(files)} files")
    return pd.DataFrame(
        {
            "motif": decode_kmer_token(np.concatenate(kmer_list)),
            "bq": list(np.concatenate(bq_list)),
            "score": np.concatenate(score_list),
        }
    )


def decode_kmer_token(kmer_token):
    """
    Bases of the kmer_token array that deeprm train prep writes.

    The array holds one row per block and the ASCII code of each base of the context block (A, C, G, T), with 0
    where a block was padded past a read end.

    Args:
        kmer_token (numpy.ndarray): Array of shape (blocks, context-block length).

    Returns:
        list: One string per block, with N for padding.
    """
    codes = np.asarray(kmer_token).astype(np.uint8)
    codes[codes == 0] = ord("N")
    return [row.tobytes().decode("ascii") for row in codes]


def seq_to_onehot(seq: str):
    """
    Converts a nucleotide sequence to a one-hot encoded matrix.

    Args:
        seq (str): Nucleotide sequence (A, C, G, T/U); another letter, such as N, gets a row of zeros.

    Returns:
        numpy.ndarray: One-hot encoded matrix of the sequence.
    """
    seq = seq.upper()
    seq = seq.replace("T", "U")
    mapping = dict(zip("ACGU", range(4)))
    mapped = [mapping.get(i, 4) for i in seq]
    result = np.eye(5)[mapped][:, :4].astype(float)
    return result


def motif_counts(motifs):
    """
    Counts the bases at each position of motifs of one length.

    Args:
        motifs (list): Motifs (A, C, G, T/U); other letters, such as N, are not counted.

    Returns:
        numpy.ndarray: Counts of A, C, G and U at each position, of shape (motif length, 4).
    """
    lengths = {len(motif) for motif in motifs}
    if len(lengths) != 1:
        raise ValueError(f"The motifs differ in length ({', '.join(map(str, sorted(lengths)))} bases)")
    codes = np.frombuffer("".join(motifs).upper().replace("T", "U").encode("ascii"), dtype=np.uint8)
    codes = codes.reshape(len(motifs), -1)
    return np.stack([(codes == ord(base)).sum(axis=0) for base in "ACGU"], axis=1).astype(float)


def center_kmer(motif, k=5):
    """
    The k-mer at the centre of a motif, which holds the site.

    Args:
        motif (str): Motif of a context block.
        k (int): k-mer length.

    Returns:
        str: The k bases around the centre base.
    """
    start = len(motif) // 2 - k // 2
    return motif[start : start + k]


def bq_positions(length, cb_len=None):
    """
    Positions in the context block of the base qualities of a block.

    deeprm train prep leaves out the k-mer edges, so its base qualities sit in the middle of the block: for example,
    17 values of a 21-nt block sit at positions 2 to 18.

    Args:
        length (int): Number of base qualities per block.
        cb_len (int): Context-block length in bases; None places the first value at position 0.

    Returns:
        numpy.ndarray: Position of each base quality.
    """
    offset = max((cb_len - length) // 2, 0) if cb_len else 0
    return np.arange(length) + offset


def motif_cdf(block_df_dict, color_dict, output):
    """
    Calculate and plot the cumulative distribution function (CDF) of 5-mer motifs in the blocks.

    Args:
        block_df_dict (dict): Dictionary of DataFrames, each containing block data.
        color_dict (dict): Dictionary mapping block names to colors for plotting.
        output (str): Output directory to save the CDF plot and data.

    Returns:
        None
    """
    motif_cdf_dict = {}
    for block_name, block_df in block_df_dict.items():
        motif_arr = block_df["motif"].apply(center_kmer).value_counts()
        motif_cnt_sorted = np.sort(motif_arr.to_numpy())[::-1]
        motif_cdf = np.cumsum(motif_cnt_sorted) / np.sum(motif_cnt_sorted)
        motif_cdf_dict[block_name] = motif_cdf

    with open(f"{output}/motif_cdf.pkl", "wb") as f:
        pickle.dump(motif_cdf_dict, f)

    plt.rcParams.update({"font.size": 24})
    fig, ax = plt.subplots(figsize=(20, 20))
    for block_name, block_df in block_df_dict.items():
        ax.plot(motif_cdf_dict[block_name], label=f"{block_name} (n={len(block_df):,})", color=color_dict[block_name])
    ax.set_title("5-mer Motif CDF")
    ax.set_xlabel("Motif")
    ax.set_ylabel("CDF")
    ax.legend()
    plt.savefig(f"{output}/motif_cdf.png", dpi=300)
    plt.close(fig)
    return None


def motif_composition(block_df_dict, output):
    """
    Plot ratio of nucleotides in each position.
    Each nucleotide is represented as a box, and the height of the box is the ratio of the nucleotide.

    Args:
        block_df_dict (dict): Dictionary of DataFrames, each containing block data.
        output (str): Output directory to save the motif composition plot and data (one table per input, in order).

    Returns:
        None
    """

    motif_sum_list = []
    for block_name, block_df in block_df_dict.items():
        motif_sum = motif_counts(block_df["motif"].tolist())
        total = motif_sum.sum(axis=1, keepdims=True)
        motif_sum = np.divide(motif_sum, total, out=np.zeros_like(motif_sum), where=total > 0)
        motif_sum = pd.DataFrame(motif_sum, columns=["A", "C", "G", "U"])
        motif_sum_list.append(motif_sum)

        fig, ax = plt.subplots(1, 1, figsize=(20, 20))

        motif_sum.plot(kind="bar", stacked=True, ax=ax, color=["royalblue", "tomato", "forestgreen", "gold"])
        ax.set_title(f"Motif composition: ({block_name}) (n={len(block_df):,})")
        ax.set_xlabel("Position")
        ax.set_ylabel("Ratio")
        ax.legend()
        plt.savefig(f"{output}/motif_composition_{block_name.replace(' ', '_')}.png", dpi=300)
        plt.close(fig)

    with open(f"{output}/motif_composition.pickle", "wb") as f:
        pickle.dump(motif_sum_list, f)

    return None


def nucleotide_composition(block_df_dict, output):
    """
    Plot the ratio of nucleotides in each block as a pie chart, leaving out the centre base (the site).

    Args:
        block_df_dict (dict): Dictionary of DataFrames, each containing block data.
        output (str): Output directory to save the nucleotide composition plot.

    Returns:
        None
    """
    ## Plot ratio of nucleotides in pie chart
    nrows = 2 if len(block_df_dict) >= 4 else 1
    ncols = int(np.ceil(len(block_df_dict) / nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 8 * nrows), squeeze=False)
    for i, (block_name, block_df) in enumerate(block_df_dict.items()):
        ax = axes[i // ncols, i % ncols]
        motif_sum = motif_counts(block_df["motif"].tolist())
        motif_sum = np.delete(motif_sum, motif_sum.shape[0] // 2, axis=0).sum(axis=0)
        motif_sum = motif_sum / np.sum(motif_sum)
        ax.pie(motif_sum, labels=["A", "C", "G", "U"], autopct="%1.1f%%")
        ax.set_title(f"{block_name}")
        ax.set_ylabel("")
        ax.set_xlabel("")
    for ax in axes.flat[len(block_df_dict) :]:
        ax.axis("off")

    plt.savefig(f"{output}/nucleotide_composition.png", dpi=300)
    plt.close(fig)

    return None


def bq_plot(block_df_dict, color_dict, output, sample=int(1e4), comment="", cb_len=None):
    """
    Plot the distribution of base quality. Plot position-wise mean with CI95.

    Args:
        block_df_dict (dict): Dictionary of DataFrames, each containing block data.
        color_dict (dict): Dictionary mapping block names to colors for plotting.
        output (str): Output directory to save the base quality plot and data.
        sample (int): Number of samples to use for plotting. If None, use all data.
        comment (str): Comment to append to the output file name.
        cb_len (int): Context-block length, to place the base qualities on their positions in the block.

    Returns:
        None
    """

    stat_dict = {}
    for block_name, block_df in block_df_dict.items():
        if sample is not None:
            block_df = block_df.sample(min(sample, len(block_df)), random_state=_SEED)
        bq_arr = np.stack(block_df["bq"].values, axis=0).astype(float)
        bq_mean = np.mean(bq_arr, axis=0)
        bq_std = np.std(bq_arr, axis=0)
        bq_ci95 = 1.96 * bq_std / np.sqrt(len(bq_arr))
        stat_dict[block_name] = (bq_mean, bq_ci95)

    with open(f"{output}/bq_plot.pickle", "wb") as f:
        pickle.dump(stat_dict, f)

    plt.rcParams.update({"font.size": 24})
    fig, ax = plt.subplots(figsize=(20, 10))
    for block_name, block_df in block_df_dict.items():
        pos_bq_mean, pos_bq_ci95 = stat_dict[block_name]
        position = bq_positions(len(pos_bq_mean), cb_len)
        ax.plot(position, pos_bq_mean, color=color_dict[block_name], label=f"{block_name} (n={len(block_df):,})")
        ax.fill_between(
            position,
            pos_bq_mean - pos_bq_ci95,
            pos_bq_mean + pos_bq_ci95,
            color=color_dict[block_name],
            alpha=0.3,
        )
    ax.set_title("Base Quality Distribution")
    ax.set_xlabel("Position")
    ax.set_ylabel("Mean Base Quality")
    ax.legend()
    plt.savefig(f"{output}/bq_plot{comment}.png", dpi=300)
    plt.close(fig)
    return None


def score_bins(scores):
    """
    Histogram bins for block scores: one bin per score step when the scores take a few even steps (as with
    --method lev), else 50 bins.

    Args:
        scores (numpy.ndarray): Block scores of all inputs.

    Returns:
        numpy.ndarray or int: Bin edges, or the number of bins.
    """
    values = np.unique(scores)
    if len(values) < 2:
        return 1
    step = np.min(np.diff(values))
    if (values[-1] - values[0]) / step <= 100:
        return np.arange(values[0] - step / 2, values[-1] + step, step)
    return np.linspace(values[0], values[-1], 51)


def block_score_distribution(block_df_dict, color_dict, output):
    """Plot the distribution of block score

    Args:
        block_df_dict (dict): Dictionary of DataFrames, each containing block data.
        color_dict (dict): Dictionary mapping block names to colors for plotting.
        output (str): Output directory to save the block score distribution plot.

    Returns:
        None
    """

    score_dict = {}
    for block_name, block_df in block_df_dict.items():
        column = "block_score" if "block_score" in block_df.columns else "score"
        score_dict[block_name] = block_df[column].to_numpy(dtype=float)
    bins = score_bins(np.concatenate(list(score_dict.values())))

    plt.rcParams.update({"font.size": 24})
    fig, ax = plt.subplots(figsize=(20, 20))
    for block_name, scores in score_dict.items():
        sns.histplot(
            scores,
            ax=ax,
            label=f"{block_name} (n={len(scores):,})",
            color=color_dict[block_name],
            linewidth=5,
            element="step",
            stat="density",
            fill=False,
            bins=bins,
        )
    ax.set_title("Block Score Distribution")
    ax.set_xlabel("Block Score")
    ax.set_ylabel("Density")
    ax.legend()
    plt.savefig(f"{output}/block_score_distribution.png", dpi=300)
    plt.close(fig)
    return None


def plot_violin(block_df_dict, color_dict, cb_len, output, sample=int(1e5)):
    """
    Plot the distribution of base quality as a violin plot.

    Args:
        block_df_dict (dict): Dictionary of DataFrames, each containing block data.
        color_dict (dict): Dictionary mapping block names to colors for plotting.
        cb_len (int): Length of the context block.
        output (str): Output directory to save the violin plot.
        sample (int): Most blocks per input in the plot. If None, use all data.

    Returns:
        None
    """
    ## merge df
    df_list = []
    color_list = []
    for name, df in block_df_dict.items():
        if sample is not None:
            df = df.sample(min(sample, len(df)), random_state=_SEED)
        bq_arr = np.stack(df["bq"].values, axis=0).astype(int)
        position = bq_positions(bq_arr.shape[1], cb_len)
        df = pd.DataFrame({"bq": bq_arr.ravel(), "bq_idx": np.tile(position, len(bq_arr))})
        df["name"] = name
        df_list.append(df)
        color_list.append(color_dict[name])

    palette = sns.color_palette(color_list)

    df = pd.concat(df_list).reset_index(drop=True)

    fig, ax = plt.subplots(1, 1, figsize=(30, 10))
    sns.violinplot(
        x="bq_idx",
        y="bq",
        hue="name",
        data=df,
        ax=ax,
        palette=palette,
        linewidth=0.5,
        inner=None,
        hue_order=list(block_df_dict.keys()),
    )
    ax.set_title("Base Quality Distribution")
    ax.set_xlabel("Position")
    ax.set_ylabel("Base Quality")
    ax.legend()
    plt.savefig(f"{output}/bq_violin.png", dpi=300)
    plt.close(fig)

    return None


def plot_motif(perfect_block_df_dict, color_dict, args, motif_list=["AGACU", "CGACA", "UGAUC", "GAAGC", "UCAAG"]):
    """
    Plot the distribution of motifs in the perfect blocks.

    Args:
        perfect_block_df_dict (dict): Dictionary of DataFrames, each containing perfect block data.
        color_dict (dict): Dictionary mapping block names to colors for plotting.
        args: Command-line arguments containing output directory and context block length.
        motif_list (list): List of motifs to plot. Default is a predefined list of motifs.

    Returns:
        None
    """
    for block_name, block_df in perfect_block_df_dict.items():
        block_df = block_df.copy()
        block_df["kmer"] = block_df["motif"].apply(lambda x: center_kmer(x).upper().replace("T", "U"))
        perfect_block_df_dict[block_name] = block_df

    for motif in motif_list:
        motif_block_df_dict = {}
        for block_name, block_df in perfect_block_df_dict.items():
            motif_block_df = block_df[block_df["kmer"] == motif].copy()
            if len(motif_block_df) > 0:
                motif_block_df_dict[block_name] = motif_block_df
        if motif_block_df_dict:
            bq_plot(motif_block_df_dict, color_dict, args.output, sample=None, comment=f"-{motif}", cb_len=args.cb_len)
    plot_violin(perfect_block_df_dict, color_dict, args.cb_len, args.output)

    return None
