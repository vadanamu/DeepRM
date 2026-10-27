"""deeprm qc block: reads the training data of deeprm train prep (its data folder of .npz files) and DeepRM v1.1 block
files, and writes every plot and table. CPU only."""

import argparse
import os
import pickle

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from deeprm.qc import inspect_block
from deeprm.train import train_preprocess

CB_LEN = 21  # context-block length of deeprm train prep (default --target-cb-len)
TRIM = 2  # bases left out at each end of the base qualities (half the default k-mer length 5)


def _parse(argv):
    """Parses `deeprm qc block` arguments."""
    parser = argparse.ArgumentParser()
    inspect_block.add_arguments(parser)
    return parser.parse_args(argv)


def _expected_files(block_names):
    """Files that `deeprm qc block` writes for these inputs."""
    files = [
        "block_df_dict.pkl",
        "perfect_block_df_dict.pkl",
        "nucleotide_composition.png",
        "motif_composition.pickle",
        "bq_plot.pickle",
        "bq_plot.png",
        "bq_violin.png",
        "motif_cdf.pkl",
        "motif_cdf.png",
        "block_score_distribution.png",
    ]
    return files + [f"motif_composition_{name.replace(' ', '_')}.png" for name in block_names]


def _assert_written(output, block_names):
    for name in _expected_files(block_names):
        path = os.path.join(output, name)
        assert os.path.isfile(path), name
        assert os.path.getsize(path) > 0, name


def _write_data_folder(folder, seed, n_files=3, n_rows=40):
    """Writes a data folder as deeprm train prep does: its own save_npz, the read sequence stored as ASCII codes and
    each block cut from it with slice_scalar_token."""
    rng = np.random.default_rng(seed)
    os.makedirs(folder)
    pad = CB_LEN // 2
    for i in range(n_files):
        data = {key: [] for key in ("score", "segment_len_arr", "signal_token", "kmer_token", "dwell_motor_token")}
        data.update({key: [] for key in ("dwell_pore_token", "bq_token", "read_id", "pos_RM", "cb_idx")})
        data.update({key: [] for key in ("anchor_idx", "di_gap", "tl", "tr")})
        for row in range(n_rows):
            read = "".join(rng.choice(list("ACGT"), size=60))
            pos = 30
            read = read[:pos] + "A" + read[pos + 1 :]
            seq = np.frombuffer(read.encode("ascii"), dtype=np.int8)
            bq = rng.integers(3, 51, size=len(read)).astype(np.uint8)
            data["kmer_token"].append(
                train_preprocess.slice_scalar_token(seq, pos - pad, pos + pad + 1, 0, len(seq), "none", pad + 2)
            )
            bq_token = train_preprocess.slice_scalar_token(
                bq, pos - pad + TRIM, pos + pad + 1 - TRIM, 0, len(seq), "none", pad + 2 - TRIM
            )
            data["bq_token"].append(np.asarray(bq_token, dtype=np.int8))
            data["score"].append(float(rng.choice([0.5, 0.75, 1.0])))
            data["segment_len_arr"].append(rng.integers(1, 20, CB_LEN - 2 * TRIM).astype(np.uint16))
            data["signal_token"].append(rng.normal(size=1224).astype(np.float32))
            data["dwell_motor_token"].append(rng.normal(size=CB_LEN - 2 * TRIM).astype(np.float32))
            data["dwell_pore_token"].append(rng.normal(size=CB_LEN - 2 * TRIM).astype(np.float32))
            data["read_id"].append(rng.bytes(16))
            data["pos_RM"].append(pos)
            data["cb_idx"].append(row % 3)
            data["anchor_idx"].append(0)
            data["di_gap"].append(-1)
            data["tl"].append(0)
            data["tr"].append(len(read))
        train_preprocess.save_npz(os.path.join(folder, f"0-{i}.npz"), data)


@pytest.fixture(autouse=True)
def _small_png(monkeypatch):
    """Writes the plots at a low resolution, to keep the tests fast."""
    savefig = plt.savefig
    monkeypatch.setattr(plt, "savefig", lambda path, **kwargs: savefig(path, **{**kwargs, "dpi": 20}))


def test_decode_kmer_token_matches_prep_format():
    read = "GGCATTACGGACTCAAGGTACCTGAAGTCAGCTAGGAC"
    seq = np.frombuffer(read.encode("ascii"), dtype=np.int8)  # as deeprm train prep stores a read
    pad = CB_LEN // 2
    pos = 17
    token = train_preprocess.slice_scalar_token(seq, pos - pad, pos + pad + 1, 0, len(seq), "none", pad + 2)
    assert inspect_block.decode_kmer_token(np.stack([token])) == ["CGGACTCAAGGTACCTGAAGT"] == [read[7:28]]
    # --padding-mode zero pads past the read start with 0, shown as N
    token = train_preprocess.slice_scalar_token(seq, -3, CB_LEN - 3, 0, len(seq), "zero", pad + 2)
    assert inspect_block.decode_kmer_token(np.stack([token])) == ["NNN" + read[: CB_LEN - 3]]


def test_defaults_follow_train_prep():
    args = _parse(["-i", "x", "-o", "y", "-t", "m6A"])
    prep = argparse.ArgumentParser()
    train_preprocess.add_arguments(prep)
    assert args.score == train_preprocess.METHOD_DEFAULTS[prep.get_default("dag_method")]["min_score"]
    assert args.cb_len == prep.get_default("target_cb_len") == CB_LEN


def test_qc_block_reads_train_prep_data(tmp_path):
    _write_data_folder(str(tmp_path / "prep" / "a" / "data"), seed=1)
    _write_data_folder(str(tmp_path / "prep" / "m6a" / "data"), seed=2)
    output = str(tmp_path / "qc" / "new")  # made by the command
    args = _parse(
        ["-i", str(tmp_path / "prep" / "a" / "data"), str(tmp_path / "prep" / "m6a" / "data"), "-t", "A", "m6A"]
        + ["-o", output]
    )
    inspect_block.main(args)
    _assert_written(output, ["a (A)", "m6a (m6A)"])

    with open(os.path.join(output, "perfect_block_df_dict.pkl"), "rb") as f:
        tables = pickle.load(f)
    assert list(tables) == ["a (A)", "m6a (m6A)"]
    for table in tables.values():
        assert len(table) == 120
        assert {len(motif) for motif in table["motif"]} == {CB_LEN}
        assert {motif[CB_LEN // 2] for motif in table["motif"]} == {"A"}
        assert {len(bq) for bq in table["bq"]} == {CB_LEN - 2 * TRIM}
    with open(os.path.join(output, "bq_plot.pickle"), "rb") as f:
        assert len(pickle.load(f)["m6a (m6A)"][0]) == CB_LEN - 2 * TRIM
    with open(os.path.join(output, "motif_composition.pickle"), "rb") as f:
        composition = pickle.load(f)
    assert len(composition) == 2 and np.allclose(composition[1].sum(axis=1), 1.0)


def test_qc_block_score_cut_and_sample(tmp_path):
    folder = str(tmp_path / "m6a" / "data")
    _write_data_folder(folder, seed=3, n_files=4)
    first = inspect_block.read_data_folder(folder, sample=50)
    second = inspect_block.read_data_folder(folder, sample=50)
    assert len(first) == 80  # whole files are read until 50 blocks are in
    assert first["motif"].tolist() == second["motif"].tolist()

    output = str(tmp_path / "qc")
    inspect_block.main(_parse(["-i", folder, "-t", "m6A", "-o", output, "-p", "1", "-s", "50"]))
    with open(os.path.join(output, "perfect_block_df_dict.pkl"), "rb") as f:
        perfect = pickle.load(f)["m6a (m6A)"]
    with open(os.path.join(output, "block_df_dict.pkl"), "rb") as f:
        blocks = pickle.load(f)["m6a (m6A)"]
    assert len(blocks) == 50 and 0 < len(perfect) < 50 and perfect["score"].min() == 1.0


def test_qc_block_reads_v11_block_file(tmp_path):
    rng = np.random.default_rng(4)
    motifs = []
    for _ in range(60):
        motif = "".join(rng.choice(list("ACGU"), size=CB_LEN))
        motifs.append(motif[: CB_LEN // 2] + "A" + motif[CB_LEN // 2 + 1 :])
    block_df = pd.DataFrame(
        {
            "motif": motifs,
            "bq": [list(rng.integers(3, 51, size=CB_LEN)) for _ in motifs],
            "score": rng.choice([0.9, 1.0], size=len(motifs)),
        }
    )
    os.makedirs(tmp_path / "run" / "blocks")
    path = str(tmp_path / "run" / "blocks" / "m6a_block.pkl")
    block_df.to_pickle(path)
    output = str(tmp_path / "qc")
    inspect_block.main(_parse(["-i", path, "-t", "m6A", "-o", output, "-c", str(CB_LEN)]))
    _assert_written(output, ["m6a_block (m6A)"])
    with open(os.path.join(output, "bq_plot.pickle"), "rb") as f:
        assert len(pickle.load(f)["m6a_block (m6A)"][0]) == CB_LEN


def test_qc_block_refuses_v12_block_pickle(tmp_path):
    path = str(tmp_path / "0.pkl")
    pd.DataFrame(
        {"read_id": [b"x" * 16], "pos_RM": [30], "score": [1.0], "cb_idx": [0], "tr": [60], "tl": [0]}
    ).to_pickle(path)
    with pytest.raises(ValueError, match="data folder"):
        inspect_block.main(_parse(["-i", path, "-t", "m6A", "-o", str(tmp_path / "qc")]))
    with pytest.raises(ValueError, match="not found"):
        inspect_block.main(_parse(["-i", str(tmp_path / "missing"), "-t", "m6A", "-o", str(tmp_path / "qc")]))
