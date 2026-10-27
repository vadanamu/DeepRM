"""The fast writers of deeprm call pileup give the same files as the plain ones: the BED lines that pandas writes, NPZ
files with the same arrays as np.savez_compressed, and np.unique's result for the site ids."""

import zipfile

import numpy as np
import pandas as pd
import pytest

from deeprm.inference import pileup_deeprm


def _pandas_bed(path, ref_names, ref_pos, ref_strand, modscore, stoichiometry, count_all, count_pos):
    """The BED writer of DeepRM before the pyarrow writer (pandas only)."""
    estimated_mod_count = np.clip(np.rint(count_all * stoichiometry).astype(np.int64), 0, count_all)
    zeros = np.zeros(len(ref_names), dtype=np.int64)
    df = pd.DataFrame(
        {
            "col1": ref_names,
            "col2": ref_pos,
            "col3": ref_pos + 1,
            "col4": ["a"] * len(ref_names),
            "col5": np.clip(np.rint(modscore * 1000).astype(int), 0, 1000),
            "col6": ref_strand,
            "col7": ref_pos,
            "col8": ref_pos + 1,
            "col9": ["255,0,0"] * len(ref_names),
            "col10": count_all,
            "col11": stoichiometry * 100,
            "col12": estimated_mod_count,
            "col13": count_all - estimated_mod_count,
            **{f"col{i}": zeros for i in range(14, 19)},
        }
    )
    df.to_csv(path, sep="\t", header=False, index=False, float_format="%.2f")


def _sites(n, seed=0, names=("ENST00000001.1", "NM_000002.3", "chr1")):
    rng = np.random.default_rng(seed)
    # stoichiometry values that stress the "%.2f" texts: exact hundredths, ties, -0.0, tiny negatives, NaN, > 1, inf
    special = np.array(
        [0.0, -0.0, 1.0, 0.125, 0.00125, 0.00005, -0.00004, 0.004999999, 0.005, 0.015, 0.105, 0.995, 0.9999999]
        + [np.nan, 1.5, np.inf, -np.inf, 1e-300, 0.5, 1 / 3, 2 / 3, 0.29, 0.57, 0.58]
    )
    stoich = np.concatenate([special, rng.uniform(0, 1, n), np.round(rng.uniform(0, 1, n), 4)])
    m = len(stoich)
    return dict(
        ref_names=rng.choice(np.array(names), m),
        ref_pos=rng.integers(0, 10**8, m),
        ref_strand=rng.choice(np.array(["+", "-"]), m),
        modscore=rng.uniform(0, 1, m),
        stoichiometry=stoich,
        count_all=rng.integers(1, 5000, m),
        count_pos=rng.integers(0, 50, m),
    )


@pytest.mark.parametrize(
    "names",
    [
        ("ENST00000001.1", "NM_000002.3", "chr1"),
        ("tx with space", "tx\twith tab", 'tx"quote'),  # tab and quote: the pandas fallback, quoted by pandas
        ("tx\nnewline", "plain"),
    ],
)
def test_bed_writer_writes_the_pandas_bytes(tmp_path, names):
    sites = _sites(20000, names=names)
    count_all = sites["count_all"]
    # count_all * NaN stoichiometry cannot be rounded to an integer: keep NaN only where count_all is irrelevant
    sites["stoichiometry"] = np.where(np.isfinite(sites["stoichiometry"]), sites["stoichiometry"], 0.0)
    pileup_deeprm.bed_formatter(**sites, output_path=str(tmp_path / "fast.bed"))
    _pandas_bed(str(tmp_path / "pandas.bed"), **sites)
    assert (tmp_path / "fast.bed").read_bytes() == (tmp_path / "pandas.bed").read_bytes()
    assert count_all.size > 0


def test_two_decimals_matches_percent_format_everywhere():
    sites = _sites(50000, seed=3)
    x = sites["stoichiometry"] * 100
    texts = pileup_deeprm._two_decimals(x)
    words = np.array(texts.dictionary.to_pylist(), dtype=object)[texts.indices.to_numpy()]
    expected = ["" if v != v else "%.2f" % v for v in x.tolist()]  # noqa: UP031 -- pandas' float_format
    assert words.tolist() == expected
    assert "-0.00" in expected, "the values must include a negative zero"


def test_bed_writer_empty_table(tmp_path):
    empty = {k: v[:0] for k, v in _sites(1).items()}
    pileup_deeprm.bed_formatter(**empty, output_path=str(tmp_path / "fast.bed"))
    _pandas_bed(str(tmp_path / "pandas.bed"), **empty)
    assert (tmp_path / "fast.bed").read_bytes() == (tmp_path / "pandas.bed").read_bytes() == b""


def test_npz_writer_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    arrays = {
        "ref_names": np.array(["ENST1", "NM_2", "chr10"] * 10),
        "ref_pos": rng.integers(0, 10**9, 30),
        "modscore": rng.uniform(0, 1, 30),
        "stoich_state": rng.integers(-1, 2, 30).astype(np.int8),
        "stoichiometry_release": np.where(rng.random(30) < 0.3, np.nan, rng.uniform(0, 1, 30)),
        "empty": np.empty(0, dtype=np.float64),
    }
    pileup_deeprm._savez(str(tmp_path / "fast.npz"), **arrays)
    np.savez_compressed(tmp_path / "numpy.npz", **arrays)
    fast, ref = np.load(tmp_path / "fast.npz", allow_pickle=False), np.load(tmp_path / "numpy.npz")
    assert fast.files == ref.files == list(arrays)
    for key in arrays:
        assert fast[key].dtype == ref[key].dtype and fast[key].tobytes() == ref[key].tobytes(), key
    with zipfile.ZipFile(tmp_path / "fast.npz") as zf:
        assert zf.testzip() is None
        assert [i.filename for i in zf.infolist()] == [k + ".npy" for k in arrays]
    with pytest.raises(ValueError):  # an object array is refused (the file must load without allow_pickle)
        pileup_deeprm._savez(str(tmp_path / "obj.npz"), names=np.array(["a", None], dtype=object))


@pytest.mark.parametrize("n", [0, 1, 2, 1000])
def test_sorted_unique_matches_np_unique(n):
    rng = np.random.default_rng(n)
    x = rng.integers(-50, 50, n).astype(np.int64)
    got = pileup_deeprm._sorted_unique(x)
    assert got.dtype == np.int64 and np.array_equal(got, np.unique(x))
