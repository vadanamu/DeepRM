"""deeprm call pileup refuses a missing BAM before any work, and writes its BED files with pandas when pyarrow does
not import."""

import argparse
import pathlib
import sys

import numpy as np
import pysam
import pytest

from deeprm.inference import inference, pileup_deeprm

WEIGHTS = pathlib.Path(inference.__file__).parent.parent / "weight" / "deeprm_weights.pt"


def _pileup_args(inp, out, bam):
    return argparse.Namespace(
        input=str(inp),
        output=str(out),
        bam=str(bam),
        thread=1,
        threshold=0.98,
        epsilon=1e-30,
        postfix="",
        slice=None,
        flip=False,
        label_div=10**9,
        annot=None,
        skip_modbam=True,
    )


@pytest.mark.parametrize("what", ["missing", "folder"])
def test_pileup_refuses_a_bam_that_is_not_a_file(tmp_path, what):
    inp = tmp_path / "molecule-level"
    inp.mkdir()
    bam = tmp_path / "typo.bam"
    if what == "folder":
        bam.mkdir()
    with pytest.raises(ValueError, match="--bam"):
        pileup_deeprm.main(_pileup_args(inp, tmp_path / "out", bam))
    assert not (tmp_path / "out").exists()


def test_call_run_refuses_a_missing_bam_before_the_inference(monkeypatch, tmp_path):
    (tmp_path / "prep").mkdir()
    ran = []
    monkeypatch.setattr(inference, "run_inference", lambda args: ran.append("run"))
    args = argparse.Namespace(
        data=str(tmp_path / "prep"),
        bam=str(tmp_path / "typo.bam"),
        output=str(tmp_path / "out"),
        model=str(WEIGHTS),
        model_type=None,
        engine="auto",
        batch=20,
        num_gpu=0,
        gpu_pool=None,
        prefetch=1,
        worker=0,
        flush=100,
        resume=False,
        output_id=None,
        postfix="",
        threshold=0.98,
        epsilon=1e-30,
        thread=None,
        label_div=10**9,
        slice=None,
        flip=False,
        skip_modbam=True,
        annot=None,
    )
    with pytest.raises(ValueError, match="--bam"):
        inference.main(args)
    assert ran == [] and not (tmp_path / "out").exists()


def _sites(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return dict(
        ref_names=rng.choice(np.array(["ENST00000001.1", "chr1"]), n),
        ref_pos=rng.integers(0, 10**8, n),
        ref_strand=rng.choice(np.array(["+", "-"]), n),
        modscore=rng.uniform(0, 1, n),
        stoichiometry=np.r_[0.0, 0.125, 0.005, rng.uniform(0, 1, n - 3)],
        count_all=rng.integers(1, 5000, n),
        count_pos=rng.integers(0, 50, n),
    )


def test_bed_is_written_with_pandas_when_pyarrow_does_not_import(tmp_path, monkeypatch):
    """A pyarrow that cannot be imported (for example one built for NumPy 1.x) must not stop the pileup."""
    sites = _sites()
    pileup_deeprm.bed_formatter(**sites, output_path=str(tmp_path / "with_pyarrow.bed"))
    # None in sys.modules makes `import pyarrow` raise ImportError
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.csv", None)
    with pytest.raises(ImportError):
        import pyarrow  # noqa: F401
    pileup_deeprm.bed_formatter(**sites, output_path=str(tmp_path / "without_pyarrow.bed"))
    assert (tmp_path / "without_pyarrow.bed").read_bytes() == (tmp_path / "with_pyarrow.bed").read_bytes()


def test_pileup_runs_without_pyarrow(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    inp = tmp_path / "molecule-level"
    inp.mkdir()
    np.savez(
        inp / "inference_0_0.npz",
        label_id=np.repeat(np.arange(1, 11, dtype=np.int64), 30),
        read_id=rng.integers(0, 2**62, size=(300, 2), dtype=np.int64),
        pred=rng.uniform(0, 1, 300).astype(np.float32),
    )
    bam = tmp_path / "x.bam"
    header = pysam.AlignmentHeader.from_dict({"HD": {"VN": "1.6"}, "SQ": [{"SN": "tx1", "LN": 5000}]})
    with pysam.AlignmentFile(str(bam), "wb", header=header):
        pass
    pysam.index(str(bam))
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.csv", None)
    pileup_deeprm.main(_pileup_args(inp, tmp_path / "out", bam))
    assert len((tmp_path / "out" / "pileup.bed").read_text().splitlines()) == 10


def test_pyarrow_floor_imports_with_numpy_2():
    """pyarrow 14 installs next to NumPy 2 but does not import with it, and pyarrow 15 requires NumPy 1; 16 is the first
    release built for NumPy 2."""
    import re

    pyproject = pathlib.Path(__file__).parent.parent / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("not run from a source tree")
    floors = [int(v) for v in re.findall(r'"pyarrow>=(\d+)', pyproject.read_text())]
    assert len(floors) == 1 and floors[0] >= 16
