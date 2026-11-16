"""Parity of the C++ pileup with the Python pileup: on the same input and settings, deeprm call pileup with the C++
binary writes the outputs of the Python pileup, fails where the Python pileup fails, and leaves nothing behind.

Skipped where no pileup binary can start (none bundled, or this CPU lacks AVX2 and FMA), or where NumPy computes the
per-read terms differently from it (without AVX512): there deeprm call pileup runs the Python pileup. DEEPRM_PILEUP_BIN
names a binary to check instead of the bundled one."""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
from test_pileup_genomic import REFFLAT, _write_refflat
from test_pileup_memory import _bam_and_rows, _outputs, _random_seq
from test_pileup_modbam_merge import LABEL_DIV, _args

from deeprm.inference import pileup, pileup_deeprm
from deeprm.utils import prep_binary


@pytest.fixture(scope="module")
def binary():
    path = prep_binary.named_path(os.environ["DEEPRM_PILEUP_BIN"]) if os.environ.get("DEEPRM_PILEUP_BIN") else None
    path = path or pileup.DEFAULT_BIN
    if prep_binary.on_noexec_mount(path):
        pytest.skip(f"{path} is on a file system where programs cannot run")
    st = prep_binary.status(path, program=prep_binary.PILEUP)
    if not st.ok:
        pytest.skip(f"no pileup binary can start here: {st.reason}")
    # The binary gives the Python pileup's results only where NumPy computes the per-read terms as it does (with
    # AVX512); elsewhere deeprm call pileup runs the Python pileup, and there is no parity to check.
    why = pileup._numpy_differs(path, 0.98, 1e-30)
    if why:
        pytest.skip(f"NumPy here does not compute {why} as the pileup binary does (for example without AVX512)")
    return path


def _input(tmp_path, n_refs=4, reads_per_ref=40, n_files=5, seed=3):
    rng = np.random.default_rng(seed)
    reads = [
        (t, int(rng.integers(0, 600)), [(0, 120)], _random_seq(rng, 120), 0, [])
        for t in range(n_refs)
        for _ in range(reads_per_ref)
    ]
    return _bam_and_rows(tmp_path, [(f"tx{i}", 1000) for i in range(n_refs)], reads, n_files=n_files, seed=seed)


def _edit_files(inp, fn):
    """Rewrites every per-read file of `inp` with fn(name, {array: values}) -> {array: values}."""
    for name in sorted(os.listdir(inp)):
        if name.endswith(".npz"):
            with np.load(inp / name) as z:
                arrays = {k: z[k] for k in z.files}
            np.savez(inp / name, **fn(name, arrays))


def _leftovers(*folders):
    return [os.path.join(str(f), n) for f in folders if os.path.isdir(f) for n in os.listdir(f) if n.startswith(".")]


def _both(tmp_path, binary, args, **settings):
    """Runs the Python pileup and deeprm call pileup with the binary on the same input and settings; returns both
    output folders."""
    out = {}
    for name in ("python", "binary"):
        a = _args(args.input, tmp_path / name, args.bam, thread=args.thread)
        for key, value in vars(args).items():
            setattr(a, key, value)
        for key, value in settings.items():
            setattr(a, key, value)
        a.output = str(tmp_path / name)
        if a.tmp_dir:
            a.tmp_dir = str(tmp_path / f"tmp_{name}")
        if name == "python":
            pileup_deeprm.main(a)
        else:
            a.pileup_bin = binary
            pileup.main(a)
        out[name] = tmp_path / name
    assert _leftovers(out["python"], out["binary"], tmp_path / "tmp_python", tmp_path / "tmp_binary") == []
    return out["python"], out["binary"]


def _same(tmp_path, binary, args, **settings):
    python, cpp = _both(tmp_path, binary, args, **settings)
    expected, got = _outputs(python), _outputs(cpp)
    assert sorted(got) == sorted(expected)
    for name in expected:
        assert got[name] == expected[name], name


def _args_for(inp, bam, thread=3):
    a = _args(inp, None, bam, thread)
    a.tmp_dir, a.report_all_stoichiometry = None, False
    return a


@pytest.mark.parametrize("thread", [1, 2, 3, 7])
def test_same_outputs_at_each_thread_count(tmp_path, binary, thread):
    inp, bam = _input(tmp_path)
    _same(tmp_path, binary, _args_for(inp, bam, thread))


@pytest.mark.parametrize(
    "settings",
    [
        {"skip_modbam": True},
        {"flip": True},
        {"report_all_stoichiometry": True},
        {"report_all_stoichiometry": True, "skip_modbam": True},
        {"epsilon": 1e-12},
        {"tmp_dir": "set"},
    ],
    ids=["skip-modbam", "flip", "report-all", "report-all-skip-modbam", "epsilon", "tmp-dir"],
)
def test_same_outputs_for_each_setting(tmp_path, binary, settings):
    inp, bam = _input(tmp_path)
    _same(tmp_path, binary, _args_for(inp, bam), **settings)


@pytest.mark.parametrize("run_info", ["bundled weights", "other model", "none"])
def test_same_calibration_decision(tmp_path, binary, run_info):
    """Both read deeprm_run.json the same way: the bundled weights' fixed calibration, none for another model file,
    none without the file."""
    inp, bam = _input(tmp_path)
    if run_info == "bundled weights":
        (inp / "deeprm_run.json").write_text(json.dumps({"model_name": "deeprm_v12", "site_rule_calibration": True}))
    elif run_info == "other model":
        (inp / "deeprm_run.json").write_text(json.dumps({"model_name": "deeprm_v12", "site_rule_calibration": False}))
    _same(tmp_path, binary, _args_for(inp, bam))


def test_same_outputs_with_the_annotation_and_a_postfix(tmp_path, binary):
    """The genomic tables (--annot) under their postfix, on the references of the annotation test."""
    rng = np.random.default_rng(5)
    sq = [("tx1", 5000), ("tx2", 5000), ("tx3", 5000)]
    reads = [(t, int(rng.integers(0, 25)), [(0, 30)], _random_seq(rng, 30), 0, []) for t in range(3) for _ in range(30)]
    inp, bam = _bam_and_rows(tmp_path, sq, reads, n_files=4)
    annot = tmp_path / "refFlat.txt"
    _write_refflat(annot, REFFLAT)
    args = _args_for(inp, bam)
    args.annot, args.postfix = str(annot), "_x"
    _same(tmp_path, binary, args)


def test_same_outputs_for_a_column_of_2d_predictions(tmp_path, binary):
    inp, bam = _input(tmp_path)
    rng = np.random.default_rng(1)
    _edit_files(
        inp,
        lambda name, a: dict(
            a, pred=np.stack([1 - a["pred"], a["pred"], rng.random(len(a["pred"]))], 1).astype(np.float32)
        ),
    )
    args = _args_for(inp, bam)
    args.slice = 1
    _same(tmp_path, binary, args)


def test_same_outputs_for_another_label_factor(tmp_path, binary):
    inp, bam = _input(tmp_path)

    def repack(name, a):
        ref, pos = np.divmod(np.abs(a["label_id"]) - 1, LABEL_DIV)
        return dict(a, label_id=np.sign(a["label_id"]) * (ref * 10**6 + pos + 1))

    _edit_files(inp, repack)
    args = _args_for(inp, bam)
    args.label_div = 10**6
    _same(tmp_path, binary, args, skip_modbam=False)


def test_same_outputs_with_sites_on_the_opposite_strand(tmp_path, binary):
    """Rows of sites on the strand opposite their transcript (negative label ids) beside those on it."""
    inp, bam = _input(tmp_path)
    _edit_files(
        inp, lambda name, a: dict(a, label_id=np.where(np.arange(len(a["label_id"])) % 3 == 0, -1, 1) * a["label_id"])
    )
    _same(tmp_path, binary, _args_for(inp, bam), skip_modbam=True)


def test_same_outputs_with_nan_rows_and_an_empty_file(tmp_path, binary):
    inp, bam = _input(tmp_path)

    def damage(name, a):
        pred = a["pred"].copy()
        pred[::17] = np.nan
        return dict(a, pred=pred)

    _edit_files(inp, damage)
    np.savez(
        inp / "inference_9_9.npz",
        label_id=np.zeros(0, np.int64),
        read_id=np.zeros((0, 2), np.int64),
        pred=np.zeros(0, np.float32),
    )
    _same(tmp_path, binary, _args_for(inp, bam))


def test_same_modbam_when_no_read_of_the_bam_has_predictions(tmp_path, binary):
    """Per-read files of other reads (another BAM's): both write a modBAM with the header and no record."""
    inp, bam = _input(tmp_path)
    rng = np.random.default_rng(9)
    _edit_files(inp, lambda name, a: dict(a, read_id=rng.integers(1, 2**62, size=a["read_id"].shape, dtype=np.int64)))
    _same(tmp_path, binary, _args_for(inp, bam))


def test_the_check_against_numpy_passes_here_and_fails_without_avx512(tmp_path, binary):
    """The binary computes as NumPy does with AVX512: the check passes in this process, and fails in a process where
    NumPy may not use AVX512 (as on a CPU without it), so that the Python pileup runs there."""
    assert pileup._numpy_differs(binary, 0.98, 1e-30) is None
    code = (
        "import numpy\n"
        "core = getattr(numpy, '_core', None) or numpy.core\n"
        "on = core._multiarray_umath.__cpu_features__\n"
        "from deeprm.inference import pileup\n"
        "print('AVX512 STILL ON' if on.get('AVX512_SKX') or on.get('X86_V4') else "
        f"repr(pileup._numpy_differs({binary!r}, 0.98, 1e-30)))"
    )
    # The names differ between NumPy versions (X86_V4 since 2.3); a name a version does not know is ignored
    off = "AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR X86_V4"
    done = subprocess.run(
        [sys.executable, "-c", code], env=dict(os.environ, NPY_DISABLE_CPU_FEATURES=off), capture_output=True, text=True
    )
    if done.returncode != 0 or done.stdout.strip() == "AVX512 STILL ON":
        pytest.skip(f"NumPy here does not turn AVX512 off: {(done.stdout + done.stderr).strip()[-200:]}")
    assert done.stdout.strip() != "None"


def _fails(tmp_path, binary, args, **settings):
    """Both paths stop with an error and write no output file and no temporary file."""
    for name in ("python", "binary"):
        a = _args_for(args.input, args.bam)
        for key, value in {**vars(args), **settings}.items():
            setattr(a, key, value)
        a.output = str(tmp_path / name)
        with pytest.raises((ValueError, RuntimeError)):  # a Python worker's error comes as a RuntimeError
            if name == "python":
                pileup_deeprm.main(a)
            else:
                a.pileup_bin = binary
                pileup.main(a)
        left = os.listdir(tmp_path / name) if os.path.isdir(tmp_path / name) else []
        assert left == [], (name, left)


def test_both_fail_without_per_read_files(tmp_path, binary):
    inp, bam = _input(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    _fails(tmp_path, binary, _args_for(empty, bam))


def test_both_fail_on_a_reference_past_the_bam(tmp_path, binary):
    inp, bam = _input(tmp_path)
    _edit_files(inp, lambda name, a: dict(a, label_id=a["label_id"] + 9 * LABEL_DIV))
    _fails(tmp_path, binary, _args_for(inp, bam), skip_modbam=True)


def test_both_fail_on_a_column_the_predictions_lack(tmp_path, binary):
    inp, bam = _input(tmp_path)
    _fails(tmp_path, binary, _args_for(inp, bam), slice=3)


def test_both_fail_without_a_valid_row(tmp_path, binary):
    inp, bam = _input(tmp_path)
    _edit_files(inp, lambda name, a: dict(a, pred=np.full(len(a["pred"]), np.nan, np.float32)))
    _fails(tmp_path, binary, _args_for(inp, bam), skip_modbam=True)


def test_the_binary_is_used(tmp_path, binary, caplog):
    """A check of the check: the runs above went through the binary, not the Python fallback."""
    import logging

    inp, bam = _input(tmp_path)
    a = _args_for(inp, bam)
    a.output, a.pileup_bin = str(tmp_path / "out"), binary
    with caplog.at_level(logging.INFO):
        pileup.main(a)
    assert "Using the C++ pileup binary" in caplog.text
    shutil.rmtree(tmp_path / "out")
