"""Tests for the v1.2 site stoichiometry rule (the rule block of deeprm.inference.pileup_deeprm) and its use in
the pileup."""

import argparse
import hashlib
import json
import logging
import math
import os
import sys

import numpy as np
import pysam
import pytest
from scipy import stats

from deeprm.inference import pileup_deeprm
from deeprm.inference.pileup_deeprm import get_mm_tag
from deeprm.inference.pileup_deeprm import main as pileup_main
from deeprm.utils.utils import BUNDLED_WEIGHTS_MD5

LOG10_CUT = np.log10(1 - 0.98)


def _reference_b3v(s, n, k, e_cal):
    """The rule written out site by site from its definition, for comparison: the terms in their divided form, in Python
    floats. Returns (state, value with NaN where not reported)."""
    z = stats.norm.ppf(0.99)
    background = 504 / 100000
    state = np.full(len(s), -1, np.int8)
    value = np.full(len(s), np.nan)
    for i, (si, ni, ki, ei) in enumerate(zip(s.tolist(), n.tolist(), k.tolist(), e_cal.tolist())):
        h = l_term = -math.inf
        if si > 0:
            h = ei / ni - min(si, 0.5)
        if ki >= 1:
            q = ki / ni
            upper = (q + z * z / (2 * ni) + z * math.sqrt(q * (1 - q) / ni + z * z / (4 * ni * ni))) / (1 + z * z / ni)
            l_term = background - upper
        if max(h, l_term) > 0:
            state[i] = 1
            value[i] = si if si > 0 else ki / ni
    return state, value


def _random_sites(n_sites=4000, seed=0):
    """Site tables with both rule terms in play: shallow and deep sites, s > 0 and s = 0, k = 0 and k >= 1."""
    rng = np.random.default_rng(seed)
    n = rng.choice([1, 3, 6, 10, 21, 22, 30, 41, 60, 300, 1437, 1438, 2000, 5000], size=n_sites).astype(np.int64)
    k = np.minimum(rng.binomial(n, rng.choice([0.0, 0.0005, 0.002, 0.01, 0.3, 0.8], size=n_sites)), n)
    s12 = np.where(rng.random(n_sites) < 0.7, rng.uniform(0.0, 1.0, n_sites), 0.0)
    s12[rng.random(n_sites) < 0.05] = 1.0
    logsum = -k * rng.uniform(1.7, 4.0, n_sites)
    logsum_cal = logsum * rng.uniform(0.3, 1.0, n_sites)
    return s12, n, k, logsum, logsum_cal


def _sums(n, k, logsum, logsum_cal):
    """Per-site sums with the base sums at the per-read cut 0.98, so the rule's own sums (at 0.98) are the same."""
    return {
        "count_all": n,
        "count_pos": k,
        "logsum_1_p_pos": logsum,
        "count_pos_rule": k,
        "logsum_1_p_pos_rule": logsum,
        "logsum_1_p_pos_calib": logsum_cal,
    }


def test_b3v_matches_reference_rule():
    s12, n, k, logsum, logsum_cal = _random_sites()
    reported, state, release, eq14 = pileup_deeprm._site_stoichiometry(_sums(n, k, logsum, logsum_cal), s12, 0.98)
    s_ref = s12 * ((LOG10_CUT * s12) > (logsum / n))
    assert np.array_equal(eq14, s_ref)
    state_ref, value_ref = _reference_b3v(s_ref, n.astype(float), k, logsum_cal / LOG10_CUT)
    assert np.array_equal(state, state_ref)
    assert np.array_equal(release, value_ref, equal_nan=True)
    at_s = (state == 1) & (release == eq14) & (eq14 > 0)
    at_kn = (state == 1) & ~at_s
    assert np.array_equal(release[at_kn], (k / n)[at_kn])
    assert at_s.any() and at_kn.any() and (state == -1).any(), "the table must exercise both states and both values"


@pytest.mark.parametrize(
    "s, n, k, e_cal, state, value",
    [
        (0.0, 1437, 1, 0.0, -1, None),  # l: k = 1 needs n >= 1438
        (0.0, 1438, 1, 0.0, 1, 1 / 1438),
        (0.0, 1773, 2, 0.0, -1, None),  # k = 2 needs n >= 1774
        (0.0, 1774, 2, 0.0, 1, 2 / 1774),
        (0.0, 5000, 0, 0.0, -1, None),  # no confident read and no stoichiometry: no term
        (0.8, 10, 6, 5.1, 1, 0.8),  # h: E_cal > n * min(s, 1/2) = 5
        (0.8, 10, 6, 4.9, -1, None),
        (0.2, 10, 2, 2.1, 1, 0.2),  # below one half, s itself: n * s = 2
        (0.2, 10, 2, 1.9, -1, None),
        (0.001, 5000, 1, 0.0, 1, 0.001),  # reported by l, but s > 0 gives the value
    ],
)
def test_b3v_term_boundaries(s, n, k, e_cal, state, value):
    got_state, got_value = pileup_deeprm._apply_b3v(np.array([s]), np.array([n]), np.array([k]), np.array([e_cal]))
    assert got_state[0] == state
    if value is None:
        assert np.isnan(got_value[0])
    else:
        assert got_value[0] == value


def test_site_stoichiometry_outputs():
    s12, n, k, logsum, logsum_cal = _random_sites(seed=1)
    reported, state, release, eq14 = pileup_deeprm._site_stoichiometry(_sums(n, k, logsum, logsum_cal), s12, 0.98)
    assert state.dtype == np.int8 and set(np.unique(state).tolist()) == {-1, 1}
    assert np.isnan(release[state == -1]).all() and (reported[state == -1] == 0).all()
    assert np.array_equal(reported[state == 1], release[state == 1]) and (reported[state == 1] > 0).all()
    # one value formula: the Eq. 14 value where it is above 0, else the confident-read fraction k / n
    value, kn, s = reported[state == 1], (k / n)[state == 1], eq14[state == 1]
    assert np.array_equal(value, np.where(s > 0, s, kn))


def test_site_rule_keeps_its_cut_whatever_the_threshold():
    """The rule's constants were measured at the per-read cut 0.98, so its states and values use its own sums at 0.98;
    only stoichiometry_eq14 (the v1.1 value) follows the per-read cut."""
    s12, n, k, logsum, logsum_cal = _random_sites(seed=2)
    sums = _sums(n, k, logsum, logsum_cal)
    at_098 = pileup_deeprm._site_stoichiometry(sums, s12, 0.98)
    for threshold in (0.5, 0.9, 0.999):
        reported, state, release, eq14 = pileup_deeprm._site_stoichiometry(sums, s12, threshold)
        assert np.array_equal(state, at_098[1]) and np.array_equal(reported, at_098[0])
        assert np.array_equal(release, at_098[2], equal_nan=True)
        assert np.array_equal(eq14, s12 * ((np.log10(1 - threshold) * s12) > (logsum / n)))
    assert not np.array_equal(pileup_deeprm._site_stoichiometry(sums, s12, 0.9)[3], at_098[3])


def test_calibrated_probability():
    # Each check is a boolean with a fixed message: on a failure pytest then prints no array derived from the
    # calibration.
    logits = np.array([-5.0, 0.0, 3.0, 8.0], dtype=np.float32)
    p = pileup_deeprm._sigmoid_np(logits)
    assert np.array_equal(pileup_deeprm._calibrated_probability(p, calibrated=False), p)
    pc = pileup_deeprm._calibrated_probability(p, calibrated=True)
    matches_definition = bool(np.allclose(pc, 1 / (1 + np.exp(-logits / pileup_deeprm._CALIBRATION)), atol=1e-5))
    assert matches_definition, "the calibrated probability is not sigmoid(logit / calibration)"
    toward_half = bool((np.abs(pc - 0.5) <= np.abs(p - 0.5) + 1e-7).all())
    assert toward_half, "the calibration moves probabilities toward 0.5"


def test_the_calibration_is_pinned():
    """The calibration is a fixed part of the v1.2 model, so an edit must fail a test. Compared by digest, as a
    boolean, so that pytest never prints the value."""
    digest = hashlib.sha256(repr(pileup_deeprm._CALIBRATION).encode()).hexdigest()
    pinned = digest == "5170c9e8991dc750dc477d528f345d2c75b324363c19733e2568fa86e12ee0ba"
    assert pinned, "the fixed calibration of the v1.2 model changed"


def test_empty_mm_tag_is_valid():
    assert get_mm_tag([], [], "ACGT") == ("A+a?;", [])
    assert get_mm_tag([1], [200], "CCCC") == ("A+a?;", [])


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


class _LogLines(logging.Handler):
    """Collects the messages of the pileup's logger (it writes to standard error and does not propagate)."""

    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture
def pileup_log():
    handler = _LogLines()
    pileup_deeprm.log.addHandler(handler)
    yield handler.lines
    pileup_deeprm.log.removeHandler(handler)


def _write_bam(path):
    header = pysam.AlignmentHeader.from_dict({"HD": {"VN": "1.6"}, "SQ": [{"SN": "tx1", "LN": 5000}]})
    with pysam.AlignmentFile(str(path), "wb", header=header):
        pass
    pysam.index(str(path))


def test_pileup_writes_two_state_stoichiometry(tmp_path, capsys, pileup_log):
    rng = np.random.default_rng(3)
    # (position, reads, fraction of confident modified reads)
    plan = [(100, 30, 0.6), (200, 600, 0.0), (300, 2000, 0.0), (400, 8, 0.5), (500, 25, 0.9)]
    label, logit = [], []
    for pos, depth, frac in plan:
        mod = rng.permutation(np.arange(depth) < round(frac * depth))
        site_logit = np.where(mod, rng.uniform(6, 9, depth), rng.uniform(-9, -5, depth))
        if pos == 300:  # one confident read and 300 weak ones among 2,000: Eq. 14 fails, so k / n is reported
            site_logit[0] = 8.0
            site_logit[1:301] = 1.0
        logit.append(site_logit)
        label.append(np.full(depth, pos + 1, dtype=np.int64))
    # a site whose state depends on the calibration: 40 reads just above the cut, and below it once calibrated
    logit.append(np.full(40, 3.95))
    label.append(np.full(40, 600 + 1, dtype=np.int64))
    p600 = pileup_deeprm._sigmoid_np(np.full(40, 3.95, dtype=np.float32))
    above_cut = bool((p600 >= 0.98).all() and (pileup_deeprm._calibrated_probability(p600) < 0.98).all())
    assert above_cut, "site 600 must be at or above the cut before the calibration and below it after"
    logit = np.concatenate(logit).astype(np.float32)
    label = np.concatenate(label)
    pred = pileup_deeprm._sigmoid_np(logit)
    inp = tmp_path / "molecule-level"
    inp.mkdir()
    np.savez(
        inp / "inference_0_0.npz",
        label_id=label,
        read_id=rng.integers(0, 2**62, size=(len(label), 2), dtype=np.int64),
        pred=pred,
    )
    with open(inp / "deeprm_run.json", "w") as f:
        json.dump({"model_name": "deeprm_v12", "site_rule_calibration": True}, f)  # as a run with the bundled weights
    bam = tmp_path / "x.bam"
    _write_bam(bam)
    out = tmp_path / "site-level"
    pileup_main(_pileup_args(inp, out, bam))
    z = np.load(out / "pileup.npz", allow_pickle=True)
    for key in ("stoich_state", "stoichiometry_release", "stoichiometry_eq14"):
        assert key in z.files
    # the calibration is fixed and not shown: no calibrated sums in the output, and nothing about it printed
    assert not any(key in z.files for key, _ in pileup_deeprm._RULE_SUMS)
    printed = capsys.readouterr()
    printed = printed.out + printed.err + "\n".join(pileup_log)
    not_shown = "temperature" not in printed and str(pileup_deeprm._CALIBRATION) not in printed
    assert not_shown, "the pileup printed the calibration"
    pos = list(z["ref_pos"])
    state = dict(zip(pos, z["stoich_state"].tolist()))
    stoich = dict(zip(pos, z["stoichiometry"].tolist()))
    eq14 = dict(zip(pos, z["stoichiometry_eq14"].tolist()))
    assert set(state.values()) <= {-1, 1}
    assert state[200] == -1 and stoich[200] == 0  # 600 reads, none confident: not reported
    assert state[300] == 1 and eq14[300] == 0 and stoich[300] == 1 / 2000  # reported by l, at k / n
    assert state[600] == -1 and stoich[600] == 0  # calibrated: no confident read pays for the stoichiometry
    # a reported site with an Eq. 14 value is reported at that value
    assert all(stoich[p] == eq14[p] for p in pos if state[p] == 1 and eq14[p] > 0)
    # every site as the rule's definition gives it, with the calibrated evidence computed here from the same reads
    e_cal = []
    for p in pos:
        p_cal = pileup_deeprm._calibrated_probability(pred[label == p + 1], calibrated=True)
        logs = np.log10(np.clip(1 - p_cal[p_cal >= 0.98], 1e-30, 1.0)).astype(np.float64)
        e_cal.append(logs.sum() / LOG10_CUT)
    state_ref, value_ref = _reference_b3v(
        z["stoichiometry_eq14"], z["count_all"].astype(float), z["count_pos"], np.array(e_cal)
    )
    assert np.array_equal(z["stoich_state"], state_ref)
    assert np.array_equal(z["stoichiometry_release"], value_ref, equal_nan=True)
    bed = (out / "pileup.bed").read_text().splitlines()
    assert len(bed) == len(pos) and all(len(line.split("\t")) == 18 for line in bed)

    # without deeprm_run.json the model is unknown: a warning, and the rule runs without the calibration
    (inp / "deeprm_run.json").unlink()
    out_nojson = tmp_path / "site-level-nojson"
    pileup_log.clear()
    pileup_main(_pileup_args(inp, out_nojson, bam))
    assert any("No deeprm_run.json" in line for line in pileup_log)
    z_nojson = np.load(out_nojson / "pileup.npz", allow_pickle=True)
    assert np.array_equal(z_nojson["modscore"], z["modscore"])
    state_nojson = dict(zip(z_nojson["ref_pos"].tolist(), z_nojson["stoich_state"].tolist()))
    assert state_nojson[200] == -1 and state_nojson[300] == 1  # neither depends on the calibration
    assert state_nojson[600] == 1  # without the calibration, the 40 confident reads pay for it

    # deeprm call run records whether the model file is the bundled v1.2 weights; the pileup follows that record
    for recorded, expected in ((True, -1), (False, 1)):
        with open(inp / "deeprm_run.json", "w") as f:
            json.dump({"model_name": "deeprm_v12", "site_rule_calibration": recorded}, f)
        out_rec = tmp_path / f"site-level-{recorded}"
        pileup_main(_pileup_args(inp, out_rec, bam))
        z_rec = np.load(out_rec / "pileup.npz")
        assert dict(zip(z_rec["ref_pos"].tolist(), z_rec["stoich_state"].tolist()))[600] == expected


def test_pileup_survives_an_empty_worker_and_reports_worker_errors(tmp_path):
    rng = np.random.default_rng(5)
    inp = tmp_path / "molecule-level"
    inp.mkdir()
    n = 50
    logit = rng.uniform(-8, 8, n).astype(np.float32)
    common = dict(label_id=np.full(n, 11, dtype=np.int64), read_id=rng.integers(0, 2**62, size=(n, 2), dtype=np.int64))
    np.savez(inp / "inference_0_0.npz", pred=pileup_deeprm._sigmoid_np(logit), **common)
    # a file whose rows are all invalid: its worker has nothing to send
    np.savez(inp / "inference_0_1.npz", pred=np.full(n, np.nan, np.float32), **common)
    bam = tmp_path / "x.bam"
    _write_bam(bam)
    args = _pileup_args(inp, tmp_path / "out", bam)
    args.thread = 2
    pileup_main(args)
    assert (tmp_path / "out" / "pileup.npz").is_file()

    # a worker that fails (here: --slice on 1-D predictions) stops the pileup with its error, instead of hanging
    args = _pileup_args(inp, tmp_path / "out_err", bam)
    args.thread = 2
    args.slice = 0
    with pytest.raises(RuntimeError, match="worker"):
        pileup_main(args)


def test_pileup_parent_failure_releases_workers(tmp_path, monkeypatch):
    """A failure in the parent while it reads the workers' results must not leave the workers waiting."""
    rng = np.random.default_rng(7)
    inp = tmp_path / "molecule-level"
    inp.mkdir()
    for i in range(3):
        logit = rng.uniform(-8, 8, 40).astype(np.float32)
        np.savez(
            inp / f"inference_0_{i}.npz",
            label_id=np.full(40, 5 + i, dtype=np.int64),
            read_id=rng.integers(0, 2**62, size=(40, 2), dtype=np.int64),
            pred=pileup_deeprm._sigmoid_np(logit),
        )
    bam = tmp_path / "x.bam"
    _write_bam(bam)

    def failing_view(meta):
        raise MemoryError("injected")

    monkeypatch.setattr(pileup_deeprm, "shm_view", failing_view)
    args = _pileup_args(inp, tmp_path / "out", bam)
    args.thread = 3
    with pytest.raises(MemoryError):
        pileup_main(args)


def test_site_sums_must_be_int64_or_float64():
    """The workers send their sums in one shared-memory block per dtype: a sum of another dtype is refused."""
    ints, floats = pileup_deeprm._sums_by_dtype(pileup_deeprm.SITE_SUMS)
    assert (ints, floats) == (pileup_deeprm.INT_SUMS, pileup_deeprm.FLOAT_SUMS)
    assert set(ints) | set(floats) == {key for key, _ in pileup_deeprm.SITE_SUMS}
    with pytest.raises(TypeError, match="p_bin"):
        pileup_deeprm._sums_by_dtype(pileup_deeprm.SITE_SUMS + (("p_bin", np.float32),))


def _child_env():
    """The environment of a fresh interpreter that imports the same deeprm as this test. sys.path goes along, so the
    child also finds dependencies that come from PYTHONPATH."""
    import deeprm

    path = [os.path.dirname(os.path.dirname(deeprm.__file__)), *(p for p in sys.path if p)]
    return dict(os.environ, PYTHONPATH=os.pathsep.join(path))


def test_rule_lives_in_the_pileup_module():
    """The rule moved into pileup_deeprm: the old module is gone, and the genomic module does not load the rule."""
    import importlib
    import subprocess

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("deeprm.inference.stoich_rule")
    env = _child_env()
    code = (
        "import sys; import deeprm.inference.pileup_genomic as g; "
        "print(g.__file__); print('deeprm.inference.pileup_deeprm' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env).stdout
    path, loaded = out.split()
    assert os.path.dirname(path) == os.path.dirname(pileup_deeprm.__file__)
    assert loaded == "False"


def test_importing_the_pileup_keeps_the_start_method():
    """The pileup forks its workers through its own context; importing it does not set the program's start method."""
    import subprocess

    env = _child_env()
    code = (
        "import multiprocessing as mp; before = mp.get_start_method(allow_none=True); "
        "import deeprm.inference.pileup_deeprm as p; "
        "print(before, mp.get_start_method(allow_none=True), p._MP.get_start_method())"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env).stdout
    before, after, context = out.split()
    assert before == after == "None" and context == "fork"


def test_a_missing_annotation_file_is_refused_before_any_work(tmp_path):
    inp = tmp_path / "molecule-level"
    inp.mkdir()
    bam = tmp_path / "x.bam"
    _write_bam(bam)
    args = _pileup_args(inp, tmp_path / "out", bam)
    args.annot = str(tmp_path / "no_such_refFlat.txt")
    with pytest.raises(ValueError, match="--annot"):
        pileup_main(args)
    assert not (tmp_path / "out").exists()


def _write_run_info(folder, info):
    with open(folder / "deeprm_run.json", "w") as f:
        json.dump(info, f)


@pytest.mark.parametrize(
    "info, calibrated, warning",
    [
        # deeprm call run records whether the model file is the bundled v1.2 weights
        ({"model_name": "deeprm_v12", "site_rule_calibration": True}, True, None),
        ({"model_name": "deeprm_v12", "site_rule_calibration": False}, False, "other than the bundled v1.2 weights"),
        # a run file of an earlier v1.2 build, without that record: decided by the model md5 it records
        ({"model_name": "deeprm_v12", "model_md5": BUNDLED_WEIGHTS_MD5}, True, None),
        ({"model_name": "deeprm_v12", "model_md5": "0" * 32}, False, "other than the bundled v1.2 weights"),
        ({"model_name": "deeprm_v12"}, False, "other than the bundled v1.2 weights"),
        ({"model_name": "some_other_model"}, False, "other than the bundled v1.2 weights"),
    ],
)
def test_the_run_file_decides_the_calibration(tmp_path, pileup_log, info, calibrated, warning):
    _write_run_info(tmp_path, info)
    assert pileup_deeprm.inputs_calibrated(str(tmp_path)) is calibrated
    if warning is None:
        assert pileup_log == []
    else:
        assert len(pileup_log) == 1 and warning in pileup_log[0]
        assert "without the v1.2 model's calibration" in pileup_log[0]


def test_a_cut_off_run_file_is_refused_with_its_path(tmp_path):
    (tmp_path / "deeprm_run.json").write_text('{"model_name": "deeprm_v12", "site_rule')
    with pytest.raises(ValueError, match="deeprm_run.json is not a complete run record"):
        pileup_deeprm.inputs_calibrated(str(tmp_path))


def test_a_bed_only_pileup_does_not_index_the_bam(tmp_path):
    """With --skip-modbam only the BAM header is read, so an unindexed BAM is not indexed (in a folder that may be
    read-only)."""
    rng = np.random.default_rng(13)
    inp = tmp_path / "molecule-level"
    inp.mkdir()
    np.savez(
        inp / "inference_0_0.npz",
        label_id=np.full(20, 101, dtype=np.int64),
        read_id=rng.integers(0, 2**62, size=(20, 2), dtype=np.int64),
        pred=pileup_deeprm._sigmoid_np(rng.uniform(-8, 8, 20).astype(np.float32)),
    )
    header = pysam.AlignmentHeader.from_dict({"HD": {"VN": "1.6"}, "SQ": [{"SN": "tx1", "LN": 5000}]})
    bam = tmp_path / "unindexed.bam"
    with pysam.AlignmentFile(str(bam), "wb", header=header):
        pass
    pileup_main(_pileup_args(inp, tmp_path / "out", bam))  # _pileup_args sets skip_modbam
    assert (tmp_path / "out" / "pileup.npz").is_file()
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("unindexed")) == ["unindexed.bam"]


def test_a_bam_path_that_is_not_a_bam_is_refused_before_any_work(tmp_path):
    sam = tmp_path / "reads.sam"
    sam.write_text("@HD\tVN:1.6\n")
    args = _pileup_args(tmp_path, tmp_path / "out", sam)
    with pytest.raises(ValueError, match="--bam: expected a .bam file, got: reads.sam"):
        pileup_deeprm._validate_args(args)


def test_no_run_file_means_no_calibration_and_a_warning(tmp_path, pileup_log):
    assert pileup_deeprm.inputs_calibrated(str(tmp_path)) is False
    assert len(pileup_log) == 1 and "No deeprm_run.json" in pileup_log[0]


def test_threshold_changes_modscore_but_not_the_site_rule(tmp_path):
    """--threshold moves modscore, count_pos and stoichiometry_eq14; the rule (states, reported values) stays at the
    cut its constants were measured at (0.98)."""
    rng = np.random.default_rng(11)
    sites = {
        100: np.r_[0.95, np.full(300, 0.7), np.full(1699, 0.01)],  # reported by l at a cut of 0.9 if it followed it
        200: np.r_[np.full(5, 0.95), np.full(1995, 0.01)],  # reported by h at a cut of 0.9 if it followed it
        300: np.r_[0.9995, 0.9995, np.full(30, 0.985), np.full(2968, 0.01)],  # reported by l at a cut of 0.999
        400: pileup_deeprm._sigmoid_np(np.r_[rng.uniform(6, 9, 18), rng.uniform(-9, -5, 12)].astype(np.float32)),
        500: pileup_deeprm._sigmoid_np(np.full(40, 3.95, dtype=np.float32)),
    }
    label = np.concatenate([np.full(len(p), pos + 1, dtype=np.int64) for pos, p in sites.items()])
    pred = np.concatenate(list(sites.values())).astype(np.float32)
    inp = tmp_path / "molecule-level"
    inp.mkdir()
    np.savez(inp / "inference_0_0.npz", label_id=label, read_id=np.zeros((len(label), 2), np.int64), pred=pred)
    _write_run_info(inp, {"model_name": "deeprm_v12", "site_rule_calibration": True})
    bam = tmp_path / "x.bam"
    _write_bam(bam)
    out = {}
    for threshold in (0.98, 0.9, 0.999):
        args = _pileup_args(inp, tmp_path / f"out-{threshold}", bam)
        args.threshold = threshold
        pileup_main(args)
        out[threshold] = (
            np.load(tmp_path / f"out-{threshold}" / "pileup.npz"),
            [line.split("\t") for line in (tmp_path / f"out-{threshold}" / "pileup.bed").read_text().splitlines()],
        )
    z98, bed98 = out[0.98]
    assert dict(zip(z98["ref_pos"].tolist(), z98["stoich_state"].tolist())) == {
        100: -1,
        200: -1,
        300: -1,
        400: 1,
        500: -1,
    }
    for threshold in (0.9, 0.999):
        z, bed = out[threshold]
        assert np.array_equal(z["ref_pos"], z98["ref_pos"])
        assert np.array_equal(z["stoich_state"], z98["stoich_state"])
        assert np.array_equal(z["stoichiometry"], z98["stoichiometry"])
        assert np.array_equal(z["stoichiometry_release"], z98["stoichiometry_release"], equal_nan=True)
        assert [row[9:13] for row in bed] == [row[9:13] for row in bed98]  # coverage, stoichiometry and counts
        # the detection side follows the per-read cut
        expected_pos = [int((sites[pos].astype(np.float32) >= threshold).sum()) for pos in z["ref_pos"].tolist()]
        assert z["count_pos"].tolist() == expected_pos
        assert not np.array_equal(z["modscore"], z98["modscore"])
        assert [row[4] for row in bed] != [row[4] for row in bed98]
        # stoichiometry_eq14 is the v1.1 value: Eq. 12 times the Eq. 14 test at the per-read cut
        s12 = z["kl_div_pos"] / (z["kl_div_neg"] + z["kl_div_pos"] + 1e-30)
        eq14 = s12 * ((np.log10(1 - threshold) * s12) > (z["logsum_1_p_pos"] / z["count_all"]))
        assert np.array_equal(z["stoichiometry_eq14"], eq14)
