"""Tests for deeprm.inference.pileup: the C++ pileup binary where it can start, the Python pileup otherwise.

The binary here is a stand-in: a script that answers --version as the C++ program does, records the options it was
given, and makes the outputs with the Python pileup (cpp/PILEUP_INTERFACE.md says what the real program must do)."""

import argparse
import contextlib
import io
import json
import logging
import os
import sys

import numpy as np
import pytest
from test_pileup_memory import _bam_and_rows, _outputs, _random_seq
from test_pileup_modbam_merge import _args

from deeprm.inference import cli, pileup, pileup_deeprm
from deeprm.utils import check_installation, prep_binary

STAND_IN = """#!{python}
import argparse, json, os, sys
if sys.argv[1:] == ["--version"]:
    print("2026-11-12T00:00:00+09:00")
    print("DeepRM Pileup v1.2.1-test")
    sys.exit(0)
with open(os.environ["STAND_IN_LOG"], "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
if os.environ.get("STAND_IN_FAIL"):
    out = sys.argv[sys.argv.index("--output") + 1]
    open(os.path.join(out, "pileup.part.bed"), "w").close()  # what the binary was writing
    os.makedirs(os.path.join(out, ".deeprm_modbam_abc123"))
    print("stand-in: failing on purpose", file=sys.stderr)
    if os.environ["STAND_IN_FAIL"] == "SIGINT":
        import signal
        os.kill(os.getpid(), signal.SIGINT)
    sys.exit(3)
p = argparse.ArgumentParser()
for name in ("input", "output", "bam", "postfix", "annot", "tmp_dir"):
    p.add_argument("--" + name.replace("_", "-"), dest=name, default=None)
for name in ("--thread", "--slice", "--label_div"):
    p.add_argument(name, type=int, default=None)
for name in ("threshold", "epsilon"):
    p.add_argument("--" + name, type=float)
p.add_argument("--calibration", choices=("on", "off"))
for name in ("flip", "skip_modbam", "report_all_stoichiometry"):
    p.add_argument("--" + name.replace("_", "-"), dest=name, action="store_true")
a = p.parse_args()
a.postfix = a.postfix or ""
from deeprm.inference import pileup_deeprm
pileup_deeprm._pileup(a, a.calibration == "on")
"""


def _stand_in(folder, monkeypatch):
    """The stand-in binary in `folder`, with its log; returns (path, log path)."""
    if prep_binary.on_noexec_mount(str(folder)):
        pytest.skip(f"{folder} does not allow programs to run")
    path = folder / "deeprm_pileup"
    path.write_text(STAND_IN.format(python=sys.executable))
    path.chmod(0o755)
    log_path = folder / "stand_in.log"
    src = os.path.dirname(os.path.dirname(os.path.dirname(pileup_deeprm.__file__)))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")])))
    monkeypatch.setenv("STAND_IN_LOG", str(log_path))
    return path, log_path


def _calls(log_path):
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text().splitlines()]


def _input(tmp_path, calibrated=False):
    rng = np.random.default_rng(3)
    reads = [
        (t, int(rng.integers(0, 400)), [(0, 300)], _random_seq(rng, 300), 0, []) for t in range(4) for _ in range(30)
    ]
    inp, bam = _bam_and_rows(tmp_path, [(f"tx{i}", 1000) for i in range(4)], reads, n_files=6)
    if calibrated:
        (inp / pileup_deeprm.RUN_INFO_NAME).write_text(json.dumps({"site_rule_calibration": True}))
    return inp, bam


@pytest.fixture(autouse=True)
def _no_named_binary(monkeypatch, tmp_path):
    """No binary named in the environment, and no bundled binary, unless a test sets one; the stand-in passes the
    check against NumPy (it has no terms-range: it is the Python pileup)."""
    monkeypatch.delenv("DEEPRM_PILEUP_BIN", raising=False)
    monkeypatch.setattr(pileup, "DEFAULT_BIN", str(tmp_path / "no_bundled_binary"))
    monkeypatch.setattr(pileup, "_numpy_differs", lambda path, threshold, epsilon: None)


def test_without_a_bundled_binary_the_python_pileup_runs(tmp_path, caplog):
    inp, bam = _input(tmp_path)
    args = _args(inp, tmp_path / "out", bam, thread=3)
    with caplog.at_level(logging.INFO):
        pileup.main(args)
    assert "this installation has no pileup binary" in caplog.text
    assert "Falling back" not in caplog.text
    assert (tmp_path / "out" / "pileup.bed").exists() and (tmp_path / "out" / "modbam_input.bam").exists()


def test_the_binary_gets_the_resolved_settings_and_gives_the_python_outputs(tmp_path, monkeypatch):
    """--thread left out is resolved once, for both paths; every setting reaches the binary under the Python
    pileup's own option name; the binary reads deeprm_run.json itself; its outputs are those of the Python pileup."""
    inp, bam = _input(tmp_path, calibrated=True)
    exe, log_path = _stand_in(tmp_path, monkeypatch)
    monkeypatch.setattr(pileup, "default_threads", lambda: 3)
    args = _args(inp, tmp_path / "binary", bam, thread=None)
    args.pileup_bin, args.tmp_dir = str(exe), str(tmp_path / "tmp")
    pileup.main(args)
    (call,) = _calls(log_path)
    assert call[call.index("--thread") + 1] == "3"
    assert call[call.index("--tmp-dir") + 1] == str(tmp_path / "tmp")
    assert call[call.index("--threshold") + 1] == "0.98"
    assert call[call.index("--label_div") + 1] == str(args.label_div)
    assert call[call.index("--calibration") + 1] == "on"  # DeepRM reads deeprm_run.json and passes its decision
    assert "--skip-modbam" not in call and "--flip" not in call
    assert "--report-all-stoichiometry" not in call

    python = _args(inp, tmp_path / "python", bam, thread=3)
    pileup_deeprm.main(python)
    assert _outputs(tmp_path / "binary") == _outputs(tmp_path / "python")


def test_a_threshold_other_than_the_default_runs_the_python_pileup(tmp_path, monkeypatch, caplog):
    """The binary has log10(1 - threshold) as NumPy gives it for the default threshold only: with another one the
    Python pileup runs, so the results stay those of the Python pileup."""
    inp, bam = _input(tmp_path)
    exe, log_path = _stand_in(tmp_path, monkeypatch)
    args = _args(inp, tmp_path / "out", bam, thread=2)
    args.pileup_bin, args.threshold = str(exe), 0.9
    with caplog.at_level(logging.INFO):
        pileup.main(args)
    assert _calls(log_path) == [] and "--threshold" in caplog.text and (tmp_path / "out" / "pileup.bed").exists()


def test_flags_reach_the_binary_only_when_set(tmp_path, monkeypatch):
    inp, bam = _input(tmp_path)
    exe, log_path = _stand_in(tmp_path, monkeypatch)
    monkeypatch.setenv("DEEPRM_PILEUP_BIN", str(exe))
    args = _args(inp, tmp_path / "out", bam, thread=2)
    args.skip_modbam, args.flip, args.report_all_stoichiometry = True, True, True
    pileup.main(args)
    (call,) = _calls(log_path)
    assert "--skip-modbam" in call and "--flip" in call and call[call.index("--calibration") + 1] == "off"
    assert "--report-all-stoichiometry" in call
    assert "--postfix" not in call and "--annot" not in call and "--slice" not in call
    assert not (tmp_path / "out" / "modbam_input.bam").exists()


def test_a_named_binary_that_cannot_start_stops_the_run(tmp_path, monkeypatch):
    inp, bam = _input(tmp_path)
    args = _args(inp, tmp_path / "out", bam, thread=2)
    args.pileup_bin = str(tmp_path / "missing_binary")
    with pytest.raises(ValueError, match="given with --pileup-bin cannot start: there is no file"):
        pileup.main(args)
    args.pileup_bin = None
    monkeypatch.setenv("DEEPRM_PILEUP_BIN", str(tmp_path / "missing_binary"))
    with pytest.raises(ValueError, match="given with DEEPRM_PILEUP_BIN cannot start"):
        pileup.main(args)
    assert not (tmp_path / "out" / "pileup.bed").exists()


def test_deeprm_call_run_checks_a_named_binary_before_the_inference(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="--pileup-bin cannot start"):
        pileup.check_named_binary(argparse.Namespace(pileup_bin=str(tmp_path / "missing_binary")))
    pileup.check_named_binary(argparse.Namespace(pileup_bin=None))  # the bundled binary, or none: nothing to check
    exe, _ = _stand_in(tmp_path, monkeypatch)
    pileup.check_named_binary(argparse.Namespace(pileup_bin=str(exe)))


def test_a_bundled_binary_that_cannot_start_falls_back_to_python(tmp_path, monkeypatch, caplog):
    inp, bam = _input(tmp_path)
    exe, log_path = _stand_in(tmp_path, monkeypatch)
    monkeypatch.setattr(pileup, "DEFAULT_BIN", str(exe))
    reason = "the binary stopped with 'Illegal instruction': this CPU lacks an instruction the binary uses"
    monkeypatch.setattr(pileup, "_binary_status", lambda path: prep_binary.Status(False, reason=reason, kind="cpu"))
    with caplog.at_level(logging.INFO):
        pileup.main(_args(inp, tmp_path / "out", bam, thread=2))
    assert "Falling back to the Python pileup" in caplog.text and "Illegal instruction" in caplog.text
    assert _calls(log_path) == [] and (tmp_path / "out" / "pileup.bed").exists()


def test_a_bundled_binary_without_its_execute_bits_gets_them_back(tmp_path, monkeypatch):
    inp, bam = _input(tmp_path)
    exe, log_path = _stand_in(tmp_path, monkeypatch)
    exe.chmod(0o644)
    monkeypatch.setattr(pileup, "DEFAULT_BIN", str(exe))
    pileup.main(_args(inp, tmp_path / "out", bam, thread=2))
    assert os.access(exe, os.X_OK) and len(_calls(log_path)) == 1


def test_a_named_binary_that_fails_stops_the_run_and_leaves_nothing(tmp_path, monkeypatch):
    inp, bam = _input(tmp_path)
    exe, log_path = _stand_in(tmp_path, monkeypatch)
    monkeypatch.setenv("STAND_IN_FAIL", "1")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "notes.txt").write_text("the user's")  # there before the run: left alone
    args = _args(inp, tmp_path / "out", bam, thread=2)
    args.pileup_bin = str(exe)
    with pytest.raises(ValueError, match="given with --pileup-bin failed \\(exit code 3\\)"):
        pileup.main(args)
    assert len(_calls(log_path)) == 1 and sorted(os.listdir(tmp_path / "out")) == ["notes.txt"]


def test_a_bundled_binary_that_fails_is_followed_by_the_python_pileup(tmp_path, monkeypatch, caplog):
    """For an input the binary cannot read but the Python pileup can: the binary's work is removed and the Python
    pileup runs, so deeprm call pileup gives the Python pileup's outcome."""
    inp, bam = _input(tmp_path)
    exe, log_path = _stand_in(tmp_path, monkeypatch)
    monkeypatch.setattr(pileup, "DEFAULT_BIN", str(exe))
    monkeypatch.setenv("STAND_IN_FAIL", "1")
    with caplog.at_level(logging.INFO):
        pileup.main(_args(inp, tmp_path / "out", bam, thread=2))
    assert "running the Python pileup instead" in caplog.text and len(_calls(log_path)) == 1
    assert sorted(os.listdir(tmp_path / "out")) == [
        "modbam_input.bam",
        "modbam_input.bam.bai",
        "pileup.bed",
        "pileup.npz",
    ]


def test_ctrl_c_in_the_binary_stops_the_run_and_leaves_nothing(tmp_path, monkeypatch):
    inp, bam = _input(tmp_path)
    exe, _ = _stand_in(tmp_path, monkeypatch)
    monkeypatch.setattr(pileup, "DEFAULT_BIN", str(exe))
    monkeypatch.setenv("STAND_IN_FAIL", "SIGINT")
    with pytest.raises(KeyboardInterrupt):
        pileup.main(_args(inp, tmp_path / "out", bam, thread=2))
    assert os.listdir(tmp_path / "out") == []


def test_where_numpy_computes_otherwise_the_python_pileup_runs(tmp_path, monkeypatch, caplog):
    """A CPU without AVX512, or another NumPy version: the bundled binary's results would not be the Python pileup's,
    so the Python pileup runs; a binary the user named runs, with a warning."""
    inp, bam = _input(tmp_path)
    exe, log_path = _stand_in(tmp_path, monkeypatch)
    monkeypatch.setattr(pileup, "_numpy_differs", lambda path, threshold, epsilon: "the per-read term kl_div_pos")
    monkeypatch.setattr(pileup, "DEFAULT_BIN", str(exe))
    with caplog.at_level(logging.INFO):
        pileup.main(_args(inp, tmp_path / "bundled", bam, thread=2))
    assert _calls(log_path) == [] and "does not compute the per-read term kl_div_pos" in caplog.text
    args = _args(inp, tmp_path / "named", bam, thread=2)
    args.pileup_bin = str(exe)
    with caplog.at_level(logging.INFO):
        pileup.main(args)
    assert len(_calls(log_path)) == 1 and "may differ from the Python pileup's" in caplog.text


def test_settings_of_deeprm_call_run_do_not_send_the_run_to_python(tmp_path):
    """deeprm call run passes its whole Namespace on; its own options are not pileup options."""
    inp, bam = _input(tmp_path)
    args = _args(inp, tmp_path / "out", bam, thread=2)
    args.model, args.gpu_pool, args.batch = "model.pt", [0, 1], 512
    assert pileup._python_only_options(args) == []


def test_the_cli_runs_the_wrapper_and_lists_its_option():
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit):
        cli._build_leaf_parser("pileup", "deeprm call pileup").parse_args(["--help"])
    assert "--pileup-bin" in out.getvalue() and "--tmp-dir" in out.getvalue()
    assert cli._LEAVES["pileup"] == "deeprm.inference.pileup"
    from deeprm.inference import inference

    parser = argparse.ArgumentParser()
    inference.add_arguments(parser)
    assert parser.get_default("pileup_bin") is None and "--pileup-bin" in parser.format_help()


def test_deeprm_check_reports_the_pileup_binary(tmp_path, monkeypatch):
    monkeypatch.setattr(prep_binary, "bundled", lambda program=prep_binary.PREP: str(tmp_path / "none"))
    info, issues = check_installation._pileup_binary_info()
    assert info["present"] is False and issues == []
    exe, _ = _stand_in(tmp_path, monkeypatch)
    monkeypatch.setenv("DEEPRM_PILEUP_BIN", str(exe))
    info, issues = check_installation._pileup_binary_info()
    assert info["starts"] and info["version"] == "DeepRM Pileup v1.2.1-test" and issues == []
    monkeypatch.setenv("DEEPRM_PILEUP_BIN", str(tmp_path / "missing_binary"))
    info, issues = check_installation._pileup_binary_info()
    assert not info["starts"] and [i.code for i in issues] == ["PILEUP_BINARY_CANNOT_START"]
    assert issues[0].severity == "ERROR" and "DEEPRM_PILEUP_BIN" in issues[0].hint


def test_status_hints_name_the_pileup_options(tmp_path):
    st = prep_binary.status(str(tmp_path / "nothing"), program=prep_binary.PILEUP)
    assert not st.ok and "--pileup-bin or DEEPRM_PILEUP_BIN" in st.hint
    st = prep_binary.status(str(tmp_path / "nothing"))  # the preprocessing's own hint is unchanged
    assert "--preprocess-bin or DEEPRM_PREPROCESS_BIN" in st.hint


def test_the_bundled_pileup_binary_starts(monkeypatch):
    """The binary that ships with DeepRM starts on Linux x86_64 with AVX2 and FMA (a binary added without its execute
    bits, for example, would fail here; the parity tests would only be skipped)."""
    import platform

    monkeypatch.undo()  # the real DEFAULT_BIN, not the fixture's
    path = pileup.DEFAULT_BIN
    if not os.path.isfile(path):
        pytest.skip(f"no bundled pileup binary at {path}")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        pytest.skip(f"the bundled binary is for Linux x86_64; this is {platform.system()} {platform.machine()}")
    with open("/proc/cpuinfo") as f:
        flags = set(next((line for line in f if line.startswith("flags")), "").split())
    if not {"avx2", "fma"} <= flags:
        pytest.skip("this CPU lacks AVX2 or FMA")
    assert os.access(path, os.X_OK), f"{path} has no execute permission"
    st = prep_binary.status(path, program=prep_binary.PILEUP)
    assert st.ok and st.version.startswith("DeepRM Pileup v"), st
