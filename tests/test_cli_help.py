"""deeprm --help of each command group. They run `python -m deeprm` with this test's sys.path, so the code under test
answers, not a deeprm console script that happens to be on PATH."""

import os
import subprocess
import sys


def _help(*group):
    """Runs `python -m deeprm <group> --help`; returns (stdout, stdout + stderr)."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(os.path.abspath(p) for p in sys.path if p))
    result = subprocess.run(
        [sys.executable, "-m", "deeprm", *group, "--help"], capture_output=True, text=True, env=env, timeout=120
    )
    assert result.returncode == 0, result.stderr
    return result.stdout, result.stdout + result.stderr


def test_cli_top_level_help():
    stdout, output = _help()
    assert "DeepRM" in output
    assert "usage" in stdout.lower()


def test_cli_inference_help():
    stdout, output = _help("call")
    assert "DeepRM Call (inference) Module" in output
    assert "usage" in stdout.lower()


def test_cli_train_help():
    stdout, output = _help("train")
    assert "DeepRM Training Module" in output
    assert "usage" in stdout.lower()


def test_cli_qc_help():
    stdout, output = _help("qc")
    assert "DeepRM QC Module" in output
    assert "usage" in stdout.lower()
