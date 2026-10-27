"""Tests for deeprm.utils: mean_phred, the logging helpers and maybe_index_bam (CPU only)."""

import hashlib
import importlib
import io
import logging
import os
import shutil
import subprocess
import sys
from array import array

import numpy as np
import pytest

from deeprm.utils import logging as deeprm_logging
from deeprm.utils.utils import maybe_index_bam, mean_phred


@pytest.mark.parametrize(
    "phred",
    [
        [30, 30, 10, 40],
        np.array([30, 30, 10, 40], dtype=np.int64),
        np.array([30, 30, 10, 40], dtype=np.uint8),
        np.array([30, 30, 10, 40], dtype=np.float32),
        array("B", [30, 30, 10, 40]),
    ],
)
def test_mean_phred(phred):
    # the mean is taken over error probabilities, not over the scores: -10 log10(mean(10^(-q/10)))
    assert mean_phred(phred) == pytest.approx(15.9303, abs=1e-4)


def test_mean_phred_of_equal_scores():
    assert mean_phred([30] * 5) == pytest.approx(30.0)


def _run_python(code, env=None):
    """Runs `code` in a new Python with this test's sys.path; returns (stdout, stderr). stderr is a pipe."""
    full_env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p), **(env or {}))
    full_env.pop("NO_COLOR", None)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=full_env, timeout=120)
    assert result.returncode == 0, result.stderr
    return result.stdout, result.stderr


def test_importing_the_logging_module_leaves_the_streams_alone():
    out, _ = _run_python(
        "import sys\n"
        "before = (type(sys.stdout), type(sys.stderr))\n"
        "import deeprm.utils.logging\n"
        "import deeprm.utils.check_installation\n"
        "print(before == (type(sys.stdout), type(sys.stderr)), 'colorama' in sys.modules)\n"
    )
    assert out.split() == ["True", "False"]


def test_log_lines_on_a_pipe_have_no_colour_and_keep_the_traceback():
    _, err = _run_python(
        "from deeprm.utils.logging import get_logger\n"
        "log = get_logger('deeprm.test_pipe')\n"
        "log.warning('first line\\nsecond line')\n"
        "try:\n"
        "    1 / 0\n"
        "except ZeroDivisionError:\n"
        "    log.exception('it failed')\n"
    )
    assert "\x1b" not in err
    assert "WARNING first line" in err and "Traceback" in err and "ZeroDivisionError" in err
    # later lines are indented under the first, the traceback too
    lines = err.splitlines()
    assert lines[1].startswith(" ") and lines[1].strip() == "second line"
    assert all(line.startswith(" ") for line in lines[3:])


class _FakeTerminal(io.StringIO):
    def isatty(self):
        return True


def test_use_color(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert deeprm_logging.use_color(io.StringIO()) is False
    assert deeprm_logging.use_color(object()) is False  # no isatty
    assert deeprm_logging.use_color(_FakeTerminal()) is True
    monkeypatch.setenv("NO_COLOR", "1")
    assert deeprm_logging.use_color(_FakeTerminal()) is False
    assert deeprm_logging.is_terminal(_FakeTerminal()) is True


def test_colour_formatter():
    record = logging.LogRecord("x", logging.WARNING, __file__, 1, "a\nb", None, None)
    plain = deeprm_logging.ColorFormatter(use_color=False).format(record)
    coloured = deeprm_logging.ColorFormatter(use_color=True).format(record)
    prefix = plain.split(" a\n")[0]  # "[time] WARNING"
    assert prefix.endswith("] WARNING") and plain == f"{prefix} a\n{' ' * (len(prefix) + 1)}b"
    assert coloured == f"{prefix} \x1b[33ma\n{' ' * (len(prefix) + 1)}b\x1b[0m"


def test_training_bars_write_no_control_codes_to_a_log_file(monkeypatch):
    pytest.importorskip("torch")
    train = importlib.import_module("deeprm.train.train")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    assert train._bar_kwargs(0) == {} and train._bar_kwargs(3) == {}
    monkeypatch.setattr(sys, "stderr", _FakeTerminal())
    assert train._bar_kwargs(1) == {"position": 1, "colour": "green"}
    monkeypatch.setenv("NO_COLOR", "1")
    assert train._bar_kwargs(1) == {"position": 1}


def test_add_log_path_twice_writes_each_line_once(tmp_path):
    log = deeprm_logging.get_logger("deeprm.test_add_log_path")
    path = tmp_path / "run.log"
    try:
        deeprm_logging.add_log_path(log, str(path))
        deeprm_logging.add_log_path(log, str(path))
        assert sum(isinstance(h, logging.FileHandler) for h in log.handlers) == 1
        log.warning("only once")
    finally:
        for handler in list(log.handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
                log.removeHandler(handler)
    text = path.read_text()
    assert text.count("only once") == 1 and text.count("Logging to file") == 1
    assert f"Logging to file: {path}" in text and "\x1b" not in text


# Tiny BAMs for maybe_index_bam: two references, four reads.
_UNSORTED = [(1, 90), (0, 50), (1, 5), (0, 10)]


def _write_bam(path, positions, with_sq=True):
    pysam = pytest.importorskip("pysam")
    header = {"HD": {"VN": "1.6"}}
    if with_sq:
        header["SQ"] = [{"SN": "chr1", "LN": 1000}, {"SN": "chr2", "LN": 1000}]
    header = pysam.AlignmentHeader.from_dict(header)
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for i, (ref, pos) in enumerate(positions):
            a = pysam.AlignedSegment(header)
            a.query_name = f"read{i}"
            a.query_sequence = "ACGTACGTAC"
            a.query_qualities = pysam.qualitystring_to_array("I" * 10)
            if with_sq:
                a.flag, a.reference_id, a.reference_start, a.mapping_quality = 0, ref, pos, 60
                a.cigartuples = [(0, 10)]
            else:
                a.flag = 4  # unmapped
            out.write(a)
    return path


def _md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def _positions(path):
    import pysam

    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as bam:
        return [(a.reference_id, a.reference_start) for a in bam.fetch(until_eof=True)]


def test_maybe_index_bam_indexes_a_sorted_bam_in_place(tmp_path):
    bam = _write_bam(tmp_path / "sorted.bam", sorted(_UNSORTED))
    assert maybe_index_bam(str(bam)) == str(bam)
    assert (tmp_path / "sorted.bam.bai").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sorted.bam", "sorted.bam.bai"]


def _index_and_log(bam):
    """Run maybe_index_bam and return its result and the messages it logged."""
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger = logging.getLogger("deeprm.utils.utils")
    logger.addHandler(handler)
    try:
        return maybe_index_bam(str(bam)), [r.getMessage() for r in records]
    finally:
        logger.removeHandler(handler)


def test_maybe_index_bam_never_rewrites_an_unsorted_input(tmp_path):
    bam = _write_bam(tmp_path / "sample.bam", _UNSORTED)
    md5 = _md5(bam)
    first, messages = _index_and_log(bam)
    assert first == str(tmp_path / "sample.sorted.bam")
    assert _positions(tmp_path / "sample.sorted.bam") == sorted(_UNSORTED)
    # the samtools message is logged as plain text, not as a quoted repr with a literal "\n"
    reasons = [m for m in messages if m.startswith("Could not index")]
    assert reasons and all("\\n" not in m and not m.endswith("'") for m in reasons)
    # the second run reuses the sorted copy; the input is unchanged (an old version sorted it onto itself)
    assert maybe_index_bam(str(bam)) == first
    assert _md5(bam) == md5 and _positions(bam) == _UNSORTED
    assert not list(tmp_path.glob("*.tmp")) and not (tmp_path / "sample.bam.bai").exists()
    # a changed input (a new modification time) gets a new copy; the old one stays, and its path is logged
    later = os.stat(bam).st_mtime + 10
    os.utime(bam, (later, later))
    second, messages = _index_and_log(bam)
    assert second == str(tmp_path / "sample.sorted.1.bam")
    assert (tmp_path / "sample.sorted.bam").exists() and _md5(bam) == md5
    assert any(str(tmp_path / "sample.sorted.bam") in m and "left in place" in m for m in messages)
    assert maybe_index_bam(str(bam)) == second


def test_maybe_index_bam_makes_a_new_copy_when_the_input_is_replaced_by_an_older_file(tmp_path):
    # The replacement keeps its older modification time (as after cp -p or rsync -t), so it is older than the sorted
    # copy of the first input. That copy holds the wrong reads and must not be reused.
    other = _write_bam(tmp_path / "other.bam", [(1, 700), (0, 600), (0, 300)])
    bam = _write_bam(tmp_path / "sample.bam", _UNSORTED)
    earlier = os.stat(bam).st_mtime - 100
    os.utime(other, (earlier, earlier))
    first = maybe_index_bam(str(bam))
    shutil.copy2(other, bam)
    second = maybe_index_bam(str(bam))
    assert second == str(tmp_path / "sample.sorted.1.bam")
    assert _positions(second) == sorted(_positions(bam)) == [(0, 300), (0, 600), (1, 700)]
    assert _positions(first) == sorted(_UNSORTED)


def test_maybe_index_bam_reuses_the_copy_of_an_input_dated_in_the_future(tmp_path):
    # The input's modification time is ahead of this machine's clock (clock skew), so every new copy is older than it.
    bam = _write_bam(tmp_path / "f.bam", _UNSORTED)
    future = os.stat(bam).st_mtime + 3600
    os.utime(bam, (future, future))
    assert [maybe_index_bam(str(bam)) for _ in range(3)] == [str(tmp_path / "f.sorted.bam")] * 3
    assert not (tmp_path / "f.sorted.1.bam").exists()
    # a copy without its record of the input (for example, one made by hand) is not reused
    (tmp_path / "f.sorted.bam.source").unlink()
    assert maybe_index_bam(str(bam)) == str(tmp_path / "f.sorted.1.bam")


def _fetched(path):
    """Reads per reference (chr1, chr2) that fetch() finds through the BAM index."""
    import pysam

    with pysam.AlignmentFile(str(path), "rb") as bam:
        return [bam.count(contig) for contig in ("chr1", "chr2")]


@pytest.mark.parametrize("index_name", ["sample.bam.bai", "sample.csi"])
def test_maybe_index_bam_rebuilds_an_index_older_than_the_bam(tmp_path, index_name):
    # The BAM was rewritten in place (re-aligned, for example) and its old index left beside it. htslib only warns
    # and fetch() misses the reads of chr2, so the index is rebuilt (in place, of the same kind).
    pysam = pytest.importorskip("pysam")
    bam = _write_bam(tmp_path / "sample.bam", [(0, 10), (0, 50)])
    pysam.index(*(["-c"] if index_name.endswith(".csi") else []), str(bam), str(tmp_path / index_name))
    _write_bam(bam, sorted(_UNSORTED))
    earlier = os.stat(bam).st_mtime - 10
    os.utime(tmp_path / index_name, (earlier, earlier))
    assert _fetched(bam) == [2, 0]  # the old index
    assert maybe_index_bam(str(bam)) == str(bam)
    assert _fetched(bam) == [2, 2]
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(["sample.bam", index_name])


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write to any file")
def test_maybe_index_bam_keeps_an_old_index_it_cannot_rewrite(tmp_path):
    # A sorted BAM in a shared folder whose index is older than the BAM (copied without -p) and belongs to someone
    # else: the index still loads, so it is used with a warning, and no sorted copy of the BAM is written.
    pysam = pytest.importorskip("pysam")
    bam = _write_bam(tmp_path / "shared.bam", sorted(_UNSORTED))
    pysam.index(str(bam))
    index = tmp_path / "shared.bam.bai"
    earlier = os.stat(bam).st_mtime - 10
    os.utime(index, (earlier, earlier))
    os.chmod(index, 0o444)
    try:
        result, messages = _index_and_log(bam)
    finally:
        os.chmod(index, 0o644)
    assert result == str(bam)
    assert any("is used as it is" in m for m in messages)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["shared.bam", "shared.bam.bai"]


def test_maybe_index_bam_indexes_a_bam_dated_in_the_future_in_place(tmp_path):
    # Its index is always older than the BAM (clock skew), so it is rebuilt on each run; no sorted copy is made.
    bam = _write_bam(tmp_path / "f.bam", sorted(_UNSORTED))
    future = os.stat(bam).st_mtime + 3600
    os.utime(bam, (future, future))
    assert [maybe_index_bam(str(bam)) for _ in range(2)] == [str(bam)] * 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["f.bam", "f.bam.bai"] and _fetched(bam) == [2, 2]


def test_maybe_index_bam_refuses_a_bam_without_sq_lines(tmp_path):
    bam = _write_bam(tmp_path / "nosq.bam", [(0, 0)], with_sq=False)
    with pytest.raises(ValueError, match="@SQ"):
        maybe_index_bam(str(bam))


def test_maybe_index_bam_input_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        maybe_index_bam(str(tmp_path / "missing.bam"))
    (tmp_path / "reads.sam").write_text("")
    with pytest.raises(ValueError, match=r"\.bam"):
        maybe_index_bam(str(tmp_path / "reads.sam"))


@pytest.mark.parametrize(
    "module", ["deeprm.train.train_compile", "deeprm.train.train_preprocess", "deeprm.train.extract_block"]
)
def test_training_prep_modules_import(module):
    importlib.import_module(module)


def test_training_prep_uses_the_shared_helpers():
    """train_preprocess.py is research code copied unchanged; it must keep finding the deeprm.utils helpers."""
    train_preprocess = importlib.import_module("deeprm.train.train_preprocess")
    from deeprm.utils import utils

    assert train_preprocess.mean_phred is utils.mean_phred
    assert train_preprocess.get_logger is deeprm_logging.get_logger
    assert train_preprocess.add_log_path is deeprm_logging.add_log_path
