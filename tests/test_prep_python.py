"""`deeprm call prep`: the Python preprocessing end to end, and how the wrapper picks a path.

The end-to-end tests write a small POD5 + BAM pair (forward reads, reverse-strand reads and reads
with a wrong move-table stride) and run ``inference_preprocess_python.main``. The fixture is
written in a child process: the test process must not touch a POD5 file before ``main`` starts
its workers, which is the condition the preprocessing itself is built to keep.
"""

import argparse
import array
import datetime
import glob
import hashlib
import multiprocessing as mp
import os
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import time
import uuid

import numpy as np
import pytest

from deeprm.inference import inference_preprocess as wrapper
from deeprm.inference import inference_preprocess_python as py_prep
from deeprm.utils import prep_binary

FIELDS = ("read_id", "label_id", "segment_len_arr", "kmer_token", "bq_token",
          "dwell_motor_token", "dwell_pore_token", "signal_token")
DTYPES = {"read_id": np.int64, "label_id": np.int64, "segment_len_arr": np.uint16, "kmer_token": np.uint8,
          "bq_token": np.uint8, "dwell_motor_token": np.float32, "dwell_pore_token": np.float32,
          "signal_token": np.float32}
COMP = str.maketrans("ACGT", "TGCA")


def _revcomp(s):
    return s.translate(COMP)[::-1]


# Reads of file 0 that the preprocessing must drop, with what is wrong with them (degenerate=True)
DEGENERATE = {2: "empty_mv", 3: "zero_mv", 4: "ts_past", 5: "short_signal"}


def _write_fixture(out, n_files=4, reads_per_file=8, seed=3, n_refs=1, ref_len=3000, degenerate=False):
    """POD5 files + a sorted, indexed BAM. Returns {read_id: (kind, BAM SEQ)} (with n_refs > 1, (kind, SEQ, ref)).

    degenerate: reads 2 to 5 of file 0 get an empty move table, a move table without a move, a signal start past the
    signal, and a move table longer than the signal (DEGENERATE)."""
    import pod5
    import pysam

    rng = np.random.default_rng(seed)
    refs = ["".join(rng.choice(list("ACGT"), ref_len)) for _ in range(n_refs)]
    now = datetime.datetime(2026, 1, 1)
    run_info = pod5.RunInfo(
        acquisition_id="a" * 32, acquisition_start_time=now, adc_max=4095, adc_min=-4096, context_tags={},
        experiment_name="t", flow_cell_id="FC1", flow_cell_product_code="FLO-PRO004RA", protocol_name="rna",
        protocol_run_id="p" * 32, protocol_start_time=now, sample_id="s1", sample_rate=4000,
        sequencing_kit="sqk-rna004", sequencer_position="1A", sequencer_position_type="promethion",
        software="test", system_name="t", system_type="t", tracking_id={},
    )
    header = {"HD": {"VN": "1.6", "SO": "unsorted"}, "SQ": [{"SN": f"tx{j + 1}", "LN": ref_len} for j in range(n_refs)]}
    reads, records = {}, []
    os.makedirs(os.path.join(out, "pod5"), exist_ok=True)
    for f in range(n_files):
        with pod5.Writer(os.path.join(out, "pod5", f"{f}.pod5")) as w:
            for i in range(reads_per_file):
                kind = "reverse" if i == 0 else ("bad_stride" if i == 1 else "forward")
                if degenerate and f == 0 and i in DEGENERATE:
                    kind = DEGENERATE[i]
                ref_id = int(rng.integers(n_refs)) if n_refs > 1 else 0  # one reference: the old random stream
                rlen = int(rng.integers(150, 300))
                start = int(rng.integers(0, ref_len - rlen))
                seq = refs[ref_id][start:start + rlen]
                dwell = rng.integers(1, 10, rlen)
                mv = np.zeros(int(dwell.sum()), dtype=np.uint8)
                mv[np.concatenate([[0], np.cumsum(dwell)[:-1]])] = 1
                ts = 100
                signal = rng.integers(40, 160, ts + mv.size * 6).astype(np.int16)
                ns = ts + mv.size * 6
                if kind == "short_signal":
                    signal = signal[: ts + mv.size * 3]
                mv_tag = [5 if kind == "bad_stride" else 6] + mv.tolist()
                if kind == "empty_mv":
                    mv_tag = [6]
                elif kind == "zero_mv":
                    mv_tag = [6] + [0] * mv.size
                if kind == "ts_past":
                    ts = signal.size + 5
                rid = uuid.uuid4()
                w.add_read(pod5.Read(
                    read_id=rid, pore=pod5.Pore(channel=1, well=1, pore_type="rna"),
                    calibration=pod5.Calibration(offset=0.0, scale=1.0), read_number=1, start_sample=0,
                    median_before=100.0, end_reason=pod5.EndReason(reason=pod5.EndReasonEnum.SIGNAL_POSITIVE,
                                                                    forced=False),
                    run_info=run_info, signal=signal))
                reads[str(rid)] = (kind, seq) if n_refs == 1 else (kind, seq, ref_id)
                records.append(dict(name=str(rid), flag=16 if kind == "reverse" else 0, pos=start, seq=seq,
                                    ref=ref_id, mv=array.array("B", mv_tag), ts=ts, ns=ns,
                                    quals=rng.integers(5, 40, rlen)))
    unsorted = os.path.join(out, "unsorted.bam")
    with pysam.AlignmentFile(unsorted, "wb", header=header) as bam:
        for r in records:
            a = pysam.AlignedSegment()
            a.query_name = r["name"]
            a.flag = r["flag"]
            a.reference_id = r["ref"]
            a.reference_start = r["pos"]
            a.mapping_quality = 60
            a.cigar = [(0, len(r["seq"]))]
            a.query_sequence = r["seq"]
            a.query_qualities = pysam.qualitystring_to_array("".join(chr(33 + int(q)) for q in r["quals"]))
            a.set_tag("MD", str(len(r["seq"])), value_type="Z")
            a.set_tag("mv", r["mv"])
            a.set_tag("ts", r["ts"], value_type="i")
            a.set_tag("ns", r["ns"], value_type="i")
            a.set_tag("sp", 0, value_type="i")
            bam.write(a)
    pysam.sort("-o", os.path.join(out, "reads.bam"), unsorted)
    pysam.index(os.path.join(out, "reads.bam"))
    return reads


@pytest.fixture(autouse=True)
def _no_prep_variables(monkeypatch):
    """The cache and index variables of the Python preprocessing change where it writes; the tests use none."""
    for name in ("DEEPRM_PREP_CACHE_DIR", "DEEPRM_PREP_KEEP_CACHE", "DEEPRM_PREP_SUMMARY_INDEX"):
        monkeypatch.delenv(name, raising=False)


def _fixture_in_child(out, q, kwargs):
    q.put(_write_fixture(out, **kwargs))


def _make_fixture(tmp_path_factory, name, **kwargs):
    out = str(tmp_path_factory.mktemp(name))
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_fixture_in_child, args=(out, q, kwargs))
    p.start()
    reads = q.get(timeout=120)
    p.join(timeout=120)
    assert p.exitcode == 0
    return out, reads


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    return _make_fixture(tmp_path_factory, "prep_fixture")


@pytest.fixture(scope="module")
def multi_ref_fixture(tmp_path_factory):
    """40 references of 60 kb and 400 reads: enough BGZF blocks and index points for the voffset partition."""
    return _make_fixture(tmp_path_factory, "prep_multi_ref", n_files=10, reads_per_file=40, n_refs=40, ref_len=60000)


@pytest.fixture(scope="module")
def degenerate_fixture(tmp_path_factory):
    return _make_fixture(tmp_path_factory, "prep_degenerate", degenerate=True)


def _run(fixture, out, *extra):
    d, _ = fixture
    p = argparse.ArgumentParser()
    py_prep.add_arguments(p)
    args = p.parse_args(["-p", os.path.join(d, "pod5"), "-b", os.path.join(d, "reads.bam"), "-o", str(out), *extra])
    assert py_prep.main(args) == 0
    rows = {f: [] for f in FIELDS}
    for path in sorted(glob.glob(os.path.join(str(out), "*.npz"))):
        with np.load(path) as z:
            assert sorted(z.files) == sorted(FIELDS)
            for f in FIELDS:
                assert z[f].dtype == DTYPES[f], f
                rows[f].append(z[f])
    return {f: np.concatenate(v) for f, v in rows.items()} if rows["label_id"] else None


def _rid(pair):
    return str(uuid.UUID(bytes=np.asarray(pair, dtype=np.int64).tobytes()))


def _digests(rows):
    n = len(rows["label_id"])
    flat = [np.ascontiguousarray(rows[f]).reshape(n, -1).view(np.uint8) for f in FIELDS]
    return sorted(hashlib.blake2b(b"".join(a[i].tobytes() for a in flat), digest_size=16).digest()
                  for i in range(n))


def test_default_run_skips_reverse_and_bad_stride_reads(fixture_dir, tmp_path):
    rows = _run(fixture_dir, tmp_path / "out", "-t", "4", "-a", "1")
    assert rows is not None and len(rows["label_id"]) > 0
    kinds = {fixture_dir[1][_rid(r)][0] for r in rows["read_id"]}
    assert kinds == {"forward"}  # -g 276 skips reverse-strand reads; a stride-5 move table is skipped
    assert (rows["label_id"] > 0).all()
    assert (rows["kmer_token"][:, 10] == ord("A")).all()


def test_reverse_strand_reads_are_taken_in_rna_sense(fixture_dir, tmp_path):
    rows = _run(fixture_dir, tmp_path / "out", "-t", "4", "-a", "1", "-g", "260")
    seen_reverse = 0
    for rid, label, kmer in zip(rows["read_id"], rows["label_id"], rows["kmer_token"]):
        kind, seq = fixture_dir[1][_rid(rid)]
        assert kind != "bad_stride"
        context = kmer.tobytes().decode("ascii")
        assert context[10] == "A"
        if kind == "reverse":
            seen_reverse += 1
            assert label < 0
            assert context in _revcomp(seq)  # the read as basecalled, not the forward reference
        else:
            assert label > 0
            assert context in seq
    assert seen_reverse > 0


def test_parallel_run_gives_the_same_rows(fixture_dir, tmp_path):
    one = _run(fixture_dir, tmp_path / "one", "-t", "1", "-a", "1")
    many = _run(fixture_dir, tmp_path / "many", "-t", "8", "-a", "1")
    assert _digests(one) == _digests(many)


def test_main_process_never_opens_a_pod5_file(fixture_dir, tmp_path, monkeypatch):
    # Workers forked after the main process opened a POD5 file can fail to open theirs ("Failed to make
    # temporary directory"), and each such file is then skipped. Files written by an older pod5 hit this;
    # the fixture's files do not, so the rule itself is checked: every open is recorded with its pid.
    import pod5

    opens = tmp_path / "opens.txt"

    class Recording(pod5.Reader):
        def __init__(self, *a, **k):
            with open(opens, "a") as fh:
                fh.write(f"{os.getpid()}\n")
            super().__init__(*a, **k)

    monkeypatch.setattr(py_prep.pod5, "Reader", Recording)
    rows = _run(fixture_dir, tmp_path / "out", "-t", "4", "-a", "1")
    pids = opens.read_text().split()
    assert rows is not None and len(pids) > 0
    assert str(os.getpid()) not in pids


def test_padding_keeps_more_sites(fixture_dir, tmp_path):
    plain = _run(fixture_dir, tmp_path / "plain", "-t", "2", "-a", "1")
    padded = _run(fixture_dir, tmp_path / "padded", "-t", "2", "-a", "1", "--zero-pad")
    assert len(padded["label_id"]) > len(plain["label_id"])


def test_options_and_defaults():
    p = argparse.ArgumentParser()
    py_prep.add_arguments(p)
    a = p.parse_args(["-p", "x", "-o", "z", "-b", "y.bam", "-t", "3", "-a", "2", "-d", "7", "-f", "9", "-l", "15"])
    assert (a.thread, a.bam_thread, a.label_div, a.dwell_shift, a.cb_len) == (3, 2, 7, 9, 15)
    d = p.parse_args(["-p", "x", "-o", "z"])
    assert d.bam is None and d.qcut == 0 and d.process_once == 1000 and d.filter_flag == 276
    assert not d.zero_pad and not d.edge_pad and not d.trim and d.bam_cmd is None


@pytest.mark.parametrize("extra, message", [
    (["--cb-len", "20"], "--cb-len must be an ODD"),
    (["--kmer-len", "4"], "--kmer-len must be an ODD"),
    (["--sig-window", "7"], "--sig-window"),
    (["--filter-flag", "-1"], "--filter-flag"),
])
def test_bad_geometry_is_refused(tmp_path, extra, message):
    p = argparse.ArgumentParser()
    py_prep.add_arguments(p)
    args = p.parse_args(["-p", str(tmp_path), "-b", "-", "-o", str(tmp_path / "o"), *extra])
    with pytest.raises(ValueError, match=message):
        py_prep.main(args)


def test_missing_bam_is_refused(tmp_path):
    p = argparse.ArgumentParser()
    py_prep.add_arguments(p)
    with pytest.raises(ValueError, match="--bam is required"):
        py_prep.main(p.parse_args(["-p", str(tmp_path), "-o", str(tmp_path / "o")]))


# The wrapper -----------------------------------------------------------------------------------


def _wrapper_args(*extra, bam="in.bam", out="out", pod5="pod5"):
    p = argparse.ArgumentParser()
    wrapper.add_arguments(p)
    return p.parse_args(["-p", pod5, "-b", bam, "-o", out, *extra])


def _binary_long_options():
    src = os.path.join(os.path.dirname(__file__), "..", "cpp", "src", "args", "ArgumentParser.cpp")
    if not os.path.exists(src):
        pytest.skip("cpp/ sources not present")
    import re

    return set(re.findall(r'\{\s*"([a-z0-9-]+)",\s*(?:no|required)_argument', open(src).read()))


def test_binary_options_exist_in_the_binary():
    options = _binary_long_options()
    assert {o.replace("_", "-") for o in wrapper._BINARY_OPTIONS} <= options
    assert "process-once" not in options


def test_binary_gets_only_its_own_options():
    cmd = wrapper._args_to_cmd(_wrapper_args("--zero-pad", "--trim", "--auto-size", "--boi", "u", "-n", "500"))
    passed = {c[2:].replace("-", "_") for c in cmd if c.startswith("--")}
    assert passed <= set(wrapper._BINARY_OPTIONS)
    assert {"zero_pad", "trim"} <= passed
    assert "edge_pad" not in passed and "bam_cmd" not in passed and "process_once" not in passed
    assert "None" not in cmd
    assert cmd[cmd.index("--boi") + 1] == "T"


def test_options_at_their_default_that_older_binaries_lack_are_not_passed():
    """--trim-threshold, --trim-penalty and --max-queue go to the binary only when set (its defaults are the same)."""
    defaults = {c[2:].replace("-", "_") for c in wrapper._args_to_cmd(_wrapper_args()) if c.startswith("--")}
    assert not defaults & {"trim_threshold", "trim_penalty", "max_queue"}
    cmd = wrapper._args_to_cmd(_wrapper_args("--trim-threshold", "7", "--trim-penalty", "-9", "-Q", "10"))
    assert cmd[cmd.index("--trim-threshold") + 1] == "7" and cmd[cmd.index("--trim-penalty") + 1] == "-9"
    assert cmd[cmd.index("--max-queue") + 1] == "10"


def test_bam_dash_next_to_bam_cmd_reaches_the_binary_as_bam_cmd_only():
    cmd = wrapper._args_to_cmd(_wrapper_args("--bam-cmd", "cat x.bam", bam="-"))
    assert "--bam" not in cmd and cmd[cmd.index("--bam-cmd") + 1] == "cat x.bam"


def test_empty_preprocess_bin_is_refused(monkeypatch, calls, tmp_path):
    monkeypatch.setattr(wrapper.subprocess, "run", _no_binary_run)
    with pytest.raises(ValueError, match="--preprocess-bin is empty"):
        wrapper.main(_wrapper_args("--preprocess-bin", "", out=str(tmp_path / "out")))
    assert calls["python"] == 0


def test_an_output_folder_with_npz_files_is_refused(monkeypatch, calls, tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "123-0-0.npz").write_bytes(b"")
    monkeypatch.setattr(wrapper.subprocess, "run", _no_binary_run)
    with pytest.raises(ValueError, match="already holds 1 .npz files"):
        wrapper.main(_wrapper_args("--auto-size", out=str(tmp_path / "out")))
    assert calls["python"] == 0


def test_a_run_that_writes_no_rows_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(wrapper.py_prep, "main", lambda args: 0)
    monkeypatch.setattr(wrapper.subprocess, "run", _no_binary_run)
    with pytest.raises(ValueError, match="wrote no rows"):
        wrapper.main(_wrapper_args("--auto-size", out=str(tmp_path / "out")))


@pytest.fixture
def calls(monkeypatch):
    monkeypatch.delenv("DEEPRM_PREPROCESS_BIN", raising=False)  # the bundled binary, unless a test names one
    seen = {"python": 0}

    def fake_python(args):
        # a successful run writes rows (an empty file stands for them; deeprm call prep fails a run that wrote none)
        seen["python"] += 1
        os.makedirs(args.output, exist_ok=True)
        open(os.path.join(args.output, "0-last.npz"), "wb").close()
        return 0

    monkeypatch.setattr(wrapper.py_prep, "main", fake_python)
    return seen


def _no_binary_run(*a, **k):
    raise AssertionError("the binary must not run")


def test_python_only_option_skips_the_binary(monkeypatch, calls, tmp_path):
    monkeypatch.setattr(wrapper.subprocess, "run", _no_binary_run)
    assert wrapper.main(_wrapper_args("--auto-size", out=str(tmp_path / "out"))) == 0
    assert calls["python"] == 1


def test_single_pod5_file_goes_to_python(monkeypatch, calls, tmp_path):
    one = tmp_path / "reads.pod5"
    one.write_bytes(b"")
    monkeypatch.setattr(wrapper.subprocess, "run", _no_binary_run)
    assert wrapper.main(_wrapper_args(pod5=str(one), out=str(tmp_path / "out"))) == 0
    assert calls["python"] == 1


CANNOT_START = prep_binary.Status(False, reason="the system library libz.so.1 cannot be loaded (file too short)",
                                  hint="Install zlib1g.")
STARTS = prep_binary.Status(True, version="DeepRM Preprocessing v1.2 test")


@pytest.mark.parametrize("bam", ["in.bam", "-"])
def test_binary_that_cannot_start_goes_to_python_and_says_why(monkeypatch, calls, tmp_path, bam, caplog):
    monkeypatch.setattr(wrapper, "_binary_status", lambda path: CANNOT_START)
    monkeypatch.setattr(wrapper.subprocess, "run", _no_binary_run)
    with caplog.at_level("WARNING"):
        assert wrapper.main(_wrapper_args(bam=bam, out=str(tmp_path))) == 0
    assert calls["python"] == 1
    assert "Falling back to Python preprocessing" in caplog.text and "libz.so.1" in caplog.text
    assert "Install zlib1g." in caplog.text


@pytest.mark.parametrize("how", ["option", "environment"])
def test_named_binary_that_cannot_start_stops_the_run(monkeypatch, calls, tmp_path, how):
    missing = tmp_path / "no_such_binary"
    extra = []
    if how == "option":
        extra = ["--preprocess-bin", str(missing)]
    else:
        monkeypatch.setenv("DEEPRM_PREPROCESS_BIN", str(missing))
    monkeypatch.setattr(wrapper.subprocess, "run", _no_binary_run)
    with pytest.raises(ValueError, match="there is no file"):
        wrapper.main(_wrapper_args(*extra, out=str(tmp_path / "out")))
    assert calls["python"] == 0


def test_named_binary_error_names_the_next_choice(monkeypatch, calls, tmp_path):
    monkeypatch.setenv("DEEPRM_PREPROCESS_BIN", "/env/deeprm_preprocess")
    monkeypatch.setattr(wrapper, "_binary_status", lambda path: prep_binary.Status(
        False, reason="the binary is for x86_64 and this machine is aarch64", kind="machine"))
    with pytest.raises(ValueError) as err:
        wrapper.main(_wrapper_args("--preprocess-bin", str(tmp_path / "mine"), out=str(tmp_path / "out")))
    assert "DEEPRM_PREPROCESS_BIN (/env/deeprm_preprocess)" in str(err.value)
    assert "Nothing to fix" not in str(err.value) and "Use a binary that runs on this system" in str(err.value)


def test_relative_binary_path_is_made_absolute(monkeypatch, calls, tmp_path):
    monkeypatch.chdir(tmp_path)
    seen = []
    monkeypatch.setattr(wrapper, "_binary_status", lambda path: seen.append(path) or CANNOT_START)
    with pytest.raises(ValueError):
        wrapper.main(_wrapper_args("--preprocess-bin", "sub/deeprm_preprocess", out="out"))
    assert seen == [os.path.join(str(tmp_path), "sub", "deeprm_preprocess")]


@pytest.mark.parametrize("bam", ["in.bam", "-"])
def test_binary_that_fails_stops_the_run(monkeypatch, calls, tmp_path, bam):
    monkeypatch.setattr(wrapper, "_binary_status", lambda path: STARTS)

    def fails(cmd, check, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(wrapper.subprocess, "run", fails)
    with pytest.raises(ValueError, match="binary failed.*It wrote no output files"):
        wrapper.main(_wrapper_args(bam=bam, out=str(tmp_path)))
    assert calls["python"] == 0


def test_a_failed_binary_with_output_says_which_files_are_incomplete(monkeypatch, calls, tmp_path):
    monkeypatch.setattr(wrapper, "_binary_status", lambda path: STARTS)

    def fails_after_writing(cmd, check, **kwargs):
        (tmp_path / "0-0.npz").write_bytes(b"")
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(wrapper.subprocess, "run", fails_after_writing)
    with pytest.raises(ValueError, match="The 1 .npz files it wrote .* are incomplete"):
        wrapper.main(_wrapper_args(out=str(tmp_path)))


def _is_open(fd):
    try:
        os.fstat(fd)
    except OSError:
        return False
    return True


def test_the_binary_inherits_the_descriptor_that_a_bam_path_names(monkeypatch, calls, tmp_path):
    """-b <(command) gives a path such as /dev/fd/63. The binary can read it only if it inherits that descriptor, and
    subprocess closes every descriptor above 2 that it is not told to pass on."""
    monkeypatch.setattr(wrapper, "_binary_status", lambda path: STARTS)
    passed = []

    def runs(cmd, check, **kwargs):
        passed.append(kwargs.get("pass_fds", ()))
        open(os.path.join(cmd[cmd.index("--output") + 1], "0-0.npz"), "wb").close()

    monkeypatch.setattr(wrapper.subprocess, "run", runs)
    r, w = os.pipe()
    os.close(w)
    unused = next(n for n in range(1000, 2000) if not _is_open(n))
    try:
        for bam, want in [(f"/dev/fd/{r}", (r,)), (f"/proc/self/fd/{r}", (r,)), ("/dev/fd/0", ()), ("/dev/stdin", ()),
                          ("in.bam", ()), (f"/dev/fd/{unused}", ())]:
            out = tmp_path / f"out{len(passed)}"
            out.mkdir()
            assert wrapper.main(_wrapper_args(bam=bam, out=str(out))) == 0
            assert passed[-1] == want, bam
    finally:
        os.close(r)


@pytest.mark.parametrize("extra, message", [
    (["--qcut", "-1"], "--qcut"),
    (["--boi", "X"], "Base of interest"),
    (["--cb-len", "20"], "--cb-len"),
    (["-t", "0"], "--thread"),
    (["--sig-window", "4"], "--sig-window"),
    (["-z", "0"], "--max-token-len"),
    (["-d", "0"], "--label-div"),
    (["-f", "-1"], "--dwell-shift"),
    (["-Q", "0"], "--max-queue"),
])
def test_values_neither_path_runs_are_refused_first(monkeypatch, calls, tmp_path, extra, message):
    monkeypatch.setattr(wrapper.subprocess, "run", _no_binary_run)
    with pytest.raises(ValueError, match=message):
        wrapper.main(_wrapper_args(*extra, out=str(tmp_path / "out")))
    assert calls["python"] == 0


# The bundled binary against the Python preprocessing -----------------------------------------------


def _bundled_binary():
    """The bundled binary; skips the test when it cannot start here, or fails it when DEEPRM_REQUIRE_BINARY=1
    (a release check must not pass by skipping)."""
    path = wrapper.DEFAULT_BIN
    st = prep_binary.status(path)
    if not st.ok:
        message = f"the bundled preprocessing binary cannot start here: {st.reason}"
        if os.environ.get("DEEPRM_REQUIRE_BINARY") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return path


@pytest.mark.parametrize("extra", [[], ["-g", "260"], ["--zero-pad", "--trim"], ["--edge-pad", "-g", "260"]])
def test_binary_and_python_give_the_same_rows(fixture_dir, tmp_path, extra):
    binary = _bundled_binary()
    d, _ = fixture_dir
    py_rows = _run(fixture_dir, tmp_path / "py", "-t", "4", "-a", "1", *extra)
    p = argparse.ArgumentParser()
    wrapper.add_arguments(p)
    args = p.parse_args(["-p", os.path.join(d, "pod5"), "-b", os.path.join(d, "reads.bam"), "-o",
                         str(tmp_path / "bin"), "-t", "4", "-a", "1", *extra])
    subprocess.run([binary] + wrapper._args_to_cmd(args), check=True, capture_output=True)
    bin_rows = {f: [] for f in FIELDS}
    for path in sorted(glob.glob(str(tmp_path / "bin" / "*.npz"))):
        with np.load(path) as z:
            for f in FIELDS:
                assert z[f].dtype == DTYPES[f], f
                bin_rows[f].append(z[f])
    bin_rows = {f: np.concatenate(v) for f, v in bin_rows.items()}

    def order(rows):
        return np.lexsort((rows["label_id"], rows["read_id"][:, 1], rows["read_id"][:, 0]))

    a, b = order(py_rows), order(bin_rows)
    assert len(a) == len(b) > 0
    for f in FIELDS:
        x, y = py_rows[f][a], bin_rows[f][b]
        if f.startswith("dwell"):
            # numpy's float32 log10 is CPU-dependent in the last bits; the model does not use dwell
            assert np.abs(x.astype(np.float64) - y.astype(np.float64)).max() <= 2e-5, f
        else:
            assert x.tobytes() == y.tobytes(), f


@pytest.mark.parametrize("path", ["python", "binary"])
@pytest.mark.parametrize("extra", [[], ["-z", "120"]])
def test_deeprm_call_run_takes_every_shard_of_both_paths(fixture_dir, tmp_path, path, extra):
    """deeprm call run skips a malformed shard (inference_dataloader.shard_problem): no shard that either path
    writes may count as one, also with another --max-token-len."""
    from deeprm.inference.inference_dataloader import NanoporeDatasetIterator
    from deeprm.model.deeprm_v12 import DEFAULT_MAX_SEQ_LEN

    out = tmp_path / path
    if path == "python":
        _run(fixture_dir, out, "-t", "2", "-a", "1", *extra)
    else:
        done = _run_binary(fixture_dir, out, "-t", "2", "-a", "1", *extra)
        assert done.returncode == 0, done.stderr[-2000:]
    paths = sorted(glob.glob(str(out / "*.npz")))
    assert paths
    # the geometry of the v1.2 model, as deeprm call run passes it to the data loader
    shards = list(NanoporeDatasetIterator(paths, max_seq_len=DEFAULT_MAX_SEQ_LEN))
    assert len(shards) == len(paths) and all(shard is not None for shard in shards)
    assert sum(len(shard["label_id"]) for shard in shards) > 0


def _run_binary(fixture, out, *extra, bam=None, bam_cmd=None, stdin=None, timeout=600, preexec_fn=None):
    binary = _bundled_binary()
    d, _ = fixture
    p = argparse.ArgumentParser()
    wrapper.add_arguments(p)
    source = ["--bam-cmd", bam_cmd] if bam_cmd else ["-b", bam or os.path.join(d, "reads.bam")]
    args = p.parse_args(["-p", os.path.join(d, "pod5"), *source, "-o", str(out), *extra])
    return subprocess.run([binary] + wrapper._args_to_cmd(args), stdin=stdin, capture_output=True, text=True,
                          timeout=timeout, preexec_fn=preexec_fn)


def _binary_with(marker, change):
    """The bundled binary when it was built with `change`, which added the message `marker` (bytes) to it; a build
    from older sources lacks it. Skips the test otherwise, or fails it when DEEPRM_REQUIRE_BINARY=1."""
    path = _bundled_binary()
    with open(path, "rb") as fh:
        if marker not in fh.read():
            message = f"the bundled preprocessing binary was built before {change}; rebuild it from cpp/"
            if os.environ.get("DEEPRM_REQUIRE_BINARY") == "1":
                pytest.fail(message)
            pytest.skip(message)
    return path


# Added to the binary together with the reset of an ignored SIGCHLD, the close of the input before the command is
# stopped after a read error, and the reason when a -b path cannot be opened.
_STREAM_STOP_MARKER = b"exit status of the BAM producer is unknown"


def _row_set(folder):
    """Every row of a folder as one bytes object, sorted (the rows, whichever file holds them)."""
    rows = []
    for path in glob.glob(os.path.join(str(folder), "*.npz")):
        with np.load(path) as z:
            cols = [z[f].reshape(len(z["label_id"]), -1) for f in FIELDS]
            rows += [b"".join(c[i].tobytes() for c in cols) for i in range(len(z["label_id"]))]
    return sorted(rows)


def _count_rows(folder):
    n = 0
    for path in glob.glob(os.path.join(str(folder), "*.npz")):
        with np.load(path) as z:
            n += len(z["label_id"])
    return n


def test_binary_with_fewer_threads_than_bam_threads_still_reads_the_bam(fixture_dir, tmp_path):
    ok = _run_binary(fixture_dir, tmp_path / "a", "-t", "4", "-a", "1")
    few = _run_binary(fixture_dir, tmp_path / "b", "-t", "1", "-a", "4")
    assert ok.returncode == 0 and few.returncode == 0
    assert _count_rows(tmp_path / "b") == _count_rows(tmp_path / "a") > 0


def test_binary_fails_on_a_truncated_bam(fixture_dir, tmp_path):
    d, _ = fixture_dir
    data = open(os.path.join(d, "reads.bam"), "rb").read()
    cut = tmp_path / "cut.bam"
    cut.write_bytes(data[: len(data) // 2])
    done = _run_binary(fixture_dir, tmp_path / "out", "-t", "2", "-a", "1", bam=str(cut))
    assert done.returncode != 0


def test_binary_fails_on_empty_standard_input(fixture_dir, tmp_path):
    with open(os.devnull, "rb") as empty:
        done = _run_binary(fixture_dir, tmp_path / "out", "-t", "2", "-a", "1", bam="-", stdin=empty)
    assert done.returncode != 0


def test_binary_fails_on_records_without_md(fixture_dir, tmp_path):
    import pysam

    d, _ = fixture_dir
    no_md = str(tmp_path / "no_md.bam")
    with pysam.AlignmentFile(os.path.join(d, "reads.bam"), "rb") as src, \
            pysam.AlignmentFile(no_md, "wb", template=src) as dst:
        for rec in src:
            rec.set_tag("MD", None)
            dst.write(rec)
    pysam.index(no_md)
    done = _run_binary(fixture_dir, tmp_path / "out", "-t", "2", "-a", "1", bam=no_md)
    assert done.returncode != 0
    assert "MD tag" in done.stdout + done.stderr


def test_binary_fails_when_it_cannot_write_its_output(fixture_dir, tmp_path):
    out = tmp_path / "read_only"
    out.mkdir()
    out.chmod(0o555)
    try:
        if os.access(out, os.W_OK):
            pytest.skip("cannot make a read-only folder here (running as root?)")
        done = _run_binary(fixture_dir, out, "-t", "2", "-a", "1")
    finally:
        out.chmod(0o755)
    assert done.returncode != 0
    assert "output files could not be written" in done.stdout + done.stderr


def test_binary_reads_the_bam_from_standard_input_and_from_a_command(fixture_dir, tmp_path):
    d, _ = fixture_dir
    bam = os.path.join(d, "reads.bam")
    assert _run_binary(fixture_dir, tmp_path / "file", "-t", "2", "-a", "1").returncode == 0
    with open(bam, "rb") as stream:
        assert _run_binary(fixture_dir, tmp_path / "stdin", "-t", "2", "-a", "1", bam="-", stdin=stream).returncode == 0
    assert _run_binary(fixture_dir, tmp_path / "cmd", "-t", "2", "-a", "1", bam_cmd=f"cat {bam}").returncode == 0
    rows = _row_set(tmp_path / "file")
    assert len(rows) > 0
    assert _row_set(tmp_path / "stdin") == rows
    assert _row_set(tmp_path / "cmd") == rows


def test_binary_fails_when_the_bam_command_fails(fixture_dir, tmp_path):
    d, _ = fixture_dir
    # the whole BAM arrives, then the command exits 3: only its exit status can tell the run it failed
    done = _run_binary(fixture_dir, tmp_path / "out", "-t", "2", "-a", "1",
                       bam_cmd=f"cat {os.path.join(d, 'reads.bam')}; exit 3")
    assert done.returncode != 0


# Inputs that must fail the run, on both paths -----------------------------------------------------

_BGZF_EOF = bytes.fromhex("1f8b08040000000000ff0600424302001b0003000000000000000000")


def _bgzf_block_starts(data):
    """The start offsets of the BGZF blocks of a BAM (the BC subfield of each block gives its size)."""
    starts, i = [], 0
    while i < len(data):
        starts.append(i)
        i += struct.unpack_from("<H", data, i + 16)[0] + 1
    return starts


def _bam_without_eof_block(fixture, tmp_path):
    """The fixture BAM cut at its last block boundary: every record is there, but the end-of-file block is not, as
    when its writer was stopped between two blocks."""
    d, _ = fixture
    data = open(os.path.join(d, "reads.bam"), "rb").read()
    starts = _bgzf_block_starts(data)
    assert data[starts[-1]:] == _BGZF_EOF
    cut = tmp_path / "no_eof.bam"
    cut.write_bytes(data[: starts[-1]])
    return str(cut)


def _python_prep(argv, stdin=None, timeout=300):
    """The Python preprocessing in a child process (it reads standard input as file 0)."""
    code = ("import sys, argparse; from deeprm.inference import inference_preprocess_python as m; "
            "p = argparse.ArgumentParser(); m.add_arguments(p); sys.exit(m.main(p.parse_args(sys.argv[1:])) or 0)")
    src = os.path.dirname(os.path.dirname(os.path.dirname(py_prep.__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([src, os.environ.get("PYTHONPATH", "")]))
    for name in ("DEEPRM_PREP_CACHE_DIR", "DEEPRM_PREP_KEEP_CACHE", "DEEPRM_PREP_SUMMARY_INDEX"):
        env.pop(name, None)
    return subprocess.run([sys.executable, "-c", code, *argv], stdin=stdin, capture_output=True, text=True,
                          timeout=timeout, env=env)


@pytest.mark.parametrize("how", ["file", "stdin", "command"])
def test_binary_fails_on_a_bam_cut_at_a_block_boundary(fixture_dir, tmp_path, how):
    cut = _bam_without_eof_block(fixture_dir, tmp_path)
    if how == "file":
        done = _run_binary(fixture_dir, tmp_path / "out", "-t", "2", "-a", "1", bam=cut)
    elif how == "stdin":
        with open(cut, "rb") as stream:
            done = _run_binary(fixture_dir, tmp_path / "out", "-t", "2", "-a", "1", bam="-", stdin=stream)
    else:
        done = _run_binary(fixture_dir, tmp_path / "out", "-t", "2", "-a", "1", bam_cmd=f"cat {cut}")
    assert done.returncode != 0
    assert "end-of-file block" in done.stdout + done.stderr


@pytest.mark.parametrize("how", ["file", "stdin", "command"])
def test_python_fails_on_a_bam_cut_at_a_block_boundary(fixture_dir, tmp_path, how):
    d, _ = fixture_dir
    cut = _bam_without_eof_block(fixture_dir, tmp_path)
    argv = ["-p", os.path.join(d, "pod5"), "-o", str(tmp_path / "out"), "-t", "2", "-a", "1"]
    if how == "file":
        done = _python_prep(argv + ["-b", cut])
    elif how == "stdin":
        # a pipe, as from a basecaller: a regular file on standard input could be seeked
        src = subprocess.Popen(["cat", cut], stdout=subprocess.PIPE)
        done = _python_prep(argv + ["-b", "-"], stdin=src.stdout)
        src.stdout.close()
        src.wait()
    else:
        done = _python_prep(argv + ["--bam-cmd", f"cat {cut}"])
    assert done.returncode != 0
    if how != "file":
        assert "end-of-file block" in done.stdout + done.stderr


def _bam_corrupt_inside(fixture, tmp_path, block=-2):
    """The fixture BAM with bytes changed inside one of its blocks (the last data block by default; 0 is the block
    that holds the header), so reading fails there."""
    d, _ = fixture
    data = bytearray(open(os.path.join(d, "reads.bam"), "rb").read())
    starts = _bgzf_block_starts(bytes(data))
    at = starts[block] + 40
    data[at:at + 20] = bytes(255 - b for b in data[at:at + 20])
    bad = tmp_path / f"corrupt{block}.bam"
    bad.write_bytes(bytes(data))
    return str(bad)


def _left_running(pidfile):
    """Whether the command whose shell wrote its pid to `pidfile` still runs (and stop it if so)."""
    pid = int(open(pidfile).read())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    os.kill(pid, 9)
    return True


@pytest.mark.parametrize("bam_thread", ["1", "4"])
@pytest.mark.parametrize("block", [0, -2], ids=["header", "record"])
@pytest.mark.parametrize("side", ["binary", "python"])
def test_a_bam_command_is_stopped_after_a_read_error(fixture_dir, tmp_path, side, block, bam_thread):
    """The header block or a record block cannot be read while the command keeps running: the run must fail
    quickly and leave no part of the command running, also with decompression threads (the default -a 4)."""
    d, _ = fixture_dir
    pidfile = tmp_path / "producer.pid"
    command = f"echo $$ > {pidfile}; cat {_bam_corrupt_inside(fixture_dir, tmp_path, block=block)}; exec sleep 300"
    start = time.monotonic()
    if side == "binary":
        done = _run_binary(fixture_dir, tmp_path / "out", "-t", "4", "-a", bam_thread, bam_cmd=command, timeout=200)
    else:
        done = _python_prep(["-p", os.path.join(d, "pod5"), "-o", str(tmp_path / "out"), "-t", "4", "-a", bam_thread,
                             "--bam-cmd", command], timeout=200)
    assert done.returncode != 0 and time.monotonic() - start < 120
    assert not _left_running(pidfile)


@pytest.mark.parametrize("block", [0, -2], ids=["header", "record"])
@pytest.mark.parametrize("side, bam_thread", [("binary", "1"), ("binary", "4"), ("python", "4")])
def test_a_bam_command_that_writes_when_it_is_stopped_ends_by_itself(fixture_dir, tmp_path, side, bam_thread, block):
    """After a read error the command gets SIGTERM. One that then still writes must not block on a full pipe until the
    SIGKILL 30 s later: the input is closed first (with the binary's decompression threads, what the command still
    writes is read and thrown away instead), so it ends by itself and writes `ended`."""
    if side == "binary":
        _binary_with(_STREAM_STOP_MARKER, "the close of the input before a failed command is stopped")
    d, _ = fixture_dir
    pidfile, ended = tmp_path / "producer.pid", tmp_path / "ended"
    bad = _bam_corrupt_inside(fixture_dir, tmp_path, block=block)
    command = (f"echo $$ > {pidfile}; trap 'head -c 20000000 /dev/zero; echo yes > {ended}; exit 0' TERM; "
               f"cat {bad}; while :; do sleep 0.1; done")
    if side == "binary":
        done = _run_binary(fixture_dir, tmp_path / "out", "-t", "4", "-a", bam_thread, bam_cmd=command, timeout=200)
    else:
        done = _python_prep(["-p", os.path.join(d, "pod5"), "-o", str(tmp_path / "out"), "-t", "4", "-a", bam_thread,
                             "--bam-cmd", command], timeout=200)
    assert done.returncode != 0
    assert ended.exists(), "the command was killed while it was blocked on the pipe"
    assert not _left_running(pidfile)


def _ignore_sigchld():
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)


def test_binary_fails_when_the_bam_command_fails_under_a_parent_that_ignores_sigchld(fixture_dir, tmp_path):
    """An ignored SIGCHLD survives exec, and then waitpid() cannot read the command's exit status."""
    _binary_with(_STREAM_STOP_MARKER, "the reset of an ignored SIGCHLD")
    d, _ = fixture_dir
    done = _run_binary(fixture_dir, tmp_path / "out", "-t", "2", "-a", "1",
                       bam_cmd=f"cat {os.path.join(d, 'reads.bam')}; exit 3", preexec_fn=_ignore_sigchld)
    assert done.returncode != 0
    assert "exited with status 3" in done.stdout + done.stderr


def _deeprm(argv, timeout=300, **kwargs):
    """`deeprm <argv>` (python -m deeprm) in a child process, with this tree's sources."""
    src = os.path.dirname(os.path.dirname(os.path.dirname(py_prep.__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([src, os.environ.get("PYTHONPATH", "")]))
    for name in ("DEEPRM_PREP_CACHE_DIR", "DEEPRM_PREP_KEEP_CACHE", "DEEPRM_PREP_SUMMARY_INDEX",
                 "DEEPRM_PREPROCESS_BIN"):
        env.pop(name, None)
    return subprocess.run([sys.executable, "-m", "deeprm", *argv], capture_output=True, text=True, timeout=timeout,
                          env=env, **kwargs)


def test_a_failed_binary_fails_the_run_under_a_parent_that_ignores_sigchld(fixture_dir, tmp_path):
    """Under an ignored SIGCHLD, subprocess reports exit code 0 for every child, so a binary that failed after
    writing part of its output would end in a run that exits 0. deeprm resets SIGCHLD when it starts."""
    if prep_binary.on_noexec_mount(str(tmp_path)):
        pytest.skip(f"{tmp_path} does not allow programs to run")
    fake = tmp_path / "fake_preprocess"
    fake.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = --version ]; then echo "DeepRM Preprocessing v0 (test)"; exit 0; fi\n'
        'while [ $# -gt 0 ]; do if [ "$1" = --output ]; then out=$2; fi; shift; done\n'
        'mkdir -p "$out" && : > "$out/0-0.npz"\n'
        'echo "DeepRM Preprocessing FAILED (test)" >&2\n'
        "exit 1\n")
    fake.chmod(0o755)
    d, _ = fixture_dir
    done = _deeprm(["call", "prep", "-p", os.path.join(d, "pod5"), "-b", os.path.join(d, "reads.bam"), "-o",
                    str(tmp_path / "out"), "--preprocess-bin", str(fake)], preexec_fn=_ignore_sigchld)
    assert done.returncode == 2, (done.stdout + done.stderr)[-2000:]
    assert "The preprocessing binary failed (exit code 1)" in done.stderr


@pytest.mark.parametrize("how", ["process substitution", "descriptor 3"])
def test_deeprm_call_prep_reads_a_bam_path_that_names_a_descriptor(fixture_dir, tmp_path, how):
    """-b <(command) and -b /dev/fd/3 (with 3< file) through deeprm call prep and the bundled binary, which must
    inherit the descriptor to read the path."""
    _bundled_binary()
    d, _ = fixture_dir
    bam = os.path.join(d, "reads.bam")
    assert _run_binary(fixture_dir, tmp_path / "file", "-t", "2", "-a", "1").returncode == 0
    prep = " ".join(shlex.quote(a) for a in [sys.executable, "-m", "deeprm", "call", "prep", "-p",
                                             os.path.join(d, "pod5"), "-o", str(tmp_path / "fd"), "-t", "2", "-a", "1"])
    script = f"{prep} -b <(cat {shlex.quote(bam)})" if how == "process substitution" else \
        f"{prep} -b /dev/fd/3 3< {shlex.quote(bam)}"
    src = os.path.dirname(os.path.dirname(os.path.dirname(py_prep.__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([src, os.environ.get("PYTHONPATH", "")]))
    env.pop("DEEPRM_PREPROCESS_BIN", None)
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode == 0, (done.stdout + done.stderr)[-2000:]
    assert "Using the accelerated preprocessing binary" in done.stderr  # not the Python preprocessing
    assert _row_set(tmp_path / "fd") and _row_set(tmp_path / "fd") == _row_set(tmp_path / "file")


@pytest.mark.parametrize("bam", ["-", "/dev/stdin"])
@pytest.mark.parametrize("side", ["binary", "python"])
def test_a_bam_read_from_a_terminal_is_refused(fixture_dir, tmp_path, side, bam):
    """-b - or -b /dev/stdin with nothing piped in would wait for keyboard input, with no message."""
    if side == "binary" and bam != "-":
        _binary_with(b"is a terminal, not a BAM", "the terminal check of a -b path")
    d, _ = fixture_dir
    leader, terminal = os.openpty()
    try:
        if side == "binary":
            done = _run_binary(fixture_dir, tmp_path / "out", "-t", "2", "-a", "1", bam=bam, stdin=terminal,
                               timeout=60)
        else:
            done = _python_prep(["-p", os.path.join(d, "pod5"), "-o", str(tmp_path / "out"), "-t", "2", "-a", "1",
                                 "-b", bam], stdin=terminal, timeout=60)
    finally:
        os.close(terminal)
        os.close(leader)
    assert done.returncode != 0
    assert "is a terminal" in done.stdout + done.stderr


def test_binary_says_why_it_cannot_open_a_bam_path(fixture_dir, tmp_path):
    """A -b path that is not a regular file is first opened by the stream reader; when that fails, say why."""
    _binary_with(_STREAM_STOP_MARKER, "the reason when a -b path cannot be opened")
    fifo = tmp_path / "unreadable"
    os.mkfifo(fifo)
    fifo.chmod(0)
    if os.access(fifo, os.R_OK):
        pytest.skip("cannot make an unreadable named pipe here (running as root?)")
    done = _run_binary(fixture_dir, tmp_path / "out", "-t", "2", "-a", "1", bam=str(fifo), timeout=60)
    assert done.returncode != 0
    assert "Permission denied" in done.stdout + done.stderr


@pytest.mark.parametrize("side", ["binary", "python"])
def test_the_bam_command_runs_in_the_callers_group_and_session(fixture_dir, tmp_path, side):
    """So that Ctrl-C and a hang-up reach it, and it can prompt on the terminal."""
    d, _ = fixture_dir
    own, parent = tmp_path / "own", tmp_path / "parent"
    bam = os.path.join(d, "reads.bam")
    command = f"ps -o pgid=,sid= -p $$ > {own}; ps -o pgid=,sid= -p $PPID > {parent}; cat {bam}"
    if side == "binary":
        done = _run_binary(fixture_dir, tmp_path / "out", "-t", "2", "-a", "1", bam_cmd=command)
    else:
        done = _python_prep(["-p", os.path.join(d, "pod5"), "-o", str(tmp_path / "out"), "-t", "2", "-a", "1",
                             "--bam-cmd", command])
    assert done.returncode == 0, (done.stdout + done.stderr)[-2000:]
    assert own.read_text().split() == parent.read_text().split() != []


@pytest.mark.parametrize("fmt", ["bam", "sam"])
@pytest.mark.parametrize("side", ["binary", "python"])
def test_a_pipe_given_by_its_path_is_read_as_a_stream(fixture_dir, tmp_path, side, fmt):
    """-b /dev/stdin (or <(command), a named pipe) with a real pipe behind it is read as a stream."""
    d, _ = fixture_dir
    source = os.path.join(d, "reads.bam") if fmt == "bam" else _sam_copy(fixture_dir, tmp_path)
    argv = ["-p", os.path.join(d, "pod5"), "-t", "2", "-a", "1"]
    if side == "binary":
        assert _run_binary(fixture_dir, tmp_path / "file", "-t", "2", "-a", "1", bam=source).returncode == 0
    else:
        assert _python_prep(argv + ["-b", source, "-o", str(tmp_path / "file")]).returncode == 0
    src = subprocess.Popen(["cat", source], stdout=subprocess.PIPE)
    if side == "binary":
        done = _run_binary(fixture_dir, tmp_path / "pipe", "-t", "2", "-a", "1", bam="/dev/stdin", stdin=src.stdout)
    else:
        done = _python_prep(argv + ["-b", "/dev/stdin", "-o", str(tmp_path / "pipe")], stdin=src.stdout)
    src.stdout.close()
    src.wait()
    assert done.returncode == 0, (done.stdout + done.stderr)[-2000:]
    assert _row_set(tmp_path / "file") and _row_set(tmp_path / "pipe") == _row_set(tmp_path / "file")
    if side == "python":  # as for -b -, a producer that fails cannot be seen, and the run says so
        assert "fails CANNOT be seen" in done.stderr


def test_python_reads_the_bam_from_standard_input_and_from_a_command(fixture_dir, tmp_path):
    d, _ = fixture_dir
    bam = os.path.join(d, "reads.bam")
    argv = ["-p", os.path.join(d, "pod5"), "-t", "2", "-a", "1"]
    assert _python_prep(argv + ["-b", bam, "-o", str(tmp_path / "file")]).returncode == 0
    with open(bam, "rb") as stream:
        assert _python_prep(argv + ["-b", "-", "-o", str(tmp_path / "stdin")], stdin=stream).returncode == 0
    assert _python_prep(argv + ["--bam-cmd", f"cat {bam}", "-o", str(tmp_path / "cmd")]).returncode == 0
    rows = _row_set(tmp_path / "file")
    assert len(rows) > 0 and _row_set(tmp_path / "stdin") == rows and _row_set(tmp_path / "cmd") == rows
    # the BAM cache is removed after a successful run (its empty folder stays)
    for out in ("file", "stdin", "cmd"):
        assert os.listdir(tmp_path / out / ".deeprm_bamcache") == []


def test_python_fails_when_the_bam_command_fails(fixture_dir, tmp_path):
    d, _ = fixture_dir
    done = _python_prep(["-p", os.path.join(d, "pod5"), "-o", str(tmp_path / "out"), "-t", "2", "-a", "1",
                         "--bam-cmd", f"cat {os.path.join(d, 'reads.bam')}; exit 3"])
    assert done.returncode != 0


def _read_to_end(f):
    data = b""
    while chunk := f.read(1 << 16):
        data += chunk
    return data


def test_the_stream_copy_reads_a_descriptor_above_1023():
    """select() refuses a descriptor of 1024 or more (in a process that holds many open files); the copy must still
    pass on every byte."""
    import resource

    high = 1100
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and hard <= high:
        pytest.skip(f"the open-file limit ({hard}) does not allow descriptor {high}")
    if soft != resource.RLIM_INFINITY and soft <= high:
        resource.setrlimit(resource.RLIMIT_NOFILE, (high + 1, hard))
    try:
        data = _BGZF_EOF * 100
        r, w = os.pipe()
        os.write(w, data)
        os.close(w)
        os.dup2(r, high)
        os.close(r)
        tail = py_prep._StreamTail(high, own_source=True)
        try:
            assert _read_to_end(tail.read_end) == data
            assert tail.ended_cleanly()
        finally:
            tail.read_end.close()
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_any_error_in_the_stream_copy_fails_the_stream(monkeypatch):
    """Not only an OSError: whatever ends the copy early must not look like a clean end of the stream."""

    class OsWithBrokenRead:
        def __getattr__(self, name):
            return getattr(os, name)

        @staticmethod
        def read(fd, n):
            raise ValueError("a failure that is not an OSError")

    r, w = os.pipe()
    os.write(w, _BGZF_EOF)
    os.close(w)
    monkeypatch.setattr(py_prep, "os", OsWithBrokenRead())
    tail = py_prep._StreamTail(r, own_source=True)
    try:
        assert _read_to_end(tail.read_end) == b""
        assert not tail.ended_cleanly() and isinstance(tail.error, ValueError)
    finally:
        tail.read_end.close()


def test_the_stream_ends_count_only_bytes_that_were_passed_on(tmp_path):
    """The checks of the stream's start and end look at the bytes pysam was given, not at bytes read but never
    passed on (here nobody reads the pipe, which holds less than the stream)."""
    stream = tmp_path / "stream"
    stream.write_bytes(_BGZF_EOF * 10000)  # 280 kB, read in one piece; the pipe holds 64 KiB
    tail = py_prep._StreamTail(os.open(stream, os.O_RDONLY), own_source=True)
    try:
        time.sleep(0.5)  # the copy reads the whole file, then waits on the full pipe
        tail.stop()
        assert tail.head == b"" and tail.tail == b""
    finally:
        tail.read_end.close()


def test_binary_fails_on_an_unreadable_pod5_file(fixture_dir, tmp_path):
    d, _ = fixture_dir
    pod5_dir = tmp_path / "pod5"
    shutil.copytree(os.path.join(d, "pod5"), pod5_dir)
    victim = pod5_dir / "1.pod5"
    victim.write_bytes(victim.read_bytes()[: victim.stat().st_size // 2])
    binary = _bundled_binary()
    p = argparse.ArgumentParser()
    wrapper.add_arguments(p)
    args = p.parse_args(["-p", str(pod5_dir), "-b", os.path.join(d, "reads.bam"), "-o", str(tmp_path / "out"),
                         "-t", "2", "-a", "1"])
    done = subprocess.run([binary] + wrapper._args_to_cmd(args), capture_output=True, text=True)
    assert done.returncode != 0
    assert "POD5 files could not be read" in done.stdout + done.stderr


def _prep(which, fixture, pod5_dir, out, *extra):
    """Run the Python preprocessing or the bundled binary on pod5_dir and the fixture's BAM."""
    d, _ = fixture
    argv = ["-p", str(pod5_dir), "-b", os.path.join(d, "reads.bam"), "-o", str(out), "-t", "2", "-a", "1", *extra]
    if which == "python":
        return _python_prep(argv)
    p = argparse.ArgumentParser()
    wrapper.add_arguments(p)
    return subprocess.run([_bundled_binary()] + wrapper._args_to_cmd(p.parse_args(argv)), capture_output=True,
                          text=True)


def _fixture_read_id(fixture, f, i):
    """The read id of read i of POD5 file f (the fixture writes 8 reads per file, in order)."""
    return list(fixture[1])[8 * f + i]


@pytest.mark.parametrize("extra", [[], ["--no-group"]])
def test_python_fails_on_an_unreadable_pod5_file(fixture_dir, tmp_path, extra):
    # As the binary (test above): the file's reads would be missing. With an index the file fails when its read ids
    # are listed, with --no-group when its reads are counted.
    pod5_dir = tmp_path / "pod5"
    shutil.copytree(os.path.join(fixture_dir[0], "pod5"), pod5_dir)
    victim = pod5_dir / "1.pod5"
    victim.write_bytes(victim.read_bytes()[: victim.stat().st_size // 2])
    done = _prep("python", fixture_dir, pod5_dir, tmp_path / "out", *extra)
    assert done.returncode != 0
    assert f"Corrupted POD5 file: {victim}" in done.stderr
    assert "1 POD5 files could not be read" in done.stderr


@pytest.mark.parametrize("which", ["python", "binary"])
def test_prep_fails_on_a_read_whose_sample_count_cannot_be_read(fixture_dir, tmp_path, which):
    # Read 0 of file 1 is a reverse-strand read, which the default -g 276 skips: the binary reads the id and sample
    # count of every read, used or not, and fails ("Row outside batch bounds"); so must the Python preprocessing (it
    # skipped the read and exited 0). The reads table holds the signal row numbers 0..7 of the file's 8 reads as
    # uint64; read 0's is set past the signal table.
    pod5_dir = tmp_path / "pod5"
    shutil.copytree(os.path.join(fixture_dir[0], "pod5"), pod5_dir)
    victim = pod5_dir / "1.pod5"
    data = bytearray(victim.read_bytes())
    signal_rows = b"".join(struct.pack("<Q", i) for i in range(8))
    at = data.find(signal_rows)
    assert at >= 0 and data.find(signal_rows, at + 1) < 0
    data[at:at + 8] = b"\xff" * 8
    victim.write_bytes(bytes(data))
    done = _prep(which, fixture_dir, pod5_dir, tmp_path / "out")
    out = done.stdout + done.stderr
    assert done.returncode != 0
    assert str(victim) in out
    assert uuid.UUID(_fixture_read_id(fixture_dir, 1, 0)).hex in out.replace("-", "")


def _copy_with_unreadable_signal(src, dst, row):
    """Copy POD5 file src to dst; read `row` gets compressed signal bytes that are not zstd data. Its id and sample
    count stay readable, so only reading its signal fails."""
    import pod5

    with pod5.Reader(src) as r, pod5.Writer(dst) as w:
        for i, rec in enumerate(r.reads()):
            read = rec.to_read()
            if i == row:
                read = pod5.CompressedRead(
                    read_id=read.read_id, pore=read.pore, calibration=read.calibration, read_number=read.read_number,
                    start_sample=read.start_sample, median_before=read.median_before, end_reason=read.end_reason,
                    run_info=read.run_info, signal_chunks=[np.full(64, 0xFF, dtype=np.uint8)],
                    signal_chunk_lengths=[len(read.signal)])
            w.add_read(read)


@pytest.mark.parametrize("which", ["python", "binary"])
@pytest.mark.parametrize("row,used", [(3, True), (1, False)])
def test_prep_fails_on_an_unreadable_signal_of_a_read_it_uses(fixture_dir, tmp_path, which, row, used):
    # The binary reads the signal only of the reads that pass the filters, so an unreadable signal fails the run for
    # read 3 of file 1 (forward) and not for read 1 (its move-table stride is 5, so it is skipped). The Python
    # preprocessing must do the same; it skipped read 3 and exited 0.
    pod5_dir = tmp_path / "pod5"
    shutil.copytree(os.path.join(fixture_dir[0], "pod5"), pod5_dir)
    victim = pod5_dir / "1.pod5"
    victim.unlink()
    p = mp.get_context("fork").Process(target=_copy_with_unreadable_signal,
                                       args=(os.path.join(fixture_dir[0], "pod5", "1.pod5"), str(victim), row))
    p.start()     # in a child: this process must not open a POD5 file (see the module docstring)
    p.join(timeout=120)
    assert p.exitcode == 0
    done = _prep(which, fixture_dir, pod5_dir, tmp_path / "out")
    if used:
        out = done.stdout + done.stderr
        assert done.returncode != 0
        assert str(victim) in out
        assert uuid.UUID(_fixture_read_id(fixture_dir, 1, row)).hex in out.replace("-", "")
    else:
        assert done.returncode == 0, done.stderr
        clean = _prep(which, fixture_dir, os.path.join(fixture_dir[0], "pod5"), tmp_path / "clean")
        assert clean.returncode == 0
        assert _row_set(tmp_path / "out") == _row_set(tmp_path / "clean")


def _write_pod5_without_reads(path):
    import pod5

    with pod5.Writer(path):
        pass


@pytest.mark.parametrize("which,extra", [("python", []), ("python", ["--no-group"]), ("binary", [])])
def test_prep_reads_a_pod5_file_without_reads_as_empty(fixture_dir, tmp_path, which, extra):
    # A valid POD5 file with no reads is not an unreadable one: the binary reads it as 0 reads and exits 0. pod5 raises
    # on its read ids and read count, so the Python preprocessing must not take that for a damaged file.
    pod5_dir = tmp_path / "pod5"
    shutil.copytree(os.path.join(fixture_dir[0], "pod5"), pod5_dir)
    p = mp.get_context("fork").Process(target=_write_pod5_without_reads, args=(str(pod5_dir / "empty.pod5"),))
    p.start()     # in a child: this process must not open a POD5 file (see the module docstring)
    p.join(timeout=120)
    assert p.exitcode == 0
    done = _prep(which, fixture_dir, pod5_dir, tmp_path / "out", *extra)
    assert done.returncode == 0, done.stderr
    clean = _prep(which, fixture_dir, os.path.join(fixture_dir[0], "pod5"), tmp_path / "clean", *extra)
    assert clean.returncode == 0
    assert _row_set(tmp_path / "out") == _row_set(tmp_path / "clean")


@pytest.mark.parametrize("broken", ["read_ids", "reads"])
def test_python_fails_when_a_pod5_file_cannot_be_read_in_phase_2(fixture_dir, tmp_path, monkeypatch, capfd, broken):
    # A file that could be counted can still fail when a signal worker lists its reads or reads a batch of them; the
    # worker then stops the run instead of skipping the file. (--no-group: no index, which also lists the reads.)
    import pod5

    class Failing(pod5.Reader):
        def _fail(self, what):
            if what == broken and os.path.basename(str(self.path)) == "1.pod5":
                raise RuntimeError("Invalid: made to fail by the test")

        @property
        def read_ids(self):
            self._fail("read_ids")
            return super().read_ids

        def reads(self, *a, **k):
            self._fail("reads")
            return super().reads(*a, **k)

    monkeypatch.setattr(py_prep.pod5, "Reader", Failing)
    d, _ = fixture_dir
    p = argparse.ArgumentParser()
    py_prep.add_arguments(p)
    args = p.parse_args(["-p", os.path.join(d, "pod5"), "-b", os.path.join(d, "reads.bam"), "-o", str(tmp_path / "out"),
                         "-t", "2", "-a", "1", "--no-group"])
    with pytest.raises(RuntimeError, match="Signal segmentation/normalization failed"):
        py_prep.main(args)
    assert f"Corrupted POD5 file: {os.path.join(d, 'pod5', '1.pod5')} (Invalid: made to fail by the test)" \
        in capfd.readouterr().err


def _sam_copy(fixture, tmp_path):
    import pysam

    d, _ = fixture
    sam = str(tmp_path / "reads.sam")
    with pysam.AlignmentFile(os.path.join(d, "reads.bam"), "rb") as src, \
            pysam.AlignmentFile(sam, "w", template=src) as dst:
        for rec in src:
            dst.write(rec)
    return sam


def test_binary_reads_a_sam_file(fixture_dir, tmp_path):
    """The 1.1.3 binary read a SAM file given with -b; this binary reads it as a stream, with the BAM's rows."""
    sam = _sam_copy(fixture_dir, tmp_path)
    assert _run_binary(fixture_dir, tmp_path / "bam", "-t", "2", "-a", "1").returncode == 0
    done = _run_binary(fixture_dir, tmp_path / "sam", "-t", "2", "-a", "1", bam=sam)
    assert done.returncode == 0 and "reading it as a stream" in done.stdout + done.stderr
    assert _row_set(tmp_path / "sam") == _row_set(tmp_path / "bam") != []


def test_both_paths_read_a_sam_stream(fixture_dir, tmp_path):
    """A SAM text stream has no BGZF end-of-file block; the end-of-file check must not refuse it."""
    d, _ = fixture_dir
    sam = _sam_copy(fixture_dir, tmp_path)
    argv = ["-p", os.path.join(d, "pod5"), "-t", "2", "-a", "1"]
    # each path against its own file-mode rows (the dwell tokens of the two paths differ in the last bits)
    assert _run_binary(fixture_dir, tmp_path / "bin_bam", "-t", "2", "-a", "1").returncode == 0
    assert _run_binary(fixture_dir, tmp_path / "bin_sam", "-t", "2", "-a", "1", bam_cmd=f"cat {sam}").returncode == 0
    assert _python_prep(argv + ["-b", os.path.join(d, "reads.bam"), "-o", str(tmp_path / "py_bam")]).returncode == 0
    done = _python_prep(argv + ["--bam-cmd", f"cat {sam}", "-o", str(tmp_path / "py_sam")])
    assert done.returncode == 0, done.stderr[-2000:]
    assert _row_set(tmp_path / "bin_bam") and _row_set(tmp_path / "bin_sam") == _row_set(tmp_path / "bin_bam")
    assert _row_set(tmp_path / "py_bam") and _row_set(tmp_path / "py_sam") == _row_set(tmp_path / "py_bam")


def _cut_sam(fixture, tmp_path, where):
    """The fixture's SAM text cut off. "mv", "seq" and "qual" end inside that field of the last record, "newline"
    ends just before the last record's newline (the record is whole, its newline missing); the last record is the
    last forward read, so its rows count. "line_end" ends right after the newline of the middle record, so it holds
    only whole records. Returns the path and the names of the records it holds in full."""
    _, reads = fixture
    lines = open(_sam_copy(fixture, tmp_path), "rb").read().split(b"\n")[:-1]
    head = [line for line in lines if line.startswith(b"@")]
    records = [line for line in lines if not line.startswith(b"@")]
    if where == "line_end":
        records, cut = records[: len(records) // 2], b""
    else:
        last = max(i for i, r in enumerate(records) if reads[r.split(b"\t")[0].decode()][0] == "forward")
        records, fields = records[: last + 1], records[last].split(b"\t")
        if where == "newline":
            cut = records[-1]
        else:
            i = {"seq": 9, "qual": 10}.get(where) or next(k for k, f in enumerate(fields) if f.startswith(b"mv:B:"))
            cut = b"\t".join(fields[:i] + [fields[i][: len(fields[i]) // 3]])
        records = records[:-1]
    path = tmp_path / f"cut_{where}.sam"
    path.write_bytes(b"".join(line + b"\n" for line in head + records) + cut)
    names = [r.split(b"\t")[0].decode() for r in records + ([cut] if where == "newline" else [])]
    return str(path), names


def _prep_reading(fixture, side, source, how, out):
    """The binary or the Python preprocessing reading `source` as a file (-b), from a pipe on standard input (-b -)
    or from a command (--bam-cmd)."""
    d, _ = fixture
    argv = ["-p", os.path.join(d, "pod5"), "-o", str(out), "-t", "2", "-a", "1"]
    src = subprocess.Popen(["cat", source], stdout=subprocess.PIPE) if how == "stdin" else None
    if side == "binary":
        bam, cmd = {"file": (source, None), "stdin": ("-", None), "command": (None, f"cat {source}")}[how]
        done = _run_binary(fixture, out, "-t", "2", "-a", "1", bam=bam, bam_cmd=cmd,
                           stdin=src.stdout if src else None)
    else:
        done = _python_prep(argv + {"file": ["-b", source], "stdin": ["-b", "-"],
                                    "command": ["--bam-cmd", f"cat {source}"]}[how],
                            stdin=src.stdout if src else None)
    if src:
        src.stdout.close()
        src.wait()
    return done


def _binary_with_the_sam_end_check():
    """The bundled binary when it checks that a SAM text ends with a newline (a build from older sources reads a cut
    one without an error); skips the test otherwise, or fails it when DEEPRM_REQUIRE_BINARY=1."""
    return _binary_with(b"does not end with a newline", "the SAM end check")


@pytest.mark.parametrize("how", ["file", "stdin", "command"])
@pytest.mark.parametrize("where", ["mv", "seq", "qual", "newline"])
@pytest.mark.parametrize("side", ["binary", "python"])
def test_a_sam_cut_inside_its_last_record_fails_the_run(fixture_dir, tmp_path, side, where, how):
    """SAM text has no end-of-file marker, and htslib reads a record cut inside its move table without an error, with
    a shortened move table (wrong rows). A last line without its newline fails the run; a cut inside SEQ or QUAL
    is also a parse error, which a stream reaches before its end."""
    if side == "binary" and where in ("mv", "newline"):
        _binary_with_the_sam_end_check()
    cut, _ = _cut_sam(fixture_dir, tmp_path, where)
    done = _prep_reading(fixture_dir, side, cut, how, tmp_path / "out")
    assert done.returncode != 0
    # the Python preprocessing checks a SAM file's end before it reads the file; otherwise the parse error comes first
    if where in ("mv", "newline") or (how == "file" and side == "python"):
        assert "does not end with a newline" in done.stdout + done.stderr


@pytest.mark.parametrize("side", ["binary", "python"])
def test_a_sam_cut_at_a_line_end_is_read(fixture_dir, tmp_path, side):
    """A SAM cut right after a newline holds only whole records: each form reads it, with the rows those records give
    in the whole BAM."""
    if side == "binary":
        _bundled_binary()
    d, _ = fixture_dir
    whole = _prep_reading(fixture_dir, side, os.path.join(d, "reads.bam"), "file", tmp_path / "whole")
    assert whole.returncode == 0
    cut, names = _cut_sam(fixture_dir, tmp_path, "line_end")
    kept = {uuid.UUID(name).bytes for name in names}
    want = [row for row in _row_set(tmp_path / "whole") if row[:16] in kept]   # read_id is the first field
    assert 0 < len(want) < len(_row_set(tmp_path / "whole"))
    for how in ("file", "stdin", "command"):
        done = _prep_reading(fixture_dir, side, cut, how, tmp_path / how)
        assert done.returncode == 0, (done.stdout + done.stderr)[-2000:]
        assert _row_set(tmp_path / how) == want, how


def test_build_sidecar_through_deeprm_call_prep(fixture_dir, tmp_path):
    """--build-sidecar writes no rows on purpose; the no-rows check must not fail it."""
    d, _ = fixture_dir
    bam = tmp_path / "reads.bam"
    shutil.copyfile(os.path.join(d, "reads.bam"), bam)
    shutil.copyfile(os.path.join(d, "reads.bam.bai"), str(bam) + ".bai")
    args = _wrapper_args("--build-sidecar", pod5=os.path.join(d, "pod5"), bam=str(bam), out=str(tmp_path / "out"))
    assert wrapper.main(args) == 0


def test_named_binary_without_execute_permission_gets_it_and_runs(fixture_dir, tmp_path):
    binary = _bundled_binary()
    if prep_binary.on_noexec_mount(str(tmp_path)):
        pytest.skip(f"{tmp_path} does not allow programs to run")
    d, _ = fixture_dir
    copy = tmp_path / "deeprm_preprocess"
    shutil.copyfile(binary, copy)
    copy.chmod(0o644)
    args = _wrapper_args("--preprocess-bin", str(copy), "-t", "2", "-a", "1", pod5=os.path.join(d, "pod5"),
                         bam=os.path.join(d, "reads.bam"), out=str(tmp_path / "out"))
    assert wrapper.main(args) == 0
    assert os.access(copy, os.X_OK) and _count_rows(tmp_path / "out") > 0


# Read shapes and layouts the default fixture does not have -------------------------------------------


def _binary_rows(fixture, out, *extra):
    """Run the bundled binary as deeprm call prep would and load its rows."""
    binary = _bundled_binary()
    d, _ = fixture
    p = argparse.ArgumentParser()
    wrapper.add_arguments(p)
    args = p.parse_args(["-p", os.path.join(d, "pod5"), "-b", os.path.join(d, "reads.bam"), "-o", str(out), *extra])
    subprocess.run([binary] + wrapper._args_to_cmd(args), check=True, capture_output=True)
    rows = {f: [] for f in FIELDS}
    for path in sorted(glob.glob(str(out / "*.npz"))):
        with np.load(path) as z:
            for f in FIELDS:
                assert z[f].dtype == DTYPES[f], f
                rows[f].append(z[f])
    return {f: np.concatenate(v) for f, v in rows.items()}


def _assert_same_rows(py_rows, bin_rows):
    def order(rows):
        return np.lexsort((rows["label_id"], rows["read_id"][:, 1], rows["read_id"][:, 0]))

    a, b = order(py_rows), order(bin_rows)
    assert len(a) == len(b) > 0
    for f in FIELDS:
        x, y = py_rows[f][a], bin_rows[f][b]
        if f.startswith("dwell"):
            # numpy's float32 log10 and the C library's round some values differently; the model does not use dwell
            assert np.abs(x.astype(np.float64) - y.astype(np.float64)).max() <= 2e-5, f
        else:
            assert x.tobytes() == y.tobytes(), f


def test_degenerate_reads_are_dropped_by_both_paths(degenerate_fixture, tmp_path):
    d, reads = degenerate_fixture
    # both paths count the reads whose move table is longer than their signal
    done = _run_binary(degenerate_fixture, tmp_path / "bin_log", "-t", "2", "-a", "1")
    assert done.returncode == 0 and "reads whose move table is longer than their signal" in done.stdout + done.stderr
    done = _python_prep(["-p", os.path.join(d, "pod5"), "-b", os.path.join(d, "reads.bam"), "-o",
                         str(tmp_path / "py_log"), "-t", "2", "-a", "1"])
    assert done.returncode == 0 and "reads whose move table is longer than their signal" in done.stdout + done.stderr
    bin_rows = _binary_rows(degenerate_fixture, tmp_path / "bin", "-t", "2", "-a", "1")
    py_rows = _run(degenerate_fixture, tmp_path / "py", "-t", "2", "-a", "1")
    dropped = {r for r, v in reads.items() if v[0] in DEGENERATE.values()}
    assert len(dropped) == len(DEGENERATE)
    assert not {_rid(pair) for pair in py_rows["read_id"]} & dropped
    _assert_same_rows(py_rows, bin_rows)


def test_voffset_partition_gives_the_rows_of_the_modulo_scan(multi_ref_fixture, tmp_path, caplog):
    d, reads = multi_ref_fixture
    with caplog.at_level("INFO"):
        rows = _run(multi_ref_fixture, tmp_path / "ranges", "-t", "8", "-a", "1")
    assert "voffset partition" in caplog.text
    # the same BAM without its index is read by the modulo scan
    plain = tmp_path / "plain"
    plain.mkdir()
    shutil.copyfile(os.path.join(d, "reads.bam"), plain / "reads.bam")
    p = argparse.ArgumentParser()
    py_prep.add_arguments(p)
    caplog.clear()
    with caplog.at_level("INFO"):
        assert py_prep.main(p.parse_args(["-p", os.path.join(d, "pod5"), "-b", str(plain / "reads.bam"),
                                          "-o", str(tmp_path / "modulo"), "-t", "8", "-a", "1"])) == 0
    assert "voffset partition" not in caplog.text
    assert _row_set(tmp_path / "modulo") == _row_set(tmp_path / "ranges")
    # label_id = (reference index * label_div + position + 1) * strand: every row names its read's reference
    label_div = 10**9
    ref_of = {rid: v[2] for rid, v in reads.items()}
    refs = (np.abs(rows["label_id"]) - 1) // label_div
    assert len(set(refs.tolist())) > 1
    assert all(ref_of[_rid(pair)] == ref for pair, ref in zip(rows["read_id"], refs))
    assert ((np.abs(rows["label_id"]) - 1) % label_div < 60000).all()


def test_binary_and_python_agree_on_many_references(multi_ref_fixture, tmp_path):
    bin_rows = _binary_rows(multi_ref_fixture, tmp_path / "bin", "-t", "8", "-a", "1")
    _assert_same_rows(_run(multi_ref_fixture, tmp_path / "py", "-t", "8", "-a", "1"), bin_rows)


def test_bucketed_pod5_groups_give_the_same_rows(fixture_dir, tmp_path, monkeypatch, caplog):
    rows = _row_set_of_run(fixture_dir, tmp_path / "groups")
    monkeypatch.setattr(py_prep, "_MAX_GROUPS", 2)
    with caplog.at_level("WARNING"):
        bucketed = _row_set_of_run(fixture_dir, tmp_path / "buckets")
    assert "Bucketing" in caplog.text
    assert bucketed == rows and len(rows) > 0


def _row_set_of_run(fixture, out):
    _run(fixture, out, "-t", "4", "-a", "1")
    return _row_set(out)
