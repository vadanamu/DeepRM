"""The memory bounds of deeprm call pileup change no output, and the container limits are read from the cgroup."""

import errno
import os
import resource
import signal
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pysam
import pytest
from test_pileup_modbam_merge import LABEL_DIV, _args, _write_input_bam, _write_per_read_files

from deeprm.inference import pileup_deeprm
from deeprm.inference.pileup_genomic import pileup_genomic
from deeprm.utils import utils


def _random_tables(n_files, seed, n_sites=2000):
    """Per-file site tables as the workers make them: sorted unique label_ids, int64 counts, and float sums whose
    per-file values are float32 (as the per-read terms are), spread over many orders of magnitude so that the order
    of the additions shows in the last bits."""
    rng = np.random.default_rng(seed)
    ids_list, sums_list = [], []
    for _ in range(n_files):
        ids = np.unique(rng.integers(-n_sites, n_sites, size=int(rng.integers(0, 400)))).astype(np.int64)
        ids = ids[ids != 0]
        sums = {}
        for key, dtype in pileup_deeprm.SITE_SUMS:
            if dtype is np.int64:
                sums[key] = rng.integers(0, 50, len(ids)).astype(np.int64)
            else:
                sums[key] = (rng.standard_normal(len(ids)) * 10.0 ** rng.integers(-8, 8, len(ids))).astype(np.float32)
        ids_list.append(ids)
        sums_list.append(sums)
    return ids_list, sums_list


def _assert_same_table(a, b):
    """Two (label_id, sums) tables are the same to the bit."""
    assert np.array_equal(a[0], b[0]) and a[0].dtype == b[0].dtype
    for key, dtype in pileup_deeprm.SITE_SUMS:
        x, y = a[1][key], b[1][key]
        assert x.dtype == y.dtype == dtype, key
        assert x.tobytes() == y.tobytes(), key


@pytest.mark.parametrize("min_sites", [1, 7, 300, 10**9])
def test_running_merge_is_the_merge_of_all_files_to_the_bit(min_sites):
    ids_list, sums_list = _random_tables(60, seed=min_sites)
    acc = pileup_deeprm._SiteSums(min_sites=min_sites)
    for ids, sums in zip(ids_list, sums_list):
        acc.add(ids, sums)
    _assert_same_table(acc.result(), pileup_deeprm._merge_site_sums(ids_list, sums_list))


def test_running_merge_of_no_file_is_an_empty_table():
    label_id, sums = pileup_deeprm._SiteSums().result()
    assert label_id.size == 0 and label_id.dtype == np.int64
    assert {key: arr.dtype for key, arr in sums.items()} == dict(pileup_deeprm.SITE_SUMS)


def test_merge_of_saved_worker_tables_is_the_merge_of_all_to_the_bit(tmp_path):
    ids_list, sums_list = _random_tables(60, seed=11)
    stems, tables = [], []
    for w in range(5):
        acc = pileup_deeprm._SiteSums(min_sites=3)
        if w != 2:  # worker 2 has no valid row: an empty table
            for f in range(w, 60, 5):
                acc.add(ids_list[f], sums_list[f])
        tables.append(acc.result())
        stem = str(tmp_path / f"sums_{w}")
        pileup_deeprm._save_table(stem, *tables[-1])
        stems.append(stem)
    expected = pileup_deeprm._merge_site_sums([t[0] for t in tables], [t[1] for t in tables])
    _assert_same_table(pileup_deeprm._merge_worker_tables(stems), expected)


def _refflat(path, n_refs):
    """One annotated transcript per reference, tx1 ... txN, each one exon of 2000 bases on chr1 or chr2."""
    with open(path, "w") as f:
        for r in range(n_refs):
            chrom, strand, start = f"chr{1 + r % 2}", "+-"[r % 2], 10000 * (r + 1)
            f.write(f"g{r}\ttx{r + 1}\t{chrom}\t{strand}\t{start}\t{start + 2000}\t{start}\t{start + 2000}\t1\t")
            f.write(f"{start},\t{start + 2000},\n")


def _outputs(folder):
    """Every output file of a pileup, as bytes; the npz as its arrays (a zip records file times), the modBAM as its
    header and records (samtools merge writes its own command line, with the temporary shard paths, into the header,
    so the header without that line), and of the modBAM index only that it is there (its offsets follow the
    header's length)."""
    out = {}
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if name.endswith(".npz"):
            with np.load(path) as z:
                out[name] = {k: (z[k].dtype.str, z[k].tobytes()) for k in z.files}
        elif name.endswith(".bam"):
            with pysam.AlignmentFile(path, "rb") as bam:
                header = [line for line in str(bam.header).splitlines() if not line.startswith("@PG\tID:samtools")]
                out[name] = (header, [r.to_string() for r in bam])
        elif name.endswith(".bai"):
            out[name] = os.path.getsize(path) > 0
        else:
            with open(path, "rb") as f:
                out[name] = f.read()
    return out


def test_pileup_outputs_do_not_depend_on_the_memory_bounds(tmp_path, monkeypatch):
    """Running merges after every file, a modBAM run every few rows and modBAM slices of one reference give the same
    files as one merge, one run and one slice per worker."""
    bam = tmp_path / "input.bam"
    reads = _write_input_bam(bam, n_reads=90, seed=3, n_refs=7)
    inp = tmp_path / "molecule-level"
    inp.mkdir()
    _write_per_read_files(inp, reads, n_files=9)
    annot = tmp_path / "annot.refflat"
    _refflat(annot, 7)

    results = []
    for k, (sites, run_rows, slice_rows) in enumerate([(10**9, 10**9, 10**9), (1, 7, 1), (5, 50, 30)]):
        monkeypatch.setattr(pileup_deeprm, "_MERGE_MIN_SITES", sites)
        monkeypatch.setattr(pileup_deeprm, "_MODBAM_RUN_ROWS", run_rows)
        monkeypatch.setattr(pileup_deeprm, "_MODBAM_SLICE_ROWS", slice_rows)
        out = tmp_path / f"site-level-{k}"
        args = _args(inp, out, bam, thread=3)
        args.annot = str(annot)
        pileup_deeprm.main(args)
        results.append(_outputs(out))
    assert "genomic_pileup.bed" in results[0] and "modbam_input.bam" in results[0]
    assert not [n for n in results[0] if n.startswith(".deeprm_")], "the temporary folders are removed"
    assert results[1] == results[0]
    assert results[2] == results[0]


def test_modbam_windows_of_one_large_reference_give_the_records_of_one_pass(tmp_path, monkeypatch):
    """One reference (as a chromosome), read in windows of a few rows: reads crossing window starts, reads that tie on
    position, and a read whose supplementary alignment far away has no row in its own span (it is written with the
    tags of no row, as in one pass) give the same records as a single pass."""
    import uuid

    rng = np.random.default_rng(9)
    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 100000}]}
    )
    records, rows = [], []
    for i in range(120):
        name = str(uuid.UUID(int=int(rng.integers(1, 2**62)) << 64 | i))
        start = int(rng.integers(0, 40)) * 25  # ties, and reads crossing any window start
        seq = "".join(rng.choice(list("ACGT"), 30))
        records.append((start, name, seq, 0))
        hi, lo = np.frombuffer(uuid.UUID(name).bytes, dtype=np.int64)
        rows += [(start + q + 1, hi, lo) for q, base in enumerate(seq) if base == "A"]
    # one read with a supplementary alignment at 50000, and rows only from its primary alignment at 10
    name = str(uuid.UUID(int=7 << 64 | 999))
    seq = "A" * 30
    records += [(10, name, seq, 0), (50000, name, seq, 2048)]
    hi, lo = np.frombuffer(uuid.UUID(name).bytes, dtype=np.int64)
    rows += [(10 + q + 1, hi, lo) for q in range(30)]
    records.sort()
    bam = tmp_path / "input.bam"
    with pysam.AlignmentFile(str(bam), "wb", header=header) as out:
        for start, qname, s, flag in records:
            a = pysam.AlignedSegment(header)
            a.query_name, a.flag, a.reference_id, a.reference_start = qname, flag, 0, start
            a.mapping_quality, a.cigartuples, a.query_sequence = 60, [(0, 30)], s
            a.query_qualities = pysam.qualitystring_to_array("I" * 30)
            out.write(a)
    pysam.index(str(bam))
    inp = tmp_path / "molecule-level"
    inp.mkdir()
    order = rng.permutation(len(rows))
    for f, part in enumerate(np.array_split(order, 5)):
        np.savez(
            inp / f"inference_0_{f}.npz",
            label_id=np.array([rows[i][0] for i in part], dtype=np.int64),
            read_id=np.array([[rows[i][1], rows[i][2]] for i in part], dtype=np.int64),
            pred=rng.uniform(0, 1, len(part)).astype(np.float32),
        )
    results = []
    for k, (run_rows, slice_rows) in enumerate([(10**9, 10**9), (37, 23)]):
        monkeypatch.setattr(pileup_deeprm, "_MODBAM_RUN_ROWS", run_rows)
        monkeypatch.setattr(pileup_deeprm, "_MODBAM_SLICE_ROWS", slice_rows)
        out = tmp_path / f"site-level-{k}"
        pileup_deeprm.main(_args(inp, out, bam, thread=2))
        results.append(_outputs(out))
    with pysam.AlignmentFile(str(tmp_path / "site-level-1" / "modbam_input.bam"), "rb") as f:
        supplementary = [r for r in f if r.is_supplementary]
    assert len(supplementary) == 1 and supplementary[0].get_tag("MM") == "A+a?;"
    assert results[1] == results[0]


def test_saved_runs_carry_the_first_and_last_position_of_each_part(tmp_path):
    """keys_span: for each part of a saved run (a run of rows of one (reference, read) in one file), the first and last
    position of its rows; the rows are saved sorted by reference and the first position of their part. The runs of a
    worker go into the same few files, each run read back as its own."""
    rows = pileup_deeprm._ReadRows()
    read_id = np.array([[1, 2], [1, 2], [3, 4], [1, 2]], dtype=np.int64)
    rows.add(np.array([0, 0, 0, 1]), read_id, np.array([50, 10, 7, 99]), np.zeros(4, np.uint8))
    rows.add(np.array([0]), np.array([[1, 2]], dtype=np.int64), np.array([70]), np.zeros(1, np.uint8))
    prefix = str(tmp_path / "worker")
    stem = rows.save_by_reference(prefix)
    rows.add(np.array([2, 2]), np.array([[5, 6], [5, 6]], dtype=np.int64), np.array([4, 3]), np.ones(2, np.uint8))
    stem2 = rows.save_by_reference(prefix)
    assert rows.save_by_reference(prefix) is None  # nothing added since: nothing saved
    ref, keys, span = (pileup_deeprm._saved(stem, name) for name in ("keys_ref", "keys", "keys_span"))
    got = [(int(r), int(k[0]), int(k[1]), int(s[0]), int(s[1])) for r, k, s in zip(ref, keys, span)]
    assert got == [(0, 1, 2, 10, 50), (0, 1, 2, 70, 70), (0, 3, 4, 7, 7), (1, 1, 2, 99, 99)]
    # the parts in the order of their rows: reference, first position and read id; and the first row of each
    part_key = [[0, 7, 3, 4], [0, 10, 1, 2], [0, 70, 1, 2], [1, 99, 1, 2]]
    assert pileup_deeprm._saved(stem, "part_key").tolist() == part_key
    assert pileup_deeprm._saved(stem, "part_at").tolist() == [0, 1, 3, 4]
    assert pileup_deeprm._saved(stem, "rows")["pos"].tolist() == [7, 50, 10, 70, 99]
    assert pileup_deeprm._saved(stem2, "rows")["pos"].tolist() == [4, 3]
    assert pileup_deeprm._saved(stem2, "keys_span").tolist() == [[3, 4, 2]]  # first, last, rows
    # where each reference's rows and parts lie, from the runs' indexes
    assert [pileup_deeprm._bounds(stem, r, r + 1) for r in (0, 1, 2)] == [(0, 4), (4, 5), (5, 5)]
    assert [pileup_deeprm._bounds(stem, r, r + 1, keys=True) for r in (0, 1, 2)] == [(0, 3), (3, 4), (4, 4)]
    assert pileup_deeprm._bounds(stem, 0, 2) == (0, 5) and pileup_deeprm._bounds(stem2, 2, 3) == (0, 2)
    slices = pileup_deeprm._ref_slices([stem, stem2], np.array([0, 1, 2]), 10)
    assert [(refs.tolist(), n) for refs, n in slices] == [([0, 1, 2], 7)]
    assert [n for _, n in pileup_deeprm._ref_slices([stem, stem2], np.array([0, 1, 2]), 4)] == [4, 3]
    assert sorted(os.listdir(tmp_path)) == ["worker." + n for n in sorted(pileup_deeprm._SAVED_FILES)]


def test_saved_maps_stay_within_the_open_file_limit(tmp_path, monkeypatch):
    """A memory map holds a file descriptor: with a low open-file limit that cannot be raised, _saved keeps only the
    maps the free descriptors allow (the least recently used are dropped) and still reads every run right."""
    monkeypatch.setattr(pileup_deeprm.resource, "getrlimit", lambda _: (316, 316))
    monkeypatch.setattr(pileup_deeprm, "_SAVED_MAPS", type(pileup_deeprm._SAVED_MAPS)())
    monkeypatch.setattr(pileup_deeprm, "_open_files", lambda: 40 + len(pileup_deeprm._SAVED_MAPS))
    stems = []
    for w in range(12):  # 12 workers, 8 files each: 96 maps would be needed
        rows = pileup_deeprm._ReadRows()
        read_id = np.array([[w, 1], [w, 1]], dtype=np.int64)
        rows.add(np.array([0, 0]), read_id, np.array([w, w + 1]), np.ones(2, np.uint8))
        stems.append(rows.save_by_reference(str(tmp_path / f"worker{w}")))
    for w, stem in enumerate(stems):
        assert pileup_deeprm._saved(stem, "rows")["pos"].tolist() == [w, w + 1]
        assert pileup_deeprm._bounds(stem, 0, 1) == (0, 2)
        assert len(pileup_deeprm._SAVED_MAPS) <= pileup_deeprm._saved_cap() == 20
    assert len(pileup_deeprm._SAVED_MAPS) == 20  # 12 workers' runs took more maps than that: some were dropped


@pytest.mark.parametrize("limit, n_files", [(256, 64), (1024, 300)])
def test_many_workers_under_a_low_open_file_limit(tmp_path, limit, n_files):
    """Many workers under a hard open-file limit that cannot be raised: the pileup completes with the outputs of a
    run without the limit (the workers' and the writers' pipes are closed once they end, and the maps are capped by
    the descriptors still free). At 1024, a common container or Slurm limit, the 64 processes shrink only a little,
    so the descriptors each process takes besides its pipes must fit too."""
    rng = np.random.default_rng(5)
    sq = [(f"chr{i}", 3000) for i in range(8)]
    reads = [(i % 8, 7 * (i // 8), [(0, 40)], _random_seq(rng, 40), 0, []) for i in range(512)]
    inp, bam = _bam_and_rows(tmp_path, sq, reads, n_files=n_files)
    out_free, out_low = tmp_path / "free", tmp_path / "low"
    pileup_deeprm.main(_args(inp, out_free, bam, thread=2))
    code = (
        "import argparse; from deeprm.inference import pileup_deeprm; "
        f"pileup_deeprm.main(argparse.Namespace(**{vars(_args(inp, out_low, bam, thread=64))!r}))"
    )
    src = os.path.dirname(os.path.dirname(os.path.dirname(pileup_deeprm.__file__)))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")]))}
    subprocess.run(
        [sys.executable, "-c", code],
        preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_NOFILE, (limit, limit)),
        env=env,
        check=True,
        timeout=600,
    )
    assert _outputs(out_low) == _outputs(out_free)


def test_ctrl_c_during_cleanup_is_held_until_it_ends():
    """Ctrl-C while the temporary files are removed does not cut the removal short: it is raised when it is done."""
    done = []
    with pytest.raises(KeyboardInterrupt):
        with pileup_deeprm._sigint_held():
            signal.raise_signal(signal.SIGINT)
            done.append("removed")
    assert done == ["removed"]
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_ctrl_c_stays_ignored_in_a_process_that_ignores_it():
    """A run started with Ctrl-C ignored (as `cmd &` in a script) keeps ignoring it during the cleanup."""
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        with pileup_deeprm._sigint_held():
            signal.raise_signal(signal.SIGINT)
            signal.raise_signal(signal.SIGINT)
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, previous)


def test_a_failed_modbam_join_leaves_no_partial_file(tmp_path, monkeypatch):
    """The modBAM is written under a temporary name and moved into place when complete: samtools failing part-way
    leaves no modBAM (and no index) in the output folder."""

    def cat(*a):
        open(a[a.index("-o") + 1], "wb").write(b"partial")
        raise pileup_deeprm.pysam.SamtoolsError("injected")

    monkeypatch.setattr(pileup_deeprm.pysam, "cat", cat)
    shard = tmp_path / "shard.bam"
    shard.write_bytes(b"x")
    out = tmp_path / "out" / "modbam_input.bam"
    out.parent.mkdir()
    with pytest.raises(pileup_deeprm.pysam.SamtoolsError):
        pileup_deeprm.merge_modbam_shards([str(shard)], str(out), 1, presorted=True)
    assert os.listdir(out.parent) == []


def test_a_modbam_never_sits_beside_another_modbams_index(tmp_path, monkeypatch):
    """The join failing between moving the modBAM and its index into place, over an earlier modBAM: the folder holds
    neither the new modBAM nor the earlier one's index (never a modBAM with another modBAM's index)."""
    inp, bam = _bam_and_rows(tmp_path, [("tx1", 2000)], [(0, 10, [(0, 30)], "ACGTA" * 6, 0, [])])
    out = tmp_path / "out" / "modbam_input.bam"
    out.parent.mkdir()
    out.write_bytes(b"earlier modBAM")
    (out.parent / "modbam_input.bam.bai").write_bytes(b"earlier index")
    shard = tmp_path / "shard.bam"
    shard.write_bytes(bam.read_bytes())
    replace = os.replace

    def replace_but_not_the_index(src, dst):
        if str(dst).endswith(".bai"):
            raise OSError("injected")
        replace(src, dst)

    monkeypatch.setattr(pileup_deeprm.os, "replace", replace_but_not_the_index)
    with pytest.raises(OSError):
        pileup_deeprm.merge_modbam_shards([str(shard)], str(out), 1, presorted=True)
    assert os.listdir(out.parent) == []


def test_comment_lines_appear_once_in_the_modbam_header(tmp_path, monkeypatch):
    """The input BAM's @CO lines are in the modBAM header once, however many shards are joined and whether they are
    joined as they are or merged and sorted (samtools merge copied them once per shard), with the same records."""
    rng = np.random.default_rng(6)
    reads = [(i % 3, 40 * (i // 3), [(0, 30)], _random_seq(rng, 30), 0, []) for i in range(90)]
    sq = [("a", 5000), ("b", 5000), ("c", 5000)]
    inp, bam = _bam_and_rows(tmp_path, sq, reads, rg=["rg1"], pg=["bc"], dorado_order=True)
    join, shards = pileup_deeprm.merge_modbam_shards, []

    def spy(paths, out_path, threads, presorted=False, header=None, shard_refs=None):
        shards.append(len(paths))
        first = not shards[1:]
        return join(paths, out_path, threads, presorted=presorted and first, header=header, shard_refs=shard_refs)

    monkeypatch.setattr(pileup_deeprm, "merge_modbam_shards", spy)
    results = []
    for path in ("cat", "merge"):
        out = tmp_path / path
        pileup_deeprm.main(_args(inp, out, bam, thread=3))
        results.append((_header(str(out / "modbam_input.bam")), _records(str(out / "modbam_input.bam"))))
    assert shards == [3, 3]
    assert [sum(line.startswith("@CO") for line in h) for h, _ in results] == [1, 1]
    assert results[0][1] == results[1][1] and results[0][0][1].startswith("@SQ")


def test_merge_order_header_splits_lines_at_newlines_only():
    """A header field may hold a carriage return (as a command line in @PG CL): the line stays whole. Lines of other
    kinds than @HD, @SQ, @RG, @PG and @CO are dropped, as samtools merge did; without comments, no @CO line."""
    text = "@HD\tVN:1.6\tSO:coordinate\n@CO\tc1\n@PG\tID:bc\tPN:bc\tCL:run\r1 --x\n@XY\tz\n@SQ\tSN:a\tLN:10\n"
    header = pysam.AlignmentHeader.from_text(text)
    assert str(pileup_deeprm._merge_order_header(header)).split("\n")[:-1] == [
        "@HD\tVN:1.6\tSO:coordinate",
        "@SQ\tSN:a\tLN:10",
        "@PG\tID:bc\tPN:bc\tCL:run\r1 --x",
        "@CO\tc1",
    ]
    assert "@CO" not in str(pileup_deeprm._merge_order_header(header, comments=False))


def _clustered_rows(tmp_path, clusters, seed):
    """Saved runs (two) of reads on reference 0 in `clusters` [(first, last position, number of reads, rows per
    read)]: each read starts in the first half of its cluster (the first read at the cluster's first position) and
    has rows to the cluster's last position, as reads that all end at a transcript's 3' end. Returns the stems and the
    number of rows."""
    rng = np.random.default_rng(seed)
    parts, k = [pileup_deeprm._ReadRows(), pileup_deeprm._ReadRows()], 0
    for first, last, n_reads, n_rows in clusters:
        for j in range(n_reads):
            start = first if j == 0 else first + int(rng.integers(0, (last - first) // 2))
            inner = np.sort(rng.choice(np.arange(start + 1, last), n_rows - 2, replace=False))
            pos = np.concatenate(([start], inner, [last])).astype(np.int64)
            read_id = np.tile(np.array([[k, k + 1]], dtype=np.int64), (n_rows, 1))
            parts[k % 2].add(np.zeros(n_rows, np.int64), read_id, pos, np.zeros(n_rows, np.uint8))
            k += 1
    stems = [part.save_by_reference(str(tmp_path / f"worker{i}")) for i, part in enumerate(parts)]
    return stems, sum(n * r for _, _, n, r in clusters)


def test_windows_of_a_reference_start_in_the_gaps_between_reads(tmp_path, monkeypatch):
    """_pos_windows starts a window just after the last row of the reads before a gap and takes whole runs of
    overlapping reads while they fit; a run too large for one window is cut inside."""
    # three runs of overlapping reads, of 100, 200 and 400 rows: [100, 199], [1000, 1599] and [5000, 5399]
    stems, n_rows = _clustered_rows(tmp_path, [(100, 199, 10, 10), (1000, 1599, 20, 10), (5000, 5399, 40, 10)], 3)
    assert pileup_deeprm._pos_windows(stems, 0, n_rows, 10**6).tolist() == [0]
    starts = pileup_deeprm._pos_windows(stems, 0, n_rows, 250).tolist()
    assert starts[:3] == [0, 200, 1600] and len(starts) == 4 and 5000 <= starts[3] < 5200
    assert pileup_deeprm._pos_windows(stems, 0, n_rows, 350).tolist()[:2] == [0, 1600]


def test_modbam_windows_cut_at_gaps_give_the_records_of_one_pass(tmp_path, monkeypatch):
    """Spliced reads (an N in the CIGAR) in clusters with gaps between them, one read spliced across a gap, read in
    windows cut at the gaps: the same records as one pass."""
    import uuid

    rng = np.random.default_rng(11)
    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 10**6}]}
    )
    records, rows = [], []
    for i in range(150):
        name = str(uuid.UUID(int=int(rng.integers(1, 2**62)) << 64 | i))
        cluster = [1000, 20000, 60000][i % 3]
        intron = 40000 if i == 7 else int(rng.integers(0, 200))  # read 7 is spliced from the 2nd cluster to the 3rd
        start = cluster + int(rng.integers(0, 300))
        seq = "".join(rng.choice(list("ACGT"), 30))
        records.append((start, name, seq, [(0, 10), (3, intron), (0, 20)] if intron else [(0, 30)]))
        hi, lo = np.frombuffer(uuid.UUID(name).bytes, dtype=np.int64)
        ref_pos = [start + q if q < 10 else start + intron + q for q in range(30)]
        rows += [(ref_pos[q] + 1, hi, lo) for q, base in enumerate(seq) if base == "A"]
    records.sort()
    bam = tmp_path / "input.bam"
    with pysam.AlignmentFile(str(bam), "wb", header=header) as out:
        for start, qname, s, cigar in records:
            a = pysam.AlignedSegment(header)
            a.query_name, a.flag, a.reference_id, a.reference_start = qname, 0, 0, start
            a.mapping_quality, a.cigartuples, a.query_sequence = 60, cigar, s
            a.query_qualities = pysam.qualitystring_to_array("I" * 30)
            out.write(a)
    pysam.index(str(bam))
    inp = tmp_path / "molecule-level"
    inp.mkdir()
    for f, part in enumerate(np.array_split(rng.permutation(len(rows)), 4)):
        np.savez(
            inp / f"inference_0_{f}.npz",
            label_id=np.array([rows[i][0] for i in part], dtype=np.int64),
            read_id=np.array([[rows[i][1], rows[i][2]] for i in part], dtype=np.int64),
            pred=rng.uniform(0, 1, len(part)).astype(np.float32),
        )
    calls, results = [], []
    windows = pileup_deeprm._pos_windows
    monkeypatch.setattr(pileup_deeprm, "_pos_windows", lambda *a: calls.append(windows(*a)) or calls[-1])
    for k, (run_rows, slice_rows) in enumerate([(10**9, 10**9), (61, 150)]):
        monkeypatch.setattr(pileup_deeprm, "_MODBAM_RUN_ROWS", run_rows)
        monkeypatch.setattr(pileup_deeprm, "_MODBAM_SLICE_ROWS", slice_rows)
        out = tmp_path / f"site-level-{k}"
        pileup_deeprm.main(_args(inp, out, bam, thread=2))
        results.append(_outputs(out))
    assert any(len(c) > 1 for c in calls)  # the second pass ran in windows
    assert results[1] == results[0]


def _bam_and_rows(tmp_path, sq, reads, rg=(), pg=(), n_files=4, seed=0, dorado_order=False):
    """An indexed, coordinate-sorted BAM of `reads` [(reference index, start, CIGAR, sequence, flag, tags)] on the
    references `sq` [(name, length)] (with @RG and @PG lines of the IDs `rg`, `pg`), and per-read files with a row
    (random probability) at each A of each primary alignment, in `n_files` files. Returns (input folder, BAM path)."""
    import uuid

    rng = np.random.default_rng(seed)
    hd = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": n, "LN": length} for n, length in sq]}
    if rg:
        hd["RG"] = [{"ID": i, "SM": "s"} for i in rg]
    if pg:
        hd["PG"] = [{"ID": i, "PN": i} for i in pg]
    header = pysam.AlignmentHeader.from_dict(hd)
    if dorado_order:  # the header lines as a Dorado BAM has them: @PG and @RG before the @SQ lines, and a comment
        lines = str(header).splitlines()
        kinds = [[x for x in lines if x.startswith(k)] for k in ("@HD", "@PG", "@RG", "@SQ")]
        ordered = kinds[0] + ["@CO\tbasecalled"] + kinds[1] + kinds[2] + kinds[3]
        header = pysam.AlignmentHeader.from_text("\n".join(ordered) + "\n")
    records, rows = [], []
    for i, (tid, start, cigar, seq, flag, tags) in enumerate(reads):
        a = pysam.AlignedSegment(header)
        a.query_name = str(uuid.UUID(int=(i + 1) << 64 | int(rng.integers(1, 2**62))))
        a.flag, a.reference_id, a.reference_start, a.mapping_quality = flag, tid, start, 60
        a.cigartuples, a.query_sequence = cigar, seq
        a.query_qualities = pysam.qualitystring_to_array("I" * len(seq))
        a.set_tags(tags)
        records.append(a)
        if flag & 0x900 == 0:
            hi, lo = np.frombuffer(uuid.UUID(a.query_name).bytes, dtype=np.int64)
            rows += [
                (tid * LABEL_DIV + r + 1, hi, lo) for q, r in a.get_aligned_pairs(matches_only=True) if seq[q] == "A"
            ]
    records.sort(key=lambda a: (a.reference_id, a.reference_start))
    bam = tmp_path / "input.bam"
    with pysam.AlignmentFile(str(bam), "wb", header=header) as out:
        for a in records:
            out.write(a)
    pysam.index(str(bam))
    inp = tmp_path / "molecule-level"
    inp.mkdir()
    for f, part in enumerate(np.array_split(np.arange(len(rows)), n_files)):  # a read's rows together, as written
        np.savez(
            inp / f"inference_0_{f}.npz",
            label_id=np.array([rows[i][0] for i in part], dtype=np.int64),
            read_id=np.array([[rows[i][1], rows[i][2]] for i in part], dtype=np.int64).reshape(-1, 2),
            pred=rng.uniform(0, 1, len(part)).astype(np.float32),
        )
    return inp, bam


def _random_seq(rng, n):
    return "".join(rng.choice(list("ACGT"), n))


def test_modbam_windows_of_a_deep_reference_load_about_their_own_rows(tmp_path, monkeypatch):
    """A deep reference whose reads all end at its 3' end (as a highly expressed transcript), read in windows: each
    window loads about the rows of the reads that start in it, not every row up to where they end, and the records
    are those of one pass."""
    rng = np.random.default_rng(21)
    reads = []
    for _ in range(120):
        start = int(rng.integers(0, 1400))
        reads.append((0, start, [(0, 1500 - start)], _random_seq(rng, 1500 - start), 0, []))
    inp, bam = _bam_and_rows(tmp_path, [("tx1", 2000)], reads)
    log = tmp_path / "loads.txt"
    load_window = pileup_deeprm._ReadRows.load_window.__func__

    def logged(cls, *a, **k):  # the writers are forked: the counts go to a file
        out = load_window(cls, *a, **k)
        with open(log, "a") as fh:
            fh.write(f"{len(out._pos)}\n")
        return out

    monkeypatch.setattr(pileup_deeprm._ReadRows, "load_window", classmethod(logged))
    results, slice_rows = [], 1500
    for k, (run_rows, cap) in enumerate([(10**9, 10**9), (997, slice_rows)]):
        monkeypatch.setattr(pileup_deeprm, "_MODBAM_RUN_ROWS", run_rows)
        monkeypatch.setattr(pileup_deeprm, "_MODBAM_SLICE_ROWS", cap)
        log.unlink(missing_ok=True)  # the loads of the pass in windows only
        out = tmp_path / f"site-level-{k}"
        pileup_deeprm.main(_args(inp, out, bam, thread=2))
        results.append(_outputs(out))
    loads = [int(x) for x in log.read_text().split()]
    n_rows = sum(len(np.load(f)["pred"]) for f in inp.glob("*.npz"))
    # a read that crosses a window's start is loaded by both windows: a few reads in each of these small windows
    assert len(loads) > 10 and max(loads) <= 2 * slice_rows and sum(loads) <= 1.5 * n_rows
    assert results[1] == results[0]


def test_modbam_records_keep_their_tags_where_samtools_merge_put_them(tmp_path):
    """The shards are joined as they are (samtools cat), so the writers move each record's RG tag and then its PG tag
    to the end, and drop one whose ID is not in the header, and give the header merge's order, as samtools merge -c -p
    did up to DeepRM 1.2: merging the modBAM again changes no record and no header line."""
    rng = np.random.default_rng(4)
    tag_sets = [
        [("RG", "rg1"), ("XA", 1), ("PG", "bc")],
        [("PG", "bc"), ("RG", "rg1")],
        [("RG", "zz"), ("XA", 2)],
        [("XA", 3)],
        [("PG", "zz")],
    ]
    reads = [(0, 40 * i, [(0, 30)], _random_seq(rng, 30), 0, tag_sets[i % 5]) for i in range(50)]
    sq = [("chr1", 10000), ("chr2", 500)]
    inp, bam = _bam_and_rows(tmp_path, sq, reads, rg=["rg1"], pg=["bc"], dorado_order=True)
    out = tmp_path / "site-level"
    pileup_deeprm.main(_args(inp, out, bam, thread=2))
    modbam = str(out / "modbam_input.bam")
    merged = str(tmp_path / "merged.bam")
    pysam.merge("-c", "-p", "-f", merged, modbam)
    assert _records(modbam) == _records(merged)
    # the header too: its lines in merge's order (the input BAM has them in Dorado's), apart from samtools's own line
    assert _header(modbam) == _header(merged) and _header(modbam)[1].startswith("@SQ")
    with pysam.AlignmentFile(modbam, "rb") as f:
        records = list(f)
    assert len(records) == 50 and all(r.get_tags()[-1][0] == "PG" for r in records if r.has_tag("PG"))
    assert not any(r.has_tag("RG") and r.get_tag("RG") == "zz" for r in records)
    assert all(
        r.get_tags()[-2:] == [("RG", "rg1"), ("PG", "bc")] for r in records if r.has_tag("XA") and r.has_tag("PG")
    )


def _records(path):
    with pysam.AlignmentFile(path, "rb") as f:
        return [r.to_string() for r in f]


def _header(path):
    with pysam.AlignmentFile(path, "rb") as f:
        return [line for line in str(f.header).splitlines() if not line.startswith("@PG\tID:samtools")]


def test_a_large_reference_is_split_also_beside_small_ones(tmp_path, monkeypatch):
    """_plan_modbam_tasks: a reference with more than two shares of the reads is written in window tasks of its own,
    also next to small references (they go into tasks of whole references), and the tasks keep the reference order."""
    monkeypatch.setattr(pileup_deeprm, "_MODBAM_SLICE_ROWS", 50)
    rng = np.random.default_rng(8)
    reads = [
        (t, int(rng.integers(0, 4000)), [(0, 30)], _random_seq(rng, 30), 0, [])
        for t, n in ((0, 6), (1, 200), (2, 6))
        for _ in range(n)
    ]
    inp, bam = _bam_and_rows(tmp_path, [("a", 5000), ("b", 5000), ("c", 5000)], reads)
    rows = pileup_deeprm._ReadRows()
    for f in sorted(inp.glob("*.npz")):
        with np.load(f) as z:
            ref, pos = pileup_deeprm._decode_label_id(z["label_id"], LABEL_DIV)
            rows.add(ref, z["read_id"], pos, np.zeros(len(pos), np.uint8))
    stem = rows.save_by_reference(str(tmp_path / "worker"))
    tasks = pileup_deeprm._plan_modbam_tasks(str(bam), [stem], np.array([0, 1, 2]), threads=2)
    assert tasks[0][0] == "refs" and tasks[0][1].tolist() == [0] and tasks[-1][0] == "refs"
    assert tasks[-1][1].tolist() == [2]
    middle = tasks[1:-1]
    assert len(middle) >= 2 and all(t[0] == "windows" and t[1] == 1 for t in middle)
    assert [t[2][0] for t in middle] == sorted(t[2][0] for t in middle) and middle[-1][3] is None


def test_a_modbam_joined_by_copying_is_that_of_samtools_and_its_index_finds_the_same_reads(tmp_path, monkeypatch):
    """The modBAM put together from the shards without reading them (copied blocks, merged shard indexes) holds the
    records and header of the one samtools joins, and its index gives the same reads as one samtools makes for it, for
    whole references and for regions, also for a reference written in several windows."""
    monkeypatch.setattr(pileup_deeprm, "_MODBAM_SLICE_ROWS", 50)
    rng = np.random.default_rng(9)
    reads = [
        (t, int(rng.integers(0, 4000)), [(0, 30)], _random_seq(rng, 30), 0, [])
        for t, n in ((0, 6), (1, 300), (2, 6), (4, 40))
        for _ in range(n)
    ]
    sq = [("a", 5000), ("b", 5000), ("c", 5000), ("d", 5000), ("e", 5000)]
    inp, bam = _bam_and_rows(tmp_path, sq, reads, rg=["rg1"], pg=["bc"], dorado_order=True)
    calls = []
    join = pileup_deeprm._join_without_reading
    monkeypatch.setattr(pileup_deeprm, "_join_without_reading", lambda *a: calls.append(join(*a)) or calls[-1])
    copied, joined = tmp_path / "copied", tmp_path / "samtools"
    pileup_deeprm.main(_args(inp, copied, bam, thread=3))
    monkeypatch.setattr(pileup_deeprm, "_copy_file_range", lambda: None)  # no copy_file_range: samtools joins
    pileup_deeprm.main(_args(inp, joined, bam, thread=3))
    assert calls == [True, False]
    a, b = str(copied / "modbam_input.bam"), str(joined / "modbam_input.bam")
    assert _records(a) == _records(b) and _header(a) == _header(b)
    pysam.index(a, str(tmp_path / "samtools_made.bai"))
    regions = [(name, None, None) for name, _ in sq]
    regions += [(name, s, s + int(rng.integers(1, 3000))) for name, _ in sq for s in rng.integers(0, 4500, 40)]
    with (
        pysam.AlignmentFile(a, "rb") as merged,
        pysam.AlignmentFile(a, "rb", index_filename=str(tmp_path / "samtools_made.bai")) as made,
    ):
        assert [tuple(s) for s in merged.get_index_statistics()] == [tuple(s) for s in made.get_index_statistics()]
        for name, start, stop in regions:
            got = [r.to_string() for r in merged.fetch(name, start, stop)]
            assert got == [r.to_string() for r in made.fetch(name, start, stop)]
    assert sum(1 for _ in pysam.AlignmentFile(a, "rb").fetch("b")) > 0


def test_the_merged_index_of_long_references_finds_the_reads_of_any_region(tmp_path, monkeypatch):
    """Long references and long reads (many 16 kb windows of the linear index, reads across windows, stretches with no
    read), one reference written in several windows: the merged index gives the reads a samtools-made index gives."""
    monkeypatch.setattr(pileup_deeprm, "_MODBAM_SLICE_ROWS", 20000)
    rng = np.random.default_rng(12)
    reads = []
    for t, n, span in ((0, 60, (0, 150000)), (1, 8, (200000, 260000)), (2, 4, (0, 50000))):
        for _ in range(n):
            length = int(rng.integers(500, 20000))
            reads.append((t, int(rng.integers(*span)), [(0, length)], _random_seq(rng, length), 0, []))
    sq = [("a", 300000), ("b", 300000), ("c", 300000)]
    inp, bam = _bam_and_rows(tmp_path, sq, reads, n_files=3)
    shards = []
    join = pileup_deeprm._join_without_reading
    monkeypatch.setattr(pileup_deeprm, "_join_without_reading", lambda *a: shards.append(len(a[0])) or join(*a))
    out = tmp_path / "copied"
    pileup_deeprm.main(_args(inp, out, bam, thread=3))
    assert shards and shards[0] >= 3
    a = str(out / "modbam_input.bam")
    pysam.index(a, str(tmp_path / "samtools_made.bai"))
    regions = [(name, None, None) for name, _ in sq]
    regions += [(name, s, s + int(rng.integers(1, 40000))) for name, _ in sq for s in rng.integers(0, 299000, 60)]
    with (
        pysam.AlignmentFile(a, "rb") as merged,
        pysam.AlignmentFile(a, "rb", index_filename=str(tmp_path / "samtools_made.bai")) as made,
    ):
        assert [tuple(s) for s in merged.get_index_statistics()] == [tuple(s) for s in made.get_index_statistics()]
        for name, start, stop in regions:
            assert [r.query_name for r in merged.fetch(name, start, stop)] == [
                r.query_name for r in made.fetch(name, start, stop)
            ]


def test_a_shard_without_an_index_is_joined_by_samtools(tmp_path, monkeypatch):
    """A shard that could not be indexed: the shards are joined by samtools, with the same modBAM."""
    rng = np.random.default_rng(10)
    reads = [
        (t, int(rng.integers(0, 4000)), [(0, 30)], _random_seq(rng, 30), 0, []) for t in range(3) for _ in range(30)
    ]
    inp, bam = _bam_and_rows(tmp_path, [("a", 5000), ("b", 5000), ("c", 5000)], reads)
    pileup_deeprm.main(_args(inp, tmp_path / "copied", bam, thread=3))
    monkeypatch.setattr(pileup_deeprm, "_index_shard", lambda path: None)
    calls = []
    join = pileup_deeprm._join_without_reading
    monkeypatch.setattr(pileup_deeprm, "_join_without_reading", lambda *a: calls.append(join(*a)) or calls[-1])
    pileup_deeprm.main(_args(inp, tmp_path / "samtools", bam, thread=3))
    assert calls == [False]
    a, b = str(tmp_path / "copied" / "modbam_input.bam"), str(tmp_path / "samtools" / "modbam_input.bam")
    assert _records(a) == _records(b) and _header(a) == _header(b)


def test_genomic_pileup_of_categorical_names_is_that_of_string_names(tmp_path):
    """pileup() hands the transcript names to pileup_genomic as a categorical column; the table is the same."""
    annot = tmp_path / "annot.refflat"
    _refflat(annot, 4)
    rng = np.random.default_rng(5)
    n = 500
    ref_arr = np.array(["tx1", "tx2", "tx3", "tx4", "tx9"])  # tx9 is not in the annotation
    codes = rng.integers(0, len(ref_arr), n)
    cols = {"ref_pos": rng.integers(0, 2100, n), "ref_strand": np.where(rng.random(n) < 0.5, "+", "-")}
    sums = {key: rng.random(n).astype(dtype) for key, dtype in pileup_deeprm.SITE_SUMS}
    keys = [key for key, _ in pileup_deeprm.SITE_SUMS]
    as_strings = pileup_genomic(pd.DataFrame({"ref_names": ref_arr[codes], **cols, **sums}), str(annot), keys)
    categorical = pd.Categorical.from_codes(codes, categories=pd.Index(ref_arr, dtype=object))
    as_codes = pileup_genomic(pd.DataFrame({"ref_names": categorical, **cols, **sums}, copy=False), str(annot), keys)
    pd.testing.assert_frame_equal(as_codes, as_strings, check_exact=True)


def _cgroup(tmp_path, files):
    d = tmp_path / "cg"
    d.mkdir(exist_ok=True)
    for name, text in files.items():
        (d / name).write_text(text)
    return str(d)


def test_available_memory_and_cpus_follow_a_cgroup_v2_limit(tmp_path, monkeypatch):
    d = _cgroup(tmp_path, {"memory.max": "1073741824\n", "memory.current": "536870912\n", "cpu.max": "250000 100000\n"})
    monkeypatch.setattr(utils, "_cgroup_dirs", lambda: [("v2", d)])
    monkeypatch.setattr("psutil.virtual_memory", lambda: types.SimpleNamespace(available=64 << 30))
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(64)), raising=False)
    assert utils.available_memory() == 512 << 20
    assert utils.available_cpus() == 3
    assert utils.default_threads() == 2
    assert utils.sort_memory_per_thread(2) == 128


def test_no_cgroup_limit_leaves_the_machine_values(tmp_path, monkeypatch):
    d = _cgroup(tmp_path, {"memory.max": "max\n", "memory.current": "536870912\n", "cpu.max": "max 100000\n"})
    monkeypatch.setattr(utils, "_cgroup_dirs", lambda: [("v2", d)])
    monkeypatch.setattr("psutil.virtual_memory", lambda: types.SimpleNamespace(available=64 << 30))
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(40)), raising=False)
    assert utils.available_memory() == 64 << 30
    assert utils.available_cpus() == 40
    assert utils.default_threads() == 38


def test_available_memory_and_cpus_follow_a_cgroup_v1_limit(tmp_path, monkeypatch):
    d = _cgroup(
        tmp_path,
        {
            "memory.limit_in_bytes": str(8 << 30),
            "memory.usage_in_bytes": str(6 << 30),
            "cpu.cfs_quota_us": "400000",
            "cpu.cfs_period_us": "100000",
        },
    )
    monkeypatch.setattr(utils, "_cgroup_dirs", lambda: [("v1", d)])
    monkeypatch.setattr("psutil.virtual_memory", lambda: types.SimpleNamespace(available=64 << 30))
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(16)), raising=False)
    assert utils.available_memory() == 2 << 30
    assert utils.available_cpus() == 4


def test_available_memory_gives_back_the_inactive_page_cache(tmp_path, monkeypatch):
    """The cgroup's usage counts the page cache; its inactive part (memory.stat) is free for the asking."""
    monkeypatch.setattr("psutil.virtual_memory", lambda: types.SimpleNamespace(available=64 << 30))
    (tmp_path / "v2").mkdir()
    (tmp_path / "v1").mkdir()
    d2 = _cgroup(
        tmp_path / "v2",
        {
            "memory.max": str(4 << 30),
            "memory.current": str(3 << 30),
            "memory.stat": f"anon 1\ninactive_file {3 << 29}\n",
        },
    )
    monkeypatch.setattr(utils, "_cgroup_dirs", lambda: [("v2", d2)])
    assert utils.available_memory() == (5 << 29)
    d1 = _cgroup(
        tmp_path / "v1",
        {
            "memory.limit_in_bytes": str(8 << 30),
            "memory.usage_in_bytes": str(6 << 30),
            "memory.stat": f"inactive_file 7\ntotal_inactive_file {1 << 30}\n",
        },
    )
    monkeypatch.setattr(utils, "_cgroup_dirs", lambda: [("v1", d1)])
    assert utils.available_memory() == 3 << 30


def test_cgroup_dirs_of_this_process_are_folders():
    for version, d in utils._cgroup_dirs():
        assert version in ("v1", "v2") and os.path.isdir(d)
    assert utils.available_memory() > 0 and utils.available_cpus() >= 1


def _random_reads(n, seed):
    """Reads with random CIGARs (M I D N S H = X, some with P or no CIGAR, which go to the one-read path), random
    strands and sequences, and rows inside and outside their alignment, some at one position twice (agreeing or not)."""
    rng = np.random.default_rng(seed)
    header = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "chr1", "LN": 10**6}]})
    reads, payloads = [], []
    for i in range(n):
        a = pysam.AlignedSegment(header)
        a.query_name = f"r{i}"
        a.reference_id = 0
        a.reference_start = int(rng.integers(0, 5000))
        a.flag = 16 if rng.random() < 0.5 else 0
        ops = []
        if rng.random() < 0.3:
            ops.append((5, int(rng.integers(1, 5))))
        if rng.random() < 0.5:
            ops.append((4, int(rng.integers(1, 8))))
        for _ in range(int(rng.integers(1, 7))):
            ops.append((int(rng.choice([0, 0, 1, 2, 3, 7, 8])), int(rng.integers(1, 30))))
        if ops[-1][0] in (1, 2, 3) or not any(o in (0, 7, 8) for o, _ in ops):
            ops.append((0, int(rng.integers(1, 30))))
        if rng.random() < 0.3:
            ops.append((4, int(rng.integers(1, 8))))
        if rng.random() < 0.05:
            ops.insert(1, (6, 2))  # padding: _query_positions leaves the read to get_aligned_pairs
        q_len = sum(n for o, n in ops if o in (0, 1, 4, 7, 8))
        alphabet = list("ACGT") if rng.random() < 0.9 else list("CG")  # some reads without an A
        a.query_sequence = "".join(rng.choice(alphabet, q_len))
        a.query_qualities = pysam.qualitystring_to_array("I" * q_len)
        if rng.random() < 0.05:
            a.cigarstring = None
        else:
            a.cigartuples = ops
        if rng.random() < 0.3:
            a.set_tag("MN", q_len + 3, "i")
        span = sum(n for o, n in ops if o in (0, 2, 3, 7, 8))
        k = int(rng.integers(0, 60))
        pos = (a.reference_start + rng.integers(-5, span + 5, k)).astype(np.int64)
        pred = rng.integers(0, 256, k).astype(np.uint8)
        if k and rng.random() < 0.3:  # a position listed twice
            pos = np.append(pos, pos[0])
            pred = np.append(pred, int(pred[0]) if rng.random() < 0.5 else (int(pred[0]) + 1) % 256).astype(np.uint8)
        reads.append(a)
        payloads.append((pos, pred))
    return header, reads, payloads


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_batched_tags_are_those_of_one_read_at_a_time(tmp_path, seed):
    header, reads, payloads = _random_reads(400, seed)
    out = {}
    for mode in ("one", "batch"):
        copies = [pysam.AlignedSegment.fromstring(r.to_string(), header) for r in reads]
        path = str(tmp_path / f"{mode}.bam")
        with pysam.AlignmentFile(path, "wb", header=header) as bam:
            if mode == "one":
                for read, payload in zip(copies, payloads):
                    pileup_deeprm._write_tagged(read, payload, bam)
            else:
                pileup_deeprm._write_batch(list(zip(copies, payloads)), bam)
        with pysam.AlignmentFile(path, "rb", check_sq=False) as bam:
            out[mode] = [r.to_string() for r in bam]
    assert len(out["one"]) == 400
    assert any("MM:Z:A+a?," in r for r in out["one"]) and any("MM:Z:A+a?;" in r for r in out["one"])
    assert out["batch"] == out["one"]


def test_an_early_failure_leaves_no_temporary_folder(tmp_path):
    """A failure before the processes start (here a bad --annot, read after the modBAM's temporary folder is made)
    leaves no temporary folder in the output folder or in --tmp-dir."""
    rng = np.random.default_rng(3)
    reads = [
        (t, int(rng.integers(0, 400)), [(0, 300)], _random_seq(rng, 300), 0, []) for t in range(2) for _ in range(5)
    ]
    inp, bam = _bam_and_rows(tmp_path, [("a", 1000), ("b", 1000)], reads)
    annot = tmp_path / "bad_annot.txt"
    annot.write_text("not\ta\trefflat\n")
    args = _args(inp, tmp_path / "out", bam, thread=2)
    args.annot, args.tmp_dir = str(annot), str(tmp_path / "tmp")
    with pytest.raises(ValueError):
        pileup_deeprm.main(args)
    for folder in (tmp_path / "out", tmp_path / "tmp"):
        assert not folder.exists() or [p for p in os.listdir(folder) if p.startswith(".deeprm_")] == []


def test_parts_on_another_file_system_are_joined_by_copying(tmp_path, monkeypatch):
    """With --tmp-dir on another file system, copy_file_range refuses to copy the modBAM parts into the output folder
    (EXDEV): they are copied through this machine instead, still without samtools, and the outputs are the same."""
    rng = np.random.default_rng(3)
    reads = [
        (t, int(rng.integers(0, 400)), [(0, 300)], _random_seq(rng, 300), 0, []) for t in range(4) for _ in range(30)
    ]
    inp, bam = _bam_and_rows(tmp_path, [(f"tx{i}", 1000) for i in range(4)], reads)
    calls = []
    join = pileup_deeprm._join_without_reading
    monkeypatch.setattr(pileup_deeprm, "_join_without_reading", lambda *a: calls.append(join(*a)) or calls[-1])
    pileup_deeprm.main(_args(inp, tmp_path / "same", bam, thread=3))

    def refuse(*a):
        raise OSError(errno.EXDEV, os.strerror(errno.EXDEV))

    monkeypatch.setattr(pileup_deeprm, "_copy_file_range", lambda: refuse)
    pileup_deeprm.main(_args(inp, tmp_path / "across", bam, thread=3))
    assert calls == [True, True]
    assert _outputs(tmp_path / "across") == _outputs(tmp_path / "same")


def test_a_tmp_dir_that_cannot_be_used_is_refused_before_any_work(tmp_path):
    """--tmp-dir is checked with the other settings, so deeprm call run refuses it before the inference."""
    bam, file = tmp_path / "x.bam", tmp_path / "a_file"
    bam.write_bytes(b"")
    file.write_text("")
    for bad in (file, file / "sub"):
        args = _args(tmp_path, tmp_path / "out", bam, thread=1)
        args.tmp_dir = str(bad)
        with pytest.raises(ValueError, match="--tmp-dir"):
            pileup_deeprm._validate_args(args)
    args.tmp_dir = str(tmp_path / "new" / "tmp")
    pileup_deeprm._validate_args(args)


@pytest.mark.parametrize("thread", [1, 2])
def test_a_reference_past_the_bam_is_reported_as_such(tmp_path, thread):
    """Per-read rows of a reference the BAM does not have (the BAM of another run): the error says so, also when the
    only site adder stops before it has written any site."""
    reads = _write_input_bam(tmp_path / "x.bam", n_reads=4)
    inp = tmp_path / "molecule-level"
    inp.mkdir()
    _write_per_read_files(inp, [(ref + 5, start, name, seq) for ref, start, name, seq in reads], n_files=2)
    args = _args(inp, tmp_path / "out", tmp_path / "x.bam", thread=thread)
    args.skip_modbam = True
    with pytest.raises(ValueError, match="outside the BAM reference range"):
        pileup_deeprm.main(args)
