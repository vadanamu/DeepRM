"""The modBAM that deeprm call pileup writes from several workers: header, read groups and record order."""

import argparse
import os
import uuid

import numpy as np
import pysam
import pytest

from deeprm.inference import pileup_deeprm

LABEL_DIV = 10**9
READ_LEN = 30
# everything a pileup without --annot leaves in its output folder, for an input BAM named input.bam
OUTPUTS = ["modbam_input.bam", "modbam_input.bam.bai", "pileup.bed", "pileup.npz"]


def _write_input_bam(path, n_reads, seed=0, n_refs=1):
    """A coordinate-sorted, indexed BAM with one @RG (every read carries its RG tag) and one @PG; the reads are dealt
    to the `n_refs` references in turn. Returns the reads as (reference index, start, name, sequence)."""
    rng = np.random.default_rng(seed)
    header = pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": f"tx{r + 1}", "LN": 2000} for r in range(n_refs)],
            "RG": [{"ID": "rg1", "SM": "sample1"}],
            "PG": [{"ID": "basecaller", "PN": "basecaller"}],
        }
    )
    reads = []
    for i in range(n_reads):
        name = str(uuid.UUID(int=int(rng.integers(1, 2**62)) << 64 | i))
        start = int(rng.integers(0, 20)) * 90  # 20 start positions for 60 reads: reads tie on position
        seq = "".join(rng.choice(list("ACGT"), READ_LEN))
        reads.append((i % n_refs, start, name, seq))
    reads.sort()
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for ref, start, name, seq in reads:
            a = pysam.AlignedSegment(header)
            a.query_name = name
            a.flag = 0
            a.reference_id = ref
            a.reference_start = start
            a.mapping_quality = 60
            a.cigartuples = [(0, READ_LEN)]
            a.query_sequence = seq
            a.query_qualities = pysam.qualitystring_to_array("I" * READ_LEN)
            a.set_tag("RG", "rg1")
            out.write(a)
    pysam.index(str(path))
    return reads


def _write_per_read_files(folder, reads, n_files, seed=1):
    """Every A of every read gets a prediction; reads are dealt to the files in turn (each read in one file)."""
    rng = np.random.default_rng(seed)
    rows = [[] for _ in range(n_files)]
    for i, (ref, start, name, seq) in enumerate(reads):
        hi, lo = np.frombuffer(uuid.UUID(name).bytes, dtype=np.int64)
        for q, base in enumerate(seq):
            if base == "A":
                rows[i % n_files].append((ref * LABEL_DIV + start + q + 1, hi, lo))
    for f, part in enumerate(rows):
        label = np.array([r[0] for r in part], dtype=np.int64)
        read_id = np.array([[r[1], r[2]] for r in part], dtype=np.int64)
        np.savez(
            folder / f"inference_0_{f}.npz",
            label_id=label,
            read_id=read_id,
            pred=rng.uniform(0, 1, len(label)).astype(np.float32),
        )


def _args(inp, out, bam, thread):
    return argparse.Namespace(
        input=str(inp),
        output=str(out),
        bam=str(bam),
        thread=thread,
        threshold=0.98,
        epsilon=1e-30,
        postfix="",
        slice=None,
        flip=False,
        label_div=LABEL_DIV,
        annot=None,
        skip_modbam=False,
    )


def _records(path):
    with pysam.AlignmentFile(str(path), "rb") as bam:
        return bam.header.to_dict(), [r.to_string() for r in bam]


def test_modbam_keeps_the_read_group_and_its_record_order(tmp_path):
    bam = tmp_path / "input.bam"
    reads = _write_input_bam(bam, n_reads=60)
    inp = tmp_path / "molecule-level"
    inp.mkdir()
    _write_per_read_files(inp, reads, n_files=6)

    runs = []
    for k in range(2):
        out = tmp_path / f"site-level-{k}"
        pileup_deeprm.main(_args(inp, out, bam, thread=4))
        runs.append(_records(out / "modbam_input.bam"))
        assert sorted(p.name for p in out.iterdir()) == OUTPUTS, "the temporary files are removed"

    header, records = runs[0]
    assert [rg["ID"] for rg in header["RG"]] == ["rg1"], "one @RG, the input's, not one renamed copy per worker"
    assert len([pg for pg in header["PG"] if pg["ID"] == "basecaller"]) == 1
    with pysam.AlignmentFile(str(tmp_path / "site-level-0" / "modbam_input.bam"), "rb") as f:
        tags = {r.get_tag("RG") for r in f}
    assert tags == {"rg1"}
    assert len(records) == len(reads)
    assert runs[1][1] == records, "the record order is the same on every run"


def _split_reads_over_files(folder, reads, n_files, seed=2):
    """Every read's A positions are split over two neighbouring files, so neighbouring workers share reads."""
    rng = np.random.default_rng(seed)
    rows = [[] for _ in range(n_files)]
    for i, (ref, start, name, seq) in enumerate(reads):
        hi, lo = np.frombuffer(uuid.UUID(name).bytes, dtype=np.int64)
        a_pos = [q for q, base in enumerate(seq) if base == "A"]
        for k, q in enumerate(a_pos):
            f = (i + (k >= len(a_pos) // 2)) % n_files
            rows[f].append((ref * LABEL_DIV + start + q + 1, hi, lo))
    for f, part in enumerate(rows):
        np.savez(
            folder / f"inference_0_{f}.npz",
            label_id=np.array([r[0] for r in part], dtype=np.int64),
            read_id=np.array([[r[1], r[2]] for r in part], dtype=np.int64),
            pred=rng.uniform(0, 1, len(part)).astype(np.float32),
        )


def test_a_read_split_over_workers_gets_one_record(tmp_path):
    bam = tmp_path / "input.bam"
    reads = _write_input_bam(bam, n_reads=40, seed=4)
    inp = tmp_path / "molecule-level"
    inp.mkdir()
    _split_reads_over_files(inp, reads, n_files=8)
    out = {}
    for thread in (1, 4, 8):
        folder = tmp_path / f"t{thread}"
        pileup_deeprm.main(_args(inp, folder, bam, thread=thread))
        out[thread] = _records(folder / "modbam_input.bam")[1]
        assert sorted(p.name for p in folder.iterdir()) == OUTPUTS, "the temporary files are removed"
    assert len(out[1]) == len(reads), "one record per alignment"
    assert out[4] == out[1] and out[8] == out[1], "the same records at any --thread"
    with pysam.AlignmentFile(str(tmp_path / "t8" / "modbam_input.bam"), "rb") as f:
        for read in f:
            n_a = read.get_forward_sequence().count("A")
            assert len(read.get_tag("ML")) == n_a, "every call of the read is in its one record"


def test_modbam_of_several_references_is_the_same_at_any_thread(tmp_path, monkeypatch):
    """With several references, --thread 4 runs several modBAM writers (one per run of references); their shards
    merge into the modBAM that one writer makes at --thread 1: one @RG and one record per read."""
    bam = tmp_path / "input.bam"
    reads = _write_input_bam(bam, n_reads=60, seed=9, n_refs=5)
    inp = tmp_path / "molecule-level"
    inp.mkdir()
    _write_per_read_files(inp, reads, n_files=6)
    n_writers = []
    modbam_buckets = pileup_deeprm._modbam_buckets

    def counted_buckets(bam_path, ref_ids, n):
        buckets = modbam_buckets(bam_path, ref_ids, n)
        n_writers.append(len(buckets))
        return buckets

    monkeypatch.setattr(pileup_deeprm, "_modbam_buckets", counted_buckets)
    out = {}
    for thread in (1, 4):
        folder = tmp_path / f"t{thread}"
        pileup_deeprm.main(_args(inp, folder, bam, thread=thread))
        out[thread] = _records(folder / "modbam_input.bam")
        assert sorted(p.name for p in folder.iterdir()) == OUTPUTS, "the temporary files are removed"
    assert n_writers[0] == 1 and n_writers[1] > 1, "--thread 4 runs more than one modBAM writer"

    header, records = out[4]
    assert [rg["ID"] for rg in header["RG"]] == ["rg1"], "one @RG, the input's, not one renamed copy per writer"
    assert len([pg for pg in header["PG"] if pg["ID"] == "basecaller"]) == 1
    with pysam.AlignmentFile(str(tmp_path / "t4" / "modbam_input.bam"), "rb") as f:
        tags, references = zip(*((r.get_tag("RG"), r.reference_name) for r in f))
    assert set(tags) == {"rg1"}
    assert set(references) == {f"tx{r + 1}" for r in range(5)}, "every reference keeps its reads"
    assert len(records) == len(reads), "one record per read"
    assert records == out[1][1], "the same records at --thread 1 and 4"


@pytest.mark.parametrize("name", ["shard_0.bam", "worker_0.bam"])
def test_an_input_bam_named_like_a_temporary_file_keeps_its_modbam(tmp_path, name):
    """The temporary files have a folder of their own, so removing them cannot remove modbam_<input BAM name>."""
    bam = tmp_path / name
    reads = _write_input_bam(bam, n_reads=20, seed=7)
    inp = tmp_path / "molecule-level"
    inp.mkdir()
    _write_per_read_files(inp, reads, n_files=2)
    out = tmp_path / "site-level"
    pileup_deeprm.main(_args(inp, out, bam, thread=2))
    expected = sorted([f"modbam_{name}", f"modbam_{name}.bai", "pileup.bed", "pileup.npz"])
    assert sorted(p.name for p in out.iterdir()) == expected
    assert len(_records(out / f"modbam_{name}")[1]) == len(reads)


def test_a_failed_modbam_step_removes_its_temporary_folder(tmp_path, monkeypatch):
    bam = tmp_path / "input.bam"
    reads = _write_input_bam(bam, n_reads=20, seed=8)
    inp = tmp_path / "molecule-level"
    inp.mkdir()
    _write_per_read_files(inp, reads, n_files=2)
    seen = {}

    def failing_write(bam_path, rows_stems, ref_ids, out_path, threads, tmp_dir):
        seen[tmp_dir] = sorted(os.listdir(tmp_dir))
        raise RuntimeError("injected")

    monkeypatch.setattr(pileup_deeprm, "_write_modbam", failing_write)
    out = tmp_path / "site-level"
    with pytest.raises(RuntimeError, match="injected"):
        pileup_deeprm.main(_args(inp, out, bam, thread=2))
    [(tmp_dir, names)] = seen.items()
    assert names, "the workers saved their rows in the temporary folder"
    assert os.path.dirname(tmp_dir) == str(out) and not os.path.exists(tmp_dir)
    assert not list(out.iterdir())


def test_merge_sorts_shards_that_are_not_sorted(tmp_path):
    """merge_modbam_shards(presorted=True) checks the merge by indexing it, and sorts it when that fails."""
    bam = tmp_path / "input.bam"
    reads = _write_input_bam(bam, n_reads=20, seed=5)
    with pysam.AlignmentFile(str(bam), "rb") as f:
        records = list(f)
    shard = tmp_path / "modbam_shard_0.bam"
    with pysam.AlignmentFile(str(bam), "rb") as template, pysam.AlignmentFile(str(shard), "wb", template=template) as o:
        for r in reversed(records):  # out of order
            o.write(r)
    out = tmp_path / "merged.bam"
    pileup_deeprm.merge_modbam_shards([str(shard)], str(out), threads=1, presorted=True)
    with pysam.AlignmentFile(str(out), "rb") as f:
        starts = [r.reference_start for r in f]
        assert f.check_index()
    assert starts == sorted(starts) and len(starts) == len(reads)
    assert not shard.exists() and not (tmp_path / "merged.bam.unsorted.bam").exists()


def test_shard_writer_without_an_index_scans_for_its_references(tmp_path):
    bam = tmp_path / "input.bam"
    reads = _write_input_bam(bam, n_reads=10, seed=6)
    (tmp_path / "input.bam.bai").unlink()
    rows = {}
    for ref, start, name, _ in reads:
        hi, lo = np.frombuffer(uuid.UUID(name).bytes, dtype=np.int64)
        rows[(ref, int(hi), int(lo))] = ([start], [200])
    out = tmp_path / "shard.bam"
    pileup_deeprm.write_modbam_worker_from_map(str(bam), str(out), rows, progress=False, references=[0])
    with pysam.AlignmentFile(str(out), "rb", check_sq=False) as f:
        assert len(list(f)) == len(reads)
