"""modBAM (MM/ML) writing must be strand-correct for genomic alignment.

MM/ML are defined relative to the ORIGINAL basecalled read orientation. For
reverse-strand alignments the BAM stores SEQ reverse-complemented, so the
writer must recover the original orientation before emitting MM/ML, and the
modified base is the RNA adenosine ('A') in that orientation.
"""

import numpy as np
import pysam
import pytest

from deeprm.inference.pileup_deeprm import (
    get_mm_tag,
    reverse_complement,
    write_modbam_worker_from_map,
)


def test_reverse_complement():
    assert reverse_complement("ACGTAC") == "GTACGT"
    assert reverse_complement("AACGN") == "NCGTT"


def _write_bam(path, name, flag, seq, pos=100):
    header = pysam.AlignmentHeader.from_dict({"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 1000}]})
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        a = pysam.AlignedSegment(header)
        a.query_name = name
        a.flag = flag
        a.reference_id = 0
        a.reference_start = pos
        a.mapping_quality = 60
        a.cigartuples = [(0, len(seq))]
        a.query_sequence = seq
        a.query_qualities = pysam.qualitystring_to_array("I" * len(seq))
        a.set_tag("MD", str(len(seq)))
        out.write(a)
    pysam.sort("-o", str(path), str(path))
    pysam.index(str(path))


@pytest.mark.parametrize(
    "flag,seq,genomic_pos,exp_fwdref_qpos",
    [
        # forward read: query_sequence == original; 'A' at index 2 -> genomic 102
        (0, "TTATTT", 102, 2),
        # reverse read: stored SEQ is revcomp of original. original = revcomp =
        # "GTACGT" has its only 'A' at original index 2, which is stored index
        # 3 ('T') -> genomic 103. The fix must still emit MM/ML and pysam must
        # map it back to forward-ref query index 3.
        (16, "ACGTAC", 103, 3),
    ],
)
def test_modbam_strand_correct(tmp_path, flag, seq, genomic_pos, exp_fwdref_qpos):
    import uuid as _uuid

    in_bam = tmp_path / "in.bam"
    name = str(_uuid.UUID(int=0x1234))
    _write_bam(in_bam, name, flag, seq)

    hi, lo = np.frombuffer(_uuid.UUID(name).bytes, dtype=np.int64)
    data_map = {(0, int(hi), int(lo)): ([genomic_pos], [200])}  # pred_u8 = 200

    out_bam = tmp_path / "out.bam"
    write_modbam_worker_from_map(str(in_bam), str(out_bam), data_map)

    with pysam.AlignmentFile(str(out_bam), "rb") as bam:
        rec = next(iter(bam))

    mb = rec.modified_bases or {}
    assert mb, "no MM/ML written (reverse-strand calls must not be dropped)"
    # exactly one m6A call, on canonical base 'A', at the expected forward-ref
    # query_sequence position, with the ML value we supplied.
    key = next(k for k in mb if k[0] == "A")
    calls = mb[key]
    assert len(calls) == 1
    qpos, qual = calls[0]
    assert qpos == exp_fwdref_qpos
    assert qual == 200
    # the base pysam reports it on, in forward-ref SEQ, is the genomic-A site
    assert rec.query_sequence[qpos] == ("A" if flag == 0 else "T")


def test_forward_read_mm_unchanged():
    """get_mm_tag on a forward read is unaffected by the reverse-strand fix."""
    seq = "ACACA"  # 'A' at 0,2,4
    mm, ml = get_mm_tag([0, 4], [10, 20], seq)
    # run-lengths over A-positions [0,2,4]: first A (idx0)=0 skipped, then to
    # idx2 of A-list -> skip 1
    assert mm == "A+a?,0,1;"
    assert ml == [10, 20]


@pytest.mark.parametrize("flag,seq,genomic_pos,exp_fwdref_qpos", [(0, "TTATTT", 102, 2), (16, "ACGTAC", 103, 3)])
def test_modbam_strand_correct_from_the_worker_table(tmp_path, flag, seq, genomic_pos, exp_fwdref_qpos):
    """The same cases, with the rows given as the pileup workers build them (a _ReadRows table)."""
    import uuid as _uuid

    from deeprm.inference.pileup_deeprm import _ReadRows

    in_bam = tmp_path / "in.bam"
    name = str(_uuid.UUID(int=0x1234))
    _write_bam(in_bam, name, flag, seq)
    hi, lo = np.frombuffer(_uuid.UUID(name).bytes, dtype=np.int64)
    rows = _ReadRows()
    rows.add(np.array([0]), np.array([[hi, lo]], dtype=np.int64), np.array([genomic_pos]), np.array([200]))
    out_bam = tmp_path / "out.bam"
    write_modbam_worker_from_map(str(in_bam), str(out_bam), rows.finish())
    with pysam.AlignmentFile(str(out_bam), "rb") as bam:
        rec = next(iter(bam))
    calls = next(v for k, v in rec.modified_bases.items() if k[0] == "A")
    assert calls == [(exp_fwdref_qpos, 200)]


def test_cigar_positions_match_get_aligned_pairs():
    """_query_positions (from the CIGAR blocks) gives the pairs get_aligned_pairs() gives, for every CIGAR operation."""
    from deeprm.inference.pileup_deeprm import _query_positions

    rng = np.random.default_rng(0)
    header = pysam.AlignmentHeader.from_dict({"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 100000}]})
    for _ in range(300):
        ops = [(int(rng.choice([4, 5])), int(rng.integers(1, 5)))] if rng.random() < 0.5 else []
        for _ in range(int(rng.integers(1, 8))):
            ops.append((int(rng.choice([0, 1, 2, 3, 6, 7, 8])), int(rng.integers(1, 12))))
        if not any(op in (0, 7, 8) for op, _ in ops):
            ops.append((0, 5))
        if rng.random() < 0.5:
            ops.append((int(rng.choice([4, 5])), int(rng.integers(1, 5))))
        a = pysam.AlignedSegment(header)
        a.reference_id, a.reference_start, a.cigartuples = 0, int(rng.integers(0, 1000)), ops
        a.query_sequence = "A" * sum(n for op, n in ops if op in (0, 1, 4, 7, 8))
        pairs = {r: q for q, r in a.get_aligned_pairs() if r is not None and q is not None}
        ref = np.arange(a.reference_start - 5, a.reference_start + 200)
        if any(op == 6 for op, _ in ops):  # padding: left to get_aligned_pairs(), which counts it as query bases
            assert _query_positions(a, ref) is None
            continue
        q, ok = _query_positions(a, ref)
        assert [int(x) for x in ref[ok]] == [r for r in ref.tolist() if r in pairs]
        assert [int(x) for x in q[ok]] == [pairs[r] for r in ref.tolist() if r in pairs]


def test_get_mm_tag_lists_each_read_base_once():
    """Two records of one read overlapping on one reference give the same read base twice; MM must not get a -1 skip.

    A base whose rows agree is listed once with that call; a base whose rows disagree is left out, because which row
    belongs to this record is unknown."""
    mm, ml = get_mm_tag([2, 2], [10, 10], "AAAA")
    assert mm == "A+a?,2;" and ml == [10]
    mm, ml = get_mm_tag([3, 1, 3], [7, 5, 7], "AAAA")
    assert mm == "A+a?,1,1;" and ml == [5, 7]
    mm, ml = get_mm_tag([3, 1, 3], [7, 5, 9], "AAAA")
    assert mm == "A+a?,1;" and ml == [5]
    mm, ml = get_mm_tag([2, 2], [10, 11], "AAAA")
    assert mm == "A+a?;" and ml == []


@pytest.mark.parametrize("flag", [2048, 2048 + 16])
def test_mn_tag_follows_the_hard_clipped_seq(tmp_path, flag):
    """A basecaller's MN (the full read length) does not fit a hard-clipped record; the writer sets it to len(SEQ)."""
    import uuid as _uuid

    name = str(_uuid.UUID(int=0x5678))
    header = pysam.AlignmentHeader.from_dict({"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 1000}]})
    seq = "ACGTA" * 8 + "TT"  # 42 bases: 40 aligned, 2 soft-clipped
    in_bam = tmp_path / "in.bam"
    with pysam.AlignmentFile(str(in_bam), "wb", header=header) as out:
        a = pysam.AlignedSegment(header)
        a.query_name = name
        a.flag = flag
        a.reference_id = 0
        a.reference_start = 100
        a.mapping_quality = 60
        a.cigartuples = [(5, 6), (0, 40), (4, 2)]  # 6H40M2S
        a.query_sequence = seq
        a.query_qualities = pysam.qualitystring_to_array("I" * len(seq))
        a.set_tag("MD", "40")
        a.set_tag("MN", 48)
        out.write(a)
    pysam.index(str(in_bam))
    hi, lo = np.frombuffer(_uuid.UUID(name).bytes, dtype=np.int64)
    genomic = [100 + i for i, b in enumerate(seq[:40]) if b == ("A" if flag == 2048 else "T")]
    data_map = {(0, int(hi), int(lo)): (genomic, [200] * len(genomic))}
    out_bam = tmp_path / "out.bam"
    write_modbam_worker_from_map(str(in_bam), str(out_bam), data_map)
    with pysam.AlignmentFile(str(out_bam), "rb") as bam:
        rec = next(iter(bam))
    assert rec.get_tag("MN") == len(seq)
    calls = next(v for k, v in (rec.modified_bases or {}).items() if k[0] == "A")
    assert len(calls) == len(genomic)
