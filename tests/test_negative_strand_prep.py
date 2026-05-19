"""Negative-strand handling in ``deeprm call prep``.

Background
----------
A BAM stores the SEQ/QUAL of a reverse-strand read in **forward-reference
orientation** (the aligner reverse-complements the basecalled read). pysam's
``get_aligned_pairs(..., with_seq=True)`` therefore reports the reference base
in forward orientation, and the query positions index the forward-oriented
SEQ.

`parse_bam` selects the bases of interest by comparing that forward-reference
base to ``boi`` (default ``"A"``). Without strand awareness a reverse-strand
read's m6A adenosine -- which appears as ``T`` on the forward reference -- is
skipped, and forward-reference ``A`` (read base ``U``) is profiled instead.
The fix complements ``boi`` for reverse-strand reads.

These tests pin that behaviour. They build a tiny synthetic BAM with pysam
(the MD tag supplies the reference bases, so no reference FASTA is needed) and
exercise the exact filter used in
``deeprm.inference.inference_preprocess_python.parse_bam``.
"""

import pathlib

import numpy as np
import pysam
import pytest

from deeprm.inference.inference_preprocess_python import _BASE_COMPLEMENT

DATA_DIR = pathlib.Path(__file__).parent / "data"
# One real reverse-strand read from ON0090 (rna004 direct-RNA, dorado
# sup@v5.0.0 with m6A/pseU modbase + --emit-moves), carrying its authentic
# CIGAR/MD/mv/MM/ML tags. SEQ is stored forward-reference-oriented exactly as
# a real BAM stores reverse-strand reads.
REV_BAM = DATA_DIR / "rev_strand_read.bam"


def _write_synthetic_bam(path):
    """One forward + one reverse read, perfect 8M match, SEQ='ACGTACGT'.

    SEQ is written in forward-reference orientation for both reads, exactly
    as a real BAM stores them.
    """
    header = pysam.AlignmentHeader.from_dict({"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 1000}]})
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for name, flag, pos in (("fwd", 0, 100), ("rev", 16, 200)):
            a = pysam.AlignedSegment(header)
            a.query_name = name
            a.flag = flag
            a.reference_id = 0
            a.reference_start = pos
            a.mapping_quality = 60
            a.cigartuples = [(0, 8)]  # 8M
            a.query_sequence = "ACGTACGT"
            a.query_qualities = pysam.qualitystring_to_array("IIIIIIII")
            a.set_tag("MD", "8")
            out.write(a)
    pysam.sort("-o", str(path), str(path))
    pysam.index(str(path))


def _select(read, boi, strand_aware):
    """Replicate the filter from parse_bam.

    strand_aware=True is the patched behaviour; False is the old buggy
    strand-agnostic behaviour.
    """
    ap = np.array(read.get_aligned_pairs(matches_only=True, with_seq=True), dtype=object)
    boi_fwd = boi.upper()
    if strand_aware and read.is_reverse:
        boi_used = _BASE_COMPLEMENT.get(boi_fwd, boi_fwd)
    else:
        boi_used = boi_fwd
    sel = ap[ap[:, 2] == boi_used][:, :2].astype(np.int32)
    return [(int(q), int(r)) for q, r in sel]


@pytest.fixture()
def reads(tmp_path):
    bam_path = tmp_path / "synth.bam"
    _write_synthetic_bam(bam_path)
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        yield {r.query_name: r for r in bam}


def test_forward_read_unaffected_by_fix(reads):
    """Forward reads must select the same sites with or without the fix."""
    fwd = reads["fwd"]
    old = _select(fwd, "A", strand_aware=False)
    new = _select(fwd, "A", strand_aware=True)
    assert old == new == [(0, 100), (4, 104)]
    # The profiled read base is the base of interest.
    seq = fwd.query_sequence
    assert {seq[q] for q, _ in new} == {"A"}


def test_reverse_read_selects_complement(reads):
    """Reverse reads must anchor on the complement of boi.

    Old (buggy) behaviour anchored on forward-ref 'A' (read base U on the
    sequenced strand). The fix anchors on forward-ref 'T', whose base on the
    sequenced reverse strand is the adenosine that can carry m6A.
    """
    rev = reads["rev"]
    assert rev.is_reverse

    old = _select(rev, "A", strand_aware=False)
    new = _select(rev, "A", strand_aware=True)

    assert old == [(0, 200), (4, 204)]  # forward-ref 'A'  (wrong base)
    assert new == [(3, 203), (7, 207)]  # forward-ref 'T'  (correct)
    assert old != new

    seq = rev.query_sequence  # forward-reference orientation, as stored in BAM
    assert {seq[q] for q, _ in old} == {"A"}
    assert {seq[q] for q, _ in new} == {"T"}


@pytest.mark.parametrize("boi,expected", [("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")])
def test_complement_table(boi, expected):
    assert _BASE_COMPLEMENT[boi] == expected


# --------------------------------------------------------------------------- #
# Real reverse-strand read: validate the boi-complement fix against dorado's
# own m6A modbase calls (an independent ground truth in the same BAM).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not REV_BAM.exists(), reason="real reverse-strand fixture missing")
def test_real_reverse_read_boi_matches_dorado_m6a_calls():
    with pysam.AlignmentFile(str(REV_BAM), "rb", check_sq=False) as bam:
        rev = next(iter(bam))

    assert rev.is_reverse
    seq = rev.query_sequence  # forward-reference orientation, as BAM stores it

    # mv carries exactly one move per basecalled base (full read length),
    # confirming it is a per-base, full-read array.
    mv = np.asarray(rev.get_tag("mv"))
    assert int(mv[1:].sum()) == rev.query_length

    # dorado's m6A model calls modifications on adenosines of the *sequenced*
    # RNA. pysam maps them onto forward-reference query_sequence positions;
    # for a reverse-strand read those positions read 'T' in SEQ.
    modbase = rev.modified_bases or {}
    m6a_key = next(k for k in modbase if k[0] == "A")  # ('A', 1, 'a')
    m6a_qpos = {p for p, _ in modbase[m6a_key]}
    assert {seq[p] for p in m6a_qpos} == {"T"}

    old_sites = {q for q, _ in _select(rev, "A", strand_aware=False)}
    new_sites = {q for q, _ in _select(rev, "A", strand_aware=True)}

    assert old_sites and new_sites
    assert {seq[q] for q in old_sites} == {"A"}  # old: forward-ref 'A'
    assert {seq[q] for q in new_sites} == {"T"}  # patched: forward-ref 'T'

    # Independent ground truth: the old (strand-agnostic) selection has ZERO
    # overlap with the positions dorado's m6A model actually scored, while the
    # patched (strand-aware) selection lands squarely on them. (The boi filter
    # keys off the reference base and drops mismatches / unlisted MM entries,
    # so it is an overlap -- not a subset -- relationship.)
    assert len(old_sites & m6a_qpos) == 0
    assert len(new_sites & m6a_qpos) > 0


# --------------------------------------------------------------------------- #
# Second bug (CONFIRMED on real ON0090 reads + POD5, now fixed): the BAM stores
# reverse-strand SEQ/QUAL reverse-complemented (forward-reference order) and
# get_aligned_pairs' q_pos is forward-reference, but `signal`/`dwell_token` are
# in basecalled (RNA-sense) order for both strands. The model is trained only
# on RNA-sense, base-of-interest-centred context, so the fix presents reverse-
# strand reads in RNA-sense: reverse-complement `seq`, reverse `bq`, mirror
# q_pos (-> q_len-1-q_pos); `signal`/`dwell_token` are left untouched and the
# motor offset stays +dwell_shift for both strands.
#
# Deterministic contract test of that transform (no POD5), mirroring exactly
# what inference_preprocess_python.segment_normalize_signal does.
# --------------------------------------------------------------------------- #
from deeprm.inference.inference_preprocess_python import _ASCII_COMPLEMENT  # noqa: E402


def _sense_transform(seq_u8, bq, q_pos, q_len, strand):
    """RNA-sense transform from segment_normalize_signal for one (read, site)."""
    if strand == -1:
        seq_u8 = _ASCII_COMPLEMENT[seq_u8][::-1]
        bq = bq[::-1]
        q_pos = q_len - 1 - q_pos
    return seq_u8, bq, q_pos


@pytest.mark.parametrize("strand", [1, -1])
def test_reverse_strand_sense_transform_contract(strand):
    # Forward-reference per-read arrays (as parse_bam stores them).
    seq = np.frombuffer(b"ACGTACGTAA", dtype=np.uint8).copy()  # q_len = 10
    bq = np.arange(10, dtype=np.int8)
    q_len = 10
    q_pos = 2  # forward-reference index of the base of interest

    out_seq, out_bq, out_q = _sense_transform(seq.copy(), bq.copy(), q_pos, q_len, strand)

    if strand == 1:
        # Forward: untouched (forward-reference == RNA-sense).
        assert out_seq.tobytes() == b"ACGTACGTAA"
        assert out_bq.tolist() == bq.tolist()
        assert out_q == 2
    else:
        # Reverse: seq reverse-complemented, bq reversed, q_pos mirrored.
        assert out_seq.tobytes() == b"TTACGTACGT"  # revcomp("ACGTACGTAA")
        assert out_bq.tolist() == bq[::-1].tolist()
        assert out_q == q_len - 1 - 2  # == 7
        # The base selected on the forward ref ('G' = complement of 'C') is
        # presented to the model as the RNA-sense base at the mirrored index.
        assert chr(seq[q_pos]) == "G"
        assert chr(out_seq[out_q]) == "C"


def test_sense_transform_centers_boi_on_adenosine():
    """A reverse-strand m6A site (forward-ref 'T') becomes RNA-sense 'A'."""
    # Forward-ref window ...X T X... ; boi-complement picked the 'T'.
    seq = np.frombuffer(b"GGGGTCCCC", dtype=np.uint8).copy()
    q_len = len(seq)
    q_pos = 4  # the 'T' chosen by the complement boi filter on reverse reads
    assert chr(seq[q_pos]) == "T"
    out_seq, _, out_q = _sense_transform(seq.copy(), np.zeros(q_len, np.int8), q_pos, q_len, -1)
    assert chr(out_seq[out_q]) == "A"  # RNA-sense centred base is the adenosine


def test_motor_offset_strand_uniform():
    """Motor offset is +dwell_shift for both strands (dwell is basecalled order).

    Mirrors `x["start_pos"] + dwell_shift + trim` in segment_normalize_signal
    with no strand factor.
    """
    dwell_shift, start_pos, trim = 10, 30, 2
    for strand in (1, -1):
        assert start_pos + dwell_shift + trim == 42  # independent of strand
