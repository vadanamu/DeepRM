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
    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 1000}]}
    )
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
