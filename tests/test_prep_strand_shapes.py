"""Strand and alignment-shape test of the preprocessing (Python and binary), against an independent oracle.

A synthetic POD5 + BAM holds forward and reverse reads of 7 alignment shapes (plain match, soft clips, indels with
mismatches, a splice, low-quality ends, and a hard clip before or after a soft clip), 3 copies each. The fixture keeps
the truth: each read's basecalled (RNA-sense) sequence, qualities and per-base dwell. Checks:
  1. the Python preprocessing and the binary give the same rows, read by read, for every shape and strand;
  2. every Python row is the RNA-sense window centred on an A: its k-mer, base-quality and segment-length tokens come
     from the basecalled read, and its label is the forward-reference position of that base with the strand as sign;
     without padding the set of sites per read is exactly the expected one (hard-clipped shapes are left out here:
     their feature shift is a known limit, see CHANGELOG);
  3. --trim removes sites at the low-quality read ends (on both strands).
"""

import argparse
import array
import datetime
import glob
import json
import multiprocessing as mp
import os
import subprocess
import uuid
from collections import defaultdict

import numpy as np
import pytest

from deeprm.inference import inference_preprocess as wrapper
from deeprm.inference import inference_preprocess_python as py_prep
from deeprm.utils import prep_binary

M, INS, D, N, S, H = 0, 1, 2, 3, 4, 5
SHAPES = {
    "M": [(M, 220)],
    "S": [(S, 6), (M, 220), (S, 9)],
    "indel": [(S, 4), (M, 70), (INS, 3), (M, 60), (D, 4), (M, 90), (S, 2)],
    "splice": [(M, 90), (N, 700), (M, 130)],
    "lowq": [(M, 260)],
    "HS": [(H, 4), (S, 3), (M, 220)],
    "SH": [(M, 220), (S, 3), (H, 4)],
}
REF_LEN = 20000
COMP = str.maketrans("ACGT", "TGCA")


def _rc(s):
    return s.translate(COMP)[::-1]


def _write_fixture(out):
    """Runs in a child process: the main process must never open a POD5 file (see the Python preprocessing)."""
    import pod5
    import pysam

    rng = np.random.default_rng(11)
    ref = "".join(rng.choice(list("ACGT"), REF_LEN))

    def build(ops, start, mism_every=0):
        full, seq, md, run, rp, hl, hr = [], [], [], 0, start, 0, 0
        for k, (op, n) in enumerate(ops):
            if op == M:
                for j in range(n):
                    b = ref[rp]
                    if mism_every and (j + 1) % mism_every == 0:
                        nb = rng.choice([x for x in "ACGT" if x != b])
                        seq.append(nb); full.append(nb); md += [str(run), b]; run = 0  # noqa: E702
                    else:
                        seq.append(b); full.append(b); run += 1  # noqa: E702
                    rp += 1
            elif op in (INS, S):
                r = "".join(rng.choice(list("ACGT"), n))
                seq.append(r); full.append(r)  # noqa: E702
            elif op == D:
                md += [str(run), "^" + ref[rp:rp + n]]; run = 0; rp += n  # noqa: E702
            elif op == N:
                rp += n
            elif op == H:
                full.append("".join(rng.choice(list("ACGT"), n)))
                hl, hr = (n, hr) if k == 0 else (hl, n)
        md.append(str(run))
        return "".join(full), "".join(seq), "".join(md), hl, hr

    os.makedirs(os.path.join(out, "pod5"), exist_ok=True)
    now = datetime.datetime(2026, 1, 1)
    run_info = pod5.RunInfo(
        acquisition_id="a" * 32, acquisition_start_time=now, adc_max=4095, adc_min=-4096, context_tags={},
        experiment_name="t", flow_cell_id="FC1", flow_cell_product_code="FLO-PRO004RA", protocol_name="rna",
        protocol_run_id="p" * 32, protocol_start_time=now, sample_id="s1", sample_rate=4000,
        sequencing_kit="sqk-rna004", sequencer_position="1A", sequencer_position_type="promethion",
        software="test", system_name="t", system_type="t", tracking_id={},
    )
    truth, records = {}, []
    with pod5.Writer(os.path.join(out, "pod5", "0.pod5")) as w:
        for kind, ops in SHAPES.items():
            for strand in ("fwd", "rev"):
                for _ in range(3):
                    start = int(rng.integers(100, REF_LEN - 2000))
                    full_bam, seq, md, hl, hr = build(ops, start, mism_every=(17 if kind == "indel" else 0))
                    n_full = len(full_bam)
                    basecalled = full_bam if strand == "fwd" else _rc(full_bam)  # RNA-sense 5' -> 3'
                    q_rna = rng.integers(20, 40, n_full)
                    if kind == "lowq":
                        q_rna[:25] = 2  # RNA 5' end
                        q_rna[-12:] = 2  # RNA 3' end
                    q_bam = q_rna if strand == "fwd" else q_rna[::-1]
                    qual = q_bam[hl:n_full - hr]
                    dwell = rng.integers(2, 10, n_full)  # time order: index 0 is the RNA 3' end
                    mv = np.zeros(int(dwell.sum()), dtype=np.uint8)
                    mv[np.concatenate([[0], np.cumsum(dwell)[:-1]])] = 1
                    ts = 100
                    signal = rng.integers(40, 160, ts + mv.size * 6).astype(np.int16)
                    rid = uuid.UUID(int=int(rng.integers(1, 2**62)) << 64 | len(truth))
                    w.add_read(pod5.Read(
                        read_id=rid, pore=pod5.Pore(channel=1, well=1, pore_type="rna"),
                        calibration=pod5.Calibration(offset=0.0, scale=1.0), read_number=1, start_sample=0,
                        median_before=100.0,
                        end_reason=pod5.EndReason(reason=pod5.EndReasonEnum.SIGNAL_POSITIVE, forced=False),
                        run_info=run_info, signal=signal))
                    truth[str(rid)] = dict(kind=kind, strand=strand, start=start, cigar=ops, seq=seq,
                                           basecalled=basecalled, q_rna=q_rna.tolist(), dwell=dwell.tolist())
                    records.append(dict(name=str(rid), flag=16 if strand == "rev" else 0, pos=start, seq=seq,
                                        cigar=ops, md=md, qual=qual, mv=array.array("B", [6] + mv.tolist()),
                                        ts=ts, ns=ts + mv.size * 6))
    header = {"HD": {"VN": "1.6", "SO": "unsorted"}, "SQ": [{"SN": "chr1", "LN": REF_LEN}]}
    unsorted = os.path.join(out, "unsorted.bam")
    with pysam.AlignmentFile(unsorted, "wb", header=header) as bam:
        for r in records:
            a = pysam.AlignedSegment()
            a.query_name = r["name"]; a.flag = r["flag"]; a.reference_id = 0; a.reference_start = r["pos"]  # noqa: E702
            a.mapping_quality = 60; a.cigartuples = r["cigar"]; a.query_sequence = r["seq"]  # noqa: E702
            a.query_qualities = array.array("B", [int(x) for x in r["qual"]])
            a.set_tag("MD", r["md"], value_type="Z"); a.set_tag("mv", r["mv"])  # noqa: E702
            a.set_tag("ts", r["ts"], value_type="i"); a.set_tag("ns", r["ns"], value_type="i")  # noqa: E702
            a.set_tag("sp", 0, value_type="i")
            bam.write(a)
    pysam.sort("-o", os.path.join(out, "reads.bam"), unsorted)
    pysam.index(os.path.join(out, "reads.bam"))
    json.dump(dict(ref=ref, reads=truth), open(os.path.join(out, "truth.json"), "w"))


@pytest.fixture(scope="module")
def shapes(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("strand_shapes"))
    p = mp.get_context("spawn").Process(target=_write_fixture, args=(d,))
    p.start(); p.join()  # noqa: E702
    assert p.exitcode == 0
    return d, json.load(open(os.path.join(d, "truth.json")))


def _rows(folder):
    rows = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        with np.load(path) as z:
            zz = {k: z[k] for k in z.files}
        for i in range(len(zz["label_id"])):
            rid = str(uuid.UUID(bytes=np.asarray(zz["read_id"][i], np.int64).tobytes()))
            rows[rid].append(dict(
                label=int(zz["label_id"][i]), kmer=zz["kmer_token"][i].tobytes(), bq=zz["bq_token"][i].tolist(),
                seg=zz["segment_len_arr"][i].tolist(),
                raw=b"".join(np.ascontiguousarray(zz[f][i]).tobytes()
                             for f in ("label_id", "segment_len_arr", "kmer_token", "bq_token", "signal_token"))))
    return rows


def _run(d, out, extra):
    p = argparse.ArgumentParser()
    py_prep.add_arguments(p)
    args = p.parse_args(["-p", os.path.join(d, "pod5"), "-b", os.path.join(d, "reads.bam"), "-o", str(out / "py"),
                         "-t", "2", "-a", "1", *extra])
    assert int(py_prep.main(args) or 0) == 0
    return _rows(str(out / "py"))


def _run_binary(d, out, extra):
    st = prep_binary.status(wrapper.DEFAULT_BIN)
    if not st.ok:
        if os.environ.get("DEEPRM_REQUIRE_BINARY") == "1":
            pytest.fail(f"the bundled preprocessing binary cannot start here: {st.reason}")
        pytest.skip(f"the bundled preprocessing binary cannot start here: {st.reason}")
    p = argparse.ArgumentParser()
    wrapper.add_arguments(p)
    args = p.parse_args(["-p", os.path.join(d, "pod5"), "-b", os.path.join(d, "reads.bam"), "-o", str(out / "bin"),
                         "-t", "2", "-a", "1", *extra])
    done = subprocess.run([wrapper.DEFAULT_BIN] + wrapper._args_to_cmd(args), capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    return _rows(str(out / "bin"))


def _aligned_pairs(t):
    q, r, out = 0, t["start"], []
    for op, n in t["cigar"]:
        if op == M:
            out += [(q + j, r + j) for j in range(n)]
            q += n; r += n  # noqa: E702
        elif op in (INS, S):
            q += n
        elif op in (D, N):
            r += n
    return out


def _oracle_failures(rows, truth, padded, trimmed):
    """Rows (and, without padding, site sets) that differ from the RNA-sense truth; hard-clipped shapes left out."""
    ref, reads, bad = truth["ref"], truth["reads"], []
    for rid, t in reads.items():
        if t["kind"] in ("HS", "SH"):
            continue
        rev = t["strand"] == "rev"
        seq, n = t["seq"], len(t["seq"])
        rna, q_rna, dwell = t["basecalled"], np.array(t["q_rna"]), np.array(t["dwell"])[::-1]
        boi_ref = "T" if rev else "A"
        cand = {r: ((n - 1 - q) if rev else q) for q, r in _aligned_pairs(t) if ref[r] == boi_ref and seq[q] == ref[r]}
        if trimmed:
            tl, tr = py_prep.quality_trimming(rna, q_rna.astype(np.int32), penalty=-15, threshold=5)
            lb, rb = (tl, tr) if (tl, tr) != (0, 0) else (0, n)
        else:
            lb, rb = 1, n
        for row in rows.get(rid, []):
            strand, r = (-1 if row["label"] < 0 else 1), abs(row["label"]) - 1
            ok = strand == (-1 if rev else 1) and r in cand
            if ok:
                j = cand[r]
                if padded:
                    ok = chr(row["kmer"][10]) == "A" and rna[j] == "A"
                else:
                    ok = (row["kmer"] == rna[j - 10:j + 11].encode()
                          and row["bq"] == [min(60, int(x)) for x in q_rna[j - 8:j + 9]]
                          and row["seg"] == dwell[j - 8:j + 9].tolist())
            if not ok:
                bad.append((t["kind"], t["strand"], "row"))
        if not padded:
            expected = {r for r, j in cand.items() if j - 10 >= lb and j + 11 <= rb and j + 19 <= rb and j + 2 >= lb}
            if expected != {abs(x["label"]) - 1 for x in rows.get(rid, [])}:
                bad.append((t["kind"], t["strand"], "site set"))
    return bad


SETTINGS = [["-g", "260"], ["-g", "260", "--trim"], ["-g", "260", "--zero-pad"], ["-g", "260", "--edge-pad", "--trim"]]


@pytest.mark.parametrize("extra", SETTINGS, ids=lambda e: " ".join(e))
def test_shapes_and_strands_against_the_truth(shapes, tmp_path, extra):
    d, truth = shapes
    rows = _run(d, tmp_path, extra)
    kinds = {(t["kind"], t["strand"]) for rid, t in truth["reads"].items() if rows.get(rid)}
    assert {(k, s) for k in SHAPES for s in ("fwd", "rev") if k not in ("HS", "SH")} <= kinds
    padded = "--zero-pad" in extra or "--edge-pad" in extra
    assert _oracle_failures(rows, truth, padded, "--trim" in extra) == []


@pytest.mark.parametrize("extra", SETTINGS + [[]], ids=lambda e: " ".join(e) or "default")
def test_binary_matches_python_for_every_shape(shapes, tmp_path, extra):
    d, truth = shapes
    py_rows, bin_rows = _run(d, tmp_path, extra), _run_binary(d, tmp_path, extra)
    differ = sorted({(truth["reads"][rid]["kind"], truth["reads"][rid]["strand"])
                     for rid in set(py_rows) | set(bin_rows)
                     if sorted(x["raw"] for x in py_rows.get(rid, []))
                     != sorted(x["raw"] for x in bin_rows.get(rid, []))})
    assert sum(map(len, py_rows.values())) > 0
    assert differ == []


def test_trim_removes_sites_at_low_quality_ends(shapes, tmp_path):
    d, truth = shapes
    plain, trimmed = _run(d, tmp_path / "plain", ["-g", "260"]), _run(d, tmp_path / "trim", ["-g", "260", "--trim"])
    for strand in ("fwd", "rev"):
        ids = [rid for rid, t in truth["reads"].items() if t["kind"] == "lowq" and t["strand"] == strand]
        assert sum(len(trimmed.get(r, [])) for r in ids) < sum(len(plain.get(r, [])) for r in ids)
