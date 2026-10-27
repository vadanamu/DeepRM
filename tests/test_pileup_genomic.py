"""The genomic pileup (deeprm call pileup --annot): transcript sites mapped to genomic positions, their sums added up,
and the genomic table scored and written the same way as the transcript table."""

import argparse
import json
import logging

import numpy as np
import pandas as pd
import pysam
import pytest

from deeprm.inference import pileup_deeprm
from deeprm.inference import pileup_genomic as pileup_genomic_module
from deeprm.inference.pileup_deeprm import main as pileup_main
from deeprm.inference.pileup_genomic import TranscriptMapper, _map_to_genome

LABEL_DIV = 10**9
# Three transcripts on chr1. tx1 (+) has two exons; tx3 (+) starts inside tx1's first exon, so tx1 position 5 and
# tx3 position 0 are the same genomic position (1005, + strand). tx2 (-) has one exon: its position 0 is 1014.
REFFLAT = [
    ("g1", "tx1", "chr1", "+", 1000, 1030, 1000, 1030, 2, "1000,1020,", "1010,1030,"),
    ("g2", "tx2", "chr1", "-", 1005, 1015, 1005, 1015, 1, "1005,", "1015,"),
    ("g3", "tx3", "chr1", "+", 1005, 1025, 1005, 1025, 1, "1005,", "1025,"),
]
BAM_REFS = ["tx1", "tx2", "tx3"]


def _write_refflat(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write("\t".join(str(x) for x in row) + "\n")


def _write_bam(path, refs):
    header = pysam.AlignmentHeader.from_dict({"HD": {"VN": "1.6"}, "SQ": [{"SN": r, "LN": 5000} for r in refs]})
    with pysam.AlignmentFile(str(path), "wb", header=header):
        pass
    pysam.index(str(path))


def _label(ref_id, pos):
    return ref_id * LABEL_DIV + pos + 1  # + strand; 1 is added so that no label is 0


def _write_reads(path, sites, seed=0):
    """sites: [(ref_id, pos, depth)]; random per-read probabilities, some confident."""
    rng = np.random.default_rng(seed)
    label, pred = [], []
    for ref_id, pos, depth in sites:
        label.append(np.full(depth, _label(ref_id, pos), dtype=np.int64))
        logit = np.where(rng.random(depth) < 0.5, rng.uniform(5, 9, depth), rng.uniform(-9, -4, depth))
        pred.append(pileup_deeprm._sigmoid_np(logit.astype(np.float32)))
    _save_reads(path, np.concatenate(label), np.concatenate(pred), rng)


def _save_reads(path, label, pred, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    np.savez(
        path,
        label_id=np.asarray(label, dtype=np.int64),
        read_id=rng.integers(0, 2**62, size=(len(label), 2), dtype=np.int64),
        pred=np.asarray(pred, dtype=np.float32),
    )


def _args(inp, out, bam, annot, thread=2):
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
        annot=str(annot),
        skip_modbam=True,
    )


@pytest.fixture
def example(tmp_path):
    inp = tmp_path / "molecule-level"
    inp.mkdir()
    # tx1 pos 5 and tx3 pos 0 share chr1:1005(+); tx2 pos 0 is chr1:1014(-); tx1 pos 12 is chr1:1022(+)
    _write_reads(inp / "inference_0_0.npz", [(0, 5, 30), (1, 0, 25), (0, 12, 12)], seed=1)
    _write_reads(inp / "inference_0_1.npz", [(2, 0, 40), (0, 5, 7)], seed=2)
    bam = tmp_path / "x.bam"
    _write_bam(bam, BAM_REFS)
    annot = tmp_path / "refFlat.txt"
    _write_refflat(annot, REFFLAT)
    return tmp_path, inp, bam, annot


def test_genomic_pileup_adds_up_sites_and_scores_them_like_transcripts(example):
    tmp_path, inp, bam, annot = example
    out = tmp_path / "site-level"
    pileup_main(_args(inp, out, bam, annot))

    tx = np.load(out / "pileup.npz", allow_pickle=False)
    gen = np.load(out / "genomic_pileup.npz", allow_pickle=False)  # strings, not objects
    assert gen["ref_names"].dtype.kind == "U" and gen["ref_strand"].dtype.kind == "U"
    assert gen.files == tx.files, "both tables have the same arrays in the same order"
    for z in (tx, gen):
        for key in z.files:
            z[key]  # every array loads without allow_pickle

    site = {
        (str(c), str(s), int(p)): i
        for i, (c, s, p) in enumerate(zip(gen["ref_names"], gen["ref_strand"], gen["ref_pos"]))
    }
    assert set(site) == {("chr1", "+", 1005), ("chr1", "-", 1014), ("chr1", "+", 1022)}

    tx_row = {(str(n), int(p)): i for i, (n, p) in enumerate(zip(tx["ref_names"], tx["ref_pos"]))}
    shared = [tx_row[("tx1", 5)], tx_row[("tx3", 0)]]
    g = site[("chr1", "+", 1005)]
    for key in ("count_all", "count_pos", "logsum_1_p_pos", "kl_div_neg", "kl_div_pos"):
        assert np.isclose(gen[key][g], tx[key][shared].sum(), rtol=0, atol=1e-9), key
    assert gen["count_all"][g] == 30 + 7 + 40
    g2 = site[("chr1", "-", 1014)]
    for key in tx.files:
        if key not in ("ref_names", "ref_pos", "ref_strand"):
            assert gen[key][g2] == tx[key][tx_row[("tx2", 0)]] or (
                np.isnan(gen[key][g2]) and np.isnan(tx[key][tx_row[("tx2", 0)]])
            ), key

    # the genomic scores are _site_scores of the genomic sums (the rule-only sums are not written, so recompute them)
    sums = {key: gen[key] for key in ("count_all", "count_pos", "logsum_1_p_pos", "kl_div_neg", "kl_div_pos")}
    for key, _ in pileup_deeprm._RULE_SUMS:  # not written; modscore and stoichiometry_eq14 do not use them
        sums[key] = np.zeros(len(gen["count_all"]))
    scores = pileup_deeprm._site_scores(sums, 0.98, 1e-30)
    assert np.array_equal(gen["modscore"], scores["modscore"])
    assert np.array_equal(gen["stoichiometry_eq14"], scores["stoichiometry_eq14"])

    bed = (out / "genomic_pileup.bed").read_text().splitlines()
    assert len(bed) == 3 and all(len(line.split("\t")) == 18 for line in bed)


def test_genomic_pileup_without_any_annotated_transcript_says_so(example):
    tmp_path, inp, bam, _ = example
    other = tmp_path / "other_refFlat.txt"
    _write_refflat(other, [("g9", "tx9", "chr2", "+", 0, 100, 0, 100, 1, "0,", "100,")])
    with pytest.raises(ValueError, match="No valid genomic position"):
        pileup_main(_args(inp, tmp_path / "out", bam, other))


def test_a_strand_other_than_plus_or_minus_is_refused_and_named(example):
    tmp_path, inp, bam, _ = example
    rows = list(REFFLAT) + [("g4", "tx4", "chr1", ".", 2000, 2100, 2000, 2100, 1, "2000,", "2100,")]
    dot = tmp_path / "dot_refFlat.txt"
    _write_refflat(dot, rows)
    with pytest.raises(ValueError, match="Invalid strand '.' for transcript tx4"):
        pileup_main(_args(inp, tmp_path / "out", bam, dot))


def test_vectorised_mapping_matches_transcript_mapper():
    """_map_to_genome maps every site as TranscriptMapper does, on random transcripts of both strands."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(300):
        n = int(rng.integers(1, 6))
        bounds = np.sort(rng.choice(np.arange(1000, 100000), 2 * n, replace=False))
        rows.append((f"tx{i}", f"chr{i % 3}", "+" if rng.random() < 0.5 else "-", bounds[0::2], bounds[1::2]))
    refflat = pd.DataFrame(rows, columns=["transcript_id", "chrom", "strand", "exonStarts", "exonEnds"])
    refflat = refflat.set_index("transcript_id")
    names = np.array([f"tx{i}" for i in rng.integers(0, 320, 20000)])  # tx300+ are not annotated
    positions = rng.integers(-5, 3000, len(names))
    sites, chrom, strand, pos = _map_to_genome(names, positions, refflat)

    expected = []
    for i, (name, p) in enumerate(zip(names, positions)):
        if name not in refflat.index:
            continue
        row = refflat.loc[name]
        g = TranscriptMapper(row["exonStarts"], row["exonEnds"], row["strand"]).map(np.array([p]))[0]
        if not np.isnan(g):
            expected.append((i, row["chrom"], row["strand"], int(g)))
    got = list(zip(sites.tolist(), chrom.tolist(), strand.tolist(), pos.tolist()))
    assert got == expected and len(expected) > 1000


def test_ensembl_names_match_without_their_version(example):
    tmp_path, inp, bam, annot = example
    out = tmp_path / "site-level"
    pileup_main(_args(inp, out, bam, annot))
    # the same reads, aligned to versioned Ensembl names, against the unversioned annotation
    versioned = tmp_path / "versioned.bam"
    _write_bam(versioned, ["ENST00000000001.4", "ENST00000000002.1", "NM_000003.2"])
    unversioned = tmp_path / "unversioned_refFlat.txt"
    renamed = {"tx1": "ENST00000000001", "tx2": "ENST00000000002", "tx3": "NM_000003.2"}
    _write_refflat(unversioned, [(r[0], renamed[r[1]], *r[2:]) for r in REFFLAT])
    out_v = tmp_path / "site-level-versioned"
    pileup_main(_args(inp, out_v, versioned, unversioned))
    a, b = np.load(out / "genomic_pileup.npz"), np.load(out_v / "genomic_pileup.npz")
    for key in a.files:
        assert np.array_equal(a[key], b[key], equal_nan=a[key].dtype.kind == "f"), key


def _row_of_site(table):
    """{(name, strand, position): row} of a site table."""
    sites = zip(table["ref_names"], table["ref_strand"], table["ref_pos"])
    return {(str(n), str(s), int(p)): i for i, (n, s, p) in enumerate(sites)}


def test_sites_on_the_strand_opposite_the_transcript_get_rows_of_their_own(example):
    """A '-' label is an A on the strand opposite the transcript (a read aligned in reverse, kept with -g 260): the
    same genomic base, on the other genomic strand. It is never added to the transcript-strand site there."""
    tmp_path, _, bam, annot = example
    inp = tmp_path / "both-strands"
    inp.mkdir()
    # tx1 (+) pos 5 is chr1:1005: 10 '+' reads at p = 0.99 and 7 '-' reads at p = 0.01.
    # tx2 (-) pos 0 is chr1:1014: 6 '-' reads at p = 0.99, so its A is on the + strand of the genome.
    label = [_label(0, 5)] * 10 + [-_label(0, 5)] * 7 + [-_label(1, 0)] * 6
    pred = [0.99] * 10 + [0.01] * 7 + [0.99] * 6
    _save_reads(inp / "inference_0_0.npz", label, pred)
    out = tmp_path / "site-level"
    pileup_main(_args(inp, out, bam, annot))

    tx = np.load(out / "pileup.npz")
    gen = np.load(out / "genomic_pileup.npz")
    tx_row, gen_row = _row_of_site(tx), _row_of_site(gen)
    expected = {
        ("tx1", "+", 5): ("chr1", "+", 1005),
        ("tx1", "-", 5): ("chr1", "-", 1005),
        ("tx2", "-", 0): ("chr1", "+", 1014),
    }
    assert set(tx_row) == set(expected)
    assert set(gen_row) == set(expected.values()), "one genomic row per transcript site, on its own strand"
    for tx_site, gen_site in expected.items():
        t, g = tx_row[tx_site], gen_row[gen_site]
        for key in tx.files:
            if key not in ("ref_names", "ref_pos", "ref_strand"):
                same = gen[key][g] == tx[key][t] or (np.isnan(gen[key][g]) and np.isnan(tx[key][t]))
                assert same, (tx_site, key)
    assert gen["count_all"][gen_row[("chr1", "+", 1005)]] == 10 and gen["count_pos"][gen_row[("chr1", "-", 1005)]] == 0
    bed = [line.split("\t") for line in (out / "genomic_pileup.bed").read_text().splitlines()]
    assert sorted((b[0], b[5], int(b[1])) for b in bed) == sorted(expected.values())


def test_refseq_names_match_without_their_version(example):
    """A UCSC refFlat lists RefSeq names without the version (NM_005517) that a RefSeq transcriptome has
    (NM_005517.4); the genomic table is the same as with matching names."""
    tmp_path, inp, bam, annot = example
    out = tmp_path / "site-level"
    pileup_main(_args(inp, out, bam, annot))
    versioned = tmp_path / "refseq.bam"
    _write_bam(versioned, ["NM_005517.4", "XM_024451418.1", "NR_046018.12"])
    unversioned = tmp_path / "refseq_refFlat.txt"
    renamed = {"tx1": "NM_005517", "tx2": "XM_024451418", "tx3": "NR_046018"}
    _write_refflat(unversioned, [(r[0], renamed[r[1]], *r[2:]) for r in REFFLAT])
    out_v = tmp_path / "site-level-refseq"
    pileup_main(_args(inp, out_v, versioned, unversioned))
    a, b = np.load(out / "genomic_pileup.npz"), np.load(out_v / "genomic_pileup.npz")
    for key in a.files:
        assert np.array_equal(a[key], b[key], equal_nan=a[key].dtype.kind == "f"), key


class _LogRecords(logging.Handler):
    """Collects the records of the genomic step's logger (it writes to standard error and does not propagate)."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def genomic_log():
    handler = _LogRecords()
    pileup_genomic_module.log.addHandler(handler)
    yield handler.records
    pileup_genomic_module.log.removeHandler(handler)


def _refflat_frame(rows):
    """rows: [(name, chrom, strand, exon starts, exon ends)] -> the frame parse_refflat returns (the columns used)."""
    frame = pd.DataFrame(
        [(n, c, s, np.array(a, dtype=np.int64), np.array(b, dtype=np.int64)) for n, c, s, a, b in rows],
        columns=["transcript_id", "chrom", "strand", "exonStarts", "exonEnds"],
    )
    return frame.set_index("transcript_id")


def test_names_are_matched_exactly_first_then_without_any_version(genomic_log):
    refflat = _refflat_frame(
        [
            ("NM_005517", "chr1", "+", [100], [200]),
            ("ENST00000000001", "chr2", "+", [100], [200]),
            ("tx.7", "chr3", "+", [100], [200]),  # the exact name wins over the name without its version
            ("tx", "chr4", "+", [100], [200]),
        ]
    )
    names = np.array(["NM_005517.4", "ENST00000000001.3", "tx.7", "NM_000001.1", "tx9"])
    sites, chrom, _, pos = _map_to_genome(names, np.full(len(names), 10), refflat)
    assert sites.tolist() == [0, 1, 2] and chrom.tolist() == ["chr1", "chr2", "chr3"] and pos.tolist() == [110] * 3
    [summary] = [r for r in genomic_log if r.getMessage().startswith("Genomic pileup:")]
    assert summary.levelno == logging.WARNING, "transcripts that were not found are a warning"
    assert "1 of 5 transcripts matched the annotation by name, 2 after removing the version, 2 not found" in (
        summary.getMessage()
    )
    assert "NM_000001.1" in summary.getMessage(), "the warning names a transcript that was not found"


def test_all_names_found_is_not_a_warning(genomic_log):
    refflat = _refflat_frame([("NM_005517", "chr1", "+", [100], [200])])
    _map_to_genome(np.array(["NM_005517"]), np.array([10]), refflat)
    [summary] = [r for r in genomic_log if r.getMessage().startswith("Genomic pileup:")]
    assert summary.levelno == logging.INFO and "0 not found" in summary.getMessage()


def _write_run_info(folder, model_name):
    with open(folder / "deeprm_run.json", "w") as f:
        json.dump({"model_name": model_name}, f)


def test_calibrated_stoichiometry_of_a_position_shared_by_two_transcripts(example):
    """At a genomic position of two transcripts, the stoichiometry rule reads the sum of both transcripts' sums,
    the calibrated evidence included (the v1.2 model's predictions, named in deeprm_run.json)."""
    tmp_path, _, bam, annot = example
    inp = tmp_path / "calibrated"
    inp.mkdir()
    _write_run_info(inp, "deeprm_v12")
    # tx1 pos 5 and tx3 pos 0 are both chr1:1005(+); each has 12 confident and 12 negative reads
    rng = np.random.default_rng(0)
    preds = {}
    for ref_id, pos in ((0, 5), (2, 0)):
        logit = np.concatenate([rng.uniform(5, 9, 12), rng.uniform(-9, -4, 12)]).astype(np.float32)
        preds[(ref_id, pos)] = pileup_deeprm._sigmoid_np(logit)
        _save_reads(inp / f"inference_0_{ref_id}.npz", [_label(ref_id, pos)] * 24, preds[(ref_id, pos)], rng)
    out = tmp_path / "site-level"
    pileup_main(_args(inp, out, bam, annot))

    # each transcript site's sums, recomputed from its reads as a worker sums them (the rule's own sums are not
    # written), then added up
    per_site = []
    for p in preds.values():
        terms = pileup_deeprm._per_read_terms(p, 0.98, 1e-30, True)
        one_site = np.zeros(len(p), dtype=np.int64)
        per_site.append(
            {key: pileup_deeprm.grouped_sum(1, one_site, terms[key]).astype(np.float64) for key in terms}
            | {"count_all": np.array([len(p)])}
        )
    summed = {key: per_site[0][key] + per_site[1][key] for key in per_site[0]}
    expected = pileup_deeprm._site_scores(summed, 0.98, 1e-30)
    first_only = pileup_deeprm._site_scores(
        summed | {"logsum_1_p_pos_calib": per_site[0]["logsum_1_p_pos_calib"]}, 0.98, 1e-30
    )
    reported = bool(expected["stoich_state"][0] == 1)
    sensitive = bool(first_only["stoich_state"][0] != expected["stoich_state"][0])

    gen = np.load(out / "genomic_pileup.npz")
    [g] = np.flatnonzero((gen["ref_strand"] == "+") & (gen["ref_pos"] == 1005))
    for key in ("count_all", "count_pos", "logsum_1_p_pos", "kl_div_neg", "kl_div_pos"):
        assert np.isclose(gen[key][g], summed[key][0], rtol=1e-12, atol=0), key
    same_state = bool(gen["stoich_state"][g] == expected["stoich_state"][0])
    same_release = bool(
        np.allclose(gen["stoichiometry_release"][g], expected["stoichiometry_release"][0], rtol=1e-12, equal_nan=True)
    )
    assert reported, "the summed evidence of both transcripts reports the site"
    assert sensitive, "one transcript's calibrated evidence alone would not report it, so the test can tell"
    assert same_state, "the genomic stoich_state is the rule on the summed transcript sums"
    assert same_release, "the genomic stoichiometry_release is the rule on the summed transcript sums"


REFFLAT_COLUMNS = [
    "transcript_id",
    "chrom",
    "strand",
    "txStart",
    "txEnd",
    "cdsStart",
    "cdsEnd",
    "exonCount",
    "exonStarts",
    "exonEnds",
]


def _random_refflat(rng, n):
    """Random refFlat rows (as parse_refflat reads them): valid ones, and ones with exons in descending order, an
    empty or a negative exon, an exon whose end goes back, a wrong exonCount, or an empty exon list."""
    rows = []
    for i in range(n):
        k = int(rng.integers(1, 6))
        bounds = np.sort(rng.choice(np.arange(1000, 100000), 2 * k, replace=False))
        starts, ends = bounds[0::2].copy(), bounds[1::2].copy()
        count = k
        j = int(rng.integers(k))
        kind = int(rng.integers(0, 9))
        if kind == 1 and k > 1:
            starts, ends = starts[::-1], ends[::-1]  # descending exons
        elif kind == 2:
            ends[j] = starts[j]  # an empty exon
        elif kind == 3:
            ends[j] = starts[j] - int(rng.integers(1, 50))  # a negative exon
        elif kind == 4 and j > 0:
            starts[j], ends[j] = starts[j - 1], ends[j - 1] - 1  # inside the exon before it: its end goes back
        elif kind == 5 and j > 0:
            starts[j] = starts[j - 1]  # the same start as the exon before it, a later end: kept
        elif kind == 6:
            count = k + int(rng.choice([-1, 1]))  # wrong exonCount (0 for one exon: dropped before the exon checks)
        elif kind == 7:
            starts = ends = np.array([], dtype=np.int64)  # an empty exon list
        comma = "," if rng.random() < 0.7 else ""
        text = [",".join(map(str, x)) + (comma if len(x) else "") for x in (starts, ends)]
        rows.append((f"g{i}", f"tx{i}", "chr1", "+-"[i % 2], 0, 100000, 0, 100000, count, *text))
    return rows


def _same_annotation(a, b):
    return (
        a.index.tolist() == b.index.tolist()
        and a.drop(columns=["exonStarts", "exonEnds"]).equals(b.drop(columns=["exonStarts", "exonEnds"]))
        and all(np.array_equal(x, y) and x.dtype == y.dtype for x, y in zip(a["exonStarts"], b["exonStarts"]))
        and all(np.array_equal(x, y) and x.dtype == y.dtype for x, y in zip(a["exonEnds"], b["exonEnds"]))
    )


def test_vectorised_exon_checks_keep_the_rows_the_row_by_row_checks_keep(tmp_path, monkeypatch):
    rows = _random_refflat(np.random.default_rng(0), 600)
    frame = pd.DataFrame([r[1:] for r in rows], columns=REFFLAT_COLUMNS)
    frame = frame[frame["exonCount"] > 0]  # as parse_refflat, before the exon checks
    starts = pileup_genomic_module._parse_exon_lists(frame["exonStarts"])
    ends = pileup_genomic_module._parse_exon_lists(frame["exonEnds"])
    assert starts is not None and ends is not None, "plain exon lists take the vectorised path"
    vectorised = pileup_genomic_module._keep_valid_exons(frame, starts, ends)
    by_row = pileup_genomic_module._keep_valid_exons_by_row(frame)
    assert 100 < len(by_row) < len(frame) - 100, "the random rows hold both kinds"
    assert _same_annotation(vectorised.set_index("transcript_id"), by_row.set_index("transcript_id"))

    # the same through parse_refflat, with the vectorised path turned off for the second parse (an empty field
    # reads as NaN, which only the row-by-row path takes, so the file has no empty exon list)
    path = tmp_path / "random_refFlat.txt"
    _write_refflat(path, [r for r in rows if r[9]])
    used = []
    keep_valid_exons = pileup_genomic_module._keep_valid_exons
    monkeypatch.setattr(pileup_genomic_module, "_keep_valid_exons", lambda *a: used.append(1) or keep_valid_exons(*a))
    fast = pileup_genomic_module.parse_refflat(path)
    assert used, "parse_refflat took the vectorised path"
    monkeypatch.setattr(pileup_genomic_module, "_parse_exon_lists", lambda column: None)
    slow = pileup_genomic_module.parse_refflat(path)
    assert len(used) == 1 and _same_annotation(fast, slow)
    first_two = {r[1]: [int(x) for x in r[9].split(",")[:2] if x] for r in rows if r[8] > 1}
    descending = {name for name, x in first_two.items() if len(x) == 2 and x[0] > x[1]}
    assert descending and not descending & set(fast.index), "exons in descending order are dropped"
