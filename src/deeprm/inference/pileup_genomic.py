"""Utilities for mapping transcript-relative coordinates to genomic coordinates
and aggregating per-site statistics across a transcriptome.

This module provides:

* `TranscriptMapper`: A vectorized mapper that converts 0-based, half-open
  transcript coordinates into genomic coordinates for both '+' and '-' strands,
  returning `np.nan` for out-of-bounds inputs.
* `parse_refflat`: A helper to parse RefFlat/RefGene/GenePred annotations into a
  normalized DataFrame.
* `pileup_genomic`: A high-level function that adds up per-site sums per
  (chrom, strand, pos). It maps all sites of all transcripts in one vectorized pass
  (the same mapping as `TranscriptMapper`). The scores are computed from these sums
  by `deeprm.inference.pileup_deeprm`, the same way as for the transcript table.

All exon intervals are assumed to be 0-based, half-open `[start, end)`, sorted by
genomic start in ascending order.
"""

import logging
import re

import numpy as np
import pandas as pd

from deeprm.utils.logging import get_logger

log = get_logger(__name__)


class TranscriptMapper:
    """Vectorized mapper from transcript coordinates to genomic coordinates.

    This class maps 0-based, half-open transcript offsets into genomic 0-based
    positions using exon interval metadata. It supports both '+' and '-' strands
    and returns `np.nan` for any input coordinate that falls outside the valid
    transcript range `[0, total_len)` or is non-finite.

    Attributes:
        starts (numpy.ndarray): 1D array of exon start genomic coordinates (0-based, inclusive),
            sorted ascending.
        ends (numpy.ndarray): 1D array of exon end genomic coordinates (0-based, exclusive),
            same shape as `starts`.
        lengths (numpy.ndarray): Exon lengths, computed as `ends - starts`.
        total_len (int): Total transcript length, i.e., `lengths.sum()`.
        strand (str): Strand symbol, either `'+'` or `'-'`.
        cumsum (numpy.ndarray): Precomputed cumulative exon lengths. For `'+'`,
            `cumsum[i]` is the total length before exon `i`; for `'-'`, it is
            computed over reversed exons to enable reverse mapping.

    Raises:
        ValueError: If `strand` is not `'+'` or `'-'`.
    """

    def __init__(self, exon_starts: np.ndarray, exon_ends: np.ndarray, strand: str):
        """Initialize the mapper with exon intervals and strand.

        Args:
            exon_starts (numpy.ndarray): 1D array of exon starts (0-based, inclusive),
                sorted ascending by genomic coordinate.
            exon_ends (numpy.ndarray): 1D array of exon ends (0-based, exclusive),
                same shape as `exon_starts`. Each `end` must be strictly greater
                than its corresponding `start`.
            strand (str): `'+'` or `'-'`.

        Raises:
            ValueError: If `strand` is not `'+'` or `'-'`.
        """
        self.starts = exon_starts
        self.ends = exon_ends
        self.lengths = self.ends - self.starts
        self.total_len = int(self.lengths.sum())
        self.strand = strand

        # Precompute cumulative sums
        if self.strand == "+":
            self.cumsum = np.concatenate(([0], np.cumsum(self.lengths)))
        elif self.strand == "-":
            self.cumsum = np.concatenate(([0], np.cumsum(self.lengths[::-1])))
        else:
            raise ValueError("strand must be '+' or '-'.")

    def map(self, coords: np.ndarray) -> np.ndarray:
        """Map transcript offsets to genomic positions (vectorized).

        Input coordinates are treated as 0-based offsets into the concatenated
        transcript exonic sequence with a valid range `[0, total_len)`. Values
        outside this range or non-finite values (NaN/Inf) yield `np.nan` in the
        output. The result is always `float64` to accommodate NaNs.

        For `'+'` strand: genomic position = `starts[idx] + offset`.
        For `'-'` strand: genomic position = `ends[idx] - 1 - offset`.

        Args:
            coords (numpy.ndarray): Array-like of transcript offsets to map. May be any
                shape; the returned array will match this shape.

        Returns:
            numpy.ndarray: Array of genomic positions (dtype `float64`) with `np.nan`
            for out-of-bounds or non-finite inputs. Positions are 0-based.

        Notes:
            * Assumes exon intervals are half-open `[start, end)` and sorted by
              genomic start ascending. It should be validated in parse_refflat().
            * No exceptions are raised for invalid `coords`; they are marked as NaN.
        """
        out = np.full(coords.shape, np.nan, dtype=np.float64)

        # Valid coords in [0, total_len)
        # also guard against inf/NaN in input by requiring finite
        valid = np.isfinite(coords) & (coords >= 0) & (coords < self.total_len)
        if not np.any(valid):
            return out

        coords = coords[valid].astype(np.int64, copy=False)

        if self.strand == "+":
            idx = np.searchsorted(self.cumsum, coords, side="right") - 1
            offsets = coords - self.cumsum[idx]
            out[valid] = (self.starts[idx] + offsets).astype(np.float64, copy=False)
        else:
            r_idx = np.searchsorted(self.cumsum, coords, side="right") - 1
            offsets = coords - self.cumsum[r_idx]
            idx = self.starts.size - r_idx - 1
            out[valid] = (self.ends[idx] - offsets - 1).astype(np.float64, copy=False)

        return out


def _parse_exon_array(value):
    if isinstance(value, np.ndarray):
        return value.astype(np.int64, copy=False)
    text = str(value).strip()
    if text.endswith(","):
        text = text[:-1]
    if text == "":
        return np.array([], dtype=np.int64)
    return np.fromstring(text, sep=",", dtype=np.int64)


# A plain exon list: non-negative integers separated by commas, with at most one trailing comma (UCSC style)
_EXON_LIST = re.compile(r"\d+(,\d+)*,?")


def _parse_exon_lists(column):
    """Parses a whole exon-list column at once.

    Returns:
        tuple: (all values of all rows as one int64 array, number of values per row), or None when a row is not a
        plain exon list; parse_refflat then parses and checks the rows one by one.
    """
    text = column.astype(str).str.strip()
    if not text.map(lambda t: t == "" or _EXON_LIST.fullmatch(t) is not None).all():
        return None
    text = text.str.removesuffix(",")
    counts = np.where(text.str.len().to_numpy() == 0, 0, text.str.count(",").to_numpy() + 1).astype(np.int64)
    joined = ",".join(t for t in text.tolist() if t)
    values = np.fromstring(joined, sep=",", dtype=np.int64) if joined else np.empty(0, dtype=np.int64)
    if len(values) != counts.sum():
        return None
    return values, counts


def _keep_valid_exons(refflat_df, starts, ends):
    """The rows of `refflat_df` whose exon lists pass the checks of parse_refflat, with the lists as int64 arrays.

    The checks: exonCount starts and exonCount ends, every exon end > its start, and starts and ends that do not
    decrease. They run on the flat arrays of _parse_exon_lists; the result is the one of the row-by-row checks.
    """
    (start_values, n_starts), (end_values, n_ends) = starts, ends
    count = refflat_df["exonCount"].to_numpy()
    right_count = np.flatnonzero((n_starts == count) & (n_ends == count))  # other rows are dropped
    lens = count[right_count]  # > 0: exonCount > 0 was checked before
    within = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)  # exon index within its row
    exon_start = start_values[np.repeat((np.cumsum(n_starts) - n_starts)[right_count], lens) + within]
    exon_end = end_values[np.repeat((np.cumsum(n_ends) - n_ends)[right_count], lens) + within]
    later = within > 0
    bad = exon_end <= exon_start
    bad[later] |= (exon_start[1:] < exon_start[:-1])[later[1:]] | (exon_end[1:] < exon_end[:-1])[later[1:]]
    row_bad = np.zeros(len(right_count), dtype=bool)
    row_bad[np.repeat(np.arange(len(right_count)), lens)[bad]] = True

    split = np.cumsum(lens)[:-1]
    good = ~row_bad
    starts_col = np.empty(int(good.sum()), dtype=object)
    ends_col = np.empty(int(good.sum()), dtype=object)
    for i, (a, b) in enumerate(
        (a, b) for a, b, g in zip(np.split(exon_start, split), np.split(exon_end, split), good) if g
    ):
        starts_col[i] = a.copy()
        ends_col[i] = b.copy()
    kept = refflat_df.iloc[right_count[good]].copy()
    kept["exonStarts"] = starts_col
    kept["exonEnds"] = ends_col
    return kept


def _keep_valid_exons_by_row(refflat_df):
    """_keep_valid_exons for exon lists that are not plain integer lists: parsed and checked one row at a time."""
    refflat_df = refflat_df.copy()
    refflat_df["exonStarts"] = refflat_df["exonStarts"].apply(_parse_exon_array)
    refflat_df["exonEnds"] = refflat_df["exonEnds"].apply(_parse_exon_array)
    refflat_df = refflat_df[refflat_df["exonStarts"].apply(len) == refflat_df["exonCount"]]
    refflat_df = refflat_df[refflat_df["exonEnds"].apply(len) == refflat_df["exonCount"]]
    refflat_df = refflat_df[refflat_df.apply(lambda x: np.all(x["exonEnds"] > x["exonStarts"]), axis=1)]
    refflat_df = refflat_df[refflat_df.apply(lambda x: np.all(x["exonStarts"][1:] >= x["exonStarts"][:-1]), axis=1)]
    refflat_df = refflat_df[refflat_df.apply(lambda x: np.all(x["exonEnds"][1:] >= x["exonEnds"][:-1]), axis=1)]
    return refflat_df


def parse_refflat(refflat_path):
    """
    Parse RefFlat/RefGene/GenePred annotation into a normalized DataFrame.

    This function reads a tab-delimited annotation file and normalizes it into a
    common schema with the following columns:
    `transcript_id`, `chrom`, `strand`, `txStart`, `txEnd`,
    `cdsStart`, `cdsEnd`, `exonCount`, `exonStarts`, `exonEnds`.

    It accepts three formats based on column count:

    * 11 columns (RefFlat): drops the first column.
    * 15 columns (RefGene): keeps the first 10 columns.
    * 10 columns (GenePred): uses as-is.

    Exon start/end lists are expected as comma-separated strings with a trailing comma
    (UCSC style) and are converted to `numpy.ndarray` of `int`. Invalid transcripts are
    filtered out based on interval consistency and ordering. The resulting DataFrame is
    indexed by `transcript_id`.

    Args:
        refflat_path (str): Path to the annotation file.

    Returns:
        pandas.DataFrame: Normalized and validated annotation indexed by `transcript_id`, with the columns
        `chrom` (str), `strand` (str; '+' or '-'), `txStart`, `txEnd`, `cdsStart`, `cdsEnd` (int), `exonCount` (int),
        and `exonStarts`, `exonEnds` (numpy.ndarray of int; 0-based, half-open).

    Raises:
        ValueError: If the file has an unexpected number of columns, a strand other than
            '+' or '-' (the message names the first such transcript), or if no valid
            transcripts remain after validation.

    Notes:
        * Validation ensures: `txEnd > txStart`, `cdsEnd >= cdsStart`, `exonCount > 0`,
          `len(exonStarts) == len(exonEnds) == exonCount`, strictly positive exon lengths,
          and non-decreasing starts/ends across exons.
    """
    col_list = [
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
    refflat_df = pd.read_csv(refflat_path, sep="\t", header=None)

    n_cols = refflat_df.shape[1]
    if n_cols == 11:
        refflat_df = refflat_df.iloc[:, 1:]
    elif n_cols == 15:
        refflat_df = refflat_df.iloc[:, :10]
    elif n_cols != 10:
        raise ValueError(
            "Invalid annotation file format. Expected 10 (GenePred), 11 (RefFlat), or 15 (RefGene) columns."
        )

    refflat_df.columns = col_list
    bad_strand = ~refflat_df["strand"].isin(["+", "-"])
    if bad_strand.any():
        first = refflat_df[bad_strand].iloc[0]
        raise ValueError(
            f"Invalid strand {first['strand']!r} for transcript {first['transcript_id']} in {refflat_path} "
            f"({int(bad_strand.sum())} rows have a strand other than '+' or '-')."
        )
    refflat_df[["txStart", "txEnd", "cdsStart", "cdsEnd", "exonCount"]] = refflat_df[
        ["txStart", "txEnd", "cdsStart", "cdsEnd", "exonCount"]
    ].astype(np.int64)

    refflat_df = refflat_df[refflat_df["txEnd"] > refflat_df["txStart"]]
    refflat_df = refflat_df[refflat_df["cdsEnd"] >= refflat_df["cdsStart"]]
    refflat_df = refflat_df[refflat_df["exonCount"] > 0]
    starts = _parse_exon_lists(refflat_df["exonStarts"])
    ends = _parse_exon_lists(refflat_df["exonEnds"])
    if starts is not None and ends is not None:
        refflat_df = _keep_valid_exons(refflat_df, starts, ends)
    else:
        refflat_df = _keep_valid_exons_by_row(refflat_df)
    refflat_df.drop_duplicates(subset="transcript_id", keep="first", inplace=True, ignore_index=True)

    if len(refflat_df) == 0:
        raise ValueError("No valid transcripts found in the annotation file.")

    refflat_df.set_index("transcript_id", inplace=True)
    return refflat_df


# A transcript name with its version: ENST00000123456.7 -> ENST00000123456, NM_005517.4 -> NM_005517
_VERSIONED_NAME = re.compile(r"^(.+)\.\d+$")


def _annotation_rows(names, annotation_index):
    """Row of each transcript name in the annotation: the exact name, or else the name without its version (.N), as
    annotations often have it (Ensembl names, and RefSeq names in UCSC tables).

    Returns:
        tuple: (row per name, -1 when not found; number of names found exactly; number found without the version).
    """
    rows = annotation_index.get_indexer(names)
    retry = np.flatnonzero(rows < 0)
    n_exact = len(rows) - len(retry)
    n_unversioned = 0
    if retry.size:
        bare = [_VERSIONED_NAME.sub(r"\1", str(name)) for name in names[retry]]
        changed = np.array([b != str(name) for b, name in zip(bare, names[retry])], dtype=bool)
        if changed.any():
            found = annotation_index.get_indexer(pd.Index(np.array(bare, dtype=object)[changed]))
            rows[retry[changed]] = found
            n_unversioned = int((found >= 0).sum())
    return rows, n_exact, n_unversioned


def _map_to_genome(names, positions, refflat_df):
    """Genomic positions of transcript sites; `TranscriptMapper` for all transcripts at once.

    The exons of every annotated transcript are laid end to end in transcript order (reversed for '-'), so one
    searchsorted finds the exon of every site.

    Args:
        names (numpy.ndarray): Transcript name per site.
        positions (numpy.ndarray): 0-based transcript position per site.
        refflat_df (pandas.DataFrame): The output of parse_refflat.

    Returns:
        tuple: (indices of the sites that map, their chrom, strand and 0-based genomic position).
    """
    codes, transcripts = pd.factorize(names, sort=False)
    rows, n_exact, n_unversioned = _annotation_rows(np.asarray(transcripts, dtype=object), refflat_df.index)
    n_missing = int((rows < 0).sum())
    example = f", for example {transcripts[np.flatnonzero(rows < 0)[0]]}" if n_missing else ""
    # a warning when sites are left out, so that a mismatched annotation does not go unnoticed
    log.log(
        logging.WARNING if n_missing else logging.INFO,
        f"Genomic pileup: {n_exact} of {len(transcripts)} transcripts matched the annotation by name, "
        f"{n_unversioned} after removing the version, {n_missing} not found{example} (their sites are left out).",
    )
    if n_unversioned:
        log.warning(
            f"{n_unversioned} transcripts matched the annotation only without their version; where the two "
            "versions have different exons, their genomic positions follow the annotation's version."
        )
    present = np.flatnonzero(rows >= 0)
    if present.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, np.empty(0, dtype=object), np.empty(0, dtype=object), empty

    annotated = refflat_df.iloc[rows[present]]
    plus = annotated["strand"].to_numpy() == "+"
    starts = annotated["exonStarts"].tolist()
    ends = annotated["exonEnds"].tolist()
    n_exons = np.array([len(x) for x in starts], dtype=np.int64)
    first = np.cumsum(n_exons) - n_exons
    within = np.arange(n_exons.sum()) - np.repeat(first, n_exons)
    exon_plus = np.repeat(plus, n_exons)
    order = np.where(exon_plus, np.repeat(first, n_exons) + within, np.repeat(first + n_exons - 1, n_exons) - within)
    exon_start = np.concatenate(starts).astype(np.int64)[order]
    exon_end = np.concatenate(ends).astype(np.int64)[order]
    exon_len = exon_end - exon_start
    tx_len = np.add.reduceat(exon_len, first)
    tx_offset = np.cumsum(tx_len) - tx_len  # start of each transcript in the laid-out coordinates
    exon_offset = np.cumsum(exon_len) - exon_len  # start of each exon in the same coordinates

    tx_of_site = np.full(len(transcripts), -1, dtype=np.int64)
    tx_of_site[present] = np.arange(len(present))
    tx_of_site = tx_of_site[codes]
    positions = np.asarray(positions, dtype=np.int64)
    inside = tx_of_site >= 0
    inside[inside] = (positions[inside] >= 0) & (positions[inside] < tx_len[tx_of_site[inside]])
    sites = np.flatnonzero(inside)
    tx = tx_of_site[sites]
    laid_out = tx_offset[tx] + positions[sites]
    exon = np.searchsorted(exon_offset, laid_out, side="right") - 1
    offset = laid_out - exon_offset[exon]
    genomic = np.where(plus[tx], exon_start[exon] + offset, exon_end[exon] - offset - 1)
    chrom = annotated["chrom"].to_numpy()[tx]
    strand = np.where(plus[tx], "+", "-").astype(object)
    return sites, chrom, strand, genomic


def pileup_genomic(input_df, annot, sum_columns):
    """Adds up per-site sums in genomic coordinates.

    Every transcript site is mapped to its genomic position (as `TranscriptMapper` maps it) and the sums are added up
    per (chrom, strand, pos). A site on the transcript's strand ('+' in `ref_strand`) gets the transcript's genomic
    strand; a site on the opposite strand ('-', from reads aligned in reverse) is the same genomic base on the other
    genomic strand, so it gets a row of its own. Transcripts are found in the annotation by their exact name, or else
    by the name without its version (.N); transcripts not in the annotation, and sites outside a transcript's exons,
    are left out. The scores (modscore, stoichiometry) are not computed here: `deeprm.inference.pileup_deeprm`
    computes them from the summed table, the same way as for the transcript table.

    Args:
        input_df (pandas.DataFrame): One row per transcript site, with these columns:

            - `ref_names` (str): Transcript ID.
            - `ref_pos` (int): Transcript-relative position (0-based).
            - The columns named in `sum_columns`.
            - Optionally `ref_strand` (str): '+' for a site on the transcript's strand, '-' for the opposite strand.
              Without it, every site is on the transcript's strand.

        annot (str): Path to a RefFlat/RefGene/GenePred file.
        sum_columns (list): The per-site sum columns to add up. Other columns of `input_df` are not carried over.

    Returns:
        pandas.DataFrame: `chrom`, `strand`, `pos` (0-based genomic position, int64) and the `sum_columns`, summed
        over the transcript sites that map to each genomic position, sorted by (chrom, strand, pos).

    Raises:
        ValueError: If `input_df` is empty, or no valid genomic position is produced (e.g., due to mismatched or
            invalid annotations).
    """
    if len(input_df) == 0:
        raise ValueError("input_df is empty. No transcript pileup entries were provided.")
    sum_columns = list(sum_columns)

    refflat_df = parse_refflat(annot)
    sites, chrom, strand, pos = _map_to_genome(
        input_df["ref_names"].to_numpy(), input_df["ref_pos"].to_numpy(), refflat_df
    )
    if len(sites) == 0:
        raise ValueError("No valid genomic position found. Please verify the annotation file.")
    if "ref_strand" in input_df:
        # a '-' site is an A on the strand opposite the transcript: the same base, on the other genomic strand
        opposite = input_df["ref_strand"].to_numpy()[sites] == "-"
        strand = np.where(opposite, np.where(strand == "+", "-", "+"), strand).astype(object)

    df = pd.DataFrame({"chrom": chrom, "strand": strand, "pos": pos})
    for col in sum_columns:
        df[col] = input_df[col].to_numpy()[sites]
    df = df.groupby(["chrom", "strand", "pos"], as_index=False).agg({col: "sum" for col in sum_columns})
    df["pos"] = df["pos"].astype(np.int64)
    return df
