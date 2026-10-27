"""
Context-block extraction for `deeprm train prep` (DAG over spacer and anchor hits).

Two extractors find the context blocks of each read: DAGLevenshteinBlockExtractor (``--method lev``, designs with
6-nt spacers) and DAGPairwiseBlockExtractor (``--method pw``, designs with 15-nt spacers; with ``di_anchor=True`` it
also handles two anchors per context block). Each read is matched against the spacer sequences, the spacer hits are
joined into the longest consistent path through a DAG, and the context block between two spacers is checked for its
anchor. Each block is reported with its read, its position, its order in the read and a score; the score is lower when
the anchor position is ambiguous.
"""
import numpy as np
from Bio import Align
import networkx as nx
import itertools as it
from collections import defaultdict

# polyleven is only needed by the Levenshtein (6-nt spacers) extractor. Keep it optional
# so the pairwise path -- which is what di-m6A uses -- runs in environments that
# do not have it.
try:
    import polyleven as pl
except ImportError:  # pragma: no cover
    pl = None

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union


# --------------------------------------------------------------------------- #
# di-m6A design constants (see the table in the module docstring)
# --------------------------------------------------------------------------- #

#: random nt between the two anchors, one entry per designed type
DEFAULT_DI_GAP_LIST: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

#: random nt between each spacer and its nearest anchor, same order.
#: equals ``20 - ceil(gap / 2)``, which keeps the anchors centred and BB at 79/80.
DEFAULT_DI_FLANK_LIST: Tuple[int, ...] = (20, 19, 19, 18, 18, 17, 17, 16, 16, 15)

#: the two 19-nt spacers of the di-m6A library (S1, S2); BB = S1 + CB + S2.
#: Given by the experimenter 2026-08-08 as RNA:
#:   S1 5'-CGACGGUGUUAGUAUCCAA-3'   S2 5'-CCAGAUCCGAAUUGUAGUC-3'
#: stored here as DNA (U -> T) because reads are basecalled in DNA alphabet.
#: These SUPERSEDE the pair recovered from notebook outputs; the lengths (19)
#: and therefore the whole block geometry are unchanged, but the alignment
#: behaviour is not, so every metric below was re-measured against these.
DEFAULT_DI_SPACER_LIST: Tuple[str, ...] = ("CGACGGTGTTAGTATCCAA", "CCAGATCCGAATTGTAGTC")


#: gap value denoting the MONO-m6A layout (a single anchor per CB)
MONO_GAP: int = -1


@dataclass(frozen=True)
class AnchorLayout:
    """One designed CB type.

    Covers both library styles:
      * mono-m6A: one anchor, ``gap = MONO_GAP (-1)``, cb_size = 2*flank + A
      * di-m6A:   two anchors, ``gap >= 0``,           cb_size = 2*flank + gap + 2A

    Attributes:
        gap: random nt between the two anchors, or MONO_GAP for a single anchor.
        flank: random nt between each spacer and its nearest anchor.
        cb_size: total CB length implied by this layout.
        offsets: anchor offsets inside the CB, ascending (1 or 2 entries).
    """
    gap: int
    flank: int
    cb_size: int
    offsets: Tuple[int, ...]

    @property
    def n_anchors(self) -> int:
        return len(self.offsets)

    @property
    def is_mono(self) -> bool:
        return len(self.offsets) == 1


#: kept as an alias so existing di-only call sites keep reading naturally
DiAnchorLayout = AnchorLayout


def build_di_layouts(gap_list: Sequence[int],
                     flank_list: Sequence[int],
                     anchor_size: int,
                     include_mono: bool = False,
                     mono_flank: Optional[int] = None) -> List[AnchorLayout]:
    """Turn the (gap, flank) design table into concrete CB layouts.

    Args:
        gap_list: random nt between the two anchors, per designed di type.
        flank_list: random nt between spacer and nearest anchor, same order.
        anchor_size: length of the anchor string.
        include_mono: also admit a single-anchor (mono-m6A) layout, so one run
            can reconstruct a pool containing both block styles.
        mono_flank: flank of the mono layout (normally ``cb_pad``). Required
            when include_mono is True.

    Returns:
        list[AnchorLayout]: mono layout first (when requested), then di layouts.
    """
    if len(gap_list) != len(flank_list):
        raise ValueError(
            f"di_gap_list and di_flank_list must have the same length "
            f"({len(gap_list)} vs {len(flank_list)})"
        )
    if len(gap_list) == 0 and not include_mono:
        raise ValueError("di_gap_list must be non-empty")

    layouts = []

    if include_mono:
        if mono_flank is None:
            raise ValueError("mono_flank is required when include_mono is True")
        mono_flank = int(mono_flank)
        if mono_flank < 0:
            raise ValueError(f"mono_flank must be >= 0, got {mono_flank}")
        layouts.append(
            AnchorLayout(
                gap=MONO_GAP,
                flank=mono_flank,
                cb_size=2 * mono_flank + anchor_size,
                offsets=(mono_flank,),
            )
        )

    seen_gaps = {MONO_GAP} if include_mono else set()
    for gap, flank in zip(gap_list, flank_list):
        gap = int(gap)
        flank = int(flank)
        if gap < 0:
            raise ValueError(
                f"di gap must be >= 0 ({MONO_GAP} is reserved for the mono "
                f"layout, added via include_mono), got {gap}"
            )
        if flank < 0:
            raise ValueError(f"di flank must be >= 0, got {flank}")
        if gap in seen_gaps:
            raise ValueError(f"duplicate di gap {gap}: gaps identify the type")
        seen_gaps.add(gap)
        layouts.append(
            AnchorLayout(
                gap=gap,
                flank=flank,
                cb_size=2 * flank + gap + 2 * anchor_size,
                offsets=(flank, flank + anchor_size + gap),
            )
        )
    return layouts


@dataclass(slots=True)
class DAGBaseConfig:
    """Common configuration for DAG-based block extractors.

    Notes:
        * anchor_list / spacer_list are stored in DNA alphabet (U->T) and uppercase.
        * anchor_shift defaults to (anchor_size - 1) for each anchor if not provided.
        * anchor_uncertainty_penalty (NEW in v3): per-extra-anchor-match penalty
          in raw-score units.  Applied AFTER DAG path selection (so it never
          affects which CBs are extracted, only their reported score).
          Default 2 (~one spacer-base-match worth).  Set to 0 to recover v1
          score semantics.
    """
    cb_pad: int
    cb_per_bb: int
    spacer_size: int
    spacer_list: Sequence[str]
    anchor_list: Sequence[str]
    anchor_shift: Optional[Sequence[int]]
    cb_size_tolerance: int
    indel_tolerance: int
    indel_penalty: int
    anchor_mismatch_penalty: int
    spacer_mismatch_penalty: int
    anchor_uncertainty_penalty: int = 2
    skip_anchor: bool = False

    # --- edge (terminal, one-sided) blocks -------------------------------- #
    # A CB is normally emitted only from an S_0 -> S_1 edge, i.e. only when BOTH
    # of its flanking spacers are on the read. Real concatemer reads start and
    # end mid-block, so the first and last CB usually have one spacer clipped
    # off and are dropped. Measured on one training library (13,896 reads): 72% of reads carry
    # >=36 nt of orphaned block sequence at the 5' end and 36% at the 3' end.
    #
    # With edge_blocks on, such a terminal CB is placed by GEOMETRY from its one
    # surviving spacer. There is no second spacer to close the measurement, so
    # the indel error inside the orphan is unknown -- this trades precision for
    # recall and is off by default. `edge_anchor_window` is how far the anchor
    # base may sit from its designed offset before the block is rejected.
    edge_blocks: bool = False
    edge_anchor_window: int = 2

    # --- di-m6A (two anchors per CB) ------------------------------------- #
    di_anchor: bool = False
    di_gap_list: Optional[Sequence[int]] = None
    di_flank_list: Optional[Sequence[int]] = None
    di_allow_partial: bool = False

    # Also admit the single-anchor (mono-m6A) layout, so one run can
    # reconstruct a pool containing both mono and di blocks. The mono layout
    # uses cb_pad as its flank and is reported as gap = MONO_GAP (-1).
    di_include_mono: bool = False

    # The designed type of THIS library, when it is known.
    #
    # Confirmed by the experimenter (2026-08-07): one molecule carries a single
    # type, AND each type is sequenced in its own run. So for a real run the
    # type is known up front, and the layout table collapses to one entry.
    # That removes the same-CB-length ambiguity entirely rather than voting it
    # away, and it is the recommended way to run a real di-m6A library.
    di_type: Optional[int] = None

    # How to resolve the type when it is NOT given.
    #   "batch" - pool evidence across every read in the batch (correct when a
    #             run holds a single type, which is the real layout)
    #   "read"  - one type per molecule, decided independently per read
    #   "none"  - resolve each context block on its own
    di_consensus: str = "batch"

    # When the type is fixed (di_type, or a batch/read consensus), drop context
    # blocks that cannot be explained by it instead of keeping each one's own
    # best-fitting layout. A block that is not the run's type is either spurious
    # or one whose anchors were destroyed; keeping it emits two WRONG positions.
    # These become training labels, so a wrong label costs more than a lost one.
    di_drop_offtype: bool = True

    # Deprecated alias for di_consensus; False maps to "none".
    di_consistent_type: bool = True

    # Periodicity between the same landmark in consecutive building blocks.
    # False keeps the original formula (cb_size * cb_per_bb + spacer_size).
    # True uses the geometrically correct cb_size * cb_per_bb + spacer_size *
    # len(spacer_list), since every BB carries one spacer per spacer_list entry
    # (BB = S1 + CB + S2 = 79/80 here, not 41/42 + 19 = 60/61).
    # Kept configurable so the original behaviour stays reproducible.
    true_bb_period: bool = False

    def __post_init__(self):
        self.spacer_list = [str(x).upper().replace("U", "T") for x in self.spacer_list]
        self.anchor_list = [str(x).upper().replace("U", "T") for x in self.anchor_list]
        if len(self.anchor_list) == 0:
            raise ValueError("anchor_list must be non-empty")
        anchor_size = len(self.anchor_list[0])
        if any(len(a) != anchor_size for a in self.anchor_list):
            raise ValueError("All anchors must have the same length")
        if self.anchor_shift is None:
            self.anchor_shift = [anchor_size - 1 for _ in self.anchor_list]
        else:
            self.anchor_shift = list(self.anchor_shift)
        if len(self.anchor_shift) != len(self.anchor_list):
            raise ValueError("anchor_shift length must match anchor_list length")
        if self.indel_tolerance < self.cb_size_tolerance:
            raise ValueError("indel_tolerance must be >= cb_size_tolerance")
        if self.anchor_uncertainty_penalty < 0:
            raise ValueError("anchor_uncertainty_penalty must be >= 0")

        if self.di_anchor:
            if self.skip_anchor:
                raise ValueError(
                    "skip_anchor and di_anchor are mutually exclusive: di mode "
                    "localises a CB *by* its anchors, so it cannot skip anchor "
                    "validation."
                )
            if self.di_gap_list is None:
                self.di_gap_list = list(DEFAULT_DI_GAP_LIST)
            if self.di_flank_list is None:
                self.di_flank_list = list(DEFAULT_DI_FLANK_LIST)
            self.di_gap_list = [int(x) for x in self.di_gap_list]
            self.di_flank_list = [int(x) for x in self.di_flank_list]
            if len(self.di_gap_list) != len(self.di_flank_list):
                raise ValueError(
                    "di_gap_list and di_flank_list must have the same length; got "
                    f"{len(self.di_gap_list)} gaps and {len(self.di_flank_list)} flanks"
                )

            # A known type is applied when the block is INTERPRETED, not by
            # deleting the other layouts. Measured: collapsing the table to one
            # layout also starves DAG path selection (the per-block penalty
            # that weights the edges is then computed from one candidate), and
            # costs ~0.4 points of recall against keeping all ten and forcing
            # the type at emission. Same type accuracy either way.
            if self.di_type is not None:
                self.di_type = int(self.di_type)
                # MONO_GAP names the single-anchor layout, which is a legitimate
                # run type once di_include_mono has put it in the table -- a
                # mono-m6A run declared with --di_type mono used to be rejected.
                allowed = list(self.di_gap_list)
                if self.di_include_mono:
                    allowed.append(MONO_GAP)
                if self.di_type not in allowed:
                    raise ValueError(
                        f"di_type {self.di_type} is not one of the designed gaps "
                        f"{self.di_gap_list}"
                        + (f" (or {MONO_GAP} for the mono layout)"
                           if self.di_include_mono else
                           "; pass di_include_mono=True to allow the mono layout")
                    )

            if not self.di_consistent_type:      # deprecated alias
                self.di_consensus = "none"
            if self.di_consensus not in ("batch", "read", "none"):
                raise ValueError(
                    f"di_consensus must be 'batch', 'read' or 'none', got "
                    f"{self.di_consensus!r}"
                )


@dataclass(slots=True)
class DAGPairwiseConfig(DAGBaseConfig):
    spacer_match_score: int = 3
    spacer_gap_open_penalty: int = 3
    spacer_gap_extend_penalty: int = 1
    spacer_edit_tolerance: int = 6
    max_iter: int = 10
    max_alignments_per_iter: int = 2000


@dataclass(slots=True)
class DAGLevenshteinConfig(DAGBaseConfig):
    skip_size_tolerance: int = 4
    spacer_mismatch_tolerance: int = 3
    spacer_size_tolerance: int = 1


############# Common functions for DAG-based block extraction #############

class DAGBlockExtractor:
    def __init__(self, config: Optional[DAGBaseConfig] = None, **kwargs):
        # Backward-compatible: allow legacy **kwargs construction.
        if config is None:
            config = DAGBaseConfig(**kwargs)

        self.config = config

        self.cb_pad = config.cb_pad
        self.cb_per_bb = config.cb_per_bb
        self.spacer_size = config.spacer_size
        self.spacer_list = list(config.spacer_list)
        self.anchor_list = list(config.anchor_list)
        self.anchor_shift = list(config.anchor_shift)
        self.cb_size_tolerance = config.cb_size_tolerance
        self.indel_tolerance = config.indel_tolerance
        self.indel_penalty = config.indel_penalty
        self.anchor_mismatch_penalty = config.anchor_mismatch_penalty
        self.spacer_mismatch_penalty = config.spacer_mismatch_penalty
        self.anchor_uncertainty_penalty = config.anchor_uncertainty_penalty   # v3: output-score only
        self.skip_anchor = config.skip_anchor

        self.di_anchor = config.di_anchor
        self.di_allow_partial = config.di_allow_partial
        self.di_type = config.di_type
        self.di_consensus = config.di_consensus
        self.di_drop_offtype = config.di_drop_offtype
        # kept for callers that still read the old name
        self.di_consistent_type = (config.di_consensus != "none")
        self.true_bb_period = config.true_bb_period

        self.edge_blocks = config.edge_blocks
        self.edge_anchor_window = config.edge_anchor_window
        self._last_node_scores = {}

        self.anchor_size = len(self.anchor_list[0])

        if self.di_anchor:
            self.layouts = build_di_layouts(
                config.di_gap_list,
                config.di_flank_list,
                self.anchor_size,
                include_mono=config.di_include_mono,
                mono_flank=self.cb_pad,
            )
            self.layout_by_gap = {lay.gap: lay for lay in self.layouts}
            self.cb_size_list = sorted({lay.cb_size for lay in self.layouts})
            # representative CB size; exact per-type geometry is handled in
            # layout_costs, this is only for coarse/legacy arithmetic.
            self.cb_size = int(round(float(np.mean(self.cb_size_list))))
        else:
            self.layouts = []
            self.layout_by_gap = {}
            self.cb_size = 2 * self.cb_pad + self.anchor_size
            self.cb_size_list = [self.cb_size]

        # Geometry used ONLY by the edge-block path, so that turning edge blocks
        # off leaves every mono code path byte-identical to v3. A mono CB is
        # flank + anchor + flank with flank = cb_pad.
        self.mono_layout = AnchorLayout(
            gap=MONO_GAP, flank=self.cb_pad,
            cb_size=2 * self.cb_pad + self.anchor_size,
            offsets=(self.cb_pad,),
        )

        self.cb_size_min = self.cb_size_list[0]
        self.cb_size_max = self.cb_size_list[-1]

        # Periodicity of one building block, per admissible CB size.
        # Original formula understates it by spacer_size * cb_per_bb because a
        # BB carries one spacer per spacer_list entry; kept as the default so
        # v3 behaviour is reproducible.
        n_spacers = len(self.spacer_list)
        self.period_by_cb = {
            cb: (cb * self.cb_per_bb + self.spacer_size * n_spacers)
            if self.true_bb_period
            else (cb * self.cb_per_bb + self.spacer_size)
            for cb in self.cb_size_list
        }
        self.bb_size = self.period_by_cb[self.cb_size_list[0]] \
            if len(self.cb_size_list) == 1 \
            else float(np.mean(list(self.period_by_cb.values())))

        self.indel_dict = self.get_integer_partition()

    def get_integer_partition(self):
        indel_dict = {}
        for spacing_error in range(-self.cb_size_tolerance, self.cb_size_tolerance + 1):
            indel_list = []
            for front_error in range(-self.indel_tolerance, self.indel_tolerance + 1):
                back_error = spacing_error - front_error
                if np.abs(front_error) + np.abs(back_error) <= self.indel_tolerance:
                    indel_list.append((front_error, back_error))
            indel_list.sort(key=lambda x: np.abs(x[0]) + np.abs(x[1]))
            indel_dict[spacing_error] = indel_list
        return indel_dict


    def dag_longest_path(self, edge_list):
        node_list = list(set([x[0] for x in edge_list] + [x[1] for x in edge_list]))

        dag = nx.DiGraph()
        dag.add_nodes_from(node_list)
        dag.add_weighted_edges_from(edge_list)
        longest_path = nx.dag_longest_path(dag, weight='weight')

        return longest_path


    def extract_blocks_read(self, seq):
        raise NotImplementedError("extract_blocks_read method should be implemented in subclasses")


    #: v4 result keys. `anchor_idx` is the anchor's index inside its CB (0 for
    #: mono blocks, 0/1 for di blocks) and `di_gap` is the identified layout
    #: (MONO_GAP = -1 for a single-anchor block).
    RESULT_KEYS = ["cb_idx", "global_idx", "pos", "score", "anchor_idx", "di_gap"]

    def extract_blocks(self, bam_df):
        """Extract blocks for a batch of reads.

        With ``di_consensus='batch'`` the designed type is decided ONCE from
        every read in the batch, which is the right scope when a run holds a
        single type. It is skipped when the layout table already has one entry
        (``di_type`` given), because then there is nothing to decide.
        """
        keys = list(self.RESULT_KEYS)
        result_dict = {key: [] for key in keys}
        empty = {key: [] for key in keys}

        seqs = []
        for seq, trim_5p, trim_3p in zip(bam_df["seq"], bam_df["tl"], bam_df["tr"]):
            if trim_5p == 0 and trim_3p == 0:
                seqs.append(None)
            else:
                seqs.append(seq[trim_5p:trim_3p].tobytes().decode("ascii"))

        use_batch = (self.di_anchor
                     and getattr(self, "di_type", None) is None
                     and getattr(self, "di_consensus", "none") == "batch"
                     and len(self.layouts) > 1
                     and hasattr(self, "analyse_read_ex"))

        if use_batch:
            analyses = [self.analyse_read_ex(s) if s is not None else ([], None)
                        for s in seqs]
            costs_all = [c for a, _ in analyses for c in self.layout_costs_of(a)]
            gap = self.consensus_gap(costs_all) if costs_all else None
            per_read = [self.emit_read(a, force_gap=gap, edge_ctx=ctx)
                        for a, ctx in analyses]
        else:
            per_read = [self.extract_blocks_read(s) if s is not None else dict(empty)
                        for s in seqs]

        for r in per_read:
            for key in keys:
                result_dict[key].append(r[key])

        return result_dict


    # ---- v3: anchor uniqueness helpers ----------------------------------

    def _count_anchor_matches_in_window(self, query, single_anchor, displacement_error):
        """Count exact anchor matches inside the indel-tolerance window.

        Uses the same indel partition (`indel_dict[displacement_error]`)
        that `validate_anchor` walks, so the count exactly mirrors the set
        of positions the anchor finder considers candidates.

        Args:
            query (str): the read substring inside the candidate CB
                (PW: read[from_pos:to_pos]; LEV: read[from_pos+spacer_size:to_pos]).
            single_anchor (str): expected anchor sequence.
            displacement_error (int): displacement_error passed to validate_anchor.

        Returns:
            int: number of positions in the window where query matches anchor.
        """
        n_matches = 0
        L = len(single_anchor)
        possible_indel_list = self.indel_dict.get(displacement_error, [])
        for front_indel, _back_indel in possible_indel_list:
            anchor_start = self.cb_pad + front_indel
            anchor_end = anchor_start + L
            if anchor_start < 0 or anchor_end > len(query):
                continue
            if query[anchor_start:anchor_end] == single_anchor:
                n_matches += 1
        return n_matches


    # ---- v4: layout engine (mono and di anchors) ------------------------

    def find_anchor_occurrences(self, query, anchor):
        """Every start offset at which `anchor` occurs in `query`."""
        occ = []
        start = query.find(anchor)
        while start != -1:
            occ.append(start)
            start = query.find(anchor, start + 1)
        return occ


    def layout_costs(self, read, from_pos, to_pos, anchor_idx):
        """Cheapest anchor placement for EACH admissible CB layout.

        Rather than enumerating indel partitions (O(tolerance^2) per layout),
        the search is inverted: locate every anchor occurrence in the window
        and read the implied indels off the geometry.  For a di layout with
        anchors observed at offsets i < j::

            e_left  = i - flank
            e_mid   = (j - i - anchor_size) - gap
            e_right = (obs_len - cb_size) - e_left - e_mid

        and for a mono layout with the anchor at i::

            e_left  = i - flank
            e_right = (obs_len - cb_size) - e_left

        A placement is admissible when the total |indel| stays within
        indel_tolerance, and is charged total_indel * indel_penalty -- the same
        scale v3 uses, so scores remain comparable.

        Keeping every layout (not just the global best) is what lets the caller
        impose a read-level consensus type and measure ambiguity.

        Returns:
            dict: {gap: (penalty, pos_tuple, balance)} for each layout with an
            admissible placement.  `pos_tuple` is in READ coordinates,
            ascending, already shifted by anchor_shift.  `balance` is
            |e_left - e_right|, a deterministic geometry-centring tiebreak.
        """
        anchor = self.anchor_list[anchor_idx]
        a_len = len(anchor)
        shift = self.anchor_shift[anchor_idx]
        tol = self.indel_tolerance

        query = read[from_pos:to_pos]
        obs_len = len(query)
        occ = self.find_anchor_occurrences(query, anchor)

        costs = {}

        def offer(gap, penalty, balance, offsets):
            prev = costs.get(gap)
            if prev is None or (penalty, balance) < (prev[0], prev[2]):
                costs[gap] = (penalty,
                              tuple(from_pos + o + shift for o in offsets),
                              balance)

        mono_layouts = [lay for lay in self.layouts if lay.is_mono]
        di_layouts = [lay for lay in self.layouts if not lay.is_mono]

        # --- mono layouts: a single anchor -------------------------------- #
        for layout in mono_layouts:
            for i in occ:
                e_left = i - layout.flank
                if abs(e_left) > tol:
                    continue
                e_right = (obs_len - layout.cb_size) - e_left
                total_indel = abs(e_left) + abs(e_right)
                if total_indel > tol:
                    continue
                offer(layout.gap, total_indel * self.indel_penalty,
                      abs(e_left - e_right), (i,))

        # --- di layouts: both anchors observed ---------------------------- #
        for oi in range(len(occ) - 1):
            i = occ[oi]
            for oj in range(oi + 1, len(occ)):
                j = occ[oj]
                observed_gap = j - i - a_len
                if observed_gap < 0:
                    continue
                for layout in di_layouts:
                    e_left = i - layout.flank
                    if abs(e_left) > tol:
                        continue
                    e_mid = observed_gap - layout.gap
                    if abs(e_left) + abs(e_mid) > tol:
                        continue
                    e_right = (obs_len - layout.cb_size) - e_left - e_mid
                    total_indel = abs(e_left) + abs(e_mid) + abs(e_right)
                    if total_indel > tol:
                        continue
                    offer(layout.gap, total_indel * self.indel_penalty,
                          abs(e_left - e_right), (i, j))

        # --- di layouts: only one anchor survived basecalling -------------- #
        # Opt-in. Always costs an extra anchor_mismatch_penalty, so a full
        # match on the same layout is preferred whenever one exists.
        if self.di_allow_partial:
            for i in occ:
                for layout in di_layouts:
                    # (a) the observed anchor is the FIRST one
                    e_left = i - layout.flank
                    if abs(e_left) <= tol:
                        e_rest = (obs_len - layout.cb_size) - e_left
                        total_indel = abs(e_left) + abs(e_rest)
                        inferred = i + a_len + layout.gap
                        if total_indel <= tol and 0 <= inferred <= obs_len - a_len:
                            offer(layout.gap,
                                  total_indel * self.indel_penalty + self.anchor_mismatch_penalty,
                                  abs(e_left - e_rest), (i, inferred))

                    # (b) the observed anchor is the SECOND one
                    e_right = (obs_len - i - a_len) - layout.flank
                    if abs(e_right) <= tol:
                        e_rest = (obs_len - layout.cb_size) - e_right
                        total_indel = abs(e_right) + abs(e_rest)
                        inferred = i - a_len - layout.gap
                        if total_indel <= tol and 0 <= inferred <= obs_len - a_len:
                            offer(layout.gap,
                                  total_indel * self.indel_penalty + self.anchor_mismatch_penalty,
                                  abs(e_right - e_rest), (inferred, i))

        return costs


    def resolve_layout(self, costs, displacement_error, force_gap=None):
        """Pick ONE layout out of `layout_costs`.

        Args:
            costs: output of layout_costs.
            displacement_error: used only for the "no anchor found" penalty,
                matching the single-anchor validator's fallback.
            force_gap: if given and admissible, resolve to this layout. Used to
                impose the read-level consensus type.

        Returns:
            tuple: (penalty, missing_anchor, pos_list, n_matches, gap), the same
            shape the PW validator returns. `n_matches` counts how many layouts
            explain this window equally well, and feeds v3's anchor-uncertainty
            score discount.
        """
        if not costs:
            penalty = abs(displacement_error) * self.indel_penalty + self.anchor_mismatch_penalty
            return penalty, 1, None, 1, MONO_GAP

        if force_gap is not None and force_gap in costs:
            gap = force_gap
        else:
            gap = min(costs, key=lambda g: (costs[g][0], costs[g][2], g))

        penalty, pos_tuple, _ = costs[gap]
        best_penalty = min(v[0] for v in costs.values())
        n_matches = sum(1 for v in costs.values() if v[0] == best_penalty)
        return penalty, 0, list(pos_tuple), max(1, n_matches), gap


    def consensus_gap(self, costs_list):
        """Pick the one designed type that best explains a whole molecule.

        One molecule carries one designed type, so the CBs of a read should all
        resolve to the same gap.

        The vote is by PLURALITY of per-CB winners, not by summed penalty: with
        a loose indel_tolerance almost every layout is admissible somewhere, so
        an admissibility count saturates, and a summed penalty lets one badly
        basecalled CB outvote several clean ones.  Counting how many CBs pick
        each layout as their own best is robust to both.  Ties in the per-CB
        minimum count for every layout that achieves it.

        Order: most per-CB wins, then most CBs where it is admissible at all,
        then lowest total penalty, then smallest gap (deterministic).

        Returns:
            int | None: consensus gap, or None if no CB had any admissible layout.
        """
        candidates = set()
        for costs in costs_list:
            candidates.update(costs.keys())
        if not candidates:
            return None

        wins = {gap: 0 for gap in candidates}
        for costs in costs_list:
            if not costs:
                continue
            best_penalty = min(v[0] for v in costs.values())
            for gap, v in costs.items():
                if v[0] == best_penalty:
                    wins[gap] += 1

        def rank(gap):
            n_ok = 0
            total = 0.0
            for costs in costs_list:
                if gap in costs:
                    n_ok += 1
                    total += costs[gap][0]
            return (-wins[gap], -n_ok, total, gap)

        return min(candidates, key=rank)


#############---Pairwise block extraction---#############


class DAGPairwiseBlockExtractor(DAGBlockExtractor):
    """
    Pairwise-alignment based block extractor.

    This implementation is generalized for any number of spacers by using the
    same spacer-state DAG model as DAGLevenshteinBlockExtractor, while retaining
    the accuracy benefit of pairwise alignment.

    Key design:
        1. Align every spacer as an individual monomer.
        2. Additionally align only the true physical bridge dimer:
               spacer_list[-1] + spacer_list[0]
           Do NOT align internal pseudo-dimers such as S1+S2, because internal
           spacers are separated by context blocks.
        3. Build a DAG over spacer-state nodes:
               node = (spacer_idx, spacer_start, spacer_end)
        4. Emit context blocks only for direct CB edges:
               S_i -> S_{i+1}
           Bridge edges:
               S_last -> S_0
           are used only for chaining and do not emit blocks.

    v3 change: validate_anchor returns `n_matches` (count of exact anchor
    hits inside the indel-tolerance window) alongside the v1 penalty/anchor.
    find_block_candidates subtracts `anchor_uncertainty_penalty * (n_matches - 1)`
    in raw-score space from the score written to `cb_info_dict` only --
    never from the dag_list edge weight that drives path selection.  So
    path selection is byte-identical to v1; only the reported score is
    discounted for ambiguous anchors.
    """

    def __init__(self, config: Optional[DAGPairwiseConfig] = None, **kwargs):
        # Backward-compatible: allow legacy **kwargs construction.
        if config is None:
            config = DAGPairwiseConfig(**kwargs)
        super().__init__(config=config)

        self.spacer_edit_tolerance = config.spacer_edit_tolerance
        self.spacer_match_score = config.spacer_match_score
        self.spacer_gap_open_penalty = config.spacer_gap_open_penalty
        self.spacer_gap_extend_penalty = config.spacer_gap_extend_penalty
        self.max_iter = config.max_iter
        self.max_alignments_per_iter = config.max_alignments_per_iter
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.max_alignments_per_iter <= 0:
            raise ValueError("max_alignments_per_iter must be positive")
        self.aligner = self.get_aligner()

        if len(self.spacer_list) != self.cb_per_bb + 1:
            raise ValueError(
                "Generalized PW mode expects len(spacer_list) == cb_per_bb + 1. "
                f"Got len(spacer_list)={len(self.spacer_list)}, cb_per_bb={self.cb_per_bb}."
            )
        if len(self.anchor_list) < self.cb_per_bb:
            raise ValueError(
                "Generalized PW mode expects at least cb_per_bb anchors. "
                f"Got len(anchor_list)={len(self.anchor_list)}, cb_per_bb={self.cb_per_bb}."
            )
        if any(len(s) != self.spacer_size for s in self.spacer_list):
            raise ValueError(
                "All spacers must have length == spacer_size. "
                f"spacer_size={self.spacer_size}, spacer lengths={[len(s) for s in self.spacer_list]}."
            )

        self.min_score_monomer = (
            self.spacer_match_score * self.spacer_size
            - max(
                self.spacer_edit_tolerance * self.spacer_mismatch_penalty,
                self.spacer_edit_tolerance * self.spacer_gap_extend_penalty
                + self.spacer_gap_open_penalty,
            )
        )
        self.min_score_dimer = (
            self.spacer_match_score * self.spacer_size * 2
            - max(
                self.spacer_edit_tolerance * 2 * self.spacer_mismatch_penalty,
                self.spacer_edit_tolerance * 2 * self.spacer_gap_extend_penalty
                + self.spacer_gap_open_penalty,
            )
        )

        self.spacer_dimer_size = self.spacer_size * 2
        self.bridge_dimer = self.spacer_list[-1] + self.spacer_list[0]

        # Maximum raw score for an emitted context-block edge in the generalized
        # PW model: one left spacer plus one right spacer, both perfectly aligned,
        # with zero anchor/indel penalty. Emitted block scores are normalized to
        # [0, 1] using this value. Bridge edges remain raw internally because they
        # are graph-chaining evidence and are not emitted as block scores.
        self.max_cb_raw_score = max(
            1.0,
            float(2 * self.spacer_size * self.spacer_match_score),
        )

        # One displacement table per admissible CB length (one entry unless the
        # di-m6A design is active, which admits 41 and 42).
        self.min_ideal_displacement_dicts = {
            cb_size: self.get_min_ideal_displacement_dict(cb_size)
            for cb_size in self.cb_size_list
        }
        self.min_ideal_displacement_dict = self.min_ideal_displacement_dicts[self.cb_size_list[0]]


    def _bounded_alignments(self, alignments):
        """Take at most `max_alignments_per_iter` alignments, highest score first.

        v3 declared and validated `max_alignments_per_iter` but never applied it:
        it called `sorted()` straight on the PairwiseAlignments iterator, which
        materialises EVERY co-optimal alignment. Biopython returns all optimal
        alignments, and their number grows combinatorially when the gap costs
        make many placements score the same -- reachable with ordinary-looking
        values of --sgo/--sge/--sms. Observed effect: worker processes at 16-17 GB
        RSS and no forward progress.

        Slicing the lazy iterator first bounds the work. When the true count is
        at or below the cap this returns exactly the same list as before (the
        slice keeps iteration order and the sort is stable), so normal runs are
        unaffected.
        """
        capped = list(it.islice(alignments, self.max_alignments_per_iter))
        capped.sort(key=lambda x: x.score, reverse=True)
        return capped


    def normalize_cb_score(self, raw_score):
        """Normalize a raw PW context-block score to the [0, 1] range."""
        return float(np.clip(float(raw_score) / self.max_cb_raw_score, 0.0, 1.0))


    def get_aligner(self):
        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        aligner.match_score = self.spacer_match_score
        aligner.mismatch_score = -self.spacer_mismatch_penalty
        aligner.open_gap_score = -self.spacer_gap_open_penalty
        aligner.extend_gap_score = -self.spacer_gap_extend_penalty
        return aligner


    def _alignment_bounds_on_read(self, alignment, target_len):
        """
        Convert a local pairwise alignment into full-target read bounds.

        This preserves the previous PW convention: the returned bounds estimate
        where the full target would sit on the read, including unaligned target
        bases at either end of a local alignment.
        """
        aligned = alignment.aligned
        if len(aligned[0]) == 0 or len(aligned[1]) == 0:
            return None

        start = aligned[1][0][0] - aligned[0][0][0]
        end = aligned[1][-1][1] + target_len - aligned[0][-1][1]
        return int(start), int(end)


    def _project_target_boundary_to_read(self, alignment, target_boundary, fallback_read_pos):
        """
        Project a target boundary coordinate to an approximate read coordinate.

        This is used to split the bridge dimer alignment into two spacer nodes.
        If the boundary falls inside an aligned block, map it directly. If it
        falls in a gap between aligned blocks, use the midpoint between adjacent
        read blocks. If no projection is possible, fall back to the simple
        geometric split.
        """
        target_blocks, read_blocks = alignment.aligned

        for (t0, t1), (r0, r1) in zip(target_blocks, read_blocks):
            if t0 <= target_boundary <= t1:
                return int(r0 + (target_boundary - t0))

        for i in range(len(target_blocks) - 1):
            left_t1 = target_blocks[i][1]
            right_t0 = target_blocks[i + 1][0]
            if left_t1 <= target_boundary <= right_t0:
                left_r1 = read_blocks[i][1]
                right_r0 = read_blocks[i + 1][0]
                return int(round((left_r1 + right_r0) / 2))

        return int(fallback_read_pos)


    def get_min_ideal_displacement_dict(self, cb_size=None):
        """
        Expected spacer-start displacement between all spacer-state pairs.

        This mirrors DAGLevenshteinBlockExtractor.get_min_ideal_displacement_dict(),
        but keeps the PW cb_size inherited from DAGBlockExtractor:
            cb_size = 2 * cb_pad + anchor_size

        v4: `cb_size` is a parameter because the di-m6A design admits more than
        one CB length (41 and 42). One table is built per admissible length and
        get_ideal_displacement keeps the best fit. With di off there is exactly
        one length and this is the original computation.
        """
        if cb_size is None:
            cb_size = self.cb_size
        min_ideal_displacement_dict = {}
        big_step_size = cb_size + self.spacer_size
        small_step_size = self.spacer_size
        last_spacer_idx = len(self.spacer_list) - 1

        for from_idx in range(len(self.spacer_list)):
            for to_idx in range(len(self.spacer_list)):
                if from_idx < to_idx:
                    small_steps = 0
                    big_steps = to_idx - from_idx
                    displacement = big_step_size * big_steps
                else:
                    small_steps = 1
                    big_steps = last_spacer_idx - from_idx + to_idx
                    displacement = big_step_size * big_steps + small_step_size

                min_ideal_displacement_dict[(from_idx, to_idx)] = (
                    displacement,
                    small_steps,
                    big_steps,
                )

        return min_ideal_displacement_dict


    def get_ideal_displacement(self, from_spacer_idx, to_spacer_idx, displacement):
        """Best-fitting ideal displacement over all admissible CB lengths.

        With one CB length (mono-m6A) the loop runs once and this is exactly the
        v3 computation. With the di-m6A design a one-step displacement may be
        41 or 42 nt, so each candidate length is evaluated and the one closest
        to the observed displacement wins.
        """
        best = None
        for cb_size in self.cb_size_list:
            table = self.min_ideal_displacement_dicts[cb_size]
            period = self.period_by_cb[cb_size]
            min_ideal_displacement, min_small_steps, min_big_steps = (
                table[(from_spacer_idx, to_spacer_idx)]
            )

            if displacement <= min_ideal_displacement:
                ideal_displacement = min_ideal_displacement
                small_steps = min_small_steps
                big_steps = min_big_steps
            else:
                periodicity = round((displacement - min_ideal_displacement) / period)
                ideal_displacement = min_ideal_displacement + periodicity * period
                small_steps = min_small_steps + periodicity
                big_steps = min_big_steps + self.cb_per_bb * periodicity

            candidate = (abs(displacement - ideal_displacement),
                         ideal_displacement, small_steps, big_steps)
            if best is None or candidate < best:
                best = candidate

        return best[1], best[2], best[3]


    def get_monomer_align_set(self, seq):
        """
        Align every individual spacer to the read.

        Returns:
            dict mapping node_id -> score, where
                node_id = (spacer_idx, start, end)
        """
        align_dict = {}
        seq_len = len(seq)

        for spacer_idx, spacer in enumerate(self.spacer_list):
            seq_masked = seq

            for _ in range(self.max_iter):
                alignments = self._bounded_alignments(
                    self.aligner.align(spacer, seq_masked))

                if len(alignments) == 0:
                    break
                if alignments[0].score < self.min_score_monomer:
                    break

                updated = set()

                for alignment in alignments:
                    score = alignment.score
                    if score < self.min_score_monomer:
                        break

                    bounds = self._alignment_bounds_on_read(alignment, self.spacer_size)
                    if bounds is None:
                        continue
                    start, end = bounds

                    if start < 0 or end > seq_len or start >= end:
                        continue

                    node_id = (int(spacer_idx), int(start), int(end))
                    if node_id not in align_dict or score > align_dict[node_id]:
                        align_dict[node_id] = float(score)

                    updated.add((int(start), int(end)))

                if len(updated) == 0:
                    break

                # Mask only for this spacer so repeated occurrences of the same
                # spacer can be found without suppressing other spacer identities.
                for start, end in updated:
                    seq_masked = seq_masked[:start] + "N" * (end - start) + seq_masked[end:]

        return align_dict


    def get_bridge_dimer_align_set(self, seq):
        """
        Align only the true physical bridge dimer: S_last + S_0.

        Returns:
            bridge_nodes: dict[node_id] -> score
            bridge_edges: dict[(from_node_id, to_node_id)] -> dimer_score

        The dimer hit is split into two spacer nodes:
            S_last at the left half
            S_0    at the right half
        and the edge S_last -> S_0 is marked as high-confidence bridge evidence.
        """
        bridge_nodes = {}
        bridge_edges = {}
        seq_len = len(seq)
        last_idx = len(self.spacer_list) - 1

        seq_masked = seq
        for _ in range(self.max_iter):
            alignments = self._bounded_alignments(
                self.aligner.align(self.bridge_dimer, seq_masked))

            if len(alignments) == 0:
                break
            if alignments[0].score < self.min_score_dimer:
                break

            updated = set()

            for alignment in alignments:
                score = alignment.score
                if score < self.min_score_dimer:
                    break

                bounds = self._alignment_bounds_on_read(alignment, self.spacer_dimer_size)
                if bounds is None:
                    continue
                start, end = bounds

                if start < 0 or end > seq_len or start >= end:
                    continue

                fallback_split = start + self.spacer_size
                split = self._project_target_boundary_to_read(
                    alignment,
                    self.spacer_size,
                    fallback_split,
                )

                # Keep split sane. If indels make the projected boundary too odd,
                # fall back to a geometric split.
                if split <= start or split >= end:
                    split = fallback_split
                if split <= start or split >= end:
                    continue

                left_node = (int(last_idx), int(start), int(split))
                right_node = (0, int(split), int(end))
                component_score = float(score) / 2.0

                if left_node not in bridge_nodes or component_score > bridge_nodes[left_node]:
                    bridge_nodes[left_node] = component_score
                if right_node not in bridge_nodes or component_score > bridge_nodes[right_node]:
                    bridge_nodes[right_node] = component_score

                edge_id = (left_node, right_node)
                if edge_id not in bridge_edges or score > bridge_edges[edge_id]:
                    bridge_edges[edge_id] = float(score)

                updated.add((int(start), int(end)))

            if len(updated) == 0:
                break

            for start, end in updated:
                seq_masked = seq_masked[:start] + "N" * (end - start) + seq_masked[end:]

        return bridge_nodes, bridge_edges


    def get_align_set(self, seq):
        """
        Combine monomer spacer hits with bridge-dimer-derived spacer hits.

        Returns:
            node_scores: dict[(spacer_idx, start, end)] -> score
            bridge_edges: dict[((last_idx, start, end), (0, start, end))] -> score
        """
        node_scores = self.get_monomer_align_set(seq)
        bridge_nodes, bridge_edges = self.get_bridge_dimer_align_set(seq)

        for node_id, score in bridge_nodes.items():
            if node_id not in node_scores or score > node_scores[node_id]:
                node_scores[node_id] = score

        return node_scores, bridge_edges


    def validate_anchor(self, read, from_pos, to_pos, anchor_idx, displacement_error,
                        force_gap=None, costs=None):
        """PW anchor validator (v4).

        Returns `(penalty, missing_anchor, pos_list, n_matches, gap)`.

        Mono path (di_anchor off): `penalty`, `missing_anchor` and `n_matches`
        are byte-identical to v3.  `pos_list` is v3's `anchor_pos` wrapped in a
        one-element list, and `gap` is MONO_GAP.  `n_matches` is the number of
        exact anchor hits inside the indel-tolerance window; it is NOT folded
        into `penalty` (that would change DAG path selection and degrade
        precision).  See module docstring for the rationale.

        di path: resolves the CB against the layout table, optionally forced to
        the read's consensus type.  `pos_list` then holds one read coordinate
        per anchor (two for a di layout).
        """
        if self.di_anchor:
            if costs is None:
                costs = self.layout_costs(read, from_pos, to_pos, anchor_idx)
            return self.resolve_layout(costs, displacement_error, force_gap=force_gap)

        displacement_error_abs = np.abs(displacement_error)
        if not self.skip_anchor:
            single_anchor = self.anchor_list[anchor_idx]
            possible_indel_list = self.indel_dict[displacement_error]
            query = read[from_pos:to_pos]
            anchor_candidate_list = [
                (displacement_error_abs * self.indel_penalty + self.anchor_mismatch_penalty, 1, None)
            ]

            for front_indel, back_indel in possible_indel_list:
                anchor_start = self.cb_pad + front_indel
                anchor_end = anchor_start + len(single_anchor)
                if anchor_start < 0 or anchor_end > len(query):
                    continue
                anchor_query = query[anchor_start:anchor_end]
                if anchor_query == single_anchor:
                    anchor_pos = from_pos + self.cb_pad + front_indel + self.anchor_shift[anchor_idx]
                    total_indel = np.abs(front_indel) + np.abs(back_indel)
                    anchor_candidate_list.append((total_indel * self.indel_penalty, 0, anchor_pos))

            anchor_candidate_list.sort(key=lambda x: (x[0], x[1]))
            base_penalty, missing_anchor, anchor_pos = anchor_candidate_list[0]

            n_matches = self._count_anchor_matches_in_window(
                query, single_anchor, displacement_error
            )
            pos_list = None if anchor_pos is None else [anchor_pos]
            return base_penalty, missing_anchor, pos_list, n_matches, MONO_GAP
        else:
            center = from_pos + (to_pos - from_pos) // 2
            return displacement_error_abs * self.indel_penalty, 0, [center], 1, MONO_GAP


    def find_block_candidates(self, seq):
        cb_info_dict = {}
        dag_list = []

        node_scores, bridge_edges = self.get_align_set(seq)
        # Per-call scratch, read immediately by analyse_read_ex for the edge
        # path. Not state: it is consumed before the next read is analysed.
        self._last_node_scores = node_scores
        node_list = sorted(node_scores.keys(), key=lambda x: (x[1], x[0], x[2]))

        if len(node_list) < 2:
            return cb_info_dict, dag_list

        for from_node in node_list:
            from_spacer_idx, start_1, end_1 = from_node
            score_1 = node_scores[from_node]

            for to_node in node_list:
                to_spacer_idx, start_2, end_2 = to_node
                score_2 = node_scores[to_node]

                # Strictly increasing read coordinate keeps the graph acyclic.
                displacement = start_2 - start_1
                if displacement <= 0:
                    continue

                ideal_displacement, small_steps, big_steps = self.get_ideal_displacement(
                    from_spacer_idx,
                    to_spacer_idx,
                    displacement,
                )
                displacement_error = displacement - ideal_displacement
                displacement_error_abs = np.abs(displacement_error)

                # Direct bridge is spacer-scale; all other transitions are
                # context-block-scale. For skipped transitions, allow tolerance
                # to grow with the number of CB-scale steps.
                if big_steps == 0 and small_steps == 1:
                    displacement_tolerance = self.spacer_edit_tolerance
                else:
                    displacement_tolerance = self.cb_size_tolerance * max(1, big_steps)

                if displacement_error_abs > displacement_tolerance:
                    continue

                is_cb = False
                pos_list = None
                gap_cb = MONO_GAP
                n_matches_cb = 1   # v3: only meaningful for CB edges; 1 = no uncertainty discount
                layout_costs_cb = None

                # Direct context-block edge S_i -> S_{i+1}; this emits a block.
                if (
                    big_steps == 1
                    and small_steps == 0
                    and displacement_error_abs <= self.cb_size_tolerance
                    and from_spacer_idx < self.cb_per_bb
                ):
                    if self.di_anchor:
                        # keep every admissible layout so extract_blocks_read
                        # can impose the read-level consensus type
                        layout_costs_cb = self.layout_costs(
                            seq, end_1, start_2, from_spacer_idx
                        )
                    penalty, missing_anchor, pos_list, n_matches_cb, gap_cb = self.validate_anchor(
                        seq,
                        end_1,
                        start_2,
                        from_spacer_idx,
                        displacement_error,
                        costs=layout_costs_cb,
                    )
                    if missing_anchor == 0:
                        is_cb = True

                # True physical bridge edge S_last -> S_0; this does not emit a block.
                elif (
                    big_steps == 0
                    and small_steps == 1
                    and from_spacer_idx == self.cb_per_bb
                    and to_spacer_idx == 0
                    and displacement_error_abs <= self.spacer_edit_tolerance
                ):
                    penalty = self.indel_penalty * displacement_error_abs

                # Skip edge through missing/noisy spacers. Keeps DAG connected.
                else:
                    if self.skip_anchor:
                        penalty = self.indel_penalty * displacement_error_abs
                    else:
                        penalty = self.indel_penalty * displacement_error_abs + self.anchor_mismatch_penalty

                # v3: total_score uses v1-style penalty (NO anchor-uncertainty
                # term).  This score drives DAG longest-path selection, so it
                # must match v1 exactly for precision parity.
                total_score = max(0.0, float(score_1 + score_2 - penalty))

                # If this exact bridge came from bridge-dimer alignment, preserve
                # the dimer's stronger evidence score.
                evidence = float(score_1 + score_2)
                bridge_score = bridge_edges.get((from_node, to_node))
                if bridge_score is not None:
                    total_score = max(total_score, float(bridge_score - penalty))
                    evidence = max(evidence, float(bridge_score))

                if is_cb:
                    # v3: subtract anchor-uniqueness penalty in raw-score
                    # space BEFORE normalize_cb_score.  This is applied AFTER
                    # total_score has been added to dag_list, so the path
                    # selection sees only the v1 score (precision unchanged).
                    uncertainty_penalty = (
                        self.anchor_uncertainty_penalty * max(0, n_matches_cb - 1)
                    )
                    output_raw = max(0.0, total_score - uncertainty_penalty)
                    cb_info_dict[(from_node, to_node)] = (
                        from_spacer_idx,  # cb_idx
                        pos_list,
                        self.normalize_cb_score(output_raw),
                        score_1,
                        score_2,
                        # v4 extras: enough to re-resolve the CB under the
                        # read-level consensus type without re-aligning.
                        gap_cb,
                        layout_costs_cb,
                        evidence,
                        int(displacement_error),
                    )

                # dag_list weights MUST be v1-equivalent (no uncertainty).
                dag_list.append((from_node, to_node, total_score))

        return cb_info_dict, dag_list


    def analyse_read_ex(self, seq):
        """Select the context-block chain, and keep what the edge path needs.

        Returns ``(selected, edge_ctx)``. ``edge_ctx`` carries the chain's first
        and last spacer nodes plus their alignment scores: a clipped terminal CB
        has exactly one surviving spacer, and that node fixes its geometry. It
        is None when edge blocks are off or the read yielded no chain.
        """
        cb_info_dict, dag_list = self.find_block_candidates(seq)
        if len(cb_info_dict) == 0 or len(dag_list) == 0:
            return [], None

        longest_path = self.dag_longest_path(dag_list)
        selected = []
        for x, y in zip(longest_path[:-1], longest_path[1:]):
            info = cb_info_dict.get((x, y))
            if info is None:
                continue
            if info[1] is None or info[2] is None:   # pos_list, score
                continue
            selected.append(info)

        edge_ctx = None
        if self.edge_blocks and longest_path:
            ns = self._last_node_scores or {}
            head, tail = longest_path[0], longest_path[-1]
            edge_ctx = {
                "seq": seq,
                "head": head, "tail": tail,
                "head_score": float(ns.get(head, 0.0)),
                "tail_score": float(ns.get(tail, 0.0)),
            }
        return selected, edge_ctx

    def analyse_read(self, seq):
        """Select the context-block chain for one read, before type resolution.

        Split out from extract_blocks_read so a caller can pool the layout
        evidence of many reads and decide the designed type once per batch.
        """
        return self.analyse_read_ex(seq)[0]

    def _edge_blocks(self, edge_ctx, gap):
        """Place the two clipped terminal CBs by geometry from their one spacer.

        A normal CB is measured between two spacers, so an indel inside it shows
        up as a displacement error and can be priced. A terminal CB has no
        closing spacer, so no such measurement exists: the anchors are placed at
        their designed offsets from the surviving spacer and accepted only if
        the anchor base is actually there, within `edge_anchor_window`. That
        window is the entire error budget, which is why this trades precision
        for recall.

        Returns ``(head_blocks, tail_blocks)`` as `selected`-shaped tuples.
        """
        if not edge_ctx:
            return [], []
        layout = (self.layout_by_gap.get(gap) if self.di_anchor
                  else self.mono_layout)
        if layout is None:
            return [], []

        seq = edge_ctx["seq"]
        n = len(seq)
        anchors = set(self.anchor_list)
        w = self.edge_anchor_window
        a_size = self.anchor_size

        def place(cb_start, node_score):
            pos_list = []
            for off in layout.offsets:
                ideal = cb_start + off
                found = None
                for d in range(0, w + 1):
                    cands = (ideal,) if d == 0 else (ideal - d, ideal + d)
                    for c in cands:
                        if 0 <= c and c + a_size <= n and seq[c:c + a_size] in anchors:
                            found = c
                            break
                    if found is not None:
                        break
                if found is None:
                    return None
                pos_list.append(found)
            # the two anchors must stay distinct and ordered
            if any(b <= a for a, b in zip(pos_list, pos_list[1:])):
                return None
            # one spacer's evidence only, so an edge block scores about half of
            # a two-spacer block and sorts below it on any score threshold
            score = self.normalize_cb_score(max(0.0, node_score))
            return (0, pos_list, score, node_score, 0.0,
                    layout.gap, None, node_score, 0)

        head_blocks, tail_blocks = [], []

        # 5' orphan: the chain begins on S_last, so the CB to its LEFT lost S_0
        h_idx, h_start, _ = edge_ctx["head"]
        if h_idx == self.cb_per_bb:
            cb_start = h_start - layout.cb_size
            if cb_start + layout.offsets[0] >= 0:
                b = place(cb_start, edge_ctx["head_score"])
                if b is not None:
                    head_blocks.append(b)

        # 3' orphan: the chain ends on S_0, so the CB to its RIGHT lost S_last
        t_idx, _, t_end = edge_ctx["tail"]
        if t_idx == 0:
            if t_end + layout.offsets[-1] + a_size <= n:
                b = place(t_end, edge_ctx["tail_score"])
                if b is not None:
                    tail_blocks.append(b)

        return head_blocks, tail_blocks

    @staticmethod
    def layout_costs_of(selected):
        """The per-context-block layout cost tables of an analysed read."""
        return [info[6] for info in selected if info[6]]

    def _reresolve(self, selected, gap):
        """Re-interpret every context block under one designed type.

        Path selection has already happened, so this cannot change which blocks
        the DAG chose -- only how each one is read, and (with di_drop_offtype)
        whether a block that contradicts the run's type is emitted at all.
        """
        out = []
        for info in selected:
            (cb_idx, pos_list, score, score_1, score_2,
             cur_gap, costs, evidence, disp_err) = info
            if costs and gap not in costs and self.di_drop_offtype:
                # cannot be the run's type: emitting it would place two anchors
                # that are known to be wrong
                continue
            if costs and cur_gap != gap and gap in costs:
                penalty, missing, new_pos, n_matches, new_gap = self.resolve_layout(
                    costs, disp_err, force_gap=gap
                )
                if missing == 0:
                    pos_list, cur_gap = new_pos, new_gap
                    raw = max(0.0, evidence - penalty)
                    raw = max(0.0, raw - self.anchor_uncertainty_penalty
                              * max(0, n_matches - 1))
                    score = self.normalize_cb_score(raw)
            out.append((cb_idx, pos_list, score, score_1, score_2,
                        cur_gap, costs, evidence, disp_err))
        return out

    def emit_read(self, selected, force_gap=None, edge_ctx=None):
        keys = list(self.RESULT_KEYS)
        result_dict = {key: [] for key in keys}

        if not selected:
            return result_dict

        # Resolve the designed type. With di_type given there is only one
        # layout and this is a no-op; otherwise the scope is the batch (decided
        # by the caller and passed in as force_gap) or this single read.
        if self.di_anchor:
            if self.di_type is not None:
                force_gap = self.di_type          # a known type always wins
            elif force_gap is None and self.di_consensus in ("read", "batch"):
                # No batch decision was handed in, so this read is the only
                # evidence available. Falling back to read scope keeps a direct
                # extract_blocks_read() call sane; the batch path always passes
                # force_gap and so never reaches here.
                costs_list = self.layout_costs_of(selected)
                force_gap = self.consensus_gap(costs_list) if costs_list else None
            if force_gap is not None:
                selected = self._reresolve(selected, force_gap)

        # Terminal CBs whose outer spacer was clipped off the read. Added after
        # type resolution so they are placed with the run's geometry, and in
        # read order so global_idx and the strictly-increasing check still hold.
        if self.edge_blocks and edge_ctx is not None:
            gap = force_gap if self.di_anchor else MONO_GAP
            if gap is not None:
                head_b, tail_b = self._edge_blocks(edge_ctx, gap)
                if head_b or tail_b:
                    selected = list(head_b) + list(selected) + list(tail_b)

        # A di CB emits TWO blocks, one per anchor. They share cb_idx,
        # global_idx and score, and are told apart by anchor_idx.
        global_idx = 0
        for info in selected:
            cb_idx, pos_list, score, gap = info[0], info[1], info[2], info[5]
            for anchor_idx, pos in enumerate(pos_list):
                result_dict["global_idx"].append(global_idx)
                result_dict["cb_idx"].append(cb_idx)
                result_dict["anchor_idx"].append(anchor_idx)
                result_dict["pos"].append(pos)
                result_dict["score"].append(score)
                result_dict["di_gap"].append(gap)
            global_idx += 1

        if len(result_dict["pos"]) == 0:
            return {key: [] for key in keys}

        # Context-block positions must be strictly increasing.
        if not np.all(np.diff(result_dict["pos"]) > 0):
            return {key: [] for key in keys}

        return result_dict

    def extract_blocks_read(self, seq, force_gap=None):
        selected, edge_ctx = self.analyse_read_ex(seq)
        return self.emit_read(selected, force_gap=force_gap, edge_ctx=edge_ctx)


#############---Levenshtein distance-based block extraction---#############

class DAGLevenshteinBlockExtractor(DAGBlockExtractor):
    def __init__(self, config: Optional[DAGLevenshteinConfig] = None, **kwargs):
        # Backward-compatible: allow legacy **kwargs construction.
        if config is None:
            config = DAGLevenshteinConfig(**kwargs)
        super().__init__(config=config)

        if self.di_anchor:
            raise NotImplementedError(
                "di-m6A (two anchors per CB) is implemented for the pairwise "
                "extractor only. The di-m6A library uses 19-nt spacers, which is "
                "the pairwise algorithm; the Levenshtein "
                "extractor is for designs with 6-nt spacers. Use dag_method='pw'."
            )

        self.nucs = "ACGT"
        self.skip_size_tolerance = config.skip_size_tolerance
        self.spacer_mismatch_tolerance = config.spacer_mismatch_tolerance
        self.spacer_size_tolerance = config.spacer_size_tolerance

        self.spacer_kmer_ed_dict = {i: self.get_ed_kmers(kmer) for i, kmer in enumerate(self.spacer_list)}

        # v3: max_cb_penalty is byte-identical to v1.  The anchor-uniqueness
        # penalty does NOT enter the path-scoring penalty (it would corrupt
        # DAG path selection); it is applied post-hoc as a multiplier on the
        # cb_info_dict output score only.
        self.max_cb_penalty = (
            self.anchor_mismatch_penalty
            + self.spacer_mismatch_penalty * self.spacer_mismatch_tolerance
            + self.indel_penalty * self.indel_tolerance
        )
        self.score_converting_func = lambda x: 1 - (x / (2 * self.max_cb_penalty))
        self.cb_size = 2 * self.cb_pad + 1
        self.bb_size = self.cb_size * self.cb_per_bb + self.spacer_size
        self.indel_dict = self.get_integer_partition()
        self.min_ideal_displacement_dict = self.get_min_ideal_displacement_dict()


    def get_min_ideal_displacement_dict(self):
        min_ideal_displacement_dict = {}
        big_step_size = self.cb_size + self.spacer_size
        small_step_size = self.spacer_size

        for from_idx in range(self.cb_per_bb + 1):
            for to_idx in range(self.cb_per_bb + 1):
                if from_idx < to_idx:
                    small_steps = 0
                    big_steps = to_idx - from_idx
                    displacement = big_step_size * big_steps
                else:
                    small_steps = 1
                    big_steps = self.cb_per_bb - from_idx + to_idx
                    displacement = big_step_size * big_steps + small_step_size
                min_ideal_displacement_dict[(from_idx, to_idx)] = (displacement, small_steps, big_steps)

        return min_ideal_displacement_dict


    def get_ideal_displacement(self, from_spacer_idx, to_spacer_idx, displacement):
        min_ideal_displacement, min_small_steps, min_big_steps = self.min_ideal_displacement_dict[
            (from_spacer_idx, to_spacer_idx)]
        if displacement <= min_ideal_displacement:
            ideal_displacement = min_ideal_displacement
            small_steps = min_small_steps
            big_steps = min_big_steps
        else:
            periodicity = round((displacement - min_ideal_displacement) / self.bb_size)
            ideal_displacement = min_ideal_displacement + periodicity * self.bb_size
            small_steps = min_small_steps + periodicity
            big_steps = min_big_steps + self.cb_per_bb * periodicity

        return ideal_displacement, small_steps, big_steps

    def get_ed_kmers(self, kmer):
        if pl is None:
            raise ImportError(
                "The Levenshtein extractor needs the 'polyleven' package "
                "(pip install polyleven). The pairwise extractor, which is what "
                "di-m6A uses, does not."
            )
        possible_nucs = ["".join(x) for x in it.product(self.nucs, repeat=len(kmer))]
        kmer_ed_dict = defaultdict(list)
        for possible_kmer in possible_nucs:
            ed = pl.levenshtein(kmer, possible_kmer, self.spacer_mismatch_tolerance)
            kmer_ed_dict[ed].append(possible_kmer)
        kmer_ed_dict[self.spacer_mismatch_tolerance + 1] = []
        return kmer_ed_dict


    def get_kmer_tuple(self, from_spacer_kmer_ed_dict, to_spacer_kmer_ed_dict):
        kmer_tuple_list = []
        for total_mismatch in range(self.spacer_mismatch_tolerance + 1):
            for front_mismatch in range(total_mismatch + 1):
                back_mismatch = total_mismatch - front_mismatch
                for from_kmer, to_kmer in it.product(from_spacer_kmer_ed_dict[front_mismatch],
                                                     to_spacer_kmer_ed_dict[back_mismatch]):
                    kmer_tuple_list.append((from_kmer, to_kmer, total_mismatch))
        return kmer_tuple_list


    def validate_anchor(self, read, from_pos, to_pos, anchor_idx, displacement_error):
        """LEV anchor validator (v3).

        Returns `(penalty, missing_anchor, anchor_pos, n_matches)`.  The
        first three are byte-identical to v1.  `n_matches` is emitted
        for downstream anchor_conf weighting (applied in
        find_block_candidates after the path is selected).  See module
        docstring for the rationale.
        """
        displacement_error_abs = np.abs(displacement_error)
        if self.skip_anchor:
            center = from_pos + self.spacer_size + (to_pos - (from_pos + self.spacer_size)) // 2
            return displacement_error_abs * self.indel_penalty, 0, center, 1

        single_anchor = self.anchor_list[anchor_idx]
        possible_indel_list = self.indel_dict[displacement_error]
        query = read[from_pos + self.spacer_size:to_pos]
        anchor_candidate_list = [
            (displacement_error_abs * self.indel_penalty + self.anchor_mismatch_penalty, 1, None)
        ]

        for front_indel, back_indel in possible_indel_list:
            anchor_start = self.cb_pad + front_indel
            anchor_end = anchor_start + len(single_anchor)
            if anchor_start < 0 or anchor_end > len(query):
                continue
            anchor_query = query[anchor_start:anchor_end]
            if anchor_query == single_anchor:
                anchor_pos = from_pos + self.spacer_size + self.cb_pad + front_indel + self.anchor_shift[anchor_idx]
                total_indel = np.abs(front_indel) + np.abs(back_indel)
                anchor_candidate_list.append((total_indel * self.indel_penalty, 0, anchor_pos))

        anchor_candidate_list.sort(key=lambda x: (x[0], x[1]))
        base_penalty, missing_anchor, anchor_pos = anchor_candidate_list[0]

        n_matches = self._count_anchor_matches_in_window(
            query, single_anchor, displacement_error
        )
        return base_penalty, missing_anchor, anchor_pos, n_matches


    def find_block_candidates(self, seq):
        kmer_pos_dict = defaultdict(list)
        for i in range(len(seq) - self.spacer_size + 1):
            kmer = seq[i:i + self.spacer_size]
            kmer_pos_dict[kmer].append(i)

        dag_list = []  ## Format: [from_pos, to_pos, score]
        cb_info_dict = {}  ## Format: {(from_pos, to_pos): [cb_idx,from_pos,to_pos,anchor_pos,score]}

        for from_spacer_idx in range(len(self.spacer_list)):

            if from_spacer_idx == self.cb_per_bb:
                single_anchor = None
            else:
                single_anchor = self.anchor_list[from_spacer_idx]

            from_spacer_kmer_ed_dict = self.spacer_kmer_ed_dict[from_spacer_idx]

            for to_spacer_idx in range(len(self.spacer_list)):

                to_spacer_kmer_ed_dict = self.spacer_kmer_ed_dict[to_spacer_idx]
                kmer_tuple_list = self.get_kmer_tuple(from_spacer_kmer_ed_dict,to_spacer_kmer_ed_dict)

                for from_kmer, to_kmer, kmer_mismatch in kmer_tuple_list:
                    for from_pos, to_pos in it.product(kmer_pos_dict[from_kmer], kmer_pos_dict[to_kmer]):

                        displacement = to_pos - from_pos
                        if displacement < 0:
                            continue

                        ideal_displacement, small_steps, big_steps = self.get_ideal_displacement(from_spacer_idx, to_spacer_idx,
                                                                                                 displacement)
                        displacement_tolerance_skip = big_steps * self.skip_size_tolerance

                        displacement_error = displacement - ideal_displacement
                        displacement_error_abs = np.abs(displacement_error)

                        if displacement_error_abs > displacement_tolerance_skip:
                            continue

                        anchor_pos = None
                        is_cb = False
                        n_matches_cb = 1   # v3: only meaningful for CB edges

                        if big_steps == 1 and small_steps == 0 and displacement_error_abs <= self.cb_size_tolerance and single_anchor is not None:
                            indel_anchor_penalty, missing_anchor, anchor_pos, n_matches_cb = self.validate_anchor(
                                seq, from_pos, to_pos, from_spacer_idx, displacement_error,
                            )
                            if missing_anchor == 0:
                                is_cb = True

                        elif big_steps == 0 and small_steps == 1 and displacement_error_abs <= self.spacer_size_tolerance:
                            indel_anchor_penalty = self.indel_penalty * displacement_error_abs

                        else:
                            indel_anchor_penalty = self.indel_penalty * displacement_error_abs + self.anchor_mismatch_penalty

                        # v3: `score` here is the v1-equivalent score; drives DAG path selection.
                        penalty = self.spacer_mismatch_penalty * kmer_mismatch + indel_anchor_penalty
                        score = self.score_converting_func(penalty)
                        from_pos_id = (from_spacer_idx, from_pos)
                        to_pos_id = (to_spacer_idx, to_pos)

                        if is_cb:
                            # v3: subtract anchor-uniqueness penalty in
                            # penalty-space BEFORE score_converting_func.
                            # Applied AFTER score has been added to dag_list,
                            # so path selection is byte-identical to v1.
                            uncertainty_penalty = (
                                self.anchor_uncertainty_penalty * max(0, n_matches_cb - 1)
                            )
                            output_score = self.score_converting_func(
                                penalty + uncertainty_penalty
                            )
                            cb_info_dict[(from_pos_id, to_pos_id)] = [from_spacer_idx, anchor_pos, output_score]

                        # dag_list weights MUST be v1-equivalent.
                        dag_list.append((from_pos_id, to_pos_id, score))

        return cb_info_dict, dag_list


    def extract_blocks_read(self, seq):
        keys = list(self.RESULT_KEYS)
        result_dict = {key: [] for key in keys}

        cb_info_dict, dag_list = self.find_block_candidates(seq)

        if len(cb_info_dict) == 0:
            return {key: [] for key in keys}

        global_idx = 0
        longest_path = self.dag_longest_path(dag_list)
        for x, y in zip(longest_path[:-1], longest_path[1:]):
            cb_idx, pos, score = cb_info_dict.get((x, y), (None, None, None))
            if pos is not None and score is not None:
                result_dict["cb_idx"].append(cb_idx)
                result_dict["global_idx"].append(global_idx)
                result_dict["anchor_idx"].append(0)
                result_dict["pos"].append(pos)
                result_dict["score"].append(score)
                result_dict["di_gap"].append(MONO_GAP)
                global_idx += 1

        return result_dict
