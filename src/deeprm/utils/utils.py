"""
Utility functions for DeepRM.
"""

import numpy as np

# The run record that deeprm call run writes beside the per-read files (molecule-level/); deeprm call pileup reads from
# it whether the per-read files come from the bundled weights. Here, in a module without PyTorch, so the pileup can
# use the name without importing inference.
RUN_INFO_NAME = "deeprm_run.json"
# md5 of the bundled weights file (weight/deeprm_weights.pt) as shipped. The pileup gives a fixed adjustment only to
# the predictions of these weights (deeprm_run.json records whether the model file has this md5).
BUNDLED_WEIGHTS_MD5 = "920a59eb2e895f48fc226a014246a1f7"


def read_run_info(path):
    """deeprm_run.json as a dict. A file that is cut off or is not JSON is refused with its path (ValueError)."""
    import json

    try:
        with open(path) as f:
            return json.load(f) or {}
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not a complete run record ({e}); use a new output folder.") from e


def save_run_info(path, info):
    """Writes deeprm_run.json under a .part name and renames it, so a stopped run never leaves half a record."""
    import json
    import os

    part = path + ".part"
    with open(part, "w") as f:
        json.dump(info, f, indent=1)
    os.replace(part, path)


def mean_phred(phred):
    """
    Calculates the mean Phred quality score.

    Args:
        phred (numpy.ndarray): Array or list of Phred quality scores.

    Returns:
        float: Mean Phred quality score.
    """
    if not isinstance(phred, np.ndarray):
        phred = np.array(phred, dtype=int)
    else:
        phred = phred.astype(int)
    return -10 * np.log10(np.mean(10 ** (-phred / 10)))


def maybe_index_bam(bam_path: str, threads: int = 1) -> str:
    """
    Ensure a BAM is indexed, and return the path of an indexed BAM with the same reads. The input BAM is never
    rewritten.

    1. The BAM already has a usable index that is not older than the BAM: return it.
    2. Otherwise index it in place (this writes <bam>.bai beside it, or rewrites the existing index); this works when
       the BAM is coordinate-sorted.
    3. Otherwise (it is not coordinate-sorted) use a coordinate-sorted copy beside it. An existing copy
       (<stem>.sorted.bam, <stem>.sorted.1.bam, ...) that has a usable index and was made from this BAM (its size and
       modification time are kept in <copy>.source) is reused. Else the BAM is sorted into the first free name of
       that list and indexed. Copies that are not reused are left in place, and their paths are logged.

    Args:
        bam_path (str): Path to the BAM file.
        threads (int): Threads for samtools sort and index.

    Returns:
        Path (as str) to the BAM that is guaranteed to be indexed (original or sorted copy).

    Raises:
        FileNotFoundError: The BAM does not exist.
        ValueError: The path does not end in .bam, or the BAM header has no @SQ lines (a BAM index needs them).
        RuntimeError: The header cannot be read, all 101 names for a sorted copy are taken, or the sorted copy's
            index is not usable.
    """
    import os
    from pathlib import Path

    import psutil
    import pysam

    from deeprm.utils.logging import get_logger

    log = get_logger(__name__)
    bam_path = Path(bam_path)
    if not bam_path.exists():
        raise FileNotFoundError(f"BAM not found: {bam_path}")
    if bam_path.suffix.lower() != ".bam":
        raise ValueError(f"Expected a .bam file, got: {bam_path.name}")

    def _index_path(p):
        """The index file htslib loads for BAM p: the first of these names that exists (None if none does)."""
        for name in (f"{p}.csi", f"{p.with_suffix('')}.csi", f"{p}.bai", f"{p.with_suffix('')}.bai"):
            if os.path.exists(name):
                return name
        return None

    def _has_usable_index(p, check_age=True) -> bool:
        index = _index_path(p)
        if index is None:
            return False
        # An index older than its BAM may be from an earlier version of it (a BAM re-aligned or re-sorted in place).
        # htslib only warns, and fetch() then misses reads or fails, so such an index is rebuilt.
        if check_age and os.path.getmtime(index) < os.path.getmtime(p):
            return False
        try:
            with pysam.AlignmentFile(str(p), "rb", check_sq=False) as bam:
                # has_index(): index file discoverable; check_index(): validate it matches & is readable
                return bool(bam.has_index()) and bool(bam.check_index())
        except Exception:
            return False

    # Fast path: already indexed & usable
    if _has_usable_index(bam_path):
        return str(bam_path)
    old_index = _index_path(bam_path)  # older than the BAM, or unusable: rebuilt in step 2

    def _keep_old_index(why):
        log.warning(
            f"{old_index} is older than {bam_path} and {why}, so it is used as it is. If the BAM was rewritten after "
            "the index was made, the modBAM may miss reads; run `samtools index` on the BAM where you can write to "
            "its index."
        )
        return str(bam_path)

    # An old index that still loads, but that this user cannot rewrite (a shared or read-only folder): keep it with a
    # warning instead of writing a sorted copy of a BAM that is already sorted.
    if old_index is not None and _has_usable_index(bam_path, check_age=False) and not os.access(old_index, os.W_OK):
        return _keep_old_index("cannot be rebuilt here (no write permission)")

    # Before attempting to index, ensure the BAM has @SQ. Without it, indexing is not possible.
    try:
        with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
            has_sq = bool(bam.header.get("SQ"))
    except Exception as e:
        raise RuntimeError(f"Failed to read BAM header: {bam_path}") from e
    if not has_sq:
        raise ValueError(f"{bam_path} has no @SQ lines in the header; cannot build a BAM index.")

    # Try indexing the BAM as-is. This fails when it is not coordinate-sorted; then use a sorted copy.
    try:
        if old_index is None:
            pysam.index(str(bam_path), nthreads=threads)
        else:
            # Overwrite the index that htslib loads, with one of the same kind.
            log.info(f"Rebuilding {old_index}: it is older than {bam_path}, or it cannot be used.")
            kind = ["-c"] if old_index.endswith(".csi") else []
            pysam.index("-@", str(threads), *kind, str(bam_path), old_index)
        # The index was just built from this BAM, so its age is not checked (a BAM dated in the future, from clock
        # skew, stays newer than any index).
        if _has_usable_index(bam_path, check_age=False):
            return str(bam_path)
        log.warning(f"Indexing {bam_path} did not give a usable index.")
    except pysam.SamtoolsError as e:
        # str() of a SamtoolsError is the repr of its message (quotes and a literal \n); .value is the plain message.
        # This is info, not a warning: an unsorted BAM is a normal input, and the lines below say what is done.
        reason = str(getattr(e, "value", e)).strip()
        if old_index is not None and _has_usable_index(bam_path, check_age=False) and "ermission denied" in reason:
            return _keep_old_index(f"could not be rebuilt ({reason})")
        log.info(f"Could not index {bam_path} (it is probably not coordinate-sorted): {reason}")

    # The names of a sorted copy, beside the input: <stem>.sorted.bam, then <stem>.sorted.1.bam ... .sorted.100.bam.
    # None of them is the input path.
    stem = bam_path.stem
    candidates = [bam_path.with_name(f"{stem}.sorted.bam")]
    candidates += [bam_path.with_name(f"{stem}.sorted.{i}.bam") for i in range(1, 101)]
    # A copy records the size and modification time of the input it was made from, in <copy>.source. It is reused
    # only while they match the input, so a replaced input (even one with an older modification time) gets a new copy,
    # and an input with a modification time in the future does not get a new copy on every run.
    st = bam_path.stat()
    made_from = f"{st.st_size} {st.st_mtime_ns}\n"  # the input a sorted copy was made from

    def _made_from_input(p) -> bool:
        try:
            return p.with_name(p.name + ".source").read_text() == made_from
        except OSError:
            return False

    existing = [p for p in candidates if p.exists()]
    for p in existing:
        if _has_usable_index(p) and _made_from_input(p):
            log.info(f"Using the existing sorted copy {p} of {bam_path}.")
            return str(p)
    for p in existing:
        log.warning(
            f"Not using the sorted copy {p}: it was made from another version of {bam_path}, or has no usable index. "
            "It is left in place; delete it if it is not needed."
        )
    sorted_path = next((p for p in candidates if not p.exists()), None)
    if sorted_path is None:
        raise RuntimeError(
            f"No free name for a sorted copy of {bam_path}: {candidates[0].name} to {candidates[-1].name} all exist."
        )

    # Sort into <name>.tmp, then rename it, so that an interrupted sort never leaves a partial <name>.
    log.info(f"Writing a coordinate-sorted copy of {bam_path}: {sorted_path}")
    tmp_path = sorted_path.with_name(sorted_path.name + ".tmp")
    pysam_thread_memory = max(1, int(0.5 * psutil.virtual_memory().available / (1024**2) / threads))
    pysam.sort("-@", str(threads), f"-m {pysam_thread_memory}M", "-O", "bam", "-o", str(tmp_path), str(bam_path))
    os.replace(tmp_path, sorted_path)
    pysam.index("-@", str(threads), str(sorted_path))

    if not _has_usable_index(sorted_path):
        raise RuntimeError(f"Sorted BAM was created but index is not usable: {sorted_path}")
    # Written last, so that a copy whose sort or index was interrupted is never reused.
    sorted_path.with_name(sorted_path.name + ".source").write_text(made_from)

    return str(sorted_path)
