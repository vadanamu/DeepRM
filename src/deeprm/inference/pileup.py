"""Aggregate per-read predictions into site-level results (pileup), with the C++ binary when it can run."""

from __future__ import annotations

import argparse
import contextlib
import functools
import os
import shutil
import signal
import subprocess
import time
from typing import List

import numpy as np

from deeprm.inference import pileup_deeprm as py_pileup
from deeprm.utils import prep_binary
from deeprm.utils.logging import get_logger
from deeprm.utils.utils import default_threads, maybe_index_bam

log = get_logger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments: those of the Python pileup, and the binary to use."""
    py_pileup.add_arguments(parser)
    parser.add_argument(
        "--pileup-bin",
        type=str,
        default=None,
        help="Path to a pileup binary to use instead of the bundled one (optional); the run stops if it cannot start",
    )


DEFAULT_BIN = prep_binary.bundled(prep_binary.PILEUP)

# The options the C++ binary takes (cpp/PILEUP_INTERFACE.md), in the order they are passed. A run that sets any other
# pileup option to a value other than its default goes to the Python pileup.
_BINARY_OPTIONS = (
    "input",
    "output",
    "bam",
    "thread",
    "threshold",
    "epsilon",
    "postfix",
    "slice",
    "flip",
    "label_div",
    "annot",
    "skip_modbam",
    "tmp_dir",
    "report_all_stoichiometry",
)
# Options the binary gives the Python pileup's results for only at their default: it has log10(1 - threshold) as
# NumPy computes it for the default threshold only, and takes any other from the C library, which can differ in the
# last bit. A run with another value goes to the Python pileup.
_BINARY_DEFAULT_ONLY = ("threshold",)
# Options of this wrapper; they do not decide which path runs.
_NOT_A_PATH_CHOICE = ("pileup_bin",)

# NumPy's results the binary must reproduce, checked before it is used: the per-read terms of float32 probabilities
# around the values that matter (bit patterns: first, count), from `deeprm_pileup terms-range`, and log10(1 - 0.98) of
# the site rule. The binary computes them as NumPy does with AVX512; NumPy computes them otherwise on a CPU without
# AVX512, and another NumPy version may too: then the Python pileup runs.
_PROBE = (
    (0x33D6BF95, 4096),  # about 1e-7
    (0x3C23D70A, 4096),  # about 0.01
    (0x3F000000, 4096),  # 0.5
    (0x3F666666, 4096),  # 0.9
    (0x3F7AE148 - 2048, 4096),  # around 0.98, the rule's cut
    (0x3F7FF000, 4096),  # up to 1
)
# the arrays of `terms-range`, in its order, with the dtype it writes them in
_TERMS = (
    ("count_pos", "u1"),
    ("logsum_1_p_pos", "<f8"),
    ("kl_div_neg", "<f4"),
    ("kl_div_pos", "<f4"),
    ("count_pos_rule", "u1"),
    ("logsum_1_p_pos_rule", "<f8"),
    ("logsum_1_p_pos_calib", "<f8"),
)
# np.log10(1 - 0.98) as the binary has it (cpp/pileup/pileup/site_scores.cc)
_BINARY_LOG10_CUT = float.fromhex("-0x1.b2efb2bd8217fp+0")


@functools.cache
def _numpy_differs(bin_path: str, threshold: float, epsilon: float) -> str | None:
    """What NumPy here computes otherwise than the binary at `bin_path`, or None when nothing (see _PROBE)."""
    if float(np.log10(1 - 0.98)) != _BINARY_LOG10_CUT:
        return "log10(1 - 0.98)"
    for calibrated in (False, True):
        for first, count in _PROBE:
            cmd = [
                bin_path,
                "terms-range",
                str(first),
                str(count),
                repr(threshold),
                repr(epsilon),
                str(int(calibrated)),
            ]
            try:
                done = subprocess.run(cmd, capture_output=True, timeout=120, check=True)
                pred = np.arange(first, first + count, dtype=np.uint32).view(np.float32)
                ours = py_pileup._per_read_terms(pred, threshold, epsilon, calibrated)
                at = 0
                for name, dtype in _TERMS:
                    theirs = np.frombuffer(done.stdout, dtype=dtype, count=count, offset=at)
                    at += count * np.dtype(dtype).itemsize
                    mine = np.asarray(ours[name])
                    if theirs.astype(mine.dtype).tobytes() != mine.tobytes():
                        return f"the per-read term {name}"
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                return f"the per-read terms (the check failed: {exc})"
    return None


class _BinaryFailed(Exception):
    """The binary started and ended with a non-zero exit code."""

    def __init__(self, code: int):
        super().__init__(code)
        self.code = code


# Whether the binary can start here (`--version` exits 0) and, if not, why: a prep_binary.Status.
_binary_status = functools.partial(prep_binary.status, program=prep_binary.PILEUP)


def _pileup_actions() -> dict:
    """The Python pileup's options by name (dest): their default and their long option string."""
    parser = argparse.ArgumentParser()
    py_pileup.add_arguments(parser)
    return {
        action.dest: (action.default, next(o for o in action.option_strings if o.startswith("--")))
        for action in parser._actions
        if action.dest != "help"
    }


def _python_only_options(args: argparse.Namespace) -> List[str]:
    """The pileup options this run sets to a value other than their default that the binary does not take, or takes
    only at its default (_BINARY_DEFAULT_ONLY). Options of deeprm call run, which passes its own settings on, are not
    pileup options and are left out."""
    found = []
    for key, (default, name) in _pileup_actions().items():
        if key in _NOT_A_PATH_CHOICE or not hasattr(args, key):
            continue
        if (key not in _BINARY_OPTIONS or key in _BINARY_DEFAULT_ONLY) and getattr(args, key) != default:
            found.append(name)
    return found


def _args_to_cmd(args: argparse.Namespace) -> List[str]:
    """The binary's command-line options for this run: each option of _BINARY_OPTIONS that has a value, under the
    name the Python pileup gives it, as --name value (a flag only when it is set)."""
    names = {key: name for key, (_, name) in _pileup_actions().items()}
    cmd: List[str] = []
    for key in _BINARY_OPTIONS:
        value = getattr(args, key, None)
        if value is None or (key == "postfix" and value == ""):
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(names[key])
        else:
            cmd.extend([names[key], str(value)])
    return cmd


def _binary_path(args: argparse.Namespace) -> tuple[str, str | None]:
    """The binary to run and the option that named it (None for the bundled binary); see prep_binary.named_path."""
    named = getattr(args, "pileup_bin", None)  # deeprm call run and a caller's own Namespace may leave it out
    if named is not None:
        if not named.strip():
            raise ValueError("--pileup-bin is empty: give the path of a pileup binary, or leave it out")
        return prep_binary.named_path(named), "--pileup-bin"
    if os.environ.get("DEEPRM_PILEUP_BIN"):
        return prep_binary.named_path(os.environ["DEEPRM_PILEUP_BIN"]), "DEEPRM_PILEUP_BIN"
    return DEFAULT_BIN, None


def _named_cannot_start(named_by: str, status: prep_binary.Status) -> ValueError:
    """The error for a binary the user named (--pileup-bin or DEEPRM_PILEUP_BIN) that cannot start."""
    hint = f" {status.hint}" if status.hint else " Use a binary that runs on this system."
    if named_by == "--pileup-bin" and os.environ.get("DEEPRM_PILEUP_BIN"):
        instead = f"the binary given with DEEPRM_PILEUP_BIN ({os.environ['DEEPRM_PILEUP_BIN']})"
    else:
        instead = "the bundled binary (or the Python pileup where that cannot start)"
    return ValueError(
        f"The pileup binary given with {named_by} cannot start: {status.reason}.{hint} Remove {named_by} to use "
        f"{instead}."
    )


def check_named_binary(args: argparse.Namespace) -> None:
    """Raises the error main() would raise for a binary named with --pileup-bin or DEEPRM_PILEUP_BIN that cannot
    start; deeprm call run calls it before the inference, so that such a binary does not stop the run hours later."""
    bin_path, named_by = _binary_path(args)
    if named_by:
        status = _binary_status(bin_path)
        if not status.ok:
            raise _named_cannot_start(named_by, status)


def main(args: argparse.Namespace):
    """Runs deeprm call pileup, and the pileup of deeprm call run: the C++ binary where it can start and the options
    allow it, else the Python pileup (deeprm.inference.pileup_deeprm).

    Both paths get the same settings: the arguments are checked once, and --thread is resolved here (the sums are
    added in an order that follows the number of processes, so the two paths must use the same number to give the
    same results).

    Args:
        args (argparse.Namespace): Parsed command-line arguments (see add_arguments).

    Returns:
        None: Results are saved in the output directory.

    Raises:
        ValueError: For a binary named with --pileup-bin or DEEPRM_PILEUP_BIN that cannot start or that fails. A
            bundled binary that fails is followed by the Python pileup.
    """
    py_pileup._validate_args(args)
    if args.thread is None:
        args.thread = default_threads()

    python_only = _python_only_options(args)
    if python_only:
        log.info("Using the Python pileup: the binary does not give its results with %s", ", ".join(python_only))
        return py_pileup.main(args)

    bin_path, named_by = _binary_path(args)
    if named_by is None and not os.path.exists(bin_path):
        log.info("Using the Python pileup: this installation has no pileup binary")
        return py_pileup.main(args)

    # A file this user cannot run (for example after an unpacking tool dropped its execute bits) gets them back. A
    # file on a noexec file system is left alone: chmod cannot help there.
    if os.path.isfile(bin_path) and not os.access(bin_path, os.X_OK) and not prep_binary.on_noexec_mount(bin_path):
        try:
            os.chmod(bin_path, os.stat(bin_path).st_mode | 0o111)
            log.info("Set the execute permission of the pileup binary at %s", bin_path)
        except OSError as exc:
            log.warning("Could not set the execute permission of %s: %s", bin_path, exc)

    # Start the binary only when it can run here; otherwise the Python pileup does the whole run. A binary the user
    # named must start.
    status = _binary_status(bin_path)
    if not status.ok:
        if named_by:
            raise _named_cannot_start(named_by, status)
        hint = f" {status.hint}" if status.hint else ""
        log.warning(
            "Falling back to the Python pileup: the binary at %s cannot start: %s.%s Run `deeprm check` for details.",
            bin_path,
            status.reason,
            hint,
        )
        return py_pileup.main(args)

    # The binary's results are the Python pileup's only where NumPy computes as the binary does
    why = _numpy_differs(bin_path, args.threshold, args.epsilon)
    if why and not named_by:
        log.info(
            "Using the Python pileup: NumPy here does not compute %s as the pileup binary does (for example on a CPU "
            "without AVX512), so the binary's results would not be those of the Python pileup",
            why,
        )
        return py_pileup.main(args)
    if why:
        log.warning(
            "NumPy here does not compute %s as the pileup binary given with %s does: its results may differ from "
            "the Python pileup's in the last digits",
            why,
            named_by,
        )
    try:
        return _run_binary(args, bin_path, status)
    except _BinaryFailed as exc:
        if named_by:
            raise ValueError(
                f"The pileup binary given with {named_by} failed (exit code {exc.code}); see its messages above."
            ) from None
        # A bundled binary that fails on an input the Python pileup can read (for example one it was not written
        # for) must not stop the run: the Python pileup runs, and fails with its own message on a bad input.
        log.warning(
            "The pileup binary failed (exit code %s; see its messages above); running the Python pileup instead.",
            exc.code,
        )
        return py_pileup.main(args)


def _binary_work(args: argparse.Namespace, name: str) -> bool:
    """Whether `name`, in the output folder or the temporary folder, is what the binary writes while it runs: the site
    tables as .part files, the modBAM and its index, and its temporary folder."""
    bam = f"modbam_{os.path.basename(args.bam)}"
    tables = [f"{stem}.part.{ext}" for stem in ("pileup", f"genomic_pileup{args.postfix}") for ext in ("bed", "npz")]
    return name in tables or name in (bam, f"{bam}.part", f"{bam}.bai") or name.startswith(".deeprm_modbam_")


def _remove_binary_work(args: argparse.Namespace, before: dict) -> None:
    """After a binary run that failed or was stopped (Ctrl-C ends the binary at once): removes what it was writing,
    as the Python pileup leaves no temporary file and no output of a run that fails. Files that were there before the
    run are left alone."""
    with py_pileup._sigint_held():
        for folder, names in before.items():
            if not os.path.isdir(folder):
                continue
            for name in sorted(set(os.listdir(folder)) - names):
                if _binary_work(args, name):
                    path = os.path.join(folder, name)
                    if os.path.isdir(path) and not os.path.islink(path):
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        with contextlib.suppress(OSError):
                            os.remove(path)


def _run_binary(args: argparse.Namespace, bin_path: str, status: prep_binary.Status) -> None:
    """Runs the binary with what the Python pileup would prepare before it starts: an indexed (and sorted) BAM when
    the modBAM is written, the output folder, and whether the per-read files get the v1.2 model's calibration
    (inputs_calibrated, with its warnings). Raises _BinaryFailed when the binary ends with an error."""
    start = time.time()
    if not args.skip_modbam:  # the modBAM is written from the BAM through its index
        args.bam = maybe_index_bam(args.bam, args.thread)
    os.makedirs(args.output, exist_ok=True)
    calibrated = py_pileup.inputs_calibrated(args.input)
    cmd = [bin_path] + _args_to_cmd(args) + ["--calibration", "on" if calibrated else "off"]
    log.info("Using the C++ pileup binary %s (%s)", bin_path, status.version)
    folders = {args.output, getattr(args, "tmp_dir", None) or args.output}
    before = {folder: set(os.listdir(folder)) if os.path.isdir(folder) else set() for folder in folders}
    try:
        code = subprocess.run(cmd).returncode
    except BaseException:  # Ctrl-C (the binary gets it too and ends), or any other stop
        _remove_binary_work(args, before)
        raise
    if code != 0:
        _remove_binary_work(args, before)
        if code == -signal.SIGINT:  # Ctrl-C that reached the binary only
            raise KeyboardInterrupt
        if code < 0:
            raise ValueError(f"The pileup binary was stopped by signal {-code}.")
        raise _BinaryFailed(code)
    log.info(f"Pileup finished in {time.time() - start:.2f} seconds.")
    return None
