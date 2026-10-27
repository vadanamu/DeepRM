"""Preprocess POD5 and BAM files for inference, with the C++ binary when it can run."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import List

from deeprm.inference import inference_preprocess_python as py_prep
from deeprm.utils import prep_binary
from deeprm.utils.logging import get_logger

log = get_logger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments, mirroring the Python preprocessing module."""
    py_prep.add_arguments(parser)
    parser.add_argument(
        "--max-queue",
        "-Q",
        type=int,
        default=1000,
        help="BAM records queued per worker when the BAM is streamed (-b - or --bam-cmd); binary only",
    )
    parser.add_argument(
        "--preprocess-bin",
        type=str,
        default=None,
        help="Path to a preprocessing binary to use instead of the bundled one (optional); the run stops if it "
        "cannot start",
    )


DEFAULT_BIN = os.path.join(Path(__file__).parent.resolve(), "deeprm_preprocess")

# The options the C++ binary accepts (cpp/src/args/ArgumentParser.cpp). The binary stops on any other option, so
# only these are passed to it; a run that sets one of the Python-only options goes to the Python preprocessing.
_BINARY_OPTIONS = (
    "pod5",
    "bam",
    "bam_cmd",
    "output",
    "thread",
    "qcut",
    "chunk",
    "max_token_len",
    "sampling",
    "boi",
    "kmer_len",
    "cb_len",
    "bam_thread",
    "dwell_shift",
    "sig_window",
    "label_div",
    "filter_flag",
    "zero_pad",
    "edge_pad",
    "trim",
    "trim_threshold",
    "trim_penalty",
    "max_queue",
)
# Options of this wrapper, and Python options the binary has no use for (it does not read the POD5 files in
# batches); neither decides which path runs.
_NOT_A_PATH_CHOICE = ("preprocess_bin", "max_queue", "process_once")
# Binary options passed only when they differ from their default (the binary's defaults are the same), so a binary
# built from older C++ code, which lacks them, can still run a run that does not set them.
_ONLY_WHEN_SET = ("trim_threshold", "trim_penalty", "max_queue")


def _defaults() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser


def _args_to_cmd(args: argparse.Namespace) -> List[str]:
    defaults = _defaults()
    cmd: List[str] = []
    for key, value in vars(args).items():
        if key not in _BINARY_OPTIONS or value is None:
            continue
        if key == "bam" and args.bam_cmd:
            # "-b -" (or an empty -b) next to --bam-cmd means the same input; the binary takes only one of them
            continue
        if key in _ONLY_WHEN_SET and value == defaults.get_default(key):
            continue
        if key == "boi":
            # The binary takes A, C, G or T only; the Python preprocessing also takes lower case and U.
            value = str(value).upper().replace("U", "T")
        opt = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(opt)
        else:
            cmd.extend([opt, str(value)])
    return cmd


def _python_only_options(args: argparse.Namespace) -> List[str]:
    """Return the Python-only options this run sets to a value other than their default."""
    defaults = argparse.ArgumentParser()
    py_prep.add_arguments(defaults)
    found = []
    for key, value in vars(args).items():
        if key in _BINARY_OPTIONS or key in _NOT_A_PATH_CHOICE:
            continue
        if value != defaults.get_default(key):
            found.append(f"--{key.replace('_', '-')}")
    return found


# Whether the binary can start here (`--version` exits 0) and, if not, why: a prep_binary.Status.
_binary_status = prep_binary.status


def _binary_path(args: argparse.Namespace) -> tuple[str, str | None]:
    """The binary to run and the option that named it (None for the bundled binary); see prep_binary.named_path."""
    if args.preprocess_bin is not None:
        if not args.preprocess_bin.strip():
            raise ValueError("--preprocess-bin is empty: give the path of a preprocessing binary, or leave it out")
        return prep_binary.named_path(args.preprocess_bin), "--preprocess-bin"
    if os.environ.get("DEEPRM_PREPROCESS_BIN"):
        return prep_binary.named_path(os.environ["DEEPRM_PREPROCESS_BIN"]), "DEEPRM_PREPROCESS_BIN"
    return DEFAULT_BIN, None


def _check_shared_values(args: argparse.Namespace) -> None:
    """Refuse values that neither path can run, with the same message on both paths."""
    boi = str(args.boi).upper()
    if boi not in ("A", "C", "G", "T", "U"):
        raise ValueError(f"Base of interest {args.boi} not supported, must be one of A,C,G,T")
    py_prep.validate_geometry(args.cb_len, args.kmer_len, args.sig_window, args.chunk)
    if args.filter_flag < 0 or args.filter_flag > 0xFFF:
        raise ValueError(f"--filter-flag must be a SAM flag mask in [0, 4095] (got {args.filter_flag})")
    for name in ("thread", "bam_thread", "sampling", "process_once", "max_token_len", "label_div", "max_queue"):
        if getattr(args, name, 1) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.qcut < 0:
        raise ValueError("--qcut must be >= 0")
    if args.dwell_shift < 0:
        raise ValueError("--dwell-shift must be >= 0")
    if args.bam_cmd and args.bam not in (None, "", "-"):
        raise ValueError("pass either --bam or --bam-cmd, not both")
    if args.bam is None and not args.bam_cmd:
        raise ValueError("--bam is required (a BAM file, or - for standard input), unless --bam-cmd is given")


def _descriptor_of(bam: str | None) -> tuple[int, ...]:
    """The open file descriptor that a -b path such as /dev/fd/63 (what <(command) gives) or /proc/self/fd/7 names.

    The binary must inherit it to read the path, and subprocess closes every descriptor above 2 that it is not told
    to pass on. A descriptor that is not open here is not passed: the binary then says that the path does not exist.
    """
    m = re.fullmatch(r"/(?:dev|proc/self)/fd/(\d+)", bam or "")
    if not m or int(m.group(1)) <= 2:
        return ()
    try:
        os.fstat(int(m.group(1)))
    except OSError:
        return ()
    return (int(m.group(1)),)


def _npz_files(folder: str) -> List[str]:
    try:
        return [name for name in os.listdir(folder) if name.endswith(".npz")]
    except (FileNotFoundError, NotADirectoryError):
        return []


def main(args: argparse.Namespace) -> int:
    """Runs `deeprm call prep`: the C++ binary where it can start and the options allow it, else the Python
    preprocessing (deeprm.inference.inference_preprocess_python).

    Args:
        args (argparse.Namespace): Parsed command-line arguments (see add_arguments).

    Returns:
        int: The exit code, 0 on success.

    Raises:
        ValueError: For example if the output folder already holds .npz files from an earlier run, or if the run
            wrote no rows.
    """
    _check_shared_values(args)
    if getattr(args, "build_sidecar", False) or getattr(args, "validate_sizing", False):
        return _run(args)  # these modes write no rows on purpose
    # The output file names change between runs (and from 1.1.3), so files left from an earlier run would not be
    # replaced, and deeprm call run would read them together with the new ones.
    old = _npz_files(args.output)
    if old:
        raise ValueError(
            f"The output folder {args.output} already holds {len(old)} .npz files from an earlier run, and deeprm call "
            "run would read them together with the new ones. Remove them, or choose another output folder."
        )
    rc = _run(args)
    if rc == 0 and not _npz_files(args.output):
        raise ValueError(
            f"The preprocessing wrote no rows to {args.output}: no read of the BAM was found in the POD5 files and "
            "passed the filters. Check that --pod5 holds the POD5 files of the reads in the BAM, that --sampling is "
            "the basecaller's stride (the messages above count the reads skipped for it), and the --qcut and "
            "--filter-flag values."
        )
    return rc


def _run(args: argparse.Namespace) -> int:
    python_only = _python_only_options(args)
    if python_only:
        log.info("Using Python preprocessing: the binary has no %s option", ", ".join(python_only))
        return int(py_prep.main(args) or 0)
    if os.path.isfile(args.pod5):
        log.info("Using Python preprocessing: the binary reads a POD5 folder, not a single file")
        return int(py_prep.main(args) or 0)

    bin_path, named_by = _binary_path(args)

    # A file this user cannot run (for example after an unpacking tool dropped its execute bits) gets them back. A
    # file on a noexec file system is left alone: chmod cannot help there.
    if os.path.isfile(bin_path) and not os.access(bin_path, os.X_OK) and not prep_binary.on_noexec_mount(bin_path):
        try:
            os.chmod(bin_path, os.stat(bin_path).st_mode | 0o111)
            log.info("Set the execute permission of the preprocessing binary at %s", bin_path)
        except OSError as exc:
            log.warning("Could not set the execute permission of %s: %s", bin_path, exc)

    # Start the binary only when it can run here; otherwise the Python preprocessing does the whole run (the
    # standard input or the --bam-cmd command has not been read yet). A binary the user named must start.
    status = _binary_status(bin_path)
    if not status.ok:
        hint = f" {status.hint}" if status.hint else ""
        if named_by:
            if named_by == "--preprocess-bin" and os.environ.get("DEEPRM_PREPROCESS_BIN"):
                instead = f"the binary given with DEEPRM_PREPROCESS_BIN ({os.environ['DEEPRM_PREPROCESS_BIN']})"
            else:
                instead = "the bundled binary (or the Python preprocessing where that cannot start)"
            raise ValueError(
                f"The preprocessing binary given with {named_by} cannot start: {status.reason}."
                f"{hint or ' Use a binary that runs on this system.'} Remove {named_by} to use {instead}."
            )
        log.warning(
            "Falling back to Python preprocessing, which is 2 to 5 times slower: the binary at %s cannot start: "
            "%s.%s Run `deeprm check` for details.",
            bin_path,
            status.reason,
            hint,
        )
        return int(py_prep.main(args) or 0)

    cmd = [str(bin_path)] + _args_to_cmd(args)
    log.info("Using the accelerated preprocessing binary %s (%s)", bin_path, status.version)
    if args.process_once != _defaults().get_default("process_once"):
        log.info("--process-once applies to the Python preprocessing only; the binary does not use it")
    try:
        subprocess.run(cmd, check=True, pass_fds=_descriptor_of(args.bam))
    except subprocess.CalledProcessError as exc:
        # The binary started and failed on this input: its own messages above say why. There is no fallback,
        # because the input may be partly read (a stream) and the output folder may hold part of the output.
        written = _npz_files(args.output)
        output = (
            f"The {len(written)} .npz files it wrote to {args.output} are incomplete: remove them before you run again."
            if written
            else "It wrote no output files."
        )
        raise ValueError(
            f"The preprocessing binary failed (exit code {exc.returncode}); see its messages above. {output}"
        ) from exc
    return 0
