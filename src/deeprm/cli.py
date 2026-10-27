# src/deeprm/cli.py
from __future__ import annotations

import argparse
import os
import signal
import sys
from argparse import RawTextHelpFormatter
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Dict, List

# -------------------------------------------------------------------
# Static registry used ONLY to render a rich top-level help page.
# This keeps top-level help detailed without importing heavy modules.
# -------------------------------------------------------------------
_HELP_REGISTRY: Dict[str, dict] = {
    "call": {
        "desc": "Call modifications (inference)",
        "module": "deeprm.inference.cli",
        "subcommands": [
            ("prep", "Preprocess POD5 and BAM files for inference"),
            ("run", "Run the model on preprocessed data, then the pileup"),
            ("pileup", "Aggregate per-read predictions into site-level results"),
        ],
    },
    "train": {
        "desc": "Prepare training data and train a model",
        "module": "deeprm.train.cli",
        "subcommands": [
            ("prep", "Prepare training data"),
            ("compile", "Compile the training dataset"),
            ("run", "Train a model on one or more GPUs"),
        ],
    },
    "qc": {
        "desc": "Quality control",
        "module": "deeprm.qc.cli",
        "subcommands": [
            ("run", "Plot read length, base quality and poly(A) length of a BAM"),
            ("alignment", "Plot alignment error rates of a BAM"),
            ("block", "Plot the training data of deeprm train prep or DeepRM v1.1"),
        ],
    },
    "check": {
        "desc": "Check the installation",
        "module": "deeprm.utils.check_installation",
    },
}


def resolved_version() -> str:
    try:
        found = version("deeprm")
        if not found:  # an incomplete egg-info, as in an unpacked sdist run with src on the path
            raise PackageNotFoundError("deeprm")
        return found
    except (PackageNotFoundError, KeyError):
        # Editable/uninstalled checkouts
        try:
            from . import __version__  # type: ignore

            return __version__
        except Exception:
            return "0+unknown"


def _format_top_description() -> str:
    return (
        "[DeepRM unified command-line interface]\n\n"
        "Usage:\n"
        "  deeprm <group> [--help]\n"
        "  deeprm <group> <subcommand> [args]\n"
    )


def _format_groups_block() -> str:
    lines: List[str] = []
    lines.append("Groups & subcommands:\n")
    for g, info in _HELP_REGISTRY.items():
        lines.append(f"  {g:<10} {info['desc']}")
        if "subcommands" in info:
            for name, desc in info["subcommands"]:
                lines.append(f"    {name:<10} {desc}")
        lines.append("")  # blank line between groups
    return "\n".join(lines).rstrip() + "\n"


def _format_tips() -> str:
    return (
        "Tips:\n"
        "  • Use '--help' after any command to see its flags.\n"
        "  • See docs: https://deeprm.readthedocs.io/\n"
    )


def _build_top_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deeprm",
        description=_format_top_description(),
        formatter_class=RawTextHelpFormatter,
        add_help=False,  # we add -h/--help manually to control placement
        epilog="\n".join(
            [
                _format_groups_block(),
                _format_tips(),
            ]
        ),
    )
    p.register("action", "parsers", argparse._SubParsersAction)  # for typing clarity

    # Top-level flags
    p.add_argument("--version", "-v", action="version", version=f"DeepRM {resolved_version()}")
    p.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit.",
    )

    # Skeleton subparsers: we *do not* import group modules here.
    sub = p.add_subparsers(dest="group", metavar="{call,train,qc,check}")
    for g, info in _HELP_REGISTRY.items():
        # 'help' here enriches the default subcommand listing in argparse,
        # but we still print a custom epilog with more details.
        sub.add_parser(g, help=info["desc"], add_help=False)

    return p


def main(argv: List[str] | None = None) -> int:
    # A parent that ignores SIGCHLD passes that on to this process, and then no child's exit status can be read:
    # subprocess reports 0 for a preprocessing binary or a --bam-cmd command that failed, so the run would succeed.
    if hasattr(signal, "SIGCHLD") and signal.getsignal(signal.SIGCHLD) == signal.SIG_IGN:
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_top_parser()

    # No args or explicit help -> print the rich top-level page
    if not argv or argv[0] in ("-h", "--help", "help"):
        parser.print_help()
        return 0

    # Parse only enough to know the selected group (still no imports)
    ns, rest = parser.parse_known_args(argv)
    if ns.group is None:
        parser.print_help()
        return 2

    # Lazy-import just the chosen group CLI; it will handle its own argparse
    group_info = _HELP_REGISTRY.get(ns.group)
    if not group_info:
        parser.error(f"Unknown group '{ns.group}'")

    mod = import_module(group_info["module"])
    # Delegate the remaining argv to the group's entry() function
    try:
        return int(mod.entry(rest) or 0)
    except ModuleNotFoundError as exc:
        # The commands import PyTorch when they start (inside entry()). Without it, say how to install it.
        if exc.name != "torch":
            raise
        from deeprm.utils import check_deps

        print(f"PyTorch is not installed. Install it with: {check_deps.torch_install_hint()}", file=sys.stderr)
        return 1
    except ValueError as exc:
        # DeepRM refuses bad arguments and unusable setups with ValueError: print the reason, not a traceback
        if os.environ.get("DEEPRM_TRACEBACK", "") == "1":
            raise
        print(f"deeprm {ns.group}: error: {exc}", file=sys.stderr)
        print("(set DEEPRM_TRACEBACK=1 to see where it was raised)", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
