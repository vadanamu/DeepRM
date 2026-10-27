# src/deeprm/inference/cli.py
from __future__ import annotations

import argparse
import sys
from importlib import import_module
from typing import List

# Map subcommand -> module path (not imported yet)
_LEAVES = {
    "prep": "deeprm.inference.inference_preprocess",
    "run": "deeprm.inference.inference",
    "pileup": "deeprm.inference.pileup_deeprm",
}


def parser() -> argparse.ArgumentParser:
    gp = argparse.ArgumentParser(
        prog="deeprm call",
        description="DeepRM Call (inference) Module",
        add_help=True,
    )
    sub = gp.add_subparsers(dest="cmd", metavar="{prep,run,pileup}")
    # Skeleton subparsers: no flags yet; no imports
    for name, help_text in [
        ("prep", "Preprocess POD5 and BAM files for inference"),
        ("run", "Run the model on preprocessed data, then the pileup"),
        ("pileup", "Aggregate per-read predictions into site-level results"),
    ]:
        sub.add_parser(name, help=help_text, add_help=False)
    return gp


def _get_doc_firstline(mod) -> str:
    """Get the first line of the module's docstring, or a default message."""
    doc = getattr(mod, "__doc__", None)
    if doc:
        doc = doc.splitlines()
        if len(doc) == 1:
            return doc[0]
        elif len(doc) > 1:
            if doc[0].strip():
                return doc[0].strip()
            elif doc[1].strip():
                return doc[1].strip()
    return "No documentation available for this command."


def _build_leaf_parser(cmd: str, prog: str) -> argparse.ArgumentParser:
    """Import only the requested leaf, ask it to add its flags, and return the parser."""
    mod = import_module(_LEAVES[cmd])
    leaf = argparse.ArgumentParser(prog=prog, add_help=True, description=_get_doc_firstline(mod))
    # Preferred light interface
    if hasattr(mod, "add_arguments") and callable(mod.add_arguments):
        mod.add_arguments(leaf)  # type: ignore
    else:
        # Fallback: if module only has parse_args(), reflect flags by calling it with --help
        pass  # Usually not needed if you implement add_arguments() in the leaf
    return leaf


def entry(argv: List[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    gp = parser()

    # If group-only help
    if not argv or argv[0] in ("-h", "--help"):
        gp.print_help()
        return 0

    # Parse only 'cmd' without importing leaves
    ns, rest = gp.parse_known_args(argv)
    if ns.cmd is None:
        gp.print_help()
        return 2
    if ns.cmd not in _LEAVES:
        gp.error(f"Unknown subcommand '{ns.cmd}'")

    # If leaf help requested (e.g., `deeprm call run --help`), import only that leaf
    if not rest or any(h in rest for h in ("-h", "--help")):
        lp = _build_leaf_parser(ns.cmd, prog=f"deeprm call {ns.cmd}")
        lp.parse_args(["--help"])  # prints help and exits(0)
        return 0

    # Normal execution path: import only the requested leaf, parse its flags, run it
    mod = import_module(_LEAVES[ns.cmd])
    lp = _build_leaf_parser(ns.cmd, prog=f"deeprm call {ns.cmd}")
    leaf_args = lp.parse_args(rest)

    # Run
    if hasattr(mod, "main") and callable(mod.main):
        return int(mod.main(leaf_args) or 0)
    gp.error(f"Subcommand '{ns.cmd}' has no entry point")
    return 2
