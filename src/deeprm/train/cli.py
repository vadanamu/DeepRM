# src/deeprm/train/cli.py
from __future__ import annotations

import argparse
import sys
from importlib import import_module
from typing import List

_LEAVES = {
    "prep": "deeprm.train.train_preprocess",
    "compile": "deeprm.train.train_compile",
    "run": "deeprm.train.train",
}


def parser() -> argparse.ArgumentParser:
    gp = argparse.ArgumentParser(prog="deeprm train", description="DeepRM Training Module", add_help=True)
    sub = gp.add_subparsers(dest="cmd", metavar="{prep,run,compile}")
    for name, help_text in [
        ("prep", "Prepare training data"),
        ("compile", "Compile the training dataset"),
        ("run", "Train a model on one or more GPUs"),
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
    if hasattr(mod, "add_arguments"):
        mod.add_arguments(leaf)  # type: ignore
    return leaf


def entry(argv: List[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    gp = parser()
    if not argv or argv[0] in ("-h", "--help"):
        gp.print_help()
        return 0

    ns, rest = gp.parse_known_args(argv)
    if ns.cmd is None:
        gp.print_help()
        return 2
    if ns.cmd not in _LEAVES:
        gp.error(f"Unknown subcommand '{ns.cmd}'")

    if not rest or any(h in rest for h in ("-h", "--help")):
        _build_leaf_parser(ns.cmd, f"deeprm train {ns.cmd}").parse_args(["--help"])
        return 0

    mod = import_module(_LEAVES[ns.cmd])
    lp = _build_leaf_parser(ns.cmd, f"deeprm train {ns.cmd}")
    leaf_args = lp.parse_args(rest)

    # Run
    if hasattr(mod, "main") and callable(mod.main):
        return int(mod.main(leaf_args) or 0)
    gp.error(f"Subcommand '{ns.cmd}' has no entry point")
    return 2
