"""Ranges of float32 bit patterns that cover every probability a per-read file can hold: 0.0 to 1.0, and -0.0."""
import subprocess

ONE = 0x3F800000  # bits of 1.0f
STEP = 1 << 23
NEGATIVE_ZERO = 0x80000000


def probability_ranges():
    """(first bit pattern, count) tasks; every count is a multiple of the tool's block or less than one block."""
    return [(first, STEP) for first in range(0, ONE, STEP)] + [(ONE, 1), (NEGATIVE_ZERO, 1)]


def run_tool(binary, *arguments):
    """stdout of one developer command of deeprm_pileup."""
    return subprocess.run([binary, *map(str, arguments)], check=True, stdout=subprocess.PIPE).stdout
