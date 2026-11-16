"""check_terms.py BINARY PILEUP_SRC [JOBS]: checks the per-read terms of deeprm_pileup against the Python pileup.

For every float32 probability (0.0 to 1.0, and -0.0) and several settings, compares each array of
`deeprm_pileup terms-range` with _per_read_terms() of PILEUP_SRC/deeprm/inference/pileup_deeprm.py, dtype and bytes.
Prints the differences per setting and RESULT IDENTICAL or DIFFERENT; exit 0 when identical."""
import multiprocessing as mp
import sys

import numpy as np

from float_ranges import probability_ranges, run_tool

BINARY, PILEUP_SRC = sys.argv[1], sys.argv[2]
sys.path.insert(0, PILEUP_SRC)
from deeprm.inference import pileup_deeprm  # noqa: E402

BLOCK = 1 << 16
# (name in _per_read_terms, dtype written by the tool, dtype _per_read_terms must return), in the tool's order
FIELDS = (
    ("count_pos", "u1", "int64"),
    ("logsum_1_p_pos", "<f8", "float64"),
    ("kl_div_neg", "<f4", "float32"),
    ("kl_div_pos", "<f4", "float32"),
    ("count_pos_rule", "u1", "int64"),
    ("logsum_1_p_pos_rule", "<f8", "float64"),
    ("logsum_1_p_pos_calib", "<f8", "float64"),
)
# (threshold, epsilon, calibrated): the defaults, uncalibrated inputs, and other option values
SETTINGS = ((0.98, 1e-30, True), (0.98, 1e-30, False), (0.9, 1e-6, True))


def tool_arrays(raw, count):
    """{name: array} from the tool's output: per block of BLOCK rows, the arrays of FIELDS in order."""
    parts = {name: [] for name, _, _ in FIELDS}
    offset = 0
    for start in range(0, count, BLOCK):
        size = min(BLOCK, count - start)
        for name, dtype, _ in FIELDS:
            parts[name].append(np.frombuffer(raw, dtype=dtype, count=size, offset=offset))
            offset += size * np.dtype(dtype).itemsize
    assert offset == len(raw)
    return {name: np.concatenate(arrays) for name, arrays in parts.items()}


def differences(task):
    first, count, (threshold, epsilon, calibrated) = task
    bits = np.arange(first, first + count, dtype=np.uint64).astype(np.uint32)
    ours = tool_arrays(run_tool(BINARY, "terms-range", first, count, threshold, epsilon, int(calibrated)), count)
    theirs = pileup_deeprm._per_read_terms(bits.view(np.float32), threshold, epsilon, calibrated)
    found = {}
    for name, dtype, expected in FIELDS:
        assert theirs[name].dtype == np.dtype(expected), (name, theirs[name].dtype)
        ref = theirs[name].astype(dtype) if dtype == "u1" else theirs[name]
        at = np.flatnonzero(ours[name].view(f"u{ref.itemsize}") != ref.view(f"u{ref.itemsize}"))
        if at.size:
            found[name] = (int(at.size), [(hex(int(bits[i])), ours[name][i].item(), ref[i].item()) for i in at[:3]])
    return count, found


if __name__ == "__main__":
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    ok = True
    with mp.get_context("fork").Pool(jobs) as pool:
        for setting in SETTINGS:
            tasks = [(first, count, setting) for first, count in probability_ranges()]
            total, bad = 0, {}
            for count, found in pool.imap(differences, tasks):
                total += count
                for name, (n, examples) in found.items():
                    seen = bad.setdefault(name, [0, []])
                    seen[0] += n
                    seen[1] = (seen[1] + examples)[:3]
            ok &= not bad
            print(f"threshold {setting[0]} epsilon {setting[1]} calibrated {setting[2]}: values {total}", flush=True)
            for name, (n, examples) in bad.items():
                print(f"  DIFF {name}: {n} values, e.g. (bits, ours, NumPy) {examples}", flush=True)
            if not bad:
                print("  SAME all 7 arrays", flush=True)
    print("RESULT", "IDENTICAL" if ok else "DIFFERENT")
    sys.exit(0 if ok else 1)
