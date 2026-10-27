"""The dwell tokens do not depend on the numpy version.

The dwell tokens of a read are scaled by two quantiles of its float32 log dwell times. numpy before 2.4, the research
code that ran with it, and the C++ binary place these quantiles in single precision; numpy 2.4 and newer place them in
double precision when q is a Python float, which on long reads moves every dwell token of the read (by up to ~4e-4 on
real libraries). Both Python preprocessing paths pass q as float32, which gives the single-precision result with every
numpy version.
"""

import numpy as np
import pytest

from deeprm.inference import inference_preprocess_python as py_prep
from deeprm.train import train_preprocess

f32 = np.float32


def single_precision_quantile(x, q):
    """np.quantile(x, q) for float32 x as numpy before 2.4 computes it (cpp/src/utils/Utils.h, quantile<float>)."""
    s = np.sort(x)
    pos = f32(s.size - 1) * f32(q)
    if pos >= f32(s.size - 1):
        return s[-1]
    lo = int(np.floor(pos))
    a, b = s[lo], s[lo + 1]
    g, d = f32(pos - f32(lo)), f32(b - a)
    return f32(b - d * (f32(1) - g)) if g >= f32(0.5) else f32(a + d * g)


def move_table(dwell):
    """A move table whose dwell times (in strides, in the order move_to_dwell reads them) are `dwell`."""
    flipped = np.concatenate([np.r_[np.zeros(d - 1, bool), True] for d in dwell])
    return np.flip(flipped)


def boundary_read(n, rng):
    """Dwell times of n moves whose 20% and 80% quantiles each fall between two different values, the only case in
    which the interpolation weight, and so its precision, matters (with equal neighbours any weight gives the same)."""
    lo_a = int(np.floor(f32(n - 1) * f32(0.2)))
    lo_b = int(np.floor(f32(n - 1) * f32(0.8)))
    dwell = np.r_[np.full(lo_a + 1, 5), np.full(lo_b - lo_a, 6), np.full(n - lo_b - 1, 8)]
    return rng.permutation(dwell)


@pytest.mark.parametrize(
    "move_to_dwell", [py_prep.move_to_dwell, train_preprocess.move_to_dwell], ids=["call", "train"]
)
def test_dwell_quantiles_are_single_precision(move_to_dwell):
    rng = np.random.default_rng(11)
    reads = [boundary_read(int(n), rng) for n in rng.integers(2000, 30000, 40)]
    reads += [rng.geometric(0.25, int(n)) for n in rng.integers(2000, 30000, 20)]
    for dwell in reads:
        x = np.log10((dwell * 6).astype(f32))
        qa, qb = single_precision_quantile(x, 0.2), single_precision_quantile(x, 0.8)
        expected = (x - max(0.1, 0.5 * (qa + qb))) / max(0.1, 1.5 * (qb - qa))
        got = move_to_dwell(move_table(dwell), 0.2, 0.8, 0.5, 1.5, 6)
        assert got.dtype == np.float32 and np.array_equal(got, expected)
