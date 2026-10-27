"""Test helper: a CPU kernel for the fa4 engine's FlashAttention-4 operation (deeprm_fa4::varlen), plain attention.

It reads keys and values as FlashAttention-4 4.0.0b35 does on H100 and B200: in whole tiles of TILE rows from each
sequence's start, bounded only by the end of the buffer. Scores past the sequence are masked to -inf (weight 0), but a
NaN or Inf value row in the tile still makes the output NaN (0 x NaN = NaN). Rows after the last packed sequence are NaN
(FA4 leaves them unwritten). So an engine that stops zeroing the padding rows of the attention output gives NaN here.
"""

import torch

from deeprm.model import deeprm_v12_engines  # noqa: F401  (registers the operation)

TILE = 128  # FlashAttention-4's key/value tile in rows (n_block_size) on sm90 and sm100
_ENABLED = False


def _cpu_varlen(q, k, v, cu_q, cu_k, max_q, max_k):
    out = torch.full((q.shape[0], q.shape[1], v.shape[-1]), float("nan"), dtype=q.dtype, device=q.device)
    starts_q, starts_k = cu_q.tolist(), cu_k.tolist()
    for i in range(len(starts_q) - 1):
        a, e, ka, ke = starts_q[i], starts_q[i + 1], starts_k[i], starts_k[i + 1]
        if e == a:
            continue
        if ke == ka:  # a sequence without keys
            out[a:e] = 0
            continue
        kt = min(ka + TILE * -(-(ke - ka) // TILE), k.shape[0])  # the end of the sequence's last tile
        qi, ki, vi = (t.float().transpose(0, 1) for t in (q[a:e], k[ka:kt], v[ka:kt]))  # [H, rows, hd]
        scores = qi @ ki.transpose(1, 2) / qi.shape[-1] ** 0.5  # [H, lq, tile rows]
        inside = torch.arange(kt - ka) < (ke - ka)
        scores = torch.where(inside, scores, torch.tensor(float("-inf")))  # a select, as FA4's mask
        out[a:e] = (torch.softmax(scores, dim=-1) @ vi).transpose(0, 1).to(out.dtype)
    return out


def enable():
    """Registers the CPU stand-in (once per process)."""
    global _ENABLED
    if not _ENABLED:
        torch.library.register_kernel("deeprm_fa4::varlen", "cpu", _cpu_varlen)
        _ENABLED = True
