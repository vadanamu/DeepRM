"""
DeepRM v1.2 m6A model (architecture deeprm_v12): the reference implementation, run as the `eager` engine.

A row is one read at one candidate A: the signal of the 17 bases around it, cut into up to 200 overlapping windows of
30 samples (one every 6). Each window gets a signal embedding plus the embedding of its base's 5-mer; six pre-norm
transformer layers (RoPE, SwiGLU) follow; a head scores each window, and the centre base's window scores are averaged
into one logit. The engines in deeprm.model.deeprm_v12_engines subclass TransformerModel and compute the same
function.

Inputs: src_kmer [B, M] (ASCII codes), src_signal [B, R], src_seg_len [B, L] (windows per base), src_dwell_bq
[B, L, E] or None. Output: [B, C], the logit (or probability) that the centre A is m6A. Base b owns windows
cs[b] .. cs[b + 1] - 1, with cs = [0, cumsum(src_seg_len[r])]; windows from cs[L] on are padding.

Shape symbols used in the docstrings and comments (symbol, v1.2 value, meaning)::

  B      varies   Rows (reads) per call.
  M      21       Bases in src_kmer.
  L      17       Bases with signal; the centre base is L // 2.
  R      1224     Signal samples per row.
  Z      30       Samples per window.
  S      200      Windows per row.
  S_max  2048     Most windows a row may have (the length of the rotary tables).
  T      96-200   Bucket width of the fast engine, in windows.
  E      0        Dwell and base-quality channels.
  D      512      Model width.
  H      8        Attention heads.
  hd     64       Width of one attention head (D / H).
  G      688      Inner width of SwiGLU.
  C      1        Outputs per row.

The engines add sizes of their own, such as W and U_pad; they are listed in deeprm.model.deeprm_v12_engines.
"""

# Editing rule: the compiled engines (deeprm_v12_engines.py) also trace `_effective_seq_len`, `process_signal`,
# `create_src_pad_mask`, `RotaryEmbedding`, `apply_rotary` and the layer modules; changing their operations, order or
# dtypes can change the engines' outputs in the last bits.

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Without max_seq_len, TransformerModel uses S_max = max(seq_len, DEFAULT_MAX_SEQ_LEN).
DEFAULT_MAX_SEQ_LEN = 2048

# The head's activations, by lin_act name. They have no parameters, so the choice does not change the weight
# file's tensor names.
HEAD_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "elu": nn.ELU}


ARCHITECTURE = "deeprm_v12"


class TransformerModel(nn.Module):
    """The deeprm_v12 model (DeepRM v1.2). Reference forward pass, never compiled.

    Numbers in parentheses are the v1.2 values.

    Args:
        d_ff: Feed-forward size before the SwiGLU 2/3 rule (1024); the inner width is G = 688.
        signal_size: Samples per window, Z; must equal signal_stride * kmer_size (30).
        block_len: Bases per row, L (17); the centre base is block_len // 2.
        seq_len: Windows per row in training (200).
        max_seq_len: Most windows a row may have, S_max; default max(seq_len, 2048).
        dwell_bq_dim: Dwell / base-quality channels per window, E (0).
        lin_depth: The head has lin_depth - 1 Linear + BatchNorm blocks before its last three Linear layers (6).
        n_label: Outputs per row, C (1: the logit that the centre A is m6A).
        t_act: Not used (the feed-forward layers are SwiGLU); accepted because model files carry it.
        n_class: Not used (n_label sets the outputs); accepted because model files carry it.
        kmer_mode: Must be "5mer"; accepted because model files carry it.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        encoder_dropout: float = 0.1,
        lin_dropout: float = 0.1,
        kmer_size: int = 5,
        signal_size: int = 25,
        block_len: int = 17,
        seq_len: int = 200,
        max_seq_len: int | None = None,
        lin_act: str = "gelu",
        dwell_bq_dim: int = 3,
        lin_depth: int = 1,
        signal_stride: int = 6,
        n_label: int = 2,
        output_logits: bool = False,
        t_act: str = "gelu",
        n_class: int | None = None,
        kmer_mode: str = "5mer",
    ) -> None:
        super().__init__()
        if kmer_mode != "5mer":
            raise ValueError(f"kmer_mode {kmer_mode!r} is not supported; the v1.2 model uses 5-mer tokens.")
        if max_seq_len is None:
            max_seq_len = max(seq_len, DEFAULT_MAX_SEQ_LEN)
        if max_seq_len < seq_len:
            raise ValueError(f"max_seq_len ({max_seq_len}) must be >= seq_len ({seq_len}).")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

        # [3126, D]: the 5**5 base-5 5-mer ids shifted by 1, so that id 0 marks padding windows. No padding_idx:
        # id 0 has a trained vector, but padding windows are never attended to and never read out.
        self.kmer_embedding = nn.Embedding(5**kmer_size + 1, d_model)
        self.signal_embedding = nn.Linear(signal_size + dwell_bq_dim, d_model)  # [Z + E] -> [D]

        # kmer_radix = [1, 5, 25, 125, 625]: the first base of a 5-mer is the least significant digit.
        ascii_to_base = torch.zeros(256, dtype=torch.long)
        for letters, code in (("Aa", 1), ("Cc", 2), ("Gg", 3), ("TtUu", 4)):
            for letter in letters:
                ascii_to_base[ord(letter)] = code
        self.register_buffer("ascii_to_base", ascii_to_base, persistent=False)
        self.register_buffer("kmer_radix", 5 ** torch.arange(kmer_size, dtype=torch.long), persistent=False)
        self.register_buffer("seq_positions", torch.arange(max_seq_len, dtype=torch.long), persistent=False)

        self.d_model = d_model
        self.transformer_encoder = RoPESwiGLUEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=encoder_dropout,
            max_seq_len=max_seq_len,
        )
        self.regression_head = RegressionHead(
            d_model=d_model,
            lin_act=lin_act,
            lin_depth=lin_depth,
            lin_dropout=lin_dropout,
            n_class=n_label,
        )
        # For multi-GPU training; in eval mode SyncBatchNorm computes exactly what BatchNorm1d does.
        self.regression_head = nn.SyncBatchNorm.convert_sync_batchnorm(self.regression_head)
        self.init_weights()

        self.kmer_size = kmer_size
        self.signal_size = signal_size
        self.signal_stride = signal_stride
        self.seq_len = seq_len
        self.max_seq_len = max_seq_len
        self.block_len = block_len
        self.output_logits = output_logits

    def init_weights(self, initrange: float = 0.1) -> None:
        """Initial weights for training (replaced when a weight file is loaded)."""
        self.kmer_embedding.weight.data.uniform_(-initrange, initrange)
        self.signal_embedding.weight.data.uniform_(-initrange, initrange)
        self.regression_head.init_weights(initrange)

    def _effective_seq_len(self, src_signal: Tensor) -> int:
        """Number of signal windows, S = (R - Z) // 6 + 1. Also traced by the compiled engines."""
        n_samples = src_signal.shape[1]
        window = self.signal_stride * self.kmer_size
        if n_samples < window:
            raise ValueError(
                f"signal_token width {n_samples} is smaller than the unfold window ({window} = "
                f"signal_stride*kmer_size)."
            )
        return (n_samples - window) // self.signal_stride + 1

    def process_kmer(self, src_kmer: Tensor, src_seg_len_flat: Tensor, eff_seq_len: int) -> Tensor:
        """The 5-mer id of every signal window, [B, S] int32; padding windows get id 0.

        src_kmer is [B, M] ASCII codes; src_seg_len_flat is [B * (L + 1)] from flatten_seg_len.
        """
        batch = src_kmer.shape[0]
        src_kmer = src_kmer.long().clamp(0, 255)  # out of place: an int64 input must not be changed for the caller
        src_kmer = self.ascii_to_base[src_kmer]  # [B, M] base codes 0..4
        # [B, M] -> [B, L, 5]: 5-mer i is bases i .. i + 4 of M. M has 2 extra bases at each end, so 5-mer i is
        # centred on base i of L.
        src_kmer = src_kmer.unfold(1, self.kmer_size, 1)
        # Base-5 number: id = 1 + sum_j code[i + j] * 5**j; the + 1 keeps id 0 for padding.
        src_kmer = src_kmer * self.kmer_radix.unsqueeze(0).unsqueeze(0)  # [B, L, 5] * [1, 1, 5]
        src_kmer = src_kmer.sum(dim=-1) + 1  # [B, L] int64
        # [B, L] -> [B, L + 1]: id 0 for the padding segment.
        src_kmer = torch.cat([src_kmer, torch.zeros(batch, 1, device=src_kmer.device, dtype=torch.long)], dim=1)
        # Repeat each id once per window of its segment. Each row's L + 1 lengths sum to S, so the reshape is exact.
        src_kmer = src_kmer.flatten()  # [B * (L + 1)]
        src_kmer = src_kmer.repeat_interleave(src_seg_len_flat)  # [B * S]
        src_kmer = src_kmer.reshape(batch, eff_seq_len)  # [B, S]
        return src_kmer.int()

    def process_signal(self, src_signal: Tensor) -> Tensor:
        """The overlapping signal windows, [B, R] -> [B, S, Z] (a view)."""
        return src_signal.unfold(1, self.signal_stride * self.kmer_size, self.signal_stride)

    def flatten_seg_len(self, src_seg_len: Tensor, eff_seq_len: int) -> Tensor:
        """Each row's segment lengths plus one padding segment, flattened: [B, L] int32 -> [B * (L + 1)] int64.

        The padding segment has S - src_seg_len[r].sum() windows, so each row's L + 1 lengths sum to S.
        """
        per_row_sum = src_seg_len.sum(dim=1, keepdims=True)  # [B, 1]: real windows per row
        if torch.any(per_row_sum > eff_seq_len):
            raise ValueError(
                f"Some rows have segment_len_arr summing > eff_seq_len={eff_seq_len}.  Either reduce input dwell or "
                f"feed a wider signal_token (more samples) so eff_seq_len grows."
            )
        # [B, L] and [B, 1] -> [B, L + 1]; cat promotes the int32 lengths to int64.
        src_seg_len_flat = torch.cat([src_seg_len, eff_seq_len - per_row_sum], dim=1)
        return src_seg_len_flat.flatten()

    def create_src_pad_mask(self, src_seg_len: Tensor, eff_seq_len: int) -> Tensor:
        """True for the padding windows, [B, S] bool."""
        positions = self.seq_positions[:eff_seq_len]  # [S]
        return positions.unsqueeze(0) >= src_seg_len.sum(dim=1, keepdim=True)  # [1, S] >= [B, 1] -> [B, S]

    def create_target_mask(self, src_seg_len: Tensor, src_seg_len_flat: Tensor, eff_seq_len: int) -> Tensor:
        """1 for the windows of the centre base, else 0, [B, S] int32.

        Row r is 1 for windows cs[c] .. cs[c + 1] - 1, with c = L // 2 (8 in v1.2) and cs as in the module docstring, so
        its sum is src_seg_len[r, c].
        """
        batch = src_seg_len.shape[0]
        width = src_seg_len.shape[1]  # L
        # Segment ids 0 .. L (L is the padding segment); True only for the centre base.
        target_mask = torch.arange(width + 1, device=src_seg_len.device, dtype=torch.int)
        target_mask = target_mask == self.block_len // 2  # [L + 1] bool
        target_mask = target_mask.repeat(batch)  # [B * (L + 1)]
        target_mask = target_mask.repeat_interleave(src_seg_len_flat)  # [B * S]
        target_mask = target_mask.reshape(batch, eff_seq_len)  # [B, S]
        return target_mask.int()

    def process_dwell_bq(self, src_dwell_bq: Tensor, src_seg_len_flat: Tensor, eff_seq_len: int) -> Tensor:
        """Each base's dwell / base-quality values repeated over its windows, [B, L, E] -> [B, S, E].

        Not used by the v1.2 weights (E = 0).
        """
        batch = src_dwell_bq.shape[0]
        channel = src_dwell_bq.shape[2]  # E
        # [B, L, E] -> [B, L + 1, E]: zeros for the padding segment.
        src_dwell_bq = torch.cat(
            [src_dwell_bq, torch.zeros(batch, 1, channel, device=src_dwell_bq.device, dtype=torch.float32)], dim=1
        )
        src_dwell_bq = src_dwell_bq.flatten(end_dim=1)  # [B * (L + 1), E]
        src_dwell_bq = src_dwell_bq.repeat_interleave(src_seg_len_flat, dim=0)  # [B * S, E]
        return src_dwell_bq.reshape(batch, eff_seq_len, channel)

    def forward(
        self,
        src_kmer: Tensor,
        src_signal: Tensor,
        src_seg_len: Tensor,
        src_dwell_bq: Tensor | None = None,
    ) -> Tensor:
        """Mean of the head's output over the centre base's windows, [B, C] float32; its sigmoid if not output_logits.

        A row whose centre base has no windows gives NaN (0 / 0). Rows do not interact in eval mode, but on a GPU
        the rounding can change with the batch size.
        """
        eff_seq_len = self._effective_seq_len(src_signal)  # S
        if eff_seq_len > self.max_seq_len:
            raise ValueError(
                f"Input would yield eff_seq_len={eff_seq_len}, but max_seq_len={self.max_seq_len}.  Rebuild with a "
                f"larger max_seq_len, or feed a narrower signal_token."
            )

        # Index arithmetic on the inputs; no trainable weights.
        with torch.no_grad():
            src_seg_len_flat = self.flatten_seg_len(src_seg_len, eff_seq_len)  # [B * (L + 1)] int64
            src_kmer = self.process_kmer(src_kmer, src_seg_len_flat, eff_seq_len)  # [B, S] int32
            src_signal = self.process_signal(src_signal)  # [B, S, Z]
            if src_dwell_bq is not None:
                src_dwell_bq = self.process_dwell_bq(src_dwell_bq, src_seg_len_flat, eff_seq_len)  # [B, S, E]
            src_pad_mask = self.create_src_pad_mask(src_seg_len, eff_seq_len)  # [B, S] bool, True = padding
            target_mask = self.create_target_mask(src_seg_len, src_seg_len_flat, eff_seq_len)  # [B, S] int32

        if src_dwell_bq is not None:
            src_signal = torch.cat([src_signal, src_dwell_bq], dim=-1)  # [B, S, Z + E]

        kmer_embedding = self.kmer_embedding(src_kmer)  # [B, S, D]
        signal_embedding = self.signal_embedding(src_signal)  # [B, S, D]
        x = kmer_embedding + signal_embedding  # no position embedding here: the layers rotate Q and K
        x = self.transformer_encoder(x, src_key_padding_mask=src_pad_mask)  # [B, S, D]
        # [B, S, D] -> [B, S, C] for every window, padding included. In eval mode the head acts on each window alone,
        # so the faster engines run it, and the last layer's queries, only on or near the centre base's windows.
        output = self.regression_head(x)

        # Readout: masked sum over the windows, divided by the number of centre windows, src_seg_len[:, L // 2].
        target_mask = target_mask.unsqueeze(-1)  # [B, S, 1] int32
        target_mask_sum = target_mask.sum(dim=1)  # [B, 1] int64
        output = output * target_mask  # [B, S, C]; float16 on a GPU under autocast
        output = output.sum(dim=1)  # [B, C] float32: autocast runs sum in float32
        output = output / target_mask_sum  # [B, C] / [B, 1] -> [B, C]
        if not self.output_logits:
            output = torch.sigmoid(output)
        return output


class RotaryEmbedding(nn.Module):
    """Rotary position embedding: cos / sin tables, [S_max, hd / 2] float32.

    Entry [p, i] is cos or sin of p * theta_i, with theta_i = base ** (-2i / hd) and base = 10000 (not a nucleotide).
    The tables are not in weight files; the compiled engines read them in their graphs.
    """

    def __init__(self, head_dim: int, max_seq_len: int = DEFAULT_MAX_SEQ_LEN, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}.")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))  # [hd / 2]: theta_i
        positions = torch.arange(max_seq_len).float()  # [S_max]
        freqs = torch.outer(positions, inv_freq)  # [S_max, hd / 2]
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, seq_len: int):
        """The first seq_len rows of both tables, each [S, hd / 2]."""
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotates each channel pair (2i, 2i + 1) of Q or K by its position's angle. Also traced by the compiled engines.

    x [B, H, S, hd], cos and sin [S, hd / 2] -> [B, H, S, hd] in x's dtype. The pairs are neighbouring channels, not
    the first and second half of the head.
    """
    x1 = x[..., 0::2]  # [B, H, S, hd / 2]: even channels
    x2 = x[..., 1::2]  # [B, H, S, hd / 2]: odd channels
    cos = cos.unsqueeze(0).unsqueeze(0).to(x.dtype)  # [1, 1, S, hd / 2]
    sin = sin.unsqueeze(0).unsqueeze(0).to(x.dtype)
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)  # [B, H, S, hd / 2, 2]
    return rotated.flatten(-2)  # [B, H, S, hd]: the pairs interleave back into channels 2i, 2i + 1


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward layer."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        d_inner = (2 * d_ff) // 3  # 2 * 1024 // 3 = 682
        d_inner = ((d_inner + 7) // 8) * 8  # G = 688
        self.w1 = nn.Linear(d_model, d_inner, bias=False)  # gate
        self.w2 = nn.Linear(d_inner, d_model, bias=False)  # output
        self.w3 = nn.Linear(d_model, d_inner, bias=False)  # value
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))  # [B, S, D] -> [B, S, G] -> [B, S, D]


class RoPESwiGLUEncoderLayer(nn.Module):
    """Pre-norm transformer layer: self-attention with rotary Q and K, then a SwiGLU feed-forward layer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        rope: RotaryEmbedding | None = None,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}.")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self.dropout_p = dropout

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Q, K and V in one matrix, weight [3D, D]: output channel j * D + h * hd + c is channel c of head h of Q
        # (j = 0), K (j = 1) or V (j = 2).
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.attn_resid_dropout = nn.Dropout(dropout)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout=dropout)
        self.ffn_resid_dropout = nn.Dropout(dropout)
        self.rope = rope  # shared by all layers

    def forward(self, x: Tensor, src_key_padding_mask: Tensor | None = None) -> Tensor:
        """One layer, [B, S, D] -> [B, S, D]. src_key_padding_mask is [B, S] bool, True = padding.

        The fast engine repeats these steps in its own layer function.
        """
        B, S, D = x.shape

        # Attention block.
        normed = self.norm1(x)  # [B, S, D]
        qkv = self.qkv(normed).reshape(B, S, 3, self.n_heads, self.head_dim)  # [B, S, 3D] -> [B, S, 3, H, hd]
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, hd]
        q, k, v = qkv.unbind(0)  # each [B, H, S, hd]
        cos, sin = self.rope(S)  # each [S, hd / 2]
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # Additive key mask [B, 1, 1, S]: -inf on padding keys. Padding windows still act as queries; the readout
        # gives them weight 0. In a row with no real window every key is masked; that row is NaN anyway (0 / 0).
        if src_key_padding_mask is not None:
            attn_mask = torch.zeros((B, 1, 1, S), device=x.device, dtype=q.dtype)
            attn_mask.masked_fill_(src_key_padding_mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        else:
            attn_mask = None
        # [B, H, S, hd], default scale 1 / sqrt(hd).
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p if self.training else 0.0, is_causal=False
        )
        attn = attn.transpose(1, 2).reshape(B, S, D)  # [B, S, H, hd] -> [B, S, D], channel h * hd + c
        attn = self.attn_out(attn)  # [B, S, D]
        x = x + self.attn_resid_dropout(attn)

        # Feed-forward block.
        x = x + self.ffn_resid_dropout(self.ffn(self.norm2(x)))
        return x


class RoPESwiGLUEncoder(nn.Module):
    """The transformer layers, sharing one rotary table, then a final LayerNorm: [B, S, D] -> [B, S, D]."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float = 0.1,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    ) -> None:
        super().__init__()
        head_dim = d_model // n_heads  # hd
        self.rope = RotaryEmbedding(head_dim, max_seq_len)  # tables [S_max, hd / 2]
        self.layers = nn.ModuleList(
            [RoPESwiGLUEncoderLayer(d_model, n_heads, d_ff, dropout, rope=self.rope) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, src_key_padding_mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)  # [B, S, D]
        return self.final_norm(x)  # [B, S, D]


class ChannelBN(nn.Module):
    """BatchNorm over the channels of every token: [B, S, D] is normalised as [B * S, D].

    In eval mode it acts on each token alone; in training, padding windows count in the batch statistics.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, x: Tensor) -> Tensor:
        B, S, D = x.shape
        return self.bn(x.reshape(B * S, D)).reshape(B, S, D)  # [B, S, D] -> [B * S, D] -> [B, S, D]


class RegressionHead(nn.Module):
    """The head that scores every token on its own (in eval mode): [B, S, D] -> [B, S, C].

    The Dropout modules do nothing in eval mode but keep their places, because the layer indices are part of the
    weight file's tensor names.
    """

    def __init__(self, d_model: int, lin_act: str, lin_depth: int, lin_dropout: float, n_class: int) -> None:
        super().__init__()
        if lin_act not in HEAD_ACTIVATIONS:
            raise ValueError(f"lin_act {lin_act!r} is not supported; use one of: {', '.join(HEAD_ACTIVATIONS)}.")
        act = HEAD_ACTIVATIONS[lin_act]
        layer_list = []
        for _ in range(lin_depth - 1):
            layer_list.append(nn.Linear(d_model, d_model))
            layer_list.append(ChannelBN(d_model))
            layer_list.append(act())
            layer_list.append(nn.Dropout(lin_dropout))
        layer_list.append(nn.Linear(d_model, d_model))
        layer_list.append(act())
        layer_list.append(nn.Linear(d_model, d_model // 4))
        layer_list.append(act())
        layer_list.append(nn.Linear(d_model // 4, n_class))
        self.lin_layers = nn.Sequential(*layer_list)

    def forward(self, x: Tensor) -> Tensor:
        return self.lin_layers(x)  # [B, S, D] -> [B, S, C]

    def init_weights(self, initrange: float = 0.1) -> None:
        for layer in self.lin_layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                if layer.bias is not None:
                    layer.bias.data.zero_()
