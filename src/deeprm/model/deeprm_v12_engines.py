"""
Faster engines for the deeprm_v12 model, and the choice between them. The reference model, deeprm.model.deeprm_v12,
runs as the `eager` engine.

* `fast` (FastEngine): rows go into length buckets; one compiled graph per bucket width plus one for the head,
  PyTorch attention. Compute capability 7.0+; uncompiled on CPU.
* `fa4` (Fa4Engine): the real tokens of a batch are packed into one dimension; one compiled graph with
  FlashAttention-4 variable-length attention. Compute capability 8.0+ and FlashAttention-4 (installed with DeepRM on
  Linux x86_64).

"auto" picks fa4 where it can run and was tested (an NVIDIA GPU of compute capability 8.0, 9.0 or 10.0, the verified
FlashAttention-4 build, a PyTorch of CUDA 12.8 or newer), else fast, else eager. fa4 skips each row's padding windows
(it pads only the batch total, to whole blocks); fast skips most of them through the length buckets. In the last layer
fa4 computes queries only for the centre tokens, and fast for W window positions per pass from the centre base's first
window. Both run the head only on a buffer of centre tokens, merge SwiGLU's w1 and w3, and fold the head BatchNorms;
results equal the reference up to floating-point rounding. Sizes that change per batch are dynamic and never 1; bucket
width and buffer size are static, which keeps the graph count small (a new buffer size U_pad compiles one more graph).

deeprm_run.json records the form of the engine that ran (engine_used, see engine_form); a run continued with
``--resume`` must run the same form.

Shape symbols ([B, C] and so on) are those of deeprm.model.deeprm_v12. The engines add these (symbol, v1.2 value,
meaning)::

  W      24       Centre-token positions per pass of the fast engine's last layer.
  N      varies   Real tokens in the batch (fa4).
  U      varies   Real centre tokens in the batch.
  U_pad  varies   Size of the centre-token buffer (see set_batch_size).
"""

# Editing rule: functions marked "Compiled", and what they call, are traced by torch.compile. Changing their
# operations, order, dtypes, shape marks or the size constants below changes outputs in the last bits. Some values
# they compute are never read: lens, n_in_window and in_window in FastEngine._bucket_graph, and the second outputs of
# _bucket_graph and Fa4Engine._packed_graph. They are kept so that the code stays the code that was checked against
# the reference; FORM (recorded in deeprm_run.json) names that code.
#
# Shape symbols that are not in the module docstring (symbol, v1.2 value, meaning). W is CENTRE_WINDOW, and U_pad
# ("capacity") is set by centre_slot_block.
#   Tq     varies   Queries per row in FastEngine._attention: T in the full layers, W in the last layer.
#   N_pad  varies   The fa4 token dimension: N rounded up to whole TOKEN_GRANULE blocks, at least MIN_TOKEN_BLOCKS.

import hashlib
import importlib
import importlib.util
import os
import platform
import shutil
import site
import sys
import sysconfig
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from deeprm.model.deeprm_v12 import ARCHITECTURE, ChannelBN, RoPESwiGLUEncoderLayer, TransformerModel, apply_rotary
from deeprm.utils import check_deps
from deeprm.utils.logging import get_logger

log = get_logger(__name__)

ENGINES = ("auto", "fa4", "fast", "eager")
FA4_CC_MAJORS = (8, 9, 10, 11, 12)  # GPU generations FlashAttention-4 4.0.0b35 has kernels for (A100 to B200, RTX 50)
# The GPUs auto chooses fa4 on: those it was tested on (A100, H100, B200). Others of FA4_CC_MAJORS run it on request.
FA4_AUTO_CCS = ((8, 0), (9, 0), (10, 0))
COMPILE_MIN_CC = 7  # torch.compile (Triton) needs compute capability 7.0 or higher

# The verified FlashAttention-4 build.
FA4_DIST = "flash-attn-4"
FA4_PIN_VERSION = "4.0.0b35"
FA4_PIN_INTERFACE_MD5 = "4470f6405b66cb9e34d45ad7555cad52"  # md5 of flash_attn/cute/interface.py
FA4_DTYPES = (torch.float16, torch.bfloat16)
FA4_SITE_ENV = "DEEPRM_FA4_SITE"
FA4_UNPINNED_ENV = "DEEPRM_FA4_UNPINNED_OK"
# fa4 is chosen with a PyTorch build of CUDA 12.8 or newer: tested with 2.10.0+cu130 (B200, H100) and 2.10.0+cu128
# (A100, B200). DeepRM installs FlashAttention-4 on Linux x86_64 with these pins (pyproject.toml lists the same, with a
# platform marker); the cu13 form installs CUTLASS's CUDA 12 and CUDA 13 libraries, and CUTLASS loads the one the
# driver supports.
FA4_REQUIREMENTS = (
    "flash-attn-4[cu13]==4.0.0b35",
    "nvidia-cutlass-dsl[cu13]==4.8.0",
    "quack-kernels==0.6.5",
    "apache-tvm-ffi==0.1.14.post1",
    "torch-c-dlpack-ext==0.1.5",
    "einops==0.8.2",
)
FA4_TORCH_CUDA = (12, 8)  # the oldest PyTorch CUDA build fa4 is chosen with
FA4_TORCH_CUDA_MAJOR = 13  # a build of this CUDA major or newer is the cu130 route (see _fa4_route)
FA4_DRIVER_CUDA = (13, 0)  # a driver for CUDA 13.0 (580 or newer) runs the CUDA 13 build of PyTorch
FA4_WARNING_RULE = "=" * 72  # frame of the fa4_warning text

COMPILE_LOGS_ENV = "DEEPRM_COMPILE_LOGS"
# torch's default of 8 graphs per function is too low for a long run: each bucket width and each new buffer size
# U_pad adds a graph. Reaching the limit stops the run (see _arm) instead of falling back to uncompiled code.
RECOMPILE_LIMIT = 128

# Fast engine sizes; they set the shapes the compiled graphs see (see the editing rule).
# T: bucket widths in signal windows. A row goes to the smallest width that holds all its real windows.
BUCKET_WIDTHS = (96, 128, 152, 200)  # the fastest set in a timing model of the benchmark inputs
CENTRE_WINDOW = 24  # W; a longer centre base takes extra passes
# Bucket row counts are padded to a multiple of batch // 64 (8 to 128), so attention sees few shapes.
ROW_GRANULE = 128
ROW_GRANULE_MIN = 8
EXTRA_PASS_GRANULE = 16  # row counts of extra passes are padded to a multiple of this, for the same reason


def _compile_logs_on():
    return os.environ.get(COMPILE_LOGS_ENV, "") == "1"


def _file_md5(path):
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except OSError as exc:
        return f"UNREADABLE({type(exc).__name__})"


def row_granule(batch):
    """Bucket row counts are padded to a multiple of this: batch // 64, clamped to 8 .. 128."""
    return max(ROW_GRANULE_MIN, min(ROW_GRANULE, int(batch) // 64))


def centre_slot_block(batch):
    """Centre-token buffer block for `batch` rows: 8,192 slots per 1,000 rows, in multiples of 256.

    U_pad is one block, or a whole multiple of it for a batch with more centre tokens: 131,072 slots at the default
    ``--batch`` of 16,000 rows. The benchmark inputs held about 7.2 centre tokens per row (71,762 in 10,000 rows).
    """
    # -(-a // b) is ceil(a / b): blocks of 256 slots, at least one
    return max(256, -(-int(batch) * 8192 // (1000 * 256)) * 256)


def bucket_widths_for(n_windows):
    """The bucket widths for `n_windows` signal windows; the top bucket is the input's own width."""
    # e.g. S = 200 -> (96, 128, 152, 200); S = 150 -> (96, 128, 150); S = 96 -> (96,)
    return tuple(w for w in BUCKET_WIDTHS if w < n_windows) + (int(n_windows),)


def compile_kwargs(backend=None):
    """The keyword arguments of every torch.compile call in this file."""
    backend = backend or "inductor"
    # dynamic=None is torch's default; the engines do not rely on its guess but mark the sizes that matter before every
    # call. fullgraph=True: a graph break is an error, not a silent split. mode='default': no CUDA graphs, no autotune.
    kwargs = dict(backend=backend, dynamic=None, fullgraph=True)
    if backend == "inductor":
        kwargs["mode"] = "default"
    return kwargs


def _compile_errors():
    """The exceptions torch.compile raises when it cannot compile (not out-of-memory: the caller handles that)."""
    import torch._dynamo.exc
    import torch._inductor.exc

    return (torch._dynamo.exc.BackendCompilerFailed, torch._inductor.exc.InductorError)


def dynamo_counts():
    """(compiled graphs, graph breaks) of this process so far."""
    from torch._dynamo.utils import counters

    return int(counters["stats"]["unique_graphs"]), int(sum(counters["graph_break"].values()))


def _is_noexec(path):
    """True if the nearest existing folder of `path` is on a mount that does not allow running files."""
    p = os.path.abspath(path)
    while not os.path.exists(p):
        p = os.path.dirname(p)
    return bool(os.statvfs(p).f_flag & os.ST_NOEXEC)


def _plan_compile_cache_dirs():
    """(inductor cache folder, Triton cache folder or None for torch's default), avoiding noexec mounts.

    Compiled kernels cannot load from a noexec mount (such as /tmp on some systems), so such a folder is replaced.
    """
    from torch._inductor.runtime.cache_dir_utils import default_cache_dir

    inductor_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR") or default_cache_dir()
    if _is_noexec(inductor_dir):
        import getpass

        inductor_dir = os.path.join(os.path.expanduser("~"), ".cache", f"torchinductor_deeprm_{getpass.getuser()}")
    triton_dir = os.environ.get("TRITON_CACHE_DIR")
    if triton_dir and _is_noexec(triton_dir):
        triton_dir = os.path.join(inductor_dir, "triton", "env_override")
    return inductor_dir, triton_dir


def _arg(args, kwargs, index, name):
    return args[index] if len(args) > index else kwargs.get(name)


def _first_tensor(args, kwargs):
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, torch.Tensor):
            return a
    return None


# ================================================================================================ FlashAttention-4
# FlashAttention-4 (PyPI flash-attn-4, import flash_attn.cute) has no PyTorch operation, so its varlen call is wrapped
# in a custom one: torch.compile records one opaque node and never traces FA4's Python code. FA4 is imported only when
# the fa4 engine is armed.

# Filled once by fa4_preflight.
_FA4 = {"varlen_fn": None, "ready": None}


def _flash_attn_dirs():
    """Folders of the flash_attn package Python would import, or [] if there is none."""
    try:
        spec = importlib.util.find_spec("flash_attn")
    except (ImportError, ValueError):
        return []
    if spec is None:
        return []
    return [str(p) for p in (spec.submodule_search_locations or [])] or [str(spec.origin)]


def _make_fa4_importable():
    """If flash_attn is not importable, adds the DEEPRM_FA4_SITE folder (with its .pth files) to sys.path."""
    if _flash_attn_dirs():
        return
    folder = os.environ.get(FA4_SITE_ENV, "").strip()
    if not folder or not os.path.isfile(os.path.join(folder, "flash_attn", "cute", "interface.py")):
        raise RuntimeError(
            f"FlashAttention-4 is not installed (`{fa4_install_command()}`, or set {FA4_SITE_ENV} to a folder that "
            "holds it). Or run with --engine fast."
        )
    before = list(sys.path)
    site.addsitedir(folder)
    added = [p for p in sys.path if p not in before]
    rest = [p for p in sys.path if p not in added]
    # before the environment's own site-packages, as if FA4 were installed there
    anchor = next((i for i, p in enumerate(rest) if p.rstrip("/").endswith("site-packages")), len(rest))
    sys.path[:] = rest[:anchor] + added + rest[anchor:]
    importlib.invalidate_caches()
    if not _flash_attn_dirs():
        raise RuntimeError(f"Added {folder} to sys.path, but flash_attn is still not importable.")


def _fa4_check():
    """Looks for a usable FlashAttention-4 build without importing it. Returns (code, detail); see fa4_status."""
    dirs = _flash_attn_dirs()
    folder = ""
    if not dirs:
        folder = os.environ.get(FA4_SITE_ENV, "").strip()
        if not folder:
            return "missing", None
        dirs = [os.path.join(folder, "flash_attn")]
    candidates = [os.path.join(d, "cute", "interface.py") for d in dirs]
    interface = next((path for path in candidates if os.path.isfile(path)), None)
    if interface is None:
        return ("site", folder) if folder else ("other", ", ".join(dirs))
    if _file_md5(interface) != FA4_PIN_INTERFACE_MD5 and os.environ.get(FA4_UNPINNED_ENV, "") != "1":
        return "unpinned", interface
    return "ok", interface


def fa4_status():
    """Checks, without importing it, that a usable FlashAttention-4 build can be found.

    Returns:
        tuple: (True, path of flash_attn/cute/interface.py) or (False, reason).
    """
    code, detail = _fa4_check()
    if code == "ok":
        return True, detail
    missing = f"`{fa4_install_command()}`" if fa4_platform() else "DeepRM installs it on Linux x86_64 only"
    return False, {
        "missing": f"FlashAttention-4 not installed ({missing})",
        "site": f"{FA4_SITE_ENV}={detail} holds no FlashAttention-4 (no flash_attn/cute/interface.py in it)",
        "other": "the importable flash_attn package is not FlashAttention-4 (no flash_attn/cute)",
        "unpinned": (
            f"FlashAttention-4 is not the verified build (`{fa4_install_command()}`, or set {FA4_UNPINNED_ENV}=1)"
        ),
    }[code]


def fa4_imports():
    """Checks that FlashAttention-4 and its dependencies import (also from DEEPRM_FA4_SITE). Returns (ok, reason)."""
    try:
        _make_fa4_importable()
        importlib.import_module("flash_attn.cute.interface")
    except Exception as exc:  # noqa: BLE001 -- auto falls back; an explicit --engine fa4 fails loudly later
        return False, f"FlashAttention-4 does not import ({_error_text(exc)})"
    return True, "ok"


def _error_text(exc):
    """'Type: first line of the message' of an exception, the message cut to 200 characters."""
    lines = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {lines[0][:200]}" if lines else type(exc).__name__


def fa4_installed() -> bool:
    """True if a usable FlashAttention-4 build can be found (see fa4_status)."""
    return fa4_status()[0]


def fa4_preflight(device):
    """Imports FlashAttention-4 and checks that it can run on `device`. Raises RuntimeError with the reason."""
    if device.type != "cuda":
        raise RuntimeError("FlashAttention-4 runs on a CUDA GPU only.")
    if is_rocm():
        raise RuntimeError(
            "FlashAttention-4 runs on NVIDIA GPUs only; this PyTorch is a ROCm build. Use --engine fast."
        )
    capability = torch.cuda.get_device_capability(device)
    if capability[0] not in FA4_CC_MAJORS:
        raise RuntimeError(
            f"FlashAttention-4 has no kernels for {torch.cuda.get_device_name(device)} (compute capability "
            f"{capability[0]}.{capability[1]}; it needs 8.0 or newer). Use --engine fast."
        )
    if _FA4["ready"] is not None:
        return _FA4["ready"]
    _make_fa4_importable()
    try:
        import importlib.metadata

        import flash_attn.cute.interface as fa4_interface
        from flash_attn.cute import flash_attn_varlen_func
    except Exception as exc:  # noqa: BLE001 -- any import failure means FA4 cannot run here
        raise RuntimeError(
            f"FlashAttention-4 cannot be imported ({type(exc).__name__}: {exc}). Install it with "
            f"`{fa4_install_command()}`, or run with --engine fast."
        ) from exc
    try:
        version = importlib.metadata.version(FA4_DIST)
    except Exception as exc:  # noqa: BLE001 -- only reported
        version = f"unknown ({type(exc).__name__})"
    interface_md5 = _file_md5(fa4_interface.__file__)
    if interface_md5 != FA4_PIN_INTERFACE_MD5 and os.environ.get(FA4_UNPINNED_ENV, "") != "1":
        raise RuntimeError(
            f"FlashAttention-4 {version} is not the verified build {FA4_PIN_VERSION} (md5 of {fa4_interface.__file__}: "
            f"{interface_md5}). Install it with `{fa4_install_command()}`, or set {FA4_UNPINNED_ENV}=1."
        )
    _FA4["varlen_fn"] = flash_attn_varlen_func
    _FA4["ready"] = dict(version=version, interface_md5=interface_md5, pinned=interface_md5 == FA4_PIN_INTERFACE_MD5)
    return _FA4["ready"]


# flash_attn_varlen_func returns the output, or a tuple whose first item is the output.
def _first(result):
    return result[0] if isinstance(result, (tuple, list)) else result


# No input is written. Only the CUDA kernel is defined; a CPU kernel for testing can be added with
# fa4_varlen.register_kernel("cpu").
@torch.library.custom_op("deeprm_fa4::varlen", mutates_args=(), device_types="cuda")
def fa4_varlen(q: Tensor, k: Tensor, v: Tensor, cu_q: Tensor, cu_k: Tensor, max_q: int, max_k: int) -> Tensor:
    """FlashAttention-4 attention over packed sequences. Compiled (as one opaque node).

    q [Nq, H, hd] (Nq = N_pad, or U_pad in the last layer); k, v [N_pad, H, hd]; cu_q, cu_k [B + 1] int32 offsets.
    Sequence i's q rows cu_q[i] .. cu_q[i + 1] - 1 attend to its k, v rows cu_k[i] .. cu_k[i + 1] - 1; padding is in
    no sequence, so no mask. Returns [Nq, H, hd] in q's dtype, contiguous; rows cu_q[-1] .. Nq - 1 are left unwritten.
    """
    # for a direct call; the engine runs fa4_preflight when it is armed
    if _FA4["varlen_fn"] is None:
        fa4_preflight(q.device)
    out = _first(
        _FA4["varlen_fn"](
            q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=max_q, max_seqlen_k=max_k, causal=False
        )
    )
    # must match the fake below (a new contiguous tensor): the compiler plans later operations from the fake's strides
    return out.contiguous()


# The fake, run on shape-only tensors while tracing. It must match the real output: [rows of q, heads of q, hd of v],
# q's dtype, contiguous. It reads only shapes, so a symbolic token count passes through.
@fa4_varlen.register_fake
def _fa4_varlen_shape(q, k, v, cu_q, cu_k, max_q, max_k):
    return q.new_empty((q.shape[0], q.shape[1], v.shape[-1]))


# ================================================================================================ shared engine base


class _EngineState:
    """Per-call counters, kept off the model: setting a model attribute on every call slows torch.compile's guards."""

    __slots__ = ("calls", "graphs_seen", "max_batch")

    def __init__(self):
        self.calls, self.graphs_seen, self.max_batch = 0, 0, 0


class _CompiledEngine(TransformerModel):
    """What the fast and fa4 engines share: compiling, fused SwiGLU, the folded head, the readout. Inference only.

    The first call with a CUDA tensor arms the engine: it wraps the graph functions in torch.compile (see _arm).
    """

    ENGINE = None  # "fast" or "fa4"
    FORM = None  # form id: the exact code this engine runs
    NEEDS_FA4 = False
    # True if _forward_batch takes `counts` (real tokens, centre tokens, longest row, any negative length) summed
    # on the host by the caller, which replaces the engine's device-to-host read of them (see inference.py).
    ACCEPTS_COUNTS = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._armed = False
        self._state = _EngineState()

    def set_batch_size(self, batch: int) -> None:
        """Sizes the centre-token buffer U_pad for batches of up to `batch` rows from the first call on.

        U_pad is a static size of the compiled graph. Without this call it follows the largest batch so far, so a run
        whose first batches are small compiles again when a larger batch comes.
        """
        self._state.max_batch = max(self._state.max_batch, int(batch))

    def compile_engine(self, device, backend=None):
        """Compiles the engine now; otherwise the first CUDA call compiles it.

        Args:
            backend: torch.compile backend; None means inductor. "eager" (no code generation) runs on the CPU too.
        """
        self._arm(torch.device(device), backend=backend)

    def forward(self, src_kmer, src_signal, src_seg_len, src_dwell_bq=None, counts=None):
        """Mean of the head's output over the centre base's windows, [B, C]; its sigmoid if not output_logits.

        Inference only (eval mode), with no dwell / base-quality input. The output has the head's dtype: fp16 under
        CUDA autocast (as deeprm call run runs it), float32 otherwise. A row whose centre base has no windows gives NaN.
        counts: optional (real tokens, centre tokens, longest row, any negative length), summed on the host by the
        caller (see deeprm.inference.inference.batch_counts). The fa4 engine uses them; the fast engine ignores them
        and reads the sizes it needs from the device.
        """
        if self.training:
            raise RuntimeError(f"The {self.ENGINE} engine is for inference only; call .eval() first.")
        if src_dwell_bq is not None:
            raise ValueError(f"The {self.ENGINE} engine takes no dwell / base-quality input; use the eager engine.")
        self._check_model(src_seg_len)
        # max_batch only grows; it sets U_pad (see set_batch_size)
        self._state.max_batch = max(self._state.max_batch, src_seg_len.shape[0])
        if counts is not None and not self.ACCEPTS_COUNTS:
            counts = None  # the engine reads the counts from the device itself
        return self._forward_batch(src_kmer, src_signal, src_seg_len, counts)

    def _forward_batch(self, src_kmer, src_signal, src_seg_len, counts=None):
        raise NotImplementedError

    def _compile(self, kwargs):
        raise NotImplementedError

    def _check_model(self, src_seg_len=None):
        """Refuses a model these engines were not written for."""
        problems = []
        if not all(isinstance(layer, RoPESwiGLUEncoderLayer) for layer in self.transformer_encoder.layers):
            problems.append("the encoder layers are not RoPESwiGLUEncoderLayer")
        for module in self.regression_head.lin_layers:
            if isinstance(module, ChannelBN) and not (
                module.bn.affine and module.bn.running_mean is not None and module.bn.running_var is not None
            ):
                problems.append("a head BatchNorm has no affine weights or running statistics (it cannot be folded)")
                break
        if src_seg_len is not None and not (0 <= self.block_len // 2 < src_seg_len.shape[1]):
            problems.append(
                f"the centre index {self.block_len // 2} is outside the segment row of width {src_seg_len.shape[1]}"
            )
        if problems:
            raise RuntimeError(f"The {self.ENGINE} engine cannot run this model: {'; '.join(problems)}.")

    # ------------------------------------------------------------------------------------------- compiled pieces
    def _swiglu_fused(self, ffn, x):
        """SwiGLU w2(silu(w1 x) * w3 x), with w1 and w3 [G, D] applied as one [2G, D] matrix. Compiled."""
        d_inner = ffn.w1.out_features  # G
        # stacked in the graph at every call; the parameters are not changed
        y = F.linear(x, torch.cat([ffn.w1.weight, ffn.w3.weight], dim=0))  # [..., D] -> [..., 2G]
        return ffn.w2(F.silu(y[..., :d_inner]) * y[..., d_inner:])  # silu(w1 x) * w3 x: [..., G]; w2: -> [..., D]

    def _head_folded(self, x):
        """The regression head with each Linear -> BatchNorm pair folded into one Linear (eval mode). Compiled.

        BN(x W^T + b) = x (W * s[:, None])^T + (b - mean) * s + beta, with s = gamma / sqrt(var + eps).
        Each module acts per token, so the head can run on just the centre tokens: [1, U_pad, D] -> [1, U_pad, C].
        """
        modules = list(self.regression_head.lin_layers)
        i = 0
        while i < len(modules):
            module = modules[i]
            following = modules[i + 1] if i + 1 < len(modules) else None
            if isinstance(module, nn.Linear) and isinstance(following, ChannelBN):
                bn = following.bn
                # s [D]; folded weight [D, D] and bias [D], computed in the graph at every call (parameters unchanged)
                scale = bn.weight * torch.rsqrt(bn.running_var + bn.eps)
                x = F.linear(x, module.weight * scale.unsqueeze(1), (module.bias - bn.running_mean) * scale + bn.bias)
                i += 2
                continue
            # activations, Dropout and the last three Linear layers run as they are
            x = module(x)
            i += 1
        return x

    def _head_on_centre_tokens(self, centre_tokens):
        """The head on the centre-token buffer, [U_pad, D] -> [U_pad, C]. Compiled (fast) or in the packed graph (fa4).

        The reference averages the head output over the centre base's windows and multiplies all others by 0, so the
        head on the centre tokens alone gives the same mean. The buffer runs as one row, [1, U_pad, D], the 3-D shape
        of the reference head's input.
        """
        return self._head_folded(centre_tokens.unsqueeze(0)).squeeze(0)

    # ------------------------------------------------------------------------------------------- eager readout
    def _readout(self, head_out, centre_count, row_start):
        """Per-row sums [B, C] float32 of head_out [U_pad, C] over each row's centre slots, and the head's dtype.

        Row r owns slots row_start[r] .. row_start[r] + centre_count[r] - 1, so with running = [0, cumsum(head_out)]
        its sum is running[row_start[r] + centre_count[r]] - running[row_start[r]]: two gathers, no loop. The running
        sum is fp64, so the other rows of a batch change a row's sum only by fp64 rounding.
        """
        # [1, C] zeros, then cumsum [U_pad, C] -> running [U_pad + 1, C] float64
        running = torch.cat(
            [
                torch.zeros(1, head_out.shape[1], dtype=torch.float64, device=head_out.device),
                torch.cumsum(head_out.double(), 0),
            ]
        )
        # a row with no centre tokens reads the same entry twice and gets 0
        count = centre_count.clamp(min=0)
        # two [B]-index gathers -> [B, C] float64 -> float32
        row_sum = (running.index_select(0, row_start + count) - running.index_select(0, row_start)).float()
        return row_sum, head_out.dtype

    def _finish(self, row_sum, out_dtype, centre_count):
        """The per-row mean, as a probability unless the model returns logits. Returns [B, C] in the head's dtype.

        That is fp16 under CUDA autocast and float32 without it; eager returns float32 in both cases (autocast runs its
        sum over the windows in float32). A row without centre tokens gives NaN, as in eager.
        """
        output = row_sum.to(out_dtype) / centre_count.unsqueeze(1)  # [B, C] / [B, 1]; 0 / 0 = NaN
        if not self.output_logits:
            output = torch.sigmoid(output)
        return output

    # ------------------------------------------------------------------------------------------- arming
    def __call__(self, *args, **kwargs):
        if self.training:
            if self._armed:
                raise RuntimeError(f"The {self.ENGINE} engine was compiled for inference; build a new model to train.")
            return nn.Module.__call__(self, *args, **kwargs)
        # The first call with a CUDA tensor arms the engine (wraps the graph functions in torch.compile).
        if not self._armed:
            x = _first_tensor(args, kwargs)
            if x is not None and x.is_cuda:
                self._arm(x.device, args, kwargs)
        if not self._armed:
            return nn.Module.__call__(self, *args, **kwargs)  # CPU: the same code, uncompiled
        state = self._state
        state.calls += 1
        started = time.time()
        try:
            out = nn.Module.__call__(self, *args, **kwargs)
        except _compile_errors() as err:
            device = _first_tensor(args, kwargs).device
            gpu = torch.cuda.get_device_name(device)
            cc = ".".join(map(str, torch.cuda.get_device_capability(device)))
            first_line = str(err).splitlines()[0] if str(err) else ""
            raise RuntimeError(
                f"The {self.ENGINE} engine could not compile the model for this GPU ({gpu}, compute capability {cc}): "
                f"{type(err).__name__}: {first_line} -- Run with --engine eager, which needs no compiler. (If the "
                "message says no C compiler was found: Triton needs gcc at run time.)"
            ) from err
        # a larger graph count means this call compiled a new graph
        graphs, breaks = dynamo_counts()
        if graphs != state.graphs_seen:
            if _compile_logs_on():
                if isinstance(out, torch.Tensor) and out.is_cuda:
                    torch.cuda.synchronize(out.device)
                x = _first_tensor(args, kwargs)
                log.info(
                    "%s engine: call %d (batch %s) compiled %d new graph(s); %d graphs, %d breaks so far; %.1f s.",
                    self.ENGINE,
                    state.calls,
                    tuple(x.shape) if x is not None else None,
                    graphs - state.graphs_seen,
                    graphs,
                    breaks,
                    time.time() - started,
                )
            state.graphs_seen = graphs
        return out

    def _arm(self, device, args=(), kwargs=None, backend=None):
        """Compiles the engine for `device`, refusing every way it could silently run differently.

        args, kwargs: the first call's arguments, checked before compiling.
        """
        import torch._dynamo
        import torch._dynamo.config as dynamo_config

        kwargs = kwargs or {}
        if self.NEEDS_FA4 and device.type == "cuda":
            fa4 = fa4_preflight(device)
        else:
            fa4 = None
        if self.training:
            raise RuntimeError(f"The {self.ENGINE} engine cannot be compiled in train mode.")
        self._check_model(_arg(args, kwargs, 2, "src_seg_len"))
        if _arg(args, kwargs, 3, "src_dwell_bq") is not None:
            raise ValueError(f"The {self.ENGINE} engine takes no dwell / base-quality input; use the eager engine.")
        dynamo_config.suppress_errors = False  # a compile error stops the run, never a silent fall back to eager
        stance = getattr(getattr(torch._dynamo.eval_frame, "_stance", None), "stance", "default")
        if getattr(dynamo_config, "disable", False) or stance != "default":
            raise RuntimeError(
                f"The {self.ENGINE} engine cannot be compiled: torch.compile is disabled (TORCH_COMPILE_DISABLE, or "
                f"compiler stance {stance!r}). Use --engine eager."
            )
        # raise the recompile limit (its name differs between torch versions) and make reaching it an error
        for name in ("recompile_limit", "cache_size_limit"):
            if hasattr(dynamo_config, name) and getattr(dynamo_config, name) < RECOMPILE_LIMIT:
                setattr(dynamo_config, name, RECOMPILE_LIMIT)
        if hasattr(dynamo_config, "fail_on_recompile_limit_hit"):
            dynamo_config.fail_on_recompile_limit_hit = True
        if _compile_logs_on():
            try:
                torch._logging.set_logs(recompiles=True)
            except Exception:  # noqa: BLE001 -- log lines only
                pass
        # compiled kernels load from the cache folders, so these must allow running files
        inductor_dir, triton_dir = _plan_compile_cache_dirs()
        blocked = [d for d in (inductor_dir, triton_dir or os.path.join(inductor_dir, "triton")) if _is_noexec(d)]
        if blocked:
            raise RuntimeError(
                f"The compile cache folder {blocked[0]} is on a mount that does not allow running files. Set "
                f"TORCHINDUCTOR_CACHE_DIR (and TRITON_CACHE_DIR) to another folder, or use --engine eager."
            )
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_dir
        if triton_dir:
            os.environ["TRITON_CACHE_DIR"] = triton_dir
        compile_args = compile_kwargs(backend=backend)
        self._compile(compile_args)
        self._armed = True
        self._state.graphs_seen = dynamo_counts()[0]
        log.info(
            "%s engine ready (%s, form %s, torch %s%s%s).",
            self.ENGINE,
            type(self).__name__,
            self.FORM,
            torch.__version__,
            f", FlashAttention-4 {fa4['version']}" if fa4 else "",
            f", test backend {backend}" if backend else "",
        )
        if _compile_logs_on():
            here = os.path.abspath(__file__)
            reference = sys.modules[TransformerModel.__module__].__file__
            log.info(
                "%s engine details: compile %r; cache %s; %s md5 %s; %s md5 %s; device %s.",
                self.ENGINE,
                compile_args,
                inductor_dir,
                here,
                _file_md5(here),
                reference,
                _file_md5(reference),
                device,
            )


# ================================================================================================ fast engine


class FastEngine(_CompiledEngine):
    """The fast engine: length buckets, PyTorch attention, one compiled graph per bucket width plus one for the head.

    The last layer computes queries for W window positions per pass, from the centre base's first window (more passes
    for a longer centre base). The centre tokens go into one buffer [U_pad + 1, D], whose last slot takes the writes
    of the positions past them, and the head runs once on its first U_pad slots.
    """

    ENGINE = "fast"
    FORM = "l12g3_56"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compiled_bucket = None
        self._compiled_head = None

    def _compile(self, kwargs):
        # the bucket graph (one graph per width; extra passes reuse it) and the head (one graph per U_pad)
        self._compiled_bucket = torch.compile(self._bucket_graph, **kwargs)
        self._compiled_head = torch.compile(self._head_on_centre_tokens, **kwargs)

    # ------------------------------------------------------------------------------------------- eager batch loop
    def _forward_batch(self, src_kmer, src_signal, src_seg_len, counts=None):
        """Shapes: src_kmer [B, M], src_signal [B, R], src_seg_len [B, L]; returns [B, C]. `counts` is unused:
        this engine needs the bucket sizes too, which it reads from the device."""
        device = src_seg_len.device
        n_rows = src_seg_len.shape[0]
        S = self._effective_seq_len(src_signal)  # (R - Z) // 6 + 1
        if S > self.max_seq_len:
            raise ValueError(f"Input would yield eff_seq_len={S}, but max_seq_len={self.max_seq_len}.")
        widths = bucket_widths_for(S)
        n_buckets = len(widths)
        window = CENTRE_WINDOW
        centre = self.block_len // 2  # L // 2: the candidate A
        seg = src_seg_len
        centre_count = seg[:, centre].long()  # [B] int64: centre tokens (windows of the centre base) of each row
        with torch.no_grad():
            row_len = seg.sum(dim=1)  # [B] int64: real tokens (windows) of each row
            # bucket[r] = #{w in widths[:-1]: row_len[r] > w}: index of the smallest width holding the row; 130 -> 2
            bucket = torch.zeros_like(row_len)
            for w in widths[:-1]:
                bucket += (row_len > w).long()
            rows_in_bucket = torch.bincount(bucket, minlength=n_buckets)  # [n_buckets]
            # n centre tokens need ceil(n / W) - 1 extra passes (0 for n = 0); ceil(S / W) - 1 suffices, as n <= S
            max_passes = -(-int(S) // window) - 1
            passes_needed = None
            if max_passes > 0:
                passes_needed = (centre_count + (window - 1)).div(window, rounding_mode="floor").sub(1).clamp(min=0)
                # rows_in_pass[j, p - 1] = rows of bucket j that need pass p (needs_pass summed by bucket)
                needs_pass = passes_needed.unsqueeze(1) >= torch.arange(1, max_passes + 1, device=device).unsqueeze(0)
                rows_in_pass = torch.zeros(n_buckets, max_passes, dtype=torch.long, device=device).index_add_(
                    0, bucket, needs_pass.long()
                )
            else:
                rows_in_pass = torch.zeros(n_buckets, 0, dtype=torch.long, device=device)
            n_centre_tokens = centre_count.clamp(min=0).sum().reshape(1)  # [1]: U
            # [2]: any row longer than S, any negative length
            bad = torch.stack([(row_len > S).any().long(), (seg < 0).any().long()])
            # every count the host needs (39 numbers for S = 200) in one device-to-host copy. torch.bincount above
            # waits for the GPU too: the size of its output depends on the values.
            host = torch.cat([rows_in_bucket, rows_in_pass.reshape(-1), n_centre_tokens, bad]).tolist()
        bucket_sizes = host[:n_buckets]
        # pass_sizes[j][p - 1]: rows of bucket j that need extra pass p
        pass_sizes = [host[n_buckets + j * max_passes : n_buckets + (j + 1) * max_passes] for j in range(n_buckets)]
        n_centre_total, bad_sum, bad_negative = host[n_buckets + n_buckets * max_passes :]
        if bad_sum:
            raise ValueError(
                f"Some rows have segment_len_arr summing > eff_seq_len={S}.  Either reduce input dwell or feed a wider "
                f"signal_token (more samples) so eff_seq_len grows."
            )
        if bad_negative:
            raise ValueError("A segment length is negative.")

        # One buffer in row order: row r owns slots row_start[r] .. row_start[r] + centre_count[r] - 1. capacity = U_pad
        # is block, or the smallest whole multiple of block that holds U; spare slot `capacity` takes padding writes.
        block = centre_slot_block(max(n_rows, self._state.max_batch))
        capacity = block
        while capacity < n_centre_total:  # more centre tokens than planned: grow by whole blocks
            capacity += block
        row_start = torch.cumsum(centre_count.clamp(min=0), 0) - centre_count.clamp(min=0)
        buffer = None
        # order [B]: rows sorted by bucket (stable), so bucket j's rows are order[taken : taken + n_j]
        order = torch.argsort(bucket, stable=True) if n_buckets > 1 else None
        taken = 0
        for bucket_index, width in enumerate(widths):
            n_bucket_rows = int(bucket_sizes[bucket_index])
            if n_bucket_rows == 0:
                continue
            rows = order[taken : taken + n_bucket_rows] if order is not None else torch.arange(n_rows, device=device)
            taken += n_bucket_rows
            n_samples = self._signal_samples_for(width)  # 6 * (T - 1) + 30
            # With several buckets the row count is padded to a multiple of row_granule with copies of the first row
            # (outputs dropped). index_select makes fresh inputs, so the graph never sees the caller's layout.
            index = self._pad_rows(rows, row_granule(n_rows)) if order is not None else rows
            centre_out = self._run_bucket(
                src_kmer.index_select(0, index),
                src_signal[:, :n_samples].index_select(0, index),
                seg.index_select(0, index),
                torch.zeros(index.shape[0], dtype=torch.long, device=device),
            )
            buffer = self._store_centre_tokens(
                buffer, capacity, centre_out, index, n_bucket_rows, 0, centre_count, row_start
            )
            # Extra passes reuse the same graph. by_need: the bucket's rows by passes needed, descending, so pass p's
            # rows are by_need[:rows_in_pass[j, p - 1]].
            if max_passes > 0 and pass_sizes[bucket_index][0]:
                # Pass p starts at centre token p * W, given as a [B] tensor so the pass number never becomes a graph
                # constant.
                by_need = rows.index_select(
                    0, torch.argsort(passes_needed.index_select(0, rows), descending=True, stable=True)
                )
                for pass_number in range(1, max_passes + 1):
                    n_pass_rows = int(pass_sizes[bucket_index][pass_number - 1])
                    if n_pass_rows == 0:
                        break
                    sub = by_need[0:n_pass_rows]
                    pass_index = self._pad_rows(sub, EXTRA_PASS_GRANULE)
                    centre_out = self._run_bucket(
                        src_kmer.index_select(0, pass_index),
                        src_signal[:, :n_samples].index_select(0, pass_index),
                        seg.index_select(0, pass_index),
                        torch.full((pass_index.shape[0],), pass_number * window, dtype=torch.long, device=device),
                    )
                    buffer = self._store_centre_tokens(
                        buffer,
                        capacity,
                        centre_out,
                        pass_index,
                        n_pass_rows,
                        pass_number * window,
                        centre_count,
                        row_start,
                    )
        if buffer is None:  # no rows
            # [0, 1]: one column, which assumes C = 1 (the v1.2 weights)
            row_sum = torch.zeros(n_rows, 1, dtype=torch.float32, device=device)
            return self._finish(row_sum, self._guess_head_dtype(device), centre_count)
        head = self._compiled_head if self._compiled_head is not None else self._head_on_centre_tokens
        # [U_pad, D]: the buffer without its spare row; static, so the head compiles once per buffer size
        centre_tokens = buffer[:capacity]
        torch._dynamo.mark_static(centre_tokens, 0)
        head_out = head(centre_tokens)  # [U_pad, C]
        row_sum, out_dtype = self._readout(head_out, centre_count, row_start)
        return self._finish(row_sum, out_dtype, centre_count)

    def _store_centre_tokens(self, buffer, capacity, centre_out, index, n_real, first_token, centre_count, row_start):
        """Writes one call's centre tokens [rows, W, D] into the buffer [U_pad + 1, D].

        Position j of real row r goes to slot row_start[r] + first_token + j if the row has that centre token; padding
        rows and positions go to the spare slot U_pad.
        """
        states, _ = centre_out
        device = states.device
        window = CENTRE_WINDOW
        # zeroed at a batch's first call; slots U .. U_pad - 1 stay 0 (the head runs on them, the readout skips them)
        if buffer is None:
            buffer = torch.zeros(capacity + 1, states.shape[-1], dtype=states.dtype, device=device)
        real_rows = index[:n_real]
        position = torch.arange(window, device=device).unsqueeze(0)  # [1, W]
        # in_window [n_real, 1]: real centre tokens among the W positions
        in_window = (centre_count.index_select(0, real_rows) - first_token).clamp(min=0, max=window).unsqueeze(1)
        # slot [n_real, W]: the target slot, or the spare slot U_pad past the row's centre tokens
        slot = torch.where(
            position < in_window,
            row_start.index_select(0, real_rows).unsqueeze(1) + first_token + position,
            torch.full_like(position, capacity),
        )
        # the padding rows of the call (copies) write only to the spare slot: slot [rows, W]
        slot = torch.cat(
            [slot, torch.full((states.shape[0] - n_real, window), capacity, dtype=slot.dtype, device=device)]
        )
        # Real slots are distinct. Only the spare slot is written many times (which write wins is undefined); it is
        # never read.
        buffer.index_copy_(0, slot.reshape(-1), states.reshape(-1, states.shape[-1]))
        return buffer

    def _signal_samples_for(self, n_windows):
        """Signal samples that hold exactly `n_windows` windows: 6 * (n_windows - 1) + 30."""
        return self.signal_stride * (n_windows - 1) + self.signal_stride * self.kmer_size

    @staticmethod
    def _pad_rows(index, granule):
        """Pads `index` to a multiple of `granule` with its first entry: [7, 3, 5], 8 -> [7, 3, 5, 7, 7, 7, 7, 7]."""
        pad = (-index.shape[0]) % max(1, int(granule))
        return torch.cat([index, index[:1].expand(pad)]) if pad else index

    # the head's output dtype, for a batch with no rows (no head call ran)
    @staticmethod
    def _guess_head_dtype(device):
        if device.type == "cuda" and torch.is_autocast_enabled("cuda"):
            return torch.get_autocast_dtype("cuda")
        return torch.float32

    def _run_bucket(self, kmer, signal, seg_len, first_centre_token):
        """Eager input preparation for one bucket call, then the bucket graph. Returns (centre token states, empty)."""
        n = kmer.shape[0]
        # Only the first pass of a one-bucket batch can have 1 row; other calls are padded to 8 or more rows.
        if n == 1:  # torch.compile specialises size 1: run 2 copies, drop the second
            kmer, signal, seg_len, first_centre_token = (
                torch.cat([t, t], 0) for t in (kmer, signal, seg_len, first_centre_token)
            )
        T = self._effective_seq_len(signal)  # T: the bucket width
        with torch.no_grad():
            # seg_len_flat [B * (L + 1)] int64; kmer_ids [B, T] int32; both made outside the graph
            seg_len_flat = self._flat_segment_lengths(seg_len, T)
            kmer_ids = self._window_kmer_ids(kmer, seg_len_flat, T)
        # set on every call: the marks belong to the tensor objects, which are new each time
        for t in (kmer_ids, signal, seg_len, first_centre_token):
            torch._dynamo.mark_dynamic(t, 0)  # one graph for any row count
        # the width stays static (else the second width would make it symbolic): one graph per width
        for t in (kmer_ids, signal):
            torch._dynamo.mark_static(t, 1)
        graph = self._compiled_bucket if self._compiled_bucket is not None else self._bucket_graph
        states, empty = graph(kmer_ids, signal, seg_len, first_centre_token)
        return states[:n], empty  # [B, W, D]: drops the copy of a 1-row call

    def _flat_segment_lengths(self, src_seg_len, eff_seq_len):
        """flatten_seg_len of the reference model, with the input check on the device (no host sync).

        [B, L] -> [B * (L + 1)] int64: each row's L segment lengths, then its padding windows T - (row sum).
        """
        per_row_sum = src_seg_len.sum(dim=1, keepdims=True)
        ok = torch.all(per_row_sum <= eff_seq_len) & torch.all(src_seg_len >= 0)
        # a failure surfaces at a later sync; the host does not wait here
        torch._assert_async(ok, "segment_len_arr sums > eff_seq_len, or a negative segment length")
        src_seg_len_flat = torch.cat([src_seg_len, eff_seq_len - per_row_sum], dim=1)
        return src_seg_len_flat.flatten()

    def _token_segment_ids(self, src_seg_len_flat, batch, eff_seq_len):
        """[B, T] int64: the base (segment) index 0..L of every window; L = padding.

        Window t is in segment #{j: cum[j] <= t}, which skips empty segments as repeat_interleave does. E.g. lengths
        [2, 0, 3], T = 6: cum = [2, 2, 5], windows get [0, 0, 2, 2, 2, 3] (3 = padding).
        """
        seg = src_seg_len_flat.reshape(batch, -1)  # [B, L + 1]
        cum = torch.cumsum(seg[:, :-1].long(), dim=1)  # [B, L]: end of each base's windows
        # positions [B, T]: 0 .. T - 1; contiguous, else searchsorted copies it with a warning
        positions = self.seq_positions[:eff_seq_len].unsqueeze(0).expand(batch, eff_seq_len).contiguous()
        return torch.searchsorted(cum, positions, right=True)  # [B, T]

    def _window_kmer_ids(self, src_kmer, src_seg_len_flat, eff_seq_len):
        """process_kmer of the reference model with a gather instead of repeat_interleave: [B, M] -> [B, T].

        repeat_interleave has a data-dependent output size, which costs a host sync on the GPU.
        """
        batch = src_kmer.shape[0]
        src_kmer = src_kmer.long().clamp(0, 255)  # [B, M] ASCII codes
        src_kmer = self.ascii_to_base[src_kmer]  # [B, M] base codes: A=1, C=2, G=3, T/U=4, other 0
        src_kmer = src_kmer.unfold(1, self.kmer_size, 1)  # [B, L, 5]: the 5-mer around each of the L bases
        src_kmer = src_kmer * self.kmer_radix.unsqueeze(0).unsqueeze(0)  # digit i times 5**i (1, 5, 25, 125, 625)
        src_kmer = src_kmer.sum(dim=-1) + 1  # [B, L]: id = 1 + sum, in 1..3125; 0 is kept for padding
        # [B, L + 1]: id 0 for the padding segment
        src_kmer = torch.cat([src_kmer, torch.zeros(batch, 1, device=src_kmer.device, dtype=torch.long)], dim=1)
        if src_kmer.shape[0] * src_kmer.shape[1] != src_seg_len_flat.shape[0]:
            raise ValueError(
                f"The k-mer input ({src_kmer.shape[1] - 1} k-mers per row) does not match the segment input "
                f"({src_seg_len_flat.shape[0] // batch - 1} segments per row)."
            )
        seg_ids = self._token_segment_ids(src_seg_len_flat, batch, eff_seq_len)  # [B, T] int64
        return torch.gather(src_kmer, 1, seg_ids).int()  # [B, T] int32: the id of each window's base

    # ------------------------------------------------------------------------------------------- compiled bucket graph
    def _bucket_graph(self, kmer_ids, src_signal, src_seg_len, first_centre_token):
        """One bucket call. Compiled: one graph per bucket width.

        kmer_ids [B, T] int32, src_signal [B, 6 * (T - 1) + 30], src_seg_len [B, L], first_centre_token [B] (0, or
        p * W for extra pass p) -> (states after final_norm of the pass's W centre positions [B, W, D], empty tensor).
        """
        T = self._effective_seq_len(src_signal)  # T, fixed in this graph (the signal width is static)
        windows = self.process_signal(src_signal)  # [B, T, Z]: overlapping windows (a view)
        pad = self.create_src_pad_mask(src_seg_len, T)  # [B, T] bool, True = padding window
        # [B, T, D]; positions enter only through the rotary embedding in attention
        x = self.kmer_embedding(kmer_ids) + self.signal_embedding(windows)
        encoder = self.transformer_encoder
        layers = encoder.layers
        n_full = len(layers) - 1
        lens = src_seg_len.sum(dim=1)  # unused; kept (see the editing rule in the module docstring)
        # all layers but the last, on every window: [B, T, D] -> [B, T, D]
        for i in range(n_full):
            x = self._encoder_layer(layers[i], x, pad, lens)
        centre = self.block_len // 2
        window = CENTRE_WINDOW
        # first [B]: window index of this pass's first centre token. n_in_window and in_window are unused (kept).
        first = src_seg_len[:, :centre].sum(dim=1) + first_centre_token
        n_in_window = (src_seg_len[:, centre].long() - first_centre_token).clamp(min=0, max=window)
        j = torch.arange(window, device=x.device)
        # the pass's W windows, clamped to T - 1; those past the row's centre tokens are computed but stored in the
        # spare slot
        position = (first.unsqueeze(1) + j.unsqueeze(0)).clamp(max=T - 1)  # [B, W]
        in_window = j.unsqueeze(0) < n_in_window.unsqueeze(1)  # noqa: F841 -- unused; kept (see the editing rule)
        states = self._last_layer_window(layers[-1], x, pad, lens, position)  # [B, W, D]
        states = encoder.final_norm(states)  # on the W positions only
        return states, torch.empty(0, dtype=states.dtype, device=states.device)

    def _attention(self, q, k, v, pad, lens):
        """The reference layer's attention: additive 0 / -inf key mask in q's dtype, then SDPA. Compiled.

        q [B, H, Tq, hd]; k, v [B, H, T, hd]; pad [B, T] (True = padding key) -> [B, Tq, D]. `lens` is unused.
        """
        n_rows, n_keys = pad.shape
        # attn_mask [B, 1, 1, T]: 0 for a real key, -inf for a padding key
        attn_mask = torch.zeros((n_rows, 1, 1, n_keys), device=q.device, dtype=q.dtype)
        attn_mask.masked_fill_(pad.unsqueeze(1).unsqueeze(1), float("-inf"))
        # [B, H, Tq, hd]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        b, h, tq, hd = out.shape
        return out.transpose(1, 2).reshape(b, tq, h * hd)  # [B, Tq, H, hd] -> [B, Tq, D]

    def _encoder_layer(self, layer, x, pad, lens):
        """RoPESwiGLUEncoderLayer.forward in eval mode, with the fused SwiGLU layer. Compiled."""
        B, S, D = x.shape  # S = T in the fast engine
        normed = layer.norm1(x)
        # [B, S, D] -> [B, S, 3D] -> [B, S, 3, H, hd] -> permute -> [3, B, H, S, hd]; unbind: q, k, v [B, H, S, hd]
        qkv = layer.qkv(normed).reshape(B, S, 3, layer.n_heads, layer.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        cos, sin = layer.rope(S)  # [S, hd / 2] each: positions 0 .. S - 1
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        attn = self._attention(q, k, v, pad, lens)
        attn = layer.attn_out(attn)  # [B, S, D]
        x = x + attn
        x = x + self._swiglu_fused(layer.ffn, layer.norm2(x))
        return x

    def _last_layer_window(self, layer, x, pad, lens, position):
        """The last layer with queries only at `position` [B, W]; K and V use every token. Compiled. Returns [B, W, D].

        Equal to the reference layer at those positions: a token's output depends only on its own input and on the K and
        V of its row.
        """
        B, S, D = x.shape  # S = T
        H, hd = layer.n_heads, layer.head_dim
        W = position.shape[1]  # W = CENTRE_WINDOW
        normed = layer.norm1(x)
        w_qkv = layer.qkv.weight  # [3D, D]: rows of q, then k, then v
        # K and V only (weight rows D .. 3D - 1): [B, S, D] -> [B, S, 2D] -> [B, S, 2, H, hd] -> [2, B, H, S, hd]
        kv = F.linear(normed, w_qkv[D:]).reshape(B, S, 2, H, hd).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        cos, sin = layer.rope(S)  # keys at positions 0 .. S - 1
        k = apply_rotary(k, cos, sin)
        # index [B, W, D]: position repeated over the channels
        index = position.unsqueeze(-1).expand(B, W, D)
        normed_window = torch.gather(normed, 1, index)  # [B, W, D]
        # queries for the W positions (weight rows 0 .. D - 1): [B, W, D] -> [B, W, H, hd] -> [B, H, W, hd]
        q = F.linear(normed_window, w_qkv[:D]).reshape(B, W, H, hd).permute(0, 2, 1, 3)
        # cos_w, sin_w [B, 1, W, hd / 2]: per-row angles, since each row's centre starts at a different window
        cos_w = layer.rope.cos_cached[position].unsqueeze(1).to(q.dtype)
        sin_w = layer.rope.sin_cached[position].unsqueeze(1).to(q.dtype)
        # rotate pairs (2i, 2i + 1) as in _rotate_packed: [B, H, W, hd / 2, 2] -> flatten -> [B, H, W, hd]
        q1, q2 = q[..., 0::2], q[..., 1::2]
        q = torch.stack([q1 * cos_w - q2 * sin_w, q1 * sin_w + q2 * cos_w], dim=-1).flatten(-2)
        attn = self._attention(q, k, v, pad, lens)  # [B, W, D]
        attn = layer.attn_out(attn)
        states = torch.gather(x, 1, index) + attn  # residual at the W positions: [B, W, D]
        states = states + self._swiglu_fused(layer.ffn, layer.norm2(states))
        return states


# ================================================================================================ fa4 engine

# N_pad = max(MIN_TOKEN_BLOCKS, ceil(N / TOKEN_GRANULE)) * TOKEN_GRANULE: N_pad changes only through the block count,
# which is dynamic. Padding tokens lie outside every sequence of cu_seqlens; their attention output is set to 0, and
# their states reach only padding slots, which are set to 0 before the head.
TOKEN_GRANULE = 1024
MIN_TOKEN_BLOCKS = 2  # so the dynamic block count is never 1, which torch.compile specialises


def _rotate_packed(x, cos, sin):
    """apply_rotary for packed tokens with per-token angles. Compiled.

    x [N_pad or U_pad, H, hd]; cos, sin [same rows, 1, hd / 2]. Pair (2i, 2i + 1) at position p turns by p * theta_i,
    theta_i = 10000 ** (-2i / hd): (x1, x2) -> (x1 cos - x2 sin, x1 sin + x2 cos).
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Fa4Engine(_CompiledEngine):
    """The fa4 engine: the real tokens of a batch packed into one dimension, FlashAttention-4 varlen attention, and the
    whole model as one compiled graph (one more for each new buffer size U_pad).

    With host counts (deeprm call run passes them) there is no host sync; without them, one host sync per batch reads
    N and U. The per-row mean is computed outside the graph.
    """

    ENGINE = "fa4"
    FORM = "u2"
    NEEDS_FA4 = True
    ACCEPTS_COUNTS = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compiled_packed = None

    def _compile(self, kwargs):
        # the whole model: one graph per signal width R and buffer size U_pad
        self._compiled_packed = torch.compile(self._packed_graph, **kwargs)

    def _forward_batch(self, src_kmer, src_signal, src_seg_len, counts=None):
        """Shapes: src_kmer [B, M], src_signal [B, R], src_seg_len [B, L]; returns [B, C].

        counts: (real tokens N, centre tokens U, longest row, any negative length), summed on the host by the
        caller (inference.batch_counts). None reads them from the device, which makes the host wait for the GPU.
        """
        device = src_seg_len.device
        n_rows = src_seg_len.shape[0]
        S = self._effective_seq_len(src_signal)  # (R - Z) // 6 + 1
        if S > self.max_seq_len:
            raise ValueError(f"Input would yield eff_seq_len={S}, but max_seq_len={self.max_seq_len}.")
        # M - 5 + 1 5-mers must match the L segment lengths
        if src_kmer.shape[1] - self.kmer_size + 1 != src_seg_len.shape[1]:
            raise ValueError(
                f"The k-mer input ({src_kmer.shape[1] - self.kmer_size + 1} k-mers per row) does not match the segment "
                f"input ({src_seg_len.shape[1]} segments per row)."
            )
        # FlashAttention-4 has no CPU kernel; one registered for deeprm_fa4::varlen (for testing) runs this on the CPU
        if device.type != "cuda" and not torch._C._dispatch_has_kernel_for_dispatch_key("deeprm_fa4::varlen", "CPU"):
            raise RuntimeError("The fa4 engine runs on a CUDA GPU only; use the fast engine on CPU.")
        if n_rows == 0:
            # [0, 1]: one column, which assumes C = 1 (the v1.2 weights)
            return torch.zeros(0, 1, dtype=self._guess_head_dtype(device), device=device)
        centre = self.block_len // 2
        kmer, signal, seg_len = src_kmer, src_signal, src_seg_len
        if n_rows == 1:  # torch.compile specialises size 1: run 2 copies, drop the second
            kmer, signal, seg_len = (torch.cat([t, t], 0) for t in (kmer, signal, seg_len))
        # N, U, "a row longer than S" and "a negative length": from the caller's host counts, or else read from the
        # device in one host sync (row_len [B], centre_count [B])
        with torch.no_grad():
            centre_count = seg_len[:, centre].long()
            if counts is None:
                row_len = seg_len.sum(dim=1)
                host = torch.stack(
                    [
                        row_len.sum(),
                        centre_count.clamp(min=0).sum(),
                        (row_len > S).any().long(),
                        (seg_len < 0).any().long(),
                    ]
                ).tolist()  # the one host sync
                n_tokens, n_centre_tokens, bad_sum, bad_negative = host
            else:
                n_tokens, n_centre_tokens, longest_row, bad_negative = counts
                bad_sum = longest_row > S
                if n_rows == 1:  # the counts are of the one row; the batch holds its 2 copies
                    n_tokens, n_centre_tokens = 2 * n_tokens, 2 * n_centre_tokens
        if bad_sum:
            raise ValueError(
                f"Some rows have segment_len_arr summing > eff_seq_len={S}.  Either reduce input dwell or feed a wider "
                f"signal_token (more samples) so eff_seq_len grows."
            )
        if bad_negative:
            raise ValueError("A segment length is negative.")
        # n_blocks = N_pad / TOKEN_GRANULE; -(-a // b) is ceil(a / b)
        n_blocks = max(MIN_TOKEN_BLOCKS, -(-int(n_tokens) // TOKEN_GRANULE))
        # capacity = U_pad: block, or for this batch only the smallest whole multiple of block that holds U
        block = centre_slot_block(max(n_rows, self._state.max_batch))
        capacity = block
        while capacity < n_centre_tokens:  # more centre tokens than planned: grow by whole blocks
            capacity += block
        # Shape carriers: their sizes pass N_pad / TOKEN_GRANULE (dynamic) and U_pad (static) into the graph. A Python
        # int would be guarded by value (a recompile per new value) and cannot be marked dynamic or static.
        token_shape = torch.empty((n_blocks, TOKEN_GRANULE), dtype=torch.uint8, device=device)
        slot_shape = torch.empty((capacity,), dtype=torch.uint8, device=device)
        # Clones own new storage at offset 0, so a caller's slice never shows the graph an offset or a gap.
        kmer_, signal_, seg_len_ = kmer.clone(), signal.clone(), seg_len.clone()
        # dim 0 (B, and N_pad / TOKEN_GRANULE) dynamic; dim 1 (M, R, L and TOKEN_GRANULE) static; U_pad static
        for t in (kmer_, signal_, seg_len_, token_shape):
            torch._dynamo.mark_dynamic(t, 0)
        for t in (kmer_, signal_, seg_len_, token_shape):
            torch._dynamo.mark_static(t, 1)
        torch._dynamo.mark_static(slot_shape, 0)
        graph = self._compiled_packed if self._compiled_packed is not None else self._packed_graph
        head_out, _ = graph(kmer_, signal_, seg_len_, token_shape, slot_shape)  # [U_pad, C]
        count = centre_count.clamp(min=0)
        row_start = torch.cumsum(count, 0) - count  # [B]: first slot of each row
        row_sum, out_dtype = self._readout(head_out, centre_count, row_start)
        return self._finish(row_sum, out_dtype, centre_count)[:n_rows]  # [B, C]; drops the copy of a 1-row batch

    _guess_head_dtype = staticmethod(FastEngine._guess_head_dtype)

    # ------------------------------------------------------------------------------------------- compiled packed graph
    def _packed_graph(self, kmer, signal, seg_len, token_shape, slot_shape):
        """The whole model on the packed tokens. Compiled: one graph per signal width R and buffer size U_pad.

        kmer [B, M], signal [B, R], seg_len [B, L]; token_shape and slot_shape carry only sizes (N_pad, U_pad).
        Returns (head output [U_pad, C], centre-token states after final_norm [U_pad, D]).
        """
        S = self._effective_seq_len(signal)  # S = (R - Z) // 6 + 1, a constant here (R is static)
        with torch.no_grad():
            plan = self._packing_plan(kmer, seg_len, token_shape, slot_shape, S)
        n_samples = signal.shape[1]  # R
        # process_signal as one gather: window [N_pad, Z] = flat indices row * R + 6 * position + z into signal [B * R].
        # A padding token reads valid samples of the last row; its result is never used.
        window = (plan["row"] * n_samples + plan["position"] * self.signal_stride).unsqueeze(1) + torch.arange(
            self.signal_stride * self.kmer_size, device=signal.device
        ).unsqueeze(0)
        token_signal = signal.reshape(-1)[window]  # [N_pad, Z]
        x = self.kmer_embedding(plan["kmer_id"]) + self.signal_embedding(token_signal)  # [N_pad, D]
        layers = self.transformer_encoder.layers
        # all layers but the last, on all packed tokens: [N_pad, D] -> [N_pad, D]
        for i in range(len(layers) - 1):
            x = self._packed_layer(layers[i], x, plan, S)
        centre_states = self._packed_last_layer(layers[-1], x, plan, S)  # [U_pad, D]
        centre_states = self.transformer_encoder.final_norm(centre_states)
        # padding slots (u >= U) are set to 0
        centre_states = torch.where(
            plan["centre_is_real"].unsqueeze(1),
            centre_states,
            torch.zeros((), dtype=centre_states.dtype, device=centre_states.device),
        )
        head_out = self._head_on_centre_tokens(centre_states)  # [U_pad, C]
        return head_out, centre_states

    def _kmer_table(self, kmer):
        """[B, L] int64: the 5-mer id (1..3125) of each base (process_kmer before the repeat). Compiled."""
        k = kmer.long().clamp(0, 255)  # [B, M] ASCII codes
        k = self.ascii_to_base[k]  # [B, M] base codes 0..4
        k = k.unfold(1, self.kmer_size, 1)  # [B, M] -> [B, L, 5]
        k = k * self.kmer_radix.unsqueeze(0).unsqueeze(0)  # digit i times 5**i
        return k.sum(dim=-1) + 1  # [B, L]

    def _packing_plan(self, kmer, seg_len, token_shape, slot_shape, S):
        """Where every packed token and every centre-token slot comes from. Integers only. Compiled.

        Per token [N_pad]: row, position (in its row), is_real, kmer_id (int32), segment (r * L + j). Per slot [U_pad]:
        centre_row, centre_position, centre_token (its packed token), centre_is_real. cu_seqlens and cu_seqlens_centre
        [B + 1] int32: start of each row's tokens and slots. Example with 3 bases per row (centre index 1), lengths
        [2, 0, 3] and [1, 2, 1], S = 8, N_pad = 12, U_pad = 4 (so N = 9, U = 2)::

            cu_seqlens         [0, 5, 9]
            segment            [0, 0, 2, 2, 2, 3, 4, 4, 5, 5, 5, 5]  (segment 1 is empty; tokens 9.. are padding)
            row                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
            position           [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6]
            cu_seqlens_centre  [0, 0, 2]  (row 0 has no centre tokens)
            centre_row         [1, 1, 1, 1]
            centre_token       [6, 7, 8, 9]
            centre_is_real     [True, True, False, False]
        """
        device = seg_len.device
        n_segments = seg_len.shape[1]  # L
        n_pad = token_shape.shape[0] * token_shape.shape[1]  # N_pad
        capacity = slot_shape.shape[0]  # U_pad
        centre = self.block_len // 2
        seg = seg_len.long()  # [B, L]
        row_len = seg.sum(dim=1)  # [B]
        ok = torch.all(row_len <= S) & torch.all(seg >= 0)
        # a failure surfaces at a later sync; the host does not wait here
        torch._assert_async(ok, "segment_len_arr sums > eff_seq_len, or a negative segment length")
        row_end = torch.cumsum(row_len, 0)  # [B]: inclusive prefix sum; row_end[-1] = N
        row_start = row_end - row_len  # [B]: packed start of every row
        seg_start_in_row = torch.cumsum(seg, dim=1) - seg  # [B, L]: first window of base j in row r
        seg_start = (row_start.unsqueeze(1) + seg_start_in_row).reshape(-1)  # [B * L]; <= N <= N_pad
        # segment[t] = #{segment starts <= t} - 1: the last segment starting at or before t. An empty segment starts
        # where the next one does, so it is skipped, as in repeat_interleave. marks[t] counts the starts at t; entry
        # N_pad takes the starts at N when N = N_pad. Padding tokens (t >= N) get B * L - 1.
        marks = torch.zeros(n_pad + 1, dtype=torch.int32, device=device).index_add(
            0, seg_start, torch.ones_like(seg_start, dtype=torch.int32)
        )
        segment = torch.cumsum(marks[:n_pad], 0) - 1  # [N_pad] int64
        row = torch.div(segment, n_segments, rounding_mode="floor")  # [N_pad]: segment // L
        token = torch.arange(n_pad, device=device)  # [N_pad]
        is_real = token < row_end[-1]  # [N_pad] bool: t < N
        # position [N_pad]: t - start of its row, clamped so padding tokens' gathers stay in range
        position = (token - row_start.index_select(0, row)).clamp(min=0, max=S - 1)
        # kmer_id [N_pad] int32: the 5-mer id of each token's segment
        kmer_id = self._kmer_table(kmer).reshape(-1).index_select(0, segment).int()
        n_centre = seg[:, centre]  # [B]
        centre_end = torch.cumsum(n_centre, 0)  # [B]: inclusive prefix sum; centre_end[-1] = U
        first_slot = centre_end - n_centre  # [B]: first centre-token slot of every row
        # The same counting for slots: centre_row[u] = #{rows whose slots start at or before u} - 1, skipping rows
        # without centre tokens. The clamp only guards the index (the host makes U <= U_pad).
        slot_marks = torch.zeros(capacity + 1, dtype=torch.int32, device=device).index_add(
            0, first_slot.clamp(max=capacity), torch.ones_like(first_slot, dtype=torch.int32)
        )
        centre_row = torch.cumsum(slot_marks[:capacity], 0) - 1  # [U_pad]: owning row of slot u
        slot = torch.arange(capacity, device=device)  # [U_pad]
        centre_is_real = slot < centre_end[-1]  # [U_pad] bool: u < U
        k_in_row = slot - first_slot.index_select(0, centre_row)  # [U_pad]: rank of slot u in its row
        centre_first_in_row = seg[:, :centre].sum(dim=1)  # [B]: position of the row's first centre token
        # centre_position [U_pad]: the slot's window index in its row (for the query's rotary angle)
        centre_position = (centre_first_in_row.index_select(0, centre_row) + k_in_row).clamp(min=0, max=S - 1)
        # centre_token [U_pad]: packed index = row start + first centre window + rank; clamped for padding slots
        centre_token = (
            row_start.index_select(0, centre_row) + centre_first_in_row.index_select(0, centre_row) + k_in_row
        ).clamp(min=0, max=n_pad - 1)
        # cu_seqlens = [0, row_end]: each row's tokens (keys, and full-layer queries). cu_seqlens_centre = [0,
        # centre_end]: each row's last-layer query slots. FA4 takes int32 offsets.
        zero = row_end.new_zeros(1)
        cu_seqlens = torch.cat([zero, row_end]).to(torch.int32)
        cu_seqlens_centre = torch.cat([zero, centre_end]).to(torch.int32)
        return dict(
            row=row,
            position=position,
            is_real=is_real,
            kmer_id=kmer_id,
            cu_seqlens=cu_seqlens,
            cu_seqlens_centre=cu_seqlens_centre,
            centre_row=centre_row,
            centre_position=centre_position,
            centre_token=centre_token,
            centre_is_real=centre_is_real,
            segment=segment,
        )

    def _varlen_attention(self, q, k, v, cu_q, cu_k, S):
        """fa4_varlen with S as both length bounds; needs fp16 or bf16 q, k, v of one dtype on the GPU. Compiled."""
        if q.device.type == "cuda" and (q.dtype not in FA4_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype):
            raise RuntimeError(
                f"FlashAttention-4 needs fp16 or bf16 q, k and v of one dtype, got {q.dtype}/{k.dtype}/{v.dtype}; run "
                f"under torch.autocast as deeprm call run does."
            )
        return torch.ops.deeprm_fa4.varlen(q, k, v, cu_q, cu_k, S, S)

    def _packed_layer(self, layer, x, plan, S):
        """RoPESwiGLUEncoderLayer.forward on the packed tokens x [N_pad, D]. Compiled.

        Rows are kept apart only by cu_seqlens (attention) and plan["position"] (rotary angles).
        """
        n_pad, D = x.shape
        H, hd = layer.n_heads, layer.head_dim
        normed = layer.norm1(x)
        qkv = layer.qkv(normed).reshape(n_pad, 3, H, hd)  # [N_pad, 3D] -> [N_pad, 3, H, hd]
        q, k, v = qkv.unbind(1)  # q, k, v [N_pad, H, hd]: FA4 takes this layout
        # cos, sin [N_pad, 1, hd / 2]: angles of each token's position in its own row, as in the reference
        cos = layer.rope.cos_cached[plan["position"]].unsqueeze(1).to(q.dtype)
        sin = layer.rope.sin_cached[plan["position"]].unsqueeze(1).to(q.dtype)
        q = _rotate_packed(q, cos, sin)
        k = _rotate_packed(k, cos, sin)
        o = self._varlen_attention(q, k, v, plan["cu_seqlens"], plan["cu_seqlens"], S)  # [N_pad, H, hd]
        # FlashAttention-4 leaves the rows after the last real token unwritten
        o = torch.where(plan["is_real"].unsqueeze(1).unsqueeze(2), o, torch.zeros((), dtype=o.dtype, device=o.device))
        x = x + layer.attn_out(o.reshape(n_pad, H * hd))  # [N_pad, D]
        x = x + self._swiglu_fused(layer.ffn, layer.norm2(x))
        return x

    def _packed_last_layer(self, layer, x, plan, S):
        """The last layer: K and V for every packed token, everything else only for the centre slots. Compiled.

        Only centre tokens reach the head, and a token's output depends only on its own input and its row's K and V.
        Returns [U_pad, D].
        """
        n_pad, D = x.shape
        H, hd = layer.n_heads, layer.head_dim
        normed = layer.norm1(x)
        w_qkv = layer.qkv.weight  # [3D, D]: rows of q, then k, then v
        # K and V only (weight rows D .. 3D - 1): [N_pad, D] -> [N_pad, 2D] -> [N_pad, 2, H, hd]
        kv = F.linear(normed, w_qkv[D:]).reshape(n_pad, 2, H, hd)
        k, v = kv.unbind(1)  # [N_pad, H, hd] each
        # cos, sin [N_pad, 1, hd / 2]
        cos = layer.rope.cos_cached[plan["position"]].unsqueeze(1).to(k.dtype)
        sin = layer.rope.sin_cached[plan["position"]].unsqueeze(1).to(k.dtype)
        k = _rotate_packed(k, cos, sin)
        centre_token = plan["centre_token"]  # [U_pad]
        capacity = centre_token.shape[0]  # U_pad
        # queries of the slots' tokens (weight rows 0 .. D - 1): [U_pad, D] -> [U_pad, H, hd]
        q = F.linear(normed.index_select(0, centre_token), w_qkv[:D]).reshape(capacity, H, hd)
        # cos_c, sin_c [U_pad, 1, hd / 2]: angles of the slots' positions
        cos_c = layer.rope.cos_cached[plan["centre_position"]].unsqueeze(1).to(q.dtype)
        sin_c = layer.rope.sin_cached[plan["centre_position"]].unsqueeze(1).to(q.dtype)
        q = _rotate_packed(q, cos_c, sin_c)
        # row r's queries (its slots) attend to row r's keys (its packed tokens): o [U_pad, H, hd]
        o = self._varlen_attention(q, k, v, plan["cu_seqlens_centre"], plan["cu_seqlens"], S)
        # FA4 leaves slots u >= U unwritten
        o = torch.where(
            plan["centre_is_real"].unsqueeze(1).unsqueeze(2), o, torch.zeros((), dtype=o.dtype, device=o.device)
        )
        # residual from the slots' own tokens: [U_pad, D]
        states = x.index_select(0, centre_token) + layer.attn_out(o.reshape(capacity, H * hd))
        states = states + self._swiglu_fused(layer.ffn, layer.norm2(states))  # [U_pad, D]
        return states


# ================================================================================================ engine selection

ENGINE_CLASSES = {"fa4": Fa4Engine, "fast": FastEngine, "eager": TransformerModel}


def is_engine_model(model_name: str) -> bool:
    """True if `model_name` names the deeprm_v12 model (so ``--engine`` applies)."""
    return model_name == ARCHITECTURE


def engine_form(engine: str) -> str:
    """The form an engine runs: the engine name and a label of its code version (recorded in deeprm_run.json)."""
    return f"{engine}/{getattr(ENGINE_CLASSES[engine], 'FORM', None) or 'reference'}"


def compile_status(device: torch.device):
    """Checks that torch.compile can compile the engines on `device`. Returns (ok, reason)."""
    if os.environ.get("TORCH_COMPILE_DISABLE", "") == "1":  # the value torch itself acts on
        return False, "TORCH_COMPILE_DISABLE=1 is set"
    try:
        dynamo = importlib.import_module("torch._dynamo")
    except Exception as exc:  # noqa: BLE001 -- e.g. a broken dependency of torch._dynamo
        return False, f"torch._dynamo cannot be imported ({type(exc).__name__})"
    if getattr(getattr(dynamo, "config", None), "disable", False):
        return False, "torch._dynamo is disabled"
    if device.type == "cuda":
        cc_major = torch.cuda.get_device_capability(device)[0]
        if cc_major < COMPILE_MIN_CC:
            return False, f"GPU compute capability {cc_major}.x is below {COMPILE_MIN_CC}.0 (torch.compile needs 7.0+)"
        try:
            from torch.utils._triton import has_triton

            if not has_triton():
                return False, "Triton is not available"
        except ImportError:
            pass
        if not _c_compiler():
            return False, "no C compiler found (Triton needs gcc or clang at run time)"
        include = _python_include()
        if not os.path.isfile(os.path.join(include, "Python.h")):
            return False, f"Python's C headers are missing (no Python.h in {include}; Triton needs them at run time)"
    return True, "ok"


def _c_compiler():
    """The C compiler Triton would use (the CC variable, else cc, gcc or clang on PATH), or None."""
    if os.environ.get("CC"):
        return os.environ["CC"]
    return next((path for path in map(shutil.which, ("cc", "gcc", "clang")) if path), None)


def _python_include():
    """The folder Triton takes Python's C headers (Python.h) from, as triton/runtime/build.py looks it up."""
    scheme = sysconfig.get_default_scheme()
    if scheme == "posix_local":  # Debian's scheme; Triton reads the posix_prefix paths instead
        scheme = "posix_prefix"
    return sysconfig.get_paths(scheme=scheme)["include"]


def is_rocm():
    """True for a ROCm build of PyTorch.

    ROCm reports AMD GPUs as "cuda" with AMD architecture numbers as the compute capability (MI300X: 9.4), so the
    capability checks alone cannot tell them apart.
    """
    return bool(getattr(torch.version, "hip", None))


def _driver_cuda():
    """Newest CUDA version the NVIDIA driver supports, as (major, minor), or None (check_deps.driver_cuda: libcuda,
    then the nvidia-smi banner)."""
    return check_deps.driver_cuda()


def fa4_platform():
    """True on Linux x86_64, the only system DeepRM installs FlashAttention-4 on (the pyproject.toml marker)."""
    return platform.system() == "Linux" and platform.machine() == "x86_64"


def _fa4_route():
    """"cu130" or "cu128": the PyTorch build fa4 runs with here.

    A CUDA 13 build gives "cu130" and a CUDA 12.8 or newer 12.x build "cu128", without reading the driver. For an older
    or CPU-only build it is the build to install instead: "cu130" on a driver for CUDA 13, else "cu128", which runs on
    every CUDA 12 driver.
    """
    built = check_deps.version_tuple(torch.version.cuda)
    if built is not None and built[0] >= FA4_TORCH_CUDA_MAJOR:
        return "cu130"
    if built is not None and built >= FA4_TORCH_CUDA:
        return "cu128"
    driver = _driver_cuda()
    return "cu130" if driver is not None and tuple(driver) >= FA4_DRIVER_CUDA else "cu128"


def torch_cuda_status():
    """Checks that PyTorch is a build fa4 is chosen with (CUDA 12.8 or newer). Returns (ok, reason)."""
    if is_rocm():
        return False, "PyTorch is a ROCm build (FlashAttention-4 runs on NVIDIA GPUs only)"
    built = check_deps.version_tuple(torch.version.cuda)
    if built is None or built < FA4_TORCH_CUDA:
        what = f"a CUDA {torch.version.cuda} build" if torch.version.cuda else "a CPU-only build"
        return False, f"PyTorch {torch.__version__} is {what} (DeepRM chooses fa4 with a CUDA 12.8 or newer build)"
    return True, "ok"


def resolve_engine(engine: str, device: torch.device):
    """Picks the engine for `device`.

    Args:
        engine: "auto", "fa4", "fast" or "eager". "auto" picks fa4 where it runs and was tested, else fast, else eager
            (see the module docstring).

    Returns:
        tuple: (engine, reason).

    Raises:
        ValueError: If `engine` is unknown, or was asked for where it cannot run.
    """
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine {engine!r}. Choose one of {ENGINES}.")
    on_cuda = device.type == "cuda"
    cc_major = torch.cuda.get_device_capability(device)[0] if on_cuda else None
    can_compile, compile_reason = compile_status(device) if on_cuda else (True, "CPU")
    if engine == "auto":
        if not on_cuda:
            return "fast", "CPU (runs uncompiled)"
        if not can_compile:
            return "eager", compile_reason
        if cc_major not in FA4_CC_MAJORS:
            return "fast", f"FlashAttention-4 has no kernels for compute capability {cc_major}.x"
        if is_rocm():
            return "fast", torch_cuda_status()[1]
        cc = tuple(torch.cuda.get_device_capability(device))
        cuda_ok, cuda_reason = torch_cuda_status()
        if cc not in FA4_AUTO_CCS:
            reason = (
                "fa4 was tested on compute capability 8.0, 9.0 and 10.0 only (A100, H100, B200); this GPU is "
                f"{cc[0]}.{cc[1]}"
            )
            if cuda_ok and fa4_status()[0]:
                reason += ". --engine fa4 runs it here"
            return "fast", reason
        if not cuda_ok:
            return "fast", cuda_reason
        fa4_ok, fa4_reason = fa4_status()
        if not fa4_ok:
            return "fast", fa4_reason
        imports, import_reason = fa4_imports()
        if not imports:
            return "fast", import_reason
        return "fa4", f"FlashAttention-4 GPU (compute capability {cc[0]}.{cc[1]})"
    if engine == "fa4":
        if not on_cuda:
            raise ValueError(
                "--engine fa4 needs a CUDA GPU (compute capability 8.0 or newer). Use --engine fast on CPU."
            )
        if is_rocm():
            raise ValueError("--engine fa4 needs an NVIDIA GPU; FlashAttention-4 does not run on AMD GPUs (ROCm).")
        if cc_major not in FA4_CC_MAJORS:
            raise ValueError(
                f"--engine fa4 needs a GPU of compute capability 8.0 or newer; this GPU is {cc_major}.x. "
                "Use --engine fast."
            )
        # The CUDA build is not checked here (torch_cuda_status): a requested fa4 runs on the user's choice.
        fa4_ok, fa4_reason = fa4_status()
        if not fa4_ok:
            raise ValueError(f"--engine fa4: {fa4_reason}. Or use --engine fast.")
    if engine in ("fa4", "fast") and not can_compile:
        raise ValueError(f"--engine {engine} compiles the model on the GPU, but {compile_reason}. Use --engine eager.")
    return engine, "requested"


def fa4_install_command():
    """The command that installs FA4_REQUIREMENTS in this Python's environment."""
    return check_deps.pip_command("install", *(f'"{r}"' for r in FA4_REQUIREMENTS))


def _torch_fa4_command():
    """The command that installs the PyTorch build of _fa4_route (2.10.0 for CUDA 13.0 or 12.8)."""
    return check_deps.torch_install_command(_fa4_route())


def _torch_fa4_step(command, pool):
    """The step `command` (_torch_fa4_command) where its build has code for every GPU of `pool`.

    On a mixed machine it never gives a build without code for one of the GPUs unless it says so. Where the build of
    `command` has no code for one (cu130 on a V100), the step is the cu128 build if that one has code for all of them
    (it has code for every GPU that cu130 has, and for the V100). Where neither has (a P100 with an A100), the step says
    that no build fa4 runs with has code for that GPU, and gives the build with code for the most GPUs and the
    --gpu-pool of those GPUs.
    """
    route = _fa4_route()
    builds = {tag: archs for tag, archs, _ in check_deps.TORCH_CUDA_BUILDS}
    best = None  # (tag, indices of the GPUs it has code for)
    for tag in dict.fromkeys((route, "cu128")):  # the builds fa4 runs with, in order of preference
        keep = [index for index, _, cc in pool if check_deps.arch_supported(cc, builds[tag])]
        if len(keep) == len(pool):
            return command if tag == route else check_deps.torch_install_command(tag)
        if best is None or len(keep) > len(best[1]):
            best = tag, keep
    tag, keep = best
    if not keep:
        return command
    install = command if tag == route else check_deps.torch_install_command(tag)
    gpus = ", ".join(
        f"GPU {index} ({name}, compute capability {cc[0]}.{cc[1]})" for index, name, cc in pool if index not in keep
    )
    return (
        f"No PyTorch build that fa4 runs with has code for {gpus}. To use fa4, install the build for the other GPUs "
        f"(`{install}`) and run deeprm call run on them only (--gpu-pool {' '.join(map(str, keep))})."
    )


def _triton_pin():
    """The Triton version this PyTorch was built with (from its package metadata), or None."""
    try:
        import importlib.metadata

        for requirement in importlib.metadata.requires("torch") or ():
            name, _, rest = requirement.partition("==")
            if name.strip() == "triton" and rest:
                return rest.split(";")[0].strip()
    except Exception:  # noqa: BLE001 -- only a hint
        pass
    return None


def _fa4_version():
    try:
        import importlib.metadata

        return importlib.metadata.version(FA4_DIST)
    except Exception:  # noqa: BLE001 -- only reported
        return "(version unknown)"


def _compile_fixes(reason):
    """The steps that let torch.compile run, for a reason compile_status gave."""
    if reason.startswith("TORCH_COMPILE_DISABLE"):
        return ["unset TORCH_COMPILE_DISABLE"]
    if reason.startswith("no C compiler"):
        return [
            "Install a C compiler, which Triton needs at run time: `sudo apt install gcc` (Ubuntu, Debian), "
            "`sudo dnf install gcc` (RHEL, Rocky, Fedora) or `conda install -c conda-forge gcc` (no root needed)."
        ]
    if reason.startswith("Python's C headers"):
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        return [
            f"Install them: `sudo apt install python{version}-dev` (Ubuntu, Debian) or "
            f"`sudo dnf install python{version}-devel` (RHEL, Rocky, Fedora).",
            f"Or, without root, use a Python that has them: make a new environment with `uv venv --python {version} "
            "--python-preference only-managed <folder>`, or with conda, and install DeepRM there.",
        ]
    if reason.startswith("Triton"):
        pin = _triton_pin()
        return [check_deps.pip_command("install", f'"triton=={pin}"')] if pin else [_torch_fa4_command()]
    if reason == "torch._dynamo is disabled":
        return ["Remove the code that sets torch._dynamo.config.disable = True."]
    return [_torch_fa4_command()]  # torch._dynamo cannot be imported: a broken PyTorch install


def _fa4_install_fix():
    """(reason, steps) if FlashAttention-4 is missing, is not the verified build or does not import; else None."""
    command = fa4_install_command()
    code, detail = _fa4_check()
    if code == "missing":
        return "FlashAttention-4 is not installed", [command]
    if code == "site":
        return (
            f"{FA4_SITE_ENV} is set to {detail}, which holds no FlashAttention-4 (no flash_attn/cute/interface.py)",
            [
                f"Set {FA4_SITE_ENV} to the site-packages folder that holds flash_attn/cute, or unset it and run "
                f"`{command}`"
            ],
        )
    if code == "other":
        return (
            f"the flash_attn package that Python finds ({detail}) is not FlashAttention-4 (FlashAttention-2 uses the "
            "same name)",
            [
                command,
                f"If {detail} is still found first, take it off PYTHONPATH or uninstall it: "
                f"`{check_deps.pip_command('uninstall', '-y', 'flash-attn')}`",
            ],
        )
    if code == "unpinned":
        md5 = _file_md5(detail)
        if md5.startswith("UNREADABLE"):
            return f"{detail} cannot be read ({md5})", [command]
        return (
            f"FlashAttention-4 {_fa4_version()} is installed; DeepRM v1.2 was verified with {FA4_PIN_VERSION}",
            [
                f"Install the verified build: `{command}` (or set {FA4_UNPINNED_ENV}=1 to use the "
                "installed build, which DeepRM did not verify)"
            ],
        )
    imports, reason = fa4_imports()
    if not imports:
        return reason, [command]
    return None


def _not_tested_cause(gpu=None):
    """The (reason, steps) of a GPU fa4 was not tested on; `gpu` names it where one warning covers several GPU types."""
    kernels_for = "these GPUs" if gpu else "this GPU"
    return (
        f"FlashAttention-4 was not tested on {gpu or 'this GPU'}, so DeepRM does not choose fa4 by itself (DeepRM v1.2 "
        "tested fa4 on A100, H100 and B200: compute capability 8.0, 9.0 and 10.0)",
        [f"Run deeprm call run with --engine fa4 (FlashAttention-4 has kernels for {kernels_for})."],
    )


def fa4_fixes(device, pool=None):
    """Why fa4 cannot run on `device`, for each cause the user can fix, and how to fix it.

    Every cause is checked, not only the first one "auto" stops at, so a compile problem does not hide a missing
    FlashAttention-4. On a GPU that FlashAttention-4 has kernels for but fa4 was not tested on (one not in
    FA4_AUTO_CCS), auto keeps fast on purpose; that is reported as the last cause, with --engine fa4 as its step.

    Args:
        device: The GPU.
        pool: (index, name, compute capability) of every GPU of the run; None for every GPU PyTorch sees. A PyTorch
            build that has no code for one of them is given as a step only with a --gpu-pool that leaves it out
            (_torch_fa4_step).

    Returns:
        None off Linux x86_64 (where DeepRM does not install FlashAttention-4), if `device` is not a GPU that
        FlashAttention-4 supports, or if nothing keeps fa4 off it; else a list of (reason, steps) in fix order, each
        step one line.
    """
    if device.type != "cuda" or is_rocm() or not fa4_platform():
        return None
    cc = tuple(torch.cuda.get_device_capability(device))
    if cc[0] not in FA4_CC_MAJORS:  # no FlashAttention-4 kernels: nothing to tell
        return None
    causes = []
    cuda_ok, cuda_reason = torch_cuda_status()
    if not cuda_ok:
        causes.append((cuda_reason, [_torch_fa4_command()]))  # a build that runs on this driver: no driver update
    compile_ok, compile_reason = compile_status(device)
    if not compile_ok:
        causes.append((compile_reason, _compile_fixes(compile_reason)))
    install = _fa4_install_fix()
    if install:
        causes.append(install)
    if cc not in FA4_AUTO_CCS:
        causes.append(_not_tested_cause())
    command = _torch_fa4_command()
    if any(command in steps for _, steps in causes):
        if pool is None:
            pool = [
                (i, torch.cuda.get_device_name(i), tuple(torch.cuda.get_device_capability(i)))
                for i in range(torch.cuda.device_count())
            ]
        step = _torch_fa4_step(command, pool)
        causes = [(reason, [step if s == command else s for s in steps]) for reason, steps in causes]
    return causes or None


def _gpu_label(indices):
    """Returns "GPU 0" or "GPUs 0, 1"."""
    return f"GPU{'s' if len(indices) > 1 else ''} {', '.join(map(str, indices))}"


def fa4_warning(gpus, uses="This run uses"):
    """The loud warning for GPUs on which fa4 is not used for a reason the user can fix, or was not tested.

    Args:
        gpus: (GPU index, device, engine chosen for it) for each GPU of the run.
        uses: How the warning names the engine in use ("This run uses" or "deeprm call run would use").

    Returns:
        None if fa4_fixes gives no cause for any GPU that does not use fa4; else (one-line summary, framed warning
        text).
    """
    pool = [
        (index, torch.cuda.get_device_name(device), tuple(torch.cuda.get_device_capability(device)))
        for index, device, _ in gpus
        if device.type == "cuda"
    ]
    found = {}  # (GPU name, compute capability, engine) -> [GPU indices, causes]
    for index, device, engine in gpus:
        if engine == "fa4" or device.type != "cuda":
            continue
        key = (torch.cuda.get_device_name(device), tuple(torch.cuda.get_device_capability(device)), engine)
        if key not in found:
            found[key] = [[], fa4_fixes(device, pool)]  # once per GPU type
        found[key][0].append(index)
    found = {key: value for key, value in found.items() if value[1]}
    if not found:
        return None
    where = "; ".join(
        f"{name} (compute capability {cc[0]}.{cc[1]}; {_gpu_label(idx)})" for (name, cc, _), (idx, _) in found.items()
    )
    if len(found) > 1:  # several GPU types: the not-tested reason names the GPUs it is about
        generic = _not_tested_cause()
        for (name, _, _), (idx, fixes) in found.items():
            fixes[:] = [_not_tested_cause(f"{name} ({_gpu_label(idx)})") if c == generic else c for c in fixes]
    engine = next(iter(found))[2]
    causes = []
    for _, fixes in found.values():
        causes += [cause for cause in fixes if cause not in causes]
    steps = []
    for _, cause_steps in causes:
        steps += [step for step in cause_steps if step not in steps]
    reasons = [reason for reason, _ in causes]
    # On a GPU fa4 was not tested on, auto keeps the same engine after the steps: the `deeprm check` line is left out
    # where the warning names such a GPU, and the steps "try" fa4 where it names no tested GPU.
    tested = [cc in FA4_AUTO_CCS for _, cc, _ in found]
    note = check_deps.pip_note() if any(" -m pip " in step for step in steps) else ""
    lines = [
        FA4_WARNING_RULE,
        f"WARNING: fa4 is not used on {where}.",
        f"{uses} the {engine} engine instead, because:",
        *(f"  - {reason}" for reason in reasons),
        "To use fa4, do these steps, then run DeepRM again:" if any(tested) else "To try fa4, do these steps:",
        *(f"  {i}. {step}" for i, step in enumerate(steps, 1)),
        *([f"Note: {note}."] if note else []),
        *(["`deeprm check` shows the engine DeepRM will then use."] if all(tested) else []),
        f"To keep {engine} without this warning, run deeprm call run with --engine {engine}.",
        FA4_WARNING_RULE,
    ]
    return f"fa4 is not used on {where}: {'; '.join(reasons)}.", "\n".join(lines)
