"""Tests for DeepRM v1.2 (deeprm_v12): weight file, engines, inference and training. CPU only."""

import argparse
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deeprm.inference import inference  # noqa: E402
from deeprm.model import deeprm_v12  # noqa: E402
from deeprm.model import deeprm_v12_engines as engines  # noqa: E402

WEIGHTS = pathlib.Path(inference.__file__).parent.parent / "weight" / "deeprm_weights.pt"
N_PARAMS = 15_908_609


@pytest.fixture(scope="module")
def save_dict():
    return torch.load(WEIGHTS, map_location="cpu", weights_only=True)


def _tensor_digest(state):
    """sha256 over every tensor's name, dtype, shape and bytes, in name order."""
    import hashlib

    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous().reshape(-1)
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(state[name].shape)).encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def test_weight_file_contents(save_dict):
    assert set(save_dict) == {"model_state_dict", "model_config", "deeprm_meta"}
    state = save_dict["model_state_dict"]
    assert len(state) == 100
    assert sum(t.numel() for n, t in state.items() if "running_" not in n and "num_batches" not in n) == N_PARAMS
    cfg = save_dict["model_config"]
    assert cfg["model_type"] == engines.ARCHITECTURE == "deeprm_v12"
    assert cfg["output_logits"] is True and cfg["no_bq"] is True and cfg["no_dwell"] is True
    assert (cfg["enc_dim"], cfg["enc_layer"], cfg["head"]) == (512, 6, 8)
    meta = save_dict["deeprm_meta"]
    assert meta["architecture"] == "deeprm_v12"
    # the stoichiometry calibration lives in the code, not in the weight file
    assert "stoich_rule" not in meta and "temperature" not in repr(save_dict["deeprm_meta"]) + repr(cfg)
    # the tensors of the verified v1.2 model, pinned by content (the file's md5 also covers the metadata)
    assert _tensor_digest(state) == "d046f6ebd933f361e93571e3c651b5e9c0aa8a5b3d93a5da3ce58331eb4b4b74"
    # only these fields: no internal run names, training classes or source checkpoints in the shipped file
    assert set(meta) == {"deeprm_version", "description", "architecture", "params", "trained_torch_version"}
    assert not any("/" in str(v) for k, v in meta.items() if k != "description"), "no paths in the shipped metadata"


def test_model_files_load_without_running_code(tmp_path):
    """A model file that holds Python objects other than tensors and plain values is refused (weights_only=True)."""
    path = tmp_path / "pickled.pt"
    config = {"model_type": "deeprm_v12"}
    torch.save({"model_state_dict": {}, "model_config": config, "args": argparse.Namespace(a=1)}, path)
    with pytest.raises(ValueError, match="weights_only=True"):
        inference.load_model_file(str(path))
    assert set(inference.load_model_file(str(WEIGHTS))) == {"model_state_dict", "model_config", "deeprm_meta"}


def test_model_name_accepts_only_deeprm_v12():
    assert inference._model_name({"model_type": "deeprm_v12"}) == "deeprm_v12"
    assert inference._model_name({"model": "deeprm_v12"}) == "deeprm_v12"
    # a file that names another module (e.g. a research checkpoint), or none, needs --model-type deeprm_v12
    other = {"model_type": "other_module"}
    assert inference._model_name(other, "deeprm_v12") == "deeprm_v12"
    assert inference._model_name({}, "deeprm_v12", {"kmer_embedding.weight": None}) == "deeprm_v12"
    assert inference._choose_engine(other, "deeprm_v12", "eager", torch.device("cpu"))[:2] == ("eager", "deeprm_v12")
    with pytest.raises(ValueError, match="names the model module 'no_such_module'.*add --model-type deeprm_v12"):
        inference._model_name({"model_type": "no_such_module"})
    with pytest.raises(ValueError, match="does not name its model architecture.*add --model-type deeprm_v12"):
        inference._model_name({}, None, {"kmer_embedding.weight": None})
    with pytest.raises(ValueError, match="--model-type 'mt_v7'"):
        inference._choose_engine({"model_type": "deeprm_v12"}, "mt_v7", "auto", torch.device("cpu"))


def test_dwell_bq_dim():
    assert inference._dwell_bq_dim({}) == 3
    assert inference._dwell_bq_dim({"no_bq": True}) == 2
    assert inference._dwell_bq_dim({"no_bq": True, "no_dwell": True}) == 0


def test_engine_classes():
    expected = {"fa4": engines.Fa4Engine, "fast": engines.FastEngine, "eager": deeprm_v12.TransformerModel}
    assert engines.ENGINE_CLASSES == expected
    assert engines.Fa4Engine.FORM == "u2" and engines.FastEngine.FORM == "l12g3_56"
    assert [engines.engine_form(e) for e in ("fa4", "fast", "eager")] == ["fa4/u2", "fast/l12g3_56", "eager/reference"]
    assert engines.is_engine_model("deeprm_v12") and not engines.is_engine_model("no_such_module")
    assert set(engines.ENGINES) == {"auto", "fa4", "fast", "eager"}
    for cls in (engines.Fa4Engine, engines.FastEngine):
        assert issubclass(cls, deeprm_v12.TransformerModel)


def test_model_code_is_two_files_with_one_internal_import():
    """deeprm_v12.py imports nothing from deeprm; deeprm_v12_engines.py imports only deeprm_v12 from deeprm.model."""
    import ast

    model_dir = pathlib.Path(deeprm_v12.__file__).parent
    imported = {}
    for name in ("deeprm_v12", "deeprm_v12_engines"):
        tree = ast.parse((model_dir / f"{name}.py").read_text())
        modules = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported[name] = sorted(m for m in modules if m.startswith("deeprm"))
    assert imported["deeprm_v12"] == []
    assert [m for m in imported["deeprm_v12_engines"] if m.startswith("deeprm.model")] == ["deeprm.model.deeprm_v12"]
    sources = list(pathlib.Path(inference.__file__).parent.parent.rglob("*.py"))
    assert len(sources) > 20
    # the old project name, an internal training-class name and an internal run label, written in parts so this file
    # does not contain them
    old_names = ("ai" + "rna", "hard" + "task", "cm" + "p5")
    # the sdist ships the tests and the docs too, so they are scanned with the package code; a copy of tests/ run
    # against an installed package has no docs next to it
    repo = pathlib.Path(__file__).resolve().parent.parent
    tests = sorted((repo / "tests").rglob("*.py"))
    assert pathlib.Path(__file__).resolve() in tests
    texts = sources + tests
    if (repo / "docs").is_dir():
        docs = sorted(p for p in (repo / "docs").rglob("*") if p.suffix in (".md", ".py", ".rst", ".txt"))
        assert len(docs) > 5
        texts += docs + [repo / "README.md", repo / "CHANGELOG.md"]
    assert [str(p) for p in texts if any(name in p.read_text().lower() for name in old_names)] == []
    # the model package holds only the v1.2 model, and nothing imports the v1.1 model module
    assert sorted(p.name for p in model_dir.glob("*.py")) == ["__init__.py", "deeprm_v12.py", "deeprm_v12_engines.py"]
    old_import = "deeprm.model" + ".deeprm_model"
    assert [str(p) for p in sources if old_import in p.read_text() or "import deeprm_model" in p.read_text()] == []


def test_engine_selection_cpu():
    cpu = torch.device("cpu")
    assert engines.resolve_engine("auto", cpu)[0] == "fast"
    assert engines.resolve_engine("eager", cpu) == ("eager", "requested")
    with pytest.raises(ValueError):
        engines.resolve_engine("fa4", cpu)
    with pytest.raises(ValueError):
        engines.resolve_engine("nope", cpu)


@pytest.mark.parametrize(
    "cc, fa4_ok, compile_ok, expected",
    [
        ((10, 0), True, True, "fa4"),
        ((10, 0), False, True, "fast"),
        ((8, 0), True, True, "fa4"),  # A100
        ((9, 0), True, True, "fa4"),  # H100, H200
        ((10, 3), True, True, "fast"),  # B300: fa4 not tested there, so auto keeps fast
        ((12, 0), True, True, "fast"),  # RTX 50: the same
        ((7, 5), True, True, "fast"),  # T4: no FlashAttention-4 kernels
        ((6, 1), True, False, "eager"),
        ((10, 0), True, False, "eager"),
    ],
)
def test_engine_selection_gpu(monkeypatch, tmp_path, cc, fa4_ok, compile_ok, expected):
    _fake_gpu(monkeypatch, tmp_path, cc, fa4_ok=fa4_ok, compile_ok=compile_ok)
    assert engines.resolve_engine("auto", torch.device("cuda", 0))[0] == expected
    if not compile_ok:
        with pytest.raises(ValueError):
            engines.resolve_engine("fast", torch.device("cuda", 0))


def _fake_gpu(monkeypatch, tmp_path, cc, fa4_ok=True, compile_ok=True, name="Test GPU"):
    """Fake an NVIDIA GPU of compute capability `cc`."""
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: cc)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: name)
    monkeypatch.setattr(torch.version, "cuda", "13.0")  # fa4 needs CUDA 12.8 or newer; CI runs CPU-only PyTorch
    monkeypatch.setattr(engines, "fa4_status", lambda: (fa4_ok, "test"))
    monkeypatch.setattr(engines, "fa4_imports", lambda: (True, "test"))
    monkeypatch.setattr(engines, "compile_status", lambda device: (compile_ok, "test"))


def test_auto_uses_fa4_where_it_was_tested_and_never_on_amd(monkeypatch, tmp_path):
    """auto picks fa4 on the GPUs it was tested on (A100, H100, B200), fast on the other FlashAttention-4 GPUs, and
    never fa4 on AMD GPUs (ROCm reports them as "cuda")."""
    gpu = torch.device("cuda", 0)
    for cc in ((8, 0), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0)):
        _fake_gpu(monkeypatch, tmp_path, cc)
        engine, reason = engines.resolve_engine("auto", gpu)
        assert engine == ("fa4" if cc in engines.FA4_AUTO_CCS else "fast") and f"{cc[0]}.{cc[1]}" in reason
    _fake_gpu(monkeypatch, tmp_path, (9, 4), name="AMD Instinct MI300X")  # 9.4 passes the 8.0 check
    monkeypatch.setattr(torch.version, "hip", "7.0.51831")
    monkeypatch.setattr(torch.version, "cuda", None)
    engine, reason = engines.resolve_engine("auto", gpu)
    assert engine == "fast" and "ROCm" in reason
    with pytest.raises(ValueError, match="AMD"):
        engines.resolve_engine("fa4", gpu)
    with pytest.raises(RuntimeError, match="ROCm"):
        engines.fa4_preflight(gpu)
    assert engines.fa4_fixes(gpu) is None  # no fa4 warning on AMD GPUs
    _fake_gpu(monkeypatch, tmp_path, (7, 5), name="Tesla T4")
    assert engines.resolve_engine("auto", gpu)[0] == "fast"
    with pytest.raises(ValueError, match="8.0 or newer"):
        engines.resolve_engine("fa4", gpu)


def test_auto_uses_eager_without_a_c_compiler(monkeypatch):
    monkeypatch.delenv("TORCH_COMPILE_DISABLE", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    monkeypatch.setattr(engines, "_c_compiler", lambda: None)
    import torch.utils._triton as triton_utils

    monkeypatch.setattr(triton_utils, "has_triton", lambda: True)
    ok, reason = engines.compile_status(torch.device("cuda", 0))
    assert ok is False and "C compiler" in reason
    assert engines.resolve_engine("auto", torch.device("cuda", 0)) == ("eager", reason)


def test_auto_uses_eager_without_python_headers(monkeypatch, tmp_path):
    """Triton builds C modules that include Python.h at run time: without the headers, auto falls back to eager."""
    monkeypatch.delenv("TORCH_COMPILE_DISABLE", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    monkeypatch.setattr(engines, "_c_compiler", lambda: "/usr/bin/gcc")
    monkeypatch.setattr(engines, "_python_include", lambda: str(tmp_path))
    import torch.utils._triton as triton_utils

    monkeypatch.setattr(triton_utils, "has_triton", lambda: True)
    ok, reason = engines.compile_status(torch.device("cuda", 0))
    assert ok is False and "Python.h" in reason and str(tmp_path) in reason
    assert engines.resolve_engine("auto", torch.device("cuda", 0)) == ("eager", reason)
    (tmp_path / "Python.h").write_text("")
    assert engines.compile_status(torch.device("cuda", 0)) == (True, "ok")


def test_python_include_is_where_this_python_keeps_its_headers():
    import sysconfig

    assert engines._python_include() in {sysconfig.get_paths(scheme=s)["include"] for s in sysconfig.get_scheme_names()}


def test_compile_status_respects_torch_compile_disable(monkeypatch):
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "1")
    assert engines.compile_status(torch.device("cpu"))[0] is False
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "false")  # torch only acts on "1"
    assert engines.compile_status(torch.device("cpu"))[0] is True


def test_compile_status_without_dynamo(monkeypatch):
    real = engines.importlib.import_module

    def fake(name, *a, **k):
        if name == "torch._dynamo":
            raise ImportError("broken")
        return real(name, *a, **k)

    monkeypatch.setattr(engines.importlib, "import_module", fake)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    ok, reason = engines.compile_status(torch.device("cuda", 0))
    assert ok is False and "torch._dynamo" in reason
    assert engines.resolve_engine("auto", torch.device("cuda", 0))[0] == "eager"
    assert engines.resolve_engine("eager", torch.device("cuda", 0))[0] == "eager"


def test_fa4_status_checks_the_verified_build(monkeypatch, tmp_path):
    import importlib.util

    if importlib.util.find_spec("flash_attn") is not None:
        pytest.skip("a flash_attn package is installed here")
    cute = tmp_path / "flash_attn" / "cute"
    cute.mkdir(parents=True)
    (cute / "interface.py").write_text("# not the verified build\n")
    monkeypatch.setenv(engines.FA4_SITE_ENV, str(tmp_path))
    monkeypatch.delenv(engines.FA4_UNPINNED_ENV, raising=False)
    assert engines.fa4_status()[0] is False
    monkeypatch.setenv(engines.FA4_UNPINNED_ENV, "1")
    assert engines.fa4_status()[0] is True
    monkeypatch.delenv(engines.FA4_SITE_ENV)
    assert engines.fa4_status()[0] is False


def test_dwell_bq_channels():
    assert inference._dwell_bq_channels({"no_bq": True, "no_dwell": True}) is None
    assert inference._dwell_bq_channels({"no_bq": True}) == [0, 1]
    assert inference._dwell_bq_channels({"no_dwell": True}) == [2]
    assert inference._dwell_bq_channels({}) == [0, 1, 2]
    data = {k: torch.zeros(2, 3) for k in ("kmer_token", "signal_token", "segment_len")}
    data["dwell_bq_token"] = torch.arange(2 * 17 * 3, dtype=torch.float32).reshape(2, 17, 3)
    assert inference.to_device(data, torch.device("cpu"), None)[3] is None
    assert inference.to_device(data, torch.device("cpu"), [0, 1])[3].shape == (2, 17, 2)
    assert torch.equal(inference.to_device(data, torch.device("cpu"), [2])[3][..., 0], data["dwell_bq_token"][..., 2])


def _synthetic_inputs(n, seed=0):
    rng = np.random.default_rng(seed)
    kmer = rng.choice(np.frombuffer(b"ACGT", dtype=np.uint8), size=(n, 21))
    kmer[:, 10] = ord("A")
    seg = rng.integers(3, 12, size=(n, 17)).astype(np.int32)
    signal = rng.normal(size=(n, 1224)).astype(np.float32)
    return torch.from_numpy(kmer.astype(np.int32)), torch.from_numpy(signal), torch.from_numpy(seg)


@pytest.mark.parametrize("engine", ["fa4", "fast", "eager"])
def test_v12_model_builds_strict_and_runs_on_cpu(save_dict, engine):
    import fa4_cpu_standin

    fa4_cpu_standin.enable()  # plain attention in place of FlashAttention-4, so fa4 runs on CPU (tests only)
    cfg = save_dict["model_config"]
    model = engines.ENGINE_CLASSES[engine](
        d_model=cfg["enc_dim"],
        n_heads=cfg["head"],
        d_ff=cfg["lin_dim"],
        n_layers=cfg["enc_layer"],
        lin_depth=cfg["lin_layer"],
        t_act=cfg["t_act"],
        lin_act=cfg["lin_act"],
        encoder_dropout=cfg["enc_dropout"],
        lin_dropout=cfg["lin_dropout"],
        kmer_size=cfg["kmer_size"],
        signal_size=cfg["signal_size"],
        block_len=cfg["block_len"],
        seq_len=cfg["seq_len"],
        signal_stride=cfg["signal_stride"],
        dwell_bq_dim=inference._dwell_bq_dim(cfg),
        kmer_mode=cfg["kmer_mode"],
        n_class=cfg["n_class"],
        n_label=cfg["n_label"],
        output_logits=cfg["output_logits"],
    )
    model.load_state_dict(save_dict["model_state_dict"], strict=True)
    model.eval()
    with torch.inference_mode():
        out = model(*_synthetic_inputs(8), None)
    pred = inference._to_numpy_pred(out)
    assert pred.shape == (8,) and pred.dtype == np.float32 and np.isfinite(pred).all()


MODEL_KW = dict(
    d_model=512,
    n_heads=8,
    d_ff=1024,
    n_layers=6,
    lin_depth=6,
    t_act="gelu",
    lin_act="gelu",
    kmer_size=5,
    signal_size=30,
    block_len=17,
    seq_len=200,
    signal_stride=6,
    dwell_bq_dim=0,
    n_label=1,
    n_class=1,
    output_logits=True,
)


# Activation indices in regression_head.lin_layers at lin_depth 6 (the v1.2 weights).
# Layer indices are part of the weight file's tensor names, so the layout must not change.
HEAD_ACT_SLOTS = (2, 6, 10, 14, 18, 21, 23)
SMALL_KW = dict(MODEL_KW, d_model=64, n_heads=4, d_ff=128, n_layers=1)  # same head layout, quicker to build


@pytest.mark.parametrize(
    "name, cls",
    [("relu", torch.nn.ReLU), ("gelu", torch.nn.GELU), ("silu", torch.nn.SiLU), ("elu", torch.nn.ELU)],
)
def test_head_activation_names(name, cls):
    layers = deeprm_v12.TransformerModel(**dict(SMALL_KW, lin_act=name)).regression_head.lin_layers
    assert [i for i, layer in enumerate(layers) if isinstance(layer, cls)] == list(HEAD_ACT_SLOTS)
    assert [i for i, layer in enumerate(layers) if isinstance(layer, torch.nn.Linear)] == [0, 4, 8, 12, 16, 20, 22, 24]
    assert len({id(layers[i]) for i in HEAD_ACT_SLOTS}) == len(HEAD_ACT_SLOTS)  # a new module in every slot
    if name == "gelu":
        assert all(layers[i].approximate == "none" for i in HEAD_ACT_SLOTS)
    assert set(deeprm_v12.HEAD_ACTIVATIONS) == {"relu", "gelu", "silu", "elu"}


@pytest.mark.parametrize("name", ["swish", "GELU", "", None])
def test_head_activation_unknown(name):
    with pytest.raises(ValueError, match="lin_act"):
        deeprm_v12.TransformerModel(**dict(SMALL_KW, lin_act=name))


def test_bundled_head_is_gelu(save_dict):
    model = inference._build_model(deeprm_v12.TransformerModel, save_dict["model_config"])
    model.load_state_dict(save_dict["model_state_dict"], strict=True)
    assert save_dict["model_config"]["lin_act"] == "gelu" and len(model.state_dict()) == 100
    layers = model.regression_head.lin_layers
    assert all(type(layers[i]) is torch.nn.GELU and layers[i].approximate == "none" for i in HEAD_ACT_SLOTS)


def test_lin_act_default():
    """The default head activation is gelu, as in the v1.2 weights."""
    import inspect

    assert inspect.signature(deeprm_v12.TransformerModel).parameters["lin_act"].default == "gelu"
    kw = dict(SMALL_KW)
    kw.pop("lin_act")
    layers = deeprm_v12.TransformerModel(**kw).regression_head.lin_layers
    assert all(type(layers[i]) is torch.nn.GELU for i in HEAD_ACT_SLOTS)
    assert _train_args().lin_act == "gelu"


def _engine(save_dict, engine):
    model = engines.ENGINE_CLASSES[engine](**MODEL_KW)
    model.load_state_dict(save_dict["model_state_dict"], strict=True)
    return model.eval()


def test_fast_matches_eager_on_cpu(save_dict):
    """The fast engine matches the reference model on CPU, up to rounding."""
    preds = []
    for engine in ("fast", "eager"):
        with torch.inference_mode():
            logits = _engine(save_dict, engine)(*_synthetic_inputs(16, 1), None)
        preds.append(inference.sigmoid_np(inference._to_numpy_pred(logits)))
    assert np.max(np.abs(preds[0] - preds[1])) < 1e-4


def test_fa4_engine_matches_the_reference_on_cpu(save_dict):
    """With plain attention standing in for FlashAttention-4, the uncompiled fa4 engine matches the others on CPU up to
    rounding (packing the tokens reorders sums)."""
    import fa4_cpu_standin

    fa4_cpu_standin.enable()
    kmer, signal, seg = _synthetic_inputs(64, 3)
    seg[0] = 0  # a row with no signal
    seg[1, :8] = 0  # a row with empty segments
    seg[2] = 11
    seg[3, 8] = 60  # a long centre base (several centre windows)
    logits = {}
    for engine in ("fa4", "fast", "eager"):
        model = _engine(save_dict, engine)
        with torch.inference_mode():
            logits[engine] = inference._to_numpy_pred(model(kmer, signal, seg, None))
            if engine == "fa4":
                one = inference._to_numpy_pred(model(kmer[5:6], signal[5:6], seg[5:6], None))
                tail = inference._to_numpy_pred(model(kmer[-3:], signal[-3:], seg[-3:], None))
                empty = model(kmer[:0], signal[:0], seg[:0], None)
    ok = ~np.isnan(logits["eager"])
    assert ok.sum() > 50
    for engine in ("fa4", "fast"):
        assert np.array_equal(np.isnan(logits[engine]), np.isnan(logits["eager"]))
        assert np.max(np.abs(logits[engine][ok] - logits["eager"][ok])) < 1e-4
    assert np.max(np.abs(one - logits["fa4"][5:6])) < 1e-4
    assert np.max(np.abs(tail[ok[-3:]] - logits["fa4"][-3:][ok[-3:]])) < 1e-4
    assert tuple(empty.shape) == (0, 1)


def test_engines_refuse_inputs_they_were_not_written_for(save_dict):
    import fa4_cpu_standin

    fa4_cpu_standin.enable()
    kmer, signal, seg = _synthetic_inputs(4, 4)
    for engine in ("fa4", "fast"):
        model = _engine(save_dict, engine)
        with torch.inference_mode(), pytest.raises(ValueError, match="k-mer"):
            model(torch.cat([kmer, kmer[:, :1]], 1), signal, seg, None)  # 22 bases for 17 segments
        with torch.inference_mode(), pytest.raises(ValueError, match="dwell"):
            model(kmer, signal, seg, torch.zeros(4, 17, 3))
        model.train()
        with pytest.raises(RuntimeError, match="inference only"):
            model(kmer, signal, seg, None)
    with pytest.raises(ValueError, match="5-mer"):
        deeprm_v12.TransformerModel(**dict(MODEL_KW, kmer_mode="1mer"))


def test_set_batch_size_fixes_the_centre_buffer(save_dict):
    model = _engine(save_dict, "fa4")
    model.set_batch_size(10_000)
    model.set_batch_size(256)  # never shrinks
    assert model._state.max_batch == 10_000
    assert engines.centre_slot_block(10_000) == 81_920


def test_sigmoid_np_matches_torch():
    x = np.linspace(-30, 30, 1001, dtype=np.float32)
    assert np.allclose(inference.sigmoid_np(x), torch.sigmoid(torch.from_numpy(x)).numpy(), atol=1e-7)


def _write_prep(path, n=48, seed=0):
    rng = np.random.default_rng(seed)
    kmer, signal, seg = _synthetic_inputs(n, seed)
    np.savez(
        path,
        segment_len_arr=seg.numpy().astype(np.uint16),
        signal_token=signal.numpy(),
        kmer_token=kmer.numpy().astype(np.uint8),
        dwell_motor_token=rng.random((n, 17)).astype(np.float32),
        dwell_pore_token=rng.random((n, 17)).astype(np.float32),
        bq_token=rng.integers(0, 40, size=(n, 17)).astype(np.uint8),
        label_id=(np.arange(n, dtype=np.int64) + 1),
        read_id=rng.integers(0, 2**62, size=(n, 2), dtype=np.int64),
    )


def _run_args(data, out, model, engine="auto", model_type=None):
    return argparse.Namespace(
        data=str(data),
        output=str(out),
        model=str(model),
        model_type=model_type,
        engine=engine,
        batch=20,
        num_gpu=0,
        gpu_pool=[],
        prefetch=1,
        worker=0,
        flush=100,
        resume=False,
        output_id=None,
    )


def test_inference_worker_v12_writes_pred(tmp_path):
    data = tmp_path / "prep"
    data.mkdir()
    _write_prep(data / "0.npz")
    out = tmp_path / "out"
    out.mkdir()
    args = _run_args(data, out, WEIGHTS)
    info = inference.write_run_info(args)
    assert info["model_name"] == "deeprm_v12" and info["device_type"] == "cpu"
    text = (out / inference.RUN_INFO_NAME).read_text()
    assert "stoich" not in text and "temperature" not in text
    inference.inference_worker(0, vars(args))
    with np.load(out / "inference_0_0.npz") as z:
        assert set(z.files) == {"label_id", "read_id", "pred"}  # the v1.1 per-read format
        pred = z["pred"]
    assert pred.shape == (48,) and pred.dtype == np.float32
    assert (pred >= 0).all() and (pred <= 1).all()


def test_call_run_defaults_and_removed_options():
    parser = argparse.ArgumentParser()
    inference.add_arguments(parser)
    args = parser.parse_args(["-i", "x", "-b", "y.bam", "-o", "z"])
    assert args.batch == 16000 and args.engine == "auto"
    for removed in (["--stoich-rule", "eq14"], ["--stoich-temperature", "1.0"]):
        with pytest.raises(SystemExit):
            parser.parse_args(["-i", "x", "-b", "y.bam", "-o", "z", *removed])
    from deeprm.inference import inference_preprocess_python

    prep = argparse.ArgumentParser()
    inference_preprocess_python.add_arguments(prep)
    assert prep.parse_args(["-p", "a", "-b", "b.bam", "-o", "c"]).qcut == 0


def test_unreadable_input_shard_is_skipped_with_a_warning(tmp_path, caplog):
    data = tmp_path / "prep"
    data.mkdir()
    _write_prep(data / "0.npz", n=12)
    (data / "1.npz").write_bytes(b"not an npz file")
    _write_prep(data / "2.npz", n=12, seed=1)
    out = tmp_path / "out"
    out.mkdir()
    args = _run_args(data, out, WEIGHTS)
    inference.write_run_info(args)
    from deeprm.inference import inference_dataloader

    inference_dataloader.READ_RETRIES, retries = 1, inference_dataloader.READ_RETRIES
    try:
        with caplog.at_level("WARNING"):
            inference.inference_worker(0, vars(args))
    finally:
        inference_dataloader.READ_RETRIES = retries
    # the other shards keep their numbers, so --resume can fill the gap once the shard is re-created
    assert sorted(p.name for p in out.glob("inference_*.npz")) == ["inference_0_0.npz", "inference_0_2.npz"]
    assert "Skipping input shard" in caplog.text and "1.npz" in caplog.text


def test_out_of_memory_runs_the_batch_in_smaller_parts(monkeypatch, caplog):
    calls = []

    def model(kmer, signal, seg_len, dwell_bq):  # a GPU that holds at most 5 rows at a time
        calls.append(kmer.shape[0])
        if kmer.shape[0] > 5:
            raise torch.OutOfMemoryError("CUDA out of memory (test)")
        return kmer[:, :1].float() * 2

    monkeypatch.setattr(inference, "to_device", lambda data, device, ch: (data["kmer_token"], None, None, None))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    batch = {"kmer_token": torch.arange(16).reshape(16, 1), "label_id": torch.arange(16)}
    fit = {"rows": 16}
    with caplog.at_level("WARNING"):
        pred = inference._predict(model, batch, torch.device("cuda"), None, None, fit)
    assert np.array_equal(pred, np.arange(16) * 2.0) and fit["rows"] == 4
    assert caplog.text.count("too little memory") == 1  # 16 -> 8 -> 4, reported once
    assert "The results stay the same apart from small float differences." in caplog.text
    calls.clear()
    inference._predict(model, batch, torch.device("cuda"), None, None, fit)
    assert calls == [4, 4, 4, 4], "the smaller size is kept for the rest of the run"
    with pytest.raises(torch.OutOfMemoryError):  # on the CPU an out-of-memory error is not retried
        inference._predict(model, batch, torch.device("cpu"), None, None, {"rows": 16})


def test_cpu_runs_a_batch_in_parts(monkeypatch):
    calls = []

    def model(kmer, signal, seg_len, dwell_bq):
        calls.append(kmer.shape[0])
        return kmer[:, :1].float()

    monkeypatch.setattr(inference, "to_device", lambda data, device, ch: (data["kmer_token"], None, None, None))
    n = inference.CPU_ROWS * 2 + 5
    batch = {"kmer_token": torch.arange(n).reshape(n, 1), "label_id": torch.arange(n)}
    pred = inference._predict(model, batch, torch.device("cpu"), None, None, {"rows": inference.CPU_ROWS})
    assert np.array_equal(pred, np.arange(n, dtype=np.float32)) and calls == [inference.CPU_ROWS] * 2 + [5]


def test_run_stops_when_no_shard_can_be_read(tmp_path):
    data = tmp_path / "prep"
    data.mkdir()
    (data / "0.npz").write_bytes(b"not an npz file")
    out = tmp_path / "out"
    out.mkdir()
    args = _run_args(data, out, WEIGHTS)
    inference.write_run_info(args)
    from deeprm.inference import inference_dataloader

    inference_dataloader.READ_RETRIES, retries = 1, inference_dataloader.READ_RETRIES
    try:
        with pytest.raises(RuntimeError, match="could use none"):
            inference.inference_worker(0, vars(args))
    finally:
        inference_dataloader.READ_RETRIES = retries


def test_resume_refuses_changed_input_files(tmp_path):
    data = tmp_path / "prep"
    data.mkdir()
    _write_prep(data / "0.npz", n=12)
    _write_prep(data / "1.npz", n=12, seed=1)
    args = _run_args(data, tmp_path, WEIGHTS)
    assert inference.write_run_info(args)["inputs"]["count"] == 2
    args.resume = True
    inference.write_run_info(args)  # same files: accepted
    (data / "1.npz").unlink()  # removing a shard renumbers the later ones, so a resume would skip or repeat some
    with pytest.raises(ValueError, match="holds other files"):
        inference.write_run_info(args)
    _write_prep(data / "1.npz", n=12, seed=1)  # re-created under the same name: accepted
    inference.write_run_info(args)


def test_output_id_outside_the_model_outputs_is_refused(tmp_path):
    data = tmp_path / "prep"
    data.mkdir()
    _write_prep(data / "0.npz", n=12)
    args = _run_args(data, tmp_path, WEIGHTS)
    args.output_id = 1
    with pytest.raises(ValueError, match="--output-id 1"):
        inference.write_run_info(args)


def test_resume_refuses_another_device_type(tmp_path):
    import json

    data = tmp_path / "prep"
    data.mkdir()
    _write_prep(data / "0.npz", n=12)
    args = _run_args(data, tmp_path, WEIGHTS)
    info = inference.write_run_info(args)
    info["device_type"] = "cuda"  # as if the first part ran on one GPU
    (tmp_path / inference.RUN_INFO_NAME).write_text(json.dumps(info))
    args.resume = True
    with pytest.raises(ValueError, match="made on the CUDA"):
        inference.write_run_info(args)


def test_explicit_engine_is_recorded(tmp_path):
    import json

    data = tmp_path / "prep"
    data.mkdir()
    _write_prep(data / "0.npz", n=12)
    out = tmp_path / "out"
    out.mkdir()
    args = _run_args(data, out, WEIGHTS, engine="eager", model_type="deeprm_v12")
    inference.write_run_info(args)
    inference.inference_worker(0, vars(args))
    info = json.loads((out / inference.RUN_INFO_NAME).read_text())
    assert info["model_name"] == "deeprm_v12"
    assert info["engine_used"]["engine"] == "eager" and info["engine_used"]["form"] == "eager/reference"


@pytest.mark.parametrize("second, common", [("fast", "fast/l12g3_56"), ("eager", "eager/reference")])
def test_mixed_gpu_pool_uses_a_common_engine(monkeypatch, tmp_path, caplog, second, common):
    """GPUs whose auto choices differ (a B200 next to an older GPU) all run one engine that every GPU can run."""

    def by_device(requested, device):
        if requested != "auto":
            return requested, "requested"
        return ("fa4", "t") if device.index == 0 else (second, "t")

    monkeypatch.setattr(engines, "resolve_engine", by_device)
    args = _run_args(tmp_path, tmp_path, WEIGHTS)
    args.num_gpu, args.gpu_pool = 2, [0, 1]
    with caplog.at_level("WARNING"):
        info = inference.write_run_info(args)
    assert info["form"] == common and "would run different engines" in caplog.text
    assert args.engine_resolved[0] == common.split("/")[0]  # the GPU processes run what the run file records


def test_fa4_engine_needs_its_gpu(tmp_path):
    """--engine fa4 is refused on CPU, before the run file is written and again by the worker."""
    data = tmp_path / "prep"
    data.mkdir()
    _write_prep(data / "0.npz", n=12)
    out = tmp_path / "out"
    out.mkdir()
    args = _run_args(data, out, WEIGHTS, engine="fa4")
    with pytest.raises(ValueError, match="CUDA GPU"):
        inference.write_run_info(args)
    assert not (out / inference.RUN_INFO_NAME).exists()
    with pytest.raises(ValueError, match="CUDA GPU"):
        inference.inference_worker(0, vars(args))


def test_resume_refuses_other_gpu_count_and_engine(tmp_path):
    data = tmp_path / "prep"
    data.mkdir()
    _write_prep(data / "0.npz", n=12)
    out = tmp_path / "out"
    out.mkdir()
    args = _run_args(data, out, WEIGHTS, engine="eager")
    inference.write_run_info(args)
    inference.inference_worker(0, vars(args))
    args.resume = True
    args.num_gpu = 2
    with pytest.raises(ValueError, match="GPU processes"):
        inference.write_run_info(args)
    args.num_gpu = 0
    inference.write_run_info(args)  # same model, same ranks: accepted, engine kept
    args.engine = "fast"
    before = (out / inference.RUN_INFO_NAME).read_text()
    with pytest.raises(ValueError, match="--engine eager"):
        inference.write_run_info(args)  # refused before the run file is rewritten
    assert (out / inference.RUN_INFO_NAME).read_text() == before
    args.engine_resolved = None  # a worker started without the parent's choice resolves the engine itself
    with pytest.raises(ValueError, match="--engine eager"):
        inference.inference_worker(0, vars(args))  # and checks it against the recorded one


def test_resume_of_a_folder_from_an_earlier_build(tmp_path):
    """A folder from an earlier build (its run file names a module, not an engine form) is not resumed."""
    import json

    data = tmp_path / "prep"
    data.mkdir()
    _write_prep(data / "0.npz", n=12)
    out = tmp_path / "out"
    out.mkdir()
    args = _run_args(data, out, WEIGHTS, engine="eager")
    inference.write_run_info(args)
    inference.inference_worker(0, vars(args))
    path = out / inference.RUN_INFO_NAME
    args.resume = True
    for recorded in ("engine_used", "model_name only"):
        info = json.loads(path.read_text())
        info["model_name"] = "an_earlier_module"
        info.pop("form", None)  # earlier builds wrote no form
        if recorded == "engine_used":
            info["engine_used"] = {"engine": "eager", "module": "an_earlier_module"}
        else:
            info.pop("engine_used", None)
        path.write_text(json.dumps(info))
        before = path.read_text()
        with pytest.raises(ValueError, match="new output folder"):
            inference.write_run_info(args)
        assert path.read_text() == before


def test_resume_before_the_worker_recorded_its_engine(tmp_path):
    """A run stopped after writing deeprm_run.json but before its worker recorded the engine can be resumed."""
    data = tmp_path / "prep"
    data.mkdir()
    _write_prep(data / "0.npz", n=12)
    out = tmp_path / "out"
    out.mkdir()
    args = _run_args(data, out, WEIGHTS, engine="eager")
    assert inference.write_run_info(args)["form"] == "eager/reference"
    args.resume = True
    inference.write_run_info(args)  # accepted
    args.engine = "fast"
    with pytest.raises(ValueError, match="--engine eager"):
        inference.write_run_info(args)


def test_engines_compile_on_cpu_with_a_test_backend(save_dict):
    """compile_engine runs on CPU with the 'eager' test backend: same outputs as uncompiled, one graph for fa4."""
    import fa4_cpu_standin
    import torch._dynamo

    fa4_cpu_standin.enable()
    kmer, signal, seg = _synthetic_inputs(32, 6)
    for engine, graphs in (("fast", None), ("fa4", 1)):
        plain = _engine(save_dict, engine)
        compiled = _engine(save_dict, engine)
        compiled.compile_engine("cpu", backend="eager")
        torch._dynamo.reset()
        before = engines.dynamo_counts()[0]
        with torch.inference_mode():
            a = inference._to_numpy_pred(plain(kmer, signal, seg, None))
            b = inference._to_numpy_pred(compiled(kmer, signal, seg, None))
        assert np.array_equal(a, b, equal_nan=True)
        if graphs is not None:
            assert engines.dynamo_counts()[0] - before == graphs


def test_resume_refuses_another_model(tmp_path):
    data = tmp_path / "prep"
    data.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    args = _run_args(data, out, WEIGHTS)
    inference.write_run_info(args)
    other = tmp_path / "other.pt"
    other.write_bytes(b"not the same model")
    args.model, args.resume = str(other), True
    with pytest.raises(ValueError):
        inference.write_run_info(args)
    (out / inference.RUN_INFO_NAME).unlink()
    (out / "inference_0_0.npz").write_bytes(b"")
    args.model = str(WEIGHTS)
    with pytest.raises(ValueError):
        inference.write_run_info(args)


def test_v11_model_files_are_refused(tmp_path):
    """DeepRM v1.1 model files get a one-line error before the run file is written, even with --model-type
    deeprm_v12."""
    data = tmp_path / "prep"
    data.mkdir()
    _write_prep(data / "0.npz", n=12)
    files = {
        "v11_weights": {"model_state_dict": {"cnn_encoder.0.weight": torch.zeros(1)}, "model_config": {"enc_dim": 256}},
        "v11_train_run": {"model_state_dict": {"x": torch.zeros(1)}, "model_config": {"model_type": "deeprm_model"}},
    }
    for name, content in files.items():
        path = tmp_path / f"{name}.pt"
        torch.save(content, path)
        for model_type in (None, "deeprm_v12"):
            out = tmp_path / f"out_{name}_{model_type}"
            out.mkdir()
            args = _run_args(data, out, path, model_type=model_type)
            with pytest.raises(ValueError, match=r"DeepRM v1\.1 model .*Run it with DeepRM v1\.1") as refused:
                inference.write_run_info(args)
            assert "\n" not in str(refused.value)
            assert not (out / inference.RUN_INFO_NAME).exists()
            with pytest.raises(ValueError, match=r"DeepRM v1\.1"):
                inference.inference_worker(0, vars(args))
    parser = argparse.ArgumentParser()
    inference.add_arguments(parser)
    assert "--model-type" not in parser.format_help()  # documented in the usage guide only, like --engine
    base = ["-i", "x", "-b", "y.bam", "-o", "z"]
    inference._validate_args(parser.parse_args(base + ["-y", "deeprm_v12"]))
    with pytest.raises(ValueError, match="--model-type deeprm_model: .*DeepRM v1\\.1"):
        inference._validate_args(parser.parse_args(base + ["-y", "deeprm_model"]))


def _train_args(*extra):
    from deeprm.train import train

    parser = argparse.ArgumentParser()
    train.add_arguments(parser)
    return parser.parse_args(["--data", "d", "--output", "o", *extra])


def test_train_run_defaults_build_the_v12_model_and_call_run_loads_it(save_dict, tmp_path):
    """deeprm train run's defaults build the v1.2 model; a file in its save format loads strictly in deeprm call run
    and predicts as the bundled one."""
    from deeprm.train import train

    cfg = vars(train._deeprm_v12_args(_train_args()))
    for key, value in save_dict["model_config"].items():
        if key not in ("model", "drop_kmer_input", "train_mode"):
            assert cfg[key] == value, key
    model = inference._build_model(deeprm_v12.TransformerModel, cfg)  # what train.main_worker builds
    model.load_state_dict(save_dict["model_state_dict"], strict=True)
    ckpt = tmp_path / "trained.pt"
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": {}, "model_config": cfg}, ckpt)
    data = tmp_path / "prep"
    data.mkdir()
    _write_prep(data / "0.npz")
    preds = []
    for name, model_file in (("trained", ckpt), ("bundled", WEIGHTS)):
        out = tmp_path / name
        out.mkdir()
        args = _run_args(data, out, model_file)
        info = inference.write_run_info(args)
        assert info["model_name"] == "deeprm_v12"
        assert info["form"] == engines.engine_form(engines.resolve_engine("auto", torch.device("cpu"))[0])
        inference.inference_worker(0, vars(args))
        with np.load(out / "inference_0_0.npz") as z:
            preds.append(z["pred"])
    assert np.array_equal(preds[0], preds[1])


def test_train_feeds_deeprm_v12_one_logit_per_row():
    from deeprm.train import train
    from deeprm.train.train_dataloader import pad_collate

    small = ["--enc-dim", "32", "--head", "4", "--lin-dim", "64", "--enc-layer", "1", "--lin-layer", "2"]
    model_args = vars(train._deeprm_v12_args(_train_args(*small)))
    model = inference._build_model(deeprm_v12.TransformerModel, model_args).train()
    kmer, signal, seg = _synthetic_inputs(8)
    zeros = np.zeros(17, dtype=np.float32)
    rows = [((seg[i].numpy(), signal[i].numpy(), kmer[i].numpy(), zeros, zeros, zeros), float(i % 2)) for i in range(8)]
    source, target = pad_collate(rows, pad_to=200, signal_stride=6, kmer_size=5)
    output, target = train.Trainer._feed_model(argparse.Namespace(gpu_id="cpu", model=model), source, target)
    assert output.shape == target.shape == (8,)
    loss = torch.nn.BCEWithLogitsLoss()(output, target)
    loss.backward()
    assert torch.isfinite(loss)


def test_train_run_refusals_need_no_gpu(tmp_path):
    from deeprm.train import train

    v11 = tmp_path / "v11.pt"
    torch.save({"model_state_dict": {"cnn_encoder.0.weight": torch.zeros(1)}, "model_config": {}}, v11)
    for extra, pattern in (
        (["--model", "deeprm_model"], r"--model deeprm_model: .*DeepRM v1\.1"),
        (["--model", "mt_v7"], "trains only the v1.2 architecture"),
        (["--signal-size", "25"], "--signal-size 25 must equal"),
        (["--load-checkpoint", str(v11)], r"DeepRM v1\.1 model"),
    ):
        with pytest.raises(ValueError, match=pattern):
            train.main(_train_args(*extra))
    for removed in (["--loss", "BCE"], ["--no-bq"], ["--t-act", "relu"]):
        with pytest.raises(SystemExit):
            _train_args(*removed)
    with pytest.raises(SystemExit):  # --lin-act takes only the names of deeprm_v12.HEAD_ACTIVATIONS
        _train_args("--lin-act", "swish")
    weights_only = train._deeprm_v12_args(_train_args("--load-checkpoint", str(WEIGHTS)))
    assert weights_only.load_weight_only is True  # the bundled file has no optimizer state


def test_train_prep_options_follow_the_method():
    from deeprm.train import train_preprocess

    parser = argparse.ArgumentParser()
    train_preprocess.add_arguments(parser)
    args = train_preprocess._resolve_method_defaults(parser.parse_args(["-p", "a", "-b", "b.bam", "-o", "c"]))
    assert args.dag_method == "lev" and args.spacer_list == ["CGACAU", "CCAUUG", "AAGCGU", "GUAGUC"]
    assert (args.cb_per_bb, args.cb_pad, args.indel_tolerance, args.postfix, args.qcut) == (3, 10, 3, "data", 7)
    args = train_preprocess._resolve_method_defaults(
        parser.parse_args(["-p", "a", "-b", "b.bam", "-o", "c", "--method", "pw", "--max_token_len", "100"])
    )
    assert (args.cb_per_bb, args.cb_pad, args.indel_tolerance, args.max_token_len) == (1, 20, 25, 100)
    assert args.spacer_list is None and args.di_consensus == "batch"  # resolved later from the design (--di or not)
    with pytest.raises(ValueError, match="does not take"):
        train_preprocess._resolve_method_defaults(
            parser.parse_args(["-p", "a", "-b", "b.bam", "-o", "c", "--method", "pw", "--kst", "5"])
        )


def test_train_prep_cfg_and_output_checks(tmp_path):
    import json

    from deeprm.train import train_preprocess

    parser = argparse.ArgumentParser()
    train_preprocess.add_arguments(parser)
    out = tmp_path / "out"

    def parse(*extra):
        return parser.parse_args(["-p", "new.pod5", "-b", "new.bam", "-o", str(out), *extra])

    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"read_bq_cutoff": 7}))  # a v1.1 key
    with pytest.raises(ValueError, match="unknown settings read_bq_cutoff"):
        train_preprocess._merge_config(parse("--cfg", str(cfg)))
    # a previous run's args.json applies, but -p, -b and -o from the command line win
    cfg.write_text(json.dumps({"pod5": "old.pod5", "bam": "old.bam", "output": "old", "dag_method": "pw"}))
    args = train_preprocess._merge_config(parse("--cfg", str(cfg)))
    assert (args.pod5, args.bam, args.output, args.dag_method) == ("new.pod5", "new.bam", str(out), "pw")
    with pytest.raises(ValueError, match="--sample works only with --reuse-blocks-dir"):
        train_preprocess._check_new_output(parse("--sample", "10"))
    train_preprocess._check_new_output(parse())  # a new folder is fine
    (out / "data").mkdir(parents=True)
    (out / "data" / "0.npz").write_bytes(b"")
    with pytest.raises(ValueError, match="already holds the output of a run"):
        train_preprocess._check_new_output(parse())


def test_train_compile_checks_its_folders(tmp_path):
    from deeprm.train import train_compile

    parser = argparse.ArgumentParser()
    train_compile.add_arguments(parser)
    (tmp_path / "empty").mkdir()
    args = parser.parse_args(["-p", str(tmp_path / "empty"), "-n", str(tmp_path / "empty"), "-o", str(tmp_path / "a")])
    with pytest.raises(ValueError, match="holds no token files"):
        train_compile.main(args)
    (tmp_path / "b").mkdir()
    args = parser.parse_args(["-p", str(tmp_path / "empty"), "-n", str(tmp_path / "empty"), "-o", str(tmp_path / "b")])
    with pytest.raises(ValueError, match="already exists"):
        train_compile.main(args)


def test_train_compile_refuses_an_empty_split(tmp_path):
    from deeprm.train import train_compile

    for label, n in (("pos", 30), ("neg", 30)):
        (tmp_path / label).mkdir()
        rng = np.random.default_rng(len(label))
        np.savez(
            tmp_path / label / "0.npz",
            segment_len_arr=rng.integers(1, 12, size=(n, 17)).astype(np.uint16),
            signal_token=rng.normal(size=(n, 1224)).astype(np.float32),
            kmer_token=rng.integers(1, 5, size=(n, 21)).astype(np.int8),
            dwell_motor_token=rng.random((n, 17)).astype(np.float32),
            dwell_pore_token=rng.random((n, 17)).astype(np.float32),
            bq_token=rng.integers(0, 40, size=(n, 17)).astype(np.int8),
            block_score=np.ones(n, np.float32),
        )
    parser = argparse.ArgumentParser()
    train_compile.add_arguments(parser)
    args = parser.parse_args(
        ["-p", str(tmp_path / "pos"), "-n", str(tmp_path / "neg"), "-o", str(tmp_path / "out"), "-t", "1", "-c", "20"]
    )
    with pytest.raises(ValueError, match="No files were written for val"):
        train_compile.main(args)


def test_train_loader_len_does_not_raise():
    """len() of a NanoporeDataLoader returns its number of batches and does not raise."""
    from deeprm.train.train_dataloader import NanoporeDataLoader

    loader = object.__new__(NanoporeDataLoader)
    loader.dataset = argparse.Namespace(
        num_gpu=1,
        drop_last=True,
        pos_file_paths=["p"] * 4,
        neg_file_paths=["n"] * 4,
        class_ratio=1,
        disk_shard_size=10,
    )
    loader.num_workers = 2
    loader.batch_size = 5
    assert len(loader) == (2 + 2) * 10 // 5 * 2


def test_cli_prints_refusals_without_a_traceback(monkeypatch, capsys):
    import types

    from deeprm import cli

    def refuse(argv):
        raise ValueError("--engine fa4 needs a CUDA GPU (sm100 / B200). Use --engine fast on CPU.")

    monkeypatch.setitem(cli._HELP_REGISTRY, "call", dict(cli._HELP_REGISTRY["call"], module="fake_call"))
    monkeypatch.setitem(__import__("sys").modules, "fake_call", types.SimpleNamespace(entry=refuse))
    monkeypatch.delenv("DEEPRM_TRACEBACK", raising=False)
    assert cli.main(["call", "run"]) == 2
    err = capsys.readouterr().err
    assert "deeprm call: error: --engine fa4 needs a CUDA GPU" in err and "Traceback" not in err
    monkeypatch.setenv("DEEPRM_TRACEBACK", "1")
    with pytest.raises(ValueError):
        cli.main(["call", "run"])


def test_run_file_says_whether_the_site_rule_calibration_applies(save_dict, tmp_path):
    """deeprm_run.json records site_rule_calibration: true only for the bundled weights (compared by md5), so a model
    trained with deeprm train run does not get the v1.2 model's calibration."""
    import json
    import shutil

    data = tmp_path / "prep"
    data.mkdir()
    _write_prep(data / "0.npz", n=12)
    copy = tmp_path / "copy_of_the_bundled_file.pt"
    shutil.copyfile(WEIGHTS, copy)
    trained = tmp_path / "trained.pt"  # the same tensors, saved as deeprm train run saves a model
    torch.save({"model_state_dict": save_dict["model_state_dict"], "model_config": save_dict["model_config"]}, trained)
    for name, model, expected in (("bundled", WEIGHTS, True), ("copy", copy, True), ("trained", trained, False)):
        out = tmp_path / name
        out.mkdir()
        info = inference.write_run_info(_run_args(data, out, model))
        recorded = json.loads((out / inference.RUN_INFO_NAME).read_text())
        assert info["site_rule_calibration"] is expected and recorded["site_rule_calibration"] is expected, name


def test_the_pinned_md5_is_that_of_the_bundled_weights():
    import hashlib

    from deeprm.utils.utils import BUNDLED_WEIGHTS_MD5

    assert hashlib.md5(WEIGHTS.read_bytes()).hexdigest() == BUNDLED_WEIGHTS_MD5


def test_resume_refuses_another_output_id(tmp_path):
    """The per-read files of a resumed run must hold the same model output as the files already written."""
    data = tmp_path / "prep"
    data.mkdir()
    _write_prep(data / "0.npz", n=12)
    two = tmp_path / "two_outputs.pt"
    torch.save({"model_state_dict": {}, "model_config": {"model_type": "deeprm_v12", "n_label": 2}}, two)
    out = tmp_path / "out"
    out.mkdir()
    args = _run_args(data, out, two)
    args.output_id = 0
    assert inference.write_run_info(args)["output_id"] == 0
    np.savez(out / "inference_0_0.npz", pred=np.zeros(1, np.float32))  # an output of that run
    args.resume, args.output_id = True, 1
    with pytest.raises(ValueError, match="made with --output-id 0; this run has --output-id 1"):
        inference.write_run_info(args)
    args.output_id = 0
    inference.write_run_info(args)  # the same output id resumes
    assert not list(out.glob("*.part"))


def test_a_cut_off_run_file_is_refused_on_resume(tmp_path):
    data = tmp_path / "prep"
    data.mkdir()
    _write_prep(data / "0.npz", n=12)
    out = tmp_path / "out"
    out.mkdir()
    (out / inference.RUN_INFO_NAME).write_text('{"model_md5": "abc')
    args = _run_args(data, out, WEIGHTS)
    args.resume = True
    with pytest.raises(ValueError, match="is not a complete run record"):
        inference.write_run_info(args)


def test_a_new_run_refuses_a_folder_with_per_read_outputs(tmp_path):
    """Without --resume, per-read files of an earlier run would stay and be added up by the pileup."""
    data = tmp_path / "prep"
    data.mkdir()
    _write_prep(data / "0.npz", n=12)
    _write_prep(data / "1.npz", n=12, seed=1)
    out = tmp_path / "out"
    out.mkdir()
    args = _run_args(data, out, WEIGHTS, engine="eager")
    inference.write_run_info(args)
    inference.inference_worker(0, vars(args))
    done = ["inference_0_0.npz", "inference_0_1.npz"]
    assert sorted(p.name for p in out.glob("inference_*.npz")) == done
    before = (out / inference.RUN_INFO_NAME).read_text()
    with pytest.raises(ValueError, match="use --resume, or a new or empty output folder") as refused:
        inference.write_run_info(args)
    assert str(out) in str(refused.value) and "\n" not in str(refused.value)
    assert (out / inference.RUN_INFO_NAME).read_text() == before
    # --resume still continues the run: the missing shard is made again and the finished one is kept
    (out / "inference_0_1.npz").unlink()
    kept = (out / "inference_0_0.npz").stat().st_mtime_ns
    args.resume = True
    inference.write_run_info(args)
    inference.inference_worker(0, vars(args))
    assert sorted(p.name for p in out.glob("inference_*.npz")) == done
    assert (out / "inference_0_0.npz").stat().st_mtime_ns == kept


def test_a_run_stopped_while_writing_leaves_no_partial_per_read_file(tmp_path, monkeypatch):
    """A per-read file is written under a .part name and renamed when complete, so a stopped run never leaves a cut-off
    .npz file, which the pileup would fail on and --resume would count as done. --resume removes the .part files."""
    import io

    class Stopped(Exception):
        pass

    savez = np.savez_compressed

    def stopped_half_way(file, **arrays):
        whole = io.BytesIO()
        savez(whole, **arrays)
        half = whole.getvalue()[: len(whole.getvalue()) // 2]
        if hasattr(file, "write"):
            file.write(half)
        else:
            pathlib.Path(file).write_bytes(half)
        raise Stopped

    data = tmp_path / "prep"
    data.mkdir()
    _write_prep(data / "0.npz", n=12)
    out = tmp_path / "out"
    out.mkdir()
    args = _run_args(data, out, WEIGHTS, engine="eager")
    inference.write_run_info(args)
    with monkeypatch.context() as patch:
        patch.setattr(np, "savez_compressed", stopped_half_way)
        with pytest.raises(Stopped):
            inference.inference_worker(0, vars(args))
    assert list(out.glob("*.npz")) == []
    args.resume = True
    inference.write_run_info(args)
    inference.inference_worker(0, vars(args))
    assert sorted(p.name for p in out.iterdir()) == ["deeprm_run.json", "inference_0_0.npz"]
    with np.load(out / "inference_0_0.npz") as z:
        assert z["pred"].shape == (12,)
    # only this rank's .part files are removed
    (out / "inference_0_1.npz.part").write_bytes(b"cut off")
    (out / "inference_1_0.npz.part").write_bytes(b"another rank")
    assert inference._discover_resume_point(str(out), 0) == 1
    assert not (out / "inference_0_1.npz.part").exists() and (out / "inference_1_0.npz.part").exists()


def test_model_files_load_onto_the_device_of_each_process(tmp_path, monkeypatch):
    """A model file saved on cuda:2 (rank 0 of deeprm train run --gpu-pool 2 3) loads onto the GPU of each process,
    in deeprm call run and in deeprm train run --load-checkpoint. Simulated on the CPU: the tensors are tagged cuda:2
    when saved, and the device each tensor would load onto is recorded instead of moving it there."""
    import torch.serialization as serialization

    from deeprm.train import train

    path = tmp_path / "saved_on_gpu2.pt"
    with monkeypatch.context() as patch:
        patch.setattr(serialization, "location_tag", lambda storage: "cuda:2")
        torch.save({"model_state_dict": {"w": torch.ones(2)}, "model_config": {"model_type": "deeprm_v12"}}, path)
    targets = []

    def record(storage, location):  # the tensor stays on the CPU
        targets.append(location)
        return storage

    class Loaded(Exception):
        pass

    load = inference.load_model_file

    def load_then_stop(file, map_location="cpu"):
        load(file, map_location=map_location)
        raise Loaded

    monkeypatch.setattr(serialization, "default_restore_location", record)
    monkeypatch.setattr(inference, "load_model_file", load_then_stop)
    monkeypatch.setattr(train, "load_model_file", load_then_stop)
    args = _run_args(tmp_path, tmp_path, path)
    # GPU numbers that differ from the rank and from the saved cuda:2
    args.num_gpu, args.gpu_pool = 2, [0, 3]
    devices_set = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: devices_set.append(str(device)))
    with pytest.raises(Loaded):
        inference.inference_worker(1, vars(args))  # rank 1 runs on cuda:3
    assert targets == ["cuda:3"]
    assert devices_set == ["cuda:3"]  # its own GPU is the process's default, before anything touches CUDA

    class Model:
        def to(self, device):
            return self

    monkeypatch.setattr(train, "setup_ddp", lambda *args: None)
    monkeypatch.setattr(train, "_build_model", lambda model_class, config: Model())
    targets.clear()
    with pytest.raises(Loaded):
        train.main_worker(1, {"gpu_pool": [1, 3], "num_gpu": 2, "load_checkpoint": str(path)})  # rank 1: cuda:3
    assert targets == ["cuda:3"]


def test_an_interrupted_gpu_rank_is_an_error(tmp_path, monkeypatch):
    """torch's spawn takes a KeyboardInterrupt in a rank for a clean exit, so a SIGINT that reaches only one rank would
    let the pileup run on incomplete outputs; inference_worker turns it into an error. On the CPU (no spawned ranks)
    a Ctrl-C stays a KeyboardInterrupt."""

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(inference, "load_model_file", interrupted)
    args = _run_args(tmp_path, tmp_path, tmp_path / "model.pt")
    args.num_gpu, args.gpu_pool = 1, [0]
    try:
        with pytest.raises(RuntimeError, match="Rank 0 was interrupted"):
            inference.inference_worker(0, vars(args))
    except KeyboardInterrupt:
        pytest.fail("a KeyboardInterrupt escaped from a GPU rank (torch's spawn would report it as a clean exit)")
    args.num_gpu, args.gpu_pool = 0, []
    with pytest.raises(KeyboardInterrupt):
        inference.inference_worker(0, vars(args))

    # A plain Ctrl-C gives the short message alone; a Ctrl-C while the rank was stopping on an error keeps that error.
    args.num_gpu, args.gpu_pool = 1, [0]
    with pytest.raises(RuntimeError) as plain:
        inference.inference_worker(0, vars(args))
    assert plain.value.__cause__ is None and "error first" not in str(plain.value)

    def interrupted_after_an_error(*args, **kwargs):
        error = ValueError("A segment length is negative.")
        interrupt = KeyboardInterrupt()
        interrupt.stopped_on = error  # as the pipelined loop records it (test_writer_close runs the loop itself)
        raise interrupt

    monkeypatch.setattr(inference, "load_model_file", interrupted_after_an_error)
    with pytest.raises(RuntimeError, match="Rank 0 was interrupted") as after_error:
        inference.inference_worker(0, vars(args))
    assert "It had stopped on this error first: ValueError('A segment length is negative.')" in str(after_error.value)
    assert isinstance(after_error.value.__cause__, ValueError)

    def interrupted_while_handling_an_error(*args, **kwargs):
        try:
            raise torch.OutOfMemoryError("CUDA out of memory (test)")
        except torch.OutOfMemoryError:
            raise KeyboardInterrupt  # a Ctrl-C during the recovery: the error is its context, not one it stopped on

    monkeypatch.setattr(inference, "load_model_file", interrupted_while_handling_an_error)
    with pytest.raises(RuntimeError, match="Rank 0 was interrupted") as recovering:
        inference.inference_worker(0, vars(args))
    assert recovering.value.__cause__ is None and "error first" not in str(recovering.value)


def test_call_run_reads_the_driver_as_deeprm_check_does(monkeypatch):
    """A CPU-only PyTorch: deeprm call run reads the driver from libcuda first, as deeprm check does. So it warns in a
    container without nvidia-smi and after a driver update without a reboot, and stays quiet where nvidia-smi finds
    no GPU or there is no driver."""
    from deeprm.utils import check_deps

    smi = {}  # what the fake nvidia-smi prints; none: nvidia-smi is missing
    monkeypatch.setattr(check_deps, "_nvidia_smi", lambda *args: smi.get("out"))
    monkeypatch.setattr(check_deps, "libcuda_cuda_version", lambda: (13, 0))
    monkeypatch.setattr(inference.torch.version, "cuda", None)
    monkeypatch.setattr(inference.platform, "machine", lambda: "x86_64")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    # no nvidia-smi (a container started without the "utility" capability): libcuda reads the driver
    advice = inference._no_gpu_advice()
    assert advice.startswith("This PyTorch is a CPU-only build. To use the GPU:") and "whl/cu130" in advice, advice
    # nvidia-smi fails after a driver update without a reboot: fix the driver first, then install the CUDA build
    smi["out"] = "Failed to initialize NVML: Driver/library version mismatch\n"
    advice = inference._no_gpu_advice()
    assert advice.startswith("`nvidia-smi` fails here") and "Restart the machine" in advice, advice
    assert "Then install a CUDA build of PyTorch:" in advice and "whl/cu130" in advice, advice
    # nvidia-smi finds no GPU (a login node with the driver): the CPU is expected
    smi["out"] = check_deps.NVIDIA_SMI_NO_GPU + "\n"
    assert inference._no_gpu_advice() is None
    # no driver
    del smi["out"]
    monkeypatch.setattr(check_deps, "libcuda_cuda_version", lambda: None)
    assert inference._no_gpu_advice() is None


def test_call_run_on_arm_says_deeprm_runs_on_the_cpu(monkeypatch):
    """A CPU-only PyTorch on Linux on ARM with an NVIDIA driver: deeprm call run says what deeprm check says (the CPU
    only, no install command)."""
    from deeprm.utils import check_deps

    monkeypatch.setattr(check_deps, "_nvidia_smi", lambda *args: None)
    monkeypatch.setattr(check_deps, "libcuda_cuda_version", lambda: (13, 0))
    monkeypatch.setattr(inference.torch.version, "cuda", None)
    monkeypatch.setattr(inference.platform, "machine", lambda: "aarch64")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    advice = inference._no_gpu_advice()
    assert advice.startswith("On Linux on ARM (aarch64), DeepRM v1.2 runs on the CPU only"), advice
    assert "pip" not in advice, advice


def test_call_run_refuses_a_missing_model_file_before_any_work(tmp_path):
    data = tmp_path / "prep"
    data.mkdir()
    _write_prep(data / "0.npz", n=12)
    parser = argparse.ArgumentParser()
    inference.add_arguments(parser)
    out = tmp_path / "out"
    missing = tmp_path / "typo.pt"
    args = parser.parse_args(["-i", str(data), "-b", "x.bam", "-o", str(out), "-m", str(missing), "-g", "0"])
    with pytest.raises(ValueError, match="model file not found") as refused:
        inference.main(args)
    assert str(missing) in str(refused.value) and not out.exists()


def test_slice_and_output_id_must_fit_the_model_outputs(tmp_path):
    """--slice picks a column of 2-D predictions: it is refused for a one-output model and with --output-id. A model
    with several outputs needs --output-id or --slice. All are refused before the run file is written."""
    data = tmp_path / "prep"
    data.mkdir()
    _write_prep(data / "0.npz", n=12)
    two = tmp_path / "two_outputs.pt"
    torch.save({"model_state_dict": {}, "model_config": {"model_type": "deeprm_v12", "n_label": 2}}, two)
    refused_cases = (
        (WEIGHTS, None, 0, "--slice 0: this model has one output"),
        (two, None, None, "This model has 2 outputs; choose one with --output-id"),
        (two, 0, 1, "--slice and --output-id"),
        (two, None, 2, "--slice 2: this model has 2 outputs, numbered 0 to 1"),
    )
    for i, (model, output_id, slice_idx, message) in enumerate(refused_cases):
        out = tmp_path / f"refused_{i}"
        out.mkdir()
        args = _run_args(data, out, model)
        args.output_id, args.slice = output_id, slice_idx
        with pytest.raises(ValueError, match=message) as refused:
            inference.write_run_info(args)
        assert "\n" not in str(refused.value)
        assert not (out / inference.RUN_INFO_NAME).exists()
    for i, (model, output_id, slice_idx) in enumerate(((two, 1, None), (two, None, 1), (WEIGHTS, 0, None))):
        out = tmp_path / f"accepted_{i}"
        out.mkdir()
        args = _run_args(data, out, model)
        args.output_id, args.slice = output_id, slice_idx
        inference.write_run_info(args)


def test_v11_refusal_names_the_option_that_gave_the_file(tmp_path):
    from deeprm.train import train

    v11 = tmp_path / "v11.pt"
    torch.save({"model_state_dict": {"cnn_encoder.0.weight": torch.zeros(1)}, "model_config": {}}, v11)
    with pytest.raises(ValueError, match=r"DeepRM v1\.1 model") as refused:
        train._deeprm_v12_args(_train_args("--load-checkpoint", str(v11)))
    assert "--load-checkpoint" in str(refused.value) and "--model" not in str(refused.value)
    with pytest.raises(ValueError, match=r"DeepRM v1\.1 model") as refused:
        inference._model_name({}, None, {"cnn_encoder.0.weight": None})
    assert "leave --model out" in str(refused.value) and "--load-checkpoint" not in str(refused.value)
