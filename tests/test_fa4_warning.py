"""The loud warning when fa4 is not used on a GPU FlashAttention-4 supports, for a reason the user can fix (CPU only:
the GPU, PyTorch build, driver and FlashAttention-4 install are faked)."""

import argparse
import pathlib
import platform
import re
import sys

import pytest

torch = pytest.importorskip("torch")

from deeprm.inference import inference  # noqa: E402
from deeprm.model import deeprm_v12_engines as engines  # noqa: E402
from deeprm.utils import check_installation  # noqa: E402

WEIGHTS = pathlib.Path(inference.__file__).parent.parent / "weight" / "deeprm_weights.pt"
GPU = torch.device("cuda", 0)
CU130 = "--index-url https://download.pytorch.org/whl/cu130"
CU128 = "--index-url https://download.pytorch.org/whl/cu128"


def _env(
    monkeypatch,
    tmp_path,
    cc=(10, 0),
    name="NVIDIA B200",
    cuda="13.0",
    driver=(13, 0),
    compile_ok=(True, "ok"),
    fa4=("ok", "/x/flash_attn/cute/interface.py"),
    imports=(True, "ok"),
):
    """A Linux x86_64 machine where everything fa4 needs is in place, unless an argument says otherwise."""
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: cc)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: name)
    monkeypatch.setattr(torch.version, "cuda", cuda)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(engines.check_deps, "driver_cuda_version", lambda: driver)
    monkeypatch.setattr(engines.check_deps, "libcuda_cuda_version", lambda: driver)
    monkeypatch.setattr(engines, "_driver_cuda", lambda: driver)
    monkeypatch.setattr(engines, "compile_status", lambda device: compile_ok)
    monkeypatch.setattr(engines, "_fa4_check", lambda: fa4)
    monkeypatch.setattr(engines, "fa4_imports", lambda: imports)


def _only(causes):
    assert causes is not None and len(causes) == 1, causes
    return causes[0]


def _mixed_gpus(monkeypatch, gpus):
    """GPUs of different kinds: gpus[i] is (compute capability, name) of GPU i. Returns their devices."""

    def index(device=None):
        return device.index if isinstance(device, torch.device) else (device or 0)

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: gpus[index(device)][0])
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: gpus[index(device)][1])
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(gpus))
    return [torch.device("cuda", i) for i in range(len(gpus))]


# --- one test per cause -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("driver", [(12, 5), (12, 8), (13, 0), None])
@pytest.mark.parametrize(
    "cc, name", [((8, 0), "NVIDIA A100-SXM4-80GB"), ((9, 0), "NVIDIA H100 80GB HBM3"), ((10, 0), "NVIDIA B200")]
)
@pytest.mark.parametrize("cuda", ["12.8", "12.9", "13.0"])
def test_a_cuda12_8_or_newer_build_runs_fa4(monkeypatch, tmp_path, cuda, cc, name, driver):
    """fa4 on every GPU FlashAttention-4 supports with a PyTorch of CUDA 12.8 or newer, whatever the driver."""
    _env(monkeypatch, tmp_path, cc=cc, name=name, cuda=cuda, driver=driver)
    engine, reason = engines.resolve_engine("auto", GPU)
    assert (engine, reason) == ("fa4", f"FlashAttention-4 GPU (compute capability {cc[0]}.{cc[1]})")
    assert engines.fa4_fixes(GPU) is None and engines.fa4_warning([(0, GPU, engine)]) is None


@pytest.mark.parametrize("driver, index", [((12, 5), CU128), ((13, 0), CU130), (None, CU128)])
def test_an_older_cuda12_build(monkeypatch, tmp_path, driver, index):
    """A build older than 12.8 gets the build that runs on this driver: no driver update."""
    _env(monkeypatch, tmp_path, cuda="12.6", driver=driver)
    assert engines.resolve_engine("auto", GPU)[0] == "fast"
    reason, steps = _only(engines.fa4_fixes(GPU))
    assert "is a CUDA 12.6 build (DeepRM chooses fa4 with a CUDA 12.8 or newer build)" in reason
    uninstall = engines.check_deps.pip_command("uninstall", "-y", "torch")
    assert len(steps) == 1 and steps[0].startswith(uninstall + " && ") and steps[0].endswith(index)
    assert engines.resolve_engine("fa4", GPU) == ("fa4", "requested")  # an explicit --engine fa4 still runs


@pytest.mark.parametrize("cuda", ["13.0", "12.8"])
def test_the_build_choice_never_reads_the_driver(monkeypatch, tmp_path, cuda):
    for fa4 in (("ok", "/x/flash_attn/cute/interface.py"), ("missing", None)):
        _env(monkeypatch, tmp_path, cuda=cuda, fa4=fa4)
        for owner, name in ((engines, "_driver_cuda"), (engines.check_deps, "driver_cuda_version"),
                            (engines.check_deps, "libcuda_cuda_version")):
            monkeypatch.setattr(owner, name, lambda: pytest.fail("the driver was read"))
        if fa4[0] == "ok":
            assert engines.resolve_engine("auto", GPU)[0] == "fa4"
        else:
            assert _only(engines.fa4_fixes(GPU))[1] == [engines.fa4_install_command()]


@pytest.mark.parametrize("fa4", [("missing", None), ("other", "/x/flash_attn")])
def test_fa4_install_causes_with_a_cuda12_build(monkeypatch, tmp_path, fa4):
    """The same FlashAttention-4 install command for every PyTorch build: it installs both CUTLASS builds."""
    _env(monkeypatch, tmp_path, cuda="12.8", driver=(12, 5), fa4=fa4)
    reason, steps = _only(engines.fa4_fixes(GPU))
    assert steps[0] == engines.fa4_install_command()
    if fa4[0] == "missing":
        engine, why = engines.resolve_engine("auto", GPU)
        assert engine == "fast" and engines.fa4_install_command() in why


def _fake_fa4_site(tmp_path, interface, cute_init=""):
    """A DEEPRM_FA4_SITE folder that holds a fake flash_attn package (its cute/interface.py has the given text)."""
    cute = tmp_path / "site" / "flash_attn" / "cute"
    cute.mkdir(parents=True)
    (tmp_path / "site" / "flash_attn" / "__init__.py").write_text("")
    (cute / "__init__.py").write_text(cute_init)
    (cute / "interface.py").write_text(interface)
    return str(tmp_path / "site")


def _forget_fake_fa4():
    import importlib

    for name in [m for m in sys.modules if m == "flash_attn" or m.startswith("flash_attn.")]:
        sys.modules.pop(name)
    importlib.invalidate_caches()


@pytest.mark.parametrize("case", ["missing", "no_import", "unpinned"])
def test_fa4_messages_name_the_install_command(monkeypatch, tmp_path, case):
    """The short messages of the import, preflight and status checks give the FlashAttention-4 install command."""
    import importlib.util

    if importlib.util.find_spec("flash_attn") is not None:
        pytest.skip("a flash_attn package is installed here")
    _env(monkeypatch, tmp_path, cuda="12.8")
    monkeypatch.delenv(engines.FA4_SITE_ENV, raising=False)
    monkeypatch.setitem(engines._FA4, "ready", None)
    extra = re.escape(engines.fa4_install_command())
    if case == "missing":
        with pytest.raises(RuntimeError, match=extra):
            engines._make_fa4_importable()
        with pytest.raises(RuntimeError, match=extra):
            engines.fa4_preflight(GPU)
        return
    if case == "no_import":
        site_dir = _fake_fa4_site(tmp_path, "import no_such_module_for_fa4_test\n")
        words = "FlashAttention-4 cannot be imported"
    else:  # importable, but its interface.py is not the verified build
        varlen = "def flash_attn_varlen_func(*args, **kwargs):\n    pass\n"
        site_dir = _fake_fa4_site(tmp_path, "# another build\n", varlen)
        words = "is not the verified build"
    monkeypatch.setenv(engines.FA4_SITE_ENV, site_dir)
    monkeypatch.setattr(sys, "path", list(sys.path))  # the site folder is added to sys.path; undo it afterwards
    try:
        with pytest.raises(RuntimeError, match=extra) as info:
            engines.fa4_preflight(GPU)
        assert words in str(info.value)
    finally:
        _forget_fake_fa4()
    monkeypatch.setattr(engines, "_fa4_check", lambda: ("unpinned", "/x/flash_attn/cute/interface.py"))
    ok, reason = engines.fa4_status()
    assert not ok and engines.fa4_install_command() in reason


def test_pyproject_installs_the_fa4_requirements():
    """pyproject.toml's dependencies hold FA4_REQUIREMENTS, each for Linux x86_64 only, and the verified PyTorch."""
    tomllib = pytest.importorskip("tomllib")  # Python 3.11 or newer
    path = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not path.is_file():
        pytest.skip("no pyproject.toml next to the tests (an installed package)")
    deps = tomllib.loads(path.read_text())["project"]["dependencies"]
    marker = 'platform_system == "Linux" and platform_machine == "x86_64"'
    assert [d for d in deps if d.endswith(marker)] == [f"{r}; {marker}" for r in engines.FA4_REQUIREMENTS]
    assert "torch==2.10.0" in deps
    assert "fa4" not in tomllib.loads(path.read_text())["project"]["optional-dependencies"]


@pytest.mark.parametrize("cc, name", [((8, 6), "NVIDIA A10"), ((8, 9), "NVIDIA L40S"), ((10, 3), "NVIDIA B300"),
                                      ((12, 0), "NVIDIA RTX PRO 6000")])
def test_auto_keeps_fast_on_gpus_fa4_was_not_tested_on(monkeypatch, tmp_path, cc, name):
    """FlashAttention-4 comes with DeepRM, so auto uses fa4 only where it was tested; --engine fa4 still runs it."""
    _env(monkeypatch, tmp_path, cc=cc, name=name)
    engine, reason = engines.resolve_engine("auto", GPU)
    assert engine == "fast" and "tested on compute capability 8.0, 9.0 and 10.0 only" in reason
    assert reason.endswith("--engine fa4 runs it here")
    assert engines.fa4_warning([(0, GPU, engine)]) is not None  # said loudly (see test_untested_gpu_warning)
    assert engines.resolve_engine("fa4", GPU) == ("fa4", "requested")
    # an older PyTorch build or no FlashAttention-4: the same reason, never "chooses fa4 with a newer build"
    for kwargs in ({"cuda": "12.6"}, {"fa4": ("missing", None)}):
        _env(monkeypatch, tmp_path, cc=cc, name=name, **kwargs)
        engine, reason = engines.resolve_engine("auto", GPU)
        assert engine == "fast" and "tested on compute capability" in reason and "runs it here" not in reason


UNTESTED = [((10, 3), "NVIDIA B300 SXM6 AC"), ((8, 6), "NVIDIA GeForce RTX 3090"), ((12, 0), "NVIDIA GeForce RTX 5090")]
NOT_TESTED = "FlashAttention-4 was not tested on this GPU, so DeepRM does not choose fa4 by itself"
TRY_FA4 = "Run deeprm call run with --engine fa4 (FlashAttention-4 has kernels for this GPU)."


@pytest.mark.parametrize("cc, name", UNTESTED)
def test_untested_gpu_warning(monkeypatch, tmp_path, cc, name):
    """On a GPU fa4 was not tested on, auto keeps fast, and the warning says so with the GPU and how to try fa4."""
    _env(monkeypatch, tmp_path, cc=cc, name=name)
    engine = engines.resolve_engine("auto", GPU)[0]
    assert engine == "fast"
    reason, steps = _only(engines.fa4_fixes(GPU))
    assert reason.startswith(NOT_TESTED) and "A100, H100 and B200" in reason and steps == [TRY_FA4]
    summary, text = engines.fa4_warning([(0, GPU, engine)])
    where = f"{name} (compute capability {cc[0]}.{cc[1]}; GPU 0)"
    assert summary == f"fa4 is not used on {where}: {reason}."
    assert text.splitlines() == [
        "=" * 72,
        f"WARNING: fa4 is not used on {where}.",
        "This run uses the fast engine instead, because:",
        f"  - {reason}",
        "To try fa4, do these steps:",
        f"  1. {TRY_FA4}",
        "To keep fast without this warning, run deeprm call run with --engine fast.",
        "=" * 72,
    ]


def test_untested_gpu_install_steps_come_first(monkeypatch, tmp_path):
    """Without FlashAttention-4, --engine fa4 stops, so installing it is step 1 and --engine fa4 step 2."""
    _env(monkeypatch, tmp_path, cc=(10, 3), name="NVIDIA B300", fa4=("missing", None))
    causes = engines.fa4_fixes(GPU)
    assert [reason[: len(NOT_TESTED)] for reason, _ in causes] == ["FlashAttention-4 is not installed", NOT_TESTED]
    lines = engines.fa4_warning([(0, GPU, "fast")])[1].splitlines()
    assert lines[lines.index("To try fa4, do these steps:") + 1 :][:2] == [
        f"  1. {engines.fa4_install_command()}",
        f"  2. {TRY_FA4}",
    ]
    assert not any("deeprm check" in line for line in lines)  # auto keeps fast after the steps


def test_tested_gpu_warning_is_unchanged(monkeypatch, tmp_path):
    """On a B200 (10.0) the warning has no untested-GPU cause and keeps its text."""
    _env(monkeypatch, tmp_path, fa4=("missing", None))
    assert engines.fa4_warning([(0, GPU, "fast")])[1].splitlines() == [
        "=" * 72,
        "WARNING: fa4 is not used on NVIDIA B200 (compute capability 10.0; GPU 0).",
        "This run uses the fast engine instead, because:",
        "  - FlashAttention-4 is not installed",
        "To use fa4, do these steps, then run DeepRM again:",
        f"  1. {engines.fa4_install_command()}",
        "`deeprm check` shows the engine DeepRM will then use.",
        "To keep fast without this warning, run deeprm call run with --engine fast.",
        "=" * 72,
    ]


def test_a_broken_pytorch_with_a_cuda12_build_gets_the_cuda12_8_build(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path, cuda="12.8", compile_ok=(False, "torch._dynamo cannot be imported (ImportError)"))
    reason, steps = _only(engines.fa4_fixes(GPU))
    assert steps[0].endswith(CU128)


def test_the_driver_is_read_from_libcuda_first(monkeypatch):
    import ctypes

    class Lib:
        def __init__(self, value):
            self.value = value

        def cuDriverGetVersion(self, ref):
            ref._obj.value = self.value
            return 0

    monkeypatch.setattr(ctypes, "CDLL", lambda name: Lib(12050))
    assert engines.check_deps.libcuda_cuda_version() == (12, 5)
    monkeypatch.setattr(engines.check_deps, "driver_cuda_version", lambda: (13, 0))
    assert engines._driver_cuda() == (12, 5)  # libcuda wins over the nvidia-smi banner
    assert engines.check_deps.driver_cuda() == (12, 5)  # the same reading for the PyTorch install hints

    def no_lib(name):
        raise OSError("libcuda.so.1: cannot open shared object file")

    monkeypatch.setattr(ctypes, "CDLL", no_lib)
    assert engines.check_deps.libcuda_cuda_version() is None
    assert engines._driver_cuda() == (13, 0)  # then the banner


@pytest.mark.parametrize(
    "compile_reason, step",
    [
        ("TORCH_COMPILE_DISABLE=1 is set", "unset TORCH_COMPILE_DISABLE"),
        ("no C compiler found (Triton needs gcc or clang at run time)", "sudo apt install gcc"),
        ("Python's C headers are missing (no Python.h in /x; Triton needs them at run time)", "-dev`"),
        ("Triton is not available", engines.check_deps.pip_command("install", '"triton==')),
        ("torch._dynamo cannot be imported (ImportError)", CU130),
    ],
)
def test_compile_causes(monkeypatch, tmp_path, compile_reason, step):
    _env(monkeypatch, tmp_path, compile_ok=(False, compile_reason))
    monkeypatch.setattr(engines, "_triton_pin", lambda: "3.6.0")  # as the PyPI CUDA build of torch pins it
    reason, steps = _only(engines.fa4_fixes(GPU))
    assert reason == compile_reason and step in steps[0]


def test_triton_missing_without_a_triton_pin(monkeypatch, tmp_path):
    """A torch build that pins no Triton (for example the CPU build): the step installs the CUDA build of torch."""
    _env(monkeypatch, tmp_path, compile_ok=(False, "Triton is not available"))
    monkeypatch.setattr(engines, "_triton_pin", lambda: None)
    reason, steps = _only(engines.fa4_fixes(GPU))
    assert reason == "Triton is not available" and steps == [engines._torch_fa4_command()]


@pytest.mark.parametrize(
    "fa4, words, step",
    [
        (("missing", None), "FlashAttention-4 is not installed", '"flash-attn-4[cu13]==4.0.0b35"'),
        (("site", "/no/such/dir"), "DEEPRM_FA4_SITE is set to /no/such/dir", "Set DEEPRM_FA4_SITE"),
        (("other", "/x/flash_attn"), "(/x/flash_attn) is not FlashAttention-4", '"flash-attn-4[cu13]==4.0.0b35"'),
    ],
)
def test_fa4_install_causes(monkeypatch, tmp_path, fa4, words, step):
    _env(monkeypatch, tmp_path, fa4=fa4)
    reason, steps = _only(engines.fa4_fixes(GPU))
    assert words in reason and step in steps[0]


def test_fa4_not_the_verified_build(monkeypatch, tmp_path):
    interface = tmp_path / "interface.py"
    interface.write_text("# another build\n")
    _env(monkeypatch, tmp_path, fa4=("unpinned", str(interface)))
    reason, steps = _only(engines.fa4_fixes(GPU))
    assert "DeepRM v1.2 was verified with 4.0.0b35" in reason
    assert engines.fa4_install_command() in steps[0] and "DEEPRM_FA4_UNPINNED_OK=1" in steps[0]


def test_fa4_import_error_keeps_its_message(monkeypatch, tmp_path):
    _env(
        monkeypatch,
        tmp_path,
        imports=(False, "FlashAttention-4 does not import (ModuleNotFoundError: No module named 'cutlass')"),
    )
    reason, steps = _only(engines.fa4_fixes(GPU))
    assert "No module named 'cutlass'" in reason and steps[0] == engines.fa4_install_command()


def test_fa4_imports_reports_the_missing_module(monkeypatch, tmp_path):
    """fa4_imports itself keeps the error message (it kept only the type before)."""
    import importlib
    import importlib.util

    if importlib.util.find_spec("flash_attn") is not None:
        pytest.skip("a flash_attn package is installed here")
    cute = tmp_path / "site" / "flash_attn" / "cute"
    cute.mkdir(parents=True)
    (cute / "interface.py").write_text("import no_such_module_for_fa4_test\n")
    monkeypatch.setenv(engines.FA4_SITE_ENV, str(tmp_path / "site"))
    monkeypatch.setattr(sys, "path", list(sys.path))  # the site folder is added to sys.path; undo it afterwards
    try:
        ok, reason = engines.fa4_imports()
    finally:  # forget the fake package, so later tests do not find it
        for name in [m for m in sys.modules if m == "flash_attn" or m.startswith("flash_attn.")]:
            sys.modules.pop(name)
        importlib.invalidate_caches()
    assert ok is False and "No module named 'no_such_module_for_fa4_test'" in reason


def test_wrong_fa4_site_folder_is_named(monkeypatch, tmp_path):
    import importlib.util

    if importlib.util.find_spec("flash_attn") is not None:
        pytest.skip("a flash_attn package is installed here")
    monkeypatch.setenv(engines.FA4_SITE_ENV, str(tmp_path / "empty"))
    assert engines.fa4_status() == (
        False,
        f"DEEPRM_FA4_SITE={tmp_path / 'empty'} holds no FlashAttention-4 (no flash_attn/cute/interface.py in it)",
    )


def test_a_compile_problem_does_not_hide_a_missing_fa4(monkeypatch, tmp_path):
    _env(
        monkeypatch,
        tmp_path,
        compile_ok=(False, "no C compiler found (Triton needs gcc or clang at run time)"),
        fa4=("missing", None),
    )
    assert engines.resolve_engine("auto", GPU)[0] == "eager"
    causes = engines.fa4_fixes(GPU)
    assert [reason.split(" ")[0] for reason, _ in causes] == ["no", "FlashAttention-4"]


def test_mixed_machine_never_gets_a_pytorch_build_without_code_for_one_of_its_gpus(monkeypatch, tmp_path):
    """P100 + A100 with the cu126 build, the one build with code for both: the fa4 step does not swap it for a build
    without P100 code (deeprm check would then ask for cu126 again); it says so and names --gpu-pool."""
    _env(monkeypatch, tmp_path, cuda="12.6")
    p100, a100 = _mixed_gpus(monkeypatch, [((6, 0), "Tesla P100-PCIE-16GB"), ((8, 0), "NVIDIA A100-SXM4-80GB")])
    text = engines.fa4_warning([(0, p100, "fast"), (1, a100, "fast")])[1]
    assert "fa4 is not used on NVIDIA A100-SXM4-80GB (compute capability 8.0; GPU 1)." in text
    steps = [line for line in text.splitlines() if line.startswith("  1. ")]
    assert len(steps) == 1 and not steps[0].endswith(CU130)
    no_build = "No PyTorch build that fa4 runs with has code for GPU 0 (Tesla P100-PCIE-16GB, compute capability 6.0)"
    assert no_build in steps[0]
    assert f"`{engines._torch_fa4_command()}`" in steps[0] and steps[0].endswith("(--gpu-pool 1).")
    assert _only(engines.fa4_fixes(a100))[1] == [steps[0][len("  1. ") :]]  # the same from every GPU PyTorch sees
    # one kind of GPU: the plain command, as before
    assert _only(engines.fa4_fixes(a100, [(1, "NVIDIA A100-SXM4-80GB", (8, 0))]))[1] == [engines._torch_fa4_command()]


V100 = ((7, 0), "Tesla V100-SXM2-32GB")
P100 = ((6, 0), "Tesla P100-PCIE-16GB")
A100 = ((8, 0), "NVIDIA A100-SXM4-80GB")


@pytest.mark.parametrize("driver", [(13, 0), (12, 5)])
def test_mixed_machine_gets_the_fa4_build_with_code_for_every_gpu(monkeypatch, tmp_path, driver):
    """V100 + A100 with the cu126 build and a driver for CUDA 13.0: cu130 has no V100 code, but cu128 has code for
    both and fa4 runs with it, so the step is cu128, with no --gpu-pool that drops the V100."""
    _env(monkeypatch, tmp_path, cuda="12.6", driver=driver)
    v100, a100 = _mixed_gpus(monkeypatch, [V100, A100])
    text = engines.fa4_warning([(0, v100, "fast"), (1, a100, "fast")])[1]
    cu128 = engines.check_deps.torch_install_command("cu128")
    assert [line for line in text.splitlines() if line.startswith("  1. ")] == [f"  1. {cu128}"]
    assert "--gpu-pool" not in text and "V100" not in text  # the V100 runs fast (no fa4 kernels): not named
    assert _only(engines.fa4_fixes(a100))[1] == [cu128]
    assert engines.check_deps.torch_index_for(driver, [V100[0], A100[0]]) == "cu128"  # deeprm check's choice agrees


def test_mixed_machine_without_a_fa4_build_for_every_gpu_keeps_the_most(monkeypatch, tmp_path):
    """P100 + V100 + A100: no build fa4 runs with has P100 code. cu128 has code for the V100 and the A100, cu130 for
    the A100 only, so the step gives cu128 and --gpu-pool 1 2 (not cu130 and --gpu-pool 2)."""
    _env(monkeypatch, tmp_path, cuda="12.6")
    devices = _mixed_gpus(monkeypatch, [P100, V100, A100])
    text = engines.fa4_warning([(i, device, "fast") for i, device in enumerate(devices)])[1]
    cu128 = engines.check_deps.torch_install_command("cu128")
    assert [line for line in text.splitlines() if line.startswith("  1. ")] == [
        "  1. No PyTorch build that fa4 runs with has code for GPU 0 (Tesla P100-PCIE-16GB, compute capability 6.0). "
        f"To use fa4, install the build for the other GPUs (`{cu128}`) and run deeprm call run on them only "
        "(--gpu-pool 1 2)."
    ]


def test_mixed_tested_and_untested_gpus_name_the_untested_one(monkeypatch, tmp_path):
    """B200 + B300 without FlashAttention-4: the not-tested reason names the B300, and the box says how to use fa4
    (the B200 is a tested GPU) without the `deeprm check` line (auto keeps fast on the B300)."""
    _env(monkeypatch, tmp_path, fa4=("missing", None))
    b200, b300 = _mixed_gpus(monkeypatch, [((10, 0), "NVIDIA B200"), ((10, 3), "NVIDIA B300")])
    summary, text = engines.fa4_warning([(0, b200, "fast"), (1, b300, "fast")])
    lines = text.splitlines()
    assert lines[1] == (
        "WARNING: fa4 is not used on NVIDIA B200 (compute capability 10.0; GPU 0); NVIDIA B300 (compute capability "
        "10.3; GPU 1)."
    )
    assert lines[3] == "  - FlashAttention-4 is not installed"
    assert lines[4].startswith("  - " + NOT_TESTED.replace("this GPU", "NVIDIA B300 (GPU 1)"))
    assert lines[5:8] == [
        "To use fa4, do these steps, then run DeepRM again:",
        f"  1. {engines.fa4_install_command()}",
        "  2. Run deeprm call run with --engine fa4 (FlashAttention-4 has kernels for these GPUs).",
    ]
    assert not any("deeprm check" in line for line in lines) and "this GPU" not in text


@pytest.mark.parametrize("cc, name", [((9, 0), "NVIDIA GH200 480GB"), ((10, 3), "NVIDIA B300")])
def test_no_fa4_hints_off_linux_x86_64(monkeypatch, tmp_path, cc, name):
    """DeepRM installs FlashAttention-4 on Linux x86_64 only (the pyproject.toml marker): no fa4 warning or install
    command elsewhere, for example on a GH200 (aarch64)."""
    _env(monkeypatch, tmp_path, cc=cc, name=name, fa4=("missing", None))
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    assert engines.fa4_fixes(GPU) is None and engines.fa4_warning([(0, GPU, "fast")]) is None
    engine, reason = engines.resolve_engine("auto", GPU)
    assert engine == "fast" and engines.fa4_install_command() not in reason
    if cc == (9, 0):
        assert reason == "FlashAttention-4 not installed (DeepRM installs it on Linux x86_64 only)"
    _fake_check_machine(monkeypatch)
    env, issues = check_installation.collect()
    assert "FA4_NOT_USED" not in [i.code for i in issues] and "fa4_warning" not in env["engine"]


def test_warning_notes_a_missing_pip_and_uv(monkeypatch, tmp_path):
    """No pip and no uv on PATH: the box keeps the pip command and says what to do first."""
    _env(monkeypatch, tmp_path, fa4=("missing", None))
    monkeypatch.setattr(engines.check_deps, "_has_pip", lambda: False)
    monkeypatch.setattr(engines.check_deps, "_has_uv", lambda: False)
    lines = engines.fa4_warning([(0, GPU, "fast")])[1].splitlines()
    step = lines.index(f"  1. {engines.fa4_install_command()}")
    assert " -m pip install " in lines[step] and lines[step + 1] == f"Note: {engines.check_deps.pip_note()}."
    _env(monkeypatch, tmp_path, cc=(10, 3), name="NVIDIA B300")  # the one step is --engine fa4: no note
    assert not any(line.startswith("Note:") for line in engines.fa4_warning([(0, GPU, "fast")])[1].splitlines())


# --- no warning: fa4 runs, or not using it is on purpose ----------------------------------------------------------


def test_no_cause_when_fa4_runs(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    assert engines.resolve_engine("auto", GPU)[0] == "fa4"
    assert engines.fa4_fixes(GPU) is None
    assert engines.fa4_warning([(0, GPU, "fa4")]) is None


def test_no_cause_on_cpu_older_gpus_or_rocm(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path, fa4=("missing", None))
    assert engines.fa4_fixes(torch.device("cpu")) is None
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (7, 5))  # T4: no FA4 kernels
    assert engines.fa4_fixes(GPU) is None
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 4))  # MI300X under ROCm
    monkeypatch.setattr(torch.version, "hip", "6.4")
    assert engines.fa4_fixes(GPU) is None


# --- the framed text ----------------------------------------------------------------------------------------------


def test_warning_text(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path, fa4=("missing", None))
    summary, text = engines.fa4_warning([(0, GPU, "fast"), (1, GPU, "fast")])
    lines = text.splitlines()
    assert lines[0] == lines[-1] == "=" * 72
    assert lines[1] == "WARNING: fa4 is not used on NVIDIA B200 (compute capability 10.0; GPUs 0, 1)."
    assert lines[2] == "This run uses the fast engine instead, because:"
    assert "  1. " + engines.fa4_install_command() in lines
    assert "--engine fast" in lines[-2]
    assert summary == (
        "fa4 is not used on NVIDIA B200 (compute capability 10.0; GPUs 0, 1): FlashAttention-4 is not installed."
    )


# --- deeprm call run: built once by the parent, printed at the start and the end ----------------------------------


def _run_args(data, out, engine="auto"):
    return argparse.Namespace(
        data=str(data),
        output=str(out),
        model=str(WEIGHTS),
        model_type=None,
        engine=engine,
        batch=20,
        num_gpu=1,
        gpu_pool=[0],
        prefetch=1,
        worker=0,
        flush=100,
        resume=False,
        output_id=None,
        postfix="",
        threshold=0.5,
        epsilon=1e-6,
        thread=None,
        label_div=10**9,
        slice=None,
        annot=None,
    )


def test_write_run_info_builds_the_warning_once_per_gpu_type(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path, fa4=("missing", None))
    calls = []
    real = engines.fa4_fixes
    monkeypatch.setattr(engines, "fa4_fixes", lambda device, *pool: calls.append(device) or real(device, *pool))
    args = _run_args(tmp_path, tmp_path)
    args.num_gpu, args.gpu_pool = 2, [0, 1]
    inference.write_run_info(args)
    assert args.engine_resolved[0] == "fast" and len(calls) == 1
    assert args.fa4_warning.splitlines()[1].startswith("WARNING: fa4 is not used on NVIDIA B200")


@pytest.mark.parametrize("case", ["engine_fast", "cpu", "dwell_model", "cc75", "fa4_runs"])
def test_write_run_info_without_a_warning(monkeypatch, tmp_path, case):
    _env(monkeypatch, tmp_path, fa4=("ok", "/x") if case == "fa4_runs" else ("missing", None))
    args = _run_args(tmp_path, tmp_path, engine="fast" if case == "engine_fast" else "auto")
    if case == "cpu":
        args.num_gpu, args.gpu_pool = 0, []
    if case == "dwell_model":
        monkeypatch.setattr(inference, "_dwell_bq_dim", lambda config: 3)  # forced to eager on purpose
    if case == "cc75":
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (7, 5))
    inference.write_run_info(args)
    assert args.fa4_warning is None


@pytest.mark.parametrize("engine", ["auto", "fast", "fa4"])
def test_write_run_info_on_an_untested_gpu(monkeypatch, tmp_path, engine):
    """deeprm call run warns on a B300 with auto; an engine the user asked for gets no warning."""
    _env(monkeypatch, tmp_path, cc=(10, 3), name="NVIDIA B300")
    args = _run_args(tmp_path, tmp_path, engine=engine)
    inference.write_run_info(args)
    assert args.engine_resolved[0] == ("fast" if engine == "auto" else engine)
    if engine == "auto":
        lines = args.fa4_warning.splitlines()
        assert lines[1] == "WARNING: fa4 is not used on NVIDIA B300 (compute capability 10.3; GPU 0)."
        assert lines[3].startswith(f"  - {NOT_TESTED}") and f"  1. {TRY_FA4}" in lines
    else:
        assert args.fa4_warning is None


@pytest.mark.parametrize("fails", [False, True])
def test_call_run_prints_the_warning_at_the_start_and_the_end(monkeypatch, tmp_path, fails):
    _env(monkeypatch, tmp_path, fa4=("missing", None))
    (tmp_path / "prep").mkdir()
    events = []

    def fake_gpus(args):
        args.num_gpu, args.gpu_pool = 1, [0]

    def fake_run(args):
        events.append("run")
        if fails:
            raise RuntimeError("worker died")

    monkeypatch.setattr(inference, "_normalize_gpu_config", fake_gpus)
    monkeypatch.setattr(inference, "run_inference", fake_run)
    from deeprm.inference import pileup_deeprm  # main() imports it (at its start, to check the pileup settings)

    monkeypatch.setattr(pileup_deeprm, "main", lambda args: events.append("pileup"))
    monkeypatch.setattr(inference.log, "warning", lambda msg, *a: events.append(msg % a if a else msg))
    args = _run_args(tmp_path / "prep", tmp_path / "out")
    if fails:
        with pytest.raises(RuntimeError, match="worker died"):
            inference.main(args)
    else:
        inference.main(args)
    boxes = [i for i, e in enumerate(events) if e.startswith("=" * 72)]
    assert len(boxes) == 2 and events[0].startswith("=" * 72) and boxes[1] == len(events) - 1
    assert events[1:-1] == (["run"] if fails else ["run", "pileup"])


def test_gpu_processes_never_print_it(monkeypatch):
    import inspect

    assert "fa4_warning" not in inspect.getsource(inference.inference_worker)
    assert "fa4_warning" not in inspect.getsource(inference.run_inference)


# --- deeprm check -------------------------------------------------------------------------------------------------


def _fake_check_machine(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(check_installation, "_nvidia_info", lambda: {"present": False})
    monkeypatch.setattr(check_installation.check_deps, "gpus_without_code", lambda torch_module: [])


def test_check_warns_and_exits_0(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path, fa4=("missing", None))
    _fake_check_machine(monkeypatch)
    printed = []
    monkeypatch.setattr(check_installation.log, "warning", lambda msg, *a: printed.append(msg % a if a else msg))
    env, issues = check_installation.collect()
    assert [i.code for i in issues if i.code == "FA4_NOT_USED"] == ["FA4_NOT_USED"]
    assert check_installation.main([]) == 0
    assert printed[-1].splitlines()[1] == "WARNING: fa4 is not used on NVIDIA B200 (compute capability 10.0; GPU 0)."
    assert "deeprm call run would use the fast engine instead" in printed[-1]


@pytest.mark.parametrize("cc, name", UNTESTED)
def test_check_warns_on_an_untested_gpu_and_exits_0(monkeypatch, tmp_path, cc, name):
    _env(monkeypatch, tmp_path, cc=cc, name=name)
    _fake_check_machine(monkeypatch)
    printed = []
    monkeypatch.setattr(check_installation.log, "warning", lambda msg, *a: printed.append(msg % a if a else msg))
    env, issues = check_installation.collect()
    assert env["engine"]["auto_engine"] == "fast"
    found = [i for i in issues if i.code == "FA4_NOT_USED"]
    assert len(found) == 1 and NOT_TESTED in found[0].message
    assert found[0].hint == "The steps are in the box at the end of this output."  # nothing is broken: no "fix"
    assert check_installation.main([]) == 0
    where = f"{name} (compute capability {cc[0]}.{cc[1]}; GPU 0)"
    assert printed[-1].splitlines()[1] == f"WARNING: fa4 is not used on {where}."
    assert "deeprm call run would use the fast engine instead" in printed[-1] and TRY_FA4 in printed[-1]


def test_check_stays_quiet_when_fa4_runs(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    _fake_check_machine(monkeypatch)
    env, issues = check_installation.collect()
    assert "FA4_NOT_USED" not in [i.code for i in issues] and "fa4_warning" not in env["engine"]


def test_check_says_fa4_on_an_a100_with_a_cuda12_8_build(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path, cc=(8, 0), name="NVIDIA A100-SXM4-80GB", cuda="12.8", driver=(12, 5))
    _fake_check_machine(monkeypatch)
    env, issues = check_installation.collect()
    assert "FA4_NOT_USED" not in [i.code for i in issues] and "fa4_warning" not in env["engine"]
    assert env["engine"]["auto_engine"] == "fa4"


def test_call_run_refuses_a_missing_annotation_file_before_the_inference(monkeypatch, tmp_path):
    """The pileup's settings are checked at the start of deeprm call run, not after the inference."""
    (tmp_path / "prep").mkdir()
    ran = []
    monkeypatch.setattr(inference, "run_inference", lambda args: ran.append("run"))
    args = _run_args(tmp_path / "prep", tmp_path / "out")
    args.annot = str(tmp_path / "no_such_refFlat.txt")
    with pytest.raises(ValueError, match="--annot"):
        inference.main(args)
    assert ran == [] and not (tmp_path / "out").exists()
