"""Tests for the PyTorch install hints, deeprm check's exit status and the CPU-fallback warning (CPU only)."""

import argparse
import shlex
import sys

import pytest

from deeprm.utils import check_deps, check_installation


def test_torch_index_for_driver():
    assert check_deps.torch_index_for(None) == "cpu"
    assert check_deps.torch_index_for((13, 0)) == "cu130"
    assert check_deps.torch_index_for((13, 2)) == "cu130"
    assert check_deps.torch_index_for((12, 9)) == "cu128"
    assert check_deps.torch_index_for((12, 4)) == "cu128"  # a CUDA 12 build runs on any CUDA 12 driver (R525+)
    assert check_deps.torch_index_for((11, 8)) is None  # no PyTorch 2.10 build for this driver


@pytest.mark.parametrize(
    "driver, ccs, expected",
    [
        ((13, 0), [(7, 5)], "cu130"),  # T4, RTX 20, GTX 16
        ((13, 0), [(10, 0), (10, 0)], "cu130"),  # B200
        ((13, 0), [(7, 0)], "cu128"),  # V100: the CUDA 13 build has no code for it
        ((13, 0), [(6, 0)], "cu126"),  # P100
        ((13, 0), [(6, 1)], "cu126"),  # P40, GTX 10 (they run the 6.0 code)
        ((12, 8), [(8, 9)], "cu128"),  # RTX 40, L4 (they run the 8.6 code)
        ((12, 2), [(10, 0)], "cu128"),
        ((13, 0), [(10, 0), (6, 0)], None),  # no single build has code for both
    ],
)
def test_torch_index_for_gpus(driver, ccs, expected):
    assert check_deps.torch_index_for(driver, ccs) == expected


def test_torch_install_hint_text():
    hint = check_deps.torch_install_hint((13, 0), [(10, 0)])
    assert check_deps.pip_command("uninstall", "-y", "torch") in hint and f"torch=={check_deps.TORCH_VERIFIED}" in hint
    assert hint.count("https://download.pytorch.org/whl/cu130") == 1
    assert "whl/cu126" in check_deps.torch_install_hint((13, 0), [(6, 0)])
    assert "whl/cpu" in check_deps.torch_install_hint(None, [])
    assert "Update the NVIDIA driver" in check_deps.torch_install_hint((11, 8), [(7, 0)])
    assert "No PyTorch" in check_deps.torch_install_hint((13, 0), [(10, 0), (6, 0)])


def test_gpus_without_code():
    class FakeCuda:
        def __init__(self, archs, ccs):
            self.archs, self.ccs = archs, ccs

        def is_available(self):
            return True

        def get_arch_list(self):
            return self.archs

        def device_count(self):
            return len(self.ccs)

        def get_device_capability(self, i):
            return self.ccs[i]

        def get_device_name(self, i):
            return f"gpu{i}"

    class FakeTorch:
        def __init__(self, archs, ccs):
            self.cuda = FakeCuda(archs, ccs)

    cu130 = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120"]
    assert check_deps.torch_build_archs(cu130) == [(7, 5), (8, 0), (8, 6), (9, 0), (10, 0), (12, 0)]
    assert check_deps.gpus_without_code(FakeTorch(cu130, [(10, 0), (7, 5), (8, 9)])) == []
    missing = check_deps.gpus_without_code(FakeTorch(cu130, [(7, 0), (6, 0)]))  # V100 and P100
    assert missing == [(0, "gpu0", (7, 0)), (1, "gpu1", (6, 0))]


def test_version_tuple():
    assert check_deps.version_tuple("2.10.0+cu130") == (2, 10)
    assert check_deps.version_tuple("1.9.0") == (1, 9)
    assert check_deps.version_tuple(None) is None
    assert check_deps.version_tuple("2.10.0") >= check_deps.TORCH_MIN > check_deps.version_tuple("2.9.1")


@pytest.mark.parametrize("severity, expected", [("ERROR", 1), ("WARN", 0), (None, 0)])
def test_check_exit_status(monkeypatch, severity, expected):
    issues = [check_installation.Issue(severity=severity, message="m", hint="h", code="C")] if severity else []
    monkeypatch.setattr(
        check_installation, "collect", lambda require_train=False, verbose=False: ({"engine": {}}, issues)
    )
    assert check_installation.main([]) == expected


def _gpu_args(num_gpu=None):
    return argparse.Namespace(gpu_pool=None, num_gpu=num_gpu)


def _driver_from_the_banner_only(monkeypatch):
    """No libcuda and a working nvidia-smi: the faked driver_cuda_version (the nvidia-smi banner) is the driver."""
    monkeypatch.setattr(check_deps, "libcuda_cuda_version", lambda: None)
    monkeypatch.setattr(check_deps, "nvidia_smi_error", lambda: None, raising=False)


def test_cpu_fallback_warns_when_a_driver_is_present(monkeypatch):
    from deeprm.inference import inference

    warnings = []
    # PyTorch sees no GPU. is_available is faked too: with CUDA_VISIBLE_DEVICES unset, the real one would start CUDA
    # with every GPU of the machine visible, for the rest of the test session.
    monkeypatch.setattr(inference.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(inference.torch.cuda, "device_count", lambda: 0)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)  # set, it would be the cause (tested below)
    _driver_from_the_banner_only(monkeypatch)
    monkeypatch.setattr(inference.check_deps, "gpu_compute_capabilities", lambda: [])
    monkeypatch.setattr(inference.check_deps, "driver_cuda_version", lambda: (13, 0))
    monkeypatch.setattr(inference.log, "warning", lambda msg, *a: warnings.append(msg % a))
    args = _gpu_args()
    inference._normalize_gpu_config(args)
    assert args.num_gpu == 0 and args.gpu_pool == []
    assert len(warnings) == 1 and "sees no GPU" in warnings[0] and "whl/cu130" in warnings[0]
    # no driver: CPU is expected, no warning
    monkeypatch.setattr(inference.check_deps, "driver_cuda_version", lambda: None)
    inference._normalize_gpu_config(_gpu_args())
    assert len(warnings) == 1
    # asking for a GPU that torch cannot see names the fix
    monkeypatch.setattr(inference.check_deps, "driver_cuda_version", lambda: (12, 8))
    with pytest.raises(ValueError, match="whl/cu128"):
        inference._normalize_gpu_config(_gpu_args(num_gpu=1))


def test_cpu_fallback_names_cuda_visible_devices(monkeypatch):
    """CUDA_VISIBLE_DEVICES hides the GPUs: say so, and do not suggest reinstalling the build that is installed."""
    from deeprm.inference import inference

    warnings = []
    monkeypatch.setattr(inference.torch.version, "cuda", "13.0")
    monkeypatch.setattr(inference.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(inference.torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(inference.check_deps, "gpu_compute_capabilities", lambda: [(10, 0)])
    monkeypatch.setattr(inference.check_deps, "driver_cuda_version", lambda: (13, 0))
    _driver_from_the_banner_only(monkeypatch)
    monkeypatch.setattr(inference.log, "warning", lambda msg, *a: warnings.append(msg % a))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    inference._normalize_gpu_config(_gpu_args())
    assert len(warnings) == 1 and "CUDA_VISIBLE_DEVICES" in warnings[0]
    assert "whl/cu130" not in warnings[0] and "pip install" not in warnings[0]
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES") as refused:
        inference._normalize_gpu_config(_gpu_args(num_gpu=1))
    assert "pip install" not in str(refused.value)


@pytest.fixture
def cuda_build_without_gpu(monkeypatch):
    """A CUDA 13.0 build of PyTorch that sees no GPU, and a driver for CUDA 13.0 with eight B200s."""
    import torch

    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(check_deps, "driver_cuda_version", lambda: (13, 0))
    monkeypatch.setattr(check_deps, "gpu_compute_capabilities", lambda: [(10, 0)] * 8)
    _driver_from_the_banner_only(monkeypatch)
    return torch


def _cuda_runtime_issue(torch_info_issues):
    found = [issue for issue in torch_info_issues if issue.code == "CUDA_RUNTIME_MISSING"]
    assert len(found) == 1
    return found[0]


def test_check_hidden_gpus_is_a_warning_without_reinstall_advice(monkeypatch, cuda_build_without_gpu):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    issue = _cuda_runtime_issue(check_installation._torch_info({"present": True})[1])
    assert issue.severity == "WARN" and "CUDA_VISIBLE_DEVICES" in issue.hint
    assert "pip install" not in issue.message + issue.hint


def test_check_without_a_driver_is_a_warning_without_reinstall_advice(monkeypatch, cuda_build_without_gpu):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(check_deps, "driver_cuda_version", lambda: None)
    issue = _cuda_runtime_issue(check_installation._torch_info({"present": False})[1])
    assert issue.severity == "WARN" and "no NVIDIA driver" in issue.message
    assert "pip" not in issue.message + issue.hint


def test_check_driver_too_old_for_the_build_is_an_error_with_the_install_hint(monkeypatch, cuda_build_without_gpu):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # the driver is checked first: it is the cause either way
    monkeypatch.setattr(check_deps, "driver_cuda_version", lambda: (12, 8))
    issue = _cuda_runtime_issue(check_installation._torch_info({"present": True})[1])
    assert issue.severity == "ERROR" and "whl/cu128" in issue.hint and "CUDA 12.8" in issue.message


def test_check_unknown_cause_is_an_error_that_names_nvidia_smi(monkeypatch, cuda_build_without_gpu):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    issue = _cuda_runtime_issue(check_installation._torch_info({"present": True})[1])
    assert issue.severity == "ERROR" and "nvidia-smi" in issue.hint


def test_cuda_unavailable_cause(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    _driver_from_the_banner_only(monkeypatch)
    monkeypatch.setattr(check_deps, "driver_cuda_version", lambda: None)
    assert check_deps.cuda_unavailable_cause("13.0") == "no_driver"
    monkeypatch.setattr(check_deps, "driver_cuda_version", lambda: (12, 4))
    assert check_deps.cuda_unavailable_cause("13.0") == "driver_too_old"
    assert check_deps.cuda_unavailable_cause("12.8") == "hidden"  # a 12.8 build runs on a driver for CUDA 12.4
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert check_deps.cuda_unavailable_cause("12.8") == "unknown"
    assert check_deps.cuda_unavailable_advice("no_driver") is None


def test_one_issue_names_every_gpu_without_code(monkeypatch):
    import torch

    ccs = [(7, 0), (6, 0)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=None: ccs[i])
    monkeypatch.setattr(check_deps, "gpus_without_code", lambda t: [(0, "V100", (7, 0)), (1, "P100", (6, 0))])
    monkeypatch.setattr(check_deps, "driver_cuda_version", lambda: (13, 0))
    issues = [i for i in check_installation._torch_info({"present": True})[1] if i.code == "TORCH_NO_CODE_FOR_GPU"]
    assert len(issues) == 1 and "GPU 0 (V100" in issues[0].message and "GPU 1 (P100" in issues[0].message


def test_install_hint_lists_each_compute_capability_once():
    hint = check_deps.torch_install_hint((13, 0), [(10, 0)] * 8)
    assert "compute capability 10.0)" in hint and "10.0, 10.0" not in hint
    assert "compute capability 7.5, 10.0)" in check_deps.torch_install_hint((13, 0), [(10, 0), (7, 5), (10, 0)])
    assert check_deps.TORCH_MIN_TEXT == "2.10"


def test_torch_install_command(monkeypatch):
    monkeypatch.setattr(check_deps, "_has_pip", lambda: True)
    command = check_deps.torch_install_command("cu130")
    python = shlex.quote(sys.executable)
    assert f'"torch=={check_deps.TORCH_VERIFIED}"' in command and command.endswith("/whl/cu130")
    assert command.startswith(f"{python} -m pip uninstall -y torch && {python} -m pip install ")
    assert f"`{command}`" in check_deps.torch_install_hint((13, 0), [(10, 0)])
    pytest.importorskip("torch")
    from deeprm.model import deeprm_v12_engines as engines

    # the fa4 warning keeps a CUDA 13 or 12.8 build, and replaces an older one by the build that runs on the driver
    for cuda, driver, tag in (("13.0", None, "cu130"), ("12.8", (13, 0), "cu128"), ("12.6", (13, 0), "cu130"),
                              ("12.6", (12, 5), "cu128"), ("12.6", None, "cu128")):
        monkeypatch.setattr(engines.torch.version, "cuda", cuda)
        monkeypatch.setattr(engines, "_driver_cuda", lambda driver=driver: driver)
        assert engines._torch_fa4_command() == check_deps.torch_install_command(tag)


def test_install_commands_use_uv_where_the_environment_has_no_pip(monkeypatch):
    """`uv venv` makes no pip, and a bare `pip` may belong to another environment: print `uv pip` for this Python."""
    monkeypatch.setattr(check_deps, "_has_pip", lambda: False)
    monkeypatch.setattr(check_deps, "_has_uv", lambda: True, raising=False)
    python = shlex.quote(sys.executable)
    command = check_deps.torch_install_command("cu126")
    assert command == (
        f"uv pip uninstall --python {python} torch && uv pip install --python {python} "
        f'"torch=={check_deps.TORCH_VERIFIED}" --index-url https://download.pytorch.org/whl/cu126'
    )
    assert check_deps.pip_command("install", "x") == f"uv pip install --python {python} x"
    monkeypatch.setattr(check_deps, "_has_pip", lambda: True)
    assert check_deps.pip_command("uninstall", "-y", "x") == f"{python} -m pip uninstall -y x"


_BANNER = "| NVIDIA-SMI 580.95.05    Driver Version: 580.95.05    CUDA Version: 13.0 |\n"
_GPU_LIST = "".join(f"{i}, GPU-{i:08x}-0000-0000-0000-000000000000, NVIDIA B200, 10.0\n" for i in range(8))
_A100 = "0, GPU-88590fd1-ea38-11be-76a0-ccb260283928, NVIDIA A100-SXM4-80GB, 8.0\n"
_MISMATCH = "Failed to initialize NVML: Driver/library version mismatch\nNVML library version: 580.95\n"


@pytest.fixture
def nvidia_smi(monkeypatch):
    """Puts nvidia-smi on PATH: call it with the banner, the GPU list, the exit status and stderr it should give. It
    returns the list of commands nvidia-smi ran. Clears the nvidia-smi cache before and after, so that no other test
    sees the canned output."""
    import shutil
    import subprocess

    real_which = shutil.which
    calls = []

    def put(banner=_BANNER, gpus=_GPU_LIST, returncode=0, stderr=""):
        def run(cmd, *a, **k):
            calls.append(cmd)
            out = gpus if any(arg.startswith("--query-gpu") for arg in cmd[1:]) else banner
            return subprocess.CompletedProcess(cmd, returncode, stdout=out, stderr=stderr)

        monkeypatch.setattr(
            check_deps.shutil, "which", lambda name: "/fake/nvidia-smi" if name == "nvidia-smi" else real_which(name)
        )
        monkeypatch.setattr(check_deps.subprocess, "run", run)
        check_deps._nvidia_smi.cache_clear()
        return calls

    yield put
    check_deps._nvidia_smi.cache_clear()


@pytest.fixture
def fake_nvidia_smi(nvidia_smi):
    """nvidia-smi on PATH with eight B200s and a driver for CUDA 13.0; returns the list of commands it ran."""
    return nvidia_smi()


def test_check_runs_nvidia_smi_at_most_twice(monkeypatch, fake_nvidia_smi):
    """One deeprm check reads the driver and the GPU list once each, and keeps the -v keys."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    env, issues = check_installation.collect()
    smi = [cmd for cmd in fake_nvidia_smi if cmd[0] == "/fake/nvidia-smi"]  # the binary check also runs a command
    assert 0 < len(smi) <= 2, fake_nvidia_smi
    assert env["nvidia"] == {
        "present": True,
        "driver_version": "580.95.05",
        "driver_supports_cuda": "13.0",
        "gpus": ["NVIDIA B200"] * 8,
        "gpu_count": 8,
    }
    assert "rocm" not in env and "built_rocm" not in env["torch"]
    assert check_deps.gpu_compute_capabilities() == [(10, 0)] * 8 and check_deps.driver_cuda_version() == (13, 0)
    assert len([cmd for cmd in fake_nvidia_smi if cmd[0] == "/fake/nvidia-smi"]) <= 2


def test_check_reports_the_engine(monkeypatch):
    """deeprm check still finds the engine deeprm call run would use (the speed test that once ran here is gone)."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    info = check_installation._engine_info()
    assert "error" not in info and info["auto_engine"] == "fast"


def test_cli_without_torch_says_how_to_install_it():
    """Without PyTorch, a command prints the install hint (exit 1); importing a module raises ModuleNotFoundError."""
    import os
    import subprocess
    import sys

    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"  # PyTorch hidden: import torch raises ModuleNotFoundError
        "try:\n"
        "    import deeprm.inference.inference\n"
        "except ModuleNotFoundError as exc:\n"
        "    print('import:', exc.name)\n"
        "from deeprm.cli import main\n"
        "sys.exit(main(['call', 'run', '--help']))\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))  # find deeprm as this test does
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 1, result.stderr
    assert "import: torch" in result.stdout and "PyTorch is not installed" in result.stderr


# The driver read from libcuda, an nvidia-smi that fails, CUDA_VISIBLE_DEVICES, the PyTorch version ---------------


def test_install_hint_reads_the_driver_from_libcuda(monkeypatch):
    """Without nvidia-smi (a container started without the "utility" capability) the driver is read from libcuda, so
    a P100 gets the cu126 build and not the CPU-only one."""
    monkeypatch.setattr(check_deps, "_nvidia_smi", lambda *args: None)
    monkeypatch.setattr(check_deps, "libcuda_cuda_version", lambda: (12, 4))
    hint = check_deps.torch_install_hint(gpu_ccs=[(6, 0)])
    assert "whl/cu126" in hint and "the NVIDIA driver supports CUDA 12.4" in hint and "whl/cpu" not in hint


def test_install_hint_never_gives_the_cpu_build_for_gpus():
    """GPUs, but no driver reading at all: say so instead of a command that would lose the GPUs."""
    hint = check_deps.torch_install_hint(None, [(6, 0)])
    assert "whl/cpu" not in hint and "could not be read" in hint and "compute capability 6.0" in hint
    assert "whl/cpu" in check_deps.torch_install_hint(None, [])  # no GPU and no driver: the CPU build


def test_no_code_hint_in_a_container_without_nvidia_smi(monkeypatch):
    """PyTorch sees a P100 that its build has no code for, nvidia-smi is missing and libcuda reads CUDA 12.4."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=None: (6, 0))
    monkeypatch.setattr(check_deps, "gpus_without_code", lambda t: [(0, "Tesla P100-PCIE-16GB", (6, 0))])
    monkeypatch.setattr(check_deps, "_nvidia_smi", lambda *args: None)
    monkeypatch.setattr(check_deps, "libcuda_cuda_version", lambda: (12, 4))
    issues = [i for i in check_installation._torch_info({"present": False})[1] if i.code == "TORCH_NO_CODE_FOR_GPU"]
    assert len(issues) == 1 and "whl/cu126" in issues[0].hint and "whl/cpu" not in issues[0].hint


def test_cuda_build_without_gpu_reads_the_driver_from_libcuda(monkeypatch, cuda_build_without_gpu):
    """No nvidia-smi, but libcuda: there is a driver, so the cause is not "no_driver" (a warning that hides errors)."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(check_deps, "driver_cuda_version", lambda: None)
    monkeypatch.setattr(check_deps, "libcuda_cuda_version", lambda: (13, 0))
    assert check_deps.cuda_unavailable_cause("13.0") == "unknown"
    assert check_deps.cuda_unavailable_cause("12.8") == "unknown"
    monkeypatch.setattr(check_deps, "libcuda_cuda_version", lambda: (12, 8))
    assert check_deps.cuda_unavailable_cause("13.0") == "driver_too_old"


def test_cpu_torch_in_a_container_without_nvidia_smi_is_reported(monkeypatch):
    """Only libcuda is there (no nvidia-smi): a CPU-only PyTorch still gets the warning and the CUDA build for the
    driver."""
    import torch

    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(check_installation.platform, "machine", lambda: "x86_64")  # on ARM the hint is the CPU
    monkeypatch.setattr(check_deps, "_nvidia_smi", lambda *args: None)
    monkeypatch.setattr(check_deps, "libcuda_cuda_version", lambda: (13, 0))
    issues = check_installation._torch_info(check_installation._nvidia_info())[1]
    found = [i for i in issues if i.code == "CPU_TORCH_ON_GPU_MACHINE"]
    assert len(found) == 1 and "whl/cu130" in found[0].hint


def test_nvidia_smi_error_is_read_from_stderr_too(nvidia_smi):
    error = "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver."
    nvidia_smi(banner="", gpus="", returncode=9, stderr=error + "\n")
    assert check_deps.nvidia_smi_error() == error and check_deps.nvidia_driver_info() == (None, None)
    nvidia_smi()
    assert check_deps.nvidia_smi_error() is None  # a working nvidia-smi
    nvidia_smi(banner="", gpus="")
    assert check_deps.nvidia_smi_error() is None  # one that prints nothing: nothing to report, as before


def test_check_when_nvidia_smi_fails_after_a_driver_update(monkeypatch, nvidia_smi):
    """'Driver/library version mismatch' (a driver updated without a reboot): libcuda still reads the driver, so it is
    neither "no NVIDIA driver" nor the CPU-only build; the advice is a restart."""
    import torch

    from deeprm.inference import inference

    nvidia_smi(banner=_MISMATCH, gpus=_MISMATCH, returncode=18)
    monkeypatch.setattr(check_deps, "libcuda_cuda_version", lambda: (13, 0))
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    nvi = check_installation._nvidia_info()
    assert nvi["present"] and nvi["error"] == "Failed to initialize NVML: Driver/library version mismatch"
    assert check_deps.cuda_unavailable_cause("13.0") == "nvidia_smi_fails"
    issue = _cuda_runtime_issue(check_installation._torch_info(nvi)[1])
    assert issue.severity == "ERROR" and "`nvidia-smi` fails here" in issue.message
    assert "Restart the machine" in issue.hint and "Driver/library version mismatch" in issue.hint
    assert "Restart the machine" in inference._no_gpu_advice()  # deeprm call run says the same
    # a CPU-only PyTorch: fix the driver first, then the CUDA build for the driver libcuda reads
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(check_installation.platform, "machine", lambda: "x86_64")  # on ARM the hint is the CPU
    found = [i for i in check_installation._torch_info(nvi)[1] if i.code == "CPU_TORCH_ON_GPU_MACHINE"]
    assert len(found) == 1 and "Restart the machine" in found[0].hint and "whl/cu130" in found[0].hint
    assert "whl/cpu" not in found[0].hint and "no NVIDIA driver was found" not in found[0].hint


def test_check_where_the_driver_finds_no_gpu(monkeypatch, nvidia_smi):
    """nvidia-smi says "No devices were found" (the driver is loaded, but no GPU is passed in or bound to it): the CPU
    is expected, as without a driver, although libcuda reads the driver."""
    import torch

    nvidia_smi(banner="No devices were found\n", gpus="No devices were found\n", returncode=6)
    monkeypatch.setattr(check_deps, "libcuda_cuda_version", lambda: (13, 0))
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    nvi = check_installation._nvidia_info()
    issue = _cuda_runtime_issue(check_installation._torch_info(nvi)[1])
    assert issue.severity == "WARN" and "finds no GPU" in issue.message and "pip" not in issue.hint
    assert check_deps.cuda_unavailable_advice(check_deps.cuda_unavailable_cause("13.0")) is None  # call run: quiet
    monkeypatch.setattr(torch.version, "cuda", None)
    assert "CPU_TORCH_ON_GPU_MACHINE" not in [i.code for i in check_installation._torch_info(nvi)[1]]


_NO_COMMUNICATION = (
    "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA "
    "driver is installed and running.\n"
)


def test_check_on_a_machine_without_a_gpu_that_has_the_driver_package(monkeypatch, nvidia_smi):
    """No GPU, but the NVIDIA driver package is installed (a GPU image on a CPU instance or a login node): the kernel
    module does not load, nvidia-smi "couldn't communicate with the NVIDIA driver", and libcuda still loads. That is
    the warning of no driver, as before, not an error that says to fix the driver."""
    import torch

    from deeprm.inference import inference

    nvidia_smi(banner=_NO_COMMUNICATION, gpus=_NO_COMMUNICATION, returncode=9)
    monkeypatch.setattr(check_deps, "libcuda_cuda_version", lambda: (13, 0))
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(check_installation.platform, "machine", lambda: "x86_64")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    nvi = check_installation._nvidia_info()
    assert check_deps.cuda_unavailable_cause("13.0") == "no_driver" and not check_deps.nvidia_driver_broken()
    issues = check_installation._torch_info(nvi)[1]
    issue = _cuda_runtime_issue(issues)
    assert issue.severity == "WARN" and "no NVIDIA driver was found" in issue.message and "pip" not in issue.hint
    assert [i.code for i in issues if i.severity == "ERROR"] == []
    assert inference._no_gpu_advice() is None  # deeprm call run stays quiet, as before
    # a CPU-only PyTorch: the warning as before, with no advice to fix the driver
    monkeypatch.setattr(torch.version, "cuda", None)
    found = [i for i in check_installation._torch_info(nvi)[1] if i.code == "CPU_TORCH_ON_GPU_MACHINE"]
    assert len(found) == 1 and found[0].hint.startswith("To use the GPU, install a CUDA build: ")
    assert "fails" not in found[0].hint and "Fix the NVIDIA driver" not in found[0].hint


@pytest.mark.parametrize(
    "error, libcuda, cause",
    [
        ("Failed to initialize NVML: Driver/library version mismatch", (13, 0), "nvidia_smi_fails"),
        ("Failed to initialize NVML: Driver/library version mismatch", None, "nvidia_smi_fails"),
        ("Failed to initialize NVML: Unknown Error", (13, 0), "nvidia_smi_fails"),  # a container that lost its GPUs
        ("Failed to initialize NVML: Insufficient Permissions", (12, 8), "nvidia_smi_fails"),
        ("Failed to initialize NVML: Unknown Error", None, "no_driver"),  # nothing shows a driver
        ("Failed to initialize NVML: Driver Not Loaded", (13, 0), "no_driver"),
        (_NO_COMMUNICATION.strip(), (13, 0), "no_driver"),
        (_NO_COMMUNICATION.strip(), None, "no_driver"),
    ],
)
def test_only_a_broken_driver_is_an_error(monkeypatch, nvidia_smi, error, libcuda, cause):
    """An error (exit 1) only where nvidia-smi shows a driver that is loaded but cannot be used: the driver/library
    mismatch, or another NVML start failure while libcuda reads a driver. Any other failure is the warning of no
    driver (exit 0), as before."""
    import torch

    nvidia_smi(banner=error + "\n", gpus=error + "\n", returncode=18)
    monkeypatch.setattr(check_deps, "libcuda_cuda_version", lambda: libcuda)
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert check_deps.cuda_unavailable_cause("12.8") == cause
    issue = _cuda_runtime_issue(check_installation._torch_info(check_installation._nvidia_info())[1])
    if cause == "nvidia_smi_fails":
        assert issue.severity == "ERROR" and f'("{error}")' in issue.hint
        assert ("Restart the machine" in issue.hint) == ("mismatch" in error)
    else:
        assert issue.severity == "WARN" and "no NVIDIA driver was found" in issue.message


def test_cpu_torch_on_arm_is_told_deeprm_runs_on_the_cpu(monkeypatch, nvidia_smi):
    """A GH200 (aarch64) with a CPU-only PyTorch: DeepRM v1.2 runs on the CPU there, so the hint gives no CUDA build."""
    import torch

    nvidia_smi(gpus="0, GPU-0d1e2f3a-4b5c-6d7e-8f90-a1b2c3d4e5f6, NVIDIA GH200 480GB, 9.0\n")
    monkeypatch.setattr(check_deps, "libcuda_cuda_version", lambda: (13, 0))
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(check_installation.platform, "machine", lambda: "aarch64")
    found = [
        i
        for i in check_installation._torch_info(check_installation._nvidia_info())[1]
        if i.code == "CPU_TORCH_ON_GPU_MACHINE"
    ]
    assert len(found) == 1 and found[0].severity == "WARN"
    assert found[0].hint == (
        "On Linux on ARM (aarch64), DeepRM v1.2 runs on the CPU only: it was not tested with the CUDA builds of "
        "PyTorch for ARM."
    )
    monkeypatch.setattr(check_installation.platform, "machine", lambda: "x86_64")  # the same machine on x86_64
    found = [
        i
        for i in check_installation._torch_info(check_installation._nvidia_info())[1]
        if i.code == "CPU_TORCH_ON_GPU_MACHINE"
    ]
    assert len(found) == 1 and "whl/cu130" in found[0].hint


@pytest.mark.parametrize(
    "value, hidden",
    [
        ("", True),
        ("-1", True),
        ("1", True),  # one GPU here: index 1 names none
        ("1,0", True),  # CUDA stops at the first entry that names no GPU
        ("0", False),
        ("0,5", False),
        ("GPU-88590fd1", False),  # the start of the A100's UUID
        ("GPU-00000000", True),
        ("MIG-5ac3a6b8-7d4e-5f1a-9c2b-0d1e2f3a4b5c", False),  # nvidia-smi does not list MIG parts
        ("none", True),
    ],
)
def test_cuda_visible_devices_hide_the_gpus_only_when_they_name_none(monkeypatch, nvidia_smi, value, hidden):
    nvidia_smi(gpus=_A100)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    assert check_deps.cuda_visible_devices_hide_gpus() is hidden


def test_cuda_visible_devices_without_a_gpu_list(monkeypatch):
    """Without nvidia-smi, an index or a UUID cannot be checked: it counts as a GPU."""
    monkeypatch.setattr(check_deps, "_nvidia_smi", lambda *args: None)
    for value, hidden in (("7", False), ("GPU-00000000", False), ("", True), ("-1", True)):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
        assert check_deps.cuda_visible_devices_hide_gpus() is hidden, value


def test_a_gpu_named_in_cuda_visible_devices_is_not_called_hidden(monkeypatch, nvidia_smi):
    """A scheduler sets CUDA_VISIBLE_DEVICES=0 and PyTorch still sees no GPU: an error that says to check nvidia-smi,
    not a warning that says to unset CUDA_VISIBLE_DEVICES."""
    import torch

    nvidia_smi(banner=_BANNER.replace("13.0", "12.8"), gpus=_A100)
    monkeypatch.setattr(check_deps, "libcuda_cuda_version", lambda: (12, 8))
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    issue = _cuda_runtime_issue(check_installation._torch_info(check_installation._nvidia_info())[1])
    assert issue.severity == "ERROR" and "nvidia-smi" in issue.hint and "CUDA_VISIBLE_DEVICES" not in issue.hint
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")  # no GPU 3 here: hidden, a warning as before
    issue = _cuda_runtime_issue(check_installation._torch_info(check_installation._nvidia_info())[1])
    assert issue.severity == "WARN" and "CUDA_VISIBLE_DEVICES is set to '3'" in issue.hint


@pytest.mark.parametrize(
    "version, code",
    [
        ("2.11.0", "TORCH_NOT_VERIFIED"),  # pip install -U torch
        ("2.10.0a0+b4e4ee8", "TORCH_NOT_VERIFIED"),  # a container's own build
        ("2.10.1", "TORCH_NOT_VERIFIED"),
        ("2.10.0+cu130", None),
        ("2.10.0", None),
        ("2.9.1+cu128", "TORCH_TOO_OLD"),
    ],
)
def test_check_wants_exactly_the_verified_pytorch(monkeypatch, version, code):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch, "__version__", version)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(check_deps, "libcuda_cuda_version", lambda: (13, 0))
    monkeypatch.setattr(check_deps, "driver_cuda_version", lambda: (13, 0))
    monkeypatch.setattr(check_deps, "gpu_compute_capabilities", lambda: [(10, 0)])
    issues = check_installation._torch_info({"present": True})[1]
    found = [i for i in issues if i.code in ("TORCH_NOT_VERIFIED", "TORCH_TOO_OLD")]
    assert [i.code for i in found] == ([code] if code else [])
    if code:
        assert found[0].severity == ("WARN" if code == "TORCH_NOT_VERIFIED" else "ERROR")
        assert f"{version} is " in found[0].message and check_deps.TORCH_VERIFIED in found[0].message
        assert f'"torch=={check_deps.TORCH_VERIFIED}"' in found[0].hint and "whl/cu130" in found[0].hint


def test_no_code_hint_on_a_mixed_machine_names_the_other_gpus(monkeypatch):
    """P100 + A100 with the cu130 build: the build for both GPUs, or this build on the A100 only (which fa4 needs)."""
    import torch

    ccs = [(6, 0), (8, 0)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=None: ccs[i])
    monkeypatch.setattr(check_deps, "gpus_without_code", lambda t: [(0, "Tesla P100-PCIE-16GB", (6, 0))])
    monkeypatch.setattr(check_deps, "libcuda_cuda_version", lambda: (13, 0))
    monkeypatch.setattr(check_deps, "driver_cuda_version", lambda: (13, 0))
    issues = [i for i in check_installation._torch_info({"present": True})[1] if i.code == "TORCH_NO_CODE_FOR_GPU"]
    assert len(issues) == 1 and "whl/cu126" in issues[0].hint
    assert "run DeepRM on the other GPUs only (--gpu-pool 1)" in issues[0].hint


def test_install_commands_without_pip_or_uv(monkeypatch):
    """No pip, and uv is not on PATH (for example `python -m venv --without-pip`): the pip form, with a note that says
    what to do first, never a `uv pip` command that cannot run."""
    monkeypatch.setattr(check_deps, "_has_pip", lambda: False)
    monkeypatch.setattr(check_deps, "_has_uv", lambda: False)
    python = shlex.quote(sys.executable)
    assert check_deps.pip_command("install", "x") == f"{python} -m pip install x"
    note = check_deps.pip_note()
    assert "no pip" in note and "uv is not on PATH" in note and f"`{python} -m ensurepip`" in note
    hint = check_deps.torch_install_hint((13, 0), [(10, 0)])
    assert hint.startswith(f"`{python} -m pip uninstall -y torch && ") and hint.endswith(f"; {note})")
    assert check_installation._command("install", "x") == f"`{python} -m pip install x` ({note})"
    monkeypatch.setattr(check_deps, "_has_uv", lambda: True)
    assert check_deps.pip_command("install", "x").startswith("uv pip install") and check_deps.pip_note() == ""


def test_uv_is_looked_up_on_path(monkeypatch, tmp_path):
    from deeprm.utils import prep_binary

    if prep_binary.on_noexec_mount(str(tmp_path)):
        pytest.skip(f"{tmp_path} does not allow programs to run, so nothing in it is found on PATH")
    monkeypatch.setenv("PATH", str(tmp_path))
    assert not check_deps._has_uv()
    (tmp_path / "uv").write_text("#!/bin/sh\n")
    (tmp_path / "uv").chmod(0o755)
    assert check_deps._has_uv()


# The C++ preprocessing binary ------------------------------------------------------------------------------------


def _binary_status(monkeypatch, ok):
    from deeprm.utils import prep_binary

    st = prep_binary.Status(True, version="DeepRM Preprocessing v1.2 test") if ok else prep_binary.Status(
        False, reason="the system library libz.so.1 cannot be loaded (file too short)", hint="Install zlib."
    )
    monkeypatch.setattr(prep_binary, "status", lambda path, timeout=60: st)
    monkeypatch.setattr(prep_binary, "libraries", lambda path: [("libz.so.1", "not found")])


def test_check_reports_a_binary_that_starts(monkeypatch):
    monkeypatch.delenv("DEEPRM_PREPROCESS_BIN", raising=False)
    _binary_status(monkeypatch, ok=True)
    info, issues = check_installation._prep_binary_info()
    assert issues == [] and info["starts"] and info["source"] == "bundled" and "v1.2 test" in info["version"]


def test_check_warns_when_the_bundled_binary_cannot_start(monkeypatch):
    monkeypatch.delenv("DEEPRM_PREPROCESS_BIN", raising=False)
    _binary_status(monkeypatch, ok=False)
    info, issues = check_installation._prep_binary_info(verbose=True)
    assert [(i.severity, i.code) for i in issues] == [("WARN", "PREP_BINARY_CANNOT_START")]
    assert "libz.so.1" in issues[0].message and "Install zlib." in issues[0].hint and "Python" in issues[0].hint
    assert info["libraries"] == ["libz.so.1 => not found"]


def test_check_fails_when_the_named_binary_cannot_start(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPRM_PREPROCESS_BIN", str(tmp_path / "mine"))
    _binary_status(monkeypatch, ok=False)
    info, issues = check_installation._prep_binary_info()
    assert [(i.severity, i.code) for i in issues] == [("ERROR", "PREP_BINARY_CANNOT_START")]
    assert info["source"] == "DEEPRM_PREPROCESS_BIN" and "DEEPRM_PREPROCESS_BIN" in issues[0].hint


def test_check_hint_when_nothing_can_make_the_binary_run(monkeypatch):
    from deeprm.utils import prep_binary

    monkeypatch.delenv("DEEPRM_PREPROCESS_BIN", raising=False)
    st = prep_binary.Status(False, reason="the binary is for x86_64 and this machine is aarch64", kind="machine")
    monkeypatch.setattr(prep_binary, "status", lambda path, timeout=60: st)
    _, issues = check_installation._prep_binary_info()
    assert issues[0].severity == "WARN" and "Until then" not in issues[0].hint
    assert "uses the Python preprocessing" in issues[0].hint


def test_check_hint_for_a_missing_execute_bit_says_prep_sets_it(monkeypatch):
    from deeprm.utils import prep_binary

    monkeypatch.delenv("DEEPRM_PREPROCESS_BIN", raising=False)
    st = prep_binary.Status(False, reason="x has no execute permission", hint="Run: chmod +x x", kind="mode")
    monkeypatch.setattr(prep_binary, "status", lambda path, timeout=60: st)
    _, issues = check_installation._prep_binary_info()
    assert "sets the execute permission itself" in issues[0].hint and "Until then" not in issues[0].hint


def test_check_keeps_the_issue_when_the_library_list_fails(monkeypatch):
    from deeprm.utils import prep_binary

    monkeypatch.delenv("DEEPRM_PREPROCESS_BIN", raising=False)
    st = prep_binary.Status(False, reason="the system library libz.so.1 cannot be loaded", hint="h", kind="library")
    monkeypatch.setattr(prep_binary, "status", lambda path, timeout=60: st)

    def broken(path):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")

    monkeypatch.setattr(prep_binary, "libraries", broken)
    info, issues = check_installation._prep_binary_info(verbose=True)
    assert [i.code for i in issues] == ["PREP_BINARY_CANNOT_START"] and "could not list" in info["libraries"][0]


@pytest.mark.parametrize("owned, severity", [(True, "WARN"), (False, "ERROR")])
def test_check_for_a_named_binary_without_execute_bit(monkeypatch, tmp_path, owned, severity):
    """deeprm call prep sets the execute bit of a binary this user owns, so check warns; otherwise prep stops."""
    from deeprm.utils import prep_binary

    monkeypatch.setenv("DEEPRM_PREPROCESS_BIN", str(tmp_path / "mine"))
    st = prep_binary.Status(False, reason="x has no execute permission", hint="Run `chmod +x x`.", kind="mode")
    monkeypatch.setattr(prep_binary, "status", lambda path, timeout=60: st)
    monkeypatch.setattr(check_installation, "_owned", lambda path: owned)
    _, issues = check_installation._prep_binary_info()
    assert [(i.severity, i.code) for i in issues] == [(severity, "PREP_BINARY_CANNOT_START")]
    if owned:
        assert "sets the execute permission itself" in issues[0].hint
