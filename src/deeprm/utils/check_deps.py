"""Checks of the NVIDIA driver, the GPUs and the PyTorch build, and the PyTorch install command that fits this machine.

Used by deeprm check, and by the advice DeepRM prints when PyTorch is missing or cannot use a GPU.
"""

import functools
import importlib.util
import os
import re
import shlex
import shutil
import subprocess
import sys

# Oldest supported PyTorch. The install hints install TORCH_VERIFIED, the tested version.
TORCH_MIN = (2, 10)
TORCH_MIN_TEXT = ".".join(map(str, TORCH_MIN))
TORCH_VERIFIED = "2.10.0"
# PyTorch 2.10.0's CUDA builds, in order of preference: (tag, GPU architectures with code, CUDA version the driver
# must support). Architectures from PyTorch v2.10.0 .ci/manywheel/build_cuda.sh. A 12.x build runs on any CUDA 12
# driver (R525+); a 13.x build needs R580+.
TORCH_CUDA_BUILDS = (
    ("cu130", ((7, 5), (8, 0), (8, 6), (9, 0), (10, 0), (12, 0)), (13, 0)),  # T4 / RTX 20 to B200 / RTX 50
    ("cu128", ((7, 0), (7, 5), (8, 0), (8, 6), (9, 0), (10, 0), (12, 0)), (12, 0)),  # adds V100
    ("cu126", ((5, 0), (6, 0), (7, 0), (7, 5), (8, 0), (8, 6), (9, 0)), (12, 0)),  # adds P100, GTX 10; no Blackwell
)
TORCH_INDEX_URL = "https://download.pytorch.org/whl/{}"
# nvidia-smi can take several seconds on a busy machine with many GPUs. A timeout counts as no nvidia-smi: the driver is
# then read from libcuda only (driver_cuda).
NVIDIA_SMI_TIMEOUT = 20
NVIDIA_SMI_NO_GPU = "No devices were found"  # what nvidia-smi prints where the driver is installed but finds no GPU


@functools.cache
def _nvidia_smi(*args):
    """Output of `nvidia-smi ARGS`, also on a non-zero exit: its stdout, or its stderr if a failed run printed nothing
    to stdout. None if it is missing, cannot start or times out."""
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return None
    try:
        result = subprocess.run([nvsmi, *args], capture_output=True, text=True, timeout=NVIDIA_SMI_TIMEOUT)
    except Exception:  # noqa: BLE001 -- a hint must never fail
        return None
    return result.stdout if result.stdout.strip() or result.returncode == 0 else result.stderr


def nvidia_driver_info():
    """(driver version, CUDA version) from the `nvidia-smi` banner, such as ("580.95.05", "13.0").

    None if nvidia-smi is missing or cannot run; either string is None if the banner does not show it.
    """
    out = _nvidia_smi()
    if out is None:
        return None
    driver = re.search(r"Driver Version:\s*([0-9.]+)", out)
    cuda = re.search(r"CUDA Version:\s*([0-9.]+)", out)
    return (driver.group(1) if driver else None, cuda.group(1) if cuda else None)


def nvidia_smi_error():
    """The first line nvidia-smi printed when it runs but shows no driver version; None if it is missing, works or
    printed nothing.

    For example "Failed to initialize NVML: Driver/library version mismatch" (a driver updated without a reboot), or
    "No devices were found".
    """
    out = _nvidia_smi()
    if out is None or "Driver Version:" in out:
        return None
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    return lines[0] if lines else None


def nvidia_smi_finds_no_gpu():
    """True if nvidia-smi says the driver finds no GPU ("No devices were found"): a machine where the CPU is
    expected."""
    return (nvidia_smi_error() or "").startswith(NVIDIA_SMI_NO_GPU)


def nvidia_driver_broken():
    """True if nvidia-smi fails in a way that shows an NVIDIA driver that is loaded but cannot be used, so a GPU is
    expected: "Driver/library version mismatch" (a driver updated without a reboot), or another "Failed to initialize
    NVML" error (other than "Driver Not Loaded") while libcuda reads a driver.

    Other failures do not count: a machine without a GPU where the NVIDIA driver package is installed prints "couldn't
    communicate with the NVIDIA driver" (its kernel module does not load), and libcuda loads there too.
    """
    error = (nvidia_smi_error() or "").lower()
    if "mismatch" in error:
        return True
    return error.startswith("failed to initialize nvml") and "not loaded" not in error and bool(libcuda_cuda_version())


def driver_cuda_version():
    """Newest CUDA version the NVIDIA driver supports, from the nvidia-smi banner, as (major, minor).

    None without nvidia-smi or a banner. driver_cuda() also reads libcuda.
    """
    info = nvidia_driver_info()
    return version_tuple(info[1]) if info else None


def libcuda_cuda_version():
    """Newest CUDA version the NVIDIA driver library supports (cuDriverGetVersion), as (major, minor).

    None if libcuda cannot be loaded. This is the library CUDA programs in this process use, and it needs no subprocess.
    """
    try:
        import ctypes

        version = ctypes.c_int(0)
        if ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(version)) != 0 or version.value <= 0:
            return None
        return version.value // 1000, version.value % 1000 // 10
    except Exception:  # noqa: BLE001 -- no NVIDIA driver library
        return None


def driver_cuda():
    """Newest CUDA version the NVIDIA driver supports, as (major, minor); None without a driver.

    Read from libcuda first, and from the nvidia-smi banner (driver_cuda_version) only if libcuda cannot be loaded:
    nvidia-smi can be missing (a container started without the "utility" capability), too slow, or failing (a driver
    updated without a reboot) while libcuda works.
    """
    return libcuda_cuda_version() or driver_cuda_version()


def _nvidia_gpu_rows():
    """Every GPU `nvidia-smi` lists, as (index, UUID, name, compute capability or None); None if it cannot run."""
    out = _nvidia_smi("--query-gpu=index,uuid,name,compute_cap", "--format=csv,noheader")
    if out is None:
        return None
    rows = []
    for line in out.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 4 and fields[0].isdigit():  # skips error lines such as "No devices were found"
            rows.append((int(fields[0]), fields[1], ", ".join(fields[2:-1]), version_tuple(fields[-1])))
    return rows


def nvidia_gpus():
    """Every GPU `nvidia-smi` lists, as (index, name, compute capability or None); None if nvidia-smi cannot run.

    nvidia-smi ignores CUDA_VISIBLE_DEVICES, so this can list GPUs that PyTorch does not see.
    """
    rows = _nvidia_gpu_rows()
    return None if rows is None else [(index, name, cc) for index, _, name, cc in rows]


def gpu_compute_capabilities():
    """Compute capability (major, minor) of every NVIDIA GPU, from `nvidia-smi`; empty if unknown."""
    return [cc for _, _, cc in nvidia_gpus() or () if cc is not None]


def arch_supported(cc, archs):
    """True if a build with code for `archs` runs on compute capability `cc`: code for X.Y runs on X.Z for Z >= Y."""
    return any(major == cc[0] and minor <= cc[1] for major, minor in archs)


def torch_build_archs(arch_list):
    """(major, minor) of each GPU architecture in torch.cuda.get_arch_list() (for example "sm_75" -> (7, 5))."""
    archs = []
    for name in arch_list or ():
        m = re.fullmatch(r"sm_([0-9]+)([0-9])a?", name)
        if m:
            archs.append((int(m.group(1)), int(m.group(2))))
    return archs


def torch_index_for(driver_cuda, gpu_ccs=()):
    """PyTorch wheel index tag for this machine, such as "cu130"; "cpu" without a driver; None if no build fits."""
    if driver_cuda is None:
        return "cpu"
    for tag, archs, need in TORCH_CUDA_BUILDS:
        if tuple(driver_cuda) >= need and all(arch_supported(cc, archs) for cc in gpu_ccs):
            return tag
    return None


def torch_install_hint(driver="auto", gpu_ccs="auto"):
    """One-line install instruction for a PyTorch build that fits this machine: the command, then the reason.

    When no build fits, it returns advice instead. It never gives the CPU-only build for GPUs it is given.

    Args:
        driver: (major, minor) CUDA version the driver supports, None for no driver, or "auto" to read it
            (driver_cuda: libcuda, then nvidia-smi).
        gpu_ccs: GPU compute capabilities, or "auto" to ask nvidia-smi.
    """
    if driver == "auto":
        driver = driver_cuda()
    if gpu_ccs == "auto":
        gpu_ccs = gpu_compute_capabilities()
    # each compute capability once: eight B200s give "10.0"
    gpus = ", ".join(f"{a}.{b}" for a, b in sorted({tuple(cc) for cc in gpu_ccs}))
    if driver is None and gpu_ccs:  # GPUs, but the driver could not be read: the CPU-only build would lose them
        return (
            f"The CUDA version of the NVIDIA driver could not be read (from libcuda or nvidia-smi), so DeepRM cannot "
            f"choose a PyTorch build for the GPUs here (compute capability {gpus}). Install the build for the driver "
            "and the GPUs (see the installation guide)."
        )
    tag = torch_index_for(driver, gpu_ccs)
    if tag is None:
        if tuple(driver) < min(need for _, _, need in TORCH_CUDA_BUILDS):
            return (
                f"Update the NVIDIA driver: it supports CUDA {driver[0]}.{driver[1]}, and PyTorch "
                f"{TORCH_VERIFIED} needs a driver for CUDA 12.0 or newer (R525+)."
            )
        return (
            f"No PyTorch {TORCH_VERIFIED} build has code for all GPUs here (compute capability {gpus}). Install the "
            "build for the GPUs you use (see the installation guide) and choose them with --gpu-pool."
        )
    if tag == "cpu":
        where = "no NVIDIA driver was found"
    else:
        where = f"the NVIDIA driver supports CUDA {driver[0]}.{driver[1]}"
        if gpus:
            where += f"; GPU compute capability {gpus}"
    note = pip_note()
    if note:
        where += f"; {note}"
    return f"`{torch_install_command(tag)}` ({where})"


def _has_pip():
    return importlib.util.find_spec("pip") is not None


def _has_uv():
    return shutil.which("uv") is not None


def pip_command(*args):
    """`<this Python> -m pip ARGS`, or `uv pip ARGS --python <this Python>` where this environment has no pip.

    `uv venv` makes environments without pip. A bare `pip` is never printed: it may belong to another environment.
    The uv form is printed only if uv is on PATH; without pip and uv, the pip form is printed and pip_note() says
    what to do first.
    """
    python = shlex.quote(sys.executable or "python")
    if _has_pip() or not _has_uv():
        return " ".join([python, "-m", "pip", *args])
    command, *rest = args
    if command == "uninstall":
        rest = [a for a in rest if a != "-y"]  # uv pip uninstall does not ask
    return " ".join(["uv", "pip", command, "--python", python, *rest])


def pip_note():
    """Where this environment has no pip and uv is not on PATH, so a pip_command cannot run as printed: what to do
    first. Else ""."""
    if _has_pip() or _has_uv():
        return ""
    python = shlex.quote(sys.executable or "python")
    return (
        f"this environment has no pip and uv is not on PATH: add pip with `{python} -m ensurepip` first, or install "
        "uv and run deeprm check again"
    )


def torch_install_command(tag):
    """Command that replaces PyTorch with the tested PyTorch version from wheel index `tag`, in this Python's
    environment."""
    url = TORCH_INDEX_URL.format(tag)
    pin = f'"torch=={TORCH_VERIFIED}"'
    return f'{pip_command("uninstall", "-y", "torch")} && {pip_command("install", pin, "--index-url", url)}'


def cuda_visible_devices_hide_gpus():
    """True if CUDA_VISIBLE_DEVICES is set and names no GPU here, so it hides every GPU from PyTorch.

    CUDA uses the entries up to the first one that names no GPU, so the first entry decides. "" and "-1" name none, nor
    does an index or a GPU UUID (or the start of one) that nvidia-smi does not list. Without a GPU list from nvidia-smi
    an index or a UUID counts as a GPU, and a MIG UUID always does (nvidia-smi lists the GPUs, not their MIG parts).
    """
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return False
    first = value.split(",")[0].strip()
    if first.isdigit() or first.startswith("GPU-"):
        rows = _nvidia_gpu_rows()
        if not rows:  # nothing to check it against
            return False
        if first.isdigit():
            return int(first) >= len(rows)
        return not any(uuid.startswith(first) for _, uuid, _, _ in rows)
    return not first.startswith("MIG-")


def cuda_unavailable_cause(built_cuda):
    """Why a CUDA build of PyTorch sees no GPU, checked in this order:

    "no_gpu" (nvidia-smi finds no GPU), "nvidia_smi_fails" (nvidia-smi shows a driver that is loaded but cannot be used,
    for example after a driver update without a reboot: nvidia_driver_broken), "no_driver" (also where nvidia-smi fails
    in another way, as on a machine without a GPU that has the driver package), "driver_too_old", "hidden"
    (CUDA_VISIBLE_DEVICES names no GPU) or "unknown".

    Only CUDA major versions are compared (`built_cuda` such as "13.0"): a 12.8 build runs on a CUDA 12.4 driver.
    """
    if nvidia_smi_finds_no_gpu():
        return "no_gpu"
    if nvidia_driver_broken():
        return "nvidia_smi_fails"
    if nvidia_smi_error() is not None:  # such as "couldn't communicate with the NVIDIA driver": no driver, as before
        return "no_driver"
    driver = driver_cuda()
    if driver is None:
        return "no_driver"
    built = version_tuple(built_cuda)
    if built is not None and tuple(driver) < (built[0], 0):
        return "driver_too_old"
    if cuda_visible_devices_hide_gpus():
        return "hidden"
    return "unknown"


def cuda_unavailable_advice(cause):
    """Advice for a cause from cuda_unavailable_cause; None for "no_driver" and "no_gpu", where the CPU is expected."""
    if cause == "nvidia_smi_fails":
        error = nvidia_smi_error() or ""
        if "mismatch" in error.lower():
            return (
                f'`nvidia-smi` fails here ("{error}"): the NVIDIA driver was updated, but the old one is still loaded. '
                "Restart the machine (or reload the NVIDIA kernel modules); `nvidia-smi` should then list the GPUs."
            )
        return (
            f'`nvidia-smi` fails here ("{error}"). Fix the NVIDIA driver so that `nvidia-smi` lists the GPUs (in a '
            "container: check that the GPUs are passed in)."
        )
    if cause == "hidden":
        value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        return (
            f"CUDA_VISIBLE_DEVICES is set to {value!r}. PyTorch uses only the GPUs it names (nvidia-smi ignores it), "
            "and here PyTorch sees none. To use the GPUs, unset it, or set it to the numbers of the GPUs to use (for "
            "example 0,1)."
        )
    if cause == "driver_too_old":
        return f"The NVIDIA driver is too old for this PyTorch build. Install the build for it: {torch_install_hint()}"
    if cause == "unknown":
        return (
            "Check that `nvidia-smi` lists the GPUs and that this user can use them (in a container: that the GPUs are "
            f"passed in). If it does, the PyTorch install may be broken; reinstall it: {torch_install_hint()}"
        )
    return None


def gpus_without_code(torch_module):
    """GPUs this PyTorch build has no code for, as (index, name, (major, minor)); empty without a GPU."""
    try:
        if not torch_module.cuda.is_available():
            return []
        archs = torch_build_archs(torch_module.cuda.get_arch_list())
        if not archs:
            return []
        missing = []
        for i in range(torch_module.cuda.device_count()):
            cc = tuple(torch_module.cuda.get_device_capability(i))
            if not arch_supported(cc, archs):
                missing.append((i, torch_module.cuda.get_device_name(i), cc))
        return missing
    except Exception:  # noqa: BLE001 -- a check must never fail
        return []


def version_tuple(version):
    """(major, minor) of a version string such as "2.10.0+cu130", or None."""
    m = re.match(r"([0-9]+)\.([0-9]+)", str(version or ""))
    return (int(m.group(1)), int(m.group(2))) if m else None
