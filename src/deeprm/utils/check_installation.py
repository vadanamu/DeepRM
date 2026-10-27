"""`deeprm check`: checks PyTorch and torchmetrics, the GPUs and the engine they would run, and the C++ preprocessing
binary, and prints each problem with a hint to fix it."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from deeprm.utils import check_deps, prep_binary
from deeprm.utils.logging import get_logger

TRAIN_EXTRA = '"deeprm[train]"'

log = get_logger(__name__)


@dataclass
class Issue:
    """One problem that deeprm check found: its severity ("ERROR" or "WARN"), message, hint (how to fix it, or None)
    and code (a fixed name such as "TORCH_MISSING")."""

    severity: str  # "ERROR" | "WARN"
    message: str
    hint: str | None = None
    code: str | None = None  # machine-usable code, e.g., "TORCH_MISSING"


def _nvidia_info() -> Dict[str, Any]:
    """What nvidia-smi reports (the "nvidia" block of `deeprm check -v`): present, driver_version,
    driver_supports_cuda, gpus (names) and gpu_count, or its error. It reads the outputs check_deps keeps, so it runs
    nothing new."""
    banner = check_deps.nvidia_driver_info()
    if banner is None:
        return {"present": False}
    info: Dict[str, Any] = {"present": True}
    error = check_deps.nvidia_smi_error()
    if error:
        info["error"] = error
    driver, cuda = banner
    if driver:
        info["driver_version"] = driver
    if cuda:
        info["driver_supports_cuda"] = cuda
    gpus = check_deps.nvidia_gpus()
    if gpus is not None:
        info["gpus"] = [name for _, name, _ in gpus]
        info["gpu_count"] = len(gpus)
    return info


def _cuda_runtime_issue(built_cuda: str) -> Issue:
    """The issue for a CUDA build of PyTorch that sees no GPU. Its severity and advice depend on the cause
    (check_deps.cuda_unavailable_cause): a warning when there is no NVIDIA driver or GPU, or CUDA_VISIBLE_DEVICES hides
    the GPUs, an error otherwise."""
    cause = check_deps.cuda_unavailable_cause(built_cuda)
    build = f"This PyTorch is a CUDA {built_cuda} build"
    slow = "DeepRM would run on the CPU, which is much slower."
    if cause == "no_driver":
        return Issue(
            severity="WARN",
            message=f"{build}, and no NVIDIA driver was found. {slow}",
            hint="If this machine has an NVIDIA GPU, install its driver (`nvidia-smi` should then list the GPU).",
            code="CUDA_RUNTIME_MISSING",
        )
    if cause == "no_gpu":
        return Issue(
            severity="WARN",
            message=f"{build}, and the NVIDIA driver finds no GPU (nvidia-smi: {check_deps.NVIDIA_SMI_NO_GPU}). {slow}",
            hint="If this machine has an NVIDIA GPU, check that the system sees it (in a container or a virtual "
            "machine: that it is passed in); `nvidia-smi` should then list it.",
            code="CUDA_RUNTIME_MISSING",
        )
    driver = check_deps.driver_cuda()
    if cause == "nvidia_smi_fails":
        message = f"{build}, but `nvidia-smi` fails here, so PyTorch cannot use a GPU. {slow}"
    elif cause == "hidden":
        message = f"{build}, but it sees no GPU. {slow}"
    elif cause == "driver_too_old":
        message = f"{build}, but the NVIDIA driver supports only CUDA {driver[0]}.{driver[1]}. {slow}"
    else:
        message = (
            f"{build}, but it cannot use a GPU here, although the NVIDIA driver supports CUDA {driver[0]}.{driver[1]}. "
            f"{slow}"
        )
    return Issue(
        severity="WARN" if cause == "hidden" else "ERROR",
        message=message,
        hint=check_deps.cuda_unavailable_advice(cause),
        code="CUDA_RUNTIME_MISSING",
    )


def _torch_info(nvi: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Issue]]:
    """PyTorch facts and issues; `nvi` is the result of _nvidia_info()."""
    issues: List[Issue] = []
    info: Dict[str, Any] = {}
    try:
        import torch  # type: ignore

        info["present"] = True
        info["version"] = getattr(torch, "__version__", None)
        info["built_cuda"] = getattr(torch.version, "cuda", None)
        # Avoid crashing when CUDA libs missing
        try:
            info["cuda_available"] = bool(torch.cuda.is_available())
            info["cuda_device_count"] = int(torch.cuda.device_count()) if info["cuda_available"] else 0
        except Exception:
            info["cuda_available"] = False
            info["cuda_device_count"] = 0
    except Exception as e:
        info["present"] = False
        issues.append(
            Issue(
                severity="ERROR",
                message=f"PyTorch is not importable ({e}).",
                hint=check_deps.torch_install_hint(),
                code="TORCH_MISSING",
            )
        )
        return info, issues

    # compute capability of each GPU PyTorch sees, for the install hints; without one they ask nvidia-smi
    try:
        seen = [tuple(torch.cuda.get_device_capability(i)) for i in range(info["cuda_device_count"])] or "auto"
    except Exception:  # noqa: BLE001 -- a check must never fail
        seen = "auto"
    version = check_deps.version_tuple(info.get("version"))
    if version is not None and version < check_deps.TORCH_MIN:
        issues.append(
            Issue(
                severity="ERROR",
                message=(
                    f"PyTorch {info['version']} is older than {check_deps.TORCH_VERIFIED}, the version DeepRM v1.2 "
                    "needs."
                ),
                hint=check_deps.torch_install_hint(gpu_ccs=seen),
                code="TORCH_TOO_OLD",
            )
        )
    elif version is not None and str(info["version"]).split("+")[0] != check_deps.TORCH_VERIFIED:
        # DeepRM requires exactly this version: a newer one (pip install -U torch) is not what it was verified with
        issues.append(
            Issue(
                severity="WARN",
                message=(
                    f"PyTorch {info['version']} is not {check_deps.TORCH_VERIFIED}, the version DeepRM v1.2 requires "
                    "and was verified with."
                ),
                hint=f"Install the verified version: {check_deps.torch_install_hint(gpu_ccs=seen)}",
                code="TORCH_NOT_VERIFIED",
            )
        )

    # Built for CUDA but sees no GPU
    if info.get("built_cuda") and not info.get("cuda_available"):
        issues.append(_cuda_runtime_issue(info["built_cuda"]))

    # A CUDA build without code for a GPU here (for example the CUDA 13 build on a V100 or a P100): one issue that
    # names every such GPU
    missing = check_deps.gpus_without_code(torch)
    if missing:
        gpus = ", ".join(f"GPU {index} ({name}, compute capability {cc[0]}.{cc[1]})" for index, name, cc in missing)
        them = "it" if len(missing) == 1 else "them"
        all_ccs = [tuple(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
        hint = f"Install a build for the GPUs here: {check_deps.torch_install_hint(gpu_ccs=all_ccs)}"
        others = [str(i) for i in range(len(all_ccs)) if i not in {index for index, _, _ in missing}]
        if others:  # a mixed machine: this build still runs the other GPUs (fa4 may need such a build)
            hint += f" Or keep this build and run DeepRM on the other GPUs only (--gpu-pool {' '.join(others)})."
        issues.append(
            Issue(
                severity="ERROR",
                message=(
                    f"This PyTorch ({info['version']}) has no code for {gpus}: DeepRM would stop with 'no kernel "
                    f"image is available' on {them}."
                ),
                hint=hint,
                code="TORCH_NO_CODE_FOR_GPU",
            )
        )

    # CPU build on a GPU machine (notice-level -> WARN). nvidia-smi is there and does not say it finds no GPU, or
    # only libcuda is (a container started without nvidia-smi).
    driver_here = nvi.get("present") or check_deps.libcuda_cuda_version() is not None
    if not info.get("built_cuda") and driver_here and not check_deps.nvidia_smi_finds_no_gpu():
        if platform.machine().lower() in ("aarch64", "arm64"):  # the installation guide: the CPU only on ARM
            hint = (
                "On Linux on ARM (aarch64), DeepRM v1.2 runs on the CPU only: it was not tested with the CUDA builds "
                "of PyTorch for ARM."
            )
        elif check_deps.nvidia_driver_broken():  # fix the driver first; its CUDA version may not be readable
            hint = check_deps.cuda_unavailable_advice("nvidia_smi_fails")
            if check_deps.driver_cuda():
                hint += f" Then install a CUDA build of PyTorch: {check_deps.torch_install_hint()}"
        else:
            hint = f"To use the GPU, install a CUDA build: {check_deps.torch_install_hint()}"
        issues.append(
            Issue(
                severity="WARN",
                message="CPU-only PyTorch detected while an NVIDIA driver is present: DeepRM will run on the CPU.",
                hint=hint,
                code="CPU_TORCH_ON_GPU_MACHINE",
            )
        )

    return info, issues


def _command(*pip_args: str) -> str:
    """check_deps.pip_command(PIP_ARGS) in backticks, followed by check_deps.pip_note() where there is one."""
    note = check_deps.pip_note()
    return f"`{check_deps.pip_command(*pip_args)}`" + (f" ({note})" if note else "")


def _torchmetrics_info(require_train: bool) -> Tuple[Dict[str, Any], List[Issue]]:
    info: Dict[str, Any] = {}
    issues: List[Issue] = []
    try:
        import torchmetrics  # type: ignore

        info["present"] = True
        info["version"] = getattr(torchmetrics, "__version__", None)
        version = check_deps.version_tuple(info["version"])
        if version is not None and version < (1, 0):
            issues.append(
                Issue(
                    severity="ERROR" if require_train else "WARN",
                    message=f"torchmetrics {info['version']} is too old for deeprm train (it needs 1.0 or newer).",
                    hint=_command("install", "-U", TRAIN_EXTRA),
                    code="TORCHMETRICS_TOO_OLD",
                )
            )
    except Exception as e:
        info["present"] = False
        sev = "ERROR" if require_train else "WARN"
        issues.append(
            Issue(
                severity=sev,
                message=f"torchmetrics is not importable ({e}).",
                hint=f"Needed only for deeprm train. Install with: {_command('install', TRAIN_EXTRA)}",
                code="TORCHMETRICS_MISSING",
            )
        )
    return info, issues


def _engine_info() -> Dict[str, Any]:
    """Which engine `deeprm call run --engine auto` would use for the v1.2 model on this machine."""
    info: Dict[str, Any] = {}
    try:
        import torch  # type: ignore

        from deeprm.model import deeprm_v12_engines as engines

        info["fa4_installed"] = engines.fa4_installed()
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        if device.type == "cuda":
            info["gpu0_compute_capability"] = "{}.{}".format(*torch.cuda.get_device_capability(device))
        engine, reason = engines.resolve_engine("auto", device)
        info["auto_engine"] = engine
        info["auto_engine_reason"] = reason
    except Exception as e:  # noqa: BLE001 -- informational only
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def _fa4_warning() -> Tuple[str, str] | None:
    """(summary, framed warning) when fa4 would not be used on a GPU here for a reason the user can fix, or because fa4
    was not tested on that GPU; else None."""
    try:
        import torch  # type: ignore

        from deeprm.model import deeprm_v12_engines as engines

        if not torch.cuda.is_available():
            return None
        gpus = []
        for index in range(torch.cuda.device_count()):
            device = torch.device("cuda", index)
            gpus.append((index, device, engines.resolve_engine("auto", device)[0]))
        return engines.fa4_warning(gpus, uses="deeprm call run would use")
    except Exception:  # noqa: BLE001 -- a check must never fail
        return None


def _owned(path: str) -> bool:
    """Whether this user may change the file's permissions (its owner, or root)."""
    try:
        return os.geteuid() in (0, os.stat(path).st_uid)
    except OSError:
        return False


def _prep_binary_info(verbose: bool = False) -> Tuple[Dict[str, Any], List[Issue]]:
    """Whether the C++ preprocessing binary that deeprm call prep would run can start here, and if not why."""
    info: Dict[str, Any] = {}
    try:
        named = os.environ.get("DEEPRM_PREPROCESS_BIN")
        path = prep_binary.named_path(named) if named else prep_binary.BUNDLED
        st = prep_binary.status(path)
        info = {"path": path, "source": "DEEPRM_PREPROCESS_BIN" if named else "bundled", "starts": st.ok}
        if st.ok:
            info["version"] = st.version
            return info, []
        info["reason"] = st.reason
        if verbose:
            try:
                info["libraries"] = [f"{name} => {where}" for name, where in prep_binary.libraries(path)]
            except Exception as e:  # noqa: BLE001 -- the list is extra detail; the issue below must still be kept
                info["libraries"] = [f"(could not list: {type(e).__name__}: {e})"]
        if st.kind == "mode" and _owned(path):
            # deeprm call prep sets the execute bits itself before it checks the binary, for a bundled and a named
            # binary alike
            message = f"The C++ preprocessing binary cannot start yet: {st.reason}."
            hint = "deeprm call prep sets the execute permission itself, so it can use the binary."
            return info, [Issue(severity="WARN", message=message, hint=hint, code="PREP_BINARY_CANNOT_START")]
        if named:
            # deeprm call prep stops when a binary given with DEEPRM_PREPROCESS_BIN cannot start
            hint = st.hint or "Use a binary that runs on this system."
            hint += " Or unset DEEPRM_PREPROCESS_BIN to use the bundled binary."
            message = f"The preprocessing binary given with DEEPRM_PREPROCESS_BIN cannot start: {st.reason}."
            return info, [Issue(severity="ERROR", message=message, hint=hint, code="PREP_BINARY_CANNOT_START")]
        if st.hint:
            hint = f"{st.hint} Until then deeprm call prep uses the Python preprocessing, which is 2 to 5 times slower."
        else:
            hint = "deeprm call prep uses the Python preprocessing on this system, which is 2 to 5 times slower."
        message = f"The C++ preprocessing binary cannot start: {st.reason}."
        return info, [Issue(severity="WARN", message=message, hint=hint, code="PREP_BINARY_CANNOT_START")]
    except Exception as e:  # noqa: BLE001 -- a check must never fail
        info["error"] = f"{type(e).__name__}: {e}"
        return info, []


def collect(require_train: bool = False, verbose: bool = False) -> Tuple[Dict[str, Any], List[Issue]]:
    """Runs the checks of deeprm check. Returns (environment details, problems found).

    Args:
        require_train: Report a missing or too old torchmetrics (needed only for deeprm train) as an error, not a
            warning.
        verbose: For a preprocessing binary that cannot start, also list the shared libraries it would load.
    """
    env: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
    }
    nvi = _nvidia_info()
    if nvi.get("present"):
        env["nvidia"] = nvi

    torch_info, torch_issues = _torch_info(nvi)
    # torchmetrics needs PyTorch: without it, the PyTorch error says all there is to say
    tmetrics_info, tmetrics_issues = (
        _torchmetrics_info(require_train=require_train) if torch_info.get("present") else ({}, [])
    )

    env["torch"] = torch_info
    env["torchmetrics"] = tmetrics_info
    env["engine"] = _engine_info()

    issues = torch_issues + tmetrics_issues
    env["preprocess_binary"], binary_issues = _prep_binary_info(verbose=verbose)
    issues += binary_issues
    fa4 = _fa4_warning()
    if fa4:
        env["engine"]["fa4_warning"] = fa4[1]
        issues.append(
            Issue(
                severity="WARN",
                message=fa4[0],
                hint="The steps are in the box at the end of this output.",
                code="FA4_NOT_USED",
            )
        )
    return env, issues


def _count(n: int, word: str) -> str:
    """Returns "1 error" or "2 errors"."""
    return f"{n} {word}{'' if n == 1 else 's'}"


def parser(prog: str | None = None) -> argparse.ArgumentParser:
    """The command-line parser of deeprm check."""
    p = argparse.ArgumentParser(
        prog=prog or "deeprm check",
        description="Check the DeepRM installation: PyTorch, torchmetrics, the GPU and the C++ preprocessing binary.",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Also print all environment details (as JSON).")
    p.add_argument(
        "--require-train",
        action="store_true",
        default=False,
        help="Treat a missing or too old torchmetrics (needed for training) as an error, not a warning.",
    )
    return p


def main(argv: List[str] | None = None) -> int:
    """Runs deeprm check with the arguments `argv` (None: the command line). Returns 1 if it found an error, else 0."""
    args = parser().parse_args(argv)
    env, issues = collect(require_train=args.require_train, verbose=args.verbose)
    warnings = [i for i in issues if i.severity == "WARN"]
    errors = [i for i in issues if i.severity == "ERROR"]

    if errors:
        log.error(f"Environment checks failed with {_count(len(errors), 'error')}:")
        for i, issue in enumerate(errors, start=1):
            log.error(f"{i}. {issue.code}")
            log.error(f"\t- {issue.message}")
            if issue.hint:
                log.error(f"\t- Hint: {issue.hint}")
        if warnings:
            verb = "was" if len(warnings) == 1 else "were"
            log.warning(f"Additionally, {_count(len(warnings), 'warning')} {verb} found:")
            for i, issue in enumerate(warnings, start=1):
                log.warning(f"{i}. {issue.code}")
                log.warning(f"\t{issue.message}")
                if issue.hint:
                    log.warning(f"\tHint: {issue.hint}")

        log.warning("Re-run with `--verbose` flag to see full environment details.")

    elif warnings:
        log.warning(f"Environment checks completed with {_count(len(warnings), 'warning')}:")
        for i, issue in enumerate(warnings, start=1):
            log.warning(f"{i}. {issue.code}")
            log.warning(f"\t{issue.message}")
            if issue.hint:
                log.warning(f"\tHint: {issue.hint}")

        log.warning("Re-run with `--verbose` flag to see full environment details.")

    else:
        log.info("Environment checks passed successfully.")

    engine = env.get("engine", {})
    if engine.get("auto_engine"):
        name, reason = engine["auto_engine"], engine["auto_engine_reason"]
        log.info(f"DeepRM v1.2 model engine on this machine: {name}. Reason: {reason}.")
    binary = env.get("preprocess_binary", {})
    if binary.get("starts"):
        log.info(f"C++ preprocessing binary: starts ({binary['version']}).")
    elif "starts" in binary:
        log.info("C++ preprocessing binary: cannot start (see PREP_BINARY_CANNOT_START above).")
    if engine.get("fa4_warning"):
        log.warning("%s", engine["fa4_warning"])  # last, so it is not lost above the other lines

    if args.verbose:
        print(json.dumps(env, indent=2))

    return 1 if errors else 0


def entry(argv: List[str] | None = None) -> int:
    """Entry point for the CLI."""
    return main(argv) if argv is not None else main(sys.argv[1:])
