"""Can the C++ preprocessing binary start here, and if not, why? Used by deeprm call prep and deeprm check.

This module imports nothing heavy (no pod5, pysam or torch), so deeprm check can use it on any system.
"""

from __future__ import annotations

import errno
import os
import platform
import re
import shutil
import struct
import subprocess
from dataclasses import dataclass
from typing import List, Tuple

# Packages that provide a library a self-built binary may link (the bundled binary needs only glibc).
LIB_PACKAGES = {
    "libz.so.1": "Ubuntu/Debian: zlib1g; RHEL/Rocky/Alma: zlib; Fedora 40+: zlib-ng-compat; conda-forge: libzlib",
    "liblzma.so.5": "Ubuntu/Debian: liblzma5; RHEL/Rocky/Alma/Fedora: xz-libs; conda-forge: liblzma",
    "libbz2.so.1.0": "Ubuntu/Debian: libbz2-1.0; conda-forge: bzip2",
    "libbz2.so.1": "RHEL/Rocky/Alma/Fedora: bzip2-libs",
}
_ELF_MACHINES = {0x3E: "x86_64", 0xB7: "aarch64", 0x28: "arm", 0x03: "i386", 0xF3: "riscv64", 0x15: "ppc64le"}
# The binary that ships in the package (the same path as deeprm.inference.inference_preprocess.DEFAULT_BIN).
BUNDLED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inference", "deeprm_preprocess")


@dataclass
class Status:
    """Whether the preprocessing binary starts here (see status), and if not, why.

    ok: `--version` ran and exited 0. version: the binary's version line. reason, hint: why not, and what to do
    (no hint when nothing on this system can make the binary run, for example on another machine type).
    kind: the reason as one word (missing, unreadable, incomplete, os, machine, loader, noexec, mode, permission,
    library, glibc, cpu, timeout, exec, other), for callers that word their message by the reason."""

    ok: bool
    version: str | None = None
    reason: str | None = None
    hint: str | None = None
    kind: str | None = None


def _elf(path: str) -> Tuple[str | None, str | None]:
    """(machine, program loader) of a 64-bit little-endian ELF file; (None, None) for any other file.

    Raises ValueError when the file is cut off inside its program headers.
    """
    with open(path, "rb") as f:
        head = f.read(64)
        if len(head) < 64 or head[:4] != b"\x7fELF" or head[4] != 2 or head[5] != 1:
            return None, None
        machine = _ELF_MACHINES.get(struct.unpack_from("<H", head, 18)[0], "unknown")
        (phoff,) = struct.unpack_from("<Q", head, 32)
        phentsize, phnum = struct.unpack_from("<HH", head, 54)
        loader = None
        for i in range(phnum):
            f.seek(phoff + i * phentsize)
            ph = f.read(max(phentsize, 40))
            if len(ph) < 40:
                raise ValueError("the file ends inside its program headers")
            # Elf64_Phdr: p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz, ...
            p_type, _, p_offset, _, _, p_filesz = struct.unpack_from("<IIQQQQ", ph)
            if p_type == 3:  # PT_INTERP: the program loader's path
                f.seek(p_offset)
                loader = f.read(p_filesz).rstrip(b"\0").decode(errors="replace")
        return machine, loader


def named_path(value: str) -> str:
    """The binary that a value of --preprocess-bin or DEEPRM_PREPROCESS_BIN names: a path is made absolute (it is run
    from a subprocess, which would otherwise look a name without a slash up on PATH); a bare name found on PATH is
    that program, otherwise the file of that name in the current folder."""
    if os.sep not in value:
        found = shutil.which(value)
        if found:
            return os.path.abspath(found)
    return os.path.abspath(value)


def on_noexec_mount(path: str) -> bool:
    """True when the file system that holds `path` does not allow running programs (mounted noexec)."""
    try:
        return bool(os.statvfs(path).f_flag & os.ST_NOEXEC)
    except (OSError, AttributeError):
        return False


def explain_failure(path: str, returncode: int, output: str) -> Status:
    """A not-ok Status from the exit code and the messages of a failed `path --version`."""
    m = re.search(r"error while loading shared libraries: ([^:]+): (.+)", output)
    if m:
        lib, why = os.path.basename(m.group(1).strip()), m.group(2).strip()
        hint = f"Install the package that provides {lib}"
        hint += f" ({LIB_PACKAGES[lib]})." if lib in LIB_PACKAGES else "."
        if os.environ.get("LD_LIBRARY_PATH"):
            hint += " LD_LIBRARY_PATH is set: a broken or wrong copy of the library in its folders gives this too."
        return Status(False, reason=f"the system library {lib} cannot be loaded ({why})", hint=hint, kind="library")
    m = re.search(r"version `(GLIBC_([0-9.]+))' not found", output)
    if m:
        here = platform.libc_ver()[1]
        older = bool(here) and _version(here) < _version(m.group(2))
        if older and not os.environ.get("LD_LIBRARY_PATH"):
            return Status(
                False,
                reason=f"this system's glibc ({here}) is older than the binary needs ({m.group(1)})",
                hint="Build the binary on this system (see the advanced installation guide).",
                kind="glibc",
            )
        hint = "Build the binary on this system (see the advanced installation guide)."
        if os.environ.get("LD_LIBRARY_PATH"):
            hint = "LD_LIBRARY_PATH is set: an older copy of a glibc library in its folders gives this. " + hint
        return Status(
            False, reason=f"a library the binary loads lacks {m.group(1)} (this system's glibc: {here or 'unknown'})",
            hint=hint, kind="glibc",
        )
    if returncode == -4:  # SIGILL
        return Status(
            False,
            reason="the binary stopped with 'Illegal instruction': this CPU lacks an instruction the binary uses",
            kind="cpu",
        )
    tail = " | ".join(output.strip().splitlines()[-3:])
    reason = f"`{path} --version` exited with code {returncode}" + (f": {tail}" if tail else "")
    return Status(False, reason=reason, kind="other")


def _version(text: str) -> Tuple[int, ...]:
    """'2.28' -> (2, 28); the digits of each dot-separated part."""
    return tuple(int("".join(c for c in part if c.isdigit()) or 0) for part in text.split("."))


def status(path: str, timeout: float = 60) -> Status:
    """Whether the binary at `path` starts on this system: runs `path --version` after cheap checks of the file.

    Never raises: anything unexpected becomes a not-ok Status, so a check of the binary can never stop a run.
    """
    try:
        return _status(os.path.abspath(path), timeout)
    except Exception as exc:  # noqa: BLE001 -- see the docstring
        return Status(False, reason=f"the binary could not be checked ({type(exc).__name__}: {exc})", kind="other")


def _status(path: str, timeout: float) -> Status:
    if not os.path.isfile(path):
        return Status(
            False,
            reason=f"there is no file at {path}",
            hint="Reinstall DeepRM, or check the path given with --preprocess-bin or DEEPRM_PREPROCESS_BIN.",
            kind="missing",
        )
    try:
        machine, loader = _elf(path)
    except OSError as exc:
        if exc.errno != errno.EACCES:
            return Status(False, reason=f"{path} cannot be read ({exc.strerror})", kind="unreadable")
        machine, loader = None, None  # execute-only (for example mode 0711): the kernel can still run it
    except (ValueError, struct.error) as exc:
        return Status(
            False,
            reason=f"{path} is not a complete program ({exc})",
            hint="Reinstall DeepRM, or copy the file again.",
            kind="incomplete",
        )
    if machine is not None:  # a file that is not ELF (for example a wrapper script) is simply run
        if platform.system() != "Linux":
            reason = f"the binary is a Linux program and this system is {platform.system()}"
            return Status(False, reason=reason, kind="os")
        if machine != platform.machine():
            reason = f"the binary is for {machine} and this machine is {platform.machine()}"
            return Status(False, reason=reason, kind="machine")
        if loader and not os.path.exists(loader):
            return Status(
                False,
                reason=f"this system has no {loader}, the glibc program loader (as on Alpine and other musl systems)",
                hint="Use a glibc-based system or container image (for example Ubuntu, Debian, Rocky or Alma Linux).",
                kind="loader",
            )
    if on_noexec_mount(path):
        return Status(
            False,
            reason=f"the folder {os.path.dirname(path)} is on a file system where programs cannot run (noexec)",
            hint="Copy deeprm_preprocess to a folder where programs can run; set DEEPRM_PREPROCESS_BIN to the copy.",
            kind="noexec",
        )
    if not os.stat(path).st_mode & 0o111:
        return Status(
            False, reason=f"{path} has no execute permission", hint=f"Run `chmod +x {path}`.", kind="mode"
        )
    try:
        done = subprocess.run([path, "--version"], capture_output=True, text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return Status(False, reason=f"`{path} --version` did not finish in {timeout:g} s", kind="timeout")
    except OSError as exc:
        if exc.errno == errno.ENOEXEC:
            reason = f"{path} is not a program this system can run (exec format error)"
            return Status(False, reason=reason, kind="exec")
        if exc.errno == errno.EACCES:
            return Status(
                False,
                reason=f"running {path} is not permitted ({exc.strerror})",
                hint=f"Check the file's permissions and the mount options of its folder (findmnt -T {path}).",
                kind="permission",
            )
        if exc.errno == errno.ENOENT:
            # The file is there (checked above), so what is missing is the program that runs it.
            return _missing_runner(path)
        return Status(False, reason=f"{path} cannot be run ({exc.strerror})", kind="exec")
    if done.returncode == 0:
        lines = done.stdout.strip().splitlines()
        # the first line is the start-time stamp; the version line names the program
        version = next((line for line in lines if "Preprocessing v" in line), lines[-1] if lines else "")
        return Status(True, version=version)
    return explain_failure(path, done.returncode, (done.stderr or "") + (done.stdout or ""))


def _missing_runner(path: str) -> Status:
    """Why a file that exists cannot be run with "No such file or directory": its script interpreter (the #! line)
    or its program loader is missing."""
    try:
        with open(path, "rb") as f:
            first = f.readline(256)
    except OSError:
        first = b""
    if first.startswith(b"#!"):
        if first.endswith(b"\r\n"):
            return Status(
                False,
                reason=f"{path} is a script with Windows (CRLF) line endings, so its #! line names a program "
                "that does not exist",
                hint=f"Convert it to Unix line endings, for example with `dos2unix {path}`.",
                kind="loader",
            )
        words = first[2:].split()
        interpreter = words[0].decode(errors="replace") if words else ""
        if interpreter and not os.path.exists(interpreter):
            return Status(
                False,
                reason=f"{path} is a script whose interpreter {interpreter} is not on this system",
                hint="Install that interpreter, or use a compiled binary.",
                kind="loader",
            )
        return Status(False, reason=f"{path} cannot be run (No such file or directory)", kind="exec")
    return Status(
        False,
        reason=f"the program loader of {path} is not on this system (for example a 32-bit program without "
        "the 32-bit C library)",
        hint="Use a binary built for this system (see the advanced installation guide).",
        kind="loader",
    )


def libraries(path: str) -> List[Tuple[str, str]]:
    """Every shared library the program loader would load for `path`, as ldd lists it, with 'not found' entries.

    Uses LD_TRACE_LOADED_OBJECTS: for a dynamically linked program the loader lists the libraries and does not run
    the program.
    """
    env = dict(os.environ, LD_TRACE_LOADED_OBJECTS="1")
    try:
        done = subprocess.run(
            [os.path.abspath(path)], capture_output=True, text=True, errors="replace", timeout=30, env=env
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [("(could not list)", str(exc))]
    found = []
    for line in done.stdout.splitlines():
        m = re.match(r"\s*(\S+)\s+=>\s+(not found|\S+)", line)
        if m:
            found.append((m.group(1), m.group(2)))
    if done.returncode != 0:
        found.append(("(loader error)", (done.stderr.strip().splitlines() or [""])[-1]))
    return found
