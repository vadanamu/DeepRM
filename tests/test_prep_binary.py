"""Tests for deeprm.utils.prep_binary: why the C++ preprocessing binary cannot start, and what it needs to run."""

import os
import platform
import shutil
import subprocess

import pytest

from deeprm.utils import prep_binary as pb

GLIBC_ONLY = {"libc.so.6", "libm.so.6", "libdl.so.2", "libpthread.so.0", "ld-linux-x86-64.so.2"}


def _bundled():
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        pytest.skip(f"the bundled binary is for Linux x86_64; this is {platform.system()} {platform.machine()}")
    if not os.path.isfile(pb.BUNDLED):
        pytest.skip(f"no bundled binary at {pb.BUNDLED}")
    return pb.BUNDLED


def test_missing_file(tmp_path):
    st = pb.status(str(tmp_path / "nothing"))
    assert not st.ok and "there is no file" in st.reason


def test_file_without_execute_permission(tmp_path):
    f = tmp_path / "prep"
    f.write_text("#!/bin/sh\nexit 0\n")
    f.chmod(0o644)
    st = pb.status(str(f))
    if pb.on_noexec_mount(str(tmp_path)):  # there the mount is the first reason: chmod would not help
        assert not st.ok and "noexec" in st.reason
    else:
        assert not st.ok and "no execute permission" in st.reason and "chmod +x" in st.hint


def test_truncated_program(tmp_path):
    with open(_bundled(), "rb") as f:
        head = f.read(64)  # the ELF header only: the program headers it points to are missing
    cut = tmp_path / "prep"
    cut.write_bytes(head)
    cut.chmod(0o755)
    st = pb.status(str(cut))
    assert not st.ok and "not a complete program" in st.reason


def test_program_for_another_machine(tmp_path):
    with open(_bundled(), "rb") as f:
        data = bytearray(f.read(1 << 16))
    data[18:20] = (0xB7).to_bytes(2, "little")  # e_machine = aarch64 (_bundled() makes sure this host is x86_64)
    other = tmp_path / "prep"
    other.write_bytes(bytes(data))
    other.chmod(0o755)
    st = pb.status(str(other))
    assert not st.ok and "is for aarch64" in st.reason and st.kind == "machine" and st.hint is None


def test_noexec_folder(tmp_path):
    if not pb.on_noexec_mount(str(tmp_path)):
        pytest.skip(f"{tmp_path} allows programs to run, so the noexec case cannot be made here")
    copy = tmp_path / "prep"
    shutil.copy(_bundled(), copy)
    copy.chmod(0o755)
    st = pb.status(str(copy))
    assert not st.ok and "noexec" in st.reason and "DEEPRM_PREPROCESS_BIN" in st.hint


@pytest.mark.parametrize("output, code, words", [
    ("prep: error while loading shared libraries: libz.so.1: cannot open shared object file: No such file or "
     "directory", 127, ["libz.so.1", "cannot open shared object file", "zlib1g"]),
    ("prep: /lib64/libc.so.6: version `GLIBC_9.99' not found (required by prep)", 1, ["is older than", "GLIBC_9.99"]),
    ("", -4, ["Illegal instruction"]),
    ("something else\nlast line", 3, ["exited with code 3", "last line"]),
])
def test_explain_failure(output, code, words, monkeypatch):
    # LD_LIBRARY_PATH changes the glibc wording; the case without it is the one checked here
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(pb.platform, "libc_ver", lambda: ("glibc", "2.28"))
    st = pb.explain_failure("prep", code, output)
    text = f"{st.reason} {st.hint or ''}"
    if "GLIBC_9.99" in output:
        assert "is older than" in st.reason, st.reason
    assert not st.ok and all(w in text for w in words), text


def test_bundled_binary_starts():
    st = pb.status(_bundled())
    assert st.ok, st.reason
    assert "Preprocessing v" in st.version


def test_bundled_binary_needs_only_glibc():
    """The bundled binary links zlib, bzip2 and liblzma statically and has its own UUID parser."""
    if shutil.which("readelf") is None:
        pytest.skip("readelf is not installed, so the binary's library list cannot be read")
    out = subprocess.run(["readelf", "-d", _bundled()], capture_output=True, text=True, check=True).stdout
    needed = {line.split("[", 1)[1].rstrip("]") for line in out.splitlines() if "(NEEDED)" in line}
    assert needed and needed <= GLIBC_ONLY, f"needs more than glibc: {sorted(needed - GLIBC_ONLY)}"
    assert "(RPATH)" not in out and "(RUNPATH)" not in out


def test_bundled_binary_ignores_broken_copies_of_libraries(tmp_path, monkeypatch):
    """A broken libz, liblzma or libuuid first on LD_LIBRARY_PATH does not stop it (it does not load them)."""
    binary = _bundled()
    for lib in ("libz.so.1", "liblzma.so.5", "libuuid.so.1", "libbz2.so.1.0", "libbz2.so.1"):
        (tmp_path / lib).write_bytes(b"")
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path))
    st = pb.status(binary)
    assert st.ok, st.reason


def test_glibc_message_with_a_newer_system_glibc_points_at_library_copies(monkeypatch):
    monkeypatch.setattr(pb.platform, "libc_ver", lambda: ("glibc", "2.39"))
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    st = pb.explain_failure("prep", 1, "prep: /x/libc.so.6: version `GLIBC_2.38' not found (required by prep)")
    assert "older than" not in st.reason and "lacks GLIBC_2.38" in st.reason
    monkeypatch.setenv("LD_LIBRARY_PATH", "/somewhere")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/x")
    st = pb.explain_failure("prep", 1, "prep: /x/libc.so.6: version `GLIBC_2.38' not found (required by prep)")
    assert "LD_LIBRARY_PATH" in st.hint


def test_bare_name_is_looked_up_on_path(tmp_path, monkeypatch):
    if pb.on_noexec_mount(str(tmp_path)):
        pytest.skip(f"{tmp_path} does not allow programs to run, so nothing in it is found on PATH")
    tool = tmp_path / "bin" / "deeprm_preprocess"
    tool.parent.mkdir()
    tool.write_text("#!/bin/sh\nexit 0\n")
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tool.parent}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.chdir(tmp_path)
    assert pb.named_path("deeprm_preprocess") == str(tool)
    assert pb.named_path("not_on_path_xyz") == str(tmp_path / "not_on_path_xyz")
    assert pb.named_path("bin/deeprm_preprocess") == str(tool)


def test_execute_only_binary_still_starts(tmp_path):
    if pb.on_noexec_mount(str(tmp_path)):
        pytest.skip(f"{tmp_path} does not allow programs to run")
    copy = tmp_path / "prep"
    shutil.copy(_bundled(), copy)
    copy.chmod(0o111)  # the kernel runs an ELF file it may not read
    try:
        st = pb.status(str(copy))
    finally:
        copy.chmod(0o755)
    assert st.ok, st.reason


def test_script_with_a_missing_interpreter(tmp_path):
    if pb.on_noexec_mount(str(tmp_path)):
        pytest.skip(f"{tmp_path} does not allow programs to run")
    script = tmp_path / "prep"
    script.write_text("#!/no/such/interpreter -x\nexit 0\n")
    script.chmod(0o755)
    st = pb.status(str(script))
    assert not st.ok and st.kind == "loader" and "/no/such/interpreter" in st.reason and st.hint


def test_script_with_windows_line_endings(tmp_path):
    if pb.on_noexec_mount(str(tmp_path)):
        pytest.skip(f"{tmp_path} does not allow programs to run")
    script = tmp_path / "prep"
    script.write_bytes(b"#!/bin/sh\r\nexit 0\r\n")
    script.chmod(0o755)
    st = pb.status(str(script))
    assert not st.ok and "CRLF" in st.reason and "dos2unix" in st.hint


def test_missing_execute_bit_hint_is_a_whole_sentence(tmp_path):
    copy = tmp_path / "prep"
    shutil.copy(_bundled(), copy)
    copy.chmod(0o644)
    st = pb.status(str(copy))
    if st.kind == "noexec":
        pytest.skip(f"{tmp_path} does not allow programs to run")
    assert st.kind == "mode" and st.hint == f"Run `chmod +x {copy}`."
