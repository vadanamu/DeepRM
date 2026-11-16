# Building deeprm_pileup

`deeprm_pileup` is a C++ program that gives the outputs of `deeprm call pileup` (DeepRM v1.2.1),
byte for byte. License: CC BY-NC-SA 4.0 (`LICENSE`); the notices of the work it contains or follows are
in `THIRD_PARTY_NOTICES`.

## Layout

| Path | Contents |
|---|---|
| `pileup/` | the program (C++23) |
| `svml/` | two assembly routines of numpy/SVML (BSD-3-Clause, `svml/LICENSE`) |
| `libdeflate/` | libdeflate v1.23 (MIT) |
| `htslib/` | htslib 1.22 with htscodecs v1.6.3 (MIT and BSD) |
| `tests/unit/` | unit tests |
| `tests/*.py` | scripts that compare the arithmetic with NumPy (not needed for a build) |
| `Makefile` | the build |

## Requirements

- x86-64 Linux with glibc 2.28 or later. The build and all checks were done on RHEL/Rocky 8.
- GCC 14 or later (the code uses C++23 library features such as `<print>`). On RHEL/Rocky 8.10 or 9:
  `dnf install gcc-toolset-14`. The Makefile uses `/opt/rh/gcc-toolset-14` when it is there; elsewhere
  pass the compilers: `make CC=/path/to/gcc CXX=/path/to/g++`.
- `make`, `cmake` 3.7 or later, `autoconf`, `automake`, and the zlib headers (`zlib-devel`).
- To run: a CPU with AVX2 and FMA. AVX512 is used when the CPU has it and is not needed.

## The two libraries

`libdeflate/` and `htslib/` must hold the sources of the two libraries. A source archive of
`deeprm_pileup` has them. When the folders are empty, get the releases from GitHub:

```bash
rmdir libdeflate htslib 2>/dev/null
git clone --depth 1 --branch v1.23 https://github.com/ebiggers/libdeflate.git libdeflate
git clone --depth 1 --branch 1.22 --recurse-submodules --shallow-submodules \
  https://github.com/samtools/htslib.git htslib
```

Both are the upstream releases, unchanged. The Makefile builds them as static libraries; htslib is
configured with libdeflate and without bzip2, lzma, libcurl and plugins (BAM needs none of them).

## Build

```bash
make -j"$(nproc)"         # the libraries, then bin/deeprm_pileup
make -j"$(nproc)" test    # bin/unit_tests; ends with "43 tests, 0 failed"
bin/deeprm_pileup -h
```

- The program needs only the system's zlib, libstdc++ and C library at run time.
- `bin/deeprm_pileup -vv` prints the build time and the git commit; outside a git repository the
  commit reads `unknown`.

## Checking a build

The outputs must be byte identical to those of DeepRM v1.2.1 run with the same `-t`:

```bash
bin/deeprm_pileup -t 45 -i <molecule-level folder> -b <BAM> [-a <annotation>] -o <output folder>
python compare_outputs.py <v1.2.1 output folder> <output folder>    # last line: RESULT IDENTICAL
```

Keep the compiler options of the Makefile, `-ffp-contract=off` above all: the floating-point results
depend on them. A different C library can change the results of `log` and `log1p`; the program checks
this when it starts and prints a warning if they differ.

## Placing it in the DeepRM repository

The preprocessing program of DeepRM (`cpp/` of https://github.com/vadanamu/DeepRM) is built
differently; when the two are put side by side, mind these points:

| | preprocessing (`cpp/`) | deeprm_pileup |
|---|---|---|
| C++ standard, compiler | C++20, the system's `g++` | C++23, GCC 14 or later |
| libdeflate | its own copy | v1.23 |
| htslib 1.22 | built with bzip2 and lzma | built without them (see above) |
| include paths | relative to `src/` | relative to the top folder (`"pileup/x.h"`) |

- The simplest placement is a folder of its own (`cpp/pileup/`, say) holding `pileup/`, `svml/`,
  `tests/` and this Makefile, so that the include paths stay as they are.
- `deeprm_pileup` can use the libraries of `cpp/` in place of its own (`LIBA` and `INCLUDES` in the
  Makefile; an htslib built with bzip2 and lzma also needs `-lbz2 -llzma` in `LIBS`). The site tables
  and the BAM records do not depend on the library versions; the compressed bytes of the npz and BAM
  files do, and `compare_outputs.py` does not compare those. Rerun the check above after such a change.

## The binary shipped with DeepRM

DeepRM ships `deeprm_pileup` in its wheel (`src/deeprm/inference/deeprm_pileup`) and runs it from
`deeprm call pileup` (see `cpp/PILEUP_INTERFACE.md`). That binary must run on any Linux x86_64 with glibc 2.28 or newer
and load nothing but the C library, as `deeprm_preprocess` does. It is built with conda-forge's compilers, whose C
library is glibc 2.28, with libstdc++, libgcc and zlib linked statically (no root needed):

```bash
conda create -y -p ./buildenv -c conda-forge --override-channels \
  gcc_linux-64=14 gxx_linux-64=14 sysroot_linux-64=2.28 zlib autoconf automake m4 make
pip install --target ./cmake cmake          # libdeflate is built with CMake
export PATH=$PWD/cmake/bin:$PWD/buildenv/bin:$PATH PYTHONPATH=$PWD/cmake CPATH=$PWD/buildenv/include LIBRARY_PATH=$PWD/buildenv/lib M4=$PWD/buildenv/bin/m4
# keep conda's library folder out of the binary's library search path
SPECS=$(ls buildenv/lib/gcc/x86_64-conda-linux-gnu/*/specs)
{ echo '*link_command:'; awk 'f && /^$/{exit} f{print} /^\*link_command:/{f=1}' $SPECS \
    | sed "s#%{!static:-rpath $PWD/buildenv/lib}##"; echo; } > norpath.specs
MK="CC=x86_64-conda-linux-gnu-gcc CXX=x86_64-conda-linux-gnu-g++ Z_LIB=-l:libz.a EXTRA_LDFLAGS=-specs=$PWD/norpath.specs"
make -j 16 $MK
make -j 16 test $MK                                   # 44 tests, 0 failed
x86_64-conda-linux-gnu-strip --strip-debug bin/deeprm_pileup  # the debug information names the build's folders
readelf -d bin/deeprm_pileup | grep NEEDED            # libc, libm, libpthread and the loader only
```

Outside a git repository, pass the commit to show in `--version`: `make GIT_HASH=<commit> ...`.
