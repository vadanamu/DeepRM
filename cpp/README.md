# DeepRM Preprocessing (C++)

The C++ preprocessing program that `deeprm call prep` runs. This page tells how to build it.

## Requirements

- GCC 10 or newer (or Clang 12 or newer) with its static C++ library (`libstdc++.a`; on RHEL-based systems the
  `libstdc++-static` package)
- GNU binutils 2.36 or newer. Ubuntu 22.04 and Debian 12 have it; RHEL, Rocky and Alma 8 and 9 do not (use the
  conda toolchain below).
- cmake 3.18 or newer (`python -m pip install cmake` gives one without root)
- autoconf, automake and make
- zlib, bzip2 and liblzma, with their headers

htslib, cnpy, mimalloc and libdeflate are included in this folder and built with the program. The POD5 library is
included as prebuilt static libraries (`pod5-file-format/`), so it is not built.

## Build

### 1. Get the source

Use the source of the DeepRM version you run (`deeprm --version` shows it):

```bash
git clone https://github.com/vadanamu/deeprm
cd deeprm
git checkout <the release tag of that version>
cd cpp
```

### 2. Get the build tools

On Ubuntu 22.04 or newer, or Debian 12 or newer:

```bash
sudo apt-get install build-essential autoconf automake cmake zlib1g-dev libbz2-dev liblzma-dev
```

On any other Linux, also without root, use conda-forge's compilers. Their glibc is 2.28, so the binary runs on any
Linux with glibc 2.28 or newer:

```bash
conda create -y -p ./buildenv -c conda-forge --override-channels \
  gcc_linux-64=13 gxx_linux-64=13 sysroot_linux-64=2.28 zlib bzip2 xz liblzma-static autoconf automake m4 make
python -m pip install cmake    # if your cmake is older than 3.18
export PATH=$PWD/buildenv/bin:$PATH CPATH=$PWD/buildenv/include LIBRARY_PATH=$PWD/buildenv/lib M4=$PWD/buildenv/bin/m4
CONDA_CC="CC=x86_64-conda-linux-gnu-gcc CXX=x86_64-conda-linux-gnu-g++"
# keep conda's library folder out of the binary's library search path
SPECS=$(ls buildenv/lib/gcc/x86_64-conda-linux-gnu/*/specs)
{ echo '*link_command:'; awk 'f && /^$/{exit} f{print} /^\*link_command:/{f=1}' "$SPECS" \
    | sed "s#%{!static:-rpath $PWD/buildenv/lib}##"; echo; } > norpath.specs
```

### 3. Build

```bash
STATIC="Z_LIB=-l:libz.a BZ2_LIB=-l:libbz2.a LZMA_LIB=-l:liblzma.a"
make -j 16 $STATIC                                                     # system tools
make -j 16 $CONDA_CC $STATIC EXTRA_LDFLAGS=-specs=$PWD/norpath.specs   # conda toolchain
```

`$STATIC` links zlib, bzip2 and liblzma into the binary, so it needs only glibc where it runs. A plain `make -j 16`
also works, but then those libraries must be installed wherever the binary runs. `make clean_all` removes all build
output.

## Test

```bash
make test-uuid $CONDA_CC       # the read-id parser
make test-sam-end $CONDA_CC    # the check for a cut-off SAM input
```

With system tools, `$CONDA_CC` is empty.

## Where the binary lands

The build writes `bin/deeprm_preprocess`. `bin/deeprm_preprocess --version` checks that it runs. To use it with
DeepRM:

```bash
export DEEPRM_PREPROCESS_BIN=$PWD/bin/deeprm_preprocess    # or: deeprm call prep --preprocess-bin <path> ...
```
