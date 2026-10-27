CXX = g++
CC = gcc
# -ffile-prefix-map keeps the build folder's absolute path out of the binary (assert messages, debug information);
# the same map goes to every vendored library built below.
PREFIX_MAP = -ffile-prefix-map=$(CURDIR)=.
CXXFLAGS = -g -std=c++20 -O2 -Wall -Wextra -pthread -ftree-vectorize $(PREFIX_MAP) # -fopt-info-vec
# The C++ runtime is linked statically and unused shared libraries are dropped, so the binary needs from the
# system only the C library, libz, libbz2 and liblzma; with Z_LIB=-l:libz.a BZ2_LIB=-l:libbz2.a
# LZMA_LIB=-l:liblzma.a (below) it needs only the C library. Build on the oldest Linux that should run it: the
# C library version is the one of the build system (or of the compiler's sysroot).
# EXTRA_LDFLAGS=<flags> adds link flags for the binary, for example a toolchain's -specs file.
EXTRA_LDFLAGS ?=
LDFLAGS = -pthread -static-libstdc++ -static-libgcc -Wl,--as-needed $(EXTRA_LDFLAGS)

# Project directories
SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj

# External libraries
HTSLIB_DIR = htslib
CNPY_DIR = cnpy
POD5_DIR = pod5-file-format
MIMALLOC_DIR = mimalloc
LIBDEFLATE_DIR = libdeflate

# Include paths
INCLUDES = -I$(INC_DIR) \
           -I$(SRC_DIR) \
           -I$(HTSLIB_DIR) \
           -I$(CNPY_DIR) \
           -I$(POD5_DIR)/include \
           -I$(MIMALLOC_DIR)/include \
           -I$(LIBDEFLATE_DIR)

# Library paths and libraries. htslib is built without libcurl (no network URLs are read), so neither
# libcurl nor OpenSSL is linked.
# Z_LIB, BZ2_LIB and LZMA_LIB pick each library: shared by default, static with -l:libz.a, -l:libbz2.a and
# -l:liblzma.a (the .a files must be on the library path; RHEL's bzip2-devel and xz-devel have none). The binary
# shipped with DeepRM is linked with all three static, so it needs nothing but the C library. bzip2 matters
# most: its shared library is named libbz2.so.1.0 on Debian and Ubuntu and libbz2.so.1 on RHEL and Rocky, so a
# binary linked to one of them does not start on the other.
Z_LIB ?= -lz
BZ2_LIB ?= -lbz2
LZMA_LIB ?= -llzma
LIBS = $(Z_LIB) $(BZ2_LIB) $(LZMA_LIB) -ldl
LIBA = $(HTSLIB_DIR)/libhts.a \
       $(CNPY_DIR)/build/libcnpy.a \
       $(POD5_DIR)/libpod5_format.a \
       $(POD5_DIR)/libarrow.a \
       $(POD5_DIR)/libzstd.a \
       $(LIBDEFLATE_A)

# NPZ compression (v1.20, cmake defaults).
LIBDEFLATE_A = $(LIBDEFLATE_DIR)/build/libdeflate.a

# Process-wide allocator. mimalloc (v2.1.7, cmake defaults) replaces the jemalloc that ships with pod5:
# the object file goes FIRST on the link line so its malloc/free/operator new override every static
# library's references, which an archive placed later would not guarantee. pod5/Arrow reference no
# jemalloc-specific API, so nothing else changes.
MIMALLOC_OBJ = $(MIMALLOC_DIR)/out/release/mimalloc.o

# Target executable
TARGET = bin/deeprm_preprocess

# Source files
SOURCES = $(shell find $(SRC_DIR) -name "*.cpp")
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Version string of -v: build time and the git commit of this folder, with * when tracked files changed.
# Outside a git checkout both git calls print nothing.
BUILD_TIME:=$(shell date -Iseconds)
GIT_MODIFIED:=$(shell git diff --quiet HEAD -- . 2>/dev/null || echo "*")
GIT_HASH:=$(shell git rev-parse --short HEAD 2>/dev/null)$(if $(shell git rev-parse --short HEAD 2>/dev/null),$(GIT_MODIFIED),)

# Default target
all: $(TARGET)

mpfr: mpfr_build
mpfr_build: CXXFLAGS += -DUSE_MPFR
mpfr_build: LIBS += -lmpfr -lgmp
mpfr_build: all

# Build htslib first
# htslib uses libdeflate for BGZF inflate/deflate and crc32, so it is built after libdeflate and
# pointed at it with paths relative to the htslib folder. htslib records its build flags in the binary,
# so no absolute path goes into them (not even a prefix map: htslib compiles its files by relative path).
$(HTSLIB_DIR)/libhts.a: $(LIBDEFLATE_A)
	@echo "Building htslib..."
	autoreconf -i $(HTSLIB_DIR)
	cd $(HTSLIB_DIR) && ./configure CC=$(CC) CFLAGS="-g -O2" --disable-ref-cache --disable-s3 \
	    --disable-libcurl --with-libdeflate CPPFLAGS=-I../$(LIBDEFLATE_DIR) LDFLAGS=-L../$(LIBDEFLATE_DIR)/build \
	    && make -j 20

# cnpy asks for cmake 3.0, which cmake 4 refuses unless told the oldest policy version to apply.
$(CNPY_DIR)/build/libcnpy.a:
	@echo "Building cnpy..."
	mkdir -p $(CNPY_DIR)/build && cd $(CNPY_DIR)/build && cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
	    -DCMAKE_CXX_FLAGS="$(PREFIX_MAP)" .. && make && cd -

# Create the object directory tree, mirroring every directory that holds a
# source file, so adding a source directory needs no Makefile change.
$(OBJ_DIR):
	@mkdir -p $(sort $(dir $(OBJECTS)))

# Build target
$(TARGET): src/build_time.h $(HTSLIB_DIR)/libhts.a $(CNPY_DIR)/build/libcnpy.a $(MIMALLOC_OBJ) $(LIBDEFLATE_A) $(OBJ_DIR) $(OBJECTS)
	@echo "Linking $(TARGET)..."
	mkdir -p bin
	$(CXX) $(LDFLAGS) $(MIMALLOC_OBJ) $(OBJECTS) -o $(TARGET) $(LIBA) $(LIBS)
	strip --strip-debug $(TARGET)  # the shipped binary carries no debug information (or the paths it holds)

# Compile source files. -MMD -MP writes each object's header dependencies (obj/*.d, included below), so an
# edited header recompiles every object that uses it; without them a class whose layout changed is seen two ways.
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -MMD -MP $(INCLUDES) -c $< -o $@

-include $(OBJECTS:.o=.d)

# mimalloc is built with its own cmake defaults with the same compiler as this project so the result
# does not depend on whatever cc cmake would otherwise find first on PATH.
$(MIMALLOC_OBJ):
	mkdir -p $(MIMALLOC_DIR)/out/release
	cd $(MIMALLOC_DIR)/out/release && cmake -DCMAKE_C_COMPILER=$(CC) -DCMAKE_C_FLAGS="$(PREFIX_MAP)" ../.. && $(MAKE)

$(LIBDEFLATE_A):
	mkdir -p $(LIBDEFLATE_DIR)/build
	cd $(LIBDEFLATE_DIR)/build && cmake -DCMAKE_C_COMPILER=$(CC) -DCMAKE_C_FLAGS="$(PREFIX_MAP)" .. && $(MAKE)

# Clean compiled objects and target (not external libraries)
clean:
	@echo "Cleaning project files..."
	rm -rf $(OBJ_DIR)
	rm -f $(TARGET)

# Clean all including external libraries
clean_all: clean
	@echo "Cleaning all including external libraries..."
	cd $(HTSLIB_DIR) && make clean
	rm -rf $(MIMALLOC_DIR)/out/release
	rm -rf $(LIBDEFLATE_DIR)/build
	rm -rf $(CNPY_DIR)/build

# Install dependencies (if needed)
install-deps:
	@echo "Please ensure the following are installed:"
	@echo "- cnpy library in ./cnpy/"
	@echo "- pod5-file-format library in ./pod5-file-format/"

# The UUID parser against libuuid's own answers (tests/uuid_vectors.tsv, made by tests/gen_uuid_vectors.py).
test-uuid:
	mkdir -p $(OBJ_DIR)
	$(CXX) -std=c++20 -O2 -Wall -Wextra -I$(SRC_DIR) tests/uuid_parse_test.cpp -o $(OBJ_DIR)/uuid_parse_test
	$(OBJ_DIR)/uuid_parse_test tests/uuid_vectors.tsv

# The stream reader's input and its check of a SAM text's last newline (src/sam/StreamInput.cpp), on files and pipes;
# links the htslib built for the binary.
test-sam-end: $(HTSLIB_DIR)/libhts.a $(LIBDEFLATE_A)
	mkdir -p $(OBJ_DIR)
	$(CXX) -std=c++20 -O2 -Wall -Wextra -pthread $(INCLUDES) tests/sam_text_end_test.cpp src/sam/StreamInput.cpp \
	    -o $(OBJ_DIR)/sam_text_end_test $(HTSLIB_DIR)/libhts.a $(LIBDEFLATE_A) $(LIBS)
	$(OBJ_DIR)/sam_text_end_test $(OBJ_DIR)

# Test build (compile only, don't link)
test-compile: $(OBJ_DIR) $(OBJECTS)
	@echo "Compilation test completed successfully"

# build_time.h - conditionally update to avoid unnecessary recompilation
src/build_time.h: .FORCE
	@echo "// this file generated by make"                    >| $@.tmp
	@echo "#ifndef DEEPRM_SRC_BUILD_TIME_H_"                   >> $@.tmp
	@echo "#define DEEPRM_SRC_BUILD_TIME_H_"                   >> $@.tmp
	@echo "#define DEEPRM_BUILD_TIME \"$(BUILD_TIME)\""         >> $@.tmp
	@echo "#define DEEPRM_GIT_HASH   \"$(GIT_HASH)\""           >> $@.tmp
	@echo "#endif  // DEEPRM_SRC_BUILD_TIME_H_"                >> $@.tmp
	@cmp -s $@.tmp $@ || mv $@.tmp $@
	@rm -f $@.tmp

.FORCE:
.PHONY: all clean clean_all install-deps test-compile test-uuid test-sam-end .FORCE
