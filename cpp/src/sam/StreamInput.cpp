/***************************************************************************************************
 *
 * Copyright (C) 2025-2026 Genome4me Incorporated - All Rights Reserved.
 *
 * This software, including its source code, embedded concepts, and associated
 * documentation, is proprietary to Genome4me Incorporated and is protected
 * under trade secret and copyright law. Unauthorized use, copying, modification,
 * distribution, or disclosure to third parties, in whole or in part, is
 * strictly prohibited unless prior written permission is granted by Genome4me
 * Incorporated. Any such unauthorized actions constitute an infringement of
 * the intellectual property rights of Genome4me Incorporated. For licensing
 * inquiries or permissions, please contact Genome4me Incorporated.
 *
 **************************************************************************************************/

#include "sam/StreamInput.h"

#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>

// htslib's interface for an hFILE backend (struct hFILE_backend, hfile_init); it is not installed with htslib, so it
// comes from the htslib source folder (-I htslib).
#include "hfile_internal.h"

namespace deeprm {
  namespace {
    // htslib's fd-backed hFILE, plus the last byte read. That byte is written by whichever thread htslib reads the
    // input in, and read only after the reader has reached the end, when htslib has handed the end over to the
    // caller through its own locks.
    struct StreamInput {
      hFILE base;
      int fd;
      bool shared;    // standard input: left open on close, as htslib's hopen("-") leaves it
      int last_byte;  // -1 until the first byte is read
    };

    // read(), write(), seek() and close() as in htslib's fd backend (hfile.c), so the input reads as with hdopen().
    ssize_t input_read(hFILE* fpv, void* buffer, size_t nbytes)
    {
      StreamInput* fp = reinterpret_cast<StreamInput*>(fpv);
      ssize_t n;
      do {
        n = read(fp->fd, buffer, nbytes);
      } while (n < 0 && errno == EINTR);
      if (n > 0)
        fp->last_byte = static_cast<unsigned char*>(buffer)[n - 1];
      return n;
    }

    ssize_t input_write(hFILE*, const void*, size_t)
    {
      errno = EBADF;  // opened for reading only
      return -1;
    }

    off_t input_seek(hFILE* fpv, off_t offset, int whence)
    {
      return lseek(reinterpret_cast<StreamInput*>(fpv)->fd, offset, whence);
    }

    int input_flush(hFILE*)
    {
      return 0;
    }

    int input_close(hFILE* fpv)
    {
      StreamInput* fp = reinterpret_cast<StreamInput*>(fpv);
      if (fp->shared)
        return 0;
      int ret;
      do {
        ret = close(fp->fd);
      } while (ret < 0 && errno == EINTR);
      return ret;
    }

    const hFILE_backend input_backend = {input_read, input_write, input_seek, input_flush, input_close};
  }

  hFILE* open_stream_input(int fd, bool shared)
  {
    // The buffer size htslib's hdopen() picks: 128 KiB for a pipe, the file's block size otherwise.
    struct stat st;
    size_t capacity = 0;
    if (fstat(fd, &st) == 0)
      capacity = S_ISFIFO(st.st_mode) ? 128 * 1024 : st.st_blksize;
    StreamInput* fp = reinterpret_cast<StreamInput*>(hfile_init(sizeof(StreamInput), "r", capacity));
    if (!fp)
      return nullptr;
    fp->fd = fd;
    fp->shared = shared;
    fp->last_byte = -1;
    fp->base.backend = &input_backend;
    return &fp->base;
  }

  bool sam_text_cut_off(const htsFile* file, hFILE* input)
  {
    return file->format.format == htsExactFormat::sam && file->format.compression == htsCompression::no_compression
        && reinterpret_cast<const StreamInput*>(input)->last_byte != '\n';
  }
}
