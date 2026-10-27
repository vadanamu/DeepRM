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

#pragma once

#include <htslib/hfile.h>
#include <htslib/hts.h>

namespace deeprm {
  // The stream reader's input: reads the file descriptor as htslib's own hdopen() does and remembers the last byte
  // it read. SAM text has no end-of-file marker, and htslib reads a last line without its newline as a whole record,
  // so a SAM cut off inside a record (in its move table, for example) reads cleanly to its end; its last byte, which
  // is then not a newline, is the only sign. The hFILE owns fd and closes it, unless `shared` (standard input, which
  // htslib's hopen("-") also leaves open); on failure (nullptr) fd stays open.
  hFILE* open_stream_input(int fd, bool shared = false);

  // True when `file`, read to its end from `input` (an hFILE from open_stream_input), is SAM text that is not
  // compressed and does not end with a newline: it was cut off inside its last record. Every SAM writer ends each
  // record with a newline. A SAM cut exactly at a line end holds only whole records and is not detected.
  bool sam_text_cut_off(const htsFile* file, hFILE* input);
}
