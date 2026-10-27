/**
 * Copyright (C) 2026 Genome4me Incorporated - All Rights Reserved.
 * This software, including its source code, embedded concepts, and associated
 * documentation, is proprietary to Genome4me Incorporated and is protected
 * under trade secret and copyright law. Unauthorized use, copying, modification,
 * distribution, or disclosure to third parties, in whole or in part, is
 * strictly prohibited unless prior written permission is granted by Genome4me
 * Incorporated. Any such unauthorized actions constitute an infringement of
 * the intellectual property rights of Genome4me Incorporated. For licensing
 * inquiries or permissions, please contact Genome4me Incorporated.
 */

#include "pod5/Pod5Catalog.h"

#include <glob.h>

#include <algorithm>
#include <functional>
#include <utility>

#include "utils/Utils.h"

namespace deeprm {
  vector<string> Pod5Catalog::list_files(const string& directory)
  {
    vector<string> files;
    string pattern = directory + "/*.pod5";
    glob_t glob_result;
    if (glob(pattern.c_str(), GLOB_TILDE, nullptr, &glob_result) == 0) {
      for (size_t i = 0; i < glob_result.gl_pathc; ++i)
        files.emplace_back(glob_result.gl_pathv[i]);
    }
    globfree(&glob_result);
    sort(files.begin(), files.end());
    return files;
  }

  Pod5Catalog::Pod5Catalog(const string& directory) : pod5_files(list_files(directory))
  {
  }

  void Pod5Catalog::scan_worker(int worker_id, const vector<string>& files, size_t first_index,
                                vector<vector<Pod5RecordMeta>>& out, atomic<size_t>& failed)
  {
    log_info() << "Starting POD5 metadata worker " << worker_id << endl;
    out.reserve(files.size());

    for (size_t i = 0; i < files.size(); ++i) {
      log_info() << "POD5 metadata worker " << worker_id << " processing " << i + 1
          << "/" << files.size() << " files" << endl;
      Pod5Reader reader(files[i]);
      vector<Pod5RecordMeta> file_records;
      if (!reader.parse_single_pod5_meta(files[i], file_records)) {
        log_err() << "Corrupted POD5 file: " << files[i] << endl;
        failed.fetch_add(1);
        continue;
      }
      for (auto& record : file_records)
        record.file_index = static_cast<uint32_t>(first_index + i);
      // Each file stays a separate vector so file-level structure survives.
      out.push_back(move(file_records));
    }
  }

  void Pod5Catalog::scan_async(int max_threads)
  {
    num_readers = min(static_cast<int>(pod5_files.size()), max_threads);
    scanned.resize(num_readers);
    log_info() << "Starting POD5 metadata reading with " << num_readers
        << " reader threads" << endl;

    vector<size_t> counts = Utils::split_sizes(pod5_files.size(), num_readers);
    size_t idx = 0;
    for (int i = 0; i < num_readers; ++i) {
      size_t count = counts[i];
      vector<string> slice(pod5_files.begin() + idx, pod5_files.begin() + idx + count);
      scan_threads.emplace_back(scan_worker, i, move(slice), idx, ref(scanned[i]), ref(failed_files));
      idx += count;
    }
  }

  void Pod5Catalog::join()
  {
    for (auto& t : scan_threads)
      t.join();
    scan_threads.clear();
    total_records = 0;
    for (const auto& reader_files : scanned)
      for (const auto& file_records : reader_files)
        total_records += file_records.size();
    log_info() << "POD5 metadata reading completed. Total records: " << total_records << endl;
  }
}  // namespace deeprm
