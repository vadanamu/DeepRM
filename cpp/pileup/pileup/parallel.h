/**
 * SPDX-License-Identifier: CC-BY-NC-SA-4.0
 *
 * Part of deeprm_pileup, which is adapted from the pileup step of DeepRM
 * v1.2.1 (Laboratory of Computational Biology, Seoul National University).
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
 * 4.0 International License; see the LICENSE file.
 */
#ifndef DEEPRM_PILEUP_PARALLEL_H_
#define DEEPRM_PILEUP_PARALLEL_H_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <expected>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace deeprm {

using Status = std::expected<void, std::string>;

// Runs function(worker, i) for every i < count on up to `threads` threads
// (worker: the index of the calling thread); each call returns a Status.
// Stops handing out work after the first error and returns it.
template <typename Function>
Status parallel_for(std::size_t threads, std::size_t count,
                    Function&& function) {
  std::atomic<std::size_t> next{0};
  std::atomic<bool> failed{false};
  std::mutex mutex;
  Status result;
  const auto work = [&](std::size_t worker) {
    while (!failed.load()) {
      const std::size_t i = next.fetch_add(1);
      if (i >= count) return;
      Status status = function(worker, i);
      if (!status) {
        const std::lock_guard<std::mutex> lock(mutex);
        if (!failed.exchange(true)) result = std::move(status);
      }
    }
  };
  {
    std::vector<std::jthread> pool;
    const std::size_t extra = std::min(std::max<std::size_t>(threads, 1), count);
    for (std::size_t t = 1; t < extra; ++t) pool.emplace_back(work, t);
    work(0);
  }
  return result;
}

}  // namespace deeprm

#endif  // DEEPRM_PILEUP_PARALLEL_H_
