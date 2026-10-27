/***************************************************************************************************
 *
 * Copyright (C) 2025 Genome4me Incorporated - All Rights Reserved.
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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// Global mutex for thread-safe logging
inline mutex& get_log_mutex()
{
  static mutex log_mtx;
  return log_mtx;
}

// Timestamp logging utility
inline string get_timestamp()
{
  auto now = chrono::system_clock::now();
  auto time_t_now = chrono::system_clock::to_time_t(now);
  struct tm tm_buf;
  if (!localtime_r(&time_t_now, &tm_buf)) {
    return "[unknown]";
  }

  ostringstream oss;
  oss << "[" << put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S");

  // Insert colon into timezone offset for ISO 8601 (+0900 -> +09:00)
  char tz_buf[8];
  if (strftime(tz_buf, sizeof(tz_buf), "%z", &tm_buf) == 0) {
    return "[unknown]";
  }
  string tz(tz_buf);
  if (tz.size() >= 5) {
    tz.insert(3, ":");
  }
  oss << tz << "]";
  return oss.str();
}

// Program start time for relative elapsed logging
inline chrono::steady_clock::time_point& get_start_time()
{
  static auto start = chrono::steady_clock::now();
  return start;
}

// Relative elapsed time string (HH:MM:SS.ss)
inline string get_elapsed()
{
  auto elapsed = chrono::steady_clock::now() - get_start_time();
  auto total_ms = chrono::duration_cast<chrono::milliseconds>(elapsed).count();
  int h = total_ms / 3600000;
  int m = (total_ms % 3600000) / 60000;
  double s = (total_ms % 60000) / 1000.0;
  ostringstream oss;
  oss << setfill('0') << setw(2) << h << ":"
      << setw(2) << m << ":"
      << fixed << setprecision(2) << setw(5) << s;
  return oss.str();
}

// Thread-safe logger that buffers output and writes atomically
class ThreadSafeLogger
{
private:
  ostringstream buffer_;
  ostream& out_;

public:
  explicit ThreadSafeLogger(ostream& os) : out_(os)
  {
    buffer_ << "[" << get_elapsed() << "] ";
  }

  ThreadSafeLogger(const ThreadSafeLogger&) = delete;
  ThreadSafeLogger& operator=(const ThreadSafeLogger&) = delete;
  ThreadSafeLogger(ThreadSafeLogger&&) = delete;
  ThreadSafeLogger& operator=(ThreadSafeLogger&&) = delete;

  ~ThreadSafeLogger()
  {
    lock_guard<mutex> lock(get_log_mutex());
    out_ << buffer_.str();
    out_.flush();
  }

  template<typename T>
  ThreadSafeLogger& operator<<(const T& val)
  {
    buffer_ << val;
    return *this;
  }

  // Handle stream manipulators like endl
  ThreadSafeLogger& operator<<(ostream& (*manip)(ostream&))
  {
    manip(buffer_);
    return *this;
  }
};

// Inline loggers prefixed with the elapsed time since program start (thread-safe)
inline ThreadSafeLogger log_info() { return ThreadSafeLogger(cout); }
inline ThreadSafeLogger log_err() { return ThreadSafeLogger(cerr); }

// Log absolute start time (call once at program start)
inline void log_start_time()
{
  get_start_time(); // initialize
  log_info() << "Program start: " << get_timestamp() << endl;
}

// Read process RSS from /proc/self/statm (Linux only)
inline size_t get_rss_mb()
{
  ifstream statm("/proc/self/statm");
  size_t size_pages = 0, rss_pages = 0;
  statm >> size_pages >> rss_pages;
  return rss_pages * 4096 / (1024 * 1024);
}

struct NormalizationFactors {
  double quantile_a = 0.2;
  double quantile_b = 0.8;
  double shift_mult = 0.48;
  double scale_mult = 0.59;
};

class Utils {
public:
  // Reads dropped because their move table is longer than their signal, over all threads of the run.
  static inline atomic<long> move_past_signal{0};

  static double mean_phred(const vector<uint8_t>& phred);
  static vector<float> move_to_dwell(const vector<bool>& move,
                                     double quantile_a, double quantile_b, double shift_mult,
                                     double scale_mult, int sampling = 6);
  static vector<vector<double>> normalize_trim_segment_signal(
    const vector<double>& signal, const vector<bool>& move,
    int sp, int ts, int ns, double quantile_a, double quantile_b,
    double shift_mult, double scale_mult, int sampling = 6);
  static vector<double> segmented_signal_to_block_range(
    const vector<vector<double>>& signal_segmented,
    int start_idx, int end_idx,
    const vector<uint16_t>& segment_len_arr,
    int kmer, int sampling, int sig_window, int pad_to);
  // numpy.quantile(method="linear") bit for bit: the virtual index and the
  // interpolation run in double and the result is rounded to T once, and
  // the upper neighbour is interpolated from when gamma >= 0.5, as numpy's
  // _lerp does. Both matter for parity with the Python reference.
  template <typename T>
  static T quantile(const vector<T>& data, double q);
  // Sizes of the `parts` contiguous slices np.array_split makes of `n`
  // items: the first n % parts slices get one extra item.
  static vector<size_t> split_sizes(size_t n, int parts);
};

// Template implementation
template <typename T>
T Utils::quantile(const vector<T>& data, double q)
{
  if (data.empty()) return 0.0;
  // Need to copy data since we will modify it
  vector<T> sorted_data(data);
  double index = q * (sorted_data.size() - 1);
  int lower = static_cast<int>(index);
  int upper = lower + 1;
  if (upper >= static_cast<int>(sorted_data.size())) {
    // Only need one element - use nth_element for O(n) performance
    nth_element(sorted_data.begin(), sorted_data.begin() + lower, sorted_data.end());
    return sorted_data[lower];
  }
  // Need two elements for interpolation
  // First, partition to find the lower element
  nth_element(sorted_data.begin(), sorted_data.begin() + lower, sorted_data.end());
  double lower_val = sorted_data[lower];

  // Find the upper element - it's the minimum element in the range [lower+1, end)
  double upper_val = *min_element(sorted_data.begin() + upper, sorted_data.end());
  double gamma = index - lower;
  double diff = upper_val - lower_val;

  return static_cast<T>(gamma >= 0.5 ? upper_val - diff * (1.0 - gamma)
                                     : lower_val + diff * gamma);
}

// float data (the dwell tokens): numpy 2 casts a Python-float q to the array's dtype, so for float32 input the
// virtual index (n - 1) * q, gamma and the interpolation are all computed in float32. Do the same, in float, so
// the quantile matches np.quantile(float32_array, q) bit for bit (the double version above can pick a different
// interpolation point when the virtual index is near an integer, and shifts every dwell token of the read).
template <>
inline float Utils::quantile<float>(const vector<float>& data, double q)
{
  if (data.empty()) return 0.0f;
  vector<float> sorted_data(data);
  const size_t n = sorted_data.size();
  const float q32 = static_cast<float>(q);
  const float vi = static_cast<float>(n - 1) * q32;
  if (vi >= static_cast<float>(n - 1)) {
    return *max_element(sorted_data.begin(), sorted_data.end());
  }
  const size_t lower = static_cast<size_t>(floor(vi));
  nth_element(sorted_data.begin(), sorted_data.begin() + lower, sorted_data.end());
  const float a = sorted_data[lower];
  const float b = *min_element(sorted_data.begin() + lower + 1, sorted_data.end());
  const float g = vi - static_cast<float>(lower);
  const float d = b - a;
  return g >= 0.5f ? b - d * (1.0f - g) : a + d * g;
}
