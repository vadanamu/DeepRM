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

#include "sam/SamDispatcher.h"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <utility>

#include <htslib/bgzf.h>
#include <htslib/hfile.h>

#include "bam/BamRecordExtractor.h"
#include "sam/StreamInput.h"
#include "utils/Utils.h"
#include "worker/StreamWorker.h"

namespace deeprm {
  namespace {
    // Reads and throws away what is still written into a pipe, until it is destroyed. It reads from its own
    // non-blocking opening of the pipe, not from a dup() of the descriptor: a dup shares the blocking mode with
    // htslib's reader, so either a read here could block for good when htslib's reader takes the bytes first, or
    // htslib's reads would stop blocking. Without /proc it does nothing.
    class PipeDrain {
    public:
      explicit PipeDrain(int pipe_fd)
      {
        if (pipe_fd >= 0)
          fd = open(("/proc/self/fd/" + to_string(pipe_fd)).c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
        if (fd >= 0)
          reader = thread([this] { run(); });
      }

      ~PipeDrain()
      {
        done = true;
        if (reader.joinable())
          reader.join();
        if (fd >= 0)
          close(fd);
      }

    private:
      void run()
      {
        char buffer[64 * 1024];
        while (!done) {
          pollfd p = {fd, POLLIN, 0};
          const int ready = poll(&p, 1, 100);
          if (ready < 0 && errno != EINTR)
            return;
          if (ready <= 0)
            continue;
          const ssize_t n = read(fd, buffer, sizeof(buffer));
          if (n == 0 || (n < 0 && errno != EAGAIN && errno != EINTR))
            return;  // every writer has closed the pipe, or the pipe cannot be read
        }
      }

      int fd = -1;
      atomic<bool> done{false};
      thread reader;
    };
  }

  SamDispatcher::SamDispatcher(const Arguments& args)
    : bam_path(args.bam_path), bam_cmd(args.bam_cmd), producer_pid(-1),
      producer_error(false),
      bq_cutoff(args.qcut), base_of_interest(args.base_of_interest),
      filter_flag(args.filter_flag),
      bam_threads(args.bam_threads),
      max_queue_per_worker(static_cast<size_t>(args.max_queue)),
      max_queue_size(max_queue_per_worker * args.cpu_count),
      worker_inflight(0),
      is_running(false), workers_ready(false), reading_complete(false),
      bam_file(nullptr), header(nullptr), stream_workers(nullptr), pod5_index(nullptr)
  {
  }

  SamDispatcher::~SamDispatcher()
  {
    stop();
    if (header) {
      sam_hdr_destroy(header);
    }
    if (bam_file) {
      sam_close(bam_file);
    }
  }

  void SamDispatcher::start()
  {
    is_running = true;

    log_info() << "Starting SAM dispatcher for " << bam_path
        << " with " << bam_threads << " decompression threads"
        << ", max_queue=" << max_queue_size << endl;
    read_thread = thread(&SamDispatcher::read_loop, this);
  }

  void SamDispatcher::stop()
  {
    // Wait for reading to complete first
    if (read_thread.joinable()) {
      read_thread.join();
    }

    // Now stop the dispatcher's own loops and wait for dispatch completion.
    // The dispatcher may be parked in the in-flight wait; its predicate
    // checks !is_running, so wake that cv too. The workers are stopped
    // separately by the runner after this returns.
    is_running = false;
    queue_cv.notify_all();
    { lock_guard<mutex> g(inflight_mutex); }
    inflight_cv.notify_all();

    if (dispatch_thread.joinable()) {
      dispatch_thread.join();
    }

    log_info() << "SAM dispatcher fully stopped" << endl;
  }

  void SamDispatcher::set_workers(const vector<unique_ptr<StreamWorker>>& workers,
                                  const unordered_map<Uuid128, int>& index)
  {
    stream_workers = &workers;
    pod5_index = &index;
    max_queue_size = max_queue_per_worker * workers.size();
    workers_ready = true;

    // Start dispatching queued data to workers
    dispatch_thread = thread(&SamDispatcher::dispatch_loop, this);
  }

  void SamDispatcher::read_loop()
  {
    // Every input is read through open_stream_input(), which remembers the last byte read, so the end of a SAM text
    // can be checked (see StreamInput.h).
    hFILE* input = nullptr;
    int pipe_fd = -1;  // the --bam-cmd command's pipe
    int open_errno = 0;
    if (!bam_cmd.empty()) {
      // Spawn the producer and stream its stdout (as --bam-cmd does in the Python preprocessing).
      // Owning the child is the point: after EOF its exit code says
      // whether the stream was complete, which plain stdin cannot.
      int fds[2];
      if (pipe(fds) != 0) {
        log_err() << "Failed to create pipe for --bam-cmd" << endl;
        producer_error = true;
        is_running = false;
        reading_complete = true;
        queue_cv.notify_all();
        return;
      }
      producer_pid = fork();
      if (producer_pid == 0) {
        // The command stays in the caller's process group and session, like a producer piped in with -b -:
        // Ctrl-C and a hang-up reach it, and a command that prompts on the terminal (ssh, sudo) can read it.
        dup2(fds[1], STDOUT_FILENO);
        close(fds[0]);
        close(fds[1]);
        execl("/bin/sh", "sh", "-c", bam_cmd.c_str(),
              static_cast<char*>(nullptr));
        _exit(127);
      }
      close(fds[1]);
      if (producer_pid < 0) {
        close(fds[0]);
        log_err() << "Failed to fork BAM producer" << endl;
        producer_error = true;
        is_running = false;
        reading_complete = true;
        queue_cv.notify_all();
        return;
      }
      pipe_fd = fds[0];
      input = open_stream_input(fds[0]);
      bam_file = input ? hts_hopen(input, "-", "r") : nullptr;
      if (!bam_file) {
        // Close the pipe here, so a producer that is still writing gets SIGPIPE instead of blocking.
        if (input)
          hclose_abruptly(input);
        else
          close(fds[0]);
      }
      log_info() << "Streaming BAM from producer: " << bam_cmd << endl;
    } else {
      // Open BAM file (supports stdin with "-", which is left open on close, as htslib leaves it)
      int fd = bam_path == "-" ? STDIN_FILENO : open(bam_path.c_str(), O_RDONLY);
      open_errno = fd < 0 ? errno : 0;
      input = fd >= 0 ? open_stream_input(fd, bam_path == "-") : nullptr;
      bam_file = input ? hts_hopen(input, bam_path.c_str(), "r") : nullptr;
      if (!bam_file) {
        if (input)
          hclose_abruptly(input);
        else if (fd >= 0 && bam_path != "-")
          close(fd);
      }
    }
    if (!bam_file) {
      log_err() << "Failed to open BAM file: "
                << (bam_cmd.empty() ? bam_path : bam_cmd)
                << (open_errno ? string(" (") + strerror(open_errno) + ")" : string()) << endl;
      // A BAM that cannot be opened fails the run, for stdin as for --bam-cmd.
      producer_error = true;
      reap_producer();
      is_running = false;
      reading_complete = true;
      queue_cv.notify_all();
      return;
    }

    // Enable multi-threaded BAM decompression
    if (bam_threads > 1) {
      if (hts_set_threads(bam_file, bam_threads) < 0) {
        log_err() << "Failed to set BAM decompression threads" << endl;
      }
    }

    header = sam_hdr_read(bam_file);
    if (!header) {
      log_err() << "Failed to read BAM header" << endl;
      // An empty or non-BAM input fails the run, for stdin as for --bam-cmd.
      producer_error = true;
      stop_after_error(pipe_fd);
      is_running = false;
      reading_complete = true;
      queue_cv.notify_all();
      return;
    }

    // Chunk-based BAM reading
    deque<bam1_t*> read_q;

    while (is_running) {
      // Read a chunk into local read_q
      while (read_q.size() < read_chunk_size) {
        bam1_t* read = bam_init1();
        int r = sam_read1(bam_file, header, read);
        if (r < 0) {
          bam_destroy1(read);
          if (r < -1) {
            // A truncated or corrupt stream, not its end: fail the run.
            log_err() << "Failed to read a BAM record (error " << r << ")" << endl;
            producer_error = true;
          } else if (bam_file->format.compression == htsCompression::bgzf && bam_file->fp.bgzf->no_eof_block) {
            // Every block read cleanly, but the BGZF end-of-file block never came: the stream was cut
            // off between two blocks (as a writer that is killed leaves it), so records are missing.
            log_err() << "The BAM stream ended without the BGZF end-of-file block, so it was cut off" << endl;
            producer_error = true;
          } else if (sam_text_cut_off(bam_file, input)) {
            // SAM text has no end-of-file marker. Its last record was cut off in the middle and was read with a
            // shortened move table or other missing fields, so its rows are wrong.
            log_err() << "The SAM input does not end with a newline, so it was cut off inside its last record" << endl;
            producer_error = true;
          }
          goto read_done;
        }
        read_q.push_back(read);
      }

      // Flush read_q into internal_queue
      {
        unique_lock<mutex> lock(queue_mutex);
        if (internal_queue.size() >= max_queue_size) {
          queue_cv.wait(lock, [this] {
            return internal_queue.size() < max_queue_size || !is_running;
          });
        }
        if (!is_running) break;
        internal_queue.insert(internal_queue.end(), read_q.begin(), read_q.end());
        read_q.clear();
      }
      queue_cv.notify_all();
    }

read_done:
    // Flush remaining reads
    if (!read_q.empty()) {
      unique_lock<mutex> lock(queue_mutex);
      internal_queue.insert(internal_queue.end(), read_q.begin(), read_q.end());
      read_q.clear();
      lock.unlock();
      queue_cv.notify_all();
    }

    reading_complete = true;
    queue_cv.notify_all();
    log_info() << "SAM dispatcher finished reading BAM file" << endl;
    if (producer_error) {
      stop_after_error(pipe_fd);
    } else {
      // Close the input before waiting for the producer: a producer still writing into a full pipe would
      // otherwise never exit, and the wait would never return.
      sam_close(bam_file);
      bam_file = nullptr;
      reap_producer();
    }
  }

  // After an error the run fails: stop the --bam-cmd command (see reap_producer) and let go of the input.
  void SamDispatcher::stop_after_error(int pipe_fd)
  {
    if (bam_threads <= 1) {
      // Without decompression threads the close cannot wait on the pipe, so the input closes first, as at a clean
      // end: a part of the command that still writes, also one that writes when it gets SIGTERM, gets SIGPIPE
      // instead of blocking on a full pipe until the SIGKILL.
      sam_close(bam_file);
      bam_file = nullptr;
      reap_producer();
      return;
    }
    // With decompression threads, closing waits for htslib's reader thread, which waits on the pipe for as long as
    // any part of the command keeps it open, so the input is left open until the process exits. While the command
    // is stopped, what it still writes is read and thrown away, so a command that writes when it gets SIGTERM is not
    // blocked on a full pipe.
    PipeDrain drain(pipe_fd);
    reap_producer();
    bam_file = nullptr;
  }

  void SamDispatcher::reap_producer()
  {
    if (producer_pid <= 0)
      return;
    int status = 0;
    pid_t done = waitpid(producer_pid, &status, WNOHANG);
    bool stopped = false;
    if (done == 0 && producer_error) {
      // After a read error the run fails anyway: stop a command that is still running instead of waiting for one
      // that may never end. Without decompression threads the input is already closed, so a part of the command
      // that still writes gets SIGPIPE; with them the input is still open (see stop_after_error), and a part that
      // outlives the command ends when this process exits.
      kill(producer_pid, SIGTERM);
      stopped = true;
      for (int i = 0; i < 300 && (done = waitpid(producer_pid, &status, WNOHANG)) == 0; ++i)
        this_thread::sleep_for(chrono::milliseconds(100));
      if (done == 0) {
        kill(producer_pid, SIGKILL);
        waitpid(producer_pid, &status, 0);
      }
    } else if (done == 0) {
      while ((done = waitpid(producer_pid, &status, 0)) < 0 && errno == EINTR) {
      }
    }
    const int wait_errno = errno;
    producer_pid = -1;
    if (stopped) {
      log_err() << "Stopped the BAM producer after the read error" << endl;
      return;
    }
    if (done < 0) {
      // The exit status cannot be read (main() undoes an ignored SIGCHLD, the usual cause), so a failed command
      // cannot be told from one that succeeded.
      producer_error = true;
      log_err() << "The exit status of the BAM producer is unknown (waitpid: " << strerror(wait_errno)
                << "); the stream may be truncated -- this run's output is not trustworthy" << endl;
      return;
    }
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
      log_info() << "BAM producer exited cleanly" << endl;
      return;
    }
    producer_error = true;
    if (WIFEXITED(status)) {
      log_err() << "BAM producer exited with status " << WEXITSTATUS(status)
                << "; the stream may be truncated -- this run's output is "
                << "not trustworthy" << endl;
    } else {
      log_err() << "BAM producer was killed by signal "
                << (WIFSIGNALED(status) ? WTERMSIG(status) : -1)
                << "; the stream may be truncated -- this run's output is "
                << "not trustworthy" << endl;
    }
  }

  void SamDispatcher::dispatch_loop()
  {
    deque<bam1_t*> dispatch_q;

    while (true) {
      // Global back-pressure, chunk-granular: while the workers' queues
      // collectively hold >= max_queue_size reads, stop draining. Waited on a
      // separate mutex BEFORE taking queue_mutex, so the stalled dispatcher
      // never blocks the reader's push; internal_queue then fills and the
      // reader stalls on its own existing bound. Total in-flight is therefore
      // capped at ~2 x max_queue_size (+ one 1024-read chunk of overshoot).
      {
        unique_lock<mutex> ilock(inflight_mutex);
        inflight_cv.wait(ilock, [this] {
          return worker_inflight.load(memory_order_relaxed) <
                     static_cast<long>(max_queue_size) ||
                 !is_running;
        });
      }

      // Pop a chunk from internal_queue
      {
        unique_lock<mutex> lock(queue_mutex);

        if (internal_queue.empty()) {
          if (reading_complete) break;
          queue_cv.wait(lock);
          continue;
        }

        if (!workers_ready) {
          lock.unlock();
          this_thread::sleep_for(chrono::milliseconds(100));
          continue;
        }

        size_t count = min(read_chunk_size, internal_queue.size());
        dispatch_q.insert(dispatch_q.end(),
                          internal_queue.begin(), internal_queue.begin() + count);
        internal_queue.erase(internal_queue.begin(), internal_queue.begin() + count);

        bool should_notify = internal_queue.size() <= max_queue_size / 2;
        lock.unlock();
        if (should_notify) queue_cv.notify_all();
      }

      // Dispatch without holding lock
      long dispatched = 0;
      for (auto* read : dispatch_q) {
        // Pre-filter applied before the index so these records cost neither a hash probe nor queue
        // residency.
        if (!BamRecordExtractor::passes_prefilter(read, filter_flag)) {
          bam_destroy1(read);
          continue;
        }

        auto it = pod5_index->find(uuid128_from_str(BamRecordExtractor::parent_id(read)));
        if (it != pod5_index->end()) {
          int worker_idx = it->second;
          if (worker_idx >= 0 && cmp_less(worker_idx, stream_workers->size())) {
            (*stream_workers)[worker_idx]->add_bam_data(read);
            ++dispatched;
          } else {
            bam_destroy1(read);
          }
        } else {
          bam_destroy1(read);
        }
      }
      dispatch_q.clear();
      // One atomic add per chunk; only reads actually handed to a worker count.
      if (dispatched > 0)
        worker_inflight.fetch_add(dispatched, memory_order_relaxed);
    }

    log_info() << "SAM dispatcher finished dispatching to workers" << endl;
  }
}
