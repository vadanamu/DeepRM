// Checks the stream reader's input (src/sam/StreamInput.cpp): every input reads the same records as htslib's own
// sam_open(), and sam_text_cut_off() is true exactly for a SAM text whose last line was cut off (inside the move
// table, inside a later tag, or just before its newline), never for a complete SAM, one cut at a line end, one with
// CRLF line ends, a BAM or a BGZF-compressed SAM. Each case is read from a file and from a pipe, with 1 and 4 htslib
// threads. It also checks that closing the input leaves standard input open, as htslib's own hopen("-") does, and
// closes any other descriptor. Exit 0 only when every case agrees.
// usage: sam_text_end_test [folder for its temporary files]
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <htslib/hts.h>
#include <htslib/sam.h>

#include "sam/StreamInput.h"

using namespace std;
using namespace deeprm;

namespace {
  // Header and records shaped like Dorado's: a move table, then the sample counts after it.
  string sam_text(int n_records)
  {
    string s = "@HD\tVN:1.6\tSO:unsorted\n@SQ\tSN:tx1\tLN:3000\n";
    for (int i = 0; i < n_records; ++i) {
      const int len = 150 + 7 * i;
      string seq, qual, mv = "mv:B:c,6";
      for (int j = 0; j < len; ++j) {
        seq += "ACGT"[(i + j * 7) % 4];
        qual += static_cast<char>('!' + (i + j) % 40);
        for (int k = 0; k < 1 + (i + j) % 4; ++k)
          mv += k == 0 ? ",1" : ",0";
      }
      s += "read" + to_string(i) + "\t0\ttx1\t" + to_string(1 + i) + "\t60\t" + to_string(len) + "M\t*\t0\t0\t" + seq
           + "\t" + qual + "\tMD:Z:" + to_string(len) + "\t" + mv + "\tts:i:100\tns:i:" + to_string(100000 + i) + "\n";
    }
    return s;
  }

  bool write_file(const string& path, const string& bytes)
  {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f)
      return false;
    const bool ok = fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
    return fclose(f) == 0 && ok;
  }

  // The records of a SAM or BAM written with htslib (mode "wb" for BAM, "wz" for BGZF-compressed SAM).
  bool convert(const string& sam_path, const string& out_path, const char* mode)
  {
    samFile* in = sam_open(sam_path.c_str(), "r");
    samFile* out = sam_open(out_path.c_str(), mode);
    sam_hdr_t* h = in ? sam_hdr_read(in) : nullptr;
    bool ok = in && out && h && sam_hdr_write(out, h) == 0;
    bam1_t* b = bam_init1();
    int r;
    while (ok && (r = sam_read1(in, h, b)) >= 0)
      ok = sam_write1(out, h, b) >= 0;
    bam_destroy1(b);
    if (h)
      sam_hdr_destroy(h);
    if (in)
      sam_close(in);
    if (out)
      ok = sam_close(out) == 0 && ok;
    return ok;
  }

  struct Result {
    bool opened = false, read_error = false, cut_off = false;
    vector<string> records;  // each record's core fields and data, as bytes
  };

  // Reads every record of `f` (`input` is its hFILE from open_stream_input, or nullptr for sam_open()).
  Result read_all(htsFile* f, hFILE* input, int threads)
  {
    Result res;
    if (!f)
      return res;
    res.opened = true;
    if (threads > 1)
      hts_set_threads(f, threads);
    sam_hdr_t* h = sam_hdr_read(f);
    if (!h) {
      res.read_error = true;
      sam_close(f);
      return res;
    }
    bam1_t* b = bam_init1();
    int r;
    while ((r = sam_read1(f, h, b)) >= 0) {
      string rec(reinterpret_cast<const char*>(&b->core), sizeof(b->core));
      rec.append(reinterpret_cast<const char*>(b->data), b->l_data);
      res.records.push_back(rec);
    }
    res.read_error = r < -1;
    if (input && !res.read_error)
      res.cut_off = sam_text_cut_off(f, input);
    bam_destroy1(b);
    sam_hdr_destroy(h);
    sam_close(f);
    return res;
  }

  Result read_through_input(int fd, const string& name, int threads)
  {
    hFILE* input = open_stream_input(fd);
    htsFile* f = input ? hts_hopen(input, name.c_str(), "r") : nullptr;
    if (!f && input)
      hclose_abruptly(input);
    return read_all(f, input, threads);
  }

  // The same bytes through a pipe, written by a child process, as a basecaller's output arrives.
  Result read_from_pipe(const string& path, int threads)
  {
    int fds[2];
    if (pipe(fds) != 0)
      return Result();
    pid_t pid = fork();
    if (pid == 0) {
      close(fds[0]);
      int in = open(path.c_str(), O_RDONLY);
      char buf[1 << 16];
      ssize_t n;
      while (in >= 0 && (n = read(in, buf, sizeof(buf))) > 0)
        if (write(fds[1], buf, n) != n)
          _exit(1);
      _exit(0);
    }
    close(fds[1]);
    Result res = read_through_input(fds[0], "-", threads);
    waitpid(pid, nullptr, 0);
    return res;
  }

  bool is_open(int fd)
  {
    return fcntl(fd, F_GETFD) != -1;
  }

  // Whether closing an input leaves the descriptor open: standard input (shared) as htslib's hopen("-") leaves it,
  // any other descriptor closed. Standard input is a pipe here, put in place for the check.
  int check_close_of_standard_input()
  {
    int fds[2];
    const int saved = dup(STDIN_FILENO);
    if (saved < 0 || pipe(fds) != 0 || dup2(fds[0], STDIN_FILENO) < 0) {
      cerr << "cannot set up standard input\n";
      return 1;
    }
    close(fds[0]);
    hFILE* htslib_stdin = hopen("-", "r");
    const bool htslib_left_open = htslib_stdin && hclose(htslib_stdin) == 0 && is_open(STDIN_FILENO);
    hFILE* ours = open_stream_input(STDIN_FILENO, true);
    const bool ours_left_open = ours && hclose(ours) == 0 && is_open(STDIN_FILENO);
    const int other = dup(STDIN_FILENO);
    hFILE* owned = other >= 0 ? open_stream_input(other) : nullptr;
    const bool owned_closed = owned && hclose(owned) == 0 && !is_open(other);
    dup2(saved, STDIN_FILENO);
    close(saved);
    close(fds[1]);
    const bool ok = htslib_left_open && ours_left_open && owned_closed;
    cout << (ok ? "ok   " : "FAIL ") << "close: standard input left open (htslib " << (htslib_left_open ? "yes" : "no")
         << ", ours " << (ours_left_open ? "yes" : "no") << "), another descriptor closed ("
         << (owned_closed ? "yes" : "no") << ")\n";
    return ok ? 0 : 1;
  }
}

int main(int argc, char** argv)
{
  string dir = argc > 1 ? argv[1] : "/tmp";
  char tmpl[4096];
  snprintf(tmpl, sizeof(tmpl), "%s/sam_text_end_XXXXXX", dir.c_str());
  if (!mkdtemp(tmpl)) {
    perror("mkdtemp");
    return 1;
  }
  dir = tmpl;
  const int n = 40;
  const string whole = sam_text(n);
  const size_t last_start = whole.rfind('\n', whole.size() - 2) + 1;
  const size_t mv_at = whole.find("\tmv:B:c,", last_start);
  const size_t ns_at = whole.find("\tns:i:", last_start);
  string crlf;
  for (char c : whole)
    crlf += c == '\n' ? string("\r\n") : string(1, c);

  struct Case {
    string name, bytes, convert_mode;
    bool cut_off;
    int records;
  };
  vector<Case> cases = {
    {"whole", whole, "", false, n},
    {"cut inside the move table", whole.substr(0, mv_at + 401), "", true, n},
    {"cut inside a tag after it", whole.substr(0, ns_at + 8), "", true, n},
    {"newline of the last line missing", whole.substr(0, whole.size() - 1), "", true, n},
    {"cut at a line end", whole.substr(0, last_start), "", false, n - 1},
    {"CRLF line ends", crlf, "", false, n},
    {"BAM", whole, "wb", false, n},
    {"BGZF-compressed SAM", whole, "wz", false, n},
  };

  const string sam_path = dir + "/whole.sam";
  if (!write_file(sam_path, whole)) {
    cerr << "cannot write " << sam_path << "\n";
    return 1;
  }
  int bad = check_close_of_standard_input();
  for (size_t i = 0; i < cases.size(); ++i) {
    const Case& c = cases[i];
    const string path = dir + "/case" + to_string(i);
    const bool made = c.convert_mode.empty() ? write_file(path, c.bytes)
                                             : convert(sam_path, path, c.convert_mode.c_str());
    if (!made) {
      cerr << "cannot write " << path << "\n";
      return 1;
    }
    for (int threads : {1, 4}) {
      // htslib's own reading of the same file is the reference for the records
      const Result want = read_all(sam_open(path.c_str(), "r"), nullptr, threads);
      for (const char* how : {"file", "pipe"}) {
        Result got = strcmp(how, "file") == 0 ? read_through_input(open(path.c_str(), O_RDONLY), path, threads)
                                              : read_from_pipe(path, threads);
        const bool ok = got.opened && !got.read_error && got.cut_off == c.cut_off
                        && static_cast<int>(got.records.size()) == c.records && got.records == want.records;
        cout << (ok ? "ok   " : "FAIL ") << c.name << " (" << how << ", " << threads << " threads): records "
             << got.records.size() << ", cut off " << (got.cut_off ? "yes" : "no") << "\n";
        bad += !ok;
      }
    }
    unlink(path.c_str());
  }
  unlink(sam_path.c_str());
  rmdir(dir.c_str());
  cout << "cases=" << cases.size() << " failures=" << bad << "\n";
  return bad == 0 ? 0 : 1;
}
