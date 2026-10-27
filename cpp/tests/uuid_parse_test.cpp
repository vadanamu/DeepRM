// Checks parse_uuid36 (src/utils/Uuid128.h) against libuuid's own answers. Each line of the vector file is
// "<input bytes as hex>\t<32 hex digits, or - when libuuid refused the text>"; the input is used as a C string,
// so an embedded NUL ends it, as it does for libuuid. Exit 0 only when every line agrees.
// usage: uuid_parse_test uuid_vectors.tsv
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "utils/Uuid128.h"

int main(int argc, char** argv)
{
  std::ifstream in(argc > 1 ? argv[1] : "uuid_vectors.tsv");
  std::string line;
  long n = 0, bad = 0;
  while (std::getline(in, line)) {
    const auto tab = line.find('\t');
    const std::string hex = line.substr(0, tab), want = line.substr(tab + 1);
    std::string text;
    for (size_t i = 0; i + 1 < hex.size(); i += 2) text.push_back(static_cast<char>(std::stoi(hex.substr(i, 2), nullptr, 16)));
    uint8_t bytes[16];
    std::string got = "-";
    if (deeprm::parse_uuid36(text.c_str(), bytes)) {
      char h[33];
      for (int i = 0; i < 16; ++i) std::snprintf(h + 2 * i, 3, "%02x", bytes[i]);
      got = h;
    }
    ++n;
    if (got != want && ++bad <= 5) std::cerr << "MISMATCH input " << hex << " want " << want << " got " << got << "\n";
  }
  std::cout << "vectors=" << n << " mismatches=" << bad << "\n";
  return (n > 0 && bad == 0) ? 0 : 1;
}
