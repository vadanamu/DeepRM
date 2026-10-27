/***************************************************************************************************
 *
 * Copyright (C) 2026 Genome4me Incorporated - All Rights Reserved.
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

#include <cstdint>
#include <cstring>
#include <functional>
#include <ostream>
#include <string>

using namespace std;

namespace deeprm {

// 16-byte read/parent UUID used as a hash-map key, replacing the 36-char
// std::string that cost ~80 B and a heap allocation per key. POD5 read ids
// arrive as raw 16 bytes (read_id_t); BAM pi tag / qname arrive as UUID text.
// All ids are assumed to be valid UUIDs (no non-UUID fallback by design).
struct Uuid128 {
  uint64_t hi;
  uint64_t lo;

  bool operator==(const Uuid128& other) const
  {
    return hi == other.hi && lo == other.lo;
  }
};

// Two uint64_t with no padding: a raw 16-byte id maps onto {hi, lo} exactly,
// which is what lets uuid128_from_bytes use a single memcpy.
static_assert(sizeof(Uuid128) == 16, "Uuid128 must be exactly 16 bytes");

// From raw 16 bytes (POD5 read_id_t): no parsing needed. A single fixed-size
// memcpy is byte-for-byte identical to the layout {hi; lo;} and endian-agnostic
// (hi/lo are only ever hashed/compared, never read as numbers). A pointer cast
// is avoided on purpose: `p` (16 parsed bytes / POD5 read_id) is not guaranteed 16-byte
// aligned, and reading a uint8_t buffer through a Uuid128* violates strict
// aliasing (undefined behavior) even when it is. The compiler lowers this
// memcpy to a single aligned/unaligned load, so there is no copy overhead.
inline Uuid128 uuid128_from_bytes(const uint8_t* p)
{
  Uuid128 u;
  memcpy(&u, p, sizeof(u));
  return u;
}

// Value of one hex digit (either case), or -1.
inline int uuid_hex_value(char c)
{
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

// Parses UUID text: exactly 36 characters, '-' at positions 8, 13, 18 and 23, hex digits (either case)
// everywhere else, giving the 16 bytes in text order. Returns false, and leaves `out` unchanged, for
// anything else. The same rule and the same bytes as libuuid's uuid_parse, which it replaces so that the
// binary does not need libuuid.so.1 (cpp/tests/uuid_vectors.tsv holds libuuid's own answers).
inline bool parse_uuid36(const char* s, uint8_t out[16])
{
  if (s == nullptr || strnlen(s, 37) != 36) return false;
  uint8_t bytes[16];
  int k = 0;
  for (int i = 0; i < 36;) {
    if (i == 8 || i == 13 || i == 18 || i == 23) {
      if (s[i] != '-') return false;
      ++i;
      continue;
    }
    const int hi = uuid_hex_value(s[i]), lo = uuid_hex_value(s[i + 1]);
    if (hi < 0 || lo < 0) return false;
    bytes[k++] = static_cast<uint8_t>((hi << 4) | lo);
    i += 2;
  }
  memcpy(out, bytes, 16);
  return true;
}

// From a UUID string (BAM pi tag or qname). A parse failure yields a zeroed
// key; callers assume valid UUIDs, so this is only a defensive floor.
inline Uuid128 uuid128_from_str(const char* s)
{
  uint8_t raw[16];
  if (!parse_uuid36(s, raw)) {
    return Uuid128{0, 0};
  }
  return uuid128_from_bytes(raw);
}

// 32-hex-char rendering for log/error messages (identifier, not canonical form).
inline string uuid128_to_hex(const Uuid128& u)
{
  static const char* kHexDigits = "0123456789abcdef";
  string s(32, '0');
  const uint8_t* p = reinterpret_cast<const uint8_t*>(&u);
  for (int i = 0; i < 16; ++i) {
    s[i * 2] = kHexDigits[p[i] >> 4];
    s[i * 2 + 1] = kHexDigits[p[i] & 0x0F];
  }
  return s;
}

inline ostream& operator<<(ostream& os, const Uuid128& u)
{
  return os << uuid128_to_hex(u);
}

} // namespace deeprm

namespace std {

template <>
struct hash<deeprm::Uuid128> {
  size_t operator()(const deeprm::Uuid128& u) const
  {
    // hi holds 8 bytes of a random UUID and is already well distributed; mix
    // lo in so both halves contribute and near-identical his don't collide.
    return static_cast<size_t>(u.hi ^ (u.lo * 0x9E3779B97F4A7C15ULL));
  }
};

} // namespace std
