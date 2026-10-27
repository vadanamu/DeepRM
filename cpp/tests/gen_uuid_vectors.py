"""Writes uuid_vectors.tsv: the input bytes as hex, a tab, then libuuid's answer (32 hex digits of the 16 bytes), or
'-' when libuuid's uuid_parse refuses the text. Input bytes are stored as hex so that any byte (high bytes, NUL,
'+', space) reaches the test unchanged. libuuid.so.1 is needed only to make the vectors, never to build or run
deeprm_preprocess. usage: python gen_uuid_vectors.py uuid_vectors.tsv
"""

import ctypes
import random
import sys
import uuid

lib = ctypes.CDLL("libuuid.so.1")
lib.uuid_parse.argtypes = [ctypes.c_char_p, ctypes.c_char_p]


def libuuid_answer(text):
    out = ctypes.create_string_buffer(16)
    return out.raw.hex() if lib.uuid_parse(text, out) == 0 else "-"


rng = random.Random(1)
good = [str(uuid.UUID(int=rng.getrandbits(128))).encode() for _ in range(2000)]
cases = list(good)
cases += [c.upper() for c in good[:300]]
cases += ["".join(ch.upper() if rng.random() < 0.5 else ch for ch in c.decode()).encode() for c in good[300:600]]
# one character replaced: hex digits, dash, other letters, '+', space, NUL and high bytes
alphabet = [ord(c) for c in "0123456789abcdefABCDEF-gGxz+ _"] + [0, 0x80, 0xA0, 0xC3, 0xFF]
for c in good[:2000]:
    s = bytearray(c)
    s[rng.randrange(36)] = rng.choice(alphabet)
    cases.append(bytes(s))
# wrong length or shifted dashes
cases += [c[:35] for c in good[:100]] + [c + b"0" for c in good[:100]] + [c[1:] + b"-" for c in good[:100]]
cases += [b"+" + c[1:] for c in good[:50]] + [b" " + c[1:] for c in good[:50]] + [c[:35] + b" " for c in good[:50]]
cases += [b"", b"-" * 36, b"0" * 36, b"00000000-0000-0000-0000-000000000000",
          b"ffffffff-ffff-ffff-ffff-ffffffffffff", b"FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF"]
with open(sys.argv[1], "w") as f:
    for c in cases:
        f.write(c.hex() + "\t" + libuuid_answer(c) + "\n")
print("vectors", len(cases), "accepted by libuuid", sum(libuuid_answer(c) != "-" for c in cases))
