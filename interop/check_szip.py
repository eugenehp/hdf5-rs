#!/usr/bin/env python3
"""SZip codec cross-validation against libsz (Homebrew libaec).

Usage:
    python3 interop/check_szip.py <dir> gen    # write oracle vectors
    cargo run --features szip --example szval -- <dir>
    python3 interop/check_szip.py <dir> check  # verify rust streams

Requires `brew install libaec`. The full-file check (cd_values + chunk
framing) lives in this repo's history; the Rust side is examples/szval.rs.
"""
import ctypes, os, sys, struct
lib = ctypes.CDLL("/opt/homebrew/opt/libaec/lib/libsz.dylib")
class SZ(ctypes.Structure):
    _fields_ = [("options_mask", ctypes.c_int), ("bits_per_pixel", ctypes.c_int),
                ("pixels_per_block", ctypes.c_int), ("pixels_per_scanline", ctypes.c_int)]
for f in (lib.SZ_BufftoBuffCompress, lib.SZ_BufftoBuffDecompress):
    f.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(SZ)]

EC, LSB, MSB, NN = 4, 8, 16, 32
def comp(data, mask, bpp, ppb, pps):
    p = SZ(mask, bpp, ppb, pps)
    dst = ctypes.create_string_buffer(len(data)*2 + 4096); n = ctypes.c_size_t(len(dst))
    assert lib.SZ_BufftoBuffCompress(dst, ctypes.byref(n), data, len(data), ctypes.byref(p)) == 0, "oracle comp fail"
    return dst.raw[:n.value]
def decomp(blob, mask, bpp, ppb, pps, outlen):
    p = SZ(mask, bpp, ppb, pps)
    dst = ctypes.create_string_buffer(outlen + 4096); n = ctypes.c_size_t(outlen)
    r = lib.SZ_BufftoBuffDecompress(dst, ctypes.byref(n), blob, len(blob), ctypes.byref(p))
    assert r == 0, f"oracle decomp fail {r}"
    return dst.raw[:outlen]

import random
random.seed(7)
ramp16 = b"".join(struct.pack("<H", (i*3) % 4000) for i in range(2048))
cases = [
    ("ec8",    EC|LSB, 8, 8, 16,  bytes(range(256))*4),
    ("nn8",    NN|LSB, 8, 8, 16,  bytes((i*7) % 251 for i in range(1000))),
    ("zeros",  NN|LSB, 8, 16, 64, bytes(2048)),
    ("const",  NN|LSB, 8, 8, 32,  bytes([42])*777),
    ("nn16",   NN|LSB, 16, 16, 128, ramp16),
    ("msb16",  NN|MSB, 16, 16, 128, b"".join(struct.pack(">H", (i*5) % 9999) for i in range(1500))),
    ("nn32",   NN|LSB, 32, 32, 256, b"".join(struct.pack("<I", i*i % 100000) for i in range(1024))),
    ("pad",    NN|LSB, 8, 8, 30,  bytes((i % 17) for i in range(901))),   # pps%ppb!=0 + partial line
    ("rand16", NN|LSB, 16, 32, 512, bytes(random.randrange(256) for _ in range(4096))),
]
d = sys.argv[1]
if sys.argv[2] == "gen":
    for name, mask, bpp, ppb, pps, data in cases:
        open(f"{d}/{name}.params", "w").write(f"{mask} {bpp} {ppb} {pps}")
        open(f"{d}/{name}.data", "wb").write(data)
        open(f"{d}/{name}.oc", "wb").write(comp(data, mask, bpp, ppb, pps))
    print("generated", len(cases), "cases")
else:
    ok = 0
    for name, mask, bpp, ppb, pps, data in cases:
        rc = open(f"{d}/{name}.rc", "rb").read()
        back = decomp(rc, mask, bpp, ppb, pps, len(data))
        assert back == data, f"{name}: oracle cannot decode rust stream correctly"
        ok += 1
    print(f"oracle decodes all {ok} rust streams OK")
