#!/usr/bin/env python3
"""Cross-validation of the pure-Rust `hdf5` crate against h5py/libhdf5.

Usage:
    python3 interop/check_h5py.py write <dir>   # create reference files for Rust to read
    python3 interop/check_h5py.py verify <dir>  # verify files written by Rust

The Rust side lives in `hdf5/examples/interop_harness.rs`.
"""
import os
import sys
import numpy as np
import h5py


def write_reference(d):
    """Files the Rust crate must be able to READ."""
    with h5py.File(f"{d}/py_earliest.h5", "w", libver="earliest") as f:
        f.create_dataset("scalar", data=np.float64(1.25))
        f.create_dataset("v", data=np.arange(20, dtype="<i8"))
        g = f.create_group("g")
        g.create_dataset("m", data=np.arange(24, dtype="<f4").reshape(4, 6),
                         chunks=(2, 3), compression="gzip", shuffle=True, fletcher32=True)
        g.attrs["name"] = "reference"
        f["soft"] = h5py.SoftLink("/g/m")
        comp = np.zeros(3, dtype=[("a", "<i4"), ("b", "<f8")])
        comp["a"] = [1, 2, 3]
        comp["b"] = [0.5, 1.5, 2.5]
        f.create_dataset("comp", data=comp)
        f.create_dataset("strs", data=np.array(["x", "yz", ""], dtype=h5py.string_dtype()))
    with h5py.File(f"{d}/py_latest.h5", "w", libver="latest") as f:
        f.create_dataset("x", data=np.arange(10, dtype="<f8"))
        g = f.create_group("g")
        g.create_dataset("y", data=np.arange(6, dtype="<i4").reshape(2, 3),
                         chunks=(1, 3), compression="gzip")
        g.attrs["meta"] = np.float32(9.5)
        # extensible-array chunk index (1 unlimited dim)
        ds = f.create_dataset("grow", shape=(4, 6), maxshape=(None, 6),
                              chunks=(2, 3), dtype="<i4")
        ds[...] = np.arange(24).reshape(4, 6)
        ds.resize((10, 6))
        ds[4:10] = np.arange(24, 60).reshape(6, 6)
        # v2 btree chunk index (2 unlimited dims) + filters
        dz = f.create_dataset("two", shape=(6, 6), maxshape=(None, None),
                              chunks=(2, 2), dtype="<i8", compression="gzip",
                              shuffle=True)
        dz[...] = np.arange(36).reshape(6, 6)
        # dense links + dense attributes
        many = f.create_group("many")
        for i in range(20):
            many.create_dataset(f"d{i:03}", data=np.int32(i))
        af = f.create_dataset("attrful", data=np.int64(1))
        for i in range(12):
            af.attrs[f"a{i:03}"] = np.float64(i)
    # big-endian data
    with h5py.File(f"{d}/py_bigendian.h5", "w", libver="earliest") as f:
        f.create_dataset("bi", data=np.array([1, -2, 300000], dtype=">i4"))
        f.create_dataset("bf", data=np.array([1.5, -2.25], dtype=">f8"))
        f.attrs["battr"] = np.array([9], dtype=">i8")
    # external link pair
    with h5py.File(f"{d}/py_ext_target.h5", "w", libver="earliest") as f:
        f.create_dataset("payload", data=np.arange(5, dtype="<i4"))
    with h5py.File(f"{d}/py_ext_source.h5", "w", libver="earliest") as f:
        f["ext"] = h5py.ExternalLink("py_ext_target.h5", "/payload")
    # committed datatype (shared dtype message), LZF, scaleoffset
    with h5py.File(f"{d}/py_misc.h5", "w", libver="earliest") as f:
        f["mytype"] = np.dtype([("a", "<i4"), ("b", "<f8")])
        ds = f.create_dataset("committed", (2,), dtype=f["mytype"])
        ds[...] = np.array([(1, .5), (2, 1.5)], dtype=f["mytype"])
        f.create_dataset("lzf", data=np.arange(1000, dtype="<i4"),
                         chunks=(100,), compression="lzf")
        f.create_dataset("scaleoffset", data=np.arange(100, dtype="<i4"),
                         chunks=(50,), scaleoffset=0)
    # huge dense attribute (fractal-heap huge object)
    with h5py.File(f"{d}/py_hugeattr.h5", "w", libver="latest") as f:
        ds = f.create_dataset("x", data=np.int32(1))
        ds.attrs["big"] = np.arange(2000, dtype="<f8")
    # virtual dataset over two sources
    layout = h5py.VirtualLayout(shape=(2, 5), dtype="<i4")
    with h5py.File(f"{d}/py_vds_src.h5", "w", libver="earliest") as f:
        for i in range(2):
            f.create_dataset(f"s{i}", data=np.arange(i*5, i*5+5, dtype="<i4"))
    for i in range(2):
        layout[i] = h5py.VirtualSource(f"{d}/py_vds_src.h5", f"s{i}", shape=(5,))
    with h5py.File(f"{d}/py_vds.h5", "w", libver="latest") as f:
        f.create_virtual_dataset("virt", layout)
    # blosc (requires hdf5plugin; skipped when unavailable)
    try:
        import hdf5plugin
        with h5py.File(f"{d}/py_blosc.h5", "w", libver="earliest") as f:
            data = np.arange(5000, dtype="<i4")
            for cname in ("blosclz", "lz4", "zstd", "zlib"):
                f.create_dataset(cname, data=data, chunks=(1000,),
                                 **hdf5plugin.Blosc(cname=cname, clevel=5,
                                                    shuffle=hdf5plugin.Blosc.SHUFFLE))
    except ImportError:
        print("hdf5plugin unavailable; blosc reference skipped")
    # SOHM files (needs the ctypes generator; best-effort)
    try:
        import subprocess
        subprocess.run([sys.executable,
                        os.path.join(os.path.dirname(__file__), "make_sohm.py"), d],
                       check=True)
    except Exception as e:
        print(f"SOHM reference skipped: {e}")
    print("reference files written")


def verify_rust(d):
    """Verify files WRITTEN by the Rust crate."""
    with h5py.File(f"{d}/rust_all.h5", "r") as f:
        assert f["scalar_f64"][()] == 6.5
        assert list(f["vec_i16"][()]) == [3, 1, 4, 1, 5]
        m = f["grp/mat_u32"][()]
        assert m.shape == (3, 4) and m[2, 3] == 11
        ds = f["chunked"]
        assert ds.chunks == (2, 2) and ds.compression == "gzip" and ds.shuffle
        assert ds[3, 3] == 15
        comp = f["compound"][()]
        assert comp[1]["id"] == 2 and abs(comp[1]["value"] - 1.5) < 1e-12
        strs = [s.decode() if isinstance(s, bytes) else s for s in f["strings"][()]]
        assert strs == ["alpha", "βeta", ""]
        seqs = [list(x) for x in f["seqs"][()]]
        assert seqs == [[1, 2], [3, 4, 5]]
        assert f.attrs["version"] == 3
        assert f["grp"].attrs["desc"].startswith("rust")
        # soft link resolves
        assert f["alias"][2, 3] == 11
        # enum dtype survives
        et = h5py.check_enum_dtype(f["colors"].dtype)
        assert et == {"RED": 1, "GREEN": 2, "BLUE": 3}, et
        assert list(f["colors"][()]) == [1, 3, 2]
    # rust-written external links resolve in h5py
    with h5py.File(f"{d}/rust_ext_source.h5", "r") as f:
        assert list(f["borrowed"][()]) == [10, 20, 30]
        lnk = f.get("borrowed", getlink=True)
        assert isinstance(lnk, h5py.ExternalLink)
    with h5py.File(f"{d}/rust_all.h5", "r") as f:
        # LZF chunks
        z = f["lzfd"]
        assert z.compression == "lzf"
        assert list(z[()]) == list(range(500))
        # dense attributes incl. one above the 64 KB compact limit
        big = f["dense_attr_target"].attrs["big"]
        assert big.shape == (10000,) and big[9999] == 9999.0
    # rust-written blosc decompresses in h5py (needs hdf5plugin)
    try:
        import hdf5plugin
        with h5py.File(f"{d}/rust_all.h5", "r") as f:
            assert list(f["bloscd"][()]) == list(range(400))
    except ImportError:
        pass
    print("rust-written file verified by h5py")


if __name__ == "__main__":
    cmd, d = sys.argv[1], sys.argv[2]
    if cmd == "write":
        write_reference(d)
    elif cmd == "verify":
        verify_rust(d)
    else:
        raise SystemExit(f"unknown command {cmd}")
