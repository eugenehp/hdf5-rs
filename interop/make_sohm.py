#!/usr/bin/env python3
"""Generate SOHM (shared object header message) reference files.

h5py exposes no API for `H5Pset_shared_mesg_nindexes`, so this calls the
libhdf5 C API directly via ctypes — loading the *same* dylib h5py links
against, which makes the returned identifiers valid inside h5py.

Usage: python3 interop/make_sohm.py <dir>
Creates <dir>/sohm_list.h5 (single index, list storage) and
<dir>/sohm_btree.h5 (two indexes, v2-btree storage forced).
"""
import ctypes
import glob
import os
import sys

import h5py
import numpy as np

SDSPACE, DTYPE, FILL, ATTR = 1 << 1, 1 << 3, 1 << 5, 1 << 12


def _lib():
    libpath = glob.glob(os.path.dirname(h5py.__file__) + "/.dylibs/libhdf5.*.dylib")
    if not libpath:  # non-macOS wheels
        libpath = glob.glob(os.path.dirname(h5py.__file__) + "/../h5py.libs/libhdf5*.so*")
    lib = ctypes.CDLL(libpath[0])
    lib.H5open()
    hid_t = ctypes.c_int64
    lib.H5Pcreate.restype = hid_t
    lib.H5Pcreate.argtypes = [hid_t]
    lib.H5Pset_shared_mesg_nindexes.argtypes = [hid_t, ctypes.c_uint]
    lib.H5Pset_shared_mesg_index.argtypes = [hid_t, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]
    lib.H5Pset_shared_mesg_phase_change.argtypes = [hid_t, ctypes.c_uint, ctypes.c_uint]
    lib.H5Fcreate.restype = hid_t
    lib.H5Fcreate.argtypes = [ctypes.c_char_p, ctypes.c_uint, hid_t, hid_t]
    return lib, hid_t


def make(lib, hid_t, path, nidx, phase=None):
    fcpl = lib.H5Pcreate(hid_t.in_dll(lib, "H5P_CLS_FILE_CREATE_ID_g").value)
    assert lib.H5Pset_shared_mesg_nindexes(fcpl, nidx) >= 0
    if nidx == 1:
        assert lib.H5Pset_shared_mesg_index(fcpl, 0, SDSPACE | DTYPE | FILL | ATTR, 30) >= 0
    else:
        assert lib.H5Pset_shared_mesg_index(fcpl, 0, DTYPE, 30) >= 0
        assert lib.H5Pset_shared_mesg_index(fcpl, 1, SDSPACE | FILL | ATTR, 30) >= 0
    if phase:
        assert lib.H5Pset_shared_mesg_phase_change(fcpl, *phase) >= 0
    fid = lib.H5Fcreate(path.encode(), 2, fcpl, 0)  # H5F_ACC_TRUNC
    assert fid >= 0
    f = h5py.File(h5py.h5f.FileID(fid))
    comp = np.dtype([("a", "<i4"), ("b", "<f8"), ("c", "<i8"), ("d", "<f4")])
    rows = np.zeros(4, dtype=comp)
    rows["a"] = [1, 2, 3, 4]
    rows["b"] = [0.5, 1.5, 2.5, 3.5]
    rows["c"] = [10, 20, 30, 40]
    rows["d"] = [9.0, 8.0, 7.0, 6.0]
    for i in range(6):
        ds = f.create_dataset(f"d{i}", (4,), dtype=comp)
        ds[...] = rows
        ds.attrs["note"] = np.arange(8, dtype="<f8") + i
    for i in range(6):
        f[f"d{i}"].attrs["common"] = np.arange(10, dtype="<i4")
    f.close()


if __name__ == "__main__":
    d = sys.argv[1]
    lib, hid_t = _lib()
    make(lib, hid_t, f"{d}/sohm_list.h5", 1)
    make(lib, hid_t, f"{d}/sohm_btree.h5", 2, phase=(0, 0))
    print("SOHM reference files written")
