import numpy as np, h5py, os
d = "/corpus"

# --- netCDF4: dimension scales, groups, unlimited dims (=> object references)
import netCDF4
nc = netCDF4.Dataset(f"{d}/nc_classic.nc", "w", format="NETCDF4")
nc.createDimension("time", None)
nc.createDimension("lat", 10)
nc.createDimension("lon", 20)
v = nc.createVariable("temp", "f4", ("time", "lat", "lon"), zlib=True, chunksizes=(1, 10, 20))
v[0:3] = np.arange(600, dtype="f4").reshape(3, 10, 20)
v.units = "K"
g = nc.createGroup("model")
gv = g.createVariable("bias", "f8", ("lat",))
gv[:] = np.linspace(0, 1, 10)
nc.setncattr("title", "corpus")
nc.close()

# --- PyTables: tables (bool -> H5T_BITFIELD), nested compound, vlarray, EArray
import tables
class Row(tables.IsDescription):
    name = tables.StringCol(16)
    value = tables.Float64Col()
    count = tables.Int32Col()
    ok = tables.BoolCol()
with tables.open_file(f"{d}/pytables.h5", "w") as h:
    t = h.create_table("/", "measurements", Row, "rows")
    for i in range(50):
        t.row["name"] = f"row{i}".encode()
        t.row["value"] = i / 7.0
        t.row["count"] = i
        t.row["ok"] = i % 2 == 0
        t.row.append()
    t.flush()
    ea = h.create_earray("/", "series", tables.Int64Atom(), (0,))
    ea.append(np.arange(1000))
    vl = h.create_vlarray("/", "ragged", tables.Float32Atom())
    for i in range(5):
        vl.append(np.arange(i + 1, dtype="f4"))
    h.create_array("/", "plain", np.arange(24).reshape(4, 6))

# --- pandas HDFStore (fixed + table formats)
import pandas as pd
df = pd.DataFrame({"a": np.arange(100), "b": np.random.default_rng(0).normal(size=100),
                   "c": [f"s{i}" for i in range(100)]})
df.to_hdf(f"{d}/pandas_fixed.h5", key="df", mode="w", format="fixed")
df.to_hdf(f"{d}/pandas_table.h5", key="df", mode="w", format="table")

# --- h5py: object + region references, MATLAB-style userblock, track_order
with h5py.File(f"{d}/refs.h5", "w") as f:
    a = f.create_dataset("a", data=np.arange(10, dtype="<i4"))
    b = f.create_dataset("b", data=np.arange(20, dtype="<f8"))
    refs = f.create_dataset("objrefs", (2,), dtype=h5py.ref_dtype)
    refs[0] = a.ref
    refs[1] = b.ref
    rr = f.create_dataset("regrefs", (1,), dtype=h5py.regionref_dtype)
    rr[0] = a.regionref[2:5]
    f.attrs["root_ref"] = a.ref
with h5py.File(f"{d}/matlab_style.h5", "w", userblock_size=512) as f:
    f.create_dataset("var", data=np.eye(3))
with open(f"{d}/matlab_style.h5", "r+b") as fh:
    fh.write(b"MATLAB 7.3 MAT-file, corpus test")
with h5py.File(f"{d}/tracked.h5", "w", track_order=True) as f:
    for n in ["zz", "aa", "mm"]:
        f.create_dataset(n, data=np.int32(1))
        f.attrs[n] = n
print("corpus generated:", sorted(os.listdir(d)))
