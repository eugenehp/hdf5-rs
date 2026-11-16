"""Write the read benchmark's input: `python3 make_bench.py out.h5`.

Read by `cargo run --release --example bench_read -- out.h5`.
"""
import sys

import h5py
import numpy as np

rows, cols = 4_000_000, 8
rng = np.random.default_rng(0)
# smooth-ish signal: compresses like real sensor data, not like noise
data = (np.cumsum(rng.integers(-3, 4, size=(rows, cols)), axis=0) % 30000).astype("<i2")

with h5py.File(sys.argv[1], "w") as f:
    f.create_dataset("contig", data=data)
    f.create_dataset("plain_chunks", data=data, chunks=(65536, cols))
    f.create_dataset("gzip_rows", data=data, chunks=(65536, cols), compression="gzip", shuffle=True)
    f.create_dataset("gzip_cols", data=data, chunks=(rows // 4, 1), compression="gzip", shuffle=True)
    f.create_dataset("shuffle_f32", data=data.astype("<f4"), chunks=(65536, cols), shuffle=True, fletcher32=True)
    f.create_dataset("be_f64", data=data.astype(">f8"), chunks=(65536, cols))
    f.create_dataset("lzf", data=data, chunks=(65536, cols), compression="lzf", shuffle=True)
    f.create_dataset("strings", data=[f"s{i}" for i in range(100_000)], dtype=h5py.string_dtype())
    many = f.create_group("many")
    for i in range(20_000):
        g = many.create_group(f"g{i}")
        g.attrs["neurodata_type"] = "TimeSeries"
        g.create_dataset("data", data=np.arange(4, dtype="<f4"))
