# hdf5-no-c-wasm — HDF5 for JavaScript

Read and write HDF5 files in the browser or Node with
[`hdf5-no-c`](../README.md) compiled to WebAssembly. No native library is
involved, and remote files are read with HTTP range requests, fetching only
what each call needs.

```bash
wasm-pack build hdf5-wasm --target web      # browsers / bundlers  -> hdf5-wasm/pkg
wasm-pack build hdf5-wasm --target nodejs --out-dir pkg-node --out-name hdf5
# SIMD kernels (shuffle filter, checksums, byte swaps): all current engines
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build hdf5-wasm --target web
```

```js
import init, { H5File, H5Remote } from "./pkg/hdf5_wasm.js";
await init();

// a local file (<input type=file>, drag and drop, fetch, ...)
const f = H5File.fromBytes(new Uint8Array(await file.arrayBuffer()));
f.keys("/");                                  // ["acquisition", "general", ...]
f.kind("acquisition/data");                   // "dataset"
f.shape("acquisition/data");                  // [200000, 16]
f.dtype("acquisition/data");                  // "float32"
f.read("acquisition/data");                   // Float32Array, row-major
f.readSlice("acquisition/data", [100, 0], [10, 16]); // only its chunks decode
f.attrNames("/"); f.readAttr("/", "session_description");

// a remote file: HEAD for the size, then Range requests
const r = await H5Remote.open("https://example.org/session.nwb", 1 << 20);
await r.shape("acquisition/ElectricalSeries/data");
await r.readSlice("acquisition/ElectricalSeries/data", [0, 0], [30000, 4]);
r.bytesFetched(); r.roundTrips();

// writing
const out = H5File.create();
out.createGroup("results");
out.writeDataset("results/trace", new Float32Array(1000), [1000], [250], 4); // chunk, gzip level
out.writeDataset("results/labels", ["a", "b"], [2]);
const bytes = out.toBytes();                  // Uint8Array -> download / upload
```

Values come back as typed arrays (`BigInt64Array` / `BigUint64Array` for
64-bit integers) and strings as arrays of strings. Shapes are plain arrays.

`H5Remote` needs a server that honours `Range` headers. For cross-origin
files, the server must also allow the `Range` request header and expose
`Content-Length` (CORS).

**Threads.** A plain wasm build runs single-threaded. To decode chunks on Web
Workers, build with `+atomics` and shared memory (nightly `-Z build-std`), and
give the library your worker pool with `hdf5::set_thread_spawner` and
`hdf5::set_num_threads` from Rust. Call it from a Worker, since the
browser's main thread may not block.

`node hdf5-wasm/tests/node.mjs [file.h5]` checks the package end to end:
in-memory round trips, plus range reads from a local HTTP server.
