# hdf5-no-c — pure-Rust HDF5

A pure-Rust implementation of HDF5 with a **drop-in API compatible with the
FFI-based [`hdf5`](https://github.com/aldanor/hdf5-rust) crate** — no C
library, no build scripts probing for libhdf5, no unsafe FFI.

Files written by this crate are valid HDF5 readable by **h5py, libhdf5 and
every other HDF5 consumer**, and it reads files written by those libraries.

```toml
[dependencies]
hdf5-no-c = "0.1"
```

The package is named `hdf5-no-c`, but the **library target is named `hdf5`**,
so all code written against the FFI crate — `use hdf5::prelude::*`, the
`#[derive(H5Type)]` macro, everything — compiles unchanged.

Repository: <https://github.com/eugenehp/hdf5-rs>

```rust
use hdf5::{File, H5Type};
use ndarray::arr2;

#[derive(H5Type, Clone, PartialEq, Debug)]
#[repr(u8)]
enum Color { R = 1, G = 2, B = 3 }

#[derive(H5Type, Clone, PartialEq, Debug)]
#[repr(C)]
struct Pixel { xy: (i64, i64), color: Color }

fn main() -> hdf5::Result<()> {
    let file = File::create("pixels.h5")?;                 // open for writing
    let group = file.create_group("dir")?;                 // create a group
    let ds = group
        .new_dataset_builder()
        .with_data(&arr2(&[
            [Pixel { xy: (1, 2), color: Color::R }, Pixel { xy: (2, 3), color: Color::B }],
            [Pixel { xy: (3, 4), color: Color::G }, Pixel { xy: (4, 5), color: Color::R }],
        ]))
        .create("pixels")?;                                // write a dataset
    let attr = ds.new_attr::<Color>().shape([3]).create("colors")?;
    attr.write(&ndarray::arr1(&[Color::R, Color::G, Color::B]))?;
    Ok(())
}
```

## Workspace

| Crate | Role |
|---|---|
| `hdf5-no-c` (lib name `hdf5`, folder `hdf5/`) | High-level API (drop-in surface of the FFI crate) + the pure-Rust format engine |
| `hdf5-no-c-types` (lib name `hdf5_types`, folder `hdf5-types/`) | Native Rust equivalents of HDF5 types (`H5Type`, `TypeDescriptor`, vlen/fixed strings & arrays, dynamic values) |
| `hdf5-no-c-derive` (lib name `hdf5_derive`, folder `hdf5-derive/`) | `#[derive(H5Type)]` for structs, tuple structs and enums |
| `hdf5-no-c-wasm` (folder `hdf5-wasm/`, npm via `wasm-pack`) | JavaScript bindings: read/write in the browser and Node, remote files over HTTP range requests ([README](hdf5-wasm/README.md)) |

All three packages carry the `-no-c` prefix on crates.io (the unprefixed
names are taken by the FFI project), while every **library** keeps its
original crate name — imports and derive output compile unchanged.

## Examples

Getting started:

```bash
cargo run --example quickstart   # typed datasets, groups, attributes
cargo run --example tour         # chunking+compression, resizing, slicing, links, vlen
```

Feature deep-dives (each self-contained): `compound`, `vlen`, `denseattr`,
`advanced_filters` (scaleoffset/nbit/bit-shuffle/zstd + dense links),
`sohm_vds_write` (shared-message tables + virtual datasets), `userblock`.

With feature flags:

```bash
cargo run --features rlx  --example rlx_tensor # datasets straight into RLX tensors
cargo run --features rlx  --example rlx_mlp    # HDF5 checkpoint -> RLX forward pass
cargo run --features mpi  --example mpi_demo   # 4 processes, one shared file
cargo run --features mpi-rs --example mpi_rs_demo # same, over the mpi-rs crate
cargo run --features szip --example szval -- <dir>  # szip codec validation
```

Validation tooling: `corpus -- <dir>` (real-world file sweep), `fuzz -- <dir>`
(mutation fuzzing), `interop_harness` (bidirectional h5py verification).

### HDF5 → RLX in five lines

```rust
let f = hdf5::File::open("model.h5")?;
let w1 = f.dataset("model/w1")?.read_tensor()?;   // any numeric dtype -> f32
let w2 = f.dataset("model/w2")?.read_tensor()?;   // chunked/compressed is fine
let y = x.matmul(&w1).relu().matmul(&w2);          // symbolic graph, fuses
let out = y.to_vec();                              // compiled forward pass
```

## Cargo features

| Feature | Effect |
|---|---|
| *(default)* | Full read/write engine; DEFLATE, shuffle, fletcher32, LZF, blosc, scale-offset, n-bit filters |
| `szip` | Pure-Rust SZip (extended-Rice/CCSDS 121.0) codec, validated against libaec |
| `mpi` | SPMD collective files over a built-in TCP mini-MPI (`hdf5::mpi`) — one shared physical file written by N Rust processes; not wire-compatible with OpenMPI |
| `mpi-rs` | Back `hdf5::mpi::Comm` with the pure-Rust [mpi-rs](https://crates.io/crates/mpi-rs) crate (rsmpi-compatible API): launch ranks with its `mpiexec`, adopt an app-initialized universe via `Comm::from_mpi_rs`, or run standalone as a singleton. Implies `mpi` |
| `rlx` | `Container::read_tensor()` — load any numeric dataset/attribute directly as an [RLX](https://crates.io/crates/rlx) tensor (`rlx@0.2.10`). Ops compose lazily, fuse, and run on RLX's bundled cpu backend — HDF5 checkpoints feed straight into inference (see `examples/rlx_mlp.rs`) |
| `complex` | Complex-number datatypes (`num-complex`) |
| `f16` | Half-precision floats (`half`) |

MSRV: 1.85. No build scripts, no C toolchain, no `HDF5_DIR` probing.

## Performance: SIMD and parallelism

Always on, no feature flag, no extra dependencies:

- **Parallel chunk I/O.** Reading decodes chunks on every core, then scatters
  them in parallel by row band: `out` is split at band edges, so each worker
  owns the bytes it writes, with no locks or `unsafe`. Writing gathers and
  filters chunks in parallel, then appends them in order, so the file is
  byte-for-byte what one thread writes. Large byte-swaps, numeric conversions
  and contiguous copies are split over threads too. Work runs on a small
  persistent pool, so a small read costs a queue push per helper, not a
  thread start. `hdf5::set_num_threads(n)` or `HDF5_NO_C_THREADS=n` sets the
  worker count (default: one per core); `1` keeps everything on the calling
  thread. `hdf5::set_thread_spawner` hands the work to threads the
  application owns instead (Web Workers, a custom pool).
- **Partial reads decode only what they touch.** `read_slice` (any
  hyperslab) and point selections read just the chunks holding a selected
  element, or just the selected runs of contiguous data, never the whole
  dataset. Decoded chunks are kept per dataset in an LRU cache (16 MiB
  default, `hdf5::set_chunk_cache_size`), so neighbouring partial reads don't
  decode the same chunk twice.
- **One pass per read.** Decoding, byte-swapping (big-endian data) and type
  conversion (e.g. `i16` on disk read as `f32`) happen as each chunk is
  placed, not as separate passes over a full-size intermediate copy. The
  filter pipeline borrows its input and sizes buffers from the known chunk
  size.
- **SIMD kernels** (`format/simd.rs`). The shuffle filter (both directions,
  element sizes 2/4/8/16) is written as a byte-zip/unzip "perfect shuffle"
  for NEON (aarch64), SSE2 (x86_64), AVX2 (x86_64: two 16-element groups per
  256-bit vector, used where calibration shows it faster) and wasm `simd128`. Scale-offset and
  N-bit decode read each packed field through one 128-bit window instead of
  bit-by-bit. Fletcher32 uses a
  closed-form block sum the compiler vectorizes. Byte-swaps and every numeric
  conversion pair (all 100 integer/float combinations, with libhdf5's
  saturating semantics) are tight typed loops that auto-vectorize; on wasm,
  explicit `simd128` kernels cover the cases LLVM leaves scalar. Each vector
  path is tested against its scalar reference.
- **Run-based chunk scatter/gather.** Chunk ↔ dataset copies move whole
  contiguous runs, with fixed-size moves for narrow runs (column chunks).
- **Metadata parsing** reads fields from a cached window of the source,
  not through one dynamic call and lock per field. Range readers check the
  last block used without taking a lock, address maps use a multiply hash
  instead of SipHash, and the per-object arena slot shrank from 440 bytes to
  about 100 (dataset descriptions are boxed).

`examples/bench_read.rs` on a 4M×8 `i16` file written by h5py
(`python3 interop/make_bench.py bench.h5`), Apple M-series, 14 threads, best
of 3. It also checks that every layout decodes to the same values:

| Workload | Before | After | After, 1 thread |
|---|---:|---:|---:|
| contiguous read | 5.1 ms | 2.1 ms | 5.2 ms |
| gzip + shuffle, row chunks | 15.6 ms | 14.3 ms | 103 ms |
| gzip + shuffle, column chunks | 133 ms | 18.8 ms | 116 ms |
| shuffle + fletcher32 `f32` | 10.2 ms | 7.1 ms | 23.8 ms |
| big-endian `f64` (256 MB) | 66.4 ms | 10.3 ms | 25.2 ms |
| `i16` read as `f32` | 176 ms | 3.8 ms | 7.3 ms |
| gzip + shuffle **write** (64 MB) | 895 ms | 70.5 ms | 510 ms |
| walk 20k groups (attr + shape) | 64.1 ms | 47.1 ms | 46.7 ms |

Kernel throughput (`cargo test --release -p hdf5-no-c simd::tests::throughput
-- --ignored --nocapture`), GB/s, on a 1 MiB chunk in cache:

| Kernel | AMD Ryzen 9 8945HS | Intel i9-12900H | scalar |
|---|---:|---:|---:|
| shuffle 4 (AVX2 / SSE2) | 57.8 / 29.2 | 32.9 / 25.3 | ~1.5 |
| unshuffle 8 (AVX2 / SSE2) | 51.7 / 25.3 | 31.5 / 31.1 | — |
| fletcher32 | 45.7 | 39.2 | 4.3 |

AVX2's byte shuffles beat SSE2's on Zen 4 but not always on Alder Lake.
Intel issues 128-bit byte shuffles on two ports and 256-bit ones on one,
so AVX2 unshuffle is up to ~30% slower there. On first use the library
times both on a chunk-sized buffer and keeps AVX2 only where it is at
least 10% faster (once per process, a few milliseconds). Fletcher32 on
x86 uses `pmaddwd` (SSE2/AVX2) on bias-shifted words, which is about 5×
the compiler's own vectorization. Blosc's byte shuffle and bitshuffle use
the same kernels. On wasm `simd128`: fletcher32 2.4 → 10.8 GB/s, 8-byte
swap 7.8 → 33.

Tested on macOS (aarch64, plus x86_64 under Rosetta), Linux x86_64 on Intel
and AMD (the full workspace suite, h5py interop in both directions, and the
benchmarks), wasm32-wasip1 (with and without SIMD), wasm32-wasip1-threads,
and headless Chrome (`hdf5-wasm/tests/browser.html`).

## Streaming writes

Writable files on disk (`File::create`, `open_rw`, `append`) are never
rebuilt in memory:

- **Flushes append.** What changed goes after the current end of the file,
  followed by fresh metadata, and only the superblock is rewritten. Data
  already in the file stays where it is (chunks are re-indexed, not
  re-encoded), and a file opened and closed without changes is not written
  at all.
- **Written datasets leave memory.** After a flush, each written dataset is
  a view of the file, like a dataset of a file opened for reading.
- **Chunk writes go straight to disk.** `write_slice` (or any write) into a
  chunked dataset decodes only the chunks it touches, updates them,
  re-encodes them in parallel and appends them to the file. `resize` of a
  chunked dataset only changes its shape. Recording loops
  (`resize` + `write_slice` + an occasional `flush`) hold one block of data
  at a time, whatever the file grows to (`examples/streaming_write.rs`).
- **Files from other writers** are edited the same way: chunks under any of
  libhdf5's chunk indexes (v1/v2 B-tree, fixed or extensible array) stay in
  place under a new v1 B-tree index, and h5py reads the result.

Old metadata stays in the file, unreferenced (libhdf5 reuses such space;
this writer does not yet), so a file flushed very often carries some dead
bytes. Rewrite it with `to_bytes()` to compact it. Files with shared-message
tables (SOHM) fall back to a whole-file rewrite on flush.

## Remote files (async range reads)

`hdf5::remote::RemoteFile` reads a file through any `AsyncRangeRead`
(`fetch` with a `Range` header, an object store, ...), with no executor
dependency. Reads run against the bytes fetched so far. Every range an
attempt is missing gets fetched concurrently, and the read then runs again:

```rust
let remote = RemoteFile::open(reader, 1 << 20).await?;
let rows = remote.read(|f| f.dataset("data")?.read_rows_raw::<f32>(0, 64)).await?;
```

Opening a file takes about 2 round-trips, and a slice fetches only its
chunks. For example, 64 rows of a gzip column-chunked dataset in a 641 MB file
written by h5py came over HTTP in 6 MB. `hdf5-wasm`'s `H5Remote` wraps this
for JavaScript.

## WebAssembly

The crate builds and passes its test suite on `wasm32-unknown-unknown` (the
browser), `wasm32-wasip1` and `wasm32-wasip1-threads`, with or without
`simd128`. The [`hdf5-wasm`](hdf5-wasm/README.md) package wraps it for
JavaScript.

- **Browser** (`wasm32-unknown-unknown`): there is no file system, so files
  live in memory. `File::from_bytes(bytes)` opens bytes you already have,
  e.g. from `fetch` or a browser `File` (read-only and zero-copy; objects
  are parsed as they are reached). `File::from_bytes_rw` opens them for
  editing, `File::create_in_memory` builds a new file, and `to_bytes()`
  exports either one. `File::open_reader` takes any `Read + Seek`. Without a
  clock, object times are 0. Without threads, everything runs inline.
- **WASI** (`wasm32-wasip1`): path-based `File::open`/`create` work against
  preopened directories. WASI has no mmap, so files are read into memory.
- **SIMD**: build with `RUSTFLAGS="-C target-feature=+simd128"` to get the
  wasm vector kernels (supported by all current browsers, Node and wasmtime).
- **Threads** (`wasm32-wasip1-threads`, `-C target-feature=+atomics`): chunk
  work runs on the persistent pool. Whether threads exist is found out at
  run time (a host that refuses the first spawn gets inline execution), and
  the crate's locks are `std::sync` on wasm, which blocks correctly there
  (`parking_lot` cannot park without `cfg(target_feature = "atomics")`).
  The default count is 1 (WASI reports no cores), so set
  `HDF5_NO_C_THREADS`. In the browser, plug Web Workers in with
  `set_thread_spawner`.

```bash
# browser target, under Node (wasm-bindgen-test-runner must match the
# lockfile's wasm-bindgen version)
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
  cargo test -p hdf5-no-c --target wasm32-unknown-unknown --test wasm
# the whole test suite on WASI, under Node's WASI
CARGO_TARGET_WASM32_WASIP1_RUNNER="node $PWD/interop/wasi_run.mjs" \
  cargo test -p hdf5-no-c --target wasm32-wasip1 --features zlib,szip
```

## What works

- **Files**: create / open (read-only, read-write, append, exclusive), flush on
  drop, userblocks (read). Unsupported objects inside a file are isolated
  per-object (the rest of the file stays readable), like libhdf5's
  open-on-demand behavior.
- **Groups**: nested creation, soft/hard links, **external links** (written as
  new-style compact link-message groups; followed transparently across files
  on read, with per-file caching), relink/unlink, iteration (by name or
  creation order), member listing.
- **Datasets**: scalar and N-dimensional (row-major, via `ndarray`),
  contiguous and chunked layouts, resizable datasets (`maxdims`/unlimited),
  hyperslab and point selections (`read_slice`/`write_slice` with `s![..]`),
  fill values, `ByteReader` (`std::io::Read + Seek`).
- **Filters**: DEFLATE/zlib (pure-Rust `miniz_oxide`), shuffle, fletcher32,
  **LZF** (pure-Rust liblzf port, bidirectional with h5py's built-in filter),
  and **blosc** (pure-Rust blosc1 frames: blosclz/LZ4/snappy/zlib encode +
  decode, zstd decode; verified against `hdf5plugin`). **ScaleOffset** and
  **N-bit** chunks decode (with nbit sign re-extension matching libhdf5's
  conversion). Per-chunk *filter masks* are honored, so chunks stored raw by
  optional filters read correctly.
- **Datatypes**: all fixed-width integers/floats, `bool`, enums, compounds
  (arbitrary nesting, name-matched field conversion), tuples, fixed arrays,
  fixed/variable-length ASCII & UTF-8 strings, variable-length arrays.
  **Big-endian data is byte-swapped transparently on read** (including inside
  compounds, arrays, vlen payloads, enum values, attributes and fill values).
  Numeric conversions replicate libhdf5's hard-conversion semantics exactly:
  integer narrowing **saturates**, negative→unsigned clamps to 0, float→int
  saturates with NaN→0 (verified against `H5Tconv` sources).
- **Attributes**: scalar/array attributes of all supported types on any
  object, including variable-length string attributes.
- **Object metadata**: modification times are written (message `0x0012`),
  parsed, and exposed via `loc_info()`.

### On-disk format

The writer emits the maximally-compatible **"earliest" format**: superblock
v0, v1 object headers, old-style (symbol-table) groups with v1 B-trees and
local heaps, v1 chunk B-trees, and global-heap collections for vlen data —
the same structures h5py writes with `libver="earliest"`, readable by every
HDF5 library since 1.8. Groups containing external links are written as
new-style compact link-message groups (the libhdf5 1.8 representation).

The reader goes much further — transcribed one-to-one from the libhdf5 C
sources (`H5B2cache.c`, `H5HFcache.c`, `H5EAcache.c`, …) and verified against
libhdf5 2.0 output:

- superblock v0–v3; object headers v1 and v2 (with continuation blocks)
- symbol-table, compact (link-message) **and dense (fractal-heap) groups**
- compact **and dense attributes**
- dataspace v1/v2; layout messages v1–v5
- chunk indexes: v1 B-tree, Single-Chunk, Implicit, Fixed Array,
  **Extensible Array** (unlimited datasets) and **v2 B-tree** (multiple
  unlimited dimensions), filtered and unfiltered
- v2 B-trees (multi-level), fractal heaps (managed + tiny + **huge** objects,
  nested indirect blocks), extensible arrays (index/super/data blocks, paged)
- **shared object-header messages**: committed (named) datatype references —
  the form h5py emits for `f["name"] = np.dtype(...)` typed datasets — and
  full **SOHM tables** (superblock extension → Shared Message Table message →
  `SMTB` master table → per-index fractal heaps), resolving shared datatypes,
  dataspaces, fill values and attributes; both list- and B-tree-indexed files
- **virtual datasets** (layout class 3): global-heap mapping blobs and
  serialized hyperslab/point/all selections decode; source datasets are read
  from the same or external files and scattered per mapping

This covers files produced by h5py/libhdf5 with `libver="latest"` through
HDF5 2.0, including dense storage and resizable chunked datasets.

## Parity proof

- **Real-world corpus** (`interop/make_corpus.py`, generated in Docker with
  netCDF4 / PyTables / pandas / h5py): dimension-scale references,
  bitfield bool columns, HDFStore fixed+table layouts, object & region
  references, MATLAB-style userblocks, tracked creation order — every
  object and attribute reads (`examples/corpus.rs`), with value spot-checks.
- **Mechanical API diff** (`interop/api_diff.sh`) against the FFI crate:
  0 missing items in `hdf5-types`/`hdf5-derive`; 1 nominal difference in
  `hdf5` (our `Maybe` is an enum, theirs a struct).
- **Fuzzed parser**: 35,000 mutated files (`examples/fuzz.rs`), zero panics
  or allocation aborts; all size fields are overflow-checked and capped.
- Userblocks now round-trip (write + content preservation), and
  `H5Pset_external`-style external storage is a checked error instead of
  silently ignored.

## Interop validation

`interop/check_h5py.py` + `hdf5/examples/interop_harness.rs` form a
bidirectional harness:

SOHM reference files are generated by `interop/make_sohm.py`, which calls
the libhdf5 C API via ctypes (h5py exposes no SOHM creation API).

```bash
# h5py -> Rust
python3 interop/check_h5py.py write /tmp/interop
cargo run --example interop_harness -- read /tmp/interop
# Rust -> h5py
cargo run --example interop_harness -- write /tmp/interop
python3 interop/check_h5py.py verify /tmp/interop
```

## Differences from the FFI crate

- All data *written* is little-endian (the native order everywhere libhdf5
  runs today); big-endian is read-only.
- Attributes larger than the 64 KB compact limit are stored **densely**
  (fractal heap + v2 btree written by this crate, with a v2 object header) —
  matching libhdf5's 1.8+ behavior; there is no practical attribute size limit.
- `Object::id()` returns synthetic identifiers (there is no C id registry);
  `from_id` is only kept for source compatibility.
- `Dataset::offset()` returns `None` (addresses are assigned at serialization).
- Error *messages* differ from libhdf5's error-stack text; `ErrorStack` exists
  for API compatibility but carries no C frames.
- Writable files are flushed incrementally (see *Streaming writes*); dead
  metadata from earlier flushes is not reclaimed. In-memory files
  (`create_in_memory`, `from_bytes_rw`) are serialized whole by `to_bytes`.
- Objects reached through external links are read-only (libhdf5 allows
  writes through them).
- Non-contiguous `ndarray` views must be standardized by the caller before
  writing (matching the FFI crate's standard-layout requirement).
- Blosc/zstd is *written* with stored (uncompressed) streams inside a fully
  valid zstd-tagged frame — no pure-Rust zstd encoder exists; reading real
  zstd streams works. All other blosc codecs compress normally.
- The `mpi` feature (below) is a pure-Rust MPI *subset* for SPMD workflows
  between Rust processes; it is **not** wire-compatible with OpenMPI/MPICH
  and cannot join a C MPI application's communicator. `globals` exposes the
  full set of constant names (synthetic identifiers).

## Also included

- **MPI subset** (cargo feature `mpi`): a pure-Rust mini-MPI
  (`hdf5::mpi::Comm` — TCP rendezvous, `rank`/`size`, `barrier`, `bcast`,
  `gather`/`allgather`, `allreduce`, plus a `spawn_workers` mini-`mpiexec`)
  with **collective HDF5 files** mirroring parallel HDF5's rules: all ranks
  perform identical metadata calls, each rank writes its own dataset
  selections, and `File::close` is collective — rank 0 gathers every rank's
  write log and produces the single physical file (single-aggregator
  two-phase I/O). Vlen writes are rejected in MPI mode, matching real
  parallel HDF5. See `examples/mpi_demo.rs` (N processes, one shared file,
  verified with h5py). With the `mpi-rs` feature, `Comm` runs on the
  [mpi-rs](https://crates.io/crates/mpi-rs) crate instead — a full pure-Rust
  MPI with an rsmpi-compatible API and a real `mpiexec` launcher
  (`Comm::init` picks the transport automatically; see
  `examples/mpi_rs_demo.rs`).
- **SZip** (cargo feature `szip`): a pure-Rust extended-Rice (CCSDS 121.0)
  codec transcribed from libaec — encode *and* decode, EC/NN coding, 8/16-bit
  samples plus byte-interleaved 32/64-bit, scanline padding, zero-block/ROS
  and second-extension low-entropy options. Validated bidirectionally against
  the real `libsz` (Homebrew libaec) at both the codec level
  (`interop/check_szip.py` + `examples/szval.rs`) and the file level: libsz
  decodes every chunk of a Rust-written `.h5`, including the `H5Zszip`
  cd_values and 4-byte size framing.
- **SOHM writing**: configure `fcpl().shared_mesg_indexes(...)` and identical
  datatype/dataspace/fill messages are deduplicated into shared-message
  tables (superblock v2 + extension + `SMTB`/`SMLI` + fractal heap), readable
  by h5py.
- **VDS writing**: `DatasetCreateBuilder::virtual_map(...)` emits layout-v4
  virtual datasets with serialized selections; h5py resolves them
  (`is_virtual == True`). Files containing VDS round-trip through rewrite.
- **ScaleOffset and N-bit encoding** (libhdf5-verified chunk formats),
  **bit-shuffle** in blosc (both directions), **dense-link writing** for
  large link-message groups, **filtered fractal heap** reading.
- **Memory-mapped, metadata-only opens**: `File::open` maps the file; both
  contiguous *and chunked/filtered* datasets stay as lazy references (the
  chunk list is collected at parse, decoding happens on first access) —
  opening huge files costs metadata only, and untouched datasets never enter
  memory or run their filters.

## License

MIT OR Apache-2.0, same as the reference implementation.

## Release checklist status

- `cargo test` — 69 tests (default), 36 (`-p hdf5 --features "szip mpi"`)
- `cargo clippy --workspace --all-features` — zero warnings
- `cargo doc --workspace --all-features` — zero warnings
- bidirectional h5py harness, real-world corpus, 35k-mutation fuzz — green
