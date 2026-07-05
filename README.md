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
| `rlx` | `Container::read_tensor()` — load any numeric dataset/attribute directly as an [RLX](https://crates.io/crates/rlx) tensor (`rlx@0.2.10`). Ops compose lazily, fuse, and run on RLX's bundled cpu backend — HDF5 checkpoints feed straight into inference (see `examples/rlx_mlp.rs`) |
| `complex` | Complex-number datatypes (`num-complex`) |
| `f16` | Half-precision floats (`half`) |

MSRV: 1.85. No build scripts, no C toolchain, no `HDF5_DIR` probing.

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
- The file is held in memory while open and serialized on flush/close/drop —
  ideal for small/medium files; not yet suited to files larger than RAM.
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
  verified with h5py).
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
  memory or run their filters. (Writes still assemble the file in memory.)

## License

MIT OR Apache-2.0, same as the reference implementation.

## Release checklist status

- `cargo test` — 69 tests (default), 36 (`-p hdf5 --features "szip mpi"`)
- `cargo clippy --workspace --all-features` — zero warnings
- `cargo doc --workspace --all-features` — zero warnings
- bidirectional h5py harness, real-world corpus, 35k-mutation fuzz — green
