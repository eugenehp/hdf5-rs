//! Pure-Rust implementation of HDF5.
//!
//! This crate provides a drop-in replacement for the FFI-based `hdf5` crate
//! (`hdf5-rust`), implemented entirely in Rust — no C library required. Files
//! it writes use the widely-compatible "earliest" HDF5 format (superblock v0,
//! v1 object headers, symbol-table groups, v1 B-trees) and can be opened by
//! libhdf5, h5py and other HDF5 consumers; it reads files written by those
//! libraries as well (object header v1/v2, superblock v0-v3).
//!
//! Some of the features include:
//!
//! - Native representation of most HDF5 types, including variable-length
//!   strings and arrays, compounds, enums and fixed arrays.
//! - Derive-macro for automatic mapping of user structs and enums to `HDF5`
//!   types.
//! - Multi-dimensional array reading/writing interface via `ndarray`,
//!   including hyperslab/point selections and resizable datasets.
//! - Pure-Rust filters: DEFLATE (zlib), shuffle, fletcher32, LZF, blosc
//!   (blosclz/LZ4/snappy/zlib + bit-shuffle), scale-offset and n-bit;
//!   SZip behind the `szip` feature.
//! - Dense and compact groups/attributes, SOHM (shared message) tables,
//!   virtual datasets, external links and committed datatypes — read and
//!   write.
//! - Memory-mapped, lazily materialized reads: opening a file costs its
//!   metadata only.
//!
//! # Cargo features
//!
//! - `zlib` — kept for FFI-crate compatibility (DEFLATE is always built in).
//! - `szip` — pure-Rust SZip (extended-Rice) codec, validated against libaec.
//! - `mpi` — SPMD collective files over a built-in TCP mini-MPI
//!   ([`mpi::Comm`]); not wire-compatible with OpenMPI/MPICH.
//! - `mpi-rs` — back [`mpi::Comm`] with the pure-Rust
//!   [`mpi-rs`](https://crates.io/crates/mpi-rs) crate (rsmpi-compatible
//!   API): launch ranks with its `mpiexec`, adopt an app-initialized
//!   universe, or run as a singleton. Implies `mpi`.
//! - `rlx` — [`Container::read_tensor`](crate::Container::read_tensor):
//!   load datasets/attributes directly as `rlx` tensors, compose ops
//!   lazily, and run them on RLX's bundled cpu backend (`to_vec`).
//! - `complex` — complex-number datatypes via `num-complex`.
//! - `f16` — half-precision floats via `half`.

#![allow(clippy::missing_safety_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::return_self_not_must_use)]
#![cfg_attr(docsrs, feature(doc_cfg))]

mod export {
    pub use crate::{
        class::from_id,
        dim::{Dimension, Ix},
        error::{silence_errors, Error, ErrorFrame, ErrorStack, ExpandedErrorStack, Result},
        hl::extents::{Extent, Extents, SimpleExtents},
        hl::selection::{Hyperslab, Selection, SliceOrIndex},
        hl::{
            Attribute, AttributeBuilder, AttributeBuilderData, AttributeBuilderEmpty,
            AttributeBuilderEmptyShape, ByteReader, Container, Conversion, Dataset, DatasetBuilder,
            DatasetBuilderData, DatasetBuilderEmpty, DatasetBuilderEmptyShape, Dataspace, Datatype,
            File, FileBuilder, Group, LinkInfo, LinkType, Location, LocationInfo, LocationToken,
            LocationType, Object, OpenMode, PropertyList, Reader, Writer,
        },
    };

    pub use crate::dim::slice_as_shape;
    #[doc(hidden)]
    pub use crate::error::{h5check, is_err_code, H5ErrorCode};
    pub use crate::hl::plist::set_vlen_manager_libc;
    pub use crate::macros::H5Get;
    pub use crate::sync::{lock_part1, lock_part2};

    pub use hdf5_derive::H5Type;
    pub use hdf5_types::H5Type;

    pub mod types {
        pub use hdf5_types::*;
    }

    pub mod dataset {
        pub use crate::hl::chunks::{ChunkInfo, ChunkInfoRef};
        pub use crate::hl::dataset::{
            Chunk, Dataset, DatasetBuilder, Maybe, DEFAULT_CHUNK_SIZE_KB,
        };
        pub use crate::hl::plist::dataset_access::*;
        pub use crate::hl::plist::dataset_create::*;
    }

    pub mod datatype {
        pub use crate::hl::datatype::{ByteOrder, Conversion, Datatype};
    }

    pub mod file {
        pub use crate::hl::file::{File, FileBuilder, OpenMode};
        pub use crate::hl::plist::file_access::*;
        pub use crate::hl::plist::file_create::*;
    }

    pub mod plist {
        pub use crate::hl::plist::dataset_access::{DatasetAccess, DatasetAccessBuilder};
        pub use crate::hl::plist::dataset_create::{DatasetCreate, DatasetCreateBuilder};
        pub use crate::hl::plist::file_access::{FileAccess, FileAccessBuilder};
        pub use crate::hl::plist::file_create::{FileCreate, FileCreateBuilder};
        pub use crate::hl::plist::link_create::{LinkCreate, LinkCreateBuilder};
        pub use crate::hl::plist::{PropertyList, PropertyListClass};

        pub mod dataset_access {
            pub use crate::hl::plist::dataset_access::*;
        }
        pub mod dataset_create {
            pub use crate::hl::plist::dataset_create::*;
        }
        pub mod file_access {
            pub use crate::hl::plist::file_access::*;
        }
        pub mod file_create {
            pub use crate::hl::plist::file_create::*;
        }
        pub mod link_create {
            pub use crate::hl::plist::link_create::*;
        }
    }

    pub mod filters {
        pub use crate::hl::filters::*;
    }
}

pub use crate::export::*;

#[macro_use]
mod macros;
#[macro_use]
mod class;

mod dim;
mod error;
#[doc(hidden)]
pub mod globals;
mod handle;
#[doc(hidden)]
pub mod sync;
mod util;

#[doc(hidden)]
pub mod format;
#[doc(hidden)]
pub mod h5i;
mod model;

mod hl;
#[cfg(feature = "mpi")]
pub mod mpi;

/// Returns the runtime version of the (emulated) HDF5 library.
///
/// The pure-Rust engine reports 1.10.x: the on-disk structures it writes are
/// the "earliest" set, fully readable by any library of version 1.8+.
pub fn library_version() -> (u8, u8, u8) {
    (1, 10, 0)
}

/// Returns true if the library is threadsafe.
///
/// The pure-Rust implementation is always threadsafe (per-file RwLocks).
pub fn is_library_threadsafe() -> bool {
    true
}

/// Validation hook (examples only): raw SZ_BufftoBuffCompress equivalent.
#[cfg(feature = "szip")]
pub fn internal_szip_compress(cd: &[u32], data: &[u8]) -> Result<Vec<u8>> {
    format::szip::sz_compress(cd, data)
}

/// Validation hook (examples only): raw SZ_BufftoBuffDecompress equivalent.
#[cfg(feature = "szip")]
pub fn internal_szip_decompress(cd: &[u32], data: &[u8], out_len: usize) -> Result<Vec<u8>> {
    format::szip::sz_decompress(cd, data, out_len)
}

#[cfg(test)]
pub mod tests {
    use crate::library_version;

    #[test]
    pub fn test_library_version() {
        assert!(library_version() >= (1, 8, 4));
    }
}
