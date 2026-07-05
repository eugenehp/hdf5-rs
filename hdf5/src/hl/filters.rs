//! Dataset filters (compression, checksums, etc.).
//!
//! The public surface mirrors the FFI crate. In this pure-Rust build the
//! supported filters are DEFLATE (`zlib`, via a pure-Rust backend, always
//! compiled in), `shuffle` and `fletcher32`. SZip/NBit/ScaleOffset variants
//! exist for API parity and can be introspected, but attempting to write with
//! them returns an error.

use crate::error::Result;
use crate::format::filters::{
    RawFilter, FILTER_BLOSC, FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_LZF, FILTER_NBIT,
    FILTER_SCALEOFFSET, FILTER_SHUFFLE,
};

/// An HDF5 filter identifier.
#[allow(non_camel_case_types)]
pub type H5Z_filter_t = i32;

pub const H5Z_FILTER_DEFLATE: H5Z_filter_t = FILTER_DEFLATE as _;
pub const H5Z_FILTER_SHUFFLE: H5Z_filter_t = FILTER_SHUFFLE as _;
pub const H5Z_FILTER_FLETCHER32: H5Z_filter_t = FILTER_FLETCHER32 as _;
pub const H5Z_FILTER_SZIP: H5Z_filter_t = 4;
pub const H5Z_FILTER_NBIT: H5Z_filter_t = FILTER_NBIT as _;
pub const H5Z_FILTER_SCALEOFFSET: H5Z_filter_t = FILTER_SCALEOFFSET as _;
pub const H5Z_FILTER_LZF: H5Z_filter_t = FILTER_LZF as _;
pub const H5Z_FILTER_BLOSC: H5Z_filter_t = FILTER_BLOSC as _;

/// SZip coding method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SZip {
    Entropy,
    NearestNeighbor,
}

/// Scale-offset filter mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleOffset {
    Integer(u16),
    FloatDScale(u8),
}

/// Blosc sub-compressor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Blosc {
    BloscLZ,
    LZ4,
    LZ4HC,
    Snappy,
    ZLib,
    ZStd,
}

/// Blosc shuffle mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BloscShuffle {
    None,
    #[default]
    Byte,
    Bit,
}

impl From<bool> for BloscShuffle {
    fn from(b: bool) -> Self {
        if b {
            Self::Byte
        } else {
            Self::None
        }
    }
}

/// A single dataset filter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Filter {
    Deflate(u8),
    Shuffle,
    Fletcher32,
    SZip(SZip, u8),
    NBit,
    ScaleOffset(ScaleOffset),
    LZF,
    Blosc(Blosc, u8, BloscShuffle),
    User(H5Z_filter_t, Vec<u32>),
}

/// Availability/encode/decode info for a filter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct FilterInfo {
    pub is_available: bool,
    pub encode_enabled: bool,
    pub decode_enabled: bool,
}

impl Filter {
    pub fn id(&self) -> H5Z_filter_t {
        match self {
            Self::Deflate(_) => H5Z_FILTER_DEFLATE,
            Self::Shuffle => H5Z_FILTER_SHUFFLE,
            Self::Fletcher32 => H5Z_FILTER_FLETCHER32,
            Self::SZip(..) => H5Z_FILTER_SZIP,
            Self::NBit => H5Z_FILTER_NBIT,
            Self::ScaleOffset(_) => H5Z_FILTER_SCALEOFFSET,
            Self::LZF => H5Z_FILTER_LZF,
            Self::Blosc(..) => H5Z_FILTER_BLOSC,
            Self::User(id, _) => *id,
        }
    }

    pub fn get_info(filter_id: H5Z_filter_t) -> FilterInfo {
        let supported = matches!(
            filter_id,
            H5Z_FILTER_DEFLATE
                | H5Z_FILTER_SHUFFLE
                | H5Z_FILTER_FLETCHER32
                | H5Z_FILTER_LZF
                | H5Z_FILTER_BLOSC
        );
        FilterInfo {
            is_available: supported,
            encode_enabled: supported,
            decode_enabled: supported,
        }
    }

    pub fn is_available(&self) -> bool {
        Self::get_info(self.id()).is_available
    }

    pub fn encode_enabled(&self) -> bool {
        Self::get_info(self.id()).encode_enabled
    }

    pub fn decode_enabled(&self) -> bool {
        Self::get_info(self.id()).decode_enabled
    }

    pub fn deflate(level: u8) -> Self {
        Self::Deflate(level)
    }

    pub fn shuffle() -> Self {
        Self::Shuffle
    }

    pub fn fletcher32() -> Self {
        Self::Fletcher32
    }

    pub fn szip(coding: SZip, px_per_block: u8) -> Self {
        Self::SZip(coding, px_per_block)
    }

    pub fn nbit() -> Self {
        Self::NBit
    }

    pub fn scale_offset(mode: ScaleOffset) -> Self {
        Self::ScaleOffset(mode)
    }

    pub fn lzf() -> Self {
        Self::LZF
    }

    pub fn blosc<T>(complib: Blosc, clevel: u8, shuffle: T) -> Self
    where
        T: Into<BloscShuffle>,
    {
        Self::Blosc(complib, clevel, shuffle.into())
    }

    pub fn blosc_blosclz<T: Into<BloscShuffle>>(clevel: u8, shuffle: T) -> Self {
        Self::blosc(Blosc::BloscLZ, clevel, shuffle)
    }

    pub fn blosc_lz4<T: Into<BloscShuffle>>(clevel: u8, shuffle: T) -> Self {
        Self::blosc(Blosc::LZ4, clevel, shuffle)
    }

    pub fn blosc_lz4hc<T: Into<BloscShuffle>>(clevel: u8, shuffle: T) -> Self {
        Self::blosc(Blosc::LZ4HC, clevel, shuffle)
    }

    pub fn blosc_snappy<T: Into<BloscShuffle>>(clevel: u8, shuffle: T) -> Self {
        Self::blosc(Blosc::Snappy, clevel, shuffle)
    }

    pub fn blosc_zlib<T: Into<BloscShuffle>>(clevel: u8, shuffle: T) -> Self {
        Self::blosc(Blosc::ZLib, clevel, shuffle)
    }

    pub fn blosc_zstd<T: Into<BloscShuffle>>(clevel: u8, shuffle: T) -> Self {
        Self::blosc(Blosc::ZStd, clevel, shuffle)
    }

    pub fn user(id: H5Z_filter_t, cdata: &[u32]) -> Self {
        Self::User(id, cdata.to_vec())
    }

    pub fn from_raw(filter_id: H5Z_filter_t, cdata: &[u32]) -> Result<Self> {
        Self::from_raw_parts(filter_id as u16, cdata)
            .ok_or_else(|| format!("unknown filter id: {filter_id}").into())
    }

    /// Interpret a raw (id, client data) pair, returning `None` for ids that
    /// don't map onto a known variant.
    pub(crate) fn from_raw_parts(id: u16, cdata: &[u32]) -> Option<Self> {
        match id as i32 {
            H5Z_FILTER_DEFLATE => Some(Self::Deflate(cdata.first().copied().unwrap_or(6) as u8)),
            H5Z_FILTER_SHUFFLE => Some(Self::Shuffle),
            H5Z_FILTER_FLETCHER32 => Some(Self::Fletcher32),
            H5Z_FILTER_SZIP => {
                let mask = cdata.first().copied().unwrap_or(0);
                let px = cdata.get(1).copied().unwrap_or(0) as u8;
                let coding = if mask & 0x20 != 0 {
                    SZip::NearestNeighbor
                } else {
                    SZip::Entropy
                };
                Some(Self::SZip(coding, px))
            }
            H5Z_FILTER_NBIT => Some(Self::NBit),
            H5Z_FILTER_LZF => Some(Self::LZF),
            H5Z_FILTER_BLOSC => {
                let clevel = cdata.get(4).copied().unwrap_or(5) as u8;
                let shuffle = match cdata.get(5).copied().unwrap_or(1) {
                    0 => BloscShuffle::None,
                    2 => BloscShuffle::Bit,
                    _ => BloscShuffle::Byte,
                };
                let complib = match cdata.get(6).copied().unwrap_or(0) {
                    1 => Blosc::LZ4,
                    2 => Blosc::LZ4HC,
                    3 => Blosc::Snappy,
                    4 => Blosc::ZLib,
                    5 => Blosc::ZStd,
                    _ => Blosc::BloscLZ,
                };
                Some(Self::Blosc(complib, clevel, shuffle))
            }
            H5Z_FILTER_SCALEOFFSET => {
                // H5Z_SO_FLOAT_DSCALE = 0, H5Z_SO_FLOAT_ESCALE = 1, H5Z_SO_INT = 2
                let mode = cdata.first().copied().unwrap_or(0);
                let factor = cdata.get(1).copied().unwrap_or(0);
                Some(Self::ScaleOffset(if mode == 2 {
                    ScaleOffset::Integer(factor as u16)
                } else {
                    ScaleOffset::FloatDScale(factor as u8)
                }))
            }
            other => Some(Self::User(other, cdata.to_vec())),
        }
    }

    /// Lower to raw (id, client data, canonical name) form for the pipeline.
    pub(crate) fn to_raw(&self) -> RawFilter {
        match self {
            Self::Deflate(level) => RawFilter {
                id: FILTER_DEFLATE,
                cdata: vec![u32::from(*level)],
                name: "deflate".into(),
            },
            Self::Shuffle => RawFilter {
                id: FILTER_SHUFFLE,
                cdata: vec![],
                name: "shuffle".into(),
            },
            Self::Fletcher32 => RawFilter {
                id: FILTER_FLETCHER32,
                cdata: vec![],
                name: "fletcher32".into(),
            },
            Self::SZip(coding, px) => RawFilter {
                id: 4,
                cdata: vec![
                    match coding {
                        SZip::Entropy => 0x04,
                        SZip::NearestNeighbor => 0x20,
                    },
                    u32::from(*px),
                ],
                name: "szip".into(),
            },
            Self::NBit => RawFilter {
                id: FILTER_NBIT,
                cdata: vec![],
                name: "nbit".into(),
            },
            // client data mirrors h5py's lzf_set_local: {filter version,
            // liblzf version, chunk size (filled in by the writer)}
            Self::LZF => RawFilter {
                id: FILTER_LZF,
                cdata: vec![4, 0x0105, 0],
                name: "lzf".into(),
            },
            // blosc_filter.c set_local layout; typesize/chunksize are filled
            // in by the writer
            Self::Blosc(complib, clevel, shuffle) => RawFilter {
                id: FILTER_BLOSC,
                cdata: vec![
                    2, // filter revision
                    2, // blosc format version
                    0, // typesize (writer fills)
                    0, // chunk size (writer fills)
                    u32::from(*clevel),
                    match shuffle {
                        BloscShuffle::None => 0,
                        BloscShuffle::Byte => 1,
                        BloscShuffle::Bit => 2,
                    },
                    match complib {
                        Blosc::BloscLZ => 0,
                        Blosc::LZ4 => 1,
                        Blosc::LZ4HC => 2,
                        Blosc::Snappy => 3,
                        Blosc::ZLib => 4,
                        Blosc::ZStd => 5,
                    },
                ],
                name: "blosc".into(),
            },
            Self::ScaleOffset(mode) => RawFilter {
                id: FILTER_SCALEOFFSET,
                cdata: match mode {
                    ScaleOffset::Integer(f) => vec![2, u32::from(*f)],
                    ScaleOffset::FloatDScale(f) => vec![0, u32::from(*f)],
                },
                name: "scaleoffset".into(),
            },
            Self::User(id, cdata) => RawFilter {
                id: *id as u16,
                cdata: cdata.clone(),
                name: String::new(),
            },
        }
    }

    /// Validate that this filter can actually be applied by this build.
    pub(crate) fn validate_writable(&self) -> Result<()> {
        match self {
            Self::Deflate(l) if *l > 9 => Err("deflate level must be <= 9".into()),
            Self::Deflate(_) | Self::Shuffle | Self::Fletcher32 | Self::LZF => Ok(()),
            // zstd streams are stored uncompressed inside the frame (no
            // pure-Rust zstd encoder exists); the frame stays fully valid
            Self::Blosc(..) => Ok(()),
            Self::NBit => Ok(()),
            Self::ScaleOffset(_) => Ok(()),
            #[cfg(feature = "szip")]
            Self::SZip(_, ppb) => {
                if *ppb == 0 || *ppb % 2 == 1 || *ppb > 32 {
                    Err("szip pixels-per-block must be even and <= 32".into())
                } else {
                    Ok(())
                }
            }
            #[cfg(not(feature = "szip"))]
            Self::SZip(..) => Err("szip requires the `szip` cargo feature".into()),
            other => {
                Err(format!("filter {other:?} is not supported by the pure-Rust engine").into())
            }
        }
    }
}

/// Returns `true` if the gzip/DEFLATE filter is available.
pub fn deflate_available() -> bool {
    true
}

/// Returns `true` if the SZip filter is available (never, in this build).
pub fn szip_available() -> bool {
    cfg!(feature = "szip")
}

/// Returns `true` if the LZF filter is available.
pub fn lzf_available() -> bool {
    true
}

/// Returns `true` if the blosc filter is available.
pub fn blosc_available() -> bool {
    true
}

/// Get the number of blosc threads (this build is single-threaded).
pub fn blosc_get_nthreads() -> u8 {
    1
}

/// Set the number of blosc threads (no-op: this build is single-threaded).
pub fn blosc_set_nthreads(_num_threads: u8) -> u8 {
    1
}

/// LZF filter id (parity constant; the filter is built in).
pub const LZF_FILTER_ID: H5Z_filter_t = 32000;
/// Blosc filter id (parity constant; the filter is built in).
pub const BLOSC_FILTER_ID: H5Z_filter_t = 32001;

/// Returns `true` if the gzip/DEFLATE filter is available (alias).
pub fn gzip_available() -> bool {
    deflate_available()
}

/// No-op: LZF is compiled in (parity with the FFI crate's registration).
pub fn register_lzf() -> Result<(), &'static str> {
    Ok(())
}

/// No-op: blosc is compiled in (parity with the FFI crate's registration).
pub fn register_blosc() -> Result<(), &'static str> {
    Ok(())
}
