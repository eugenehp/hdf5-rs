//! Pure-Rust HDF5 on-disk format: serialization and parsing.
//!
//! The engine targets the universally-compatible "earliest" format written by
//! default by libhdf5/h5py: superblock v0, object header v1, symbol-table
//! (old-style) groups backed by v1 B-trees and local heaps, contiguous and
//! chunked data layouts, a global heap for variable-length data, and the
//! DEFLATE/shuffle/fletcher32 filter pipeline. Files written here open in
//! h5py and the C library, and files those write are parsed back here.

pub mod blosc;
pub mod checksum;
pub mod convert;
pub mod datatype;
pub mod filters;
pub mod reader;
#[cfg(feature = "szip")]
pub mod szip;
pub mod v2;
pub mod vds;
pub mod writer;

use crate::error::Result;
use crate::model::FileState;

/// HDF5 superblock signature.
pub const SIGNATURE: [u8; 8] = [0x89, b'H', b'D', b'F', 0x0d, 0x0a, 0x1a, 0x0a];

/// Undefined 8-byte address / length.
pub const UNDEF: u64 = u64::MAX;

/// Size of file offsets and lengths (bytes) used by this writer.
pub const SIZEOF_ADDR: usize = 8;

/// Default group leaf-node K (max 2K symbols per SNOD).
pub const GROUP_LEAF_K: u16 = 4;
/// Default group internal-node K (B-tree fan-out for group nodes).
pub const GROUP_INTERNAL_K: u16 = 16;
/// Default chunk B-tree K (istore_k).
pub const CHUNK_K: u16 = 32;

/// Round `n` up to the next multiple of 8.
#[inline]
pub const fn align8(n: usize) -> usize {
    (n + 7) & !7
}

/// Round `n` up to the next multiple of `a` (a must be a power of two).
#[inline]
pub const fn align_up(n: u64, a: u64) -> u64 {
    (n + a - 1) & !(a - 1)
}

/// A little-endian byte-writer that tracks the current length and supports
/// back-patching previously reserved regions.
#[derive(Default)]
pub struct Buf {
    pub bytes: Vec<u8>,
}

impl Buf {
    pub fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn u8(&mut self, v: u8) {
        self.bytes.push(v);
    }

    pub fn u16(&mut self, v: u16) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }

    pub fn u32(&mut self, v: u32) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }

    pub fn u64(&mut self, v: u64) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }

    pub fn addr(&mut self, v: u64) {
        self.u64(v);
    }

    pub fn raw(&mut self, s: &[u8]) {
        self.bytes.extend_from_slice(s);
    }

    pub fn zeros(&mut self, n: usize) {
        self.bytes.resize(self.bytes.len() + n, 0);
    }

    /// Pad with zeros to the next multiple of 8.
    pub fn pad8(&mut self) {
        let target = align8(self.bytes.len());
        self.zeros(target - self.bytes.len());
    }

    pub fn patch_u16(&mut self, at: usize, v: u16) {
        self.bytes[at..at + 2].copy_from_slice(&v.to_le_bytes());
    }

    pub fn patch_u32(&mut self, at: usize, v: u32) {
        self.bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }

    pub fn patch_u64(&mut self, at: usize, v: u64) {
        self.bytes[at..at + 8].copy_from_slice(&v.to_le_bytes());
    }
}

/// A little-endian byte-reader over an in-memory file image.
pub struct Cursor<'a> {
    pub data: &'a [u8],
    pub pos: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub fn at(data: &'a [u8], pos: usize) -> Self {
        Self { data, pos }
    }

    pub fn seek(&mut self, pos: usize) {
        self.pos = pos;
    }

    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    pub fn eof(&self) -> bool {
        self.pos >= self.data.len()
    }

    pub fn u8(&mut self) -> Result<u8> {
        let b = *self.data.get(self.pos).ok_or("unexpected end of file")?;
        self.pos += 1;
        Ok(b)
    }

    pub fn u16(&mut self) -> Result<u16> {
        let s = self.take(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }

    pub fn u32(&mut self) -> Result<u32> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    pub fn u64(&mut self) -> Result<u64> {
        let s = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(s);
        Ok(u64::from_le_bytes(a))
    }

    pub fn addr(&mut self) -> Result<u64> {
        self.u64()
    }

    /// Read a length-sized integer of `size` bytes (little-endian).
    pub fn uint(&mut self, size: usize) -> Result<u64> {
        let s = self.take(size)?;
        let mut a = [0u8; 8];
        a[..size].copy_from_slice(s);
        Ok(u64::from_le_bytes(a))
    }

    pub fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            return Err("unexpected end of file".into());
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    pub fn skip(&mut self, n: usize) {
        self.pos += n;
    }

    /// Align current position up to a multiple of 8.
    pub fn align8(&mut self) {
        self.pos = align8(self.pos);
    }
}

/// Serialize an in-memory file model into a complete HDF5 byte image.
pub fn serialize(state: &FileState) -> Result<Vec<u8>> {
    writer::serialize(state)
}

/// Parse a complete HDF5 byte image into an in-memory file model.
pub fn parse(data: &[u8]) -> Result<FileState> {
    let image = std::sync::Arc::new(crate::model::FileImage::Bytes(data.to_vec()));
    reader::parse(&image, None)
}

/// Parse with a base directory for resolving virtual-dataset source files.
pub fn parse_at(data: &[u8], dir: Option<&std::path::Path>) -> Result<FileState> {
    let image = std::sync::Arc::new(crate::model::FileImage::Bytes(data.to_vec()));
    reader::parse(&image, dir)
}

/// Parse an image (memory-mapped or owned); large contiguous datasets stay
/// lazily referenced into the image instead of being copied up front.
pub fn parse_image(
    image: &std::sync::Arc<crate::model::FileImage>,
    dir: Option<&std::path::Path>,
) -> Result<FileState> {
    reader::parse(image, dir)
}
