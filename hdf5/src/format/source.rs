//! Byte sources for the pure-Rust HDF5 parser.
//!
//! Parsing is random-access: the reader jumps to object-header, B-tree and heap
//! addresses scattered throughout the file. A [`ByteSource`] abstracts *where*
//! those bytes come from, so the same parser runs over either
//!
//! * a fully-present image (mmap or owned `Vec`) — zero-copy, or
//! * a lazy [`Read`] + [`Seek`] reader that fetches and caches 1 MiB blocks on
//!   demand ([`RangeSource`]).
//!
//! The lazy variant lets metadata be read from very large — or remote,
//! range-request-backed — files without materializing the whole thing: only the
//! blocks actually touched (superblock, object headers, B-tree/heap nodes, and
//! any small chunks a lazy dataset reads) are fetched.
//!
//! `slice` returns a reference valid for the lifetime of the source. Every
//! source materializes bytes into storage that is never moved or freed while the
//! source lives (a borrowed slice, an `Arc`-held image, or boxed blocks), so the
//! [`Cursor`](crate::format::Cursor) can hold a returned slice across further
//! reads exactly as it did over a plain `&[u8]`.

use std::io::{Read, Seek, SeekFrom};
use std::sync::Mutex;

use crate::error::Result;
use crate::model::FileImage;

/// Default fetch granularity for [`RangeSource`] (must stay a power of two).
/// 4 MiB: HDF5 metadata (object headers, heaps, B-tree nodes) usually clusters,
/// so a wider block captures more per round-trip. Files written incrementally
/// scatter it instead; see [`RangeSource::with_block`].
pub const DEFAULT_BLOCK: usize = 4 << 20; // 4 MiB

/// Random-access, read-only view of a file's bytes.
pub trait ByteSource {
    /// Bytes `[offset, offset + len)`. Errors if the range runs past the end.
    fn slice(&self, offset: u64, len: usize) -> Result<&[u8]>;
    /// Total length of the underlying file in bytes.
    fn len(&self) -> u64;
    /// The largest run of bytes at hand around `offset`, as `(start, bytes)`
    /// with `start <= offset`: the whole image for in-memory sources, the
    /// cached block for a lazy one. Lets a [`Cursor`](crate::format::Cursor)
    /// read field after field without a call per field.
    fn window(&self, _offset: u64) -> Option<(u64, &[u8])> {
        None
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[inline]
fn end_of(offset: u64, len: usize) -> Result<u64> {
    offset
        .checked_add(len as u64)
        .ok_or_else(|| "length overflow".into())
}

/// `[offset, offset + len)` of a slice held in memory. File offsets are `u64`
/// on every target, so one past what a 32-bit `usize` holds is out of range
/// here rather than wrapped onto other bytes.
#[inline]
fn in_memory(data: &[u8], offset: u64, len: usize) -> Result<&[u8]> {
    let start = usize::try_from(offset).map_err(|_| "unexpected end of file")?;
    let end = start.checked_add(len).ok_or("length overflow")?;
    data.get(start..end)
        .ok_or_else(|| "unexpected end of file".into())
}

/// Zero-copy source over a borrowed, fully-present slice (a message body, or a
/// mmap/owned image).
pub struct MemSource<'a> {
    data: &'a [u8],
}

impl<'a> MemSource<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data }
    }
}

impl ByteSource for MemSource<'_> {
    fn slice(&self, offset: u64, len: usize) -> Result<&[u8]> {
        in_memory(self.data, offset, len)
    }
    fn window(&self, _: u64) -> Option<(u64, &[u8])> {
        Some((0, self.data))
    }
    fn len(&self) -> u64 {
        self.data.len() as u64
    }
}

/// Owned source over a shared file image (mmap or bytes). Held by lazy datasets
/// so the image outlives the parse.
pub struct ImageSource {
    image: std::sync::Arc<FileImage>,
}

impl ImageSource {
    pub fn new(image: std::sync::Arc<FileImage>) -> Self {
        Self { image }
    }
}

impl ByteSource for ImageSource {
    fn slice(&self, offset: u64, len: usize) -> Result<&[u8]> {
        in_memory(&self.image, offset, len)
    }
    fn window(&self, _: u64) -> Option<(u64, &[u8])> {
        Some((0, &self.image))
    }
    fn len(&self) -> u64 {
        self.image.len() as u64
    }
}

/// Lazy source backed by a `Read + Seek` reader.
///
/// Fetches BLOCK-aligned regions on demand and caches them. Blocks are boxed and
/// appended, never removed or resized, so their heap addresses are stable — a
/// slice handed out by [`slice`](ByteSource::slice) stays valid for as long as
/// the source lives, even across later fetches (which may grow the index `Vec`;
/// the `Box` contents themselves never move).
pub struct RangeSource<R: Read + Seek> {
    reader: Mutex<R>,
    total: u64,
    block: u64,
    cache: crate::lock::Mutex<Blocks>,
    /// The block the last request was served from: checked first, without
    /// the lock — consecutive reads mostly land in the same block.
    last: std::sync::atomic::AtomicPtr<Block>,
}

/// A fetched `(start, bytes)` block.
type Block = (u64, Box<[u8]>);

#[derive(Default)]
struct Blocks {
    /// Append-only blocks, each boxed: neither the record nor its bytes move
    /// or are freed while the source lives (`last` points at a record).
    #[allow(clippy::vec_box)]
    list: Vec<Box<Block>>,
    /// `block_start -> newest covering block index` (a lookup shortcut; the old
    /// block, if superseded by a larger one, remains in `list` and valid).
    index: crate::util::AddrMap<u64, usize>,
}

impl<R: Read + Seek> RangeSource<R> {
    /// Build a source, discovering the length via `seek(End)`.
    pub fn new(mut reader: R) -> Result<Self> {
        let total = reader.seek(SeekFrom::End(0))?;
        Ok(Self::with_len(reader, total))
    }

    /// Build a source with a known total length (skips the `seek(End)` probe).
    pub fn with_len(reader: R, total: u64) -> Self {
        Self {
            reader: Mutex::new(reader),
            total,
            block: DEFAULT_BLOCK as u64,
            cache: Default::default(),
            last: Default::default(),
        }
    }

    /// Fetch `block` bytes at a time (a power of two). Smaller costs more
    /// round-trips where metadata clusters, larger reads more it does not need
    /// where metadata is scattered.
    pub fn with_block(mut self, block: usize) -> Result<Self> {
        if !block.is_power_of_two() {
            return Err(format!("block size {block} is not a power of two").into());
        }
        self.block = block as u64;
        Ok(self)
    }

    /// Number of blocks fetched so far (diagnostics / tests).
    pub fn blocks_fetched(&self) -> usize {
        self.cache.lock().list.len()
    }

    /// SAFETY invariant helper: pointer into a boxed block, valid for `&self`.
    unsafe fn stable(&self, ptr: *const u8, len: usize) -> &[u8] {
        std::slice::from_raw_parts(ptr, len)
    }
}

impl<R: Read + Seek> RangeSource<R> {
    /// The cached block covering `[offset, end)`, fetched if there is none:
    /// `(start, bytes)`.
    fn block_for(&self, offset: u64, end: u64) -> Result<(u64, &[u8])> {
        use std::sync::atomic::Ordering::{Acquire, Release};
        let last = self.last.load(Acquire);
        if !last.is_null() {
            // SAFETY: `last` points at a boxed record in `list`, never moved or
            // freed while the source lives
            let (start, block) = unsafe { &*last };
            if *start <= offset && *start + block.len() as u64 >= end {
                return Ok((*start, unsafe { self.stable(block.as_ptr(), block.len()) }));
            }
        }
        let wstart = offset & !(self.block - 1);
        let mut cache = self.cache.lock();
        // A cached block for this window already covers the request.
        if let Some(&i) = cache.index.get(&wstart) {
            let record: &Block = &cache.list[i];
            let (start, block) = record;
            if *start + block.len() as u64 >= end {
                self.last
                    .store(record as *const Block as *mut Block, Release);
                // SAFETY: `block` is boxed and never moved/freed for `&self`'s
                // lifetime, so this reference outlives the guard.
                return Ok((*start, unsafe { self.stable(block.as_ptr(), block.len()) }));
            }
        }
        // Fetch a fresh BLOCK-aligned region covering the whole request.
        let wend = std::cmp::max(wstart + self.block, end).min(self.total);
        let mut buf = vec![0u8; usize::try_from(wend - wstart).map_err(|_| "read too large")?];
        {
            let mut r = self.reader.lock().unwrap();
            r.seek(SeekFrom::Start(wstart))?;
            r.read_exact(&mut buf)?;
        }
        let idx = cache.list.len();
        cache.list.push(Box::new((wstart, buf.into_boxed_slice())));
        cache.index.insert(wstart, idx);
        let record: &Block = &cache.list[idx];
        self.last
            .store(record as *const Block as *mut Block, Release);
        let block = &record.1;
        // SAFETY: as above — the boxed block's heap storage is stable for the
        // source's lifetime regardless of `list` growth.
        Ok((wstart, unsafe { self.stable(block.as_ptr(), block.len()) }))
    }
}

impl<R: Read + Seek> ByteSource for RangeSource<R> {
    fn slice(&self, offset: u64, len: usize) -> Result<&[u8]> {
        let end = end_of(offset, len)?;
        if end > self.total {
            return Err("unexpected end of file".into());
        }
        if len == 0 {
            return Ok(&[]);
        }
        let (start, block) = self.block_for(offset, end)?;
        let rel = (offset - start) as usize;
        Ok(&block[rel..rel + len])
    }

    fn window(&self, offset: u64) -> Option<(u64, &[u8])> {
        if offset >= self.total {
            return None;
        }
        self.block_for(offset, offset + 1).ok()
    }

    fn len(&self) -> u64 {
        self.total
    }
}

/// A file on disk that this process appends to while it is open.
///
/// The bytes present when it was opened are memory-mapped (read into memory
/// where the platform has no mmap, as on WASI). Bytes appended since — chunks
/// written through, metadata of incremental flushes — are read back with plain
/// reads, and kept, like [`RangeSource`]'s blocks, so a returned slice stays
/// valid for the source's lifetime.
pub struct FileSource {
    file: Mutex<std::fs::File>,
    /// The file as it was when opened (`None`: it was empty / just created).
    image: Option<std::sync::Arc<FileImage>>,
    len: std::sync::atomic::AtomicU64,
    /// Read-back bytes past the image: append-only, boxed, never moved.
    tail: crate::lock::Mutex<Vec<Block>>,
}

impl std::fmt::Debug for FileSource {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "FileSource {{ len: {} }}", self.len())
    }
}

impl FileSource {
    /// Open `path` for reading and appending.
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)?;
        let len = file.metadata()?.len();
        let image = if len == 0 {
            None
        } else {
            // SAFETY: as for every mmap of a file: it must not be truncated
            // underneath us; this crate only ever appends to, and rewrites the
            // superblock of, a file it holds open this way
            Some(std::sync::Arc::new(
                match unsafe { memmap2::Mmap::map(&file) } {
                    Ok(m) => FileImage::Mmap(m),
                    Err(e) if e.kind() == std::io::ErrorKind::Unsupported => {
                        FileImage::Bytes(std::fs::read(path)?)
                    }
                    Err(e) => return Err(e.into()),
                },
            ))
        };
        Ok(Self {
            file: Mutex::new(file),
            image,
            len: len.into(),
            tail: Default::default(),
        })
    }

    /// Append `bytes` at the end of the file; returns where they start.
    pub fn append(&self, bytes: &[u8]) -> Result<u64> {
        use std::io::Write;
        let mut file = self.file.lock().unwrap_or_else(|e| e.into_inner());
        let at = self.len.load(std::sync::atomic::Ordering::Acquire);
        file.seek(SeekFrom::Start(at))?;
        file.write_all(bytes)?;
        self.len.store(
            at + bytes.len() as u64,
            std::sync::atomic::Ordering::Release,
        );
        Ok(at)
    }

    /// Overwrite bytes in place — only for the superblock, which is read once
    /// at open and never handed out as a slice afterwards.
    pub fn write_at(&self, offset: u64, bytes: &[u8]) -> Result<()> {
        use std::io::Write;
        let mut file = self.file.lock().unwrap_or_else(|e| e.into_inner());
        file.seek(SeekFrom::Start(offset))?;
        file.write_all(bytes)?;
        file.flush()?;
        Ok(())
    }

    /// Bytes past the image, read back (and kept) on first use.
    fn tail(&self, offset: u64, len: usize) -> Result<&[u8]> {
        let mut tail = self.tail.lock();
        for (start, bytes) in tail.iter().rev() {
            if *start <= offset && offset + len as u64 <= *start + bytes.len() as u64 {
                let rel = (offset - start) as usize;
                // SAFETY: boxed bytes, never moved or freed while `self` lives
                return Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr().add(rel), len) });
            }
        }
        // a little extra: the next read is usually just after this one
        let end = (offset + len.max(64 << 10) as u64).min(self.len());
        let mut buf = vec![0u8; usize::try_from(end - offset).map_err(|_| "read too large")?];
        {
            let mut file = self.file.lock().unwrap_or_else(|e| e.into_inner());
            file.seek(SeekFrom::Start(offset))?;
            file.read_exact(&mut buf)?;
        }
        tail.push((offset, buf.into_boxed_slice()));
        let bytes = &tail.last().expect("just pushed").1;
        // SAFETY: as above
        Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr(), len) })
    }
}

impl ByteSource for FileSource {
    fn slice(&self, offset: u64, len: usize) -> Result<&[u8]> {
        let end = end_of(offset, len)?;
        if end > self.len() {
            return Err("unexpected end of file".into());
        }
        match &self.image {
            Some(image) if end <= image.len() as u64 => in_memory(image, offset, len),
            _ => self.tail(offset, len),
        }
    }

    fn window(&self, offset: u64) -> Option<(u64, &[u8])> {
        let image = self.image.as_ref()?;
        (offset < image.len() as u64).then(|| (0, &image[..]))
    }

    fn len(&self) -> u64 {
        self.len.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// A source filled from outside, for async I/O: a read of bytes not yet
/// here fails, and the range is noted ([`missing`](Self::missing)) for the
/// caller to fetch — all of an attempt's misses at once — and
/// [`insert`](Self::insert) before trying again. See
/// [`crate::remote::RemoteFile`].
pub struct FetchSource {
    total: u64,
    block: u64,
    blocks: crate::lock::Mutex<FetchBlocks>,
    missing: crate::lock::Mutex<Vec<(u64, u64)>>,
}

#[derive(Default)]
struct FetchBlocks {
    list: Vec<Block>,
    /// block start -> the widest block fetched from there
    index: crate::util::AddrMap<u64, usize>,
}

impl FetchSource {
    /// A source for a file of `total` bytes, fetched `block` bytes (a power
    /// of two) at a time.
    pub fn new(total: u64, block: usize) -> Result<Self> {
        if !block.is_power_of_two() {
            return Err(format!("block size {block} is not a power of two").into());
        }
        Ok(Self {
            total,
            block: block as u64,
            blocks: Default::default(),
            missing: Default::default(),
        })
    }

    /// The ranges reads asked for and did not find, as block-aligned
    /// `(start, end)` to fetch; clears the list.
    pub fn missing(&self) -> Vec<(u64, u64)> {
        let mut ranges = std::mem::take(&mut *self.missing.lock());
        ranges.sort_unstable();
        ranges.dedup_by(|b, a| {
            // `b` follows `a` in order: one fetch covers both when they share
            // a start (the wider wins)
            if a.0 == b.0 {
                a.1 = a.1.max(b.1);
                true
            } else {
                false
            }
        });
        ranges
    }

    /// Hand over fetched bytes: `bytes` are the file's from `start` on.
    pub fn insert(&self, start: u64, bytes: Vec<u8>) {
        let mut blocks = self.blocks.lock();
        let widest = blocks
            .index
            .get(&start)
            .map(|&i| blocks.list[i].1.len())
            .unwrap_or(0);
        if bytes.len() > widest {
            let i = blocks.list.len();
            blocks.list.push((start, bytes.into_boxed_slice()));
            blocks.index.insert(start, i);
        }
    }

    /// Bytes held so far.
    pub fn bytes_fetched(&self) -> u64 {
        self.blocks
            .lock()
            .list
            .iter()
            .map(|b| b.1.len() as u64)
            .sum()
    }
}

impl ByteSource for FetchSource {
    fn slice(&self, offset: u64, len: usize) -> Result<&[u8]> {
        let end = end_of(offset, len)?;
        if end > self.total {
            return Err("unexpected end of file".into());
        }
        if len == 0 {
            return Ok(&[]);
        }
        let wstart = offset & !(self.block - 1);
        {
            let blocks = self.blocks.lock();
            if let Some(&i) = blocks.index.get(&wstart) {
                let (start, bytes) = &blocks.list[i];
                if *start + bytes.len() as u64 >= end {
                    let rel = (offset - start) as usize;
                    // SAFETY: boxed, never moved or freed while `self` lives
                    return Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr().add(rel), len) });
                }
            }
        }
        let wend = (wstart + self.block).max(end).min(self.total);
        self.missing.lock().push((wstart, wend));
        Err("not fetched yet".into())
    }

    fn len(&self) -> u64 {
        self.total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor as IoCursor;

    #[test]
    fn range_source_reads_and_caches() {
        let data: Vec<u8> = (0..(3 * DEFAULT_BLOCK as u32 + 17))
            .map(|i| i as u8)
            .collect();
        let src = RangeSource::new(IoCursor::new(data.clone())).unwrap();
        assert_eq!(src.len(), data.len() as u64);
        // read spanning a block boundary
        let a = src.slice(DEFAULT_BLOCK as u64 - 4, 8).unwrap();
        assert_eq!(a, &data[DEFAULT_BLOCK - 4..DEFAULT_BLOCK + 4]);
        // a second read in the first block reuses the cached block
        let b = src.slice(10, 4).unwrap();
        assert_eq!(b, &data[10..14]);
        // earlier slice still valid after more fetches
        assert_eq!(a, &data[DEFAULT_BLOCK - 4..DEFAULT_BLOCK + 4]);
        // out of range
        assert!(src.slice(data.len() as u64, 1).is_err());
    }
}
