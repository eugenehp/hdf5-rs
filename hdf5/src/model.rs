//! In-memory object model for the pure-Rust HDF5 engine.
//!
//! An open file is represented as an arena of objects (groups, datasets, named
//! datatypes) linked together, plus per-object attributes. All high-level API
//! handles refer into this shared, interior-mutable model. On flush/close the
//! whole model is serialized into a valid HDF5 byte stream (see
//! [`crate::format`]); on open it is parsed back out.

use std::path::PathBuf;

use hdf5_types::TypeDescriptor;

use crate::hl::extents::Extents;
use crate::hl::filters::Filter;

/// Index of an object within a [`FileState`] arena.
pub type ObjId = usize;

/// The raw bytes of an opened file: either memory-mapped (zero-copy, paged in
/// by the OS on demand) or an owned buffer.
pub enum FileImage {
    Mmap(memmap2::Mmap),
    Bytes(Vec<u8>),
}

impl std::ops::Deref for FileImage {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Self::Mmap(m) => m,
            Self::Bytes(b) => b,
        }
    }
}

impl std::fmt::Debug for FileImage {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "FileImage({} bytes)", self.len())
    }
}

/// Deferred dataset bytes: read from the backing [`ByteSource`] on first access
/// so opening a file costs metadata only. The source may be a fully-present
/// image (mmap/bytes) or a lazy range-reader, so this works identically for
/// local and streamed files.
/// A zeroed buffer of `n` bytes, or an error when it cannot be had: sizes
/// here come from the file, and a damaged one must not abort the process.
pub(crate) fn zeroed(n: usize) -> crate::error::Result<Vec<u8>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| format!("not enough memory for {n} bytes; read it in parts"))?;
    v.resize(n, 0);
    Ok(v)
}

/// A dataset's chunks, shared between readers.
pub type ChunkList = std::sync::Arc<Vec<crate::format::reader::ChunkEntry>>;

#[derive(Clone)]
pub struct LazyData {
    pub src: std::sync::Arc<dyn crate::format::source::ByteSource + Send + Sync>,
    /// File base address (superblock offset).
    pub base: u64,
    /// Logical dataset size in bytes (output buffer size).
    pub logical: u64,
    pub kind: LazyKind,
    /// The walked chunk index `(offsets, size, filter_mask, addr)`, memoized by
    /// the first read that needs all of it, so random access does not re-walk
    /// the index per call. Chunks written since the file was opened (see
    /// [`LazyData::write_pick`]) are only here — the on-disk index is
    /// rewritten at the next flush.
    pub resolved_chunks: std::sync::Arc<crate::lock::RwLock<Option<ChunkList>>>,
    /// Big-endian on disk: the width to reverse as read, else 0.
    pub swap: usize,
    /// Decoded chunks kept between partial reads.
    pub cache: std::sync::Arc<crate::cache::ChunkCache>,
}

impl std::fmt::Debug for LazyData {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("LazyData")
            .field("base", &self.base)
            .field("logical", &self.logical)
            .field("kind", &self.kind)
            .finish()
    }
}

/// What must be done to materialize the deferred bytes.
#[derive(Clone, Debug)]
pub enum LazyKind {
    /// One contiguous run: `image[offset..offset + len]`, zero-padded to
    /// `logical`.
    Contiguous { offset: u64, len: u64 },
    /// Chunked storage: the chunk *index* is walked, then every chunk decoded
    /// through the filter pipeline and scattered into place — all on first
    /// access. Storing the index descriptor (not the walked chunk list) means
    /// opening a file never enumerates chunks.
    Chunked {
        index: crate::format::reader::ChunkIndex,
        dims: Vec<u64>,
        chunk_dims: Vec<u64>,
        elem_size: usize,
        filters: Vec<crate::format::filters::RawFilter>,
    },
}

impl LazyData {
    /// Bytes of rows `[r0, r1)` along the leading axis, decoding ONLY the
    /// chunks that overlap them (the full-dataset path decodes every chunk).
    /// `row_bytes` = element size × product of the trailing dims.
    pub fn materialize_rows(
        &self,
        r0: u64,
        r1: u64,
        row_bytes: usize,
    ) -> crate::error::Result<Vec<u8>> {
        self.native(self.stored_rows(r0, r1, row_bytes)?)
    }

    /// `bytes` in the order the model holds: little-endian.
    fn native(&self, mut bytes: Vec<u8>) -> crate::error::Result<Vec<u8>> {
        self.swap_in_place(&mut bytes);
        Ok(bytes)
    }

    fn swap_in_place(&self, bytes: &mut [u8]) {
        if self.swap > 1 {
            let width = self.swap;
            crate::par::units_mut(bytes, width, |part| {
                crate::format::simd::swap_bytes(width, part)
            });
        }
    }

    /// The whole dataset, into `out` (zeroed, of the logical size): nothing is
    /// kept here, and nothing copied twice.
    pub fn materialize_into(&self, out: &mut [u8]) -> crate::error::Result<()> {
        // big-endian numbers: swapped as they are placed, not in a second pass
        if self.swap > 1 {
            let copy = |s: &[u8], d: &mut [u8]| d.copy_from_slice(s);
            if self.materialize_converted(out, self.swap, self.swap, &copy)? {
                return Ok(());
            }
        }
        let src: &dyn crate::format::source::ByteSource = &*self.src;
        match &self.kind {
            LazyKind::Contiguous { offset, len } => {
                let n = (*len).min(out.len() as u64) as usize;
                // spread over cores: from an mmap, the page faults are most of it
                crate::par::zip_units(src.slice(*offset, n)?, 1, &mut out[..n], 1, |s, d| {
                    d.copy_from_slice(s)
                });
            }
            LazyKind::Chunked {
                dims,
                chunk_dims,
                elem_size,
                filters,
                ..
            } => {
                let chunks = self.chunk_list()?;
                crate::format::reader::materialize_chunks(
                    src, self.base, &chunks, dims, chunk_dims, *elem_size, filters, out,
                )?;
            }
        }
        self.swap_in_place(out);
        Ok(())
    }

    /// The whole dataset into `out` as elements of `out_es` bytes, each run
    /// of stored elements (`es` bytes, little-endian) passed through `conv`
    /// as it is placed — decode, byte-swap and conversion in one pass, with
    /// no intermediate copy of the dataset. `false` (and `out` untouched)
    /// where that would be slower: chunks whose rows are only a few elements
    /// wide, where a per-run call would cost more than a second pass.
    pub fn materialize_converted(
        &self,
        out: &mut [u8],
        es: usize,
        out_es: usize,
        conv: &(dyn Fn(&[u8], &mut [u8]) + Sync),
    ) -> crate::error::Result<bool> {
        let swap = self.swap;
        // big-endian: swap a stack buffer's worth at a time, then convert
        let swapped = |s: &[u8], d: &mut [u8]| {
            let mut tmp = [0u8; 4096];
            let step = (4096 / es).max(1) * es;
            for (sc, dc) in s.chunks(step).zip(d.chunks_mut(step / es * out_es)) {
                let t = &mut tmp[..sc.len()];
                t.copy_from_slice(sc);
                crate::format::simd::swap_bytes(swap, t);
                conv(t, dc);
            }
        };
        let f: &(dyn Fn(&[u8], &mut [u8]) + Sync) = if swap > 1 { &swapped } else { conv };
        let src: &dyn crate::format::source::ByteSource = &*self.src;
        match &self.kind {
            LazyKind::Contiguous { offset, len } => {
                let n = (*len).min(self.logical) as usize / es * es;
                let stored = src.slice(*offset, n)?;
                crate::par::zip_units(stored, es, &mut out[..n / es * out_es], out_es, f);
            }
            LazyKind::Chunked {
                dims,
                chunk_dims,
                elem_size,
                filters,
                ..
            } => {
                if chunk_dims.last().is_some_and(|&run| run < 16) || *elem_size != es {
                    return Ok(false);
                }
                let chunks = self.chunks_for_rows(0, dims.first().copied().unwrap_or(1))?;
                crate::format::reader::materialize_chunks_with(
                    src,
                    self.base,
                    &chunks,
                    dims,
                    chunk_dims,
                    *elem_size,
                    filters,
                    out,
                    Some((out_es, f)),
                )?;
            }
        }
        Ok(true)
    }

    fn stored_rows(&self, r0: u64, r1: u64, row_bytes: usize) -> crate::error::Result<Vec<u8>> {
        let src: &dyn crate::format::source::ByteSource = &*self.src;
        let n = (r1 - r0) as usize * row_bytes;
        match &self.kind {
            LazyKind::Contiguous { offset, len } => {
                let mut v = zeroed(n)?;
                let start = r0 * row_bytes as u64;
                if start < *len {
                    // no more than n, which is a usize
                    let take = (*len - start).min(n as u64) as usize;
                    v[..take].copy_from_slice(src.slice(*offset + start, take)?);
                }
                Ok(v)
            }
            LazyKind::Chunked {
                dims,
                chunk_dims,
                elem_size,
                filters,
                ..
            } => {
                let rank = dims.len();
                let c0 = chunk_dims.first().copied().unwrap_or(1);
                let chunks = self.chunks_for_rows(r0, r1)?;
                let wanted: Vec<crate::format::reader::ChunkEntry> = chunks
                    .iter()
                    .filter(|(off, ..)| rank > 0 && off[0] < r1 && off[0] + c0 > r0)
                    .cloned()
                    .collect();
                let mut out = zeroed(n)?;
                crate::format::reader::materialize_chunk_rows(
                    src,
                    self.base,
                    &wanted,
                    dims,
                    chunk_dims,
                    *elem_size,
                    filters,
                    r0,
                    r1,
                    &mut out,
                    Some(&self.cache),
                )?;
                Ok(out)
            }
        }
    }

    /// The walked chunk index, or just the part covering rows `[r0, r1)`
    /// where the index allows a partial walk (and the whole index is not yet
    /// memoized).
    fn chunks_for_rows(&self, r0: u64, r1: u64) -> crate::error::Result<ChunkList> {
        let LazyKind::Chunked {
            index,
            dims,
            chunk_dims,
            ..
        } = &self.kind
        else {
            return Ok(Default::default());
        };
        if let Some(c) = &*self.resolved_chunks.read() {
            return Ok(c.clone());
        }
        let src: &dyn crate::format::source::ByteSource = &*self.src;
        if !dims.is_empty() {
            if let Some(part) = crate::format::reader::resolve_chunk_rows(
                src,
                self.base,
                index,
                dims.len(),
                (r0, r1, chunk_dims[0]),
            )? {
                return Ok(std::sync::Arc::new(part));
            }
        }
        self.chunk_list()
    }

    /// Every stored chunk, walking the index the first time.
    pub fn chunk_list(&self) -> crate::error::Result<ChunkList> {
        let LazyKind::Chunked {
            index,
            dims,
            chunk_dims,
            elem_size,
            ..
        } = &self.kind
        else {
            return Ok(Default::default());
        };
        let mut memo = self.resolved_chunks.write();
        if let Some(c) = &*memo {
            return Ok(c.clone());
        }
        let src: &dyn crate::format::source::ByteSource = &*self.src;
        let c = std::sync::Arc::new(crate::format::reader::resolve_chunks(
            src, self.base, index, dims, chunk_dims, *elem_size,
        )?);
        *memo = Some(c.clone());
        Ok(c)
    }

    /// The elements `pick` selects, packed in pick order (little-endian),
    /// reading only the stored runs or chunks they lie in. `dims` are the
    /// dataset's.
    pub(crate) fn pick(
        &self,
        dims: &[u64],
        elem_size: usize,
        pick: &crate::format::reader::Pick,
    ) -> crate::error::Result<Vec<u8>> {
        let src: &dyn crate::format::source::ByteSource = &*self.src;
        let bytes = pick
            .len()
            .checked_mul(elem_size)
            .ok_or("selection too large")?;
        let mut out = zeroed(bytes)?;
        match &self.kind {
            LazyKind::Contiguous { offset, len } => {
                crate::format::reader::contiguous_pick_into(
                    src,
                    (*offset, *len),
                    dims,
                    elem_size,
                    pick,
                    &mut out,
                )?;
            }
            LazyKind::Chunked {
                dims,
                chunk_dims,
                elem_size,
                filters,
                ..
            } => {
                let (r0, r1) = pick.row_span().unwrap_or((0, 0));
                let chunks = self.chunks_for_rows(r0, r1)?;
                let cs = crate::format::reader::ChunkSource {
                    src,
                    base: self.base,
                    dims,
                    chunk_dims,
                    elem_size: *elem_size,
                    filters,
                    cache: Some(&self.cache),
                };
                crate::format::reader::chunks_pick_into(&cs, &chunks, pick, &mut out)?;
            }
        }
        self.native(out)
    }

    /// Write `values` (stored layout, little-endian, in pick order) into the
    /// picked elements, straight to `file`: only the chunks the pick touches
    /// are decoded, updated and appended; the chunk list here is updated, and
    /// the on-disk index is rewritten at the next flush. Chunked data only.
    pub(crate) fn write_pick(
        &self,
        file: &crate::format::source::FileSource,
        pick: &crate::format::reader::Pick,
        values: &[u8],
        fill: &[u8],
    ) -> crate::error::Result<()> {
        let LazyKind::Chunked {
            dims,
            chunk_dims,
            elem_size,
            filters,
            ..
        } = &self.kind
        else {
            return Err("write_pick: not chunked".into());
        };
        let chunks = self.chunk_list()?;
        let cs = crate::format::reader::ChunkSource {
            src: &*self.src,
            base: self.base,
            dims,
            chunk_dims,
            elem_size: *elem_size,
            filters,
            cache: Some(&self.cache),
        };
        let base = self.base;
        let mut append =
            |bytes: &[u8]| -> crate::error::Result<u64> { Ok(file.append(bytes)? - base) };
        let written = crate::format::reader::chunks_write_pick(
            &cs,
            &chunks,
            pick,
            values,
            fill,
            &mut append,
        )?;
        drop(chunks);
        // merge: a rewritten chunk replaces its old entry, a new one is added
        let rank = dims.len();
        let mut memo = self.resolved_chunks.write();
        let list = std::sync::Arc::make_mut(memo.get_or_insert_with(Default::default));
        let mut at: std::collections::HashMap<Vec<u64>, usize> = list
            .iter()
            .enumerate()
            .map(|(i, c)| (c.0[..rank].to_vec(), i))
            .collect();
        for entry in written {
            let key = entry.0[..rank].to_vec();
            match at.get(&key) {
                Some(&i) => list[i] = entry,
                None => {
                    at.insert(key, list.len());
                    list.push(entry);
                }
            }
        }
        Ok(())
    }

    /// The dataset reshaped to `dims` without reading it: its chunks stay
    /// where they are (the list is walked first, as some indexes are laid out
    /// by the old shape); new regions read as fill until written.
    pub(crate) fn resized(&self, dims: &[u64]) -> crate::error::Result<Self> {
        let LazyKind::Chunked {
            index,
            chunk_dims,
            elem_size,
            filters,
            ..
        } = &self.kind
        else {
            return Err("resized: not chunked".into());
        };
        let chunks = self.chunk_list()?;
        Ok(Self {
            logical: dims.iter().product::<u64>() * *elem_size as u64,
            kind: LazyKind::Chunked {
                index: index.clone(),
                dims: dims.to_vec(),
                chunk_dims: chunk_dims.clone(),
                elem_size: *elem_size,
                filters: filters.clone(),
            },
            resolved_chunks: std::sync::Arc::new(crate::lock::RwLock::new(Some(chunks))),
            ..self.clone()
        })
    }

    /// The dataset's size where it must fit in memory at once.
    fn logical_in_memory(&self) -> crate::error::Result<usize> {
        usize::try_from(self.logical)
            .map_err(|_| format!("dataset too large to load here ({} bytes)", self.logical).into())
    }

    /// Load the deferred bytes into a fresh logical buffer.
    pub fn materialize_bytes(&self) -> crate::error::Result<Vec<u8>> {
        self.stored_bytes()
    }

    fn stored_bytes(&self) -> crate::error::Result<Vec<u8>> {
        let n = self.logical_in_memory()?;
        // a dataset bigger than memory is an error to read whole, not an abort
        let mut out = Vec::new();
        out.try_reserve_exact(n).map_err(|_| {
            format!("not enough memory to read {n} bytes at once; read it in parts")
        })?;
        out.resize(n, 0);
        self.materialize_into(&mut out)?;
        Ok(out)
    }
}

/// Data-layout class of a dataset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayoutClass {
    /// Data stored inline within the object header.
    Compact,
    /// Data stored as a single contiguous block.
    Contiguous,
    /// Data stored as filtered/unfiltered chunks indexed by a v1 B-tree.
    Chunked(Vec<u64>),
    /// Virtual dataset: data mapped from source datasets per the mappings.
    Virtual(Vec<crate::format::vds::VdsMapping>),
}

/// Fill-value policy for a dataset.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum FillValue {
    /// No fill value defined.
    #[default]
    Undefined,
    /// Library default (zero) fill value.
    Default,
    /// User-defined fill value (raw bytes in the dataset's datatype layout).
    UserDefined(Vec<u8>),
}

/// The target of a link in a group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LinkTarget {
    /// Hard link to an object in the same file (arena index).
    Hard(ObjId),
    /// Soft (symbolic) link to a path within the same file.
    Soft(String),
    /// External link to an object in another file.
    External { file: String, path: String },
}

/// A named link within a group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Link {
    pub name: String,
    pub target: LinkTarget,
    pub creation_order: i64,
    pub utf8: bool,
}

/// A group's contents.
#[derive(Clone, Debug, Default)]
pub struct GroupData {
    pub links: Vec<Link>,
    pub track_order: bool,
    index: NameIndex,
}

/// Members a group has before a name is looked up in an index, not by scanning.
const INDEXED_LINKS: usize = 32;

/// Link names to their place in `links`, built on the first lookup. `links` is
/// changed freely, so the index is only ever a hint: a hit is checked against
/// the link it names, and a miss falls back to scanning.
#[derive(Default)]
struct NameIndex(crate::lock::RwLock<Option<bbtree::unsync::BBTreeMap<String, u32>>>);

impl Clone for NameIndex {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl std::fmt::Debug for NameIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("NameIndex")
    }
}

impl GroupData {
    pub fn find(&self, name: &str) -> Option<&Link> {
        self.find_index(name).map(|i| &self.links[i])
    }

    pub fn find_index(&self, name: &str) -> Option<usize> {
        if self.links.len() >= INDEXED_LINKS {
            let hit = match &*self.index.0.read() {
                Some(names) => names.get(name).map(|i| *i as usize),
                None => None,
            };
            if let Some(i) = hit.filter(|i| self.links.get(*i).is_some_and(|l| l.name == name)) {
                return Some(i);
            }
        }
        let found = self.links.iter().position(|l| l.name == name);
        if self.links.len() >= INDEXED_LINKS && found.is_some() {
            // not there, or stale: this group's links have changed since
            let mut names = bbtree::unsync::BBTreeMap::new();
            for (i, link) in self.links.iter().enumerate().rev() {
                names.insert(link.name.clone(), i as u32);
            }
            *self.index.0.write() = Some(names);
        }
        found
    }
}

/// A dataset's metadata and data.
#[derive(Clone, Debug)]
pub struct DatasetData {
    /// Datatype in disk representation (see [`crate::format::convert`]).
    pub dtype: TypeDescriptor,
    pub dims: Vec<u64>,
    pub maxdims: Vec<Option<u64>>,
    pub layout: LayoutClass,
    pub filters: Vec<Filter>,
    pub fill: FillValue,
    /// Raw element bytes in the disk datatype layout, row-major (C order).
    /// Vlen components are `{len, store_idx}` slots referencing `vlen`.
    pub data: Vec<u8>,
    /// Side store for variable-length payloads (1-based indices).
    pub vlen: crate::format::convert::VlenStore,
    /// Present while the data has not been copied out of the file image.
    pub lazy: Option<LazyData>,
    pub is_scalar: bool,
    pub is_null: bool,
    /// Object references: the objects whose addresses are written in place
    /// of `data`, under the reference datatype (`H5R_OBJECT`).
    pub refs: Option<Vec<ObjId>>,
}

impl DatasetData {
    /// Copy lazily-referenced bytes out of the file image. Decoding errors
    /// surface on first access, matching libhdf5's read-time error behavior.
    pub fn materialize(&mut self) -> crate::error::Result<()> {
        if let Some(l) = self.lazy.take() {
            match l.materialize_bytes() {
                Ok(v) => self.data = v,
                Err(e) => {
                    self.lazy = Some(l); // stay lazy so a retry is possible
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    pub fn extents(&self) -> Extents {
        if self.is_null {
            Extents::Null
        } else if self.is_scalar {
            Extents::Scalar
        } else {
            use crate::hl::extents::{Extent, SimpleExtents};
            let ext: Vec<Extent> = self
                .dims
                .iter()
                .zip(self.maxdims.iter())
                .map(|(&d, &m)| Extent::new(d as usize, m.map(|v| v as usize)))
                .collect();
            Extents::Simple(SimpleExtents::from_vec(ext))
        }
    }

    /// Number of elements in the dataspace.
    pub fn num_elements(&self) -> usize {
        if self.is_null {
            0
        } else if self.is_scalar {
            1
        } else {
            self.dims.iter().product::<u64>() as usize
        }
    }
}

/// An attribute attached to an object.
#[derive(Clone, Debug)]
pub struct AttrData {
    pub name: String,
    /// Datatype in disk representation.
    pub dtype: TypeDescriptor,
    pub dims: Vec<u64>,
    pub is_scalar: bool,
    pub is_null: bool,
    /// Raw element bytes in the disk datatype layout, row-major.
    pub data: Vec<u8>,
    /// Side store for variable-length payloads (1-based indices).
    pub vlen: crate::format::convert::VlenStore,
    /// Object references: the objects whose addresses are written in place
    /// of `data`, under the reference datatype (`H5R_OBJECT`).
    pub refs: Option<Vec<ObjId>>,
}

impl AttrData {
    pub fn num_elements(&self) -> usize {
        if self.is_null {
            0
        } else if self.is_scalar {
            1
        } else {
            self.dims.iter().product::<u64>() as usize
        }
    }

    pub fn extents(&self) -> Extents {
        if self.is_null {
            Extents::Null
        } else if self.is_scalar {
            Extents::Scalar
        } else {
            use crate::hl::extents::{Extent, SimpleExtents};
            let ext: Vec<Extent> = self
                .dims
                .iter()
                .map(|&d| Extent::new(d as usize, Some(d as usize)))
                .collect();
            Extents::Simple(SimpleExtents::from_vec(ext))
        }
    }
}

/// The kind of an object stored in the arena.
#[derive(Clone, Debug)]
pub enum ObjectKind {
    Group(GroupData),
    /// Boxed: a dataset's description is ~5× a group's, and every arena slot
    /// (most of them lazy stubs, in a big file) would otherwise be that size.
    Dataset(Box<DatasetData>),
    NamedType(TypeDescriptor),
    /// An object present in the file but not parseable by this build; the
    /// string holds the reason. Accessing it errors; writing a file that
    /// contains one errors (preventing silent data loss on rewrite).
    Unsupported(String),
    /// A not-yet-parsed object in a lazily-opened file, identified by its
    /// object-header address. Replaced in place by the real kind on first
    /// navigation (see [`FileState::force`]).
    Lazy {
        oh_addr: u64,
    },
}

/// A single object in the arena, with its attributes.
#[derive(Clone, Debug)]
pub struct ObjectNode {
    pub kind: ObjectKind,
    pub attrs: Vec<AttrData>,
    pub comment: Option<String>,
    /// Number of hard links referencing this object (for lifetime bookkeeping).
    pub refcount: u32,
    /// Object modification time (seconds since epoch; 0 = unset).
    pub mtime: u32,
}

impl ObjectNode {
    pub fn new(kind: ObjectKind) -> Self {
        Self {
            kind,
            attrs: Vec::new(),
            comment: None,
            refcount: 0,
            mtime: 0,
        }
    }

    pub fn attr_index(&self, name: &str) -> Option<usize> {
        self.attrs.iter().position(|a| a.name == name)
    }
}

/// The mutable state of an open file: the object arena plus file-level metadata.
#[derive(Clone, Debug)]
pub struct FileState {
    pub objects: Vec<Option<ObjectNode>>,
    pub root: ObjId,
    pub userblock: u64,
    /// Userblock content (empty = zeros); preserved across rewrites.
    pub userblock_data: Vec<u8>,
    pub read_only: bool,
    pub next_creation_order: i64,
    /// SOHM configuration: (message-type flag bits, min message size) per
    /// index; empty = no shared-message tables.
    pub sohm: Vec<(u16, u32)>,
    /// Present for lazily-opened files: parses further objects on demand.
    /// `None` for eager/writable files.
    pub loader: Option<std::sync::Arc<crate::format::reader::LazyLoader>>,
    /// Object-header address -> object id, for resolving HDF5 object
    /// references. A reference token *is* an address, so without this a
    /// dereference has no way back to the object the parser already built.
    pub addrs: crate::util::AddrMap<u64, ObjId>,
    /// The file on disk, for a writable path-backed file: flushes append to
    /// it, and chunk writes go straight to it (see
    /// [`crate::format::flush_append`]).
    pub appendable: Option<std::sync::Arc<crate::format::source::FileSource>>,
    /// Changed since it was opened or last flushed: a clean file is not
    /// written at all.
    pub dirty: bool,
}

impl FileState {
    /// Create a new file state with an empty root group.
    pub fn new_empty() -> Self {
        let root_node = ObjectNode::new(ObjectKind::Group(GroupData::default()));
        let mut objects = vec![Some(root_node)];
        // The root group is referenced by the file itself.
        if let Some(Some(n)) = objects.get_mut(0) {
            n.refcount = 1;
        }
        Self {
            objects,
            root: 0,
            userblock: 0,
            userblock_data: Vec::new(),
            read_only: false,
            next_creation_order: 0,
            sohm: Vec::new(),
            loader: None,
            addrs: Default::default(),
            appendable: None,
            dirty: false,
        }
    }

    /// Force a lazily-opened object into its real form (no-op if already parsed
    /// or the file is eager). See [`crate::format::reader::force_object`].
    pub fn force(&mut self, id: ObjId) -> crate::error::Result<()> {
        if matches!(
            self.try_get(id).map(|n| &n.kind),
            Some(ObjectKind::Lazy { .. })
        ) {
            crate::format::reader::force_object(self, id)?;
        }
        Ok(())
    }

    /// Like [`resolve`](Self::resolve) but forces lazy objects along the path,
    /// so navigation of a lazily-opened file parses only what it touches.
    pub fn resolve_force(&mut self, start: ObjId, path: &str) -> Option<ObjId> {
        let mut cur = if path.starts_with('/') {
            self.root
        } else {
            start
        };
        self.force(cur).ok()?;
        for comp in path.split('/').filter(|s| !s.is_empty() && *s != ".") {
            let target = match &self.try_get(cur)?.kind {
                ObjectKind::Group(g) => g.find(comp)?.target.clone(),
                _ => return None,
            };
            cur = match target {
                LinkTarget::Hard(id) => id,
                LinkTarget::Soft(t) => self.resolve_force(self.root, &t)?,
                LinkTarget::External { .. } => return None,
            };
            self.force(cur).ok()?;
        }
        Some(cur)
    }

    pub fn get(&self, id: ObjId) -> &ObjectNode {
        self.objects[id].as_ref().expect("dangling object id")
    }

    pub fn get_mut(&mut self, id: ObjId) -> &mut ObjectNode {
        self.objects[id].as_mut().expect("dangling object id")
    }

    pub fn try_get(&self, id: ObjId) -> Option<&ObjectNode> {
        self.objects.get(id).and_then(|o| o.as_ref())
    }

    pub fn alloc(&mut self, kind: ObjectKind) -> ObjId {
        let id = self.objects.len();
        self.objects.push(Some(ObjectNode::new(kind)));
        id
    }

    pub fn next_order(&mut self) -> i64 {
        let n = self.next_creation_order;
        self.next_creation_order += 1;
        n
    }

    pub fn is_group(&self, id: ObjId) -> bool {
        matches!(
            self.try_get(id).map(|n| &n.kind),
            Some(ObjectKind::Group(_))
        )
    }

    pub fn is_dataset(&self, id: ObjId) -> bool {
        matches!(
            self.try_get(id).map(|n| &n.kind),
            Some(ObjectKind::Dataset(_))
        )
    }

    pub fn group_data(&self, id: ObjId) -> Option<&GroupData> {
        match &self.try_get(id)?.kind {
            ObjectKind::Group(g) => Some(g),
            _ => None,
        }
    }

    pub fn dataset_data(&self, id: ObjId) -> Option<&DatasetData> {
        match &self.try_get(id)?.kind {
            ObjectKind::Dataset(d) => Some(d),
            _ => None,
        }
    }

    /// Resolve a possibly-nested path to an object id, starting from `start`.
    /// Absolute paths (beginning with `/`) start from the root group. Follows
    /// hard and soft links; external links are not followed.
    pub fn resolve(&self, start: ObjId, path: &str) -> Option<ObjId> {
        let mut cur = if path.starts_with('/') {
            self.root
        } else {
            start
        };
        for comp in path.split('/').filter(|s| !s.is_empty() && *s != ".") {
            let group = match &self.try_get(cur)?.kind {
                ObjectKind::Group(g) => g,
                _ => return None,
            };
            let link = group.find(comp)?;
            cur = match &link.target {
                LinkTarget::Hard(id) => *id,
                LinkTarget::Soft(target) => self.resolve(self.root, target)?,
                LinkTarget::External { .. } => return None,
            };
        }
        Some(cur)
    }

    /// Find one path from the root to the given object (BFS). Returns "/" for
    /// the root. Returns `None` if unreachable.
    pub fn path_of(&self, target: ObjId) -> Option<String> {
        if target == self.root {
            return Some("/".to_string());
        }
        let mut queue: std::collections::VecDeque<(ObjId, String)> =
            std::collections::VecDeque::new();
        let mut seen = std::collections::HashSet::new();
        queue.push_back((self.root, String::new()));
        seen.insert(self.root);
        while let Some((cur, prefix)) = queue.pop_front() {
            if let Some(ObjectKind::Group(g)) = self.try_get(cur).map(|n| &n.kind) {
                for link in &g.links {
                    if let LinkTarget::Hard(id) = link.target {
                        let path = format!("{prefix}/{}", link.name);
                        if id == target {
                            return Some(path);
                        }
                        if self.is_group(id) && seen.insert(id) {
                            queue.push_back((id, path));
                        }
                    }
                }
            }
        }
        None
    }

    /// Materialize every lazy dataset (required before serialization).
    pub fn materialize_all(&mut self) -> crate::error::Result<()> {
        for slot in self.objects.iter_mut().flatten() {
            if let ObjectKind::Dataset(d) = &mut slot.kind {
                d.materialize()?;
            }
        }
        Ok(())
    }

    /// Parse every object a lazily opened file has not reached yet. Forcing a
    /// group adds its children to the arena, which this then reaches in turn.
    pub fn force_all(&mut self) -> crate::error::Result<()> {
        let mut id = 0;
        while id < self.objects.len() {
            self.force(id)?;
            id += 1;
        }
        Ok(())
    }

    /// Recompute hard-link reference counts across the arena.
    pub fn recount(&mut self) {
        let n = self.objects.len();
        let mut counts = vec![0u32; n];
        counts[self.root] = 1;
        for node in self.objects.iter().flatten() {
            if let ObjectKind::Group(g) = &node.kind {
                for link in &g.links {
                    if let LinkTarget::Hard(id) = link.target {
                        if id < n {
                            counts[id] += 1;
                        }
                    }
                }
            }
        }
        for (id, slot) in self.objects.iter_mut().enumerate() {
            if let Some(node) = slot {
                node.refcount = counts[id];
            }
        }
    }
}

/// The shared, interior-mutable state backing an open file.
pub(crate) struct FileInner {
    pub path: Option<PathBuf>,
    #[allow(dead_code)]
    pub mode: crate::hl::file::OpenMode,
    pub state: crate::lock::RwLock<FileState>,
    /// Synthetic id for this file handle.
    pub id: crate::h5i::hid_t,
    /// Files opened while following external links, keyed by the link's
    /// file-name string (kept open for the lifetime of this file).
    pub externals: crate::lock::Mutex<std::collections::HashMap<String, std::sync::Arc<FileInner>>>,
    /// MPI collective-file state (feature `mpi`).
    #[cfg(feature = "mpi")]
    pub mpi: crate::lock::Mutex<Option<crate::mpi::MpiFile>>,
}

/// The current time in seconds since the Unix epoch (0 on clock error).
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
pub(crate) fn now() -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as u32)
        .unwrap_or(0)
}

/// `wasm32-unknown-unknown` (the browser) has no clock — `SystemTime::now`
/// panics there — so object times are 0, as on a clock error.
#[cfg(all(target_family = "wasm", target_os = "unknown"))]
pub(crate) fn now() -> u32 {
    0
}
