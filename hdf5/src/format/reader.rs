//! Parsing of an HDF5 byte image into the in-memory model.
//!
//! Coverage is deliberately broader than the writer: superblock versions 0-3,
//! object header versions 1 and 2 (including continuation blocks), old-style
//! symbol-table groups (B-tree v1 + local heap + SNOD) and new-style compact
//! groups (link messages), contiguous/compact/chunked layouts (chunk B-tree
//! v1), the filter pipeline, and the global heap for variable-length data.

use std::collections::HashMap;

use crate::util::AddrMap;

use hdf5_types::TypeDescriptor;

use super::convert::{disk_size, has_vlen, VlenStore};
use super::datatype::OrderTree;
use super::source::{ByteSource, ImageSource, MemSource};
use super::v2::{read_v2btree, ExtensibleArray, FractalHeap};
use super::{datatype, filters as filt, Cursor, SIGNATURE, UNDEF};
use crate::error::Result;
use crate::hl::filters::Filter;
use crate::model::{
    AttrData, DatasetData, FileState, FillValue, GroupData, LayoutClass, Link, LinkTarget, ObjId,
    ObjectKind,
};

// Object-header message types.
const MSG_NIL: u16 = 0x0000;
const MSG_DATASPACE: u16 = 0x0001;
const MSG_LINK_INFO: u16 = 0x0002;
const MSG_DATATYPE: u16 = 0x0003;
const MSG_FILL_OLD: u16 = 0x0004;
const MSG_FILL: u16 = 0x0005;
const MSG_LINK: u16 = 0x0006;
const MSG_LAYOUT: u16 = 0x0008;
const MSG_GROUP_INFO: u16 = 0x000A;
const MSG_FILTER: u16 = 0x000B;
const MSG_ATTRIBUTE: u16 = 0x000C;
const MSG_COMMENT: u16 = 0x000D;
const MSG_MOD_TIME_OLD: u16 = 0x000E;
const MSG_CONTINUATION: u16 = 0x0010;
const MSG_SYMBOL_TABLE: u16 = 0x0011;
const MSG_SHMESG_TABLE: u16 = 0x000F;
const MSG_MOD_TIME: u16 = 0x0012;
const MSG_ATTRIBUTE_INFO: u16 = 0x0015;

/// Parsed dataspace: (dims, maxdims, is_scalar, is_null).
type ParsedDataspace = (Vec<u64>, Vec<Option<u64>>, bool, bool);

/// Everything extracted from one object header.
#[derive(Default)]
struct ObjHeader {
    dataspace: Option<ParsedDataspace>,
    dtype: Option<(TypeDescriptor, OrderTree)>,
    fill: FillValue,
    layout: Option<RawLayout>,
    filters: Vec<filt::RawFilter>,
    attrs: Vec<AttrData>,
    comment: Option<String>,
    symbol_table: Option<(u64, u64)>, // btree, heap
    links: Vec<Link>,
    is_group_like: bool,
    /// Dense link storage: (fractal heap addr, name-index v2 btree addr).
    dense_links: Option<(u64, u64)>,
    /// Dense attribute storage: (fractal heap addr, name-index v2 btree addr).
    dense_attrs: Option<(u64, u64)>,
    /// Object modification time (seconds since epoch).
    mtime: u32,
    /// Shared Message Table message: (master table address, num indexes).
    shmesg_table: Option<(u64, u8)>,
}

/// How the chunks of a chunked dataset are indexed on disk. Cheap to carry
/// (just addresses) so it can be stashed in [`crate::model::LazyData`] and the
/// index walked lazily, on first data access, instead of at open time.
#[derive(Clone, Debug)]
// Reachable through `model::LazyKind::Chunked::index`, whose enum is `pub`
// inside the private `mod model`. Declaring the leaf `pub(crate)` made that
// look like a visibility leak; nothing here is reachable outside the crate.
pub enum ChunkIndex {
    /// Version 1 B-tree (the "earliest" index, also written by this crate).
    BtreeV1 { addr: u64 },
    /// A single chunk covering the whole dataset (layout v4/v5, type 1).
    Single {
        addr: u64,
        filtered_size: Option<u32>,
        filter_mask: u32,
    },
    /// Implicit index: unfiltered chunks laid out contiguously (type 2).
    Implicit { addr: u64 },
    /// Fixed array index (type 3).
    FixedArray { header_addr: u64 },
    /// Extensible array index (type 4).
    ExtArray { header_addr: u64 },
    /// v2 B-tree index (type 5).
    BtreeV2 { header_addr: u64 },
}

enum RawLayout {
    Compact(Vec<u8>),
    Contiguous {
        addr: u64,
        size: u64,
    },
    Chunked {
        index: ChunkIndex,
        chunk_dims: Vec<u64>,
    },
    Virtual {
        gh_addr: u64,
        gh_idx: u32,
    },
}

pub fn parse(
    image: &std::sync::Arc<crate::model::FileImage>,
    dir: Option<&std::path::Path>,
) -> Result<FileState> {
    let src: std::sync::Arc<dyn ByteSource + Send + Sync> =
        std::sync::Arc::new(ImageSource::new(image.clone()));
    parse_source(src, dir, 0)
}

/// Parse from any [`ByteSource`] — a fully-present image or a lazy range-reader.
/// Eagerly parses the whole object hierarchy.
pub fn parse_source(
    src: std::sync::Arc<dyn ByteSource + Send + Sync>,
    dir: Option<&std::path::Path>,
    depth: usize,
) -> Result<FileState> {
    parse_source_opt(src, dir, depth, false)
}

/// Parse lazily: only the root object is read up front; every child becomes a
/// [`ObjectKind::Lazy`] stub parsed on first navigation. Opening a huge
/// many-object file (e.g. IBL NWB) is then O(root), and reading a single path
/// touches only the objects along it. Backed by a [`LazyLoader`] stashed in the
/// returned state.
pub fn parse_source_lazy(
    src: std::sync::Arc<dyn ByteSource + Send + Sync>,
    dir: Option<&std::path::Path>,
) -> Result<FileState> {
    parse_source_opt(src, dir, 0, true)
}

fn parse_source_opt(
    src: std::sync::Arc<dyn ByteSource + Send + Sync>,
    dir: Option<&std::path::Path>,
    depth: usize,
    lazy: bool,
) -> Result<FileState> {
    if depth > 8 {
        return Err("virtual dataset source nesting too deep".into());
    }
    // stored (base-relative) root object-header address
    let (root_oh_addr, base, ext_addr) = find_superblock(&*src)?;
    // a lazily opened file is read-only: its user block is never written back,
    // so it is not read (it can be any size, and is often most of a download)
    let userblock_data = if lazy {
        Vec::new()
    } else {
        src.slice(
            0,
            usize::try_from(base.min(src.len())).map_err(|_| "user block too large")?,
        )?
        .to_vec()
    };
    let mut state = FileState::new_empty();
    let mut memo: AddrMap<u64, ObjId> = AddrMap::default();
    let mut gheap: AddrMap<u64, HashMap<u32, Vec<u8>>> = AddrMap::default();
    // Assigned unconditionally at the end of the parser block below; the empty
    // Vec it used to start with was always overwritten before any read.
    let sohm_indexes: Vec<(u16, u64)>;
    {
        let mut p = Parser {
            src: src.clone(),
            base,
            state: &mut state,
            memo: &mut memo,
            gheap_cache: &mut gheap,
            pending_vds: Vec::new(),
            dir: dir.map(std::path::Path::to_path_buf),
            depth,
            sohm_indexes: Vec::new(),
            pending_lazy: None,
            lazy,
        };
        // Load the SOHM master table (if any) from the superblock extension
        // before parsing objects, so shared-message references resolve.
        if ext_addr != UNDEF {
            p.load_sohm_table(ext_addr)?;
        }
        p.state.userblock = base;
        p.state.userblock_data = userblock_data;
        // Replace the default-constructed root with the parsed one.
        let root = p.parse_object(root_oh_addr)?;
        p.state.root = root;
        // Materialize virtual datasets now that every source exists. (Lazy files
        // defer this: VDS sources are forced on access instead.)
        if !lazy {
            p.materialize_vds()?;
        }
        sohm_indexes = std::mem::take(&mut p.sohm_indexes);
    }
    // Keep the address -> id map: object references are addresses, and for an
    // eagerly-parsed file the parser's memo is the only record of where each
    // object came from.
    state.addrs = memo.clone();
    if !lazy {
        state.recount();
    } else {
        state.loader = Some(std::sync::Arc::new(LazyLoader {
            src,
            base,
            dir: dir.map(std::path::Path::to_path_buf),
            memo: std::sync::Mutex::new(memo),
            gheap: std::sync::Mutex::new(gheap),
            sohm_indexes,
        }));
    }
    Ok(state)
}

/// Everything needed to parse further objects of a lazily-opened file on demand.
pub struct LazyLoader {
    pub(crate) src: std::sync::Arc<dyn ByteSource + Send + Sync>,
    pub(crate) base: u64,
    pub(crate) dir: Option<std::path::PathBuf>,
    pub(crate) memo: std::sync::Mutex<AddrMap<u64, ObjId>>,
    pub(crate) gheap: std::sync::Mutex<AddrMap<u64, HashMap<u32, Vec<u8>>>>,
    pub(crate) sohm_indexes: Vec<(u16, u64)>,
}

impl std::fmt::Debug for LazyLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "LazyLoader {{ base: {} }}", self.base)
    }
}

/// Parse the object backing a [`ObjectKind::Lazy`] stub `id` in place, replacing
/// it with the real kind/attrs and recording *its* children as new lazy stubs.
pub(crate) fn force_object(state: &mut FileState, id: ObjId) -> Result<()> {
    let oh_addr = match &state.get(id).kind {
        crate::model::ObjectKind::Lazy { oh_addr } => *oh_addr,
        _ => return Ok(()),
    };
    let pending = with_parser(state, Vec::new(), |p| {
        p.parse_object_into(id, oh_addr)?;
        Ok(std::mem::take(&mut p.pending_vds))
    })?;
    if pending.is_empty() {
        return Ok(());
    }
    // A virtual dataset reads its sources now. Those in this file may be in
    // groups not parsed yet, and forcing them needs the loader the parser above
    // held, so it happens between the two.
    let sources = with_parser(state, Vec::new(), |p| {
        let mut sources = Vec::new();
        for (_, gh_addr, gh_idx) in &pending {
            let blob = p.read_gheap_object(*gh_addr, *gh_idx)?;
            for m in super::vds::parse_vds_blob(&blob)? {
                if m.source_file == "." {
                    sources.push(m.source_dset);
                }
            }
        }
        Ok(sources)
    })?;
    for path in sources {
        state.resolve_force(state.root, &path);
    }
    with_parser(state, pending, |p| p.materialize_vds())
}

/// Read a lazy stub's attributes from its object header, leaving it a stub:
/// its links, or its data, are the expensive part, and are not touched. Forcing
/// it later reads them again, with the rest.
pub(crate) fn peek_attrs(state: &mut FileState, id: ObjId) -> Result<()> {
    let crate::model::ObjectKind::Lazy { oh_addr } = state.get(id).kind else {
        return Ok(());
    };
    with_parser(state, Vec::new(), |p| {
        let mut oh = p.parse_object_header(oh_addr)?;
        if let Some((fheap, name_bt2)) = oh.dense_attrs {
            if fheap != UNDEF && name_bt2 != UNDEF {
                p.load_dense_attrs(fheap, name_bt2, &mut oh.attrs)?;
            }
        }
        p.state.get_mut(id).attrs = oh.attrs;
        Ok(())
    })
}

/// Run `f` with a parser over a lazily opened file's loader.
fn with_parser<T>(
    state: &mut FileState,
    pending_vds: Vec<(ObjId, u64, u32)>,
    f: impl FnOnce(&mut Parser) -> Result<T>,
) -> Result<T> {
    let loader = state
        .loader
        .clone()
        .ok_or("lazily-parsed object but the file has no loader")?;
    let mut memo = loader.memo.lock().unwrap();
    let mut gheap = loader.gheap.lock().unwrap();
    let mut p = Parser {
        src: loader.src.clone(),
        base: loader.base,
        state,
        memo: &mut memo,
        gheap_cache: &mut gheap,
        pending_vds,
        dir: loader.dir.clone(),
        depth: 0,
        sohm_indexes: loader.sohm_indexes.clone(),
        pending_lazy: None,
        lazy: true,
    };
    f(&mut p)
}

/// Resolve an object-header address to an object id, parsing the object if the
/// file has not walked to it yet.
///
/// This is what an HDF5 **object reference** needs: the reference token *is* the
/// address of an object header. MATLAB `-v7.3` stores cell arrays this way,
/// pointing into `/#refs#` — and a lazily-opened file may never have expanded
/// that group, so a plain map lookup is not enough.
pub(crate) fn object_at(state: &mut FileState, oh_addr: u64) -> Result<ObjId> {
    if oh_addr == UNDEF || oh_addr == 0 {
        return Err("object reference is undefined".into());
    }
    if let Some(&id) = state.addrs.get(&oh_addr) {
        return Ok(id);
    }
    if let Some(loader) = state.loader.clone() {
        let known = loader.memo.lock().unwrap().get(&oh_addr).copied();
        if let Some(id) = known {
            state.addrs.insert(oh_addr, id);
            return Ok(id);
        }
    }
    // Not reached yet. Only a lazy file can get here — an eager parse has
    // already visited everything reachable and recorded it above.
    let loader = state
        .loader
        .clone()
        .ok_or("object reference points outside the objects this file parsed")?;
    let id = state.alloc(crate::model::ObjectKind::Lazy { oh_addr });
    loader.memo.lock().unwrap().insert(oh_addr, id);
    state.addrs.insert(oh_addr, id);
    force_object(state, id)?;
    Ok(id)
}

/// Locate the superblock (it may sit after a userblock at offsets 512, 1024, ...)
/// and return (root object header address, base address, superblock
/// extension address).
fn find_superblock(src: &dyn ByteSource) -> Result<(u64, u64, u64)> {
    let mut offset = 0u64;
    loop {
        if offset + 96 > src.len() {
            return Err("not an HDF5 file (superblock signature not found)".into());
        }
        if *src.slice(offset, 8)? == SIGNATURE {
            break;
        }
        offset = if offset == 0 { 512 } else { offset * 2 };
    }
    let mut c = Cursor::at(src, offset + 8);
    let version = c.u8()?;
    match version {
        0 | 1 => {
            let _free_ver = c.u8()?;
            let _root_ver = c.u8()?;
            let _rsv = c.u8()?;
            let _shmsg_ver = c.u8()?;
            let size_off = c.u8()?;
            let size_len = c.u8()?;
            let _rsv = c.u8()?;
            if size_off != 8 || size_len != 8 {
                return Err(format!(
                    "unsupported offset/length sizes: {size_off}/{size_len} (only 8/8 supported)"
                )
                .into());
            }
            let _leaf_k = c.u16()?;
            let _internal_k = c.u16()?;
            let _flags = c.u32()?;
            if version == 1 {
                let _indexed_k = c.u16()?;
                let _rsv = c.u16()?;
            }
            let base = c.u64()?;
            let _free_addr = c.u64()?;
            let _eof = c.u64()?;
            let _driver = c.u64()?;
            // root group symbol table entry (stored base-relative)
            let _link_name_off = c.u64()?;
            let oh_addr = c.u64()?;
            Ok((oh_addr, base, UNDEF))
        }
        2 | 3 => {
            let size_off = c.u8()?;
            let size_len = c.u8()?;
            if size_off != 8 || size_len != 8 {
                return Err("unsupported offset/length sizes (only 8/8 supported)".into());
            }
            let _flags = c.u8()?;
            let base = c.u64()?;
            let ext_addr = c.u64()?;
            let _eof = c.u64()?;
            let root_oh = c.u64()?;
            let _checksum = c.u32()?;
            Ok((root_oh, base, ext_addr))
        }
        v => Err(format!("unsupported superblock version {v}").into()),
    }
}

struct Parser<'a> {
    src: std::sync::Arc<dyn ByteSource + Send + Sync>,
    base: u64,
    state: &'a mut FileState,
    memo: &'a mut AddrMap<u64, ObjId>,
    gheap_cache: &'a mut AddrMap<u64, HashMap<u32, Vec<u8>>>,
    /// Virtual datasets awaiting materialization: (arena id, heap addr, idx).
    pending_vds: Vec<(ObjId, u64, u32)>,
    /// Directory of the file being parsed (for VDS source resolution).
    dir: Option<std::path::PathBuf>,
    depth: usize,
    /// SOHM indexes: (message-type flag bits, fractal heap address).
    sohm_indexes: Vec<(u16, u64)>,
    /// Set by read_dataset_data when the dataset can stay lazily referenced.
    pending_lazy: Option<crate::model::LazyData>,
    /// When true, child objects are recorded as [`ObjectKind::Lazy`] stubs and
    /// parsed only on navigation (see [`force_object`]).
    lazy: bool,
}

impl Parser<'_> {
    /// Resolve a child object header: parse it now (eager) or stub it (lazy).
    fn child(&mut self, oh_addr: u64) -> Result<ObjId> {
        if self.lazy {
            Ok(self.lazy_child(oh_addr))
        } else {
            self.parse_object(oh_addr)
        }
    }

    /// Allocate (or reuse) a [`ObjectKind::Lazy`] stub for an unparsed object.
    fn lazy_child(&mut self, oh_addr: u64) -> ObjId {
        if let Some(&id) = self.memo.get(&oh_addr) {
            return id;
        }
        let id = self.state.alloc(crate::model::ObjectKind::Lazy { oh_addr });
        self.state.get_mut(id).refcount = 1;
        self.memo.insert(oh_addr, id);
        id
    }

    /// Parse the object at `oh_addr` into the existing arena slot `id` (used to
    /// force a lazy stub). Its own children are recorded as lazy stubs.
    fn parse_object_into(&mut self, id: ObjId, oh_addr: u64) -> Result<()> {
        match self.parse_object_inner(oh_addr) {
            Ok((kind, attrs, comment, mtime, vds_ref)) => {
                let node = self.state.get_mut(id);
                node.kind = kind;
                node.attrs = attrs;
                node.comment = comment;
                node.mtime = mtime;
                if let Some((gh_addr, gh_idx)) = vds_ref {
                    self.pending_vds.push((id, gh_addr, gh_idx));
                }
            }
            Err(e) => {
                self.state.get_mut(id).kind = crate::model::ObjectKind::Unsupported(e.to_string());
            }
        }
        Ok(())
    }
    /// Borrow the backing source as a trait object for `Cursor`/helper calls.
    #[inline]
    fn src(&self) -> &dyn ByteSource {
        &*self.src
    }

    fn addr(&self, a: u64) -> u64 {
        self.base + a
    }

    /// Parse the object at the given object-header address into the arena.
    ///
    /// Per-object failures do not abort the whole file: the object becomes
    /// [`ObjectKind::Unsupported`] and only erroring when accessed, matching
    /// libhdf5's open-on-demand behavior.
    fn parse_object(&mut self, oh_addr: u64) -> Result<ObjId> {
        if let Some(&id) = self.memo.get(&oh_addr) {
            return Ok(id);
        }
        // looked up once: reading the environment takes a process-wide lock
        static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *TRACE.get_or_init(|| std::env::var_os("HDF5_TRACE").is_some()) {
            static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n % 25 == 0 {
                eprintln!("[trace] {n} objects parsed");
            }
        }
        // Reserve an arena slot before recursing so cycles terminate.
        let placeholder = self.state.alloc(ObjectKind::Group(GroupData::default()));
        self.memo.insert(oh_addr, placeholder);
        match self.parse_object_inner(oh_addr) {
            Ok((kind, attrs, comment, mtime, vds_ref)) => {
                let node = self.state.get_mut(placeholder);
                node.kind = kind;
                node.attrs = attrs;
                node.comment = comment;
                node.mtime = mtime;
                if let Some((gh_addr, gh_idx)) = vds_ref {
                    self.pending_vds.push((placeholder, gh_addr, gh_idx));
                }
            }
            Err(e) => {
                self.state.get_mut(placeholder).kind = ObjectKind::Unsupported(e.to_string());
            }
        }
        Ok(placeholder)
    }

    #[allow(clippy::type_complexity)]
    fn parse_object_inner(
        &mut self,
        oh_addr: u64,
    ) -> Result<(
        ObjectKind,
        Vec<AttrData>,
        Option<String>,
        u32,
        Option<(u64, u32)>,
    )> {
        let mut oh = self.parse_object_header(oh_addr)?;
        let vds_ref = match &oh.layout {
            Some(RawLayout::Virtual { gh_addr, gh_idx }) => Some((*gh_addr, *gh_idx)),
            _ => None,
        };

        // Dense attributes: fetch attribute messages from the fractal heap
        // via the name-index v2 btree.
        if let Some((fheap, name_bt2)) = oh.dense_attrs {
            if fheap != UNDEF && name_bt2 != UNDEF {
                self.load_dense_attrs(fheap, name_bt2, &mut oh.attrs)?;
            }
        }

        let kind = if oh.symbol_table.is_some()
            || oh.is_group_like
            || !oh.links.is_empty()
            || oh.dense_links.is_some()
        {
            // group: old-style (symbol table), compact (link messages),
            // or dense (fractal heap + v2 btree)
            let mut group = GroupData::default();
            if let Some((btree, heap)) = oh.symbol_table {
                self.parse_symbol_table(btree, heap, &mut group)?;
            }
            for link in oh.links {
                group.links.push(link);
            }
            if let Some((fheap, name_bt2)) = oh.dense_links {
                if fheap != UNDEF && name_bt2 != UNDEF {
                    self.load_dense_links(fheap, name_bt2, &mut group)?;
                }
            }
            ObjectKind::Group(group)
        } else if let (Some((dtype, order)), Some(space)) = (&oh.dtype, &oh.dataspace) {
            if oh.layout.is_some() {
                let (dims, maxdims, is_scalar, is_null) = space.clone();
                let dtype = dtype.clone();
                let order = order.clone();
                let filters: Vec<Filter> = oh
                    .filters
                    .iter()
                    .filter_map(|rf| Filter::from_raw_parts(rf.id, &rf.cdata))
                    .collect();
                self.pending_lazy = None;
                let Some(raw_layout) = oh.layout.as_ref() else {
                    return Err("dataset object header lacks a layout message".into());
                };
                let (layout, mut data, mut vlen) = self.read_dataset_data(
                    raw_layout,
                    &dtype,
                    &dims,
                    &oh.filters,
                    is_scalar,
                    is_null,
                )?;
                let mut lazy = self.pending_lazy.take();
                if !order.is_none() {
                    if let Some(l) = lazy.take() {
                        if let OrderTree::SwapLeaf(w) = order {
                            // plain numbers stay lazy, swapped as they are read
                            lazy = Some(crate::model::LazyData { swap: w, ..l });
                        } else {
                            // anything else needs the eager postprocess below
                            data = l.materialize_bytes()?;
                        }
                    }
                }
                // Normalize big-endian data to little-endian.
                let mut fill = oh.fill.clone();
                if !order.is_none() {
                    let esize = disk_size(&dtype);
                    let n = if is_null {
                        0
                    } else if is_scalar {
                        1
                    } else {
                        dims.iter().product::<u64>() as usize
                    };
                    if lazy.is_none() {
                        apply_order_all(&order, &mut data, esize, n, &mut vlen);
                    }
                    if let FillValue::UserDefined(fv) = &mut fill {
                        if fv.len() >= esize {
                            let mut dummy = VlenStore::new();
                            apply_order(&order, &mut fv[..esize], &mut dummy);
                        }
                    }
                }
                ObjectKind::Dataset(Box::new(DatasetData {
                    dtype,
                    dims,
                    maxdims,
                    layout,
                    filters,
                    fill,
                    data,
                    vlen,
                    lazy,
                    is_scalar,
                    is_null,
                    refs: None,
                }))
            } else {
                ObjectKind::NamedType(dtype.clone())
            }
        } else if let Some((dtype, _)) = &oh.dtype {
            ObjectKind::NamedType(dtype.clone())
        } else {
            // A bare header with no recognizable class: treat as empty group.
            ObjectKind::Group(GroupData::default())
        };
        Ok((kind, oh.attrs, oh.comment, oh.mtime, vds_ref))
    }

    /// Fill in the data of virtual datasets by reading their source datasets
    /// (in this file or external files) and scattering per the mappings.
    fn materialize_vds(&mut self) -> Result<()> {
        let pending = std::mem::take(&mut self.pending_vds);
        if pending.is_empty() {
            return Ok(());
        }
        let mut src_cache: HashMap<String, FileState> = HashMap::new();
        for (obj, gh_addr, gh_idx) in pending {
            let blob = self.read_gheap_object(gh_addr, gh_idx)?;
            let mappings = super::vds::parse_vds_blob(&blob)?;
            if let ObjectKind::Dataset(dd) = &mut self.state.get_mut(obj).kind {
                dd.materialize()?;
            }
            let (dims, esize, mut data, fill) = {
                let d = self
                    .state
                    .dataset_data(obj)
                    .ok_or("pending VDS is not a dataset")?;
                (
                    d.dims.clone(),
                    disk_size(&d.dtype),
                    d.data.clone(),
                    d.fill.clone(),
                )
            };
            // start from the fill value where defined
            if let FillValue::UserDefined(fv) = &fill {
                if fv.len() >= esize {
                    for cell in data.chunks_mut(esize) {
                        cell.copy_from_slice(&fv[..esize]);
                    }
                }
            }
            for m in &mappings {
                // load the source dataset's dims and raw data
                let source = if m.source_file == "." {
                    let root = self.state.root;
                    if let Some(id) = self.state.resolve(root, &m.source_dset) {
                        if let ObjectKind::Dataset(dd) = &mut self.state.get_mut(id).kind {
                            dd.materialize()?;
                        }
                    }
                    self.state
                        .resolve(root, &m.source_dset)
                        .and_then(|id| self.state.dataset_data(id))
                        .map(|d| (d.dims.clone(), disk_size(&d.dtype), d.data.clone()))
                } else {
                    if !src_cache.contains_key(&m.source_file) {
                        let mut path = std::path::PathBuf::from(&m.source_file);
                        if path.is_relative() {
                            if let Some(dir) = &self.dir {
                                path = dir.join(path);
                            }
                        }
                        let bytes = std::fs::read(&path).map_err(|e| {
                            format!("unable to open VDS source '{}': {e}", m.source_file)
                        })?;
                        let img = std::sync::Arc::new(crate::model::FileImage::Bytes(bytes));
                        let vsrc: std::sync::Arc<dyn ByteSource + Send + Sync> =
                            std::sync::Arc::new(ImageSource::new(img));
                        let mut st = parse_source(vsrc, path.parent(), self.depth + 1)?;
                        st.materialize_all()?;
                        src_cache.insert(m.source_file.clone(), st);
                    }
                    let st = &src_cache[&m.source_file];
                    st.resolve(st.root, &m.source_dset)
                        .and_then(|id| st.dataset_data(id))
                        .map(|d| (d.dims.clone(), disk_size(&d.dtype), d.data.clone()))
                };
                let Some((sdims, sesize, sdata)) = source else {
                    return Err(format!(
                        "VDS source dataset '{}' not found in '{}'",
                        m.source_dset, m.source_file
                    )
                    .into());
                };
                if sesize != esize {
                    return Err("VDS source datatype size mismatch".into());
                }
                let virt_idx = m.virt_sel.linear_indices(&dims)?;
                let src_idx = m.src_sel.linear_indices(&sdims)?;
                for (v, s) in virt_idx.iter().zip(src_idx.iter()) {
                    let (v, s) = ((*v as usize) * esize, (*s as usize) * esize);
                    if v + esize <= data.len() && s + esize <= sdata.len() {
                        data[v..v + esize].copy_from_slice(&sdata[s..s + esize]);
                    }
                }
            }
            if let ObjectKind::Dataset(d) = &mut self.state.get_mut(obj).kind {
                d.data = data;
                d.layout = LayoutClass::Virtual(mappings);
            }
        }
        Ok(())
    }

    /// Load dense links (fractal heap + name-index v2 btree, record type 5).
    fn load_dense_links(&mut self, fheap: u64, name_bt2: u64, group: &mut GroupData) -> Result<()> {
        let src = self.src.clone();
        let heap = FractalHeap::parse(&*src, self.base, fheap)?;
        let btree = read_v2btree(&*src, self.base, name_bt2)?;
        if btree.btree_type != 5 {
            return Err("dense link index has unexpected btree type".into());
        }
        let mut links = Vec::new();
        for rec in &btree.records {
            // record: hash(4) + heap id (7 bytes)
            if rec.len() < 4 {
                continue;
            }
            let msg = heap.get(&rec[4..])?;
            if let Some(link) = self.parse_link_message(&msg)? {
                links.push(link);
            }
        }
        // btree order is by name hash; sort by creation order for stability
        links.sort_by_key(|l| l.creation_order);
        group.links.extend(links);
        Ok(())
    }

    /// Load dense attributes (fractal heap + name-index v2 btree, type 8).
    fn load_dense_attrs(
        &mut self,
        fheap: u64,
        name_bt2: u64,
        attrs: &mut Vec<AttrData>,
    ) -> Result<()> {
        let src = self.src.clone();
        let heap = FractalHeap::parse(&*src, self.base, fheap)?;
        let btree = read_v2btree(&*src, self.base, name_bt2)?;
        if btree.btree_type != 8 {
            return Err("dense attribute index has unexpected btree type".into());
        }
        for rec in &btree.records {
            // record: heap id (8) + message flags (1) + corder (4) + hash (4)
            if rec.len() < 9 {
                continue;
            }
            let flags = rec[8];
            if flags & 0x03 != 0 {
                return Err("shared attribute messages are not supported".into());
            }
            let msg = heap.get(&rec[..8])?;
            if let Some(attr) = self.parse_attribute(&msg)? {
                attrs.push(attr);
            }
        }
        Ok(())
    }

    /// Parse a v1 or v2 object header (with continuations) at `addr`.
    fn parse_object_header(&mut self, addr: u64) -> Result<ObjHeader> {
        let pos = self.addr(addr);
        if pos + 4 > self.src.len() {
            return Err("object header address out of bounds".into());
        }
        if self.src.slice(pos, 4)? == b"OHDR" {
            self.parse_object_header_v2(addr)
        } else {
            self.parse_object_header_v1(addr)
        }
    }

    fn parse_object_header_v1(&mut self, addr: u64) -> Result<ObjHeader> {
        let src = self.src.clone();
        let mut c = Cursor::at(&*src, self.addr(addr));
        let version = c.u8()?;
        if version != 1 {
            return Err(format!("unsupported object header version {version}").into());
        }
        let _rsv = c.u8()?;
        let num_msgs = c.u16()?;
        let _refcount = c.u32()?;
        let hdr_size = c.u32()? as usize;
        c.skip(4); // pad to 8-byte boundary

        let mut oh = ObjHeader::default();
        // Blocks to scan: (start, len). Continuations append more.
        let mut blocks = vec![(c.pos, hdr_size)];
        let mut msgs_read = 0u16;
        let mut bi = 0;
        while bi < blocks.len() {
            let (start, len) = blocks[bi];
            let mut mc = Cursor::at(&*src, start);
            let end = start + len as u64;
            while mc.pos + 8 <= end && msgs_read < num_msgs {
                let mtype = mc.u16()?;
                let msize = mc.u16()? as usize;
                let mflags = mc.u8()?;
                mc.skip(3);
                let body_start = mc.pos;
                if body_start + msize as u64 > end || body_start + msize as u64 > self.src.len() {
                    break;
                }
                let body = self.src.slice(body_start, msize)?.to_vec();
                self.handle_message(mtype, mflags, &body, &mut oh, &mut blocks)?;
                mc.seek(body_start + msize as u64);
                msgs_read += 1;
            }
            bi += 1;
        }
        Ok(oh)
    }

    fn parse_object_header_v2(&mut self, addr: u64) -> Result<ObjHeader> {
        let src = self.src.clone();
        let mut c = Cursor::at(&*src, self.addr(addr));
        let sig = c.take(4)?;
        if sig != b"OHDR" {
            return Err("bad OHDR signature".into());
        }
        let version = c.u8()?;
        if version != 2 {
            return Err(format!("unsupported OHDR version {version}").into());
        }
        let flags = c.u8()?;
        if flags & 0x20 != 0 {
            let _access_time = c.u32()?;
            let _mod_time = c.u32()?;
            let _change_time = c.u32()?;
            let _birth_time = c.u32()?;
        }
        if flags & 0x10 != 0 {
            let _max_compact = c.u16()?;
            let _min_dense = c.u16()?;
        }
        let size_bytes = 1usize << (flags & 0x03);
        let chunk0_size = c.uint(size_bytes)? as usize;
        let track_order = flags & 0x04 != 0;

        let mut oh = ObjHeader {
            is_group_like: false,
            ..ObjHeader::default()
        };
        let _ = track_order;

        // message block: chunk0 (ends before 4-byte checksum)
        let mut blocks = vec![(c.pos, chunk0_size)];
        let creation_order_tracked = flags & 0x04 != 0;
        let mut bi = 0;
        while bi < blocks.len() {
            let (start, len) = blocks[bi];
            let mut mc = Cursor::at(&*src, start);
            let end = start + len as u64;
            // continuation blocks begin with the OCHK signature
            if bi > 0 {
                let sig = mc.take(4)?;
                if sig != b"OCHK" {
                    return Err("bad OCHK signature".into());
                }
            }
            let payload_end = if bi == 0 { end } else { end - 4 }; // trailing checksum in OCHK
            loop {
                // v2 message header: type(1), size(2), flags(1), [creation order(2)]
                let hdr_len = if creation_order_tracked { 6 } else { 4 };
                if mc.pos + hdr_len > payload_end {
                    break;
                }
                let mtype = mc.u8()? as u16;
                let msize = mc.u16()? as usize;
                let mflags = mc.u8()?;
                if creation_order_tracked {
                    let _order = mc.u16()?;
                }
                let body_start = mc.pos;
                if body_start + msize as u64 > payload_end
                    || body_start + msize as u64 > self.src.len()
                {
                    break;
                }
                let body = self.src.slice(body_start, msize)?.to_vec();
                self.handle_message(mtype, mflags, &body, &mut oh, &mut blocks)?;
                mc.seek(body_start + msize as u64);
            }
            bi += 1;
        }
        Ok(oh)
    }

    fn handle_message(
        &mut self,
        mtype: u16,
        mflags: u8,
        body: &[u8],
        oh: &mut ObjHeader,
        blocks: &mut Vec<(u64, usize)>,
    ) -> Result<()> {
        // Message flag bit 1: the body is a *shared message* reference, not
        // the message itself (H5Oshared.c).
        if mflags & 0x02 != 0 {
            return self.handle_shared_message(mtype, body, oh);
        }
        let body_src = MemSource::new(body);
        let mut c = Cursor::new(&body_src);
        match mtype {
            MSG_NIL => {}
            MSG_DATASPACE => {
                oh.dataspace = Some(parse_dataspace(&mut c)?);
            }
            MSG_DATATYPE => {
                let mut order = OrderTree::None;
                let desc = datatype::decode_ordered(&mut c, &mut order)?;
                oh.dtype = Some((desc, order));
            }
            MSG_FILL_OLD => {
                // old fill value: size(u32) + bytes
                let size = c.u32()? as usize;
                if size > 0 {
                    let v = c.take(size)?.to_vec();
                    oh.fill = FillValue::UserDefined(v);
                }
            }
            MSG_FILL => {
                let version = c.u8()?;
                if version <= 2 {
                    let _alloc_time = c.u8()?;
                    let _fill_time = c.u8()?;
                    let defined = c.u8()?;
                    if version < 2 || defined != 0 {
                        let size = c.u32()? as usize;
                        if size > 0 {
                            let v = c.take(size)?.to_vec();
                            oh.fill = FillValue::UserDefined(v);
                        } else if defined != 0 {
                            oh.fill = FillValue::Default;
                        }
                    }
                } else {
                    // version 3: flags byte
                    let flags = c.u8()?;
                    let defined = flags & 0x20 != 0;
                    if defined {
                        let size = c.u32()? as usize;
                        let v = c.take(size)?.to_vec();
                        oh.fill = if v.is_empty() {
                            FillValue::Default
                        } else {
                            FillValue::UserDefined(v)
                        };
                    }
                }
            }
            MSG_LAYOUT => {
                oh.layout = Some(parse_layout(&mut c)?);
            }
            MSG_FILTER => {
                oh.filters = parse_filter_pipeline(&mut c)?;
            }
            MSG_ATTRIBUTE => {
                if let Some(attr) = self.parse_attribute(body)? {
                    oh.attrs.push(attr);
                }
            }
            MSG_COMMENT => {
                let mut name = Vec::new();
                while let Ok(b) = c.u8() {
                    if b == 0 {
                        break;
                    }
                    name.push(b);
                }
                oh.comment = Some(String::from_utf8_lossy(&name).into_owned());
            }
            MSG_CONTINUATION => {
                let addr = c.u64()?;
                let len = c.u64()? as usize;
                blocks.push((self.addr(addr), len));
            }
            MSG_SYMBOL_TABLE => {
                let btree = c.u64()?;
                let heap = c.u64()?;
                oh.symbol_table = Some((btree, heap));
            }
            MSG_LINK_INFO => {
                oh.is_group_like = true;
                // Link Info: version, flags, [max corder], fheap addr,
                // name-index btree addr, [corder btree addr]
                let _version = c.u8()?;
                let flags = c.u8()?;
                if flags & 0x01 != 0 {
                    let _max_corder = c.u64()?;
                }
                let fheap = c.u64()?;
                let name_bt2 = c.u64()?;
                if fheap != UNDEF && name_bt2 != UNDEF {
                    oh.dense_links = Some((fheap, name_bt2));
                }
            }
            MSG_GROUP_INFO => {
                oh.is_group_like = true;
            }
            MSG_LINK => {
                if let Some(link) = self.parse_link_message(body)? {
                    oh.links.push(link);
                }
            }
            MSG_ATTRIBUTE_INFO => {
                // Attribute Info: version, flags, [max creation index (2)],
                // fheap addr, name-index btree addr, [corder btree addr]
                let _version = c.u8()?;
                let flags = c.u8()?;
                if flags & 0x01 != 0 {
                    let _max_index = c.u16()?;
                }
                let fheap = c.u64()?;
                let name_bt2 = c.u64()?;
                if fheap != UNDEF && name_bt2 != UNDEF {
                    oh.dense_attrs = Some((fheap, name_bt2));
                }
            }
            MSG_MOD_TIME => {
                let version = c.u8()?;
                if version == 1 {
                    c.skip(3);
                    oh.mtime = c.u32()?;
                }
            }
            MSG_SHMESG_TABLE => {
                // Shared Message Table: version, master table addr, nindexes
                let _version = c.u8()?;
                let table_addr = c.u64()?;
                let nindexes = c.u8()?;
                oh.shmesg_table = Some((table_addr, nindexes));
            }
            MSG_MOD_TIME_OLD => {
                // 14-byte ASCII YYYYMMDDhhmmss -- parse best-effort; ignore
                // failures (the field is informational only).
            }
            _ => {
                // Unknown/unneeded message: skip.
            }
        }
        Ok(())
    }

    /// Parse the superblock extension's Shared Message Table message and the
    /// SOHM master table (`SMTB`) it points to, recording each index's
    /// message-type flags and fractal heap address.
    fn load_sohm_table(&mut self, ext_addr: u64) -> Result<()> {
        let ext = match self.parse_object_header(ext_addr) {
            Ok(oh) => oh,
            // extensions can hold messages we don't parse; ignore failures
            Err(_) => return Ok(()),
        };
        let Some((table_addr, nindexes)) = ext.shmesg_table else {
            return Ok(());
        };
        if table_addr == UNDEF || nindexes == 0 {
            return Ok(());
        }
        let src = self.src.clone();
        let mut c = Cursor::at(&*src, self.addr(table_addr));
        let sig = c.take(4)?;
        if sig != b"SMTB" {
            return Err("bad SMTB signature".into());
        }
        for _ in 0..nindexes {
            let _list_version = c.u8()?;
            let _index_type = c.u8()?; // 1 = list, 2 = v2 btree (write-side only)
            let mesg_types = c.u16()?;
            let _min_mesg_size = c.u32()?;
            let _list_max = c.u16()?;
            let _btree_min = c.u16()?;
            let _num_messages = c.u16()?;
            let _index_addr = c.addr()?;
            let heap_addr = c.addr()?;
            self.sohm_indexes.push((mesg_types, heap_addr));
        }
        Ok(())
    }

    /// Resolve a shared-message reference (H5Oshared.c layout):
    /// v1: version, 1 skipped, 6 reserved, 8 skipped (heap addr), OH addr;
    /// v2: version, type, OH addr; v3: version, type, then OH addr for
    /// committed messages or a SOHM heap id (unsupported).
    fn handle_shared_message(&mut self, mtype: u16, body: &[u8], oh: &mut ObjHeader) -> Result<()> {
        let body_src = MemSource::new(body);
        let mut c = Cursor::new(&body_src);
        let version = c.u8()?;
        let share_type = c.u8()?; // ignored (forced to COMMITTED) for v1
        let oh_addr = match version {
            1 => {
                c.skip(6); // reserved
                c.skip(8); // local heap address (unused)
                c.u64()?
            }
            2 => c.u64()?,
            3 => {
                if share_type == 1 {
                    // SOHM: the body holds a fractal heap ID into the SOHM
                    // index heap covering this message type
                    let heap_id = c.take(8)?.to_vec();
                    let flag = 1u16 << (mtype & 0x0f);
                    let heap_addr = self
                        .sohm_indexes
                        .iter()
                        .find(|(types, _)| types & flag != 0)
                        .map(|(_, addr)| *addr)
                        .ok_or("no SOHM index covers this message type")?;
                    if heap_addr == UNDEF {
                        return Err("SOHM index heap not allocated".into());
                    }
                    let heap = FractalHeap::parse(self.src(), self.base, heap_addr)?;
                    let body = heap.get(&heap_id)?;
                    let mut blocks = Vec::new();
                    return self.handle_message(mtype, 0, &body, oh, &mut blocks);
                }
                c.u64()?
            }
            v => return Err(format!("unsupported shared message version {v}").into()),
        };
        // Fetch the real message from the referenced object header.
        let target = self.parse_object_header(oh_addr)?;
        match mtype {
            MSG_DATATYPE => {
                if target.dtype.is_none() {
                    return Err("shared datatype target has no datatype message".into());
                }
                oh.dtype = target.dtype;
                Ok(())
            }
            MSG_DATASPACE => {
                oh.dataspace = target.dataspace;
                Ok(())
            }
            MSG_FILL => {
                oh.fill = target.fill;
                Ok(())
            }
            MSG_FILTER => {
                oh.filters = target.filters;
                Ok(())
            }
            t => Err(format!("shared message of type {t:#06x} is not supported").into()),
        }
    }

    fn parse_link_message(&mut self, body: &[u8]) -> Result<Option<Link>> {
        let body_src = MemSource::new(body);
        let mut c = Cursor::new(&body_src);
        let version = c.u8()?;
        if version != 1 {
            return Ok(None);
        }
        let flags = c.u8()?;
        let link_type = if flags & 0x08 != 0 { c.u8()? } else { 0 };
        let creation_order = if flags & 0x04 != 0 {
            c.u64()? as i64
        } else {
            0
        };
        let charset = if flags & 0x10 != 0 { c.u8()? } else { 0 };
        let len_size = 1usize << (flags & 0x03);
        let name_len = c.uint(len_size)? as usize;
        let name = String::from_utf8_lossy(c.take(name_len)?).into_owned();
        let target = match link_type {
            0 => {
                let oh_addr = c.u64()?;
                let child = self.child(oh_addr)?;
                LinkTarget::Hard(child)
            }
            1 => {
                let len = c.u16()? as usize;
                let path = String::from_utf8_lossy(c.take(len)?).into_owned();
                LinkTarget::Soft(path)
            }
            64 => {
                let len = c.u16()? as usize;
                let blob = c.take(len)?;
                // external link: version/flags byte, then file\0path\0
                let mut parts = blob[1..].split(|&b| b == 0);
                let file = String::from_utf8_lossy(parts.next().unwrap_or(&[])).into_owned();
                let path = String::from_utf8_lossy(parts.next().unwrap_or(&[])).into_owned();
                LinkTarget::External { file, path }
            }
            _ => return Ok(None),
        };
        Ok(Some(Link {
            name,
            target,
            creation_order,
            utf8: charset == 1,
        }))
    }

    /// Parse an old-style group: B-tree v1 of SNODs + local heap.
    fn parse_symbol_table(&mut self, btree: u64, heap: u64, group: &mut GroupData) -> Result<()> {
        let heap_data = self.parse_local_heap(heap)?;
        let mut snods = Vec::new();
        self.walk_group_btree(btree, &mut snods)?;
        // reach every node first: a source fetching as it goes then learns them
        // all at once, not one per attempt
        let mut failed = None;
        for addr in &snods {
            if let Err(e) = self.src.slice(self.addr(*addr), 8) {
                failed.get_or_insert(e);
            }
        }
        if let Some(e) = failed {
            return Err(e);
        }
        for snod_addr in snods {
            let src = self.src.clone();
            let mut c = Cursor::at(&*src, self.addr(snod_addr));
            let sig = c.take(4)?;
            if sig != b"SNOD" {
                return Err("bad SNOD signature".into());
            }
            let _ver = c.u8()?;
            let _rsv = c.u8()?;
            let nsyms = c.u16()?;
            for _ in 0..nsyms {
                let name_off = c.u64()? as usize;
                let oh_addr = c.u64()?;
                let cache_type = c.u32()?;
                let _rsv = c.u32()?;
                let scratch = c.take(16)?;
                let name = read_heap_string(&heap_data, name_off)?;
                let target = if cache_type == 2 {
                    let val_off =
                        u32::from_le_bytes([scratch[0], scratch[1], scratch[2], scratch[3]])
                            as usize;
                    LinkTarget::Soft(read_heap_string(&heap_data, val_off)?)
                } else {
                    let child = self.child(oh_addr)?;
                    LinkTarget::Hard(child)
                };
                let order = self.state.next_order();
                group.links.push(Link {
                    name,
                    target,
                    creation_order: order,
                    utf8: false,
                });
            }
        }
        Ok(())
    }

    fn walk_group_btree(&mut self, addr: u64, out: &mut Vec<u64>) -> Result<()> {
        self.walk_group_btree_at(addr, None, out)
    }

    /// One node and below. Each child must sit exactly one level lower: a
    /// damaged file whose child points back up the tree (a cycle) is then an
    /// error, not unbounded recursion — levels are a `u8`, so the depth is too.
    fn walk_group_btree_at(
        &mut self,
        addr: u64,
        expect: Option<u8>,
        out: &mut Vec<u64>,
    ) -> Result<()> {
        if addr == UNDEF {
            return Ok(());
        }
        let src = self.src.clone();
        let mut c = Cursor::at(&*src, self.addr(addr));
        let sig = c.take(4)?;
        if sig != b"TREE" {
            return Err("bad TREE signature".into());
        }
        let node_type = c.u8()?;
        if node_type != 0 {
            return Err("expected group B-tree node".into());
        }
        let level = c.u8()?;
        if expect.is_some_and(|l| l != level) {
            return Err("group B-tree node at the wrong level (damaged file)".into());
        }
        let entries = c.u16()? as usize;
        let _left = c.u64()?;
        let _right = c.u64()?;
        let _key0 = c.u64()?;
        let mut failed = None;
        for _ in 0..entries {
            let child = c.u64()?;
            let _key = c.u64()?;
            if level == 0 {
                out.push(child);
            } else if let Err(e) = self.walk_group_btree_at(child, Some(level - 1), out) {
                failed.get_or_insert(e); // as walk_chunk_btree
            }
        }
        failed.map_or(Ok(()), Err)
    }

    fn parse_local_heap(&mut self, addr: u64) -> Result<Vec<u8>> {
        let src = self.src.clone();
        let mut c = Cursor::at(&*src, self.addr(addr));
        let sig = c.take(4)?;
        if sig != b"HEAP" {
            return Err("bad HEAP signature".into());
        }
        let _ver = c.u8()?;
        c.skip(3);
        let seg_size = c.u64()?;
        let _free_off = c.u64()?;
        let seg_addr = c.u64()?;
        let start = self.addr(seg_addr);
        if start + seg_size > self.src.len() {
            return Err("local heap out of bounds".into());
        }
        let seg_size = usize::try_from(seg_size).map_err(|_| "local heap too large")?;
        Ok(self.src.slice(start, seg_size)?.to_vec())
    }

    /// Materialize a dataset's raw bytes from its layout. Vlen heap references
    /// are rewritten into `{len, store_idx}` model slots with payloads loaded
    /// into the returned side store.
    fn read_dataset_data(
        &mut self,
        layout: &RawLayout,
        dtype: &TypeDescriptor,
        dims: &[u64],
        raw_filters: &[filt::RawFilter],
        is_scalar: bool,
        is_null: bool,
    ) -> Result<(LayoutClass, Vec<u8>, VlenStore)> {
        let elem_size = disk_size(dtype);
        let total_elems: u64 = if is_null {
            0
        } else if is_scalar {
            1
        } else {
            let mut n: u64 = 1;
            for &d in dims {
                n = n.checked_mul(d).ok_or("dataset dimension overflow")?;
            }
            n
        };
        let logical_size = total_elems
            .checked_mul(elem_size as u64)
            .ok_or("dataset size overflow")?;
        // the size stays u64 until something is allocated: a dataset larger than
        // this target's address space can still be opened, shaped and read in part
        // Asked only where the parse itself would allocate the dataset: data
        // kept lazily (a 183 GB recording, read a few rows at a time) has no
        // size limit here. Sanity bound: never materialize more than 64 GiB
        // from a parse (a corrupt size field must error, not abort).
        let in_memory = || {
            if logical_size > 1 << 36 {
                return Err(format!(
                    "dataset too large to materialize ({logical_size} bytes)"
                ));
            }
            usize::try_from(logical_size)
                .map_err(|_| format!("dataset too large to load here ({logical_size} bytes)"))
        };

        let (class, mut raw) = match layout {
            RawLayout::Compact(bytes) => {
                let mut v = bytes.clone();
                v.resize(in_memory()?, 0);
                (LayoutClass::Compact, v)
            }
            RawLayout::Contiguous { addr, size } => {
                if *addr == UNDEF || logical_size == 0 {
                    (LayoutClass::Contiguous, vec![0u8; in_memory()?])
                } else {
                    let start = self.addr(*addr);
                    // as libhdf5 (H5D__contig_check): allocated storage must
                    // hold the whole dataspace — a damaged dataspace claiming
                    // billions of elements is caught here, not decoded
                    if *size < logical_size {
                        return Err(format!(
                            "contiguous storage ({size} bytes) is smaller than the dataspace ({logical_size} bytes)"
                        )
                        .into());
                    }
                    let n = (*size).min(logical_size);
                    if start.checked_add(n).is_none_or(|e| e > self.src.len()) {
                        return Err("contiguous data out of bounds".into());
                    }
                    if raw_filters.is_empty() && !has_vlen(dtype) {
                        // defer the copy: reference the file image directly
                        self.pending_lazy = Some(crate::model::LazyData {
                            src: self.src.clone(),
                            base: self.base,
                            logical: logical_size,
                            resolved_chunks: Default::default(),
                            cache: Default::default(),
                            swap: 0,
                            kind: crate::model::LazyKind::Contiguous {
                                offset: start,
                                len: n,
                            },
                        });
                        (LayoutClass::Contiguous, Vec::new())
                    } else {
                        let n = usize::try_from(n).map_err(|_| "dataset too large to load here")?;
                        let mut v = self.src.slice(start, n)?.to_vec();
                        v.resize(in_memory()?, 0);
                        (LayoutClass::Contiguous, v)
                    }
                }
            }
            RawLayout::Chunked { index, chunk_dims } => {
                // chunk_dims includes the trailing element-size dim; drop it
                let cd: Vec<u64> = chunk_dims[..chunk_dims.len().saturating_sub(1)].to_vec();
                // as libhdf5: one chunk extent per dataset axis, none zero (a
                // damaged layout would otherwise reach indexing by axis)
                if cd.len() != dims.len() || cd.contains(&0) {
                    return Err(format!(
                        "chunk shape {cd:?} does not fit a dataset of rank {}",
                        dims.len()
                    )
                    .into());
                }
                // only N-bit decodes here; everything else defers, and must not
                // allocate the whole dataset just to be opened
                let eager = raw_filters.iter().any(|f| f.id == filt::FILTER_NBIT);
                let mut data = if eager {
                    vec![0u8; in_memory()?]
                } else {
                    Vec::new()
                };
                self.read_chunked(
                    index,
                    dims,
                    &cd,
                    elem_size,
                    raw_filters,
                    logical_size,
                    &mut data,
                )?;
                // Vlen elements are resolved against the global heap below, which
                // needs the chunk bytes NOW. A deferred chunk read leaves `data`
                // zeroed, so every vlen string decoded as "" (h5py writes any
                // resizable string dataset — e.g. EEGUnity's epoch_meta — chunked).
                if has_vlen(dtype) {
                    if let Some(l) = self.pending_lazy.take() {
                        data = l.materialize_bytes()?;
                    }
                }
                (LayoutClass::Chunked(cd), data)
            }
            RawLayout::Virtual { .. } => {
                // mappings + data filled in later by materialize_vds()
                (LayoutClass::Virtual(Vec::new()), vec![0u8; in_memory()?])
            }
        };

        // N-bit datasets store only `precision` bits per element; libhdf5's
        // datatype conversion re-extends the sign when reading into a wider
        // native type. Replicate that for signed integers.
        if let Some(nb) = raw_filters.iter().find(|f| f.id == filt::FILTER_NBIT) {
            if matches!(dtype, TypeDescriptor::Integer(_)) && nb.cdata.get(1) == Some(&0) {
                let (precision, offset) = (
                    nb.cdata.get(6).copied().unwrap_or(0) as usize,
                    nb.cdata.get(7).copied().unwrap_or(0) as usize,
                );
                if precision > 0 && precision + offset <= elem_size * 8 && offset == 0 {
                    for cell in raw.chunks_exact_mut(elem_size) {
                        let mut b = [0u8; 8];
                        b[..elem_size].copy_from_slice(cell);
                        let mut v = u64::from_le_bytes(b);
                        if precision < 64 && v & (1 << (precision - 1)) != 0 {
                            v |= u64::MAX << precision; // sign-extend
                        }
                        cell.copy_from_slice(&v.to_le_bytes()[..elem_size]);
                    }
                }
            }
        }

        // Load vlen payloads into a side store and normalize the slots.
        let mut store = VlenStore::new();
        if has_vlen(dtype) && !raw.is_empty() {
            for cell in raw.chunks_exact_mut(elem_size) {
                self.load_vlen_elem(dtype, cell, &mut store)?;
            }
        }

        Ok((class, raw, store))
    }

    /// Rewrite one element's on-disk vlen references `{len, gcol_addr, idx}`
    /// into model slots `{len, store_idx, 0}`, loading payloads into `store`.
    fn load_vlen_elem(
        &mut self,
        dtype: &TypeDescriptor,
        elem: &mut [u8],
        store: &mut VlenStore,
    ) -> Result<()> {
        match dtype {
            TypeDescriptor::VarLenAscii
            | TypeDescriptor::VarLenUnicode
            | TypeDescriptor::VarLenArray(_) => {
                let len = u32::from_le_bytes(elem[0..4].try_into().unwrap());
                let addr = u64::from_le_bytes(elem[4..12].try_into().unwrap());
                let idx = u32::from_le_bytes(elem[12..16].try_into().unwrap());
                let bytes = if addr == 0 || addr == UNDEF || idx == 0 {
                    Vec::new()
                } else {
                    self.read_gheap_object(addr, idx)?
                };
                let store_idx = if bytes.is_empty() && len == 0 {
                    0u32
                } else {
                    store.push(bytes);
                    store.len() as u32
                };
                elem[0..4].copy_from_slice(&len.to_le_bytes());
                elem[4..8].copy_from_slice(&store_idx.to_le_bytes());
                elem[8..16].fill(0);
            }
            TypeDescriptor::Compound(c) => {
                for f in &c.fields {
                    let fs = disk_size(&f.ty);
                    if f.offset + fs <= elem.len() {
                        self.load_vlen_elem(&f.ty, &mut elem[f.offset..f.offset + fs], store)?;
                    }
                }
            }
            TypeDescriptor::FixedArray(base, n) => {
                let bs = disk_size(base);
                for i in 0..*n {
                    self.load_vlen_elem(base, &mut elem[i * bs..(i + 1) * bs], store)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn read_chunked(
        &mut self,
        index: &ChunkIndex,
        dims: &[u64],
        chunk_dims: &[u64],
        elem_size: usize,
        raw_filters: &[filt::RawFilter],
        logical: u64,
        out: &mut [u8],
    ) -> Result<()> {
        // NBit needs post-decode sign extension in the parser, so it must
        // resolve + decode eagerly. Everything else defers the *entire* index
        // walk to first data access: opening a file never enumerates chunks, so
        // metadata reads stay O(object header) even for datasets with millions
        // of chunks (the giant Neuropixels/IBL case).
        if raw_filters.iter().any(|f| f.id == filt::FILTER_NBIT) {
            let chunks = resolve_chunks(self.src(), self.base, index, dims, chunk_dims, elem_size)?;
            return materialize_chunks(
                self.src(),
                self.base,
                &chunks,
                dims,
                chunk_dims,
                elem_size,
                raw_filters,
                out,
            );
        }
        self.pending_lazy = Some(crate::model::LazyData {
            src: self.src.clone(),
            base: self.base,
            logical,
            resolved_chunks: Default::default(),
            cache: Default::default(),
            swap: 0,
            kind: crate::model::LazyKind::Chunked {
                index: index.clone(),
                dims: dims.to_vec(),
                chunk_dims: chunk_dims.to_vec(),
                elem_size,
                filters: raw_filters.to_vec(),
            },
        });
        Ok(())
    }

    /// Parse a Fixed Array chunk index (FAHD header + FADB data block),
    /// returning one `(chunk_addr, filtered_size, filter_mask)` per chunk in
    /// row-major order.
    fn read_fixed_array(
        src: &dyn ByteSource,
        base: u64,
        header_addr: u64,
    ) -> Result<Vec<(u64, Option<u32>, u32)>> {
        let mut c = Cursor::at(src, base + header_addr);
        let sig = c.take(4)?;
        if sig != b"FAHD" {
            return Err("bad FAHD signature".into());
        }
        let _ver = c.u8()?;
        let client_id = c.u8()?; // 0 = chunks, 1 = filtered chunks
        let entry_size = c.u8()? as usize;
        let page_bits = c.u8()? as usize;
        let max_num_entries = c.u64()? as usize;
        let data_block_addr = c.u64()?;
        let _checksum = c.u32()?;
        if data_block_addr == UNDEF {
            return Ok(vec![(UNDEF, None, 0); max_num_entries]);
        }

        let mut d = Cursor::at(src, base + data_block_addr);
        let sig = d.take(4)?;
        if sig != b"FADB" {
            return Err("bad FADB signature".into());
        }
        let _ver = d.u8()?;
        let _client = d.u8()?;
        let _hdr_addr = d.u64()?;
        let page_size = 1usize << page_bits;
        let paged = max_num_entries > page_size;
        let mut entries = Vec::with_capacity((max_num_entries).min(1 << 16));
        if paged {
            // page bitmap + checksum, then pages of elements each followed by
            // their own checksum
            let npages = (max_num_entries).div_ceil(page_size);
            let bitmap_bytes = npages.div_ceil(8);
            d.skip(bitmap_bytes);
            let _checksum = d.u32()?;
            for p in 0..npages {
                let in_page = page_size.min(max_num_entries - p * page_size);
                for _ in 0..in_page {
                    entries.push(parse_fa_entry(&mut d, client_id, entry_size)?);
                }
                let _page_checksum = d.u32()?;
            }
        } else {
            for _ in 0..max_num_entries {
                entries.push(parse_fa_entry(&mut d, client_id, entry_size)?);
            }
        }
        Ok(entries)
    }

    /// Collect a v1 chunk B-tree's chunks, or with `rows = (r0, r1, chunk_rows)`
    /// only those that can hold rows `[r0, r1)`. Keys sort by offset, the first
    /// dimension most significant, and each child's chunks lie between its key
    /// and the next, so children past the rows are never read.
    fn walk_chunk_btree(
        src: &dyn ByteSource,
        base: u64,
        addr: u64,
        rank: usize,
        rows: Option<(u64, u64, u64)>,
        out: &mut Vec<ChunkEntry>,
    ) -> Result<()> {
        Self::walk_chunk_btree_at(src, base, addr, rank, rows, None, out)
    }

    /// One node and below; each child exactly one level lower (see
    /// [`Self::walk_group_btree_at`]).
    fn walk_chunk_btree_at(
        src: &dyn ByteSource,
        base: u64,
        addr: u64,
        rank: usize,
        rows: Option<(u64, u64, u64)>,
        expect: Option<u8>,
        out: &mut Vec<ChunkEntry>,
    ) -> Result<()> {
        if addr == UNDEF {
            return Ok(());
        }
        let mut c = Cursor::at(src, base + addr);
        let sig = c.take(4)?;
        if sig != b"TREE" {
            return Err("bad chunk TREE signature".into());
        }
        let node_type = c.u8()?;
        if node_type != 1 {
            return Err("expected chunk B-tree node".into());
        }
        let level = c.u8()?;
        if expect.is_some_and(|l| l != level) {
            return Err("chunk B-tree node at the wrong level (damaged file)".into());
        }
        let entries = c.u16()? as usize;
        let _left = c.u64()?;
        let _right = c.u64()?;
        let read_key = |c: &mut Cursor| -> Result<(u32, u32, Vec<u64>)> {
            let size = c.u32()?;
            let mask = c.u32()?;
            let mut offsets = Vec::with_capacity((rank + 1).min(1 << 16));
            for _ in 0..=rank {
                offsets.push(c.u64()?);
            }
            Ok((size, mask, offsets))
        };
        let mut node = Vec::with_capacity(entries.min(1 << 16));
        for _ in 0..entries {
            let key = read_key(&mut c)?;
            node.push((key, c.u64()?));
        }
        if level == 0 {
            out.extend(node.into_iter().map(|((s, m, o), child)| (o, s, m, child)));
            return Ok(());
        }
        // the key after the last child bounds it
        let last = match rows {
            Some(_) => Some(read_key(&mut c)?.2),
            None => None,
        };
        // on, past a child that fails: a source that fails on bytes it has yet to
        // fetch then learns all it is missing in one walk, not one per child
        let mut failed = None;
        for i in 0..node.len() {
            if let Some((r0, r1, chunk_rows)) = rows {
                let from = node[i].0 .2.first().copied().unwrap_or(0);
                let to = match node.get(i + 1) {
                    Some(next) => next.0 .2.first(),
                    None => last.as_ref().and_then(|k| k.first()),
                };
                let to = to.copied().unwrap_or(u64::MAX);
                if from >= r1 || to.saturating_add(chunk_rows) <= r0 {
                    continue;
                }
            }
            if let Err(e) =
                Self::walk_chunk_btree_at(src, base, node[i].1, rank, rows, Some(level - 1), out)
            {
                failed.get_or_insert(e);
            }
        }
        failed.map_or(Ok(()), Err)
    }

    /// Fetch one global-heap object by collection address and index.
    fn read_gheap_object(&mut self, addr: u64, idx: u32) -> Result<Vec<u8>> {
        if !self.gheap_cache.contains_key(&addr) {
            let col = self.parse_gheap_collection(addr)?;
            self.gheap_cache.insert(addr, col);
        }
        Ok(self.gheap_cache[&addr]
            .get(&idx)
            .cloned()
            .unwrap_or_default())
    }

    fn parse_gheap_collection(&mut self, addr: u64) -> Result<HashMap<u32, Vec<u8>>> {
        let src = self.src.clone();
        let mut c = Cursor::at(&*src, self.addr(addr));
        let sig = c.take(4)?;
        if sig != b"GCOL" {
            return Err("bad GCOL signature".into());
        }
        let _ver = c.u8()?;
        c.skip(3);
        let col_size = c.u64()?;
        let end = self.addr(addr) + col_size;
        let mut map = HashMap::new();
        while c.pos + 16 <= end {
            let idx = c.u16()?;
            let _refcount = c.u16()?;
            c.skip(4);
            let size = c.u64()? as usize;
            if idx == 0 {
                break;
            }
            let data = c.take(size)?.to_vec();
            // objects padded to 8
            let pad = (8 - (size % 8)) % 8;
            c.skip(pad);
            map.insert(idx as u32, data);
        }
        Ok(map)
    }

    fn parse_attribute(&mut self, body: &[u8]) -> Result<Option<AttrData>> {
        let body_src = MemSource::new(body);
        let mut c = Cursor::new(&body_src);
        let version = c.u8()?;
        match version {
            1 => {
                let _rsv = c.u8()?;
                let name_size = c.u16()? as usize;
                let dt_size = c.u16()? as usize;
                let ds_size = c.u16()? as usize;
                let name_raw = c.take(super::align8(name_size))?;
                let name = cstr(&name_raw[..name_size]);
                let dt_start = c.pos;
                let mut dt_cur = Cursor::at(&body_src, dt_start);
                let mut order = OrderTree::None;
                let dtype = datatype::decode_ordered(&mut dt_cur, &mut order)?;
                c.seek(dt_start + (super::align8(dt_size)) as u64);
                let ds_start = c.pos;
                let mut ds_cur = Cursor::at(&body_src, ds_start);
                let (dims, _maxdims, is_scalar, is_null) = parse_dataspace(&mut ds_cur)?;
                c.seek(ds_start + (super::align8(ds_size)) as u64);
                if c.pos > (body.len()) as u64 {
                    return Err("attribute message truncated".into());
                }
                let data = body[c.pos as usize..].to_vec(); // within body, checked above
                self.finish_attribute(name, dtype, order, dims, is_scalar, is_null, data)
            }
            2 | 3 => {
                let _flags = c.u8()?;
                let name_size = c.u16()? as usize;
                let dt_size = c.u16()? as usize;
                let ds_size = c.u16()? as usize;
                if version == 3 {
                    let _charset = c.u8()?;
                }
                let name = cstr(c.take(name_size)?);
                let dt_start = c.pos;
                let mut dt_cur = Cursor::at(&body_src, dt_start);
                let mut order = OrderTree::None;
                let dtype = datatype::decode_ordered(&mut dt_cur, &mut order)?;
                c.seek(dt_start + dt_size as u64);
                let ds_start = c.pos;
                let mut ds_cur = Cursor::at(&body_src, ds_start);
                let (dims, _maxdims, is_scalar, is_null) = parse_dataspace(&mut ds_cur)?;
                c.seek(ds_start + ds_size as u64);
                if c.pos > (body.len()) as u64 {
                    return Err("attribute message truncated".into());
                }
                let data = body[c.pos as usize..].to_vec(); // within body, checked above
                self.finish_attribute(name, dtype, order, dims, is_scalar, is_null, data)
            }
            _ => Ok(None),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_attribute(
        &mut self,
        name: String,
        dtype: TypeDescriptor,
        order: OrderTree,
        dims: Vec<u64>,
        is_scalar: bool,
        is_null: bool,
        mut data: Vec<u8>,
    ) -> Result<Option<AttrData>> {
        let n = if is_null {
            0
        } else if is_scalar {
            1
        } else {
            let mut n: u64 = 1;
            for &d in &dims {
                n = n.checked_mul(d).ok_or("attribute dimension overflow")?;
            }
            n as usize
        };
        let elem_size = disk_size(&dtype);
        let logical = n.checked_mul(elem_size).ok_or("attribute size overflow")?;
        if logical as u64 > 1 << 33 {
            return Err("attribute too large to materialize".into());
        }
        data.truncate(logical);
        data.resize(logical, 0);
        let mut store = VlenStore::new();
        if has_vlen(&dtype) && !data.is_empty() {
            for i in 0..n {
                let off = i * elem_size;
                self.load_vlen_elem(&dtype, &mut data[off..off + elem_size], &mut store)?;
            }
        }
        if !order.is_none() {
            apply_order_all(&order, &mut data, elem_size, n, &mut store);
        }
        Ok(Some(AttrData {
            name,
            dtype,
            dims,
            is_scalar,
            is_null,
            data,
            vlen: store,
            refs: None,
        }))
    }
}

/// Parse one Fixed Array element: for client id 0 a plain chunk address; for
/// client id 1 an address plus stored size and filter mask.
fn parse_fa_entry(
    d: &mut Cursor,
    client_id: u8,
    entry_size: usize,
) -> Result<(u64, Option<u32>, u32)> {
    if client_id == 0 {
        let addr = d.u64()?;
        Ok((addr, None, 0))
    } else {
        // entry = address (8) + chunk size (entry_size - 12) + filter mask (4)
        let addr = d.u64()?;
        let size_len = entry_size.saturating_sub(12).max(1);
        let size = d.uint(size_len)? as u32;
        let mask = d.u32()?;
        Ok((addr, Some(size), mask))
    }
}

/// Enumerate the base offsets of every chunk in row-major order.
fn chunk_grid(dims: &[u64], chunk_dims: &[u64]) -> Vec<Vec<u64>> {
    let rank = dims.len();
    if rank == 0 {
        return vec![vec![]];
    }
    let nchunks: Vec<u64> = dims
        .iter()
        .zip(chunk_dims)
        .map(|(&d, &c)| ((d).div_ceil(c)).max(1))
        .collect();
    let total: u64 = nchunks.iter().product();
    let mut out = Vec::with_capacity((total as usize).min(1 << 16));
    let mut idx = vec![0u64; rank];
    for _ in 0..total {
        out.push(idx.iter().zip(chunk_dims).map(|(&i, &c)| i * c).collect());
        for k in (0..rank).rev() {
            idx[k] += 1;
            if idx[k] < nchunks[k] {
                break;
            }
            idx[k] = 0;
        }
    }
    out
}

/// [`apply_order`] over the first `n` elements of `data`; plain numbers in one
/// vectorized pass.
fn apply_order_all(
    order: &OrderTree,
    data: &mut [u8],
    esize: usize,
    n: usize,
    store: &mut VlenStore,
) {
    let end = (n * esize).min(data.len());
    let data = &mut data[..end];
    match order {
        OrderTree::SwapLeaf(w) if *w == esize => super::simd::swap_bytes(esize, data),
        _ => {
            for elem in data.chunks_exact_mut(esize.max(1)) {
                apply_order(order, elem, store);
            }
        }
    }
}

/// Byte-swap one element (already in model layout, vlen slots normalized)
/// from big-endian to little-endian, per its order tree. Vlen payloads in the
/// side store are swapped in place.
fn apply_order(order: &OrderTree, elem: &mut [u8], store: &mut VlenStore) {
    match order {
        OrderTree::None => {}
        OrderTree::SwapLeaf(w) => {
            if *w <= elem.len() {
                elem[..*w].reverse();
            }
        }
        OrderTree::Compound(fields) => {
            for (off, sub) in fields {
                if *off < elem.len() {
                    apply_order(sub, &mut elem[*off..], store);
                }
            }
        }
        OrderTree::Array { n, stride, inner } => {
            for i in 0..*n {
                let s = i * stride;
                if s + stride <= elem.len() {
                    apply_order(inner, &mut elem[s..s + stride], store);
                }
            }
        }
        OrderTree::VarLen { base_stride, inner } => {
            // model slot: {len u32, store_idx u32, reserved}
            if elem.len() >= 8 {
                let idx = u32::from_le_bytes(elem[4..8].try_into().unwrap()) as usize;
                if idx > 0 && idx <= store.len() {
                    // take the entry out to keep the borrow checker happy with
                    // (rare) nested vlen recursion
                    let mut entry = std::mem::take(&mut store[idx - 1]);
                    let mut i = 0;
                    while (i + 1) * base_stride <= entry.len() {
                        apply_order(
                            inner,
                            &mut entry[i * base_stride..(i + 1) * base_stride],
                            store,
                        );
                        i += 1;
                    }
                    store[idx - 1] = entry;
                }
            }
        }
    }
}

fn cstr(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// Read a NUL-terminated string out of a local heap data segment.
fn read_heap_string(heap: &[u8], offset: usize) -> Result<String> {
    if offset >= heap.len() {
        return Err("heap string offset out of bounds".into());
    }
    Ok(cstr(&heap[offset..]))
}

fn parse_dataspace(c: &mut Cursor) -> Result<ParsedDataspace> {
    let version = c.u8()?;
    match version {
        1 => {
            let rank = c.u8()? as usize;
            let flags = c.u8()?;
            c.skip(5); // reserved
            let mut dims = Vec::with_capacity((rank).min(1 << 16));
            for _ in 0..rank {
                dims.push(c.u64()?);
            }
            let maxdims = if flags & 1 != 0 {
                let mut m = Vec::with_capacity((rank).min(1 << 16));
                for _ in 0..rank {
                    let v = c.u64()?;
                    m.push(if v == UNDEF { None } else { Some(v) });
                }
                m
            } else {
                dims.iter().map(|&d| Some(d)).collect()
            };
            Ok((dims, maxdims, rank == 0, false))
        }
        2 => {
            let rank = c.u8()? as usize;
            let flags = c.u8()?;
            let space_type = c.u8()?;
            let mut dims = Vec::with_capacity((rank).min(1 << 16));
            for _ in 0..rank {
                dims.push(c.u64()?);
            }
            let maxdims = if flags & 1 != 0 {
                let mut m = Vec::with_capacity((rank).min(1 << 16));
                for _ in 0..rank {
                    let v = c.u64()?;
                    m.push(if v == UNDEF { None } else { Some(v) });
                }
                m
            } else {
                dims.iter().map(|&d| Some(d)).collect()
            };
            Ok((dims, maxdims, space_type == 0, space_type == 2))
        }
        v => Err(format!("unsupported dataspace version {v}").into()),
    }
}

fn parse_layout(c: &mut Cursor) -> Result<RawLayout> {
    let version = c.u8()?;
    match version {
        1 | 2 => {
            let rank = c.u8()? as usize;
            let class = c.u8()?;
            c.skip(5); // reserved
            match class {
                0 => {
                    // compact: dims then size+data
                    let mut _dims = Vec::new();
                    for _ in 0..rank {
                        _dims.push(c.u32()?);
                    }
                    let size = c.u32()? as usize;
                    Ok(RawLayout::Compact(c.take(size)?.to_vec()))
                }
                1 => {
                    let addr = c.u64()?;
                    for _ in 0..rank {
                        let _ = c.u32()?;
                    }
                    Ok(RawLayout::Contiguous {
                        addr,
                        size: u64::MAX,
                    })
                }
                2 => {
                    let btree = c.u64()?;
                    let mut chunk_dims = Vec::with_capacity((rank).min(1 << 16));
                    for _ in 0..rank {
                        chunk_dims.push(c.u32()? as u64);
                    }
                    Ok(RawLayout::Chunked {
                        index: ChunkIndex::BtreeV1 { addr: btree },
                        chunk_dims,
                    })
                }
                _ => Err("unknown layout class".into()),
            }
        }
        3 => {
            let class = c.u8()?;
            match class {
                0 => {
                    let size = c.u16()? as usize;
                    Ok(RawLayout::Compact(c.take(size)?.to_vec()))
                }
                1 => {
                    let addr = c.u64()?;
                    let size = c.u64()?;
                    Ok(RawLayout::Contiguous { addr, size })
                }
                2 => {
                    let ndims = c.u8()? as usize;
                    let btree = c.u64()?;
                    let mut chunk_dims = Vec::with_capacity((ndims).min(1 << 16));
                    for _ in 0..ndims {
                        chunk_dims.push(c.u32()? as u64);
                    }
                    Ok(RawLayout::Chunked {
                        index: ChunkIndex::BtreeV1 { addr: btree },
                        chunk_dims,
                    })
                }
                _ => Err("unknown layout class".into()),
            }
        }
        // versions 4 and 5 share the same structure (v5 is written by HDF5 2.0)
        4 | 5 => {
            let class = c.u8()?;
            match class {
                0 => {
                    let size = c.u16()? as usize;
                    Ok(RawLayout::Compact(c.take(size)?.to_vec()))
                }
                3 => {
                    // virtual: global heap reference to the mapping blob
                    let gh_addr = c.u64()?;
                    let gh_idx = c.u32()?;
                    Ok(RawLayout::Virtual { gh_addr, gh_idx })
                }
                1 => {
                    let addr = c.u64()?;
                    let size = c.u64()?;
                    Ok(RawLayout::Contiguous { addr, size })
                }
                2 => {
                    let flags = c.u8()?;
                    let ndims = c.u8()? as usize;
                    let enc_size = c.u8()? as usize;
                    let mut chunk_dims = Vec::with_capacity((ndims).min(1 << 16));
                    for _ in 0..ndims {
                        chunk_dims.push(c.uint(enc_size)?);
                    }
                    let index_type = c.u8()?;
                    let index = match index_type {
                        1 => {
                            // single chunk; when filtered (flags bit 1), the
                            // stored size and filter mask precede the address
                            if flags & 0x02 != 0 {
                                let size = c.uint(8)? as u32;
                                let mask = c.u32()?;
                                let addr = c.u64()?;
                                ChunkIndex::Single {
                                    addr,
                                    filtered_size: Some(size),
                                    filter_mask: mask,
                                }
                            } else {
                                let addr = c.u64()?;
                                ChunkIndex::Single {
                                    addr,
                                    filtered_size: None,
                                    filter_mask: 0,
                                }
                            }
                        }
                        2 => {
                            let addr = c.u64()?;
                            ChunkIndex::Implicit { addr }
                        }
                        3 => {
                            let _page_bits = c.u8()?;
                            let header_addr = c.u64()?;
                            ChunkIndex::FixedArray { header_addr }
                        }
                        4 => {
                            // extensible array: 5 creation-parameter bytes
                            // (all present in the EA header too), then addr
                            c.skip(5);
                            let header_addr = c.u64()?;
                            ChunkIndex::ExtArray { header_addr }
                        }
                        5 => {
                            // v2 btree: node size (4) + split (1) + merge (1)
                            // (all present in the BTHD header too), then addr
                            c.skip(6);
                            let header_addr = c.u64()?;
                            ChunkIndex::BtreeV2 { header_addr }
                        }
                        t => return Err(format!("unknown chunk index type {t}").into()),
                    };
                    Ok(RawLayout::Chunked { index, chunk_dims })
                }
                _ => Err("unknown layout class".into()),
            }
        }
        v => Err(format!("unsupported layout version {v}").into()),
    }
}

/// Public wrapper used by the fractal-heap reader for filtered heaps.
pub(crate) fn parse_filter_pipeline_public(c: &mut Cursor) -> Result<Vec<filt::RawFilter>> {
    parse_filter_pipeline(c)
}

fn parse_filter_pipeline(c: &mut Cursor) -> Result<Vec<filt::RawFilter>> {
    let version = c.u8()?;
    let nfilters = c.u8()? as usize;
    if version == 1 {
        c.skip(2);
        c.skip(4);
    }
    let mut filters = Vec::with_capacity((nfilters).min(1 << 16));
    for _ in 0..nfilters {
        let id = c.u16()?;
        let name_len = if version == 1 || id >= 256 {
            c.u16()? as usize
        } else {
            0
        };
        let _flags = c.u16()?;
        let nvals = c.u16()? as usize;
        let name = if name_len > 0 {
            let raw = c.take(name_len)?;
            cstr(raw)
        } else {
            String::new()
        };
        let mut cdata = Vec::with_capacity((nvals).min(1 << 16));
        for _ in 0..nvals {
            cdata.push(c.u32()?);
        }
        if version == 1 && nvals % 2 != 0 {
            c.skip(4);
        }
        filters.push(filt::RawFilter { id, cdata, name });
    }
    Ok(filters)
}

/// The chunks a read of rows `[r0, r1)` needs from a v1 B-tree index, walking only
/// that part of it; `None` for the other indexes, which are resolved whole.
#[allow(clippy::type_complexity)]
pub(crate) fn resolve_chunk_rows(
    src: &dyn ByteSource,
    base: u64,
    index: &ChunkIndex,
    rank: usize,
    (r0, r1, chunk_rows): (u64, u64, u64),
) -> Result<Option<Vec<ChunkEntry>>> {
    let ChunkIndex::BtreeV1 { addr } = index else {
        return Ok(None);
    };
    let mut chunks = Vec::new();
    Parser::walk_chunk_btree(
        src,
        base,
        *addr,
        rank,
        Some((r0, r1, chunk_rows)),
        &mut chunks,
    )?;
    Ok(Some(chunks))
}

/// Walk a chunk index into the full `(scaled_offsets, stored_size, mask, addr)`
/// chunk list. Deferred to first data access (called from
/// [`crate::model::LazyData::materialize_bytes`]) so opening a file never walks
/// the — possibly enormous — chunk index.
/// A walked chunk-index entry: `(offsets, stored size, filter mask, address)`.
pub(crate) type ChunkEntry = (Vec<u64>, u32, u32, u64);

pub(crate) fn resolve_chunks(
    src: &dyn ByteSource,
    base: u64,
    index: &ChunkIndex,
    dims: &[u64],
    chunk_dims: &[u64],
    elem_size: usize,
) -> Result<Vec<ChunkEntry>> {
    let rank = dims.len();
    let chunk_elems: usize = chunk_dims.iter().map(|&c| c as usize).product();
    let chunk_bytes = chunk_elems * elem_size;
    let mut chunks: Vec<ChunkEntry> = Vec::new();
    match index {
        ChunkIndex::BtreeV1 { addr } => {
            if *addr != UNDEF {
                Parser::walk_chunk_btree(src, base, *addr, rank, None, &mut chunks)?;
            }
        }
        ChunkIndex::Single {
            addr,
            filtered_size,
            filter_mask,
        } => {
            if *addr != UNDEF {
                let size = filtered_size.unwrap_or(chunk_bytes as u32);
                chunks.push((vec![0u64; rank], size, *filter_mask, *addr));
            }
        }
        ChunkIndex::Implicit { addr } => {
            if *addr != UNDEF {
                for (i, offsets) in chunk_grid(dims, chunk_dims).into_iter().enumerate() {
                    chunks.push((
                        offsets,
                        chunk_bytes as u32,
                        0,
                        addr + (i * chunk_bytes) as u64,
                    ));
                }
            }
        }
        ChunkIndex::FixedArray { header_addr } => {
            if *header_addr != UNDEF {
                let entries = Parser::read_fixed_array(src, base, *header_addr)?;
                for (offsets, entry) in chunk_grid(dims, chunk_dims).into_iter().zip(entries) {
                    let (addr, size, mask) = entry;
                    if addr == UNDEF || addr == 0 {
                        continue; // unallocated chunk
                    }
                    let size = size.unwrap_or(chunk_bytes as u32);
                    chunks.push((offsets, size, mask, addr));
                }
            }
        }
        ChunkIndex::ExtArray { header_addr } => {
            if *header_addr != UNDEF {
                let ea = ExtensibleArray::parse(src, base, *header_addr)?;
                for (i, offsets) in chunk_grid(dims, chunk_dims).into_iter().enumerate() {
                    let Some(elem) = ea.element(i as u64)? else {
                        continue;
                    };
                    // client 0: addr(8); client 1: addr(8)+size+mask(4)
                    let elem_src = MemSource::new(&elem);
                    let mut ec = Cursor::new(&elem_src);
                    let addr = ec.addr()?;
                    if addr == UNDEF || addr == 0 {
                        continue;
                    }
                    let (size, mask) = if ea.elmt_size > 8 {
                        let size_len = ea.elmt_size - 8 - 4;
                        let size = ec.uint(size_len)? as u32;
                        let mask = ec.u32()?;
                        (size, mask)
                    } else {
                        (chunk_bytes as u32, 0)
                    };
                    chunks.push((offsets, size, mask, addr));
                }
            }
        }
        ChunkIndex::BtreeV2 { header_addr } => {
            if *header_addr != UNDEF {
                let bt = read_v2btree(src, base, *header_addr)?;
                for rec in &bt.records {
                    let rec_src = MemSource::new(rec);
                    let mut rc = Cursor::new(&rec_src);
                    let addr = rc.addr()?;
                    let (size, mask) = match bt.btree_type {
                        10 => (chunk_bytes as u32, 0),
                        11 => {
                            // filtered: chunk size length = record len minus
                            // addr(8) + mask(4) + scaled offsets
                            let size_len = rec
                                .len()
                                .checked_sub(8 + 4 + rank * 8)
                                .ok_or("bad v2 btree chunk record size")?;
                            let size = rc.uint(size_len)? as u32;
                            let mask = rc.u32()?;
                            (size, mask)
                        }
                        t => return Err(format!("unexpected chunk btree record type {t}").into()),
                    };
                    // scaled offsets (chunk offset / chunk dim), rank of them
                    let mut offsets = Vec::with_capacity((rank + 1).min(1 << 16));
                    for &dim in chunk_dims.iter().take(rank) {
                        offsets.push(rc.u64()? * dim);
                    }
                    offsets.push(0);
                    if addr != UNDEF && addr != 0 {
                        chunks.push((offsets, size, mask, addr));
                    }
                }
            }
        }
    }
    Ok(chunks)
}

/// Bytes in one chunk, `Π chunk_dims · elem_size`, computed in `u64` so a
/// 32-bit target (wasm32) neither truncates a dimension nor overflows the
/// product; capped at 2³³ bytes and at what this target can address.
fn chunk_byte_size(chunk_dims: &[u64], elem_size: usize) -> Result<usize> {
    let bytes = chunk_dims
        .iter()
        .try_fold(elem_size as u64, |acc, &c| acc.checked_mul(c))
        .filter(|&b| b <= 1 << 33)
        .ok_or("chunk too large")?;
    usize::try_from(bytes).map_err(|_| "chunk too large for this target".into())
}
/// Decode and scatter a chunk list into a logical dataset buffer. Shared by
/// the eager parse path and lazy materialization ([`crate::model::LazyData`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn materialize_chunks(
    src: &dyn ByteSource,
    base: u64,
    chunks: &[ChunkEntry],
    dims: &[u64],
    chunk_dims: &[u64],
    elem_size: usize,
    raw_filters: &[filt::RawFilter],
    out: &mut [u8],
) -> Result<()> {
    materialize_chunks_with(
        src,
        base,
        chunks,
        dims,
        chunk_dims,
        elem_size,
        raw_filters,
        out,
        None,
    )
}

/// [`materialize_chunks`], placing elements through `tf`: `out` then holds
/// elements of the transform's size.
#[allow(clippy::too_many_arguments)]
pub(crate) fn materialize_chunks_with(
    src: &dyn ByteSource,
    base: u64,
    chunks: &[ChunkEntry],
    dims: &[u64],
    chunk_dims: &[u64],
    elem_size: usize,
    raw_filters: &[filt::RawFilter],
    out: &mut [u8],
    tf: Transform,
) -> Result<()> {
    touch_chunks(src, base, chunks)?;
    let chunk_bytes = chunk_byte_size(chunk_dims, elem_size)?;
    if dims.is_empty() {
        // a scalar: one chunk of one element
        for chunk in decode_batch(src, base, chunks, elem_size, raw_filters, chunk_bytes, None)? {
            scatter_chunk(&chunk, dims, chunk_dims, &[], elem_size, out, tf);
        }
        return Ok(());
    }
    chunk_rows_into(
        src,
        base,
        chunks,
        dims,
        chunk_dims,
        elem_size,
        raw_filters,
        (0, dims[0]),
        out,
        None,
        tf,
    )
}

/// Rows out of unfiltered chunks: a chunk is stored row-major, so the rows
/// wanted are one run of its bytes, and only that run is read. A chunk can be
/// millions of rows.
#[allow(clippy::too_many_arguments)]
fn plain_chunk_rows(
    src: &dyn ByteSource,
    base: u64,
    chunks: &[ChunkEntry],
    dims: &[u64],
    chunk_dims: &[u64],
    elem_size: usize,
    (r0, r1): (u64, u64),
    odims: &[u64],
    out: &mut [u8],
) -> Result<()> {
    let rank = dims.len();
    let inner = chunk_dims[1..].iter().product::<u64>() * elem_size as u64;
    let runs: Vec<_> = chunks
        .iter()
        .filter_map(|(offsets, size, _, addr)| {
            let lo = offsets[0].max(r0);
            let hi = (offsets[0] + chunk_dims[0]).min(r1).min(dims[0]);
            let start = (lo - offsets[0]) * inner;
            let len = (hi - lo) * inner;
            // past what is stored, as a short final chunk can be: read as zeros
            (lo < hi && start + len <= u64::from(*size)).then_some((
                offsets,
                lo,
                hi,
                base + addr + start,
                len,
            ))
        })
        .collect();
    let mut failed = None;
    for (.., at, len) in &runs {
        if let Err(e) = src.slice(*at, *len as usize) {
            failed.get_or_insert(e); // as touch_chunks
        }
    }
    if let Some(e) = failed {
        return Err(e);
    }
    for (offsets, lo, hi, at, len) in runs {
        let sub = src.slice(at, len as usize)?;
        let mut sub_cd = chunk_dims.to_vec();
        sub_cd[0] = hi - lo;
        let mut sub_base = offsets[..rank].to_vec();
        sub_base[0] = lo - r0;
        scatter_chunk(sub, odims, &sub_cd, &sub_base, elem_size, out, None);
    }
    Ok(())
}

/// Chunks decoded in one batch: bounds what is held at once.
const DECODE_BATCH: usize = 64;

/// One chunk's logical bytes: borrowed from the source, decoded here, or
/// shared with the dataset's [`ChunkCache`].
pub(crate) enum Decoded<'a> {
    Borrowed(&'a [u8]),
    Owned(Vec<u8>),
    Shared(std::sync::Arc<[u8]>),
}

impl std::ops::Deref for Decoded<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Self::Borrowed(b) => b,
            Self::Owned(v) => v,
            Self::Shared(a) => a,
        }
    }
}

/// One batch of chunks' logical bytes (each `chunk_bytes`, zero-padded), in
/// order: decoded through the filters on every core, borrowed straight from
/// the source when unfiltered and whole. With a `cache`, decoded chunks are
/// taken from it when there and put in it when not.
fn decode_batch<'a>(
    src: &'a dyn ByteSource,
    base: u64,
    batch: &[ChunkEntry],
    elem_size: usize,
    raw_filters: &[filt::RawFilter],
    chunk_bytes: usize,
    cache: Option<&crate::cache::ChunkCache>,
) -> Result<Vec<Decoded<'a>>> {
    let mut out: Vec<Option<Decoded<'a>>> = Vec::with_capacity(batch.len());
    let mut stored = Vec::with_capacity(batch.len());
    for (i, (_, size, mask, addr)) in batch.iter().enumerate() {
        let start = base + addr;
        if let Some(hit) = cache
            .filter(|_| !raw_filters.is_empty())
            .and_then(|c| c.get(start))
        {
            out.push(Some(Decoded::Shared(hit)));
            continue;
        }
        if start
            .checked_add(u64::from(*size))
            .is_none_or(|e| e > src.len())
        {
            return Err("chunk data out of bounds".into());
        }
        let bytes = src.slice(start, *size as usize)?;
        if raw_filters.is_empty() {
            out.push(Some(if bytes.len() >= chunk_bytes {
                Decoded::Borrowed(bytes)
            } else {
                let mut chunk = Vec::with_capacity(chunk_bytes);
                chunk.extend_from_slice(bytes);
                chunk.resize(chunk_bytes, 0);
                Decoded::Owned(chunk)
            }));
        } else {
            out.push(None);
            stored.push((i, start, bytes, *mask));
        }
    }
    let decoded = crate::par::map(&stored, |(_, _, bytes, mask)| {
        filt::reverse_sized(raw_filters, elem_size, bytes, *mask, chunk_bytes).map(|mut c| {
            c.resize(chunk_bytes, 0);
            c
        })
    });
    for ((i, start, ..), chunk) in stored.iter().zip(decoded) {
        let chunk = chunk?;
        out[*i] = Some(match cache {
            Some(cache) => {
                let shared: std::sync::Arc<[u8]> = chunk.into();
                cache.put(*start, shared.clone());
                Decoded::Shared(shared)
            }
            None => Decoded::Owned(chunk),
        });
    }
    Ok(out
        .into_iter()
        .map(|c| c.expect("every chunk placed"))
        .collect())
}

/// Decode `chunks` and place rows `[r0, r1)` of the dataset (`dims`, rank ≥
/// 1) into `out`, which holds just those rows.
///
/// Per batch, chunks decode in parallel; they are then scattered in
/// parallel too, by *row band*: chunks sharing leading-axis rows go to one
/// worker, and `out` is split at band edges, so every worker owns the bytes
/// it writes.
#[allow(clippy::too_many_arguments)]
fn chunk_rows_into(
    src: &dyn ByteSource,
    base: u64,
    chunks: &[ChunkEntry],
    dims: &[u64],
    chunk_dims: &[u64],
    elem_size: usize,
    raw_filters: &[filt::RawFilter],
    (r0, r1): (u64, u64),
    out: &mut [u8],
    cache: Option<&crate::cache::ChunkCache>,
    tf: Transform,
) -> Result<()> {
    let rank = dims.len();
    let chunk_bytes = chunk_byte_size(chunk_dims, elem_size)?;
    let out_es = tf.map_or(elem_size, |(size, _)| size);
    let row_bytes = dims[1..].iter().product::<u64>() as usize * out_es;
    let inner = chunk_dims[1..].iter().product::<u64>() as usize * elem_size;
    if row_bytes == 0 || r0 >= r1 {
        return Ok(());
    }
    // the dataset rows a chunk contributes
    let rows = |offsets: &[u64]| {
        let lo = offsets[0].max(r0);
        let hi = (offsets[0] + chunk_dims[0]).min(r1).min(dims[0]);
        (lo, hi.max(lo))
    };
    // place one chunk's rows into `band`, which holds the rows from `row`
    let place = |offsets: &[u64], chunk: &[u8], band: &mut [u8], row: u64| {
        let (lo, hi) = rows(offsets);
        let skip = (lo - offsets[0]) as usize;
        let keep = (hi - lo) as usize;
        let Some(sub) = chunk.get(skip * inner..(skip + keep) * inner) else {
            return;
        };
        let mut bdims = dims.to_vec();
        bdims[0] = (band.len() / row_bytes) as u64;
        let mut sub_cd = chunk_dims.to_vec();
        sub_cd[0] = keep as u64;
        let mut sub_base = offsets[..rank].to_vec();
        sub_base[0] = lo - row;
        scatter_chunk(sub, &bdims, &sub_cd, &sub_base, elem_size, band, tf);
    };
    let out_rows = (out.len() / row_bytes) as u64;
    for batch in chunks.chunks(DECODE_BATCH) {
        let decoded = decode_batch(src, base, batch, elem_size, raw_filters, chunk_bytes, cache)?;
        // bands: chunks grouped by the rows they cover, in row order
        let mut order: Vec<usize> = (0..batch.len())
            .filter(|&i| {
                let (lo, hi) = rows(&batch[i].0);
                lo < hi && hi - r0 <= out_rows
            })
            .collect();
        order.sort_by_key(|&i| rows(&batch[i].0));
        let mut bands: Vec<((u64, u64), Vec<usize>)> = Vec::new();
        for i in order {
            let span = rows(&batch[i].0);
            match bands.last_mut() {
                Some((last, members)) if *last == span => members.push(i),
                _ => bands.push((span, vec![i])),
            }
        }
        let disjoint = bands.windows(2).all(|w| w[0].0 .1 <= w[1].0 .0);
        if !disjoint || bands.len() < 2 || crate::par::num_threads() < 2 {
            for (_, members) in &bands {
                for &i in members {
                    place(&batch[i].0, &decoded[i], out, r0);
                }
            }
            continue;
        }
        let mut work = Vec::with_capacity(bands.len());
        let mut rest = &mut out[..];
        let mut at = r0; // the dataset row `rest` starts at
        for ((lo, hi), members) in bands {
            let (_, tail) = std::mem::take(&mut rest).split_at_mut((lo - at) as usize * row_bytes);
            let (band, tail) = tail.split_at_mut((hi - lo) as usize * row_bytes);
            rest = tail;
            at = hi;
            work.push((lo, band, members));
        }
        crate::par::for_each(work, |(lo, band, members)| {
            for i in members {
                place(&batch[i].0, &decoded[i], band, lo);
            }
        });
    }
    Ok(())
}

/// Reach every chunk before decoding any: a source that fails on bytes it has
/// yet to fetch then learns them all at once, not one per attempt.
fn touch_chunks(src: &dyn ByteSource, base: u64, chunks: &[ChunkEntry]) -> Result<()> {
    let mut failed = None;
    for (_, size, _, addr) in chunks {
        if let Err(e) = src.slice(base + addr, *size as usize) {
            failed.get_or_insert(e);
        }
    }
    failed.map_or(Ok(()), Err)
}

/// [`materialize_chunks`] restricted to rows `[r0, r1)` of the leading axis:
/// `out` holds only those rows. Callers pass just the overlapping chunks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn materialize_chunk_rows(
    src: &dyn ByteSource,
    base: u64,
    chunks: &[ChunkEntry],
    dims: &[u64],
    chunk_dims: &[u64],
    elem_size: usize,
    raw_filters: &[filt::RawFilter],
    r0: u64,
    r1: u64,
    out: &mut [u8],
    cache: Option<&crate::cache::ChunkCache>,
) -> Result<()> {
    let rank = dims.len();
    if rank == 0 {
        return materialize_chunks(
            src,
            base,
            chunks,
            dims,
            chunk_dims,
            elem_size,
            raw_filters,
            out,
        );
    }
    // Output is a (r1 - r0, dims[1..]) row-major array.
    let mut odims = dims.to_vec();
    odims[0] = r1 - r0;
    if raw_filters.is_empty() {
        return plain_chunk_rows(
            src,
            base,
            chunks,
            dims,
            chunk_dims,
            elem_size,
            (r0, r1),
            &odims,
            out,
        );
    }
    touch_chunks(src, base, chunks)?;
    chunk_rows_into(
        src,
        base,
        chunks,
        dims,
        chunk_dims,
        elem_size,
        raw_filters,
        (r0, r1),
        out,
        cache,
        None,
    )
}

/// What a partial read picks out of a dataset.
pub(crate) enum Pick {
    /// Coordinates per axis (each in output order); the output is their
    /// cartesian product, row-major — a hyperslab.
    Axes(Vec<Vec<u64>>),
    /// Individual elements, by coordinates, in output order.
    Points(Vec<Vec<u64>>),
}

impl Pick {
    pub fn len(&self) -> usize {
        match self {
            Self::Axes(axes) => axes.iter().map(Vec::len).product(),
            Self::Points(points) => points.len(),
        }
    }

    /// The leading-axis rows the pick lies within, `[lo, hi)`.
    pub fn row_span(&self) -> Option<(u64, u64)> {
        let rows: Box<dyn Iterator<Item = u64> + '_> = match self {
            Self::Axes(axes) => Box::new(axes.first()?.iter().copied()),
            Self::Points(points) => Box::new(points.iter().filter_map(|p| p.first().copied())),
        };
        let (lo, hi) = rows.fold((u64::MAX, 0), |(lo, hi), r| (lo.min(r), hi.max(r + 1)));
        (lo < hi).then_some((lo, hi))
    }
}

/// Where chunks come from, and how they are laid out.
pub(crate) struct ChunkSource<'a> {
    pub src: &'a dyn ByteSource,
    pub base: u64,
    pub dims: &'a [u64],
    pub chunk_dims: &'a [u64],
    pub elem_size: usize,
    pub filters: &'a [filt::RawFilter],
    pub cache: Option<&'a crate::cache::ChunkCache>,
}

/// The elements `pick` selects, out of chunked storage, into `out` (packed in
/// pick order). Only chunks holding a selected element are decoded — in
/// parallel, and through the dataset's chunk cache.
pub(crate) fn chunks_pick_into(
    cs: &ChunkSource,
    chunks: &[ChunkEntry],
    pick: &Pick,
    out: &mut [u8],
) -> Result<()> {
    let rank = cs.dims.len();
    let es = cs.elem_size;
    let cd = cs.chunk_dims;
    let chunk_bytes = chunk_byte_size(cd, es)?;
    if rank == 0 || pick.len() == 0 {
        if let (Some(first), true) = (chunks.first(), pick.len() > 0) {
            let decoded = decode_batch(
                cs.src,
                cs.base,
                std::slice::from_ref(first),
                es,
                cs.filters,
                chunk_bytes,
                cs.cache,
            )?;
            let n = es.min(out.len()).min(decoded[0].len());
            out[..n].copy_from_slice(&decoded[0][..n]);
        }
        return Ok(());
    }
    let mut cstrides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        cstrides[i] = cstrides[i + 1] * cd[i + 1] as usize;
    }
    let grid = |off: &[u64]| -> Option<Vec<u64>> {
        (0..rank)
            .map(|i| (off[i] % cd[i] == 0).then(|| off[i] / cd[i]))
            .collect()
    };
    match pick {
        Pick::Axes(axes) => {
            // per axis: chunk number -> [(coordinate in chunk, output index)]
            let per_axis: Vec<HashMap<u64, Vec<(usize, usize)>>> = axes
                .iter()
                .zip(cd)
                .map(|(coords, &c)| {
                    let mut m: HashMap<u64, Vec<(usize, usize)>> = HashMap::new();
                    for (o, &x) in coords.iter().enumerate() {
                        m.entry(x / c).or_default().push(((x % c) as usize, o));
                    }
                    m
                })
                .collect();
            let mut ostrides = vec![1usize; rank];
            for i in (0..rank - 1).rev() {
                ostrides[i] = ostrides[i + 1] * axes[i + 1].len();
            }
            type Lists<'l> = Vec<&'l [(usize, usize)]>;
            let wanted: Vec<(ChunkEntry, Lists)> = chunks
                .iter()
                .filter_map(|chunk| {
                    let g = grid(&chunk.0)?;
                    let lists = (0..rank)
                        .map(|i| per_axis[i].get(&g[i]).map(Vec::as_slice))
                        .collect::<Option<Vec<_>>>()?;
                    Some((chunk.clone(), lists))
                })
                .collect();
            for batch in wanted.chunks(DECODE_BATCH) {
                let entries: Vec<ChunkEntry> = batch.iter().map(|(c, _)| c.clone()).collect();
                let decoded = decode_batch(
                    cs.src,
                    cs.base,
                    &entries,
                    es,
                    cs.filters,
                    chunk_bytes,
                    cs.cache,
                )?;
                for ((_, lists), chunk) in batch.iter().zip(&decoded) {
                    copy_picked(lists, &cstrides, &ostrides, es, |from, to, n| {
                        if let (Some(s), Some(d)) =
                            (chunk.get(from..from + n), out.get_mut(to..to + n))
                        {
                            d.copy_from_slice(s);
                        }
                    });
                }
            }
        }
        Pick::Points(points) => {
            // chunk grid position -> [(element in chunk, output index)]
            let mut by_chunk: HashMap<Vec<u64>, Vec<(usize, usize)>> = HashMap::new();
            for (o, p) in points.iter().enumerate() {
                let key: Vec<u64> = (0..rank).map(|i| p[i] / cd[i]).collect();
                let lin: usize = (0..rank)
                    .map(|i| (p[i] % cd[i]) as usize * cstrides[i])
                    .sum();
                by_chunk.entry(key).or_default().push((lin, o));
            }
            let wanted: Vec<(ChunkEntry, &Vec<(usize, usize)>)> = chunks
                .iter()
                .filter_map(|c| Some((c.clone(), by_chunk.get(&grid(&c.0)?)?)))
                .collect();
            for batch in wanted.chunks(DECODE_BATCH) {
                let entries: Vec<ChunkEntry> = batch.iter().map(|(c, _)| c.clone()).collect();
                let decoded = decode_batch(
                    cs.src,
                    cs.base,
                    &entries,
                    es,
                    cs.filters,
                    chunk_bytes,
                    cs.cache,
                )?;
                for ((_, picks), chunk) in batch.iter().zip(&decoded) {
                    for &(lin, o) in picks.iter() {
                        let (from, to) = (lin * es, o * es);
                        if let (Some(s), Some(d)) =
                            (chunk.get(from..from + es), out.get_mut(to..to + es))
                        {
                            d.copy_from_slice(s);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Write `values` (stored layout, one element per picked element, in pick
/// order) into chunked storage: each chunk the pick touches is decoded (or
/// started from `fill`, when it was never written), updated, re-encoded on
/// every core, and handed to `append`, which stores it and returns its
/// address. Returns the touched chunks' new index entries.
pub(crate) fn chunks_write_pick(
    cs: &ChunkSource,
    chunks: &[ChunkEntry],
    pick: &Pick,
    values: &[u8],
    fill: &[u8],
    append: &mut dyn FnMut(&[u8]) -> Result<u64>,
) -> Result<Vec<ChunkEntry>> {
    let rank = cs.dims.len();
    let es = cs.elem_size;
    let cd = cs.chunk_dims;
    let chunk_bytes = chunk_byte_size(cd, es)?;
    if rank == 0 {
        return Err("writing a scalar through chunks is not supported".into());
    }
    let mut cstrides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        cstrides[i] = cstrides[i + 1] * cd[i + 1] as usize;
    }
    // what lands in each touched chunk, keyed by the chunk's grid position
    enum Into<'p> {
        Lists(Vec<&'p [(usize, usize)]>),
        Points(Vec<(usize, usize)>),
    }
    let per_axis: Vec<HashMap<u64, Vec<(usize, usize)>>>;
    let mut touched: Vec<(Vec<u64>, Into)> = Vec::new();
    let mut ostrides = vec![1usize; rank];
    match pick {
        Pick::Axes(axes) => {
            per_axis = axes
                .iter()
                .zip(cd)
                .map(|(coords, &c)| {
                    let mut m: HashMap<u64, Vec<(usize, usize)>> = HashMap::new();
                    for (o, &x) in coords.iter().enumerate() {
                        m.entry(x / c).or_default().push(((x % c) as usize, o));
                    }
                    m
                })
                .collect();
            for i in (0..rank - 1).rev() {
                ostrides[i] = ostrides[i + 1] * axes[i + 1].len();
            }
            // every combination of touched chunk numbers, row-major
            let keys: Vec<Vec<u64>> = per_axis
                .iter()
                .map(|m| {
                    let mut k: Vec<u64> = m.keys().copied().collect();
                    k.sort_unstable();
                    k
                })
                .collect();
            if keys.iter().any(Vec::is_empty) {
                return Ok(Vec::new());
            }
            let mut at = vec![0usize; rank];
            loop {
                let grid: Vec<u64> = (0..rank).map(|i| keys[i][at[i]]).collect();
                let lists = (0..rank)
                    .map(|i| per_axis[i][&grid[i]].as_slice())
                    .collect();
                touched.push((grid, Into::Lists(lists)));
                let mut i = rank;
                loop {
                    if i == 0 {
                        break;
                    }
                    i -= 1;
                    at[i] += 1;
                    if at[i] < keys[i].len() {
                        break;
                    }
                    at[i] = 0;
                }
                if at.iter().all(|&a| a == 0) {
                    break;
                }
            }
        }
        Pick::Points(points) => {
            let mut by_chunk: HashMap<Vec<u64>, Vec<(usize, usize)>> = HashMap::new();
            for (o, p) in points.iter().enumerate() {
                let key: Vec<u64> = (0..rank).map(|i| p[i] / cd[i]).collect();
                let lin: usize = (0..rank)
                    .map(|i| (p[i] % cd[i]) as usize * cstrides[i])
                    .sum();
                by_chunk.entry(key).or_default().push((lin, o));
            }
            let mut v: Vec<_> = by_chunk.into_iter().collect();
            v.sort_by(|a, b| a.0.cmp(&b.0));
            touched = v.into_iter().map(|(k, p)| (k, Into::Points(p))).collect();
        }
    }
    // where each touched chunk is stored now, if anywhere
    let existing: HashMap<Vec<u64>, &ChunkEntry> = chunks
        .iter()
        .filter_map(|c| {
            let grid: Option<Vec<u64>> = (0..rank)
                .map(|i| (c.0[i] % cd[i] == 0).then(|| c.0[i] / cd[i]))
                .collect();
            Some((grid?, c))
        })
        .collect();
    let mut written = Vec::with_capacity(touched.len());
    for batch in touched.chunks(DECODE_BATCH) {
        let stored: Vec<ChunkEntry> = batch
            .iter()
            .filter_map(|(grid, _)| existing.get(grid).map(|c| (*c).clone()))
            .collect();
        let decoded = decode_batch(cs.src, cs.base, &stored, es, cs.filters, chunk_bytes, None)?;
        let mut decoded = decoded.into_iter();
        // update each chunk, then encode them all at once
        let mut updated: Vec<Vec<u8>> = Vec::with_capacity(batch.len());
        for (grid, into) in batch {
            let mut chunk = match existing.get(grid) {
                Some(_) => decoded.next().expect("one per stored chunk").to_vec(),
                None => fill_chunk(fill, es, chunk_bytes),
            };
            match into {
                Into::Lists(lists) => {
                    copy_picked(lists, &cstrides, &ostrides, es, |to, from, n| {
                        if let (Some(d), Some(s)) =
                            (chunk.get_mut(to..to + n), values.get(from..from + n))
                        {
                            d.copy_from_slice(s);
                        }
                    })
                }
                Into::Points(points) => {
                    for &(lin, o) in points {
                        let (to, from) = (lin * es, o * es);
                        if let (Some(d), Some(s)) =
                            (chunk.get_mut(to..to + es), values.get(from..from + es))
                        {
                            d.copy_from_slice(s);
                        }
                    }
                }
            }
            updated.push(chunk);
        }
        let encoded = crate::par::map(&updated, |chunk| {
            if cs.filters.is_empty() {
                Ok(chunk.clone())
            } else {
                filt::apply(cs.filters, es, chunk)
            }
        });
        for (((grid, _), chunk), bytes) in batch.iter().zip(&updated).zip(encoded) {
            let bytes = bytes?;
            let addr = append(&bytes)?;
            if let Some(cache) = cs.cache {
                cache.put(cs.base + addr, std::sync::Arc::from(chunk.as_slice()));
            }
            let mut offsets: Vec<u64> = (0..rank).map(|i| grid[i] * cd[i]).collect();
            offsets.push(0);
            written.push((offsets, bytes.len() as u32, 0, addr));
        }
    }
    Ok(written)
}

/// A chunk of `fill` (one element's bytes; zeros when empty).
fn fill_chunk(fill: &[u8], es: usize, chunk_bytes: usize) -> Vec<u8> {
    if fill.len() == es && fill.iter().any(|&b| b != 0) {
        fill.iter().copied().cycle().take(chunk_bytes).collect()
    } else {
        vec![0u8; chunk_bytes]
    }
}

/// The elements `pick` selects out of contiguous storage at `offset` (`len`
/// bytes stored), reading only the runs it needs.
pub(crate) fn contiguous_pick_into(
    src: &dyn ByteSource,
    (offset, len): (u64, u64),
    dims: &[u64],
    elem_size: usize,
    pick: &Pick,
    out: &mut [u8],
) -> Result<()> {
    let rank = dims.len();
    let mut strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1] as usize;
    }
    let mut failed = None;
    let mut read = |from: usize, to: usize, n: usize| {
        // past what is stored reads as zeros, as a whole read does
        let avail = (len.saturating_sub(from as u64)).min(n as u64) as usize;
        if avail == 0 || failed.is_some() {
            return;
        }
        match src.slice(offset + from as u64, avail) {
            Ok(s) => {
                if let Some(d) = out.get_mut(to..to + avail) {
                    d.copy_from_slice(s);
                }
            }
            Err(e) => failed = Some(e),
        }
    };
    match pick {
        Pick::Axes(axes) if rank > 0 => {
            let lists: Vec<Vec<(usize, usize)>> = axes
                .iter()
                .map(|a| {
                    a.iter()
                        .enumerate()
                        .map(|(o, &c)| (c as usize, o))
                        .collect()
                })
                .collect();
            let lists: Vec<&[(usize, usize)]> = lists.iter().map(Vec::as_slice).collect();
            let mut ostrides = vec![1usize; rank];
            for i in (0..rank - 1).rev() {
                ostrides[i] = ostrides[i + 1] * axes[i + 1].len();
            }
            copy_picked(&lists, &strides, &ostrides, elem_size, &mut read);
        }
        Pick::Axes(_) => read(0, 0, elem_size),
        Pick::Points(points) => {
            for (o, p) in points.iter().enumerate() {
                let lin: usize = (0..rank).map(|i| p[i] as usize * strides[i]).sum();
                read(lin * elem_size, o * elem_size, elem_size);
            }
        }
    }
    failed.map_or(Ok(()), Err)
}

/// Copy a hyperslab pick out of one block (a chunk, or the whole dataset):
/// `lists[i]` holds `(coordinate in block, output index)` along axis `i`.
/// Consecutive coordinates that land consecutively in the output are merged
/// into one run, so `copy(from_byte, to_byte, bytes)` is called per run.
fn copy_picked(
    lists: &[&[(usize, usize)]],
    strides: &[usize],
    ostrides: &[usize],
    es: usize,
    mut copy: impl FnMut(usize, usize, usize),
) {
    let rank = lists.len();
    let last = rank - 1;
    let mut runs: Vec<(usize, usize, usize)> = Vec::new();
    for &(c, o) in lists[last] {
        match runs.last_mut() {
            Some((c0, o0, n)) if *c0 + *n == c && *o0 + *n == o => *n += 1,
            _ => runs.push((c, o, 1)),
        }
    }
    if lists[..last].iter().any(|l| l.is_empty()) {
        return;
    }
    let mut at = vec![0usize; last];
    loop {
        let (mut from, mut to) = (0, 0);
        for i in 0..last {
            let (c, o) = lists[i][at[i]];
            from += c * strides[i];
            to += o * ostrides[i];
        }
        for &(c, o, n) in &runs {
            copy(
                (from + c * strides[last]) * es,
                (to + o * ostrides[last]) * es,
                n * es,
            );
        }
        let mut i = last;
        loop {
            if i == 0 {
                return;
            }
            i -= 1;
            at[i] += 1;
            if at[i] < lists[i].len() {
                break;
            }
            at[i] = 0;
        }
    }
}

/// Copy one chunk (row-major, `chunk_dims`, at `base`) into its place in the
/// row-major dataset `out` (`dims`); parts past the dataset edge are dropped.
fn scatter_chunk(
    chunk: &[u8],
    dims: &[u64],
    chunk_dims: &[u64],
    base: &[u64],
    elem_size: usize,
    out: &mut [u8],
    tf: Transform,
) {
    let Some((oes, f)) = tf else {
        if dims.is_empty() {
            let n = elem_size.min(chunk.len()).min(out.len());
            out[..n].copy_from_slice(&chunk[..n]);
            return;
        }
        return copy_runs(dims, chunk_dims, base, elem_size, |bytes, c, g| {
            if let (Some(from), Some(to)) = (chunk.get(c..c + bytes), out.get_mut(g..g + bytes)) {
                to.copy_from_slice(from);
            }
        });
    };
    if dims.is_empty() {
        if let (Some(from), Some(to)) = (chunk.get(..elem_size), out.get_mut(..oes)) {
            f(from, to);
        }
        return;
    }
    copy_runs(dims, chunk_dims, base, elem_size, |bytes, c, g| {
        let to = g / elem_size * oes;
        let n = bytes / elem_size * oes;
        if let (Some(from), Some(to)) = (chunk.get(c..c + bytes), out.get_mut(to..to + n)) {
            f(from, to);
        }
    });
}

/// How stored elements land in the output: `None` copies them; `Some((size,
/// f))` hands each run to `f`, which writes elements of `size` bytes —
/// byte-swapped or converted on the way, so no second pass over the output
/// is needed.
pub(crate) type Transform<'a> = Option<(usize, &'a (dyn Fn(&[u8], &mut [u8]) + Sync))>;

/// The inverse of [`scatter_chunk`]: one chunk's elements out of the dataset
/// `data`; elements past the dataset edge are left as they are in `chunk`.
pub(crate) fn gather_chunk(
    data: &[u8],
    dims: &[u64],
    chunk_dims: &[u64],
    base: &[u64],
    elem_size: usize,
    chunk: &mut [u8],
) {
    if dims.is_empty() {
        let n = elem_size.min(data.len()).min(chunk.len());
        chunk[..n].copy_from_slice(&data[..n]);
        return;
    }
    copy_runs(dims, chunk_dims, base, elem_size, |bytes, c, g| {
        if let (Some(to), Some(from)) = (chunk.get_mut(c..c + bytes), data.get(g..g + bytes)) {
            to.copy_from_slice(from);
        }
    });
}

/// Walk the contiguous runs shared by a chunk (`chunk_dims` at `base`) and
/// the dataset (`dims`, rank ≥ 1): `copy(bytes, chunk_offset, data_offset)`
/// per run, all in bytes.
///
/// A run is the chunk's extent along the last axis — and, while the chunk
/// spans an axis whole, the axes before it too. The axis just outside the
/// run is walked in a tight strided loop; run sizes of 1–16 bytes are copied
/// as fixed-size moves rather than `memcpy` calls, which is what keeps
/// narrow chunks (one column of a table) fast.
fn copy_runs(
    dims: &[u64],
    chunk_dims: &[u64],
    base: &[u64],
    elem_size: usize,
    mut copy: impl FnMut(usize, usize, usize),
) {
    let rank = dims.len();
    let mut strides = vec![1usize; rank];
    let mut cstrides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1] as usize;
        cstrides[i] = cstrides[i + 1] * chunk_dims[i + 1] as usize;
    }
    // the part of the chunk inside the dataset, per axis
    let extent: Vec<usize> = (0..rank)
        .map(|i| chunk_dims[i].min(dims[i].saturating_sub(base[i])) as usize)
        .collect();
    if extent.contains(&0) {
        return;
    }
    let mut last = rank - 1;
    let mut run = extent[last];
    while last > 0 && base[last] == 0 && chunk_dims[last] == dims[last] {
        last -= 1;
        run *= extent[last];
    }
    let bytes = run * elem_size;
    let origin: usize = (0..rank)
        .map(|i| base[i] as usize * strides[i])
        .sum::<usize>()
        * elem_size;
    if last == 0 {
        return copy(bytes, 0, origin);
    }
    // the strided axis, and the odometer over the ones before it
    let axis = last - 1;
    let (count, cstep, gstep) = (
        extent[axis],
        cstrides[axis] * elem_size,
        strides[axis] * elem_size,
    );
    let mut coord = vec![0usize; axis];
    loop {
        let c0: usize = (0..axis).map(|i| coord[i] * cstrides[i]).sum::<usize>() * elem_size;
        let g0: usize =
            origin + (0..axis).map(|i| coord[i] * strides[i]).sum::<usize>() * elem_size;
        let mut line = |bytes: usize| {
            for k in 0..count {
                copy(bytes, c0 + k * cstep, g0 + k * gstep);
            }
        };
        // a constant `bytes` lets each copy compile to one load/store
        match bytes {
            1 => line(1),
            2 => line(2),
            4 => line(4),
            8 => line(8),
            16 => line(16),
            b => line(b),
        }
        let mut i = axis;
        loop {
            if i == 0 {
                return;
            }
            i -= 1;
            coord[i] += 1;
            if coord[i] < extent[i] {
                break;
            }
            coord[i] = 0;
        }
    }
}
