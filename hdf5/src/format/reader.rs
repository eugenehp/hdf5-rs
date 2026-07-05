//! Parsing of an HDF5 byte image into the in-memory model.
//!
//! Coverage is deliberately broader than the writer: superblock versions 0-3,
//! object header versions 1 and 2 (including continuation blocks), old-style
//! symbol-table groups (B-tree v1 + local heap + SNOD) and new-style compact
//! groups (link messages), contiguous/compact/chunked layouts (chunk B-tree
//! v1), the filter pipeline, and the global heap for variable-length data.

use std::collections::HashMap;

use hdf5_types::TypeDescriptor;

use super::convert::{disk_size, has_vlen, VlenStore};
use super::datatype::OrderTree;
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

/// How the chunks of a chunked dataset are indexed on disk.
enum ChunkIndex {
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
    parse_depth(image, dir, 0)
}

fn parse_depth(
    image: &std::sync::Arc<crate::model::FileImage>,
    dir: Option<&std::path::Path>,
    depth: usize,
) -> Result<FileState> {
    if depth > 8 {
        return Err("virtual dataset source nesting too deep".into());
    }
    let data: &[u8] = image;
    // stored (base-relative) root object-header address
    let (root_oh_addr, base, ext_addr) = find_superblock(data)?;
    let mut p = Parser {
        data,
        image: image.clone(),
        base,
        state: FileState::new_empty(),
        memo: HashMap::new(),
        gheap_cache: HashMap::new(),
        pending_vds: Vec::new(),
        dir: dir.map(std::path::Path::to_path_buf),
        depth,
        sohm_indexes: Vec::new(),
        pending_lazy: None,
    };
    // Load the SOHM master table (if any) from the superblock extension
    // before parsing objects, so shared-message references resolve.
    if ext_addr != UNDEF {
        p.load_sohm_table(ext_addr)?;
    }
    // Record the userblock (superblock offset == base) so rewrites keep it.
    p.state.userblock = base;
    p.state.userblock_data = data[..(base as usize).min(data.len())].to_vec();
    // Replace the default-constructed root with the parsed one.
    let root = p.parse_object(root_oh_addr)?;
    p.state.root = root;
    // Materialize virtual datasets now that every source in this file exists.
    p.materialize_vds()?;
    p.state.recount();
    Ok(p.state)
}

/// Locate the superblock (it may sit after a userblock at offsets 512, 1024, ...)
/// and return (root object header address, base address, superblock
/// extension address).
fn find_superblock(data: &[u8]) -> Result<(u64, u64, u64)> {
    let mut offset = 0usize;
    loop {
        if offset + 96 > data.len() {
            return Err("not an HDF5 file (superblock signature not found)".into());
        }
        if data[offset..offset + 8] == SIGNATURE {
            break;
        }
        offset = if offset == 0 { 512 } else { offset * 2 };
    }
    let mut c = Cursor::at(data, offset + 8);
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
    data: &'a [u8],
    image: std::sync::Arc<crate::model::FileImage>,
    base: u64,
    state: FileState,
    memo: HashMap<u64, ObjId>,
    gheap_cache: HashMap<u64, HashMap<u32, Vec<u8>>>,
    /// Virtual datasets awaiting materialization: (arena id, heap addr, idx).
    pending_vds: Vec<(ObjId, u64, u32)>,
    /// Directory of the file being parsed (for VDS source resolution).
    dir: Option<std::path::PathBuf>,
    depth: usize,
    /// SOHM indexes: (message-type flag bits, fractal heap address).
    sohm_indexes: Vec<(u16, u64)>,
    /// Set by read_dataset_data when the dataset can stay lazily referenced.
    pending_lazy: Option<crate::model::LazyData>,
}

impl<'a> Parser<'a> {
    fn addr(&self, a: u64) -> usize {
        (self.base + a) as usize
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
                    // byte-swapped data needs the eager postprocess below
                    if let Some(l) = lazy.take() {
                        data = l.materialize_bytes()?;
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
                    for i in 0..n {
                        apply_order(&order, &mut data[i * esize..(i + 1) * esize], &mut vlen);
                    }
                    if let FillValue::UserDefined(fv) = &mut fill {
                        if fv.len() >= esize {
                            let mut dummy = VlenStore::new();
                            apply_order(&order, &mut fv[..esize], &mut dummy);
                        }
                    }
                }
                ObjectKind::Dataset(DatasetData {
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
                })
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
                        let mut st = parse_depth(&img, path.parent(), self.depth + 1)?;
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
        let heap = FractalHeap::parse(self.data, self.base, fheap)?;
        let btree = read_v2btree(self.data, self.base, name_bt2)?;
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
        let heap = FractalHeap::parse(self.data, self.base, fheap)?;
        let btree = read_v2btree(self.data, self.base, name_bt2)?;
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
        if pos + 4 > self.data.len() {
            return Err("object header address out of bounds".into());
        }
        if &self.data[pos..pos + 4] == b"OHDR" {
            self.parse_object_header_v2(addr)
        } else {
            self.parse_object_header_v1(addr)
        }
    }

    fn parse_object_header_v1(&mut self, addr: u64) -> Result<ObjHeader> {
        let mut c = Cursor::at(self.data, self.addr(addr));
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
            let mut mc = Cursor::at(self.data, start);
            let end = start + len;
            while mc.pos + 8 <= end && msgs_read < num_msgs {
                let mtype = mc.u16()?;
                let msize = mc.u16()? as usize;
                let mflags = mc.u8()?;
                mc.skip(3);
                let body_start = mc.pos;
                if body_start + msize > end || body_start + msize > self.data.len() {
                    break;
                }
                self.handle_message(
                    mtype,
                    mflags,
                    &self.data[body_start..body_start + msize],
                    &mut oh,
                    &mut blocks,
                )?;
                mc.seek(body_start + msize);
                msgs_read += 1;
            }
            bi += 1;
        }
        Ok(oh)
    }

    fn parse_object_header_v2(&mut self, addr: u64) -> Result<ObjHeader> {
        let mut c = Cursor::at(self.data, self.addr(addr));
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
            let mut mc = Cursor::at(self.data, start);
            let end = start + len;
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
                if body_start + msize > payload_end || body_start + msize > self.data.len() {
                    break;
                }
                self.handle_message(
                    mtype,
                    mflags,
                    &self.data[body_start..body_start + msize],
                    &mut oh,
                    &mut blocks,
                )?;
                mc.seek(body_start + msize);
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
        blocks: &mut Vec<(usize, usize)>,
    ) -> Result<()> {
        // Message flag bit 1: the body is a *shared message* reference, not
        // the message itself (H5Oshared.c).
        if mflags & 0x02 != 0 {
            return self.handle_shared_message(mtype, body, oh);
        }
        let mut c = Cursor::new(body);
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
        let mut c = Cursor::at(self.data, self.addr(table_addr));
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
        let mut c = Cursor::new(body);
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
                    let heap = FractalHeap::parse(self.data, self.base, heap_addr)?;
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
        let mut c = Cursor::new(body);
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
                let child = self.parse_object(oh_addr)?;
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
        for snod_addr in snods {
            let mut c = Cursor::at(self.data, self.addr(snod_addr));
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
                    let child = self.parse_object(oh_addr)?;
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
        if addr == UNDEF {
            return Ok(());
        }
        let mut c = Cursor::at(self.data, self.addr(addr));
        let sig = c.take(4)?;
        if sig != b"TREE" {
            return Err("bad TREE signature".into());
        }
        let node_type = c.u8()?;
        if node_type != 0 {
            return Err("expected group B-tree node".into());
        }
        let level = c.u8()?;
        let entries = c.u16()? as usize;
        let _left = c.u64()?;
        let _right = c.u64()?;
        let _key0 = c.u64()?;
        for _ in 0..entries {
            let child = c.u64()?;
            let _key = c.u64()?;
            if level == 0 {
                out.push(child);
            } else {
                self.walk_group_btree(child, out)?;
            }
        }
        Ok(())
    }

    fn parse_local_heap(&mut self, addr: u64) -> Result<Vec<u8>> {
        let mut c = Cursor::at(self.data, self.addr(addr));
        let sig = c.take(4)?;
        if sig != b"HEAP" {
            return Err("bad HEAP signature".into());
        }
        let _ver = c.u8()?;
        c.skip(3);
        let seg_size = c.u64()? as usize;
        let _free_off = c.u64()?;
        let seg_addr = c.u64()?;
        let start = self.addr(seg_addr);
        if start + seg_size > self.data.len() {
            return Err("local heap out of bounds".into());
        }
        Ok(self.data[start..start + seg_size].to_vec())
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
        let total_elems: usize = if is_null {
            0
        } else if is_scalar {
            1
        } else {
            let mut n: u64 = 1;
            for &d in dims {
                n = n.checked_mul(d).ok_or("dataset dimension overflow")?;
            }
            n as usize
        };
        let logical_size = total_elems
            .checked_mul(elem_size)
            .ok_or("dataset size overflow")?;
        // sanity bound: never materialize more than 64 GiB from a parse (a
        // corrupt size field must error, not abort on allocation)
        if logical_size as u64 > 1 << 36 {
            return Err(format!("dataset too large to materialize ({logical_size} bytes)").into());
        }

        let (class, mut raw) = match layout {
            RawLayout::Compact(bytes) => {
                let mut v = bytes.clone();
                v.resize(logical_size, 0);
                (LayoutClass::Compact, v)
            }
            RawLayout::Contiguous { addr, size } => {
                if *addr == UNDEF || logical_size == 0 {
                    (LayoutClass::Contiguous, vec![0u8; logical_size])
                } else {
                    let start = self.addr(*addr);
                    let n = (*size as usize).min(logical_size);
                    if start.checked_add(n).is_none_or(|e| e > self.data.len()) {
                        return Err("contiguous data out of bounds".into());
                    }
                    if raw_filters.is_empty() && !has_vlen(dtype) {
                        // defer the copy: reference the file image directly
                        self.pending_lazy = Some(crate::model::LazyData {
                            image: self.image.clone(),
                            base: self.base,
                            logical: logical_size,
                            kind: crate::model::LazyKind::Contiguous {
                                offset: start,
                                len: n,
                            },
                        });
                        (LayoutClass::Contiguous, Vec::new())
                    } else {
                        let mut v = self.data[start..start + n].to_vec();
                        v.resize(logical_size, 0);
                        (LayoutClass::Contiguous, v)
                    }
                }
            }
            RawLayout::Chunked { index, chunk_dims } => {
                // chunk_dims includes the trailing element-size dim; drop it
                let cd: Vec<u64> = chunk_dims[..chunk_dims.len().saturating_sub(1)].to_vec();
                let mut data = vec![0u8; logical_size];
                self.read_chunked(index, dims, &cd, elem_size, raw_filters, &mut data)?;
                (LayoutClass::Chunked(cd), data)
            }
            RawLayout::Virtual { .. } => {
                // mappings + data filled in later by materialize_vds()
                (LayoutClass::Virtual(Vec::new()), vec![0u8; logical_size])
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
                    for e in 0..total_elems {
                        let cell = &mut raw[e * elem_size..(e + 1) * elem_size];
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
            for i in 0..total_elems {
                let off = i * elem_size;
                self.load_vlen_elem(dtype, &mut raw[off..off + elem_size], &mut store)?;
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

    fn read_chunked(
        &mut self,
        index: &ChunkIndex,
        dims: &[u64],
        chunk_dims: &[u64],
        elem_size: usize,
        raw_filters: &[filt::RawFilter],
        out: &mut [u8],
    ) -> Result<()> {
        let rank = dims.len();
        let chunk_elems: usize = chunk_dims.iter().map(|&c| c as usize).product();
        let chunk_bytes = chunk_elems * elem_size;
        // Collect (offsets, stored_size, filter_mask, addr) for every chunk.
        let mut chunks: Vec<(Vec<u64>, u32, u32, u64)> = Vec::new();
        match index {
            ChunkIndex::BtreeV1 { addr } => {
                if *addr != UNDEF {
                    self.walk_chunk_btree(*addr, rank, &mut chunks)?;
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
                    let entries = self.read_fixed_array(*header_addr)?;
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
                    let ea = ExtensibleArray::parse(self.data, self.base, *header_addr)?;
                    for (i, offsets) in chunk_grid(dims, chunk_dims).into_iter().enumerate() {
                        let Some(elem) = ea.element(i as u64)? else {
                            continue;
                        };
                        // client 0: addr(8); client 1: addr(8)+size+mask(4)
                        let mut ec = Cursor::new(&elem);
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
                    let bt = read_v2btree(self.data, self.base, *header_addr)?;
                    for rec in &bt.records {
                        let mut rc = Cursor::new(rec);
                        let addr = rc.addr()?;
                        let (size, mask) = match bt.btree_type {
                            10 => (chunk_bytes as u32, 0),
                            11 => {
                                // filtered: chunk size length = record len
                                // minus addr(8) + mask(4) + scaled offsets
                                let size_len = rec
                                    .len()
                                    .checked_sub(8 + 4 + rank * 8)
                                    .ok_or("bad v2 btree chunk record size")?;
                                let size = rc.uint(size_len)? as u32;
                                let mask = rc.u32()?;
                                (size, mask)
                            }
                            t => {
                                return Err(format!("unexpected chunk btree record type {t}").into())
                            }
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
        // Defer the decode: keep only the chunk list (metadata) and load on
        // first access. NBit stays eager (post-decode sign extension happens
        // in the parser).
        let has_nbit = raw_filters.iter().any(|f| f.id == filt::FILTER_NBIT);
        if !has_nbit {
            self.pending_lazy = Some(crate::model::LazyData {
                image: self.image.clone(),
                base: self.base,
                logical: out.len(),
                kind: crate::model::LazyKind::Chunked {
                    chunks,
                    dims: dims.to_vec(),
                    chunk_dims: chunk_dims.to_vec(),
                    elem_size,
                    filters: raw_filters.to_vec(),
                },
            });
            return Ok(());
        }
        materialize_chunks(
            self.data,
            self.base,
            &chunks,
            dims,
            chunk_dims,
            elem_size,
            raw_filters,
            out,
        )
    }

    /// Parse a Fixed Array chunk index (FAHD header + FADB data block),
    /// returning one `(chunk_addr, filtered_size, filter_mask)` per chunk in
    /// row-major order.
    fn read_fixed_array(&mut self, header_addr: u64) -> Result<Vec<(u64, Option<u32>, u32)>> {
        let mut c = Cursor::at(self.data, self.addr(header_addr));
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

        let mut d = Cursor::at(self.data, self.addr(data_block_addr));
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

    fn walk_chunk_btree(
        &mut self,
        addr: u64,
        rank: usize,
        out: &mut Vec<(Vec<u64>, u32, u32, u64)>,
    ) -> Result<()> {
        if addr == UNDEF {
            return Ok(());
        }
        let mut c = Cursor::at(self.data, self.addr(addr));
        let sig = c.take(4)?;
        if sig != b"TREE" {
            return Err("bad chunk TREE signature".into());
        }
        let node_type = c.u8()?;
        if node_type != 1 {
            return Err("expected chunk B-tree node".into());
        }
        let level = c.u8()?;
        let entries = c.u16()? as usize;
        let _left = c.u64()?;
        let _right = c.u64()?;
        for _ in 0..entries {
            // key
            let size = c.u32()?;
            let mask = c.u32()?;
            let mut offsets = Vec::with_capacity((rank + 1).min(1 << 16));
            for _ in 0..=rank {
                offsets.push(c.u64()?);
            }
            let child = c.u64()?;
            if level == 0 {
                out.push((offsets, size, mask, child));
            } else {
                self.walk_chunk_btree(child, rank, out)?;
            }
        }
        Ok(())
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
        let mut c = Cursor::at(self.data, self.addr(addr));
        let sig = c.take(4)?;
        if sig != b"GCOL" {
            return Err("bad GCOL signature".into());
        }
        let _ver = c.u8()?;
        c.skip(3);
        let col_size = c.u64()? as usize;
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
        let mut c = Cursor::new(body);
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
                let mut dt_cur = Cursor::at(body, dt_start);
                let mut order = OrderTree::None;
                let dtype = datatype::decode_ordered(&mut dt_cur, &mut order)?;
                c.seek(dt_start + super::align8(dt_size));
                let ds_start = c.pos;
                let mut ds_cur = Cursor::at(body, ds_start);
                let (dims, _maxdims, is_scalar, is_null) = parse_dataspace(&mut ds_cur)?;
                c.seek(ds_start + super::align8(ds_size));
                if c.pos > body.len() {
                    return Err("attribute message truncated".into());
                }
                let data = body[c.pos..].to_vec();
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
                let mut dt_cur = Cursor::at(body, dt_start);
                let mut order = OrderTree::None;
                let dtype = datatype::decode_ordered(&mut dt_cur, &mut order)?;
                c.seek(dt_start + dt_size);
                let ds_start = c.pos;
                let mut ds_cur = Cursor::at(body, ds_start);
                let (dims, _maxdims, is_scalar, is_null) = parse_dataspace(&mut ds_cur)?;
                c.seek(ds_start + ds_size);
                if c.pos > body.len() {
                    return Err("attribute message truncated".into());
                }
                let data = body[c.pos..].to_vec();
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
            for i in 0..n {
                apply_order(
                    &order,
                    &mut data[i * elem_size..(i + 1) * elem_size],
                    &mut store,
                );
            }
        }
        Ok(Some(AttrData {
            name,
            dtype,
            dims,
            is_scalar,
            is_null,
            data,
            vlen: store,
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

/// Decode and scatter a chunk list into a logical dataset buffer. Shared by
/// the eager parse path and lazy materialization ([`crate::model::LazyData`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn materialize_chunks(
    data: &[u8],
    base: u64,
    chunks: &[(Vec<u64>, u32, u32, u64)],
    dims: &[u64],
    chunk_dims: &[u64],
    elem_size: usize,
    raw_filters: &[filt::RawFilter],
    out: &mut [u8],
) -> Result<()> {
    let rank = dims.len();
    let chunk_elems: usize = chunk_dims.iter().map(|&c| c as usize).product();
    let chunk_bytes = chunk_elems * elem_size;
    if chunk_bytes > (1 << 33) {
        return Err("chunk too large".into());
    }
    for (offsets, size, mask, addr) in chunks {
        let start = (base + addr) as usize;
        if start
            .checked_add(*size as usize)
            .is_none_or(|e| e > data.len())
        {
            return Err("chunk data out of bounds".into());
        }
        let stored = &data[start..start + *size as usize];
        let mut chunk = if raw_filters.is_empty() {
            stored.to_vec()
        } else {
            filt::reverse_masked(raw_filters, elem_size, stored, *mask)?
        };
        chunk.resize(chunk_bytes, 0);
        scatter_chunk(&chunk, dims, chunk_dims, &offsets[..rank], elem_size, out);
    }
    Ok(())
}

/// Copy one chunk's elements into the row-major logical array.
fn scatter_chunk(
    chunk: &[u8],
    dims: &[u64],
    chunk_dims: &[u64],
    base: &[u64],
    elem_size: usize,
    out: &mut [u8],
) {
    let rank = dims.len();
    if rank == 0 {
        let n = elem_size.min(chunk.len()).min(out.len());
        out[..n].copy_from_slice(&chunk[..n]);
        return;
    }
    let mut strides = vec![1u64; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    let mut cstrides = vec![1u64; rank];
    for i in (0..rank - 1).rev() {
        cstrides[i] = cstrides[i + 1] * chunk_dims[i + 1];
    }
    let chunk_total: usize = chunk_dims.iter().map(|&c| c as usize).product();
    let mut coord = vec![0u64; rank];
    for lin in 0..chunk_total {
        let mut rem = lin as u64;
        for i in 0..rank {
            coord[i] = rem / cstrides[i];
            rem %= cstrides[i];
        }
        let mut inside = true;
        let mut global_lin = 0u64;
        for i in 0..rank {
            let g = base[i] + coord[i];
            if g >= dims[i] {
                inside = false;
                break;
            }
            global_lin += g * strides[i];
        }
        if inside {
            let src = lin * elem_size;
            let dst = (global_lin as usize) * elem_size;
            if src + elem_size <= chunk.len() && dst + elem_size <= out.len() {
                out[dst..dst + elem_size].copy_from_slice(&chunk[src..src + elem_size]);
            }
        }
    }
}
