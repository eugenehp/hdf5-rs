//! Serialization of the in-memory model into an HDF5 byte image (superblock v0,
//! object header v1, old-style symbol-table groups, contiguous/chunked layouts,
//! global-heap collections for variable-length data).

use std::collections::HashMap;

use super::convert::{disk_size, has_vlen, slot_parts, VlenStore};
use super::{
    align8, datatype, filters as filt, Buf, CHUNK_K, GROUP_INTERNAL_K, GROUP_LEAF_K, SIGNATURE,
    UNDEF,
};
use crate::error::Result;
use crate::hl::filters::Filter;
use crate::model::{
    AttrData, DatasetData, FileState, FillValue, LayoutClass, LinkTarget, ObjId, ObjectKind,
};

// Object-header message type codes.
const MSG_DATASPACE: u16 = 0x0001;
const MSG_LINK_INFO: u16 = 0x0002;
const MSG_LINK: u16 = 0x0006;
const MSG_GROUP_INFO: u16 = 0x000A;
const MSG_MOD_TIME: u16 = 0x0012;
const MSG_DATATYPE: u16 = 0x0003;
const MSG_FILL: u16 = 0x0005;
const MSG_LAYOUT: u16 = 0x0008;
const MSG_FILTER: u16 = 0x000B;
const MSG_ATTRIBUTE: u16 = 0x000C;
const MSG_COMMENT: u16 = 0x000D;
const MSG_SYMBOL_TABLE: u16 = 0x0011;
const MSG_ATTRIBUTE_INFO: u16 = 0x0015;

// Fractal-heap creation parameters for dense attribute storage
// (H5Oprivate.h H5O_FHEAP_* / H5Adense.c H5A_NAME_BT2_*).
const FH_WIDTH: usize = 4;
const FH_START_BLOCK: u64 = 1024;
const FH_MAX_DIRECT: u64 = 64 * 1024;
const FH_MAX_HEAP_BITS: usize = 40;
const FH_MAX_MAN_SIZE: usize = 4096;
const FH_ID_LEN: usize = 8;
const ATTR_BT2_SPLIT: u8 = 100;
const ATTR_BT2_MERGE: u8 = 40;

/// On-disk reference to a serialized object, used to fill in symbol table
/// entries in the parent group.
#[derive(Clone, Copy)]
struct Ref {
    oh_addr: u64,
    cache_type: u32,
    btree: u64,
    heap: u64,
}

struct Writer<'a> {
    state: &'a FileState,
    buf: Buf,
    memo: HashMap<ObjId, Ref>,
    visiting: Vec<ObjId>,
    /// Shared-message stores, one per configured SOHM index.
    sohm: Vec<SohmIndex>,
    /// File positions of shared-ref heap-id placeholders: (pos, (index, obj)).
    sohm_patches: Vec<(usize, (usize, usize))>,
}

/// Write-side state of one SOHM index: deduplicated message bodies plus the
/// bump allocator that assigns stable fractal-heap IDs at insert time.
struct SohmIndex {
    flags: u16,
    min_size: u32,
    objects: Vec<Vec<u8>>,
    hashes: Vec<u32>,
    refcounts: Vec<u32>,
    ids: Vec<Vec<u8>>,
    map: HashMap<Vec<u8>, usize>,
}

/// Serialize a file state into a complete HDF5 image.
pub fn serialize(state: &FileState) -> Result<Vec<u8>> {
    let sohm = state
        .sohm
        .iter()
        .map(|&(flags, min_size)| SohmIndex {
            flags,
            min_size,
            objects: Vec::new(),
            hashes: Vec::new(),
            refcounts: Vec::new(),
            ids: Vec::new(),
            map: HashMap::new(),
        })
        .collect();
    let mut w = Writer {
        state,
        buf: Buf::new(),
        memo: HashMap::new(),
        visiting: Vec::new(),
        sohm,
        sohm_patches: Vec::new(),
    };
    // Reserve the 96-byte superblock at offset 0.
    w.buf.zeros(96);
    let root = w.write_object(state.root)?;
    // Shared-message tables (superblock extension) when configured.
    let ext_addr = w.write_sohm_tables()?;
    let eof = align8(w.buf.len()) as u64;
    w.buf.bytes.resize(eof as usize, 0);
    let ub = state.userblock;
    // the EOF field is absolute: it includes the userblock (h5py layout)
    if let Some(ext) = ext_addr {
        w.write_superblock_v2(root.oh_addr, ext, ub + eof, ub);
    } else {
        w.write_superblock(root.oh_addr, root.btree, root.heap, ub + eof, ub);
    }
    if ub > 0 {
        // prepend the userblock; stored addresses stay base-relative and the
        // superblock base-address field carries the block size (h5py layout)
        let mut out = state.userblock_data.clone();
        out.resize(ub as usize, 0);
        out.extend_from_slice(&w.buf.bytes);
        return Ok(out);
    }
    Ok(w.buf.bytes)
}

impl<'a> Writer<'a> {
    /// Append `bytes` at the next 8-aligned offset, returning its address.
    fn append(&mut self, bytes: &[u8]) -> u64 {
        self.buf.pad8();
        let addr = self.buf.len() as u64;
        self.buf.raw(bytes);
        addr
    }

    fn write_superblock(&mut self, root_oh: u64, btree: u64, heap: u64, eof: u64, base: u64) {
        let mut b = Buf::new();
        b.raw(&SIGNATURE);
        b.u8(0); // superblock version
        b.u8(0); // free space version
        b.u8(0); // root group symbol table version
        b.u8(0); // reserved
        b.u8(0); // shared header message version
        b.u8(8); // size of offsets
        b.u8(8); // size of lengths
        b.u8(0); // reserved
        b.u16(GROUP_LEAF_K);
        b.u16(GROUP_INTERNAL_K);
        b.u32(0); // file consistency flags
        b.u64(base); // base address (userblock size)
        b.u64(UNDEF); // free space info address
        b.u64(eof); // end of file address
        b.u64(UNDEF); // driver info block address
                      // root group symbol table entry
        b.u64(0); // link name offset
        b.u64(root_oh); // object header address
        b.u32(1); // cache type = 1 (group with cached btree/heap)
        b.u32(0); // reserved
        b.u64(btree);
        b.u64(heap);
        debug_assert_eq!(b.len(), 96);
        self.buf.bytes[..96].copy_from_slice(&b.bytes);
    }

    fn write_object(&mut self, id: ObjId) -> Result<Ref> {
        if let Some(r) = self.memo.get(&id) {
            return Ok(*r);
        }
        if self.visiting.contains(&id) {
            return Err("cyclic hard links are not supported by the writer".into());
        }
        self.visiting.push(id);
        let node = self.state.get(id);
        let r = match &node.kind {
            ObjectKind::Group(_) => self.write_group(id)?,
            ObjectKind::Dataset(d) => {
                let d = d.clone();
                self.write_dataset(id, &d)?
            }
            ObjectKind::NamedType(t) => {
                let t = t.clone();
                let mut msgs = Vec::new();
                msgs.push((MSG_DATATYPE, 0, datatype::encode(&t)));
                let dense = self.append_comment_and_attrs(id, &mut msgs)?;
                let oh = self.write_object_header_auto(&msgs, dense)?;
                Ref {
                    oh_addr: oh,
                    cache_type: 0,
                    btree: UNDEF,
                    heap: UNDEF,
                }
            }
            ObjectKind::Unsupported(reason) => {
                // Refuse to rewrite a file containing an object this build
                // could not parse -- writing would silently drop its contents.
                return Err(format!(
                    "cannot write file: it contains an object unsupported by this build ({reason})"
                )
                .into());
            }
        };
        self.visiting.pop();
        self.memo.insert(id, r);
        Ok(r)
    }

    fn write_group(&mut self, id: ObjId) -> Result<Ref> {
        let group = match &self.state.get(id).kind {
            ObjectKind::Group(g) => g.clone(),
            _ => unreachable!(),
        };

        // Old-style symbol-table groups cannot hold external links (symbol
        // table entries only encode hard/soft targets); such groups are
        // written as new-style compact link-message groups instead -- the
        // same representation libhdf5 1.8+ uses.
        if group
            .links
            .iter()
            .any(|l| matches!(l.target, LinkTarget::External { .. }))
        {
            return self.write_group_compact(id, &group);
        }

        // Serialize members and gather symbol table entries.
        struct SymEntry {
            name: String,
            name_off: u64,
            oh_addr: u64,
            cache_type: u32,
            scratch: [u8; 16],
        }
        let mut heap = HeapBuilder::new();
        let mut entries: Vec<SymEntry> = Vec::new();

        // Sort links by name up front so heap offsets are assigned in sorted
        // order (not required by the format, but keeps files deterministic).
        let mut links = group.links.clone();
        links.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));

        for link in &links {
            match &link.target {
                LinkTarget::Hard(cid) => {
                    let r = self.write_object(*cid)?;
                    let name_off = heap.add(&link.name);
                    let mut scratch = [0u8; 16];
                    if r.cache_type == 1 {
                        scratch[..8].copy_from_slice(&r.btree.to_le_bytes());
                        scratch[8..].copy_from_slice(&r.heap.to_le_bytes());
                    }
                    entries.push(SymEntry {
                        name: link.name.clone(),
                        name_off,
                        oh_addr: r.oh_addr,
                        cache_type: r.cache_type,
                        scratch,
                    });
                }
                LinkTarget::Soft(target) => {
                    let name_off = heap.add(&link.name);
                    let value_off = heap.add(target);
                    let mut scratch = [0u8; 16];
                    scratch[..4].copy_from_slice(&(value_off as u32).to_le_bytes());
                    entries.push(SymEntry {
                        name: link.name.clone(),
                        name_off,
                        oh_addr: UNDEF,
                        cache_type: 2,
                        scratch,
                    });
                }
                LinkTarget::External { .. } => {
                    return Err("external links are not yet supported by the writer".into());
                }
            }
        }
        entries.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));

        // Write the local heap (data segment + header).
        let heap_addr = self.write_local_heap(&heap);

        // Write SNODs (max 2*leaf_K per node) and a group B-tree over them.
        let leaf_cap = (2 * GROUP_LEAF_K) as usize;
        let mut snods: Vec<(u64, u64)> = Vec::new(); // (max_name_off, snod_addr)
        for chunk in entries.chunks(leaf_cap) {
            let snod_addr = self.write_snod(
                chunk
                    .iter()
                    .map(|e| (e.name_off, e.oh_addr, e.cache_type, e.scratch)),
            );
            let max_off = chunk.last().unwrap().name_off;
            snods.push((max_off, snod_addr));
        }
        let btree_addr = self.write_group_btree(&snods)?;

        // Object header: symbol table message + comment + attributes.
        let mut msgs = Vec::new();
        let mut sym = Buf::new();
        sym.u64(btree_addr);
        sym.u64(heap_addr);
        msgs.push((MSG_SYMBOL_TABLE, 0, sym.bytes));
        let dense = self.append_comment_and_attrs(id, &mut msgs)?;
        let oh = self.write_object_header_auto(&msgs, dense)?;

        Ok(Ref {
            oh_addr: oh,
            cache_type: 1,
            btree: btree_addr,
            heap: heap_addr,
        })
    }

    /// Write a new-style "compact" group: an object header carrying Link
    /// Info + Group Info + one Link message per member.
    fn write_group_compact(&mut self, id: ObjId, group: &crate::model::GroupData) -> Result<Ref> {
        let mut msgs: Vec<(u16, u8, Vec<u8>)> = Vec::new();
        // Link Info (v0, no creation-order tracking, no dense storage)
        let mut li = Buf::new();
        li.u8(0); // version
        li.u8(0); // flags
        li.u64(UNDEF); // fractal heap address
        li.u64(UNDEF); // name-index v2 btree address
        msgs.push((MSG_LINK_INFO, 0, li.bytes));
        // Group Info (v0, defaults)
        let mut gi = Buf::new();
        gi.u8(0);
        gi.u8(0);
        msgs.push((MSG_GROUP_INFO, 0, gi.bytes));
        // more than 8 links: dense storage (fractal heap + name btree),
        // mirroring libhdf5's compact->dense transition; otherwise one Link
        // message per member
        if group.links.len() > 8 {
            let (fheap, bt2) = self.write_dense_links(group)?;
            let mut li = Buf::new();
            li.u8(0);
            li.u8(0);
            li.u64(fheap);
            li.u64(bt2);
            msgs[0] = (MSG_LINK_INFO, 0, li.bytes); // replace the UNDEF Link Info
        } else {
            let mut links = group.links.clone();
            links.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
            for link in &links {
                let target_oh = if let LinkTarget::Hard(cid) = &link.target {
                    Some(self.write_object(*cid)?.oh_addr)
                } else {
                    None
                };
                msgs.push((MSG_LINK, 0, encode_link_message(link, target_oh)?));
            }
        }
        let dense = self.append_comment_and_attrs(id, &mut msgs)?;
        let oh = self.write_object_header_auto(&msgs, dense)?;
        Ok(Ref {
            oh_addr: oh,
            cache_type: 0,
            btree: UNDEF,
            heap: UNDEF,
        })
    }

    fn write_local_heap(&mut self, heap: &HeapBuilder) -> u64 {
        let mut data = heap.data.clone();
        // Trailing free block: {next=1 (end of list), size}
        let free_off = data.len();
        let free_size = 16u64;
        let mut fb = Buf::new();
        fb.u64(1);
        fb.u64(free_size);
        data.extend_from_slice(&fb.bytes);
        let data_seg_size = data.len() as u64;
        let data_addr = self.append(&data);

        let mut h = Buf::new();
        h.raw(b"HEAP");
        h.u8(0);
        h.zeros(3);
        h.u64(data_seg_size);
        h.u64(free_off as u64);
        h.u64(data_addr);
        self.append(&h.bytes)
    }

    fn write_snod<I>(&mut self, syms: I) -> u64
    where
        I: Iterator<Item = (u64, u64, u32, [u8; 16])>,
    {
        let entries: Vec<_> = syms.collect();
        let mut b = Buf::new();
        b.raw(b"SNOD");
        b.u8(1);
        b.u8(0);
        b.u16(entries.len() as u16);
        for (name_off, oh_addr, cache_type, scratch) in &entries {
            b.u64(*name_off);
            b.u64(*oh_addr);
            b.u32(*cache_type);
            b.u32(0);
            b.raw(scratch);
        }
        // Pad node to full on-disk size: 8 + 2*leaf_K entries * 40 bytes.
        let full = 8 + (2 * GROUP_LEAF_K as usize) * 40;
        if b.len() < full {
            b.zeros(full - b.len());
        }
        self.append(&b.bytes)
    }

    /// Write a (possibly multi-level) group B-tree over the given SNODs.
    /// `snods` is a list of `(max_name_offset, snod_addr)` in sorted order.
    /// Node keys follow the convention: `key[0] = 0` (empty string) for the
    /// left-most node, and `key[i+1]` = max name offset within child `i`.
    fn write_group_btree(&mut self, snods: &[(u64, u64)]) -> Result<u64> {
        let cap = 2 * GROUP_INTERNAL_K as usize;
        let node_size = 24 + cap * 16 + 8;

        // level 0: children are SNODs
        let mut level = 0u8;
        // (first_key, addr, last_key) per node of the current level
        let mut nodes: Vec<(u64, u64, u64)> = Vec::new();
        {
            let chunks: Vec<&[(u64, u64)]> = if snods.is_empty() {
                vec![&[][..]]
            } else {
                snods.chunks(cap).collect()
            };
            let mut prev_last = 0u64;
            for chunk in chunks {
                let mut b = Buf::new();
                b.raw(b"TREE");
                b.u8(0);
                b.u8(0);
                b.u16(chunk.len() as u16);
                b.u64(UNDEF); // left sibling (patched below if needed)
                b.u64(UNDEF); // right sibling
                let first_key = prev_last;
                b.u64(first_key);
                let mut last_key = first_key;
                for (max_off, addr) in chunk {
                    b.u64(*addr);
                    b.u64(*max_off);
                    last_key = *max_off;
                }
                if b.len() < node_size {
                    b.zeros(node_size - b.len());
                }
                let addr = self.append(&b.bytes);
                nodes.push((first_key, addr, last_key));
                prev_last = last_key;
            }
            // patch sibling pointers
            self.patch_siblings(&nodes);
        }

        // build internal levels until a single root remains
        while nodes.len() > 1 {
            level += 1;
            let mut parents: Vec<(u64, u64, u64)> = Vec::new();
            let groups: Vec<&[(u64, u64, u64)]> = nodes.chunks(cap).collect();
            for grp in groups {
                let mut b = Buf::new();
                b.raw(b"TREE");
                b.u8(0);
                b.u8(level);
                b.u16(grp.len() as u16);
                b.u64(UNDEF);
                b.u64(UNDEF);
                let first_key = grp.first().unwrap().0;
                b.u64(first_key);
                let mut last_key = first_key;
                for (_, addr, last) in grp {
                    b.u64(*addr);
                    b.u64(*last);
                    last_key = *last;
                }
                if b.len() < node_size {
                    b.zeros(node_size - b.len());
                }
                let addr = self.append(&b.bytes);
                parents.push((first_key, addr, last_key));
            }
            self.patch_siblings(&parents);
            nodes = parents;
        }
        Ok(nodes[0].1)
    }

    /// Patch left/right sibling pointers of a freshly written node level.
    fn patch_siblings(&mut self, nodes: &[(u64, u64, u64)]) {
        for i in 0..nodes.len() {
            let addr = nodes[i].1 as usize;
            if i > 0 {
                self.buf.patch_u64(addr + 8, nodes[i - 1].1);
            }
            if i + 1 < nodes.len() {
                self.buf.patch_u64(addr + 16, nodes[i + 1].1);
            }
        }
    }

    /// Write a global-heap collection holding the store's buffers; returns the
    /// collection address. Object indices equal store indices (1-based).
    fn write_gheap(&mut self, store: &VlenStore) -> u64 {
        let mut b = Buf::new();
        b.raw(b"GCOL");
        b.u8(1);
        b.zeros(3);
        let size_at = b.len();
        b.u64(0); // collection size (patched)
        for (i, obj) in store.iter().enumerate() {
            b.u16((i + 1) as u16);
            b.u16(1); // refcount
            b.zeros(4);
            b.u64(obj.len() as u64);
            b.raw(obj);
            b.pad8();
        }
        // free-space object header
        let free_hdr_at = b.len();
        b.u16(0);
        b.u16(0);
        b.zeros(4);
        b.u64(0); // free size (patched)
                  // pad collection to a 4096 multiple like libhdf5 (harmless, plays nice
                  // with C-library heap caching)
        let total = b.len().div_ceil(4096) * 4096;
        b.zeros(total - b.len());
        b.patch_u64(size_at, total as u64);
        b.patch_u64(free_hdr_at + 8, (total - free_hdr_at) as u64);
        self.append(&b.bytes)
    }

    /// Rewrite model vlen slots `{len, store_idx}` into on-disk global heap
    /// references `{len, collection_addr, obj_idx}` throughout `data`.
    fn fixup_vlen(
        &mut self,
        dtype: &hdf5_types::TypeDescriptor,
        data: &mut [u8],
        n: usize,
        gcol_addr: u64,
    ) {
        let esize = disk_size(dtype);
        for i in 0..n {
            let start = i * esize;
            if start + esize <= data.len() {
                fixup_vlen_elem(dtype, &mut data[start..start + esize], gcol_addr);
            }
        }
    }

    fn write_dataset(&mut self, id: ObjId, d: &DatasetData) -> Result<Ref> {
        if d.lazy.is_some() {
            return Err("internal: lazy dataset not materialized before serialization".into());
        }
        let elem_size = disk_size(&d.dtype);
        let n_elems = d.num_elements();

        // Resolve vlen store into a real global-heap collection first.
        let mut data = d.data.clone();
        if has_vlen(&d.dtype) && !d.vlen.is_empty() {
            let gcol = self.write_gheap(&d.vlen);
            self.fixup_vlen(&d.dtype.clone(), &mut data, n_elems, gcol);
        }

        let chunk_elems = match &d.layout {
            LayoutClass::Chunked(c) => c.iter().map(|&x| x as usize).product(),
            _ => n_elems,
        };
        let raw_filters = lower_filters(&d.filters, d, chunk_elems.max(1));

        // Data layout message + data blocks.
        let layout_msg = match &d.layout {
            LayoutClass::Virtual(mappings) => {
                let blob = super::vds::encode_vds_blob(mappings)?;
                let store = vec![blob];
                let gcol = self.write_gheap(&store);
                let mut m = Buf::new();
                m.u8(4); // layout version 4 (virtual requires >= 4)
                m.u8(3); // class: virtual
                m.u64(gcol);
                m.u32(1); // global heap object index
                m.bytes
            }
            LayoutClass::Contiguous | LayoutClass::Compact => {
                let (data_addr, size) = if data.is_empty() {
                    (UNDEF, (n_elems * elem_size) as u64)
                } else {
                    (self.append(&data), data.len() as u64)
                };
                let mut m = Buf::new();
                m.u8(3); // layout version 3
                m.u8(1); // contiguous
                m.u64(data_addr);
                m.u64(size);
                m.bytes
            }
            LayoutClass::Chunked(chunk_dims) => {
                let btree = self.write_chunks(d, &data, chunk_dims, &raw_filters, elem_size)?;
                let rank = d.dims.len();
                let mut m = Buf::new();
                m.u8(3); // layout version 3
                m.u8(2); // chunked
                m.u8((rank + 1) as u8); // dimensionality (incl. element-size dim)
                m.u64(btree);
                for &c in chunk_dims {
                    m.u32(c as u32);
                }
                m.u32(elem_size as u32); // element size (last "dimension")
                m.bytes
            }
        };

        let mut msgs = Vec::new();
        let (f1, b1) = self.maybe_share(
            MSG_DATASPACE,
            encode_dataspace(&d.dims, &d.maxdims, d.is_scalar, d.is_null),
        );
        msgs.push((MSG_DATASPACE, f1, b1));
        let (f2, b2) = self.maybe_share(MSG_DATATYPE, datatype::encode(&d.dtype));
        msgs.push((MSG_DATATYPE, f2, b2));
        let (f3, b3) = self.maybe_share(MSG_FILL, encode_fill(&d.fill));
        msgs.push((MSG_FILL, f3, b3));
        if !raw_filters.is_empty() {
            msgs.push((MSG_FILTER, 0, encode_filter_pipeline(&raw_filters)));
        }
        msgs.push((MSG_LAYOUT, 0, layout_msg));
        let dense = self.append_comment_and_attrs(id, &mut msgs)?;
        let oh = self.write_object_header_auto(&msgs, dense)?;
        Ok(Ref {
            oh_addr: oh,
            cache_type: 0,
            btree: UNDEF,
            heap: UNDEF,
        })
    }

    /// Split a dataset's logical data into chunks, filter each, write the chunk
    /// data blocks, and build the chunk B-tree; return the B-tree address.
    fn write_chunks(
        &mut self,
        d: &DatasetData,
        data: &[u8],
        chunk_dims: &[u64],
        raw_filters: &[filt::RawFilter],
        elem_size: usize,
    ) -> Result<u64> {
        let rank = d.dims.len();
        let chunk_elems: usize = chunk_dims.iter().map(|&c| c as usize).product();
        let chunk_bytes = chunk_elems * elem_size;

        // number of chunks along each dimension
        let nchunks: Vec<usize> = d
            .dims
            .iter()
            .zip(chunk_dims.iter())
            .map(|(&dim, &c)| ((dim).div_ceil(c)).max(1) as usize)
            .collect();
        let total_chunks: usize = nchunks.iter().product();

        struct ChunkEntry {
            offsets: Vec<u64>, // rank+1 coords (last = 0)
            size: u32,
            filter_mask: u32,
            addr: u64,
        }
        let mut chunk_entries: Vec<ChunkEntry> = Vec::new();

        let mut idx = vec![0usize; rank];
        for _ in 0..total_chunks {
            let base: Vec<u64> = idx
                .iter()
                .zip(chunk_dims)
                .map(|(&i, &c)| i as u64 * c)
                .collect();
            let mut chunk = vec![0u8; chunk_bytes];
            gather_chunk(data, &d.dims, chunk_dims, &base, elem_size, &mut chunk);

            let filtered = if raw_filters.is_empty() {
                chunk
            } else {
                filt::apply(raw_filters, elem_size, &chunk)?
            };
            let size = filtered.len() as u32;
            let addr = self.append(&filtered);
            let mut offsets = base.clone();
            offsets.push(0);
            chunk_entries.push(ChunkEntry {
                offsets,
                size,
                filter_mask: 0,
                addr,
            });

            // advance N-D chunk index (row-major)
            let mut k = rank;
            while k > 0 {
                k -= 1;
                idx[k] += 1;
                if idx[k] < nchunks[k] {
                    break;
                }
                idx[k] = 0;
            }
        }

        // Encode a chunk key.
        let key_size = 8 + (rank + 1) * 8;
        let enc_key = |size: u32, mask: u32, offs: &[u64]| -> Vec<u8> {
            let mut k = Buf::new();
            k.u32(size);
            k.u32(mask);
            for &o in offs {
                k.u64(o);
            }
            k.bytes
        };
        // "End" key: strictly greater than every chunk offset — bump the most
        // significant dimension by one chunk, put the element size last
        // (matches libhdf5's right-most key).
        let end_key = {
            let mut offs: Vec<u64> = chunk_entries
                .last()
                .map(|e| e.offsets.clone())
                .unwrap_or_else(|| vec![0; rank + 1]);
            if rank > 0 {
                offs[0] += chunk_dims[0];
            }
            offs[rank] = elem_size as u64;
            enc_key(0, 0, &offs)
        };

        // Build leaf nodes.
        let cap = 2 * CHUNK_K as usize;
        let node_size = 24 + cap * (key_size + 8) + key_size;
        struct BtNode {
            first_key: Vec<u8>,
            last_key: Vec<u8>,
            addr: u64,
        }
        let mut nodes: Vec<BtNode> = Vec::new();
        let groups: Vec<&[ChunkEntry]> = if chunk_entries.is_empty() {
            vec![&[][..]]
        } else {
            chunk_entries.chunks(cap).collect()
        };
        let ngroups = groups.len();
        for (gi, grp) in groups.into_iter().enumerate() {
            // trailing key = first key of next group, or the end key
            let trailing = if gi + 1 < ngroups {
                let ne = &chunk_entries[(gi + 1) * cap];
                enc_key(ne.size, ne.filter_mask, &ne.offsets)
            } else {
                end_key.clone()
            };
            let mut b = Buf::new();
            b.raw(b"TREE");
            b.u8(1);
            b.u8(0);
            b.u16(grp.len() as u16);
            b.u64(UNDEF);
            b.u64(UNDEF);
            let first_key = grp
                .first()
                .map(|e| enc_key(e.size, e.filter_mask, &e.offsets))
                .unwrap_or_else(|| enc_key(0, 0, &vec![0u64; rank + 1]));
            for e in grp {
                b.raw(&enc_key(e.size, e.filter_mask, &e.offsets));
                b.u64(e.addr);
            }
            b.raw(&trailing);
            if b.len() < node_size {
                b.zeros(node_size - b.len());
            }
            let addr = self.append(&b.bytes);
            nodes.push(BtNode {
                first_key,
                last_key: trailing,
                addr,
            });
        }
        // sibling pointers
        let addrs: Vec<(u64, u64, u64)> = nodes.iter().map(|n| (0, n.addr, 0)).collect();
        self.patch_siblings(&addrs);

        // Internal levels.
        let mut level = 0u8;
        while nodes.len() > 1 {
            level += 1;
            let mut parents: Vec<BtNode> = Vec::new();
            let idxs: Vec<usize> = (0..nodes.len()).collect();
            for grp in idxs.chunks(cap) {
                let mut b = Buf::new();
                b.raw(b"TREE");
                b.u8(1);
                b.u8(level);
                b.u16(grp.len() as u16);
                b.u64(UNDEF);
                b.u64(UNDEF);
                for &i in grp {
                    b.raw(&nodes[i].first_key);
                    b.u64(nodes[i].addr);
                }
                let last = &nodes[*grp.last().unwrap()];
                b.raw(&last.last_key);
                if b.len() < node_size {
                    b.zeros(node_size - b.len());
                }
                let first_key = nodes[grp[0]].first_key.clone();
                let last_key = last.last_key.clone();
                let addr = self.append(&b.bytes);
                parents.push(BtNode {
                    first_key,
                    last_key,
                    addr,
                });
            }
            let addrs: Vec<(u64, u64, u64)> = parents.iter().map(|n| (0, n.addr, 0)).collect();
            self.patch_siblings(&addrs);
            nodes = parents;
        }
        Ok(nodes[0].addr)
    }

    /// Append comment/mtime/attribute messages; returns `true` when dense
    /// attribute storage was used (requiring a version-2 object header).
    fn append_comment_and_attrs(
        &mut self,
        id: ObjId,
        msgs: &mut Vec<(u16, u8, Vec<u8>)>,
    ) -> Result<bool> {
        let node = self.state.get(id);
        if node.mtime != 0 {
            let mut b = Buf::new();
            b.u8(1); // version
            b.zeros(3);
            b.u32(node.mtime);
            msgs.push((MSG_MOD_TIME, 0, b.bytes));
        }
        if let Some(comment) = node.comment.clone() {
            let mut b = Buf::new();
            b.raw(comment.as_bytes());
            b.u8(0);
            msgs.push((MSG_COMMENT, 0, b.bytes));
        }
        let attrs = node.attrs.clone();
        let mut encoded = Vec::with_capacity(attrs.len());
        let mut oversized = false;
        for attr in &attrs {
            let msg = self.encode_attribute(attr)?;
            // v1 header messages carry a 16-bit size; larger attributes are
            // stored densely (fractal heap + v2 btree), like libhdf5 1.8+.
            if super::align8(msg.len()) > u16::MAX as usize {
                oversized = true;
            }
            encoded.push(msg);
        }
        if !oversized {
            for msg in encoded {
                msgs.push((MSG_ATTRIBUTE, 0, msg));
            }
            Ok(false)
        } else {
            let (fheap, bt2) = self.write_dense_attributes(&attrs)?;
            let mut ai = Buf::new();
            ai.u8(0); // version
            ai.u8(0); // flags: no creation-order tracking
            ai.u64(fheap);
            ai.u64(bt2);
            msgs.push((MSG_ATTRIBUTE_INFO, 0, ai.bytes));
            Ok(true)
        }
    }

    /// If SOHM is configured for this message type and the body is large
    /// enough, deduplicate it into the shared store and return a shared-ref
    /// message (v3, SOHM) with the "shared" flag; otherwise pass through.
    fn maybe_share(&mut self, mtype: u16, body: Vec<u8>) -> (u8, Vec<u8>) {
        let flag = 1u16 << (mtype & 0x0f);
        let Some(ix) = self
            .sohm
            .iter()
            .position(|s| s.flags & flag != 0 && body.len() >= s.min_size as usize)
        else {
            return (0, body);
        };
        if body.len() > FH_MAX_MAN_SIZE {
            return (0, body); // keep huge messages inline
        }
        let store = &mut self.sohm[ix];
        let obj = match store.map.get(&body) {
            Some(&i) => {
                store.refcounts[i] += 1;
                i
            }
            None => {
                let i = store.objects.len();
                store.hashes.push(super::checksum::checksum(&body));
                store.refcounts.push(1);
                store.map.insert(body.clone(), i);
                store.objects.push(body);
                // heap ids are assigned when the heap is emitted; reserve slot
                store.ids.push(Vec::new());
                i
            }
        };
        // A placeholder shared ref; the real heap id is patched in
        // write_sohm_tables (ids are stable under append, so we record the
        // patch position instead of precomputing the allocator).
        let mut b = Buf::new();
        b.u8(3); // shared message version 3
        b.u8(1); // type: stored in SOHM heap
                 // patch marker: index + object number, fixed 8 bytes
        b.u32(ix as u32);
        b.u32(obj as u32);
        (0x02, b.bytes)
    }

    /// Emit the SOHM heaps, lists, master table and superblock-extension
    /// object header; patch all shared refs with real heap IDs. Returns the
    /// extension address when SOHM is active.
    fn write_sohm_tables(&mut self) -> Result<Option<u64>> {
        if self.sohm.is_empty() {
            return Ok(None);
        }
        // 1. heaps (ids assigned here)
        let mut heap_addrs = Vec::with_capacity(self.sohm.len());
        for ix in 0..self.sohm.len() {
            let objects = self.sohm[ix].objects.clone();
            let (addr, ids) = if objects.is_empty() {
                (UNDEF, Vec::new())
            } else {
                self.write_fractal_heap(&objects, FH_ID_LEN, FH_MAX_HEAP_BITS)?
            };
            self.sohm[ix].ids = ids;
            heap_addrs.push(addr);
        }
        // 2. patch shared refs: scan for the v3/SOHM marker pattern we wrote
        //    (2-byte prefix "\x03\x01" + ix u32 + obj u32) inside object
        //    headers is unambiguous because we recorded exact positions...
        //    positions were not recorded; instead rewrite via the marker map.
        self.patch_sohm_refs()?;
        // 3. lists + master table
        let mut list_bufs = Vec::with_capacity(self.sohm.len());
        for s in &self.sohm {
            let list_max = 512usize.max(s.objects.len());
            let mut l = Buf::new();
            l.raw(b"SMLI");
            let mut order: Vec<usize> = (0..s.objects.len()).collect();
            order.sort_by_key(|&i| s.hashes[i]);
            for &i in &order {
                l.u8(0); // location: in heap
                l.u32(s.hashes[i]);
                l.u32(s.refcounts[i]);
                l.raw(&s.ids[i]);
            }
            let sum = super::checksum::checksum(&l.bytes);
            l.u32(sum);
            let full = 4 + list_max * 17 + 4;
            if l.len() < full {
                l.zeros(full - l.len());
            }
            list_bufs.push(l.bytes);
        }
        let mut index_addrs = Vec::with_capacity(list_bufs.len());
        for b in &list_bufs {
            let b = b.clone();
            index_addrs.push(self.append(&b));
        }
        let mut t = Buf::new();
        t.raw(b"SMTB");
        for (i, s) in self.sohm.iter().enumerate() {
            t.u8(0); // list version
            t.u8(1); // index type: list
            t.u16(s.flags);
            t.u32(s.min_size);
            t.u16(512u16.max(s.objects.len() as u16)); // list max
            t.u16(40); // btree min
            t.u16(s.objects.len() as u16);
            t.u64(index_addrs[i]);
            t.u64(heap_addrs[i]);
        }
        let sum = super::checksum::checksum(&t.bytes);
        t.u32(sum);
        let table_addr = self.append(&t.bytes);
        // 4. superblock extension object header with the 0x000F message
        let mut sm = Buf::new();
        sm.u8(0); // version
        sm.u64(table_addr);
        sm.u8(self.sohm.len() as u8);
        let msgs = vec![(0x000Fu16, 0u8, sm.bytes)];
        let ext = self.write_object_header(&msgs)?;
        Ok(Some(ext))
    }

    /// Patch shared-ref placeholders with real heap IDs (positions recorded
    /// while writing object headers).
    fn patch_sohm_refs(&mut self) -> Result<()> {
        for (pos, (ix, obj)) in std::mem::take(&mut self.sohm_patches) {
            let id = self.sohm[ix].ids[obj].clone();
            self.buf.bytes[pos..pos + 8].copy_from_slice(&id);
        }
        Ok(())
    }

    /// Write a version-2 superblock pointing at a superblock extension.
    fn write_superblock_v2(&mut self, root_oh: u64, ext_addr: u64, eof: u64, base: u64) {
        let mut b = Buf::new();
        b.raw(&SIGNATURE);
        b.u8(2); // superblock version
        b.u8(8); // size of offsets
        b.u8(8); // size of lengths
        b.u8(0); // file consistency flags
        b.u64(base); // base address (userblock size)
        b.u64(ext_addr);
        b.u64(eof);
        b.u64(root_oh);
        let sum = super::checksum::checksum(&b.bytes);
        b.u32(sum);
        self.buf.bytes[..b.len()].copy_from_slice(&b.bytes);
    }

    /// Write dense attribute storage: serialized v3 attribute messages in a
    /// fractal heap, indexed by a name v2 btree. Returns (fheap, name_bt2).
    fn write_dense_attributes(&mut self, attrs: &[AttrData]) -> Result<(u64, u64)> {
        let mut names = Vec::with_capacity(attrs.len());
        let mut objects = Vec::with_capacity(attrs.len());
        for attr in attrs {
            names.push(attr.name.clone());
            objects.push(self.encode_attribute_v3(attr)?);
        }
        let (fheap_addr, heap_ids) =
            self.write_fractal_heap(&objects, FH_ID_LEN, FH_MAX_HEAP_BITS)?;
        // name-index v2 btree (type 8): {heap id 8, flags 1, corder 4, hash 4}
        let mut records: Vec<Vec<u8>> = Vec::with_capacity(objects.len());
        for (i, name) in names.iter().enumerate() {
            let hash = super::checksum::checksum(name.as_bytes());
            let mut rec = Buf::new();
            rec.raw(&heap_ids[i]);
            rec.u8(0);
            rec.u32(i as u32);
            rec.u32(hash);
            records.push(rec.bytes);
        }
        records.sort_by_key(|r| u32::from_le_bytes(r[13..17].try_into().unwrap()));
        let bt2 = self.write_v2btree_leaf(8, 17, &records)?;
        Ok((fheap_addr, bt2))
    }

    /// Write dense link storage: serialized link messages in a fractal heap
    /// (7-byte heap ids), indexed by a name v2 btree (type 5).
    fn write_dense_links(&mut self, group: &crate::model::GroupData) -> Result<(u64, u64)> {
        let mut names = Vec::with_capacity(group.links.len());
        let mut objects = Vec::with_capacity(group.links.len());
        for link in &group.links {
            let target_oh = if let LinkTarget::Hard(cid) = &link.target {
                Some(self.write_object(*cid)?.oh_addr)
            } else {
                None
            };
            names.push(link.name.clone());
            objects.push(encode_link_message(link, target_oh)?);
        }
        // H5G dense heaps use 7-byte ids: offset 4 bytes (32 heap bits) + len 2
        let (fheap_addr, heap_ids) = self.write_fractal_heap(&objects, 7, 32)?;
        // name-index v2 btree (type 5): {hash 4, heap id 7}
        let mut records: Vec<Vec<u8>> = Vec::with_capacity(objects.len());
        for (i, name) in names.iter().enumerate() {
            let hash = super::checksum::checksum(name.as_bytes());
            let mut rec = Buf::new();
            rec.u32(hash);
            rec.raw(&heap_ids[i]);
            records.push(rec.bytes);
        }
        records.sort_by_key(|r| u32::from_le_bytes(r[0..4].try_into().unwrap()));
        let bt2 = self.write_v2btree_leaf(5, 11, &records)?;
        Ok((fheap_addr, bt2))
    }

    /// Write a fractal heap holding `objects`; returns the header address and
    /// one heap ID (of `id_len` bytes) per object.
    fn write_fractal_heap(
        &mut self,
        objects: &[Vec<u8>],
        id_len: usize,
        max_heap_bits: usize,
    ) -> Result<(u64, Vec<Vec<u8>>)> {
        const FRHP_SIZE: usize = 4 + 1 + 2 + 2 + 1 + 4 + 8 * 12 + 2 + 8 + 8 + 2 + 2 + 8 + 2 + 4;
        self.buf.pad8();
        let fheap_addr = self.buf.len() as u64;
        self.buf.zeros(FRHP_SIZE);

        let heap_off_size = max_heap_bits.div_ceil(8);
        let heap_len_size = 2usize;
        debug_assert!(1 + heap_off_size + heap_len_size <= id_len);
        let dblock_prefix = 4 + 1 + 8 + heap_off_size; // no dblock checksums

        struct DBlock {
            block_off: u64,
            size: u64,
            data: Vec<u8>,
        }
        let row_size = |r: usize| -> u64 {
            if r == 0 {
                FH_START_BLOCK
            } else {
                FH_START_BLOCK << (r - 1)
            }
        };
        let mut dblocks: Vec<DBlock> = Vec::new();
        let mut next_slot = 0usize;
        let mut heap_ids: Vec<Vec<u8>> = vec![vec![0u8; id_len]; objects.len()];
        let mut huge: Vec<(usize, Vec<u8>)> = Vec::new();
        let mut man_nobjs = 0u64;

        for (i, obj) in objects.iter().enumerate() {
            if obj.len() > FH_MAX_MAN_SIZE {
                huge.push((i, obj.clone()));
                continue;
            }
            while !dblocks
                .last()
                .map(|b| b.data.len() + obj.len() <= b.size as usize)
                .unwrap_or(false)
            {
                let row = next_slot / FH_WIDTH;
                let col = next_slot % FH_WIDTH;
                next_slot += 1;
                let mut block_off = 0u64;
                for r in 0..row {
                    block_off += row_size(r) * FH_WIDTH as u64;
                }
                block_off += row_size(row) * col as u64;
                let size = row_size(row);
                if size > FH_MAX_DIRECT {
                    return Err("dense storage exceeds direct-block capacity".into());
                }
                let mut data = Vec::with_capacity(size as usize);
                data.resize(dblock_prefix, 0);
                dblocks.push(DBlock {
                    block_off,
                    size,
                    data,
                });
            }
            let b = dblocks.last_mut().unwrap();
            let heap_off = b.block_off + b.data.len() as u64;
            b.data.extend_from_slice(obj);
            man_nobjs += 1;
            let id = &mut heap_ids[i];
            id[0] = 0x00; // managed
            id[1..1 + heap_off_size].copy_from_slice(&heap_off.to_le_bytes()[..heap_off_size]);
            id[1 + heap_off_size..1 + heap_off_size + heap_len_size]
                .copy_from_slice(&(obj.len() as u64).to_le_bytes()[..heap_len_size]);
        }

        let mut dblock_addrs = Vec::with_capacity(dblocks.len());
        for b in &mut dblocks {
            let mut d = Buf::new();
            d.raw(b"FHDB");
            d.u8(0);
            d.u64(fheap_addr);
            d.raw(&b.block_off.to_le_bytes()[..heap_off_size]);
            debug_assert_eq!(d.len(), dblock_prefix);
            d.raw(&b.data[dblock_prefix..]);
            d.zeros(b.size as usize - d.len());
            dblock_addrs.push(self.append(&d.bytes));
        }

        let (root_addr, cur_rows, man_size) = match dblocks.len() {
            0 => (UNDEF, 0u16, 0u64),
            1 => (dblock_addrs[0], 0u16, dblocks[0].size),
            n => {
                let nrows = (n).div_ceil(FH_WIDTH);
                let mut ib = Buf::new();
                ib.raw(b"FHIB");
                ib.u8(0);
                ib.u64(fheap_addr);
                ib.zeros(heap_off_size);
                for slot in 0..nrows * FH_WIDTH {
                    ib.u64(dblock_addrs.get(slot).copied().unwrap_or(UNDEF));
                }
                let sum = super::checksum::checksum(&ib.bytes);
                ib.u32(sum);
                let addr = self.append(&ib.bytes);
                (addr, nrows as u16, dblocks.iter().map(|b| b.size).sum())
            }
        };

        let mut huge_bt2 = UNDEF;
        let mut huge_size = 0u64;
        let huge_id_size = (id_len - 1).min(8);
        if !huge.is_empty() {
            let mut records = Vec::with_capacity(huge.len());
            for (n, (i, obj)) in huge.iter().enumerate() {
                let addr = self.append(obj);
                huge_size += obj.len() as u64;
                let hid = (n + 1) as u64;
                let mut rec = Buf::new();
                rec.u64(addr);
                rec.u64(obj.len() as u64);
                rec.u64(hid);
                records.push(rec.bytes);
                let id = &mut heap_ids[*i];
                id[0] = 0x10; // huge
                id[1..1 + huge_id_size].copy_from_slice(&hid.to_le_bytes()[..huge_id_size]);
            }
            huge_bt2 = self.write_v2btree_leaf(1, 24, &records)?;
        }

        let mut h = Buf::new();
        h.raw(b"FRHP");
        h.u8(0);
        h.u16(id_len as u16);
        h.u16(0);
        h.u8(0);
        h.u32(FH_MAX_MAN_SIZE as u32);
        h.u64(huge.len() as u64 + 1);
        h.u64(huge_bt2);
        h.u64(0);
        h.u64(UNDEF);
        h.u64(man_size);
        h.u64(man_size);
        h.u64(man_size);
        h.u64(man_nobjs);
        h.u64(huge_size);
        h.u64(huge.len() as u64);
        h.u64(0);
        h.u64(0);
        h.u16(FH_WIDTH as u16);
        h.u64(FH_START_BLOCK);
        h.u64(FH_MAX_DIRECT);
        h.u16(max_heap_bits as u16);
        h.u16(1);
        h.u64(root_addr);
        h.u16(cur_rows);
        let sum = super::checksum::checksum(&h.bytes);
        h.u32(sum);
        debug_assert_eq!(h.len(), FRHP_SIZE);
        self.buf.bytes[fheap_addr as usize..fheap_addr as usize + FRHP_SIZE]
            .copy_from_slice(&h.bytes);
        Ok((fheap_addr, heap_ids))
    }

    /// Write a depth-0 v2 btree (header + one leaf) holding `records`.
    fn write_v2btree_leaf(
        &mut self,
        btree_type: u8,
        rrec_size: usize,
        records: &[Vec<u8>],
    ) -> Result<u64> {
        // leaf: sig4 ver1 type1 + records + checksum4, padded to node_size
        let used = 4 + 1 + 1 + records.len() * rrec_size + 4;
        let node_size = used.max(512);
        let mut leaf = Buf::new();
        leaf.raw(b"BTLF");
        leaf.u8(0);
        leaf.u8(btree_type);
        for r in records {
            debug_assert_eq!(r.len(), rrec_size);
            leaf.raw(r);
        }
        let sum = super::checksum::checksum(&leaf.bytes);
        leaf.u32(sum);
        leaf.zeros(node_size - leaf.len());
        let leaf_addr = self.append(&leaf.bytes);

        let mut h = Buf::new();
        h.raw(b"BTHD");
        h.u8(0);
        h.u8(btree_type);
        h.u32(node_size as u32);
        h.u16(rrec_size as u16);
        h.u16(0); // depth
        h.u8(ATTR_BT2_SPLIT);
        h.u8(ATTR_BT2_MERGE);
        h.u64(leaf_addr);
        h.u16(records.len() as u16);
        h.u64(records.len() as u64);
        let sum = super::checksum::checksum(&h.bytes);
        h.u32(sum);
        Ok(self.append(&h.bytes))
    }

    /// Encode an attribute message using version 3 (unpadded, with charset),
    /// the encoding libhdf5 uses inside dense storage.
    fn encode_attribute_v3(&mut self, attr: &AttrData) -> Result<Vec<u8>> {
        let mut data = attr.data.clone();
        if has_vlen(&attr.dtype) && !attr.vlen.is_empty() {
            let gcol = self.write_gheap(&attr.vlen);
            self.fixup_vlen(&attr.dtype.clone(), &mut data, attr.num_elements(), gcol);
        }
        let dt = datatype::encode(&attr.dtype);
        let ds = encode_dataspace(
            &attr.dims,
            &attr.dims.iter().map(|&d| Some(d)).collect::<Vec<_>>(),
            attr.is_scalar,
            attr.is_null,
        );
        let mut b = Buf::new();
        b.u8(3); // version
        b.u8(0); // flags
        b.u16((attr.name.len() + 1) as u16);
        b.u16(dt.len() as u16);
        b.u16(ds.len() as u16);
        b.u8(if attr.name.is_ascii() { 0 } else { 1 }); // charset
        b.raw(attr.name.as_bytes());
        b.u8(0);
        b.raw(&dt);
        b.raw(&ds);
        b.raw(&data);
        Ok(b.bytes)
    }

    fn encode_attribute(&mut self, attr: &AttrData) -> Result<Vec<u8>> {
        // Resolve vlen payloads into a global heap collection.
        let mut data = attr.data.clone();
        if has_vlen(&attr.dtype) && !attr.vlen.is_empty() {
            let gcol = self.write_gheap(&attr.vlen);
            self.fixup_vlen(&attr.dtype.clone(), &mut data, attr.num_elements(), gcol);
        }

        let mut b = Buf::new();
        b.u8(1); // version 1
        b.u8(0); // reserved
        let name_len = attr.name.len() + 1; // incl. NUL
        let dt = datatype::encode(&attr.dtype);
        let ds = encode_dataspace(
            &attr.dims,
            &attr.dims.iter().map(|&d| Some(d)).collect::<Vec<_>>(),
            attr.is_scalar,
            attr.is_null,
        );
        b.u16(name_len as u16);
        b.u16(dt.len() as u16);
        b.u16(ds.len() as u16);
        let start = b.len();
        b.raw(attr.name.as_bytes());
        b.u8(0);
        b.zeros(align8(name_len) - (b.len() - start));
        let start = b.len();
        b.raw(&dt);
        b.zeros(align8(dt.len()) - (b.len() - start));
        let start = b.len();
        b.raw(&ds);
        b.zeros(align8(ds.len()) - (b.len() - start));
        b.raw(&data);
        Ok(b.bytes)
    }

    /// Write a v1 or v2 object header depending on `dense` (libhdf5 only
    /// honors dense attribute storage in version-2 headers).
    fn write_object_header_auto(
        &mut self,
        msgs: &[(u16, u8, Vec<u8>)],
        dense: bool,
    ) -> Result<u64> {
        if dense {
            self.write_object_header_v2(msgs)
        } else {
            self.write_object_header(msgs)
        }
    }

    /// Write a version-2 ("OHDR") object header: unaligned 4-byte message
    /// prefixes and a trailing Jenkins checksum over the whole chunk.
    fn write_object_header_v2(&mut self, msgs: &[(u16, u8, Vec<u8>)]) -> Result<u64> {
        let mut body = Buf::new();
        for (ty, mflags, data) in msgs {
            if *ty > 0xff {
                return Err("v2 object header message type out of range".into());
            }
            if data.len() > u16::MAX as usize {
                return Err("object header message too large".into());
            }
            body.u8(*ty as u8);
            body.u16(data.len() as u16);
            body.u8(*mflags);
            body.raw(data);
        }
        let mut h = Buf::new();
        h.raw(b"OHDR");
        h.u8(2); // version
        h.u8(0x02); // flags: chunk-0 size stored as u32; nothing else
        h.u32(body.len() as u32);
        h.raw(&body.bytes);
        let sum = super::checksum::checksum(&h.bytes);
        h.u32(sum);
        let addr = self.append(&h.bytes);
        self.record_sohm_patches(msgs, addr as usize + 10, 4, false);
        Ok(addr)
    }

    /// Write a v1 object header holding the given messages, exactly sized.
    fn write_object_header(&mut self, msgs: &[(u16, u8, Vec<u8>)]) -> Result<u64> {
        let mut body = Buf::new();
        for (ty, mflags, data) in msgs {
            body.u16(*ty);
            let _ = mflags;
            let padded = align8(data.len());
            // v1 header message sizes are 16-bit; larger payloads (e.g. an
            // attribute holding >64KB of data) cannot be represented and
            // would silently corrupt the file if truncated.
            if padded > u16::MAX as usize {
                return Err(format!(
                    "object header message too large ({padded} bytes > 65535); \
                     store large data in a dataset instead of an attribute"
                )
                .into());
            }
            body.u16(padded as u16);
            body.u8(*mflags);
            body.zeros(3);
            body.raw(data);
            body.zeros(padded - data.len());
        }
        let hdr_size = body.len();

        let mut h = Buf::new();
        h.u8(1); // version
        h.u8(0); // reserved
        h.u16(msgs.len() as u16);
        h.u32(1); // reference count
        h.u32(hdr_size as u32);
        h.zeros(4); // pad to 16
        h.raw(&body.bytes);
        let addr = self.append(&h.bytes);
        self.record_sohm_patches(msgs, addr as usize + 16, 8, true);
        Ok(addr)
    }

    /// Record heap-id patch positions for shared-ref messages within a header
    /// just written at `base` (message headers of `hdr_len` bytes each).
    fn record_sohm_patches(
        &mut self,
        msgs: &[(u16, u8, Vec<u8>)],
        base: usize,
        hdr_len: usize,
        pad8: bool,
    ) {
        if self.sohm.is_empty() {
            return;
        }
        let mut off = base;
        for (_, mflags, data) in msgs {
            let body = off + hdr_len;
            if mflags & 0x02 != 0 && data.len() >= 10 && data[0] == 3 && data[1] == 1 {
                let ix = u32::from_le_bytes(data[2..6].try_into().unwrap()) as usize;
                let obj = u32::from_le_bytes(data[6..10].try_into().unwrap()) as usize;
                self.sohm_patches.push((body + 2, (ix, obj)));
            }
            let len = if pad8 { align8(data.len()) } else { data.len() };
            off = body + len;
        }
    }
}

/// Rewrite one element's vlen slots from `{len, store_idx}` to
/// `{len, gcol_addr, obj_idx}`.
fn fixup_vlen_elem(desc: &hdf5_types::TypeDescriptor, elem: &mut [u8], gcol_addr: u64) {
    use hdf5_types::TypeDescriptor::*;
    match desc {
        VarLenAscii | VarLenUnicode | VarLenArray(_) => {
            let (len, idx) = slot_parts(elem);
            elem[..4].copy_from_slice(&len.to_le_bytes());
            if idx == 0 {
                elem[4..12].copy_from_slice(&0u64.to_le_bytes());
                elem[12..16].copy_from_slice(&0u32.to_le_bytes());
            } else {
                elem[4..12].copy_from_slice(&gcol_addr.to_le_bytes());
                elem[12..16].copy_from_slice(&idx.to_le_bytes());
            }
        }
        Compound(c) => {
            for f in &c.fields {
                let fs = disk_size(&f.ty);
                if f.offset + fs <= elem.len() {
                    fixup_vlen_elem(&f.ty, &mut elem[f.offset..f.offset + fs], gcol_addr);
                }
            }
        }
        FixedArray(base, n) => {
            let bs = disk_size(base);
            for i in 0..*n {
                fixup_vlen_elem(base, &mut elem[i * bs..(i + 1) * bs], gcol_addr);
            }
        }
        _ => {}
    }
}

/// Encode a version-1 Link message (object header type 0x0006).
fn encode_link_message(link: &crate::model::Link, hard_target_oh: Option<u64>) -> Result<Vec<u8>> {
    let mut b = Buf::new();
    b.u8(1); // version
    let name_len = link.name.len();
    let len_class: u8 = if name_len < 0x100 {
        0
    } else if name_len < 0x1_0000 {
        1
    } else {
        2
    };
    let type_present = !matches!(link.target, LinkTarget::Hard(_));
    let mut flags = len_class;
    if type_present {
        flags |= 0x08;
    }
    if link.utf8 {
        flags |= 0x10;
    }
    b.u8(flags);
    if type_present {
        b.u8(match &link.target {
            LinkTarget::Hard(_) => 0,
            LinkTarget::Soft(_) => 1,
            LinkTarget::External { .. } => 64,
        });
    }
    if link.utf8 {
        b.u8(1); // charset = UTF-8
    }
    match len_class {
        0 => b.u8(name_len as u8),
        1 => b.u16(name_len as u16),
        _ => b.u32(name_len as u32),
    }
    b.raw(link.name.as_bytes());
    match &link.target {
        LinkTarget::Hard(_) => {
            b.u64(hard_target_oh.ok_or("missing hard link target")?);
        }
        LinkTarget::Soft(path) => {
            b.u16(path.len() as u16);
            b.raw(path.as_bytes());
        }
        LinkTarget::External { file, path } => {
            let blob_len = 1 + file.len() + 1 + path.len() + 1;
            b.u16(blob_len as u16);
            b.u8(0); // external link version/flags byte
            b.raw(file.as_bytes());
            b.u8(0);
            b.raw(path.as_bytes());
            b.u8(0);
        }
    }
    Ok(b.bytes)
}

/// A local-heap data segment builder that deduplicates strings and 8-aligns
/// each entry; offset 0 is the reserved empty-string sentinel.
struct HeapBuilder {
    data: Vec<u8>,
    offsets: HashMap<String, u64>,
}

impl HeapBuilder {
    fn new() -> Self {
        Self {
            data: vec![0u8; 8],
            offsets: HashMap::new(),
        }
    }

    fn add(&mut self, s: &str) -> u64 {
        if let Some(&off) = self.offsets.get(s) {
            return off;
        }
        let off = self.data.len() as u64;
        self.data.extend_from_slice(s.as_bytes());
        self.data.push(0);
        while self.data.len() % 8 != 0 {
            self.data.push(0);
        }
        self.offsets.insert(s.to_string(), off);
        off
    }
}

/// Copy the elements belonging to one chunk out of a row-major logical array.
fn gather_chunk(
    data: &[u8],
    dims: &[u64],
    chunk_dims: &[u64],
    base: &[u64],
    elem_size: usize,
    out: &mut [u8],
) {
    let rank = dims.len();
    if rank == 0 {
        let n = elem_size.min(data.len()).min(out.len());
        out[..n].copy_from_slice(&data[..n]);
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
            let src = (global_lin as usize) * elem_size;
            let dst = lin * elem_size;
            if src + elem_size <= data.len() {
                out[dst..dst + elem_size].copy_from_slice(&data[src..src + elem_size]);
            }
        }
    }
}

fn encode_dataspace(
    dims: &[u64],
    maxdims: &[Option<u64>],
    is_scalar: bool,
    is_null: bool,
) -> Vec<u8> {
    let mut b = Buf::new();
    if is_null {
        // Version 2 dataspace is required to express a NULL space.
        b.u8(2);
        b.u8(0); // rank
        b.u8(0); // flags
        b.u8(2); // type = null
        return b.bytes;
    }
    b.u8(1); // version
    if is_scalar {
        b.u8(0); // rank
        b.u8(0); // flags
        b.zeros(5);
        return b.bytes;
    }
    let rank = dims.len();
    b.u8(rank as u8);
    b.u8(1); // flags: max dimensions present
    b.zeros(5);
    for &d in dims {
        b.u64(d);
    }
    for (i, m) in maxdims.iter().enumerate() {
        match m {
            Some(v) => b.u64(*v),
            None => b.u64(UNDEF),
        }
        let _ = i;
    }
    // If maxdims is shorter than rank (shouldn't happen), pad with dims.
    for &d in dims.iter().skip(maxdims.len()) {
        b.u64(d);
    }
    b.bytes
}

fn encode_fill(fill: &FillValue) -> Vec<u8> {
    let mut b = Buf::new();
    b.u8(2); // version 2
    b.u8(2); // space allocation time = late
    b.u8(0); // fill value write time = at allocation
    match fill {
        FillValue::Undefined => {
            b.u8(0); // fill value undefined
        }
        FillValue::Default => {
            b.u8(1);
            b.u32(0); // size 0 => library default
        }
        FillValue::UserDefined(v) => {
            b.u8(1);
            b.u32(v.len() as u32);
            b.raw(v);
        }
    }
    b.bytes
}

fn encode_filter_pipeline(filters: &[filt::RawFilter]) -> Vec<u8> {
    let mut b = Buf::new();
    b.u8(1); // version 1
    b.u8(filters.len() as u8);
    b.u16(0); // reserved
    b.u32(0); // reserved
    for f in filters {
        b.u16(f.id);
        let name_bytes = f.name.as_bytes();
        let name_len = if name_bytes.is_empty() {
            0
        } else {
            align8(name_bytes.len() + 1)
        };
        b.u16(name_len as u16);
        b.u16(1); // flags: optional (matches libhdf5 defaults)
        b.u16(f.cdata.len() as u16);
        if name_len > 0 {
            let start = b.len();
            b.raw(name_bytes);
            b.u8(0);
            b.zeros(name_len - (b.len() - start));
        }
        for &v in &f.cdata {
            b.u32(v);
        }
        // client data padded to a multiple of 8 (v1)
        if f.cdata.len() % 2 != 0 {
            b.u32(0);
        }
    }
    b.bytes
}

/// Lower high-level filters to raw (id, client-data) form for the pipeline,
/// filling in filter-specific client data the way libhdf5's "set local"
/// callbacks do (element size for shuffle, datatype info for scaleoffset...).
fn lower_filters(filters: &[Filter], d: &DatasetData, chunk_elems: usize) -> Vec<filt::RawFilter> {
    use hdf5_types::TypeDescriptor as TD;
    let elem_size = disk_size(&d.dtype);
    filters
        .iter()
        .map(|f| {
            let mut raw = f.to_raw();
            if raw.id == filt::FILTER_SHUFFLE {
                raw.cdata = vec![elem_size as u32];
            }
            if raw.id == filt::FILTER_BLOSC && raw.cdata.len() >= 4 {
                raw.cdata[2] = elem_size as u32;
                raw.cdata[3] = (chunk_elems * elem_size) as u32;
            }
            if raw.id == filt::FILTER_NBIT {
                // full-precision datatypes: "no need to compress" flag set
                raw.cdata = vec![
                    8,
                    1,
                    chunk_elems as u32,
                    1,
                    elem_size as u32,
                    0,
                    (elem_size * 8) as u32,
                    0,
                ];
            }
            if raw.id == 4 {
                // szip set_local (H5Zszip.c): [mask, bpp, ppb, pps]
                let mask = raw.cdata.first().copied().unwrap_or(0x04);
                let ppb = raw.cdata.get(1).copied().unwrap_or(32).clamp(2, 32);
                let bpp = (elem_size * 8) as u32;
                let npoints = chunk_elems as u32;
                let scanline0 = match &d.layout {
                    LayoutClass::Chunked(c) => c.last().copied().unwrap_or(1) as u32,
                    _ => *d.dims.last().unwrap_or(&1) as u32,
                };
                let mut scanline = scanline0;
                if scanline < ppb {
                    if npoints < ppb {
                        scanline = npoints; // rejected below by codec if < ppb
                    } else {
                        scanline = (ppb * 128).min(npoints);
                    }
                } else if scanline <= 4096 {
                    scanline = scanline.min(ppb * 128);
                } else {
                    scanline = ppb * 128;
                }
                // little-endian data
                raw.cdata = vec![(mask & !0x10) | 0x08, bpp, ppb, scanline];
            }
            if raw.id == filt::FILTER_SCALEOFFSET {
                let (class, sign) = match &d.dtype {
                    TD::Integer(_) => (0u32, 1u32),
                    TD::Unsigned(_) => (0, 0),
                    TD::Float(_) => (1, 1),
                    _ => (u32::MAX, 0), // rejected by the codec
                };
                let (filavail, filval) = match &d.fill {
                    FillValue::UserDefined(v) => (1u32, v.clone()),
                    _ => (0, Vec::new()),
                };
                let mut cd = vec![
                    raw.cdata.first().copied().unwrap_or(2), // scale type
                    raw.cdata.get(1).copied().unwrap_or(0),  // scale factor
                    chunk_elems as u32,
                    class,
                    elem_size as u32,
                    sign,
                    0, // little-endian
                    filavail,
                ];
                for i in (0..filval.len()).step_by(4) {
                    let mut w = 0u32;
                    for j in 0..4.min(filval.len() - i) {
                        w |= u32::from(filval[i + j]) << (j * 8);
                    }
                    cd.push(w);
                }
                // H5Z_SCALEOFFSET_TOTAL_NPARMS: the decoder requires exactly
                // 20 client-data words
                cd.resize(20, 0);
                raw.cdata = cd;
            }
            raw
        })
        .collect()
}
