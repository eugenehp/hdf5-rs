//! Readers for the "version 2" metadata structures introduced in HDF5 1.8+:
//! v2 B-trees (`BTHD`/`BTIN`/`BTLF`), fractal heaps (`FRHP`/`FHIB`/`FHDB`) and
//! extensible arrays (`EAHD`/`EAIB`/`EASB`/`EADB`).
//!
//! Layouts are transcribed one-to-one from the libhdf5 sources
//! (`H5B2cache.c`, `H5HFcache.c`/`H5HFdtable.c`, `H5EAcache.c`/`H5EAhdr.c`)
//! and verified against files produced by libhdf5 2.0/h5py.

use super::{Cursor, UNDEF};
use crate::error::Result;

/// `H5VM_log2_gen`: floor(log2(n)) for n >= 1.
fn log2_gen(n: u64) -> u32 {
    63 - n.leading_zeros().min(63)
}

/// `H5VM_log2_of2`: log2 of a power of two.
fn log2_of2(n: u64) -> u32 {
    n.trailing_zeros()
}

/// `H5VM_limit_enc_size`: bytes needed to encode values in `0..=limit`.
fn limit_enc_size(limit: u64) -> usize {
    (log2_gen(limit.max(1)) as usize / 8) + 1
}

// ---------------------------------------------------------------------------
// v2 B-tree (H5B2)
// ---------------------------------------------------------------------------

/// A fully-walked v2 B-tree: raw records in tree order.
pub struct V2Btree {
    pub btree_type: u8,
    pub records: Vec<Vec<u8>>,
}

struct B2Level {
    /// max records per node at this depth
    max_nrec: u64,
    /// cumulative max records in a subtree rooted at this depth
    cum_max_nrec: u64,
    /// bytes used to encode a child's cumulative record count
    cum_max_nrec_size: usize,
}

/// Read and walk an entire v2 B-tree, returning its raw records in order.
pub fn read_v2btree(data: &[u8], base: u64, hdr_addr: u64) -> Result<V2Btree> {
    let mut c = Cursor::at(data, (base + hdr_addr) as usize);
    let sig = c.take(4)?;
    if sig != b"BTHD" {
        return Err("bad BTHD signature".into());
    }
    let version = c.u8()?;
    if version != 0 {
        return Err(format!("unsupported v2 btree header version {version}").into());
    }
    let btree_type = c.u8()?;
    let node_size = c.u32()? as usize;
    let rrec_size = c.u16()? as usize;
    let depth = c.u16()? as usize;
    let _split = c.u8()?;
    let _merge = c.u8()?;
    let root_addr = c.addr()?;
    let root_nrec = c.u16()? as u64;
    let _root_all_nrec = c.u64()?;
    let _checksum = c.u32()?;

    // Per-depth node capacity info, mirroring H5B2__hdr_init.
    // depth 0 = leaves.
    const B2_LEAF_PREFIX: usize = 4 + 1 + 1 + 4; // magic+ver+type+checksum
    const B2_INT_PREFIX: usize = 4 + 1 + 1 + 4;
    let mut levels: Vec<B2Level> = Vec::with_capacity((depth + 1).min(1 << 16));
    let leaf_max = ((node_size - B2_LEAF_PREFIX) / rrec_size) as u64;
    levels.push(B2Level {
        max_nrec: leaf_max,
        cum_max_nrec: leaf_max,
        cum_max_nrec_size: 0,
    });
    for u in 1..=depth {
        let prev = &levels[u - 1];
        let cum_size = limit_enc_size(prev.cum_max_nrec);
        // pointer to a child at depth u-1: addr + nrec + (all_nrec if u > 1)
        let ptr_size = 8 + limit_enc_size(prev.max_nrec) + if u > 1 { cum_size } else { 0 };
        let max_nrec = ((node_size - (B2_INT_PREFIX + ptr_size)) / (rrec_size + ptr_size)) as u64;
        let cum_max_nrec = ((max_nrec + 1) * prev.cum_max_nrec) + max_nrec;
        levels.push(B2Level {
            max_nrec,
            cum_max_nrec,
            cum_max_nrec_size: cum_size,
        });
    }

    let mut records = Vec::new();
    if root_addr != UNDEF && root_nrec > 0 {
        walk_b2_node(
            data,
            base,
            root_addr,
            root_nrec,
            depth,
            btree_type,
            rrec_size,
            &levels,
            &mut records,
        )?;
    }
    Ok(V2Btree {
        btree_type,
        records,
    })
}

#[allow(clippy::too_many_arguments)]
fn walk_b2_node(
    data: &[u8],
    base: u64,
    addr: u64,
    nrec: u64,
    depth: usize,
    btree_type: u8,
    rrec_size: usize,
    levels: &[B2Level],
    out: &mut Vec<Vec<u8>>,
) -> Result<()> {
    let mut c = Cursor::at(data, (base + addr) as usize);
    let sig = c.take(4)?;
    let expected: &[u8] = if depth == 0 { b"BTLF" } else { b"BTIN" };
    if sig != expected {
        return Err(format!(
            "bad v2 btree node signature (expected {})",
            String::from_utf8_lossy(expected)
        )
        .into());
    }
    let _ver = c.u8()?;
    let ntype = c.u8()?;
    if ntype != btree_type {
        return Err("v2 btree node type mismatch".into());
    }
    // records first (nrec * rrec_size), then child pointers for internal nodes
    let rec_start = c.pos;
    let recs: Vec<Vec<u8>> = (0..nrec)
        .map(|i| {
            let s = rec_start + (i as usize) * rrec_size;
            data[s..s + rrec_size].to_vec()
        })
        .collect();
    if depth == 0 {
        out.extend(recs);
        return Ok(());
    }
    c.seek(rec_start + (nrec as usize) * rrec_size);
    // nrec+1 child pointers to depth-1 nodes
    let child_level = &levels[depth - 1];
    let nrec_size = limit_enc_size(child_level.max_nrec);
    let cum_size = levels[depth].cum_max_nrec_size;
    let mut children = Vec::with_capacity((nrec as usize + 1).min(1 << 16));
    for _ in 0..=nrec {
        let caddr = c.addr()?;
        let cnrec = c.uint(nrec_size)?;
        if depth > 1 {
            let _all = c.uint(cum_size)?;
        }
        children.push((caddr, cnrec));
    }
    // in-order traversal: child0, rec0, child1, rec1, ..., recN-1, childN
    for (i, (caddr, cnrec)) in children.iter().enumerate() {
        walk_b2_node(
            data,
            base,
            *caddr,
            *cnrec,
            depth - 1,
            btree_type,
            rrec_size,
            levels,
            out,
        )?;
        if i < recs.len() {
            out.push(recs[i].clone());
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Fractal heap (H5HF)
// ---------------------------------------------------------------------------

/// A parsed fractal heap, sufficient to fetch managed and tiny objects.
pub struct FractalHeap<'a> {
    data: &'a [u8],
    base: u64,
    pub id_len: usize,
    #[allow(dead_code)] // parsed for completeness; direct-block checksums are skipped over
    checksum_dblocks: bool,
    // doubling table parameters
    table_width: usize,
    start_block_size: u64,
    max_direct_size: u64,
    #[allow(dead_code)] // used via heap_off_size, kept for debugging
    max_heap_bits: usize,
    root_addr: u64,
    cur_rows: usize,
    heap_off_size: usize,
    heap_len_size: usize,
    root_filtered_size: u64,
    root_filter_mask: u32,
    filters: Vec<crate::format::filters::RawFilter>,
    huge_bt2_addr: u64,
    huge_ids_direct: bool,
    huge_id_size: usize,
}

impl<'a> FractalHeap<'a> {
    pub fn parse(data: &'a [u8], base: u64, addr: u64) -> Result<Self> {
        let mut c = Cursor::at(data, (base + addr) as usize);
        let sig = c.take(4)?;
        if sig != b"FRHP" {
            return Err("bad FRHP signature".into());
        }
        let version = c.u8()?;
        if version != 0 {
            return Err(format!("unsupported fractal heap version {version}").into());
        }
        let id_len = c.u16()? as usize;
        let io_filter_len = c.u16()? as usize;
        let flags = c.u8()?;
        let max_man_size = c.u32()? as u64;
        let _next_huge_id = c.u64()?;
        let huge_bt2_addr = c.addr()?;
        let _free_space = c.u64()?;
        let _fs_addr = c.addr()?;
        let _man_size = c.u64()?;
        let _man_alloc = c.u64()?;
        let _man_iter_off = c.u64()?;
        let _man_nobjs = c.u64()?;
        let _huge_size = c.u64()?;
        let _huge_nobjs = c.u64()?;
        let _tiny_size = c.u64()?;
        let _tiny_nobjs = c.u64()?;
        // doubling table
        let table_width = c.u16()? as usize;
        let start_block_size = c.u64()?;
        let max_direct_size = c.u64()?;
        let max_heap_bits = c.u16()? as usize;
        let _start_rows = c.u16()?;
        let root_addr = c.addr()?;
        let cur_rows = c.u16()? as usize;
        // filtered heaps: root direct block stored size + mask + pipeline
        let (root_filtered_size, root_filter_mask, filters) = if io_filter_len > 0 {
            let sz = c.u64()?;
            let mask = c.u32()?;
            let mut pc = Cursor::at(c.data, c.pos);
            let filters = crate::format::reader::parse_filter_pipeline_public(&mut pc)?;
            (sz, mask, filters)
        } else {
            (0, 0, Vec::new())
        };
        let heap_off_size = max_heap_bits.div_ceil(8);
        // H5HF_hdr: heap_len_size = min(max_dir_blk_off_size, limit_enc_size(max_man_size))
        let max_dir_blk_off_size = limit_enc_size(max_direct_size);
        let heap_len_size = max_dir_blk_off_size.min(limit_enc_size(max_man_size));
        // H5HF__huge_init (unfiltered): huge ids are direct when the id can
        // hold addr + length inline; otherwise a v2 btree lookup is required.
        let huge_ids_direct = 8 + 8 <= id_len.saturating_sub(1);
        let huge_id_size = if huge_ids_direct {
            16
        } else {
            id_len.saturating_sub(1).min(8)
        };
        Ok(Self {
            data,
            base,
            id_len,
            checksum_dblocks: flags & 0x02 != 0,
            table_width,
            start_block_size,
            max_direct_size,
            max_heap_bits,
            root_addr,
            cur_rows,
            heap_off_size,
            heap_len_size,
            root_filtered_size,
            root_filter_mask,
            filters,
            huge_bt2_addr,
            huge_ids_direct,
            huge_id_size,
        })
    }

    /// Fetch and (when the heap is filtered) decompress one direct block.
    fn dblock_bytes(&self, addr: u64, stored: u64, mask: u32, plain_size: u64) -> Result<Vec<u8>> {
        let start = (self.base + addr) as usize;
        if self.filters.is_empty() {
            let end = start + plain_size as usize;
            if end > self.data.len() {
                return Err("fractal heap block out of bounds".into());
            }
            Ok(self.data[start..end].to_vec())
        } else {
            let end = start + stored as usize;
            if end > self.data.len() {
                return Err("fractal heap block out of bounds".into());
            }
            crate::format::filters::reverse_masked(&self.filters, 1, &self.data[start..end], mask)
        }
    }

    fn first_row_bits(&self) -> u32 {
        log2_of2(self.start_block_size) + log2_of2(self.table_width as u64)
    }

    fn max_direct_rows(&self) -> usize {
        (log2_of2(self.max_direct_size) - log2_of2(self.start_block_size)) as usize + 2
    }

    fn row_block_size(&self, row: usize) -> u64 {
        if row == 0 {
            self.start_block_size
        } else {
            self.start_block_size << (row - 1)
        }
    }

    /// `H5HF__dtable_lookup`: map a heap offset to (row, col).
    fn dtable_lookup(&self, off: u64) -> (usize, usize) {
        if off < self.start_block_size * self.table_width as u64 {
            (0, (off / self.start_block_size) as usize)
        } else {
            let high_bit = log2_gen(off);
            let off_mask = 1u64 << high_bit;
            let row = (high_bit - self.first_row_bits()) as usize + 1;
            let col = ((off - off_mask) / self.row_block_size(row)) as usize;
            (row, col)
        }
    }

    /// Find the direct block containing heap offset `off`, walking
    /// (possibly nested) indirect blocks. Returns (addr, block_off,
    /// stored_size, filter_mask); stored size/mask are meaningful only for
    /// filtered heaps.
    fn locate_dblock(&self, off: u64) -> Result<(u64, u64, u64, u32)> {
        if self.cur_rows == 0 {
            // root is a single direct block covering offset 0
            return Ok((
                self.root_addr,
                0,
                self.root_filtered_size,
                self.root_filter_mask,
            ));
        }
        self.locate_in_iblock(self.root_addr, self.cur_rows, 0, off)
    }

    fn locate_in_iblock(
        &self,
        iblock_addr: u64,
        nrows: usize,
        iblock_off: u64,
        off: u64,
    ) -> Result<(u64, u64, u64, u32)> {
        let mut c = Cursor::at(self.data, (self.base + iblock_addr) as usize);
        let sig = c.take(4)?;
        if sig != b"FHIB" {
            return Err("bad FHIB signature".into());
        }
        let _ver = c.u8()?;
        let _hdr_addr = c.addr()?;
        let block_off = c.uint(self.heap_off_size)?;
        debug_assert_eq!(block_off, iblock_off);
        let entries_start = c.pos;

        let (row, col) = self.dtable_lookup(off - block_off);
        if row >= nrows {
            return Err("fractal heap offset beyond indirect block rows".into());
        }
        let max_direct_rows = self.max_direct_rows();
        let entry = row * self.table_width + col;
        // filtered heaps store {addr, size, mask} per direct entry
        let dstride = if self.filters.is_empty() {
            8
        } else {
            8 + 8 + 4
        };
        if row < max_direct_rows {
            let mut ec = Cursor::at(self.data, entries_start + entry * dstride);
            let dblock_addr = ec.addr()?;
            let (fsize, fmask) = if self.filters.is_empty() {
                (0, 0)
            } else {
                (ec.u64()?, ec.u32()?)
            };
            if dblock_addr == UNDEF {
                return Err("fractal heap direct block not allocated".into());
            }
            // block offset of this direct block within the heap
            let dblock_off = block_off + self.row_col_offset(row, col);
            Ok((dblock_addr, dblock_off, fsize, fmask))
        } else {
            // indirect-block entry: skip the direct entries, then indirect addrs
            let ndirect = max_direct_rows.min(nrows) * self.table_width;
            let ientry = entry - ndirect;
            let mut ec = Cursor::at(self.data, entries_start + ndirect * dstride + ientry * 8);
            let child_addr = ec.addr()?;
            if child_addr == UNDEF {
                return Err("fractal heap indirect block not allocated".into());
            }
            // child indirect block at (row, col) covers this offset; its row
            // count follows the doubling-table geometry
            let child_off = block_off + self.row_col_offset(row, col);
            let child_nrows =
                (log2_of2(self.row_block_size(row)) - self.first_row_bits()) as usize + 1;
            self.locate_in_iblock(child_addr, child_nrows, child_off, off)
        }
    }

    /// Heap offset of the block at (row, col) relative to its indirect block.
    fn row_col_offset(&self, row: usize, col: usize) -> u64 {
        let mut off = 0u64;
        for r in 0..row {
            off += self.row_block_size(r) * self.table_width as u64;
        }
        off + self.row_block_size(row) * col as u64
    }

    /// Fetch an object by its heap ID.
    pub fn get(&self, id: &[u8]) -> Result<Vec<u8>> {
        if id.is_empty() {
            return Err("empty heap id".into());
        }
        let id_type = (id[0] >> 4) & 0x03;
        match id_type {
            0 => {
                // managed
                let mut c = Cursor::new(&id[1..]);
                let off = c.uint(self.heap_off_size)?;
                let len = c.uint(self.heap_len_size)? as usize;
                let (dblock_addr, dblock_off, fsize, fmask) = self.locate_dblock(off)?;
                // Managed heap offsets are absolute within the heap's address
                // space, which *includes* each block's prefix bytes
                // (H5HF__man_op_real: data = block image + (obj_off - block_off)).
                if self.filters.is_empty() {
                    let start = (self.base + dblock_addr + (off - dblock_off)) as usize;
                    if start + len > self.data.len() {
                        return Err("fractal heap object out of bounds".into());
                    }
                    Ok(self.data[start..start + len].to_vec())
                } else {
                    // decompress the whole block, then slice the object
                    let block =
                        self.dblock_bytes(dblock_addr, fsize, fmask, self.start_block_size)?;
                    let s = (off - dblock_off) as usize;
                    if s + len > block.len() {
                        return Err("fractal heap object out of bounds".into());
                    }
                    Ok(block[s..s + len].to_vec())
                }
            }
            2 => {
                // tiny (normal form: length in low nibble)
                let len = (id[0] & 0x0f) as usize + 1;
                if 1 + len > id.len() {
                    return Err("tiny heap object exceeds id length".into());
                }
                Ok(id[1..1 + len].to_vec())
            }
            1 => {
                // huge object (H5HFhuge.c): either the id embeds {addr, len}
                // directly, or it names a record in the huge v2 btree
                let mut c = Cursor::new(&id[1..]);
                let (addr, len) = if self.huge_ids_direct {
                    (c.addr()?, c.u64()? as usize)
                } else {
                    let want = c.uint(self.huge_id_size)?;
                    if self.huge_bt2_addr == UNDEF {
                        return Err("huge object btree missing".into());
                    }
                    let bt = read_v2btree(self.data, self.base, self.huge_bt2_addr)?;
                    if bt.btree_type != 1 {
                        return Err(format!(
                            "unsupported huge-object btree type {}",
                            bt.btree_type
                        )
                        .into());
                    }
                    // record: {addr(8), len(8), id(8)}
                    let mut found = None;
                    for rec in &bt.records {
                        let mut rc = Cursor::new(rec);
                        let addr = rc.addr()?;
                        let len = rc.u64()?;
                        let rec_id = rc.u64()?;
                        if rec_id == want {
                            found = Some((addr, len as usize));
                            break;
                        }
                    }
                    found.ok_or("huge object id not found in btree")?
                };
                let start = (self.base + addr) as usize;
                if start + len > self.data.len() {
                    return Err("huge object out of bounds".into());
                }
                Ok(self.data[start..start + len].to_vec())
            }
            _ => Err("unknown fractal heap id type".into()),
        }
    }
}

// ---------------------------------------------------------------------------
// Extensible array (H5EA)
// ---------------------------------------------------------------------------

/// A parsed extensible array header plus lookup machinery.
pub struct ExtensibleArray<'a> {
    data: &'a [u8],
    base: u64,
    pub elmt_size: usize,
    idx_blk_elmts: usize,
    data_blk_min_elmts: usize,
    sup_blk_min_data_ptrs: usize,
    #[allow(dead_code)] // used via arr_off_size/nsblks, kept for debugging
    max_nelmts_bits: usize,
    max_dblk_page_nelmts_bits: usize,
    index_blk_addr: u64,
    // derived
    arr_off_size: usize,
    nsblks: usize,
    sblk_info: Vec<SblkInfo>,
}

#[derive(Clone, Copy)]
struct SblkInfo {
    ndblks: usize,
    dblk_nelmts: usize,
    start_idx: u64,
    start_dblk: u64,
}

impl<'a> ExtensibleArray<'a> {
    pub fn parse(data: &'a [u8], base: u64, addr: u64) -> Result<Self> {
        let mut c = Cursor::at(data, (base + addr) as usize);
        let sig = c.take(4)?;
        if sig != b"EAHD" {
            return Err("bad EAHD signature".into());
        }
        let version = c.u8()?;
        if version != 0 {
            return Err(format!("unsupported extensible array version {version}").into());
        }
        let _client_id = c.u8()?;
        let elmt_size = c.u8()? as usize;
        let max_nelmts_bits = c.u8()? as usize;
        let idx_blk_elmts = c.u8()? as usize;
        let data_blk_min_elmts = c.u8()? as usize;
        let sup_blk_min_data_ptrs = c.u8()? as usize;
        let max_dblk_page_nelmts_bits = c.u8()? as usize;
        // statistics (6 lengths) + index block address
        let _nsblks_created = c.u64()?;
        let _sblk_size = c.u64()?;
        let _ndblks_created = c.u64()?;
        let _dblk_size = c.u64()?;
        let _max_idx_set = c.u64()?;
        let _nelmts_realized = c.u64()?;
        let index_blk_addr = c.addr()?;
        let _checksum = c.u32()?;

        // H5EA__hdr_init
        let nsblks = 1 + (max_nelmts_bits - log2_of2(data_blk_min_elmts as u64) as usize);
        let mut sblk_info = Vec::with_capacity((nsblks).min(1 << 16));
        let mut start_idx = 0u64;
        let mut start_dblk = 0u64;
        for u in 0..nsblks {
            let ndblks = 1usize << (u / 2);
            let dblk_nelmts = (1usize << u.div_ceil(2)) * data_blk_min_elmts;
            sblk_info.push(SblkInfo {
                ndblks,
                dblk_nelmts,
                start_idx,
                start_dblk,
            });
            start_idx += (ndblks * dblk_nelmts) as u64;
            start_dblk += ndblks as u64;
        }

        Ok(Self {
            data,
            base,
            elmt_size,
            idx_blk_elmts,
            data_blk_min_elmts,
            sup_blk_min_data_ptrs,
            max_nelmts_bits,
            max_dblk_page_nelmts_bits,
            index_blk_addr,
            arr_off_size: max_nelmts_bits.div_ceil(8),
            nsblks,
            sblk_info,
        })
    }

    fn dblk_page_nelmts(&self) -> usize {
        1usize << self.max_dblk_page_nelmts_bits
    }

    /// Number of super blocks whose data blocks are addressed directly from
    /// the index block (`H5EA_SBLK_FIRST_IDX`).
    fn iblock_nsblks(&self) -> usize {
        2 * log2_of2(self.sup_blk_min_data_ptrs as u64) as usize
    }

    fn iblock_ndblk_addrs(&self) -> usize {
        2 * (self.sup_blk_min_data_ptrs - 1)
    }

    /// Fetch one raw element by linear index; returns `None` for elements in
    /// unallocated blocks.
    pub fn element(&self, idx: u64) -> Result<Option<Vec<u8>>> {
        if self.index_blk_addr == UNDEF {
            return Ok(None);
        }
        let ib_pos = (self.base + self.index_blk_addr) as usize;
        let mut c = Cursor::at(self.data, ib_pos);
        let sig = c.take(4)?;
        if sig != b"EAIB" {
            return Err("bad EAIB signature".into());
        }
        let _ver = c.u8()?;
        let _cls = c.u8()?;
        let _hdr = c.addr()?;
        let elmts_start = c.pos;
        let dblk_addrs_start = elmts_start + self.idx_blk_elmts * self.elmt_size;
        let sblk_addrs_start = dblk_addrs_start + self.iblock_ndblk_addrs() * 8;

        if (idx as usize) < self.idx_blk_elmts {
            let s = elmts_start + (idx as usize) * self.elmt_size;
            return Ok(Some(self.data[s..s + self.elmt_size].to_vec()));
        }
        let elmt_idx = idx - self.idx_blk_elmts as u64;
        let sblk_idx = log2_gen(elmt_idx / self.data_blk_min_elmts as u64 + 1) as usize;
        if sblk_idx >= self.nsblks {
            return Err("extensible array index out of range".into());
        }
        let info = self.sblk_info[sblk_idx];
        let within = elmt_idx - info.start_idx;
        let dblk_nelmts = info.dblk_nelmts;

        if sblk_idx < self.iblock_nsblks() {
            // data block addressed directly from the index block
            let dblk_idx = info.start_dblk as usize + (within / dblk_nelmts as u64) as usize;
            let mut ac = Cursor::at(self.data, dblk_addrs_start + dblk_idx * 8);
            let dblk_addr = ac.addr()?;
            if dblk_addr == UNDEF {
                return Ok(None);
            }
            self.read_dblock_element(
                dblk_addr,
                dblk_nelmts,
                (within % dblk_nelmts as u64) as usize,
            )
        } else {
            // data block inside a super block
            let sblk_slot = sblk_idx - self.iblock_nsblks();
            let mut ac = Cursor::at(self.data, sblk_addrs_start + sblk_slot * 8);
            let sblk_addr = ac.addr()?;
            if sblk_addr == UNDEF {
                return Ok(None);
            }
            let mut sc = Cursor::at(self.data, (self.base + sblk_addr) as usize);
            let sig = sc.take(4)?;
            if sig != b"EASB" {
                return Err("bad EASB signature".into());
            }
            let _ver = sc.u8()?;
            let _cls = sc.u8()?;
            let _hdr = sc.addr()?;
            let _block_off = sc.uint(self.arr_off_size)?;
            // page-init bitmasks (present when data blocks are paged)
            let paged = dblk_nelmts > self.dblk_page_nelmts();
            let npages = if paged {
                dblk_nelmts / self.dblk_page_nelmts()
            } else {
                0
            };
            let page_init_size = if paged { npages.div_ceil(8) } else { 0 };
            let dblk_addrs_pos = sc.pos + info.ndblks * page_init_size;
            let dblk_idx = (within / dblk_nelmts as u64) as usize;
            let mut ac = Cursor::at(self.data, dblk_addrs_pos + dblk_idx * 8);
            let dblk_addr = ac.addr()?;
            if dblk_addr == UNDEF {
                return Ok(None);
            }
            self.read_dblock_element(
                dblk_addr,
                dblk_nelmts,
                (within % dblk_nelmts as u64) as usize,
            )
        }
    }

    fn read_dblock_element(
        &self,
        dblk_addr: u64,
        nelmts: usize,
        elmt_idx: usize,
    ) -> Result<Option<Vec<u8>>> {
        let mut c = Cursor::at(self.data, (self.base + dblk_addr) as usize);
        let sig = c.take(4)?;
        if sig != b"EADB" {
            return Err("bad EADB signature".into());
        }
        let _ver = c.u8()?;
        let _cls = c.u8()?;
        let _hdr = c.addr()?;
        let _block_off = c.uint(self.arr_off_size)?;
        let paged = nelmts > self.dblk_page_nelmts();
        if !paged {
            let s = c.pos + elmt_idx * self.elmt_size;
            if s + self.elmt_size > self.data.len() {
                return Err("extensible array element out of bounds".into());
            }
            Ok(Some(self.data[s..s + self.elmt_size].to_vec()))
        } else {
            // paged: prefix is followed by its checksum, then fixed-size pages
            // of (page_nelmts * elmt_size + checksum) each
            let page_nelmts = self.dblk_page_nelmts();
            let pages_start = c.pos + 4; // prefix checksum
            let page_size = page_nelmts * self.elmt_size + 4;
            let page_idx = elmt_idx / page_nelmts;
            let within = elmt_idx % page_nelmts;
            let s = pages_start + page_idx * page_size + within * self.elmt_size;
            if s + self.elmt_size > self.data.len() {
                return Err("extensible array element out of bounds".into());
            }
            Ok(Some(self.data[s..s + self.elmt_size].to_vec()))
        }
    }
}
