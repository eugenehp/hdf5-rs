//! Virtual dataset (VDS) support: decoding the global-heap mapping blob
//! (`H5Dvirtual.c`, heap encoding versions 0 and 1) and the serialized
//! dataspace selections it contains (`H5Shyper.c`/`H5Sall.c`/`H5Spoint.c`).

use super::{Buf, Cursor};
use crate::error::Result;

/// A deserialized dataspace selection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SerSelection {
    All,
    None,
    /// Regular hyperslab: per-dim (start, stride, count, block); `count`
    /// `None` = unlimited (H5S_UNLIMITED).
    Regular(Vec<(u64, u64, Option<u64>, u64)>),
    /// Irregular hyperslab: list of (start, end) corner pairs per block.
    Blocks(Vec<(Vec<u64>, Vec<u64>)>),
    /// Point list.
    Points(Vec<Vec<u64>>),
}

impl SerSelection {
    /// Enumerate selected linear (row-major) indices within `dims`.
    pub fn linear_indices(&self, dims: &[u64]) -> Result<Vec<u64>> {
        let rank = dims.len();
        let mut strides = vec![1u64; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        let total: u64 = dims.iter().product();
        match self {
            Self::All => Ok((0..total).collect()),
            Self::None => Ok(vec![]),
            Self::Regular(sel) => {
                if sel.len() != rank {
                    return Err("selection rank mismatch".into());
                }
                // per-dim coordinate lists
                let mut axes: Vec<Vec<u64>> = Vec::with_capacity((rank).min(1 << 16));
                for (d, &(start, stride, count, block)) in sel.iter().enumerate() {
                    let stride = stride.max(1);
                    let count = match count {
                        Some(c) => c,
                        None => {
                            // unlimited: as many full blocks as fit the extent
                            if dims[d] >= start + block {
                                (dims[d] - start - block) / stride + 1
                            } else {
                                0
                            }
                        }
                    };
                    let mut coords = Vec::new();
                    for i in 0..count {
                        let base = start + i * stride;
                        for j in 0..block {
                            if base + j < dims[d] {
                                coords.push(base + j);
                            }
                        }
                    }
                    axes.push(coords);
                }
                let n: usize = axes.iter().map(Vec::len).product();
                let mut out = Vec::with_capacity((n).min(1 << 16));
                let mut idx = vec![0usize; rank];
                if axes.iter().any(Vec::is_empty) {
                    return Ok(vec![]);
                }
                for _ in 0..n {
                    let mut lin = 0u64;
                    for (i, &j) in idx.iter().enumerate() {
                        lin += axes[i][j] * strides[i];
                    }
                    out.push(lin);
                    for k in (0..rank).rev() {
                        idx[k] += 1;
                        if idx[k] < axes[k].len() {
                            break;
                        }
                        idx[k] = 0;
                    }
                }
                Ok(out)
            }
            Self::Blocks(blocks) => {
                let mut out = Vec::new();
                for (start, end) in blocks {
                    if start.len() != rank || end.len() != rank {
                        return Err("selection rank mismatch".into());
                    }
                    // iterate the axis-aligned box [start, end] inclusive
                    let mut coord = start.clone();
                    loop {
                        let mut lin = 0u64;
                        for i in 0..rank {
                            lin += coord[i] * strides[i];
                        }
                        out.push(lin);
                        let mut k = rank;
                        loop {
                            if k == 0 {
                                return Ok(out);
                            }
                            k -= 1;
                            coord[k] += 1;
                            if coord[k] <= end[k] {
                                break;
                            }
                            coord[k] = start[k];
                            if k == 0 {
                                // finished this block
                                break;
                            }
                        }
                        if coord.iter().zip(start).all(|(c, s)| c == s) {
                            break;
                        }
                    }
                }
                Ok(out)
            }
            Self::Points(points) => {
                let mut out = Vec::with_capacity(points.len().min(1 << 16));
                for p in points {
                    let mut lin = 0u64;
                    for i in 0..rank {
                        lin += p[i] * strides[i];
                    }
                    out.push(lin);
                }
                Ok(out)
            }
        }
    }
}

/// Deserialize one dataspace selection (H5S_select_deserialize framing).
pub fn parse_selection(c: &mut Cursor) -> Result<SerSelection> {
    let sel_type = c.u32()?;
    match sel_type {
        // H5S_SEL_NONE / H5S_SEL_ALL: version(4) + padding(4) + length(4)
        0 | 3 => {
            let _version = c.u32()?;
            c.skip(8);
            Ok(if sel_type == 0 {
                SerSelection::None
            } else {
                SerSelection::All
            })
        }
        // H5S_SEL_POINTS
        1 => {
            let version = c.u32()?;
            let enc_size = if version >= 2 {
                let s = c.u8()?;
                s as usize
            } else {
                c.skip(8); // padding + length
                4
            };
            let rank = c.u32()? as usize;
            let npoints = c.uint(if version >= 2 { enc_size } else { 4 })? as usize;
            let mut points = Vec::with_capacity((npoints).min(1 << 16));
            for _ in 0..npoints {
                let mut p = Vec::with_capacity((rank).min(1 << 16));
                for _ in 0..rank {
                    p.push(c.uint(if version >= 2 { enc_size } else { 4 })?);
                }
                points.push(p);
            }
            Ok(SerSelection::Points(points))
        }
        // H5S_SEL_HYPERSLABS
        2 => {
            let version = c.u32()?;
            let (flags, enc_size) = match version {
                1 => {
                    c.skip(8); // padding + length
                    (0u8, 4usize)
                }
                2 => {
                    let f = c.u8()?;
                    c.skip(4); // padding
                    (f, 8usize)
                }
                3 => {
                    let f = c.u8()?;
                    let s = c.u8()? as usize;
                    (f, s)
                }
                v => return Err(format!("unsupported hyperslab selection version {v}").into()),
            };
            let rank = c.u32()? as usize;
            const REGULAR: u8 = 0x01;
            if version >= 2 && flags & REGULAR != 0 {
                let mut sel = Vec::with_capacity((rank).min(1 << 16));
                let unlimited = match enc_size {
                    2 => u16::MAX as u64,
                    4 => u32::MAX as u64,
                    _ => u64::MAX,
                };
                for _ in 0..rank {
                    let start = c.uint(enc_size)?;
                    let stride = c.uint(enc_size)?;
                    let count = c.uint(enc_size)?;
                    let block = c.uint(enc_size)?;
                    let count = if count == unlimited {
                        None
                    } else {
                        Some(count)
                    };
                    let block = if block == unlimited { u64::MAX } else { block };
                    sel.push((start, stride, count, block));
                }
                Ok(SerSelection::Regular(sel))
            } else {
                // irregular: number of blocks, then start/end corner pairs
                let nblocks = c.uint(enc_size.min(8))? as usize;
                let mut blocks = Vec::with_capacity((nblocks).min(1 << 16));
                for _ in 0..nblocks {
                    let mut start = Vec::with_capacity((rank).min(1 << 16));
                    let mut end = Vec::with_capacity((rank).min(1 << 16));
                    for _ in 0..rank {
                        start.push(c.uint(enc_size)?);
                    }
                    for _ in 0..rank {
                        end.push(c.uint(enc_size)?);
                    }
                    blocks.push((start, end));
                }
                Ok(SerSelection::Blocks(blocks))
            }
        }
        t => Err(format!("unsupported selection type {t}").into()),
    }
}

/// One virtual-to-source mapping.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VdsMapping {
    pub source_file: String,
    pub source_dset: String,
    pub src_sel: SerSelection,
    pub virt_sel: SerSelection,
}

/// Decode the VDS global-heap block (encoding versions 0 and 1).
pub fn parse_vds_blob(blob: &[u8]) -> Result<Vec<VdsMapping>> {
    let mut c = Cursor::new(blob);
    let version = c.u8()?;
    if version > 1 {
        return Err(format!("unsupported VDS heap encoding version {version}").into());
    }
    let nentries = c.u64()? as usize;
    let mut out: Vec<VdsMapping> = Vec::with_capacity((nentries).min(1 << 16));
    let mut first_same_file: Option<usize> = None;

    const FILE_SHARED: u8 = 0x01;
    const DSET_SHARED: u8 = 0x02;
    const SAME_FILE: u8 = 0x04;

    for i in 0..nentries {
        let flags = if version >= 1 { c.u8()? } else { 0 };

        let source_file = if flags & SAME_FILE != 0 {
            if first_same_file.is_none() {
                first_same_file = Some(i);
            }
            ".".to_string()
        } else if flags & FILE_SHARED != 0 {
            let origin = c.u64()? as usize;
            out.get(origin)
                .ok_or("bad shared source-file index")?
                .source_file
                .clone()
        } else {
            read_cstr(&mut c)?
        };

        let source_dset = if flags & DSET_SHARED != 0 {
            let origin = c.u64()? as usize;
            out.get(origin)
                .ok_or("bad shared source-dataset index")?
                .source_dset
                .clone()
        } else {
            read_cstr(&mut c)?
        };

        let src_sel = parse_selection(&mut c)?;
        let virt_sel = parse_selection(&mut c)?;
        out.push(VdsMapping {
            source_file,
            source_dset,
            src_sel,
            virt_sel,
        });
    }
    Ok(out)
}

fn read_cstr(c: &mut Cursor) -> Result<String> {
    let mut s = Vec::new();
    loop {
        let b = c.u8()?;
        if b == 0 {
            break;
        }
        s.push(b);
    }
    Ok(String::from_utf8_lossy(&s).into_owned())
}

/// Serialize one selection (H5S_select_serialize framing). `All` uses the
/// version-1 encoding; hyperslabs use the version-2 regular encoding.
pub fn encode_selection(sel: &SerSelection) -> Result<Vec<u8>> {
    let mut b = Buf::new();
    match sel {
        SerSelection::All | SerSelection::None => {
            b.u32(if matches!(sel, SerSelection::All) {
                3
            } else {
                0
            });
            b.u32(1); // version
            b.u32(0); // padding
            b.u32(0); // extra info length
        }
        SerSelection::Regular(dims) => {
            b.u32(2); // H5S_SEL_HYPERSLABS
            b.u32(2); // version 2
            b.u8(0x01); // flags: regular
            b.u32(0); // padding
            b.u32(dims.len() as u32);
            for &(start, stride, count, block) in dims {
                b.u64(start);
                b.u64(stride.max(1));
                b.u64(count.unwrap_or(u64::MAX));
                b.u64(block);
            }
        }
        SerSelection::Blocks(_) | SerSelection::Points(_) => {
            return Err("only ALL and regular hyperslab selections can be written".into())
        }
    }
    Ok(b.bytes)
}

/// Serialize a VDS mapping list into a global-heap block (encoding v0).
pub fn encode_vds_blob(mappings: &[VdsMapping]) -> Result<Vec<u8>> {
    let mut b = Buf::new();
    b.u8(0); // heap encoding version 0: no per-entry flags
    b.u64(mappings.len() as u64);
    for m in mappings {
        b.raw(m.source_file.as_bytes());
        b.u8(0);
        b.raw(m.source_dset.as_bytes());
        b.u8(0);
        b.raw(&encode_selection(&m.src_sel)?);
        b.raw(&encode_selection(&m.virt_sel)?);
    }
    let sum = super::checksum::checksum(&b.bytes);
    b.u32(sum);
    Ok(b.bytes)
}
