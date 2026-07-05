//! The HDF5 filter pipeline: DEFLATE (zlib), byte-shuffle and fletcher32.
//!
//! Filters operate on raw chunk (or contiguous) bytes. On write they are applied
//! in listed order; on read they are reversed in the opposite order. Each filter
//! is identified by its HDF5 filter id and a small array of client-data words.

use std::io::{Read, Write};

use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;

use crate::error::Result;

pub const FILTER_DEFLATE: u16 = 1;
pub const FILTER_SHUFFLE: u16 = 2;
pub const FILTER_FLETCHER32: u16 = 3;
pub const FILTER_NBIT: u16 = 5;
pub const FILTER_SCALEOFFSET: u16 = 6;
/// The registered id of the LZF filter (h5py's default fast compressor).
pub const FILTER_SZIP: u16 = 4;
pub const FILTER_LZF: u16 = 32000;
/// The registered id of the blosc filter.
pub const FILTER_BLOSC: u16 = 32001;

/// A single filter lowered to its id and client-data words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawFilter {
    pub id: u16,
    pub cdata: Vec<u32>,
    pub name: String,
}

/// Apply the filter pipeline (forward) to `data`, given the dataset element
/// size. Returns the filtered bytes.
pub fn apply(filters: &[RawFilter], element_size: usize, data: &[u8]) -> Result<Vec<u8>> {
    let mut buf = data.to_vec();
    for f in filters {
        buf = apply_one(f, element_size, &buf)?;
    }
    Ok(buf)
}

/// Reverse the filter pipeline on `data` (filters applied in reverse order).
pub fn reverse(filters: &[RawFilter], element_size: usize, data: &[u8]) -> Result<Vec<u8>> {
    reverse_masked(filters, element_size, data, 0)
}

/// Reverse the pipeline, skipping filters whose bit is set in `filter_mask`
/// (chunks store a mask of *skipped* optional filters, indexed by pipeline
/// position).
pub fn reverse_masked(
    filters: &[RawFilter],
    element_size: usize,
    data: &[u8],
    filter_mask: u32,
) -> Result<Vec<u8>> {
    let mut buf = data.to_vec();
    for (i, f) in filters.iter().enumerate().rev() {
        if filter_mask & (1 << i) != 0 {
            continue; // filter was skipped when the chunk was written
        }
        buf = reverse_one(f, element_size, &buf)?;
    }
    Ok(buf)
}

fn apply_one(f: &RawFilter, element_size: usize, data: &[u8]) -> Result<Vec<u8>> {
    match f.id {
        FILTER_DEFLATE => {
            let level = f.cdata.first().copied().unwrap_or(6).min(9);
            let mut enc = ZlibEncoder::new(Vec::new(), Compression::new(level));
            enc.write_all(data)?;
            Ok(enc.finish()?)
        }
        FILTER_SHUFFLE => {
            // The pipeline message stores the element size as client data.
            let esize = f.cdata.first().map(|&v| v as usize).unwrap_or(element_size);
            Ok(shuffle(esize.max(1), data))
        }
        FILTER_FLETCHER32 => {
            let mut out = data.to_vec();
            let sum = fletcher32(data);
            out.extend_from_slice(&sum.to_le_bytes());
            Ok(out)
        }
        FILTER_LZF => Ok(lzf_compress(data)),
        FILTER_BLOSC => {
            // cd_values (blosc_filter.c): [0]=filter rev, [1]=blosc format,
            // [2]=typesize, [3]=chunksize, [4]=clevel, [5]=shuffle, [6]=codec
            let typesize = f.cdata.get(2).copied().unwrap_or(element_size as u32) as usize;
            let clevel = f.cdata.get(4).copied().unwrap_or(5) as u8;
            let shuffle = f.cdata.get(5).copied().unwrap_or(1) as u8;
            // cd[6] is the compressor CODE (BLOSC_ZSTD=5 etc.); the frame
            // stores the FORMAT id (zstd=4, zlib=3, lz4hc shares lz4's 1)
            let format = match f.cdata.get(6).copied().unwrap_or(0) {
                1 | 2 => super::blosc::CODEC_LZ4,
                3 => super::blosc::CODEC_SNAPPY,
                4 => super::blosc::CODEC_ZLIB,
                5 => super::blosc::CODEC_ZSTD,
                _ => super::blosc::CODEC_BLOSCLZ,
            };
            super::blosc::compress(format, clevel, shuffle, typesize.max(1), data)
        }
        FILTER_SCALEOFFSET => scaleoffset_compress(&f.cdata, data),
        #[cfg(feature = "szip")]
        FILTER_SZIP => {
            // H5Zszip framing: u32 LE original size + SZ stream
            let mut out = (data.len() as u32).to_le_bytes().to_vec();
            out.extend_from_slice(&super::szip::sz_compress(&f.cdata, data)?);
            Ok(out)
        }
        FILTER_NBIT => {
            // full-precision datatypes: the "no need to compress" path
            // (cd_values[1] = 1) stores the chunk verbatim, exactly what
            // libhdf5's set_local computes for our type system
            Ok(data.to_vec())
        }
        other => Err(format!("filter id {other} is not supported for writing").into()),
    }
}

fn reverse_one(f: &RawFilter, element_size: usize, data: &[u8]) -> Result<Vec<u8>> {
    match f.id {
        FILTER_DEFLATE => {
            let mut dec = ZlibDecoder::new(data);
            let mut out = Vec::new();
            dec.read_to_end(&mut out)?;
            Ok(out)
        }
        FILTER_SHUFFLE => {
            let esize = f.cdata.first().map(|&v| v as usize).unwrap_or(element_size);
            Ok(unshuffle(esize.max(1), data))
        }
        FILTER_FLETCHER32 => {
            if data.len() < 4 {
                return Err("fletcher32: data too short".into());
            }
            let (body, sum_bytes) = data.split_at(data.len() - 4);
            let stored =
                u32::from_le_bytes([sum_bytes[0], sum_bytes[1], sum_bytes[2], sum_bytes[3]]);
            let computed = fletcher32(body);
            if stored != computed {
                return Err("fletcher32 checksum mismatch".into());
            }
            Ok(body.to_vec())
        }
        FILTER_LZF => lzf_decompress(data),
        FILTER_SCALEOFFSET => scaleoffset_decompress(&f.cdata, data),
        FILTER_NBIT => nbit_decompress(&f.cdata, data),
        FILTER_BLOSC => super::blosc::decompress(data),
        #[cfg(feature = "szip")]
        FILTER_SZIP => {
            if data.len() < 4 {
                return Err("szip: chunk too short".into());
            }
            let n = u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;
            super::szip::sz_decompress(&f.cdata, &data[4..], n)
        }
        other => Err(format!("filter id {other} is not supported for reading").into()),
    }
}

/// Byte-shuffle: gather the i-th byte of every element together.
fn shuffle(element_size: usize, data: &[u8]) -> Vec<u8> {
    if element_size <= 1 {
        return data.to_vec();
    }
    let n = data.len() / element_size;
    let mut out = vec![0u8; data.len()];
    let leftover = data.len() - n * element_size;
    for j in 0..element_size {
        for i in 0..n {
            out[j * n + i] = data[i * element_size + j];
        }
    }
    // trailing bytes that don't form a whole element are copied verbatim
    if leftover > 0 {
        let tail = n * element_size;
        out[tail..].copy_from_slice(&data[tail..]);
    }
    out
}

/// Inverse of [`shuffle`].
fn unshuffle(element_size: usize, data: &[u8]) -> Vec<u8> {
    if element_size <= 1 {
        return data.to_vec();
    }
    let n = data.len() / element_size;
    let mut out = vec![0u8; data.len()];
    let leftover = data.len() - n * element_size;
    for j in 0..element_size {
        for i in 0..n {
            out[i * element_size + j] = data[j * n + i];
        }
    }
    if leftover > 0 {
        let tail = n * element_size;
        out[tail..].copy_from_slice(&data[tail..]);
    }
    out
}

/// HDF5's fletcher32 checksum (`H5_checksum_fletcher32`): operates on 16-bit
/// big-endian word pairs, with an odd trailing byte contributing `byte << 8`.
pub fn fletcher32(data: &[u8]) -> u32 {
    let mut sum1: u32 = 0;
    let mut sum2: u32 = 0;
    let mut len = data.len() / 2; // number of 16-bit words
    let mut pos = 0;
    while len > 0 {
        let mut tlen = len.min(360);
        len -= tlen;
        while tlen > 0 {
            let word = ((data[pos] as u32) << 8) | data[pos + 1] as u32;
            sum1 = sum1.wrapping_add(word);
            sum2 = sum2.wrapping_add(sum1);
            pos += 2;
            tlen -= 1;
        }
        sum1 = (sum1 & 0xffff).wrapping_add(sum1 >> 16);
        sum2 = (sum2 & 0xffff).wrapping_add(sum2 >> 16);
    }
    // Handle an odd trailing byte.
    if data.len() % 2 != 0 {
        let word = (data[data.len() - 1] as u32) << 8;
        sum1 = sum1.wrapping_add(word);
        sum2 = sum2.wrapping_add(sum1);
        sum1 = (sum1 & 0xffff).wrapping_add(sum1 >> 16);
        sum2 = (sum2 & 0xffff).wrapping_add(sum2 >> 16);
    }
    sum1 = (sum1 & 0xffff).wrapping_add(sum1 >> 16);
    sum2 = (sum2 & 0xffff).wrapping_add(sum2 >> 16);
    (sum2 << 16) | sum1
}

// ---------------------------------------------------------------------------
// LZF (Marc Lehmann's liblzf format, as used by h5py's built-in LZF filter)
// ---------------------------------------------------------------------------

const LZF_MAX_OFF: usize = 1 << 13; // 13-bit back-reference offsets
const LZF_MAX_REF: usize = (1 << 8) + (1 << 3); // 264: max match length
const LZF_MAX_LIT: usize = 1 << 5; // 32: max literal run
const LZF_HLOG: usize = 14;

#[inline]
fn lzf_hash(a: u8, b: u8, c: u8) -> usize {
    let v = (u32::from(a) << 16) | (u32::from(b) << 8) | u32::from(c);
    (v.wrapping_mul(2_654_435_761) >> (32 - LZF_HLOG)) as usize
}

/// Compress `input` into a valid LZF stream. Incompressible data degrades to
/// pure literal runs (still valid LZF), so this never fails -- matching the
/// "optional filter" semantics where the stored chunk must always decode.
// Loop shapes intentionally mirror the C reference implementation.
#[allow(clippy::needless_range_loop, clippy::explicit_counter_loop)]
pub fn lzf_compress(input: &[u8]) -> Vec<u8> {
    let n = input.len();
    let mut out = Vec::with_capacity((n + n / 16 + 8).min(1 << 16));
    let mut htab = vec![usize::MAX; 1 << LZF_HLOG];
    let mut lit_start = 0usize;
    let mut pos = 0usize;

    let emit_literals = |out: &mut Vec<u8>, from: usize, to: usize| {
        let mut s = from;
        while s < to {
            let run = (to - s).min(LZF_MAX_LIT);
            out.push((run - 1) as u8);
            out.extend_from_slice(&input[s..s + run]);
            s += run;
        }
    };

    while pos + 2 < n {
        let h = lzf_hash(input[pos], input[pos + 1], input[pos + 2]);
        let cand = htab[h];
        htab[h] = pos;
        if cand != usize::MAX
            && pos - cand <= LZF_MAX_OFF
            && cand + 2 < n
            && input[cand] == input[pos]
            && input[cand + 1] == input[pos + 1]
            && input[cand + 2] == input[pos + 2]
        {
            // extend the match
            let max_len = (n - pos).min(LZF_MAX_REF);
            let mut mlen = 3;
            while mlen < max_len && input[cand + mlen] == input[pos + mlen] {
                mlen += 1;
            }
            emit_literals(&mut out, lit_start, pos);
            let off = pos - cand - 1;
            let l = mlen - 2;
            if l < 7 {
                out.push(((off >> 8) as u8) | ((l as u8) << 5));
            } else {
                out.push(((off >> 8) as u8) | (7 << 5));
                out.push((l - 7) as u8);
            }
            out.push((off & 0xff) as u8);
            pos += mlen;
            lit_start = pos;
        } else {
            pos += 1;
        }
    }
    emit_literals(&mut out, lit_start, n);
    out
}

/// Decompress an LZF stream.
// Loop shapes intentionally mirror the C reference implementation.
#[allow(clippy::needless_range_loop, clippy::explicit_counter_loop)]
pub fn lzf_decompress(input: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity((input.len().min(1 << 16)) * 3);
    let mut i = 0usize;
    while i < input.len() {
        let ctrl = input[i] as usize;
        i += 1;
        if ctrl < 32 {
            // literal run of ctrl + 1 bytes
            let len = ctrl + 1;
            if i + len > input.len() {
                return Err("lzf: truncated literal run".into());
            }
            out.extend_from_slice(&input[i..i + len]);
            i += len;
        } else {
            // back reference
            let mut len = ctrl >> 5;
            if len == 7 {
                if i >= input.len() {
                    return Err("lzf: truncated length byte".into());
                }
                len += input[i] as usize;
                i += 1;
            }
            len += 2;
            if i >= input.len() {
                return Err("lzf: truncated offset byte".into());
            }
            let off = ((ctrl & 0x1f) << 8) | input[i] as usize;
            i += 1;
            let mut rpos = out
                .len()
                .checked_sub(off + 1)
                .ok_or("lzf: back reference before start of output")?;
            for _ in 0..len {
                let b = out[rpos];
                out.push(b);
                rpos += 1;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod lzf_tests {
    use super::{lzf_compress, lzf_decompress};

    #[test]
    fn lzf_roundtrip() {
        for data in [
            b"".to_vec(),
            b"a".to_vec(),
            b"hello hello hello hello hello".to_vec(),
            (0..10_000u32)
                .flat_map(|i| (i % 251).to_le_bytes())
                .collect::<Vec<u8>>(),
            vec![0u8; 65536],
            (0..=255u8).cycle().take(100_000).collect::<Vec<u8>>(),
        ] {
            let c = lzf_compress(&data);
            assert_eq!(lzf_decompress(&c).unwrap(), data);
        }
    }
}

// ---------------------------------------------------------------------------
// ScaleOffset filter (H5Zscaleoffset.c) -- decode only
// ---------------------------------------------------------------------------

/// Decode a scaleoffset-compressed chunk. `cdata` layout (set_local):
/// `[0]`=scale type, `[1]`=scale factor, `[2]`=nelmts, `[3]`=class (0 int,
/// 1 float), `[4]`=size, `[5]`=sign, `[6]`=order, `[7]`=fill defined,
/// `[8..]`=fill value bytes.
pub fn scaleoffset_decompress(cdata: &[u32], data: &[u8]) -> Result<Vec<u8>> {
    if cdata.len() < 8 {
        return Err("scaleoffset: missing parameters".into());
    }
    let scale_type = cdata[0];
    let scale_factor = cdata[1];
    let nelmts = cdata[2] as usize;
    let class = cdata[3];
    let size = cdata[4] as usize;
    let order = cdata[6];
    let filavail = cdata[7] == 1;
    if order != 0 {
        return Err("scaleoffset: big-endian datasets are not supported".into());
    }
    if !(1..=8).contains(&size) {
        return Err("scaleoffset: bad element size".into());
    }
    if data.len() < 21 {
        return Err("scaleoffset: buffer too short".into());
    }
    // header: minbits (u32 LE), minval size (u8), minval (LE); data at 21
    let minbits = u32::from_le_bytes(data[0..4].try_into().unwrap());
    let minval_size = (data[4] as usize).min(8);
    let mut mv = [0u8; 8];
    mv[..minval_size].copy_from_slice(&data[5..5 + minval_size]);
    let minval = u64::from_le_bytes(mv);
    if minbits as usize > size * 8 {
        return Err("scaleoffset: minbits exceeds type size".into());
    }

    let dtype_len = size * 8;
    let mut out = vec![0u8; nelmts * size];
    if minbits != 0 {
        // bit-unpack (little-endian branch of H5Z__scaleoffset_decompress)
        let packed = &data[21..];
        let mut j = 0usize;
        let mut bits_to_fill = 8usize;
        let begin_i = size - 1 - (dtype_len - minbits as usize) / 8;
        for e in 0..nelmts {
            let base = e * size;
            for k in (0..=begin_i).rev() {
                let bits_to_copy = if k == begin_i {
                    8 - (dtype_len - minbits as usize) % 8
                } else {
                    8
                };
                unpack_bits(
                    &mut out[base + k],
                    packed,
                    &mut j,
                    &mut bits_to_fill,
                    bits_to_copy,
                    0,
                )?;
            }
        }
    }

    // post-decompression
    let mask: u64 = if minbits as usize >= 64 {
        u64::MAX
    } else {
        (1u64 << minbits) - 1
    };
    let fill_bytes = |cd: &[u32], size: usize| -> Vec<u8> {
        let mut v = Vec::with_capacity((size).min(1 << 16));
        for i in 0..size {
            let word = cd.get(8 + i / 4).copied().unwrap_or(0);
            v.push((word >> ((i % 4) * 8)) as u8);
        }
        v
    };
    if class == 0 {
        // integer: value = (raw == 2^minbits - 1 && fill defined) ? fill : raw + minval
        let filval = fill_bytes(cdata, size);
        for e in 0..nelmts {
            let cell = &mut out[e * size..(e + 1) * size];
            let mut b = [0u8; 8];
            b[..size].copy_from_slice(cell);
            let raw = u64::from_le_bytes(b);
            let v = if filavail && raw == mask {
                let mut f = [0u8; 8];
                f[..size].copy_from_slice(&filval);
                u64::from_le_bytes(f)
            } else {
                raw.wrapping_add(minval)
            };
            cell.copy_from_slice(&v.to_le_bytes()[..size]);
        }
    } else if class == 1 && scale_type == 0 {
        // float variable-minbits (D-scale): value = raw_signed / 10^D + min
        let d_val = f64::from(scale_factor);
        let filval = fill_bytes(cdata, size);
        for e in 0..nelmts {
            let cell = &mut out[e * size..(e + 1) * size];
            match size {
                4 => {
                    let raw = i32::from_le_bytes(cell.try_into().unwrap());
                    let v = if filavail && raw as u32 as u64 == mask {
                        f32::from_le_bytes(filval[..4].try_into().unwrap())
                    } else {
                        let min = f32::from_le_bytes(minval.to_le_bytes()[..4].try_into().unwrap());
                        (raw as f32) / 10f32.powf(d_val as f32) + min
                    };
                    cell.copy_from_slice(&v.to_le_bytes());
                }
                8 => {
                    let raw = i64::from_le_bytes(cell.try_into().unwrap());
                    let v = if filavail && raw as u64 == mask {
                        f64::from_le_bytes(filval[..8].try_into().unwrap())
                    } else {
                        let min = f64::from_le_bytes(minval.to_le_bytes());
                        (raw as f64) / 10f64.powf(d_val) + min
                    };
                    cell.copy_from_slice(&v.to_le_bytes());
                }
                _ => return Err("scaleoffset: unsupported float size".into()),
            }
        }
    } else {
        return Err("scaleoffset: unsupported scale type".into());
    }
    Ok(out)
}

/// Copy `bits_to_copy` bits from the MSB-first packed stream into one output
/// byte (shared by the scaleoffset and nbit unpackers; `dat_offset` shifts the
/// bits up within the byte, as nbit's offset handling requires).
// Loop shapes intentionally mirror the C reference implementation.
#[allow(clippy::needless_range_loop, clippy::explicit_counter_loop)]
fn unpack_bits(
    out_byte: &mut u8,
    packed: &[u8],
    j: &mut usize,
    bits_to_fill: &mut usize,
    mut bits_to_copy: usize,
    dat_offset: usize,
) -> Result<()> {
    if *j >= packed.len() {
        return Err("bit-packed stream too short".into());
    }
    let mut val = packed[*j];
    if *bits_to_fill > bits_to_copy {
        *out_byte = (((val as usize >> (*bits_to_fill - bits_to_copy))
            & !(usize::MAX << bits_to_copy))
            << dat_offset) as u8;
        *bits_to_fill -= bits_to_copy;
    } else {
        *out_byte = ((((val as usize) & !(usize::MAX << *bits_to_fill))
            << (bits_to_copy - *bits_to_fill))
            << dat_offset) as u8;
        bits_to_copy -= *bits_to_fill;
        *j += 1;
        *bits_to_fill = 8;
        if bits_to_copy == 0 {
            return Ok(());
        }
        if *j >= packed.len() {
            return Err("bit-packed stream too short".into());
        }
        val = packed[*j];
        *out_byte |= (((val as usize >> (*bits_to_fill - bits_to_copy))
            & !(usize::MAX << bits_to_copy))
            << dat_offset) as u8;
        *bits_to_fill -= bits_to_copy;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// N-bit filter (H5Znbit.c) -- decode only, atomic datatypes
// ---------------------------------------------------------------------------

/// Decode an nbit-compressed chunk. `cdata`: `[0]`=nparms, `[1]`=no-op flag,
/// `[2]`=nelmts, `[3]`=class, `[4]`=size, `[5]`=order, `[6]`=precision,
/// `[7]`=offset.
// Loop shapes intentionally mirror the C reference implementation.
#[allow(clippy::needless_range_loop, clippy::explicit_counter_loop)]
pub fn nbit_decompress(cdata: &[u32], data: &[u8]) -> Result<Vec<u8>> {
    if cdata.len() < 3 {
        return Err("nbit: missing parameters".into());
    }
    if cdata[1] != 0 {
        // "no need to compress" flag: data stored verbatim
        return Ok(data.to_vec());
    }
    if cdata.len() < 8 {
        return Err("nbit: missing parameters".into());
    }
    let nelmts = cdata[2] as usize;
    let class = cdata[3];
    if class != 1 {
        return Err("nbit: only atomic datatypes are supported".into());
    }
    let size = cdata[4] as usize;
    let order = cdata[5];
    let precision = cdata[6] as usize;
    let offset = cdata[7] as usize;
    if order != 0 {
        return Err("nbit: big-endian datasets are not supported".into());
    }
    if precision > size * 8 || precision + offset > size * 8 || precision == 0 {
        return Err("nbit: bad precision/offset".into());
    }
    let datatype_len = size * 8;
    let mut out = vec![0u8; nelmts * size];
    let mut j = 0usize;
    let mut buf_len = 8usize;
    // little-endian branch of H5Z__nbit_decompress_one_atomic
    let begin_i = if (precision + offset) % 8 != 0 {
        (precision + offset) / 8
    } else {
        (precision + offset) / 8 - 1
    };
    let end_i = offset / 8;
    for e in 0..nelmts {
        let base = e * size;
        for k in (end_i..=begin_i).rev() {
            let (dat_len, dat_offset) = if begin_i != end_i {
                if k == begin_i {
                    (8 - (datatype_len - precision - offset) % 8, 0)
                } else if k == end_i {
                    let l = 8 - offset % 8;
                    (l, 8 - l)
                } else {
                    (8, 0)
                }
            } else {
                (precision, offset % 8)
            };
            unpack_bits(
                &mut out[base + k],
                data,
                &mut j,
                &mut buf_len,
                dat_len,
                dat_offset,
            )?;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// ScaleOffset compression (H5Zscaleoffset.c encode path)
// ---------------------------------------------------------------------------

/// Number of bits needed to hold `num` distinct values (H5Z__scaleoffset_log2).
fn so_log2(num: u64) -> u32 {
    if num <= 1 {
        0
    } else {
        64 - (num - 1).leading_zeros()
    }
}

/// Compress a chunk with the scaleoffset filter (integer and float D-scale).
// Loop shapes intentionally mirror the C reference implementation.
#[allow(clippy::needless_range_loop, clippy::explicit_counter_loop)]
pub fn scaleoffset_compress(cdata: &[u32], data: &[u8]) -> Result<Vec<u8>> {
    if cdata.len() < 8 {
        return Err("scaleoffset: missing parameters".into());
    }
    let scale_type = cdata[0];
    let d_val = f64::from(cdata[1]);
    let nelmts = cdata[2] as usize;
    let class = cdata[3];
    let size = cdata[4] as usize;
    let signed = cdata[5] == 1;
    let filavail = cdata[7] == 1;
    if data.len() < nelmts * size {
        return Err("scaleoffset: short input".into());
    }
    let dtype_len = (size * 8) as u32;

    let read_cell = |i: usize| -> u64 {
        let mut b = [0u8; 8];
        b[..size].copy_from_slice(&data[i * size..(i + 1) * size]);
        u64::from_le_bytes(b)
    };
    let sign_extend = |v: u64| -> i64 {
        if size == 8 {
            v as i64
        } else {
            let shift = 64 - size * 8;
            ((v << shift) as i64) >> shift
        }
    };
    let filval: u64 = {
        let mut b = [0u8; 8];
        for i in 0..size {
            let w = cdata.get(8 + i / 4).copied().unwrap_or(0);
            b[i] = (w >> ((i % 4) * 8)) as u8;
        }
        u64::from_le_bytes(b)
    };

    // transform values to non-negative offsets + compute minbits/minval
    let mut vals = vec![0u64; nelmts];
    let (minbits, minval): (u32, u64);
    if class == 0 {
        // integer
        let mut min = i128::MAX;
        let mut max = i128::MIN;
        let as_key = |raw: u64| -> i128 {
            if signed {
                i128::from(sign_extend(raw))
            } else {
                i128::from(raw)
            }
        };
        let filkey = as_key(filval);
        let mut any = false;
        for i in 0..nelmts {
            let k = as_key(read_cell(i));
            if filavail && k == filkey {
                continue;
            }
            any = true;
            min = min.min(k);
            max = max.max(k);
        }
        if !any {
            min = 0;
            max = 0;
        }
        let span = (max - min + 1) as u64;
        let mb = if filavail {
            so_log2(span + 1)
        } else {
            so_log2(span)
        };
        minbits = mb.min(dtype_len);
        minval = min as u64; // truncated to type width on decode
        for i in 0..nelmts {
            let raw = read_cell(i);
            vals[i] = if minbits == dtype_len {
                raw
            } else if filavail && as_key(raw) == filkey {
                (1u64 << minbits) - 1
            } else {
                (as_key(raw) - min) as u64
            };
        }
    } else if class == 1 && scale_type == 0 {
        // float D-scale
        let getf = |i: usize| -> f64 {
            match size {
                4 => f64::from(f32::from_le_bytes(
                    data[i * 4..i * 4 + 4].try_into().unwrap(),
                )),
                _ => f64::from_le_bytes(data[i * 8..i * 8 + 8].try_into().unwrap()),
            }
        };
        let filf = match size {
            4 => f64::from(f32::from_le_bytes(
                filval.to_le_bytes()[..4].try_into().unwrap(),
            )),
            _ => f64::from_le_bytes(filval.to_le_bytes()),
        };
        let p = 10f64.powf(d_val);
        let tol = 10f64.powf(-d_val);
        let is_fill = |v: f64| filavail && (v - filf).abs() < tol;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut any = false;
        for i in 0..nelmts {
            let v = getf(i);
            if is_fill(v) {
                continue;
            }
            any = true;
            min = min.min(v);
            max = max.max(v);
        }
        if !any {
            min = 0.0;
            max = 0.0;
        }
        let span = ((max * p).round() as i64 - (min * p).round() as i64 + 1) as u64;
        let mb = if filavail {
            so_log2(span + 1)
        } else {
            so_log2(span)
        };
        minbits = mb.min(dtype_len);
        // minval stores the bit pattern of `min` in the float type
        minval = match size {
            4 => {
                let mut b = [0u8; 8];
                b[..4].copy_from_slice(&(min as f32).to_le_bytes());
                u64::from_le_bytes(b)
            }
            _ => u64::from_le_bytes(min.to_le_bytes()),
        };
        for i in 0..nelmts {
            let v = getf(i);
            vals[i] = if minbits == dtype_len {
                read_cell(i)
            } else if is_fill(v) {
                (1u64 << minbits) - 1
            } else {
                ((v * p).round() - (min * p).round()) as i64 as u64
            };
        }
    } else {
        return Err("scaleoffset: unsupported scale type".into());
    }

    // header: minbits(4) + minval size byte + minval(8) + pad to 21
    let mut out = Vec::with_capacity((21 + (nelmts * minbits as usize + 7).min(1 << 16)) / 8);
    out.extend_from_slice(&minbits.to_le_bytes());
    out.push(8);
    out.extend_from_slice(&minval.to_le_bytes());
    out.resize(21, 0);
    if minbits > 0 {
        // MSB-first bit packing, mirroring the decode loops
        let mut cur = 0u8;
        let mut bits_free = 8usize;
        let begin_i = size - 1 - ((dtype_len - minbits) / 8) as usize;
        for &v in vals.iter().take(nelmts) {
            let bytes = v.to_le_bytes();
            for k in (0..=begin_i).rev() {
                let bits_to_copy = if k == begin_i {
                    8 - ((dtype_len - minbits) % 8) as usize
                } else {
                    8
                };
                let mut chunk = bytes[k] & (!0u8 >> (8 - bits_to_copy));
                let mut n = bits_to_copy;
                while n > 0 {
                    if n <= bits_free {
                        cur |= chunk << (bits_free - n);
                        bits_free -= n;
                        n = 0;
                    } else {
                        cur |= chunk >> (n - bits_free);
                        n -= bits_free;
                        chunk &= !0u8 >> (8 - n);
                        bits_free = 0;
                    }
                    if bits_free == 0 {
                        out.push(cur);
                        cur = 0;
                        bits_free = 8;
                    }
                }
            }
        }
        if bits_free < 8 {
            out.push(cur);
        }
    }
    // libhdf5 sizes the compressed buffer as n*minbits/8 + 1 (integer
    // division plus one spare byte); match it exactly
    let expect = 21 + (nelmts * size * minbits as usize) / (size * 8) + 1;
    out.resize(expect.max(out.len()), 0);
    Ok(out)
}
