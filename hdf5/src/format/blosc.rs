//! Pure-Rust blosc1 frame codec (HDF5 filter id 32001), transcribed from
//! c-blosc `blosc.c`/`blosclz.c` and verified against hdf5plugin output.
//!
//! Frame: 16-byte header {version, versionlz, flags, typesize, nbytes u32,
//! blocksize u32, cbytes u32}, then (unless memcpyed) one i32 start offset per
//! block, then per-block streams. Each block is split into `typesize` streams
//! when eligible; every stream is `{csize: i32}` + payload, with
//! `csize == neblock` meaning "stored uncompressed". Byte-shuffle is applied
//! per block. Codec id lives in flags bits 5-7.

use std::io::Read;

use crate::error::Result;

const FLAG_SHUFFLE: u8 = 0x01;
const FLAG_MEMCPYED: u8 = 0x02;
const FLAG_BITSHUFFLE: u8 = 0x04;
const FLAG_DONT_SPLIT: u8 = 0x10;

pub const CODEC_BLOSCLZ: u8 = 0;
pub const CODEC_LZ4: u8 = 1;
pub const CODEC_SNAPPY: u8 = 2;
pub const CODEC_ZLIB: u8 = 3;
pub const CODEC_ZSTD: u8 = 4;

const MAX_SPLITS: usize = 16;
const MIN_BUFFERSIZE: usize = 128;

/// Decompress a blosc1 frame.
pub fn decompress(src: &[u8]) -> Result<Vec<u8>> {
    if src.len() < 16 {
        return Err("blosc: frame too short".into());
    }
    let _version = src[0];
    let _versionlz = src[1];
    let flags = src[2];
    let typesize = src[3] as usize;
    let nbytes = u32::from_le_bytes(src[4..8].try_into().unwrap()) as usize;
    let blocksize = u32::from_le_bytes(src[8..12].try_into().unwrap()) as usize;
    let cbytes = u32::from_le_bytes(src[12..16].try_into().unwrap()) as usize;
    if cbytes > src.len() {
        return Err("blosc: cbytes exceeds input".into());
    }
    if nbytes == 0 {
        return Ok(Vec::new());
    }
    if flags & FLAG_MEMCPYED != 0 {
        if 16 + nbytes > src.len() {
            return Err("blosc: memcpyed frame too short".into());
        }
        return Ok(src[16..16 + nbytes].to_vec());
    }
    let do_bitshuffle = flags & FLAG_BITSHUFFLE != 0 && typesize >= 1;
    let codec = flags >> 5;
    let blocksize = blocksize.max(1);
    let nblocks = nbytes.div_ceil(blocksize);
    let leftover = nbytes % blocksize;
    let bstarts = 16;
    if bstarts + nblocks * 4 > src.len() {
        return Err("blosc: truncated block starts".into());
    }
    let mut out = vec![0u8; nbytes];
    let do_shuffle = flags & FLAG_SHUFFLE != 0 && typesize > 1;
    let dont_split = flags & FLAG_DONT_SPLIT != 0;

    for j in 0..nblocks {
        let start = u32::from_le_bytes(
            src[bstarts + j * 4..bstarts + j * 4 + 4]
                .try_into()
                .unwrap(),
        ) as usize;
        let is_leftover = j == nblocks - 1 && leftover > 0;
        let bsize = if is_leftover { leftover } else { blocksize };
        // split policy (blosc_d)
        let nsplits = if !dont_split
            && (1..=MAX_SPLITS).contains(&typesize)
            && blocksize / typesize.max(1) >= MIN_BUFFERSIZE
            && !is_leftover
        {
            typesize
        } else {
            1
        };
        let neblock = bsize / nsplits;
        let mut block = Vec::with_capacity((bsize).min(1 << 16));
        let mut off = start;
        for _ in 0..nsplits {
            if off + 4 > src.len() {
                return Err("blosc: truncated stream size".into());
            }
            let csize = i32::from_le_bytes(src[off..off + 4].try_into().unwrap());
            off += 4;
            if csize < 0 {
                return Err("blosc: negative stream size".into());
            }
            let csize = csize as usize;
            if csize == 0 {
                // stream of zeros
                block.resize(block.len() + neblock, 0);
                continue;
            }
            if off + csize > src.len() {
                return Err("blosc: truncated stream".into());
            }
            let payload = &src[off..off + csize];
            off += csize;
            if csize == neblock {
                block.extend_from_slice(payload); // stored uncompressed
            } else {
                let plain = match codec {
                    CODEC_BLOSCLZ => blosclz_decompress(payload, neblock)?,
                    CODEC_LZ4 => lz4_flex::block::decompress(payload, neblock)
                        .map_err(|e| format!("blosc/lz4: {e}"))?,
                    CODEC_SNAPPY => {
                        let mut d = snap::raw::Decoder::new();
                        d.decompress_vec(payload)
                            .map_err(|e| format!("blosc/snappy: {e}"))?
                    }
                    CODEC_ZLIB => {
                        let mut d = flate2::read::ZlibDecoder::new(payload);
                        let mut v = Vec::with_capacity((neblock).min(1 << 16));
                        d.read_to_end(&mut v)
                            .map_err(|e| format!("blosc/zlib: {e}"))?;
                        v
                    }
                    CODEC_ZSTD => {
                        // capped at the sub-block's size: a bomb stops there
                        let mut v = Vec::with_capacity((neblock).min(1 << 16));
                        zstd_rs::Decompressor::new()
                            .decompress(payload, None, neblock, &mut v)
                            .map_err(|e| format!("blosc/zstd: {e}"))?;
                        v
                    }
                    c => return Err(format!("blosc: unknown codec {c}").into()),
                };
                if plain.len() != neblock {
                    return Err("blosc: stream decompressed to wrong size".into());
                }
                block.extend_from_slice(&plain);
            }
        }
        if block.len() != bsize {
            return Err("blosc: block decompressed to wrong size".into());
        }
        if do_shuffle {
            // splits store the shuffled streams; unshuffle the whole block
            block = unshuffle_bytes(typesize, &block);
        }
        if do_bitshuffle {
            block = bitunshuffle_block(typesize, &block);
        }
        out[j * blocksize..j * blocksize + bsize].copy_from_slice(&block);
    }
    Ok(out)
}

/// Compress into a blosc1 frame. Writing uses single-stream blocks
/// (`DONT_SPLIT`) with the requested codec; incompressible streams are stored
/// raw, so output is always valid.
pub fn compress(
    codec: u8,
    _clevel: u8,
    shuffle: u8,
    typesize: usize,
    data: &[u8],
) -> Result<Vec<u8>> {
    let nbytes = data.len();
    let typesize = if (1..=255).contains(&typesize) {
        typesize
    } else {
        1
    };
    let do_shuffle = shuffle == 1 && typesize > 1;
    let do_bitshuffle = shuffle == 2;
    // block size: keep simple; 256 KB rounded to a typesize multiple
    let mut blocksize = (256 * 1024).min(nbytes.max(1));
    if blocksize % typesize != 0 {
        blocksize -= blocksize % typesize;
        blocksize = blocksize.max(typesize);
    }
    let nblocks = nbytes.div_ceil(blocksize.max(1));

    let mut out = Vec::with_capacity((nbytes / 2 + 64).min(1 << 16));
    out.extend_from_slice(&[
        2, // blosc format version
        1, // blosclz format version
        (codec << 5)
            | FLAG_DONT_SPLIT
            | if do_shuffle { FLAG_SHUFFLE } else { 0 }
            | if do_bitshuffle { FLAG_BITSHUFFLE } else { 0 },
        typesize as u8,
    ]);
    out.extend_from_slice(&(nbytes as u32).to_le_bytes());
    out.extend_from_slice(&(blocksize as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // cbytes patched below
    let bstarts_at = out.len();
    out.resize(out.len() + nblocks * 4, 0);

    for j in 0..nblocks {
        let start = j * blocksize;
        let bsize = blocksize.min(nbytes - start);
        let raw = &data[start..start + bsize];
        let shuffled;
        let block: &[u8] = if do_shuffle {
            shuffled = shuffle_bytes(typesize, raw);
            &shuffled
        } else if do_bitshuffle {
            shuffled = bitshuffle_block(typesize, raw);
            &shuffled
        } else {
            raw
        };
        let off = out.len() as u32;
        out[bstarts_at + j * 4..bstarts_at + j * 4 + 4].copy_from_slice(&off.to_le_bytes());
        let compressed = match codec {
            CODEC_BLOSCLZ => blosclz_compress(block),
            CODEC_LZ4 => Some(lz4_flex::block::compress(block)),
            CODEC_SNAPPY => {
                let mut e = snap::raw::Encoder::new();
                e.compress_vec(block).ok()
            }
            CODEC_ZLIB => {
                use std::io::Write;
                let mut e =
                    flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(6));
                e.write_all(block).ok().and_then(|()| e.finish().ok())
            }
            _ => None, // zstd encode not available pure-Rust; store raw
        };
        match compressed {
            Some(c) if c.len() < bsize => {
                out.extend_from_slice(&(c.len() as i32).to_le_bytes());
                out.extend_from_slice(&c);
            }
            _ => {
                out.extend_from_slice(&(bsize as i32).to_le_bytes());
                out.extend_from_slice(block);
            }
        }
    }
    let cbytes = out.len() as u32;
    out[12..16].copy_from_slice(&cbytes.to_le_bytes());
    Ok(out)
}

/// Blosc's byte shuffle is HDF5's (trailing partial element copied as is):
/// the same SIMD kernels.
fn shuffle_bytes(typesize: usize, data: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; data.len()];
    super::simd::shuffle(typesize, data, &mut out);
    out
}

fn unshuffle_bytes(typesize: usize, data: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; data.len()];
    super::simd::unshuffle(typesize, data, &mut out);
    out
}

// ---------------------------------------------------------------------------
// blosclz codec (blosclz.c)
// ---------------------------------------------------------------------------

const BLZ_MAX_DISTANCE: usize = 8191;

/// Decompress a blosclz stream into exactly `outlen` bytes.
// Loop shape intentionally mirrors c-blosc's blosclz.c.
#[allow(clippy::explicit_counter_loop)]
pub fn blosclz_decompress(input: &[u8], outlen: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity((outlen).min(1 << 16));
    if input.is_empty() {
        return Ok(out);
    }
    let mut ip = 0usize;
    let mut ctrl = (input[ip] & 31) as usize;
    ip += 1;
    loop {
        if ctrl >= 32 {
            // match
            let mut len = (ctrl >> 5) - 1;
            let mut ofs = (ctrl & 31) << 8;
            if len == 7 - 1 {
                loop {
                    if ip >= input.len() {
                        return Err("blosclz: truncated match length".into());
                    }
                    let code = input[ip] as usize;
                    ip += 1;
                    len += code;
                    if code != 255 {
                        break;
                    }
                }
            }
            if ip >= input.len() {
                return Err("blosclz: truncated match".into());
            }
            let code = input[ip] as usize;
            ip += 1;
            len += 3;
            let mut dist = ofs + code;
            if code == 255 && ofs == 31 << 8 {
                if ip + 1 >= input.len() {
                    return Err("blosclz: truncated far match".into());
                }
                ofs = (input[ip] as usize) << 8;
                ofs += input[ip + 1] as usize;
                ip += 2;
                dist = ofs + BLZ_MAX_DISTANCE;
            }
            // final reference: one further back (ref-- in the C code)
            let dist = dist + 1;
            if dist > out.len() {
                return Err("blosclz: back reference before start".into());
            }
            // Faithful to blosclz.c: the next control byte is read *before*
            // the copy, and a stream that ends at a match token terminates
            // WITHOUT performing that final copy.
            if ip >= input.len() {
                break;
            }
            ctrl = input[ip] as usize;
            ip += 1;
            let mut rpos = out.len() - dist;
            for _ in 0..len {
                let b = out[rpos];
                out.push(b);
                rpos += 1;
            }
        } else {
            // literal run of ctrl + 1 bytes
            let run = ctrl + 1;
            if ip + run > input.len() {
                return Err("blosclz: truncated literal run".into());
            }
            out.extend_from_slice(&input[ip..ip + run]);
            ip += run;
            if ip >= input.len() {
                break;
            }
            ctrl = input[ip] as usize;
            ip += 1;
        }
        if out.len() > outlen {
            return Err("blosclz: output overrun".into());
        }
    }
    if out.len() != outlen {
        return Err("blosclz: output size mismatch".into());
    }
    Ok(out)
}

/// Compress with blosclz; `None` when the input is too small to bother.
pub fn blosclz_compress(input: &[u8]) -> Option<Vec<u8>> {
    let n = input.len();
    if n < 16 {
        return None;
    }
    const HLOG: usize = 14;
    let mut htab = vec![usize::MAX; 1 << HLOG];
    let hash = |a: u8, b: u8, c: u8| -> usize {
        let v = (u32::from(a) << 16) | (u32::from(b) << 8) | u32::from(c);
        (v.wrapping_mul(2_654_435_761) >> (32 - HLOG)) as usize
    };
    let mut out: Vec<u8> = Vec::with_capacity((n).min(1 << 16));
    let mut lit_start = 0usize;
    let mut pos = 0usize;

    let emit_literals = |out: &mut Vec<u8>, data: &[u8], from: usize, to: usize| {
        let mut s = from;
        while s < to {
            let run = (to - s).min(32);
            out.push((run - 1) as u8);
            out.extend_from_slice(&data[s..s + run]);
            s += run;
        }
    };

    while pos + 2 < n {
        let h = hash(input[pos], input[pos + 1], input[pos + 2]);
        let cand = htab[h];
        htab[h] = pos;
        let dist = if cand == usize::MAX {
            usize::MAX
        } else {
            pos - cand
        };
        // distance encodes as (dist - 1) in 5+8 bits; avoid the far-match
        // marker pattern (high == 31 && low == 255)
        if dist != usize::MAX
            && (1..=BLZ_MAX_DISTANCE).contains(&dist)
            && !((dist - 1) >> 8 == 31 && (dist - 1) & 0xff == 255)
            && pos != lit_start // a match token may not begin the stream
            && input[cand] == input[pos]
            && input[cand + 1] == input[pos + 1]
            && input[cand + 2] == input[pos + 2]
        {
            // never let a match reach the end of the stream: the decoder
            // (like blosclz.c) drops a match token with no following byte
            let max_len = (n - pos).saturating_sub(1);
            if max_len < 3 {
                pos += 1;
                continue;
            }
            let mut mlen = 3;
            while mlen < max_len && input[cand + mlen] == input[pos + mlen] {
                mlen += 1;
            }
            emit_literals(&mut out, input, lit_start, pos);
            let d1 = dist - 1;
            let l = mlen - 3; // decoder: len = X-1 [+ext] + 3
            if l < 6 {
                // X = l + 1 in 1..=6
                out.push((((l + 1) << 5) | (d1 >> 8)) as u8);
            } else {
                // X = 7 with 255-continuation encoding of (l - 6)
                out.push(((7 << 5) | (d1 >> 8)) as u8);
                let mut rest = l - 6;
                while rest >= 255 {
                    out.push(255);
                    rest -= 255;
                }
                out.push(rest as u8);
            }
            out.push((d1 & 0xff) as u8);
            pos += mlen;
            lit_start = pos;
        } else {
            pos += 1;
        }
    }
    emit_literals(&mut out, input, lit_start, n);
    if out.len() < n {
        Some(out)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// bit-shuffle (bitshuffle-generic.c scalar path, little-endian)
// ---------------------------------------------------------------------------

/// 8x8 bit-matrix transpose (TRANS_BIT_8X8).
#[inline]
fn trans_bit_8x8(mut x: u64) -> u64 {
    let mut t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AA;
    x = x ^ t ^ (t << 7);
    t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCC;
    x = x ^ t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x0000_0000_F0F0_F0F0;
    x = x ^ t ^ (t << 28);
    x
}

/// Forward bit-shuffle of one block (blosc `bitshuffle()` wrapper semantics:
/// only a multiple-of-8-elements prefix is transformed; the rest is copied).
pub fn bitshuffle_block(typesize: usize, data: &[u8]) -> Vec<u8> {
    let size = data.len() / typesize;
    if size % 8 != 0 || size == 0 {
        return data.to_vec();
    }
    let nbyte = size * typesize;
    let nbyte_bitrow = nbyte / 8;
    // 1. byte-transpose within elements
    let mut a = shuffle_bytes(typesize, &data[..nbyte]);
    // 2. bit-transpose within each u64 word, then byte k of word ii to
    //    b[k * nbyte_bitrow + ii] — an 8-byte shuffle of the words
    trans_bit_words(&mut a);
    let mut b = vec![0u8; nbyte];
    super::simd::shuffle(8, &a, &mut b);
    debug_assert_eq!(nbyte_bitrow * 8, nbyte);
    // 3. transpose 8 x typesize blocks of (size/8) bytes
    let blk = size / 8;
    let mut out = vec![0u8; data.len()];
    for ii in 0..8 {
        for jj in 0..typesize {
            let src = (ii * typesize + jj) * blk;
            let dst = (jj * 8 + ii) * blk;
            out[dst..dst + blk].copy_from_slice(&b[src..src + blk]);
        }
    }
    out[nbyte..].copy_from_slice(&data[nbyte..]);
    out
}

/// Inverse of [`bitshuffle_block`].
pub fn bitunshuffle_block(typesize: usize, data: &[u8]) -> Vec<u8> {
    let size = data.len() / typesize;
    if size % 8 != 0 || size == 0 {
        return data.to_vec();
    }
    let nbyte = size * typesize;
    let nbyte_bitrow = nbyte / 8;
    let blk = size / 8;
    // inverse step 3
    let mut b = vec![0u8; nbyte];
    for ii in 0..8 {
        for jj in 0..typesize {
            let src = (jj * 8 + ii) * blk;
            let dst = (ii * typesize + jj) * blk;
            b[dst..dst + blk].copy_from_slice(&data[src..src + blk]);
        }
    }
    // inverse step 2: gather each word's bytes back, transpose (its own
    // inverse)
    let _ = nbyte_bitrow;
    let mut a = vec![0u8; nbyte];
    super::simd::unshuffle(8, &b, &mut a);
    trans_bit_words(&mut a);
    // inverse step 1
    let mut out = vec![0u8; data.len()];
    super::simd::unshuffle(typesize, &a, &mut out[..nbyte]);
    out[nbyte..].copy_from_slice(&data[nbyte..]);
    out
}

/// [`trans_bit_8x8`] on every 8-byte word, in place — shifts and masks on
/// 64-bit lanes, which the compiler vectorizes.
fn trans_bit_words(words: &mut [u8]) {
    for w in words.chunks_exact_mut(8) {
        let x = trans_bit_8x8(u64::from_le_bytes(w.try_into().unwrap()));
        w.copy_from_slice(&x.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scalar bitshuffle this crate shipped (validated against
    /// hdf5plugin), kept as the reference for the vectorized one.
    fn bitshuffle_reference(typesize: usize, data: &[u8]) -> Vec<u8> {
        let size = data.len() / typesize;
        if size % 8 != 0 || size == 0 {
            return data.to_vec();
        }
        let nbyte = size * typesize;
        let rows = nbyte / 8;
        let mut a = vec![0u8; nbyte];
        for j in 0..typesize {
            for i in 0..size {
                a[j * size + i] = data[i * typesize + j];
            }
        }
        let mut b = vec![0u8; nbyte];
        for ii in 0..rows {
            let x = trans_bit_8x8(u64::from_le_bytes(
                a[ii * 8..ii * 8 + 8].try_into().unwrap(),
            ));
            for (kk, &byte) in x.to_le_bytes().iter().enumerate() {
                b[kk * rows + ii] = byte;
            }
        }
        let blk = size / 8;
        let mut out = vec![0u8; data.len()];
        for ii in 0..8 {
            for jj in 0..typesize {
                let (src, dst) = ((ii * typesize + jj) * blk, (jj * 8 + ii) * blk);
                out[dst..dst + blk].copy_from_slice(&b[src..src + blk]);
            }
        }
        out[nbyte..].copy_from_slice(&data[nbyte..]);
        out
    }

    #[test]
    fn bitshuffle_matches_reference() {
        let mut x = 0x1234_5678u32;
        let data: Vec<u8> = (0..40_000)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                x as u8
            })
            .collect();
        for ts in [1usize, 2, 4, 8, 16, 3] {
            for len in [8 * ts * 16, 8 * ts * 1000 + 5, 8 * ts * 3] {
                let d = &data[..len.min(data.len())];
                let want = bitshuffle_reference(ts, d);
                assert_eq!(bitshuffle_block(ts, d), want, "typesize {ts} len {len}");
                assert_eq!(
                    bitunshuffle_block(ts, &want),
                    d,
                    "inverse typesize {ts} len {len}"
                );
            }
        }
    }

    #[test]
    fn blosclz_roundtrip() {
        for data in [
            b"hello hello hello hello hello hello".to_vec(),
            (0..50_000u32)
                .flat_map(|i| ((i * 7) % 253).to_le_bytes())
                .collect::<Vec<u8>>(),
            vec![42u8; 10_000],
        ] {
            if let Some(c) = blosclz_compress(&data) {
                assert_eq!(blosclz_decompress(&c, data.len()).unwrap(), data);
            }
        }
    }

    #[test]
    fn frame_roundtrip_all_codecs() {
        let data: Vec<u8> = (0..300_000u32)
            .flat_map(|i| ((i / 3) as u16).to_le_bytes())
            .collect();
        for codec in [
            CODEC_BLOSCLZ,
            CODEC_LZ4,
            CODEC_SNAPPY,
            CODEC_ZLIB,
            CODEC_ZSTD,
        ] {
            for shuffle in [0u8, 1, 2] {
                let c = compress(codec, 5, shuffle, 2, &data).unwrap();
                assert_eq!(
                    decompress(&c).unwrap(),
                    data,
                    "codec {codec} shuffle {shuffle}"
                );
            }
        }
    }
}
