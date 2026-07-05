//! SZip (extended-Rice / CCSDS 121.0) codec, transcribed from libaec
//! (`decode.c`, `encode.c`, `sz_compat.c`) and validated against the
//! Homebrew `libsz` implementation. Only the options HDF5 uses are
//! supported: EC/NN coding, LSB/MSB sample order, 8/16/32/64 bits per pixel
//! (32/64-bit data is byte-interleaved and coded as 8-bit samples, exactly
//! like `SZ_BufftoBuffCompress`).

use crate::error::Result;

pub const SZ_EC_OPTION_MASK: u32 = 4;
pub const SZ_LSB_OPTION_MASK: u32 = 8;
pub const SZ_MSB_OPTION_MASK: u32 = 16;
pub const SZ_NN_OPTION_MASK: u32 = 32;
pub const SZ_RAW_OPTION_MASK: u32 = 128;

const ROS: u32 = 5; // "rest of segment" zero-block marker

// ---------------------------------------------------------------------------
// bit I/O (MSB-first within bytes)
// ---------------------------------------------------------------------------

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize, // bit position
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn bits(&mut self, n: u32) -> Result<u64> {
        let mut v = 0u64;
        for _ in 0..n {
            let byte = self.pos / 8;
            if byte >= self.data.len() {
                return Err("szip: stream truncated".into());
            }
            let bit = (self.data[byte] >> (7 - (self.pos % 8))) & 1;
            v = (v << 1) | u64::from(bit);
            self.pos += 1;
        }
        Ok(v)
    }

    /// Fundamental-sequence value: number of 0 bits before the next 1.
    fn fs(&mut self) -> Result<u32> {
        let mut n = 0u32;
        loop {
            let byte = self.pos / 8;
            if byte >= self.data.len() {
                return Err("szip: stream truncated in FS".into());
            }
            let bit = (self.data[byte] >> (7 - (self.pos % 8))) & 1;
            self.pos += 1;
            if bit == 1 {
                return Ok(n);
            }
            n += 1;
            if n > 1 << 28 {
                return Err("szip: unterminated FS".into());
            }
        }
    }
}

#[derive(Default)]
struct BitWriter {
    out: Vec<u8>,
    cur: u8,
    nbits: u8,
}

impl BitWriter {
    fn put(&mut self, v: u64, n: u32) {
        for i in (0..n).rev() {
            self.cur = (self.cur << 1) | (((v >> i) & 1) as u8);
            self.nbits += 1;
            if self.nbits == 8 {
                self.out.push(self.cur);
                self.cur = 0;
                self.nbits = 0;
            }
        }
    }

    fn fs(&mut self, v: u32) {
        for _ in 0..v {
            self.put(0, 1);
        }
        self.put(1, 1);
    }

    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.out.push(self.cur << (8 - self.nbits));
        }
        self.out
    }
}

// ---------------------------------------------------------------------------
// AEC core
// ---------------------------------------------------------------------------

fn id_len_for(bps: u32) -> u32 {
    if bps > 16 {
        5
    } else if bps > 8 {
        4
    } else {
        3
    }
}

/// Second-extension table: for gamma index m, (group i, first index of group).
fn se_table() -> [(u32, u32); 91] {
    let mut t = [(0u32, 0u32); 91];
    let mut k = 0usize;
    for i in 0..13u32 {
        let ms = k as u32;
        for _ in 0..=i {
            t[k] = (i, ms);
            k += 1;
        }
    }
    t
}

/// Decode `n_samples` samples from an AEC stream.
fn aec_decode(
    src: &[u8],
    bps: u32,
    block_size: usize,
    rsi: usize,
    pp: bool,
    n_samples: usize,
) -> Result<Vec<u32>> {
    let mut r = BitReader::new(src);
    let id_len = id_len_for(bps);
    let modi = 1u64 << id_len;
    let xmax: u32 = if bps == 32 {
        u32::MAX
    } else {
        (1u32 << bps) - 1
    };
    let se = se_table();
    let rsi_samples = rsi * block_size;

    let mut raw: Vec<u32> = Vec::with_capacity((n_samples + block_size).min(1 << 16));
    let mut out: Vec<u32> = Vec::with_capacity((n_samples + block_size).min(1 << 16));

    // Unmap one completed RSI (libaec FLUSH, unsigned branch).
    let unmap_rsi = |raw: &mut Vec<u32>, out: &mut Vec<u32>| {
        if !pp {
            out.append(raw);
            return;
        }
        let mut data = raw[0]; // reference sample, stored verbatim
        out.push(data);
        let med = xmax / 2 + 1;
        for &d in &raw[1..] {
            let half_d = (d >> 1) + (d & 1);
            let mask = if data & med != 0 { xmax } else { 0 };
            if half_d <= (mask ^ data) {
                if d & 1 == 1 {
                    data = data.wrapping_sub((d >> 1) + 1);
                } else {
                    data = data.wrapping_add(d >> 1);
                }
            } else {
                data = mask ^ d;
            }
            out.push(data);
        }
        raw.clear();
    };

    while out.len() < n_samples {
        let in_rsi = raw.len(); // samples decoded so far in the current RSI
        let refflag = pp && in_rsi == 0;
        let ebs = if refflag { block_size - 1 } else { block_size };
        let id = r.bits(id_len)?;
        if id == 0 {
            // low-entropy: 1 more bit selects second-extension vs zero-block
            let id2 = r.bits(1)?;
            if refflag {
                raw.push(r.bits(bps)? as u32);
            }
            if id2 == 1 {
                // second extension: pairs from FS gamma values
                let mut count = if refflag { 1 } else { 0 };
                while count < block_size {
                    let m = r.fs()? as usize;
                    if m >= se.len() {
                        return Err("szip: bad second-extension gamma".into());
                    }
                    let (i, ms) = se[m];
                    let d1 = m as u32 - ms;
                    if count % 2 == 0 {
                        raw.push(i - d1);
                        count += 1;
                        if count >= block_size {
                            // d1 belongs to the next block position; libaec
                            // still emits it within this block only
                        }
                    }
                    raw.push(d1);
                    count += 1;
                }
            } else {
                // zero block(s)
                let fsv = r.fs()?;
                let mut zero_blocks = fsv + 1;
                if zero_blocks == ROS {
                    let b = (in_rsi + if refflag { 1 } else { 0 }) / block_size;
                    let left_in_rsi = rsi - b;
                    let left_in_seg = 64 - (b % 64);
                    zero_blocks = left_in_rsi.min(left_in_seg) as u32;
                } else if zero_blocks > ROS {
                    zero_blocks -= 1;
                }
                let zero_samples = zero_blocks as usize * block_size - usize::from(refflag);
                raw.resize(raw.len() + zero_samples, 0);
            }
        } else if id == modi - 1 {
            // uncompressed: block_size raw samples (ref occupies slot 0)
            for _ in 0..block_size {
                raw.push(r.bits(bps)? as u32);
            }
        } else {
            // split-sample, k = id - 1
            let k = (id - 1) as u32;
            if refflag {
                raw.push(r.bits(bps)? as u32);
            }
            let base = raw.len();
            for _ in 0..ebs {
                raw.push(r.fs()? << k);
            }
            if k > 0 {
                for i in 0..ebs {
                    raw[base + i] += r.bits(k)? as u32;
                }
            }
        }
        if raw.len() >= rsi_samples {
            debug_assert_eq!(raw.len(), rsi_samples);
            unmap_rsi(&mut raw, &mut out);
        }
    }
    if !raw.is_empty() {
        unmap_rsi(&mut raw, &mut out);
    }
    out.truncate(n_samples);
    Ok(out)
}

/// Encode samples as an AEC stream (split-sample and uncompressed options
/// only — a valid encoder choice the reference decoder accepts).
fn aec_encode(samples: &[u32], bps: u32, block_size: usize, rsi: usize, pp: bool) -> Vec<u8> {
    let mut w = BitWriter::default();
    let id_len = id_len_for(bps);
    let modi = 1u64 << id_len;
    let kmax = (modi - 3) as u32; // ids 1..=modi-2 carry k = id-1
    let xmax: u64 = if bps == 32 {
        u32::MAX as u64
    } else {
        (1u64 << bps) - 1
    };
    let rsi_samples = rsi * block_size;

    for rsi_chunk in samples.chunks(rsi_samples) {
        // preprocess: reference + mapped deltas
        let mut coded: Vec<u32> = Vec::with_capacity((rsi_samples).min(1 << 16));
        if pp {
            let mut prev = u64::from(rsi_chunk[0]);
            coded.push(rsi_chunk[0]);
            for &x in &rsi_chunk[1..] {
                let x = u64::from(x);
                let d = x as i64 - prev as i64;
                let theta = prev.min(xmax - prev) as i64;
                let m = if d >= 0 && d <= theta {
                    (2 * d) as u64
                } else if d < 0 && -d <= theta {
                    (-2 * d - 1) as u64
                } else {
                    (theta + d.abs()) as u64
                };
                coded.push(m as u32);
                prev = x;
            }
        } else {
            coded.extend_from_slice(rsi_chunk);
        }
        // pad the trailing partial block with zero deltas
        let nblocks = coded.len().div_ceil(block_size);
        coded.resize(nblocks * block_size, 0);

        for (bi, block) in coded.chunks(block_size).enumerate() {
            let refflag = pp && bi == 0;
            let vals = if refflag { &block[1..] } else { block };
            // choose the cheapest k, versus the uncompressed option
            let mut best_k = 0u32;
            let mut best_cost = u64::MAX;
            for k in 0..=kmax {
                let mut cost = u64::from(k) * vals.len() as u64 + vals.len() as u64;
                for &v in vals {
                    cost += u64::from(v >> k);
                    if cost >= best_cost {
                        break;
                    }
                }
                if cost < best_cost {
                    best_cost = cost;
                    best_k = k;
                }
            }
            let uncomp_cost =
                block.len() as u64 * u64::from(bps) - if refflag { u64::from(bps) } else { 0 };
            if best_cost >= uncomp_cost {
                w.put(modi - 1, id_len); // uncompressed
                for &v in block {
                    w.put(u64::from(v), bps);
                }
            } else {
                w.put(u64::from(best_k) + 1, id_len);
                if refflag {
                    w.put(u64::from(block[0]), bps);
                }
                for &v in vals {
                    w.fs(v >> best_k);
                }
                if best_k > 0 {
                    for &v in vals {
                        w.put(u64::from(v) & ((1 << best_k) - 1), best_k);
                    }
                }
            }
        }
    }
    w.finish()
}

// ---------------------------------------------------------------------------
// SZ compatibility layer (sz_compat.c semantics)
// ---------------------------------------------------------------------------

struct SzParams {
    mask: u32,
    bpp: u32,
    ppb: usize,
    pps: usize,
}

fn params(cdata: &[u32]) -> Result<SzParams> {
    if cdata.len() < 4 {
        return Err("szip: missing filter parameters".into());
    }
    let p = SzParams {
        mask: cdata[0],
        bpp: cdata[1],
        ppb: cdata[2] as usize,
        pps: cdata[3] as usize,
    };
    if p.pps == 0
        || p.pps > 4096
        || p.ppb == 0
        || p.ppb % 2 == 1
        || p.bpp == 0
        || (p.bpp > 32 && p.bpp != 64)
    {
        return Err("szip: invalid parameters".into());
    }
    if p.mask & SZ_RAW_OPTION_MASK != 0 {
        return Err("szip: raw option not supported".into());
    }
    Ok(p)
}

fn interleave(src: &[u8], ws: usize) -> Vec<u8> {
    let n = src.len();
    let words = n / ws;
    let mut out = vec![0u8; n];
    for i in 0..words {
        for j in 0..ws {
            out[j * words + i] = src[i * ws + j];
        }
    }
    out
}

fn deinterleave(src: &[u8], ws: usize) -> Vec<u8> {
    let n = src.len();
    let words = n / ws;
    let mut out = vec![0u8; n];
    for i in 0..words {
        for j in 0..ws {
            out[i * ws + j] = src[j * words + i];
        }
    }
    out
}

/// Read samples from bytes per the LSB/MSB option.
fn to_samples(data: &[u8], bps: u32, msb: bool) -> Vec<u32> {
    match bps {
        b if b <= 8 => data.iter().map(|&x| u32::from(x)).collect(),
        b if b <= 16 => data
            .chunks_exact(2)
            .map(|c| {
                if msb {
                    u32::from(u16::from_be_bytes([c[0], c[1]]))
                } else {
                    u32::from(u16::from_le_bytes([c[0], c[1]]))
                }
            })
            .collect(),
        _ => data
            .chunks_exact(4)
            .map(|c| {
                if msb {
                    u32::from_be_bytes([c[0], c[1], c[2], c[3]])
                } else {
                    u32::from_le_bytes([c[0], c[1], c[2], c[3]])
                }
            })
            .collect(),
    }
}

fn from_samples(samples: &[u32], bps: u32, msb: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity((samples.len().min(1 << 16)) * (bps as usize).div_ceil(8));
    for &s in samples {
        match bps {
            b if b <= 8 => out.push(s as u8),
            b if b <= 16 => out.extend_from_slice(&if msb {
                (s as u16).to_be_bytes()
            } else {
                (s as u16).to_le_bytes()
            }),
            _ => out.extend_from_slice(&if msb {
                s.to_be_bytes()
            } else {
                s.to_le_bytes()
            }),
        }
    }
    out
}

/// `SZ_BufftoBuffCompress` equivalent.
pub fn sz_compress(cdata: &[u32], data: &[u8]) -> Result<Vec<u8>> {
    let p = params(cdata)?;
    let pp = p.mask & SZ_NN_OPTION_MASK != 0;
    let msb = p.mask & SZ_MSB_OPTION_MASK != 0;
    let inter = p.bpp == 32 || p.bpp == 64;
    let bps = if inter { 8 } else { p.bpp };
    let pixel_size = if bps > 16 {
        4
    } else if bps > 8 {
        2
    } else {
        1
    };
    let rsi = (p.pps).div_ceil(p.ppb);

    let buf;
    let src: &[u8] = if inter {
        buf = interleave(data, (p.bpp / 8) as usize);
        &buf
    } else {
        data
    };

    // pad each scanline to rsi * block_size samples when pps % ppb != 0,
    // repeating the last pixel (NN) or zeros (EC) like add_padding()
    let line_bytes = p.pps * pixel_size;
    let padded;
    let src: &[u8] = if p.pps % p.ppb != 0 {
        let padded_line = rsi * p.ppb * pixel_size;
        let mut v = Vec::with_capacity((src.len().min(1 << 16)).div_ceil(line_bytes) * padded_line);
        let mut i = 0;
        while i < src.len() {
            let ls = line_bytes.min(src.len() - i);
            v.extend_from_slice(&src[i..i + ls]);
            i += ls;
            let pad = padded_line - ls;
            if pp && ls >= pixel_size {
                let last = &src[i - pixel_size..i];
                for _ in 0..pad / pixel_size {
                    v.extend_from_slice(last);
                }
            } else {
                v.resize(v.len() + pad, 0);
            }
        }
        padded = v;
        &padded
    } else {
        src
    };

    let samples = to_samples(src, bps, msb);
    Ok(aec_encode(&samples, bps, p.ppb, rsi, pp))
}

/// `SZ_BufftoBuffDecompress` equivalent; `out_len` is the original size.
pub fn sz_decompress(cdata: &[u32], data: &[u8], out_len: usize) -> Result<Vec<u8>> {
    let p = params(cdata)?;
    let pp = p.mask & SZ_NN_OPTION_MASK != 0;
    let msb = p.mask & SZ_MSB_OPTION_MASK != 0;
    let inter = p.bpp == 32 || p.bpp == 64;
    let bps = if inter { 8 } else { p.bpp };
    let pixel_size: usize = if bps > 16 {
        4
    } else if bps > 8 {
        2
    } else {
        1
    };
    let rsi = (p.pps).div_ceil(p.ppb);
    let pad_scanline = p.pps % p.ppb != 0;

    let line_samples = p.pps;
    let padded_line = rsi * p.ppb;
    let out_samples = out_len / pixel_size;
    let n_samples = if pad_scanline {
        (out_samples).div_ceil(line_samples) * padded_line
    } else {
        out_samples
    };

    let samples = aec_decode(data, bps, p.ppb, rsi, pp, n_samples)?;
    let samples: Vec<u32> = if pad_scanline {
        samples
            .chunks(padded_line)
            .flat_map(|line| line[..line_samples.min(line.len())].to_vec())
            .collect()
    } else {
        samples
    };
    let mut bytes = from_samples(&samples, bps, msb);
    bytes.truncate(out_len);
    if inter {
        bytes = deinterleave(&bytes, (p.bpp / 8) as usize);
    }
    Ok(bytes)
}
