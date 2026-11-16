//! SIMD kernels for the hot byte-level loops: HDF5's shuffle filter, byte
//! swapping and the fletcher32 checksum.
//!
//! Shuffle (and its inverse) is a byte transpose. For element sizes 2, 4, 8
//! and 16 it is done 16 elements at a time as `log2(size)` rounds of
//! byte-unzip (even/odd bytes of two vectors) or byte-zip (interleave two
//! vectors) — the "perfect shuffle" recursion — so each target only has to
//! supply those two primitives:
//!
//! * `aarch64` — NEON `uzp1/uzp2` and `zip1/zip2` (always available),
//! * `x86_64` — SSE2 `unpack` and `packus` (always available),
//! * `wasm32` with `simd128` enabled (`-C target-feature=+simd128`) —
//!   `i8x16.shuffle`.
//!
//! Every other target, element size and the tail of every buffer go through
//! the scalar code, which the tests check the vector paths against.

/// Byte-shuffle into `out`: byte `j` of element `i` goes to `out[j * n + i]`.
/// Trailing bytes that do not make a whole element are copied verbatim.
pub fn shuffle(esize: usize, data: &[u8], out: &mut [u8]) {
    debug_assert_eq!(data.len(), out.len());
    let n = data.len() / esize.max(1);
    if esize <= 1 || n == 0 {
        out.copy_from_slice(data);
        return;
    }
    let done = vector::shuffle(esize, data, out, n);
    for j in 0..esize {
        let plane = &mut out[j * n..(j + 1) * n];
        for (i, b) in plane.iter_mut().enumerate().skip(done) {
            *b = data[i * esize + j];
        }
    }
    let tail = n * esize;
    out[tail..].copy_from_slice(&data[tail..]);
}

/// Inverse of [`shuffle`].
pub fn unshuffle(esize: usize, data: &[u8], out: &mut [u8]) {
    debug_assert_eq!(data.len(), out.len());
    let n = data.len() / esize.max(1);
    if esize <= 1 || n == 0 {
        out.copy_from_slice(data);
        return;
    }
    let done = vector::unshuffle(esize, data, out, n);
    match esize {
        2 => scalar_unshuffle::<2>(data, n, done, out),
        4 => scalar_unshuffle::<4>(data, n, done, out),
        8 => scalar_unshuffle::<8>(data, n, done, out),
        _ => {
            for j in 0..esize {
                for i in done..n {
                    out[i * esize + j] = data[j * n + i];
                }
            }
        }
    }
    let tail = n * esize;
    out[tail..].copy_from_slice(&data[tail..]);
}

/// Elements `[from, n)` of an unshuffle, the element size a constant so the
/// compiler unrolls the planes.
fn scalar_unshuffle<const E: usize>(data: &[u8], n: usize, from: usize, out: &mut [u8]) {
    let plane: [&[u8]; E] = std::array::from_fn(|j| &data[j * n..(j + 1) * n]);
    for (i, element) in out[..n * E].chunks_exact_mut(E).enumerate().skip(from) {
        for j in 0..E {
            element[j] = plane[j][i];
        }
    }
}

/// Reverse the bytes of every `width`-byte element (big ↔ little endian).
/// A trailing partial element is left alone.
pub fn swap_bytes(width: usize, buf: &mut [u8]) {
    // fixed widths compile to vector byte-reversals on every SIMD target
    macro_rules! swap {
        ($buf:expr, $t:ty) => {
            for e in $buf.chunks_exact_mut(std::mem::size_of::<$t>()) {
                let v = <$t>::from_ne_bytes(e.try_into().unwrap()).swap_bytes();
                e.copy_from_slice(&v.to_ne_bytes());
            }
        };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    let buf = {
        let done = wasm::swap_bytes(width, buf);
        &mut buf[done..]
    };
    match width {
        0 | 1 => {}
        2 => swap!(buf, u16),
        4 => swap!(buf, u32),
        8 => swap!(buf, u64),
        16 => swap!(buf, u128),
        w => buf.chunks_exact_mut(w).for_each(<[u8]>::reverse),
    }
}

/// HDF5's fletcher32 checksum (`H5_checksum_fletcher32`): 16-bit big-endian
/// words, an odd trailing byte contributing `byte << 8`, both sums folded
/// every 360 words.
///
/// Within a 360-word block, `sum1`/`sum2` only add (wrapping, mod 2³²), so
/// the block's contribution is computed in closed form —
/// `sum2 += k·sum1 + Σ (k − i)·wᵢ`, `sum1 += Σ wᵢ` — two independent
/// reductions the compiler vectorizes, instead of a serial dependency chain.
pub fn fletcher32(data: &[u8]) -> u32 {
    let fold = |s: u32| (s & 0xffff).wrapping_add(s >> 16);
    let (mut sum1, mut sum2) = (0u32, 0u32);
    let words = data.len() / 2;
    for block in data[..words * 2].chunks(720) {
        let k = (block.len() / 2) as u32;
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        let (skip, mut plain, mut weighted) = wasm::fletcher_block(block, k);
        #[cfg(target_arch = "x86_64")]
        let (skip, mut plain, mut weighted) = fletcher_x86::block(block, k);
        #[cfg(not(any(
            target_arch = "x86_64",
            all(target_arch = "wasm32", target_feature = "simd128")
        )))]
        let (skip, mut plain, mut weighted) = (0, 0u32, 0u32);
        for (i, w) in block.chunks_exact(2).enumerate().skip(skip) {
            let w = u32::from(u16::from_be_bytes([w[0], w[1]]));
            plain = plain.wrapping_add(w);
            weighted = weighted.wrapping_add(w.wrapping_mul(k - i as u32));
        }
        sum2 = sum2
            .wrapping_add(k.wrapping_mul(sum1))
            .wrapping_add(weighted);
        sum1 = sum1.wrapping_add(plain);
        sum1 = fold(sum1);
        sum2 = fold(sum2);
    }
    if data.len() % 2 != 0 {
        sum1 = sum1.wrapping_add(u32::from(data[data.len() - 1]) << 8);
        sum2 = sum2.wrapping_add(sum1);
        sum1 = fold(sum1);
        sum2 = fold(sum2);
    }
    (fold(sum2) << 16) | fold(sum1)
}

/// Fletcher32 block sums on x86 with `pmaddwd` (16-bit multiply, pairwise
/// add): SSE2 everywhere, AVX2 where present. LLVM's own vectorization of
/// the weighted sum widens to 32-bit multiplies and runs at half the speed.
///
/// `pmaddwd` multiplies *signed* words, so each big-endian word `w` goes in
/// as `s = w ^ 0x8000` (= `w − 32768` as an `i16`) and the 32768 is added
/// back in closed form: `Σw = Σs + 32768·n`, `Σ(k−i)·w = Σ(k−i)·s +
/// 32768·Σ(k−i)`. All sums are mod 2³², exactly as the scalar code's.
#[cfg(target_arch = "x86_64")]
mod fletcher_x86 {
    use std::arch::x86_64::*;

    /// `(words done, Σ wᵢ, Σ (k − i)·wᵢ)` for one block of `k ≤ 360` words.
    pub fn block(block: &[u8], k: u32) -> (usize, u32, u32) {
        // SAFETY: the AVX2 kernel only runs where the CPU has AVX2; SSE2 is
        // part of x86_64
        let (n, plain, weighted) = unsafe {
            if std::is_x86_feature_detected!("avx2") {
                avx2(block, k)
            } else {
                sse2(block, k)
            }
        };
        let n32 = n as u32;
        let bias_plain = 32768u32.wrapping_mul(n32);
        // Σ_{i<n} (k − i) = n·k − n(n−1)/2
        let weights = n32
            .wrapping_mul(k)
            .wrapping_sub(n32.wrapping_mul(n32.wrapping_sub(1)) / 2);
        (
            n,
            plain.wrapping_add(bias_plain),
            weighted.wrapping_add(32768u32.wrapping_mul(weights)),
        )
    }

    #[inline(always)]
    unsafe fn hsum128(v: __m128i) -> u32 {
        let mut lanes = [0u32; 4];
        _mm_storeu_si128(lanes.as_mut_ptr().cast(), v);
        lanes.iter().fold(0u32, |a, &b| a.wrapping_add(b))
    }

    /// Big-endian 16-bit words to signed `w − 32768`.
    #[inline(always)]
    unsafe fn words128(v: __m128i) -> __m128i {
        let native = _mm_or_si128(_mm_slli_epi16(v, 8), _mm_srli_epi16(v, 8));
        _mm_xor_si128(native, _mm_set1_epi16(i16::MIN))
    }

    unsafe fn sse2(block: &[u8], k: u32) -> (usize, u32, u32) {
        let groups = block.len() / 16;
        let k = k as i16;
        let mut weights = _mm_setr_epi16(k, k - 1, k - 2, k - 3, k - 4, k - 5, k - 6, k - 7);
        let (eight, ones) = (_mm_set1_epi16(8), _mm_set1_epi16(1));
        let (mut plain, mut weighted) = (_mm_setzero_si128(), _mm_setzero_si128());
        for g in block[..groups * 16].chunks_exact(16) {
            let s = words128(_mm_loadu_si128(g.as_ptr().cast()));
            plain = _mm_add_epi32(plain, _mm_madd_epi16(s, ones));
            weighted = _mm_add_epi32(weighted, _mm_madd_epi16(s, weights));
            weights = _mm_sub_epi16(weights, eight);
        }
        (groups * 8, hsum128(plain), hsum128(weighted))
    }

    #[target_feature(enable = "avx2")]
    unsafe fn avx2(block: &[u8], k: u32) -> (usize, u32, u32) {
        let groups = block.len() / 32;
        let k = k as i16;
        let mut weights = _mm256_setr_epi16(
            k,
            k - 1,
            k - 2,
            k - 3,
            k - 4,
            k - 5,
            k - 6,
            k - 7,
            k - 8,
            k - 9,
            k - 10,
            k - 11,
            k - 12,
            k - 13,
            k - 14,
            k - 15,
        );
        let (sixteen, ones) = (_mm256_set1_epi16(16), _mm256_set1_epi16(1));
        let flip = _mm256_set1_epi16(i16::MIN);
        let (mut plain, mut weighted) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        for g in block[..groups * 32].chunks_exact(32) {
            let v = _mm256_loadu_si256(g.as_ptr().cast());
            let native = _mm256_or_si256(_mm256_slli_epi16(v, 8), _mm256_srli_epi16(v, 8));
            let s = _mm256_xor_si256(native, flip);
            plain = _mm256_add_epi32(plain, _mm256_madd_epi16(s, ones));
            weighted = _mm256_add_epi32(weighted, _mm256_madd_epi16(s, weights));
            weights = _mm256_sub_epi16(weights, sixteen);
        }
        let plain = _mm_add_epi32(
            _mm256_castsi256_si128(plain),
            _mm256_extracti128_si256::<1>(plain),
        );
        let weighted = _mm_add_epi32(
            _mm256_castsi256_si128(weighted),
            _mm256_extracti128_si256::<1>(weighted),
        );
        (groups * 16, hsum128(plain), hsum128(weighted))
    }
}

/// The vector halves of shuffle/unshuffle: each handles whole groups of 16
/// elements from the start and returns how many elements it did.
mod vector {
    #[cfg(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    use super::arch::{load, store, unzip_even, unzip_odd, zip_hi, zip_lo, V};

    #[cfg(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    pub fn shuffle(esize: usize, data: &[u8], out: &mut [u8], n: usize) -> usize {
        // AVX2, where the CPU has it, takes 32 elements at a time first
        #[cfg(target_arch = "x86_64")]
        let from = super::avx2::shuffle(esize, data, out, n) / 16;
        #[cfg(not(target_arch = "x86_64"))]
        let from = 0;
        match esize {
            2 => shuffle_e::<2>(data, out, n, from),
            4 => shuffle_e::<4>(data, out, n, from),
            8 => shuffle_e::<8>(data, out, n, from),
            16 => shuffle_e::<16>(data, out, n, from),
            _ => 0,
        }
    }

    #[cfg(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    pub fn unshuffle(esize: usize, data: &[u8], out: &mut [u8], n: usize) -> usize {
        // AVX2, where the CPU has it, takes 32 elements at a time first
        #[cfg(target_arch = "x86_64")]
        let from = super::avx2::unshuffle(esize, data, out, n) / 16;
        #[cfg(not(target_arch = "x86_64"))]
        let from = 0;
        match esize {
            2 => unshuffle_e::<2>(data, out, n, from),
            4 => unshuffle_e::<4>(data, out, n, from),
            8 => unshuffle_e::<8>(data, out, n, from),
            16 => unshuffle_e::<16>(data, out, n, from),
            _ => 0,
        }
    }

    /// The 128-bit kernels alone (no AVX2 first), for calibration.
    #[cfg(target_arch = "x86_64")]
    pub fn sse(forward: bool, esize: usize, data: &[u8], out: &mut [u8], n: usize) -> usize {
        match (forward, esize) {
            (true, 2) => shuffle_e::<2>(data, out, n, 0),
            (true, 4) => shuffle_e::<4>(data, out, n, 0),
            (true, 8) => shuffle_e::<8>(data, out, n, 0),
            (true, 16) => shuffle_e::<16>(data, out, n, 0),
            (false, 2) => unshuffle_e::<2>(data, out, n, 0),
            (false, 4) => unshuffle_e::<4>(data, out, n, 0),
            (false, 8) => unshuffle_e::<8>(data, out, n, 0),
            (false, 16) => unshuffle_e::<16>(data, out, n, 0),
            _ => 0,
        }
    }

    /// 16 elements (`E` vectors of input) at a time: `log2(E)` unzip rounds
    /// leave vector `j` holding byte `j` of all 16.
    #[cfg(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    #[inline(always)]
    fn shuffle_e<const E: usize>(data: &[u8], out: &mut [u8], n: usize, from: usize) -> usize {
        let groups = n / 16;
        for g in from..groups {
            let src = &data[g * 16 * E..(g + 1) * 16 * E];
            // SAFETY: each load reads 16 bytes inside `src`
            let mut v: [V; E] = std::array::from_fn(|k| unsafe { load(&src[k * 16..]) });
            let mut round = E;
            while round > 1 {
                let w: [V; E] = std::array::from_fn(|k| {
                    if k < E / 2 {
                        unzip_even(v[2 * k], v[2 * k + 1])
                    } else {
                        let k = k - E / 2;
                        unzip_odd(v[2 * k], v[2 * k + 1])
                    }
                });
                v = w;
                round /= 2;
            }
            for (j, vj) in v.iter().enumerate() {
                let at = j * n + g * 16;
                // SAFETY: plane `j` has `n` bytes, and `g * 16 + 16 <= n`
                unsafe { store(&mut out[at..at + 16], *vj) };
            }
        }
        groups * 16
    }

    /// Inverse of [`shuffle_e`]: `log2(E)` zip rounds over one vector from
    /// each plane leave the 16 elements in order.
    #[cfg(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    #[inline(always)]
    fn unshuffle_e<const E: usize>(data: &[u8], out: &mut [u8], n: usize, from: usize) -> usize {
        let groups = n / 16;
        for g in from..groups {
            // SAFETY: plane `j` has `n` bytes, and `g * 16 + 16 <= n`
            let mut v: [V; E] = std::array::from_fn(|j| unsafe { load(&data[j * n + g * 16..]) });
            let mut round = E;
            while round > 1 {
                let w: [V; E] = std::array::from_fn(|k| {
                    let (a, b) = (v[k / 2], v[k / 2 + E / 2]);
                    if k % 2 == 0 {
                        zip_lo(a, b)
                    } else {
                        zip_hi(a, b)
                    }
                });
                v = w;
                round /= 2;
            }
            let dst = &mut out[g * 16 * E..(g + 1) * 16 * E];
            for (k, vk) in v.iter().enumerate() {
                // SAFETY: each store writes 16 bytes inside `dst`
                unsafe { store(&mut dst[k * 16..k * 16 + 16], *vk) };
            }
        }
        groups * 16
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    pub fn shuffle(_: usize, _: &[u8], _: &mut [u8], _: usize) -> usize {
        0
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    pub fn unshuffle(_: usize, _: &[u8], _: &mut [u8], _: usize) -> usize {
        0
    }
}

/// AVX2 (detected at run time): the same zip/unzip recursion on 32-byte
/// vectors. AVX2's byte unpack/pack work within each 128-bit lane, so the
/// two lanes carry two independent 16-element groups, and a lane exchange on
/// the way in or out puts each group's bytes back together.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// Tests turn AVX2 off to time and check the SSE2 path on AVX2 machines.
    #[cfg(test)]
    pub static DISABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    /// Tests force AVX2 on, past the calibration, to check it everywhere.
    #[cfg(test)]
    pub static FORCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    /// The calibrated choice, `[unshuffle, shuffle][2, 4, 8, 16 bytes]`.
    #[cfg(test)]
    pub fn choice() -> Option<[[bool; 4]; 2]> {
        std::is_x86_feature_detected!("avx2").then(calibrate)
    }

    /// Whether to use AVX2 for this direction and element size. AVX2 is not
    /// a sure win: AMD Zen 4 runs these kernels up to ~1.9× faster with it,
    /// while Intel Alder Lake, which issues 128-bit byte shuffles on two
    /// ports but 256-bit ones on one, runs unshuffle ~30% *slower*. So the
    /// first call times both on a chunk-sized buffer and keeps AVX2 only
    /// where it wins clearly (once per process, a few milliseconds).
    fn available(forward: bool, esize: usize) -> bool {
        #[cfg(test)]
        if DISABLED.load(std::sync::atomic::Ordering::Relaxed) {
            return false;
        }
        #[cfg(test)]
        if FORCED.load(std::sync::atomic::Ordering::Relaxed) {
            return std::is_x86_feature_detected!("avx2") && [2, 4, 8, 16].contains(&esize);
        }
        static CHOICE: std::sync::OnceLock<[[bool; 4]; 2]> = std::sync::OnceLock::new();
        let slot = match esize {
            2 => 0,
            4 => 1,
            8 => 2,
            16 => 3,
            _ => return false,
        };
        std::is_x86_feature_detected!("avx2")
            && CHOICE.get_or_init(calibrate)[forward as usize][slot]
    }

    fn calibrate() -> [[bool; 4]; 2] {
        // chunk-sized: a typical chunk (and its output) overflows L2, and
        // which kernel wins depends on it (Alder Lake: AVX2 in L1, SSE2 past)
        let data: Vec<u8> = (0..1u32 << 20)
            .map(|i| i.wrapping_mul(2_654_435_761) as u8)
            .collect();
        let mut out = vec![0u8; data.len()];
        let mut choice = [[false; 4]; 2];
        for (forward, row) in [false, true].into_iter().zip(&mut choice) {
            for (slot, esize) in [2usize, 4, 8, 16].into_iter().enumerate() {
                let n = data.len() / esize;
                let mut best = [f64::MAX; 2];
                // alternate, best of several: robust to a stray interruption
                for _ in 0..4 {
                    for (k, b) in best.iter_mut().enumerate() {
                        let t = std::time::Instant::now();
                        for _ in 0..1 {
                            if k == 0 {
                                super::vector::sse(forward, esize, &data, &mut out, n);
                            } else if forward {
                                // SAFETY: AVX2 checked by the caller
                                unsafe { dispatch(true, esize, &data, &mut out, n) };
                            } else {
                                unsafe { dispatch(false, esize, &data, &mut out, n) };
                            }
                        }
                        *b = b.min(t.elapsed().as_secs_f64());
                    }
                }
                // only a clear win: on hybrid CPUs (P- and E-cores) a close
                // call measured on one core can reverse on the other
                row[slot] = best[1] * 1.1 < best[0];
            }
        }
        choice
    }

    /// # Safety
    /// AVX2 must be present.
    unsafe fn dispatch(
        forward: bool,
        esize: usize,
        data: &[u8],
        out: &mut [u8],
        n: usize,
    ) -> usize {
        match (forward, esize) {
            (true, 2) => shuffle_e::<2>(data, out, n),
            (true, 4) => shuffle_e::<4>(data, out, n),
            (true, 8) => shuffle_e::<8>(data, out, n),
            (true, 16) => shuffle_e::<16>(data, out, n),
            (false, 2) => unshuffle_e::<2>(data, out, n),
            (false, 4) => unshuffle_e::<4>(data, out, n),
            (false, 8) => unshuffle_e::<8>(data, out, n),
            (false, 16) => unshuffle_e::<16>(data, out, n),
            _ => 0,
        }
    }

    /// Elements done, from the start (a multiple of 32).
    pub fn shuffle(esize: usize, data: &[u8], out: &mut [u8], n: usize) -> usize {
        if !available(true, esize) {
            return 0;
        }
        // SAFETY: AVX2 is present
        unsafe { dispatch(true, esize, data, out, n) }
    }

    /// Elements done, from the start (a multiple of 32).
    pub fn unshuffle(esize: usize, data: &[u8], out: &mut [u8], n: usize) -> usize {
        if !available(false, esize) {
            return 0;
        }
        // SAFETY: AVX2 is present
        unsafe { dispatch(false, esize, data, out, n) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn unzip_even(a: __m256i, b: __m256i) -> __m256i {
        let m = _mm256_set1_epi16(0x00ff);
        _mm256_packus_epi16(_mm256_and_si256(a, m), _mm256_and_si256(b, m))
    }

    #[target_feature(enable = "avx2")]
    unsafe fn unzip_odd(a: __m256i, b: __m256i) -> __m256i {
        _mm256_packus_epi16(_mm256_srli_epi16(a, 8), _mm256_srli_epi16(b, 8))
    }

    /// 32 elements (two groups of 16, one per lane) at a time.
    #[target_feature(enable = "avx2")]
    unsafe fn shuffle_e<const E: usize>(data: &[u8], out: &mut [u8], n: usize) -> usize {
        let pairs = n / 32;
        let mut v = [_mm256_setzero_si256(); E];
        let mut w = [_mm256_setzero_si256(); E];
        for g in 0..pairs {
            let src = &data[g * 32 * E..(g + 1) * 32 * E];
            // lane 0: the first group's k-th vector, lane 1: the second's
            for (k, vk) in v.iter_mut().enumerate() {
                let lo = _mm_loadu_si128(src[k * 16..].as_ptr().cast());
                let hi = _mm_loadu_si128(src[16 * E + k * 16..].as_ptr().cast());
                *vk = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi);
            }
            let mut round = E;
            while round > 1 {
                for k in 0..E / 2 {
                    w[k] = unzip_even(v[2 * k], v[2 * k + 1]);
                    w[k + E / 2] = unzip_odd(v[2 * k], v[2 * k + 1]);
                }
                v = w;
                round /= 2;
            }
            // vector j: byte j of all 32 elements, in order
            for (j, vj) in v.iter().enumerate() {
                let at = j * n + g * 32;
                _mm256_storeu_si256(out[at..at + 32].as_mut_ptr().cast(), *vj);
            }
        }
        pairs * 32
    }

    /// Inverse of [`shuffle_e`].
    #[target_feature(enable = "avx2")]
    unsafe fn unshuffle_e<const E: usize>(data: &[u8], out: &mut [u8], n: usize) -> usize {
        let pairs = n / 32;
        let mut v = [_mm256_setzero_si256(); E];
        let mut w = [_mm256_setzero_si256(); E];
        for g in 0..pairs {
            for (j, vj) in v.iter_mut().enumerate() {
                let at = j * n + g * 32;
                *vj = _mm256_loadu_si256(data[at..at + 32].as_ptr().cast());
            }
            let mut round = E;
            while round > 1 {
                for k in 0..E {
                    let (a, b) = (v[k / 2], v[k / 2 + E / 2]);
                    w[k] = if k % 2 == 0 {
                        _mm256_unpacklo_epi8(a, b)
                    } else {
                        _mm256_unpackhi_epi8(a, b)
                    };
                }
                v = w;
                round /= 2;
            }
            // lane 0 of each vector belongs to the first group, lane 1 to the
            // second: pair vectors up to store each group's 32-byte runs
            let dst = &mut out[g * 32 * E..(g + 1) * 32 * E];
            for m in 0..E / 2 {
                let (x, y) = (v[2 * m], v[2 * m + 1]);
                let first = _mm256_permute2x128_si256::<0x20>(x, y);
                let second = _mm256_permute2x128_si256::<0x31>(x, y);
                _mm256_storeu_si256(dst[32 * m..].as_mut_ptr().cast(), first);
                _mm256_storeu_si256(dst[16 * E + 32 * m..].as_mut_ptr().cast(), second);
            }
        }
        pairs * 32
    }
}

/// The per-target primitives on 16-byte vectors. Loads/stores are unaligned;
/// callers guarantee 16 readable/writable bytes.
#[cfg(target_arch = "aarch64")]
mod arch {
    use std::arch::aarch64::*;
    pub type V = uint8x16_t;

    #[inline(always)]
    pub unsafe fn load(s: &[u8]) -> V {
        vld1q_u8(s.as_ptr())
    }
    #[inline(always)]
    pub unsafe fn store(d: &mut [u8], v: V) {
        vst1q_u8(d.as_mut_ptr(), v)
    }
    #[inline(always)]
    pub fn zip_lo(a: V, b: V) -> V {
        unsafe { vzip1q_u8(a, b) }
    }
    #[inline(always)]
    pub fn zip_hi(a: V, b: V) -> V {
        unsafe { vzip2q_u8(a, b) }
    }
    #[inline(always)]
    pub fn unzip_even(a: V, b: V) -> V {
        unsafe { vuzp1q_u8(a, b) }
    }
    #[inline(always)]
    pub fn unzip_odd(a: V, b: V) -> V {
        unsafe { vuzp2q_u8(a, b) }
    }
}

#[cfg(target_arch = "x86_64")]
mod arch {
    use std::arch::x86_64::*;
    pub type V = __m128i;

    #[inline(always)]
    pub unsafe fn load(s: &[u8]) -> V {
        _mm_loadu_si128(s.as_ptr().cast())
    }
    #[inline(always)]
    pub unsafe fn store(d: &mut [u8], v: V) {
        _mm_storeu_si128(d.as_mut_ptr().cast(), v)
    }
    #[inline(always)]
    pub fn zip_lo(a: V, b: V) -> V {
        unsafe { _mm_unpacklo_epi8(a, b) }
    }
    #[inline(always)]
    pub fn zip_hi(a: V, b: V) -> V {
        unsafe { _mm_unpackhi_epi8(a, b) }
    }
    /// Even bytes: mask each 16-bit lane to its low byte, then pack (no
    /// value exceeds 255, so the saturation never triggers).
    #[inline(always)]
    pub fn unzip_even(a: V, b: V) -> V {
        unsafe {
            let m = _mm_set1_epi16(0x00ff);
            _mm_packus_epi16(_mm_and_si128(a, m), _mm_and_si128(b, m))
        }
    }
    #[inline(always)]
    pub fn unzip_odd(a: V, b: V) -> V {
        unsafe { _mm_packus_epi16(_mm_srli_epi16(a, 8), _mm_srli_epi16(b, 8)) }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod arch {
    use std::arch::wasm32::*;
    pub type V = v128;

    #[inline(always)]
    pub unsafe fn load(s: &[u8]) -> V {
        v128_load(s.as_ptr().cast())
    }
    #[inline(always)]
    pub unsafe fn store(d: &mut [u8], v: V) {
        v128_store(d.as_mut_ptr().cast(), v)
    }
    #[inline(always)]
    pub fn zip_lo(a: V, b: V) -> V {
        u8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(a, b)
    }
    #[inline(always)]
    pub fn zip_hi(a: V, b: V) -> V {
        u8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(a, b)
    }
    #[inline(always)]
    pub fn unzip_even(a: V, b: V) -> V {
        u8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30>(a, b)
    }
    #[inline(always)]
    pub fn unzip_odd(a: V, b: V) -> V {
        u8x16_shuffle::<1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31>(a, b)
    }
}

/// wasm `simd128` kernels for what LLVM leaves scalar there (on NEON/SSE2
/// the plain loops above auto-vectorize).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use std::arch::wasm32::*;

    /// Byte-reverse whole 16-byte groups of `width`-byte elements; returns
    /// the bytes done.
    pub fn swap_bytes(width: usize, buf: &mut [u8]) -> usize {
        fn run<const W: usize>(buf: &mut [u8], f: impl Fn(v128) -> v128) -> usize {
            let done = buf.len() / 16 * 16;
            for g in buf[..done].chunks_exact_mut(16) {
                // SAFETY: `g` is 16 bytes
                unsafe { v128_store(g.as_mut_ptr().cast(), f(v128_load(g.as_ptr().cast()))) }
            }
            done
        }
        match width {
            2 => run::<2>(buf, |v| {
                u8x16_shuffle::<1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14>(v, v)
            }),
            4 => run::<4>(buf, |v| {
                u8x16_shuffle::<3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12>(v, v)
            }),
            8 => run::<8>(buf, |v| {
                u8x16_shuffle::<7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8>(v, v)
            }),
            16 => run::<16>(buf, |v| {
                u8x16_shuffle::<15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0>(v, v)
            }),
            _ => 0,
        }
    }

    /// One fletcher32 block's `(words done, Σ wᵢ, Σ (k − i)·wᵢ)`, 8 words
    /// (one vector) at a time.
    pub fn fletcher_block(block: &[u8], k: u32) -> (usize, u32, u32) {
        let groups = block.len() / 16;
        let (mut plain, mut weighted) = (u32x4_splat(0), u32x4_splat(0));
        let k = k as i32;
        let mut w_lo = i32x4(k, k - 1, k - 2, k - 3);
        let mut w_hi = i32x4(k - 4, k - 5, k - 6, k - 7);
        let eight = u32x4_splat(8);
        for g in block[..groups * 16].chunks_exact(16) {
            // SAFETY: `g` is 16 bytes
            let v = unsafe { v128_load(g.as_ptr().cast()) };
            // big-endian words → native u16 lanes → u32 lanes
            let v = u8x16_shuffle::<1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14>(v, v);
            let (lo, hi) = (u32x4_extend_low_u16x8(v), u32x4_extend_high_u16x8(v));
            plain = u32x4_add(plain, u32x4_add(lo, hi));
            weighted = u32x4_add(
                weighted,
                u32x4_add(i32x4_mul(lo, w_lo), i32x4_mul(hi, w_hi)),
            );
            w_lo = u32x4_sub(w_lo, eight);
            w_hi = u32x4_sub(w_hi, eight);
        }
        let sum = |v: v128| {
            (0..4).fold(0u32, |a, i| {
                a.wrapping_add(match i {
                    0 => u32x4_extract_lane::<0>(v),
                    1 => u32x4_extract_lane::<1>(v),
                    2 => u32x4_extract_lane::<2>(v),
                    _ => u32x4_extract_lane::<3>(v),
                })
            })
        };
        (groups * 8, sum(plain), sum(weighted))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(n: usize) -> Vec<u8> {
        let mut x = 0x2545_f491u32;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                x as u8
            })
            .collect()
    }

    fn shuffle_ref(e: usize, data: &[u8]) -> Vec<u8> {
        let n = data.len() / e;
        let mut out = data.to_vec();
        for j in 0..e {
            for i in 0..n {
                out[j * n + i] = data[i * e + j];
            }
        }
        out
    }

    /// The original serial loop from `H5_checksum_fletcher32`.
    fn fletcher32_ref(data: &[u8]) -> u32 {
        let (mut sum1, mut sum2) = (0u32, 0u32);
        let mut len = data.len() / 2;
        let mut pos = 0;
        while len > 0 {
            let mut tlen = len.min(360);
            len -= tlen;
            while tlen > 0 {
                let word = (u32::from(data[pos]) << 8) | u32::from(data[pos + 1]);
                sum1 = sum1.wrapping_add(word);
                sum2 = sum2.wrapping_add(sum1);
                pos += 2;
                tlen -= 1;
            }
            sum1 = (sum1 & 0xffff).wrapping_add(sum1 >> 16);
            sum2 = (sum2 & 0xffff).wrapping_add(sum2 >> 16);
        }
        if data.len() % 2 != 0 {
            sum1 = sum1.wrapping_add(u32::from(data[data.len() - 1]) << 8);
            sum2 = sum2.wrapping_add(sum1);
            sum1 = (sum1 & 0xffff).wrapping_add(sum1 >> 16);
            sum2 = (sum2 & 0xffff).wrapping_add(sum2 >> 16);
        }
        sum1 = (sum1 & 0xffff).wrapping_add(sum1 >> 16);
        sum2 = (sum2 & 0xffff).wrapping_add(sum2 >> 16);
        (sum2 << 16) | sum1
    }

    #[test]
    fn shuffle_matches_scalar() {
        shuffle_matches_scalar_once();
        // where there is AVX2: both paths, whatever the calibration picks
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            use std::sync::atomic::Ordering::Relaxed;
            super::avx2::FORCED.store(true, Relaxed);
            shuffle_matches_scalar_once();
            super::avx2::FORCED.store(false, Relaxed);
            super::avx2::DISABLED.store(true, Relaxed);
            shuffle_matches_scalar_once();
            super::avx2::DISABLED.store(false, Relaxed);
        }
    }

    fn shuffle_matches_scalar_once() {
        for e in [1, 2, 3, 4, 5, 8, 12, 16] {
            for len in [0, 1, 7, 31, 32, 33, 255, 256, 257, 4096 + 3, 100_003] {
                let data = bytes(len);
                let mut out = vec![0; len];
                shuffle(e, &data, &mut out);
                assert_eq!(out, shuffle_ref(e, &data), "shuffle e={e} len={len}");
                let mut back = vec![0; len];
                unshuffle(e, &out, &mut back);
                assert_eq!(back, data, "unshuffle e={e} len={len}");
            }
        }
    }

    #[test]
    fn fletcher32_matches_reference() {
        for len in [0, 1, 2, 3, 719, 720, 721, 1440, 5000, 100_001] {
            let data = bytes(len);
            assert_eq!(fletcher32(&data), fletcher32_ref(&data), "len {len}");
        }
        // all-ones words: the largest sums the folding has to absorb
        for len in [720, 721, 72_000, 72_001] {
            let data = vec![0xffu8; len];
            assert_eq!(fletcher32(&data), fletcher32_ref(&data), "0xff len {len}");
        }
    }

    #[test]
    fn swap_bytes_reverses() {
        for w in [2, 3, 4, 8, 16] {
            let data = bytes(w * 37 + 1);
            let mut got = data.clone();
            swap_bytes(w, &mut got);
            let mut want = data.clone();
            want[..w * 37].chunks_exact_mut(w).for_each(<[u8]>::reverse);
            assert_eq!(got, want, "width {w}");
        }
    }

    /// `cargo test --release -p hdf5-no-c simd::tests::throughput -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn throughput() {
        let data = bytes(64 << 20);
        let mut out = vec![0; data.len()];
        let time = |name: &str, f: &mut dyn FnMut()| {
            let t = std::time::Instant::now();
            for _ in 0..5 {
                f();
            }
            let s = t.elapsed().as_secs_f64() / 5.0;
            println!("{name:14} {:7.2} GB/s", data.len() as f64 / s / 1e9);
        };
        for e in [2, 4, 8] {
            time(&format!("shuffle {e}"), &mut || shuffle(e, &data, &mut out));
            time(&format!("unshuffle {e}"), &mut || {
                unshuffle(e, &data, &mut out)
            });
        }
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            super::avx2::DISABLED.store(true, std::sync::atomic::Ordering::Relaxed);
            for e in [2, 4, 8] {
                time(&format!("shuffle {e} sse2"), &mut || {
                    shuffle(e, &data, &mut out)
                });
                time(&format!("unshuffle {e} sse2"), &mut || {
                    unshuffle(e, &data, &mut out)
                });
            }
            super::avx2::DISABLED.store(false, std::sync::atomic::Ordering::Relaxed);
        }
        time("shuffle 4 ref", &mut || {
            std::hint::black_box(shuffle_ref(4, &data));
        });
        time("fletcher32", &mut || {
            std::hint::black_box(fletcher32(&data));
        });
        time("fletcher ref", &mut || {
            std::hint::black_box(fletcher32_ref(&data));
        });
        time("swap 8", &mut || swap_bytes(8, &mut out));

        // a typical chunk (1 MiB), in cache: compute, not memory, decides
        let chunk = bytes(1 << 20);
        let mut cout = vec![0; chunk.len()];
        let time_chunk = |name: &str, f: &mut dyn FnMut()| {
            let t = std::time::Instant::now();
            for _ in 0..500 {
                f();
            }
            let s = t.elapsed().as_secs_f64() / 500.0;
            println!("{name:20} {:7.2} GB/s", chunk.len() as f64 / s / 1e9);
        };
        for e in [2, 4, 8] {
            time_chunk(&format!("1MiB shuffle {e}"), &mut || {
                shuffle(e, &chunk, &mut cout)
            });
            time_chunk(&format!("1MiB unshuffle {e}"), &mut || {
                unshuffle(e, &chunk, &mut cout)
            });
        }
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            println!(
                "calibrated AVX2 use [unshuffle, shuffle][2,4,8,16 B]: {:?}",
                super::avx2::choice()
            );
            super::avx2::FORCED.store(true, std::sync::atomic::Ordering::Relaxed);
            for e in [2, 4, 8] {
                time_chunk(&format!("1MiB shuffle {e} avx2"), &mut || {
                    shuffle(e, &chunk, &mut cout)
                });
                time_chunk(&format!("1MiB unshuffle {e} avx2"), &mut || {
                    unshuffle(e, &chunk, &mut cout)
                });
            }
            super::avx2::FORCED.store(false, std::sync::atomic::Ordering::Relaxed);
            super::avx2::DISABLED.store(true, std::sync::atomic::Ordering::Relaxed);
            for e in [2, 4, 8] {
                time_chunk(&format!("1MiB shuffle {e} sse2"), &mut || {
                    shuffle(e, &chunk, &mut cout)
                });
                time_chunk(&format!("1MiB unshuffle {e} sse2"), &mut || {
                    unshuffle(e, &chunk, &mut cout)
                });
            }
            super::avx2::DISABLED.store(false, std::sync::atomic::Ordering::Relaxed);
        }
        time_chunk("1MiB fletcher32", &mut || {
            std::hint::black_box(fletcher32(&chunk));
        });
        time_chunk("1MiB swap 8", &mut || swap_bytes(8, &mut cout));
    }
}
