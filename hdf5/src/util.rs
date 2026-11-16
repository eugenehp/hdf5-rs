//! Small string utilities (API-compatible subset of the FFI crate's `util`).
#![allow(dead_code)]

use crate::error::Result;

/// Convert a fixed-size zero-padded byte buffer into a `String`.
pub fn string_from_fixed_bytes(bytes: &[u8], len: usize) -> String {
    let end = bytes[..len.min(bytes.len())]
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(len.min(bytes.len()));
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// Copy a string into a fixed-size zero-padded byte buffer.
pub fn string_to_fixed_bytes(s: &str, buf: &mut [u8]) {
    let n = s.len().min(buf.len());
    buf[..n].copy_from_slice(&s.as_bytes()[..n]);
    for b in &mut buf[n..] {
        *b = 0;
    }
}

/// Validate that a string contains no interior NUL bytes.
pub fn to_cstring(s: &str) -> Result<std::ffi::CString> {
    std::ffi::CString::new(s).map_err(Into::into)
}

/// Hashes file addresses (and other plain integers) with one multiply: the
/// keys are not attacker-chosen in a way that matters for a parser that
/// already trusts the file's layout, and `SipHash` showed up in profiles of
/// opening many-object files.
#[derive(Default, Clone, Copy)]
pub struct AddrHasher(u64);

impl std::hash::Hasher for AddrHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(u64::from(b));
        }
    }
    fn write_u64(&mut self, n: u64) {
        // Fibonacci hashing; rotate so the high (well-mixed) bits reach the
        // low bits hashbrown buckets by, while its tag uses the top bits
        self.0 = (self.0 ^ n)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .rotate_left(26);
    }
    fn write_u32(&mut self, n: u32) {
        self.write_u64(u64::from(n));
    }
    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
}

/// A map keyed by file address (or another integer), hashed by [`AddrHasher`].
pub type AddrMap<K, V> = std::collections::HashMap<K, V, std::hash::BuildHasherDefault<AddrHasher>>;
