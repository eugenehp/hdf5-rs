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
