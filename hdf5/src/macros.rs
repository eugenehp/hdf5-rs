//! Public macros kept for source compatibility with the FFI crate.
//!
//! In the pure-Rust implementation there is no global library lock and no
//! C error stack, so these are thin pass-throughs.

/// Run an expression (formerly: while holding the global HDF5 library lock).
#[macro_export]
macro_rules! h5lock {
    ($expr:expr) => {{
        $expr
    }};
}

/// Evaluate an expression yielding a `Result` (formerly: check a C call
/// against the HDF5 error stack).
#[macro_export]
macro_rules! h5call {
    ($expr:expr) => {{
        $crate::Result::from(Ok($expr))
    }};
}

/// Like `h5call!` but unwraps with `?`.
#[macro_export]
macro_rules! h5try {
    ($expr:expr) => {
        match h5call!($expr) {
            Ok(v) => v,
            Err(e) => return Err(e),
        }
    };
}

/// Parity trait: types retrievable from a property-list getter by value.
pub trait H5Get: Copy + Default {}

impl<T: Copy + Default> H5Get for T {}
