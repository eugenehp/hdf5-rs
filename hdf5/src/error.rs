//! Error handling for the pure-Rust HDF5 implementation.
//!
//! The public surface mirrors the FFI-based `hdf5` crate (`Error`, `Result`,
//! `ErrorStack`, `ErrorFrame`, `ExpandedErrorStack`, `silence_errors`) so that
//! downstream code keeps compiling, but the implementation is entirely native
//! Rust: there is no C library error stack to unwind, so errors are carried as
//! descriptive messages.

use std::error::Error as StdError;
use std::fmt::{self, Debug, Display};

/// The error type for HDF5 operations.
#[derive(Clone, PartialEq, Eq)]
pub enum Error {
    /// An error originating in the pure-Rust HDF5 engine, with a message.
    Internal(String),
    /// A structured error stack (kept for API compatibility).
    HDF5(ErrorStack),
}

impl Error {
    pub fn query() -> Option<Self> {
        None
    }

    #[doc(hidden)]
    pub fn description(&self) -> &str {
        match self {
            Self::Internal(msg) => msg.as_str(),
            Self::HDF5(stack) => stack.description(),
        }
    }
}

impl From<&str> for Error {
    fn from(desc: &str) -> Self {
        Self::Internal(desc.into())
    }
}

impl From<String> for Error {
    fn from(desc: String) -> Self {
        Self::Internal(desc)
    }
}

impl From<&String> for Error {
    fn from(desc: &String) -> Self {
        Self::Internal(desc.clone())
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Self::Internal(format!("I/O error: {err}"))
    }
}

impl From<std::ffi::NulError> for Error {
    fn from(err: std::ffi::NulError) -> Self {
        Self::Internal(format!("null error: {err}"))
    }
}

impl From<std::convert::Infallible> for Error {
    fn from(_: std::convert::Infallible) -> Self {
        unreachable!("Infallible error can never be constructed")
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.description())
    }
}

impl Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl StdError for Error {
    fn description(&self) -> &str {
        Error::description(self)
    }
}

/// The result type for HDF5 operations.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// A single frame of an error stack (kept for API compatibility).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ErrorFrame {
    desc: String,
    func: String,
    major: String,
    minor: String,
}

impl ErrorFrame {
    pub fn new(desc: &str, func: &str, major: &str, minor: &str) -> Self {
        Self {
            desc: desc.into(),
            func: func.into(),
            major: major.into(),
            minor: minor.into(),
        }
    }

    pub fn desc(&self) -> &str {
        &self.desc
    }

    pub fn description(&self) -> &str {
        &self.desc
    }

    pub fn detail(&self) -> Option<String> {
        Some(format!(
            "Error in {}(): {} [{}: {}]",
            self.func, self.desc, self.major, self.minor
        ))
    }
}

/// A structured stack of error frames (kept for API compatibility).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ErrorStack {
    frames: Vec<ErrorFrame>,
    description: Option<String>,
}

impl ErrorStack {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, frame: ErrorFrame) {
        self.frames.push(frame);
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn top(&self) -> Option<&ErrorFrame> {
        self.frames.first()
    }

    pub fn description(&self) -> &str {
        if let Some(ref desc) = self.description {
            desc.as_str()
        } else if let Some(frame) = self.frames.last() {
            frame.description()
        } else {
            "unknown error"
        }
    }

    pub fn expand(&self) -> ExpandedErrorStack {
        ExpandedErrorStack {
            frames: self.frames.clone(),
        }
    }
}

/// An expanded (human-readable) error stack (kept for API compatibility).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ExpandedErrorStack {
    frames: Vec<ErrorFrame>,
}

impl ExpandedErrorStack {
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }
}

impl Display for ExpandedErrorStack {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, frame) in self.frames.iter().enumerate() {
            if let Some(detail) = frame.detail() {
                writeln!(f, "  {i}: {detail}")?;
            }
        }
        Ok(())
    }
}

/// Silences the library's error output.
///
/// In the pure-Rust implementation there is no C-library error printing, so
/// this is a no-op provided for API compatibility.
pub fn silence_errors(_silence: bool) {}

/// Internal helper mirroring the FFI crate's `h5check`.
///
/// In the FFI crate this checked the return value of a C call against the error
/// stack; here it simply forwards `Result` values unchanged.
#[doc(hidden)]
#[inline]
pub fn h5check<T>(value: Result<T>) -> Result<T> {
    value
}

/// Construct an internal error from a formatted message.
macro_rules! fail {
    ($e:expr) => {
        return Err($crate::error::Error::from($e))
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err($crate::error::Error::from(format!($fmt, $($arg)*)))
    };
}

/// Ensure a condition holds or return an internal error.
macro_rules! ensure {
    ($expr:expr, $($arg:tt)*) => {
        if !($expr) {
            fail!($($arg)*);
        }
    };
}

pub(crate) use ensure;
pub(crate) use fail;

/// Values that can carry an HDF5 error code (parity trait).
pub trait H5ErrorCode: Copy {
    fn is_err_code(value: Self) -> bool;
}

impl H5ErrorCode for crate::h5i::hid_t {
    fn is_err_code(value: Self) -> bool {
        value < 0
    }
}

impl H5ErrorCode for libc::c_int {
    fn is_err_code(value: Self) -> bool {
        value < 0
    }
}

/// Returns `true` if `value` represents an HDF5 error code.
pub fn is_err_code<T: H5ErrorCode>(value: T) -> bool {
    T::is_err_code(value)
}
