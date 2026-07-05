#![recursion_limit = "1024"]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::redundant_pub_crate)]
#![allow(clippy::must_use_candidate)]

//! Types that can be stored and retrieved from a `HDF5` dataset
//!
//! Crate features:
//! * `h5-alloc`: Use the `hdf5` allocator for varlen types and dynamic values.
//!   This is necessary on platforms which uses different allocators
//!   in different libraries (e.g. dynamic libraries on windows),
//!   or if `hdf5-c` is compiled with the MEMCHECKER option.
//!   This option is forced on in the case of using a `windows` DLL.

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

mod array;
pub mod dyn_value;
mod h5type;
mod string;

#[cfg(feature = "complex")]
mod complex;

pub use self::array::VarLenArray;
pub use self::dyn_value::{DynValue, OwnedDynValue};
pub use self::h5type::{
    CompoundField, CompoundType, EnumMember, EnumType, FloatSize, H5Type, IntSize, TypeDescriptor,
};
pub use self::string::{FixedAscii, FixedUnicode, StringError, VarLenAscii, VarLenUnicode};

// Pure-Rust implementation: variable-length allocations always go through libc's
// allocator. There is no C HDF5 library to interoperate with, so a single
// consistent allocator is used for all `hvl_t`-style data. The `hvl_t` memory
// layout is preserved for compatibility with the type system's `#[repr(C)]` types.
pub(crate) unsafe fn malloc(n: usize) -> *mut core::ffi::c_void {
    libc::malloc(n)
}

pub(crate) unsafe fn free(ptr: *mut core::ffi::c_void) {
    libc::free(ptr);
}

pub const USING_H5_ALLOCATOR: bool = false;
