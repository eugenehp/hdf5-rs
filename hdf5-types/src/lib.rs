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

// Pure-Rust implementation: every variable-length allocation (`hvl_t` data,
// `char*` strings) goes through this one allocator, so allocation and release
// always match. There is no C HDF5 library to interoperate with; the `hvl_t`
// memory layout is preserved for the type system's `#[repr(C)]` types.
//
// Built on `std::alloc` (no libc), so the crate also builds for
// `wasm32-unknown-unknown`. `free` takes no size, like C's, so each block
// carries its size in a header; blocks are aligned to `ALIGN`, as `malloc`'s are.

const ALIGN: usize = 16;
/// Header in front of every block: the requested size, padded to `ALIGN`.
const HEADER: usize = ALIGN;

fn block_layout(n: usize) -> Option<std::alloc::Layout> {
    std::alloc::Layout::from_size_align(n.checked_add(HEADER)?, ALIGN).ok()
}

/// `malloc(n)`: an `ALIGN`-aligned block of `n` bytes, or null when out of memory.
/// Release it with [`free`] only.
#[doc(hidden)]
pub unsafe fn malloc(n: usize) -> *mut core::ffi::c_void {
    let Some(layout) = block_layout(n) else {
        return core::ptr::null_mut();
    };
    // SAFETY: `layout` has a non-zero size (at least `HEADER`).
    let base = unsafe { std::alloc::alloc(layout) };
    if base.is_null() {
        return core::ptr::null_mut();
    }
    // SAFETY: the block holds `HEADER` (≥ size_of::<usize>) bytes before the data.
    unsafe {
        base.cast::<usize>().write(n);
        base.add(HEADER).cast()
    }
}

/// `free(ptr)` for a block from [`malloc`]; null is a no-op.
#[doc(hidden)]
pub unsafe fn free(ptr: *mut core::ffi::c_void) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: `ptr` came from `malloc`, which put the size `HEADER` bytes before it.
    unsafe {
        let base = ptr.cast::<u8>().sub(HEADER);
        let n = base.cast::<usize>().read();
        std::alloc::dealloc(
            base,
            block_layout(n).expect("hdf5_types::free: corrupt block header"),
        );
    }
}

/// `strlen`: bytes before the first NUL.
pub(crate) unsafe fn strlen(ptr: *const u8) -> usize {
    let mut n = 0;
    // SAFETY: the caller guarantees a NUL-terminated buffer.
    while unsafe { *ptr.add(n) } != 0 {
        n += 1;
    }
    n
}

pub const USING_H5_ALLOCATOR: bool = false;

#[cfg(test)]
mod alloc_tests {
    #[test]
    fn malloc_free_round_trip() {
        unsafe {
            for n in [0usize, 1, 7, 15, 16, 17, 255, 4096, 1 << 20] {
                let p = super::malloc(n).cast::<u8>();
                assert!(!p.is_null(), "n={n}");
                assert_eq!(p as usize % super::ALIGN, 0, "n={n} misaligned");
                for i in 0..n {
                    p.add(i).write(i as u8);
                }
                for i in 0..n {
                    assert_eq!(*p.add(i), i as u8);
                }
                super::free(p.cast());
            }
            super::free(core::ptr::null_mut());
            assert!(super::malloc(usize::MAX).is_null());
            assert_eq!(super::strlen(b"abc\0def\0".as_ptr()), 3);
            assert_eq!(super::strlen(b"\0".as_ptr()), 0);
        }
    }
}
