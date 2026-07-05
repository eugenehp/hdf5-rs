//! The `ObjectClass` trait: common machinery for typed wrappers over handles.
//!
//! Mirrors the FFI crate: every high-level type is a `#[repr(transparent)]`
//! newtype over [`Handle`], and `Deref` between them is implemented via
//! transmutation, which is sound because of the shared representation.

use std::fmt;
use std::mem;
use std::ptr::{self, addr_of};

use crate::error::Result;
use crate::h5i::{hid_t, H5I_type_t};
use crate::handle::Handle;

pub trait ObjectClass: Sized {
    const NAME: &'static str;
    const VALID_TYPES: &'static [H5I_type_t];

    fn from_handle(handle: Handle) -> Self;

    fn handle(&self) -> &Handle;

    fn short_repr(&self) -> Option<String> {
        None
    }

    fn validate(&self) -> Result<()> {
        Ok(())
    }

    fn from_id(id: hid_t) -> Result<Self> {
        // Synthetic ids cannot be looked up in a global registry in the
        // pure-Rust implementation; constructing an object from a raw id is
        // only supported for invalid ids (matching FFI-crate error behavior).
        Err(format!("Invalid {} id: {}", Self::NAME, id).into())
    }

    fn invalid() -> Self {
        Self::from_handle(Handle::new_invalid())
    }

    fn is_valid_id_type(tp: H5I_type_t) -> bool {
        Self::VALID_TYPES.is_empty() || Self::VALID_TYPES.contains(&tp)
    }

    unsafe fn transmute<T: ObjectClass>(&self) -> &T {
        &*(self as *const Self).cast::<T>()
    }

    unsafe fn transmute_mut<T: ObjectClass>(&mut self) -> &mut T {
        &mut *(self as *mut Self).cast::<T>()
    }

    unsafe fn cast_unchecked<T: ObjectClass>(self) -> T {
        let obj = ptr::read(addr_of!(self).cast());
        mem::forget(self);
        obj
    }

    fn cast<T: ObjectClass>(self) -> Result<T> {
        let id_type = self.handle().id_type();
        if T::is_valid_id_type(id_type) {
            Ok(unsafe { self.cast_unchecked() })
        } else {
            Err(format!(
                "unable to cast {} ({:?}) into {}",
                Self::NAME,
                id_type,
                T::NAME
            )
            .into())
        }
    }

    fn debug_fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if !self.handle().is_valid_user_id() {
            write!(f, "<HDF5 {}: invalid id>", Self::NAME)
        } else if let Some(d) = self.short_repr() {
            write!(f, "<HDF5 {}: {}>", Self::NAME, d)
        } else {
            write!(f, "<HDF5 {}>", Self::NAME)
        }
    }
}

/// Convert a raw id into a typed object (kept for API compatibility; always
/// errors in the pure-Rust implementation since raw ids are synthetic).
pub unsafe fn from_id<T: ObjectClass>(id: hid_t) -> Result<T> {
    T::from_id(id)
}
