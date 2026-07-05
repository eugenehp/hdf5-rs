//! Object identifier types.
//!
//! Pure-Rust analogues of the `hdf5-sys` `h5i` identifiers used throughout the
//! public API (`Object::id`, `Object::id_type`, ...).

/// An HDF5 object identifier.
///
/// In the FFI crate this is the C library `hid_t`. Here it is a synthetic,
/// process-unique integer used only for identity/debugging.
#[allow(non_camel_case_types)]
pub type hid_t = i64;

/// Sentinel value for an invalid identifier.
pub const H5I_INVALID_HID: hid_t = -1;

/// The type of an HDF5 object identifier.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum H5I_type_t {
    H5I_UNINIT = -2,
    H5I_BADID = -1,
    H5I_FILE = 1,
    H5I_GROUP,
    H5I_DATATYPE,
    H5I_DATASPACE,
    H5I_DATASET,
    H5I_MAP,
    H5I_ATTR,
    H5I_VFL,
    H5I_VOL,
    H5I_GENPROP_CLS,
    H5I_GENPROP_LST,
    H5I_ERROR_CLASS,
    H5I_ERROR_MSG,
    H5I_ERROR_STACK,
    H5I_NTYPES,
}

pub use self::H5I_type_t::*;
