//! A pure-Rust handle to an HDF5 object.
//!
//! In the FFI crate a `Handle` wraps a C `hid_t` registered in the library.
//! Here it carries a reference-counted payload describing the object: either a
//! locator into an open file's object arena (groups, datasets, attributes,
//! named types) or a transient object (datatype, dataspace, property list).

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use hdf5_types::TypeDescriptor;

use crate::h5i::{hid_t, H5I_type_t, H5I_INVALID_HID};
use crate::hl::dataspace::DataspaceState;
use crate::hl::plist::PlistState;
use crate::model::{FileInner, ObjId};

static NEXT_ID: AtomicI64 = AtomicI64::new(1);

pub(crate) fn next_id() -> hid_t {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// The payload carried by a handle.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Payload {
    Invalid,
    File(Arc<FileInner>),
    Group {
        file: Arc<FileInner>,
        id: ObjId,
    },
    Dataset {
        file: Arc<FileInner>,
        id: ObjId,
    },
    Attribute {
        file: Arc<FileInner>,
        owner: ObjId,
        name: String,
    },
    NamedType {
        file: Arc<FileInner>,
        id: ObjId,
    },
    Datatype(TypeDescriptor),
    Dataspace(DataspaceState),
    PropertyList(PlistState),
}

pub(crate) struct HandleData {
    pub id: hid_t,
    pub payload: Payload,
}

/// A handle to an HDF5 object.
#[derive(Clone)]
pub struct Handle {
    pub(crate) inner: Arc<HandleData>,
}

impl std::fmt::Debug for Handle {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Handle({}, {:?})", self.id(), self.id_type())
    }
}

impl Handle {
    /// Parity with the FFI crate's non-owning borrow. Synthetic identifiers
    /// have no global registry, so this fails like `from_id` does.
    pub fn try_borrow(id: crate::h5i::hid_t) -> crate::error::Result<Self> {
        Err(format!("invalid id: {id} (no id registry in the pure-Rust build)").into())
    }

    pub(crate) fn new(payload: Payload) -> Self {
        Self {
            inner: Arc::new(HandleData {
                id: next_id(),
                payload,
            }),
        }
    }

    pub const fn invalid() -> Self {
        // Note: not const-constructible with Arc; provide a runtime invalid.
        panic!("use Handle::new_invalid()")
    }

    pub(crate) fn new_invalid() -> Self {
        Self::new(Payload::Invalid)
    }

    pub fn id(&self) -> hid_t {
        if matches!(self.inner.payload, Payload::Invalid) {
            H5I_INVALID_HID
        } else {
            self.inner.id
        }
    }

    pub fn incref(&self) {}

    pub fn decref(&self) {}

    pub fn is_valid_user_id(&self) -> bool {
        !matches!(self.inner.payload, Payload::Invalid)
    }

    pub fn is_valid_id(&self) -> bool {
        self.is_valid_user_id()
    }

    pub fn refcount(&self) -> u32 {
        Arc::strong_count(&self.inner) as u32
    }

    pub fn id_type(&self) -> H5I_type_t {
        use H5I_type_t::*;
        match &self.inner.payload {
            Payload::Invalid => H5I_BADID,
            Payload::File(_) => H5I_FILE,
            Payload::Group { .. } => H5I_GROUP,
            Payload::Dataset { .. } => H5I_DATASET,
            Payload::Attribute { .. } => H5I_ATTR,
            Payload::NamedType { .. } | Payload::Datatype(_) => H5I_DATATYPE,
            Payload::Dataspace(_) => H5I_DATASPACE,
            Payload::PropertyList(_) => H5I_GENPROP_LST,
        }
    }

    // --- payload accessors used by the high-level API ---

    pub(crate) fn payload(&self) -> &Payload {
        &self.inner.payload
    }

    /// The backing file, for file-resident objects.
    pub(crate) fn file(&self) -> Option<&Arc<FileInner>> {
        match &self.inner.payload {
            Payload::File(f)
            | Payload::Group { file: f, .. }
            | Payload::Dataset { file: f, .. }
            | Payload::Attribute { file: f, .. }
            | Payload::NamedType { file: f, .. } => Some(f),
            _ => None,
        }
    }

    /// The arena object id, for objects that have one (file root, group,
    /// dataset, named type, or the owner of an attribute).
    pub(crate) fn obj_id(&self) -> Option<ObjId> {
        match &self.inner.payload {
            Payload::File(f) => Some(f.state.read().root),
            Payload::Group { id, .. }
            | Payload::Dataset { id, .. }
            | Payload::NamedType { id, .. } => Some(*id),
            Payload::Attribute { owner, .. } => Some(*owner),
            _ => None,
        }
    }

    pub(crate) fn datatype_desc(&self) -> Option<&TypeDescriptor> {
        match &self.inner.payload {
            Payload::Datatype(d) => Some(d),
            _ => None,
        }
    }

    pub(crate) fn dataspace_state(&self) -> Option<&DataspaceState> {
        match &self.inner.payload {
            Payload::Dataspace(s) => Some(s),
            _ => None,
        }
    }

    pub(crate) fn plist_state(&self) -> Option<&PlistState> {
        match &self.inner.payload {
            Payload::PropertyList(s) => Some(s),
            _ => None,
        }
    }
}
