//! Property lists: containers of settings controlling HDF5 operations.

pub mod common;
pub mod dataset_access;
pub mod dataset_create;
pub mod file_access;
pub mod file_create;
pub mod link_create;

use std::fmt::{self, Debug, Display};
use std::ops::Deref;
use std::str::FromStr;

use crate::class::ObjectClass;
use crate::error::Result;
use crate::h5i::H5I_type_t;
use crate::handle::{Handle, Payload};
use crate::hl::object::Object;

/// The payload of a property-list handle.
#[derive(Clone, Debug)]
pub(crate) enum PlistState {
    FileCreate(file_create::FileCreateData),
    FileAccess(file_access::FileAccessData),
    DatasetCreate(dataset_create::DatasetCreateData),
    DatasetAccess(dataset_access::DatasetAccessData),
    LinkCreate(link_create::LinkCreateData),
}

impl PlistState {
    pub(crate) fn class(&self) -> PropertyListClass {
        match self {
            Self::FileCreate(_) => PropertyListClass::FileCreate,
            Self::FileAccess(_) => PropertyListClass::FileAccess,
            Self::DatasetCreate(_) => PropertyListClass::DatasetCreate,
            Self::DatasetAccess(_) => PropertyListClass::DatasetAccess,
            Self::LinkCreate(_) => PropertyListClass::LinkCreate,
        }
    }

    pub(crate) fn properties(&self) -> Vec<String> {
        match self {
            Self::FileCreate(_) => file_create::PROPERTY_NAMES
                .iter()
                .map(|s| s.to_string())
                .collect(),
            Self::FileAccess(_) => file_access::PROPERTY_NAMES
                .iter()
                .map(|s| s.to_string())
                .collect(),
            Self::DatasetCreate(_) => dataset_create::PROPERTY_NAMES
                .iter()
                .map(|s| s.to_string())
                .collect(),
            Self::DatasetAccess(_) => dataset_access::PROPERTY_NAMES
                .iter()
                .map(|s| s.to_string())
                .collect(),
            Self::LinkCreate(_) => link_create::PROPERTY_NAMES
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }
}

/// Property list class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PropertyListClass {
    AttributeCreate,
    DatasetAccess,
    DatasetCreate,
    DataTransfer,
    DatatypeAccess,
    DatatypeCreate,
    FileAccess,
    FileCreate,
    FileMount,
    GroupAccess,
    GroupCreate,
    LinkAccess,
    LinkCreate,
    ObjectCopy,
    ObjectCreate,
    StringCreate,
}

impl PropertyListClass {
    pub fn name(self) -> &'static str {
        match self {
            Self::AttributeCreate => "attribute create",
            Self::DatasetAccess => "dataset access",
            Self::DatasetCreate => "dataset create",
            Self::DataTransfer => "data transfer",
            Self::DatatypeAccess => "datatype access",
            Self::DatatypeCreate => "datatype create",
            Self::FileAccess => "file access",
            Self::FileCreate => "file create",
            Self::FileMount => "file mount",
            Self::GroupAccess => "group access",
            Self::GroupCreate => "group create",
            Self::LinkAccess => "link access",
            Self::LinkCreate => "link create",
            Self::ObjectCopy => "object copy",
            Self::ObjectCreate => "object create",
            Self::StringCreate => "string create",
        }
    }
}

impl Display for PropertyListClass {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for PropertyListClass {
    type Err = crate::error::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "attribute create" => Ok(Self::AttributeCreate),
            "dataset access" => Ok(Self::DatasetAccess),
            "dataset create" => Ok(Self::DatasetCreate),
            "data transfer" => Ok(Self::DataTransfer),
            "datatype access" => Ok(Self::DatatypeAccess),
            "datatype create" => Ok(Self::DatatypeCreate),
            "file access" => Ok(Self::FileAccess),
            "file create" => Ok(Self::FileCreate),
            "file mount" => Ok(Self::FileMount),
            "group access" => Ok(Self::GroupAccess),
            "group create" => Ok(Self::GroupCreate),
            "link access" => Ok(Self::LinkAccess),
            "link create" => Ok(Self::LinkCreate),
            "object copy" => Ok(Self::ObjectCopy),
            "object create" => Ok(Self::ObjectCreate),
            "string create" => Ok(Self::StringCreate),
            _ => Err(format!("invalid property list class: {s}").into()),
        }
    }
}

/// A generic property list object.
#[repr(transparent)]
#[derive(Clone)]
pub struct PropertyList(pub(crate) Handle);

impl ObjectClass for PropertyList {
    const NAME: &'static str = "property list";
    const VALID_TYPES: &'static [H5I_type_t] = &[H5I_type_t::H5I_GENPROP_LST];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }

    fn short_repr(&self) -> Option<String> {
        self.class().ok().map(|c| c.to_string())
    }
}

impl Debug for PropertyList {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.debug_fmt(f)
    }
}

impl Deref for PropertyList {
    type Target = Object;

    fn deref(&self) -> &Object {
        unsafe { self.transmute() }
    }
}

impl PartialEq for PropertyList {
    fn eq(&self, other: &Self) -> bool {
        match (self.0.plist_state(), other.0.plist_state()) {
            (Some(a), Some(b)) => plist_state_eq(a, b),
            _ => false,
        }
    }
}

impl Eq for PropertyList {}

fn plist_state_eq(a: &PlistState, b: &PlistState) -> bool {
    match (a, b) {
        (PlistState::FileCreate(x), PlistState::FileCreate(y)) => x == y,
        (PlistState::FileAccess(x), PlistState::FileAccess(y)) => x == y,
        (PlistState::DatasetCreate(x), PlistState::DatasetCreate(y)) => x == y,
        (PlistState::DatasetAccess(x), PlistState::DatasetAccess(y)) => x == y,
        (PlistState::LinkCreate(x), PlistState::LinkCreate(y)) => x == y,
        _ => false,
    }
}

impl PropertyList {
    pub(crate) fn from_state(state: PlistState) -> Self {
        Self(Handle::new(Payload::PropertyList(state)))
    }

    pub(crate) fn state(&self) -> Result<&PlistState> {
        self.0
            .plist_state()
            .ok_or_else(|| "invalid property list handle".into())
    }

    /// Copies the property list.
    pub fn copy(&self) -> Self {
        match self.state() {
            Ok(s) => Self::from_state(s.clone()),
            Err(_) => Self::invalid(),
        }
    }

    /// Queries whether a property name exists in the property list.
    pub fn has(&self, property: &str) -> bool {
        self.state()
            .map(|s| s.properties().iter().any(|p| p == property))
            .unwrap_or(false)
    }

    /// Returns the current number of registered properties.
    pub fn len(&self) -> usize {
        self.state().map(|s| s.properties().len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the names of all registered properties.
    pub fn properties(&self) -> Vec<String> {
        self.state().map(|s| s.properties()).unwrap_or_default()
    }

    /// Returns the class of the property list.
    pub fn class(&self) -> Result<PropertyListClass> {
        self.state().map(|s| s.class())
    }

    /// Returns `true` if the property list belongs to the given class.
    pub fn is_class(&self, class: PropertyListClass) -> bool {
        self.class().map(|c| c == class).unwrap_or(false)
    }
}

/// No-op: vlen memory always uses libc malloc/free in this implementation.
pub fn set_vlen_manager_libc(_plist: crate::h5i::hid_t) -> crate::error::Result<()> {
    Ok(())
}
