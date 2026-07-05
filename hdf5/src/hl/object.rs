//! The base `Object` type: any HDF5 entity referenced through a handle.

use std::fmt::{self, Debug};

use crate::class::ObjectClass;
use crate::error::Result;
use crate::h5i::{hid_t, H5I_type_t};
use crate::handle::Handle;
use crate::hl::{
    Attribute, Container, Dataset, Dataspace, Datatype, File, Group, Location, PropertyList,
};

/// Any HDF5 object that can be referenced through an identifier.
#[repr(transparent)]
#[derive(Clone)]
pub struct Object(pub(crate) Handle);

impl ObjectClass for Object {
    const NAME: &'static str = "object";
    const VALID_TYPES: &'static [H5I_type_t] = &[];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }
}

impl Debug for Object {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.debug_fmt(f)
    }
}

impl Object {
    pub fn id(&self) -> hid_t {
        self.0.id()
    }

    /// Returns reference count if the handle is valid and 0 otherwise.
    pub fn refcount(&self) -> u32 {
        self.handle().refcount()
    }

    /// Returns `true` if the object has a valid unlocked identifier.
    pub fn is_valid(&self) -> bool {
        self.handle().is_valid_user_id()
    }

    /// Returns type of the object.
    pub fn id_type(&self) -> H5I_type_t {
        self.handle().id_type()
    }

    pub(crate) fn try_borrow(&self) -> Result<Handle> {
        Ok(self.0.clone())
    }
}

macro_rules! impl_downcast {
    ($func:ident, $tp:ty) => {
        impl Object {
            #[doc = concat!("Downcast the object into `", stringify!($tp), "` if possible.")]
            pub fn $func(&self) -> Result<$tp> {
                self.clone().cast()
            }
        }
    };
}

impl_downcast!(as_file, File);
impl_downcast!(as_group, Group);
impl_downcast!(as_dataset, Dataset);
impl_downcast!(as_location, Location);
impl_downcast!(as_attr, Attribute);
impl_downcast!(as_container, Container);
impl_downcast!(as_datatype, Datatype);
impl_downcast!(as_dataspace, Dataspace);
impl_downcast!(as_plist, PropertyList);
