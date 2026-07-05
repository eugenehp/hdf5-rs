//! HDF5 attributes and their builders.

use std::fmt::{self, Debug};
use std::ops::Deref;

use ndarray::ArrayView;

use hdf5_types::{H5Type, TypeDescriptor};

use crate::class::ObjectClass;
use crate::error::Result;
use crate::format::convert::{disk_size, to_disk_repr, VlenStore};
use crate::h5i::H5I_type_t;
use crate::handle::{Handle, Payload};
use crate::hl::container::Container;
use crate::hl::datatype::Conversion;
use crate::hl::extents::Extents;
use crate::hl::location::Location;
use crate::model::AttrData;

/// An HDF5 attribute.
#[repr(transparent)]
#[derive(Clone)]
pub struct Attribute(Handle);

impl ObjectClass for Attribute {
    const NAME: &'static str = "attribute";
    const VALID_TYPES: &'static [H5I_type_t] = &[H5I_type_t::H5I_ATTR];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }
}

impl Debug for Attribute {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.debug_fmt(f)
    }
}

impl Deref for Attribute {
    type Target = Container;

    fn deref(&self) -> &Container {
        unsafe { self.transmute() }
    }
}

impl Attribute {
    /// Returns the names of all attributes attached to the given location.
    pub fn attr_names(obj: &Location) -> Result<Vec<String>> {
        obj.attr_names()
    }

    /// Returns this attribute's own name.
    pub fn name(&self) -> String {
        match self.0.payload() {
            Payload::Attribute { name, .. } => name.clone(),
            _ => String::new(),
        }
    }
}

/// A builder for new attributes.
pub struct AttributeBuilder {
    parent: Result<Handle>,
    packed: bool,
}

impl AttributeBuilder {
    pub fn new(parent: &Location) -> Self {
        Self {
            parent: parent.try_borrow(),
            packed: false,
        }
    }

    pub fn packed(mut self, packed: bool) -> Self {
        self.packed = packed;
        self
    }

    pub fn empty<T: H5Type>(self) -> AttributeBuilderEmpty {
        self.empty_as(&T::type_descriptor())
    }

    pub fn empty_as(self, type_desc: &TypeDescriptor) -> AttributeBuilderEmpty {
        AttributeBuilderEmpty {
            builder: self,
            type_desc: type_desc.clone(),
            extents: Extents::Scalar,
        }
    }

    pub fn with_data<'d, A, T, D>(self, data: A) -> AttributeBuilderData<'d, T, D>
    where
        A: Into<ArrayView<'d, T, D>>,
        T: H5Type,
        D: ndarray::Dimension,
    {
        let view = data.into();
        AttributeBuilderData {
            builder: self,
            data: view,
            type_desc: T::type_descriptor(),
            conv: Conversion::Soft,
        }
    }

    pub fn with_data_as<'d, A, T, D>(
        self,
        data: A,
        type_desc: &TypeDescriptor,
    ) -> AttributeBuilderData<'d, T, D>
    where
        A: Into<ArrayView<'d, T, D>>,
        T: H5Type,
        D: ndarray::Dimension,
    {
        let view = data.into();
        AttributeBuilderData {
            builder: self,
            data: view,
            type_desc: type_desc.clone(),
            conv: Conversion::Soft,
        }
    }

    fn create_attr(
        &self,
        name: &str,
        type_desc: &TypeDescriptor,
        extents: &Extents,
    ) -> Result<Attribute> {
        let parent = self.parent.clone()?;
        let file = parent.file().ok_or("parent is not file-resident")?.clone();
        let owner = parent.obj_id().ok_or("parent has no location")?;

        let type_desc = if self.packed {
            type_desc.to_packed_repr()
        } else {
            type_desc.clone()
        };
        let disk_desc = to_disk_repr(&type_desc);
        let esize = disk_size(&disk_desc);

        let (dims, is_scalar, is_null) = match extents {
            Extents::Null => (vec![], false, true),
            Extents::Scalar => (vec![], true, false),
            Extents::Simple(se) => (
                se.dims().iter().map(|&d| d as u64).collect::<Vec<u64>>(),
                false,
                false,
            ),
        };
        let n: usize = if is_null {
            0
        } else if is_scalar {
            1
        } else {
            dims.iter().product::<u64>() as usize
        };

        let mut state = file.state.write();
        if state.read_only {
            return Err("unable to create attribute: file is read-only".into());
        }
        let node = state.get_mut(owner);
        if node.attr_index(name).is_some() {
            return Err(format!("attribute '{name}' already exists").into());
        }
        node.attrs.push(AttrData {
            name: name.to_string(),
            dtype: disk_desc,
            dims,
            is_scalar,
            is_null,
            data: vec![0u8; n * esize],
            vlen: VlenStore::new(),
        });
        drop(state);
        Ok(Attribute::from_handle(Handle::new(Payload::Attribute {
            file,
            owner,
            name: name.to_string(),
        })))
    }
}

/// Attribute builder with a datatype chosen.
pub struct AttributeBuilderEmpty {
    builder: AttributeBuilder,
    type_desc: TypeDescriptor,
    extents: Extents,
}

impl AttributeBuilderEmpty {
    pub fn shape<S: Into<Extents>>(self, extents: S) -> AttributeBuilderEmptyShape {
        AttributeBuilderEmptyShape {
            builder: self.builder,
            type_desc: self.type_desc,
            extents: extents.into(),
        }
    }

    pub fn packed(mut self, packed: bool) -> Self {
        self.builder = self.builder.packed(packed);
        self
    }

    pub fn create<'n, T: Into<&'n str>>(self, name: T) -> Result<Attribute> {
        self.builder
            .create_attr(name.into(), &self.type_desc, &self.extents)
    }
}

/// Attribute builder with datatype and shape chosen.
pub struct AttributeBuilderEmptyShape {
    builder: AttributeBuilder,
    type_desc: TypeDescriptor,
    extents: Extents,
}

impl AttributeBuilderEmptyShape {
    pub fn packed(mut self, packed: bool) -> Self {
        self.builder = self.builder.packed(packed);
        self
    }

    pub fn create<'n, T: Into<&'n str>>(&self, name: T) -> Result<Attribute> {
        self.builder
            .create_attr(name.into(), &self.type_desc, &self.extents)
    }
}

/// Attribute builder holding data to write on creation.
pub struct AttributeBuilderData<'d, T, D> {
    builder: AttributeBuilder,
    data: ArrayView<'d, T, D>,
    type_desc: TypeDescriptor,
    conv: Conversion,
}

impl<'d, T, D> AttributeBuilderData<'d, T, D>
where
    T: H5Type,
    D: ndarray::Dimension,
{
    pub fn conversion(mut self, conv: Conversion) -> Self {
        self.conv = conv;
        self
    }

    pub fn no_convert(mut self) -> Self {
        self.conv = Conversion::NoOp;
        self
    }

    pub fn packed(mut self, packed: bool) -> Self {
        self.builder = self.builder.packed(packed);
        self
    }

    pub fn create<'n, N: Into<&'n str>>(&self, name: N) -> Result<Attribute> {
        let shape: Vec<usize> = self.data.shape().to_vec();
        let extents = Extents::from(&shape[..]);
        let attr = self
            .builder
            .create_attr(name.into(), &self.type_desc, &extents)?;
        let writer = crate::hl::container::Writer::new(&attr).conversion(self.conv);
        let slice = self
            .data
            .as_slice()
            .ok_or("input array is not contiguous or not in standard layout")?;
        writer.write_raw(slice)?;
        Ok(attr)
    }
}
