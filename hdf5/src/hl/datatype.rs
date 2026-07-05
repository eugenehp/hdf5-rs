//! HDF5 datatypes.

use std::borrow::Borrow;
use std::fmt::{self, Debug};
use std::ops::Deref;

use hdf5_types::{H5Type, TypeDescriptor};

use crate::class::ObjectClass;
use crate::error::Result;
use crate::h5i::H5I_type_t;
use crate::handle::{Handle, Payload};
use crate::hl::object::Object;

/// Datatype conversion path quality.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Conversion {
    /// Type layouts are identical: a plain copy suffices.
    #[default]
    NoOp = 1,
    /// A compiled (hard) conversion exists.
    Hard,
    /// A soft (generic) conversion exists.
    Soft,
}

/// Byte order of a datatype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteOrder {
    LittleEndian,
    BigEndian,
    Vax,
    Mixed,
    None,
}

/// An HDF5 datatype object.
#[repr(transparent)]
#[derive(Clone)]
pub struct Datatype(Handle);

impl ObjectClass for Datatype {
    const NAME: &'static str = "datatype";
    const VALID_TYPES: &'static [H5I_type_t] = &[H5I_type_t::H5I_DATATYPE];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }

    fn short_repr(&self) -> Option<String> {
        self.to_descriptor().ok().map(|d| d.to_string())
    }
}

impl Debug for Datatype {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.debug_fmt(f)
    }
}

impl Deref for Datatype {
    type Target = Object;

    fn deref(&self) -> &Object {
        unsafe { self.transmute() }
    }
}

impl PartialEq for Datatype {
    fn eq(&self, other: &Self) -> bool {
        match (self.to_descriptor(), other.to_descriptor()) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Datatype {}

impl Datatype {
    pub(crate) fn from_descriptor_internal(desc: TypeDescriptor) -> Self {
        Self(Handle::new(Payload::Datatype(desc)))
    }

    pub(crate) fn descriptor(&self) -> Result<TypeDescriptor> {
        if let Some(d) = self.0.datatype_desc() {
            return Ok(d.clone());
        }
        // named datatype living in a file
        if let (Some(file), Some(id)) = (self.0.file(), self.0.obj_id()) {
            let state = file.state.read();
            if let crate::model::ObjectKind::NamedType(t) = &state.get(id).kind {
                return Ok(t.clone());
            }
        }
        Err("invalid datatype handle".into())
    }

    /// Returns the size of the datatype in bytes.
    pub fn size(&self) -> usize {
        self.descriptor()
            .map(|d| crate::format::convert::disk_size(&d))
            .unwrap_or(0)
    }

    /// Returns the byte order of the datatype.
    pub fn byte_order(&self) -> ByteOrder {
        // The pure-Rust engine always reads/writes little-endian data.
        match self.descriptor() {
            Ok(
                TypeDescriptor::Compound(_)
                | TypeDescriptor::FixedAscii(_)
                | TypeDescriptor::FixedUnicode(_)
                | TypeDescriptor::VarLenAscii
                | TypeDescriptor::VarLenUnicode,
            ) => ByteOrder::None,
            Ok(_) => ByteOrder::LittleEndian,
            Err(_) => ByteOrder::None,
        }
    }

    /// Returns the best conversion path to the destination type, if any.
    pub fn conv_path<D>(&self, dst: D) -> Option<Conversion>
    where
        D: Borrow<Self>,
    {
        let src = self.to_descriptor().ok()?;
        let dst = dst.borrow().to_descriptor().ok()?;
        conversion_path(&src, &dst)
    }

    /// Returns the conversion path to a Rust type, if any.
    pub fn conv_to<T: H5Type>(&self) -> Option<Conversion> {
        let dst = Self::from_type::<T>().ok()?;
        self.conv_path(dst)
    }

    /// Returns the conversion path from a Rust type, if any.
    pub fn conv_from<T: H5Type>(&self) -> Option<Conversion> {
        let src = Self::from_type::<T>().ok()?;
        src.conv_path(self)
    }

    /// Returns true if this datatype is identical to the Rust type's.
    pub fn is<T: H5Type>(&self) -> bool {
        self.to_descriptor()
            .map(|d| d == T::type_descriptor())
            .unwrap_or(false)
    }

    /// Extract the type descriptor.
    pub fn to_descriptor(&self) -> Result<TypeDescriptor> {
        self.descriptor()
    }

    /// Create a datatype from a Rust type.
    pub fn from_type<T: H5Type>() -> Result<Self> {
        Self::from_descriptor(&T::type_descriptor())
    }

    /// Create a datatype from a type descriptor.
    pub fn from_descriptor(desc: &TypeDescriptor) -> Result<Self> {
        Ok(Self::from_descriptor_internal(desc.clone()))
    }
}

/// Compute the conversion path between two descriptors.
pub(crate) fn conversion_path(src: &TypeDescriptor, dst: &TypeDescriptor) -> Option<Conversion> {
    use TypeDescriptor::*;
    if src == dst {
        return Some(Conversion::NoOp);
    }
    match (src, dst) {
        // numeric width/signedness/int-float conversions: hard paths
        (Integer(_) | Unsigned(_) | Float(_), Integer(_) | Unsigned(_) | Float(_)) => {
            Some(Conversion::Hard)
        }
        (Compound(a), Compound(b)) => {
            if a.fields.len() != b.fields.len() {
                return None;
            }
            let mut worst = Conversion::NoOp;
            for bf in &b.fields {
                let af = a.fields.iter().find(|f| f.name == bf.name)?;
                let path = conversion_path(&af.ty, &bf.ty)?;
                worst = worst
                    .max(path)
                    .max(Conversion::Soft.min(if af.offset == bf.offset {
                        Conversion::NoOp
                    } else {
                        Conversion::Soft
                    }));
            }
            Some(worst)
        }
        (FixedArray(a, n), FixedArray(b, m)) if n == m => conversion_path(a, b),
        (VarLenArray(a), VarLenArray(b)) => conversion_path(a, b),
        (FixedAscii(_), FixedAscii(_))
        | (FixedUnicode(_), FixedUnicode(_))
        | (FixedAscii(_), FixedUnicode(_)) => Some(Conversion::Soft),
        (VarLenAscii, VarLenUnicode) | (VarLenAscii, VarLenAscii) => Some(Conversion::Soft),
        (VarLenUnicode, VarLenUnicode) => Some(Conversion::NoOp),
        // libhdf5 converts among all string forms (fixed<->vlen, any charset)
        (FixedUnicode(_), FixedAscii(_)) | (VarLenUnicode, VarLenAscii) => Some(Conversion::Soft),
        (FixedAscii(_) | FixedUnicode(_), VarLenAscii | VarLenUnicode) => Some(Conversion::Soft),
        (VarLenAscii | VarLenUnicode, FixedAscii(_) | FixedUnicode(_)) => Some(Conversion::Soft),
        (Enum(e), Boolean) | (Boolean, Enum(e)) if e.size as usize == 1 => Some(Conversion::Soft),
        _ => None,
    }
}
