//! HDF5 dataspaces: extents plus an optional selection.

use std::fmt::{self, Debug};
use std::ops::Deref;

use crate::class::ObjectClass;
use crate::error::Result;
use crate::h5i::H5I_type_t;
use crate::handle::{Handle, Payload};
use crate::hl::extents::{Extent, Extents, Ix};
use crate::hl::object::Object;
use crate::hl::selection::{RawSelection, Selection};

/// The payload of a dataspace handle.
#[derive(Clone, Debug)]
pub(crate) struct DataspaceState {
    pub extents: Extents,
    pub selection: RawSelection,
}

/// An HDF5 dataspace object.
#[repr(transparent)]
#[derive(Clone)]
pub struct Dataspace(Handle);

impl ObjectClass for Dataspace {
    const NAME: &'static str = "dataspace";
    const VALID_TYPES: &'static [H5I_type_t] = &[H5I_type_t::H5I_DATASPACE];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }

    fn short_repr(&self) -> Option<String> {
        self.state().map(|s| format!("{}", s.extents)).ok()
    }
}

impl Debug for Dataspace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.debug_fmt(f)
    }
}

impl Deref for Dataspace {
    type Target = Object;

    fn deref(&self) -> &Object {
        unsafe { self.transmute() }
    }
}

impl Dataspace {
    /// Applies a raw selection to a copy of this dataspace.
    pub fn select_raw<S: Into<RawSelection>>(&self, raw_sel: S) -> Result<Self> {
        let state = self.state()?;
        Ok(Self::from_state(DataspaceState {
            extents: state.extents.clone(),
            selection: raw_sel.into(),
        }))
    }

    /// Returns the current selection in raw form.
    pub fn get_raw_selection(&self) -> Result<RawSelection> {
        Ok(self.state()?.selection.clone())
    }

    pub(crate) fn from_state(state: DataspaceState) -> Self {
        Self(Handle::new(Payload::Dataspace(state)))
    }

    pub(crate) fn from_extents_internal(extents: Extents) -> Self {
        Self::from_state(DataspaceState {
            extents,
            selection: RawSelection::All,
        })
    }

    pub(crate) fn state(&self) -> Result<&DataspaceState> {
        self.0
            .dataspace_state()
            .ok_or_else(|| "invalid dataspace handle".into())
    }

    /// Create a new dataspace from the given extents.
    pub fn try_new<T: Into<Extents>>(extents: T) -> Result<Self> {
        let extents = extents.into();
        if !extents.is_valid() {
            return Err(format!("invalid extents: {extents}").into());
        }
        Ok(Self::from_extents_internal(extents))
    }

    /// Copy the dataspace.
    pub fn copy(&self) -> Self {
        match self.state() {
            Ok(s) => Self::from_state(s.clone()),
            Err(_) => Self::invalid(),
        }
    }

    pub fn ndim(&self) -> usize {
        self.state().map(|s| s.extents.ndim()).unwrap_or(0)
    }

    pub fn shape(&self) -> Vec<Ix> {
        self.state().map(|s| s.extents.dims()).unwrap_or_default()
    }

    pub fn maxdims(&self) -> Vec<Option<Ix>> {
        self.state()
            .map(|s| s.extents.maxdims())
            .unwrap_or_default()
    }

    pub fn is_resizable(&self) -> bool {
        self.state()
            .map(|s| s.extents.is_resizable())
            .unwrap_or(false)
    }

    pub fn is_null(&self) -> bool {
        self.state().map(|s| s.extents.is_null()).unwrap_or(false)
    }

    pub fn is_scalar(&self) -> bool {
        self.state().map(|s| s.extents.is_scalar()).unwrap_or(false)
    }

    pub fn is_simple(&self) -> bool {
        self.state().map(|s| s.extents.is_simple()).unwrap_or(false)
    }

    pub fn is_valid(&self) -> bool {
        self.state().map(|s| s.extents.is_valid()).unwrap_or(false)
    }

    pub fn size(&self) -> usize {
        self.state().map(|s| s.extents.size()).unwrap_or(0)
    }

    /// Serialize the dataspace (extents only) into bytes.
    ///
    /// Note: this uses a crate-internal encoding, not `H5Sencode` bytes.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let s = self.state()?;
        let dims = s.extents.dims();
        let maxdims = s.extents.maxdims();
        let mut out = Vec::new();
        let kind: u8 = if s.extents.is_null() {
            2
        } else if s.extents.is_scalar() {
            0
        } else {
            1
        };
        out.push(kind);
        out.push(dims.len() as u8);
        for (d, m) in dims.iter().zip(&maxdims) {
            out.extend_from_slice(&(*d as u64).to_le_bytes());
            out.extend_from_slice(&m.map(|v| v as u64).unwrap_or(u64::MAX).to_le_bytes());
        }
        Ok(out)
    }

    /// Deserialize a dataspace previously produced by [`Dataspace::encode`].
    pub fn decode<T>(buf: T) -> Result<Self>
    where
        T: AsRef<[u8]>,
    {
        let buf = buf.as_ref();
        if buf.len() < 2 {
            return Err("dataspace decode: buffer too short".into());
        }
        let kind = buf[0];
        let ndim = buf[1] as usize;
        match kind {
            2 => Self::try_new(Extents::Null),
            0 => Self::try_new(Extents::Scalar),
            _ => {
                let mut extents = Vec::with_capacity(ndim);
                let mut pos = 2;
                for _ in 0..ndim {
                    if pos + 16 > buf.len() {
                        return Err("dataspace decode: buffer too short".into());
                    }
                    let d = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
                    let m = u64::from_le_bytes(buf[pos + 8..pos + 16].try_into().unwrap());
                    pos += 16;
                    extents.push(Extent::new(
                        d as usize,
                        if m == u64::MAX {
                            None
                        } else {
                            Some(m as usize)
                        },
                    ));
                }
                Self::try_new(Extents::simple(
                    crate::hl::extents::SimpleExtents::from_vec(extents),
                ))
            }
        }
    }

    /// Get the extents of the dataspace.
    pub fn extents(&self) -> Result<Extents> {
        self.state().map(|s| s.extents.clone())
    }

    /// The number of elements in the current selection.
    pub fn selection_size(&self) -> usize {
        match self.state() {
            Ok(s) => s
                .selection
                .linear_indices(&s.extents.dims())
                .map(|v| v.len())
                .unwrap_or_else(|_| self.size()),
            Err(_) => 0,
        }
    }

    /// Return a copy of this dataspace with the given selection applied.
    pub fn select<S: Into<Selection>>(&self, selection: S) -> Result<Self> {
        let state = self.state()?;
        let selection = selection.into();
        let shape = state.extents.dims();
        let raw = selection.into_raw(&shape)?;
        Ok(Self::from_state(DataspaceState {
            extents: state.extents.clone(),
            selection: raw,
        }))
    }

    /// Get the current selection.
    pub fn get_selection(&self) -> Result<Selection> {
        let state = self.state()?;
        Selection::from_raw(state.selection.clone())
    }
}
