//! Dataset access property list.

use std::fmt::{self, Debug};
use std::ops::Deref;

use crate::class::ObjectClass;
use crate::error::Result;
use crate::h5i::H5I_type_t;
use crate::handle::Handle;
use crate::hl::plist::{PlistState, PropertyList};

pub use crate::hl::plist::file_access::ChunkCache;

pub(crate) const PROPERTY_NAMES: &[&str] = &["chunk_cache", "efile_prefix", "virtual_view"];

/// View of missing mapped elements in a virtual dataset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VirtualView {
    #[default]
    FirstMissing,
    LastAvailable,
}

/// The data carried by a dataset-access property list.
#[derive(Clone, Debug, PartialEq, Default)]
pub(crate) struct DatasetAccessData {
    pub chunk_cache: ChunkCache,
    pub efile_prefix: String,
    pub virtual_view: VirtualView,
    pub virtual_printf_gap: usize,
}

/// Dataset access property list.
#[repr(transparent)]
#[derive(Clone)]
pub struct DatasetAccess(Handle);

impl ObjectClass for DatasetAccess {
    const NAME: &'static str = "dataset access property list";
    const VALID_TYPES: &'static [H5I_type_t] = &[H5I_type_t::H5I_GENPROP_LST];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }
}

impl Debug for DatasetAccess {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.debug_fmt(f)
    }
}

impl Deref for DatasetAccess {
    type Target = PropertyList;

    fn deref(&self) -> &PropertyList {
        unsafe { self.transmute() }
    }
}

impl PartialEq for DatasetAccess {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl Eq for DatasetAccess {}

impl Default for DatasetAccess {
    fn default() -> Self {
        Self::try_new().unwrap()
    }
}

impl DatasetAccess {
    pub(crate) fn from_data(data: DatasetAccessData) -> Self {
        Self(PropertyList::from_state(PlistState::DatasetAccess(data)).0)
    }

    pub(crate) fn data(&self) -> DatasetAccessData {
        match self.0.plist_state() {
            Some(PlistState::DatasetAccess(d)) => d.clone(),
            _ => DatasetAccessData::default(),
        }
    }

    pub fn try_new() -> Result<Self> {
        Ok(Self::from_data(DatasetAccessData::default()))
    }

    pub fn copy(&self) -> Self {
        Self::from_data(self.data())
    }

    pub fn build() -> DatasetAccessBuilder {
        DatasetAccessBuilder::new()
    }

    pub fn get_chunk_cache(&self) -> Result<ChunkCache> {
        Ok(self.data().chunk_cache)
    }

    pub fn chunk_cache(&self) -> ChunkCache {
        self.data().chunk_cache
    }

    pub fn get_efile_prefix(&self) -> Result<String> {
        Ok(self.data().efile_prefix)
    }

    pub fn efile_prefix(&self) -> String {
        self.data().efile_prefix
    }

    pub fn get_virtual_view(&self) -> Result<VirtualView> {
        Ok(self.data().virtual_view)
    }

    pub fn virtual_view(&self) -> VirtualView {
        self.data().virtual_view
    }

    pub fn get_virtual_printf_gap(&self) -> Result<usize> {
        Ok(self.data().virtual_printf_gap)
    }

    pub fn virtual_printf_gap(&self) -> usize {
        self.data().virtual_printf_gap
    }
}

/// Builder for dataset access property lists.
#[derive(Clone, Debug, Default)]
pub struct DatasetAccessBuilder {
    data: DatasetAccessData,
}

impl DatasetAccessBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_plist(plist: &DatasetAccess) -> Result<Self> {
        Ok(Self { data: plist.data() })
    }

    pub fn chunk_cache(&mut self, nslots: usize, nbytes: usize, w0: f64) -> &mut Self {
        self.data.chunk_cache = ChunkCache { nslots, nbytes, w0 };
        self
    }

    pub fn efile_prefix(&mut self, prefix: &str) -> &mut Self {
        self.data.efile_prefix = prefix.into();
        self
    }

    pub fn virtual_view(&mut self, view: VirtualView) -> &mut Self {
        self.data.virtual_view = view;
        self
    }

    pub fn virtual_printf_gap(&mut self, gap_size: usize) -> &mut Self {
        self.data.virtual_printf_gap = gap_size;
        self
    }

    pub fn apply(&self, plist: &mut DatasetAccess) -> Result<()> {
        *plist = DatasetAccess::from_data(self.data.clone());
        Ok(())
    }

    pub fn finish(&self) -> Result<DatasetAccess> {
        Ok(DatasetAccess::from_data(self.data.clone()))
    }
}
