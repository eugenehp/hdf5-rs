//! Dataset creation property list.

use std::fmt::{self, Debug};
use std::ops::Deref;

use bitflags::bitflags;

use hdf5_types::OwnedDynValue;

use crate::class::ObjectClass;
use crate::dim::Dimension;
use crate::error::Result;
use crate::h5i::H5I_type_t;
use crate::handle::Handle;
use crate::hl::filters::{Blosc, BloscShuffle, Filter, SZip, ScaleOffset};
use crate::hl::plist::common::{AttrCreationOrder, AttrPhaseChange};
use crate::hl::plist::{PlistState, PropertyList};

pub(crate) const PROPERTY_NAMES: &[&str] = &[
    "filters",
    "alloc_time",
    "fill_time",
    "fill_value",
    "chunk",
    "layout",
    "external",
    "obj_track_times",
    "attr_phase_change",
    "attr_creation_order",
];

/// Dataset storage layout class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Layout {
    Compact,
    #[default]
    Contiguous,
    Chunked,
    Virtual,
}

/// Storage space allocation timing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocTime {
    Early,
    Incr,
    Late,
}

/// Fill value writing timing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FillTime {
    #[default]
    IfSet,
    Alloc,
    Never,
}

/// Fill value status.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FillValue {
    Undefined,
    #[default]
    Default,
    UserDefined,
}

bitflags! {
    /// Options for chunked datasets.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
    pub struct ChunkOpts: u32 {
        const DONT_FILTER_PARTIAL_CHUNKS = 0x1;
    }
}

/// One virtual-dataset mapping (API-compatible with the FFI crate).
#[derive(Clone, Debug)]
pub struct VirtualMapping {
    pub src_filename: String,
    pub src_dataset: String,
    pub src_extents: crate::hl::extents::Extents,
    pub src_selection: crate::hl::selection::Selection,
    pub vds_extents: crate::hl::extents::Extents,
    pub vds_selection: crate::hl::selection::Selection,
}

/// An external file specification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalFile {
    pub name: String,
    pub offset: usize,
    pub size: usize,
}

/// The data carried by a dataset-creation property list.
#[derive(Clone, Debug, Default)]
pub(crate) struct DatasetCreateData {
    pub filters: Vec<Filter>,
    pub alloc_time: Option<AllocTime>,
    pub fill_time: FillTime,
    pub fill_value: Option<OwnedDynValue>,
    pub chunk: Option<Vec<usize>>,
    pub layout: Layout,
    pub external: Vec<ExternalFile>,
    pub virtual_maps: Vec<VirtualMapping>,
    pub obj_track_times: bool,
    pub attr_phase_change: AttrPhaseChange,
    pub attr_creation_order: AttrCreationOrder,
    pub chunk_opts: ChunkOpts,
}

impl PartialEq for DatasetCreateData {
    fn eq(&self, other: &Self) -> bool {
        self.filters == other.filters
            && self.alloc_time == other.alloc_time
            && self.fill_time == other.fill_time
            && self.chunk == other.chunk
            && self.layout == other.layout
            && self.external == other.external
            && self.obj_track_times == other.obj_track_times
            && self.attr_phase_change == other.attr_phase_change
            && self.attr_creation_order == other.attr_creation_order
            && self.chunk_opts == other.chunk_opts
    }
}

/// Dataset creation property list.
#[repr(transparent)]
#[derive(Clone)]
pub struct DatasetCreate(Handle);

impl ObjectClass for DatasetCreate {
    const NAME: &'static str = "dataset create property list";
    const VALID_TYPES: &'static [H5I_type_t] = &[H5I_type_t::H5I_GENPROP_LST];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }
}

impl Debug for DatasetCreate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.debug_fmt(f)
    }
}

impl Deref for DatasetCreate {
    type Target = PropertyList;

    fn deref(&self) -> &PropertyList {
        unsafe { self.transmute() }
    }
}

impl PartialEq for DatasetCreate {
    fn eq(&self, other: &Self) -> bool {
        self.data() == other.data()
    }
}

impl Eq for DatasetCreate {}

impl Default for DatasetCreate {
    fn default() -> Self {
        Self::try_new().unwrap()
    }
}

impl DatasetCreate {
    pub(crate) fn from_data(data: DatasetCreateData) -> Self {
        Self(PropertyList::from_state(PlistState::DatasetCreate(data)).0)
    }

    pub(crate) fn data(&self) -> DatasetCreateData {
        match self.0.plist_state() {
            Some(PlistState::DatasetCreate(d)) => d.clone(),
            _ => DatasetCreateData::default(),
        }
    }

    pub fn try_new() -> Result<Self> {
        Ok(Self::from_data(DatasetCreateData::default()))
    }

    pub fn copy(&self) -> Self {
        Self::from_data(self.data())
    }

    pub fn build() -> DatasetCreateBuilder {
        DatasetCreateBuilder::new()
    }

    pub fn filters(&self) -> Vec<Filter> {
        self.data().filters
    }

    pub fn has_filters(&self) -> bool {
        !self.data().filters.is_empty()
    }

    pub fn alloc_time(&self) -> AllocTime {
        self.data().alloc_time.unwrap_or(AllocTime::Late)
    }

    pub fn fill_time(&self) -> FillTime {
        self.data().fill_time
    }

    pub fn fill_value_defined(&self) -> FillValue {
        match self.data().fill_value {
            Some(_) => FillValue::UserDefined,
            None => FillValue::Default,
        }
    }

    pub fn fill_value(&self, tp: &hdf5_types::TypeDescriptor) -> Result<Option<OwnedDynValue>> {
        let _ = tp;
        Ok(self.data().fill_value)
    }

    pub fn fill_value_as<T: hdf5_types::H5Type + Clone>(&self) -> Result<Option<T>> {
        match self.data().fill_value {
            Some(v) => Ok(v.cast::<T>().ok()),
            None => Ok(None),
        }
    }

    pub fn chunk(&self) -> Option<Vec<usize>> {
        self.data().chunk
    }

    pub fn layout(&self) -> Layout {
        self.data().layout
    }

    pub fn chunk_opts(&self) -> Option<ChunkOpts> {
        Some(self.data().chunk_opts)
    }

    // get_* aliases (parity with the FFI crate)

    pub fn get_filters(&self) -> Result<Vec<Filter>> {
        Ok(self.filters())
    }

    pub fn get_alloc_time(&self) -> Result<AllocTime> {
        Ok(self.alloc_time())
    }

    pub fn get_fill_time(&self) -> Result<FillTime> {
        Ok(self.fill_time())
    }

    pub fn get_fill_value_defined(&self) -> Result<FillValue> {
        Ok(self.fill_value_defined())
    }

    pub fn get_fill_value(&self, tp: &hdf5_types::TypeDescriptor) -> Result<Option<OwnedDynValue>> {
        self.fill_value(tp)
    }

    pub fn get_fill_value_as<T: crate::H5Type + Clone>(&self) -> Result<Option<T>> {
        self.fill_value_as::<T>()
    }

    pub fn get_chunk(&self) -> Result<Option<Vec<usize>>> {
        Ok(self.chunk())
    }

    pub fn get_layout(&self) -> Result<Layout> {
        Ok(self.layout())
    }

    pub fn get_chunk_opts(&self) -> Result<Option<ChunkOpts>> {
        Ok(self.chunk_opts())
    }

    pub fn get_external(&self) -> Result<Vec<ExternalFile>> {
        Ok(self.external())
    }

    pub fn get_virtual_map(&self) -> Result<Vec<VirtualMapping>> {
        Ok(self.virtual_map())
    }

    pub fn all_filters_avail(&self) -> bool {
        self.filters().iter().all(|f| f.is_available())
    }

    pub fn external(&self) -> Vec<ExternalFile> {
        self.data().external
    }

    pub fn virtual_map(&self) -> Vec<VirtualMapping> {
        self.data().virtual_maps
    }

    pub fn obj_track_times(&self) -> bool {
        self.data().obj_track_times
    }

    pub fn attr_phase_change(&self) -> AttrPhaseChange {
        self.data().attr_phase_change
    }

    pub fn attr_creation_order(&self) -> AttrCreationOrder {
        self.data().attr_creation_order
    }
}

/// Builder for dataset creation property lists.
#[derive(Clone, Debug, Default)]
pub struct DatasetCreateBuilder {
    data: DatasetCreateData,
}

impl DatasetCreateBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_plist(plist: &DatasetCreate) -> Result<Self> {
        Ok(Self { data: plist.data() })
    }

    pub fn set_filters(&mut self, filters: &[Filter]) -> &mut Self {
        self.data.filters = filters.to_vec();
        self
    }

    pub fn deflate(&mut self, level: u8) -> &mut Self {
        self.data.filters.push(Filter::deflate(level));
        self
    }

    pub fn shuffle(&mut self) -> &mut Self {
        self.data.filters.push(Filter::shuffle());
        self
    }

    pub fn fletcher32(&mut self) -> &mut Self {
        self.data.filters.push(Filter::fletcher32());
        self
    }

    pub fn szip(&mut self, coding: SZip, px_per_block: u8) -> &mut Self {
        self.data.filters.push(Filter::szip(coding, px_per_block));
        self
    }

    pub fn nbit(&mut self) -> &mut Self {
        self.data.filters.push(Filter::nbit());
        self
    }

    pub fn lzf(&mut self) -> &mut Self {
        self.data.filters.push(Filter::lzf());
        self
    }

    pub fn blosc<T>(&mut self, complib: Blosc, clevel: u8, shuffle: T) -> &mut Self
    where
        T: Into<BloscShuffle>,
    {
        self.data
            .filters
            .push(Filter::blosc(complib, clevel, shuffle));
        self
    }

    pub fn blosc_blosclz<T: Into<BloscShuffle>>(&mut self, clevel: u8, shuffle: T) -> &mut Self {
        self.blosc(Blosc::BloscLZ, clevel, shuffle)
    }

    pub fn blosc_lz4<T: Into<BloscShuffle>>(&mut self, clevel: u8, shuffle: T) -> &mut Self {
        self.blosc(Blosc::LZ4, clevel, shuffle)
    }

    pub fn blosc_lz4hc<T: Into<BloscShuffle>>(&mut self, clevel: u8, shuffle: T) -> &mut Self {
        self.blosc(Blosc::LZ4HC, clevel, shuffle)
    }

    pub fn blosc_snappy<T: Into<BloscShuffle>>(&mut self, clevel: u8, shuffle: T) -> &mut Self {
        self.blosc(Blosc::Snappy, clevel, shuffle)
    }

    pub fn blosc_zlib<T: Into<BloscShuffle>>(&mut self, clevel: u8, shuffle: T) -> &mut Self {
        self.blosc(Blosc::ZLib, clevel, shuffle)
    }

    pub fn blosc_zstd<T: Into<BloscShuffle>>(&mut self, clevel: u8, shuffle: T) -> &mut Self {
        self.blosc(Blosc::ZStd, clevel, shuffle)
    }

    pub fn scale_offset(&mut self, mode: ScaleOffset) -> &mut Self {
        self.data.filters.push(Filter::scale_offset(mode));
        self
    }

    pub fn add_filter(&mut self, id: crate::hl::filters::H5Z_filter_t, cdata: &[u32]) -> &mut Self {
        self.data.filters.push(Filter::user(id, cdata));
        self
    }

    pub fn clear_filters(&mut self) -> &mut Self {
        self.data.filters.clear();
        self
    }

    pub fn alloc_time(&mut self, alloc_time: Option<AllocTime>) -> &mut Self {
        self.data.alloc_time = alloc_time;
        self
    }

    pub fn fill_time(&mut self, fill_time: FillTime) -> &mut Self {
        self.data.fill_time = fill_time;
        self
    }

    pub fn fill_value<T: Into<OwnedDynValue>>(&mut self, fill_value: T) -> &mut Self {
        self.data.fill_value = Some(fill_value.into());
        self
    }

    pub fn no_fill_value(&mut self) -> &mut Self {
        self.data.fill_value = None;
        self
    }

    pub fn chunk<D: Dimension>(&mut self, chunk: D) -> &mut Self {
        self.data.chunk = Some(chunk.dims());
        self.data.layout = Layout::Chunked;
        self
    }

    pub fn no_chunk(&mut self) -> &mut Self {
        self.data.chunk = None;
        if self.data.layout == Layout::Chunked {
            self.data.layout = Layout::Contiguous;
        }
        self
    }

    pub fn layout(&mut self, layout: Layout) -> &mut Self {
        self.data.layout = layout;
        self
    }

    pub fn chunk_opts(&mut self, opts: ChunkOpts) -> &mut Self {
        self.data.chunk_opts = opts;
        self
    }

    pub fn external(&mut self, name: &str, offset: usize, size: usize) -> &mut Self {
        self.data.external.push(ExternalFile {
            name: name.into(),
            offset,
            size,
        });
        self
    }

    #[allow(clippy::too_many_arguments)]
    pub fn virtual_map<F, D, E1, S1, E2, S2>(
        &mut self,
        src_filename: F,
        src_dataset: D,
        src_extents: E1,
        src_selection: S1,
        vds_extents: E2,
        vds_selection: S2,
    ) -> &mut Self
    where
        F: AsRef<str>,
        D: AsRef<str>,
        E1: Into<crate::hl::extents::Extents>,
        S1: Into<crate::hl::selection::Selection>,
        E2: Into<crate::hl::extents::Extents>,
        S2: Into<crate::hl::selection::Selection>,
    {
        self.data.virtual_maps.push(VirtualMapping {
            src_filename: src_filename.as_ref().to_string(),
            src_dataset: src_dataset.as_ref().to_string(),
            src_extents: src_extents.into(),
            src_selection: src_selection.into(),
            vds_extents: vds_extents.into(),
            vds_selection: vds_selection.into(),
        });
        self.data.layout = Layout::Virtual;
        self
    }

    pub fn obj_track_times(&mut self, track_times: bool) -> &mut Self {
        self.data.obj_track_times = track_times;
        self
    }

    pub fn attr_phase_change(&mut self, max_compact: u32, min_dense: u32) -> &mut Self {
        self.data.attr_phase_change = AttrPhaseChange {
            max_compact,
            min_dense,
        };
        self
    }

    pub fn attr_creation_order(&mut self, attr_creation_order: AttrCreationOrder) -> &mut Self {
        self.data.attr_creation_order = attr_creation_order;
        self
    }

    pub fn apply(&self, plist: &mut DatasetCreate) -> Result<()> {
        *plist = DatasetCreate::from_data(self.data.clone());
        Ok(())
    }

    pub fn finish(&self) -> Result<DatasetCreate> {
        for f in &self.data.filters {
            f.validate_writable()?;
        }
        Ok(DatasetCreate::from_data(self.data.clone()))
    }
}
