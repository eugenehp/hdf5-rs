//! HDF5 datasets and their builders.

use std::fmt::{self, Debug};
use std::ops::Deref;

use ndarray::ArrayView;

use hdf5_types::{H5Type, OwnedDynValue, TypeDescriptor};

use crate::class::ObjectClass;
use crate::dim::Dimension;
use crate::error::Result;
use crate::format::convert::{disk_size, to_disk_repr, VlenStore};
use crate::h5i::H5I_type_t;
use crate::handle::{Handle, Payload};
use crate::hl::container::Container;
use crate::hl::datatype::Conversion;
use crate::hl::extents::{Extents, Ix};
use crate::hl::filters::{Blosc, BloscShuffle, Filter, SZip, ScaleOffset};
use crate::hl::group::Group;
use crate::hl::plist::dataset_access::{DatasetAccess, DatasetAccessBuilder};
use crate::hl::plist::dataset_create::{
    AllocTime, DatasetCreate, DatasetCreateBuilder, FillTime, Layout,
};
use crate::hl::plist::link_create::LinkCreateBuilder;

/// Default chunk size hint (parity constant).
pub const DEFAULT_CHUNK_SIZE_KB: usize = 64 * 1024;

use crate::model::{
    DatasetData, FillValue as ModelFill, LayoutClass, Link, LinkTarget, ObjectKind,
};

/// Chunking strategy for a new dataset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Chunk {
    /// Chunk shape is exactly this.
    Exact(Vec<Ix>),
    /// Chunk shape is computed to target a minimum size in KB.
    MinKB(usize),
    /// No chunking.
    None,
}

/// An HDF5 dataset.
#[repr(transparent)]
#[derive(Clone)]
pub struct Dataset(pub(crate) Handle);

impl ObjectClass for Dataset {
    const NAME: &'static str = "dataset";
    const VALID_TYPES: &'static [H5I_type_t] = &[H5I_type_t::H5I_DATASET];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }

    fn short_repr(&self) -> Option<String> {
        Some(format!("\"{}\"", self.name()))
    }
}

impl Debug for Dataset {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.debug_fmt(f)
    }
}

impl Deref for Dataset {
    type Target = Container;

    fn deref(&self) -> &Container {
        unsafe { self.transmute() }
    }
}

impl Dataset {
    fn with_dataset<R>(&self, f: impl FnOnce(&DatasetData) -> R) -> Result<R> {
        let file = self.0.file().ok_or("dataset is not file-resident")?;
        let id = self.0.obj_id().ok_or("dataset has no location")?;
        let state = file.state.read();
        state
            .dataset_data(id)
            .map(f)
            .ok_or_else(|| "object is not a dataset".into())
    }

    /// Returns a copy of the dataset access property list.
    pub fn access_plist(&self) -> Result<DatasetAccess> {
        DatasetAccess::try_new()
    }

    /// A short alias for `access_plist()`.
    pub fn dapl(&self) -> Result<DatasetAccess> {
        self.access_plist()
    }

    /// Returns a copy of the dataset creation property list.
    pub fn create_plist(&self) -> Result<DatasetCreate> {
        self.with_dataset(|d| {
            let mut b = DatasetCreateBuilder::new();
            b.set_filters(&d.filters);
            match &d.layout {
                LayoutClass::Compact => {
                    b.layout(Layout::Compact);
                }
                LayoutClass::Contiguous => {
                    b.layout(Layout::Contiguous);
                }
                LayoutClass::Chunked(c) => {
                    b.chunk(c.iter().map(|&x| x as usize).collect::<Vec<_>>());
                }
                LayoutClass::Virtual(_) => {
                    b.layout(Layout::Virtual);
                }
            }
            match &d.fill {
                ModelFill::Undefined | ModelFill::Default => {}
                ModelFill::UserDefined(bytes) => {
                    // reconstruct a dynamic value from raw bytes when possible
                    if let Ok(v) = fill_from_bytes(&d.dtype, bytes) {
                        b.fill_value(v);
                    }
                }
            }
            b.finish()
        })?
    }

    /// A short alias for `create_plist()`.
    pub fn dcpl(&self) -> Result<DatasetCreate> {
        self.create_plist()
    }

    /// Returns `true` if this dataset is resizable along at least one axis.
    pub fn is_resizable(&self) -> bool {
        self.with_dataset(|d| d.maxdims.iter().any(|m| m.is_none()))
            .unwrap_or(false)
    }

    /// Returns `true` if this dataset has a chunked layout.
    pub fn is_chunked(&self) -> bool {
        self.with_dataset(|d| matches!(d.layout, LayoutClass::Chunked(_)))
            .unwrap_or(false)
    }

    /// Returns the dataset layout.
    pub fn layout(&self) -> Layout {
        self.with_dataset(|d| match &d.layout {
            LayoutClass::Compact => Layout::Compact,
            LayoutClass::Contiguous => Layout::Contiguous,
            LayoutClass::Chunked(_) => Layout::Chunked,
            LayoutClass::Virtual(_) => Layout::Virtual,
        })
        .unwrap_or(Layout::Contiguous)
    }

    /// Returns the number of chunks used by this dataset, if chunked.
    pub fn num_chunks(&self) -> Option<usize> {
        self.with_dataset(|d| match &d.layout {
            LayoutClass::Chunked(c) => Some(
                d.dims
                    .iter()
                    .zip(c.iter())
                    .map(|(&dim, &ch)| ((dim).div_ceil(ch)).max(1) as usize)
                    .product(),
            ),
            _ => None,
        })
        .ok()
        .flatten()
    }

    /// Returns the chunk shape, if chunked.
    pub fn chunk(&self) -> Option<Vec<Ix>> {
        self.with_dataset(|d| match &d.layout {
            LayoutClass::Chunked(c) => Some(c.iter().map(|&x| x as usize).collect()),
            _ => None,
        })
        .ok()
        .flatten()
    }

    /// Returns the address of the dataset's data within the file, if known.
    ///
    /// The pure-Rust engine only assigns addresses at serialization time, so
    /// this always returns `None`.
    pub fn offset(&self) -> Option<u64> {
        None
    }

    /// Returns the fill value, if defined.
    pub fn fill_value(&self) -> Result<Option<OwnedDynValue>> {
        self.with_dataset(|d| match &d.fill {
            ModelFill::UserDefined(bytes) => fill_from_bytes(&d.dtype, bytes).map(Some),
            _ => Ok(None),
        })?
    }

    /// Resizes the dataset to new dimensions (within `maxdims` bounds).
    pub fn resize<D: Dimension>(&self, shape: D) -> Result<()> {
        let file = self.0.file().ok_or("dataset is not file-resident")?.clone();
        let id = self.0.obj_id().ok_or("dataset has no location")?;
        let new_dims: Vec<u64> = shape.dims().iter().map(|&d| d as u64).collect();
        let mut state = file.state.write();
        if state.read_only {
            return Err("unable to resize: file is read-only".into());
        }
        match &mut state.get_mut(id).kind {
            ObjectKind::Dataset(d) => {
                if new_dims.len() != d.dims.len() {
                    return Err(format!(
                        "resize: rank mismatch ({} != {})",
                        new_dims.len(),
                        d.dims.len()
                    )
                    .into());
                }
                if !matches!(d.layout, LayoutClass::Chunked(_)) {
                    return Err("only chunked datasets can be resized".into());
                }
                for (i, (&nd, m)) in new_dims.iter().zip(&d.maxdims).enumerate() {
                    if let Some(m) = m {
                        if nd > *m {
                            return Err(
                                format!("resize: dim {i} exceeds maximum ({nd} > {m})").into()
                            );
                        }
                    }
                }
                // Reshape the row-major data buffer in place.
                let esize = disk_size(&d.dtype);
                let old_dims = d.dims.clone();
                let new_len: usize = new_dims.iter().product::<u64>() as usize * esize;
                let mut new_data = vec![0u8; new_len];
                copy_overlap(&d.data, &old_dims, &mut new_data, &new_dims, esize);
                d.data = new_data;
                d.dims = new_dims;
                Ok(())
            }
            _ => Err("object is not a dataset".into()),
        }
    }

    /// Returns all filters attached to this dataset.
    pub fn filters(&self) -> Vec<Filter> {
        self.with_dataset(|d| d.filters.clone()).unwrap_or_default()
    }
}

/// Reconstruct an `OwnedDynValue` for simple scalar fill values.
fn fill_from_bytes(dtype: &TypeDescriptor, bytes: &[u8]) -> Result<OwnedDynValue> {
    use hdf5_types::{FloatSize, IntSize};
    macro_rules! from_le {
        ($t:ty) => {{
            let mut a = [0u8; std::mem::size_of::<$t>()];
            let n = a.len().min(bytes.len());
            a[..n].copy_from_slice(&bytes[..n]);
            OwnedDynValue::new(<$t>::from_le_bytes(a))
        }};
    }
    Ok(match dtype {
        TypeDescriptor::Integer(IntSize::U1) => from_le!(i8),
        TypeDescriptor::Integer(IntSize::U2) => from_le!(i16),
        TypeDescriptor::Integer(IntSize::U4) => from_le!(i32),
        TypeDescriptor::Integer(IntSize::U8) => from_le!(i64),
        TypeDescriptor::Unsigned(IntSize::U1) => from_le!(u8),
        TypeDescriptor::Unsigned(IntSize::U2) => from_le!(u16),
        TypeDescriptor::Unsigned(IntSize::U4) => from_le!(u32),
        TypeDescriptor::Unsigned(IntSize::U8) => from_le!(u64),
        TypeDescriptor::Float(FloatSize::U4) => from_le!(f32),
        TypeDescriptor::Float(FloatSize::U8) => from_le!(f64),
        TypeDescriptor::Boolean => OwnedDynValue::new(bytes.first().copied().unwrap_or(0) != 0),
        _ => return Err("unsupported fill value type".into()),
    })
}

/// Copy the overlapping region between two row-major buffers of different shapes.
fn copy_overlap(src: &[u8], src_dims: &[u64], dst: &mut [u8], dst_dims: &[u64], esize: usize) {
    let rank = src_dims.len();
    if rank == 0 {
        let n = esize.min(src.len()).min(dst.len());
        dst[..n].copy_from_slice(&src[..n]);
        return;
    }
    let overlap: Vec<u64> = src_dims
        .iter()
        .zip(dst_dims)
        .map(|(&a, &b)| a.min(b))
        .collect();
    if overlap.contains(&0) {
        return;
    }
    let mut sstr = vec![1u64; rank];
    let mut dstr = vec![1u64; rank];
    for i in (0..rank - 1).rev() {
        sstr[i] = sstr[i + 1] * src_dims[i + 1];
        dstr[i] = dstr[i + 1] * dst_dims[i + 1];
    }
    let total: u64 = overlap.iter().product();
    let mut coord = vec![0u64; rank];
    for _ in 0..total {
        let mut s = 0u64;
        let mut d = 0u64;
        for i in 0..rank {
            s += coord[i] * sstr[i];
            d += coord[i] * dstr[i];
        }
        let (s, d) = ((s as usize) * esize, (d as usize) * esize);
        if s + esize <= src.len() && d + esize <= dst.len() {
            dst[d..d + esize].copy_from_slice(&src[s..s + esize]);
        }
        for i in (0..rank).rev() {
            coord[i] += 1;
            if coord[i] < overlap[i] {
                break;
            }
            coord[i] = 0;
        }
    }
}

/// Shared builder state for new datasets.
#[derive(Clone)]
pub struct DatasetBuilder {
    parent: Result<Handle>,
    dcpl: DatasetCreateBuilder,
    dapl: DatasetAccessBuilder,
    lcpl: LinkCreateBuilder,
    packed: bool,
    chunk: Option<Chunk>,
}

impl DatasetBuilder {
    /// Returns the link-creation property list builder for this dataset.
    pub fn link_create_plist(&mut self) -> &mut LinkCreateBuilder {
        &mut self.lcpl
    }

    /// Alias for [`Self::link_create_plist`].
    pub fn lcpl(&mut self) -> &mut LinkCreateBuilder {
        &mut self.lcpl
    }

    pub fn set_link_create_plist(&mut self, lcpl: &crate::plist::LinkCreate) -> &mut Self {
        self.lcpl = LinkCreateBuilder::from_plist(lcpl).unwrap_or_default();
        self
    }

    /// Alias for [`Self::set_link_create_plist`].
    pub fn set_lcpl(&mut self, lcpl: &crate::plist::LinkCreate) -> &mut Self {
        self.set_link_create_plist(lcpl)
    }

    pub fn with_link_create_plist<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&mut LinkCreateBuilder) -> &mut LinkCreateBuilder,
    {
        f(&mut self.lcpl);
        self
    }

    /// Alias for [`Self::with_link_create_plist`].
    pub fn with_lcpl<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&mut LinkCreateBuilder) -> &mut LinkCreateBuilder,
    {
        self.with_link_create_plist(f)
    }

    pub fn new(parent: &Group) -> Self {
        Self {
            parent: parent.try_borrow(),
            dcpl: DatasetCreateBuilder::new(),
            dapl: DatasetAccessBuilder::new(),
            lcpl: LinkCreateBuilder::new(),
            packed: false,
            chunk: None,
        }
    }

    pub fn packed(mut self, packed: bool) -> Self {
        self.packed = packed;
        self
    }

    /// Datatype fixed to `T`; shape given later.
    pub fn empty<T: H5Type>(self) -> DatasetBuilderEmpty {
        self.empty_as(&T::type_descriptor())
    }

    /// Datatype given as a descriptor; shape given later.
    pub fn empty_as(self, type_desc: &TypeDescriptor) -> DatasetBuilderEmpty {
        DatasetBuilderEmpty {
            builder: self,
            type_desc: type_desc.clone(),
            extents: Extents::Scalar,
        }
    }

    /// Data (and shape/type) given by an ndarray.
    pub fn with_data<'d, A, T, D>(self, data: A) -> DatasetBuilderData<'d, T, D>
    where
        A: Into<ArrayView<'d, T, D>>,
        T: H5Type,
        D: ndarray::Dimension,
    {
        let view = data.into();
        let type_desc = T::type_descriptor();
        DatasetBuilderData {
            builder: self,
            data: view,
            type_desc,
            conv: Conversion::Soft,
        }
    }

    /// Data given by an ndarray with an explicit file datatype.
    pub fn with_data_as<'d, A, T, D>(
        self,
        data: A,
        type_desc: &TypeDescriptor,
    ) -> DatasetBuilderData<'d, T, D>
    where
        A: Into<ArrayView<'d, T, D>>,
        T: H5Type,
        D: ndarray::Dimension,
    {
        let view = data.into();
        DatasetBuilderData {
            builder: self,
            data: view,
            type_desc: type_desc.clone(),
            conv: Conversion::Soft,
        }
    }

    // --- dcpl/dapl/lcpl delegation ---

    pub fn set_filters(mut self, filters: &[Filter]) -> Self {
        self.dcpl.set_filters(filters);
        self
    }

    pub fn deflate(mut self, level: u8) -> Self {
        self.dcpl.deflate(level);
        self
    }

    pub fn shuffle(mut self) -> Self {
        self.dcpl.shuffle();
        self
    }

    pub fn fletcher32(mut self) -> Self {
        self.dcpl.fletcher32();
        self
    }

    pub fn szip(mut self, coding: SZip, px_per_block: u8) -> Self {
        self.dcpl.szip(coding, px_per_block);
        self
    }

    pub fn nbit(mut self) -> Self {
        self.dcpl.nbit();
        self
    }

    pub fn lzf(mut self) -> Self {
        self.dcpl.lzf();
        self
    }

    pub fn blosc<T: Into<BloscShuffle>>(mut self, complib: Blosc, clevel: u8, shuffle: T) -> Self {
        self.dcpl.blosc(complib, clevel, shuffle);
        self
    }

    pub fn blosc_blosclz<T: Into<BloscShuffle>>(mut self, clevel: u8, shuffle: T) -> Self {
        self.dcpl.blosc_blosclz(clevel, shuffle);
        self
    }

    pub fn blosc_lz4<T: Into<BloscShuffle>>(mut self, clevel: u8, shuffle: T) -> Self {
        self.dcpl.blosc_lz4(clevel, shuffle);
        self
    }

    pub fn blosc_lz4hc<T: Into<BloscShuffle>>(mut self, clevel: u8, shuffle: T) -> Self {
        self.dcpl.blosc_lz4hc(clevel, shuffle);
        self
    }

    pub fn blosc_snappy<T: Into<BloscShuffle>>(mut self, clevel: u8, shuffle: T) -> Self {
        self.dcpl.blosc_snappy(clevel, shuffle);
        self
    }

    pub fn blosc_zlib<T: Into<BloscShuffle>>(mut self, clevel: u8, shuffle: T) -> Self {
        self.dcpl.blosc_zlib(clevel, shuffle);
        self
    }

    pub fn blosc_zstd<T: Into<BloscShuffle>>(mut self, clevel: u8, shuffle: T) -> Self {
        self.dcpl.blosc_zstd(clevel, shuffle);
        self
    }

    pub fn scale_offset(mut self, mode: ScaleOffset) -> Self {
        self.dcpl.scale_offset(mode);
        self
    }

    pub fn add_filter(mut self, id: crate::hl::filters::H5Z_filter_t, cdata: &[u32]) -> Self {
        self.dcpl.add_filter(id, cdata);
        self
    }

    pub fn clear_filters(mut self) -> Self {
        self.dcpl.clear_filters();
        self
    }

    pub fn alloc_time(mut self, alloc_time: Option<AllocTime>) -> Self {
        self.dcpl.alloc_time(alloc_time);
        self
    }

    pub fn fill_time(mut self, fill_time: FillTime) -> Self {
        self.dcpl.fill_time(fill_time);
        self
    }

    pub fn fill_value<V: Into<OwnedDynValue>>(mut self, fill_value: V) -> Self {
        self.dcpl.fill_value(fill_value);
        self
    }

    pub fn no_fill_value(mut self) -> Self {
        self.dcpl.no_fill_value();
        self
    }

    pub fn chunk<D: Dimension>(mut self, chunk: D) -> Self {
        self.chunk = Some(Chunk::Exact(chunk.dims()));
        self
    }

    pub fn chunk_min_kb(mut self, size: usize) -> Self {
        self.chunk = Some(Chunk::MinKB(size));
        self
    }

    pub fn no_chunk(mut self) -> Self {
        self.chunk = Some(Chunk::None);
        self
    }

    pub fn layout(mut self, layout: Layout) -> Self {
        self.dcpl.layout(layout);
        self
    }

    pub fn obj_track_times(mut self, track_times: bool) -> Self {
        self.dcpl.obj_track_times(track_times);
        self
    }

    pub fn attr_phase_change(mut self, max_compact: u32, min_dense: u32) -> Self {
        self.dcpl.attr_phase_change(max_compact, min_dense);
        self
    }

    pub fn attr_creation_order(
        mut self,
        order: crate::hl::plist::common::AttrCreationOrder,
    ) -> Self {
        self.dcpl.attr_creation_order(order);
        self
    }

    pub fn chunk_cache(mut self, nslots: usize, nbytes: usize, w0: f64) -> Self {
        self.dapl.chunk_cache(nslots, nbytes, w0);
        self
    }

    pub fn efile_prefix(mut self, prefix: &str) -> Self {
        self.dapl.efile_prefix(prefix);
        self
    }

    pub fn create_intermediate_group(mut self, create: bool) -> Self {
        self.lcpl.create_intermediate_group(create);
        self
    }

    /// Alias for [`Self::set_create_plist`].
    pub fn set_dcpl(self, dcpl: &DatasetCreate) -> Self {
        self.set_create_plist(dcpl)
    }

    /// Configure the dataset-creation plist via a closure.
    pub fn with_dcpl<F>(mut self, f: F) -> Self
    where
        F: Fn(&mut crate::plist::DatasetCreateBuilder) -> &mut crate::plist::DatasetCreateBuilder,
    {
        f(&mut self.dcpl);
        self
    }

    /// Alias for [`Self::set_access_plist`].
    pub fn set_dapl(self, dapl: &DatasetAccess) -> Self {
        self.set_access_plist(dapl)
    }

    /// Configure the dataset-access plist via a closure.
    pub fn with_dapl<F>(mut self, f: F) -> Self
    where
        F: Fn(&mut crate::plist::DatasetAccessBuilder) -> &mut crate::plist::DatasetAccessBuilder,
    {
        f(&mut self.dapl);
        self
    }

    pub fn set_create_plist(mut self, dcpl: &DatasetCreate) -> Self {
        self.dcpl = DatasetCreateBuilder::from_plist(dcpl).unwrap_or_default();
        self
    }

    pub fn set_access_plist(mut self, dapl: &DatasetAccess) -> Self {
        self.dapl = DatasetAccessBuilder::from_plist(dapl).unwrap_or_default();
        self
    }

    /// Create the dataset in the file.
    pub(crate) fn create_dataset(
        &self,
        name: Option<&str>,
        type_desc: &TypeDescriptor,
        extents: &Extents,
    ) -> Result<Dataset> {
        let parent = self.parent.clone()?;
        let file = parent.file().ok_or("parent is not file-resident")?.clone();
        let parent_id = parent.obj_id().ok_or("parent has no location")?;

        let type_desc = if self.packed {
            type_desc.to_packed_repr()
        } else {
            type_desc.clone()
        };
        let disk_desc = to_disk_repr(&type_desc);
        let esize = disk_size(&disk_desc);

        let (dims, maxdims, is_scalar, is_null) = match extents {
            Extents::Null => (vec![], vec![], false, true),
            Extents::Scalar => (vec![], vec![], true, false),
            Extents::Simple(se) => (
                se.dims().iter().map(|&d| d as u64).collect::<Vec<u64>>(),
                se.maxdims()
                    .iter()
                    .map(|m| m.map(|v| v as u64))
                    .collect::<Vec<Option<u64>>>(),
                false,
                false,
            ),
        };

        let dcpl = self.dcpl.finish()?;
        let dcpl_data = dcpl.data();
        if !dcpl_data.external.is_empty() {
            return Err("external storage (H5Pset_external) is not supported; \
                 store the data in the file or use an external link"
                .into());
        }

        // Virtual dataset: mappings only, no storage of its own.
        if !dcpl_data.virtual_maps.is_empty() {
            use crate::format::vds::{SerSelection, VdsMapping};
            use crate::hl::selection::{RawSelection, Selection};
            let to_ser = |sel: &Selection, dims: &[usize]| -> Result<SerSelection> {
                Ok(match sel.clone().into_raw(dims)? {
                    RawSelection::All => SerSelection::All,
                    RawSelection::None => SerSelection::None,
                    RawSelection::RegularHyperslab(h) => SerSelection::Regular(
                        h.iter()
                            .map(|s| {
                                (
                                    s.start as u64,
                                    s.step as u64,
                                    s.count.map(|c| c as u64),
                                    s.block as u64,
                                )
                            })
                            .collect(),
                    ),
                    RawSelection::Points(p) => SerSelection::Points(
                        p.rows()
                            .into_iter()
                            .map(|r| r.iter().map(|&x| x as u64).collect())
                            .collect(),
                    ),
                    RawSelection::ComplexHyperslab => {
                        return Err("complex hyperslabs cannot be written".into())
                    }
                })
            };
            let vdims: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
            let mut mappings = Vec::with_capacity(dcpl_data.virtual_maps.len());
            for vm in &dcpl_data.virtual_maps {
                mappings.push(VdsMapping {
                    source_file: vm.src_filename.clone(),
                    source_dset: vm.src_dataset.clone(),
                    src_sel: to_ser(&vm.src_selection, &vm.src_extents.dims())?,
                    virt_sel: to_ser(&vm.vds_selection, &vdims)?,
                });
            }
            let n_elems: usize = if is_null {
                0
            } else if is_scalar {
                1
            } else {
                dims.iter().product::<u64>() as usize
            };
            let dataset = DatasetData {
                dtype: disk_desc,
                dims,
                maxdims,
                layout: LayoutClass::Virtual(mappings),
                filters: Vec::new(),
                fill: ModelFill::Default,
                data: vec![0u8; n_elems * esize],
                vlen: VlenStore::new(),
                lazy: None,
                is_scalar,
                is_null,
            };
            let mut state = file.state.write();
            if state.read_only {
                return Err("unable to create dataset: file is read-only".into());
            }
            let new_id = state.alloc(ObjectKind::Dataset(dataset));
            state.get_mut(new_id).mtime = crate::model::now();
            if let Some(name) = name {
                let order = state.next_order();
                match &mut state.get_mut(parent_id).kind {
                    ObjectKind::Group(g) => g.links.push(Link {
                        name: name.to_string(),
                        target: LinkTarget::Hard(new_id),
                        creation_order: order,
                        utf8: !name.is_ascii(),
                    }),
                    _ => return Err("parent is not a group".into()),
                }
                state.get_mut(new_id).refcount += 1;
            }
            drop(state);
            return Ok(Dataset::from_handle(Handle::new(Payload::Dataset {
                file,
                id: new_id,
            })));
        }

        // Determine chunking.
        let resizable = maxdims.iter().any(|m| m.is_none());
        let has_filters = !dcpl_data.filters.is_empty();
        let chunk_setting = self
            .chunk
            .clone()
            .or_else(|| dcpl_data.chunk.as_ref().map(|c| Chunk::Exact(c.clone())));
        let layout = match chunk_setting {
            Some(Chunk::None) => {
                if resizable {
                    return Err("resizable datasets require chunking".into());
                }
                if has_filters {
                    return Err("filtered datasets require chunking".into());
                }
                LayoutClass::Contiguous
            }
            Some(Chunk::Exact(c)) => {
                if c.len() != dims.len() {
                    return Err(
                        format!("chunk rank {} != dataset rank {}", c.len(), dims.len()).into(),
                    );
                }
                if c.contains(&0) {
                    return Err("chunk dimensions must be positive".into());
                }
                LayoutClass::Chunked(c.iter().map(|&x| x as u64).collect())
            }
            Some(Chunk::MinKB(kb)) => LayoutClass::Chunked(auto_chunk(&dims, esize, kb * 1024)),
            None => {
                if is_scalar || is_null {
                    if has_filters || resizable {
                        return Err("scalar datasets cannot be chunked".into());
                    }
                    LayoutClass::Contiguous
                } else if has_filters || resizable {
                    // default chunk shape: whole current shape (or 1 in
                    // unlimited zero-sized dims)
                    LayoutClass::Chunked(dims.iter().map(|&d| d.max(1)).collect::<Vec<u64>>())
                } else {
                    LayoutClass::Contiguous
                }
            }
        };

        for f in &dcpl_data.filters {
            f.validate_writable()?;
        }
        if matches!(layout, LayoutClass::Chunked(_)) && (is_scalar || is_null) {
            return Err("scalar datasets cannot be chunked".into());
        }

        // Fill value.
        let n_elems: usize = if is_null {
            0
        } else if is_scalar {
            1
        } else {
            dims.iter().product::<u64>() as usize
        };
        let (fill, init_byte) = match &dcpl_data.fill_value {
            Some(v) => {
                // convert the dynamic value's bytes into the file layout
                let src_desc = v.type_descriptor().clone();
                let raw = dyn_value_bytes(v);
                let mut store = VlenStore::new();
                let converted = crate::format::convert::mem_to_model(
                    &src_desc, &disk_desc, &raw, &mut store, 1,
                )
                .map_err(|e| format!("invalid fill value: {e}"))?;
                (ModelFill::UserDefined(converted), None)
            }
            None => (ModelFill::Default, Some(0u8)),
        };
        let mut data = vec![init_byte.unwrap_or(0); n_elems * esize];
        if let ModelFill::UserDefined(fv) = &fill {
            for i in 0..n_elems {
                data[i * esize..(i + 1) * esize].copy_from_slice(fv);
            }
        }

        let dataset = DatasetData {
            dtype: disk_desc,
            dims,
            maxdims,
            layout,
            filters: dcpl_data.filters.clone(),
            fill,
            data,
            vlen: VlenStore::new(),
            lazy: None,
            is_scalar,
            is_null,
        };

        let mut state = file.state.write();
        if state.read_only {
            return Err("unable to create dataset: file is read-only".into());
        }
        let new_id = state.alloc(ObjectKind::Dataset(dataset));
        state.get_mut(new_id).mtime = crate::model::now();
        if let Some(name) = name {
            // resolve intermediate path
            let (parent_id, leaf) = match name.rfind('/') {
                Some(pos) => {
                    let (dir, leaf) = name.split_at(pos);
                    let dir = if dir.is_empty() { "/" } else { dir };
                    let pid = state
                        .resolve(parent_id, dir)
                        .ok_or_else(|| format!("parent group '{dir}' not found"))?;
                    (pid, &leaf[1..])
                }
                None => (parent_id, name),
            };
            if state
                .group_data(parent_id)
                .map(|g| g.find(leaf).is_some())
                .unwrap_or(false)
            {
                return Err(format!("unable to create dataset: '{leaf}' already exists").into());
            }
            let order = state.next_order();
            match &mut state.get_mut(parent_id).kind {
                ObjectKind::Group(g) => g.links.push(Link {
                    name: leaf.to_string(),
                    target: LinkTarget::Hard(new_id),
                    creation_order: order,
                    utf8: !leaf.is_ascii(),
                }),
                _ => return Err("parent is not a group".into()),
            }
            state.get_mut(new_id).refcount += 1;
        }
        drop(state);
        Ok(Dataset::from_handle(Handle::new(Payload::Dataset {
            file,
            id: new_id,
        })))
    }
}

/// Extract the little-endian bytes of a scalar `OwnedDynValue`.
///
/// `hdf5-types` does not expose the value's internal buffer, so the value is
/// recovered through its typed `cast` API; only fixed-size scalar fill values
/// are supported (matching what can be reconstructed on read).
fn dyn_value_bytes(v: &OwnedDynValue) -> Vec<u8> {
    let size = v.type_descriptor().size();
    let mut out = vec![0u8; size];
    macro_rules! try_cast {
        ($($t:ty),*) => {$(
            if let Ok(x) = v.clone().cast::<$t>() {
                let bytes = x.to_le_bytes();
                out[..bytes.len()].copy_from_slice(&bytes);
                return out;
            }
        )*};
    }
    try_cast!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
    if let Ok(x) = v.clone().cast::<bool>() {
        out[0] = u8::from(x);
    }
    out
}

/// Compute a chunk shape targeting `min_bytes` per chunk.
fn auto_chunk(dims: &[u64], esize: usize, min_bytes: usize) -> Vec<u64> {
    let mut chunk: Vec<u64> = dims.iter().map(|&d| d.max(1)).collect();
    loop {
        let bytes = chunk.iter().product::<u64>() as usize * esize;
        if bytes <= min_bytes.max(esize) {
            break;
        }
        // halve the largest dimension
        if let Some((i, _)) = chunk.iter().enumerate().max_by_key(|(_, &c)| c) {
            if chunk[i] <= 1 {
                break;
            }
            chunk[i] = chunk[i].div_ceil(2);
        } else {
            break;
        }
    }
    chunk
}

/// Dataset builder with a datatype chosen (shape defaults to scalar).
pub struct DatasetBuilderEmpty {
    builder: DatasetBuilder,
    type_desc: TypeDescriptor,
    extents: Extents,
}

impl DatasetBuilderEmpty {
    /// Sets the dataset shape.
    pub fn shape<S: Into<Extents>>(self, extents: S) -> DatasetBuilderEmptyShape {
        DatasetBuilderEmptyShape {
            builder: self.builder,
            type_desc: self.type_desc,
            extents: extents.into(),
        }
    }

    /// Creates the (scalar) dataset.
    pub fn create<'n, T: Into<Maybe<&'n str>>>(self, name: T) -> Result<Dataset> {
        let name: Maybe<&str> = name.into();
        self.builder
            .create_dataset(name.into(), &self.type_desc, &self.extents)
    }
}

// dcpl delegation for the typed builder stages
macro_rules! delegate_builder {
    ($ty:ident) => {
        impl $ty {
            pub fn set_filters(mut self, filters: &[Filter]) -> Self {
                self.builder = self.builder.set_filters(filters);
                self
            }
            pub fn deflate(mut self, level: u8) -> Self {
                self.builder = self.builder.deflate(level);
                self
            }
            pub fn shuffle(mut self) -> Self {
                self.builder = self.builder.shuffle();
                self
            }
            pub fn fletcher32(mut self) -> Self {
                self.builder = self.builder.fletcher32();
                self
            }
            pub fn lzf(mut self) -> Self {
                self.builder = self.builder.lzf();
                self
            }
            pub fn blosc<T: Into<BloscShuffle>>(
                mut self,
                complib: Blosc,
                clevel: u8,
                shuffle: T,
            ) -> Self {
                self.builder = self.builder.blosc(complib, clevel, shuffle);
                self
            }
            pub fn blosc_blosclz<T: Into<BloscShuffle>>(mut self, clevel: u8, shuffle: T) -> Self {
                self.builder = self.builder.blosc_blosclz(clevel, shuffle);
                self
            }
            pub fn blosc_lz4<T: Into<BloscShuffle>>(mut self, clevel: u8, shuffle: T) -> Self {
                self.builder = self.builder.blosc_lz4(clevel, shuffle);
                self
            }
            pub fn blosc_lz4hc<T: Into<BloscShuffle>>(mut self, clevel: u8, shuffle: T) -> Self {
                self.builder = self.builder.blosc_lz4hc(clevel, shuffle);
                self
            }
            pub fn blosc_snappy<T: Into<BloscShuffle>>(mut self, clevel: u8, shuffle: T) -> Self {
                self.builder = self.builder.blosc_snappy(clevel, shuffle);
                self
            }
            pub fn blosc_zlib<T: Into<BloscShuffle>>(mut self, clevel: u8, shuffle: T) -> Self {
                self.builder = self.builder.blosc_zlib(clevel, shuffle);
                self
            }
            pub fn blosc_zstd<T: Into<BloscShuffle>>(mut self, clevel: u8, shuffle: T) -> Self {
                self.builder = self.builder.blosc_zstd(clevel, shuffle);
                self
            }
            pub fn chunk<D: Dimension>(mut self, chunk: D) -> Self {
                self.builder = self.builder.chunk(chunk);
                self
            }
            pub fn chunk_min_kb(mut self, size: usize) -> Self {
                self.builder = self.builder.chunk_min_kb(size);
                self
            }
            pub fn no_chunk(mut self) -> Self {
                self.builder = self.builder.no_chunk();
                self
            }
            pub fn fill_value<V: Into<OwnedDynValue>>(mut self, fill_value: V) -> Self {
                self.builder = self.builder.fill_value(fill_value);
                self
            }
            pub fn no_fill_value(mut self) -> Self {
                self.builder = self.builder.no_fill_value();
                self
            }
            pub fn packed(mut self, packed: bool) -> Self {
                self.builder = self.builder.packed(packed);
                self
            }
        }
    };
}

delegate_builder!(DatasetBuilderEmpty);

/// Dataset builder with datatype and shape chosen.
pub struct DatasetBuilderEmptyShape {
    builder: DatasetBuilder,
    type_desc: TypeDescriptor,
    extents: Extents,
}

impl DatasetBuilderEmptyShape {
    pub fn create<'n, T: Into<Maybe<&'n str>>>(&self, name: T) -> Result<Dataset> {
        let name: Maybe<&str> = name.into();
        self.builder
            .create_dataset(name.into(), &self.type_desc, &self.extents)
    }
}

delegate_builder!(DatasetBuilderEmptyShape);

/// Dataset builder holding the data to write on creation.
pub struct DatasetBuilderData<'d, T, D> {
    builder: DatasetBuilder,
    data: ArrayView<'d, T, D>,
    type_desc: TypeDescriptor,
    conv: Conversion,
}

impl<'d, T, D> DatasetBuilderData<'d, T, D>
where
    T: H5Type,
    D: ndarray::Dimension,
{
    /// Set the maximum allowed conversion level.
    pub fn conversion(mut self, conv: Conversion) -> Self {
        self.conv = conv;
        self
    }

    /// Disallow all type conversions.
    pub fn no_convert(mut self) -> Self {
        self.conv = Conversion::NoOp;
        self
    }

    pub fn packed(mut self, packed: bool) -> Self {
        self.builder = self.builder.packed(packed);
        self
    }

    // filter/chunk delegation
    pub fn set_filters(mut self, filters: &[Filter]) -> Self {
        self.builder = self.builder.set_filters(filters);
        self
    }
    pub fn deflate(mut self, level: u8) -> Self {
        self.builder = self.builder.deflate(level);
        self
    }
    pub fn shuffle(mut self) -> Self {
        self.builder = self.builder.shuffle();
        self
    }
    pub fn fletcher32(mut self) -> Self {
        self.builder = self.builder.fletcher32();
        self
    }
    pub fn lzf(mut self) -> Self {
        self.builder = self.builder.lzf();
        self
    }
    pub fn blosc<S: Into<BloscShuffle>>(mut self, complib: Blosc, clevel: u8, shuffle: S) -> Self {
        self.builder = self.builder.blosc(complib, clevel, shuffle);
        self
    }
    pub fn blosc_blosclz<S: Into<BloscShuffle>>(mut self, clevel: u8, shuffle: S) -> Self {
        self.builder = self.builder.blosc_blosclz(clevel, shuffle);
        self
    }
    pub fn blosc_lz4<S: Into<BloscShuffle>>(mut self, clevel: u8, shuffle: S) -> Self {
        self.builder = self.builder.blosc_lz4(clevel, shuffle);
        self
    }
    pub fn blosc_lz4hc<S: Into<BloscShuffle>>(mut self, clevel: u8, shuffle: S) -> Self {
        self.builder = self.builder.blosc_lz4hc(clevel, shuffle);
        self
    }
    pub fn blosc_snappy<S: Into<BloscShuffle>>(mut self, clevel: u8, shuffle: S) -> Self {
        self.builder = self.builder.blosc_snappy(clevel, shuffle);
        self
    }
    pub fn blosc_zlib<S: Into<BloscShuffle>>(mut self, clevel: u8, shuffle: S) -> Self {
        self.builder = self.builder.blosc_zlib(clevel, shuffle);
        self
    }
    pub fn blosc_zstd<S: Into<BloscShuffle>>(mut self, clevel: u8, shuffle: S) -> Self {
        self.builder = self.builder.blosc_zstd(clevel, shuffle);
        self
    }
    pub fn chunk<Dim: Dimension>(mut self, chunk: Dim) -> Self {
        self.builder = self.builder.chunk(chunk);
        self
    }
    pub fn chunk_min_kb(mut self, size: usize) -> Self {
        self.builder = self.builder.chunk_min_kb(size);
        self
    }
    pub fn no_chunk(mut self) -> Self {
        self.builder = self.builder.no_chunk();
        self
    }

    /// Creates the dataset and writes the data into it.
    pub fn create<'n, N: Into<Maybe<&'n str>>>(&self, name: N) -> Result<Dataset> {
        let name: Maybe<&str> = name.into();
        let shape: Vec<usize> = self.data.shape().to_vec();
        let extents = Extents::from(&shape[..]);
        let ds = self
            .builder
            .create_dataset(name.into(), &self.type_desc, &extents)?;
        let writer = crate::hl::container::Writer::new(&ds).conversion(self.conv);
        let slice = self
            .data
            .as_slice()
            .ok_or("input array is not contiguous or not in standard layout")?;
        writer.write_raw(slice)?;
        Ok(ds)
    }
}

/// Optional-name helper: allows passing either `&str` or `()` to `create`.
#[derive(Clone, Copy, Debug)]
pub enum Maybe<T> {
    Some(T),
    None,
}

impl<'a> From<&'a str> for Maybe<&'a str> {
    fn from(s: &'a str) -> Self {
        Self::Some(s)
    }
}

impl<T> From<()> for Maybe<T> {
    fn from((): ()) -> Self {
        Self::None
    }
}

impl<T> From<Maybe<T>> for Option<T> {
    fn from(v: Maybe<T>) -> Self {
        match v {
            Maybe::Some(x) => Some(x),
            Maybe::None => None,
        }
    }
}
