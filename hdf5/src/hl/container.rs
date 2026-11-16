//! `Container`: common read/write interface for datasets and attributes.

use std::fmt::{self, Debug};
use std::mem;
use std::ops::Deref;
use std::sync::Arc;

use ndarray::{Array, Array1, Array2, ArrayD, ArrayView, ArrayView1};

use hdf5_types::{H5Type, TypeDescriptor};

use crate::class::ObjectClass;
use crate::error::Result;
use crate::format::convert::{self, disk_size, VlenStore};
use crate::h5i::H5I_type_t;
use crate::handle::{Handle, Payload};
use crate::hl::dataspace::{Dataspace, DataspaceState};
use crate::hl::datatype::{conversion_path, Conversion, Datatype};
use crate::hl::extents::Ix;
use crate::hl::location::Location;
use crate::hl::selection::{RawSelection, Selection};
use crate::model::{FileInner, ObjectKind};

/// A snapshot of a container's metadata and data.
pub(crate) struct ContainerData {
    pub dtype: TypeDescriptor,
    pub dims: Vec<u64>,
    pub is_scalar: bool,
    pub is_null: bool,
    pub data: Vec<u8>,
    pub vlen: VlenStore,
}

impl ContainerData {
    fn num_elements(&self) -> usize {
        if self.is_null {
            0
        } else if self.is_scalar {
            1
        } else {
            self.dims.iter().product::<u64>() as usize
        }
    }
}

/// An object that has both a datatype and a dataspace (dataset or attribute).
#[repr(transparent)]
#[derive(Clone)]
pub struct Container(pub(crate) Handle);

impl ObjectClass for Container {
    const NAME: &'static str = "container";
    const VALID_TYPES: &'static [H5I_type_t] = &[H5I_type_t::H5I_DATASET, H5I_type_t::H5I_ATTR];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }
}

impl Debug for Container {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.debug_fmt(f)
    }
}

impl Deref for Container {
    type Target = Location;

    fn deref(&self) -> &Location {
        unsafe { self.transmute() }
    }
}

impl Container {
    /// Load a snapshot of this container's metadata and raw data.
    pub(crate) fn snapshot(&self) -> Result<ContainerData> {
        self.load(true)
    }

    /// A snapshot of the metadata alone: `data` and `vlen` are left empty, so
    /// asking what a dataset is does not read, or copy, what it holds.
    fn metadata(&self) -> Result<ContainerData> {
        self.load(false)
    }

    fn load(&self, with_data: bool) -> Result<ContainerData> {
        let file = self.0.file().ok_or("container is not file-resident")?;
        match self.0.payload() {
            Payload::Dataset { id, .. } => {
                // materialize lazily-referenced data on first access
                if with_data {
                    let state = file.state.read();
                    let needs = state
                        .dataset_data(*id)
                        .map(|d| d.lazy.is_some())
                        .unwrap_or(false);
                    drop(state);
                    if needs {
                        let mut state = file.state.write();
                        if let crate::model::ObjectKind::Dataset(d) = &mut state.get_mut(*id).kind {
                            d.materialize()?;
                        }
                    }
                }
                let state = file.state.read();
                let d = state.dataset_data(*id).ok_or("not a dataset")?;
                Ok(ContainerData {
                    dtype: d.dtype.clone(),
                    dims: d.dims.clone(),
                    is_scalar: d.is_scalar,
                    is_null: d.is_null,
                    data: if with_data {
                        d.data.clone()
                    } else {
                        Vec::new()
                    },
                    vlen: if with_data {
                        d.vlen.clone()
                    } else {
                        VlenStore::default()
                    },
                })
            }
            Payload::Attribute { owner, name, .. } => {
                let state = file.state.read();
                let node = state.try_get(*owner).ok_or("dangling attribute owner")?;
                let idx = node.attr_index(name).ok_or("attribute not found")?;
                let a = &node.attrs[idx];
                Ok(ContainerData {
                    dtype: a.dtype.clone(),
                    dims: a.dims.clone(),
                    is_scalar: a.is_scalar,
                    is_null: a.is_null,
                    data: a.data.clone(),
                    vlen: a.vlen.clone(),
                })
            }
            _ => Err("object is not a dataset or attribute".into()),
        }
    }

    /// Store new raw data (and vlen store) into this container.
    pub(crate) fn store(&self, data: Vec<u8>, vlen: VlenStore) -> Result<()> {
        let file: &Arc<FileInner> = self.0.file().ok_or("container is not file-resident")?;
        let mut state = file.state.write();
        state.dirty = true;
        if state.read_only {
            return Err("unable to write: file is read-only".into());
        }
        match self.0.payload() {
            Payload::Dataset { id, .. } => match &mut state.get_mut(*id).kind {
                ObjectKind::Dataset(d) => {
                    d.data = data;
                    d.vlen = vlen;
                    Ok(())
                }
                _ => Err("not a dataset".into()),
            },
            Payload::Attribute { owner, name, .. } => {
                let node = state.get_mut(*owner);
                let idx = node.attr_index(name).ok_or("attribute not found")?;
                node.attrs[idx].data = data;
                node.attrs[idx].vlen = vlen;
                Ok(())
            }
            _ => Err("object is not a dataset or attribute".into()),
        }
    }

    /// Returns the datatype of this container.
    pub fn dtype(&self) -> Result<Datatype> {
        let snap = self.metadata()?;
        Datatype::from_descriptor(&snap.dtype)
    }

    /// Returns the dataspace of this container.
    pub fn space(&self) -> Result<Dataspace> {
        let snap = self.metadata()?;
        let extents = if snap.is_null {
            crate::hl::extents::Extents::Null
        } else if snap.is_scalar {
            crate::hl::extents::Extents::Scalar
        } else {
            // preserve maxdims for datasets
            if let Payload::Dataset { file, id } = self.0.payload() {
                let state = file.state.read();
                let d = state.dataset_data(*id).ok_or("not a dataset")?;
                d.extents()
            } else {
                use crate::hl::extents::{Extent, SimpleExtents};
                crate::hl::extents::Extents::Simple(SimpleExtents::from_vec(
                    snap.dims
                        .iter()
                        .map(|&d| Extent::new(d as usize, Some(d as usize)))
                        .collect(),
                ))
            }
        };
        Ok(Dataspace::from_state(DataspaceState {
            extents,
            selection: RawSelection::All,
        }))
    }

    /// Returns the shape of the container's dataspace.
    pub fn shape(&self) -> Vec<Ix> {
        self.metadata()
            .map(|s| s.dims.iter().map(|&d| d as usize).collect())
            .unwrap_or_default()
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.metadata().map(|s| s.dims.len()).unwrap_or(0)
    }

    /// Returns the total number of elements.
    pub fn size(&self) -> usize {
        self.metadata().map(|s| s.num_elements()).unwrap_or(0)
    }

    /// Returns whether the dataspace is scalar.
    pub fn is_scalar(&self) -> bool {
        self.metadata().map(|s| s.is_scalar).unwrap_or(false)
    }

    /// Returns the amount of storage used by the data, in bytes.
    pub fn storage_size(&self) -> u64 {
        // report without forcing materialization
        if let (Some(file), Payload::Dataset { id, .. }) = (self.0.file(), self.0.payload()) {
            let state = file.state.read();
            if let Some(d) = state.dataset_data(*id) {
                return d
                    .lazy
                    .as_ref()
                    .map(|l| match &l.kind {
                        crate::model::LazyKind::Contiguous { len, .. } => *len,
                        // Resolve the chunk index on demand (deferred from open);
                        // storage_size is an explicit query, not a hot path.
                        crate::model::LazyKind::Chunked {
                            index,
                            dims,
                            chunk_dims,
                            elem_size,
                            ..
                        } => crate::format::reader::resolve_chunks(
                            &*l.src, l.base, index, dims, chunk_dims, *elem_size,
                        )
                        .map(|chunks| chunks.iter().map(|(_, size, _, _)| u64::from(*size)).sum())
                        .unwrap_or(0),
                    })
                    .unwrap_or(d.data.len() as u64);
            }
        }
        self.snapshot().map(|s| s.data.len() as u64).unwrap_or(0)
    }

    /// Rows `[r0, r1)` of the leading axis as raw elements, for random access
    /// into large datasets (e.g. one epoch of an `(N, C, T)` array).
    ///
    /// Unlike `read_slice`, this never clones the whole dataset: lazily stored
    /// data decodes only the chunks overlapping the rows (the chunk index is
    /// walked once and memoized); materialized data is sliced in place. `T`
    /// must be the exact on-disk element type (no conversion), non-vlen.
    pub fn read_rows_raw<T: H5Type>(&self, r0: usize, r1: usize) -> Result<Vec<T>> {
        let file = self.0.file().ok_or("container is not file-resident")?;
        let Payload::Dataset { id, .. } = self.0.payload() else {
            return Err("read_rows_raw: not a dataset".into());
        };
        let state = file.state.read();
        let d = state.dataset_data(*id).ok_or("not a dataset")?;
        if T::type_descriptor() != d.dtype || convert::has_vlen(&d.dtype) {
            return Err(format!(
                "read_rows_raw: element type {:?} must equal the on-disk type {:?}",
                T::type_descriptor(),
                d.dtype
            )
            .into());
        }
        if d.dims.is_empty() || r0 > r1 || r1 as u64 > d.dims[0] {
            return Err(format!(
                "read_rows_raw: rows {r0}..{r1} out of range for dims {:?}",
                d.dims
            )
            .into());
        }
        let elem = disk_size(&d.dtype);
        if elem != mem::size_of::<T>() {
            return Err("read_rows_raw: element size mismatch".into());
        }
        let row_bytes = elem * d.dims[1..].iter().map(|&x| x as usize).product::<usize>();
        let bytes = match &d.lazy {
            Some(l) => l.materialize_rows(r0 as u64, r1 as u64, row_bytes)?,
            None => d.data[r0 * row_bytes..r1 * row_bytes].to_vec(),
        };
        let n = bytes.len() / elem;
        let mut out: Vec<T> = Vec::with_capacity(n);
        // SAFETY: T is a plain H5 scalar of exactly `elem` bytes (checked above)
        // and the bytes are in native (little-endian) order: lazy data swaps as
        // it materializes.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, n * elem);
            out.set_len(n);
        }
        Ok(out)
    }

    /// Creates a reader wrapper for this container.
    pub fn as_reader(&self) -> Reader<'_> {
        Reader::new(self)
    }

    /// Creates a writer wrapper for this container.
    pub fn as_writer(&self) -> Writer<'_> {
        Writer::new(self)
    }

    /// Creates a byte reader over the container's raw data.
    pub fn as_byte_reader(&self) -> Result<ByteReader> {
        ByteReader::new(self)
    }

    // --- convenience read methods (delegate to Reader) ---

    pub fn read<T: H5Type, D: ndarray::Dimension>(&self) -> Result<Array<T, D>> {
        self.as_reader().read()
    }

    pub fn read_raw<T: H5Type>(&self) -> Result<Vec<T>> {
        self.as_reader().read_raw()
    }

    pub fn read_1d<T: H5Type>(&self) -> Result<Array1<T>> {
        self.as_reader().read_1d()
    }

    pub fn read_slice_1d<T, S>(&self, selection: S) -> Result<Array1<T>>
    where
        T: H5Type,
        S: TryInto<Selection>,
        crate::error::Error: From<S::Error>,
    {
        self.as_reader().read_slice_1d(selection)
    }

    pub fn read_2d<T: H5Type>(&self) -> Result<Array2<T>> {
        self.as_reader().read_2d()
    }

    pub fn read_slice_2d<T, S>(&self, selection: S) -> Result<Array2<T>>
    where
        T: H5Type,
        S: TryInto<Selection>,
        crate::error::Error: From<S::Error>,
    {
        self.as_reader().read_slice_2d(selection)
    }

    pub fn read_dyn<T: H5Type>(&self) -> Result<ArrayD<T>> {
        self.as_reader().read_dyn()
    }

    /// Alias for [`Container::shape`] (parity with the FFI crate).
    pub fn get_shape(&self) -> Result<Vec<usize>> {
        Ok(self.shape())
    }

    /// Reads the container as an [`rlx`] tensor (feature `rlx`).
    ///
    /// Numeric data of any width converts to `f32` host data via the same
    /// checked conversion rules as [`Container::read_raw`]; the tensor's
    /// shape mirrors the dataspace. Non-numeric datatypes are an error.
    ///
    /// The returned tensor is an RLX graph constant: further ops compose
    /// lazily and fuse at materialization, so HDF5 checkpoints feed directly
    /// into inference —
    ///
    /// ```ignore
    /// let w1 = file.dataset("model/w1")?.read_tensor()?;
    /// let w2 = file.dataset("model/w2")?.read_tensor()?;
    /// let y = x.matmul(&w1).relu().matmul(&w2);
    /// let out = y.to_vec(); // compiles + runs on RLX's cpu backend
    /// ```
    ///
    /// See `examples/rlx_tensor.rs` and `examples/rlx_mlp.rs`.
    #[cfg(feature = "rlx")]
    pub fn read_tensor(&self) -> Result<rlx::tensor::Tensor> {
        use hdf5_types::TypeDescriptor as TD;
        match self.dtype()?.to_descriptor()? {
            TD::Integer(_) | TD::Unsigned(_) | TD::Float(_) | TD::Boolean | TD::Enum(_) => {}
            other => return Err(format!("read_tensor: {other} is not a numeric datatype").into()),
        }
        let shape: Vec<usize> = self.shape();
        let data: Vec<f32> = self.read_raw()?;
        Ok(rlx::tensor::Tensor::from_vec(data, shape))
    }

    /// Reads the raw element bytes in file layout (little-endian, packed).
    /// Works for every supported datatype; variable-length elements are
    /// returned as their store-resolved payloads concatenated after the
    /// fixed-size slots.
    pub fn read_bytes(&self) -> Result<Vec<u8>> {
        let snap = self.snapshot()?;
        let mut out = snap.data;
        for payload in snap.vlen.iter() {
            out.extend_from_slice(payload);
        }
        Ok(out)
    }

    pub fn read_slice<T, S, D>(&self, selection: S) -> Result<Array<T, D>>
    where
        T: H5Type,
        S: TryInto<Selection>,
        crate::error::Error: From<S::Error>,
        D: ndarray::Dimension,
    {
        self.as_reader().read_slice(selection)
    }

    pub fn read_scalar<T: H5Type>(&self) -> Result<T> {
        self.as_reader().read_scalar()
    }

    // --- convenience write methods (delegate to Writer) ---

    pub fn write<'b, A, T, D>(&self, arr: A) -> Result<()>
    where
        A: Into<ArrayView<'b, T, D>>,
        T: H5Type,
        D: ndarray::Dimension,
    {
        self.as_writer().write(arr)
    }

    pub fn write_raw<'b, A, T>(&self, arr: A) -> Result<()>
    where
        A: Into<ArrayView1<'b, T>>,
        T: H5Type,
    {
        self.as_writer().write_raw(arr)
    }

    pub fn write_slice<'b, A, T, S, D>(&self, arr: A, selection: S) -> Result<()>
    where
        A: Into<ArrayView<'b, T, D>>,
        T: H5Type,
        S: TryInto<Selection>,
        crate::error::Error: From<S::Error>,
        D: ndarray::Dimension,
    {
        self.as_writer().write_slice(arr, selection)
    }

    pub fn write_scalar<T: H5Type>(&self, val: &T) -> Result<()> {
        self.as_writer().write_scalar(val)
    }
}

/// Convert model-layout bytes into an owned `Vec<T>`.
fn bytes_into_vec<T: H5Type>(bytes: Vec<u8>, n: usize) -> Result<Vec<T>> {
    let tsize = mem::size_of::<T>();
    if tsize * n != bytes.len() {
        return Err(format!(
            "element size mismatch: expected {} bytes, got {}",
            tsize * n,
            bytes.len()
        )
        .into());
    }
    let mut out: Vec<T> = Vec::with_capacity(n);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr().cast::<u8>(), bytes.len());
        out.set_len(n);
    }
    // any vlen pointers in `bytes` now belong to `out`; dropping a Vec<u8>
    // frees the buffer alone, and never follows them
    Ok(out)
}

/// `n` zeroed elements, from the allocator's zero pages. Only for types all
/// zero bytes are valid for: plain numbers.
fn zeroed_vec<T>(n: usize) -> Result<Vec<T>> {
    if n == 0 || mem::size_of::<T>() == 0 {
        return Ok(Vec::new());
    }
    let layout = std::alloc::Layout::array::<T>(n).map_err(|_| "allocation too large")?;
    // SAFETY: the layout is non-zero sized, and the Vec takes the block with the
    // same layout, its `n` elements initialized by the zeroing
    unsafe {
        let p = std::alloc::alloc_zeroed(layout).cast::<T>();
        if p.is_null() {
            // a dataset bigger than memory is an error to read whole
            return Err(format!(
                "not enough memory to read {} bytes at once; read it in parts",
                layout.size()
            )
            .into());
        }
        Ok(Vec::from_raw_parts(p, n, n))
    }
}

/// Is this a plain number, the same in memory as in the model?
fn is_plain_number(desc: &TypeDescriptor) -> bool {
    matches!(
        desc,
        TypeDescriptor::Integer(_) | TypeDescriptor::Unsigned(_) | TypeDescriptor::Float(_)
    )
}

/// View a `&[T]` as raw bytes.
fn slice_as_bytes<T>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), std::mem::size_of_val(slice)) }
}

fn check_conversion(
    required: Conversion,
    src: &TypeDescriptor,
    dst: &TypeDescriptor,
) -> Result<()> {
    match conversion_path(src, dst) {
        Some(path) if path <= required => Ok(()),
        Some(path) => Err(format!(
            "conversion path {path:?} exceeds allowed {required:?} ({src} -> {dst})"
        )
        .into()),
        None => Err(format!("no conversion path from {src} to {dst}").into()),
    }
}

/// A reader wrapper with conversion settings.
pub struct Reader<'a> {
    obj: &'a Container,
    conv: Conversion,
}

impl<'a> Reader<'a> {
    pub fn new(obj: &'a Container) -> Self {
        Self {
            obj,
            conv: Conversion::Soft,
        }
    }

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

    fn read_elements<T: H5Type>(&self, selection: Option<Selection>) -> Result<(Vec<T>, Vec<Ix>)> {
        match &selection {
            None => {
                if let Some(read) = self.read_numbers::<T>()? {
                    return Ok(read);
                }
            }
            Some(sel) => {
                if let Some(read) = self.read_picked::<T>(sel)? {
                    return Ok(read);
                }
            }
        }
        let snap = self.obj.snapshot()?;
        let dst_desc = T::type_descriptor();
        check_conversion(self.conv, &snap.dtype, &dst_desc)?;

        let shape: Vec<Ix> = snap.dims.iter().map(|&d| d as usize).collect();
        let esize = disk_size(&snap.dtype);

        match selection {
            None => {
                let n = snap.num_elements();
                let mem = convert::model_to_mem(&snap.dtype, &dst_desc, &snap.data, &snap.vlen, n)?;
                Ok((bytes_into_vec(mem, n)?, shape))
            }
            Some(sel) => {
                if self.obj.handle().id_type() == H5I_type_t::H5I_ATTR {
                    return Err("Slicing cannot be used on attribute datasets".into());
                }
                let out_shape = sel.out_shape(&shape)?;
                let raw = sel.into_raw(&shape)?;
                let indices = raw.linear_indices(&shape)?;
                let mut gathered = Vec::with_capacity(indices.len() * esize);
                for &i in &indices {
                    let start = i * esize;
                    if start + esize > snap.data.len() {
                        return Err("selection out of bounds".into());
                    }
                    gathered.extend_from_slice(&snap.data[start..start + esize]);
                }
                let mem = convert::model_to_mem(
                    &snap.dtype,
                    &dst_desc,
                    &gathered,
                    &snap.vlen,
                    indices.len(),
                )?;
                Ok((bytes_into_vec(mem, indices.len())?, out_shape))
            }
        }
    }

    /// A selection out of a lazily stored dataset: only the stored runs or
    /// chunks holding selected elements are read and decoded (through the
    /// dataset's chunk cache) — never the whole dataset. `None` when the data
    /// is already in memory or holds variable-length values.
    fn read_picked<T: H5Type>(&self, sel: &Selection) -> Result<Option<(Vec<T>, Vec<Ix>)>> {
        let Payload::Dataset { id, .. } = self.obj.0.payload() else {
            return Ok(None);
        };
        let file = self.obj.0.file().ok_or("container is not file-resident")?;
        let lazy = {
            let state = file.state.read();
            state.dataset_data(*id).ok_or("not a dataset")?.lazy.clone()
        };
        let Some(lazy) = lazy else {
            return Ok(None);
        };
        let meta = self.obj.metadata()?;
        if convert::has_vlen(&meta.dtype) {
            return Ok(None);
        }
        let dst = T::type_descriptor();
        check_conversion(self.conv, &meta.dtype, &dst)?;
        let shape: Vec<Ix> = meta.dims.iter().map(|&d| d as usize).collect();
        let out_shape = sel.out_shape(&shape)?;
        let pick = sel.clone().into_raw(&shape)?.pick(&shape)?;
        let n = pick.len();
        let esize = disk_size(&meta.dtype);
        let bytes = lazy.pick(&meta.dims, esize, &pick)?;
        let mut out = zeroed_vec::<T>(n)?;
        let numbers = is_plain_number(&meta.dtype) && is_plain_number(&dst);
        if numbers {
            // SAFETY: `out` holds `n` plain numbers, valid whatever their bytes
            let into = unsafe {
                std::slice::from_raw_parts_mut(
                    out.as_mut_ptr().cast::<u8>(),
                    n * mem::size_of::<T>(),
                )
            };
            if convert::convert_numbers(&meta.dtype, &dst, &bytes, into) {
                return Ok(Some((out, out_shape)));
            }
        }
        drop(out);
        let mem = convert::model_to_mem(&meta.dtype, &dst, &bytes, &meta.vlen, n)?;
        Ok(Some((bytes_into_vec(mem, n)?, out_shape)))
    }

    /// A whole dataset of plain numbers: straight into the result when it is
    /// the type asked for, else through one conversion, with no copy kept in
    /// the file. `None` for anything else.
    fn read_numbers<T: H5Type>(&self) -> Result<Option<(Vec<T>, Vec<Ix>)>> {
        let Payload::Dataset { id, .. } = self.obj.0.payload() else {
            return Ok(None);
        };
        let file = self.obj.0.file().ok_or("container is not file-resident")?;
        let meta = self.obj.metadata()?;
        let dst = T::type_descriptor();
        if !is_plain_number(&meta.dtype) || !is_plain_number(&dst) {
            return Ok(None);
        }
        check_conversion(self.conv, &meta.dtype, &dst)?;
        let n = meta.num_elements();
        let shape: Vec<Ix> = meta.dims.iter().map(|&d| d as usize).collect();
        let lazy = {
            let state = file.state.read();
            state.dataset_data(*id).ok_or("not a dataset")?.lazy.clone()
        };
        if meta.dtype == dst {
            let mut out = zeroed_vec::<T>(n)?;
            // SAFETY: `out` holds `n` plain numbers, valid whatever their bytes
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    out.as_mut_ptr().cast::<u8>(),
                    n * mem::size_of::<T>(),
                )
            };
            match &lazy {
                Some(l) => l.materialize_into(bytes)?,
                None => {
                    let state = file.state.read();
                    let d = state.dataset_data(*id).ok_or("not a dataset")?;
                    if d.data.len() < bytes.len() {
                        return Err("source data too short".into());
                    }
                    bytes.copy_from_slice(&d.data[..bytes.len()]);
                }
            }
            return Ok(Some((out, shape)));
        }
        // converted straight into the result, no intermediate buffer
        let mut out = zeroed_vec::<T>(n)?;
        // SAFETY: `out` holds `n` plain numbers, valid whatever their bytes
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), n * mem::size_of::<T>())
        };
        let ssize = disk_size(&meta.dtype);
        let into = |data: &[u8], bytes: &mut [u8]| -> Result<()> {
            let data = data.get(..n * ssize).ok_or("source data too short")?;
            if !convert::convert_numbers(&meta.dtype, &dst, data, bytes) {
                bytes.copy_from_slice(&convert::model_to_mem(
                    &meta.dtype,
                    &dst,
                    data,
                    &meta.vlen,
                    n,
                )?);
            }
            Ok(())
        };
        let fused = |l: &crate::model::LazyData, bytes: &mut [u8]| {
            let conv = |s: &[u8], d: &mut [u8]| {
                convert::convert_numbers(&meta.dtype, &dst, s, d);
            };
            l.materialize_converted(bytes, ssize, mem::size_of::<T>(), &conv)
        };
        match &lazy {
            // decoded, swapped and converted in one pass where it can be
            Some(l) if convert::numbers_convertible(&meta.dtype, &dst) && fused(l, bytes)? => {}
            Some(l) => into(&l.materialize_bytes()?, bytes)?,
            None => {
                let state = file.state.read();
                let d = state.dataset_data(*id).ok_or("not a dataset")?;
                into(&d.data, bytes)?;
            }
        }
        Ok(Some((out, shape)))
    }

    pub fn read_raw<T: H5Type>(&self) -> Result<Vec<T>> {
        Ok(self.read_elements(None)?.0)
    }

    pub fn read<T: H5Type, D: ndarray::Dimension>(&self) -> Result<Array<T, D>> {
        let (vec, shape) = self.read_elements::<T>(None)?;
        let arr = ArrayD::from_shape_vec(ndarray::IxDyn(&shape), vec)
            .map_err(|e| format!("shape error: {e}"))?;
        arr.into_dimensionality::<D>()
            .map_err(|e| format!("dimensionality error: {e}").into())
    }

    pub fn read_scalar<T: H5Type>(&self) -> Result<T> {
        let (mut vec, _) = self.read_elements::<T>(None)?;
        if vec.len() != 1 {
            return Err(format!("expected scalar, got {} elements", vec.len()).into());
        }
        Ok(vec.remove(0))
    }

    pub fn read_1d<T: H5Type>(&self) -> Result<Array1<T>> {
        self.read()
    }

    pub fn read_2d<T: H5Type>(&self) -> Result<Array2<T>> {
        self.read()
    }

    pub fn read_dyn<T: H5Type>(&self) -> Result<ArrayD<T>> {
        self.read()
    }

    pub fn read_slice<T, S, D>(&self, selection: S) -> Result<Array<T, D>>
    where
        T: H5Type,
        S: TryInto<Selection>,
        crate::error::Error: From<S::Error>,
        D: ndarray::Dimension,
    {
        let sel: Selection = selection.try_into().map_err(crate::error::Error::from)?;
        let (vec, shape) = self.read_elements::<T>(Some(sel))?;
        let arr = ArrayD::from_shape_vec(ndarray::IxDyn(&shape), vec)
            .map_err(|e| format!("shape error: {e}"))?;
        arr.into_dimensionality::<D>()
            .map_err(|e| format!("dimensionality error: {e}").into())
    }

    pub fn read_slice_1d<T, S>(&self, selection: S) -> Result<Array1<T>>
    where
        T: H5Type,
        S: TryInto<Selection>,
        crate::error::Error: From<S::Error>,
    {
        self.read_slice(selection)
    }

    pub fn read_slice_2d<T, S>(&self, selection: S) -> Result<Array2<T>>
    where
        T: H5Type,
        S: TryInto<Selection>,
        crate::error::Error: From<S::Error>,
    {
        self.read_slice(selection)
    }
}

/// A writer wrapper with conversion settings.
pub struct Writer<'a> {
    obj: &'a Container,
    conv: Conversion,
}

impl<'a> Writer<'a> {
    pub fn new(obj: &'a Container) -> Self {
        Self {
            obj,
            conv: Conversion::Soft,
        }
    }

    pub fn conversion(mut self, conv: Conversion) -> Self {
        self.conv = conv;
        self
    }

    pub fn no_convert(mut self) -> Self {
        self.conv = Conversion::NoOp;
        self
    }

    /// Log a dataset write for MPI collective aggregation (no-op when the
    /// file is not MPI-attached or the target is an attribute).
    #[cfg(feature = "mpi")]
    fn mpi_log(&self, ranges: &[(u64, u64)], converted: &[u8], store: &VlenStore) -> Result<()> {
        use crate::handle::Payload;
        let (file, id) = match self.obj.handle().payload() {
            Payload::Dataset { file, id } => (file, *id),
            _ => return Ok(()), // attributes are collective metadata
        };
        let mut guard = file.mpi.lock();
        let Some(mpi) = guard.as_mut() else {
            return Ok(());
        };
        if mpi.comm.rank() == 0 {
            // rank 0's writes are already in the model it will serialize
            return Ok(());
        }
        if !store.is_empty() {
            return Err("variable-length data cannot be written in MPI mode \
                 (parallel HDF5 imposes the same restriction)"
                .into());
        }
        let state = file.state.read();
        let path = state.path_of(id).ok_or("MPI log: dataset has no path")?;
        drop(state);
        let bytes = if ranges.len() == 1 && ranges[0] == (0, converted.len() as u64) {
            converted.to_vec()
        } else {
            let mut b = Vec::with_capacity(ranges.len() * 8);
            let mut src = 0usize;
            for &(_, l) in ranges {
                b.extend_from_slice(&converted[src..src + l as usize]);
                src += l as usize;
            }
            b
        };
        mpi.log.push(crate::mpi::LogEntry {
            path,
            ranges: ranges.to_vec(),
            bytes,
        });
        Ok(())
    }

    fn write_elements<T: H5Type>(&self, data: &[T], selection: Option<Selection>) -> Result<()> {
        if self.write_through(data, selection.as_ref())? {
            return Ok(());
        }
        let snap = self.obj.snapshot()?;
        let src_desc = T::type_descriptor();
        check_conversion(self.conv, &src_desc, &snap.dtype)?;

        let shape: Vec<Ix> = snap.dims.iter().map(|&d| d as usize).collect();
        let esize = disk_size(&snap.dtype);
        let bytes = slice_as_bytes(data);

        match selection {
            None => {
                let n = snap.num_elements();
                if data.len() != n {
                    return Err(
                        format!("write: expected {} elements, got {}", n, data.len()).into(),
                    );
                }
                let mut store = VlenStore::new();
                let model = convert::mem_to_model(&src_desc, &snap.dtype, bytes, &mut store, n)?;
                #[cfg(feature = "mpi")]
                self.mpi_log(&[(0, model.len() as u64)], &model, &store)?;
                self.obj.store(model, store)
            }
            Some(sel) => {
                if self.obj.handle().id_type() == H5I_type_t::H5I_ATTR {
                    return Err("Slicing cannot be used on attribute datasets".into());
                }
                let raw = sel.into_raw(&shape)?;
                let indices = raw.linear_indices(&shape)?;
                if data.len() != indices.len() {
                    return Err(format!(
                        "write_slice: expected {} elements, got {}",
                        indices.len(),
                        data.len()
                    )
                    .into());
                }
                let mut model_data = snap.data;
                let mut store = snap.vlen;
                let converted =
                    convert::mem_to_model(&src_desc, &snap.dtype, bytes, &mut store, data.len())?;
                for (k, &i) in indices.iter().enumerate() {
                    let dst = i * esize;
                    let src = k * esize;
                    if dst + esize > model_data.len() {
                        return Err("selection out of bounds".into());
                    }
                    model_data[dst..dst + esize].copy_from_slice(&converted[src..src + esize]);
                }
                #[cfg(feature = "mpi")]
                {
                    let ranges: Vec<(u64, u64)> = indices
                        .iter()
                        .map(|&i| ((i * esize) as u64, esize as u64))
                        .collect();
                    self.mpi_log(&ranges, &converted, &store)?;
                }
                self.obj.store(model_data, store)
            }
        }
    }

    /// A write into a chunked dataset whose chunks are in the file being
    /// written: only the chunks it touches are read, updated and appended to
    /// the file — the dataset is never loaded whole. `false` for any other
    /// dataset (the write then goes to the in-memory copy).
    fn write_through<T: H5Type>(&self, data: &[T], selection: Option<&Selection>) -> Result<bool> {
        let Payload::Dataset { id, .. } = self.obj.0.payload() else {
            return Ok(false);
        };
        let file = self.obj.0.file().ok_or("container is not file-resident")?;
        #[cfg(feature = "mpi")]
        if file.mpi.lock().is_some() {
            return Ok(false); // collective files log writes for the merge
        }
        let mut state = file.state.write();
        let Some(disk) = state.appendable.clone() else {
            return Ok(false);
        };
        let d = state.dataset_data(*id).ok_or("not a dataset")?;
        let Some(lazy) = d.lazy.clone() else {
            return Ok(false);
        };
        let ours = Arc::as_ptr(&disk) as *const ();
        if lazy.swap != 0
            || Arc::as_ptr(&lazy.src) as *const () != ours
            || !matches!(lazy.kind, crate::model::LazyKind::Chunked { .. })
            || convert::has_vlen(&d.dtype)
            || d.dims.is_empty()
        {
            return Ok(false);
        }
        let (dtype, dims) = (d.dtype.clone(), d.dims.clone());
        let fill = match &d.fill {
            crate::model::FillValue::UserDefined(v) => v.clone(),
            _ => Vec::new(),
        };
        state.dirty = true;
        if state.read_only {
            return Err("unable to write: file is read-only".into());
        }
        let src_desc = T::type_descriptor();
        check_conversion(self.conv, &src_desc, &dtype)?;
        let shape: Vec<Ix> = dims.iter().map(|&d| d as usize).collect();
        let pick = match selection {
            None => crate::hl::selection::RawSelection::All.pick(&shape)?,
            Some(sel) => sel.clone().into_raw(&shape)?.pick(&shape)?,
        };
        if data.len() != pick.len() {
            return Err(format!(
                "write: expected {} elements, got {}",
                pick.len(),
                data.len()
            )
            .into());
        }
        let mut store = VlenStore::new();
        let values = convert::mem_to_model(
            &src_desc,
            &dtype,
            slice_as_bytes(data),
            &mut store,
            data.len(),
        )?;
        lazy.write_pick(&disk, &pick, &values, &fill)?;
        Ok(true)
    }

    pub fn write<'b, A, T, D>(&self, arr: A) -> Result<()>
    where
        A: Into<ArrayView<'b, T, D>>,
        T: H5Type,
        D: ndarray::Dimension,
    {
        let view = arr.into();
        let expected = self.obj.shape();
        let got: Vec<usize> = view.shape().to_vec();
        // scalar containers accept 0-dim arrays; otherwise shapes must match
        if !(expected.is_empty() && got.is_empty()) && expected != got {
            return Err(format!("write: shape mismatch: {got:?} != {expected:?}").into());
        }
        let slice = view
            .as_slice()
            .ok_or("input array is not contiguous or not in standard layout")?;
        self.write_elements(slice, None)
    }

    pub fn write_raw<'b, A, T>(&self, arr: A) -> Result<()>
    where
        A: Into<ArrayView1<'b, T>>,
        T: H5Type,
    {
        let view = arr.into();
        let slice = view
            .as_slice()
            .ok_or("input array is not contiguous or not in standard layout")?;
        self.write_elements(slice, None)
    }

    pub fn write_slice<'b, A, T, S, D>(&self, arr: A, selection: S) -> Result<()>
    where
        A: Into<ArrayView<'b, T, D>>,
        T: H5Type,
        S: TryInto<Selection>,
        crate::error::Error: From<S::Error>,
        D: ndarray::Dimension,
    {
        let sel: Selection = selection.try_into().map_err(crate::error::Error::from)?;
        let view = arr.into();
        let slice = view
            .as_slice()
            .ok_or("input array is not contiguous or not in standard layout")?;
        self.write_elements(slice, Some(sel))
    }

    pub fn write_scalar<T: H5Type>(&self, val: &T) -> Result<()> {
        self.write_elements(std::slice::from_ref(val), None)
    }
}

/// A byte-oriented reader over a container's raw (fixed-size) data,
/// implementing `std::io::Read` and `std::io::Seek`.
pub struct ByteReader {
    data: Vec<u8>,
    pos: u64,
}

impl ByteReader {
    pub fn new(obj: &Container) -> Result<Self> {
        let snap = obj.snapshot()?;
        if convert::has_vlen(&snap.dtype) {
            return Err("cannot read variable-length data as bytes".into());
        }
        Ok(Self {
            data: snap.data,
            pos: 0,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl std::io::Read for ByteReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let pos = self.pos.min(self.data.len() as u64) as usize;
        let n = buf.len().min(self.data.len() - pos);
        buf[..n].copy_from_slice(&self.data[pos..pos + n]);
        self.pos += n as u64;
        Ok(n)
    }
}

impl std::io::Seek for ByteReader {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        use std::io::SeekFrom;
        let new = match pos {
            SeekFrom::Start(o) => o as i64,
            SeekFrom::End(o) => self.data.len() as i64 + o,
            SeekFrom::Current(o) => self.pos as i64 + o,
        };
        if new < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "seek before start",
            ));
        }
        self.pos = new as u64;
        Ok(self.pos)
    }
}
