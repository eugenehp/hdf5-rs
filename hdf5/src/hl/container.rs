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
        let file = self.0.file().ok_or("container is not file-resident")?;
        match self.0.payload() {
            Payload::Dataset { id, .. } => {
                // materialize lazily-referenced data on first access
                {
                    let state = file.state.read();
                    let needs = state
                        .dataset_data(*id)
                        .map(|d| d.lazy.is_some())
                        .unwrap_or(false);
                    drop(state);
                    if needs {
                        let mut state = file.state.write();
                        if let crate::model::ObjectKind::Dataset(d) = &mut state.get_mut(*id).kind {
                            d.materialize();
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
                    data: d.data.clone(),
                    vlen: d.vlen.clone(),
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
        let snap = self.snapshot()?;
        Datatype::from_descriptor(&snap.dtype)
    }

    /// Returns the dataspace of this container.
    pub fn space(&self) -> Result<Dataspace> {
        let snap = self.snapshot()?;
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
        self.snapshot()
            .map(|s| s.dims.iter().map(|&d| d as usize).collect())
            .unwrap_or_default()
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.snapshot().map(|s| s.dims.len()).unwrap_or(0)
    }

    /// Returns the total number of elements.
    pub fn size(&self) -> usize {
        self.snapshot().map(|s| s.num_elements()).unwrap_or(0)
    }

    /// Returns whether the dataspace is scalar.
    pub fn is_scalar(&self) -> bool {
        self.snapshot().map(|s| s.is_scalar).unwrap_or(false)
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
                    .map(|l| l.len as u64)
                    .unwrap_or(d.data.len() as u64);
            }
        }
        self.snapshot().map(|s| s.data.len() as u64).unwrap_or(0)
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
    // ownership of any vlen pointers has moved into `out`
    mem::forget(bytes);
    Ok(out)
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
