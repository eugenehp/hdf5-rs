//! `Location`: a named object within a file (file, group, dataset, named type).

use std::fmt::{self, Debug};
use std::ops::Deref;

use crate::class::ObjectClass;
use crate::error::Result;
use crate::h5i::H5I_type_t;
use crate::handle::{Handle, Payload};
use crate::hl::attribute::{Attribute, AttributeBuilder, AttributeBuilderEmpty};
use crate::hl::file::File;
use crate::hl::object::Object;
use crate::model::{FileInner, LinkTarget, ObjId, ObjectKind};
use hdf5_types::H5Type;
use std::sync::Arc;

/// A locatable object's type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocationType {
    Group,
    Dataset,
    NamedDatatype,
}

/// An opaque, unique token identifying an object within a file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LocationToken(pub(crate) u64);

/// Metadata information about a location.
#[derive(Clone, Debug)]
pub struct LocationInfo {
    pub fileno: u64,
    pub token: LocationToken,
    pub loc_type: LocationType,
    pub num_links: usize,
    pub atime: i64,
    pub mtime: i64,
    pub ctime: i64,
    pub btime: i64,
    pub num_attrs: usize,
}

/// Any object that has a location within a file.
#[repr(transparent)]
#[derive(Clone)]
pub struct Location(pub(crate) Handle);

impl ObjectClass for Location {
    const NAME: &'static str = "location";
    const VALID_TYPES: &'static [H5I_type_t] = &[
        H5I_type_t::H5I_FILE,
        H5I_type_t::H5I_GROUP,
        H5I_type_t::H5I_DATATYPE,
        H5I_type_t::H5I_DATASET,
        H5I_type_t::H5I_ATTR,
    ];

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

impl Debug for Location {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.debug_fmt(f)
    }
}

impl Deref for Location {
    type Target = Object;

    fn deref(&self) -> &Object {
        unsafe { self.transmute() }
    }
}

/// Maximum depth of link indirection while resolving a path (soft links,
/// external links), mirroring libhdf5's `nlinks` traversal limit.
const MAX_LINK_DEPTH: usize = 32;

/// Open (and cache) the file referenced by an external link. External files
/// are opened read-only; relative names resolve against the referring file's
/// directory (libhdf5's primary search location).
pub(crate) fn open_external(parent: &Arc<FileInner>, name: &str) -> Result<Arc<FileInner>> {
    if let Some(f) = parent.externals.lock().get(name) {
        return Ok(f.clone());
    }
    let mut path = std::path::PathBuf::from(name);
    if path.is_relative() {
        if let Some(dir) = parent.path.as_ref().and_then(|p| p.parent()) {
            path = dir.join(path);
        }
    }
    let bytes =
        std::fs::read(&path).map_err(|e| format!("unable to open external file '{name}': {e}"))?;
    let mut state = crate::format::parse(&bytes)?;
    state.read_only = true;
    let inner = Arc::new(FileInner {
        path: Some(path),
        mode: crate::hl::file::OpenMode::Read,
        state: crate::lock::RwLock::new(state),
        id: crate::handle::next_id(),
        externals: crate::lock::Mutex::new(std::collections::HashMap::new()),
        #[cfg(feature = "mpi")]
        mpi: crate::lock::Mutex::new(None),
    });
    parent
        .externals
        .lock()
        .insert(name.to_string(), inner.clone());
    Ok(inner)
}

/// Resolve a path to an object, following soft links within a file and
/// external links across files.
pub(crate) fn resolve_path_cross(
    file: &Arc<FileInner>,
    start: ObjId,
    path: &str,
    depth: usize,
) -> Result<(Arc<FileInner>, ObjId)> {
    if depth > MAX_LINK_DEPTH {
        return Err("too many levels of link indirection".into());
    }
    let mut cur_file = file.clone();
    let mut cur = if path.starts_with('/') {
        cur_file.state.read().root
    } else {
        start
    };
    for comp in path.split('/').filter(|s| !s.is_empty() && *s != ".") {
        // Parse a lazily-stubbed group before reading its links (no-op eagerly).
        cur_file.state.write().force(cur).ok();
        let target = {
            let state = cur_file.state.read();
            let group = match &state.try_get(cur).ok_or("dangling object")?.kind {
                ObjectKind::Group(g) => g,
                ObjectKind::Unsupported(reason) => {
                    return Err(format!("object is unsupported by this build: {reason}").into())
                }
                _ => return Err(format!("'{comp}' is not reachable (not a group)").into()),
            };
            group
                .find(comp)
                .cloned()
                .ok_or_else(|| format!("object '{comp}' not found"))?
        };
        match target.target {
            LinkTarget::Hard(id) => cur = id,
            LinkTarget::Soft(t) => {
                let root = cur_file.state.read().root;
                let (f, id) = resolve_path_cross(&cur_file, root, &t, depth + 1)?;
                cur_file = f;
                cur = id;
            }
            LinkTarget::External {
                file: fname,
                path: tpath,
            } => {
                let ext = open_external(&cur_file, &fname)?;
                let root = ext.state.read().root;
                let (f, id) = resolve_path_cross(&ext, root, &tpath, depth + 1)?;
                cur_file = f;
                cur = id;
            }
        }
    }
    Ok((cur_file, cur))
}

impl Location {
    /// Returns the full path of the object within the file.
    pub fn name(&self) -> String {
        match (self.0.file(), self.0.obj_id()) {
            (Some(file), Some(id)) => file.state.read().path_of(id).unwrap_or_else(|| "/".into()),
            _ => String::new(),
        }
    }

    /// Returns the name of the file containing the object.
    pub fn filename(&self) -> String {
        self.0
            .file()
            .and_then(|f| f.path.as_ref().map(|p| p.to_string_lossy().into_owned()))
            .unwrap_or_default()
    }

    /// Returns a handle to the file containing the object.
    pub fn file(&self) -> Result<File> {
        let file = self.0.file().ok_or("object is not file-resident")?;
        Ok(File::from_handle(Handle::new(Payload::File(file.clone()))))
    }

    /// Returns the comment attached to the object, if any.
    pub fn comment(&self) -> Option<String> {
        let file = self.0.file()?;
        let id = self.0.obj_id()?;
        file.state
            .read()
            .try_get(id)
            .and_then(|n| n.comment.clone())
    }

    /// Sets or overwrites the comment attached to the object.
    pub fn set_comment(&self, comment: &str) -> Result<()> {
        let file = self.0.file().ok_or("object is not file-resident")?;
        let id = self.0.obj_id().ok_or("object has no location")?;
        let mut state = file.state.write();
        state.dirty = true;
        if state.read_only {
            return Err("unable to modify: file is read-only".into());
        }
        state.get_mut(id).comment = Some(comment.to_string());
        Ok(())
    }

    /// Clears the comment attached to the object.
    pub fn clear_comment(&self) -> Result<()> {
        let file = self.0.file().ok_or("object is not file-resident")?;
        let id = self.0.obj_id().ok_or("object has no location")?;
        let mut state = file.state.write();
        state.dirty = true;
        if state.read_only {
            return Err("unable to modify: file is read-only".into());
        }
        state.get_mut(id).comment = None;
        Ok(())
    }

    /// Returns information about this location.
    pub fn loc_info(&self) -> Result<LocationInfo> {
        let file = self.0.file().ok_or("object is not file-resident")?;
        let id = self.0.obj_id().ok_or("object has no location")?;
        file.state.write().force(id).ok();
        let state = file.state.read();
        let node = state.try_get(id).ok_or("dangling object")?;
        let loc_type = match &node.kind {
            ObjectKind::Group(_) => LocationType::Group,
            ObjectKind::Dataset(_) => LocationType::Dataset,
            ObjectKind::NamedType(_) => LocationType::NamedDatatype,
            ObjectKind::Unsupported(reason) => {
                return Err(format!("object is unsupported by this build: {reason}").into())
            }
            ObjectKind::Lazy { .. } => return Err("object could not be parsed".into()),
        };
        Ok(LocationInfo {
            fileno: file.id as u64,
            token: LocationToken(id as u64),
            loc_type,
            num_links: node.refcount as usize,
            atime: node.mtime as i64,
            mtime: node.mtime as i64,
            ctime: node.mtime as i64,
            btime: node.mtime as i64,
            num_attrs: node.attrs.len(),
        })
    }

    /// Returns the type of this location.
    pub fn loc_type(&self) -> Result<LocationType> {
        Ok(self.loc_info()?.loc_type)
    }

    /// Returns information about the object at the given relative path.
    pub fn loc_info_by_name(&self, name: &str) -> Result<LocationInfo> {
        let obj = self.open_by_path(name)?;
        obj.loc_info()
    }

    /// Returns the type of the object at the given relative path.
    pub fn loc_type_by_name(&self, name: &str) -> Result<LocationType> {
        Ok(self.loc_info_by_name(name)?.loc_type)
    }

    /// Opens the object identified by a location token.
    pub fn open_by_token(&self, token: LocationToken) -> Result<Self> {
        let file = self.0.file().ok_or("object is not file-resident")?;
        let id = token.0 as usize;
        file.state.write().force(id).ok();
        let state = file.state.read();
        let node = state.try_get(id).ok_or("invalid location token")?;
        let payload = match &node.kind {
            ObjectKind::Group(_) => Payload::Group {
                file: file.clone(),
                id,
            },
            ObjectKind::Dataset(_) => Payload::Dataset {
                file: file.clone(),
                id,
            },
            ObjectKind::NamedType(_) => Payload::NamedType {
                file: file.clone(),
                id,
            },
            ObjectKind::Unsupported(reason) => {
                return Err(format!("object is unsupported by this build: {reason}").into())
            }
            ObjectKind::Lazy { .. } => return Err("object could not be parsed".into()),
        };
        Ok(Self::from_handle(Handle::new(payload)))
    }

    /// Resolve a path relative to this location into a new `Location`,
    /// following soft links and (cross-file) external links.
    pub(crate) fn open_by_path(&self, path: &str) -> Result<Self> {
        let file = self.0.file().ok_or("object is not file-resident")?;
        let start = self.0.obj_id().ok_or("object has no location")?;
        let (file, id) = resolve_path_cross(file, start, path, 0)
            .map_err(|e| crate::error::Error::from(format!("unable to open '{path}': {e}")))?;
        file.state.write().force(id).ok();
        let state = file.state.read();
        let node = state.try_get(id).ok_or("dangling object")?;
        let payload = match &node.kind {
            ObjectKind::Group(_) => Payload::Group {
                file: file.clone(),
                id,
            },
            ObjectKind::Dataset(_) => Payload::Dataset {
                file: file.clone(),
                id,
            },
            ObjectKind::NamedType(_) => Payload::NamedType {
                file: file.clone(),
                id,
            },
            ObjectKind::Lazy { .. } => return Err("object could not be parsed".into()),
            ObjectKind::Unsupported(reason) => {
                return Err(format!("object is unsupported by this build: {reason}").into())
            }
        };
        drop(state);
        Ok(Self::from_handle(Handle::new(payload)))
    }

    // --- attributes ---

    /// Creates a new attribute builder for a specific type.
    pub fn new_attr<T: H5Type>(&self) -> AttributeBuilderEmpty {
        AttributeBuilder::new(self).empty::<T>()
    }

    /// Creates a new generic attribute builder.
    pub fn new_attr_builder(&self) -> AttributeBuilder {
        AttributeBuilder::new(self)
    }

    /// Opens an existing attribute by name.
    pub fn attr(&self, name: &str) -> Result<Attribute> {
        let file = self.0.file().ok_or("object is not file-resident")?;
        let id = self.0.obj_id().ok_or("object has no location")?;
        {
            let state = file.state.read();
            let node = state.try_get(id).ok_or("dangling object")?;
            if node.attr_index(name).is_none() {
                return Err(format!("unable to open attribute '{name}'").into());
            }
        }
        Ok(Attribute::from_handle(Handle::new(Payload::Attribute {
            file: file.clone(),
            owner: id,
            name: name.to_string(),
        })))
    }

    /// Returns the names of all attributes attached to this object.
    pub fn attr_names(&self) -> Result<Vec<String>> {
        let file = self.0.file().ok_or("object is not file-resident")?;
        let id = self.0.obj_id().ok_or("object has no location")?;
        let state = file.state.read();
        let node = state.try_get(id).ok_or("dangling object")?;
        Ok(node.attrs.iter().map(|a| a.name.clone()).collect())
    }
}

/// An HDF5 **object reference** (`H5R_OBJECT`): the 8-byte address of an object
/// header, as stored in a reference-typed dataset or attribute.
///
/// MATLAB `-v7.3` files use these for cell arrays — the cell dataset holds one
/// reference per element, each pointing at the real data under `/#refs#`.
///
/// The datatype is read as a fixed array of 8 bytes (the reference token), which
/// is exactly how the format layer decodes datatype class 7, so
/// `dataset.read_raw::<ObjectReference1>()` needs no conversion.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ObjectReference1 {
    token: [u8; 8],
}

impl ObjectReference1 {
    /// The object-header address this reference points at.
    pub fn address(&self) -> u64 {
        u64::from_le_bytes(self.token)
    }

    /// Build a reference to a given object-header address.
    pub fn from_address(addr: u64) -> Self {
        Self {
            token: addr.to_le_bytes(),
        }
    }

    /// True for the null reference, which HDF5 writes as all-zero.
    pub fn is_null(&self) -> bool {
        self.token == [0u8; 8]
    }
}

impl Debug for ObjectReference1 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ObjectReference1(0x{:x})", self.address())
    }
}

unsafe impl H5Type for ObjectReference1 {
    fn type_descriptor() -> hdf5_types::TypeDescriptor {
        use hdf5_types::{IntSize, TypeDescriptor};
        TypeDescriptor::FixedArray(Box::new(TypeDescriptor::Unsigned(IntSize::U1)), 8)
    }
}

/// What an [`ObjectReference1`] resolved to.
#[derive(Debug)]
pub enum ReferencedObject {
    Group(crate::hl::group::Group),
    Dataset(crate::hl::dataset::Dataset),
}

impl Location {
    /// The arena ids of `targets`, which must be objects of this file.
    fn reference_targets(&self, targets: &[&Location]) -> Result<Vec<crate::model::ObjId>> {
        let file = self.handle().file().ok_or("not file-resident")?;
        targets
            .iter()
            .map(|t| {
                let same = t.handle().file().is_some_and(|f| Arc::ptr_eq(f, file));
                if !same {
                    return Err("an object reference to another file".into());
                }
                t.handle()
                    .obj_id()
                    .ok_or_else(|| "a reference to an object without a location".into())
            })
            .collect()
    }

    fn set_attr_refs(&self, name: &str, refs: Vec<crate::model::ObjId>) -> Result<()> {
        let file = self.handle().file().ok_or("not file-resident")?;
        let owner = self.handle().obj_id().ok_or("no location")?;
        let mut state = file.state.write();
        let node = state.get_mut(owner);
        let i = node.attr_index(name).ok_or("attribute vanished")?;
        node.attrs[i].refs = Some(refs);
        Ok(())
    }

    /// Creates a scalar attribute holding an object reference (`H5R_OBJECT`)
    /// to `target`, an object of this file — as NWB's `table` attributes.
    pub fn new_reference_attr(&self, name: &str, target: &Location) -> Result<Attribute> {
        let refs = self.reference_targets(&[target])?;
        let attr = self.new_attr::<ObjectReference1>().create(name)?;
        self.set_attr_refs(name, refs)?;
        Ok(attr)
    }

    /// Creates an attribute holding object references to `targets`, in order.
    pub fn new_reference_array_attr(&self, name: &str, targets: &[&Location]) -> Result<Attribute> {
        let refs = self.reference_targets(targets)?;
        let attr = self
            .new_attr::<ObjectReference1>()
            .shape([refs.len()])
            .create(name)?;
        self.set_attr_refs(name, refs)?;
        Ok(attr)
    }

    /// Resolve an object reference against the file this location belongs to.
    ///
    /// The reference token is an object-header address; a lazily-opened file may
    /// not have walked to that object yet, in which case it is parsed on demand.
    pub fn dereference(&self, reference: &ObjectReference1) -> Result<ReferencedObject> {
        if reference.is_null() {
            return Err("null object reference".into());
        }
        let file = self
            .0
            .file()
            .ok_or("cannot dereference against a location with no file")?
            .clone();
        let addr = reference.address();
        let id = {
            let mut state = file.state.write();
            crate::format::reader::object_at(&mut state, addr)?
        };
        let is_group = file.state.read().is_group(id);
        Ok(if is_group {
            ReferencedObject::Group(crate::hl::group::Group::from_handle(Handle::new(
                Payload::Group { file, id },
            )))
        } else {
            ReferencedObject::Dataset(crate::hl::dataset::Dataset::from_handle(Handle::new(
                Payload::Dataset { file, id },
            )))
        })
    }
}
