//! In-memory object model for the pure-Rust HDF5 engine.
//!
//! An open file is represented as an arena of objects (groups, datasets, named
//! datatypes) linked together, plus per-object attributes. All high-level API
//! handles refer into this shared, interior-mutable model. On flush/close the
//! whole model is serialized into a valid HDF5 byte stream (see
//! [`crate::format`]); on open it is parsed back out.

use std::path::PathBuf;

use hdf5_types::TypeDescriptor;

use crate::hl::extents::Extents;
use crate::hl::filters::Filter;

/// Index of an object within a [`FileState`] arena.
pub type ObjId = usize;

/// The raw bytes of an opened file: either memory-mapped (zero-copy, paged in
/// by the OS on demand) or an owned buffer.
pub enum FileImage {
    Mmap(memmap2::Mmap),
    Bytes(Vec<u8>),
}

impl std::ops::Deref for FileImage {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Self::Mmap(m) => m,
            Self::Bytes(b) => b,
        }
    }
}

impl std::fmt::Debug for FileImage {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "FileImage({} bytes)", self.len())
    }
}

/// Deferred dataset bytes: a slice of the file image, loaded on first access
/// so opening a file does not copy every dataset into memory.
#[derive(Clone, Debug)]
pub struct LazyData {
    pub image: std::sync::Arc<FileImage>,
    pub offset: usize,
    pub len: usize,
    /// Logical size (>= len; trailing bytes are zero-filled).
    pub logical: usize,
}

/// Data-layout class of a dataset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayoutClass {
    /// Data stored inline within the object header.
    Compact,
    /// Data stored as a single contiguous block.
    Contiguous,
    /// Data stored as filtered/unfiltered chunks indexed by a v1 B-tree.
    Chunked(Vec<u64>),
    /// Virtual dataset: data mapped from source datasets per the mappings.
    Virtual(Vec<crate::format::vds::VdsMapping>),
}

/// Fill-value policy for a dataset.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum FillValue {
    /// No fill value defined.
    #[default]
    Undefined,
    /// Library default (zero) fill value.
    Default,
    /// User-defined fill value (raw bytes in the dataset's datatype layout).
    UserDefined(Vec<u8>),
}

/// The target of a link in a group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LinkTarget {
    /// Hard link to an object in the same file (arena index).
    Hard(ObjId),
    /// Soft (symbolic) link to a path within the same file.
    Soft(String),
    /// External link to an object in another file.
    External { file: String, path: String },
}

/// A named link within a group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Link {
    pub name: String,
    pub target: LinkTarget,
    pub creation_order: i64,
    pub utf8: bool,
}

/// A group's contents.
#[derive(Clone, Debug, Default)]
pub struct GroupData {
    pub links: Vec<Link>,
    pub track_order: bool,
}

impl GroupData {
    pub fn find(&self, name: &str) -> Option<&Link> {
        self.links.iter().find(|l| l.name == name)
    }

    pub fn find_index(&self, name: &str) -> Option<usize> {
        self.links.iter().position(|l| l.name == name)
    }
}

/// A dataset's metadata and data.
#[derive(Clone, Debug)]
pub struct DatasetData {
    /// Datatype in disk representation (see [`crate::format::convert`]).
    pub dtype: TypeDescriptor,
    pub dims: Vec<u64>,
    pub maxdims: Vec<Option<u64>>,
    pub layout: LayoutClass,
    pub filters: Vec<Filter>,
    pub fill: FillValue,
    /// Raw element bytes in the disk datatype layout, row-major (C order).
    /// Vlen components are `{len, store_idx}` slots referencing `vlen`.
    pub data: Vec<u8>,
    /// Side store for variable-length payloads (1-based indices).
    pub vlen: crate::format::convert::VlenStore,
    /// Present while the data has not been copied out of the file image.
    pub lazy: Option<LazyData>,
    pub is_scalar: bool,
    pub is_null: bool,
}

impl DatasetData {
    /// Copy lazily-referenced bytes out of the file image.
    pub fn materialize(&mut self) {
        if let Some(l) = self.lazy.take() {
            let mut v = Vec::with_capacity(l.logical);
            v.extend_from_slice(&l.image[l.offset..l.offset + l.len]);
            v.resize(l.logical, 0);
            self.data = v;
        }
    }

    pub fn extents(&self) -> Extents {
        if self.is_null {
            Extents::Null
        } else if self.is_scalar {
            Extents::Scalar
        } else {
            use crate::hl::extents::{Extent, SimpleExtents};
            let ext: Vec<Extent> = self
                .dims
                .iter()
                .zip(self.maxdims.iter())
                .map(|(&d, &m)| Extent::new(d as usize, m.map(|v| v as usize)))
                .collect();
            Extents::Simple(SimpleExtents::from_vec(ext))
        }
    }

    /// Number of elements in the dataspace.
    pub fn num_elements(&self) -> usize {
        if self.is_null {
            0
        } else if self.is_scalar {
            1
        } else {
            self.dims.iter().product::<u64>() as usize
        }
    }
}

/// An attribute attached to an object.
#[derive(Clone, Debug)]
pub struct AttrData {
    pub name: String,
    /// Datatype in disk representation.
    pub dtype: TypeDescriptor,
    pub dims: Vec<u64>,
    pub is_scalar: bool,
    pub is_null: bool,
    /// Raw element bytes in the disk datatype layout, row-major.
    pub data: Vec<u8>,
    /// Side store for variable-length payloads (1-based indices).
    pub vlen: crate::format::convert::VlenStore,
}

impl AttrData {
    pub fn num_elements(&self) -> usize {
        if self.is_null {
            0
        } else if self.is_scalar {
            1
        } else {
            self.dims.iter().product::<u64>() as usize
        }
    }

    pub fn extents(&self) -> Extents {
        if self.is_null {
            Extents::Null
        } else if self.is_scalar {
            Extents::Scalar
        } else {
            use crate::hl::extents::{Extent, SimpleExtents};
            let ext: Vec<Extent> = self
                .dims
                .iter()
                .map(|&d| Extent::new(d as usize, Some(d as usize)))
                .collect();
            Extents::Simple(SimpleExtents::from_vec(ext))
        }
    }
}

/// The kind of an object stored in the arena.
// group/dataset payloads legitimately differ in size; boxing would ripple
// through every accessor for no measurable gain
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum ObjectKind {
    Group(GroupData),
    Dataset(DatasetData),
    NamedType(TypeDescriptor),
    /// An object present in the file but not parseable by this build; the
    /// string holds the reason. Accessing it errors; writing a file that
    /// contains one errors (preventing silent data loss on rewrite).
    Unsupported(String),
}

/// A single object in the arena, with its attributes.
#[derive(Clone, Debug)]
pub struct ObjectNode {
    pub kind: ObjectKind,
    pub attrs: Vec<AttrData>,
    pub comment: Option<String>,
    /// Number of hard links referencing this object (for lifetime bookkeeping).
    pub refcount: u32,
    /// Object modification time (seconds since epoch; 0 = unset).
    pub mtime: u32,
}

impl ObjectNode {
    pub fn new(kind: ObjectKind) -> Self {
        Self {
            kind,
            attrs: Vec::new(),
            comment: None,
            refcount: 0,
            mtime: 0,
        }
    }

    pub fn attr_index(&self, name: &str) -> Option<usize> {
        self.attrs.iter().position(|a| a.name == name)
    }
}

/// The mutable state of an open file: the object arena plus file-level metadata.
#[derive(Clone, Debug)]
pub struct FileState {
    pub objects: Vec<Option<ObjectNode>>,
    pub root: ObjId,
    pub userblock: u64,
    /// Userblock content (empty = zeros); preserved across rewrites.
    pub userblock_data: Vec<u8>,
    pub read_only: bool,
    pub next_creation_order: i64,
    /// SOHM configuration: (message-type flag bits, min message size) per
    /// index; empty = no shared-message tables.
    pub sohm: Vec<(u16, u32)>,
}

impl FileState {
    /// Create a new file state with an empty root group.
    pub fn new_empty() -> Self {
        let root_node = ObjectNode::new(ObjectKind::Group(GroupData::default()));
        let mut objects = vec![Some(root_node)];
        // The root group is referenced by the file itself.
        if let Some(Some(n)) = objects.get_mut(0) {
            n.refcount = 1;
        }
        Self {
            objects,
            root: 0,
            userblock: 0,
            userblock_data: Vec::new(),
            read_only: false,
            next_creation_order: 0,
            sohm: Vec::new(),
        }
    }

    pub fn get(&self, id: ObjId) -> &ObjectNode {
        self.objects[id].as_ref().expect("dangling object id")
    }

    pub fn get_mut(&mut self, id: ObjId) -> &mut ObjectNode {
        self.objects[id].as_mut().expect("dangling object id")
    }

    pub fn try_get(&self, id: ObjId) -> Option<&ObjectNode> {
        self.objects.get(id).and_then(|o| o.as_ref())
    }

    pub fn alloc(&mut self, kind: ObjectKind) -> ObjId {
        let id = self.objects.len();
        self.objects.push(Some(ObjectNode::new(kind)));
        id
    }

    pub fn next_order(&mut self) -> i64 {
        let n = self.next_creation_order;
        self.next_creation_order += 1;
        n
    }

    pub fn is_group(&self, id: ObjId) -> bool {
        matches!(
            self.try_get(id).map(|n| &n.kind),
            Some(ObjectKind::Group(_))
        )
    }

    pub fn is_dataset(&self, id: ObjId) -> bool {
        matches!(
            self.try_get(id).map(|n| &n.kind),
            Some(ObjectKind::Dataset(_))
        )
    }

    pub fn group_data(&self, id: ObjId) -> Option<&GroupData> {
        match &self.try_get(id)?.kind {
            ObjectKind::Group(g) => Some(g),
            _ => None,
        }
    }

    pub fn dataset_data(&self, id: ObjId) -> Option<&DatasetData> {
        match &self.try_get(id)?.kind {
            ObjectKind::Dataset(d) => Some(d),
            _ => None,
        }
    }

    /// Resolve a possibly-nested path to an object id, starting from `start`.
    /// Absolute paths (beginning with `/`) start from the root group. Follows
    /// hard and soft links; external links are not followed.
    pub fn resolve(&self, start: ObjId, path: &str) -> Option<ObjId> {
        let mut cur = if path.starts_with('/') {
            self.root
        } else {
            start
        };
        for comp in path.split('/').filter(|s| !s.is_empty() && *s != ".") {
            let group = match &self.try_get(cur)?.kind {
                ObjectKind::Group(g) => g,
                _ => return None,
            };
            let link = group.find(comp)?;
            cur = match &link.target {
                LinkTarget::Hard(id) => *id,
                LinkTarget::Soft(target) => self.resolve(self.root, target)?,
                LinkTarget::External { .. } => return None,
            };
        }
        Some(cur)
    }

    /// Find one path from the root to the given object (BFS). Returns "/" for
    /// the root. Returns `None` if unreachable.
    pub fn path_of(&self, target: ObjId) -> Option<String> {
        if target == self.root {
            return Some("/".to_string());
        }
        let mut queue: std::collections::VecDeque<(ObjId, String)> =
            std::collections::VecDeque::new();
        let mut seen = std::collections::HashSet::new();
        queue.push_back((self.root, String::new()));
        seen.insert(self.root);
        while let Some((cur, prefix)) = queue.pop_front() {
            if let Some(ObjectKind::Group(g)) = self.try_get(cur).map(|n| &n.kind) {
                for link in &g.links {
                    if let LinkTarget::Hard(id) = link.target {
                        let path = format!("{prefix}/{}", link.name);
                        if id == target {
                            return Some(path);
                        }
                        if self.is_group(id) && seen.insert(id) {
                            queue.push_back((id, path));
                        }
                    }
                }
            }
        }
        None
    }

    /// Materialize every lazy dataset (required before serialization).
    pub fn materialize_all(&mut self) {
        for slot in self.objects.iter_mut().flatten() {
            if let ObjectKind::Dataset(d) = &mut slot.kind {
                d.materialize();
            }
        }
    }

    /// Recompute hard-link reference counts across the arena.
    pub fn recount(&mut self) {
        let n = self.objects.len();
        let mut counts = vec![0u32; n];
        counts[self.root] = 1;
        for node in self.objects.iter().flatten() {
            if let ObjectKind::Group(g) = &node.kind {
                for link in &g.links {
                    if let LinkTarget::Hard(id) = link.target {
                        if id < n {
                            counts[id] += 1;
                        }
                    }
                }
            }
        }
        for (id, slot) in self.objects.iter_mut().enumerate() {
            if let Some(node) = slot {
                node.refcount = counts[id];
            }
        }
    }
}

/// The shared, interior-mutable state backing an open file.
pub(crate) struct FileInner {
    pub path: Option<PathBuf>,
    #[allow(dead_code)]
    pub mode: crate::hl::file::OpenMode,
    pub state: parking_lot::RwLock<FileState>,
    /// Synthetic id for this file handle.
    pub id: crate::h5i::hid_t,
    /// Files opened while following external links, keyed by the link's
    /// file-name string (kept open for the lifetime of this file).
    pub externals: parking_lot::Mutex<std::collections::HashMap<String, std::sync::Arc<FileInner>>>,
    /// MPI collective-file state (feature `mpi`).
    #[cfg(feature = "mpi")]
    pub mpi: parking_lot::Mutex<Option<crate::mpi::MpiFile>>,
}

/// The current time in seconds since the Unix epoch (0 on clock error).
pub(crate) fn now() -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as u32)
        .unwrap_or(0)
}
