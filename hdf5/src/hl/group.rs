//! HDF5 groups.

use std::fmt::{self, Debug};
use std::ops::Deref;

use crate::class::ObjectClass;
use crate::error::Result;
use crate::h5i::H5I_type_t;
use crate::handle::{Handle, Payload};
use crate::hl::dataset::{Dataset, DatasetBuilder, DatasetBuilderEmpty};
use crate::hl::datatype::Datatype;
use crate::hl::location::Location;
use crate::model::{GroupData, Link, LinkTarget, ObjectKind};
use hdf5_types::H5Type;

/// Iteration order for group traversal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TraversalOrder {
    #[default]
    Name,
    Creation,
}

/// Directionality of iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum IterationOrder {
    Increasing,
    Decreasing,
    #[default]
    Native,
}

/// Type of a link within a group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkType {
    Hard,
    Soft,
    External,
}

/// Metadata about a link.
#[derive(Clone, Copy, Debug)]
pub struct LinkInfo {
    pub link_type: LinkType,
    pub creation_order: Option<i64>,
    pub is_utf8: bool,
}

/// An HDF5 group.
#[repr(transparent)]
#[derive(Clone)]
pub struct Group(Handle);

impl ObjectClass for Group {
    const NAME: &'static str = "group";
    const VALID_TYPES: &'static [H5I_type_t] = &[H5I_type_t::H5I_FILE, H5I_type_t::H5I_GROUP];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }

    fn short_repr(&self) -> Option<String> {
        let members = match self.len() {
            0 => "empty".to_owned(),
            1 => "1 member".to_owned(),
            x => format!("{x} members"),
        };
        Some(format!("\"{}\" ({})", self.name(), members))
    }
}

impl Debug for Group {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.debug_fmt(f)
    }
}

impl Deref for Group {
    type Target = Location;

    fn deref(&self) -> &Location {
        unsafe { self.transmute() }
    }
}

impl Group {
    fn with_group<R>(&self, f: impl FnOnce(&GroupData) -> R) -> Result<R> {
        let file = self.0.file().ok_or("group is not file-resident")?;
        let id = self.0.obj_id().ok_or("group has no location")?;
        let state = file.state.read();
        match &state.try_get(id).ok_or("dangling group")?.kind {
            ObjectKind::Group(g) => Ok(f(g)),
            _ => Err("object is not a group".into()),
        }
    }

    /// Returns the number of objects in the group.
    pub fn len(&self) -> u64 {
        self.with_group(|g| g.links.len() as u64).unwrap_or(0)
    }

    /// Returns true if the group has no members.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Creates a new group (and intermediate groups as needed).
    pub fn create_group(&self, name: &str) -> Result<Self> {
        let file = self.0.file().ok_or("group is not file-resident")?.clone();
        let start = self.0.obj_id().ok_or("group has no location")?;
        let mut state = file.state.write();
        if state.read_only {
            return Err("unable to create group: file is read-only".into());
        }
        let mut cur = if name.starts_with('/') {
            state.root
        } else {
            start
        };
        let mut created = cur;
        for comp in name.split('/').filter(|s| !s.is_empty()) {
            let existing = state.group_data(cur).and_then(|g| g.find(comp).cloned());
            match existing {
                Some(link) => match link.target {
                    LinkTarget::Hard(id) if state.is_group(id) => {
                        cur = id;
                        created = id;
                    }
                    _ => return Err(format!("unable to create group at '{comp}'").into()),
                },
                None => {
                    let new_id = state.alloc(ObjectKind::Group(GroupData::default()));
                    let order = state.next_order();
                    match &mut state.get_mut(cur).kind {
                        ObjectKind::Group(g) => g.links.push(Link {
                            name: comp.to_string(),
                            target: LinkTarget::Hard(new_id),
                            creation_order: order,
                            utf8: !comp.is_ascii(),
                        }),
                        _ => return Err("parent is not a group".into()),
                    }
                    state.get_mut(new_id).refcount += 1;
                    state.get_mut(new_id).mtime = crate::model::now();
                    cur = new_id;
                    created = new_id;
                }
            }
        }
        drop(state);
        Ok(Self::from_handle(Handle::new(Payload::Group {
            file,
            id: created,
        })))
    }

    /// Opens an existing group by (possibly nested) name.
    pub fn group(&self, name: &str) -> Result<Self> {
        let loc = self.open_by_path(name)?;
        let file = loc.0.file().unwrap().clone();
        let id = loc.0.obj_id().unwrap();
        if !file.state.read().is_group(id) {
            return Err(format!("'{name}' is not a group").into());
        }
        Ok(Self::from_handle(Handle::new(Payload::Group { file, id })))
    }

    /// Creates a soft link.
    pub fn link_soft(&self, target: &str, link_name: &str) -> Result<()> {
        self.add_link(link_name, |_| Ok(LinkTarget::Soft(target.to_string())))
    }

    /// Creates a hard link to an existing object.
    pub fn link_hard(&self, target: &str, link_name: &str) -> Result<()> {
        let file = self.0.file().ok_or("group is not file-resident")?.clone();
        let start = self.0.obj_id().ok_or("group has no location")?;
        let target_id = {
            let state = file.state.read();
            state
                .resolve(start, target)
                .ok_or_else(|| format!("target '{target}' not found"))?
        };
        self.add_link(link_name, |_| Ok(LinkTarget::Hard(target_id)))?;
        let mut state = file.state.write();
        state.get_mut(target_id).refcount += 1;
        Ok(())
    }

    /// Creates an external link to an object in another file.
    pub fn link_external(
        &self,
        target_file_name: &str,
        target: &str,
        link_name: &str,
    ) -> Result<()> {
        self.add_link(link_name, |_| {
            Ok(LinkTarget::External {
                file: target_file_name.to_string(),
                path: target.to_string(),
            })
        })
    }

    fn add_link(
        &self,
        link_name: &str,
        make: impl FnOnce(&crate::model::FileState) -> Result<LinkTarget>,
    ) -> Result<()> {
        let file = self.0.file().ok_or("group is not file-resident")?.clone();
        let start = self.0.obj_id().ok_or("group has no location")?;
        let mut state = file.state.write();
        if state.read_only {
            return Err("unable to modify links: file is read-only".into());
        }
        // resolve the parent group of the link name (supports nested paths)
        let (parent, leaf) = match link_name.rfind('/') {
            Some(pos) => {
                let (dir, leaf) = link_name.split_at(pos);
                let dir = if dir.is_empty() { "/" } else { dir };
                let parent = state
                    .resolve(start, dir)
                    .ok_or_else(|| format!("parent group '{dir}' not found"))?;
                (parent, &leaf[1..])
            }
            None => (start, link_name),
        };
        if state
            .group_data(parent)
            .map(|g| g.find(leaf).is_some())
            .unwrap_or(false)
        {
            return Err(format!("link '{leaf}' already exists").into());
        }
        let target = make(&state)?;
        let order = state.next_order();
        match &mut state.get_mut(parent).kind {
            ObjectKind::Group(g) => {
                g.links.push(Link {
                    name: leaf.to_string(),
                    target,
                    creation_order: order,
                    utf8: !leaf.is_ascii(),
                });
                Ok(())
            }
            _ => Err("parent is not a group".into()),
        }
    }

    /// Renames (relinks) an object.
    pub fn relink(&self, name: &str, path: &str) -> Result<()> {
        let file = self.0.file().ok_or("group is not file-resident")?.clone();
        let start = self.0.obj_id().ok_or("group has no location")?;
        let mut state = file.state.write();
        if state.read_only {
            return Err("unable to relink: file is read-only".into());
        }
        let idx = state
            .group_data(start)
            .and_then(|g| g.find_index(name))
            .ok_or_else(|| format!("link '{name}' not found"))?;
        let link = match &mut state.get_mut(start).kind {
            ObjectKind::Group(g) => g.links.remove(idx),
            _ => unreachable!(),
        };
        // insert at new path
        let (parent, leaf) = match path.rfind('/') {
            Some(pos) => {
                let (dir, leaf) = path.split_at(pos);
                let dir = if dir.is_empty() { "/" } else { dir };
                let parent = state
                    .resolve(start, dir)
                    .ok_or_else(|| format!("parent group '{dir}' not found"))?;
                (parent, &leaf[1..])
            }
            None => (start, path),
        };
        match &mut state.get_mut(parent).kind {
            ObjectKind::Group(g) => {
                g.links.push(Link {
                    name: leaf.to_string(),
                    ..link
                });
                Ok(())
            }
            _ => Err("target parent is not a group".into()),
        }
    }

    /// Removes a link from the group.
    pub fn unlink(&self, name: &str) -> Result<()> {
        let file = self.0.file().ok_or("group is not file-resident")?.clone();
        let start = self.0.obj_id().ok_or("group has no location")?;
        let mut state = file.state.write();
        if state.read_only {
            return Err("unable to unlink: file is read-only".into());
        }
        let idx = state
            .group_data(start)
            .and_then(|g| g.find_index(name))
            .ok_or_else(|| format!("link '{name}' not found"))?;
        let link = match &mut state.get_mut(start).kind {
            ObjectKind::Group(g) => g.links.remove(idx),
            _ => unreachable!(),
        };
        if let LinkTarget::Hard(id) = link.target {
            let node = state.get_mut(id);
            node.refcount = node.refcount.saturating_sub(1);
        }
        Ok(())
    }

    /// Returns true if a link with the given name exists.
    pub fn link_exists(&self, name: &str) -> bool {
        let (file, start) = match (self.0.file(), self.0.obj_id()) {
            (Some(f), Some(s)) => (f, s),
            _ => return false,
        };
        let state = file.state.read();
        // resolve all but last component, then check final link existence
        match name.rfind('/') {
            Some(pos) => {
                let (dir, leaf) = name.split_at(pos);
                let dir = if dir.is_empty() { "/" } else { dir };
                state
                    .resolve(start, dir)
                    .and_then(|p| state.group_data(p))
                    .map(|g| g.find(&leaf[1..]).is_some())
                    .unwrap_or(false)
            }
            None => state
                .group_data(start)
                .map(|g| g.find(name).is_some())
                .unwrap_or(false),
        }
    }

    /// Instantiates a typed dataset builder.
    pub fn new_dataset<T: H5Type>(&self) -> DatasetBuilderEmpty {
        self.new_dataset_builder().empty::<T>()
    }

    /// Instantiates a generic dataset builder.
    pub fn new_dataset_builder(&self) -> DatasetBuilder {
        DatasetBuilder::new(self)
    }

    /// Opens an existing dataset by name.
    pub fn dataset(&self, name: &str) -> Result<Dataset> {
        let loc = self.open_by_path(name)?;
        let file = loc.0.file().unwrap().clone();
        let id = loc.0.obj_id().unwrap();
        if !file.state.read().is_dataset(id) {
            return Err(format!("'{name}' is not a dataset").into());
        }
        Ok(Dataset::from_handle(Handle::new(Payload::Dataset {
            file,
            id,
        })))
    }

    /// Visits all links in the group, folding a value through the callback.
    /// The callback returns `true` to continue iteration.
    pub fn iter_visit<F, G>(
        &self,
        iteration_order: IterationOrder,
        traversal_order: TraversalOrder,
        mut val: G,
        op: F,
    ) -> Result<G>
    where
        F: Fn(&Self, &str, LinkInfo, &mut G) -> bool,
    {
        let mut links = self.with_group(|g| g.links.clone())?;
        match traversal_order {
            TraversalOrder::Name => links.sort_by(|a, b| a.name.cmp(&b.name)),
            TraversalOrder::Creation => links.sort_by_key(|l| l.creation_order),
        }
        if iteration_order == IterationOrder::Decreasing {
            links.reverse();
        }
        for link in links {
            let info = LinkInfo {
                link_type: match link.target {
                    LinkTarget::Hard(_) => LinkType::Hard,
                    LinkTarget::Soft(_) => LinkType::Soft,
                    LinkTarget::External { .. } => LinkType::External,
                },
                creation_order: Some(link.creation_order),
                is_utf8: link.utf8,
            };
            if !op(self, &link.name, info, &mut val) {
                break;
            }
        }
        Ok(val)
    }

    /// Visits all links with default ordering.
    pub fn iter_visit_default<F, G>(&self, val: G, op: F) -> Result<G>
    where
        F: Fn(&Self, &str, LinkInfo, &mut G) -> bool,
    {
        self.iter_visit(
            IterationOrder::default(),
            TraversalOrder::default(),
            val,
            op,
        )
    }

    fn members_of_kind(
        &self,
        want_group: bool,
        want_dataset: bool,
        want_type: bool,
    ) -> Result<Vec<Location>> {
        let file = self.0.file().ok_or("group is not file-resident")?.clone();
        let start = self.0.obj_id().ok_or("group has no location")?;
        let mut out = Vec::new();
        let state = file.state.read();
        let mut links = state.group_data(start).ok_or("not a group")?.links.clone();
        links.sort_by(|a, b| a.name.cmp(&b.name));
        for link in links {
            if let LinkTarget::Hard(id) = link.target {
                if let Some(node) = state.try_get(id) {
                    let payload = match &node.kind {
                        ObjectKind::Group(_) if want_group => Some(Payload::Group {
                            file: file.clone(),
                            id,
                        }),
                        ObjectKind::Dataset(_) if want_dataset => Some(Payload::Dataset {
                            file: file.clone(),
                            id,
                        }),
                        ObjectKind::NamedType(_) if want_type => Some(Payload::NamedType {
                            file: file.clone(),
                            id,
                        }),
                        _ => None,
                    };
                    if let Some(p) = payload {
                        out.push(Location::from_handle(Handle::new(p)));
                    }
                }
            }
        }
        Ok(out)
    }

    /// Returns all groups in the group, non-recursively.
    pub fn groups(&self) -> Result<Vec<Self>> {
        Ok(self
            .members_of_kind(true, false, false)?
            .into_iter()
            .map(|l| Self::from_handle(l.0))
            .collect())
    }

    /// Returns all datasets in the group, non-recursively.
    pub fn datasets(&self) -> Result<Vec<Dataset>> {
        Ok(self
            .members_of_kind(false, true, false)?
            .into_iter()
            .map(|l| Dataset::from_handle(l.0))
            .collect())
    }

    /// Returns all named datatypes in the group, non-recursively.
    pub fn named_datatypes(&self) -> Result<Vec<Datatype>> {
        Ok(self
            .members_of_kind(false, false, true)?
            .into_iter()
            .map(|l| Datatype::from_handle(l.0))
            .collect())
    }

    /// Returns the names of all group members.
    pub fn member_names(&self) -> Result<Vec<String>> {
        self.iter_visit_default(vec![], |_, name, _, names| {
            names.push(name.to_owned());
            true
        })
    }
}
