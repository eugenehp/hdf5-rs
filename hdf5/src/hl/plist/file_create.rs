//! File creation property list.

use std::fmt::{self, Debug};
use std::ops::Deref;

use bitflags::bitflags;

use crate::class::ObjectClass;
use crate::error::Result;
use crate::h5i::H5I_type_t;
use crate::handle::Handle;
use crate::hl::plist::common::{AttrCreationOrder, AttrPhaseChange};
use crate::hl::plist::{PlistState, PropertyList};

pub(crate) const PROPERTY_NAMES: &[&str] = &[
    "userblock",
    "sym_k",
    "istore_k",
    "shared_mesg_phase_change",
    "shared_mesg_indexes",
    "obj_track_times",
    "attr_phase_change",
    "attr_creation_order",
];

/// Size of object offsets and lengths in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SizeofInfo {
    pub sizeof_addr: usize,
    pub sizeof_size: usize,
}

impl Default for SizeofInfo {
    fn default() -> Self {
        Self {
            sizeof_addr: 8,
            sizeof_size: 8,
        }
    }
}

/// Symbol table B-tree parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SymbolTableInfo {
    pub tree_rank: u32,
    pub node_size: u32,
}

impl Default for SymbolTableInfo {
    fn default() -> Self {
        Self {
            tree_rank: 16,
            node_size: 4,
        }
    }
}

/// Shared message phase-change parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PhaseChangeInfo {
    pub max_list: u32,
    pub min_btree: u32,
}

bitflags! {
    /// Types of messages that can be shared.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
    pub struct SharedMessageType: u32 {
        // values match H5O_SHMESG_*_FLAG: 1 << message_type_id
        const NONE = 0x0;
        const SIMPLE_DATASPACE = 1 << 0x1;
        const DATATYPE = 1 << 0x3;
        const FILL_VALUE = 1 << 0x5;
        const FILTER_PIPELINE = 1 << 0xb;
        const ATTRIBUTE = 1 << 0xc;
        const ALL = (1 << 0x1) | (1 << 0x3) | (1 << 0x5) | (1 << 0xb) | (1 << 0xc);
    }
}

/// Configuration of one shared message index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct SharedMessageIndex {
    pub message_types: SharedMessageType,
    pub min_message_size: u32,
}

/// File space management strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileSpaceStrategy {
    FreeSpaceManager {
        paged: bool,
        persist: bool,
        threshold: u64,
    },
    PageAggregation,
    None,
}

impl Default for FileSpaceStrategy {
    fn default() -> Self {
        Self::FreeSpaceManager {
            paged: false,
            persist: false,
            threshold: 1,
        }
    }
}

/// The data carried by a file-creation property list.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FileCreateData {
    pub userblock: u64,
    pub sym_k: SymbolTableInfo,
    pub istore_k: u32,
    pub shared_mesg_phase_change: PhaseChangeInfo,
    pub shared_mesg_indexes: Vec<SharedMessageIndex>,
    pub obj_track_times: bool,
    pub attr_phase_change: AttrPhaseChange,
    pub attr_creation_order: AttrCreationOrder,
    pub file_space_page_size: u64,
    pub file_space_strategy: FileSpaceStrategy,
}

impl Default for FileCreateData {
    fn default() -> Self {
        Self {
            userblock: 0,
            sym_k: SymbolTableInfo::default(),
            istore_k: 32,
            shared_mesg_phase_change: PhaseChangeInfo::default(),
            shared_mesg_indexes: Vec::new(),
            obj_track_times: true,
            attr_phase_change: AttrPhaseChange::default(),
            attr_creation_order: AttrCreationOrder::default(),
            file_space_page_size: 4096,
            file_space_strategy: FileSpaceStrategy::default(),
        }
    }
}

/// File creation property list.
#[repr(transparent)]
#[derive(Clone)]
pub struct FileCreate(Handle);

impl ObjectClass for FileCreate {
    const NAME: &'static str = "file create property list";
    const VALID_TYPES: &'static [H5I_type_t] = &[H5I_type_t::H5I_GENPROP_LST];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }
}

impl Debug for FileCreate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.debug_fmt(f)
    }
}

impl Deref for FileCreate {
    type Target = PropertyList;

    fn deref(&self) -> &PropertyList {
        unsafe { self.transmute() }
    }
}

impl PartialEq for FileCreate {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl Eq for FileCreate {}

impl Default for FileCreate {
    fn default() -> Self {
        Self::try_new().unwrap()
    }
}

impl FileCreate {
    pub(crate) fn from_data(data: FileCreateData) -> Self {
        Self(PropertyList::from_state(PlistState::FileCreate(data)).0)
    }

    pub(crate) fn data(&self) -> FileCreateData {
        match self.0.plist_state() {
            Some(PlistState::FileCreate(d)) => d.clone(),
            _ => FileCreateData::default(),
        }
    }

    pub fn try_new() -> Result<Self> {
        Ok(Self::from_data(FileCreateData::default()))
    }

    pub fn copy(&self) -> Self {
        Self::from_data(self.data())
    }

    pub fn build() -> FileCreateBuilder {
        FileCreateBuilder::new()
    }

    // getters

    pub fn userblock(&self) -> u64 {
        self.data().userblock
    }

    pub fn sizes(&self) -> SizeofInfo {
        SizeofInfo::default()
    }

    pub fn sym_k(&self) -> SymbolTableInfo {
        self.data().sym_k
    }

    pub fn istore_k(&self) -> u32 {
        self.data().istore_k
    }

    pub fn shared_mesg_phase_change(&self) -> PhaseChangeInfo {
        self.data().shared_mesg_phase_change
    }

    pub fn shared_mesg_indexes(&self) -> Vec<SharedMessageIndex> {
        self.data().shared_mesg_indexes
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

    pub fn file_space_page_size(&self) -> u64 {
        self.data().file_space_page_size
    }

    pub fn file_space_strategy(&self) -> FileSpaceStrategy {
        self.data().file_space_strategy
    }

    // aliased getters (parity with the FFI crate)

    pub fn get_userblock(&self) -> Result<u64> {
        Ok(self.userblock())
    }

    pub fn get_sizes(&self) -> Result<SizeofInfo> {
        Ok(self.sizes())
    }

    pub fn get_sym_k(&self) -> Result<SymbolTableInfo> {
        Ok(self.sym_k())
    }

    pub fn get_istore_k(&self) -> Result<u32> {
        Ok(self.istore_k())
    }

    pub fn get_shared_mesg_phase_change(&self) -> Result<PhaseChangeInfo> {
        Ok(self.shared_mesg_phase_change())
    }

    pub fn get_shared_mesg_indexes(&self) -> Result<Vec<SharedMessageIndex>> {
        Ok(self.shared_mesg_indexes())
    }

    pub fn get_obj_track_times(&self) -> Result<bool> {
        Ok(self.obj_track_times())
    }

    pub fn get_attr_phase_change(&self) -> Result<AttrPhaseChange> {
        Ok(self.attr_phase_change())
    }

    pub fn get_attr_creation_order(&self) -> Result<AttrCreationOrder> {
        Ok(self.attr_creation_order())
    }

    pub fn get_file_space_page_size(&self) -> Result<u64> {
        Ok(self.file_space_page_size())
    }

    pub fn get_file_space_strategy(&self) -> Result<FileSpaceStrategy> {
        Ok(self.file_space_strategy())
    }
}

/// Builder for file creation property lists.
#[derive(Clone, Debug, Default)]
pub struct FileCreateBuilder {
    data: FileCreateData,
}

impl FileCreateBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_plist(plist: &FileCreate) -> Result<Self> {
        Ok(Self { data: plist.data() })
    }

    pub fn userblock(&mut self, size: u64) -> &mut Self {
        self.data.userblock = size;
        self
    }

    pub fn sym_k(&mut self, tree_rank: u32, node_size: u32) -> &mut Self {
        self.data.sym_k = SymbolTableInfo {
            tree_rank,
            node_size,
        };
        self
    }

    pub fn istore_k(&mut self, ik: u32) -> &mut Self {
        self.data.istore_k = ik;
        self
    }

    pub fn shared_mesg_phase_change(&mut self, max_list: u32, min_btree: u32) -> &mut Self {
        self.data.shared_mesg_phase_change = PhaseChangeInfo {
            max_list,
            min_btree,
        };
        self
    }

    pub fn shared_mesg_indexes(&mut self, indexes: &[SharedMessageIndex]) -> &mut Self {
        self.data.shared_mesg_indexes = indexes.to_vec();
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

    pub fn file_space_page_size(&mut self, page_size: u64) -> &mut Self {
        self.data.file_space_page_size = page_size;
        self
    }

    pub fn file_space_strategy(&mut self, strategy: FileSpaceStrategy) -> &mut Self {
        self.data.file_space_strategy = strategy;
        self
    }

    pub fn apply(&self, plist: &mut FileCreate) -> Result<()> {
        *plist = FileCreate::from_data(self.data.clone());
        Ok(())
    }

    pub fn finish(&self) -> Result<FileCreate> {
        Ok(FileCreate::from_data(self.data.clone()))
    }
}
