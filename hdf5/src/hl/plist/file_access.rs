//! File access property list.

use std::fmt::{self, Debug};
use std::ops::Deref;

use crate::class::ObjectClass;
use crate::error::Result;
use crate::h5i::H5I_type_t;
use crate::handle::Handle;
use crate::hl::plist::{PlistState, PropertyList};

pub(crate) const PROPERTY_NAMES: &[&str] = &[
    "driver",
    "fclose_degree",
    "alignment",
    "chunk_cache",
    "meta_block_size",
    "sieve_buf_size",
    "gc_references",
    "small_data_block_size",
    "libver_bounds",
];

/// File close degree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FileCloseDegree {
    #[default]
    Default,
    Weak,
    Semi,
    Strong,
}

/// Memory ("core") file driver settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoreDriver {
    pub increment: usize,
    pub filebacked: bool,
    pub write_tracking: usize,
}

impl Default for CoreDriver {
    fn default() -> Self {
        Self {
            increment: 1 << 20,
            filebacked: false,
            write_tracking: 0,
        }
    }
}

/// Family file driver settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct FamilyDriver {
    pub member_size: usize,
}

/// Split file driver settings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitDriver {
    pub meta_ext: String,
    pub raw_ext: String,
}

impl Default for SplitDriver {
    fn default() -> Self {
        Self {
            meta_ext: "-m.h5".into(),
            raw_ext: "-r.h5".into(),
        }
    }
}

/// The file driver to use.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum FileDriver {
    #[default]
    Sec2,
    Stdio,
    Core(CoreDriver),
    Family(FamilyDriver),
    Split(SplitDriver),
    Log,
}

/// Raw data chunk cache parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChunkCache {
    pub nslots: usize,
    pub nbytes: usize,
    pub w0: f64,
}

impl Default for ChunkCache {
    fn default() -> Self {
        Self {
            nslots: 521,
            nbytes: 1 << 20,
            w0: 0.75,
        }
    }
}

impl Eq for ChunkCache {}

/// Library version bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LibraryVersion {
    #[default]
    Earliest,
    V18,
    V110,
    V112,
    V114,
}

impl LibraryVersion {
    pub fn is_earliest(self) -> bool {
        self == Self::Earliest
    }
}

/// Alignment properties.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Alignment {
    pub threshold: u64,
    pub alignment: u64,
}

impl Default for Alignment {
    fn default() -> Self {
        Self {
            threshold: 1,
            alignment: 1,
        }
    }
}

/// Library version bounds for objects written to the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct LibVerBounds {
    pub low: LibraryVersion,
    pub high: LibraryVersion,
}

/// The data carried by a file-access property list.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FileAccessData {
    pub driver: FileDriver,
    pub evict_on_close: bool,
    pub write_tracking: usize,
    pub metadata_read_attempts: u32,
    pub mdc_config: MetadataCacheConfig,
    pub mdc_image_config: CacheImageConfig,
    pub mdc_log_options: CacheLogOptions,
    pub page_buffer_size: PageBufferSize,
    pub elink_file_cache_size: u32,
    pub all_coll_metadata_ops: bool,
    pub coll_metadata_write: bool,
    pub fclose_degree: FileCloseDegree,
    pub alignment: Alignment,
    pub chunk_cache: ChunkCache,
    pub meta_block_size: u64,
    pub sieve_buf_size: usize,
    pub gc_references: bool,
    pub small_data_block_size: u64,
    pub libver_bounds: LibVerBounds,
}

impl Default for FileAccessData {
    fn default() -> Self {
        Self {
            driver: FileDriver::default(),
            evict_on_close: false,
            write_tracking: 0,
            metadata_read_attempts: 1,
            mdc_config: MetadataCacheConfig::default(),
            mdc_image_config: CacheImageConfig::default(),
            mdc_log_options: CacheLogOptions::default(),
            page_buffer_size: PageBufferSize::default(),
            elink_file_cache_size: 0,
            all_coll_metadata_ops: false,
            coll_metadata_write: false,
            fclose_degree: FileCloseDegree::default(),
            alignment: Alignment::default(),
            chunk_cache: ChunkCache::default(),
            meta_block_size: 2048,
            sieve_buf_size: 64 * 1024,
            gc_references: false,
            small_data_block_size: 2048,
            libver_bounds: LibVerBounds::default(),
        }
    }
}

/// File access property list.
#[repr(transparent)]
#[derive(Clone)]
pub struct FileAccess(Handle);

impl ObjectClass for FileAccess {
    const NAME: &'static str = "file access property list";
    const VALID_TYPES: &'static [H5I_type_t] = &[H5I_type_t::H5I_GENPROP_LST];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }
}

impl Debug for FileAccess {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.debug_fmt(f)
    }
}

impl Deref for FileAccess {
    type Target = PropertyList;

    fn deref(&self) -> &PropertyList {
        unsafe { self.transmute() }
    }
}

impl PartialEq for FileAccess {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl Eq for FileAccess {}

impl Default for FileAccess {
    fn default() -> Self {
        Self::try_new().unwrap()
    }
}

impl FileAccess {
    pub(crate) fn from_data(data: FileAccessData) -> Self {
        Self(PropertyList::from_state(PlistState::FileAccess(data)).0)
    }

    pub(crate) fn data(&self) -> FileAccessData {
        match self.0.plist_state() {
            Some(PlistState::FileAccess(d)) => d.clone(),
            _ => FileAccessData::default(),
        }
    }

    pub fn try_new() -> Result<Self> {
        Ok(Self::from_data(FileAccessData::default()))
    }

    pub fn copy(&self) -> Self {
        Self::from_data(self.data())
    }

    pub fn build() -> FileAccessBuilder {
        FileAccessBuilder::new()
    }

    pub fn get_driver(&self) -> Result<FileDriver> {
        self.driver()
    }

    pub fn get_evict_on_close(&self) -> Result<bool> {
        Ok(self.data().evict_on_close)
    }

    pub fn evict_on_close(&self) -> bool {
        self.data().evict_on_close
    }

    pub fn get_metadata_read_attempts(&self) -> Result<u32> {
        Ok(self.data().metadata_read_attempts)
    }

    pub fn metadata_read_attempts(&self) -> u32 {
        self.data().metadata_read_attempts
    }

    pub fn get_mdc_config(&self) -> Result<MetadataCacheConfig> {
        Ok(self.data().mdc_config)
    }

    pub fn mdc_config(&self) -> MetadataCacheConfig {
        self.data().mdc_config
    }

    pub fn get_mdc_image_config(&self) -> Result<CacheImageConfig> {
        Ok(self.data().mdc_image_config)
    }

    pub fn mdc_image_config(&self) -> CacheImageConfig {
        self.data().mdc_image_config
    }

    pub fn get_mdc_log_options(&self) -> Result<CacheLogOptions> {
        Ok(self.data().mdc_log_options)
    }

    pub fn mdc_log_options(&self) -> CacheLogOptions {
        self.data().mdc_log_options
    }

    pub fn get_page_buffer_size(&self) -> Result<PageBufferSize> {
        Ok(self.data().page_buffer_size)
    }

    pub fn page_buffer_size(&self) -> PageBufferSize {
        self.data().page_buffer_size
    }

    pub fn get_elink_file_cache_size(&self) -> Result<u32> {
        Ok(self.data().elink_file_cache_size)
    }

    pub fn elink_file_cache_size(&self) -> u32 {
        self.data().elink_file_cache_size
    }

    pub fn get_all_coll_metadata_ops(&self) -> Result<bool> {
        Ok(self.data().all_coll_metadata_ops)
    }

    pub fn all_coll_metadata_ops(&self) -> bool {
        self.data().all_coll_metadata_ops
    }

    pub fn get_coll_metadata_write(&self) -> Result<bool> {
        Ok(self.data().coll_metadata_write)
    }

    pub fn coll_metadata_write(&self) -> bool {
        self.data().coll_metadata_write
    }

    pub fn write_tracking(&self) -> usize {
        self.data().write_tracking
    }

    pub fn driver(&self) -> Result<FileDriver> {
        Ok(self.data().driver)
    }

    pub fn get_fclose_degree(&self) -> Result<FileCloseDegree> {
        Ok(self.data().fclose_degree)
    }

    pub fn fclose_degree(&self) -> FileCloseDegree {
        self.data().fclose_degree
    }

    pub fn get_alignment(&self) -> Result<Alignment> {
        Ok(self.data().alignment)
    }

    pub fn alignment(&self) -> Alignment {
        self.data().alignment
    }

    pub fn get_chunk_cache(&self) -> Result<ChunkCache> {
        Ok(self.data().chunk_cache)
    }

    pub fn chunk_cache(&self) -> ChunkCache {
        self.data().chunk_cache
    }

    pub fn get_meta_block_size(&self) -> Result<u64> {
        Ok(self.data().meta_block_size)
    }

    pub fn meta_block_size(&self) -> u64 {
        self.data().meta_block_size
    }

    pub fn get_sieve_buf_size(&self) -> Result<usize> {
        Ok(self.data().sieve_buf_size)
    }

    pub fn sieve_buf_size(&self) -> usize {
        self.data().sieve_buf_size
    }

    pub fn get_gc_references(&self) -> Result<bool> {
        Ok(self.data().gc_references)
    }

    pub fn gc_references(&self) -> bool {
        self.data().gc_references
    }

    pub fn get_small_data_block_size(&self) -> Result<u64> {
        Ok(self.data().small_data_block_size)
    }

    pub fn small_data_block_size(&self) -> u64 {
        self.data().small_data_block_size
    }

    pub fn get_libver_bounds(&self) -> Result<LibVerBounds> {
        Ok(self.data().libver_bounds)
    }

    pub fn libver(&self) -> LibraryVersion {
        self.data().libver_bounds.low
    }
}

/// Builder for file access property lists.
#[derive(Clone, Debug, Default)]
pub struct FileAccessBuilder {
    data: FileAccessData,
}

impl FileAccessBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_plist(plist: &FileAccess) -> Result<Self> {
        Ok(Self { data: plist.data() })
    }

    pub fn driver(&mut self, driver: &FileDriver) -> &mut Self {
        self.data.driver = driver.clone();
        self
    }

    pub fn evict_on_close(&mut self, v: bool) -> &mut Self {
        self.data.evict_on_close = v;
        self
    }

    pub fn write_tracking(&mut self, page_size: usize) -> &mut Self {
        self.data.write_tracking = page_size;
        self
    }

    pub fn metadata_read_attempts(&mut self, n: u32) -> &mut Self {
        self.data.metadata_read_attempts = n;
        self
    }

    pub fn mdc_config(&mut self, c: &MetadataCacheConfig) -> &mut Self {
        self.data.mdc_config = c.clone();
        self
    }

    pub fn mdc_image_config(&mut self, c: &CacheImageConfig) -> &mut Self {
        self.data.mdc_image_config = *c;
        self
    }

    pub fn mdc_log_options(&mut self, o: &CacheLogOptions) -> &mut Self {
        self.data.mdc_log_options = o.clone();
        self
    }

    /// Alias for [`Self::mdc_log_options`] (FFI-crate name).
    pub fn log_options(&mut self, o: &CacheLogOptions) -> &mut Self {
        self.mdc_log_options(o)
    }

    pub fn page_buffer_size(&mut self, p: &PageBufferSize) -> &mut Self {
        self.data.page_buffer_size = *p;
        self
    }

    pub fn elink_file_cache_size(&mut self, n: u32) -> &mut Self {
        self.data.elink_file_cache_size = n;
        self
    }

    pub fn all_coll_metadata_ops(&mut self, v: bool) -> &mut Self {
        self.data.all_coll_metadata_ops = v;
        self
    }

    pub fn coll_metadata_write(&mut self, v: bool) -> &mut Self {
        self.data.coll_metadata_write = v;
        self
    }

    /// Alias for the latest library-version bounds (FFI-crate name).
    pub fn libver_latest(&mut self) -> &mut Self {
        self.latest_libver()
    }

    pub fn multi(&mut self, d: &MultiDriver) -> &mut Self {
        let _ = d; // accepted for parity; the engine has one storage backend
        self
    }

    pub fn multi_options(
        &mut self,
        files: &[MultiFile],
        layout: &MultiLayout,
        relax: bool,
    ) -> &mut Self {
        let _ = (files, layout, relax);
        self
    }

    pub fn direct(&mut self) -> &mut Self {
        self
    }

    pub fn direct_options(
        &mut self,
        alignment: usize,
        block_size: usize,
        cbuf_size: usize,
    ) -> &mut Self {
        let _ = (alignment, block_size, cbuf_size);
        self
    }

    /// MPI-IO driver (parity name). With the `mpi` feature, attach the
    /// communicator via `hdf5::mpi::create`/`open` instead; without it this
    /// is accepted and ignored.
    pub fn mpio(&mut self) -> &mut Self {
        self
    }

    pub fn sec2(&mut self) -> &mut Self {
        self.data.driver = FileDriver::Sec2;
        self
    }

    pub fn stdio(&mut self) -> &mut Self {
        self.data.driver = FileDriver::Stdio;
        self
    }

    pub fn core(&mut self) -> &mut Self {
        self.data.driver = FileDriver::Core(CoreDriver::default());
        self
    }

    pub fn core_options(&mut self, increment: usize, filebacked: bool) -> &mut Self {
        self.data.driver = FileDriver::Core(CoreDriver {
            increment,
            filebacked,
            write_tracking: 0,
        });
        self
    }

    pub fn core_filebacked(&mut self, filebacked: bool) -> &mut Self {
        self.data.driver = FileDriver::Core(CoreDriver {
            filebacked,
            ..CoreDriver::default()
        });
        self
    }

    pub fn family(&mut self) -> &mut Self {
        self.data.driver = FileDriver::Family(FamilyDriver::default());
        self
    }

    pub fn family_options(&mut self, member_size: usize) -> &mut Self {
        self.data.driver = FileDriver::Family(FamilyDriver { member_size });
        self
    }

    pub fn split(&mut self) -> &mut Self {
        self.data.driver = FileDriver::Split(SplitDriver::default());
        self
    }

    pub fn split_options(&mut self, meta_ext: &str, raw_ext: &str) -> &mut Self {
        self.data.driver = FileDriver::Split(SplitDriver {
            meta_ext: meta_ext.into(),
            raw_ext: raw_ext.into(),
        });
        self
    }

    pub fn log(&mut self) -> &mut Self {
        self.data.driver = FileDriver::Log;
        self
    }

    pub fn fclose_degree(&mut self, degree: FileCloseDegree) -> &mut Self {
        self.data.fclose_degree = degree;
        self
    }

    pub fn alignment(&mut self, threshold: u64, alignment: u64) -> &mut Self {
        self.data.alignment = Alignment {
            threshold,
            alignment,
        };
        self
    }

    pub fn chunk_cache(&mut self, nslots: usize, nbytes: usize, w0: f64) -> &mut Self {
        self.data.chunk_cache = ChunkCache { nslots, nbytes, w0 };
        self
    }

    pub fn meta_block_size(&mut self, size: u64) -> &mut Self {
        self.data.meta_block_size = size;
        self
    }

    pub fn sieve_buf_size(&mut self, size: usize) -> &mut Self {
        self.data.sieve_buf_size = size;
        self
    }

    pub fn gc_references(&mut self, gc: bool) -> &mut Self {
        self.data.gc_references = gc;
        self
    }

    pub fn small_data_block_size(&mut self, size: u64) -> &mut Self {
        self.data.small_data_block_size = size;
        self
    }

    pub fn libver_bounds(&mut self, low: LibraryVersion, high: LibraryVersion) -> &mut Self {
        self.data.libver_bounds = LibVerBounds { low, high };
        self
    }

    pub fn libver_earliest(&mut self) -> &mut Self {
        self.libver_bounds(LibraryVersion::Earliest, LibraryVersion::V114)
    }

    pub fn libver_v18(&mut self) -> &mut Self {
        self.libver_bounds(LibraryVersion::V18, LibraryVersion::V114)
    }

    pub fn libver_v110(&mut self) -> &mut Self {
        self.libver_bounds(LibraryVersion::V110, LibraryVersion::V114)
    }

    pub fn libver_v112(&mut self) -> &mut Self {
        self.libver_bounds(LibraryVersion::V112, LibraryVersion::V114)
    }

    pub fn libver_v114(&mut self) -> &mut Self {
        self.libver_bounds(LibraryVersion::V114, LibraryVersion::V114)
    }

    pub fn latest_libver(&mut self) -> &mut Self {
        self.libver_v114()
    }

    pub fn apply(&self, plist: &mut FileAccess) -> Result<()> {
        *plist = FileAccess::from_data(self.data.clone());
        Ok(())
    }

    pub fn finish(&self) -> Result<FileAccess> {
        Ok(FileAccess::from_data(self.data.clone()))
    }
}

// ---------------------------------------------------------------------------
// Metadata-cache / paging knobs (accepted and stored; the pure-Rust engine
// has no metadata cache, so these are configuration echoes for API parity)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheIncreaseMode {
    Off,
    Threshold,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlashIncreaseMode {
    Off,
    AddSpace,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheDecreaseMode {
    Off,
    Threshold,
    AgeOut,
    AgeOutWithThreshold,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetadataWriteStrategy {
    ProcessZeroOnly,
    Distributed,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetadataCacheConfig {
    pub rpt_fcn_enabled: bool,
    pub open_trace_file: bool,
    pub close_trace_file: bool,
    pub trace_file_name: String,
    pub evictions_enabled: bool,
    pub set_initial_size: bool,
    pub initial_size: usize,
    pub min_clean_fraction: f64,
    pub max_size: usize,
    pub min_size: usize,
    pub epoch_length: i64,
    pub incr_mode: CacheIncreaseMode,
    pub lower_hr_threshold: f64,
    pub increment: f64,
    pub apply_max_increment: bool,
    pub max_increment: usize,
    pub flash_incr_mode: FlashIncreaseMode,
    pub flash_multiple: f64,
    pub flash_threshold: f64,
    pub decr_mode: CacheDecreaseMode,
    pub upper_hr_threshold: f64,
    pub decrement: f64,
    pub apply_max_decrement: bool,
    pub max_decrement: usize,
    pub epochs_before_eviction: i32,
    pub apply_empty_reserve: bool,
    pub empty_reserve: f64,
    pub dirty_bytes_threshold: usize,
    pub metadata_write_strategy: MetadataWriteStrategy,
}

impl Default for MetadataCacheConfig {
    fn default() -> Self {
        Self {
            rpt_fcn_enabled: false,
            open_trace_file: false,
            close_trace_file: false,
            trace_file_name: String::new(),
            evictions_enabled: true,
            set_initial_size: true,
            initial_size: 1 << 21,
            min_clean_fraction: 0.3,
            max_size: 1 << 25,
            min_size: 1 << 20,
            epoch_length: 50_000,
            incr_mode: CacheIncreaseMode::Threshold,
            lower_hr_threshold: 0.9,
            increment: 2.0,
            apply_max_increment: true,
            max_increment: 1 << 22,
            flash_incr_mode: FlashIncreaseMode::AddSpace,
            flash_multiple: 1.4,
            flash_threshold: 0.25,
            decr_mode: CacheDecreaseMode::AgeOutWithThreshold,
            upper_hr_threshold: 0.999,
            decrement: 0.9,
            apply_max_decrement: true,
            max_decrement: 1 << 20,
            epochs_before_eviction: 3,
            apply_empty_reserve: true,
            empty_reserve: 0.1,
            dirty_bytes_threshold: 1 << 18,
            metadata_write_strategy: MetadataWriteStrategy::Distributed,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheImageConfig {
    pub generate_image: bool,
    pub save_resize_status: bool,
    pub entry_ageout: i32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CacheLogOptions {
    pub is_enabled: bool,
    pub location: String,
    pub start_on_access: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PageBufferSize {
    pub buf_size: usize,
    pub min_meta_perc: u32,
    pub min_raw_perc: u32,
}

/// Multi-driver per-family file layout (parity; stored, not acted upon).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MultiLayout {
    pub mem_super: u8,
    pub mem_btree: u8,
    pub mem_draw: u8,
    pub mem_gheap: u8,
    pub mem_lheap: u8,
    pub mem_object: u8,
}

/// One member of a multi-driver family (parity).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MultiFile {
    pub name: String,
    pub addr: u64,
}

/// Multi file driver configuration (parity; stored, not acted upon).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MultiDriver {
    pub files: Vec<MultiFile>,
    pub layout: MultiLayout,
    pub relax: bool,
}

/// Direct (O_DIRECT) driver configuration (parity; stored, not acted upon).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DirectDriver {
    pub alignment: usize,
    pub block_size: usize,
    pub cbuf_size: usize,
}

bitflags::bitflags! {
    /// Log-driver flags (parity).
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct LogFlags: u64 {
        const TRUNCATE = 0x1;
        const META_IO = 0x2;
        const LOC_READ = 0x4;
        const LOC_WRITE = 0x8;
        const LOC_SEEK = 0x10;
        const LOC_IO = 0x1c;
        const FILE_READ = 0x20;
        const FILE_WRITE = 0x40;
        const FILE_IO = 0x60;
        const FLAVOR = 0x80;
        const NUM_READ = 0x100;
        const NUM_WRITE = 0x200;
        const NUM_SEEK = 0x400;
        const NUM_TRUNCATE = 0x800;
        const NUM_IO = 0xf00;
        const TIME_OPEN = 0x1000;
        const TIME_STAT = 0x2000;
        const TIME_READ = 0x4000;
        const TIME_WRITE = 0x8000;
        const TIME_SEEK = 0x10000;
        const TIME_CLOSE = 0x20000;
        const TIME_IO = 0x3f000;
        const ALLOC = 0x40000;
        const ALL = 0x7ffff;
    }
}

/// Log driver options (parity; stored, not acted upon).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LogOptions {
    pub logfile: Option<String>,
    pub flags: LogFlags,
    pub buf_size: usize,
}

/// MPI-IO driver marker (parity). Real collective files are provided by the
/// `mpi` feature via `hdf5::mpi::create`/`open`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MpioDriver;
