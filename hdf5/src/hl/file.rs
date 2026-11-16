//! HDF5 files.

use std::fmt::{self, Debug};
use std::fs;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::class::ObjectClass;
use crate::error::Result;
use crate::format;
use crate::h5i::H5I_type_t;
use crate::handle::{next_id, Handle, Payload};
use crate::hl::group::Group;
use crate::hl::plist::file_access::{FileAccess, FileAccessBuilder};
use crate::hl::plist::file_create::{FileCreate, FileCreateBuilder};
use crate::model::{FileInner, FileState};

/// File opening mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenMode {
    /// Open a file as read-only, file must exist.
    Read,
    /// Open a file as read/write, file must exist.
    ReadWrite,
    /// Create a file, truncate if exists.
    Create,
    /// Create a file, fail if exists.
    CreateExcl,
    /// Open a file as read/write if exists, create otherwise.
    Append,
}

/// An HDF5 file object.
#[repr(transparent)]
#[derive(Clone)]
pub struct File(Handle);

impl ObjectClass for File {
    const NAME: &'static str = "file";
    const VALID_TYPES: &'static [H5I_type_t] = &[H5I_type_t::H5I_FILE];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }

    fn short_repr(&self) -> Option<String> {
        let basename = Path::new(&self.filename())
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let mode = if self.is_read_only() {
            "read-only"
        } else {
            "read/write"
        };
        Some(format!("\"{basename}\" ({mode})"))
    }
}

impl Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.debug_fmt(f)
    }
}

impl Deref for File {
    type Target = Group;

    fn deref(&self) -> &Group {
        unsafe { self.transmute() }
    }
}

impl File {
    fn inner(&self) -> Result<&Arc<FileInner>> {
        self.0.file().ok_or_else(|| "invalid file handle".into())
    }

    /// Opens a file as read-only, file must exist.
    pub fn open<P: AsRef<Path>>(filename: P) -> Result<Self> {
        Self::open_as(filename, OpenMode::Read)
    }

    /// Opens a file as read/write, file must exist.
    pub fn open_rw<P: AsRef<Path>>(filename: P) -> Result<Self> {
        Self::open_as(filename, OpenMode::ReadWrite)
    }

    /// Creates a file, truncates if exists.
    pub fn create<P: AsRef<Path>>(filename: P) -> Result<Self> {
        Self::open_as(filename, OpenMode::Create)
    }

    /// Creates a file, fails if exists.
    pub fn create_excl<P: AsRef<Path>>(filename: P) -> Result<Self> {
        Self::open_as(filename, OpenMode::CreateExcl)
    }

    /// Opens a file as read/write if exists, creates otherwise.
    pub fn append<P: AsRef<Path>>(filename: P) -> Result<Self> {
        Self::open_as(filename, OpenMode::Append)
    }

    /// Creates an empty file that lives only in memory — no path, nothing is
    /// written to disk (`flush`/`close`/drop are no-ops). Get the HDF5 bytes with
    /// [`File::to_bytes`]; reopen them with [`File::open_reader`]. This is how a
    /// browser (wasm32, no filesystem) writes HDF5.
    ///
    /// ```
    /// let f = hdf5::File::create_in_memory().unwrap();
    /// f.new_dataset_builder().with_data(&[1i32, 2, 3]).create("x").unwrap();
    /// let bytes = f.to_bytes().unwrap();
    /// let g = hdf5::File::open_reader(std::io::Cursor::new(bytes)).unwrap();
    /// assert_eq!(g.dataset("x").unwrap().read_raw::<i32>().unwrap(), [1, 2, 3]);
    /// ```
    pub fn create_in_memory() -> Result<Self> {
        let mut state = FileState::new_empty();
        state.get_mut(state.root).mtime = crate::model::now();
        let inner = Arc::new(FileInner {
            path: None,
            mode: OpenMode::Create,
            state: crate::lock::RwLock::new(state),
            id: next_id(),
            externals: crate::lock::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "mpi")]
            mpi: crate::lock::Mutex::new(None),
        });
        Ok(File::from_handle(Handle::new(Payload::File(inner))))
    }

    /// Opens HDF5 bytes already in memory (read-only) — no file system
    /// needed, so this is the way in on `wasm32-unknown-unknown`, e.g. with
    /// the bytes of a `fetch` or a browser `File`. The buffer is used in
    /// place, and objects are parsed as they are reached.
    ///
    /// ```
    /// let f = hdf5::File::create_in_memory().unwrap();
    /// f.new_dataset_builder().with_data(&[1.5f64, 2.5]).create("x").unwrap();
    /// let g = hdf5::File::from_bytes(f.to_bytes().unwrap()).unwrap();
    /// assert_eq!(g.dataset("x").unwrap().read_raw::<f64>().unwrap(), [1.5, 2.5]);
    /// ```
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Result<Self> {
        let image = Arc::new(crate::model::FileImage::Bytes(bytes.into()));
        let source = crate::format::source::ImageSource::new(image);
        let mut state = format::reader::parse_source_lazy(Arc::new(source), None)?;
        state.read_only = true;
        Ok(Self::in_memory(state, OpenMode::Read))
    }

    /// Opens HDF5 bytes in memory for editing: changes stay in memory, and
    /// [`to_bytes`](Self::to_bytes) gives the updated file.
    ///
    /// ```
    /// let f = hdf5::File::create_in_memory().unwrap();
    /// f.new_dataset_builder().with_data(&[1u8, 2]).create("a").unwrap();
    /// let g = hdf5::File::from_bytes_rw(f.to_bytes().unwrap()).unwrap();
    /// g.new_dataset_builder().with_data(&[3u8]).create("b").unwrap();
    /// let h = hdf5::File::from_bytes(g.to_bytes().unwrap()).unwrap();
    /// assert_eq!(h.member_names().unwrap(), ["a", "b"]);
    /// ```
    pub fn from_bytes_rw(bytes: impl Into<Vec<u8>>) -> Result<Self> {
        let image = Arc::new(crate::model::FileImage::Bytes(bytes.into()));
        let mut state = format::parse_image(&image, None)?;
        state.read_only = false;
        Ok(Self::in_memory(state, OpenMode::ReadWrite))
    }

    fn in_memory(state: FileState, mode: OpenMode) -> Self {
        let inner = Arc::new(FileInner {
            path: None,
            mode,
            state: crate::lock::RwLock::new(state),
            id: next_id(),
            externals: crate::lock::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "mpi")]
            mpi: crate::lock::Mutex::new(None),
        });
        File::from_handle(Handle::new(Payload::File(inner)))
    }

    /// The file serialized as HDF5 bytes (what `flush` would write), for any
    /// file — in-memory, path-backed, or opened from a reader.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let inner = self.inner()?;
        let mut state = inner.state.write();
        state.force_all()?;
        state.materialize_all()?;
        // a lazy open leaves the user block unread
        if state.userblock_data.is_empty() && state.userblock > 0 {
            if let Some(loader) = state.loader.clone() {
                let size = usize::try_from(state.userblock).map_err(|_| "user block too large")?;
                state.userblock_data = loader.src.slice(0, size)?.to_vec();
            }
        }
        format::serialize(&state)
    }

    /// Opens a file (read-only) from any `Read + Seek` source instead of a path.
    ///
    /// The parser reads through the source lazily, fetching only the bytes it
    /// touches (superblock, object headers, B-tree/heap nodes) and — for lazily
    /// materialized datasets — only the chunks actually read. Nothing is copied
    /// up front, so this streams metadata from very large or remote files (e.g.
    /// an HTTP range-request reader) without materializing the whole file.
    ///
    /// ```no_run
    /// use std::io::Cursor;
    /// let bytes: Vec<u8> = std::fs::read("data.h5").unwrap();
    /// let file = hdf5::File::open_reader(Cursor::new(bytes)).unwrap();
    /// let shape = file.dataset("acquisition/series/data").unwrap().shape();
    /// ```
    pub fn open_reader<R>(reader: R) -> Result<Self>
    where
        R: std::io::Read + std::io::Seek + Send + 'static,
    {
        Self::open_reader_with_block(reader, crate::format::source::DEFAULT_BLOCK)
    }

    /// [`open_reader`](Self::open_reader), fetching `block` bytes at a time (a
    /// power of two). Over a network each read is a round-trip: larger suits
    /// files whose metadata clusters, smaller those that scatter it through
    /// the data, as files written while recording do.
    pub fn open_reader_with_block<R>(reader: R, block: usize) -> Result<Self>
    where
        R: std::io::Read + std::io::Seek + Send + 'static,
    {
        Self::from_source(std::sync::Arc::new(
            crate::format::source::RangeSource::new(reader)?.with_block(block)?,
        ))
    }

    /// Open (read-only, lazily) from any byte source.
    pub(crate) fn from_source(
        src: std::sync::Arc<dyn crate::format::source::ByteSource + Send + Sync>,
    ) -> Result<Self> {
        // Lazy parse: only the root is read now; children are parsed on demand,
        // so opening a huge many-object file is O(root).
        let mut state = crate::format::reader::parse_source_lazy(src, None)?;
        state.read_only = true;
        let inner = Arc::new(FileInner {
            path: None,
            mode: OpenMode::Read,
            state: crate::lock::RwLock::new(state),
            id: next_id(),
            externals: crate::lock::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "mpi")]
            mpi: crate::lock::Mutex::new(None),
        });
        Ok(File::from_handle(Handle::new(Payload::File(inner))))
    }

    /// Opens a file with the given mode.
    pub fn open_as<P: AsRef<Path>>(filename: P, mode: OpenMode) -> Result<Self> {
        FileBuilder::new().open_as(filename, mode)
    }

    /// Creates a file builder.
    pub fn with_options() -> FileBuilder {
        FileBuilder::new()
    }

    /// Returns the file size in bytes (as it would be serialized right now).
    pub fn size(&self) -> u64 {
        match self.inner() {
            Ok(inner) => {
                let mut state = inner.state.write();
                if state.materialize_all().is_err() {
                    return 0;
                }
                format::serialize(&state)
                    .map(|b| b.len() as u64)
                    .unwrap_or(0)
            }
            Err(_) => 0,
        }
    }

    /// Returns the free space in the file in bytes (always 0: the pure-Rust
    /// writer compacts on serialization).
    pub fn free_space(&self) -> u64 {
        0
    }

    /// Returns true if the file was opened in a read-only mode.
    pub fn is_read_only(&self) -> bool {
        self.inner()
            .map(|i| i.state.read().read_only)
            .unwrap_or(true)
    }

    /// Returns the userblock size in bytes.
    pub fn userblock(&self) -> u64 {
        self.inner().map(|i| i.state.read().userblock).unwrap_or(0)
    }

    /// Flushes the in-memory model to storage.
    pub fn flush(&self) -> Result<()> {
        let inner = self.inner()?;
        #[cfg(feature = "mpi")]
        if inner.mpi.lock().is_some() {
            // MPI mode: data becomes visible at the collective close
            return Ok(());
        }
        let mut state = inner.state.write();
        if state.read_only {
            return Ok(());
        }
        // an in-memory file (`create_in_memory`) has nowhere to flush to
        let Some(path) = inner.path.as_ref() else {
            return Ok(());
        };
        write_out(&mut state, path)
    }

    /// Closes the file, flushing pending changes.
    pub fn close(self) -> Result<()> {
        #[cfg(feature = "mpi")]
        {
            let inner = self.inner()?;
            let mut guard = inner.mpi.lock();
            if let Some(mpi) = guard.as_mut() {
                // Collective close: rank 0 merges all write logs and writes
                // the single physical file; other ranks write nothing.
                let mut state = inner.state.write();
                let write = mpi.collective_merge(&mut state)?;
                let comm = mpi.comm.clone();
                *guard = None; // detach so flush()/Drop stay local no-ops
                if write {
                    state.materialize_all()?;
                    let path = inner.path.as_ref().ok_or("file has no backing path")?;
                    let bytes = format::serialize(&state)?;
                    fs::write(path, bytes)?;
                }
                state.read_only = true; // suppress any Drop-flush afterwards
                drop(state);
                drop(guard);
                comm.barrier()?; // file is complete when close() returns
                return Ok(());
            }
        }
        self.flush()
    }

    /// Build an in-memory replica of a to-be-created file for a non-root
    /// MPI rank: same path, nothing persisted until the collective close.
    #[cfg(feature = "mpi")]
    pub(crate) fn create_mpi_replica<P: AsRef<Path>>(filename: P) -> Result<Self> {
        let path = std::path::PathBuf::from(filename.as_ref());
        let mut state = FileState::new_empty();
        state.get_mut(state.root).mtime = crate::model::now();
        let inner = Arc::new(FileInner {
            path: Some(path),
            mode: OpenMode::Create,
            state: crate::lock::RwLock::new(state),
            id: next_id(),
            externals: crate::lock::Mutex::new(std::collections::HashMap::new()),
            mpi: crate::lock::Mutex::new(None),
        });
        Ok(File::from_handle(Handle::new(Payload::File(inner))))
    }

    /// Attach MPI collective-file state (called by `hdf5::mpi`).
    #[cfg(feature = "mpi")]
    pub(crate) fn mpi_attach(&self, comm: crate::mpi::Comm) -> Result<()> {
        let inner = self.inner()?;
        *inner.mpi.lock() = Some(crate::mpi::MpiFile {
            comm,
            log: Vec::new(),
        });
        Ok(())
    }

    /// Returns a copy of the file access property list.
    pub fn access_plist(&self) -> Result<FileAccess> {
        FileAccess::try_new()
    }

    /// A short alias for `access_plist()`.
    pub fn fapl(&self) -> Result<FileAccess> {
        self.access_plist()
    }

    /// Returns a copy of the file creation property list.
    pub fn create_plist(&self) -> Result<FileCreate> {
        let mut b = FileCreateBuilder::new();
        b.userblock(self.userblock());
        b.finish()
    }

    /// A short alias for `create_plist()`.
    pub fn fcpl(&self) -> Result<FileCreate> {
        self.create_plist()
    }
}

impl Drop for File {
    fn drop(&mut self) {
        // Flush on the last file handle so `File::create(..)? ... drop` persists.
        if let Some(inner) = self.0.file() {
            #[cfg(feature = "mpi")]
            if inner.mpi.lock().is_some() {
                // MPI mode without a collective close: never write locally
                return;
            }
            // The FileInner Arc is shared by every object handle in the file;
            // flush whenever a File handle is dropped and the file is writable.
            let mut state = inner.state.write();
            // an in-memory file has nowhere to go: nothing to load or encode
            if let (false, Some(path)) = (state.read_only, inner.path.as_ref()) {
                let _ = write_out(&mut state, path);
            }
        }
    }
}

/// A builder to create or open files.
#[derive(Default)]
pub struct FileBuilder {
    fapl: FileAccessBuilder,
    fcpl: FileCreateBuilder,
}

impl FileBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Access the file-access property list builder.
    pub fn access_plist(&mut self) -> &mut FileAccessBuilder {
        &mut self.fapl
    }

    /// A short alias for `access_plist()`.
    pub fn fapl(&mut self) -> &mut FileAccessBuilder {
        &mut self.fapl
    }

    /// Access the file-create property list builder.
    pub fn create_plist(&mut self) -> &mut FileCreateBuilder {
        &mut self.fcpl
    }

    /// A short alias for `create_plist()`.
    pub fn fcpl(&mut self) -> &mut FileCreateBuilder {
        &mut self.fcpl
    }

    pub fn set_access_plist(&mut self, fapl: &FileAccess) -> Result<&mut Self> {
        self.fapl = FileAccessBuilder::from_plist(fapl)?;
        Ok(self)
    }

    pub fn set_fapl(&mut self, fapl: &FileAccess) -> Result<&mut Self> {
        self.set_access_plist(fapl)
    }

    pub fn set_create_plist(&mut self, fcpl: &FileCreate) -> Result<&mut Self> {
        self.fcpl = FileCreateBuilder::from_plist(fcpl)?;
        Ok(self)
    }

    pub fn set_fcpl(&mut self, fcpl: &FileCreate) -> Result<&mut Self> {
        self.set_create_plist(fcpl)
    }

    pub fn with_access_plist<F>(&mut self, func: F) -> &mut Self
    where
        F: Fn(&mut FileAccessBuilder) -> &mut FileAccessBuilder,
    {
        func(&mut self.fapl);
        self
    }

    pub fn with_fapl<F>(&mut self, func: F) -> &mut Self
    where
        F: Fn(&mut FileAccessBuilder) -> &mut FileAccessBuilder,
    {
        self.with_access_plist(func)
    }

    pub fn with_create_plist<F>(&mut self, func: F) -> &mut Self
    where
        F: Fn(&mut FileCreateBuilder) -> &mut FileCreateBuilder,
    {
        func(&mut self.fcpl);
        self
    }

    pub fn with_fcpl<F>(&mut self, func: F) -> &mut Self
    where
        F: Fn(&mut FileCreateBuilder) -> &mut FileCreateBuilder,
    {
        self.with_create_plist(func)
    }

    // --- open/create ---

    pub fn open<P: AsRef<Path>>(&self, filename: P) -> Result<File> {
        self.open_as(filename, OpenMode::Read)
    }

    pub fn open_rw<P: AsRef<Path>>(&self, filename: P) -> Result<File> {
        self.open_as(filename, OpenMode::ReadWrite)
    }

    pub fn create<P: AsRef<Path>>(&self, filename: P) -> Result<File> {
        self.open_as(filename, OpenMode::Create)
    }

    pub fn create_excl<P: AsRef<Path>>(&self, filename: P) -> Result<File> {
        self.open_as(filename, OpenMode::CreateExcl)
    }

    pub fn append<P: AsRef<Path>>(&self, filename: P) -> Result<File> {
        self.open_as(filename, OpenMode::Append)
    }

    pub fn open_as<P: AsRef<Path>>(&self, filename: P, mode: OpenMode) -> Result<File> {
        let path: PathBuf = filename.as_ref().to_path_buf();
        let exists = path.exists();

        let state = match mode {
            OpenMode::Read | OpenMode::ReadWrite => {
                if !exists {
                    return Err(
                        format!("unable to open file: '{}' (not found)", path.display()).into(),
                    );
                }
                if mode == OpenMode::ReadWrite {
                    open_appendable(&path)?
                } else {
                    // memory-map the file: the OS pages data in on demand and
                    // lazily-referenced datasets avoid up-front copies entirely
                    // (WASI has no mmap: read the file into an owned buffer instead)
                    let file = fs::File::open(&path)?;
                    let image = std::sync::Arc::new(match unsafe { memmap2::Mmap::map(&file) } {
                        Ok(m) => crate::model::FileImage::Mmap(m),
                        Err(e) if e.kind() == std::io::ErrorKind::Unsupported => {
                            crate::model::FileImage::Bytes(fs::read(&path)?)
                        }
                        Err(e) => return Err(e.into()),
                    });
                    // read-only, objects are parsed as they are reached: opening a
                    // file costs its root, however many objects it holds
                    let source = crate::format::source::ImageSource::new(image.clone());
                    let mut state = format::reader::parse_source_lazy(
                        std::sync::Arc::new(source),
                        path.parent(),
                    )?;
                    state.read_only = true;
                    state
                }
            }
            OpenMode::Create => {
                let mut state = FileState::new_empty();
                let fcpl = self.fcpl.finish()?;
                state.userblock = fcpl.userblock();
                state.sohm = fcpl
                    .shared_mesg_indexes()
                    .iter()
                    .filter(|ix| !ix.message_types.is_empty())
                    .map(|ix| (ix.message_types.bits() as u16, ix.min_message_size))
                    .collect();
                state.get_mut(state.root).mtime = crate::model::now();
                state
            }
            OpenMode::CreateExcl => {
                if exists {
                    return Err(format!(
                        "unable to create file: '{}' already exists",
                        path.display()
                    )
                    .into());
                }
                let mut state = FileState::new_empty();
                state.userblock = self.fcpl.finish()?.userblock();
                state
            }
            OpenMode::Append => {
                if exists {
                    open_appendable(&path)?
                } else {
                    FileState::new_empty()
                }
            }
        };

        let inner = Arc::new(FileInner {
            path: Some(path),
            mode,
            state: crate::lock::RwLock::new(state),
            id: next_id(),
            externals: crate::lock::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "mpi")]
            mpi: crate::lock::Mutex::new(None),
        });

        // Persist immediately on create so the file exists on disk.
        if matches!(mode, OpenMode::Create | OpenMode::CreateExcl)
            || (mode == OpenMode::Append && !exists)
        {
            let mut state = inner.state.write();
            let path = inner.path.as_ref().unwrap();
            fs::write(path, format::serialize(&state)?)?;
            // from here on, flushes append to it
            state.appendable = Some(Arc::new(crate::format::source::FileSource::open(path)?));
        }

        Ok(File::from_handle(Handle::new(Payload::File(inner))))
    }
}

/// Open `path` for writing: parsed through a [`FileSource`] that later flushes
/// append to, so data already in the file is never loaded or rewritten.
///
/// [`FileSource`]: crate::format::source::FileSource
fn open_appendable(path: &Path) -> Result<FileState> {
    let file = Arc::new(crate::format::source::FileSource::open(path)?);
    let mut state = format::reader::parse_source(file.clone(), path.parent(), 0)?;
    state.read_only = false;
    state.appendable = Some(file);
    Ok(state)
}

/// Write a file's changes to `path`: appended to it where possible (see
/// [`format::flush_append`]), else the whole file rewritten.
fn write_out(state: &mut FileState, path: &Path) -> Result<()> {
    if !state.dirty {
        return Ok(());
    }
    if format::flush_append(state)? {
        state.dirty = false;
        return Ok(());
    }
    state.materialize_all()?;
    let bytes = format::serialize(state)?;
    let reopen = state.appendable.take().is_some();
    fs::write(path, bytes)?;
    if reopen {
        state.appendable = Some(Arc::new(crate::format::source::FileSource::open(path)?));
    }
    state.dirty = false;
    Ok(())
}
