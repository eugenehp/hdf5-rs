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
                state.materialize_all();
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
        state.materialize_all();
        let path = inner.path.as_ref().ok_or("file has no backing path")?;
        let bytes = format::serialize(&state)?;
        fs::write(path, bytes)?;
        Ok(())
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
                    state.materialize_all();
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
            state: parking_lot::RwLock::new(state),
            id: next_id(),
            externals: parking_lot::Mutex::new(std::collections::HashMap::new()),
            mpi: parking_lot::Mutex::new(None),
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
            if !state.read_only {
                state.materialize_all();
                if let Some(path) = inner.path.as_ref() {
                    if let Ok(bytes) = format::serialize(&state) {
                        let _ = fs::write(path, bytes);
                    }
                }
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
                // memory-map the file: the OS pages data in on demand and
                // lazily-referenced datasets avoid up-front copies entirely
                let file = fs::File::open(&path)?;
                let image = std::sync::Arc::new(crate::model::FileImage::Mmap(unsafe {
                    memmap2::Mmap::map(&file)?
                }));
                let mut state = format::parse_image(&image, path.parent())?;
                state.read_only = mode == OpenMode::Read;
                state
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
                    let bytes = fs::read(&path)?;
                    let mut state = format::parse_at(&bytes, path.parent())?;
                    state.read_only = false;
                    state
                } else {
                    FileState::new_empty()
                }
            }
        };

        let inner = Arc::new(FileInner {
            path: Some(path),
            mode,
            state: parking_lot::RwLock::new(state),
            id: next_id(),
            externals: parking_lot::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "mpi")]
            mpi: parking_lot::Mutex::new(None),
        });

        // Persist immediately on create so the file exists on disk.
        if matches!(mode, OpenMode::Create | OpenMode::CreateExcl)
            || (mode == OpenMode::Append && !exists)
        {
            let state = inner.state.read();
            let bytes = format::serialize(&state)?;
            fs::write(inner.path.as_ref().unwrap(), bytes)?;
        }

        Ok(File::from_handle(Handle::new(Payload::File(inner))))
    }
}
