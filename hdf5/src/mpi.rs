//! A pure-Rust MPI subset (cargo feature `mpi`) sufficient for SPMD parallel
//! HDF5 workflows between Rust processes on one host (or across hosts via
//! TCP). This is **not** interoperable with OpenMPI/MPICH — it exists so
//! that N processes can cooperatively produce a single HDF5 file with
//! MPI-style collective semantics.
//!
//! Model (mirroring parallel HDF5's rules):
//! - all ranks open the file collectively and perform *identical* metadata
//!   calls in the same order (create_group / create_dataset / attributes);
//! - each rank writes its own dataset selections independently;
//! - [`crate::File::close`] is **collective**: rank 0 gathers every rank's write
//!   log, merges it, and writes the single physical file. Variable-length
//!   types cannot be written in MPI mode (the same restriction real
//!   parallel HDF5 imposes).
//!
//! Rendezvous: rank 0 listens on `H5MPI_PORT`; other ranks connect and
//! identify themselves. `spawn_workers` launches N copies of the current
//! executable with the environment prepared (a minimal `mpiexec -n`).
//!
//! # Alternative transport: the `mpi-rs` crate (cargo feature `mpi-rs`)
//!
//! With the `mpi-rs` feature, [`Comm`] can instead run on top of the
//! [`mpi-rs`](https://crates.io/crates/mpi-rs) crate — a pure-Rust MPI
//! implementation with an rsmpi-compatible API and a real `mpiexec`
//! launcher. [`Comm::init`] picks the transport automatically:
//!
//! - `H5MPI_RANK` set → the built-in TCP star (spawned via [`spawn_workers`]);
//! - otherwise → the `mpi-rs` runtime: the `mpiexec`-launched world, the
//!   application's own already-initialized universe, or a singleton
//!   (1 rank) when run standalone.
//!
//! Everything above the communicator — [`create`], [`open`], collective
//! close — is identical for both transports.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use crate::error::Result;

/// Reduction operations for [`Comm::allreduce_u64`] / [`Comm::allreduce_f64`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Sum,
    Min,
    Max,
}

enum Links {
    /// Rank 0: one stream per other rank (index 0 unused).
    Root(Vec<Mutex<TcpStream>>),
    /// Other ranks: single stream to rank 0.
    Leaf(Mutex<TcpStream>),
    /// Transport delegated to the `mpi-rs` crate. Holds the universe when
    /// this `Comm` initialized MPI itself (`None` = adopted an environment
    /// the application initialized), keeping it alive for the comm's life.
    #[cfg(feature = "mpi-rs")]
    MpiRs { _universe: Option<mpi_rs::Universe> },
}

/// Point-to-point send over the mpi-rs world (length-delimited, like frames).
#[cfg(feature = "mpi-rs")]
fn mpirs_send(rank: usize, data: &[u8]) {
    use mpi_rs::point_to_point::Destination;
    use mpi_rs::topology::Communicator;
    mpi_rs::topology::SimpleCommunicator::world()
        .process_at_rank(rank as mpi_rs::Rank)
        .send(data);
}

/// Point-to-point receive from a specific rank over the mpi-rs world.
#[cfg(feature = "mpi-rs")]
fn mpirs_recv(rank: usize) -> Vec<u8> {
    use mpi_rs::point_to_point::Source;
    use mpi_rs::topology::Communicator;
    mpi_rs::topology::SimpleCommunicator::world()
        .process_at_rank(rank as mpi_rs::Rank)
        .receive_vec::<u8>()
        .0
}

/// An MPI-style communicator over TCP with a star topology through rank 0.
#[derive(Clone)]
pub struct Comm {
    rank: usize,
    size: usize,
    links: Arc<Links>,
}

impl std::fmt::Debug for Comm {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Comm(rank {} of {})", self.rank, self.size)
    }
}

fn send_frame(s: &mut TcpStream, data: &[u8]) -> Result<()> {
    s.write_all(&(data.len() as u64).to_le_bytes())?;
    s.write_all(data)?;
    Ok(())
}

fn recv_frame(s: &mut TcpStream) -> Result<Vec<u8>> {
    let mut len = [0u8; 8];
    s.read_exact(&mut len)?;
    let n = u64::from_le_bytes(len) as usize;
    let mut buf = vec![0u8; n];
    s.read_exact(&mut buf)?;
    Ok(buf)
}

impl Comm {
    /// Join the world communicator. Collective: blocks until every rank has
    /// joined.
    ///
    /// When `H5MPI_RANK` / `H5MPI_SIZE` / `H5MPI_PORT` are set (a world
    /// launched via [`spawn_workers`]), joins the built-in TCP transport.
    /// Otherwise, with the `mpi-rs` feature, initializes (or adopts) the
    /// `mpi-rs` runtime: an `mpiexec`-launched world or a singleton.
    pub fn init() -> Result<Self> {
        #[cfg(feature = "mpi-rs")]
        if std::env::var("H5MPI_RANK").is_err() {
            return Ok(Self::wrap_mpi_rs(mpi_rs::initialize()));
        }
        Self::init_tcp()
    }

    /// Build a communicator over an `mpi-rs` universe the application
    /// initialized itself (`mpi::initialize()` from the `mpi-rs` crate).
    /// The universe is owned by the returned `Comm` (and its clones);
    /// dropping the last clone finalizes MPI.
    #[cfg(feature = "mpi-rs")]
    pub fn from_mpi_rs(universe: mpi_rs::Universe) -> Self {
        Self::wrap_mpi_rs(Some(universe))
    }

    #[cfg(feature = "mpi-rs")]
    fn wrap_mpi_rs(universe: Option<mpi_rs::Universe>) -> Self {
        use mpi_rs::topology::Communicator;
        let world = mpi_rs::topology::SimpleCommunicator::world();
        Self {
            rank: world.rank() as usize,
            size: world.size() as usize,
            links: Arc::new(Links::MpiRs { _universe: universe }),
        }
    }

    fn init_tcp() -> Result<Self> {
        let rank: usize = std::env::var("H5MPI_RANK")
            .map_err(|_| "H5MPI_RANK not set (launch via mpi::spawn_workers)")?
            .parse()
            .map_err(|_| "bad H5MPI_RANK")?;
        let size: usize = std::env::var("H5MPI_SIZE")
            .map_err(|_| "H5MPI_SIZE not set")?
            .parse()
            .map_err(|_| "bad H5MPI_SIZE")?;
        let port: u16 = std::env::var("H5MPI_PORT")
            .map_err(|_| "H5MPI_PORT not set")?
            .parse()
            .map_err(|_| "bad H5MPI_PORT")?;
        let host = std::env::var("H5MPI_HOST").unwrap_or_else(|_| "127.0.0.1".into());
        if size == 0 || rank >= size {
            return Err("invalid H5MPI_RANK/H5MPI_SIZE".into());
        }
        let links = if rank == 0 {
            let listener = TcpListener::bind(("0.0.0.0", port))?;
            let mut slots: Vec<Option<TcpStream>> = (0..size).map(|_| None).collect();
            for _ in 1..size {
                let (mut s, _) = listener.accept()?;
                s.set_nodelay(true).ok();
                let hello = recv_frame(&mut s)?;
                let r = u32::from_le_bytes(hello.get(..4).ok_or("bad hello")?.try_into().unwrap())
                    as usize;
                if r == 0 || r >= size || slots[r].is_some() {
                    return Err("duplicate or invalid rank in rendezvous".into());
                }
                slots[r] = Some(s);
            }
            // confirm the world is complete
            let mut streams = Vec::with_capacity(size);
            for (i, slot) in slots.into_iter().enumerate() {
                match slot {
                    Some(mut s) => {
                        send_frame(&mut s, b"ok")?;
                        streams.push(Mutex::new(s));
                    }
                    None if i == 0 => {
                        // self slot: placeholder connection to our own listener
                        let s = TcpStream::connect(("127.0.0.1", port))?;
                        streams.push(Mutex::new(s));
                    }
                    None => unreachable!(),
                }
            }
            Links::Root(streams)
        } else {
            let mut last_err: crate::error::Error = "unreachable".into();
            let mut stream = None;
            for _ in 0..600 {
                match TcpStream::connect((host.as_str(), port)) {
                    Ok(mut s) => {
                        s.set_nodelay(true).ok();
                        send_frame(&mut s, &(rank as u32).to_le_bytes())?;
                        let ack = recv_frame(&mut s)?;
                        if ack != b"ok" {
                            return Err("rendezvous rejected".into());
                        }
                        stream = Some(s);
                        break;
                    }
                    Err(e) => {
                        last_err = e.into();
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                }
            }
            let s = stream.ok_or_else(|| {
                crate::error::Error::from(format!("cannot reach rank 0: {last_err}"))
            })?;
            Links::Leaf(Mutex::new(s))
        };
        Ok(Self {
            rank,
            size,
            links: Arc::new(links),
        })
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn size(&self) -> usize {
        self.size
    }

    fn send_to_root(&self, data: &[u8]) -> Result<()> {
        match &*self.links {
            Links::Leaf(s) => send_frame(&mut s.lock().unwrap(), data),
            Links::Root(_) => Err("send_to_root called on rank 0".into()),
            #[cfg(feature = "mpi-rs")]
            Links::MpiRs { .. } if self.rank == 0 => Err("send_to_root called on rank 0".into()),
            #[cfg(feature = "mpi-rs")]
            Links::MpiRs { .. } => {
                mpirs_send(0, data);
                Ok(())
            }
        }
    }

    fn recv_from_root(&self) -> Result<Vec<u8>> {
        match &*self.links {
            Links::Leaf(s) => recv_frame(&mut s.lock().unwrap()),
            Links::Root(_) => Err("recv_from_root called on rank 0".into()),
            #[cfg(feature = "mpi-rs")]
            Links::MpiRs { .. } if self.rank == 0 => Err("recv_from_root called on rank 0".into()),
            #[cfg(feature = "mpi-rs")]
            Links::MpiRs { .. } => Ok(mpirs_recv(0)),
        }
    }

    fn root_send(&self, rank: usize, data: &[u8]) -> Result<()> {
        match &*self.links {
            Links::Root(streams) => send_frame(&mut streams[rank].lock().unwrap(), data),
            Links::Leaf(_) => Err("root_send called on a non-root rank".into()),
            #[cfg(feature = "mpi-rs")]
            Links::MpiRs { .. } if self.rank != 0 => Err("root_send called on a non-root rank".into()),
            #[cfg(feature = "mpi-rs")]
            Links::MpiRs { .. } => {
                mpirs_send(rank, data);
                Ok(())
            }
        }
    }

    fn root_recv(&self, rank: usize) -> Result<Vec<u8>> {
        match &*self.links {
            Links::Root(streams) => recv_frame(&mut streams[rank].lock().unwrap()),
            Links::Leaf(_) => Err("root_recv called on a non-root rank".into()),
            #[cfg(feature = "mpi-rs")]
            Links::MpiRs { .. } if self.rank != 0 => Err("root_recv called on a non-root rank".into()),
            #[cfg(feature = "mpi-rs")]
            Links::MpiRs { .. } => Ok(mpirs_recv(rank)),
        }
    }

    /// Collective barrier.
    pub fn barrier(&self) -> Result<()> {
        #[cfg(feature = "mpi-rs")]
        if let Links::MpiRs { .. } = &*self.links {
            use mpi_rs::collective::CommunicatorCollectives;
            mpi_rs::topology::SimpleCommunicator::world().barrier();
            return Ok(());
        }
        if self.rank == 0 {
            for r in 1..self.size {
                self.root_recv(r)?;
            }
            for r in 1..self.size {
                self.root_send(r, b"")?;
            }
        } else {
            self.send_to_root(b"")?;
            self.recv_from_root()?;
        }
        Ok(())
    }

    /// Collective broadcast from `root`; every rank returns the root's bytes.
    pub fn bcast(&self, data: &[u8], root: usize) -> Result<Vec<u8>> {
        // relay through rank 0
        let at0: Option<Vec<u8>> = if root == 0 {
            if self.rank == 0 {
                Some(data.to_vec())
            } else {
                None
            }
        } else if self.rank == root {
            self.send_to_root(data)?;
            None
        } else if self.rank == 0 {
            Some(self.root_recv(root)?)
        } else {
            None
        };
        if self.rank == 0 {
            let payload = at0.unwrap();
            for r in 1..self.size {
                self.root_send(r, &payload)?;
            }
            Ok(payload)
        } else {
            self.recv_from_root()
        }
    }

    /// Collective gather to `root`. Returns `Some(vec_per_rank)` on the root
    /// and `None` elsewhere.
    pub fn gather(&self, data: &[u8], root: usize) -> Result<Option<Vec<Vec<u8>>>> {
        if self.rank == 0 {
            let mut parts = vec![data.to_vec()];
            for r in 1..self.size {
                parts.push(self.root_recv(r)?);
            }
            if root == 0 {
                Ok(Some(parts))
            } else {
                // relay the whole set to the requested root
                let mut blob = Vec::new();
                for p in &parts {
                    blob.extend_from_slice(&(p.len() as u64).to_le_bytes());
                    blob.extend_from_slice(p);
                }
                self.root_send(root, &blob)?;
                Ok(None)
            }
        } else {
            self.send_to_root(data)?;
            if self.rank == root {
                let blob = self.recv_from_root()?;
                let mut parts = Vec::with_capacity(self.size);
                let mut c = &blob[..];
                for _ in 0..self.size {
                    let n = u64::from_le_bytes(c[..8].try_into().unwrap()) as usize;
                    parts.push(c[8..8 + n].to_vec());
                    c = &c[8 + n..];
                }
                Ok(Some(parts))
            } else {
                Ok(None)
            }
        }
    }

    /// Collective all-gather; every rank returns every rank's bytes.
    pub fn allgather(&self, data: &[u8]) -> Result<Vec<Vec<u8>>> {
        let gathered = self.gather(data, 0)?;
        let blob = if self.rank == 0 {
            let mut blob = Vec::new();
            for p in gathered.unwrap() {
                blob.extend_from_slice(&(p.len() as u64).to_le_bytes());
                blob.extend_from_slice(&p);
            }
            blob
        } else {
            Vec::new()
        };
        let blob = self.bcast(&blob, 0)?;
        let mut parts = Vec::with_capacity(self.size);
        let mut c = &blob[..];
        for _ in 0..self.size {
            let n = u64::from_le_bytes(c[..8].try_into().unwrap()) as usize;
            parts.push(c[8..8 + n].to_vec());
            c = &c[8 + n..];
        }
        Ok(parts)
    }

    /// Collective reduction over one `u64` per rank.
    pub fn allreduce_u64(&self, value: u64, op: Op) -> Result<u64> {
        let parts = self.allgather(&value.to_le_bytes())?;
        let vals = parts
            .iter()
            .map(|p| u64::from_le_bytes(p[..8].try_into().unwrap()));
        Ok(match op {
            Op::Sum => vals.sum(),
            Op::Min => vals.min().unwrap(),
            Op::Max => vals.max().unwrap(),
        })
    }

    /// Collective reduction over one `f64` per rank.
    pub fn allreduce_f64(&self, value: f64, op: Op) -> Result<f64> {
        let parts = self.allgather(&value.to_le_bytes())?;
        let vals = parts
            .iter()
            .map(|p| f64::from_le_bytes(p[..8].try_into().unwrap()));
        Ok(match op {
            Op::Sum => vals.sum(),
            Op::Min => vals.fold(f64::INFINITY, f64::min),
            Op::Max => vals.fold(f64::NEG_INFINITY, f64::max),
        })
    }
}

/// Spawn `n` copies of the current executable with the `H5MPI_*` environment
/// prepared (a minimal single-host `mpiexec -n`). Returns the children; the
/// caller waits on them. Not called from within a rank.
pub fn spawn_workers(n: usize) -> Result<Vec<std::process::Child>> {
    let exe = std::env::current_exe()?;
    // reserve a port by binding to :0, then release it for rank 0
    let port = TcpListener::bind(("127.0.0.1", 0))?.local_addr()?.port();
    let mut children = Vec::with_capacity(n);
    for rank in 0..n {
        children.push(
            std::process::Command::new(&exe)
                .args(std::env::args().skip(1))
                .env("H5MPI_RANK", rank.to_string())
                .env("H5MPI_SIZE", n.to_string())
                .env("H5MPI_PORT", port.to_string())
                .spawn()?,
        );
    }
    Ok(children)
}

/// Returns true when running inside a spawned rank — either one launched by
/// [`spawn_workers`] or (with the `mpi-rs` feature) by mpi-rs's `mpiexec`.
pub fn is_worker() -> bool {
    if std::env::var("H5MPI_RANK").is_ok() {
        return true;
    }
    #[cfg(feature = "mpi-rs")]
    if std::env::var("MPI_PMI_ROOT").is_ok() {
        return true;
    }
    false
}

// ---------------------------------------------------------------------------
// HDF5 integration: collective files
// ---------------------------------------------------------------------------

/// Collectively create a file: every rank must call this with the same path.
/// All ranks perform identical metadata operations (SPMD); each rank writes
/// its own dataset selections; [`crate::File::close`] is collective and
/// produces the single physical file (written by rank 0).
pub fn create<P: AsRef<std::path::Path>>(path: P, comm: &Comm) -> Result<crate::File> {
    // only rank 0 touches the filesystem during the run
    let file = if comm.rank() == 0 {
        crate::File::create(path)?
    } else {
        crate::File::create_mpi_replica(path)?
    };
    file.mpi_attach(comm.clone())?;
    comm.barrier()?;
    Ok(file)
}

/// Collectively open an existing file read-write. Every rank parses the same
/// bytes; the collective close writes rank 0's merged view.
pub fn open<P: AsRef<std::path::Path>>(path: P, comm: &Comm) -> Result<crate::File> {
    // rank 0 opens first so a racing create-then-open elsewhere cannot bite
    let file = if comm.rank() == 0 {
        let f = crate::File::open_rw(&path)?;
        comm.barrier()?;
        f
    } else {
        comm.barrier()?;
        crate::File::open_rw(&path)?
    };
    file.mpi_attach(comm.clone())?;
    comm.barrier()?;
    Ok(file)
}

/// One logged dataset write: byte ranges into the dataset's model buffer.
pub(crate) struct LogEntry {
    pub path: String,
    /// (destination byte offset, length) runs into the dataset data buffer.
    pub ranges: Vec<(u64, u64)>,
    /// Concatenated bytes for the runs, in order.
    pub bytes: Vec<u8>,
}

/// Per-file MPI state attached to `FileInner`.
pub(crate) struct MpiFile {
    pub comm: Comm,
    pub log: Vec<LogEntry>,
}

fn encode_log(log: &[LogEntry]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&(log.len() as u64).to_le_bytes());
    for e in log {
        b.extend_from_slice(&(e.path.len() as u32).to_le_bytes());
        b.extend_from_slice(e.path.as_bytes());
        b.extend_from_slice(&(e.ranges.len() as u64).to_le_bytes());
        for &(o, l) in &e.ranges {
            b.extend_from_slice(&o.to_le_bytes());
            b.extend_from_slice(&l.to_le_bytes());
        }
        b.extend_from_slice(&(e.bytes.len() as u64).to_le_bytes());
        b.extend_from_slice(&e.bytes);
    }
    b
}

fn decode_log(mut c: &[u8]) -> Result<Vec<LogEntry>> {
    let mut take = |n: usize| -> Result<&[u8]> {
        if c.len() < n {
            return Err("truncated MPI write log".into());
        }
        let (a, b) = c.split_at(n);
        c = b;
        Ok(a)
    };
    let n = u64::from_le_bytes(take(8)?.try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let pl = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        let path = String::from_utf8_lossy(take(pl)?).into_owned();
        let nr = u64::from_le_bytes(take(8)?.try_into().unwrap()) as usize;
        let mut ranges = Vec::with_capacity(nr);
        for _ in 0..nr {
            let o = u64::from_le_bytes(take(8)?.try_into().unwrap());
            let l = u64::from_le_bytes(take(8)?.try_into().unwrap());
            ranges.push((o, l));
        }
        let bl = u64::from_le_bytes(take(8)?.try_into().unwrap()) as usize;
        let bytes = take(bl)?.to_vec();
        out.push(LogEntry {
            path,
            ranges,
            bytes,
        });
    }
    Ok(out)
}

impl MpiFile {
    /// Collective close: gather all ranks' write logs on rank 0, replay them
    /// into rank 0's model, and let rank 0 (alone) write the file.
    /// Returns true if this rank should perform the physical write.
    pub(crate) fn collective_merge(&mut self, state: &mut crate::model::FileState) -> Result<bool> {
        let encoded = encode_log(&self.log);
        let gathered = self.comm.gather(&encoded, 0)?;
        if self.comm.rank() != 0 {
            self.log.clear();
            return Ok(false);
        }
        state.materialize_all()?;
        for (rank, blob) in gathered.unwrap().into_iter().enumerate() {
            // rank 0's own writes are already in its model
            if rank == 0 {
                continue;
            }
            for e in decode_log(&blob)? {
                let id = state
                    .resolve(state.root, &e.path)
                    .ok_or_else(|| format!("MPI merge: no dataset at {}", e.path))?;
                let crate::model::ObjectKind::Dataset(d) = &mut state.get_mut(id).kind else {
                    return Err(format!("MPI merge: {} is not a dataset", e.path).into());
                };
                d.materialize()?;
                let mut src = 0usize;
                for &(o, l) in &e.ranges {
                    let (o, l) = (o as usize, l as usize);
                    if o + l > d.data.len() || src + l > e.bytes.len() {
                        return Err("MPI merge: write out of bounds".into());
                    }
                    d.data[o..o + l].copy_from_slice(&e.bytes[src..src + l]);
                    src += l;
                }
            }
        }
        self.log.clear();
        Ok(true)
    }
}

#[cfg(all(test, feature = "mpi-rs"))]
mod tests {
    use super::*;

    /// Singleton world over the mpi-rs backend: `init` without any launcher
    /// env must yield rank 0 of 1, working collectives, and a collective
    /// file that closes into a readable single-rank HDF5 file.
    /// (Multi-rank operation is exercised by `examples/mpi_rs_demo.rs`
    /// under mpi-rs's `mpiexec`.)
    #[test]
    fn mpi_rs_singleton_collective_file() {
        assert!(std::env::var("H5MPI_RANK").is_err(), "test env leaked");
        let comm = Comm::init().unwrap();
        assert_eq!((comm.rank(), comm.size()), (0, 1));

        comm.barrier().unwrap();
        assert_eq!(comm.bcast(b"tok", 0).unwrap(), b"tok");
        assert_eq!(comm.allgather(b"me").unwrap(), vec![b"me".to_vec()]);
        assert_eq!(comm.allreduce_u64(7, Op::Sum).unwrap(), 7);
        assert_eq!(comm.allreduce_f64(2.5, Op::Max).unwrap(), 2.5);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mpi_rs_singleton.h5");
        let f = super::create(&path, &comm).unwrap();
        let ds = f.new_dataset::<i32>().shape([comm.size(), 3]).create("x").unwrap();
        ds.write_slice(&ndarray::arr1(&[1i32, 2, 3]), ndarray::s![0..1, ..])
            .unwrap();
        f.close().unwrap();

        let f = crate::File::open(&path).unwrap();
        let x: Vec<i32> = f.dataset("x").unwrap().read_raw().unwrap();
        assert_eq!(x, [1, 2, 3]);
    }
}
