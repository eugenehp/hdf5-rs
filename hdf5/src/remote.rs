//! Reading files over async I/O — HTTP range requests from the browser
//! (`fetch`), object stores, anything that cannot block.
//!
//! The parser is synchronous and random-access, so reads run against a
//! [`FetchSource`]: a read of bytes not yet fetched fails and is noted. After
//! each attempt, every noted range is fetched at once (concurrently), handed
//! to the source, and the read runs again on a freshly opened view of the
//! file. Fetched bytes are kept, so a retry costs only parsing, and an
//! attempt usually finds all the bytes one level of metadata needs: opening a
//! file takes a few round-trips, a dataset read one or two more.
//!
//! ```no_run
//! # async fn demo(reader: impl hdf5::remote::AsyncRangeRead) -> hdf5::Result<()> {
//! use hdf5::remote::RemoteFile;
//! let remote = RemoteFile::open(reader, 4 << 20).await?;
//! let shape = remote.read(|f| Ok(f.dataset("data")?.shape())).await?;
//! let first = remote.read(|f| f.dataset("data")?.read_rows_raw::<f32>(0, 10)).await?;
//! # Ok(()) }
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use crate::error::Result;
use crate::format::source::FetchSource;
use crate::hl::file::File;

/// Async random access to a file's bytes — e.g. `fetch` with a `Range`
/// header.
pub trait AsyncRangeRead {
    /// The file's size in bytes.
    fn len(&self) -> u64;

    /// Whether the file is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes `[offset, offset + len)`.
    fn read_range(&self, offset: u64, len: usize)
        -> impl Future<Output = std::io::Result<Vec<u8>>>;
}

/// A file read through an [`AsyncRangeRead`].
pub struct RemoteFile<R> {
    reader: R,
    src: Arc<FetchSource>,
    round_trips: std::sync::atomic::AtomicUsize,
}

/// Attempts before giving up on a read that keeps finding more to fetch.
const MAX_ATTEMPTS: usize = 64;

impl<R: AsyncRangeRead> RemoteFile<R> {
    /// Open the file: its superblock and root group are fetched now. `block`
    /// (a power of two) is the fetch size: larger means fewer round-trips
    /// where metadata clusters, smaller fewer wasted bytes where it does not.
    pub async fn open(reader: R, block: usize) -> Result<Self> {
        let src = Arc::new(FetchSource::new(reader.len(), block)?);
        let remote = Self {
            reader,
            src,
            round_trips: 0.into(),
        };
        remote.read(|_| Ok(())).await?;
        Ok(remote)
    }

    /// Run `op` against the file, fetching whatever it turns out to need.
    /// `op` may run several times (each on a fresh view of the file), so it
    /// should only read.
    pub async fn read<T>(&self, mut op: impl FnMut(&File) -> Result<T>) -> Result<T> {
        for _ in 0..MAX_ATTEMPTS {
            let result = File::from_source(self.src.clone()).and_then(|f| op(&f));
            let missing = self.src.missing();
            if missing.is_empty() {
                return result;
            }
            // whatever the attempt made of the gaps, it is not the answer:
            // fetch them all, then go again
            self.round_trips
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let fetches = missing.iter().map(|&(start, end)| async move {
                let bytes = self
                    .reader
                    .read_range(start, (end - start) as usize)
                    .await?;
                if bytes.len() as u64 != end - start {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "short range read",
                    ));
                }
                Ok((start, bytes))
            });
            for fetched in join_all(fetches.collect()).await {
                let (start, bytes) = fetched?;
                self.src.insert(start, bytes);
            }
        }
        Err("remote read kept needing more data".into())
    }

    /// Round-trips made so far (each fetching every range one attempt
    /// needed).
    pub fn round_trips(&self) -> usize {
        self.round_trips.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Bytes fetched so far.
    pub fn bytes_fetched(&self) -> u64 {
        self.src.bytes_fetched()
    }
}

/// Poll all `futures` together; their outputs, in order. (No executor
/// dependency: this works under any.)
async fn join_all<F: Future>(futures: Vec<F>) -> Vec<F::Output> {
    struct JoinAll<F: Future> {
        futures: Vec<Pin<Box<F>>>,
        done: Vec<Option<F::Output>>,
    }
    impl<F: Future> Future for JoinAll<F> {
        type Output = Vec<F::Output>;
        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            // SAFETY-free: the boxed futures are pinned on the heap, and
            // nothing here is structurally pinned
            let this = Pin::into_inner(self);
            let mut pending = false;
            for (f, slot) in this.futures.iter_mut().zip(this.done.iter_mut()) {
                if slot.is_none() {
                    match f.as_mut().poll(cx) {
                        Poll::Ready(v) => *slot = Some(v),
                        Poll::Pending => pending = true,
                    }
                }
            }
            if pending {
                Poll::Pending
            } else {
                Poll::Ready(
                    this.done
                        .iter_mut()
                        .map(|d| d.take().expect("ready"))
                        .collect(),
                )
            }
        }
    }
    impl<F: Future> Unpin for JoinAll<F> {}
    let done = futures.iter().map(|_| None).collect();
    JoinAll {
        futures: futures.into_iter().map(Box::pin).collect(),
        done,
    }
    .await
}
