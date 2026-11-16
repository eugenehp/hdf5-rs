//! Decoded-chunk cache: what libhdf5 calls the raw-data chunk cache.
//!
//! Partial reads (`read_slice`, `read_rows_raw`, point selections) of a
//! filtered dataset decode whole chunks; reading neighbouring rows again would
//! decode the same chunks again. Each lazily stored dataset keeps its most
//! recently used decoded chunks, up to [`chunk_cache_size`] bytes, and later
//! partial reads take them from here. Whole-dataset reads bypass it: they
//! touch every chunk once and would only evict what partial reads keep.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Default budget per dataset. libhdf5's default is 1 MiB, which a single
/// typical compressed chunk already exceeds (libhdf5 then skips caching it);
/// 16 MiB keeps a working set of several chunks.
pub const DEFAULT_CHUNK_CACHE: usize = 16 << 20;

static BUDGET: AtomicUsize = AtomicUsize::new(DEFAULT_CHUNK_CACHE);

/// Set how many bytes of decoded chunks each dataset may keep between partial
/// reads (default 16 MiB). `0` disables the cache.
pub fn set_chunk_cache_size(bytes: usize) {
    BUDGET.store(bytes, Ordering::Relaxed);
}

/// The per-dataset decoded-chunk budget (see [`set_chunk_cache_size`]).
pub fn chunk_cache_size() -> usize {
    BUDGET.load(Ordering::Relaxed)
}

/// Least-recently-used decoded chunks of one dataset, keyed by file address.
#[derive(Default)]
pub struct ChunkCache {
    inner: crate::lock::Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    map: crate::util::AddrMap<u64, (Arc<[u8]>, u64)>,
    bytes: usize,
    tick: u64,
}

impl ChunkCache {
    pub fn get(&self, addr: u64) -> Option<Arc<[u8]>> {
        let mut inner = self.inner.lock();
        inner.tick += 1;
        let tick = inner.tick;
        let (chunk, used) = inner.map.get_mut(&addr)?;
        *used = tick;
        Some(chunk.clone())
    }

    pub fn put(&self, addr: u64, chunk: Arc<[u8]>) {
        let budget = chunk_cache_size();
        if chunk.len() > budget {
            return;
        }
        let mut inner = self.inner.lock();
        while inner.bytes + chunk.len() > budget {
            // evict the least recently used; a cache holds few chunks
            let Some((&oldest, _)) = inner.map.iter().min_by_key(|(_, (_, used))| *used) else {
                break;
            };
            if let Some((gone, _)) = inner.map.remove(&oldest) {
                inner.bytes -= gone.len();
            }
        }
        inner.tick += 1;
        let tick = inner.tick;
        inner.bytes += chunk.len();
        if let Some((old, _)) = inner.map.insert(addr, (chunk, tick)) {
            inner.bytes -= old.len();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evicts_least_recently_used() {
        let cache = ChunkCache::default();
        let mb = |b: u8| Arc::<[u8]>::from(vec![b; 6 << 20]);
        cache.put(1, mb(1));
        cache.put(2, mb(2));
        assert!(cache.get(1).is_some()); // 1 is now the most recent
        cache.put(3, mb(3)); // 18 MiB > 16: evicts 2
        assert!(cache.get(2).is_none());
        assert_eq!(cache.get(1).unwrap()[0], 1);
        assert_eq!(cache.get(3).unwrap()[0], 3);
    }
}
