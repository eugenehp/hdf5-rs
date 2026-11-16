//! Data parallelism on a small built-in pool of persistent `std` threads —
//! no thread-pool dependency.
//!
//! Chunk decoding/encoding, numeric conversion and byte-swapping of large
//! buffers fan out over [`num_threads`] workers. Work is handed out one item
//! at a time from a shared counter, so chunks of uneven cost (one compresses
//! 10×, the next stores raw) still keep every core busy.
//!
//! Whether threads exist is found out at run time: `wasm32-wasip1-threads`
//! (and wasm hosts with thread support) spawn them, while
//! `wasm32-unknown-unknown` or plain `wasm32-wasip1` refuse the first spawn —
//! from then on everything runs inline on the calling thread, with identical
//! results. On wasm the default count is 1 (the host reports no cores); ask
//! for more with [`set_num_threads`] or `HDF5_NO_C_THREADS`.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Requested worker count; 0 = one per available core.
static THREADS: AtomicUsize = AtomicUsize::new(0);

/// Below this many bytes an element-wise pass (swap, convert) stays on one
/// thread: spawning costs more than it saves.
pub(crate) const MIN_PAR_BYTES: usize = 4 << 20;

/// Set once a spawn has failed: this host has no threads.
static NO_THREADS: AtomicBool = AtomicBool::new(false);

/// Set how many threads chunk decoding/encoding and large conversions may
/// use. `0` restores the default: one per available core, or the
/// `HDF5_NO_C_THREADS` environment variable when set. `1` makes every
/// operation run on the calling thread.
pub fn set_num_threads(n: usize) {
    THREADS.store(n, Ordering::Relaxed);
}

/// The number of threads parallel operations use (see [`set_num_threads`]).
pub fn num_threads() -> usize {
    if IN_WORKER.with(std::cell::Cell::get) {
        return 1;
    }
    if NO_THREADS.load(Ordering::Relaxed) && spawner().is_none() {
        return 1;
    }
    match THREADS.load(Ordering::Relaxed) {
        0 => default_threads(),
        n => n,
    }
}

fn default_threads() -> usize {
    static DEFAULT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *DEFAULT.get_or_init(|| {
        std::env::var("HDF5_NO_C_THREADS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
    })
}

/// A unit of work handed to a [`set_thread_spawner`] spawner.
pub type Task = Box<dyn FnOnce() + Send + 'static>;

type Spawner = std::sync::Arc<dyn Fn(Task) + Send + Sync>;

static SPAWNER: std::sync::RwLock<Option<Spawner>> = std::sync::RwLock::new(None);

/// Run parallel work on threads the application provides, instead of
/// `std::thread` — for hosts where the standard library cannot spawn, such as
/// the browser (`wasm32-unknown-unknown` built with `+atomics` and shared
/// memory), where `spawn` would post each task to a Web Worker.
///
/// Tasks may run on any thread, or never: a task still queued when the
/// caller has finished the work itself is cancelled, so a fixed-size pool
/// whose own threads call into this crate cannot deadlock. The calling thread
/// does block while started tasks finish, so call into this crate from a
/// Worker, not the browser's main thread, which may not block. Parallel
/// operations started inside a task run inline. Pair with [`set_num_threads`] (the browser
/// reports one core).
pub fn set_thread_spawner(spawn: impl Fn(Task) + Send + Sync + 'static) {
    *SPAWNER.write().unwrap_or_else(|e| e.into_inner()) = Some(std::sync::Arc::new(spawn));
}

/// Go back to spawning `std` threads.
pub fn clear_thread_spawner() {
    *SPAWNER.write().unwrap_or_else(|e| e.into_inner()) = None;
}

fn spawner() -> Option<Spawner> {
    SPAWNER.read().unwrap_or_else(|e| e.into_inner()).clone()
}

thread_local! {
    /// Set while this thread runs parallel work: nested parallel calls then
    /// stay inline instead of oversubscribing (or exhausting a host's pool).
    static IN_WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// `work` with this thread marked as a worker.
fn as_worker(work: &(dyn Fn() + Sync)) {
    let was = IN_WORKER.with(|w| w.replace(true));
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            IN_WORKER.with(|w| w.set(self.0));
        }
    }
    let _restore = Restore(was);
    work();
}

/// Run `work` on `n` threads (the caller's included), re-raising any panic.
fn run(n: usize, work: impl Fn() + Sync) {
    if n <= 1 {
        return work();
    }
    if let Some(spawn) = spawner() {
        return run_hosted(n, &work, &*spawn);
    }
    let helpers = pool().grow(n - 1);
    if helpers == 0 {
        return work();
    }
    run_hosted(helpers + 1, &work, &|task| pool().push(task));
}

/// Persistent worker threads, started as first needed and kept: a read
/// costs a queue push per helper, not a thread start — which on wasm (a
/// thread is a Worker instantiating the module) takes milliseconds.
struct Pool {
    queue: std::sync::Mutex<std::collections::VecDeque<Task>>,
    ready: std::sync::Condvar,
    workers: std::sync::Mutex<usize>,
}

fn pool() -> &'static Pool {
    static POOL: std::sync::OnceLock<Pool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| Pool {
        queue: Default::default(),
        ready: Default::default(),
        workers: std::sync::Mutex::new(0),
    })
}

impl Pool {
    /// Start workers until there are `want`; returns how many there are (fewer
    /// where the host refuses threads).
    fn grow(&'static self, want: usize) -> usize {
        let mut workers = self.workers.lock().unwrap_or_else(|e| e.into_inner());
        while *workers < want && !NO_THREADS.load(Ordering::Relaxed) {
            let spawned = std::thread::Builder::new()
                .name("hdf5-no-c".into())
                .spawn(move || self.serve());
            match spawned {
                Ok(_) => *workers += 1,
                Err(_) => NO_THREADS.store(true, Ordering::Relaxed),
            }
        }
        (*workers).min(want)
    }

    fn push(&self, task: Task) {
        self.queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(task);
        self.ready.notify_one();
    }

    fn serve(&self) {
        loop {
            let task = {
                let mut queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
                loop {
                    if let Some(task) = queue.pop_front() {
                        break task;
                    }
                    queue = self.ready.wait(queue).unwrap_or_else(|e| e.into_inner());
                }
            };
            task(); // never unwinds: `run_hosted` catches the work's panics
        }
    }
}

/// [`run`] on a spawner (the application's, or the pool): tasks borrow
/// `work`, and this does not return (or unwind) before every task has
/// finished with it.
fn run_hosted(n: usize, work: &(dyn Fn() + Sync), spawn: &(dyn Fn(Task) + Send + Sync)) {
    use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
    type Latch = (
        std::sync::Mutex<(usize, Option<Box<dyn std::any::Any + Send>>)>,
        std::sync::Condvar,
    );
    let latch: std::sync::Arc<Latch> = Default::default();
    // SAFETY: the borrow is only used by tasks that start before the caller
    // stops waiting: the caller cancels every task not yet started (below),
    // a cancelled task never touches it, and the wait outlasts the rest —
    // `pending` drops to 0 only after each started one is done with it
    let shared: &'static (dyn Fn() + Sync) = unsafe { std::mem::transmute(work) };
    let mut started = Vec::with_capacity(n - 1);
    for _ in 1..n {
        latch.0.lock().unwrap_or_else(|e| e.into_inner()).0 += 1;
        let latch = latch.clone();
        let claim = std::sync::Arc::new(AtomicBool::new(false));
        started.push(claim.clone());
        spawn(Box::new(move || {
            if claim.swap(true, Ordering::AcqRel) {
                return; // cancelled: the caller did the work, and moved on
            }
            let result = catch_unwind(AssertUnwindSafe(|| as_worker(shared)));
            let mut state = latch.0.lock().unwrap_or_else(|e| e.into_inner());
            state.0 -= 1;
            if let Err(panic) = result {
                state.1.get_or_insert(panic);
            }
            latch.1.notify_all();
        }));
    }
    let mine = catch_unwind(AssertUnwindSafe(|| as_worker(work)));
    // The work is handed out item by item, so once the caller's share
    // returns nothing is left to start: cancel helpers that have not begun.
    // A helper can then never be needed — so a fixed-size pool whose own
    // threads call into this crate cannot deadlock waiting for a free thread.
    let cancelled = started
        .iter()
        .filter(|claim| !claim.swap(true, Ordering::AcqRel))
        .count();
    let mut state = latch.0.lock().unwrap_or_else(|e| e.into_inner());
    state.0 -= cancelled;
    while state.0 > 0 {
        state = latch.1.wait(state).unwrap_or_else(|e| e.into_inner());
    }
    let theirs = state.1.take();
    drop(state);
    if let Err(panic) = mine {
        resume_unwind(panic);
    }
    if let Some(panic) = theirs {
        resume_unwind(panic);
    }
}

/// `items.iter().map(f).collect()`, the calls spread over the workers.
pub(crate) fn map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let n = num_threads().min(items.len());
    if n <= 1 {
        return items.iter().map(f).collect();
    }
    let next = AtomicUsize::new(0);
    let done = crate::lock::Mutex::new(Vec::with_capacity(items.len()));
    run(n, || {
        let mut mine = Vec::new();
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(item) = items.get(i) else { break };
            mine.push((i, f(item)));
        }
        done.lock().extend(mine);
    });
    let mut done = done.into_inner();
    done.sort_unstable_by_key(|(i, _)| *i);
    done.into_iter().map(|(_, r)| r).collect()
}

/// Call `f` on every item, consuming them, spread over the workers.
pub(crate) fn for_each<T: Send>(items: Vec<T>, f: impl Fn(T) + Sync) {
    let n = num_threads().min(items.len());
    if n <= 1 {
        return items.into_iter().for_each(f);
    }
    let queue = crate::lock::Mutex::new(items.into_iter());
    run(n, || loop {
        let Some(item) = queue.lock().next() else {
            break;
        };
        f(item);
    });
}

/// `f(src_part, dst_part)` over matching pieces of `src` and `dst`, cut at
/// whole elements (`s_unit` source bytes ↔ `d_unit` destination bytes); in
/// parallel when there is enough to be worth it.
pub(crate) fn zip_units(
    src: &[u8],
    s_unit: usize,
    dst: &mut [u8],
    d_unit: usize,
    f: impl Fn(&[u8], &mut [u8]) + Sync,
) {
    let elems = (src.len() / s_unit.max(1)).min(dst.len() / d_unit.max(1));
    let n = num_threads();
    if n <= 1 || src.len().max(dst.len()) < MIN_PAR_BYTES || s_unit == 0 || d_unit == 0 {
        return f(src, dst);
    }
    // a few pieces per thread so a slow core does not hold up the rest
    let per = elems.div_ceil(n * 4).max(1);
    let pieces: Vec<_> = src
        .chunks(per * s_unit)
        .zip(dst.chunks_mut(per * d_unit))
        .collect();
    for_each(pieces, |(s, d)| f(s, d));
}

/// `f` over `buf` in place, cut at whole `unit`-byte elements; in parallel
/// when there is enough to be worth it.
pub(crate) fn units_mut(buf: &mut [u8], unit: usize, f: impl Fn(&mut [u8]) + Sync) {
    let n = num_threads();
    if n <= 1 || buf.len() < MIN_PAR_BYTES || unit == 0 {
        return f(buf);
    }
    let per = (buf.len() / unit).div_ceil(n * 4).max(1) * unit;
    for_each(buf.chunks_mut(per).collect(), f);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_host_spawner_runs_the_work() {
        use std::sync::atomic::AtomicUsize;
        static SPAWNED: AtomicUsize = AtomicUsize::new(0);
        set_thread_spawner(|task| {
            SPAWNED.fetch_add(1, Ordering::Relaxed);
            std::thread::spawn(task);
        });
        let items: Vec<u64> = (0..10_000).collect();
        // nested parallel work inside a task stays inline
        let out = map(&items, |&x| map(&[x, x], |&y| y + 1).iter().sum::<u64>());
        clear_thread_spawner();
        assert_eq!(out, items.iter().map(|x| 2 * x + 2).collect::<Vec<_>>());
        if num_threads() > 1 {
            assert!(SPAWNED.load(Ordering::Relaxed) > 0);
        }

        // A fixed pool of two threads whose own jobs call back in: every
        // pool thread waits inside a parallel call while that call's helper
        // tasks sit in the same pool's queue. Unstarted helpers are
        // cancelled, so this finishes instead of deadlocking.
        use std::sync::mpsc;
        if std::thread::Builder::new().spawn(|| {}).is_err() {
            return; // no threads here (plain wasm): nothing to deadlock
        }
        let (tx, rx) = mpsc::channel::<Task>();
        let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));
        for _ in 0..2 {
            let rx = rx.clone();
            std::thread::spawn(move || loop {
                let task = rx.lock().unwrap().recv();
                match task {
                    Ok(task) => task(),
                    Err(_) => break,
                }
            });
        }
        let tx = std::sync::Arc::new(std::sync::Mutex::new(tx));
        let pool = tx.clone();
        set_thread_spawner(move |task| {
            let _ = pool.lock().unwrap().send(task);
        });
        let (done_tx, done_rx) = mpsc::channel();
        for job in 0..2u64 {
            let done_tx = done_tx.clone();
            let items: Vec<u64> = (0..1000).map(|x| x + job).collect();
            tx.lock()
                .unwrap()
                .send(Box::new(move || {
                    let sum: u64 = map(&items, |&x| x * 2).iter().sum();
                    let _ = done_tx.send(sum);
                }))
                .unwrap();
        }
        let mut sums: Vec<u64> = (0..2)
            .map(|_| {
                done_rx
                    .recv_timeout(std::time::Duration::from_secs(20))
                    .expect("deadlocked")
            })
            .collect();
        clear_thread_spawner();
        sums.sort_unstable();
        assert_eq!(sums, [999_000, 1_001_000]);
    }

    #[test]
    fn map_keeps_order() {
        let items: Vec<u32> = (0..1000).collect();
        let out = map(&items, |&x| x * 2);
        assert_eq!(out, items.iter().map(|x| x * 2).collect::<Vec<_>>());
    }

    #[test]
    fn units_cover_everything() {
        let mut buf = vec![1u8; MIN_PAR_BYTES * 2 + 6];
        units_mut(&mut buf, 2, |part| {
            assert_eq!(part.len() % 2, 0);
            part.iter_mut().for_each(|b| *b += 1);
        });
        assert!(buf.iter().all(|&b| b == 2));
    }

    #[test]
    fn zip_matches_serial() {
        let src: Vec<u8> = (0..MIN_PAR_BYTES as u32 + 10).map(|i| i as u8).collect();
        let mut dst = vec![0u8; src.len() * 2];
        zip_units(&src, 1, &mut dst, 2, |s, d| {
            for (a, b) in s.iter().zip(d.chunks_exact_mut(2)) {
                b[0] = *a;
            }
        });
        assert!(src.iter().zip(dst.chunks_exact(2)).all(|(a, b)| b[0] == *a));
    }
}
