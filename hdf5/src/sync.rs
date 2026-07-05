//! Synchronization helpers (API-compatible with the FFI crate).
//!
//! The pure-Rust engine uses fine-grained per-file locks instead of a global
//! library lock, so `sync` simply executes the closure.

/// Executes the closure (formerly: while holding the global library lock).
pub fn sync<T, F>(func: F) -> T
where
    F: FnOnce() -> T,
{
    func()
}

/// A re-entrant mutex type alias kept for compatibility.
pub type ReentrantMutex<T> = parking_lot::ReentrantMutex<T>;

/// Parity stubs for the FFI crate's two-phase global-lock test hooks.
pub fn lock_part1() {}
pub fn lock_part2() {}
