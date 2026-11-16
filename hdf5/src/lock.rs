//! The crate's locks: `parking_lot` natively, `std::sync` on wasm.
//!
//! `parking_lot` blocks a contended thread through its wasm "parker" only
//! when compiled with `cfg(target_feature = "atomics")`, which stable Rust
//! never sets — so on `wasm32-wasip1-threads` a contended `parking_lot` lock
//! panics ("Parking not supported on this platform"). The standard library
//! there is built with atomics and waits with `memory.atomic.wait`, so wasm
//! builds use thin `std::sync` wrappers with `parking_lot`'s (non-poisoning)
//! API instead.

#[cfg(not(target_family = "wasm"))]
pub(crate) use parking_lot::{Mutex, RwLock};

#[cfg(target_family = "wasm")]
pub(crate) use self::std_locks::{Mutex, RwLock};

#[cfg(target_family = "wasm")]
mod std_locks {
    use std::sync::{self, PoisonError};

    /// `std::sync::Mutex` without poisoning: a panic elsewhere does not make
    /// the data unreachable, as with `parking_lot`.
    #[derive(Debug, Default)]
    pub struct Mutex<T: ?Sized>(sync::Mutex<T>);

    impl<T> Mutex<T> {
        pub const fn new(value: T) -> Self {
            Self(sync::Mutex::new(value))
        }
        pub fn into_inner(self) -> T {
            self.0.into_inner().unwrap_or_else(PoisonError::into_inner)
        }
    }

    impl<T: ?Sized> Mutex<T> {
        pub fn lock(&self) -> sync::MutexGuard<'_, T> {
            self.0.lock().unwrap_or_else(PoisonError::into_inner)
        }
    }

    /// `std::sync::RwLock` without poisoning.
    #[derive(Debug, Default)]
    pub struct RwLock<T: ?Sized>(sync::RwLock<T>);

    impl<T> RwLock<T> {
        pub const fn new(value: T) -> Self {
            Self(sync::RwLock::new(value))
        }
    }

    impl<T: ?Sized> RwLock<T> {
        pub fn read(&self) -> sync::RwLockReadGuard<'_, T> {
            self.0.read().unwrap_or_else(PoisonError::into_inner)
        }
        pub fn write(&self) -> sync::RwLockWriteGuard<'_, T> {
            self.0.write().unwrap_or_else(PoisonError::into_inner)
        }
    }
}
