//! Property types shared across several property list classes.

use bitflags::bitflags;

/// Attribute storage phase-change thresholds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttrPhaseChange {
    pub max_compact: u32,
    pub min_dense: u32,
}

impl Default for AttrPhaseChange {
    fn default() -> Self {
        Self {
            max_compact: 8,
            min_dense: 6,
        }
    }
}

bitflags! {
    /// Attribute creation-order tracking flags.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
    pub struct AttrCreationOrder: u32 {
        const TRACKED = 0x1;
        const INDEXED = 0x2;
    }
}
