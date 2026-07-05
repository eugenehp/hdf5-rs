//! Chunk introspection helpers.

use crate::error::Result;
use crate::hl::dataset::Dataset;
use crate::model::LayoutClass;

/// Information on a single dataset chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkInfo {
    /// Array with a size equal to the dataset's rank, with the offset of the
    /// chunk in the dataset's logical space.
    pub offset: Vec<u64>,
    /// Filter mask indicating which filters were skipped for this chunk.
    pub filter_mask: u32,
    /// The address of the chunk in the file (unassigned until serialization).
    pub addr: u64,
    /// The size of the chunk in bytes (unfiltered, in-memory estimate).
    pub size: u64,
}

impl Dataset {
    /// Returns information on the chunk at the given index, if chunked.
    pub fn chunk_info(&self, index: usize) -> Option<ChunkInfo> {
        let file = self.0.file()?;
        let id = self.0.obj_id()?;
        let state = file.state.read();
        let d = state.dataset_data(id)?;
        let chunk_dims = match &d.layout {
            LayoutClass::Chunked(c) => c.clone(),
            _ => return None,
        };
        let nchunks: Vec<u64> = d
            .dims
            .iter()
            .zip(&chunk_dims)
            .map(|(&dim, &c)| ((dim).div_ceil(c)).max(1))
            .collect();
        let total: u64 = nchunks.iter().product();
        if (index as u64) >= total {
            return None;
        }
        // decompose row-major chunk index into coordinates
        let mut rem = index as u64;
        let mut strides = vec![1u64; nchunks.len()];
        for i in (0..nchunks.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * nchunks[i + 1];
        }
        let mut offset = Vec::with_capacity(nchunks.len());
        for i in 0..nchunks.len() {
            offset.push((rem / strides[i]) * chunk_dims[i]);
            rem %= strides[i];
        }
        let esize = crate::format::convert::disk_size(&d.dtype) as u64;
        let size = chunk_dims.iter().product::<u64>() * esize;
        Some(ChunkInfo {
            offset,
            filter_mask: 0,
            addr: 0,
            size,
        })
    }

    /// Visits information on all chunks; the callback returns non-negative to
    /// continue iteration.
    pub fn chunks_visit<F>(&self, mut callback: F) -> Result<()>
    where
        F: FnMut(&ChunkInfo) -> i32,
    {
        let n = self.num_chunks().unwrap_or(0);
        for i in 0..n {
            if let Some(info) = self.chunk_info(i) {
                if callback(&info) < 0 {
                    break;
                }
            }
        }
        Ok(())
    }
}

/// Borrowed variant of `ChunkInfo` (FFI-crate parity).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkInfoRef<'a> {
    pub offset: &'a [u64],
    pub filter_mask: u32,
    pub addr: u64,
    pub size: u64,
}

impl ChunkInfoRef<'_> {
    /// Indexes of filters skipped for this chunk (mask bit set).
    pub fn disabled_filters(&self) -> Vec<usize> {
        (0..32)
            .filter(|i| self.filter_mask & (1 << i) != 0)
            .collect()
    }
}
