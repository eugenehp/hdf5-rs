use std::convert::{TryFrom, TryInto};
use std::fmt::{self, Display};
use std::ops::{Deref, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use ndarray::{self, s, Array1, Array2, ArrayView1, ArrayView2};

use crate::error::{ensure, fail, Error, Result};
use crate::hl::extents::Ix;

fn check_coords(coords: &Array2<Ix>, shape: &[Ix]) -> Result<()> {
    if coords.shape() == [0, 0] {
        return Ok(());
    }
    let ndim = coords.shape()[1];
    ensure!(
        ndim == shape.len(),
        "Slice ndim ({}) != shape ndim ({})",
        ndim,
        shape.len()
    );
    for (i, &dim) in shape.iter().enumerate() {
        for &d in coords.slice(s![.., i]) {
            ensure!(
                d < dim,
                "Index {} out of bounds for axis {} with size {}",
                d,
                i,
                dim
            );
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawSlice {
    pub start: Ix,
    pub step: Ix,
    pub count: Option<Ix>,
    pub block: Ix,
}

impl RawSlice {
    pub fn new(start: Ix, step: Ix, count: Option<Ix>, block: Ix) -> Self {
        Self {
            start,
            step,
            count,
            block,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawHyperslab {
    dims: Vec<RawSlice>,
}

impl Deref for RawHyperslab {
    type Target = [RawSlice];

    fn deref(&self) -> &Self::Target {
        &self.dims
    }
}

impl RawHyperslab {
    fn is_none(&self) -> bool {
        self.iter().any(|s| s.count == Some(0))
    }

    fn is_all(&self, shape: &[Ix]) -> bool {
        if self.is_empty() {
            return true;
        }
        for (slice, &dim) in self.iter().zip(shape) {
            let count = match slice.count {
                Some(count) => count,
                None => return false,
            };
            if slice.start != 0 || slice.step != slice.block || count * slice.block != dim {
                return false;
            }
        }
        true
    }
}

impl From<Vec<RawSlice>> for RawHyperslab {
    fn from(dims: Vec<RawSlice>) -> Self {
        Self { dims }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum RawSelection {
    None,
    #[default]
    All,
    Points(Array2<Ix>),
    RegularHyperslab(RawHyperslab),
    ComplexHyperslab,
}

impl From<RawHyperslab> for RawSelection {
    fn from(hyper: RawHyperslab) -> Self {
        Self::RegularHyperslab(hyper)
    }
}

impl From<Vec<RawSlice>> for RawSelection {
    fn from(dims: Vec<RawSlice>) -> Self {
        Self::RegularHyperslab(dims.into())
    }
}

impl RawSelection {
    /// Enumerate the linear (row-major) element indices selected within an
    /// array of the given shape, in selection order.
    pub(crate) fn linear_indices(&self, shape: &[Ix]) -> Result<Vec<usize>> {
        let total: usize = shape.iter().product();
        let strides = row_major_strides(shape);
        match self {
            Self::None => Ok(vec![]),
            Self::All => Ok((0..total).collect()),
            Self::Points(coords) => {
                check_coords(coords, shape)?;
                let mut out = Vec::with_capacity(coords.nrows());
                for row in coords.rows() {
                    let mut lin = 0usize;
                    for (i, &c) in row.iter().enumerate() {
                        lin += c * strides[i];
                    }
                    out.push(lin);
                }
                Ok(out)
            }
            Self::RegularHyperslab(hyper) => {
                ensure!(hyper.len() == shape.len(), "selection rank mismatch");
                // per-dim list of selected coordinates
                let mut axes: Vec<Vec<usize>> = Vec::with_capacity(hyper.len());
                for (slice, &dim) in hyper.iter().zip(shape) {
                    let count = match slice.count {
                        Some(c) => c,
                        None => {
                            // unlimited: fill up to the dim end
                            if slice.block == 0 || slice.step == 0 {
                                0
                            } else if dim > slice.start {
                                (dim - slice.start - slice.block) / slice.step + 1
                            } else {
                                0
                            }
                        }
                    };
                    let mut coords = Vec::with_capacity(count * slice.block);
                    for i in 0..count {
                        let base = slice.start + i * slice.step;
                        for j in 0..slice.block {
                            let c = base + j;
                            ensure!(c < dim, "selection out of bounds: {} >= {} in axis", c, dim);
                            coords.push(c);
                        }
                    }
                    axes.push(coords);
                }
                // cartesian product in row-major order
                let n: usize = axes.iter().map(Vec::len).product();
                let mut out = Vec::with_capacity(n);
                let mut idx = vec![0usize; axes.len()];
                if axes.iter().any(|a| a.is_empty()) {
                    return Ok(vec![]);
                }
                for _ in 0..n {
                    let mut lin = 0usize;
                    for (i, &j) in idx.iter().enumerate() {
                        lin += axes[i][j] * strides[i];
                    }
                    out.push(lin);
                    for k in (0..idx.len()).rev() {
                        idx[k] += 1;
                        if idx[k] < axes[k].len() {
                            break;
                        }
                        idx[k] = 0;
                    }
                }
                Ok(out)
            }
            Self::ComplexHyperslab => fail!("Complex hyperslabs are not supported"),
        }
    }
}

fn row_major_strides(shape: &[Ix]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// A selector of a one-dimensional array
///
/// The following examples will use an array of 11 elements
/// to illustrate the various selections. The active elements
/// are marked with an `s`.
/// ```text
/// // An array of 11 elements
/// x x x x x x x x x x x
/// ```
///
/// ```text
/// Index(4)
/// _ _ _ _ s _ _ _ _ _ _
/// ```
/// ```text
/// Slice { start: 0, step: 3, end: 4, block: 1 }
/// s _ _ s _ _ _ _ _ _ _
/// ```
/// ```text
/// SliceTo { start: 2, step: 3, end: 8, block: 1 }
/// _ _ s _ _ s _ _ _ _ _
/// ```
/// ```text
/// SliceCount { start: 1, step: 3, count: 2, block: 1 }
/// _ s _ _ s _ _ s _ _ _
/// ```
/// ```text
/// Unlimited { start: 0, step: 3, block: 1 }
/// s _ _ s _ _ s _ _ s _
/// ```
/// ```text
/// Unlimited { start: 2, step: 3, block: 1 }
/// _ _ s _ _ s _ _ s _ _
/// ```
/// ```text
/// Unlimited { start: 0, step: 4, block: 2 }
/// s s _ _ s s _ _ s s _
/// ```
///
/// See also [`this hdf5 tutorial`](https://support.hdfgroup.org/HDF5/Tutor/select.html)
/// for more information on hyperslab selections.
#[derive(Clone, Copy, Debug, Eq)]
pub enum SliceOrIndex {
    /// A single index
    Index(Ix),
    /// Up to the given index
    SliceTo {
        start: Ix,
        step: Ix,
        end: Ix,
        block: Ix,
    },
    /// The given count of elements
    SliceCount {
        start: Ix,
        step: Ix,
        count: Ix,
        block: Ix,
    },
    /// An unlimited hyperslab
    Unlimited { start: Ix, step: Ix, block: Ix },
}

impl PartialEq for SliceOrIndex {
    fn eq(&self, other: &Self) -> bool {
        use SliceOrIndex::{Index, SliceCount, SliceTo, Unlimited};
        match (self, other) {
            (Index(s), Index(o)) => s == o,
            (
                SliceTo {
                    start: sstart,
                    step: sstep,
                    end: send,
                    block: sblock,
                },
                SliceTo {
                    start: ostart,
                    step: ostep,
                    end: oend,
                    block: oblock,
                },
            ) => (sstart == ostart) & (sstep == ostep) & (send == oend) & (sblock == oblock),
            (
                SliceCount {
                    start: sstart,
                    step: sstep,
                    count: scount,
                    block: sblock,
                },
                SliceCount {
                    start: ostart,
                    step: ostep,
                    count: ocount,
                    block: oblock,
                },
            ) => (sstart == ostart) & (sstep == ostep) & (scount == ocount) & (sblock == oblock),
            (
                Unlimited {
                    start: sstart,
                    step: sstep,
                    block: sblock,
                },
                Unlimited {
                    start: ostart,
                    step: ostep,
                    block: oblock,
                },
            ) => (sstart == ostart) & (sstep == ostep) & (sblock == oblock),
            (
                SliceTo {
                    start: sstart,
                    step: sstep,
                    end: _,
                    block: sblock,
                },
                SliceCount {
                    start: ostart,
                    step: ostep,
                    count: ocount,
                    block: oblock,
                },
            ) => {
                if (sstart != ostart) | (sstep != ostep) | (sblock != oblock) {
                    return false;
                }
                self.count().unwrap() == *ocount
            }
            (SliceCount { .. }, SliceTo { .. }) => other == self,
            _ => false,
        }
    }
}

impl SliceOrIndex {
    pub fn to_unlimited(self) -> Result<Self> {
        Ok(match self {
            Self::Index(_) => fail!("Cannot make index selection unlimited"),
            Self::SliceTo {
                start, step, block, ..
            }
            | Self::SliceCount {
                start, step, block, ..
            }
            | Self::Unlimited { start, step, block } => Self::Unlimited { start, step, block },
        })
    }

    pub fn is_index(self) -> bool {
        matches!(self, Self::Index(_))
    }

    pub fn is_slice(self) -> bool {
        matches!(
            self,
            Self::SliceTo { .. } | Self::SliceCount { .. } | Self::Unlimited { .. }
        )
    }

    pub fn is_unlimited(self) -> bool {
        matches!(self, Self::Unlimited { .. })
    }

    fn set_blocksize(self, blocksize: Ix) -> Result<Self> {
        Ok(match self {
            Self::Index(_) => fail!("Cannot set blocksize for index selection"),
            Self::SliceTo {
                start, step, end, ..
            } => Self::SliceTo {
                start,
                step,
                end,
                block: blocksize,
            },
            Self::SliceCount {
                start, step, count, ..
            } => Self::SliceCount {
                start,
                step,
                count,
                block: blocksize,
            },
            Self::Unlimited { start, step, .. } => Self::Unlimited {
                start,
                step,
                block: blocksize,
            },
        })
    }

    /// Number of elements contained in the `SliceOrIndex`
    fn count(self) -> Option<usize> {
        use SliceOrIndex::{Index, SliceCount, SliceTo, Unlimited};
        match self {
            Index(_) => Some(1),
            SliceTo {
                start,
                step,
                end,
                block,
            } => Some((start + block.saturating_sub(1)..end).step_by(step).count()),
            SliceCount { count, .. } => Some(count),
            Unlimited { .. } => None,
        }
    }
}

impl TryFrom<ndarray::SliceInfoElem> for SliceOrIndex {
    type Error = Error;
    fn try_from(slice: ndarray::SliceInfoElem) -> Result<Self, Self::Error> {
        Ok(match slice {
            ndarray::SliceInfoElem::Index(index) => match Ix::try_from(index) {
                Err(_) => fail!("Index must be non-negative"),
                Ok(index) => Self::Index(index),
            },
            ndarray::SliceInfoElem::Slice { start, end, step } => {
                let start =
                    Ix::try_from(start).map_err(|_| Error::from("Index must be non-negative"))?;
                let step =
                    Ix::try_from(step).map_err(|_| Error::from("Step must be non-negative"))?;
                let end = end.map(|end| {
                    Ix::try_from(end).map_err(|_| Error::from("End must be non-negative"))
                });
                match end {
                    Some(Ok(end)) => Self::SliceTo {
                        start,
                        step,
                        end,
                        block: 1,
                    },
                    None => Self::Unlimited {
                        start,
                        step,
                        block: 1,
                    },
                    Some(Err(e)) => return Err(e),
                }
            }
            ndarray::SliceInfoElem::NewAxis => fail!("ndarray NewAxis can not be mapped to hdf5"),
        })
    }
}

impl TryFrom<ndarray::SliceInfoElem> for Hyperslab {
    type Error = Error;
    fn try_from(slice: ndarray::SliceInfoElem) -> Result<Self, Self::Error> {
        Ok(vec![slice.try_into()?].into())
    }
}

impl TryFrom<ndarray::SliceInfoElem> for Selection {
    type Error = Error;
    fn try_from(slice: ndarray::SliceInfoElem) -> Result<Self, Self::Error> {
        Hyperslab::try_from(slice).map(Into::into)
    }
}

impl From<RangeFull> for SliceOrIndex {
    fn from(_r: RangeFull) -> Self {
        Self::Unlimited {
            start: 0,
            step: 1,
            block: 1,
        }
    }
}

impl TryFrom<ndarray::Slice> for SliceOrIndex {
    type Error = std::num::TryFromIntError;
    fn try_from(slice: ndarray::Slice) -> Result<Self, Self::Error> {
        let ndarray::Slice { start, end, step } = slice;
        let start = start.try_into()?;
        let step = step.try_into()?;
        let end = end.map(TryInto::try_into);
        match end {
            Some(Ok(end)) => Ok(Self::SliceTo {
                start,
                end,
                step,
                block: 1,
            }),
            None => Ok(Self::Unlimited {
                start,
                step,
                block: 1,
            }),
            Some(Err(e)) => Err(e),
        }
    }
}

impl From<usize> for SliceOrIndex {
    fn from(val: usize) -> Self {
        Self::Index(val as _)
    }
}

impl From<usize> for Hyperslab {
    fn from(slice: usize) -> Self {
        (slice,).into()
    }
}

impl From<usize> for Selection {
    fn from(slice: usize) -> Self {
        Hyperslab::from(slice).into()
    }
}

impl From<Range<usize>> for SliceOrIndex {
    fn from(val: Range<usize>) -> Self {
        Self::SliceTo {
            start: val.start as _,
            step: 1,
            end: val.end,
            block: 1,
        }
    }
}

impl From<Range<usize>> for Hyperslab {
    fn from(val: Range<usize>) -> Self {
        vec![val.into()].into()
    }
}

impl From<Range<usize>> for Selection {
    fn from(val: Range<usize>) -> Self {
        Hyperslab::from(val).into()
    }
}

impl From<RangeToInclusive<usize>> for SliceOrIndex {
    fn from(val: RangeToInclusive<usize>) -> Self {
        let end = val.end + 1;
        Self::SliceTo {
            start: 0,
            step: 1,
            end,
            block: 1,
        }
    }
}

impl From<RangeToInclusive<usize>> for Hyperslab {
    fn from(val: RangeToInclusive<usize>) -> Self {
        vec![val.into()].into()
    }
}

impl From<RangeToInclusive<usize>> for Selection {
    fn from(val: RangeToInclusive<usize>) -> Self {
        Hyperslab::from(val).into()
    }
}

impl From<RangeFrom<usize>> for SliceOrIndex {
    fn from(val: RangeFrom<usize>) -> Self {
        Self::Unlimited {
            start: val.start,
            step: 1,
            block: 1,
        }
    }
}

impl From<RangeFrom<usize>> for Hyperslab {
    fn from(val: RangeFrom<usize>) -> Self {
        vec![val.into()].into()
    }
}

impl From<RangeFrom<usize>> for Selection {
    fn from(val: RangeFrom<usize>) -> Self {
        Hyperslab::from(val).into()
    }
}

impl From<RangeInclusive<usize>> for SliceOrIndex {
    fn from(val: RangeInclusive<usize>) -> Self {
        Self::SliceTo {
            start: *val.start(),
            step: 1,
            end: *val.end() + 1,
            block: 1,
        }
    }
}

impl From<RangeInclusive<usize>> for Hyperslab {
    fn from(val: RangeInclusive<usize>) -> Self {
        vec![val.into()].into()
    }
}

impl From<RangeInclusive<usize>> for Selection {
    fn from(val: RangeInclusive<usize>) -> Self {
        Hyperslab::from(val).into()
    }
}

impl From<RangeTo<usize>> for SliceOrIndex {
    fn from(val: RangeTo<usize>) -> Self {
        Self::SliceTo {
            start: 0,
            step: 1,
            end: val.end,
            block: 1,
        }
    }
}

impl From<RangeTo<usize>> for Hyperslab {
    fn from(val: RangeTo<usize>) -> Self {
        vec![val.into()].into()
    }
}

impl From<RangeTo<usize>> for Selection {
    fn from(val: RangeTo<usize>) -> Self {
        Hyperslab::from(val).into()
    }
}

impl Display for SliceOrIndex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Self::Index(index) => write!(f, "{index}")?,
            Self::SliceTo {
                start,
                end,
                step,
                block,
            } => {
                if start != 0 {
                    write!(f, "{start}")?;
                }
                write!(f, "..")?;
                write!(f, "{end}")?;
                if step != 1 {
                    write!(f, ";{step}")?;
                }
                if block != 1 {
                    write!(f, "(Bx{block})")?;
                }
            }
            Self::SliceCount {
                start,
                step,
                count,
                block,
            } => {
                if start != 0 {
                    write!(f, "{start}")?;
                }
                write!(f, "+{count}")?;
                if step != 1 {
                    write!(f, ";{step}")?;
                }
                if block != 1 {
                    write!(f, "(Bx{block})")?;
                }
            }
            Self::Unlimited { start, step, block } => {
                if start != 0 {
                    write!(f, "{start}")?;
                }
                // \u{221e} = ∞
                write!(f, "..\u{221e}")?;
                if step != 1 {
                    write!(f, ";{step}")?;
                }
                if block != 1 {
                    write!(f, "(Bx{block})")?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// A descriptor of a selection of an N-dimensional array.
///
/// The Hyperslab consists of [`slices`](SliceOrIndex) in N dimensions,
/// spanning an N-dimensional hypercube. This type is used as a [`selector`](Selection)
/// for retrieving and putting data to a [`Container`](crate::Container).
pub struct Hyperslab {
    dims: Vec<SliceOrIndex>,
}

impl Hyperslab {
    pub fn new<T: Into<Self>>(hyper: T) -> Self {
        hyper.into()
    }

    pub fn try_new<T: TryInto<Self>>(hyper: T) -> Result<Self, T::Error> {
        hyper.try_into()
    }

    pub fn is_unlimited(&self) -> bool {
        self.iter().any(|&s| s.is_unlimited())
    }

    pub fn unlimited_axis(&self) -> Option<usize> {
        self.iter()
            .enumerate()
            .find_map(|(i, s)| if s.is_unlimited() { Some(i) } else { None })
    }

    pub fn set_unlimited(&self, axis: usize) -> Result<Self> {
        if axis < self.len() {
            let mut hyper = self.clone();
            hyper.dims[axis] = hyper.dims[axis].to_unlimited()?;
            Ok(hyper)
        } else {
            fail!("Invalid axis for making hyperslab unlimited: {}", axis);
        }
    }

    pub fn set_block(&self, axis: usize, blocksize: Ix) -> Result<Self> {
        ensure!(
            axis < self.len(),
            "Invalid axis for changing the slice to block-like: {}",
            axis
        );
        let mut hyper = self.clone();
        hyper.dims[axis] = hyper.dims[axis].set_blocksize(blocksize)?;
        Ok(hyper)
    }

    #[doc(hidden)]
    pub fn into_raw<S: AsRef<[Ix]>>(self, shape: S) -> Result<RawHyperslab> {
        let shape = shape.as_ref();
        let ndim = shape.len();
        ensure!(
            self.len() == ndim,
            "Slice ndim ({}) != shape ndim ({})",
            self.len(),
            ndim
        );
        //let n_unlimited: usize = self.iter().map(|s| s.is_unlimited() as usize).sum();
        //ensure!(n_unlimited <= 1, "Expected at most 1 unlimited dimension, got {}", n_unlimited);
        let hyper = RawHyperslab::from(
            self.iter()
                .zip(shape)
                .enumerate()
                .map(|(i, (slice, &dim))| slice_info_to_raw(i, slice, dim))
                .collect::<Result<Vec<_>>>()?,
        );
        Ok(hyper)
    }

    #[doc(hidden)]
    #[allow(clippy::needless_pass_by_value)]
    pub fn from_raw(hyper: RawHyperslab) -> Result<Self> {
        let mut dims = vec![];
        for (axis, slice) in hyper.iter().enumerate() {
            ensure!(
                slice.step >= slice.block,
                "Blocks can not overlap (axis: {})",
                axis
            );
            dims.push(match slice.count {
                Some(count) => SliceOrIndex::SliceCount {
                    start: slice.start,
                    step: slice.step,
                    count,
                    block: slice.block,
                },
                None => SliceOrIndex::Unlimited {
                    start: slice.start,
                    step: slice.step,
                    block: slice.block,
                },
            });
        }
        Ok(Self { dims })
    }
}

impl Deref for Hyperslab {
    type Target = [SliceOrIndex];

    fn deref(&self) -> &Self::Target {
        &self.dims
    }
}

impl From<Vec<SliceOrIndex>> for Hyperslab {
    fn from(dims: Vec<SliceOrIndex>) -> Self {
        Self { dims }
    }
}

impl From<()> for Hyperslab {
    fn from((): ()) -> Self {
        vec![].into()
    }
}

impl From<RangeFull> for Hyperslab {
    fn from(_: RangeFull) -> Self {
        (0..).into()
    }
}

impl From<SliceOrIndex> for Hyperslab {
    fn from(slice: SliceOrIndex) -> Self {
        vec![slice].into()
    }
}

impl TryFrom<ndarray::Slice> for Hyperslab {
    type Error = Error;
    fn try_from(slice: ndarray::Slice) -> Result<Self, Self::Error> {
        Ok(vec![SliceOrIndex::try_from(slice).map_err(|_| Error::from("Invalid slice"))?].into())
    }
}

impl<T, Din, Dout> TryFrom<ndarray::SliceInfo<T, Din, Dout>> for Hyperslab
where
    T: AsRef<[ndarray::SliceInfoElem]>,
    Din: ndarray::Dimension,
    Dout: ndarray::Dimension,
{
    type Error = Error;
    fn try_from(slice: ndarray::SliceInfo<T, Din, Dout>) -> Result<Self, Self::Error> {
        slice
            .deref()
            .as_ref()
            .iter()
            .copied()
            .map(TryInto::try_into)
            .collect::<Result<Vec<_>>>()
            .map(Into::into)
    }
}

/// Turns `SliceOrIndex` into real dimensions given `dim` as the maximum dimension
fn slice_info_to_raw(axis: usize, slice: &SliceOrIndex, dim: Ix) -> Result<RawSlice> {
    let err_msg = || format!("out of bounds for axis {axis} with size {dim}");
    let (start, step, count, block) = match *slice {
        SliceOrIndex::Index(index) => {
            ensure!(index < dim, "Index {} {}", index, err_msg());
            (index, 1, 1, 1)
        }
        SliceOrIndex::SliceTo {
            start,
            step,
            end,
            block,
        } => {
            ensure!(step >= 1, "Slice stride {} < 1 for axis {}", step, axis);
            ensure!(start <= dim, "Slice start {} {}", start, err_msg());
            ensure!(end <= dim, "Slice end {} {}", end, err_msg());
            ensure!(step > 0, "Stride {} {}", step, err_msg());
            let count = slice.count().unwrap();
            (start, step, count, block)
        }
        SliceOrIndex::SliceCount {
            start,
            step,
            count,
            block,
        } => {
            ensure!(step >= 1, "Slice stride {} < 1 for axis {}", step, axis);
            ensure!(start <= dim as _, "Slice start {} {}", start, err_msg());
            let end = start + block.saturating_sub(1) + step * count.saturating_sub(1);
            ensure!(end <= dim, "Slice end {} {}", end, err_msg());
            (start, step, count, block)
        }
        SliceOrIndex::Unlimited { start, step, block } => {
            // Replace infinite slice with one limited by the current dimension
            return slice_info_to_raw(
                axis,
                &SliceOrIndex::SliceTo {
                    start,
                    step,
                    end: dim,
                    block,
                },
                dim,
            );
        }
    };
    Ok(RawSlice {
        start,
        step,
        count: Some(count),
        block,
    })
}

impl Display for Hyperslab {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let slice: &[_] = self.as_ref();
        write!(f, "(")?;
        for (i, s) in slice.iter().enumerate() {
            if i != 0 {
                write!(f, ", ")?;
            }
            write!(f, "{s}")?;
        }
        if slice.len() == 1 {
            write!(f, ",")?;
        }
        write!(f, ")")
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// A selection used for reading and writing to a [`Container`](crate::Container).
#[derive(Default)]
pub enum Selection {
    #[default]
    All,
    Points(Array2<Ix>),
    Hyperslab(Hyperslab),
}

impl Selection {
    pub fn new<T: Into<Self>>(selection: T) -> Self {
        selection.into()
    }

    pub fn try_new<T: TryInto<Self>>(selection: T) -> Result<Self, T::Error> {
        selection.try_into()
    }

    #[doc(hidden)]
    pub fn into_raw<S: AsRef<[Ix]>>(self, shape: S) -> Result<RawSelection> {
        let shape = shape.as_ref();
        Ok(match self {
            Self::All => RawSelection::All,
            Self::Points(coords) => {
                check_coords(&coords, shape)?;
                if coords.shape()[0] == 0 {
                    RawSelection::None
                } else {
                    RawSelection::Points(coords)
                }
            }
            Self::Hyperslab(hyper) => {
                let hyper = hyper.into_raw(shape)?;
                if hyper.is_none() {
                    RawSelection::None
                } else if hyper.is_all(shape) {
                    RawSelection::All
                } else {
                    RawSelection::RegularHyperslab(hyper)
                }
            }
        })
    }

    #[doc(hidden)]
    pub fn from_raw(selection: RawSelection) -> Result<Self> {
        Ok(match selection {
            RawSelection::None => Self::Points(Array2::default((0, 0))),
            RawSelection::All => Self::All,
            RawSelection::Points(coords) => Self::Points(coords),
            RawSelection::RegularHyperslab(hyper) => Hyperslab::from_raw(hyper)?.into(),
            RawSelection::ComplexHyperslab => fail!("Cannot convert complex hyperslabs"),
        })
    }

    pub fn in_ndim(&self) -> Option<usize> {
        match self {
            Self::All => None,
            Self::Points(ref points) => {
                if points.shape() == [0, 0] {
                    None
                } else {
                    Some(points.shape()[1])
                }
            }
            Self::Hyperslab(ref hyper) => Some(hyper.len()),
        }
    }

    pub fn out_ndim(&self) -> Option<usize> {
        match self {
            Self::All => None,
            Self::Points(ref points) => Some(usize::from(points.shape() != [0, 0])),
            Self::Hyperslab(ref hyper) => {
                Some(hyper.iter().map(|&s| usize::from(s.is_slice())).sum())
            }
        }
    }

    pub fn out_shape<S: AsRef<[Ix]>>(&self, in_shape: S) -> Result<Vec<Ix>> {
        let in_shape = in_shape.as_ref();
        match self {
            Self::All => Ok(in_shape.to_owned()),
            Self::Points(ref points) => {
                check_coords(points, in_shape).and(Ok(if points.shape() == [0, 0] {
                    vec![]
                } else {
                    vec![points.shape()[0]]
                }))
            }
            Self::Hyperslab(ref hyper) => hyper
                .clone()
                .into_raw(in_shape)?
                .iter()
                .zip(hyper.iter())
                .filter_map(|(&r, &s)| match (r.count, s.is_index()) {
                    (Some(_), true) => None,
                    (Some(count), false) => Some(Ok(count * r.block)),
                    (None, _) => {
                        Some(Err("Unable to get the shape for unlimited hyperslab".into()))
                    }
                })
                .collect(),
        }
    }

    pub fn is_all(&self) -> bool {
        self == &Self::All
    }

    pub fn is_points(&self) -> bool {
        if let Self::Points(ref points) = self {
            points.shape() != [0, 0]
        } else {
            false
        }
    }

    pub fn is_none(&self) -> bool {
        if let Self::Points(points) = self {
            points.shape() == [0, 0]
        } else {
            false
        }
    }

    pub fn is_hyperslab(&self) -> bool {
        matches!(self, Self::Hyperslab(_))
    }
}

impl Display for Selection {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::All => write!(f, ".."),
            Self::Points(ref points) => {
                if points.shape() == [0, 0] {
                    write!(f, "[]")
                } else {
                    write!(f, "{points}")
                }
            }
            Self::Hyperslab(hyper) => write!(f, "{hyper}"),
        }
    }
}

impl From<&Self> for Selection {
    fn from(sel: &Self) -> Self {
        sel.clone()
    }
}

impl From<RangeFull> for Selection {
    fn from(_: RangeFull) -> Self {
        Self::All
    }
}

impl From<()> for Selection {
    fn from((): ()) -> Self {
        Hyperslab::from(()).into()
    }
}

impl From<SliceOrIndex> for Selection {
    fn from(slice: SliceOrIndex) -> Self {
        Self::Hyperslab(slice.into())
    }
}

impl From<Hyperslab> for Selection {
    fn from(hyper: Hyperslab) -> Self {
        Self::Hyperslab(hyper)
    }
}

impl TryFrom<ndarray::Slice> for Selection {
    type Error = Error;
    fn try_from(slice: ndarray::Slice) -> Result<Self, Self::Error> {
        Hyperslab::try_from(slice).map(Into::into)
    }
}

impl<T, Din, Dout> TryFrom<ndarray::SliceInfo<T, Din, Dout>> for Selection
where
    T: AsRef<[ndarray::SliceInfoElem]>,
    Din: ndarray::Dimension,
    Dout: ndarray::Dimension,
{
    type Error = Error;
    fn try_from(slice: ndarray::SliceInfo<T, Din, Dout>) -> Result<Self, Self::Error> {
        Hyperslab::try_from(slice).map(Into::into)
    }
}

impl From<Array2<Ix>> for Selection {
    fn from(points: Array2<Ix>) -> Self {
        Self::Points(points)
    }
}

impl From<Array1<Ix>> for Selection {
    fn from(points: Array1<Ix>) -> Self {
        let n = points.len();
        Self::Points(if n == 0 {
            Array2::zeros((0, 0))
        } else {
            points.insert_axis(ndarray::Axis(1))
        })
    }
}

impl From<ArrayView2<'_, Ix>> for Selection {
    fn from(points: ArrayView2<'_, Ix>) -> Self {
        points.to_owned().into()
    }
}

impl From<ArrayView1<'_, Ix>> for Selection {
    fn from(points: ArrayView1<'_, Ix>) -> Self {
        points.to_owned().into()
    }
}

impl From<&Array2<Ix>> for Selection {
    fn from(points: &Array2<Ix>) -> Self {
        points.clone().into()
    }
}

impl From<&Array1<Ix>> for Selection {
    fn from(points: &Array1<Ix>) -> Self {
        points.clone().into()
    }
}

impl From<Vec<Ix>> for Selection {
    fn from(points: Vec<Ix>) -> Self {
        Array1::from(points).into()
    }
}

impl From<&[Ix]> for Selection {
    fn from(points: &[Ix]) -> Self {
        ArrayView1::from(points).into()
    }
}

impl<const N: usize> From<[Ix; N]> for Selection {
    fn from(points: [Ix; N]) -> Self {
        points.as_ref().into()
    }
}

impl<const N: usize> From<&[Ix; N]> for Selection {
    fn from(points: &[Ix; N]) -> Self {
        points.as_ref().into()
    }
}

macro_rules! impl_tuple {
    () => ();

    ($head:ident, $($tail:ident,)*) => (
        #[allow(non_snake_case)]
        impl<$head, $($tail,)*> From<($head, $($tail,)*)> for Hyperslab
            where $head: Into<SliceOrIndex>, $($tail: Into<SliceOrIndex>,)*
        {
            fn from(slice: ($head, $($tail,)*)) -> Self {
                let ($head, $($tail,)*) = slice;
                vec![($head).into(), $(($tail).into(),)*].into()
            }
        }

        #[allow(non_snake_case)]
        impl<$head, $($tail,)*> From<($head, $($tail,)*)> for Selection
            where $head: Into<SliceOrIndex>, $($tail: Into<SliceOrIndex>,)*
        {
            fn from(slice: ($head, $($tail,)*)) -> Self {
                Hyperslab::from(slice).into()
            }
        }

        impl_tuple! { $($tail,)* }
    )
}

impl_tuple! { T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, }
