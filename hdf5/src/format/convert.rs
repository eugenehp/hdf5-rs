//! Layout conversion between user memory and the file/model representation.
//!
//! Element layouts differ between memory and disk for variable-length types:
//! in memory a vlen string is a pointer (8 bytes) and a vlen sequence is an
//! `hvl_t` (16 bytes); on disk both are 16-byte global-heap references
//! `{len: u32, collection_addr: u64, index: u32}`.
//!
//! The in-memory *model* stores dataset/attribute bytes in **disk layout**,
//! with heap references replaced by indices into a per-object side store:
//! `{len: u32, store_idx: u32, reserved: u64}` (index is 1-based, 0 = empty).
//! Conversion to and from real pointers happens only at the API boundary, so
//! the model never owns raw pointers and stays safely cloneable.
//!
//! Compound conversion matches source and destination fields **by name**, so a
//! packed on-disk compound can be read into a `#[repr(C)]` Rust struct with
//! different offsets, and vice versa.

use hdf5_types::{CompoundType, TypeDescriptor};

use crate::error::Result;

/// Width of a pointer / `usize` in this target's memory layout (8 on 64-bit,
/// 4 on wasm32): `char*` strings and `hvl_t { len: usize, ptr }` are laid out
/// with it, so it must never be hard-coded.
const PTR: usize = std::mem::size_of::<usize>();

/// A `usize`/pointer read from memory-layout bytes at `off`.
fn read_word(s: &[u8], off: usize) -> usize {
    usize::from_ne_bytes(s[off..off + PTR].try_into().unwrap())
}

/// Size of one element of `desc` in the on-disk/model layout.
pub fn disk_size(desc: &TypeDescriptor) -> usize {
    match desc {
        TypeDescriptor::VarLenAscii | TypeDescriptor::VarLenUnicode => 16,
        TypeDescriptor::VarLenArray(_) => 16,
        TypeDescriptor::Compound(c) => c.size, // stored descriptors carry disk offsets
        TypeDescriptor::FixedArray(base, n) => disk_size(base) * n,
        other => other.size(),
    }
}

/// Does this type contain any variable-length component?
pub fn has_vlen(desc: &TypeDescriptor) -> bool {
    match desc {
        TypeDescriptor::VarLenAscii
        | TypeDescriptor::VarLenUnicode
        | TypeDescriptor::VarLenArray(_) => true,
        TypeDescriptor::Compound(c) => c.fields.iter().any(|f| has_vlen(&f.ty)),
        TypeDescriptor::FixedArray(base, _) => has_vlen(base),
        _ => false,
    }
}

/// Transform a memory-layout descriptor into the disk-layout descriptor stored
/// in the model and encoded into datatype messages.
///
/// Compounds that contain vlen members get their field offsets recomputed by
/// packing disk sizes in index order (memory offsets are meaningless on disk
/// once member sizes change); other compounds keep their given offsets, which
/// is what libhdf5 does when handed a memory type at creation time.
pub fn to_disk_repr(desc: &TypeDescriptor) -> TypeDescriptor {
    match desc {
        TypeDescriptor::Compound(c) => {
            let mut fields: Vec<_> = c.fields.clone();
            fields.sort_by_key(|f| f.index);
            for f in &mut fields {
                f.ty = to_disk_repr(&f.ty);
            }
            if c.fields.iter().any(|f| has_vlen(&f.ty)) {
                let mut offset = 0usize;
                for f in &mut fields {
                    f.offset = offset;
                    offset += disk_size(&f.ty);
                }
                TypeDescriptor::Compound(CompoundType {
                    fields,
                    size: offset,
                })
            } else {
                TypeDescriptor::Compound(CompoundType {
                    fields,
                    size: c.size,
                })
            }
        }
        TypeDescriptor::FixedArray(base, n) => {
            TypeDescriptor::FixedArray(Box::new(to_disk_repr(base)), *n)
        }
        TypeDescriptor::VarLenArray(base) => {
            TypeDescriptor::VarLenArray(Box::new(to_disk_repr(base)))
        }
        other => other.clone(),
    }
}

/// A vlen side store: buffers referenced by 1-based index from model slots.
pub type VlenStore = Vec<Vec<u8>>;

fn model_slot(len: u32, idx: u32) -> [u8; 16] {
    let mut s = [0u8; 16];
    s[..4].copy_from_slice(&len.to_le_bytes());
    s[4..8].copy_from_slice(&idx.to_le_bytes());
    s
}

pub fn slot_parts(slot: &[u8]) -> (u32, u32) {
    let len = u32::from_le_bytes(slot[0..4].try_into().unwrap());
    let idx = u32::from_le_bytes(slot[4..8].try_into().unwrap());
    (len, idx)
}

/// A numeric value in transit between differently-sized representations.
enum Numeric {
    Signed(i64),
    Unsigned(u64),
    Float(f64),
}

fn read_numeric(desc: &TypeDescriptor, s: &[u8]) -> Option<Numeric> {
    use hdf5_types::{FloatSize, IntSize};
    use TypeDescriptor::*;
    Some(match desc {
        Integer(IntSize::U1) => Numeric::Signed(i8::from_le_bytes([s[0]]) as i64),
        Integer(IntSize::U2) => Numeric::Signed(i16::from_le_bytes([s[0], s[1]]) as i64),
        Integer(IntSize::U4) => Numeric::Signed(i32::from_le_bytes(s[..4].try_into().ok()?) as i64),
        Integer(IntSize::U8) => Numeric::Signed(i64::from_le_bytes(s[..8].try_into().ok()?)),
        Unsigned(IntSize::U1) => Numeric::Unsigned(s[0] as u64),
        Unsigned(IntSize::U2) => Numeric::Unsigned(u16::from_le_bytes([s[0], s[1]]) as u64),
        Unsigned(IntSize::U4) => {
            Numeric::Unsigned(u32::from_le_bytes(s[..4].try_into().ok()?) as u64)
        }
        Unsigned(IntSize::U8) => Numeric::Unsigned(u64::from_le_bytes(s[..8].try_into().ok()?)),
        Float(FloatSize::U4) => Numeric::Float(f32::from_le_bytes(s[..4].try_into().ok()?) as f64),
        Float(FloatSize::U8) => Numeric::Float(f64::from_le_bytes(s[..8].try_into().ok()?)),
        #[cfg(feature = "f16")]
        Float(FloatSize::U2) => Numeric::Float(half::f16::from_le_bytes([s[0], s[1]]).to_f64()),
        _ => return None,
    })
}

fn write_numeric(desc: &TypeDescriptor, v: &Numeric, d: &mut [u8]) -> bool {
    use hdf5_types::{FloatSize, IntSize};
    use TypeDescriptor::*;
    // Saturating conversions, matching libhdf5's default conversion-exception
    // behavior (H5T_CONV_* macros): integer overflow clamps to the destination
    // min/max, negative-to-unsigned clamps to 0. Float sources use Rust `as`,
    // which already matches libhdf5 (saturates; NaN -> 0; +/-Inf -> MAX/MIN).
    #[allow(clippy::cast_lossless)]
    fn clamp_i(v: &Numeric, min: i64, max: i64) -> i64 {
        match v {
            Numeric::Signed(x) => (*x).clamp(min, max),
            Numeric::Unsigned(x) => {
                if *x > max as u64 {
                    max
                } else {
                    *x as i64
                }
            }
            Numeric::Float(x) => {
                // `as` saturates and maps NaN to 0 -- identical to libhdf5
                (*x as i64).clamp(min, max)
            }
        }
    }
    fn clamp_u(v: &Numeric, max: u64) -> u64 {
        match v {
            Numeric::Signed(x) => {
                if *x < 0 {
                    0
                } else {
                    (*x as u64).min(max)
                }
            }
            Numeric::Unsigned(x) => (*x).min(max),
            Numeric::Float(x) => (*x as u64).min(max),
        }
    }
    let f = match v {
        Numeric::Signed(x) => *x as f64,
        Numeric::Unsigned(x) => *x as f64,
        Numeric::Float(x) => *x,
    };
    match desc {
        Integer(IntSize::U1) => {
            d[..1]
                .copy_from_slice(&(clamp_i(v, i8::MIN as i64, i8::MAX as i64) as i8).to_le_bytes());
        }
        Integer(IntSize::U2) => {
            d[..2].copy_from_slice(
                &(clamp_i(v, i16::MIN as i64, i16::MAX as i64) as i16).to_le_bytes(),
            );
        }
        Integer(IntSize::U4) => {
            d[..4].copy_from_slice(
                &(clamp_i(v, i32::MIN as i64, i32::MAX as i64) as i32).to_le_bytes(),
            );
        }
        Integer(IntSize::U8) => {
            d[..8].copy_from_slice(&clamp_i(v, i64::MIN, i64::MAX).to_le_bytes());
        }
        Unsigned(IntSize::U1) => {
            d[..1].copy_from_slice(&(clamp_u(v, u8::MAX as u64) as u8).to_le_bytes());
        }
        Unsigned(IntSize::U2) => {
            d[..2].copy_from_slice(&(clamp_u(v, u16::MAX as u64) as u16).to_le_bytes());
        }
        Unsigned(IntSize::U4) => {
            d[..4].copy_from_slice(&(clamp_u(v, u32::MAX as u64) as u32).to_le_bytes());
        }
        Unsigned(IntSize::U8) => {
            d[..8].copy_from_slice(&clamp_u(v, u64::MAX).to_le_bytes());
        }
        Float(FloatSize::U4) => d[..4].copy_from_slice(&(f as f32).to_le_bytes()),
        Float(FloatSize::U8) => d[..8].copy_from_slice(&f.to_le_bytes()),
        #[cfg(feature = "f16")]
        Float(FloatSize::U2) => {
            d[..2].copy_from_slice(&half::f16::from_f64(f).to_le_bytes());
        }
        _ => return false,
    }
    true
}

/// Convert a scalar numeric element between two layouts if both sides are
/// numeric and they differ; returns `true` if handled.
fn convert_numeric(src: &TypeDescriptor, dst: &TypeDescriptor, s: &[u8], d: &mut [u8]) -> bool {
    if src == dst {
        return false;
    }
    match read_numeric(src, s) {
        Some(v) => write_numeric(dst, &v, d),
        None => false,
    }
}

/// Check two descriptors describe convertible data (same logical structure).
fn check_compatible(src: &TypeDescriptor, dst: &TypeDescriptor) -> Result<()> {
    use TypeDescriptor::*;
    let ok = match (src, dst) {
        // numeric conversions (width/signedness/int-float) are supported,
        // mirroring libhdf5's hard conversion paths
        (Integer(_) | Unsigned(_) | Float(_), Integer(_) | Unsigned(_) | Float(_)) => true,
        (Boolean, Boolean) => true,
        (Enum(a), Enum(b)) => a.size == b.size && a.signed == b.signed,
        // Booleans are stored as an enum on disk; allow both directions.
        (Enum(e), Boolean) | (Boolean, Enum(e)) => e.size as usize == 1,
        (FixedAscii(_), FixedAscii(_)) | (FixedUnicode(_), FixedUnicode(_)) => true,
        (FixedAscii(_), FixedUnicode(_)) | (FixedUnicode(_), FixedAscii(_)) => true,
        (VarLenAscii | VarLenUnicode, VarLenAscii | VarLenUnicode) => true,
        (FixedAscii(_) | FixedUnicode(_), VarLenAscii | VarLenUnicode) => true,
        (VarLenAscii | VarLenUnicode, FixedAscii(_) | FixedUnicode(_)) => true,
        (VarLenArray(a), VarLenArray(b)) => return check_compatible(a, b),
        (FixedArray(a, n), FixedArray(b, m)) => {
            if n != m {
                return Err(format!("array length mismatch: {n} vs {m}").into());
            }
            return check_compatible(a, b);
        }
        (Compound(a), Compound(b)) => {
            for bf in &b.fields {
                let af = a
                    .fields
                    .iter()
                    .find(|f| f.name == bf.name)
                    .ok_or_else(|| format!("missing compound field '{}'", bf.name))?;
                check_compatible(&af.ty, &bf.ty)?;
            }
            true
        }
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(format!("incompatible datatypes: {src} vs {dst}").into())
    }
}

/// A plain number type, as stored: little-endian in the model, native in
/// user memory (the same on every little-endian target).
trait Prim: Copy {
    const N: usize;
    fn get(b: &[u8]) -> Self;
    fn put(self, b: &mut [u8]);
}

/// The conversion libhdf5's hard paths (`H5T_conv_*`) perform: integers
/// saturate to the destination range (negative → 0 for unsigned), floats
/// go through `as` — which already saturates, maps NaN to 0 and ±Inf to
/// MAX/MIN, like the C library.
trait Sat<D> {
    fn sat(self) -> D;
}

macro_rules! prim {
    ($($t:ty),*) => {$(
        impl Prim for $t {
            const N: usize = std::mem::size_of::<$t>();
            #[inline(always)]
            fn get(b: &[u8]) -> Self {
                <$t>::from_le_bytes(b.try_into().unwrap())
            }
            #[inline(always)]
            fn put(self, b: &mut [u8]) {
                b.copy_from_slice(&self.to_le_bytes())
            }
        }
    )*};
}
prim!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

macro_rules! sat_int {
    ($s:ty => $($d:ty),*) => {$(
        impl Sat<$d> for $s {
            #[inline(always)]
            #[allow(unused_comparisons)]
            fn sat(self) -> $d {
                match <$d>::try_from(self) {
                    Ok(v) => v,
                    Err(_) if self < 0 => <$d>::MIN,
                    Err(_) => <$d>::MAX,
                }
            }
        }
    )*};
}
macro_rules! sat_as {
    ($s:ty => $($d:ty),*) => {$(
        impl Sat<$d> for $s {
            #[inline(always)]
            fn sat(self) -> $d {
                self as $d
            }
        }
    )*};
}
macro_rules! sat_all {
    ($($s:ty),*) => {$(
        sat_int!($s => i8, i16, i32, i64, u8, u16, u32, u64);
        sat_as!($s => f32, f64);
    )*};
}
sat_all!(i8, i16, i32, i64, u8, u16, u32, u64);
sat_as!(f32 => i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
sat_as!(f64 => i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

/// Converts to every plain number type.
trait ToAll:
    Prim
    + Sat<i8>
    + Sat<i16>
    + Sat<i32>
    + Sat<i64>
    + Sat<u8>
    + Sat<u16>
    + Sat<u32>
    + Sat<u64>
    + Sat<f32>
    + Sat<f64>
{
}
impl<T> ToAll for T where
    T: Prim
        + Sat<i8>
        + Sat<i16>
        + Sat<i32>
        + Sat<i64>
        + Sat<u8>
        + Sat<u16>
        + Sat<u32>
        + Sat<u64>
        + Sat<f32>
        + Sat<f64>
{
}

/// One tight, vectorizable loop per type pair — split over threads for big
/// buffers.
fn cast<S: Sat<D> + Prim, D: Prim>(data: &[u8], out: &mut [u8]) {
    crate::par::zip_units(data, S::N, out, D::N, |s, d| {
        for (s, d) in s.chunks_exact(S::N).zip(d.chunks_exact_mut(D::N)) {
            S::get(s).sat().put(d);
        }
    });
}

fn cast_from<S: ToAll>(dst: &TypeDescriptor, data: &[u8], out: &mut [u8]) -> bool {
    use hdf5_types::{FloatSize, IntSize};
    use TypeDescriptor::*;
    match dst {
        Integer(IntSize::U1) => cast::<S, i8>(data, out),
        Integer(IntSize::U2) => cast::<S, i16>(data, out),
        Integer(IntSize::U4) => cast::<S, i32>(data, out),
        Integer(IntSize::U8) => cast::<S, i64>(data, out),
        Unsigned(IntSize::U1) => cast::<S, u8>(data, out),
        Unsigned(IntSize::U2) => cast::<S, u16>(data, out),
        Unsigned(IntSize::U4) => cast::<S, u32>(data, out),
        Unsigned(IntSize::U8) => cast::<S, u64>(data, out),
        Float(FloatSize::U4) => cast::<S, f32>(data, out),
        Float(FloatSize::U8) => cast::<S, f64>(data, out),
        _ => return false,
    }
    true
}

/// Whether [`convert_numbers`] handles this pair.
pub(crate) fn numbers_convertible(src: &TypeDescriptor, dst: &TypeDescriptor) -> bool {
    use hdf5_types::FloatSize;
    use TypeDescriptor::*;
    let plain = |t: &TypeDescriptor| {
        matches!(
            t,
            Integer(_) | Unsigned(_) | Float(FloatSize::U4 | FloatSize::U8)
        )
    };
    plain(src) && plain(dst)
}

/// Convert plain numbers (`data`: `src` elements) into `out` (`dst`
/// elements, sized by the caller) in one pass. `false` if either side is not
/// a plain integer/float, leaving `out` untouched.
pub(crate) fn convert_numbers(
    src: &TypeDescriptor,
    dst: &TypeDescriptor,
    data: &[u8],
    out: &mut [u8],
) -> bool {
    use hdf5_types::{FloatSize, IntSize};
    use TypeDescriptor::*;
    if src == dst && matches!(src, Integer(_) | Unsigned(_) | Float(_)) {
        let n = data.len().min(out.len());
        crate::par::zip_units(&data[..n], 1, &mut out[..n], 1, |s, d| d.copy_from_slice(s));
        return true;
    }
    match src {
        Integer(IntSize::U1) => cast_from::<i8>(dst, data, out),
        Integer(IntSize::U2) => cast_from::<i16>(dst, data, out),
        Integer(IntSize::U4) => cast_from::<i32>(dst, data, out),
        Integer(IntSize::U8) => cast_from::<i64>(dst, data, out),
        Unsigned(IntSize::U1) => cast_from::<u8>(dst, data, out),
        Unsigned(IntSize::U2) => cast_from::<u16>(dst, data, out),
        Unsigned(IntSize::U4) => cast_from::<u32>(dst, data, out),
        Unsigned(IntSize::U8) => cast_from::<u64>(dst, data, out),
        Float(FloatSize::U4) => cast_from::<f32>(dst, data, out),
        Float(FloatSize::U8) => cast_from::<f64>(dst, data, out),
        _ => false,
    }
}

/// Convert `n` elements from model/disk layout into user-memory layout.
///
/// The returned buffer is laid out per `dst` (`dst.size()` stride) and any
/// vlen components are freshly `malloc`-ed, matching the `hdf5-types` layouts;
/// ownership passes to the caller (typically via transmutation into `Vec<T>`).
pub fn model_to_mem(
    src: &TypeDescriptor,
    dst: &TypeDescriptor,
    data: &[u8],
    store: &VlenStore,
    n: usize,
) -> Result<Vec<u8>> {
    check_compatible(src, dst)?;
    let ssize = disk_size(src);
    let dsize = dst.size();
    let mut out = vec![0u8; n * dsize];
    if data.len() >= n * ssize && convert_numbers(src, dst, &data[..n * ssize], &mut out) {
        return Ok(out);
    }
    for i in 0..n {
        let s = data
            .get(i * ssize..(i + 1) * ssize)
            .ok_or("source data too short")?;
        elem_to_mem(src, dst, s, &mut out[i * dsize..(i + 1) * dsize], store)?;
    }
    Ok(out)
}

fn elem_to_mem(
    src: &TypeDescriptor,
    dst: &TypeDescriptor,
    s: &[u8],
    d: &mut [u8],
    store: &VlenStore,
) -> Result<()> {
    use TypeDescriptor::*;
    match (src, dst) {
        (VarLenAscii | VarLenUnicode, VarLenAscii | VarLenUnicode) => {
            let (len, idx) = slot_parts(s);
            let bytes: &[u8] = if idx == 0 {
                &[]
            } else {
                store
                    .get(idx as usize - 1)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[])
            };
            let len = (len as usize).min(bytes.len());
            let ptr = unsafe {
                let p = hdf5_types::malloc(len + 1).cast::<u8>();
                if p.is_null() {
                    return Err("out of memory".into());
                }
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, len);
                *p.add(len) = 0;
                p
            };
            d.copy_from_slice(&(ptr as usize).to_ne_bytes());
        }
        (VarLenArray(sb), VarLenArray(db)) => {
            let (len, idx) = slot_parts(s);
            let n = len as usize;
            let src_stride = disk_size(sb);
            let dst_stride = db.size();
            let bytes: &[u8] = if idx == 0 {
                &[]
            } else {
                store
                    .get(idx as usize - 1)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[])
            };
            let ptr = unsafe {
                let p = hdf5_types::malloc((n * dst_stride).max(1)).cast::<u8>();
                if p.is_null() {
                    return Err("out of memory".into());
                }
                p
            };
            for i in 0..n {
                let sd = bytes
                    .get(i * src_stride..(i + 1) * src_stride)
                    .ok_or("vlen store entry too short")?;
                let dd =
                    unsafe { std::slice::from_raw_parts_mut(ptr.add(i * dst_stride), dst_stride) };
                elem_to_mem(sb, db, sd, dd, store)?;
            }
            // hvl_t { len: usize, ptr }
            d[..PTR].copy_from_slice(&n.to_ne_bytes());
            d[PTR..2 * PTR].copy_from_slice(&(ptr as usize).to_ne_bytes());
        }
        (Compound(sc), Compound(dc)) => {
            for df in &dc.fields {
                let sf = sc.fields.iter().find(|f| f.name == df.name).unwrap();
                let ss = disk_size(&sf.ty);
                let ds = df.ty.size();
                let sslice = s
                    .get(sf.offset..sf.offset + ss)
                    .ok_or("compound field out of bounds")?;
                let dslice = &mut d[df.offset..df.offset + ds];
                elem_to_mem(&sf.ty, &df.ty, sslice, dslice, store)?;
            }
        }
        (FixedArray(sb, n), FixedArray(db, _)) => {
            let ss = disk_size(sb);
            let ds = db.size();
            for i in 0..*n {
                elem_to_mem(
                    sb,
                    db,
                    &s[i * ss..(i + 1) * ss],
                    &mut d[i * ds..(i + 1) * ds],
                    store,
                )?;
            }
        }
        (FixedAscii(_) | FixedUnicode(_), FixedAscii(_) | FixedUnicode(_)) => {
            let n = s.len().min(d.len());
            d[..n].copy_from_slice(&s[..n]);
            for b in &mut d[n..] {
                *b = 0;
            }
        }
        (FixedAscii(_) | FixedUnicode(_), VarLenAscii | VarLenUnicode) => {
            // trim trailing NUL padding, then hand out a C string
            let end = s.iter().rposition(|&b| b != 0).map(|i| i + 1).unwrap_or(0);
            let bytes = &s[..end];
            let ptr = unsafe {
                let p = hdf5_types::malloc(bytes.len() + 1).cast::<u8>();
                if p.is_null() {
                    return Err("out of memory".into());
                }
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len());
                *p.add(bytes.len()) = 0;
                p
            };
            d.copy_from_slice(&(ptr as usize).to_ne_bytes());
        }
        (VarLenAscii | VarLenUnicode, FixedAscii(_) | FixedUnicode(_)) => {
            let (len, idx) = slot_parts(s);
            let bytes: &[u8] = if idx == 0 {
                &[]
            } else {
                store
                    .get(idx as usize - 1)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[])
            };
            let n = (len as usize).min(bytes.len()).min(d.len());
            d[..n].copy_from_slice(&bytes[..n]);
            for b in &mut d[n..] {
                *b = 0;
            }
        }
        (Enum(_), Boolean) | (Boolean, Enum(_)) => {
            d[0] = s[0];
        }
        _ => {
            if !convert_numeric(src, dst, s, d) {
                // identical scalar layouts
                let n = d.len().min(s.len());
                d[..n].copy_from_slice(&s[..n]);
            }
        }
    }
    Ok(())
}

/// Convert `n` elements from user-memory layout into model/disk layout,
/// pushing vlen payloads into `store`. The source memory is only read; the
/// caller retains ownership of any pointers inside it.
pub fn mem_to_model(
    src: &TypeDescriptor,
    dst: &TypeDescriptor,
    data: &[u8],
    store: &mut VlenStore,
    n: usize,
) -> Result<Vec<u8>> {
    check_compatible(src, dst)?;
    let ssize = src.size();
    let dsize = disk_size(dst);
    let mut out = vec![0u8; n * dsize];
    if data.len() >= n * ssize && convert_numbers(src, dst, &data[..n * ssize], &mut out) {
        return Ok(out);
    }
    for i in 0..n {
        let s = data
            .get(i * ssize..(i + 1) * ssize)
            .ok_or("source data too short")?;
        elem_to_model(src, dst, s, &mut out[i * dsize..(i + 1) * dsize], store)?;
    }
    Ok(out)
}

fn elem_to_model(
    src: &TypeDescriptor,
    dst: &TypeDescriptor,
    s: &[u8],
    d: &mut [u8],
    store: &mut VlenStore,
) -> Result<()> {
    use TypeDescriptor::*;
    match (src, dst) {
        (VarLenAscii | VarLenUnicode, VarLenAscii | VarLenUnicode) => {
            let ptr = read_word(s, 0) as *const u8;
            let bytes = if ptr.is_null() {
                Vec::new()
            } else {
                let mut v = Vec::new();
                let mut p = ptr;
                unsafe {
                    while *p != 0 {
                        v.push(*p);
                        p = p.add(1);
                    }
                }
                v
            };
            let len = bytes.len() as u32;
            let idx = if bytes.is_empty() {
                0
            } else {
                store.push(bytes);
                store.len() as u32
            };
            d.copy_from_slice(&model_slot(len, idx));
        }
        (VarLenArray(sb), VarLenArray(db)) => {
            // hvl_t { len: usize, ptr }
            let len = read_word(s, 0);
            let ptr = read_word(s, PTR) as *const u8;
            let src_stride = sb.size();
            let dst_stride = disk_size(db);
            let mut buf = vec![0u8; len * dst_stride];
            for i in 0..len {
                let sd = unsafe { std::slice::from_raw_parts(ptr.add(i * src_stride), src_stride) };
                elem_to_model(
                    sb,
                    db,
                    sd,
                    &mut buf[i * dst_stride..(i + 1) * dst_stride],
                    store,
                )?;
            }
            let idx = if len == 0 {
                0
            } else {
                store.push(buf);
                store.len() as u32
            };
            d.copy_from_slice(&model_slot(len as u32, idx));
        }
        (Compound(sc), Compound(dc)) => {
            for df in &dc.fields {
                let sf = sc
                    .fields
                    .iter()
                    .find(|f| f.name == df.name)
                    .ok_or_else(|| format!("missing compound field '{}'", df.name))?;
                let ss = sf.ty.size();
                let ds = disk_size(&df.ty);
                let sslice = s
                    .get(sf.offset..sf.offset + ss)
                    .ok_or("compound field out of bounds")?;
                let s_copy = sslice.to_vec();
                elem_to_model(
                    &sf.ty,
                    &df.ty,
                    &s_copy,
                    &mut d[df.offset..df.offset + ds],
                    store,
                )?;
            }
        }
        (FixedArray(sb, n), FixedArray(db, _)) => {
            let ss = sb.size();
            let ds = disk_size(db);
            for i in 0..*n {
                let s_copy = s[i * ss..(i + 1) * ss].to_vec();
                elem_to_model(sb, db, &s_copy, &mut d[i * ds..(i + 1) * ds], store)?;
            }
        }
        (FixedAscii(_) | FixedUnicode(_), FixedAscii(_) | FixedUnicode(_)) => {
            let n = s.len().min(d.len());
            d[..n].copy_from_slice(&s[..n]);
            for b in &mut d[n..] {
                *b = 0;
            }
        }
        (FixedAscii(_) | FixedUnicode(_), VarLenAscii | VarLenUnicode) => {
            let end = s.iter().rposition(|&b| b != 0).map(|i| i + 1).unwrap_or(0);
            let bytes = s[..end].to_vec();
            let len = bytes.len() as u32;
            let idx = if bytes.is_empty() {
                0
            } else {
                store.push(bytes);
                store.len() as u32
            };
            d.copy_from_slice(&model_slot(len, idx));
        }
        (VarLenAscii | VarLenUnicode, FixedAscii(_) | FixedUnicode(_)) => {
            let ptr = read_word(s, 0) as *const u8;
            let mut n = 0usize;
            unsafe {
                while n < d.len() && !ptr.is_null() && *ptr.add(n) != 0 {
                    d[n] = *ptr.add(n);
                    n += 1;
                }
            }
            for b in &mut d[n..] {
                *b = 0;
            }
        }
        (Enum(_), Boolean) | (Boolean, Enum(_)) => {
            d[0] = s[0];
        }
        _ => {
            if !convert_numeric(src, dst, s, d) {
                let n = d.len().min(s.len());
                d[..n].copy_from_slice(&s[..n]);
            }
        }
    }
    Ok(())
}
