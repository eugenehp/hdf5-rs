//! Encoding and decoding of HDF5 datatype messages (message type `0x0003`) to
//! and from [`TypeDescriptor`].
//!
//! Integers, floats, strings, compounds and enums are written using datatype
//! version 1, arrays using version 2 — matching what libhdf5/h5py emit — so the
//! results are byte-compatible on read. Decoding accepts versions 1–3.

use hdf5_types::{
    CompoundField, CompoundType, EnumMember, EnumType, FloatSize, IntSize, TypeDescriptor,
};

use super::{align8, Buf, Cursor};
use crate::error::Result;

/// Byte-order information mirroring a decoded descriptor tree, used to
/// normalize big-endian file data to little-endian at load time (the same
/// normalization libhdf5 performs via its order-conversion paths).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrderTree {
    /// Nothing below needs swapping.
    None,
    /// Swap this scalar leaf (width in bytes).
    SwapLeaf(usize),
    /// Recurse into compound fields at their disk offsets.
    Compound(Vec<(usize, OrderTree)>),
    /// `n` consecutive elements of `stride` bytes each.
    Array {
        n: usize,
        stride: usize,
        inner: Box<OrderTree>,
    },
    /// Variable-length data: swap base elements inside the side store.
    VarLen {
        base_stride: usize,
        inner: Box<OrderTree>,
    },
}

impl OrderTree {
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    fn from_parts(parts: Vec<(usize, OrderTree)>) -> Self {
        if parts.iter().all(|(_, t)| t.is_none()) {
            Self::None
        } else {
            Self::Compound(parts)
        }
    }
}

// Datatype classes.
const CLASS_INT: u8 = 0;
const CLASS_BITFIELD: u8 = 4;
const CLASS_OPAQUE: u8 = 5;
const CLASS_REFERENCE: u8 = 7;
const CLASS_FLOAT: u8 = 1;
const CLASS_STRING: u8 = 3;
const CLASS_COMPOUND: u8 = 6;
const CLASS_ENUM: u8 = 8;
const CLASS_VLEN: u8 = 9;
const CLASS_ARRAY: u8 = 10;

/// Encode a type descriptor into datatype-message bytes.
pub fn encode(desc: &TypeDescriptor) -> Vec<u8> {
    let mut b = Buf::new();
    encode_into(&mut b, desc);
    b.bytes
}

fn class_version_byte(version: u8, class: u8) -> u8 {
    (version << 4) | class
}

fn encode_into(b: &mut Buf, desc: &TypeDescriptor) {
    match desc {
        TypeDescriptor::Integer(size) => encode_int(b, *size as usize, true),
        TypeDescriptor::Unsigned(size) => encode_int(b, *size as usize, false),
        TypeDescriptor::Float(size) => encode_float(b, *size),
        TypeDescriptor::Boolean => encode_bool(b),
        TypeDescriptor::Enum(e) => encode_enum(b, e),
        TypeDescriptor::Compound(c) => encode_compound(b, c),
        TypeDescriptor::FixedArray(base, n) => encode_array(b, base, *n),
        TypeDescriptor::FixedAscii(n) => encode_string(b, *n, false),
        TypeDescriptor::FixedUnicode(n) => encode_string(b, *n, true),
        TypeDescriptor::VarLenAscii => encode_vlen_string(b, false),
        TypeDescriptor::VarLenUnicode => encode_vlen_string(b, true),
        TypeDescriptor::VarLenArray(base) => encode_vlen_seq(b, base),
    }
}

fn encode_int(b: &mut Buf, size: usize, signed: bool) {
    b.u8(class_version_byte(1, CLASS_INT));
    // bitfield: bit0 = byte order (0=LE), bit3 = signed
    b.u8(if signed { 0x08 } else { 0x00 });
    b.u8(0);
    b.u8(0);
    b.u32(size as u32);
    b.u16(0); // bit offset
    b.u16((size * 8) as u16); // precision
}

fn encode_float(b: &mut Buf, size: FloatSize) {
    b.u8(class_version_byte(1, CLASS_FLOAT));
    let (bytes, precision, exp_loc, exp_size, mant_loc, mant_size, bias, sign_loc) = match size {
        FloatSize::U4 => (4usize, 32u16, 23u8, 8u8, 0u8, 23u8, 127u32, 31u8),
        FloatSize::U8 => (8, 64, 52, 11, 0, 52, 1023, 63),
        #[cfg(feature = "f16")]
        FloatSize::U2 => (2, 16, 10, 5, 0, 10, 15, 15),
    };
    // bitfield: [0x20 (LE, normalization=implied MSB), sign location, 0]
    b.u8(0x20);
    b.u8(sign_loc);
    b.u8(0);
    b.u32(bytes as u32);
    b.u16(0); // bit offset
    b.u16(precision);
    b.u8(exp_loc);
    b.u8(exp_size);
    b.u8(mant_loc);
    b.u8(mant_size);
    b.u32(bias);
}

fn encode_string(b: &mut Buf, n: usize, unicode: bool) {
    b.u8(class_version_byte(1, CLASS_STRING));
    // bits 0-3: padding (1 = null pad), bits 4-7: charset (0=ascii, 1=utf8)
    let charset = if unicode { 1u8 } else { 0 };
    b.u8((charset << 4) | 0x01);
    b.u8(0);
    b.u8(0);
    b.u32(n as u32);
}

fn encode_bool(b: &mut Buf) {
    let e = EnumType {
        size: IntSize::U1,
        signed: true,
        members: vec![
            EnumMember {
                name: "FALSE".into(),
                value: 0,
            },
            EnumMember {
                name: "TRUE".into(),
                value: 1,
            },
        ],
    };
    encode_enum(b, &e);
}

fn encode_enum(b: &mut Buf, e: &EnumType) {
    b.u8(class_version_byte(1, CLASS_ENUM));
    b.u16(e.members.len() as u16);
    b.u8(0);
    let base_size = e.size as usize;
    b.u32(base_size as u32);
    // base integer type
    encode_int(b, base_size, e.signed);
    // member names: null-terminated, padded to a multiple of 8 (v1)
    for m in &e.members {
        let start = b.len();
        b.raw(m.name.as_bytes());
        b.u8(0);
        let padded = align8(b.len() - start);
        b.zeros(padded - (b.len() - start));
    }
    // member values, each `base_size` bytes, little-endian
    for m in &e.members {
        b.raw(&m.value.to_le_bytes()[..base_size]);
    }
}

fn encode_compound(b: &mut Buf, c: &CompoundType) {
    b.u8(class_version_byte(1, CLASS_COMPOUND));
    b.u16(c.fields.len() as u16);
    b.u8(0);
    b.u32(c.size as u32);
    let mut fields: Vec<&CompoundField> = c.fields.iter().collect();
    fields.sort_by_key(|f| f.index);
    for f in fields {
        // name null-terminated, padded to a multiple of 8
        let start = b.len();
        b.raw(f.name.as_bytes());
        b.u8(0);
        let padded = align8(b.len() - start);
        b.zeros(padded - (b.len() - start));
        // byte offset (4 bytes, v1)
        b.u32(f.offset as u32);
        // v1 member "array" baggage: dimensionality(1) + reserved(3) +
        // permutation(4) + reserved(4) + 4 dimension sizes (16) = 28 bytes
        b.zeros(28);
        // member datatype
        encode_into(b, &f.ty);
    }
}

fn encode_array(b: &mut Buf, base: &TypeDescriptor, n: usize) {
    // Array datatype, version 2 (includes permutation indices), matching h5py.
    b.u8(class_version_byte(2, CLASS_ARRAY));
    b.u8(0);
    b.u8(0);
    b.u8(0);
    b.u32((super::convert::disk_size(base) * n) as u32);
    b.u8(1); // dimensionality
    b.u8(0);
    b.u8(0);
    b.u8(0);
    b.u32(n as u32); // dim size
    b.u32(0); // permutation index (v2)
    encode_into(b, base);
}

fn encode_vlen_string(b: &mut Buf, unicode: bool) {
    b.u8(class_version_byte(1, CLASS_VLEN));
    // bits 0-3 = type (1 = string), bits 4-7 = padding (0 = null term)
    b.u8(0x01);
    // charset byte: bits 0-3 = charset (0=ascii, 1=utf8)
    b.u8(if unicode { 0x01 } else { 0x00 });
    b.u8(0);
    b.u32(16); // on-disk size: {len: u32, gheap addr: u64, index: u32}
               // base type: 8-bit character
    encode_int(b, 1, false);
}

fn encode_vlen_seq(b: &mut Buf, base: &TypeDescriptor) {
    b.u8(class_version_byte(1, CLASS_VLEN));
    b.u8(0x00); // type = sequence
    b.u8(0);
    b.u8(0);
    b.u32(16); // on-disk size: {len: u32, gheap addr: u64, index: u32}
    encode_into(b, base);
}

/// Decode a datatype message starting at the cursor's current position.
pub fn decode(c: &mut Cursor) -> Result<TypeDescriptor> {
    let mut order = OrderTree::None;
    decode_ordered(c, &mut order)
}

/// Decode a datatype message, also reporting which components are stored
/// big-endian (so callers can byte-swap the data to little-endian).
pub fn decode_ordered(c: &mut Cursor, order: &mut OrderTree) -> Result<TypeDescriptor> {
    let start = c.pos;
    let vc = c.u8()?;
    let version = vc >> 4;
    let class = vc & 0x0f;
    let bitfield = [c.u8()?, c.u8()?, c.u8()?];
    let size = c.u32()? as usize;
    match class {
        CLASS_INT => {
            // bit 0 of the class bitfield is the byte order (1 = big-endian);
            // BE data is byte-swapped to LE at load via the order tree.
            let swapped = bitfield[0] & 0x01 != 0;
            let _bit_offset = c.u16()?;
            let _precision = c.u16()?;
            let signed = (bitfield[0] & 0x08) != 0;
            let isize = IntSize::from_int(size).ok_or("invalid integer size")?;
            *order = if swapped {
                OrderTree::SwapLeaf(size)
            } else {
                OrderTree::None
            };
            Ok(if signed {
                TypeDescriptor::Integer(isize)
            } else {
                TypeDescriptor::Unsigned(isize)
            })
        }
        CLASS_FLOAT => {
            if bitfield[0] & 0x01 != 0 {
                *order = OrderTree::SwapLeaf(size);
            }
            let _bit_offset = c.u16()?;
            let _precision = c.u16()?;
            let _exp_loc = c.u8()?;
            let _exp_size = c.u8()?;
            let _mant_loc = c.u8()?;
            let _mant_size = c.u8()?;
            let _bias = c.u32()?;
            let fsize = FloatSize::from_int(size).ok_or("unsupported float size")?;
            Ok(TypeDescriptor::Float(fsize))
        }
        CLASS_STRING => {
            let charset = (bitfield[0] >> 4) & 0x0f;
            Ok(if charset == 1 {
                TypeDescriptor::FixedUnicode(size)
            } else {
                TypeDescriptor::FixedAscii(size)
            })
        }
        CLASS_ENUM => {
            let nmembers = u16::from_le_bytes([bitfield[0], bitfield[1]]) as usize;
            let mut base_order = OrderTree::None;
            let base = decode_ordered(c, &mut base_order)?;
            let base_swapped = !base_order.is_none();
            let (base_size, signed) = match &base {
                TypeDescriptor::Integer(s) => (*s as usize, true),
                TypeDescriptor::Unsigned(s) => (*s as usize, false),
                _ => return Err("enum base must be integer".into()),
            };
            let mut names = Vec::with_capacity((nmembers).min(1 << 16));
            for _ in 0..nmembers {
                let nstart = c.pos;
                let mut name = Vec::new();
                loop {
                    let byte = c.u8()?;
                    if byte == 0 {
                        break;
                    }
                    name.push(byte);
                }
                // v1/v2: names padded to a multiple of 8
                if version < 3 {
                    let consumed = c.pos - nstart;
                    let pad = align8(consumed) - consumed;
                    c.skip(pad);
                }
                names.push(String::from_utf8_lossy(&name).into_owned());
            }
            let mut members = Vec::with_capacity((nmembers).min(1 << 16));
            for name in names {
                let raw = c.take(base_size)?;
                let mut v = [0u8; 8];
                // enum member values are stored in the base type's byte order
                if base_swapped {
                    for (i, b) in raw.iter().rev().enumerate() {
                        v[i] = *b;
                    }
                } else {
                    v[..base_size].copy_from_slice(raw);
                }
                members.push(EnumMember {
                    name,
                    value: u64::from_le_bytes(v),
                });
            }
            *order = base_order;
            let isize = IntSize::from_int(base_size).ok_or("invalid enum base size")?;
            let etype = EnumType {
                size: isize,
                signed,
                members,
            };
            // Recognize the boolean pattern.
            if is_bool_enum(&etype) {
                Ok(TypeDescriptor::Boolean)
            } else {
                Ok(TypeDescriptor::Enum(etype))
            }
        }
        CLASS_COMPOUND => {
            let nfields = u16::from_le_bytes([bitfield[0], bitfield[1]]) as usize;
            let mut fields = Vec::with_capacity((nfields).min(1 << 16));
            let mut field_orders = Vec::with_capacity((nfields).min(1 << 16));
            for index in 0..nfields {
                let nstart = c.pos;
                let mut name = Vec::new();
                loop {
                    let byte = c.u8()?;
                    if byte == 0 {
                        break;
                    }
                    name.push(byte);
                }
                if version < 3 {
                    let consumed = c.pos - nstart;
                    let pad = align8(consumed) - consumed;
                    c.skip(pad);
                }
                let offset = if version >= 3 {
                    // variable-size offset (bytes needed to hold `size`)
                    let nbytes = bytes_for(size);
                    c.uint(nbytes)? as usize
                } else {
                    c.u32()? as usize
                };
                if version == 1 {
                    // skip the 28-byte array baggage
                    c.skip(28);
                }
                let mut forder = OrderTree::None;
                let ty = decode_ordered(c, &mut forder)?;
                field_orders.push((offset, forder));
                fields.push(CompoundField {
                    name: String::from_utf8_lossy(&name).into_owned(),
                    ty,
                    offset,
                    index,
                });
            }
            *order = OrderTree::from_parts(field_orders);
            Ok(TypeDescriptor::Compound(CompoundType { fields, size }))
        }
        CLASS_ARRAY => {
            let ndim = c.u8()? as usize;
            if version == 2 {
                c.skip(3); // reserved
            }
            let mut total = 1usize;
            let mut dims = Vec::with_capacity((ndim).min(1 << 16));
            for _ in 0..ndim {
                let d = c.u32()? as usize;
                dims.push(d);
                total *= d;
            }
            if version == 2 {
                // permutation indices
                c.skip(4 * ndim);
            }
            let mut base_order = OrderTree::None;
            let base = decode_ordered(c, &mut base_order)?;
            let stride = super::convert::disk_size(&base);
            let mut desc = base;
            if !base_order.is_none() {
                *order = OrderTree::Array {
                    n: total,
                    stride,
                    inner: Box::new(base_order),
                };
            }
            for &d in dims.iter().rev() {
                desc = TypeDescriptor::FixedArray(Box::new(desc), d);
            }
            Ok(desc)
        }
        CLASS_VLEN => {
            let vlen_type = bitfield[0] & 0x0f;
            let mut base_order = OrderTree::None;
            let base = decode_ordered(c, &mut base_order)?;
            if vlen_type != 1 && !base_order.is_none() {
                *order = OrderTree::VarLen {
                    base_stride: super::convert::disk_size(&base),
                    inner: Box::new(base_order),
                };
            }
            if vlen_type == 1 {
                // vlen string
                let charset = bitfield[1] & 0x0f;
                Ok(if charset == 1 {
                    TypeDescriptor::VarLenUnicode
                } else {
                    TypeDescriptor::VarLenAscii
                })
            } else {
                Ok(TypeDescriptor::VarLenArray(Box::new(base)))
            }
        }
        CLASS_BITFIELD => {
            // bitfields read as unsigned integers of the same width (the
            // h5py convention); BE handled like integers
            let swapped = bitfield[0] & 0x01 != 0;
            let _bit_offset = c.u16()?;
            let _precision = c.u16()?;
            let isize = IntSize::from_int(size).ok_or("invalid bitfield size")?;
            *order = if swapped {
                OrderTree::SwapLeaf(size)
            } else {
                OrderTree::None
            };
            Ok(TypeDescriptor::Unsigned(isize))
        }
        CLASS_OPAQUE | CLASS_REFERENCE => {
            // opaque blobs and object/region references (8/12 bytes) read as
            // fixed byte arrays; the raw bytes are the reference tokens
            if class == CLASS_OPAQUE {
                // skip the ASCII tag (length = multiple of 8 from bitfield[0])
                c.skip(bitfield[0] as usize);
            }
            *order = OrderTree::None;
            Ok(TypeDescriptor::FixedArray(
                Box::new(TypeDescriptor::Unsigned(IntSize::U1)),
                size,
            ))
        }
        _ => {
            let _ = start;
            Err(format!("unsupported datatype class {class}").into())
        }
    }
}

fn is_bool_enum(e: &EnumType) -> bool {
    e.size == IntSize::U1
        && e.members.len() == 2
        && e.members.iter().any(|m| m.name == "FALSE" && m.value == 0)
        && e.members.iter().any(|m| m.name == "TRUE" && m.value == 1)
}

/// Minimum number of bytes needed to represent `n`.
fn bytes_for(n: usize) -> usize {
    let mut bits: usize = 0;
    let mut v = n;
    while v > 0 {
        bits += 1;
        v >>= 1;
    }
    bits.div_ceil(8).max(1)
}
