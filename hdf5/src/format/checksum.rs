//! Jenkins `lookup3` checksum, used by HDF5 for version-2 metadata structures
//! (superblock v2/v3, v2 B-trees, fractal heaps, object header v2). Needed for
//! reading files written in the newer format.

#[inline]
fn rot(x: u32, k: u32) -> u32 {
    x.rotate_left(k)
}

/// Compute the HDF5 metadata checksum (Bob Jenkins `hashlittle`) over `data`.
pub fn checksum(data: &[u8]) -> u32 {
    let mut length = data.len();
    let initval: u32 = 0;
    let mut a: u32 = 0xdead_beefu32
        .wrapping_add(length as u32)
        .wrapping_add(initval);
    let mut b = a;
    let mut c = a;

    let mut offset = 0usize;

    // Process 12-byte blocks.
    while length > 12 {
        a = a.wrapping_add(u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]));
        b = b.wrapping_add(u32::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]));
        c = c.wrapping_add(u32::from_le_bytes([
            data[offset + 8],
            data[offset + 9],
            data[offset + 10],
            data[offset + 11],
        ]));

        // mix(a, b, c)
        a = a.wrapping_sub(c);
        a ^= rot(c, 4);
        c = c.wrapping_add(b);
        b = b.wrapping_sub(a);
        b ^= rot(a, 6);
        a = a.wrapping_add(c);
        c = c.wrapping_sub(b);
        c ^= rot(b, 8);
        b = b.wrapping_add(a);
        a = a.wrapping_sub(c);
        a ^= rot(c, 16);
        c = c.wrapping_add(b);
        b = b.wrapping_sub(a);
        b ^= rot(a, 19);
        a = a.wrapping_add(c);
        c = c.wrapping_sub(b);
        c ^= rot(b, 4);
        b = b.wrapping_add(a);

        offset += 12;
        length -= 12;
    }

    // Handle the last (probably partial) block of 1..=12 bytes.
    let mut tail = [0u8; 12];
    tail[..length].copy_from_slice(&data[offset..offset + length]);
    if length > 0 {
        a = a.wrapping_add(u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]));
    }
    if length > 4 {
        b = b.wrapping_add(u32::from_le_bytes([tail[4], tail[5], tail[6], tail[7]]));
    }
    if length > 8 {
        c = c.wrapping_add(u32::from_le_bytes([tail[8], tail[9], tail[10], tail[11]]));
    }

    if length == 0 {
        return c; // zero-length data => c unchanged
    }

    // final(a, b, c)
    c ^= b;
    c = c.wrapping_sub(rot(b, 14));
    a ^= c;
    a = a.wrapping_sub(rot(c, 11));
    b ^= a;
    b = b.wrapping_sub(rot(a, 25));
    c ^= b;
    c = c.wrapping_sub(rot(b, 16));
    a ^= c;
    a = a.wrapping_sub(rot(c, 4));
    b ^= a;
    b = b.wrapping_sub(rot(a, 14));
    c ^= b;
    c = c.wrapping_sub(rot(b, 24));

    c
}

#[cfg(test)]
mod tests {
    use super::checksum;

    #[test]
    fn known_vectors() {
        // Bob Jenkins lookup3 hashlittle reference values (initval = 0).
        assert_eq!(checksum(b"Four score and seven years ago"), 0x17770551);
        assert_eq!(checksum(b""), 0xdeadbeef);
    }
}
