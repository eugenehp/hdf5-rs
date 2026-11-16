//! Slices and point selections of lazily stored datasets read only what they
//! select: the stored runs (contiguous) or the chunks holding a selected
//! element (chunked) — never the whole dataset.

use std::io::{Cursor, Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;

use hdf5::{File, Selection};
use ndarray::{arr2, s, Array1, Array2, Array3, ArrayD};

fn cube() -> Array3<f32> {
    Array3::from_shape_fn((400, 30, 20), |(i, j, k)| (i * 600 + j * 20 + k) as f32)
}

fn bytes() -> Vec<u8> {
    let f = File::create_in_memory().unwrap();
    let data = cube();
    f.new_dataset_builder()
        .chunk((25, 10, 20))
        .shuffle()
        .deflate(3)
        .with_data(&data)
        .create("gzip")
        .unwrap();
    f.new_dataset_builder()
        .with_data(&data)
        .create("contig")
        .unwrap();
    f.new_dataset_builder()
        .chunk((25, 10, 20))
        .with_data(&data)
        .create("raw")
        .unwrap();
    f.to_bytes().unwrap()
}

/// Counts the bytes a reader hands out.
struct Counted(Cursor<Vec<u8>>, Arc<AtomicU64>);
impl Read for Counted {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.0.read(buf)?;
        self.1.fetch_add(n as u64, Relaxed);
        Ok(n)
    }
}
impl Seek for Counted {
    fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
        self.0.seek(to)
    }
}

#[test]
fn slices_match_the_full_array() {
    let want = cube();
    let f = File::from_bytes(bytes()).unwrap();
    for name in ["gzip", "contig", "raw"] {
        let d = f.dataset(name).unwrap();
        let a: ArrayD<f32> = d.read_slice(s![10..20, 3..4, ..]).unwrap();
        assert_eq!(a, want.slice(s![10..20, 3..4, ..]).into_dyn(), "{name}");
        let b: Array2<f32> = d.read_slice(s![7, 1..29;4, 5..]).unwrap();
        assert_eq!(b, want.slice(s![7, 1..29;4, 5..]), "{name}");
        let c: Array1<f64> = d.read_slice(s![399, 29, ..]).unwrap();
        assert_eq!(c, want.slice(s![399, 29, ..]).mapv(f64::from), "{name}");
        let e: ArrayD<f32> = d.read_slice(s![..;37, ..;9, ..;7]).unwrap();
        assert_eq!(e, want.slice(s![..;37, ..;9, ..;7]).into_dyn(), "{name}");
        // the same again: from the chunk cache
        let e2: ArrayD<f32> = d.read_slice(s![..;37, ..;9, ..;7]).unwrap();
        assert_eq!(e, e2);
        let points = Selection::from(arr2(&[[0, 0, 0], [399, 29, 19], [123, 4, 5], [0, 0, 0]]));
        let p: Array1<f32> = d.read_slice(points).unwrap();
        assert_eq!(
            p.to_vec(),
            [
                want[[0, 0, 0]],
                want[[399, 29, 19]],
                want[[123, 4, 5]],
                want[[0, 0, 0]]
            ],
            "{name}"
        );
    }
}

#[test]
fn a_small_slice_reads_a_small_part() {
    let bytes = bytes();
    let total = bytes.len() as u64;
    for name in ["gzip", "contig", "raw"] {
        let read = Arc::new(AtomicU64::new(0));
        // small fetch blocks, so what is read tracks what is asked for
        let f =
            File::open_reader_with_block(Counted(Cursor::new(bytes.clone()), read.clone()), 4096)
                .unwrap();
        let d = f.dataset(name).unwrap();
        let before = read.load(Relaxed);
        let a: ArrayD<f32> = d.read_slice(s![200..205, 12..14, ..]).unwrap();
        assert_eq!(a, cube().slice(s![200..205, 12..14, ..]).into_dyn());
        let used = read.load(Relaxed) - before;
        assert!(used < total / 20, "{name}: read {used} of {total} bytes");
    }
}
