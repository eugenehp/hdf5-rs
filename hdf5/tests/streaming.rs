//! Writing a file bigger than you want in memory: flushes append to the file
//! (data already there is never rewritten), written datasets are then held as
//! views of the file, and writes into chunked datasets go straight to disk,
//! one touched chunk at a time.

use hdf5::File;
use ndarray::{s, Array2};

/// `tempfile::tempdir`, also under WASI (see tests/test_roundtrip.rs).
fn tempdir() -> std::io::Result<tempfile::TempDir> {
    #[cfg(target_os = "wasi")]
    let _ = tempfile::env::override_temp_dir(std::path::Path::new("/tmp"));
    tempfile::tempdir()
}

fn block(step: usize, rows: usize) -> Array2<f32> {
    Array2::from_shape_fn((rows, 8), |(i, j)| ((step * rows + i) * 8 + j) as f32)
}

#[test]
fn a_recording_grows_the_file_by_what_is_written() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("rec.h5");
    let (steps, rows) = (60, 500);
    let mut sizes = Vec::new();
    {
        let f = File::create(&path).unwrap();
        let ds = f
            .new_dataset::<f32>()
            .shape((0.., 8))
            .chunk((500, 8))
            .shuffle()
            .deflate(1)
            .create("signal")
            .unwrap();
        f.flush().unwrap();
        for step in 0..steps {
            ds.resize(((step + 1) * rows, 8)).unwrap();
            ds.write_slice(&block(step, rows), s![step * rows..(step + 1) * rows, ..])
                .unwrap();
            if step % 10 == 9 {
                f.flush().unwrap();
                sizes.push(std::fs::metadata(&path).unwrap().len());
            }
            // reading back mid-recording sees what was written
            let last: Array2<f32> = ds.read_slice(s![step * rows.., ..]).unwrap();
            assert_eq!(last, block(step, rows));
        }
    }
    // each flush adds about the same (new chunks + metadata): linear, not
    // the whole file again
    let growth: Vec<u64> = sizes.windows(2).map(|w| w[1] - w[0]).collect();
    let (lo, hi) = (*growth.iter().min().unwrap(), *growth.iter().max().unwrap());
    assert!(hi < lo * 3, "flush growth {growth:?}");
    let f = File::open(&path).unwrap();
    let all: Array2<f32> = f.dataset("signal").unwrap().read_2d().unwrap();
    assert_eq!(all.shape(), [steps * rows, 8]);
    for step in 0..steps {
        assert_eq!(
            all.slice(s![step * rows..(step + 1) * rows, ..]),
            block(step, rows)
        );
    }
}

#[test]
fn editing_a_file_rewrites_only_what_changed() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("edit.h5");
    let data = Array2::from_shape_fn((20_000, 16), |(i, j)| (i * 16 + j) as i32);
    {
        let f = File::create(&path).unwrap();
        f.new_dataset_builder()
            .chunk((1000, 16))
            .deflate(4)
            .with_data(&data)
            .create("big")
            .unwrap();
        f.new_dataset_builder()
            .with_data(&[1.5f64, 2.5])
            .create("small")
            .unwrap();
    }
    let before = std::fs::metadata(&path).unwrap().len();
    {
        let f = File::open_rw(&path).unwrap();
        let patch = Array2::from_elem((3, 16), -7i32);
        f.dataset("big")
            .unwrap()
            .write_slice(&patch, s![10_500..10_503, ..])
            .unwrap();
        f.new_dataset_builder()
            .with_data(&[9u8; 4])
            .create("added")
            .unwrap();
    }
    let after = std::fs::metadata(&path).unwrap().len();
    // one re-encoded chunk and fresh metadata, not another copy of the file
    assert!(after - before < before / 4, "{before} -> {after}");
    // opened and closed without changes: not written at all
    drop(File::open_rw(&path).unwrap());
    assert_eq!(std::fs::metadata(&path).unwrap().len(), after);

    let f = File::open(&path).unwrap();
    let mut want = data.clone();
    want.slice_mut(s![10_500..10_503, ..]).fill(-7);
    assert_eq!(f.dataset("big").unwrap().read_2d::<i32>().unwrap(), want);
    assert_eq!(
        f.dataset("small").unwrap().read_raw::<f64>().unwrap(),
        [1.5, 2.5]
    );
    assert_eq!(
        f.dataset("added").unwrap().read_raw::<u8>().unwrap(),
        [9; 4]
    );
}

/// A recording far bigger than memory (the shape of a 183 GB DANDI
/// Neuropixels series): it opens, and rows read through a range reader. (A
/// whole read is the OS's call: refused where memory is not overcommitted,
/// an error then rather than an abort.)
#[test]
fn a_dataset_bigger_than_memory_opens_and_reads_in_part() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("huge.h5");
    let (rows, channels) = (238_479_133usize, 384usize);
    {
        let f = File::create(&path).unwrap();
        let ds = f
            .new_dataset::<i16>()
            .shape((0.., channels))
            .chunk((232_890, 1))
            .deflate(1)
            .create("data")
            .unwrap();
        f.flush().unwrap();
        ds.resize((rows, channels)).unwrap();
        let head = ndarray::Array2::from_shape_fn((10, channels), |(i, c)| (i * 1000 + c) as i16);
        ds.write_slice(&head, s![0..10, ..]).unwrap();
    }
    let reader = std::fs::File::open(&path).unwrap();
    let f = File::open_reader(reader).unwrap();
    let ds = f.dataset("data").unwrap();
    assert_eq!(ds.shape(), [rows, channels]);
    let got = ds.read_rows_raw::<i16>(0, 10).unwrap();
    assert_eq!(got[..3], [0, 1, 2]);
    assert_eq!(got[9 * channels + 5], 9005);
    // never written: fill
    assert!(ds
        .read_rows_raw::<i16>(rows - 2, rows)
        .unwrap()
        .iter()
        .all(|&v| v == 0));
}
