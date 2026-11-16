//! Parallel chunk encode/decode must give exactly what one thread gives:
//! odd chunk shapes, partial edge chunks, row ranges that cut through
//! chunks, filters and type conversion.

use hdf5::File;
use ndarray::{s, Array3, ArrayD, IxDyn};

fn cube() -> Array3<i32> {
    Array3::from_shape_fn((97, 13, 7), |(i, j, k)| {
        (i * 1000 + j * 10 + k) as i32 - 50_000
    })
}

fn file() -> File {
    let f = File::create_in_memory().unwrap();
    let data = cube();
    for (name, chunk) in [
        ("tall", (10, 13, 7)),
        ("odd", (7, 5, 3)),
        ("cols", (97, 1, 1)),
        ("tiny", (1, 2, 2)),
    ] {
        f.new_dataset_builder()
            .chunk(chunk)
            .shuffle()
            .deflate(1)
            .with_data(&data)
            .create(name)
            .unwrap();
        f.new_dataset_builder()
            .chunk(chunk)
            .with_data(&data)
            .create(format!("{name}_raw").as_str())
            .unwrap();
    }
    File::from_bytes(f.to_bytes().unwrap()).unwrap()
}

#[test]
fn every_thread_count_reads_the_same() {
    let want = cube();
    let flat: Vec<i32> = want.iter().copied().collect();
    let wide: Vec<f64> = flat.iter().map(|&v| f64::from(v)).collect();
    for threads in [1, 2, 3, 8, 0] {
        hdf5::set_num_threads(threads);
        let f = file();
        for name in ["tall", "odd", "cols", "tiny"] {
            for name in [name.to_string(), format!("{name}_raw")] {
                let d = f.dataset(&name).unwrap();
                assert_eq!(d.read_raw::<i32>().unwrap(), flat, "{name} x{threads}");
                assert_eq!(d.read_raw::<f64>().unwrap(), wide, "{name} f64 x{threads}");
                let rows = d.read_rows_raw::<i32>(11, 40).unwrap();
                let want_rows: Vec<i32> = want.slice(s![11..40, .., ..]).iter().copied().collect();
                assert_eq!(rows, want_rows, "{name} rows x{threads}");
                let sel: ArrayD<i32> = d.read_slice(s![5..90;3, 2..11, 1..6]).unwrap();
                let want_sel = want.slice(s![5..90;3, 2..11, 1..6]).to_owned().into_dyn();
                assert_eq!(sel, want_sel, "{name} slice x{threads}");
            }
        }
    }
    hdf5::set_num_threads(0);
}

#[test]
fn written_in_parallel_reads_back_serially() {
    let data = ArrayD::from_shape_fn(IxDyn(&[300, 41]), |ix| (ix[0] * 41 + ix[1]) as u16);
    let bytes = |threads| {
        hdf5::set_num_threads(threads);
        let f = File::create_in_memory().unwrap();
        f.new_dataset_builder()
            .chunk((16, 8))
            .shuffle()
            .deflate(6)
            .fletcher32()
            .with_data(&data)
            .create("x")
            .unwrap();
        f.to_bytes().unwrap()
    };
    let (one, many) = (bytes(1), bytes(8));
    hdf5::set_num_threads(0);
    // same layout, chunk for chunk (object times may differ, sizes cannot)
    assert_eq!(one.len(), many.len());
    for file in [one, many] {
        let back: ArrayD<u16> = File::from_bytes(file)
            .unwrap()
            .dataset("x")
            .unwrap()
            .read()
            .unwrap();
        assert_eq!(back, data);
    }
}
