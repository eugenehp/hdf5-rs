//! The browser target, `wasm32-unknown-unknown`: no file system, no clock, no
//! threads — files live in memory (`create_in_memory` / `to_bytes` /
//! `from_bytes`). Run with
//!
//! ```text
//! CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
//!     cargo test -p hdf5-no-c --target wasm32-unknown-unknown --test wasm
//! ```
//!
//! (add `RUSTFLAGS="-C target-feature=+simd128"` for the SIMD kernels). On
//! other targets the same checks run as ordinary tests.

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
use wasm_bindgen_test::wasm_bindgen_test as test;

use hdf5::types::VarLenUnicode;
use hdf5::File;

fn signal(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.01).sin() * 1000.0).collect()
}

/// A file with every filter the browser is likely to meet, as bytes.
fn written() -> Vec<u8> {
    let f = File::create_in_memory().unwrap();
    let data = ndarray::Array2::from_shape_vec((4096, 6), signal(4096 * 6)).unwrap();
    f.new_dataset_builder()
        .chunk((512, 6))
        .shuffle()
        .deflate(4)
        .fletcher32()
        .with_data(&data)
        .create("gzip")
        .unwrap();
    f.new_dataset_builder()
        .chunk((1000, 1))
        .shuffle()
        .lzf()
        .with_data(&data)
        .create("lzf_cols")
        .unwrap();
    f.new_dataset_builder()
        .with_data(&data.mapv(|v| v as i16))
        .create("contig_i16")
        .unwrap();
    let g = f.create_group("meta").unwrap();
    let names: Vec<VarLenUnicode> = ["α", "beta", "γάμμα"]
        .iter()
        .map(|s| s.parse().unwrap())
        .collect();
    g.new_dataset_builder()
        .with_data(&names)
        .create("names")
        .unwrap();
    g.new_attr::<u32>()
        .create("version")
        .unwrap()
        .write_scalar(&7)
        .unwrap();
    f.to_bytes().unwrap()
}

#[test]
fn round_trip_in_memory() {
    let want = signal(4096 * 6);
    let f = File::from_bytes(written()).unwrap();
    assert_eq!(f.dataset("gzip").unwrap().read_raw::<f32>().unwrap(), want);
    assert_eq!(
        f.dataset("lzf_cols").unwrap().read_raw::<f32>().unwrap(),
        want
    );
    let ints: Vec<i16> = want.iter().map(|&v| v as i16).collect();
    let contig = f.dataset("contig_i16").unwrap();
    assert_eq!(contig.read_raw::<i16>().unwrap(), ints);
    // converted on read
    let wide: Vec<f64> = ints.iter().map(|&v| f64::from(v)).collect();
    assert_eq!(contig.read_raw::<f64>().unwrap(), wide);
    let rows = f
        .dataset("gzip")
        .unwrap()
        .read_rows_raw::<f32>(1000, 1010)
        .unwrap();
    assert_eq!(rows, want[6000..6060]);
    let names = f
        .dataset("meta/names")
        .unwrap()
        .read_raw::<VarLenUnicode>()
        .unwrap();
    assert_eq!(names[2].as_str(), "γάμμα");
    let version: u32 = f
        .group("meta")
        .unwrap()
        .attr("version")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(version, 7);
}

#[test]
fn reader_and_edit_in_memory() {
    let bytes = written();
    let f = File::open_reader(std::io::Cursor::new(bytes.clone())).unwrap();
    assert_eq!(
        f.member_names().unwrap(),
        ["contig_i16", "gzip", "lzf_cols", "meta"]
    );
    let g = File::from_bytes_rw(bytes).unwrap();
    g.new_dataset_builder()
        .with_data(&[1u8, 2, 3])
        .create("added")
        .unwrap();
    let h = File::from_bytes(g.to_bytes().unwrap()).unwrap();
    assert_eq!(
        h.dataset("added").unwrap().read_raw::<u8>().unwrap(),
        [1, 2, 3]
    );
    assert_eq!(
        h.dataset("gzip").unwrap().read_raw::<f32>().unwrap(),
        signal(4096 * 6)
    );
}

#[test]
fn no_file_system_is_an_error_not_a_crash() {
    // on the browser target std::fs is unsupported: this must fail cleanly
    let missing = File::open("/definitely/not/here.h5");
    assert!(missing.is_err());
}
