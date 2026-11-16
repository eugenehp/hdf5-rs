//! h5py-written fixture (tests/data/h5py_chunked_rows.h5, regenerate with the
//! script in the header of `chunked_rows_fixture` below):
//! - variable-length strings in a CHUNKED (resizable) dataset read back intact
//!   (they decoded as "" while chunk bytes were deferred);
//! - `read_rows_raw` on gzip data whose chunks span several rows (4×2×5 over an
//!   11×3×5 array: partial row ranges, a partial last chunk, a chunk split on a
//!   trailing axis) equals the same rows of the full read.

use hdf5::types::VarLenUnicode;

fn file() -> hdf5::File {
    hdf5::File::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/h5py_chunked_rows.h5"
    ))
    .unwrap()
}

#[test]
fn chunked_vlen_strings() {
    let f = file();
    let v: Vec<String> = f
        .dataset("vlen_chunked")
        .unwrap()
        .read_raw::<VarLenUnicode>()
        .unwrap()
        .iter()
        .map(|s| s.as_str().to_string())
        .collect();
    assert_eq!(v, ["ucddb002", "ucddb002", "ab", "Ünïcødé", ""]);
}

#[test]
fn read_rows_raw_matches_full_read() {
    // Rows are read from a SEPARATE handle that is never fully read: a full
    // read materializes the dataset, after which read_rows_raw slices in place
    // and the chunk-selective path would go untested.
    let (f_full, f_rows) = (file(), file());
    for name in ["rows_gzip", "rows_plain"] {
        let full: Vec<f32> = f_full.dataset(name).unwrap().read_raw::<f32>().unwrap();
        let ds = f_rows.dataset(name).unwrap();
        let row = 3 * 5;
        for r0 in 0..=11usize {
            for r1 in r0..=11usize {
                let got = ds.read_rows_raw::<f32>(r0, r1).unwrap();
                assert_eq!(got, full[r0 * row..r1 * row], "{name} rows {r0}..{r1}");
            }
        }
        assert!(ds.read_rows_raw::<f32>(0, 12).is_err());
        assert!(
            ds.read_rows_raw::<f64>(0, 1).is_err(),
            "type must match exactly"
        );
    }
}
