//! Read SOHM (shared object header message) files produced by libhdf5.
use hdf5::File;

#[derive(hdf5::H5Type, Clone, PartialEq, Debug)]
#[repr(C)]
struct Row {
    a: i32,
    b: f64,
    c: i64,
    d: f32,
}

fn main() -> hdf5::Result<()> {
    // reference files are produced by `python3 interop/make_sohm.py <dir>`
    let Some(d) = std::env::args().nth(1) else {
        println!("usage: sohm <dir with sohm_list.h5/sohm_btree.h5>; skipping");
        return Ok(());
    };
    for fname in ["sohm_list", "sohm_btree"] {
        let f = File::open(format!("{d}/{fname}.h5"))?;
        for i in 0..6 {
            let ds = f.dataset(&format!("d{i}"))?;
            let rows: Vec<Row> = ds.read_raw()?;
            assert_eq!(rows.len(), 4, "{fname}/d{i}");
            assert_eq!(
                rows[2],
                Row {
                    a: 3,
                    b: 2.5,
                    c: 30,
                    d: 7.0
                },
                "{fname}/d{i}"
            );
            // shared attribute messages
            let common: Vec<i32> = ds.attr("common")?.read_raw()?;
            assert_eq!(common, (0..10).collect::<Vec<i32>>(), "{fname}/d{i} common");
            let note: Vec<f64> = ds.attr("note")?.read_raw()?;
            assert_eq!(note[7], 7.0 + i as f64, "{fname}/d{i} note");
        }
        println!("{fname}: 6 datasets via SOHM (shared dtype/dataspace/fill/attr) OK");
    }
    Ok(())
}
