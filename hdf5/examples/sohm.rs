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
    let d = "/private/tmp/claude-501/-Users-Shared-hdf5-rs/0ebd41ce-ec10-4ac9-9239-8cf4acc80aa8/scratchpad";
    for fname in ["sohm1", "sohm2"] {
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
