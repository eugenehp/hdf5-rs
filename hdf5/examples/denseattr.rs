//! Dense attribute storage: attributes above the 64 KB compact limit
//! (fractal heap + v2 B-tree, v2 object header).
use hdf5::File;
use ndarray::Array1;

fn main() -> hdf5::Result<()> {
    let d = std::env::temp_dir().display().to_string();
    let path = format!("{d}/rust_denseattr.h5");
    {
        let f = File::create(&path)?;
        let ds = f.new_dataset::<i32>().create("x")?;
        ds.write_scalar(&7)?;
        // 100 KB attribute -> forces dense storage + huge heap object
        let big = Array1::from_shape_fn(12_800, |i| i as f64);
        ds.new_attr::<f64>()
            .shape([12_800])
            .create("big")?
            .write(&big)?;
        // several ordinary attributes alongside (managed objects)
        for i in 0..6 {
            ds.new_attr::<i32>()
                .create(format!("small{i}").as_str())?
                .write_scalar(&(i as i32))?;
        }
        // one mid-size (managed, but needs a larger direct block row)
        let mid = Array1::from_shape_fn(400, |i| i as f32);
        ds.new_attr::<f32>()
            .shape([400])
            .create("mid")?
            .write(&mid)?;
        f.close()?;
    }
    // read back with our own reader
    let f = File::open(&path)?;
    let ds = f.dataset("x")?;
    assert_eq!(ds.attr_names()?.len(), 8);
    let big: Vec<f64> = ds.attr("big")?.read_raw()?;
    assert_eq!(big.len(), 12_800);
    assert_eq!(big[12_799], 12_799.0);
    let mid: Vec<f32> = ds.attr("mid")?.read_raw()?;
    assert_eq!(mid[399], 399.0);
    assert_eq!(ds.attr("small3")?.read_scalar::<i32>()?, 3);
    println!("rust dense-attr (100KB attr) roundtrip OK");
    Ok(())
}
