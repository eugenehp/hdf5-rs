use hdf5::File;
use ndarray::Array2;

#[derive(hdf5::H5Type, Clone, PartialEq, Debug)]
#[repr(C)]
struct AB {
    a: i32,
    b: f64,
}

fn main() -> hdf5::Result<()> {
    let d = "/private/tmp/claude-501/-Users-Shared-hdf5-rs/0ebd41ce-ec10-4ac9-9239-8cf4acc80aa8/scratchpad";

    // 1. Committed datatypes referenced via SHARED datatype messages
    let f = File::open(format!("{d}/committed.h5"))?;
    let rows: Vec<AB> = f.dataset("d")?.read_raw()?;
    assert_eq!(
        rows,
        vec![
            AB { a: 1, b: 0.5 },
            AB { a: 2, b: 1.5 },
            AB { a: 3, b: 2.5 }
        ]
    );
    // the committed type itself is a named datatype object
    let dt = f.group("/")?.named_datatypes()?;
    assert_eq!(dt.len(), 1);
    let f = File::open(format!("{d}/committed_latest.h5"))?;
    let v: Vec<i64> = f.dataset("dl")?.read_raw()?;
    assert_eq!(v, vec![10, 20, 30, 40]);
    println!("shared/committed datatype messages OK");

    // 2. Huge fractal-heap objects: 16KB dense attribute
    let f = File::open(format!("{d}/hugeattr.h5"))?;
    let ds = f.dataset("x")?;
    assert_eq!(ds.attr_names()?.len(), 11);
    let big: Vec<f64> = ds.attr("big")?.read_raw()?;
    assert_eq!(big.len(), 2000);
    assert_eq!(big[1999], 1999.0);
    assert_eq!(ds.attr("small7")?.read_scalar::<f64>()?, 7.0);
    println!("huge heap objects (16KB dense attr) OK");

    // 3. LZF: read h5py's file
    let f = File::open(format!("{d}/lzf.h5"))?;
    let z: Array2<i32> = f.dataset("z")?.read_2d()?;
    for (i, v) in z.iter().enumerate() {
        assert_eq!(*v, i as i32);
    }
    println!("read h5py LZF OK");

    // 4. LZF: write from Rust
    let path = format!("{d}/rust_lzf.h5");
    {
        let f = File::create(&path)?;
        let data = Array2::from_shape_fn((64, 64), |(i, j)| (i * 64 + j) as i64);
        f.new_dataset_builder()
            .with_data(&data)
            .chunk((16, 64))
            .lzf()
            .create("z")?;
        f.close()?;
    }
    let f = File::open(&path)?;
    let z: Array2<i64> = f.dataset("z")?.read_2d()?;
    assert_eq!(z[[63, 63]], 4095);
    println!("rust LZF roundtrip OK");
    Ok(())
}
