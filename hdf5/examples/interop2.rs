use hdf5::File;

fn main() -> hdf5::Result<()> {
    // 1. Read the h5py-generated reference files
    let f = File::open("/private/tmp/claude-501/-Users-Shared-hdf5-rs/0ebd41ce-ec10-4ac9-9239-8cf4acc80aa8/scratchpad/ref_simple.h5")?;
    let s: i32 = f.dataset("scalar_i32")?.read_scalar()?;
    assert_eq!(s, 42);
    let v: ndarray::Array1<f64> = f.dataset("vec_f64")?.read_1d()?;
    assert_eq!(v.as_slice().unwrap(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let m: ndarray::Array2<i32> = f.dataset("grp/mat_i32")?.read_2d()?;
    assert_eq!(m[[2, 3]], 11);
    let ra: i32 = f.attr("root_attr")?.read_scalar()?;
    assert_eq!(ra, 7);
    println!("read h5py ref_simple.h5 OK");

    // 2. Read the chunked+gzip+shuffle file
    let f = File::open("/private/tmp/claude-501/-Users-Shared-hdf5-rs/0ebd41ce-ec10-4ac9-9239-8cf4acc80aa8/scratchpad/chk.h5")?;
    let c: ndarray::Array2<i32> = f.dataset("c")?.read_2d()?;
    assert_eq!(c.shape(), [6, 4]);
    for (i, x) in c.iter().enumerate() {
        assert_eq!(*x, i as i32);
    }
    assert!(f.dataset("c")?.is_chunked());
    assert_eq!(f.dataset("c")?.chunk(), Some(vec![2, 2]));
    println!("read h5py chunked+gzip+shuffle OK");

    // 3. Read the vlen file
    let f = File::open("/private/tmp/claude-501/-Users-Shared-hdf5-rs/0ebd41ce-ec10-4ac9-9239-8cf4acc80aa8/scratchpad/vl.h5")?;
    let s: Vec<hdf5::types::VarLenUnicode> = f.dataset("s")?.read_raw()?;
    assert_eq!(s[0].as_str(), "hi");
    assert_eq!(s[1].as_str(), "worldly");
    let v: Vec<hdf5::types::VarLenArray<i32>> = f.dataset("v")?.read_raw()?;
    assert_eq!(v[0].as_slice(), &[1, 2, 3]);
    assert_eq!(v[1].as_slice(), &[4]);
    println!("read h5py vlen strings + sequences OK");

    // 4. Write chunked+deflate+shuffle, resize, slices
    let path = "/private/tmp/claude-501/-Users-Shared-hdf5-rs/0ebd41ce-ec10-4ac9-9239-8cf4acc80aa8/scratchpad/rust_chunked.h5";
    {
        let f = File::create(path)?;
        let ds = f
            .new_dataset::<f32>()
            .chunk((1, 5, 5))
            .shape((1.., 5, 5))
            .deflate(3)
            .shuffle()
            .create("var")?;
        let arr = ndarray::Array2::from_shape_fn((5, 5), |(j, i)| (10 * j + i) as f32);
        ds.write_slice(&arr, (0, .., ..))?;
        ds.resize((3, 5, 5))?;
        ds.write_slice(&(&arr * 2.0), (2, .., ..))?;
        f.close()?;
    }
    let f = File::open(path)?;
    let ds = f.dataset("var")?;
    assert_eq!(ds.shape(), vec![3, 5, 5]);
    let back: ndarray::Array2<f32> = ds.read_slice((2, .., ..))?;
    assert_eq!(back[[4, 4]], 88.0);
    println!("rust chunked+resize+slice roundtrip OK");
    Ok(())
}
