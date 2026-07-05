use hdf5::File;
fn main() -> hdf5::Result<()> {
    let f = File::open("/private/tmp/claude-501/-Users-Shared-hdf5-rs/0ebd41ce-ec10-4ac9-9239-8cf4acc80aa8/scratchpad/latest.h5")?;
    let x: ndarray::Array1<f64> = f.dataset("x")?.read_1d()?;
    assert_eq!(x[9], 9.0);
    let y: ndarray::Array2<i32> = f.dataset("g/y")?.read_2d()?;
    assert_eq!(y[[1, 2]], 5);
    let meta: f32 = f.group("g")?.attr("meta")?.read_scalar()?;
    assert_eq!(meta, 9.5);
    let ys: ndarray::Array2<i32> = f.dataset("softy")?.read_2d()?;
    assert_eq!(ys, y);
    println!("read h5py libver=latest (OHDR v2 + link messages) OK");
    Ok(())
}
