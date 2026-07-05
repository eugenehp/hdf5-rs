fn main() -> hdf5::Result<()> {
    let d = "/private/tmp/claude-501/-Users-Shared-hdf5-rs/0ebd41ce-ec10-4ac9-9239-8cf4acc80aa8/scratchpad";
    // write with userblock via fcpl
    let mut b = hdf5::File::with_options();
    b.fcpl().userblock(512);
    let f = b.create(format!("{d}/rust_ub.h5"))?;
    f.new_dataset_builder()
        .with_data(&ndarray::Array1::from_iter(0..7i32))
        .create("x")?;
    f.close()?;
    // reopen h5py's userblock file, rewrite, confirm preservation
    let f = hdf5::File::open_rw(format!("{d}/ub.h5"))?;
    let x: Vec<i32> = f.dataset("x")?.read_raw()?;
    assert_eq!(x, vec![0, 1, 2, 3, 4]);
    f.new_dataset_builder()
        .with_data(&ndarray::arr1(&[9i32]))
        .create("y")?;
    f.close()?;
    println!("rust userblock write + rewrite OK");
    Ok(())
}
