use hdf5::File;

fn main() -> hdf5::Result<()> {
    let d = "/private/tmp/claude-501/-Users-Shared-hdf5-rs/0ebd41ce-ec10-4ac9-9239-8cf4acc80aa8/scratchpad";

    // 1. Read h5py-written external links (cross-file follow)
    let f = File::open(format!("{d}/ext_source.h5"))?;
    let v: Vec<i32> = f.dataset("ext")?.read_raw()?;
    assert_eq!(v, vec![0, 1, 2, 3, 4]);
    assert_eq!(f.dataset("own")?.read_scalar::<i32>()?, 1);
    println!("read h5py external link OK");

    // 2. Write external links from Rust (compact link-message group)
    {
        let t = File::create(format!("{d}/rust_ext_target.h5"))?;
        t.new_dataset_builder()
            .with_data(&ndarray::arr1(&[10i64, 20, 30]))
            .create("data")?;
        t.close()?;
        let s = File::create(format!("{d}/rust_ext_source.h5"))?;
        s.link_external("rust_ext_target.h5", "/data", "borrowed")?;
        s.new_dataset_builder()
            .with_data(&ndarray::arr1(&[1i16]))
            .create("mine")?;
        s.create_group("sub")?
            .link_external("rust_ext_target.h5", "/data", "nested")?;
        s.close()?;
    }
    // 3. Follow our own external links back
    let f = File::open(format!("{d}/rust_ext_source.h5"))?;
    let v: Vec<i64> = f.dataset("borrowed")?.read_raw()?;
    assert_eq!(v, vec![10, 20, 30]);
    let v2: Vec<i64> = f.dataset("sub/nested")?.read_raw()?;
    assert_eq!(v2, vec![10, 20, 30]);
    println!("rust external link roundtrip OK");
    Ok(())
}
