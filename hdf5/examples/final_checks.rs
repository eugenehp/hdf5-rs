use hdf5::File;

fn main() -> hdf5::Result<()> {
    let d = "/private/tmp/claude-501/-Users-Shared-hdf5-rs/0ebd41ce-ec10-4ac9-9239-8cf4acc80aa8/scratchpad";
    // Tolerant parsing: file with an unsupported (reference-type) dataset
    let f = File::open(format!("{d}/mixed_support.h5"))?;
    let n: Vec<i32> = f.dataset("normal")?.read_raw()?;
    assert_eq!(n, vec![0, 1, 2, 3]);
    assert_eq!(f.dataset("also_normal")?.read_scalar::<f64>()?, 2.5);
    match f.dataset("refs") {
        Err(e) => println!("unsupported object correctly isolated: {e}"),
        Ok(_) => panic!("reference dataset should be unsupported"),
    }
    // it still shows up in listings
    assert!(f.member_names()?.contains(&"refs".to_string()));

    // mtime: create a file, check loc_info reports a plausible timestamp
    let path = format!("{d}/mtime.h5");
    {
        let f = File::create(&path)?;
        f.new_dataset::<i32>().create("d")?;
        f.close()?;
    }
    let f = File::open(&path)?;
    let info = f.dataset("d")?.loc_info()?;
    assert!(info.mtime > 1_700_000_000, "mtime = {}", info.mtime);
    println!("tolerant parse + mtime OK");
    Ok(())
}
