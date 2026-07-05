//! Regression example: userblock write + content preservation on rewrite.
use hdf5::File;

fn main() -> hdf5::Result<()> {
    let path = std::env::temp_dir().join("userblock_demo.h5");

    // create with a 512-byte userblock via the fcpl
    let mut b = File::with_options();
    b.fcpl().userblock(512);
    let f = b.create(&path)?;
    f.new_dataset_builder()
        .with_data(&ndarray::Array1::from_iter(0..7i32))
        .create("x")?;
    f.close()?;

    // stamp user content into the block (e.g. a MATLAB-style header)
    {
        use std::io::{Seek, Write};
        let mut fh = std::fs::OpenOptions::new().write(true).open(&path)?;
        fh.seek(std::io::SeekFrom::Start(0))?;
        fh.write_all(b"MY-CUSTOM-HEADER v1")?;
    }

    // reopen + rewrite: the userblock content must survive
    let f = File::open_rw(&path)?;
    assert_eq!(f.userblock(), 512);
    f.new_dataset_builder()
        .with_data(&ndarray::arr1(&[9i32]))
        .create("y")?;
    f.close()?;

    let bytes = std::fs::read(&path)?;
    assert!(
        bytes.starts_with(b"MY-CUSTOM-HEADER v1"),
        "userblock content lost"
    );
    let f = File::open(&path)?;
    assert_eq!(
        f.dataset("x")?.read_raw::<i32>()?,
        (0..7).collect::<Vec<_>>()
    );
    assert_eq!(f.dataset("y")?.read_scalar::<i32>()?, 9);
    println!("userblock write + content preservation OK");
    Ok(())
}
