//! A tour of the crate's main capabilities: chunking + compression,
//! resizable datasets, hyperslab slicing, links, and variable-length types.
//!
//! ```bash
//! cargo run --example tour
//! ```
use hdf5::types::VarLenUnicode;
use hdf5::File;
use ndarray::{s, Array1, Array2};

fn main() -> hdf5::Result<()> {
    let path = std::env::temp_dir().join("tour.h5");
    let file = File::create(&path)?;

    // 1. chunked + compressed dataset (deflate + shuffle, like h5py gzip)
    let data = Array2::from_shape_fn((100, 50), |(i, j)| (i * 50 + j) as i32);
    file.new_dataset_builder()
        .with_data(&data)
        .chunk((25, 50))
        .deflate(6)
        .shuffle()
        .create("measurements")?;

    // 2. resizable (unlimited) dataset: append rows over time
    let log = file
        .new_dataset::<f64>()
        .shape((0.., 4)) // 0 rows now, unlimited growth
        .chunk((64, 4))
        .create("log")?;
    for step in 0..3 {
        let row = Array1::from_shape_fn(4, |c| (step * 4 + c) as f64);
        log.resize((step + 1, 4))?;
        log.write_slice(&row, s![step..step + 1, ..])?;
    }

    // 3. hyperslab reads: every other row of a block
    let ds = file.dataset("measurements")?;
    let block: Array2<i32> = ds.read_slice_2d(s![10..20;2, 0..5])?;
    assert_eq!(block.shape(), [5, 5]);

    // 4. soft links and iteration
    file.link_soft("/measurements", "/alias")?;
    let names = file.member_names()?;

    // 5. variable-length strings
    let words: Vec<VarLenUnicode> = ["pure", "rust", "hdf5"]
        .iter()
        .map(|s| s.parse().unwrap())
        .collect();
    file.new_dataset_builder()
        .with_data(&words)
        .create("words")?;
    file.close()?;

    // read everything back
    let file = File::open(&path)?;
    let logged: Array2<f64> = file.dataset("log")?.read_2d()?;
    let via_alias: Array2<i32> = file.dataset("alias")?.read_2d()?;
    let words: Vec<VarLenUnicode> = file.dataset("words")?.read_raw()?;

    println!("members        = {names:?}");
    println!("log rows       = {}", logged.nrows());
    println!("alias[99][49]  = {}", via_alias[[99, 49]]);
    println!("words          = {words:?}");
    assert_eq!(logged.nrows(), 3);
    assert_eq!(via_alias[[99, 49]], 4999);
    Ok(())
}
