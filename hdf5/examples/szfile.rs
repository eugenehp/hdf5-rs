//! Write a szip-compressed file (feature = szip) for external validation.
use hdf5::filters::{Filter, SZip};
use hdf5::File;
use ndarray::Array1;

fn main() -> hdf5::Result<()> {
    let f = File::create(std::env::args().nth(1).unwrap_or_else(|| {
        std::env::temp_dir()
            .join("rust_szip.h5")
            .display()
            .to_string()
    }))?;
    let data = Array1::from_shape_fn(4000, |i| (i as i32) % 977);
    f.new_dataset_builder()
        .with_data(&data)
        .chunk((1000,))
        .set_filters(&[Filter::SZip(SZip::NearestNeighbor, 16)])
        .create("d")?;
    f.close()?;
    println!("rust szip file written");
    Ok(())
}
