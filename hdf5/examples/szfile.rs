use hdf5::filters::{Filter, SZip};
use hdf5::File;
use ndarray::Array1;

fn main() -> hdf5::Result<()> {
    let f = File::create("/private/tmp/claude-501/-Users-Shared-hdf5-rs/0ebd41ce-ec10-4ac9-9239-8cf4acc80aa8/scratchpad/rust_szip.h5")?;
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
