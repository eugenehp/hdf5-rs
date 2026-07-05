//! Load HDF5 datasets directly as RLX tensors (feature = rlx).
//!
//! ```bash
//! cargo run --features rlx --example rlx_tensor
//! ```
use hdf5::File;
use ndarray::Array2;

fn main() -> hdf5::Result<()> {
    let path = std::env::temp_dir().join("rlx_tensor.h5");
    {
        let f = File::create(&path)?;
        let m = Array2::from_shape_fn((8, 16), |(i, j)| (i * 16 + j) as i32);
        f.new_dataset_builder()
            .with_data(&m)
            .chunk((4, 16))
            .deflate(4)
            .create("weights")?;
        f.new_dataset_builder()
            .with_data(&ndarray::arr1(&[0.5f64, 1.5, 2.5]))
            .create("bias")?;
    }
    let f = File::open(&path)?;

    // datasets load as f32 host tensors, whatever their on-disk numeric type
    let w = f.dataset("weights")?.read_tensor()?;
    let b = f.dataset("bias")?.read_tensor()?;
    println!(
        "weights tensor dims = {:?}, dtype = {:?}",
        w.dims(),
        w.dtype()
    );
    println!("bias    tensor dims = {:?}", b.dims());
    assert_eq!(w.dims(), vec![8, 16]);
    assert_eq!(b.dims(), vec![3]);

    // the tensors are symbolic RLX graph constants: ops compose lazily and
    // the whole expression fuses at materialization (RLX's cpu backend)
    let scaled = &w * 2.0f32;
    println!("composed op dims    = {:?}", scaled.dims());
    let vals = b.to_vec();
    println!("bias values         = {vals:?}");
    assert_eq!(vals, vec![0.5, 1.5, 2.5]);
    println!("rlx tensors loaded OK");
    Ok(())
}
