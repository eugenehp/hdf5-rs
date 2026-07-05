//! End-to-end RLX workflow: store MLP weights in an HDF5 file, load them
//! back as tensors, build the forward graph, and run it on the host.
//!
//! ```bash
//! cargo run --features rlx-eval --example rlx_mlp
//! ```
//!
//! The weights file is ordinary HDF5 — h5py, netCDF tooling and this crate
//! all read it — so checkpoints written by Python training code (h5py /
//! Keras-style layouts) feed straight into RLX inference in Rust.
use hdf5::File;
use ndarray::Array2;

fn main() -> hdf5::Result<()> {
    let path = std::env::temp_dir().join("rlx_mlp.h5");

    // -- "training" side: persist a tiny 4 -> 8 -> 2 MLP -----------------
    {
        let f = File::create(&path)?;
        let model = f.create_group("model")?;
        // deterministic pseudo-weights; any numeric dtype works (they load
        // as f32), and chunked+compressed layouts are fine too
        let w1 = Array2::from_shape_fn((4, 8), |(i, j)| ((i * 8 + j) as f32 * 0.1).sin());
        let w2 = Array2::from_shape_fn((8, 2), |(i, j)| ((i * 2 + j) as f32 * 0.2).cos());
        model
            .new_dataset_builder()
            .with_data(&w1)
            .deflate(4)
            .chunk((4, 8))
            .create("w1")?;
        model.new_dataset_builder().with_data(&w2).create("w2")?;
        model
            .new_dataset_builder()
            .with_data(&ndarray::arr1(&[0.1f32, -0.1]))
            .create("b2")?;
        f.close()?;
    }

    // -- inference side: HDF5 -> RLX tensors -> forward pass -------------
    let f = File::open(&path)?;
    let model = f.group("model")?;
    let w1 = model.dataset("w1")?.read_tensor()?;
    let w2 = model.dataset("w2")?.read_tensor()?;
    let b2 = model.dataset("b2")?.read_tensor()?;
    println!(
        "loaded w1 {:?}, w2 {:?}, b2 {:?}",
        w1.dims(),
        w2.dims(),
        b2.dims()
    );

    // build the graph symbolically; nothing runs until materialization
    let x = rlx::tensor::Tensor::from_vec(vec![1.0, 0.5, -0.5, 2.0], [1, 4]);
    let y = &x.matmul(&w1).relu().matmul(&w2) + &b2;

    // with `rlx-eval`, to_vec() compiles + runs the fused graph on the host
    let out = y.to_vec();
    println!("mlp([1, 0.5, -0.5, 2]) = {out:?}");
    assert_eq!(out.len(), 2);
    assert!(out.iter().all(|v| v.is_finite()));
    println!("rlx forward pass OK");
    Ok(())
}
