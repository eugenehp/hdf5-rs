//! Record more data than you want in memory: `streaming_write <out.h5>`.
//!
//! The dataset grows chunk by chunk; each `write_slice` goes straight to the
//! file (only the chunks it touches are held, briefly), and each `flush`
//! appends the new metadata rather than rewriting the file.
use ndarray::{s, Array2};

fn main() -> hdf5::Result<()> {
    let path = std::env::args().nth(1).expect("an output path");
    let (blocks, rows, channels) = (200, 2048, 32);
    let f = hdf5::File::create(&path)?;
    let ds = f
        .new_dataset::<i16>()
        .shape((0.., channels))
        .chunk((rows, channels))
        .shuffle()
        .deflate(1)
        .create("ephys")?;
    f.flush()?;
    for b in 0..blocks {
        let data = Array2::from_shape_fn((rows, channels), |(i, c)| {
            (((b * rows + i) as f32 * 0.01 + c as f32).sin() * 1000.0) as i16
        });
        ds.resize(((b + 1) * rows, channels))?;
        ds.write_slice(&data, s![b * rows..(b + 1) * rows, ..])?;
        if b % 50 == 49 {
            f.flush()?;
        }
    }
    drop(f);
    // and edit it in place: one chunk rewritten, a dataset added
    let f = hdf5::File::open_rw(&path)?;
    f.dataset("ephys")?
        .write_slice(&Array2::<i16>::zeros((10, channels)), s![5000..5010, ..])?;
    f.new_dataset_builder()
        .with_data(&[1u32, 2, 3])
        .create("events")?;
    println!("{} rows x {channels} channels written", blocks * rows);
    Ok(())
}
