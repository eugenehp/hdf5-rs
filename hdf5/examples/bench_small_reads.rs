//! Many small partial reads — the per-read overhead of parallelism (thread
//! start-up, on wasm a Worker per thread) dominates here. In memory, so it
//! runs on every target: `bench_small_reads [threads]`.
use std::time::Instant;

fn main() {
    if let Some(n) = std::env::args().nth(1).and_then(|a| a.parse().ok()) {
        hdf5::set_num_threads(n);
    }
    let f = hdf5::File::create_in_memory().unwrap();
    let data = ndarray::Array2::from_shape_fn((200_000, 16), |(i, j)| (i * 16 + j) as f32);
    f.new_dataset_builder()
        .chunk((1000, 16))
        .shuffle()
        .deflate(4)
        .with_data(&data)
        .create("x")
        .unwrap();
    let f = hdf5::File::from_bytes(f.to_bytes().unwrap()).unwrap();
    let d = f.dataset("x").unwrap();
    let t = Instant::now();
    let mut sum = 0f32;
    for i in 0..3000 {
        let r = (i * 6151) % 199_990;
        let rows = d
            .read_slice_2d::<f32, _>(ndarray::s![r..r + 10, ..])
            .unwrap();
        sum += rows[[0, 0]];
    }
    println!(
        "3000 reads of 10 rows: {:.1} ms ({} threads, checksum {sum})",
        t.elapsed().as_secs_f64() * 1e3,
        hdf5::num_threads()
    );
}
