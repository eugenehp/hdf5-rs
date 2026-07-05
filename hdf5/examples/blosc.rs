use hdf5::filters::{Blosc, BloscShuffle};
use hdf5::File;
use ndarray::Array2;

fn main() -> hdf5::Result<()> {
    let d = "/private/tmp/claude-501/-Users-Shared-hdf5-rs/0ebd41ce-ec10-4ac9-9239-8cf4acc80aa8/scratchpad";
    // 1. read hdf5plugin-written blosc datasets (all codecs, shuffle variants)
    let f = File::open(format!("{d}/blosc.h5"))?;
    for name in ["blosclz", "lz4", "zstd", "zlib", "noshuffle"] {
        let a: Array2<i32> = f.dataset(name)?.read_2d()?;
        for (i, v) in a.iter().enumerate() {
            assert_eq!(*v, i as i32, "{name}[{i}]");
        }
    }
    println!("read hdf5plugin blosc (blosclz/lz4/zstd/zlib, shuffle+no) OK");

    // 2. write blosc from Rust with each supported codec
    let path = format!("{d}/rust_blosc.h5");
    {
        let f = File::create(&path)?;
        let data = Array2::from_shape_fn((64, 128), |(i, j)| (i * 128 + j) as i64);
        f.new_dataset_builder()
            .with_data(&data)
            .chunk((16, 128))
            .blosc_blosclz(5, BloscShuffle::Byte)
            .create("blz")?;
        f.new_dataset_builder()
            .with_data(&data)
            .chunk((16, 128))
            .blosc_lz4(5, true)
            .create("lz4")?;
        f.new_dataset_builder()
            .with_data(&data)
            .chunk((16, 128))
            .blosc_zlib(5, BloscShuffle::None)
            .create("zl")?;
        f.new_dataset_builder()
            .with_data(&data)
            .chunk((16, 128))
            .blosc(Blosc::Snappy, 5, BloscShuffle::Byte)
            .create("sn")?;
        f.close()?;
    }
    let f = File::open(&path)?;
    for name in ["blz", "lz4", "zl", "sn"] {
        let a: Array2<i64> = f.dataset(name)?.read_2d()?;
        assert_eq!(a[[63, 127]], 8191, "{name}");
    }
    println!("rust blosc roundtrip (blosclz/lz4/zlib/snappy) OK");
    Ok(())
}
