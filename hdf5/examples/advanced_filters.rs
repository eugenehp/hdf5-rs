//! Regression example: scaleoffset/nbit/bit-shuffle/zstd filters and
//! dense-link groups, written by this crate and read back.
use hdf5::filters::{Blosc, BloscShuffle, Filter, ScaleOffset};
use hdf5::File;
use ndarray::Array1;

fn main() -> hdf5::Result<()> {
    let d = std::env::temp_dir().display().to_string();
    let path = format!("{d}/rust_round6a.h5");
    {
        let f = File::create(&path)?;
        let ints = Array1::from_shape_fn(1000, |i| (i as i32) - 200);
        f.new_dataset_builder()
            .with_data(&ints)
            .chunk((250,))
            .set_filters(&[Filter::ScaleOffset(ScaleOffset::Integer(0))])
            .create("soi")?;
        let floats = Array1::from_shape_fn(500, |i| i as f64 / 7.0);
        f.new_dataset_builder()
            .with_data(&floats)
            .chunk((250,))
            .set_filters(&[Filter::ScaleOffset(ScaleOffset::FloatDScale(3))])
            .create("sof")?;
        f.new_dataset_builder()
            .with_data(&ints)
            .chunk((250,))
            .set_filters(&[Filter::NBit])
            .create("nb")?;
        f.new_dataset_builder()
            .with_data(&ints)
            .chunk((250,))
            .blosc(Blosc::LZ4, 5, BloscShuffle::Bit)
            .create("bitshuf")?;
        f.new_dataset_builder()
            .with_data(&ints)
            .chunk((250,))
            .blosc_zstd(5, true)
            .create("zstd_store")?;
        // dense links: compact group with an external link + 12 members
        let t = File::create(format!("{d}/r6_target.h5"))?;
        t.new_dataset::<i32>().create("p")?.write_scalar(&5)?;
        t.close()?;
        let g = f.create_group("dense")?;
        g.link_external("r6_target.h5", "/p", "ext")?;
        for i in 0..12 {
            g.new_dataset::<i32>()
                .create(format!("m{i:02}").as_str())?
                .write_scalar(&(i as i32))?;
        }
        f.close()?;
    }
    // our own roundtrip
    let f = File::open(&path)?;
    let soi: Vec<i32> = f.dataset("soi")?.read_raw()?;
    assert_eq!(soi[999], 799);
    let sof: Vec<f64> = f.dataset("sof")?.read_raw()?;
    assert!((sof[499] - 499.0 / 7.0).abs() < 1e-3);
    let nb: Vec<i32> = f.dataset("nb")?.read_raw()?;
    assert_eq!(nb[0], -200);
    let bs: Vec<i32> = f.dataset("bitshuf")?.read_raw()?;
    assert_eq!(bs[999], 799);
    let zs: Vec<i32> = f.dataset("zstd_store")?.read_raw()?;
    assert_eq!(zs[999], 799);
    let g = f.group("dense")?;
    assert_eq!(g.len(), 13);
    assert_eq!(g.dataset("m07")?.read_scalar::<i32>()?, 7);
    assert_eq!(g.dataset("ext")?.read_scalar::<i32>()?, 5);
    println!("rust roundtrip: scaleoffset/nbit/bitshuffle/zstd/dense-links OK");
    Ok(())
}
