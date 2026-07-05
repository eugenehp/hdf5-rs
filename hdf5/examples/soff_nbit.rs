use hdf5::File;
use ndarray::Array2;

fn main() -> hdf5::Result<()> {
    let d = "/private/tmp/claude-501/-Users-Shared-hdf5-rs/0ebd41ce-ec10-4ac9-9239-8cf4acc80aa8/scratchpad";
    // scaleoffset: integer (auto minbits) and float D-scale
    let f = File::open(format!("{d}/scaleoffset.h5"))?;
    let soi: Array2<i32> = f.dataset("soi")?.read_2d()?;
    for (i, v) in soi.iter().enumerate() {
        assert_eq!(*v, i as i32);
    }
    let sof: Array2<f64> = f.dataset("sof")?.read_2d()?;
    for (i, v) in sof.iter().enumerate() {
        let expect = i as f64 / 63.0;
        assert!((v - expect).abs() < 1e-3, "sof[{i}]: {v} vs {expect}");
    }
    println!("scaleoffset (int + float D-scale) OK");

    // nbit: 12-bit signed ints in i32 containers
    let f = File::open(format!("{d}/nbit.h5"))?;
    let nb: Vec<i32> = f.dataset("nb")?.read_raw()?;
    let expected: Vec<i32> = (0..32).map(|i| ((i * 37) % 2048) - 1024).collect();
    assert_eq!(nb, expected);
    println!("nbit (12-bit signed) OK");
    Ok(())
}
