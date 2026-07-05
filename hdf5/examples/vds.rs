use hdf5::File;
use ndarray::Array2;

fn main() -> hdf5::Result<()> {
    let d = "/private/tmp/claude-501/-Users-Shared-hdf5-rs/0ebd41ce-ec10-4ac9-9239-8cf4acc80aa8/scratchpad";
    let f = File::open(format!("{d}/vds.h5"))?;
    let ds = f.dataset("virt")?;
    assert_eq!(ds.layout(), hdf5::dataset::Layout::Virtual);
    let v: Array2<i32> = ds.read_2d()?;
    assert_eq!(v.shape(), [4, 10]);
    for (i, x) in v.iter().enumerate() {
        assert_eq!(*x, i as i32, "element {i}");
    }
    println!("virtual dataset (4 sources, cross-file) OK");
    Ok(())
}
