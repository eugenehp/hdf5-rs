//! Basic roundtrip: groups, datasets, attributes, links.
use hdf5::File;
use ndarray::arr2;

fn main() -> hdf5::Result<()> {
    let path = std::env::temp_dir().join("rust_out.h5");
    let path = path.to_str().unwrap();
    {
        let file = File::create(path)?;
        let g = file.create_group("grp")?;
        let ds = g
            .new_dataset_builder()
            .with_data(&arr2(&[[1i32, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]))
            .create("mat")?;
        ds.new_attr::<f64>().create("pi")?.write_scalar(&std::f64::consts::PI)?;
        file.new_dataset_builder()
            .with_data(&ndarray::arr1(&[1.5f64, 2.5, 3.5]))
            .create("vec")?;
        file.new_attr::<i64>().create("answer")?.write_scalar(&42)?;
        file.close()?;
    }
    // reopen with our own reader
    let f = File::open(path)?;
    let mat: ndarray::Array2<i32> = f.dataset("grp/mat")?.read_2d()?;
    assert_eq!(mat[[2, 3]], 12);
    let pi: f64 = f.dataset("grp/mat")?.attr("pi")?.read_scalar()?;
    assert!((pi - std::f64::consts::PI).abs() < 1e-12);
    let ans: i64 = f.attr("answer")?.read_scalar()?;
    assert_eq!(ans, 42);
    println!("rust roundtrip OK");
    Ok(())
}
