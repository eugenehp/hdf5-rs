use hdf5::File;
use ndarray::Array2;

fn main() -> hdf5::Result<()> {
    let d = "/private/tmp/claude-501/-Users-Shared-hdf5-rs/0ebd41ce-ec10-4ac9-9239-8cf4acc80aa8/scratchpad";

    // 1. Big-endian data, now byte-swapped transparently
    let f = File::open(format!("{d}/be2.h5"))?;
    let bi: Vec<i32> = f.dataset("bi")?.read_raw()?;
    assert_eq!(bi, vec![1, -2, 300000]);
    let bf: Vec<f64> = f.dataset("bf")?.read_raw()?;
    assert_eq!(bf, vec![1.5, -2.25]);
    let bu: Vec<u16> = f.dataset("bu")?.read_raw()?;
    assert_eq!(bu, vec![1, 65535]);
    let battr: Vec<i64> = f.attr("battr")?.read_raw()?;
    assert_eq!(battr, vec![9]);
    #[derive(hdf5::H5Type, Clone, PartialEq, Debug)]
    #[repr(C)]
    struct AB {
        a: i32,
        b: f32,
    }
    let bc: Vec<AB> = f.dataset("bcomp")?.read_raw()?;
    assert_eq!(bc, vec![AB { a: 7, b: 0.5 }, AB { a: -8, b: 1.25 }]);
    println!("big-endian swap OK");

    // 2. Extensible-array chunk index (resizable, latest format)
    let f = File::open(format!("{d}/ea.h5"))?;
    let g: Array2<i32> = f.dataset("grow")?.read_2d()?;
    assert_eq!(g.shape(), [10, 6]);
    for (i, v) in g.iter().enumerate() {
        assert_eq!(*v, i as i32);
    }
    let gz: Array2<f32> = f.dataset("growz")?.read_2d()?;
    assert_eq!(gz.shape(), [8, 4]);
    assert_eq!(gz[[7, 3]], 31.0);
    println!("extensible-array index OK");

    // 3. v2 B-tree chunk index (2 unlimited dims)
    let f = File::open(format!("{d}/bt2.h5"))?;
    let t: Array2<i64> = f.dataset("two")?.read_2d()?;
    assert_eq!(t.shape(), [6, 6]);
    for (i, v) in t.iter().enumerate() {
        assert_eq!(*v, i as i64);
    }
    let tz: Array2<i32> = f.dataset("twoz")?.read_2d()?;
    assert_eq!(tz[[3, 3]], 15);
    println!("v2 B-tree index OK");

    // 4. Dense links + dense attributes (fractal heap)
    let f = File::open(format!("{d}/dense.h5"))?;
    let many = f.group("many")?;
    assert_eq!(many.len(), 30);
    for i in [0usize, 7, 15, 29] {
        let v: i32 = many.dataset(&format!("d{i:03}"))?.read_scalar()?;
        assert_eq!(v, i as i32);
    }
    let names = many.member_names()?;
    assert_eq!(names.len(), 30);
    assert_eq!(names[0], "d000");
    let ds = f.dataset("attrful")?;
    let attr_names = ds.attr_names()?;
    assert_eq!(attr_names.len(), 20);
    for i in [0usize, 9, 19] {
        let v: f64 = ds.attr(&format!("a{i:03}"))?.read_scalar()?;
        assert_eq!(v, i as f64 / 2.0);
    }
    println!("dense links + attributes OK");
    Ok(())
}
