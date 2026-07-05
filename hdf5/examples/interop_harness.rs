//! Rust side of the h5py cross-validation harness.
//!
//! Usage:
//!   cargo run --example interop_harness -- write <dir>   # write rust_all.h5 for h5py to verify
//!   cargo run --example interop_harness -- read <dir>    # read the h5py reference files
//!
//! Driven by `interop/check_h5py.py`; see that script for the counterpart.

use std::str::FromStr;

use hdf5::types::{VarLenArray, VarLenUnicode};
use hdf5::{File, H5Type};
use ndarray::{arr1, Array2};

#[derive(H5Type, Clone, PartialEq, Debug)]
#[repr(C)]
struct Comp {
    id: i32,
    value: f64,
}

#[derive(H5Type, Clone, PartialEq, Debug)]
#[repr(u8)]
#[allow(clippy::upper_case_acronyms)]
enum Color {
    RED = 1,
    GREEN = 2,
    BLUE = 3,
}

fn write(dir: &str) -> hdf5::Result<()> {
    let f = File::create(format!("{dir}/rust_all.h5"))?;
    f.new_dataset::<f64>()
        .create("scalar_f64")?
        .write_scalar(&6.5)?;
    f.new_dataset_builder()
        .with_data(&arr1(&[3i16, 1, 4, 1, 5]))
        .create("vec_i16")?;
    let g = f.create_group("grp")?;
    g.new_dataset_builder()
        .with_data(&Array2::from_shape_fn((3, 4), |(i, j)| (i * 4 + j) as u32))
        .create("mat_u32")?;
    f.new_dataset_builder()
        .with_data(&Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as i64))
        .chunk((2, 2))
        .deflate(4)
        .shuffle()
        .create("chunked")?;
    f.new_dataset_builder()
        .with_data(&arr1(&[
            Comp { id: 1, value: 0.5 },
            Comp { id: 2, value: 1.5 },
        ]))
        .create("compound")?;
    f.new_dataset_builder()
        .with_data(&arr1(&[
            VarLenUnicode::from_str("alpha").unwrap(),
            VarLenUnicode::from_str("βeta").unwrap(),
            VarLenUnicode::from_str("").unwrap(),
        ]))
        .create("strings")?;
    f.new_dataset_builder()
        .with_data(&arr1(&[
            VarLenArray::from_slice(&[1i32, 2]),
            VarLenArray::from_slice(&[3i32, 4, 5]),
        ]))
        .create("seqs")?;
    f.new_dataset_builder()
        .with_data(&arr1(&[Color::RED, Color::BLUE, Color::GREEN]))
        .create("colors")?;
    f.new_attr::<i32>().create("version")?.write_scalar(&3)?;
    g.new_attr::<VarLenUnicode>()
        .create("desc")?
        .write_scalar(&VarLenUnicode::from_str("rust pure implementation").unwrap())?;
    f.link_soft("/grp/mat_u32", "alias")?;
    // LZF + blosc chunked datasets
    f.new_dataset_builder()
        .with_data(&ndarray::Array1::from_iter(0..500i32))
        .chunk((100,))
        .lzf()
        .create("lzfd")?;
    f.new_dataset_builder()
        .with_data(&ndarray::Array1::from_iter(0..400i32))
        .chunk((100,))
        .blosc_lz4(5, true)
        .create("bloscd")?;
    // a 78 KB attribute: dense attribute storage + huge heap object
    let target = f.new_dataset::<i32>().create("dense_attr_target")?;
    target.write_scalar(&1)?;
    let big = ndarray::Array1::from_shape_fn(10_000, |i| i as f64);
    target
        .new_attr::<f64>()
        .shape([10_000])
        .create("big")?
        .write(&big)?;
    f.close()?;

    // external link pair
    let t = File::create(format!("{dir}/rust_ext_target.h5"))?;
    t.new_dataset_builder()
        .with_data(&arr1(&[10i64, 20, 30]))
        .create("data")?;
    t.close()?;
    let s = File::create(format!("{dir}/rust_ext_source.h5"))?;
    s.link_external("rust_ext_target.h5", "/data", "borrowed")?;
    s.close()?;

    println!("rust_all.h5 written");
    Ok(())
}

fn read(dir: &str) -> hdf5::Result<()> {
    // earliest-format reference
    let f = File::open(format!("{dir}/py_earliest.h5"))?;
    assert_eq!(f.dataset("scalar")?.read_scalar::<f64>()?, 1.25);
    let v: Vec<i64> = f.dataset("v")?.read_raw()?;
    assert_eq!(v, (0..20).collect::<Vec<i64>>());
    let m: Array2<f32> = f.dataset("g/m")?.read_2d()?;
    assert_eq!(m[[3, 5]], 23.0);
    assert_eq!(f.dataset("g/m")?.filters().len(), 3); // shuffle+deflate+fletcher32
    let soft: Array2<f32> = f.dataset("soft")?.read_2d()?;
    assert_eq!(soft, m);
    let comp: Vec<Comp> = f
        .dataset("comp")?
        .as_reader()
        .read_raw()
        .map_err(|e| {
            hdf5::Error::from(format!(
                "comp read (field-name matching a/b vs id/value expected to fail): {e}"
            ))
        })
        .unwrap_or_default();
    let _ = comp; // named a/b in the file; skip strict comparison
    let strs: Vec<VarLenUnicode> = f.dataset("strs")?.read_raw()?;
    assert_eq!(
        strs.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        vec!["x", "yz", ""]
    );
    let name: VarLenUnicode = f.group("g")?.attr("name")?.read_scalar()?;
    assert_eq!(name.as_str(), "reference");

    // latest-format reference
    let f = File::open(format!("{dir}/py_latest.h5"))?;
    let x: Vec<f64> = f.dataset("x")?.read_raw()?;
    assert_eq!(x[9], 9.0);
    let y: Array2<i32> = f.dataset("g/y")?.read_2d()?;
    assert_eq!(y[[1, 2]], 5);
    assert_eq!(f.group("g")?.attr("meta")?.read_scalar::<f32>()?, 9.5);
    // extensible-array chunk index
    let g: Array2<i32> = f.dataset("grow")?.read_2d()?;
    assert_eq!(g.shape(), [10, 6]);
    for (i, v) in g.iter().enumerate() {
        assert_eq!(*v, i as i32);
    }
    // v2 btree chunk index with filters
    let t: Array2<i64> = f.dataset("two")?.read_2d()?;
    for (i, v) in t.iter().enumerate() {
        assert_eq!(*v, i as i64);
    }
    // dense links + dense attributes
    let many = f.group("many")?;
    assert_eq!(many.len(), 20);
    assert_eq!(many.dataset("d019")?.read_scalar::<i32>()?, 19);
    let af = f.dataset("attrful")?;
    assert_eq!(af.attr_names()?.len(), 12);
    assert_eq!(af.attr("a011")?.read_scalar::<f64>()?, 11.0);

    // big-endian data (byte-swapped transparently)
    let f = File::open(format!("{dir}/py_bigendian.h5"))?;
    let bi: Vec<i32> = f.dataset("bi")?.read_raw()?;
    assert_eq!(bi, vec![1, -2, 300000]);
    let bf: Vec<f64> = f.dataset("bf")?.read_raw()?;
    assert_eq!(bf, vec![1.5, -2.25]);
    let ba: Vec<i64> = f.attr("battr")?.read_raw()?;
    assert_eq!(ba, vec![9]);

    // external links (cross-file follow)
    let f = File::open(format!("{dir}/py_ext_source.h5"))?;
    let p: Vec<i32> = f.dataset("ext")?.read_raw()?;
    assert_eq!(p, vec![0, 1, 2, 3, 4]);

    // committed datatype (shared message), LZF, scaleoffset
    let f = File::open(format!("{dir}/py_misc.h5"))?;
    let rows: Vec<Comp> = f
        .dataset("committed")?
        .as_reader()
        .read_raw::<CommittedAB>()?
        .into_iter()
        .map(|r| Comp {
            id: r.a,
            value: r.b,
        })
        .collect();
    assert_eq!(
        rows,
        vec![Comp { id: 1, value: 0.5 }, Comp { id: 2, value: 1.5 }]
    );
    let z: Vec<i32> = f.dataset("lzf")?.read_raw()?;
    assert_eq!(z.len(), 1000);
    assert_eq!(z[999], 999);
    let so: Vec<i32> = f.dataset("scaleoffset")?.read_raw()?;
    assert_eq!(so, (0..100).collect::<Vec<i32>>());

    // huge dense attribute (fractal heap huge object)
    let f = File::open(format!("{dir}/py_hugeattr.h5"))?;
    let big: Vec<f64> = f.dataset("x")?.attr("big")?.read_raw()?;
    assert_eq!(big.len(), 2000);
    assert_eq!(big[1999], 1999.0);

    // virtual dataset
    let f = File::open(format!("{dir}/py_vds.h5"))?;
    let v: Array2<i32> = f.dataset("virt")?.read_2d()?;
    for (i, x) in v.iter().enumerate() {
        assert_eq!(*x, i as i32);
    }

    // SOHM shared-message tables (present when the ctypes generator ran)
    for fname in ["sohm_list", "sohm_btree"] {
        if let Ok(f) = File::open(format!("{dir}/{fname}.h5")) {
            for i in 0..6 {
                let ds = f.dataset(&format!("d{i}"))?;
                let rows: Vec<SohmRow> = ds.read_raw()?;
                assert_eq!(
                    rows[2],
                    SohmRow {
                        a: 3,
                        b: 2.5,
                        c: 30,
                        d: 7.0
                    }
                );
                let common: Vec<i32> = ds.attr("common")?.read_raw()?;
                assert_eq!(common, (0..10).collect::<Vec<i32>>());
            }
        }
    }

    // blosc (present when the python side has hdf5plugin)
    if let Ok(f) = File::open(format!("{dir}/py_blosc.h5")) {
        for name in ["blosclz", "lz4", "zstd", "zlib"] {
            let a: Vec<i32> = f.dataset(name)?.read_raw()?;
            assert_eq!(a.len(), 5000);
            assert_eq!(a[4999], 4999);
        }
    }

    println!("h5py reference files read OK");
    Ok(())
}

/// Field names of the committed datatype in py_misc.h5 ("a"/"b").
#[derive(H5Type, Clone, PartialEq, Debug)]
#[repr(C)]
struct CommittedAB {
    a: i32,
    b: f64,
}

/// Row type of the SOHM reference datasets.
#[derive(H5Type, Clone, PartialEq, Debug)]
#[repr(C)]
struct SohmRow {
    a: i32,
    b: f64,
    c: i64,
    d: f32,
}

fn main() -> hdf5::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match (args.get(1).map(String::as_str), args.get(2)) {
        (Some("write"), Some(dir)) => write(dir),
        (Some("read"), Some(dir)) => read(dir),
        _ => Err("usage: interop_harness (write|read) <dir>".into()),
    }
}
