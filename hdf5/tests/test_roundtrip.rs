//! End-to-end roundtrip tests for the pure-Rust HDF5 engine.

use std::str::FromStr;

use ndarray::{arr1, arr2, s, Array1, Array2, ArrayD};
use tempfile::tempdir;

use hdf5::types::{FixedAscii, FixedUnicode, VarLenArray, VarLenAscii, VarLenUnicode};
use hdf5::{File, H5Type};

#[derive(H5Type, Clone, PartialEq, Debug)]
#[repr(u8)]
enum Color {
    Red = 1,
    Green = 2,
    Blue = 3,
}

#[derive(H5Type, Clone, PartialEq, Debug)]
#[repr(C)]
struct Pixel {
    xy: (i64, i64),
    color: Color,
}

#[derive(H5Type, Clone, PartialEq, Debug)]
#[repr(C)]
struct Mixed {
    id: u32,
    value: f64,
    tag: FixedAscii<6>,
}

#[test]
fn scalar_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("scalar.h5");
    {
        let f = File::create(&path).unwrap();
        f.new_dataset::<i32>()
            .create("i")
            .unwrap()
            .write_scalar(&42i32)
            .unwrap();
        f.new_dataset::<f64>()
            .create("f")
            .unwrap()
            .write_scalar(&2.5f64)
            .unwrap();
        f.new_dataset::<bool>()
            .create("b")
            .unwrap()
            .write_scalar(&true)
            .unwrap();
    }
    let f = File::open(&path).unwrap();
    assert_eq!(f.dataset("i").unwrap().read_scalar::<i32>().unwrap(), 42);
    assert_eq!(f.dataset("f").unwrap().read_scalar::<f64>().unwrap(), 2.5);
    assert!(f.dataset("b").unwrap().read_scalar::<bool>().unwrap());
    assert!(f.dataset("i").unwrap().is_scalar());
    assert_eq!(f.dataset("i").unwrap().ndim(), 0);
}

#[test]
fn multidim_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("nd.h5");
    let a2 = Array2::from_shape_fn((7, 3), |(i, j)| (100 * i + j) as i64);
    let a3 = ArrayD::from_shape_fn(ndarray::IxDyn(&[2, 3, 4]), |ix| {
        (ix[0] * 12 + ix[1] * 4 + ix[2]) as f32
    });
    {
        let f = File::create(&path).unwrap();
        f.new_dataset_builder().with_data(&a2).create("a2").unwrap();
        f.new_dataset_builder()
            .with_data(&a3.view())
            .create("a3")
            .unwrap();
    }
    let f = File::open(&path).unwrap();
    assert_eq!(f.dataset("a2").unwrap().read_2d::<i64>().unwrap(), a2);
    assert_eq!(f.dataset("a3").unwrap().read_dyn::<f32>().unwrap(), a3);
    assert_eq!(f.dataset("a2").unwrap().shape(), vec![7, 3]);
    assert_eq!(f.dataset("a3").unwrap().size(), 24);
}

#[test]
fn groups_links_iteration() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("groups.h5");
    {
        let f = File::create(&path).unwrap();
        let g = f.create_group("a/b/c").unwrap();
        assert_eq!(g.name(), "/a/b/c");
        f.create_group("z").unwrap();
        g.new_dataset::<i32>().create("leaf").unwrap();
        f.link_soft("/a/b/c", "shortcut").unwrap();
        f.link_hard("/a/b/c/leaf", "hard_leaf").unwrap();
    }
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.member_names().unwrap(),
        vec!["a", "hard_leaf", "shortcut", "z"]
    );
    assert!(f.link_exists("a/b"));
    assert!(!f.link_exists("nope"));
    let g = f.group("shortcut").unwrap(); // via soft link
    assert_eq!(g.member_names().unwrap(), vec!["leaf"]);
    let d = f.dataset("hard_leaf").unwrap(); // via hard link
    assert!(d.is_scalar());
    assert_eq!(f.group("a").unwrap().groups().unwrap().len(), 1);
    assert_eq!(f.group("a/b/c").unwrap().datasets().unwrap().len(), 1);
}

#[test]
fn unlink_and_relink() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("links.h5");
    let f = File::create(&path).unwrap();
    f.create_group("g1").unwrap();
    f.new_dataset::<i32>().create("d1").unwrap();
    assert_eq!(f.len(), 2);
    f.relink("d1", "d2").unwrap();
    assert!(f.link_exists("d2"));
    assert!(!f.link_exists("d1"));
    f.unlink("d2").unwrap();
    assert_eq!(f.len(), 1);
    drop(f);
    let f = File::open(&path).unwrap();
    assert_eq!(f.member_names().unwrap(), vec!["g1"]);
}

#[test]
fn attributes_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("attrs.h5");
    {
        let f = File::create(&path).unwrap();
        let g = f.create_group("g").unwrap();
        g.new_attr::<i32>()
            .create("scalar")
            .unwrap()
            .write_scalar(&7)
            .unwrap();
        g.new_attr::<f32>()
            .shape([4])
            .create("vec")
            .unwrap()
            .write(&arr1(&[1.0f32, 2.0, 3.0, 4.0]))
            .unwrap();
        let v = VarLenUnicode::from_str("metadata ünïcode").unwrap();
        g.new_attr::<VarLenUnicode>()
            .create("note")
            .unwrap()
            .write_scalar(&v)
            .unwrap();
    }
    let f = File::open(&path).unwrap();
    let g = f.group("g").unwrap();
    assert_eq!(g.attr_names().unwrap(), vec!["scalar", "vec", "note"]);
    assert_eq!(g.attr("scalar").unwrap().read_scalar::<i32>().unwrap(), 7);
    let vec: Array1<f32> = g.attr("vec").unwrap().read_1d().unwrap();
    assert_eq!(vec, arr1(&[1.0f32, 2.0, 3.0, 4.0]));
    let note: VarLenUnicode = g.attr("note").unwrap().read_scalar().unwrap();
    assert_eq!(note.as_str(), "metadata ünïcode");
}

#[test]
fn compound_enum_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("compound.h5");
    let data = arr2(&[
        [
            Pixel {
                xy: (1, 2),
                color: Color::Red,
            },
            Pixel {
                xy: (3, 4),
                color: Color::Blue,
            },
        ],
        [
            Pixel {
                xy: (5, 6),
                color: Color::Green,
            },
            Pixel {
                xy: (7, 8),
                color: Color::Red,
            },
        ],
    ]);
    {
        let f = File::create(&path).unwrap();
        f.new_dataset_builder()
            .with_data(&data)
            .create("px")
            .unwrap();
    }
    let f = File::open(&path).unwrap();
    let back: Array2<Pixel> = f.dataset("px").unwrap().read_2d().unwrap();
    assert_eq!(back, data);
    // datatype descriptor equality through the file
    let dt = f.dataset("px").unwrap().dtype().unwrap();
    assert_eq!(dt.to_descriptor().unwrap(), Pixel::type_descriptor());
    assert_eq!(dt.size(), Pixel::type_descriptor().size());
}

#[test]
fn mixed_compound_with_strings() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("mixed.h5");
    let rows = vec![
        Mixed {
            id: 1,
            value: 0.5,
            tag: FixedAscii::from_ascii(b"one").unwrap(),
        },
        Mixed {
            id: 2,
            value: 1.5,
            tag: FixedAscii::from_ascii(b"two").unwrap(),
        },
    ];
    {
        let f = File::create(&path).unwrap();
        f.new_dataset_builder()
            .with_data(&arr1(&rows))
            .create("rows")
            .unwrap();
    }
    let f = File::open(&path).unwrap();
    let back: Vec<Mixed> = f.dataset("rows").unwrap().read_raw().unwrap();
    assert_eq!(back, rows);
}

#[test]
fn strings_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("strings.h5");
    {
        let f = File::create(&path).unwrap();
        let va = vec![
            VarLenAscii::from_ascii("plain").unwrap(),
            VarLenAscii::from_ascii("").unwrap(),
        ];
        f.new_dataset_builder()
            .with_data(&arr1(&va))
            .create("va")
            .unwrap();
        let fu = vec![
            FixedUnicode::<12>::from_str("üni").unwrap(),
            FixedUnicode::<12>::from_str("code").unwrap(),
        ];
        f.new_dataset_builder()
            .with_data(&arr1(&fu))
            .create("fu")
            .unwrap();
    }
    let f = File::open(&path).unwrap();
    let va: Vec<VarLenAscii> = f.dataset("va").unwrap().read_raw().unwrap();
    assert_eq!(va[0].as_str(), "plain");
    assert_eq!(va[1].as_str(), "");
    let fu: Vec<FixedUnicode<12>> = f.dataset("fu").unwrap().read_raw().unwrap();
    assert_eq!(fu[0].as_str(), "üni");
    assert_eq!(fu[1].as_str(), "code");
}

#[test]
fn vlen_arrays_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("vla.h5");
    {
        let f = File::create(&path).unwrap();
        let data = vec![
            VarLenArray::from_slice(&[1u16, 2]),
            VarLenArray::from_slice(&[3u16, 4, 5, 6]),
        ];
        f.new_dataset_builder()
            .with_data(&arr1(&data))
            .create("vla")
            .unwrap();
    }
    let f = File::open(&path).unwrap();
    let back: Vec<VarLenArray<u16>> = f.dataset("vla").unwrap().read_raw().unwrap();
    assert_eq!(back[0].as_slice(), &[1, 2]);
    assert_eq!(back[1].as_slice(), &[3, 4, 5, 6]);
}

#[test]
fn chunked_filters_resize() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("chunk.h5");
    let base = Array2::from_shape_fn((16, 16), |(i, j)| (i * 16 + j) as i32);
    {
        let f = File::create(&path).unwrap();
        let ds = f
            .new_dataset::<i32>()
            .chunk((4, 4))
            .shape((16.., 16))
            .deflate(6)
            .shuffle()
            .fletcher32()
            .create("data")
            .unwrap();
        ds.write(&base).unwrap();
        ds.resize((32, 16)).unwrap();
        ds.write_slice(&base, s![16..32, ..]).unwrap();
    }
    let f = File::open(&path).unwrap();
    let ds = f.dataset("data").unwrap();
    assert!(ds.is_chunked());
    assert!(ds.is_resizable());
    assert_eq!(ds.chunk(), Some(vec![4, 4]));
    assert_eq!(ds.shape(), vec![32, 16]);
    assert_eq!(ds.filters().len(), 3);
    let all: Array2<i32> = ds.read_2d().unwrap();
    assert_eq!(all.slice(s![..16, ..]), base);
    assert_eq!(all.slice(s![16.., ..]), base);
    // partial read
    let part: Array2<i32> = ds.read_slice(s![4..8, 8..12]).unwrap();
    assert_eq!(part[[0, 0]], base[[4, 8]]);
}

#[test]
fn slicing_and_selections() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("slice.h5");
    let data = Array2::from_shape_fn((10, 10), |(i, j)| (i * 10 + j) as i64);
    {
        let f = File::create(&path).unwrap();
        f.new_dataset_builder()
            .with_data(&data)
            .create("d")
            .unwrap();
    }
    let f = File::open(&path).unwrap();
    let ds = f.dataset("d").unwrap();
    // stepped slice
    let sub: Array2<i64> = ds.read_slice(s![0..10;2, 0..4]).unwrap();
    assert_eq!(sub.shape(), [5, 4]);
    assert_eq!(sub[[1, 1]], data[[2, 1]]);
    // index (drops the axis)
    let row: Array1<i64> = ds.read_slice_1d(s![3, ..]).unwrap();
    assert_eq!(row, data.row(3));
    // write through a slice
    let fw = File::open_rw(&path).unwrap();
    let dsw = fw.dataset("d").unwrap();
    dsw.write_slice(&arr1(&[-1i64, -2, -3]), s![0, 0..3])
        .unwrap();
    let head: Array1<i64> = dsw.read_slice_1d(s![0, 0..3]).unwrap();
    assert_eq!(head, arr1(&[-1i64, -2, -3]));
}

#[test]
fn fill_values() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("fill.h5");
    {
        let f = File::create(&path).unwrap();
        f.new_dataset::<f64>()
            .fill_value(7.5f64)
            .shape([4])
            .create("filled")
            .unwrap();
    }
    let f = File::open(&path).unwrap();
    let ds = f.dataset("filled").unwrap();
    let v: Array1<f64> = ds.read_1d().unwrap();
    assert_eq!(v, arr1(&[7.5f64; 4]));
    let fv = ds.fill_value().unwrap().unwrap();
    assert_eq!(fv.cast::<f64>().ok(), Some(7.5));
}

#[test]
fn byte_reader() {
    use std::io::{Read, Seek, SeekFrom};
    let dir = tempdir().unwrap();
    let path = dir.path().join("bytes.h5");
    {
        let f = File::create(&path).unwrap();
        f.new_dataset_builder()
            .with_data(&arr1(&[0x01020304u32, 0x05060708]))
            .create("d")
            .unwrap();
    }
    let f = File::open(&path).unwrap();
    let mut r = f.dataset("d").unwrap().as_byte_reader().unwrap();
    assert_eq!(r.len(), 8);
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).unwrap();
    assert_eq!(buf, [0x04, 0x03, 0x02, 0x01]); // little-endian
    r.seek(SeekFrom::Start(4)).unwrap();
    r.read_exact(&mut buf).unwrap();
    assert_eq!(buf, [0x08, 0x07, 0x06, 0x05]);
}

#[test]
fn numeric_conversions() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("conv.h5");
    {
        let f = File::create(&path).unwrap();
        let ds = f.new_dataset::<i64>().shape([3]).create("wide").unwrap();
        ds.write(&arr1(&[1i32, 2, 3])).unwrap(); // i32 -> i64 write conversion
    }
    let f = File::open(&path).unwrap();
    let ds = f.dataset("wide").unwrap();
    let as_i64: Vec<i64> = ds.read_raw().unwrap();
    assert_eq!(as_i64, vec![1, 2, 3]);
    let as_f32: Vec<f32> = ds.read_raw().unwrap(); // i64 -> f32 read conversion
    assert_eq!(as_f32, vec![1.0, 2.0, 3.0]);
    // no_convert must reject mismatched types
    assert!(ds.as_reader().no_convert().read_raw::<i32>().is_err());
}

#[test]
fn errors_are_reported() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("errs.h5");
    let f = File::create(&path).unwrap();
    f.new_dataset::<i32>().create("d").unwrap();
    // duplicate name
    assert!(f.new_dataset::<i32>().create("d").is_err());
    // missing object
    assert!(f.dataset("missing").is_err());
    assert!(f.group("d").is_err()); // wrong kind
    drop(f);
    // read-only writes fail
    let f = File::open(&path).unwrap();
    assert!(f.create_group("g").is_err());
    assert!(f.dataset("d").unwrap().write_scalar(&1i32).is_err());
    // exclusive create on existing file fails
    assert!(File::create_excl(&path).is_err());
}

#[test]
fn large_group_many_links() {
    // exceeds one SNOD (8 symbols) and one btree leaf (32 SNODs = 256 links)
    let dir = tempdir().unwrap();
    let path = dir.path().join("many.h5");
    {
        let f = File::create(&path).unwrap();
        for i in 0..300 {
            f.new_dataset::<i32>()
                .create(format!("d{i:04}").as_str())
                .unwrap()
                .write_scalar(&(i as i32))
                .unwrap();
        }
    }
    let f = File::open(&path).unwrap();
    assert_eq!(f.len(), 300);
    assert_eq!(
        f.dataset("d0299").unwrap().read_scalar::<i32>().unwrap(),
        299
    );
    assert_eq!(f.dataset("d0000").unwrap().read_scalar::<i32>().unwrap(), 0);
}

#[test]
fn many_chunks_multilevel_btree() {
    // more than 64 chunks forces a multi-node (and multi-level) chunk btree
    let dir = tempdir().unwrap();
    let path = dir.path().join("chunks.h5");
    let data = Array2::from_shape_fn((40, 40), |(i, j)| (i * 40 + j) as u32);
    {
        let f = File::create(&path).unwrap();
        let ds = f
            .new_dataset::<u32>()
            .chunk((4, 4))
            .shape((40, 40))
            .create("d")
            .unwrap();
        ds.write(&data).unwrap(); // 100 chunks
    }
    let f = File::open(&path).unwrap();
    let ds = f.dataset("d").unwrap();
    assert_eq!(ds.num_chunks(), Some(100));
    assert_eq!(ds.read_2d::<u32>().unwrap(), data);
}

#[test]
fn dataspace_and_datatype_api() {
    use hdf5::{Dataspace, Datatype, Extents};
    let space = Dataspace::try_new((5, 10)).unwrap();
    assert_eq!(space.shape(), vec![5, 10]);
    assert_eq!(space.ndim(), 2);
    assert_eq!(space.size(), 50);
    assert!(!space.is_resizable());
    let enc = space.encode().unwrap();
    let dec = Dataspace::decode(&enc).unwrap();
    assert_eq!(dec.shape(), vec![5, 10]);

    let null = Dataspace::try_new(Extents::Null).unwrap();
    assert!(null.is_null());
    let scalar = Dataspace::try_new(Extents::Scalar).unwrap();
    assert!(scalar.is_scalar());

    let dt = Datatype::from_type::<f64>().unwrap();
    assert_eq!(dt.size(), 8);
    assert!(dt.is::<f64>());
    assert!(!dt.is::<f32>());
}

#[test]
fn overwrite_and_append_modes() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("modes.h5");
    {
        let f = File::create(&path).unwrap();
        f.new_dataset::<i32>()
            .create("d")
            .unwrap()
            .write_scalar(&1)
            .unwrap();
    }
    {
        let f = File::append(&path).unwrap();
        assert!(f.link_exists("d"));
        f.new_dataset::<i32>()
            .create("d2")
            .unwrap()
            .write_scalar(&2)
            .unwrap();
    }
    {
        let f = File::open(&path).unwrap();
        assert_eq!(f.len(), 2);
    }
    {
        // truncate
        let f = File::create(&path).unwrap();
        assert_eq!(f.len(), 0);
        drop(f);
    }
}

#[test]
fn saturating_numeric_conversions() {
    // libhdf5 semantics: int narrowing saturates (not wraps), negative ->
    // unsigned clamps to 0, float->int saturates with NaN -> 0.
    let dir = tempdir().unwrap();
    let path = dir.path().join("sat.h5");
    let f = File::create(&path).unwrap();
    let ds = f.new_dataset::<i8>().shape([4]).create("i8").unwrap();
    ds.write(&arr1(&[300i64, -300, 5, i64::MAX])).unwrap();
    let v: Vec<i8> = ds.read_raw().unwrap();
    assert_eq!(v, vec![127, -128, 5, 127]);

    let du = f.new_dataset::<u8>().shape([3]).create("u8").unwrap();
    du.write(&arr1(&[-5i32, 300, 42])).unwrap();
    let v: Vec<u8> = du.read_raw().unwrap();
    assert_eq!(v, vec![0, 255, 42]);

    let di = f.new_dataset::<i16>().shape([3]).create("i16").unwrap();
    di.write(&arr1(&[f64::NAN, 1e10, -1e10])).unwrap();
    let v: Vec<i16> = di.read_raw().unwrap();
    assert_eq!(v, vec![0, i16::MAX, i16::MIN]);
}

#[test]
fn external_links() {
    let dir = tempdir().unwrap();
    let target = dir.path().join("target.h5");
    let source = dir.path().join("source.h5");
    {
        let t = File::create(&target).unwrap();
        t.new_dataset_builder()
            .with_data(&arr1(&[7i32, 8, 9]))
            .create("data")
            .unwrap();
        t.close().unwrap();
        let s = File::create(&source).unwrap();
        s.link_external("target.h5", "/data", "ext").unwrap();
        s.new_dataset::<i32>()
            .create("own")
            .unwrap()
            .write_scalar(&1)
            .unwrap();
        s.close().unwrap();
    }
    let f = File::open(&source).unwrap();
    // follow the cross-file link
    let v: Vec<i32> = f.dataset("ext").unwrap().read_raw().unwrap();
    assert_eq!(v, vec![7, 8, 9]);
    // link metadata
    let info = f
        .iter_visit_default(
            vec![],
            |_, name, info, acc: &mut Vec<(String, hdf5::LinkType)>| {
                acc.push((name.to_string(), info.link_type));
                true
            },
        )
        .unwrap();
    assert!(info.contains(&("ext".to_string(), hdf5::LinkType::External)));
}

#[test]
fn modification_time_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("mtime.h5");
    {
        let f = File::create(&path).unwrap();
        f.new_dataset::<i32>().create("d").unwrap();
        f.close().unwrap();
    }
    let f = File::open(&path).unwrap();
    let info = f.dataset("d").unwrap().loc_info().unwrap();
    assert!(
        info.mtime > 1_600_000_000,
        "mtime not preserved: {}",
        info.mtime
    );
}

#[test]
fn lzf_filter_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("lzf.h5");
    let data = Array2::from_shape_fn((32, 32), |(i, j)| ((i * 32 + j) % 7) as i64);
    {
        let f = File::create(&path).unwrap();
        f.new_dataset_builder()
            .with_data(&data)
            .chunk((8, 32))
            .lzf()
            .create("z")
            .unwrap();
    }
    let f = File::open(&path).unwrap();
    let ds = f.dataset("z").unwrap();
    assert_eq!(ds.filters(), vec![hdf5::filters::Filter::LZF]);
    assert_eq!(ds.read_2d::<i64>().unwrap(), data);
}

#[test]
fn globals_are_exposed() {
    // API-parity: dereferencing globals must compile and yield distinct ids.
    let a = *hdf5::globals::H5T_NATIVE_INT;
    let b = *hdf5::globals::H5T_NATIVE_DOUBLE;
    assert_ne!(a, b);
    let _ = hdf5::globals::H5P_DEFAULT;
}

#[test]
fn blosc_filter_roundtrip() {
    use hdf5::filters::{Blosc, BloscShuffle};
    let dir = tempdir().unwrap();
    let path = dir.path().join("blosc.h5");
    let data = Array2::from_shape_fn((40, 50), |(i, j)| ((i * 50 + j) % 13) as i32);
    {
        let f = File::create(&path).unwrap();
        f.new_dataset_builder()
            .with_data(&data)
            .chunk((10, 50))
            .blosc_blosclz(5, BloscShuffle::Byte)
            .create("blz")
            .unwrap();
        f.new_dataset_builder()
            .with_data(&data)
            .chunk((10, 50))
            .blosc_lz4(5, true)
            .create("lz4")
            .unwrap();
        f.new_dataset_builder()
            .with_data(&data)
            .chunk((10, 50))
            .blosc(Blosc::ZLib, 5, BloscShuffle::None)
            .create("zl")
            .unwrap();
        f.new_dataset_builder()
            .with_data(&data)
            .chunk((10, 50))
            .blosc_snappy(5, BloscShuffle::Byte)
            .create("sn")
            .unwrap();
        // zstd frames are written with stored (uncompressed) streams
        f.new_dataset_builder()
            .with_data(&data)
            .chunk((10, 50))
            .blosc_zstd(5, true)
            .create("zs")
            .unwrap();
        // bit-shuffle
        f.new_dataset_builder()
            .with_data(&data)
            .chunk((10, 50))
            .blosc_lz4(5, hdf5::filters::BloscShuffle::Bit)
            .create("bit")
            .unwrap();
    }
    let f = File::open(&path).unwrap();
    for name in ["blz", "lz4", "zl", "sn", "zs", "bit"] {
        assert_eq!(
            f.dataset(name).unwrap().read_2d::<i32>().unwrap(),
            data,
            "{name}"
        );
    }
}

#[test]
fn dense_attributes_large() {
    // attributes above the 64 KB compact limit switch to dense storage
    let dir = tempdir().unwrap();
    let path = dir.path().join("dense.h5");
    let big = Array1::from_shape_fn(20_000, |i| i as f64); // 156 KB
    {
        let f = File::create(&path).unwrap();
        let ds = f.new_dataset::<i32>().create("x").unwrap();
        ds.new_attr::<f64>()
            .shape([20_000])
            .create("big")
            .unwrap()
            .write(&big)
            .unwrap();
        for i in 0..5 {
            ds.new_attr::<i32>()
                .create(format!("small{i}").as_str())
                .unwrap()
                .write_scalar(&i)
                .unwrap();
        }
    }
    let f = File::open(&path).unwrap();
    let ds = f.dataset("x").unwrap();
    assert_eq!(ds.attr_names().unwrap().len(), 6);
    let back: Array1<f64> = ds.attr("big").unwrap().read_1d().unwrap();
    assert_eq!(back, big);
    assert_eq!(ds.attr("small3").unwrap().read_scalar::<i32>().unwrap(), 3);
}

#[cfg(feature = "szip")]
#[test]
fn szip_filter_roundtrip() {
    use hdf5::filters::{Filter, SZip};
    let dir = tempdir().unwrap();
    let path = dir.path().join("szip.h5");
    let ints = Array1::from_shape_fn(3000, |i| (i as i32 / 3) - 100);
    let shorts = Array1::from_shape_fn(2000, |i| (i as i16).wrapping_mul(7));
    {
        let f = File::create(&path).unwrap();
        f.new_dataset_builder()
            .with_data(&ints)
            .chunk((500,))
            .set_filters(&[Filter::SZip(SZip::NearestNeighbor, 16)])
            .create("nn32")
            .unwrap();
        f.new_dataset_builder()
            .with_data(&shorts)
            .chunk((512,))
            .set_filters(&[Filter::SZip(SZip::Entropy, 8)])
            .create("ec16")
            .unwrap();
    }
    let f = File::open(&path).unwrap();
    assert_eq!(f.dataset("nn32").unwrap().read_1d::<i32>().unwrap(), ints);
    assert_eq!(f.dataset("ec16").unwrap().read_1d::<i16>().unwrap(), shorts);
    let filters = f.dataset("nn32").unwrap().filters();
    assert!(
        matches!(filters[0], Filter::SZip(SZip::NearestNeighbor, _)),
        "{filters:?}"
    );
}
