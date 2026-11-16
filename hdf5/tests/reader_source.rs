//! `File::open_reader` (lazy Read+Seek source) must read identically to the
//! path/mmap-based `File::open`, including chunked data materialized on demand.
use std::io::Cursor;

/// `tempfile::tempdir`, also under WASI, which has no `temp_dir` (the test
/// runner, `interop/wasi_run.mjs`, preopens the host's as `/tmp`).
fn tempdir() -> std::io::Result<tempfile::TempDir> {
    #[cfg(target_os = "wasi")]
    let _ = tempfile::env::override_temp_dir(std::path::Path::new("/tmp"));
    tempfile::tempdir()
}

fn fixture(name: &str) -> String {
    let here = env!("CARGO_MANIFEST_DIR");
    for cand in [format!("{here}/{name}"), format!("{here}/../{name}")] {
        if std::path::Path::new(&cand).exists() {
            return cand;
        }
    }
    name.to_string()
}

fn read_i32(path: &str, dset: &str) -> (Vec<usize>, Vec<i32>) {
    let f = hdf5::File::open(path).unwrap();
    let d = f.dataset(dset).unwrap();
    (d.shape(), d.read_raw::<i32>().unwrap_or_default())
}

fn read_i32_via_reader(path: &str, dset: &str) -> (Vec<usize>, Vec<i32>) {
    let bytes = std::fs::read(path).unwrap();
    let f = hdf5::File::open_reader(Cursor::new(bytes)).unwrap();
    let d = f.dataset(dset).unwrap();
    (d.shape(), d.read_raw::<i32>().unwrap_or_default())
}

#[test]
fn open_reader_matches_open_chunked() {
    let (s1, v1) = read_i32(&fixture("chunking.h5"), "variable");
    let (s2, v2) = read_i32_via_reader(&fixture("chunking.h5"), "variable");
    assert_eq!(s1, s2, "shape mismatch");
    assert_eq!(v1, v2, "data mismatch");
    assert!(!v1.is_empty());
}

#[test]
fn open_reader_lists_members() {
    let bytes = std::fs::read(fixture("chunking.h5")).unwrap();
    let f = hdf5::File::open_reader(Cursor::new(bytes)).unwrap();
    let names = f.group("/").unwrap().member_names().unwrap();
    assert!(names.iter().any(|n| n == "variable"));
}

/// A lazily opened file has not parsed a group until something walks into it;
/// `link_exists` must answer for the file, not for what has been parsed.
#[test]
fn open_reader_link_exists_walks_unparsed_groups() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("nested.h5");
    {
        let f = hdf5::File::create(&path).unwrap();
        let g = f.create_group("a/b/c").unwrap();
        g.new_dataset::<f64>().shape(3).create("d").unwrap();
        f.group("a")
            .unwrap()
            .link_soft("/nowhere", "dangling")
            .unwrap();
    }
    let open = || hdf5::File::open_reader(Cursor::new(std::fs::read(&path).unwrap())).unwrap();
    // each from a fresh file: nothing has walked there first
    assert!(open().link_exists("a/b"));
    assert!(open().link_exists("a/b/c"));
    assert!(open().link_exists("a/b/c/d"));
    assert!(open().link_exists("/a/b/c/d"));
    assert!(!open().link_exists("a/b/nope"));
    assert!(!open().link_exists("a/nope/c"));
    assert!(!open().link_exists("a/b/c/d/e"));
    // a link can exist with nothing at its target
    assert!(open().link_exists("a/dangling"));
    // from a group handle, as well as the file
    assert!(open().group("a").unwrap().link_exists("b/c/d"));
}

/// Asking a dataset what it is must not read what it holds: over a range
/// reader that is the whole download, for a shape.
#[test]
fn open_reader_metadata_leaves_data_unread() {
    use std::io::{Read, Seek, SeekFrom};
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::sync::Arc;

    struct Counted(Cursor<Vec<u8>>, Arc<AtomicU64>);
    impl Read for Counted {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let n = self.0.read(buf)?;
            self.1.fetch_add(n as u64, Relaxed);
            Ok(n)
        }
    }
    impl Seek for Counted {
        fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
            self.0.seek(to)
        }
    }

    let dir = tempdir().unwrap();
    let path = dir.path().join("big.h5");
    let n = 8 << 20; // 64 MB of f64, many read blocks
    let values: Vec<f64> = (0..n).map(|i| i as f64).collect();
    {
        let f = hdf5::File::create(&path).unwrap();
        f.new_dataset_builder()
            .with_data(&values)
            .create("big")
            .unwrap();
    }
    let bytes = std::fs::read(&path).unwrap();
    let total = bytes.len() as u64;
    let read = Arc::new(AtomicU64::new(0));
    let f = hdf5::File::open_reader(Counted(Cursor::new(bytes), read.clone())).unwrap();
    let d = f.dataset("big").unwrap();
    assert_eq!(d.shape(), [n]);
    assert_eq!(d.ndim(), 1);
    assert_eq!(d.size(), n);
    assert!(!d.is_scalar());
    assert_eq!(d.space().unwrap().shape(), [n]);
    assert!(d.dtype().unwrap().is::<f64>());
    let seen = read.load(Relaxed);
    assert!(seen < total / 4, "metadata read {seen} of {total} bytes");
    // and the data is still there when asked for
    assert_eq!(d.read_raw::<f64>().unwrap(), values);
}

/// A virtual dataset's source in the same file may sit in a group nothing has
/// walked into yet.
#[test]
fn open_reader_virtual_dataset_reaches_unparsed_source() {
    let path = format!("{}/tests/data/vds_same_file.h5", env!("CARGO_MANIFEST_DIR"));
    let want: Vec<i32> = (0..12).collect();
    let eager = hdf5::File::open(&path).unwrap();
    assert_eq!(eager.dataset("v").unwrap().read_raw::<i32>().unwrap(), want);
    let f = hdf5::File::open_reader(Cursor::new(std::fs::read(&path).unwrap())).unwrap();
    assert_eq!(f.dataset("v").unwrap().read_raw::<i32>().unwrap(), want);
}

/// The user block is left unread on a lazy open, not lost: writing the file
/// back must still carry it.
#[test]
fn open_reader_to_bytes_keeps_the_user_block() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("ub.h5");
    {
        let f = hdf5::File::with_options()
            .with_fcpl(|p| p.userblock(512))
            .create(&path)
            .unwrap();
        f.new_dataset_builder()
            .with_data(&[1i32, 2, 3])
            .create("x")
            .unwrap();
    }
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[..5].copy_from_slice(b"hello");
    let f = hdf5::File::open_reader(Cursor::new(bytes)).unwrap();
    assert_eq!(f.userblock(), 512);
    let out = f.to_bytes().unwrap();
    assert_eq!(&out[..5], b"hello");
}

#[test]
fn open_reader_with_block_reads_the_same_and_checks_the_size() {
    let bytes = std::fs::read(fixture("chunking.h5")).unwrap();
    let (_, want) = read_i32(&fixture("chunking.h5"), "variable");
    for block in [512, 4096, 1 << 20] {
        let f = hdf5::File::open_reader_with_block(Cursor::new(bytes.clone()), block).unwrap();
        assert_eq!(
            f.dataset("variable").unwrap().read_raw::<i32>().unwrap(),
            want
        );
    }
    assert!(hdf5::File::open_reader_with_block(Cursor::new(bytes), 1000).is_err());
}

/// Rows of a chunked dataset read only the index nodes over them: on a long
/// recording the index alone is far more than the rows wanted.
#[test]
fn open_reader_rows_walk_only_their_part_of_the_chunk_index() {
    use std::io::{Read, Seek, SeekFrom};
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::sync::Arc;

    struct Counted(Cursor<Vec<u8>>, Arc<AtomicU64>);
    impl Read for Counted {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let n = self.0.read(buf)?;
            self.1.fetch_add(n as u64, Relaxed);
            Ok(n)
        }
    }
    impl Seek for Counted {
        fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
            self.0.seek(to)
        }
    }

    let dir = tempdir().unwrap();
    let path = dir.path().join("rows.h5");
    let (rows, cols) = (200_000usize, 2usize);
    let values: Vec<i32> = (0..(rows * cols) as i32).collect();
    {
        let f = hdf5::File::create(&path).unwrap();
        f.new_dataset_builder()
            .chunk((8, 1))
            .with_data(&ndarray::Array2::from_shape_vec((rows, cols), values.clone()).unwrap())
            .create("d")
            .unwrap();
    }
    let bytes = std::fs::read(&path).unwrap();
    let total = bytes.len() as u64;
    for (r0, r1) in [(0, 3), (5, 21), (100_003, 100_050), (rows - 9, rows)] {
        let read = Arc::new(AtomicU64::new(0));
        let f = hdf5::File::open_reader_with_block(
            Counted(Cursor::new(bytes.clone()), read.clone()),
            4096,
        )
        .unwrap();
        let got = f
            .dataset("d")
            .unwrap()
            .read_rows_raw::<i32>(r0, r1)
            .unwrap();
        assert_eq!(got, values[r0 * cols..r1 * cols], "rows {r0}..{r1}");
        let seen = read.load(Relaxed);
        assert!(
            seen < total / 20,
            "rows {r0}..{r1} read {seen} of {total} bytes"
        );
    }
}

/// Big-endian numbers stay lazy: read whole or by rows, on either kind of
/// open, they come out the same, and asking their shape reads none of them.
#[test]
fn open_reader_big_endian_stays_lazy_and_reads_right() {
    let path = format!("{}/tests/data/big_endian.h5", env!("CARGO_MANIFEST_DIR"));
    let eager = hdf5::File::open(&path).unwrap();
    let lazy = hdf5::File::open_reader(Cursor::new(std::fs::read(&path).unwrap())).unwrap();
    let floats: Vec<f64> = (0..4096).map(f64::from).collect();
    let ints: Vec<i32> = (-500..500).collect();
    for f in [&eager, &lazy] {
        let c = f.dataset("contiguous").unwrap();
        assert_eq!(c.shape(), [1024, 4]);
        assert_eq!(c.read_rows_raw::<f64>(100, 103).unwrap(), floats[400..412]);
        assert_eq!(c.read_raw::<f64>().unwrap(), floats);
        let d = f.dataset("chunked").unwrap();
        assert_eq!(d.read_rows_raw::<i32>(15, 33).unwrap(), ints[60..132]);
        assert_eq!(d.read_raw::<i32>().unwrap(), ints);
    }
    // the shape of a big-endian dataset reads none of its 32 kB, in 512 B blocks
    {
        use std::io::{Read, Seek, SeekFrom};
        use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
        use std::sync::Arc;
        struct Counted(Cursor<Vec<u8>>, Arc<AtomicU64>);
        impl Read for Counted {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                let n = self.0.read(buf)?;
                self.1.fetch_add(n as u64, Relaxed);
                Ok(n)
            }
        }
        impl Seek for Counted {
            fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
                self.0.seek(to)
            }
        }
        let read = Arc::new(AtomicU64::new(0));
        let bytes = std::fs::read(&path).unwrap();
        let f = hdf5::File::open_reader_with_block(Counted(Cursor::new(bytes), read.clone()), 512)
            .unwrap();
        assert_eq!(f.dataset("contiguous").unwrap().shape(), [1024, 4]);
        let seen = read.load(Relaxed);
        assert!(seen < 16 << 10, "the shape read {seen} bytes");
    }
    #[derive(hdf5::H5Type, Clone, Copy, PartialEq, Debug)]
    #[repr(C)]
    struct Pair {
        a: i16,
        b: f32,
    }
    let want: Vec<Pair> = (0..5)
        .map(|i| Pair {
            a: i as i16,
            b: i as f32 * 0.5,
        })
        .collect();
    assert_eq!(
        lazy.dataset("compound")
            .unwrap()
            .read_raw::<Pair>()
            .unwrap(),
        want
    );
}

/// Every shape the run-by-run copy has to get right, on either kind of open:
/// chunks cut off at the edges, spanning dimensions whole or not, and numbers
/// converted as they are read.
#[test]
fn layouts_read_whole_and_converted() {
    let path = format!("{}/tests/data/layouts.h5", env!("CARGO_MANIFEST_DIR"));
    let eager = hdf5::File::open_as(&path, hdf5::file::OpenMode::ReadWrite);
    let lazy = hdf5::File::open(&path).unwrap();
    let reader = hdf5::File::open_reader(Cursor::new(std::fs::read(&path).unwrap())).unwrap();
    let mut files = vec![lazy, reader];
    files.extend(eager);
    for f in &files {
        let d3: Vec<i32> = (0..37 * 11 * 5).collect();
        assert_eq!(f.dataset("d3").unwrap().read_raw::<i32>().unwrap(), d3);
        assert_eq!(
            f.dataset("d3").unwrap().read_rows_raw::<i32>(7, 9).unwrap(),
            d3[7 * 55..9 * 55]
        );
        let d1: Vec<f64> = (0..1000).map(f64::from).collect();
        assert_eq!(f.dataset("d1").unwrap().read_raw::<f64>().unwrap(), d1);
        let wide: Vec<i16> = (0..700).collect();
        assert_eq!(f.dataset("wide").unwrap().read_raw::<i16>().unwrap(), wide);
        // unfiltered chunks: only the rows' own bytes, across a chunk boundary,
        // chunks that span the columns and chunks that do not, to the last row
        let rows = |name: &str, r0, r1| {
            f.dataset(name)
                .unwrap()
                .read_rows_raw::<i32>(r0, r1)
                .unwrap()
        };
        let w = |r0, r1| {
            f.dataset("wide")
                .unwrap()
                .read_rows_raw::<i16>(r0, r1)
                .unwrap()
        };
        assert_eq!(w(14, 18), wide[14 * 7..18 * 7]);
        assert_eq!(w(96, 100), wide[96 * 7..]);
        let cols: Vec<i32> = (0..200 * 6).collect();
        assert_eq!(rows("cols", 30, 75), cols[180..450]);
        assert_eq!(rows("cols", 190, 200), cols[1140..]);
        assert_eq!(rows("cols", 0, 200), cols);
        let tall: Vec<u16> = (0..450).collect();
        assert_eq!(f.dataset("tall").unwrap().read_raw::<u16>().unwrap(), tall);
        // converted
        let as_f64: Vec<f64> = (0..37 * 11 * 5).map(f64::from).collect();
        assert_eq!(f.dataset("d3").unwrap().read_raw::<f64>().unwrap(), as_f64);
        let i8s: Vec<f64> = (-128..128).map(f64::from).collect();
        assert_eq!(f.dataset("i8").unwrap().read_raw::<f64>().unwrap(), i8s);
        assert_eq!(
            f.dataset("u32").unwrap().read_raw::<f64>().unwrap(),
            [0.0, 1.0, 4294967295.0]
        );
        assert_eq!(
            f.dataset("u32").unwrap().read_raw::<i64>().unwrap(),
            [0, 1, 4294967295]
        );
        assert_eq!(
            f.dataset("f32").unwrap().read_raw::<f64>().unwrap(),
            [0.5, -1.25]
        );
        assert_eq!(
            f.dataset("i64").unwrap().read_raw::<i64>().unwrap(),
            [-5, 1 << 40]
        );
        // narrowed: clamped, as libhdf5 does
        assert_eq!(f.dataset("i8").unwrap().read_raw::<u8>().unwrap()[0], 0);
        assert_eq!(
            f.dataset("u32").unwrap().read_raw::<i32>().unwrap()[2],
            i32::MAX
        );
        assert_eq!(
            f.dataset("scalar").unwrap().read_scalar::<i32>().unwrap(),
            7
        );
        // chunks never written read as zero
        let mut sparse = vec![0i32; 40];
        for (i, v) in sparse[8..16].iter_mut().enumerate() {
            *v = i as i32 + 1;
        }
        assert_eq!(
            f.dataset("sparse").unwrap().read_raw::<i32>().unwrap(),
            sparse
        );
    }
}

/// A large group's name index is only a hint: it must never answer for what
/// has since been unlinked, renamed or added.
#[test]
fn large_group_lookup_survives_changes() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("many.h5");
    let f = hdf5::File::create(&path).unwrap();
    for i in 0..100 {
        f.create_group(&format!("g{i}")).unwrap();
    }
    assert!(f.link_exists("g50"));
    assert!(!f.link_exists("missing"));
    f.unlink("g50").unwrap();
    assert!(!f.link_exists("g50"));
    assert!(f.group("g50").is_err());
    assert!(f.link_exists("g99"));
    f.create_group("g50").unwrap();
    assert!(f.link_exists("g50"));
    f.relink("g10", "renamed").unwrap();
    assert!(f.link_exists("renamed"));
    assert!(!f.link_exists("g10"));
    assert!(f.group("g98").is_ok());
    assert_eq!(f.member_names().unwrap().len(), 100);
}

/// A member's attributes are in its object header: reading one walks neither
/// its own members nor its siblings'.
#[test]
fn member_attr_leaves_the_member_unparsed() {
    use std::io::{Read, Seek, SeekFrom};
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::sync::Arc;
    struct Counted(Cursor<Vec<u8>>, Arc<AtomicU64>);
    impl Read for Counted {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let n = self.0.read(buf)?;
            self.1.fetch_add(n as u64, Relaxed);
            Ok(n)
        }
    }
    impl Seek for Counted {
        fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
            self.0.seek(to)
        }
    }
    use hdf5::types::VarLenUnicode;
    let dir = tempdir().unwrap();
    let path = dir.path().join("kinds.h5");
    {
        let f = hdf5::File::create(&path).unwrap();
        let big = f.create_group("top/big").unwrap();
        let kind: VarLenUnicode = "Images".parse().unwrap();
        big.new_attr::<VarLenUnicode>()
            .create("kind")
            .unwrap()
            .write_scalar(&kind)
            .unwrap();
        for i in 0..3000 {
            big.create_group(&format!("g{i}")).unwrap();
        }
        f.group("top")
            .unwrap()
            .link_soft("/top/big", "soft")
            .unwrap();
        f.group("top")
            .unwrap()
            .link_soft("/nowhere", "dangling")
            .unwrap();
    }
    let open = |read: &Arc<AtomicU64>| {
        let bytes = std::fs::read(&path).unwrap();
        hdf5::File::open_reader_with_block(Counted(Cursor::new(bytes), read.clone()), 512).unwrap()
    };
    let (peek, walk) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    let f = open(&peek);
    let top = f.group("top").unwrap();
    let a = top.member_attr("big", "kind").unwrap();
    assert_eq!(a.read_scalar::<VarLenUnicode>().unwrap().as_str(), "Images");
    assert!(top.member_attr("big", "none").is_err());
    assert!(top.member_attr("none", "kind").is_err());
    let peeked = peek.load(Relaxed); // following a link below opens the group
                                     // a soft link is followed, a dangling one is an error, and neither hangs
    let a = top.member_attr("soft", "kind").unwrap();
    assert_eq!(a.read_scalar::<VarLenUnicode>().unwrap().as_str(), "Images");
    assert!(top.member_attr("dangling", "kind").is_err());
    // the member still opens whole, attribute and all
    let g = open(&walk).group("top/big").unwrap();
    assert_eq!(g.member_names().unwrap().len(), 3000);
    let walked = walk.load(Relaxed);
    assert!(
        peeked * 10 < walked,
        "{peeked} read, against {walked} to open it"
    );
    let f = open(&Arc::new(AtomicU64::new(0)));
    let top = f.group("top").unwrap();
    top.member_attr("big", "kind").unwrap();
    assert_eq!(
        top.group("big")
            .unwrap()
            .attr("kind")
            .unwrap()
            .read_scalar::<VarLenUnicode>()
            .unwrap()
            .as_str(),
        "Images"
    );
}
