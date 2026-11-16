//! Read speed on a file written by h5py: `bench_read <file>`
//! (generate it with `python3 interop/make_bench.py <file>`).
use std::time::Instant;

use hdf5::types::VarLenUnicode;

fn time(name: &str, mut f: impl FnMut()) {
    let mut best = f64::MAX;
    for _ in 0..3 {
        let s = Instant::now();
        f();
        best = best.min(s.elapsed().as_secs_f64());
    }
    println!("{name:16} {:9.1} ms", best * 1000.0);
}

fn main() {
    let path = std::env::args().nth(1).expect("a file");
    if std::env::var_os("BENCH_ONLY_WALK").is_some() {
        for _ in 0..50 {
            walk(&path);
        }
        return;
    }
    // every layout decodes to the same values
    let f = hdf5::File::open(&path).unwrap();
    let want = f.dataset("contig").unwrap().read_raw::<i16>().unwrap();
    for name in ["plain_chunks", "gzip_rows", "gzip_cols", "lzf"] {
        assert!(
            f.dataset(name).unwrap().read_raw::<i16>().unwrap() == want,
            "{name}"
        );
    }
    let as_f64: Vec<f64> = want.iter().map(|&v| f64::from(v)).collect();
    assert!(f.dataset("be_f64").unwrap().read_raw::<f64>().unwrap() == as_f64);
    let as_f32: Vec<f32> = want.iter().map(|&v| f32::from(v)).collect();
    assert!(f.dataset("shuffle_f32").unwrap().read_raw::<f32>().unwrap() == as_f32);
    assert!(
        f.dataset("plain_chunks")
            .unwrap()
            .read_raw::<f32>()
            .unwrap()
            == as_f32
    );
    let rows = f
        .dataset("gzip_cols")
        .unwrap()
        .read_rows_raw::<i16>(3_000_000, 3_000_064)
        .unwrap();
    assert!(rows[..] == want[3_000_000 * 8..3_000_064 * 8]);
    drop(f);
    println!("values check out; {} threads", hdf5::num_threads());
    for name in ["contig", "plain_chunks", "gzip_rows", "gzip_cols"] {
        time(name, || {
            let f = hdf5::File::open(&path).unwrap();
            std::hint::black_box(f.dataset(name).unwrap().read_raw::<i16>().unwrap());
        });
    }
    time("shuffle_f32", || {
        let f = hdf5::File::open(&path).unwrap();
        std::hint::black_box(f.dataset("shuffle_f32").unwrap().read_raw::<f32>().unwrap());
    });
    time("be_f64", || {
        let f = hdf5::File::open(&path).unwrap();
        std::hint::black_box(f.dataset("be_f64").unwrap().read_raw::<f64>().unwrap());
    });
    time("i16_as_f32", || {
        let f = hdf5::File::open(&path).unwrap();
        std::hint::black_box(
            f.dataset("plain_chunks")
                .unwrap()
                .read_raw::<f32>()
                .unwrap(),
        );
    });
    time("lzf", || {
        let f = hdf5::File::open(&path).unwrap();
        std::hint::black_box(f.dataset("lzf").unwrap().read_raw::<i16>().unwrap());
    });
    let f = hdf5::File::open(&path).unwrap();
    let data = f.dataset("contig").unwrap().read_2d::<i16>().unwrap();
    time("write_gzip", || {
        let out = hdf5::File::create_in_memory().unwrap();
        out.new_dataset_builder()
            .chunk((65536, 8))
            .shuffle()
            .deflate(4)
            .with_data(&data)
            .create("d")
            .unwrap();
        std::hint::black_box(out.to_bytes().unwrap());
    });
    time("rows_mid", || {
        let f = hdf5::File::open(&path).unwrap();
        std::hint::black_box(
            f.dataset("gzip_cols")
                .unwrap()
                .read_rows_raw::<i16>(3_000_000, 3_000_064)
                .unwrap(),
        );
    });
    time("strings", || {
        std::hint::black_box(
            f.dataset("strings")
                .unwrap()
                .read_raw::<VarLenUnicode>()
                .unwrap(),
        );
    });
    time("open", || {
        std::hint::black_box(hdf5::File::open(&path).unwrap());
    });
    time("open_reader", || {
        let r = std::fs::File::open(&path).unwrap();
        std::hint::black_box(hdf5::File::open_reader(r).unwrap());
    });
    time("walk20k", || walk(&path));
}

/// Every one of 20k groups: its attribute and dataset shape.
fn walk(path: &str) {
    let r = std::fs::File::open(path).unwrap();
    let f = hdf5::File::open_reader(r).unwrap();
    let g = f.group("many").unwrap();
    for n in g.member_names().unwrap() {
        let s = g.group(&n).unwrap();
        std::hint::black_box(s.attr("neurodata_type").unwrap());
        std::hint::black_box(s.dataset("data").unwrap().shape());
    }
}
