//! Deterministic mutation fuzzer: parse mutated corpus files; any panic is a
//! bug (errors are fine). Usage: fuzz <dir> [mutations-per-file]; replay a
//! saved crash with panics shown: fuzz <file>
use std::panic::{catch_unwind, AssertUnwindSafe};

fn main() {
    let dir = std::env::args().nth(1).expect("usage: fuzz <dir> [n]");
    if std::path::Path::new(&dir).is_file() {
        let r = hdf5::format::parse(&std::fs::read(&dir).unwrap());
        println!("parsed: {}", if r.is_ok() { "ok" } else { "error" });
        return;
    }
    let n: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    std::panic::set_hook(Box::new(|_| {})); // silence expected-panic spam
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut rng = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut files = 0;
    let mut panics = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(&dir).unwrap().flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let p = entry.path();
        if !matches!(
            p.extension().and_then(|e| e.to_str()),
            Some("h5" | "nc" | "hdf5")
        ) {
            continue;
        }
        let orig = std::fs::read(&p).unwrap();
        files += 1;
        for i in 0..n {
            let mut data = orig.clone();
            // 1-8 random byte mutations (flip / random / truncate)
            for _ in 0..=(rng() % 8) {
                if data.is_empty() {
                    break;
                }
                let pos = (rng() as usize) % data.len();
                match rng() % 3 {
                    0 => data[pos] ^= (rng() & 0xff) as u8,
                    1 => data[pos] = (rng() & 0xff) as u8,
                    _ => data.truncate(pos.max(64)),
                }
            }
            let r = catch_unwind(AssertUnwindSafe(|| {
                let _ = hdf5::format::parse(&data);
                // and lazily, through every read path a caller can reach
                if let Ok(f) = hdf5::File::from_bytes(data.clone()) {
                    walk(&f, 0);
                }
            }));
            if r.is_err() {
                panics.push((p.clone(), i));
                std::fs::write(
                    format!(
                        "{dir}/panic_{}_{i}.bin",
                        p.file_stem().unwrap().to_string_lossy()
                    ),
                    &data,
                )
                .ok();
            }
        }
    }
    let _ = std::panic::take_hook();
    if panics.is_empty() {
        println!("fuzz: {files} files x {n} mutations, no panics");
    } else {
        println!(
            "fuzz: {} PANICS: {:?}",
            panics.len(),
            &panics[..panics.len().min(5)]
        );
        std::process::exit(1);
    }
}

/// Every group and dataset below `g`: shape, type, rows, a slice, the whole.
fn walk(g: &hdf5::Group, depth: usize) {
    let Ok(names) = g.member_names() else { return };
    for n in names.iter().take(64) {
        if let Ok(sub) = g.group(n) {
            if depth < 16 {
                walk(&sub, depth + 1);
            }
        } else if let Ok(d) = g.dataset(n) {
            let _ = (d.shape(), d.dtype());
            let _ = d.read_rows_raw::<f64>(0, 4);
            let _ = d.read_rows_raw::<i16>(0, 4);
            // whole reads only where small: a mutated shape can claim terabytes
            if d.size() < 1 << 22 {
                let _ = d.read_slice::<f64, _, ndarray::IxDyn>(ndarray::s![..]);
                let _ = d.read_raw::<f64>();
            }
        }
    }
}
