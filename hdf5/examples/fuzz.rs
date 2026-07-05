//! Deterministic mutation fuzzer: parse mutated corpus files; any panic is a
//! bug (errors are fine). Usage: fuzz <dir> [mutations-per-file]
use std::panic::{catch_unwind, AssertUnwindSafe};

fn main() {
    let dir = std::env::args().nth(1).expect("usage: fuzz <dir> [n]");
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
