//! Corpus reader: opens every .h5/.nc file in a directory, visits every
//! group/dataset/attribute, and reads every dataset's bytes. Reports
//! per-file results; exits nonzero if anything is unreadable.
use hdf5::{File, Group};

/// Read a container with the natural typed API for its dtype, falling back
/// to raw file-layout bytes for compounds/references/opaque data.
fn read_any(c: &hdf5::Container) -> hdf5::Result<()> {
    use hdf5::types::TypeDescriptor as TD;
    use hdf5::types::VarLenUnicode;
    match c.dtype()?.to_descriptor()? {
        TD::Integer(_) => c.read_raw::<i64>().map(drop),
        TD::Unsigned(_) | TD::Boolean => c.read_raw::<u64>().map(drop),
        TD::Float(_) => c.read_raw::<f64>().map(drop),
        TD::FixedAscii(_) | TD::FixedUnicode(_) | TD::VarLenAscii | TD::VarLenUnicode => {
            c.read_raw::<VarLenUnicode>().map(drop)
        }
        _ => c.read_bytes().map(drop),
    }
}

fn visit(g: &Group, stats: &mut (usize, usize, Vec<String>)) {
    for name in g.member_names().unwrap_or_default() {
        if let Ok(sub) = g.group(&name) {
            visit(&sub, stats);
            continue;
        }
        match g.dataset(&name) {
            Ok(ds) => {
                stats.0 += 1;
                if let Err(e) = read_any(&ds) {
                    stats.2.push(format!("{}/{name}: read: {e}", g.name()));
                }
                for an in ds.attr_names().unwrap_or_default() {
                    stats.1 += 1;
                    match ds.attr(&an).and_then(|a| read_any(&a)) {
                        Ok(_) => {}
                        Err(e) => stats.2.push(format!("{}/{name}@{an}: {e}", g.name())),
                    }
                }
            }
            Err(e) => stats.2.push(format!("{}/{name}: open: {e}", g.name())),
        }
    }
    for an in g.attr_names().unwrap_or_default() {
        stats.1 += 1;
        if let Err(e) = g.attr(&an).and_then(|a| read_any(&a)) {
            stats.2.push(format!("{}@{an}: {e}", g.name()));
        }
    }
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: corpus <dir>");
    let mut total_fail = 0;
    let mut entries: Vec<_> = std::fs::read_dir(&dir).unwrap().flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let p = entry.path();
        let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !matches!(ext, "h5" | "nc" | "hdf5") {
            continue;
        }
        match File::open(&p) {
            Ok(f) => {
                let mut stats = (0usize, 0usize, Vec::new());
                visit(&f.as_group().unwrap(), &mut stats);
                let name = p.file_name().unwrap().to_string_lossy().into_owned();
                if stats.2.is_empty() {
                    println!("OK   {name}: {} datasets, {} attrs", stats.0, stats.1);
                } else {
                    println!("FAIL {name}: {} problems", stats.2.len());
                    for e in &stats.2 {
                        println!("       {e}");
                    }
                    total_fail += stats.2.len();
                }
            }
            Err(e) => {
                println!("FAIL {}: open: {e}", p.display());
                total_fail += 1;
            }
        }
    }
    if total_fail > 0 {
        std::process::exit(1);
    }
    // value spot-checks against the generators' known content
    let f = File::open(format!("{dir}/nc_classic.nc")).unwrap();
    let t: Vec<f32> = f.dataset("temp").unwrap().read_raw().unwrap();
    assert_eq!(t.len(), 600);
    assert_eq!(t[599], 599.0);
    let f = File::open(format!("{dir}/pytables.h5")).unwrap();
    let s: Vec<i64> = f.dataset("series").unwrap().read_raw().unwrap();
    assert_eq!(s, (0..1000).collect::<Vec<i64>>());
    let p: Vec<i64> = f.dataset("plain").unwrap().read_raw().unwrap();
    assert_eq!(p, (0..24).collect::<Vec<i64>>());
    let f = File::open(format!("{dir}/pandas_fixed.h5")).unwrap();
    let a: Vec<i64> = f.dataset("df/block0_values").unwrap().read_raw().unwrap();
    assert_eq!(a[..5], [0, 1, 2, 3, 4]);
    println!("corpus fully readable + spot values verified");
}
