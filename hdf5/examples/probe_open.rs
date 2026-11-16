//! Open a (possibly damaged) file and walk it: `probe_open <file>`. Errors
//! are fine; a crash, or a long stall, is a bug.
fn walk(g: &hdf5::Group, depth: usize) {
    let Ok(names) = g.member_names() else { return };
    for n in names {
        if let Ok(sub) = g.group(&n) {
            if depth < 32 {
                walk(&sub, depth + 1);
            }
        } else if let Ok(d) = g.dataset(&n) {
            let _ = d.shape();
            let _ = d.dtype();
            // the partial-read paths too: rows, a slice, a whole read
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = d.read_rows_raw::<f64>(0, 10);
                let _ = d.read_rows_raw::<i16>(0, 10);
                let _ = d.read_slice::<f64, _, ndarray::IxDyn>(ndarray::s![..]);
                let _ = d.read_raw::<f64>();
            }));
            if caught.is_err() {
                println!("PANIC reading {n}");
            }
        }
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("a file");
    let t = std::time::Instant::now();
    match hdf5::File::open(&path) {
        Ok(f) => walk(&f, 0),
        Err(e) => println!("open: {e}"),
    }
    println!("walked in {:.2} s", t.elapsed().as_secs_f64());
}
