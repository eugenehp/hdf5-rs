//! Edit an existing file in place: `edit_in_place <file.h5>` — writes into
//! three chunked datasets (whatever chunk index the file uses), grows one,
//! and adds a dataset; the file is appended to, not rewritten.
use ndarray::{s, Array2};

fn main() -> hdf5::Result<()> {
    let path = std::env::args().nth(1).expect("a file");
    let before = std::fs::metadata(&path)?.len();
    let f = hdf5::File::open_rw(&path)?;
    let patch = Array2::from_elem((5, 4), -1i32);
    for name in ["fixed", "grow", "two"] {
        f.dataset(name)?
            .write_slice(&patch, s![20_000..20_005, 2..6])?;
    }
    let grow = f.dataset("grow")?;
    grow.resize((60_000, 8))?;
    grow.write_slice(&Array2::from_elem((10_000, 8), 7i32), s![50_000.., ..])?;
    f.new_dataset_builder()
        .with_data(&[1.5f64, 2.5])
        .create("added")?;
    drop(f);
    println!("{before} -> {} bytes", std::fs::metadata(&path)?.len());
    Ok(())
}
