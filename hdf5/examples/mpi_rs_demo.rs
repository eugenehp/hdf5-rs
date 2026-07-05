//! SPMD parallel write with the `mpi-rs` transport (feature = mpi-rs).
//!
//! Standalone run — a singleton MPI world (1 rank):
//!
//! ```bash
//! cargo run --example mpi_rs_demo --features mpi-rs
//! ```
//!
//! Real SPMD run — any number of ranks via mpi-rs's launcher:
//!
//! ```bash
//! cargo install mpi-rs               # provides `mpiexec`
//! cargo build --example mpi_rs_demo --features mpi-rs
//! mpiexec -n 4 target/debug/examples/mpi_rs_demo
//! ```
use hdf5::mpi::{self, Comm, Op};
use ndarray::Array1;

const COLS: usize = 1000;

fn main() -> hdf5::Result<()> {
    let path = std::env::temp_dir().join("mpi_rs_shared.h5");
    // no H5MPI_* env set, so this initializes the mpi-rs runtime: the
    // mpiexec-launched world, or a singleton when run directly
    let comm = Comm::init()?;
    let (rank, size) = (comm.rank(), comm.size());
    println!("rank {rank} of {size} up");

    // collective metadata: identical calls on every rank
    let f = mpi::create(&path, &comm)?;
    let ds = f.new_dataset::<f64>().shape([size, COLS]).create("matrix")?;
    let total = f.new_dataset::<u64>().create("checksum")?;

    // independent data: each rank writes its own row
    let row = Array1::from_shape_fn(COLS, |j| (rank * COLS + j) as f64);
    ds.write_slice(&row, ndarray::s![rank..rank + 1, ..])?;

    // collective reduction over mpi-rs, written by rank 0 only
    let local: u64 = row.iter().map(|&x| x as u64).sum();
    let global = comm.allreduce_u64(local, Op::Sum)?;
    if rank == 0 {
        total.write_scalar(&global)?;
    }
    let token = comm.bcast(b"hello", 0)?;
    assert_eq!(token, b"hello");

    f.close()?; // collective: rank 0 merges every rank's writes

    if rank == 0 {
        let f = hdf5::File::open(&path)?;
        let m: ndarray::Array2<f64> = f.dataset("matrix")?.read_2d()?;
        assert_eq!(m.shape(), [size, COLS]);
        for (i, v) in m.iter().enumerate() {
            assert_eq!(*v, i as f64, "element {i}");
        }
        let sum: u64 = f.dataset("checksum")?.read_scalar()?;
        assert_eq!(sum, (0..size * COLS).map(|x| x as u64).sum::<u64>());
        println!("mpi-rs: {size} rank(s) wrote one shared file; merged output verified OK");
    }
    Ok(())
}
