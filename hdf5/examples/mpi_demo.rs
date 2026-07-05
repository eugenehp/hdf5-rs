//! SPMD parallel-write demo for the pure-Rust MPI subset (feature = mpi).
//! Run without env: spawns 4 ranks of itself, waits, verifies the merged file.
use hdf5::mpi::{self, Comm, Op};
use ndarray::Array1;

const RANKS: usize = 4;
const COLS: usize = 1000;

fn path() -> String {
    let dir = std::env::var("H5MPI_DEMO_DIR").unwrap_or_else(|_| "/tmp".into());
    format!("{dir}/mpi_shared.h5")
}

fn rank_main() -> hdf5::Result<()> {
    let comm = Comm::init()?;
    let (rank, size) = (comm.rank(), comm.size());

    // collective metadata: identical calls on every rank
    let f = mpi::create(path(), &comm)?;
    let g = f.create_group("results")?;
    let ds = g
        .new_dataset::<f64>()
        .shape([size, COLS])
        .create("matrix")?;
    let total = f.new_dataset::<u64>().create("checksum")?;

    // independent data: each rank writes its own row
    let row = Array1::from_shape_fn(COLS, |j| (rank * COLS + j) as f64);
    ds.write_slice(&row, ndarray::s![rank..rank + 1, ..])?;

    // a collective reduction, written by rank 0 only
    let local: u64 = row.iter().map(|&x| x as u64).sum();
    let global = comm.allreduce_u64(local, Op::Sum)?;
    if rank == 0 {
        total.write_scalar(&global)?;
    }
    // sanity: bcast + barrier work
    let token = comm.bcast(b"hello", 0)?;
    assert_eq!(token, b"hello");
    comm.barrier()?;

    f.close()?; // collective: rank 0 merges and writes
    Ok(())
}

fn main() -> hdf5::Result<()> {
    if mpi::is_worker() {
        return rank_main();
    }
    // launcher: spawn RANKS copies of ourselves and wait
    let children = mpi::spawn_workers(RANKS)?;
    for mut c in children {
        let st = c.wait().map_err(|e| format!("wait: {e}"))?;
        if !st.success() {
            return Err("a rank failed".into());
        }
    }
    // verify the merged file with our own reader
    let f = hdf5::File::open(path())?;
    let m: ndarray::Array2<f64> = f.dataset("results/matrix")?.read_2d()?;
    assert_eq!(m.shape(), [RANKS, COLS]);
    for (i, v) in m.iter().enumerate() {
        assert_eq!(*v, i as f64, "element {i}");
    }
    let sum: u64 = f.dataset("checksum")?.read_scalar()?;
    assert_eq!(sum, (0..RANKS * COLS).map(|x| x as u64).sum::<u64>());
    println!("mpi: {RANKS} ranks wrote one shared file; merged output verified OK");
    Ok(())
}
