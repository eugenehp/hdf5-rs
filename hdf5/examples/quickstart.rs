//! Quickstart: create a file, write a typed dataset, read it back.
//!
//! ```bash
//! cargo run --example quickstart
//! ```
use hdf5::{File, H5Type};
use ndarray::{arr2, Array2};

#[derive(H5Type, Clone, PartialEq, Debug)]
#[repr(u8)]
enum Quality {
    Good = 1,
    Suspect = 2,
    Bad = 3,
}

#[derive(H5Type, Clone, PartialEq, Debug)]
#[repr(C)]
struct Sample {
    sensor: i32,
    value: f64,
    quality: Quality,
}

fn main() -> hdf5::Result<()> {
    let path = std::env::temp_dir().join("quickstart.h5");

    // -- write ---------------------------------------------------------
    let file = File::create(&path)?;
    let run = file.create_group("run_001")?;

    // a 2-D float dataset from an ndarray
    let grid: Array2<f64> = arr2(&[[0.0, 0.5, 1.0], [1.5, 2.0, 2.5]]);
    run.new_dataset_builder().with_data(&grid).create("grid")?;

    // a dataset of derived structs (compound type with an enum field)
    let samples = vec![
        Sample {
            sensor: 7,
            value: 0.23,
            quality: Quality::Good,
        },
        Sample {
            sensor: 7,
            value: 9.99,
            quality: Quality::Bad,
        },
        Sample {
            sensor: 8,
            value: 0.25,
            quality: Quality::Suspect,
        },
    ];
    run.new_dataset_builder()
        .with_data(&samples)
        .create("samples")?;

    // attributes on any object
    run.new_attr::<f64>()
        .create("temperature_c")?
        .write_scalar(&21.5)?;
    file.close()?;

    // -- read ----------------------------------------------------------
    let file = File::open(&path)?;
    let grid: Array2<f64> = file.dataset("run_001/grid")?.read_2d()?;
    let samples: Vec<Sample> = file.dataset("run_001/samples")?.read_raw()?;
    let temp: f64 = file
        .group("run_001")?
        .attr("temperature_c")?
        .read_scalar()?;

    println!("grid sum      = {}", grid.sum());
    println!("first sample  = {:?}", samples[0]);
    println!("temperature   = {temp} °C");
    assert_eq!(samples.len(), 3);
    Ok(())
}
