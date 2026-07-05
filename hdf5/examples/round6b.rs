use hdf5::plist::file_create::{SharedMessageIndex, SharedMessageType};
use hdf5::{File, H5Type};
use ndarray::Array1;

#[derive(H5Type, Clone, PartialEq, Debug)]
#[repr(C)]
struct Wide {
    a: i64,
    b: f64,
    c: i32,
    d: u32,
    e: f32,
    f: i16,
}

fn main() -> hdf5::Result<()> {
    let d = "/private/tmp/claude-501/-Users-Shared-hdf5-rs/0ebd41ce-ec10-4ac9-9239-8cf4acc80aa8/scratchpad";
    // 1. SOHM writing: shared dtype+dataspace messages across 6 datasets
    let p1 = format!("{d}/rust_sohm.h5");
    {
        let mut b = File::with_options();
        b.fcpl().shared_mesg_indexes(&[SharedMessageIndex {
            message_types: SharedMessageType::DATATYPE | SharedMessageType::SIMPLE_DATASPACE,
            min_message_size: 30,
        }]);
        let f = b.create(&p1)?;
        let rows: Vec<Wide> = (0..8)
            .map(|i| Wide {
                a: i,
                b: i as f64 / 2.0,
                c: i as i32,
                d: i as u32,
                e: 0.5,
                f: 7,
            })
            .collect();
        for k in 0..6 {
            f.new_dataset_builder()
                .with_data(&Array1::from(rows.clone()))
                .create(format!("d{k}").as_str())?;
        }
        f.close()?;
    }
    let f = File::open(&p1)?;
    for k in 0..6 {
        let rows: Vec<Wide> = f.dataset(&format!("d{k}"))?.read_raw()?;
        assert_eq!(rows[7].a, 7);
        assert_eq!(rows[7].b, 3.5);
    }
    println!("rust SOHM write roundtrip OK");

    // 2. VDS writing
    let src_p = format!("{d}/rust_vds_src.h5");
    {
        let f = File::create(&src_p)?;
        for i in 0..3 {
            f.new_dataset_builder()
                .with_data(&Array1::from_shape_fn(10, |j| (i * 10 + j) as i32))
                .create(format!("s{i}").as_str())?;
        }
        f.close()?;
    }
    let p2 = format!("{d}/rust_vds.h5");
    {
        let f = File::create(&p2)?;
        let mut dcpl = hdf5::plist::DatasetCreateBuilder::new();
        for i in 0..3usize {
            dcpl.virtual_map(
                "rust_vds_src.h5",
                format!("/s{i}"),
                [10usize],
                hdf5::Selection::All,
                [3usize, 10],
                (i..i + 1, ..),
            );
        }
        f.new_dataset_builder()
            .set_create_plist(&dcpl.finish()?)
            .empty::<i32>()
            .shape([3, 10])
            .create("virt")?;
        f.close()?;
    }
    let f = File::open(&p2)?;
    let v: ndarray::Array2<i32> = f.dataset("virt")?.read_2d()?;
    for (i, x) in v.iter().enumerate() {
        assert_eq!(*x, i as i32);
    }
    println!("rust VDS write roundtrip OK");
    Ok(())
}
