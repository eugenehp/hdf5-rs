//! Writing object references (`H5R_OBJECT`): datasets and attributes of them,
//! written, read back, and followed to their objects. (h5py follows them
//! too: `interop/check_h5py.py verify`, on `interop_harness write`'s
//! rust_refs.h5.)

use hdf5::{File, ObjectReference1, ReferencedObject};

/// The `id` attribute of the object a reference points at (an object
/// reached by address does not know its path, so its name is no help).
fn name(f: &File, r: &ObjectReference1) -> u8 {
    match f.dereference(r).unwrap() {
        ReferencedObject::Group(g) => g.attr("id").unwrap().read_scalar::<u8>().unwrap(),
        ReferencedObject::Dataset(d) => d.attr("id").unwrap().read_scalar::<u8>().unwrap(),
    }
}

#[test]
fn references_written_and_followed() {
    let path = std::env::temp_dir().join(format!("hdf5-refs-{}.h5", std::process::id()));
    {
        let f = File::create(&path).unwrap();
        let a = f.create_group("devices/a").unwrap();
        let b = f.create_group("devices/b").unwrap();
        a.new_attr::<u8>()
            .create("id")
            .unwrap()
            .write_scalar(&1)
            .unwrap();
        b.new_attr::<u8>()
            .create("id")
            .unwrap()
            .write_scalar(&2)
            .unwrap();
        let values = f.new_dataset::<f64>().shape([3]).create("values").unwrap();
        values.write(&[1.0, 2.0, 3.0]).unwrap();
        // a column of references (as NWB's electrodes/group), targets in
        // other groups, one written after the column
        let table = f.create_group("table").unwrap();
        table
            .new_attr::<u8>()
            .create("id")
            .unwrap()
            .write_scalar(&3)
            .unwrap();
        table.new_reference_dataset("group", &[&a, &b, &a]).unwrap();
        // a region of the table, elsewhere, pointing at it (as NWB's
        // DynamicTableRegion `table` attribute)
        let region = f
            .create_group("acquisition")
            .unwrap()
            .new_dataset::<i32>()
            .shape([2])
            .create("electrodes")
            .unwrap();
        region.write(&[0, 2]).unwrap();
        region
            .new_attr::<u8>()
            .create("id")
            .unwrap()
            .write_scalar(&4)
            .unwrap();
        region.new_reference_attr("table", &table).unwrap();
        values
            .new_reference_array_attr("sources", &[&b, &region])
            .unwrap();
        f.flush().unwrap();
    }
    if let Ok(keep) = std::env::var("HDF5_REFS_KEEP") {
        std::fs::copy(&path, keep).unwrap();
    }
    let f = File::open(&path).unwrap();
    let column = f
        .dataset("table/group")
        .unwrap()
        .read_raw::<ObjectReference1>()
        .unwrap();
    let names: Vec<u8> = column.iter().map(|r| name(&f, r)).collect();
    assert_eq!(names, [1, 2, 1]);
    let table = f
        .dataset("acquisition/electrodes")
        .unwrap()
        .attr("table")
        .unwrap()
        .read_scalar::<ObjectReference1>()
        .unwrap();
    assert_eq!(name(&f, &table), 3);
    let sources = f
        .dataset("values")
        .unwrap()
        .attr("sources")
        .unwrap()
        .read_raw::<ObjectReference1>()
        .unwrap();
    assert_eq!(
        sources.iter().map(|r| name(&f, r)).collect::<Vec<_>>(),
        [2, 4]
    );
    // the data around them are as written
    assert_eq!(
        f.dataset("values").unwrap().read_raw::<f64>().unwrap(),
        [1.0, 2.0, 3.0]
    );
    drop(f);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_reference_to_an_enclosing_group_is_refused_on_writing() {
    let path = std::env::temp_dir().join(format!("hdf5-refs-self-{}.h5", std::process::id()));
    let f = File::create(&path).unwrap();
    let g = f.create_group("g").unwrap();
    let inner = g.new_dataset::<u8>().shape([1]).create("x").unwrap();
    inner.new_reference_attr("parent", &g).unwrap();
    assert!(f.flush().is_err());
    let other = File::create(
        std::env::temp_dir().join(format!("hdf5-refs-other-{}.h5", std::process::id())),
    )
    .unwrap();
    assert!(inner
        .new_reference_attr("elsewhere", &other.create_group("o").unwrap())
        .is_err());
}
