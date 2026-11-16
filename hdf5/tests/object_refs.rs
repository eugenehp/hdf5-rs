//! HDF5 object references (`H5R_OBJECT`): reading a reference-typed dataset and
//! resolving each token back to the group or dataset it addresses.
//!
//! This is the shape MATLAB `-v7.3` uses for cell arrays — the cell dataset
//! holds one reference per element, pointing into `/#refs#`. `object_refs.h5`
//! is the same structure written by h5py: two dataset targets and one group
//! target, referenced from `/refs`.

use hdf5::{ObjectReference1, ReferencedObject};

fn fixture() -> String {
    let here = env!("CARGO_MANIFEST_DIR");
    for cand in [
        format!("{here}/object_refs.h5"),
        format!("{here}/../object_refs.h5"),
    ] {
        if std::path::Path::new(&cand).exists() {
            return cand;
        }
    }
    panic!("object_refs.h5 not found");
}

#[test]
fn a_reference_dataset_reads_as_tokens_and_resolves_to_its_targets() {
    let f = hdf5::File::open(fixture()).unwrap();
    let refs = f
        .dataset("refs")
        .unwrap()
        .read_raw::<ObjectReference1>()
        .unwrap();
    assert_eq!(refs.len(), 3, "three references were written");
    assert!(
        refs.iter().all(|r| !r.is_null()),
        "h5py wrote real addresses, not the null reference"
    );

    // Two dataset targets, in order, with the values h5py stored.
    match f.dereference(&refs[0]).unwrap() {
        ReferencedObject::Dataset(d) => {
            assert_eq!(d.read_raw::<i32>().unwrap(), vec![0, 1, 2, 3])
        }
        other => panic!("refs[0] should be a dataset, got {other:?}"),
    }
    match f.dereference(&refs[1]).unwrap() {
        ReferencedObject::Dataset(d) => {
            assert_eq!(d.read_raw::<i32>().unwrap(), vec![10, 11, 12])
        }
        other => panic!("refs[1] should be a dataset, got {other:?}"),
    }
    // ...and a group target, which must not be mistaken for a dataset.
    match f.dereference(&refs[2]).unwrap() {
        ReferencedObject::Group(_) => {}
        other => panic!("refs[2] should be a group, got {other:?}"),
    }
}

#[test]
fn a_reference_round_trips_through_its_address() {
    let f = hdf5::File::open(fixture()).unwrap();
    let refs = f
        .dataset("refs")
        .unwrap()
        .read_raw::<ObjectReference1>()
        .unwrap();
    let rebuilt = ObjectReference1::from_address(refs[0].address());
    assert_eq!(rebuilt, refs[0]);
    // Resolving the rebuilt token reaches the same object.
    match f.dereference(&rebuilt).unwrap() {
        ReferencedObject::Dataset(d) => {
            assert_eq!(d.read_raw::<i32>().unwrap(), vec![0, 1, 2, 3])
        }
        other => panic!("expected a dataset, got {other:?}"),
    }
}

#[test]
fn the_null_reference_is_refused_rather_than_resolved() {
    let f = hdf5::File::open(fixture()).unwrap();
    let null = ObjectReference1::default();
    assert!(null.is_null());
    assert!(
        f.dereference(&null).is_err(),
        "a null reference addresses nothing and must not resolve"
    );
}
