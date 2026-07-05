//! Variable-length strings and arrays, fixed-size strings.
use hdf5::types::{FixedAscii, VarLenArray, VarLenUnicode};
use hdf5::File;
use std::str::FromStr;

fn main() -> hdf5::Result<()> {
    let path = std::env::temp_dir().join("rust_vlen.h5");
    let path = path.to_str().unwrap();
    {
        let f = File::create(path)?;
        let strs = vec![
            VarLenUnicode::from_str("hello").unwrap(),
            VarLenUnicode::from_str("wörld✓").unwrap(),
            VarLenUnicode::from_str("").unwrap(),
        ];
        f.new_dataset_builder()
            .with_data(&ndarray::arr1(&strs))
            .create("strings")?;
        let seqs = vec![
            VarLenArray::from_slice(&[1i32, 2, 3]),
            VarLenArray::from_slice(&[42i32]),
            VarLenArray::from_slice(&[] as &[i32]),
        ];
        f.new_dataset_builder()
            .with_data(&ndarray::arr1(&seqs))
            .create("seqs")?;
        let fixed = vec![
            FixedAscii::<8>::from_ascii(b"abc").unwrap(),
            FixedAscii::<8>::from_ascii(b"defgh").unwrap(),
        ];
        f.new_dataset_builder()
            .with_data(&ndarray::arr1(&fixed))
            .create("fixed")?;
        // vlen string attribute
        let attr = f.new_attr::<VarLenUnicode>().create("note")?;
        attr.write_scalar(&VarLenUnicode::from_str("attached note").unwrap())?;
        f.close()?;
    }
    // roundtrip
    let f = File::open(path)?;
    let s: Vec<VarLenUnicode> = f.dataset("strings")?.read_raw()?;
    assert_eq!(s[0].as_str(), "hello");
    assert_eq!(s[1].as_str(), "wörld✓");
    assert_eq!(s[2].as_str(), "");
    let q: Vec<VarLenArray<i32>> = f.dataset("seqs")?.read_raw()?;
    assert_eq!(q[0].as_slice(), &[1, 2, 3]);
    assert_eq!(q[1].as_slice(), &[42]);
    assert_eq!(q[2].len(), 0);
    let x: Vec<FixedAscii<8>> = f.dataset("fixed")?.read_raw()?;
    assert_eq!(x[0].as_str(), "abc");
    assert_eq!(x[1].as_str(), "defgh");
    let note: VarLenUnicode = f.attr("note")?.read_scalar()?;
    assert_eq!(note.as_str(), "attached note");
    println!("rust vlen roundtrip OK");
    Ok(())
}
