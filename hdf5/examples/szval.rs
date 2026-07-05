//! SZip codec validation against libsz-generated vectors (feature = szip).
fn main() -> hdf5::Result<()> {
    let d = std::env::args().nth(1).unwrap();
    let names = [
        "ec8", "nn8", "zeros", "const", "nn16", "msb16", "nn32", "pad", "rand16",
    ];
    for name in names {
        let p = std::fs::read_to_string(format!("{d}/{name}.params"))?;
        let cd: Vec<u32> = p.split_whitespace().map(|x| x.parse().unwrap()).collect();
        let data = std::fs::read(format!("{d}/{name}.data"))?;
        let oc = std::fs::read(format!("{d}/{name}.oc"))?;
        // 1. decode the oracle's stream
        let dec = hdf5::internal_szip_decompress(&cd, &oc, data.len())
            .map_err(|e| format!("{name}: decode: {e}"))?;
        assert_eq!(dec, data, "{name}: decode mismatch");
        // 2. encode ourselves for the oracle to check
        let enc = hdf5::internal_szip_compress(&cd, &data)?;
        std::fs::write(format!("{d}/{name}.rc"), &enc)?;
    }
    println!("rust decodes all {} oracle streams OK", names.len());
    Ok(())
}
