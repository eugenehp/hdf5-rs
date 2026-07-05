use hdf5::{File, H5Type};

#[derive(H5Type, Clone, PartialEq, Debug)]
#[repr(u8)]
pub enum Color {
    R = 1,
    G = 2,
    B = 3,
}

#[derive(H5Type, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct Pixel {
    xy: (i64, i64),
    color: Color,
}

fn main() -> hdf5::Result<()> {
    let path = "/private/tmp/claude-501/-Users-Shared-hdf5-rs/0ebd41ce-ec10-4ac9-9239-8cf4acc80aa8/scratchpad/rust_compound.h5";
    {
        let file = File::create(path)?;
        let group = file.create_group("dir")?;
        let pix = |x, y, c| Pixel {
            xy: (x, y),
            color: c,
        };
        group
            .new_dataset_builder()
            .with_data(&ndarray::arr2(&[
                [pix(1, 2, Color::R), pix(2, 3, Color::B)],
                [pix(3, 4, Color::G), pix(4, 5, Color::R)],
            ]))
            .create("pixels")?;
        let attr = group
            .dataset("pixels")?
            .new_attr::<Color>()
            .shape([3])
            .create("colors")?;
        attr.write(&ndarray::arr1(&[Color::R, Color::G, Color::B]))?;
        // soft link
        file.link_soft("/dir/pixels", "pixels_alias")?;
        file.close()?;
    }
    let file = File::open(path)?;
    let ds = file.dataset("dir/pixels")?;
    let px: ndarray::Array2<Pixel> = ds.read_2d()?;
    assert_eq!(
        px[[1, 0]],
        Pixel {
            xy: (3, 4),
            color: Color::G
        }
    );
    let colors: Vec<Color> = ds.attr("colors")?.read_raw()?;
    assert_eq!(colors, vec![Color::R, Color::G, Color::B]);
    // follow the soft link
    let ds2 = file.dataset("pixels_alias")?;
    assert_eq!(ds2.shape(), vec![2, 2]);
    // slicing
    use ndarray::s;
    let row: ndarray::Array2<Pixel> = ds.read_slice(s![1.., ..])?;
    assert_eq!(row.shape(), [1, 2]);
    assert_eq!(
        row[[0, 1]],
        Pixel {
            xy: (4, 5),
            color: Color::R
        }
    );
    println!("rust compound+enum+softlink roundtrip OK");
    Ok(())
}
