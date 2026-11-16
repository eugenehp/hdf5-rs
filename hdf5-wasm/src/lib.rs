//! JavaScript bindings for `hdf5-no-c`: read and write HDF5 files in the
//! browser or Node, no native library involved.
//!
//! ```js
//! import init, { H5File, H5Remote } from "hdf5-no-c-wasm";
//! await init();                                  // (--target web)
//! const f = H5File.fromBytes(new Uint8Array(await file.arrayBuffer()));
//! f.keys("/");                                   // ["acquisition", ...]
//! f.shape("acquisition/data");                   // [200000, 16]
//! f.read("acquisition/data");                    // Float32Array
//! f.readSlice("acquisition/data", [100, 0], [10, 16]);
//!
//! const r = await H5Remote.open("https://example.org/big.h5");
//! await r.readSlice("acquisition/data", [0, 0], [10, 16]); // range requests
//! ```
//!
//! Numbers come back as typed arrays (64-bit integers as `BigInt64Array` /
//! `BigUint64Array`), strings as arrays of strings, in row-major order; shapes
//! are arrays of numbers.

use std::rc::Rc;

use hdf5::types::{FloatSize, IntSize, TypeDescriptor, VarLenUnicode};
use hdf5::{Container, File, Hyperslab, Selection, SliceOrIndex};
use js_sys::{Array, Promise};
use ndarray::{ArrayD, IxDyn};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

type JsResult<T> = Result<T, JsError>;

fn err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

/// A dataset or attribute, by `path` (and attribute `name`, if any).
fn container(file: &File, path: &str, attr: Option<&str>) -> hdf5::Result<Container> {
    let path = path.trim_start_matches('/');
    match attr {
        None => Ok((*file.dataset(path)?).clone()),
        Some(name) => {
            let loc = match path {
                "" => (***file).clone(),
                p => match file.group(p) {
                    Ok(g) => (*g).clone(),
                    Err(_) => (**file.dataset(p)?).clone(),
                },
            };
            Ok((*loc.attr(name)?).clone())
        }
    }
}

/// `start`/`count` per axis as a selection.
fn hyperslab(start: &[f64], count: &[f64]) -> Selection {
    let dims: Vec<SliceOrIndex> = start
        .iter()
        .zip(count)
        .map(|(&s, &c)| SliceOrIndex::SliceCount {
            start: s as usize,
            step: 1,
            count: c as usize,
            block: 1,
        })
        .collect();
    Selection::from(Hyperslab::from(dims))
}

/// A dataset's values (all, or `sel`) as the matching JS array.
fn read_js(c: &Container, sel: Option<Selection>) -> hdf5::Result<JsValue> {
    fn values<T: hdf5::H5Type>(c: &Container, sel: Option<Selection>) -> hdf5::Result<Vec<T>> {
        match sel {
            None => c.read_raw::<T>(),
            Some(sel) => Ok(c.read_slice::<T, _, IxDyn>(sel)?.into_raw_vec()),
        }
    }
    use TypeDescriptor::*;
    let desc = c.dtype()?.to_descriptor()?;
    Ok(match desc {
        Float(FloatSize::U4) => js_sys::Float32Array::from(&values::<f32>(c, sel)?[..]).into(),
        Float(FloatSize::U8) => js_sys::Float64Array::from(&values::<f64>(c, sel)?[..]).into(),
        Integer(IntSize::U1) => js_sys::Int8Array::from(&values::<i8>(c, sel)?[..]).into(),
        Integer(IntSize::U2) => js_sys::Int16Array::from(&values::<i16>(c, sel)?[..]).into(),
        Integer(IntSize::U4) => js_sys::Int32Array::from(&values::<i32>(c, sel)?[..]).into(),
        Integer(IntSize::U8) => js_sys::BigInt64Array::from(&values::<i64>(c, sel)?[..]).into(),
        Unsigned(IntSize::U1) => js_sys::Uint8Array::from(&values::<u8>(c, sel)?[..]).into(),
        Unsigned(IntSize::U2) => js_sys::Uint16Array::from(&values::<u16>(c, sel)?[..]).into(),
        Unsigned(IntSize::U4) => js_sys::Uint32Array::from(&values::<u32>(c, sel)?[..]).into(),
        Unsigned(IntSize::U8) => js_sys::BigUint64Array::from(&values::<u64>(c, sel)?[..]).into(),
        Boolean => {
            let v: Vec<u8> = values::<bool>(c, sel)?.into_iter().map(u8::from).collect();
            js_sys::Uint8Array::from(&v[..]).into()
        }
        FixedAscii(_) | FixedUnicode(_) | VarLenAscii | TypeDescriptor::VarLenUnicode => {
            values::<hdf5::types::VarLenUnicode>(c, sel)?
                .iter()
                .map(|s| JsValue::from_str(s.as_str()))
                .collect::<Array>()
                .into()
        }
        other => return Err(format!("reading {other} into JavaScript is not supported").into()),
    })
}

/// The JS-facing name of a datatype.
fn dtype_name(c: &Container) -> hdf5::Result<String> {
    use TypeDescriptor::*;
    Ok(match c.dtype()?.to_descriptor()? {
        Float(FloatSize::U4) => "float32".into(),
        Float(FloatSize::U8) => "float64".into(),
        Integer(s) => format!("int{}", s as usize * 8),
        Unsigned(s) => format!("uint{}", s as usize * 8),
        Boolean => "bool".into(),
        FixedAscii(_) | FixedUnicode(_) | VarLenAscii | TypeDescriptor::VarLenUnicode => {
            "string".into()
        }
        other => other.to_string(),
    })
}

fn keys(file: &File, path: &str) -> hdf5::Result<Vec<String>> {
    match path.trim_start_matches('/') {
        "" => file.member_names(),
        p => file.group(p)?.member_names(),
    }
}

fn kind(file: &File, path: &str) -> hdf5::Result<String> {
    let p = path.trim_start_matches('/');
    if p.is_empty() || file.group(p).is_ok() {
        Ok("group".into())
    } else if file.dataset(p).is_ok() {
        Ok("dataset".into())
    } else {
        Err(format!("'{path}' not found").into())
    }
}

fn attr_names(file: &File, path: &str) -> hdf5::Result<Vec<String>> {
    match path.trim_start_matches('/') {
        "" => file.attr_names(),
        p => match file.group(p) {
            Ok(g) => g.attr_names(),
            Err(_) => file.dataset(p)?.attr_names(),
        },
    }
}

/// An HDF5 file held in memory.
#[wasm_bindgen]
pub struct H5File {
    file: File,
}

#[wasm_bindgen]
impl H5File {
    /// Open the bytes of an HDF5 file.
    #[wasm_bindgen(js_name = fromBytes)]
    pub fn from_bytes(bytes: Vec<u8>) -> JsResult<H5File> {
        Ok(Self {
            file: File::from_bytes_rw(bytes).map_err(err)?,
        })
    }

    /// A new, empty file.
    pub fn create() -> JsResult<H5File> {
        Ok(Self {
            file: File::create_in_memory().map_err(err)?,
        })
    }

    /// Member names of the group at `path` (`"/"` for the root).
    pub fn keys(&self, path: &str) -> JsResult<Vec<String>> {
        keys(&self.file, path).map_err(err)
    }

    /// `"group"` or `"dataset"`.
    pub fn kind(&self, path: &str) -> JsResult<String> {
        kind(&self.file, path).map_err(err)
    }

    /// The dataset's shape.
    pub fn shape(&self, path: &str) -> JsResult<Array> {
        let c = container(&self.file, path, None).map_err(err)?;
        Ok(c.shape()
            .into_iter()
            .map(|d| JsValue::from(d as f64))
            .collect())
    }

    /// The dataset's element type: `"float32"`, `"int16"`, `"string"`, ...
    pub fn dtype(&self, path: &str) -> JsResult<String> {
        dtype_name(&container(&self.file, path, None).map_err(err)?).map_err(err)
    }

    /// The whole dataset, row-major.
    pub fn read(&self, path: &str) -> JsResult<JsValue> {
        read_js(&container(&self.file, path, None).map_err(err)?, None).map_err(err)
    }

    /// `count[i]` elements from `start[i]` along each axis, row-major — only
    /// the chunks holding them are decoded.
    #[wasm_bindgen(js_name = readSlice)]
    pub fn read_slice(&self, path: &str, start: Vec<f64>, count: Vec<f64>) -> JsResult<JsValue> {
        let c = container(&self.file, path, None).map_err(err)?;
        read_js(&c, Some(hyperslab(&start, &count))).map_err(err)
    }

    /// Attribute names of the object at `path`.
    #[wasm_bindgen(js_name = attrNames)]
    pub fn attr_names(&self, path: &str) -> JsResult<Vec<String>> {
        attr_names(&self.file, path).map_err(err)
    }

    /// An attribute's value (an array, even for a scalar).
    #[wasm_bindgen(js_name = readAttr)]
    pub fn read_attr(&self, path: &str, name: &str) -> JsResult<JsValue> {
        read_js(&container(&self.file, path, Some(name)).map_err(err)?, None).map_err(err)
    }

    /// Create a group (its parent must exist).
    #[wasm_bindgen(js_name = createGroup)]
    pub fn create_group(&self, path: &str) -> JsResult<()> {
        self.file
            .create_group(path.trim_start_matches('/'))
            .map(drop)
            .map_err(err)
    }

    /// Create a dataset from a typed array (or an array of strings) of
    /// `shape`; chunked by `chunk` and gzip-compressed at level `gzip` when
    /// given.
    #[wasm_bindgen(js_name = writeDataset)]
    pub fn write_dataset(
        &self,
        path: &str,
        data: JsValue,
        shape: Vec<f64>,
        chunk: Option<Vec<f64>>,
        gzip: Option<u8>,
    ) -> JsResult<()> {
        let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let path = path.trim_start_matches('/');
        let chunk: Option<Vec<usize>> = chunk.map(|c| c.iter().map(|&d| d as usize).collect());
        macro_rules! write {
            ($vals:expr) => {{
                let arr = ArrayD::from_shape_vec(IxDyn(&shape), $vals).map_err(err)?;
                let mut b = self.file.new_dataset_builder();
                if let Some(c) = &chunk {
                    b = b.chunk(c.clone());
                }
                if let Some(level) = gzip {
                    b = b.deflate(level);
                }
                b.with_data(&arr).create(path).map(drop).map_err(err)
            }};
        }
        macro_rules! typed {
            ($($js:ty),*) => {$(
                if let Some(a) = data.dyn_ref::<$js>() {
                    return write!(a.to_vec());
                }
            )*};
        }
        typed!(
            js_sys::Float32Array,
            js_sys::Float64Array,
            js_sys::Int8Array,
            js_sys::Int16Array,
            js_sys::Int32Array,
            js_sys::BigInt64Array,
            js_sys::Uint8Array,
            js_sys::Uint16Array,
            js_sys::Uint32Array,
            js_sys::BigUint64Array
        );
        if let Some(a) = data.dyn_ref::<Array>() {
            let strings = a
                .iter()
                .map(|v| {
                    v.as_string()
                        .ok_or_else(|| err("arrays must hold strings"))?
                        .parse::<VarLenUnicode>()
                        .map_err(err)
                })
                .collect::<JsResult<Vec<VarLenUnicode>>>()?;
            return write!(strings);
        }
        Err(err("data must be a typed array or an array of strings"))
    }

    /// The file's bytes, e.g. to download or upload.
    #[wasm_bindgen(js_name = toBytes)]
    pub fn to_bytes(&self) -> JsResult<Vec<u8>> {
        self.file.to_bytes().map_err(err)
    }
}

/// Range requests through the host's global `fetch` (browsers, Node 18+).
struct Fetch {
    url: String,
    len: u64,
}

fn call(target: &JsValue, method: &str, args: &[JsValue]) -> Result<JsValue, JsValue> {
    let f: js_sys::Function = js_sys::Reflect::get(target, &method.into())?.dyn_into()?;
    f.apply(target, &args.iter().collect::<Array>())
}

async fn fetch(url: &str, method: &str, range: Option<(u64, u64)>) -> Result<JsValue, JsValue> {
    let init = js_sys::Object::new();
    js_sys::Reflect::set(&init, &"method".into(), &method.into())?;
    if let Some((start, end)) = range {
        let headers = js_sys::Object::new();
        js_sys::Reflect::set(
            &headers,
            &"Range".into(),
            &format!("bytes={start}-{}", end - 1).into(),
        )?;
        js_sys::Reflect::set(&init, &"headers".into(), &headers)?;
    }
    let promise: Promise =
        call(&js_sys::global(), "fetch", &[url.into(), init.into()])?.dyn_into()?;
    let response = wasm_bindgen_futures::JsFuture::from(promise).await?;
    let ok = js_sys::Reflect::get(&response, &"ok".into())?;
    if !ok.is_truthy() {
        let status = js_sys::Reflect::get(&response, &"status".into())?;
        return Err(format!("{method} {url}: HTTP {}", status.as_f64().unwrap_or(0.0)).into());
    }
    Ok(response)
}

fn io(e: JsValue) -> std::io::Error {
    std::io::Error::other(e.as_string().unwrap_or_else(|| format!("{e:?}")))
}

impl hdf5::remote::AsyncRangeRead for Fetch {
    fn len(&self) -> u64 {
        self.len
    }

    async fn read_range(&self, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
        let response = fetch(&self.url, "GET", Some((offset, offset + len as u64)))
            .await
            .map_err(io)?;
        let buffer: Promise = call(&response, "arrayBuffer", &[])
            .map_err(io)?
            .dyn_into()
            .map_err(io)?;
        let buffer = wasm_bindgen_futures::JsFuture::from(buffer)
            .await
            .map_err(io)?;
        let bytes = js_sys::Uint8Array::new(&buffer).to_vec();
        if bytes.len() != len {
            return Err(std::io::Error::other(
                "the server ignored the Range header (no partial content)",
            ));
        }
        Ok(bytes)
    }
}

/// An HDF5 file read over HTTP range requests: only the bytes each call
/// needs are downloaded. Every method returns a `Promise`.
#[wasm_bindgen]
pub struct H5Remote {
    remote: Rc<hdf5::remote::RemoteFile<Fetch>>,
}

#[wasm_bindgen]
impl H5Remote {
    /// Open `url` (its size from a `HEAD` request), fetching `block` bytes
    /// per range request (default 1 MiB, a power of two).
    pub async fn open(url: String, block: Option<u32>) -> JsResult<H5Remote> {
        let head = fetch(&url, "HEAD", None)
            .await
            .map_err(|e| err(format!("{e:?}")))?;
        let headers =
            js_sys::Reflect::get(&head, &"headers".into()).map_err(|e| err(format!("{e:?}")))?;
        let len = call(&headers, "get", &["content-length".into()])
            .ok()
            .and_then(|v| v.as_string())
            .and_then(|v| v.parse::<u64>().ok())
            .ok_or_else(|| err("the server sent no Content-Length"))?;
        let reader = Fetch { url, len };
        let block = block.map_or(1 << 20, |b| b as usize);
        let remote = hdf5::remote::RemoteFile::open(reader, block)
            .await
            .map_err(err)?;
        Ok(Self {
            remote: Rc::new(remote),
        })
    }

    /// Range requests made so far (each fetching all one step needed).
    #[wasm_bindgen(js_name = roundTrips)]
    pub fn round_trips(&self) -> usize {
        self.remote.round_trips()
    }

    /// Bytes downloaded so far.
    #[wasm_bindgen(js_name = bytesFetched)]
    pub fn bytes_fetched(&self) -> f64 {
        self.remote.bytes_fetched() as f64
    }

    /// As [`H5File::keys`].
    pub fn keys(&self, path: String) -> Promise {
        self.run(move |f| {
            keys(f, &path).map(|k| k.into_iter().map(JsValue::from).collect::<Array>().into())
        })
    }

    /// As [`H5File::kind`].
    pub fn kind(&self, path: String) -> Promise {
        self.run(move |f| kind(f, &path).map(JsValue::from))
    }

    /// As [`H5File::shape`].
    pub fn shape(&self, path: String) -> Promise {
        self.run(move |f| {
            let c = container(f, &path, None)?;
            Ok(c.shape()
                .into_iter()
                .map(|d| JsValue::from(d as f64))
                .collect::<Array>()
                .into())
        })
    }

    /// As [`H5File::dtype`].
    pub fn dtype(&self, path: String) -> Promise {
        self.run(move |f| dtype_name(&container(f, &path, None)?).map(JsValue::from))
    }

    /// As [`H5File::read`].
    pub fn read(&self, path: String) -> Promise {
        self.run(move |f| read_js(&container(f, &path, None)?, None))
    }

    /// As [`H5File::read_slice`]: only the chunks holding the slice are
    /// downloaded.
    #[wasm_bindgen(js_name = readSlice)]
    pub fn read_slice(&self, path: String, start: Vec<f64>, count: Vec<f64>) -> Promise {
        self.run(move |f| read_js(&container(f, &path, None)?, Some(hyperslab(&start, &count))))
    }

    /// As [`H5File::attr_names`].
    #[wasm_bindgen(js_name = attrNames)]
    pub fn attr_names(&self, path: String) -> Promise {
        self.run(move |f| {
            attr_names(f, &path).map(|k| k.into_iter().map(JsValue::from).collect::<Array>().into())
        })
    }

    /// As [`H5File::read_attr`].
    #[wasm_bindgen(js_name = readAttr)]
    pub fn read_attr(&self, path: String, name: String) -> Promise {
        self.run(move |f| read_js(&container(f, &path, Some(&name))?, None))
    }

    fn run(&self, op: impl Fn(&File) -> hdf5::Result<JsValue> + 'static) -> Promise {
        let remote = self.remote.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            remote
                .read(|f| op(f))
                .await
                .map_err(|e| JsValue::from(JsError::new(&e.to_string())))
        })
    }
}
