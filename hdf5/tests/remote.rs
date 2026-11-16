//! Reading over async range requests (`hdf5::remote`): values match, a
//! handful of round-trips, and a partial read fetches a small part.

use std::cell::Cell;
use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll};

use hdf5::remote::{AsyncRangeRead, RemoteFile};
use ndarray::{s, Array2};

/// Serves a byte buffer; each read first answers "not yet" once, like a
/// network would, so reads really are concurrent futures.
struct Server {
    bytes: Vec<u8>,
    requests: Cell<usize>,
}

impl AsyncRangeRead for Server {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }
    fn read_range(
        &self,
        offset: u64,
        len: usize,
    ) -> impl Future<Output = std::io::Result<Vec<u8>>> {
        self.requests.set(self.requests.get() + 1);
        let bytes = self.bytes[offset as usize..offset as usize + len].to_vec();
        let mut waited = false;
        std::future::poll_fn(move |cx| {
            if waited {
                Poll::Ready(Ok(bytes.clone()))
            } else {
                waited = true;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        })
    }
}

/// The smallest executor: poll until ready.
fn block_on<F: Future>(f: F) -> F::Output {
    let mut f = pin!(f);
    let mut cx = Context::from_waker(std::task::Waker::noop());
    loop {
        if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
    }
}

fn data() -> Array2<f32> {
    Array2::from_shape_fn((200_000, 16), |(i, j)| ((i * 16 + j) % 10_007) as f32)
}

fn file_bytes() -> Vec<u8> {
    let f = hdf5::File::create_in_memory().unwrap();
    f.create_group("acq").unwrap();
    f.create_group("meta").unwrap();
    f.new_dataset_builder()
        .chunk((5000, 16))
        .shuffle()
        .deflate(3)
        .with_data(&data())
        .create("acq/data")
        .unwrap();
    for i in 0..50 {
        f.create_group(&format!("meta/g{i}")).unwrap();
    }
    f.to_bytes().unwrap()
}

#[test]
fn remote_reads_fetch_what_they_need() {
    let bytes = file_bytes();
    let total = bytes.len() as u64;
    let server = Server {
        bytes,
        requests: Cell::new(0),
    };
    block_on(async {
        let remote = RemoteFile::open(server, 16 << 10).await.unwrap();
        let opened = remote.round_trips();
        assert!(opened <= 6, "open took {opened} round-trips");
        eprintln!("open: {opened} round-trips");
        let shape = remote
            .read(|f| Ok(f.dataset("acq/data")?.shape()))
            .await
            .unwrap();
        assert_eq!(shape, [200_000, 16]);
        let names = remote
            .read(|f| f.group("meta")?.member_names())
            .await
            .unwrap();
        assert_eq!(names.len(), 50);
        // a slice in the middle: its chunks, not the dataset
        let before = remote.bytes_fetched();
        let part: Array2<f32> = remote
            .read(|f| f.dataset("acq/data")?.read_slice(s![100_000..100_010, ..]))
            .await
            .unwrap();
        assert_eq!(part, data().slice(s![100_000..100_010, ..]));
        let used = remote.bytes_fetched() - before;
        assert!(used < total / 10, "slice fetched {used} of {total} bytes");
        assert!(
            remote.round_trips() < 12,
            "{} round-trips",
            remote.round_trips()
        );
        // and the whole thing
        let all: Array2<f32> = remote
            .read(|f| f.dataset("acq/data")?.read_2d())
            .await
            .unwrap();
        assert_eq!(all, data());
    });
}
