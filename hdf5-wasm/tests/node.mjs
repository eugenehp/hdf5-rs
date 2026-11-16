// End-to-end check of the npm package under Node:
//   wasm-pack build hdf5-wasm --target nodejs --out-dir pkg-node --out-name hdf5
//   node hdf5-wasm/tests/node.mjs [some-big-file.h5]
// In memory: write, read back, slices, attributes. Over HTTP: a local server
// with Range support, read through H5Remote, checking only a little is fetched.
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import http from "node:http";
import fs from "node:fs";

const require = createRequire(import.meta.url);
const { H5File, H5Remote } = require("../pkg-node/hdf5.js");

// --- in memory ---------------------------------------------------------------
const f = H5File.create();
f.createGroup("acq");
const rows = 1000, cols = 6;
const signal = Float32Array.from({ length: rows * cols }, (_, i) => Math.sin(i / 10));
f.writeDataset("acq/signal", signal, [rows, cols], [100, cols], 4);
f.writeDataset("acq/counts", Int16Array.from([1, -2, 3, -4]), [2, 2]);
f.writeDataset("labels", ["α", "beta", "γ"], [3]);
const bytes = f.toBytes();

const g = H5File.fromBytes(bytes);
assert.deepEqual(g.keys("/"), ["acq", "labels"]);
assert.deepEqual(g.keys("acq"), ["counts", "signal"]);
assert.equal(g.kind("acq"), "group");
assert.equal(g.kind("acq/signal"), "dataset");
assert.deepEqual(g.shape("acq/signal"), [rows, cols]);
assert.equal(g.dtype("acq/signal"), "float32");
assert.equal(g.dtype("acq/counts"), "int16");
assert.deepEqual(g.read("acq/signal"), signal);
assert.deepEqual(Array.from(g.read("acq/counts")), [1, -2, 3, -4]);
assert.deepEqual(g.read("labels"), ["α", "beta", "γ"]);
const part = g.readSlice("acq/signal", [500, 2], [3, 2]);
assert.deepEqual(Array.from(part), [500, 501, 502].flatMap((r) => [signal[r * cols + 2], signal[r * cols + 3]]));
console.log("in-memory: ok");

// --- over HTTP ------------------------------------------------------------------
let path = process.argv[2];
if (!path) {
  path = "/tmp/hdf5-wasm-node-test.h5";
  const big = H5File.create();
  const n = 2_000_000;
  big.writeDataset("x", Float64Array.from({ length: n }, (_, i) => i * 0.5), [n], [10_000], 1);
  fs.writeFileSync(path, big.toBytes());
}
const size = fs.statSync(path).size;
const server = http.createServer((req, res) => {
  const range = /bytes=(\d+)-(\d+)/.exec(req.headers.range ?? "");
  if (req.method === "HEAD" || !range) {
    res.writeHead(200, { "content-length": size, "accept-ranges": "bytes" });
    return req.method === "HEAD" ? res.end() : fs.createReadStream(path).pipe(res);
  }
  const [start, end] = [Number(range[1]), Number(range[2])];
  res.writeHead(206, { "content-length": end - start + 1, "content-range": `bytes ${start}-${end}/${size}` });
  fs.createReadStream(path, { start, end }).pipe(res);
});
await new Promise((ok) => server.listen(0, ok));
const url = `http://127.0.0.1:${server.address().port}/file.h5`;
try {
  const r = await H5Remote.open(url, 1 << 16);
  const names = await r.keys("/");
  console.log("remote keys:", names.join(", "), `(${r.roundTrips()} round-trips)`);
  if (names.includes("gzip_cols")) {
    // the h5py-written benchmark file (interop/make_bench.py)
    const shape = await r.shape("gzip_cols");
    const fromGzip = await r.readSlice("gzip_cols", [3_000_000, 0], [64, shape[1]]);
    const fromContig = await r.readSlice("contig", [3_000_000, 0], [64, shape[1]]);
    assert.deepEqual(fromGzip, fromContig);
  } else {
    const x = await r.readSlice("x", [1_234_567], [5]);
    assert.deepEqual(Array.from(x), [0, 1, 2, 3, 4].map((i) => (1_234_567 + i) * 0.5));
  }
  const fetched = r.bytesFetched();
  console.log(`remote: ok — fetched ${(fetched / 1e6).toFixed(2)} MB of ${(size / 1e6).toFixed(1)} MB in ${r.roundTrips()} round-trips`);
  assert.ok(fetched < size / 10);
} finally {
  server.close();
}
