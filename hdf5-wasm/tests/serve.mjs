// Static file server with HTTP Range support, for the browser test:
//   node hdf5-wasm/tests/serve.mjs <port> [big-file.h5]
// Serves the hdf5-wasm directory; /big.h5 maps to the given file.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const [port, big] = [Number(process.argv[2] ?? 8000), process.argv[3]];
const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm" };

http
  .createServer((req, res) => {
    const url = new URL(req.url, "http://x");
    const file = url.pathname === "/big.h5" && big ? big : path.join(root, decodeURIComponent(url.pathname));
    if (!file.startsWith(root) && file !== big) return res.writeHead(403).end();
    fs.stat(file, (e, st) => {
      if (e || !st.isFile()) return res.writeHead(404).end();
      const head = { "content-type": types[path.extname(file)] ?? "application/octet-stream", "accept-ranges": "bytes" };
      const range = /bytes=(\d+)-(\d+)/.exec(req.headers.range ?? "");
      if (req.method === "HEAD") return res.writeHead(200, { ...head, "content-length": st.size }).end();
      if (!range) {
        res.writeHead(200, { ...head, "content-length": st.size });
        return fs.createReadStream(file).pipe(res);
      }
      const [start, end] = [Number(range[1]), Math.min(Number(range[2]), st.size - 1)];
      res.writeHead(206, { ...head, "content-length": end - start + 1, "content-range": `bytes ${start}-${end}/${st.size}` });
      fs.createReadStream(file, { start, end }).pipe(res);
    });
  })
  .listen(port, () => console.log(`serving ${root} on :${port}`));
