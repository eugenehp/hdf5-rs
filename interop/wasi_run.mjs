// Run a wasm32-wasip1 binary (e.g. a cargo test harness) under Node's WASI:
//   CARGO_TARGET_WASM32_WASIP1_RUNNER="node interop/wasi_run.mjs" \
//     cargo test --target wasm32-wasip1
// The system temp dir is preopened as /tmp, the current dir as ".", and the
// current dir's parent at its absolute path. WASI has no `temp_dir`, so tests
// point `tempfile` at /tmp themselves.
import { readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname } from "node:path";
import { WASI } from "node:wasi";
import { argv, cwd, env, exit } from "node:process";

const [file, ...args] = argv.slice(2);
const wasi = new WASI({
  version: "preview1",
  args: [file, ...args],
  env: { ...env, TMPDIR: "/tmp" },
  // the cwd's parent (the workspace, when cargo runs a package's tests) also
  // at its own absolute path, for CARGO_MANIFEST_DIR fixtures
  preopens: { "/tmp": tmpdir(), ".": ".", [dirname(cwd())]: dirname(cwd()) },
  returnOnExit: true,
});
const module = await WebAssembly.compile(await readFile(file));
const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
exit(wasi.start(instance));
