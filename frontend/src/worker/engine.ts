/**
 * The game engine for the browser build: Pyodide runs the project's Python
 * (app_api.py through browser_entry.py) and its C core compiled to
 * WebAssembly, inside this worker so bot thinking never blocks the page.
 *
 * Requests are answered one at a time, in order. Between requests the
 * worker analyses the user's decisions for the post-game review, one per
 * task, so a new request waits at most for one analysis.
 *
 * PYODIDE_VERSION must match the Emscripten version the core is built with
 * (ohhell/native build_wasm): Pyodide 314.0.8 uses Emscripten 5.0.3.
 */

const PYODIDE_VERSION = "314.0.8";
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

interface Entry {
  call(method: string, path: string, body: string): string;
  review_step(): boolean;
}

interface EngineRequest {
  id: number;
  method: string;
  path: string;
  body: unknown;
}

const scope = self as unknown as {
  postMessage(message: unknown): void;
  onmessage: ((event: MessageEvent) => void) | null;
};

let ready: Promise<Entry> | null = null;
const queue: EngineRequest[] = [];
let busy = false;
let reviewing = false;

async function boot(base: string): Promise<Entry> {
  const { loadPyodide } = await import(/* @vite-ignore */ `${PYODIDE_URL}pyodide.mjs`);
  const pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
  const manifest = await (await fetch(`${base}py/manifest.json`, { cache: "no-cache" })).json();
  const [, archive] = await Promise.all([
    pyodide.loadPackage("numpy"),
    fetch(`${base}py/${manifest.archive}`).then((r) => {
      if (!r.ok) throw new Error(`Could not download the game code (${r.status}).`);
      return r.arrayBuffer();
    }),
  ]);
  pyodide.unpackArchive(archive, "zip", { extractDir: "/home/pyodide/app" });
  pyodide.runPython("import sys; sys.path.insert(0, '/home/pyodide/app')");
  return pyodide.pyimport("browser_entry") as Entry;
}

async function pump() {
  if (busy || !ready) return;
  busy = true;
  try {
    const entry = await ready;
    while (queue.length) {
      const request = queue.shift()!;
      try {
        const body = request.body === null ? "" : JSON.stringify(request.body);
        const [status, data] = JSON.parse(entry.call(request.method, request.path, body));
        scope.postMessage({ id: request.id, status, data });
      } catch (err) {
        scope.postMessage({ id: request.id, error: String(err) });
      }
    }
  } catch (err) {
    // The engine failed to load: fail everything waiting.
    for (const request of queue.splice(0)) scope.postMessage({ id: request.id, error: String(err) });
  } finally {
    busy = false;
  }
  scheduleReview();
}

function scheduleReview() {
  if (reviewing) return;
  reviewing = true;
  setTimeout(async () => {
    reviewing = false;
    if (busy || queue.length || !ready) return;
    try {
      if ((await ready).review_step()) scheduleReview();
    } catch {
      // A broken engine is reported through requests.
    }
  }, 0);
}

scope.onmessage = (event: MessageEvent) => {
  const message = event.data;
  if (message.type === "start") {
    ready ??= boot(message.base);
    void pump();
  } else if (message.type === "request") {
    queue.push(message as EngineRequest);
    void pump();
  }
};
