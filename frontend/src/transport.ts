/**
 * How API requests travel. Local development sends them to website.py; the
 * GitHub Pages build (VITE_ENGINE=browser) runs the same Python API in a Web
 * Worker with Pyodide, so no server is needed.
 */

export interface Reply {
  status: number;
  data: any;
}

export const inBrowser = import.meta.env.VITE_ENGINE === "browser";

const SERVER_DOWN = inBrowser
  ? "The game engine failed to start in this browser."
  : "Can't reach the game server. Is website.py running?";

type Pending = { resolve: (reply: Reply) => void; reject: (error: Error) => void };

let worker: Worker | null = null;
let nextId = 0;
const pending = new Map<number, Pending>();

function engine(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL("./worker/engine.ts", import.meta.url), { type: "module" });
  worker.onmessage = (event: MessageEvent) => {
    const { id, status, data, error } = event.data;
    const waiting = pending.get(id);
    if (!waiting) return;
    pending.delete(id);
    if (error) waiting.reject(new Error(`${SERVER_DOWN} ${error}`));
    else waiting.resolve({ status, data });
  };
  worker.onerror = (event) => {
    for (const waiting of pending.values()) waiting.reject(new Error(`${SERVER_DOWN} ${event.message ?? ""}`));
    pending.clear();
  };
  worker.postMessage({ type: "start", base: import.meta.env.BASE_URL });
  return worker;
}

/** Send one API request (path starting /api/) and return its status and JSON reply. */
export async function send(path: string, method = "GET", body?: unknown): Promise<Reply> {
  if (inBrowser) {
    const id = nextId++;
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject });
      engine().postMessage({ type: "request", id, method, path, body: body ?? null });
    });
  }
  let response: Response;
  try {
    response = await fetch(path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new Error(SERVER_DOWN);
  }
  const data = await response.json().catch(() => {
    throw new Error(`The game server sent an unreadable response. ${SERVER_DOWN}`);
  });
  return { status: response.status, data };
}
