import { copyFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

/** GitHub Pages has no server to route /bid-game, so give it its own copy of the page. */
const bidGamePage = (): Plugin => ({
  name: "bid-game-page",
  apply: "build",
  closeBundle() {
    const dist = resolve(__dirname, "dist");
    mkdirSync(resolve(dist, "bid-game"), { recursive: true });
    copyFileSync(resolve(dist, "index.html"), resolve(dist, "bid-game", "index.html"));
  },
});

// `vite build --mode pages` (npm run build:pages) makes the GitHub Pages site:
// served from /10-1/, with the game engine running in the browser (.env.pages).
// Every other mode talks to website.py and is served from /.
export default defineConfig(({ mode }) => ({
  base: mode === "pages" ? "/10-1/" : "/",
  plugins: [react(), ...(mode === "pages" ? [bidGamePage()] : [])],
  worker: { format: "es" },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:5000",
    },
  },
}));
