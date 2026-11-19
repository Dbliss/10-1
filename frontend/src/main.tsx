import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { MotionConfig } from "motion/react";
import "@fontsource-variable/fraunces";
import "@fontsource-variable/manrope";
import "./styles.css";
import App from "./App";
import BidGame from "./bidgame/BidGame";

// The server (or, on GitHub Pages, a copy of this page) answers /bid-game;
// the path, after the site's base, picks the screen.
const isBidGame = window.location.pathname.replace(/\/+$/, "").endsWith("/bid-game");

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <MotionConfig reducedMotion="user">{isBidGame ? <BidGame /> : <App />}</MotionConfig>
  </StrictMode>,
);
