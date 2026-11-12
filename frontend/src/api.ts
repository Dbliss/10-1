import type { Advice, GameState } from "./types";

type Game = GameState | null;

async function request<T = Game>(path: string, method = "GET", body?: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api/game${path}`, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new Error("Can't reach the game server. Is website.py running?");
  }

  const data = await response.json().catch(() => {
    throw new Error("The game server sent an unreadable response. Is website.py running?");
  });
  if (!response.ok) {
    throw new Error(data?.error ?? "The game server rejected that action.");
  }
  return data;
}

export const api = {
  get: () => request(""),
  create: (players: number, handSize: number) =>
    request("", "POST", { players, hand_size: handSize }),
  end: () => request("", "DELETE"),
  bid: (bid: number) => request("/bid", "POST", { bid }),
  play: (card: string) => request("/play", "POST", { card }),
  advance: () => request("/advance", "POST"),
  advice: () => request<Advice>("/advice"),
};
