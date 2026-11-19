import type { Advice, GameReview, GameState } from "./types";
import { send } from "./transport";

type Game = GameState | null;

async function request<T = Game>(path: string, method = "GET", body?: unknown): Promise<T> {
  const { status, data } = await send(`/api/game${path}`, method, body);
  if (status >= 400) {
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
  review: () => request<GameReview>("/review"),
};
