import type { BidOption, Player, Round, Situation } from "./types";

export const signed = (value: number) =>
  (value > 0 ? "+" : value < 0 ? "−" : "") + Math.abs(value).toFixed(1);

export const pct = (p: number) => (p > 0 && p < 0.005 ? "<1%" : `${Math.round(p * 100)}%`);

export const tone = (value: number) => (value > 0 ? "gain" : value < 0 ? "loss" : "");

export const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

export function ordinal(n: number) {
  const suffix = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return n + (suffix[(v - 20) % 10] || suffix[v] || suffix[0]);
}

export const option = (sit: Situation, bid: number): BidOption =>
  sit.options.find((o) => o.bid === bid)!;

export const bestOption = (sit: Situation) => option(sit, sit.best_bid);

/** Whether every player has bid on this hand. */
export const complete = (round: Round | undefined, players: Player[]): round is Round =>
  !!round && players.every((_, p) => round.bids[p] !== undefined);

/** Each player's total expected points over the finished hands. */
export const totals = (rounds: Round[], players: Player[]) =>
  players.map((_, p) =>
    rounds.reduce(
      (sum, round) =>
        complete(round, players) ? sum + option(round.sit, round.bids[p]).expected_points : sum,
      0,
    ),
  );

export function load<T>(key: string, fallback: T): T {
  try {
    const value = localStorage.getItem(key);
    return value === null ? fallback : (JSON.parse(value) as T);
  } catch {
    return fallback;
  }
}

export function save(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Private windows may refuse storage; the game works without it.
  }
}
