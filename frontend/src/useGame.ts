import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { Awaiting, GameState } from "./types";

/** How long to show each automatic step before moving on, in milliseconds. */
const AUTO_ADVANCE_DELAY: Partial<Record<Awaiting, number>> = {
  bot: 0,
  collect_trick: 1600,
};

/**
 * The shortest time each bot move stays on screen before the next one, so
 * instant decisions still appear one at a time. The bot thinks during this
 * time, so slower decisions are not delayed further.
 */
const BOT_MOVE_MIN_MS = 250;

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export function useGame() {
  const [game, setGame] = useState<GameState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const inFlight = useRef(false);
  const shownAt = useRef(0);

  const run = useCallback(async (action: () => Promise<GameState | null>) => {
    if (inFlight.current) return;
    inFlight.current = true;
    try {
      setGame(await action());
      shownAt.current = performance.now();
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      inFlight.current = false;
    }
  }, []);

  useEffect(() => {
    run(api.get).finally(() => setLoading(false));
  }, [run]);

  useEffect(() => {
    if (!game) return;
    const delay = AUTO_ADVANCE_DELAY[game.awaiting];
    if (delay === undefined) return;
    const step =
      game.awaiting === "bot"
        ? async () => {
            const next = await api.advance();
            await sleep(BOT_MOVE_MIN_MS - (performance.now() - shownAt.current));
            return next;
          }
        : api.advance;
    const timer = setTimeout(() => run(step), delay);
    return () => clearTimeout(timer);
  }, [game, run]);

  return {
    game,
    loading,
    error,
    retry: () => run(api.get),
    newGame: (players: number, handSize: number) => run(() => api.create(players, handSize)),
    endGame: () => run(api.end),
    bid: (value: number) => run(() => api.bid(value)),
    play: (cardId: string) => run(() => api.play(cardId)),
    nextRound: () => run(api.advance),
  };
}
