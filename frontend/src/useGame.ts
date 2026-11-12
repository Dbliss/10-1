import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { Awaiting, GameState } from "./types";

/** How long to show each automatic step before moving on, in milliseconds. */
const AUTO_ADVANCE_DELAY: Partial<Record<Awaiting, number>> = {
  bot: 750,
  collect_trick: 1600,
};

export function useGame() {
  const [game, setGame] = useState<GameState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const inFlight = useRef(false);

  const run = useCallback(async (action: () => Promise<GameState | null>) => {
    if (inFlight.current) return;
    inFlight.current = true;
    try {
      setGame(await action());
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
    const timer = setTimeout(() => run(api.advance), delay);
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
