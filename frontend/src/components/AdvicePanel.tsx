import { useEffect, useState } from "react";
import { api } from "../api";
import { cardName } from "../cards";
import type { Advice, GameState } from "../types";

const signed = (value: number) => (value > 0 ? `+${value.toFixed(1)}` : value.toFixed(1));

function BidRows({ advice }: { advice: Extract<Advice, { kind: "bid" }> }) {
  return (
    <>
      <div className="advice-row advice-head">
        <span>Bid</span>
        <span>Chance to make</span>
        <span>Avg points</span>
      </div>
      {advice.options.map((option) => (
        <div
          key={option.bid}
          className={`advice-row ${option.bid === advice.recommended ? "advice-best" : ""}`}
        >
          <span>{option.bid}</span>
          <span>{Math.round(option.make_chance * 100)}%</span>
          <span>{signed(option.expected_points)}</span>
        </div>
      ))}
    </>
  );
}

function CardRows({ advice }: { advice: Extract<Advice, { kind: "card" }> }) {
  return (
    <>
      <div className="advice-row advice-row-card advice-head">
        <span>Card</span>
        <span>Avg points</span>
      </div>
      {advice.options.map((option) => (
        <div
          key={option.card.id}
          className={`advice-row advice-row-card ${
            option.card.id === advice.recommended ? "advice-best" : ""
          }`}
        >
          <span>{cardName(option.card)}</span>
          <span>{signed(option.expected_points)}</span>
        </div>
      ))}
    </>
  );
}

/** Shows the bot's own analysis of the decision the user is facing. */
export function AdvicePanel({ game }: { game: GameState }) {
  const [advice, setAdvice] = useState<Advice | null>(null);
  const [thinking, setThinking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const yourTurn = game.awaiting === "user_bid" || game.awaiting === "user_play";
  const decision = `${game.round}-${game.awaiting}-${game.hand.length}-${game.trick.length}`;

  useEffect(() => {
    setAdvice(null);
    setError(null);
  }, [decision]);

  if (!yourTurn) return null;

  const ask = async () => {
    setThinking(true);
    setError(null);
    try {
      setAdvice(await api.advice());
    } catch (err) {
      setError(err instanceof Error ? err.message : "The bot couldn't answer.");
    } finally {
      setThinking(false);
    }
  };

  return (
    <section className="advice" aria-live="polite">
      {advice === null ? (
        <button type="button" className="btn btn-quiet btn-wide" onClick={ask} disabled={thinking}>
          {thinking ? "Thinking…" : "Ask the bot"}
        </button>
      ) : advice.kind === "bid" ? (
        <BidRows advice={advice} />
      ) : (
        <CardRows advice={advice} />
      )}
      {error && <p className="advice-error">{error}</p>}
    </section>
  );
}
