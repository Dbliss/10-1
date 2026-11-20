import { useEffect, type ReactNode } from "react";
import { motion } from "motion/react";
import type { GameReview, GameState } from "../types";
import { ReviewTeaser } from "./GameReview";

function Dialog({ title, children }: { title: string; children: ReactNode }) {
  return (
    <motion.div className="dialog-backdrop" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      <motion.div
        className="dialog"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        initial={{ y: 24, scale: 0.97 }}
        animate={{ y: 0, scale: 1 }}
      >
        <h2>{title}</h2>
        {children}
      </motion.div>
    </motion.div>
  );
}

/** A panel that slides up from the bottom of a small screen. Wrap in AnimatePresence. */
export function Sheet({ title, onClose, children }: { title: string; onClose: () => void; children: ReactNode }) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <motion.div
      className="sheet-backdrop"
      onClick={onClose}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.section
        className="sheet"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onClick={(event) => event.stopPropagation()}
        initial={{ y: "100%" }}
        animate={{ y: 0 }}
        exit={{ y: "100%" }}
        transition={{ type: "spring", stiffness: 420, damping: 40 }}
      >
        <header className="sheet-head">
          <h2>{title}</h2>
          <button type="button" className="btn btn-quiet" onClick={onClose} autoFocus>
            Close
          </button>
        </header>
        <div className="sheet-body">{children}</div>
      </motion.section>
    </motion.div>
  );
}

export function RoundSummaryDialog({ game, onNext }: { game: GameState; onNext: () => void }) {
  const last = game.history[game.history.length - 1];
  const nextSize = game.round_sizes[game.round];

  return (
    <Dialog title={`Round ${last.round} results`}>
      <table className="results">
        <thead>
          <tr>
            <th scope="col">Player</th>
            <th scope="col">Bid</th>
            <th scope="col">Won</th>
            <th scope="col">Points</th>
            <th scope="col">Total</th>
          </tr>
        </thead>
        <tbody>
          {game.players.map((player, seat) => {
            const result = last.results[seat];
            const tone = result.points > 0 ? "gain" : result.points < 0 ? "loss" : "";
            return (
              <tr key={player.name} className={player.is_human ? "row-you" : ""}>
                <th scope="row">{player.name}</th>
                <td>{result.bid}</td>
                <td>{result.tricks}</td>
                <td className={tone}>{result.points > 0 ? `+${result.points}` : result.points}</td>
                <td>{result.score}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <button type="button" className="btn btn-primary btn-wide" onClick={onNext} autoFocus>
        {nextSize === 0 ? "Deal blind round" : `Deal ${nextSize} ${nextSize === 1 ? "card" : "cards"}`}
      </button>
    </Dialog>
  );
}

function winnerTitle(game: GameState) {
  const best = Math.max(...game.players.map((player) => player.score));
  const winners = game.players.filter((player) => player.score === best);
  if (winners.length > 1) return `Tied: ${winners.map((player) => player.name).join(", ")}`;
  return winners[0].is_human ? "You win" : `${winners[0].name} wins`;
}

interface GameOverDialogProps {
  game: GameState;
  review: GameReview | null;
  onReview: () => void;
  onPlayAgain: () => void;
  onChangeSettings: () => void;
}

export function GameOverDialog({
  game,
  review,
  onReview,
  onPlayAgain,
  onChangeSettings,
}: GameOverDialogProps) {
  const standings = [...game.players].sort((a, b) => b.score - a.score);

  return (
    <Dialog title={winnerTitle(game)}>
      <ol className="standings">
        {standings.map((player) => (
          <li key={player.name} className={player.is_human ? "row-you" : ""}>
            <span>{player.name}</span>
            <strong>{player.score}</strong>
          </li>
        ))}
      </ol>
      <ReviewTeaser review={review} />
      <button type="button" className="btn btn-primary btn-wide" onClick={onReview} autoFocus>
        Review your game
      </button>
      <button type="button" className="btn btn-quiet btn-wide" onClick={onPlayAgain}>
        Play again
      </button>
      <button type="button" className="btn btn-quiet btn-wide" onClick={onChangeSettings}>
        Change settings
      </button>
    </Dialog>
  );
}
