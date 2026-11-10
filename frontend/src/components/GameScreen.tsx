import { useEffect, useState } from "react";
import type { GameState } from "../types";
import { AdvicePanel } from "./AdvicePanel";
import { GameOverDialog, RoundSummaryDialog } from "./Dialogs";
import { Hand } from "./Hand";
import { Scoreboard } from "./Scoreboard";
import { StatusBar } from "./StatusBar";
import { Table } from "./Table";

interface GameScreenProps {
  game: GameState;
  onBid: (bid: number) => void;
  onPlay: (cardId: string) => void;
  onNextRound: () => void;
  onPlayAgain: () => void;
  onQuit: () => void;
}

function QuitButton({ onQuit }: { onQuit: () => void }) {
  const [confirming, setConfirming] = useState(false);

  useEffect(() => {
    if (!confirming) return;
    const timer = setTimeout(() => setConfirming(false), 3000);
    return () => clearTimeout(timer);
  }, [confirming]);

  return (
    <button
      type="button"
      className={`btn btn-quiet ${confirming ? "btn-danger" : ""}`}
      onClick={() => (confirming ? onQuit() : setConfirming(true))}
    >
      {confirming ? "Abandon this game?" : "Quit"}
    </button>
  );
}

function RoundPips({ game }: { game: GameState }) {
  const sizes = game.round_sizes;
  return (
    <ol className="pips" aria-label={`Round ${game.round} of ${game.total_rounds}`}>
      {sizes.map((size, i) => {
        const round = i + 1;
        const state = round === game.round ? "now" : round < game.round ? "done" : "next";
        return (
          <li key={round} className={`pip pip-${state}`} title={size === 0 ? "Blind round" : undefined} aria-hidden="true">
            {size}
          </li>
        );
      })}
    </ol>
  );
}

export function GameScreen({
  game,
  onBid,
  onPlay,
  onNextRound,
  onPlayAgain,
  onQuit,
}: GameScreenProps) {
  return (
    <div className="game">
      <header className="topbar">
        <span className="brand">Oh Hell</span>
        <RoundPips game={game} />
        <span className="round-compact">
          Round {game.round}/{game.total_rounds}
        </span>
        <QuitButton onQuit={onQuit} />
      </header>

      <main className="play">
        <StatusBar game={game} />
        <Table game={game} onBid={onBid} />
        <Hand game={game} onPlay={onPlay} />
      </main>

      <aside className="sidebar">
        <Scoreboard game={game} />
        <AdvicePanel game={game} />
      </aside>

      {game.phase === "round_over" && <RoundSummaryDialog game={game} onNext={onNextRound} />}
      {game.phase === "game_over" && (
        <GameOverDialog game={game} onPlayAgain={onPlayAgain} onChangeSettings={onQuit} />
      )}
    </div>
  );
}
