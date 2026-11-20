import { useEffect, useState } from "react";
import { AnimatePresence } from "motion/react";
import type { GameState } from "../types";
import { useMediaQuery, useRevealOnScrollUp } from "../useMediaQuery";
import { AdvicePanel } from "./AdvicePanel";
import { GameOverDialog, RoundSummaryDialog, Sheet } from "./Dialogs";
import { GameReviewScreen, useReview } from "./GameReview";
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
  const over = game.phase === "game_over";
  const { review, error: reviewError } = useReview(over);
  const [reviewing, setReviewing] = useState(false);
  useEffect(() => {
    if (!over) setReviewing(false);
  }, [over]);

  // Below this width the sidebar no longer fits: everything must fit on one
  // screen, so scores and advice move into sheets behind buttons.
  const compact = useMediaQuery("(max-width: 1040px)");
  const [headerShown, revealHeader] = useRevealOnScrollUp(compact);
  const [sheet, setSheet] = useState<"scores" | "advice" | null>(null);
  const yourTurn = game.awaiting === "user_bid" || game.awaiting === "user_play";

  useEffect(() => {
    if (!compact || (!yourTurn && sheet === "advice")) setSheet(null);
  }, [compact, yourTurn, sheet]);

  const sheetActions = compact && (
    <>
      <button type="button" className="btn btn-quiet btn-small" onClick={() => setSheet("scores")}>
        Scores
      </button>
      {yourTurn && (
        <button type="button" className="btn btn-quiet btn-small btn-hint" onClick={() => setSheet("advice")}>
          Hint
        </button>
      )}
    </>
  );

  return (
    <div className={`game ${compact ? "game-compact" : ""}`}>
      <header
        className={`topbar ${compact ? "topbar-auto" : ""} ${headerShown ? "topbar-shown" : ""}`}
        onPointerDown={compact ? revealHeader : undefined}
      >
        <span className="brand">10-to-1</span>
        <RoundPips game={game} />
        <span className="round-compact">
          Round {game.round}/{game.total_rounds}
        </span>
        <QuitButton onQuit={onQuit} />
      </header>

      <main className="play">
        <StatusBar game={game} actions={sheetActions} />
        <Table game={game} onBid={onBid} />
        <Hand game={game} onPlay={onPlay} />
      </main>

      {!compact && (
        <aside className="sidebar">
          <AdvicePanel game={game} />
          <Scoreboard game={game} />
        </aside>
      )}

      <AnimatePresence>
        {sheet === "scores" && (
          <Sheet key="scores" title="Scores" onClose={() => setSheet(null)}>
            <Scoreboard game={game} heading={false} />
          </Sheet>
        )}
        {sheet === "advice" && (
          <Sheet key="advice" title="Bot's advice" onClose={() => setSheet(null)}>
            <AdvicePanel game={game} autoAsk />
          </Sheet>
        )}
      </AnimatePresence>

      {game.phase === "round_over" && <RoundSummaryDialog game={game} onNext={onNextRound} />}
      {over && !reviewing && (
        <GameOverDialog
          game={game}
          review={review}
          onReview={() => setReviewing(true)}
          onPlayAgain={onPlayAgain}
          onChangeSettings={onQuit}
        />
      )}
      {over && reviewing && (
        <GameReviewScreen review={review} error={reviewError} onClose={() => setReviewing(false)} />
      )}
    </div>
  );
}
