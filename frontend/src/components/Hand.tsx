import { motion } from "motion/react";
import { cardImage, cardName, sortHand } from "../cards";
import type { GameState } from "../types";

export function Hand({ game, onPlay }: { game: GameState; onPlay: (cardId: string) => void }) {
  const cards = sortHand(game.hand, game.trump?.suit);
  const playing = game.awaiting === "user_play";
  const middle = (cards.length - 1) / 2;

  if (game.blind) {
    return (
      <div className="hand" key={game.round}>
        {game.players[0].num_cards > 0 && (
          <button type="button" className="hand-card blind-card"
            disabled={!playing} aria-label="Play your hidden card"
            onClick={() => onPlay("blind")}>
            <span aria-hidden="true">?</span>
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="hand" key={game.round} style={{ "--n": cards.length } as React.CSSProperties}>
      {cards.map((card, i) => {
        const playable = game.playable.includes(card.id);
        const offset = i - middle;
        const restY = offset * offset * 0.6 - (playable ? 14 : 0);
        return (
          <motion.button
            key={card.id}
            type="button"
            layout="position"
            className={`hand-card ${playing && !playable ? "hand-card-blocked" : ""}`}
            style={{ zIndex: i + 1 }}
            disabled={!playable}
            aria-label={cardName(card)}
            onClick={() => onPlay(card.id)}
            initial={{ y: 70, opacity: 0, rotate: 0 }}
            animate={{ y: restY, opacity: 1, rotate: offset * 1.6 }}
            whileHover={playable ? { y: restY - 18 } : undefined}
            whileFocus={playable ? { y: restY - 18 } : undefined}
            transition={{
              type: "spring",
              stiffness: 320,
              damping: 30,
              opacity: { delay: i * 0.035, duration: 0.2 },
            }}
          >
            <img src={cardImage(card)} alt="" draggable={false} />
          </motion.button>
        );
      })}
    </div>
  );
}
