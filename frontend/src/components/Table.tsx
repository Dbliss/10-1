import { useLayoutEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { cardImage, cardName } from "../cards";
import type { GameState, Player } from "../types";

/** Seat 0 (the user) sits at the bottom; turn order runs clockwise from there. */
const seatAngle = (seat: number, seats: number) => Math.PI / 2 + (seat * 2 * Math.PI) / seats;

function useElementSize() {
  const ref = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useLayoutEffect(() => {
    const element = ref.current!;
    const observer = new ResizeObserver(() =>
      setSize({ width: element.clientWidth, height: element.clientHeight }),
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return [ref, size] as const;
}

function SeatPlate({ player, seat, game }: { player: Player; seat: number; game: GameState }) {
  const angle = seatAngle(seat, game.players.length);
  const isActive = game.current_player === seat;
  const isDealer = game.dealer === seat;
  const isWinner = game.trick_winner === seat;

  let bidTone = "";
  if (player.bid !== null && player.tricks_won > player.bid) bidTone = "bid-over";
  else if (player.bid !== null && player.tricks_won === player.bid) bidTone = "bid-met";

  return (
    <div
      className={`seat ${isActive ? "seat-active" : ""} ${isWinner ? "seat-winner" : ""}`}
      style={{
        left: `${50 + Math.cos(angle) * 42}%`,
        top: `${50 + Math.sin(angle) * 40}%`,
      }}
    >
      {!player.is_human && (
        <div className="seat-cards" aria-label={`${player.num_cards} cards left`}>
          {Array.from({ length: player.num_cards }, (_, i) => (
            <span key={i} className="mini-back" />
          ))}
        </div>
      )}
      <div className="seat-plate">
        <div className="seat-head">
          <span className="seat-name">{player.name}</span>
          {isDealer && (
            <span className="dealer-chip" title="Dealer">
              D
            </span>
          )}
          <span className="seat-score">{player.score}</span>
        </div>
        <div className={`seat-bid ${bidTone}`}>
          {player.bid === null ? (
            <span className="seat-bid-empty">No bid yet</span>
          ) : (
            <motion.span
              key={player.bid}
              initial={{ scale: 1.5, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
            >
              Won {player.tricks_won} of {player.bid}
            </motion.span>
          )}
        </div>
      </div>
    </div>
  );
}

function TrickCards({ game, width, height }: { game: GameState; width: number; height: number }) {
  const seats = game.players.length;

  // Once a trick is collected the winner is cleared, but its cards still need
  // to know where to fly as they leave the table.
  const lastWinner = useRef(0);
  if (game.trick_winner !== null) lastWinner.current = game.trick_winner;

  const seatPoint = (seat: number) => {
    const angle = seatAngle(seat, seats);
    return { x: Math.cos(angle) * width * 0.42, y: Math.sin(angle) * height * 0.4 };
  };
  const collectTo = seatPoint(lastWinner.current);

  return (
    <AnimatePresence custom={collectTo}>
      {game.trick.map(({ seat, card }, order) => {
        const angle = seatAngle(seat, seats);
        const from = seatPoint(seat);
        const isWinner = game.trick_winner === seat;
        return (
          <motion.div
            key={card.id}
            className={`trick-card ${isWinner ? "trick-winner" : ""}`}
            style={{ zIndex: order + 1 }}
            custom={collectTo}
            initial={{ x: from.x, y: from.y, scale: 0.7, opacity: 0 }}
            animate={{
              x: Math.cos(angle) * Math.min(width * 0.14, 118),
              y: Math.sin(angle) * Math.min(height * 0.17, 74),
              scale: isWinner ? 1.08 : 1,
              rotate: ((seat * 37) % 11) - 5,
              opacity: 1,
            }}
            variants={{
              collected: (to: { x: number; y: number }) => ({
                x: to.x,
                y: to.y,
                scale: 0.3,
                opacity: 0,
                transition: { duration: 0.4, ease: "easeIn" },
              }),
            }}
            exit="collected"
            transition={{ type: "spring", stiffness: 260, damping: 26 }}
          >
            <img src={cardImage(card)} alt={`${game.players[seat].name}: ${cardName(card)}`} />
          </motion.div>
        );
      })}
    </AnimatePresence>
  );
}

function BidPanel({ game, onBid }: { game: GameState; onBid: (bid: number) => void }) {
  const options = Array.from({ length: game.hand_size + 1 }, (_, i) => i);
  // legal_bids empties as soon as the bid lands, while this panel is fading out.
  const blocked =
    game.awaiting === "user_bid"
      ? options.find((option) => !game.legal_bids.includes(option))
      : undefined;
  return (
    <motion.div
      className="bid-panel"
      initial={{ opacity: 0, scale: 0.92 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.92 }}
    >
      <h2>Your bid</h2>
      <div className="bid-options">
        {options.map((option) => (
          <button
            key={option}
            type="button"
            className="bid-option"
            disabled={option === blocked}
            onClick={() => onBid(option)}
          >
            {option}
          </button>
        ))}
      </div>
      {blocked !== undefined && (
        <p className="bid-note">As dealer you can't bid {blocked}: bids can't add up to the hand size.</p>
      )}
    </motion.div>
  );
}

export function Table({ game, onBid }: { game: GameState; onBid: (bid: number) => void }) {
  const [ref, { width, height }] = useElementSize();

  return (
    <div className="table-wrap">
      <div className="felt" ref={ref}>
        {game.players.map((player, seat) => (
          <SeatPlate key={player.name} player={player} seat={seat} game={game} />
        ))}
        <TrickCards game={game} width={width} height={height} />
        <AnimatePresence>
          {game.awaiting === "user_bid" && <BidPanel key="bid" game={game} onBid={onBid} />}
        </AnimatePresence>
      </div>
    </div>
  );
}
