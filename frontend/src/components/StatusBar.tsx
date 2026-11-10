import { cardImage, cardName, isRed, SUIT_GLYPH } from "../cards";
import type { GameState } from "../types";

function statusMessage(game: GameState): string {
  const { awaiting, phase, players, current_player, trick_winner, lead_suit, hand } = game;

  switch (awaiting) {
    case "user_bid":
      return "Your bid. How many tricks will you take?";
    case "user_play":
      if (!lead_suit) return "Your lead. Play any card.";
      return hand.some((card) => card.suit === lead_suit)
        ? `Your turn. Follow ${lead_suit.toLowerCase()}.`
        : `No ${lead_suit.toLowerCase()} left. Play any card.`;
    case "bot":
      return `${players[current_player!].name} is ${phase === "bidding" ? "bidding" : "playing"}…`;
    case "collect_trick":
      return players[trick_winner!].is_human ? "You take the trick." : `${players[trick_winner!].name} takes the trick.`;
    case "next_round":
      return "Round complete.";
    case "nothing":
      return "Game over.";
  }
}

function bidSummary(game: GameState): string {
  const total = game.players.reduce((sum, player) => sum + (player.bid ?? 0), 0);
  if (game.phase === "bidding") return `${total} bid so far`;
  if (total === game.hand_size) return `${total} bid, exactly enough`;
  return total > game.hand_size
    ? `${total} bid, ${total - game.hand_size} over`
    : `${total} bid, ${game.hand_size - total} under`;
}

export function StatusBar({ game }: { game: GameState }) {
  const { trump } = game;
  const yourTurn = game.awaiting === "user_bid" || game.awaiting === "user_play";

  return (
    <div className="status">
      {trump && (
        <div className="trump">
          <img className="trump-card" src={cardImage(trump)} alt={`Trump card: ${cardName(trump)}`} />
          <div>
            <span className="label">Trump</span>
            <strong className={isRed(trump.suit) ? "suit-red" : "suit-light"}>
              {SUIT_GLYPH[trump.suit]} {trump.suit}
            </strong>
          </div>
        </div>
      )}

      <p className={`status-message ${yourTurn ? "status-yours" : ""}`} aria-live="polite">
        {statusMessage(game)}
      </p>

      <div className="tally">
        <span className="label">
          {game.hand_size} {game.hand_size === 1 ? "trick" : "tricks"}
        </span>
        <strong>{bidSummary(game)}</strong>
      </div>
    </div>
  );
}
