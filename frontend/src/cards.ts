import type { Card, Suit } from "./types";

export const SUIT_GLYPH: Record<Suit, string> = {
  Spades: "♠",
  Hearts: "♥",
  Clubs: "♣",
  Diamonds: "♦",
};

export const isRed = (suit: Suit) => suit === "Hearts" || suit === "Diamonds";

export const cardImage = (card: Pick<Card, "rank" | "suit">) =>
  `/cards/${card.rank.toLowerCase()}_of_${card.suit.toLowerCase()}.png`;

export const cardName = (card: Card) => `${card.rank} of ${card.suit}`;

const SUITS: Suit[] = ["Spades", "Hearts", "Clubs", "Diamonds"];
const RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"];

/**
 * Group a hand by the suit each card plays as, trump first, strongest on the
 * left of each suit. The two bowers therefore lead the trump group.
 */
export function sortHand(hand: Card[], trump: Suit | undefined): Card[] {
  const suitOrder = (suit: Suit) => (suit === trump ? -1 : SUITS.indexOf(suit));
  return [...hand].sort(
    (a, b) => suitOrder(a.plays_as) - suitOrder(b.plays_as) || b.power - a.power,
  );
}

export function preloadCardImages() {
  for (const suit of SUITS) {
    for (const rank of RANKS) {
      new Image().src = cardImage({ rank, suit });
    }
  }
}
