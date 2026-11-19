import type { Card, Suit } from "../types";

/** A seat at the table. Seats are listed in table order, the user first. */
export interface Seat {
  name: string;
  is_you: boolean;
  is_dealer: boolean;
  /** Position in the bidding: 0 for the seat left of the dealer. */
  order: number;
  bid: number | null;
}

export interface BidOption {
  bid: number;
  expected_points: number;
  make_chance: number;
  /** Chance of taking exactly k tricks after this bid, for k = 0 .. tricks. */
  tricks: number[];
}

/** One random bidding decision, with the analysis of every legal bid. */
export interface Situation {
  players: number;
  /** Cards dealt: 0 for the blind round. */
  hand_size: number;
  /** Tricks in the round: 1 for the blind round. */
  tricks: number;
  blind: boolean;
  /** How many seats bid before the user. */
  position: number;
  /** The seat that bids first. */
  first: number;
  is_dealer: boolean;
  trump: Card;
  trump_suit: Suit;
  hand: Card[];
  seats: Seat[];
  legal_bids: number[];
  forbidden_bid: number | null;
  best_bid: number;
  options: BidOption[];
}

export interface Player {
  name: string;
  color: string;
}

export interface Round {
  sit: Situation;
  /** Each player's bid, by player index; missing until they bid. */
  bids: number[];
}
