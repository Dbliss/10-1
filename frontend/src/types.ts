export type Suit = "Hearts" | "Diamonds" | "Clubs" | "Spades";

export interface Card {
  id: string;
  suit: Suit;
  rank: string;
}

export interface Player {
  name: string;
  is_human: boolean;
  num_cards: number;
  bid: number | null;
  tricks_won: number;
  score: number;
}

export interface RoundResult {
  bid: number;
  tricks: number;
  points: number;
  score: number;
}

export interface RoundHistory {
  round: number;
  hand_size: number;
  trump: Card;
  results: RoundResult[];
}

export type Advice =
  | {
      kind: "bid";
      recommended: number;
      options: { bid: number; expected_points: number; make_chance: number }[];
    }
  | {
      kind: "card";
      recommended: string;
      options: { card: Card; expected_points: number }[];
    };

export type Phase ="bidding" | "playing" | "trick_complete" | "round_over" | "game_over";

export type Awaiting =
  | "user_bid"
  | "user_play"
  | "bot"
  | "collect_trick"
  | "next_round"
  | "nothing";

export interface GameState {
  phase: Phase;
  awaiting: Awaiting;
  round: number;
  total_rounds: number;
  hand_size: number;
  trump: Card | null;
  dealer: number;
  current_player: number | null;
  players: Player[];
  hand: Card[];
  legal_bids: number[];
  playable: string[];
  trick: { seat: number; card: Card }[];
  lead_suit: Suit | null;
  trick_winner: number | null;
  history: RoundHistory[];
}
