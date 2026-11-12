"""Pure rules for one round of Oh Hell.

``RoundState`` is the full truth, including every hand. Bots never see it: they
get an ``Observation``, which holds only what that seat is allowed to know.
Search code can turn an observation plus a guess at the hidden hands back into
a ``RoundState`` with ``Observation.determinize``.
"""

from dataclasses import dataclass

from .cards import EFFECTIVE_SUIT, POWER, SUIT_CARDS, SUIT_OF, cards_in
from .rules import Rules

BIDDING, PLAYING, DONE = 0, 1, 2
NO_BID = -1
NO_SUIT = -1
NO_CARD = -1
BLIND_CARD = -2  # Play the sole hidden card without revealing its identity.


def legal_bids(rules, hand_size, bids):
    """List the bids open to the next bidder, given the bids made so far."""
    options = list(range(hand_size + 1))
    made = [b for b in bids if b != NO_BID]
    if rules.dealer_restriction and len(made) == rules.num_players - 1:
        forbidden = hand_size - sum(made)
        if forbidden in options:
            options.remove(forbidden)
    return options


def legal_card_mask(hand, lead_suit, trump):
    """Cards a hand may play; the left bower follows trump, not its own suit."""
    if lead_suit == NO_SUIT:
        return hand
    return (hand & SUIT_CARDS[trump][lead_suit]) or hand


class RoundState:
    __slots__ = (
        'rules', 'n', 'hand_size', 'trump_card', 'trump_suit', 'dealer',
        'hands', 'bids', 'bids_made', 'to_act', 'phase', 'leader', 'lead_suit',
        'best_card', 'best_seat', 'trick_size', 'table', 'tricks_won',
        'played', 'voids', 'history', 'blind',
    )

    def __init__(self, rules, hands, trump_card, dealer, blind=False):
        self.rules = rules
        self.n = rules.num_players
        self.hand_size = hands[0].bit_count()
        if blind and any(hand.bit_count() != 1 for hand in hands):
            raise ValueError('A blind round requires one card per player.')
        self.blind = blind
        self.trump_card = trump_card
        self.trump_suit = SUIT_OF[trump_card]
        self.dealer = dealer
        self.hands = list(hands)
        self.bids = [NO_BID] * self.n
        self.bids_made = 0
        self.to_act = (dealer + 1) % self.n
        self.phase = BIDDING
        self.leader = self.to_act
        self.lead_suit = NO_SUIT
        self.best_card = NO_CARD
        self.best_seat = -1
        self.trick_size = 0
        self.table = 0  # cards in the unfinished trick
        self.tricks_won = [0] * self.n
        self.played = 0  # every card played so far, including the table
        self.voids = [0] * self.n  # per seat, a 4-bit set of suits shown void
        self.history = []  # (seat, card) in play order; None inside search

    def clone(self):
        """Copy for search. The copy drops the play history."""
        copy = RoundState.__new__(RoundState)
        copy.rules = self.rules
        copy.n = self.n
        copy.hand_size = self.hand_size
        copy.blind = self.blind
        copy.trump_card = self.trump_card
        copy.trump_suit = self.trump_suit
        copy.dealer = self.dealer
        copy.hands = self.hands[:]
        copy.bids = self.bids[:]
        copy.bids_made = self.bids_made
        copy.to_act = self.to_act
        copy.phase = self.phase
        copy.leader = self.leader
        copy.lead_suit = self.lead_suit
        copy.best_card = self.best_card
        copy.best_seat = self.best_seat
        copy.trick_size = self.trick_size
        copy.table = self.table
        copy.tricks_won = self.tricks_won[:]
        copy.played = self.played
        copy.voids = self.voids[:]
        copy.history = None
        return copy

    def legal_actions(self):
        if self.phase == BIDDING:
            return legal_bids(self.rules, self.hand_size, self.bids)
        if self.phase == PLAYING:
            if self.blind:
                return [BLIND_CARD]
            return cards_in(legal_card_mask(self.hands[self.to_act], self.lead_suit, self.trump_suit))
        return []

    def apply(self, action):
        """Validate and perform a bid or a card for the seat to act."""
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action {action} in phase {self.phase}.")
        if self.phase == BIDDING:
            self._place_bid(action)
        else:
            if self.blind:
                action = cards_in(self.hands[self.to_act])[0]
            self.play_unchecked(action)

    def _place_bid(self, bid):
        self.bids[self.to_act] = bid
        self.bids_made += 1
        if self.bids_made == self.n:
            self.phase = PLAYING
            self.leader = self.to_act = (self.dealer + 1) % self.n
        else:
            self.to_act = (self.to_act + 1) % self.n

    def play_unchecked(self, card):
        """Play a card known to be legal. This is the hot path for search."""
        seat = self.to_act
        bit = 1 << card
        suit = EFFECTIVE_SUIT[self.trump_suit][card]
        self.hands[seat] ^= bit
        self.played |= bit
        self.table |= bit

        if self.trick_size == 0:
            self.lead_suit = suit
            self.best_card = card
            self.best_seat = seat
        else:
            if suit != self.lead_suit:
                self.voids[seat] |= 1 << self.lead_suit
            trump = self.trump_suit
            best = self.best_card
            best_suit = EFFECTIVE_SUIT[trump][best]
            if ((POWER[trump][card] > POWER[trump][best]) if suit == best_suit
                    else (suit == trump)):
                self.best_card = card
                self.best_seat = seat

        if self.history is not None:
            self.history.append((seat, card))

        self.trick_size += 1
        if self.trick_size < self.n:
            self.to_act = (seat + 1) % self.n
            return

        winner = self.best_seat
        self.tricks_won[winner] += 1
        self.leader = self.to_act = winner
        self.trick_size = 0
        self.lead_suit = NO_SUIT
        self.best_card = NO_CARD
        self.best_seat = -1
        self.table = 0
        if not self.hands[winner]:
            self.phase = DONE

    def points(self):
        score = self.rules.scoring.points
        return [score(b, t) for b, t in zip(self.bids, self.tricks_won)]

    def observation(self, seat):
        return Observation(
            rules=self.rules, n=self.n, hand_size=self.hand_size,
            trump_card=self.trump_card, trump_suit=self.trump_suit,
            dealer=self.dealer, seat=seat, hand=0 if self.blind else self.hands[seat],
            bids=tuple(self.bids), to_act=self.to_act, phase=self.phase,
            leader=self.leader, lead_suit=self.lead_suit,
            best_card=self.best_card, best_seat=self.best_seat,
            trick_size=self.trick_size, table=self.table,
            tricks_won=tuple(self.tricks_won), played=self.played,
            voids=tuple(self.voids), history=tuple(self.history),
            blind=self.blind,
        )


@dataclass(frozen=True, slots=True)
class Observation:
    """Everything one seat knows: its own hand and the public game record."""

    rules: Rules
    n: int
    hand_size: int
    trump_card: int
    trump_suit: int
    dealer: int
    seat: int
    hand: int
    bids: tuple
    to_act: int
    phase: int
    leader: int
    lead_suit: int
    best_card: int
    best_seat: int
    trick_size: int
    table: int
    tricks_won: tuple
    played: int
    voids: tuple
    history: tuple
    blind: bool = False

    def legal_actions(self):
        if self.phase == BIDDING:
            return legal_bids(self.rules, self.hand_size, self.bids)
        if self.phase == PLAYING:
            if self.blind:
                return [BLIND_CARD]
            return cards_in(legal_card_mask(self.hand, self.lead_suit, self.trump_suit))
        return []

    def cards_held(self, seat):
        """How many cards a seat still holds."""
        return self.hand_size - sum(1 for s, _ in self.history if s == seat)

    def determinize(self, hands):
        """Build a searchable state from a guess at every hand."""
        state = RoundState.__new__(RoundState)
        state.rules = self.rules
        state.n = self.n
        state.hand_size = self.hand_size
        state.blind = self.blind
        state.trump_card = self.trump_card
        state.trump_suit = self.trump_suit
        state.dealer = self.dealer
        state.hands = list(hands)
        state.bids = list(self.bids)
        state.bids_made = sum(b != NO_BID for b in self.bids)
        state.to_act = self.to_act
        state.phase = self.phase
        state.leader = self.leader
        state.lead_suit = self.lead_suit
        state.best_card = self.best_card
        state.best_seat = self.best_seat
        state.trick_size = self.trick_size
        state.table = self.table
        state.tricks_won = list(self.tricks_won)
        state.played = self.played
        state.voids = list(self.voids)
        state.history = None
        return state
