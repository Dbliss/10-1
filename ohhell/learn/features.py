"""Turning what a seat knows into input vectors for the models.

Suits are reordered so the same situation looks the same whichever suit is
trump: trumps come first, then the side suits, longest in hand first.
"""

import numpy as np

from ..cards import ABOVE, NUM_RANKS, RANK_OF, SUIT_MASK, SUIT_OF
from ..engine import NO_BID, NO_SUIT

MAX_TRICKS = 10
_RANK_BITS = (1 << NUM_RANKS) - 1
_BITS = ((np.arange(1 << NUM_RANKS)[:, None] >> np.arange(NUM_RANKS)) & 1).astype(np.float32)

BID_INPUTS = 52 + 13 + 7 + 10 + 4 + 11 + 1
PLAY_INPUTS = 52 * 4 + 4 + 9 + 3 + 7 + 3 + 1 + 4 + 2


def _suit_order(trump, hand, played=0):
    def key(suit):
        return (-(hand & SUIT_MASK[suit]).bit_count(),
                (played & SUIT_MASK[suit]).bit_count(), suit)
    return [trump] + sorted((s for s in range(4) if s != trump), key=key)


def _encode(mask, order):
    return _BITS[[(mask >> (NUM_RANKS * s)) & _RANK_BITS for s in order]].ravel()


def bid_matrix(hand, trump_card, n, hand_size, seat, dealer, bids, options):
    """One row per candidate bid."""
    order = _suit_order(SUIT_OF[trump_card], hand)
    base = np.zeros(BID_INPUTS, dtype=np.float32)
    base[:52] = _encode(hand, order)
    base[52 + RANK_OF[trump_card]] = 1.0
    base[65 + n - 2] = 1.0
    base[72 + hand_size - 1] = 1.0

    position = (seat - dealer - 1) % n
    earlier = [b for b in bids if b != NO_BID]
    total = sum(earlier)
    base[82] = position / (n - 1)
    base[83] = float(position == 0)
    base[84] = total / hand_size
    base[85] = (total - hand_size * len(earlier) / n) / hand_size

    rows = np.tile(base, (len(options), 1))
    for i, bid in enumerate(options):
        rows[i, 86 + bid] = 1.0
        rows[i, 97] = bid / hand_size
    return rows


def play_matrix(view, seat, hand, cards):
    """One row per candidate card. ``view`` is a RoundState or an Observation."""
    n = view.n
    trump = view.trump_suit
    played = view.played | (1 << view.trump_card)
    order = _suit_order(trump, hand, played)
    place = {suit: i for i, suit in enumerate(order)}
    seen = played | hand
    bids = view.bids
    won = view.tricks_won
    left = max(1, hand.bit_count())

    base = np.zeros(PLAY_INPUTS, dtype=np.float32)
    base[:52] = _encode(hand, order)
    base[52:104] = _encode(played, order)
    best = view.best_card
    if best >= 0:
        base[104 + 13 * place[SUIT_OF[best]] + RANK_OF[best]] = 1.0

    at = 208
    if view.lead_suit != NO_SUIT:
        base[at + place[view.lead_suit]] = 1.0
    at += 4
    need = bids[seat] - won[seat]
    base[at + max(-2, min(6, need)) + 2] = 1.0
    at += 9
    base[at] = need / left
    base[at + 1] = left / MAX_TRICKS
    base[at + 2] = view.trick_size / (n - 1)
    at += 3
    base[at + n - 2] = 1.0
    at += 7

    wanting = behind_wanting = 0
    needed = 0
    for offset in range(1, n):
        other = (seat + offset) % n
        short = bids[other] - won[other]
        if short > 0:
            wanting += 1
            needed += short
            if offset < n - view.trick_size:
                behind_wanting += 1
    base[at] = wanting / (n - 1)
    base[at + 1] = behind_wanting / (n - 1)
    base[at + 2] = needed / left
    at += 3
    if view.best_seat >= 0:
        base[at] = float(bids[view.best_seat] > won[view.best_seat])
    at += 1
    for i, suit in enumerate(order):
        base[at + i] = sum(
            view.voids[s] >> suit & 1 for s in range(n) if s != seat) / (n - 1)
    at += 4

    rows = np.tile(base, (len(cards), 1))
    for i, card in enumerate(cards):
        suit = SUIT_OF[card]
        rows[i, 156 + 13 * place[suit] + RANK_OF[card]] = 1.0
        if best < 0:
            wins = True
        elif suit == SUIT_OF[best]:
            wins = card > best
        else:
            wins = suit == trump
        rows[i, at] = float(wins)
        rows[i, at + 1] = float(not ABOVE[card] & ~seen)
    return rows
