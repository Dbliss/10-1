"""Turning what a seat knows into input vectors for the models.

Suits are reordered so the same situation looks the same whichever suit is
trump: trumps come first, then the side suits, longest in hand first.
"""

import numpy as np

from ..cards import LEFT_BOWER, NUM_RANKS, OUTRANKS, RANK_OF, SUIT_CARDS, SUIT_OF, beats, partner_suit
from ..engine import NO_BID, NO_SUIT

MAX_TRICKS = 10
_RANK_BITS = (1 << NUM_RANKS) - 1
_BITS = ((np.arange(1 << NUM_RANKS)[:, None] >> np.arange(NUM_RANKS)) & 1).astype(np.float32)

BID_INPUTS = 52 + 13 + 7 + 10 + 4 + 11 + 1
# Seat-bid layout: the 98 inputs above, then for each bidding position (0 is
# the first bidder after the dealer) whether it bid before us, its bid over
# the hand size and whether that bid was zero, then how many seats still bid
# after us as a one-hot. Old 98-input networks keep using the first layout.
SEAT_SLOTS = 7
BID_SEAT_INPUTS = BID_INPUTS + 3 * SEAT_SLOTS + SEAT_SLOTS + 1
# Bid bower layout: the 98 inputs, then a one-hot of the trump partner suit's
# slot (1-3 in the suit order) and whether the hand holds the left bower. The
# card bits keep the left bower as a Jack in that partner slot, so without
# these the network cannot tell it from a side-suit Jack.
BID_BOWER_INPUTS = BID_INPUTS + 4
PLAY_INPUTS = 52 * 4 + 4 + 9 + 3 + 7 + 3 + 1 + 4 + 2
# Bower layout: the card bits sit in printed suits, so the left bower shows as
# a Jack in one of the side-suit slots, and nothing said which slot is trump's
# colour partner. Four more inputs follow the 241 above: a one-hot of the
# partner suit's slot (1-3 in the suit order), and whether the candidate card
# is the left bower. Old 241-input networks keep using the first layout.
PLAY_BOWER_INPUTS = PLAY_INPUTS + 4


def _suit_order(trump, hand, played=0):
    suits = SUIT_CARDS[trump]

    def key(suit):
        return (-(hand & suits[suit]).bit_count(),
                (played & suits[suit]).bit_count(), suit)
    return [trump] + sorted((s for s in range(4) if s != trump), key=key)


def _encode(mask, order):
    return _BITS[[(mask >> (NUM_RANKS * s)) & _RANK_BITS for s in order]].ravel()


def bid_hand_part(hand, trump_card, n, hand_size):
    """The first 82 bid inputs: the hand, trump rank, players and hand size."""
    order = _suit_order(SUIT_OF[trump_card], hand)
    part = np.zeros(82, dtype=np.float32)
    part[:52] = _encode(hand, order)
    part[52 + RANK_OF[trump_card]] = 1.0
    part[65 + n - 2] = 1.0
    part[72 + hand_size - 1] = 1.0
    return part


def bid_bower_part(hand, trump_card):
    """The 4 bower inputs: the partner suit's slot one-hot, then the left bower held."""
    trump = SUIT_OF[trump_card]
    order = _suit_order(trump, hand)
    part = np.zeros(4, dtype=np.float32)
    part[order.index(partner_suit(trump)) - 1] = 1.0
    part[3] = float(hand >> LEFT_BOWER[trump] & 1)
    return part


def bid_context(bids, n, hand_size, seat, dealer, inputs=BID_INPUTS):
    """Inputs from 82 on that describe the earlier bids, for many tables at once.

    ``bids`` holds one row of every seat's bid per table, ``NO_BID`` for seats
    yet to bid; the earlier bidders are the ``position`` seats after the
    dealer. Columns for the candidate bid (86-97) are left at zero, and so
    are the bower layout's hand columns (98-101), which ``bid_bower_part`` fills.
    """
    bids = np.asarray(bids, dtype=np.float64).reshape(-1, n)
    rows = np.zeros((len(bids), inputs - 82))
    position = (seat - dealer - 1) % n
    seats = [(dealer + 1 + j) % n for j in range(position)]
    earlier = bids[:, seats]
    total = earlier.sum(axis=1)
    rows[:, 0] = position / (n - 1)
    rows[:, 1] = float(position == 0)
    rows[:, 2] = total / hand_size
    rows[:, 3] = (total - hand_size * position / n) / hand_size
    if inputs == BID_SEAT_INPUTS:
        at = BID_INPUTS - 82
        for j in range(position):
            rows[:, at + 3 * j] = 1.0
            rows[:, at + 3 * j + 1] = earlier[:, j] / hand_size
            rows[:, at + 3 * j + 2] = earlier[:, j] == 0
        rows[:, at + 3 * SEAT_SLOTS + n - 1 - position] = 1.0
    return rows.astype(np.float32)


def bid_matrix(hand, trump_card, n, hand_size, seat, dealer, bids, options,
               inputs=BID_INPUTS):
    """One row per candidate bid, with ``inputs`` columns (98, 102 or 127)."""
    base = np.zeros(inputs, dtype=np.float32)
    base[:82] = bid_hand_part(hand, trump_card, n, hand_size)
    base[82:] = bid_context(bids, n, hand_size, seat, dealer, inputs)[0]
    if inputs == BID_BOWER_INPUTS:
        base[BID_INPUTS:] = bid_bower_part(hand, trump_card)
    rows = np.tile(base, (len(options), 1))
    for i, bid in enumerate(options):
        rows[i, 86 + bid] = 1.0
        rows[i, 97] = bid / hand_size
    return rows


def play_matrix(view, seat, hand, cards, inputs=PLAY_INPUTS):
    """One row per candidate card, with ``inputs`` columns (241 or 245).

    ``view`` is a RoundState or an Observation.
    """
    n = view.n
    trump = view.trump_suit
    played = view.played | (1 << view.trump_card)
    order = _suit_order(trump, hand, played)
    place = {suit: i for i, suit in enumerate(order)}
    seen = played | hand
    bids = view.bids
    won = view.tricks_won
    left = max(1, hand.bit_count())

    base = np.zeros(inputs, dtype=np.float32)
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
        wins = best < 0 or beats(card, best, trump)
        rows[i, at] = float(wins)
        rows[i, at + 1] = float(not OUTRANKS[trump][card] & ~seen)
    if inputs > PLAY_INPUTS:
        rows[:, PLAY_INPUTS - 1 + place[partner_suit(trump)]] = 1.0
        for i, card in enumerate(cards):
            rows[i, PLAY_INPUTS + 3] = float(card == LEFT_BOWER[trump])
    return rows
