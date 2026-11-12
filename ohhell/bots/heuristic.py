"""Hand-written bidding and play.

This is the measured baseline, and ``choose_card`` is also the policy that
simulations use to finish a hand, so it has to be fast.
"""

import math

from ..cards import (
    EFFECTIVE_SUIT, OUTRANKS, POWER, RANK_OF, SUIT_CARDS, SUIT_OF, bottom, cards_in,
    highest, lowest, top,
)
from ..engine import NO_SUIT
from .base import Bot


def _poisson_at_most(k, mean):
    term = total = math.exp(-mean)
    for i in range(1, k + 1):
        term *= mean / i
        total += term
    return total


def card_win_probs(hand, trump_card, n, hand_size, leads_first=False):
    """Estimate, card by card, the chance each card in a hand takes a trick."""
    trump = SUIT_OF[trump_card]
    suits = SUIT_CARDS[trump]
    outranks = OUTRANKS[trump]
    power = POWER[trump]
    seen = hand | (1 << trump_card)
    unseen = 51 - hand_size
    miss = max(0.0, 1.0 - (n - 1) * hand_size / unseen)
    trump_held = (hand & suits[trump]).bit_count()
    short_sides = sum(
        (hand & suits[s]).bit_count() <= 1 for s in range(4) if s != trump)
    # The turned-up card is always a trump: the left bower is never turned.
    other_trumps = suits[trump].bit_count() - trump_held - 1
    ruff_rate = hand_size * other_trumps / unseen
    opponent_has_trump = 1.0 - math.exp(-ruff_rate)

    probs = []
    for suit in range(4):
        cards = sorted(cards_in(hand & suits[suit]), key=power.__getitem__, reverse=True)
        length = len(cards)
        for position, card in enumerate(cards):
            higher = (outranks[card] & ~seen).bit_count()
            clear = miss ** higher
            if suit == trump:
                ruff = 0.0
                if hand_size >= 3:
                    ruff = min(0.55, 0.10 * (trump_held - 1) + 0.12 * short_sides)
                probs.append(clear + (1.0 - clear) * ruff)
                continue
            outstanding = suits[suit].bit_count() - length
            opponent_length = hand_size * outstanding / unseen
            short = _poisson_at_most(position, opponent_length)
            safe = (1.0 - short * opponent_has_trump) ** (n - 1)
            if leads_first and position == 0:
                reached = 1.0
            else:
                reached = 1.0 - 0.75 ** hand_size
            probs.append(clear * safe * reached)
    return probs


def trick_distribution(probs):
    """Chance of taking exactly k tricks if each card wins independently."""
    dist = [1.0]
    for p in probs:
        grown = [0.0] * (len(dist) + 1)
        for k, weight in enumerate(dist):
            grown[k] += weight * (1.0 - p)
            grown[k + 1] += weight * p
        dist = grown
    return dist


def expected_tricks(hand, trump_card, n, hand_size, leads_first=False):
    return sum(card_win_probs(hand, trump_card, n, hand_size, leads_first))


def best_bid_for(dist, options, scoring):
    """Pick the bid with the highest expected points under a trick distribution."""
    def value(bid):
        return sum(p * scoring.points(bid, k) for k, p in enumerate(dist))
    return max(options, key=value)


def _is_boss(card, seen, trump):
    return not (OUTRANKS[trump][card] & ~seen)


def _lead(hand, trump, seen, want):
    suits = SUIT_CARDS[trump]
    trumps = hand & suits[trump]
    if want:
        if trumps:
            best_trump = top(trumps, trump)
            if _is_boss(best_trump, seen, trump) and trumps.bit_count() >= 2:
                return best_trump
        best = None
        for suit in range(4):
            cards = hand & suits[suit]
            if cards and suit != trump:
                high = highest(cards)
                if (_is_boss(high, seen, trump)
                        and (best is None or RANK_OF[high] > RANK_OF[best])):
                    best = high
        if best is not None:
            return best
        if trumps and _is_boss(top(trumps, trump), seen, trump):
            return top(trumps, trump)
        # No sure winner. Lead low from the shortest side suit to set up a ruff.
        shortest = None
        for suit in range(4):
            cards = hand & suits[suit]
            if cards and suit != trump:
                if shortest is None or cards.bit_count() < shortest.bit_count():
                    shortest = cards
        return lowest(shortest) if shortest else bottom(trumps, trump)

    # Trying to lose: lead whatever has the most unseen cards above it.
    best = None
    best_score = -99
    for suit in range(4):
        cards = hand & suits[suit]
        if cards:
            card = bottom(cards, trump)
            score = (OUTRANKS[trump][card] & ~seen).bit_count() - (3 if suit == trump else 0)
            if score > best_score:
                best, best_score = card, score
    return best


def _highest_rank(mask, trump):
    """The card of highest rank across side suits, preferring the shorter suit."""
    best = None
    best_key = None
    for suit in range(4):
        cards = mask & SUIT_CARDS[trump][suit]
        if cards:
            high = highest(cards)
            key = (RANK_OF[high], -cards.bit_count())
            if best_key is None or key > best_key:
                best, best_key = high, key
    return best


def _lowest_rank(mask, trump):
    """The card of lowest rank across side suits."""
    best = None
    for suit in range(4):
        cards = mask & SUIT_CARDS[trump][suit]
        if cards:
            low = lowest(cards)
            if best is None or RANK_OF[low] < RANK_OF[best]:
                best = low
    return best


def table_read(seat, n, bids, won, trick_size, left):
    """What the other bids say about this trick, for ``choose_card``.

    Returns whether a seat still to play wants tricks, and whether the table
    needs more tricks in total than the ``left`` still to be played.
    """
    behind_wants = False
    total = 0
    for offset in range(n):
        other = (seat + offset) % n
        short = bids[other] - won[other]
        if short > 0:
            total += short
            if 0 < offset < n - trick_size:
                behind_wants = True
    return behind_wants, total > left


def choose_card(hand, trump, lead_suit, best_card, trick_size, n, seen, need,
                behind_wants=True, scarce=False):
    """Pick a card from one player's point of view.

    ``seen`` is every card that player knows the location of, and ``need`` is
    their bid minus the tricks they have taken. ``behind_wants`` and ``scarce``
    come from ``table_read``; the defaults ignore the other players.
    """
    want = need > 0
    if lead_suit == NO_SUIT:
        return _lead(hand, trump, seen, want)

    last = trick_size == n - 1
    suits = SUIT_CARDS[trump]
    outranks = OUTRANKS[trump]
    following = hand & suits[lead_suit]
    best_suit = EFFECTIVE_SUIT[trump][best_card]

    if following:
        winners = following & outranks[best_card] if best_suit == lead_suit else 0
        losers = following ^ winners
        if want:
            if not winners:
                return bottom(losers, trump)
            if last or not behind_wants:  # nobody left will fight for this trick
                return bottom(winners, trump)
            high = top(winners, trump)
            if _is_boss(high, seen, trump) or trick_size * 2 >= n or not losers or scarce:
                return high
            return bottom(losers, trump)
        if losers:
            return top(losers, trump)
        return (top(winners, trump) if last or not behind_wants
                else bottom(winners, trump))

    trumps = hand & suits[trump]
    sides = hand ^ trumps
    trumped = best_suit == trump

    if want:
        if trumps:
            if not trumped:
                return bottom(trumps, trump)
            over = trumps & outranks[best_card]
            if over:
                return bottom(over, trump)
        return _lowest_rank(sides, trump) if sides else bottom(trumps, trump)

    if trumped:
        under = trumps & ~outranks[best_card]
        if under:
            return top(under, trump)
    if sides:
        return _highest_rank(sides, trump)
    return top(trumps, trump) if last else bottom(trumps, trump)


def state_card(state, react=False):
    """Run ``choose_card`` for whoever is to act in a full ``RoundState``."""
    seat = state.to_act
    hand = state.hands[seat]
    read = (table_read(seat, state.n, state.bids, state.tricks_won, state.trick_size,
                       hand.bit_count()) if react else ())
    return choose_card(
        hand, state.trump_suit, state.lead_suit, state.best_card,
        state.trick_size, state.n,
        state.played | hand | (1 << state.trump_card),
        state.bids[seat] - state.tricks_won[seat], *read)


class HeuristicBot(Bot):
    """Bids from per-card odds and plays to reach or avoid its target."""

    def __init__(self, seed=0, react=False):
        super().__init__(seed)
        self.react = react

    def bid(self, obs):
        leads = obs.seat == (obs.dealer + 1) % obs.n
        probs = card_win_probs(obs.hand, obs.trump_card, obs.n, obs.hand_size, leads)
        return best_bid_for(trick_distribution(probs), obs.legal_actions(),
                            obs.rules.scoring)

    def play(self, obs):
        read = (table_read(obs.seat, obs.n, obs.bids, obs.tricks_won, obs.trick_size,
                           obs.hand.bit_count()) if self.react else ())
        return choose_card(
            obs.hand, obs.trump_suit, obs.lead_suit, obs.best_card,
            obs.trick_size, obs.n,
            obs.played | obs.hand | (1 << obs.trump_card),
            obs.bids[obs.seat] - obs.tricks_won[obs.seat], *read)
