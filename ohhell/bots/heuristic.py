"""Hand-written bidding and play.

This is the measured baseline, and ``choose_card`` is also the policy that
simulations use to finish a hand, so it has to be fast.
"""

import math

from ..cards import ABOVE, RANK_OF, SUIT_MASK, SUIT_OF, cards_in, highest, lowest
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
    seen = hand | (1 << trump_card)
    unseen = 51 - hand_size
    miss = max(0.0, 1.0 - (n - 1) * hand_size / unseen)
    trump_held = (hand & SUIT_MASK[trump]).bit_count()
    short_sides = sum(
        (hand & SUIT_MASK[s]).bit_count() <= 1 for s in range(4) if s != trump)
    other_trumps = 13 - trump_held - 1
    ruff_rate = hand_size * other_trumps / unseen
    opponent_has_trump = 1.0 - math.exp(-ruff_rate)

    probs = []
    for suit in range(4):
        cards = cards_in(hand & SUIT_MASK[suit])
        cards.reverse()  # highest first
        length = len(cards)
        for position, card in enumerate(cards):
            higher = (ABOVE[card] & ~seen).bit_count()
            clear = miss ** higher
            if suit == trump:
                ruff = 0.0
                if hand_size >= 3:
                    ruff = min(0.55, 0.10 * (trump_held - 1) + 0.12 * short_sides)
                probs.append(clear + (1.0 - clear) * ruff)
                continue
            outstanding = 13 - length - (SUIT_OF[trump_card] == suit)
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


def _is_boss(card, seen):
    return not (ABOVE[card] & ~seen)


def _lead(hand, trump, seen, want):
    trumps = hand & SUIT_MASK[trump]
    if want:
        if trumps:
            top = highest(trumps)
            if _is_boss(top, seen) and trumps.bit_count() >= 2:
                return top
        best = None
        for suit in range(4):
            cards = hand & SUIT_MASK[suit]
            if cards and suit != trump:
                top = highest(cards)
                if _is_boss(top, seen) and (best is None or RANK_OF[top] > RANK_OF[best]):
                    best = top
        if best is not None:
            return best
        if trumps and _is_boss(highest(trumps), seen):
            return highest(trumps)
        # No sure winner. Lead low from the shortest side suit to set up a ruff.
        shortest = None
        for suit in range(4):
            cards = hand & SUIT_MASK[suit]
            if cards and suit != trump:
                if shortest is None or cards.bit_count() < shortest.bit_count():
                    shortest = cards
        return lowest(shortest if shortest else trumps)

    # Trying to lose: lead whatever has the most unseen cards above it.
    best = None
    best_score = -99
    for suit in range(4):
        cards = hand & SUIT_MASK[suit]
        if cards:
            card = lowest(cards)
            score = (ABOVE[card] & ~seen).bit_count() - (3 if suit == trump else 0)
            if score > best_score:
                best, best_score = card, score
    return best


def _highest_rank(mask):
    """The card of highest rank across suits, preferring the shorter suit."""
    best = None
    best_key = None
    for suit in range(4):
        cards = mask & SUIT_MASK[suit]
        if cards:
            top = highest(cards)
            key = (RANK_OF[top], -cards.bit_count())
            if best_key is None or key > best_key:
                best, best_key = top, key
    return best


def _lowest_rank(mask):
    best = None
    for suit in range(4):
        cards = mask & SUIT_MASK[suit]
        if cards:
            low = lowest(cards)
            if best is None or RANK_OF[low] < RANK_OF[best]:
                best = low
    return best


def choose_card(hand, trump, lead_suit, best_card, trick_size, n, seen, need):
    """Pick a card from one player's point of view.

    ``seen`` is every card that player knows the location of, and ``need`` is
    their bid minus the tricks they have taken.
    """
    want = need > 0
    if lead_suit == NO_SUIT:
        return _lead(hand, trump, seen, want)

    last = trick_size == n - 1
    following = hand & SUIT_MASK[lead_suit]
    best_suit = SUIT_OF[best_card]

    if following:
        winners = following & ABOVE[best_card] if best_suit == lead_suit else 0
        losers = following ^ winners
        if want:
            if not winners:
                return lowest(losers)
            if last:
                return lowest(winners)
            top = highest(winners)
            if _is_boss(top, seen) or trick_size * 2 >= n or not losers:
                return top
            return lowest(losers)
        if losers:
            return highest(losers)
        return highest(winners) if last else lowest(winners)

    trumps = hand & SUIT_MASK[trump]
    sides = hand ^ trumps
    trumped = best_suit == trump

    if want:
        if trumps:
            if not trumped:
                return lowest(trumps)
            over = trumps & ABOVE[best_card]
            if over:
                return lowest(over)
        return _lowest_rank(sides) if sides else lowest(trumps)

    if trumped:
        under = trumps & ~ABOVE[best_card]
        if under:
            return highest(under)
    if sides:
        return _highest_rank(sides)
    return highest(trumps) if last else lowest(trumps)


def state_card(state):
    """Run ``choose_card`` for whoever is to act in a full ``RoundState``."""
    seat = state.to_act
    hand = state.hands[seat]
    return choose_card(
        hand, state.trump_suit, state.lead_suit, state.best_card,
        state.trick_size, state.n,
        state.played | hand | (1 << state.trump_card),
        state.bids[seat] - state.tricks_won[seat])


class HeuristicBot(Bot):
    """Bids from per-card odds and plays to reach or avoid its target."""

    def bid(self, obs):
        leads = obs.seat == (obs.dealer + 1) % obs.n
        probs = card_win_probs(obs.hand, obs.trump_card, obs.n, obs.hand_size, leads)
        return best_bid_for(trick_distribution(probs), obs.legal_actions(),
                            obs.rules.scoring)

    def play(self, obs):
        return choose_card(
            obs.hand, obs.trump_suit, obs.lead_suit, obs.best_card,
            obs.trick_size, obs.n,
            obs.played | obs.hand | (1 << obs.trump_card),
            obs.bids[obs.seat] - obs.tricks_won[obs.seat])
