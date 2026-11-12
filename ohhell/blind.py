"""Blind-round bidding from public information, without inspecting any hand."""

from functools import lru_cache
from math import comb

from .cards import SUIT_OF, beats


@lru_cache(maxsize=1024)
def win_probability(players, trump_card, leads):
    """Exact chance of winning the single trick before any cards are shown.

    A leader can win with a non-trump when everyone follows another suit;
    other seats must beat the lead card as well as all remaining opponents.
    Cards are drawn without replacement, excluding the visible trump card.
    """
    deck = [card for card in range(52) if card != trump_card]
    trump = SUIT_OF[trump_card]
    if leads:
        return sum(
            comb(sum(not beats(other, card, trump) for other in deck if other != card),
                 players - 1)
            for card in deck
        ) / (51 * comb(50, players - 1))
    total = 0
    for lead in deck:
        for card in deck:
            if card == lead or not beats(card, lead, trump):
                continue
            safe = sum(not beats(other, card, trump)
                       for other in deck if other != lead and other != card)
            total += comb(safe, players - 2)
    return total / (51 * 50 * comb(49, players - 2))


def bid_values(obs):
    probability = win_probability(obs.n, obs.trump_card, obs.seat == obs.leader)
    score = obs.rules.scoring.points
    return {
        bid: ((1 - probability) * score(bid, 0) + probability * score(bid, 1))
        for bid in obs.legal_actions()
    }, probability
