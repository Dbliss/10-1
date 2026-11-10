"""The kinds of opponent the bots plan for, from beginners to strong players."""

from dataclasses import dataclass

from .bots.heuristic import state_card
from .cards import cards_in
from .engine import legal_card_mask


@dataclass(frozen=True)
class Kind:
    name: str
    card_noise: float  # chance of playing a random legal card
    bid_one: float  # chance a bid is off by one
    bid_two: float  # chance a bid is off by two
    bid_sigma: float  # how far a bid may stray from a hand's estimate
    play_doubt: float  # chance a play is unlike the rule-based bot's


SKILLED = Kind('skilled', 0.0, 0.15, 0.0, 1.2, 0.15)
CASUAL = Kind('casual', 0.35, 0.4, 0.1, 1.7, 0.35)
NOVICE = Kind('novice', 1.0, 0.4, 0.3, 2.5, 1.0)

# What the previous default bot assumed: everyone plays by the book, with some
# tolerance for plays that don't fit.
PREVIOUS = Kind('previous', 0.0, 0.0, 0.0, 1.4, 0.75)

KINDS = (SKILLED, CASUAL, NOVICE, PREVIOUS)
FIELDS = {
    'mix': (0.4, 0.3, 0.3, 0.0),
    'skilled': (1.0, 0.0, 0.0, 0.0),
    'casual': (0.0, 1.0, 0.0, 0.0),
    'novice': (0.0, 0.0, 1.0, 0.0),
    'previous': (0.0, 0.0, 0.0, 1.0),
}


def noisy_bid(bid, kind, rng, options):
    """Move a bid by the kind's usual error, if the result is legal."""
    roll = rng.random()
    if roll < kind.bid_two:
        size = 2
    elif roll < kind.bid_two + kind.bid_one:
        size = 1
    else:
        return bid
    moved = bid + rng.choice((-size, size))
    return moved if moved in options else bid


class PopulationPolicy:
    """Finishes a deal with each seat playing according to its kind."""

    def __init__(self, kinds, rng, react=False):
        self.noise = [kind.card_noise for kind in kinds]
        self.rng = rng
        self.react = react

    def __call__(self, state):
        noise = self.noise[state.to_act]
        if noise and self.rng.random() < noise:
            hand = state.hands[state.to_act]
            return self.rng.choice(cards_in(legal_card_mask(hand, state.lead_suit)))
        return state_card(state, self.react)
