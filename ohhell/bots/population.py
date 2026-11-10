"""Bots that imitate each kind of opponent, for testing against."""

import random

from ..opponents import FIELDS, KINDS, noisy_bid
from .heuristic import HeuristicBot


class KindBot(HeuristicBot):
    """The rule-based bot with one kind of opponent's mistakes added."""

    def __init__(self, seed=0, kind='novice'):
        super().__init__(seed)
        self.kind = next(k for k in KINDS if k.name == kind)

    def bid(self, obs):
        return noisy_bid(super().bid(obs), self.kind, self.rng, obs.legal_actions())

    def play(self, obs):
        if self.kind.card_noise and self.rng.random() < self.kind.card_noise:
            return self.rng.choice(obs.legal_actions())
        return super().play(obs)


def mixed_bot(seed):
    """One opponent drawn from the planning mix, chosen by its seed."""
    kind = random.Random(seed).choices(KINDS, FIELDS['mix'])[0]
    return KindBot(seed=seed, kind=kind.name)
