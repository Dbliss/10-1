"""The interface every bot implements, and a random reference bot."""

import random

from ..engine import BIDDING


class Bot:
    """Receives only an ``Observation`` and returns a bid or a card."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, obs):
        return self.bid(obs) if obs.phase == BIDDING else self.play(obs)

    def bid(self, obs):
        raise NotImplementedError

    def play(self, obs):
        raise NotImplementedError


class RandomBot(Bot):
    """Picks uniformly among legal actions."""

    def bid(self, obs):
        return self.rng.choice(obs.legal_actions())

    def play(self, obs):
        return self.rng.choice(obs.legal_actions())
