"""The interface every bot implements, and a random reference bot."""

import random

from ..engine import BIDDING


class Bot:
    """Receives only an ``Observation`` and returns a bid or a card."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, obs):
        if obs.blind:
            if obs.phase != BIDDING:
                return obs.legal_actions()[0]
            return self.blind_bid(obs)
        return self.bid(obs) if obs.phase == BIDDING else self.play(obs)

    def blind_bid(self, obs):
        from ..blind import bid_values
        values, _ = bid_values(obs)
        return max(values, key=values.get)

    def bid(self, obs):
        raise NotImplementedError

    def play(self, obs):
        raise NotImplementedError


class RandomBot(Bot):
    """Picks uniformly among legal actions."""

    def bid(self, obs):
        return self.rng.choice(obs.legal_actions())

    blind_bid = bid

    def play(self, obs):
        return self.rng.choice(obs.legal_actions())
