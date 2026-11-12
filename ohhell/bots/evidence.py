"""Bidding with sample-counted model priors."""

import math

from .learned import HybridBot, _expected
from .simulation import BidAnalysis, SimulationBot


class EvidenceBot(HybridBot):
    """Keep hybrid card play; let actual bid samples determine prior strength.

    The model contributes ``bid_prior_worlds`` virtual deals to each bid.
    The heuristic bid shortlist is unchanged. Blind bids still use the exact
    blind-round calculation inherited from Bot.
    """

    def __init__(self, seed=0, bid_prior_worlds=64, **kwargs):
        if not math.isfinite(bid_prior_worlds) or bid_prior_worlds < 0:
            raise ValueError('bid_prior_worlds must be finite and nonnegative.')
        if 'bid_blend' in kwargs:
            raise ValueError('Use bid_prior_worlds instead of bid_blend with evidence.')
        super().__init__(seed=seed, **kwargs)
        self.bid_prior_worlds = bid_prior_worlds
        self.last_bid_search = None

    def analyse_bids(self, obs):
        simulated = SimulationBot.analyse_bids(self, obs)
        options = list(simulated.distributions)
        predicted = self.models.bid_distributions(
            obs.hand, obs.trump_card, obs.n, obs.hand_size, obs.seat, obs.dealer,
            obs.bids, options)
        score = obs.rules.scoring
        prior = self.bid_prior_worlds
        blend = prior / (prior + simulated.samples)
        distributions = {
            bid: [(1 - blend) * s + blend * p
                  for s, p in zip(simulated.distributions[bid], predicted[bid])]
            for bid in options
        }
        self.last_bid_search = {
            'options': options, 'samples': simulated.samples, 'model_weight': blend,
        }
        return BidAnalysis(distributions, {
            bid: _expected(dist, bid, score) for bid, dist in distributions.items()
        }, simulated.samples)
