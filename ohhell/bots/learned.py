"""Bots that use the models trained by self-play."""

from ..learn.models import load_models
from .base import Bot
from .simulation import BidAnalysis, SimulationBot


def _expected(dist, bid, score):
    return sum(p * score.points(bid, k) for k, p in enumerate(dist))


class NetBot(Bot):
    """Decides instantly from the models alone, with no simulation."""

    def __init__(self, seed=0, models=None, bid_models=None, play_models=None):
        super().__init__(seed)
        self.models = load_models(models, bid_models, play_models)

    def bid(self, obs):
        options = obs.legal_actions()
        dists = self.models.bid_distributions(
            obs.hand, obs.trump_card, obs.n, obs.hand_size, obs.seat, obs.dealer,
            obs.bids, options)
        score = obs.rules.scoring
        return max(options, key=lambda b: _expected(dists[b], b, score))

    def play(self, obs):
        cards = obs.legal_actions()
        if len(cards) == 1:
            return cards[0]
        values = self.models.card_values(obs, obs.seat, obs.hand, cards)
        return cards[values.index(max(values))]


class HybridBot(SimulationBot):
    """Simulation with the models acting as a prior.

    Bids average the simulated and predicted trick odds, ``bid_blend`` being
    the model's share (0 bids from the simulation alone). For cards, the model
    shortlists the ``shortlist`` most promising options, and its estimate counts
    as ``prior_worlds`` extra simulated deals. Exact endgames ignore the model.

    ``blend_pivot`` (0 = off) scales both model weights by card count:
    ``(cards - 1) / (blend_pivot - 1)``, cards being the hand size for bids and
    the cards in hand for play. One card trusts the simulation alone, the pivot
    gives the fixed weights, and larger hands lean further on the model (the
    bid share is capped at 1).
    """

    def __init__(self, seed=0, models=None, shortlist=4, prior_worlds=16,
                 bid_blend=0.5, inference=True, bid_models=None, play_models=None,
                 blend_pivot=0, **kwargs):
        super().__init__(seed=seed, inference=inference, **kwargs)
        if blend_pivot and blend_pivot <= 1:
            raise ValueError('blend_pivot must be 0 (off) or greater than 1.')
        self.models = load_models(models, bid_models, play_models)
        self.shortlist = shortlist
        self.prior_worlds = prior_worlds
        self.bid_blend = bid_blend
        self.blend_pivot = blend_pivot

    def model_scale(self, cards):
        """The multiplier on the model's weight at this card count."""
        if not self.blend_pivot:
            return 1.0
        return max(0.0, (cards - 1) / (self.blend_pivot - 1))

    def analyse_bids(self, obs):
        simulated = super().analyse_bids(obs)
        blend = min(1.0, self.bid_blend * self.model_scale(obs.hand_size))
        if not blend:  # bids from the simulation alone
            return simulated
        predicted = self.models.bid_distributions(
            obs.hand, obs.trump_card, obs.n, obs.hand_size, obs.seat, obs.dealer,
            obs.bids, list(simulated.distributions))
        distributions = {
            bid: [(1.0 - blend) * s + blend * p for s, p in zip(sim, predicted[bid])]
            for bid, sim in simulated.distributions.items()
        }
        score = obs.rules.scoring
        return BidAnalysis(distributions, {
            bid: _expected(dist, bid, score) for bid, dist in distributions.items()
        })

    def play(self, obs):
        cards = obs.legal_actions()
        if len(cards) == 1:
            return cards[0]
        remaining = obs.hand.bit_count() * obs.n - obs.trick_size
        if remaining <= self.endgame_cards:
            return super().play(obs)

        predicted = dict(zip(cards, self.models.card_values(
            obs, obs.seat, obs.hand, cards)))
        kept = sorted(cards, key=predicted.get, reverse=True)[:self.shortlist]
        simulated = self.evaluate_cards(obs, kept)
        weight = self.prior_worlds * self.model_scale(obs.hand.bit_count())
        return max(kept, key=lambda c: (self.worlds * simulated[c] + weight * predicted[c])
                   / (self.worlds + weight))
