"""Experimental NNUE card evaluator; all v1 search and bidding stay intact."""

from ..learn.nnue import Evaluator, load_network
from .learned import HybridBot


class NNUEModels:
    def __init__(self, original, evaluator):
        self.bid_distributions = original.bid_distributions
        self.card_values = evaluator.card_values
        self.evaluator = evaluator


class NNUEBot(HybridBot):
    def __init__(self, seed=0, models=None, bid_models=None, play_models=None, **kwargs):
        super().__init__(seed=seed, bid_models=bid_models or models, **kwargs)
        self.models = NNUEModels(self.models, Evaluator(load_network(play_models or models)))
