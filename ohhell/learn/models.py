"""Loading trained models and asking them questions."""

from functools import lru_cache
from pathlib import Path

import numpy as np

from ..cards import cards_in
from ..engine import legal_card_mask
from .features import MAX_TRICKS, bid_matrix, play_matrix
from .nn import MLP

MODEL_DIR = Path(__file__).parent.parent / 'models'
VALUE_SCALE = 30.0  # play model outputs are points divided by this


class Models:
    def __init__(self, bid, play):
        self.bid = bid
        self.play = play

    def bid_distributions(self, hand, trump_card, n, hand_size, seat, dealer, bids,
                          options):
        """For each bid, the predicted chance of taking exactly k tricks."""
        logits = self.bid(bid_matrix(hand, trump_card, n, hand_size, seat, dealer,
                                     bids, options))
        logits = logits[:, :hand_size + 1]
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        return {bid: probs[i].tolist() for i, bid in enumerate(options)}

    def card_values(self, view, seat, hand, cards):
        """Predicted final round points for this seat after each card."""
        output = self.play(play_matrix(view, seat, hand, cards))[:, 0]
        return (output * VALUE_SCALE).tolist()


class NetPolicy:
    """Picks the card the play model likes best, for use inside simulations."""

    def __init__(self, models):
        self.models = models

    def __call__(self, state):
        seat = state.to_act
        hand = state.hands[seat]
        cards = cards_in(legal_card_mask(hand, state.lead_suit))
        if len(cards) == 1:
            return cards[0]
        values = self.models.card_values(state, seat, hand, cards)
        return cards[values.index(max(values))]


@lru_cache(maxsize=8)
def load_models(directory=None):
    folder = Path(directory) if directory else MODEL_DIR
    return Models(MLP.load(folder / 'bid.npz'), MLP.load(folder / 'play.npz'))


__all__ = ['MAX_TRICKS', 'Models', 'NetPolicy', 'load_models']
