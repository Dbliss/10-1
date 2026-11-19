"""Compact public-information action scorer, shared with native/core.c.

32-32-16-1 float MLP. Scores are in round points, with the same VALUE_SCALE
as the root network. No other seat's sampled hand enters these features.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

from ..cards import EFFECTIVE_SUIT, OUTRANKS, POWER, SUIT_CARDS, beats, cards_in
from ..engine import legal_card_mask
from .models import VALUE_SCALE
from .nn import MLP

FEATURE_VERSION = 1
SIZES = (32, 32, 16, 1)
MODEL_DIR = Path(__file__).parents[1] / 'models' / 'rollout_candidate'


def feature_matrix(obs, seat, hand, cards):
    t = obs.trump_suit
    seen = obs.played | hand | (1 << obs.trump_card)
    need = obs.bids[seat] - obs.tricks_won[seat]
    call = sum(max(0, b - w) for b, w in zip(obs.bids, obs.tricks_won)) - hand.bit_count()
    score = obs.rules.scoring
    common = [hand.bit_count() / 10, (hand & SUIT_CARDS[t][t]).bit_count() / 10,
              need / 10, obs.bids[seat] / 10, obs.tricks_won[seat] / 10,
              obs.n / 8, obs.trick_size / obs.n, obs.hand_size / 10,
              (obs.leader - seat) % obs.n / obs.n, (obs.dealer - seat) % obs.n / obs.n,
              -1 if obs.best_seat < 0 else (obs.best_seat - seat) % obs.n / obs.n,
              call / 10, score.points_per_trick / 10, score.made_zero / 10,
              score.penalty_per_trick / 10, float(obs.lead_suit < 0),
              float(need <= 0), float(need >= hand.bit_count())]
    rows = []
    for card in cards:
        suit = EFFECTIVE_SUIT[t][card]
        above = OUTRANKS[t][card]
        suit_cards = SUIT_CARDS[t][suit]
        below = suit_cards & ~above & ~(1 << card)
        rows.append([POWER[t][card] / 14, float(suit == t),
                     float(obs.best_card < 0 or beats(card, obs.best_card, t)),
                     float(not above & ~seen), (hand & suit_cards).bit_count() / 10,
                     (above & ~seen).bit_count() / 14, (above & hand).bit_count() / 14,
                     (below & hand).bit_count() / 14,
                     *common,
                     sum(bool(v & (1 << suit)) for v in obs.voids) / obs.n,
                     sum(bool(v & (1 << t)) for v in obs.voids) / obs.n,
                     *[float(suit == s) for s in (t, 3 - t,
                         *[s for s in range(4) if s not in (t, 3 - t)])]])
    return np.asarray(rows, dtype=np.float32).reshape(-1, SIZES[0])


def save_network(model, path):
    arrays = {f'w{i}': w for i, w in enumerate(model.weights)}
    arrays.update({f'b{i}': b for i, b in enumerate(model.biases)})
    np.savez_compressed(path, **arrays, architecture=np.asarray('ohhell-rollout-float'),
                        feature_version=np.asarray(FEATURE_VERSION))


@lru_cache(maxsize=8)
def load_network(directory=None):
    path = (Path(directory) if directory else MODEL_DIR) / 'play.npz'
    with np.load(path, allow_pickle=False) as data:
        if (str(data['architecture']) != 'ohhell-rollout-float'
                or int(data['feature_version']) != FEATURE_VERSION):
            raise ValueError('Unsupported rollout architecture/feature version')
    model = MLP.load(path)
    if (len(model.weights) != 3 or len(model.biases) != 3
            or any(w.shape != (a, b) or bias.shape != (b,)
                   for w, bias, a, b in zip(model.weights, model.biases, SIZES, SIZES[1:]))
            or any(p.dtype != np.float32 or not np.isfinite(p).all() for p in model.parameters())):
        raise ValueError('Invalid rollout weights')
    for p in model.parameters():
        p.flags.writeable = False
    return model


class RolloutPolicy:
    def __init__(self, model):
        self.model = model

    def card_values(self, obs, seat, hand, cards):
        return (self.model(feature_matrix(obs, seat, hand, cards))[:, 0] * VALUE_SCALE).tolist()

    def __call__(self, state):
        seat = state.to_act
        hand = state.hands[seat]
        cards = cards_in(legal_card_mask(hand, state.lead_suit, state.trump_suit))
        values = self.card_values(state, seat, hand, cards)
        return cards[values.index(max(values))]
