"""Float NNUE action values with stable sparse features and incremental sums.

The feature-transformer preactivation is updated by adding/removing changed
feature rows. Candidate cards branch from the same position accumulator. No
hidden hands enter this representation. This prototype is not quantized.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

from ..cards import OUTRANKS, SUIT_OF, beats, cards_in
from .models import VALUE_SCALE
from .nn import MLP

FEATURE_VERSION = 1
POSITION_INPUTS = 430
ACTION_OFFSET = POSITION_INPUTS
INPUTS = POSITION_INPUTS + 54
MODEL_DIR = Path(__file__).parents[1] / 'models' / 'nnue_candidate'


def canonical_card(card, trump):
    # Fixed rotation, never reordered according to cards held/played.
    return ((SUIT_OF[card] - trump) % 4) * 13 + card % 13


def position_features(obs, seat, hand):
    if obs.blind or not 1 <= obs.hand_size <= 10:
        raise ValueError('NNUE features require a visible hand of size 1..10')
    x = np.zeros(INPUTS, dtype=np.float32)
    trump = obs.trump_suit
    for offset, mask in ((0, hand), (52, obs.played)):
        for card in cards_in(mask):
            x[offset + canonical_card(card, trump)] = 1
    if obs.best_card >= 0:
        x[104 + canonical_card(obs.best_card, trump)] = 1
    x[156 + obs.trump_card % 13] = 1
    x[169 + obs.n - 2] = 1
    x[176 + obs.hand_size - 1] = 1
    x[186 + obs.trick_size] = 1
    x[194 + (obs.leader - seat) % obs.n] = 1
    x[202 + (obs.dealer - seat) % obs.n] = 1
    x[210 + (0 if obs.best_seat < 0 else 1 + (obs.best_seat - seat) % obs.n)] = 1
    for relative in range(obs.n):
        other = (seat + relative) % obs.n
        at = 219 + relative * 26
        if obs.bids[other] < 0:
            raise ValueError('NNUE play features require completed bidding')
        x[at + obs.bids[other]] = 1
        x[at + 11 + obs.tricks_won[other]] = 1
        for suit in range(4):
            x[at + 22 + (suit - trump) % 4] = (obs.voids[other] >> suit) & 1
    scoring = obs.rules.scoring
    x[427:430] = (scoring.points_per_trick / 10, scoring.made_zero / 10,
                  scoring.penalty_per_trick / 10)
    return x


def action_features(obs, hand, card):
    best = obs.best_card
    trump = obs.trump_suit
    wins = best < 0 or beats(card, best, trump)
    unseen = ~(obs.played | hand | (1 << obs.trump_card))
    return canonical_card(card, trump), float(wins), float(not OUTRANKS[trump][card] & unseen)


def feature_matrix(obs, seat, hand, cards):
    rows = np.tile(position_features(obs, seat, hand), (len(cards), 1))
    for row, card in zip(rows, cards):
        index, wins, top = action_features(obs, hand, card)
        row[ACTION_OFFSET + index] = 1
        row[-2:] = wins, top
    return rows


def save_network(model, path):
    arrays = {f'w{i}': w for i, w in enumerate(model.weights)}
    arrays.update({f'b{i}': b for i, b in enumerate(model.biases)})
    arrays.update(architecture=np.asarray('ohhell-nnue-float'),
                  feature_version=np.asarray(FEATURE_VERSION))
    np.savez_compressed(path, **arrays)


@lru_cache(maxsize=8)
def load_network(directory=None):
    path = (Path(directory) if directory else MODEL_DIR) / 'play.npz'
    if not path.exists():
        raise FileNotFoundError(
            f'No NNUE checkpoint at {path}. The experimental candidate was removed; '
            'train one with `python -m ohhell.learn.train_nnue` or pass play_models=FOLDER.')
    with np.load(path, allow_pickle=False) as data:
        if (str(data['architecture']) != 'ohhell-nnue-float'
                or int(data['feature_version']) != FEATURE_VERSION):
            raise ValueError('Unsupported NNUE architecture/feature version')
    model = MLP.load(path)
    if (model.weights[0].shape[0] != INPUTS or model.weights[-1].shape[1] != 1
            or any(not np.isfinite(p).all() for p in model.parameters())):
        raise ValueError('Invalid NNUE weights')
    # All mutation lives in each bot's evaluator, never in cached weights.
    for p in model.parameters():
        p.flags.writeable = False
    return model


class Evaluator:
    def __init__(self, network):
        self.network = network
        self.previous = None
        self.accumulator = None
        self.updates = 0
        self.rebuilds = 0

    def accumulate(self, features):
        w = self.network.weights[0]
        if self.previous is None or self.updates >= 64:
            active = np.flatnonzero(features)
            self.accumulator = (self.network.biases[0] +
                                (w[active] * features[active, None]).sum(axis=0))
            self.updates = 0
            self.rebuilds += 1
        else:
            delta = features - self.previous
            changed = np.flatnonzero(delta)
            if len(changed):
                self.accumulator += (w[changed] * delta[changed, None]).sum(axis=0)
            self.updates += 1
        self.previous = features.copy()
        return self.accumulator

    def card_values(self, obs, seat, hand, cards):
        base = self.accumulate(position_features(obs, seat, hand))
        first = np.tile(base, (len(cards), 1))
        w = self.network.weights[0]
        for row, card in zip(first, cards):
            index, wins, top = action_features(obs, hand, card)
            row += w[ACTION_OFFSET + index] + wins * w[-2] + top * w[-1]
        output = np.maximum(first, 0)
        for i in range(1, len(self.network.weights)):
            output = output @ self.network.weights[i] + self.network.biases[i]
            if i < len(self.network.weights) - 1:
                output = np.maximum(output, 0)
        return (output[:, 0] * VALUE_SCALE).tolist()
