"""Small correction to the score gaps between close bids on long hands."""

from functools import lru_cache
import json
import math

from .bid_calibration import coefficient_path


GAP_LIMIT = 6.0
MIN_HAND = 8
N_FEATURES = 11


@lru_cache(maxsize=8)
def load_weights(path):
    weights = tuple(json.loads(coefficient_path(path).read_text())['weights'])
    if len(weights) != N_FEATURES or any(not math.isfinite(w) for w in weights):
        raise ValueError('Invalid bid ranking weights')
    return weights


def features(odds, bid, players, before):
    size = len(odds) - 1
    below = sum(odds[:bid])
    above = 1.0 - odds[bid] - below
    short = sum((bid - k) * p for k, p in enumerate(odds[:bid]))
    extra = sum((k - bid) * odds[k] for k in range(bid + 1, len(odds)))
    make = odds[bid]
    return (float(bid == 0), bid / size, make, below, short, extra,
            above * bid / size, make * bid / size, make * size / 10,
            make * players / 8, below * before / (players - 1))


def contenders(values):
    best = max(values, key=values.get)
    return best, [b for b, value in values.items() if values[best] - value <= GAP_LIMIT]


def pick(values, distributions, size, players, before, weights):
    if size < MIN_HAND:
        return max(values, key=values.get)
    best, candidates = contenders(values)
    if len(candidates) == 1:
        return best
    return max(candidates, key=lambda b: values[b] + sum(
        w * x for w, x in zip(weights, features(distributions[b], b, players, before))))
