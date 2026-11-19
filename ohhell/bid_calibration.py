"""Adjust exact-bid odds once, after the cheap bid rollouts finish."""

from functools import lru_cache
import json
import math
from pathlib import Path


def coefficient_path(path):
    path = Path(path)
    if not path.is_absolute() and not path.is_file():
        path = Path(__file__).resolve().parent.parent / path
    return path.resolve()


@lru_cache(maxsize=8)
def load_coefficients(path):
    weights = tuple(json.loads(coefficient_path(path).read_text())['log_odds_weights'])
    if len(weights) != 4 or any(not math.isfinite(w) or abs(w) > 10 for w in weights):
        raise ValueError('Invalid bid calibration coefficients')
    return weights


def features(hand_size, bid):
    return (1.0, hand_size / 10, float(bid == 0), bid / hand_size)


def adjust(distribution, hand_size, bid, weights):
    multiplier = math.exp(sum(w * x for w, x in zip(weights, features(hand_size, bid))))
    normalizer = 1.0 + distribution[bid] * (multiplier - 1.0)
    return [p * (multiplier if k == bid else 1.0) / normalizer
            for k, p in enumerate(distribution)]
