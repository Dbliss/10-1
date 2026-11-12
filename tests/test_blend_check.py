"""The blend check must record without changing v2's play, and score correctly."""

import random

import pytest

from ohhell.arena import deal_round
from ohhell.bots import make_bot
from ohhell.engine import DONE, RoundState
from ohhell.learn.blend_check import Recorder, bid_losses, card_losses, compare
from ohhell.rules import Rules, Scoring

SPEC = 'hybrid:field=previous,worlds=8,bid_worlds=16,fast=0'


def test_recorder_plays_exactly_as_v2():
    kwargs = dict(field='previous', models='ohhell/models',
                  worlds=8, bid_worlds=16, fast=False)
    hands, trump = deal_round(random.Random(3), 4, 6)
    a, b = (RoundState(Rules(4, 6), hands, trump, 0) for _ in range(2))
    recorders = [Recorder(seed=s, **kwargs) for s in range(4)]
    v2 = [make_bot(SPEC, s) for s in range(4)]
    while a.phase != DONE:
        x = recorders[a.to_act].act(a.observation(a.to_act))
        y = v2[b.to_act].act(b.observation(b.to_act))
        assert x == y
        a.apply(x)
        b.apply(y)
    assert sum(len(r.bids) for r in recorders) == 4


def test_scores_and_differences():
    scoring = Scoring()
    bid = {'kind': 'bid', 'round': 1, 'bid': 1, 'tricks': 1, 'points': scoring.points(1, 1),
           'sim': [0.0, 1.0], 'net': [0.5, 0.5]}
    assert bid_losses(bid, 0.0, scoring)['brier'] == 0.0
    assert bid_losses(bid, 1.0, scoring)['brier'] == pytest.approx(0.5)
    card = {'kind': 'card', 'round': 1, 'points': 10, 'sim': 10.0, 'net': 0.0}
    assert card_losses(card, 0.5)['points_se'] == 25.0
    result = compare([bid, dict(bid, round=2)], bid_losses, ('brier',), 0.5, scoring)
    assert result['brier']['best_w'] == 0.0
    base = next(row for row in result['brier']['by_w'] if row['w'] == 0.5)
    assert base['diff_vs_v2'] == 0.0
