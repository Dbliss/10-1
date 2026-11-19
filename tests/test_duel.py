import math
from types import SimpleNamespace

import pytest

from ohhell import duel


def test_rounds_per_game_follow_the_down_blind_up_format():
    assert duel.rounds_per_game(4) == 21
    assert duel.rounds_per_game(6) == 17  # 51 // 6 = 8 cards at most
    assert duel.rounds_per_game(8) == 13


def test_draw_round_is_reproducible_and_uses_the_field():
    field = (('skilled', 1.0), ('casual', 1.0))
    a = duel.draw_round(123, field)
    b = duel.draw_round(123, field)
    assert a[0] == b[0] and a[2:] == b[2:]
    players, rules, label = a[0], a[1], a[2]
    assert label in rules.hand_sizes
    assert set(a[7]) <= {'skilled', 'casual'} and len(a[7]) == players


def test_an_equal_duel_round_has_zero_difference():
    row = duel.duel_round((0, 77, 'skilled', 'skilled', (('casual', 1.0),)))
    assert row['diff'] == 0 and row['diverged'] is None


def test_obf_margin_matches_the_tournament_rule():
    assert duel.obf_radius(10.0, 100, 400) == pytest.approx(duel.FINAL_Z * 10 * 20 / 100)


def test_decide_uses_planned_margins():
    args = SimpleNamespace(min_rounds=10, max_rounds=1000, stopping="obf", plan=4000, futility=True,
                           smallest_gain=1.0)
    stats = duel.Running()
    for x in [5.0, 6.0] * 50:
        stats.add(x)
    verdict, radius = duel.decide(stats, args)
    assert verdict == 'better' and radius > 0
    zero = duel.Running()
    for _ in range(20):
        zero.add(0.0)
    assert duel.decide(zero, args)[0] == 'no gain'


def test_confidence_sequence_is_wider_than_a_plain_interval():
    plain = 1.96 * 10 / math.sqrt(4000)
    assert duel.cs_radius(10.0, 4000, 4000) > plain


def test_budgeted_bots_repeat_exactly():
    from ohhell import native
    if not native.available:
        pytest.skip('native core not built')
    spec = 'hybrid:field=previous,inference=1,card_budget=40,bid_budget=40,endgame_budget=20'
    row = duel.duel_round((1, 5, spec, spec, (('skilled', 1.0),)))
    assert row['diff'] == 0 and row['diverged'] is None
