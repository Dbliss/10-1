import math
import random

from ohhell.arena import deal_round
from ohhell.bots import make_bot
from ohhell.engine import RoundState
from ohhell.learn.bid_floor import _seat_by_seat, sample_loss
from ohhell.rules import Rules


def test_seat_by_seat_keeps_our_hand_and_reproduces_earlier_bids():
    rng = random.Random(3)
    n, size = 4, 5
    rules = Rules(n, 10)
    hands, trump = deal_round(rng, n, size)
    state = RoundState(rules, hands, trump, dealer=0)
    bots = [make_bot('heuristic', seat) for seat in range(n)]
    order = []
    ours = 3
    while state.to_act != ours:
        order.append(state.to_act)
        state.apply(bots[state.to_act].act(state.observation(state.to_act)))
    drawn, tries = _seat_by_seat(rng, rules, size, ours, hands[ours], trump, 0,
                                 order, state.bids, bots, 500)
    assert drawn is not None and tries >= len(order)
    assert drawn[ours] == hands[ours]
    assert all(h.bit_count() == size for h in drawn)
    union = 0
    for h in drawn:
        assert not union & h
        union |= h
    assert not (union >> trump) & 1
    world = RoundState(rules, drawn, trump, 0)
    for seat in order:
        assert bots[seat].act(world.observation(seat)) == state.bids[seat]
        world.apply(state.bids[seat])


def test_sample_loss_smoothing():
    assert math.isclose(sample_loss([1, 1, 2, 1], 1, 2, 0.0), -math.log(0.75))
    assert math.isclose(sample_loss([0, 0], 1, 1, 0.5), -math.log(0.5 / 3))
    assert sample_loss([], 0, 3, 0.25) is None
