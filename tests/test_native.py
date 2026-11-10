"""The compiled core must behave like the Python code it speeds up."""

import random
import time

import pytest

from ohhell import native
from ohhell.arena import deal_round
from ohhell.bots import make_bot
from ohhell.bots.heuristic import expected_tricks, state_card, table_read
from ohhell.bots.simulation import SimulationBot, shift_tricks
from ohhell.engine import BIDDING, PLAYING, RoundState
from ohhell.rules import Rules

needs_core = pytest.mark.skipif(not native.available, reason='core not built')


def random_round(rng, players=4, hand_size=8):
    hands, trump = deal_round(rng, players, hand_size)
    state = RoundState(Rules(players, hand_size), hands, trump, 0)
    while state.phase == BIDDING:
        state.apply(rng.choice(state.legal_actions()))
    return state


@needs_core
def test_card_choice_matches_python_everywhere():
    rng = random.Random(1)
    checked = 0
    for _ in range(3000):
        state = random_round(rng, rng.choice([2, 3, 4, 6, 8]), rng.randint(1, 6))
        while state.phase == PLAYING:
            seat = state.to_act
            hand = state.hands[seat]
            assert state_card(state) == native.library.oh_choose_card(
                hand, state.trump_suit, state.lead_suit, state.best_card, state.trick_size,
                state.n, state.played | hand | (1 << state.trump_card),
                state.bids[seat] - state.tricks_won[seat], 1, 0)
            behind, scarce = table_read(seat, state.n, state.bids, state.tricks_won,
                                        state.trick_size, hand.bit_count())
            assert state_card(state, True) == native.library.oh_choose_card(
                hand, state.trump_suit, state.lead_suit, state.best_card, state.trick_size,
                state.n, state.played | hand | (1 << state.trump_card),
                state.bids[seat] - state.tricks_won[seat], int(behind), int(scarce))
            checked += 1
            state.apply(rng.choice(state.legal_actions()))
    assert checked > 10000


@needs_core
def test_trick_estimates_match_python():
    rng = random.Random(2)
    for _ in range(3000):
        n = rng.choice([2, 4, 6, 8])
        size = rng.randint(1, min(10, 51 // n))
        hands, trump = deal_round(rng, n, size)
        leads = rng.random() < 0.5
        assert expected_tricks(hands[0], trump, n, size, leads) == pytest.approx(
            native.library.oh_expected_tricks(hands[0], trump, n, size, int(leads)), abs=1e-9)


@needs_core
@pytest.mark.parametrize('field', ['mix', 'previous'])
@pytest.mark.parametrize('inference', [False, True])
@pytest.mark.parametrize('plays', [0, 6, 21, 30])
def test_card_values_agree_with_python(field, inference, plays):
    rng = random.Random(plays + 7)
    state = random_round(rng)
    for _ in range(plays):
        state.apply(rng.choice(state.legal_actions()))
    obs = state.observation(state.to_act)
    kwargs = dict(seed=3, worlds=1200, inference=inference, field=field)
    slow = SimulationBot(fast=False, **kwargs).evaluate_cards(obs)
    fast = SimulationBot(fast=True, **kwargs).evaluate_cards(obs)
    for card in slow:
        assert fast[card] == pytest.approx(slow[card], abs=5.0)


@needs_core
@pytest.mark.parametrize('field', ['mix', 'previous'])
@pytest.mark.parametrize('bids_made', [0, 2, 3])
def test_bid_odds_agree_with_python(field, bids_made):
    rng = random.Random(bids_made)
    hands, trump = deal_round(rng, 4, 8)
    state = RoundState(Rules(4, 8), hands, trump, 0)
    for _ in range(bids_made):
        state.apply(rng.choice(state.legal_actions()))
    obs = state.observation(state.to_act)
    kwargs = dict(seed=5, bid_worlds=1000, inference=True, field=field)
    slow = SimulationBot(fast=False, **kwargs).analyse_bids(obs)
    fast = SimulationBot(fast=True, **kwargs).analyse_bids(obs)
    assert slow.distributions.keys() == fast.distributions.keys()
    for bid in slow.values:
        assert fast.values[bid] == pytest.approx(slow.values[bid], abs=5.0)


@pytest.mark.parametrize('fast', [False, True])
def test_time_limit_stops_a_turn_on_time(fast):
    if fast and not native.available:
        pytest.skip('core not built')
    rng = random.Random(4)
    state = random_round(rng, 6, 8)
    obs = state.observation(state.to_act)
    bot = SimulationBot(seed=1, worlds=8, fast=fast, time_ms=20)
    start = time.perf_counter()
    bot.act(obs)
    assert time.perf_counter() - start < 0.5
    assert bot.deadline is None


def test_confidence_shifts_the_trick_odds():
    assert shift_tricks([0.0, 1.0, 0.0], 0.5) == [0.0, 0.5, 0.5]
    assert shift_tricks([0.0, 1.0, 0.0], -0.5) == [0.5, 0.5, 0.0]
    assert shift_tricks([0.5, 0.5], 1) == [0.0, 1.0]


def test_confidence_moves_bids_the_right_way():
    rng = random.Random(9)
    totals = {-0.75: 0, 0.0: 0, 0.75: 0}
    for game in range(150):
        hands, trump = deal_round(rng, 6, 8)
        state = RoundState(Rules(6, 8), hands, trump, 5)
        obs = state.observation(state.to_act)
        for confidence in totals:
            bot = make_bot(f'simulation:confidence={confidence}', game)
            totals[confidence] += bot.bid(obs)
    assert totals[-0.75] < totals[0.0] < totals[0.75]
