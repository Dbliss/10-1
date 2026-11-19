"""Own-bid calibration must not change the opponent model or card policy."""

import random

import pytest

from ohhell import native
from ohhell.arena import deal_round
from ohhell.bots import make_bot
from ohhell.bots.simulation import SimulationBot
from ohhell.cards import FULL_DECK, card_from_id, cards_in, mask_of
from ohhell.engine import RoundState
from ohhell.rules import Rules


def screenshot_position():
    hand = mask_of(card_from_id(name) for name in (
        'Jack_of_Hearts', '7_of_Hearts', '4_of_Hearts',
        'Jack_of_Spades', '7_of_Spades', 'King_of_Clubs',
        'Queen_of_Clubs', '2_of_Clubs', 'King_of_Diamonds', '10_of_Diamonds'))
    trump = card_from_id('2_of_Hearts')
    rest = cards_in(FULL_DECK & ~hand & ~(1 << trump))
    hands = [hand] + [mask_of(rest[i:i + 10]) for i in (0, 10, 20)]
    state = RoundState(Rules(4, 10), hands, trump, 2)
    state.apply(2)  # AI 3 has bid; it is now the user's turn.
    return state.observation(0)


def test_calibrated_own_bid_includes_five_for_the_reported_hand():
    obs = screenshot_position()
    old = SimulationBot(fast=False)
    candidate = SimulationBot(fast=False, bid_estimate=3)
    assert old._bid_options(obs) == [0, 1, 2, 3, 4]
    assert candidate._bid_options(obs) == [1, 2, 3, 4, 5]
    assert candidate.estimate == candidate.belief.estimate == 1


@pytest.mark.parametrize('version', [1, 2, 3])
def test_default_and_explicit_zero_preserve_existing_shortlists(version):
    rng = random.Random(913)
    old = SimulationBot(fast=False, estimate=version)
    zero = SimulationBot(fast=False, estimate=version, bid_estimate=0)
    for _ in range(60):
        n = rng.randint(2, 8)
        size = rng.randint(1, min(10, 51 // n))
        hands, trump = deal_round(rng, n, size)
        state = RoundState(Rules(n, size), hands, trump, rng.randrange(n))
        obs = state.observation(state.to_act)
        assert old._bid_options(obs) == zero._bid_options(obs)
        options = SimulationBot(fast=False, bid_estimate=3)._bid_options(obs)
        assert options and set(options) <= set(obs.legal_actions())


@pytest.mark.parametrize('fast', [False, True])
def test_same_options_have_identical_analysis_with_own_bid_override(fast):
    if fast and not native.available:
        pytest.skip('native core not built')
    settings = dict(seed=123, fast=fast, inference=True, field='mix', bid_worlds=64)
    old = SimulationBot(**settings)
    candidate = SimulationBot(bid_estimate=3, **settings)
    obs = screenshot_position()
    # The shortlist is the only changed input. Equal options must produce
    # bit-identical samples and values, with original opponent bidding.
    a = old._analyse_bids(obs, [2, 3, 4])
    b = candidate._analyse_bids(obs, [2, 3, 4])
    assert a == b
    assert candidate.belief.estimate == 1
    if fast:
        assert candidate.simulator.params.estimate == 1


def test_hybrid_accepts_independent_selector_without_changing_opponents():
    bot = make_bot('hybrid:bid_estimate=3,worlds=4,bid_worlds=4,fast=0', seed=7)
    assert bot.bid_estimate == 3 and bot.estimate == bot.belief.estimate == 1
    result = bot.analyse_bids(screenshot_position())
    assert list(result.values) == [1, 2, 3, 4, 5]


def test_invalid_own_bid_estimate_fails_early():
    with pytest.raises(ValueError, match='bid_estimate'):
        SimulationBot(fast=False, bid_estimate=4)
