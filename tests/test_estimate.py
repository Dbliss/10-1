"""The version 2 quick trick estimate (ohhell/estimate.py) and its C port."""

import random

import pytest

from ohhell import estimate, native
from ohhell.arena import deal_round
from ohhell.bots import make_bot
from ohhell.bots.simulation import SimulationBot, heuristic_bid
from ohhell.cards import SUIT_CARDS, SUIT_OF, card_from_id, cards_in, mask_of
from ohhell.engine import BIDDING, RoundState
from ohhell.rules import Rules, Scoring

needs_core = pytest.mark.skipif(not native.available, reason='core not built')


def ids(*names):
    return mask_of(card_from_id(name) for name in names)


def test_weights_match_feature_lists():
    side, trump = estimate._weights()
    assert len(side) == len(estimate.SIDE) == native.EST_SIDE
    assert len(trump) == len(estimate.TRUMP) == native.EST_TRUMP
    rows = estimate.card_features(ids('Ace_of_Spades', '2_of_Hearts'), card_from_id('5_of_Hearts'), 4, 2)
    for _, is_trump, features, _ in rows:
        assert len(features) == len(trump if is_trump else side)


def test_top_trumps_are_certain():
    hand = ids('Jack_of_Hearts', 'Jack_of_Diamonds', 'Ace_of_Hearts', 'King_of_Hearts', 'Queen_of_Hearts')
    trump = card_from_id('2_of_Hearts')
    assert estimate.certain_tricks(hand, trump) == 5
    assert estimate.expected_tricks(hand, trump, 4, 5) == pytest.approx(5.0)
    # A gap above the Queen (the King unseen) leaves only the bowers and Ace certain.
    gap = ids('Jack_of_Hearts', 'Jack_of_Diamonds', 'Ace_of_Hearts', 'Queen_of_Hearts', '3_of_Clubs')
    assert estimate.certain_tricks(gap, trump) == 3


def test_a_turned_up_card_counts_as_seen():
    # With the right bower turned up, the left bower is the top trump left.
    hand = ids('Jack_of_Diamonds', '4_of_Clubs')
    assert estimate.certain_tricks(hand, card_from_id('Jack_of_Hearts')) == 1
    assert estimate.certain_tricks(hand, card_from_id('9_of_Hearts')) == 0


def test_leading_first_helps_an_ace():
    hand = ids('Ace_of_Spades', '2_of_Clubs', '3_of_Clubs', '4_of_Diamonds',
               '5_of_Diamonds', '6_of_Clubs', '7_of_Diamonds', '8_of_Clubs', '9_of_Diamonds', '3_of_Spades')
    trump = card_from_id('2_of_Hearts')
    ace = card_from_id('Ace_of_Spades')
    first = estimate.card_win_probs(hand, trump, 4, 10, position=0)[ace]
    second = estimate.card_win_probs(hand, trump, 4, 10, position=1)[ace]
    # Known limit: the dealer's (last seat's) Aces are rated close to the
    # leader's, although in play-outs they are thrown away more often. Seat
    # totals are still calibrated; see BOT_EXPERIMENTS.md.
    assert first > second + 0.03
    assert first > 0.8


def test_probabilities_are_probabilities():
    rng = random.Random(4)
    for _ in range(300):
        n = rng.randint(2, 8)
        h = rng.randint(1, min(10, 51 // n))
        hands, trump = deal_round(rng, n, h)
        probs = estimate.card_win_probs(hands[0], trump, n, h, rng.randrange(n))
        assert sorted(probs) == sorted(cards_in(hands[0]))
        assert all(0.0 <= p <= 1.0 for p in probs.values())


@needs_core
@pytest.mark.parametrize('version', [2, 3])
def test_native_matches_python(version):
    sim = native.Simulator(1, True, 4, 'previous', 12, Scoring(), estimate=version)
    rng = random.Random(9)
    for _ in range(2000):
        n = rng.randint(2, 8)
        h = rng.randint(1, min(10, 51 // n))
        hands, trump = deal_round(rng, n, h)
        position = rng.randrange(n)
        expected = estimate.card_win_probs(hands[0], trump, n, h, position, version=version)
        total, probs = sim.expected_tricks2(hands[0], trump, n, h, position)
        # The C code lists cards suit by suit, strongest first, as Python builds them.
        t = SUIT_OF[trump]
        order = [c for row in estimate.card_features(hands[0], trump, n, h, position)
                 for c in [row[0]]]
        assert total == pytest.approx(sum(expected.values()), abs=1e-9)
        assert probs == pytest.approx([expected[c] for c in order], abs=1e-9)
        assert len(order) == len(cards_in(hands[0] & sum(SUIT_CARDS[t])))


def test_simulation_option_selects_the_estimate():
    rng = random.Random(2)
    hands, trump = deal_round(rng, 4, 8)
    state = RoundState(Rules(4, 8), hands, trump, 0)
    obs = state.observation(1)
    old = make_bot('simulation:worlds=4,bid_worlds=4', 1)
    new = make_bot('simulation:worlds=4,bid_worlds=4,estimate=2', 1)
    assert old.estimate == 1 and new.estimate == 2
    assert set(new._bid_options(obs)) <= set(obs.legal_actions())
    assert heuristic_bid(hands[2], obs, 2, 2) in range(9)
    with pytest.raises(ValueError):
        SimulationBot(estimate=4)


@pytest.mark.parametrize('fast', [True, False])
def test_bot_with_new_estimate_bids_and_plays(fast):
    if fast and not native.available:
        pytest.skip('core not built')
    rng = random.Random(5)
    bot = SimulationBot(seed=3, worlds=8, bid_worlds=8, inference=True, field='previous',
                        fast=fast, estimate=2)
    hands, trump = deal_round(rng, 5, 6)
    state = RoundState(Rules(5, 6), hands, trump, 2)
    while state.phase == BIDDING:
        state.apply(bot.act(state.observation(state.to_act)))
    for _ in range(10):
        state.apply(bot.act(state.observation(state.to_act)))


def test_version_2_is_padded_and_unspread():
    side, trump, spread = estimate._coefficients(2)
    assert len(side) == len(estimate.SIDE)
    assert side[-len(estimate.DISCARDS):] == (0.0,) * len(estimate.DISCARDS)
    assert spread == estimate.NO_SPREAD
    dist = [0.2, 0.5, 0.3]
    assert estimate.widen(dist, 1.0) == dist


def test_version_3_spread_changes_the_bid_distribution():
    assert estimate.spread(3) != estimate.NO_SPREAD
    rng = random.Random(6)
    hands, trump = deal_round(rng, 4, 8)
    probs = estimate.card_win_probs(hands[0], trump, 4, 8, 1, version=3)
    plain = estimate.trick_distribution(list(probs.values()))
    spread = estimate.bid_distribution(hands[0], trump, 4, 8, 1, version=3)
    assert sum(spread) == pytest.approx(1.0)
    assert spread != pytest.approx(plain)
    # A temperature below 1 sharpens: the likeliest count gets likelier.
    if estimate.spread_temperature(8, 4, estimate.spread(3)) < 1.0:
        assert max(spread) > max(plain)


@needs_core
@pytest.mark.parametrize('version', [2, 3])
def test_native_bid_odds_agree_with_python(version):
    """Imagined bidders (with the spread) and inference agree between C and Python."""
    rng = random.Random(10 + version)
    hands, trump = deal_round(rng, 4, 7)
    state = RoundState(Rules(4, 7), hands, trump, 0)
    state.apply(state.legal_actions()[1])  # one earlier bid, so inference is on
    obs = state.observation(state.to_act)
    kwargs = dict(seed=4, bid_worlds=3000, inference=True, field='previous', estimate=version)
    fast = SimulationBot(fast=True, **kwargs).analyse_bids(obs).distributions
    slow = SimulationBot(fast=False, **kwargs).analyse_bids(obs).distributions
    assert fast.keys() == slow.keys()
    for bid in fast:
        assert fast[bid] == pytest.approx(slow[bid], abs=0.04)


@needs_core
@pytest.mark.parametrize('version', [1, 2, 3])
def test_native_imagined_bids_match_python(version):
    sim = native.Simulator(1, True, 4, 'previous', 12, Scoring(), estimate=version)
    rng = random.Random(3)
    for _ in range(600):
        n = rng.randint(2, 8)
        h = rng.randint(1, min(10, 51 // n))
        hands, trump = deal_round(rng, n, h)
        dealer = rng.randrange(n)
        obs = RoundState(Rules(n, min(10, 51 // n)), hands, trump, dealer).observation(0)
        for seat in range(n):
            assert sim.heuristic_bid(hands[seat], trump, n, h, seat, dealer) == heuristic_bid(
                hands[seat], obs, seat, version)
