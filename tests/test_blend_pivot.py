"""Card-count blend: the model's weight grows with the cards held."""

import random
from types import SimpleNamespace

import pytest

from ohhell.arena import deal_round, play_game
from ohhell.bots import make_bot
from ohhell.bots.learned import HybridBot
from ohhell.engine import BIDDING, RoundState
from ohhell.rules import Rules


def bid_obs(hand_size, players=4, seed=7):
    hands, trump = deal_round(random.Random(seed), players, hand_size)
    state = RoundState(Rules(players, hand_size), hands, trump, players - 1)
    return state.observation(state.to_act)


def play_obs(hand_size, players=4, seed=8):
    hands, trump = deal_round(random.Random(seed), players, hand_size)
    state = RoundState(Rules(players, hand_size), hands, trump, players - 1)
    while state.phase == BIDDING:
        state.apply(state.legal_actions()[0])
    return state.observation(state.to_act)


@pytest.mark.parametrize('cards,scale', [(1, 0.0), (3, 0.4), (6, 1.0), (10, 1.8)])
def test_scale_is_linear_from_one_card_to_the_pivot(cards, scale):
    assert HybridBot(blend_pivot=6).model_scale(cards) == pytest.approx(scale)
    assert HybridBot().model_scale(cards) == 1.0


def test_rejects_pivot_of_one_or_less():
    with pytest.raises(ValueError):
        HybridBot(blend_pivot=1)


@pytest.mark.parametrize('hand_size,share', [(1, 0.0), (3, 0.2), (6, 0.5), (10, 0.9)])
def test_bid_model_share_follows_hand_size(monkeypatch, hand_size, share):
    obs = bid_obs(hand_size)
    bot = HybridBot(blend_pivot=6)
    width = hand_size + 1
    sim = [1.0] + [0.0] * hand_size      # simulation: no tricks
    net = [0.0, 1.0] + [0.0] * (hand_size - 1)  # model: one trick
    bot.models = SimpleNamespace(bid_distributions=lambda *args: {
        b: list(net) for b in args[-1]})
    monkeypatch.setattr(
        'ohhell.bots.simulation.SimulationBot.analyse_bids',
        lambda self, o: SimpleNamespace(distributions={0: list(sim)}))
    dist = bot.analyse_bids(obs).distributions[0]
    assert len(dist) == width
    assert dist[:2] == pytest.approx([1 - share, share])


def test_pivot_hand_size_matches_fixed_blend_exactly():
    kwargs = dict(seed=13, bid_worlds=32, fast=False, field='previous')
    obs = bid_obs(6)
    fixed = HybridBot(**kwargs).analyse_bids(obs)
    pivot = HybridBot(blend_pivot=6, **kwargs).analyse_bids(obs)
    assert pivot.distributions == fixed.distributions


def test_one_card_bids_skip_the_model():
    bot = HybridBot(blend_pivot=6, bid_worlds=8, fast=False, field='previous')
    bot.models = SimpleNamespace(bid_distributions=lambda *a: pytest.fail('model called'))
    bot.analyse_bids(bid_obs(1))


def test_play_prior_follows_cards_in_hand(monkeypatch):
    obs = play_obs(9)
    held = obs.hand.bit_count()
    bot = HybridBot(blend_pivot=6, worlds=32, field='previous', endgame_cards=0)
    cards = obs.legal_actions()
    assert len(cards) > 1
    # Simulation prefers the first card by 1 point; the model prefers the
    # second by 10. With prior 16 * (held - 1) / 5 against 32 deals, the model
    # wins once its weight * 11 exceeds 32 * 1.
    bot.models = SimpleNamespace(card_values=lambda *a: [
        10.0 if c == cards[1] else 0.0 for c in cards])
    monkeypatch.setattr(bot, 'evaluate_cards', lambda o, kept: {
        c: (1.0 if c == cards[0] else 0.0) for c in kept})
    assert bot.play(obs) == cards[1]
    assert 16 * (held - 1) / 5 * 11 > 32

    bot.blend_pivot = 100  # nearly no model weight
    assert bot.play(obs) == cards[0]


@pytest.mark.parametrize('players', [3, 8])
def test_full_game_including_blind_round(players):
    bots = [make_bot('hybrid:field=previous,blend_pivot=6,time_ms=1', s) for s in range(players)]
    totals, _, _ = play_game(bots, Rules(players, 3), players)
    assert len(totals) == players
