"""Adaptive search must explore omitted cards and respect its total budget."""

import random
import time
from types import SimpleNamespace

import pytest

from ohhell import native
from ohhell.arena import deal_round, play_game
from ohhell.bots import make_bot
from ohhell.bots.adaptive import AdaptiveBot, CardRace
from ohhell.bots.learned import HybridBot
from ohhell.engine import BIDDING, RoundState
from ohhell.rules import Rules


def playing_obs(players=4, size=6):
    hands, trump = deal_round(random.Random(7), players, size)
    state = RoundState(Rules(players, size), hands, trump, players - 1)
    while state.phase == BIDDING:
        state.apply(state.legal_actions()[0])
    return state.observation(state.to_act)


def test_simulations_can_overturn_a_wrong_model_prior():
    race = CardRace({1: 30.0, 2: 0.0}, 16)
    race.add({1: 0.0, 2: 20.0}, 4)
    assert race.best() == 1  # the prior still dominates with little evidence
    race.add({1: 0.0, 2: 20.0}, 100)
    assert race.best() == 2
    assert race.value(2) == pytest.approx(2080 / 120)


def test_zero_prior_can_choose_before_any_samples():
    assert CardRace({1: -2.0, 2: 5.0}, 0).best() == 2


def test_race_preserves_totals_and_concentrates_samples_on_survivors():
    race = CardRace(dict.fromkeys(range(6), 0.0), 0)
    race.add({card: card * 10 for card in race.active}, 8)
    race.halve()
    assert race.active == [5, 4, 3]
    race.add({card: card * 10 for card in race.active}, 16)
    race.halve()
    assert race.active == [5, 4]
    assert race.samples == {0: 8, 1: 8, 2: 8, 3: 24, 4: 24, 5: 24}


def test_search_can_select_a_card_outside_the_old_top_four(monkeypatch):
    obs = playing_obs(size=8)
    cards = obs.legal_actions()
    bot = AdaptiveBot(worlds=32, prior_worlds=1)
    bot.models = SimpleNamespace(card_values=lambda *_: list(reversed(range(len(cards)))))
    evaluated = []
    target = cards[-1]

    def batch(_obs, active, count):
        evaluated.append((list(active), count))
        return {card: 100.0 if card == target else -100.0 for card in active}

    monkeypatch.setattr(bot, '_batch', batch)
    assert bot.play(obs) == target
    assert evaluated[0][0] == cards  # every legal action gets a chance
    samples = bot.last_search['samples']
    assert all(count > 0 for count in samples.values())
    assert samples[target] > min(samples.values())
    assert sum(samples.values()) <= bot.worlds * len(cards)


def test_expired_turn_does_not_start_more_simulations(monkeypatch):
    obs = playing_obs()
    bot = AdaptiveBot()
    bot.deadline = time.perf_counter() - 1

    def unexpected_batch(*_):
        pytest.fail('Sampling continued after the deadline')

    monkeypatch.setattr(bot, '_batch', unexpected_batch)
    assert bot.play(obs) in obs.legal_actions()
    assert not any(bot.last_search['samples'].values())


def test_stages_share_one_turn_deadline(monkeypatch):
    import ohhell.bots.adaptive as module

    obs = playing_obs()
    bot = AdaptiveBot()
    clock = [0.0]
    monkeypatch.setattr(module, 'time', SimpleNamespace(perf_counter=lambda: clock[0]))
    bot.deadline = 0.002
    calls = []

    def batch(_obs, cards, count):
        calls.append(clock[0])
        clock[0] += 0.00025
        return dict.fromkeys(cards, 0.0)

    monkeypatch.setattr(bot, '_batch', batch)
    assert bot.play(obs) in obs.legal_actions()
    assert all(start < bot.deadline for start in calls)
    assert clock[0] <= bot.deadline + 0.00025


@pytest.mark.parametrize('fast', [False, True])
def test_adaptive_bot_finishes_full_games_with_both_backends(fast):
    if fast and not native.available:
        pytest.skip('core not built')
    spec = f'adaptive:field=previous,worlds=8,bid_worlds=3,fast={int(fast)},endgame_cards=0'
    bots = [make_bot(spec, seat) for seat in range(4)]
    totals, _, _ = play_game(bots, Rules(4, 3), 19)
    assert len(totals) == 4


def test_bidding_and_exact_endgames_match_current_bot():
    hands, trump = deal_round(random.Random(5), 4, 2)
    state = RoundState(Rules(4, 2), hands, trump, 3)
    kwargs = dict(seed=8, field='previous', worlds=8, bid_worlds=8)
    current, candidate = HybridBot(**kwargs), AdaptiveBot(**kwargs)
    assert current.bid(state.observation(0)) == candidate.bid(state.observation(0))
    while state.phase == BIDDING:
        state.apply(state.legal_actions()[0])
    obs = state.observation(state.to_act)
    assert current.play(obs) == candidate.play(obs)
    assert candidate.last_search is None


@pytest.mark.parametrize('batch', [0, -1, 1025, 2.5])
def test_invalid_batch_size_is_rejected(batch):
    with pytest.raises(ValueError, match='batch_worlds'):
        AdaptiveBot(batch_worlds=batch)
