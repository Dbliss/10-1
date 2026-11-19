"""Noise-aware exact endgame search (endgame_noise) and per-table thresholds (endgame_table)."""

import random

import pytest

from ohhell import native
from ohhell.arena import deal_round
from ohhell.bots.heuristic import HeuristicBot
from ohhell.bots.simulation import ENDGAME_BY_PLAYERS, SimulationBot
from ohhell.cards import cards_in
from ohhell.engine import BIDDING, PLAYING, RoundState, legal_card_mask
from ohhell.rules import Rules
from ohhell.search import distinct_moves, distinct_moves_counted, solve, solve_noisy, solve_root

needs_core = pytest.mark.skipif(not native.available, reason='native core not built')


def brute_noisy(state, root, noise):
    """Reference: every legal card, no merging, no memo."""
    if state.phase != PLAYING:
        return tuple(float(x) for x in state.points())
    seat = state.to_act
    legal = cards_in(legal_card_mask(state.hands[seat], state.lead_suit, state.trump_suit))
    best, values = None, []
    for card in legal:
        child = state.clone()
        child.play_unchecked(card)
        value = brute_noisy(child, root, noise)
        values.append(value)
        if (best is None or value[seat] > best[seat]
                or (value[seat] == best[seat] and value[root] < best[root])):
            best = value
    eps = noise[seat]
    if not eps:
        return best
    return tuple((1 - eps) * b + eps * sum(v[i] for v in values) / len(values)
                 for i, b in enumerate(best))


def position(seed, players, hand_size, played=0):
    """A round after bidding (random bids) and ``played`` heuristic cards."""
    rng = random.Random(seed)
    hands, trump = deal_round(rng, players, hand_size)
    state = RoundState(Rules(players, hand_size), hands, trump, rng.randrange(players))
    while state.phase == BIDDING:
        state.apply(rng.choice(state.legal_actions()))
    bot = HeuristicBot(seed)
    for _ in range(played):
        state.apply(bot.act(state.observation(state.to_act)))
    return state, rng


@pytest.mark.parametrize('seed', range(30))
def test_noisy_search_matches_brute_force(seed):
    state, rng = position(seed, random.Random(seed).choice([3, 4]), 3)
    noise = [rng.choice([0.0, 0.35, 1.0]) for _ in state.hands]
    root = state.to_act
    expected = brute_noisy(state, root, noise)
    actual = solve_noisy(state, root, noise, {})
    assert actual == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize('seed', range(20))
def test_no_noise_equals_plain_search(seed):
    state, _ = position(seed, 4, 4, played=2)
    root = state.to_act
    noise = [0.0] * len(state.hands)
    assert solve_noisy(state, root, noise, {}) == pytest.approx(solve(state, root, {}))


@pytest.mark.parametrize('seed', range(20))
def test_merged_moves_count_every_legal_card(seed):
    state, _ = position(seed, 3, 8, played=seed % 7)
    kept, sizes = distinct_moves_counted(state)
    legal = legal_card_mask(state.hands[state.to_act], state.lead_suit, state.trump_suit)
    assert kept == distinct_moves(state)
    assert sum(sizes) == legal.bit_count() and min(sizes) >= 1


def test_noise_changes_the_value_of_a_risky_line():
    """Some deal where a noisy opponent makes a perfect-play line worse for us."""
    for seed in range(200):
        state, _ = position(seed, 3, 3)
        root = state.to_act
        cards = state.legal_actions()
        plain = solve_root(state, cards, root)
        noisy = solve_root(state, cards, root, [0.35] * 3)
        if any(noisy[c] < plain[c] - 1e-9 for c in cards):
            return
    pytest.fail('noise never lowered a value')


@needs_core
@pytest.mark.parametrize('seed', range(40))
def test_native_noisy_search_matches_python(seed):
    rng = random.Random(seed)
    players = rng.choice([3, 4, 5])
    state, rng = position(seed, players, rng.choice([3, 4]), played=rng.randrange(3))
    obs = state.observation(state.to_act)
    cards = obs.legal_actions()
    noise = [rng.choice([0.0, 0.35, 1.0]) for _ in range(players)]
    bot = SimulationBot(seed=seed)
    native_values = bot.simulator.solve_noisy_scores(obs, state.hands, noise, cards)
    python_values = solve_root(state, cards, obs.seat, noise)
    for card in cards:
        assert native_values[card] == pytest.approx(python_values[card], abs=1e-9)


def test_endgame_table_sets_threshold_per_table_size():
    bot = SimulationBot(endgame_table=1, fast=False)
    assert [bot.endgame_limit(n) for n in range(3, 9)] == [ENDGAME_BY_PLAYERS[n] for n in range(3, 9)]
    assert SimulationBot(endgame_cards=12, fast=False).endgame_limit(4) == 12


@needs_core
def test_native_simulator_uses_the_table_threshold():
    bot = SimulationBot(endgame_table=1, endgame_budget=7, card_budget=3)
    state, _ = position(1, 4, 4)  # 16 cards left: exact at 4 players only with the table (19)
    obs = state.observation(state.to_act)
    assert bot.simulator._card_budget(obs) == 7
    plain = SimulationBot(endgame_budget=7, card_budget=3)
    assert plain.simulator._card_budget(obs) == 3


@needs_core
@pytest.mark.parametrize('noise_on', [0, 1])
def test_bot_plays_a_mixed_field_endgame(noise_on):
    state, _ = position(3, 4, 4)
    obs = state.observation(state.to_act)
    bot = SimulationBot(seed=1, field='mix', endgame_noise=noise_on, endgame_cards=16,
                        endgame_budget=50, card_budget=1)
    values = bot.evaluate_cards(obs)
    assert set(values) == set(obs.legal_actions())
    assert bot.last_card_worlds >= 50
