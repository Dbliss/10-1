"""Experimental components must stay separate, private and reproducible."""

from dataclasses import replace
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from ohhell import native, tournament
from ohhell.arena import play_game
from ohhell.bots import DEFAULT_SPEC, make_bot
from ohhell.bots.components import TimeBank, touching_groups
from ohhell.bots.learned import HybridBot
from ohhell.bots.simulation import SimulationBot
from ohhell.cards import card_from_id, mask_of
from ohhell.engine import PLAYING, RoundState
from ohhell.learn.cases import build_bank, provenance, restore
from ohhell.learn.nn import MLP
from ohhell.learn.position_bench import label_position
from ohhell.learn.rollout import (RolloutPolicy, SIZES, feature_matrix, load_network,
                                  save_network)
from ohhell.opponent_reading import OpponentTracker
from ohhell.rules import Rules
from ohhell.search import playout
from test_native import needs_core, random_round


@pytest.fixture
def compact(tmp_path):
    model = MLP(SIZES, seed=44)
    save_network(model, tmp_path / 'play.npz')
    return tmp_path


def test_default_remains_promoted_v2():
    assert DEFAULT_SPEC == 'hybrid:field=previous,time_ms=2'
    bot = make_bot(DEFAULT_SPEC)
    assert not bot.play_evidence and not bot.touching
    assert bot.rollout_policy is None and bot.opponent_tracker is None
    assert bot.policy_name == 'heuristic' and bot.play_evaluator is None


@needs_core
@pytest.mark.parametrize('field', ['previous', 'mix'])
@pytest.mark.parametrize('exact', [0, 52])
def test_dedupe_preserves_fixed_budget_values_and_rng(field, exact):
    # Small exact states are tractable. One candidate pool per turn; oversample
    # 1 deliberately repeats many draws. No stochastic field is reordered.
    rng = random.Random(91)
    savings = 0
    for seed in range(8):
        state = random_round(rng, 4, 3)
        obs = state.observation(state.to_act)
        args = dict(seed=seed, worlds=32, inference=True, oversample=1,
                    field=field, endgame_cards=exact)
        plain, reused = SimulationBot(**args), SimulationBot(dedupe=1, **args)
        assert reused.evaluate_cards(obs) == plain.evaluate_cards(obs)
        assert plain.simulator.rng.value == reused.simulator.rng.value
        assert reused.last_card_worlds == 32
        savings += 32 - reused.simulator.last_card_playouts
    assert (savings > 0) == (field == 'previous')


def test_python_dedupe_preserves_fixed_budget_values():
    state = random_round(random.Random(4), 4, 4)
    obs = state.observation(state.to_act)
    args = dict(seed=17, worlds=16, inference=True, oversample=1, field='previous', fast=False)
    assert SimulationBot(dedupe=1, **args).evaluate_cards(obs) == SimulationBot(**args).evaluate_cards(obs)


@needs_core
def test_compact_python_c_values_and_continuations_match(compact):
    model = load_network(str(compact))
    rng = random.Random(22)
    checked = 0
    for mode in ('learned_self', 'learned_all'):
        bot = SimulationBot(policy=mode, rollout_models=str(compact), endgame_cards=0)
        for i in range(30):
            n = (3, 4, 5, 6, 7, 8)[i % 6]
            state = random_round(rng, n, min(5, 51 // n))
            for _ in range(i % (n * 3)):
                state.apply(rng.choice(state.legal_actions()))
            obs = state.observation(state.to_act)
            cards = obs.legal_actions()
            expected = RolloutPolicy(model).card_values(obs, obs.seat, obs.hand, cards)
            actual = bot.simulator.rollout_values(obs, cards)
            assert list(actual.values()) == pytest.approx(expected, abs=0.00015)
            from ohhell.opponents import SKILLED
            policy = bot._policy([SKILLED] * n, obs.seat)
            scores = bot.simulator.rollout_scores(obs, state.hands, cards)
            for card in cards:
                child = state.clone()
                child.play_unchecked(card)
                playout(child, policy)
                assert scores[card] == child.points()[obs.seat]
                checked += 1
    assert checked > 100


def test_compact_bowers_and_private_hands(compact):
    ids = lambda *names: mask_of(card_from_id(n) for n in names)
    hands = [ids('Jack_of_Diamonds', 'Ace_of_Hearts', '3_of_Clubs'),
             ids('2_of_Clubs', '4_of_Clubs', '5_of_Clubs'),
             ids('2_of_Spades', '3_of_Spades', '4_of_Spades')]
    state = RoundState(Rules(3, 3), hands, card_from_id('2_of_Hearts'), 2)
    while state.phase != PLAYING:
        state.apply(state.legal_actions()[0])
    obs = state.observation(0)
    left, ace = card_from_id('Jack_of_Diamonds'), card_from_id('Ace_of_Hearts')
    x = feature_matrix(obs, 0, obs.hand, [left, ace])
    assert x[0, 0] > x[1, 0] and x[:, 1].tolist() == [1, 1]
    state.hands[1], state.hands[2] = state.hands[2], state.hands[1]
    assert np.array_equal(x, feature_matrix(state.observation(0), 0, obs.hand, [left, ace]))


@pytest.mark.parametrize('count,chosen', [(2, 1), (32, 1), (512, 0)])
def test_play_prior_counts_completed_worlds(monkeypatch, count, chosen):
    state = random_round(random.Random(7), 4, 5)
    obs = state.observation(state.to_act)
    bot = HybridBot(play_evidence=1, endgame_cards=0, shortlist=2)
    first, second = obs.legal_actions()[:2]
    bot.models = SimpleNamespace(card_values=lambda *args: [
        2 if c == second else 0 if c == first else -100 for c in args[-1]])
    def evaluate(o, cards):
        bot.last_card_worlds = count
        return {c: 0 if c == second else 0.5 for c in cards}
    monkeypatch.setattr(bot, 'evaluate_cards', evaluate)
    assert bot.play(obs) == (first, second)[chosen]


def test_touching_group_respects_table_cards_and_bowers():
    state = random_round(random.Random(9), 4, 4)
    obs = state.observation(state.to_act)
    t = obs.trump_suit
    right, left, ace = t * 13 + 9, (3 - t) * 13 + 9, t * 13 + 12
    obs = replace(obs, hand=(1 << right) | (1 << left) | (1 << ace), played=0, table=0)
    assert touching_groups(obs, [right, ace, left]) == [[right, ace, left]]
    obs = replace(obs, hand=(1 << right) | (1 << ace), table=1 << left, played=1 << left)
    assert touching_groups(obs, [right, ace]) == [[right], [ace]]


def test_bank_earns_credit_and_retains_overrun_debt():
    bank = TimeBank(0.002, capacity=4)
    assert bank.begin() == 0.002
    bank.finish(0.0005)
    assert bank.begin() == pytest.approx(0.0035)
    bank.finish(0.010)
    assert bank.credit == pytest.approx(-0.0065)
    assert bank.begin() == 1e-6
    bank.charge(0.001)
    assert bank.credit == pytest.approx(-0.0075)


@needs_core
@pytest.mark.parametrize('fast', [0, 1])
def test_experimental_components_finish_blind_down_up_game(compact, fast):
    spec = (f'hybrid:field=previous,policy=learned_self,rollout_models={compact},'
            f'policy_inference=1,typing=1,play_evidence=1,controller=bank,time_ms=1,fast={fast},'
            'worlds=2,bid_worlds=2')
    bots = [make_bot(spec, s) for s in range(3)]
    scores, _, _ = play_game(bots, Rules(3, 3), 17)
    assert len(scores) == 3
    assert all(b.opponent_tracker.rounds == 6 for b in bots)


def test_typing_uses_only_completed_public_rounds_and_excludes_blind():
    tracker = OpponentTracker('previous')
    state = random_round(random.Random(12), 4, 4)
    with pytest.raises(ValueError, match='completed'):
        tracker.observe_round(state.observation(1))
    from ohhell.bots.heuristic import state_card
    while state.phase == PLAYING:
        state.apply(state_card(state))
    prior = tracker.weights(4)
    tracker.observe_round(state.observation(1))
    after = tracker.weights(4)
    assert after[1] == pytest.approx(prior[1])
    assert any(after[s] != prior[s] for s in (0, 2, 3))
    assert after[0] != after[2]  # per seat, not pooled into one kind


@needs_core
def test_native_sampler_receives_persistent_seat_priors():
    state = random_round(random.Random(12), 4, 4)
    obs = state.observation(state.to_act)
    bot = SimulationBot(field='previous', inference=False)
    weights = [[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
    bot.simulator.set_seat_weights(weights)
    _, kinds = bot.simulator.sample_worlds(obs, 8)
    for seat in range(obs.n):
        assert set(kinds[:, seat]) == {0 if seat == obs.seat else weights[seat].index(1)}


def test_reference_does_not_consult_saved_hidden_deal():
    case = next(c for c in build_bank(31, 1, 4, players=(4,))
                if c['part'] == 'play' and not c['blind'] and 'choice' in c['tags'])
    changed = dict(case, hands=list(case['hands']))
    own = restore(case).to_act
    others = [s for s in range(4) if s != own]
    a, b = others[:2]
    changed['hands'][a], changed['hands'][b] = changed['hands'][b], changed['hands'][a]
    left = label_position(case, 'hybrid:field=previous,worlds=8', repeats=1)
    right = label_position(changed, 'hybrid:field=previous,worlds=8', repeats=1)
    assert left['outcomes'] == right['outcomes']


def test_compact_fingerprint_prevents_changed_weights_resuming(tmp_path, compact):
    args = SimpleNamespace(seed=1, max_sets=130, smallest_gain=1, time_ms=2, resume=False)
    spec = f'hybrid:policy=learned_self,rollout_models={compact}'
    path = tmp_path / 'run.meta'
    tournament.check_run_config(path, args, [spec])
    assert str(compact / 'play.npz') in provenance(spec)['models']
    assert 'source_sha256' in json.loads(path.read_text())
    save_network(MLP(SIZES, seed=3), compact / 'play.npz')
    args.resume = True
    with pytest.raises(SystemExit, match='different rules, models or settings'):
        tournament.check_run_config(path, args, [spec])


def test_mean_time_gate_withholds_promotion_and_counts_confirmation():
    from test_tournament import board
    b = board()
    b.args = SimpleNamespace(**{k: getattr(b.args, k) for k in (
        'max_sets', 'smallest_gain', 'label', 'seed', 'time_ms', 'reference')}, mean_time_ratio=1.05)
    b.repeat_target = 100
    for i in range(100):
        b.add_repeat([0, 1, 2], [[10 + i % 3, 0, 0]], [[0.003, 1, 0.003], [0.002, 1, 0.002], [0, 0, 0]])
    b.finish()
    assert b.verdict == 'over time'
    assert b.clocks[0][1] == 100


def test_missing_native_requires_explicit_python_oracle(monkeypatch):
    monkeypatch.setattr(native, 'available', False)
    with pytest.raises(RuntimeError, match='fast=0'):
        SimulationBot()
    assert SimulationBot(fast=False).simulator is None


@needs_core
def test_early_stop_finishes_clear_positions_without_spending_full_deadline():
    rng = random.Random(99)
    found = False
    for _ in range(20):
        state = random_round(rng, 4, 6)
        obs = state.observation(state.to_act)
        bot = SimulationBot(field='previous', inference=False, worlds=32,
                            endgame_cards=0, stop_z=3, stop_min=32)
        # A generous ceiling makes this a stopping-mechanics test, not a speed
        # benchmark. Some positions stay ambiguous; at least one is clear.
        import time
        started = time.perf_counter()
        bot.simulator.evaluate_cards(obs, obs.legal_actions(), 32, time_left=0.1)
        elapsed = time.perf_counter() - started
        if elapsed < 0.05 and bot.simulator.last_card_worlds >= 32:
            found = True
            break
    assert found


def test_learned_play_likelihood_uses_normalized_policy_probabilities(compact):
    from ohhell.belief import Belief
    from ohhell.bots.heuristic import state_card
    from ohhell.engine import legal_card_mask
    from ohhell.opponents import PREVIOUS
    state = random_round(random.Random(5), 4, 4)
    opening = list(state.hands)
    seat = state.to_act
    cards = state.legal_actions()
    assert len(cards) > 1
    chosen = cards[-1]
    policy = RolloutPolicy(load_network(str(compact)))
    values = policy.card_values(state, seat, opening[seat], cards)
    weights = np.exp((np.asarray(values) - max(values)) / 10)
    expected = np.log(PREVIOUS.play_doubt / len(cards) + (1 - PREVIOUS.play_doubt)
                      * weights[-1] / weights.sum())
    state.apply(chosen)
    observer = state.to_act
    obs = state.observation(observer)
    belief = Belief()
    belief.card_policy = policy
    assert belief._play_log_weight(obs, opening, [PREVIOUS] * 4) == pytest.approx(expected)


@needs_core
def test_learned_native_inference_matches_python_on_public_replay(compact):
    from ohhell.opponents import KINDS
    rng = random.Random(123)
    bot = SimulationBot(policy_inference=1, rollout_models=str(compact))
    for i in range(20):
        state = random_round(rng, 4, 5)
        for _ in range(i % 14):
            state.apply(rng.choice(state.legal_actions()))
        obs = state.observation(state.to_act)
        indices = [0, 1, 2, 3]
        actual = bot.simulator.world_log_weight(obs, state.hands, indices)
        expected = bot.belief.log_weight(obs, state.hands, [KINDS[k] for k in indices])
        assert actual == pytest.approx(expected, abs=0.0001)


@needs_core
def test_compact_and_native_outcomes_use_observed_scoring(compact):
    from ohhell.opponents import SKILLED
    from ohhell.rules import Scoring
    state = random_round(random.Random(55), 4, 4)
    state.rules = Rules(4, 4, scoring=Scoring(17, 23, 7))
    obs = state.observation(state.to_act)
    cards = obs.legal_actions()
    bot = SimulationBot(policy='learned_all', rollout_models=str(compact))
    expected = bot.rollout_policy.card_values(obs, obs.seat, obs.hand, cards)
    assert list(bot.simulator.rollout_values(obs, cards).values()) == pytest.approx(expected, abs=0.0001)
    actual = bot.simulator.rollout_scores(obs, state.hands, cards)
    for card in cards:
        child = state.clone()
        child.play_unchecked(card)
        playout(child, bot._policy([SKILLED] * 4, obs.seat))
        assert actual[card] == child.points()[obs.seat]


@pytest.mark.parametrize('stage', ['dedupe', 'touching', 'play-evidence', 'early-stop',
                                    'bank', 'policy-self', 'policy-all', 'typing', 'policy-inference'])
def test_launcher_changes_one_setting_per_comparison(stage):
    from ohhell.learn.search_experiment import stage_options
    candidate, baseline, extra = stage_options(stage)
    def options(spec):
        return dict(pair.split('=') for pair in spec.split(':')[1].split(','))
    c, b = options(candidate), options(baseline)
    assert sum(c.get(k) != b.get(k) for k in c.keys() | b.keys()) == 1


@needs_core
def test_existing_search_label_selfplay_accepts_learned_continuation(compact):
    from ohhell.learn.models import MODEL_DIR
    from ohhell.learn.selfplay import _generate_search
    spec = (f'hybrid:field=previous,policy=learned_self,rollout_models={compact},'
            'models={folder},worlds=2,bid_worlds=2')
    arrays = _generate_search((17, 3, str(MODEL_DIR), 1.0, 0.0, 2, 2, 1.0, 1.0, spec))
    assert len(arrays[0]) > 0 and len(arrays[3]) > 0
    assert all(np.isfinite(a).all() for a in arrays)
