"""Incremental NNUE equivalence, privacy, training splits and bot isolation."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from ohhell.arena import play_game
from ohhell.bots import DEFAULT_SPEC, make_bot
from ohhell.learn.cases import build_bank, provenance, restore
from ohhell.learn.models import VALUE_SCALE, load_models
from ohhell.learn.nn import MLP
from ohhell.learn.nnue import (
    INPUTS, Evaluator, canonical_card, feature_matrix, load_network,
    position_features, save_network,
)
from ohhell.learn.train_nnue import dataset, fit_separated, metrics
from ohhell.rules import Rules
from ohhell.tournament import check_run_config


@pytest.fixture(scope='module')
def cases():
    return [c for c in build_bank(17, 3, 3, players=(2, 4, 8))
            if c['part'] == 'play' and not c['blind']]


def test_incremental_matches_dense_through_positions_rounds_and_rebuilds(cases):
    network = MLP([INPUTS, 32, 16, 1], seed=9)
    evaluator = Evaluator(network)
    # Includes trick completions, changing targets, and jumps between rounds.
    for case in cases[::3]:
        state = restore(case)
        obs = state.observation(state.to_act)
        cards = obs.legal_actions()
        expected = network(feature_matrix(obs, obs.seat, obs.hand, cards))[:, 0] * VALUE_SCALE
        actual = evaluator.card_values(obs, obs.seat, obs.hand, cards)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=5e-5)
        # Candidate branches must not mutate the cached base position.
        again = evaluator.card_values(obs, obs.seat, obs.hand, list(reversed(cards)))
        np.testing.assert_allclose(again, list(reversed(actual)), rtol=2e-5, atol=5e-5)
    assert evaluator.rebuilds > 1


def test_feature_layout_is_sparse_stable_and_uses_no_hidden_hand(cases):
    case = next(c for c in cases if 'hand=3' in c['tags'])
    state = restore(case)
    obs = state.observation(state.to_act)
    features = position_features(obs, obs.seat, obs.hand)
    assert np.count_nonzero(features) < 100
    card = obs.legal_actions()[0]
    smaller = obs.hand ^ (1 << card)
    after = position_features(obs, obs.seat, smaller)
    assert np.flatnonzero(features != after).tolist() == [canonical_card(card, obs.trump_suit)]
    # Same public observation, different actual opponent hands: same features.
    for seat in range(state.n):
        if seat != state.to_act:
            state.hands[seat] = 0
    other = state.observation(state.to_act)
    np.testing.assert_array_equal(features, position_features(other, other.seat, other.hand))
    with pytest.raises(ValueError, match='visible'):
        position_features(replace(obs, blind=True, hand=0), obs.seat, 0)


def test_checkpoint_version_roundtrip_and_private_accumulators(tmp_path, cases):
    model = MLP([INPUTS, 16, 8, 1], seed=5)
    save_network(model, tmp_path / 'play.npz')
    loaded = load_network(str(tmp_path))
    assert not loaded.weights[0].flags.writeable
    left = make_bot(f'nnue:play_models={tmp_path},fast=0,worlds=1,bid_worlds=1')
    right = make_bot(f'nnue:play_models={tmp_path},fast=0,worlds=1,bid_worlds=1')
    assert left.models.evaluator is not right.models.evaluator
    assert left.models.evaluator.network is right.models.evaluator.network
    state = restore(cases[0])
    obs = state.observation(state.to_act)
    left.models.card_values(obs, obs.seat, obs.hand, obs.legal_actions())
    assert right.models.evaluator.accumulator is None
    assert DEFAULT_SPEC == 'hybrid:field=previous,time_ms=2'


def test_bidding_and_endgame_stay_v1(tmp_path, cases, monkeypatch):
    save_network(MLP([INPUTS, 16, 8, 1]), tmp_path / 'play.npz')
    spec = f'nnue:play_models={tmp_path},field=previous,fast=0,worlds=2,bid_worlds=2'
    candidate = make_bot(spec, 19)
    baseline = make_bot('hybrid:field=previous,fast=0,worlds=2,bid_worlds=2', 19)
    bidding = next(c for c in build_bank(17, 1, 2, players=(4,))
                   if c['part'] == 'bid' and not c['blind'])
    state = restore(bidding)
    obs = state.observation(state.to_act)
    assert candidate.analyse_bids(obs).distributions == baseline.analyse_bids(obs).distributions
    state = restore(next(c for c in cases if 'hand=2' in c['tags'] and 'players=4' in c['tags']))
    obs = state.observation(state.to_act)
    monkeypatch.setattr(candidate.models, 'card_values', lambda *args: pytest.fail('endgame used NNUE'))
    assert candidate.act(obs) in obs.legal_actions()


@pytest.mark.parametrize('fast', [0, 1])
def test_nnue_finishes_complete_blind_inclusive_game(tmp_path, fast):
    save_network(MLP([INPUTS, 16, 8, 1]), tmp_path / 'play.npz')
    spec = f'nnue:play_models={tmp_path},field=previous,fast={fast},worlds=1,bid_worlds=1'
    totals, _, _ = play_game([make_bot(spec, s) for s in range(4)], Rules(4, 3), 91)
    assert len(totals) == 4


def test_fingerprints_cover_selected_nnue_and_bid_weights(tmp_path):
    folder = tmp_path / 'weights'
    folder.mkdir()
    save_network(MLP([INPUTS, 16, 8, 1]), folder / 'play.npz')
    spec = f'nnue:play_models={folder}'
    proof = provenance(spec)
    assert str(folder / 'play.npz') in proof['models']
    args = SimpleNamespace(seed=1, max_sets=130, smallest_gain=1, time_ms=2, resume=False)
    path = tmp_path / 'run.meta'
    check_run_config(path, args, [spec])
    save_network(MLP([INPUTS, 16, 8, 1], seed=2), folder / 'play.npz')
    args.resume = True
    with pytest.raises(SystemExit, match='different rules, models or settings'):
        check_run_config(path, args, [spec])


def test_training_selects_with_external_validation_and_updates_partial_batch(cases):
    groups = dataset(iter(cases), teacher=load_models())
    assert not groups['train']['deals'] & groups['validation']['deals']
    assert not groups['train']['deals'] & groups['test']['deals']
    tiny = {'x': groups['train']['x'][:7], 'y': groups['train']['y'][:7]}
    model = MLP([INPUTS, 16, 8, 1], seed=0)
    before = [p.copy() for p in model.parameters()]
    history = fit_separated(model, tiny, groups['validation'], epochs=2,
                            batch_size=1024, seed=0, log=lambda _: None)
    assert len(history) == 2
    assert any(not np.array_equal(a, b) for a, b in zip(before, model.parameters()))
    assert np.isfinite(metrics(model, groups['test'])['rmse_points'])


def test_training_rejects_deal_leakage(cases):
    chosen = next(c for c in cases if len(restore(c).legal_actions()) > 1)
    corrupted = [dict(chosen, split=split) for split in ('train', 'validation', 'test')]
    with pytest.raises(ValueError, match='multiple splits'):
        dataset(corrupted, teacher=load_models())
