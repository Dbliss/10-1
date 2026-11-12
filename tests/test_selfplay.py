"""Self-play data generation produces well-formed, legal training rows."""

import random

import numpy as np

from ohhell.arena import deal_round
from ohhell.engine import BIDDING, PLAYING, RoundState
from ohhell.learn.features import BID_INPUTS, MAX_TRICKS, PLAY_INPUTS
from ohhell.learn.models import MODEL_DIR, VALUE_SCALE, load_models
from ohhell.learn.selfplay import SEARCH_SPEC, SelfPlayPolicy, _generate
from ohhell.opponents import SKILLED
from ohhell.rules import Rules


def test_generated_rows_have_matching_shapes_and_sane_labels():
    bid_x, bid_k, bid_limit, play_x, play_y = _generate((5, 40, str(MODEL_DIR), 1.0, 0.05, 0.15, 0.3))
    assert bid_x.shape[1] == BID_INPUTS and play_x.shape[1] == PLAY_INPUTS
    assert len(bid_x) == len(bid_k) == len(bid_limit) > 0
    assert len(play_x) == len(play_y) > 0
    assert (bid_k >= 0).all() and (bid_k <= bid_limit).all()
    assert (bid_limit <= MAX_TRICKS).all()
    assert np.abs(play_y * VALUE_SCALE).max() <= 100


def test_model_policy_only_plays_legal_cards():
    rng = random.Random(3)
    models = load_models()
    for _ in range(20):
        n = rng.choice([2, 4, 7])
        hands, trump = deal_round(rng, n, 5)
        state = RoundState(Rules(n, 5), hands, trump, 0)
        while state.phase == BIDDING:
            state.apply(rng.choice(state.legal_actions()))
        policy = SelfPlayPolicy(models, [SKILLED] * n, [True, False] * n, rng, 0.1)
        while state.phase == PLAYING:
            state.apply(policy(state))


def test_search_labels_are_distributions_and_points():
    from ohhell import native
    import pytest
    if not native.available:
        pytest.skip('native core not built')
    from ohhell.learn.selfplay import _generate_search
    bid_x, bid_p, bid_limit, play_x, play_y = _generate_search(
        (7, 30, str(MODEL_DIR), 1.0, 0.05, 32, 16, 0.15, 0.3, SEARCH_SPEC))
    assert bid_x.shape[1] == BID_INPUTS and play_x.shape[1] == PLAY_INPUTS
    assert len(bid_x) == len(bid_p) == len(bid_limit) > 0
    assert bid_p.shape[1] == MAX_TRICKS + 1
    assert np.allclose(bid_p.sum(axis=1), 1.0, atol=1e-5)
    above = np.arange(MAX_TRICKS + 1) > bid_limit[:, None]
    assert (bid_p[above] == 0).all()
    assert len(play_x) == len(play_y) > 0
    assert np.abs(play_y * VALUE_SCALE).max() <= 100


def test_soft_loss_matches_hard_loss_on_one_hot_targets():
    from ohhell.learn.nn import soft_softmax_loss, softmax_loss
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((6, MAX_TRICKS + 1)).astype(np.float32)
    limits = np.array([3, 5, 10, 1, 7, 2])
    hard = np.array([2, 0, 9, 1, 7, 2])
    soft = np.eye(MAX_TRICKS + 1, dtype=np.float32)[hard]
    batch = np.arange(6)
    a, ga = softmax_loss(limits)(hard, batch)(logits.copy())
    b, gb = soft_softmax_loss(limits)(soft, batch)(logits.copy())
    assert abs(a - b) < 1e-5 and np.allclose(ga, gb, atol=1e-5)


def test_search_labels_follow_the_given_spec_and_frozen_bids():
    from ohhell import native
    import pytest
    if not native.available:
        pytest.skip('native core not built')
    from ohhell.learn.selfplay import _generate_search
    spec = 'hybrid:field=previous,estimate=2,bid_blend=0,time_ms=1,models={folder}'
    bid_x, _, _, play_x, play_y = _generate_search(
        (8, 20, str(MODEL_DIR), 1.0, 0.0, 16, 16, 0.3, 0.0, spec))
    assert len(bid_x) == 0  # bid labelling off
    assert len(play_x) == len(play_y) > 0


def test_widened_play_network_predicts_what_it_did_before():
    import random
    from ohhell.arena import deal_round
    from ohhell.engine import RoundState
    from ohhell.learn.features import PLAY_BOWER_INPUTS, play_matrix
    from ohhell.learn.nn import MLP
    from ohhell.learn.selfplay import widen_play
    from ohhell.rules import Rules
    play = MLP.load(MODEL_DIR / 'play.npz')
    wide = widen_play(MLP.load(MODEL_DIR / 'play.npz'), PLAY_BOWER_INPUTS)
    rng = random.Random(4)
    hands, trump = deal_round(rng, 4, 6)
    state = RoundState(Rules(4, 6), hands, trump, 0)
    while state.phase != 2:
        state.apply(state.legal_actions()[0])
    obs = state.observation(state.to_act)
    cards = obs.legal_actions()
    narrow = play(play_matrix(obs, obs.seat, obs.hand, cards))
    widened = wide(play_matrix(obs, obs.seat, obs.hand, cards, PLAY_BOWER_INPUTS))
    assert np.allclose(narrow, widened, atol=1e-5)
