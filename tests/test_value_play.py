import random

import numpy as np

from ohhell.arena import deal_round
from ohhell.engine import NO_BID, PLAYING, RoundState
from ohhell.learn import bid_policy as bp
from ohhell.learn import value_play as vp
from ohhell.rules import Rules


def _midround(seed=5, n=4, hand_size=6, tricks=2):
    rng = random.Random(seed)
    hands, trump = deal_round(rng, n, hand_size)
    state = RoundState(Rules(n, min(10, 51 // n)), hands, trump, 0)
    while state.phase != PLAYING:
        state.apply(state.legal_actions()[0])
    while state.tricks_won and sum(state.tricks_won) < tricks:
        state.apply(state.legal_actions()[-1])
    return state


def test_play_features_mark_every_card_once():
    st = _midround()
    assert st.trick_size == 0
    for hero in range(st.n):
        x = vp.features(st.hands, st.played, st.trump_card, st.hand_size, st.bids,
                        st.tricks_won, st.leader, hero, 0)
        assert x.shape == (vp.INPUTS,)
        cards = x[:52 * vp.OWNERS].reshape(52, vp.OWNERS)
        assert np.all(cards.sum(axis=1) == 1)
        assert cards[:, 0].sum() == st.hands[hero].bit_count()
        assert cards[:, vp.PLAYED].sum() == 2 * st.n
        assert cards[:, vp.UPCARD].sum() == 1


def test_native_play_value_matches_python():
    import pytest
    from ohhell import native
    if not native.available or not hasattr(native.Simulator, 'play_value_points'):
        pytest.skip('native card value not built')
    from ohhell.rules import Scoring
    model = vp.MLP([vp.INPUTS, 32, 16, 11], seed=2)
    sim = native.Simulator(1, False, 4, 'previous', 12, Scoring())
    sim.set_play_value(model)
    score = Scoring().points
    for seed in range(12):
        n = 3 + seed % 4
        st = _midround(seed=seed, n=n, hand_size=min(3 + seed % 5, 51 // n), tricks=1 + seed % 2)
        if st.phase != PLAYING:
            continue
        hero = st.to_act
        x = vp.features(st.hands, st.played, st.trump_card, st.hand_size, st.bids,
                        st.tricks_won, st.leader, hero, 0)
        probs = vp.predict(model, x[None], np.array([st.hand_size]))[0]
        want = sum(p * score(st.bids[hero], k) for k, p in enumerate(probs[:st.hand_size + 1]))
        got = sim.play_value_points(st.observation(hero), st.hands)
        assert abs(got - want) < 1e-3


def test_bid_policy_features_only_use_earlier_bids():
    rng = random.Random(2)
    hands, trump = deal_round(rng, 5, 4)
    seat, dealer = 3, 0
    a = bp.features(hands[seat], trump, 5, 4, seat, dealer, [NO_BID, 1, 0, NO_BID, NO_BID])
    b = bp.features(hands[seat], trump, 5, 4, seat, dealer, [NO_BID, 1, 0, NO_BID, NO_BID])
    assert a.shape == (bp.INPUTS,) and np.array_equal(a, b)
    assert a[90 + 2] == 1.0  # third bidder after the dealer
