"""Bid-network training options: bower inputs, resampled labels, strong
opponents and wider hidden layers."""

import random

import numpy as np
import pytest

from ohhell.arena import deal_round
from ohhell.cards import card_from_id, mask_of
from ohhell.learn.features import (BID_BOWER_INPUTS, BID_INPUTS, MAX_TRICKS, bid_bower_part,
                                   bid_matrix)
from ohhell.learn.models import MODEL_DIR, Models
from ohhell.learn.nn import MLP
from ohhell.learn.selfplay import _generate, widen_bid, widen_hidden

V2 = MODEL_DIR


def ids(*names):
    return mask_of(card_from_id(name) for name in names)


def _v2():
    return Models(MLP.load(V2 / 'bid.npz'), MLP.load(V2 / 'play.npz'))


def _spots(count, seed):
    rng = random.Random(seed)
    for _ in range(count):
        n = rng.randint(2, 8)
        size = rng.randint(1, min(MAX_TRICKS, 51 // n))
        hands, trump = deal_round(rng, n, size)
        dealer = rng.randrange(n)
        seat = rng.randrange(n)
        position = (seat - dealer - 1) % n
        bids = [-1] * n
        for j in range(position):
            bids[(dealer + 1 + j) % n] = rng.randint(0, size)
        yield rng, hands[seat], trump, n, size, seat, dealer, bids


def test_bower_part_marks_the_partner_slot_and_the_left_bower():
    trump = card_from_id('2_of_Hearts')  # left bower: Jack of Diamonds
    with_left = ids('Jack_of_Diamonds', '3_of_Clubs', '4_of_Clubs')
    # Order: Hearts (trump, the left bower counts), Clubs (two), then Diamonds
    # and Spades (none) by suit number, so Diamonds is slot 2.
    assert bid_bower_part(with_left, trump).tolist() == [0.0, 1.0, 0.0, 1.0]
    without = ids('3_of_Diamonds', '4_of_Diamonds', '5_of_Diamonds', '3_of_Clubs')
    # Diamonds is now the longest side suit, slot 1, and no left bower.
    assert bid_bower_part(without, trump).tolist() == [1.0, 0.0, 0.0, 0.0]
    side_jack = ids('Jack_of_Clubs', '3_of_Clubs')
    assert bid_bower_part(side_jack, trump)[3] == 0.0


def test_bower_layout_keeps_the_first_98_inputs():
    for _, hand, trump, n, size, seat, dealer, bids in _spots(50, 1):
        options = list(range(size + 1))
        narrow = bid_matrix(hand, trump, n, size, seat, dealer, bids, options)
        wide = bid_matrix(hand, trump, n, size, seat, dealer, bids, options, BID_BOWER_INPUTS)
        assert wide.shape == (len(options), BID_BOWER_INPUTS)
        assert np.array_equal(wide[:, :BID_INPUTS], narrow)
        assert np.array_equal(wide[:, BID_INPUTS:],
                              np.tile(bid_bower_part(hand, trump), (len(options), 1)))


def test_widened_networks_predict_exactly_as_before():
    v2 = _v2()
    bower = Models(widen_bid(MLP.load(V2 / 'bid.npz'), BID_BOWER_INPUTS), v2.play)
    bigger = Models(widen_hidden(MLP.load(V2 / 'bid.npz'), [384, 192], seed=3), v2.play)
    assert bower.bid_inputs == BID_BOWER_INPUTS
    assert [w.shape[1] for w in bigger.bid.weights] == [384, 192, MAX_TRICKS + 1]
    for _, hand, trump, n, size, seat, dealer, bids in _spots(100, 2):
        options = list(range(size + 1))
        before = v2.bid_distributions(hand, trump, n, size, seat, dealer, bids, options)
        for models in (bower, bigger):
            after = models.bid_distributions(hand, trump, n, size, seat, dealer, bids, options)
            for bid in options:
                assert after[bid] == pytest.approx(before[bid], abs=1e-5)


def test_widen_hidden_refuses_to_shrink():
    with pytest.raises(ValueError):
        widen_hidden(MLP.load(V2 / 'bid.npz'), [100, 50])


def test_batched_choice_matches_one_at_a_time_in_the_bower_layout():
    rng0 = np.random.default_rng(0)
    bid = MLP.load(V2 / 'bid.npz')
    widen_bid(bid, BID_BOWER_INPUTS)
    bid.weights[0][BID_INPUTS:] = rng0.standard_normal((4, bid.weights[0].shape[1])) * 0.5
    models = Models(bid, _v2().play)
    for rng, hand, trump, n, size, seat, dealer, bids in _spots(150, 4):
        options = list(range(size + 1))
        score = np.array([[(10 + b) if k == b else -abs(k - b) for k in range(size + 1)]
                          for b in range(MAX_TRICKS + 1)], dtype=float)[:, :size + 1]
        score = np.pad(score, ((0, 0), (0, 0)))
        dists = models.bid_distributions(hand, trump, n, size, seat, dealer, bids, options)
        slow = max(options, key=lambda b: float(np.dot(dists[b], score[b])))
        fast = models.choose_bids([hand], [bids], trump, n, size, seat, dealer,
                                  [options], [[True] * len(options)], score)
        assert fast[0] == slow


def test_resampled_labels_are_trick_distributions():
    bid_x, bid_p, bid_limit, play_x, play_y = _generate(
        (7, 20, str(V2), 1.0, 0.05, 0.0, 1.0, {'label_worlds': 3}))
    assert len(bid_x) == len(bid_p) == len(bid_limit) > 0
    assert bid_p.shape[1] == MAX_TRICKS + 1
    assert np.allclose(bid_p.sum(axis=1), 1.0)
    for row, limit in zip(bid_p, bid_limit):
        assert not row[limit + 1:].any()
        # Three deals, so every probability is a multiple of a third.
        assert np.allclose(row * 3, np.round(row * 3), atol=1e-5)
    assert len(play_x) == 0


def test_strong_opponents_play_full_rounds():
    spec = 'hybrid:field=previous,worlds=4,bid_worlds=8'
    bid_x, bid_k, bid_limit, _, _ = _generate(
        (8, 12, str(V2), 0.5, 0.05, 0.0, 1.0, {'strong_share': 1.0, 'strong_spec': spec}))
    assert len(bid_x) == len(bid_k) > 0
    assert bid_k.dtype == np.int64
    assert ((0 <= bid_k) & (bid_k <= bid_limit)).all()
