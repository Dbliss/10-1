import random

import numpy as np

from ohhell.arena import deal_round
from ohhell.cards import LEFT_BOWER, RIGHT_BOWER, make_card, partner_suit
from ohhell.engine import NO_BID
from ohhell.learn import value_bid as vb
from ohhell.rules import Rules


def test_slots_are_a_bijection_with_bowers_on_top():
    for trump in range(4):
        slot = vb.SLOT[trump]
        assert sorted(slot) == list(range(52))
        assert slot[RIGHT_BOWER[trump]] == 0
        assert slot[LEFT_BOWER[trump]] == 1
        assert slot[make_card(trump, 12)] == 2  # Ace of trumps next
        # Same-colour suit (12 cards, its Jack is a trump) follows the 14 trumps.
        assert slot[make_card(partner_suit(trump), 12)] == 14


def test_features_mark_every_card_once_and_are_relative_to_the_hero():
    rng = random.Random(4)
    hands, trump = deal_round(rng, 4, 7)
    bids = [2, 0, 3, 1]
    for hero in range(4):
        x = vb.features(hands, trump, 7, 1, bids, hero, 0)
        assert x.shape == (vb.INPUTS,)
        cards = x[:52 * vb.OWNERS].reshape(52, vb.OWNERS)
        assert np.all(cards.sum(axis=1) == 1)
        assert cards[:, 0].sum() == 7  # hero's own hand
        assert cards[:, vb.UPCARD].sum() == 1
        assert cards[:, vb.UNDEALT].sum() == 52 - 28 - 1
        at = 52 * vb.OWNERS
        assert x[at + 1] == bids[hero] / 10.0
    # Rotating every seat leaves the hero's view unchanged.
    turned = hands[1:] + hands[:1]
    assert np.array_equal(vb.features(hands, trump, 7, 1, bids, 1, 0),
                          vb.features(turned, trump, 7, 0, bids[1:] + bids[:1], 0, 0))


class _Obs:
    def __init__(self, seat, dealer, bids, hand_size):
        self.seat, self.dealer, self.bids, self.hand_size = seat, dealer, bids, hand_size
        self.rules = Rules(len(bids), 10)


def test_our_bid_respects_the_dealer_rule_for_an_imagined_dealer():
    obs = _Obs(seat=1, dealer=3, bids=[1, NO_BID, NO_BID, NO_BID], hand_size=4)
    out = vb._with_our_bid([1, NO_BID, 2, 1], 0, obs)
    assert out == [1, 0, 2, 0]  # dealer moved off the sum of 4
    assert vb._with_our_bid([1, NO_BID, 1, 1], 0, obs) == [1, 0, 1, 1]


def test_predictions_are_distributions_up_to_the_hand_size():
    model = vb.MLP([vb.INPUTS, 16, 11], seed=0)
    rng = random.Random(1)
    hands, trump = deal_round(rng, 5, 3)
    x = np.stack([vb.features(hands, trump, 3, 0, [1, 0, 1, 2, 0], h, 0) for h in range(5)])
    probs = vb.predict(model, x, np.full(5, 3))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs[:, 4:] < 1e-6)


def test_native_value_odds_match_python():
    import pytest
    from ohhell import native
    if not native.available:
        pytest.skip('native core not built')
    model = vb.MLP([vb.INPUTS, 32, 16, 11], seed=3)
    for b in model.biases:
        b[:] = np.random.default_rng(1).standard_normal(b.shape) * 0.3
    net = native.ValueNet(model)
    rng = random.Random(8)
    for _ in range(40):
        n = rng.randint(3, 8)
        hand_size = rng.randint(1, min(10, 51 // n))
        hands, trump = deal_round(rng, n, hand_size)
        dealer, hero = rng.randrange(n), rng.randrange(n)
        bids = [rng.randint(0, hand_size) for _ in range(n)]
        x = vb.features(hands, trump, hand_size, dealer, bids, hero, 0)
        want = vb.predict(model, x[None], np.array([hand_size]))[0][:hand_size + 1]
        got = native.value_odds(net, hands, trump, hand_size, dealer, bids, hero)
        assert np.allclose(got, want, atol=1e-5)
