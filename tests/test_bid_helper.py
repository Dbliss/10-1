import random

import pytest

from bid_helper import recommend
from website import app

HAND = ['Ace_of_Spades', 'Jack_of_Hearts', 'King_of_Hearts', '2_of_Clubs', '7_of_Diamonds']


def test_recommends_a_listed_bid():
    result = recommend(4, HAND, 'Hearts', [1, 2], rng=random.Random(1))
    bids = [o['bid'] for o in result['options']]
    assert result['recommended'] in bids
    assert all(0 <= o['make_chance'] <= 1 for o in result['options'])
    assert not result['is_dealer']


def test_dealer_cannot_make_bids_add_up():
    result = recommend(4, HAND, '9_of_Hearts', [1, 2, 0], rng=random.Random(1))
    assert result['is_dealer'] and result['forbidden_bid'] == 2
    assert 2 not in [o['bid'] for o in result['options']]


def test_blind_round():
    result = recommend(5, [], 'Spades', [1], blind=True, rng=random.Random(1))
    assert sorted(o['bid'] for o in result['options']) == [0, 1]


@pytest.mark.parametrize('kwargs', [
    dict(players=4, hand=HAND, trump='Ace_of_Spades', bids_before=[]),   # trump in hand
    dict(players=4, hand=HAND, trump='Hearts', bids_before=[1, 1, 1, 1]),  # too many bids
    dict(players=4, hand=HAND, trump='Hearts', bids_before=[6]),          # bid above hand
    dict(players=4, hand=[], trump='Hearts', bids_before=[]),             # empty hand
    dict(players=9, hand=HAND, trump='Hearts', bids_before=[]),
])
def test_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        recommend(**kwargs)


def test_endpoint_and_page():
    client = app.test_client()
    ok = client.post('/api/bid-helper', json={'players': 3, 'hand': HAND, 'trump': 'Spades', 'bids': [2]})
    assert ok.status_code == 200 and 'recommended' in ok.get_json()
    bad = client.post('/api/bid-helper', json={'players': 3, 'hand': ['Nope'], 'trump': 'Spades'})
    assert bad.status_code == 400 and 'error' in bad.get_json()
    assert b'Bid Helper' in client.get('/bid').data
