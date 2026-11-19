import random

import pytest

from bid_game import random_situation, situation
from website import app


def check(result):
    bids = [o['bid'] for o in result['options']]
    assert bids == result['legal_bids'] and result['best_bid'] in bids
    for option in result['options']:
        assert len(option['tricks']) == result['tricks'] + 1
        assert sum(option['tricks']) == pytest.approx(1, abs=0.01)
        assert option['make_chance'] == option['tricks'][option['bid']]
    best = max(o['expected_points'] for o in result['options'])
    assert next(o for o in result['options'] if o['bid'] == result['best_bid'])['expected_points'] == best


def test_every_legal_bid_is_analysed():
    result = situation(random.Random(1), players=4, hand_size=6, position=1)
    check(result)
    assert result['legal_bids'] == list(range(7)) and len(result['hand']) == 6
    seats = result['seats']  # table order, the user first
    assert seats[0]['is_you'] and seats[0]['order'] == 1 and seats[0]['bid'] is None
    assert result['first'] == 3 and seats[3]['order'] == 0 and seats[3]['bid'] is not None
    assert seats[2]['is_dealer'] and seats[2]['order'] == 3


def test_dealer_cannot_make_bids_add_up():
    result = situation(random.Random(2), players=3, hand_size=4, position=2)
    check(result)
    before = sum(s['bid'] for s in result['seats'][1:])
    assert result['is_dealer'] and result['seats'][0]['is_you'] and result['seats'][0]['is_dealer']
    assert result['forbidden_bid'] == (4 - before if before <= 4 else None)


def test_blind_round_hides_the_hand():
    result = situation(random.Random(3), players=5, hand_size=0, position=0)
    check(result)
    assert result['blind'] and result['hand'] == [] and result['tricks'] == 1


def test_random_situations_offer_a_choice():
    rng = random.Random(4)
    for _ in range(3):
        result = random_situation(rng)
        check(result)
        assert len(result['legal_bids']) > 1 and result['seats'][0]['order'] == result['position']


def test_endpoint_and_page():
    client = app.test_client()
    response = client.post('/api/bid-game/round')
    assert response.status_code == 200
    check(response.get_json())


def test_page_is_the_frontend():
    response = app.test_client().get('/bid-game')
    assert response.status_code in (200, 503)  # 503 until the frontend is built
    if response.status_code == 200:
        assert b'id="root"' in response.data
