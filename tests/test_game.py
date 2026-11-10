import random

import pytest

from game import Game

FAST = {'bot_spec': 'heuristic', 'advisor_spec': 'simulation:worlds=3,bid_worlds=3'}


def play_to_the_end(game, rng):
    """Act as the UI would, checking the JSON at every step."""
    game.start_round()
    seen_trick_complete = False
    for _ in range(5000):
        state = game.get_state()
        if state['awaiting'] == 'nothing':
            return state, seen_trick_complete
        if state['awaiting'] == 'user_bid':
            assert game.receive_bid('You', rng.choice(state['legal_bids'])) is None
        elif state['awaiting'] == 'user_play':
            assert state['playable']
            assert game.play_card('You', rng.choice(state['playable'])) is None
        else:
            if state['awaiting'] == 'collect_trick':
                seen_trick_complete = True
                assert len(state['trick']) == game.num_players
                assert state['trick_winner'] is not None
            game.advance()
    raise AssertionError('game did not finish')


@pytest.mark.parametrize('players,hand', [(2, 3), (4, 10), (8, 6)])
def test_a_whole_game_reaches_game_over(players, hand):
    state, shown = play_to_the_end(Game(players, hand, **FAST), random.Random(players))
    assert shown
    assert state['phase'] == 'game_over'
    sequence = list(range(hand, -1, -1)) + list(range(1, hand + 1))
    assert len(state['history']) == 2 * hand + 1
    assert state['round_sizes'] == sequence
    assert [r['round_size'] for r in state['history']] == sequence
    assert [r['hand_size'] for r in state['history']] == [max(1, size) for size in sequence]
    assert [r['blind'] for r in state['history']].count(True) == 1
    final = state['history'][-1]['results']
    assert [p['score'] for p in state['players']] == [r['score'] for r in final]


def test_round_points_follow_the_house_scoring():
    state, _ = play_to_the_end(Game(4, 6, **FAST), random.Random(1))
    for entry in state['history']:
        assert sum(r['tricks'] for r in entry['results']) == entry['hand_size']
        for r in entry['results']:
            if r['bid'] == r['tricks']:
                assert r['points'] == (10 if r['bid'] == 0 else 10 * r['bid'])
            else:
                assert r['points'] == -10 * abs(r['bid'] - r['tricks'])


def test_state_shows_only_the_users_cards():
    game = Game(4, 5, **FAST)
    game.start_round()
    state = game.get_state()
    assert len(state['hand']) == 5
    assert all('hand' not in p and 'cards' not in p for p in state['players'])


def test_bad_input_is_rejected_without_changing_anything():
    game = Game(2, 3, **FAST)
    game.dealer_index = 1
    game.start_round()
    assert game.current_player_index() == 0
    assert game.receive_bid('You', -1)
    assert game.receive_bid('You', 4)
    assert game.play_card('You', 'nonsense')
    assert game.current_player_index() == 0 and game.phase == 'bidding'


def test_dealer_cannot_make_the_bids_add_up():
    game = Game(2, 3, **FAST)
    game.dealer_index = 0
    game.start_round()
    game.advance()  # the bot, left of the dealer, bids first
    other = game.state.bids[1]
    forbidden = 3 - other
    state = game.get_state()
    assert forbidden not in state['legal_bids']
    assert len(state['legal_bids']) == 3
    assert 'dealer' in game.receive_bid('You', forbidden)
    assert game.phase == 'bidding'
    assert game.receive_bid('You', state['legal_bids'][0]) is None


def test_bots_never_break_the_dealer_rule():
    for seed in range(20):
        random.seed(seed)
        state, _ = play_to_the_end(Game(4, 5, **FAST), random.Random(seed))
        for entry in state['history']:
            assert sum(r['bid'] for r in entry['results']) != entry['hand_size']


def test_advice_covers_the_users_turn_only():
    game = Game(2, 3, **FAST)
    game.dealer_index = 1
    game.start_round()
    advice = game.advice()
    assert advice['kind'] == 'bid'
    assert advice['recommended'] in [o['bid'] for o in advice['options']]

    game.receive_bid('You', 1)
    assert game.advice() is None
    game.advance()
    advice = game.advice()
    assert advice['kind'] == 'card'
    assert advice['recommended'] in game.get_state()['playable']


def test_default_bots_and_advisor_load_and_act():
    game = Game(4, 3)
    game.dealer_index = 3
    game.start_round()
    assert game.advice()['kind'] == 'bid'
    game.receive_bid('You', game.get_state()['legal_bids'][0])
    game.advance()
    assert game.state.bids[1] >= 0


def test_web_routes():
    flask = pytest.importorskip('flask')
    import website
    client = website.app.test_client()
    assert client.post('/api/game', json={'players': 9}).status_code == 400
    assert client.post('/api/game', json={'players': 8, 'hand_size': 10}).status_code == 400
    website.game = None
    assert client.get('/api/game/advice').status_code == 404
    state = client.post('/api/game', json={'players': 4, 'hand_size': 2}).get_json()
    assert state['phase'] == 'bidding'
    assert client.post('/api/game/bid', json={'bid': ''}).status_code == 400
    assert client.post('/api/game/advance').status_code == 200
