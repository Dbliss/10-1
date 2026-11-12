import json
import random

import pytest

from game import Game
from ohhell.arena import deal_round, play_game
from ohhell.blind import win_probability
from ohhell.bots import make_bot
from ohhell.cards import card_id, cards_in
from ohhell.engine import BIDDING, BLIND_CARD, DONE, PLAYING, RoundState
from ohhell.rules import Rules


@pytest.mark.parametrize('size', [1, 3, 10])
def test_full_round_schedule(size):
    assert Rules(4, size).hand_sizes == (
        tuple(range(size, -1, -1)) + tuple(range(1, size + 1)))


def blind_state():
    hands, trump = deal_round(random.Random(2), 4, 1)
    return RoundState(Rules(), hands, trump, 3, blind=True)


def test_blind_observations_hide_own_and_opponents_cards():
    state = blind_state()
    for seat in range(state.n):
        obs = state.observation(seat)
        assert obs.blind and obs.hand == 0 and obs.history == ()
        assert not hasattr(obs, 'hands')
        assert obs.cards_held(seat) == 1
    assert state.clone().blind
    assert state.observation(0).determinize(state.hands).blind


@pytest.mark.parametrize('spec', ['random', 'heuristic', 'net', 'simulation',
                                 'inference', 'hybrid', 'skilled', 'novice', 'casual'])
def test_every_bot_can_finish_blind_round_without_seeing_hand(spec):
    state = blind_state()
    bot = make_bot(spec)
    while state.phase != DONE:
        obs = state.observation(state.to_act)
        assert obs.hand == 0
        action = bot.act(obs)
        assert action in obs.legal_actions()
        if obs.phase == PLAYING:
            assert action == BLIND_CARD
        state.apply(action)
    assert sum(state.tricks_won) == 1
    assert sum(state.bids) != 1


@pytest.mark.parametrize('players', [2, 3, 4, 8])
@pytest.mark.parametrize('trump', [0, 12, 25, 51])
def test_blind_win_probabilities_sum_to_one(players, trump):
    lead = win_probability(players, trump, True)
    other = win_probability(players, trump, False)
    assert 0 <= other <= lead <= 1
    assert lead + (players - 1) * other == pytest.approx(1)


@pytest.mark.parametrize('trump', [0, 9, 25, 48])
def test_blind_win_probability_matches_every_two_player_deal(trump):
    """Exact odds agree with the engine, bowers included (9 is a Jack)."""
    wins = deals = 0
    deck = [card for card in range(52) if card != trump]
    for lead in deck:
        for other in deck:
            if other == lead:
                continue
            state = RoundState(Rules(2, 1), [1 << lead, 1 << other], trump, 1, blind=True)
            state.apply(0)
            state.apply(0)
            state.apply(BLIND_CARD)
            state.apply(BLIND_CARD)
            wins += state.tricks_won[0]
            deals += 1
    assert win_probability(2, trump, True) == pytest.approx(wins / deals)


def test_blind_api_hides_card_ids_and_reveals_only_when_played():
    game = Game(4, 1, bot_spec='heuristic')
    game.round_number = 1  # second round is blind
    game.dealer_index = 2  # rotates to 3, so human bids and leads first
    game.start_round()
    own_card = card_id(cards_in(game.state.hands[0])[0])
    state = game.get_state()
    assert state['blind'] and state['hand'] == [] and state['hand_size'] == 1
    assert own_card not in json.dumps(state)
    advice = game.advice()
    assert advice['kind'] == 'bid'
    assert own_card not in json.dumps(advice)
    game.receive_bid('You', 0)
    while game.phase == 'bidding':
        game.advance()
    assert game.get_state()['playable'] == ['blind']
    assert game.advice() == {'kind': 'blind', 'recommended': 'blind'}
    assert game.play_card('You', own_card) == 'Play your hidden card.'
    assert game.play_card('You', 'blind') is None
    assert game.get_state()['trick'][0]['card']['id'] == own_card


def test_arena_uses_the_same_schedule_and_hidden_observations():
    class Recorder:
        def __init__(self):
            self.rounds = []
            self.bot = make_bot('heuristic')

        def act(self, obs):
            if obs.phase == BIDDING:
                self.rounds.append(0 if obs.blind else obs.hand_size)
            if obs.blind:
                assert obs.hand == 0
            return self.bot.act(obs)

    bots = [Recorder() for _ in range(4)]
    play_game(bots, Rules(4, 2), 1)
    assert all(bot.rounds == [2, 1, 0, 1, 2] for bot in bots)
