"""The bower layout of the play inputs, simulation-only bids and bid_sigma."""

import numpy as np
import pytest

from ohhell.bots import make_bot
from ohhell.cards import card_from_id, mask_of
from ohhell.engine import PLAYING, RoundState
from ohhell.learn.features import PLAY_BOWER_INPUTS, PLAY_INPUTS, play_matrix
from ohhell.rules import Rules


def ids(*names):
    return mask_of(card_from_id(name) for name in names)


def test_bower_inputs_mark_the_partner_suit_and_the_left_bower():
    trump = card_from_id('2_of_Hearts')  # left bower: Jack of Diamonds
    hands = [ids('Jack_of_Diamonds', '3_of_Clubs', '4_of_Clubs'),
             ids('5_of_Spades', '6_of_Spades', '7_of_Spades'),
             ids('8_of_Clubs', '9_of_Clubs', '10_of_Clubs')]
    state = RoundState(Rules(3, 3), hands, trump, 2)
    while state.phase != PLAYING:
        state.apply(state.legal_actions()[0])
    obs = state.observation(0)
    cards = obs.legal_actions()
    narrow = play_matrix(obs, 0, obs.hand, cards)
    wide = play_matrix(obs, 0, obs.hand, cards, PLAY_BOWER_INPUTS)
    assert wide.shape == (len(cards), PLAY_BOWER_INPUTS)
    assert np.array_equal(wide[:, :PLAY_INPUTS], narrow)
    # Suit order: Hearts (trump), Clubs (two held), then the empty Diamonds
    # and Spades by suit number, so Diamonds (the partner suit) is slot 2.
    assert wide[0, PLAY_INPUTS:PLAY_INPUTS + 3].tolist() == [0.0, 1.0, 0.0]
    left = cards.index(card_from_id('Jack_of_Diamonds'))
    assert wide[:, PLAY_INPUTS + 3].tolist() == [float(i == left) for i in range(len(cards))]


def test_zero_blend_bids_from_the_simulation_alone():
    import random
    from ohhell.arena import deal_round
    hands, trump = deal_round(random.Random(3), 4, 6)
    state = RoundState(Rules(4, 6), hands, trump, 3)
    obs = state.observation(0)
    blended = make_bot('hybrid:field=previous,bid_worlds=64,fast=0', 5).analyse_bids(obs)
    alone = make_bot('hybrid:field=previous,bid_worlds=64,fast=0,bid_blend=0', 5).analyse_bids(obs)
    plain = make_bot('simulation:field=previous,inference=1,bid_worlds=64,fast=0', 5).analyse_bids(obs)
    assert alone.distributions == plain.distributions
    assert alone.distributions != blended.distributions


@pytest.mark.parametrize('fast', [False, True])
def test_bid_sigma_overrides_every_kind(fast):
    from ohhell import native
    if fast and not native.available:
        pytest.skip('native core not built')
    bot = make_bot(f'simulation:inference=1,bid_sigma=0.9,fast={int(fast)}', 1)
    assert bot.belief.bid_sigma == 0.9
    if fast:
        assert all(kind.bid_sigma == 0.9 for kind in bot.simulator.params.kinds)
