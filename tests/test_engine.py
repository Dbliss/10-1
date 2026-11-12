import random

import pytest

from ohhell.arena import deal_round, play_game
from ohhell.belief import Belief, sample_world
from ohhell.bots import make_bot
from ohhell.cards import (
    EFFECTIVE_SUIT, card_from_id, card_id, cards_in, make_card, mask_of,
)
from ohhell.bots.heuristic import state_card
from ohhell.engine import BIDDING, DONE, PLAYING, RoundState
from ohhell.opponents import NOVICE, SKILLED, noisy_bid
from ohhell.rules import Rules, Scoring
from ohhell.search import distinct_moves, playout, solve, solve_root

CLUBS, DIAMONDS, HEARTS, SPADES = range(4)
TWO, THREE, FOUR, FIVE, SIX, SEVEN = range(6)
JACK, QUEEN, KING, ACE = 9, 10, 11, 12


def C(suit, rank):
    return make_card(suit, rank)


def table(hands, trump, dealer=3, **rules):
    n = len(hands)
    return RoundState(Rules(n, max(len(h) for h in hands), **rules),
                      [mask_of(h) for h in hands], trump, dealer)


def bid_all(state, bids):
    for bid in bids:
        state.apply(bid)


@pytest.mark.parametrize('bid,tricks,points', [
    (0, 0, 10), (1, 1, 10), (3, 3, 30), (10, 10, 100),
    (0, 2, -20), (3, 1, -20), (2, 5, -30),
])
def test_scoring_matches_house_rules(bid, tricks, points):
    assert Scoring().points(bid, tricks) == points


def test_card_ids_round_trip():
    for card in range(52):
        assert card_from_id(card_id(card)) == card


def test_bid_order_starts_left_of_dealer_and_first_bidder_leads():
    state = table([[C(CLUBS, TWO)], [C(CLUBS, THREE)], [C(CLUBS, FOUR)], [C(CLUBS, FIVE)]],
                  C(HEARTS, TWO), dealer=1)
    assert state.to_act == 2
    bid_all(state, [0, 0, 0, 0])
    assert state.phase == PLAYING and state.to_act == 2


def test_must_follow_suit():
    state = table([[C(CLUBS, TWO), C(SPADES, ACE)], [C(CLUBS, THREE), C(DIAMONDS, TWO)]],
                  C(HEARTS, TWO), dealer=1)
    bid_all(state, [0, 0])
    state.apply(C(CLUBS, TWO))
    assert state.legal_actions() == [C(CLUBS, THREE)]
    with pytest.raises(ValueError):
        state.apply(C(DIAMONDS, TWO))


@pytest.mark.parametrize('plays,winner', [
    ([C(CLUBS, TWO), C(CLUBS, ACE), C(CLUBS, KING), C(CLUBS, THREE)], 1),
    ([C(CLUBS, ACE), C(DIAMONDS, ACE), C(CLUBS, TWO), C(CLUBS, THREE)], 0),
    ([C(CLUBS, ACE), C(HEARTS, THREE), C(CLUBS, TWO), C(CLUBS, THREE)], 1),
    ([C(CLUBS, ACE), C(HEARTS, THREE), C(HEARTS, FOUR), C(CLUBS, THREE)], 2),
    ([C(CLUBS, ACE), C(HEARTS, FOUR), C(HEARTS, THREE), C(CLUBS, THREE)], 1),
])
def test_trick_winner(plays, winner):
    state = table([[c] for c in plays], C(HEARTS, ACE))
    bid_all(state, [0, 0, 0, 0])
    for card in plays:
        state.apply(card)
    assert state.phase == DONE
    assert state.tricks_won[winner] == 1 and sum(state.tricks_won) == 1


def test_dealer_restriction_only_binds_last_bidder():
    hands = [[C(CLUBS, r)] for r in range(3)]
    free = table(hands, C(HEARTS, TWO), dealer=2, dealer_restriction=False)
    bid_all(free, [1, 0])
    assert free.legal_actions() == [0, 1]
    hooked = table(hands, C(HEARTS, TWO), dealer=2, dealer_restriction=True)
    assert hooked.legal_actions() == [0, 1]
    bid_all(hooked, [0, 0])
    assert hooked.legal_actions() == [0]


def test_observation_hides_other_hands():
    hands = [[C(CLUBS, TWO)], [C(SPADES, ACE)]]
    obs = table(hands, C(HEARTS, TWO), dealer=1).observation(0)
    assert obs.hand == 1 << C(CLUBS, TWO)
    assert not hasattr(obs, 'hands')


def test_void_is_recorded_when_a_seat_fails_to_follow():
    state = table([[C(CLUBS, TWO), C(CLUBS, FIVE)], [C(SPADES, ACE), C(SPADES, TWO)]],
                  C(HEARTS, TWO), dealer=1)
    bid_all(state, [0, 0])
    state.apply(C(CLUBS, TWO))
    state.apply(C(SPADES, TWO))
    assert state.voids[1] == 1 << CLUBS and state.voids[0] == 0


# Bowers, with hearts as trump: the Jack of Hearts is the right bower and the
# Jack of Diamonds, the other red Jack, is the left bower.
RIGHT, LEFT = C(HEARTS, JACK), C(DIAMONDS, JACK)


@pytest.mark.parametrize('plays,winner', [
    ([C(HEARTS, ACE), RIGHT, C(HEARTS, KING), C(HEARTS, TWO)], 1),
    ([C(HEARTS, ACE), LEFT, C(HEARTS, KING), C(HEARTS, TWO)], 1),
    ([LEFT, C(HEARTS, ACE), RIGHT, C(HEARTS, TWO)], 2),
    ([RIGHT, LEFT, C(HEARTS, ACE), C(HEARTS, KING)], 0),
    # The left bower trumps a diamond lead rather than following it.
    ([C(DIAMONDS, ACE), LEFT, C(DIAMONDS, KING), C(DIAMONDS, TWO)], 1),
    # A Jack of a black suit ranks as an ordinary Jack.
    ([C(CLUBS, 8), C(CLUBS, JACK), C(CLUBS, QUEEN), C(SPADES, JACK)], 2),
])
def test_bowers_top_the_trump_suit(plays, winner):
    state = table([[c] for c in plays], C(HEARTS, FIVE))
    bid_all(state, [0, 0, 0, 0])
    for card in plays:
        state.apply(card)
    assert state.tricks_won[winner] == 1 and sum(state.tricks_won) == 1


def test_left_bower_follows_trump_not_its_printed_suit():
    hands = [[C(HEARTS, TWO), C(DIAMONDS, TWO)],
             [LEFT, C(CLUBS, TWO)],
             [C(DIAMONDS, THREE), C(SPADES, TWO)]]
    state = table(hands, C(HEARTS, FIVE), dealer=2)
    bid_all(state, [0, 0, 0])
    state.apply(C(HEARTS, TWO))
    # The only heart the seat holds is the left bower, so it must be played.
    assert state.legal_actions() == [LEFT]

    state = table(hands, C(HEARTS, FIVE), dealer=2)
    bid_all(state, [0, 0, 0])
    state.apply(C(DIAMONDS, TWO))
    # Holding no diamond by effective suit, the seat may play anything.
    assert state.legal_actions() == [C(CLUBS, TWO), LEFT]
    state.apply(LEFT)
    assert state.voids[1] == 1 << DIAMONDS
    state.apply(C(DIAMONDS, THREE))
    assert state.tricks_won == [0, 1, 0]


def test_leading_the_left_bower_leads_trump():
    state = table([[LEFT, C(CLUBS, TWO)], [C(DIAMONDS, ACE), C(HEARTS, TWO)]],
                  C(HEARTS, FIVE), dealer=1)
    bid_all(state, [0, 0])
    state.apply(LEFT)
    assert state.lead_suit == HEARTS
    assert state.legal_actions() == [C(HEARTS, TWO)]


def test_turned_up_right_bower_leaves_left_bower_on_top():
    plays = [C(HEARTS, ACE), LEFT]
    state = table([[c] for c in plays], RIGHT, dealer=1)
    bid_all(state, [0, 0])
    for card in plays:
        state.apply(card)
    assert state.tricks_won == [0, 1]


def test_touching_trumps_include_the_bowers():
    def moves(mine, other):
        state = table([mine, other], C(HEARTS, TWO), dealer=1)
        bid_all(state, [0, 0])
        return distinct_moves(state)

    spades = [C(SPADES, TWO), C(SPADES, THREE), C(SPADES, FOUR)]
    # King, Ace and both bowers run unbroken, so they are a single move.
    assert moves([C(HEARTS, KING), C(HEARTS, ACE), LEFT, RIGHT],
                 [C(HEARTS, QUEEN)] + spades) == [RIGHT]
    # An Ace in another hand splits the King from the bowers.
    assert moves([C(HEARTS, KING), C(HEARTS, QUEEN), LEFT, RIGHT],
                 [C(HEARTS, ACE)] + spades) == [C(HEARTS, KING), RIGHT]


def mid_round(seed, players=4, hand_size=8, plays=13):
    rng = random.Random(seed)
    hands, trump = deal_round(rng, players, hand_size)
    state = RoundState(Rules(players, hand_size), hands, trump, 0)
    while state.phase == BIDDING:
        state.apply(rng.choice(state.legal_actions()))
    for _ in range(plays):
        state.apply(rng.choice(state.legal_actions()))
    return state


@pytest.mark.parametrize('seed', range(30))
@pytest.mark.parametrize('inference', [False, True])
def test_sampled_worlds_respect_everything_known(seed, inference):
    state = mid_round(seed)
    obs = state.observation(state.to_act)
    rng = random.Random(seed)
    for hands, kinds in Belief(inference=inference).worlds(obs, 8, rng):
        assert len(kinds) == obs.n
        assert hands[obs.seat] == obs.hand
        all_cards = 0
        for s, hand in enumerate(hands):
            assert hand.bit_count() == state.hands[s].bit_count()
            assert not all_cards & hand
            all_cards |= hand
            for card in cards_in(hand):
                assert not obs.voids[s] >> EFFECTIVE_SUIT[obs.trump_suit][card] & 1
        assert not all_cards & obs.played
        assert not all_cards >> obs.trump_card & 1


def test_sampling_succeeds_with_a_tight_full_table():
    for seed in range(100):
        state = mid_round(seed, players=8, hand_size=6, plays=30)
        sample_world(state.observation(state.to_act), random.Random(seed))


def brute_force(state, root):
    """Same recursion as ``solve`` with no pruning or memo."""
    if state.phase != PLAYING:
        return tuple(state.points())
    seat = state.to_act
    best = None
    for card in state.legal_actions():
        child = state.clone()
        child.play_unchecked(card)
        value = brute_force(child, root)
        if (best is None or value[seat] > best[seat]
                or (value[seat] == best[seat] and value[root] < best[root])):
            best = value
    return best


@pytest.mark.parametrize('seed', range(40))
def test_exact_search_keeps_the_root_value_of_brute_force(seed):
    rng = random.Random(seed)
    players = rng.choice([2, 3, 4])
    hands, trump = deal_round(rng, players, 3)
    state = RoundState(Rules(players, 3), hands, trump, 0)
    while state.phase == BIDDING:
        state.apply(rng.choice(state.legal_actions()))
    seat = state.to_act
    exact = brute_force(state, seat)[seat]
    fast = max(solve_root(state, state.legal_actions(), seat).values())
    assert exact == fast


def test_touching_cards_count_as_one_move():
    state = table([[C(CLUBS, FOUR), C(CLUBS, FIVE), C(CLUBS, SEVEN)],
                   [C(CLUBS, SIX), C(SPADES, TWO), C(SPADES, THREE)]],
                  C(HEARTS, TWO), dealer=1)
    bid_all(state, [0, 0])
    assert len(distinct_moves(state)) == 2


@pytest.mark.parametrize('spec', [
    'random', 'heuristic',
    'simulation:worlds=3,bid_worlds=3',
    'inference:worlds=3,bid_worlds=3,oversample=2',
    'net', 'novice', 'casual', 'mixed',
    'hybrid:worlds=3,bid_worlds=3,oversample=2',
])
@pytest.mark.parametrize('players,hand', [(2, 5), (4, 6), (8, 6)])
@pytest.mark.parametrize('restriction', [False, True])
def test_bots_finish_legal_games(spec, players, hand, restriction):
    # state.apply rejects any illegal action, so finishing proves legality.
    rules = Rules(players, hand, restriction)
    totals, _, _ = play_game([make_bot(spec, s) for s in range(players)], rules, 7)
    assert len(totals) == players


def test_same_seed_means_same_game():
    rules = Rules(4, 5)
    make = lambda: [make_bot('heuristic', s) for s in range(4)]
    assert play_game(make(), rules, 11)[0] == play_game(make(), rules, 11)[0]


def test_playout_finishes_with_every_trick_awarded():
    state = mid_round(3)
    copy = state.clone()
    playout(copy)
    assert copy.phase == DONE and sum(copy.tricks_won) == 8


def test_noisy_bids_stay_legal():
    rng = random.Random(1)
    options = [0, 1, 3]
    for _ in range(200):
        assert noisy_bid(1, NOVICE, rng, options) in options


def test_inference_recognises_how_an_opponent_has_been_playing():
    """Deals are weighted up for the kind that explains the plays so far."""
    rng = random.Random(4)
    hands, trump = deal_round(rng, 2, 8)
    state = RoundState(Rules(2, 8), hands, trump, 1)
    state.apply(4)
    state.apply(5)
    while state.hands[0].bit_count() > 3 or state.to_act != 0:
        cards = state.legal_actions()
        state.apply(state_card(state) if state.to_act == 1 else cards[0])
    obs = state.observation(0)
    worlds = [state.hands[0], state.hands[1]]
    belief = Belief()
    assert (belief.log_weight(obs, worlds, [SKILLED, SKILLED])
            > belief.log_weight(obs, worlds, [SKILLED, NOVICE]))
