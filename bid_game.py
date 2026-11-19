"""Random bidding situations for the bidding mini game (``/bid-game``).

Each situation is a fresh deal with a random number of players, hand size and
seat in the bidding order. The seats before the user bid from their own cards
with the game's default bot. The answer travels with the situation: for every
legal bid, the advisor's chance of each trick count and the average points.
The page keeps the answer hidden until the bids are locked in.
"""

import random

from game import Game
from ohhell.arena import deal_round
from ohhell.blind import bid_values
from ohhell.bots import DEFAULT_SPEC, make_bot
from ohhell.cards import (
    EFFECTIVE_SUIT, POWER, RANK_NAMES, RANK_OF, SUIT_NAMES, SUIT_OF, card_id, cards_in,
)
from ohhell.engine import RoundState
from ohhell.rules import Rules

ROUNDS = 10
BLIND_CHANCE = 0.08
# The seats before the user only need a sensible bid; the advisor gets longer.
BIDDER_SPEC = f'{DEFAULT_SPEC},{Game.LONG_THINK},time_ms=60'
ADVISOR_SPEC = f'{Game.ADVISOR_SPEC},time_ms=700'


def _card_json(card, trump):
    return {
        'id': card_id(card),
        'suit': SUIT_NAMES[SUIT_OF[card]],
        'rank': RANK_NAMES[RANK_OF[card]],
        'plays_as': SUIT_NAMES[EFFECTIVE_SUIT[trump][card]],
        'power': POWER[trump][card],
    }


def _sorted_hand(hand, trump):
    """Trumps first, then each other suit, strongest card first."""
    def key(card):
        suit = EFFECTIVE_SUIT[trump][card]
        return (suit != trump, suit, -POWER[trump][card])
    return sorted(cards_in(hand), key=key)


def _analyse(obs, rng):
    """Trick odds and average points for every legal bid."""
    score = obs.rules.scoring.points
    legal = obs.legal_actions()
    if obs.blind:
        values, win = bid_values(obs)
        return {b: ([1 - win, win], values[b]) for b in legal}

    advisor = make_bot(ADVISOR_SPEC, rng.getrandbits(32))
    advisor.start_clock()
    analysis = advisor.analyse_bids(obs)
    result = {b: (list(analysis.distributions[b]), analysis.values[b])
              for b in analysis.values}
    # The advisor only studies its shortlist. The user may pick anything, so
    # the remaining bids are played out too.
    rest = [b for b in legal if b not in result]
    if rest:
        advisor.start_clock()
        extra = advisor._analyse_bids(obs, rest)
        for b in rest:
            dist = list(extra.distributions[b])
            result[b] = (dist, sum(p * score(b, k) for k, p in enumerate(dist)))
    return result


def situation(rng=random, players=None, hand_size=None, position=None):
    """Deal a random bidding decision and analyse every legal bid.

    ``hand_size`` 0 is the blind round. ``position`` is how many seats bid
    before the user. Anything left as None is chosen at random.
    """
    if players is None:
        players = rng.randint(Game.MIN_PLAYERS, Game.MAX_PLAYERS)
    if not Game.MIN_PLAYERS <= players <= Game.MAX_PLAYERS:
        raise ValueError(f'Players must be between {Game.MIN_PLAYERS} and {Game.MAX_PLAYERS}.')
    max_hand = Game.max_hand_size(players)
    if hand_size is None:
        hand_size = 0 if rng.random() < BLIND_CHANCE else rng.randint(1, max_hand)
    if not 0 <= hand_size <= max_hand:
        raise ValueError(f'Hand size must be between 0 and {max_hand} for {players} players.')
    if position is None:
        position = rng.randrange(players)
    if not 0 <= position < players:
        raise ValueError(f'At most {players - 1} seats can bid before you.')
    blind = hand_size == 0
    tricks = max(1, hand_size)

    # Seat 0 is the user. The first bidder sits left of the dealer.
    first = -position % players
    dealer = (first - 1) % players
    hands, trump_card = deal_round(rng, players, tricks)
    state = RoundState(Rules(players, max_hand), hands, trump_card, dealer, blind=blind)
    for _ in range(position):
        seat = state.to_act
        state.apply(make_bot(BIDDER_SPEC, rng.getrandbits(32)).act(state.observation(seat)))

    obs = state.observation(0)
    legal = obs.legal_actions()
    analysis = _analyse(obs, rng)
    options = [{'bid': b,
                'expected_points': round(analysis[b][1], 1),
                'make_chance': round(analysis[b][0][b], 3),
                'tricks': [round(p, 3) for p in analysis[b][0]]}
               for b in legal]
    best = max(options, key=lambda o: o['expected_points'])['bid']

    trump = state.trump_suit
    # Listed in table order, seat 0 (the user) first; ``order`` is the
    # position in the bidding, 0 for the seat left of the dealer.
    seats = []
    for seat in range(players):
        order = (seat - first) % players
        seats.append({
            'name': 'You' if seat == 0 else f'AI {seat}',
            'is_you': seat == 0,
            'is_dealer': seat == dealer,
            'order': order,
            'bid': state.bids[seat] if order < position else None,
        })
    return {
        'players': players,
        'hand_size': hand_size,
        'tricks': tricks,
        'blind': blind,
        'position': position,
        'first': first,
        'is_dealer': position == players - 1,
        'trump': _card_json(trump_card, trump),
        'trump_suit': SUIT_NAMES[trump],
        'hand': [] if blind else [_card_json(c, trump) for c in _sorted_hand(state.hands[0], trump)],
        'seats': seats,
        'legal_bids': legal,
        'forbidden_bid': next((b for b in range(tricks + 1) if b not in legal), None),
        'best_bid': best,
        'options': options,
    }


def random_situation(rng=random):
    """A random situation that leaves a real choice of bids."""
    while True:
        result = situation(rng)
        if len(result['legal_bids']) > 1:
            return result
