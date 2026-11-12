"""Bid recommendations for a real-life hand.

Give the cards you hold, the number of players, the trump (the turned-up card,
or just its suit) and the bids already made before you. The game's advisor
then estimates each bid's chance of being made exactly. The advisor is the
game's default bot (the current best version) with more sampled deals per
decision. The calibrated quick estimate (``ohhell/estimate.py``) is shown only
as a note; it does not drive the recommendation.
"""

import random

from game import Game
from ohhell import estimate
from ohhell.blind import bid_values
from ohhell.bots import make_bot
from ohhell.cards import JACK, SUIT_NAMES, card_from_id, make_card, mask_of
from ohhell.engine import RoundState
from ohhell.rules import Rules

ADVISOR_SPEC = Game.ADVISOR_SPEC


def _parse_card(text):
    try:
        return card_from_id(text)
    except (AttributeError, ValueError):
        raise ValueError(f'Unknown card: {text!r}.')


def recommend(players, hand, trump, bids_before, blind=False, rng=random):
    """Return the analysis for the bidder after ``bids_before``.

    ``hand`` is a list of card ids such as ``"Jack_of_Hearts"``. ``trump`` is
    either a card id (the turned-up card) or a suit name; with only a suit,
    a non-Jack card of that suit outside the hand stands in for it.
    ``bids_before`` lists the bids in table order, starting left of the dealer.
    """
    players = int(players)
    if not Game.MIN_PLAYERS <= players <= Game.MAX_PLAYERS:
        raise ValueError(f'Players must be between {Game.MIN_PLAYERS} and {Game.MAX_PLAYERS}.')

    cards = [_parse_card(c) for c in hand]
    if len(set(cards)) != len(cards):
        raise ValueError('The hand lists a card twice.')
    if blind:
        if cards:
            raise ValueError('The blind round has no visible hand.')
        hand_size = 1
    else:
        hand_size = len(cards)
        max_hand = 51 // players
        if not 1 <= hand_size <= max_hand:
            raise ValueError(f'Pick between 1 and {max_hand} cards for {players} players.')

    if trump in SUIT_NAMES:
        suit = SUIT_NAMES.index(trump)
        choices = [make_card(suit, r) for r in range(13)
                   if r != JACK and make_card(suit, r) not in cards]
        trump_card = rng.choice(choices)
    else:
        trump_card = _parse_card(trump)
        if trump_card in cards:
            raise ValueError('The turned-up trump card cannot also be in your hand.')

    bids = [int(b) for b in bids_before]
    if len(bids) >= players:
        raise ValueError(f'At most {players - 1} bids can come before yours.')
    if any(not 0 <= b <= hand_size for b in bids):
        raise ValueError(f'Each bid must be between 0 and {hand_size}.')

    # Seat 0 is the user. The first bidder sits left of the dealer.
    first = -len(bids) % players
    dealer = (first - 1) % players

    # Opponent hands are placeholders; the advisor only sees seat 0's view.
    deck = [c for c in range(52) if c != trump_card and c not in cards]
    rng.shuffle(deck)
    own = cards if not blind else [deck.pop()]
    hands = [mask_of(own)] + [
        mask_of(deck[i * hand_size:(i + 1) * hand_size]) for i in range(players - 1)]
    rules = Rules(players, max(1, min(Game.MAX_HAND_SIZE, 51 // players)))
    state = RoundState(rules, hands, trump_card, dealer, blind=blind)
    for bid in bids:
        state.apply(bid)
    obs = state.observation(0)

    if blind:
        values, probability = bid_values(obs)
        options = [{'bid': b, 'expected_points': round(v, 1),
                    'make_chance': round(probability if b else 1 - probability, 3)}
                   for b, v in values.items()]
    else:
        advisor = make_bot(ADVISOR_SPEC, rng.getrandbits(32))
        analysis = advisor.analyse_bids(obs)
        options = [{'bid': b, 'expected_points': round(analysis.values[b], 1),
                    'make_chance': round(analysis.distributions[b][b], 3)}
                   for b in sorted(analysis.values)]

    legal = obs.legal_actions()
    if blind:
        quick = certain = None
    else:
        hand_mask = mask_of(cards)
        quick = round(estimate.expected_tricks(hand_mask, trump_card, players, hand_size,
                                               len(bids)), 2)
        certain = estimate.certain_tricks(hand_mask, trump_card)
    best = max(options, key=lambda o: o['expected_points'])['bid']
    return {
        'recommended': best,
        'options': options,
        'legal_bids': legal,
        'forbidden_bid': next((b for b in range(hand_size + 1) if b not in legal), None),
        'hand_size': 0 if blind else hand_size,
        'is_dealer': len(bids) == players - 1,
        'trump_card': None if trump in SUIT_NAMES else trump,
        'trump_suit': SUIT_NAMES[obs.trump_suit],
        'quick_estimate': quick,
        'certain_tricks': certain,
    }
