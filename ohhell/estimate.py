"""Quick trick estimate for bidding, versions 2 and 3.

``card_win_probs`` in ``bots/heuristic.py`` is the original rule of thumb. It
stays as it is because the trained networks take it as an input. This module
is a calibrated replacement for decisions that only need a better estimate,
such as which bids the simulation should compare.

Each card gets exact counting features, then a logistic model fitted to
play-outs (``python -m ohhell.learn.calibrate_estimate``) turns them into a
win chance. The features are:

* **Falling first**: your i-th highest card of a suit (0 = highest) wins if
  at most i higher unseen cards were dealt to opponents, because those fall
  on your earlier cards. Exact hypergeometric chance.
* **Length**: every opponent holds at most i cards of the suit, so by the
  time you play this card they are out of it.
* **Ruff risk** (side suits): an opponent is out of the suit by then and
  still holds a trump.
* **Move order**: whether you lead the first trick, where you sit, and how
  strong your other cards are, since winning any trick gives you the lead.
* **Trump shape**: side-suit voids and singletons that let low trumps ruff.
* **Discards** (version 3): short other side suits, where a seat trying to
  lose throws away its highest side card, often an Ace.

Trumps with no unseen higher trump win whichever trick they are played in,
so they are certain tricks and get probability 1.

Version 2 is fitted on tables that bid with the original rule of thumb
(``models/trick_estimate.json``). Version 3 (``models/trick_estimate3.json``)
adds the discard features and a fitted spread, and is fitted on tables that
bid with version 3 itself, so its imagined bidders and its training tables
agree. Version 2 weights are padded with zeros for the discard features and
use no spread, so they give exactly what they always did.
"""

import json
import math
from functools import lru_cache
from pathlib import Path

from .cards import OUTRANKS, POWER, SUIT_CARDS, SUIT_OF, cards_in

COEFFICIENTS = Path(__file__).parent / 'models' / 'trick_estimate.json'
COEFFICIENT_FILES = {2: COEFFICIENTS,
                     3: Path(__file__).parent / 'models' / 'trick_estimate3.json'}
NO_SPREAD = (1.0, 0.0, 0.0)

# Feature order is shared with card_win_probs2 in native/core.c.
COMMON = ('bias', 'logit_top', 'logit_clear', 'logit_long', 'leads', 'leads_top',
          'others', 'others_top', 'position', 'index', 'length', 'hand', 'players', 'one_card',
          # Leading, or sitting early, helps most when this card is a likely winner.
          'leads_top_win', 'position_top', 'last', 'last_top')
SIDE_BASE = COMMON + ('log_safe', 'log_safe_top')
TRUMP_BASE = COMMON + ('voids', 'singletons', 'ruff_low')
# Interactions: each signal's weight changes with hand size and table size,
# and a one-card round is a different game (you play your card in the first trick).
SCALED = ('logit_top', 'logit_clear', 'logit_long', 'leads', 'leads_top', 'others', 'position',
          'leads_top_win', 'position_top', 'last', 'last_top')
SCALES = ('hand', 'one_card', 'players')
SIDE_SCALED = SCALED + ('log_safe',)
TRUMP_SCALED = SCALED + ('ruff_low',)
# Version 3: other side suits that are void (1) or singleton (0.5), so this
# seat may discard on them; a seat trying to lose discards its highest card.
# Then the top card of a suit in the last seat (the dealer, whose bid the
# dealer rule can force off) and at big tables, where Aces were overrated.
# An Ace's logit_top saturates, so last*logit_top singles out the dealer's
# boss cards, which the seat's other top cards otherwise averaged away.
DISCARDS = ('short_others', 'short_others_top', 'short_others_top*players',
            'last_first', 'first*players', 'last_first*hand', 'first*players*hand',
            'last*logit_top', 'last*logit_top*hand')
SIDE = SIDE_BASE + tuple(f'{f}*{g}' for g in SCALES for f in SIDE_SCALED) + DISCARDS
TRUMP = TRUMP_BASE + tuple(f'{f}*{g}' for g in SCALES for f in TRUMP_SCALED)

_EPS = 1e-4


@lru_cache(maxsize=None)
def _hyper_pmf(population, successes, draws):
    """P(k successes) for k = 0..draws, drawing without replacement."""
    total = math.comb(population, draws)
    return tuple(
        math.comb(successes, k) * math.comb(population - successes, draws - k) / total
        for k in range(draws + 1))


def _at_most(pmf, k):
    return sum(pmf[:k + 1]) if k >= 0 else 0.0


def _logit(p):
    p = min(1.0 - _EPS, max(_EPS, p))
    return math.log(p / (1.0 - p))


@lru_cache(maxsize=None)
def _ruff_chance(unseen, in_suit, trumps, hand_size, index):
    """P(one opponent holds at most ``index`` of the suit and at least one trump)."""
    pmf = _hyper_pmf(unseen, in_suit, hand_size)
    rest = unseen - in_suit
    chance = 0.0
    for held in range(min(index, hand_size) + 1):
        if pmf[held]:
            others = hand_size - held
            no_trump = (math.comb(rest - trumps, others) / math.comb(rest, others)
                        if others <= rest else 0.0)
            chance += pmf[held] * (1.0 - no_trump)
    return chance


def card_features(hand, trump_card, n, hand_size, position=0):
    """Return ``(card, is_trump, features, certain)`` for every card in the hand.

    ``position`` is the seat's place in the first trick: 0 leads, n-1 plays last.
    """
    trump = SUIT_OF[trump_card]
    suits = SUIT_CARDS[trump]
    outranks = OUTRANKS[trump]
    power = POWER[trump]
    seen = hand | (1 << trump_card)
    unseen = 51 - hand_size
    dealt = (n - 1) * hand_size
    unseen_trumps = (suits[trump] & ~seen).bit_count()
    lengths = [(hand & suits[s]).bit_count() for s in range(4)]
    voids = sum(lengths[s] == 0 for s in range(4) if s != trump)
    singletons = sum(lengths[s] == 1 for s in range(4) if s != trump)
    shortness = [(lengths[s] == 0) + 0.5 * (lengths[s] == 1) for s in range(4)]
    leads = position == 0
    last = n > 1 and position == n - 1

    raw = []
    for suit in range(4):
        cards = sorted(cards_in(hand & suits[suit]), key=power.__getitem__, reverse=True)
        outstanding = (suits[suit] & ~seen).bit_count()
        length_pmf = _hyper_pmf(unseen, outstanding, hand_size)
        for index, card in enumerate(cards):
            higher = (outranks[card] & ~seen).bit_count()
            fall = _hyper_pmf(unseen, higher, dealt)
            top = _at_most(fall, index)
            clear = fall[0]
            longest = _at_most(length_pmf, index) ** (n - 1)
            is_trump = suit == trump
            if is_trump:
                safe = 1.0
                structural = max(top, longest)
            else:
                safe = (1.0 - _ruff_chance(unseen, outstanding, unseen_trumps,
                                           hand_size, index)) ** (n - 1)
                structural = max(top, longest) * safe
            short_others = sum(shortness[s] for s in range(4) if s != trump and s != suit)
            raw.append((card, is_trump, index, len(cards), top, clear, longest, safe,
                        structural, is_trump and higher == 0, short_others))

    total = sum(r[8] for r in raw)
    rows = []
    for (card, is_trump, index, length, top, clear, longest, safe, structural, certain,
         short_others) in raw:
        others = total - structural
        first = 1.0 if index == 0 else 0.0
        seat = position / max(1, n - 1)
        f = {
            'bias': 1.0, 'logit_top': _logit(top), 'logit_clear': _logit(clear),
            'logit_long': _logit(longest), 'leads': float(leads),
            'leads_top': float(leads) * first, 'others': others, 'others_top': others * first,
            'position': seat, 'index': float(index), 'length': float(length),
            'hand': hand_size / 10.0, 'players': n / 8.0, 'one_card': float(hand_size == 1),
            'leads_top_win': float(leads) * first * top, 'position_top': seat * top,
            'last': float(last), 'last_top': float(last) * top,
        }
        if is_trump:
            f['voids'] = float(voids)
            f['singletons'] = float(singletons)
            f['ruff_low'] = (voids + 0.5 * singletons) * (1.0 - top)
            base, scaled = TRUMP_BASE, TRUMP_SCALED
        else:
            f['log_safe'] = math.log(max(_EPS, safe))
            f['log_safe_top'] = f['log_safe'] * first
            base, scaled = SIDE_BASE, SIDE_SCALED
        features = [f[name] for name in base]
        features += [f[name] * f[g] for g in SCALES for name in scaled]
        if not is_trump:
            features += [short_others, short_others * top, short_others * top * f['players'],
                         f['last'] * first, first * f['players'],
                         f['last'] * first * f['hand'], first * f['players'] * f['hand'],
                         f['last'] * f['logit_top'], f['last'] * f['logit_top'] * f['hand']]
        rows.append((card, is_trump, features, certain))
    return rows


@lru_cache(maxsize=4)
def _coefficients(version=2):
    """``(side, trump, spread)`` for an estimate version.

    Shorter weight lists (version 2 predates the discard features) are padded
    with zeros, and a missing spread means none.
    """
    data = json.loads(COEFFICIENT_FILES[version].read_text())
    side = tuple(data['side']) + (0.0,) * (len(SIDE) - len(data['side']))
    return side, tuple(data['trump']), tuple(data.get('spread', NO_SPREAD))


def _weights(version=2):
    side, trump, _ = _coefficients(version)
    return side, trump


def spread(version=2):
    return _coefficients(version)[2]


def card_win_probs(hand, trump_card, n, hand_size, position=0, weights=None, version=2):
    """Chance each card wins a trick, as ``{card: probability}``."""
    side, trump = weights or _weights(version)
    probs = {}
    for card, is_trump, features, certain in card_features(hand, trump_card, n, hand_size, position):
        if certain:
            probs[card] = 1.0
            continue
        w = trump if is_trump else side
        z = sum(a * b for a, b in zip(w, features))
        probs[card] = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
    return probs


def certain_tricks(hand, trump_card):
    """Trumps with no unseen higher trump: each wins the trick it is played in."""
    trump = SUIT_OF[trump_card]
    seen = hand | (1 << trump_card)
    return sum(1 for card in cards_in(hand & SUIT_CARDS[trump][trump])
               if not OUTRANKS[trump][card] & ~seen)


def trick_distribution(probs):
    """P(exactly k tricks), treating cards as independent, k = 0..len(probs)."""
    dist = [1.0]
    for p in probs:
        grown = [0.0] * (len(dist) + 1)
        for k, weight in enumerate(dist):
            grown[k] += weight * (1.0 - p)
            grown[k + 1] += weight * p
        dist = grown
    return dist


def spread_temperature(hand_size, n, coefficients=NO_SPREAD):
    a, b, c = coefficients
    return max(0.2, a + b * hand_size / 10.0 + c * n / 8.0)


def widen(dist, temperature):
    """Flatten (above 1) or sharpen (below 1) a trick distribution.

    Treating cards as independent misjudges how spread out the trick count
    is, because a hand's cards win or lose together. Each probability is
    raised to ``1 / temperature`` and the result renormalised.
    """
    if temperature == 1.0:
        return list(dist)
    powered = [p ** (1.0 / temperature) if p > 0.0 else 0.0 for p in dist]
    total = sum(powered)
    return [p / total for p in powered]


def bid_distribution(hand, trump_card, n, hand_size, position=0, version=2):
    """P(exactly k tricks) under this version, spread included, for choosing a bid."""
    probs = card_win_probs(hand, trump_card, n, hand_size, position, version=version)
    dist = trick_distribution(list(probs.values()))
    return widen(dist, spread_temperature(hand_size, n, spread(version)))


def expected_tricks(hand, trump_card, n, hand_size, position=0, version=2):
    return sum(card_win_probs(hand, trump_card, n, hand_size, position,
                              version=version).values())


def first_trick_position(seat, dealer, n):
    """0 for the player left of the dealer, who leads the first trick."""
    return (seat - dealer - 1) % n
