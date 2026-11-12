"""One-card bidding that reads every earlier bid as a range of cards.

Each bidder is assumed to follow this same rule. The first bidder (who also
leads) bids 1 when the exact chance that nobody beats their card is over a
half. A later bidder assumes each earlier bidder holds a card from the set
that would have produced their bid, and works out the exact chance of winning
against those ranges plus random cards for everyone still to bid. A bid of 1
from an earlier player therefore narrows them to the top of the deck, and
each further 1 narrows the next player more.

Chances are exact counts over every deal consistent with the ranges: each
earlier seat gets a different card from its own range, and the seats still
to bid share the rest of the deck. Larger hands use the hybrid bot unchanged.
"""

import json
from collections import defaultdict
from functools import lru_cache
from itertools import product
from pathlib import Path

from ..cards import DECK_SIZE, FULL_DECK, NUM_RANKS, RANK_OF, SUIT_OF, beats, cards_in
from ..engine import NO_BID
from ..rules import Scoring
from .learned import HybridBot


def _falling(n, k):
    """n * (n - 1) * ... * (n - k + 1), and 0 when n < k."""
    if n < k:
        return 0
    out = 1
    for i in range(k):
        out *= n - i
    return out


def matchings(sets):
    """Ways to give every set a different card from itself.

    Cards that belong to exactly the same sets are interchangeable, so the
    count runs over those groups rather than over single cards.
    """
    if not sets:
        return 1
    groups = defaultdict(int)
    for card in cards_in(_union(sets)):
        pattern = 0
        for index, members in enumerate(sets):
            if members >> card & 1:
                pattern |= 1 << index
        groups[pattern] += 1
    full = (1 << len(sets)) - 1
    ways = {0: 1}
    for pattern, size in groups.items():
        grown = defaultdict(int)
        for taken, count in ways.items():
            free = pattern & ~taken
            subset = free
            while True:
                used = subset.bit_count()
                if used <= size:
                    grown[taken | subset] += count * _falling(size, used)
                if not subset:
                    break
                subset = (subset - 1) & free
        ways = grown
    return ways.get(full, 0)


def _union(sets):
    out = 0
    for members in sets:
        out |= members
    return out


@lru_cache(maxsize=None)
def _not_beating(card, trump):
    """Cards that would not take the trick from ``card``."""
    out = 0
    for other in range(DECK_SIZE):
        if other != card and not beats(other, card, trump):
            out |= 1 << other
    return out


@lru_cache(maxsize=None)
def _beaten_by(card, trump):
    """Leads that ``card`` takes."""
    out = 0
    for lead in range(DECK_SIZE):
        if lead != card and beats(card, lead, trump):
            out |= 1 << lead
    return out


def win_probability(card, trump_card, players, ranges):
    """Exact chance ``card`` wins the trick, sitting after ``len(ranges)`` bidders.

    ``ranges`` holds, in bidding order, the cards each earlier bidder may
    hold. The first bidder leads. If the ranges cannot all hold at once
    (someone has bid against this rule), the latest are dropped until they can.
    """
    trump = SUIT_OF[trump_card]
    deck = FULL_DECK & ~(1 << trump_card) & ~(1 << card)
    position = len(ranges)
    safe = _not_beating(card, trump) & deck
    constraints = [r & deck for r in ranges]
    total = matchings(constraints)
    while not total:
        for index in range(len(constraints) - 1, -1, -1):
            if constraints[index] != deck:
                constraints[index] = deck
                break
        total = matchings(constraints)
    if position:
        allowed = [_beaten_by(card, trump)] + [safe] * (position - 1)
        wins = matchings([c & a for c, a in zip(constraints, allowed)])
    else:
        wins = 1
    later = players - 1 - position
    rest = deck.bit_count() - position
    return wins / total * _falling(safe.bit_count() - position, later) / _falling(rest, later)


def preferred_bid(probability, scoring, options=(0, 1)):
    """The bid among ``options`` worth the most at this chance of winning."""
    def value(bid):
        return (1 - probability) * scoring.points(bid, 0) + probability * scoring.points(bid, 1)
    return max(options, key=lambda b: (value(b), -b))


@lru_cache(maxsize=None)
def ranges_for(players, trump_card, bids, scoring):
    """The card range each bidder in ``bids`` (bidding order) is read to hold."""
    if not bids:
        return ()
    earlier = ranges_for(players, trump_card, bids[:-1], scoring)
    choices = FULL_DECK & ~(1 << trump_card)
    held = 0
    for card in cards_in(choices):
        chance = win_probability(card, trump_card, players, earlier)
        if preferred_bid(chance, scoring) == bids[-1]:
            held |= 1 << card
    return earlier + (held,)


def bidding_order(obs):
    first = (obs.dealer + 1) % obs.n
    return [(first + i) % obs.n for i in range(obs.n)]


def earlier_bids(obs):
    """The bids made before this seat, in bidding order."""
    order = bidding_order(obs)
    earlier = tuple(obs.bids[s] for s in order[:order.index(obs.seat)])
    if any(b == NO_BID for b in earlier):
        raise ValueError('Earlier seats have not all bid.')
    return earlier


def one_card_analysis(obs):
    """Chance of winning and the bid this rule makes, computed live."""
    scoring = obs.rules.scoring
    ranges = ranges_for(obs.n, obs.trump_card, earlier_bids(obs), scoring)
    card = cards_in(obs.hand)[0]
    chance = win_probability(card, obs.trump_card, obs.n, ranges)
    return chance, preferred_bid(chance, scoring, obs.legal_actions())


# ---------------------------------------------------------------------------
# Precomputed table. Ranges depend only on the table size, the trump card and
# the earlier bids. Relabelling suits so that trump becomes clubs, and its
# same-colour partner spades, leaves the game unchanged, so the table holds
# 13 trump ranks per table size. Every bidding sequence up to the dealer is
# stored: a seat bids 1 exactly when its card is in the range for its earlier
# bids followed by a 1.

TABLE_FILE = Path(__file__).resolve().parent.parent / 'models' / 'one_card_ranges.json'
TABLE_SCORING = Scoring()
_SUIT_MAPS = ((0, 1, 2, 3), (1, 0, 3, 2), (1, 3, 0, 2), (3, 1, 2, 0))  # [trump][suit]


def canonical_card(card, trump_suit):
    """The card after relabelling suits so ``trump_suit`` becomes clubs."""
    return _SUIT_MAPS[trump_suit][SUIT_OF[card]] * NUM_RANKS + RANK_OF[card]


def _bid_key(bids):
    return ''.join(str(b) for b in bids)


def build_table(players, trump_rank):
    """Every range for one table size and trump rank, under the default scoring."""
    ranges_for.cache_clear()
    out = {}
    for length in range(1, players + 1):
        for bids in product((0, 1), repeat=length):
            out[_bid_key(bids)] = ranges_for(players, trump_rank, bids, TABLE_SCORING)[-1]
    ranges_for.cache_clear()
    return out


@lru_cache(maxsize=1)
def load_table():
    data = json.loads(TABLE_FILE.read_text())
    return {int(n): {int(r): table for r, table in ranks.items()}
            for n, ranks in data['tables'].items()}


def table_bid(obs):
    """This rule's bid, looked up in the precomputed table."""
    trump = obs.trump_suit
    table = load_table()[obs.n][RANK_OF[obs.trump_card]]
    card = canonical_card(cards_in(obs.hand)[0], trump)
    wants = 1 if table[_bid_key(earlier_bids(obs) + (1,))] >> card & 1 else 0
    options = obs.legal_actions()
    return wants if wants in options else options[0]


class RangesBot(HybridBot):
    """The hybrid bot, with one-card bids read from earlier bids as ranges.

    ``live=1`` recomputes the ranges instead of using the table (slow on big
    tables until the cache warms). Scoring other than the default is always
    computed live.
    """

    def __init__(self, seed=0, live=0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.live = bool(live)

    def bid(self, obs):
        if obs.hand_size == 1 and not obs.blind:
            if self.live or obs.rules.scoring != TABLE_SCORING:
                return one_card_analysis(obs)[1]
            return table_bid(obs)
        return super().bid(obs)
