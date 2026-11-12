import random
from itertools import permutations

import pytest

from ohhell.arena import deal_round, play_game
from ohhell.bots import make_bot
from ohhell.bots.ranges import (
    RangesBot, canonical_card, earlier_bids, load_table, matchings,
    one_card_analysis, ranges_for, table_bid, win_probability,
)
from ohhell.cards import (
    EFFECTIVE_SUIT, FULL_DECK, POWER, SUIT_OF, beats, cards_in, make_card, mask_of,
)
from ohhell.engine import BIDDING, RoundState
from ohhell.rules import Rules, Scoring

MODELS = 'models=ohhell/models'


def brute_matchings(sets):
    cards = cards_in(FULL_DECK)
    return sum(all(s >> c & 1 for s, c in zip(sets, pick))
               for pick in permutations([c for c in cards if any(s >> c & 1 for s in sets)],
                                        len(sets)))


def test_matchings_counts_distinct_representatives():
    rng = random.Random(3)
    for _ in range(60):
        k = rng.randrange(0, 4)
        pool = rng.sample(range(52), 7)
        sets = [mask_of(rng.sample(pool, rng.randrange(0, 6))) for _ in range(k)]
        assert matchings(sets) == brute_matchings(sets)


def brute_win(card, trump_card, players, ranges):
    trump = SUIT_OF[trump_card]
    deck = [c for c in range(52) if c not in (card, trump_card)]
    wins = total = 0
    for others in permutations(deck, players - 1):
        if any(not r >> c & 1 for r, c in zip(ranges, others)):
            continue
        total += 1
        k = len(ranges)
        table = list(others[:k]) + [card] + list(others[k:])
        best = table[0]
        for c in table[1:]:
            if beats(c, best, trump):
                best = c
        wins += best == card
    return wins / total


@pytest.mark.parametrize('bids', [(), (1,), (0,), (1, 1), (0, 1)])
def test_win_probability_is_exact_for_three_players(bids):
    trump_card = make_card(2, 4)  # hearts trump
    ranges = ranges_for(3, trump_card, bids[:2], Scoring())
    for card in (make_card(2, 9), make_card(1, 9), make_card(2, 12), make_card(0, 12),
                 make_card(3, 3)):
        position = len(ranges)
        if position < 3:
            expected = brute_win(card, trump_card, 3, ranges)
            assert win_probability(card, trump_card, 3, ranges) == pytest.approx(expected)


def test_first_bidder_bids_one_exactly_when_more_likely_to_win():
    trump_card = make_card(0, 3)
    (held,) = ranges_for(4, trump_card, (1,), Scoring())
    for card in range(52):
        if card != trump_card:
            assert (held >> card & 1) == (win_probability(card, trump_card, 4, ()) > 0.5)


def test_each_one_bid_narrows_the_next_range():
    for players in (3, 4, 5):
        bids = ()
        sizes = []
        for _ in range(players - 1):
            bids += (1,)
            sizes.append(ranges_for(players, make_card(1, 7), bids, Scoring())[-1].bit_count())
        assert sizes == sorted(sizes, reverse=True)
        assert sizes[0] > sizes[-1]


def test_inconsistent_bids_still_give_a_probability():
    trump_card = make_card(0, 0)
    right = make_card(0, 9)
    # Two earlier seats both claim the only card in a range.
    chance = win_probability(make_card(0, 12), trump_card, 4, (1 << right, 1 << right))
    assert 0.0 <= chance <= 1.0


def test_canonical_relabelling_preserves_ranking():
    for trump in range(4):
        for a in range(52):
            ca = canonical_card(a, trump)
            assert (EFFECTIVE_SUIT[0][ca] == 0) == (EFFECTIVE_SUIT[trump][a] == trump)
            assert POWER[0][ca] == POWER[trump][a]
            for b in range(52):
                assert beats(a, b, trump) == beats(ca, canonical_card(b, trump), 0)


def one_card_observations(seed, count):
    rng = random.Random(seed)
    for _ in range(count):
        players = rng.randrange(3, 9)
        rules = Rules(players, 1)
        hands, trump_card = deal_round(rng, players, 1)
        state = RoundState(rules, hands, trump_card, rng.randrange(players))
        for _ in range(rng.randrange(players)):
            state.apply(rng.choice(state.legal_actions()))
        yield state.observation(state.to_act)


def test_table_matches_live_calculation():
    load_table.cache_clear()
    for obs in one_card_observations(11, 150):
        assert table_bid(obs) == one_card_analysis(obs)[1]


def test_bids_are_legal_and_respect_dealer_rule():
    for obs in one_card_observations(12, 200):
        assert table_bid(obs) in obs.legal_actions()


def test_registry_and_larger_hands_use_hybrid(monkeypatch):
    bot = make_bot(f'ranges:field=previous,{MODELS}')
    assert isinstance(bot, RangesBot)
    rng = random.Random(5)
    hands, trump_card = deal_round(rng, 4, 3)
    obs = RoundState(Rules(4, 3), hands, trump_card, 0).observation(1)
    called = []
    monkeypatch.setattr('ohhell.bots.learned.HybridBot.analyse_bids',
                        lambda self, o: called.append(o) or type('A', (), {
                            'values': {0: 0.0, 1: 1.0}})())
    assert bot.bid(obs) == 1 and called


def test_full_game_runs():
    rules = Rules(4, 3)
    bots = [make_bot(f'ranges:field=previous,time_ms=1,{MODELS}', seed=s) for s in range(4)]
    totals, _, _ = play_game(bots, rules, seed=2)
    assert len(totals) == 4


def test_earlier_bids_in_bidding_order():
    rules = Rules(4, 1)
    rng = random.Random(1)
    hands, trump_card = deal_round(rng, 4, 1)
    state = RoundState(rules, hands, trump_card, dealer=2)  # seat 3 bids first
    state.apply(1)
    state.apply(0)
    obs = state.observation(state.to_act)
    assert obs.phase == BIDDING and obs.seat == 1
    assert earlier_bids(obs) == (1, 0)
