"""The optional play styles: off by default, and the same in Python and C."""

import random

import pytest

from ohhell import native
from ohhell.arena import play_game
from ohhell.bots import make_bot
from ohhell.bots.heuristic import (
    EVERY_SEAT, SCRAMBLE_FOLLOW, SCRAMBLE_LEAD, SCRAMBLE_RUFF, choose_card, state_card, table_call,
    table_read,
)
from ohhell.bots.simulation import SimulationBot
from ohhell.cards import mask_of
from ohhell.engine import NO_SUIT, PLAYING
from ohhell.opponents import SKILLED, PopulationPolicy
from ohhell.rules import Rules
from ohhell.search import playout
from test_native import needs_core, random_round

ALL = SCRAMBLE_FOLLOW | SCRAMBLE_RUFF | SCRAMBLE_LEAD
TRUMP = 0  # the left bower is then the Jack of suit 3; sides below use suits 1 and 2


def card(suit, rank):
    """Rank 0 is a two, 9 a Jack, 12 an Ace."""
    return suit * 13 + rank


def walk(seed, rounds):
    """Every decision point in some randomly played rounds."""
    rng = random.Random(seed)
    for _ in range(rounds):
        state = random_round(rng, rng.choice([2, 3, 4, 6, 8]), rng.randint(1, 6))
        while state.phase == PLAYING:
            yield state
            state.apply(rng.choice(state.legal_actions()))


def test_call_counts_wanted_tricks_against_tricks_left():
    assert table_call([2, 1, 0, 3], [0, 0, 0, 1], 3) == 2
    assert table_call([2, 1, 0, 3], [2, 3, 1, 0], 4) == -1  # taking too many wants nothing


def test_styles_do_nothing_unless_the_round_is_over_called_and_tricks_are_wanted():
    checked = 0
    for state in walk(1, 1500):
        seat = state.to_act
        hand = state.hands[seat]
        need = state.bids[seat] - state.tricks_won[seat]
        call = table_call(state.bids, state.tricks_won, hand.bit_count())
        plain = state_card(state)
        assert state_card(state, False, ALL, seat + 1) == plain  # another seat's style
        if need <= 0 or call <= 0:
            assert state_card(state, False, ALL, seat) == plain
            checked += 1
    assert checked > 3000


def test_over_called_follower_wins_with_the_top_card():
    best = card(1, 5)
    hand = mask_of([card(1, 11), card(1, 1)])
    args = (hand, TRUMP, 1, best, 1, 4, hand | 1 << best, 1, True, False)
    assert choose_card(*args) == card(1, 1)
    assert choose_card(*args, SCRAMBLE_FOLLOW, 0) == card(1, 1)
    assert choose_card(*args, SCRAMBLE_FOLLOW, 1) == card(1, 11)
    assert choose_card(*args, SCRAMBLE_RUFF | SCRAMBLE_LEAD, 1) == card(1, 1)


def test_over_called_ruff_uses_a_trump_likely_to_hold():
    best = card(1, 5)
    hand = mask_of([card(0, 0), card(0, 7), card(0, 11), card(2, 3)])
    # The Ace, right bower, Queen and ten of trumps have gone: only the left
    # bower beats the King or the nine.
    gone = mask_of([card(0, 12), card(0, 9), card(0, 10), card(0, 8)])
    args = (hand, TRUMP, 1, best, 1, 4, hand | gone | 1 << best, 1, True, False)
    assert choose_card(*args) == card(0, 0)
    assert choose_card(*args, SCRAMBLE_RUFF, 1) == card(0, 7)
    assert choose_card(*args, SCRAMBLE_RUFF, 2) == card(0, 11)
    # Last to play, the lowest trump is enough.
    last = (hand, TRUMP, 1, best, 3, 4, hand | gone | 1 << best, 1, True, False)
    assert choose_card(*last, SCRAMBLE_RUFF, 2) == card(0, 0)


def test_over_called_ruff_keeps_a_top_trump_for_later():
    best = card(1, 5)
    hand = mask_of([card(0, 9), card(2, 3)])  # the right bower wins whenever it is played
    args = (hand, TRUMP, 1, best, 1, 4, hand | 1 << best, 1, True, False)
    assert choose_card(*args, SCRAMBLE_RUFF, 2) == card(0, 9)  # the only trump
    hand |= 1 << card(0, 2)
    args = (hand, TRUMP, 1, best, 1, 4, hand | 1 << best, 1, True, False)
    assert choose_card(*args, SCRAMBLE_RUFF, 2) == card(0, 2)


def test_over_called_leader_tries_a_king_under_an_unseen_ace():
    hand = mask_of([card(1, 11), card(1, 2), card(2, 3)])
    args = (hand, TRUMP, NO_SUIT, -1, 0, 4, hand | 1 << card(0, 4), 1, True, False)
    assert choose_card(*args) == card(2, 3)
    assert choose_card(*args, SCRAMBLE_LEAD, 0) == card(2, 3)
    assert choose_card(*args, SCRAMBLE_LEAD, 1) == card(1, 11)
    # A sure winner is still led first.
    ace = hand | 1 << card(2, 12)
    args = (ace, TRUMP, NO_SUIT, -1, 0, 4, ace | 1 << card(0, 4), 1, True, False)
    assert choose_card(*args, SCRAMBLE_LEAD, 1) == card(2, 12)


@needs_core
def test_native_card_choice_matches_python_for_every_style():
    rng = random.Random(3)
    checked = changed = 0
    for state in walk(2, 2500):
        seat = state.to_act
        hand = state.hands[seat]
        seen = state.played | hand | (1 << state.trump_card)
        need = state.bids[seat] - state.tricks_won[seat]
        call = table_call(state.bids, state.tricks_won, hand.bit_count())
        react = rng.random() < 0.3
        read = (table_read(seat, state.n, state.bids, state.tricks_won, state.trick_size,
                           hand.bit_count()) if react else (True, False))
        for style in range(ALL + 1):
            expected = state_card(state, react, style, seat)
            assert expected == native.library.oh_choose_card_style(
                hand, state.trump_suit, state.lead_suit, state.best_card, state.trick_size,
                state.n, seen, need, int(read[0]), int(read[1]), style, call)
            changed += expected != state_card(state, react)
        checked += 1
    assert checked > 8000
    assert changed > 100  # the styles do alter some plays


def test_population_policy_applies_a_style_to_one_seat_only():
    differs = 0
    for state in walk(4, 300):
        if state.trick_size or state.hands[0].bit_count() < 3:
            continue
        plain, styled = state.clone(), state.clone()
        playout(plain, PopulationPolicy([SKILLED] * state.n, random.Random(1)))
        playout(styled, PopulationPolicy([SKILLED] * state.n, random.Random(1), False, ALL, 0))
        differs += plain.tricks_won != styled.tricks_won
        ghost = state.clone()  # no such seat: nobody changes
        playout(ghost, PopulationPolicy([SKILLED] * state.n, random.Random(1), False, ALL, -1))
        assert ghost.tricks_won == plain.tricks_won
    assert differs > 5


@needs_core
@pytest.mark.parametrize('plays', [0, 6, 21])
def test_styled_card_values_agree_with_python(plays):
    rng = random.Random(plays + 11)
    state = random_round(rng)
    for _ in range(plays):
        state.apply(rng.choice(state.legal_actions()))
    obs = state.observation(state.to_act)
    kwargs = dict(seed=3, worlds=1200, inference=True, field='previous', style=ALL)
    slow = SimulationBot(fast=False, **kwargs).evaluate_cards(obs)
    fast = SimulationBot(fast=True, **kwargs).evaluate_cards(obs)
    for option in slow:
        assert fast[option] == pytest.approx(slow[option], abs=5.0)


@pytest.mark.parametrize('spec', [
    f'heuristic:style={ALL}', f'hybrid:field=previous,time_ms=2,style={ALL}'])
@pytest.mark.parametrize('players', [3, 8])
def test_full_games_complete_with_a_style(spec, players):
    bots = [make_bot(spec, 1)] + [make_bot('skilled', seat) for seat in range(1, players)]
    totals, _, _ = play_game(bots, Rules(players, 51 // players if players == 8 else 10), 5)
    assert len(totals) == players


@needs_core
@pytest.mark.parametrize('plays', [6, 21])
def test_card_values_agree_when_every_seat_has_the_style(plays):
    rng = random.Random(plays + 23)
    state = random_round(rng)
    for _ in range(plays):
        state.apply(rng.choice(state.legal_actions()))
    obs = state.observation(state.to_act)
    kwargs = dict(seed=3, worlds=1200, inference=True, field='previous', style=ALL,
                  style_seats='all')
    slow = SimulationBot(fast=False, **kwargs).evaluate_cards(obs)
    fast = SimulationBot(fast=True, **kwargs).evaluate_cards(obs)
    for option in slow:
        assert fast[option] == pytest.approx(slow[option], abs=5.0)


def test_every_seat_style_reaches_the_opponents():
    differs = 0
    for state in walk(6, 300):
        if state.trick_size or state.hands[0].bit_count() < 3:
            continue
        own, everyone = state.clone(), state.clone()
        playout(own, PopulationPolicy([SKILLED] * state.n, random.Random(1), False, ALL, 0))
        playout(everyone, PopulationPolicy([SKILLED] * state.n, random.Random(1), False, ALL,
                                           EVERY_SEAT))
        differs += own.tricks_won != everyone.tricks_won
    assert differs > 5


def test_style_seats_must_be_self_or_all():
    with pytest.raises(ValueError):
        SimulationBot(style=ALL, style_seats='others')


def test_cash_leads_the_highest_sure_side_winner_only_when_tricks_are_wanted():
    from ohhell.bots.learned import HybridBot
    from ohhell.cards import EFFECTIVE_SUIT, OUTRANKS
    led = 0
    for state in walk(8, 400):
        seat = state.to_act
        obs = state.observation(seat)
        always, over = HybridBot(cash=1, fast=False), HybridBot(cash=2, fast=False)
        sure = always.sure_side_lead(obs)
        need = state.bids[seat] - state.tricks_won[seat]
        if state.trick_size or need <= 0:
            assert sure is None
            continue
        call = table_call(state.bids, state.tricks_won, obs.hand.bit_count())
        assert over.sure_side_lead(obs) == (sure if call > 0 else None)
        if sure is not None:
            trump = state.trump_suit
            unseen = ~(obs.played | obs.hand | (1 << obs.trump_card))
            assert obs.hand >> sure & 1 and EFFECTIVE_SUIT[trump][sure] != trump
            assert not OUTRANKS[trump][sure] & unseen
            led += 1
    assert led > 20


def test_cash_must_be_a_known_setting_and_plays_full_games():
    from ohhell.bots.learned import HybridBot
    with pytest.raises(ValueError):
        HybridBot(cash=4)
    bots = [make_bot('hybrid:field=previous,time_ms=2,cash=1', 1)] + [
        make_bot('skilled', seat) for seat in range(1, 4)]
    assert len(play_game(bots, Rules(4, 10), 7)[0]) == 4
