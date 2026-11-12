"""Concrete card-game regression and process-local native tree isolation."""

import random

import pytest

from ohhell import native
from ohhell.arena import deal_round
from ohhell.bots.planning import InformationTree, PlanningBot
from ohhell.engine import BIDDING, RoundState
from ohhell.opponents import PopulationPolicy, SKILLED
from ohhell.rules import Rules
from ohhell.search import playout


def test_later_card_choices_can_rescue_a_hand_fixed_rollouts_miss():
    # Under bowers; the deal before them no longer fooled fixed rollouts.
    state = RoundState(Rules(2, 3), [154652377088, 274877909056], 23, 1)
    state.apply(1)
    state.apply(0)
    fixed = {}
    for card in state.legal_actions():
        trial = state.clone()
        trial.play_unchecked(card)
        playout(trial)
        fixed[card] = trial.points()[0]
    assert fixed == {25: -20, 34: -10, 37: -10}
    tree = InformationTree([34, 25, 37])
    policy = PopulationPolicy([SKILLED] * 2, random.Random(1))
    for _ in range(1200):
        tree.run(state.clone(), policy, 0)
    result = tree.result()
    chosen = max(tree.cards, key=result['visits'].get)
    assert chosen in (34, 37)
    assert result['totals'][chosen] / result['visits'][chosen] > 5
    assert result['deep_visits'] > 100


@pytest.mark.parametrize('fast', [False, True])
def test_one_search_does_not_inherit_another_searchs_tree(fast):
    if fast and not native.available:
        pytest.skip('core not built')
    hands, trump = deal_round(random.Random(15), 4, 6)
    state = RoundState(Rules(4, 6), hands, trump, 3)
    while state.phase == BIDDING:
        state.apply(state.legal_actions()[0])
    obs = state.observation(state.to_act)
    settings = dict(seed=17, fast=fast, worlds=32, endgame_cards=0)
    a = PlanningBot(**settings)
    first = a.play(obs)
    PlanningBot(seed=99, fast=fast, worlds=10, endgame_cards=0).play(obs)
    b = PlanningBot(**settings)
    assert b.play(obs) == first
    assert b.last_search == a.last_search
