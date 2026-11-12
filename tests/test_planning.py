"""History search must value information without seeing the sampled secret."""

import random
import time
from types import SimpleNamespace

import pytest

from ohhell import native
from ohhell.arena import deal_round, play_game
from ohhell.bots import DEFAULT_SPEC, make_bot
from ohhell.bots.learned import HybridBot
from ohhell.bots.planning import InformationTree, PlanningBot
from ohhell.engine import BIDDING, DONE, PLAYING, RoundState
from ohhell.rules import Rules
from ohhell.tournament import check_run_config


class ProbeGame:
    """Safe earns 6. Probe costs 1, then guessing a hidden bit earns 10.

    An opponent's public response may reveal the bit. Nothing in the legal
    actions or our own rollout policy depends on that hidden bit.
    """

    def __init__(self, secret, revealing=True):
        self.secret = secret
        self.revealing = revealing
        self.phase = PLAYING
        self.to_act = 0
        self.stage = 0
        self.reward = 0

    def legal_actions(self):
        return [0, 1] if self.stage == 0 else [2, 3]

    def play_unchecked(self, card):
        if self.stage == 0:
            if card == 1:  # safe
                self.reward, self.phase = 6, DONE
            else:
                self.to_act = 1
        elif self.stage == 1:
            self.to_act = 0
        else:
            self.reward = 9 if card == 2 + self.secret else -1
            self.phase = DONE
        self.stage += 1

    def points(self):
        return [self.reward, 0]


class ProbePolicy:
    react = False

    def __call__(self, state):
        if state.to_act == 1:
            return 10 + state.secret if state.revealing else 10
        return state.legal_actions()[0]


@pytest.mark.parametrize('revealing,expected', [(True, 0), (False, 1)])
def test_probe_is_valuable_only_when_response_supplies_useful_information(
        monkeypatch, revealing, expected):
    monkeypatch.setattr('ohhell.bots.planning.state_card',
                        lambda state, *_: state.legal_actions()[0])
    tree = InformationTree([1, 0], exploration=10)
    for i in range(4000):
        tree.run(ProbeGame(i % 2, revealing), ProbePolicy(), 0)
    result = tree.result()
    assert max(result['visits'], key=result['visits'].get) == expected
    assert result['deep_visits'] > 0
    # Identical public responses share a node, even across different secrets.
    probe = tree.root.children[0]
    assert set(probe.children) == ({10, 11} if revealing else {10})
    if revealing:
        assert probe.children[10].children[2].visits > probe.children[10].children[3].visits
        assert probe.children[11].children[3].visits > probe.children[11].children[2].visits


def playing_obs(size=6):
    hands, trump = deal_round(random.Random(7), 4, size)
    state = RoundState(Rules(4, size), hands, trump, 3)
    while state.phase == BIDDING:
        state.apply(state.legal_actions()[0])
    return state.observation(state.to_act)


@pytest.mark.parametrize('fast', [False, True])
def test_all_legal_cards_receive_real_visits_and_budget_is_shared(fast):
    if fast and not native.available:
        pytest.skip('core not built')
    obs = playing_obs(size=8)
    bot = PlanningBot(fast=fast, worlds=20, tree_batch=3, endgame_cards=0)
    card = bot.play(obs)
    result = bot.last_search
    assert card in obs.legal_actions()
    assert set(result['visits']) == set(obs.legal_actions())
    assert len(result['visits']) > 4  # no top-four model shortlist
    assert min(result['visits'].values()) > 0
    assert sum(result['visits'].values()) == result['iterations'] == 20 * len(result['visits'])
    assert result['deep_visits'] > 0
    assert result['nodes'] <= 32768


def test_expired_deadline_uses_model_order_without_sampling(monkeypatch):
    obs = playing_obs()
    bot = PlanningBot()
    target = obs.legal_actions()[-1]
    bot.models = SimpleNamespace(card_values=lambda *args: [
        float(c == target) for c in args[-1]])
    bot.deadline = time.perf_counter() - 1
    bot.simulator = SimpleNamespace(plan_cards=lambda *_: pytest.fail('expired turn'))
    assert bot.play(obs) == target
    assert bot.last_search['iterations'] == 0


def test_python_search_checks_deadline_between_completed_simulations(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr('ohhell.bots.planning.time',
                        SimpleNamespace(perf_counter=lambda: clock[0]))
    bot = PlanningBot(fast=False, tree_batch=8, endgame_cards=0)
    obs = playing_obs()
    hands = obs.determinize(deal_round(random.Random(7), 4, 6)[0]).hands
    kinds = bot.belief._kinds(obs, bot.rng)
    bot.belief.worlds = lambda *args: [(hands, kinds)] * args[1]
    starts = []
    original = InformationTree.run

    def run(tree, *args):
        starts.append(clock[0])
        original(tree, *args)
        clock[0] += .00075

    monkeypatch.setattr(InformationTree, 'run', run)
    bot.deadline = .002
    assert bot.play(obs) in obs.legal_actions()
    assert starts == [0, .00075, .0015]
    assert bot.last_search['iterations'] == 3


@pytest.mark.parametrize('fast', [False, True])
def test_bidding_and_exact_endgames_preserve_v1(fast):
    if fast and not native.available:
        pytest.skip('core not built')
    kwargs = dict(seed=8, fast=fast, field='previous', worlds=8, bid_worlds=8)
    current, candidate = HybridBot(**kwargs), PlanningBot(**kwargs)
    hands, trump = deal_round(random.Random(5), 4, 2)
    state = RoundState(Rules(4, 2), hands, trump, 3)
    assert candidate.bid(state.observation(0)) == current.bid(state.observation(0))
    while state.phase == BIDDING:
        state.apply(state.legal_actions()[0])
    obs = state.observation(state.to_act)
    assert candidate.play(obs) == current.play(obs)
    assert candidate.last_search is None


@pytest.mark.parametrize('fast', [False, True])
def test_complete_down_blind_up_games(fast):
    if fast and not native.available:
        pytest.skip('core not built')
    spec = f'planning:field=previous,worlds=4,bid_worlds=3,fast={int(fast)},endgame_cards=0'
    totals, _, _ = play_game([make_bot(spec, s) for s in range(4)], Rules(4, 3), 19)
    assert len(totals) == 4
    assert DEFAULT_SPEC == 'hybrid:field=previous,time_ms=2'


def test_tree_capacity_exhaustion_finishes_with_rollout(monkeypatch):
    monkeypatch.setattr('ohhell.bots.planning.state_card',
                        lambda state, *_: state.legal_actions()[0])
    tree = InformationTree([0, 1], max_nodes=1)
    tree.run(ProbeGame(0), ProbePolicy(), 0)
    assert tree.iterations == 1
    assert tree.nodes == 1


@pytest.mark.parametrize('settings', [dict(exploration=0), dict(exploration=float('nan')),
                                     dict(tree_batch=0), dict(tree_batch=1025),
                                     dict(tree_batch=2.5), dict(worlds=0)])
def test_invalid_search_settings(settings):
    with pytest.raises(ValueError):
        PlanningBot(**settings)


def test_tournament_fingerprints_planning_checkpoints(tmp_path):
    weights = tmp_path / 'weights'
    weights.mkdir()
    (weights / 'bid.npz').write_bytes(b'bid')
    (weights / 'play.npz').write_bytes(b'play')
    args = SimpleNamespace(seed=1, max_sets=130, smallest_gain=1, time_ms=2, resume=False)
    manifest = tmp_path / 'run.meta'
    specs = [f'planning:models={weights}']
    check_run_config(manifest, args, specs)
    args.resume = True
    check_run_config(manifest, args, specs)
    (weights / 'play.npz').write_bytes(b'changed')
    with pytest.raises(SystemExit, match='different rules, models or settings'):
        check_run_config(manifest, args, specs)
