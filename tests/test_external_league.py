"""Cross-project adapters: real upstream assets, public inputs, common rules."""

import ctypes
import json
import math
import random
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ohhell.arena import deal_round
from ohhell.bots.base import Bot
from ohhell.bots.external import (CampbellBot, DiminishingBot, DRLRuleBot, campbell_network,
                                  diminishing_inputs, diminishing_library, network_probability)
from ohhell.cards import EFFECTIVE_SUIT, POWER, beats, cards_in, mask_of
from ohhell.engine import BIDDING, PLAYING, RoundState
from ohhell.external_league import Board, DEFAULT_ROSTER, play_set, serve
from ohhell.external_setup import ASSETS, verify_assets
from ohhell.rules import Rules

needs_assets = pytest.mark.skipif(not (ASSETS / 'manifest.json').exists(),
                                 reason='run python -m ohhell.external_league --prepare for upstream integration tests')


def random_position(seed=17, n=5, size=5, plays=0):
    rng = random.Random(seed)
    hands, trump = deal_round(rng, n, size)
    state = RoundState(Rules(n, size), hands, trump, n - 1)
    while state.phase == BIDDING:
        state.apply(rng.choice(state.legal_actions()))
    for _ in range(plays):
        state.apply(rng.choice(state.legal_actions()))
    return state


@needs_assets
@pytest.mark.parametrize('n', range(2, 9))
def test_original_campbell_shapes_and_scalar_inference_match(n):
    rng = random.Random(n)
    state = random_position(n, n, min(5, 51 // n))
    bot = CampbellBot()
    obs = state.observation(state.to_act)
    cards = obs.legal_actions()
    for name, rows in [('ovl', [bot._ovl_features(obs, c) for c in cards]),
                       ('ivl', [bot._ivl_features(obs, c) for c in cards])]:
        network = campbell_network(n, name)
        assert len(rows[0]) == network[0].shape[1]
        actual = network_probability(network, rows)
        w0, b0, w1, b1 = network
        expected = []
        for row in rows:
            hidden = [max(0, bias + sum(w * x for w, x in zip(weights, row)))
                      for weights, bias in zip(w0, b0)]
            value = b1[0] + sum(w * h for w, h in zip(w1[0], hidden))
            expected.append(1 / (1 + math.exp(-value)))
        assert actual == pytest.approx(expected, abs=1e-12)


@needs_assets
def test_upstream_rank_grid_matches_bowers_in_every_suit():
    rng = random.Random(33)
    library = diminishing_library()
    for trump in range(4):
        for n in (3, 4, 5, 8):
            for _ in range(60):
                cards = rng.sample(range(52), n)
                grid = [v for c in cards for v in (EFFECTIVE_SUIT[trump][c], POWER[trump][c])]
                actual = library.league_dim_winner(n, trump, (ctypes.c_int * len(grid))(*grid))
                winner = 0
                for i in range(1, n):
                    if beats(cards[i], cards[winner], trump):
                        winner = i
                assert actual == winner


@needs_assets
def test_actual_upstream_source_is_pinned_and_present():
    manifest = verify_assets()
    assert len(manifest['files']) == 26
    assert manifest['revisions']['diminishing'][1] == '305af70fe0c5341aeb70d5e9d2411bb57c4e7c67'
    from ohhell.bots.external import drl_module
    assert drl_module().RuleBasedPlayer.__module__ == 'upstream_drl_ohhell_agent'


@needs_assets
def test_adapters_never_receive_saved_opponent_hands():
    state = random_position(12, 5, 5)
    obs = state.observation(state.to_act)
    before, reverse = diminishing_inputs(obs, [0] * 5)
    own = obs.seat
    others = [s for s in range(5) if s != own]
    a, b = others[:2]
    state.hands[a], state.hands[b] = state.hands[b], state.hands[a]
    changed = state.observation(own)
    after, new_reverse = diminishing_inputs(changed, [0] * 5)
    assert before == after and reverse == new_reverse
    # Unknown cards contain no other seat identities.
    assert set(before[0]) <= {-2, -1, own}
    bot = CampbellBot()
    for card in obs.legal_actions():
        assert bot._ovl_features(obs, card) == bot._ovl_features(changed, card)


@needs_assets
@pytest.mark.parametrize('algorithm', ['mc', 'mcts'])
def test_timed_upstream_bid_uses_completed_simulations(algorithm):
    # Clubs right/left/ace win every trick in every world. Dividing by the
    # original fixed 100000 budget after a timed cut-off would incorrectly bid 0.
    hands = [mask_of([9, 48, 12]), mask_of([13, 14, 15]), mask_of([26, 27, 28])]
    state = RoundState(Rules(3, 3), hands, 0, 2)
    bot = DiminishingBot(17, algorithm, time_ms=2, budget=100000)
    assert bot.act(state.observation(0)) == 3


@needs_assets
def test_upstream_failure_cannot_silently_fall_back(monkeypatch):
    state = random_position()
    obs = state.observation(state.to_act)
    monkeypatch.setattr('ohhell.bots.external.diminishing_library', lambda: SimpleNamespace(
        league_dim_choose=lambda *args: -999))
    with pytest.raises(RuntimeError, match='could not produce'):
        DiminishingBot().act(obs)


@needs_assets
@pytest.mark.parametrize('algorithm', ['mc', 'mcts'])
def test_upstream_score_heuristic_stays_finite_with_large_public_score_gaps(algorithm):
    for seed in range(5):
        state = random_position(seed + 70, 5, 3)
        obs = state.observation(state.to_act)
        bot = DiminishingBot(seed, algorithm, time_ms=2)
        bot.scores = [-1000, 1000, 0, 500, -500]
        assert bot.act(obs) in obs.legal_actions()


@needs_assets
def test_drl_dealer_adjustment_is_explicit_and_counted():
    hands, trump = deal_round(random.Random(4), 4, 4)
    state = RoundState(Rules(4, 4), hands, trump, 3)
    for _ in range(3):
        state.apply(0)
    obs = state.observation(3)
    bot = DRLRuleBot()
    bot._prepare(obs).bidding = lambda *args: 4
    assert bot.bid(obs) == 3
    assert bot.legal_bid_adjustments == 1


@needs_assets
@pytest.mark.parametrize('n', [3, 4, 5])
def test_real_upstream_entries_complete_rotated_blind_inclusive_games(n):
    row = play_set((0, 1234, DEFAULT_ROSTER[:n], 3, 1))
    assert len(row['scores']) == n
    assert len(row['rounds']) == n
    for rotation, score in enumerate(row['scores']):
        assert len(row['rounds'][rotation]) == 7
        assert score == [sum(r[i] for r in row['rounds'][rotation]) for i in range(n)]


def test_each_player_gets_every_seat_on_identical_deals(monkeypatch):
    roster = ('a', 'b', 'c')
    seen = {entry: [] for entry in roster}
    class Spy(Bot):
        def __init__(self, entry):
            super().__init__()
            self.entry = entry
            self.first = True
        def act(self, obs):
            if self.first:
                seen[self.entry].append((obs.seat, obs.hand, obs.trump_card))
                self.first = False
            return obs.legal_actions()[0]
    monkeypatch.setattr('ohhell.external_league.create_bot', lambda entry, *args: Spy(entry))
    row = play_set((0, 1234, roster, 3, 2))
    assert sorted(seen['a']) == sorted(seen['b']) == sorted(seen['c'])
    assert [s for s, _, _ in sorted(seen['a'])] == [0, 1, 2]


def test_board_intervals_cluster_by_deal_set_not_rotations():
    config = {'label': 'test', 'roster': ['v2', 'campbell'], 'sets': 2, 'rules': {}, 'time_ms': 2}
    board = Board(config)
    for i, score in enumerate((10, 14)):
        board.add({'index': i, 'seed': i, 'scores': [[score, 0]] * 2,
                   'clocks': [[0.002, 1, 0.002], [0.001, 1, 0.001]],
                   'bid_adjustments': [0, 0], 'feature_saturations': [0, 0]})
    report = board.snapshot()
    assert report['games'] == 4
    assert report['standings'][0]['mean'] == 12
    assert report['standings'][0]['margin'] == pytest.approx(3.92)
    assert report['pairs'][0]['edge'] == 12
    assert report['pairs'][0]['margin'] == pytest.approx(3.92)
    assert report['standings'][0]['win_share'] == 1


def test_dashboard_exposes_readonly_stats_and_rules():
    from urllib.request import urlopen
    board = Board({'label': 'test', 'roster': ['v2', 'campbell'], 'sets': 1,
                   'rules': {'max_hand': 3}, 'time_ms': 2})
    server = serve(board, 0)
    try:
        base = f'http://localhost:{server.server_address[1]}'
        stats = json.load(urlopen(base + '/stats'))
        assert stats['rules']['max_hand'] == 3
        assert stats['sets'] == 0
        assert b'Origins and rule adjustments' in urlopen(base).read()
    finally:
        server.shutdown()
        server.server_close()


@needs_assets
def test_captured_mcts_pruning_position_does_not_corrupt_heap():
    # Run in its own process: the old implementation corrupts the native heap
    # and terminates Python on this real, late-round public position.
    fixture = Path(__file__).parent / 'fixtures' / 'external_mcts_pruning.json'
    program = '''
import json, sys
from ohhell.bots.external import DiminishingBot
from ohhell.engine import Observation
from ohhell.rules import Rules
r = json.load(open(sys.argv[1]))
fields = r['observation']
obs = Observation(rules=Rules(fields['n'], 10), **fields)
for budget in (100, 1000, 10000, 100000):
    bot = DiminishingBot(r['bot']['seed'], 'mcts', time_ms=0, budget=budget)
    bot.calls = r['bot']['calls']
    bot.scores = r['arrays'][5]
    chosen = bot.act(obs)
    assert chosen == 38 and chosen in obs.legal_actions(), chosen
print('All four pruning regressions passed')
'''
    result = subprocess.run([sys.executable, '-c', program, str(fixture.resolve())],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'All four pruning regressions passed' in result.stdout


def test_native_failure_report_names_public_seed_context(tmp_path):
    from ohhell.external_league import native_failure_message
    (tmp_path / '123.fault.log').write_text('Windows fatal exception: heap corruption')
    (tmp_path / '123.context.json').write_text(json.dumps(
        {'pid': 123, 'index': 10, 'seed': 202611140, 'rotation': 2, 'hand': 10}))
    message = native_failure_message(tmp_path)
    assert 'set 10, seed 202611140, rotation 2, hand 10' in message
    assert 'Completed sets are preserved' in message
    assert 'no bot was substituted' in message
