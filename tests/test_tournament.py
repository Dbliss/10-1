import io
import json
from collections import Counter

import numpy as np
import pytest

from ohhell import tournament as t
from ohhell.bots import VERSIONS, make_bot, version_title


def test_table_cycle_covers_three_to_eight_players():
    assert Counter(t.table_size(i) for i in range(len(t.TABLE_CYCLE))) == Counter(t.PLAYER_SPLIT)
    assert set(t.PLAYER_SPLIT) == {3, 4, 5, 6, 7, 8}


def test_hand_sizes_fit_every_table():
    assert t.rules_for(3).max_hand == 10
    assert t.rules_for(4).max_hand == 10
    assert t.rules_for(8).max_hand == 6


def test_each_check_is_balanced_across_table_sizes():
    every = 1300 // t.LOOKS
    counts = Counter(t.table_size(i) for i in range(every))
    assert counts == Counter({n: w * every // 13 for n, w in t.PLAYER_SPLIT.items()})


def test_lobby_is_labelled_with_versions():
    lobby = t.build_lobby('heuristic', VERSIONS[-1].spec, 2, 'My idea')
    names = [name for name, _, _ in lobby]
    assert names[0] == f'v{VERSIONS[-1].number + 1} My idea'
    assert names[1] == VERSIONS[-1].title
    assert not any('old' in name.lower().split() or 'new' in name.lower().split() for name in names)
    assert {'skilled', 'casual', 'novice'} <= {spec for _, _, spec in lobby}
    for _, _, spec in lobby[2:6]:
        assert 'confidence=' in spec


def test_unknown_baseline_is_not_given_a_version():
    assert version_title('random', 'Baseline') == 'Baseline'


def test_every_pool_member_can_be_built():
    for _, _, spec in t.HUMAN_LIKE:
        make_bot(spec, 1)


def pool_specs():
    return ['heuristic', 'heuristic'] + ['novice', 'casual', 'skilled', 'heuristic', 'random', 'mixed']


@pytest.mark.parametrize('players', [3, 4, 5, 6, 7, 8])
def test_set_seats_candidate_and_baseline_in_every_game(players):
    index = t.TABLE_CYCLE.index(players)
    _, members, games, clocks, rounds = t.play_set((index, 1234, pool_specs()))
    assert len(members) == len(set(members)) == players
    assert t.CANDIDATE in members and t.BASELINE in members
    assert len(games) == players and all(len(scores) == players for scores in games)
    assert len(clocks) == players
    sizes = t.rules_for(players).hand_sizes
    assert len(rounds) == players and all(len(game) == len(sizes) for game in rounds)
    for scores, game in zip(games, rounds):
        assert [sum(row[place] for row in game) for place in range(players)] == scores


def test_round_breakdown_groups_by_hand_size():
    members = [t.CANDIDATE, t.BASELINE, 2]
    sizes = t.rules_for(3).hand_sizes
    # candidate takes every round, baseline is second, the third player scores nothing
    game = [[11, 1, 0] for _ in sizes]
    stats = t.round_stats(members, [game, game, game])
    assert sorted(stats) == sorted(set(sizes))
    means, wins = stats[sizes[0]]
    assert means == [11, 1, 0] and wins == [1.0, 0.0, 0.0]

    b = t.Scoreboard(Args(), [('a', 'candidate', 'x'), ('b', 'current', 'y'), ('c', 'novice', 'z')], [])
    b.add_set(members, [[11 * len(sizes), len(sizes), 0]] * 3, [[0.0, 0, 0.0]] * 3,
              [game, game, game])
    snap = {r['cards']: r for r in b.snapshot()['rounds']}
    assert snap[0]['edge'] == 10 and snap[0]['win_edge'] == 1.0
    assert snap[sizes[0]]['mean'] == [11, 1, 0]
    assert [r['cards'] for r in b.snapshot()['rounds']] == sorted(set(sizes), reverse=True)


def test_same_set_is_repeatable():
    first = t.play_set((0, 77, pool_specs()))
    second = t.play_set((0, 77, pool_specs()))
    assert first[1:3] == second[1:3]


def test_win_shares_split_ties_and_sum_to_one():
    shares = t.win_shares([[10, 10, 0], [0, 5, 20]])
    assert shares == pytest.approx([0.25, 0.25, 0.5])
    assert sum(shares) == pytest.approx(1)


def test_planned_margin_shrinks_to_the_final_value():
    assert t.planned_margin(10, 1000, 1000) == pytest.approx(t.FINAL_Z * 10 / np.sqrt(1000))
    assert t.planned_margin(10, 100, 1000) == pytest.approx(10 * t.planned_margin(10, 1000, 1000))


def test_decide_only_rules_when_certain():
    assert t.decide(5, 15, 1000, 1, 1000) == 'better'
    assert t.decide(-5, 15, 1000, 1, 1000) == 'worse'
    assert t.decide(0, 15, 1000, 3, 1000) == 'no gain'
    assert t.decide(0, 15, 1000, 0.5, 1000) is None
    assert t.decide(100, 15, 10, 1, 1000) is None


def test_ten_checks_keep_the_error_rate_near_five_percent():
    rng = np.random.default_rng(5)
    sets, runs = 130, 40000
    totals = rng.standard_normal((runs, t.LOOKS)).cumsum(axis=1) * np.sqrt(sets)
    seen = sets * np.arange(1, t.LOOKS + 1)
    margin = t.FINAL_Z * np.sqrt(seen[-1]) * np.ones(t.LOOKS)  # total-score form of the margin
    wrong = (np.abs(totals) > margin).any(axis=1).mean()
    assert 0.04 < wrong < 0.06


def test_repeat_needs_a_clear_win():
    assert t.confirmed(2, 15, 1000)
    assert not t.confirmed(0.1, 15, 1000)
    assert not t.confirmed(-2, 15, 1000)
    assert not t.confirmed(2, None, 1)


class Args:
    max_sets = 130
    smallest_gain = 1.0
    label = 'x'
    seed = 0
    time_ms = 2
    reference = False


def board():
    lobby = t.build_lobby('heuristic', 'heuristic', 2, 'x')
    return t.Scoreboard(Args(), lobby, [])


def test_reference_candidate_runs_untimed_and_others_keep_the_limit():
    lobby = t.build_lobby('simulation:worlds=8', VERSIONS[-1].spec, 2, 'Untimed', reference=True)
    name, role, spec = lobby[0]
    assert spec == 'simulation:worlds=8' and role == 'candidate'
    assert 'reference' in name and not name.startswith('v')
    for _, _, other in lobby[1:6]:
        assert other.endswith(',time_ms=2')
    make_bot(spec, 1)


def test_reference_run_never_decides_or_promotes():
    class Reference(Args):
        reference = True

    b = t.Scoreboard(Reference(), t.build_lobby('heuristic', 'heuristic', 2, 'x', True), [])
    members = [t.CANDIDATE, t.BASELINE, 2]
    for i in range(130):  # a clear, consistent lead that would pass a normal check
        assert b.add_set(members, [[20 + i % 3, 0, 0]] * 3, [[0.0, 0, 0.0]] * 3) is None
    b.finish()
    snap = b.snapshot()
    assert b.verdict == 'reference' and snap['reference'] is True
    edge = b.edges[t.CANDIDATE]
    assert snap['players'][0]['edge_interval'] == pytest.approx(edge.interval)


def test_reference_flag_is_saved_without_changing_older_run_settings(tmp_path):
    from types import SimpleNamespace
    args = SimpleNamespace(seed=1, max_sets=130, smallest_gain=1, time_ms=2, resume=False)
    plain, ref = tmp_path / 'plain.meta', tmp_path / 'ref.meta'
    t.check_run_config(plain, args, ['heuristic'])
    assert 'reference' not in json.loads(plain.read_text())
    args.reference = True
    t.check_run_config(ref, args, ['heuristic'])
    assert json.loads(ref.read_text())['reference'] is True
    args.resume = True
    with pytest.raises(SystemExit):
        t.check_run_config(plain, args, ['heuristic'])


def test_board_tracks_tables_and_only_decides_at_checks():
    b = board()
    members = [t.CANDIDATE, t.BASELINE, 2]
    games = [[10, 0, 5], [12, 1, 4], [11, 0, 6]]
    clocks = [[0.0, 0, 0.0] for _ in members]
    for _ in range(64):
        assert b.add_set(members, games, clocks) is None
    # Identical sets have no spread, so no decision can be made.
    assert b.add_set(members, games, clocks) is None
    snap = b.snapshot()
    table = next(row for row in snap['tables'] if row['players'] == 3)
    assert table['sets'] == 65 and table['edge'] == pytest.approx(10.667, abs=0.01)
    assert snap['win_edge'] == pytest.approx(1.0)
    assert snap['players'][3]['mean'] is None  # never seated


def test_repeat_stage_decides_the_final_verdict():
    b = board()
    members = [t.CANDIDATE, t.BASELINE, 2]
    for i in range(13):
        b.add_set(members, [[i % 3, 0, 0]] * 3, [[0.0, 0, 0.0]] * 3)
    b.decision = 'better'
    assert list(b.start_repeat()) == list(range(130, 143))
    for i in range(13):
        b.add_repeat(members, [[i % 3 - 1, 0, 0]] * 3, [[0.0, 0, 0.0]] * 3)
    b.finish()
    assert b.verdict == 'unconfirmed'


class SavedFile:
    name = 'old.jsonl'

    def __init__(self, record):
        self.text = json.dumps(record) + '\n'

    def exists(self):
        return True

    def open(self):
        return io.StringIO(self.text)


def test_old_results_cannot_be_resumed():
    with pytest.raises(SystemExit):
        t.load_previous(SavedFile([0, [[1] * 6] * 6, [[0, 0, 0]] * 6]))


def test_saved_results_are_read_by_index():
    record = [7, [0, 1, 2], [[1, 2, 3]] * 3, [[0, 0, 0]] * 3]
    assert t.load_previous(SavedFile(record)) == {7: record}


def test_history_ignores_smoke_reports_and_preserves_tournament_results(tmp_path, monkeypatch):
    monkeypatch.setattr(t, 'GATES', tmp_path)
    summary = {'label': 'Completed test', 'verdict': 'inconclusive', 'sets': 130,
               'players': [{'edge': 1.25, 'edge_interval': 2.5}]}
    (tmp_path / 'tournament.json').write_text(json.dumps(summary))
    smoke = tmp_path / 'Planning_information_smoke_2026-11-10.json'
    smoke_text = json.dumps({'purpose': 'Smoke check', 'players': 4,
                             'report': {'edge': 2.8, 'interval': 7.8}})
    smoke.write_text(smoke_text)
    assert t.load_history() == [{'label': 'Completed test', 'verdict': 'inconclusive',
                                 'sets': 130, 'edge': 1.25, 'margin': 2.5}]
    assert smoke.read_text() == smoke_text


@pytest.mark.parametrize('data', ['{unfinished', 'null', '[]', '{"players": []}',
                                  '{"players": [4]}', '{"players": [{}]}',
                                  '{"label":"x","verdict":"inconclusive","sets":0,'
                                  '"players":[{"edge":null,"edge_interval":null}]}'])
def test_incompatible_history_cannot_prevent_startup(tmp_path, monkeypatch, data):
    monkeypatch.setattr(t, 'GATES', tmp_path)
    (tmp_path / 'report.json').write_text(data)
    assert t.load_history() == []


def test_resume_rejects_old_rules_and_changed_settings(tmp_path):
    from types import SimpleNamespace
    args = SimpleNamespace(seed=1, max_sets=130, smallest_gain=1, time_ms=2, resume=True)
    path = tmp_path / 'run.meta'
    with pytest.raises(SystemExit, match='different rules'):
        t.check_run_config(path, args, ['heuristic'])
    args.resume = False
    t.check_run_config(path, args, ['heuristic'])
    args.resume = True
    t.check_run_config(path, args, ['heuristic'])
    with pytest.raises(SystemExit, match='different rules'):
        t.check_run_config(path, args, ['random'])


def test_tournament_fingerprints_the_estimate_file(tmp_path):
    import json
    from types import SimpleNamespace
    from ohhell.estimate import COEFFICIENT_FILES
    from ohhell.tournament import check_run_config
    args = SimpleNamespace(seed=1, max_sets=130, smallest_gain=1, time_ms=10, resume=False)
    for version in (2, 3):
        manifest = tmp_path / f'run{version}.meta'
        check_run_config(manifest, args, [f'simulation:estimate={version}'])
        saved = json.loads(manifest.read_text())['weights']
        assert str(COEFFICIENT_FILES[version].resolve()) in saved


def test_candidate_alone_can_get_more_time():
    lobby = t.build_lobby(VERSIONS[-1].spec, VERSIONS[-1].spec, 2, 'x', candidate_time_ms=10)
    assert lobby[0][2].endswith(',time_ms=10')
    assert all(spec.endswith(',time_ms=2') for _, _, spec in lobby[1:6])
    assert make_bot(lobby[0][2]).time_limit == 0.01
    assert make_bot(lobby[1][2]).time_limit == 0.002
