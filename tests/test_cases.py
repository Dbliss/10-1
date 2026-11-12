"""Case-bank isolation, counterfactual labels, privacy and training exports."""

import json

import numpy as np
import pytest

from ohhell.engine import BIDDING, DONE
from ohhell.learn.cases import build_bank, evaluate, export_rows, label_case, restore


@pytest.fixture(scope='module')
def bank():
    return list(build_bank(seed=17, repeats=2, max_hand=2, players=(2, 4, 8)))


def test_saved_cases_replay_legally_and_cover_required_dimensions(bank):
    assert bank == list(build_bank(seed=17, repeats=2, max_hand=2, players=(2, 4, 8)))
    assert len({c['id'] for c in bank}) == len(bank)
    tags = {tag for case in bank for tag in case['tags']}
    assert {'hand=0', 'hand=1', 'hand=2', 'players=2', 'players=4', 'players=8',
            'target_met', 'overshot', 'unreachable', 'early', 'late', 'choice', 'forced'} <= tags
    splits = {}
    for case in bank:
        state = restore(case)
        assert state.phase != DONE
        assert state.legal_actions()
        assert (state.phase == BIDDING) == (case['part'] == 'bid')
        assert splits.setdefault(case['deal_id'], case['split']) == case['split']
        if case['blind']:
            assert state.observation(state.to_act).hand == 0


def test_every_forced_action_has_exact_one_card_outcome(bank):
    case = next(c for c in bank if c['part'] == 'play' and not c['blind']
                and 'hand=1' in c['tags'])
    state = restore(case)
    row = label_case(case, opponents='heuristic')
    assert [o['action'] for o in row['outcomes']] == state.legal_actions()
    for outcome in row['outcomes']:
        trial = restore(case)
        trial.apply(outcome['action'])
        while trial.phase != DONE:
            trial.apply(trial.legal_actions()[0])
        assert outcome['score'] == trial.points()[state.to_act]
        assert outcome['tricks'] == [trial.tricks_won[state.to_act]]


def test_label_bids_reacts_legally_and_repeats_deterministically(bank):
    case = next(c for c in bank if c['part'] == 'bid' and len(c['actions']) == 0)
    row = label_case(case, samples=2)
    assert row == label_case(case, samples=2)
    assert {o['action'] for o in row['outcomes']} == set(restore(case).legal_actions())
    assert all(len(o['tricks']) == 2 for o in row['outcomes'])


def test_zero_bid_tactical_case_has_distinct_known_card_values():
    # Leading the low club loses both tricks. Leading the ace of spades
    # necessarily wins the first. The heuristic opponent discards its club,
    # leaving us to win the second too. Hearts are trump; neither holds one.
    case = {'id': 'zero-bid-tactic', 'deal_id': 'tactic', 'seed': 1,
            'split': 'test', 'profile': 'zero', 'part': 'play',
            'rules': {'num_players': 2, 'max_hand': 2},
            'hands': [(1 << 0) | (1 << 51), (1 << 1) | (1 << 13)],
            'trump': 26, 'dealer': 1, 'blind': False, 'actions': [0, 0], 'tags': []}
    row = label_case(case, opponents='heuristic')
    assert {o['action']: o['score'] for o in row['outcomes']} == {0: 10, 51: -20}


def test_completed_manifest_rejects_changed_or_partial_data(tmp_path):
    from ohhell.learn.cases import checked_metadata, file_hash, write_json
    path = tmp_path / 'bank.jsonl'
    path.write_text('{}\n', encoding='utf-8')
    with pytest.raises(FileNotFoundError):
        checked_metadata(path)
    write_json(str(path) + '.meta.json', {'version': 1, 'sha256': file_hash(path)})
    assert checked_metadata(path)['version'] == 1
    path.write_text('{"changed":true}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='fingerprint'):
        checked_metadata(path)


def test_bots_receive_only_observations_including_blind(monkeypatch, bank):
    from ohhell.engine import Observation
    import ohhell.learn.cases as module

    class Observer:
        def act(self, obs):
            assert isinstance(obs, Observation)
            assert not hasattr(obs, 'hands')
            if obs.blind:
                assert obs.hand == 0
            return obs.legal_actions()[0]

    monkeypatch.setattr(module, 'make_bot', lambda *args: Observer())
    case = next(c for c in bank if c['blind'] and c['part'] == 'bid')
    row = label_case(case)
    result = evaluate([row], 'heuristic', 'heuristic', 'bid', case['split'])
    assert result['summary']['all']['deal_mean_edge'] == 0


@pytest.mark.parametrize('part', ['bid', 'play'])
def test_export_matches_model_inputs_and_keeps_deal_ids(tmp_path, bank, part):
    from ohhell.learn.features import BID_INPUTS, PLAY_INPUTS
    selected = [c for c in bank if c['part'] == part and c['split'] == 'train'][:20]
    rows = [label_case(c) for c in selected]
    path = tmp_path / 'rows.npz'
    report = export_rows(rows, part, 'train', path)
    with np.load(path) as data:
        assert data['x'].shape == (report['rows'], BID_INPUTS if part == 'bid' else PLAY_INPUTS)
        assert len(data['y']) == len(data['deal_id']) == report['rows']
        assert set(data['deal_id']) <= {c['deal_id'] for c in selected if not c['blind']}
        assert np.isfinite(data['x']).all() and np.isfinite(data['y']).all()


def test_evaluation_identical_bot_has_zero_edge_and_rejects_missing_labels(bank):
    cases = [c for c in bank if c['part'] == 'bid' and c['split'] == 'train'][:10]
    rows = [label_case(c) for c in cases]
    report = evaluate(rows, 'heuristic', 'heuristic', 'bid', 'train')
    assert report['summary']['all']['deal_mean_edge'] == 0
    assert len(report['decisions']) == len(rows)
    damaged = json.loads(json.dumps(rows[0]))
    damaged['outcomes'].pop()
    with pytest.raises(ValueError, match='exactly the legal'):
        evaluate([damaged], 'heuristic', 'heuristic', 'bid', 'train')
