"""Fitted bid shares: the table option, the fit, and the timed advisor."""

import json
import random
from types import SimpleNamespace

import pytest

import ohhell.tournament as t
from game import Game
from ohhell.arena import deal_round
from ohhell.bots import DEFAULT_SPEC, make_bot
from ohhell.bots.learned import HybridBot, load_bid_table
from ohhell.engine import RoundState
from ohhell.learn import bid_blend_fit as fit
from ohhell.rules import Rules


def bid_obs(hand_size, players=4, seed=7):
    hands, trump = deal_round(random.Random(seed), players, hand_size)
    state = RoundState(Rules(players, hand_size), hands, trump, players - 1)
    return state.observation(state.to_act)


def write_table(path, share):
    path.write_text(json.dumps({'share': share}))
    return str(path).replace('\\', '/')


def test_table_sets_the_share_and_other_cells_keep_the_default(tmp_path):
    bot = HybridBot(bid_table=write_table(tmp_path / 't.json', {'3,4': 0.2, '1,4': 0.0}))
    assert bot.bid_share(3, 4) == 0.2
    assert bot.bid_share(1, 4) == 0.0
    assert bot.bid_share(3, 5) == 0.5  # no entry: v2's share
    assert HybridBot().bid_share(3, 4) == 0.5


def test_table_share_blends_the_two_predictions(monkeypatch, tmp_path):
    obs = bid_obs(3)
    bot = HybridBot(bid_table=write_table(tmp_path / 't.json', {'3,4': 0.2}))
    bot.models = SimpleNamespace(bid_distributions=lambda *args: {
        b: [0.0, 1.0, 0.0, 0.0] for b in args[-1]})
    monkeypatch.setattr(
        'ohhell.bots.simulation.SimulationBot.analyse_bids',
        lambda self, o: SimpleNamespace(distributions={0: [1.0, 0.0, 0.0, 0.0]}))
    assert bot.analyse_bids(obs).distributions[0][:2] == pytest.approx([0.8, 0.2])


def test_zero_share_does_not_ask_the_network(tmp_path):
    bot = HybridBot(seed=3, bid_worlds=8, field='previous',
                    bid_table=write_table(tmp_path / 't.json', {'3,4': 0.0}))

    def fail(*args):
        raise AssertionError('the bid network was consulted')
    bot.models = SimpleNamespace(bid_distributions=fail)
    assert bot.analyse_bids(bid_obs(3)).values


def test_table_rejects_shares_outside_zero_to_one(tmp_path):
    with pytest.raises(ValueError):
        load_bid_table(write_table(tmp_path / 'bad.json', {'3,4': 1.5}))


def test_spec_loads_a_table_and_plays_a_bid(tmp_path):
    path = write_table(tmp_path / 't.json', {'5,4': 0.3})
    bot = make_bot(f'hybrid:field=previous,time_ms=1,bid_table={path}', 1)
    obs = bid_obs(5)
    assert bot.act(obs) in obs.legal_actions()


def test_tournament_fingerprints_the_table(tmp_path):
    path = write_table(tmp_path / 't.json', {'5,4': 0.3})
    args = SimpleNamespace(seed=1, max_sets=130, smallest_gain=1, time_ms=10, resume=False)
    meta = tmp_path / 'run.meta'
    spec = f'hybrid:field=previous,bid_table={path}'
    t.check_run_config(meta, args, [spec])
    assert any(name.endswith('t.json') for name in json.loads(meta.read_text())['weights'])
    (tmp_path / 't.json').write_text(json.dumps({'share': {'5,4': 0.4}}))
    args.resume = True
    with pytest.raises(SystemExit):
        t.check_run_config(meta, args, [spec])


def record(cards, n, right, number):
    """Two options. ``right`` names the source that knows what happens:
    one trick. The other source expects none."""
    knows, wrong = [0.1, 0.9], [0.9, 0.1]
    sim, net = (knows, wrong) if right == 'sim' else (wrong, knows)
    return {'round': number, 'n': n, 'cards': cards, 'before': 0, 'samples': 100,
            'options': [0, 1], 'sim': [sim, sim], 'net': [net, net],
            'tricks': [1, 1], 'points': [-10, 11]}


def test_choice_follows_the_share():
    r = record(1, 4, 'sim', 0)
    assert r['options'][fit.choice(r, 0.0)] == 1  # the simulation expects a trick
    assert r['options'][fit.choice(r, 1.0)] == 0


def test_log_loss_prefers_the_source_that_was_right():
    r = record(1, 4, 'net', 0)
    assert fit.log_loss(r, 1.0) < fit.log_loss(r, 0.5) < fit.log_loss(r, 0.0)


def test_fit_gives_each_cell_its_own_share():
    records = ([record(1, 4, 'sim', i) for i in range(300)]
               + [record(9, 4, 'net', 1000 + i) for i in range(300)])
    table = fit.fit_table(records, shrink=10)
    assert table[1, 4] < 0.2 and table[9, 4] > 0.8
    gain, _ = fit.paired(records, fit.table_share(table), lambda r: 0.5)
    assert gain >= 0


def test_sparse_cells_borrow_from_their_row_and_column():
    records = ([record(2, 4, 'sim', i) for i in range(300)]
               + [record(3, 5, 'sim', 1000 + i) for i in range(300)]
               + [record(2, 5, 'net', 500)])  # one position against its neighbours
    assert fit.fit_table(records, shrink=200)[2, 5] < 0.5
    assert fit.fit_table(records, shrink=0)[2, 5] == 1.0


def test_collected_positions_have_an_outcome_for_every_option():
    rows = fit._rounds((7_100_000, 6, 1))
    assert rows
    for row in rows:
        count = len(row['options'])
        assert count >= 2 and row['samples'] > 0
        assert len(row['sim']) == len(row['net']) == len(row['tricks']) == len(row['points']) == count
        for sim, net, tricks in zip(row['sim'], row['net'], row['tricks']):
            assert sum(sim) == pytest.approx(1) and sum(net) == pytest.approx(1, abs=1e-5)
            assert 0 <= tricks <= row['cards']


def test_fit_report_runs_and_writes_a_loadable_table(tmp_path, capsys):
    records = tmp_path / 'r.jsonl'
    rows = ([record(1, 4, 'sim', i) for i in range(60)]
            + [record(9, 4, 'net', 100 + i) for i in range(60)])
    records.write_text('\n'.join(json.dumps(r) for r in [{'meta': {'time_ms': 10}}] + rows))
    out = tmp_path / 'table.json'
    fit.main(['fit', str(records), '--shrink', '5', '--write', str(out)])
    assert 'Fitted share table' in capsys.readouterr().out
    table = load_bid_table(str(out).replace('\\', '/'))
    assert table[1, 4] < 0.5 < table[9, 4]


def test_game_bots_and_advisor_think_longer_than_a_tournament_turn():
    assert DEFAULT_SPEC.endswith('time_ms=2')  # the tournament default is unchanged
    assert make_bot(Game.BOT_SPEC).time_limit == Game.BOT_TIME_MS / 1000 >= 0.1
    assert make_bot(Game.ADVISOR_SPEC).time_limit == Game.ADVISOR_TIME_MS / 1000 >= 0.5


def test_started_clock_keeps_sampling_until_the_time_is_up():
    obs = bid_obs(5)
    once = make_bot('hybrid:field=previous,bid_worlds=16,time_ms=30', 1)
    once.analyse_bids(obs)  # no clock: one block of 16, as the advisor used to
    assert once.simulator.last_bid_samples == 16
    timed = make_bot('hybrid:field=previous,bid_worlds=16,time_ms=30', 1)
    timed.start_clock()
    timed.analyse_bids(obs)
    assert timed.simulator.last_bid_samples > 16 * 4


def test_waiting_bid_helper_requests_give_way_to_a_newer_one(monkeypatch):
    import threading

    import app_api
    import website
    started, release = threading.Event(), threading.Event()
    calls = []

    def slow(*args, **kwargs):
        calls.append(args)
        started.set()
        release.wait(5)
        return {'recommended': 0}
    monkeypatch.setattr(app_api, 'recommend', slow)
    results = {}

    def ask(name):
        results[name] = website.app.test_client().post('/api/bid-helper', json={}).status_code
    first = threading.Thread(target=ask, args=('first',))
    first.start()
    assert started.wait(5)
    second = threading.Thread(target=ask, args=('second',))
    second.start()
    while website.api.helper_requests < 2:
        pass
    third = threading.Thread(target=ask, args=('third',))
    third.start()
    while website.api.helper_requests < 3:
        pass
    release.set()
    for thread in (first, second, third):
        thread.join(5)
    assert results == {'first': 200, 'second': 409, 'third': 200}
    assert len(calls) == 2
