"""Train on paired live/cheap points differences, then score frozen bid choices."""

import argparse
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import statistics
from types import SimpleNamespace

import numpy as np

from ..bid_rank import GAP_LIMIT, MIN_HAND, N_FEATURES, contenders, features, load_weights, pick
from ..duel import FIELD, REFERENCE_SPEC, draw_round, extend, rounds_per_game
from ..engine import NO_BID
from ..rules import Scoring
from ..tournament import check_run_config
from .bid_play_audit import _live_counts, _worlds, bid_position


def collect_position(job):
    seed, count = job
    if draw_round(seed, FIELD)[2] < MIN_HAND:
        return None
    prepared = bid_position(seed)
    if prepared is None:
        return None
    obs, bot, _, _ = prepared
    bot.simulator.bid_budget = 4096
    current = bot.analyse_bids(obs)
    best, bids = contenders(current.values)
    if len(bids) < 2:
        return None
    sample = _worlds(obs, bot.simulator, bids, count, seed)
    cheap = np.zeros((len(bids), obs.hand_size + 1))
    bot.simulator.score_bids(obs, *sample, cheap)
    live = _live_counts(obs, sample, extend(REFERENCE_SPEC, 'card_budget=128,endgame_budget=128'),
                        seed + (1 << 31), own_only=False)
    score = obs.rules.scoring.points
    before = sum(b != NO_BID for b in obs.bids)
    return dict(seed=seed, base=best, bids=bids, base_values=[current.values[b] for b in bids],
                features=[features(current.distributions[b], b, obs.n, before) for b in bids],
                corrections=[(1 - bot.bid_share(obs.hand_size, obs.n)) * float(row @ np.array(
                    [score(b, k) for k in range(obs.hand_size + 1)])) / count
                    for b, row in zip(bids, live - cheap)])


def fit(rows):
    x, y = [], []
    for row in rows:
        base = row['bids'].index(row['base'])
        vectors = np.array(row['features'])
        for i in range(len(vectors)):
            if i == base:
                continue
            gap = row['base_values'][base] - row['base_values'][i]
            weight = math.sqrt((1 - gap / (GAP_LIMIT + 1)) / (len(vectors) - 1))
            x.append(weight * (vectors[i] - vectors[base]))
            y.append(weight * (row['corrections'][i] - row['corrections'][base]))
    if not x:
        raise ValueError('Training requires at least one paired position')
    x, y = np.array(x), np.array(y)
    weights = np.linalg.solve(x.T @ x + 3.0 * np.eye(N_FEATURES), x.T @ y)
    return dict(weights=weights.tolist(), training_positions=len(rows), paired_comparisons=len(y))


def evaluate(rows, weights, historical):
    score = Scoring().points
    results = []
    for row in rows:
        bids = row['options'] if historical else row['bids']
        size = row['cards']
        players = row['n'] if historical else row['players']
        if historical:
            before = row['before']
            cheap, blend = row['sim'], 0.25
        else:
            prepared = bid_position(row['seed']) if size >= MIN_HAND else None
            before = sum(b != NO_BID for b in prepared[0].bids) if prepared else 0
            cheap, blend = row['cheap'], row['blend']
        odds = {b: [(1 - blend) * p + blend * q for p, q in zip(sim, net)]
                for b, sim, net in zip(bids, cheap, row['net'])}
        expected = {b: sum(p * score(b, k) for k, p in enumerate(dist)) for b, dist in odds.items()}
        base = max(expected, key=expected.get)
        chosen = pick(expected, odds, size, players, before, weights)
        values = row['points'] if historical else [sum(p * score(b, k) for k, p in enumerate(dist))
                      for b, dist in zip(bids, row['reference'])]
        delta = values[bids.index(chosen)] - values[bids.index(base)]
        results.append((size, chosen - base, delta, delta * rounds_per_game(players)))
    report = {}
    for name, subset in [('all', results), ('8_to_10_cards', [r for r in results if r[0] >= MIN_HAND])]:
        if not subset:
            continue
        edges = [r[3] for r in subset]
        changed = [r for r in subset if r[1]]
        report[name] = dict(positions=len(subset), changed=len(changed),
                            raised=sum(r[1] > 0 for r in changed),
                            points_per_changed_hand=statistics.mean(r[2] for r in changed) if changed else None,
                            points_per_game=statistics.mean(edges),
                            margin_95=1.96 * statistics.stdev(edges) / math.sqrt(len(edges)) if len(edges) > 1 else None)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    collect = actions.add_parser('collect')
    collect.add_argument('--seed', type=int, required=True)
    collect.add_argument('--draws', type=int, required=True)
    collect.add_argument('--worlds', type=int, required=True)
    collect.add_argument('--workers', type=int, required=True)
    train = actions.add_parser('fit')
    train.add_argument('source', type=Path)
    validate = actions.add_parser('evaluate')
    validate.add_argument('model')
    validate.add_argument('source', type=Path)
    validate.add_argument('--historical', action='store_true')
    for action in (collect, train, validate):
        action.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output already exists')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.action == 'collect':
        if min(args.draws, args.worlds, args.workers) <= 0:
            parser.error('counts must be positive')
        config = SimpleNamespace(seed=args.seed, max_sets=args.draws, smallest_gain=2, time_ms=0, resume=False)
        check_run_config(args.out.with_suffix('.meta'), config, [REFERENCE_SPEC, *(spec for spec, _ in FIELD)])
        jobs = ((args.seed + i, args.worlds) for i in range(args.draws))
        kept = 0
        with multiprocessing.Pool(args.workers) as pool, args.out.open('x') as output:
            for i, row in enumerate(pool.imap(collect_position, jobs, chunksize=4), 1):
                if row is not None:
                    output.write(json.dumps(row) + '\n')
                    kept += 1
                if i % 200 == 0:
                    output.flush()
                    print(f'{i}/{args.draws} drawn, {kept} labeled', flush=True)
        print(f'{kept} labeled positions', flush=True)
        return
    data = args.source.read_bytes()
    rows = [r for line in data.splitlines() if 'meta' not in (r := json.loads(line))]
    if args.action == 'fit':
        result = dict(**fit(rows), training_sha256=hashlib.sha256(data).hexdigest())
    else:
        result = evaluate(rows, load_weights(args.model), args.historical)
    with args.out.open('x') as output:
        output.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
