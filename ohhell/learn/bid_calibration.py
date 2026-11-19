"""Fit a cheap exact-bid correction, then validate it on fresh positions."""

import argparse
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import statistics
from types import SimpleNamespace

import numpy as np

from ..bid_calibration import adjust, features, load_coefficients
from ..bots import make_bot
from ..bots.simulation import SimulationBot
from ..duel import FIELD, REFERENCE_SPEC, extend
from ..tournament import check_run_config
from .bid_play_audit import _live_counts, _worlds, bid_position, summarize


def fit(source):
    raw = source.read_bytes()
    x, offset, target, importance = [], [], [], []
    rows = [json.loads(line) for line in raw.splitlines()]
    for row in rows:
        for i, b in enumerate(row['bids']):
            p = row['cheap'][i][b]
            if 0 < p < 1:
                x.append(features(row['cards'], b))
                offset.append(math.log(p / (1 - p)))
                target.append(row['reference'][i][b])
                importance.append(1 / len(row['bids']))
    x, offset, target, importance = map(np.array, (x, offset, target, importance))
    weights = np.zeros(4)
    ridge = 2.0 * np.eye(4)
    for _ in range(20):
        p = 1 / (1 + np.exp(-np.clip(offset + x @ weights, -30, 30)))
        gradient = x.T @ (importance * (p - target)) + ridge @ weights
        hessian = (x.T * (importance * p * (1 - p))) @ x + ridge
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if np.linalg.norm(step) < 1e-8:
            break
    return dict(log_odds_weights=weights.tolist(), training_positions=len(rows),
                training_source=source.as_posix(), training_sha256=hashlib.sha256(raw).hexdigest())


def validate_position(job):
    seed, count, model = job
    prepared = bid_position(seed)
    if prepared is None:
        return None
    obs, bot, bids, bot_seed = prepared
    bot.simulator.bid_budget = 4096
    raw = SimulationBot.analyse_bids(bot, obs).distributions
    net = bot.models.bid_distributions(obs.hand, obs.trump_card, obs.n, obs.hand_size,
                                       obs.seat, obs.dealer, obs.bids, bids)
    reference = make_bot(REFERENCE_SPEC, bot_seed + (1 << 30))
    worlds = _worlds(obs, reference.simulator, bids, count, seed + (1 << 30))
    spec = extend(REFERENCE_SPEC, 'card_budget=128,endgame_budget=128')
    live = _live_counts(obs, worlds, spec, seed + (1 << 32), own_only=False)
    weights = load_coefficients(model)
    return dict(seed=seed, players=obs.n, cards=obs.hand_size, bids=bids,
                cheap=[raw[b] for b in bids], net=[net[b] for b in bids],
                blend=bot.bid_share(obs.hand_size, obs.n), reference=(live / count).tolist(),
                calibrated=[adjust(raw[b], obs.hand_size, b, weights) for b in bids])


def validation_report(rows):
    names = ('cheap', 'calibrated')
    report = summarize(rows, names)
    measured = [(r, summarize([r], names)['all']['policies']) for r in rows]
    for label, group in [('all', measured), ('8_to_10_cards', [(r, m) for r, m in measured if r['cards'] >= 8])]:
        if len(group) < 2:
            continue
        changes = {}
        for metric in ('brier', 'ranked_score', 'make_bias'):
            diffs = [m['calibrated'][metric] - m['cheap'][metric] for _, m in group]
            changes[metric] = dict(delta=statistics.mean(diffs),
                                   margin_95=1.96 * statistics.stdev(diffs) / math.sqrt(len(diffs)))
        report[label]['calibration_minus_cheap'] = changes
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    train = commands.add_parser('fit')
    train.add_argument('source', type=Path)
    train.add_argument('--out', type=Path, required=True)
    check = commands.add_parser('validate')
    check.add_argument('model', type=Path)
    check.add_argument('--seed', type=int, required=True)
    check.add_argument('--positions', type=int, required=True)
    check.add_argument('--worlds', type=int, default=32)
    check.add_argument('--workers', type=int, default=8)
    check.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.action == 'fit':
        with args.out.open('x') as output:
            json.dump(fit(args.source), output, indent=2)
            output.write('\n')
        return

    if min(args.positions, args.worlds, args.workers) < 1:
        parser.error('counts must be positive')
    manifest, summary = (args.out.with_suffix(ext) for ext in ('.meta', '.json'))
    if any(p.exists() for p in (args.out, manifest, summary)):
        parser.error('output already exists')
    model = args.model.as_posix()
    load_coefficients(model)
    config = SimpleNamespace(seed=args.seed, max_sets=args.positions, smallest_gain=2, time_ms=0, resume=False)
    check_run_config(manifest, config, [extend(REFERENCE_SPEC, f'bid_calibration_file={model}'),
                                      *(spec for spec, _ in FIELD)])
    jobs = ((args.seed + i, args.worlds, model) for i in range(args.positions))
    rows = []
    with multiprocessing.Pool(args.workers) as pool, args.out.open('x') as output:
        for i, row in enumerate(pool.imap(validate_position, jobs), 1):
            if row is not None:
                rows.append(row)
                output.write(json.dumps(row) + '\n')
            if i % 50 == 0:
                output.flush()
                print(f'{i}/{args.positions} positions', flush=True)
    results = dict(draws=args.positions, reference_worlds=args.worlds, **validation_report(rows))
    summary.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
