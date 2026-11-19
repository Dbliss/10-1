"""Learned bid likelihood: how the lobby really bids, for reading hidden hands.

v2 judges a sampled deal by how well each earlier bid fits a rule-of-thumb
trick estimate of that seat's sampled hand, with a fixed spread. This fits a
network to P(bid | hand, table, position, earlier bids) on rounds the
tournament lobby really played (every style mixed, as a reader would face
it), so a sampled deal can be weighted by the chance its hands produce the
bids we saw.

    python -m ohhell.learn.bid_policy train runs/value_bid/train.jsonl --out runs/bid_policy/model
    python -m ohhell.learn.bid_policy screen --model runs/bid_policy/model --value runs/value_bid/model

``screen`` repeats ``value_bid screen`` with the deals drawn differently:
``learned``: ``--pool`` deals drawn with no bid reading, weighted by this
network's likelihood of every earlier bid, resampled to ``--worlds``;
``v2``: v2's own sampler (as ``value_bid``). Same positions, same value
network, so the difference is only the reading of earlier bids.
"""

import argparse
import hashlib
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..belief import sample_world
from ..engine import NO_BID
from .features import bid_bower_part, bid_context, bid_hand_part
from .nn import MLP, fit, softmax_loss

INPUTS = 82 + 4 + 4 + 8 + 1


def features(hand, trump_card, n, hand_size, seat, dealer, bids, out=None):
    """Inputs for ``seat``'s bid, given the bids before it (later seats NO_BID)."""
    x = np.zeros(INPUTS, dtype=np.float32) if out is None else out
    x[:82] = bid_hand_part(hand, trump_card, n, hand_size)
    position = (seat - dealer - 1) % n
    earlier = [bids[(dealer + 1 + j) % n] for j in range(position)]
    row = bid_context([b if b != NO_BID else 0 for b in bids], n, hand_size, seat, dealer)[0]
    x[82:86] = row[:4]
    x[86:90] = bid_bower_part(hand, trump_card)
    x[90 + min(position, 7)] = 1.0
    # Sum still needed for the dealer's forbidden bid: only the dealer is restricted.
    x[98] = float(position == n - 1) * (hand_size - sum(earlier)) / max(1, hand_size)
    return x


def _training(path, seed):
    rows = []
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if 'meta' not in r:
            rows.append(r)
    test = [r for r in rows if r['round'] % 20 == 0]
    train = [r for r in rows if r['round'] % 20 != 0]
    return train, test


def _matrix(rows):
    count = sum(r['n'] for r in rows)
    x = np.zeros((count, INPUTS), dtype=np.float32)
    y = np.zeros(count, dtype=np.int64)
    limit = np.zeros(count, dtype=np.int64)
    i = 0
    for r in rows:
        n, dealer = r['n'], r['dealer']
        for seat in range(n):
            position = (seat - dealer - 1) % n
            before = [NO_BID] * n
            for j in range(position):
                s = (dealer + 1 + j) % n
                before[s] = r['bids'][s]
            features(r['hands'][seat], r['trump'], n, r['cards'], seat, dealer, before, out=x[i])
            y[i], limit[i] = r['bids'][seat], r['cards']
            i += 1
    return x, y, limit


def predict(model, x, limit):
    logits = model(x)
    logits = logits - logits.max(axis=1, keepdims=True)
    logits[np.arange(logits.shape[1]) > limit[:, None]] = -1e4
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def train(args):
    train_rows, test_rows = _training(args.data, args.seed)
    x, y, limit = _matrix(train_rows)
    print(f'{len(train_rows):,} rounds, {len(x):,} bids', flush=True)
    sizes = [INPUTS] + [int(h) for h in args.hidden.split(',')] + [11]
    model = MLP(sizes, seed=args.seed)
    fit(model, x, y, softmax_loss(limit), epochs=args.epochs, rate=args.rate, seed=args.seed,
        log=lambda t: print(t, flush=True))
    tx, ty, tlimit = _matrix(test_rows)
    probs = predict(model, tx, tlimit)
    loss = float(-np.log(np.maximum(probs[np.arange(len(ty)), ty], 1e-9)).mean())
    top1 = float((probs.argmax(axis=1) == ty).mean())
    print(f'test {len(test_rows):,} rounds: log loss {loss:.4f}, top-1 {top1:.3f}', flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    model.save(args.out / 'policy.npz')
    digest = hashlib.sha256((args.out / 'policy.npz').read_bytes()).hexdigest()
    (args.out / 'training.json').write_text(json.dumps({
        'data': str(args.data), 'sizes': sizes, 'epochs': args.epochs, 'seed': args.seed,
        'test_log_loss': loss, 'test_top1': top1, 'policy_sha256': digest}, indent=2))
    print(f'saved sha256 {digest}', flush=True)


def load_policy(folder):
    path = Path(folder)
    return MLP.load(path if path.suffix == '.npz' else path / 'policy.npz')


def bid_log_likelihood(model, obs, worlds_hands):
    """Per sampled deal, the summed log chance of every earlier bid we saw.

    ``obs`` is the reader's observation; each entry of ``worlds_hands`` holds
    every seat's opening hand.
    """
    n, dealer = obs.n, obs.dealer
    seats = [s for s in range(n) if s != obs.seat and obs.bids[s] != NO_BID]
    if not seats:
        return np.zeros(len(worlds_hands))
    rows, limits, labels = [], [], []
    for hands in worlds_hands:
        for seat in seats:
            position = (seat - dealer - 1) % n
            before = [NO_BID] * n
            for j in range(position):
                s = (dealer + 1 + j) % n
                before[s] = obs.bids[s]
            rows.append(features(hands[seat], obs.trump_card, n, obs.hand_size, seat, dealer,
                                 before))
            labels.append(obs.bids[seat])
            limits.append(obs.hand_size)
    probs = predict(model, np.stack(rows), np.array(limits))
    picked = np.log(np.maximum(probs[np.arange(len(labels)), labels], 1e-6))
    return picked.reshape(len(worlds_hands), len(seats)).sum(axis=1)


def learned_worlds(model, obs, pool, count, rng, kinds_of):
    """``count`` deals: ``pool`` uniform draws weighted by the learned bid likelihood."""
    candidates = [sample_world(obs, rng) for _ in range(pool)]
    logs = bid_log_likelihood(model, obs, candidates)
    weights = np.exp(logs - logs.max())
    picks = rng.choices(range(pool), weights=weights.tolist(), k=count)
    return [(candidates[i], kinds_of(obs, rng)) for i in picks]


def _screen_chunk(job):
    records, value_path, policy_path, worlds, pool, time_ms, seed = job
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    from .value_bid import _lobby, _rebuild, _value_odds, load_value
    value = load_value(value_path)
    policy = load_policy(policy_path)
    specs, _ = _lobby(time_ms)
    out = []
    for record in records:
        state, bot, ours = _rebuild(record, specs)
        obs = state.observation(ours)
        options = record['options']
        predicted = bot.models.bid_distributions(
            obs.hand, obs.trump_card, obs.n, obs.hand_size, obs.seat, obs.dealer, obs.bids, options)
        if not all(np.allclose(predicted[b], s, atol=1e-5) for b, s in zip(options, record['net'])):
            out.append(None)
            continue
        rng = random.Random(seed * 1000003 + record['round'])
        v2 = _value_odds(value, bot.belief.worlds(obs, worlds, rng), obs, options, bot, rng)
        learned = _value_odds(value, learned_worlds(policy, obs, pool, worlds, rng,
                                                    bot.belief._kinds), obs, options, bot, rng)
        out.append(dict(record, value=v2.tolist(), learned=learned.tolist()))
    return out


def screen(args):
    from .bid_blend_fit import load
    from .value_bid import _blend, _summary
    meta, records = load(args.data)
    if args.limit:
        records = records[:args.limit]
    chunks = [records[i:i + 40] for i in range(0, len(records), 40)]
    kept, dropped, started = [], 0, time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_screen_chunk, (c, str(args.value), str(args.model), args.worlds,
                                               args.pool, meta['time_ms'], args.seed)) for c in chunks]
        for i, future in enumerate(as_completed(futures), 1):
            for r in future.result():
                if r is None:
                    dropped += 1
                else:
                    kept.append(r)
            if i % 50 == 0 or i == len(chunks):
                print(f'{i}/{len(chunks)} chunks, kept {len(kept):,}, {time.monotonic() - started:.0f}s',
                      flush=True)
    methods = {
        'v2 (0.5 sim + 0.5 net)': _blend('sim', 0.5, 'net', 0.5),
        'value, v2 deals': _blend('value', 1.0),
        'value, learned reading': _blend('learned', 1.0),
        '1/3 each, v2 deals': _blend('value', 1 / 3, 'net', 1 / 3, 'sim'),
        '1/3 each, learned reading': _blend('learned', 1 / 3, 'net', 1 / 3, 'sim'),
        '0.5 learned + 0.5 net': _blend('learned', 0.5, 'net', 0.5),
    }
    groups = {'all': kept, 'no earlier bids': [r for r in kept if r['before'] == 0],
              '1-2 earlier bids': [r for r in kept if 1 <= r['before'] <= 2],
              '3+ earlier bids': [r for r in kept if r['before'] >= 3]}
    report = {'meta': {'data': str(args.data), 'value': str(args.value), 'policy': str(args.model),
                       'worlds': args.worlds, 'pool': args.pool, 'kept': len(kept),
                       'dropped': dropped},
              'groups': {k: dict(count=len(v), **_summary(v, methods)) for k, v in groups.items() if v}}
    for name, entry in report['groups'].items():
        print(f'\n== {name}: {entry["count"]:,} positions ==')
        for method, stats in entry.items():
            if method != 'count':
                print(f'  {method:28s} log loss {stats["log_loss"]:.4f}   points vs v2 '
                      f'{stats["points_vs_v2"]:+.3f} +/- {stats["interval"]:.3f}')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('data', type=Path)
    t.add_argument('--out', type=Path, default=Path('runs/bid_policy/model'))
    t.add_argument('--hidden', default='128,64')
    t.add_argument('--epochs', type=int, default=10)
    t.add_argument('--rate', type=float, default=1e-3)
    t.add_argument('--seed', type=int, default=1)
    s = sub.add_parser('screen')
    s.add_argument('--data', type=Path, default=Path('runs/value_bid/confirm_t10.jsonl'))
    s.add_argument('--model', type=Path, default=Path('runs/bid_policy/model'))
    s.add_argument('--value', type=Path, default=Path('runs/value_bid/model'))
    s.add_argument('--worlds', type=int, default=64)
    s.add_argument('--pool', type=int, default=1024)
    s.add_argument('--seed', type=int, default=3)
    s.add_argument('--limit', type=int, default=0)
    s.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    s.add_argument('--report', type=Path, default=None)
    args = parser.parse_args()
    {'train': train, 'screen': screen}[args.command](args)


if __name__ == '__main__':
    main()
