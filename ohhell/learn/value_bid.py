"""Screen a learned value of a full deal for bidding, against the bid network and simulation.

The simulation scores a bid by sampling deals and playing each out with the
rule-based player for every seat. This replaces the play-out with a network
that looks at the whole deal once bidding is over (every hand, the turned-up
card, every bid) and predicts the trick count of one seat. Its targets come
from rounds really played by the tournament lobby, where nobody saw another
hand, so it learns how such deals actually turn out, not how the rule bot
would play them.

``collect`` plays rounds at the tournament's table sizes with its lobby
(every simulating bot at ``--time-ms``) and stores each deal, the bids and
every seat's tricks. ``train`` fits the network. ``screen`` rebuilds the
positions of a ``bid_blend_fit collect`` file (default
``runs/bid_blend/t10.jsonl``), keeping only those whose earlier bids replay
exactly (checked by the stored network odds), and scores against the real
outcomes there:

- ``sim`` and ``net``: as stored (the simulation and bid network alone);
- ``value``: v2's own deal sampling and imagined later bids, with the value
  network in place of the play-out;
- ``oracle``: the value network on the true deal (later bids imagined): what
  ``value`` would reach if the deal sampling were perfect.

It reports log loss of the observed trick count and the paired points of the
bid each would choose against v2's 0.5 blend. One playout per option, as in
``bid_blend_fit``. A screen, not a tournament result.

    python -m ohhell.learn.value_bid collect --rounds 150000 --out runs/value_bid/train.jsonl
    python -m ohhell.learn.value_bid train runs/value_bid/train.jsonl --out runs/value_bid/model
    python -m ohhell.learn.value_bid screen --model runs/value_bid/model --report runs/value_bid/screen.json
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

from ..arena import deal_round
from ..bots import DEFAULT_SPEC, make_bot
from ..bots.simulation import heuristic_bid
from ..cards import EFFECTIVE_SUIT, POWER, SUIT_OF, cards_in, partner_suit
from ..engine import BIDDING, DONE, NO_BID, RoundState
from ..opponents import noisy_bid
from ..tournament import BASELINE, CANDIDATE, build_lobby, rules_for, table_size
from .nn import MLP, fit, softmax_loss

OWNERS = 10  # relative seats 0-7, the turned-up card, not dealt
UPCARD, UNDEALT = 8, 9
TYPES = ('v2', 'skilled', 'casual', 'novice')
INPUTS = 52 * OWNERS + 8 * 4 + 6 + 10 + 8 + 7 + len(TYPES)
FLOOR = 1e-3


def _slots(trump):
    """Card -> slot: trumps by power, then the same-colour suit, then the other two."""
    others = [s for s in range(4) if s not in (trump, partner_suit(trump))]

    def key(card):
        suit = EFFECTIVE_SUIT[trump][card]
        group = 0 if suit == trump else 1 if suit == partner_suit(trump) else 2 + others.index(suit)
        return group, -POWER[trump][card]
    order = sorted(range(52), key=key)
    slot = [0] * 52
    for index, card in enumerate(order):
        slot[card] = index
    return slot


SLOT = [_slots(t) for t in range(4)]
SLOT_BASE = np.arange(52) * OWNERS


def deal_part(hands, trump_card, hero):
    """The 520 card-owner inputs, relative to ``hero``."""
    n = len(hands)
    slot = SLOT[SUIT_OF[trump_card]]
    owner = np.full(52, UNDEALT, dtype=np.int64)
    owner[slot[trump_card]] = UPCARD
    for seat, hand in enumerate(hands):
        rel = (seat - hero) % n
        for card in cards_in(hand):
            owner[slot[card]] = rel
    return SLOT_BASE + owner


def features(hands, trump_card, hand_size, dealer, bids, hero, hero_type, out=None):
    """One input row for ``hero`` once every bid in ``bids`` is known."""
    n = len(hands)
    x = np.zeros(INPUTS, dtype=np.float32) if out is None else out
    x[deal_part(hands, trump_card, hero)] = 1.0
    at = 52 * OWNERS
    for rel in range(n):
        bid = bids[(hero + rel) % n]
        x[at + 4 * rel:at + 4 * rel + 4] = (1.0, bid / 10.0, bid == 0, bid / hand_size)
    at += 32
    x[at + n - 3] = 1.0
    at += 6
    x[at + hand_size - 1] = 1.0
    at += 10
    x[at + (hero - dealer - 1) % n] = 1.0
    at += 8
    x[at + max(-3, min(3, sum(bids) - hand_size)) + 3] = 1.0
    at += 7
    x[at + hero_type] = 1.0
    return x


def _lobby(time_ms):
    lobby = build_lobby(DEFAULT_SPEC, DEFAULT_SPEC, time_ms, 'stand-in')
    specs = [spec for _, _, spec in lobby]
    types = [TYPES.index(role) if role in TYPES else 0 for _, role, _ in lobby]
    return specs, types


def _play_to_end(state, bots):
    while state.phase != DONE:
        seat = state.to_act
        state.apply(bots[seat].act(state.observation(seat)))


# --- collect -------------------------------------------------------------------

def _collect_rounds(job):
    first, count, time_ms = job
    specs, types = _lobby(time_ms)
    out = []
    for number in range(first, first + count):
        rng = random.Random(number)
        n = table_size(number)
        rules = rules_for(n)
        members = [CANDIDATE, BASELINE] + rng.sample(range(BASELINE + 1, len(specs)), n - 2)
        rng.shuffle(members)
        bots = [make_bot(specs[m], number * 31 + seat) for seat, m in enumerate(members)]
        hand_size = rng.randint(1, rules.max_hand)
        hands, trump = deal_round(rng, n, hand_size)
        state = RoundState(rules, hands, trump, rng.randrange(n))
        _play_to_end(state, bots)
        out.append({'round': number, 'n': n, 'cards': hand_size, 'hands': hands,
                    'trump': trump, 'dealer': state.dealer, 'bids': list(state.bids),
                    'tricks': list(state.tricks_won), 'types': [types[m] for m in members]})
    return out


def collect(args):
    args.out.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    chunk = 50
    jobs = [(args.first + start, min(chunk, args.rounds - start), args.time_ms)
            for start in range(0, args.rounds, chunk)]
    started, done = time.monotonic(), 0
    with args.out.open('w') as output, ProcessPoolExecutor(max_workers=args.workers) as pool:
        output.write(json.dumps({'meta': {'rounds': args.rounds, 'first': args.first,
                                          'time_ms': args.time_ms, 'spec': DEFAULT_SPEC}}) + '\n')
        for future in as_completed([pool.submit(_collect_rounds, job) for job in jobs]):
            for record in future.result():
                output.write(json.dumps(record) + '\n')
            done += 1
            if done % 50 == 0 or done == len(jobs):
                output.flush()
                print(f'{done}/{len(jobs)} chunks, {time.monotonic() - started:.0f}s', flush=True)


# --- train ---------------------------------------------------------------------

def _load(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if 'meta' not in row:
            rows.append(row)
    return rows


def _matrix(rows):
    count = sum(r['n'] for r in rows)
    x = np.zeros((count, INPUTS), dtype=np.float32)
    y = np.zeros(count, dtype=np.int64)
    limit = np.zeros(count, dtype=np.int64)
    group = np.zeros(count, dtype=np.int64)
    i = 0
    for r in rows:
        for hero in range(r['n']):
            features(r['hands'], r['trump'], r['cards'], r['dealer'], r['bids'], hero,
                     r['types'][hero], out=x[i])
            y[i], limit[i], group[i] = r['tricks'][hero], r['cards'], r['round']
            i += 1
    return x, y, limit, group


def train(args):
    rows = _load(args.data)
    # Hold out whole rounds for the final test (fit() also holds out 5% for selection).
    test_rows = [r for r in rows if r['round'] % 20 == 0]
    train_rows = [r for r in rows if r['round'] % 20 != 0]
    x, y, limit, _ = _matrix(train_rows)
    print(f'{len(train_rows):,} training rounds, {len(x):,} seat rows', flush=True)
    sizes = [INPUTS] + [int(h) for h in args.hidden.split(',')] + [11]
    model = MLP(sizes, seed=args.seed)
    fit(model, x, y, softmax_loss(limit), epochs=args.epochs, rate=args.rate, seed=args.seed,
        log=lambda text: print(text, flush=True))
    tx, ty, tlimit, _ = _matrix(test_rows)
    probs = predict(model, tx, tlimit)
    test = float(-np.log(np.maximum(probs[np.arange(len(ty)), ty], 1e-9)).mean())
    print(f'test rounds {len(test_rows):,}, rows {len(tx):,}: log loss {test:.4f}', flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    model.save(args.out / 'value.npz')
    digest = hashlib.sha256((args.out / 'value.npz').read_bytes()).hexdigest()
    (args.out / 'training.json').write_text(json.dumps({
        'data': str(args.data), 'data_sha256': hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        'sizes': sizes, 'epochs': args.epochs, 'rate': args.rate, 'seed': args.seed,
        'train_rounds': len(train_rows), 'test_rounds': len(test_rows),
        'test_log_loss': test, 'value_sha256': digest}, indent=2))
    print(f'saved {args.out / "value.npz"} sha256 {digest}', flush=True)


def load_value(folder):
    """The value network in ``folder``/value.npz (or a path to the file)."""
    from .models import REPO_ROOT
    path = Path(folder)
    if path.is_dir() or path.suffix != '.npz':
        path = path / 'value.npz'
    if not path.is_absolute() and not path.exists() and (REPO_ROOT / path).exists():
        path = REPO_ROOT / path
    model = MLP.load(path)
    if model.weights[0].shape[0] != INPUTS or model.weights[-1].shape[1] != 11:
        raise ValueError(f'{path} is not a {INPUTS}-input value network.')
    return model


def predict(model, x, limit):
    logits = model(x)
    logits = logits - logits.max(axis=1, keepdims=True)
    logits[np.arange(logits.shape[1]) > limit[:, None]] = -1e4
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


# --- screen --------------------------------------------------------------------

def _rebuild(record, specs):
    """State, bots and our seat at our bid, replaying the earlier bids."""
    number = record['round']
    rng = random.Random(number)
    n = table_size(number)
    rules = rules_for(n)
    members = [CANDIDATE, BASELINE] + rng.sample(range(BASELINE + 1, len(specs)), n - 2)
    rng.shuffle(members)
    bots = [make_bot(specs[m], number * 31 + seat) for seat, m in enumerate(members)]
    ours = members.index(CANDIDATE)
    hand_size = rng.randint(1, rules.max_hand)
    hands, trump = deal_round(rng, n, hand_size)
    state = RoundState(rules, hands, trump, rng.randrange(n))
    while state.phase == BIDDING and state.to_act != ours:
        seat = state.to_act
        state.apply(bots[seat].act(state.observation(seat)))
    return state, bots[ours], ours


def _imagined_bids(obs, hands, kinds, bot, rng):
    """Every seat's bid in one world, as v2's bid simulation imagines it (ours left NO_BID)."""
    bids = list(obs.bids)
    for seat in range(obs.n):
        if bids[seat] == NO_BID and seat != obs.seat:
            bids[seat] = noisy_bid(heuristic_bid(hands[seat], obs, seat, bot.estimate),
                                   kinds[seat], rng, range(obs.hand_size + 1))
    return bids


def _with_our_bid(bids, bid, obs):
    out = list(bids)
    out[obs.seat] = bid
    if (obs.rules.dealer_restriction and obs.seat != obs.dealer
            and obs.bids[obs.dealer] == NO_BID and sum(out) == obs.hand_size):
        out[obs.dealer] += 1 if out[obs.dealer] == 0 else -1
    return out


def _value_odds(model, worlds, obs, options, bot, rng):
    rows, limits = [], []
    for hands, kinds in worlds:
        bids = _imagined_bids(obs, hands, kinds, bot, rng)
        for bid in options:
            rows.append(features(hands, obs.trump_card, obs.hand_size, obs.dealer,
                                 _with_our_bid(bids, bid, obs), obs.seat, 0))
            limits.append(obs.hand_size)
    probs = predict(model, np.stack(rows), np.array(limits)).reshape(len(worlds), len(options), 11)
    return probs.mean(axis=0)[:, :obs.hand_size + 1]


def _screen_chunk(job):
    records, model_path, worlds, time_ms, seed = job
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    model = MLP.load(model_path)
    specs, _ = _lobby(time_ms)
    out = []
    for record in records:
        state, bot, ours = _rebuild(record, specs)
        obs = state.observation(ours)
        options = record['options']
        predicted = bot.models.bid_distributions(
            obs.hand, obs.trump_card, obs.n, obs.hand_size, obs.seat, obs.dealer, obs.bids, options)
        if not all(np.allclose(predicted[b], stored, atol=1e-5)
                   for b, stored in zip(options, record['net'])):
            out.append(None)  # earlier bids did not replay exactly
            continue
        rng = random.Random(seed * 1000003 + record['round'])
        sampled = bot.belief.worlds(obs, worlds, rng)
        value = _value_odds(model, sampled, obs, options, bot, rng)
        truth = [(state.hands, [bot.belief._kinds(obs, rng)[s] for s in range(obs.n)])]
        oracle = _value_odds(model, truth, obs, options, bot, rng)
        out.append(dict(record, value=value.tolist(), oracle=oracle.tolist()))
    return out


def _points(bid, k):
    from ..rules import Scoring
    global _SCORE
    if _SCORE is None:
        _SCORE = Scoring().points
    return _SCORE(bid, k)


_SCORE = None


def _choose(record, mix):
    """Index of the option with the best expected points under ``mix(record, i)``."""
    best, best_value = 0, -math.inf
    for i, bid in enumerate(record['options']):
        dist = mix(record, i)
        value = sum(p * _points(bid, k) for k, p in enumerate(dist))
        if value > best_value:
            best, best_value = i, value
    return best


def _blend(a, wa, b=None, wb=0.0, c=None):
    def mix(record, i):
        rows = [np.asarray(record[a][i]) * wa]
        if b:
            rows.append(np.asarray(record[b][i][:len(record[a][i])]) * wb)
        if c:
            rows.append(np.asarray(record[c][i][:len(record[a][i])]) * (1.0 - wa - wb))
        return sum(rows)
    return mix


def _summary(records, methods):
    report = {}
    v2 = _blend('sim', 0.5, 'net', 0.5)
    base = [r['points'][_choose(r, v2)] for r in records]
    for name, mix in methods.items():
        loss = []
        for r in records:
            for i, k in enumerate(r['tricks']):
                loss.append(-math.log(max(float(mix(r, i)[k]), FLOOR)))
        diff = [r['points'][_choose(r, mix)] - b for r, b in zip(records, base)]
        mean = sum(diff) / len(diff)
        se = math.sqrt(sum((d - mean) ** 2 for d in diff) / max(1, len(diff) - 1) / len(diff))
        report[name] = {'log_loss': sum(loss) / len(loss), 'points_vs_v2': mean,
                        'interval': 1.96 * se}
    return report


def screen(args):
    from .bid_blend_fit import load
    meta, records = load(args.data)
    if args.limit:
        records = records[:args.limit]
    model_path = str(args.model / 'value.npz')
    chunks = [records[i:i + 40] for i in range(0, len(records), 40)]
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    started, kept, dropped, done = time.monotonic(), [], 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_screen_chunk, (c, model_path, args.worlds, meta['time_ms'], args.seed))
                   for c in chunks]
        for future in as_completed(futures):
            for r in future.result():
                if r is None:
                    dropped += 1
                else:
                    kept.append(r)
            done += 1
            if done % 50 == 0 or done == len(chunks):
                print(f'{done}/{len(chunks)} chunks, kept {len(kept):,}, dropped {dropped:,}, '
                      f'{time.monotonic() - started:.0f}s', flush=True)
    methods = {
        'sim': _blend('sim', 1.0), 'net': _blend('net', 1.0), 'v2 (0.5 sim + 0.5 net)': _blend('sim', 0.5, 'net', 0.5),
        'value': _blend('value', 1.0), 'oracle (true deal)': _blend('oracle', 1.0),
        '0.5 value + 0.5 net': _blend('value', 0.5, 'net', 0.5),
        '0.35 value + 0.65 net': _blend('value', 0.35, 'net', 0.65),
        '0.33 each': _blend('value', 1 / 3, 'net', 1 / 3, 'sim'),
    }
    groups = {'all': kept}
    for lo, hi in ((1, 1), (2, 3), (4, 6), (7, 10)):
        groups[f'{lo}-{hi} cards'] = [r for r in kept if lo <= r['cards'] <= hi]
    for n in range(3, 9):
        groups[f'{n} players'] = [r for r in kept if r['n'] == n]
    report = {'meta': {'data': str(args.data), 'model': model_path, 'worlds': args.worlds,
                       'seed': args.seed, 'kept': len(kept), 'dropped': dropped,
                       'value_sha256': hashlib.sha256(Path(model_path).read_bytes()).hexdigest()},
              'groups': {name: dict(count=len(rows), **_summary(rows, methods))
                         for name, rows in groups.items() if rows}}
    for name, entry in report['groups'].items():
        print(f'\n== {name}: {entry["count"]:,} positions ==')
        for method, stats in entry.items():
            if method == 'count':
                continue
            print(f'  {method:24s} log loss {stats["log_loss"]:.4f}   points vs v2 '
                  f'{stats["points_vs_v2"]:+.3f} +/- {stats["interval"]:.3f}')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--rounds', type=int, default=150000)
    c.add_argument('--first', type=int, default=8000000)
    c.add_argument('--time-ms', type=float, default=2.0)
    c.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    c.add_argument('--out', type=Path, default=Path('runs/value_bid/train.jsonl'))
    t = sub.add_parser('train')
    t.add_argument('data', type=Path)
    t.add_argument('--out', type=Path, default=Path('runs/value_bid/model'))
    t.add_argument('--hidden', default='256,128')
    t.add_argument('--epochs', type=int, default=12)
    t.add_argument('--rate', type=float, default=1e-3)
    t.add_argument('--seed', type=int, default=1)
    s = sub.add_parser('screen')
    s.add_argument('--data', type=Path, default=Path('runs/bid_blend/t10.jsonl'))
    s.add_argument('--model', type=Path, default=Path('runs/value_bid/model'))
    s.add_argument('--worlds', type=int, default=64)
    s.add_argument('--seed', type=int, default=1)
    s.add_argument('--limit', type=int, default=0)
    s.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    s.add_argument('--report', type=Path, default=None)
    args = parser.parse_args()
    {'collect': collect, 'train': train, 'screen': screen}[args.command](args)


if __name__ == '__main__':
    main()
