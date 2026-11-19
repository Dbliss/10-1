"""Screen a learned value of a position for card play, against v2's card choices.

v2 scores each card mostly by sampling deals and playing each to the end of
the round with the rule-based player for every seat. This replaces that
long play-out: after our card, the current trick is finished the same way,
then a network predicts our final trick count from the whole position at
the start of the next trick (every hand, what has been played, bids, tricks
won, who leads). Its targets are real outcomes of rounds played by the
tournament lobby, where nobody saw another hand.

``collect`` plays lobby rounds (every simulating bot at ``--time-ms``) and
stores every trick-start position with the final tricks. ``train`` fits the
network. ``positions`` plays lobby rounds and, at one card decision of the
v2 seat per round (outside the exact endgame), stores v2's own choice and
one real outcome for every legal card (the round finished by the real bots).
``screen`` rebuilds each position and compares, by the points of the chosen
card against v2's choice on the same position:

- ``value``: v2's deal sampling, our card, the trick finished by the
  simulated population, then the value network;
- ``sim``: the same deals played out to the end (v2's simulation, in Python);
- ``net``: the play network alone; and blends using v2's weights.

    python -m ohhell.learn.value_play collect --rounds 300000 --out runs/value_play/train.jsonl
    python -m ohhell.learn.value_play train runs/value_play/train.jsonl --out runs/value_play/model
    python -m ohhell.learn.value_play positions --rounds 40000 --out runs/value_play/positions.jsonl
    python -m ohhell.learn.value_play screen runs/value_play/positions.jsonl --model runs/value_play/model
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
from ..bots import make_bot
from ..cards import SUIT_OF, cards_in
from ..engine import DONE, PLAYING, RoundState
from ..opponents import PopulationPolicy
from ..rules import Scoring
from ..tournament import BASELINE, CANDIDATE, rules_for, table_size
from .nn import MLP, fit, softmax_loss
from .value_bid import SLOT, TYPES, _lobby

OWNERS = 11  # holder 0-7 relative to the hero, played, turned-up card, not dealt
PLAYED, UPCARD, UNDEALT = 8, 9, 10
INPUTS = 52 * OWNERS + 8 * 7 + 8 + 6 + 10 + 10 + 11 + len(TYPES)
SLOT_BASE = np.arange(52) * OWNERS
ENDGAME_CARDS = 12  # v2's exact endgame starts at this many cards in all hands


def features(hands, played, trump_card, hand_size, bids, won, leader, hero, hero_type,
             out=None):
    """One input row for ``hero`` at the start of a trick."""
    n = len(hands)
    x = np.zeros(INPUTS, dtype=np.float32) if out is None else out
    slot = SLOT[SUIT_OF[trump_card]]
    owner = np.full(52, UNDEALT, dtype=np.int64)
    for card in cards_in(played):
        owner[slot[card]] = PLAYED
    owner[slot[trump_card]] = UPCARD
    for seat, hand in enumerate(hands):
        rel = (seat - hero) % n
        for card in cards_in(hand):
            owner[slot[card]] = rel
    x[SLOT_BASE + owner] = 1.0
    at = 52 * OWNERS
    need_total = 0
    for rel in range(n):
        seat = (hero + rel) % n
        bid, got = bids[seat], won[seat]
        need = bid - got
        need_total += max(0, need)
        x[at + 7 * rel:at + 7 * rel + 7] = (1.0, bid / 10.0, bid == 0, got / 10.0,
                                            max(0, need) / 10.0, need < 0, need == 0)
    at += 56
    x[at + (leader - hero) % n] = 1.0
    at += 8
    x[at + n - 3] = 1.0
    at += 6
    x[at + hand_size - 1] = 1.0
    at += 10
    left = hands[hero].bit_count()
    x[at + left - 1] = 1.0
    at += 10
    x[at + max(-5, min(5, need_total - left)) + 5] = 1.0
    at += 11
    x[at + hero_type] = 1.0
    return x


def _deal(number, specs):
    rng = random.Random(number)
    n = table_size(number)
    rules = rules_for(n)
    members = [CANDIDATE, BASELINE] + rng.sample(range(BASELINE + 1, len(specs)), n - 2)
    rng.shuffle(members)
    hand_size = rng.randint(1, rules.max_hand)
    hands, trump = deal_round(rng, n, hand_size)
    state = RoundState(rules, hands, trump, rng.randrange(n))
    return state, members, rng


# --- collect -------------------------------------------------------------------

def _collect_rounds(job):
    first, count, time_ms = job
    specs, types = _lobby(time_ms)
    out = []
    for number in range(first, first + count):
        state, members, _ = _deal(number, specs)
        bots = [make_bot(specs[m], number * 31 + seat) for seat, m in enumerate(members)]
        snaps = []
        while state.phase != DONE:
            if state.phase == PLAYING and state.trick_size == 0:
                snaps.append([list(state.hands), state.played, list(state.tricks_won), state.leader])
            seat = state.to_act
            state.apply(bots[seat].act(state.observation(seat)))
        out.append({'round': number, 'n': state.n, 'cards': state.hand_size,
                    'trump': state.trump_card, 'dealer': state.dealer, 'bids': list(state.bids),
                    'tricks': list(state.tricks_won), 'types': [types[m] for m in members],
                    'snaps': snaps})
    return out


def _run_chunks(work, jobs, out_path, meta, workers, every=50):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    started, done = time.monotonic(), 0
    with out_path.open('w') as output, ProcessPoolExecutor(max_workers=workers) as pool:
        output.write(json.dumps({'meta': meta}) + '\n')
        for future in as_completed([pool.submit(work, job) for job in jobs]):
            for record in future.result():
                output.write(json.dumps(record) + '\n')
            done += 1
            if done % every == 0 or done == len(jobs):
                output.flush()
                print(f'{done}/{len(jobs)} chunks, {time.monotonic() - started:.0f}s', flush=True)


def collect(args):
    chunk = 50
    jobs = [(args.first + s, min(chunk, args.rounds - s), args.time_ms)
            for s in range(0, args.rounds, chunk)]
    _run_chunks(_collect_rounds, jobs, args.out,
                {'rounds': args.rounds, 'first': args.first, 'time_ms': args.time_ms}, args.workers)


# --- train ---------------------------------------------------------------------

def _rows(path):
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if 'meta' not in row:
            yield row


def _matrix(rows, keep, seed):
    rng = random.Random(seed)
    picks = []
    for r in rows:
        for snap in r['snaps']:
            for hero in range(r['n']):
                if rng.random() < keep:
                    picks.append((r, snap, hero))
    x = np.zeros((len(picks), INPUTS), dtype=np.float32)
    y = np.zeros(len(picks), dtype=np.int64)
    limit = np.zeros(len(picks), dtype=np.int64)
    for i, (r, (hands, played, won, leader), hero) in enumerate(picks):
        features(hands, played, r['trump'], r['cards'], r['bids'], won, leader, hero,
                 r['types'][hero], out=x[i])
        y[i], limit[i] = r['tricks'][hero], r['cards']
    return x, y, limit


def train(args):
    rows = list(_rows(args.data))
    test_rows = [r for r in rows if r['round'] % 20 == 0]
    train_rows = [r for r in rows if r['round'] % 20 != 0]
    x, y, limit = _matrix(train_rows, args.keep, args.seed)
    print(f'{len(train_rows):,} training rounds, {len(x):,} rows', flush=True)
    sizes = [INPUTS] + [int(h) for h in args.hidden.split(',')] + [11]
    model = MLP(sizes, seed=args.seed)
    fit(model, x, y, softmax_loss(limit), epochs=args.epochs, rate=args.rate, seed=args.seed,
        log=lambda text: print(text, flush=True))
    del x, y, limit
    tx, ty, tlimit = _matrix(test_rows, 0.25, args.seed + 1)
    probs = predict(model, tx, tlimit)
    test = float(-np.log(np.maximum(probs[np.arange(len(ty)), ty], 1e-9)).mean())
    print(f'test rounds {len(test_rows):,}, rows {len(tx):,}: log loss {test:.4f}', flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    model.save(args.out / 'value.npz')
    digest = hashlib.sha256((args.out / 'value.npz').read_bytes()).hexdigest()
    (args.out / 'training.json').write_text(json.dumps({
        'data': str(args.data), 'sizes': sizes, 'epochs': args.epochs, 'rate': args.rate,
        'seed': args.seed, 'keep': args.keep, 'train_rounds': len(train_rows),
        'test_rounds': len(test_rows), 'test_log_loss': test, 'value_sha256': digest}, indent=2))
    print(f'saved {args.out / "value.npz"} sha256 {digest}', flush=True)


def predict(model, x, limit):
    logits = model(x)
    logits = logits - logits.max(axis=1, keepdims=True)
    logits[np.arange(logits.shape[1]) > limit[:, None]] = -1e4
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def load_value(folder):
    from .models import REPO_ROOT
    path = Path(folder)
    if path.suffix != '.npz':
        path = path / 'value.npz'
    if not path.is_absolute() and not path.exists() and (REPO_ROOT / path).exists():
        path = REPO_ROOT / path
    model = MLP.load(path)
    if model.weights[0].shape[0] != INPUTS:
        raise ValueError(f'{path} is not a {INPUTS}-input card value network.')
    return model


# --- positions -----------------------------------------------------------------

def _position_rounds(job):
    first, count, time_ms, rate = job
    specs, _ = _lobby(time_ms)
    out = []
    for number in range(first, first + count):
        state, members, rng = _deal(number, specs)
        bots = [make_bot(specs[m], number * 31 + seat) for seat, m in enumerate(members)]
        ours = members.index(CANDIDATE)
        record = None
        while state.phase != DONE and record is None:
            seat = state.to_act
            obs = state.observation(seat)
            if (seat == ours and state.phase == PLAYING and len(obs.legal_actions()) > 1
                    and sum(h.bit_count() for h in state.hands) - state.trick_size > ENDGAME_CARDS
                    and rng.random() < rate):
                cards = obs.legal_actions()
                choice = bots[seat].act(obs)
                points, tricks = [], []
                for card in cards:
                    branch = state.clone()
                    branch.history = list(state.history)
                    branch.apply(card)
                    while branch.phase != DONE:
                        s = branch.to_act
                        branch.apply(bots[s].act(branch.observation(s)))
                    points.append(branch.points()[ours])
                    tricks.append(branch.tricks_won[ours])
                record = {'round': number, 'n': state.n, 'cards': state.hand_size,
                          'seat': ours, 'history': [list(p) for p in state.history],
                          'bids': list(state.bids), 'options': cards, 'v2': choice,
                          'points': points, 'tricks': tricks,
                          'left': state.hands[ours].bit_count()}
                break
            state.apply(bots[seat].act(obs))
        if record:
            out.append(record)
    return out


def positions(args):
    chunk = 40
    jobs = [(args.first + s, min(chunk, args.rounds - s), args.time_ms, args.rate)
            for s in range(0, args.rounds, chunk)]
    _run_chunks(_position_rounds, jobs, args.out,
                {'rounds': args.rounds, 'first': args.first, 'time_ms': args.time_ms,
                 'rate': args.rate}, args.workers)


# --- screen --------------------------------------------------------------------

def _rebuild(record, specs):
    state, _, _ = _deal(record['round'], specs)
    order = [(state.dealer + 1 + i) % state.n for i in range(state.n)]
    for seat in order:
        state.apply(record['bids'][seat])
    for seat, card in record['history']:
        assert state.to_act == seat
        state.apply(card)
    return state


def _finish_trick(state, policy):
    while state.phase == PLAYING and state.trick_size:
        state.play_unchecked(policy(state))


def _screen_chunk(job):
    records, model_path, worlds, time_ms, seed = job
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    model = load_value(model_path)
    specs, _ = _lobby(time_ms)
    score = Scoring().points
    out = []
    for record in records:
        state = _rebuild(record, specs)
        seat = record['seat']
        obs = state.observation(seat)
        cards = record['options']
        assert sorted(obs.legal_actions()) == sorted(cards)
        bot = make_bot(specs[CANDIDATE], record['round'] * 31 + seat)
        rng = random.Random(seed * 1000003 + record['round'])
        net = bot.models.card_values(obs, seat, obs.hand, cards)
        bid = obs.bids[seat]
        sim = np.zeros(len(cards))
        val = np.zeros(len(cards))
        rows, owners, exact = [], [], []
        sampled = bot.belief.worlds(obs, worlds, rng)
        for hands, kinds in sampled:
            policy = PopulationPolicy(kinds, rng)
            for i, card in enumerate(cards):
                st = obs.determinize(hands)
                st.history = None
                st.play_unchecked(card)
                _finish_trick(st, policy)
                if st.phase == DONE:
                    exact.append((i, score(bid, st.tricks_won[seat])))
                else:
                    rows.append(features(st.hands, st.played, st.trump_card, st.hand_size,
                                         st.bids, st.tricks_won, st.leader, seat, 0))
                    owners.append(i)
                while st.phase == PLAYING:
                    st.play_unchecked(policy(st))
                sim[i] += st.points()[seat]
        if rows:
            probs = predict(model, np.stack(rows), np.full(len(rows), state.hand_size))
            table = np.array([score(bid, k) for k in range(11)], dtype=np.float64)
            for i, ev in zip(owners, probs @ table):
                val[i] += ev
        for i, points in exact:
            val[i] += points
        sim /= len(sampled)
        val /= len(sampled)
        out.append(dict(record, sim=sim.tolist(), value=val.tolist(),
                        net=[float(v) for v in net]))
    return out


def _pick(record, scores, shortlist=None):
    cards = record['options']
    order = list(range(len(cards)))
    if shortlist:
        order = sorted(order, key=lambda i: record['net'][i], reverse=True)[:shortlist]
    return max(order, key=lambda i: scores[i])


def _methods():
    def blend(key, weight=32.0, prior=16.0, shortlist=4):
        def choose(r):
            s = [(weight * a + prior * b) / (weight + prior) for a, b in zip(r[key], r['net'])]
            return _pick(r, s, shortlist)
        return choose
    return {
        'net': lambda r: _pick(r, r['net']),
        'sim': lambda r: _pick(r, r['sim']),
        'value': lambda r: _pick(r, r['value']),
        'v2-like (sim, net prior 16, top 4)': blend('sim'),
        'value, net prior 16, top 4': blend('value'),
        'value, net prior 16, all cards': blend('value', shortlist=None),
        'value + sim, net prior 16, top 4': lambda r: _pick(r, [
            (16 * a + 16 * b + 16 * c) / 48 for a, b, c in zip(r['value'], r['sim'], r['net'])], 4),
    }


def _summary(records):
    report = {}
    for name, choose in _methods().items():
        diff, agree = [], 0
        for r in records:
            i = choose(r)
            v2 = r['options'].index(r['v2'])
            diff.append(r['points'][i] - r['points'][v2])
            agree += i == v2
        mean = sum(diff) / len(diff)
        se = math.sqrt(sum((d - mean) ** 2 for d in diff) / max(1, len(diff) - 1) / len(diff))
        report[name] = {'points_vs_v2': mean, 'interval': 1.96 * se, 'same_as_v2': agree / len(records)}
    return report


def screen(args):
    meta, records = {}, []
    for line in Path(args.data).read_text().splitlines():
        row = json.loads(line)
        if 'meta' in row:
            meta = row['meta']
        else:
            records.append(row)
    if args.limit:
        records = records[:args.limit]
    chunks = [records[i:i + 25] for i in range(0, len(records), 25)]
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    started, kept, done = time.monotonic(), [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_screen_chunk, (c, str(args.model), args.worlds, meta['time_ms'],
                                               args.seed)) for c in chunks]
        for future in as_completed(futures):
            kept.extend(future.result())
            done += 1
            if done % 50 == 0 or done == len(chunks):
                print(f'{done}/{len(chunks)} chunks, {len(kept):,} positions, '
                      f'{time.monotonic() - started:.0f}s', flush=True)
    groups = {'all': kept}
    for lo, hi in ((2, 3), (4, 6), (7, 10)):
        groups[f'{lo}-{hi} cards left'] = [r for r in kept if lo <= r['left'] <= hi]
    for lo, hi in ((3, 3), (4, 4), (5, 8)):
        groups[f'{lo}-{hi} players'] = [r for r in kept if lo <= r['n'] <= hi]
    report = {'meta': {'data': str(args.data), 'model': str(args.model), 'worlds': args.worlds,
                       'seed': args.seed, 'positions': len(kept)},
              'groups': {k: dict(count=len(v), **_summary(v)) for k, v in groups.items() if v}}
    for name, entry in report['groups'].items():
        print(f'\n== {name}: {entry["count"]:,} positions ==')
        for method, stats in entry.items():
            if method != 'count':
                print(f'  {method:38s} {stats["points_vs_v2"]:+.3f} +/- {stats["interval"]:.3f}'
                      f'   same card as v2 {stats["same_as_v2"]:.0%}')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({**report, 'records': kept}))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    workers = max(1, (os.cpu_count() or 2) - 2)
    c = sub.add_parser('collect')
    c.add_argument('--rounds', type=int, default=300000)
    c.add_argument('--first', type=int, default=9000000)
    c.add_argument('--time-ms', type=float, default=2.0)
    c.add_argument('--workers', type=int, default=workers)
    c.add_argument('--out', type=Path, default=Path('runs/value_play/train.jsonl'))
    t = sub.add_parser('train')
    t.add_argument('data', type=Path)
    t.add_argument('--out', type=Path, default=Path('runs/value_play/model'))
    t.add_argument('--hidden', default='256,128')
    t.add_argument('--epochs', type=int, default=10)
    t.add_argument('--rate', type=float, default=1e-3)
    t.add_argument('--keep', type=float, default=0.25)
    t.add_argument('--seed', type=int, default=1)
    p = sub.add_parser('positions')
    p.add_argument('--rounds', type=int, default=40000)
    p.add_argument('--first', type=int, default=9500000)
    p.add_argument('--time-ms', type=float, default=2.0)
    p.add_argument('--rate', type=float, default=0.35)
    p.add_argument('--workers', type=int, default=workers)
    p.add_argument('--out', type=Path, default=Path('runs/value_play/positions.jsonl'))
    s = sub.add_parser('screen')
    s.add_argument('data', type=Path)
    s.add_argument('--model', type=Path, default=Path('runs/value_play/model'))
    s.add_argument('--worlds', type=int, default=64)
    s.add_argument('--seed', type=int, default=1)
    s.add_argument('--limit', type=int, default=0)
    s.add_argument('--workers', type=int, default=workers)
    s.add_argument('--report', type=Path, default=None)
    args = parser.parse_args()
    {'collect': collect, 'train': train, 'positions': positions, 'screen': screen}[args.command](args)


if __name__ == '__main__':
    main()
