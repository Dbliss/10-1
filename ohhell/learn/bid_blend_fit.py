"""Fit how much the bid network should count against the simulation.

``collect`` deals rounds at the tournament's table sizes with its lobby. At
one v2 seat's bid it keeps what the simulation alone and the network alone
predict for every shortlisted bid, then plays the real deal out once after
each of those bids (the later seats bid and everyone plays as they really
would, v2 playing our seat). Every option therefore has an observed outcome,
not only the bid v2 chose.

``fit`` uses those records to

- score fixed network shares (0 = simulation only, 0.5 = v2, 1 = network
  only) by the points their chosen bid really made, by hand size, table size
  and bidding position;
- fit a table of network shares per hand size and table size by the log loss
  of the observed trick count, shrunk towards neighbouring cells, on a
  training split of rounds; and
- report the fitted table's points on the held-out rounds.

A playout is one noisy sample of each option's value, with v2 playing the
cards. This screens a bid blend; it is not a tournament result.

    python -m ohhell.learn.bid_blend_fit collect --rounds 24000 --time-ms 10 --out runs/bid_blend/t10.jsonl
    python -m ohhell.learn.bid_blend_fit fit runs/bid_blend/t10.jsonl --write ohhell/models/bid_blend.json
"""

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ..arena import deal_round
from ..bots import DEFAULT_SPEC, make_bot
from ..bots.simulation import SimulationBot
from ..engine import BIDDING, DONE, NO_BID, RoundState
from ..tournament import BASELINE, CANDIDATE, build_lobby, rules_for, table_size

GRID = [i / 20 for i in range(21)]
FLOOR = 1e-3
V2_SHARE = 0.5


def _finish(state, bots):
    while state.phase != DONE:
        seat = state.to_act
        state.apply(bots[seat].act(state.observation(seat)))


def _rounds(job):
    first, count, time_ms = job
    specs = [spec for _, _, spec in build_lobby(DEFAULT_SPEC, DEFAULT_SPEC, time_ms, 'stand-in')]
    out = []
    for number in range(first, first + count):
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
        bot = bots[ours]
        obs = state.observation(ours)
        options = bot._bid_options(obs)
        if len(options) < 2:
            continue
        bot.deadline = time.perf_counter() + bot.time_limit
        simulated = SimulationBot.analyse_bids(bot, obs)
        bot.deadline = None
        predicted = bot.models.bid_distributions(
            obs.hand, obs.trump_card, obs.n, obs.hand_size, obs.seat, obs.dealer,
            obs.bids, options)
        tricks, points = [], []
        for bid in options:
            branch = state.clone()
            branch.history = []  # clone drops it; nothing is played while bidding
            branch.apply(bid)
            _finish(branch, bots)
            tricks.append(branch.tricks_won[ours])
            points.append(branch.points()[ours])
        out.append({
            'round': number, 'n': n, 'cards': hand_size,
            'before': sum(b != NO_BID for b in obs.bids), 'samples': simulated.samples,
            'options': options,
            'sim': [simulated.distributions[b] for b in options],
            'net': [predicted[b] for b in options],
            'tricks': tricks, 'points': points,
        })
    return out


def collect(args):
    args.out.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    chunk = 40
    jobs = [(args.first + start, min(chunk, args.rounds - start), args.time_ms)
            for start in range(0, args.rounds, chunk)]
    started = time.monotonic()
    done = 0
    with args.out.open('w') as output, ProcessPoolExecutor(max_workers=args.workers) as pool:
        output.write(json.dumps({'meta': {
            'rounds': args.rounds, 'first': args.first, 'time_ms': args.time_ms,
            'spec': DEFAULT_SPEC, 'workers': args.workers}}) + '\n')
        for future in as_completed([pool.submit(_rounds, job) for job in jobs]):
            for record in future.result():
                output.write(json.dumps(record) + '\n')
            output.flush()
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f'{done}/{len(jobs)} chunks, {time.monotonic() - started:.0f}s', flush=True)


def load(path):
    meta, records = {}, []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if 'meta' in row:
            meta = row['meta']
        else:
            records.append(row)
    return meta, records


def _points_table():
    from ..rules import Scoring
    score = Scoring().points
    return [[score(b, k) for k in range(11)] for b in range(11)]


POINTS = None


def choice(record, share):
    """Index of the option a bot with this network share would bid."""
    global POINTS
    POINTS = POINTS or _points_table()
    best, best_value = 0, -math.inf
    for i, bid in enumerate(record['options']):
        row = POINTS[bid]
        value = sum(((1.0 - share) * s + share * p) * row[k]
                    for k, (s, p) in enumerate(zip(record['sim'][i], record['net'][i])))
        if value > best_value:
            best, best_value = i, value
    return best


def log_loss(record, share):
    """Summed over the options: -log of the chance given to what happened."""
    total = 0.0
    for sim, net, k in zip(record['sim'], record['net'], record['tricks']):
        total -= math.log(max((1.0 - share) * sim[k] + share * net[k], FLOOR))
    return total


def mean_se(values):
    count = len(values)
    if not count:
        return 0.0, 0.0
    mean = sum(values) / count
    if count < 2:
        return mean, 0.0
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / (count - 1) / count)


def paired(records, share_of, base_of):
    """Mean points per bid of one rule minus another on the same records."""
    return mean_se([r['points'][choice(r, share_of(r))] - r['points'][choice(r, base_of(r))]
                    for r in records])


def fit_table(records, shrink):
    """A network share per (hand size, table size), by penalised log loss.

    Each cell takes the grid share with the lowest loss, where the loss adds
    ``shrink`` pseudo-observations of the losses of the same hand size at
    every table and of the same table at every hand size. That lets sparse
    cells borrow from their row and column without forcing a line through
    them.
    """
    cells, by_cards, by_n = {}, {}, {}
    for r in records:
        losses = [log_loss(r, w) / len(r['options']) for w in GRID]
        for store, key in ((cells, (r['cards'], r['n'])), (by_cards, r['cards']), (by_n, r['n'])):
            entry = store.setdefault(key, [0, [0.0] * len(GRID)])
            entry[0] += 1
            entry[1] = [a + b for a, b in zip(entry[1], losses)]
    table = {}
    for (cards, n), (count, sums) in cells.items():
        row = [s / by_cards[cards][0] for s in by_cards[cards][1]]
        column = [s / by_n[n][0] for s in by_n[n][1]]
        total = [sums[i] + shrink * 0.5 * (row[i] + column[i]) for i in range(len(GRID))]
        table[(cards, n)] = GRID[total.index(min(total))]
    return table


def table_share(table, default=V2_SHARE):
    return lambda r: table.get((r['cards'], r['n']), default)


def _fmt(pair):
    return f'{pair[0]:+.3f}+/-{1.96 * pair[1]:.3f}'


def fit(args):
    meta, records = load(args.records)
    print(f'{len(records)} bid positions, {meta}')
    groups = (
        ('all', lambda r: True),
        *((f'{c} cards', lambda r, c=c: r['cards'] == c) for c in range(1, 11)),
        *((f'{n} players', lambda r, n=n: r['n'] == n) for n in range(3, 9)),
        ('first to bid', lambda r: r['before'] == 0),
        ('later bidder', lambda r: r['before'] > 0),
        ('1-3 cards, 3-4 players', lambda r: r['cards'] <= 3 and r['n'] <= 4),
        ('1-3 cards, 5+ players', lambda r: r['cards'] <= 3 and r['n'] >= 5),
        ('7+ cards', lambda r: r['cards'] >= 7),
    )
    base = lambda r: V2_SHARE
    report = {'meta': meta, 'positions': len(records), 'groups': {}}
    print('\nPoints per bid against v2 (network share 0.5), +/- 95%. Log loss per option.')
    print(f"{'group':<24}{'bids':>7}{'deals':>7}  {'sim only':>16}  {'net only':>16}"
          f"  {'ll sim':>7}{'ll v2':>7}{'ll net':>7}  best share (ll)")
    for name, keep in groups:
        rows = [r for r in records if keep(r)]
        if not rows:
            continue
        sim_only = paired(rows, lambda r: 0.0, base)
        net_only = paired(rows, lambda r: 1.0, base)
        options = sum(len(r['options']) for r in rows)
        loss = {w: sum(log_loss(r, w) for r in rows) / options for w in GRID}
        best = min(loss, key=loss.get)
        samples = sum(r['samples'] for r in rows) / len(rows)
        print(f'{name:<24}{len(rows):>7}{samples:>7.0f}  {_fmt(sim_only):>16}  {_fmt(net_only):>16}'
              f'  {loss[0.0]:>7.4f}{loss[0.5]:>7.4f}{loss[1.0]:>7.4f}  {best:.2f}')
        report['groups'][name] = {
            'bids': len(rows), 'mean_sim_deals': samples,
            'sim_only_vs_v2': sim_only, 'net_only_vs_v2': net_only,
            'log_loss': {str(w): loss[w] for w in GRID}, 'best_share_log_loss': best}

    # Fit on some rounds, judge on the rest; then the other way round.
    print('\nFitted share table, judged on rounds it was not fitted on (points per bid vs v2):')
    folds = args.folds
    held = []
    for fold in range(folds):
        train = [r for r in records if r['round'] % folds != fold]
        test = [r for r in records if r['round'] % folds == fold]
        table = fit_table(train, args.shrink)
        share = table_share(table)
        held += [(r, share(r)) for r in test]
    fitted_of = dict((id(r), w) for r, w in held)
    fitted = lambda r: fitted_of[id(r)]
    report['held_out'] = {}
    for name, keep in groups:
        rows = [r for r in records if keep(r)]
        if not rows:
            continue
        gain = paired(rows, fitted, base)
        versus_sim = paired(rows, fitted, lambda r: 0.0)
        print(f'  {name:<24}{len(rows):>7}  vs v2 {_fmt(gain):>16}   vs sim only {_fmt(versus_sim):>16}')
        report['held_out'][name] = {'vs_v2': gain, 'vs_sim_only': versus_sim}

    table = fit_table(records, args.shrink)
    print('\nNetwork share fitted on every round (rows: cards, columns: players 3-8)')
    for cards in range(1, 11):
        print(f'  {cards:>2}  ' + '  '.join(
            f"{table[(cards, n)]:.2f}" if (cards, n) in table else '  - ' for n in range(3, 9)))
    report['table'] = {f'{cards},{n}': w for (cards, n), w in sorted(table.items())}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1))
    if args.write:
        args.write.write_text(json.dumps({
            'what': 'bid network share by "cards,players"; 0 is the simulation alone',
            'fitted_on': str(args.records), 'time_ms': meta.get('time_ms'),
            'positions': len(records), 'shrink': args.shrink, 'default': V2_SHARE,
            'share': report['table']}, indent=1))
        print(f'Wrote {args.write}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='command', required=True)
    c = commands.add_parser('collect')
    c.add_argument('--rounds', type=int, default=24000)
    c.add_argument('--time-ms', type=float, default=10)
    c.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    c.add_argument('--first', type=int, default=7_000_000, help='first round seed')
    c.add_argument('--out', type=Path, required=True)
    f = commands.add_parser('fit')
    f.add_argument('records', type=Path)
    f.add_argument('--shrink', type=float, default=200,
                   help='pseudo-positions each cell borrows from its row and column')
    f.add_argument('--folds', type=int, default=5)
    f.add_argument('--report', type=Path, default=None)
    f.add_argument('--write', type=Path, default=None, help='save the fitted share table')
    args = parser.parse_args(argv)
    (collect if args.command == 'collect' else fit)(args)


if __name__ == '__main__':
    main()
