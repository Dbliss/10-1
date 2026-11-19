"""How low could the bid log loss go? Two reference forecasters for v2's bids.

The bid network scores about 0.99 log loss on the bid-blend screen. Zero is not
reachable: at bid time the hidden cards, the later bids and the play are
unknown. This tool measures where the floor lies, for v2's own chosen bid, at
the tournament's tables and lobby (the same setup as ``bid_blend_fit``).

At each position v2 bids for real. Then, from the position just after that
bid:

- ``actual``: one real playout. Every forecaster is scored on this outcome.
- ``oracle`` (lower limit): K more playouts of the *same* deal with freshly
  seeded bots. Knows every hand; only the bots' play-out randomness is left.
  No forecaster with less information can do better on average.
- ``reader`` (the floor for our seat, approximately): K deals drawn seat by
  seat in bidding order. Each earlier bidder's hand is redrawn from the cards
  left until that seat's real bot, freshly seeded, makes its real bid; the
  rest is dealt at random and the real bots play it out. This is close to the
  Bayes forecaster for a seat that knows every opponent's style perfectly.
  It is approximate because a later seat's bid does not reweight an earlier
  seat's hand (card removal is the only link it keeps).
- ``exact`` (check on ``reader``): whole deals redrawn and kept only when
  every earlier bidder reproduces its real bid. Exact but often fails with
  several bids made, so it is compared with ``reader`` only where it reached
  K deals.
- ``blind`` (an achievable upper reference): K deals with the hidden hands
  resampled with no regard to the earlier bids, the real bids kept on the
  table, played out by the real bots. Ignores what the bids say about hands.

Also kept: the network's and v2's blended trick odds for the chosen bid.

Sample forecasts use add-``alpha`` smoothing over 0..hand size. With K deals
the estimate is biased upwards; ``report`` also gives K/2 and the 1/K
extrapolation ``2 L(K) - L(K/2)``.

    python -m ohhell.learn.bid_floor collect --positions 2000 --deals 24 --out runs/bid_floor/t10.jsonl
    python -m ohhell.learn.bid_floor report runs/bid_floor/t10.jsonl
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
from ..cards import mask_of
from ..engine import BIDDING, DONE, NO_BID, RoundState
from ..tournament import BASELINE, CANDIDATE, build_lobby, rules_for, table_size

FLOOR = 1e-3


def _finish(state, bots):
    while state.phase != DONE:
        seat = state.to_act
        state.apply(bots[seat].act(state.observation(seat)))
    return state


def _copy(state):
    branch = state.clone()
    branch.history = []  # clone drops it; nothing has been played yet
    return branch


def _fresh(specs, members, seed):
    return [make_bot(specs[m], seed + seat) for seat, m in enumerate(members)]


def _unseen(our_hand, trump):
    return [c for c in range(52) if c != trump and not (our_hand >> c) & 1]


def _fill(fixed, pool, n, hand_size, first=None):
    """Deal ``pool`` (already shuffled) to the seats not in ``fixed``;
    ``first`` takes the first ``hand_size`` cards."""
    hands, at = [None] * n, 0
    for seat in ([first] if first is not None else []) + list(range(n)):
        if hands[seat] is None:
            if seat in fixed:
                hands[seat] = fixed[seat]
            else:
                hands[seat] = mask_of(pool[at:at + hand_size])
                at += hand_size
    return hands


def _resample(rng, n, hand_size, ours, our_hand, trump):
    pool = _unseen(our_hand, trump)
    rng.shuffle(pool)
    return _fill({ours: our_hand}, pool, n, hand_size)


def _seat_by_seat(rng, rules, hand_size, ours, our_hand, trump, dealer, order, bids, bots, max_tries):
    """Hidden hands drawn one earlier bidder at a time, each kept once that
    seat's bot reproduces its real bid. None if ``max_tries`` runs out."""
    n = rules.num_players
    pool = _unseen(our_hand, trump)
    fixed = {ours: our_hand}
    tries = 0
    for i, seat in enumerate(order):
        while True:
            if tries >= max_tries:
                return None, tries
            tries += 1
            rng.shuffle(pool)
            hands = _fill(fixed, pool, n, hand_size, first=seat)
            world = RoundState(rules, hands, trump, dealer)
            for earlier in order[:i]:
                world.apply(bids[earlier])
            if bots[seat].act(world.observation(seat)) == bids[seat]:
                fixed[seat] = hands[seat]
                pool = pool[hand_size:]
                break
    rng.shuffle(pool)
    return _fill(fixed, pool, n, hand_size), tries


def _position(number, specs, deals, max_tries, exact_tries):
    rng = random.Random(number)
    n = table_size(number)
    rules = rules_for(n)
    members = [CANDIDATE, BASELINE] + rng.sample(range(BASELINE + 1, len(specs)), n - 2)
    rng.shuffle(members)
    bots = [make_bot(specs[m], number * 31 + seat) for seat, m in enumerate(members)]
    ours = members.index(CANDIDATE)
    hand_size = rng.randint(1, rules.max_hand)
    hands, trump = deal_round(rng, n, hand_size)
    dealer = rng.randrange(n)
    state = RoundState(rules, hands, trump, dealer)
    order = []
    while state.phase == BIDDING and state.to_act != ours:
        seat = state.to_act
        order.append(seat)
        state.apply(bots[seat].act(state.observation(seat)))
    bot = bots[ours]
    obs = state.observation(ours)
    if len(obs.legal_actions()) < 2:
        return None

    captured = {}
    analyse = bot.analyse_bids

    def capture(o):
        captured['analysis'] = analyse(o)
        return captured['analysis']
    bot.analyse_bids = capture
    bid = bot.act(obs)
    bot.analyse_bids = analyse
    blended = captured['analysis'].distributions.get(bid) if 'analysis' in captured else None
    net = bot.models.bid_distributions(
        obs.hand, obs.trump_card, obs.n, obs.hand_size, obs.seat, obs.dealer,
        obs.bids, [bid])[bid]

    after = _copy(state)
    after.apply(bid)
    actual = _finish(_copy(after), bots).tricks_won[ours]
    seed = number * 1009 + 7

    oracle = []
    for i in range(deals):
        oracle.append(_finish(_copy(after), _fresh(specs, members, seed + 100 * i)).tricks_won[ours])

    sampler = random.Random(seed)
    blind = []
    for i in range(deals):
        world = RoundState(rules, _resample(sampler, n, hand_size, ours, obs.hand, trump), trump, dealer)
        for seat in order:
            world.apply(state.bids[seat])
        world.apply(bid)
        blind.append(_finish(world, _fresh(specs, members, seed + 100 * (deals + i))).tricks_won[ours])

    reader, reader_tries = [], 0
    for i in range(deals):
        fresh = _fresh(specs, members, seed + 100 * (2 * deals + i))
        hands, used = _seat_by_seat(sampler, rules, hand_size, ours, obs.hand, trump, dealer,
                                    order, state.bids, fresh, max_tries)
        reader_tries += used
        if hands is None:
            continue
        world = RoundState(rules, hands, trump, dealer)
        for seat in order:
            world.apply(state.bids[seat])
        world.apply(bid)
        reader.append(_finish(world, fresh).tricks_won[ours])

    exact, tries = [], 0
    while len(exact) < deals and tries < exact_tries:
        world = RoundState(rules, _resample(sampler, n, hand_size, ours, obs.hand, trump), trump, dealer)
        fresh = _fresh(specs, members, seed + 100 * (3 * deals + tries))
        tries += 1
        for seat in order:
            if fresh[seat].act(world.observation(seat)) != state.bids[seat]:
                break
            world.apply(state.bids[seat])
        else:
            world.apply(bid)
            exact.append(_finish(world, fresh).tricks_won[ours])

    return {
        'round': number, 'n': n, 'cards': hand_size, 'before': len(order), 'bid': bid,
        'net': net, 'v2': blended, 'actual': actual,
        'oracle': oracle, 'blind': blind, 'reader': reader, 'reader_tries': reader_tries,
        'exact': exact, 'exact_tries': tries,
    }


def _chunk(job):
    first, count, time_ms, deals, max_tries, exact_tries = job
    specs = [spec for _, _, spec in build_lobby(DEFAULT_SPEC, DEFAULT_SPEC, time_ms, 'stand-in')]
    out = []
    for number in range(first, first + count):
        started = time.perf_counter()
        record = _position(number, specs, deals, max_tries, exact_tries)
        if record:
            record['seconds'] = round(time.perf_counter() - started, 2)
            out.append(record)
    return out


def collect(args):
    args.out.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    chunk = args.chunk
    jobs = [(args.first + start, min(chunk, args.positions - start), args.time_ms, args.deals, args.max_tries,
             args.exact_tries)
            for start in range(0, args.positions, chunk)]
    started = time.monotonic()
    done = 0
    with args.out.open('w') as output, ProcessPoolExecutor(max_workers=args.workers) as pool:
        output.write(json.dumps({'meta': {
            'positions': args.positions, 'first': args.first, 'time_ms': args.time_ms,
            'deals': args.deals, 'max_tries': args.max_tries, 'exact_tries': args.exact_tries, 'spec': DEFAULT_SPEC,
            'workers': args.workers}}) + '\n')
        for future in as_completed([pool.submit(_chunk, job) for job in jobs]):
            for record in future.result():
                output.write(json.dumps(record) + '\n')
            output.flush()
            done += 1
            print(f'{done}/{len(jobs)} chunks, {time.monotonic() - started:.0f}s', flush=True)


def sample_loss(outcomes, actual, cards, alpha):
    if not outcomes:
        return None
    count = sum(1 for k in outcomes if k == actual)
    return -math.log((count + alpha) / (len(outcomes) + alpha * (cards + 1)))


def dist_loss(dist, actual):
    return -math.log(max(dist[actual], FLOOR))


def mean_ci(values):
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, float('nan')
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, 1.96 * math.sqrt(var / len(values))


def report(args):
    meta, records = {}, []
    for line in Path(args.records).read_text().splitlines():
        row = json.loads(line)
        if 'meta' in row:
            meta = row['meta']
        else:
            records.append(row)
    deals = meta.get('deals', 24)
    half = deals // 2
    short = sum(len(r['reader']) < deals for r in records)
    print(f'{len(records)} positions, {meta}')
    print(f'reader short of {deals} deals at {short} positions; exact reached {deals} at '
          f'{sum(len(r["exact"]) >= deals for r in records)}; mean seconds per position '
          f'{sum(r["seconds"] for r in records) / len(records):.1f}')
    print(f'Log loss of the actual outcome, +/- 95%. Sample forecasts: K={deals} deals, '
          f'add-{args.alpha} smoothing; (K/2) and ~1/K extrapolation alongside.')

    def loss(r, key, k=None):
        if key in ('net', 'v2'):
            return dist_loss(r[key], r['actual']) if r[key] else None
        outcomes = r[key][:k] if k else r[key]
        return sample_loss(outcomes, r['actual'], r['cards'], args.alpha)

    def row(label, rows, key):
        vals = [v for v in (loss(r, key) for r in rows) if v is not None]
        if not vals:
            return
        m, ci = mean_ci(vals)
        text = f'  {label:<10}{m:7.3f} +/- {ci:.3f}'
        if key not in ('net', 'v2'):
            h = [v for v in (loss(r, key, half) for r in rows) if v is not None]
            hm = sum(h) / len(h)
            text += f'   ({hm:.3f})  ~{2 * m - hm:.3f}'
        print(text)

    def gap(label, rows, key):
        diffs = [loss(r, 'net') - loss(r, key) for r in rows if loss(r, key) is not None]
        m, ci = mean_ci(diffs)
        print(f'  network - {label:<7}{m:+7.3f} +/- {ci:.3f}')

    groups = [('all', lambda r: True)]
    groups += [(f'{c} cards', lambda r, c=c: r['cards'] == c) for c in range(1, 11)]
    groups += [(f'{n} players', lambda r, n=n: r['n'] == n) for n in range(3, 9)]
    groups += [('first to bid', lambda r: r['before'] == 0), ('later bidder', lambda r: r['before'] > 0)]
    for name, keep in groups:
        rows = [r for r in records if keep(r)]
        if not rows:
            continue
        print(f'\n== {name}: {len(rows)} positions')
        for label, key in (('network', 'net'), ('v2 blend', 'v2'), ('blind', 'blind'),
                           ('reader', 'reader'), ('oracle', 'oracle')):
            row(label, rows, key)
        gap('reader', rows, 'reader')
        gap('oracle', rows, 'oracle')
        checked = [r for r in rows if len(r['exact']) >= deals and r['before'] > 0]
        if checked:
            print(f'  -- exact check, {len(checked)} positions with bids before ours')
            for label, key in (('network', 'net'), ('reader', 'reader'), ('exact', 'exact')):
                row(label, checked, key)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--positions', type=int, default=2000)
    c.add_argument('--first', type=int, default=7300000)
    c.add_argument('--time-ms', type=float, default=10.0)
    c.add_argument('--deals', type=int, default=24)
    c.add_argument('--max-tries', type=int, default=200,
                   help='bid tries per reader deal, over all earlier seats')
    c.add_argument('--exact-tries', type=int, default=100, help='whole-deal tries for the exact check')
    c.add_argument('--chunk', type=int, default=5)
    c.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    c.add_argument('--out', type=Path, required=True)
    r = sub.add_parser('report')
    r.add_argument('records', type=Path)
    r.add_argument('--alpha', type=float, default=0.25)
    args = parser.parse_args(argv)
    collect(args) if args.command == 'collect' else report(args)


if __name__ == '__main__':
    main()
