"""Fit the bid sigma that inference uses, from the tournament lobby's own bids.

Inference weights a sampled deal by how well each earlier bid fits the quick
trick estimate of that seat's sampled hand: a Gaussian with ``bid_sigma``
tricks of spread. That spread was set by hand (1.4) for the original rule of
thumb. Here, rounds are dealt at the tournament's table sizes and every seat
bids with a tournament lobby member (v2, its leaning versions, and the
human-like players, drawn as the tournament draws them; the candidate's own
seat is played by v2 as a stand-in). The bids and hands are saved, and the
maximum-likelihood sigma for each estimate version is the root mean square of
``bid - estimate`` over the seats a candidate would be judging.

    python -m ohhell.learn.bid_sigma --rounds 4000 --time-ms 10 --out runs/estimate3/lobby_bids.json
"""

import argparse
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .. import estimate
from ..arena import deal_round
from ..bots import DEFAULT_SPEC, make_bot
from ..bots.heuristic import expected_tricks
from ..engine import BIDDING, RoundState
from ..tournament import BASELINE, CANDIDATE, build_lobby, rules_for, table_size


def _bids(job):
    first, count, time_ms = job
    lobby = build_lobby(DEFAULT_SPEC, DEFAULT_SPEC, time_ms, 'stand-in')
    specs = [spec for _, _, spec in lobby]
    records = []
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
        while state.phase == BIDDING:
            seat = state.to_act
            state.apply(bots[seat].act(state.observation(seat)))
        for seat in range(n):
            if members[seat] != CANDIDATE:
                records.append({'n': n, 'hand_size': hand_size, 'hand': hands[seat],
                                'trump_card': trump, 'dealer': state.dealer, 'seat': seat,
                                'bid': state.bids[seat], 'member': lobby[members[seat]][1]})
    return records


def collect(rounds, time_ms, workers, first=0):
    chunk = max(1, rounds // (workers * 4))
    jobs = [(first + start, min(chunk, rounds - start), time_ms)
            for start in range(0, rounds, chunk)]
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [r for part in pool.map(_bids, jobs) for r in part]


def residuals(records, version):
    out = []
    for r in records:
        position = estimate.first_trick_position(r['seat'], r['dealer'], r['n'])
        if version == 1:
            guess = expected_tricks(r['hand'], r['trump_card'], r['n'], r['hand_size'],
                                    position == 0)
        else:
            guess = estimate.expected_tricks(r['hand'], r['trump_card'], r['n'],
                                             r['hand_size'], position, version=version)
        out.append(r['bid'] - guess)
    return out


def summary(records, version):
    gaps = residuals(records, version)
    count = len(gaps)
    by_member = {}
    for r, gap in zip(records, gaps):
        by_member.setdefault(r['member'], []).append(gap)
    return {
        'version': version, 'bids': count,
        'mean_gap': sum(gaps) / count,
        'sigma': math.sqrt(sum(g * g for g in gaps) / count),
        'by_member': {m: {'bids': len(g), 'mean_gap': sum(g) / len(g),
                          'rms': math.sqrt(sum(x * x for x in g) / len(g))}
                      for m, g in sorted(by_member.items())},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--rounds', type=int, default=4000)
    parser.add_argument('--time-ms', type=float, default=10)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--first', type=int, default=5_000_000, help='first round seed')
    parser.add_argument('--out', type=Path, required=True,
                        help='JSON file for the bids (reused if it exists)')
    parser.add_argument('--versions', default='1,2,3')
    args = parser.parse_args(argv)

    if args.out.exists():
        data = json.loads(args.out.read_text())
        records = data['records']
    else:
        records = collect(args.rounds, args.time_ms, args.workers, args.first)
        data = {'rounds': args.rounds, 'time_ms': args.time_ms, 'first': args.first,
                'baseline': DEFAULT_SPEC, 'records': records}
    data['fits'] = [summary(records, int(v)) for v in args.versions.split(',')]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data))
    for fit in data['fits']:
        print(f"estimate {fit['version']}: {fit['bids']} bids, mean bid - estimate "
              f"{fit['mean_gap']:+.3f}, sigma {fit['sigma']:.3f}")
        for member, row in fit['by_member'].items():
            print(f"    {member:10} {row['bids']:>6}  gap {row['mean_gap']:+.3f}  rms {row['rms']:.3f}")


if __name__ == '__main__':
    main()
