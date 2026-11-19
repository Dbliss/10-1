"""How a bot fares in over- and under-called rounds.

The call of a round is the total of the bids minus the tricks available:
above zero the table wants more tricks than exist, below zero fewer. Each
bot named takes seat 0 of the same games against the same opponents, so the
tables can be compared and the points differ only through that seat's play.

    python -m ohhell.learn.call_report heuristic "heuristic:style=7" --games 2000

This describes play. It is not the tournament and decides nothing.
"""

import argparse
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor

from ..arena import deal_round
from ..bots import DEFAULT_SPEC, make_bot
from ..engine import DONE, RoundState
from ..rules import Rules

SPAN = 3  # calls beyond +/- SPAN are counted with +/- SPAN


def play(job):
    """One game. Returns (seat 0's points, [(hand size, call, bid, tricks, points)])."""
    spec, opponents, rules, game = job
    n = rules.num_players
    bots = [make_bot(spec, game * 1009 + 100)] + [
        make_bot(opponents[(seat - 1) % len(opponents)], game * 1009 + seat)
        for seat in range(1, n)]
    rng = random.Random(game)  # deals depend only on the seed, as in the arena
    dealer = rng.randrange(n)
    total = 0
    rounds = []
    for hand_size in rules.hand_sizes:
        hands, trump = deal_round(rng, n, max(1, hand_size))
        state = RoundState(rules, hands, trump, dealer, blind=hand_size == 0)
        while state.phase != DONE:
            state.apply(bots[state.to_act].act(state.observation(state.to_act)))
        points = state.points()[0]
        total += points
        rounds.append((hand_size, sum(state.bids) - max(1, hand_size), state.bids[0],
                       state.tricks_won[0], points))
        dealer = (dealer + 1) % n
    return total, rounds


def tabulate(games, min_hand):
    """Per call: [rounds, made, too many, too few, points, tricks minus bid]."""
    rows = {}
    for _, rounds in games:
        for hand_size, call, bid, tricks, points in rounds:
            if hand_size < min_hand:
                continue
            row = rows.setdefault(max(-SPAN, min(SPAN, call)), [0, 0, 0, 0, 0, 0])
            row[0] += 1
            row[1] += tricks == bid
            row[2] += tricks > bid
            row[3] += tricks < bid
            row[4] += points
            row[5] += tricks - bid
    return rows


def interval(values):
    mean = sum(values) / len(values)
    spread = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
    return mean, 1.96 * math.sqrt(spread / len(values))


def report(specs, opponents, rules, games, first_seed=0, workers=None):
    """Play every spec on the same games. Returns {spec: [(total, rounds), ...]}."""
    jobs = [(spec, opponents, rules, first_seed + g) for spec in specs for g in range(games)]
    if workers == 1:
        results = [play(job) for job in jobs]
    else:
        os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(play, jobs, chunksize=max(1, len(jobs) // 200)))
    return {spec: results[i * games:(i + 1) * games] for i, spec in enumerate(specs)}


def show(results, min_hand):
    first = None
    for spec, games in results.items():
        totals = [total for total, _ in games]
        mean, margin = interval(totals)
        print(f'\n{spec}: {mean:.1f} +/- {margin:.1f} points per game, {len(games)} games')
        if first is None:
            first = totals
        else:
            edge, margin = interval([a - b for a, b in zip(totals, first)])
            print(f'  against the first bot on the same games: {edge:+.2f} +/- {margin:.2f}')
        print(f'  rounds of {min_hand}+ cards')
        print('  call   rounds   made   too many   too few   points/round   tricks-bid')
        rows = tabulate(games, min_hand)
        for call in sorted(rows):
            count, made, over, under, points, gap = rows[call]
            edge = '<=' if call == -SPAN else '>=' if call == SPAN else '  '
            print(f'  {edge}{call:+d}  {count:7d}  {made / count:5.1%}  {over / count:9.1%}'
                  f'  {under / count:8.1%}  {points / count:13.2f}  {gap / count:+11.2f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('specs', nargs='*', default=[DEFAULT_SPEC],
                        help='bots to put in seat 0, one table each (default: the current bot)')
    parser.add_argument('--opponents', default='skilled,casual,novice',
                        help='the other seats, repeated in order to fill the table; '
                             'use ";" between specs that contain commas')
    parser.add_argument('--games', type=int, default=200)
    parser.add_argument('--players', type=int, default=4)
    parser.add_argument('--hand', type=int, default=10)
    parser.add_argument('--min-hand', type=int, default=3,
                        help='smallest hand size shown in the tables')
    parser.add_argument('--first-seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args()
    separator = ';' if ';' in args.opponents else ','
    opponents = [spec for spec in args.opponents.split(separator) if spec]
    rules = Rules(args.players, args.hand)
    print(f'{args.players} players, opening hand {args.hand}, seeds from {args.first_seed}, '
          f'seat 0 against {", ".join(opponents)}')
    show(report(args.specs, opponents, rules, args.games, args.first_seed, args.workers),
         args.min_hand)


if __name__ == '__main__':
    main()
