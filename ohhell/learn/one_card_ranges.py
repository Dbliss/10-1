"""Build the one-card range table used by the ``ranges`` bot.

    python -m ohhell.learn.one_card_ranges --workers 12

Writes ``ohhell/models/one_card_ranges.json``. Masks are over canonical cards
(trump relabelled to clubs, its partner to spades).
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

from ..bots.ranges import TABLE_FILE, TABLE_SCORING, build_table
from ..cards import NUM_RANKS

PLAYERS = range(2, 9)


def _job(key):
    players, rank = key
    return key, build_table(players, rank)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    start = time.perf_counter()
    jobs = [(n, r) for n in reversed(PLAYERS) for r in range(NUM_RANKS)]
    tables = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for (players, rank), table in pool.map(_job, jobs):
            tables.setdefault(str(players), {})[str(rank)] = table
    scoring = TABLE_SCORING
    TABLE_FILE.write_text(json.dumps({
        'description': 'One-card bid ranges; see ohhell/bots/ranges.py.',
        'scoring': [scoring.points_per_trick, scoring.made_zero, scoring.penalty_per_trick],
        'tables': {n: tables[n] for n in sorted(tables, key=int)},
    }, separators=(',', ':')))
    print(f'Wrote {TABLE_FILE} in {time.perf_counter() - start:.0f} s')


if __name__ == '__main__':
    main()
