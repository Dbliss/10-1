"""Measure completed play samples and elapsed time; never infer strength."""

import argparse
from pathlib import Path
import random
import time

from .. import native
from ..arena import deal_round
from ..bots import make_bot
from ..bots.heuristic import state_card
from ..engine import BIDDING, PLAYING, RoundState
from ..rules import Rules
from .cases import provenance, write_json


def measure(spec, seed=20261113, positions=24):
    if not native.available:
        raise RuntimeError('Build native core with python -m ohhell.native')
    rng = random.Random(seed)
    rows = []
    while len(rows) < positions:
        n = (3, 4, 5, 6, 7, 8)[len(rows) % 6]
        size = min(10, 51 // n)
        hands, trump = deal_round(rng, n, size)
        state = RoundState(Rules(n, size), hands, trump, rng.randrange(n))
        while state.phase == BIDDING:
            state.apply(make_bot('heuristic').act(state.observation(state.to_act)))
        for _ in range((len(rows) // 6) % 4 * n):
            if state.phase == PLAYING:
                state.play_unchecked(state_card(state))
        obs = state.observation(state.to_act)
        if len(obs.legal_actions()) < 2:
            continue
        bot = make_bot(spec, seed + len(rows))
        if not getattr(bot, 'simulator', None):
            raise ValueError('Diagnostics require a native simulation bot')
        started = time.perf_counter()
        action = bot.act(obs)
        rows.append({'players': n, 'hand_size': size, 'held': obs.hand.bit_count(),
                     'legal': len(obs.legal_actions()), 'action': action,
                     'elapsed_ms': (time.perf_counter() - started) * 1000,
                     'counted_worlds': bot.last_card_worlds,
                     'executed_worlds': bot.simulator.last_card_playouts})
    return {'spec': provenance(spec), 'seed': seed, 'positions': rows,
            'native_library': str(native.LIBRARY),
            'mean_ms': sum(r['elapsed_ms'] for r in rows) / len(rows),
            'mean_counted_worlds': sum(r['counted_worlds'] for r in rows) / len(rows),
            'mean_executed_worlds': sum(r['executed_worlds'] for r in rows) / len(rows),
            'limitations': 'Isolated card turns only, including Python inference and sampling. '
            'Not game strength, independent samples, a stable benchmark, or a per-playout timer.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('specs', nargs='+')
    parser.add_argument('--positions', type=int, default=24)
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.positions < 1 or Path(args.out).exists():
        parser.error('positive positions and a fresh output filename required')
    reports = [measure(spec, args.seed, args.positions) for spec in args.specs]
    write_json(args.out, reports)
    for report in reports:
        print(f"{report['spec']['spec']}: {report['mean_ms']:.3f} ms, "
              f"{report['mean_counted_worlds']:.1f} counted / "
              f"{report['mean_executed_worlds']:.1f} executed worlds")


if __name__ == '__main__':
    main()
