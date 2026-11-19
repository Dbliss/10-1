"""Measure the current game's bid shortlist against an all-legal search."""

import argparse
import json
import math
import multiprocessing
from pathlib import Path
import statistics

from ..bots import make_bot
from ..duel import FIELD, REFERENCE_SPEC, draw_round, extend, rounds_per_game
from ..engine import RoundState
from ..tournament import check_run_config


def audit_position(job):
    seed, worlds = job
    n, rules, label, hands, trump, dealer, seat, opponents, seeds = draw_round(seed, FIELD)
    row = {'seed': seed, 'players': n, 'cards': label}
    if label == 0:
        return row  # Blind bids already use exact odds.

    state = RoundState(rules, list(hands), trump, dealer)
    while state.to_act != seat:
        bidder = state.to_act
        bot = make_bot(opponents[bidder], seeds[bidder])
        state.apply(bot.act(state.observation(bidder)))
    obs = state.observation(seat)
    spec = extend(REFERENCE_SPEC, f'bid_budget={worlds}')
    bot = make_bot(spec, seeds[seat])
    legal = obs.legal_actions()
    shortlist = bot._bid_options(obs)
    row.update(legal=legal, shortlist=shortlist)
    if shortlist == legal:
        return row

    # Rank both sets from one analysis with the same sampled deals and blend.
    bot._bid_options = lambda _: legal
    values = bot.analyse_bids(obs).values
    best = max(legal, key=values.get)
    shortlisted = max(shortlist, key=values.get)
    row.update(best=best, shortlisted=shortlisted, values=values,
               samples=bot.simulator.last_bid_samples)
    if best not in shortlist and values[best] > values[shortlisted]:
        # Score the nominated pair on a second stream to check how much of
        # the estimated advantage survives independent draws.
        verify = make_bot(spec, seeds[seat] + (1 << 30))
        verify._bid_options = lambda _: sorted([best, shortlisted])
        held_out = verify.analyse_bids(obs).values
        row['verify_gap'] = held_out[best] - held_out[shortlisted]
    return row


def summarize(rows):
    misses = [row for row in rows if 'verify_gap' in row]
    gaps = [row['values'][row['best']] - row['values'][row['shortlisted']]
            for row in misses]
    validated = [row.get('verify_gap', 0.0) * rounds_per_game(row['players']) for row in rows]
    return {
        'positions': len(rows),
        'pruned': sum(row.get('legal') != row.get('shortlist') for row in rows),
        'misses': len(misses),
        'above_shortlist': sum(row['best'] > max(row['shortlist']) for row in misses),
        'confirmed_positive': sum(row['verify_gap'] > 0 for row in misses),
        'mean_gap_on_misses': statistics.mean(gaps) if gaps else 0.0,
        'max_gap': max(gaps, default=0.0),
        'verified_points_per_game': statistics.mean(validated),
        'verified_margin_95': 1.96 * statistics.stdev(validated) / math.sqrt(len(rows)) if len(rows) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--rounds', type=int, required=True)
    parser.add_argument('--worlds', type=int, default=8192)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if min(args.rounds, args.worlds, args.workers) < 1:
        parser.error('rounds, worlds and workers must be positive')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = args.out.with_suffix('.meta')
    if any(path.exists() for path in (args.out, manifest, args.out.with_suffix('.json'))):
        parser.error('output already exists')
    args.max_sets, args.smallest_gain, args.time_ms = args.rounds, 2.0, 0
    args.resume = False
    check_run_config(manifest, args, [extend(REFERENCE_SPEC, f'bid_budget={args.worlds}'),
                                    *(spec for spec, _ in FIELD)])
    jobs = ((args.seed + i, args.worlds) for i in range(args.rounds))
    rows = []
    with args.out.open('x') as output, multiprocessing.Pool(args.workers) as pool:
        for row in pool.imap(audit_position, jobs, chunksize=8):
            rows.append(row)
            output.write(json.dumps(row) + '\n')
            if len(rows) % 500 == 0:
                print(f'{len(rows):,}/{args.rounds:,} positions', flush=True)
    result = summarize(rows)
    args.out.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
