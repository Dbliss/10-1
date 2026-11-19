"""How many sampled deals does the exact endgame need? Points lost per decision.

For each endgame position (3-8 players, from 8 cards left up to the table's
``ENDGAME_BY_PLAYERS`` threshold) the bot's own sampler draws a long run of
deals, in the bot's block size, and every legal card is solved exactly in
each deal (noise-aware, as ``endgame_noise=1``). The run is then cut into
consecutive chunks of N deals: each chunk is one bot decision with N deals.
Its choice is scored against the mean over every OTHER deal (held out), so

    loss = reference value of the reference's best card
           - reference value of the chunk's choice   (points, this decision)

This measures only the cost of too few deals. The reference shares the
exact search's perfect-information bias, so it cannot say whether exact
search beats play-outs (a duel does that). A finite reference overstates
the loss (its own best card is picked on its noise), so the report also
gives a lower bound: the best card picked on half the held-out deals and
scored on the other half. The truth lies between the two. Positions come from heuristic bids and play, so bid reading is
better calibrated than against the real field.

Seconds per deal are measured under whatever load the machine has; run it
alone for 250 ms deal counts that match a duel.

Hard time cap: ``--minutes`` (at most 115). Workers stop starting deals a
few minutes before the cap, the pool is terminated at the cap, and results
are written every 30 seconds, so a stop loses at most the positions in
progress.

    python -m ohhell.learn.endgame_deals run --minutes 110 --workers 18
    python -m ohhell.learn.endgame_deals report runs/endgame_deals/<file>.json
"""

import argparse
from datetime import datetime, timezone
import math
import multiprocessing as mp
from pathlib import Path
import random
import time

from .. import native
from ..arena import deal_round
from ..bots import make_bot
from ..bots.heuristic import state_card
from ..bots.simulation import ENDGAME_BY_PLAYERS
from ..cards import FULL_DECK
from ..engine import BIDDING, PLAYING, RoundState
from ..opponents import KINDS
from ..rules import Rules
from .cases import provenance, write_json

SPEC = ('hybrid:field=mix,inference=1,shortlist=13,touching=1,bid_sigma=0.76,'
        'bid_blend=0.25,endgame_noise=1,endgame_table=1')
CHUNKS = (10, 25, 50, 100, 200, 400)
MIN_LEFT = 8
MAX_MINUTES = 115
STOP_EARLY = 180  # seconds before the cap that workers stop starting deals


def make_position(rng):
    """Heuristic bids and play to a random endgame point; None if no choice."""
    n = rng.choice((3, 4, 5, 6, 7, 8))
    limit = ENDGAME_BY_PLAYERS[n]
    target = rng.randint(MIN_LEFT, limit)
    largest = min(10, 51 // n)
    sizes = [s for s in range(2, largest + 1) if s * n >= target]
    if not sizes:
        return None
    size = rng.choice(sizes)
    hands, trump = deal_round(rng, n, size)
    state = RoundState(Rules(n, size), hands, trump, rng.randrange(n))
    bidder = make_bot('heuristic')
    while state.phase == BIDDING:
        state.apply(bidder.act(state.observation(state.to_act)))
    while state.phase == PLAYING:
        held = state.hands[state.to_act].bit_count()
        left = held * n - state.trick_size
        if left <= target:
            obs = state.observation(state.to_act)
            if len(obs.legal_actions()) > 1:
                return obs
        if left < MIN_LEFT:
            return None
        state.play_unchecked(state_card(state))
    return None


def chunk_losses(values, size):
    """Loss and agreement for each consecutive chunk of ``size`` deals."""
    import numpy as np
    deals = len(values)
    if deals < 2 * size:
        return None
    total = values.sum(axis=0)
    losses, lows, agree, ref_se = [], [], [], []
    for start in range(0, deals - size + 1, size):
        chunk = values[start:start + size]
        rest = np.concatenate((values[:start], values[start + size:]))
        reference = (total - chunk.sum(axis=0)) / len(rest)
        chosen = int(np.argmax(chunk.mean(axis=0)))
        best = int(np.argmax(reference))
        # Upper bound: the reference's own best card is picked on its noise.
        losses.append(float(reference[best] - reference[chosen]))
        # Lower bound: best card picked on one half, scored on the other.
        half = len(rest) // 2
        pick = int(np.argmax(rest[:half].mean(axis=0)))
        lows.append(float(rest[half:, pick].mean() - rest[half:, chosen].mean()))
        agree.append(int(chosen == best))
        gap = rest[:, best] - rest[:, chosen]
        ref_se.append(float(gap.std(ddof=1) / math.sqrt(len(rest))) if chosen != best else 0.0)
    count = len(losses)
    return {'loss': sum(losses) / count, 'loss_low': sum(lows) / count,
            'agree': sum(agree) / count, 'chunks': count, 'ref_se': sum(ref_se) / count}


def measure(job):
    """Solve one position's deals until the reference count or its time is up."""
    import numpy as np
    seed, spec, reference, position_seconds, deadline = job
    rng = random.Random(seed)
    obs = None
    while obs is None:
        obs = make_position(rng)
    bot = make_bot(spec, seed)
    sim = bot.simulator
    cards = obs.legal_actions()
    stop = min(time.time() + position_seconds, deadline)
    rows, spent = [], 0.0
    while len(rows) < reference and time.time() < stop:
        hands, kinds = sim.sample_worlds(obs, min(bot.worlds, reference - len(rows)))
        for h, k in zip(hands, kinds):
            noise = [KINDS[int(k[s])].card_noise if s < obs.n else 0.0
                     for s in range(len(k))]
            started = time.perf_counter()
            values = sim.solve_noisy_scores(obs, [int(x) for x in h], noise, cards)
            spent += time.perf_counter() - started
            rows.append([values[c] for c in cards])
            if time.time() >= stop:
                break
    values = np.array(rows, dtype=np.float64)
    seen = obs.hand | obs.played | (1 << obs.trump_card)
    held = obs.hand.bit_count()
    return {
        'seed': seed, 'players': obs.n, 'hand_size': obs.hand_size,
        'cards_left': held * obs.n - obs.trick_size,
        'opponent_cards': held * obs.n - obs.trick_size - held,
        'unseen': (FULL_DECK & ~seen).bit_count(),
        'legal': len(cards), 'deals': len(rows),
        'ms_per_deal': 1000.0 * spent / max(1, len(rows)),
        'complete': len(rows) >= reference,
        'by_deals': {str(size): chunk_losses(values, size) for size in CHUNKS},
    }


def run(args):
    if not native.available:
        raise RuntimeError('Build the native core first: python -m ohhell.native')
    if not 0 < args.minutes <= MAX_MINUTES:
        raise SystemExit(f'--minutes must be between 1 and {MAX_MINUTES}.')
    started = time.time()
    cap = started + 60 * args.minutes
    deadline = cap - min(STOP_EARLY, 60 * args.minutes / 4)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = Path(args.out or f'runs/endgame_deals/endgame_deals_{stamp}.json')
    meta = {
        'started': datetime.now(timezone.utc).isoformat(), 'spec': args.spec,
        'seed': args.seed, 'reference_deals': args.reference,
        'position_seconds': args.position_seconds, 'minutes': args.minutes,
        'workers': args.workers, 'chunks': CHUNKS, 'min_left': MIN_LEFT,
        'thresholds': ENDGAME_BY_PLAYERS, 'provenance': provenance(args.spec),
    }
    records, saved = [], time.time()

    def save(final=False):
        meta['finished' if final else 'updated'] = datetime.now(timezone.utc).isoformat()
        meta['elapsed_minutes'] = (time.time() - started) / 60
        write_json(out, {'meta': meta, 'positions': records})

    print(f'Writing {out}; hard stop at {datetime.fromtimestamp(cap):%H:%M}.', flush=True)
    pool = mp.Pool(args.workers)
    pending, next_seed = [], args.seed
    try:
        while True:
            now = time.time()
            if now >= cap:
                print('Time cap reached; stopping workers.', flush=True)
                break
            while now < deadline and len(pending) < 2 * args.workers and (
                    not args.positions or next_seed - args.seed < args.positions):
                pending.append(pool.apply_async(measure, ((
                    next_seed, args.spec, args.reference, args.position_seconds, deadline),)))
                next_seed += 1
            if not pending:
                break
            done = [p for p in pending if p.ready()]
            for p in done:
                pending.remove(p)
                records.append(p.get())
                if len(records) % 25 == 0:
                    print(f'{len(records)} positions, {(time.time() - started) / 60:.1f} min',
                          flush=True)
            if time.time() - saved > 30:
                save()
                saved = time.time()
            if not done:
                time.sleep(0.2)
    finally:
        pool.terminate()
        pool.join()
        save(final=True)
    print(f'{len(records)} positions in {(time.time() - started) / 60:.1f} min -> {out}')
    report_records(records)


BUCKETS = ((0, 8), (9, 12), (13, 16), (17, 20), (21, 52))


def report_records(records):
    def line(label, group, key):
        cells = []
        for size in CHUNKS:
            rows = [r['by_deals'][str(size)] for r in group if r['by_deals'].get(str(size))]
            if len(rows) < 5:
                cells.append(f'{"":>16}')
                continue
            losses = [r[key] for r in rows]
            mean = sum(losses) / len(losses)
            sd = math.sqrt(sum((x - mean) ** 2 for x in losses) / (len(losses) - 1))
            cells.append(f'{mean:7.3f}+/-{1.96 * sd / math.sqrt(len(losses)):5.3f} ')
        ms = sorted(r['ms_per_deal'] for r in group)
        median = ms[len(ms) // 2] if ms else 0.0
        deals250 = 250.0 / median if median else float('inf')
        print(f'{label:<20}{len(group):>6}{median:>9.2f}{deals250:>8.0f} ' + ''.join(cells))

    def table(key, title, full):
        header = ''.join(f'{"N=" + str(s):>16}' for s in CHUNKS)
        print(f'\n{title}')
        print(f'{"group":<20}{"pos":>6}{"ms/deal":>9}{"d@250":>8} {header}')
        line('all', records, key)
        for low, high in BUCKETS:
            line(f'opp cards {low}-{high}',
                 [r for r in records if low <= r['opponent_cards'] <= high], key)
        if not full:
            return
        for n in range(3, 9):
            group = [r for r in records if r['players'] == n]
            line(f'{n}p left <=12', [r for r in group if r['cards_left'] <= 12], key)
            line(f'{n}p left >12', [r for r in group if r['cards_left'] > 12], key)

    print('\nPoints lost per decision by deals N (mean +/- 95%). ms/deal: median exact')
    print('solve of all legal cards, under the load of this run; d@250 = 250 / ms/deal.')
    table('loss', 'UPPER bound (reference picks its own best card; overstates):', True)
    table('loss_low', 'LOWER bound (best card picked and scored on separate halves):', True)
    short = sum(not r['complete'] for r in records)
    print(f'\n{short} of {len(records)} positions stopped short of the reference count '
          f'(time cap per position); their large-N cells use fewer chunks or are blank.')


def report(args):
    import json
    data = json.loads(Path(args.path).read_text(encoding='utf-8'))
    print({k: data['meta'][k] for k in ('spec', 'reference_deals', 'minutes', 'workers')})
    report_records(data['positions'])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--spec', default=SPEC)
    r.add_argument('--minutes', type=float, default=110)
    r.add_argument('--workers', type=int, default=18)
    r.add_argument('--reference', type=int, default=2000, help='deals per position')
    r.add_argument('--position-seconds', type=float, default=240)
    r.add_argument('--positions', type=int, default=0, help='0: until the time cap')
    r.add_argument('--seed', type=int, default=20261119)
    r.add_argument('--out')
    p = sub.add_parser('report')
    p.add_argument('path')
    args = parser.parse_args()
    {'run': run, 'report': report}[args.command](args)


if __name__ == '__main__':
    main()
