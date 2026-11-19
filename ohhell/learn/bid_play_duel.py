"""Short paired screen of live card search inside the bid simulation."""

import argparse
import json
import math
import multiprocessing
from pathlib import Path
import random
import statistics
import time
from types import SimpleNamespace

import numpy as np

from ..bots import make_bot
from ..duel import FIELD, REFERENCE_SPEC, draw_round, extend, play, rounds_per_game
from ..engine import BIDDING
from ..tournament import check_run_config
from .bid_play_audit import _live_counts, _worlds


class TestSeat:
    def __init__(self, seed, live_bids, bid_ms, rollout_budget, card_budget):
        self.cards = make_bot(extend(REFERENCE_SPEC, f'card_budget={card_budget},endgame_budget={card_budget}'), seed)
        # Independent card RNG makes identical bids lead to identical play.
        self.bidder = make_bot(extend(REFERENCE_SPEC, f'time_ms={bid_ms:g}'), seed ^ (1 << 29))
        self.live_bids = live_bids
        self.bid_ms = bid_ms
        self.rollout_spec = extend(REFERENCE_SPEC, f'card_budget={rollout_budget},endgame_budget={rollout_budget}')
        self.rng = random.Random(seed ^ (1 << 30))
        self.bid = self.elapsed = None
        self.worlds = 0
        self.network_fallback = False

    def act(self, obs):
        if obs.phase != BIDDING:
            return self.cards.act(obs)
        started = time.perf_counter()
        if self.live_bids and not obs.blind:
            self.bid = self._live_bid(obs, started + self.bid_ms / 1000)
        else:
            self.bid = self.bidder.act(obs)
            if not obs.blind:
                self.worlds = self.bidder.simulator.last_bid_samples
        self.elapsed = (time.perf_counter() - started) * 1000
        return self.bid

    def _live_bid(self, obs, deadline):
        bot = self.bidder
        options = bot._bid_options(obs)
        if len(options) == 1:
            return options[0]
        prior = bot.models.bid_distributions(obs.hand, obs.trump_card, obs.n,
                                             obs.hand_size, obs.seat, obs.dealer,
                                             obs.bids, options)
        counts = np.zeros((len(options), obs.hand_size + 1))
        while time.perf_counter() < deadline:
            worlds = _worlds(obs, bot.simulator, options, 32, self.rng.randrange(1 << 30))
            block = _live_counts(obs, worlds, self.rollout_spec, self.rng.randrange(1 << 30),
                                 own_only=False, deadline=deadline)
            counts += block
            if block[0].sum() < 32:
                break
        self.worlds = int(counts[0].sum())
        # A complete world supplies one result for every candidate. Use the
        # existing network when the deadline allows no complete world.
        self.network_fallback = self.worlds == 0
        blend = 1.0 if self.network_fallback else bot.bid_share(obs.hand_size, obs.n)
        values = {}
        for b, tally in zip(options, counts):
            odds = (1 - blend) * tally / max(1, self.worlds) + blend * np.array(prior[b])
            values[b] = sum(p * obs.rules.scoring.points(b, k) for k, p in enumerate(odds))
        return max(values, key=values.get)


def duel_round(job):
    seed, bid_ms, rollout_budget, card_budget = job
    n, rules, label, hands, trump, dealer, seat, opponents, seeds = draw_round(seed, FIELD)
    results = {}
    for live in (False, True) if seed % 2 else (True, False):
        test = TestSeat(seeds[seat], live, bid_ms, rollout_budget, card_budget)
        bots = [test if s == seat else make_bot(opponents[s], seeds[s]) for s in range(n)]
        points, _, _, actions = play(rules, label, hands, trump, dealer, bots, seat)
        results[live] = dict(points=points[seat], bid=test.bid, bid_ms=test.elapsed,
                             worlds=test.worlds, network_fallback=test.network_fallback,
                             actions=actions)
    base, live = results[False], results[True]
    if base['bid'] == live['bid'] and base['actions'] != live['actions']:
        raise RuntimeError('Identical bids should give identical actual play')
    del base['actions'], live['actions']
    return dict(seed=seed, cards=label, players=n, base=base, live=live,
                edge=(live['points'] - base['points']) * rounds_per_game(n))


def summarize(rows):
    ordinary = [r for r in rows if r['cards']]
    edges = [r['edge'] for r in rows]
    timing = {side: dict(mean_ms=statistics.mean(r[side]['bid_ms'] for r in ordinary),
                         median_worlds=statistics.median(r[side]['worlds'] for r in ordinary))
              for side in ('base', 'live')} if ordinary else {}
    return dict(rounds=len(rows), ordinary_bids=len(ordinary),
                changed_bids=sum(r['live']['bid'] != r['base']['bid'] for r in rows),
                edge=statistics.mean(edges),
                margin_95=1.96 * statistics.stdev(edges) / math.sqrt(len(edges)) if len(edges) > 1 else None,
                network_fallbacks=sum(r['live']['network_fallback'] for r in ordinary),
                **timing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--rounds', type=int, required=True)
    parser.add_argument('--bid-ms', type=float, default=250)
    parser.add_argument('--rollout-budget', type=int, default=128)
    parser.add_argument('--card-budget', type=int, default=1024)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if min(args.rounds, args.bid_ms, args.rollout_budget, args.card_budget, args.workers) <= 0:
        parser.error('counts and budgets must be positive')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest, summary = (args.out.with_suffix(ext) for ext in ('.meta', '.json'))
    if any(p.exists() for p in (args.out, manifest, summary)):
        parser.error('output already exists')
    config = SimpleNamespace(seed=args.seed, max_sets=args.rounds, smallest_gain=2,
                             time_ms=args.bid_ms, resume=False)
    check_run_config(manifest, config, [REFERENCE_SPEC, *(spec for spec, _ in FIELD)])
    meta = dict(bid_ms=args.bid_ms, rollout_budget=args.rollout_budget,
                card_budget=args.card_budget, workers=args.workers, field=FIELD)
    manifest.write_text(json.dumps(dict(json.loads(manifest.read_text()), **meta), indent=2) + '\n')
    jobs = ((args.seed + i, args.bid_ms, args.rollout_budget, args.card_budget) for i in range(args.rounds))
    rows = []
    with multiprocessing.Pool(args.workers) as pool, args.out.open('x') as output:
        for row in pool.imap(duel_round, jobs):
            rows.append(row)
            output.write(json.dumps(row) + '\n')
            if len(rows) % 100 == 0:
                output.flush()
                print(f'{len(rows)}/{args.rounds} paired rounds', flush=True)
    result = dict(**meta, **summarize(rows))
    summary.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
