"""Which simulation/network blend predicts outcomes best, at each card count?

Rounds are dealt at the tournament's table sizes with its lobby (v2, its
leaning versions and the human-like players). The candidate and baseline
seats are both played by v2 with a recorder that keeps, at every decision,
what the simulation alone and the network alone predicted. Play then
continues as v2 would, and each prediction is scored against what actually
happened to that seat:

- Bids, by hand size: the trick distribution for the bid v2 made, scored by
  Brier, log loss (probabilities floored at 1e-3) and squared error of the
  expected points.
- Cards, by cards in hand: the predicted round points of the card v2 played,
  scored by squared error. Exact endgames are skipped (they ignore the network).

Each blend ``w`` (the network's share) is scored as ``(1 - w) sim + w net``.
v2 uses 0.5 for bids and 16 / (32 + 16) = 1/3 for cards. Differences against
v2's blend come with standard errors clustered by round.

Only the chosen action's outcome is observed, and play after a bid is v2's,
so this measures calibration on v2's own decisions, not game strength.

    python -m ohhell.learn.blend_check --rounds 30000 --out runs/blend_check/check.json
"""

import argparse
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..arena import deal_round
from ..bots import DEFAULT_SPEC, _parse_value, make_bot
from ..bots.learned import HybridBot
from ..bots.simulation import BidAnalysis, SimulationBot
from ..engine import DONE, RoundState
from ..tournament import BASELINE, CANDIDATE, build_lobby, rules_for, table_size

GRID = [i / 20 for i in range(21)]
V2_BID, V2_PLAY = 0.5, 16 / 48
FLOOR = 1e-3


class Recorder(HybridBot):
    """v2, noting the simulation's and the network's predictions as it goes."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bids, self.cards = [], []

    def analyse_bids(self, obs):
        simulated = SimulationBot.analyse_bids(self, obs)
        predicted = self.models.bid_distributions(
            obs.hand, obs.trump_card, obs.n, obs.hand_size, obs.seat, obs.dealer,
            obs.bids, list(simulated.distributions))
        self.bids.append({'sim': simulated.distributions, 'samples': simulated.samples,
                          'net': {b: list(p) for b, p in predicted.items()}})
        blend = self.bid_blend  # as HybridBot.analyse_bids, without simulating twice
        distributions = {
            bid: [(1.0 - blend) * s + blend * p for s, p in zip(sim, predicted[bid])]
            for bid, sim in simulated.distributions.items()
        }
        score = obs.rules.scoring
        return BidAnalysis(distributions, {
            bid: sum(p * score.points(bid, k) for k, p in enumerate(dist))
            for bid, dist in distributions.items()
        })

    def play(self, obs):
        cards = obs.legal_actions()
        remaining = obs.hand.bit_count() * obs.n - obs.trick_size
        if len(cards) == 1 or remaining <= self.endgame_cards:
            return super().play(obs)
        predicted = dict(zip(cards, self.models.card_values(obs, obs.seat, obs.hand, cards)))
        kept = sorted(cards, key=predicted.get, reverse=True)[:self.shortlist]
        simulated = self.evaluate_cards(obs, kept)
        weight = self.prior_worlds
        card = max(kept, key=lambda c: (self.worlds * simulated[c] + weight * predicted[c])
                   / (self.worlds + weight))
        self.cards.append({'cards': obs.hand.bit_count(), 'sim': simulated[card],
                           'net': predicted[card]})
        return card


def _rounds(job):
    first, count, time_ms = job
    lobby = build_lobby(DEFAULT_SPEC, DEFAULT_SPEC, time_ms, 'stand-in')
    specs = [spec for _, _, spec in lobby]
    name, _, args = specs[BASELINE].partition(':')
    assert name == 'hybrid', specs[BASELINE]
    recorder_kwargs = {}
    for pair in args.split(','):
        key, _, value = pair.partition('=')
        recorder_kwargs[key] = _parse_value(value)
    out = []
    for number in range(first, first + count):
        rng = random.Random(number)
        n = table_size(number)
        rules = rules_for(n)
        members = [CANDIDATE, BASELINE] + rng.sample(range(BASELINE + 1, len(specs)), n - 2)
        rng.shuffle(members)
        bots = []
        for seat, m in enumerate(members):
            seed = number * 31 + seat
            if m in (CANDIDATE, BASELINE):
                bots.append(Recorder(seed=seed, **recorder_kwargs))
            else:
                bots.append(make_bot(specs[m], seed))
        hand_size = rng.randint(1, rules.max_hand)
        hands, trump = deal_round(rng, n, hand_size)
        state = RoundState(rules, hands, trump, rng.randrange(n))
        while state.phase != DONE:
            seat = state.to_act
            state.apply(bots[seat].act(state.observation(seat)))
        points = state.points()
        for seat, bot in enumerate(bots):
            if not isinstance(bot, Recorder):
                continue
            bid, tricks = state.bids[seat], state.tricks_won[seat]
            for b in bot.bids:
                if bid not in b['sim']:
                    continue  # not reached when v2 picks from its own shortlist
                out.append({'kind': 'bid', 'round': number, 'n': n, 'cards': hand_size,
                            'bid': bid, 'tricks': tricks, 'points': points[seat],
                            'sim': b['sim'][bid], 'net': b['net'][bid],
                            'samples': b['samples']})
            for c in bot.cards:
                out.append({'kind': 'card', 'round': number, 'n': n, 'cards': c['cards'],
                            'points': points[seat], 'sim': c['sim'], 'net': c['net']})
    return out


def collect(rounds, time_ms, workers, first):
    chunk = max(1, rounds // (workers * 8))
    jobs = [(first + start, min(chunk, rounds - start), time_ms)
            for start in range(0, rounds, chunk)]
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [r for part in pool.map(_rounds, jobs) for r in part]


def bid_losses(r, w, scoring):
    dist = [(1 - w) * s + w * p for s, p in zip(r['sim'], r['net'])]
    k = r['tricks']
    brier = sum((p - (i == k)) ** 2 for i, p in enumerate(dist))
    nll = -math.log(max(dist[k], FLOOR))
    expected = sum(p * scoring.points(r['bid'], i) for i, p in enumerate(dist))
    return {'brier': brier, 'nll': nll, 'points_se': (expected - r['points']) ** 2}


def card_losses(r, w, scoring=None):
    return {'points_se': ((1 - w) * r['sim'] + w * r['net'] - r['points']) ** 2}


def compare(records, losses, metrics, base_w, scoring):
    """Mean loss per blend, and each blend's difference from base_w with a
    round-clustered standard error."""
    by_round = {}
    for r in records:
        by_round.setdefault(r['round'], []).append(r)
    count = len(records)
    result = {}
    for metric in metrics:
        sums = {w: [] for w in GRID}
        for rows in by_round.values():
            base = sum(losses(r, base_w, scoring)[metric] for r in rows)
            for w in GRID:
                total = sum(losses(r, w, scoring)[metric] for r in rows)
                sums[w].append((total, total - base))
        rows_out = []
        clusters = len(by_round)
        for w in GRID:
            totals = [t for t, _ in sums[w]]
            diffs = [d for _, d in sums[w]]
            mean_diff = sum(diffs) / count
            # Ratio estimator: per-cluster residuals around mean_diff * size.
            sizes = [len(rows) for rows in by_round.values()]
            resid = [d - mean_diff * s for d, s in zip(diffs, sizes)]
            se = (math.sqrt(sum(e * e for e in resid) * clusters / max(1, clusters - 1))
                  / count)
            rows_out.append({'w': w, 'mean': sum(totals) / count,
                             'diff_vs_v2': mean_diff, 'se': se})
        best = min(rows_out, key=lambda row: row['mean'])
        result[metric] = {'by_w': rows_out, 'best_w': best['w'],
                          'best_diff': best['diff_vs_v2'], 'best_se': best['se']}
    return result


def report(records, scoring):
    out = {'bids': {}, 'cards': {}}
    for kind, key, losses, metrics, base in (
            ('bid', 'bids', bid_losses, ('brier', 'nll', 'points_se'), V2_BID),
            ('card', 'cards', card_losses, ('points_se',), V2_PLAY)):
        rows = [r for r in records if r['kind'] == kind]
        for cards in sorted({r['cards'] for r in rows}):
            group = [r for r in rows if r['cards'] == cards]
            out[key][cards] = {'samples': len(group),
                               'rounds': len({r['round'] for r in group}),
                               **compare(group, losses, metrics, base, scoring)}
    return out


def _sim_only(v):
    return v['by_w'][0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--rounds', type=int, default=30000)
    parser.add_argument('--time-ms', type=float, default=2)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--first', type=int, default=6_000_000, help='first round seed')
    parser.add_argument('--out', type=Path, required=True,
                        help='JSON file for records and report (records reused if it exists)')
    args = parser.parse_args(argv)

    from ..rules import Scoring
    scoring = Scoring()
    if args.out.exists():
        data = json.loads(args.out.read_text())
        records = data['records']
    else:
        records = collect(args.rounds, args.time_ms, args.workers, args.first)
        data = {'rounds': args.rounds, 'time_ms': args.time_ms, 'first': args.first,
                'spec': DEFAULT_SPEC, 'records': records}
    data['report'] = report(records, scoring)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data))

    print('Bids by hand size: best network share per metric '
          '(difference from v2 0.5, +/- 2 se; negative is better)')
    for cards, row in data['report']['bids'].items():
        parts = []
        for metric in ('brier', 'nll', 'points_se'):
            m = row[metric]
            sim = _sim_only(m)
            parts.append(f"{metric} w={m['best_w']:.2f} {m['best_diff']:+.4f}+/-{2*m['best_se']:.4f}"
                         f" (sim-only {sim['diff_vs_v2']:+.4f})")
        print(f"  {cards:>2} cards, {row['samples']:>6} bids: " + '; '.join(parts))
    print('Cards by cards in hand: best network share for squared points error '
          '(difference from v2 1/3)')
    for cards, row in data['report']['cards'].items():
        m = row['points_se']
        sim, net = _sim_only(m), m['by_w'][-1]
        print(f"  {cards:>2} cards, {row['samples']:>6} plays: w={m['best_w']:.2f} "
              f"{m['best_diff']:+.3f}+/-{2*m['best_se']:.3f}  sim-only {sim['diff_vs_v2']:+.3f}  "
              f"net-only {net['diff_vs_v2']:+.3f}  (v2 mse {m['by_w'][0]['mean'] - sim['diff_vs_v2']:.2f})")


if __name__ == '__main__':
    main()
