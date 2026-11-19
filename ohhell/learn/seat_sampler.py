"""Offline diagnostic: deal hidden hands seat by seat instead of whole-deal resampling.

The bot reads opponents' bids in ``Belief.worlds``: it deals ``count *
oversample`` deals at random (respecting known cards and shown voids),
weights each by ``exp(-(bid - E[tricks | opening hand])^2 / 2 sigma^2)`` for
every opponent that has bid, and redraws ``count`` deals in proportion. With
several bidders few random deals fit every bid, so the redraw may collapse.

``seat_worlds`` targets the same distribution another way:

1. Opponents that have bid are dealt first, most extreme bid first
   (|bid - hand_size / players| largest). Each seat's hidden hand is drawn
   uniformly from the cards left that its shown voids allow, and accepted with
   probability a(h) = L(bid | h) / max L = exp(-(bid - E)^2 / 2 sigma^2) (the
   same Gaussian, sigma and estimate as ``log_weight``; max L is 1, at E = bid).
   No hard cut-off.
2. The other opponents and the undealt stock are filled at random last.
3. Each finished deal carries an importance weight
       w = prod_s C(A_s, k_s)  *  prod_{bidders s} Z_s(P_s)
   where A_s is the number of cards seat s could take from the pool P_s left
   when it was dealt, k_s its hidden hand size, and Z_s(P_s) = mean of a(h)
   over hands h drawn uniformly from P_s: the seat's acceptance rate given
   the cards left. The first factor makes the sequential void-respecting deal
   uniform over consistent deals; the second removes the bias of accepting
   each seat against its own bid only (a later seat that struggled to fit
   shows a low Z and the deal is down-weighted). Derivation: the proposal is
   prod_s U_s(h_s) a(h_s) / Z_s, the target prior x prod_s a(h_s).

   Z_s is estimated (``estimator``):
   - ``draws`` (default): the mean of a(h) over ``extra`` further uniform
     draws from P_s, independent of the accepted hand. Unbiased for Z_s given
     P_s; the product over seats is unbiased for the weight, so this is a
     valid random-weight (pseudo-marginal) importance sampler.
   - ``tries``: 1 / (number of tries the seat needed). Biased: for a
     geometric count T, E[1/T] = -p ln p / (1 - p), not p.
   - ``none``: no correction (shows what the correction is worth).
4. Tries per seat are capped (``cap``); hitting it restarts the deal. Restarts
   slightly under-represent pools that are hard to fit, so they are counted.

Budget: both samplers are charged one unit per evaluation of the trick
estimate for one seat's hand. ``Belief.worlds`` with 64 x 4 candidates costs
256 x (number of bidders) units; ``seat_worlds`` gets the same and keeps
dealing until it is spent.

Diagnostic (``python -m ohhell.learn.seat_sampler run``): positions are real
bid and card decisions from v2 games (bowers, cumulative down / blind / up,
tournament table sizes, fixed seed); blind rounds and positions with no
opponent bid are skipped. Both samplers use bids only (``play_reading``
off), so they target the same distribution. ``exact`` checks small rounds
against whole-deal rejection sampling.
"""

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..arena import deal_round
from ..belief import Belief, sample_world
from ..bots import make_bot
from ..bots.heuristic import expected_tricks
from ..cards import EFFECTIVE_SUIT, FULL_DECK, cards_in
from ..engine import BIDDING, DONE, NO_BID, PLAYING, RoundState
from ..opponents import PREVIOUS
from ..tournament import PLAYER_SPLIT, rules_for

SIGMA = PREVIOUS.bid_sigma  # what v2 (field=previous) assumes for every opponent


class Counter:
    """Counts trick-estimate evaluations (the shared budget unit)."""

    def __init__(self):
        self.calls = 0

    def accept(self, obs, seat, opening):
        self.calls += 1
        first = (obs.dealer + 1) % obs.n
        estimate = expected_tricks(opening, obs.trump_card, obs.n, obs.hand_size, seat == first)
        return math.exp(-(obs.bids[seat] - estimate) ** 2 / (2 * SIGMA * SIGMA))


def _played_by(obs):
    played = [0] * obs.n
    for seat, card in obs.history:
        played[seat] |= 1 << card
    return played


def _hidden_sizes(obs):
    sizes = [obs.hand_size] * obs.n
    for seat, _ in obs.history:
        sizes[seat] -= 1
    return sizes


def _allowed(obs, seat, pool):
    suit_of = EFFECTIVE_SUIT[obs.trump_suit]
    voids = obs.voids[seat]
    return [c for c in pool if not voids >> suit_of[c] & 1]


def _order(obs):
    """Opponents that bid, most extreme first; then the rest."""
    others = [s for s in range(obs.n) if s != obs.seat]
    mean = obs.hand_size / obs.n
    bidders = sorted((s for s in others if obs.bids[s] != NO_BID),
                     key=lambda s: -abs(obs.bids[s] - mean))
    return bidders, [s for s in others if obs.bids[s] == NO_BID]


def seat_worlds(obs, budget, rng, estimator='draws', extra=8, cap=200, counter=None):
    """Weighted deals ([(hands, weight)], stats) using about ``budget`` evaluations."""
    counter = counter or Counter()
    start = counter.calls
    bidders, rest = _order(obs)
    sizes = _hidden_sizes(obs)
    played = _played_by(obs)
    known = obs.hand | obs.played | (1 << obs.trump_card)
    unseen = cards_in(FULL_DECK & ~known)
    deals, restarts, dead_ends, tries_all = [], 0, 0, []
    while counter.calls - start < budget:
        pool = list(unseen)
        hands = [0] * obs.n
        hands[obs.seat] = obs.hand
        log_w = 0.0
        ok = True
        for seat in bidders + rest:
            k = sizes[seat]
            allowed = _allowed(obs, seat, pool)
            if len(allowed) < k:
                ok, dead_ends = False, dead_ends + 1
                break
            log_w += math.lgamma(len(allowed) + 1) - math.lgamma(k + 1) - math.lgamma(len(allowed) - k + 1)
            if seat in bidders:
                tries = 0
                while True:
                    tries += 1
                    hand = sum(1 << c for c in rng.sample(allowed, k))
                    if rng.random() < counter.accept(obs, seat, hand | played[seat]):
                        break
                    if tries >= cap:
                        ok = False
                        break
                if not ok:
                    restarts += 1
                    break
                tries_all.append(tries)
                if estimator == 'draws':
                    z = sum(counter.accept(obs, seat, sum(1 << c for c in rng.sample(allowed, k))
                                           | played[seat]) for _ in range(extra)) / extra
                elif estimator == 'tries':
                    z = 1.0 / tries
                else:
                    z = 1.0
                if z <= 0:
                    ok = False
                    break
                log_w += math.log(z)
            else:
                hand = sum(1 << c for c in rng.sample(allowed, k))
            hands[seat] = hand
            pool = [c for c in pool if not hand >> c & 1]
        if ok:
            deals.append((hands, log_w))
    if not deals:
        return [], {'restarts': restarts, 'dead_ends': dead_ends, 'tries': tries_all,
                    'calls': counter.calls - start}
    top = max(w for _, w in deals)
    weighted = [(h, math.exp(w - top)) for h, w in deals]
    return weighted, {'restarts': restarts, 'dead_ends': dead_ends, 'tries': tries_all,
                      'calls': counter.calls - start}


def current_worlds(obs, rng, count=64, oversample=4, counter=None):
    """v2's whole-deal resampling, bids only. Returns (resampled deals with weight 1,
    the candidates with their importance weights, evaluations)."""
    counter = counter or Counter()
    start = counter.calls
    bidders, _ = _order(obs)
    played = _played_by(obs)
    candidates = []
    for _ in range(count * oversample):
        hands = sample_world(obs, rng)
        w = 1.0
        for seat in bidders:
            w *= counter.accept(obs, seat, hands[seat] | played[seat])
        candidates.append((hands, w))
    total = sum(w for _, w in candidates)
    if total <= 0:
        picks = [rng.choice(candidates)[0] for _ in range(count)]
    else:
        picks = rng.choices([h for h, _ in candidates], weights=[w for _, w in candidates], k=count)
    return [(h, 1.0) for h in picks], candidates, counter.calls - start


def ess(weighted):
    s = sum(w for _, w in weighted)
    s2 = sum(w * w for _, w in weighted)
    return s * s / s2 if s2 else 0.0


def distinct(weighted):
    return len({tuple(h) for h, w in weighted if w > 0})


def calibration(obs, weighted, truth):
    """(mean log P(true owner) over hidden cards, mean log P(true trump count) over
    opponents), each predicted from the weighted deals with one pseudo-deal of
    uniform smoothing."""
    others = [s for s in range(obs.n) if s != obs.seat]
    known = obs.hand | obs.played | (1 << obs.trump_card)
    hidden = cards_in(FULL_DECK & ~known)
    total = sum(w for _, w in weighted)
    bins = len(others) + 1  # opponents and the undealt stock
    card_log = 0.0
    for c in hidden:
        owner = next((s for s in others if truth[s] >> c & 1), None)
        mass = sum(w for h, w in weighted
                   if (owner is not None and h[owner] >> c & 1)
                   or (owner is None and not any(h[s] >> c & 1 for s in others)))
        card_log += math.log((mass + 1.0 / bins) / (total + 1.0))
    suit_of = EFFECTIVE_SUIT[obs.trump_suit]
    trumps = lambda hand: sum(1 for c in cards_in(hand) if suit_of[c] == obs.trump_suit)
    sizes = _hidden_sizes(obs)
    trump_log = 0.0
    for s in others:
        true_count = trumps(truth[s])
        mass = sum(w for h, w in weighted if trumps(h[s]) == true_count)
        trump_log += math.log((mass + 1.0 / (sizes[s] + 1)) / (total + 1.0))
    return card_log / max(1, len(hidden)), trump_log / max(1, len(others))


# --- positions from v2 games ------------------------------------------------------

PLAY_SPEC = 'hybrid:field=previous,inference=1,worlds=16,bid_worlds=32'  # v2, fixed budget


def _positions(seed, games, rate):
    """(obs, true hidden hands) at sampled bid and card decisions of v2 games."""
    rng = random.Random(seed)
    out = []
    for g in range(games):
        n = rng.choices(list(PLAYER_SPLIT), weights=list(PLAYER_SPLIT.values()))[0]
        rules = rules_for(n)
        bots = [make_bot(PLAY_SPEC, seed * 101 + g * 11 + s) for s in range(n)]
        dealer = rng.randrange(n)
        for label in rules.hand_sizes:
            hands, trump = deal_round(rng, n, max(1, label))
            state = RoundState(rules, hands, trump, dealer, blind=label == 0)
            while state.phase != DONE:
                seat = state.to_act
                obs = state.observation(seat)
                if (label and len(obs.legal_actions()) > 1 and rng.random() < rate
                        and any(b != NO_BID for s, b in enumerate(obs.bids) if s != seat)):
                    out.append((obs, list(state.hands)))
                state.apply(bots[seat].act(obs))
            dealer = (dealer + 1) % n
    return out


def _evaluate(job):
    seed, games, rate, extra, sigma = job
    global SIGMA
    SIGMA = sigma  # this worker's bid-likelihood spread, for every sampler
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    rows = []
    for index, (obs, truth) in enumerate(_positions(seed, games, rate)):
        rng = random.Random(seed * 7919 + index)
        bidders, _ = _order(obs)
        row = {'phase': 'bid' if obs.phase == BIDDING else 'card', 'n': obs.n,
               'hand_size': obs.hand_size, 'bidders': len(bidders),
               'extreme': any(obs.bids[s] == 0 and obs.hand_size >= 5
                              or obs.bids[s] >= max(2, obs.hand_size - 1) for s in bidders)}
        started = time.perf_counter()
        resampled, candidates, budget = current_worlds(obs, rng)
        row['cur_s'] = time.perf_counter() - started
        row['cur_ess'] = ess(candidates)
        row['cur_distinct'] = distinct(resampled)
        row['cur_card'], row['cur_trump'] = calibration(obs, resampled, truth)
        row['budget'] = budget
        uniform = [(sample_world(obs, rng), 1.0) for _ in range(64)]
        row['uni_card'], row['uni_trump'] = calibration(obs, uniform, truth)
        started = time.perf_counter()
        weighted, stats = seat_worlds(obs, budget, rng, extra=extra)
        row['seat_s'] = time.perf_counter() - started
        row['seat_deals'] = len(weighted)
        row['seat_ess'] = ess(weighted) if weighted else 0.0
        row['seat_distinct'] = distinct(weighted) if weighted else 0
        row['seat_restarts'] = stats['restarts']
        row['seat_dead'] = stats['dead_ends']
        row['seat_tries'] = sum(stats['tries']) / len(stats['tries']) if stats['tries'] else 0.0
        if weighted:
            row['seat_card'], row['seat_trump'] = calibration(obs, weighted, truth)
        else:
            row['seat_card'] = row['seat_trump'] = None
        # The cheap variant: weights from 1/tries, no extra draws (biased).
        cheap, cheap_stats = seat_worlds(obs, budget, rng, estimator='tries')
        row['cheap_deals'] = len(cheap)
        row['cheap_ess'] = ess(cheap) if cheap else 0.0
        row['cheap_card'], row['cheap_trump'] = (calibration(obs, cheap, truth) if cheap
                                                 else (None, None))
        rows.append(row)
    return rows


def _table(rows, key, label):
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    lines = [f'\n{label}', '| group | positions | ESS cur / seat | distinct cur (of 64) / seat (deals) '
             '| card log-lik uniform / cur / seat | trump log-lik uniform / cur / seat '
             '| restarts per position | ms cur / seat |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for k in sorted(groups):
        g = groups[k]
        ok = [r for r in g if r['seat_card'] is not None]
        m = lambda name, rs=g: sum(r[name] for r in rs) / len(rs)
        lines.append(
            f'| {k} | {len(g)} | {m("cur_ess"):.1f} / {m("seat_ess"):.1f} '
            f'| {m("cur_distinct"):.1f} / {m("seat_distinct"):.1f} ({m("seat_deals"):.0f}) '
            f'| {m("uni_card", ok):.4f} / {m("cur_card", ok):.4f} / {m("seat_card", ok):.4f} '
            f'| {m("uni_trump", ok):.4f} / {m("cur_trump", ok):.4f} / {m("seat_trump", ok):.4f} '
            f'| {m("seat_restarts"):.2f} | {1000 * m("cur_s"):.0f} / {1000 * m("seat_s"):.0f} |')
    return '\n'.join(lines)


def run(args):
    jobs = [(args.seed + i, args.games, args.rate, args.extra, args.sigma) for i in range(args.jobs)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for part in pool.map(_evaluate, jobs):
            rows.extend(part)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'args': {k: str(v) for k, v in vars(args).items()},
                                    'rows': rows}))
    text = [f'{len(rows)} positions (seed {args.seed}, {args.jobs} x {args.games} games, rate {args.rate}, '
            f'sigma {args.sigma})']
    ok = [r for r in rows if r['seat_card'] is not None and r['cheap_card'] is not None]
    m = lambda k: sum(r[k] for r in ok) / len(ok)
    text.append('')
    text.append(f'Summary over {len(ok)} positions | ESS: current {m("cur_ess"):.1f} of 256, '
                f'seat {m("seat_ess"):.1f} ({m("seat_deals"):.0f} deals), cheap seat {m("cheap_ess"):.1f} '
                f'({m("cheap_deals"):.0f} deals) | card log-lik: uniform {m("uni_card"):.4f}, current '
                f'{m("cur_card"):.4f}, seat {m("seat_card"):.4f}, cheap seat {m("cheap_card"):.4f} | '
                f'trump log-lik: uniform {m("uni_trump"):.4f}, current {m("cur_trump"):.4f}, seat '
                f'{m("seat_trump"):.4f}, cheap seat {m("cheap_trump"):.4f}')
    text.append(_table(rows, lambda r: 'all', 'All positions'))
    text.append(_table(rows, lambda r: r['phase'], 'By decision'))
    text.append(_table(rows, lambda r: min(r['bidders'], 4), 'By opponents that have bid (4 = 4+)'))
    text.append(_table(rows, lambda r: ('1-3' if r['hand_size'] <= 3 else '4-6' if r['hand_size'] <= 6
                                         else '7-10'), 'By hand size'))
    text.append(_table(rows, lambda r: 'extreme bid' if r['extreme'] else 'no extreme bid',
                       'Extreme bids (0 in 5+ cards, or hand size - 1 or more)'))
    report = '\n'.join(text)
    print(report)
    args.out.with_suffix('.md').write_text(report)


# --- exactness check on small rounds ---------------------------------------------

def _marginals(obs, weighted):
    """P(card c with seat s) for every hidden card and opponent."""
    others = [s for s in range(obs.n) if s != obs.seat]
    total = sum(w for _, w in weighted)
    known = obs.hand | obs.played | (1 << obs.trump_card)
    out = {}
    for c in cards_in(FULL_DECK & ~known):
        for s in others:
            out[c, s] = sum(w for h, w in weighted if h[s] >> c & 1) / total
    return out


def rejection_reference(obs, rng, accepted):
    """Exact sampler for the same target: whole deals accepted with prob prod a."""
    bidders, _ = _order(obs)
    played = _played_by(obs)
    counter = Counter()
    out = []
    while len(out) < accepted:
        hands = sample_world(obs, rng)
        p = 1.0
        for s in bidders:
            p *= counter.accept(obs, s, hands[s] | played[s])
        if rng.random() < p:
            out.append((hands, 1.0))
    return out


def exact(args):
    rng = random.Random(args.seed)
    results = []
    cases = 0
    while cases < args.cases:
        n = rng.choice([3, 4])
        hand_size = rng.choice([2, 3])
        hands, trump = deal_round(rng, n, hand_size)
        state = RoundState(rules_for(n), hands, trump, rng.randrange(n))
        bots = [make_bot('skilled', s) for s in range(n)]
        while state.to_act != (state.dealer + 0) % n and state.phase == BIDDING:
            state.apply(bots[state.to_act].act(state.observation(state.to_act)))
        obs = state.observation(state.to_act)
        if not any(b != NO_BID for s, b in enumerate(obs.bids) if s != obs.seat):
            continue
        cases += 1
        reference = _marginals(obs, rejection_reference(obs, rng, args.reference))
        row = {'n': n, 'hand_size': hand_size, 'bids': [b for b in obs.bids]}
        # Noise floor: a second, independent exact reference of the same size.
        second = _marginals(obs, rejection_reference(obs, rng, args.reference))
        floor = [abs(second[k] - reference[k]) for k in reference]
        row['reference noise'] = {'mean_abs': sum(floor) / len(floor), 'max_abs': max(floor)}
        for name, make in (
                ('current (64x4)', lambda: current_worlds(obs, rng, count=args.deals // 4,
                                                          oversample=4)[1]),
                ('seat, draws', lambda: seat_worlds(obs, args.budget, rng, 'draws')[0]),
                ('seat, tries', lambda: seat_worlds(obs, args.budget, rng, 'tries')[0]),
                ('seat, no correction', lambda: seat_worlds(obs, args.budget, rng, 'none')[0])):
            # Average each run's self-normalised marginals (runs' weights have
            # different scales, so they must not be pooled raw).
            runs = [_marginals(obs, make()) for _ in range(args.repeats)]
            marg = {k: sum(m[k] for m in runs) / len(runs) for k in reference}
            errors = [abs(marg[k] - reference[k]) for k in reference]
            row[name] = {'mean_abs': sum(errors) / len(errors), 'max_abs': max(errors)}
        results.append(row)
        print(json.dumps(row), flush=True)
    summary = {}
    for name in ('reference noise', 'current (64x4)', 'seat, draws', 'seat, tries',
                 'seat, no correction'):
        summary[name] = {
            'mean_abs': sum(r[name]['mean_abs'] for r in results) / len(results),
            'max_abs': max(r[name]['max_abs'] for r in results)}
    print(json.dumps(summary, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'cases': results, 'summary': summary,
                                    'args': {k: str(v) for k, v in vars(args).items()}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--seed', type=int, default=20261117)
    r.add_argument('--jobs', type=int, default=16, help='independent game batches')
    r.add_argument('--games', type=int, default=3, help='games per batch')
    r.add_argument('--rate', type=float, default=0.12, help='share of decisions sampled')
    r.add_argument('--extra', type=int, default=8, help='extra draws per seat for Z')
    r.add_argument('--sigma', type=float, default=SIGMA,
                   help='bid-likelihood spread for every sampler (v2 uses 1.4; 0.76 is fitted)')
    r.add_argument('--workers', type=int, default=4)
    r.add_argument('--out', type=Path, default=Path('runs/seat_sampler/diagnostic.json'))
    e = sub.add_parser('exact')
    e.add_argument('--seed', type=int, default=11)
    e.add_argument('--cases', type=int, default=12)
    e.add_argument('--reference', type=int, default=20000, help='accepted rejection deals')
    e.add_argument('--budget', type=int, default=2000, help='evaluations per seat-sampler run')
    e.add_argument('--deals', type=int, default=256, help='current sampler candidates per run')
    e.add_argument('--repeats', type=int, default=20, help='independent runs pooled per sampler')
    e.add_argument('--out', type=Path, default=Path('runs/seat_sampler/exact.json'))
    args = parser.parse_args()
    {'run': run, 'exact': exact}[args.command](args)


if __name__ == '__main__':
    main()
