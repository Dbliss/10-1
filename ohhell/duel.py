"""Paired round duels: is a candidate bot better than a baseline, fast.

Built for long thinking times (250 ms to 1 s a turn), where a tournament
that times every seat through every rotation costs hours. Each sample is one
round, dealt at random from what a real game contains:

- table size drawn with the tournament's weights (PLAYER_SPLIT), the round
  label uniform over that table's 21 (or fewer) rounds, including the blind
  round, a random dealer and a random seat under test;
- the round is played twice on the same cards: once with the baseline in the
  test seat, once with the candidate. The other seats are the same opponents
  with the same seeds both times, on a fixed search budget (no clock), so
  they repeat their decisions exactly while the table looks the same. The
  only difference between the two plays is the test seat's policy. Where both
  versions choose the same actions the difference is exactly zero.
- only the test seat thinks for ``--time-ms``, as fixed deal budgets: the
  game bot gets its idle-machine deals per ms (REFERENCE_RATES) x time; any
  other version gets that scaled by its deals per turn relative to the game
  bot, both timed together on ``--calibrate-rounds`` interleaved rounds (so
  machine load cancels). Same compute scale in every duel, exactly
  repeatable: two equal bots never disagree and every nonzero difference
  comes from a real change of decision. ``--timed`` keeps the clock instead.

Bots in this project never see the running game score, and every round is
dealt afresh, so a game's score is the sum of independent rounds. Each
sample's difference is multiplied by its table's rounds per game, so the
mean of the samples estimates the points per game gained, matching the
tournament's unit. Bots that remember earlier rounds (``typing=1``) are not
measured fairly here: each round starts with fresh bots.

The run is checked at ten planned looks (every ``--max-rounds`` / 10), with
the tournament's O'Brien-Fleming margins (5% error over all looks), and
stops at the first look where the edge clears zero (better / worse) or its
upper end falls below ``--smallest-gain`` (no gain); ``--stopping cs`` uses
a confidence sequence checked every ``--check-every`` rounds instead, valid
under any monitoring but wider. A "better" is then repeated on unseen deals with as many
rounds and must clear zero one-sided at 95%, as the tournament's
confirmation does.

    python -m ohhell.duel CANDIDATE --baseline BASELINE --time-ms 250 --label NAME

Results: ``runs/duels/LABEL.{meta,jsonl,json}``; ``--resume`` continues.
"""

import argparse
import json
import math
import multiprocessing
import os
import random
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from .arena import deal_round
from .bots import make_bot
from .engine import DONE, RoundState
from .tournament import FINAL_Z, LOOKS, PLAYER_SPLIT, check_run_config, rules_for

HERE = Path(__file__).parent
DUELS = HERE.parent / 'runs' / 'duels'
# Stand-ins for the people and bots a game bot plays against. Fixed budgets so
# they repeat themselves exactly; weights are relative.
FIELD = (
    ('hybrid:field=previous,inference=1,worlds=16,bid_worlds=32', 3.0),
    ('skilled', 1.0),
    ('casual', 1.0),
)
CONFIRM_Z = 1.645


def parse_field(text):
    if not text:
        return FIELD
    field = []
    for part in text.split(';'):
        spec, _, weight = part.rpartition('@')
        field.append((spec, float(weight)) if spec else (part, 1.0))
    return tuple(field)


def extend(spec, extra):
    return f'{spec}{"," if ":" in spec else ":"}{extra}'


def rounds_per_game(players):
    return len(rules_for(players).hand_sizes)


def draw_round(seed, field):
    """Everything about one sampled round that does not depend on the bots."""
    rng = random.Random(seed)
    players = rng.choices(list(PLAYER_SPLIT), weights=list(PLAYER_SPLIT.values()))[0]
    rules = rules_for(players)
    label = rng.choice(rules.hand_sizes)
    hands, trump = deal_round(rng, players, max(1, label))
    dealer = rng.randrange(players)
    seat = rng.randrange(players)
    specs, weights = zip(*field)
    opponents = [rng.choices(specs, weights=weights)[0] for _ in range(players)]
    seeds = [rng.randrange(1 << 30) for _ in range(players)]
    return players, rules, label, hands, trump, dealer, seat, opponents, seeds


def play(rules, label, hands, trump, dealer, bots, seat):
    """Play one round; returns (points, test seat seconds, choices, actions)."""
    state = RoundState(rules, list(hands), trump, dealer, blind=label == 0)
    seconds, choices, actions = 0.0, 0, []
    while state.phase != DONE:
        to_act = state.to_act
        obs = state.observation(to_act)
        if to_act == seat and len(obs.legal_actions()) > 1:
            started = time.perf_counter()
            action = bots[to_act].act(obs)
            seconds += time.perf_counter() - started
            choices += 1
        else:
            action = bots[to_act].act(obs)
        actions.append(action)
        state.apply(action)
    return state.points(), seconds, choices, actions


def duel_round(job):
    index, seed, candidate, baseline, field = job
    players, rules, label, hands, trump, dealer, seat, opponents, seeds = draw_round(seed, field)
    results = {}
    order = ('baseline', 'candidate') if index % 2 == 0 else ('candidate', 'baseline')
    for side in order:
        spec = candidate if side == 'candidate' else baseline
        bots = [make_bot(spec if s == seat else opponents[s], seeds[s]) for s in range(players)]
        results[side] = play(rules, label, hands, trump, dealer, bots, seat)
    (cp, cs, cc, ca), (bp, bs, bc, ba) = results['candidate'], results['baseline']
    first = next((i for i, (x, y) in enumerate(zip(ca, ba)) if x != y), None)
    return {'i': index, 'seed': seed, 'n': players, 'label': label, 'seat': seat,
            'cand': cp[seat], 'base': bp[seat], 'diff': cp[seat] - bp[seat],
            'x': (cp[seat] - bp[seat]) * rounds_per_game(players),
            'diverged': first, 'cand_ms': 1000 * cs, 'base_ms': 1000 * bs,
            'cand_choices': cc, 'base_choices': bc}


def calibrate_round(job):
    """Deals a timed test seat completes per bid, card and endgame turn in one round."""
    seed, spec, field = job
    players, rules, label, hands, trump, dealer, seat, opponents, seeds = draw_round(seed, field)
    bots = [make_bot(spec if s == seat else opponents[s], seeds[s]) for s in range(players)]
    state = RoundState(rules, list(hands), trump, dealer, blind=label == 0)
    counts = {'bid': [], 'card': [], 'endgame': []}
    bot = bots[seat]
    while state.phase != DONE:
        to_act = state.to_act
        obs = state.observation(to_act)
        action = bots[to_act].act(obs)
        if to_act == seat and len(obs.legal_actions()) > 1 and not obs.blind:
            if state.phase == 0:
                counts['bid'].append(bot.simulator.last_bid_samples)
            else:
                remaining = obs.hand.bit_count() * obs.n - obs.trick_size
                kind = 'endgame' if remaining <= bot.endgame_limit(obs.n) else 'card'
                counts[kind].append(bot.last_card_worlds)
        state.apply(action)
    return counts


# The game bot's deals per millisecond of thinking on this machine when idle
# (250 ms, 400 rounds, 18 workers, faster core; 17 November 2026). Budgets
# are tied to these, not to whatever load the machine has while a duel
# calibrates. Before 17 November: 'hybrid:field=previous,inference=1' at
# 39.0 card / 71.8 bid / 41.2 endgame on the old core.
# 18 November: the game bot became field=mix (user decision). Its rates are
# the field=previous game bot's (118.8 / 218.3 / 97.5) scaled by the deals
# ratio measured in the '250ms field mix' duel's interleaved calibration
# (25,915 / 44,705 / 33,400 deals per 250 ms turn).
REFERENCE_SPEC = 'hybrid:field=mix,inference=1,shortlist=13,touching=1,bid_sigma=0.76,bid_blend=0.25'
REFERENCE_RATES = {'card': 103.66, 'bid': 178.82, 'endgame': 133.6}


def _measure(specs, args, pool):
    """Mean deals per bid, card and endgame turn of each timed spec, measured
    together: rounds alternate between the specs on the same seeds in one
    pool, so a change of machine load hits all of them alike."""
    jobs = [(args.seed + 20_000_000 + i, spec, args.field_specs)
            for i in range(args.calibrate_rounds) for spec in specs]
    totals = {spec: {'bid': [], 'card': [], 'endgame': []} for spec in specs}
    for (_, spec, _), counts in zip(jobs, pool.imap(calibrate_round, jobs)):
        for key, values in counts.items():
            totals[spec][key].extend(values)
    return {spec: {k: (sum(v) / len(v) if v else 0.0, len(v)) for k, v in kinds.items()}
            for spec, kinds in totals.items()}


def calibrate(raw, args, pool):
    """Each spec in ``raw`` with fixed deal budgets for ``args.time_ms``.

    The reference spec (the game's bot) gets REFERENCE_RATES x time. Any other
    spec gets that times its deals per turn relative to the reference, both
    timed at ``args.time_ms`` in the same interleaved run. So a bot that is
    twice as slow per deal gets half the deals, whatever the load, and every
    duel at a given time uses the same compute scale.
    """
    timed = {spec: extend(spec, f'time_ms={args.time_ms:g}') for spec in set(raw)}
    reference = extend(REFERENCE_SPEC, f'time_ms={args.time_ms:g}')
    others = sorted(t for s, t in timed.items() if s != REFERENCE_SPEC)
    measured = _measure([reference] + others, args, pool) if others else {}
    result, info = [], {}
    load = None
    if measured:
        ref = measured[reference]
        load = {k: ref[k][0] / (REFERENCE_RATES[k] * args.time_ms) for k in REFERENCE_RATES}
    for spec in raw:
        budgets = {}
        for kind, rate in REFERENCE_RATES.items():
            ratio = 1.0
            if spec != REFERENCE_SPEC:
                own, ref = measured[timed[spec]][kind][0], measured[reference][kind][0]
                ratio = own / ref if own and ref else 1.0
            budgets[kind] = max(1, round(rate * args.time_ms * ratio))
        info[spec] = {'budgets': budgets, 'measured': measured.get(timed[spec])}
        result.append(extend(spec, f'card_budget={budgets["card"]},bid_budget={budgets["bid"]},'
                                   f'endgame_budget={budgets["endgame"]}'))
        print(f'budgets for {spec}: {budgets}', flush=True)
    if load:
        print(f'machine load during calibration: reference ran at '
              f'{", ".join(f"{k} {v:.0%}" for k, v in load.items())} of its idle rate', flush=True)
    return result, {'specs': info, 'reference': REFERENCE_SPEC, 'rates': REFERENCE_RATES,
                    'load': load, 'rounds': args.calibrate_rounds, 'time_ms': args.time_ms}


class Running:
    __slots__ = ('n', 'mean', 'm2')

    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def add(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    @property
    def sd(self):
        return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0.0


def cs_radius(sd, n, planned, alpha=0.05):
    """Half-width of a two-sided (1 - alpha) asymptotic confidence sequence.

    Robbins' normal mixture (Waudby-Smith et al. 2021), tuned to be tightest
    near ``planned`` samples. It stays valid however often the run is checked.
    """
    rho2 = (-2 * math.log(alpha) + math.log(-2 * math.log(alpha) + 1)) / planned
    return sd * math.sqrt(2 * (n * rho2 + 1) / (n * n * rho2)
                          * math.log(math.sqrt(n * rho2 + 1) / alpha))


def obf_radius(sd, n, max_rounds):
    """Margin under the tournament's O'Brien-Fleming plan (10 looks, 5% overall).

    Only valid when checked at the planned looks (every max_rounds / LOOKS).
    """
    return FINAL_Z * sd * math.sqrt(max_rounds) / n


def radius_for(stats, args):
    if args.stopping == 'obf':
        return obf_radius(stats.sd, stats.n, args.max_rounds)
    return cs_radius(stats.sd, stats.n, args.plan)


def decide(stats, args):
    if stats.n < args.min_rounds:
        return None, None
    radius = radius_for(stats, args)
    low, high = stats.mean - radius, stats.mean + radius
    if low > 0:
        return 'better', radius
    if high < 0:
        return 'worse', radius
    if high < args.smallest_gain:
        return 'no gain', radius
    # Futility: stopping when even a plain 95% upper bound is below the
    # smallest gain worth having can only prevent a "better", never cause
    # one, so the 5% error for improvements still holds.
    if args.futility and stats.mean + 1.96 * stats.sd / math.sqrt(stats.n) < args.smallest_gain:
        return 'no gain', radius
    if stats.n >= args.max_rounds:
        return 'inconclusive', radius
    return None, radius


def summary(rows):
    def group(key):
        out = {}
        for r in rows:
            out.setdefault(key(r), []).append(r)
        return {str(k): {'rounds': len(v), 'edge_per_round': sum(r['diff'] for r in v) / len(v),
                         'diverged': sum(r['diverged'] is not None for r in v) / len(v)}
                for k, v in sorted(out.items())}
    timed = [r for r in rows if r['cand_choices'] and r['base_choices']]
    return {
        'by_players': group(lambda r: r['n']), 'by_label': group(lambda r: r['label']),
        'diverged': sum(r['diverged'] is not None for r in rows) / max(1, len(rows)),
        'cand_ms_per_choice': sum(r['cand_ms'] for r in timed) / max(1, sum(r['cand_choices'] for r in timed)),
        'base_ms_per_choice': sum(r['base_ms'] for r in timed) / max(1, sum(r['base_choices'] for r in timed)),
    }


def run_stage(name, first, count, saved, output, args, specs, stop_early):
    stats, rows = Running(), []
    started = time.monotonic()
    jobs = [(i, args.seed + first + i, specs[0], specs[1], args.field_specs)
            for i in range(count)]
    todo = [job for job in jobs if (name, job[0]) not in saved]
    verdict = radius = None

    def take(row):
        nonlocal verdict, radius
        rows.append(row)
        stats.add(row['x'])
        every = max(1, args.max_rounds // LOOKS) if args.stopping == 'obf' else args.check_every
        if stop_early and stats.n % every == 0:
            verdict, radius = decide(stats, args)
        return verdict

    for job in jobs:
        if (name, job[0]) in saved and take(saved[name, job[0]]):
            return verdict, radius, stats, rows
        if (name, job[0]) not in saved:
            break
    with multiprocessing.Pool(args.workers) as pool:
        for row in pool.imap(duel_round, [j for j in todo if j[0] >= stats.n]):
            row['stage'] = name
            output.write(json.dumps(row) + '\n')
            output.flush()
            if take(row):
                break
            if stats.n % args.report_every == 0:
                r = radius_for(stats, args) if stats.n > 1 else float('nan')
                rate = stats.n / max(1e-9, time.monotonic() - started)
                print(f'{name} {stats.n:,} rounds: {stats.mean:+.2f} pts/game, gate margin +/- {r:.2f} '
                      f'({rate:.1f} rounds/s)', flush=True)
    if stop_early and verdict is None:
        verdict, radius = decide(stats, args)
        verdict = verdict or 'inconclusive'
    return verdict, radius, stats, rows


def run(args):
    DUELS.mkdir(parents=True, exist_ok=True)
    label = args.label.replace(' ', '_')
    base = DUELS / label
    limit = f'time_ms={args.time_ms:g}' if args.time_ms else None
    raw = [args.candidate, args.baseline]
    timed = [extend(s, limit) if limit else s for s in raw]
    args.field_specs = parse_field(args.field)
    meta_path = base.parent / (base.name + '.meta')
    calibration = None
    if args.resume and meta_path.exists():
        saved_meta = json.loads(meta_path.read_text())
        specs, calibration = [saved_meta['candidate'], saved_meta['baseline']], saved_meta.get('calibration')
    elif limit and not args.timed:
        with multiprocessing.Pool(args.workers) as pool:
            specs, calibration = calibrate(raw, args, pool)
    else:
        specs = timed
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / 'meta'
        check_run_config(path, SimpleNamespace(max_sets=args.max_rounds, smallest_gain=args.smallest_gain,
                                               time_ms=0 if calibration else args.time_ms,
                                               resume=False,
                                               seed=args.seed),
                         specs + [s for s, _ in args.field_specs])
        meta = json.loads(path.read_text())
    meta.update({'kind': 'duel', 'candidate': specs[0], 'baseline': specs[1],
                 'field': args.field_specs, 'seed': args.seed, 'plan': args.plan,
                 'min_rounds': args.min_rounds, 'check_every': args.check_every,
                 'calibration': calibration, 'time_ms': args.time_ms})
    meta = json.loads(json.dumps(meta))
    if args.resume:
        if not meta_path.exists() or json.loads(meta_path.read_text()) != meta:
            raise SystemExit('Saved duel differs in specs, models, source or settings; use a new --label.')
    else:
        meta_path.write_text(json.dumps(meta, indent=2))
    saved = {}
    rows_path = base.parent / (base.name + '.jsonl')
    if args.resume and rows_path.exists():
        for line in rows_path.read_text().splitlines():
            row = json.loads(line)
            saved[row['stage'], row['i']] = row
    print(f'Candidate: {specs[0]}\nBaseline:  {specs[1]}\nField: {args.field_specs}', flush=True)
    started = time.monotonic()
    with rows_path.open('a' if args.resume else 'w') as output:
        verdict, radius, stats, rows = run_stage('main', 0, args.max_rounds, saved, output, args, specs, True)
        print(f'main: {verdict} after {stats.n:,} rounds, {stats.mean:+.2f} +/- {radius or 0:.2f} '
              f'pts/game', flush=True)
        result = {'label': args.label, 'verdict': verdict, 'main': {
            'rounds': stats.n, 'edge': stats.mean, 'cs_radius': radius, 'sd': stats.sd,
            **summary(rows)}}
        if verdict == 'better':
            _, _, rep, rep_rows = run_stage('confirm', 10_000_000, stats.n, saved, output, args,
                                            specs, False)
            margin = CONFIRM_Z * rep.sd / math.sqrt(rep.n) if rep.n > 1 else float('inf')
            ok = rep.mean - margin > 0
            result['confirm'] = {'rounds': rep.n, 'edge': rep.mean, 'one_sided_margin': margin,
                                 'passed': ok, **summary(rep_rows)}
            result['verdict'] = 'better' if ok else 'not confirmed'
            print(f'confirmation: {rep.mean:+.2f} pts/game, one-sided 95% margin {margin:.2f}: '
                  f'{"passed" if ok else "failed"}', flush=True)
    result['seconds'] = time.monotonic() - started
    (base.parent / (base.name + '.json')).write_text(json.dumps(result, indent=2))
    print(f'verdict: {result["verdict"]} ({result["seconds"]:.0f} s)', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('candidate')
    parser.add_argument('--baseline', default=REFERENCE_SPEC,
                        help="default: the game's bot (Game.LONG_THINK settings)")
    parser.add_argument('--time-ms', type=float, default=250.0,
                        help='time per turn for the test seat (0: leave the specs as given)')
    parser.add_argument('--field', default=None,
                        help="opponents as 'spec@weight;spec@weight' (fixed budgets, no clock)")
    parser.add_argument('--label', required=True)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--max-rounds', type=int, default=60000)
    parser.add_argument('--stopping', choices=('obf', 'cs'), default='obf',
                        help="obf: ten planned looks (O'Brien-Fleming, as the tournament); "
                             'cs: confidence sequence, checked every --check-every rounds')
    parser.add_argument('--plan', type=int, default=4000,
                        help='rounds where the confidence sequence is tightest')
    parser.add_argument('--min-rounds', type=int, default=400)
    parser.add_argument('--check-every', type=int, default=20)
    parser.add_argument('--report-every', type=int, default=200)
    parser.add_argument('--no-futility', dest='futility', action='store_false',
                        help='only stop for no gain when the planned margin rules it out')
    parser.add_argument('--smallest-gain', type=float, default=2.0,
                        help='points per game below which a gain is not worth having')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument('--timed', action='store_true',
                        help='keep the clock for the test seat instead of calibrated deal budgets '
                             '(noisier: equal bots then disagree on close calls)')
    parser.add_argument('--calibrate-rounds', type=int, default=300,
                        help='rounds per bot used to measure deals per turn at --time-ms')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.seed is None:
        args.seed = int.from_bytes(args.label.encode(), 'little') % 1_000_000_007
    run(args)


if __name__ == '__main__':
    main()
