"""Compare bid forecasts with fixed opponent bids and matched sampled deals."""

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
from ..bots.heuristic import state_card
from ..duel import FIELD, REFERENCE_SPEC, draw_round, extend, rounds_per_game
from ..engine import DONE, NO_BID, RoundState
from ..native import MAX_SEATS
from ..opponents import KINDS, noisy_bid
from ..tournament import check_run_config
from .value_bid import _with_our_bid


POLICIES = ('cheap', 'live_self', 'live_all')


def _worlds(obs, sampler, options, count, seed):
    hands, kinds = sampler.sample_worlds(obs, count)
    if len(hands) != count:
        raise ValueError('Native sampler returned fewer worlds than requested')
    rng = random.Random(seed)
    tables = np.zeros((count, len(options), MAX_SEATS), dtype=np.int32)
    for w, (deal, players) in enumerate(zip(hands, kinds)):
        bids = list(obs.bids)
        for s, bid in enumerate(bids):
            if bid == NO_BID and s != obs.seat:
                imagined = sampler.heuristic_bid(int(deal[s]), obs.trump_card, obs.n,
                                                  obs.hand_size, s, obs.dealer)
                bids[s] = noisy_bid(imagined, KINDS[players[s]], rng, range(obs.hand_size + 1))
        for i, option in enumerate(options):
            tables[w, i, :obs.n] = _with_our_bid(bids, option, obs)
    return hands, kinds, tables


def _live_counts(obs, worlds, spec, seed, own_only, deadline=None):
    hands, kinds, tables = worlds
    counts = np.zeros((tables.shape[1], obs.hand_size + 1))
    rng = random.Random(seed)
    for deal, players, alternatives in zip(hands, kinds, tables):
        finishes = []
        fixed = [int(h) for h in deal[:obs.n]]
        noise = [KINDS[k].card_noise for k in players[:obs.n]]
        branch_seed = rng.randrange(1 << 30)
        for i, bids in enumerate(alternatives):
            state = RoundState(obs.rules, fixed.copy(), obs.trump_card, obs.dealer)
            for _ in fixed:
                state.apply(int(bids[state.to_act]))
            # Common seeds across candidate bids and the two live policies.
            cards_rng = [random.Random(branch_seed + s) for s in range(obs.n)]
            bots = [make_bot(spec, branch_seed + 8 + s)
                    if noise[s] < 1 and (not own_only or s == obs.seat) else None
                    for s in range(obs.n)]
            while state.phase != DONE:
                if deadline is not None and time.perf_counter() >= deadline:
                    return counts
                s = state.to_act
                if noise[s] and cards_rng[s].random() < noise[s]:
                    card = cards_rng[s].choice(state.legal_actions())
                elif bots[s] is None:
                    card = state_card(state)
                else:
                    card = bots[s].act(state.observation(s))
                state.play_unchecked(card)
            finishes.append(state.tricks_won[obs.seat])
        for i, tricks in enumerate(finishes):
            counts[i, tricks] += 1
    return counts


def bid_position(seed):
    n, rules, size, hands, trump, dealer, seat, opponents, seeds = draw_round(seed, FIELD)
    if not size:
        return None
    state = RoundState(rules, list(hands), trump, dealer)
    while state.to_act != seat:
        s = state.to_act
        state.apply(make_bot(opponents[s], seeds[s]).act(state.observation(s)))
    obs = state.observation(seat)
    bot = make_bot(REFERENCE_SPEC, seeds[seat])
    options = bot._bid_options(obs)
    if len(options) < 2:
        return None
    return obs, bot, options, seeds[seat]


def audit_position(job):
    seed, count, card_budget = job
    prepared = bid_position(seed)
    if prepared is None:
        return None
    obs, bot, options, bot_seed = prepared
    size, n, trump, seat, dealer = obs.hand_size, obs.n, obs.trump_card, obs.seat, obs.dealer

    sample = _worlds(obs, bot.simulator, options, count, seed)
    cheap = np.zeros((len(options), size + 1))
    bot.simulator.score_bids(obs, *sample, cheap)
    live_spec = extend(REFERENCE_SPEC, f'card_budget={card_budget},endgame_budget={card_budget}')
    own = _live_counts(obs, sample, live_spec, seed + (1 << 31), own_only=True)
    live = _live_counts(obs, sample, live_spec, seed + (1 << 31), own_only=False)

    held_out = make_bot(REFERENCE_SPEC, bot_seed + (1 << 30))
    unseen = _worlds(obs, held_out.simulator, options, count, seed + (1 << 30))
    reference = _live_counts(obs, unseen, live_spec, seed + (1 << 32), own_only=False)
    net = bot.models.bid_distributions(obs.hand, trump, n, size, seat, dealer, obs.bids, options)
    return dict(seed=seed, players=n, cards=size, bids=options,
                cheap=(cheap / count).tolist(), live_self=(own / count).tolist(),
                live_all=(live / count).tolist(), reference=(reference / count).tolist(),
                net=[net[b] for b in options], blend=bot.bid_share(size, n))


def summarize(rows, policies=POLICIES):
    from ..rules import Scoring

    scoring = Scoring()
    records = []
    for row in rows:
        bids = row['bids']
        actual = np.array(row['reference'])
        values = np.array([[scoring.points(b, k) for k in range(row['cards'] + 1)] for b in bids])
        entry = {}
        for policy in policies:
            pred = np.array(row[policy])
            mixed = (1 - row['blend']) * pred + row['blend'] * np.array(row['net'])
            winner = int(np.argmax(np.sum(mixed * values, axis=1)))
            make = pred[np.arange(len(bids)), bids]
            made = actual[np.arange(len(bids)), bids]
            entry[policy] = dict(
                brier=float(np.mean(make**2 - 2 * make * made + made)),
                make_bias=float(np.mean(make - made)),
                trick_bias=float(np.mean((pred - actual) @ np.arange(row['cards'] + 1))),
                ranked_score=float(np.mean(np.sum((np.cumsum(pred - actual, axis=1))**2, axis=1))),
                winner=winner, held_out_points=float(actual[winner] @ values[winner]),
            )
        records.append((row, entry))

    result = {}
    for label, group in [('all', records), ('8_to_10_cards', [(r, m) for r, m in records if r['cards'] >= 8])]:
        if not group:
            continue
        policy_results = {}
        for policy in policies:
            metrics = {key: statistics.mean(m[policy][key] for _, m in group)
                       for key in ('brier', 'make_bias', 'trick_bias', 'ranked_score')}
            edges = [(m[policy]['held_out_points'] - m['cheap']['held_out_points']) * rounds_per_game(r['players'])
                     for r, m in group]
            metrics.update(changed_bids=sum(m[policy]['winner'] != m['cheap']['winner'] for _, m in group),
                           model_points_per_game=statistics.mean(edges),
                           margin_95=1.96 * statistics.stdev(edges) / math.sqrt(len(edges)) if len(edges) > 1 else 0)
            policy_results[policy] = metrics
        result[label] = dict(positions=len(group), policies=policy_results)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--positions', type=int, required=True)
    parser.add_argument('--worlds', type=int, default=32)
    parser.add_argument('--card-budget', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if min(args.positions, args.worlds, args.card_budget, args.workers) < 1:
        parser.error('counts and budgets must be positive')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest, summary = (args.out.with_suffix(ext) for ext in ('.meta', '.json'))
    if any(p.exists() for p in (args.out, manifest, summary)):
        parser.error('output already exists')
    config = SimpleNamespace(seed=args.seed, max_sets=args.positions, smallest_gain=2,
                             time_ms=0, resume=False)
    check_run_config(manifest, config, [REFERENCE_SPEC, extend(REFERENCE_SPEC,
                     f'card_budget={args.card_budget},endgame_budget={args.card_budget}'),
                     *(spec for spec, _ in FIELD)])
    jobs = ((args.seed + i, args.worlds, args.card_budget) for i in range(args.positions))
    rows = []
    with multiprocessing.Pool(args.workers) as pool, args.out.open('x') as output:
        for i, row in enumerate(pool.imap(audit_position, jobs, chunksize=1), 1):
            if row is not None:
                rows.append(row)
                output.write(json.dumps(row) + '\n')
            if i % 25 == 0:
                output.flush()
                print(f'{i}/{args.positions} positions', flush=True)
    result = dict(worlds_per_split=args.worlds, live_card_budget=args.card_budget,
                  drawn_positions=args.positions, **summarize(rows))
    summary.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
