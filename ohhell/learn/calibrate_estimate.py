"""Fit and check the version 2 quick trick estimate (``ohhell/estimate.py``).

Deals random rounds of 2-8 players and every hand size, has every seat bid
with the original rule of thumb and play with the rule-based card player
(the same policy the simulations use), and records whether each card won its
trick. A logistic model per card type (trump or side suit) is fitted on one
set of deals and scored, against the original formula, on unseen deals.

    python -m ohhell.learn.calibrate_estimate --train 60000 --test 20000 --write

``--bidders 2`` or ``3`` has the table bid with that estimate version instead
(its saved coefficients and spread), so the fit describes tables that bid the
way the simulation imagines them. ``--version 3`` writes the version 3 file.
Version 3 also fits a spread: how much flatter or sharper than independent
cards the trick count is, as ``estimate.widen``'s temperature.

    python -m ohhell.learn.calibrate_estimate --version 3 --bidders 2 --write
    python -m ohhell.learn.calibrate_estimate --version 3 --bidders 3 --write
"""

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import estimate
from ..arena import deal_round
from ..bots.heuristic import best_bid_for, card_win_probs, state_card, trick_distribution
from ..cards import EFFECTIVE_SUIT, POWER, SUIT_CARDS, SUIT_OF, beats, cards_in
from ..engine import BIDDING, DONE, RoundState
from ..rules import Rules


def _old_probs(hand, trump_card, n, hand_size, leads):
    """The original estimate keyed by card (it lists cards in this order)."""
    trump = SUIT_OF[trump_card]
    order = []
    for suit in range(4):
        order += sorted(cards_in(hand & SUIT_CARDS[trump][suit]),
                        key=POWER[trump].__getitem__, reverse=True)
    return dict(zip(order, card_win_probs(hand, trump_card, n, hand_size, leads)))


def _winning_cards(history, n, trump):
    winners = set()
    for i in range(0, len(history), n):
        trick = history[i:i + n]
        best = trick[0][1]
        for _, card in trick[1:]:
            if beats(card, best, trump):
                best = card
        winners.add(best)
    return winners


def _bid(state, seat, old, bidders, rules):
    if bidders == 1:
        dist = trick_distribution(list(old[seat].values()))
    else:
        dist = estimate.bid_distribution(
            state.hands[seat], state.trump_card, state.n, state.hand_size,
            estimate.first_trick_position(seat, state.dealer, state.n), version=bidders)
    return best_bid_for(dist, state.legal_actions(), rules.scoring)


def play_rounds(count, seed, bidders=1):
    """Yield one record per seat: its round, cards, features and outcome.

    ``bidders`` is the estimate version every seat bids with (1, the original
    rule of thumb, unless given).
    """
    rng = random.Random(seed)
    for _ in range(count):
        n = rng.randint(2, 8)
        top = min(10, 51 // n)
        hand_size = rng.randint(1, top)
        dealer = rng.randrange(n)
        hands, trump_card = deal_round(rng, n, hand_size)
        rules = Rules(n, top)
        state = RoundState(rules, hands, trump_card, dealer)
        first = (dealer + 1) % n
        old = [_old_probs(hands[s], trump_card, n, hand_size, s == first) for s in range(n)]
        while state.phase == BIDDING:
            state.apply(_bid(state, state.to_act, old, bidders, rules))
        while state.phase != DONE:
            state.play_unchecked(state_card(state))
        winners = _winning_cards(state.history, n, SUIT_OF[trump_card])
        for seat in range(n):
            position = estimate.first_trick_position(seat, dealer, n)
            yield {
                'n': n, 'hand_size': hand_size, 'hand': hands[seat], 'trump_card': trump_card,
                'position': position, 'old': old[seat], 'winners': winners,
                'tricks': state.tricks_won[seat],
                'rows': estimate.card_features(hands[seat], trump_card, n, hand_size, position),
            }


def fit_logistic(x, y, l2=1e-3, iterations=30):
    """Newton's method for logistic regression with a small ridge penalty."""
    w = np.zeros(x.shape[1])
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))
        gradient = x.T @ (p - y) + l2 * len(y) * w
        hessian = (x * (p * (1 - p))[:, None]).T @ x + l2 * len(y) * np.eye(x.shape[1])
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w


def fit(records):
    groups = {False: ([], []), True: ([], [])}
    for record in records:
        for card, is_trump, features, certain in record['rows']:
            if not certain:
                xs, ys = groups[is_trump]
                xs.append(features)
                ys.append(card in record['winners'])
    weights = {}
    for is_trump, (xs, ys) in groups.items():
        x, y = np.array(xs), np.array(ys, dtype=float)
        weights['trump' if is_trump else 'side'] = fit_logistic(x, y).tolist()
    return weights, {k: len(v[1]) for k, v in (('side', groups[False]), ('trump', groups[True]))}


def _nll(dist, k):
    return -math.log(max(1e-9, dist[k]))


def fit_spread(records, weights, passes=4):
    """The ``(a, b, c)`` of ``estimate.spread_temperature`` with the lowest
    trick-count NLL, by coordinate search on a shrinking grid."""
    pairs = (weights['side'], weights['trump'])
    dists, ks, hands, ns = [], [], [], []
    for record in records:
        probs = estimate.card_win_probs(record['hand'], record['trump_card'], record['n'],
                                        record['hand_size'], record['position'], pairs)
        dist = trick_distribution(list(probs.values()))
        dists.append(dist + [0.0] * (11 - len(dist)))
        ks.append(record['tricks'])
        hands.append(record['hand_size'])
        ns.append(record['n'])
    logs = np.log(np.clip(np.array(dists), 1e-12, None))
    present = np.arange(11)[None, :] <= np.array(hands)[:, None]
    ks, hands, ns = np.array(ks), np.array(hands), np.array(ns)

    def nll(coefficients):
        a, b, c = coefficients
        temperature = np.maximum(0.2, a + b * hands / 10.0 + c * ns / 8.0)
        scaled = np.where(present, logs / temperature[:, None], -np.inf)
        top = scaled.max(axis=1, keepdims=True)
        norm = top[:, 0] + np.log(np.exp(scaled - top).sum(axis=1))
        return float((norm - scaled[np.arange(len(ks)), ks]).mean())

    best = [1.0, 0.0, 0.0]
    step = 0.4
    for _ in range(passes):
        for i in range(3):
            grid = best[i] + step * np.linspace(-1.0, 1.0, 17)
            scores = []
            for value in grid:
                trial = list(best)
                trial[i] = float(value)
                scores.append(nll(trial))
            best[i] = float(grid[int(np.argmin(scores))])
        step /= 4.0
    best = tuple(round(v, 4) for v in best)
    return best, nll((1.0, 0.0, 0.0)), nll(best)


def evaluate(records, weights):
    """Compare the original and the new estimate on the same seats.

    The new estimate's trick-count NLL includes its spread, if it has one.
    """
    temperature = weights.get('spread', estimate.NO_SPREAD)
    weights = (weights['side'], weights['trump'])
    by_size = defaultdict(lambda: defaultdict(float))
    tables = defaultdict(lambda: defaultdict(float))
    cards = defaultdict(float)
    calibration = defaultdict(lambda: [0, 0.0, 0])
    seats = defaultdict(lambda: [0, 0, 0.0, 0.0])
    for record in records:
        new = estimate.card_win_probs(record['hand'], record['trump_card'], record['n'],
                                      record['hand_size'], record['position'], weights)
        old = record['old']
        tricks = record['tricks']
        # Move order: side-suit Aces in rounds of 7+ cards, by seat in the first trick.
        if record['hand_size'] >= 7:
            n, position = record['n'], record['position']
            where = 'lead' if position == 0 else 'last' if position == n - 1 else 'middle'
            for card, is_trump, features, _ in record['rows']:
                if not is_trump and card % 13 == 12:
                    b = seats[(min(n, 6), where)]
                    b[0] += 1
                    b[1] += card in record['winners']
                    b[2] += new[card]
                    b[3] += old[card]
        for name, probs in (('old', old), ('new', new)):
            pred = sum(probs.values())
            row = by_size[record['hand_size']]
            row['count_' + name] += 1
            row['pred_' + name] += pred
            row['sq_' + name] += (pred - tricks) ** 2
            dist = trick_distribution(list(probs.values()))
            if name == 'new':
                dist = estimate.widen(dist, estimate.spread_temperature(
                    record['hand_size'], record['n'], temperature))
            row['nll_' + name] += _nll(dist, tricks)
            key = (record['n'], record['hand_size'], record['trump_card'], id(record['winners']))
            tables[key][name] += pred
            for card, p in probs.items():
                won = card in record['winners']
                cards['brier_' + name] += (p - won) ** 2
                cards['count_' + name] += 1
                if name == 'new':
                    kind = 'trump' if EFFECTIVE_SUIT[SUIT_OF[record['trump_card']]][card] == SUIT_OF[record['trump_card']] else 'side'
                    b = calibration[(kind, min(9, int(p * 10)))]
                    b[0] += 1
                    b[1] += p
                    b[2] += won
        by_size[record['hand_size']]['actual'] += tricks
        by_size[record['hand_size']]['seats'] += 1
        tables[key]['hand_size'] = record['hand_size']

    table_gap = defaultdict(lambda: defaultdict(float))
    for table in tables.values():
        gap = table_gap[int(table['hand_size'])]
        gap['tables'] += 1
        gap['old'] += table['old'] - table['hand_size']
        gap['new'] += table['new'] - table['hand_size']

    report = {'by_hand_size': {}, 'cards': {}, 'calibration': {}, 'side_aces_by_seat': {}}
    order = {'lead': 0, 'middle': 1, 'last': 2}
    for (n, where), (count, won, p_new, p_old) in sorted(seats.items(), key=lambda kv: (kv[0][0], order[kv[0][1]])):
        report['side_aces_by_seat'][f'{n}{"+" if n == 6 else ""} players, {where}'] = {
            'aces': count, 'actual': won / count, 'new': p_new / count, 'old': p_old / count}
    totals = defaultdict(float)
    for size in sorted(by_size):
        row = by_size[size]
        seats = row['seats']
        entry = {'seats': int(seats), 'actual': row['actual'] / seats}
        for name in ('old', 'new'):
            entry[name] = {
                'bias': (row['pred_' + name] - row['actual']) / seats,
                'rmse': math.sqrt(row['sq_' + name] / seats),
                'nll': row['nll_' + name] / seats,
                'table_sum_gap': table_gap[size][name] / table_gap[size]['tables'],
            }
            totals['sq_' + name] += row['sq_' + name]
            totals['nll_' + name] += row['nll_' + name]
            totals['bias_' + name] += row['pred_' + name] - row['actual']
        totals['seats'] += seats
        report['by_hand_size'][size] = entry
    for name in ('old', 'new'):
        report['cards'][name] = {
            'brier': cards['brier_' + name] / cards['count_' + name],
            'bias': totals['bias_' + name] / totals['seats'],
            'rmse': math.sqrt(totals['sq_' + name] / totals['seats']),
            'nll': totals['nll_' + name] / totals['seats'],
        }
    for (kind, bucket), (count, p, won) in sorted(calibration.items()):
        report['calibration'][f'{kind} {bucket / 10:.1f}'] = {
            'cards': count, 'predicted': p / count, 'actual': won / count}
    return report


def print_report(report):
    print('Per seat, unseen deals (old = original formula, new = version 2)')
    print(' hand  seats  actual |   old bias  rmse   nll  table |   new bias  rmse   nll  table')
    for size, e in report['by_hand_size'].items():
        o, n = e['old'], e['new']
        print(f'  {size:>3} {e["seats"]:>6}  {e["actual"]:5.2f}  | '
              f'{o["bias"]:+9.2f} {o["rmse"]:5.2f} {o["nll"]:5.2f} {o["table_sum_gap"]:+6.2f} | '
              f'{n["bias"]:+9.2f} {n["rmse"]:5.2f} {n["nll"]:5.2f} {n["table_sum_gap"]:+6.2f}')
    for name in ('old', 'new'):
        c = report['cards'][name]
        print(f'  {name}: bias {c["bias"]:+.3f}  rmse {c["rmse"]:.3f}  '
              f'trick-count NLL {c["nll"]:.3f}  card Brier {c["brier"]:.4f}')
    print('Side-suit Aces, 7+ cards, by seat in the first trick')
    for key, c in report['side_aces_by_seat'].items():
        print(f'  {key:20} {c["aces"]:>6}  actual {c["actual"]:.1%}  new {c["new"]:.1%}  old {c["old"]:.1%}')
    print('New estimate calibration by predicted win chance')
    for key, c in report['calibration'].items():
        if c['cards'] >= 100:
            print(f'  {key:10} {c["cards"]:>8}  predicted {c["predicted"]:.3f}  actual {c["actual"]:.3f}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--train', type=int, default=60000, help='rounds to fit on')
    parser.add_argument('--test', type=int, default=20000, help='unseen rounds to score on')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--test-seed', type=int, default=1_000_001)
    parser.add_argument('--version', type=int, choices=(2, 3), default=2,
                        help='which estimate file to write; 3 adds a fitted spread')
    parser.add_argument('--bidders', type=int, choices=(1, 2, 3), default=1,
                        help='estimate version the table bids with (1: rule of thumb)')
    parser.add_argument('--spread-rounds', type=int, default=30000,
                        help='version 3: training rounds used to fit the spread')
    parser.add_argument('--write', action='store_true',
                        help="save the coefficients to that version's file")
    parser.add_argument('--report', type=Path, help='also save the report as JSON')
    args = parser.parse_args(argv)

    start = time.perf_counter()
    weights, rows = fit(play_rounds(args.train, args.seed, args.bidders))
    if args.version == 2:
        # Version 2 has no discard features: keep its original layout.
        weights['side'] = weights['side'][:len(estimate.SIDE) - len(estimate.DISCARDS)]
    print(f'Fitted on {args.train} rounds ({rows["side"]} side-suit and {rows["trump"]} trump '
          f'cards) in {time.perf_counter() - start:.0f} s')
    spread_info = None
    if args.version == 3:
        rounds = min(args.spread_rounds, args.train)
        weights['spread'], before, after = fit_spread(
            list(play_rounds(rounds, args.seed, args.bidders)), weights)
        spread_info = {'rounds': rounds, 'nll_without': before, 'nll_with': after}
        print(f'Spread {weights["spread"]}: train trick-count NLL {before:.4f} -> {after:.4f}')
    report = evaluate(list(play_rounds(args.test, args.test_seed, args.bidders)), weights)
    print_report(report)

    meta = {'train_rounds': args.train, 'seed': args.seed, 'test_rounds': args.test,
            'test_seed': args.test_seed, 'rows': rows, 'version': args.version,
            'bidders': args.bidders, 'spread_fit': spread_info,
            'features': {'side': estimate.SIDE, 'trump': estimate.TRUMP},
            'test_summary': report['cards']}
    if args.write:
        target = estimate.COEFFICIENT_FILES[args.version]
        target.write_text(json.dumps({**weights, 'trained': meta}, indent=1))
        print(f'Wrote {target}')
    if args.report:
        args.report.write_text(json.dumps({'trained': meta, **report}, indent=1))


if __name__ == '__main__':
    main()
