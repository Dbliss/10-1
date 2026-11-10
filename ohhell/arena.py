"""Measure bots against each other on identical deals.

For each game seed, one control table is all baseline bots. Then the candidate
takes each seat in turn against the baseline, on exactly the same cards. The
reported edge is the candidate's score minus the control score from that seat,
averaged across seats, so the luck of the deal mostly cancels.

    python -m ohhell.arena heuristic legacy --games 500
"""

import argparse
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from .bots import make_bot
from .cards import mask_of
from .engine import DONE, RoundState
from .rules import Rules


def deal_round(rng, n, hand_size):
    deck = list(range(52))
    rng.shuffle(deck)
    hands = [mask_of(deck[1 + i * hand_size:1 + (i + 1) * hand_size]) for i in range(n)]
    return hands, deck[0]


def play_game(bots, rules, seed, timed_seat=None):
    """Play a whole game. Returns (total per seat, seconds, decisions)."""
    rng = random.Random(seed)  # deals depend only on the seed, never on play
    n = rules.num_players
    dealer = rng.randrange(n)
    totals = [0] * n
    seconds = 0.0
    decisions = 0
    for hand_size in rules.hand_sizes:
        hands, trump = deal_round(rng, n, hand_size)
        state = RoundState(rules, hands, trump, dealer)
        while state.phase != DONE:
            seat = state.to_act
            obs = state.observation(seat)
            if seat == timed_seat:
                start = time.perf_counter()
                action = bots[seat].act(obs)
                seconds += time.perf_counter() - start
                decisions += 1
            else:
                action = bots[seat].act(obs)
            state.apply(action)
        for seat, points in enumerate(state.points()):
            totals[seat] += points
        dealer = (dealer + 1) % n
    return totals, seconds, decisions


@dataclass
class Report:
    candidate: str
    baseline: str
    games: int
    edge: float
    interval: float
    candidate_score: float
    baseline_score: float
    win_rate: float
    fair_win_rate: float
    ms_per_decision: float

    def __str__(self):
        return (
            f"{self.candidate} vs {self.baseline}: "
            f"edge {self.edge:+.1f} +/- {self.interval:.1f} points per game "
            f"({self.candidate_score:.1f} vs {self.baseline_score:.1f}), "
            f"wins {self.win_rate:.1%} (fair {self.fair_win_rate:.1%}), "
            f"{self.ms_per_decision:.1f} ms/decision, {self.games} games")


def _table_seed(game, label):
    return (game * 1009 + label) & 0x7FFFFFFF


def _compare_one(job):
    candidate, baseline, rules, game = job
    n = rules.num_players
    control = [make_bot(baseline, _table_seed(game, s)) for s in range(n)]
    control_totals, _, _ = play_game(control, rules, game)

    edges, scores, wins = [], [], 0.0
    seconds, decisions = 0.0, 0
    for seat in range(n):
        bots = [make_bot(baseline, _table_seed(game, s)) for s in range(n)]
        bots[seat] = make_bot(candidate, _table_seed(game, 100 + seat))
        totals, spent, count = play_game(bots, rules, game, timed_seat=seat)
        seconds += spent
        decisions += count
        edges.append(totals[seat] - control_totals[seat])
        scores.append(totals[seat])
        top = max(totals)
        if totals[seat] == top:
            wins += 1.0 / totals.count(top)
    return (sum(edges) / n, sum(scores) / n, sum(control_totals) / n,
            wins / n, seconds, decisions)


def compare(candidate, baseline, rules, games, workers=None, first_seed=0):
    jobs = [(candidate, baseline, rules, first_seed + g) for g in range(games)]
    if workers == 1:
        results = [_compare_one(job) for job in jobs]
    else:
        os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_compare_one, jobs, chunksize=max(1, games // 80)))

    edges = [r[0] for r in results]
    mean = sum(edges) / games
    spread = sum((e - mean) ** 2 for e in edges) / max(1, games - 1)
    interval = 1.96 * math.sqrt(spread / games)
    total_seconds = sum(r[4] for r in results)
    total_decisions = sum(r[5] for r in results)
    return Report(
        candidate, baseline, games, mean, interval,
        sum(r[1] for r in results) / games,
        sum(r[2] for r in results) / games,
        sum(r[3] for r in results) / games,
        1.0 / rules.num_players,
        1000.0 * total_seconds / max(1, total_decisions))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('candidate')
    parser.add_argument('baseline')
    parser.add_argument('--games', type=int, default=200)
    parser.add_argument('--players', type=int, default=4)
    parser.add_argument('--hand', type=int, default=10)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--no-dealer-restriction', action='store_true')
    parser.add_argument('--first-seed', type=int, default=0)
    args = parser.parse_args()
    rules = Rules(args.players, args.hand, not args.no_dealer_restriction)
    print(compare(args.candidate, args.baseline, rules, args.games,
                  args.workers, args.first_seed), flush=True)


if __name__ == '__main__':
    main()
