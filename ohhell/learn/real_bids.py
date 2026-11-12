"""Retrain the bid model on what happens when the real bot plays its hand.

``train.py`` learns trick odds by finishing each round with the rule-based
policy, which is much weaker at hitting a target than the bot that actually
plays. Here full rounds are played by the bot itself, with each bid sometimes
nudged so that the model also sees how bids it wouldn't choose turn out.

    python -m ohhell.learn.real_bids --out ohhell/models/real_bids --minutes 10
"""

import argparse
import multiprocessing
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np

from ..arena import deal_round
from ..bots import DEFAULT_SPEC, make_bot
from ..engine import BIDDING, DONE, RoundState
from ..rules import Rules
from .features import BID_INPUTS, MAX_TRICKS, bid_matrix
from .models import MODEL_DIR
from .nn import MLP, fit, softmax_loss

# Six players is what the tournament tests and four is the game's usual table.
PLAYER_WEIGHTS = {2: 1, 3: 2, 4: 6, 5: 2, 6: 6, 7: 1, 8: 1}
LEANS = (0.0, 0.0, 0.0, 0.23, 0.36, -0.25, -0.5)  # as in the tournament lobby
NUDGE_RATE = 0.2


def _play_rounds(job):
    """Play ``rounds`` rounds and return (inputs, tricks taken, hand sizes)."""
    seed, rounds, spec = job
    rng = random.Random(seed)
    players = list(PLAYER_WEIGHTS)
    weights = list(PLAYER_WEIGHTS.values())
    xs, tricks, limits = [], [], []

    for _ in range(rounds):
        n = rng.choices(players, weights)[0]
        top = min(MAX_TRICKS, 51 // n)
        hand_size = rng.randint(1, top)
        hands, trump = deal_round(rng, n, hand_size)
        state = RoundState(Rules(n, top), hands, trump, rng.randrange(n))
        bots = [
            make_bot(f'{spec},confidence={rng.choice(LEANS)}', rng.getrandbits(31))
            for _ in range(n)
        ]

        rows = []
        while state.phase == BIDDING:
            seat = state.to_act
            options = state.legal_actions()
            bid = bots[seat].act(state.observation(seat))
            if rng.random() < NUDGE_RATE:
                nudged = bid + rng.choice((-1, 1))
                if nudged in options:
                    bid = nudged
            rows.append((seat, bid_matrix(state.hands[seat], trump, n, hand_size, seat,
                                          state.dealer, list(state.bids), [bid])))
            state.apply(bid)

        while state.phase != DONE:
            state.apply(bots[state.to_act].act(state.observation(state.to_act)))

        for seat, row in rows:
            xs.append(row)
            tricks.append(state.tricks_won[seat])
            limits.append(hand_size)

    return (np.concatenate(xs) if xs else np.zeros((0, BID_INPUTS), np.float32),
            np.array(tricks, dtype=np.int64), np.array(limits, dtype=np.int64))


def collect(spec, minutes, workers, seed):
    """Play rounds on every worker until the time is up."""
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    stop = time.monotonic() + minutes * 60.0
    parts = []
    jobs = ((seed * 100003 + i, 50, spec) for i in range(10 ** 9))
    with multiprocessing.Pool(workers) as pool:
        for part in pool.imap_unordered(_play_rounds, jobs):
            parts.append(part)
            if len(parts) % 200 == 0:
                print(f'{sum(len(p[0]) for p in parts):,} bids', flush=True)
            if time.monotonic() >= stop:
                break
    return [np.concatenate([p[i] for p in parts]) for i in range(3)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('--spec', default=DEFAULT_SPEC, help='the bot that plays the rounds')
    parser.add_argument('--play-from', default=str(MODEL_DIR), help='folder to copy play.npz from')
    parser.add_argument('--minutes', type=float, default=10.0)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    bid_x, bid_k, bid_limit = collect(args.spec, args.minutes, args.workers, args.seed)
    print(f'{len(bid_x):,} bids from real play', flush=True)
    model = MLP([BID_INPUTS, 192, 96, MAX_TRICKS + 1], seed=args.seed)
    log = lambda text: print(text, flush=True)
    fit(model, bid_x, bid_k, softmax_loss(bid_limit), epochs=args.epochs, seed=args.seed, log=log)

    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    model.save(folder / 'bid.npz')
    shutil.copy(Path(args.play_from) / 'play.npz', folder / 'play.npz')
    print(f'saved to {folder}', flush=True)


if __name__ == '__main__':
    main()
