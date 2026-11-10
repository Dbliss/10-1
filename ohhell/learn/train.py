"""Train the bid and play models by self-play.

Each generation plays many rounds with the previous generation's policy. At a
sampled decision, every option is tried from that exact position and finished
by the policy, and the input features describe only what the seat could see.
Averaged over many deals, the model therefore learns each option's value given
the seat's information, without ever being shown hidden cards as input.

    python -m ohhell.learn.train --generations 2 --rounds 200000
"""

import argparse
import math
import os
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..arena import deal_round
from ..bots.heuristic import (
    best_bid_for, card_win_probs, state_card, trick_distribution,
)
from ..engine import BIDDING, NO_BID, PLAYING, RoundState
from ..rules import Rules
from ..search import playout
from .features import BID_INPUTS, MAX_TRICKS, PLAY_INPUTS, bid_matrix, play_matrix
from .models import MODEL_DIR, VALUE_SCALE, NetPolicy, load_models
from .nn import MLP, fit, huber_loss, softmax_loss

PLAYER_WEIGHTS = {2: 1, 3: 2, 4: 6, 5: 2, 6: 1, 7: 1, 8: 1}
BID_SAMPLE_RATE = 0.3
PLAY_SAMPLE_RATE = 0.15
RANDOM_CARD_RATE = 0.1
BID_NOISE_RATE = 0.3


def _estimate(hand, state, seat):
    leads = seat == (state.dealer + 1) % state.n
    return card_win_probs(hand, state.trump_card, state.n, state.hand_size, leads)


def _choose_bid(state, seat, models, rng):
    hand = state.hands[seat]
    options = state.legal_actions()
    score = state.rules.scoring
    if models is None:
        dist = trick_distribution(_estimate(hand, state, seat))
        bid = best_bid_for(dist, options, score)
    else:
        dists = models.bid_distributions(
            hand, state.trump_card, state.n, state.hand_size, seat, state.dealer,
            state.bids, options)
        values = {
            b: sum(p * score.points(b, k) for k, p in enumerate(d))
            for b, d in dists.items()
        }
        bid = max(values, key=values.get)
    if rng.random() < BID_NOISE_RATE:
        moved = bid + rng.choice((-1, 1))
        if moved in options:
            bid = moved
    return bid


def _record_bids(state, earlier, policy, rng, xs, ks, limits):
    """Try several bids per sampled seat from the same deal."""
    for seat in range(state.n):
        if rng.random() >= BID_SAMPLE_RATE:
            continue
        hand = state.hands[seat]
        estimate = sum(_estimate(hand, state, seat))
        low = max(0, math.floor(estimate) - 2)
        high = min(state.hand_size, math.ceil(estimate) + 3)
        options = list(range(low, high + 1))
        xs.append(bid_matrix(hand, state.trump_card, state.n, state.hand_size, seat,
                             state.dealer, earlier[seat], options))
        for bid in options:
            trial = state.clone()
            trial.bids[seat] = bid
            playout(trial, policy)
            ks.append(trial.tricks_won[seat])
            limits.append(state.hand_size)


def _generate(job):
    """Play ``rounds`` rounds and return training arrays."""
    seed, rounds, model_dir = job
    rng = random.Random(seed)
    models = load_models(model_dir) if model_dir else None
    policy = NetPolicy(models) if models else state_card
    players = list(PLAYER_WEIGHTS)
    weights = list(PLAYER_WEIGHTS.values())
    bid_x, bid_k, bid_limit, play_x, play_y = [], [], [], [], []

    for _ in range(rounds):
        n = rng.choices(players, weights)[0]
        top = min(MAX_TRICKS, 51 // n)
        hand_size = rng.randint(1, top)
        hands, trump = deal_round(rng, n, hand_size)
        state = RoundState(Rules(n, top), hands, trump, rng.randrange(n))
        state.history = None

        earlier = {}
        while state.phase == BIDDING:
            seat = state.to_act
            earlier[seat] = list(state.bids)
            state.apply(_choose_bid(state, seat, models, rng))
        _record_bids(state, earlier, policy, rng, bid_x, bid_k, bid_limit)

        while state.phase == PLAYING:
            seat = state.to_act
            cards = state.legal_actions()
            if len(cards) > 1 and rng.random() < PLAY_SAMPLE_RATE:
                play_x.append(play_matrix(state, seat, state.hands[seat], cards))
                for card in cards:
                    trial = state.clone()
                    trial.play_unchecked(card)
                    playout(trial, policy)
                    play_y.append(trial.points()[seat] / VALUE_SCALE)
            if len(cards) > 1 and rng.random() < RANDOM_CARD_RATE:
                state.play_unchecked(rng.choice(cards))
            else:
                state.play_unchecked(policy(state))

    return (np.concatenate(bid_x) if bid_x else np.zeros((0, BID_INPUTS), np.float32),
            np.array(bid_k, dtype=np.int64), np.array(bid_limit, dtype=np.int64),
            np.concatenate(play_x) if play_x else np.zeros((0, PLAY_INPUTS), np.float32),
            np.array(play_y, dtype=np.float32))


def generate(rounds, model_dir, seed, workers):
    # Workers each do small matrix maths, so stop them spawning BLAS threads.
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    chunks = workers * 8
    size = rounds // chunks
    jobs = [(seed * 100003 + i, size, model_dir) for i in range(chunks)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_generate, jobs))
    return [np.concatenate([p[i] for p in parts]) for i in range(5)]


def train_generation(index, rounds, previous, out, workers, epochs):
    began = time.time()
    bid_x, bid_k, bid_limit, play_x, play_y = generate(
        rounds, previous, index + 1, workers)
    print(f"generation {index}: {len(bid_x):,} bid and {len(play_x):,} play samples "
          f"in {time.time() - began:.0f}s", flush=True)

    log = lambda text: print(text, flush=True)
    bid = MLP([BID_INPUTS, 192, 96, MAX_TRICKS + 1], seed=index)
    print("bid model", flush=True)
    fit(bid, bid_x, bid_k, softmax_loss(bid_limit), epochs=epochs, seed=index, log=log)
    play = MLP([PLAY_INPUTS, 320, 160, 1], seed=index + 50)
    print("play model", flush=True)
    fit(play, play_x, play_y, huber_loss(), epochs=epochs, seed=index, log=log)

    folder = Path(out) / f'gen{index}'
    folder.mkdir(parents=True, exist_ok=True)
    bid.save(folder / 'bid.npz')
    play.save(folder / 'play.npz')
    print(f"generation {index} saved to {folder} after {time.time() - began:.0f}s",
          flush=True)
    return str(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--generations', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=200000)
    parser.add_argument('--later-rounds', type=int, default=80000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--out', default=str(MODEL_DIR / 'runs'))
    args = parser.parse_args()

    previous = None
    for index in range(args.generations):
        rounds = args.rounds if index == 0 else args.later_rounds
        previous = train_generation(index, rounds, previous, args.out, args.workers,
                                    args.epochs)


if __name__ == '__main__':
    main()
