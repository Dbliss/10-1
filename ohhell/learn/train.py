"""Train the bid and play models on simulated tables of mixed opponents.

At a sampled decision, every option is tried from that exact position and the
round is finished with each opponent playing as its kind would. The input
features describe only what the seat could see, so averaged over many deals the
model learns each option's value given the seat's information.

    python -m ohhell.learn.train --out runs
"""

import argparse
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..arena import deal_round
from ..bots.heuristic import best_bid_for, card_win_probs, trick_distribution
from ..engine import BIDDING, PLAYING, RoundState, legal_bids
from ..opponents import FIELDS, KINDS, NOVICE, SKILLED, PopulationPolicy, noisy_bid
from ..rules import Rules
from ..search import playout
from .features import BID_INPUTS, MAX_TRICKS, PLAY_INPUTS, bid_matrix, play_matrix
from .models import MODEL_DIR, VALUE_SCALE
from .nn import MLP, fit, huber_loss, softmax_loss

PLAYER_WEIGHTS = {2: 1, 3: 2, 4: 6, 5: 2, 6: 1, 7: 1, 8: 1}
BID_SAMPLE_RATE = 0.3
PLAY_SAMPLE_RATE = 0.15


def _estimate(hand, state, seat):
    leads = seat == (state.dealer + 1) % state.n
    return card_win_probs(hand, state.trump_card, state.n, state.hand_size, leads)


def _learner_policy(kinds, seat, rng):
    """The sampled seat plays well; everyone else plays as their kind."""
    table = list(kinds)
    table[seat] = SKILLED
    return PopulationPolicy(table, rng)


def _record_bids(state, earlier, kinds, rng, xs, ks, limits):
    """Try each plausible legal bid for sampled seats on the same deal."""
    for seat in range(state.n):
        if kinds[seat] is NOVICE or rng.random() >= BID_SAMPLE_RATE:
            continue
        hand = state.hands[seat]
        estimate = sum(_estimate(hand, state, seat))
        low = max(0, math.floor(estimate) - 2)
        high = min(state.hand_size, math.ceil(estimate) + 3)
        legal = legal_bids(state.rules, state.hand_size, earlier[seat])
        options = [b for b in legal if low <= b <= high] or legal
        xs.append(bid_matrix(hand, state.trump_card, state.n, state.hand_size, seat,
                             state.dealer, earlier[seat], options))
        policy = _learner_policy(kinds, seat, rng)
        dealer = state.dealer
        for bid in options:
            trial = state.clone()
            trial.bids[seat] = bid
            # A different bid here can leave the dealer's bid illegal.
            if seat != dealer and sum(trial.bids) == state.hand_size:
                trial.bids[dealer] += 1 if trial.bids[dealer] == 0 else -1
            playout(trial, policy)
            ks.append(trial.tricks_won[seat])
            limits.append(state.hand_size)


def _generate(job):
    """Play ``rounds`` rounds and return training arrays."""
    seed, rounds = job
    rng = random.Random(seed)
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
        kinds = rng.choices(KINDS, FIELDS['mix'], k=n)

        earlier = {}
        while state.phase == BIDDING:
            seat = state.to_act
            earlier[seat] = list(state.bids)
            options = state.legal_actions()
            dist = trick_distribution(_estimate(state.hands[seat], state, seat))
            bid = best_bid_for(dist, options, state.rules.scoring)
            state.apply(noisy_bid(bid, kinds[seat], rng, options))
        _record_bids(state, earlier, kinds, rng, bid_x, bid_k, bid_limit)

        table = PopulationPolicy(kinds, rng)
        while state.phase == PLAYING:
            seat = state.to_act
            cards = state.legal_actions()
            if (len(cards) > 1 and kinds[seat] is not NOVICE
                    and rng.random() < PLAY_SAMPLE_RATE):
                play_x.append(play_matrix(state, seat, state.hands[seat], cards))
                policy = _learner_policy(kinds, seat, rng)
                for card in cards:
                    trial = state.clone()
                    trial.play_unchecked(card)
                    playout(trial, policy)
                    play_y.append(trial.points()[seat] / VALUE_SCALE)
            state.play_unchecked(table(state))

    return (np.concatenate(bid_x) if bid_x else np.zeros((0, BID_INPUTS), np.float32),
            np.array(bid_k, dtype=np.int64), np.array(bid_limit, dtype=np.int64),
            np.concatenate(play_x) if play_x else np.zeros((0, PLAY_INPUTS), np.float32),
            np.array(play_y, dtype=np.float32))


def generate(rounds, seed, workers):
    # Workers each do small matrix maths, so stop them spawning BLAS threads.
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    chunks = workers * 8
    jobs = [(seed * 100003 + i, rounds // chunks) for i in range(chunks)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_generate, jobs))
    return [np.concatenate([p[i] for p in parts]) for i in range(5)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--rounds', type=int, default=400000)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--out', default=str(MODEL_DIR / 'candidate'),
                        help='where to save; never the default folder, which holds the current best')
    args = parser.parse_args()

    began = time.time()
    bid_x, bid_k, bid_limit, play_x, play_y = generate(args.rounds, args.seed, args.workers)
    print(f"{len(bid_x):,} bid and {len(play_x):,} play samples "
          f"in {time.time() - began:.0f}s", flush=True)

    log = lambda text: print(text, flush=True)
    bid = MLP([BID_INPUTS, 192, 96, MAX_TRICKS + 1], seed=args.seed)
    print("bid model", flush=True)
    fit(bid, bid_x, bid_k, softmax_loss(bid_limit), epochs=args.epochs, seed=args.seed, log=log)
    play = MLP([PLAY_INPUTS, 320, 160, 1], seed=args.seed + 50)
    print("play model", flush=True)
    fit(play, play_x, play_y, huber_loss(), epochs=args.epochs, seed=args.seed, log=log)

    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    bid.save(folder / 'bid.npz')
    play.save(folder / 'play.npz')
    print(f"saved to {folder} after {time.time() - began:.0f}s", flush=True)


if __name__ == '__main__':
    main()
