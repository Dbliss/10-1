"""Score bid networks on the same unseen self-play labels.

The labels are v2's rollout labels (one finish of the true deal under v2's
networks and the mixed population) on deals no run trained on. They are made
with v2's bid network widened to the bower layout, which predicts exactly
what v2 does, so the rows carry every input any layout needs: 98-input
networks read the first 98 columns, 102-input networks all of them.

    python -m ohhell.learn.bid_holdout runs/selfplay/bid_control/iter3 runs/selfplay/bid_bower/iter3
"""

import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from .features import BID_BOWER_INPUTS, BID_INPUTS
from .models import MODEL_DIR
from .nn import MLP, softmax_loss
from .selfplay import generate, widen_bid

V2 = MODEL_DIR


def holdout(rounds, seed, workers):
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        widen_bid(MLP.load(V2 / 'bid.npz'), BID_BOWER_INPUTS).save(folder / 'bid.npz')
        MLP.load(V2 / 'play.npz').save(folder / 'play.npz')
        bid_x, bid_k, bid_limit, _, _ = generate(
            rounds, seed, workers, folder, 0.6, 0.05, play_rate=0.0)
    return bid_x, bid_k, bid_limit


def score(model, bid_x, bid_k, bid_limit):
    inputs = model.weights[0].shape[0]
    if inputs not in (BID_INPUTS, BID_BOWER_INPUTS):
        raise ValueError(f'{inputs}-input networks are not covered')
    x = bid_x[:, :inputs]
    total, top1 = 0.0, 0
    for piece in np.array_split(np.arange(len(x)), max(1, len(x) // 8192)):
        logits = model(x[piece])
        loss, _ = softmax_loss(bid_limit)(bid_k, piece)(logits.copy())
        total += loss * len(piece)
        masked = np.where(np.arange(logits.shape[1]) > bid_limit[piece][:, None], -np.inf, logits)
        top1 += int((masked.argmax(axis=1) == bid_k[piece]).sum())
    return total / len(x), top1 / len(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('folders', nargs='+', help='folders holding bid.npz')
    parser.add_argument('--rounds', type=int, default=40000)
    parser.add_argument('--seed', type=int, default=987655)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    tick = time.time()
    bid_x, bid_k, bid_limit = holdout(args.rounds, args.seed, args.workers)
    print(f'{len(bid_x):,} held-out bid rows from {args.rounds:,} rounds, seed {args.seed} '
          f'({time.time() - tick:.0f}s)')
    for folder in [str(V2)] + args.folders:
        nll, top1 = score(MLP.load(Path(folder) / 'bid.npz'), bid_x, bid_k, bid_limit)
        print(f'{folder}: NLL {nll:.4f}  top-1 {top1:.3f}')


if __name__ == '__main__':
    main()
