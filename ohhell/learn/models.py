"""Loading trained models and asking them questions."""

from functools import lru_cache
from pathlib import Path

import numpy as np

from .features import (BID_BOWER_INPUTS, BID_INPUTS, bid_bower_part, bid_context,
                       bid_hand_part, bid_matrix, play_matrix)
from .nn import MLP

MODEL_DIR = Path(__file__).parent.parent / 'models'
VALUE_SCALE = 30.0  # play model outputs are points divided by this
MAX_BID_ROWS = 11  # bids 0-10, the bid network's candidate-bid inputs 86-96


class Models:
    def __init__(self, bid, play):
        self.bid = bid
        self.play = play
        self.bid_inputs = bid.weights[0].shape[0]
        self.play_inputs = play.weights[0].shape[0]

    def bid_distributions(self, hand, trump_card, n, hand_size, seat, dealer, bids,
                          options):
        """For each bid, the predicted chance of taking exactly k tricks."""
        logits = self.bid(bid_matrix(hand, trump_card, n, hand_size, seat, dealer,
                                     bids, options, self.bid_inputs))
        logits = logits[:, :hand_size + 1]
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        return {bid: probs[i].tolist() for i, bid in enumerate(options)}

    def choose_bids(self, hands, tables, trump_card, n, hand_size, seat, dealer,
                    candidates, valid, score):
        """The bid with the best predicted points for ``seat`` at many tables.

        ``hands[t]`` is the seat's hand at table ``t`` and ``tables[t]`` every
        seat's bid so far there. ``candidates[t]`` lists the bids to compare
        there, with ``valid[t]`` False for padding or illegal entries (each
        row needs one valid entry). ``score[b, k]`` is the points for bidding b
        and taking k. The first layer is shared across a row's candidates,
        which differ in two inputs only.
        """
        w0, b0 = self.bid.weights[0], self.bid.biases[0]
        cache = {}
        hand_rows = np.empty((len(hands), 82), dtype=np.float32)
        for t, hand in enumerate(hands):
            row = cache.get(hand)
            if row is None:
                row = cache[hand] = bid_hand_part(hand, trump_card, n, hand_size)
            hand_rows[t] = row
        context = bid_context(tables, n, hand_size, seat, dealer, self.bid_inputs)
        if self.bid_inputs == BID_BOWER_INPUTS:
            context[:, BID_INPUTS - 82:] = [bid_bower_part(hand, trump_card) for hand in hands]
        shared = hand_rows @ w0[:82] + context @ w0[82:] + b0
        bids = np.arange(MAX_BID_ROWS)
        offsets = w0[86 + bids] + (bids / hand_size)[:, None].astype(np.float32) * w0[97]
        candidates = np.asarray(candidates)
        x = np.maximum(shared[:, None, :] + offsets[candidates], 0.0)
        x = x.reshape(-1, w0.shape[1])
        weights, biases = self.bid.weights[1:], self.bid.biases[1:]
        for w, b in zip(weights[:-1], biases[:-1]):
            x = np.maximum(x @ w + b, 0.0)
        width = hand_size + 1
        logits = (x @ weights[-1] + biases[-1])[:, :width].reshape(*candidates.shape, width)
        logits -= logits.max(axis=2, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=2, keepdims=True)
        values = (probs * score[candidates]).sum(axis=2)
        values[~np.asarray(valid)] = -np.inf
        return candidates[np.arange(len(candidates)), values.argmax(axis=1)]

    def card_values(self, view, seat, hand, cards):
        """Predicted final round points for this seat after each card."""
        output = self.play(play_matrix(view, seat, hand, cards, self.play_inputs))[:, 0]
        return (output * VALUE_SCALE).tolist()


REPO_ROOT = MODEL_DIR.parents[1]


def _folder(path):
    """A model folder as given, or relative to the repository when the working
    directory differs, so specs like ``models=ohhell/models/x`` work anywhere."""
    folder = Path(path)
    if not folder.is_absolute() and not folder.exists() and (REPO_ROOT / folder).exists():
        return REPO_ROOT / folder
    return folder


def model_paths(directory=None, bid_models=None, play_models=None):
    """Resolve each checkpoint independently for one-change comparisons."""
    folder = _folder(directory) if directory else MODEL_DIR
    bid_folder = _folder(bid_models) if bid_models else folder
    play_folder = _folder(play_models) if play_models else folder
    return bid_folder / 'bid.npz', play_folder / 'play.npz'


@lru_cache(maxsize=8)
def load_models(directory=None, bid_models=None, play_models=None):
    bid, play = model_paths(directory, bid_models, play_models)
    return Models(MLP.load(bid), MLP.load(play))
