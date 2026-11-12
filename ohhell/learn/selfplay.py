"""Self-play policy iteration for the bid and play models.

Training in ``train.py`` labels every option by finishing the round with the
rule-based population, so the models learn values under the heuristic's play.
Here the models label their own data instead. Each iteration:

1. Deals rounds where a share of seats bid and play with the current models
   (the rest are the usual skilled, casual and novice population, so the
   models keep meeting human-like players).
2. At sampled decisions of model seats, tries every option from that exact
   position and finishes the round with the deciding seat following the
   current models: the value of each option under the models' own policy.
3. Warm-starts the networks from the current weights and fits them to those
   values, so acting greedily on them is one step of policy improvement.
4. Checks the network-only bot of the new weights against the starting
   weights on identical deals, and saves the weights for that iteration.

    python -m ohhell.learn.selfplay --start ohhell/models --out runs/selfplay/bowers

With ``--labels search`` the loop learns from the search instead. Model seats
are the full hybrid bot (v2's spec with this iteration's weights, search
included) acting from their own observations. At sampled decisions the bot's
search scores every option from that observation, over a fixed number of
sampled deals: the trick odds of each bid and the average points after each
legal card. The networks are fitted to those scores, then the next iteration
searches with the new networks.

    python -m ohhell.learn.selfplay --labels search --start ohhell/models --out runs/selfplay/search

``--bid-inputs 127`` widens a 98-input bid network to the seat-bid layout
(each earlier bid by seat and the number still to bid; see ``features.py``).
The new inputs start with zero weights, so the widened network predicts
exactly what the old one did until training moves them. ``--freeze-play``
keeps the starting play network and labels only bids.

    python -m ohhell.learn.selfplay --start ohhell/models --bid-inputs 127 --freeze-play --out runs/selfplay/seat_bids

``--play-inputs 245`` widens a 241-input play network to the bower layout
(which side-suit slot is trump's colour partner, and whether a card is the
left bower), again with zero weights to start. ``--freeze-bid`` keeps the
starting bid network and labels only cards. ``--spec`` replaces the bot that
plays and labels in search mode (``{folder}`` becomes the iteration's nets),
so the play network can be retrained around a candidate's own bidding:

    python -m ohhell.learn.selfplay --labels search --freeze-bid --play-inputs 245 --spec "hybrid:field=previous,estimate=3,bid_blend=0,time_ms=2,models={folder}" --start ohhell/models --out runs/selfplay/sim_bids

Bid-network options for rollout labels, each meant to be tested alone:
``--bid-inputs 102`` adds the bower inputs (which side-suit slot is trump's
colour partner, and whether the hand holds the left bower). ``--label-worlds
K`` scores each bid over K deals resampled from what the bidder knew, with
later seats rebidding in reply, and fits the averaged trick odds instead of
one finish of the true deal. ``--strong-share`` seats some non-model players
as an untimed hybrid bot (``--strong-spec``), at the table and in finishes.
``--bid-hidden 384,192`` grows the hidden layers, function-preserving.
"""

import argparse
import shutil
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..arena import compare, deal_round
from ..belief import Belief
from ..bots.heuristic import best_bid_for, card_win_probs, state_card, trick_distribution
from ..cards import cards_in
from ..engine import BIDDING, PLAYING, RoundState, legal_bids, legal_card_mask
from ..opponents import FIELDS, KINDS, NOVICE, noisy_bid
from ..rules import Rules
from .features import (
    BID_BOWER_INPUTS, BID_INPUTS, BID_SEAT_INPUTS, MAX_TRICKS, PLAY_BOWER_INPUTS, PLAY_INPUTS, bid_matrix, play_matrix,
)
from .models import VALUE_SCALE, load_models
from .nn import MLP, fit, huber_loss, soft_softmax_loss, softmax_loss
from .train import PLAYER_WEIGHTS

BID_SAMPLE_RATE = 0.3
PLAY_SAMPLE_RATE = 0.15
SEARCH_SPEC = 'hybrid:field=previous,time_ms=2,models={folder}'  # v2 with given nets


class SelfPlayPolicy:
    """Model seats play the play model's best card, strong seats ask their bot,
    and the rest play as their kind."""

    def __init__(self, models, kinds, model_seats, rng, epsilon=0.0, strong=None):
        self.models = models
        self.noise = [kind.card_noise for kind in kinds]
        self.model_seats = model_seats
        self.rng = rng
        self.epsilon = epsilon
        self.strong = strong or {}

    def __call__(self, state):
        seat = state.to_act
        hand = state.hands[seat]
        if self.model_seats[seat]:
            cards = cards_in(legal_card_mask(hand, state.lead_suit, state.trump_suit))
            if len(cards) == 1:
                return cards[0]
            if self.epsilon and self.rng.random() < self.epsilon:
                return self.rng.choice(cards)
            values = self.models.play(play_matrix(state, seat, hand, cards,
                                                  self.models.play_inputs))[:, 0]
            return cards[int(values.argmax())]
        bot = self.strong.get(seat)
        if bot is not None:
            cards = cards_in(legal_card_mask(hand, state.lead_suit, state.trump_suit))
            return cards[0] if len(cards) == 1 else bot.act(state.observation(seat))
        noise = self.noise[seat]
        if noise and self.rng.random() < noise:
            return self.rng.choice(cards_in(
                legal_card_mask(hand, state.lead_suit, state.trump_suit)))
        return state_card(state)


def _clone(state):
    """A copy that keeps the play history when the state has one (strong
    seats read their observation, which needs it)."""
    copy = state.clone()
    if state.history is not None:
        copy.history = list(state.history)
    return copy


def _finish(state, policy):
    while state.phase == PLAYING:
        state.play_unchecked(policy(state))
    return state


def _model_bid(models, state, seat, options):
    dists = models.bid_distributions(
        state.hands[seat], state.trump_card, state.n, state.hand_size, seat,
        state.dealer, state.bids, options)
    score = state.rules.scoring
    return max(options, key=lambda b: sum(p * score.points(b, k)
                                          for k, p in enumerate(dists[b])))


def _heuristic_bid(state, seat, options):
    leads = seat == (state.dealer + 1) % state.n
    probs = card_win_probs(state.hands[seat], state.trump_card, state.n,
                           state.hand_size, leads)
    return best_bid_for(trick_distribution(probs), options, state.rules.scoring)


def _table_bid(state, models, kinds, model_seats, strong, rng):
    """The bid the seat to act makes at this table."""
    seat = state.to_act
    options = state.legal_actions()
    if model_seats[seat]:
        return _model_bid(models, state, seat, options)
    bot = strong.get(seat)
    if bot is not None:
        return bot.act(state.observation(seat))
    return noisy_bid(_heuristic_bid(state, seat, options), kinds[seat], rng, options)


def _bid_label_options(state, seat, earlier):
    """The bids to label: legal bids from the heuristic estimate -2 to +3."""
    legal = legal_bids(state.rules, state.hand_size, earlier)
    leads = seat == (state.dealer + 1) % state.n
    estimate = sum(card_win_probs(state.hands[seat], state.trump_card, state.n,
                                  state.hand_size, leads))
    low = max(0, math.floor(estimate) - 2)
    high = min(state.hand_size, math.ceil(estimate) + 3)
    return [b for b in legal if low <= b <= high] or legal


def _resampled_odds(state, seat, options, worlds, belief, models, kinds, model_seats,
                    strong, rng):
    """For each option, the seat's trick counts averaged over ``worlds`` deals
    drawn from what the seat knows when it bids (its hand and the earlier
    bids, weighted by how well each deal explains them). In each deal the
    seat makes the bid, the later seats bid as their policy would in reply,
    and the round is finished with the table's policies. Every option sees
    the same deals."""
    obs = state.observation(seat)
    order = [(state.dealer + 1 + j) % state.n for j in range(state.n)]
    earlier = order[:order.index(seat)]
    counts = {bid: np.zeros(MAX_TRICKS + 1, dtype=np.float32) for bid in options}
    for hands, _ in belief.worlds(obs, worlds, rng):
        base = RoundState(state.rules, hands, state.trump_card, state.dealer)
        if not strong:
            base.history = None
        for other in earlier:
            base.apply(state.bids[other])
        policy = SelfPlayPolicy(models, kinds, model_seats, rng, strong=strong)
        for bid in options:
            trial = _clone(base)
            trial.apply(bid)
            while trial.phase == BIDDING:
                trial.apply(_table_bid(trial, models, kinds, model_seats, strong, rng))
            _finish(trial, policy)
            counts[bid][trial.tricks_won[seat]] += 1.0
    return {bid: c / worlds for bid, c in counts.items()}


def _strong_bots(spec, count, seed):
    from ..bots import make_bot
    return [make_bot(spec, seed * 8 + i) for i in range(count)]


def _generate(job):
    """Play ``rounds`` rounds with the given models and return training arrays.

    ``options`` may set ``label_worlds`` (above 0: bid labels are trick odds
    over that many resampled deals, see ``_resampled_odds``; 0: one finish of
    the true deal), ``strong_share`` (the chance each non-model seat is the
    ``strong_spec`` bot rather than a population kind) and ``strong_spec``.
    """
    seed, rounds, folder, model_share, epsilon, play_rate, bid_rate, *rest = job
    options = rest[0] if rest else {}
    label_worlds = options.get('label_worlds', 0)
    strong_share = options.get('strong_share', 0.0)
    models = load_models(folder)
    inputs = models.bid_inputs
    play_inputs = models.play_inputs
    rng = random.Random(seed)
    players = list(PLAYER_WEIGHTS)
    weights = list(PLAYER_WEIGHTS.values())
    bots = _strong_bots(options['strong_spec'], 8, seed) if strong_share else []
    belief = Belief(inference=True, field='mix') if label_worlds else None
    bid_x, bid_k, bid_limit, play_x, play_y = [], [], [], [], []

    for _ in range(rounds):
        n = rng.choices(players, weights)[0]
        top = min(MAX_TRICKS, 51 // n)
        hand_size = rng.randint(1, top)
        hands, trump = deal_round(rng, n, hand_size)
        state = RoundState(Rules(n, top), hands, trump, rng.randrange(n))
        kinds = rng.choices(KINDS, FIELDS['mix'], k=n)
        model_seats = [rng.random() < model_share for _ in range(n)]
        if not any(model_seats):
            model_seats[rng.randrange(n)] = True
        strong = {}
        if strong_share:
            strong = {seat: bots[seat] for seat in range(n)
                      if not model_seats[seat] and rng.random() < strong_share}
        if not strong and not label_worlds:
            state.history = None

        earlier = {}
        while state.phase == BIDDING:
            seat = state.to_act
            earlier[seat] = list(state.bids)
            if label_worlds and model_seats[seat] and rng.random() < bid_rate:
                labelled = _bid_label_options(state, seat, earlier[seat])
                odds = _resampled_odds(state, seat, labelled, label_worlds, belief, models,
                                       kinds, model_seats, strong, rng)
                bid_x.append(bid_matrix(state.hands[seat], trump, n, hand_size, seat,
                                        state.dealer, earlier[seat], labelled, inputs))
                bid_k.extend(odds[bid] for bid in labelled)
                bid_limit.extend([hand_size] * len(labelled))
            state.apply(_table_bid(state, models, kinds, model_seats, strong, rng))

        # Bid labels (true-deal mode): the seat's tricks after each plausible
        # bid, finishing this deal once under the table's policies.
        for seat in range(0 if label_worlds else n):
            if not model_seats[seat] or rng.random() >= bid_rate:
                continue
            labelled = _bid_label_options(state, seat, earlier[seat])
            bid_x.append(bid_matrix(state.hands[seat], trump, n, hand_size, seat,
                                    state.dealer, earlier[seat], labelled, inputs))
            policy = SelfPlayPolicy(models, kinds, model_seats, rng, strong=strong)
            for bid in labelled:
                trial = _clone(state)
                trial.bids[seat] = bid
                if seat != state.dealer and sum(trial.bids) == hand_size:
                    trial.bids[state.dealer] += 1 if trial.bids[state.dealer] == 0 else -1
                _finish(trial, policy)
                bid_k.append(trial.tricks_won[seat])
                bid_limit.append(hand_size)

        # Play labels: the seat's points after each legal card, under the models.
        table = SelfPlayPolicy(models, kinds, model_seats, rng, epsilon, strong)
        rollout = SelfPlayPolicy(models, kinds, model_seats, rng, strong=strong)
        while state.phase == PLAYING:
            seat = state.to_act
            cards = state.legal_actions()
            if (len(cards) > 1 and model_seats[seat]
                    and rng.random() < play_rate):
                play_x.append(play_matrix(state, seat, state.hands[seat], cards, play_inputs))
                for card in cards:
                    trial = _clone(state)
                    trial.play_unchecked(card)
                    _finish(trial, rollout)
                    play_y.append(trial.points()[seat] / VALUE_SCALE)
            state.play_unchecked(table(state))

    if label_worlds:
        labels = np.array(bid_k, dtype=np.float32).reshape(-1, MAX_TRICKS + 1)
    else:
        labels = np.array(bid_k, dtype=np.int64)
    return (np.concatenate(bid_x) if bid_x else np.zeros((0, inputs), np.float32),
            labels, np.array(bid_limit, dtype=np.int64),
            np.concatenate(play_x) if play_x else np.zeros((0, play_inputs), np.float32),
            np.array(play_y, dtype=np.float32))


def _search_bid_options(obs):
    """The bids to label: the heuristic estimate -2 to +3, as in rollout mode."""
    leads = obs.seat == (obs.dealer + 1) % obs.n
    estimate = sum(card_win_probs(obs.hand, obs.trump_card, obs.n, obs.hand_size, leads))
    low = max(0, math.floor(estimate) - 2)
    high = min(obs.hand_size, math.ceil(estimate) + 3)
    legal = obs.legal_actions()
    return [b for b in legal if low <= b <= high] or legal


def _generate_search(job):
    """Play rounds with the full hybrid bot and label options with its search."""
    (seed, rounds, folder, model_share, epsilon, bid_worlds, card_worlds, play_rate,
     bid_rate, spec) = job
    from ..bots import make_bot
    spec = spec.format(folder=Path(folder).as_posix())
    bots = [make_bot(spec, seed * 8 + i) for i in range(8)]
    inputs = bots[0].models.bid_inputs
    play_inputs = bots[0].models.play_inputs
    if bots[0].simulator is None:
        raise RuntimeError('search labels need the native core: python -m ohhell.native')
    rng = random.Random(seed)
    players = list(PLAYER_WEIGHTS)
    weights = list(PLAYER_WEIGHTS.values())
    bid_x, bid_p, bid_limit, play_x, play_y = [], [], [], [], []

    for _ in range(rounds):
        n = rng.choices(players, weights)[0]
        top = min(MAX_TRICKS, 51 // n)
        hand_size = rng.randint(1, top)
        hands, trump = deal_round(rng, n, hand_size)
        state = RoundState(Rules(n, top), hands, trump, rng.randrange(n))
        kinds = rng.choices(KINDS, FIELDS['mix'], k=n)
        model_seats = [rng.random() < model_share for _ in range(n)]
        if not any(model_seats):
            model_seats[rng.randrange(n)] = True

        while state.phase == BIDDING:
            seat = state.to_act
            options = state.legal_actions()
            if not model_seats[seat]:
                state.apply(noisy_bid(_heuristic_bid(state, seat, options), kinds[seat],
                                      rng, options))
                continue
            obs = state.observation(seat)
            bot = bots[seat]
            if rng.random() < bid_rate:
                labelled = _search_bid_options(obs)
                odds = bot.simulator.analyse_bids(obs, labelled, bid_worlds)
                bid_x.append(bid_matrix(obs.hand, trump, n, hand_size, seat, obs.dealer,
                                        list(obs.bids), labelled, inputs))
                for bid in labelled:
                    row = np.zeros(MAX_TRICKS + 1, dtype=np.float32)
                    row[:hand_size + 1] = odds[bid]
                    bid_p.append(row)
                    bid_limit.append(hand_size)
            state.apply(bot.act(obs))

        noise = [kind.card_noise for kind in kinds]
        while state.phase == PLAYING:
            seat = state.to_act
            if not model_seats[seat]:
                if noise[seat] and rng.random() < noise[seat]:
                    state.play_unchecked(rng.choice(state.legal_actions()))
                else:
                    state.play_unchecked(state_card(state))
                continue
            cards = state.legal_actions()
            if len(cards) == 1:
                state.play_unchecked(cards[0])
                continue
            obs = state.observation(seat)
            bot = bots[seat]
            if rng.random() < play_rate:
                values = bot.simulator.evaluate_cards(obs, cards, card_worlds)
                play_x.append(play_matrix(obs, seat, obs.hand, cards, play_inputs))
                play_y.extend(values[c] / VALUE_SCALE for c in cards)
            if epsilon and rng.random() < epsilon:
                state.play_unchecked(rng.choice(cards))
            else:
                state.play_unchecked(bot.act(obs))

    return (np.concatenate(bid_x) if bid_x else np.zeros((0, inputs), np.float32),
            np.array(bid_p, dtype=np.float32).reshape(-1, MAX_TRICKS + 1),
            np.array(bid_limit, dtype=np.int64),
            np.concatenate(play_x) if play_x else np.zeros((0, play_inputs), np.float32),
            np.array(play_y, dtype=np.float32))


def widen_bid(model, inputs):
    """Add zero-weight inputs at the end so the network's outputs are unchanged."""
    w0 = model.weights[0]
    if w0.shape[0] > inputs:
        raise ValueError(f'bid network already has {w0.shape[0]} inputs')
    extra = np.zeros((inputs - w0.shape[0], w0.shape[1]), dtype=w0.dtype)
    model.weights[0] = np.concatenate([w0, extra])
    return model


def widen_play(model, inputs):
    """The same for the play network (241 inputs to the 245-input bower layout)."""
    return widen_bid(model, inputs)


def widen_hidden(model, hidden, seed=0, scale=0.1):
    """Grow the hidden layers to ``hidden`` units without changing the outputs.

    New units get small random incoming weights and zero outgoing weights,
    so the network predicts exactly what it did until training moves them.
    """
    old = [w.shape[1] for w in model.weights[:-1]]
    if len(hidden) != len(old) or any(h < o for h, o in zip(hidden, old)):
        raise ValueError(f'cannot grow hidden layers {old} to {list(hidden)}')
    rng = np.random.default_rng(seed)
    sizes = [model.weights[0].shape[0], *hidden, model.weights[-1].shape[1]]
    weights, biases = [], []
    for i, (w, b) in enumerate(zip(model.weights, model.biases)):
        rows, cols = sizes[i], sizes[i + 1]
        grown = np.zeros((rows, cols), dtype=w.dtype)
        grown[:w.shape[0], :w.shape[1]] = w
        if cols > w.shape[1]:  # incoming weights of the new units
            grown[:, w.shape[1]:] = (rng.standard_normal((rows, cols - w.shape[1]))
                                     * scale * np.sqrt(2.0 / rows))
        # Rows from new units below stay zero in the old columns, so old units
        # (and the output) are unchanged.
        weights.append(grown)
        bias = np.zeros(cols, dtype=b.dtype)
        bias[:len(b)] = b
        biases.append(bias)
    model.weights, model.biases = weights, biases
    return model


def generate(rounds, seed, workers, folder, model_share, epsilon, labels='rollout',
             bid_worlds=256, card_worlds=128, play_rate=PLAY_SAMPLE_RATE,
             bid_rate=BID_SAMPLE_RATE, spec=SEARCH_SPEC, options=None):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    chunks = workers * 4
    if labels == 'search':
        work = _generate_search
        jobs = [(seed * 100003 + i, rounds // chunks, str(folder), model_share, epsilon,
                 bid_worlds, card_worlds, play_rate, bid_rate, spec) for i in range(chunks)]
    else:
        work = _generate
        jobs = [(seed * 100003 + i, rounds // chunks, str(folder), model_share, epsilon,
                 play_rate, bid_rate, options or {}) for i in range(chunks)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(work, jobs))
    return [np.concatenate([p[i] for p in parts]) for i in range(5)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--start', default='ohhell/models',
                        help='folder holding the bid.npz and play.npz to improve')
    parser.add_argument('--out', default='runs/selfplay/bowers')
    parser.add_argument('--iterations', type=int, default=4)
    parser.add_argument('--rounds', type=int, default=150000, help='rounds per iteration')
    parser.add_argument('--epochs', type=int, default=4, help='epochs per iteration')
    parser.add_argument('--model-share', type=float, default=0.6,
                        help='chance each seat is played by the current models')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='chance a model seat plays a random card off the sampled path')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--eval-games', type=int, default=300)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--labels', choices=('rollout', 'search'), default='rollout',
                        help='rollout: finish the true deal with the models; '
                             'search: the full hybrid bot\'s search scores')
    parser.add_argument('--bid-worlds', type=int, default=256,
                        help='search labels: sampled deals per bid label')
    parser.add_argument('--card-worlds', type=int, default=128,
                        help='search labels: sampled deals per card label')
    parser.add_argument('--bid-inputs', type=int,
                        choices=(BID_INPUTS, BID_BOWER_INPUTS, BID_SEAT_INPUTS),
                        default=None, help=f'widen the starting bid network to '
                        f'{BID_BOWER_INPUTS} inputs (the bower layout: partner suit slot, '
                        f'left bower held) or {BID_SEAT_INPUTS} (earlier bids by seat, '
                        f'seats still to bid)')
    parser.add_argument('--bid-hidden', default=None,
                        help="grow the starting bid network's hidden layers, e.g. 384,192 "
                             "(new units start with zero outgoing weights)")
    parser.add_argument('--label-worlds', type=int, default=0,
                        help='rollout labels: score each bid over this many deals resampled '
                             'from what the bidder knows, with later seats rebidding '
                             '(0: one finish of the true deal)')
    parser.add_argument('--strong-share', type=float, default=0.0,
                        help='rollout labels: chance each non-model seat is --strong-spec '
                             'instead of a population kind, at the table and in finishes')
    parser.add_argument('--strong-spec',
                        default='hybrid:field=previous,'
                                'worlds=8,bid_worlds=32',
                        help='the strong opponent (untimed, so its strength does not '
                             'depend on machine load)')
    parser.add_argument('--freeze-play', action='store_true',
                        help='keep the starting play network; label and fit bids only')
    parser.add_argument('--freeze-bid', action='store_true',
                        help='keep the starting bid network; label and fit cards only')
    parser.add_argument('--play-inputs', type=int, choices=(PLAY_INPUTS, PLAY_BOWER_INPUTS),
                        default=None, help=f'widen the starting play network to '
                        f'{PLAY_BOWER_INPUTS} inputs (the bower layout)')
    parser.add_argument('--spec', default=SEARCH_SPEC,
                        help="search labels: the bot that plays and labels, with {folder} "
                             "for the iteration's networks")
    args = parser.parse_args()
    search = args.labels == 'search'
    bid_loss = soft_softmax_loss if search or args.label_worlds else softmax_loss
    if search and (args.label_worlds or args.strong_share):
        parser.error('--label-worlds and --strong-share apply to rollout labels only')
    options = {'label_worlds': args.label_worlds, 'strong_share': args.strong_share,
               'strong_spec': args.strong_spec}
    bid_hidden = [int(h) for h in args.bid_hidden.split(',')] if args.bid_hidden else None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.freeze_bid and args.freeze_play:
        parser.error('--freeze-bid and --freeze-play leave nothing to train')
    start = Path(args.start)
    widen_bid_to = (args.bid_inputs if args.bid_inputs and
                    MLP.load(start / 'bid.npz').weights[0].shape[0] != args.bid_inputs else None)
    widen_play_to = (args.play_inputs if args.play_inputs and
                     MLP.load(start / 'play.npz').weights[0].shape[0] != args.play_inputs else None)
    if bid_hidden and [w.shape[1] for w in MLP.load(start / 'bid.npz').weights[:-1]] == bid_hidden:
        bid_hidden = None
    if widen_bid_to or widen_play_to or bid_hidden:
        widened = out / 'start'
        widened.mkdir(parents=True, exist_ok=True)
        if widen_bid_to or bid_hidden:
            bid = MLP.load(start / 'bid.npz')
            if widen_bid_to:
                widen_bid(bid, widen_bid_to)
                print(f"widened {start / 'bid.npz'} to {widen_bid_to} inputs", flush=True)
            if bid_hidden:
                widen_hidden(bid, bid_hidden, seed=args.seed)
                print(f"grew {start / 'bid.npz'} hidden layers to {bid_hidden}", flush=True)
            bid.save(widened / 'bid.npz')
        else:
            shutil.copyfile(start / 'bid.npz', widened / 'bid.npz')
        if widen_play_to:
            widen_play(MLP.load(start / 'play.npz'), widen_play_to).save(widened / 'play.npz')
            print(f"widened {start / 'play.npz'} to {widen_play_to} inputs", flush=True)
        else:
            shutil.copyfile(start / 'play.npz', widened / 'play.npz')
        start = widened
    play_rate = 0.0 if args.freeze_play else PLAY_SAMPLE_RATE
    bid_rate = 0.0 if args.freeze_bid else BID_SAMPLE_RATE
    current = start
    history = []
    began = time.time()
    for iteration in range(1, args.iterations + 1):
        tick = time.time()
        seed = args.seed * 1000 + iteration
        bid_x, bid_k, bid_limit, play_x, play_y = generate(
            args.rounds, seed, args.workers, current, args.model_share, args.epsilon,
            args.labels, args.bid_worlds, args.card_worlds, play_rate, bid_rate, args.spec,
            options)
        print(f"iteration {iteration}: {len(bid_x):,} bid and {len(play_x):,} play "
              f"samples in {time.time() - tick:.0f}s", flush=True)

        log = lambda text: print(text, flush=True)
        bid = MLP.load(current / 'bid.npz')
        play = MLP.load(current / 'play.npz')
        if args.freeze_bid:
            print("bid model kept from the start", flush=True)
        else:
            print("bid model", flush=True)
            fit(bid, bid_x, bid_k, bid_loss(bid_limit), epochs=args.epochs, seed=seed, log=log)
        if args.freeze_play:
            print("play model kept from the start", flush=True)
        else:
            print("play model", flush=True)
            fit(play, play_x, play_y, huber_loss(), epochs=args.epochs, seed=seed, log=log)
        del bid_x, bid_k, bid_limit, play_x, play_y

        folder = out / f'iter{iteration}'
        folder.mkdir(parents=True, exist_ok=True)
        bid.save(folder / 'bid.npz')
        play.save(folder / 'play.npz')
        current = folder

        if search:
            candidate = args.spec.format(folder=folder.as_posix())
            baseline = args.spec.format(folder=start.as_posix())
        else:
            candidate, baseline = f'net:models={folder.as_posix()}', f'net:models={start.as_posix()}'
        report = compare(candidate, baseline, Rules(4, 10), args.eval_games, args.workers,
                         first_seed=900000 + iteration * 10000)
        print(f"iteration {iteration} check: {report}", flush=True)
        history.append({
            'iteration': iteration, 'folder': folder.as_posix(), 'seed': seed,
            'check': f'{candidate} vs {baseline}',
            'edge_vs_start': report.edge, 'interval': report.interval,
            'games': report.games, 'seconds': round(time.time() - tick),
        })
        (out / 'history.json').write_text(json.dumps({
            'start': start.as_posix(), 'settings': vars(args), 'iterations': history,
        }, indent=2))
    print(f"done after {time.time() - began:.0f}s", flush=True)


if __name__ == '__main__':
    main()
