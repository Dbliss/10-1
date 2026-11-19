"""Explicit adaptations of pinned public Oh Hell-family players.

These are separate entries, never silent fallbacks for unavailable neural
models. All runtime information comes from the ordinary Observation API.
"""

import ctypes
from functools import lru_cache
import importlib.util
import json
from types import SimpleNamespace

import numpy as np

from ..cards import EFFECTIVE_SUIT, OUTRANKS, POWER, SUIT_CARDS, beats, cards_in
from ..engine import BIDDING
from ..external_setup import ASSETS, build_diminishing
from .base import Bot


@lru_cache(maxsize=8)
def campbell_network(n, name):
    path = ASSETS / 'campbell' / f'OhHell/resources/models/N{n}/D1/T0/{name}.txt'
    arrays = [np.asarray(json.loads(block.replace('{', '[').replace('}', ']')), dtype=np.float64)
              for block in path.read_text().strip().split('\n\n')]
    if len(arrays) != 4 or arrays[2].shape != (1, arrays[0].shape[0]):
        raise ValueError(f'Invalid upstream network: {path}')
    for a in arrays:
        a.flags.writeable = False
    return tuple(arrays)


def network_probability(network, rows):
    w0, b0, w1, b1 = network
    x = np.asarray(rows, dtype=np.float64)
    hidden = np.maximum(x @ w0.T + b0, 0)
    z = (hidden @ w1.T + b1)[:, 0]
    return 1 / (1 + np.exp(-np.clip(z, -700, 700)))


def one_hot_inputs(chunks):
    row = []
    for value, width in chunks:
        # Bower trump counts can exceed the Ace-high training layout. Preserve
        # that layout and explicitly saturate the categorical count at its max.
        value = min(width, max(0, value))
        row.extend(float(i + 1 == value) for i in range(width))
    return row


def trick_distribution(probabilities):
    result = np.array([1.0])
    for p in probabilities:
        result = np.convolve(result, [1 - p, p])
    return result


class CampbellBot(Bot):
    """Python port of AiStrategyModuleOI with its original OVL/IVL weights.

    Bids use the table's score function. Play retains the upstream independent
    card probabilities and probability-of-making-the-bid objective. Ties use
    stable legal-card order instead of Java Hashtable iteration order.
    """
    def __init__(self, seed=0):
        super().__init__(seed)
        self.feature_saturations = 0

    @staticmethod
    def _known(obs):
        return obs.played | obs.hand | (1 << obs.trump_card)

    def _ovl_features(self, obs, card, offset=0, void_offset=0):
        t = obs.trump_suit
        suit = EFFECTIVE_SUIT[t][card]
        seen = self._known(obs)
        max_h = min(10, 51 // obs.n)
        chunks = [(obs.hand.bit_count() - offset, max_h)]
        for relative in range(obs.n):
            seat = (obs.seat + relative) % obs.n
            value = (obs.bids[seat] + 1 if obs.phase == BIDDING else
                     max(0, obs.bids[seat] - obs.tricks_won[seat]) + 1)
            chunks.append((value, max_h + 1))
        voids = sum(not obs.hand & SUIT_CARDS[t][s] for s in range(4))
        unseen_trumps = (SUIT_CARDS[t][t] & ~seen).bit_count() + 1
        unseen_suit = (SUIT_CARDS[t][suit] & ~seen).bit_count() + 1
        self.feature_saturations += int(unseen_trumps > 13) + int(unseen_suit > 13)
        chunks.extend([(voids + void_offset + 1, 4), (unseen_trumps, 13),
                       (2 if suit == t else 1, 2), (unseen_suit, 13),
                       (13 - (OUTRANKS[t][card] & ~seen).bit_count(), 13)])
        return one_hot_inputs(chunks)

    def _ivl_features(self, obs, card):
        t = obs.trump_suit
        seen = self._known(obs)
        max_h = min(10, 51 // obs.n)
        chunks = []
        position = (obs.seat - obs.leader) % obs.n
        for relative in range(1, obs.n):
            seat = (obs.seat + relative) % obs.n
            value = max(0, obs.bids[seat] - obs.tricks_won[seat]) + 1
            chunks.append((value if position + relative < obs.n else 0, max_h + 1))
        led = EFFECTIVE_SUIT[t][card] if obs.lead_suit < 0 else obs.lead_suit
        suit = EFFECTIVE_SUIT[t][card]
        chunks.extend([((SUIT_CARDS[t][t] & ~seen).bit_count() + 1, 13),
                       (2 if led == t else 1, 2),
                       ((SUIT_CARDS[t][led] & ~seen).bit_count() + 1, 13),
                       (2 if suit == t else 1, 2),
                       (13 - (OUTRANKS[t][card] & ~seen).bit_count(), 13)])
        return one_hot_inputs(chunks)

    def bid(self, obs):
        cards = cards_in(obs.hand)
        probabilities = network_probability(campbell_network(obs.n, 'ovl'),
                                             [self._ovl_features(obs, c) for c in cards])
        distribution = trick_distribution(probabilities)
        return max(obs.legal_actions(), key=lambda bid: sum(
            p * obs.rules.scoring.points(bid, tricks) for tricks, p in enumerate(distribution)))

    def play(self, obs):
        legal = obs.legal_actions()
        if len(legal) == 1:
            return legal[0]
        cards = cards_in(obs.hand)
        t = obs.trump_suit
        sizes = [(obs.hand & SUIT_CARDS[t][s]).bit_count() for s in range(4)]
        base = network_probability(campbell_network(obs.n, 'ovl'),
                                  [self._ovl_features(obs, c, 1) for c in cards])
        singleton = network_probability(campbell_network(obs.n, 'ovl'),
                                       [self._ovl_features(obs, c, 1, 1) for c in cards])
        immediate = network_probability(campbell_network(obs.n, 'ivl'),
                                        [self._ivl_features(obs, c) for c in legal])
        need = max(0, obs.bids[obs.seat] - obs.tricks_won[obs.seat])
        values = {}
        for card, chance in zip(legal, immediate):
            if obs.best_card >= 0 and not beats(card, obs.best_card, t):
                chance = 0.0
            probabilities = (singleton if sizes[EFFECTIVE_SUIT[t][card]] == 1 else base).copy()
            probabilities[cards.index(card)] = chance
            dist = trick_distribution(probabilities)
            values[card] = dist[need] if need < len(dist) else 0.0
        return max(legal, key=values.get)


@lru_cache(maxsize=1)
def drl_module():
    path = ASSETS / 'drl' / 'agent.py'
    spec = importlib.util.spec_from_file_location('upstream_drl_ohhell_agent', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def drl_order():
    # Upstream card_order[a,b] is 1 when b beats a, including trumping.
    order = np.zeros((4, 52, 52), dtype=int)
    for t in range(4):
        for a in range(52):
            for b in range(52):
                order[t, a, b] = int(a != b and beats(b, a, t))
    return order


class DRLRuleBot(Bot):
    """Execute the pinned upstream RuleBasedPlayer, with bower-aware tables."""
    def __init__(self, seed=0):
        super().__init__(seed)
        self.player = None
        self.legal_bid_adjustments = 0

    def _prepare(self, obs):
        if self.player is None:
            hps = {'env': {'PLAYERS': obs.n, 'TRICKS': 10, 'CARDS': 52, 'SUIT': 4, 'SEED': 0}}
            self.player = drl_module().RuleBasedPlayer(obs.seat, 'DRL upstream rule', hps, None)
        self.player.hand = np.asarray(cards_in(obs.hand))
        self.player.env = SimpleNamespace(card_order=drl_order(), num_round=obs.hand_size)
        return self.player

    def bid(self, obs):
        bid = int(self._prepare(obs).bidding(0, obs.trump_suit, obs.bids))
        legal = obs.legal_actions()
        if bid not in legal:
            self.legal_bid_adjustments += 1
            bid = min(legal, key=lambda b: (abs(b - bid), b))
        return bid

    def play(self, obs):
        player = self._prepare(obs)
        # The upstream caches its rank probabilities during bidding. Rebuild
        # them if called on a standalone saved position (not hidden history).
        if player.winning_probs is None:
            player.bidding(0, obs.trump_suit, obs.bids)
        return int(player.playing(None, obs.trick_size, obs.seat,
            np.asarray(obs.legal_actions()), None, obs.trump_suit, obs.lead_suit,
            obs.bids, obs.tricks_won, obs.history, None if obs.best_card < 0 else obs.best_card))


@lru_cache(maxsize=1)
def diminishing_library():
    library = ctypes.CDLL(str(build_diminishing()))
    ints = ctypes.POINTER(ctypes.c_int)
    library.league_dim_choose.restype = ctypes.c_int
    library.league_dim_choose.argtypes = [ctypes.c_int] * 8 + [ctypes.c_double, ctypes.c_uint] + [ints] * 7
    library.league_dim_winner.restype = ctypes.c_int
    library.league_dim_winner.argtypes = [ctypes.c_int, ctypes.c_int, ints]
    return library


def diminishing_inputs(obs, scores):
    t = obs.trump_suit
    grid = [-2] * 60
    reverse = {}
    known = obs.played | (1 << obs.trump_card)
    for card in range(52):
        code = EFFECTIVE_SUIT[t][card] * 15 + POWER[t][card]
        reverse[code] = card
        grid[code] = obs.seat if obs.hand & (1 << card) else -2 if known & (1 << card) else -1
    trick = [-1] * (obs.n * 2)
    if obs.trick_size:
        for i, (_, card) in enumerate(obs.history[-obs.trick_size:]):
            trick[2 * i:2 * i + 2] = EFFECTIVE_SUIT[t][card], POWER[t][card]
    order = [(obs.leader + i) % obs.n for i in range(obs.n)]
    voids = [int(bool(obs.voids[p] & (1 << s))) for s in range(4) for p in range(obs.n)]
    return (grid, trick, order, list(obs.bids), list(obs.tricks_won), list(scores), voids), reverse


class DiminishingBot(Bot):
    def __init__(self, seed=0, algorithm='mc', time_ms=2, budget=100000):
        super().__init__(seed)
        if algorithm not in ('mc', 'mcts') or budget < 15 or time_ms < 0:
            raise ValueError('Invalid DiminishingBridge settings')
        self.algorithm = algorithm
        self.time_ms = time_ms
        self.budget = budget
        self.seed = seed
        self.calls = 0
        self.scores = None
        self.legal_bid_adjustments = 0

    def _choose(self, obs):
        if self.scores is None:
            self.scores = [0] * obs.n
        arrays, reverse = diminishing_inputs(obs, self.scores)
        arrays = [(ctypes.c_int * len(a))(*a) for a in arrays]
        forbidden = next((b for b in range(obs.hand_size + 1) if b not in obs.legal_actions()), -1) if obs.phase == BIDDING else -1
        action = diminishing_library().league_dim_choose(obs.n, obs.hand_size, obs.seat,
            obs.trump_suit, int(obs.phase == BIDDING), forbidden,
            int(self.algorithm == 'mcts'), self.budget, self.time_ms / 1000,
            (self.seed + self.calls * 1009) & 0xFFFFFFFF, *arrays)
        self.calls += 1
        if action == -999:
            raise RuntimeError(f'Upstream {self.algorithm} could not produce a valid sampled decision')
        if obs.phase == BIDDING:
            legal = obs.legal_actions()
            if action not in legal:
                self.legal_bid_adjustments += 1
                action = min(legal, key=lambda b: (abs(b - action), b))
            return action
        if action not in reverse or reverse[action] not in obs.legal_actions():
            raise RuntimeError(f'Upstream {self.algorithm} returned illegal card code {action}')
        return reverse[action]

    bid = _choose

    def play(self, obs):
        legal = obs.legal_actions()
        return legal[0] if len(legal) == 1 else self._choose(obs)

    def observe_round(self, obs):
        if self.scores is None:
            self.scores = [0] * obs.n
        for s, (bid, won) in enumerate(zip(obs.bids, obs.tricks_won)):
            self.scores[s] += obs.rules.scoring.points(bid, won)
