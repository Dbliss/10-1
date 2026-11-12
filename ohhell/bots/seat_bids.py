"""Hybrid bidding that imagines later bidders with the bid network.

v2's hybrid search samples deals and fills in each seat still to bid with the
rule-of-thumb bid for its imagined hand. ``SeatBidsBot`` asks the bid network
instead: in every sampled deal and for every bid we might make, each later
seat in turn picks the bid the network expects to score best for its imagined
hand, seeing every bid made before it (ours included). To fit the time
allowance it compares only the bids from one below the rule-of-thumb
estimate's floor to one above its ceiling. The seat's kind then
adds its usual bidding error. Card play, blending and endgames are hybrid's.

``imagine=0`` keeps hybrid's rule-of-thumb later bidders, so the network
inputs and the imagined bidders can be tested apart.
"""

import time

import numpy as np

from .. import native
from ..engine import NO_BID, PLAYING
from ..native import MAX_SEATS
from ..opponents import KINDS, PopulationPolicy, noisy_bid
from ..search import playout
from .learned import HybridBot
from .heuristic import card_win_probs
from .simulation import BidAnalysis

WINDOW = 4  # floor(estimate) - 1 to ceil(estimate) + 1 spans at most four bids


class SeatBidsBot(HybridBot):
    def __init__(self, seed=0, imagine=1, block=8, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.imagine = bool(imagine)
        self.block = block

    @staticmethod
    def later_bidders(obs):
        """Seats still to bid after this one, in bidding order."""
        return [s for s in ((obs.seat + j) % obs.n for j in range(1, obs.n))
                if obs.bids[s] == NO_BID]

    def _analyse_bids(self, obs, options):
        later = self.later_bidders(obs)
        if not self.imagine or not later:
            return super()._analyse_bids(obs, options)
        width = obs.hand_size + 1
        points = obs.rules.scoring.points
        score = np.array([[points(b, k) for k in range(width)] for b in range(width)],
                         dtype=np.float64)
        counts = np.zeros((len(options), width), dtype=np.float64)
        timed = self.deadline is not None
        block = self.block if timed else self.bid_worlds
        total = 0
        while True:
            hands, kinds = self._draw(obs, block)
            bids = self.imagine_bids(obs, hands, kinds, options, later, score)
            self._score(obs, hands, kinds, bids, counts)
            total += len(hands)
            if not timed or time.perf_counter() >= self.deadline:
                break
        self.last_bid_samples = total
        distributions = {b: (counts[i] / total).tolist() for i, b in enumerate(options)}
        values = {b: float(counts[i] @ score[b]) / total for i, b in enumerate(options)}
        return BidAnalysis(distributions, values, total)

    def imagine_bids(self, obs, hands, kinds, options, later, score):
        """Every seat's bid, shape (worlds, options, MAX_SEATS), after each of
        our options, with the later seats bidding in order by the network."""
        n, size = obs.n, obs.hand_size
        worlds = len(hands)
        tables = np.full((worlds, len(options), MAX_SEATS), NO_BID, dtype=np.int32)
        tables[:, :, :n] = obs.bids
        tables[:, :, obs.seat] = options
        flat = tables.reshape(-1, MAX_SEATS)
        restricted = obs.rules.dealer_restriction
        leader = (obs.dealer + 1) % n
        for seat in later:
            low, high = self._windows(hands[:, seat], obs, seat == leader)
            low = np.repeat(low, len(options))
            high = np.repeat(high, len(options))
            candidates = low[:, None] + np.arange(WINDOW)
            valid = (candidates <= high[:, None]) & (candidates <= size)
            if restricted and seat == obs.dealer:
                forbidden = size - (flat[:, :n].sum(axis=1) - flat[:, seat])
                valid &= candidates != forbidden[:, None]
            candidates = np.minimum(candidates, size)
            seat_hands = [int(h) for h in np.repeat(hands[:, seat], len(options))]
            chosen = self.models.choose_bids(
                seat_hands, flat[:, :n], obs.trump_card, n, size, seat, obs.dealer,
                candidates, valid, score)
            row_kinds = np.repeat(kinds[:, seat], len(options))
            for r, kind_index in enumerate(row_kinds):
                kind = KINDS[kind_index]
                if kind.bid_one or kind.bid_two:
                    legal = list(range(size + 1))
                    if restricted and seat == obs.dealer:
                        legal.remove(int(forbidden[r])) if 0 <= forbidden[r] <= size else None
                    chosen[r] = noisy_bid(int(chosen[r]), kind, self.rng, legal)
            flat[:, seat] = chosen
        return tables

    @staticmethod
    def _windows(hands, obs, leads):
        """Per imagined hand, the bids the network chooses among: one below
        the rule-of-thumb trick estimate's floor to one above its ceiling.
        Over random spots this window holds the network's unrestricted choice
        about 97% of the time, at about a third of the cost."""
        if native.available:
            expected = native.library.oh_expected_tricks
            estimates = np.array([expected(int(h), obs.trump_card, obs.n, obs.hand_size,
                                           int(leads)) for h in hands])
        else:
            estimates = np.array([
                sum(card_win_probs(int(h), obs.trump_card, obs.n, obs.hand_size, leads))
                for h in hands])
        low = np.maximum(0, np.floor(estimates).astype(np.int64) - 1)
        high = np.minimum(obs.hand_size, np.ceil(estimates).astype(np.int64) + 1)
        return low, high

    def _draw(self, obs, count):
        if self.simulator:
            return self.simulator.sample_worlds(obs, count)
        hands = np.zeros((count, MAX_SEATS), dtype=np.uint64)
        kinds = np.zeros((count, MAX_SEATS), dtype=np.int32)
        done = 0
        for world_hands, world_kinds in self.belief.worlds(obs, count, self.rng):
            hands[done, :obs.n] = world_hands
            kinds[done, :obs.n] = [KINDS.index(k) for k in world_kinds]
            done += 1
        return hands[:done], kinds[:done]

    def _score(self, obs, hands, kinds, bids, counts):
        if self.simulator:
            self.simulator.score_bids(obs, hands, kinds, bids, counts)
            return
        n = obs.n
        for w in range(len(hands)):
            base = obs.determinize([int(h) for h in hands[w, :n]])
            base.phase = PLAYING
            base.leader = base.to_act = (obs.dealer + 1) % n
            policy = PopulationPolicy([KINDS[k] for k in kinds[w, :n]], self.rng,
                                      self.react)
            for o in range(bids.shape[1]):
                state = base.clone()
                state.bids[:] = [int(b) for b in bids[w, o, :n]]
                playout(state, policy)
                counts[o, state.tricks_won[obs.seat]] += 1
