"""Bidding and play by simulating likely deals."""

import math
import time
from dataclasses import dataclass

from .. import estimate as estimate_v2
from .. import native
from ..belief import Belief
from ..engine import NO_BID, PLAYING
from ..opponents import PopulationPolicy, noisy_bid
from ..rules import Scoring
from ..search import playout, solve_root
from .base import Bot
from .heuristic import (
    HeuristicBot, best_bid_for, card_win_probs, trick_distribution,
)


@dataclass
class BidAnalysis:
    """For each bid considered, the trick odds and the expected points."""

    distributions: dict
    values: dict
    samples: int = 0


def heuristic_bid(hand, obs, seat, estimate=1):
    options = range(obs.hand_size + 1)
    if estimate >= 2:
        position = estimate_v2.first_trick_position(seat, obs.dealer, obs.n)
        dist = estimate_v2.bid_distribution(
            hand, obs.trump_card, obs.n, obs.hand_size, position, version=estimate)
        return best_bid_for(dist, options, obs.rules.scoring)
    leads = seat == (obs.dealer + 1) % obs.n
    probs = card_win_probs(hand, obs.trump_card, obs.n, obs.hand_size, leads)
    return best_bid_for(trick_distribution(probs), options, obs.rules.scoring)


def shift_tricks(dist, delta):
    """Move a trick distribution up or down by ``delta`` tricks.

    Fractions split each probability between two neighbouring counts, and
    anything pushed past either end stays at that end.
    """
    whole = math.floor(delta)
    part = delta - whole
    last = len(dist) - 1
    moved = [0.0] * len(dist)
    for k, p in enumerate(dist):
        moved[max(0, min(last, k + whole))] += p * (1.0 - part)
        moved[max(0, min(last, k + whole + 1))] += p * part
    return moved


class SimulationBot(HeuristicBot):
    """Samples possible deals and scores each option by playing them out.

    ``worlds`` and ``bid_worlds`` set how many deals are sampled. Once at most
    ``endgame_cards`` cards remain, play-outs are replaced by exact search in
    deals where every opponent is the skilled kind. ``field`` is the mix of
    opponents to plan for.

    ``confidence`` makes the bot misjudge its own hand when bidding, in tricks:
    positive believes it will take more than it will, negative fewer.

    ``time_ms`` caps each turn: sampling continues in blocks of ``worlds`` or
    ``bid_worlds`` until the time is up. ``react`` makes the simulated players
    respond to the other seats' bids.

    ``estimate`` picks the quick trick estimate used to shortlist our bids,
    imagine the bids of seats still to bid, and judge (with inference) how
    well a sampled deal explains the bids already made: 1 is the original
    rule of thumb, 2 the calibrated estimate in ``ohhell/estimate.py``, 3 its
    refit with discard features and a fitted spread.

    ``bid_sigma`` replaces every opponent kind's ``bid_sigma``: how far, in
    tricks, inference expects a bid to stray from the quick estimate.
    """

    def __init__(self, seed=0, worlds=32, bid_worlds=64, endgame_cards=12,
                 inference=False, oversample=4, field='mix', confidence=0.0,
                 fast=True, time_ms=None, react=False, estimate=1, bid_sigma=None):
        super().__init__(seed, react=react)
        if estimate not in (1, 2, 3):
            raise ValueError('estimate must be 1, 2 or 3.')
        self.estimate = estimate
        self.time_limit = time_ms / 1000.0 if time_ms else None
        self.deadline = None
        self.worlds = worlds
        self.bid_worlds = bid_worlds
        self.endgame_cards = endgame_cards
        self.belief = Belief(inference=inference, oversample=oversample, field=field,
                             react=react, estimate=estimate, bid_sigma=bid_sigma)
        self.confidence = confidence
        self.simulator = None
        if fast and native.available:
            self.simulator = native.Simulator(
                seed, inference, oversample, field, endgame_cards, Scoring(), react, estimate,
                bid_sigma)

    def act(self, obs):
        self.deadline = time.perf_counter() + self.time_limit if self.time_limit else None
        try:
            return super().act(obs)
        finally:
            self.deadline = None

    def _time_left(self):
        """Seconds left this turn, or 0.0 when the turn has no limit."""
        if self.deadline is None:
            return 0.0
        return max(1e-6, self.deadline - time.perf_counter())

    def _sample_worlds(self, obs, block):
        """Yield worlds in blocks, stopping when this turn's time runs out."""
        while True:
            for world in self.belief.worlds(obs, block, self.rng):
                yield world
                if self.deadline is not None and time.perf_counter() >= self.deadline:
                    return
            if self.deadline is None:
                return

    @staticmethod
    def _respect_dealer_rule(state, obs):
        """Move an imagined dealer's bid if it would break the dealer rule."""
        if (obs.rules.dealer_restriction and obs.seat != obs.dealer
                and obs.bids[obs.dealer] == NO_BID
                and sum(state.bids) == obs.hand_size):
            state.bids[obs.dealer] += 1 if state.bids[obs.dealer] == 0 else -1

    def analyse_bids(self, obs):
        """Estimate P(exactly k tricks) if we bid b, for each plausible b."""
        return self._analyse_bids(obs, self._bid_options(obs))

    def _bid_options(self, obs):
        """The current bot's heuristic shortlist, in legal bid order."""
        legal = obs.legal_actions()
        if self.estimate >= 2:
            position = estimate_v2.first_trick_position(obs.seat, obs.dealer, obs.n)
            estimate = estimate_v2.expected_tricks(
                obs.hand, obs.trump_card, obs.n, obs.hand_size, position,
                version=self.estimate)
            low = max(estimate_v2.certain_tricks(obs.hand, obs.trump_card),
                      math.floor(estimate) - 1)
            high = min(obs.hand_size, math.ceil(estimate) + 2)
            return [b for b in legal if low <= b <= high] or legal
        leads = obs.seat == (obs.dealer + 1) % obs.n
        probs = card_win_probs(obs.hand, obs.trump_card, obs.n, obs.hand_size, leads)
        estimate = sum(probs)
        low = max(0, math.floor(estimate) - 1)
        high = min(obs.hand_size, math.ceil(estimate) + 2)
        return [b for b in legal if low <= b <= high] or legal

    def _analyse_bids(self, obs, options):
        if self.simulator:
            distributions = self.simulator.analyse_bids(
                obs, options, self.bid_worlds, self._time_left())
            samples = self.simulator.last_bid_samples
        else:
            distributions = self._simulate_bids(obs, options)
            samples = self.last_bid_samples
        score = obs.rules.scoring.points
        values = {
            b: sum(p * score(b, k) for k, p in enumerate(dist))
            for b, dist in distributions.items()
        }
        return BidAnalysis(distributions, values, samples)

    def _simulate_bids(self, obs, options):
        counts = {b: [0] * (obs.hand_size + 1) for b in options}
        total = 0
        for hands, kinds in self._sample_worlds(obs, self.bid_worlds):
            total += 1
            base = obs.determinize(hands)
            base.phase = PLAYING
            base.leader = base.to_act = (obs.dealer + 1) % obs.n
            for seat in range(obs.n):
                if obs.bids[seat] == NO_BID and seat != obs.seat:
                    base.bids[seat] = noisy_bid(
                        heuristic_bid(hands[seat], obs, seat, self.estimate), kinds[seat], self.rng,
                        range(obs.hand_size + 1))
            policy = PopulationPolicy(kinds, self.rng, self.react)
            for bid in options:
                state = base.clone()
                state.bids[obs.seat] = bid
                self._respect_dealer_rule(state, obs)
                playout(state, policy)
                counts[bid][state.tricks_won[obs.seat]] += 1

        self.last_bid_samples = total
        return {b: [c / total for c in tally] for b, tally in counts.items()}

    def bid(self, obs):
        analysis = self.analyse_bids(obs)
        if not self.confidence:
            return max(analysis.values, key=analysis.values.get)
        score = obs.rules.scoring.points

        def believed_value(bid):
            shifted = shift_tricks(analysis.distributions[bid], self.confidence)
            return sum(p * score(bid, k) for k, p in enumerate(shifted))
        return max(analysis.distributions, key=believed_value)

    def evaluate_cards(self, obs, cards=None):
        """Average points this seat ends the round with after each card."""
        cards = obs.legal_actions() if cards is None else cards
        if self.simulator:
            return self.simulator.evaluate_cards(obs, cards, self.worlds, self._time_left())
        remaining = obs.hand.bit_count() * obs.n - obs.trick_size
        exact = remaining <= self.endgame_cards
        totals = dict.fromkeys(cards, 0.0)
        count = 0
        for hands, kinds in self._sample_worlds(obs, self.worlds):
            count += 1
            base = obs.determinize(hands)
            if exact and not any(kind.card_noise for kind in kinds):
                values = solve_root(base, cards, obs.seat)
                for card in cards:
                    totals[card] += values[card]
                continue
            policy = PopulationPolicy(kinds, self.rng, self.react)
            for card in cards:
                state = base.clone()
                state.play_unchecked(card)
                playout(state, policy)
                totals[card] += state.points()[obs.seat]
        return {card: total / count for card, total in totals.items()}

    def play(self, obs):
        cards = obs.legal_actions()
        if len(cards) == 1:
            return cards[0]
        values = self.evaluate_cards(obs, cards)
        default = super().play(obs)
        best = max(values.values())
        if values[default] >= best - 1e-9:
            return default
        return max(values, key=values.get)
