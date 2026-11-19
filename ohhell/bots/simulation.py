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
from .components import FixedController, TimeBank
from .heuristic import (
    EVERY_SEAT, HeuristicBot, best_bid_for, card_win_probs, trick_distribution,
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


# Exact-search thresholds (cards left) per table size for endgame_table=1:
# the deepest where one sampled deal's exact solve still leaves ~100 deals in
# a 250 ms turn (median), timed on 19 November 2026 under duel load.
ENDGAME_BY_PLAYERS = {3: 18, 4: 19, 5: 21, 6: 22, 7: 24, 8: 24}


class SimulationBot(HeuristicBot):
    """Samples possible deals and scores each option by playing them out.

    ``worlds`` and ``bid_worlds`` set how many deals are sampled. Once at most
    ``endgame_cards`` cards remain, play-outs are replaced by exact search in
    deals where every opponent is the skilled kind. ``field`` is the mix of
    opponents to plan for. ``endgame_noise=1`` searches the other deals
    exactly too, a noisy seat playing its best card with probability
    ``1 - card_noise`` and a random legal card otherwise (``solve_noisy``).
    ``endgame_table=1`` replaces ``endgame_cards`` by ENDGAME_BY_PLAYERS.

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

    ``style`` (the bits in ``heuristic.py``) changes how our own seat is
    assumed to play the rest of a simulated round. Simulated opponents keep
    the plain rules unless ``style_seats`` is ``'all'``, which also judges
    their past plays by the style when inference is on.

    ``dedupe`` plays each distinct deterministic resampled deal once per
    native block (per turn in Python), counting every draw's multiplicity.
    Fixed-budget values are the same; repeated rollout work is avoided.
    """

    def __init__(self, seed=0, worlds=32, bid_worlds=64, endgame_cards=12,
                 inference=False, oversample=4, field='mix', confidence=0.0,
                 fast=True, time_ms=None, react=False, estimate=1, bid_sigma=None,
                 style=0, style_seats='self', dedupe=0, sampler='belief',
                 policy='heuristic', rollout_models=None, policy_inference=0,
                 policy_temperature=10, controller='fixed', bank_capacity=4,
                 stop_z=0, stop_min=32, typing=0, native_required=0, reading='all',
                 card_budget=0, bid_budget=0, endgame_budget=0, endgame_noise=0,
                 endgame_table=0):
        super().__init__(seed, react=react, style=style)
        self.last_card_worlds = 0  # deals the last card evaluation counted
        if style_seats not in ('self', 'all'):
            raise ValueError("style_seats must be 'self' or 'all'.")
        self.style_all = style_seats == 'all'
        if estimate not in (1, 2, 3):
            raise ValueError('estimate must be 1, 2 or 3.')
        self.estimate = estimate
        self.time_limit = time_ms / 1000.0 if time_ms else None
        if sampler not in ('belief', 'uniform') or policy not in ('heuristic', 'learned_self', 'learned_all'):
            raise ValueError('Unknown sampler or rollout policy.')
        if controller not in ('fixed', 'bank') or policy_temperature <= 0 or stop_z < 0 or stop_min < 2:
            raise ValueError('Invalid controller, temperature or stopping settings.')
        if worlds < 1 or bid_worlds < 1 or oversample < 1:
            raise ValueError('worlds, bid_worlds and oversample must be positive.')
        self.controller = (TimeBank(self.time_limit, bank_capacity) if controller == 'bank'
                           else FixedController(self.time_limit))
        self.dedupe = bool(dedupe)
        self.policy_name = policy
        self.rollout_policy = None
        self.opponent_tracker = None
        if typing:
            from ..opponent_reading import OpponentTracker
            self.opponent_tracker = OpponentTracker(field, react, estimate, bid_sigma)
        self.deadline = None
        self.worlds = worlds
        self.bid_worlds = bid_worlds
        self.endgame_cards = endgame_cards
        if endgame_table not in (0, 1):
            raise ValueError('endgame_table must be 0 or 1.')
        self.endgame_table = dict(ENDGAME_BY_PLAYERS) if endgame_table else {}
        self.endgame_noise = bool(endgame_noise)
        inference = bool(inference) and sampler == 'belief'
        self.belief = Belief(inference=inference, oversample=oversample, field=field,
                             react=react, estimate=estimate, bid_sigma=bid_sigma,
                             style=style if self.style_all else 0)
        if reading not in ('all', 'bids'):
            raise ValueError("reading must be 'all' or 'bids'.")
        self.belief.play_reading = reading == 'all'
        self.confidence = confidence
        self.simulator = None
        if fast and not native.available:
            raise RuntimeError('Native simulator is not built for the current core.c. '
                               'Run python -m ohhell.native; use fast=0 only for the Python oracle.')
        if fast and native.available:
            self.simulator = native.Simulator(
                seed, inference, oversample, field, endgame_cards, Scoring(), react, estimate,
                bid_sigma, style, self.style_all, bool(dedupe))
        if native_required and self.simulator is None:
            raise RuntimeError('Native simulator required; build with python -m ohhell.native')
        if stop_z and self.simulator is None:
            raise RuntimeError('Early stopping requires the native simulator.')
        if policy != 'heuristic' or policy_inference:
            from ..learn.rollout import RolloutPolicy, load_network
            self.rollout_policy = RolloutPolicy(load_network(rollout_models))
            if policy_inference:
                self.belief.card_policy = self.rollout_policy
                self.belief.policy_temperature = policy_temperature
            if self.simulator:
                self.simulator.set_rollout(self.rollout_policy.model,
                    {'heuristic': 0, 'learned_self': 1, 'learned_all': 2}[policy],
                    policy_inference, policy_temperature)
        if (card_budget or bid_budget or endgame_budget) and (self.simulator is None or time_ms):
            raise ValueError('Deal budgets need the native simulator and no time_ms.')
        if self.simulator:
            # Fixed deal budgets: untimed search that repeats exactly (see ohhell.duel).
            self.simulator.card_budget = int(card_budget)
            self.simulator.bid_budget = int(bid_budget)
            self.simulator.endgame_budget = int(endgame_budget)
            self.simulator.endgame_table = self.endgame_table
            self.simulator.params.endgame_noise = int(self.endgame_noise)
            self.simulator.params.bid_only = int(reading == 'bids')
            self.simulator.params.stop_z = stop_z
            self.simulator.params.stop_min = stop_min

    def endgame_limit(self, players):
        """Cards left at or below which card play is searched exactly."""
        return self.endgame_table.get(players, self.endgame_cards)

    def act(self, obs):
        started = time.perf_counter()
        choice = len(obs.legal_actions()) > 1
        allowance = self.controller.begin() if choice else self.time_limit
        self.deadline = started + allowance if allowance else None
        self.last_card_worlds = 0
        if self.opponent_tracker:
            weights = self.opponent_tracker.weights(obs.n)
            self.belief.seat_weights = weights
            if self.simulator:
                self.simulator.set_seat_weights(weights)
        try:
            return super().act(obs)
        finally:
            self.deadline = None
            if choice:
                self.controller.finish(time.perf_counter() - started)

    def start_clock(self):
        """Start this turn's time allowance by hand.

        ``act`` does this itself. Callers that ask for an analysis directly
        (``analyse_bids``, ``evaluate_cards``) must call this first, or the
        bot samples one fixed block of deals and ignores ``time_ms``.
        """
        self.deadline = time.perf_counter() + self.time_limit if self.time_limit else None

    def observe_round(self, obs):
        if self.opponent_tracker:
            started = time.perf_counter()
            self.opponent_tracker.observe_round(obs)
            self.controller.charge(time.perf_counter() - started)

    def _policy(self, kinds, seat):
        population = PopulationPolicy(kinds, self.rng, self.react, self.style,
                                      EVERY_SEAT if self.style_all else seat)
        if self.rollout_policy is None or self.policy_name == 'heuristic':
            return population
        def decide(state):
            acting = state.to_act
            if (self.policy_name == 'learned_all' or acting == seat):
                noise = kinds[acting].card_noise
                if not noise or self.rng.random() >= noise:
                    return self.rollout_policy(state)
                from ..cards import cards_in
                from ..engine import legal_card_mask
                return self.rng.choice(cards_in(legal_card_mask(
                    state.hands[acting], state.lead_suit, state.trump_suit)))
            return population(state)
        return decide

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
            policy = self._policy(kinds, obs.seat)
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
            values = self.simulator.evaluate_cards(obs, cards, self.worlds, self._time_left())
            self.last_card_worlds = self.simulator.last_card_worlds
            return values
        remaining = obs.hand.bit_count() * obs.n - obs.trick_size
        exact = remaining <= self.endgame_limit(obs.n)
        totals = dict.fromkeys(cards, 0.0)
        count = 0
        cache = {}
        for hands, kinds in self._sample_worlds(obs, self.worlds):
            count += 1
            key = (tuple(hands), tuple(kinds)) if self.dedupe and not any(k.card_noise for k in kinds) else None
            if key is not None and key in cache:
                for card in cards:
                    totals[card] += cache[key][card]
                continue
            base = obs.determinize(hands)
            noisy = any(kind.card_noise for kind in kinds)
            if exact and (not noisy or self.endgame_noise):
                values = solve_root(base, cards, obs.seat,
                                    [kind.card_noise for kind in kinds] if noisy else None)
                for card in cards:
                    totals[card] += values[card]
                if key is not None:
                    cache[key] = values
                continue
            policy = self._policy(kinds, obs.seat)
            values = {}
            for card in cards:
                state = base.clone()
                state.play_unchecked(card)
                playout(state, policy)
                values[card] = state.points()[obs.seat]
                totals[card] += values[card]
            if key is not None:
                cache[key] = values
        self.last_card_worlds = count
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
