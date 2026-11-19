"""Persistent per-seat kind posteriors learned only from completed public rounds."""

import math

from .belief import Belief
from .bots.heuristic import expected_tricks, state_card
from .engine import DONE, PLAYING, RoundState
from .opponents import FIELDS, KINDS


class OpponentTracker:
    def __init__(self, field='mix', react=False, estimate=1, bid_sigma=None):
        # A small floor lets an initially unlikely kind recover from the prior.
        self.prior = [(w + 0.025) / 1.1 for w in FIELDS[field]]
        self.logs = []
        self.react = react
        self.belief = Belief(estimate=estimate, bid_sigma=bid_sigma)
        self.rounds = 0

    def weights(self, n):
        if not self.logs:
            self.logs = [[math.log(p) for p in self.prior] for _ in range(n)]
        if len(self.logs) != n:
            raise ValueError('Opponent tracker cannot change table size within a game')
        result = []
        for row in self.logs:
            top = max(row)
            values = [math.exp(v - top) for v in row]
            total = sum(values)
            result.append([v / total for v in values])
        return result

    def observe_round(self, obs):
        if obs.phase != DONE or len(obs.history) != obs.n * obs.hand_size:
            raise ValueError('Typing requires a completed, fully public round')
        if obs.blind:
            return  # blind bids were made without seeing these hands
        self.weights(obs.n)
        opening = [0] * obs.n
        for seat, card in obs.history:
            opening[seat] |= 1 << card
        replay = RoundState(obs.rules, opening, obs.trump_card, obs.dealer)
        replay.bids = list(obs.bids)
        replay.bids_made = obs.n
        replay.phase = PLAYING
        likelihood = [[0.0] * len(KINDS) for _ in range(obs.n)]
        for seat in range(obs.n):
            if seat == obs.seat:
                continue
            if self.belief.estimate >= 2:
                from .estimate import expected_tricks as calibrated, first_trick_position
                estimate = calibrated(opening[seat], obs.trump_card, obs.n, obs.hand_size,
                    first_trick_position(seat, obs.dealer, obs.n), version=self.belief.estimate)
            else:
                estimate = expected_tricks(opening[seat], obs.trump_card, obs.n,
                                           obs.hand_size, seat == (obs.dealer + 1) % obs.n)
            for k, kind in enumerate(KINDS):
                sigma = self.belief.bid_sigma or kind.bid_sigma
                likelihood[seat][k] -= (obs.bids[seat] - estimate) ** 2 / (2 * sigma ** 2) + math.log(sigma)
        for seat, card in obs.history:
            options = replay.legal_actions()
            if seat != obs.seat and len(options) > 1:
                predicted = state_card(replay, self.react)
                for k, kind in enumerate(KINDS):
                    chance = kind.play_doubt / len(options)
                    if card == predicted:
                        chance += 1 - kind.play_doubt
                    likelihood[seat][k] += math.log(chance)
            replay.play_unchecked(card)
        for seat in range(obs.n):
            if seat == obs.seat:
                continue
            row = [old + new for old, new in zip(self.logs[seat], likelihood[seat])]
            top = max(row)
            # Bound odds to avoid permanent certainty under a misspecified model.
            self.logs[seat] = [max(-12, v - top) for v in row]
        self.rounds += 1
