"""Bidding and play by simulating likely deals."""

import math
from dataclasses import dataclass

from ..belief import Belief
from ..engine import NO_BID, PLAYING
from ..search import playout, solve_root
from .base import Bot
from .heuristic import (
    HeuristicBot, best_bid_for, card_win_probs, state_card, trick_distribution,
)


@dataclass
class BidAnalysis:
    """For each bid considered, the trick odds and the expected points."""

    distributions: dict
    values: dict


def heuristic_bid(hand, obs, seat):
    leads = seat == (obs.dealer + 1) % obs.n
    probs = card_win_probs(hand, obs.trump_card, obs.n, obs.hand_size, leads)
    options = range(obs.hand_size + 1)
    return best_bid_for(trick_distribution(probs), options, obs.rules.scoring)


class SimulationBot(HeuristicBot):
    """Samples possible deals and scores each option by playing them out.

    ``worlds`` and ``bid_worlds`` set how many deals are sampled. Once at most
    ``endgame_cards`` cards remain, play-outs are replaced by exact search.
    """

    def __init__(self, seed=0, worlds=32, bid_worlds=64, endgame_cards=12,
                 inference=False, oversample=4):
        super().__init__(seed)
        self.worlds = worlds
        self.bid_worlds = bid_worlds
        self.endgame_cards = endgame_cards
        self.belief = Belief(inference=inference, oversample=oversample)
        self.rollout_policy = state_card

    @staticmethod
    def _respect_dealer_rule(state, obs):
        """Move an imagined dealer's bid if it would break the dealer rule."""
        if (obs.rules.dealer_restriction and obs.seat != obs.dealer
                and obs.bids[obs.dealer] == NO_BID
                and sum(state.bids) == obs.hand_size):
            state.bids[obs.dealer] += 1 if state.bids[obs.dealer] == 0 else -1

    def analyse_bids(self, obs):
        """Estimate P(exactly k tricks) if we bid b, for each plausible b."""
        leads = obs.seat == (obs.dealer + 1) % obs.n
        probs = card_win_probs(obs.hand, obs.trump_card, obs.n, obs.hand_size, leads)
        estimate = sum(probs)
        low = max(0, math.floor(estimate) - 1)
        high = min(obs.hand_size, math.ceil(estimate) + 2)
        legal = obs.legal_actions()
        options = [b for b in legal if low <= b <= high] or legal

        worlds = self.belief.worlds(obs, self.bid_worlds, self.rng)
        counts = {b: [0] * (obs.hand_size + 1) for b in options}
        for hands in worlds:
            base = obs.determinize(hands)
            base.phase = PLAYING
            base.leader = base.to_act = (obs.dealer + 1) % obs.n
            for seat in range(obs.n):
                if obs.bids[seat] == NO_BID and seat != obs.seat:
                    base.bids[seat] = heuristic_bid(hands[seat], obs, seat)
            for bid in options:
                state = base.clone()
                state.bids[obs.seat] = bid
                self._respect_dealer_rule(state, obs)
                playout(state, self.rollout_policy)
                counts[bid][state.tricks_won[obs.seat]] += 1

        total = len(worlds)
        score = obs.rules.scoring.points
        distributions = {b: [c / total for c in tally] for b, tally in counts.items()}
        values = {
            b: sum(p * score(b, k) for k, p in enumerate(dist))
            for b, dist in distributions.items()
        }
        return BidAnalysis(distributions, values)

    def bid(self, obs):
        analysis = self.analyse_bids(obs)
        return max(analysis.values, key=analysis.values.get)

    def evaluate_cards(self, obs, cards=None):
        """Average points this seat ends the round with after each card."""
        cards = obs.legal_actions() if cards is None else cards
        worlds = self.belief.worlds(obs, self.worlds, self.rng)
        remaining = obs.hand.bit_count() * obs.n - obs.trick_size
        exact = remaining <= self.endgame_cards
        totals = dict.fromkeys(cards, 0.0)
        for hands in worlds:
            base = obs.determinize(hands)
            if exact:
                values = solve_root(base, cards, obs.seat)
                for card in cards:
                    totals[card] += values[card]
                continue
            for card in cards:
                state = base.clone()
                state.play_unchecked(card)
                playout(state, self.rollout_policy)
                totals[card] += state.points()[obs.seat]
        return {card: total / len(worlds) for card, total in totals.items()}

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
