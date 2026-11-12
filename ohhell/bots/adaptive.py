"""Search every legal card, then spend the remaining budget on contenders."""

import math
import time

from ..opponents import PopulationPolicy
from ..search import playout
from .learned import HybridBot


class CardRace:
    """Successive halving with a model prior measured in sampled deals.

    A stage compares its active cards on the same sampled worlds. Totals and
    actual sample counts survive elimination stages, so the prior naturally
    fades rather than always counting as a fixed fraction of the decision.
    """

    def __init__(self, predicted, prior_worlds):
        self.active = list(predicted)
        self.predicted = predicted
        self.prior_worlds = prior_worlds
        self.totals = dict.fromkeys(predicted, 0.0)
        self.samples = dict.fromkeys(predicted, 0)

    def add(self, means, count):
        for card in self.active:
            self.totals[card] += means[card] * count
            self.samples[card] += count

    def value(self, card):
        weight = self.prior_worlds
        count = self.samples[card]
        if not count and not weight:
            return self.predicted[card]
        return (self.totals[card] + weight * self.predicted[card]) / (count + weight)

    def ranked(self):
        return sorted(self.active, key=self.value, reverse=True)

    def halve(self):
        # Keep at least two cards so the final budget still compares alternatives.
        self.active = self.ranked()[:max(2, math.ceil(len(self.active) / 2))]

    def best(self):
        return max(self.active, key=self.value)


class AdaptiveBot(HybridBot):
    """Hybrid bidding with adaptive card search; exact endgames stay unchanged.

    Timed search divides the remaining turn among successive halving stages.
    Untimed search uses ``worlds * number_of_legal_cards`` card/deal evaluations
    in total. ``batch_worlds`` bounds the work between deadline checks.
    """

    def __init__(self, seed=0, batch_worlds=4, **kwargs):
        if not isinstance(batch_worlds, int) or not 1 <= batch_worlds <= 1024:
            raise ValueError('batch_worlds must be an integer from 1 to 1024.')
        super().__init__(seed=seed, **kwargs)
        if self.worlds < 1 or self.prior_worlds < 0:
            raise ValueError('worlds must be positive and prior_worlds nonnegative.')
        self.batch_worlds = batch_worlds
        self.last_search = None

    def _batch(self, obs, cards, count):
        if self.simulator:
            # One complete batch, not a fresh full turn allowance for each call.
            return self.simulator.evaluate_cards(obs, cards, count, 0.0)
        totals = dict.fromkeys(cards, 0.0)
        for hands, kinds in self.belief.worlds(obs, count, self.rng):
            base = obs.determinize(hands)
            policy = PopulationPolicy(kinds, self.rng, self.react)
            for card in cards:
                state = base.clone()
                state.play_unchecked(card)
                playout(state, policy)
                totals[card] += state.points()[obs.seat]
        return {card: value / count for card, value in totals.items()}

    def play(self, obs):
        self.last_search = None
        cards = obs.legal_actions()
        remaining = obs.hand.bit_count() * obs.n - obs.trick_size
        if len(cards) == 1 or remaining <= self.endgame_cards:
            return super().play(obs)

        predicted = dict(zip(cards, self.models.card_values(
            obs, obs.seat, obs.hand, cards)))
        race = CardRace(predicted, self.prior_worlds)
        budget = max(len(cards), int(self.worlds) * len(cards))
        stages = 1
        width = len(cards)
        while width > 2:
            width = math.ceil(width / 2)
            stages += 1

        for stage in range(stages):
            left = stages - stage
            stage_budget = budget // left
            stage_stop = (time.perf_counter() +
                          max(0.0, self.deadline - time.perf_counter()) / left
                          if self.deadline is not None else None)
            spent = 0
            while True:
                now = time.perf_counter()
                # If model inference used the turn, return its ranking. Otherwise
                # finish at least one batch before pruning any untested card.
                if self.deadline is not None and now >= self.deadline:
                    break
                if stage_stop is None:
                    count = min(self.batch_worlds,
                                max(0, stage_budget - spent) // len(race.active))
                    if not count:
                        break
                else:
                    if spent and now >= stage_stop:
                        break
                    count = self.batch_worlds
                means = self._batch(obs, race.active, count)
                race.add(means, count)
                spent += count * len(race.active)
            budget -= spent
            if not spent:
                break
            if stage != stages - 1:
                race.halve()

        self.last_search = {
            'legal_cards': len(cards),
            'samples': dict(race.samples),
            'finalists': list(race.active),
            'values': {card: race.value(card) for card in cards},
        }
        return race.best()
