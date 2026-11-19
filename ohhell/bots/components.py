"""Small, observation-only seams for experimental hybrid components."""

from typing import Protocol

from ..cards import EFFECTIVE_SUIT, OUTRANKS


class Sampler(Protocol):
    def worlds(self, obs, count, rng): ...


class Evaluator(Protocol):
    def card_values(self, obs, seat, hand, cards): ...


class Policy(Protocol):
    def __call__(self, state): ...


def touching_groups(obs, cards):
    """Group ranks with no possibly live outside card between them.

    This is an experimental shortlist rule, not a rollout equivalence proof:
    a rank-sensitive continuation can still distinguish the two cards.
    """
    trump = obs.trump_suit
    outside = ~(obs.played | obs.hand | (1 << obs.trump_card)) | obs.table
    groups = []
    for card in cards:  # caller's model preference order
        for group in groups:
            other = group[0]
            if EFFECTIVE_SUIT[trump][card] != EFFECTIVE_SUIT[trump][other]:
                continue
            between = (OUTRANKS[trump][card] ^ OUTRANKS[trump][other])
            between &= ~((1 << card) | (1 << other))
            if not between & outside:
                group.append(card)
                break
        else:
            groups.append([card])
    return groups


class TimeBank:
    """Pay actual elapsed time, allowing only previously earned credit.

    Only decisions with a choice earn/pay time, matching the tournament clock.
    Debt is carried, never forgiven by the capacity limit. Native deadlines
    remain soft; this controller cannot interrupt one expensive sampled deal.
    """
    def __init__(self, seconds, capacity=4):
        if seconds is None or seconds <= 0 or capacity < 1:
            raise ValueError('A time bank needs positive time_ms and capacity >= 1.')
        self.seconds = seconds
        self.capacity = capacity * seconds
        self.credit = 0.0

    def begin(self):
        return max(1e-6, self.seconds + self.credit)

    def finish(self, elapsed):
        self.credit = min(self.capacity, self.credit + self.seconds - elapsed)

    def charge(self, elapsed):
        self.credit -= elapsed


class FixedController:
    def __init__(self, seconds):
        self.seconds = seconds

    def begin(self):
        return self.seconds

    def finish(self, elapsed):
        pass

    def charge(self, elapsed):
        pass
