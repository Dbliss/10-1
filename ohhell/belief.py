"""Guessing the hidden cards.

``sample_world`` draws a deal consistent with the hard facts: known cards, how
many cards each seat holds, and the suits each seat has shown it lacks.

``Belief`` also guesses what kind of player each opponent is, and adds soft
evidence by importance resampling.
"""

import math

from . import estimate as estimate_v2
from .bots.heuristic import expected_tricks, state_card
from .cards import EFFECTIVE_SUIT, FULL_DECK, cards_in
from .engine import NO_BID, PLAYING, RoundState, legal_card_mask
from .opponents import FIELDS, KINDS, SKILLED


def _assign(cards, capacity, voids, suit_of, rng):
    """Deal ``cards`` to bins at random, or return None at a dead end.

    ``suit_of`` maps a card to its effective suit for this trump.
    """
    bins = [0] * len(capacity)
    room = capacity[:]
    for card in cards:
        suit_bit = 1 << suit_of[card]
        total = 0
        for i, space in enumerate(room):
            if space and not voids[i] & suit_bit:
                total += space
        if not total:
            return None
        pick = rng.randrange(total)
        for i, space in enumerate(room):
            if space and not voids[i] & suit_bit:
                pick -= space
                if pick < 0:
                    bins[i] |= 1 << card
                    room[i] -= 1
                    break
    return bins


def _assign_exactly(cards, capacity, voids, suit_of, rng):
    """Backtracking fallback for the rare deals random dealing can't finish."""
    order = list(range(len(capacity)))
    bins = [0] * len(capacity)
    room = capacity[:]
    dead = set()

    def place(index):
        if index == len(cards):
            return True
        key = (suit_of[cards[index]], index, tuple(room))
        if key in dead:
            return False
        suit_bit = 1 << suit_of[cards[index]]
        rng.shuffle(order)
        for i in order[:]:
            if room[i] and not voids[i] & suit_bit:
                room[i] -= 1
                bins[i] |= 1 << cards[index]
                if place(index + 1):
                    return True
                room[i] += 1
                bins[i] ^= 1 << cards[index]
        dead.add(key)
        return False

    if not place(0):
        raise ValueError("No deal is consistent with this observation.")
    return bins


def sample_world(obs, rng):
    """Return one guess at every seat's current hand."""
    known = obs.hand | obs.played | (1 << obs.trump_card)
    unknown = cards_in(FULL_DECK ^ known)

    held = [obs.hand_size] * obs.n
    for seat, _ in obs.history:
        held[seat] -= 1

    others = [s for s in range(obs.n) if s != obs.seat]
    capacity = [held[s] for s in others]
    voids = [obs.voids[s] for s in others]
    # The undealt stock is one more bin, and it can hold any suit.
    capacity.append(len(unknown) - sum(capacity))
    voids.append(0)

    # Deal the suits the most players lack first, to avoid dead ends.
    suit_of = EFFECTIVE_SUIT[obs.trump_suit]
    pressure = [sum(v >> s & 1 for v in voids) for s in range(4)]
    bins = None
    for _ in range(40):
        rng.shuffle(unknown)
        unknown.sort(key=lambda c: -pressure[suit_of[c]])
        bins = _assign(unknown, capacity, voids, suit_of, rng)
        if bins is not None:
            break
    if bins is None:
        bins = _assign_exactly(unknown, capacity, voids, suit_of, rng)

    hands = [0] * obs.n
    hands[obs.seat] = obs.hand
    for index, seat in enumerate(others):
        hands[seat] = bins[index]
    return hands


class Belief:
    """Samples deals together with a kind of player for each opponent.

    With inference on, a deal and its kinds are kept more often when they
    explain the bids and plays so far. A kind that plays erratically explains
    any card equally, while a skilled kind explains sensible cards far better,
    so the sample shifts towards whatever the table has actually shown.
    """

    def __init__(self, inference=True, oversample=4, field='mix', react=False, estimate=1,
                 bid_sigma=None):
        self.react = react
        self.estimate = estimate  # 1: original rule of thumb, 2 or 3: estimate.py
        self.bid_sigma = bid_sigma  # overrides every kind's bid_sigma when set
        self.inference = inference
        self.oversample = oversample
        self.weights = FIELDS[field]

    def _kinds(self, obs, rng):
        kinds = rng.choices(KINDS, self.weights, k=obs.n)
        kinds[obs.seat] = SKILLED
        return kinds

    def worlds(self, obs, count, rng):
        """Return ``count`` pairs of (every seat's hand, every seat's kind)."""
        if not self.inference or not self._has_evidence(obs):
            return [(sample_world(obs, rng), self._kinds(obs, rng)) for _ in range(count)]

        candidates = [
            (sample_world(obs, rng), self._kinds(obs, rng))
            for _ in range(count * self.oversample)
        ]
        logs = [self.log_weight(obs, hands, kinds) for hands, kinds in candidates]
        top = max(logs)
        weights = [math.exp(v - top) for v in logs]
        return rng.choices(candidates, weights=weights, k=count)

    @staticmethod
    def _has_evidence(obs):
        return any(b != NO_BID for s, b in enumerate(obs.bids) if s != obs.seat)

    def log_weight(self, obs, hands, kinds):
        played_by = [0] * obs.n
        for seat, card in obs.history:
            played_by[seat] |= 1 << card
        opening = [hands[s] | played_by[s] for s in range(obs.n)]

        total = 0.0
        first = (obs.dealer + 1) % obs.n
        for seat in range(obs.n):
            if seat != obs.seat and obs.bids[seat] != NO_BID:
                if self.estimate >= 2:
                    estimate = estimate_v2.expected_tricks(
                        opening[seat], obs.trump_card, obs.n, obs.hand_size,
                        estimate_v2.first_trick_position(seat, obs.dealer, obs.n),
                        version=self.estimate)
                else:
                    estimate = expected_tricks(
                        opening[seat], obs.trump_card, obs.n, obs.hand_size, seat == first)
                sigma = self.bid_sigma or kinds[seat].bid_sigma
                total -= (obs.bids[seat] - estimate) ** 2 / (2.0 * sigma ** 2) + math.log(sigma)

        if obs.phase == PLAYING and obs.history:
            total += self._play_log_weight(obs, opening, kinds)
        return total

    def _play_log_weight(self, obs, opening, kinds):
        replay = RoundState(obs.rules, opening, obs.trump_card, obs.dealer)
        replay.history = None
        replay.bids = list(obs.bids)
        replay.bids_made = obs.n
        replay.phase = PLAYING
        replay.leader = replay.to_act = (obs.dealer + 1) % obs.n

        total = 0.0
        for seat, card in obs.history:
            if seat != obs.seat:
                options = legal_card_mask(replay.hands[seat], replay.lead_suit, replay.trump_suit).bit_count()
                if options > 1:
                    doubt = kinds[seat].play_doubt
                    chance = doubt / options
                    if state_card(replay, self.react) == card:
                        chance += 1.0 - doubt
                    total += math.log(chance)
            replay.play_unchecked(card)
        return total
