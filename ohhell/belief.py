"""Guessing the hidden cards.

``sample_world`` draws a deal consistent with the hard facts: known cards, how
many cards each seat holds, and the suits each seat has shown it lacks.

``Belief`` adds soft evidence on top by importance resampling. Worlds in which
an opponent's bid and plays look natural are kept more often.
"""

import math

from .bots.heuristic import expected_tricks, state_card
from .cards import FULL_DECK, SUIT_MASK, SUIT_OF, cards_in
from .engine import NO_BID, PLAYING, RoundState, legal_card_mask


def _assign(cards, capacity, voids, rng):
    """Deal ``cards`` to bins at random, or return None at a dead end."""
    bins = [0] * len(capacity)
    room = capacity[:]
    for card in cards:
        suit_bit = 1 << SUIT_OF[card]
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


def _assign_exactly(cards, capacity, voids, rng):
    """Backtracking fallback for the rare deals random dealing can't finish."""
    order = list(range(len(capacity)))
    bins = [0] * len(capacity)
    room = capacity[:]
    dead = set()

    def place(index):
        if index == len(cards):
            return True
        key = (SUIT_OF[cards[index]], index, tuple(room))
        if key in dead:
            return False
        suit_bit = 1 << SUIT_OF[cards[index]]
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
    pressure = [sum(v >> s & 1 for v in voids) for s in range(4)]
    bins = None
    for _ in range(40):
        rng.shuffle(unknown)
        unknown.sort(key=lambda c: -pressure[SUIT_OF[c]])
        bins = _assign(unknown, capacity, voids, rng)
        if bins is not None:
            break
    if bins is None:
        bins = _assign_exactly(unknown, capacity, voids, rng)

    hands = [0] * obs.n
    hands[obs.seat] = obs.hand
    for index, seat in enumerate(others):
        hands[seat] = bins[index]
    return hands


class Belief:
    """Samples worlds, optionally weighted by opponents' bids and plays.

    The opponent model is the heuristic bot, loosened so that it tolerates
    players who don't behave like it: ``bid_sigma`` is how far a bid may stray
    from the estimate, and ``play_miss`` is how much less likely a world becomes
    when the observed card isn't the one the heuristic would have played.
    """

    def __init__(self, inference=True, oversample=4, bid_sigma=1.4, play_miss=0.5):
        self.inference = inference
        self.oversample = oversample
        self.bid_sigma = bid_sigma
        self.play_log_miss = math.log(play_miss)

    def worlds(self, obs, count, rng):
        if not self.inference or not self._has_evidence(obs):
            return [sample_world(obs, rng) for _ in range(count)]

        candidates = [sample_world(obs, rng) for _ in range(count * self.oversample)]
        logs = [self.log_weight(obs, hands) for hands in candidates]
        top = max(logs)
        weights = [math.exp(v - top) for v in logs]
        return rng.choices(candidates, weights=weights, k=count)

    @staticmethod
    def _has_evidence(obs):
        return any(b != NO_BID for s, b in enumerate(obs.bids) if s != obs.seat)

    def log_weight(self, obs, hands):
        played_by = [0] * obs.n
        for seat, card in obs.history:
            played_by[seat] |= 1 << card
        opening = [hands[s] | played_by[s] for s in range(obs.n)]

        total = 0.0
        first = (obs.dealer + 1) % obs.n
        spread = 2.0 * self.bid_sigma ** 2
        for seat in range(obs.n):
            if seat != obs.seat and obs.bids[seat] != NO_BID:
                estimate = expected_tricks(
                    opening[seat], obs.trump_card, obs.n, obs.hand_size, seat == first)
                total -= (obs.bids[seat] - estimate) ** 2 / spread

        if obs.phase == PLAYING and obs.history:
            total += self._play_log_weight(obs, opening)
        return total

    def _play_log_weight(self, obs, opening):
        replay = RoundState(obs.rules, opening, obs.trump_card, obs.dealer)
        replay.history = None
        replay.bids = list(obs.bids)
        replay.bids_made = obs.n
        replay.phase = PLAYING
        replay.leader = replay.to_act = (obs.dealer + 1) % obs.n

        total = 0.0
        for seat, card in obs.history:
            if seat != obs.seat:
                options = legal_card_mask(replay.hands[seat], replay.lead_suit)
                if options & (options - 1) and state_card(replay) != card:
                    total += self.play_log_miss
            replay.play_unchecked(card)
        return total
