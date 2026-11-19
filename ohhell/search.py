"""Playing out a known deal, either by policy or by exact search."""

from .bots.heuristic import state_card
from .cards import EFFECTIVE_SUIT, OUTRANKS, POWER, cards_in
from .engine import PLAYING, legal_card_mask


def playout(state, policy=state_card):
    """Finish the round in place, every seat following ``policy``."""
    while state.phase == PLAYING:
        state.play_unchecked(policy(state))
    return state


def distinct_moves(state):
    """Legal cards, skipping those equivalent to a touching card already kept.

    Two cards in a suit are interchangeable when no card still in someone's
    hand or on the table sits between them. Cards are walked in order of
    strength so that the bowers sit next to the Ace of trump.
    """
    trump = state.trump_suit
    legal = legal_card_mask(state.hands[state.to_act], state.lead_suit, trump)
    power = POWER[trump]
    outranks = OUTRANKS[trump]
    suit_of = EFFECTIVE_SUIT[trump]
    cards = sorted(cards_in(legal), key=lambda c: (suit_of[c], power[c]))
    if len(cards) < 2:
        return cards
    live = state.table
    for hand in state.hands:
        live |= hand
    kept = [cards[0]]
    for card in cards[1:]:
        between = outranks[kept[-1]] & ~outranks[card] & ~(1 << card)
        same_suit = (outranks[kept[-1]] >> card) & 1
        if not (same_suit and not between & live):
            kept.append(card)
        else:
            kept[-1] = card
    return kept


def solve(state, root, memo):
    """Exact play-out where each seat maximises its own points.

    Returns every seat's final points. Ties go against ``root``, which keeps
    the value from relying on an opponent's generosity.
    """
    if state.phase != PLAYING:
        return tuple(state.points())

    key = None
    if state.trick_size == 0:
        key = (tuple(state.hands), state.leader, tuple(state.tricks_won))
        known = memo.get(key)
        if known is not None:
            return known

    seat = state.to_act
    best = None
    for card in distinct_moves(state):
        child = state.clone()
        child.play_unchecked(card)
        value = solve(child, root, memo)
        if (best is None or value[seat] > best[seat]
                or (value[seat] == best[seat] and value[root] < best[root])):
            best = value

    if key is not None:
        memo[key] = best
    return best


def distinct_moves_counted(state):
    """``distinct_moves`` with how many legal cards each kept card stands for."""
    kept = distinct_moves(state)
    legal = cards_in(legal_card_mask(state.hands[state.to_act], state.lead_suit,
                                     state.trump_suit))
    order = {card: i for i, card in enumerate(kept)}
    trump = state.trump_suit
    power = POWER[trump]
    suit_of = EFFECTIVE_SUIT[trump]
    sizes = [0] * len(kept)
    for card in legal:
        # A merged card goes to the weakest kept card of its suit at or above it.
        owner = min((k for k in kept if suit_of[k] == suit_of[card] and power[k] >= power[card]),
                    key=lambda k: power[k])
        sizes[order[owner]] += 1
    return kept, sizes


def solve_noisy(state, root, noise, memo):
    """``solve`` for seats that play a random legal card with chance ``noise[seat]``.

    The mover's best line (ties against ``root``) has weight ``1 - noise`` and
    the mean over every legal card has weight ``noise``. Matches the C core.
    """
    if state.phase != PLAYING:
        return tuple(float(x) for x in state.points())

    key = None
    if state.trick_size == 0:
        key = (tuple(state.hands), state.leader, tuple(state.tricks_won))
        known = memo.get(key)
        if known is not None:
            return known

    seat = state.to_act
    eps = noise[seat]
    best = None
    mean = [0.0] * len(state.hands)
    total = 0
    for card, size in zip(*distinct_moves_counted(state)):
        child = state.clone()
        child.play_unchecked(card)
        value = solve_noisy(child, root, noise, memo)
        if (best is None or value[seat] > best[seat]
                or (value[seat] == best[seat] and value[root] < best[root])):
            best = value
        if eps > 0:
            mean = [m + size * v for m, v in zip(mean, value)]
            total += size
    if eps > 0:
        best = tuple((1 - eps) * b + eps * m / total for b, m in zip(best, mean))

    if key is not None:
        memo[key] = best
    return best


def solve_root(state, cards, seat, noise=None):
    """Value, for ``seat``, of each candidate card under exact search.

    With ``noise`` (chance per seat of a random card), uses ``solve_noisy``;
    ``seat`` itself is always taken to play its best card.
    """
    memo = {}
    values = {}
    if noise is not None:
        noise = list(noise)
        noise[seat] = 0.0
    for card in cards:
        child = state.clone()
        child.play_unchecked(card)
        values[card] = (solve(child, seat, memo) if noise is None
                        else solve_noisy(child, seat, noise, memo))[seat]
    return values
