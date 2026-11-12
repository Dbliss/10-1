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


def solve_root(state, cards, seat):
    """Value, for ``seat``, of each candidate card under exact search."""
    memo = {}
    values = {}
    for card in cards:
        child = state.clone()
        child.play_unchecked(card)
        values[card] = solve(child, seat, memo)[seat]
    return values
