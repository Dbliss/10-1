"""Cards as integers 0-51 and sets of cards as 52-bit masks.

A card is ``suit * 13 + rank``, with rank 0 for a two and 12 for an ace, so
within a suit a higher integer always means a higher card.
"""

NUM_SUITS = 4
NUM_RANKS = 13
DECK_SIZE = 52
FULL_DECK = (1 << DECK_SIZE) - 1

SUIT_NAMES = ('Clubs', 'Diamonds', 'Hearts', 'Spades')
RANK_NAMES = ('2', '3', '4', '5', '6', '7', '8', '9', '10',
              'Jack', 'Queen', 'King', 'Ace')

SUIT_OF = tuple(card // NUM_RANKS for card in range(DECK_SIZE))
RANK_OF = tuple(card % NUM_RANKS for card in range(DECK_SIZE))
SUIT_MASK = tuple(((1 << NUM_RANKS) - 1) << (NUM_RANKS * s) for s in range(NUM_SUITS))
# Every card of the same suit that outranks the given card.
ABOVE = tuple(SUIT_MASK[SUIT_OF[c]] & ~((2 << c) - 1) for c in range(DECK_SIZE))


def make_card(suit, rank):
    return suit * NUM_RANKS + rank


def card_id(card):
    """Name a card the way the web app does, e.g. ``Jack_of_Hearts``."""
    return f"{RANK_NAMES[RANK_OF[card]]}_of_{SUIT_NAMES[SUIT_OF[card]]}"


def card_from_id(text):
    rank, _, suit = text.partition('_of_')
    return make_card(SUIT_NAMES.index(suit), RANK_NAMES.index(rank))


def mask_of(cards):
    mask = 0
    for card in cards:
        mask |= 1 << card
    return mask


def cards_in(mask):
    """List the cards in a mask, lowest integer first."""
    cards = []
    while mask:
        low = mask & -mask
        cards.append(low.bit_length() - 1)
        mask ^= low
    return cards


def highest(mask):
    return mask.bit_length() - 1


def lowest(mask):
    return (mask & -mask).bit_length() - 1


# ---------------------------------------------------------------------------
# Bowers. The Jack of trump (right bower) is the highest card, then the other
# Jack of the same colour (left bower), then Ace down to two. The left bower is
# a trump for every purpose, so in that round the trump suit has 14 cards and
# the same-colour suit has 12. Every table below is indexed by trump suit.

JACK = 9


def partner_suit(suit):
    """The other suit of the same colour: clubs/spades, diamonds/hearts."""
    return 3 - suit


RIGHT_BOWER = tuple(make_card(t, JACK) for t in range(NUM_SUITS))
LEFT_BOWER = tuple(make_card(partner_suit(t), JACK) for t in range(NUM_SUITS))


def _effective_suit(trump, card):
    return trump if card == LEFT_BOWER[trump] else SUIT_OF[card]


def _power(trump, card):
    if card == RIGHT_BOWER[trump]:
        return NUM_RANKS + 1
    if card == LEFT_BOWER[trump]:
        return NUM_RANKS
    return RANK_OF[card]


# The suit a card belongs to when following, leading and showing voids.
EFFECTIVE_SUIT = tuple(
    tuple(_effective_suit(t, c) for c in range(DECK_SIZE)) for t in range(NUM_SUITS))
# Strength within the effective suit: 0-12 for two to Ace, then 13 and 14 for
# the left and right bowers.
POWER = tuple(tuple(_power(t, c) for c in range(DECK_SIZE)) for t in range(NUM_SUITS))
# The cards of each effective suit.
SUIT_CARDS = tuple(
    tuple(mask_of(c for c in range(DECK_SIZE) if EFFECTIVE_SUIT[t][c] == s)
          for s in range(NUM_SUITS))
    for t in range(NUM_SUITS))
# Every card of the same effective suit that outranks the given card.
OUTRANKS = tuple(
    tuple(mask_of(o for o in range(DECK_SIZE)
                  if EFFECTIVE_SUIT[t][o] == EFFECTIVE_SUIT[t][c]
                  and POWER[t][o] > POWER[t][c])
          for c in range(DECK_SIZE))
    for t in range(NUM_SUITS))


BOWERS = tuple((1 << RIGHT_BOWER[t]) | (1 << LEFT_BOWER[t]) for t in range(NUM_SUITS))


def top(mask, trump):
    """The strongest card in a non-empty mask of one effective suit."""
    if mask & BOWERS[trump]:
        right = RIGHT_BOWER[trump]
        return right if mask >> right & 1 else LEFT_BOWER[trump]
    return mask.bit_length() - 1


def bottom(mask, trump):
    """The weakest card in a non-empty mask of one effective suit."""
    rest = mask & ~BOWERS[trump]
    if rest:
        return (rest & -rest).bit_length() - 1
    left = LEFT_BOWER[trump]
    return left if mask >> left & 1 else RIGHT_BOWER[trump]


def beats(card, best, trump):
    """Whether ``card`` takes a trick currently held by ``best``."""
    suit = EFFECTIVE_SUIT[trump][card]
    if suit == EFFECTIVE_SUIT[trump][best]:
        return POWER[trump][card] > POWER[trump][best]
    return suit == trump
