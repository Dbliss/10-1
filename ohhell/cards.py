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
