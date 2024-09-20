# player.py

import random
from card import Card

class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []
        self.bid = None
        self.tricks_won = 0
        self.score = 0

    def make_bid(self, bids_so_far, players_left, total_cards, known_cards, trump):
        # Simple AI bid logic
        # Estimate based on high cards and trump cards
        high_cards = [card for card in self.hand if card.rank in ['Ace', 'King', 'Queen', 'Jack']]
        trump_cards = [card for card in self.hand if card.suit == trump]
        estimate = int(len(high_cards) * 0.7 + len(trump_cards) * 0.4)
        # Introduce slight randomness
        estimate += random.choice([0, 1])
        return max(estimate, 0)  # Ensure bid is not negative

    def play_card(self, lead_suit, played_cards, known_cards, trump):
        # Simple AI play logic
        playable_cards = [card for card in self.hand if card.suit == lead_suit]
        if playable_cards:
            # Play the lowest card that can win the trick if possible
            winning_card = None
            for card in playable_cards:
                if not winning_card or card.value() > winning_card.value():
                    winning_card = card
            if winning_card:
                return winning_card
            else:
                return playable_cards[0]
        else:
            # No cards to follow suit
            # Play the lowest trump if possible
            trump_cards = [card for card in self.hand if card.suit == trump]
            if trump_cards:
                return min(trump_cards, key=lambda c: c.value())
            else:
                # Play the lowest card
                return min(self.hand, key=lambda c: c.value())
