# deck.py

from card import Card
import random

class Deck:
    def __init__(self):
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
        self.cards = [Card(suit, rank) for suit in suits for rank in ranks]
        self.trump_card = None

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, num_cards):
        if num_cards > len(self.cards):
            raise ValueError("Not enough cards to deal.")
        dealt_cards = self.cards[:num_cards]
        self.cards = self.cards[num_cards:]
        return dealt_cards

    def flip_trump(self):
        if not self.cards:
            return None
        self.trump_card = self.cards.pop(0)
        return self.trump_card

    def get_card(self, card_str):
        for card in self.cards:
            if str(card) == card_str:
                return card
        return None

