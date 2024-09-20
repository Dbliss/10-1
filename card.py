# card.py

class Card:
    def __init__(self, suit, rank):
        self.suit = suit  # 'Hearts', 'Diamonds', 'Clubs', 'Spades'
        self.rank = rank  # '2' to '10', 'Jack', 'Queen', 'King', 'Ace'
    
    def value(self):
        if self.rank.isdigit():
            return int(self.rank)
        elif self.rank == 'Jack':
            return 11
        elif self.rank == 'Queen':
            return 12
        elif self.rank == 'King':
            return 13
        elif self.rank == 'Ace':
            return 14
    
    def __str__(self):
        return f"{self.rank}_of_{self.suit}"
    
    def __repr__(self):
        return self.__str__()
