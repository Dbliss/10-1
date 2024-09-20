from deck import Deck
from player import Player
from image_loader import load_card_images
import random

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']

class Game:
    def __init__(self):
        self.round_number = 0
        self.num_players = 8  # Including the user
        self.players = []
        self.deck = Deck()
        self.player = Player("You")
        self.ai_players = [Player(f"AI {i}") for i in range(1, self.num_players)]
        self.trump_card = None
        self.tricks_won = {player.name: 0 for player in self.players}  # Track tricks won
        self.bids = []
        self.bidders = set()  # Set to keep track of who has bid
        self.dealer_index = random.randint(0, self.num_players - 1)  # Randomly select the first dealer
        self.initialize_game()

    def initialize_game(self):
        self.players = self.ai_players[0:2] + [self.player] + self.ai_players[2:self.num_players]
        load_card_images()

    def determine_dealer(self):
        # Move the dealer clockwise
        self.dealer_index = (self.dealer_index + 1) % self.num_players

    def start_round(self):
        self.bids.clear()
        self.bidders.clear()  # Clear the set of bidders
        self.deck = Deck()
        self.deck.shuffle()
        self.tricks_won = {player.name: 0 for player in self.players}  # Reset tricks won
        self.trump_card = self.deck.flip_trump()

        # Deal cards based on the round number
        for player in self.players:
            player.hand = self.deck.deal(self.round_number if self.round_number > 0 else 1)
            player.bid = None  # Reset bid for the new round

        return self.get_card_image(self.trump_card), {player.name: [self.get_card_image(card) for card in player.hand] for player in self.players}

    def receive_bid(self, player_name, bid):
        player = next((p for p in self.players if p.name == player_name), None)
        if player and player_name not in self.bidders:
            player.bid = bid
            self.bids.append(bid)  # Append the actual bid amount
            self.bidders.add(player_name)  # Track that this player has bid
            if len(self.bidders) == self.num_players:
                self.determine_dealer()  # Determine new dealer
                return "All bids are in, dealer is now: " + self.players[self.dealer_index].name
        return None

    def manage_turns(self):
        start_index = (self.dealer_index + 1) % self.num_players
        for idx in range(self.num_players):
            current_player_index = (start_index + idx) % self.num_players
            current_player = self.players[current_player_index]

            if current_player.name == "You":
                return "Your turn to play.", True  # Player's turn to play
            else:
                ai_bid = random.randint(0, self.round_number)  # AI bid example
                self.receive_bid(current_player.name, ai_bid)
        return "All bids are in.", False  # Indicate bidding complete

    def play_card(self, player_name, card_str):
        player = next((p for p in self.players if p.name == player_name), None)
        if not player:
            return f"Player {player_name} not found."

        card = self.deck.get_card(card_str)
        if card not in player.hand:
            return f"{player_name} cannot play {card_str}."

        playable_cards = self.get_playable_cards(player)
        if card not in playable_cards:
            return f"You must follow suit if possible."

        player.hand.remove(card)
        self.tricks_won[player.name] += 1  # Increment tricks won
        self.log_action(f"{player_name} played {card}")

        return None

    def get_playable_cards(self, player):
        if not self.trick_cards:
            return player.hand
        else:
            follow_suit_cards = [card for card in player.hand if card.suit == self.lead_suit]
            return follow_suit_cards if follow_suit_cards else player.hand

    def calculate_scores(self):
        scores = {}
        for player in self.players:
            if player.tricks_won == player.bid:
                if player.bid == 0:
                    player.score += 10
                else:
                    earned = 10 * player.bid
                    player.score += earned
            else:
                penalty = 10 * abs(player.tricks_won - player.bid)
                player.score -= penalty
            scores[player.name] = player.score

        return scores

    def log_action(self, action_text):
        self.log.append(action_text)

    def get_card_image(self, card):
        """Return the path to the card image based on its suit and rank."""
        card_image_path = f"/assets/cards/{card.rank}_of_{card.suit}.png"  # Example path where card images are stored
        return card_image_path
