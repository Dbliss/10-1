from deck import Deck
from player import Player
from image_loader import load_card_images
import random
from dash import html

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
JACK_RULE = True  # Enable the jack rule

class Game:
    def __init__(self):
        self.total_rounds = 0
        self.round_number = 0
        self.current_round_index = 0
        self.num_players = 8  # Including the user
        self.players = []
        self.deck = Deck()
        self.player = Player("You")
        self.ai_players = [Player(f"AI {i}") for i in range(1, self.num_players)]
        self.trump_card = None
        self.tricks_won = {player.name: 0 for player in self.players}  # Track tricks won
        self.bidding_order = []
        self.bids = []
        self.bidders = set()  # Set to keep track of who has bid
        self.play_order = []
        self.current_player_index = 0
        self.tricks_played = 0
        self.trick_cards = {}
        self.lead_suit = None
        self.played_cards = []
        self.total_cards = 0  # Total tricks in the current round
        self.log = []
        self.dealer_index = random.randint(0, self.num_players - 1)  # Randomly select the first dealer
        self.initialize_game()
        
    def initialize_game(self):
        self.players = self.ai_players[0:2] + [self.player] + self.ai_players[2:self.num_players]
        load_card_images()

    def determine_dealer(self):
        # Move the dealer clockwise
        self.dealer_index = (self.dealer_index + 1) % self.num_players
        return self.dealer_index

    def start_round(self, round_number):
        self.round_number = round_number
        self.bids.clear()
        self.bidders.clear()  # Clear the set of bidders
        self.deck = Deck()
        self.deck.shuffle()
        self.tricks_played = 0
        self.trick_cards = {}
        self.lead_suit = None
        
        self.total_cards = round_number if round_number > 0 else 1
        for player in self.players:
            player.hand = self.deck.deal(self.total_cards)
            player.bid = None  # Reset bid for the new round

        # Flip the trump card correctly
        self.trump_card = self.deck.flip_trump()
        self.trump = self.trump_card.suit if self.trump_card else None

        # Return trump card and hands
        return (
            self.get_card_image(self.trump_card),  # Return the image path for the trump card
            {player.name: [self.get_card_image(card) for card in player.hand] for player in self.players}
        )

    def receive_bid(self, player_name, bid):
        player = next((p for p in self.players if p.name == player_name), None)
        if player:
            if not (0 <= bid <= self.round_number):
                return f"Please enter a valid bid between 0 and {self.round_number}."
            
            # Check to see if this player has already bid
            if player_name not in self.bidders:
                player.bid = bid
                self.bids.append(bid)  # Append the actual bid amount
                self.bidders.add(player_name)  # Track that this player has bid
                self.log_action(f"{player_name} bids {bid}")

                # Check if all players have bid
                if len(self.bidders) == len(self.players):
                    self.determine_dealer()  # Determine new dealer
                    return "All bids are in, dealer is now: " + self.players[self.dealer_index].name
        return None

    def manage_turns(self):
        # Start with the player after the dealer
        start_index = (self.dealer_index + 1) % self.num_players

        for idx in range(self.num_players):
            current_player_index = (start_index + idx) % self.num_players
            current_player = self.players[current_player_index]

            if current_player.name == "You":
                return "Your turn to bid.", True  # Player's turn to bid
            else:
                # AI bidding logic
                ai_bid = random.randint(0, self.round_number)  # AI bid example
                self.receive_bid(current_player.name, ai_bid)
                self.log_action(f"{current_player.name} bids {ai_bid}")

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
        self.played_cards.append(card)
        self.trick_cards[player] = card
        if not self.is_round_zero:
            self.known_cards.append(str(card))
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
                    self.log_action(f"{player.name} successfully bid 0 and earns +10 points.")
                else:
                    earned = 10 * player.bid
                    player.score += earned
                    self.log_action(f"{player.name} successfully bid {player.bid} and earns +{earned} points.")
            else:
                penalty = 10 * abs(player.tricks_won - player.bid)
                player.score -= penalty
                self.log_action(f"{player.name} missed their bid by {abs(player.tricks_won - player.bid)} and loses -{penalty} points.")
            scores[player.name] = player.score

        return scores

    def log_action(self, action_text):
        self.log.append(action_text)

    def get_card_image(self, card):
        """Return the path to the card image based on its suit and rank."""
        card_image_path = f"/assets/cards/{card.rank}_of_{card.suit}.png"  # Example path where card images are stored
        return card_image_path
