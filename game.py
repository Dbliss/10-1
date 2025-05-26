from deck import Deck
from player import Player
from image_loader import load_card_images
import random
import time

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']

class Game:
    def __init__(self, num_players=4, num_rounds=10):
        """Create a new game instance.

        Parameters
        ----------
        num_players : int, optional
            Total number of players including the user. The game UI supports up
            to 8 seats and any unused seats will remain empty.
        """
        self.round_number = 0  # current round index (1 based)
        self.num_players = num_players  # Including the user
        self.total_rounds = num_rounds
        self.current_cards = num_rounds
        self.players = []
        self.deck = Deck()
        self.player = Player("You")
        self.ai_players = [Player(f"AI {i}") for i in range(1, self.num_players)]
        self.trump_card = None
        # Track tricks won for each player; will be populated once players are initialized
        self.tricks_won = {}
        self.trick_piles = {}
        self.bids = []
        self.bidders = set()  # Set to keep track of who has bid
        self.dealer_index = random.randint(0, self.num_players - 1)  # Randomly select the first dealer
        self.leader_index = (self.dealer_index + 1) % self.num_players
        # Track the cards played in the current trick and which suit was led
        self.trick_cards = []
        self.lead_suit = None
        # Simple list to record actions for debugging/game history
        self.log = []
        # Game phase: 'bidding' or 'playing'
        self.phase = 'bidding'
        self.initialize_game()

    def initialize_game(self):
        """Initialise the list of players for the game."""
        # The user always takes the first seat and the remaining seats are
        # filled with AI players up to ``num_players``.
        self.players = [self.player] + self.ai_players[: self.num_players - 1]
        load_card_images()
        # initialise trick piles for each player
        self.trick_piles = {player.name: [] for player in self.players}

    def determine_dealer(self):
        # Move the dealer clockwise
        self.dealer_index = (self.dealer_index + 1) % self.num_players
        self.leader_index = (self.dealer_index + 1) % self.num_players

    def start_round(self):
        self.bids.clear()
        self.bidders.clear()  # Clear the set of bidders
        self.deck = Deck()
        self.deck.shuffle()
        self.tricks_won = {player.name: 0 for player in self.players}  # Reset tricks won
        self.trick_piles = {player.name: [] for player in self.players}
        self.trump_card = self.deck.flip_trump()
        # Reset trick state for the new round
        self.trick_cards = []
        self.lead_suit = None
        self.phase = 'bidding'

        # Determine how many cards to deal this round
        num_cards = max(self.current_cards, 1)
        self.round_number = self.total_rounds - self.current_cards + 1
        for player in self.players:
            player.hand = self.deck.deal(num_cards)
            player.bid = None  # Reset bid for the new round
        self.current_cards -= 1

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
        if self.phase != 'bidding':
            return "Bidding complete.", False

        start_index = (self.dealer_index + 1) % self.num_players
        for idx in range(self.num_players):
            current_player_index = (start_index + idx) % self.num_players
            current_player = self.players[current_player_index]

            if current_player.bid is not None:
                continue

            if current_player.name == "You":
                return "Your turn to bid.", True
            else:
                ai_bid = random.randint(0, self.round_number)
                self.receive_bid(current_player.name, ai_bid)
                time.sleep(0.5)

        if len(self.bidders) == self.num_players:
            self.phase = 'playing'
            return "All bids are in.", False

        return "Waiting for bids.", False

    def play_card(self, player_name, card):
        """Play ``card`` for ``player_name``.

        Parameters
        ----------
        player_name : str
            Name of the player.
        card : Card or str
            ``Card`` instance or string representation of the card.
        """

        player = next((p for p in self.players if p.name == player_name), None)
        if not player:
            return f"Player {player_name} not found."

        # Enforce turn order based on leader and cards already played
        expected_index = (self.leader_index + len(self.trick_cards)) % self.num_players
        expected_player = self.players[expected_index]
        if player != expected_player:
            return f"It is not {player_name}'s turn."

        # Allow card to be provided as a string (e.g. "2_of_Hearts")
        if isinstance(card, str):
            card_obj = next((c for c in player.hand if str(c) == card), None)
            if not card_obj:
                return f"{player_name} cannot play {card}."
            card = card_obj

        if card not in player.hand:
            return f"{player_name} cannot play {card}."

        playable_cards = self.get_playable_cards(player)
        if card not in playable_cards:
            return f"You must follow suit if possible."

        player.hand.remove(card)
        self.trick_cards.append((player.name, card))
        if not self.lead_suit:
            self.lead_suit = card.suit
        self.log_action(f"{player_name} played {card}")

        if len(self.trick_cards) == self.num_players:
            self.resolve_trick()

        return None

    def autoplay_until_player(self):
        """Have AI players play until it is the user's turn or the trick ends."""
        if self.phase != 'playing':
            return
        while len(self.trick_cards) < self.num_players:
            idx = (self.leader_index + len(self.trick_cards)) % self.num_players
            player = self.players[idx]
            if player.name == "You":
                break
            playable = self.get_playable_cards(player)
            if playable:
                time.sleep(0.5)
                self.play_card(player.name, playable[0])

    def get_playable_cards(self, player):
        if not self.trick_cards:
            return player.hand
        else:
            follow_suit_cards = [card for card in player.hand if card.suit == self.lead_suit]
            return follow_suit_cards if follow_suit_cards else player.hand

    def resolve_trick(self):
        """Determine the winner of the current trick and move the cards"""
        if not self.trick_cards:
            return

        winning_name, winning_card = self.trick_cards[0]
        for name, card in self.trick_cards[1:]:
            if card.suit == winning_card.suit and card.value() > winning_card.value():
                winning_name, winning_card = name, card
            elif card.suit == self.trump_card.suit and winning_card.suit != self.trump_card.suit:
                winning_name, winning_card = name, card

        self.tricks_won[winning_name] += 1
        self.trick_piles[winning_name].extend([card for _, card in self.trick_cards])
        self.trick_cards = []
        self.lead_suit = None
        self.leader_index = next((i for i,p in enumerate(self.players) if p.name == winning_name), self.leader_index)

        # If all players are out of cards, begin a new round
        if all(len(p.hand) == 0 for p in self.players):
            if self.current_cards > 0:
                self.start_round()
                # Let AI bid until it's the player's turn
                self.manage_turns()
                # After bidding, autoplay AI cards if they lead
                self.autoplay_until_player()

    def autoplay_trick(self):
        """Automatically play a trick using simple logic"""
        start = self.leader_index
        for offset in range(self.num_players):
            idx = (start + offset) % self.num_players
            player = self.players[idx]
            if player.hand:
                card = player.hand[0]
                self.play_card(player.name, card)

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

    def get_state(self):
        """Return a serializable representation of the current game state."""
        return {
            'round': self.round_number,
            'trump_card': self.get_card_image(self.trump_card) if self.trump_card else None,
            'dealer': self.players[self.dealer_index].name if self.players else None,
            'phase': self.phase,
            'players': [
                {
                    'name': p.name,
                    'cards': [self.get_card_image(c) for c in p.hand] if p.name == 'You' else [],
                    'num_cards': len(p.hand),
                    'bid': p.bid,
                    'score': p.score
                }
                for p in self.players
            ],
            'trick': [
                {'player': name, 'card': self.get_card_image(card)} for name, card in self.trick_cards
            ],
            'won_piles': {
                name: [self.get_card_image(c) for c in pile] for name, pile in self.trick_piles.items()
            }
        }
