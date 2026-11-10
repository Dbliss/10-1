import random

from ohhell.arena import deal_round
from ohhell.bots import DEFAULT_SPEC, make_bot
from ohhell.cards import (
    RANK_NAMES, RANK_OF, SUIT_NAMES, SUIT_OF, card_from_id, card_id, cards_in,
)
from ohhell.engine import BIDDING, DONE, NO_BID, NO_SUIT, PLAYING, RoundState
from ohhell.rules import Rules


class Game:
    """One game of Oh Hell for the web app: a human in seat 0 against bots.

    The rules live in ``ohhell.engine``. This class adds the pacing the UI needs:

    ``bidding`` -> ``playing`` <-> ``trick_complete`` -> ``round_over``
    -> ``bidding`` for the next round, or ``game_over`` after the last one.

    Nothing happens on its own. The user acts through ``receive_bid`` and
    ``play_card``, and everything else (a bot's turn, collecting a finished
    trick, dealing the next round) is one call to ``advance``, which lets the
    UI pace and animate each step.
    """

    HUMAN_NAME = "You"
    MIN_PLAYERS = 2
    MAX_PLAYERS = 8
    MAX_HAND_SIZE = 10
    BOT_SPEC = DEFAULT_SPEC
    ADVISOR_SPEC = f'{DEFAULT_SPEC},worlds=64,bid_worlds=128'

    @classmethod
    def max_hand_size(cls, num_players):
        """Largest opening hand that leaves a card to turn up as trump."""
        return min(cls.MAX_HAND_SIZE, 51 // num_players)

    def __init__(self, num_players=4, num_rounds=10, bot_spec=None, advisor_spec=None):
        """Create a game of ``num_rounds`` rounds, dealing that many cards first."""
        self.rules = Rules(num_players, num_rounds)
        self.num_players = num_players
        self.total_rounds = num_rounds
        self.names = [self.HUMAN_NAME] + [f"AI {i}" for i in range(1, num_players)]
        self.scores = [0] * num_players
        spec = bot_spec or self.BOT_SPEC
        self.bots = [None] + [
            make_bot(spec, random.getrandbits(32)) for _ in range(1, num_players)]
        self.advisor_spec = advisor_spec or self.ADVISOR_SPEC
        self.dealer_index = random.randrange(num_players)
        self.round_number = 0
        self.state = None
        self.phase = 'bidding'
        self.finished_trick = None  # (seat, card) pairs while a trick is on show
        self.history = []  # one entry per finished round, for the scoreboard

    def start_round(self):
        if self.round_number > 0:
            self.dealer_index = (self.dealer_index + 1) % self.num_players
        hand_size = self.total_rounds - self.round_number
        self.round_number += 1
        hands, trump = deal_round(random, self.num_players, hand_size)
        self.state = RoundState(self.rules, hands, trump, self.dealer_index)
        self.finished_trick = None
        self.phase = 'bidding'

    def current_player_index(self):
        """Seat of the player who must act, or None between turns."""
        return self.state.to_act if self.phase in ('bidding', 'playing') else None

    def advance(self):
        """Perform the next step that does not need input from the user.

        Does nothing when the game is waiting on the user or has finished.
        """
        if self.phase == 'trick_complete':
            self.collect_trick()
        elif self.phase == 'round_over':
            self.start_round()
        elif self.phase in ('bidding', 'playing') and self.state.to_act != 0:
            seat = self.state.to_act
            action = self.bots[seat].act(self.state.observation(seat))
            if self.phase == 'bidding':
                self._place_bid(action)
            else:
                self._play(action)

    def receive_bid(self, player_name, bid):
        """Record a bid. Returns an error message if the bid is rejected."""
        if self.phase != 'bidding':
            return "Bidding is over."
        if self.names[self.state.to_act] != player_name:
            return f"It is not {player_name}'s turn to bid."
        if not 0 <= bid <= self.state.hand_size:
            return f"Bid must be between 0 and {self.state.hand_size}."
        if bid not in self.state.legal_actions():
            return f"The dealer can't bid {bid}: the bids can't add up to the hand size."
        self._place_bid(bid)
        return None

    def _place_bid(self, bid):
        self.state.apply(bid)
        if self.state.phase == PLAYING:
            self.phase = 'playing'

    def play_card(self, player_name, card):
        """Play a card, named like ``Jack_of_Hearts``, for ``player_name``.

        Returns an error message if the play is rejected.
        """
        if self.phase != 'playing':
            return "Cards cannot be played right now."
        if self.names[self.state.to_act] != player_name:
            return f"It is not {player_name}'s turn."

        try:
            number = card_from_id(card)
        except (AttributeError, ValueError):
            number = None
        if number is None or not self.state.hands[self.state.to_act] >> number & 1:
            return f"{player_name} does not hold that card."
        if number not in self.state.legal_actions():
            return "You must follow suit if possible."

        self._play(number)
        return None

    def _play(self, card):
        self.state.apply(card)
        if self.state.trick_size == 0:
            self.finished_trick = self.state.history[-self.num_players:]
            self.phase = 'trick_complete'

    def collect_trick(self):
        """Clear the finished trick; the winner leads the next one."""
        self.finished_trick = None
        self.phase = 'playing'
        if self.state.phase == DONE:
            self.calculate_scores()
            self.phase = 'round_over' if self.round_number < self.total_rounds else 'game_over'

    def calculate_scores(self):
        """Add this round's points and record the round for the scoreboard."""
        results = []
        for seat, points in enumerate(self.state.points()):
            self.scores[seat] += points
            results.append({
                'bid': self.state.bids[seat],
                'tricks': self.state.tricks_won[seat],
                'points': points,
                'score': self.scores[seat],
            })
        self.history.append({
            'round': self.round_number,
            'hand_size': self.state.hand_size,
            'trump': self._card_json(self.state.trump_card),
            'results': results,
        })

    def advice(self):
        """The bot's analysis of the user's current decision, or None.

        Works only from what the user's own seat can see.
        """
        current = self.current_player_index()
        if current != 0:
            return None
        advisor = make_bot(self.advisor_spec, random.getrandbits(32))
        obs = self.state.observation(0)
        if obs.phase == BIDDING:
            analysis = advisor.analyse_bids(obs)
            best = max(analysis.values, key=analysis.values.get)
            return {
                'kind': 'bid',
                'recommended': best,
                'options': [
                    {
                        'bid': bid,
                        'expected_points': round(analysis.values[bid], 1),
                        'make_chance': round(analysis.distributions[bid][bid], 3),
                    }
                    for bid in sorted(analysis.values)
                ],
            }
        values = advisor.evaluate_cards(obs)
        best = max(values, key=values.get)
        return {
            'kind': 'card',
            'recommended': card_id(best),
            'options': [
                {'card': self._card_json(card), 'expected_points': round(value, 1)}
                for card, value in sorted(values.items(), key=lambda kv: -kv[1])
            ],
        }

    def get_state(self):
        """Return what the user is allowed to see, as JSON-friendly data."""
        state = self.state
        current = self.current_player_index()
        users_turn = current == 0
        trick = self._trick_on_table()

        return {
            'phase': self.phase,
            'awaiting': self._awaiting(current),
            'round': self.round_number,
            'total_rounds': self.total_rounds,
            'hand_size': state.hand_size,
            'trump': self._card_json(state.trump_card),
            'dealer': self.dealer_index,
            'current_player': current,
            'players': [
                {
                    'name': name,
                    'is_human': seat == 0,
                    'num_cards': state.hands[seat].bit_count(),
                    'bid': None if state.bids[seat] == NO_BID else state.bids[seat],
                    'tricks_won': state.tricks_won[seat],
                    'score': self.scores[seat],
                }
                for seat, name in enumerate(self.names)
            ],
            'hand': [self._card_json(c) for c in cards_in(state.hands[0])],
            'legal_bids': (
                state.legal_actions() if users_turn and self.phase == 'bidding' else []
            ),
            'playable': (
                [card_id(c) for c in state.legal_actions()]
                if users_turn and self.phase == 'playing' else []
            ),
            'trick': [
                {'seat': seat, 'card': self._card_json(card)} for seat, card in trick
            ],
            'lead_suit': SUIT_NAMES[SUIT_OF[trick[0][1]]] if trick else None,
            'trick_winner': state.leader if self.phase == 'trick_complete' else None,
            'history': self.history,
        }

    def _trick_on_table(self):
        if self.finished_trick is not None:
            return self.finished_trick
        size = self.state.trick_size
        return self.state.history[-size:] if size else []

    def _awaiting(self, current):
        """Name the next thing that has to happen, so the UI can react to it."""
        if self.phase == 'trick_complete':
            return 'collect_trick'
        if self.phase == 'round_over':
            return 'next_round'
        if self.phase == 'game_over':
            return 'nothing'
        if current != 0:
            return 'bot'
        return 'user_bid' if self.phase == 'bidding' else 'user_play'

    @staticmethod
    def _card_json(card):
        return {
            'id': card_id(card),
            'suit': SUIT_NAMES[SUIT_OF[card]],
            'rank': RANK_NAMES[RANK_OF[card]],
        }
