"""The game's JSON API, independent of any web server.

``website.py`` (Flask) serves it over HTTP; in the browser build a Web
Worker running Pyodide calls ``handle`` directly (``frontend/src/worker``).
Both take the same paths, bodies and replies, so the frontend is unchanged
apart from how a request travels.

``handle(method, path, body)`` returns ``(status, data)``: 200 with the
reply, or an HTTP error status with ``{'error': message}``. ``review_step``
analyses one of the user's decisions for the post-game review; callers run
it whenever no request is waiting.
"""

import threading

from bid_game import random_situation
from bid_helper import recommend
from game import Game


class _Rejected(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


class AppApi:
    def __init__(self):
        self.game = None  # one game at a time per server (or per browser tab)
        # The bots' native simulator is not thread-safe and thinks for up to
        # a second, so anything that makes a bot think takes turns.
        self.thinking = threading.Lock()
        self.helper_requests = 0  # the newest bid-helper request; older ones give up

    def handle(self, method, path, body=None):
        body = body if isinstance(body, dict) else {}
        route = self.ROUTES.get((method.upper(), path.rstrip('/') or '/'))
        if route is None:
            return 404, {'error': f'No such request: {method} {path}'}
        try:
            return 200, route(self, body)
        except _Rejected as rejected:
            return rejected.status, {'error': rejected.message}

    def review_step(self):
        """Analyse the oldest unreviewed decision; False if there is none."""
        current = self.game
        if current is None or not current.review.pending():
            return False
        with self.thinking:
            return current.review.analyse_next()

    # Requests -------------------------------------------------------------

    def _state(self, error=None):
        if error:
            raise _Rejected(400, error)
        return self.game.get_state() if self.game else None

    def _require_game(self):
        if self.game is None:
            raise _Rejected(404, 'No game in progress.')

    def get_game(self, body):
        return self._state()

    def new_game(self, body):
        try:
            players = int(body.get('players', 4))
            hand_size = int(body.get('hand_size', 10))
        except (TypeError, ValueError):
            return self._state('Players and hand size must be numbers.')
        if not Game.MIN_PLAYERS <= players <= Game.MAX_PLAYERS:
            return self._state(f'Players must be between {Game.MIN_PLAYERS} and {Game.MAX_PLAYERS}.')
        max_hand = Game.max_hand_size(players)
        if not 1 <= hand_size <= max_hand:
            return self._state(f'Hand size must be between 1 and {max_hand} for {players} players.')
        self.game = Game(num_players=players, num_rounds=hand_size)
        self.game.start_round()
        return self._state()

    def end_game(self, body):
        self.game = None
        return self._state()

    def bid(self, body):
        self._require_game()
        try:
            value = int(body.get('bid'))
        except (TypeError, ValueError):
            return self._state('Bid must be a number.')
        return self._state(self.game.receive_bid(Game.HUMAN_NAME, value))

    def play(self, body):
        self._require_game()
        return self._state(self.game.play_card(Game.HUMAN_NAME, body.get('card')))

    def advance(self, body):
        self._require_game()
        with self.thinking:
            self.game.advance()
        return self._state()

    def advice(self, body):
        self._require_game()
        with self.thinking:
            result = self.game.advice()
        if result is None:
            return self._state('It is not your turn.')
        return result

    def review(self, body):
        """The review of the user's decisions, with whatever is analysed so far."""
        self._require_game()
        return self.game.review_report()

    def bid_helper(self, body):
        """Recommend a bid for a hand entered by hand, outside any game."""
        self.helper_requests += 1
        mine = self.helper_requests
        with self.thinking:
            if mine != self.helper_requests:
                # The page has already asked about a newer hand and ignores this answer.
                raise _Rejected(409, 'Superseded by a newer request.')
            try:
                return recommend(
                    body.get('players', 4), body.get('hand', []), body.get('trump', ''),
                    body.get('bids', []), blind=bool(body.get('blind')))
            except (TypeError, ValueError) as err:
                raise _Rejected(400, str(err))

    def bid_game_round(self, body):
        """Deal one random bidding situation for the mini game, with its answer."""
        with self.thinking:
            return random_situation()

    ROUTES = {
        ('GET', '/api/game'): get_game,
        ('POST', '/api/game'): new_game,
        ('DELETE', '/api/game'): end_game,
        ('POST', '/api/game/bid'): bid,
        ('POST', '/api/game/play'): play,
        ('POST', '/api/game/advance'): advance,
        ('GET', '/api/game/advice'): advice,
        ('GET', '/api/game/review'): review,
        ('POST', '/api/bid-helper'): bid_helper,
        ('POST', '/api/bid-game/round'): bid_game_round,
    }
