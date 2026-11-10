"""JSON API for the Oh Hell game, and host for the built React frontend."""

from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

from game import Game

FRONTEND_DIST = Path(__file__).parent / 'frontend' / 'dist'

app = Flask(__name__, static_folder=None)

game = None  # The server hosts a single game at a time.


def respond(error=None):
    """Return the current game state, or a 400 when an action was rejected."""
    if error:
        return jsonify({'error': error}), 400
    return jsonify(game.get_state() if game else None)


def require_game():
    if game is None:
        abort(404, description='No game in progress.')


@app.errorhandler(404)
def not_found(err):
    if request.path.startswith('/api/'):
        return jsonify({'error': err.description}), 404
    return err


@app.get('/api/game')
def get_game():
    return respond()


@app.post('/api/game')
def new_game():
    global game
    data = request.get_json(silent=True) or {}
    try:
        players = int(data.get('players', 4))
        hand_size = int(data.get('hand_size', 10))
    except (TypeError, ValueError):
        return respond('Players and hand size must be numbers.')

    if not Game.MIN_PLAYERS <= players <= Game.MAX_PLAYERS:
        return respond(f'Players must be between {Game.MIN_PLAYERS} and {Game.MAX_PLAYERS}.')
    max_hand = Game.max_hand_size(players)
    if not 1 <= hand_size <= max_hand:
        return respond(f'Hand size must be between 1 and {max_hand} for {players} players.')

    game = Game(num_players=players, num_rounds=hand_size)
    game.start_round()
    return respond()


@app.delete('/api/game')
def end_game():
    global game
    game = None
    return respond()


@app.post('/api/game/bid')
def bid():
    require_game()
    data = request.get_json(silent=True) or {}
    try:
        value = int(data.get('bid'))
    except (TypeError, ValueError):
        return respond('Bid must be a number.')
    return respond(game.receive_bid(Game.HUMAN_NAME, value))


@app.post('/api/game/play')
def play():
    require_game()
    data = request.get_json(silent=True) or {}
    return respond(game.play_card(Game.HUMAN_NAME, data.get('card')))


@app.post('/api/game/advance')
def advance():
    require_game()
    game.advance()
    return respond()


@app.get('/api/game/advice')
def advice():
    require_game()
    result = game.advice()
    if result is None:
        return respond('It is not your turn.')
    return jsonify(result)


@app.get('/', defaults={'path': 'index.html'})
@app.get('/<path:path>')
def frontend(path):
    if not (FRONTEND_DIST / 'index.html').exists():
        return ('Frontend not built. Run "npm run build" in frontend/, '
                'or use the Vite dev server.'), 503
    return send_from_directory(FRONTEND_DIST, path)


if __name__ == '__main__':
    app.run(debug=True)
