from flask import Flask, render_template, jsonify, request
from game import Game

app = Flask(__name__, static_folder='assets', static_url_path='/assets')

game = Game()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    data = request.get_json() or {}
    players = int(data.get('players', 4))
    rounds = int(data.get('rounds', 10))
    global game
    game = Game(num_players=players, num_rounds=rounds)
    game.start_round()
    game.autoplay_until_player()
    return jsonify({'status': 'started'})

@app.route('/reset', methods=['POST'])
def reset():
    global game
    game = Game()
    return jsonify({'status': 'reset'})

@app.route('/bid', methods=['POST'])
def bid():
    data = request.get_json()
    bid_value = int(data.get('bid', 0))
    game.receive_bid('You', bid_value)
    game.manage_turns()
    game.autoplay_until_player()
    return jsonify({'status': 'bid received'})

@app.route('/play', methods=['POST'])
def play():
    data = request.get_json()
    card = data.get('card')
    err = game.play_card('You', card)
    if err:
        return jsonify({'error': err}), 400
    game.autoplay_until_player()
    return jsonify({'status': 'card played'})

@app.route('/state')
def state():
    st = game.get_state()
    return jsonify(st)

if __name__ == '__main__':
    app.run(debug=True)
