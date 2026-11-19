"""Local web server for the Oh Hell game: the JSON API (app_api.py) over HTTP,
and host for the built React frontend. The GitHub Pages build runs the same
API in the browser instead (frontend/src/worker)."""

import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from app_api import AppApi

FRONTEND_DIST = Path(__file__).parent / 'frontend' / 'dist'
BID_HELPER_PAGE = Path(__file__).parent / 'bid_helper.html'

app = Flask(__name__, static_folder=None)
api = AppApi()  # The server hosts a single game at a time.
reviewer = None  # the thread that analyses the user's decisions for the review


def review_in_background():
    """Analyse the user's decisions one at a time, between the bots' turns."""
    while True:
        if not api.review_step():
            time.sleep(0.25)
            continue
        time.sleep(0)  # let a waiting request take the lock first


def start_reviewer():
    global reviewer
    if reviewer is None:
        reviewer = threading.Thread(target=review_in_background, daemon=True)
        reviewer.start()


@app.route('/api/<path:path>', methods=['GET', 'POST', 'DELETE'])
def api_request(path):
    status, data = api.handle(request.method, f'/api/{path}', request.get_json(silent=True))
    if request.method == 'POST' and path == 'game' and status == 200:
        start_reviewer()
    return jsonify(data), status


@app.get('/bid')
def bid_helper_page():
    return send_from_directory(BID_HELPER_PAGE.parent, BID_HELPER_PAGE.name)


@app.get('/bid-game')
def bid_game_page():
    """The bid game is part of the React frontend, which reads the path."""
    return frontend('index.html')


@app.get('/', defaults={'path': 'index.html'})
@app.get('/<path:path>')
def frontend(path):
    if not (FRONTEND_DIST / 'index.html').exists():
        return ('Frontend not built. Run "npm run build" in frontend/, '
                'or use the Vite dev server.'), 503
    return send_from_directory(FRONTEND_DIST, path)


if __name__ == '__main__':
    app.run(debug=True)
