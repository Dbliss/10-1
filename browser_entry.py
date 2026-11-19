"""Entry point for the browser build: the Web Worker (frontend/src/worker/engine.ts)
imports this module in Pyodide and calls ``call`` once per API request.

Replies are JSON text encoded as Flask's ``jsonify`` does (sorted keys,
integer keys as strings), so the frontend sees the same data either way.
"""

import json
import traceback

from app_api import AppApi

api = AppApi()


def call(method, path, body_json=''):
    """One request; returns JSON text of ``[status, data]``."""
    try:
        status, data = api.handle(method, path, json.loads(body_json) if body_json else None)
    except Exception:  # what Flask would turn into a 500 page
        traceback.print_exc()
        status, data = 500, {'error': 'The game engine hit an internal error.'}
    return json.dumps([status, data], sort_keys=True)


def review_step():
    """Analyse one decision for the post-game review; False when none is waiting."""
    try:
        return bool(api.review_step())
    except Exception:
        traceback.print_exc()
        return False
