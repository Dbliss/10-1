# Oh Hell

Play the card game **Oh Hell** (10 to 1) in the browser against bots. A React frontend talks to a Flask API, and the bots are in a separate Python package.

## Rules

Each round, players bid how many tricks they will take. Hands run from the opening size down to 1, with a card turned up for trumps. You must follow suit if you can.

- A made bid scores 10 per trick, and a made zero scores 10.
- A missed bid loses 10 per trick off.
- The dealer bids last and may not make the bids add up to the hand size.

Scoring and the dealer rule are set in `ohhell/rules.py` and `game.py`.

## Project layout

- `ohhell/`: the rules engine, bots, arena and training code. It knows nothing about the web app.
- `game.py`: runs one game on that engine and paces it for the UI.
- `website.py`: the Flask JSON API, which also serves the built frontend.
- `tests/`: run with `python -m pytest`.
- `frontend/`: the React app (Vite and TypeScript). Card images are in `frontend/public/cards`.

## Setup

You need Python 3.10 or later and Node.js.

```bash
pip install -r requirements.txt
cd frontend
npm install
```

## Running the game

Build the frontend, then start the server from the repository root:

```bash
cd frontend
npm run build
cd ..
python website.py
```

Open <http://localhost:5000/>.

## Working on the frontend

Run the API and the Vite dev server in two terminals to get hot reloading:

```bash
python website.py
```

```bash
cd frontend
npm run dev
```

Open <http://localhost:5173/>. Vite forwards `/api` requests to Flask.

## API

The server holds one game at a time. Every endpoint returns the full game state as JSON, or `null` when no game exists.

| Method and path | Body | Action |
| --- | --- | --- |
| `GET /api/game` | | Read the current state |
| `POST /api/game` | `{players, hand_size}` | Start a new game |
| `DELETE /api/game` | | End the game |
| `POST /api/game/bid` | `{bid}` | Place the user's bid |
| `POST /api/game/play` | `{card}` | Play a card from the user's hand, e.g. `"Queen_of_Hearts"` |
| `POST /api/game/advance` | | Run the next automatic step: a bot's turn, collecting a finished trick, or dealing the next round |

| `GET /api/game/advice` | | The bot's analysis of the user's current decision, using only what the user can see |

The `awaiting` field in the state says what has to happen next.

## Bots

Bots receive only what their seat can see. Each is named by a spec, such as `simulation:worlds=16`.

| Spec | Approach |
| --- | --- |
| `random` | Picks any legal action |
| `heuristic` | Hand-written odds and play |
| `net` | The trained models alone, with no simulation |
| `simulation` | Samples possible deals and scores each option by playing them out, with exact search near the end of a round |
| `inference` | Simulation, favouring deals that fit opponents' bids and plays |
| `hybrid` | Inference with the models as a prior. This is the bot used in the game |

Compare two bots on identical deals:

```bash
python -m ohhell.arena hybrid heuristic --games 300
```

Build the optional C core, which makes simulations about 40 times faster. It needs `gcc`, and without it the bots use the Python code:

```bash
python -m ohhell.native
```

Retrain the models on simulated tables of skilled, casual and random opponents:

```bash
python -m ohhell.learn.train --out ohhell/models/candidate
```

## Tournament

Tests one candidate bot against the current version at tables of 3 to 8 players, weighted towards 4. The other seats are drawn each set from versions that lean passive or aggressive and from skilled, casual and novice players. Every set is one preset game played from every seat on identical cards, and every simulating bot gets the same time per turn, so spending more compute can't win. Open <http://localhost:8765/> to watch.

```bash
python -m ohhell.tournament "hybrid:field=previous,react=1" --label "Reactive rollouts"
```

The label is the candidate's name, and it is shown with the next version number. Each label gets its own deals, or pick them with `--seed`.

The run is checked ten times, every 130 sets, and stops at a check where the candidate is definitely better, definitely worse, or clearly gains less than one point per game. A candidate that looks better then plays the same number of sets again on unseen deals, and only counts as better if it wins those too. Score decides the result, and the dashboard warns if win share clearly disagrees.

If it wins, add it to `VERSIONS` in `ohhell/bots/__init__.py` with its number and name. The last entry is the bot the game uses.

Results are saved in `runs/gates/`. Add `--resume` to continue an interrupted run.
