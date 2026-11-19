# Oh Hell

Play the card game **Oh Hell** (10 to 1) in the browser against bots. A React frontend talks to a Flask API, and the bots are in a separate Python package.

## Rules

Each round, players bid how many tricks they will take. Hands run from the opening size down to 1, then a blind round labelled 0, then back up from 1 to the opening size. Starting at 10 gives 21 rounds. Scores carry through the whole game. In the blind round everyone gets one card, all hands stay hidden until played, and bids are 0 or 1. A card is turned up for trumps each round. You must follow suit if you can.

Trumps rank with bowers. The Jack of trumps (right bower) is the highest card, then the other Jack of the same colour (left bower), then Ace, King, Queen, 10 down to 2. The left bower counts as a trump in every way: it follows a trump lead, it cannot be used to follow its printed suit, and leading it leads trumps. Side suits rank Ace high as usual.

- A made bid scores 10 per trick, and a made zero scores 10.
- A missed bid loses 10 per trick off.
- The dealer bids last and may not make the bids add up to the hand size.

Scoring and the dealer rule are set in `ohhell/rules.py` and `game.py`.

## Project layout

- `ohhell/`: the rules engine, bots, arena and training code. It knows nothing about the web app.
- `game.py`: runs one game on that engine and paces it for the UI.
- `app_api.py`: the JSON API itself, with no web server. `website.py` serves it
  with Flask (and serves the built frontend); the browser build calls it
  through `browser_entry.py`.
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

## Browser build (GitHub Pages)

The site at <https://dbliss.github.io/10-1/> needs no server: a Web Worker
(`frontend/src/worker/engine.ts`) runs the project's Python in
[Pyodide](https://pyodide.org) 314.0.8 and the C core compiled to
WebAssembly. Pyodide and NumPy load from the jsDelivr CDN; the game code,
models and core come from one zip built by `browser_build.py`. The first
visit downloads about 15 MB and the engine starts in a few seconds. Bots keep
their thinking time and get about 80% of the deals the native core gets.

Building it needs [Emscripten](https://emscripten.org) **5.0.3**, the version
Pyodide 314.0.8 is built with (change both together):

```bash
python browser_build.py --emcc path/to/emcc   # frontend/public/py/app-HASH.zip
cd frontend
npm run build:pages                            # dist/, served from /10-1/
npm run preview:pages                          # http://localhost:4173/10-1/
```

`.github/workflows/pages.yml` does the same on every push to `main` and
publishes `frontend/dist` (one-time setup: Settings > Pages > Source:
GitHub Actions). The standalone `/bid` helper page is not part of the
browser build. Each browser tab plays its own game; nothing is sent to a
server.

## Bid helper

For a real game, open <http://localhost:5000/bid> while `python website.py` is
running. Choose the number of players, the trump suit (and the turned-up card if
you know it), tap the cards in your hand, and enter the bids made before yours.
It shows the recommended bid and, for each likely bid, the chance of taking
exactly that many tricks and the average points. It uses the same advisor as the
in-game advice, which thinks for about a second per answer (`ADVISOR_TIME_MS` in
`game.py`; the AI opponents get `BOT_TIME_MS` per turn). The page is plain HTML, so it needs no frontend build.

## Bid game

<http://localhost:5000/bid-game> is a ten-hand bidding quiz. Each hand is a random
deal: 2 to 8 players, 1 to 10 cards (sometimes the blind round) and a random seat
in the bidding order, with the seats before you already bid by the default bot.
After you bid it shows the chance of making each bid, its expected points and the
spread of tricks taken. A hand scores the expected points of your bid. In
two-player mode both players bid the same hands on one screen, one looking away
while the other bids. The page is part of the React frontend
(`frontend/src/bidgame/`), so run `npm run build` after changing it; the deals
come from `bid_game.py`.

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
| `POST /api/bid-helper` | `{players, hand, trump, bids, blind}` | Bid advice for an entered hand, outside any game. `trump` is a card id or a suit name; `bids` are the bids before yours, starting left of the dealer |
| `POST /api/bid-game/round` | | Deal one random bidding situation for the bid game, with the analysis of every legal bid |

The `awaiting` field in the state says what has to happen next.

## Bots

Read [the bot experiment history](BOT_EXPERIMENTS.md) before proposing another
change. The default bot is **v3**, `hybrid:field=previous,inference=0,time_ms=2`:
v2 (v1's hybrid search with networks retrained under the bowers rules and
improved by self-play, `python -m ohhell.learn.selfplay`) without reading
opponents. Reading weighs each sampled deal by how well it explains the bids
and plays so far; at 2 ms a turn it took most of the time, and six times as
many unweighted deals scored better (+5.2 points a game, confirmed on unseen
deals). At 10 ms reading was ahead again, so the game's opponents, the
advisor and the bid game, which think much longer, keep reading
(`Game.LONG_THINK`). v2's networks are the only weights kept, in
`ohhell/models/bid.npz` and `ohhell/models/play.npz`; any bot spec without
`models=` uses them.

Bots receive only what their seat can see. Each is named by a spec, such as `simulation:worlds=16`.

| Spec | Approach |
| --- | --- |
| `random` | Picks any legal action |
| `heuristic` | Hand-written odds and play |
| `net` | The trained models alone, with no simulation |
| `simulation` | Samples possible deals and scores each option by playing them out, with exact search near the end of a round |
| `inference` | Simulation, favouring deals that fit opponents' bids and plays |
| `hybrid` | Inference with the models as a prior. This is the bot used in the game |
| `adaptive` | Hybrid bidding with card search over every legal move; successive halving focuses simulations on contenders and the model prior fades with actual samples |
| `evidence` | Hybrid with the bid model weighted against the actual completed simulation count |
| `planning` | Hybrid bidding with a shared tree over visible card histories; explores future choices instead of evaluating only fixed-policy continuations |

Compare two bots on identical deals:

```bash
python -m ohhell.arena hybrid heuristic --games 300
```

Build the optional C core, which makes simulations about 40 times faster. It needs `gcc`, and without it the bots use the Python code:

```bash
python -m ohhell.native
```

### Quick trick estimate

Simulations use a fast rule-of-thumb trick estimate for three jobs: choosing
which of our bids to simulate, imagining the bids of players still to bid, and
judging (with inference) how well a sampled deal explains the bids already
made. The original formula (`card_win_probs` in `ohhell/bots/heuristic.py`)
underestimates large hands, by about a trick per player at 10 cards. It stays
unchanged because the trained networks take it as an input.

`ohhell/estimate.py` is a calibrated replacement, chosen with `estimate=2` on
any simulation bot (the C core has a matching version). It counts exactly how
likely higher cards are to fall on your earlier ones, how likely opponents are
to run out of a suit or trump it, and where you sit in the first trick. A
logistic model fitted to play-outs turns those features into each card's win
chance. Trumps with nothing higher unseen are certain tricks. Refit and score it
against the original on unseen deals with:

```bash
python -m ohhell.learn.calibrate_estimate --train 100000 --test 30000 --write
```

The bid helper shows it as a note but recommends bids with the default bot. It
is not the bot's default: its arena smoke check against v2 was negative (see [the experiment log](BOT_EXPERIMENTS.md)).

For separate bidding/card-play case banks, cached outcome labels, component
reports and training-data exports, see [BOT_CASES.md](BOT_CASES.md).

An experimental floating point NNUE card evaluator is documented in
[NNUE_EXPERIMENT.md](NNUE_EXPERIMENT.md); no playing-strength improvement was
established and its trained checkpoint has been removed, so retrain one before
using the `nnue` bot.

Retrain the models on simulated tables of skilled, casual and random opponents:

```bash
python -m ohhell.learn.train --out ohhell/models/candidate
```

Compare checkpoint changes one at a time. `models=FOLDER` selects both models;
`bid_models=FOLDER` or `play_models=FOLDER` overrides just that model, keeping the
other model from `models` (or the current default). These options work with
`hybrid` and `net`. For example, test newly trained bidding weights with the
current card-play weights:

```powershell
.venv\Scripts\python.exe -m ohhell.tournament "hybrid:field=previous,bid_models=ohhell/models/candidate" --label "Candidate bids full game" --workers 6 --port 8768
```

Open <http://localhost:8768/>. This uses the full down/blind/up game and the
current v2 baseline, including belief inference. It does not promote the
candidate automatically. Saved runs fingerprint both selected weight files;
resuming after either changes is rejected.

The larger card-search candidate is `adaptive:field=previous`. Unlike `hybrid`,
it starts with every legal card instead of the model's top four. It samples
possible deals in small batches, successively halves the field, and spends the
remaining turn comparing the finalists. Model estimates count as
`prior_worlds` samples, so increasing evidence can overturn the initial ranking.
Bidding and exact endgames use the current hybrid implementation. This is an
experimental candidate. Its completed tournament was inconclusive; a strength
improvement has not been established. The command below is retained for reference.

```powershell
.venv\Scripts\python.exe -m ohhell.tournament "adaptive:field=previous" --label "Adaptive all-card search" --workers 6 --port 8769
```

Open <http://localhost:8769/>. Every simulating bot gets the same 2 ms turn
allowance. `batch_worlds` (default 4) controls work between deadline checks;
a started batch completes before the next check, so the allowance is soft.
Without a time limit, `worlds` sets the total card/deal budget to
`worlds * number of legal cards`, redistributed across the elimination stages.

## Duels at long thinking times

The game's opponents think for 750 ms a turn, where a tournament (every seat
timed, every rotation) takes many hours. `ohhell.duel` answers "is this
candidate better than the game's bot at this time?" much faster:

```powershell
.venv\Scripts\python.exe -m ohhell.duel "hybrid:field=previous,inference=1,endgame_cards=16" --time-ms 250 --label "Endgame 16 at 250 ms"
```

- Each sample is one round, dealt at random from what a game contains (table
  sizes weighted as in the tournament, all round labels including the blind
  round, random dealer and seat). It is played twice on the same cards, once
  with the baseline (default: the game's bot, v2 search with reading) in the
  test seat and once with the candidate. The other seats are the same
  opponents with the same seeds on fixed search budgets, so the only
  difference is the test seat's choices.
- Only the test seat thinks long, on fixed deal budgets: the game bot gets
  its idle-machine deals per millisecond (`REFERENCE_RATES` in
  `ohhell/duel.py`) times `--time-ms`, and any other version that scaled by
  its speed relative to the game bot, measured with both timed together so
  machine load cancels. Exactly repeatable, so two identical bots score
  exactly the same and every difference comes from a changed decision.
  Re-measure `REFERENCE_RATES` on a different machine.
- Differences are scaled to points per game. The run is checked at ten
  planned looks with the tournament's O'Brien-Fleming margins (5% error
  overall) and stops at better, worse or no gain (`--smallest-gain`, 2 points
  a game); a "better" is repeated on unseen deals and must clear zero
  one-sided at 95%.
- Results are in `runs/duels/LABEL.{meta,jsonl,json}`; `--resume` continues.
  Bots that remember earlier rounds (`typing=1`) need whole games, so use
  the tournament for those.

Experimental `bid_estimate=3` uses the existing fitted v3 formula only for
our own bid shortlist. Unlike `estimate=3`, it leaves imagined opponents,
inference and network inputs unchanged. Default `bid_estimate=0` preserves
existing behaviour. Run `.venv\Scripts\python.exe scripts/run_bid_estimate_experiment.py`
for an unattended 50 ms screen, followed by 250 ms only if confirmed.
Progress: <http://localhost:8781/>. Every completed duel is logged in
`BOT_EXPERIMENTS.md`; the runner never changes the default bot.

At 250 ms on 18 workers a duel runs about 6 rounds a second; a full plan of
60,000 rounds takes under three hours and resolves about +/-2.5 points a
game, against roughly 8 hours for a 1,300-set tournament at 50 ms.

## Tournament

See [the strategy audit](BOT_STRATEGY_ANALYSIS.md) for the assessment of bidding,
table pressure, bailout cards and information gathering, plus remaining limits.

The larger strategy candidate is `planning:field=previous`. It replaces hybrid
card search with a tree shared across sampled hidden deals. A path is the
sequence of publicly played cards, so identical visible histories share our
future decisions even when the sampled opponent hands differ. Opponents still
follow the existing population policy. Each simulation expands one untried
choice and finishes with the rule-based rollout; revisited histories let the
tree improve later choices. This can discover plans the fixed rollout misses.

Every legal root card gets a chance. The play model orders initial exploration
only; model scores are never mixed into the final result. Search balances
observed score and exploration (`exploration=30`, in points); the most visited
root move wins, with mean score breaking ties. Bidding, exact endgames,
model checkpoints, blind rounds and belief inference are unchanged.
This is one card-search replacement, with no evidence-weighted bidding or
candidate-checkpoint change added.

```powershell
.venv\Scripts\python.exe -m ohhell.tournament "planning:field=previous" --label "Planning information v1" --workers 6 --port 8770
```

Open <http://localhost:8770/>. Keep long comparisons sequential so they do not
compete for CPU. The current default remains v1 until main and confirmation
gates both pass. A full strength evaluation of this candidate is still needed.

The native core has been built locally. After a fresh checkout or C-source
edit, rebuild it with `.venv\Scripts\python.exe -m ohhell.native`. Windows
builds use source-specific DLL filenames so an existing run can keep its
loaded core. Without the core the candidate has a Python fallback, but at
2 ms it can explore substantially less. `worlds` without a time limit budgets
`worlds * legal_cards` rollouts; timed search uses the shared turn allowance.
`tree_batch=8` controls deals per sampling batch, with time checked after each
rollout. Search stores at most 32,768 nodes per turn; at the limit, continuations
use the rollout policy. `last_search` exposes visits, totals, completed rollouts,
node count and visits to later choices for local inspection. Finite sampling,
the fallback policy and assumed opponent behavior remain approximations.

The bidding candidate `evidence:field=previous` gives the bid model the weight
of 64 virtual deals (`bid_prior_worlds=64`). The model receives more weight
when few deals finish and less when many finish. Both the native and Python
simulators report the actual completed deal count. The bid shortlist, card
play and blind bidding use the existing hybrid behavior. This is experimental; use the
tournament to establish whether it improves scores.

```powershell
.venv\Scripts\python.exe -m ohhell.tournament "evidence:field=previous" --label "Evidence weighted bids" --workers 6 --port 8768
```

Open <http://localhost:8768/>. This isolates the bidding blend change using
the current default weights. It does not promote the candidate or overwrite
a checkpoint.

Tests one candidate bot against the current version at tables of 3 to 8 players, weighted towards 4. The other seats are drawn each set from versions that lean passive or aggressive and from skilled, casual and novice players. Every set is one preset game played from every seat on identical cards, and every simulating bot gets the same time per turn, so spending more compute can't win. Open <http://localhost:8765/> to watch.

```bash
python -m ohhell.tournament "hybrid:field=previous,react=1" --label "Reactive rollouts"
```

The label is the candidate's name, and it is shown with the next version number. Each label gets its own deals, or pick them with `--seed`.

The run is checked ten times, every 130 sets, and stops at a check where the candidate is definitely better, definitely worse, or clearly gains less than one point per game. A candidate that looks better then plays the same number of sets again on unseen deals, and only counts as better if it wins those too. Score decides the result, and the dashboard warns if win share clearly disagrees.

If it wins, add it to `VERSIONS` in `ohhell/bots/__init__.py` with its number and name. The last entry is the bot the game uses.

Results are saved in `runs/gates/`. Add `--resume` to continue an interrupted run.
