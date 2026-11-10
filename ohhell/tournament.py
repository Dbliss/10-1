"""Test one candidate bot against the current version in a varied lobby, and watch it live.

Each set is one preset game played once per seat on identical cards, so every
player gets every hand. The table has 3 to 8 players: the candidate, the current
version, and opponents drawn from a pool of leaning versions and human-like
players. Every simulating bot gets the same time per turn.

The result is checked at ten planned points and can stop at any of them. A win
must then be repeated on deals the run has not seen before it counts.

    python -m ohhell.tournament "hybrid:field=previous,react=1" --label "Reactive rollouts"
    python -m ohhell.tournament SPEC --baseline SPEC --time-ms 2 --port 8766

Then open http://localhost:8765/. Each result is saved in runs/gates/.
"""

import argparse
import json
import math
import multiprocessing
import os
import random
import re
import threading
import time
import zlib
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .arena import play_game
from .bots import DEFAULT_SPEC, VERSIONS, make_bot, version_title
from .rules import Rules

HERE = Path(__file__).parent
PAGE = HERE / 'tournament.html'
GATES = HERE.parent / 'runs' / 'gates'

CANDIDATE = 0  # the lobby slot of the bot being tested
BASELINE = 1  # the lobby slot it is compared with
TIME_MS = 2  # the time every simulating bot gets per turn
MAX_HAND = 10

# How often each table size comes up, in line with what the models train on.
PLAYER_SPLIT = {3: 2, 4: 6, 5: 2, 6: 1, 7: 1, 8: 1}
TABLE_CYCLE = [players for players, weight in PLAYER_SPLIT.items() for _ in range(weight)]
random.Random(0).shuffle(TABLE_CYCLE)

# (label after the version title, role, what to add to its spec)
LEANING_VERSIONS = (
    # Aggressive values are smaller because higher bids score more, so the same
    # shift moves bids further up than down. These match the passive levels'
    # average bid change: about 0.18 and 0.39 tricks.
    ('little bit aggressive', 'up1', ',confidence=0.23'),
    ('a bit aggressive', 'up2', ',confidence=0.36'),
    ('a bit passive', 'down2', ',confidence=-0.5'),
    ('little bit passive', 'down1', ',confidence=-0.25'),
)
HUMAN_LIKE = (
    ('Skilled player', 'skilled', 'skilled'),
    ('Casual player', 'casual', 'casual'),
    ('Novice player', 'novice', 'novice'),
)

LOOKS = 10  # how many planned checks the run gets
FINAL_Z = 2.087  # O'Brien-Fleming constant for ten looks at a 5% error rate
CONFIRM_Z = 1.645  # a repeat must clear zero by this much (one-sided 5%)
MIN_SETS = 50  # too few sets give an unreliable estimate of the spread


def extend(spec, extra):
    """Add ``,key=value`` settings to a spec that may have none yet."""
    return spec + extra if ':' in spec else spec + ':' + extra[1:]


def table_size(index):
    return TABLE_CYCLE[index % len(TABLE_CYCLE)]


def rules_for(players):
    return Rules(players, min(MAX_HAND, 51 // players))


def games_in(sets, first=0):
    """How many games sets ``first`` to ``first + sets - 1`` contain."""
    return sum(table_size(first + index) for index in range(sets))


def build_lobby(candidate, baseline, time_ms, label):
    """Slot 0 is the candidate, slot 1 the current version, the rest the pool."""
    limit = f',time_ms={time_ms}'
    next_number = VERSIONS[-1].number + 1
    current = version_title(baseline, 'Baseline')
    lobby = [
        (version_title(candidate, f'v{next_number} {label}'), 'candidate',
         extend(candidate, limit)),
        (current, 'current', extend(baseline, limit)),
    ]
    for lean, role, setting in LEANING_VERSIONS:
        lobby.append((f'{current}, {lean}', role, extend(extend(baseline, setting), limit)))
    return lobby + list(HUMAN_LIKE)


def planned_margin(sd, sets, max_sets):
    """Margin on a mean under the O'Brien-Fleming plan.

    Early checks demand far more evidence, so the error rate stays at 5% over
    all ten. A plain 95% margin would be wrong because the run can stop at any
    of them.
    """
    return FINAL_Z * sd * math.sqrt(max_sets) / sets


def decide(edge, sd, sets, smallest_gain, max_sets):
    """Return the verdict once one is certain, otherwise None."""
    if sets < MIN_SETS or not sd:
        return None
    margin = planned_margin(sd, sets, max_sets)
    if edge - margin > 0:
        return 'better'
    if edge + margin < 0:
        return 'worse'
    if edge + margin < smallest_gain:
        return 'no gain'
    return None


def confirmed(edge, sd, sets):
    """Whether a repeat on unseen deals also came out ahead."""
    return bool(sd) and edge - CONFIRM_Z * sd / math.sqrt(sets) > 0


class Timed:
    """Records how long a bot spends on the turns where it has a real choice."""

    def __init__(self, bot, clock):
        self.bot = bot
        self.clock = clock

    def act(self, obs):
        started = time.perf_counter()
        action = self.bot.act(obs)
        spent = time.perf_counter() - started
        if len(obs.legal_actions()) > 1:
            self.clock[0] += spent
            self.clock[1] += 1
            self.clock[2] = max(self.clock[2], spent)
        return action


def play_set(job):
    """Play one preset game once per seat.

    Returns (index, lobby slots at the table, scores per game, clocks), with
    scores and clocks in the order of the slots.
    """
    index, number, specs = job
    players = table_size(index)
    rules = rules_for(players)
    rng = random.Random(number)
    members = [CANDIDATE, BASELINE] + rng.sample(range(BASELINE + 1, len(specs)), players - 2)
    rng.shuffle(members)  # so nobody always sits behind the same bot
    games = []
    clocks = [[0.0, 0, 0.0] for _ in members]
    for rotation in range(players):
        places = [(seat + rotation) % players for seat in range(players)]
        bots = [
            Timed(make_bot(specs[members[place]],
                           (number * 7919 + rotation * 101 + members[place]) & 0x7FFFFFFF),
                  clocks[place])
            for place in places
        ]
        totals, _, _ = play_game(bots, rules, number)
        scores = [0] * players
        for seat, place in enumerate(places):
            scores[place] = totals[seat]
        games.append(scores)
    return index, members, games, clocks


def win_shares(games):
    """Share of the games each place won, splitting ties."""
    shares = [0.0] * len(games[0])
    for scores in games:
        top = max(scores)
        winners = scores.count(top)
        for place, score in enumerate(scores):
            if score == top:
                shares[place] += 1.0 / winners
    return [share / len(games) for share in shares]


def set_means(games):
    return [sum(scores[place] for scores in games) / len(games) for place in range(len(games[0]))]


class Running:
    """Mean and variance of a stream of numbers, updated in constant time."""

    __slots__ = ('n', 'mean', 'm2')

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    @property
    def sd(self):
        return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else None

    @property
    def interval(self):
        return 1.96 * self.sd / math.sqrt(self.n) if self.n > 1 else None


class Scoreboard:
    """Everything the dashboard shows. Written by the main thread, read by the server."""

    def __init__(self, args, lobby, history):
        self.args = args
        self.lobby = lobby
        self.history = history
        self.total = games_in(args.max_sets)
        self.look_every = max(1, args.max_sets // LOOKS)
        self.played = 0
        self.scores = [Running() for _ in lobby]
        self.edges = [Running() for _ in lobby]
        self.wins = [Running() for _ in lobby]
        self.win_edge = Running()
        self.tables = {players: Running() for players in PLAYER_SPLIT}
        self.best = [None] * len(lobby)
        self.clocks = [[0.0, 0, 0.0] for _ in lobby]
        self.decision = None  # what the planned checks concluded
        self.verdict = None  # the final word, once the run is over
        self.repeat = Running()
        self.repeat_target = 0
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.finished_at = None
        self.window = deque([(self.started, 0)], maxlen=240)

    @property
    def sets(self):
        return self.edges[CANDIDATE].n

    def add_set(self, members, games, clocks):
        """Count a set from the main run. Returns a decision if this check reached one."""
        with self.lock:
            self.played += len(games)
            place = members.index(BASELINE)
            means, shares = set_means(games), win_shares(games)
            for at, slot in enumerate(members):
                for scores in games:
                    self.scores[slot].add(scores[at])
                    if self.best[slot] is None or scores[at] > self.best[slot]:
                        self.best[slot] = scores[at]
                self.edges[slot].add(means[at] - means[place])
                self.wins[slot].add(shares[at])
                self.clocks[slot][0] += clocks[at][0]
                self.clocks[slot][1] += clocks[at][1]
                self.clocks[slot][2] = max(self.clocks[slot][2], clocks[at][2])
            new = members.index(CANDIDATE)
            self.win_edge.add(shares[new] - shares[place])
            self.tables[len(members)].add(means[new] - means[place])
            if self.sets % self.look_every == 0:
                edge = self.edges[CANDIDATE]
                self.decision = decide(edge.mean, edge.sd, edge.n,
                                       self.args.smallest_gain, self.args.max_sets)
            return self.decision

    def start_repeat(self):
        """Make the winner play as many sets again, on deals it hasn't seen."""
        with self.lock:
            self.repeat_target = self.sets
            self.total += games_in(self.repeat_target, self.args.max_sets)
            return range(self.args.max_sets, self.args.max_sets + self.repeat_target)

    def add_repeat(self, members, games, clocks):
        with self.lock:
            self.played += len(games)
            means = set_means(games)
            self.repeat.add(means[members.index(CANDIDATE)] - means[members.index(BASELINE)])

    def mark_progress(self):
        with self.lock:
            self.window.append((time.monotonic(), self.played))

    def finish(self):
        with self.lock:
            if self.repeat_target:
                passed = confirmed(self.repeat.mean, self.repeat.sd, self.repeat.n)
                self.verdict = 'better' if passed else 'unconfirmed'
            else:
                self.verdict = self.decision or 'inconclusive'
            self.finished_at = time.monotonic()

    def _margin(self, slot):
        edge = self.edges[slot]
        if slot == CANDIDATE and edge.n > 1 and edge.sd:
            return planned_margin(edge.sd, edge.n, self.args.max_sets)
        return edge.interval

    def _disagree(self):
        """True when score and winning clearly point in opposite directions."""
        edge, wins = self.edges[CANDIDATE], self.win_edge
        return bool(wins.n > 1 and abs(wins.mean) > wins.interval and edge.mean * wins.mean < 0)

    def snapshot(self):
        with self.lock:
            now = self.finished_at or time.monotonic()
            then, count = self.window[0]
            done = self.played
            rate = (done - count) / (now - then) if now > then and done > count else None
            remaining = self.total - done
            sets = self.sets
            wins = self.win_edge
            return {
                'label': self.args.label,
                'seed': self.args.seed,
                'games': done,
                'sets': sets,
                'max_sets': self.args.max_sets,
                'next_check': min(self.args.max_sets,
                                  (sets // self.look_every + 1) * self.look_every),
                'total': self.total,
                'elapsed': now - self.started,
                'rate': rate,
                'eta': remaining / rate if rate and remaining else (0 if not remaining else None),
                'finished': self.finished_at is not None,
                'verdict': self.verdict,
                'stage': 'repeat' if self.repeat_target else 'main',
                'repeat': {
                    'sets': self.repeat.n,
                    'target': self.repeat_target,
                    'edge': self.repeat.mean if self.repeat.n else None,
                    'margin': self.repeat.interval,
                },
                'smallest_gain': self.args.smallest_gain,
                'time_ms': self.args.time_ms,
                'round_format': 'down-blind-up',
                'baseline': BASELINE,
                'history': self.history,
                'win_edge': wins.mean if wins.n else None,
                'win_edge_interval': wins.interval,
                'disagree': self._disagree(),
                'tables': [
                    {
                        'players': players,
                        'cards': rules_for(players).max_hand,
                        'sets': self.tables[players].n,
                        'edge': self.tables[players].mean if self.tables[players].n else None,
                        'margin': self.tables[players].interval,
                    }
                    for players in sorted(self.tables)
                ],
                'players': [
                    {
                        'name': name,
                        'role': role,
                        'spec': spec,
                        'mean': self.scores[i].mean if self.scores[i].n else None,
                        'interval': self.scores[i].interval,
                        'sd': self.scores[i].sd,
                        'max': self.best[i],
                        'edge': self.edges[i].mean if self.edges[i].n else None,
                        'edge_interval': self._margin(i),
                        'win_share': self.wins[i].mean if self.wins[i].n else None,
                        'ms_per_turn': (1000.0 * self.clocks[i][0] / self.clocks[i][1]
                                        if self.clocks[i][1] else None),
                        'longest_ms': 1000.0 * self.clocks[i][2],
                    }
                    for i, (name, role, spec) in enumerate(self.lobby)
                ],
            }


def serve(board, port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self._send(PAGE.read_bytes(), 'text/html; charset=utf-8')
            elif self.path == '/stats':
                self._send(json.dumps(board.snapshot()).encode(), 'application/json')
            else:
                self.send_error(404)

        def _send(self, body, kind):
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def load_previous(results):
    """Read saved sets as a dict from set index to the saved record."""
    saved = {}
    if results.exists():
        with results.open() as lines:
            for line in lines:
                record = json.loads(line)
                if len(record) != 4:
                    raise SystemExit(f'{results.name} was saved by an older tournament format '
                                     'and cannot be resumed. Use a new --label.')
                saved[record[0]] = record
    return saved


def load_history():
    """Earlier verdicts, oldest first, for the dashboard."""
    entries = []
    for path in sorted(GATES.glob('*.json'), key=lambda p: p.stat().st_mtime):
        entry = json.loads(path.read_text())
        new = entry['players'][0]
        entries.append({
            'label': entry['label'], 'verdict': entry['verdict'], 'sets': entry['sets'],
            'edge': new['edge'], 'margin': new['edge_interval'],
        })
    return entries


def check_run_config(path, args, specs):
    """Never mix saved sets from different rules, weights, or bot settings."""
    import hashlib
    from .learn.models import MODEL_DIR

    weights = {}
    for spec in specs:
        name, _, settings = spec.partition(':')
        if name not in ('net', 'hybrid'):
            continue
        options = dict(pair.split('=', 1) for pair in settings.split(',') if '=' in pair)
        folder = Path(options.get('models', MODEL_DIR))
        for filename in ('bid.npz', 'play.npz'):
            model = folder / filename
            weights[str(model.resolve())] = hashlib.sha256(model.read_bytes()).hexdigest()
    config = {
        'round_format': 'down-blind-up', 'specs': specs, 'seed': args.seed,
        'max_sets': args.max_sets, 'smallest_gain': args.smallest_gain,
        'time_ms': args.time_ms, 'player_split': list(PLAYER_SPLIT.items()),
        'max_hand': MAX_HAND, 'weights': weights,
    }
    # Normalize tuples so the saved JSON compares identically on resume.
    config = json.loads(json.dumps(config))
    if args.resume:
        if not path.exists() or json.loads(path.read_text()) != config:
            raise SystemExit('Saved run uses different rules, models or settings. '
                             'Start a new run with a new --label and without --resume.')
    else:
        path.write_text(json.dumps(config, indent=2))


def play_stage(indices, add, saved, specs, board, output, args):
    """Feed sets to ``add`` in order, stopping as soon as it returns a verdict."""
    fresh = []
    for index in indices:
        if index in saved:
            if add(*saved[index][1:]):
                return
        else:
            fresh.append((index, args.seed + index, specs))
    if not fresh:
        return
    last_mark = 0.0
    with multiprocessing.Pool(args.workers) as pool:
        for record in pool.imap(play_set, fresh):  # in order, so the checks are reproducible
            output.write(json.dumps(list(record)) + '\n')
            stop = add(*record[1:])
            now = time.monotonic()
            if now - last_mark >= 0.25:
                last_mark = now
                board.mark_progress()
            if stop:
                return


def run(args):
    # Each worker only does small matrix sums, so one maths thread each is fastest.
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    GATES.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r'[^\w.-]+', '_', args.label)
    results = GATES / f'{slug}.jsonl'
    summary = GATES / f'{slug}.json'

    lobby = build_lobby(args.candidate, args.baseline, args.time_ms, args.label)
    specs = [spec for _, _, spec in lobby]
    check_run_config(GATES / f'{slug}.meta', args, specs)
    board = Scoreboard(args, lobby, load_history())
    saved = load_previous(results) if args.resume else {}
    server = serve(board, args.port)
    print(f'Dashboard: http://localhost:{args.port}/', flush=True)

    with results.open('a' if args.resume else 'w', buffering=1) as output:
        play_stage(range(args.max_sets), board.add_set, saved, specs, board, output, args)
        if board.decision == 'better':
            play_stage(board.start_repeat(), board.add_repeat, saved, specs, board, output, args)

    board.mark_progress()
    board.finish()
    snapshot = board.snapshot()
    summary.write_text(json.dumps(snapshot, indent=2))
    new, repeat = snapshot['players'][CANDIDATE], snapshot['repeat']
    print(f"{new['name']}: {snapshot['verdict'].upper()}  edge {new['edge']:+.2f} "
          f"+/- {new['edge_interval']:.2f} points after {snapshot['sets']} sets "
          f"({snapshot['games']} games, {snapshot['elapsed']:.0f}s)", flush=True)
    if repeat['sets']:
        print(f"  repeat on unseen deals: {repeat['edge']:+.2f} +/- {repeat['margin']:.2f} "
              f"over {repeat['sets']} sets", flush=True)
    if snapshot['win_edge'] is not None:
        print(f"  win share against {lobby[BASELINE][0]}: {snapshot['win_edge']:+.3f} "
              f"+/- {snapshot['win_edge_interval']:.3f}", flush=True)
    for table in snapshot['tables']:
        if table['edge'] is not None:
            print(f"  {table['players']} players: {table['edge']:+.2f} over {table['sets']} sets",
                  flush=True)
    for player in snapshot['players']:
        if player['mean'] is not None:
            print(f"  {player['name']:<36} avg {player['mean']:7.2f}   "
                  f"{player['ms_per_turn']:.2f} ms/turn   longest {player['longest_ms']:.1f} ms",
                  flush=True)

    if not args.exit_when_done:
        print('Dashboard is still running. Press Ctrl+C to stop.', flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    server.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('candidate', nargs='?',
                        default=f"{DEFAULT_SPEC},models={HERE / 'models' / 'candidate'}")
    parser.add_argument('--baseline', default=DEFAULT_SPEC)
    parser.add_argument('--label', default=None,
                        help='the name of the candidate, used for its version and saved result')
    parser.add_argument('--time-ms', type=float, default=TIME_MS, help='time per turn for every bot')
    parser.add_argument('--smallest-gain', type=float, default=1.0,
                        help='points per game below which a change is not worth having')
    parser.add_argument('--max-sets', type=int, default=1300,
                        help='multiples of 130 keep every check balanced across table sizes')
    parser.add_argument('--seed', type=int, default=None,
                        help='first preset game; by default each label gets its own deals')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--resume', action='store_true', help='keep results from the last run')
    parser.add_argument('--exit-when-done', action='store_true')
    args = parser.parse_args()
    args.label = args.label or re.sub(r'[^\w.-]+', '_', args.candidate)[:60]
    if args.seed is None:
        args.seed = zlib.crc32(args.label.encode()) & 0x3FFFFFFF
    run(args)


if __name__ == '__main__':
    main()
