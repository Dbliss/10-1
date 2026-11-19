"""Run the adapted public Oh Hell-family bots together on rotated identical deals.

This is a reference league, not the default-bot promotion gate. All entries
play the repository's bower/down-blind-up rules and standard scoring.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
import faulthandler
import hashlib
from http.server import BaseHTTPRequestHandler
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import threading
import time

from .arena import deal_round
from .bots import DEFAULT_SPEC, make_bot
from .engine import DONE, RoundState
from .external_setup import ASSETS, ROOT, acquire, build_diminishing, sha256, verify_assets
from .rules import Rules

PAGE = Path(__file__).with_name('external_league.html')
ENTRIES = {
    'v2': {'name': 'v2 Self-play bowers', 'origin': 'This repository',
           'adaptation': 'Promoted weights; ordinary hybrid search at the selected soft time limit.'},
    'campbell': {'name': 'Campbell OI neural (adapted)',
                 'origin': 'https://github.com/campbellsoup37/OhHell',
                 'adaptation': 'Original per-table OVL/IVL weights; NumPy port; bower-aware effective suits '
                               'and relative ranks, saturated original count features, table scoring for bids. '
                               'Play retains probability-of-making objective; deterministic card-order ties.'},
    'diminishing-mc': {'name': 'DiminishingBridge MC (adapted)',
                       'origin': 'https://github.com/thijssnelleman/DiminishingBridge',
                       'adaptation': 'Pinned C++ MC/random-rollout player with bower rank-grid, shared scoring, '
                                     'observed public cumulative scores, safety fixes and soft deadlines. '
                                     'Mean action values account for unequal completed samples.'},
    'diminishing-mcts': {'name': 'DiminishingBridge MCTS (adapted)',
                         'origin': 'https://github.com/thijssnelleman/DiminishingBridge',
                         'adaptation': 'Pinned visible-history MCTS player with the same rule/safety/deadline '
                                       'adaptations; upstream future-score heuristic still assumes descending rounds.'},
    'drl-rule': {'name': 'DRL-POMDP heuristic (adapted)',
                 'origin': 'https://github.com/jonas-schumacher/drl-pomdp',
                 'adaptation': 'Executes upstream RuleBasedPlayer, with bower-aware comparison tables and '
                               'nearest legal bid when the dealer restriction forbids its choice. '
                               'No released DQN checkpoint found; this is explicitly the heuristic.'},
}
DEFAULT_ROSTER = tuple(ENTRIES)
EXCLUDED = [
    {'name': 'SkatZero', 'reason': 'Excluded at user request: incompatible 32-card, three-player declarer/defender game.'},
    {'name': 'BotEuchre', 'reason': 'Excluded at user request: incompatible 24-card partnership game.'},
    {'name': 'DRL-POMDP DQN', 'reason': 'No pretrained checkpoint shipped in the pinned repository; heuristic entry only.'},
]

_WORKER_DIAGNOSTICS = None
_WORKER_FAULT_LOG = None


def initialize_worker_diagnostics(folder):
    """Keep a native fault trace and public seed context outside timed moves."""
    global _WORKER_DIAGNOSTICS, _WORKER_FAULT_LOG
    _WORKER_DIAGNOSTICS = Path(folder)
    _WORKER_DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
    _WORKER_FAULT_LOG = (_WORKER_DIAGNOSTICS / f'{os.getpid()}.fault.log').open('a')
    faulthandler.enable(_WORKER_FAULT_LOG, all_threads=True)


def update_worker_context(index, seed, **context):
    if _WORKER_DIAGNOSTICS is not None:
        (_WORKER_DIAGNOSTICS / f'{os.getpid()}.context.json').write_text(json.dumps(
            {'pid': os.getpid(), 'index': index, 'seed': seed, **context}))


def native_failure_message(folder):
    folder = Path(folder)
    failures = []
    for trace in sorted(folder.glob('*.fault.log')):
        if not trace.stat().st_size:
            continue
        path = trace.with_name(trace.name.removesuffix('.fault.log') + '.context.json')
        if path.exists():
            try:
                context = json.loads(path.read_text())
                failures.append(f"set {context['index']}, seed {context['seed']}, "
                                f"rotation {context.get('rotation', '?')}, hand {context.get('hand', '?')}")
            except (OSError, ValueError, KeyError):
                failures.append(f'worker {trace.name.removesuffix(".fault.log")} (incomplete context file)')
    detail = '; '.join(failures) if failures else 'see worker context files for in-flight deal seeds'
    return f'Native league worker terminated ({detail}). Fault logs: {folder}. Completed sets are preserved; no bot was substituted.'


def create_bot(entry, seed, time_ms):
    from .bots.external import CampbellBot, DRLRuleBot, DiminishingBot
    if entry == 'v2':
        return make_bot(f'hybrid:field=previous,time_ms={time_ms}', seed)
    if entry == 'campbell':
        return CampbellBot(seed)
    if entry == 'drl-rule':
        return DRLRuleBot(seed)
    if entry in ('diminishing-mc', 'diminishing-mcts'):
        return DiminishingBot(seed, entry.removeprefix('diminishing-'), time_ms)
    raise ValueError(f'Unknown entry {entry}')


def play_set(job):
    index, seed, roster, max_hand, time_ms = job
    update_worker_context(index, seed, status='starting')
    n = len(roster)
    from .bots.external import campbell_network, drl_module, drl_order, diminishing_library
    if 'campbell' in roster:
        campbell_network(n, 'ovl')
        campbell_network(n, 'ivl')
    if 'drl-rule' in roster:
        drl_module()
        drl_order()
    if any(r.startswith('diminishing-') for r in roster):
        diminishing_library()
    rules = Rules(n, max_hand)
    scores, rounds = [], []
    clocks = [[0.0, 0, 0.0] for _ in roster]
    adjustments = [0] * n
    saturations = [0] * n
    lineup = list(range(n))
    random.Random(seed ^ 0x5E471).shuffle(lineup)
    for rotation in range(n):
        places = [lineup[(seat + rotation) % n] for seat in range(n)]
        bots = [create_bot(roster[place], (seed * 7919 + rotation * 101 + place) & 0x7FFFFFFF,
                           time_ms) for place in places]
        rng = random.Random(seed)  # immutable deals, independent of bot random state
        dealer = rng.randrange(n)
        totals = [0] * n
        round_rows = []
        for size in rules.hand_sizes:
            update_worker_context(index, seed, status='playing', rotation=rotation, hand=size)
            hands, trump = deal_round(rng, n, max(1, size))
            state = RoundState(rules, hands, trump, dealer, blind=size == 0)
            while state.phase != DONE:
                seat = state.to_act
                obs = state.observation(seat)
                started = time.perf_counter()
                action = bots[seat].act(obs)
                spent = time.perf_counter() - started
                slot = places[seat]
                if len(obs.legal_actions()) > 1:
                    clocks[slot][0] += spent
                    clocks[slot][1] += 1
                    clocks[slot][2] = max(clocks[slot][2], spent)
                if action not in obs.legal_actions():
                    raise RuntimeError(f'{roster[slot]} returned illegal action {action} at seed {seed}')
                state.apply(action)
            points = state.points()
            row = [0] * n
            for seat, bot in enumerate(bots):
                started = time.perf_counter()
                bot.observe_round(state.observation(seat))
                clocks[places[seat]][0] += time.perf_counter() - started
                totals[seat] += points[seat]
                row[places[seat]] = points[seat]
            round_rows.append(row)
            dealer = (dealer + 1) % n
        row = [0] * n
        for seat, bot in enumerate(bots):
            slot = places[seat]
            row[slot] = totals[seat]
            adjustments[slot] += getattr(bot, 'legal_bid_adjustments', 0)
            saturations[slot] += getattr(bot, 'feature_saturations', 0)
        scores.append(row)
        rounds.append(round_rows)
    update_worker_context(index, seed, status='completed')
    return {'index': index, 'seed': seed, 'scores': scores, 'clocks': clocks,
            'bid_adjustments': adjustments, 'feature_saturations': saturations, 'rounds': rounds}


def mean_interval(values):
    mean = sum(values) / len(values) if values else None
    margin = (1.96 * math.sqrt(sum((v - mean) ** 2 for v in values) /
                              (len(values) - 1) / len(values)) if len(values) > 1 else None)
    return mean, margin


class Board:
    def __init__(self, config):
        self.config = config
        self.rows = []
        self.started = time.monotonic()
        self.finished = False
        self.error = None
        self.lock = threading.Lock()

    def add(self, row):
        with self.lock:
            if row['index'] != len(self.rows):
                raise ValueError('League sets must be contiguous and ordered')
            self.rows.append(row)

    def snapshot(self):
        with self.lock:
            rows = list(self.rows)
        roster = self.config['roster']
        n = len(roster)
        means = [[sum(g[i] for g in row['scores']) / n for i in range(n)] for row in rows]
        win_sets = [[] for _ in roster]
        for row in rows:
            for i in range(n):
                win_sets[i].append(sum((1 / g.count(max(g))) if g[i] == max(g) else 0
                                       for g in row['scores']) / n)
        standings = []
        for i, entry in enumerate(roster):
            mean, margin = mean_interval([m[i] for m in means])
            win, win_margin = mean_interval(win_sets[i])
            seconds = sum(r['clocks'][i][0] for r in rows)
            choices = sum(r['clocks'][i][1] for r in rows)
            standings.append(dict(ENTRIES[entry], id=entry, mean=mean, margin=margin,
                win_share=win, win_margin=win_margin, ms_per_choice=1000 * seconds / choices if choices else None,
                longest_ms=max((1000 * r['clocks'][i][2] for r in rows), default=0),
                bid_adjustments=sum(r['bid_adjustments'][i] for r in rows),
                feature_saturations=sum(r['feature_saturations'][i] for r in rows)))
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                edge, margin = mean_interval([m[i] - m[j] for m in means])
                pairs.append({'a': roster[i], 'b': roster[j], 'edge': edge, 'margin': margin})
        return {'label': self.config['label'], 'sets': len(rows), 'target_sets': self.config['sets'],
                'games': len(rows) * n, 'players': n, 'elapsed': time.monotonic() - self.started,
                'finished': self.finished, 'error': self.error, 'standings': standings,
                'pairs': pairs, 'rules': self.config['rules'], 'excluded': EXCLUDED,
                'time_ms': self.config['time_ms'],
                'limitations': 'Adapted players under common Oh Hell rules, not original-game rankings. '
                'Soft deadlines; actual times vary. Ordinary descriptive 95% intervals clustered '
                'by deal seed, not sequential significance or promotion evidence.'}


def serve(board, port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                data, kind = PAGE.read_bytes(), 'text/html; charset=utf-8'
            elif self.path == '/stats':
                data, kind = json.dumps(board.snapshot()).encode(), 'application/json'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    from .tournament import ExclusiveServer
    for candidate in range(port, port + 50):
        try:
            server = ExclusiveServer(('127.0.0.1', candidate), Handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            return server
        except OSError:
            continue
    raise RuntimeError('No available dashboard port')


def provenance(roster):
    from . import native
    files = sorted((ROOT / 'ohhell').rglob('*.py'))
    files += sorted((ROOT / 'ohhell').rglob('*.cpp'))
    files += sorted((ROOT / 'ohhell' / 'native').glob('*.c'))
    model_paths = [ROOT / 'ohhell' / 'models' / f'{p}.npz' for p in ('bid', 'play')]
    return {'upstream': verify_assets(),
            'source': {p.relative_to(ROOT).as_posix(): sha256(p) for p in files},
            'models': {p.relative_to(ROOT).as_posix(): sha256(p) for p in model_paths},
            'diminishing_library_sha256': sha256(build_diminishing()),
            'native_library_sha256': sha256(native.LIBRARY),
            'adaptations': {entry: ENTRIES[entry]['adaptation'] for entry in roster},
            'excluded': EXCLUDED}


def prepare():
    manifest = acquire()
    dll = build_diminishing()
    from . import native
    if not native.available:
        native.build()
        import importlib
        importlib.reload(native)
    if not native.available:
        raise RuntimeError('Current native Oh Hell core unavailable')
    # Load root and upstream weights before measured decisions.
    from .bots.external import campbell_network, drl_module, diminishing_library
    make_bot(DEFAULT_SPEC)
    for n in range(2, 9):
        campbell_network(n, 'ovl')
        campbell_network(n, 'ivl')
    drl_module()
    diminishing_library()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', help='download/verify/build only, no games')
    parser.add_argument('--check-only', action='store_true', help='one small complete rotated set, no strength claim')
    parser.add_argument('--sets', type=int, default=100)
    parser.add_argument('--hand', type=int, default=10)
    parser.add_argument('--time-ms', type=float, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=202611130)
    parser.add_argument('--port', type=int, default=8781)
    parser.add_argument('--label')
    parser.add_argument('--roster', default=','.join(DEFAULT_ROSTER))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--exit-when-done', action='store_true')
    args = parser.parse_args()
    roster = tuple(args.roster.split(','))
    if len(set(roster)) != len(roster) or not 2 <= len(roster) <= 8 or any(r not in ENTRIES for r in roster):
        parser.error('roster needs 2..8 distinct known entries')
    if (args.sets < 1 or args.workers < 1 or not math.isfinite(args.time_ms) or args.time_ms <= 0
            or not 1 <= args.hand <= min(10, 51 // len(roster))):
        parser.error('invalid sets, workers, time or opening hand size')
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    prepare()
    if args.prepare:
        print('Pinned sources/models verified; both native cores ready. No games run.')
        return
    if args.check_only:
        # A complete blind-inclusive sequence and every seat rotation, in a worker
        # process so a native crash cannot masquerade as a successful preflight.
        pool = ProcessPoolExecutor(max_workers=1)
        try:
            row = pool.submit(play_set, (0, args.seed, roster, min(3, args.hand), args.time_ms)).result(timeout=120)
        except BaseException:
            pool.terminate_workers()
            raise
        else:
            pool.shutdown()
        print(f'Completed {len(row["scores"])} rotated games, including blind 0; every action legal.')
        return
    if args.resume and not args.label:
        parser.error('--resume requires the original --label')
    label = args.label or f"External_OhHell_{datetime.now(timezone.utc):%Y%m%d-%H%M%S-%f}"
    if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in label):
        parser.error('label may contain only letters, digits, underscores and hyphens')
    config = {'label': label, 'roster': roster, 'sets': args.sets, 'seed': args.seed,
              'time_ms': args.time_ms, 'rules': {'players': len(roster), 'max_hand': args.hand,
              'round_format': 'down-blind-up', 'trumps': 'bowers', 'dealer_restriction': True,
              'scoring': {'points_per_trick': 10, 'made_zero': 10, 'penalty_per_trick': 10}},
              'provenance': provenance(roster)}
    config = json.loads(json.dumps(config))
    folder = ROOT / 'runs' / 'external_leagues'
    folder.mkdir(parents=True, exist_ok=True)
    meta, ledger, summary = [folder / f'{label}.{suffix}' for suffix in ('meta', 'jsonl', 'json')]
    if args.resume:
        if not meta.exists() or json.loads(meta.read_text()) != config:
            raise SystemExit('Resume refused: rules, roster, source, models or settings changed')
    else:
        if any(p.exists() for p in (meta, ledger, summary)):
            raise SystemExit('Run label exists; use a fresh label or --resume')
        meta.write_text(json.dumps(config, indent=2) + '\n')
    board = Board(config)
    if args.resume and ledger.exists():
        for line in ledger.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row['seed'] != args.seed + row['index']:
                    raise ValueError('Invalid saved seed')
                board.add(row)
    server = serve(board, args.port)
    print(f'Dashboard: http://localhost:{server.server_address[1]}/\nSaved run: {label}', flush=True)
    jobs = [(i, args.seed + i, roster, args.hand, args.time_ms) for i in range(len(board.rows), args.sets)]
    diagnostics = folder / f'{label}.diagnostics' / datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    try:
        with ledger.open('a' if args.resume else 'x', buffering=1) as output:
            pool = ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker_diagnostics,
                                       initargs=(str(diagnostics),))
            try:
                for row in pool.map(play_set, jobs):
                    output.write(json.dumps(row) + '\n')
                    board.add(row)
                    summary.write_text(json.dumps(board.snapshot(), indent=2) + '\n')
                    print(f'{len(board.rows)}/{args.sets} sets ({len(board.rows) * len(roster)} games)', flush=True)
            except BrokenProcessPool as error:
                pool.terminate_workers()
                raise RuntimeError(native_failure_message(diagnostics)) from error
            except BaseException:
                pool.terminate_workers()
                raise
            else:
                pool.shutdown()
        board.finished = True
    except BaseException as error:
        board.error = f'{type(error).__name__}: {error}'
        raise
    finally:
        summary.write_text(json.dumps(board.snapshot(), indent=2) + '\n')
        if args.exit_when_done or board.error:
            server.shutdown()
    snapshot = board.snapshot()
    for row in sorted(snapshot['standings'], key=lambda r: r['mean'], reverse=True):
        margin = '?' if row['margin'] is None else f'{row["margin"]:.2f}'
        print(f'{row["name"]}: {row["mean"]:.2f} +/- {margin} points, {row["win_share"]:.1%} win share, '
              f'{row["ms_per_choice"]:.2f} ms/choice')
    if not args.exit_when_done:
        print('Dashboard remains available. Ctrl+C to close.', flush=True)
        try:
            while True:
                time.sleep(30)
        except KeyboardInterrupt:
            server.shutdown()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
