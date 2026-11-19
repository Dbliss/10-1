"""Unattended own-bid-only screen, then 250 ms validation if confirmed.

Run from the repository root: python scripts/run_bid_estimate_experiment.py
Each duel includes its own unseen-deal confirmation. Defaults are never changed.
The local read-only progress page is served on http://localhost:8781/.
"""

import html
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'runs' / 'duels'
BASELINE = 'hybrid:field=mix,inference=1,shortlist=13,touching=1,bid_sigma=0.76,bid_blend=0.25'
CANDIDATE = BASELINE + ',bid_estimate=3'
STAGES = [(50, 2026111903), (250, 2026111904)]
PAGE = RESULTS / 'own_bid_estimate_progress.html'


def progress(label, status):
    log = RESULTS / f'{label}.log'
    tail = log.read_text(encoding='utf-8', errors='replace')[-10000:] if log.exists() else ''
    PAGE.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="15"><title>Bid estimate duel</title>'
        '<style>body{font:18px system-ui;max-width:1100px;margin:40px auto;'
        'background:#0c1730;color:#e6edff}pre{white-space:pre-wrap;font-size:14px}'
        'a{color:#9dd5ff}</style><h1>Own-bid estimate duel</h1>'
        f'<p>{html.escape(status)}</p><p>{html.escape(label)}</p>'
        '<p>Only our bid shortlist uses the fitted v3 formula. Defaults unchanged. '
        'Refreshes every 15 seconds. Positive points favour the candidate.</p>'
        f'<p><a href="/{label}.log">Full log</a> | '
        f'<a href="/{label}.json">Final result (when complete)</a></p>'
        f'<pre>{html.escape(tail)}</pre></html>', encoding='utf-8')


def record(label, result):
    meta = json.loads((RESULTS / f'{label}.meta').read_text())
    main = result['main']
    confirmation = result.get('confirm')
    text = (
        f'\n\n## Completed: own-bid-only fitted estimate, {meta["time_ms"]:g} ms '
        f'(19 November 2026)\n\n'
        'Single strategy change: `bid_estimate=3` replaces only our own bid '
        'shortlist estimator with the existing fitted v3 formula. Opponent '
        'bidding, inference, network inputs, blend and card play are unchanged. '
        'This isolates the estimator from the earlier estimate=2/3 experiments.\n\n'
        f'Candidate `{CANDIDATE}` against `{BASELINE}`. '
        f'Seed {meta["seed"]}, allowance {meta["time_ms"]:g} ms, '
        '18 workers, calibrated fixed budgets (interleaved timing), '
        'max 60,000 main rounds, 10 O’Brien–Fleming looks, smallest gain '
        '2 points/game. Bowers, down/blind/up round weights, 3–8 players '
        'with tournament weights; field previous hybrid x3 / skilled / casual.\n\n'
        f'**Verdict: {result["verdict"]}**. Main {main["rounds"]:,} rounds: '
        f'{main["edge"]:+.3f} points/game, gate margin +/- '
        f'{main["cs_radius"]:.3f}; diverged {main["diverged"]:.2%}. '
        f'Mean decision ms candidate / baseline: '
        f'{main["cand_ms_per_choice"]:.2f} / {main["base_ms_per_choice"]:.2f}.\n'
    )
    if confirmation:
        text += (f'Unseen-deal confirmation {confirmation["rounds"]:,} rounds: '
                 f'{confirmation["edge"]:+.3f}, one-sided 95% margin '
                 f'{confirmation["one_sided_margin"]:.3f}; '
                 f'passed={confirmation["passed"]}.\n')
    else:
        text += 'No unseen-deal confirmation: main gate did not pass.\n'
    text += (f'Source SHA-256 `{meta["source_sha256"]}`. Model/coefficient '
             f'SHA-256 fingerprints: `{json.dumps(meta["weights"], sort_keys=True)}`.\n'
             f'Calibration: `{json.dumps(meta["calibration"], sort_keys=True)}`.\n'
             f'Results `runs/duels/{label}.{{meta,jsonl,json,log}}`. '
             'Defaults unchanged. A 50 ms result alone does not establish '
             'strength at the game’s 750 ms allowance.\n')
    with (ROOT / 'BOT_EXPERIMENTS.md').open('a', encoding='utf-8') as output:
        output.write(text)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(RESULTS), **kwargs)

        def do_GET(self):
            if self.path == '/':
                self.path = '/own_bid_estimate_progress.html'
            super().do_GET()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 8781), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    for allowance, seed in STAGES:
        label = f'{allowance}ms_own_bid_estimate_v3'
        # Never overwrite a previous run under the same label.
        if any((RESULTS / f'{label}.{suffix}').exists()
               for suffix in ('meta', 'jsonl', 'json', 'log')):
            raise SystemExit(f'{label} already exists; use a fresh label or resume manually.')
        command = [sys.executable, '-u', '-m', 'ohhell.duel', CANDIDATE,
                   '--baseline', BASELINE, '--time-ms', str(allowance),
                   '--label', label, '--seed', str(seed), '--workers', '18',
                   '--max-rounds', '60000', '--report-every', '1000']
        print('Starting:', subprocess.list2cmdline(command), flush=True)
        with (RESULTS / f'{label}.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                progress(label, f'Running at {allowance} ms; main gate and automatic confirmation.')
                time.sleep(10)
        if process.returncode:
            progress(label, f'Run failed (exit {process.returncode}); see log. Defaults unchanged.')
            print(f'{label} failed; see its log.', flush=True)
            break
        result = json.loads((RESULTS / f'{label}.json').read_text())
        record(label, result)
        print(f'{label}: {result["verdict"]}; experiment log updated.', flush=True)
        progress(label, f'Completed: {result["verdict"]}. Experiment log updated; defaults unchanged.')
        if result['verdict'] != 'better':
            break
    # Keep the local results page available after completion.
    print('Finished. Progress and results: http://localhost:8781/', flush=True)
    server_thread.join()


if __name__ == '__main__':
    main()
