"""Build native code and start one isolated, user-run search experiment."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

STAGES = ('time', 'dedupe', 'touching', 'play-evidence', 'early-stop',
          'bank', 'policy-self', 'policy-all', 'typing', 'policy-inference')


def stage_options(stage):
    baseline = 'hybrid:field=previous'
    candidate = baseline
    extra = []
    if stage == 'time':
        extra = ['--candidate-time-ms', '10']
    elif stage == 'dedupe':
        candidate += ',dedupe=1'
    elif stage == 'touching':
        candidate += ',touching=1'
    elif stage == 'play-evidence':
        candidate += ',play_evidence=1'
    elif stage == 'early-stop':
        candidate += ',stop_z=3'
    elif stage == 'bank':
        baseline += ',stop_z=3'
        candidate += ',stop_z=3,controller=bank'
        extra = ['--mean-time-ratio', '1.05']
    elif stage == 'policy-self':
        candidate += ',policy=learned_self'
    elif stage == 'policy-all':
        baseline += ',policy=learned_self'
        candidate += ',policy=learned_all'
    elif stage == 'typing':
        candidate += ',typing=1'
    elif stage == 'policy-inference':
        candidate += ',policy_inference=1'
    else:
        raise ValueError(f'Unknown stage: {stage}')
    return candidate, baseline, extra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=STAGES, nargs='?', default='time')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--port', type=int, default=8780)
    parser.add_argument('--seed', type=int, default=1302026113)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    root = Path(__file__).resolve().parents[2]
    candidate, baseline, extra = stage_options(args.stage)
    label = f"Search {args.stage} {datetime.now(timezone.utc):%Y%m%d-%H%M%S-%f}"
    subprocess.run([sys.executable, '-m', 'ohhell.native'], cwd=root, check=True)
    command = [sys.executable, '-m', 'ohhell.tournament', candidate,
               '--baseline', baseline, '--label', label, '--time-ms', '2',
               '--workers', str(args.workers), '--port', str(args.port),
               '--seed', str(args.seed), *extra]
    if args.check_only:
        command.append('--preflight-only')
    print(f'Candidate: {candidate}\nBaseline: {baseline}', flush=True)
    subprocess.run(command, cwd=root, check=True)


if __name__ == '__main__':
    main()
