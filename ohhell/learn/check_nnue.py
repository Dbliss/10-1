"""Saved NNUE numerical/timing diagnostics and three full-game smoke checks.

This is not an arena comparison or a tournament. Microtimings are local only.
"""

import argparse
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import random
import time

import numpy as np

from ..arena import play_game
from ..bots import DEFAULT_SPEC, make_bot
from ..rules import Rules
from .cases import checked_metadata, file_hash, provenance, read_rows, restore, write_json
from .models import VALUE_SCALE, load_models
from .nnue import Evaluator, feature_matrix, load_network


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--play-models', default=None)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    from pathlib import Path
    if Path(args.out).exists():
        parser.error('choose a fresh output file')
    checked_metadata(args.bank)
    selected = []
    for case in read_rows(args.bank):
        if case['part'] == 'play' and not case['blind'] and case['split'] == 'test':
            state = restore(case)
            if len(state.legal_actions()) > 1:
                selected.append(case)
    chosen = random.Random(20261110).sample(selected, min(256, len(selected)))
    if not chosen:
        parser.error('bank has no visible test choices')
    observations = []
    for case in chosen:
        state = restore(case)
        observations.append(state.observation(state.to_act))
    network = load_network(args.play_models)
    evaluator = Evaluator(network)
    maximum_error = 0.0
    for obs in observations:
        cards = obs.legal_actions()
        actual = np.asarray(evaluator.card_values(obs, obs.seat, obs.hand, cards))
        dense = network(feature_matrix(obs, obs.seat, obs.hand, cards))[:, 0] * VALUE_SCALE
        maximum_error = max(maximum_error, float(np.max(abs(actual - dense))))
    teacher = load_models()
    timings = {}
    functions = {
        'current_mlp': lambda obs: teacher.card_values(obs, obs.seat, obs.hand, obs.legal_actions()),
        'nnue_incremental': lambda obs: evaluator.card_values(obs, obs.seat, obs.hand, obs.legal_actions()),
        'nnue_dense_same_weights': lambda obs: network(feature_matrix(
            obs, obs.seat, obs.hand, obs.legal_actions())),
    }
    for name, function in functions.items():
        for obs in observations[:10]:
            function(obs)
        started = time.perf_counter()
        for _ in range(5):
            for obs in observations:
                function(obs)
        timings[name] = (time.perf_counter() - started) * 1e6 / (5 * len(observations))
    candidate = 'nnue:field=previous,time_ms=2'
    if args.play_models:
        candidate += f',play_models={args.play_models}'
    smoke = []
    for players, hand in ((3, 10), (4, 10), (8, 6)):
        bots = [make_bot(candidate, 17)] + [make_bot(DEFAULT_SPEC, 17 + s)
                                          for s in range(1, players)]
        scores, seconds, decisions = play_game(bots, Rules(players, hand), 17, timed_seat=0)
        smoke.append({'players': players, 'opening_hand': hand, 'seed': 17,
                      'scores': scores, 'candidate_seconds': seconds,
                      'candidate_decisions': decisions})
    report = {'candidate': provenance(candidate), 'baseline': provenance(DEFAULT_SPEC),
              'bank_sha256': file_hash(args.bank), 'selected_test_cases': len(chosen),
              'maximum_incremental_error_points': maximum_error,
              'microseconds_per_card_values_call': timings, 'full_game_smoke': smoke,
              'limitations': 'Three games verify completion, not strength. Timings include '
              'feature extraction on sampled test positions, not search or end-to-end '
              'turn latency; no speed or score gain is established.'}
    write_json(args.out, report)
    print({'maximum_error_points': maximum_error, 'microseconds': timings,
           'completed_games': len(smoke)}, flush=True)


if __name__ == '__main__':
    main()
