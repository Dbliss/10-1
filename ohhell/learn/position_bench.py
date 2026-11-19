"""Cache deep observation-only play references and report decision gaps.

References share the chosen continuation's biases. A point gap here is a
diagnostic against that reference, never measured game points lost.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from ..bots import DEFAULT_SPEC, make_bot
from .cases import (VERSION, checked_metadata, evaluate, file_hash, provenance,
                    read_rows, restore, write_json)


def label_position(case, reference, repeats=2):
    state = restore(case)
    obs = state.observation(state.to_act)
    cards = obs.legal_actions()
    scores = {c: [] for c in cards}
    counts = []
    for repeat in range(repeats):
        bot = make_bot(reference, case['seed'] + 1009 * repeat)
        if not hasattr(bot, 'evaluate_cards'):
            raise ValueError('Reference must provide simulation card values')
        bot.deadline = time.perf_counter() + bot.time_limit if bot.time_limit else None
        try:
            values = bot.evaluate_cards(obs, cards)
        finally:
            bot.deadline = None
        counts.append(bot.last_card_worlds)
        for card in cards:
            scores[card].append(values[card])
    return {'case': case, 'outcomes': [
        {'action': card, 'score': sum(values) / len(values), 'replicate_scores': values}
        for card, values in scores.items()], 'reference_worlds': counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    label = commands.add_parser('label')
    label.add_argument('--bank', required=True)
    label.add_argument('--reference', default='hybrid:field=previous,worlds=512')
    label.add_argument('--repeats', type=int, default=2)
    label.add_argument('--limit', type=int, default=1000, help='stratified selection, all splits')
    label.add_argument('--out', required=True)
    score = commands.add_parser('evaluate')
    score.add_argument('--labels', required=True)
    score.add_argument('--candidate', default=DEFAULT_SPEC)
    score.add_argument('--baseline', default=DEFAULT_SPEC)
    score.add_argument('--split', choices=('train', 'validation', 'test'), default='test')
    score.add_argument('--out', required=True)
    args = parser.parse_args()
    target = Path(args.out)
    if target.exists() or Path(str(target) + '.meta.json').exists():
        parser.error('choose a fresh output filename')
    began = time.perf_counter()
    if args.command == 'label':
        if args.repeats < 1 or args.limit < 1:
            parser.error('repeats and limit must be positive')
        metadata = checked_metadata(args.bank)
        if metadata.get('trump_ranking') != 'bowers':
            parser.error('bank must identify current bower rules; rebuild legacy banks')
        # Round-robin strata, avoiding a prefix containing only the small tables.
        groups = {}
        for case in read_rows(args.bank):
            if case['part'] != 'play' or case['blind']:
                continue
            state = restore(case)
            if len(state.legal_actions()) < 2:
                continue
            key = (state.n, state.hand_size, case['split'], state.trick_size == 0)
            groups.setdefault(key, []).append(case)
        selected = []
        depth = 0
        while len(selected) < args.limit:
            layer = [rows[depth] for _, rows in sorted(groups.items()) if depth < len(rows)]
            if not layer:
                break
            selected.extend(layer[:args.limit - len(selected)])
            depth += 1
        if not selected:
            parser.error('no visible play choices in bank')
        target.parent.mkdir(parents=True, exist_ok=True)
        coverage = Counter()
        with target.open('x', encoding='utf-8') as stream:
            for i, case in enumerate(selected, 1):
                stream.write(json.dumps(label_position(case, args.reference, args.repeats)) + '\n')
                coverage.update([case['split'], *case['tags']])
                if i % 100 == 0:
                    print(f'{i}/{len(selected)} positions', flush=True)
        write_json(str(target) + '.meta.json', {
            'version': VERSION, 'sha256': file_hash(target), 'settings': vars(args),
            'bank': metadata, 'reference': provenance(args.reference),
            'coverage': dict(coverage), 'seconds': time.perf_counter() - began,
            'label_kind': 'sampled observation-only rollout/exact values; not optimal play',
            'limitations': 'Reference uses its own sampler and continuation; systematic '
            'errors and strategy fusion are shared. Independent rounds, no persistent reading.'})
        print(f'Saved {len(selected)} positions to {target}')
    else:
        metadata = checked_metadata(args.labels)
        result = evaluate(read_rows(args.labels), args.candidate, args.baseline, 'play', args.split)
        result.update(labels_sha256=file_hash(args.labels), label_metadata=metadata,
                      limitations='Gaps against a sampled observation-only reference, not actual '
                      'game points lost. Shared model errors are invisible. Independent positions '
                      'cannot evaluate cross-round typing or banked time.')
        write_json(target, result)
        print(json.dumps(result['summary']['all'], indent=2))


if __name__ == '__main__':
    main()
