"""Reproducible, independent bid/play benchmarks and cached training rows.

Run ``python -m ohhell.learn.cases --help``. Labels are counterfactual outcomes
under frozen continuations on saved deals, NOT perfect-information optima.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

from ..arena import deal_round
from ..bots import DEFAULT_SPEC, make_bot
from ..cards import SUIT_CARDS
from ..engine import BIDDING, PLAYING, DONE, RoundState
from ..rules import Rules

VERSION = 1
PROFILES = ('heuristic', 'zero', 'high', 'random')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def checked_metadata(path):
    metadata = json.loads(Path(str(path) + '.meta.json').read_text(encoding='utf-8'))
    if metadata.get('version') != VERSION or metadata.get('sha256') != file_hash(path):
        raise ValueError('Input version or fingerprint differs from its completed manifest')
    return metadata


def provenance(spec):
    """Record selected checkpoints and source, including uncommitted changes."""
    from .models import model_paths
    _, _, args = spec.partition(':')
    opts = dict(pair.split('=', 1) for pair in args.split(',') if '=' in pair)
    if spec.partition(':')[0] == 'nnue':
        from .nnue import MODEL_DIR as NNUE_DIR
        opts.setdefault('play_models', opts.get('models', str(NNUE_DIR)))
    paths = model_paths(opts.get('models'), opts.get('bid_models'), opts.get('play_models'))
    sources = sorted(Path(__file__).parents[1].rglob('*.py'))
    sources += sorted((Path(__file__).parents[1] / 'native').glob('*.c'))
    return {'spec': spec, 'models': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in paths if p.exists()},
            'source_sha256': digest({str(p.relative_to(Path(__file__).parents[1])):
                                    hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})}


def restore(case):
    state = RoundState(Rules(**case['rules']), case['hands'], case['trump'],
                       case['dealer'], blind=case['blind'])
    for action in case['actions']:
        state.apply(action)
    return state


def tags(state):
    seat = state.to_act
    result = [f'players={state.n}', f'hand={0 if state.blind else state.hand_size}',
              f'position={(seat - state.dealer - 1) % state.n}',
              f'trump_count={(state.hands[seat] & SUIT_CARDS[state.trump_suit][state.trump_suit]).bit_count()}'
              if not state.blind else 'hidden_hand']
    if state.phase == PLAYING:
        need = state.bids[seat] - state.tricks_won[seat]
        left = state.hands[seat].bit_count()
        result += [f'need={need}', f'trick_position={state.trick_size}',
                   'early' if left * 2 > state.hand_size else 'late',
                   'overshot' if need < 0 else 'unreachable' if need > left else
                   'target_met' if need == 0 else 'needs_tricks',
                   'forced' if len(state.legal_actions()) == 1 else 'choice']
    return result


def build_bank(seed=1, repeats=4, max_hand=10, players=range(2, 9)):
    """Stratify every supported table/hand size; retain every bid and play.

    Related snapshots and alternate bid profiles share a deal-level split.
    Each repeat rotates dealer and cycles deliberately different bid profiles.
    """
    for n in players:
        top = min(max_hand, 51 // n)
        for label in range(top + 1):
            for repeat in range(repeats):
                key = [VERSION, seed, n, label, repeat]
                deal_seed = int(digest(key)[:15], 16)
                rng = random.Random(deal_seed)
                hands, trump = deal_round(rng, n, max(1, label))
                deal_id = digest([n, label, hands, trump])
                bucket = int(deal_id[:8], 16) % 10
                split = 'train' if bucket < 8 else 'validation' if bucket == 8 else 'test'
                for profile in PROFILES:
                    dealer = repeat % n
                    state = RoundState(Rules(n, top), hands, trump, dealer, blind=label == 0)
                    bots = [make_bot('heuristic', deal_seed + s) for s in range(n)]
                    actions = []
                    while state.phase != DONE:
                        case = {'id': digest([key, profile, len(actions)]), 'deal_id': deal_id,
                                'seed': deal_seed, 'split': split, 'profile': profile,
                                'part': 'bid' if state.phase == BIDDING else 'play',
                                'rules': {'num_players': n, 'max_hand': top,
                                          'dealer_restriction': True},
                                'hands': hands, 'trump': trump, 'dealer': dealer,
                                'blind': label == 0, 'actions': list(actions), 'tags': tags(state)}
                        yield case
                        options = state.legal_actions()
                        if state.phase == BIDDING and profile != 'heuristic':
                            action = (min(options) if profile == 'zero' else max(options)
                                      if profile == 'high' else rng.choice(options))
                        else:
                            action = bots[state.to_act].act(state.observation(state.to_act))
                        state.apply(action)
                        actions.append(action)


def label_case(case, continuation='heuristic', opponents='mixed', samples=1):
    """Force every legal root action, then use fixed observation-only policies.

    Other bidders respond afresh to a forced bid. Common seeds couple options.
    Samples vary policy randomness, not the saved hidden deal.
    """
    if samples < 1:
        raise ValueError('samples must be positive')
    state = restore(case)
    seat = state.to_act
    outcomes = []
    for action in state.legal_actions():
        scores, tricks = [], []
        for sample in range(samples):
            trial = restore(case)
            bots = [make_bot(continuation if s == seat else opponents,
                             case['seed'] + sample * 1009 + s) for s in range(state.n)]
            trial.apply(action)
            while trial.phase != DONE:
                acting = trial.to_act
                trial.apply(bots[acting].act(trial.observation(acting)))
            scores.append(trial.points()[seat])
            tricks.append(trial.tricks_won[seat])
        outcomes.append({'action': action, 'score': sum(scores) / samples, 'tricks': tricks})
    return {'case': case, 'outcomes': outcomes}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def evaluate(rows, candidate, baseline, part, split='test'):
    groups = defaultdict(list)
    details = []
    elapsed = 0.0
    for row in rows:
        case = row['case']
        if case['part'] != part or case['split'] != split:
            continue
        state = restore(case)
        obs = state.observation(state.to_act)
        values = {o['action']: o['score'] for o in row['outcomes']}
        if set(values) != set(obs.legal_actions()):
            raise ValueError('Labels do not cover exactly the legal actions')
        start = time.perf_counter()
        chosen = make_bot(candidate, case['seed']).act(obs)
        elapsed += time.perf_counter() - start
        control = make_bot(baseline, case['seed']).act(obs)
        if chosen not in values or control not in values:
            raise ValueError(f'Illegal decision in {case["id"]}')
        detail = {'id': case['id'], 'deal_id': case['deal_id'], 'chosen': chosen,
                  'baseline': control, 'score': values[chosen],
                  'edge': values[chosen] - values[control],
                  'hindsight_gap': max(values.values()) - values[chosen]}
        details.append(detail)
        for tag in ['all', *case['tags'], f'profile={case["profile"]}']:
            groups[tag].append(detail)
    if not details:
        raise ValueError('No cases match the requested part/split')
    summary = {}
    for tag, items in sorted(groups.items()):
        # Average within a deal first: snapshots are correlated.
        deals = defaultdict(list)
        for item in items:
            deals[item['deal_id']].append(item['edge'])
        edges = [sum(v) / len(v) for v in deals.values()]
        summary[tag] = {'cases': len(items), 'deals': len(deals),
                        'deal_mean_edge': sum(edges) / len(edges),
                        'mean_score': sum(i['score'] for i in items) / len(items),
                        'mean_hindsight_gap': sum(i['hindsight_gap'] for i in items) / len(items)}
    return {'part': part, 'split': split, 'candidate': provenance(candidate),
            'baseline': provenance(baseline), 'summary': summary, 'decisions': details,
            'candidate_ms_including_setup': elapsed * 1000 / len(details),
            'limitations': 'Frozen continuation on saved hidden deals; hindsight gap is not '
            'a perfect-play accuracy measure. This diagnostic does not promote a bot.'}


def export_rows(rows, part, split, output):
    import numpy as np
    from .features import BID_INPUTS, PLAY_INPUTS, bid_matrix, play_matrix
    from .models import VALUE_SCALE
    xs, ys, limits, ids = [], [], [], []
    skipped = 0
    for row in rows:
        case = row['case']
        if case['part'] != part or case['split'] != split:
            continue
        if case['blind']:
            skipped += 1  # current networks have no blind input encoding
            continue
        state = restore(case)
        obs = state.observation(state.to_act)
        for outcome in row['outcomes']:
            action = outcome['action']
            features = (bid_matrix(obs.hand, obs.trump_card, obs.n, obs.hand_size,
                                   obs.seat, obs.dealer, obs.bids, [action]) if part == 'bid'
                        else play_matrix(obs, obs.seat, obs.hand, [action]))
            for tricks in outcome['tricks']:
                xs.append(features[0])
                ys.append(tricks if part == 'bid' else obs.rules.scoring.points(
                    obs.bids[obs.seat], tricks) / VALUE_SCALE)
                limits.append(obs.hand_size)
                ids.append(case['deal_id'])
    if not xs:
        raise ValueError('No non-blind rows match the requested part/split')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, x=np.asarray(xs, dtype=np.float32).reshape(
        -1, BID_INPUTS if part == 'bid' else PLAY_INPUTS),
        y=np.asarray(ys, dtype=np.int64 if part == 'bid' else np.float32),
        limits=np.asarray(limits), deal_id=np.asarray(ids),
        part=part, split=split, skipped_blind_cases=skipped)
    return {'rows': len(xs), 'skipped_blind_cases': skipped, 'part': part, 'split': split}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build', help='Save predetermined cases (no model training)')
    build.add_argument('--out', required=True)
    build.add_argument('--seed', type=int, default=1)
    build.add_argument('--repeats', type=int, default=4)
    build.add_argument('--max-hand', type=int, default=10)
    label = commands.add_parser('label', help='Cache all legal action outcomes once')
    label.add_argument('--bank', required=True)
    label.add_argument('--out', required=True)
    label.add_argument('--continuation', default='heuristic')
    label.add_argument('--opponents', default='mixed')
    label.add_argument('--samples', type=int, default=1)
    for name in ('evaluate', 'export'):
        sub = commands.add_parser(name)
        sub.add_argument('--labels', required=True)
        sub.add_argument('--part', choices=('bid', 'play'), required=True)
        sub.add_argument('--split', choices=('train', 'validation', 'test'), default='test')
        sub.add_argument('--out', required=True)
        if name == 'evaluate':
            sub.add_argument('--candidate', default=DEFAULT_SPEC)
            sub.add_argument('--baseline', default=DEFAULT_SPEC)
    args = parser.parse_args()
    if args.command == 'export' and Path(args.out).suffix != '.npz':
        parser.error('training export output must end in .npz')
    if Path(args.out).exists() or Path(str(args.out) + '.meta.json').exists():
        parser.error('output already exists; choose a fresh filename')
    if args.command in ('build', 'label'):
        if args.command == 'build':
            if args.repeats < 1 or not 1 <= args.max_hand <= 10:
                parser.error('repeats must be positive; max-hand must be 1..10')
            rows = build_bank(args.seed, args.repeats, args.max_hand)
            metadata = {'version': VERSION, 'settings': vars(args),
                        'rules': asdict(Rules()), 'sequence': 'down/blind/up',
                        'scope': 'Independent round decisions, not cumulative game evaluation'}
        else:
            if args.samples < 1:
                parser.error('samples must be positive')
            if Path(args.bank).resolve() == Path(args.out).resolve():
                parser.error('label output must differ from the input bank')
            checked_metadata(args.bank)
            rows = (label_case(c, args.continuation, args.opponents, args.samples)
                    for c in read_rows(args.bank))
            metadata = {'version': VERSION, 'settings': vars(args),
                        'bank_sha256': file_hash(args.bank),
                        'continuation': provenance(args.continuation),
                        'opponents': provenance(args.opponents),
                        'label_kind': 'frozen-policy counterfactual, not optimal-action labels'}
        target = Path(args.out)
        if target.exists():
            parser.error('output already exists; choose a fresh filename')
        target.parent.mkdir(parents=True, exist_ok=True)
        coverage = Counter()
        with target.open('x', encoding='utf-8') as stream:
            for count, row in enumerate(rows, 1):
                stream.write(json.dumps(row) + '\n')
                case = row.get('case', row)
                coverage.update([case['part'], case['split'], *case['tags']])
                if count % 1000 == 0:
                    print(f'{count:,} cases', flush=True)
        metadata['coverage'] = dict(coverage)
        metadata['sha256'] = file_hash(target)
        write_json(str(target) + '.meta.json', metadata)
        print(json.dumps(dict(coverage), indent=2))
    else:
        label_metadata = checked_metadata(args.labels)
        rows = read_rows(args.labels)
        if args.command == 'evaluate':
            result = evaluate(rows, args.candidate, args.baseline, args.part, args.split)
            result['labels_sha256'] = file_hash(args.labels)
            result['label_metadata'] = label_metadata
            write_json(args.out, result)
            print(json.dumps(result['summary']['all'], indent=2))
        else:
            result = export_rows(rows, args.part, args.split, args.out)
            result['labels_sha256'] = file_hash(args.labels)
            result['label_metadata'] = label_metadata
            write_json(str(args.out) + '.meta.json', result)
            print(json.dumps(result))


if __name__ == '__main__':
    main()
