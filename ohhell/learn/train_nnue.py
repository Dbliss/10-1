"""Train the experimental NNUE on saved cases, with deal-separated validation.

Default targets distill the frozen v1 play network. --labels instead learns
cached frozen-policy round scores. Neither mode uses test rows for selection.
"""

import argparse
import os
from pathlib import Path
import time

# Small matrices suffer from excessive BLAS threading; set before NumPy loads.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np

from .cases import checked_metadata, file_hash, provenance, read_rows, restore, write_json
from .models import VALUE_SCALE, load_models
from .nn import MLP, huber_loss
from .nnue import INPUTS, MODEL_DIR, feature_matrix, save_network


def dataset(rows, teacher=None, labeled=False):
    groups = {split: {'x': [], 'y': [], 'cases': [], 'deals': set()}
              for split in ('train', 'validation', 'test')}
    for row in rows:
        case = row['case'] if labeled else row
        if case['part'] != 'play' or case['blind']:
            continue
        state = restore(case)
        obs = state.observation(state.to_act)
        cards = obs.legal_actions()
        if len(cards) < 2:
            continue
        features = feature_matrix(obs, obs.seat, obs.hand, cards)
        if labeled:
            values = {o['action']: o['score'] for o in row['outcomes']}
            if set(values) != set(cards):
                raise ValueError('Incomplete action labels')
            targets = [values[c] / VALUE_SCALE for c in cards]
        else:
            targets = np.asarray(teacher.card_values(obs, obs.seat, obs.hand, cards)) / VALUE_SCALE
        group = groups[case['split']]
        group['x'].append(features)
        group['y'].extend(targets)
        group['cases'].append((case['id'], len(cards)))
        group['deals'].add(case['deal_id'])
    for split, group in groups.items():
        if not group['x']:
            raise ValueError(f'No non-blind choice cases in {split}; use a larger bank')
        group['x'] = np.concatenate(group['x'])
        group['y'] = np.asarray(group['y'], dtype=np.float32)
    splits = list(groups)
    for i, left in enumerate(splits):
        for right in splits[i + 1:]:
            if groups[left]['deals'] & groups[right]['deals']:
                raise ValueError('Related deals appear in multiple splits')
    return groups


def predictions(model, x, batch_size=2048):
    return np.concatenate([model(x[start:start + batch_size])[:, 0]
                           for start in range(0, len(x), batch_size)])


def metrics(model, group):
    pred = predictions(model, group['x'])
    target = group['y']
    agreed = 0
    lost = []
    start = 0
    for _, count in group['cases']:
        values = target[start:start + count]
        selected = int(np.argmax(pred[start:start + count]))
        gap = float(max(values) - values[selected]) * VALUE_SCALE
        agreed += gap < 1e-5
        lost.append(gap)
        start += count
    return {'rows': len(target), 'choice_cases': len(group['cases']),
            'deals': len(group['deals']),
            'rmse_points': float(np.sqrt(np.mean((pred - target) ** 2)) * VALUE_SCALE),
            'target_best_agreement': agreed / len(group['cases']),
            'mean_target_choice_gap_points': float(np.mean(lost))}


def fit_separated(model, training, validation, epochs=20, batch_size=1024,
                  seed=1, rate=0.001, log=print):
    """Adam with complete final minibatches and external validation only."""
    rng = np.random.default_rng(seed)
    params = model.parameters()
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    bind = huber_loss()
    best = [p.copy() for p in params]
    best_loss = float('inf')
    step = 0
    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(training['y']))
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            _, grads = model._gradients(training['x'][batch], bind(training['y'], batch))
            step += 1
            for p, grad, m, v in zip(params, grads, first, second):
                m *= 0.9
                m += 0.1 * grad
                v *= 0.999
                v += 0.001 * grad * grad
                p -= rate * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
        error = predictions(model, validation['x']) - validation['y']
        loss = float(np.mean(np.where(abs(error) <= 1, 0.5 * error ** 2, abs(error) - 0.5)))
        if not np.isfinite(loss):
            raise ValueError('Training produced a non-finite validation loss')
        history.append(loss)
        log(f'epoch {epoch + 1}/{epochs}: validation Huber {loss:.6f}')
        if loss < best_loss:
            best_loss = loss
            best = [p.copy() for p in params]
    for p, saved in zip(params, best):
        p[...] = saved
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--bank', help='Distill current play-network scores on these cases')
    source.add_argument('--labels', help='Learn cached frozen-policy round scores instead')
    parser.add_argument('--teacher-models', default=None)
    parser.add_argument('--out', default=str(MODEL_DIR))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=20261110)
    args = parser.parse_args()
    if min(args.epochs, args.hidden, args.batch_size) < 1:
        parser.error('epochs, hidden and batch size must be positive')
    folder = Path(args.out)
    if folder.exists() and any(folder.iterdir()):
        parser.error('choose a new empty output directory to preserve previous candidates')
    path = args.labels or args.bank
    source_metadata = checked_metadata(path)
    began = time.perf_counter()
    teacher = load_models(play_models=args.teacher_models) if args.bank else None
    groups = dataset(read_rows(path), teacher, labeled=bool(args.labels))
    prepared = time.perf_counter()
    print({key: {'rows': len(value['y']), 'deals': len(value['deals'])}
           for key, value in groups.items()}, flush=True)
    model = MLP([INPUTS, args.hidden, 32, 1], seed=args.seed)
    initial_validation = metrics(model, groups['validation'])
    history = fit_separated(model, groups['train'], groups['validation'], args.epochs,
                            args.batch_size, args.seed, log=lambda s: print(s, flush=True))
    trained = time.perf_counter()
    report = {'settings': vars(args), 'source_sha256': file_hash(path),
              'source_metadata': source_metadata,
              'target': 'frozen-policy scores' if args.labels else 'v1 network distillation',
              'teacher': provenance('net' + (f':play_models={args.teacher_models}'
                                            if args.teacher_models else '')) if args.bank else None,
              'initial_validation': initial_validation, 'validation_loss_history': history,
              'validation': metrics(model, groups['validation']),
              'test': metrics(model, groups['test']),
              'preparation_seconds': prepared - began, 'training_seconds': trained - prepared,
              'limitations': 'Float NNUE, not quantized. Target fidelity is not playing '
              'strength. Only validation selects weights; test is measured once afterward.'}
    folder.mkdir(parents=True, exist_ok=True)
    save_network(model, folder / 'play.npz')
    report['checkpoint_sha256'] = file_hash(folder / 'play.npz')
    write_json(folder / 'training.json', report)
    print({key: report[key] for key in ('validation', 'test', 'preparation_seconds',
                                       'training_seconds', 'checkpoint_sha256')}, flush=True)


if __name__ == '__main__':
    main()
