"""Distill the current play network into a compact native rollout policy."""

import argparse
import os
from pathlib import Path
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

from . import train_nnue
from .cases import checked_metadata, file_hash, provenance, read_rows, write_json
from .models import load_models
from .nn import MLP
from .rollout import MODEL_DIR, SIZES, feature_matrix, save_network


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--bank')
    source.add_argument('--labels', help='Fit observation-only deep-reference scores instead')
    parser.add_argument('--out', default=str(MODEL_DIR))
    parser.add_argument('--teacher-models')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20261113)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    folder = Path(args.out)
    if folder.exists() and any(folder.iterdir()):
        parser.error('choose a new empty output directory')
    path = args.bank or args.labels
    metadata = checked_metadata(path)
    if (metadata.get('trump_ranking') != 'bowers'
            and metadata.get('bank', {}).get('trump_ranking') != 'bowers'):
        parser.error('source must identify current bower rules; rebuild legacy banks/labels')
    began = time.perf_counter()
    groups = train_nnue.dataset(read_rows(path), load_models(play_models=args.teacher_models)
        if args.bank else None, labeled=bool(args.labels), feature_builder=feature_matrix)
    model = MLP(SIZES, seed=args.seed)
    history = train_nnue.fit_separated(model, groups['train'], groups['validation'],
                                      epochs=args.epochs, seed=args.seed)
    folder.mkdir(parents=True, exist_ok=True)
    save_network(model, folder / 'play.npz')
    report = {'settings': vars(args), 'rules': 'bowers; independent saved round positions',
              'target': 'v2 network distillation' if args.bank else 'deep sampled reference scores',
              'bank': metadata, 'teacher': provenance('net' + (
                  f':play_models={args.teacher_models}' if args.teacher_models else '')) if args.bank else None,
              'validation_loss_history': history,
              'training': {'rows': len(groups['train']['y']),
                           'choice_cases': len(groups['train']['cases']),
                           'deals': len(groups['train']['deals'])},
              'validation': train_nnue.metrics(model, groups['validation']),
              'test': train_nnue.metrics(model, groups['test']),
              'checkpoint_sha256': file_hash(folder / 'play.npz'),
              'seconds': time.perf_counter() - began,
              'limitations': 'Target fidelity is not playing strength or proof of policy improvement. '
              'Only own hand and public state; validation selects weights; test measured afterward.'}
    write_json(folder / 'training.json', report)
    print(report['test'], flush=True)


if __name__ == '__main__':
    main()
