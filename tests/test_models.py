"""Checkpoint comparisons must change only the requested part of the bot."""

from types import SimpleNamespace

import numpy as np
import pytest

from ohhell.arena import play_game
from ohhell.bots import make_bot
from ohhell.learn.models import MODEL_DIR, load_models, model_paths
from ohhell.learn.nn import MLP
from ohhell.rules import Rules
from ohhell.tournament import check_run_config


@pytest.fixture
def candidate_dir(tmp_path):
    """A second checkpoint pair that differs from the default weights."""
    for name in ('bid', 'play'):
        copy = MLP.load(MODEL_DIR / f'{name}.npz')
        copy.weights[0] = copy.weights[0] + np.float32(0.01)
        copy.save(tmp_path / f'{name}.npz')
    return tmp_path


def same_network(left, right):
    return all(np.array_equal(a, b)
               for a, b in zip(left.parameters(), right.parameters()))


@pytest.mark.parametrize('kind', ['net', 'hybrid'])
@pytest.mark.parametrize('part', ['bid', 'play'])
def test_single_checkpoint_override_preserves_the_other_network(kind, part, candidate_dir):
    current = load_models()
    candidate = load_models(str(candidate_dir))
    bot = make_bot(f'{kind}:{part}_models={candidate_dir}')
    other = 'play' if part == 'bid' else 'bid'
    assert same_network(getattr(bot.models, part), getattr(candidate, part))
    assert same_network(getattr(bot.models, other), getattr(current, other))
    # These checkpoints are actually different, so this is a real experiment.
    assert not same_network(getattr(current, part), getattr(candidate, part))


def test_overrides_take_precedence_over_combined_model_directory():
    bid, play = model_paths('combined', 'bids', 'cards')
    assert str(bid.as_posix()) == 'bids/bid.npz'
    assert str(play.as_posix()) == 'cards/play.npz'
    assert model_paths('combined')[0].as_posix() == 'combined/bid.npz'


@pytest.mark.parametrize('kind', ['net', 'hybrid'])
def test_candidate_bidding_checkpoint_finishes_full_game(kind, candidate_dir):
    spec = f'{kind}:bid_models={candidate_dir}'
    if kind == 'hybrid':
        spec += ',field=previous,worlds=2,bid_worlds=2'
    totals, _, _ = play_game([make_bot(spec, seat) for seat in range(4)], Rules(4, 3), 13)
    assert len(totals) == 4


@pytest.mark.parametrize('part', ['bid', 'play'])
def test_resume_fingerprints_separately_selected_weights(tmp_path, part):
    weights = tmp_path / 'weights'
    weights.mkdir()
    selected = weights / f'{part}.npz'
    selected.write_bytes(b'first checkpoint')
    args = SimpleNamespace(seed=1, max_sets=130, smallest_gain=1, time_ms=2, resume=False)
    manifest = tmp_path / 'run.meta'
    specs = [f'hybrid:{part}_models={weights}']
    check_run_config(manifest, args, specs)
    args.resume = True
    check_run_config(manifest, args, specs)
    selected.write_bytes(b'changed checkpoint')
    with pytest.raises(SystemExit, match='different rules, models or settings'):
        check_run_config(manifest, args, specs)
