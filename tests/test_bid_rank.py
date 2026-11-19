import json

import pytest

from ohhell.bid_rank import N_FEATURES
from ohhell.bots import make_bot
from ohhell.duel import REFERENCE_SPEC, extend
from ohhell.learn.bid_play_audit import bid_position
from ohhell.native import available


@pytest.mark.skipif(not available, reason='native core not built')
def test_zero_weight_ranker_preserves_bid_and_sample_count(tmp_path):
    obs, _, _, seed = bid_position(947334371)
    model = tmp_path / 'rank.json'
    model.write_text(json.dumps({'weights': [0.0] * N_FEATURES}))
    spec = extend(REFERENCE_SPEC, 'bid_budget=512')
    baseline = make_bot(spec, seed)
    candidate = make_bot(extend(spec, f'bid_rank_file={model}'), seed)
    assert obs.hand_size >= 8
    assert candidate.act(obs) == baseline.act(obs)
    assert candidate.simulator.last_bid_samples == baseline.simulator.last_bid_samples == 512
