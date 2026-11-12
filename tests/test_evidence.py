"""Evidence bidding must use real sample counts and preserve the bid shortlist."""

import random
from types import SimpleNamespace

import pytest

from ohhell import native
from ohhell.arena import deal_round, play_game
from ohhell.bots import make_bot
from ohhell.bots.evidence import EvidenceBot
from ohhell.bots.learned import HybridBot
from ohhell.engine import BIDDING, RoundState
from ohhell.rules import Rules
from ohhell.tournament import check_run_config


def bid_obs():
    hands, trump = deal_round(random.Random(7), 4, 8)
    state = RoundState(Rules(4, 8), hands, trump, 3)
    return state.observation(state.to_act)


@pytest.mark.parametrize('samples,weight', [(4, 64 / 68), (64, 0.5), (256, 0.2)])
def test_actual_samples_control_prior_and_can_overturn_it(monkeypatch, samples, weight):
    obs = bid_obs()
    bot = EvidenceBot()
    # Model expects exactly one trick, rollout expects zero for every bid.
    bot.models = SimpleNamespace(bid_distributions=lambda *args: {
        b: [0., 1.] + [0.] * 7 for b in args[-1]})
    monkeypatch.setattr(bot, '_bid_options', lambda _: [0, 1])

    def simulate(_obs, options):
        return SimpleNamespace(samples=samples, distributions={
            b: [1., 0.] + [0.] * 7 for b in options})

    monkeypatch.setattr(bot, '_analyse_bids', simulate)
    analysis = bot.analyse_bids(obs)
    assert analysis.samples == samples
    assert bot.last_bid_search['model_weight'] == pytest.approx(weight)
    assert analysis.distributions[0][:2] == pytest.approx([1 - weight, weight])
    assert max(analysis.values, key=analysis.values.get) == (1 if samples == 4 else 0)


def test_model_cannot_change_heuristic_shortlist_or_add_illegal_bids(monkeypatch):
    obs = bid_obs()
    bot = EvidenceBot(bid_worlds=2, fast=False)
    monkeypatch.setattr(bot, '_bid_options', lambda _: [0, 1])
    bot.models = SimpleNamespace(bid_distributions=lambda *args: {
        b: [0.] * b + [1.] + [0.] * (8 - b) for b in args[-1]})
    result = bot.analyse_bids(obs)
    assert set(result.distributions) == {0, 1}
    assert result.samples == 2
    assert all(sum(dist) == pytest.approx(1) for dist in result.distributions.values())

    # This seat is the dealer, and a zero bid is forbidden by the auction.
    hands, trump = deal_round(random.Random(2), 4, 3)
    state = RoundState(Rules(4, 3), hands, trump, 3)
    for bid in (1, 1, 1):
        state.apply(bid)
    dealer_obs = state.observation(3)
    actual = EvidenceBot(bid_worlds=2, fast=False).analyse_bids(dealer_obs)
    assert 0 not in actual.distributions
    assert set(actual.distributions) <= set(dealer_obs.legal_actions())


@pytest.mark.parametrize('fast', [False, True])
def test_backends_report_completed_worlds(fast):
    if fast and not native.available:
        pytest.skip('core not built')
    bot = EvidenceBot(bid_worlds=7, fast=fast)
    result = bot.analyse_bids(bid_obs())
    assert result.samples == 7
    assert bot.last_bid_search['model_weight'] == pytest.approx(64 / 71)


def test_card_play_matches_hybrid():
    hands, trump = deal_round(random.Random(8), 4, 6)
    state = RoundState(Rules(4, 6), hands, trump, 3)
    while state.phase == BIDDING:
        state.apply(state.legal_actions()[0])
    obs = state.observation(state.to_act)
    kwargs = dict(seed=10, worlds=4, field='previous')
    assert EvidenceBot(**kwargs).play(obs) == HybridBot(**kwargs).play(obs)


@pytest.mark.parametrize('fast', [False, True])
def test_64_completed_worlds_reproduce_current_half_blend(fast):
    if fast and not native.available:
        pytest.skip('core not built')
    kwargs = dict(seed=13, bid_worlds=64, fast=fast, field='previous')
    current = HybridBot(**kwargs).analyse_bids(bid_obs())
    candidate = EvidenceBot(**kwargs).analyse_bids(bid_obs())
    assert candidate.distributions == current.distributions
    assert candidate.values == current.values


@pytest.mark.parametrize('fast', [False, True])
def test_full_game_including_blind_round(fast):
    if fast and not native.available:
        pytest.skip('core not built')
    spec = f'evidence:field=previous,bid_worlds=2,worlds=2,fast={int(fast)}'
    totals, _, _ = play_game([make_bot(spec, s) for s in range(4)], Rules(4, 3), 21)
    assert len(totals) == 4


def test_evidence_resume_detects_changed_checkpoint(tmp_path):
    weights = tmp_path / 'weights'
    weights.mkdir()
    selected = weights / 'bid.npz'
    selected.write_bytes(b'first')
    args = SimpleNamespace(seed=1, max_sets=130, smallest_gain=1, time_ms=2, resume=False)
    manifest = tmp_path / 'run.meta'
    specs = [f'evidence:bid_models={weights}']
    check_run_config(manifest, args, specs)
    args.resume = True
    check_run_config(manifest, args, specs)
    selected.write_bytes(b'changed')
    with pytest.raises(SystemExit, match='different rules, models or settings'):
        check_run_config(manifest, args, specs)


@pytest.mark.parametrize('prior', [-1, float('nan'), float('inf')])
def test_invalid_prior_is_rejected(prior):
    with pytest.raises(ValueError, match='bid_prior_worlds'):
        EvidenceBot(bid_prior_worlds=prior)


def test_zero_prior_uses_only_rollouts():
    bot = EvidenceBot(bid_prior_worlds=0, bid_worlds=2, fast=False)
    bot.analyse_bids(bid_obs())
    assert bot.last_bid_search['model_weight'] == 0
