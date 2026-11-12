"""Seat-bid inputs for the bid network and imagined later bidders."""

import random

import numpy as np
import pytest

from ohhell import native
from ohhell.arena import deal_round, play_game
from ohhell.bots import make_bot
from ohhell.bots.learned import HybridBot
from ohhell.bots.seat_bids import SeatBidsBot
from ohhell.cards import RANK_OF, SUIT_OF
from ohhell.engine import BIDDING, NO_BID, RoundState
from ohhell.learn.features import (
    BID_INPUTS, BID_SEAT_INPUTS, SEAT_SLOTS, _encode, _suit_order, bid_matrix,
)
from ohhell.learn.models import Models, load_models
from ohhell.learn.nn import MLP
from ohhell.learn.selfplay import _generate, widen_bid
from ohhell.rules import Rules, Scoring

V2 = 'ohhell/models'


def _previous_bid_matrix(hand, trump_card, n, hand_size, seat, dealer, bids, options):
    """The 98-input layout as it was before seat bids were added."""
    order = _suit_order(SUIT_OF[trump_card], hand)
    base = np.zeros(98, dtype=np.float32)
    base[:52] = _encode(hand, order)
    base[52 + RANK_OF[trump_card]] = 1.0
    base[65 + n - 2] = 1.0
    base[72 + hand_size - 1] = 1.0
    position = (seat - dealer - 1) % n
    earlier = [b for b in bids if b != NO_BID]
    total = sum(earlier)
    base[82] = position / (n - 1)
    base[83] = float(position == 0)
    base[84] = total / hand_size
    base[85] = (total - hand_size * len(earlier) / n) / hand_size
    rows = np.tile(base, (len(options), 1))
    for i, bid in enumerate(options):
        rows[i, 86 + bid] = 1.0
        rows[i, 97] = bid / hand_size
    return rows


def _random_spots(count, seed):
    rng = random.Random(seed)
    for _ in range(count):
        n = rng.randint(2, 8)
        size = rng.randint(1, min(10, 51 // n))
        hands, trump = deal_round(rng, n, size)
        dealer, seat = rng.randrange(n), rng.randrange(n)
        bids = [NO_BID] * n
        for j in range((seat - dealer - 1) % n):
            bids[(dealer + 1 + j) % n] = rng.randint(0, size)
        yield rng, hands[seat], trump, n, size, seat, dealer, bids


def _score(size):
    points = Scoring().points
    return np.array([[points(b, k) for k in range(size + 1)] for b in range(size + 1)])


def _wide_v2():
    v2 = load_models(V2)
    return v2, Models(widen_bid(MLP.load(f'{V2}/bid.npz'), BID_SEAT_INPUTS), v2.play)


def test_old_layout_is_unchanged_and_new_inputs_are_appended():
    for _, hand, trump, n, size, seat, dealer, bids in _random_spots(300, 1):
        options = list(range(size + 1))
        old = _previous_bid_matrix(hand, trump, n, size, seat, dealer, bids, options)
        assert np.array_equal(bid_matrix(hand, trump, n, size, seat, dealer, bids, options), old)
        new = bid_matrix(hand, trump, n, size, seat, dealer, bids, options, BID_SEAT_INPUTS)
        assert new.shape == (len(options), BID_SEAT_INPUTS)
        assert np.array_equal(new[:, :BID_INPUTS], old)
        position = (seat - dealer - 1) % n
        slots = new[0, BID_INPUTS:BID_INPUTS + 3 * SEAT_SLOTS].reshape(SEAT_SLOTS, 3)
        for j in range(SEAT_SLOTS):
            if j < position:
                bid = bids[(dealer + 1 + j) % n]
                assert slots[j].tolist() == pytest.approx([1.0, bid / size, float(bid == 0)])
            else:
                assert not slots[j].any()
        still = new[0, BID_INPUTS + 3 * SEAT_SLOTS:]
        assert still.sum() == 1 and still[n - 1 - position] == 1


def test_widened_network_predicts_exactly_as_before():
    v2, wide = _wide_v2()
    assert wide.bid_inputs == BID_SEAT_INPUTS and v2.bid_inputs == BID_INPUTS
    for _, hand, trump, n, size, seat, dealer, bids in _random_spots(100, 2):
        options = list(range(size + 1))
        before = v2.bid_distributions(hand, trump, n, size, seat, dealer, bids, options)
        after = wide.bid_distributions(hand, trump, n, size, seat, dealer, bids, options)
        for bid in options:
            assert after[bid] == pytest.approx(before[bid], abs=1e-6)


@pytest.mark.parametrize('wide', [False, True])
def test_batched_choice_matches_one_at_a_time(wide):
    models = _wide_v2()[wide]
    for rng, hand, trump, n, size, seat, dealer, bids in _random_spots(150, 3):
        options = list(range(size + 1))
        score = _score(size)
        legal = np.ones(size + 1, dtype=bool)
        if size:
            legal[rng.randrange(size + 1)] = False
        dists = models.bid_distributions(hand, trump, n, size, seat, dealer, bids, options)
        value = lambda b: float(np.dot(dists[b], score[b]))
        slow = max((b for b in options if legal[b]), key=value)
        fast = models.choose_bids([hand], [bids], trump, n, size, seat, dealer,
                                  [options], [legal], score)
        assert fast[0] == slow
        # A window of candidates, padded with an invalid repeat.
        low = rng.randrange(size + 1)
        window = [b for b in range(low, min(size, low + 2) + 1)]
        padded = window + [window[-1]] * (4 - len(window))
        valid = [i < len(window) for i in range(4)]
        fast = models.choose_bids([hand], [bids], trump, n, size, seat, dealer,
                                  [padded], [valid], score)
        assert fast[0] == max(window, key=value)


def _bidding_obs(seed, n=4, size=6, bids_before=1):
    hands, trump = deal_round(random.Random(seed), n, size)
    state = RoundState(Rules(n, size), hands, trump, 0)
    for _ in range(bids_before):
        state.apply(state.legal_actions()[1])
    return state.observation(state.to_act)


def test_imagined_bids_fill_later_seats_in_order_and_respect_the_dealer():
    obs = _bidding_obs(4, n=5, size=6, bids_before=2)
    bot = SeatBidsBot(seed=1, field='previous', models=V2, fast=False)
    later = bot.later_bidders(obs)
    assert later == [(obs.seat + j) % obs.n for j in range(1, obs.n - 2)]
    assert later[-1] == obs.dealer
    options = list(range(obs.hand_size + 1))
    hands, kinds = bot._draw(obs, 12)
    tables = bot.imagine_bids(obs, hands, kinds, options, later, _score(obs.hand_size))
    assert tables.shape == (12, len(options), native.MAX_SEATS)
    for w in range(12):
        for o, bid in enumerate(options):
            row = tables[w, o, :obs.n]
            assert row[obs.seat] == bid
            assert all(row[s] == obs.bids[s] for s in range(obs.n) if obs.bids[s] != NO_BID)
            assert (row >= 0).all() and (row <= obs.hand_size).all()
            assert row.sum() != obs.hand_size  # the dealer's bid stays legal
            assert (tables[w, o, obs.n:] == NO_BID).all()


def test_later_bidders_see_our_bid():
    """The imagined bids depend on the option we are considering."""
    obs = _bidding_obs(9, n=4, size=8, bids_before=0)
    bot = SeatBidsBot(seed=1, field='previous', models=V2, fast=False)
    later = bot.later_bidders(obs)
    options = list(range(obs.hand_size + 1))
    hands, kinds = bot._draw(obs, 40)
    tables = bot.imagine_bids(obs, hands, kinds, options, later, _score(obs.hand_size))
    assert any(len({tuple(tables[w, o, later]) for o in range(len(options))}) > 1
               for w in range(40))


@pytest.mark.parametrize('fast', [False, True])
def test_analysis_counts_complete_worlds(fast):
    if fast and not native.available:
        pytest.skip('core not built')
    obs = _bidding_obs(5)
    bot = SeatBidsBot(seed=3, bid_worlds=24, field='previous', models=V2, fast=fast)
    options = bot._bid_options(obs)
    result = bot._analyse_bids(obs, options)
    assert result.samples == 24 == bot.last_bid_samples
    for bid, dist in result.distributions.items():
        assert len(dist) == obs.hand_size + 1 and sum(dist) == pytest.approx(1.0)
        assert result.values[bid] == pytest.approx(
            sum(p * Scoring().points(bid, k) for k, p in enumerate(dist)))


def test_native_and_python_scoring_agree_on_fixed_bids():
    if not native.available:
        pytest.skip('core not built')
    obs = _bidding_obs(6, n=4, size=5)
    fast = SeatBidsBot(seed=2, field='previous', models=V2)
    hands, kinds = fast._draw(obs, 30)
    options = list(range(obs.hand_size + 1))
    bids = fast.imagine_bids(obs, hands, kinds, options, fast.later_bidders(obs),
                             _score(obs.hand_size))
    width = obs.hand_size + 1
    in_c = np.zeros((len(options), width))
    in_python = np.zeros((len(options), width))
    fast._score(obs, hands, kinds, bids, in_c)
    slow = SeatBidsBot(seed=2, field='previous', models=V2, fast=False)
    slow._score(obs, hands, kinds, bids, in_python)
    # The 'previous' kind plays without card noise, so play-outs are identical.
    assert np.array_equal(in_c, in_python)


def test_imagine_off_and_last_bidder_match_hybrid():
    kwargs = dict(seed=13, bid_worlds=32, field='previous', models=V2)
    obs = _bidding_obs(7)
    hybrid = HybridBot(**kwargs).analyse_bids(obs)
    off = SeatBidsBot(imagine=0, **kwargs).analyse_bids(obs)
    assert off.distributions == hybrid.distributions
    dealer_obs = _bidding_obs(7, bids_before=3)
    assert SeatBidsBot.later_bidders(dealer_obs) == []
    assert (SeatBidsBot(**kwargs).analyse_bids(dealer_obs).distributions
            == HybridBot(**kwargs).analyse_bids(dealer_obs).distributions)


@pytest.mark.parametrize('fast', [False, True])
def test_full_game_including_blind_round(fast):
    if fast and not native.available:
        pytest.skip('core not built')
    spec = f'seatbids:field=previous,bid_worlds=4,worlds=2,fast={int(fast)},models={V2}'
    totals, _, _ = play_game([make_bot(spec, s) for s in range(4)], Rules(4, 3), 21)
    assert len(totals) == 4


def test_timed_bids_stay_near_the_allowance():
    if not native.available:
        pytest.skip('core not built')
    import time
    bot = make_bot(f'seatbids:field=previous,time_ms=2,models={V2}', 1)
    obs = _bidding_obs(11, n=6, size=8, bids_before=0)
    bot.act(obs)  # warm up
    began = time.perf_counter()
    for _ in range(20):
        bot.act(obs)
    assert (time.perf_counter() - began) / 20 < 0.006  # 2 ms plus one block of worlds


def test_selfplay_rows_use_the_networks_width(tmp_path):
    wide = tmp_path / 'wide'
    wide.mkdir()
    widen_bid(MLP.load(f'{V2}/bid.npz'), BID_SEAT_INPUTS).save(wide / 'bid.npz')
    (wide / 'play.npz').write_bytes(open(f'{V2}/play.npz', 'rb').read())
    bid_x, bid_k, _, play_x, _ = _generate((5, 30, str(wide), 1.0, 0.05, 0.0, 0.3))
    assert bid_x.shape[1] == BID_SEAT_INPUTS and len(bid_x) == len(bid_k) > 0
    assert len(play_x) == 0  # play labels off when the play network is frozen


def test_tournament_fingerprints_seat_bid_weights(tmp_path):
    from types import SimpleNamespace
    from ohhell.tournament import check_run_config
    weights = tmp_path / 'weights'
    weights.mkdir()
    (weights / 'bid.npz').write_bytes(b'first')
    (weights / 'play.npz').write_bytes(b'play')
    args = SimpleNamespace(seed=1, max_sets=130, smallest_gain=1, time_ms=2, resume=False)
    manifest = tmp_path / 'run.meta'
    specs = [f'seatbids:models={weights}']
    check_run_config(manifest, args, specs)
    args.resume = True
    (weights / 'bid.npz').write_bytes(b'changed')
    with pytest.raises(SystemExit, match='different rules, models or settings'):
        check_run_config(manifest, args, specs)
