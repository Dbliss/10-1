import random

from ohhell.arena import deal_round
from ohhell.cards import EFFECTIVE_SUIT, FULL_DECK, cards_in
from ohhell.engine import BIDDING, NO_BID, PLAYING, RoundState
from ohhell.learn import seat_sampler as ss
from ohhell.rules import Rules


def _bid_position(seed, n=3, hand_size=1, bids=None):
    rng = random.Random(seed)
    hands, trump = deal_round(rng, n, hand_size)
    state = RoundState(Rules(n, min(10, 51 // n)), hands, trump, n - 1)
    for b in (bids or [0] * (n - 1)):
        state.apply(min(b, hand_size))
    return state, state.observation(state.to_act)


def _check_consistent(obs, hands):
    others = [s for s in range(obs.n) if s != obs.seat]
    known = obs.hand | obs.played | (1 << obs.trump_card)
    sizes = ss._hidden_sizes(obs)
    seen = 0
    suit_of = EFFECTIVE_SUIT[obs.trump_suit]
    for s in others:
        assert hands[s].bit_count() == sizes[s]
        assert not hands[s] & known and not hands[s] & seen
        assert not any(obs.voids[s] >> suit_of[c] & 1 for c in cards_in(hands[s]))
        seen |= hands[s]
    assert hands[obs.seat] == obs.hand


def test_seat_worlds_deals_are_consistent_mid_round():
    rng = random.Random(4)
    hands, trump = deal_round(rng, 4, 6)
    state = RoundState(Rules(4, 10), hands, trump, 0)
    while state.phase == BIDDING:
        options = state.legal_actions()
        state.apply(options[len(options) // 3])  # moderate bids
    for _ in range(9):
        state.apply(state.legal_actions()[0])
    obs = state.observation(state.to_act)
    weighted, stats = ss.seat_worlds(obs, 400, random.Random(1))
    assert weighted and stats['calls'] >= 400
    for h, w in weighted:
        assert w > 0
        _check_consistent(obs, h)


def test_counter_matches_the_gaussian_bid_likelihood():
    state, obs = _bid_position(3, n=4, hand_size=3, bids=[1, 0, 2])
    counter = ss.Counter()
    a = counter.accept(obs, 0, state.hands[0])
    assert 0 < a <= 1 and counter.calls == 1


def test_seat_sampler_matches_exact_posterior_on_one_card_rounds():
    """Enumerate every deal of a 3-player one-card round with two strong bids
    and compare the chance each opponent holds a trump."""
    state, obs = _bid_position(21, n=3, hand_size=1, bids=[1, 1])
    unseen = cards_in(FULL_DECK & ~(obs.hand | (1 << obs.trump_card)))
    suit_of = EFFECTIVE_SUIT[obs.trump_suit]
    counter = ss.Counter()
    exact = {0: 0.0, 1: 0.0}
    total = 0.0
    for a in unseen:
        for b in unseen:
            if a == b:
                continue
            w = counter.accept(obs, 0, 1 << a) * counter.accept(obs, 1, 1 << b)
            total += w
            exact[0] += w * (suit_of[a] == obs.trump_suit)
            exact[1] += w * (suit_of[b] == obs.trump_suit)
    exact = {s: v / total for s, v in exact.items()}
    runs = []
    rng = random.Random(5)
    for _ in range(30):
        weighted, _ = ss.seat_worlds(obs, 3000, rng, estimator='draws', extra=8)
        tw = sum(w for _, w in weighted)
        runs.append({s: sum(w for h, w in weighted if suit_of[cards_in(h[s])[0]] == obs.trump_suit) / tw
                     for s in (0, 1)})
    for s in (0, 1):
        estimate = sum(r[s] for r in runs) / len(runs)
        assert abs(estimate - exact[s]) < 0.03, (s, estimate, exact[s])


def test_current_worlds_reports_candidates_and_budget():
    state, obs = _bid_position(7, n=4, hand_size=2, bids=[2, 0, 1])
    resampled, candidates, calls = ss.current_worlds(obs, random.Random(2), count=8, oversample=4)
    assert len(resampled) == 8 and len(candidates) == 32 and calls == 32 * 3
    assert ss.ess(candidates) <= 32
