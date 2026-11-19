"""Bots that use the models trained by self-play."""

import json
from functools import lru_cache
from pathlib import Path

from .. import bid_calibration, bid_rank
from ..cards import EFFECTIVE_SUIT, OUTRANKS, RANK_OF, cards_in
from ..engine import NO_BID, NO_SUIT
from ..learn.models import load_models
from .base import Bot
from .heuristic import table_call
from .simulation import BidAnalysis, SimulationBot


def bid_table_path(path):
    """The table file as given, or relative to the repository."""
    from ..learn.models import REPO_ROOT
    file = Path(path)
    if not file.is_absolute() and not file.exists() and (REPO_ROOT / file).exists():
        return REPO_ROOT / file
    return file


@lru_cache(maxsize=8)
def load_bid_table(path):
    """{(hand size, players): network share} from a ``bid_blend_fit`` file."""
    table = {}
    for key, share in json.loads(bid_table_path(path).read_text())['share'].items():
        cards, players = key.split(',')
        if not 0.0 <= share <= 1.0:
            raise ValueError(f'Bid share {share} for {key} is outside 0 to 1.')
        table[int(cards), int(players)] = float(share)
    return table


def _expected(dist, bid, score):
    return sum(p * score.points(bid, k) for k, p in enumerate(dist))


class NetBot(Bot):
    """Decides instantly from the models alone, with no simulation."""

    def __init__(self, seed=0, models=None, bid_models=None, play_models=None):
        super().__init__(seed)
        self.models = load_models(models, bid_models, play_models)

    def bid(self, obs):
        options = obs.legal_actions()
        dists = self.models.bid_distributions(
            obs.hand, obs.trump_card, obs.n, obs.hand_size, obs.seat, obs.dealer,
            obs.bids, options)
        score = obs.rules.scoring
        return max(options, key=lambda b: _expected(dists[b], b, score))

    def play(self, obs):
        cards = obs.legal_actions()
        if len(cards) == 1:
            return cards[0]
        values = self.models.card_values(obs, obs.seat, obs.hand, cards)
        return cards[values.index(max(values))]


class HybridBot(SimulationBot):
    """Simulation with the models acting as a prior.

    Bids average the simulated and predicted trick odds, ``bid_blend`` being
    the model's share (0 bids from the simulation alone). For cards, the model
    shortlists the ``shortlist`` most promising options, and its estimate counts
    as ``prior_worlds`` extra simulated deals. Exact endgames ignore the model.

    ``blend_pivot`` (0 = off) scales both model weights by card count:
    ``(cards - 1) / (blend_pivot - 1)``, cards being the hand size for bids and
    the cards in hand for play. One card trusts the simulation alone, the pivot
    gives the fixed weights, and larger hands lean further on the model (the
    bid share is capped at 1).

    ``bid_table`` names a JSON file of fitted bid shares per hand size and
    table size (written by ``ohhell.learn.bid_blend_fit``). Where it has an
    entry, that replaces ``bid_blend`` and any ``blend_pivot`` scaling for
    bids; card play is untouched.

    ``cash`` (0 = off) overrides the search when we are on lead, still want
    tricks and hold a side-suit card no unseen card beats: lead the highest
    such card. 1 does so always, 2 only in over-called rounds, 3 unless the
    round is under-called. Exact endgames are left alone.

    ``value_bids`` names a folder holding ``value.npz`` from
    ``ohhell.learn.value_bid``: a network that predicts our trick count from
    a whole deal once every bid is known. In each deal the bid simulation
    samples, it is asked as well as (or, with ``value_playouts=0``, instead
    of) the play-out, and its averaged odds take ``value_share`` of the final
    bid odds; the rest is the usual simulation/network blend. Needs the
    native core.

    ``value_play`` names a folder holding ``value.npz`` from
    ``ohhell.learn.value_play``. Outside the exact endgame, each sampled deal
    plays our card, finishes the trick with the simulated population and
    asks that network for our expected points instead of playing the round
    out (``value_play_mode=value``), or averages the two (``mix``). The play
    network prior, shortlist and endgame are unchanged.
    """

    def __init__(self, seed=0, models=None, shortlist=4, prior_worlds=16,
                 bid_blend=0.5, inference=True, bid_models=None, play_models=None,
                 blend_pivot=0, cash=0, play_evidence=0, touching=0,
                 evaluator='mlp', bid_table=None, value_bids=None, value_share=1 / 3,
                 value_playouts=1, value_play=None, value_play_mode='value',
                 bid_calibration_file=None, bid_rank_file=None, **kwargs):
        super().__init__(seed=seed, inference=inference, **kwargs)
        if bid_calibration_file and value_bids:
            raise ValueError('Bid calibration is fitted for cheap rollouts, not full-deal value models.')
        if bid_rank_file and (bid_calibration_file or value_bids or self.confidence):
            raise ValueError('Bid ranking expects the ordinary hybrid forecast.')
        self.bid_calibration = bid_calibration.load_coefficients(bid_calibration_file) if bid_calibration_file else None
        self.bid_rank = bid_rank.load_weights(bid_rank_file) if bid_rank_file else None
        self.value_net = None
        if value_bids:
            if self.simulator is None:
                raise RuntimeError('value_bids needs the native simulator.')
            if not 0.0 < value_share <= 1.0:
                raise ValueError('value_share must be above 0 and at most 1.')
            from ..learn.value_bid import load_value
            self.value_net = load_value(value_bids)
            self.simulator.set_value(self.value_net)
            self.value_share = float(value_share)
            self.value_playouts = bool(value_playouts)
        self.play_value = None
        if value_play:
            if self.simulator is None:
                raise RuntimeError('value_play needs the native simulator.')
            if value_play_mode not in ('value', 'mix'):
                raise ValueError("value_play_mode must be 'value' or 'mix'.")
            from ..learn.value_play import load_value as load_play_value
            self.play_value = load_play_value(value_play)
            self.simulator.set_play_value(self.play_value)
            self.value_play_mode = value_play_mode
        self.bid_table = load_bid_table(str(bid_table)) if bid_table else None
        if blend_pivot and blend_pivot <= 1:
            raise ValueError('blend_pivot must be 0 (off) or greater than 1.')
        self.models = load_models(models, bid_models, play_models)
        self.shortlist = shortlist
        self.prior_worlds = prior_worlds
        self.bid_blend = bid_blend
        self.blend_pivot = blend_pivot
        if cash not in (0, 1, 2, 3):
            raise ValueError('cash must be 0, 1, 2 or 3.')
        self.cash = cash
        self.play_evidence = bool(play_evidence)
        self.touching = bool(touching)
        if evaluator not in ('mlp', 'compact'):
            raise ValueError('evaluator must be mlp or compact.')
        self.play_evaluator = None  # default follows subclasses replacing self.models
        if evaluator == 'compact':
            from ..learn.rollout import RolloutPolicy, load_network
            self.play_evaluator = RolloutPolicy(load_network(kwargs.get('rollout_models')))

    def sure_side_lead(self, obs):
        """The card ``cash`` would lead here, or None."""
        if obs.lead_suit != NO_SUIT or obs.bids[obs.seat] <= obs.tricks_won[obs.seat]:
            return None
        call = table_call(obs.bids, obs.tricks_won, obs.hand.bit_count())
        if (self.cash == 2 and call <= 0) or (self.cash == 3 and call < 0):
            return None
        trump = obs.trump_suit
        unseen = ~(obs.played | obs.hand | (1 << obs.trump_card))
        sure = [card for card in cards_in(obs.hand)
                if EFFECTIVE_SUIT[trump][card] != trump and not OUTRANKS[trump][card] & unseen]
        return max(sure, key=lambda card: (RANK_OF[card], -card)) if sure else None

    def model_scale(self, cards):
        """The multiplier on the model's weight at this card count."""
        if not self.blend_pivot:
            return 1.0
        return max(0.0, (cards - 1) / (self.blend_pivot - 1))

    def bid_share(self, hand_size, players):
        """The bid network's share of the blended trick odds."""
        if self.bid_table is not None and (hand_size, players) in self.bid_table:
            return self.bid_table[hand_size, players]
        return min(1.0, self.bid_blend * self.model_scale(hand_size))

    def _analyse_bids_value(self, obs):
        options = self._bid_options(obs)
        simulated, valued, samples = self.simulator.analyse_bids_value(
            obs, options, self.bid_worlds, self._time_left(), self.value_playouts)
        self.last_bid_samples = samples
        predicted = self.models.bid_distributions(
            obs.hand, obs.trump_card, obs.n, obs.hand_size, obs.seat, obs.dealer,
            obs.bids, options)
        blend = self.bid_share(obs.hand_size, obs.n) if self.value_playouts else 1.0
        share = self.value_share
        distributions = {
            bid: [(1.0 - share) * ((1.0 - blend) * s + blend * p) + share * v
                  for s, p, v in zip(simulated[bid], predicted[bid], valued[bid])]
            for bid in options
        }
        score = obs.rules.scoring
        return BidAnalysis(distributions, {
            bid: _expected(dist, bid, score) for bid, dist in distributions.items()
        }, samples)

    def analyse_bids(self, obs):
        if self.value_net is not None:
            return self._analyse_bids_value(obs)
        simulated = super().analyse_bids(obs)
        if self.bid_calibration is not None and not obs.blind:
            adjusted = {bid: bid_calibration.adjust(dist, obs.hand_size, bid, self.bid_calibration)
                        for bid, dist in simulated.distributions.items()}
            score = obs.rules.scoring
            simulated = BidAnalysis(adjusted, {bid: _expected(dist, bid, score)
                                              for bid, dist in adjusted.items()}, simulated.samples)
        blend = self.bid_share(obs.hand_size, obs.n)
        if not blend:  # bids from the simulation alone
            return simulated
        predicted = self.models.bid_distributions(
            obs.hand, obs.trump_card, obs.n, obs.hand_size, obs.seat, obs.dealer,
            obs.bids, list(simulated.distributions))
        distributions = {
            bid: [(1.0 - blend) * s + blend * p for s, p in zip(sim, predicted[bid])]
            for bid, sim in simulated.distributions.items()
        }
        score = obs.rules.scoring
        return BidAnalysis(distributions, {
            bid: _expected(dist, bid, score) for bid, dist in distributions.items()
        })

    def bid(self, obs):
        if self.bid_rank is None or obs.blind or obs.hand_size < bid_rank.MIN_HAND:
            return super().bid(obs)
        analysis = self.analyse_bids(obs)
        return bid_rank.pick(analysis.values, analysis.distributions, obs.hand_size,
                             obs.n, sum(b != NO_BID for b in obs.bids), self.bid_rank)

    def play(self, obs):
        cards = obs.legal_actions()
        if len(cards) == 1:
            return cards[0]
        remaining = obs.hand.bit_count() * obs.n - obs.trick_size
        if remaining <= self.endgame_limit(obs.n):
            return super().play(obs)
        if self.cash:
            sure = self.sure_side_lead(obs)
            if sure is not None:
                return sure

        evaluator = self.play_evaluator or self.models
        predicted = dict(zip(cards, evaluator.card_values(
            obs, obs.seat, obs.hand, cards)))
        ranked = sorted(cards, key=predicted.get, reverse=True)
        if self.touching:
            from .components import touching_groups
            ranked = [group[0] for group in touching_groups(obs, ranked)]
        kept = ranked[:self.shortlist]
        if self.play_value is not None:
            means, valued = self.simulator.evaluate_cards_value(
                obs, kept, self.worlds, self._time_left(), self.value_play_mode == 'mix')
            self.last_card_worlds = self.simulator.last_card_worlds
            simulated = (valued if self.value_play_mode == 'value'
                         else {c: 0.5 * (means[c] + valued[c]) for c in kept})
        else:
            simulated = self.evaluate_cards(obs, kept)
        weight = self.prior_worlds * self.model_scale(obs.hand.bit_count())
        count = self.last_card_worlds if self.play_evidence else self.worlds
        return max(kept, key=lambda c: (count * simulated[c] + weight * predicted[c])
                   / (count + weight))
