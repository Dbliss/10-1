"""Optional compiled core for the simulation bots.

Build it with ``python -m ohhell.native``. Bots require the library unless
explicitly configured with ``fast=0`` for the Python reference implementation.
"""

import ctypes
import hashlib
import platform
import shutil
import subprocess
from pathlib import Path

from ..opponents import FIELDS, KINDS

MAX_SEATS = 8
EST_SIDE = 65   # len(estimate.SIDE); core.c must agree
EST_TRUMP = 57  # len(estimate.TRUMP)
HERE = Path(__file__).parent
SOURCE = HERE / 'core.c'
# Windows locks a loaded DLL. Build a source-specific file so an existing
# tournament can keep using its own core while a new candidate is compiled.
_SOURCE_TAG = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16]
# In the browser (Pyodide) the core is a WebAssembly side module built
# ahead of time by build_wasm(); its name also carries the source tag.
WASM = platform.system() == 'Emscripten'
LIBRARY = HERE / (f'core_{_SOURCE_TAG}.dll' if platform.system() == 'Windows'
                  else f'core_{_SOURCE_TAG}.wasm.so' if WASM else 'core.so')


class _Obs(ctypes.Structure):
    _fields_ = [
        ('n', ctypes.c_int), ('hand_size', ctypes.c_int), ('trump_card', ctypes.c_int),
        ('dealer', ctypes.c_int), ('seat', ctypes.c_int), ('phase', ctypes.c_int),
        ('lead_suit', ctypes.c_int), ('best_card', ctypes.c_int), ('best_seat', ctypes.c_int),
        ('trick_size', ctypes.c_int), ('to_act', ctypes.c_int), ('leader', ctypes.c_int),
        ('bids', ctypes.c_int * MAX_SEATS), ('tricks_won', ctypes.c_int * MAX_SEATS),
        ('voids', ctypes.c_int * MAX_SEATS),
        ('hand', ctypes.c_uint64), ('played', ctypes.c_uint64), ('table', ctypes.c_uint64),
        ('history_len', ctypes.c_int),
        ('history_seat', ctypes.c_int * 52), ('history_card', ctypes.c_int * 52),
    ]


class _Kind(ctypes.Structure):
    _fields_ = [(name, ctypes.c_double) for name in (
        'card_noise', 'bid_one', 'bid_two', 'bid_sigma', 'play_doubt')]


class _RolloutNet(ctypes.Structure):
    _fields_ = [(name, ctypes.c_float * size) for name, size in (
        ('w0', 1024), ('w1', 512), ('w2', 16), ('b0', 32), ('b1', 16), ('b2', 1))]


class _ValueNet(ctypes.Structure):
    _fields_ = [(name, ctypes.POINTER(ctypes.c_float))
                for name in ('w0', 'b0', 'w1', 'b1', 'w2', 'b2')] + [
        ('inputs', ctypes.c_int), ('h1', ctypes.c_int), ('h2', ctypes.c_int),
        ('outputs', ctypes.c_int)]


class ValueNet:
    """A value_bid network held for the C core (keeps its arrays alive)."""

    def __init__(self, model):
        import numpy as np
        if len(model.weights) != 3:
            raise ValueError('The native value network needs exactly two hidden layers.')
        self.arrays = [np.ascontiguousarray(a, dtype=np.float32) for pair in
                       zip(model.weights, model.biases) for a in pair]
        pointer = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        w0, b0, w1, b1, w2, b2 = self.arrays
        self.struct = _ValueNet(pointer(w0), pointer(b0), pointer(w1), pointer(b1),
                                pointer(w2), pointer(b2), w0.shape[0], w0.shape[1],
                                w1.shape[1], w2.shape[1])


def value_odds(net, hands, trump_card, hand_size, dealer, bids, hero, hero_type=0):
    """One deal's trick odds from the C value network (for checking parity)."""
    n = len(hands)
    out = (ctypes.c_double * (hand_size + 1))()
    library.oh_value_odds(ctypes.byref(net.struct), (ctypes.c_uint64 * MAX_SEATS)(*hands),
                          n, trump_card, hand_size, dealer, (ctypes.c_int * MAX_SEATS)(*bids),
                          hero, hero_type, out)
    return list(out)


class _Params(ctypes.Structure):
    _fields_ = [
        ('worlds', ctypes.c_int), ('endgame_cards', ctypes.c_int),
        ('inference', ctypes.c_int), ('oversample', ctypes.c_int),
        ('points_per_trick', ctypes.c_int), ('made_zero', ctypes.c_int),
        ('penalty_per_trick', ctypes.c_int),
        ('weights', ctypes.c_double * len(KINDS)),
        ('kinds', _Kind * len(KINDS)),
        ('time_limit', ctypes.c_double),
        ('react', ctypes.c_int),
        ('estimate', ctypes.c_int),
        ('est_side', ctypes.c_double * EST_SIDE),
        ('est_trump', ctypes.c_double * EST_TRUMP),
        ('est_spread', ctypes.c_double * 3),
        ('style', ctypes.c_int), ('style_all', ctypes.c_int),
        ('dedupe', ctypes.c_int),
        ('rollout', ctypes.POINTER(_RolloutNet)),
        ('rollout_seats', ctypes.c_int), ('policy_inference', ctypes.c_int),
        ('policy_temperature', ctypes.c_double), ('stop_z', ctypes.c_double),
        ('stop_min', ctypes.c_int), ('seat_priors', ctypes.c_int),
        ('seat_weights', (ctypes.c_double * len(KINDS)) * MAX_SEATS),
        ('bid_only', ctypes.c_int),
        ('budget', ctypes.c_int),
        ('endgame_noise', ctypes.c_int),
    ]


def build():
    if platform.system() == 'Windows' and LIBRARY.exists():
        return LIBRARY
    compiler = shutil.which('gcc') or shutil.which('cc')
    if compiler is None:
        raise RuntimeError('No C compiler found. Install gcc or cc; fast=0 selects the Python oracle.')
    command = [compiler, '-O3', '-march=native', '-flto', '-shared', '-fPIC',
               '-o', str(LIBRARY), str(SOURCE), '-lm']
    if platform.system() == 'Windows':
        command[-1:-1] = ['-static-libgcc']
    subprocess.run(command, check=True)
    return LIBRARY


def build_wasm(emcc='emcc', out_dir=HERE):
    """Compile core.c for Pyodide (Emscripten side module) as core_TAG.wasm.so.

    The Emscripten version must match the Pyodide release the site loads
    (``frontend/src/pyodide.ts``).
    """
    target = Path(out_dir) / f'core_{_SOURCE_TAG}.wasm.so'
    subprocess.run([emcc, '-O3', '-fPIC', '-sSIDE_MODULE=1', '-o', str(target), str(SOURCE)],
                   check=True)
    return target


def _load():
    if not LIBRARY.exists():
        return None
    if (platform.system() != 'Windows' and not WASM
            and LIBRARY.stat().st_mtime < SOURCE.stat().st_mtime):
        return None
    try:
        library = ctypes.CDLL(str(LIBRARY))
    except OSError:
        return None
    library.oh_choose_card.restype = ctypes.c_int
    library.oh_choose_card.argtypes = [
        ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    library.oh_choose_card_style.restype = ctypes.c_int
    library.oh_choose_card_style.argtypes = library.oh_choose_card.argtypes + [
        ctypes.c_int, ctypes.c_int]
    library.oh_expected_tricks.restype = ctypes.c_double
    library.oh_expected_tricks.argtypes = [
        ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    library.oh_expected_tricks2.restype = ctypes.c_double
    library.oh_expected_tricks2.argtypes = [
        ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(_Params), ctypes.POINTER(ctypes.c_double)]
    library.oh_heuristic_bid.restype = ctypes.c_int
    library.oh_heuristic_bid.argtypes = [
        ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(_Params)]
    library.oh_plan_cards.restype = ctypes.c_int
    library.oh_plan_cards.argtypes = [
        ctypes.POINTER(_Obs), ctypes.POINTER(_Params), ctypes.POINTER(ctypes.c_int),
        ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint64)]
    library.oh_sample_worlds.restype = ctypes.c_int
    library.oh_sample_worlds.argtypes = [
        ctypes.POINTER(_Obs), ctypes.POINTER(_Params), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint64)]
    library.oh_score_bids.restype = ctypes.c_int
    library.oh_score_bids.argtypes = [
        ctypes.POINTER(_Obs), ctypes.POINTER(_Params), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint64)]
    library.oh_value_odds.restype = None
    library.oh_value_odds.argtypes = [
        ctypes.POINTER(_ValueNet), ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double)]
    library.oh_analyse_bids_value.restype = ctypes.c_int
    library.oh_analyse_bids_value.argtypes = [
        ctypes.POINTER(_Obs), ctypes.POINTER(_Params), ctypes.c_int, ctypes.POINTER(ctypes.c_int),
        ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(_ValueNet), ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
    library.oh_vplay_points.restype = ctypes.c_double
    library.oh_vplay_points.argtypes = [
        ctypes.POINTER(_ValueNet), ctypes.POINTER(_Obs), ctypes.POINTER(_Params),
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_int]
    library.oh_evaluate_cards_value.restype = ctypes.c_int
    library.oh_evaluate_cards_value.argtypes = [
        ctypes.POINTER(_Obs), ctypes.POINTER(_Params), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(_ValueNet), ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
    library.oh_last_card_worlds.restype = ctypes.c_long
    library.oh_last_card_worlds.argtypes = []
    library.oh_last_card_playouts.restype = ctypes.c_long
    library.oh_last_card_playouts.argtypes = []
    library.oh_rollout_values.restype = None
    library.oh_rollout_values.argtypes = [ctypes.POINTER(_Obs), ctypes.POINTER(_Params),
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    library.oh_rollout_scores.restype = None
    library.oh_rollout_scores.argtypes = [ctypes.POINTER(_Obs), ctypes.POINTER(_Params),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double)]
    library.oh_solve_noisy_known.restype = None
    library.oh_solve_noisy_known.argtypes = [ctypes.POINTER(_Obs), ctypes.POINTER(_Params),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    library.oh_world_log_weight.restype = ctypes.c_double
    library.oh_world_log_weight.argtypes = [ctypes.POINTER(_Obs), ctypes.POINTER(_Params),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int)]
    return library


library = _load()
available = library is not None


class Simulator:
    """Runs a bot's simulations in C. One per bot, as it carries the random state."""

    def __init__(self, seed, inference, oversample, field, endgame_cards, scoring, react=False,
                 estimate=1, bid_sigma=None, style=0, style_all=False, dedupe=False):
        self.rng = ctypes.c_uint64((seed * 2 + 1) * 0x9E3779B97F4A7C15 % (1 << 64) or 1)
        self.params = _Params(
            0, endgame_cards, int(inference), oversample,
            scoring.points_per_trick, scoring.made_zero, scoring.penalty_per_trick,
            (ctypes.c_double * len(KINDS))(*FIELDS[field]),
            (_Kind * len(KINDS))(*[
                _Kind(k.card_noise, k.bid_one, k.bid_two, bid_sigma or k.bid_sigma,
                      k.play_doubt)
                for k in KINDS]),
            0.0, int(react), int(estimate),
        )
        self.params.est_spread[:] = (1.0, 0.0, 0.0)
        self.params.style = int(style)
        self.params.style_all = int(style_all)
        self.params.dedupe = int(dedupe)
        self.last_card_worlds = self.last_card_playouts = 0
        if estimate in (2, 3):
            from ..estimate import _coefficients
            side, trump, spread = _coefficients(estimate)
            if len(side) != EST_SIDE or len(trump) != EST_TRUMP:
                raise ValueError('Trick estimate weights do not match the native core.')
            self.params.est_side[:] = side
            self.params.est_trump[:] = trump
            self.params.est_spread[:] = spread
        elif estimate != 1:
            raise ValueError('estimate must be 1, 2 or 3.')
        self.obs = _Obs()
        # Exact-search threshold: endgame_table maps table sizes to their own.
        self.endgame_cards = endgame_cards
        self.endgame_table = {}
        # Fixed deal budgets (0: off) for untimed, exactly repeatable search.
        self.card_budget = self.bid_budget = self.endgame_budget = 0

    def _card_budget(self, obs):
        self.params.endgame_cards = self.endgame_limit(obs.n)
        remaining = obs.hand.bit_count() * obs.n - obs.trick_size
        return self.endgame_budget if remaining <= self.params.endgame_cards else self.card_budget

    def endgame_limit(self, players):
        """Cards left at or below which card play is searched exactly."""
        return self.endgame_table.get(players, self.endgame_cards)

    def set_rollout(self, model, seats=1, inference=False, temperature=10):
        self.rollout_network = _RolloutNet()
        for i, weights in enumerate(model.weights):
            getattr(self.rollout_network, f'w{i}')[:] = weights.ravel().tolist()
        for i, bias in enumerate(model.biases):
            getattr(self.rollout_network, f'b{i}')[:] = bias.tolist()
        self.params.rollout = ctypes.pointer(self.rollout_network)
        self.params.rollout_seats = seats
        self.params.policy_inference = int(inference)
        self.params.policy_temperature = temperature

    def set_seat_weights(self, weights):
        self.params.seat_priors = 1
        for seat, row in enumerate(weights):
            self.params.seat_weights[seat][:] = row

    def rollout_values(self, obs, cards):
        if not self.params.rollout:
            raise ValueError('No rollout network configured')
        self._load_obs(obs)
        chosen = (ctypes.c_int * len(cards))(*cards)
        values = (ctypes.c_double * len(cards))()
        library.oh_rollout_values(ctypes.byref(self.obs), ctypes.byref(self.params),
                                  chosen, len(cards), values)
        return dict(zip(cards, values))

    def rollout_scores(self, obs, hands, cards):
        """Known-deal parity diagnostic, deterministic play without exact search."""
        self._load_obs(obs)
        dealt = (ctypes.c_uint64 * MAX_SEATS)(*hands)
        chosen = (ctypes.c_int * len(cards))(*cards)
        values = (ctypes.c_double * len(cards))()
        library.oh_rollout_scores(ctypes.byref(self.obs), ctypes.byref(self.params),
                                 dealt, chosen, len(cards), values)
        return dict(zip(cards, values))

    def solve_noisy_scores(self, obs, hands, noise, cards):
        """Known-deal parity diagnostic for the noise-aware exact search."""
        self._load_obs(obs)
        dealt = (ctypes.c_uint64 * MAX_SEATS)(*hands)
        chances = (ctypes.c_double * MAX_SEATS)(*noise)
        chosen = (ctypes.c_int * len(cards))(*cards)
        values = (ctypes.c_double * len(cards))()
        library.oh_solve_noisy_known(ctypes.byref(self.obs), ctypes.byref(self.params),
                                     dealt, chances, chosen, len(cards), values)
        return dict(zip(cards, values))

    def world_log_weight(self, obs, hands, kinds):
        """Inference parity diagnostic on a supplied deal and kind indices."""
        self._load_obs(obs)
        dealt = (ctypes.c_uint64 * MAX_SEATS)(*hands)
        types = (ctypes.c_int * MAX_SEATS)(*kinds)
        return library.oh_world_log_weight(ctypes.byref(self.obs), ctypes.byref(self.params),
                                           dealt, types)

    def expected_tricks2(self, hand, trump_card, n, hand_size, position):
        """The native version 2 estimate, as (total, per-card list)."""
        probs = (ctypes.c_double * 13)()
        total = library.oh_expected_tricks2(hand, trump_card, n, hand_size, position,
                                            ctypes.byref(self.params), probs)
        return total, list(probs[:hand_size])

    def heuristic_bid(self, hand, trump_card, n, hand_size, seat, dealer):
        """The bid this simulator imagines for an unbid seat (before noise)."""
        return library.oh_heuristic_bid(hand, trump_card, n, hand_size, seat, dealer,
                                        ctypes.byref(self.params))

    def _load_obs(self, o):
        scoring = o.rules.scoring
        self.params.points_per_trick = scoring.points_per_trick
        self.params.made_zero = scoring.made_zero
        self.params.penalty_per_trick = scoring.penalty_per_trick
        c = self.obs
        c.n, c.hand_size, c.trump_card, c.dealer, c.seat, c.phase = (
            o.n, o.hand_size, o.trump_card, o.dealer, o.seat, o.phase)
        c.lead_suit, c.best_card, c.best_seat, c.trick_size, c.to_act, c.leader = (
            o.lead_suit, o.best_card, o.best_seat, o.trick_size, o.to_act, o.leader)
        c.bids[:o.n] = o.bids
        c.tricks_won[:o.n] = o.tricks_won
        c.voids[:o.n] = o.voids
        c.hand, c.played, c.table = o.hand, o.played, o.table
        c.history_len = len(o.history)
        if o.history:
            seats, cards = zip(*o.history)
            c.history_seat[:len(seats)] = seats
            c.history_card[:len(cards)] = cards

    def evaluate_cards(self, obs, cards, worlds, time_left=0.0):
        """Average points for this seat after each card.

        With ``time_left`` seconds, keeps drawing blocks of ``worlds`` until then.
        """
        self._load_obs(obs)
        self.params.worlds = worlds
        self.params.time_limit = time_left
        self.params.budget = self._card_budget(obs)
        chosen = (ctypes.c_int * len(cards))(*cards)
        means = (ctypes.c_double * len(cards))()
        if not library.oh_evaluate_cards(
                ctypes.byref(self.obs), ctypes.byref(self.params), chosen, len(cards),
                means, ctypes.byref(self.rng)):
            raise ValueError('No deal is consistent with this observation.')
        self.last_card_worlds = library.oh_last_card_worlds()
        self.last_card_playouts = library.oh_last_card_playouts()
        return dict(zip(cards, means))

    def analyse_bids(self, obs, options, worlds, time_left=0.0):
        """For each bid, the fraction of deals taking exactly k tricks."""
        self._load_obs(obs)
        self.params.worlds = worlds
        self.params.time_limit = time_left
        self.params.budget = self.bid_budget
        width = obs.hand_size + 1
        chosen = (ctypes.c_int * len(options))(*options)
        counts = (ctypes.c_double * (len(options) * width))()
        total = library.oh_analyse_bids(
            ctypes.byref(self.obs), ctypes.byref(self.params),
            int(obs.rules.dealer_restriction), chosen, len(options), counts,
            ctypes.byref(self.rng))
        if not total:
            raise ValueError('No deal is consistent with this observation.')
        self.last_bid_samples = total
        return {
            bid: [c / total for c in counts[i * width:(i + 1) * width]]
            for i, bid in enumerate(options)
        }

    def set_value(self, model):
        self.value_network = ValueNet(model)

    def analyse_bids_value(self, obs, options, worlds, time_left=0.0, playouts=True):
        """(play-out odds, value-network odds, deals used) for each bid."""
        self._load_obs(obs)
        self.params.worlds = worlds
        self.params.time_limit = time_left
        self.params.budget = self.bid_budget
        width = obs.hand_size + 1
        chosen = (ctypes.c_int * len(options))(*options)
        counts = (ctypes.c_double * (len(options) * width))()
        valued = (ctypes.c_double * (len(options) * width))()
        total = library.oh_analyse_bids_value(
            ctypes.byref(self.obs), ctypes.byref(self.params),
            int(obs.rules.dealer_restriction), chosen, len(options), counts, valued,
            ctypes.byref(self.value_network.struct), int(playouts), ctypes.byref(self.rng))
        if total < 0:
            raise ValueError('The value network does not fit the native core.')
        if not total:
            raise ValueError('No deal is consistent with this observation.')
        self.last_bid_samples = total
        split = lambda c: {bid: [x / total for x in c[i * width:(i + 1) * width]]
                           for i, bid in enumerate(options)}
        return split(counts), split(valued), total

    def set_play_value(self, model):
        self.play_value_network = ValueNet(model)

    def play_value_points(self, obs, hands, hero_type=0):
        """The card value network's expected points for obs.seat at this trick start."""
        self._load_obs(obs)
        return library.oh_vplay_points(
            ctypes.byref(self.play_value_network.struct), ctypes.byref(self.obs),
            ctypes.byref(self.params), (ctypes.c_uint64 * MAX_SEATS)(*hands), hero_type)

    def evaluate_cards_value(self, obs, cards, worlds, time_left=0.0, playouts=True):
        """(play-out means, value-network means) of this seat's points after each card."""
        self._load_obs(obs)
        self.params.worlds = worlds
        self.params.time_limit = time_left
        self.params.budget = self._card_budget(obs)
        chosen = (ctypes.c_int * len(cards))(*cards)
        means = (ctypes.c_double * len(cards))()
        valued = (ctypes.c_double * len(cards))()
        total = library.oh_evaluate_cards_value(
            ctypes.byref(self.obs), ctypes.byref(self.params), chosen, len(cards), means,
            valued, ctypes.byref(self.play_value_network.struct), int(playouts),
            ctypes.byref(self.rng))
        if total < 0:
            raise ValueError('The card value network does not fit the native core.')
        if not total:
            raise ValueError('No deal is consistent with this observation.')
        self.last_card_worlds = self.last_card_playouts = total
        return dict(zip(cards, means)), dict(zip(cards, valued))

    def sample_worlds(self, obs, count):
        """Draw deals as ``analyse_bids`` would: (hands, kinds), each an
        integer array of shape (count, MAX_SEATS)."""
        import numpy as np
        self._load_obs(obs)
        hands = np.zeros((count, MAX_SEATS), dtype=np.uint64)
        kinds = np.zeros((count, MAX_SEATS), dtype=np.int32)
        done = library.oh_sample_worlds(
            ctypes.byref(self.obs), ctypes.byref(self.params), count,
            hands.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            kinds.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), ctypes.byref(self.rng))
        if not done:
            raise ValueError('No deal is consistent with this observation.')
        return hands[:done], kinds[:done]

    def score_bids(self, obs, hands, kinds, bids, counts):
        """Play out every world after every option with the given bids.

        ``bids`` has shape (worlds, options, MAX_SEATS); tricks taken by this
        seat are added to ``counts``, shape (options, hand_size + 1).
        """
        import numpy as np
        self._load_obs(obs)
        hands = np.ascontiguousarray(hands, dtype=np.uint64)
        kinds = np.ascontiguousarray(kinds, dtype=np.int32)
        bids = np.ascontiguousarray(bids, dtype=np.int32)
        assert counts.flags.c_contiguous and counts.dtype == np.float64
        library.oh_score_bids(
            ctypes.byref(self.obs), ctypes.byref(self.params), bids.shape[1], len(hands),
            hands.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            kinds.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            bids.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            counts.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.byref(self.rng))

    def plan_cards(self, obs, cards, iterations, exploration, batch, time_left=0.0):
        """Search a shared visible-history tree across sampled deals."""
        if not 1 <= batch <= 1024 or iterations < 1:
            raise ValueError('Positive iterations and a batch from 1 to 1024 required.')
        self._load_obs(obs)
        self.params.worlds = iterations
        self.params.time_limit = time_left
        chosen = (ctypes.c_int * len(cards))(*cards)
        totals = (ctypes.c_double * len(cards))()
        visits = (ctypes.c_int * len(cards))()
        diagnostics = (ctypes.c_int * 2)()
        done = library.oh_plan_cards(
            ctypes.byref(self.obs), ctypes.byref(self.params), chosen, len(cards),
            exploration, batch, totals, visits, diagnostics, ctypes.byref(self.rng))
        if not done:
            raise ValueError('No deal is consistent with this observation.')
        return {
            'visits': dict(zip(cards, visits)), 'totals': dict(zip(cards, totals)),
            'iterations': done, 'nodes': diagnostics[0], 'deep_visits': diagnostics[1],
        }
