"""Optional compiled core for the simulation bots.

Build it with ``python -m ohhell.native``. Without the library, the bots fall
back to the slower Python code.
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
LIBRARY = HERE / (f'core_{_SOURCE_TAG}.dll' if platform.system() == 'Windows' else 'core.so')


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
    ]


def build():
    if platform.system() == 'Windows' and LIBRARY.exists():
        return LIBRARY
    compiler = shutil.which('gcc') or shutil.which('cc')
    if compiler is None:
        raise RuntimeError('No C compiler found. The bots will use the Python code.')
    command = [compiler, '-O3', '-march=native', '-flto', '-shared', '-fPIC',
               '-o', str(LIBRARY), str(SOURCE), '-lm']
    if platform.system() == 'Windows':
        command[-1:-1] = ['-static-libgcc']
    subprocess.run(command, check=True)
    return LIBRARY


def _load():
    if not LIBRARY.exists():
        return None
    if platform.system() != 'Windows' and LIBRARY.stat().st_mtime < SOURCE.stat().st_mtime:
        return None
    try:
        library = ctypes.CDLL(str(LIBRARY))
    except OSError:
        return None
    library.oh_choose_card.restype = ctypes.c_int
    library.oh_choose_card.argtypes = [
        ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_int]
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
    return library


library = _load()
available = library is not None


class Simulator:
    """Runs a bot's simulations in C. One per bot, as it carries the random state."""

    def __init__(self, seed, inference, oversample, field, endgame_cards, scoring, react=False,
                 estimate=1, bid_sigma=None):
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
        chosen = (ctypes.c_int * len(cards))(*cards)
        means = (ctypes.c_double * len(cards))()
        if not library.oh_evaluate_cards(
                ctypes.byref(self.obs), ctypes.byref(self.params), chosen, len(cards),
                means, ctypes.byref(self.rng)):
            raise ValueError('No deal is consistent with this observation.')
        return dict(zip(cards, means))

    def analyse_bids(self, obs, options, worlds, time_left=0.0):
        """For each bid, the fraction of deals taking exactly k tricks."""
        self._load_obs(obs)
        self.params.worlds = worlds
        self.params.time_limit = time_left
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
