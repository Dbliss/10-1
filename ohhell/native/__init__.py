"""Optional compiled core for the simulation bots.

Build it with ``python -m ohhell.native``. Without the library, the bots fall
back to the slower Python code.
"""

import ctypes
import platform
import shutil
import subprocess
from pathlib import Path

from ..opponents import FIELDS, KINDS

MAX_SEATS = 8
HERE = Path(__file__).parent
SOURCE = HERE / 'core.c'
LIBRARY = HERE / ('core.dll' if platform.system() == 'Windows' else 'core.so')


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
    ]


def build():
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
    if LIBRARY.stat().st_mtime < SOURCE.stat().st_mtime:
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
    return library


library = _load()
available = library is not None


class Simulator:
    """Runs a bot's simulations in C. One per bot, as it carries the random state."""

    def __init__(self, seed, inference, oversample, field, endgame_cards, scoring, react=False):
        self.rng = ctypes.c_uint64((seed * 2 + 1) * 0x9E3779B97F4A7C15 % (1 << 64) or 1)
        self.params = _Params(
            0, endgame_cards, int(inference), oversample,
            scoring.points_per_trick, scoring.made_zero, scoring.penalty_per_trick,
            (ctypes.c_double * len(KINDS))(*FIELDS[field]),
            (_Kind * len(KINDS))(*[
                _Kind(k.card_noise, k.bid_one, k.bid_two, k.bid_sigma, k.play_doubt)
                for k in KINDS]),
            0.0, int(react),
        )
        self.obs = _Obs()

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
        return {
            bid: [c / total for c in counts[i * width:(i + 1) * width]]
            for i, bid in enumerate(options)
        }
