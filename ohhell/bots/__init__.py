"""Bot registry. A bot is named by a spec like ``simulation:worlds=16,inference=1``."""

from dataclasses import dataclass

from .base import Bot, RandomBot
from .heuristic import HeuristicBot


@dataclass(frozen=True)
class Version:
    number: int
    name: str
    spec: str

    @property
    def title(self):
        return f'v{self.number} {self.name}'


# Every promoted bot, oldest first. Add a line when a candidate wins the
# tournament. The last one is the bot the game uses. Only the current
# version's weights are kept, in ohhell/models/; v1 (Hybrid, the same search
# with Ace-high networks) is recorded in BOT_EXPERIMENTS.md.
VERSIONS = (
    # v1's hybrid search with networks retrained under bowers, then improved
    # by self-play (ohhell/learn/selfplay.py). Passed main and confirmation.
    Version(2, 'Self-play bowers nets', 'hybrid:field=previous,time_ms=2'),
)
DEFAULT_SPEC = VERSIONS[-1].spec


def version_title(spec, otherwise):
    """The registered title of a spec, or ``otherwise`` if it was never promoted."""
    return next((v.title for v in VERSIONS if v.spec == spec), otherwise)


def _simulation(**kwargs):
    from .simulation import SimulationBot
    return SimulationBot(**kwargs)


def _inference(**kwargs):
    from .simulation import SimulationBot
    kwargs.setdefault('inference', True)
    return SimulationBot(**kwargs)


def _kind(name):
    def build(seed=0):
        from .population import KindBot
        return KindBot(seed=seed, kind=name)
    return build


def _mixed(seed=0):
    from .population import mixed_bot
    return mixed_bot(seed)


def _net(**kwargs):
    from .learned import NetBot
    return NetBot(**kwargs)


def _hybrid(**kwargs):
    from .learned import HybridBot
    return HybridBot(**kwargs)


def _adaptive(**kwargs):
    from .adaptive import AdaptiveBot
    return AdaptiveBot(**kwargs)


def _evidence(**kwargs):
    from .evidence import EvidenceBot
    return EvidenceBot(**kwargs)


def _planning(**kwargs):
    from .planning import PlanningBot
    return PlanningBot(**kwargs)


def _seat_bids(**kwargs):
    from .seat_bids import SeatBidsBot
    return SeatBidsBot(**kwargs)


def _ranges(**kwargs):
    from .ranges import RangesBot
    return RangesBot(**kwargs)


def _nnue(**kwargs):
    from .nnue import NNUEBot
    return NNUEBot(**kwargs)


REGISTRY = {
    'random': RandomBot,
    'heuristic': HeuristicBot,
    'skilled': _kind('skilled'),
    'novice': _kind('novice'),
    'casual': _kind('casual'),
    'mixed': _mixed,
    'simulation': _simulation,
    'inference': _inference,
    'net': _net,
    'hybrid': _hybrid,
    'adaptive': _adaptive,
    'evidence': _evidence,
    'planning': _planning,
    'nnue': _nnue,
    'seatbids': _seat_bids,
    'ranges': _ranges,
}


def _parse_value(text):
    for convert in (int, float):
        try:
            return convert(text)
        except ValueError:
            pass
    return text


def make_bot(spec, seed=0):
    name, _, args = spec.partition(':')
    kwargs = {}
    if args:
        for pair in args.split(','):
            key, _, value = pair.partition('=')
            kwargs[key] = _parse_value(value)
    return REGISTRY[name](seed=seed, **kwargs)
