"""Bot registry. A bot is named by a spec like ``simulation:worlds=16,inference=1``."""

from .base import Bot, RandomBot
from .heuristic import HeuristicBot


def _simulation(**kwargs):
    from .simulation import SimulationBot
    return SimulationBot(**kwargs)


def _inference(**kwargs):
    from .simulation import SimulationBot
    kwargs.setdefault('inference', True)
    return SimulationBot(**kwargs)


def _net(**kwargs):
    from .learned import NetBot
    return NetBot(**kwargs)


def _hybrid(**kwargs):
    from .learned import HybridBot
    return HybridBot(**kwargs)


REGISTRY = {
    'random': RandomBot,
    'heuristic': HeuristicBot,
    'simulation': _simulation,
    'inference': _inference,
    'net': _net,
    'hybrid': _hybrid,
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
