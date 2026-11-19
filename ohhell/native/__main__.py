"""python -m ohhell.native [--wasm [EMCC]]: build the core for this machine, or for the browser."""

import sys

from . import build, build_wasm

if len(sys.argv) > 1 and sys.argv[1] == '--wasm':
    print(f'built {build_wasm(*sys.argv[2:3])}')
else:
    print(f'built {build()}')
