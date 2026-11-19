"""Package the game's Python for the browser build (GitHub Pages).

    python browser_build.py [--emcc PATH]

Writes ``frontend/public/py/app-HASH.zip`` (the Python the game imports,
the trained models, and the C core compiled to WebAssembly) and
``frontend/public/py/manifest.json`` naming it, which the Web Worker reads
(``frontend/src/worker/engine.ts``). The archive name changes with its
contents, so browsers never mix an old cached archive with a new page.

The WebAssembly core is built with Emscripten if it is missing
(``ohhell/native build_wasm``; the version must match the Pyodide release).
"""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / 'frontend' / 'public' / 'py'
# The modules the API imports at the top level, plus the worker's entry point.
ROOT_MODULES = ['app_api.py', 'browser_entry.py', 'game.py', 'review.py', 'bid_helper.py',
                'bid_game.py']


def files(core):
    for name in ROOT_MODULES:
        yield HERE / name
    for path in sorted((HERE / 'ohhell').rglob('*')):
        if '__pycache__' in path.parts or not path.is_file():
            continue
        relative = path.relative_to(HERE / 'ohhell')
        if path.suffix == '.py' or relative.parts[0] == 'models':
            yield path
    yield HERE / 'ohhell' / 'native' / 'core.c'  # its hash names the core to load
    yield core


def build(emcc):
    from ohhell import native
    core = native.HERE / f'core_{native._SOURCE_TAG}.wasm.so'
    if not core.exists():
        native.build_wasm(emcc)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files(core):
            info = zipfile.ZipInfo(path.relative_to(HERE).as_posix(), date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    data = buffer.getvalue()
    name = f'app-{hashlib.sha256(data).hexdigest()[:12]}.zip'
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob('app-*.zip'):
        old.unlink()
    (OUT / name).write_bytes(data)
    (OUT / 'manifest.json').write_text(json.dumps({'archive': name}) + '\n')
    return OUT / name, len(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--emcc', default='emcc', help='Emscripten compiler (5.0.3 for Pyodide 314.0.8)')
    path, size = build(parser.parse_args().emcc)
    print(f'wrote {path} ({size / 1e6:.2f} MB)')
