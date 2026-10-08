"""
Install Annoy (a PaCMAP dependency) built without -ffast-math.

Annoy publishes no wheels, so pip compiles it from source.  Its setup.py adds
-ffast-math on every platform except Windows; with current Apple clang (and
potentially other compilers) the resulting library returns only the queried
item itself from every nearest-neighbour search, which makes PaCMAP fail.
This script compiles Annoy with its other default flags and without
-ffast-math, then checks that the build returns neighbours.

Run it before ``pip install -r requirements.txt`` when setting up a build
environment (``make venv`` and the CI workflow do).  The version is read from
requirements.txt so the pin stays in one place.  Windows builds use MSVC, which
is not affected, so nothing is done there.

Usage:
    python install_annoy.py
"""
import os
import re
import subprocess
import sys
from pathlib import Path

# Annoy's own defaults for non-Windows platforms, minus -ffast-math.
COMPILER_ARGS = ['-D_CRT_SECURE_NO_WARNINGS', '-fpermissive', '-O3', '-fno-associative-math']

SELF_TEST = (
    "import numpy as np\n"
    "from annoy import AnnoyIndex\n"
    "x = np.random.default_rng(0).normal(size=(500, 7)).astype(np.float32)\n"
    "t = AnnoyIndex(7, 'euclidean')\n"
    "[t.add_item(i, x[i]) for i in range(len(x))]\n"
    "t.build(10)\n"
    "n = [len(t.get_nns_by_item(i, 20)) for i in range(len(x))]\n"
    "assert min(n) == 20, f'Annoy returned only {min(n)} neighbours'\n"
)


def pinned_requirement() -> str:
    text = (Path(__file__).parent / 'requirements.txt').read_text()
    match = re.search(r'^annoy\s*==\s*([\w.]+)', text, re.MULTILINE | re.IGNORECASE)
    return f'annoy=={match.group(1)}' if match else 'annoy'


def main() -> int:
    if os.name == 'nt':
        print('Windows builds use MSVC; Annoy needs no special flags.')
        return 0
    env = dict(os.environ, ANNOY_COMPILER_ARGS=','.join(COMPILER_ARGS))
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', '--no-binary', 'annoy', '--no-cache-dir',
         '--force-reinstall', '--no-deps', pinned_requirement()], env=env)
    subprocess.check_call([sys.executable, '-c', SELF_TEST])
    print('Annoy built and verified.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
