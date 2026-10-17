"""
Install openTSNE built with OpenMP on macOS.

openTSNE's macOS wheels are compiled without OpenMP, so its affinity,
positive-gradient and Barnes-Hut code run on one thread whatever n_jobs is
(an M-series Mac took 15.5 s on 1 thread and 14.6 s on 8 for a 5,000-event
fit).  The Windows and Linux wheels already use OpenMP, so nothing is done
there.

This script compiles openTSNE from source against Homebrew's libomp (header
and library), the same libomp.dylib the SOM and AutoSpectral optimisation
kernels link, so in a source checkout all three share one OpenMP runtime.
scikit-learn's own copy (sklearn/.dylibs) cannot be used: Apple clang emits
calls such as __kmpc_dispatch_deinit that it does not export, and openTSNE
then fails to import ("symbol not found in flat namespace").

The build is then checked:
  * openTSNE's _tsne extension links libomp;
  * every OpenMP entry point the extensions call is exported by that libomp;
  * openTSNE imports and loads that libomp;
  * a Barnes-Hut fit on all cores is clearly faster than on one.

Run it after ``pip install -r requirements.txt`` and ``brew install libomp``.
The version is read from requirements.txt.

Usage:
    python install_opentsne.py
"""
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

MIN_SPEEDUP = 1.25

SELF_TEST = r'''
import ctypes, os, sys, time
import numpy as np
import openTSNE

libc = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
libc._dyld_get_image_name.restype = ctypes.c_char_p
images = [libc._dyld_get_image_name(i).decode() for i in range(libc._dyld_image_count())]
omp = sorted({os.path.realpath(p) for p in images if 'libomp' in os.path.basename(p)})
print('libomp loaded:', omp)
assert os.path.realpath(sys.argv[2]) in omp, f'openTSNE did not load {sys.argv[2]}'

rng = np.random.default_rng(0)
centres = rng.normal(scale=4.0, size=(8, 7))
x = (centres[rng.integers(0, 8, 5000)] + rng.normal(size=(5000, 7))).astype(np.float32)

def fit(n_jobs):
    t = time.perf_counter()
    openTSNE.TSNE(n_jobs=n_jobs, negative_gradient_method='bh', n_iter=100,
                  early_exaggeration_iter=50, random_state=42).fit(x)
    return time.perf_counter() - t

fit(1)
one, many = fit(1), fit(-1)
speedup = one / many
print(f'Barnes-Hut 5,000 events: {one:.1f}s on 1 thread, {many:.1f}s on {os.cpu_count()}: '
      f'{speedup:.2f}x')
assert speedup >= float(sys.argv[1]), f'no speed-up from threads ({speedup:.2f}x)'
'''


def pinned_requirement() -> str:
    text = (Path(__file__).parent / 'requirements.txt').read_text()
    match = re.search(r'^opentsne\s*==\s*([\w.]+)', text, re.MULTILINE | re.IGNORECASE)
    return f'opentsne=={match.group(1)}' if match else 'opentsne'


def run(cmd, **kw) -> str:
    """stdout of a command; on failure its output is printed before raising."""
    result = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if result.returncode != 0:
        print(result.stdout, result.stderr, sep='\n', file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result.stdout


def package_dir(package: str) -> Path:
    """A package's directory in this environment, found without importing it."""
    return Path(run([sys.executable, '-c',
                     'import importlib.util, sys; '
                     'print(importlib.util.find_spec(sys.argv[1]).submodule_search_locations[0])',
                     package]).strip())


def libomp_ref(binary: Path) -> str | None:
    """The libomp load command of a Mach-O binary, or None."""
    for line in run(['otool', '-L', str(binary)]).splitlines()[1:]:
        path = line.strip().split(' (')[0]
        if os.path.basename(path) == 'libomp.dylib':
            return path
    return None


def omp_symbols(binary: Path, undefined: bool) -> set[str]:
    """OpenMP runtime symbols (__kmpc_*, omp_*) a binary imports or exports."""
    out = run(['nm', '-u' if undefined else '-gU', str(binary)])
    names = {line.split()[-1] for line in out.splitlines() if line.strip()}
    return {n for n in names if n.startswith(('___kmpc_', '_omp_'))}


def main() -> int:
    if platform.system() != 'Darwin':
        print('openTSNE wheels for this platform already use OpenMP; nothing to do.')
        return 0
    try:
        omp_prefix = Path(run(['brew', '--prefix', 'libomp']).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        print('Homebrew libomp not found: run `brew install libomp` first.')
        return 1
    libomp = omp_prefix / 'lib' / 'libomp.dylib'
    if not (omp_prefix / 'include' / 'omp.h').exists() or not libomp.exists():
        print(f'omp.h or libomp.dylib not found under {omp_prefix}: '
              f'run `brew install libomp` first.')
        return 1

    env = dict(os.environ,
               CPPFLAGS=f"{os.environ.get('CPPFLAGS', '')} -I{omp_prefix / 'include'}".strip(),
               LDFLAGS=f"{os.environ.get('LDFLAGS', '')} -L{libomp.parent}".strip(),
               MACOSX_DEPLOYMENT_TARGET=os.environ.get('MACOSX_DEPLOYMENT_TARGET', '11.0'))
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', '--no-binary', 'opentsne', '--no-cache-dir',
         '--force-reinstall', '--no-deps', pinned_requirement()], env=env)

    exported = omp_symbols(libomp, undefined=False)
    linked = []
    for ext in sorted(package_dir('openTSNE').rglob('*.so')):
        ref = libomp_ref(ext)
        if ref is None:
            continue
        missing = omp_symbols(ext, undefined=True) - exported
        if missing:
            print(f'{ext.name} calls OpenMP entry points {sorted(missing)} that {libomp} '
                  f'does not export.')
            return 1
        linked.append(f'{ext.name} -> {ref}')
    print('Linked to libomp:', '; '.join(linked) or 'none')
    if not any(entry.startswith('_tsne.') for entry in linked):
        print('openTSNE was built without OpenMP (omp.h not picked up).')
        return 1

    subprocess.check_call([sys.executable, '-c', SELF_TEST, str(MIN_SPEEDUP), str(libomp)])
    print('openTSNE built with OpenMP and verified.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
