"""
drc_jit.py — numba threading layer, JIT warm-up and safe numba sections
=======================================================================
Companion module to ``dr_clustering_tab.py`` (not a plugin: the filename
does not end in ``_tab.py``).

UMAP (with pynndescent) and PaCMAP are numba code.  Most of UMAP's and
pynndescent's functions are compiled afresh in every session (they are not
``cache=True``).  The first fit of 4,096 or more events and the first embed
compile about 9 s of code on the 4-core reference VM (an Apple M-series Mac),
more on slower machines.  It was 13 s while the transform searched with
pynndescent, whose query code alone took 5.5 s to compile; the search now
uses hnswlib (drc_umap_index).  ``start_warmup`` compiles it on a background thread while
the user is still configuring, so the first real run does not pay it.

Threading layer
---------------
Numba's 'omp' layer is safe when two threads run parallel numba code at
once; its 'workqueue' fallback is not (concurrent use can deadlock or
abort the process).  NUMBA_THREADING_LAYER is set to 'omp' when this module
is imported, before numba is.  ``ensure_threading_layer`` checks that the
OpenMP layer loads and falls back to 'workqueue' if it does not.  In
practice the layer is 'workqueue' once UMAP is imported: pynndescent's
``__init__`` replaces 'omp' with 'tbb', or 'workqueue' when TBB is missing
(as it is with the plugin's requirements), and on macOS numba's OpenMP pool
does not find a libomp anyway.  The two layers ran UMAP equally fast on the
reference VM (fit of 50,000 events 6.1-6.3 s vs 6.2-6.4 s, embed of 40,000
3.3-3.8 s vs 3.4-3.6 s), so the only cost of 'workqueue' is that a run
started during the warm-up waits for it.

``numba_section`` serialises numba work (warm-up, fits, transforms) while
the layer is 'workqueue', and is a no-op under 'omp'.  Every UMAP and
PaCMAP call in the plugin runs inside it, so the warm-up thread and a run
can never execute parallel numba code at the same time under 'workqueue'.

openTSNE threads
----------------
openTSNE's Barnes-Hut gradient and affinities use OpenMP only where its
extension was compiled with it: the Windows and Linux wheels are, the macOS
wheels are not (install_opentsne.py builds it with OpenMP).
``opentsne_has_openmp`` tells the two apart, for the choice of gradient
method (drc_time_estimate.tsne_gradient_method).

Pure Python; no Qt.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import threading
import time

import numpy as np

import drc_umap_index
from drc_logging import get_logger

log = get_logger(__name__)

# Must be set before the first numba import (numba reads it at import time)
LAYER_FROM_USER = 'NUMBA_THREADING_LAYER' in os.environ
if not LAYER_FROM_USER:
    os.environ['NUMBA_THREADING_LAYER'] = 'omp'

WARMUP_EVENTS = 5_000          # above UMAP's 4,096-event exact-kNN path
WARMUP_EMBED_EVENTS = 500
WARMUP_THREADS = 2             # compiles the parallel path without taking every core

_layer_lock = threading.Lock()
_layer_checked = False
_numba_lock = threading.RLock()

_warmup_lock = threading.Lock()
_warmup_threads: dict[tuple, threading.Thread] = {}
_warmed: set[tuple] = set()


def ensure_threading_layer() -> None:
    """
    Make sure Numba's 'omp' threading layer can load before UMAP or PaCMAP
    compile or load cached parallel code, and fall back to 'workqueue' if
    not.

    Numba's OpenMP pool links against '@rpath/libomp.dylib' on macOS, which
    wheel installs do not provide; it resolves only if a libomp of that name
    is already loaded into the process (scikit-learn's bundled copy is
    imported first to supply one).  A threading layer the user set through
    NUMBA_THREADING_LAYER is left alone.  Safe to call repeatedly from any
    thread.
    """
    global _layer_checked
    with _layer_lock:
        if _layer_checked:
            return
        _layer_checked = True
        if LAYER_FROM_USER:
            return
        try:
            import sklearn.utils._openmp_helpers  # noqa: F401  (loads a libomp)
        except Exception:
            pass
        try:
            from numba.np.ufunc import omppool  # noqa: F401
        except ImportError:
            import numba
            numba.config.THREADING_LAYER = 'workqueue'
            log.warning("Numba's OpenMP threading layer could not be loaded; using "
                        "'workqueue'. UMAP and PaCMAP still run multi-threaded, but "
                        "their numba work is serialised.")


def layer_is_threadsafe() -> bool:
    """True when concurrent parallel numba calls from different threads are safe."""
    ensure_threading_layer()
    import numba
    layer = str(numba.config.THREADING_LAYER).lower()
    return layer in ('omp', 'tbb', 'threadsafe')


@contextlib.contextmanager
def numba_section(waiting=None):
    """Hold the numba lock while the threading layer is not thread-safe.

    waiting: optional callable(str) told when the section has to wait (for
    the warm-up, or a cancelled run still finishing)."""
    if layer_is_threadsafe():
        yield
        return
    if not _numba_lock.acquire(blocking=False):
        if waiting is not None:
            waiting("Waiting for numba compilation in the background to finish …")
        _numba_lock.acquire()
    try:
        yield
    finally:
        _numba_lock.release()


_OPENMP_MARKERS = (b'GOMP_', b'__kmpc_', b'_vcomp_')   # GCC, clang/LLVM, MSVC
_opentsne_openmp: bool | None = None


def opentsne_has_openmp() -> bool:
    """True when openTSNE's compiled gradient code (openTSNE._tsne) calls an
    OpenMP runtime, i.e. its Barnes-Hut gradient uses every thread asked for.
    Checked once, from the extension file's bytes, without importing it."""
    global _opentsne_openmp
    if _opentsne_openmp is None:
        try:
            import importlib.util
            origin = importlib.util.find_spec('openTSNE._tsne').origin
            with open(origin, 'rb') as f:
                blob = f.read()
            _opentsne_openmp = any(marker in blob for marker in _OPENMP_MARKERS)
        except Exception:
            log.exception("could not inspect openTSNE for OpenMP; assuming one thread")
            _opentsne_openmp = False
        log.info("openTSNE built with OpenMP: %s", _opentsne_openmp)
    return _opentsne_openmp


def _warmup_key(algo: str, parallel: bool) -> tuple:
    return (algo, bool(parallel))


def is_warm(algo: str, parallel: bool) -> bool:
    return _warmup_key(algo, parallel) in _warmed


def _warmup_umap(parallel: bool) -> None:
    import umap as umap_lib
    rng = np.random.default_rng(0)
    centres = rng.normal(scale=4.0, size=(8, 7))
    data = (centres[rng.integers(0, 8, WARMUP_EVENTS)]
            + rng.normal(size=(WARMUP_EVENTS, 7))).astype(np.float32)
    reducer = umap_lib.UMAP(
        n_neighbors=15, n_components=2, n_epochs=10,
        random_state=None if parallel else 42,
        n_jobs=WARMUP_THREADS if parallel else 1,
        low_memory=False, verbose=False)
    reducer.fit(data)
    # The plugin's transform searches an hnswlib index (drc_umap_index), so
    # pynndescent's query code is never compiled
    drc_umap_index.attach_search_index(reducer, WARMUP_THREADS if parallel else 1)
    reducer.transform(data[:WARMUP_EMBED_EVENTS])


def _run_warmup(algo: str, parallel: bool) -> None:
    key = _warmup_key(algo, parallel)
    start = time.perf_counter()
    try:
        ensure_threading_layer()
        with numba_section():
            if algo == 'UMAP':
                _warmup_umap(parallel)
        _warmed.add(key)
        log.info("JIT warm-up for %s (%s) done in %.1fs", algo,
                 'parallel' if parallel else 'serial', time.perf_counter() - start)
    except Exception:
        log.exception("JIT warm-up for %s failed; the run compiles on first use", algo)
    finally:
        with _warmup_lock:
            _warmup_threads.pop(key, None)


def start_warmup(algo: str, parallel: bool) -> bool:
    """Compile *algo*'s numba code on a background thread, once per session
    and code path (parallel when UMAP runs without a fixed seed).

    Returns True when a warm-up was started."""
    if algo != 'UMAP':
        return False
    key = _warmup_key(algo, parallel)
    with _warmup_lock:
        if key in _warmed or key in _warmup_threads:
            return False
        thread = threading.Thread(target=_run_warmup, args=(algo, parallel),
                                  name=f'drc-jit-warmup-{algo}', daemon=True)
        _warmup_threads[key] = thread
    thread.start()
    return True


def ensure_warm(algo: str, parallel: bool) -> None:
    """Compile *algo*'s numba code now (from a worker thread), joining a
    warm-up already in progress."""
    if algo != 'UMAP' or is_warm(algo, parallel):
        return
    start_warmup(algo, parallel)
    with _warmup_lock:
        thread = _warmup_threads.get(_warmup_key(algo, parallel))
    if thread is not None:
        thread.join()


def wait_for_warmups(timeout: float = 60.0) -> None:
    """Join running warm-ups (called at interpreter exit, so numba code is not
    torn down mid-call)."""
    with _warmup_lock:
        threads = list(_warmup_threads.values())
    deadline = time.monotonic() + timeout
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))


atexit.register(wait_for_warmups)
