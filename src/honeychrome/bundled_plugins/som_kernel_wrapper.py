"""
som_kernel_wrapper.py
----------------------
Python interface to the compiled _som_kernel cffi extension (som_kernel.c):
OpenMP batch self-organising map training and nearest-code mapping.

Used by the DR/Clustering plugin's FlowSOM path (flowsom_consensus.py) and
by autospectral_functions.get_som_codes() for AutoSpectral autofluorescence
and fluorophore-variant clustering, which falls back to KMeans when this
kernel is unavailable.

Usage
-----
    from som_kernel_wrapper import train_som_batch, map_to_codes, SOM_KERNEL_AVAILABLE

    if SOM_KERNEL_AVAILABLE:
        codes = train_som_batch(data, init_codes, nhbrdist, radii,
                                 dist=2, n_threads=0)
        node_ids, dists = map_to_codes(data, codes, dist=2, n_threads=0)
        # node_ids: ndarray (n,) int32, 0-based

The module tries to import _som_kernel from the same directory as this
file (i.e. the compiled cffi extension). SOM_KERNEL_AVAILABLE is False and
both functions raise ImportError if the extension is absent.
"""

import os
import sys
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load the compiled extension
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import _som_kernel as _lib
    SOM_KERNEL_AVAILABLE = True
    logger.info('som_kernel_wrapper: compiled SOM kernel loaded.')
except ImportError:
    _lib = None
    SOM_KERNEL_AVAILABLE = False
    logger.info('som_kernel_wrapper: compiled SOM kernel not found — '
                'run build_som_kernel.py to enable.')


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def train_som_batch(
    data:       np.ndarray,
    init_codes: np.ndarray,
    nhbrdist:   np.ndarray,
    radii:      np.ndarray,
    dist:       int = 2,
    n_threads:  int = 0,
) -> np.ndarray:
    """
    Train a batch SOM.

    data       : (n, px) training events
    init_codes : (ncodes, px) initial codebook (e.g. a random sample of
        data rows -- pick this in Python so the seed stays under caller
        control)
    nhbrdist   : (ncodes, ncodes) grid neighbourhood distances
    radii      : (n_epochs,) neighbourhood radius per epoch, strictly > 0
    dist       : 1=manhattan, 2=euclidean (default), 3=chebyshev, 4=cosine
    n_threads  : OpenMP threads, 0 = all available cores (default)

    Returns codes : ndarray (ncodes, px) float64.
    """
    if not SOM_KERNEL_AVAILABLE:
        raise ImportError(
            'Compiled SOM kernel not available. '
            'Run build_som_kernel.py first.'
        )

    data       = np.ascontiguousarray(data,       dtype=np.float64)
    init_codes = np.ascontiguousarray(init_codes, dtype=np.float64)
    nhbrdist   = np.ascontiguousarray(nhbrdist,   dtype=np.float64)
    radii      = np.ascontiguousarray(radii,      dtype=np.float64)

    n, px    = data.shape
    ncodes   = init_codes.shape[0]
    n_epochs = radii.shape[0]

    ffi = _lib.ffi
    lib = _lib.lib

    codes_out = np.empty((ncodes, px), dtype=np.float64)

    lib.som_train_batch(
        ffi.cast('double *', data.ctypes.data),
        ffi.cast('double *', init_codes.ctypes.data),
        ffi.cast('double *', nhbrdist.ctypes.data),
        ffi.cast('double *', radii.ctypes.data),
        ffi.cast('double *', codes_out.ctypes.data),
        n, px, ncodes, n_epochs,
        dist, n_threads,
    )

    return codes_out


def map_to_codes(
    data:      np.ndarray,
    codes:     np.ndarray,
    dist:      int = 2,
    n_threads: int = 0,
):
    """
    Map each row of data to its nearest code.

    Returns (node_ids, dists):
        node_ids : ndarray (n,) int32, 0-based nearest-code index
        dists    : ndarray (n,) float64, distance to that code
    """
    if not SOM_KERNEL_AVAILABLE:
        raise ImportError(
            'Compiled SOM kernel not available. '
            'Run build_som_kernel.py first.'
        )

    data  = np.ascontiguousarray(data,  dtype=np.float64)
    codes = np.ascontiguousarray(codes, dtype=np.float64)

    n, px  = data.shape
    ncodes = codes.shape[0]

    ffi = _lib.ffi
    lib = _lib.lib

    node_ids = np.empty(n, dtype=np.int32)
    dists    = np.empty(n, dtype=np.float64)

    lib.som_map_to_codes(
        ffi.cast('double *', data.ctypes.data),
        ffi.cast('double *', codes.ctypes.data),
        ffi.cast('int32_t *', node_ids.ctypes.data),
        ffi.cast('double *', dists.ctypes.data),
        n, px, ncodes,
        dist, n_threads,
    )

    return node_ids, dists