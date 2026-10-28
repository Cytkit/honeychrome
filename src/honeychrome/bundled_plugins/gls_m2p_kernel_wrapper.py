"""
gls_m2p_kernel_wrapper.py
-------------------------
Python interface to the compiled _gls_m2p_kernel pybind11 extension
(gls_m2p_kernel.cpp): M2pg from raw events, the per-cell AF start included.
Called by controller_components/gls_unmix.unmix_m2pg, which falls back to
NumPy when the extension is absent. Build it with
build_autospectral_opt_kernel.py (CMake, both kernels).
"""

import logging
import os
import sys

import numpy as np

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import _gls_m2p_kernel as _lib
    GLS_M2P_KERNEL_AVAILABLE = True
except ImportError:
    _lib = None
    GLS_M2P_KERNEL_AVAILABLE = False
    logger.info('gls_m2p_kernel_wrapper: compiled kernel not found; noise-model unmixing runs in NumPy.')

# Events solved together in the kernel's batched (structure-of-arrays) solve.
BATCH_LANES = 8


def _envelope_arg(env: dict | None) -> dict | None:
    if env is None:
        return None
    out = {'null_var': np.ascontiguousarray(env['null_var'], dtype=np.float64),
           'mean_rows': [int(r) for r in np.atleast_1d(env.get('mean_rows', [0]))]}
    if out['null_var'].ndim == 2:
        out['null_var'] = out['null_var'][np.newaxis]
    if env.get('spread') is not None:
        out['spread'] = np.ascontiguousarray(env['spread'], dtype=np.float64)
    if 'ceiling' in env and 'saturated_frac' in env:
        out['ceiling'] = float(env['ceiling'])
        out['saturated_frac'] = float(env['saturated_frac'])
    return out


def _arr(a):
    return None if a is None else np.ascontiguousarray(a, dtype=np.float64)


def unmix_m2p_kernel(raw, spectra, af_spectra, envelope, sigma0_sq, g, c, bead_var=None, tau_spread=None,
                     prior_lambda=1.0, n_iter=2, ceiling=0.0, n_threads=0, af_choice_weights=None,
                     batch_lanes=BATCH_LANES) -> dict:
    """
    ``af_spectra`` None or empty for no AF; ``bead_var`` (F,) is the prior's
    negative variance without AF; ``tau_spread`` (F x F) the spillover
    spread added to it; ``af_choice_weights`` (D,) whitens the per-cell AF
    choice (M2pg). Returns ``unmixed`` (n, F), ``af_idx`` (0-based),
    ``af_scale``, ``chi2``.
    """
    if _lib is None:
        raise ImportError('the _gls_m2p_kernel extension is not built (build_autospectral_opt_kernel.py)')
    d = np.asarray(spectra).shape[1]
    af = np.zeros((0, d)) if af_spectra is None else np.ascontiguousarray(af_spectra, dtype=np.float64)
    out = _lib.unmix_m2p(
        np.ascontiguousarray(raw, dtype=np.float64), np.ascontiguousarray(spectra, dtype=np.float64), af,
        _envelope_arg(envelope), _arr(sigma0_sq), _arr(g), _arr(c), 1.0, _arr(bead_var), _arr(tau_spread),
        float(prior_lambda), int(n_iter), float(ceiling or 0.0), int(n_threads), False, _arr(af_choice_weights),
        int(batch_lanes))
    out['af_idx'] = np.asarray(out['af_idx'], dtype=np.int64)
    return out
