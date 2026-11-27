"""
gls_m2p_kernel_wrapper.py
-------------------------
Python interface to the compiled _gls_m2p_kernel pybind11 extension:
M2pg from raw events, the per-cell AF start included (gls_m2p_kernel.cpp),
called by controller_components/gls_unmix.unmix_m2pg, and the AF shape bend
of per-cell AF (af_bend_kernel.cpp), called by
controller_components/af_shape_bend.bend_fluorophores. Both callers fall
back to NumPy when the extension is absent. Build it with
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

# The extension holds the per-cell AF shape bend (absent from builds made before it was added).
BEND_R1_AVAILABLE = GLS_M2P_KERNEL_AVAILABLE and hasattr(_lib, 'bend_r1')

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
                     batch_lanes=BATCH_LANES, fixed_af_idx=None, fixed_af_scale=None, bend_shapes=None,
                     bend_var=None) -> dict:
    """
    ``af_spectra`` None or empty for no AF; ``bead_var`` (F,) is the prior's
    negative variance without AF; ``tau_spread`` (F x F) the spillover
    spread added to it; ``af_choice_weights`` (D,) whitens the per-cell AF
    choice (M2pg). ``fixed_af_idx`` (n, 0-based) and ``fixed_af_scale`` (n,)
    fix each event's AF row and abundance (only the fluorophores are
    solved); a kernel built before this option raises TypeError. With the
    AF fixed or without AF, ``bend_shapes`` (L, D) and ``bend_var`` (n, L)
    integrate out each event's shape terms (the AF shape bend, a
    fluorophore's spectral variability); an event uses only the shapes
    whose variance is positive for it. A kernel built before the option
    raises TypeError, and one built before shapes were accepted without AF
    raises ValueError for them. Returns ``unmixed`` (n, F), ``af_idx``
    (0-based), ``af_scale``, ``chi2``.
    """
    if _lib is None:
        raise ImportError('the _gls_m2p_kernel extension is not built (build_autospectral_opt_kernel.py)')
    d = np.asarray(spectra).shape[1]
    af = np.zeros((0, d)) if af_spectra is None else np.ascontiguousarray(af_spectra, dtype=np.float64)
    optional = {}
    if fixed_af_idx is not None:
        optional = {'fixed_af_idx': np.ascontiguousarray(fixed_af_idx, dtype=np.int64),
                    'fixed_af_scale': _arr(fixed_af_scale)}
    if bend_shapes is not None and (fixed_af_idx is not None or not len(af)):
        optional.update(bend_shapes=_arr(bend_shapes), bend_var=_arr(bend_var))
    out = _lib.unmix_m2p(
        np.ascontiguousarray(raw, dtype=np.float64), np.ascontiguousarray(spectra, dtype=np.float64), af,
        _envelope_arg(envelope), _arr(sigma0_sq), _arr(g), _arr(c), 1.0, _arr(bead_var), _arr(tau_spread),
        float(prior_lambda), int(n_iter), float(ceiling or 0.0), int(n_threads), False, _arr(af_choice_weights),
        int(batch_lanes), **optional)
    out['af_idx'] = np.asarray(out['af_idx'], dtype=np.int64)
    return out


def bend_r1_kernel(d, u, mu, t2, spectra, unmix_matrix, shapes, noise, spread=None, robust_z=4.0, robust_passes=2,
                   positive_offset_sd=2.0, n_threads=0) -> np.ndarray:
    """
    The per-cell AF shape bend of events chosen to bend (see
    af_shape_bend.bend_fluorophores for the method): leftover ``d`` (n, D),
    per-cell AF fluorophores ``u`` (n, F), fitted signal ``mu`` (n, D), the
    bend's prior variance ``t2`` (n, L), ``spectra`` and ``unmix_matrix``
    (F, D), ``shapes`` (L, D), ``noise`` = (sigma0_sq, g, c) per detector and
    the spillover ``spread`` (F, F) or None. Returns the bent fluorophores
    (n, F). An event whose system is singular keeps ``u`` (logged).
    """
    if not BEND_R1_AVAILABLE:
        raise ImportError('the _gls_m2p_kernel extension is not built with the AF shape bend '
                          '(build_autospectral_opt_kernel.py)')
    sigma0_sq, g, c = noise
    out = _lib.bend_r1(_arr(d), _arr(u), _arr(mu), _arr(t2), _arr(spectra), _arr(unmix_matrix), _arr(shapes),
                       _arr(sigma0_sq), _arr(g), _arr(c), _arr(spread), float(robust_z), int(robust_passes),
                       float(positive_offset_sd), int(n_threads))
    if out['n_failed']:
        logger.warning('AF shape bend: %d of %d events kept their unbent fluorophores (singular system).',
                       out['n_failed'], len(out['unmixed']))
    return out['unmixed']
