"""
autospectral_opt_kernel_wrapper.py
------------------------------------
Python interface to the compiled _autospectral_opt_kernel pybind11 extension
(shared Armadillo core, see core_joint_unmix.hpp/.cpp and
CONTEXT_AutoSpectral.md §4).

Usage
-----
    from autospectral_opt_kernel_wrapper import (
        unmix_autospectral_joint, AUTOSPECTRAL_OPT_KERNEL_AVAILABLE,
    )

    if AUTOSPECTRAL_OPT_KERNEL_AVAILABLE:
        result = unmix_autospectral_joint(
            raw_data_in=raw_fl,             # (N, D) float64
            spectra=spectra,                # (F, D) float64, no AF row
            af_spectra=af_spectra,          # (nAF, D) float64, nAF >= 2
            fluor_names=fluor_names,        # list[str], len F
            pos_thresholds=pos_thresholds,  # (F,) float64, unmixed-space
            variants=variants,              # list[dict(name, v_mats, delta_obs)]
            n_passes=1, n_threads=1, cell_weight=False,
            noise_floor=None, alpha=0.5, collinear_thresh=0.5,
            joint_pair_resolution=True, n_af_passes=1, refine_af_quantile=0.5,
            exact_variant_scan=False,
        )
        # result : ndarray (N, F+2) — [fluor abundances | AF abundance | AF index (1-based)]

The module tries to import _autospectral_opt_kernel from the same directory
as this file (the compiled pybind11 extension).
AUTOSPECTRAL_OPT_KERNEL_AVAILABLE is False and unmix_autospectral_joint
raises ImportError if the extension is absent — run
build_autospectral_opt_kernel.py first.
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
    import _autospectral_opt_kernel as _lib
    AUTOSPECTRAL_OPT_KERNEL_AVAILABLE = True
    logger.info('autospectral_opt_kernel_wrapper: compiled kernel loaded.')
except ImportError:
    _lib = None
    AUTOSPECTRAL_OPT_KERNEL_AVAILABLE = False
    logger.info(
        'autospectral_opt_kernel_wrapper: compiled kernel not found — '
        'run build_autospectral_opt_kernel.py to enable.'
    )


def unmix_autospectral_joint(
    raw_data_in: np.ndarray,
    spectra: np.ndarray,
    af_spectra: np.ndarray,
    fluor_names: list,
    pos_thresholds: np.ndarray,
    variants: list,
    n_passes: int = 1,
    n_threads: int = 1,
    cell_weight: bool = False,
    noise_floor: np.ndarray | None = None,
    alpha: float = 0.5,
    collinear_thresh: float = 0.5,
    joint_pair_resolution: bool = True,
    n_af_passes: int = 1,
    refine_af_quantile: float = 0.5,
    exact_variant_scan: bool = False,
) -> np.ndarray:
    """
    Call the compiled joint AF + fluorophore-variant unmixing kernel.

    Parameters
    ----------
    raw_data_in : (N, D) float64 — raw fluorescence events.
    spectra     : (F, D) float64 — reference fluorophore spectra (no AF row).
    af_spectra  : (nAF, D) float64 — AF candidate spectra (nAF >= 2, mirrors
        unmix.autospectral.rcpp()'s own validation).
    fluor_names : list[str], length F, same row order as `spectra`.
    pos_thresholds : (F,) float64 — per-fluorophore positivity threshold in
        unmixed space, gates whether joint variant optimisation is attempted
        for a cell.
    variants : list of dict, one entry per *optimizable* fluorophore (may be
        empty — AF-only mode). Each dict:
            {'name': str, 'v_mats': (n_variants, D) float64,
             'delta_obs': (n_variants, D) float64}
        `delta_obs` is `v_mats - reference_row` (matches R's `delta.list`,
        see get_spectral_variants.R line ~481 — NOT raw per-event data).
    noise_floor : None, scalar-shaped (1,), or (D,) float64. None -> 125.0
        everywhere (matches the R/C++ default).
    exact_variant_scan : score candidate variants with the closed-form
        single-endmember swap instead of the fixed-abundance residual
        approximation (unmix.autospectral.rcpp's exact.variant.scan). Ignored
        when cell_weight is True.

    Returns
    -------
    ndarray (N, F+2) — columns [fluor_1 .. fluor_F, AF abundance, AF index
    (1-based)].
    """
    if not AUTOSPECTRAL_OPT_KERNEL_AVAILABLE:
        raise ImportError(
            'Compiled AutoSpectral Optimization kernel not available. '
            'Run build_autospectral_opt_kernel.py first.'
        )

    raw_data_in    = np.ascontiguousarray(raw_data_in, dtype=np.float64)
    spectra        = np.ascontiguousarray(spectra, dtype=np.float64)
    af_spectra     = np.ascontiguousarray(af_spectra, dtype=np.float64)
    pos_thresholds = np.ascontiguousarray(pos_thresholds, dtype=np.float64)

    variants_c = [
        {
            'name': v['name'],
            'v_mats': np.ascontiguousarray(v['v_mats'], dtype=np.float64),
            'delta_obs': np.ascontiguousarray(v['delta_obs'], dtype=np.float64),
        }
        for v in variants
    ]

    nf_arg = None
    if noise_floor is not None:
        nf_arg = np.ascontiguousarray(np.atleast_1d(noise_floor), dtype=np.float64)

    return _lib.unmix_autospectral_joint(
        raw_data_in,
        spectra,
        af_spectra,
        list(fluor_names),
        pos_thresholds,
        variants_c,
        n_passes,
        n_threads,
        cell_weight,
        nf_arg,
        alpha,
        collinear_thresh,
        joint_pair_resolution,
        n_af_passes,
        refine_af_quantile,
        exact_variant_scan,
    )
