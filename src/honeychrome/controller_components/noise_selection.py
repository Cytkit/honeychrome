"""
noise_selection.py
------------------
Automatic event selection for the noise model fit (gls_unmix): the scatter
core of a sample and its singlets, independent of the user's gates. No Qt.

``scatter_gate(fsc_a, ssc_a, fsc_h=None)``
    1. The density mode of log10 FSC-A x log10 SSC-A: an 80 x 80 histogram
       over the 0.5-99.5th percentiles, 5 x 5 box-smoothed.
    2. Concentration steps from there (FAST-MCD style): 15 steps of h =
       50%, each centred on the median of the inner half, with the
       covariance scaled by the consistency factor 3.26 and 0.01^2 added to
       its diagonal, so that a saturated axis does not collapse it.
    3. The gate: Mahalanobis^2 below 5.991 (95%).
    4. Singlets, with FSC-H: |log10(A / H) - median| < 4 MAD.

    Saturated events are kept (BD beads saturate SSC on the Aurora and FSC
    and SSC on the Attune Xenith); events with non-positive or non-finite
    FSC-A or SSC-A are not.

``select_events(scatter, columns)``
    The mask of one sample and its counts ``{n_root, n_scatter,
    n_singlet}``; ``columns`` are the positions of FSC-A, SSC-A and FSC-H
    (or None) in the sample's scatter array (``scatter_columns``).

The steps differ from particle_type.scatter_core (start at the density
mode, median centre, consistency factor, saturated events kept), so that
function is not reused: its thresholds were calibrated on its own steps.
"""

from __future__ import annotations

import numpy as np

CHI2_95 = 5.991
N_STEPS = 15
CONSISTENCY = 3.26
SINGLET_MADS = 4.0
_HIST_BINS = 80
_SMOOTH = 5


def _density_mode(xy: np.ndarray) -> tuple:
    """``(centre, starting covariance)`` from the smoothed 2-D histogram of the finite events."""
    lo = np.percentile(xy, 0.5, axis=0)
    hi = np.percentile(xy, 99.5, axis=0)
    hist, ex, ey = np.histogram2d(xy[:, 0], xy[:, 1], bins=_HIST_BINS,
                                  range=[[lo[0], hi[0] + 1e-9], [lo[1], hi[1] + 1e-9]])
    pad = _SMOOTH // 2
    padded = np.pad(hist, pad)
    weight = 1.0 / (_SMOOTH * _SMOOTH)
    smooth = sum(padded[i:i + _HIST_BINS, j:j + _HIST_BINS] * weight for i in range(_SMOOTH) for j in range(_SMOOTH))
    i, j = np.unravel_index(int(np.argmax(smooth)), smooth.shape)
    centre = np.array([0.5 * (ex[i] + ex[i + 1]), 0.5 * (ey[j] + ey[j + 1])])
    return centre, np.diag(((hi - lo) / 20.0) ** 2)


def _mahalanobis2(xy: np.ndarray, centre: np.ndarray, cov: np.ndarray) -> np.ndarray:
    diff = xy - centre
    return np.einsum('ij,jk,ik->i', diff, np.linalg.inv(cov), diff)


def scatter_gate(fsc_a: np.ndarray, ssc_a: np.ndarray, fsc_h: np.ndarray | None = None,
                 cutoff: float = CHI2_95) -> np.ndarray:
    """Boolean mask of the scatter core (and singlets with ``fsc_h``); see the module docstring."""
    core = scatter_core_gate(fsc_a, ssc_a, cutoff)
    return core if fsc_h is None else singlet_band(fsc_a, fsc_h, core)


def scatter_core_gate(fsc_a: np.ndarray, ssc_a: np.ndarray, cutoff: float = CHI2_95) -> np.ndarray:
    """Steps 1-3 of the module docstring."""
    fsc_a = np.asarray(fsc_a, dtype=np.float64)
    ssc_a = np.asarray(ssc_a, dtype=np.float64)
    ok = np.isfinite(fsc_a) & np.isfinite(ssc_a) & (fsc_a > 0) & (ssc_a > 0)
    if ok.sum() < 3:
        return ok
    xy = np.column_stack([np.log10(np.clip(fsc_a, 1.0, None)), np.log10(np.clip(ssc_a, 1.0, None))])
    centre, cov = _density_mode(xy[ok])
    reg = np.diag([0.01 ** 2, 0.01 ** 2])
    for _ in range(N_STEPS):
        md2 = _mahalanobis2(xy, centre, cov)
        md2[~ok] = np.inf
        inner = (md2 <= np.quantile(md2[ok], 0.5)) & (md2 < 4.0 * cutoff)
        if inner.sum() < 3:
            break
        centre = np.median(xy[inner], axis=0)
        cov = np.cov(xy[inner].T) * CONSISTENCY + reg
    return ok & (_mahalanobis2(xy, centre, cov) < cutoff)


def singlet_band(fsc_a: np.ndarray, fsc_h: np.ndarray, core: np.ndarray) -> np.ndarray:
    """Step 4: the events of ``core`` within the singlet band of log10(FSC-A / FSC-H)."""
    if not core.any():
        return core
    ratio = (np.log10(np.clip(np.asarray(fsc_a, dtype=np.float64), 1.0, None))
             - np.log10(np.clip(np.asarray(fsc_h, dtype=np.float64), 1.0, None)))
    med = np.median(ratio[core])
    mad = 1.4826 * np.median(np.abs(ratio[core] - med))
    return core & (np.abs(ratio - med) < SINGLET_MADS * max(mad, 1e-3))


def scatter_columns(scatter_names: list, fsc_a: str | None, ssc_a: str | None,
                    fsc_h: str | None = None) -> tuple | None:
    """
    ``(i_fsc_a, i_ssc_a, i_fsc_h or None)`` in ``scatter_names``, or None
    without FSC-A and SSC-A. FSC-H defaults to FSC-A's name with -H for -A.
    """
    if fsc_a not in scatter_names or ssc_a not in scatter_names:
        return None
    if fsc_h is None and str(fsc_a).endswith('-A'):
        fsc_h = str(fsc_a)[:-2] + '-H'
    return (scatter_names.index(fsc_a), scatter_names.index(ssc_a),
            scatter_names.index(fsc_h) if fsc_h in scatter_names else None)


def select_events(scatter: np.ndarray, columns: tuple) -> tuple:
    """``(mask, counts)`` of one sample's events (see the module docstring)."""
    scatter = np.asarray(scatter)
    i_a, i_s, i_h = columns
    core = scatter_core_gate(scatter[:, i_a], scatter[:, i_s])
    mask = core if i_h is None else singlet_band(scatter[:, i_a], scatter[:, i_h], core)
    return mask, {'n_root': int(len(scatter)), 'n_scatter': int(core.sum()), 'n_singlet': int(mask.sum())}
