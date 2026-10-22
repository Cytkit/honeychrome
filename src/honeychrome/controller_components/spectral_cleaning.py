"""
spectral_cleaning.py

Public API:
    exclude_saturated(events, ceiling, threshold_frac=0.99)
    find_empirical_peak(spec_events, af_mean)
    cosine_filter(spec_events, af_median, peak_ch_idx, ...)
    knn_scatter_match(spec_pos, scatter_pos, spec_neg, scatter_neg, k=3)
    background_check(pos_events, neg_events, neg_scatter, spectral_sub, spectrum, peak_ch_idx, ...)
    CleanResult

Background check (v1): only controls cleaned against an unstained tube are
checked; internal-negative controls have no independent null and are skipped.
SpectralCleaner also skips controls whose spectrum passes the reference
library cosine QC; it runs when the cosine fails or there is no reference.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class CleanResult:
    spectral_sub:             np.ndarray        # kNN-subtracted spectral events
    scatter_pos:              np.ndarray        # scatter of selected positives
    scatter_neg_matched:      np.ndarray        # mean kNN-matched neg scatter
    n_removed_saturation:     int
    n_surviving_positive:     int
    empirical_peak_ch_idx:    int
    expected_peak_ch_idx:     int
    cs_vals:                  np.ndarray | None = None
    cosine_selected_idx:      np.ndarray | None = None
    warnings:                 list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# exclude_saturated
# ---------------------------------------------------------------------------

def exclude_saturated(
    events: np.ndarray,
    ceiling: float,
    threshold_frac: float = 0.999,
) -> tuple[np.ndarray, int, np.ndarray]:
    """
    ...
    Returns
    -------
    filtered_events : rows surviving the filter
    n_removed : number of rows removed
    saturated_mask : boolean mask (True = saturated, length = input row count)
    """
    sat_threshold = ceiling * threshold_frac
    saturated_mask = np.any(events >= sat_threshold, axis=1)
    n_removed = int(saturated_mask.sum())
    return events[~saturated_mask], n_removed, saturated_mask


# ---------------------------------------------------------------------------
# select_positive_events
# ---------------------------------------------------------------------------
def find_empirical_peak(
    spec_events: np.ndarray,
    af_mean: np.ndarray,
) -> int:
    """
    AF orthogonalisation -> empirical peak column index.

    Projects events onto the unit AF vector, subtracts the projection,
    returns the column with the highest mean in the orthogonalised space.
    """
    norm = np.sqrt(np.dot(af_mean, af_mean)) + 1e-9
    v_unit = af_mean / norm
    proj = spec_events @ v_unit
    mat_orth = spec_events - np.outer(proj, v_unit)
    return int(np.argmax(mat_orth.mean(axis=0)))

def knn_scatter_match(
    spec_pos: np.ndarray,      # (n_pos, n_ch)
    scatter_pos: np.ndarray,   # (n_pos, 2) FSC/SSC selected positives
    spec_neg: np.ndarray,      # (n_neg, n_ch)
    scatter_neg: np.ndarray,   # (n_neg, 2) FSC/SSC negative events
    k: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-event AF subtraction via kNN scatter matching.

    Returns
    -------
    spectral_sub        : (n_pos, n_ch) AF-subtracted spectral events
    matched_neg_scatter : (n_pos, 2) mean scatter of matched negatives
                         (for display in ScatterCleaningViewer)
    """
    from sklearn.neighbors import NearestNeighbors

    if len(scatter_neg) == 0 or len(scatter_pos) == 0:
        return spec_pos.copy(), np.empty((0, 2))

    k_actual = min(k, len(scatter_neg))
    nn = NearestNeighbors(n_neighbors=k_actual, algorithm='ball_tree')
    nn.fit(scatter_neg)
    _, indices = nn.kneighbors(scatter_pos)         # (n_pos, k)

    af_matched = spec_neg[indices].mean(axis=1)     # (n_pos, n_ch)
    spectral_sub = spec_pos - af_matched
    matched_neg_scatter = scatter_neg[indices].mean(axis=1)  # (n_pos, 2)

    return spectral_sub, matched_neg_scatter

def cosine_filter(
    spec_events: np.ndarray,
    af_median: np.ndarray,
    peak_ch_idx: int,
    n_candidates: int = 1000,
    n_spectral: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select events least similar to AF by cosine similarity.

    1. Top n_candidates events by peak channel value.
    2. Cosine similarity of each to af_median.
    3. Keep n_spectral with the lowest cosine similarity.

    Returns
    -------
    selected_idx : (n_spectral,) indices into spec_events
    cs_vals      : (n_candidates,) cosine similarity values (for QC plot)
    """
    n_cand = min(n_candidates, len(spec_events))
    peak_vals = spec_events[:, peak_ch_idx]
    top_idx = np.argpartition(peak_vals, -n_cand)[-n_cand:]

    top_mat  = spec_events[top_idx]
    norm_af   = np.sqrt(np.dot(af_median, af_median)) + 1e-9
    norm_rows = np.linalg.norm(top_mat, axis=1) + 1e-9
    cs_vals   = (top_mat @ af_median) / (norm_rows * norm_af)

    n_spec       = min(n_spectral, n_cand)
    keep_order   = np.argsort(cs_vals)[:n_spec]
    selected_idx = top_idx[keep_order]

    return selected_idx, cs_vals

# ---------------------------------------------------------------------------
# background_check
# ---------------------------------------------------------------------------

BACKGROUND_CHECK_VERSION = 1
BACKGROUND_RATIO_MAX = 2.0            # control / unstained peak signal below this: no clear stain
BACKGROUND_ANGLE_MAX_DEG = 15.0       # spectrum within this angle of the AF subspace: AF-shaped
BACKGROUND_POSITIVE_FRACTION_MIN = 0.01   # fallback: events above the unstained 99.5th percentile
BACKGROUND_AF_COMPONENTS = 3          # AF subspace = unstained mean + this many principal components
BACKGROUND_NULL_SPLITS = 5
BACKGROUND_SUBSPACE_EVENTS = 20_000
_MIN_NULL_CANDIDATES = 20
_MIN_NULL_KEPT = 10


def background_check(
    pos_events: np.ndarray,
    neg_events: np.ndarray,
    neg_scatter: np.ndarray,
    spectral_sub: np.ndarray,
    spectrum: np.ndarray,
    peak_ch_idx: int,
    n_candidates: int,
    n_spectral: int,
) -> dict:
    """
    Is a cleaned control distinguishable from its unstained negative?

    ratio : median peak-channel signal of the control's cleaned events
        (``spectral_sub``) over the same statistic from the identical
        selection (top peak-channel quantile, cosine filter against the
        unstained median, kNN scatter subtraction) run on the unstained
        itself, split in half (one half as "control", the other as its
        negative), median over 5 splits. The
        candidate and kept fractions match the control's, so both sides
        are the same order statistic.
    angle_deg : angle between ``spectrum`` and the span of the unstained
        mean and its top principal components.
    positive_fraction : fraction of control events above the unstained
        99.5th percentile in the peak channel; decides brightness when
        the ratio is undefined.
    verdict : 'background' (dim and AF-shaped), 'weak' (dim only) or None.
    """
    rng = np.random.default_rng(0)
    n_pos, n_neg = len(pos_events), len(neg_events)

    n_cand = min(n_candidates, n_pos)
    cand_fraction = n_cand / max(n_pos, 1)
    keep_fraction = min(n_spectral, n_cand) / max(n_cand, 1)

    control_signal = float(np.median(spectral_sub[:, peak_ch_idx])) if len(spectral_sub) else float('nan')

    null_signals = []
    if n_neg >= 2 * _MIN_NULL_CANDIDATES:
        af_median = np.median(neg_events, axis=0)
        for _ in range(BACKGROUND_NULL_SPLITS):
            perm = rng.permutation(n_neg)
            half_a, half_b = perm[: n_neg // 2], perm[n_neg // 2:]
            null_cand = max(_MIN_NULL_CANDIDATES, round(cand_fraction * len(half_a)))
            null_keep = max(_MIN_NULL_KEPT, round(keep_fraction * null_cand))
            spec_a = neg_events[half_a]
            sel, _ = cosine_filter(spec_a, af_median, peak_ch_idx,
                                   n_candidates=null_cand, n_spectral=null_keep)
            sub, _ = knn_scatter_match(spec_a[sel], neg_scatter[half_a][sel],
                                       neg_events[half_b], neg_scatter[half_b])
            null_signals.append(float(np.median(sub[:, peak_ch_idx])))
    null_signal = float(np.median(null_signals)) if null_signals else float('nan')
    ratio = (control_signal / null_signal
             if np.isfinite(null_signal) and null_signal > 0 and np.isfinite(control_signal)
             else float('nan'))

    threshold = np.quantile(neg_events[:, peak_ch_idx], 0.995)
    positive_fraction = float(np.mean(pos_events[:, peak_ch_idx] > threshold))

    sub_neg = (neg_events if n_neg <= BACKGROUND_SUBSPACE_EVENTS
               else neg_events[rng.choice(n_neg, BACKGROUND_SUBSPACE_EVENTS, replace=False)])
    mean = sub_neg.mean(axis=0)
    _, _, vt = np.linalg.svd(sub_neg - mean, full_matrices=False)
    basis, _ = np.linalg.qr(np.vstack([mean, vt[:BACKGROUND_AF_COMPONENTS]]).T)
    s = np.asarray(spectrum, dtype=float)
    s_norm = np.linalg.norm(s)
    cos = np.linalg.norm(basis.T @ s) / s_norm if s_norm > 0 else 1.0
    angle_deg = float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))

    bright = (ratio >= BACKGROUND_RATIO_MAX if np.isfinite(ratio)
              else positive_fraction >= BACKGROUND_POSITIVE_FRACTION_MIN)
    if bright:
        verdict = None
    else:
        verdict = 'background' if angle_deg < BACKGROUND_ANGLE_MAX_DEG else 'weak'

    def _finite(x: float):
        return x if np.isfinite(x) else None  # JSON-safe

    return {
        'ratio': _finite(ratio),
        'control_signal': _finite(control_signal),
        'null_signal': _finite(null_signal),
        'angle_deg': angle_deg,
        'positive_fraction': positive_fraction,
        'verdict': verdict,
    }
