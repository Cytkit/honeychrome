"""
spectral_refinement.py

Second pass over cleaned single-stained control spectra, run as part of
Clean Controls. A port of AutoSpectral's refine.fluorophore.spectra().

The first pass measures each spectrum on the few hundred cleanest events the
cosine filter selects. Checking a spectrum against the same selection it was
fit from cannot surface bias that selection introduced, so each row is
re-measured here on the control's whole gated population:

    1. Row update (refine_row). The control is unmixed against its own row
       alone -- a single-stained control has only one dye present -- and the
       raw detector trace is regressed on that abundance in abundance bins
       (extract_raw_signature). The slope is the candidate spectrum, and
       the intercept absorbs the mean autofluorescence. A candidate is
       adopted only if it passes every acceptance gate in RefineSettings,
       and the fit is repeated until the accepted change converges.
    2. Residual spillover check (crosstalk_check). Once every row in the
       panel exists, each control is unmixed against the full panel and a
       robust (Huber) line is fitted of every other fluorophore's abundance
       on the control's own. A non-zero slope is residual spillover.

The check only sees the part of a row error lying in the span of the panel
(unmixing projects a row error through the unmixing matrix), while the raw
regression in step 1 sees all of it. Step 2 therefore reports and does not
update rows.

No Qt.

Public API:
    huber_fit(x, y, ...)
    residual_spillover(source, targets, ...)
    extract_raw_signature(raw, abundance, current_row, ...)
    refine_row(raw, row, settings=None)
    subsample_negative(negative, ...)
    make_check_pool(raw, row, negative=None, af_basis=None, ...)
    crosstalk_check(pool, spectra, labels, source, ...)
    RefineSettings, RowRefinement, CheckPool
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

_EPS = np.finfo(float).eps
_MAD_SCALE = 1.4826
AF_TARGET = 'AF'


# ---------------------------------------------------------------------------
# Settings and results
# ---------------------------------------------------------------------------

@dataclass
class RefineSettings:
    """Defaults match refine.fluorophore.spectra()."""
    n_iter: int = 3                  # maximum refit passes per row
    intercept: bool = True           # absorb a constant background offset
    n_levels: int = 60               # maximum abundance bins
    min_bin_events: int = 50
    min_events: int = 200
    max_angle: float = 5.0           # degrees; larger candidate changes are rejected
    min_explained: float = 0.8
    max_explained: float = 1.2
    max_resid: float = 0.05
    max_intercept: float = 0.05
    min_bg_align: float = -0.9
    convergence_deg: float = 0.5
    step: float = 1.0                # fraction of each accepted change applied
    unstained_threshold: float = 0.99
    unstained_margin: float = 1.3
    max_crosstalk: float = 0.1       # residual spillover slope reported as a warning
    max_check_events: int = 20_000


@dataclass
class RowRefinement:
    row: np.ndarray                  # final row, L-inf normalised
    accepted: bool                   # any candidate adopted
    log: list = field(default_factory=list)


@dataclass
class CheckPool:
    """Events kept from one control for the full-panel spillover check."""
    raw: np.ndarray                  # bright events plus a share of the bulk
    negative: np.ndarray | None = None
    af_basis: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Robust line fits
# ---------------------------------------------------------------------------

def _huber_weights(resid: np.ndarray, k: float) -> np.ndarray:
    """Huber weights per row of *resid* (targets x events)."""
    med = np.median(resid, axis=1, keepdims=True)
    scale = np.maximum(np.median(np.abs(resid - med), axis=1, keepdims=True) / 0.6745, _EPS)
    return np.minimum(1.0, k / np.maximum(np.abs(resid / scale), _EPS))


def huber_fit(x, y, k: float = 1.345, max_iter: int = 100, tol: float = 1e-4, start=None):
    """
    Huber-weighted IRLS line fits with an intercept, one per column of *y*.

    x     : (n,) or (n, p) predictors, shared by every column of y.
    y     : (n,) or (n, m) responses.
    start : optional (p + 1,) or (p + 1, m) warm start, intercept first; the
            first weights then come from its residuals instead of OLS.

    The scale is the MAD of each column's residuals (about their median,
    divided by 0.6745), recomputed every iteration. A column converges when
    no coefficient moves by more than ``tol`` times its largest coefficient
    (at least 1). Columns that do not converge within ``max_iter``, or whose
    weighted predictors have no spread, return zero coefficients, so an
    unidentifiable fit is visible rather than silently replaced by OLS.

    Returns ``(coef, converged)``: coef is (p + 1, m), intercept in row 0.
    A 1-D y gives a (p + 1,) coef and a scalar converged.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    squeeze = y.ndim == 1
    if x.ndim == 1:
        x = x[:, None]
    if squeeze:
        y = y[:, None]
    n, m = y.shape
    design = np.column_stack([np.ones(n), x])
    q = design.shape[1]
    yt = np.ascontiguousarray(y.T)                     # targets x events

    coef = np.zeros((q, m))
    converged = np.zeros(m, dtype=bool)

    if n >= q + 1:
        weights = np.ones((m, n))
        if start is not None:
            start = np.asarray(start, dtype=np.float64).reshape(q, -1)
            coef = np.broadcast_to(start, (q, m)).copy()
            weights = _huber_weights(yt - (design @ coef).T, k)

        pair_products = [(i, j, design[:, i] * design[:, j]) for i in range(q) for j in range(i, q)]
        active = np.arange(m)
        for _ in range(max_iter):
            if active.size == 0:
                break
            w = weights[active]
            ya = yt[active]
            xtwx = np.empty((active.size, q, q))
            for i, j, prod in pair_products:
                xtwx[:, i, j] = xtwx[:, j, i] = w @ prod
            xtwy = (w * ya) @ design

            # centred weighted scatter of the predictors must be non-singular
            sw = xtwx[:, 0, 0]
            scatter = xtwx[:, 1:, 1:] - np.einsum('ci,cj->cij', xtwx[:, 0, 1:], xtwx[:, 0, 1:]) \
                / np.maximum(sw, _EPS)[:, None, None]
            floor = 1e-12 * np.maximum(np.einsum('cii->c', xtwx[:, 1:, 1:]), _EPS)
            solvable = (sw > 0) & (np.linalg.eigvalsh(scatter)[:, 0] > floor)

            new = np.zeros((active.size, q))
            try:
                new = np.linalg.solve(xtwx, xtwy[:, :, None])[:, :, 0]
            except np.linalg.LinAlgError:
                for c in range(active.size):
                    try:
                        new[c] = np.linalg.solve(xtwx[c], xtwy[c])
                    except np.linalg.LinAlgError:
                        solvable[c] = False
            solvable &= np.all(np.isfinite(new), axis=1)
            new[~solvable] = 0.0

            weights[active] = _huber_weights(ya - new @ design.T, k)
            new = new.T
            moved = np.abs(new - coef[:, active]).max(axis=0)
            limit = tol * np.maximum(np.abs(new).max(axis=0), 1.0)
            coef[:, active] = new

            done = (moved < limit) & solvable
            converged[active[done]] = True
            active = active[~done & solvable]

    coef[:, ~converged] = 0.0
    if squeeze:
        return coef[:, 0], bool(converged[0])
    return coef, converged


def residual_spillover(source, targets, threshold_source=None, max_events: int = 20_000,
                       min_events: int = 200, rng=None, **huber_kwargs) -> dict | None:
    """
    Robust residual spillover from one or more source abundances into each
    target abundance, on a population in which only the sources are present
    (a single-stained control, or one barcode combination).

    source           : (n,) or (n, p) source abundances.
    targets          : (n, m) target abundances.
    threshold_source : scalar or (p,) positivity boundary of each source.

    Every event above threshold in any source is kept and the bulk below is
    subsampled, so at most ``max_events`` are fitted unless the bright
    events alone exceed it: the bulk sits at the origin and carries no
    leverage, so the slopes are unchanged while the fit's cost stops scaling
    with file size. Without a threshold, events are subsampled uniformly.

    Returns None below ``min_events``, otherwise a dict:
        slope     : (m,) for one source, (p, m) for several
        intercept : (m,)
        converged : (m,) bool; non-converged slopes are zero
        n         : events fitted
        span      : range of each source over all events
        noise     : MAD-based sd of each target over all events
    """
    source = np.asarray(source, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim == 1:
        targets = targets[:, None]
    single = source.ndim == 1
    src = source[:, None] if single else source
    n = src.shape[0]
    if n < min_events or targets.shape[1] == 0:
        return None
    rng = np.random.default_rng(0) if rng is None else rng

    index = np.arange(n)
    if n > max_events:
        if threshold_source is None:
            index = np.sort(rng.choice(n, max_events, replace=False))
        else:
            thr = np.broadcast_to(np.asarray(threshold_source, dtype=np.float64), (src.shape[1],))
            bright = np.any(src > thr, axis=1)
            bulk = np.flatnonzero(~bright)
            n_bulk = max(max_events - int(bright.sum()), min_events)
            if bulk.size > n_bulk:
                bulk = rng.choice(bulk, n_bulk, replace=False)
            index = np.sort(np.concatenate([np.flatnonzero(bright), bulk]))

    coef, converged = huber_fit(src[index], targets[index], **huber_kwargs)
    span = src.max(axis=0) - src.min(axis=0)
    noise = _MAD_SCALE * np.median(np.abs(targets - np.median(targets, axis=0)), axis=0)
    return {
        'slope': coef[1] if single else coef[1:],
        'intercept': coef[0],
        'converged': converged,
        'n': int(index.size),
        'span': float(span[0]) if single else span,
        'noise': noise,
    }


# ---------------------------------------------------------------------------
# Raw-space signature
# ---------------------------------------------------------------------------

def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = float(a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), _EPS)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def extract_raw_signature(raw: np.ndarray, abundance: np.ndarray, current_row: np.ndarray,
                          intercept: bool = True, n_levels: int = 60, min_bin_events: int = 50,
                          min_events: int = 200) -> dict | None:
    """
    Re-measure one fluorophore's spectrum from raw detector data of a
    population in which it is the only dye (extract.raw.signature() with the
    target as the only active row).

    raw         : (n, d) raw events.
    abundance   : (n,) the fluorophore's abundance under ``current_row``.
    current_row : (d,) the spectrum being re-measured.

    Events are binned by abundance quantile, as finely as ``min_bin_events``
    per bin allows up to ``n_levels``, and the bin-mean detector trace is
    regressed on the bin-mean abundance. The slope, clamped at zero and
    peak-normalised, is the candidate.

    Autofluorescence is deliberately not given a regressor of its own. Any
    per-event AF proxy built from the raw data (such as the projection onto
    the unstained mean spectrum b) also carries the dye's own loading on it,
    c = row . b / |b|^2, and a joint fit then returns row - c b rather than
    the row. Left in the intercept and residual, AF independent of the dye
    only biases the slope by its variance relative to the dye's.

    Returns None when the population cannot support a fit, otherwise a dict
    with ``signature`` (d,), ``signature_raw`` (d,) and ``stats``:
        n_events, n_bins, x_span, explained, explained_total, resid_rel,
        intercept_rel, bg_align, clamp_frac, deg_change, peak_curr,
        peak_new, peak_new_rel
    ``bg_align`` near -1 means the intercept is trading a background floor
    against the slope.
    """
    raw = np.asarray(raw, dtype=np.float64)
    abundance = np.asarray(abundance, dtype=np.float64)
    current_row = np.asarray(current_row, dtype=np.float64)
    n = raw.shape[0]
    if n < min_events:
        return None

    n_levels_use = max(3, min(int(n_levels), n // int(min_bin_events)))
    brk = np.unique(np.quantile(abundance, np.linspace(0.0, 1.0, n_levels_use + 1)))
    if brk.size < 3:
        return None
    n_cut = brk.size - 1
    bin_id = np.clip(np.searchsorted(brk, abundance, side='left'), 1, n_cut) - 1
    counts = np.bincount(bin_id, minlength=n_cut)
    present = np.flatnonzero(counts)
    if present.size < 3:
        return None
    member = sparse.csr_matrix((np.ones(n), (bin_id, np.arange(n))), shape=(n_cut, n))
    y_bin = np.asarray(member @ raw)[present] / counts[present, None]
    xt = np.asarray(member @ abundance)[present] / counts[present]
    n_bins = present.size
    if xt.max() - xt.min() <= 0:
        return None

    design = np.column_stack([np.ones(n_bins), xt]) if intercept else xt[:, None]
    coef = np.linalg.lstsq(design, y_bin, rcond=None)[0]
    fitted = design @ coef
    offset = coef[0].copy() if intercept else np.zeros(raw.shape[1])
    signature_raw = coef[1 if intercept else 0].copy()
    offset[~np.isfinite(offset)] = 0.0
    signature_raw[~np.isfinite(signature_raw)] = 0.0

    resid_rel = np.linalg.norm(y_bin - fitted) / max(np.linalg.norm(y_bin), _EPS)
    top_signal = np.linalg.norm(xt.max() * signature_raw)
    intercept_rel = np.linalg.norm(offset) / max(top_signal, _EPS)

    clamped = np.maximum(signature_raw, 0.0)
    clamp_frac = np.sum(np.maximum(-signature_raw, 0.0)) / max(np.sum(np.abs(signature_raw)), _EPS)
    if clamped.max() <= 0:
        return None
    signature = clamped / clamped.max()

    top = int(np.argmax(xt))
    top_norm = np.linalg.norm(y_bin[top])
    explained = np.linalg.norm(xt[top] * signature_raw) / top_norm if top_norm > 0 else 0.0
    explained_total = np.linalg.norm(fitted[top]) / top_norm if top_norm > 0 else 0.0

    bg_align = float('nan')
    if intercept:
        denom = np.linalg.norm(offset) * np.linalg.norm(signature_raw)
        if denom > 0:
            bg_align = float(offset @ signature_raw / denom)

    peak_curr = int(np.argmax(current_row))
    peak_new = int(np.argmax(signature))
    stats = {
        'n_events': int(n),
        'n_bins': int(n_bins),
        'x_span': float(xt.max() - xt.min()),
        'explained': float(explained),
        'explained_total': float(explained_total),
        'resid_rel': float(resid_rel),
        'intercept_rel': float(intercept_rel),
        'bg_align': bg_align,
        'clamp_frac': float(clamp_frac),
        'deg_change': _angle_deg(current_row, signature),
        'peak_curr': peak_curr,
        'peak_new': peak_new,
        'peak_new_rel': float(current_row[peak_new] / max(current_row.max(), _EPS)),
    }
    return {'signature': signature, 'signature_raw': signature_raw, 'stats': stats}


def _reject_reason(stats: dict, s: RefineSettings) -> str | None:
    if stats['deg_change'] > s.max_angle:
        return 'angle'
    if (not np.isfinite(stats['explained_total']) or stats['explained_total'] < s.min_explained
            or stats['explained_total'] > s.max_explained):
        return 'explained'
    if stats['resid_rel'] > s.max_resid:
        return 'fit'
    if stats['intercept_rel'] > s.max_intercept:
        return 'offset'
    if np.isfinite(stats['bg_align']) and stats['bg_align'] < s.min_bg_align:
        return 'bg_align'
    return None


def refine_row(raw: np.ndarray, row: np.ndarray, settings: RefineSettings | None = None) -> RowRefinement:
    """
    Re-measure one row on its own single-stained control.

    raw : (n, d) gated raw events of the control.
    row : (d,) the cleaned spectrum.

    Each pass unmixes the control against the current row alone, refits the
    row (extract_raw_signature) and applies the candidate only if it passes
    every gate in ``settings``. Passes stop at ``settings.n_iter``, at the
    first rejected candidate (the data and row are then unchanged, so a
    repeat would be rejected too), or once an accepted change falls below
    ``settings.convergence_deg``.

    Returns a RowRefinement with the final row and one log dict per pass.
    """
    s = settings or RefineSettings()
    raw = np.asarray(raw, dtype=np.float64)
    current = np.asarray(row, dtype=np.float64)
    current = current / max(current.max(), _EPS)

    log = []
    accepted_any = False
    for iteration in range(1, s.n_iter + 1):
        abundance = raw @ current / max(float(current @ current), _EPS)
        candidate = extract_raw_signature(
            raw, abundance, current, intercept=s.intercept, n_levels=s.n_levels,
            min_bin_events=s.min_bin_events, min_events=s.min_events,
        )
        reject = 'no_fit' if candidate is None else _reject_reason(candidate['stats'], s)
        entry = {'iter': iteration, 'accepted': reject is None, 'reject': reject}
        if candidate is not None:
            for key in ('deg_change', 'explained_total', 'resid_rel', 'intercept_rel', 'bg_align'):
                value = candidate['stats'][key]
                entry[key] = value if np.isfinite(value) else None    # JSON-safe
            entry['n_bins'] = candidate['stats']['n_bins']
            entry['n_events'] = candidate['stats']['n_events']
        log.append(entry)
        if reject is not None:
            break

        new = s.step * candidate['signature'] + (1.0 - s.step) * current
        current = new / max(new.max(), _EPS)
        accepted_any = True
        if candidate['stats']['deg_change'] < s.convergence_deg:
            break

    return RowRefinement(row=current, accepted=accepted_any, log=log)


# ---------------------------------------------------------------------------
# Full-panel residual spillover check
# ---------------------------------------------------------------------------

def _unmixing_matrix(spectra: np.ndarray, af_basis: np.ndarray | None) -> np.ndarray:
    basis = spectra if af_basis is None else np.vstack([spectra, af_basis / max(af_basis.max(), _EPS)])
    return np.linalg.solve(basis @ basis.T, basis)


def _source_threshold(source_abundance: np.ndarray, negative_abundance: np.ndarray | None,
                      s: RefineSettings) -> float:
    """Unstained-based positivity boundary (margin x quantile of the
    unstained's own abundance), or the control's 10th percentile without
    a usable unstained."""
    if negative_abundance is not None and negative_abundance.size >= s.min_events:
        return float(s.unstained_margin * np.quantile(negative_abundance, s.unstained_threshold))
    return float(np.quantile(source_abundance, 0.10))


def subsample_negative(negative: np.ndarray, settings: RefineSettings | None = None,
                       rng=None) -> np.ndarray:
    """Uniform float32 subsample of an unstained control, at most
    ``max_check_events`` events; returned without copying when it already
    is one."""
    s = settings or RefineSettings()
    negative = np.asarray(negative)
    if len(negative) > s.max_check_events:
        rng = np.random.default_rng(0) if rng is None else rng
        negative = negative[np.sort(rng.choice(len(negative), s.max_check_events, replace=False))]
    return np.asarray(negative, dtype=np.float32)


def make_check_pool(raw: np.ndarray, row: np.ndarray, negative: np.ndarray | None = None,
                    af_basis: np.ndarray | None = None, settings: RefineSettings | None = None,
                    rng=None) -> CheckPool:
    """
    Keep the events of one control that crosstalk_check() will fit, so the
    check can run once the whole panel is known without re-reading files.

    Brightness is judged on the control's abundance under its own row, which
    tracks its full-panel abundance closely in a single-stained control.
    Every bright event is kept, capped at ``max_check_events``, and the bulk
    fills the remainder. ``negative`` is subsampled uniformly to the same
    cap; pass the result of subsample_negative() to share one copy between
    controls with the same unstained. Arrays are stored as float32.
    """
    s = settings or RefineSettings()
    rng = np.random.default_rng(0) if rng is None else rng
    raw = np.asarray(raw)
    row = np.asarray(row, dtype=np.float64)
    norm2 = max(float(row @ row), _EPS)
    own = raw @ row / norm2
    neg_pool = None if negative is None else subsample_negative(negative, s, rng)
    neg_own = None if neg_pool is None else neg_pool @ row / norm2
    threshold = _source_threshold(own, neg_own, s)

    cap = s.max_check_events
    bright = np.flatnonzero(own > threshold)
    bulk = np.flatnonzero(own <= threshold)
    if bright.size > cap:
        bright = rng.choice(bright, cap, replace=False)
    n_bulk = max(cap - bright.size, s.min_events)
    if bulk.size > n_bulk:
        bulk = rng.choice(bulk, n_bulk, replace=False)
    keep = np.sort(np.concatenate([bright, bulk]))

    return CheckPool(
        raw=raw[keep].astype(np.float32),
        negative=neg_pool,
        af_basis=None if af_basis is None else np.asarray(af_basis, dtype=np.float64),
    )


def crosstalk_check(pool: CheckPool, spectra: np.ndarray, labels, source: str,
                    settings: RefineSettings | None = None, rng=None) -> list[dict]:
    """
    Residual spillover of one single-stained control into every other
    fluorophore of the full panel.

    pool    : the control's CheckPool. With an ``af_basis`` the unstained
              mean spectrum is unmixed as an extra endmember, so
              autofluorescence is extracted as in Honeychrome's own unmixing.
    spectra : (f, d) the full panel, rows in ``labels`` order.
    source  : label of the control's own fluorophore.

    Each target's abundance is fitted on the source abundance with
    residual_spillover(). The source positivity threshold comes from the
    pool's unstained (``unstained_margin`` x its ``unstained_threshold``
    quantile) or, without one, the control's 10th percentile.

    With an ``af_basis``, the AF abundance is a target too (labelled
    ``'AF'``): an autofluorescence-shaped row error is in the span of the
    AF endmember and shows up there rather than in another fluorophore.

    Returns one dict per target: target, slope, intercept, converged, n,
    noise, and ``flagged`` (|slope| > ``max_crosstalk``). Empty when the pool
    is too small to fit.
    """
    s = settings or RefineSettings()
    labels = list(labels)
    w = _unmixing_matrix(np.asarray(spectra, dtype=np.float64), pool.af_basis)
    k = labels.index(source)
    abundance = pool.raw.astype(np.float64) @ w.T
    neg_source = None
    if pool.negative is not None:
        neg_source = pool.negative.astype(np.float64) @ w[k]
    threshold = _source_threshold(abundance[:, k], neg_source, s)

    if pool.af_basis is not None:
        labels = labels + [AF_TARGET]
    target_idx = [i for i in range(len(labels)) if i != k]
    fit = residual_spillover(
        abundance[:, k], abundance[:, target_idx], threshold_source=threshold,
        max_events=s.max_check_events, min_events=s.min_events, rng=rng,
    )
    if fit is None:
        return []
    return [{
        'target': labels[i],
        'slope': float(fit['slope'][c]),
        'intercept': float(fit['intercept'][c]),
        'converged': bool(fit['converged'][c]),
        'n': fit['n'],
        'noise': float(fit['noise'][c]),
        'flagged': bool(abs(fit['slope'][c]) > s.max_crosstalk),
    } for c, i in enumerate(target_idx)]
