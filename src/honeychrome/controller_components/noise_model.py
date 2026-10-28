"""
noise_model.py
--------------
Detector noise model for the AutoSpectral tab's Noise page and noise-weighted
unmixing (gls_unmix.py). No Qt. With every option off, the same numerics as
the GLS Test plugin's gls_noise_model.py.

Model
    Var(y_d) = sigma0_d^2 + mu_d / kappa_d (+ c_d mu_d^2)

    ``sigma0_d^2`` is the additive floor of detector d (read noise), and
    ``kappa_d`` its photon conversion factor, so ``mu_d / kappa_d`` is shot
    noise; the optional ``c_d`` term absorbs variance growing with the square
    of the signal (shape variation the projection does not span). Photon
    partitioning across detectors adds no covariance: Poisson photons split
    by a spectrum give independent Poisson counts per detector, so the
    covariance is diagonal.

    Options (``NoiseSettings``):

    * ``sigma0_by_class``: one ``sigma0`` per particle class ('cells',
      'beads'; ``Group.cls``) with ``kappa`` shared. ``sigma0_sq`` is then
      the first class's row and ``sigma0_sq_by_class`` (K x D) holds all;
      ``for_class`` gives the model of one class.
    * ``per_control_quadratic``: one non-negative ``c_g mu^2`` per control
      group (zero for negatives) in the joint fit, absorbing control-specific
      excess variance so that it does not bias ``kappa``. The ``c_g`` are
      stored (``group_c``) for calibration and QC; ``variance`` does not use
      them, so unmixing is unchanged by them.
    * ``sigma0_fallback = 'laser'``: a detector whose ``sigma0`` was clamped
      high or raised to the floor takes the median ``sigma0`` of the
      unflagged detectors on its laser (``lasers``), per class; with two
      classes, the other class's value at that detector scaled by the
      per-laser level ratio when the other class identifies it. Fewer than 3
      unflagged detectors on the laser: the class's global median. The value
      never exceeds the detector's upper bound. With ``'floor'`` the clamp or
      floor value is kept.

Estimation (joint mixing NNLS)
    Each group of events is projected by OLS onto its own design:

    * a single-stain control onto its reference spectrum, the mean of its
      variant deltas, its variant basis and (cell controls) the AF basis of
      its negative;
    * the unstained sample onto the AF basis;
    * a bead negative onto its mean only, or onto its own AF basis.

    With heteroscedastic noise the OLS residual of detector d mixes in the
    variance of every detector: ``E[r_d^2] = sum_e M_de V_e(mu_e)`` with
    ``M = (I - H) o (I - H)`` and ``H`` the detector-space hat matrix. Events
    are split into quantile bins of the group's abundance (a control's own
    coefficient, the unstained sample's first AF coefficient), and each bin
    gives one equation per detector: its robust residual variance against
    the bin-mean fitted signal. All groups, bins and detectors are stacked
    and one non-negative least-squares problem is solved for ``s =
    sigma0^2``, ``g = 1 / kappa`` (and ``c``, ``c_g``), weighted by ``n /
    v^2`` (the variance of a variance), with IRLS passes on the fitted
    variance. The first solve uses the squared normalised MAD of each bin;
    later rounds standardise the residuals by the current model before
    taking the MAD (``bin_stats``), which removes the scale-mixture bias of a
    MAD over events of different brightness.

    Afterwards ``s_d`` is clamped above at the lowest bin variance any group
    shows at d (divided by ``M_dd``, an upper bound, widened by the bin
    variance's sampling SD) and below at ``floor_frac`` of the median ``s``,
    so a dark detector on bead events cannot take an unbounded weight; ``g``
    is refitted with ``s`` fixed when a clamp applied. A detector with no
    shot-noise information (``g = 0``) takes the median ``g`` and is
    flagged.

Calibration
    The model is judged predictively on held-out events projected with the
    same designs: per detector and intensity tranche, the squared normalised
    MAD of ``r_d / sqrt(sum_e M_de V_e)`` (1 when the model is right), and
    per group the share of events whose GLS chi-square has p < 0.001. Each
    set uses its class's ``sigma0`` and, for a control, its own ``c_g``.

Clamped data (``negatives_clamped``)
    Data whose negatives have no values at or below zero on any detector
    (share below ``CLAMP_MIN_SHARE``) are clamped at zero by the export, and
    their noise is truncated there; gls_unmix then unmixes them with
    Poisson weights with a floor (``poisson_model``) or flat weights
    (``flat_model``).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

NOISE_MODEL_VERSION = 2
READABLE_VERSIONS = (1, 2)
FLAG_SIGMA0_HIGH = 1      # clamped at the lowest bin variance
FLAG_SIGMA0_LOW = 2       # raised to the floor
FLAG_KAPPA_POOLED = 4     # no shot-noise information; median 1/kappa
FLAG_SIGMA0_FALLBACK = 8  # sigma0 from the same laser's other detectors
FLAG_LABELS = ((FLAG_SIGMA0_HIGH, 'sigma0 clamped high'), (FLAG_SIGMA0_LOW, 'sigma0 raised to floor'),
               (FLAG_KAPPA_POOLED, 'kappa pooled'), (FLAG_SIGMA0_FALLBACK, 'sigma0 from same laser'))
CLASSES = ('cells', 'beads')
CLAMP_MIN_SHARE = 0.01
CLAMP_MIN_EVENTS = 200
_FALLBACK_MIN_DETECTORS = 3
_TINY = 1e-12


def robust_var(x: np.ndarray, axis=0) -> np.ndarray:
    """Squared normalised MAD."""
    med = np.median(x, axis=axis, keepdims=True)
    return (1.4826 * np.median(np.abs(x - med), axis=axis)) ** 2


def flag_text(flags: int) -> str:
    return ', '.join(label for bit, label in FLAG_LABELS if int(flags) & bit)


# ---------------------------------------------------------------------------
# Groups and projection
# ---------------------------------------------------------------------------

@dataclass
class Group:
    """
    Events of one source with their projection design. ``design`` (rows x
    D) is None for a mean-only group (bead negative). ``offset`` (D,) is
    subtracted before projecting and added back to the fitted signal (bead
    background). ``bin_row`` is the design row whose coefficient is the
    binning variable. ``cls`` is the particle class ('cells', 'beads').
    """
    name: str
    kind: str
    raw: np.ndarray
    design: np.ndarray | None = None
    offset: np.ndarray | None = None
    bin_row: int = 0
    cls: str = 'cells'


@dataclass
class Projection:
    coef_bin: np.ndarray      # (n,) binning variable
    fit: np.ndarray           # (n, D) fitted signal, offset included
    resid: np.ndarray         # (n, D)
    centred: np.ndarray       # (n, D) events minus offset
    q: np.ndarray | None      # (k, D) orthonormal rows spanning the design
    M: np.ndarray             # (D, D)


def orthonormal_rows(design: np.ndarray, rtol: float = 1e-8) -> np.ndarray:
    """Orthonormal rows spanning ``design``, dropping directions below ``rtol`` of the largest."""
    _, sv, vt = np.linalg.svd(np.asarray(design, dtype=np.float64), full_matrices=False)
    keep = sv > rtol * max(float(sv[0]), _TINY)
    return vt[keep]


def mixing_matrix(q: np.ndarray | None, n_detectors: int, n_events: int = 0) -> np.ndarray:
    """``(I - H) o (I - H)``; for a mean-only group ``(1 - 1/n) I``."""
    if q is None:
        return np.eye(n_detectors) * (1.0 - 1.0 / max(n_events, 2))
    resid_maker = np.eye(n_detectors) - q.T @ q
    return resid_maker * resid_maker


def project(group: Group) -> Projection:
    raw = np.asarray(group.raw, dtype=np.float64)
    n, d = raw.shape
    offset = np.zeros(d) if group.offset is None else np.asarray(group.offset, dtype=np.float64)
    centred = raw - offset
    if group.design is None:
        mean = centred.mean(axis=0)
        fit = np.broadcast_to(mean + offset, raw.shape).copy()
        return Projection(np.zeros(n), fit, centred - mean, centred, None, mixing_matrix(None, d, n))
    design = np.asarray(group.design, dtype=np.float64)
    q = orthonormal_rows(design)
    within = (centred @ q.T) @ q
    coef = centred @ np.linalg.pinv(design)
    return Projection(coef[:, group.bin_row], within + offset, centred - within, centred, q,
                      mixing_matrix(q, d))


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

@dataclass
class Bins:
    """Per-bin statistics of one group: (B,) counts and (B, D) arrays."""
    n: np.ndarray
    var: np.ndarray
    mu: np.ndarray
    mu2: np.ndarray
    centre: np.ndarray


def bin_members(proj: Projection, n_bins: int, min_bin_events: int, mean_only: bool) -> list:
    """Event indices of each quantile bin of the binning variable (one bin for a mean-only group)."""
    n = len(proj.fit)
    if mean_only or n < 2 * min_bin_events:
        return [np.arange(n)] if n >= min_bin_events else []
    nb = int(max(1, min(n_bins, n // min_bin_events)))
    edges = np.quantile(proj.coef_bin, np.linspace(0.0, 1.0, nb + 1))
    which = np.clip(np.searchsorted(edges, proj.coef_bin, side='right') - 1, 0, nb - 1)
    members = [np.flatnonzero(which == b) for b in range(nb)]
    return [m for m in members if len(m) >= min_bin_events]


def bin_stats(proj: Projection, members: list, pred: np.ndarray | None = None) -> Bins:
    """
    Per bin and detector: the residual variance, the mean fitted signal
    (clamped at 0) and its mean square.

    Without ``pred`` the variance is the squared normalised MAD of the
    residual. Events in a bin differ in signal, so the residual is a scale
    mixture and its MAD understates the mean variance, more so the wider
    the spread of shot noise in the bin. With ``pred`` (the current model's
    residual variance per event) the variance is the bin mean of ``pred``
    times the squared normalised MAD of ``resid / sqrt(pred)``, which is
    homoscedastic when the model's shape is right.
    """
    mu = np.maximum(proj.fit, 0.0)
    d = mu.shape[1]
    if not members:
        empty = np.zeros((0, d))
        return Bins(np.zeros(0, dtype=np.int64), empty, empty, empty, np.zeros(0))
    if pred is None:
        var = [robust_var(proj.resid[m]) for m in members]
    else:
        var = [pred[m].mean(axis=0) * robust_var(proj.resid[m] / np.sqrt(pred[m])) for m in members]
    return Bins(
        n=np.array([len(m) for m in members], dtype=np.int64),
        var=np.vstack(var),
        mu=np.vstack([mu[m].mean(axis=0) for m in members]),
        mu2=np.vstack([(mu[m] ** 2).mean(axis=0) for m in members]),
        centre=np.array([float(np.median(proj.coef_bin[m])) for m in members]),
    )


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------

@dataclass
class NoiseModel:
    """
    Per-detector ``sigma0_sq``, ``g = 1 / kappa`` and ``c`` (D,), with
    ``flags`` (bit mask per detector, see FLAG_LABELS), the binned fit table
    and the settings it was fitted with. With a sigma0 per particle class,
    ``sigma0_sq_by_class`` (K x D) and ``flags_by_class`` hold one row per
    entry of ``classes``, ``sigma0_sq`` is the first row and ``flags`` the
    union. ``group_c`` maps a control group's name to its ``c_g`` (per-
    control quadratic; not used by ``variance``).
    """
    detectors: list
    sigma0_sq: np.ndarray
    g: np.ndarray
    c: np.ndarray
    flags: np.ndarray
    fit_table: dict = field(default_factory=dict)
    groups: list = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    source: str = 'fitted'
    fingerprint: str = ''
    classes: list = field(default_factory=list)
    sigma0_sq_by_class: np.ndarray | None = None
    flags_by_class: np.ndarray | None = None
    group_c: dict = field(default_factory=dict)

    @property
    def kappa(self) -> np.ndarray:
        with np.errstate(divide='ignore'):
            return np.where(self.g > 0, 1.0 / np.maximum(self.g, _TINY), np.inf)

    def sigma0_sq_for(self, cls: str | None = None) -> np.ndarray:
        """``sigma0^2`` of a particle class; the model's first class when ``cls`` is None or not fitted."""
        if cls is None or self.sigma0_sq_by_class is None or cls not in self.classes:
            return self.sigma0_sq
        return self.sigma0_sq_by_class[self.classes.index(cls)]

    def for_class(self, cls: str | None) -> 'NoiseModel':
        """The model with ``sigma0_sq`` of ``cls`` (itself when it has one sigma0)."""
        if self.sigma0_sq_by_class is None or cls is None or cls not in self.classes:
            return self
        k = self.classes.index(cls)
        return replace(self, sigma0_sq=self.sigma0_sq_by_class[k],
                       flags=self.flags_by_class[k] if self.flags_by_class is not None else self.flags)

    def variance(self, mu: np.ndarray, sigma0_mult: float = 1.0, cls: str | None = None) -> np.ndarray:
        """``sigma0^2 m^2 + mu+ / kappa + c mu+^2`` with ``mu+ = max(mu, 0)`` and ``m`` the sigma0 multiplier."""
        mu_pos = np.maximum(mu, 0.0)
        var = mu_pos * self.g
        var += (sigma0_mult ** 2) * self.sigma0_sq_for(cls)
        if np.any(self.c):
            var += self.c * mu_pos * mu_pos
        return np.maximum(var, _TINY)

    def crossover(self) -> np.ndarray:
        """Signal at which shot noise equals the floor, ``sigma0^2 kappa``."""
        return self.sigma0_sq * self.kappa


def flat_model(detectors: list, phi: float) -> NoiseModel:
    """A noise model with the same variance ``phi`` at every detector and signal (flat weights)."""
    d = len(detectors)
    return NoiseModel(list(detectors), np.full(d, float(phi)), np.zeros(d), np.zeros(d),
                      np.zeros(d, dtype=np.int64), source='flat')


def poisson_model(detectors: list, g: np.ndarray, floor: float, scale: float = 1.0) -> NoiseModel:
    """Poisson weights with a floor: ``V_d = scale g_d (floor + mu+)``, ``g = 1 / kappa`` per detector."""
    g = np.asarray(g, dtype=np.float64) * float(scale)
    d = len(detectors)
    return NoiseModel(list(detectors), g * float(floor), g, np.zeros(d), np.zeros(d, dtype=np.int64),
                      source='poisson')


def dispersion_scale(model: NoiseModel, proj: Projection) -> float:
    """Observed / predicted residual variance of a projected group, median over detectors."""
    observed = robust_var(proj.resid)
    predicted = np.median(predicted_resid_var(model, proj), axis=0)
    return float(np.median(observed / np.maximum(predicted, _TINY)))


def negatives_clamped(negatives: list, min_share: float = CLAMP_MIN_SHARE,
                      min_events: int = CLAMP_MIN_EVENTS) -> tuple:
    """
    ``(clamped, share)``: whether no detector has ``min_share`` or more of
    its values at or below zero over the negative events (list of (n, D)
    arrays), and that share per detector. Not clamped with fewer than
    ``min_events`` events.
    """
    events = [np.asarray(x) for x in negatives if x is not None and len(x)]
    if not events:
        return False, None
    stacked = np.vstack(events)
    share = np.mean(stacked <= 0, axis=0)
    if len(stacked) < min_events:
        return False, share
    return bool(np.all(share < min_share)), share


def _weighted_nnls(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``argmin_{x >= 0} sum w (a x - b)^2`` through a QR reduction to the column count."""
    from scipy.optimize import nnls

    sw = np.sqrt(np.maximum(w, 0.0))
    aw = a * sw[:, np.newaxis]
    bw = b * sw
    scale = np.linalg.norm(aw, axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    aw /= scale
    q, r = np.linalg.qr(aw, mode='reduced')
    sol = nnls(r, q.T @ bw, maxiter=50 * a.shape[1])[0]
    return sol / scale


@dataclass
class _Rows:
    a_s: np.ndarray
    a_g: np.ndarray
    a_c: np.ndarray | None
    c_sum: np.ndarray
    b: np.ndarray
    n: np.ndarray
    group: np.ndarray
    bin: np.ndarray
    det: np.ndarray
    mu_own: np.ndarray


def _stack_rows(projections: list, bins: list, quadratic: bool = True) -> _Rows:
    """One row per group, bin and detector; ``a_c`` only when ``quadratic`` (``c_sum`` is its row sum)."""
    a_s, a_g, a_c, c_sum, b, n, grp, bn, det, mu_own = [], [], [], [], [], [], [], [], [], []
    for gi, (proj, bi) in enumerate(zip(projections, bins)):
        d = proj.M.shape[0]
        for k in range(len(bi.n)):
            a_s.append(proj.M)
            a_g.append(proj.M * bi.mu[k][np.newaxis, :])
            if quadratic:
                a_c.append(proj.M * bi.mu2[k][np.newaxis, :])
            c_sum.append(proj.M @ bi.mu2[k])
            b.append(bi.var[k])
            n.append(np.full(d, bi.n[k], dtype=np.float64))
            grp.append(np.full(d, gi))
            bn.append(np.full(d, k))
            det.append(np.arange(d))
            mu_own.append(bi.mu[k])
    if not b:
        raise ValueError('No group has enough events for a noise model bin.')
    return _Rows(np.vstack(a_s), np.vstack(a_g), np.vstack(a_c) if quadratic else None, np.concatenate(c_sum),
                 np.concatenate(b), np.concatenate(n), np.concatenate(grp), np.concatenate(bn),
                 np.concatenate(det), np.concatenate(mu_own))


@dataclass
class NoiseSettings:
    """
    ``n_restandardise`` rounds recompute the bin variances from residuals
    standardised by the current model (``bin_stats``); ``n_irls`` passes
    reweight each solve by the fitted variance. ``bound_sds`` widens the
    upper clamp on ``sigma0^2`` by that many sampling SDs beyond the
    expected lowest of the detector's bins (``_sigma0_upper_bound``).
    ``per_control_quadratic``, ``sigma0_by_class`` and ``sigma0_fallback``
    are described in the module docstring.
    """
    n_bins: int = 20
    min_bin_events: int = 50
    fit_quadratic: bool = False
    n_irls: int = 2
    n_restandardise: int = 2
    floor_frac: float = 0.01
    bound_sds: float = 1.0
    per_control_quadratic: bool = False
    sigma0_by_class: bool = False
    sigma0_fallback: str = 'floor'


_MAD_VAR_REL = 2.72       # n x relative variance of a squared normalised MAD (Gaussian)


@dataclass
class _Layout:
    """Column layout of the joint fit: K sigma0 blocks, g, optional c, then one c_g per control group."""
    d: int
    k: int
    cls_of_group: np.ndarray
    quadratic: bool
    controls: np.ndarray       # group indices with a c_g column

    @property
    def n_cols(self) -> int:
        return (self.k + 1 + int(self.quadratic)) * self.d + len(self.controls)

    def design(self, rows: _Rows, with_s: bool = True) -> np.ndarray:
        blocks = []
        if with_s:
            blocks.extend(self.s_blocks(rows))
        blocks.append(rows.a_g)
        if self.quadratic:
            blocks.append(rows.a_c)
        if len(self.controls):
            a_cg = np.zeros((len(rows.b), len(self.controls)))
            for j, gi in enumerate(self.controls):
                on = rows.group == gi
                a_cg[on, j] = rows.c_sum[on]
            blocks.append(a_cg)
        return np.hstack(blocks)

    def s_blocks(self, rows: _Rows) -> list:
        if self.k == 1:
            return [rows.a_s]
        cls = self.cls_of_group[rows.group]
        return [rows.a_s * (cls == k)[:, np.newaxis] for k in range(self.k)]

    def split(self, sol: np.ndarray, n_groups: int) -> tuple:
        """``(s (K, D), g, c, c_g (n_groups,))``."""
        d, k = self.d, self.k
        s = sol[:k * d].reshape(k, d).copy()
        g = sol[k * d:(k + 1) * d].copy()
        pos = (k + 1) * d
        c = sol[pos:pos + d].copy() if self.quadratic else np.zeros(d)
        pos += d if self.quadratic else 0
        cg = np.zeros(n_groups)
        cg[self.controls] = sol[pos:pos + len(self.controls)]
        return s, g, c, cg

    def rest(self, g: np.ndarray, c: np.ndarray, cg: np.ndarray) -> np.ndarray:
        """The non-sigma0 part of a solution vector."""
        return np.concatenate([g] + ([c] if self.quadratic else []) + [cg[self.controls]])


def _solve_rows(a: np.ndarray, b: np.ndarray, n: np.ndarray, n_irls: int) -> np.ndarray:
    sol = _weighted_nnls(a, b, n / np.maximum(b, _TINY) ** 2)
    for _ in range(max(0, int(n_irls))):
        sol = _weighted_nnls(a, b, n / np.maximum(a @ sol, _TINY) ** 2)
    return sol


def _sigma0_upper_bound(rows: _Rows, ok: np.ndarray, projections: list, bins: list, d: int,
                        extra_sds: float) -> np.ndarray:
    """
    Upper bound on ``sigma0_d^2``: ``V_d <= v_bd / M_dd`` in every bin. Each
    bin's variance has a relative sampling SD of about ``sqrt(2.72 / n)``,
    and the smallest of ``N`` such estimates lies about ``sqrt(2 ln N)`` SDs
    low, so each bound is divided by ``1 - z * rel_sd`` with ``z = sqrt(2 ln
    N_d) + extra_sds`` before the minimum is taken; bins too noisy for that
    (``z * rel_sd >= 0.9``) are skipped. Only rows in ``ok`` count.
    """
    m_diag = np.concatenate([np.tile(np.diag(p.M), len(bi.n)) for p, bi in zip(projections, bins)])
    rel = np.sqrt(_MAD_VAR_REL / np.maximum(rows.n, 1.0))
    n_rows = np.bincount(rows.det[ok], minlength=d).astype(np.float64)
    z = np.sqrt(2.0 * np.log(np.maximum(n_rows, 2.0))) + extra_sds
    shrink = 1.0 - z[rows.det] * rel
    usable = ok & (shrink > 0.1)
    bound = np.where(usable, rows.b / np.maximum(m_diag, _TINY) / np.where(usable, shrink, 1.0), np.inf)
    upper = np.full(d, np.inf)
    np.minimum.at(upper, rows.det, bound)
    return upper


def _laser_fallback(s: np.ndarray, flagged: np.ndarray, upper: np.ndarray, lasers) -> tuple:
    """
    ``(s, applied)``: ``sigma0^2`` (K, D) with each flagged detector of each
    class replaced as the module docstring describes (``sigma0_fallback =
    'laser'``), capped at its upper bound.
    """
    k_n, d = s.shape
    las = np.asarray([str(x) for x in lasers]) if lasers is not None else np.full(d, '')
    sd = np.sqrt(np.maximum(s, 0.0))
    out = sd.copy()
    applied = np.zeros((k_n, d), dtype=bool)
    for k in range(k_n):
        ok_k = ~flagged[k] & (sd[k] > 0)
        if not ok_k.any():
            continue
        global_med = float(np.median(sd[k, ok_k]))
        for i in np.flatnonzero(flagged[k]):
            same = las == las[i]
            value = None
            for j in range(k_n):
                if j == k or flagged[j, i] or sd[j, i] <= 0:
                    continue
                both = ~flagged[k] & ~flagged[j] & (sd[k] > 0) & (sd[j] > 0)
                ref = both & same if np.count_nonzero(both & same) >= _FALLBACK_MIN_DETECTORS else both
                if ref.any():
                    value = sd[j, i] * float(np.median(sd[k, ref] / sd[j, ref]))
                    break
            if value is None:
                peers = ok_k & same
                peers[i] = False
                value = float(np.median(sd[k, peers])) if np.count_nonzero(peers) >= _FALLBACK_MIN_DETECTORS \
                    else global_med
            out[k, i] = min(value, float(np.sqrt(upper[k, i]))) if np.isfinite(upper[k, i]) else value
            applied[k, i] = True
    return out ** 2, applied


def fit_noise_model(groups: list, detectors: list, settings: NoiseSettings | None = None,
                    lasers: list | None = None) -> NoiseModel:
    """Joint mixing NNLS over every group (see the module docstring). ``lasers`` (D,) for the laser fallback."""
    s = settings or NoiseSettings()
    d = len(detectors)
    n_groups = len(groups)
    if s.sigma0_by_class:
        classes = [c for c in CLASSES if any(gr.cls == c for gr in groups)] or [CLASSES[0]]
    else:
        classes = [groups[0].cls if groups else CLASSES[0]]
    cls_of_group = np.array([classes.index(gr.cls) if gr.cls in classes else 0 for gr in groups], dtype=np.int64)
    k_n = len(classes)
    controls = (np.array([i for i, gr in enumerate(groups) if gr.kind == 'control'], dtype=np.int64)
                if s.per_control_quadratic else np.zeros(0, dtype=np.int64))
    lay = _Layout(d, k_n, cls_of_group, bool(s.fit_quadratic), controls)

    projections = []
    for gr in groups:
        proj = project(gr)
        proj.centred = None          # used by the calibration's chi-square only
        projections.append(proj)
    members = [bin_members(p, s.n_bins, s.min_bin_events, gr.design is None) for p, gr in zip(projections, groups)]

    def solve(bins):
        rows = _stack_rows(projections, bins, s.fit_quadratic)
        ok = np.isfinite(rows.b) & (rows.b > 0)
        if not ok.any():
            raise ValueError('Every binned residual variance is zero or not finite.')
        a = lay.design(rows)
        sol = _solve_rows(a if ok.all() else a[ok], rows.b[ok], rows.n[ok], s.n_irls)
        del a
        return (rows, ok) + lay.split(sol, n_groups)

    def floored(s_mat):
        out = s_mat.copy()
        for k in range(k_n):
            positive = s_mat[k][s_mat[k] > 0]
            out[k] = np.maximum(s_mat[k], s.floor_frac * float(np.median(positive)) if positive.size else _TINY)
        return out

    bins = [bin_stats(p, m) for p, m in zip(projections, members)]
    rows, ok, s_mat, g, c, cg = solve(bins)
    for _ in range(max(0, int(s.n_restandardise))):
        s_cur = floored(s_mat)
        bins = [bin_stats(p, m, _group_resid_var(p, s_cur[cls_of_group[i]], g, c, cg[i]))
                for i, (p, m) in enumerate(zip(projections, members))]
        rows, ok, s_mat, g, c, cg = solve(bins)

    flags = np.zeros((k_n, d), dtype=np.int64)
    upper = np.full((k_n, d), np.inf)
    for k in range(k_n):
        ok_k = ok & (cls_of_group[rows.group] == k)
        if not ok_k.any():
            continue
        upper[k] = _sigma0_upper_bound(rows, ok_k, projections, bins, d, s.bound_sds)
        high = s_mat[k] > upper[k]
        s_mat[k] = np.where(high, upper[k], s_mat[k])
        flags[k, high] |= FLAG_SIGMA0_HIGH
        positive = s_mat[k][s_mat[k] > 0]
        floor = s.floor_frac * float(np.median(positive if positive.size else rows.b[ok_k]))
        low = s_mat[k] < floor
        s_mat[k] = np.maximum(s_mat[k], floor)
        flags[k, low] |= FLAG_SIGMA0_LOW

    clamped = (flags & (FLAG_SIGMA0_HIGH | FLAG_SIGMA0_LOW)) > 0
    if s.sigma0_fallback == 'laser' and clamped.any():
        s_mat, applied = _laser_fallback(s_mat, clamped, upper, lasers)
        flags[applied] |= FLAG_SIGMA0_FALLBACK

    if clamped.any():
        fixed = np.hstack(lay.s_blocks(rows))[ok] @ s_mat.ravel()
        rest = lay.design(rows, with_s=False)[ok]
        current = fixed + rest @ lay.rest(g, c, cg)
        sub = _weighted_nnls(rest, rows.b[ok] - fixed, rows.n[ok] / np.maximum(current, _TINY) ** 2)
        _s, g, c, cg = lay.split(np.concatenate([np.zeros(k_n * d), sub]), n_groups)

    g_pos = g[g > 0]
    if g_pos.size == 0:
        raise ValueError('No detector shows shot noise: the controls do not span enough intensity.')
    pooled = g <= 1e-6 * float(np.median(g_pos))
    g = np.where(pooled, float(np.median(g_pos)), g)
    flags[:, pooled] |= FLAG_KAPPA_POOLED

    group_c = {groups[i].name: float(cg[i]) for i in controls}
    model = NoiseModel(
        list(detectors), s_mat[0].copy(), g, np.maximum(c, 0.0), np.bitwise_or.reduce(flags, axis=0),
        groups=[{'name': gr.name, 'kind': gr.kind, 'class': classes[cls_of_group[i]], 'n_fit': int(len(gr.raw)),
                 'n_bins': int(len(bi.n)), 'c_group': float(cg[i])}
                for i, (gr, bi) in enumerate(zip(groups, bins))],
        settings={'n_bins': s.n_bins, 'min_bin_events': s.min_bin_events,
                  'fit_quadratic': bool(s.fit_quadratic), 'n_irls': s.n_irls,
                  'n_restandardise': s.n_restandardise, 'floor_frac': s.floor_frac,
                  'bound_sds': s.bound_sds, 'per_control_quadratic': bool(s.per_control_quadratic),
                  'sigma0_by_class': bool(s.sigma0_by_class), 'sigma0_fallback': s.sigma0_fallback},
        classes=list(classes), group_c=group_c)
    if k_n > 1:
        model.sigma0_sq_by_class = s_mat.copy()
        model.flags_by_class = flags.copy()
    coef = np.concatenate([s_mat.ravel(), lay.rest(g, model.c, cg)])
    model.fit_table = {
        'group': rows.group, 'bin': rows.bin, 'detector': rows.det, 'mu': rows.mu_own,
        'observed': rows.b, 'predicted': lay.design(rows) @ coef, 'n': rows.n.astype(np.int64),
        'centre': np.concatenate([np.repeat(bi.centre, d) for bi in bins]),
        'group_c': cg.copy(),
    }
    return model


def _group_resid_var(proj: Projection, sigma0_sq: np.ndarray, g: np.ndarray, c: np.ndarray,
                     c_group: float = 0.0) -> np.ndarray:
    """``sum_e M_de V_e(mu_e)`` per event of one group, with its sigma0 and its own ``c_g``."""
    mu = np.maximum(proj.fit, 0.0)
    var = sigma0_sq + g * mu
    if np.any(c) or c_group:
        var = var + (c + c_group) * mu * mu
    return np.maximum(var, _TINY) @ proj.M.T


# ---------------------------------------------------------------------------
# Calibration on held-out events
# ---------------------------------------------------------------------------

@dataclass
class CalibrationSet:
    """Held-out events of one group, projected with that group's design."""
    name: str
    kind: str
    proj: Projection
    cls: str = 'cells'


def calibration_sets(groups: list) -> list:
    return [CalibrationSet(g.name, g.kind, project(g), g.cls) for g in groups]


def _set_variance(model: NoiseModel, mu: np.ndarray, sigma0_mult: float, cls: str | None,
                  c_group: float) -> np.ndarray:
    var = model.variance(mu, sigma0_mult, cls)
    if c_group:
        mu_pos = np.maximum(mu, 0.0)
        var = var + c_group * mu_pos * mu_pos
    return var


def predicted_resid_var(model: NoiseModel, proj: Projection, sigma0_mult: float = 1.0, cls: str | None = None,
                        c_group: float = 0.0) -> np.ndarray:
    """``sum_e M_de V_e(mu_e)`` per event and detector."""
    return _set_variance(model, proj.fit, sigma0_mult, cls, c_group) @ proj.M.T


def gls_chi2(model: NoiseModel, proj: Projection, sigma0_mult: float = 1.0, chunk: int = 20_000,
             cls: str | None = None, c_group: float = 0.0) -> tuple:
    """
    Per-event chi-square of the GLS fit on the group's design, with the
    variance at the OLS fitted signal; ``(chi2, df)``. Mean-only groups use
    the residual from the group mean.
    """
    v = _set_variance(model, proj.fit, sigma0_mult, cls, c_group)
    d = v.shape[1]
    if proj.q is None:
        return np.sum(proj.resid ** 2 / v, axis=1), d
    q = proj.q
    k = q.shape[0]
    out = np.empty(len(v))
    for start in range(0, len(v), chunk):
        end = min(start + chunk, len(v))
        w = 1.0 / v[start:end]
        y = proj.centred[start:end]
        gram = np.einsum('kd,bd,ld->bkl', q, w, q)
        rhs = np.einsum('kd,bd->bk', q, w * y)
        theta = np.linalg.solve(gram, rhs[:, :, np.newaxis])[:, :, 0]
        r = y - theta @ q
        out[start:end] = np.sum(r * r * w, axis=1)
    return out, d - k


def _tranche_ratios(z: np.ndarray, mu: np.ndarray, n_tranches: int, min_events: int) -> tuple:
    """(D, T) squared normalised MAD of z in quantile tranches of each detector's mu, and counts."""
    d = z.shape[1]
    ratio = np.full((d, n_tranches), np.nan)
    count = np.zeros((d, n_tranches), dtype=np.int64)
    level = np.full((d, n_tranches), np.nan)
    for j in range(d):
        edges = np.quantile(mu[:, j], np.linspace(0.0, 1.0, n_tranches + 1))
        which = np.clip(np.searchsorted(edges, mu[:, j], side='right') - 1, 0, n_tranches - 1)
        for t in range(n_tranches):
            sel = which == t
            count[j, t] = int(sel.sum())
            if count[j, t] >= min_events:
                ratio[j, t] = (1.4826 * np.median(np.abs(z[sel, j]))) ** 2
                level[j, t] = float(np.median(mu[sel, j]))
    return ratio, count, level


def calibrate(model: NoiseModel, sets: list, n_tranches: int = 10, band: tuple = (0.8, 1.25),
              min_events: int = 30, sigma0_mult: float = 1.0, p_cut: float = 1e-3) -> dict:
    """
    Observed / predicted residual variance on held-out events.

    ``ratio`` (D, T) pools every set, ``ratio_controls`` the single-stain
    controls only (Tier A), each over quantile tranches of the detector's
    own fitted signal; z-scores are centred per set and detector first.
    ``in_band`` and ``in_band_controls`` are the shares of filled cells
    inside ``band``. ``groups`` has one row per set: events, median per-
    detector ratio, share of detectors in band, share of events with GLS
    chi-square p below ``p_cut`` and median chi-square / df. Each set is
    predicted with its class's sigma0 and, for a control, its own ``c_g``
    (``per_control_c`` is then True).
    """
    from scipy.stats import chi2 as chi2_dist

    zs, mus, kinds, rows = [], [], [], []
    for cs in sets:
        c_group = model.group_c.get(cs.name, 0.0) if cs.kind == 'control' else 0.0
        p = predicted_resid_var(model, cs.proj, sigma0_mult, cs.cls, c_group)
        z = cs.proj.resid / np.sqrt(p)
        z = z - np.median(z, axis=0)
        per_det = (1.4826 * np.median(np.abs(z), axis=0)) ** 2
        chi, df = gls_chi2(model, cs.proj, sigma0_mult, cls=cs.cls, c_group=c_group)
        df = max(int(df), 1)
        zs.append(z)
        mus.append(np.maximum(cs.proj.fit, 0.0))
        kinds.append(np.full(len(z), cs.kind, dtype=object))
        rows.append({
            'name': cs.name, 'kind': cs.kind, 'n': int(len(z)),
            'median_ratio': float(np.median(per_det)),
            'in_band': float(np.mean((per_det >= band[0]) & (per_det <= band[1]))),
            'chi2_p_small': float(np.mean(chi2_dist.sf(chi, df) < p_cut)),
            'chi2_per_df': float(np.median(chi) / df),
            'per_detector': per_det,
        })
    out = {'groups': rows, 'band': band, 'n_tranches': n_tranches, 'sigma0_mult': sigma0_mult,
           'per_control_c': bool(model.group_c)}
    if not zs:
        return out
    z_all, mu_all, kind_all = np.vstack(zs), np.vstack(mus), np.concatenate(kinds)
    for suffix, sel in (('', np.ones(len(z_all), dtype=bool)), ('_controls', kind_all == 'control')):
        if sel.sum() < n_tranches * min_events:
            out['ratio' + suffix] = None
            out['in_band' + suffix] = np.nan
            continue
        ratio, count, level = _tranche_ratios(z_all[sel], mu_all[sel], n_tranches, min_events)
        filled = np.isfinite(ratio)
        out['ratio' + suffix] = ratio
        out['count' + suffix] = count
        out['level' + suffix] = level
        out['in_band' + suffix] = float(np.mean((ratio[filled] >= band[0]) & (ratio[filled] <= band[1]))) \
            if filled.any() else np.nan
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def noise_fingerprint(spectra: np.ndarray, detectors: list, sources: list, params: dict) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(np.asarray(spectra, dtype=np.float64)).tobytes())
    h.update(json.dumps({'detectors': list(detectors), 'sources': sources, 'params': params,
                         'version': NOISE_MODEL_VERSION}, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def save_noise_model(model: NoiseModel, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = {f'fit_{k}': np.asarray(v) for k, v in model.fit_table.items()}
    extra = {}
    if model.sigma0_sq_by_class is not None:
        extra['sigma0_sq_by_class'] = model.sigma0_sq_by_class
        extra['flags_by_class'] = model.flags_by_class
    np.savez(path, sigma0_sq=model.sigma0_sq, g=model.g, c=model.c, flags=model.flags,
             detectors=np.array(json.dumps(model.detectors)), groups=np.array(json.dumps(model.groups)),
             settings=np.array(json.dumps(model.settings)), source=np.array(model.source),
             fingerprint=np.array(model.fingerprint), version=np.array(NOISE_MODEL_VERSION),
             classes=np.array(json.dumps(model.classes)), group_c=np.array(json.dumps(model.group_c)),
             **extra, **table)


def load_noise_model(path, fingerprint: str) -> NoiseModel | None:
    """The saved model (version 1 or 2) when its fingerprint matches, else None."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data['fingerprint']) != fingerprint or int(data['version']) not in READABLE_VERSIONS:
                return None
            table = {k[4:]: np.array(data[k]) for k in data.files if k.startswith('fit_')}
            model = NoiseModel(json.loads(str(data['detectors'])), np.array(data['sigma0_sq']),
                               np.array(data['g']), np.array(data['c']), np.array(data['flags']),
                               fit_table=table, groups=json.loads(str(data['groups'])),
                               settings=json.loads(str(data['settings'])), source=str(data['source']),
                               fingerprint=fingerprint)
            if 'classes' in data.files:
                model.classes = json.loads(str(data['classes']))
                model.group_c = json.loads(str(data['group_c']))
            if 'sigma0_sq_by_class' in data.files:
                model.sigma0_sq_by_class = np.array(data['sigma0_sq_by_class'])
                model.flags_by_class = np.array(data['flags_by_class'])
            return model
    except Exception as exc:
        logger.warning('load_noise_model: could not read %s (%s)', path, exc)
        return None


def detector_table(model: NoiseModel) -> tuple[list, list]:
    """``(header, rows)``: sigma0 (per class when fitted per class), kappa, c, crossover signal and flags."""
    by_class = model.sigma0_sq_by_class is not None
    s_head = [f'sigma0 ({c})' for c in model.classes] if by_class else ['sigma0']
    header = ['Detector'] + s_head + ['kappa', 'c', 'sigma0^2 kappa', 'Flags']
    kappa = model.kappa
    cross = model.crossover()
    rows = []
    for j, name in enumerate(model.detectors):
        if by_class:
            sig = [f'{np.sqrt(model.sigma0_sq_by_class[k, j]):,.1f}' for k in range(len(model.classes))]
            flags = '; '.join(f'{cls}: {flag_text(model.flags_by_class[k, j])}'
                              for k, cls in enumerate(model.classes) if model.flags_by_class[k, j])
        else:
            sig = [f'{np.sqrt(model.sigma0_sq[j]):,.1f}']
            flags = flag_text(model.flags[j])
        rows.append([name] + sig + [f'{kappa[j]:.4g}', f'{model.c[j]:.3g}', f'{cross[j]:,.0f}', flags])
    return header, rows


def groups_table(calibration: dict, selected: dict | None = None) -> tuple[list, list]:
    """``(header, rows)``: one row per held-out set; ``selected`` maps a set's name to its 'selected / root' text."""
    header = ['Source', 'Kind', 'Events', 'Median obs/pred', 'Detectors in band', 'chi2 p < 0.001',
              'Median chi2/df']
    if selected is not None:
        header.insert(3, 'Selected / root')
    rows = []
    for r in calibration.get('groups', []):
        row = [r['name'], r['kind'], f'{r["n"]:,}', f'{r["median_ratio"]:.3f}', f'{r["in_band"]:.0%}',
               f'{r["chi2_p_small"]:.1%}', f'{r["chi2_per_df"]:.2f}']
        if selected is not None:
            row.insert(3, selected.get(r['name'], ''))
        rows.append(row)
    return header, rows
