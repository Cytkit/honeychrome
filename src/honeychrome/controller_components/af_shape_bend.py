"""
af_shape_bend.py
----------------
Lets the AF spectrum of high-AF cells bend a little along learnt shapes. No Qt.

In plain terms
    Per-cell AF gives each cell the closest AF spectrum in the sample's AF
    library, scaled to the cell. In cells with a lot of AF (lung and liver
    macrophages, dead cells, the high-AF tail of most tissues) the cell's real
    AF spectrum still differs a little from every library spectrum, and at
    high AF a small difference in shape leaves a large error in the
    fluorophores whose spectra resemble AF (BUV395, BUV496, BV421, BV510 and
    their neighbours): their spread widens with AF. Here each high-AF cell's
    AF may bend along a few shapes learnt from the unstained sample its AF
    profile was extracted from, by about as much as AF of that brightness
    varies in that sample, and no more. How far to bend is read from the
    detectors the panel cannot explain and from the fluorophores that should
    be close to 0; each fluorophore's correction is worked out without that
    fluorophore's own value, so a real dim signal for a dye cannot be taken
    for AF. Cells below the unstained sample's 80th AF percentile keep their
    per-cell AF result exactly; the AF Abundance and AF Index never change.

The maths
    Learning, on the AF profile's unstained sample (LEARN_EVENTS events,
    unsaturated), unmixed with the noise-aware per-cell AF: each event has a
    variant a_j, amount k, leftover d = y - k a_j, fluorophores u = P d and
    fitted signal mu = k a_j + max(u, 0) S (S: fluorophore spectra, P the OLS
    unmixing matrix). V(mu) = sigma0^2 + g mu + c mu^2 is the noise model.

    * Shapes E (unit rows, detectors): the library's principal directions
      (SVD of the centred library, those with a singular value above
      LIBRARY_SV_FLOOR of the largest), then the principal directions of the
      noise-scaled leftover of the top 10% AF events, each event's own
      variant direction removed (an amount error is not a shape). Each is
      orthogonalised against the library's mean AF spectrum and those kept,
      and kept only if it spreads over at least MIN_PARTICIPATION detectors
      (1 / sum e^4), up to N_SHAPES from each source. A bend along the AF
      spectrum itself would be a change in AF amount, which per-cell AF
      already sets; without the mean removed, a library whose variants
      differ in only one or two directions puts every variant inside the
      shapes' span and the bend is not identifiable. A shape confined to one
      or two detectors is a detector effect, not an AF shape.
    * Bend size by AF: each event's unpenalised bend c_hat is the
      least-squares fit of d on [a_j; E] (its own variant included), and
      n its noise variance under V(mu). In each AF percentile bin
      (BIN_EDGES), sigma_l^2 = median(c_hat_l^2 / k^2) / 0.4549 -
      median(n_l / k^2), floored at 0: how much the shape varies per unit
      of AF beyond what noise alone would give (0.4549 is the median of a
      chi-square with one degree of freedom, so a median of squares reads
      as a variance).
    * Threshold k0: the 80th percentile of k.

    Applying, to events with k >= k0:

    * Prior on the bend c (one value per shape): Normal(0, T), T =
      diag(k^2 sigma_l^2(k)), sigma^2 interpolated in log k between bins.
    * Outside term: the fit of c E to d with the fluorophores refitted
      around it, with detector weights W = 1 / V(mu): A = E W E' -
      E W S' (S W S')^-1 S W E', and b likewise with d. ROBUST_PASSES times,
      detectors whose leftover after the bend is beyond ROBUST_Z noise SDs
      are down-weighted (Huber), so one detector's excess does not bend the
      whole spectrum.
    * Fluorophore term: g = E P' (each shape's effect on the fluorophores)
      and omega_f = w_f / var_f, var_f the fluorophore's noise variance
      (V(mu) through P^2, plus the spillover spread) and w_f how likely the
      fluorophore is to be near 0 (1 up to POSITIVE_OFFSET_SD SDs above 0,
      a normal tail beyond).
    * c = (A + g Omega g' + T^-1)^-1 (b + g Omega u). Fluorophore f is
      corrected by the bend estimated without its own term, exactly by a
      rank-one downdate: (g_f' c - omega_f u_f q_f) / (1 - omega_f q_f),
      q_f = g_f' (A + g Omega g' + T^-1)^-1 g_f. Reported: u minus that.

Setting (``process['af_shape_bend']``): on unless set to False. It acts only
with the noise-aware AF abundance; the bend travels on the sample's
``af_posterior.AFPosterior`` (``bend``). Learnt bends are stored in
``<experiment>/AutoSpectral/af_bends.json``, keyed by the learning sample and
a fingerprint of its prior (see af_posterior).

v1 simplifications
    * One bend per AF library, learnt on the unstained sample the library's
      first profile was extracted from, used for every sample unmixed with
      that library (stained samples, controls and the unstained sample
      itself).
    * The bend size is the learnt one (scale 1) and the threshold the 80th
      percentile; neither is chosen per sample.
    * Events with a saturated detector are not bent (their re-fit on the
      unsaturated detectors keeps the per-cell AF result).
    * Only per-cell AF (R1) reports bent fluorophores; M2pg and the
      clamped-data weights take the per-cell AF amount and row unbent, and
      spectral variants (R2) are unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.special import erfc

logger = logging.getLogger(__name__)

SHAPE_BEND_KEY = 'af_shape_bend'
BEND_VERSION = 1
BENDS_FILE = 'af_bends.json'
LEARN_EVENTS = 50_000
LEARN_SEED = 11
N_SHAPES = 3
SHAPE_QUANTILE = 0.90
MIN_PARTICIPATION = 4.0
LIBRARY_SV_FLOOR = 1e-6
BIN_EDGES = (0, 50, 80, 90, 95, 99, 100)
MIN_BIN_EVENTS = 20
FROM_QUANTILE = 0.80
POSITIVE_OFFSET_SD = 2.0
ROBUST_Z = 4.0
ROBUST_PASSES = 2
CHUNK = 2000
CHI2_1_MEDIAN = 0.45493642311957283
_TINY = 1e-300
_FIXED = 1e30


# ---------------------------------------------------------------------------
# Setting
# ---------------------------------------------------------------------------

def shape_bend_on(process: dict | None) -> bool:
    """The experiment's AF shape bend setting (on when unset)."""
    return bool((process or {}).get(SHAPE_BEND_KEY, True))


def set_shape_bend(process: dict, on: bool) -> None:
    process[SHAPE_BEND_KEY] = bool(on)


# ---------------------------------------------------------------------------
# The learnt bend
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class AFBend:
    """What one AF library's bend learnt on its unstained sample (see the module docstring)."""
    shapes: np.ndarray        # (L, D) unit rows
    log_k: np.ndarray         # (B,) median log AF of each bin
    sigma2: np.ndarray        # (B, L) shape variance per unit AF squared
    from_k: float             # events with AF at least this bend
    n_events: int
    fingerprint: str
    source: str

    @property
    def active(self) -> bool:
        return len(self.shapes) > 0 and len(self.log_k) > 0 and bool(np.isfinite(self.from_k))

    def to_dict(self) -> dict:
        return {'shapes': np.asarray(self.shapes).tolist(), 'n_detectors': int(np.asarray(self.shapes).shape[1]),
                'log_k': np.asarray(self.log_k).tolist(), 'sigma2': np.asarray(self.sigma2).tolist(),
                'from_k': float(self.from_k) if np.isfinite(self.from_k) else None,
                'n_events': int(self.n_events), 'fingerprint': self.fingerprint, 'source': self.source}

    @classmethod
    def from_dict(cls, d: dict) -> 'AFBend':
        shapes = np.asarray(d['shapes'], dtype=np.float64).reshape(-1, int(d['n_detectors']))
        log_k = np.asarray(d['log_k'], dtype=np.float64).reshape(-1)
        return cls(shapes=shapes, log_k=log_k,
                   sigma2=np.asarray(d['sigma2'], dtype=np.float64).reshape(len(log_k), len(shapes)),
                   from_k=float('inf') if d.get('from_k') is None else float(d['from_k']),
                   n_events=int(d['n_events']), fingerprint=str(d['fingerprint']), source=str(d['source']))


def empty_bend(n_detectors: int, n_events: int = 0, fingerprint: str = '', source: str = '') -> AFBend:
    """A bend that changes nothing (too few events, or no shape spread over enough detectors)."""
    return AFBend(shapes=np.zeros((0, n_detectors)), log_k=np.zeros(0), sigma2=np.zeros((0, 0)),
                  from_k=float('inf'), n_events=n_events, fingerprint=fingerprint, source=source)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def noise_variance(mu: np.ndarray, noise: tuple) -> np.ndarray:
    sigma0_sq, g, c = noise
    mu = np.maximum(mu, 0.0)
    return sigma0_sq + mu * (g + c * mu)


def _unit_rows(rows: np.ndarray) -> np.ndarray:
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), _TINY)


def _participation(v: np.ndarray) -> float:
    """Effective number of detectors a unit vector spreads over."""
    return float(1.0 / max(np.sum(v ** 4), _TINY))


def _precision(t2: np.ndarray) -> np.ndarray:
    return np.where(t2 > 0.0, 1.0 / np.maximum(t2, _TINY), _FIXED)


def prior_variance(k: np.ndarray, bend: AFBend) -> np.ndarray:
    """(k sigma_l(k))^2 per event and shape, sigma^2 interpolated in log k between the bins."""
    log_k = np.log(np.maximum(k, _TINY))
    s2 = np.column_stack([np.interp(log_k, bend.log_k, bend.sigma2[:, l]) for l in range(len(bend.shapes))])
    return s2 * (k ** 2)[:, np.newaxis]


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------

def learn_shapes(d: np.ndarray, k: np.ndarray, j: np.ndarray, mu: np.ndarray, af_spectra: np.ndarray,
                 noise: tuple) -> np.ndarray:
    """The bend's shapes (see the module docstring): library directions, then leftover directions."""
    n_det = af_spectra.shape[1]
    lib_dirs = np.zeros((0, n_det))
    if len(af_spectra) > 1:
        _u, sv, vt = np.linalg.svd(af_spectra - af_spectra.mean(axis=0), full_matrices=False)
        lib_dirs = vt[sv > LIBRARY_SV_FLOOR * max(float(sv[0]), _TINY)]
    top = k >= np.quantile(k[k > 0.0], SHAPE_QUANTILE)
    a = _unit_rows(af_spectra)[j[top]]
    left = d[top] - np.einsum('nd,nd->n', d[top], a)[:, np.newaxis] * a
    sd = np.sqrt(noise_variance(np.median(mu[top], axis=0), noise))
    res_dirs = _unit_rows(np.linalg.svd(left / sd, full_matrices=False)[2] * sd)
    mean = af_spectra.mean(axis=0)
    anchor = [mean / max(float(np.linalg.norm(mean)), _TINY)]
    kept = []
    for candidates in (_unit_rows(lib_dirs), res_dirs):
        taken = 0
        for v in candidates:
            if taken == N_SHAPES:
                break
            for q in anchor + kept:
                v = v - (v @ q) * q
            norm = np.linalg.norm(v)
            if norm < 1e-6:
                continue
            v = v / norm
            if _participation(v) >= MIN_PARTICIPATION:
                kept.append(v)
                taken += 1
    return np.array(kept).reshape(len(kept), n_det)


def bend_readings(d: np.ndarray, j: np.ndarray, mu: np.ndarray, af_spectra: np.ndarray, shapes: np.ndarray,
                  noise: tuple) -> tuple:
    """Each event's unpenalised bend along the shapes (its variant direction fitted alongside) and its noise variance."""
    c_hat = np.zeros((len(d), len(shapes)))
    noise_part = np.zeros_like(c_hat)
    var = noise_variance(mu, noise)
    a = _unit_rows(af_spectra)
    for v in np.unique(j):
        rows = j == v
        solve = np.linalg.pinv(np.vstack([a[v][np.newaxis], shapes]).T)[1:]
        c_hat[rows] = d[rows] @ solve.T
        noise_part[rows] = var[rows] @ (solve ** 2).T
    return c_hat, noise_part


def sigma_by_k(k: np.ndarray, c_hat: np.ndarray, noise_part: np.ndarray) -> tuple:
    """(median log AF of each bin, sigma_l^2 per bin and shape), bins with too few events left out."""
    pos = k > 0.0
    edges = np.percentile(k[pos], BIN_EDGES)
    centres, sigma2 = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = pos & (k >= lo) & (k <= hi)
        if sel.sum() < MIN_BIN_EVENTS:
            continue
        k2 = k[sel, np.newaxis] ** 2
        s2 = np.median(c_hat[sel] ** 2 / k2, axis=0) / CHI2_1_MEDIAN - np.median(noise_part[sel] / k2, axis=0)
        centres.append(float(np.median(np.log(k[sel]))))
        sigma2.append(np.maximum(s2, 0.0))
    return np.array(centres), np.array(sigma2).reshape(len(centres), c_hat.shape[1])


def learn_bend(raw_fl: np.ndarray, log_ssc: np.ndarray | None, precomputed: dict, af_spectra: np.ndarray,
               posterior, fingerprint: str = '', source: str = '') -> AFBend:
    """
    The bend of an AF library learnt on an unstained sample's unsaturated raw
    fluorescence events, unmixed with that sample's noise-aware posterior
    (``af_posterior.AFPosterior``, without a bend).
    """
    from honeychrome.controller_components.autospectral_functions import apply_af_unmixing

    lib = np.asarray(af_spectra, dtype=np.float64)
    n = len(raw_fl)
    empty = empty_bend(lib.shape[1], n, fingerprint, source)
    res = apply_af_unmixing(raw_fl, precomputed, lib, handle_saturation=False, af_posterior=posterior,
                            log_ssc=log_ssc, shape_bend=False)
    k = np.asarray(res['af_scale'], dtype=np.float64)
    j = np.asarray(res['af_idx'], dtype=np.int64) - 1
    pos = k > 0.0
    if pos.sum() < MIN_BIN_EVENTS * (len(BIN_EDGES) - 1):
        return empty
    u = np.asarray(res['unmixed'], dtype=np.float64)
    d = raw_fl - k[:, np.newaxis] * lib[j]
    mu = k[:, np.newaxis] * lib[j] + np.maximum(u, 0.0) @ np.asarray(precomputed['S_t'], dtype=np.float64).T
    noise = posterior.noise
    shapes = learn_shapes(d, k, j, mu, lib, noise)
    if not len(shapes):
        return empty
    log_k, sigma2 = sigma_by_k(k, *bend_readings(d, j, mu, lib, shapes, noise))
    if not len(log_k):
        return empty
    return AFBend(shapes=shapes, log_k=log_k, sigma2=sigma2, from_k=float(np.quantile(k[pos], FROM_QUANTILE)),
                  n_events=n, fingerprint=fingerprint, source=source)


# ---------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------

def _weighted_outside(d: np.ndarray, s: np.ndarray, e: np.ndarray, w: np.ndarray) -> tuple:
    """Outside term with detector weights w (n, D): E Q_W E' and E Q_W d per event (see the module docstring)."""
    n, n_l = len(d), len(e)
    a, b = np.empty((n, n_l, n_l)), np.empty((n, n_l))
    for start in range(0, n, CHUNK):
        sl = slice(start, min(n, start + CHUNK))
        sw = s[np.newaxis] * w[sl, np.newaxis, :]
        x = sw @ e.T
        m = sw @ s.T
        mx = np.linalg.solve(m, x)
        mz = np.linalg.solve(m, np.einsum('cfd,cd->cf', sw, d[sl])[:, :, np.newaxis])[:, :, 0]
        ew = e[np.newaxis] * w[sl, np.newaxis, :]
        a[sl] = ew @ e.T - np.transpose(x, (0, 2, 1)) @ mx
        b[sl] = np.einsum('cld,cd->cl', ew, d[sl]) - np.einsum('cfl,cf->cl', x, mz)
    return a, b


def _weighted_residual(d: np.ndarray, s: np.ndarray, e: np.ndarray, w: np.ndarray, c: np.ndarray) -> np.ndarray:
    """d - c E minus its fluorophore fit with detector weights w: what each detector has left."""
    r = d - c @ e
    out = np.empty_like(r)
    for start in range(0, len(r), CHUNK):
        sl = slice(start, min(len(r), start + CHUNK))
        sw = s[np.newaxis] * w[sl, np.newaxis, :]
        z = np.einsum('cfd,cd->cf', sw, r[sl])
        out[sl] = r[sl] - np.linalg.solve(sw @ s.T, z[:, :, np.newaxis])[:, :, 0] @ s
    return out


def _solve(a: np.ndarray, b: np.ndarray, t2: np.ndarray) -> np.ndarray:
    a = a + _precision(t2)[:, :, np.newaxis] * np.eye(a.shape[1])[np.newaxis]
    return np.linalg.solve(a, b[:, :, np.newaxis])[:, :, 0]


def _robust_outside(d: np.ndarray, s: np.ndarray, e: np.ndarray, w0: np.ndarray, t2: np.ndarray) -> tuple:
    """Noise-weighted outside term with Huber down-weighting of detectors far from 0 after the bend."""
    w = w0
    for _ in range(ROBUST_PASSES):
        a, b = _weighted_outside(d, s, e, w)
        c = _solve(a, b, t2)
        z = np.abs(_weighted_residual(d, s, e, w, c)) * np.sqrt(w0)
        w = w0 * np.minimum(1.0, ROBUST_Z / np.maximum(z, _TINY))
    return _weighted_outside(d, s, e, w)


def _leave_own_out(a_full: np.ndarray, c: np.ndarray, g: np.ndarray, omega: np.ndarray,
                   u: np.ndarray) -> np.ndarray:
    """Each fluorophore's correction from the bend estimated without its own term (rank-one downdate)."""
    delta = np.empty_like(u)
    for start in range(0, len(u), CHUNK):
        sl = slice(start, min(len(u), start + CHUNK))
        h = np.linalg.solve(a_full[sl], np.broadcast_to(g, (sl.stop - sl.start,) + g.shape))
        q = np.einsum('lf,nlf->nf', g, h)
        om = omega[sl]
        delta[sl] = (c[sl] @ g - om * u[sl] * q) / np.maximum(1.0 - om * q, 1e-12)
    return delta


def bend_fluorophores(d: np.ndarray, u: np.ndarray, k: np.ndarray, mu: np.ndarray, precomputed: dict,
                      noise: tuple, bend: AFBend) -> np.ndarray:
    """Bent fluorophores of events with leftover d, per-cell AF fluorophores u, AF k and fitted signal mu."""
    from honeychrome.controller_components.autospectral_functions import _spread_variance

    e = bend.shapes
    p = np.asarray(precomputed['P'], dtype=np.float64)
    s = np.asarray(precomputed['S_t'], dtype=np.float64).T
    g = e @ p.T
    t2 = prior_variance(k, bend)
    var_d = noise_variance(mu, noise)
    a, b = _robust_outside(d, s, e, 1.0 / np.maximum(var_d, _TINY), t2)
    var_f = var_d @ (p ** 2).T
    env = precomputed.get('envelope')
    if env is not None:
        spread = _spread_variance(u, env)
        if spread is not None:
            var_f = var_f + spread
    var_f = np.maximum(var_f, 1e-12)
    sd = np.sqrt(var_f)
    omega = erfc(np.maximum(u - POSITIVE_OFFSET_SD * sd, 0.0) / (np.sqrt(2.0) * sd)) / var_f
    a_full = a + np.einsum('lf,nf,mf->nlm', g, omega, g)
    a_full = a_full + _precision(t2)[:, :, np.newaxis] * np.eye(len(e))[np.newaxis]
    rhs = b + (omega * u) @ g.T
    c = np.linalg.solve(a_full, rhs[:, :, np.newaxis])[:, :, 0]
    return u - _leave_own_out(a_full, c, g, omega, u)


def apply_bend(raw_fl: np.ndarray, unmixed: np.ndarray, k: np.ndarray, j: np.ndarray, precomputed: dict,
               af_spectra: np.ndarray, noise: tuple, bend: AFBend) -> np.ndarray:
    """
    Per-cell AF fluorophores ``unmixed`` (AF k on variant j, 0-based) with the
    events at or above the bend's threshold bent; the rest returned unchanged.
    """
    if bend is None or not bend.active or bend.shapes.shape[1] != raw_fl.shape[1]:
        return unmixed
    sel = k >= bend.from_k
    if not sel.any():
        return unmixed
    lib = np.asarray(af_spectra, dtype=np.float64)
    a = k[sel, np.newaxis] * lib[j[sel]]
    u = unmixed[sel]
    mu = a + np.maximum(u, 0.0) @ np.asarray(precomputed['S_t'], dtype=np.float64).T
    out = np.array(unmixed)
    out[sel] = bend_fluorophores(raw_fl[sel] - a, u, k[sel], mu, precomputed, noise, bend)
    return out
