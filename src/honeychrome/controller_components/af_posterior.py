"""
af_posterior.py
---------------
Noise-aware per-cell AF abundance. No Qt.

In plain terms
    A cell's AF abundance is read from the part of its spectrum that no dye
    in the panel can explain. On a cell that is bright for a dye, that part
    also carries the dye's photon noise, so the reading is mostly noise; the
    legacy method then clamps negative readings at 0, which piles cells up
    at exactly 0 (their AF left in the dye channels) and leaves a tail of
    cells with too much AF taken out. The noise-aware abundance weighs each
    cell's own reading against what cells of the same size (SSC) in the same
    sample have, by how informative the reading is: a dim cell with a clean
    AF reading keeps its own value, a bright cell whose reading is mostly
    dye noise is pulled towards the typical AF of cells of its size. A
    second, broad component lets cells with clearly unusual AF (dead cells,
    high-AF myeloid cells) keep it. The AF spectrum chosen for each cell is
    unchanged.

The maths
    For an event with chosen AF variant j (``_assign_af_chunk``):

    * k_hat = r_j'y / D_j, not clamped (r_j: the variant's out-of-span
      residual, D_j: r_j'r_j floored as in ``_joint_l2_library_terms``).
    * Its sampling variance s2 = sum_d r_jd^2 V_d(mu) / D_j^2, with V_d the
      experiment's noise model (cells or beads) at mu = max(u, 0) S + k_typ
      a_j, u the AF-free unmix and k_typ the sample's typical AF. Without a
      signal-squared term (c = 0) this is a_j + max(u, 0) . W[:, j] with
      per-variant constants a and W, one fluorophore-length product per
      event.
    * s2 is multiplied by the sample's choice shrinkage c: the robust
      variance of k_hat on the chosen variants over that on the library's
      mean spectrum, within [0.05, 1]. Choosing among many variants fits
      part of each event's noise, so its AF reading varies less than a fixed
      spectrum's would; c says how much.
    * Prior, fitted on the sample's own events: a core Normal(m, tau2) with
      m = intercept + slope log SSC (median regression on residuals
      standardised by sqrt(tau2 + c s2)) and tau2 set so the median
      standardised squared residual equals that of a normal (0.4549); a
      broad Normal(m + delta, tau2_broad) of weight pi_broad by EM. Both
      truncated at 0.
    * Reported AF: the posterior mean. Per component, w = v / (v + c s2),
      mu_p = m + w (k_hat - m), sd_p = sqrt(w c s2), mean of the part above
      0 = mu_p + sd_p phi(a) / (1 - Phi(a)), a = -mu_p / sd_p; components
      weighted by prior weight x N(k_hat; m, v + c s2) x Phi(mu_p / sd_p).
      Always > 0, smooth through low values, close to k_hat where the
      event's spectrum determines its AF.
    * Fluorophores = u - k_post v_j, as for any AF value.

Priors are fitted once per sample on a seeded subsample of its file
(FIT_EVENTS events, unsaturated only), so they never depend on which events
a call unmixes, and stored in ``<experiment>/AutoSpectral/af_priors.json``
keyed by sample and by a fingerprint of what decides them (the file, the AF
library and its envelope, the panel spectra, the noise model of the
sample's particle class). Workers fit and store them; the file is written
under a lock and replaced atomically, and a lost write only means a refit
(the fit is deterministic).

AF shape bend: with ``process['af_shape_bend']`` on (the default), the
posterior also carries the bend of the sample's AF library (af_shape_bend),
learnt on the unstained sample the library's first profile was extracted
from and stored in ``<experiment>/AutoSpectral/af_bends.json`` keyed by that
sample and a fingerprint of its prior.

Setting (``process['af_abundance']``): ``'noise_aware'`` or ``'clamped'``;
missing means clamped (experiments made before the setting existed). New
experiments start noise-aware (``settings.process_default``). Noise-aware
needs a stored noise model; without one the abundance stays clamped.

v1 simplifications
    * The prior mean depends on log SSC only.
    * A sample unmixed with corrected spectra (Spectral Correction) uses the
      prior fitted with the base spectra.
    * The sample being acquired, and a sample with fewer than MIN_EVENTS
      usable events, use the prior of the unstained sample its first AF
      profile was extracted from, fitted under the sample's own AF library;
      without one, the abundance is clamped. Once recorded, a sample's own
      prior is fitted the next time it is unmixed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.special import erfcx, log_ndtr

from honeychrome.controller_components import af_shape_bend as asb

logger = logging.getLogger(__name__)

AF_ABUNDANCE_KEY = 'af_abundance'
NOISE_AWARE = 'noise_aware'
CLAMPED = 'clamped'

PRIOR_VERSION = 1
PRIORS_FILE = 'af_priors.json'
FIT_EVENTS = 20_000
FIT_SEED = 7
MIN_EVENTS = 2_000
MAX_ENTRIES_PER_SAMPLE = 4
CHUNK = 50_000

TAU2_FLOOR = 0.01            # x the median noise variance
CHI2_1_MEDIAN = 0.45493642311957283
IRLS_EPS = 1e-3
CORE_ITERATIONS = 30
MIXTURE_ITERATIONS = 60
SHRINKAGE_RANGE = (0.05, 1.0)
_TINY = 1e-300


# ---------------------------------------------------------------------------
# Setting
# ---------------------------------------------------------------------------

def af_abundance_mode(process: dict | None) -> str:
    """The experiment's AF abundance setting: NOISE_AWARE or CLAMPED (the default when unset)."""
    value = (process or {}).get(AF_ABUNDANCE_KEY)
    return NOISE_AWARE if value == NOISE_AWARE else CLAMPED


def set_af_abundance_mode(process: dict, mode: str) -> None:
    process[AF_ABUNDANCE_KEY] = NOISE_AWARE if mode == NOISE_AWARE else CLAMPED


# ---------------------------------------------------------------------------
# Prior and posterior
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AFPrior:
    """A sample's fitted AF prior (see the module docstring)."""
    intercept: float
    slope: float
    tau2: float
    delta: float
    tau2_broad: float
    pi_broad: float
    choice_shrinkage: float
    k_typical: float
    log_ssc_median: float
    n_events: int
    fingerprint: str
    source: str

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> 'AFPrior':
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


@dataclass(frozen=True, eq=False)
class AFPosterior:
    """
    Everything the noise-aware abundance needs for one sample: its prior and
    the per-detector noise model of its particle class (sigma0_sq, g, c),
    plus the name of the SSC channel the prior mean depends on and, when the
    AF shape bend is on, the bend of the sample's AF library
    (af_shape_bend.AFBend).
    """
    prior: AFPrior
    sigma0_sq: np.ndarray
    g: np.ndarray
    c: np.ndarray
    ssc_name: str | None = None
    bend: asb.AFBend | None = None

    @property
    def fingerprint(self) -> str:
        return self.prior.fingerprint

    @property
    def noise(self) -> tuple:
        return self.sigma0_sq, self.g, self.c

    def subset(self, keep: np.ndarray) -> 'AFPosterior':
        """
        The same posterior on a subset of detectors (events with saturated
        detectors), without the bend, which is learnt on every detector.
        """
        return replace(self, sigma0_sq=self.sigma0_sq[keep], g=self.g[keep], c=self.c[keep], bend=None)

    def log_ssc(self, raw_event_data: np.ndarray, pnn_raw) -> np.ndarray | None:
        """log SSC of each event (SSC below 1 read as 1); None when the channel is not in ``pnn_raw``."""
        pnn = list(pnn_raw or [])
        if self.ssc_name is None or self.ssc_name not in pnn:
            return None
        col = pnn.index(self.ssc_name)
        if col >= raw_event_data.shape[1]:
            return None
        return np.log(np.maximum(np.asarray(raw_event_data[:, col], dtype=np.float64), 1.0))

    def abundance(self, k_hat: np.ndarray, unmixed: np.ndarray, rows: np.ndarray, precomputed: dict,
                  af_spectra: np.ndarray, k_denom: np.ndarray, log_ssc: np.ndarray | None) -> np.ndarray:
        """
        Posterior mean AF of events with unclamped estimates ``k_hat`` on
        variants ``rows``, AF-free unmix ``unmixed``. Without ``log_ssc``
        every event takes the prior mean at the sample's median log SSC.
        """
        p = self.prior
        s2 = af_variance(np.maximum(unmixed, 0.0), rows, precomputed, af_spectra, k_denom,
                         self.sigma0_sq, self.g, self.c, p.k_typical)
        s2 *= p.choice_shrinkage
        if log_ssc is None:
            log_ssc = np.full(len(k_hat), p.log_ssc_median)
        return posterior_mixture(k_hat, s2, log_ssc, p)


def af_variance(u_pos: np.ndarray, rows: np.ndarray, precomputed: dict, af_spectra: np.ndarray,
                k_denom: np.ndarray, sigma0_sq: np.ndarray, g: np.ndarray, c: np.ndarray,
                k_typical: float) -> np.ndarray:
    """
    Sampling variance of each event's AF estimate on variant ``rows[i]``,
    sum_d r_d^2 V_d(mu) / D^2, with mu the event's AF-free fitted signal
    (``u_pos`` S) plus ``k_typical`` of the variant. With c = 0, V_d is linear
    in mu and the sum splits into a per-variant constant
    a_j = sum_d r_dj^2 (sigma0_d^2 + g_d k_typical a_jd) plus u_pos . W[:, j],
    W = S (g r^2) (fluorophores x variants); otherwise the detector
    variances are formed in full.
    """
    r2 = precomputed['r_library'] ** 2                      # detectors x variants
    s_t = precomputed['S_t']                                # detectors x fluorophores
    k_typ = max(float(k_typical), 0.0)
    rows = np.asarray(rows, dtype=np.intp)
    if not np.any(c):
        a = r2.T @ sigma0_sq + k_typ * np.sum(r2 * (g[:, np.newaxis] * af_spectra.T), axis=0)
        w = s_t.T @ (g[:, np.newaxis] * r2)                 # fluorophores x variants
        total = a[rows] + np.einsum('if,if->i', u_pos, w.T[rows])
    else:
        signal = np.maximum(u_pos @ s_t.T + k_typ * af_spectra[rows], 0.0)
        var = sigma0_sq + signal * (g + c * signal)
        total = np.einsum('ij,ij->i', var, r2.T[rows])
    return np.maximum(total, _TINY) / k_denom[rows] ** 2


def posterior_components(k_hat: np.ndarray, s2: np.ndarray, means: list, variances: list,
                         weights: list) -> np.ndarray:
    """
    Mean of the posterior AF for a prior that is a mixture of normals
    truncated at 0 and k_hat ~ N(k, s2). Each component gives the posterior
    of a single normal prior; components are combined by prior weight x the
    likelihood of k_hat under them x their posterior mass above 0. Computed
    on the log scale. The mean of a normal truncated at 0 uses the inverse
    Mills ratio phi(a) / (1 - Phi(a)) = sqrt(2 / pi) / erfcx(a / sqrt(2)),
    which stays accurate when the posterior lies far below 0 (large a),
    where the ratio of phi and the tail probability would overflow.
    """
    log_mass, cond_mean = [], []
    for m, v, wt in zip(means, variances, weights):
        tot = v + s2
        w = v / tot
        mu_p = m + w * (k_hat - m)
        sd_p = np.sqrt(np.maximum(w * s2, _TINY))
        a = -mu_p / sd_p
        above = log_ndtr(-a)
        log_mass.append(np.log(max(wt, _TINY)) - 0.5 * np.log(2.0 * np.pi * tot)
                        - 0.5 * (k_hat - m) ** 2 / tot + above)
        cond_mean.append(mu_p + sd_p * (np.sqrt(2.0 / np.pi) / erfcx(np.maximum(a / np.sqrt(2.0), -26.0))))
    log_mass = np.array(log_mass)
    top = log_mass.max(axis=0)
    resp = np.exp(log_mass - np.where(np.isfinite(top), top, 0.0))
    total = resp.sum(axis=0)
    resp = np.where(total > 0.0, resp / np.where(total > 0.0, total, 1.0), 1.0 / len(log_mass))
    return np.sum(resp * np.array(cond_mean), axis=0)


def posterior_mixture(k_hat: np.ndarray, s2: np.ndarray, log_ssc: np.ndarray, prior: AFPrior) -> np.ndarray:
    """Posterior mean AF under a sample's two-component prior."""
    m = prior.intercept + prior.slope * log_ssc
    return posterior_components(k_hat, s2, [m, m + prior.delta], [prior.tau2, prior.tau2_broad],
                                [1.0 - prior.pi_broad, prior.pi_broad])


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _robust_sd(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _mean_row_choice(numerator: np.ndarray, precomputed: dict, mean_rows: np.ndarray) -> np.ndarray:
    """Per event, the library mean row (of combined profiles) whose fit removes the most residual."""
    if mean_rows.size == 1:
        return np.full(len(numerator), int(mean_rows[0]), dtype=np.intp)
    gain = numerator[:, mean_rows] ** 2 / precomputed['r_dots'][mean_rows]
    return mean_rows[np.argmax(gain, axis=1)]


def af_estimates(raw_fl: np.ndarray, precomputed: dict, af_spectra: np.ndarray) -> dict:
    """
    Every event's chosen variant (as ``apply_af_unmixing``), its unclamped
    AF estimate on it (``k_hat``) and on the library's mean spectrum
    (``k_mean``), the AF-free unmix ``u`` and ``k_denom``.
    """
    from honeychrome.controller_components import autospectral_functions as asf

    w, k_denom, c_fluor = asf._joint_l2_library_terms(precomputed, af_spectra)
    envelope = precomputed.get('envelope')
    mean_rows = asf._envelope_mean_rows(envelope) if envelope is not None else np.array([0], dtype=np.intp)
    n = len(raw_fl)
    best = np.empty(n, dtype=np.intp)
    k_hat = np.empty(n)
    k_mean = np.empty(n)
    u = np.empty((n, precomputed['P'].shape[0]))
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        chunk = np.ascontiguousarray(raw_fl[start:end], dtype=np.float64)
        best_j, numerator, unmixed = asf._assign_af_chunk(chunk, precomputed, w, k_denom, c_fluor, envelope)
        rows = np.arange(end - start)
        m = _mean_row_choice(numerator, precomputed, mean_rows)
        best[start:end] = best_j
        k_hat[start:end] = numerator[rows, best_j] / k_denom[best_j]
        k_mean[start:end] = numerator[rows, m] / k_denom[m]
        u[start:end] = unmixed
    return {'best': best, 'k_hat': k_hat, 'k_mean': k_mean, 'u': u, 'k_denom': k_denom}


def _tau2_from_median(resid: np.ndarray, s2: np.ndarray, floor: float) -> float:
    """
    tau2 at which the median of resid^2 / (tau2 + s2) equals the median of a
    chi-square with one degree of freedom: the cell-to-cell AF variance once
    each event's own noise is allowed for, set by the middle of the
    distribution so tails do not inflate it. Bisection on a log scale.
    """
    r2 = resid * resid
    if np.median(r2 / (floor + s2)) <= CHI2_1_MEDIAN:
        return floor
    lo, hi = np.log(floor), np.log(max(float(np.median(r2)) / CHI2_1_MEDIAN, floor * 2.0))
    while np.median(r2 / (np.exp(hi) + s2)) > CHI2_1_MEDIAN:
        hi += np.log(4.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.median(r2 / (np.exp(mid) + s2)) > CHI2_1_MEDIAN:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


def fit_core(k_hat: np.ndarray, s2: np.ndarray, log_ssc: np.ndarray) -> dict:
    """
    Prior mean linear in log SSC and the cell-to-cell AF variance tau2, both
    resistant to tails: the mean by a median regression of the standardised
    residuals (iteratively reweighted least squares), tau2 from
    _tau2_from_median, updated in turn. Every event is used.
    """
    design = np.column_stack([np.ones_like(log_ssc), log_ssc])
    coef = np.array([float(np.median(k_hat)), 0.0])
    floor = TAU2_FLOOR * float(np.median(s2))
    tau2 = floor
    for _ in range(CORE_ITERATIONS):
        resid = k_hat - design @ coef
        tau2 = _tau2_from_median(resid, s2, floor)
        scale = np.sqrt(tau2 + s2)
        z = np.abs(resid) / scale
        w = 1.0 / (scale * scale * np.maximum(z, IRLS_EPS))
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(design * sw[:, np.newaxis], k_hat * sw, rcond=None)
    return {'intercept': float(coef[0]), 'slope': float(coef[1]), 'tau2': float(tau2)}


def _log_normal(x: np.ndarray, var: np.ndarray) -> np.ndarray:
    return -0.5 * np.log(2.0 * np.pi * var) - 0.5 * x * x / var


def fit_mixture(k_hat: np.ndarray, s2: np.ndarray, log_ssc: np.ndarray, core: dict) -> dict:
    """
    The broad component beside the core (kept as fitted): mean m + delta,
    variance tau2_broad (at least 4 x tau2), weight pi_broad, by EM on every
    event allowing for its own noise s2.
    """
    resid = k_hat - (core['intercept'] + core['slope'] * log_ssc)
    v1 = core['tau2']
    floor = 4.0 * v1
    pi_b, delta = 0.05, 0.0
    v2 = max(4.0 * _robust_sd(resid) ** 2, floor)
    for _ in range(MIXTURE_ITERATIONS):
        l1 = np.log(1.0 - pi_b) + _log_normal(resid, v1 + s2)
        l2 = np.log(pi_b) + _log_normal(resid - delta, v2 + s2)
        r_b = 1.0 / (1.0 + np.exp(np.clip(l1 - l2, -700.0, 700.0)))
        w = 1.0 / (v2 + s2)
        delta = float(np.sum(r_b * w * resid) / max(np.sum(r_b * w), _TINY))
        e = resid - delta
        rw2 = r_b * w * w
        v2 = max(float(np.sum(rw2 * (e * e - s2)) / max(np.sum(rw2), _TINY)), floor)
        pi_b = float(np.clip(np.mean(r_b), 1e-4, 0.5))
    return {'delta': delta, 'tau2_broad': v2, 'pi_broad': pi_b}


def fit_af_prior(raw_fl: np.ndarray, log_ssc: np.ndarray, precomputed: dict, af_spectra: np.ndarray,
                 sigma0_sq: np.ndarray, g: np.ndarray, c: np.ndarray, fingerprint: str = '',
                 source: str = '') -> AFPrior:
    """A sample's AF prior from its (unsaturated) raw fluorescence events and log SSC."""
    est = af_estimates(raw_fl, precomputed, af_spectra)
    k_hat = est['k_hat']
    k_typical = max(float(np.median(k_hat)), 0.0)
    s2 = af_variance(np.maximum(est['u'], 0.0), est['best'], precomputed, af_spectra, est['k_denom'],
                     sigma0_sq, g, c, k_typical)
    shrink = _robust_sd(k_hat) ** 2 / max(_robust_sd(est['k_mean']) ** 2, 1e-12)
    shrink = float(np.clip(shrink, *SHRINKAGE_RANGE))
    s2c = shrink * s2
    core = fit_core(k_hat, s2c, log_ssc)
    mix = fit_mixture(k_hat, s2c, log_ssc, core)
    return AFPrior(intercept=core['intercept'], slope=core['slope'], tau2=core['tau2'], delta=mix['delta'],
                   tau2_broad=mix['tau2_broad'], pi_broad=mix['pi_broad'], choice_shrinkage=shrink,
                   k_typical=k_typical, log_ssc_median=float(np.median(log_ssc)), n_events=int(len(k_hat)),
                   fingerprint=fingerprint, source=source)


# ---------------------------------------------------------------------------
# Stored priors
# ---------------------------------------------------------------------------

_store_lock = threading.Lock()
_fit_locks: dict = {}
_memory: dict = {}          # {priors file: {sample key: {fingerprint: entry}}}
_unfittable: set = set()    # (experiment, sample key, fingerprint) with too few usable events


def _priors_path(experiment_dir) -> Path:
    from honeychrome.controller_components.autospectral_functions import AF_SUBDIR
    return Path(experiment_dir) / AF_SUBDIR / PRIORS_FILE


def _bends_path(experiment_dir) -> Path:
    from honeychrome.controller_components.autospectral_functions import AF_SUBDIR
    return Path(experiment_dir) / AF_SUBDIR / asb.BENDS_FILE


def _read_store(path: Path, version: int = PRIOR_VERSION) -> dict:
    if str(path) in _memory:
        return _memory[str(path)]
    data = {}
    try:
        with open(path) as f:
            raw = json.load(f)
        if raw.get('version') == version:
            data = raw.get('samples') or {}
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as exc:
        logger.warning('AF priors: %s not read (%s); its entries will be fitted again.', path.name, exc)
    _memory[str(path)] = data
    return data


def _load_entry(path: Path, version: int, sample_key: str, fingerprint: str) -> dict | None:
    with _store_lock:
        return (_read_store(path, version).get(str(sample_key)) or {}).get(fingerprint)


def load_prior(experiment_dir, sample_key: str, fingerprint: str) -> AFPrior | None:
    """The stored prior of ``sample_key`` with this fingerprint, or None."""
    entry = _load_entry(_priors_path(experiment_dir), PRIOR_VERSION, sample_key, fingerprint)
    if entry is None:
        return None
    try:
        return AFPrior.from_dict(entry)
    except (KeyError, TypeError):
        return None


def load_bend(experiment_dir, sample_key: str, fingerprint: str) -> asb.AFBend | None:
    """The stored AF shape bend learnt on ``sample_key`` with this fingerprint, or None."""
    entry = _load_entry(_bends_path(experiment_dir), asb.BEND_VERSION, sample_key, fingerprint)
    if entry is None:
        return None
    try:
        return asb.AFBend.from_dict(entry)
    except (KeyError, TypeError, ValueError):
        return None


def store_prior(experiment_dir, sample_key: str, prior: AFPrior) -> None:
    """Store a fitted prior, keeping the newest MAX_ENTRIES_PER_SAMPLE fingerprints of each sample."""
    _store_entry(_priors_path(experiment_dir), PRIOR_VERSION, sample_key, prior.fingerprint, prior.to_dict())


def store_bend(experiment_dir, sample_key: str, bend: asb.AFBend) -> None:
    """Store a learnt AF shape bend, as ``store_prior``."""
    _store_entry(_bends_path(experiment_dir), asb.BEND_VERSION, sample_key, bend.fingerprint, bend.to_dict())


def _store_entry(path: Path, version: int, sample_key: str, fingerprint: str, entry: dict) -> None:
    """Write one entry under a lock, merged with what is on disk, keeping the newest entries of each sample."""
    with _store_lock:
        data = _read_store(path, version)
        try:
            with open(path) as f:
                on_disk = json.load(f)
            if on_disk.get('version') == version:
                for key, entries in (on_disk.get('samples') or {}).items():
                    merged = dict(entries)
                    merged.update(data.get(key) or {})
                    data[key] = merged
        except (OSError, ValueError):
            pass
        entries = dict(data.get(str(sample_key)) or {})
        entries.pop(fingerprint, None)
        entries[fingerprint] = entry
        while len(entries) > MAX_ENTRIES_PER_SAMPLE:
            entries.pop(next(iter(entries)))
        data[str(sample_key)] = entries
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f'{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp')
            with open(tmp, 'w') as f:
                json.dump({'version': version, 'samples': data}, f)
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning('AF priors: could not write %s (%s)', path.name, exc)


def _fit_lock(key) -> threading.Lock:
    with _store_lock:
        lock = _fit_locks.get(key)
        if lock is None:
            lock = _fit_locks[key] = threading.Lock()
        return lock


# ---------------------------------------------------------------------------
# Experiment context
# ---------------------------------------------------------------------------

def ssc_channel(raw_settings: dict, pnn_raw) -> str | None:
    """The SSC area channel: the experiment's scatter parameter when it is in ``pnn_raw``, else the first SSC*-A."""
    pnn = list(pnn_raw or [])
    scatter = (raw_settings or {}).get('scatter_param') or []
    if len(scatter) > 1 and scatter[1] in pnn and re.search(r'SSC', str(scatter[1])):
        return scatter[1]
    return next((name for name in pnn if re.search(r'SSC.*-A$', name)), None)


def _digest(*parts) -> str:
    from honeychrome.controller_components.unmix_method import digest
    return digest(*parts)


@dataclass(frozen=True, eq=False)
class AFNoiseContext:
    """
    What resolving a sample's noise-aware AF needs, read on the main thread
    (``noise_context``) and safe to hand to workers: the noise model per
    particle class, the bead samples, where each AF profile came from, and
    how to read a sample's raw events.
    """
    experiment_dir: str
    cell_noise: tuple                 # (sigma0_sq, g, c)
    bead_noise: tuple
    noise_fingerprint: str
    bead_samples: frozenset
    references: dict                  # {profile name: sample key it was extracted from}
    pnn_raw: tuple
    whitelisted: bool
    fl_ids_raw: tuple
    ssc_name: str | None
    shape_bend: bool = False

    def _noise_for(self, sample_key) -> tuple:
        return self.bead_noise if str(sample_key) in self.bead_samples else self.cell_noise

    def reference_for(self, af) -> str | None:
        for name in getattr(af, 'profile_names', ()) or ():
            ref = self.references.get(name)
            if ref:
                return str(ref)
        return None

    def _fingerprint(self, sample_key: str, af, noise: tuple) -> str | None:
        try:
            st = (Path(self.experiment_dir) / sample_key).stat()
        except OSError:
            return None
        env = af.precomputed.get('envelope') if af.precomputed else None
        env_part = None if env is None else [env.get('null_var'), env.get('spread'), env.get('mean_rows'),
                                             env.get('ceiling'), env.get('saturated_frac')]
        return _digest(PRIOR_VERSION, FIT_EVENTS, FIT_SEED, str(sample_key), [st.st_size, st.st_mtime_ns],
                       list(af.profile_names), np.asarray(af.spectra, dtype=np.float64),
                       np.asarray(af.precomputed['S_t'], dtype=np.float64), env_part, list(noise),
                       self.noise_fingerprint, self.ssc_name)

    def _read_unsaturated(self, sample_key: str, af, n_events: int, seed: int) -> tuple | None:
        """A seeded subsample's unsaturated raw fluorescence and log SSC; None with fewer than MIN_EVENTS."""
        from honeychrome.controller_components.sample_loader import read_raw_subsample

        try:
            raw, _index, _n_file, _n_kept = read_raw_subsample(Path(self.experiment_dir) / sample_key, self,
                                                               sample_key, n_events, seed=seed)
        except Exception as exc:
            logger.warning('AF priors: %s not read (%s)', sample_key, exc)
            return None
        if raw is None or len(raw) == 0 or max(self.fl_ids_raw, default=-1) >= raw.shape[1]:
            return None
        raw_fl = np.ascontiguousarray(raw[:, list(self.fl_ids_raw)], dtype=np.float64)
        env = af.precomputed.get('envelope')
        keep = np.ones(len(raw_fl), dtype=bool)
        if env is not None:
            keep = ~(raw_fl >= env['saturated_frac'] * env['ceiling']).any(axis=1)
        if keep.sum() < MIN_EVENTS:
            return None
        pnn = list(self.pnn_raw)
        if self.ssc_name in pnn:
            log_ssc = np.log(np.maximum(raw[keep, pnn.index(self.ssc_name)], 1.0))
        else:
            log_ssc = np.zeros(int(keep.sum()))
        return raw_fl[keep], log_ssc

    def _fit(self, sample_key: str, af, noise: tuple, fingerprint: str) -> AFPrior | None:
        events = self._read_unsaturated(sample_key, af, FIT_EVENTS, FIT_SEED)
        if events is None:
            return None
        raw_fl, log_ssc = events
        prior = fit_af_prior(raw_fl, log_ssc, af.precomputed, np.asarray(af.spectra, dtype=np.float64),
                             *noise, fingerprint=fingerprint, source=str(sample_key))
        logger.info('AF priors: fitted %s (%d events; typical AF %.1f, choice shrinkage %.2f)',
                    sample_key, prior.n_events, prior.k_typical, prior.choice_shrinkage)
        return prior

    def _prior_for(self, sample_key: str, af, noise: tuple) -> AFPrior | None:
        fingerprint = self._fingerprint(sample_key, af, noise)
        if fingerprint is None:
            return None
        prior = load_prior(self.experiment_dir, sample_key, fingerprint)
        if prior is not None:
            return prior
        key = (self.experiment_dir, sample_key, fingerprint)
        if key in _unfittable:
            return None
        with _fit_lock(key):
            prior = load_prior(self.experiment_dir, sample_key, fingerprint)
            if prior is None and key not in _unfittable:
                prior = self._fit(sample_key, af, noise, fingerprint)
                if prior is not None:
                    store_prior(self.experiment_dir, sample_key, prior)
                else:
                    _unfittable.add(key)
        return prior

    def _bend_for(self, sample_key: str, af) -> asb.AFBend | None:
        """
        The AF shape bend of library ``af`` learnt on ``sample_key`` (its
        reference unstained sample), stored or learnt now; None when the
        sample has no prior or too few usable events.
        """
        noise = self._noise_for(sample_key)
        prior = self._prior_for(sample_key, af, noise)
        if prior is None:
            return None
        fingerprint = _digest(asb.BEND_VERSION, asb.LEARN_EVENTS, asb.LEARN_SEED, prior.fingerprint)
        bend = load_bend(self.experiment_dir, sample_key, fingerprint)
        if bend is not None:
            return bend
        key = ('bend', self.experiment_dir, sample_key, fingerprint)
        if key in _unfittable:
            return None
        with _fit_lock(key):
            bend = load_bend(self.experiment_dir, sample_key, fingerprint)
            if bend is None and key not in _unfittable:
                events = self._read_unsaturated(sample_key, af, asb.LEARN_EVENTS, asb.LEARN_SEED)
                if events is not None:
                    posterior = AFPosterior(prior=prior, sigma0_sq=noise[0], g=noise[1], c=noise[2],
                                            ssc_name=self.ssc_name)
                    bend = asb.learn_bend(events[0], events[1], af.precomputed,
                                          np.asarray(af.spectra, dtype=np.float64), posterior,
                                          fingerprint=fingerprint, source=str(sample_key))
                    logger.info('AF shape bend: learnt on %s (%d events; %d shapes; from AF %.1f)',
                                sample_key, bend.n_events, len(bend.shapes), bend.from_k)
                    store_bend(self.experiment_dir, sample_key, bend)
                else:
                    _unfittable.add(key)
        return bend

    def posterior_for(self, af, sample_key, live: bool = False) -> AFPosterior | None:
        """
        The noise-aware posterior for ``sample_key`` unmixed with the AF
        library ``af``: its own prior, or while it is being acquired (``live``)
        or when it has too few usable events, its AF reference's prior under
        the same library. None (clamped) when neither can be had. With the
        AF shape bend on, the posterior carries the library's bend learnt on
        that reference sample (none without a reference).
        """
        if af is None or not af.precomputed or sample_key is None:
            return None
        key = str(sample_key)
        reference = self.reference_for(af)
        candidates = [] if live else [key]
        if reference is not None and reference not in candidates:
            candidates.append(reference)
        for candidate in candidates:
            noise = self._noise_for(candidate)
            if len(noise[0]) != np.asarray(af.spectra).shape[1]:
                logger.warning('AF priors: the noise model and the AF library have different detectors; '
                               'the AF abundance is clamped.')
                return None
            prior = self._prior_for(candidate, af, noise)
            if prior is not None:
                if candidate != key:
                    logger.info('AF priors: %s uses the prior of %s', key, candidate)
                bend = None
                if self.shape_bend and reference is not None:
                    bend = self._bend_for(reference, af)
                return AFPosterior(prior=prior, sigma0_sq=noise[0], g=noise[1], c=noise[2],
                                   ssc_name=self.ssc_name, bend=bend)
        return None

    def attach(self, af, sample_key, live: bool = False):
        """``af`` with the sample's posterior attached (``af`` unchanged when it has one or none can be had)."""
        if af is None or getattr(af, 'posterior', None) is not None:
            return af
        posterior = self.posterior_for(af, sample_key, live)
        return af if posterior is None else replace(af, posterior=posterior)


def _noise_arrays(model) -> tuple:
    return (np.asarray(model.sigma0_sq, dtype=np.float64), np.asarray(model.g, dtype=np.float64),
            np.asarray(model.c, dtype=np.float64))


def noise_context(controller, pnn_raw=None, whitelisted=None, fl_ids_raw=None) -> AFNoiseContext | None:
    """
    The experiment's noise-aware AF context, or None when the setting is
    clamped or no noise model is stored for the current detectors. Main
    thread. ``pnn_raw``, ``whitelisted`` and ``fl_ids_raw`` describe how raw
    events are read (default: the channel whitelist when set, as
    ``sample_loader.snapshot_unmix_state``).
    """
    experiment = getattr(controller, 'experiment', None)
    process = getattr(experiment, 'process', None)
    if af_abundance_mode(process) != NOISE_AWARE or getattr(controller, 'experiment_dir', None) is None:
        return None
    from honeychrome.controller_components import unmix_method
    from honeychrome.controller_components.particle_type import is_bead_sample

    noise = unmix_method._noise_state(controller)
    if noise is None:
        return None
    raw = experiment.settings['raw']
    if pnn_raw is None:
        whitelist = raw.get('whitelisted_pnn') or None
        pnn_raw = list(whitelist or raw['event_channels_pnn'])
        whitelisted = whitelist is not None
        full = raw['event_channels_pnn']
        ids = getattr(controller, 'filtered_raw_fluorescence_channel_ids', None)
        if ids is None:
            ids = raw.get('fluorescence_channel_ids') or []
        fl_ids_raw = [pnn_raw.index(full[i]) for i in ids if full[i] in pnn_raw]
    samples = experiment.samples
    beads = frozenset(str(k) for k in (samples.get('all_samples') or {}) if is_bead_sample(samples, k))
    references = {str(n): str(e.get('source_fcs')) for n, e in (process.get('af_profiles') or {}).items()
                  if e.get('source_fcs')}
    return AFNoiseContext(
        experiment_dir=str(controller.experiment_dir), cell_noise=_noise_arrays(noise.model_for(False)),
        bead_noise=_noise_arrays(noise.model_for(True)),
        noise_fingerprint=str(noise.entry.get('fingerprint')), bead_samples=beads, references=references,
        pnn_raw=tuple(pnn_raw), whitelisted=bool(whitelisted), fl_ids_raw=tuple(fl_ids_raw),
        ssc_name=ssc_channel(raw, pnn_raw), shape_bend=asb.shape_bend_on(process))


def signature(context: AFNoiseContext | None) -> str | None:
    """What the noise-aware abundance adds to an unmixing signature; None when it is not in effect."""
    if context is None:
        return None
    return _digest(PRIOR_VERSION, context.noise_fingerprint, context.cell_noise, context.bead_noise,
                   context.ssc_name, asb.BEND_VERSION if context.shape_bend else None)
