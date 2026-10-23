"""
fix_my_unmix.py
---------------
Fix My Unmix: corrects the part of a reference-spectrum error that lies
inside the row space of the spectra, from fully stained samples, using the
fact that a marker-negative population must read zero in every channel it is
negative for. Developed from AutoSpectral's ``fix.my.unmix()``. Numerics
only, no Qt; the experiment-level run (sample groups, unstained match,
storage) is ``spectral_correction.py``.

It complements row-shape corrections from single-stain controls, which
cannot see in-span error.

Background
    Every event is unmixed by the main app's per-cell AF
    (``autospectral_functions.apply_af_unmixing``: the positivity-weighted AF
    choice with the profile's negative envelope, saturated detectors
    re-fitted, no positivity prior on the fluorophores) with the AF library
    assigned to its own sample (``project_events``); samples run together
    keep their own libraries (``concatenate``). ``Background`` holds the
    fixed background row of the design (the AF mean) and every AF spectrum
    the events were unmixed with.
Phase one
    Residual spillover, pair by pair. For every source fluorophore, its
    clearly positive events are those above both the spread-scaled
    unstained threshold and the upper bound of the sample's own negative
    population in that channel (its mode plus the mirrored lower tail,
    ``negative_bound``), which excludes the tilted tail of a collinear
    negative population. For every other channel a Huber line is fitted
    through those events that are negative for the channel, anchored at the
    robust centre of the source-negative bulk (``anchored_slopes``): the
    bulk fixes the origin but not the slope. Target negativity is a band
    around the current line, after a first one-sided pass, so the source's
    own cell-to-cell variation is trimmed on both sides; negativity is
    re-decided with the current slope removed (``max_mask_passes``).

    Each coefficient earns a trust weight from a prior variance (spectral
    variants propagated through the unmixing matrix) against the binned
    envelope's or the fit's standard error, from the agreement between the
    envelope and median slopes, from its agreement with a joint lasso of the
    channel over every other fluorophore (``glasso_check``), from how many
    clearly positive source events the sample has beyond the unstained
    sample's rate (``source_events_low`` to ``source_events_high``), and
    from the share of the source's positive population it leaves (or newly
    pushes, ``hypernegative_rule``) below the channel's negative boundary.
    With ``glasso_warm`` the lasso is fitted after the pair slopes and its
    first negativity mask has them removed: without it, a source whose
    positive residual spillover lifts its positive events above the target's
    boundary is missing from the lasso's fit, the lasso returns zero or the
    wrong sign and the agreement zeroes the trust of a well-fitted slope.
    With ``hypernegative_dominant`` only events the source dominates are
    judged, so another dye's error, which makes its events read positive for
    the source, does not count against the source's coefficient.
    The optional AF gate (``af_gate``) drops a source's step when the change
    it implies for that row looks like an AF spectrum. The spillover matrix
    is updated by the trust-damped error and decayed toward zero where
    trust is weak, until the ``convergence_quantile`` of the off-diagonal
    error (``stop_rule='all'``) or of the step applied (``'applied'``) falls
    below ``convergence_threshold`` (with ``'settled'``: the error of every
    coefficient with at least ``stop_trust`` trust, for ``stop_confirm``
    iterations in a row, so slopes too weakly trusted ever to finish do not
    hold phase one open), or delta stalls (``stall_rule``: its
    3-iteration trend rises, or it sets no new minimum for
    ``stall_patience`` iterations). ``seed_streams`` gives every random
    draw its own generator keyed by (seed, purpose, fluorophore), so a
    change in one population's count moves no other draw.
Phase two
    Each changed fluorophore's signature (``phase_two_rows``) is re-measured
    from background-subtracted raw data of the events it dominates
    (``extract_raw_signature``, every other panel abundance fitted jointly)
    and adopted only if it passes every gate. Because the regressors are the
    phase-one-compensated abundances, the in-span part of a re-measured row
    is the back-solved phase-one row plus noise; ``phase_two_adopt=
    'backsolved'`` adopts phase one's own row when the candidate passes. The
    optional residual gate compares the OLS residual of the fluorophore's
    positive events before and after the change; it is blind to the in-span
    part of a change by construction. A row with too few events for the
    joint fit is re-measured with every other row held at its spectrum;
    ``phase_two_fallback`` adopts that row (default), phase one's
    back-solved row, or rejects it.
Plausibility bounds
    A detector of a row may change by at most ``plausibility_ratio`` x its
    value in the reference row (the spectra the run, or ``smooth_fix``'s
    rounds, started from) or ``plausibility_floor``, whichever is greater
    (defaults 1 and 0.01: a value can at most double, a detector with no
    signal can gain 1% of the peak). Real bead-to-cell errors fit inside
    this box; a dim marker co-expressed on a dye's cells, or AF taken up
    by it, does not, because it adds signal where the dye had none. Phase
    one takes the part of each back-solved row outside the box back out of
    its coefficients by least squares over the other rows (to the bound),
    so a contaminated coefficient stops steering phase two's regressors;
    phase two rejects a row with more than ``plausibility_reject`` of its
    change (norm) out of bounds ('implausible') and resets the detectors
    out of bounds of the row it adopts to the reference value
    (``plausibility_reset``; else to the bound). The box bounds back-solved
    rows, not single pair coefficients or whole phase-one steps: bounding
    those holds back real corrections.
Shared directions (off by default)
    ``shared_direction``: when mutually unrelated sources (cosine below
    ``shared_cosine``) carry coefficients of the same sign into one channel,
    and each source's whole row change points at that channel's dye
    (``shared_alignment``), the dye is more likely expressed on their cells
    than every one of their rows wrong in the same way: each such
    coefficient's fixed point is shrunk by (number of unrelated sources) **
    -``shared_power``. Bead-to-cell errors share directions too (systematic
    bead differences), so the check costs real corrections there.
AF penalty (off by default)
    ``af_penalty``: a source's step is damped when its row change is more
    AF-like (squared cosine to the nearest AF spectrum) than the dye itself
    by more than ``af_penalty_margin``, to nothing ``af_penalty_width``
    further on, so an AF-like dye may still move in AF-like directions.
Fluorophores kept fixed
    ``fixed_fluorophores`` names rows that are never corrected: a fixed
    fluorophore is no pair's source in phase one (no evidence is gathered
    and its off-diagonal spillover stays zero, so its back-solved row is its
    spectrum) and phase two leaves its spectrum as it is ('fixed'). It still
    counts as a target channel, as a lasso regressor and in the background.
    With ``exclude_fixed_targets`` (off by default) a fixed fluorophore is 
    also no pair's target channel, no lasso target or regressor and no 
    phase-two leakage target: it stays in the design, the background and the
    dominance of events, so its signal is still unmixed, but evidence in its
    channel never changes another row.
Smooth updates
    ``smooth_fix`` runs Fix My Unmix in rounds on fresh subsamples, each
    started from the current spectra, and moves the spectra a step
    1 / (round + 1) toward each round's result, so a subsample's noise is
    averaged rather than adopted. The Spectral Correction page uses it
    unless ``smooth_updates`` is switched off.

The pair slopes run in a compiled kernel (``bundled_plugins/
fix_my_unmix_kernel.cpp``, built by ``build_fix_my_unmix_kernel.py``) when it
is present, else in numpy, with the same results.

Conventions
    * Spectra are fluorophores x detectors, L-inf normalised; raw data are
      events x detectors in the same detector order.
    * Spillover matrices are source (rows) x target (columns); compensated
      abundance is ``abundance @ inv(spillover)``.
    * Index ``-1`` means "none" wherever R used 0 (dominant fluorophore, AF
      row of an event unmixed without AF).

v1 simplifications
    * The design's background row (conditioning, leakage, the variant
      prior) is a single AF mean row even when samples run together use
      different AF libraries; samples without AF add none.
"""

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


def _load_kernel():
    """The compiled pair-slope kernel from bundled_plugins/, or None."""
    from honeychrome.controller_components.plugin_loaders import bundled_plugins_path
    if str(bundled_plugins_path) not in sys.path:
        sys.path.insert(0, str(bundled_plugins_path))
    try:
        import _fmu_kernel
    except (ImportError, OSError):
        return None
    if not hasattr(_fmu_kernel.lib, 'fmu_anchored_batch'):     # built from an older kernel source
        return None
    return _fmu_kernel


_kernel = _load_kernel()
KERNEL_AVAILABLE = _kernel is not None

_EPS = np.finfo(float).eps
AF_NAME = 'AF'
RESIDUAL_GATES = ('off', 'no_increase', 'decrease')
HYPERNEGATIVE_RULES = ('after', 'new')
PHASE_TWO_ROWS = ('all', 'flagged')
PHASE_TWO_ADOPT = ('remeasured', 'backsolved')
PHASE_TWO_FALLBACK = ('remeasured', 'backsolved', 'reject')
STOP_RULES = ('all', 'applied', 'trusted', 'settled')
DISAGREEMENT_REFERENCES = ('envelope', 'anchored', 'envelope_corrected')
STALL_RULES = ('trend', 'patience')
# generator keys of seed_streams, one stream per purpose and fluorophore
_STREAM_EVIDENCE, _STREAM_LASSO, _STREAM_SLOPES, _STREAM_DOWNSAMPLE, _STREAM_SMOOTH = 1, 2, 3, 4, 5


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class FixSettings:
    """
    Settings of a run. Where a setting comes from ``fix.my.unmix()`` it keeps
    R's default, except ``spread_kappa`` (3: the Residual Model spread is
    narrower than the real cell-to-cell variation of stained cells) and
    ``convergence_quantile`` (1: phase one runs until the largest
    off-diagonal error is below ``convergence_threshold``). R's estimator
    options that proved less accurate (the bulk in the fit, the one-sided
    cut, the binned envelope and the joint lasso as estimators) are not
    offered.

    The defaults were chosen on bead-to-cell spectral errors with known cell
    references and checked on real panels. ``glasso_warm`` and
    ``hypernegative_dominant`` are on; the other stop and stall rules,
    ``seed_streams`` and the non-joint fallbacks other than 'remeasured' are
    off (no gain in accuracy, or a loss; 'applied' can stop after one partly
    trusted step, below the flag threshold). The plausibility box (ratio 1,
    floor 0.003, reject 0.5, reset) is on: it removes most of the
    co-expression and AF taken up on real panels at no cost on bead-to-cell
    errors. The shared-direction check and the AF penalty are off.
    ``stop_rule='settled'`` and ``disagreement_reference='envelope_corrected'``
    lose no accuracy against 'all' and 'envelope' and are a quarter faster
    with smooth updates; 'anchored' remains an option.

    Six defaults come from a factorial scan on bead-to-cell errors (Aurora,
    FACSDiscover A8, ID7000) checked on three real panels over three seeds
    and disjoint halves of the events: ``max_angle`` 30 (a large real
    correction, e.g. a tandem on the A8, needs about 20 degrees),
    ``phase_two_rows`` 'all' (rows phase one did not move are re-measured
    too), ``coefficient_decay`` 0, ``convergence_threshold`` 0.003,
    ``plausibility_floor`` 0.003 and ``unstained_threshold`` 0.95 for the
    upper boundary (the lower one keeps the 0.99 quantile,
    ``lower_unstained_threshold``). Together they cut the summed
    bead-to-cell error by a fifth and the residual spillover between the
    scored pairs of a fully stained panel by about two fifths, and lower the
    change into detectors where a dye has no signal on two of the three
    panels, at no cost in run time or in agreement between halves of the
    events. ``smooth_updates``,
    ``smooth_rounds`` and ``smooth_fraction`` are read by ``smooth_fix`` and
    ``spectral_correction``, not by ``fix_my_unmix``.
    """
    max_iter: int = 20
    downsample: int = 0                     # 0: off; else events kept (stratified)
    downsample_background_frac: float = 0.3
    downsample_min_stratum: int = 2000
    unstained_threshold: float = 0.95
    lower_unstained_threshold: float = 0.99   # the lower (hypernegative) boundary's quantile, 1 - this, of the unstained
    unstained_margin: float = 1.3
    spread_kappa: float = 3.0
    negative_tail: float = 0.001            # tail share of the sample's negative population left above its bound
    envelope_quantile: float = 0.05
    compare_quantile: float = 0.5
    min_negative_frac: float = 0.10
    max_disagreement: float = 0.5
    disagreement_reference: str = 'envelope_corrected'  # slope the binned median is compared with, see DISAGREEMENT_REFERENCES
    envelope_scale_quantile: float = 0.25   # 'envelope_corrected': upper quantile of each bin's lower-tail scale
    min_hypernegative_frac: float = 0.01
    max_hypernegative_frac: float = 0.05
    min_hypernegative_events: int = 200
    hypernegative_scope: str = 'pair'       # or 'exclusive': leave out events other candidates move
    hypernegative_rule: str = 'after'       # see HYPERNEGATIVE_RULES
    hypernegative_dominant: bool = True     # judge only the source's own (dominated) positive events
    leakage_prior: bool = True
    span_fraction: float = 0.6
    source_events_low: int = 200            # clearly positive source events beyond the unstained rate: no trust below
    source_events_high: int = 400           # ... full trust above
    min_negative_events: int = 200
    min_bin_negative: int = 25
    min_span: float = 5.0
    min_rise: float = 1.0
    n_levels: int = 60
    min_bin_events: int = 50
    n_levels_pair: int = 10
    multivariate: bool = True
    ridge: float = 1e-6
    max_truncated_events: int = 20_000
    negative_bound_events: int = 50_000     # events of a channel's negative population behind its upper bound
    max_mask_passes: int = 3
    mask_tolerance: float = 0.05
    glasso_check: bool = True               # scale pair trust by agreement with the joint lasso
    glasso_warm: bool = True                # ... fitted after the pair slopes, its first mask removing them
    glasso_n_lambda: int = 40
    glasso_lambda_min_ratio: float = 1e-3
    glasso_margin_frac: float = 0.05
    glasso_tolerance: float = 1e-4
    glasso_refit: bool = True               # least squares on the lasso's selection (relaxed lasso)
    af_gate: bool = False                   # drop row changes that look like an AF spectrum
    af_gate_max: float = 0.8                # ... cos^2 to the nearest AF library spectrum above this
    plausibility_ratio: float = 1.0         # 0: off; else a detector may change by at most this x its reference value
    plausibility_floor: float = 0.003       # ... or by this (peak = 1), whichever is greater
    plausibility_reject: float = 0.5        # reject a row when more than this share of its change is out of bounds
    plausibility_reset: bool = True         # an adopted row's detector out of bounds keeps its reference value
    shared_direction: bool = False          # shrink coefficients that unrelated sources share into one channel
    shared_cosine: float = 0.5              # ... sources above this cosine to each other are related (count once)
    shared_min_slope: float = 0.01          # ... a source shares a direction with a coefficient of at least this
    shared_alignment: float = 0.5           # ... and its whole row change at least this cosine to the channel's dye
    shared_power: float = 1.0               # ... coefficient x (unrelated sources sharing it) ** -power
    af_penalty: bool = False                # damp a source's step that is more AF-like than the dye itself
    af_penalty_margin: float = 0.3          # ... no damping up to the dye's own AF likeness + margin
    af_penalty_width: float = 0.3           # ... no trust beyond that + width
    source_dominant: bool = True
    spread_addback: bool = False
    max_coefficient: float = 0.5
    convergence_threshold: float = 0.003
    convergence_quantile: float = 1.0
    stop_rule: str = 'settled'              # see STOP_RULES: every fitted error, the step applied, trusted, or settled
    stall_rule: str = 'trend'               # see STALL_RULES: 3-iteration trend of delta, or no new minimum
    stall_patience: int = 3                 # ... for stall_patience iterations
    stop_trust: float = 0.05                # 'settled': a coefficient below this trust does not hold phase one open
    stop_confirm: int = 2                   # 'settled': iterations in a row every trusted error must be below threshold
    coefficient_decay: float = 0.0
    update_spectra: bool = True
    phase_two_rows: str = 'all'             # see PHASE_TWO_ROWS
    phase_two_adopt: str = 'remeasured'     # see PHASE_TWO_ADOPT
    phase_two_fallback: str = 'remeasured'  # see PHASE_TWO_FALLBACK: a row too small for the joint fit
    step: float = 1.0
    intercept: bool = True
    min_explained: float = 0.8
    max_explained: float = 1.2
    max_resid: float = 0.03
    max_intercept: float = 0.03
    min_bg_align: float = -0.9
    max_angle: float = 30.0
    max_clamp_frac: float = 0.15
    max_anchor: float = 0.10
    max_vif: float = 500.0
    max_condition_increase: float = 1.05
    peak_shift_min_rel: float = 0.7
    leakage_margin: float = 0.05
    min_deg_start: float = 0.1
    huber_k: float = 1.345
    huber_max_iter: int = 100
    huber_tol: float = 1e-4
    residual_gate: str = 'off'              # phase two: 'no_increase' or 'decrease' of the residual
    residual_margin: float = 0.0            # relative tolerance of the residual gate
    use_kernel: bool = True                 # compiled pair slopes when built
    n_threads: int = 0                      # kernel threads; 0 = all cores
    keep_history: bool = False
    seed: int = 42
    seed_streams: bool = False              # one generator per purpose and fluorophore, not one shared
    smooth_updates: bool = True             # smooth_fix: rounds averaged, not one run adopted
    smooth_rounds: int = 5
    smooth_fraction: float = 0.5            # share of the stained events each round draws
    fixed_fluorophores: tuple = ()          # rows never corrected; names not in the panel ignored
    exclude_fixed_targets: bool = False     # fixed rows are also no target channel, lasso regressor or leakage target


# ---------------------------------------------------------------------------
# Small linear-algebra helpers
# ---------------------------------------------------------------------------

def unmixing_matrix(design: np.ndarray) -> np.ndarray:
    """OLS unmixing matrix (rows x detectors) of a design (rows x detectors)."""
    return np.linalg.solve(design @ design.T, design)


def unmix_ols(raw: np.ndarray, design: np.ndarray) -> np.ndarray:
    return raw @ unmixing_matrix(design).T


def cosine_similarity(spectra: np.ndarray) -> np.ndarray:
    norms = np.maximum(np.linalg.norm(spectra, axis=1), _EPS)
    unit = spectra / norms[:, None]
    return unit @ unit.T


def hotspot_matrix(spectra: np.ndarray) -> np.ndarray:
    """Unmixing spread scale, ``sqrt(|pinv(cosine similarity)|)``."""
    return np.sqrt(np.abs(np.linalg.pinv(cosine_similarity(spectra))))


def condition_number(spectra: np.ndarray) -> float:
    sv = np.linalg.svd(spectra, compute_uv=False)
    return float(round(sv.max() / max(sv.min(), _EPS), 2))


def row_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle in degrees between matching rows of ``a`` and ``b``."""
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    cs = np.einsum('ij,ij->i', a, b) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), _EPS)
    return np.degrees(np.arccos(np.clip(cs, -1.0, 1.0)))


def _mad(x: np.ndarray, axis=0) -> np.ndarray:
    med = np.median(x, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=axis)


def _weighted_line(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Closed-form weighted least-squares slope and its standard error."""
    sw = w.sum()
    xm = (w * x).sum() / sw
    ym = (w * y).sum() / sw
    sxx = (w * (x - xm) ** 2).sum()
    if not sxx > 0:
        return np.nan, np.nan
    slope = (w * (x - xm) * (y - ym)).sum() / sxx
    resid = y - (ym - slope * xm + slope * x)
    dof = len(x) - 2
    se = np.sqrt((w * resid ** 2).sum() / dof / sxx) if dof > 0 else np.nan
    return float(slope), float(se)



# ---------------------------------------------------------------------------
# Autofluorescence background
# ---------------------------------------------------------------------------

@dataclass
class Background:
    """
    The fixed background of a run. ``basis`` (rows x detectors: the AF mean
    row of the run's main AF library, or no rows without AF) is the
    background part of the design used for conditioning, leakage and the
    variant prior; ``af_spectra`` holds every AF spectrum the events were
    unmixed with (the AF gate and the AF likeness of a change), in the
    order of the projections' ``af_idx``.
    """
    basis: np.ndarray
    af_spectra: np.ndarray

    @property
    def n_bg(self) -> int:
        return int(self.basis.shape[0])


def background_for(libraries: list, n_detectors: int) -> Background:
    """
    Background of AF libraries (each spectra x detectors, row 0 its mean;
    None for a sample without AF), stacked in the given order: the basis is
    the first library's mean row.
    """
    used = [np.asarray(lib, dtype=np.float64) for lib in libraries if lib is not None and len(lib)]
    if not used:
        empty = np.zeros((0, n_detectors))
        return Background(basis=empty, af_spectra=empty.copy())
    return Background(basis=used[0][:1].copy(), af_spectra=np.vstack(used))


def project_events(raw: np.ndarray, spectra_fluor: np.ndarray, af_precomputed: dict | None = None,
                   af_spectra: np.ndarray | None = None) -> dict:
    """
    Raw fluorescence events unmixed by per-cell AF with one AF library
    (``apply_af_unmixing``; ``af_precomputed`` built on ``spectra_fluor``,
    scored with its envelope when it has one), or by OLS without one.

    Returns ``abundance`` (n, F, OLS space, uncompensated), ``residual``
    (n, D: the raw events less each event's AF), ``af_idx`` (0-based into
    ``af_spectra``, -1 without AF) and ``af_scale``.
    """
    from honeychrome.controller_components.autospectral_functions import apply_af_unmixing

    raw = np.ascontiguousarray(raw, dtype=np.float64)
    spectra_fluor = np.asarray(spectra_fluor, dtype=np.float64)
    n = len(raw)
    if af_precomputed is None or af_spectra is None or len(af_spectra) == 0:
        return {'abundance': unmix_ols(raw, spectra_fluor), 'residual': raw.copy(),
                'af_idx': np.full(n, -1, dtype=np.int64), 'af_scale': np.zeros(n)}
    lib = np.asarray(af_spectra, dtype=np.float64)
    out = apply_af_unmixing(raw, af_precomputed, lib)
    idx = np.asarray(out['af_idx'], dtype=np.int64) - 1
    scale = np.asarray(out['af_scale'], dtype=np.float64)
    return {'abundance': np.asarray(out['unmixed'], dtype=np.float64),
            'residual': raw - scale[:, None] * lib[idx], 'af_idx': idx, 'af_scale': scale}


def concatenate(projections: list, offsets: list | None = None) -> dict:
    """
    Projections of several samples as one; ``offsets`` (one per projection)
    shift each one's ``af_idx`` into the stacked AF library of
    ``background_for``.
    """
    if not projections:
        raise ValueError('Nothing to concatenate.')
    offsets = [0] * len(projections) if offsets is None else list(offsets)
    out = {}
    for key in ('abundance', 'residual', 'af_scale'):
        out[key] = np.concatenate([np.asarray(p[key], dtype=np.float64) for p in projections], axis=0)
    out['af_idx'] = np.concatenate([np.where(np.asarray(p['af_idx']) >= 0, np.asarray(p['af_idx']) + o, -1)
                                    for p, o in zip(projections, offsets)]).astype(np.int64)
    return out


def af_likeness(change: np.ndarray, af_spectra: np.ndarray) -> float:
    """
    How much a spectral change looks like autofluorescence: the largest
    squared cosine between ``change`` (detectors) and any AF library
    spectrum; 0 for no change or no AF.
    """
    change = np.asarray(change, dtype=np.float64)
    norm_c = float(np.linalg.norm(change))
    lib = np.asarray(af_spectra, dtype=np.float64)
    if not norm_c > 0 or lib.size == 0:
        return 0.0
    cos = (lib @ change) / np.maximum(np.linalg.norm(lib, axis=1) * norm_c, _EPS)
    return float(np.max(cos ** 2))


# ---------------------------------------------------------------------------
# Plausibility bounds
# ---------------------------------------------------------------------------

def plausible_change(reference: np.ndarray, ratio: float, floor: float) -> np.ndarray:
    """
    The largest change each detector of a row may take (peak-normalised
    reference rows): ``ratio`` x the reference value or ``floor``,
    whichever is greater. A detector with no starting signal may gain no
    more than the floor; a detector's value cannot more than double at
    ratio 1.
    """
    ref = np.maximum(np.asarray(reference, dtype=np.float64), 0.0)
    return np.maximum(ratio * ref, floor)


def clip_to_plausible(row: np.ndarray, reference: np.ndarray, allow: np.ndarray,
                      reset: bool = False) -> tuple[np.ndarray, float]:
    """
    ``row`` with each detector's change from ``reference`` clipped to
    +/- ``allow`` (with ``reset``, a detector out of bounds takes no change
    at all), peak-normalised and non-negative, and the share of the change
    (Euclidean norm) that was out of bounds.
    """
    ref = np.asarray(reference, dtype=np.float64)
    change = np.asarray(row, dtype=np.float64) - ref
    norm_c = float(np.linalg.norm(change))
    if not norm_c > 0:
        return np.asarray(row, dtype=np.float64).copy(), 0.0
    kept = np.clip(change, -allow, allow)
    if reset:
        kept = np.where(np.abs(change) > allow, 0.0, change)
    out = np.maximum(ref + kept, 0.0)
    peak = out.max()
    out = out / peak if peak > 0 else ref.copy()
    return out, float(np.linalg.norm(change - np.clip(change, -allow, allow)) / norm_c)


def _plausible_project(spill: np.ndarray, spectra: np.ndarray, reference: np.ndarray, allow: np.ndarray,
                       passes: int = 3, rows: np.ndarray | None = None,
                       basis: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    ``spill`` with each row's off-diagonal coefficients adjusted, by least
    squares over the other rows' spectra, so that its back-solved row moves
    back toward reference +/- allow where it lies outside (``passes``
    rounds; a part of the excess outside the panel's span cannot be taken
    back and is left to the clip of the adopted row). Returns the
    spillover and, per row, the share of its change from the reference that
    was out of bounds before the first pass (0 for rows inside). ``rows``
    (bool) limits the rows adjusted and ``basis`` (bool) the rows whose
    coefficients may be adjusted (default all).
    """
    out = spill.copy()
    n = out.shape[0]
    share = np.zeros(n)
    rows = np.ones(n, dtype=bool) if rows is None else np.asarray(rows, dtype=bool)
    basis_ok = np.ones(n, dtype=bool) if basis is None else np.asarray(basis, dtype=bool)
    for f in np.flatnonzero(rows):
        others = np.flatnonzero(basis_ok & (np.arange(n) != f))
        if others.size == 0:
            continue
        cols = spectra[others].T                            # D x (others)
        for k in range(passes):
            row = np.maximum(out[f] @ spectra, 0.0)
            peak = row.max()
            if not peak > 0:
                break
            normed = row / peak
            clipped, frac = clip_to_plausible(normed, reference[f], allow[f])
            if k == 0:
                share[f] = frac
            if frac <= 1e-6:
                break
            delta = (clipped - normed) * peak
            coef = np.linalg.lstsq(cols, delta, rcond=None)[0]
            out[f, others] += coef
    return out, share


def cosine_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n, m) cosines between the rows of ``a`` and of ``b`` (0 for a zero row)."""
    na = np.linalg.norm(a, axis=1, keepdims=True)
    nb = np.linalg.norm(b, axis=1, keepdims=True)
    return (a @ b.T) / np.maximum(na * nb.T, _EPS)


def shared_direction_counts(marker: np.ndarray, trust: np.ndarray, candidates: list, cosine: np.ndarray,
                            s: 'FixSettings', aligned: np.ndarray | None = None) -> np.ndarray:
    """
    Per coefficient (source, channel), how many mutually unrelated sources
    (cosine below ``shared_cosine``; with ``aligned``, only the (source,
    channel) pairs it marks count) carry a trusted slope of at least ``shared_min_slope``
    with the same sign into that channel; 1 where it is not shared or the
    source is related to the channel. Real residual spillover is a
    property of each source's row, so unrelated rows erring toward the same
    fluorophore point to that fluorophore being expressed on their cells.
    """
    counts = np.ones_like(marker)
    by_channel: dict = {}
    for src, ch, _r in candidates:
        if aligned is not None and not aligned[src, ch]:
            continue
        if trust[src, ch] > 0 and abs(marker[src, ch]) >= s.shared_min_slope:
            by_channel.setdefault((ch, marker[src, ch] > 0), []).append(src)
    for (ch, _sign), members in by_channel.items():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda m: -abs(marker[m, ch]))
        picked = []
        for m in members:
            if all(cosine[m, p] < s.shared_cosine for p in picked):
                picked.append(m)
        if len(picked) >= 2:
            counts[members, ch] = len(picked)
    return counts


# ---------------------------------------------------------------------------
# Positivity boundaries and the variant prior
# ---------------------------------------------------------------------------

def align_spread(spread: np.ndarray | None, n: int) -> np.ndarray:
    """Finite, non-negative, zero-diagonal source x target spread (``.align.spillover.spread``)."""
    if spread is None:
        return np.zeros((n, n))
    s = np.where(np.isfinite(spread), spread, 0.0)
    s = np.maximum(s, 0.0)
    np.fill_diagonal(s, 0.0)
    return s


def spread_thresholds(unmixed: np.ndarray, thresholds: np.ndarray, spread: np.ndarray | None,
                      kappa: float, side: str = 'upper') -> np.ndarray:
    """
    Per-event boundaries (``get.spread.thresholds()``): the flat threshold
    widened by ``kappa * sqrt(sum_a SS[a, f] * max(x_a, 0))``, added for the
    upper side and subtracted for the lower. Flat when ``spread`` is None or
    zero.
    """
    n = unmixed.shape[0]
    if spread is None or not np.any(spread):
        return np.broadcast_to(thresholds, (n, len(thresholds))).copy()
    sign = 1.0 if side == 'upper' else -1.0
    return thresholds[None, :] + sign * kappa * np.sqrt(np.maximum(unmixed, 0.0) @ spread)


def variant_leakage_prior(spectra_fluor: np.ndarray, deltas: list, extra_rows: np.ndarray | None = None,
                          span_fraction: float = 0.6, min_variants: int = 3) -> dict | None:
    """
    Prior variance of each residual spillover coefficient
    (``get.variant.leakage.prior()``). Each fluorophore's variant deltas
    (variants x detectors, or None) are pushed through the production
    unmixing matrix: ``variance[f, c]`` is the spillover from ``f`` into
    ``c`` a plausible reference error could produce. A ridge sized from
    ``span_fraction`` keeps directions the variants never explored possible.
    Fluorophores with fewer than ``min_variants`` take the panel median.

    Returns ``{'variance': (F, F) NaN diagonal, 'fallback': [indices]}``, or
    None when no fluorophore has enough variants.
    """
    n_fluor, n_det = spectra_fluor.shape
    design = spectra_fluor if extra_rows is None else np.vstack([extra_rows, spectra_fluor])
    unmixing = unmixing_matrix(design)[design.shape[0] - n_fluor:]
    variance = np.full((n_fluor, n_fluor), np.nan)
    fallback = []
    for j in range(n_fluor):
        delta = None if j >= len(deltas) else deltas[j]
        if delta is None or np.asarray(delta).shape[0] < min_variants:
            fallback.append(j)
            continue
        delta = np.asarray(delta, dtype=np.float64)
        if delta.shape[1] != n_det:
            raise ValueError(f'Variant deltas of row {j} have {delta.shape[1]} detectors, spectra have {n_det}.')
        sigma_e = np.cov(delta, rowvar=False)
        sigma_e[np.diag_indices(n_det)] += (1.0 - span_fraction) / span_fraction * np.trace(sigma_e) / n_det
        variance[j] = np.maximum(np.diag(unmixing @ sigma_e @ unmixing.T), 0.0)
    if len(fallback) == n_fluor:
        return None
    if fallback:
        median = np.nanmedian(variance, axis=0)
        for j in fallback:
            variance[j] = median
    np.fill_diagonal(variance, np.nan)
    return {'variance': variance, 'fallback': fallback}


def _dynamic_range(unmixed: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.maximum(np.quantile(unmixed, 0.999, axis=0) - thresholds, _EPS)


def _dominance(unmixed: np.ndarray, thresholds: np.ndarray, threshold_matrix: np.ndarray,
               dyn: np.ndarray | None = None) -> np.ndarray:
    """
    Winning fluorophore per event, scored against its own boundary over each
    channel's dynamic range (``dyn``, by default that of ``unmixed``); -1 for
    none.
    """
    dyn = _dynamic_range(unmixed, thresholds) if dyn is None else dyn
    score = np.maximum(unmixed - threshold_matrix, 0.0) / dyn
    dom = np.argmax(score, axis=1)
    dom[score[np.arange(len(dom)), dom] <= 0] = -1
    return dom


def half_sample_mode(x: np.ndarray) -> float:
    """
    Robust mode (Bickel's half-sample mode): repeatedly keep the densest
    half of the sorted values until three or fewer remain. Needs no
    bandwidth, so the long positive tail of a channel does not blur it.
    """
    y = np.sort(np.asarray(x, dtype=np.float64)[np.isfinite(x)])
    if y.size == 0:
        return np.nan
    while y.size > 3:
        h = int(np.ceil(y.size / 2))
        widths = y[h - 1:] - y[:y.size - h + 1]
        i = int(np.argmin(widths))
        y = y[i:i + h]
    if y.size == 3:
        lo, hi = y[1] - y[0], y[2] - y[1]
        return float(y[:2].mean() if lo < hi else y[1:].mean() if hi < lo else y[1])
    return float(y.mean())


def negative_bound(x: np.ndarray, t_event: np.ndarray, tail: float, rng, max_events: int = 50_000,
                   min_events: int = 200) -> float:
    """
    Upper bound of a sample's own negative population in one channel, as the
    autogating plugin's ``tail`` threshold: the events below their own
    positivity boundary ``t_event`` (so a majority-positive marker still
    gives the negative mode) are mirrored about their mode, and the bound is
    the mirrored population's ``1 - tail`` quantile,
    ``2 mode - quantile(lower half, 2 tail)``. The spread of the negative
    population in the stained sample (other dyes' spillover spread,
    collinearity) is in this bound; the unstained threshold does not see
    it. -inf when too few events are negative.
    """
    neg = np.asarray(x, dtype=np.float64)[np.asarray(x) <= t_event]
    if neg.size < min_events:
        return -np.inf
    if neg.size > max_events:
        neg = neg[rng.choice(neg.size, max_events, replace=False)]
    mode = half_sample_mode(neg)
    lower = neg[neg <= mode]
    if lower.size < 5:
        return -np.inf
    return float(2.0 * mode - np.quantile(lower, min(max(2.0 * tail, 0.0), 1.0)))


def smoothstep(value, low: float, high: float):
    """0 at or below ``low``, 1 at or above ``high``, a smooth S-curve between."""
    t = np.clip((np.asarray(value, dtype=np.float64) - low) / max(high - low, _EPS), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# Pair estimator
# ---------------------------------------------------------------------------

def _huber_slope_origin(x: np.ndarray, y: np.ndarray, start_slope: float, k: float, max_iter: int,
                        tol: float) -> tuple[float, float]:
    """
    Huber IRLS slope of a line through the origin, warm-started at
    ``start_slope``, and its standard error (robust scale over
    ``sqrt(sum w x^2)``). ``(0, NaN)`` when it does not converge.
    """
    b = float(start_slope)
    for _ in range(int(max_iter)):
        r = y - b * x
        scale = max(np.median(np.abs(r - np.median(r))) / 0.6745, _EPS)
        w = np.minimum(1.0, k / np.maximum(np.abs(r / scale), _EPS))
        sxx = float(w @ (x * x))
        if not sxx > 0:
            return 0.0, np.nan
        b_new = float(w @ (x * y)) / sxx
        settled = abs(b_new - b) < tol * max(abs(b_new), 1.0)
        b = b_new
        if settled:
            r = y - b * x
            scale = max(np.median(np.abs(r - np.median(r))) / 0.6745, _EPS)
            w = np.minimum(1.0, k / np.maximum(np.abs(r / scale), _EPS))
            return b, scale / np.sqrt(max(float(w @ (x * x)), _EPS))
    return 0.0, np.nan


def _anchored_numpy(xb, Yb, Tb, Lb, xk, Yk, Tk, start, floor: int, s: FixSettings):
    m = Yb.shape[1]
    slope_out = np.full(m, np.nan)
    se_out = np.full(m, np.nan)
    n_out = np.zeros(m, dtype=np.int64)
    for j in range(m):
        slope_curr = start[j]
        if not np.isfinite(slope_curr) or abs(slope_curr) > s.max_coefficient:
            slope_curr = 0.0
        yb, yk = Yb[:, j], Yk[:, j]
        tb, tk = Tb[:, j], Tk[:, j]
        lb = None if Lb is None else Lb[:, j]
        for p in range(int(s.max_mask_passes) + 1):
            neg_k = (yk - slope_curr * xk) < tk
            resid_b = yb - slope_curr * xb
            neg_b = resid_b < tb
            if lb is not None and p > 0:
                neg_b &= resid_b > lb
            if neg_k.sum() < floor or neg_b.sum() < floor:
                break
            x0 = float(np.median(xk[neg_k]))
            y0 = float(np.median(yk[neg_k]))
            slope_next, se = _huber_slope_origin(xb[neg_b] - x0, yb[neg_b] - y0, slope_curr, s.huber_k,
                                                 s.huber_max_iter, s.huber_tol)
            n_out[j] = int(neg_b.sum())
            slope_out[j] = slope_next
            se_out[j] = se
            valid = abs(slope_next) <= s.max_coefficient
            settled = abs(slope_next - slope_curr) <= s.mask_tolerance * abs(slope_next)
            slope_curr = slope_next
            if not valid or settled:
                break
    return slope_out, n_out, se_out


def _anchored_kernel(xb, Yb, Tb, Lb, xk, Yk, Tk, start, floor: int, s: FixSettings):
    ffi, lib = _kernel.ffi, _kernel.lib
    m = Yb.shape[1]

    def arr(a):
        return np.ascontiguousarray(a, dtype=np.float64)

    xb_c, xk_c, st = arr(xb), arr(xk), arr(start)
    yb_t, tb_t, yk_t, tk_t = arr(Yb.T), arr(Tb.T), arr(Yk.T), arr(Tk.T)
    lb_t = None if Lb is None else arr(Lb.T)
    slope_out = np.empty(m, dtype=np.float64)
    se_out = np.empty(m, dtype=np.float64)
    n_out = np.empty(m, dtype=np.int32)

    def ptr(kind, a):
        return ffi.NULL if a is None else ffi.cast(kind, a.ctypes.data)

    lib.fmu_anchored_batch(
        ptr('double *', xb_c), ptr('double *', yb_t), ptr('double *', tb_t), ptr('double *', lb_t), int(len(xb_c)),
        ptr('double *', xk_c), ptr('double *', yk_t), ptr('double *', tk_t), int(len(xk_c)),
        ptr('double *', st), int(m), float(s.max_coefficient), int(s.max_mask_passes), float(s.mask_tolerance),
        int(floor), int(lb_t is not None), float(s.huber_k), int(s.huber_max_iter), float(s.huber_tol),
        int(s.n_threads), ptr('double *', slope_out), ptr('int *', n_out), ptr('double *', se_out))
    return slope_out, n_out.astype(np.int64), se_out


def anchored_slopes(x: np.ndarray, Y: np.ndarray, T_target: np.ndarray, t_source: np.ndarray, start: np.ndarray,
                    rng, s: FixSettings,
                    T_lower: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Residual spillover slope from one source into every target, fitted on
    the source's clearly positive events (``x > t_source``) through the
    robust centre of the source-negative bulk.

    The bulk fixes where the line passes through the origin but does not
    set its slope. Its own spread in source and target comes from other
    dyes' spillover spread, the noise covariance of collinear rows and AF
    residuals, none of which is spillover from the source; in a fit over all
    target-negative events it would dominate a small positive population.

    Negativity for the target is decided by its upper boundary ``T_target``
    with the current slope removed, and re-decided up to
    ``max_mask_passes`` times. With ``T_lower`` (the per-event negative
    boundaries) every pass after the first also drops positive events below
    the lower boundary: a one-sided cut removes only the upper tail of the
    source's own cell-to-cell spectral variation and so pulls the slope
    down, the more so the narrower the boundary is against that variation,
    while a band around the current line trims both tails. The first pass
    stays one-sided, so a large negative coefficient is found before the
    band centres on it. The bulk is subsampled to ``max_truncated_events``
    for its centre; the positive events are capped at the same number.
    Runs in the compiled kernel when it is built and ``use_kernel`` is set,
    with the same results.

    Returns ``(slope, n, se)``: NaN where fewer than ``min_bin_negative``
    positive or bulk events are target-negative.
    """
    m = Y.shape[1]
    floor = max(int(s.min_bin_negative), 3)
    bright_all = x > t_source
    bright = np.flatnonzero(bright_all)
    bulk = np.flatnonzero(~bright_all)
    if m == 0 or bright.size < floor or bulk.size < floor:
        return np.full(m, np.nan), np.zeros(m, dtype=np.int64), np.full(m, np.nan)
    if bulk.size > s.max_truncated_events:
        bulk = np.sort(rng.choice(bulk, int(s.max_truncated_events), replace=False))
    if bright.size > s.max_truncated_events:
        bright = np.sort(rng.choice(bright, int(s.max_truncated_events), replace=False))
    args = (x[bright], Y[bright], T_target[bright], None if T_lower is None else T_lower[bright],
            x[bulk], Y[bulk], T_target[bulk], start, floor, s)
    if _kernel is not None and s.use_kernel:
        return _anchored_kernel(*args)
    return _anchored_numpy(*args)


def kernel_status() -> str:
    """One line on whether the compiled pair-slope kernel is in use."""
    if _kernel is None:
        return 'numpy (compiled kernel not built or out of date: run build_fix_my_unmix_kernel.py)'
    return f'compiled kernel, up to {int(_kernel.lib.fmu_max_threads())} thread(s)'


def _cut_bins(values: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    """1-based right-closed bins, as R's ``cut()`` with ``include.lowest = TRUE``."""
    return np.clip(np.searchsorted(breaks, values, side='left'), 1, len(breaks) - 1)


def envelope_slopes(x: np.ndarray, Y: np.ndarray, t_source: np.ndarray, T_target: np.ndarray,
                    spread_var: np.ndarray, neg_var: np.ndarray, source_mask: np.ndarray | None,
                    start: np.ndarray, rng, s: FixSettings, T_lower: np.ndarray | None = None) -> dict | None:
    """
    Residual spillover of one source into every target, with the
    diagnostics its trust weight needs (after ``.fix.envelope.slope.batch()``).

    ``t_source`` is the per-event clearly-positive boundary of the source.
    Events positive for the source but dominated by another fluorophore are
    dropped first: their source abundance is that fluorophore read through
    a spectral error. ``slope_anchored`` and its ``se_fit`` come from
    ``anchored_slopes``. The binned envelope slope (``slope``, with ``se``)
    is a weighted line through the ``envelope_quantile`` of the
    target-negative events in quantile bins of the source-positive
    abundance, and ``slope_alt`` the same through the ``compare_quantile``
    of all events, so ``disagreement`` between the two reports
    co-expression. The lower envelope falls with the spread as the source
    brightens, so it runs steeper than the line it bounds. With
    ``disagreement_reference='anchored'`` the median is compared with the
    anchored slope instead, which has no such bias but moves with any
    co-expression inside the band, as the median does; with
    ``'envelope_corrected'`` each bin's envelope is moved back to its centre
    by the bin's own lower-tail scale, ``(q(envelope_scale_quantile) -
    q(envelope_quantile)) / (z(envelope_quantile) - z(envelope_scale_quantile))``,
    which needs no spread model and stays below most co-expression. Either
    way the envelope's own line still gives the standard error. ``coverage`` is the share of target-negative events in
    the brightest third of the source population, the identifiability test
    for co-expression. ``n_positive`` is the number of positive source
    events used.

    ``Y``, ``T_target`` and ``T_lower`` are (n, m); a transposed view of a
    target-major (m, n) array avoids every copy.

    Returns a dict of (m,) arrays, or None when too few events remain.
    """
    m = Y.shape[1]
    Yt = np.ascontiguousarray(Y.T)                          # targets x events
    Tt = np.ascontiguousarray(T_target.T)
    Lt = None if T_lower is None else np.ascontiguousarray(T_lower.T)
    if source_mask is not None:
        keep = np.flatnonzero(source_mask | (x <= t_source))
        x, t_source = x[keep], t_source[keep]
        Yt, Tt = np.take(Yt, keep, axis=1), np.take(Tt, keep, axis=1)
        Lt = None if Lt is None else np.take(Lt, keep, axis=1)
    n = len(x)
    if n < s.min_negative_events:
        return None

    seed = int(rng.integers(0, 2 ** 31 - 1))
    slope_t, n_t, se_fit = anchored_slopes(x, Yt.T, Tt.T, t_source, start, np.random.default_rng(seed), s,
                                           None if Lt is None else Lt.T)
    slope_mask = np.where(np.isfinite(slope_t) & (np.abs(slope_t) <= s.max_coefficient), slope_t, 0.0)
    neg = (Yt - slope_mask[:, None] * x[None, :]) < Tt      # targets x events
    pos = x > t_source

    coverage = np.ones(m)
    if pos.sum() >= s.min_bin_negative:
        bright = np.flatnonzero(pos)
        cut = np.quantile(x[bright], 2.0 / 3.0)
        top = bright[x[bright] >= cut]
        if top.size >= s.min_bin_negative:
            coverage = neg[:, top].mean(axis=1)

    span = np.zeros(m)
    for k in range(m):
        sel = x[neg[k]]
        span[k] = np.ptp(sel) if sel.size > 1 else 0.0

    out = {'slope': np.full(m, np.nan), 'slope_anchored': slope_t, 'se': np.full(m, np.nan),
           'slope_alt': np.full(m, np.nan), 'disagreement': np.full(m, np.nan), 'coverage': coverage,
           'n': n_t, 'span': span, 'noise': np.sqrt(np.maximum(neg_var, 0.0)), 'se_fit': se_fit,
           'n_positive': int(pos.sum())}

    if pos.sum() < s.min_negative_events or (~pos).sum() < s.min_bin_negative:
        return out
    brk = np.unique(np.quantile(x[pos], np.linspace(0.0, 1.0, s.n_levels_pair + 1)))
    if brk.size < 4:
        return out
    pos_idx = np.flatnonzero(pos)
    bin_of = _cut_bins(x[pos_idx], brk)
    bins = np.unique(bin_of)
    if bins.size < 5:
        return out
    groups = [pos_idx[bin_of == b] for b in bins]
    centre = np.array([np.median(x[g]) for g in groups])
    bin_n = np.array([g.size for g in groups], dtype=np.float64)
    neg_n = np.array([neg[:, g].sum(axis=1) for g in groups], dtype=np.float64)    # (B, m)
    usable = neg_n >= s.min_bin_negative
    sd_bin = np.sqrt(np.maximum(np.outer(np.maximum(centre, 0.0), spread_var) + neg_var[None, :], 0.0))
    z = -norm.ppf([s.envelope_quantile, s.compare_quantile]) if s.spread_addback else np.zeros(2)
    var_floor = np.maximum(sd_bin ** 2, _EPS)
    w_env = neg_n / var_floor
    w_cmp = bin_n[:, None] / var_floor

    env_val = np.full((len(groups), m), np.nan)
    med_val = np.empty((len(groups), m))
    corrected = s.disagreement_reference == 'envelope_corrected'
    scale_val = np.full((len(groups), m), np.nan) if corrected else None
    for b, g in enumerate(groups):
        yg = np.take(Yt, g, axis=1)
        med_val[b] = np.quantile(yg, s.compare_quantile, axis=1)
        env_val[b] = _masked_quantile(yg, np.take(neg, g, axis=1), s.envelope_quantile, s.min_bin_negative)
        if corrected:
            scale_val[b] = _masked_quantile(yg, np.take(neg, g, axis=1), s.envelope_scale_quantile,
                                            s.min_bin_negative)

    def fit(value, weight):
        slope = np.full(m, np.nan)
        se = np.full(m, np.nan)
        for k in range(m):
            keep = usable[:, k] & np.isfinite(value[:, k])
            if keep.sum() < 4:
                continue
            slope[k], se[k] = _weighted_line(centre[keep], value[keep, k], weight[keep, k])
        return slope, se

    env_slope, env_se = fit(env_val + z[0] * sd_bin, w_env)
    cmp_slope, _ = fit(med_val + z[1] * sd_bin, w_cmp)
    out['slope'] = env_slope
    out['se'] = env_se
    out['slope_alt'] = cmp_slope
    if s.disagreement_reference == 'envelope':
        ref_slope = env_slope
    elif s.disagreement_reference == 'anchored':
        ref_slope = np.where(np.isfinite(slope_t) & (np.abs(slope_t) <= s.max_coefficient), slope_t, np.nan)
    else:
        z_env, z_scale = -norm.ppf([s.envelope_quantile, s.envelope_scale_quantile])
        ref_slope, _ = fit(env_val + z_env * (scale_val - env_val) / (z_env - z_scale), w_env)
    out['disagreement'] = np.abs(ref_slope - cmp_slope) / (np.abs(ref_slope) + np.abs(cmp_slope) + _EPS)
    out['span'] = np.full(m, centre.max() - centre.min())
    return out


def _masked_quantile(values: np.ndarray, mask: np.ndarray, q: float, min_count: int) -> np.ndarray:
    """
    Per row of ``values`` (rows x events), the ``q`` quantile (numpy's
    default linear method) of the entries where ``mask`` is set; NaN for a
    row with fewer than ``min_count`` such entries.
    """
    count = mask.sum(axis=1)
    ordered = np.sort(np.where(mask, values, np.inf), axis=1)
    out = np.full(values.shape[0], np.nan)
    ok = count >= max(int(min_count), 1)
    if not ok.any():
        return out
    rows = np.flatnonzero(ok)
    pos = q * (count[rows] - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, count[rows] - 1)
    frac = pos - lo
    a = ordered[rows, lo]
    b = ordered[rows, hi]
    out[rows] = a + (b - a) * frac
    return out


# ---------------------------------------------------------------------------
# Joint lasso estimator (correct.spectra.glasso)
# ---------------------------------------------------------------------------

def _lasso_path(X: np.ndarray, y: np.ndarray, lambdas: np.ndarray | None, n_lambda: int,
                lambda_min_ratio: float) -> dict:
    """
    Lasso path of ``y`` on ``X`` with centred, standardised predictors and
    an unpenalised intercept (``.glasso.lasso.path()``): minimises
    ``||y - a - Z b||^2 / 2n + lambda * |b|_1`` by coordinate descent on the
    Gram matrix. ``lambdas`` None builds the path from ``lambda_max`` down to
    ``lambda_min_ratio`` of it. Returns ``lambda`` (L,), ``beta`` (p, L) on
    the original scale and ``intercept`` (L,).
    """
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import lasso_path

    n, p = X.shape
    x_centre = X.mean(axis=0)
    y_mean = float(y.mean())
    xc = X - x_centre
    yc = y - y_mean
    x_scale = np.sqrt(np.mean(xc * xc, axis=0))
    x_scale = np.where(x_scale > 0, x_scale, 1.0)
    z = xc / x_scale
    xy = z.T @ yc
    if lambdas is None:
        lambda_max = float(np.max(np.abs(xy))) / n
        if not np.isfinite(lambda_max) or lambda_max <= 0:
            return {'lambda': np.zeros(n_lambda), 'beta': np.zeros((p, n_lambda)),
                    'intercept': np.full(n_lambda, y_mean)}
        lambdas = lambda_max * lambda_min_ratio ** np.linspace(0.0, 1.0, int(n_lambda))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        _alphas, coefs, _gaps = lasso_path(z, yc, alphas=np.asarray(lambdas, dtype=np.float64),
                                           precompute=z.T @ z, Xy=xy, tol=1e-4, max_iter=10_000)
    beta = coefs / x_scale[:, None]
    return {'lambda': np.asarray(lambdas, dtype=np.float64), 'beta': beta, 'intercept': y_mean - x_centre @ beta}


def glasso_row(X: np.ndarray, y: np.ndarray, s: FixSettings) -> tuple[np.ndarray, float]:
    """
    One row of the residual spillover matrix by neighbourhood-selection
    lasso (``.glasso.select.lambda()``). The penalty comes from a two-way
    split-half holdout over a shared grid: the largest ``lambda`` (sparsest
    fit) giving up no more than ``glasso_margin_frac`` of the error
    reduction on offer is refitted on all events. With ``glasso_refit`` the
    selected sources are then refitted by least squares, without the
    penalty's shrinkage (R keeps the shrunken coefficients, which leaves
    every correction short by the penalty). Returns the coefficients and
    the chosen ``lambda``.
    """
    full = _lasso_path(X, y, None, s.glasso_n_lambda, s.glasso_lambda_min_ratio)
    grid = full['lambda']
    if not np.any(grid > 0):
        return np.zeros(X.shape[1]), 0.0
    half = np.arange(len(y)) % 2

    def holdout(train, test):
        fit = _lasso_path(X[train], y[train], grid, s.glasso_n_lambda, s.glasso_lambda_min_ratio)
        pred = X[test] @ fit['beta'] + fit['intercept'][None, :]
        return np.mean((y[test][:, None] - pred) ** 2, axis=0)

    total = holdout(half == 0, half == 1) + holdout(half == 1, half == 0)
    best = int(np.argmin(total))
    reduction = total[0] - total[best]
    tolerated = np.ones(len(total), dtype=bool) if not (np.isfinite(reduction) and reduction > 0) \
        else total <= total[best] + s.glasso_margin_frac * reduction
    li = int(np.flatnonzero(tolerated)[np.argmax(grid[tolerated])])
    beta = full['beta'][:, li].copy()
    support = np.flatnonzero(beta)
    if s.glasso_refit and support.size:
        design = np.column_stack([np.ones(len(y)), X[:, support]])
        beta[support] = np.linalg.lstsq(design, y, rcond=None)[0][1:]
    return beta, float(grid[li])


def glasso_matrix(u: np.ndarray, tm: np.ndarray, thresholds: np.ndarray, neg_var: np.ndarray, rng,
                  s: FixSettings, warm: np.ndarray | None = None) -> np.ndarray:
    """
    Residual spillover matrix (source x target, zero diagonal) by
    ``correct.spectra.glasso()``'s phase one: every target's compensated
    abundance over its own negative events is regressed jointly on every
    other fluorophore's abundance with an L1 penalty, so a source that only
    correlates with the target through a third dye has that correlation
    absorbed by the third dye's coefficient. Negativity is re-decided with
    the fitted row removed up to ``max_mask_passes`` times. Coefficients
    below ``glasso_tolerance`` or above ``max_coefficient`` are zeroed, and
    a selected source must span ``min_span`` times its threshold over the
    target-negative events and rise by ``min_rise`` noise SDs across it.
    ``rng`` is one generator, or one per target. With ``warm`` (source x
    target slopes) the first negativity mask has them removed, so a source
    whose spillover lifts its positive events above the target's boundary
    is not left out of the fit that should find it.
    """
    n_fluor = u.shape[1]
    rngs = rng if isinstance(rng, (list, tuple)) else [rng] * n_fluor
    out = np.zeros((n_fluor, n_fluor))
    for t in range(n_fluor):
        rng = rngs[t]
        sources = np.array([k for k in range(n_fluor) if k != t])
        lifted = 0.0 if warm is None else u[:, sources] @ warm[sources, t]
        neg = np.flatnonzero(u[:, t] - lifted <= tm[:, t])
        if neg.size < s.min_negative_events:
            continue
        beta = None
        for _p in range(max(int(s.max_mask_passes), 1)):
            idx = neg if neg.size <= s.max_truncated_events else \
                np.sort(rng.choice(neg, int(s.max_truncated_events), replace=False))
            b, _lam = glasso_row(u[np.ix_(idx, sources)], u[idx, t], s)
            b[np.abs(b) < s.glasso_tolerance] = 0.0
            b[np.abs(b) > s.max_coefficient] = 0.0
            beta = b
            nxt = np.flatnonzero((u[:, t] - u[:, sources] @ b) <= tm[:, t])
            if nxt.size < s.min_negative_events:
                break
            moved = np.setxor1d(neg, nxt, assume_unique=True).size
            settled = moved / max(neg.size, 1) <= s.mask_tolerance
            neg = nxt
            if settled:
                break
        noise = np.sqrt(max(float(neg_var[t]), 0.0))
        for k in np.flatnonzero(beta):
            span = float(np.ptp(u[neg, sources[k]])) if neg.size > 1 else 0.0
            if not (span > s.min_span * abs(thresholds[sources[k]]) and abs(beta[k]) * span >= s.min_rise * noise):
                beta[k] = 0.0
        out[sources, t] = beta
    return out


# ---------------------------------------------------------------------------
# Raw-space signature
# ---------------------------------------------------------------------------

def _vif(x: np.ndarray) -> np.ndarray:
    """Variance inflation factor of each column (``.signature.vif()``)."""
    if x.shape[1] < 2:
        return np.ones(x.shape[1])
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.corrcoef(x, rowvar=False)
    r = np.where(np.isfinite(r), r, 0.0)
    np.fill_diagonal(r, 1.0)
    try:
        inv = np.linalg.inv(r)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(r)
    return np.maximum(np.diag(inv), 1.0)


def extract_raw_signature(raw: np.ndarray, spectra: np.ndarray, abundance: np.ndarray, target: int,
                          active: list | None = None, intercept: bool = True, multivariate: bool = True,
                          ridge: float = 1e-6, n_levels: int = 60, min_bin_events: int = 50,
                          min_events: int = 200, background_raw: np.ndarray | None = None,
                          detector_names: list | None = None) -> dict | None:
    """
    One fluorophore's signature from background-subtracted raw data of its
    own population (``extract.raw.signature()``).

    raw       : (n, D) raw events of the population, background removed.
    spectra   : (F, D) current spectra; rows other than ``target`` are fixed.
    abundance : (n, F) current abundance estimates of the same events.
    active    : fluorophore indices whose contribution is removed (default
                all); the target is always first.

    Events are binned by the target's abundance quantiles, as finely as
    ``min_bin_events`` per bin allows up to ``n_levels``. With
    ``multivariate`` every active abundance is a regressor of the bin-mean
    detector trace (ridge-stabilised, intercept unpenalised), so an error
    in a neighbouring row is not booked against the target. Too few bins
    for that falls back to subtracting the other rows at their current
    spectra and fitting the target alone.

    Returns None when the population cannot support a fit, otherwise
    ``signature`` (non-negative, L-inf normalised), ``signature_raw`` and
    ``stats``: n_events, n_bins, n_active, joint, x_span, explained,
    explained_total, resid_rel, intercept_rel, bg_align, clamp_frac,
    anchor_rel, vif_target, vif_max, vif_partner, deg_change, peak_curr,
    peak_new, peak_new_rel.
    """
    raw = np.asarray(raw, dtype=np.float64)
    n_fluor = spectra.shape[0]
    active = list(range(n_fluor)) if active is None else list(active)
    active = [target] + [a for a in active if a != target]
    n_floor = len(active) + 3
    if multivariate and n_levels < n_floor:
        raise ValueError(f'The joint signature fit needs more abundance bins than regressors; raise the '
                         f'bin count to at least {n_floor}.')
    n = raw.shape[0]
    if n < min_events:
        return None
    x = np.asarray(abundance, dtype=np.float64)[:, active]
    x_target = x[:, 0]
    n_levels_use = max(3, min(int(n_levels), n // int(min_bin_events)))
    fit_joint = multivariate and n_levels_use >= n_floor

    brk = np.unique(np.quantile(x_target, np.linspace(0.0, 1.0, n_levels_use + 1)))
    if brk.size < 3:
        return None
    bin_of = _cut_bins(x_target, brk) - 1
    counts = np.bincount(bin_of, minlength=brk.size - 1)
    present = np.flatnonzero(counts)
    if present.size < 3:
        return None
    counts_p = counts[present][:, None]
    y_bin = np.column_stack([np.bincount(bin_of, weights=raw[:, d], minlength=brk.size - 1)
                             for d in range(raw.shape[1])])[present] / counts_p
    x_bin = np.column_stack([np.bincount(bin_of, weights=x[:, a], minlength=brk.size - 1)
                             for a in range(x.shape[1])])[present] / counts_p
    xt = x_bin[:, 0]
    if xt.max() - xt.min() <= 0:
        return None
    nuisance = active[1:]
    n_bins = present.size

    if fit_joint and nuisance and n_bins >= x_bin.shape[1] + 2:
        design = np.column_stack([np.ones(n_bins), x_bin]) if intercept else x_bin
        scale = np.sqrt(np.mean(design ** 2, axis=0))
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        zd = design / scale
        penalty = np.diag(np.full(zd.shape[1], ridge))
        if intercept:
            penalty[0, 0] = 0.0
        try:
            coef = np.linalg.solve(zd.T @ zd + penalty, zd.T @ y_bin)
        except np.linalg.LinAlgError:
            return None
        coef = coef / scale[:, None]
        offset = coef[0].copy() if intercept else np.zeros(raw.shape[1])
        signature_raw = coef[1 if intercept else 0].copy()
        fitted = design @ coef
        y_res = y_bin
        fitted_total = fitted
    else:
        nuisance_fitted = x_bin[:, 1:] @ spectra[nuisance] if nuisance else np.zeros_like(y_bin)
        y_res = y_bin - nuisance_fitted
        if intercept:
            d2 = np.column_stack([np.ones(n_bins), xt])
            coef = np.linalg.lstsq(d2, y_res, rcond=None)[0]
            offset = coef[0].copy()
            signature_raw = coef[1].copy()
            fitted = d2 @ coef
        else:
            den = float(xt @ xt)
            if not np.isfinite(den) or den <= 0:
                return None
            offset = np.zeros(raw.shape[1])
            signature_raw = (xt @ y_res) / den
            fitted = np.outer(xt, signature_raw)
        fitted_total = fitted + nuisance_fitted

    offset[~np.isfinite(offset)] = 0.0
    signature_raw[~np.isfinite(signature_raw)] = 0.0
    resid_rel = np.linalg.norm(y_res - fitted) / max(np.linalg.norm(y_res), _EPS)
    top_signal = max(np.linalg.norm(xt.max() * signature_raw), _EPS)
    intercept_rel = np.linalg.norm(offset) / top_signal
    clamped = np.maximum(signature_raw, 0.0)
    clamp_frac = np.sum(np.maximum(-signature_raw, 0.0)) / max(np.sum(np.abs(signature_raw)), _EPS)
    if clamped.max() <= 0:
        return None
    signature = clamped / clamped.max()

    top = int(np.argmax(xt))
    top_norm = np.linalg.norm(y_bin[top])
    explained = np.linalg.norm(xt[top] * signature_raw) / top_norm if top_norm > 0 else 0.0
    for_total = fitted if fit_joint else fitted_total
    explained_total = np.linalg.norm(for_total[top]) / top_norm if top_norm > 0 else 0.0

    anchor_rel = np.nan
    if background_raw is not None and len(background_raw) > 0:
        anchor_rel = np.linalg.norm(np.asarray(background_raw).mean(axis=0)) / top_signal

    vif = _vif(x)
    vif_partner = -1
    if x.shape[1] > 1:
        with np.errstate(invalid='ignore', divide='ignore'):
            r = np.corrcoef(x, rowvar=False)[0, 1:]
        r = np.where(np.isfinite(r), r, 0.0)
        vif_partner = active[1 + int(np.argmax(np.abs(r)))]

    bg_align = np.nan
    if intercept:
        on, sn = np.linalg.norm(offset), np.linalg.norm(signature_raw)
        if on > 0 and sn > 0:
            bg_align = float(offset @ signature_raw / (on * sn))

    current = spectra[target]
    peak_curr = int(np.argmax(current))
    peak_new = int(np.argmax(signature))
    names = detector_names if detector_names is not None else [str(i) for i in range(raw.shape[1])]
    stats = {
        'n_events': int(n), 'n_bins': int(n_bins), 'n_active': len(active), 'joint': bool(fit_joint),
        'x_span': float(xt.max() - xt.min()), 'explained': float(explained),
        'explained_total': float(explained_total), 'resid_rel': float(resid_rel),
        'intercept_rel': float(intercept_rel), 'bg_align': float(bg_align), 'clamp_frac': float(clamp_frac),
        'anchor_rel': float(anchor_rel), 'vif_target': float(vif[0]), 'vif_max': float(vif.max()),
        'vif_partner': int(vif_partner), 'deg_change': float(row_angle(current, signature)[0]),
        'peak_curr': names[peak_curr], 'peak_new': names[peak_new],
        'peak_new_rel': float(current[peak_new] / max(current.max(), _EPS)),
    }
    return {'signature': signature, 'signature_raw': signature_raw, 'stats': stats}


def leakage(raw: np.ndarray, design: np.ndarray, target_row: int, panel_rows: list,
            rng, max_events: int = 20_000) -> float:
    """
    Median off-target abundance in a fluorophore's own population as a
    fraction of its own median (``.fix.leakage()``); NaN when the own
    median is not positive.
    """
    if len(raw) > max_events:
        raw = raw[rng.choice(len(raw), max_events, replace=False)]
    unmixed = unmix_ols(raw, design)
    on = np.median(unmixed[:, target_row])
    if not np.isfinite(on) or on <= 0:
        return np.nan
    off = [r for r in panel_rows if r != target_row]
    if not off:
        return np.nan
    return float(np.sum(np.abs(np.median(unmixed[:, off], axis=0))) / on)


def residual_change(raw: np.ndarray, design: np.ndarray, design_trial: np.ndarray, rng,
                    max_events: int = 20_000) -> float:
    """
    Relative change of the OLS residual sum of squares of ``raw`` when the
    design changes from ``design`` to ``design_trial`` (same events for
    both); NaN when there are no events or no residual. A change that lies
    inside the span of the design leaves the residual unchanged, so this
    only sees the out-of-span part of a signature change.
    """
    if len(raw) == 0:
        return np.nan
    if len(raw) > max_events:
        raw = raw[rng.choice(len(raw), max_events, replace=False)]

    def rss(d):
        r = raw - unmix_ols(raw, d) @ d
        return float(np.einsum('ij,ij->', r, r))

    before = rss(design)
    if not before > 0:
        return np.nan
    return rss(design_trial) / before - 1.0


def stratified_sample(abundance: np.ndarray, threshold_matrix: np.ndarray, n_total: int,
                      background_frac: float, min_stratum: int, rng) -> np.ndarray:
    """
    Event subsample stratified by positivity (``.fix.stratified.sample()``):
    every fluorophore's positive population has its own quota, with a floor
    that keeps a small population whole, and events positive for nothing
    are capped at ``background_frac`` of the total. Co-positive events
    belong to every stratum they are positive in.
    """
    n = abundance.shape[0]
    if n <= n_total:
        return np.arange(n)
    positive = (abundance - threshold_matrix) > 0
    background = np.flatnonzero(~positive.any(axis=1))
    pos_idx = [np.flatnonzero(positive[:, j]) for j in range(abundance.shape[1])]
    n_bg = min(background.size, int(round(background_frac * n_total)))
    budget = n_total - n_bg
    sizes = np.array([p.size for p in pos_idx])
    floor_n = np.minimum(sizes, int(min_stratum))
    if floor_n.sum() > budget:
        quota = np.minimum(sizes, np.floor(floor_n * budget / max(floor_n.sum(), 1)).astype(np.int64))
    else:
        spare = np.maximum(sizes - floor_n, 0)
        leftover = budget - floor_n.sum()
        extra = leftover * spare / spare.sum() if spare.sum() > 0 else np.zeros(len(sizes))
        quota = np.minimum(sizes, floor_n + np.round(extra).astype(np.int64))
    chosen = [p if p.size <= k else rng.choice(p, int(k), replace=False) for p, k in zip(pos_idx, quota)]
    chosen_pos = np.unique(np.concatenate(chosen)) if chosen else np.empty(0, dtype=np.int64)
    chosen_bg = background if n_bg >= background.size else rng.choice(background, n_bg, replace=False)
    return np.concatenate([chosen_pos, chosen_bg]).astype(np.int64)


# ---------------------------------------------------------------------------
# fix.my.unmix
# ---------------------------------------------------------------------------

@dataclass
class FixResult:
    fluor_names: list
    detector_names: list
    spectra_start: np.ndarray
    spectra: np.ndarray
    spectra_backsolved: np.ndarray
    spillover: np.ndarray
    compensation: np.ndarray
    trust: np.ndarray
    coefficient_log: list
    source_log: list
    signature_log: list
    convergence_log: list
    spillover_history: list | None
    thresholds: np.ndarray
    neg_thresholds: np.ndarray
    dominant_counts: np.ndarray
    af_assignment: list | None
    prior_fallback: list | None
    recovery: list | None
    n_events: int
    n_events_fit: int
    settings: dict = field(default_factory=dict)
    background: Background | None = field(default=None, repr=False)
    smooth_log: list | None = None          # smooth_fix: one entry per round

    @property
    def accepted(self) -> np.ndarray:
        out = np.zeros(len(self.fluor_names), dtype=bool)
        for row in self.signature_log:
            out[row['index']] = row['accepted']
        return out


def _thresholds(unstained_comp: np.ndarray, s: FixSettings) -> tuple[np.ndarray, np.ndarray]:
    """
    Flat upper and lower boundaries from the unstained events. The lower one
    has its own quantile: a share of a correctly unmixed population as large
    as 1 - its quantile lies below it, so it must stay well under
    ``max_hypernegative_frac`` or the hypernegative check takes the trust of
    correct coefficients.
    """
    upper = s.unstained_margin * np.quantile(unstained_comp, s.unstained_threshold, axis=0)
    lower = s.unstained_margin * np.quantile(unstained_comp, 1.0 - s.lower_unstained_threshold, axis=0)
    return upper, lower


def _reject_reason(st: dict, threshold: float, s: FixSettings) -> str | None:
    """First acceptance gate a candidate fails, cheapest and most decisive first."""
    if st['x_span'] <= s.min_span * abs(threshold):
        return 'span'
    et = st['explained_total']
    if not np.isfinite(et) or et < s.min_explained or et > s.max_explained:
        return 'explained'
    if st['resid_rel'] > s.max_resid:
        return 'fit'
    if st['intercept_rel'] > s.max_intercept:
        return 'offset'
    if np.isfinite(st['bg_align']) and st['bg_align'] < s.min_bg_align:
        return 'background.confound'
    if st['clamp_frac'] > s.max_clamp_frac:
        return 'negative.mass'
    if st['deg_change'] > s.max_angle:
        return 'angle'
    if np.isfinite(st['anchor_rel']) and st['anchor_rel'] > s.max_anchor:
        return 'background'
    if st['vif_target'] > s.max_vif:
        return 'collinear'
    if st['peak_new'] != st['peak_curr'] and st['peak_new_rel'] < s.peak_shift_min_rel:
        return 'peak.shift'
    return None


def _check_settings(s: FixSettings) -> None:
    if s.phase_two_rows not in PHASE_TWO_ROWS:
        raise ValueError(f'Unknown phase-two row selection "{s.phase_two_rows}".')
    if s.phase_two_adopt not in PHASE_TWO_ADOPT:
        raise ValueError(f'Unknown phase-two adoption "{s.phase_two_adopt}".')
    if s.phase_two_fallback not in PHASE_TWO_FALLBACK:
        raise ValueError(f'Unknown phase-two fallback "{s.phase_two_fallback}".')
    if s.stop_rule not in STOP_RULES:
        raise ValueError(f'Unknown stop rule "{s.stop_rule}".')
    if s.stall_rule not in STALL_RULES:
        raise ValueError(f'Unknown stall rule "{s.stall_rule}".')
    if s.stop_rule == 'settled' and (not 0.0 < s.stop_trust <= 1.0 or int(s.stop_confirm) < 1):
        raise ValueError('The settled stop rule needs 0 < stop_trust <= 1 and stop_confirm >= 1.')
    if s.disagreement_reference not in DISAGREEMENT_REFERENCES:
        raise ValueError(f'Unknown disagreement reference "{s.disagreement_reference}".')
    if not 0.0 < s.envelope_quantile < s.envelope_scale_quantile < 1.0:
        raise ValueError('The envelope scale quantile must lie between the envelope quantile and 1.')
    if int(s.smooth_rounds) < 1 or not 0.0 < s.smooth_fraction <= 1.0:
        raise ValueError('Smooth updates need at least one round and 0 < fraction <= 1.')
    if s.hypernegative_scope not in ('pair', 'exclusive'):
        raise ValueError(f'Unknown hypernegative scope "{s.hypernegative_scope}".')
    if s.hypernegative_rule not in HYPERNEGATIVE_RULES:
        raise ValueError(f'Unknown hypernegative rule "{s.hypernegative_rule}".')
    if s.residual_gate not in RESIDUAL_GATES:
        raise ValueError(f'Unknown residual gate "{s.residual_gate}".')
    if not (0 <= s.source_events_low < s.source_events_high):
        raise ValueError('The source event range needs 0 <= low < high.')
    if not (0.0 < s.negative_tail < 0.5):
        raise ValueError('The negative tail must lie between 0 and 0.5.')
    if not (0.5 < s.unstained_threshold < 1.0 and 0.5 < s.lower_unstained_threshold < 1.0):
        raise ValueError('The unstained quantiles must lie between 0.5 and 1.')
    if s.plausibility_ratio < 0 or s.plausibility_floor < 0 or not 0.0 < s.plausibility_reject <= 1.0:
        raise ValueError('Plausibility bounds need ratio >= 0, floor >= 0 and 0 < reject share <= 1.')
    if not 0.0 < s.shared_cosine < 1.0 or s.shared_min_slope < 0 or s.shared_power < 0 or \
            not 0.0 <= s.shared_alignment <= 1.0:
        raise ValueError('The shared-direction check needs 0 < cosine < 1, slope >= 0 and power >= 0.')
    if s.af_penalty_width <= 0:
        raise ValueError('The AF penalty needs a positive width.')


def source_evidence(u: np.ndarray, u0: np.ndarray, thresholds: np.ndarray, spread: np.ndarray | None,
                    s: FixSettings, rng, n_events: int | None = None) -> dict:
    """
    Clearly positive events of every source fluorophore and the weight they
    earn.

    An event is clearly positive for a source when it is above both its
    spread-scaled unstained boundary and the upper bound of the stained
    sample's own negative population in that channel (``negative_bound``),
    and (with ``source_dominant``) the source is its dominant fluorophore.
    The unstained sample, compensated the same way and judged by the same
    rule, gives the rate of spurious clear positives (AF outliers, noise
    tails); ``excess`` is the count in ``u`` minus that rate times
    ``n_events`` (default ``len(u)``; the whole sample when ``u`` is a
    stratified subsample, which keeps small populations whole). The weight
    rises smoothly from 0 at ``source_events_low`` excess events to 1 at
    ``source_events_high``.

    Returns ``t_clear`` (n, F) per-event boundaries, ``bound`` (F,), per
    source ``n_clear``, ``baseline``, ``excess`` and ``weight``, and each
    event's ``dominant`` fluorophore (None without ``source_dominant``).
    ``rng`` is one generator, or one per fluorophore.
    """
    n_fluor = u.shape[1]
    n_events = len(u) if n_events is None else int(n_events)
    rngs = rng if isinstance(rng, (list, tuple)) else [rng] * n_fluor
    tm = spread_thresholds(u, thresholds, spread, s.spread_kappa)
    tm0 = spread_thresholds(u0, thresholds, spread, s.spread_kappa)
    dyn = _dynamic_range(u, thresholds)
    bound = np.array([negative_bound(u[:, f], tm[:, f], s.negative_tail, rngs[f], s.negative_bound_events)
                      for f in range(n_fluor)])
    t_clear = np.maximum(tm, bound[None, :])
    clear = u > t_clear
    clear0 = u0 > np.maximum(tm0, bound[None, :])
    dominant = None
    if s.source_dominant:
        own = np.arange(n_fluor)[None, :]
        dominant = _dominance(u, thresholds, tm, dyn)
        clear &= dominant[:, None] == own
        clear0 &= _dominance(u0, thresholds, tm0, dyn)[:, None] == own
    n_clear = clear.sum(axis=0)
    baseline = clear0.mean(axis=0) * n_events
    excess = n_clear - baseline
    return {'t_clear': t_clear, 'bound': bound, 'n_clear': n_clear, 'baseline': baseline, 'excess': excess,
            'weight': smoothstep(excess, s.source_events_low, s.source_events_high), 'dominant': dominant}


def fix_my_unmix(spectra: np.ndarray, fluor_names: list, unstained: dict, stained: dict,
                 background: Background, settings: FixSettings | None = None, spread: np.ndarray | None = None,
                 variant_deltas: list | None = None, true_spectra: np.ndarray | None = None,
                 detector_names: list | None = None, progress=None,
                 reference_spectra: np.ndarray | None = None) -> FixResult:
    """
    Correct in-span reference-spectrum error from fully stained events.

    spectra        : (F, D) reference spectra; a row named ``AF`` is left
                     out (the per-cell AF library replaces it).
    unstained      : the unstained sample's events unmixed by
                     ``project_events`` over the non-AF rows (thresholds).
    stained        : the fully stained events, likewise (``concatenate``
                     for several samples).
    background     : the AF background of both (``background_for``).
    spread         : (F, F) Spillover Spreading Matrix over ``fluor_names``
                     (source x target), or None for flat boundaries.
    variant_deltas : per row of ``spectra``, a (variants x D) delta matrix
                     or None; enables the variant leakage prior.
    true_spectra   : (F, D) ground truth, purely diagnostic (``recovery``).
    progress       : optional ``progress(text)`` callback.
    reference_spectra : (F, D) rows the plausibility bounds are measured
                     from (default ``spectra``; ``smooth_fix`` passes the
                     spectra its rounds started from).

    See the module docstring for the method. Returns a FixResult whose
    ``spectra`` keep every row that failed a gate.
    """
    s = settings or FixSettings()
    _check_settings(s)
    say = progress or (lambda _t: None)
    rng = np.random.default_rng(s.seed)

    def streams(purpose: int, n: int):
        """The shared generator, or with ``seed_streams`` one per index keyed by (seed, purpose, index)."""
        if not s.seed_streams:
            return rng
        return [np.random.default_rng([s.seed, purpose, i]) for i in range(n)]

    spectra = np.asarray(spectra, dtype=np.float64)
    names = list(fluor_names)
    if len(names) != spectra.shape[0]:
        raise ValueError('fluor_names must name every row of spectra.')
    fl = [i for i, nm in enumerate(names) if nm != AF_NAME]
    n_fluor = len(fl)
    if n_fluor < 2:
        raise ValueError('At least two panel fluorophores are required.')
    fnames = [names[i] for i in fl]
    names_idx = {n: i for i, n in enumerate(fnames)}
    spectra_fluor = spectra[fl]
    fixed_names = [str(n) for n in (s.fixed_fluorophores or ())]
    unknown = [n for n in fixed_names if n not in fnames]
    if unknown:
        logger.warning('Fix My Unmix: fluorophores to keep fixed not in the panel, ignored: %s', ', '.join(unknown))
    fixed = np.array([n in fixed_names for n in fnames], dtype=bool)
    excluded = fixed if s.exclude_fixed_targets else np.zeros(n_fluor, dtype=bool)
    kept = np.flatnonzero(~excluded)
    for fit in (stained, unstained):
        if fit['residual'].shape[1] != spectra.shape[1]:
            raise ValueError('The spectra and the spectral channels of the data do not match.')
        if fit['abundance'].shape[1] != n_fluor:
            raise ValueError('The events were not unmixed with the panel\'s fluorophores.')
    if len(stained['abundance']) == 0 or len(unstained['abundance']) < 3:
        raise ValueError('Fix My Unmix needs fully stained and unstained events.')
    if s.multivariate and s.update_spectra and s.n_levels < n_fluor + 3:
        raise ValueError(f'The joint signature fit needs at least {n_fluor + 3} abundance bins for '
                         f'{n_fluor} fluorophores; raise the bin count.')
    det_names = detector_names or [f'D{i}' for i in range(spectra.shape[1])]
    spread_fl = None
    if spread is not None:
        spread = np.asarray(spread, dtype=np.float64)
        spread_fl = align_spread(spread[np.ix_(fl, fl)] if spread.shape[0] == len(names) else spread, n_fluor)

    bg = background
    if bg.basis.shape[1] != spectra.shape[1]:
        raise ValueError('The AF background and the spectra have different detectors.')
    design = np.vstack([bg.basis, spectra_fluor])
    if design.shape[0] > design.shape[1]:
        raise ValueError('The AF mean plus the panel exceeds the detector count.')

    # rows the plausibility bounds, the shared-direction cosines and the dyes' own AF likeness are measured from
    reference = spectra_fluor if reference_spectra is None else np.asarray(reference_spectra, dtype=np.float64)
    if reference.shape[0] == len(names) and len(names) != n_fluor:
        reference = reference[fl]
    if reference.shape != spectra_fluor.shape:
        raise ValueError('reference_spectra must match the panel spectra.')
    reference = reference / np.maximum(reference.max(axis=1, keepdims=True), _EPS)
    plausible = s.plausibility_ratio > 0
    allow = plausible_change(reference, s.plausibility_ratio, s.plausibility_floor) if plausible else None
    ref_cosine = cosine_similarity(reference) if s.shared_direction else None
    dye_af = None
    if s.af_penalty:
        dye_af = np.array([af_likeness(reference[i], bg.af_spectra) for i in range(n_fluor)])

    prior = None
    if s.leakage_prior and variant_deltas is not None:
        prior = variant_leakage_prior(spectra_fluor, [variant_deltas[i] for i in fl], bg.basis, s.span_fraction)

    def joint_lasso(u, tm, thresholds, neg_var, warm=None):
        """``glasso_matrix`` over the channels that are not excluded (zero rows and columns for those)."""
        rngs = streams(_STREAM_LASSO, n_fluor)
        if not excluded.any():
            return glasso_matrix(u, tm, thresholds, neg_var, rngs, s, warm)
        rngs = [rngs[k] for k in kept] if isinstance(rngs, list) else rngs
        out = np.zeros((n_fluor, n_fluor))
        if kept.size < 2:
            return out
        out[np.ix_(kept, kept)] = glasso_matrix(u[:, kept], tm[:, kept], thresholds[kept], neg_var[kept], rngs, s,
                                                None if warm is None else warm[np.ix_(kept, kept)])
        return out

    stained_fit = {k: stained[k] for k in ('abundance', 'residual', 'af_idx', 'af_scale')}
    unstained_fit = {k: unstained[k] for k in ('abundance', 'residual', 'af_idx', 'af_scale')}
    stained_full = stained_fit
    n_events = len(stained_fit['abundance'])

    if s.downsample and s.downsample < n_events:
        thr_rough = s.unstained_margin * np.quantile(unstained_fit['abundance'], s.unstained_threshold, axis=0)
        tm_rough = spread_thresholds(stained_fit['abundance'], thr_rough, spread_fl, s.spread_kappa)
        keep = stratified_sample(stained_fit['abundance'], tm_rough, int(s.downsample),
                                 s.downsample_background_frac, s.downsample_min_stratum,
                                 rng if not s.seed_streams else np.random.default_rng([s.seed, _STREAM_DOWNSAMPLE]))
        stained_fit = {k: v[keep] for k, v in stained_fit.items()}

    # ---- phase one --------------------------------------------------------
    eye = np.eye(n_fluor)
    off_diag = ~eye.astype(bool)
    spill = eye.copy()
    trust = eye.copy()
    slope_warm = np.zeros((n_fluor, n_fluor))
    convergence_log = []
    history = [] if s.keep_history else None
    delta_history = [np.nan, np.nan, np.nan]
    delta_best, since_best = np.inf, 0
    below_run = 0                           # iterations in a row below the threshold
    pair_log = []
    source_log = []
    stop_reason = 'iteration limit'
    for it in range(1, int(s.max_iter) + 1):
        try:
            comp = np.linalg.inv(spill)
        except np.linalg.LinAlgError:
            stop_reason = 'singular spillover'
            logger.warning('Fix My Unmix: spillover matrix became singular; stopping.')
            break
        say(f'Phase one: iteration {it}')
        u = stained_fit['abundance'] @ comp
        u0 = unstained_fit['abundance'] @ comp
        thresholds, neg_thresholds = _thresholds(u0, s)
        neg_var = _mad(u0) ** 2
        tm = spread_thresholds(u, thresholds, spread_fl, s.spread_kappa)
        ntm = spread_thresholds(u, neg_thresholds, spread_fl, s.spread_kappa, side='lower')
        evidence = source_evidence(u, u0, thresholds, spread_fl, s, streams(_STREAM_EVIDENCE, n_fluor), n_events)
        t_clear = evidence['t_clear']
        dominant = evidence['dominant']

        marker = eye.copy()
        trust = eye.copy()
        pair_log = []
        source_log = []
        rows_of = {}
        candidates = []
        joint = None
        anchored = np.zeros((n_fluor, n_fluor))
        if s.glasso_check and not s.glasso_warm:
            say(f'Phase one: iteration {it}, joint lasso')
            joint = joint_lasso(u, tm, thresholds, neg_var)
        slope_rngs = streams(_STREAM_SLOPES, n_fluor)
        u_t = np.ascontiguousarray(u.T)                     # channels x events
        tm_t = np.ascontiguousarray(tm.T)
        ntm_t = np.ascontiguousarray(ntm.T)
        for src in range(n_fluor):
            chans = [c for c in range(n_fluor) if c != src and not excluded[c]]
            w_src = 0.0 if fixed[src] else float(evidence['weight'][src])
            source_log.append({'source': fnames[src], 'negative_bound': float(evidence['bound'][src]),
                               'clear_events': int(evidence['n_clear'][src]),
                               'baseline_events': float(evidence['baseline'][src]),
                               'excess_events': float(evidence['excess'][src]), 'source_weight': w_src,
                               'af_likeness': np.nan, 'af_penalty': np.nan, 'plausible_project': 0.0, 'af_gated': False,
                               'fixed': bool(fixed[src])})
            est = None
            if w_src > 0:
                spread_var = np.zeros(len(chans)) if spread_fl is None else spread_fl[src, chans]
                est = envelope_slopes(u[:, src], u_t[chans].T, t_clear[:, src], tm_t[chans].T, spread_var,
                                      neg_var[chans], None if dominant is None else dominant == src,
                                      slope_warm[src, chans],
                                      slope_rngs[src] if s.seed_streams else slope_rngs, s, ntm_t[chans].T)
            rows_of[src] = []
            for ci, ch in enumerate(chans):
                w = 0.0
                row = {'source': fnames[src], 'channel': fnames[ch], 'slope_anchored': np.nan, 'slope': np.nan,
                       'slope_median': np.nan, 'se': np.nan, 'se_fit': np.nan, 'disagreement': np.nan,
                       'coverage': np.nan, 'span': np.nan,
                       'slope_glasso': np.nan if joint is None else float(joint[src, ch]),
                       'glasso_agreement': np.nan, 'source_weight': w_src, 'hypernegative_base': np.nan,
                       'hypernegative_after': np.nan, 'hypernegative_new': np.nan,
                       'shared_sources': 1, 'shared_factor': 1.0, 'trust': 0.0}
                slope_use = np.nan
                if est is not None:
                    slope_use = est['slope_anchored'][ci]
                    if np.isfinite(slope_use):
                        slope_warm[src, ch] = slope_use
                        if abs(slope_use) <= s.max_coefficient:
                            anchored[src, ch] = slope_use
                    row.update(slope_anchored=slope_use, slope=est['slope'][ci], slope_median=est['slope_alt'][ci],
                               se=est['se'][ci], se_fit=est['se_fit'][ci], disagreement=est['disagreement'][ci],
                               coverage=est['coverage'][ci], span=est['span'][ci])
                if (est is not None and np.isfinite(slope_use)
                        and est['coverage'][ci] >= s.min_negative_frac
                        and abs(slope_use) <= s.max_coefficient
                        and abs(slope_use) * est['span'][ci] >= s.min_rise * est['noise'][ci]
                        and est['span'][ci] > s.min_span * abs(thresholds[src])):
                    prior_var = slope_use ** 2 if prior is None else prior['variance'][src, ch]
                    if not np.isfinite(prior_var) or prior_var <= 0:
                        prior_var = slope_use ** 2
                    se = est['se'][ci]
                    if not (np.isfinite(se) and se > 0):
                        se = est['se_fit'][ci]
                    # a coefficient whose uncertainty cannot be judged earns no trust
                    w = prior_var / (prior_var + se ** 2) if np.isfinite(se) and se > 0 else 0.0
                    w *= w_src
                    dis = est['disagreement'][ci]
                    if np.isfinite(dis) and dis > s.max_disagreement:
                        w *= s.max_disagreement / dis
                    if joint is not None:
                        g = float(joint[src, ch])
                        agree = 0.0 if g == 0.0 else max(0.0, 1.0 - abs(slope_use - g) / (abs(slope_use) + abs(g)))
                        row['glasso_agreement'] = agree
                        w *= agree
                    marker[src, ch] = slope_use
                    candidates.append((src, ch, len(pair_log)))
                trust[src, ch] = w
                row['trust'] = w
                rows_of[src].append(len(pair_log))
                pair_log.append(row)

        if s.glasso_check and s.glasso_warm:
            say(f'Phase one: iteration {it}, joint lasso')
            joint = joint_lasso(u, tm, thresholds, neg_var, anchored)
            for src, ch, r in candidates:
                g = float(joint[src, ch])
                agree = 0.0 if g == 0.0 else max(0.0, 1.0 - abs(marker[src, ch] - g) / (abs(marker[src, ch]) + abs(g)))
                trust[src, ch] *= agree
                pair_log[r].update(slope_glasso=g, glasso_agreement=agree, trust=float(trust[src, ch]))

        # A negative correlation between a source and a corrected channel has
        # no biological explanation, so trust falls linearly from
        # min_hypernegative_frac to zero at max_hypernegative_frac of the
        # source's clearly positive population left below the channel's
        # negative boundary ('after'), or of that population the coefficient
        # newly pushes below it ('new': a coefficient that only shrinks an
        # existing hypernegative population is not penalised for what
        # remains). 'exclusive' leaves out events whose value in the channel
        # the other candidates into it would move by more than the channel's
        # flat negative boundary, so a second error into the same channel
        # from a co-expressed source does not block this one.
        upos = np.maximum(u, 0.0)
        frac_range = max(s.max_hypernegative_frac - s.min_hypernegative_frac, _EPS)
        for src, ch, r in candidates:
            source_pos = u[:, src] > t_clear[:, src]
            if s.hypernegative_dominant and dominant is not None:
                # an event another fluorophore dominates reads positive for the source only through
                # that fluorophore's error, and removing the source's slope pushes it below zero
                source_pos &= dominant == src
            if s.hypernegative_scope == 'exclusive':
                others = [sr for sr, c, _r in candidates if c == ch and sr != src]
                if others:
                    shift = upos[:, others] @ np.abs(trust[others, ch] * marker[others, ch])
                    source_pos &= shift <= abs(neg_thresholds[ch])
            if source_pos.sum() < s.min_hypernegative_events:
                continue
            boundary = ntm[source_pos, ch]
            values = u[source_pos, ch]
            base = float(np.mean(values < boundary))
            after = float(np.mean(values - marker[src, ch] * u[source_pos, src] < boundary))
            judged = after if s.hypernegative_rule == 'after' else max(after - base, 0.0)
            trust[src, ch] *= 1.0 - min(max(judged - s.min_hypernegative_frac, 0.0) / frac_range, 1.0)
            pair_log[r].update(hypernegative_base=base, hypernegative_after=after,
                               hypernegative_new=max(after - base, 0.0), trust=float(trust[src, ch]))

        slope_error = marker - eye
        # the error each step applies (the shared-direction check moves it)
        step_error = slope_error
        if s.shared_direction:
            # the whole coefficient (what the spillover already holds plus the error left) is judged and
            # shrunk, so the shrunk coefficient is the fixed point; a damped step would only slow the approach
            # to the same fixed point
            whole = (spill - eye) + slope_error
            aligned = None
            if s.shared_alignment > 0:
                # co-expression moves a source's row toward the co-expressed dye's spectrum; a real row error
                # reaches several channels at once, each coefficient only one part of its direction
                change = (whole * off_diag) @ spectra_fluor
                aligned = cosine_between(change, spectra_fluor) >= s.shared_alignment
            shared = shared_direction_counts(whole, trust, candidates, ref_cosine, s, aligned)
            factor = np.where(shared > 1, shared ** -s.shared_power, 1.0)
            step_error = np.where(factor < 1.0, factor * whole - (spill - eye), slope_error)
            for src, ch, r in candidates:
                if shared[src, ch] > 1:
                    pair_log[r].update(shared_sources=int(shared[src, ch]), shared_factor=float(factor[src, ch]))

        # the change each source's back-solved row would take this step
        row_change = ((trust * step_error) * off_diag) @ (spill @ spectra_fluor)
        for src in range(n_fluor):
            like = af_likeness(row_change[src], bg.af_spectra)
            source_log[src]['af_likeness'] = like
            if s.af_penalty and like > 0:
                # an AF-like dye may move in AF-like directions; an AF-unlike one only as far as the margin
                low = dye_af[src] + s.af_penalty_margin
                keep = 1.0 - float(smoothstep(like, low, low + s.af_penalty_width))
                source_log[src]['af_penalty'] = keep
                if keep < 1.0:
                    trust[src, off_diag[src]] *= keep
                    for r in rows_of[src]:
                        pair_log[r]['trust'] = float(trust[src, names_idx[pair_log[r]['channel']]])
            if s.af_gate and like > s.af_gate_max:
                trust[src, off_diag[src]] = 0.0
                source_log[src]['af_gated'] = True
                for r in rows_of[src]:
                    pair_log[r]['trust'] = 0.0

        # the next fit starts from the slope this step leaves behind
        slope_warm *= 1.0 - trust

        # 'all' judges every fitted error, trusted or not; 'applied' the step this iteration takes;
        # 'trusted' the whole error of every coefficient with any trust (a partly trusted step is small
        # while its error is not); 'settled' the whole error of every coefficient with at least
        # stop_trust trust, so a slope too weakly trusted ever to finish does not hold phase one open
        if s.stop_rule == 'all':
            judged_error = slope_error
        elif s.stop_rule == 'applied':
            judged_error = trust * slope_error
        elif s.stop_rule == 'trusted':
            judged_error = np.where(trust > 0, slope_error, 0.0)
        else:
            judged_error = np.where(trust >= s.stop_trust, slope_error, 0.0)
        off = np.abs(judged_error[off_diag])
        delta = float(np.std(judged_error, ddof=1))
        delta_q = float(np.quantile(off, s.convergence_quantile))
        delta_max = float(off.max())
        n_fitted = int((trust > 0).sum() - n_fluor)
        convergence_log.append({'iter': it, 'delta': delta, 'delta_quantile': delta_q, 'delta_max': delta_max,
                                'n_fitted': n_fitted})
        logger.info('Fix My Unmix: iter %d, delta %.5f, q%.0f %.5f, max %.5f, %d coefficient(s) fitted',
                    it, delta, 100 * s.convergence_quantile, delta_q, delta_max, n_fitted)

        # trust damps the step only, so the fixed point stays the identity
        nxt = spill + (trust * step_error) @ spill
        decay = 1.0 - s.coefficient_decay * (1.0 - trust)
        np.fill_diagonal(decay, 1.0)
        nxt = nxt * decay
        if history is not None:
            history.append({'marker_spillover': marker.copy(), 'trust': trust.copy(), 'spillover_next': nxt.copy(),
                            'coefficient_log': [dict(r) for r in pair_log],
                            'source_log': [dict(r) for r in source_log]})
        diag = np.diag(nxt)
        if not np.all(np.isfinite(diag)) or np.any(diag <= 0):
            stop_reason = 'diagonal left the positive range'
            logger.warning('Fix My Unmix: spillover diagonal left the positive range; stopping.')
            break
        nxt = nxt / diag[:, None]
        if plausible:
            # the part of a source's back-solved row outside its box is taken back out of its coefficients
            # (to the bound: resetting detectors here, mid-fit, holds back real corrections on their way)
            # (fixed rows are not adjusted; excluded channels take no coefficient)
            nxt, moved_back = _plausible_project(nxt, spectra_fluor, reference, allow, rows=~fixed, basis=~excluded)
            for src in np.flatnonzero(moved_back > 0):
                source_log[src]['plausible_project'] = float(moved_back[src])
        spill = nxt
        below_run = below_run + 1 if delta_q < s.convergence_threshold else 0
        if below_run >= (int(s.stop_confirm) if s.stop_rule == 'settled' else 1):
            stop_reason = 'converged'
            break
        delta_history = delta_history[1:] + [delta]
        if delta < delta_best:
            delta_best, since_best = delta, 0
        else:
            since_best += 1
        if s.stall_rule == 'patience':
            stalled = since_best >= int(s.stall_patience)
        else:
            stalled = bool(np.all(np.isfinite(delta_history)) and np.mean(np.diff(delta_history)) >= 0)
        if stalled:
            stop_reason = 'stalled'
            break

    comp = np.linalg.inv(spill)
    u0_final = unstained_fit['abundance'] @ comp
    thresholds_final, neg_thresholds_final = _thresholds(u0_final, s)
    u_full = stained_full['abundance'] @ comp
    tm_full = spread_thresholds(u_full, thresholds_final, spread_fl, s.spread_kappa)
    dom_full = _dominance(u_full, thresholds_final, tm_full)
    dominant_counts = np.bincount(dom_full + 1, minlength=n_fluor + 1)

    backsolved = spill @ spectra_fluor
    row_max = backsolved.max(axis=1)
    row_max = np.where(np.isfinite(row_max) & (row_max > 0), row_max, 1.0)
    backsolved = backsolved / row_max[:, None]

    # ---- phase two --------------------------------------------------------
    spectra_new = spectra_fluor.copy()
    signature_log = []
    if s.update_spectra:
        u = stained_fit['abundance'] @ comp
        residual = stained_fit['residual']
        tm = spread_thresholds(u, thresholds_final, spread_fl, s.spread_kappa)
        dominant = _dominance(u, thresholds_final, tm)
        background_idx = np.flatnonzero(dominant < 0)
        background_raw = residual[background_idx] if background_idx.size else None
        cond_curr = condition_number(design)
        k_bg = bg.n_bg
        panel_rows = [k_bg + i for i in range(n_fluor) if not excluded[i]]
        # a row phase one left alone has no in-span change to re-measure
        flagged = np.max(np.abs(spill - eye), axis=1) >= s.convergence_threshold
        for f in range(n_fluor):
            say(f'Phase two: {fnames[f]}')
            idx = np.flatnonzero(dominant == f)
            reject = 'not.flagged' if s.phase_two_rows == 'flagged' and not flagged[f] else None
            if fixed[f]:
                reject = 'fixed'
            cand = None
            if reject is None:
                cand = extract_raw_signature(
                    residual[idx], spectra_new, u[idx], f, None, s.intercept, s.multivariate, s.ridge,
                    s.n_levels, s.min_bin_events, s.min_negative_events, background_raw, det_names)
                if cand is None:
                    reject = 'no.fit'
            if reject is None:
                reject = _reject_reason(cand['stats'], thresholds_final[f], s)
            # a row with too few events for the joint fit was re-measured with every other row held at
            # its current spectrum, which books their errors against it
            not_joint = cand is not None and s.multivariate and n_fluor > 1 and not cand['stats']['joint']
            if reject is None and not_joint and s.phase_two_fallback == 'reject':
                reject = 'not.joint'
            leak_before = leak_after = cond_after = resid_change = like = implausible = np.nan
            trial = None
            if reject is None:
                # 'backsolved' keeps phase two as the judge but adopts phase one's own row
                use_backsolved = s.phase_two_adopt == 'backsolved' or (
                    not_joint and s.phase_two_fallback == 'backsolved')
                target = backsolved[f] if use_backsolved else cand['signature']
                stepped = np.maximum(spectra_new[f] + s.step * (target - spectra_new[f]), 0.0)
                if stepped.max() <= 0:
                    reject = 'no.fit'
                else:
                    trial = spectra_new.copy()
                    trial[f] = stepped / stepped.max()
                    if plausible:
                        # each detector keeps within the plausibility box of the reference row
                        trial[f], implausible = clip_to_plausible(trial[f], reference[f], allow[f], s.plausibility_reset)
                        if implausible > s.plausibility_reject:
                            reject = 'implausible'
                    like = af_likeness(trial[f] - spectra_new[f], bg.af_spectra)
                    if reject is None and s.af_gate and like > s.af_gate_max:
                        reject = 'af.like'
            if reject is None:
                design_trial = np.vstack([bg.basis, trial])
                cond_after = condition_number(design_trial)
                leak_seed = s.seed + (f + 1) * 1009
                leak_before = leakage(residual[idx], design, k_bg + f, panel_rows, np.random.default_rng(leak_seed))
                leak_after = leakage(residual[idx], design_trial, k_bg + f, panel_rows,
                                     np.random.default_rng(leak_seed))
                positive = np.flatnonzero(u[:, f] > tm[:, f])
                resid_change = residual_change(residual[positive], design, design_trial,
                                               np.random.default_rng(leak_seed + 1))
                if cond_after > s.max_condition_increase * cond_curr:
                    reject = 'conditioning'
                elif np.isfinite(leak_before) and np.isfinite(leak_after) and \
                        leak_after >= (1.0 + s.leakage_margin) * leak_before:
                    reject = 'leakage'
                elif np.isfinite(resid_change) and (
                        (s.residual_gate == 'no_increase' and resid_change > s.residual_margin)
                        or (s.residual_gate == 'decrease' and resid_change > -s.residual_margin)):
                    reject = 'residual'
                if reject is None:
                    spectra_new = trial
                    design = design_trial
                    cond_curr = cond_after
            entry = {'index': f, 'fluorophore': fnames[f], 'n_events': int(idx.size)}
            st = cand['stats'] if cand is not None else {}
            for key in ('n_bins', 'joint', 'x_span', 'explained', 'explained_total', 'resid_rel', 'intercept_rel',
                        'bg_align', 'clamp_frac', 'anchor_rel', 'vif_target', 'vif_max', 'deg_change',
                        'peak_curr', 'peak_new', 'peak_new_rel'):
                entry[key] = st.get(key, np.nan)
            entry['vif_partner'] = fnames[st['vif_partner']] if st.get('vif_partner', -1) >= 0 else ''
            entry.update(af_likeness=like, implausible_frac=implausible, leak_before=leak_before, leak_after=leak_after,
                         condition_after=cond_after, resid_change=resid_change, accepted=reject is None,
                         reason='accepted' if reject is None else reject)
            signature_log.append(entry)

    recovery = None
    if true_spectra is not None:
        truth = np.asarray(true_spectra, dtype=np.float64)
        truth = truth[fl] if truth.shape[0] == len(names) else truth
        deg_start = row_angle(spectra_fluor, truth)
        deg_phase1 = row_angle(backsolved, truth)
        deg_after = row_angle(spectra_new, truth)
        accepted = {r['index']: r['accepted'] for r in signature_log}
        recovery = []
        for f in range(n_fluor):
            rec = 0.0 if deg_start[f] < s.min_deg_start else (deg_start[f] - deg_after[f]) / deg_start[f]
            recovery.append({'fluorophore': fnames[f], 'deg_start': float(deg_start[f]),
                             'deg_phase1': float(deg_phase1[f]), 'deg_after': float(deg_after[f]),
                             'recovered': float(rec), 'accepted': bool(accepted.get(f, False))})

    idx = stained_fit['af_idx']
    counts = np.bincount(idx[idx >= 0], minlength=len(bg.af_spectra))
    af_assignment = [{'af_row': int(j), 'n_events': int(counts[j]),
                      'mean_abundance': float(stained_fit['af_scale'][idx == j].mean())}
                     for j in np.argsort(-counts) if counts[j] > 0]

    settings_dict = asdict(s)
    settings_dict['stop_reason'] = stop_reason
    settings_dict['kernel'] = 'compiled' if (_kernel is not None and s.use_kernel) else 'numpy'
    return FixResult(
        fluor_names=fnames, detector_names=det_names, spectra_start=spectra_fluor, spectra=spectra_new,
        spectra_backsolved=backsolved, spillover=spill, compensation=comp, trust=trust,
        coefficient_log=pair_log, source_log=source_log, signature_log=signature_log,
        convergence_log=convergence_log, spillover_history=history, thresholds=thresholds_final,
        neg_thresholds=neg_thresholds_final, dominant_counts=dominant_counts, af_assignment=af_assignment,
        prior_fallback=None if prior is None else [fnames[j] for j in prior['fallback']],
        recovery=recovery, n_events=n_events, n_events_fit=len(stained_fit['abundance']),
        settings=settings_dict, background=bg,
    )


def _normalise_rows(rows: np.ndarray) -> np.ndarray:
    rows = np.maximum(rows, 0.0)
    return rows / np.maximum(rows.max(axis=1, keepdims=True), _EPS)


def smooth_fix(spectra: np.ndarray, fluor_names: list, project, n_stained: int, rounds: int | None = None,
               fraction: float | None = None, settings: FixSettings | None = None,
               true_spectra: np.ndarray | None = None, progress=None, **kwargs) -> FixResult:
    """
    Fix My Unmix as a stochastic approximation: ``rounds`` runs, each on a
    fresh random ``fraction`` of the stained events re-unmixed with the
    current spectra, and each moving the spectra a step ``1 / (r + 1)``
    (Robbins-Monro) toward that run's result. A single run adopts the
    re-measured rows, so its subsample's noise; the average of the rounds'
    results keeps what they agree on.

    spectra    : (F, D) panel spectra, no AF row.
    project    : ``project(spectra, keep) -> (unstained, stained, background)``,
                 the unstained events and the stained events ``keep``
                 (indices into ``n_stained``) unmixed with ``spectra``
                 (``project_events``), and their ``Background``.
    rounds, fraction : default ``settings.smooth_rounds`` and
                 ``settings.smooth_fraction``.
    kwargs     : passed to ``fix_my_unmix`` (spread, variant_deltas,
                 detector_names).

    Returns the last round's FixResult with ``spectra`` the averaged
    spectra, ``spectra_start`` the input, ``smooth_log`` the rounds,
    ``recovery`` against ``true_spectra`` over all rounds, and in
    ``signature_log`` every row the rounds moved accepted (its reason and
    ``rounds_accepted`` say in how many rounds); the spillover, trust and
    logs are the last round's.
    """
    s = settings or FixSettings()
    rounds = int(s.smooth_rounds if rounds is None else rounds)
    fraction = float(s.smooth_fraction if fraction is None else fraction)
    say = progress or (lambda _t: None)
    start = np.asarray(spectra, dtype=np.float64)
    if AF_NAME in fluor_names:
        raise ValueError('smooth_fix takes the panel spectra without an AF row.')
    if not 0.0 < fraction <= 1.0 or rounds < 1:
        raise ValueError('smooth_fix needs rounds >= 1 and 0 < fraction <= 1.')
    rng = np.random.default_rng([s.seed, _STREAM_SMOOTH])
    current = start.copy()
    reference = _normalise_rows(np.asarray(kwargs.get('reference_spectra', start), dtype=np.float64))
    allow = plausible_change(reference, s.plausibility_ratio, s.plausibility_floor) \
        if s.plausibility_ratio > 0 else None
    n_keep = max(1, int(round(fraction * n_stained)))
    log, result = [], None
    for r in range(int(rounds)):
        say(f'Round {r + 1} of {rounds}')
        keep = np.arange(n_stained) if n_keep >= n_stained else np.sort(rng.choice(n_stained, n_keep, replace=False))
        unstained, stained, background = project(current, keep)
        s_round = FixSettings(**{**asdict(s), 'seed': s.seed + r})
        result = fix_my_unmix(current, fluor_names, unstained, stained, background, s_round,
                              progress=lambda t, r=r: say(f'Round {r + 1}: {t}'),
                              **{'reference_spectra': start, **kwargs})
        moved = row_angle(current, result.spectra)
        # only the rows this round changed take a step, so the others stay exactly as they were
        step = np.any(result.spectra != current, axis=1)
        current[step] = _normalise_rows(current[step] + (result.spectra[step] - current[step]) / (r + 1))
        if allow is not None:
            for f in np.flatnonzero(step):
                current[f] = clip_to_plausible(current[f], reference[f], allow[f], s.plausibility_reset)[0]
        log.append({'round': r + 1, 'n_events': int(len(keep)), 'stop_reason': result.settings['stop_reason'],
                    'iterations': len(result.convergence_log),
                    'accepted': [result.fluor_names[i] for i in np.flatnonzero(result.accepted)],
                    'max_deg_round': float(moved.max()) if moved.size else 0.0})
    # a row counts as accepted when the rounds moved it; its reason says in how many rounds it was
    moved = np.any(current != start, axis=1)
    for entry in result.signature_log:
        f = entry['index']
        k = sum(result.fluor_names[f] in r['accepted'] for r in log)
        entry['rounds_accepted'] = k
        if moved[f]:
            entry['accepted'] = True
            entry['reason'] = 'accepted' if k == len(log) else f'accepted in {k} of {len(log)} rounds'
    result.spectra_start = start
    result.spectra = current
    result.smooth_log = log
    result.settings = {**result.settings, 'smooth_rounds': int(rounds), 'smooth_fraction': float(fraction),
                       'seed': s.seed}
    if true_spectra is not None:
        truth = np.asarray(true_spectra, dtype=np.float64)
        deg_start, deg_after = row_angle(start, truth), row_angle(current, truth)
        result.recovery = [{'fluorophore': nm, 'deg_start': float(deg_start[f]), 'deg_phase1': np.nan,
                            'deg_after': float(deg_after[f]),
                            'recovered': 0.0 if deg_start[f] < s.min_deg_start
                            else float((deg_start[f] - deg_after[f]) / deg_start[f]),
                            'accepted': bool(row_angle(start[f:f + 1], current[f:f + 1])[0] > 0)}
                           for f, nm in enumerate(result.fluor_names)]
    return result
