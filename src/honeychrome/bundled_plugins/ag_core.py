"""
ag_core.py — automated gating engine shared by the gating plugins
=================================================================

Pure computation, no Qt: gate threshold algorithms, boundary calculation,
gate application, population statistics, and the ``.agmodel`` file format.
Used by the Gating Model Builder (training) and Automated Gating
(application) plugins. The file name does not end in ``_tab.py``, so the
plugin loader does not treat it as a tab.

The gate algorithms are ported from the AutoGating R package
(MIT licence, © 2026 Oliver Burton).

Data conventions
----------------
* Event arrays are already transformed (logicle etc.), one column per
  channel; ``channel_index`` maps channel name → column.
* A gate definition (``gate_def``) is a dict with ``gate_name``,
  ``gate_type`` (``singlets``, ``1dsep``, ``2dsep``, ``free``,
  ``replicate``), ``gate_marker_x``/``gate_marker_y``, ``parent_gate``,
  ``parent_popul``, ``populations`` ({name: {label, label_pos, region}}),
  ``algorithm``, ``gate_param``, ``stats_parent`` and ``origin_gate``.
* Boundaries are ``{gate_name: {pop_name: entry}}``. Each entry has a
  display polygon ``boundary``, ``threshold_x``/``threshold_y`` for
  threshold gates, a ``region`` saying which side of the thresholds the
  population lies on, and ``n_events``/``fraction`` from the data it was
  computed on.

Threshold gates are open-ended: a population is every event on its side of
the thresholds, however far outside the display range. Polygon gates
(``singlets``, ``free``) contain the events inside their polygon.

Model application modes
-----------------------
``fixed``        — apply the stored boundaries as they are.
``recalculate``  — re-run the gate's algorithm on each sample, keep the
                   stored boundary as a reference, and flag (or replace)
                   results that move further than ``drift_limit``.
"""

from __future__ import annotations

import json
import logging
import math
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

_LOG_ROOT = 'honeychrome.plugins.autogating'
_log_configured = False


def get_logger(module_name: str) -> logging.Logger:
    """Logger under the automated-gating plugins' root logger.

    The level comes from the HONEYCHROME_AG_LOGLEVEL environment variable
    (default INFO). A stdout handler is added unless the host application
    already configured one for this branch.
    """
    global _log_configured
    if not _log_configured:
        import os
        import sys
        root = logging.getLogger(_LOG_ROOT)
        level = os.environ.get('HONEYCHROME_AG_LOGLEVEL', 'INFO').upper()
        root.setLevel(getattr(logging, level, logging.INFO))
        if not root.handlers:
            handler = logging.StreamHandler(stream=sys.stdout)
            handler.setFormatter(logging.Formatter('[AutoGating %(levelname)s] %(message)s'))
            root.addHandler(handler)
            root.propagate = False
        _log_configured = True
    short = module_name.rsplit('.', 1)[-1] if module_name else 'plugin'
    return logging.getLogger(f'{_LOG_ROOT}.{short}')


log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

GATE_TYPES: tuple[str, ...] = ('singlets', '1dsep', '2dsep', 'free', 'replicate')
THRESHOLD_TYPES = frozenset({'1dsep', '2dsep'})
POLYGON_TYPES = frozenset({'singlets', 'free'})

# 'imported' keeps the boundary stored in the gate definition's ``template``
# (e.g. a gate imported from a FlowJo workspace) instead of calculating one.
ALGORITHM_IMPORTED = 'imported'

ALGORITHMS_BY_GATE_TYPE: dict[str, list[str]] = {
    'singlets':  ['singlets', ALGORITHM_IMPORTED],
    '1dsep':     ['tail', 'kde_min', 'mixture', 'otsu', 'bimodal', ALGORITHM_IMPORTED],
    '2dsep':     ['tail', 'kde_min', 'mixture', 'otsu', 'bimodal', ALGORITHM_IMPORTED],
    'free':      ['all', 'ellipse', ALGORITHM_IMPORTED],
    'replicate': ['replicate'],
}

MODEL_FORMAT_VERSION = 2
MODE_FIXED = 'fixed'
MODE_RECALCULATE = 'recalculate'
APPLICATION_MODES = (MODE_FIXED, MODE_RECALCULATE)
DEFAULT_DRIFT_LIMIT = 0.05
MIN_EVENTS = 10

REGION_BELOW = 'below'
REGION_ABOVE = 'above'
REGION_INSIDE = 'inside'
QUADRANT_REGIONS = ('x-y-', 'x+y-', 'x-y+', 'x+y+')

ALWAYS_EXCLUDED_CHANNELS = frozenset({'Time', 'ribbon', 'event_id'})


def default_populations_for_type(gate_type: str) -> dict:
    """Default populations for a new gate, with their regions."""
    if gate_type == '1dsep':
        return {
            'neg': {'label': 'neg', 'label_pos': 1, 'region': REGION_BELOW},
            'pos': {'label': 'pos', 'label_pos': 2, 'region': REGION_ABOVE},
        }
    if gate_type == '2dsep':
        return {
            'DN': {'label': 'DN', 'label_pos': 3, 'region': 'x-y-'},
            'X+': {'label': 'X+', 'label_pos': 4, 'region': 'x+y-'},
            'Y+': {'label': 'Y+', 'label_pos': 1, 'region': 'x-y+'},
            'DP': {'label': 'DP', 'label_pos': 2, 'region': 'x+y+'},
        }
    if gate_type in POLYGON_TYPES:
        return {'in': {'label': 'in', 'label_pos': 1, 'region': REGION_INSIDE}}
    return {}


def default_algorithm_for_type(gate_type: str) -> str:
    """Starting algorithm for a gate type (``tail`` for threshold gates)."""
    if gate_type in THRESHOLD_TYPES:
        return 'tail'
    if gate_type == 'free':
        return 'all'
    return gate_type


def population_key(gate_name: str, pop_name: str) -> str:
    """Stable identifier of a population across the model: ``gate/pop``."""
    return f"{gate_name}/{pop_name}"


def split_population_key(key: str) -> tuple[str, str]:
    gate, _, pop = str(key).partition('/')
    return gate, pop


def transform_columns(unmixed: np.ndarray, channels: list[str], transforms: dict) -> np.ndarray:
    """float32 copy of *unmixed* with each channel's display transform applied.

    transforms : {channel: Transform}; channels without one (or whose
                 Transform has no ``xform``) are passed through.
    """
    out = np.asarray(unmixed, dtype=np.float32).copy()
    for col, ch in enumerate(channels):
        if col >= out.shape[1]:
            break
        xform = getattr(transforms.get(ch), 'xform', None)
        if xform is None:
            continue
        out[:, col] = np.asarray(xform.apply(out[:, col].astype(np.float64)), dtype=np.float32)
    return out


# ---------------------------------------------------------------------------
# Threshold algorithms (1D, transformed scale)
# ---------------------------------------------------------------------------

def trim_quantiles(data: np.ndarray, q: float = 0.001) -> np.ndarray:
    """Drop rows outside the [q, 1 − q] quantiles on any column."""
    data = np.asarray(data)
    if len(data) == 0 or q <= 0:
        return data
    squeeze = data.ndim == 1
    arr = data.reshape(-1, 1) if squeeze else data
    lo = np.quantile(arr, q, axis=0)
    hi = np.quantile(arr, 1 - q, axis=0)
    keep = np.all((arr >= lo) & (arr <= hi), axis=1)
    out = arr[keep]
    return out.ravel() if squeeze else out


def crop_to_region(col: np.ndarray, lo_frac: float, hi_frac: float) -> np.ndarray:
    """Restrict *col* to a window of its own [min, max] range.

    ``lo_frac``/``hi_frac`` are fractions of the empirical range (not
    percentiles), as R's ``region.min``/``region.max``.
    """
    col = np.asarray(col)
    if len(col) == 0:
        return col
    c_min, c_max = float(col.min()), float(col.max())
    floor = (1 - lo_frac) * c_min + lo_frac * c_max
    ceiling = (1 - hi_frac) * c_min + hi_frac * c_max
    return col[(col >= floor) & (col <= ceiling)]


def _kde_grid(x: np.ndarray, n_grid: int = 512):
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(x, bw_method='scott')
    grid = np.linspace(float(x.min()), float(x.max()), n_grid)
    return grid, kde(grid)


def kde_mode(x: np.ndarray) -> float:
    """Location of the highest kernel-density peak."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or float(x.max()) == float(x.min()):
        return float(np.median(x)) if len(x) else 0.0
    grid, density = _kde_grid(x)
    return float(grid[int(np.argmax(density))])


def threshold_tail(x: np.ndarray, tail_fraction: float = 0.01) -> float:
    """Upper tail of the negative population, assuming it is symmetric.

    The events below the density mode are mirrored about the mode to
    estimate the whole negative population; the threshold is that
    population's ``1 − tail_fraction`` quantile, i.e.
    ``2·mode − quantile(lower half, 2·tail_fraction)``.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < MIN_EVENTS:
        return float(np.median(x)) if len(x) else 0.0
    mode = kde_mode(x)
    lower = x[x <= mode]
    if len(lower) < 5:
        return float(np.quantile(x, 1 - tail_fraction))
    q = min(max(2.0 * float(tail_fraction), 0.0), 1.0)
    return float(2.0 * mode - np.quantile(lower, q))


def threshold_kde_min(x: np.ndarray) -> float:
    """Density minimum between the two highest density peaks.

    Falls back to ``threshold_tail`` when the density has a single peak.
    """
    from scipy.signal import find_peaks
    x = np.asarray(x, dtype=float)
    if len(x) < 20 or float(x.max()) == float(x.min()):
        return float(np.median(x)) if len(x) else 0.0
    grid, density = _kde_grid(x)
    peaks, _ = find_peaks(density)
    if len(peaks) < 2:
        return threshold_tail(x)
    top_two = sorted(peaks[np.argsort(density[peaks])[::-1][:2]])
    p1, p2 = int(top_two[0]), int(top_two[1])
    valley = p1 + int(np.argmin(density[p1:p2 + 1]))
    return float(grid[valley])


def threshold_mixture(x: np.ndarray, n_components: int = 2, seed: int = 0) -> float:
    """Density crossover between the lowest- and highest-mean components
    of a Gaussian mixture."""
    from scipy.optimize import brentq
    from scipy.stats import norm
    from sklearn.mixture import GaussianMixture

    x = np.asarray(x, dtype=float)
    if len(x) < 20:
        return float(np.median(x)) if len(x) else 0.0
    gm = GaussianMixture(n_components=int(n_components), covariance_type='full',
                         random_state=seed, max_iter=200)
    gm.fit(x.reshape(-1, 1))
    means = gm.means_.ravel()
    weights = gm.weights_.ravel()
    stds = np.sqrt(gm.covariances_.ravel())
    order = np.argsort(means)
    mu1, mu2 = means[order[0]], means[order[-1]]
    w1, w2 = weights[order[0]], weights[order[-1]]
    s1, s2 = stds[order[0]], stds[order[-1]]

    def diff(v):
        return w1 * norm.pdf(v, mu1, s1) - w2 * norm.pdf(v, mu2, s2)

    try:
        return float(brentq(diff, mu1, mu2))
    except ValueError:
        return float((mu1 + mu2) / 2.0)


def threshold_otsu(x: np.ndarray, n_bins: int = 256) -> float:
    """Otsu (1979): the histogram split that maximises between-class
    variance.

    Split ``i`` puts bins ``0..i`` in the lower class, so the threshold is
    the upper edge of bin ``i``: with ``x < t`` as the negative side, every
    event in the lower class falls below it.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 20 or float(x.max()) == float(x.min()):
        return float(np.median(x)) if len(x) else 0.0
    hist, edges = np.histogram(x, bins=n_bins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    hist = hist.astype(float)
    w0 = np.cumsum(hist)
    w1 = np.cumsum(hist[::-1])[::-1]
    m0 = np.cumsum(hist * centres) / np.maximum(w0, 1e-12)
    m1 = (np.cumsum((hist * centres)[::-1]) / np.maximum(w1[::-1], 1e-12))[::-1]
    between = w0[:-1] * w1[1:] * (m0[:-1] - m1[1:]) ** 2
    return float(edges[int(np.argmax(between)) + 1])


def threshold_bimodal(x: np.ndarray) -> float:
    """Valley between two clearly separated modes (see threshold_kde_min)."""
    return threshold_kde_min(x)


def compute_threshold(x: np.ndarray, algorithm: str, params: dict | None = None) -> float:
    """Run the named threshold algorithm; unknown names use ``tail``."""
    params = params or {}
    tf = float(params.get('tail_fraction', 0.01))
    nc = int(params.get('mixture_components', 2))
    if algorithm == 'kde_min':
        return threshold_kde_min(x)
    if algorithm == 'mixture':
        return threshold_mixture(x, n_components=nc)
    if algorithm == 'otsu':
        return threshold_otsu(x)
    if algorithm == 'bimodal':
        return threshold_bimodal(x)
    return threshold_tail(x, tail_fraction=tf)


def bimodality_delta_bic(x: np.ndarray, max_events: int = 5000, trim: float = 0.001,
                         seed: int = 0) -> float:
    """BIC(1 component) − BIC(2 components) of Gaussian mixtures on *x*.

    Large positive values favour two populations. Uses a seeded subsample
    of at most *max_events* after quantile trimming.
    """
    from sklearn.mixture import GaussianMixture

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = trim_quantiles(x, trim)
    if len(x) < 50:
        return 0.0
    if len(x) > max_events:
        x = np.random.default_rng(seed).choice(x, max_events, replace=False)
    xs = x.reshape(-1, 1)
    bics = []
    for k in (1, 2):
        gm = GaussianMixture(n_components=k, covariance_type='full',
                             random_state=seed, max_iter=200)
        gm.fit(xs)
        bics.append(gm.bic(xs))
    return float(bics[0] - bics[1])


def recommend_algorithm(gate_def: dict, data: np.ndarray, channel_index: dict,
                        min_delta_bic: float = 10.0) -> str:
    """Suggest an algorithm for *gate_def* from the parent-population data.

    Threshold gates get ``mixture`` when any gated axis looks bimodal
    (ΔBIC > *min_delta_bic*), otherwise ``tail``. Other gate types keep
    their default algorithm.
    """
    gate_type = gate_def.get('gate_type', 'free')
    if gate_type not in THRESHOLD_TYPES:
        return gate_def.get('algorithm') or default_algorithm_for_type(gate_type)
    axes = [gate_def.get('gate_marker_x')]
    if gate_type == '2dsep':
        axes.append(gate_def.get('gate_marker_y'))
    for ch in axes:
        if ch in channel_index and len(data) >= 100:
            if bimodality_delta_bic(data[:, channel_index[ch]]) > min_delta_bic:
                return 'mixture'
    return 'tail'


# ---------------------------------------------------------------------------
# Gate tree
# ---------------------------------------------------------------------------

def gates_by_name(gate_defs: list[dict]) -> dict[str, dict]:
    return {g['gate_name']: g for g in gate_defs if g.get('gate_name')}


def ordered_gate_defs(gate_defs: list[dict]) -> list[dict]:
    """Gate definitions ordered so every gate follows its parent and origin.

    Ties are broken by ``gate_number`` then name. Raises ValueError on a
    cycle.
    """
    by_name = gates_by_name(gate_defs)

    def sort_key(g):
        try:
            num = int(g.get('gate_number', 10 ** 6))
        except (TypeError, ValueError):
            num = 10 ** 6
        return (num, str(g.get('gate_name', '')))

    remaining = sorted(gate_defs, key=sort_key)
    placed: set[str] = set()
    out: list[dict] = []
    while remaining:
        progressed = False
        for g in list(remaining):
            deps = {d for d in (g.get('parent_gate'), g.get('origin_gate')) if d}
            deps &= set(by_name)
            if deps <= placed:
                out.append(g)
                placed.add(g['gate_name'])
                remaining.remove(g)
                progressed = True
                break
        if not progressed:
            names = [g.get('gate_name') for g in remaining]
            raise ValueError(f"Gate hierarchy has a cycle among: {names}")
    return out


def effective_gate_def(gate_def: dict, by_name: dict[str, dict]) -> dict:
    """For a replicate gate, the origin's type, axes and populations under
    the replicate's own name and parent. Other gates are returned as is."""
    if gate_def.get('gate_type') != 'replicate':
        return gate_def
    origin = by_name.get(gate_def.get('origin_gate') or '')
    seen = {gate_def.get('gate_name')}
    while origin is not None and origin.get('gate_type') == 'replicate':
        if origin.get('gate_name') in seen:
            return gate_def
        seen.add(origin.get('gate_name'))
        origin = by_name.get(origin.get('origin_gate') or '')
    if origin is None:
        return gate_def
    merged = deepcopy(origin)
    for key in ('gate_name', 'gate_number', 'parent_gate', 'parent_popul',
                'stats_parent', 'origin_gate'):
        merged[key] = gate_def.get(key)
    merged['replicate_of'] = origin.get('gate_name')
    return merged


def gate_channels(gate_def: dict) -> list[str]:
    """Channels a gate reads (y only for 2D gate types)."""
    chans = [gate_def.get('gate_marker_x')]
    if gate_def.get('gate_type') in ('2dsep', 'singlets', 'free') and gate_def.get('gate_marker_y'):
        chans.append(gate_def.get('gate_marker_y'))
    return [c for c in chans if c]


def population_regions(gate_def: dict, boundary: dict | None = None) -> dict[str, str]:
    """{population: region} for a gate.

    Taken from the boundary entries or population definitions when set;
    otherwise inferred from population order (threshold gates: below/above
    for 1D, DN/X+/Y+/DP order for 2D).
    """
    pops = list((gate_def.get('populations') or {}).keys()) or list((boundary or {}).keys())
    gate_type = gate_def.get('gate_type')
    regions: dict[str, str] = {}
    for i, pop in enumerate(pops):
        region = ((boundary or {}).get(pop) or {}).get('region') \
            or ((gate_def.get('populations') or {}).get(pop) or {}).get('region')
        if not region:
            if gate_type == '1dsep':
                region = REGION_BELOW if i == 0 else REGION_ABOVE
            elif gate_type == '2dsep':
                region = QUADRANT_REGIONS[i] if i < 4 else None
            else:
                region = REGION_INSIDE
        if region:
            regions[pop] = region
    return regions


# ---------------------------------------------------------------------------
# Boundary calculation
# ---------------------------------------------------------------------------

def _rect(x_lo: float, x_hi: float, y_lo: float = 0.0, y_hi: float = 1.0) -> list:
    return [[x_lo, y_lo], [x_hi, y_lo], [x_hi, y_hi], [x_lo, y_hi], [x_lo, y_lo]]


class GateCalculator:
    """Compute one gate's boundary from its parent population's events.

    axis_limits : {channel: (lo, hi)} display limits, used only to draw the
                  rectangles of threshold gates. Missing channels use the
                  data range.
    """

    def __init__(self, axis_limits: dict | None = None):
        self.axis_limits = dict(axis_limits or {})

    def _limits(self, channel: str, values: np.ndarray) -> tuple[float, float]:
        lim = self.axis_limits.get(channel)
        if lim is not None and len(lim) >= 2:
            return float(lim[0]), float(lim[1])
        if len(values):
            return float(np.min(values)), float(np.max(values))
        return 0.0, 1.0

    def calculate(self, data: np.ndarray, gate_def: dict, channel_index: dict,
                  trained: dict | None = None, by_name: dict | None = None) -> dict:
        """Return ``{pop_name: entry}`` for *gate_def*, or {} if it cannot be
        computed (missing channels, too few events, no origin boundary).

        trained : boundaries computed earlier in the same run; replicate
                  gates copy their origin's boundary from here.
        """
        by_name = by_name or {}
        gate_type = gate_def.get('gate_type', 'free')
        origin_name = gate_def.get('replicate_of') or (
            gate_def.get('origin_gate') if gate_type == 'replicate' else None)
        if origin_name:
            origin = (trained or {}).get(origin_name)
            return deepcopy(origin) if origin else {}
        for ch in gate_channels(gate_def):
            if ch not in channel_index:
                return {}
        dispatch = {
            'singlets': self._calc_singlets,
            '1dsep': self._calc_1dsep,
            '2dsep': self._calc_2dsep,
            'free': self._calc_free,
        }
        if gate_def.get('algorithm') == ALGORITHM_IMPORTED:
            calc = self._calc_imported
        else:
            calc = dispatch.get(gate_type, self._calc_free)
        result = calc(np.asarray(data), gate_def, channel_index)
        pops = gate_def.get('populations') or {}
        for pop_name, entry in result.items():
            entry['label'] = (pops.get(pop_name) or {}).get('label', pop_name)
        return result

    def _calc_imported(self, data, gate_def, channel_index) -> dict:
        """The boundary stored in the gate's ``template``. Threshold gates
        keep their thresholds and redraw their rectangles to the axis limits."""
        template = gate_def.get('template') or {}
        if not template:
            return {}
        if gate_def.get('gate_type') in THRESHOLD_TYPES:
            first = next(iter(template.values()))
            tx, ty = first.get('threshold_x'), first.get('threshold_y')
            if tx is None or (gate_def.get('gate_type') == '2dsep' and ty is None):
                return {}
            x = data[:, channel_index[gate_def['gate_marker_x']]] if len(data) else None
            y = None
            if gate_def.get('gate_type') == '2dsep' and len(data):
                y = data[:, channel_index[gate_def['gate_marker_y']]]
            return self.threshold_boundaries(gate_def, float(tx),
                                             None if ty is None else float(ty), x, y)
        out = {}
        for pop, entry in template.items():
            keep = {k: deepcopy(entry[k]) for k in ('boundary', 'range', 'threshold_x',
                                                     'threshold_y', 'region') if k in entry}
            out[pop] = keep
        return out

    def _calc_singlets(self, data, gate_def, channel_index) -> dict:
        """Line fit through the main diagonal of the two scatter channels,
        with iterative outlier removal; the gate is a parallelogram of
        ± width_sd residual SDs around the line."""
        gp = gate_def.get('gate_param', {}) or {}
        q = float(gp.get('quantile_trim', 0.001))
        width_sd = float(gp.get('width_sd', 3.0))
        ix = channel_index[gate_def['gate_marker_x']]
        iy = channel_index[gate_def['gate_marker_y']]
        xy = trim_quantiles(data[:, [ix, iy]], q)
        if len(xy) < 50:
            return {}
        x, y = xy[:, 0], xy[:, 1]
        keep = np.ones(len(x), dtype=bool)
        slope, intercept = 1.0, 0.0
        for _ in range(3):
            slope, intercept = np.polyfit(x[keep], y[keep], 1)
            resid = y[keep] - (slope * x[keep] + intercept)
            sigma = float(resid.std())
            if sigma == 0:
                break
            keep[keep] = np.abs(resid) < width_sd * sigma
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        sigma = float((y[keep] - (slope * x[keep] + intercept)).std()) or 1.0
        offset = width_sd * sigma
        x_lo, x_hi = float(x.min()), float(x.max())
        poly = [
            [x_lo, slope * x_lo + intercept - offset],
            [x_hi, slope * x_hi + intercept - offset],
            [x_hi, slope * x_hi + intercept + offset],
            [x_lo, slope * x_lo + intercept + offset],
            [x_lo, slope * x_lo + intercept - offset],
        ]
        pop = next(iter(gate_def.get('populations') or {'in': {}}))
        return {pop: {'boundary': [[float(a), float(b)] for a, b in poly],
                      'threshold_x': None, 'threshold_y': None, 'region': REGION_INSIDE}}

    def _calc_1dsep(self, data, gate_def, channel_index) -> dict:
        gp = gate_def.get('gate_param', {}) or {}
        q = float(gp.get('quantile_trim', 0.001))
        ch = gate_def['gate_marker_x']
        full = data[:, channel_index[ch]]
        col = trim_quantiles(full, q)
        if len(col) < MIN_EVENTS:
            return {}
        calc = crop_to_region(col, float(gp.get('axis_lo', 0.01)), float(gp.get('axis_hi', 0.99)))
        if len(calc) < MIN_EVENTS:
            calc = col
        t = compute_threshold(calc, gate_def.get('algorithm', 'tail'), gp)
        return self.threshold_boundaries(gate_def, t, None, full, None)

    def _calc_2dsep(self, data, gate_def, channel_index) -> dict:
        gp = gate_def.get('gate_param', {}) or {}
        q = float(gp.get('quantile_trim', 0.001))
        ix = channel_index[gate_def['gate_marker_x']]
        iy = channel_index[gate_def['gate_marker_y']]
        xy = trim_quantiles(data[:, [ix, iy]], q)
        if len(xy) < 20:
            return {}
        default_algo = gate_def.get('algorithm', 'tail')

        def axis_params(axis: str) -> dict:
            p = dict(gp)
            p['tail_fraction'] = float(gp.get(f'tail_fraction_{axis}', gp.get('tail_fraction', 0.01)))
            p['mixture_components'] = int(gp.get(f'mixture_components_{axis}',
                                                 gp.get('mixture_components', 2)))
            return p

        thresholds = []
        for k, axis in enumerate(('x', 'y')):
            col = xy[:, k]
            calc = crop_to_region(col, float(gp.get(f'axis_lo_{axis}', 0.01)),
                                  float(gp.get(f'axis_hi_{axis}', 0.99)))
            if len(calc) < MIN_EVENTS:
                calc = col
            algo = gp.get(f'algorithm_{axis}') or default_algo
            thresholds.append(compute_threshold(calc, algo, axis_params(axis)))
        return self.threshold_boundaries(gate_def, thresholds[0], thresholds[1],
                                         data[:, ix], data[:, iy])

    def threshold_boundaries(self, gate_def: dict, tx: float, ty: float | None,
                             x_values: np.ndarray | None = None,
                             y_values: np.ndarray | None = None) -> dict:
        """Boundary entries for a threshold gate at (tx[, ty])."""
        x_ch = gate_def.get('gate_marker_x')
        x_lo, x_hi = self._limits(x_ch, np.asarray(x_values if x_values is not None else []))
        x_lo, x_hi = min(x_lo, tx), max(x_hi, tx)
        regions = population_regions(gate_def)
        out = {}
        if gate_def.get('gate_type') == '1dsep':
            for pop, region in regions.items():
                box = _rect(x_lo, tx) if region == REGION_BELOW else _rect(tx, x_hi)
                out[pop] = {'boundary': box, 'threshold_x': float(tx),
                            'threshold_y': None, 'region': region}
            return out
        y_ch = gate_def.get('gate_marker_y')
        y_lo, y_hi = self._limits(y_ch, np.asarray(y_values if y_values is not None else []))
        y_lo, y_hi = min(y_lo, ty), max(y_hi, ty)
        for pop, region in regions.items():
            xs = (x_lo, tx) if region.startswith('x-') else (tx, x_hi)
            ys = (y_lo, ty) if region.endswith('y-') else (ty, y_hi)
            out[pop] = {'boundary': _rect(xs[0], xs[1], ys[0], ys[1]),
                        'threshold_x': float(tx), 'threshold_y': float(ty), 'region': region}
        return out

    def _calc_free(self, data, gate_def, channel_index) -> dict:
        gp = gate_def.get('gate_param', {}) or {}
        q = float(gp.get('quantile_trim', 0.001))
        ix = channel_index[gate_def['gate_marker_x']]
        pop = next(iter(gate_def.get('populations') or {'in': {}}))
        y_ch = gate_def.get('gate_marker_y')
        if not y_ch:
            col = trim_quantiles(data[:, ix], q)
            if len(col) < MIN_EVENTS:
                return {}
            lo, hi = float(col.min()), float(col.max())
            return {pop: {'boundary': _rect(lo, hi), 'range': [lo, hi],
                          'threshold_x': None, 'threshold_y': None, 'region': REGION_INSIDE}}
        cols = trim_quantiles(data[:, [ix, channel_index[y_ch]]], q)
        if len(cols) < 4:
            return {}
        if gate_def.get('algorithm') == 'ellipse':
            poly = _ellipse_polygon(cols)
        else:
            poly = _convex_hull_polygon(cols)
        if poly is None:
            return {}
        return {pop: {'boundary': poly, 'threshold_x': None, 'threshold_y': None,
                      'region': REGION_INSIDE}}


def _convex_hull_polygon(xy: np.ndarray) -> list | None:
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(xy)
    except Exception:
        return None
    verts = [[float(a), float(b)] for a, b in xy[hull.vertices]]
    verts.append(verts[0])
    return verts


def _ellipse_polygon(xy: np.ndarray, n_points: int = 64, n_sd: float = 3.0) -> list | None:
    try:
        cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
        vals, vecs = np.linalg.eigh(np.cov(xy.T))
    except Exception:
        return None
    a = n_sd * math.sqrt(abs(vals[0]))
    b = n_sd * math.sqrt(abs(vals[1]))
    angle = math.atan2(vecs[1, 0], vecs[0, 0])
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    xr = a * np.cos(t) * math.cos(angle) - b * np.sin(t) * math.sin(angle) + cx
    yr = a * np.cos(t) * math.sin(angle) + b * np.sin(t) * math.cos(angle) + cy
    poly = [[float(p), float(q)] for p, q in zip(xr, yr)]
    poly.append(poly[0])
    return poly


# ---------------------------------------------------------------------------
# Gate application
# ---------------------------------------------------------------------------

def population_masks(gate_def: dict, boundary: dict, data: np.ndarray,
                     channel_index: dict) -> dict[str, np.ndarray]:
    """Boolean masks over *data* rows for each population of one gate.

    *gate_def* must be the effective definition (see effective_gate_def).
    Returns {} when a channel is missing or the boundary is empty.
    """
    if not boundary:
        return {}
    chans = gate_channels(gate_def)
    if any(ch not in channel_index for ch in chans):
        return {}
    gate_type = gate_def.get('gate_type')
    n = len(data)
    x = data[:, channel_index[chans[0]]] if chans else np.zeros(n)
    regions = population_regions(gate_def, boundary)
    masks: dict[str, np.ndarray] = {}

    if gate_type in THRESHOLD_TYPES:
        first = next(iter(boundary.values()))
        tx = first.get('threshold_x')
        if tx is None:
            return {}
        above_x = x >= float(tx)
        if gate_type == '1dsep':
            for pop in boundary:
                region = regions.get(pop, REGION_ABOVE)
                masks[pop] = above_x.copy() if region == REGION_ABOVE else ~above_x
            return masks
        ty = first.get('threshold_y')
        if ty is None or len(chans) < 2:
            return {}
        above_y = data[:, channel_index[chans[1]]] >= float(ty)
        for pop in boundary:
            region = regions.get(pop, 'x+y+')
            mx = above_x if region.startswith('x+') else ~above_x
            my = above_y if region.endswith('y+') else ~above_y
            masks[pop] = mx & my
        return masks

    from matplotlib.path import Path as MplPath
    for pop, entry in boundary.items():
        rng = entry.get('range')
        if rng is not None and len(chans) == 1:
            masks[pop] = (x >= float(rng[0])) & (x <= float(rng[1]))
            continue
        coords = np.asarray(entry.get('boundary') or [], dtype=float)
        if coords.ndim != 2 or coords.shape[0] < 3 or len(chans) < 2:
            masks[pop] = np.zeros(n, dtype=bool)
            continue
        pts = np.column_stack([x, data[:, channel_index[chans[1]]]])
        masks[pop] = MplPath(coords).contains_points(pts)
    return masks


def boundary_drift(gate_def: dict, boundary: dict, reference: dict) -> float | None:
    """How far a recalculated boundary moved from the reference, in
    transformed units: the largest threshold shift for threshold gates,
    the centroid distance for polygon gates."""
    if not boundary or not reference:
        return None
    gate_type = gate_def.get('gate_type')
    a = next(iter(boundary.values()))
    b = next(iter(reference.values()))
    if gate_type in THRESHOLD_TYPES:
        shifts = []
        for key in ('threshold_x', 'threshold_y'):
            if a.get(key) is not None and b.get(key) is not None:
                shifts.append(abs(float(a[key]) - float(b[key])))
        return max(shifts) if shifts else None
    shifts = []
    for pop, entry in boundary.items():
        ref = reference.get(pop)
        if not ref:
            continue
        pa = np.asarray(entry.get('boundary') or [], dtype=float)
        pb = np.asarray(ref.get('boundary') or [], dtype=float)
        if pa.ndim == 2 and pb.ndim == 2 and len(pa) and len(pb):
            shifts.append(float(np.linalg.norm(pa.mean(axis=0) - pb.mean(axis=0))))
    return max(shifts) if shifts else None


@dataclass
class GatingResult:
    """Outcome of running a gate hierarchy on one event array.

    boundaries   : {gate: {pop: entry}} as applied, with n_events/fraction.
    masks        : {gate: {pop: bool mask over the array's rows}}.
    parent_masks : {gate: bool mask of the events the gate was applied to}.
    status       : {gate: {'source', 'drift', 'flag', 'message'}}; source is
                   'trained', 'fixed', 'recalculated' or 'reference';
                   flag is None, 'drift', 'failed' or 'skipped'.
    """
    boundaries: dict = field(default_factory=dict)
    masks: dict = field(default_factory=dict)
    parent_masks: dict = field(default_factory=dict)
    status: dict = field(default_factory=dict)

    def population_mask(self, gate_name: str, pop_name: str) -> np.ndarray | None:
        return (self.masks.get(gate_name) or {}).get(pop_name)


def run_gating(gate_defs: list[dict], data: np.ndarray, channel_index: dict,
               reference: dict | None = None, modes: dict | None = None,
               default_mode: str = MODE_FIXED,
               drift_limit: float = DEFAULT_DRIFT_LIMIT,
               drift_action: str = 'flag',
               axis_limits: dict | None = None) -> GatingResult:
    """Run a gate hierarchy on one transformed event array.

    Without *reference* every gate is calculated from the data (training).
    With *reference*, each gate uses its mode from *modes* (default
    *default_mode*): ``fixed`` applies the reference boundary;
    ``recalculate`` computes a new one, falls back to the reference if that
    fails, and flags a shift larger than *drift_limit*. With
    ``drift_action='reference'`` a flagged gate uses the reference instead.

    Each gate is applied to its parent population only, in hierarchy order.
    """
    data = np.asarray(data)
    n = len(data)
    by_name = gates_by_name(gate_defs)
    calc = GateCalculator(axis_limits)
    modes = modes or {}
    result = GatingResult()
    root = np.ones(n, dtype=bool)

    for gate_def in ordered_gate_defs(gate_defs):
        name = gate_def['gate_name']
        eff = effective_gate_def(gate_def, by_name)
        parent_gate = gate_def.get('parent_gate')
        if parent_gate:
            parent_pop = gate_def.get('parent_popul')
            parent_mask = result.population_mask(parent_gate, parent_pop)
            if parent_mask is None:
                result.status[name] = {'source': None, 'drift': None, 'flag': 'skipped',
                                       'message': f"parent population {parent_gate}/{parent_pop} "
                                                  "is not available"}
                continue
        else:
            parent_mask = root
        parent_data = data[parent_mask]

        ref = (reference or {}).get(name)
        mode = modes.get(name, default_mode) if reference is not None else MODE_RECALCULATE
        status = {'source': None, 'drift': None, 'flag': None, 'message': ''}
        boundary: dict = {}

        if mode == MODE_FIXED:
            boundary = deepcopy(ref) if ref else {}
            status['source'] = 'fixed'
            if not boundary:
                status['flag'] = 'failed'
                status['message'] = 'no stored boundary'
        else:
            try:
                boundary = calc.calculate(parent_data, eff, channel_index,
                                          trained=result.boundaries, by_name=by_name)
            except Exception as exc:
                log.warning("gate %s: calculation failed: %s", name, exc)
                status['message'] = str(exc)
                boundary = {}
            status['source'] = 'trained' if reference is None else 'recalculated'
            if not boundary:
                if ref:
                    boundary = deepcopy(ref)
                    status['source'] = 'reference'
                    status['flag'] = 'failed'
                    status['message'] = status['message'] or 'too few events to calculate'
                else:
                    status['flag'] = 'failed'
                    status['message'] = status['message'] or 'too few events to calculate'
            elif ref:
                drift = boundary_drift(eff, boundary, ref)
                status['drift'] = drift
                if drift is not None and drift > drift_limit:
                    status['flag'] = 'drift'
                    status['message'] = f"moved {drift:.3g} from the reference"
                    if drift_action == 'reference':
                        boundary = deepcopy(ref)
                        status['source'] = 'reference'

        result.status[name] = status
        if not boundary:
            continue
        masks_parent = population_masks(eff, boundary, parent_data, channel_index)
        if not masks_parent:
            status['flag'] = status['flag'] or 'failed'
            status['message'] = status['message'] or 'gate channels not in the data'
            continue
        parent_idx = np.flatnonzero(parent_mask)
        full_masks = {}
        n_parent = int(parent_mask.sum())
        for pop, m in masks_parent.items():
            full = np.zeros(n, dtype=bool)
            full[parent_idx[m]] = True
            full_masks[pop] = full
            if pop in boundary:
                k = int(m.sum())
                boundary[pop]['n_events'] = k
                boundary[pop]['fraction'] = float(k / n_parent) if n_parent else 0.0
        result.masks[name] = full_masks
        result.parent_masks[name] = parent_mask
        result.boundaries[name] = boundary
    return result


def train_boundaries(gate_defs: list[dict], data: np.ndarray, channel_index: dict,
                     axis_limits: dict | None = None) -> GatingResult:
    """Calculate every gate from pooled training data (see run_gating)."""
    return run_gating(gate_defs, data, channel_index, reference=None, axis_limits=axis_limits)


# ---------------------------------------------------------------------------
# Population statistics
# ---------------------------------------------------------------------------

def _resolve_stats_parent(spec, result: GatingResult) -> tuple[str, np.ndarray | None]:
    """Mask of a stats-parent spec: 'root', 'gate/pop', or a gate name whose
    applied-to events are used."""
    if spec in (None, '', 'root'):
        return 'root', None
    spec = str(spec)
    if '/' in spec:
        gate, pop = split_population_key(spec)
        return spec, result.population_mask(gate, pop)
    return spec, result.parent_masks.get(spec)


def population_table(gate_defs: list[dict], result: GatingResult,
                     n_total: int | None = None) -> list[dict]:
    """One row per population with counts and fractions.

    Columns: key ('gate/pop'), gate, population, label, n_events,
    parent_key, n_parent, fraction_of_parent, and — when the gate's
    ``stats_parent`` names a reference for this population — stats_parent,
    n_stats_parent and fraction_of_stats_parent. ``stats_parent`` maps a
    population name to 'root', 'gate/pop', or a gate name.
    """
    rows = []
    for gate_def in ordered_gate_defs(gate_defs):
        name = gate_def['gate_name']
        masks = result.masks.get(name)
        if not masks:
            continue
        parent_mask = result.parent_masks.get(name)
        n_parent = int(parent_mask.sum()) if parent_mask is not None else int(n_total or 0)
        parent_key = (population_key(gate_def['parent_gate'], gate_def.get('parent_popul'))
                      if gate_def.get('parent_gate') else 'root')
        pops = gate_def.get('populations') or {}
        stats_parent = gate_def.get('stats_parent') or {}
        for pop, m in masks.items():
            k = int(m.sum())
            row = {
                'key': population_key(name, pop),
                'gate': name,
                'population': pop,
                'label': (pops.get(pop) or {}).get('label', pop),
                'n_events': k,
                'parent_key': parent_key,
                'n_parent': n_parent,
                'fraction_of_parent': float(k / n_parent) if n_parent else float('nan'),
            }
            spec = stats_parent.get(pop) if isinstance(stats_parent, dict) else None
            if spec:
                label, sp_mask = _resolve_stats_parent(spec, result)
                n_sp = int(sp_mask.sum()) if sp_mask is not None else int(n_total or len(m))
                row['stats_parent'] = label
                row['n_stats_parent'] = n_sp
                row['fraction_of_stats_parent'] = float(k / n_sp) if n_sp else float('nan')
            rows.append(row)
    return rows


def population_medians(gate_defs: list[dict], result: GatingResult, data: np.ndarray,
                       channel_index: dict, channels: list[str],
                       min_events: int = 1) -> dict[tuple[str, str], float]:
    """Median of each channel within each population: {(pop key, channel): value}.
    Populations with fewer than *min_events* events are left out."""
    out: dict[tuple[str, str], float] = {}
    cols = [(ch, channel_index[ch]) for ch in channels if ch in channel_index]
    for gate_def in gate_defs:
        name = gate_def['gate_name']
        for pop, m in (result.masks.get(name) or {}).items():
            if int(m.sum()) < min_events:
                continue
            sub = data[m]
            key = population_key(name, pop)
            for ch, ci in cols:
                out[(key, ch)] = float(np.median(sub[:, ci]))
    return out


def ancestor_channels(gate_defs: list[dict], gate_name: str) -> list[str]:
    """Channels used by a gate and all its ancestors (its 'type' markers)."""
    by_name = gates_by_name(gate_defs)
    chans: list[str] = []
    seen = set()
    cur = by_name.get(gate_name)
    while cur is not None and cur.get('gate_name') not in seen:
        seen.add(cur.get('gate_name'))
        for ch in gate_channels(effective_gate_def(cur, by_name)):
            if ch not in chans:
                chans.append(ch)
        cur = by_name.get(cur.get('parent_gate') or '')
    return chans


# ---------------------------------------------------------------------------
# Transforms and portability
# ---------------------------------------------------------------------------

TRANSFORM_KEYS = ('id', 'scale_t', 'linear_a', 'logicle_w', 'logicle_m', 'logicle_a',
                  'log_m', 'limits')


def transform_params(transform) -> dict:
    """JSON-safe parameters of a Honeychrome Transform object."""
    out = {}
    for key in TRANSFORM_KEYS:
        value = getattr(transform, key, None)
        if isinstance(value, np.generic):
            value = value.item()
        if key == 'limits' and value is not None:
            value = [float(v) for v in value]
        out[key] = value
    return out


class _Identity:
    def apply(self, v):
        return np.asarray(v, dtype=float)

    def inverse(self, v):
        return np.asarray(v, dtype=float)


def make_transform(params: dict | None):
    """Object with apply()/inverse() for a transform parameter dict.

    Linear (id 0), logicle (1) and log (2) use FlowKit, as Honeychrome's
    Transform does; anything else (e.g. time) is the identity.
    """
    if not params:
        return _Identity()
    tid = params.get('id')
    from flowkit import transforms as fk
    if tid == 0:
        return fk.LinearTransform(param_t=float(params.get('scale_t', 262144)),
                                  param_a=float(params.get('linear_a', 100)))
    if tid == 1:
        return fk.LogicleTransform(param_t=float(params.get('scale_t', 262144)),
                                   param_w=float(params.get('logicle_w', 0.5)),
                                   param_m=float(params.get('logicle_m', 4.5)),
                                   param_a=float(params.get('logicle_a', 0.0)))
    if tid == 2:
        return fk.LogTransform(param_t=float(params.get('scale_t', 262144)),
                               param_m=float(params.get('log_m', 6)))
    return _Identity()


def same_transform(a: dict | None, b: dict | None, tol: float = 1e-9) -> bool:
    """True when two parameter dicts define the same transform (limits are
    display-only and ignored)."""
    if not a or not b:
        return False
    for key in TRANSFORM_KEYS:
        if key == 'limits':
            continue
        va, vb = a.get(key), b.get(key)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if abs(float(va) - float(vb)) > tol * max(1.0, abs(float(va))):
                return False
        elif va != vb:
            return False
    return True


def densify_polygon(coords, points_per_edge: int = 64) -> np.ndarray:
    """Insert evenly spaced points along each polygon edge, so the shape
    survives a non-linear change of axis transform."""
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        return coords
    out = []
    for a, b in zip(coords[:-1], coords[1:]):
        t = np.linspace(0.0, 1.0, points_per_edge, endpoint=False)[:, None]
        out.append(a + t * (b - a))
    out.append(coords[-1:])
    return np.vstack(out)


def _convert(values, src, dst) -> np.ndarray:
    return np.asarray(dst.apply(src.inverse(np.asarray(values, dtype=float))), dtype=float)


def remap_boundaries(gate_defs: list[dict], boundaries: dict, src_params: dict,
                     dst_params: dict, transform_factory=make_transform,
                     points_per_edge: int = 64) -> tuple[dict, dict]:
    """Convert boundaries from one set of channel transforms to another.

    src_params / dst_params : {channel: transform parameter dict}, keyed by
        the channel names used in *gate_defs*. Channels whose transforms
        match, or that have no source parameters, are left unchanged.

    Returns (new boundaries, {channel: 'same' | 'remapped' | 'unknown'}).
    Thresholds convert exactly; polygon vertices are densified along each
    edge before conversion.
    """
    report: dict[str, str] = {}
    cache: dict[str, tuple] = {}

    def converter(ch):
        if ch in cache:
            return cache[ch]
        sp, dp = (src_params or {}).get(ch), (dst_params or {}).get(ch)
        if not sp or not dp:
            report[ch] = 'unknown'
            cache[ch] = None
        elif same_transform(sp, dp):
            report[ch] = 'same'
            cache[ch] = None
        else:
            report[ch] = 'remapped'
            cache[ch] = (transform_factory(sp), transform_factory(dp))
        return cache[ch]

    by_name = gates_by_name(gate_defs)
    out = deepcopy(boundaries)
    for name, entries in out.items():
        gdef = by_name.get(name)
        if gdef is None:
            continue
        eff = effective_gate_def(gdef, by_name)
        chans = gate_channels(eff)
        convs = [converter(ch) for ch in chans]
        if not any(convs):
            continue
        for entry in entries.values():
            for key, k in (('threshold_x', 0), ('threshold_y', 1)):
                if entry.get(key) is not None and k < len(convs) and convs[k]:
                    entry[key] = float(_convert([entry[key]], *convs[k])[0])
            if entry.get('range') is not None and convs[0]:
                entry['range'] = [float(v) for v in _convert(entry['range'], *convs[0])]
            coords = np.asarray(entry.get('boundary') or [], dtype=float)
            if coords.ndim == 2 and coords.shape[1] == 2 and len(coords):
                if eff.get('gate_type') in POLYGON_TYPES:
                    coords = densify_polygon(coords, points_per_edge)
                for k in range(min(2, len(convs))):
                    if convs[k]:
                        coords[:, k] = _convert(coords[:, k], *convs[k])
                entry['boundary'] = coords.tolist()
    return out, report


def remap_model(model: dict, dst_params: dict,
                transform_factory=make_transform) -> tuple[dict, list[str]]:
    """Convert a normalised model's boundaries to the transforms *dst_params*.

    *dst_params* is {channel: transform parameter dict} for the experiment
    the model is being used in. The model's own ``transforms`` are the
    source; a model without them (format 1) is returned unchanged, since
    normalize_model has already warned about it.

    Returns (model, messages). The returned model's ``transforms`` describe
    the converted boundaries.
    """
    src = model.get('transforms') or {}
    if not src:
        return model, []
    gate_defs = model.get('gate_definitions') or []
    new_boundaries, report = remap_boundaries(
        gate_defs, model.get('trained_boundaries') or {},
        src, dst_params, transform_factory=transform_factory,
    )
    out = deepcopy(model)
    out['trained_boundaries'] = new_boundaries
    for key in ('template', 'template_original'):
        templates = {g['gate_name']: g[key] for g in gate_defs if g.get(key)}
        if not templates:
            continue
        new_templates, template_report = remap_boundaries(
            gate_defs, templates, src, dst_params, transform_factory=transform_factory,
        )
        for ch, status in template_report.items():
            report.setdefault(ch, status)
        for g in out['gate_definitions']:
            if g.get('gate_name') in new_templates:
                g[key] = new_templates[g['gate_name']]
    remapped = sorted(ch for ch, s in report.items() if s == 'remapped')
    for ch in remapped:
        out['transforms'][ch] = deepcopy(dst_params[ch])
    messages = []
    if remapped:
        messages.append(
            "Boundaries were converted to this experiment's axis transforms on: "
            + ", ".join(remapped)
        )
    no_source = sorted(ch for ch, s in report.items() if s == 'unknown' and ch not in src)
    no_target = sorted(ch for ch, s in report.items()
                       if s == 'unknown' and ch in src and ch not in (dst_params or {}))
    if no_source:
        messages.append(
            "The model has no stored transform for: " + ", ".join(no_source)
            + ". Boundaries on these channels are used as stored."
        )
    if no_target:
        messages.append(
            "This experiment has no transform for: " + ", ".join(no_target)
            + ". Boundaries on these channels are used as stored."
        )
    return out, messages


def rename_channels(gate_defs: list[dict], alignment: dict) -> list[dict]:
    """Copy of *gate_defs* with axis channels renamed via {old: new}."""
    out = deepcopy(gate_defs)
    for g in out:
        for key in ('gate_marker_x', 'gate_marker_y'):
            if g.get(key) in alignment:
                g[key] = alignment[g[key]]
    return out


def rekey(mapping: dict | None, alignment: dict) -> dict:
    """Copy of a per-channel dict with keys renamed via {old: new}."""
    return {alignment.get(k, k): v for k, v in (mapping or {}).items()}


# ---------------------------------------------------------------------------
# .agmodel format
# ---------------------------------------------------------------------------

def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def to_jsonable(obj):
    """Copy of *obj* with numpy values converted to JSON types and
    non-finite floats replaced by None."""
    return _jsonable(obj)


def serialisable_boundaries(boundaries: dict) -> dict:
    """Boundaries with numpy values converted for JSON."""
    return _jsonable(boundaries)


def new_model(gate_defs: list[dict], boundaries: dict, *, channel_map: dict,
              transforms: dict, training_summary: dict, metadata: dict,
              af: dict | None = None, qc: dict | None = None,
              application: dict | None = None) -> dict:
    """Assemble a format-2 model dict."""
    modes = {g['gate_name']: MODE_FIXED for g in gate_defs}
    app = {'mode_per_gate': modes, 'default_mode': MODE_FIXED,
           'drift_limit': DEFAULT_DRIFT_LIMIT, 'drift_action': 'flag'}
    app.update(application or {})
    meta = dict(metadata)
    meta.setdefault('n_gates', len(gate_defs))
    meta.setdefault('n_populations', sum(len(g.get('populations') or {}) for g in gate_defs))
    return _jsonable({
        'format_version': MODEL_FORMAT_VERSION,
        'metadata': meta,
        'gate_definitions': gate_defs,
        'trained_boundaries': boundaries,
        'channel_map': channel_map,
        'transforms': transforms,
        'training_summary': training_summary,
        'af': af or {'sample_af_profiles': {}},
        'qc': qc or {'enabled': False, 'settings': {}, 'fingerprints': {}},
        'application': app,
    })


def normalize_model(model: dict) -> tuple[dict, list[str]]:
    """Validate a loaded model and upgrade format 1 to format 2.

    Returns (model, warnings). Raises ValueError for unsupported versions or
    a malformed gate list.
    """
    warnings: list[str] = []
    version = model.get('format_version')
    if version not in (1, 2):
        raise ValueError(f"Unsupported model format_version {version!r}.")
    m = deepcopy(model)
    gate_defs = m.get('gate_definitions')
    if not isinstance(gate_defs, list):
        raise ValueError("Model has no gate_definitions list.")
    for g in gate_defs:
        if not g.get('gate_name'):
            raise ValueError("A gate definition has no gate_name.")
        g.setdefault('populations', {})
        g.setdefault('gate_param', {})
        g.setdefault('stats_parent', {})
        g.setdefault('origin_gate', None)
        if not g.get('parent_gate'):
            g['parent_gate'] = None
            g['parent_popul'] = 'root'
    ordered_gate_defs(gate_defs)
    m.setdefault('trained_boundaries', {})
    m.setdefault('channel_map', {})
    if version == 1:
        warnings.append(
            "This model was saved before transform settings were recorded; "
            "gates are applied as stored, without checking that the experiment's "
            "axis transforms match."
        )
        m['transforms'] = {}
        m.setdefault('af', {'sample_af_profiles': {}})
        m.setdefault('qc', {'enabled': False, 'settings': {}, 'fingerprints': {}})
        m['format_version'] = MODEL_FORMAT_VERSION
    m.setdefault('transforms', {})
    m.setdefault('training_summary', {})
    m.setdefault('af', {'sample_af_profiles': {}})
    m.setdefault('qc', {'enabled': False, 'settings': {}, 'fingerprints': {}})
    app = m.setdefault('application', {})
    app.setdefault('default_mode', MODE_FIXED)
    app.setdefault('drift_limit', DEFAULT_DRIFT_LIMIT)
    app.setdefault('drift_action', 'flag')
    modes = app.setdefault('mode_per_gate', {})
    for g in gate_defs:
        if modes.get(g['gate_name']) not in APPLICATION_MODES:
            modes[g['gate_name']] = app['default_mode']
    untrained = [g['gate_name'] for g in gate_defs
                 if g['gate_name'] not in m['trained_boundaries']]
    if untrained:
        warnings.append("Untrained gate(s): " + ", ".join(untrained))
    return m, warnings


def read_model(path) -> tuple[dict, list[str]]:
    """Load and normalise an ``.agmodel`` file."""
    with open(path, 'r', encoding='utf-8') as fh:
        return normalize_model(json.load(fh))


def write_model(path, model: dict) -> None:
    """Write a model dict as indented JSON.

    The file is written beside *path* and then moved into place, so an
    interrupted write leaves any previous file intact.
    """
    import os
    from pathlib import Path

    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(_jsonable(model), fh, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Export to a FlowKit GatingStrategy
# ---------------------------------------------------------------------------

def model_to_gating_strategy(gate_defs: list[dict], boundaries: dict, transformations: dict,
                             gate_name_prefix: str = ''):
    """Build a FlowKit GatingStrategy from model gates, in Honeychrome's
    conventions: coordinates on the transformed scale with
    ``transformation_ref`` set to the channel.

    transformations : {channel: Transform} (only ``.xform`` is used).

    Threshold gates become one open-ended rectangle gate per population;
    polygon gates become polygon gates. Each population is a gate named
    ``{prefix}{gate}/{pop}``, nested under its parent population.
    """
    import flowkit as fk
    from flowkit import gates as fk_gates

    def dim(ch, lo=None, hi=None):
        tref = ch if getattr(transformations.get(ch), 'xform', None) is not None else None
        return fk.Dimension(ch, compensation_ref='uncompensated', transformation_ref=tref,
                            range_min=lo, range_max=hi)

    gs = fk.GatingStrategy()
    for ch, tr in transformations.items():
        if getattr(tr, 'xform', None) is not None:
            gs.transformations[ch] = tr.xform
    by_name = gates_by_name(gate_defs)
    paths: dict[str, tuple] = {}
    for gate_def in ordered_gate_defs(gate_defs):
        name = gate_def['gate_name']
        entries = boundaries.get(name)
        if not entries:
            continue
        eff = effective_gate_def(gate_def, by_name)
        chans = gate_channels(eff)
        if gate_def.get('parent_gate'):
            parent_key = population_key(gate_def['parent_gate'], gate_def.get('parent_popul'))
            if parent_key not in paths:
                continue
            gate_path = paths[parent_key] + (gate_name_prefix + parent_key,)
        else:
            gate_path = ('root',)
        regions = population_regions(eff, entries)
        for pop, entry in entries.items():
            key = population_key(name, pop)
            gname = gate_name_prefix + key
            gtype = eff.get('gate_type')
            if gtype in THRESHOLD_TYPES:
                region = regions.get(pop, '')
                tx = entry.get('threshold_x')
                if gtype == '1dsep':
                    lo, hi = (tx, None) if region == REGION_ABOVE else (None, tx)
                    dims = [dim(chans[0], lo, hi)]
                else:
                    ty = entry.get('threshold_y')
                    xr = (tx, None) if region.startswith('x+') else (None, tx)
                    yr = (ty, None) if region.endswith('y+') else (None, ty)
                    dims = [dim(chans[0], *xr), dim(chans[1], *yr)]
                gate = fk_gates.RectangleGate(gname, dims)
            else:
                verts = [tuple(v) for v in (entry.get('boundary') or [])]
                if len(chans) < 2 or len(verts) < 3:
                    continue
                if verts[0] == verts[-1]:
                    verts = verts[:-1]
                gate = fk_gates.PolygonGate(gname, [dim(chans[0]), dim(chans[1])], verts)
            gs.add_gate(gate, gate_path=gate_path)
            paths[key] = gate_path
    return gs
