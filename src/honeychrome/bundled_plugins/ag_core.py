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
* A ``1dsep`` gate separates on ``gate_marker_x``; ``gate_marker_y`` is the
  view channel it is drawn against (None: a histogram of the separation
  channel). ``sep_axis`` ('x' or 'y') is the plot axis the separation
  channel is drawn on (see ``plot_channels``).
* A threshold gate axis whose algorithm is ``replicate`` takes its
  threshold from elsewhere (``gate_param['replicate_x']`` /
  ``['replicate_y']``): ``{'gate': name}`` copies the threshold another gate
  has on the same channel; ``{'population': 'gate/pop', 'algorithm': a}``
  calculates it with algorithm *a* on the events of that population, which
  is usually marked as a reference population (``populations[pop]
  ['reference']``): one whose biology is known and whose events are
  plentiful, so its positive/negative boundary can be transferred.
  ``populations[pop]['anchor']`` marks the anchor population
  (``anchor_population``): the default reference of replicated thresholds
  in the Gating Model Builder and the default abundance anchor of
  enrichment analysis.
* An imported gate's ``template`` may hold its shapes as drawn: entries
  with ``shape_channels`` (the channels of the plot it was drawn on) and a
  closed ``boundary`` polygon, or a ``range`` for a 1-D shape. While the
  gate's algorithm is ``imported`` these shapes are applied exactly; the
  gate's type, regions and ``proposed_algorithm`` describe the trainable
  gate it converts to.

Threshold gates are open-ended: a population is every event on its side of
the thresholds, however far outside the display range. Polygon gates
(``singlets``, ``free``) and shapes as drawn contain the events inside their
polygon or range; when display limits are given, events beyond them are
tested where the plots show them, on the axis edge, so a shape drawn up to
an axis includes the off-scale events past it.

Model application modes
-----------------------
``fixed``        — apply the stored boundaries as they are.
``recalculate``  — re-run the gate's algorithm on each sample, keep the
                   stored boundary as a reference, and flag (or replace)
                   results that move further than ``drift_limit``.
"""

from __future__ import annotations

import hashlib
import heapq
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

# 'replicate' takes a threshold from another gate or a reference population.
ALGORITHM_REPLICATE = 'replicate'

# Offered in order of preference.
ALGORITHMS_BY_GATE_TYPE: dict[str, list[str]] = {
    'singlets':  ['singlets', ALGORITHM_IMPORTED],
    '1dsep':     ['mixture', 'tail', ALGORITHM_REPLICATE, 'otsu', 'bimodal', 'kde_min',
                  ALGORITHM_IMPORTED],
    '2dsep':     ['mixture', 'tail', ALGORITHM_REPLICATE, 'otsu', 'bimodal', 'kde_min',
                  ALGORITHM_IMPORTED],
    'free':      ['cells', 'all', ALGORITHM_IMPORTED],
    'replicate': ['replicate'],
}
# Algorithms that calculate a threshold from data (replicate sources use one),
# in order of preference.
THRESHOLD_ALGORITHMS = ('mixture', 'tail', 'otsu', 'bimodal', 'kde_min')

# Colour (RGB) of each algorithm: its name in algorithm pickers and its
# threshold or boundary on review plots.
ALGORITHM_COLOURS: dict[str, tuple] = {
    'mixture':           (142, 68, 173),
    'tail':              (220, 110, 0),
    ALGORITHM_REPLICATE: (30, 100, 220),
    'otsu':              (0, 140, 140),
    'bimodal':           (200, 40, 120),
    'kde_min':           (140, 95, 40),
    ALGORITHM_IMPORTED:  (30, 160, 50),
}
DEFAULT_ALGORITHM_COLOUR = (30, 160, 50)


def algorithm_colour(algorithm: str | None) -> tuple:
    """RGB colour of an algorithm (see ALGORITHM_COLOURS)."""
    return ALGORITHM_COLOURS.get(algorithm or '', DEFAULT_ALGORITHM_COLOUR)


def default_axis_lo(algorithm: str | None) -> float:
    """Default lower crop (``axis_lo``) of a threshold calculation.

    ``tail`` mirrors the events below the negative population's mode, so
    cropping the bottom of the axis would trim the very events it measures;
    it starts uncropped. Other algorithms crop the lowest 1% of the range.
    """
    return 0.0 if algorithm == 'tail' else 0.01

MODEL_FORMAT_VERSION = 2
DEFAULT_TAIL_FRACTION = 0.0001
# Calculated thresholds are multiplied by this factor in raw (untransformed)
# units: 2 doubles the threshold intensity, 0.5 halves it.
DEFAULT_THRESHOLD_FACTOR = 1.0
THRESHOLD_FACTOR_RANGE = (0.01, 100.0)
DEFAULT_EDGE_DENSITY = 0.002        # cells: density fall-off, fraction of the peak
CELL_SIDE_CELLS = 'cells'           # cells: keep the events above the debris cut
CELL_SIDE_DEBRIS = 'debris'         # cells: keep the events below it
# Tail fraction of kde_min's fallback when the density has a single peak.
KDE_MIN_FALLBACK_TAIL = 0.01
MODE_FIXED = 'fixed'
MODE_RECALCULATE = 'recalculate'
APPLICATION_MODES = (MODE_FIXED, MODE_RECALCULATE)
DEFAULT_DRIFT_LIMIT = 0.05
MIN_EVENTS = 10
# Above this many events, kernel densities are computed from finely binned
# events (see _kde_grid) instead of summing one kernel per event.
KDE_EXACT_MAX_EVENTS = 20_000

REGION_BELOW = 'below'
REGION_ABOVE = 'above'
REGION_INSIDE = 'inside'
QUADRANT_REGIONS = ('x-y-', 'x+y-', 'x-y+', 'x+y+')
REGION_JOIN = '|'


def region_quadrants(region: str | None) -> tuple:
    """Quadrants of a 2dsep region, in QUADRANT_REGIONS order. A region is
    one quadrant ('x+y-') or several joined with '|' ('x-y-|x-y+' is the
    whole x- side; three quadrants make an L shape)."""
    parts = set(str(region or '').split(REGION_JOIN))
    return tuple(q for q in QUADRANT_REGIONS if q in parts)


def join_quadrants(quadrants) -> str:
    """Region string of a set of quadrants (see region_quadrants)."""
    parts = set(quadrants)
    return REGION_JOIN.join(q for q in QUADRANT_REGIONS if q in parts)


def region_sign(region: str | None, axis: str) -> str | None:
    """'+' or '-' when every quadrant of a 2dsep region lies on that side
    of the *axis* ('x' or 'y') threshold, else None."""
    k = 1 if axis == 'x' else 3
    signs = {q[k] for q in region_quadrants(region)}
    return signs.pop() if len(signs) == 1 else None


def quadrant_mask(region: str | None, above_x: np.ndarray, above_y: np.ndarray) -> np.ndarray:
    """Events of a 2dsep region given which events are at or above each
    threshold."""
    out = np.zeros(len(above_x), dtype=bool)
    for q in region_quadrants(region):
        mx = above_x if q[1] == '+' else ~above_x
        my = above_y if q[3] == '+' else ~above_y
        out |= mx & my
    return out


def threshold_sides(gate_def: dict, axis: str) -> set:
    """Sides ('-', '+') of the other axis on which a 2dsep gate's threshold
    on *axis* ('x' or 'y') separates two populations, or a population from
    none (a joined region needs it on one side only)."""
    owner = {}
    for pop, region in population_regions(gate_def).items():
        for q in region_quadrants(region):
            owner[q] = pop
    out = set()
    for s in '-+':
        pair = (f'x-y{s}', f'x+y{s}') if axis == 'x' else (f'x{s}y-', f'x{s}y+')
        if owner.get(pair[0]) != owner.get(pair[1]):
            out.add(s)
    return out


def region_polygon(region: str | None, tx: float, ty: float, x_lo: float, x_hi: float,
                   y_lo: float, y_hi: float) -> list:
    """Closed outline of a 2dsep region's quadrants within the limits: a
    rectangle for one quadrant or a side, an L for three quadrants, two
    rectangles meeting at the threshold crossing for opposite quadrants."""
    quads = set(region_quadrants(region))
    xs = {'-': (x_lo, tx), '+': (tx, x_hi)}
    ys = {'-': (y_lo, ty), '+': (ty, y_hi)}
    if len(quads) == 4:
        return _rect(x_lo, x_hi, y_lo, y_hi)
    if len(quads) == 1:
        q = next(iter(quads))
        return _rect(*xs[q[1]], *ys[q[3]])
    if len(quads) == 2:
        sx, sy = {q[1] for q in quads}, {q[3] for q in quads}
        if len(sx) == 1:
            return _rect(*xs[sx.pop()], y_lo, y_hi)
        if len(sy) == 1:
            return _rect(x_lo, x_hi, *ys[sy.pop()])
        a, b = sorted(quads)
        ra, rb = _rect(*xs[a[1]], *ys[a[3]]), _rect(*xs[b[1]], *ys[b[3]])
        return _opposite_outline(ra, rb, tx, ty)
    missing = (set(QUADRANT_REGIONS) - quads).pop() if len(quads) == 3 else None
    if missing is None:
        return []
    cx = x_lo if missing[1] == '-' else x_hi
    cy = y_lo if missing[3] == '-' else y_hi
    ox = x_hi if missing[1] == '-' else x_lo
    oy = y_hi if missing[3] == '-' else y_lo
    pts = [[tx, cy], [ox, cy], [ox, oy], [cx, oy], [cx, ty], [tx, ty], [tx, cy]]
    return [[float(a), float(b)] for a, b in pts]


def _opposite_outline(ra: list, rb: list, tx: float, ty: float) -> list:
    """Outline of two rectangles touching only at (tx, ty), traced through
    that point."""
    def from_corner(r):
        pts = r[:-1]
        k = min(range(len(pts)), key=lambda i: abs(pts[i][0] - tx) + abs(pts[i][1] - ty))
        return pts[k:] + pts[:k]
    a, b = from_corner(ra), from_corner(rb)
    pts = a + [a[0]] + b[1:] + [b[0]]
    return [[float(x), float(y)] for x, y in pts]

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


def axis_algorithm(gate_def: dict, axis: str) -> str:
    """Algorithm of one axis ('x' or 'y') of a threshold gate: the per-axis
    ``algorithm_x``/``algorithm_y`` of a 2dsep gate, else the gate's."""
    default = gate_def.get('algorithm') or 'tail'
    if gate_def.get('gate_type') != '2dsep':
        return default
    return (gate_def.get('gate_param') or {}).get(f'algorithm_{axis}') or default


def threshold_axes(gate_def: dict) -> tuple:
    """The data axes a threshold gate separates on: ('x',) or ('x', 'y')."""
    return ('x', 'y') if gate_def.get('gate_type') == '2dsep' else ('x',)


def axis_channel(gate_def: dict, axis: str):
    return gate_def.get('gate_marker_x' if axis == 'x' else 'gate_marker_y')


def replicate_sources(gate_def: dict) -> dict:
    """{axis: source} for the axes of a threshold gate that replicate a
    threshold (see the module notes); a missing source is None."""
    if gate_def.get('gate_type') not in THRESHOLD_TYPES \
            or gate_def.get('algorithm') == ALGORITHM_IMPORTED:
        return {}
    gp = gate_def.get('gate_param') or {}
    out = {}
    for axis in threshold_axes(gate_def):
        if axis_algorithm(gate_def, axis) == ALGORITHM_REPLICATE:
            src = gp.get(f'replicate_{axis}')
            out[axis] = src if isinstance(src, dict) and (src.get('gate') or src.get('population')) \
                else None
    return out


def gate_dependencies(gate_def: dict) -> set:
    """Names of the gates that must be calculated before *gate_def*: its
    parent, a replicate gate's origin, and the gates or reference
    populations its replicated thresholds come from."""
    deps = {d for d in (gate_def.get('parent_gate'), gate_def.get('origin_gate')) if d}
    for src in replicate_sources(gate_def).values():
        if not src:
            continue
        if src.get('gate'):
            deps.add(src['gate'])
        elif src.get('population'):
            deps.add(split_population_key(src['population'])[0])
    return deps


def reference_populations(gate_defs: list[dict]) -> list[str]:
    """'gate/pop' keys of every population marked as a reference."""
    out = []
    for g in gate_defs:
        for pop, meta in (g.get('populations') or {}).items():
            if (meta or {}).get('reference'):
                out.append(population_key(g['gate_name'], pop))
    return out


ANCHOR_FLAG = 'anchor'


def viability_channels(channels) -> set:
    """Channels whose fluorophore is a viability dye (``is.viability`` in
    Honeychrome's fluorophore database, matched by name and synonyms)."""
    return set(_viability_lookup(tuple(sorted({str(c) for c in channels if c}))))


_VIABILITY_CACHE: dict = {}


def _viability_lookup(channels: tuple) -> tuple:
    if channels in _VIABILITY_CACHE:
        return _VIABILITY_CACHE[channels]
    try:
        from honeychrome.controller_components.label_matching import (
            get_fluorophore_db, match_fluorophore)
        db = get_fluorophore_db()
    except Exception:
        return ()
    viable = {r.get('fluorophore') for r in db
              if str(r.get('is.viability', '')).strip().upper() == 'TRUE'}
    out = tuple(ch for ch in channels if match_fluorophore(ch, db) in viable)
    _VIABILITY_CACHE[channels] = out
    return out


def anchor_population(gate_defs: list[dict]) -> str | None:
    """'gate/pop' key of the anchor population: the one marked with
    ANCHOR_FLAG; else the viable population of the deepest gate using a
    viability dye channel (its population on the dye's negative side, or
    its only population); else the deepest singlets population; else the
    population labelled 'Cells'. None when there is none of these."""
    for g in gate_defs:
        for pop, meta in (g.get('populations') or {}).items():
            if (meta or {}).get(ANCHOR_FLAG):
                return population_key(g['gate_name'], pop)
    ordered = ordered_gate_defs(gate_defs)
    by_name = gates_by_name(gate_defs)
    depth: dict = {}
    for g in ordered:
        parent = g.get('parent_gate')
        depth[g['gate_name']] = depth.get(parent, -1) + 1 if parent in by_name else 0
    used = set()
    for g in gate_defs:
        used |= set(referenced_channels(g))
    viability = viability_channels(used)

    def deepest(gates):
        return max(gates, key=lambda g: depth.get(g['gate_name'], 0)) if gates else None

    via = deepest([g for g in ordered if viability & set(referenced_channels(g))])
    if via is not None:
        pops = list(population_regions(via).items())
        if len(pops) == 1:
            return population_key(via['gate_name'], pops[0][0])
        dye = next(c for c in referenced_channels(via) if c in viability)
        for pop, region in pops:
            if via.get('gate_type') == '1dsep' and via.get('gate_marker_x') == dye \
                    and region == REGION_BELOW:
                return population_key(via['gate_name'], pop)
            if via.get('gate_type') == '2dsep':
                axis = 'x' if via.get('gate_marker_x') == dye else 'y'
                if region_sign(region, axis) == '-':
                    return population_key(via['gate_name'], pop)
        return population_key(via['gate_name'], pops[0][0])
    singlets = deepest([g for g in ordered if g.get('gate_type') == 'singlets'])
    if singlets is not None:
        return population_key(singlets['gate_name'], next(iter(singlets.get('populations') or {'': 0})))
    for g in ordered:
        for pop, meta in (g.get('populations') or {}).items():
            if str((meta or {}).get('label') or pop).strip().lower() == 'cells':
                return population_key(g['gate_name'], pop)
    return None


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
    """Gaussian kernel density of *x* (Scott's bandwidth, as scipy's
    gaussian_kde) at *n_grid* points spanning its range.

    Up to KDE_EXACT_MAX_EVENTS events it is scipy's gaussian_kde. Beyond
    that the events are linearly binned on a grid at least eight times
    finer than the output (and fine enough for four points per
    bandwidth), and the bin weights convolved with the kernel, out to
    where it underflows as scipy's does. The density then agrees with
    gaussian_kde to about 1e-5 of its peak at a cost linear in the number
    of events rather than proportional to events × grid points (about
    10 ms instead of 8 s for a million events).
    """
    x = np.asarray(x, dtype=float)
    grid = np.linspace(float(x.min()), float(x.max()), n_grid)
    if len(x) > KDE_EXACT_MAX_EVENTS and np.isfinite(x).all():
        density = _binned_kde(x, grid)
        if density is not None:
            return grid, density
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(x, bw_method='scott')
    return grid, kde(grid)


def _binned_kde(x: np.ndarray, grid: np.ndarray, max_refine: int = 2048):
    """Binned Gaussian kernel density of *x* at the evenly spaced *grid*
    (see _kde_grid); None when the bandwidth is too small for the range."""
    n = len(x)
    lo, hi = float(grid[0]), float(grid[-1])
    bandwidth = float(np.std(x, ddof=1)) * n ** -0.2
    if not bandwidth > 0 or hi <= lo:
        return None
    n_grid = len(grid)
    refine = max(8, int(np.ceil(4.0 * (hi - lo) / ((n_grid - 1) * bandwidth))))
    if refine > max_refine:
        return None
    m = (n_grid - 1) * refine + 1
    step = (hi - lo) / (m - 1)
    pos = (x - lo) / step
    left = np.clip(np.floor(pos).astype(np.int64), 0, m - 2)
    frac = pos - left
    weights = (np.bincount(left, weights=1.0 - frac, minlength=m)
               + np.bincount(left + 1, weights=frac, minlength=m))
    # Kernel out to 37.5 bandwidths, where exp() underflows to 0, or across
    # the whole grid if that is shorter
    half = min(int(np.ceil(37.5 * bandwidth / step)), m - 1)
    offsets = np.arange(-half, half + 1) * step
    kernel = np.exp(-0.5 * (offsets / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi) * n)
    density = np.convolve(weights, kernel)[half:half + m]
    return density[::refine]


def kde_mode(x: np.ndarray) -> float:
    """Location of the highest kernel-density peak."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or float(x.max()) == float(x.min()):
        return float(np.median(x)) if len(x) else 0.0
    grid, density = _kde_grid(x)
    return float(grid[int(np.argmax(density))])


def threshold_tail(x: np.ndarray, tail_fraction: float = DEFAULT_TAIL_FRACTION) -> float:
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

    Falls back to ``threshold_tail`` (KDE_MIN_FALLBACK_TAIL) when the
    density has a single peak.
    """
    from scipy.signal import find_peaks
    x = np.asarray(x, dtype=float)
    if len(x) < 20 or float(x.max()) == float(x.min()):
        return float(np.median(x)) if len(x) else 0.0
    grid, density = _kde_grid(x)
    peaks, _ = find_peaks(density)
    if len(peaks) < 2:
        return threshold_tail(x, tail_fraction=KDE_MIN_FALLBACK_TAIL)
    top_two = sorted(peaks[np.argsort(density[peaks])[::-1][:2]])
    p1, p2 = int(top_two[0]), int(top_two[1])
    valley = p1 + int(np.argmin(density[p1:p2 + 1]))
    return float(grid[valley])


MIXTURE_CUTS = ('outer', 'lower', 'upper')


def threshold_mixture(x: np.ndarray, n_components: int = 2, seed: int = 0,
                      cut: str = 'outer') -> float:
    """Density crossover between two components of a Gaussian mixture,
    ordered by mean: the lowest and highest (*cut* 'outer'), the lowest
    two ('lower') or the highest two ('upper'). With two components the
    three are the same."""
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
    lo, hi = {'lower': (0, 1), 'upper': (-2, -1)}.get(cut, (0, -1))
    mu1, mu2 = means[order[lo]], means[order[hi]]
    w1, w2 = weights[order[lo]], weights[order[hi]]
    s1, s2 = stds[order[lo]], stds[order[hi]]

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
    tf = float(params.get('tail_fraction', DEFAULT_TAIL_FRACTION))
    nc = int(params.get('mixture_components', 2))
    if algorithm == 'kde_min':
        return threshold_kde_min(x)
    if algorithm == 'mixture':
        return threshold_mixture(x, n_components=nc, cut=str(params.get('mixture_cut', 'outer')))
    if algorithm == 'otsu':
        return threshold_otsu(x)
    if algorithm == 'bimodal':
        return threshold_bimodal(x)
    return threshold_tail(x, tail_fraction=tf)


def threshold_factor(gate_param: dict | None, suffix: str = '') -> float:
    """Threshold factor of one axis (``threshold_factor`` + *suffix*, falling
    back to ``threshold_factor``)."""
    gp = gate_param or {}
    try:
        return float(gp.get(f'threshold_factor{suffix}',
                            gp.get('threshold_factor', DEFAULT_THRESHOLD_FACTOR)))
    except (TypeError, ValueError):
        return DEFAULT_THRESHOLD_FACTOR


def _xform(transform):
    """Object with apply()/inverse() for a Honeychrome Transform, a transform
    parameter dict or such an object; None for the identity."""
    if transform is None:
        return None
    if isinstance(transform, dict):
        return make_transform(transform)
    xform = getattr(transform, 'xform', None)
    if xform is not None:
        return xform
    if hasattr(transform, 'apply') and hasattr(transform, 'inverse'):
        return transform
    return None


def scale_threshold(t: float, factor: float, transform=None) -> tuple[float, str]:
    """*t* (display units) multiplied by *factor* in raw units.

    Returns (threshold, note); the note is non-empty when the factor could
    not be applied (a threshold at or below zero in raw units).
    """
    f = float(factor)
    if f == 1.0 or not np.isfinite(f) or f <= 0:
        return float(t), ''
    xf = _xform(transform)
    raw = float(np.asarray(xf.inverse(np.array([float(t)])), dtype=float)[0]) if xf else float(t)
    if not raw > 0:
        return float(t), (f"threshold {raw:.4g} is not above zero in raw units; "
                          f"factor {f:g} not applied")
    new = raw * f
    return (float(np.asarray(xf.apply(np.array([new])), dtype=float)[0]) if xf else new), ''


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
    """Gate definitions ordered so every gate follows the gates it depends
    on (see gate_dependencies).

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

    ranked = sorted(gate_defs, key=sort_key)
    waiting: dict[str, list[int]] = {}
    unmet: list[set[str]] = []
    ready: list[int] = []
    for rank, g in enumerate(ranked):
        deps = gate_dependencies(g) & set(by_name)
        unmet.append(deps)
        if not deps:
            ready.append(rank)
        for dep in deps:
            waiting.setdefault(dep, []).append(rank)
    heapq.heapify(ready)
    placed: set[str] = set()
    done = [False] * len(ranked)
    out: list[dict] = []
    while ready:
        rank = heapq.heappop(ready)
        g = ranked[rank]
        done[rank] = True
        out.append(g)
        name = g['gate_name']
        if name in placed:
            continue
        placed.add(name)
        for dependent in waiting.get(name, ()):
            unmet[dependent].discard(name)
            if not unmet[dependent]:
                heapq.heappush(ready, dependent)
    if len(out) < len(ranked):
        names = [g.get('gate_name') for rank, g in enumerate(ranked) if not done[rank]]
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


def separation_axis(gate_def: dict) -> str:
    """Plot axis ('x' or 'y') a 1dsep gate's separation channel is drawn on.
    Always 'x' for other gate types and for histograms."""
    if gate_def.get('gate_type') != '1dsep' or not gate_def.get('gate_marker_y'):
        return 'x'
    return 'y' if gate_def.get('sep_axis') == 'y' else 'x'


def plot_channels(gate_def: dict) -> tuple:
    """(x channel, y channel or None) of the plot a gate is drawn on."""
    x = gate_def.get('gate_marker_x') or None
    y = gate_def.get('gate_marker_y') or None
    if separation_axis(gate_def) == 'y':
        return y, x
    return x, y


def is_shape_boundary(boundary: dict | None) -> bool:
    """True when boundary entries are shapes as drawn (see module notes)."""
    return bool(boundary) and all(
        isinstance(e, dict) and e.get('shape_channels') for e in boundary.values())


def shape_channels(gate_def: dict) -> tuple:
    """Channels of the plot an imported gate's template was drawn on, or ()."""
    template = gate_def.get('template') or {}
    if not is_shape_boundary(template):
        return ()
    return tuple(c for c in next(iter(template.values()))['shape_channels'] if c)


def boundary_channels(gate_def: dict, boundary: dict | None) -> list[str]:
    """Channels *boundary* is applied on: a shape's own channels, else the
    gate's (see gate_channels)."""
    if is_shape_boundary(boundary):
        return [c for c in next(iter(boundary.values()))['shape_channels'] if c]
    return gate_channels(gate_def)


def boundary_plot_channels(gate_def: dict, boundary: dict | None) -> tuple:
    """(x channel, y channel or None) to draw *boundary* on: a shape's own
    channels, else the gate's plot axes (see plot_channels)."""
    if is_shape_boundary(boundary):
        chans = [c for c in next(iter(boundary.values()))['shape_channels'] if c]
        return chans[0], (chans[1] if len(chans) > 1 else None)
    return plot_channels(gate_def)


def referenced_channels(gate_def: dict) -> list[str]:
    """Every channel a gate reads or is drawn on, including a 1dsep view
    channel and the channels of an imported shape."""
    out = list(gate_channels(gate_def))
    for ch in (gate_def.get('gate_marker_y'),) + shape_channels(gate_def):
        if ch and ch not in out:
            out.append(ch)
    return out


def display_entries(gate_def: dict, entries: dict) -> dict:
    """Boundary entries in plot coordinates (see plot_channels): a 1dsep
    gate drawn with its separation channel on the y axis has its threshold
    as ``threshold_y`` and its polygons transposed. Shapes as drawn and
    every other gate are returned unchanged."""
    if separation_axis(gate_def) != 'y' or is_shape_boundary(entries):
        return entries
    out = {}
    for pop, entry in (entries or {}).items():
        e = dict(entry)
        e['threshold_y'], e['threshold_x'] = entry.get('threshold_x'), None
        coords = np.asarray(entry.get('boundary') or [], dtype=float)
        if coords.ndim == 2 and coords.shape[1] == 2:
            e['boundary'] = coords[:, ::-1].tolist()
        out[pop] = e
    return out


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
    transforms  : {channel: Transform or parameter dict} the data are in,
                  for threshold factors (raw units). Missing channels are
                  untransformed.
    notes       : messages from the last ``calculate`` (factors not applied).
    """

    def __init__(self, axis_limits: dict | None = None, transforms: dict | None = None):
        self.axis_limits = dict(axis_limits or {})
        self.transforms = dict(transforms or {})
        self.notes: list[str] = []

    def _scaled(self, t: float, gate_param: dict, suffix: str, channel: str) -> float:
        t, note = scale_threshold(t, threshold_factor(gate_param, suffix),
                                  self.transforms.get(channel))
        if note:
            self.notes.append(f"{channel}: {note}")
        return t

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
        self.notes = []
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
        """The boundary stored in the gate's ``template``. Shapes as drawn
        are kept as they are; threshold gates keep their thresholds and
        redraw their rectangles to the axis limits."""
        template = gate_def.get('template') or {}
        if not template:
            return {}
        if is_shape_boundary(template):
            if any(ch not in channel_index for ch in shape_channels(gate_def)):
                return {}
            return deepcopy(template)
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
        # The band runs the full width of the axis, so the gate is open along
        # the diagonal rather than ending at the extremes of the training data.
        lim_lo, lim_hi = self._limits(gate_def['gate_marker_x'], x)
        x_lo, x_hi = min(float(x.min()), lim_lo), max(float(x.max()), lim_hi)
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
        fixed = gp.get('fixed_threshold_x')
        if fixed is not None:
            # A replicated threshold needs no events of this population.
            return self.threshold_boundaries(gate_def, float(fixed), None, full, None)
        if gate_def.get('algorithm') == ALGORITHM_REPLICATE:
            return {}
        col = trim_quantiles(full, q)
        if len(col) < MIN_EVENTS:
            return {}
        algo = gate_def.get('algorithm', 'tail')
        calc = crop_to_region(col, float(gp.get('axis_lo', default_axis_lo(algo))),
                              float(gp.get('axis_hi', 0.99)))
        if len(calc) < MIN_EVENTS:
            calc = col
        t = self._scaled(compute_threshold(calc, algo, gp), gp, '', ch)
        return self.threshold_boundaries(gate_def, t, None, full, None)

    def _calc_2dsep(self, data, gate_def, channel_index) -> dict:
        gp = gate_def.get('gate_param', {}) or {}
        q = float(gp.get('quantile_trim', 0.001))
        ix = channel_index[gate_def['gate_marker_x']]
        iy = channel_index[gate_def['gate_marker_y']]
        xy = trim_quantiles(data[:, [ix, iy]], q)
        both_fixed = all(gp.get(f'fixed_threshold_{a}') is not None for a in ('x', 'y'))
        if len(xy) < 20 and not both_fixed:
            return {}
        default_algo = gate_def.get('algorithm', 'tail')

        def axis_params(axis: str) -> dict:
            p = dict(gp)
            p['tail_fraction'] = float(gp.get(f'tail_fraction_{axis}',
                                              gp.get('tail_fraction', DEFAULT_TAIL_FRACTION)))
            p['mixture_components'] = int(gp.get(f'mixture_components_{axis}',
                                                 gp.get('mixture_components', 2)))
            p['mixture_cut'] = gp.get(f'mixture_cut_{axis}', gp.get('mixture_cut', 'outer'))
            return p

        thresholds = []
        for k, axis in enumerate(('x', 'y')):
            fixed = gp.get(f'fixed_threshold_{axis}')
            if fixed is not None:
                thresholds.append(float(fixed))
                continue
            col = xy[:, k]
            algo = gp.get(f'algorithm_{axis}') or default_algo
            if algo == ALGORITHM_REPLICATE:
                return {}
            calc = crop_to_region(col, float(gp.get(f'axis_lo_{axis}', default_axis_lo(algo))),
                                  float(gp.get(f'axis_hi_{axis}', 0.99)))
            if len(calc) < MIN_EVENTS:
                calc = col
            t = compute_threshold(calc, algo, axis_params(axis))
            thresholds.append(self._scaled(t, gp, f'_{axis}', axis_channel(gate_def, axis)))
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
            out[pop] = {'boundary': region_polygon(region, tx, ty, x_lo, x_hi, y_lo, y_hi),
                        'threshold_x': float(tx), 'threshold_y': float(ty), 'region': region}
        return out

    def _calc_free(self, data, gate_def, channel_index) -> dict:
        gp = gate_def.get('gate_param', {}) or {}
        q = float(gp.get('quantile_trim', 0.001))
        ix = channel_index[gate_def['gate_marker_x']]
        pop = next(iter(gate_def.get('populations') or {'in': {}}))
        y_ch = gate_def.get('gate_marker_y')
        if gate_def.get('algorithm') == 'cells' and y_ch:
            return self._calc_cells(data, gate_def, channel_index, pop)
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
            poly = _ellipse_polygon(cols, n_points=ELLIPSE_VERTICES)
        else:
            poly = _convex_hull_polygon(cols)
        if poly is None:
            return {}
        return {pop: {'boundary': poly, 'threshold_x': None, 'threshold_y': None,
                      'region': REGION_INSIDE}}

    def _calc_cells(self, data, gate_def, channel_index, pop) -> dict:
        """Cell gate on a pair of scatter channels: a rectangle from the
        debris cut on the forward-scatter axis to where the cells' density
        ends on each side.

        The debris cut is the density valley left of the highest peak of the
        forward-scatter axis (``gate_param['debris_axis']``, 'x' or 'y'),
        or, when no peak lies left of it, the point where the density left of
        the peak first falls below ``debris_density`` (default 0.05) of the
        peak height. On the events above the cut, the other three sides are
        the outermost points where each axis's density is still at least
        its fall-off (``edge_density_x`` / ``edge_density_y``, else
        ``edge_density``, default DEFAULT_EDGE_DENSITY) of its peak
        (density_edges). ``cell_side`` CELL_SIDE_DEBRIS keeps the events
        below the cut instead (a debris gate): the cut is then the upper
        side on the forward-scatter axis and the other sides come from the
        events below it. A pile of events at the top of either axis
        (saturation, more than ``saturation_fraction`` of the events,
        default 0.0005) is cut just below the pile.
        """
        gp = gate_def.get('gate_param', {}) or {}
        q = float(gp.get('quantile_trim', 0.001))
        chans = [gate_def['gate_marker_x'], gate_def['gate_marker_y']]
        xy = data[:, [channel_index[c] for c in chans]]
        if len(xy) < 50:
            return {}
        d = 1 if gp.get('debris_axis') == 'y' else 0
        col = trim_quantiles(xy[:, d], q)
        cut = debris_cut(col, float(gp.get('debris_density', 0.05)))
        debris = gp.get('cell_side') == CELL_SIDE_DEBRIS
        kept = xy[xy[:, d] < cut] if debris else xy[xy[:, d] >= cut]
        if len(kept) < 50:
            kept = xy
        bounds = []
        for k, ch in enumerate(chans):
            edge = float(gp.get(f'edge_density_{"xy"[k]}',
                                gp.get('edge_density', DEFAULT_EDGE_DENSITY)))
            values = xy[:, k]
            lo, hi = density_edges(trim_quantiles(kept[:, k], q), edge)
            if k == d:
                if debris:
                    hi = cut
                else:
                    lo = cut
            pile = saturation_cut(values, float(gp.get('saturation_fraction', 0.0005)))
            if pile is not None:
                hi = min(hi, pile)
            bounds.append((lo, hi))
        (x0, x1), (y0, y1) = bounds
        return {pop: {'boundary': _rect(x0, x1, y0, y1), 'threshold_x': None,
                      'threshold_y': None, 'region': REGION_INSIDE}}


def debris_cut(x: np.ndarray, rel_density: float = 0.05) -> float:
    """Forward-scatter value separating debris from cells (see
    GateCalculator._calc_cells)."""
    from scipy.signal import find_peaks
    x = np.asarray(x, dtype=float)
    if len(x) < 20 or float(x.max()) == float(x.min()):
        return float(x.min()) if len(x) else 0.0
    grid, density = _kde_grid(x)
    top = int(np.argmax(density))
    peaks, _ = find_peaks(density, prominence=0.05 * float(density[top]))
    left = [int(p) for p in peaks if p < top]
    if left:
        start = max(left)
        return float(grid[start + int(np.argmin(density[start:top + 1]))])
    below = np.flatnonzero(density[:top] < rel_density * float(density[top]))
    return float(grid[below[-1]]) if len(below) else float(grid[0])


def density_edges(x: np.ndarray, rel_density: float = DEFAULT_EDGE_DENSITY) -> tuple[float, float]:
    """(lo, hi): the lowest and highest points where the density of *x* is
    at least *rel_density* of its peak height (the extent of the events,
    leaving out sparse tails; a second population above that height is
    kept)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 20 or float(x.max()) == float(x.min()):
        return (float(x.min()), float(x.max())) if len(x) else (0.0, 0.0)
    grid, density = _kde_grid(x)
    keep = np.flatnonzero(density >= rel_density * float(density.max()))
    return float(grid[keep[0]]), float(grid[keep[-1]])


def saturation_cut(values: np.ndarray, min_fraction: float = 0.0005) -> float | None:
    """Just below a pile of events at the maximum of *values* (a saturated
    detector), or None when there is no such pile."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    v_max, v_min = float(values.max()), float(values.min())
    span = (v_max - v_min) or 1.0
    edge = v_max - 1e-3 * span
    if float(np.mean(values >= edge)) <= min_fraction:
        return None
    return edge - 1e-3 * span


def _convex_hull_polygon(xy: np.ndarray) -> list | None:
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(xy)
    except Exception:
        return None
    verts = [[float(a), float(b)] for a, b in xy[hull.vertices]]
    verts.append(verts[0])
    return verts


ELLIPSE_VERTICES = 10


def ellipse_area_scale(n_points: int) -> float:
    """Radius factor giving an n-vertex polygon the area of its ellipse."""
    return math.sqrt(2.0 * math.pi / (n_points * math.sin(2.0 * math.pi / n_points)))


def _ellipse_polygon(xy: np.ndarray, n_points: int = ELLIPSE_VERTICES,
                     n_sd: float = 3.0) -> list | None:
    """The 3 SD ellipse of *xy* as an n-vertex polygon of the same area
    (models saved with the retired 'ellipse' algorithm)."""
    try:
        cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
        vals, vecs = np.linalg.eigh(np.cov(xy.T))
    except Exception:
        return None
    k = ellipse_area_scale(n_points)
    a = k * n_sd * math.sqrt(abs(vals[0]))
    b = k * n_sd * math.sqrt(abs(vals[1]))
    angle = math.atan2(vecs[1, 0], vecs[0, 0])
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    xr = a * np.cos(t) * math.cos(angle) - b * np.sin(t) * math.sin(angle) + cx
    yr = a * np.cos(t) * math.sin(angle) + b * np.sin(t) * math.cos(angle) + cy
    poly = [[float(p), float(q)] for p, q in zip(xr, yr)]
    poly.append(poly[0])
    return poly


def ellipse_as_polygon(coords, n_points: int = ELLIPSE_VERTICES) -> list:
    """The ellipse traced by the closed polygon *coords* as an n-vertex
    polygon of the same area, vertices spaced evenly in the ellipse's own
    angle: used to turn stored ellipses into 10-vertex polygons.

    The ellipse is recovered from the polygon's area moments (centroid and
    second moments), which a densely traced ellipse matches exactly.
    """
    pts = np.asarray(coords, dtype=float)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 3:
        return [list(map(float, p)) for p in coords]
    x, y = pts[:, 0], pts[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-15:
        return [list(map(float, p)) for p in coords]
    cx = ((x + xn) * cross).sum() / (6.0 * area)
    cy = ((y + yn) * cross).sum() / (6.0 * area)
    sxx = ((x * x + x * xn + xn * xn) * cross).sum() / (12.0 * area) - cx * cx
    syy = ((y * y + y * yn + yn * yn) * cross).sum() / (12.0 * area) - cy * cy
    sxy = ((x * yn + 2 * x * y + 2 * xn * yn + xn * y) * cross).sum() / (24.0 * area) - cx * cy
    vals, vecs = np.linalg.eigh(np.array([[sxx, sxy], [sxy, syy]]))
    # A uniform ellipse with semi-axes a, b has second moments a²/4 and b²/4.
    axes = 2.0 * np.sqrt(np.clip(vals, 0.0, None)) * ellipse_area_scale(n_points)
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    out = [[float(cx + axes[0] * math.cos(a) * vecs[0, 0] + axes[1] * math.sin(a) * vecs[0, 1]),
            float(cy + axes[0] * math.cos(a) * vecs[1, 0] + axes[1] * math.sin(a) * vecs[1, 1])]
           for a in t]
    out.append(list(out[0]))
    return out


# ---------------------------------------------------------------------------
# Replicated thresholds
# ---------------------------------------------------------------------------

def _threshold_on_channel(gate_def: dict, entries: dict, channel: str):
    """The threshold a calculated threshold gate has on *channel*, or None."""
    if not entries or is_shape_boundary(entries) \
            or gate_def.get('gate_type') not in THRESHOLD_TYPES:
        return None
    first = next(iter(entries.values()))
    if gate_def.get('gate_marker_x') == channel and first.get('threshold_x') is not None:
        return float(first['threshold_x'])
    if gate_def.get('gate_type') == '2dsep' and gate_def.get('gate_marker_y') == channel \
            and first.get('threshold_y') is not None:
        return float(first['threshold_y'])
    return None


def resolve_replicates(gate_def: dict, data: np.ndarray, channel_index: dict,
                       result: 'GatingResult', by_name: dict,
                       transforms: dict | None = None,
                       notes: list | None = None) -> tuple[dict, str]:
    """Copy of *gate_def* (an effective definition) with the thresholds of
    its replicated axes fixed in ``gate_param['fixed_threshold_x/y']``.

    *result* holds the gates applied so far to *data* (boundaries and
    population masks over its rows). Returns (definition, '') or
    (None, reason) when a source is missing or cannot be calculated.
    A threshold calculated on a reference population is multiplied by the
    axis's threshold factor (raw units, *transforms*); one copied from a
    gate is copied as it is. Factors not applied are added to *notes*.
    """
    sources = replicate_sources(gate_def)
    if not sources:
        return gate_def, ''
    out = deepcopy(gate_def)
    gp = out.setdefault('gate_param', {})
    for axis, src in sources.items():
        channel = axis_channel(gate_def, axis)
        if not src:
            return None, f"no replicate source chosen for {channel}"
        if src.get('gate'):
            origin = by_name.get(src['gate'])
            if origin is None:
                return None, f"replicate source gate '{src['gate']}' does not exist"
            eff = effective_gate_def(origin, by_name)
            t = _threshold_on_channel(eff, result.boundaries.get(src['gate']), channel)
            if t is None:
                return None, (f"gate '{src['gate']}' has no threshold on {channel} "
                              "to replicate")
        else:
            key = src['population']
            mask = result.population_mask(*split_population_key(key))
            if mask is None or channel not in channel_index:
                return None, f"reference population {key} is not available"
            values = trim_quantiles(np.asarray(data)[mask, channel_index[channel]],
                                    float(gp.get('quantile_trim', 0.001)))
            if len(values) < MIN_EVENTS:
                return None, f"reference population {key} has too few events"
            suffix = f'_{axis}' if gate_def.get('gate_type') == '2dsep' else ''
            algo = src.get('algorithm') if src.get('algorithm') in THRESHOLD_ALGORITHMS else 'tail'
            calc = crop_to_region(values, float(gp.get(f'axis_lo{suffix}', default_axis_lo(algo))),
                                  float(gp.get(f'axis_hi{suffix}', 0.99)))
            if len(calc) < MIN_EVENTS:
                calc = values
            params = {'tail_fraction': gp.get(f'tail_fraction{suffix}',
                                              gp.get('tail_fraction', DEFAULT_TAIL_FRACTION)),
                      'mixture_components': gp.get(f'mixture_components{suffix}',
                                                   gp.get('mixture_components', 2)),
                      'mixture_cut': gp.get(f'mixture_cut{suffix}', gp.get('mixture_cut', 'outer'))}
            t, note = scale_threshold(compute_threshold(calc, algo, params),
                                      threshold_factor(gp, suffix),
                                      (transforms or {}).get(channel))
            if note and notes is not None:
                notes.append(f"{channel}: {note}")
        gp[f'fixed_threshold_{axis}'] = float(t)
    return out, ''


# ---------------------------------------------------------------------------
# Gate application
# ---------------------------------------------------------------------------

def _edge_clamped(values: np.ndarray, limits) -> np.ndarray:
    """*values* with off-scale events moved just inside the display limits,
    where the plots draw them."""
    if limits is None or len(limits) < 2:
        return values
    lo, hi = float(limits[0]), float(limits[1])
    if not hi > lo:
        return values
    eps = 1e-6 * (hi - lo)
    return np.clip(values, lo + eps, hi - eps)

def population_masks(gate_def: dict, boundary: dict, data: np.ndarray,
                     channel_index: dict, axis_limits: dict | None = None) -> dict[str, np.ndarray]:
    """Boolean masks over *data* rows for each population of one gate.

    *gate_def* must be the effective definition (see effective_gate_def).
    axis_limits : {channel: (lo, hi)} display limits; polygons and ranges
                  test off-scale events on the axis edge (see the module
                  notes). Thresholds are open-ended and ignore them.
    Returns {} when a channel is missing or the boundary is empty.
    """
    if not boundary:
        return {}
    chans = gate_channels(gate_def)
    if any(ch not in channel_index for ch in chans):
        return {}
    limits = axis_limits or {}
    if is_shape_boundary(boundary):
        return _shape_masks(boundary, data, channel_index, limits)
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
            masks[pop] = quadrant_mask(regions.get(pop, 'x+y+'), above_x, above_y)
        return masks

    from matplotlib.path import Path as MplPath
    x = _edge_clamped(x, limits.get(chans[0])) if chans else x
    y = (_edge_clamped(data[:, channel_index[chans[1]]], limits.get(chans[1]))
         if len(chans) > 1 else None)
    for pop, entry in boundary.items():
        rng = entry.get('range')
        if rng is not None and len(chans) == 1:
            masks[pop] = (x >= float(rng[0])) & (x <= float(rng[1]))
            continue
        coords = np.asarray(entry.get('boundary') or [], dtype=float)
        if coords.ndim != 2 or coords.shape[0] < 3 or y is None:
            masks[pop] = np.zeros(n, dtype=bool)
            continue
        masks[pop] = MplPath(coords).contains_points(np.column_stack([x, y]))
    return masks


def _shape_masks(boundary: dict, data: np.ndarray, channel_index: dict,
                 axis_limits: dict | None = None) -> dict[str, np.ndarray]:
    """Masks of shapes as drawn: inside each polygon, or within each range,
    on the entry's own ``shape_channels``, with off-scale events on the
    axis edge (see population_masks). {} when a channel is missing."""
    from matplotlib.path import Path as MplPath
    limits = axis_limits or {}
    n = len(data)
    masks: dict[str, np.ndarray] = {}
    for pop, entry in boundary.items():
        chans = [c for c in entry['shape_channels'] if c]
        if any(c not in channel_index for c in chans):
            return {}
        x = _edge_clamped(data[:, channel_index[chans[0]]], limits.get(chans[0]))
        if len(chans) == 1:
            rng = entry.get('range')
            masks[pop] = ((x >= float(rng[0])) & (x <= float(rng[1]))
                          if rng is not None else np.zeros(n, dtype=bool))
            continue
        coords = np.asarray(entry.get('boundary') or [], dtype=float)
        if coords.ndim != 2 or coords.shape[0] < 3:
            masks[pop] = np.zeros(n, dtype=bool)
            continue
        y = _edge_clamped(data[:, channel_index[chans[1]]], limits.get(chans[1]))
        masks[pop] = MplPath(coords).contains_points(np.column_stack([x, y]))
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


class GatingStopped(Exception):
    """Raised by run_gating when its stop_check asks it to stop."""


@dataclass
class GatingResult:
    """Outcome of running a gate hierarchy on one event array.

    boundaries   : {gate: {pop: entry}} as applied, with n_events/fraction.
    masks        : {gate: {pop: bool mask over the array's rows}}.
    parent_masks : {gate: bool mask of the events the gate was applied to}.
    status       : {gate: {'source', 'drift', 'flag', 'message'}}; source is
                   'trained', 'fixed', 'recalculated' or 'reference';
                   flag is None, 'drift', 'failed' or 'skipped'.
    stopped      : True when the run was stopped before every gate was
                   applied (see run_gating's keep_partial).
    """
    boundaries: dict = field(default_factory=dict)
    masks: dict = field(default_factory=dict)
    parent_masks: dict = field(default_factory=dict)
    status: dict = field(default_factory=dict)
    stopped: bool = False

    def population_mask(self, gate_name: str, pop_name: str) -> np.ndarray | None:
        return (self.masks.get(gate_name) or {}).get(pop_name)


def run_gating(gate_defs: list[dict], data: np.ndarray, channel_index: dict,
               reference: dict | None = None, modes: dict | None = None,
               default_mode: str = MODE_FIXED,
               drift_limit: float = DEFAULT_DRIFT_LIMIT,
               drift_action: str = 'flag',
               axis_limits: dict | None = None,
               stop_check=None, keep_partial: bool = False,
               transforms: dict | None = None) -> GatingResult:
    """Run a gate hierarchy on one transformed event array.

    Without *reference* every gate is calculated from the data (training).
    With *reference*, each gate uses its mode from *modes* (default
    *default_mode*): ``fixed`` applies the reference boundary;
    ``recalculate`` computes a new one, falls back to the reference if that
    fails, and flags a shift larger than *drift_limit*. With
    ``drift_action='reference'`` a flagged gate uses the reference instead.

    Each gate is applied to its parent population only, in hierarchy order.
    *stop_check*, when given, is called before each gate; when it returns
    True the run stops by raising GatingStopped, or, with *keep_partial*,
    returns the gates applied so far with ``stopped`` set.
    *transforms* (see GateCalculator) are needed for threshold factors.
    """
    data = np.asarray(data)
    n = len(data)
    by_name = gates_by_name(gate_defs)
    calc = GateCalculator(axis_limits, transforms)
    modes = modes or {}
    result = GatingResult()
    root = np.ones(n, dtype=bool)

    for gate_def in ordered_gate_defs(gate_defs):
        if stop_check is not None and stop_check():
            if keep_partial:
                result.stopped = True
                return result
            raise GatingStopped()
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
            notes: list[str] = []
            try:
                resolved, reason = resolve_replicates(eff, data, channel_index, result, by_name,
                                                      transforms=calc.transforms, notes=notes)
                if resolved is None:
                    status['message'] = reason
                    boundary = {}
                else:
                    boundary = calc.calculate(parent_data, resolved, channel_index,
                                              trained=result.boundaries, by_name=by_name)
                    notes.extend(calc.notes)
                    if notes:
                        log.warning("gate %s: %s", name, '; '.join(notes))
                        status['message'] = '; '.join(notes)
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
        masks_parent = population_masks(eff, boundary, parent_data, channel_index,
                                        axis_limits=calc.axis_limits)
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
                     axis_limits: dict | None = None, done: dict | None = None,
                     stop_check=None, transforms: dict | None = None) -> GatingResult:
    """Calculate every gate from pooled training data (see run_gating).

    done       : {gate: boundary} already trained (a paused training run);
                 these are applied as they are and only the other gates are
                 calculated.
    stop_check : called before each gate; when it returns True training
                 stops and the gates trained so far are returned with
                 ``stopped`` set.
    """
    if not done:
        return run_gating(gate_defs, data, channel_index, reference=None,
                          axis_limits=axis_limits, stop_check=stop_check, keep_partial=True,
                          transforms=transforms)
    modes = {g['gate_name']: (MODE_FIXED if g['gate_name'] in done else MODE_RECALCULATE)
             for g in gate_defs}
    result = run_gating(gate_defs, data, channel_index, reference=dict(done), modes=modes,
                        axis_limits=axis_limits, stop_check=stop_check, keep_partial=True,
                        transforms=transforms)
    for status in result.status.values():
        if status.get('source') in ('recalculated', 'fixed'):
            status['source'] = 'trained'
    return result


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
        for entry in entries.values():
            shape = [c for c in (entry.get('shape_channels') or []) if c]
            convs = [converter(ch) for ch in (shape or chans)]
            if not any(convs):
                continue
            for key, k in (('threshold_x', 0), ('threshold_y', 1)):
                if entry.get(key) is not None and k < len(convs) and convs[k]:
                    entry[key] = float(_convert([entry[key]], *convs[k])[0])
            if entry.get('range') is not None and convs[0]:
                entry['range'] = [float(v) for v in _convert(entry['range'], *convs[0])]
            coords = np.asarray(entry.get('boundary') or [], dtype=float)
            if coords.ndim == 2 and coords.shape[1] == 2 and len(coords):
                if eff.get('gate_type') in POLYGON_TYPES or shape:
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


def gates_on_channel(gate_defs: list[dict], boundaries: dict, channel: str) -> set:
    """Names of the gates whose definition, imported shapes or stored
    boundary use *channel*."""
    by_name = gates_by_name(gate_defs)
    out = {g['gate_name'] for g in gate_defs
           if channel in referenced_channels(effective_gate_def(g, by_name))}
    for name, entries in (boundaries or {}).items():
        if any(channel in ((e or {}).get('shape_channels') or []) for e in (entries or {}).values()):
            out.add(name)
    return out


def remap_channel_transform(gate_defs: list[dict], trained_boundaries: dict, channel: str,
                            old_params: dict, new_params: dict,
                            transform_factory=make_transform,
                            points_per_edge: int = 1) -> bool:
    """Convert, in place, every boundary on *channel* from one transform to another.

    Used when the axis transform of a channel is changed while a model is
    open, so that gates keep their place in the data. Covers the trained
    boundaries and each gate's ``template`` and ``template_original``.
    The transform is changed in many small steps (an axis being dragged), so
    polygon edges are not subdivided by default: only the vertices move, and
    repeated conversions do not multiply them.
    Returns True when the two transforms differ and boundaries were converted.
    """
    if not old_params or not new_params or same_transform(old_params, new_params):
        return False
    src, dst = {channel: old_params}, {channel: new_params}
    names = gates_on_channel(gate_defs, trained_boundaries, channel)
    converted, _ = remap_boundaries(gate_defs,
                                    {n: trained_boundaries[n] for n in names if n in trained_boundaries},
                                    src, dst, transform_factory=transform_factory,
                                    points_per_edge=points_per_edge)
    trained_boundaries.update(converted)
    for key in ('template', 'template_original'):
        templates = {g['gate_name']: g[key] for g in gate_defs
                     if g.get(key) and g['gate_name'] in names}
        if not templates:
            continue
        converted, _ = remap_boundaries(gate_defs, templates, src, dst,
                                        transform_factory=transform_factory,
                                        points_per_edge=points_per_edge)
        for g in gate_defs:
            if g.get('gate_name') in converted:
                g[key] = converted[g['gate_name']]
    return True


def rename_channels(gate_defs: list[dict], alignment: dict) -> list[dict]:
    """Copy of *gate_defs* with axis channels, and the channels of imported
    shapes, renamed via {old: new}."""
    out = deepcopy(gate_defs)
    for g in out:
        for key in ('gate_marker_x', 'gate_marker_y'):
            if g.get(key) in alignment:
                g[key] = alignment[g[key]]
        for key in ('template', 'template_original'):
            if g.get(key):
                g[key] = rename_boundary_channels({'g': g[key]}, alignment)['g']
    return out


def rename_boundary_channels(boundaries: dict, alignment: dict) -> dict:
    """Copy of {gate: {pop: entry}} with each shape's channels renamed via
    {old: new}."""
    out = deepcopy(boundaries or {})
    for entries in out.values():
        for entry in (entries or {}).values():
            if isinstance(entry, dict) and entry.get('shape_channels'):
                entry['shape_channels'] = [alignment.get(c, c) for c in entry['shape_channels']]
    return out


def rekey(mapping: dict | None, alignment: dict) -> dict:
    """Copy of a per-channel dict with keys renamed via {old: new}.

    A renamed entry takes precedence over an unrenamed one that already has
    the new name."""
    mapping = mapping or {}
    out = {k: v for k, v in mapping.items() if k not in alignment}
    out.update({alignment[k]: v for k, v in mapping.items() if k in alignment})
    return out


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


def _digest_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    return str(obj)


def content_digest(obj) -> str:
    """SHA-256 of *obj* written as sorted JSON: a quick test of whether
    nested state has changed. Numpy values are converted; other objects
    count by their text."""
    try:
        text = json.dumps(obj, sort_keys=True, default=_digest_default)
    except (TypeError, ValueError):
        text = json.dumps(_jsonable(obj), sort_keys=True, default=_digest_default)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


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
    retired = migrate_retired_algorithms(gate_defs, m['trained_boundaries'])
    if retired:
        warnings.append(
            "The 'ellipse' algorithm is no longer offered; these gates now use 'cells' "
            "and their stored ellipses were converted to 10-vertex polygons (train again "
            "to calculate them with 'cells'): " + ", ".join(retired))
    untrained = [g['gate_name'] for g in gate_defs
                 if g['gate_name'] not in m['trained_boundaries']]
    if untrained:
        warnings.append("Untrained gate(s): " + ", ".join(untrained))
    return m, warnings


def migrate_retired_algorithms(gate_defs: list[dict], boundaries: dict) -> list[str]:
    """Replace the retired 'ellipse' algorithm with 'cells', in place, and
    turn the stored ellipses of those gates into 10-vertex polygons of the
    same area. Returns the names of the gates changed."""
    changed = []
    for g in gate_defs:
        hit = False
        if g.get('algorithm') == 'ellipse':
            g['algorithm'] = 'cells'
            hit = True
            for entry in (boundaries.get(g['gate_name']) or {}).values():
                coords = entry.get('boundary')
                if coords and len(coords) > ELLIPSE_VERTICES + 1:
                    entry['boundary'] = ellipse_as_polygon(coords)
        if g.get('proposed_algorithm') == 'ellipse':
            g['proposed_algorithm'] = 'cells'
            hit = True
        if hit:
            changed.append(g['gate_name'])
    return changed


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

    Threshold gates become one open-ended rectangle gate per population (a
    region of three quadrants, or of two opposite ones, a polygon gate on
    its outline); polygon gates become polygon gates. Each population is a gate named
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
                    sx, sy = region_sign(region, 'x'), region_sign(region, 'y')
                    n_quads = len(region_quadrants(region))
                    if not ((n_quads == 1) or (n_quads == 2 and (sx or sy))):
                        verts = [tuple(v) for v in (entry.get('boundary') or [])]
                        if len(verts) < 3:
                            continue
                        if verts[0] == verts[-1]:
                            verts = verts[:-1]
                        gs.add_gate(fk_gates.PolygonGate(gname, [dim(chans[0]), dim(chans[1])],
                                                         verts), gate_path=gate_path)
                        paths[key] = gate_path
                        continue
                    xr = {'+': (tx, None), '-': (None, tx)}.get(sx, (None, None))
                    yr = {'+': (ty, None), '-': (None, ty)}.get(sy, (None, None))
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
