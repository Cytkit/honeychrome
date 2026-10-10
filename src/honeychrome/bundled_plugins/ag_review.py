"""
ag_review.py — review of imported gates in the Gating Model Builder
====================================================================

Pure computation, no Qt: chooses which populations can be reviewed, groups
a population's daughter gates by the parameters they use (one biplot per
parameter set), compares Honeychrome's population fractions with the
counts recorded at import, and rewrites a gate's imported boundary when the
user moves it.

Imported gates carry a ``template`` (their boundary in display units),
``template_original`` (the boundary as imported) and a ``source`` record
(see ag_flowjo.build_gate_definitions). The file name does not end in
``_tab.py``, so the plugin loader does not treat it as a tab.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

import ag_core

__all__ = [
    'ROOT', 'ABS_TOLERANCE', 'REL_TOLERANCE',
    'reference_boundaries', 'population_choices', 'daughter_groups',
    'imported_fraction', 'fraction_discrepancy', 'matches_import_sample',
    'with_thresholds', 'with_polygon', 'with_range', 'reset_template',
    'polygon_vertices', 'convert_target', 'is_free_shape',
]

ROOT = ('', 'root')

# A fraction of parent differs from the import's when it moves by this many
# fractional units (0.02 = 2 percentage points) or by this relative amount.
ABS_TOLERANCE = 0.02
REL_TOLERANCE = 0.05
_MIN_REFERENCE = 0.0005


def reference_boundaries(gate_defs: list[dict], trained: dict | None) -> dict:
    """{gate: boundary entries} used to apply the hierarchy for review: the
    trained boundary where there is one, else the imported template."""
    out = {}
    for g in gate_defs:
        name = g['gate_name']
        boundary = (trained or {}).get(name) or g.get('template')
        if boundary:
            out[name] = boundary
    return out


def _channels(gate_def: dict, by_name: dict) -> tuple:
    return tuple(ag_core.gate_channels(ag_core.effective_gate_def(gate_def, by_name)))


def population_choices(gate_defs: list[dict]) -> list[dict]:
    """Populations that have daughter gates, root first, in hierarchy order.

    Each choice is ``{'gate', 'pop', 'label', 'n_gates'}``; the root has
    gate None and pop 'root'. Populations of imported gates are labelled
    with their own names (the FlowJo or hierarchy names); others as
    ``gate / population``.
    """
    ordered = ag_core.ordered_gate_defs(gate_defs)
    by_name = ag_core.gates_by_name(gate_defs)
    counts: dict = {}
    for g in ordered:
        key = (g.get('parent_gate') or None, g.get('parent_popul') or 'root')
        counts[key] = counts.get(key, 0) + 1

    choices = []
    if (None, 'root') in counts:
        choices.append({'gate': None, 'pop': 'root', 'label': 'All events (root)',
                        'n_gates': counts[(None, 'root')]})
    for g in ordered:
        gate_def = by_name[g['gate_name']]
        imported = bool(gate_def.get('source'))
        for pop in (gate_def.get('populations') or {}):
            n = counts.get((gate_def['gate_name'], pop))
            if not n:
                continue
            label = pop if imported else f"{gate_def['gate_name']} / {pop}"
            choices.append({'gate': gate_def['gate_name'], 'pop': pop,
                            'label': label, 'n_gates': n})
    return choices


def daughter_groups(gate_defs: list[dict], parent_gate, parent_pop) -> list[dict]:
    """Daughter gates of one population grouped by parameter set.

    Returns ``[{'channels': (x,) or (x, y), 'gates': [gate_def, ...]}]`` in
    hierarchy order; each group is drawn as one biplot (or histogram).
    """
    by_name = ag_core.gates_by_name(gate_defs)
    parent_gate = parent_gate or None
    parent_pop = parent_pop or 'root'
    groups: dict = {}
    for g in ag_core.ordered_gate_defs(gate_defs):
        if (g.get('parent_gate') or None) != parent_gate:
            continue
        if parent_gate is not None and (g.get('parent_popul') or 'root') != parent_pop:
            continue
        chans = _channels(g, by_name)
        if not chans:
            continue
        groups.setdefault(chans, []).append(by_name[g['gate_name']])
    return [{'channels': chans, 'gates': gates} for chans, gates in groups.items()]


def imported_fraction(gate_def: dict, pop: str):
    """Fraction of its parent that *pop* held in the imported sample, from the
    counts recorded at import, or None when they were not recorded."""
    src = gate_def.get('source') or {}
    count = (src.get('counts') or {}).get(pop)
    parent = src.get('parent_count')
    if not count or not parent:
        return None
    return float(count) / float(parent)


def fraction_discrepancy(current, imported, abs_tol: float = ABS_TOLERANCE,
                         rel_tol: float = REL_TOLERANCE) -> bool:
    """True when *current* differs from *imported* by at least *abs_tol*
    (fractional units) or *rel_tol* (relative to *imported*)."""
    if current is None or imported is None:
        return False
    diff = abs(float(current) - float(imported))
    if diff >= abs_tol:
        return True
    return imported >= _MIN_REFERENCE and diff / imported >= rel_tol


def matches_import_sample(gate_def: dict, sample_key: str) -> bool:
    """True when *sample_key* is the sample the counts were recorded on."""
    name = (gate_def.get('source') or {}).get('sample')
    if not name or not sample_key:
        return False
    key = str(sample_key).replace('\\', '/').rsplit('/', 1)[-1]
    return key.lower() == str(name).replace('\\', '/').rsplit('/', 1)[-1].lower()


def _labelled(gate_def: dict, entries: dict) -> dict:
    pops = gate_def.get('populations') or {}
    for name, entry in entries.items():
        entry['label'] = (pops.get(name) or {}).get('label', name)
    return entries


def with_thresholds(gate_def: dict, tx: float, ty: float | None,
                    axis_limits: dict | None = None) -> dict:
    """Template entries for a threshold gate moved to (tx[, ty])."""
    calc = ag_core.GateCalculator(axis_limits)
    return _labelled(gate_def, calc.threshold_boundaries(gate_def, float(tx),
                                                          None if ty is None else float(ty)))


def with_polygon(gate_def: dict, pop: str, polygon) -> dict:
    """Template entries with *pop*'s polygon replaced (closed)."""
    entries = deepcopy(gate_def.get('template') or {})
    pts = [[float(x), float(y)] for x, y in polygon]
    if pts and pts[0] != pts[-1]:
        pts.append(list(pts[0]))
    entry = entries.setdefault(pop, {'threshold_x': None, 'threshold_y': None,
                                     'region': ag_core.REGION_INSIDE})
    entry['boundary'] = pts
    entry.pop('range', None)
    return _labelled(gate_def, entries)


def with_range(gate_def: dict, pop: str, lo: float, hi: float) -> dict:
    """Template entries with a 1-D range gate's limits replaced."""
    entries = deepcopy(gate_def.get('template') or {})
    lo, hi = float(min(lo, hi)), float(max(lo, hi))
    entry = entries.setdefault(pop, {'threshold_x': None, 'threshold_y': None,
                                     'region': ag_core.REGION_INSIDE})
    entry['range'] = [lo, hi]
    entry['boundary'] = [[lo, 0.0], [hi, 0.0], [hi, 1.0], [lo, 1.0], [lo, 0.0]]
    return _labelled(gate_def, entries)


def reset_template(gate_def: dict) -> dict:
    """The boundary as imported, or {} when no original was kept."""
    return deepcopy(gate_def.get('template_original') or {})


def polygon_vertices(polygon, max_vertices: int = 32) -> list:
    """An open vertex list for editing: the closing point dropped and, for
    long polygons, evenly thinned to *max_vertices*."""
    pts = np.asarray(polygon, dtype=float)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) > max_vertices:
        pts = pts[np.linspace(0, len(pts), max_vertices, endpoint=False).astype(int)]
    return [[float(x), float(y)] for x, y in pts]


def is_free_shape(gate_def: dict) -> bool:
    """True for a gate whose imported boundary is a drawn shape that no
    learning algorithm reproduces (a free polygon or 1-D range)."""
    return gate_def.get('gate_type') == 'free'


def convert_target(gate_def: dict, recommendation: str | None = None):
    """The learning algorithm an imported gate converts to.

    Returns ``(algorithm, note)``; *algorithm* is None when the gate is not
    using its imported boundary. Threshold gates use the recommended
    algorithm when there is one, else 'tail'; singlet gates use 'singlets'.
    A free polygon can only become 'ellipse' (the 3 SD ellipse of its
    parent population), which does not reproduce the drawn shape, so the
    note says so.
    """
    if gate_def.get('algorithm') != ag_core.ALGORITHM_IMPORTED:
        return None, ''
    gate_type = gate_def.get('gate_type')
    allowed = [a for a in ag_core.ALGORITHMS_BY_GATE_TYPE.get(gate_type, [])
               if a != ag_core.ALGORITHM_IMPORTED]
    if gate_type in ag_core.THRESHOLD_TYPES:
        return (recommendation if recommendation in allowed else 'tail'), ''
    if gate_type == 'singlets':
        return 'singlets', ''
    if gate_type == 'free' and 'ellipse' in allowed:
        return 'ellipse', ("A drawn shape cannot be learned: 'ellipse' fits an ellipse to the "
                           "parent population instead of keeping the drawn polygon.")
    return (allowed[0] if allowed else None), ''
