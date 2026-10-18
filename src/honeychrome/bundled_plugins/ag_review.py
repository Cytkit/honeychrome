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

Also here: keeping gates that share a name (``name_group``, e.g. every
'CD80+') on one algorithm, the sources a replicated threshold can come
from, and deleting, copying and pasting gates and branches.
"""

from __future__ import annotations

import json
from copy import deepcopy

import numpy as np

import ag_core

__all__ = [
    'ROOT', 'ABS_TOLERANCE', 'REL_TOLERANCE',
    'reference_boundaries', 'population_choices', 'population_tree', 'daughter_groups',
    'imported_fraction', 'fraction_discrepancy', 'matches_import_sample',
    'with_thresholds', 'with_polygon', 'with_range', 'reset_template',
    'polygon_vertices', 'convert_target', 'is_free_shape', 'drawn_channels',
    'trainable_algorithms', 'trainable_definition', 'has_drawn_shape', 'split_2dsep',
    'histogram_proposal', 'name_group', 'sync_members', 'sync_fingerprint',
    'apply_name_sync', 'sync_conflicts', 'conflicting_gate_names', 'gates_below_population',
    'feed_forward', 'conversion_check', 'SYNC_USER', 'SYNC_AUTO', 'SYNC_DRAWN',
    'replicate_choices', 'descendants', 'delete_gates', 'delete_population',
    'copy_gates', 'paste_gates', 'rename_references', 'convert_threshold_regions',
    'orphaned_by_populations', 'rename_populations', 'population_ancestors',
    'overlapping_populations', 'overlap_flags', 'OVERLAP_LIMIT', 'OVERLAP_MIN_EVENTS',
    'RULE_HISTOGRAM', 'RULE_FRACTION', 'positive_fractions', 'proposed_axis_algorithms',
    'CENTRED_BELOW_ZERO', 'SPANS_ZERO_BELOW', 'REPLICATE_POSITIVE', 'SEPARATES_MIN',
    'FIT_ALGORITHMS', 'FIT_CANDIDATES', 'FIT_LABELS', 'axis_cuts', 'separating_axis_dropped',
    'without_axis',
    'default_reference', 'split_steps', 'conversion_evidence', 'evidence_axes',
    'reference_algorithm', 'proposed_definition', 'apply_conversion', 'default_source_axes',
]

ROOT = ('', 'root')

# proposal_rule of an imported threshold gate: its proposed algorithm follows
# the share of the parent on each axis's positive side (see histogram_proposal).
RULE_HISTOGRAM = 'histogram'    # histogram gates (from the import's counts)
RULE_FRACTION = 'fraction'      # other threshold gates (review samples only)

# A fraction of parent differs from the import's when it moves by this many
# fractional units (0.02 = 2 percentage points) or by this relative amount.
ABS_TOLERANCE = 0.02
REL_TOLERANCE = 0.05
_MIN_REFERENCE = 0.0005
# Two populations overlap when they share at least this share of the events
# in either (intersection over union) and neither lies below the other.
OVERLAP_LIMIT = 0.5
OVERLAP_MIN_EVENTS = 20
# A population sharing this share of its events with a population above it
# is treated as that population when looking for overlaps.
OVERLAP_SAME_AS_ANCESTOR = 0.9


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
    return drawn_channels(ag_core.effective_gate_def(gate_def, by_name))


def has_drawn_shape(gate_def: dict) -> bool:
    """True when the gate's template holds its shapes as drawn."""
    return ag_core.is_shape_boundary(gate_def.get('template'))


def drawn_channels(gate_def: dict) -> tuple:
    """Channels of the plot a gate is reviewed on: the plot its shapes were
    drawn on when it has them, else the plot its type is drawn on."""
    shape = ag_core.shape_channels(gate_def)
    if shape:
        return shape
    return tuple(c for c in ag_core.plot_channels(gate_def) if c)


def trainable_algorithms(gate_def: dict) -> list[str]:
    """Learning algorithms a gate's type offers (``imported`` excluded)."""
    return [a for a in ag_core.ALGORITHMS_BY_GATE_TYPE.get(gate_def.get('gate_type'), [])
            if a != ag_core.ALGORITHM_IMPORTED]


def trainable_definition(gate_def: dict, algorithm: str | None = None,
                         axis_algorithms: dict | None = None) -> dict:
    """Copy of *gate_def* using a learning algorithm: *algorithm*, else the
    gate's own when it is not ``imported``, else its conversion target.
    *axis_algorithms* ({axis: algorithm}) fills the per-axis algorithms of
    a 2dsep gate that has none set."""
    out = deepcopy(gate_def)
    if algorithm is None:
        algorithm = gate_def.get('algorithm')
        if algorithm == ag_core.ALGORITHM_IMPORTED:
            algorithm = convert_target(gate_def)[0]
    out['algorithm'] = algorithm
    if axis_algorithms and out.get('gate_type') == '2dsep':
        gp = out.setdefault('gate_param', {})
        for axis, algo in axis_algorithms.items():
            gp.setdefault(f'algorithm_{axis}', algo)
    return out


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


def population_tree(gate_defs: list[dict]) -> list[dict]:
    """Every population of the hierarchy as tree nodes, parents before children.

    Each node is ``{'gate', 'pop', 'label', 'parent', 'n_gates', 'view'}``.
    ``parent`` is the ``(gate, pop)`` key of the population the node's gate is
    drawn on, None for the root; the root has gate '' and pop 'root'.
    ``n_gates`` counts the gates drawn on the population. ``view`` is the
    ``(gate, pop)`` whose daughter gates are shown to review the node: the
    node itself when gates are drawn on it, else its parent (the plot it is
    gated on). Labels follow ``population_choices``; reference populations
    and the anchor population (ag_core.anchor_population) are marked.
    """
    ordered = ag_core.ordered_gate_defs(gate_defs)
    by_name = ag_core.gates_by_name(gate_defs)
    if not ordered:
        return []

    def key_of(gate, pop):
        return (gate or '', pop or 'root')

    anchor = ag_core.anchor_population(gate_defs)
    anchor_node = key_of(*ag_core.split_population_key(anchor)) if anchor else None

    counts: dict = {}
    for g in ordered:
        k = key_of(g.get('parent_gate'), g.get('parent_popul') if g.get('parent_gate') else 'root')
        counts[k] = counts.get(k, 0) + 1

    root = key_of(None, 'root')
    nodes = [{'gate': '', 'pop': 'root', 'label': 'All events (root)', 'parent': None,
              'n_gates': counts.get(root, 0), 'view': root}]
    known = {root}
    for g in ordered:
        gate_def = by_name[g['gate_name']]
        parent_gate = gate_def.get('parent_gate') or ''
        parent = key_of(parent_gate, gate_def.get('parent_popul') if parent_gate else 'root')
        if parent not in known:
            parent = root
        imported = bool(gate_def.get('source'))
        for pop in (gate_def.get('populations') or {}):
            key = key_of(gate_def['gate_name'], pop)
            if key in known:
                continue
            n = counts.get(key, 0)
            label = pop if imported else f"{gate_def['gate_name']} / {pop}"
            if ((gate_def.get('populations') or {}).get(pop) or {}).get('reference'):
                label += '  [reference]'
            if key_of(gate_def['gate_name'], pop) == anchor_node:
                label += '  [anchor]'
            nodes.append({'gate': gate_def['gate_name'], 'pop': pop, 'label': label,
                          'parent': parent, 'n_gates': n, 'view': key if n else parent})
            known.add(key)
    return nodes


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


def histogram_proposal(gate_def: dict, positive_fraction, axis: str = 'x') -> str | None:
    """The algorithm the import's rule proposes for one axis of an imported
    threshold gate whose proposal follows the rule (``proposal_rule``
    RULE_HISTOGRAM or RULE_FRACTION), given the fraction of its parent on
    the positive side: 'mixture' above 25%, else 'tail'. *positive_fraction*
    is a number (the x axis) or {axis: fraction} (see positive_fractions).
    None when the rule does not apply or the fraction is unknown."""
    if gate_def.get('proposal_rule') not in (RULE_HISTOGRAM, RULE_FRACTION):
        return None
    if isinstance(positive_fraction, dict):
        positive_fraction = positive_fraction.get(axis)
    if positive_fraction is None:
        return None
    import ag_flowjo
    return ag_flowjo.histogram_algorithm(positive_fraction)


def positive_fractions(gate_def: dict, fractions: dict) -> dict:
    """{axis: fraction of the parent on the positive side} of a threshold
    gate, from its populations' fractions of the parent (*fractions*, e.g.
    the shapes as drawn on the review samples): the populations wholly on
    the positive side, else one minus those wholly on the negative side.
    Axes with neither are left out."""
    if gate_def.get('gate_type') not in ag_core.THRESHOLD_TYPES:
        return {}
    regions = ag_core.population_regions(gate_def)
    out = {}
    for axis in ag_core.threshold_axes(gate_def):
        def side(region):
            if gate_def.get('gate_type') == '1dsep':
                return '+' if region == ag_core.REGION_ABOVE else '-'
            return ag_core.region_sign(region, axis)
        plus = [fractions.get(p) for p, r in regions.items() if side(r) == '+']
        minus = [fractions.get(p) for p, r in regions.items() if side(r) == '-']
        if plus and None not in plus:
            out[axis] = float(sum(plus))
        elif minus and None not in minus:
            out[axis] = 1.0 - float(sum(minus))
    return out


def proposed_axis_algorithms(gate_def: dict, recommendation: str | None = None,
                             positive_fraction=None, evidence: dict | None = None) -> dict:
    """{axis: algorithm} a threshold gate trains with: its own per-axis
    algorithms once converted; while imported, the conversion target on x
    (convert_target) and, on y of a 2dsep gate, the algorithm chosen for y,
    else the rule's for y (evidence_axes when *evidence* from the review
    events is given, else histogram_proposal), else x's. The axes are those
    of evidence['definition'] when given (a gate converting to 1dsep has x
    only). {} for other gates."""
    definition = (evidence or {}).get('definition') or gate_def
    if gate_def.get('algorithm') != ag_core.ALGORITHM_IMPORTED:
        if gate_def.get('gate_type') not in ag_core.THRESHOLD_TYPES:
            return {}
        return {a: ag_core.axis_algorithm(gate_def, a) for a in ag_core.threshold_axes(gate_def)}
    if definition.get('gate_type') not in ag_core.THRESHOLD_TYPES:
        return {}
    axes = ag_core.threshold_axes(definition)
    ruled = evidence_axes(evidence) if evidence and gate_def.get('proposal_rule') in (
        RULE_HISTOGRAM, RULE_FRACTION) else {}
    x = convert_target(gate_def, recommendation, positive_fraction, evidence)[0]
    out = {'x': x}
    if 'y' in axes:
        gp = gate_def.get('gate_param') or {}
        y = gp.get('algorithm_y') or (ruled['y'][0] if 'y' in ruled else
                                      histogram_proposal(gate_def, positive_fraction, 'y'))
        out['y'] = y if y in trainable_algorithms(definition) else x
    return out


def convert_target(gate_def: dict, recommendation: str | None = None,
                   positive_fraction=None, evidence: dict | None = None):
    """The learning algorithm an imported gate converts to.

    Returns ``(algorithm, note)``; *algorithm* is None when the gate is not
    using its imported boundary. A gate still on the import's rule (see
    histogram_proposal) uses the rules on the review events (evidence_axes)
    when *evidence* is given, else *positive_fraction* (a number or {axis:
    fraction}) when one is given;
    otherwise the algorithm proposed at import comes first, then for
    threshold gates the recommended algorithm when there is one, else
    'tail'; singlet gates use 'singlets' and scatter shapes 'cells'.
    """
    if gate_def.get('algorithm') != ag_core.ALGORITHM_IMPORTED:
        return None, ''
    gate_type = gate_def.get('gate_type')
    allowed = trainable_algorithms(gate_def)
    if evidence and gate_def.get('proposal_rule') in (RULE_HISTOGRAM, RULE_FRACTION):
        ruled = evidence_axes(evidence).get('x')
        if ruled and ruled[0] in allowed:
            return ruled[0], ruled[1]
    by_rule = histogram_proposal(gate_def, positive_fraction)
    if by_rule in allowed:
        return by_rule, ''
    proposed = gate_def.get('proposed_algorithm')
    if proposed in allowed:
        return proposed, ''
    if gate_type in ag_core.THRESHOLD_TYPES:
        return (recommendation if recommendation in allowed else 'tail'), ''
    if gate_type == 'singlets':
        return 'singlets', ''
    if gate_type == 'free' and 'cells' in allowed:
        return 'cells', ''
    return (allowed[0] if allowed else None), ''


def convert_threshold_regions(populations: dict, old_type: str, old_x, old_y,
                              new_type: str, new_x, new_y) -> tuple[dict | None, str]:
    """Populations of a threshold gate changed between 1dsep and 2dsep,
    keeping their names (so the gates below them stay attached) and their
    sides of the shared separation parameter.

    2dsep → 1dsep separating on one of the two parameters: each population
    takes its side of that parameter; populations that span both sides, or
    share a side, cannot be kept (Split into 1D gates instead).
    1dsep → 2dsep with the separation parameter on one axis: each side
    becomes that half of the plot (two quadrants).

    Returns (populations, '') or (None, reason). Other type changes return
    (populations unchanged, '').
    """
    out = deepcopy(populations)
    if (old_type, new_type) == ('2dsep', '1dsep'):
        axis = 'x' if new_x == old_x else ('y' if new_x == old_y else None)
        if axis is None:
            return None, "The separation parameter must be one of the 2dsep gate's two parameters."
        sides: dict = {}
        for pop, meta in out.items():
            sign = ag_core.region_sign(meta.get('region'), axis)
            if sign is None:
                return None, (f"'{pop}' lies on both sides of {new_x}; use Split into 1D gates "
                              "instead.")
            if sign in sides:
                return None, (f"'{sides[sign]}' and '{pop}' are on the same side of {new_x}; "
                              "use Split into 1D gates instead.")
            sides[sign] = pop
            meta['region'] = ag_core.REGION_ABOVE if sign == '+' else ag_core.REGION_BELOW
            meta['label_pos'] = 2 if sign == '+' else 1
        return out, ''
    if (old_type, new_type) == ('1dsep', '2dsep'):
        axis = 'x' if old_x == new_x else ('y' if old_x == new_y else None)
        if axis is None:
            return None, "One of the 2dsep gate's parameters must be the 1dsep separation parameter."
        for pop, meta in out.items():
            sign = '+' if meta.get('region') == ag_core.REGION_ABOVE else '-'
            k = 1 if axis == 'x' else 3
            meta['region'] = ag_core.join_quadrants(q for q in ag_core.QUADRANT_REGIONS
                                                    if q[k] == sign)
        return out, ''
    return out, ''


# ---------------------------------------------------------------------------
# Conversion proposals from the review events
# ---------------------------------------------------------------------------

# 1dsep gates on their parent's separation channel, from the review events:
CENTRED_BELOW_ZERO = 0.40     # this much of the parent below raw zero: tail
SPANS_ZERO_BELOW = 0.05       # at least this much below zero: best fit of FIT_ALGORITHMS
REPLICATE_POSITIVE = 0.40     # less below zero and more drawn positive: replicate
# A single drawn shape's axis whose edges leave out at most this share of
# the parent's events within the shape's band on the other axis separates
# nothing: the gate converts to 1dsep on the other axis.
SEPARATES_MIN = 0.01
FIT_ALGORITHMS = ('tail', 'mixture')
# Candidates the shapes as drawn are fitted with: label → (algorithm, parameters).
FIT_CANDIDATES = {
    'tail': ('tail', {}),
    'mixture': ('mixture', {'mixture_components': 2, 'mixture_cut': 'outer'}),
    'mixture3_lower': ('mixture', {'mixture_components': 3, 'mixture_cut': 'lower'}),
    'mixture3_upper': ('mixture', {'mixture_components': 3, 'mixture_cut': 'upper'}),
}
FIT_LABELS = {'tail': 'tail', 'mixture': 'mixture',
              'mixture3_lower': 'mixture (3 components, lower cut)',
              'mixture3_upper': 'mixture (3 components, upper cut)'}
FIT_MAX_EVENTS = 10_000
EVIDENCE_MIN_EVENTS = 50


def _agreement(drawn: dict, calculated: dict) -> float | None:
    """Mean over the drawn populations of the events in both / events in
    the larger of drawn and calculated."""
    scores = []
    for pop, d in drawn.items():
        c = calculated.get(pop)
        if c is None:
            scores.append(0.0)
            continue
        nd, nc = int(d.sum()), int(c.sum())
        scores.append(float((c & d).sum()) / max(nd, nc) if max(nd, nc) else 1.0)
    return float(np.mean(scores)) if scores else None


def _shape_extent(gate_def: dict, pop: str, channel: str):
    """(lo, hi) of population *pop*'s drawn shape on *channel*, or None."""
    entry = (gate_def.get('template') or {}).get(pop) or {}
    chans = list(entry.get('shape_channels') or [])
    if channel not in chans:
        return None
    k = chans.index(channel)
    if entry.get('range') is not None and k == 0:
        return float(entry['range'][0]), float(entry['range'][1])
    coords = np.asarray(entry.get('boundary') or [], dtype=float)
    if coords.ndim != 2 or coords.shape[1] <= k or not len(coords):
        return None
    return float(coords[:, k].min()), float(coords[:, k].max())


def axis_cuts(gate_def: dict, parent: np.ndarray, index: dict) -> dict:
    """{axis: share of the parent's events within the drawn shape's band on
    the other axis that the shape's edges on this axis leave out}, for a
    2dsep gate drawn as a single shape; {} otherwise."""
    if gate_def.get('gate_type') != '2dsep' or len(gate_def.get('populations') or {}) != 1 \
            or not has_drawn_shape(gate_def):
        return {}
    pop = next(iter(gate_def['populations']))
    chans = {a: ag_core.axis_channel(gate_def, a) for a in ('x', 'y')}
    ext = {a: _shape_extent(gate_def, pop, ch) for a, ch in chans.items()}
    if any(e is None for e in ext.values()) or any(ch not in index for ch in chans.values()):
        return {}
    inside = {a: (parent[:, index[chans[a]]] >= ext[a][0]) & (parent[:, index[chans[a]]] <= ext[a][1])
              for a in ('x', 'y')}
    out = {}
    for a, other in (('x', 'y'), ('y', 'x')):
        band = inside[other]
        if int(band.sum()) >= 20:
            out[a] = 1.0 - float(inside[a][band].mean())
    return out


def separating_axis_dropped(gate_def: dict, parent: np.ndarray, index: dict) -> str | None:
    """The axis of a single-shape 2dsep gate that separates nothing (see
    axis_cuts and SEPARATES_MIN), or None. When both qualify, the one
    leaving out fewer events."""
    cuts = axis_cuts(gate_def, parent, index)
    if len(cuts) != 2:
        return None
    axis = min(cuts, key=cuts.get)
    return axis if cuts[axis] <= SEPARATES_MIN else None


def without_axis(gate_def: dict, axis: str) -> dict:
    """Copy of a 2dsep gate as a 1dsep gate separating on its other axis,
    drawn against the dropped one on the same plot axes. Populations keep
    their names and sides; per-axis parameters of the kept axis become the
    1dsep's."""
    keep = 'y' if axis == 'x' else 'x'
    sep, view = ag_core.axis_channel(gate_def, keep), ag_core.axis_channel(gate_def, axis)
    pops, reason = convert_threshold_regions(gate_def.get('populations') or {}, '2dsep',
                                             gate_def.get('gate_marker_x'),
                                             gate_def.get('gate_marker_y'), '1dsep', sep, view)
    if pops is None:
        raise ValueError(reason)
    out = deepcopy(gate_def)
    out.update(gate_type='1dsep', gate_marker_x=sep, gate_marker_y=view, sep_axis=keep,
               populations=pops)
    gp = out.get('gate_param') or {}
    new_gp = {k: v for k, v in gp.items()
              if not (k.endswith('_x') or k.endswith('_y')) and k != 'algorithm'}
    for key in ('tail_fraction', 'mixture_components', 'axis_lo', 'axis_hi'):
        if f'{key}_{keep}' in gp:
            new_gp[key] = gp[f'{key}_{keep}']
    if isinstance(gp.get(f'replicate_{keep}'), dict):
        new_gp['replicate_x'] = deepcopy(gp[f'replicate_{keep}'])
    out['gate_param'] = new_gp
    if gp.get(f'algorithm_{keep}') and out.get('algorithm') != ag_core.ALGORITHM_IMPORTED:
        out['algorithm'] = gp[f'algorithm_{keep}']
    return out


_UNSET = object()


def default_reference(gate_defs: list[dict], gate_def: dict, anchor=_UNSET) -> str | None:
    """The anchor population (ag_core.anchor_population, or *anchor* when
    the caller has it) when it lies above the population *gate_def* is
    drawn on; None otherwise (a gate drawn on the anchor, above it or
    beside it has no default reference)."""
    if anchor is _UNSET:
        anchor = ag_core.anchor_population(gate_defs)
    if anchor is None:
        return None
    by_name = ag_core.gates_by_name(gate_defs)
    g = by_name.get(gate_def.get('parent_gate'))
    seen = set()
    while g is not None and g['gate_name'] not in seen and g.get('parent_gate'):
        seen.add(g['gate_name'])
        if ag_core.population_key(g['parent_gate'], g.get('parent_popul') or 'root') == anchor:
            return anchor
        g = by_name.get(g['parent_gate'])
    return None


def _drawn_threshold(gate_def: dict) -> float | None:
    """A 1dsep gate's drawn threshold on its separation channel: the lower
    edge of the shapes on the positive side, else the upper edge of those
    on the negative side."""
    ch = gate_def.get('gate_marker_x')
    regions = ag_core.population_regions(gate_def)
    above = [_shape_extent(gate_def, p, ch) for p, r in regions.items() if r == ag_core.REGION_ABOVE]
    below = [_shape_extent(gate_def, p, ch) for p, r in regions.items() if r == ag_core.REGION_BELOW]
    if above and None not in above:
        return min(e[0] for e in above)
    if below and None not in below:
        return max(e[1] for e in below)
    return None


def split_steps(gate_def: dict, names: dict | None = None) -> list[dict] | None:
    """Steps for split_2dsep when a 2dsep gate has one population on a whole
    side of one channel and one population in each quadrant on the other
    side: that channel first, then the other on that side. Each step's
    'algorithm' is 'tail' (the gates are proposed afresh after the split).
    None for other layouts."""
    if gate_def.get('gate_type') != '2dsep':
        return None
    regions = ag_core.population_regions(gate_def)
    if len(regions) != 3:
        return None
    chans = {'x': gate_def.get('gate_marker_x'), 'y': gate_def.get('gate_marker_y')}
    for a, b in (('x', 'y'), ('y', 'x')):
        whole = [p for p, r in regions.items()
                 if ag_core.region_sign(r, a) is not None and ag_core.region_sign(r, b) is None]
        if len(whole) != 1:
            continue
        side = ag_core.region_sign(regions[whole[0]], a)
        rest = [r for p, r in regions.items() if p != whole[0]]
        if all(len(ag_core.region_quadrants(r)) == 1 and ag_core.region_sign(r, a) not in (None, side)
               for r in rest) and len({ag_core.region_sign(r, b) for r in rest}) == 2:
            return [{'channel': chans[a], 'algorithm': 'tail', 'view': chans[b], 'sep_axis': a},
                    {'channel': chans[b], 'algorithm': 'tail', 'view': chans[a], 'sep_axis': b}]
    return None


def _apply_candidate(definition: dict, axis: str, label: str) -> None:
    """Give one axis of a threshold gate the FIT_CANDIDATES entry *label*,
    in place."""
    algo, params = FIT_CANDIDATES[label]
    gp = definition.setdefault('gate_param', {})
    suffix = f'_{axis}' if definition.get('gate_type') == '2dsep' else ''
    if suffix:
        gp[f'algorithm_{axis}'] = algo
    else:
        definition['algorithm'] = algo
    for key in ('mixture_components', 'mixture_cut'):
        gp.pop(f'{key}{suffix}', None)
    for key, value in params.items():
        gp[f'{key}{suffix}'] = value


def _fit_scores(gate_def: dict, parent: np.ndarray, index: dict, drawn: dict,
                calc, candidates: dict) -> dict:
    """{tuple of per-axis candidate labels: agreement with the drawn masks}
    for every combination of *candidates* ({axis: [FIT_CANDIDATES labels]},
    [None] for an axis kept as it is). Each candidate's thresholds are
    calculated once; the combinations reuse them."""
    import itertools
    axes = list(candidates)
    labels = sorted({lab for a in axes for lab in candidates[a] if lab})
    thresholds = {}
    for label in labels or [None]:
        d = deepcopy(gate_def)
        if d.get('gate_type') == '2dsep':
            d['algorithm'] = 'tail'
        for a in axes:
            if label is not None and label in candidates[a]:
                _apply_candidate(d, a, label)
        try:
            boundary = calc.calculate(parent, d, index)
        except Exception:
            boundary = {}
        if boundary:
            first = next(iter(boundary.values()))
            thresholds[label] = {a: first.get(f'threshold_{a}') for a in axes}
    if not thresholds:
        return {}
    any_t = next(iter(thresholds.values()))
    out = {}
    for combo in itertools.product(*(candidates[a] for a in axes)):
        if any(lab is not None and lab not in thresholds for lab in combo):
            continue
        d = deepcopy(gate_def)
        d['algorithm'] = 'tail'
        gp = d.setdefault('gate_param', {})
        for a, lab in zip(axes, combo):
            t = (thresholds[lab] if lab is not None else any_t).get(a)
            if t is None:
                break
            gp[f'fixed_threshold_{a}'] = float(t)
        else:
            try:
                boundary = calc.calculate(parent, d, index)
                masks = ag_core.population_masks(d, boundary, parent, index) if boundary else {}
            except Exception:
                masks = {}
            score = _agreement(drawn, masks) if masks else None
            if score is not None:
                out[combo] = score
    return out


def conversion_evidence(gate_def: dict, gate_defs: list[dict], data: np.ndarray, index: dict,
                        result, axis_limits: dict | None, zeros: dict, anchor=_UNSET) -> dict:
    """What the review events say about converting an imported gate.

    *result* gates *data* with the hierarchy as reviewed (parent masks and
    population masks); *zeros* is {channel: raw zero in display units}.
    Returns {} when the gate has no drawn shape or too few parent events,
    else a dict with:

    definition        the gate the proposal applies to: the gate itself,
                      or as 1dsep when an axis of a single drawn shape
                      separates nothing (drop_axis);
    drop_axis         that axis, or None;
    positive          {axis: fraction of the parent on the positive side}
                      of the shapes as drawn (positive_fractions);
    below_zero        {axis: fraction of the parent below raw zero};
    fit               {per-axis algorithm tuple: agreement with the shapes
                      as drawn} over FIT_ALGORITHMS (2dsep gates, and 1dsep
                      gates spanning zero but not centred on it);
    reference         default reference population (default_reference)
                      of a 1dsep gate, with reference_positive (its share
                      on the positive side of the drawn threshold);
    split             split_steps of a 2dsep gate, or None;
    cell_side         for a scatter shape: CELL_SIDE_DEBRIS when most drawn
                      events lie below the debris cut, else CELL_SIDE_CELLS.
    """
    name = gate_def['gate_name']
    parent_mask = result.parent_masks.get(name) if result is not None else None
    template = gate_def.get('template') or {}
    if parent_mask is None or not ag_core.is_shape_boundary(template) \
            or int(parent_mask.sum()) < EVIDENCE_MIN_EVENTS:
        return {}
    parent = data[parent_mask]
    if len(parent) > FIT_MAX_EVENTS:
        rng = np.random.default_rng(42)
        parent = parent[np.sort(rng.choice(len(parent), FIT_MAX_EVENTS, replace=False))]
    if any(ch not in index for ch in ag_core.shape_channels(gate_def)):
        return {}
    drawn = ag_core.population_masks(gate_def, template, parent, index, axis_limits=axis_limits)
    evidence = {'definition': gate_def, 'drop_axis': None, 'n': int(parent_mask.sum())}
    gate_type = gate_def.get('gate_type')
    if gate_type == 'free':
        gp = gate_def.get('gate_param') or {}
        d = 1 if gp.get('debris_axis') == 'y' else 0
        chans = ag_core.shape_channels(gate_def)
        if len(chans) == 2 and drawn:
            col = parent[:, index[chans[d]]]
            cut = ag_core.debris_cut(ag_core.trim_quantiles(col, float(gp.get('quantile_trim', 0.001))),
                                     float(gp.get('debris_density', 0.05)))
            inside = np.any(np.stack(list(drawn.values())), axis=0)
            if inside.any():
                below = float((col[inside] < cut).mean())
                evidence['cell_side'] = (ag_core.CELL_SIDE_DEBRIS if below > 0.5
                                         else ag_core.CELL_SIDE_CELLS)
        return evidence
    if gate_type not in ag_core.THRESHOLD_TYPES:
        return evidence
    definition = gate_def
    drop = separating_axis_dropped(gate_def, parent, index)
    if drop is not None:
        try:
            definition = without_axis(gate_def, drop)
            evidence['drop_axis'] = drop
        except ValueError:
            definition = gate_def
    evidence['definition'] = definition
    fractions = {p: float(m.mean()) for p, m in drawn.items()}
    evidence['positive'] = positive_fractions(definition, fractions)
    below = {}
    for axis in ag_core.threshold_axes(definition):
        ch = ag_core.axis_channel(definition, axis)
        if ch in index and ch in zeros:
            below[axis] = float((parent[:, index[ch]] < float(zeros[ch])).mean())
    evidence['below_zero'] = below
    evidence['split'] = split_steps(definition)
    calc = ag_core.GateCalculator(axis_limits)
    gp = definition.get('gate_param') or {}
    if definition.get('gate_type') == '2dsep':
        candidates = {a: ([None] if gp.get(f'algorithm_{a}') else list(FIT_CANDIDATES))
                      for a in ('x', 'y')}
        if ag_core.ALGORITHM_REPLICATE not in (gp.get('algorithm_x'), gp.get('algorithm_y')):
            evidence['fit'] = _fit_scores(definition, parent, index, drawn, calc, candidates)
    else:
        b0 = below.get('x')
        if b0 is not None and SPANS_ZERO_BELOW <= b0 < CENTRED_BELOW_ZERO:
            evidence['fit'] = _fit_scores(definition, parent, index, drawn, calc,
                                          {'x': list(FIT_CANDIDATES)})
        ref = default_reference(gate_defs, definition, anchor)
        threshold = _drawn_threshold(definition)
        ch = definition.get('gate_marker_x')
        if ref is not None and threshold is not None and ch in index:
            mask = result.population_mask(*ag_core.split_population_key(ref))
            if mask is not None and int(mask.sum()) >= EVIDENCE_MIN_EVENTS:
                evidence['reference'] = ref
                evidence['reference_positive'] = float((data[mask, index[ch]] >= threshold).mean())
    return evidence


def evidence_axes(evidence: dict) -> dict:
    """{axis: (algorithm, reason, parameters)} the rules propose for the
    gate in evidence['definition'] (see conversion_evidence); {} when the
    evidence does not cover it. *parameters* are gate_param entries for the
    axis without suffix (e.g. mixture_components and mixture_cut of a
    3-component mixture).

    1dsep: tail when at least CENTRED_BELOW_ZERO of the parent lies below
    zero; the better fit of FIT_ALGORITHMS to the shapes as drawn when at
    least SPANS_ZERO_BELOW does; replicate (from the default reference
    population) when less does and more than REPLICATE_POSITIVE of the
    parent is drawn positive; otherwise the 25% rule (histogram_proposal).
    2dsep: the best-fitting combination of FIT_ALGORITHMS, else the 25% rule
    per axis.
    """
    d = evidence.get('definition')
    if not d or d.get('gate_type') not in ag_core.THRESHOLD_TYPES:
        return {}
    pos = evidence.get('positive') or {}
    fit = evidence.get('fit') or {}

    def by_rule(axis):
        f = pos.get(axis)
        if f is None:
            return None
        import ag_flowjo
        algo = ag_flowjo.histogram_algorithm(f)
        return algo, f"{100 * f:.1f}% of the parent drawn positive", {}

    def fitted(label, why):
        algo, params = FIT_CANDIDATES[label]
        return algo, f"{why}best fit to the shapes as drawn: {FIT_LABELS[label]}", dict(params)

    if d.get('gate_type') == '2dsep':
        if fit:
            combo = max(fit, key=fit.get)
            why = f"agreement {fit[combo]:.2f}; "
            return {a: fitted(lab, why) for a, lab in zip(('x', 'y'), combo) if lab is not None}
        return {a: r for a in ('x', 'y') if (r := by_rule(a)) is not None}
    b0 = (evidence.get('below_zero') or {}).get('x')
    if b0 is None:
        r = by_rule('x')
        return {'x': r} if r else {}
    if b0 >= CENTRED_BELOW_ZERO:
        return {'x': ('tail', f"{100 * b0:.0f}% of the parent below zero (centred on zero)", {})}
    if b0 >= SPANS_ZERO_BELOW:
        if fit:
            combo = max(fit, key=fit.get)
            return {'x': fitted(combo[0], f"{100 * b0:.0f}% of the parent below zero; agreement "
                                          f"{fit[combo]:.2f}; ")}
        r = by_rule('x')
        return {'x': r} if r else {}
    f = pos.get('x')
    if f is not None and f > REPLICATE_POSITIVE and evidence.get('reference'):
        return {'x': (ag_core.ALGORITHM_REPLICATE,
                      f"{100 * f:.0f}% of the parent drawn positive and {100 * b0:.1f}% below "
                      "zero: its own events cannot place the threshold", {})}
    r = by_rule('x')
    return {'x': r} if r else {}


def reference_algorithm(evidence: dict) -> str:
    """Algorithm the anchor population calculates a replicated threshold
    with: the 25% rule on its share above the drawn threshold."""
    import ag_flowjo
    return ag_flowjo.histogram_algorithm(evidence.get('reference_positive'))


def proposed_definition(gate_def: dict, recommendation: str | None = None,
                        positive_fraction=None, evidence: dict | None = None) -> dict:
    """The trainable gate an imported gate converts to: as 1dsep when the
    evidence drops an axis, with the proposed algorithms (an axis chosen by
    hand keeps its choice), a default reference population for a proposed
    replicate axis without a source (marked 'default', to be confirmed),
    and the debris side for a scatter shape drawn on the debris. A gate no
    longer imported is returned as it is (a copy)."""
    if gate_def.get('algorithm') != ag_core.ALGORITHM_IMPORTED:
        return deepcopy(gate_def)
    evidence = evidence or {}
    base = gate_def
    if evidence.get('drop_axis') and gate_def.get('gate_type') == '2dsep':
        try:
            base = without_axis(gate_def, evidence['drop_axis'])
        except ValueError:
            base = gate_def
    axes = proposed_axis_algorithms(gate_def, recommendation, positive_fraction, evidence)
    algorithm = axes.get('x') or convert_target(gate_def, recommendation, positive_fraction,
                                                evidence)[0]
    out = trainable_definition(dict(base, algorithm=ag_core.ALGORITHM_IMPORTED), algorithm, axes)
    gp = out.setdefault('gate_param', {})
    ruled = evidence_axes(evidence) if evidence and gate_def.get('proposal_rule') in (
        RULE_HISTOGRAM, RULE_FRACTION) else {}
    suffix_of = (lambda a: f'_{a}') if out.get('gate_type') == '2dsep' else (lambda a: '')
    for axis, algo in axes.items():
        rule = ruled.get(axis)
        if rule and rule[0] == algo:
            for k, v in rule[2].items():
                gp.setdefault(f'{k}{suffix_of(axis)}', v)
        key = f'replicate_{axis}'
        if algo == ag_core.ALGORITHM_REPLICATE and not isinstance(gp.get(key), dict) \
                and evidence.get('reference'):
            gp[key] = {'population': evidence['reference'],
                       'algorithm': reference_algorithm(evidence), 'default': True}
    if evidence.get('cell_side') and out.get('gate_type') == 'free' and 'cell_side' not in gp:
        gp['cell_side'] = evidence['cell_side']
    return out


def apply_conversion(gate_def: dict, proposed: dict) -> None:
    """Make an imported gate the trainable gate *proposed*
    (proposed_definition), in place: type, channels, separation axis,
    populations, parameters and algorithm. The shapes as drawn stay."""
    for key in ('gate_type', 'gate_marker_x', 'gate_marker_y', 'sep_axis', 'populations',
                'gate_param', 'algorithm'):
        if key in proposed:
            gate_def[key] = deepcopy(proposed[key])


def default_source_axes(gate_def: dict) -> list[str]:
    """Axes whose replicate source is a default reference population not
    yet confirmed by the user."""
    gp = gate_def.get('gate_param') or {}
    return [a for a in ('x', 'y') if isinstance(gp.get(f'replicate_{a}'), dict)
            and gp[f'replicate_{a}'].get('default')]


def orphaned_by_populations(gate_defs: list[dict], gate_name: str, removed) -> list[str]:
    """Gates drawn on one of the *removed* populations of *gate_name*."""
    removed = set(removed)
    return [g['gate_name'] for g in gate_defs
            if g.get('parent_gate') == gate_name and g.get('parent_popul') in removed]


def rename_populations(gate_defs: list[dict], gate_name: str, renames: dict,
                       trained: dict | None = None) -> None:
    """Rename populations of *gate_name* ({old: new}) everywhere, in place:
    the gate's shapes as drawn, import record and trained boundaries, and
    the parents, replicate sources and statistics parents of other gates."""
    renames = {o: n for o, n in renames.items() if o != n}
    if not renames:
        return

    def key(value):
        gate, pop = ag_core.split_population_key(str(value))
        if gate == gate_name and pop in renames:
            return ag_core.population_key(gate_name, renames[pop])
        return value

    for g in gate_defs:
        if g.get('gate_name') == gate_name:
            for field in ('template', 'template_original'):
                entries = g.get(field)
                if isinstance(entries, dict):
                    g[field] = {renames.get(p, p): e for p, e in entries.items()}
            src = g.get('source') or {}
            for field in ('populations', 'counts'):
                if isinstance(src.get(field), dict):
                    src[field] = {renames.get(p, p): v for p, v in src[field].items()}
        if g.get('parent_gate') == gate_name and g.get('parent_popul') in renames:
            g['parent_popul'] = renames[g['parent_popul']]
        gp = g.get('gate_param') or {}
        for axis in ('x', 'y'):
            src = gp.get(f'replicate_{axis}')
            if isinstance(src, dict) and src.get('population'):
                src['population'] = key(src['population'])
        sp = g.get('stats_parent')
        if isinstance(sp, dict):
            for k, v in list(sp.items()):
                sp[k] = key(v)
    if trained and isinstance(trained.get(gate_name), dict):
        trained[gate_name] = {renames.get(p, p): e for p, e in trained[gate_name].items()}


def _split_name(text: str) -> str:
    return str(text).replace('/', '∕').strip()


def split_2dsep(gate_defs: list[dict], gate_name: str, steps: list[dict],
                names: dict | None = None, keep_drawn: bool = False
                ) -> tuple[list[dict], list[str]]:
    """Replace a 2dsep gate with sequential 1dsep gates.

    *steps* gives the order: two dicts with 'channel' (one of the gate's
    two channels), 'algorithm', 'view' (view channel or None for a
    histogram) and 'sep_axis'. The first step's gate separates the parent
    population on its channel into '<channel>-' and '<channel>+' (the
    channel as named in *names*, e.g. its antigen, when given). On each
    side holding any of the 2dsep gate's populations, a second gate
    separates on the other channel; it carries those populations under
    their own names, so gates drawn on them keep their parent population
    and move to it. Per-axis parameters of the 2dsep gate
    (``algorithm_x``, ``tail_fraction_y``, …) go to the matching gate.

    *keep_drawn* (for a gate with shapes as drawn): the new gates stay
    imported, each population keeping its drawn shape (a side population
    of the first gate is drawn as the convex hull of the shapes on that
    side), with each step's algorithm as the proposal and the gate's
    proposal rule.

    Returns (new gate definition list, names of the new gates); raises
    ValueError when a new name is taken or the steps do not match the gate.
    """
    by_name = ag_core.gates_by_name(gate_defs)
    g = by_name.get(gate_name)
    if g is None or g.get('gate_type') != '2dsep':
        raise ValueError(f"'{gate_name}' is not a 2dsep gate.")
    axes = {g.get('gate_marker_x'): 'x', g.get('gate_marker_y'): 'y'}
    if len(steps) != 2 or {st.get('channel') for st in steps} != set(axes):
        raise ValueError("The steps must separate on the gate's two channels.")
    first, second = steps
    a_axis, b_axis = axes[first['channel']], axes[second['channel']]
    params = deepcopy(g.get('gate_param') or {})

    def param_for(axis: str) -> dict:
        p = {'quantile_trim': params.get('quantile_trim', 0.001),
             'tail_fraction': float(params.get(f'tail_fraction_{axis}',
                                               params.get('tail_fraction',
                                                          ag_core.DEFAULT_TAIL_FRACTION))),
             'mixture_components': int(params.get(f'mixture_components_{axis}',
                                                  params.get('mixture_components', 2)))}
        for key in ('axis_lo', 'axis_hi'):
            value = params.get(f'{key}_{axis}', 0.0)
            if value:
                p[key] = float(value)
        return p

    sign = ag_core.region_sign

    regions = ag_core.population_regions(g)
    spanning = [p for p, r in regions.items() if sign(r, a_axis) is None]
    if spanning:
        raise ValueError(
            f"{', '.join(repr(p) for p in spanning)} lies on both sides of "
            f"{first['channel']}; separate on {second['channel']} first.")
    src = g.get('source') or {}
    names = names or {}
    a = _split_name(names.get(first['channel']) or first['channel'])
    b = _split_name(names.get(second['channel']) or second['channel'])
    drawn = keep_drawn and has_drawn_shape(g)
    template = g.get('template') or {}
    side_members: dict[str, list] = {}
    name_a = f"{gate_name} · {a}"
    side_pop = {'-': f"{a}-", '+': f"{a}+"}
    gate_a = {
        'gate_name': name_a, 'gate_type': '1dsep',
        'gate_marker_x': first['channel'], 'gate_marker_y': first.get('view'),
        'sep_axis': first.get('sep_axis') or 'x',
        'parent_gate': g.get('parent_gate'), 'parent_popul': g.get('parent_popul', 'root'),
        'populations': {
            side_pop['-']: {'label': side_pop['-'], 'label_pos': 1, 'region': ag_core.REGION_BELOW},
            side_pop['+']: {'label': side_pop['+'], 'label_pos': 2, 'region': ag_core.REGION_ABOVE},
        },
        'algorithm': first['algorithm'], 'gate_param': param_for(a_axis),
        'stats_parent': {}, 'origin_gate': None,
    }
    new = [gate_a]
    moved: dict[str, str] = {}
    for side in ('-', '+'):
        pops = {p: r for p, r in regions.items() if sign(r, a_axis) == side}
        if not pops:
            continue
        side_members[side] = list(pops)
        whole = [p for p, r in pops.items() if sign(r, b_axis) is None]
        if whole:
            if len(pops) > 1:
                raise ValueError(f"'{whole[0]}' covers the whole {a}{side} side, which other "
                                 "populations of the gate share.")
            p = whole[0]
            meta = dict((g.get('populations') or {}).get(p) or {})
            side_meta = gate_a['populations'].pop(side_pop[side])
            gate_a['populations'][p] = dict(side_meta, label=meta.get('label', p))
            moved[p] = name_a
            continue
        name_b = f"{gate_name} · {b} ({a}{side})"
        populations = {}
        for p, r in pops.items():
            region = ag_core.REGION_ABOVE if sign(r, b_axis) == '+' else ag_core.REGION_BELOW
            meta = dict((g.get('populations') or {}).get(p) or {})
            populations[p] = {'label': meta.get('label', p),
                              'label_pos': 2 if region == ag_core.REGION_ABOVE else 1,
                              'region': region}
            moved[p] = name_b
        gate_b = {
            'gate_name': name_b, 'gate_type': '1dsep',
            'gate_marker_x': second['channel'], 'gate_marker_y': second.get('view'),
            'sep_axis': second.get('sep_axis') or 'x',
            'parent_gate': name_a, 'parent_popul': side_pop[side],
            'populations': populations,
            'algorithm': second['algorithm'], 'gate_param': param_for(b_axis),
            'stats_parent': {}, 'origin_gate': None,
        }
        if drawn:
            gate_b['template'] = {p: dict(deepcopy(template[p]), region=meta['region'])
                                  for p, meta in populations.items() if p in template}
        if src.get('populations'):
            gate_b['source'] = {
                'origin': src.get('origin'), 'workspace': src.get('workspace'),
                'gating_source': src.get('gating_source'), 'sample': src.get('sample'),
                'parent_count': None,
                'populations': {p: v for p, v in src['populations'].items() if p in populations},
                'counts': {p: v for p, v in (src.get('counts') or {}).items() if p in populations},
            }
        new.append(gate_b)

    if drawn:
        entries = {}
        for side, members in side_members.items():
            pop = next((p for p in gate_a['populations'] if p in members), None) \
                or side_pop[side]
            region = gate_a['populations'][pop]['region']
            if pop in template:
                entries[pop] = dict(deepcopy(template[pop]), region=region)
                continue
            shapes = [np.asarray(template[p].get('boundary') or [], dtype=float)
                      for p in members if p in template]
            shapes = [c for c in shapes if c.ndim == 2 and len(c) >= 3]
            if not shapes:
                continue
            first_entry = template[members[0]]
            hull = ag_core._convex_hull_polygon(np.vstack(shapes))
            if hull is None:
                continue
            entries[pop] = {'shape_channels': list(first_entry.get('shape_channels') or []),
                            'boundary': hull, 'region': region, 'label': pop}
        gate_a['template'] = entries
        for d in new:
            step = first if d is gate_a else second
            d['proposed_algorithm'] = step['algorithm']
            d['algorithm'] = ag_core.ALGORITHM_IMPORTED
            d['proposal_rule'] = g.get('proposal_rule') or RULE_FRACTION
            d['template_original'] = deepcopy(d.get('template') or {})
            d['name_group'] = d['gate_name']

    taken = {d['gate_name'] for d in gate_defs if d['gate_name'] != gate_name}
    clash = [d['gate_name'] for d in new if d['gate_name'] in taken]
    if clash:
        raise ValueError("Gate name(s) already exist: " + ', '.join(f"'{c}'" for c in clash))

    out = []
    for d in gate_defs:
        if d['gate_name'] == gate_name:
            out.extend(new)
            continue
        d = deepcopy(d)
        if d.get('parent_gate') == gate_name:
            target = moved.get(d.get('parent_popul'))
            if target is None:
                raise ValueError(f"'{d['gate_name']}' is drawn on '{d.get('parent_popul')}', "
                                 f"which is not a population of '{gate_name}'.")
            d['parent_gate'] = target
        out.append(d)
    for k, d in enumerate(ag_core.ordered_gate_defs(out), start=1):
        d['gate_number'] = k
    out.sort(key=lambda d: d['gate_number'])
    return out, [d['gate_name'] for d in new]


# ---------------------------------------------------------------------------
# Gates that share a name
# ---------------------------------------------------------------------------

SYNC_USER = 'user'      # the algorithm was chosen for this gate by the user
SYNC_AUTO = 'auto'      # the algorithm was copied from same-named gates
SYNC_DRAWN = 'drawn'    # the user returned this gate to its shapes as drawn

# gate_param keys that are part of a gate's trainable configuration.
SYNC_PARAM_KEYS = (
    'algorithm_x', 'algorithm_y', 'replicate_x', 'replicate_y',
    'tail_fraction', 'tail_fraction_x', 'tail_fraction_y',
    'mixture_components', 'mixture_components_x', 'mixture_components_y',
    'axis_lo', 'axis_hi', 'axis_lo_x', 'axis_hi_x', 'axis_lo_y', 'axis_hi_y',
    'quantile_trim', 'width_sd', 'debris_axis', 'debris_density', 'edge_density',
    'edge_density_x', 'edge_density_y', 'cell_side', 'saturation_fraction',
    'mixture_cut', 'mixture_cut_x', 'mixture_cut_y',
)


def name_group(gate_def: dict) -> str:
    """The name a gate shares with its replicates: the name it was imported
    with (``name_group``), else its own name."""
    return gate_def.get('name_group') or gate_def['gate_name']


def _sync_key(gate_def: dict) -> tuple:
    return (name_group(gate_def), gate_def.get('gate_type'),
            gate_def.get('gate_marker_x'), gate_def.get('gate_marker_y'))


def sync_members(gate_defs: list[dict], gate_name: str) -> list[dict]:
    """The gates treated as replicates of *gate_name*: the same imported
    name, type and channels. Empty when the gate has no such partners."""
    by_name = ag_core.gates_by_name(gate_defs)
    g = by_name.get(gate_name)
    if g is None:
        return []
    key = _sync_key(g)
    members = [o for o in gate_defs if _sync_key(o) == key]
    return members if len(members) > 1 else []


def sync_fingerprint(gate_def: dict) -> str:
    """The gate's trainable configuration as a comparable string: its
    algorithm and the gate_param keys in SYNC_PARAM_KEYS (a default
    reference population counts as 'default', whichever it is)."""
    gp = gate_def.get('gate_param') or {}
    params = {}
    for k in SYNC_PARAM_KEYS:
        if k not in gp:
            continue
        v = gp[k]
        if isinstance(v, dict) and v.get('default'):
            v = {'default': True, 'algorithm': v.get('algorithm')}
        params[k] = v
    return json.dumps({'algorithm': gate_def.get('algorithm'), 'params': params},
                      sort_keys=True, default=str)


def _synced_params(gate_defs: list[dict], target: dict, tgp: dict, anchor=_UNSET) -> dict:
    """*target*'s gate_param with the SYNC_PARAM_KEYS of *tgp*; a default
    reference population becomes the target's own (default_reference)."""
    gp = {k: v for k, v in (target.get('gate_param') or {}).items() if k not in SYNC_PARAM_KEYS}
    for k in SYNC_PARAM_KEYS:
        if k not in tgp:
            continue
        v = deepcopy(tgp[k])
        if isinstance(v, dict) and v.get('default'):
            ref = default_reference(gate_defs, target, anchor)
            v = dict(v, population=ref) if ref else None
        gp[k] = v
    return gp


def _source_gates(tgp: dict) -> set:
    """Gates a configuration's replicate sources come from, leaving out
    default reference populations (each gate has its own)."""
    out = set()
    for src in (tgp.get('replicate_x'), tgp.get('replicate_y')):
        if not isinstance(src, dict) or src.get('default'):
            continue
        if src.get('gate'):
            out.add(src['gate'])
        elif src.get('population'):
            out.add(ag_core.split_population_key(src['population'])[0])
    return out


def _user_set(gate_def: dict) -> bool:
    """Converted to trainable by the user (gates converted before syncing
    existed count as the user's choice)."""
    return (gate_def.get('algorithm') != ag_core.ALGORITHM_IMPORTED
            and gate_def.get('sync_origin') not in (SYNC_AUTO, SYNC_DRAWN))


def _follows_sync(gate_def: dict) -> bool:
    """Updated to match its replicates: still as drawn, or set automatically."""
    if gate_def.get('sync_origin') == SYNC_DRAWN:
        return False
    return (gate_def.get('algorithm') == ag_core.ALGORITHM_IMPORTED
            or gate_def.get('sync_origin') == SYNC_AUTO)


def _group_label(members: list[dict]) -> str:
    return name_group(members[0])


def apply_name_sync(gate_defs: list[dict], gate_name: str) -> tuple[list[str], str | None]:
    """Bring the replicates of *gate_name* in line with the configuration
    the user chose for them.

    The user-set members (see _user_set) decide. When they all agree, every
    member that is still as drawn, or was set automatically, takes their
    algorithm and parameters and is marked SYNC_AUTO; members the user set,
    or returned to their shapes as drawn, are left alone. A member is also
    left alone when the configuration would make it replicate a threshold
    from itself or from a gate below it. When the user-set members disagree
    nothing changes and the group's name is returned as a conflict.

    Returns (names of the gates changed, conflicting group name or None).
    """
    members = sync_members(gate_defs, gate_name)
    if not members:
        return [], None
    chosen = [m for m in members if _user_set(m)]
    if not chosen:
        return [], None
    prints = {sync_fingerprint(m) for m in chosen}
    if len(prints) > 1:
        return [], _group_label(members)
    template = chosen[0]
    target = sync_fingerprint(template)
    tgp = template.get('gate_param') or {}
    sources = _source_gates(tgp)
    anchor = _UNSET
    changed = []
    for m in members:
        if not _follows_sync(m):
            continue
        if m.get('algorithm') != ag_core.ALGORITHM_IMPORTED and sync_fingerprint(m) == target:
            continue
        below = {m['gate_name']} | descendants(gate_defs, m['gate_name'])
        if sources & below:
            continue
        if anchor is _UNSET:
            anchor = ag_core.anchor_population(gate_defs)
        m['gate_param'] = _synced_params(gate_defs, m, tgp, anchor)
        m['algorithm'] = template.get('algorithm')
        m['sync_origin'] = SYNC_AUTO
        changed.append(m['gate_name'])
    return changed, None


def sync_conflicts(gate_defs: list[dict]) -> list[str]:
    """Names of the groups of same-named gates whose user-set members use
    different configurations, so the others are not updated."""
    seen, out = set(), []
    for g in gate_defs:
        key = _sync_key(g)
        if key in seen:
            continue
        seen.add(key)
        members = sync_members(gate_defs, g['gate_name'])
        prints = {sync_fingerprint(m) for m in members if _user_set(m)}
        if len(prints) > 1:
            out.append(_group_label(members))
    return out


def conflicting_gate_names(gate_defs: list[dict]) -> set:
    """Names of every gate in a same-named group whose user-set members use
    different configurations (see sync_conflicts)."""
    seen, out = set(), set()
    for g in gate_defs:
        key = _sync_key(g)
        if key in seen:
            continue
        seen.add(key)
        members = sync_members(gate_defs, g['gate_name'])
        if len({sync_fingerprint(m) for m in members if _user_set(m)}) > 1:
            out |= {m['gate_name'] for m in members}
    return out


def gates_below_population(gate_defs: list[dict], parent_gate, parent_pop: str) -> set:
    """Names of every gate drawn on population (parent_gate, parent_pop) or
    on any population below it; parent_gate None is the root."""
    parent_gate = parent_gate or None
    on_it = {g['gate_name'] for g in gate_defs
             if (g.get('parent_gate') or None) == parent_gate
             and (parent_gate is None or (g.get('parent_popul') or 'root') == parent_pop)}
    out = set(on_it)
    for name in on_it:
        out |= descendants(gate_defs, name)
    return out


def feed_forward(gate_defs: list[dict], gate_name: str,
                 everywhere: bool = False) -> tuple[list[str], list[str]]:
    """Give the gates that share *gate_name*'s name, type and channels and
    lie below the population it is drawn on (anywhere in the model with
    *everywhere*) the same trainable configuration (algorithm and the
    SYNC_PARAM_KEYS parameters), in place.

    The copies count as the user's choice (SYNC_USER). A gate is left
    alone when the configuration would make it replicate a threshold from
    itself or from a gate below it.

    Returns (names changed, names skipped).
    """
    by_name = ag_core.gates_by_name(gate_defs)
    source = by_name.get(gate_name)
    if source is None or source.get('algorithm') == ag_core.ALGORITHM_IMPORTED:
        return [], []
    key = _sync_key(source)
    if everywhere:
        scope = {g['gate_name'] for g in gate_defs} - {gate_name}
    else:
        scope = gates_below_population(gate_defs, source.get('parent_gate'),
                                       source.get('parent_popul') or 'root') - {gate_name}
    tgp = source.get('gate_param') or {}
    sources = _source_gates(tgp)
    anchor = _UNSET
    changed, skipped = [], []
    for g in ag_core.ordered_gate_defs(gate_defs):
        name = g['gate_name']
        if name not in scope or _sync_key(g) != key:
            continue
        if sources & ({name} | descendants(gate_defs, name)):
            skipped.append(name)
            continue
        target = by_name[name]
        if anchor is _UNSET:
            anchor = ag_core.anchor_population(gate_defs)
        gp = _synced_params(gate_defs, target, tgp, anchor)
        if (target.get('algorithm') == source.get('algorithm')
                and sync_fingerprint(dict(target, gate_param=gp)) == sync_fingerprint(target)):
            continue
        target['gate_param'] = gp
        target['algorithm'] = source.get('algorithm')
        target['sync_origin'] = SYNC_USER
        changed.append(name)
    return changed, skipped


def conversion_check(gate_def: dict, drawn: dict, proposed: dict, drawn_fractions: dict,
                     proposed_fractions: dict, shift_limit: float = 0.1,
                     fraction_limit: float = 0.1) -> dict | None:
    """Compare a gate's imported boundary (*drawn*) with its trainable form
    (*proposed*) on the same events.

    shift    : the largest move of a threshold, in transformed axis units
               (thresholds of both; None when either is a shape).
    fraction : the largest change of a population's fraction of the parent.

    Returns {'shift', 'fraction', 'message'} when either exceeds its limit,
    else None.
    """
    shift = None
    first_d = next(iter(drawn.values()), {}) if drawn else {}
    first_p = next(iter(proposed.values()), {}) if proposed else {}
    if not ag_core.is_shape_boundary(drawn) and first_d and first_p:
        moves = [abs(float(first_p[k]) - float(first_d[k])) for k in ('threshold_x', 'threshold_y')
                 if first_d.get(k) is not None and first_p.get(k) is not None]
        shift = max(moves) if moves else None
    pops = set(drawn_fractions) & set(proposed_fractions)
    fraction = max((abs(float(proposed_fractions[p]) - float(drawn_fractions[p])) for p in pops),
                   default=None)
    over_shift = shift is not None and shift > shift_limit
    over_fraction = fraction is not None and fraction > fraction_limit
    if not (over_shift or over_fraction):
        return None
    parts = []
    if shift is not None:
        parts.append(f"threshold moves {shift:.3f} axis units")
    if fraction is not None:
        parts.append(f"a population changes by {100 * fraction:.1f} percentage points")
    return {'shift': shift, 'fraction': fraction,
            'message': "Trainable form far from the imported boundary: " + "; ".join(parts) + "."}


# ---------------------------------------------------------------------------
# Replicated thresholds
# ---------------------------------------------------------------------------

def descendants(gate_defs: list[dict], gate_name: str) -> set:
    """Names of every gate below *gate_name*."""
    out, frontier = set(), {gate_name}
    while frontier:
        frontier = {g['gate_name'] for g in gate_defs if g.get('parent_gate') in frontier} - out
        out |= frontier
    return out


def replicate_choices(gate_defs: list[dict], gate_name: str, axis: str,
                      anchor=_UNSET) -> list[tuple]:
    """Sources a replicated threshold of *gate_name* on *axis* can come from,
    as (label, source) pairs: other threshold gates separating on the same
    channel (their threshold is copied), then the reference populations
    (the threshold is calculated on their events), the anchor population
    when it lies above the gate (default_reference) and the population the
    axis replicates from now. The gate itself and the gates below it are
    left out, since they are calculated after it."""
    by_name = ag_core.gates_by_name(gate_defs)
    g = by_name.get(gate_name)
    if g is None:
        return []
    channel = ag_core.axis_channel(g, axis)
    excluded = {gate_name} | descendants(gate_defs, gate_name)
    out = []
    for o in ag_core.ordered_gate_defs(gate_defs):
        if o['gate_name'] in excluded:
            continue
        eff = ag_core.effective_gate_def(o, by_name)
        if eff.get('gate_type') not in ag_core.THRESHOLD_TYPES:
            continue
        chans = [eff.get('gate_marker_x')]
        if eff.get('gate_type') == '2dsep':
            chans.append(eff.get('gate_marker_y'))
        if channel in chans:
            out.append((f"Copy from gate: {o['gate_name']}", {'gate': o['gate_name']}))
    for key in ag_core.reference_populations(gate_defs):
        gate, pop = ag_core.split_population_key(key)
        if gate in excluded:
            continue
        out.append((f"Reference population: {pop} ({gate})", {'population': key, 'algorithm': 'tail'}))
    listed = {src.get('population') for _label, src in out}
    anchor = default_reference(gate_defs, g, anchor)
    current = (g.get('gate_param') or {}).get(f'replicate_{axis}')
    extra = [anchor] + ([current.get('population')] if isinstance(current, dict) else [])
    for key in extra:
        if not key or key in listed:
            continue
        gate, pop = ag_core.split_population_key(key)
        if gate in excluded:
            continue
        listed.add(key)
        kind = "Anchor population" if key == anchor else "Population"
        out.append((f"{kind}: {pop} ({gate})", {'population': key, 'algorithm': 'tail'}))
    return out


# ---------------------------------------------------------------------------
# Deleting, copying and pasting
# ---------------------------------------------------------------------------

def _clear_sources(gate_defs: list[dict], removed: set) -> list[str]:
    """Clear replicate sources that point at removed gates or their
    populations. Returns the names of the gates affected."""
    hit = []
    for g in gate_defs:
        gp = g.get('gate_param') or {}
        for axis in ('x', 'y'):
            src = gp.get(f'replicate_{axis}')
            if not isinstance(src, dict):
                continue
            gate = src.get('gate') or ag_core.split_population_key(src.get('population') or '')[0]
            if gate in removed:
                gp[f'replicate_{axis}'] = None
                if g['gate_name'] not in hit:
                    hit.append(g['gate_name'])
    return hit


def delete_gates(gate_defs: list[dict], names) -> tuple[list[dict], list[str], list[str]]:
    """Remove gates with everything below them, and old-style replicate
    gates copying them.

    Returns (remaining definitions, removed names, names of remaining gates
    whose replicate source was removed and must be chosen again).
    """
    removed = set()
    frontier = set(names)
    while frontier:
        removed |= frontier
        nxt = set()
        for g in gate_defs:
            if g['gate_name'] in removed:
                continue
            if g.get('parent_gate') in removed or (
                    g.get('gate_type') == 'replicate' and g.get('origin_gate') in removed):
                nxt.add(g['gate_name'])
        frontier = nxt
    out = [deepcopy(g) for g in gate_defs if g['gate_name'] not in removed]
    orphaned = _clear_sources(out, removed)
    return out, [g['gate_name'] for g in gate_defs if g['gate_name'] in removed], orphaned


def delete_population(gate_defs: list[dict], gate_name: str, pop: str
                      ) -> tuple[list[dict], list[str], list[str]]:
    """Remove one population of a gate with the gates below it; a gate left
    with no population is removed. Returns as delete_gates."""
    by_name = ag_core.gates_by_name(gate_defs)
    g = by_name.get(gate_name)
    if g is None or pop not in (g.get('populations') or {}):
        return [deepcopy(x) for x in gate_defs], [], []
    if len(g['populations']) == 1:
        return delete_gates(gate_defs, [gate_name])
    below = [o['gate_name'] for o in gate_defs
             if o.get('parent_gate') == gate_name and o.get('parent_popul') == pop]
    out, removed, orphaned = delete_gates(gate_defs, below) if below else (
        [deepcopy(x) for x in gate_defs], [], [])
    target = ag_core.gates_by_name(out)[gate_name]
    target['populations'].pop(pop, None)
    for key in ('template', 'template_original'):
        if target.get(key):
            target[key].pop(pop, None)
    src = target.get('source') or {}
    for key in ('populations', 'counts'):
        if isinstance(src.get(key), dict):
            src[key].pop(pop, None)
    for o in out:
        sp = o.get('stats_parent')
        if isinstance(sp, dict):
            for k, v in list(sp.items()):
                if v == ag_core.population_key(gate_name, pop):
                    sp.pop(k)
    gone_key = ag_core.population_key(gate_name, pop)
    for o in out:
        gp = o.get('gate_param') or {}
        for axis in ('x', 'y'):
            src_ = gp.get(f'replicate_{axis}')
            if isinstance(src_, dict) and src_.get('population') == gone_key:
                gp[f'replicate_{axis}'] = None
                if o['gate_name'] not in orphaned:
                    orphaned.append(o['gate_name'])
    return out, removed, orphaned


def copy_gates(gate_defs: list[dict], names, branch: bool = False) -> list[dict]:
    """Copies of the named gates (and, with *branch*, every gate below
    them), parents before children, for pasting elsewhere."""
    wanted = set(names)
    if branch:
        for n in list(wanted):
            wanted |= descendants(gate_defs, n)
    return [deepcopy(g) for g in ag_core.ordered_gate_defs(gate_defs) if g['gate_name'] in wanted]


def _paste_name(base: str, place: str, taken: set) -> str:
    cand = f"{base} ({place})"
    stem, k = cand, 2
    while cand in taken:
        cand = f"{stem} {k}"
        k += 1
    return cand


def paste_gates(gate_defs: list[dict], clipboard: list[dict], target_gate, target_pop: str
                ) -> tuple[list[dict], list[str]]:
    """Add copies of *clipboard* (see copy_gates) below the population
    (target_gate, target_pop); target_gate None or '' is the root.

    Copied gates whose parent was not copied are drawn on the target;
    the others keep their place in the copied branch. Each copy is named
    after its name group and the population it now sits in, keeps the name
    group (so it follows its same-named gates), and is untrained.
    Replicate sources pointing inside the copied branch follow the copies.

    Returns (new definition list, names of the new gates).
    """
    target_gate = target_gate or None
    taken = {g['gate_name'] for g in gate_defs}
    copied = {g['gate_name'] for g in clipboard}
    renamed: dict[str, str] = {}
    new = []
    inner = set()
    number = max((int(g.get('gate_number', 0) or 0) for g in gate_defs), default=0)
    for g in clipboard:
        c = deepcopy(g)
        if c.get('parent_gate') in copied:
            place = c.get('parent_popul') or 'root'
            inner.add(id(c))
        else:
            place = target_pop if target_gate else 'All events'
            c['parent_gate'], c['parent_popul'] = target_gate, (target_pop if target_gate else 'root')
        base = name_group(g)
        name = _paste_name(base, place, taken)
        taken.add(name)
        renamed[g['gate_name']] = name
        c['name_group'] = base
        c['gate_name'] = name
        c.pop('sync_origin', None)
        for meta in (c.get('populations') or {}).values():
            if isinstance(meta, dict):
                meta.pop(ag_core.ANCHOR_FLAG, None)
        number += 1
        c['gate_number'] = number
        new.append(c)
    for c in new:
        if id(c) in inner:
            c['parent_gate'] = renamed[c['parent_gate']]
        if c.get('origin_gate') in renamed:
            c['origin_gate'] = renamed[c['origin_gate']]
        gp = c.get('gate_param') or {}
        for axis in ('x', 'y'):
            src = gp.get(f'replicate_{axis}')
            if not isinstance(src, dict):
                continue
            if src.get('gate') in renamed:
                src['gate'] = renamed[src['gate']]
            elif src.get('population'):
                gate, pop = ag_core.split_population_key(src['population'])
                if gate in renamed:
                    src['population'] = ag_core.population_key(renamed[gate], pop)
        sp = c.get('stats_parent')
        if isinstance(sp, dict):
            for k, v in list(sp.items()):
                gate, pop = ag_core.split_population_key(str(v))
                if gate in renamed:
                    sp[k] = ag_core.population_key(renamed[gate], pop)
    out = [deepcopy(g) for g in gate_defs] + new
    ag_core.ordered_gate_defs(out)
    return out, [c['gate_name'] for c in new]


def rename_references(gate_defs: list[dict], old: str, new: str) -> None:
    """Point every reference to gate *old* at *new*, in place: parents,
    replicate origins and sources, and statistics parents."""
    for g in gate_defs:
        if g.get('parent_gate') == old:
            g['parent_gate'] = new
        if g.get('origin_gate') == old:
            g['origin_gate'] = new
        gp = g.get('gate_param') or {}
        for axis in ('x', 'y'):
            src = gp.get(f'replicate_{axis}')
            if not isinstance(src, dict):
                continue
            if src.get('gate') == old:
                src['gate'] = new
            elif src.get('population'):
                gate, pop = ag_core.split_population_key(src['population'])
                if gate == old:
                    src['population'] = ag_core.population_key(new, pop)
        sp = g.get('stats_parent')
        if isinstance(sp, dict):
            for k, v in list(sp.items()):
                gate, pop = ag_core.split_population_key(str(v))
                if pop and gate == old:
                    sp[k] = ag_core.population_key(new, pop)
                elif v == old:
                    sp[k] = new


# ---------------------------------------------------------------------------
# Overlapping populations
# ---------------------------------------------------------------------------

def population_ancestors(gate_defs: list[dict]) -> dict:
    """{(gate, pop): set of (gate, pop) populations above it}."""
    by_name = ag_core.gates_by_name(gate_defs)
    out: dict = {}

    def above(gate_name: str, seen: set) -> set:
        g = by_name.get(gate_name)
        parent = g.get('parent_gate') if g else None
        if not parent or parent in seen or parent not in by_name:
            return set()
        key = (parent, g.get('parent_popul'))
        return {key} | above(parent, seen | {gate_name})

    for g in gate_defs:
        chain = above(g['gate_name'], set())
        for pop in (g.get('populations') or {}):
            out[(g['gate_name'], pop)] = chain
    return out


def overlapping_populations(gate_defs: list[dict], masks: dict, limit: float = OVERLAP_LIMIT,
                            min_events: int = OVERLAP_MIN_EVENTS) -> list[tuple]:
    """Populations defined twice in the hierarchy: ``[(a, b, share)]`` with
    *a* and *b* ``(gate, pop)`` keys holding largely the same events
    (*share*, the intersection over the union, at least *limit*) although
    neither lies below the other.

    Only populations that gates are drawn on (branches of the hierarchy,
    not end-point marker gates) and that are distinct from every population
    above them are compared: a population sharing at least
    OVERLAP_SAME_AS_ANCESTOR of its events with one of its ancestors is
    represented by that ancestor, so each population defined twice is
    reported once rather than for every gate below it. Populations of one
    gate are not compared, nor populations with fewer than *min_events*
    events.

    masks : {(gate, pop): boolean event mask}, all over the same events.
    """
    ancestors = population_ancestors(gate_defs)
    parents = {(g.get('parent_gate'), g.get('parent_popul')) for g in gate_defs
               if g.get('parent_gate')}
    keys = [k for k, m in masks.items() if m is not None and int(np.count_nonzero(m)) >= min_events]
    if len(keys) < 2:
        return []
    index = {k: i for i, k in enumerate(keys)}
    mat = np.vstack([np.asarray(masks[k], dtype=np.float32) for k in keys])
    sizes = mat.sum(axis=1)
    inter = mat @ mat.T
    union = sizes[:, None] + sizes[None, :] - inter
    share = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    distinct = np.array([k in parents and not any(
        index.get(a) is not None and share[index[k], index[a]] >= OVERLAP_SAME_AS_ANCESTOR
        for a in ancestors.get(k, ())) for k in keys])
    out = []
    rows, cols = np.nonzero(np.triu((share >= limit) & distinct[:, None] & distinct[None, :], k=1))
    for i, j in zip(rows.tolist(), cols.tolist()):
        a, b = keys[i], keys[j]
        if a[0] == b[0] or a in ancestors.get(b, ()) or b in ancestors.get(a, ()):
            continue
        out.append((a, b, float(share[i, j])))
    out.sort(key=lambda t: -t[2])
    return out


def overlap_flags(pairs: list[tuple]) -> dict:
    """{gate: {'pairs': [[pop, other gate, other pop, share]], 'message'}}
    from overlapping_populations, each pair listed under both gates."""
    flags: dict = {}
    for (ga, pa), (gb, pb), share in pairs:
        for (g, p), (og, op) in (((ga, pa), (gb, pb)), ((gb, pb), (ga, pa))):
            flags.setdefault(g, {'pairs': []})['pairs'].append([p, og, op, round(share, 4)])
    for g, flag in flags.items():
        parts = [f"'{p}' shares {100 * sh:.0f}% of its events with '{op}' ({og})"
                 for p, og, op, sh in flag['pairs'][:5]]
        more = len(flag['pairs']) - 5
        flag['message'] = ("Overlap: " + "; ".join(parts)
                           + (f"; and {more} more" if more > 0 else '') + ".")
    return flags
