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
    'reference_boundaries', 'population_choices', 'population_tree', 'daughter_groups',
    'imported_fraction', 'fraction_discrepancy', 'matches_import_sample',
    'with_thresholds', 'with_polygon', 'with_range', 'reset_template',
    'polygon_vertices', 'convert_target', 'is_free_shape', 'drawn_channels',
    'trainable_algorithms', 'trainable_definition', 'has_drawn_shape', 'split_2dsep',
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


def trainable_definition(gate_def: dict, algorithm: str | None = None) -> dict:
    """Copy of *gate_def* using a learning algorithm: *algorithm*, else the
    gate's own when it is not ``imported``, else its conversion target."""
    out = deepcopy(gate_def)
    if algorithm is None:
        algorithm = gate_def.get('algorithm')
        if algorithm == ag_core.ALGORITHM_IMPORTED:
            algorithm = convert_target(gate_def)[0]
    out['algorithm'] = algorithm
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
    gated on). Labels follow ``population_choices``.
    """
    ordered = ag_core.ordered_gate_defs(gate_defs)
    by_name = ag_core.gates_by_name(gate_defs)
    if not ordered:
        return []

    def key_of(gate, pop):
        return (gate or '', pop or 'root')

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


def convert_target(gate_def: dict, recommendation: str | None = None):
    """The learning algorithm an imported gate converts to.

    Returns ``(algorithm, note)``; *algorithm* is None when the gate is not
    using its imported boundary. The algorithm proposed at import comes
    first; otherwise threshold gates use the recommended algorithm when
    there is one, else 'tail', and singlet gates use 'singlets'.
    A free polygon can only become 'ellipse' (the 3 SD ellipse of its
    parent population), which does not reproduce the drawn shape, so the
    note says so.
    """
    if gate_def.get('algorithm') != ag_core.ALGORITHM_IMPORTED:
        return None, ''
    gate_type = gate_def.get('gate_type')
    allowed = trainable_algorithms(gate_def)
    proposed = gate_def.get('proposed_algorithm')
    if proposed in allowed and gate_type != 'free':
        return proposed, ''
    if gate_type in ag_core.THRESHOLD_TYPES:
        return (recommendation if recommendation in allowed else 'tail'), ''
    if gate_type == 'singlets':
        return 'singlets', ''
    if gate_type == 'free' and 'ellipse' in allowed:
        return 'ellipse', ("A drawn shape cannot be learned: 'ellipse' fits an ellipse to the "
                           "parent population instead of keeping the drawn polygon.")
    return (allowed[0] if allowed else None), ''


def _split_name(text: str) -> str:
    return str(text).replace('/', '∕').strip()


def split_2dsep(gate_defs: list[dict], gate_name: str, steps: list[dict]) -> tuple[list[dict], list[str]]:
    """Replace a 2dsep gate with sequential 1dsep gates.

    *steps* gives the order: two dicts with 'channel' (one of the gate's
    two channels), 'algorithm', 'view' (view channel or None for a
    histogram) and 'sep_axis'. The first step's gate separates the parent
    population on its channel into '<channel>-' and '<channel>+'. On each
    side holding any of the 2dsep gate's populations, a second gate
    separates on the other channel; it carries those populations under
    their own names, so gates drawn on them keep their parent population
    and move to it. Per-axis parameters of the 2dsep gate
    (``algorithm_x``, ``tail_fraction_y``, …) go to the matching gate.

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
                                               params.get('tail_fraction', 0.01))),
             'mixture_components': int(params.get(f'mixture_components_{axis}',
                                                  params.get('mixture_components', 2)))}
        for key in ('axis_lo', 'axis_hi'):
            value = params.get(f'{key}_{axis}', 0.0)
            if value:
                p[key] = float(value)
        return p

    def sign(region: str, axis: str) -> str:
        return region[1] if axis == 'x' else region[3]

    regions = ag_core.population_regions(g)
    src = g.get('source') or {}
    a = _split_name(first['channel'])
    b = _split_name(second['channel'])
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
        if src.get('populations'):
            gate_b['source'] = {
                'origin': src.get('origin'), 'workspace': src.get('workspace'),
                'gating_source': src.get('gating_source'), 'sample': src.get('sample'),
                'parent_count': None,
                'populations': {p: v for p, v in src['populations'].items() if p in populations},
                'counts': {p: v for p, v in (src.get('counts') or {}).items() if p in populations},
            }
        new.append(gate_b)

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
