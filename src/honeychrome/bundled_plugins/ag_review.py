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
    'apply_name_sync', 'sync_conflicts', 'SYNC_USER', 'SYNC_AUTO', 'SYNC_DRAWN',
    'replicate_choices', 'descendants', 'delete_gates', 'delete_population',
    'copy_gates', 'paste_gates', 'rename_references',
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
    gated on). Labels follow ``population_choices``; reference populations
    are marked.
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
            if ((gate_def.get('populations') or {}).get(pop) or {}).get('reference'):
                label += '  [reference]'
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


def histogram_proposal(gate_def: dict, positive_fraction) -> str | None:
    """The algorithm the histogram rule proposes for an imported histogram
    gate whose proposal was set by the rule (``proposal_rule``), given the
    fraction of its parent on the positive side: 'mixture' above 25%, else
    'tail'. None when the rule does not apply."""
    if gate_def.get('proposal_rule') != 'histogram' or positive_fraction is None:
        return None
    import ag_flowjo
    return ag_flowjo.histogram_algorithm(positive_fraction)


def convert_target(gate_def: dict, recommendation: str | None = None,
                   positive_fraction=None):
    """The learning algorithm an imported gate converts to.

    Returns ``(algorithm, note)``; *algorithm* is None when the gate is not
    using its imported boundary. A histogram gate still on the import's rule
    (see histogram_proposal) uses *positive_fraction* when one is given;
    otherwise the algorithm proposed at import comes first, then for
    threshold gates the recommended algorithm when there is one, else
    'tail'; singlet gates use 'singlets' and scatter shapes 'cells'.
    """
    if gate_def.get('algorithm') != ag_core.ALGORITHM_IMPORTED:
        return None, ''
    gate_type = gate_def.get('gate_type')
    allowed = trainable_algorithms(gate_def)
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
    'quantile_trim', 'width_sd', 'debris_axis', 'debris_density', 'saturation_fraction',
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
    algorithm and the gate_param keys in SYNC_PARAM_KEYS."""
    gp = gate_def.get('gate_param') or {}
    return json.dumps({'algorithm': gate_def.get('algorithm'),
                       'params': {k: gp[k] for k in SYNC_PARAM_KEYS if k in gp}},
                      sort_keys=True, default=str)


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
    sources = {src.get('gate') for src in (tgp.get('replicate_x'), tgp.get('replicate_y'))
               if isinstance(src, dict) and src.get('gate')}
    sources |= {ag_core.split_population_key(src['population'])[0]
                for src in (tgp.get('replicate_x'), tgp.get('replicate_y'))
                if isinstance(src, dict) and src.get('population')}
    changed = []
    for m in members:
        if not _follows_sync(m):
            continue
        if m.get('algorithm') != ag_core.ALGORITHM_IMPORTED and sync_fingerprint(m) == target:
            continue
        below = {m['gate_name']} | descendants(gate_defs, m['gate_name'])
        if sources & below:
            continue
        gp = {k: v for k, v in (m.get('gate_param') or {}).items() if k not in SYNC_PARAM_KEYS}
        gp.update({k: deepcopy(tgp[k]) for k in SYNC_PARAM_KEYS if k in tgp})
        m['gate_param'] = gp
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


def replicate_choices(gate_defs: list[dict], gate_name: str, axis: str) -> list[tuple]:
    """Sources a replicated threshold of *gate_name* on *axis* can come from,
    as (label, source) pairs: other threshold gates separating on the same
    channel (their threshold is copied), then the reference populations
    (the threshold is calculated on their events). The gate itself and the
    gates below it are left out, since they are calculated after it."""
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
