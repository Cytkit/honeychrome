"""
ag_flowjo.py — FlowJo workspace import for the Gating Model Builder
====================================================================

Pure computation, no Qt and no controller: reads a FlowJo 10 workspace
(``.wsp``) or workspace template (``.wspt``), matches the parameters its
gates use to the experiment's unmixed channels, and converts the gates into
Gating Model Builder gate definitions with their imported boundaries.

The file name does not end in ``_tab.py``, so the plugin loader does not
treat it as a tab.

Coordinates
-----------
FlowJo stores gate coordinates in raw (untransformed) parameter units, and
draws polygon and ellipse edges on its own display scale (biex, logicle,
arcsinh or log). The parser keeps every coordinate in raw units. Conversion
to Honeychrome's display scale happens only in ``build_gate_definitions``:
rectangle bounds convert exactly; polygon edges are densified on FlowJo's
display scale first, so their shape survives the change of transform.

Workflow
--------
1. ``read_workspace(path)``         → FJWorkspace with one FJSource per group
                                       or sample that carries gates.
2. ``gated_parameters(source)``     → the FlowJo parameters the gates use.
3. ``align_parameters(...)``        → proposed {FlowJo parameter: channel}.
4. ``plan_import(source, mapping)`` → which populations import and which are
                                       dropped, with reasons.
5. ``build_gate_definitions(...)``  → gate definitions for the builder, each
                                       carrying its imported boundary as
                                       ``template`` (used by the ``imported``
                                       algorithm in ag_core).
"""

from __future__ import annotations

import difflib
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

import ag_core

__all__ = [
    'FJTransform', 'FJGate', 'FJPopulation', 'FJSource', 'FJWorkspace',
    'read_workspace', 'parse_workspace', 'gated_parameters', 'parameter_stem',
    'ExperimentChannels', 'align_parameters', 'mapping_problems',
    'MATCH_NAME', 'MATCH_FLUOROPHORE', 'MATCH_FLUOROPHORE_LABEL_DIFFERS',
    'MATCH_ANTIGEN', 'MATCH_FUZZY', 'MATCH_MANUAL', 'CONFIDENT_MATCHES',
    'ImportPlan', 'plan_import', 'build_gate_definitions',
    'ALGORITHM_IMPORTED',
]

ALGORITHM_IMPORTED = ag_core.ALGORITHM_IMPORTED

MATCH_NAME = 'name'                       # identical channel name (scatter, time)
MATCH_FLUOROPHORE = 'fluorophore'         # same fluorophore, label agrees or is absent
MATCH_FLUOROPHORE_LABEL_DIFFERS = 'fluorophore_label_differs'
MATCH_ANTIGEN = 'antigen'                 # fluorophore unknown; matched on the $PnS label
MATCH_FUZZY = 'fuzzy'                     # closest name only
MATCH_MANUAL = 'manual'                   # chosen by the user
CONFIDENT_MATCHES = frozenset({MATCH_NAME, MATCH_FLUOROPHORE, MATCH_MANUAL})

_POPULATION_NODES = ('Population', 'AndNode', 'OrNode', 'NotNode')
_AREA_SUFFIX = re.compile(r'-A$')
_HEIGHT_WIDTH_SUFFIX = re.compile(r'-[HW]$')
_DEFAULT_PREFIXES = ('Comp-', 'Unmixed-', 'Spectral-')


def _local(name: str) -> str:
    """Tag or attribute name without its XML namespace."""
    return name.rsplit('}', 1)[-1] if '}' in name else name.rsplit(':', 1)[-1]


def _attr(el, name: str, default=None):
    """Attribute by local name, ignoring any namespace prefix."""
    for k, v in el.attrib.items():
        if _local(k) == name:
            return v
    return default


def _children(el, name: str):
    return [c for c in el if _local(c.tag) == name]


def _child(el, name: str):
    for c in el:
        if _local(c.tag) == name:
            return c
    return None


def _float(v):
    if v is None or v == '':
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ---------------------------------------------------------------------------
# Parsed workspace
# ---------------------------------------------------------------------------

@dataclass
class FJTransform:
    """A FlowJo display transform for one parameter.

    kind   : 'linear', 'log', 'logicle', 'biex', 'fasinh', 'hyperlog' or the
             element name of anything else.
    params : the element's attributes (local names, values as strings).
    """
    kind: str
    params: dict = field(default_factory=dict)

    def display(self):
        """FlowKit transform reproducing FlowJo's display scale, or None
        when the scale is linear (edges stay straight) or unknown."""
        p = {k: _float(v) for k, v in self.params.items()}
        from flowkit import transforms as fk
        try:
            if self.kind == 'biex':
                return fk.WSPBiexTransform(
                    negative=p.get('neg') or 0.0, width=p.get('width') or -10.0,
                    positive=p.get('pos') or 4.418540,
                    max_value=p.get('maxRange') or 262144.0)
            if self.kind == 'logicle':
                return fk.LogicleTransform(
                    param_t=p.get('T') or 262144.0, param_w=p.get('W') or 0.5,
                    param_m=p.get('M') or 4.5, param_a=p.get('A') or 0.0)
            if self.kind == 'fasinh':
                return fk.AsinhTransform(
                    param_t=p.get('T') or 262144.0, param_m=p.get('M') or 4.5,
                    param_a=p.get('A') or 0.0)
            if self.kind == 'hyperlog':
                return fk.HyperlogTransform(
                    param_t=p.get('T') or 262144.0, param_w=p.get('W') or 0.5,
                    param_m=p.get('M') or 4.5, param_a=p.get('A') or 0.0)
            if self.kind == 'log':
                return fk.LogTransform(param_t=p.get('offset') or 1.0,
                                       param_m=p.get('decades') or 4.0)
        except Exception:
            return None
        return None


@dataclass
class FJGate:
    """One FlowJo gate in raw parameter units.

    kind      : 'rectangle', 'polygon', 'ellipse', or the XML element name of
                an unsupported gate type.
    channels  : FlowJo parameter names ($PnN), one per dimension.
    bounds    : rectangle (min, max) per dimension; None is open-ended.
    vertices  : polygon vertices.
    foci/edge : ellipse foci and the four edge points FlowJo stores.
    inside    : False when FlowJo selects the events outside the shape.
    quad_id   : FlowJo's quadrant index for quadrant pieces, else None.
    """
    kind: str
    channels: list = field(default_factory=list)
    bounds: list = field(default_factory=list)
    vertices: list = field(default_factory=list)
    foci: list = field(default_factory=list)
    edge: list = field(default_factory=list)
    inside: bool = True
    quad_id: int | None = None
    gate_id: str = ''
    parent_id: str | None = None


@dataclass
class FJPopulation:
    """A node of a FlowJo gating tree.

    node_type is 'Population' for a gated population; Boolean nodes
    ('AndNode', 'OrNode', 'NotNode') carry no gate of their own.
    path is the population names from the top of the tree to this node.
    """
    name: str
    path: tuple
    node_type: str = 'Population'
    gate: FJGate | None = None
    count: int | None = None
    children: list = field(default_factory=list)

    @property
    def key(self) -> str:
        return '/'.join(self.path)


@dataclass
class FJSource:
    """A gating tree that can be imported: a group's template gates or one
    sample's own gates, with the parameter labels and FlowJo transforms
    that apply to it."""
    kind: str
    name: str
    roots: list = field(default_factory=list)
    sample_id: str | None = None
    labels: dict = field(default_factory=dict)
    transforms: dict = field(default_factory=dict)
    count: int | None = None

    def iter_populations(self):
        """Every node, parents before children."""
        stack = list(reversed(self.roots))
        while stack:
            pop = stack.pop()
            yield pop
            stack.extend(reversed(pop.children))

    @property
    def n_populations(self) -> int:
        return sum(1 for _ in self.iter_populations())

    @property
    def title(self) -> str:
        kind = 'Group' if self.kind == 'group' else 'Sample'
        return f"{kind}: {self.name} ({self.n_populations} populations)"


@dataclass
class FJWorkspace:
    path: str
    flowjo_version: str = ''
    sources: list = field(default_factory=list)
    affixes: list = field(default_factory=list)
    cytometer: str = ''

    def default_source(self) -> FJSource | None:
        """The group with the most populations, else the sample with the most."""
        groups = [s for s in self.sources if s.kind == 'group' and s.n_populations]
        if groups:
            return max(groups, key=lambda s: s.n_populations)
        samples = [s for s in self.sources if s.n_populations]
        return max(samples, key=lambda s: s.n_populations) if samples else None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def read_workspace(path) -> FJWorkspace:
    """Parse a FlowJo ``.wsp`` or ``.wspt`` file."""
    root = ET.parse(str(path)).getroot()
    return parse_workspace(root, str(path))


def parse_workspace(root, path: str = '') -> FJWorkspace:
    """Parse a FlowJo workspace from its XML root element."""
    if _local(root.tag) != 'Workspace':
        raise ValueError("Not a FlowJo workspace: the root element is "
                         f"<{_local(root.tag)}>, not <Workspace>.")
    ws = FJWorkspace(path=path, flowjo_version=_attr(root, 'flowJoVersion', '') or '')
    ws.affixes = _matrix_affixes(root)

    cyt_transforms = {}
    cyts = _child(root, 'Cytometers')
    if cyts is not None:
        for cyt in _children(cyts, 'Cytometer'):
            ws.cytometer = ws.cytometer or (_attr(cyt, 'cyt', '') or '')
            store = _child(cyt, 'TransformStore')
            if store is not None:
                for xf_parent in store.iter():
                    if _local(xf_parent.tag) == 'Transforms':
                        cyt_transforms.update(_parse_transforms(xf_parent))

    samples = {}
    sample_list = _child(root, 'SampleList')
    for s in (_children(sample_list, 'Sample') if sample_list is not None else []):
        node = _child(s, 'SampleNode')
        if node is None:
            continue
        sid = _attr(node, 'sampleID') or _attr(_child(s, 'DataSet') or node, 'sampleID')
        xf_el = _child(s, 'Transformations')
        transforms = dict(cyt_transforms)
        if xf_el is not None:
            transforms.update(_parse_transforms(xf_el))
        labels = _parameter_labels(_child(s, 'Keywords'))
        src = FJSource(kind='sample', name=_attr(node, 'name', '') or f'Sample {sid}',
                       sample_id=sid, labels=labels, transforms=transforms,
                       count=_int(_attr(node, 'count')))
        src.roots = _parse_subpopulations(node, ())
        samples[sid] = src

    groups = []
    groups_el = _child(root, 'Groups')
    for gnode in (_children(groups_el, 'GroupNode') if groups_el is not None else []):
        roots = _parse_subpopulations(gnode, ())
        if not roots:
            continue
        group = _child(gnode, 'Group')
        refs = []
        if group is not None:
            sr = _child(group, 'SampleRefs')
            refs = [_attr(r, 'sampleID') for r in (_children(sr, 'SampleRef') if sr is not None else [])]
        member = next((samples[r] for r in refs if r in samples), None)
        groups.append(FJSource(
            kind='group', name=_attr(gnode, 'name', '') or 'Group', roots=roots,
            labels=dict(member.labels) if member else {},
            transforms=dict(member.transforms) if member else dict(cyt_transforms),
        ))
    ws.sources = groups + [s for s in samples.values()]
    return ws


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _matrix_affixes(root) -> list:
    """(prefix, suffix) of every compensation or unmixing matrix."""
    out = []
    for el in root.iter():
        if _local(el.tag) == 'spilloverMatrix':
            pair = (_attr(el, 'prefix', '') or '', _attr(el, 'suffix', '') or '')
            if pair != ('', '') and pair not in out:
                out.append(pair)
    for p in _DEFAULT_PREFIXES:
        if (p, '') not in out:
            out.append((p, ''))
    return out


def _parse_transforms(el) -> dict:
    """{parameter name: FJTransform} from a Transformations/Transforms element."""
    out = {}
    for xf in el:
        params = {_local(k): v for k, v in xf.attrib.items()}
        for p in xf.iter():
            if _local(p.tag) == 'parameter':
                name = _attr(p, 'name')
                if name:
                    out[name] = FJTransform(kind=_local(xf.tag), params=params)
    return out


def _parameter_labels(keywords_el) -> dict:
    """{$PnN: $PnS} from a sample's Keywords element."""
    if keywords_el is None:
        return {}
    kw = {}
    for k in _children(keywords_el, 'Keyword'):
        kw[_attr(k, 'name', '')] = _attr(k, 'value', '') or ''
    labels = {}
    for key, value in kw.items():
        m = re.fullmatch(r'\$P(\d+)N', key)
        if m:
            labels[value] = (kw.get(f'$P{m.group(1)}S') or '').strip()
    return labels


def _parse_subpopulations(node, path: tuple) -> list:
    sub = _child(node, 'Subpopulations')
    if sub is None:
        return []
    out = []
    for child in sub:
        tag = _local(child.tag)
        if tag not in _POPULATION_NODES:
            continue
        name = _attr(child, 'name', '') or ''
        pop = FJPopulation(name=name, path=path + (name,), node_type=tag,
                           count=_int(_attr(child, 'count')))
        if tag == 'Population':
            gate_el = _child(child, 'Gate')
            if gate_el is not None:
                pop.gate = _parse_gate(gate_el)
        pop.children = _parse_subpopulations(child, pop.path)
        out.append(pop)
    return out


def _dimension_name(dim) -> str:
    for el in dim.iter():
        if _local(el.tag) in ('fcs-dimension', 'new-dimension'):
            return _attr(el, 'name', '') or ''
    return ''


def _vertices(el) -> list:
    out = []
    for v in _children(el, 'vertex'):
        coords = [_float(_attr(c, 'value')) for c in _children(v, 'coordinate')]
        if len(coords) >= 2 and None not in coords[:2]:
            out.append((coords[0], coords[1]))
    return out


def _parse_gate(gate_el) -> FJGate:
    shape = next((c for c in gate_el if _local(c.tag).endswith('Gate')), None)
    if shape is None:
        return FJGate(kind='none', gate_id=_attr(gate_el, 'id', '') or '')
    tag = _local(shape.tag)
    dims = _children(shape, 'dimension')
    kind = {'RectangleGate': 'rectangle', 'PolygonGate': 'polygon',
            'EllipsoidGate': 'ellipse'}.get(tag, tag)
    gate = FJGate(
        kind=kind,
        channels=[_dimension_name(d) for d in dims],
        inside=(_attr(shape, 'eventsInside', '1') or '1') != '0',
        gate_id=_attr(gate_el, 'id', '') or '',
        parent_id=_attr(gate_el, 'parent_id'),
    )
    qid = _int(_attr(shape, 'quadId'))
    gate.quad_id = qid if qid is not None and qid >= 0 else None
    if kind == 'rectangle':
        gate.bounds = [(_float(_attr(d, 'min')), _float(_attr(d, 'max'))) for d in dims]
    elif kind == 'polygon':
        gate.vertices = _vertices(shape)
    elif kind == 'ellipse':
        foci = _child(shape, 'foci')
        edge = _child(shape, 'edge')
        gate.foci = _vertices(foci) if foci is not None else []
        gate.edge = _vertices(edge) if edge is not None else []
    return gate


def gated_parameters(source: FJSource) -> list[str]:
    """FlowJo parameters used by the source's gates, in order of first use."""
    out = []
    for pop in source.iter_populations():
        for ch in (pop.gate.channels if pop.gate else []):
            if ch and ch not in out:
                out.append(ch)
    return out


def parameter_stem(name: str, affixes) -> str:
    """Parameter name without its matrix prefix/suffix (e.g. 'Comp-')."""
    stem = str(name)
    for prefix, suffix in affixes or [(p, '') for p in _DEFAULT_PREFIXES]:
        if prefix and stem.startswith(prefix):
            stem = stem[len(prefix):]
        if suffix and stem.endswith(suffix):
            stem = stem[:-len(suffix)]
    return stem


# ---------------------------------------------------------------------------
# Parameter alignment
# ---------------------------------------------------------------------------

@dataclass
class ExperimentChannels:
    """The experiment side of the alignment.

    channels    : unmixed channel names ($PnN), in order.
    fluorescence: the fluorescence channels among them.
    scatter     : the scatter channels.
    antigens    : {channel: antigen} from the spectral model.
    transforms  : {channel: transform parameter dict} (ag_core.transform_params).
    singlet_pair: (area, height-or-width) forward-scatter pair, or None.
    """
    channels: list
    fluorescence: list = field(default_factory=list)
    scatter: list = field(default_factory=list)
    antigens: dict = field(default_factory=dict)
    transforms: dict = field(default_factory=dict)
    singlet_pair: tuple | None = None


def _norm(name: str) -> str:
    return re.sub(r'[\s_\-.+]+', '', str(name or '')).lower()


def _label_tokens(label: str) -> list[str]:
    """A $PnS label split into marker names: 'CD8_CD11b' → ['CD8', 'CD11b']."""
    return [t.strip() for t in re.split(r'[_/,;|]+', str(label or '')) if t.strip()]


def align_parameters(params: list[str], labels: dict, exp: ExperimentChannels,
                     affixes=None, fluor_match=None, marker_match=None) -> dict[str, dict]:
    """Propose an experiment channel for each FlowJo parameter.

    Matching order:
      1. the same channel name, after removing the matrix prefix
         (scatter, time and anything else named identically);
      2. the same fluorophore (fluorophore_database names and synonyms) on
         an area parameter; the $PnS label is checked against the channel's
         antigen and a disagreement is reported;
      3. the $PnS label against the channel antigens, marker by marker for
         combined labels such as 'CD4_IgM';
      4. the closest channel name (difflib, cutoff 0.8).
    Fluorescence height and width parameters are not matched: Honeychrome
    unmixes area only.

    fluor_match  : callable name → canonical fluorophore or None.
    marker_match : callable name → canonical marker or None.

    Returns {parameter: {'channel', 'method', 'note', 'label'}}.
    """
    fluor_match = fluor_match or (lambda _n: None)
    marker_match = marker_match or (lambda _n: None)
    scatter = set(exp.scatter)

    def safe(fn, value):
        try:
            return fn(value)
        except Exception:
            return None

    exp_fluor: dict[str, list[str]] = {}
    for ch in exp.fluorescence:
        key = safe(fluor_match, ch) or ch
        exp_fluor.setdefault(_norm(key), []).append(ch)

    def marker_key(name):
        return _norm(safe(marker_match, name) or name)

    exp_marker: dict[str, list[str]] = {}
    for ch in exp.fluorescence:
        antigen = (exp.antigens.get(ch) or '').strip()
        if antigen and antigen != ch:
            exp_marker.setdefault(marker_key(antigen), []).append(ch)

    out = {}
    for p in params:
        label = (labels or {}).get(p, '') or ''
        stem = parameter_stem(p, affixes)
        channel, method, note = None, None, ''
        tokens = [marker_key(t) for t in _label_tokens(label)]

        if stem in exp.channels or p in exp.channels:
            channel, method = (stem if stem in exp.channels else p), MATCH_NAME
        elif _HEIGHT_WIDTH_SUFFIX.search(stem) and not any(
                stem.startswith(s.rsplit('-', 1)[0]) for s in scatter):
            note = 'Height and width parameters are not unmixed in Honeychrome.'
        else:
            base = _AREA_SUFFIX.sub('', stem)
            canon = safe(fluor_match, base)
            hits = exp_fluor.get(_norm(canon), []) if canon else []
            if len(hits) == 1:
                channel, method = hits[0], MATCH_FLUOROPHORE
                antigen = (exp.antigens.get(channel) or '').strip()
                if tokens and antigen and antigen != channel \
                        and marker_key(antigen) not in tokens:
                    method = MATCH_FLUOROPHORE_LABEL_DIFFERS
                    note = f"FlowJo label '{label}', experiment antigen '{antigen}'."
            elif len(hits) > 1:
                channel, method = hits[0], MATCH_FUZZY
                note = 'Fluorophore is on more than one channel: ' + ', '.join(hits)
            if channel is None and tokens:
                found = []
                for t in tokens:
                    for ch in exp_marker.get(t, []):
                        if ch not in found:
                            found.append(ch)
                if len(found) == 1:
                    channel, method = found[0], MATCH_ANTIGEN
                    fl = f" ({canon})" if canon else ''
                    note = f"Matched on the label '{label}'; fluorophore{fl} not found."
                elif len(found) > 1:
                    channel, method = found[0], MATCH_FUZZY
                    note = f"Label '{label}' matches " + ', '.join(found)
            if channel is None:
                pool = list(exp.fluorescence) + list(exp.scatter)
                close = difflib.get_close_matches(base, pool, n=1, cutoff=0.8)
                if close:
                    channel, method = close[0], MATCH_FUZZY
                    note = 'Closest channel name only.'
        out[p] = {'channel': channel, 'method': method, 'note': note, 'label': label}
    return out


def mapping_problems(mapping: dict) -> list[str]:
    """Reasons a {FlowJo parameter: channel or None} mapping cannot be
    finalised. A dropped parameter is mapped to None explicitly and is not a
    problem; a parameter missing from the mapping is."""
    problems = []
    used: dict[str, list[str]] = {}
    for p, ch in mapping.items():
        if ch:
            used.setdefault(ch, []).append(p)
    for ch, ps in used.items():
        stems = {_AREA_SUFFIX.sub('', parameter_stem(p, None)) for p in ps}
        if len(stems) > 1:
            problems.append(f"{ch} is assigned to more than one FlowJo parameter: "
                            + ", ".join(ps))
    return problems


# ---------------------------------------------------------------------------
# Import planning: which populations survive the mapping
# ---------------------------------------------------------------------------

@dataclass
class ImportPlan:
    """keep    : population keys that will be imported, in tree order.
    dropped : {population key: reason}; a dropped population's descendants
              are dropped with it.
    """
    keep: list = field(default_factory=list)
    dropped: dict = field(default_factory=dict)

    def messages(self) -> list[str]:
        """One line per dropped branch (descendants summarised)."""
        lines = []
        tops = [k for k in self.dropped
                if not any(k != o and k.startswith(o + '/') for o in self.dropped)]
        for k in tops:
            n_desc = sum(1 for o in self.dropped if o.startswith(k + '/'))
            tail = f" and {n_desc} population(s) below it" if n_desc else ''
            lines.append(f"Dropped '{k}'{tail}: {self.dropped[k]}")
        return lines


def _unsupported_reason(pop: FJPopulation) -> str | None:
    if pop.node_type != 'Population':
        return f"Boolean gates ({pop.node_type.replace('Node', '')}) are not imported."
    g = pop.gate
    if g is None or g.kind == 'none':
        return 'The population has no gate.'
    if g.kind not in ('rectangle', 'polygon', 'ellipse'):
        return f"{g.kind} gates are not supported."
    if not g.inside:
        return 'Gates that select events outside the shape are not supported.'
    if g.kind == 'polygon' and (len(g.vertices) < 3 or len(g.channels) != 2):
        return 'The polygon is incomplete.'
    if g.kind == 'ellipse' and (len(g.edge) < 4 or len(g.channels) != 2):
        return 'The ellipse is incomplete.'
    if g.kind == 'rectangle' and not any(lo is not None or hi is not None for lo, hi in g.bounds):
        return 'The rectangle has no bounds.'
    return None


def plan_import(source: FJSource, mapping: dict) -> ImportPlan:
    """Decide which populations import under *mapping*.

    A population is dropped when its gate type is unsupported or any of its
    parameters maps to no channel; every population below it is dropped
    with it."""
    plan = ImportPlan()

    def visit(pop: FJPopulation, parent_dropped: str | None):
        reason = None
        if parent_dropped is not None:
            reason = f"its parent '{parent_dropped}' was dropped."
        else:
            reason = _unsupported_reason(pop)
            if reason is None:
                missing = [ch for ch in pop.gate.channels if not mapping.get(ch)]
                if missing:
                    reason = 'no channel for ' + ', '.join(missing) + '.'
        if reason is None:
            plan.keep.append(pop.key)
            for c in pop.children:
                visit(c, None)
        else:
            plan.dropped[pop.key] = reason
            top = parent_dropped if parent_dropped is not None else pop.key
            for c in pop.children:
                visit(c, top)

    for r in source.roots:
        visit(r, None)
    return plan


# ---------------------------------------------------------------------------
# Conversion to gate definitions
# ---------------------------------------------------------------------------

def _safe_name(name: str) -> str:
    """Population and gate names may not contain '/' (it separates gate and
    population in population keys)."""
    return str(name).replace('/', '∕').strip() or 'gate'


class _Axes:
    """Raw ↔ display conversions for one experiment channel and the FlowJo
    parameter mapped to it."""

    def __init__(self, fj_param: str, channel: str, source: FJSource, exp: ExperimentChannels):
        self.fj_param = fj_param
        self.channel = channel
        fjt = source.transforms.get(fj_param)
        self.fj = fjt.display() if fjt is not None else None
        params = exp.transforms.get(channel)
        self.hc = ag_core.make_transform(params) if params else None
        self.hc_linear = not params or params.get('id') == 0
        lim = (params or {}).get('limits')
        self.limits = (float(lim[0]), float(lim[1])) if lim and len(lim) >= 2 else None
        self.zero = float(self.to_display([0.0])[0])

    def to_display(self, raw):
        raw = np.asarray(raw, dtype=float)
        return np.asarray(self.hc.apply(raw), dtype=float) if self.hc is not None else raw

    def to_fj(self, raw):
        raw = np.asarray(raw, dtype=float)
        return np.asarray(self.fj.apply(raw), dtype=float) if self.fj is not None else raw

    def from_fj(self, disp):
        disp = np.asarray(disp, dtype=float)
        return np.asarray(self.fj.inverse(disp), dtype=float) if self.fj is not None else disp


def _polygon_display(raw_xy, ax: _Axes, ay: _Axes, points_per_edge: int = 32) -> list:
    """Raw polygon (FlowJo edges straight on FlowJo's display scale) →
    closed polygon on Honeychrome's display scale."""
    raw_xy = np.asarray(raw_xy, dtype=float)
    if len(raw_xy) and not np.allclose(raw_xy[0], raw_xy[-1]):
        raw_xy = np.vstack([raw_xy, raw_xy[:1]])
    straight = (ax.fj is None and ay.fj is None and ax.hc_linear and ay.hc_linear)
    if not straight:
        fj_xy = np.column_stack([ax.to_fj(raw_xy[:, 0]), ay.to_fj(raw_xy[:, 1])])
        fj_xy = ag_core.densify_polygon(fj_xy, points_per_edge)
        raw_xy = np.column_stack([ax.from_fj(fj_xy[:, 0]), ay.from_fj(fj_xy[:, 1])])
    disp = np.column_stack([ax.to_display(raw_xy[:, 0]), ay.to_display(raw_xy[:, 1])])
    return [[float(a), float(b)] for a, b in disp]


def _ellipse_raw(gate: FJGate, ax: _Axes, ay: _Axes, n_points: int = 96) -> list:
    """Raw polygon tracing a FlowJo ellipse. FlowJo's four edge points are
    the ends of the two axes on its display scale."""
    e = np.asarray(gate.edge[:4], dtype=float)
    fj = np.column_stack([ax.to_fj(e[:, 0]), ay.to_fj(e[:, 1])])
    centre = fj.mean(axis=0)
    u = (fj[0] - fj[1]) / 2.0
    v = (fj[2] - fj[3]) / 2.0
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    pts = centre + np.outer(np.cos(t), u) + np.outer(np.sin(t), v)
    raw = np.column_stack([ax.from_fj(pts[:, 0]), ay.from_fj(pts[:, 1])])
    return [tuple(p) for p in raw]


def _side(lo, zero_disp: float) -> str:
    """'+' when an interval (display units) lies above the channel's raw
    zero, '-' when it contains or lies below it."""
    if lo is not None and lo > zero_disp:
        return '+'
    return '-'


def _threshold_for(lo, hi, side: str):
    return lo if side == '+' else hi


def _axis_sign(piece, axis: int) -> str:
    """'+' or '-' side of a threshold piece on one axis."""
    if len(piece.channels) == 2:
        return piece.region[1] if axis == 0 else piece.region[3]
    return '+' if piece.region == ag_core.REGION_ABOVE else '-'


@dataclass
class _Piece:
    """One FlowJo population converted to display units, before grouping."""
    pop: FJPopulation
    kind: str                      # 'threshold', 'polygon', 'range'
    channels: list                 # experiment channels
    display_bounds: list = field(default_factory=list)
    polygon: list = field(default_factory=list)
    region: str = ''
    notes: list = field(default_factory=list)
    singlets: bool = False


def _piece(pop: FJPopulation, source: FJSource, mapping: dict,
           exp: ExperimentChannels) -> _Piece:
    g = pop.gate
    chans = [mapping[c] for c in g.channels]
    axes = [_Axes(fj, ch, source, exp) for fj, ch in zip(g.channels, chans)]
    scatter = set(exp.scatter)
    if g.kind in ('polygon', 'ellipse'):
        raw = g.vertices if g.kind == 'polygon' else _ellipse_raw(g, axes[0], axes[1])
        poly = _polygon_display(raw, axes[0], axes[1])
        single = bool(exp.singlet_pair) and set(chans) == set(exp.singlet_pair)
        notes = ['Ellipse traced as a polygon.'] if g.kind == 'ellipse' else []
        return _Piece(pop, 'polygon', chans, polygon=poly, notes=notes, singlets=single)

    bounds = []
    for (lo, hi), ax in zip(g.bounds, axes):
        dlo = float(ax.to_display([lo])[0]) if lo is not None else None
        dhi = float(ax.to_display([hi])[0]) if hi is not None else None
        bounds.append((dlo, dhi))
    has_scatter = any(c in scatter for c in chans)
    if len(chans) == 2 and has_scatter:
        x0, x1 = _limits_fill(bounds[0], axes[0])
        y0, y1 = _limits_fill(bounds[1], axes[1])
        poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        return _Piece(pop, 'polygon', chans, display_bounds=bounds, polygon=poly)
    if len(chans) == 1 and has_scatter:
        lo, hi = _limits_fill(bounds[0], axes[0])
        return _Piece(pop, 'range', chans, display_bounds=[(lo, hi)])

    sides = [_side(lo, ax.zero) for (lo, _hi), ax in zip(bounds, axes)]
    notes = []
    for (lo, hi), side, ax, fj in zip(bounds, sides, axes, g.channels):
        outer = hi if side == '+' else lo
        if outer is None:
            continue
        if ax.limits is not None and (outer >= ax.limits[1] or outer <= ax.limits[0]):
            continue
        which = 'upper' if side == '+' else 'lower'
        notes.append(f"{fj}: the {which} bound is not kept (threshold gates are open-ended).")
    if len(chans) == 1:
        region = ag_core.REGION_ABOVE if sides[0] == '+' else ag_core.REGION_BELOW
    else:
        region = f"x{sides[0]}y{sides[1]}"
    return _Piece(pop, 'threshold', chans, display_bounds=bounds, region=region, notes=notes)


def _limits_fill(bound, ax: _Axes) -> tuple[float, float]:
    """Replace open rectangle sides with the display limits."""
    lo, hi = bound
    lim = ax.limits or (0.0, 1.0)
    return (lo if lo is not None else float(lim[0]), hi if hi is not None else float(lim[1]))


def _merge_threshold(pieces: list[_Piece], axis: int) -> tuple[float, list[str]]:
    """One threshold (display units) for an axis shared by sibling pieces:
    the midpoint between the highest '-' side bound and the lowest '+' side
    bound, or the mean when all pieces lie on one side."""
    neg, pos = [], []
    for p in pieces:
        lo, hi = p.display_bounds[axis]
        sign = _axis_sign(p, axis)
        t = _threshold_for(lo, hi, sign)
        if t is None:
            continue
        (pos if sign == '+' else neg).append(t)
    notes = []
    if neg and pos:
        a, b = max(neg), min(pos)
        t = 0.5 * (a + b)
        if abs(a - b) > 1e-9:
            notes.append(f"FlowJo bounds {a:.3f} and {b:.3f} are joined at {t:.3f} (display units).")
        return t, notes
    vals = neg or pos
    t = float(np.mean(vals)) if vals else 0.5
    if len(vals) > 1 and (max(vals) - min(vals)) > 1e-9:
        notes.append(f"FlowJo bounds {min(vals):.3f}–{max(vals):.3f} are averaged to {t:.3f}.")
    return t, notes


def _inner_thresholds(piece: _Piece) -> list:
    """The kept bound of each axis of a threshold piece (display units)."""
    return [_threshold_for(lo, hi, _axis_sign(piece, k))
            for k, (lo, hi) in enumerate(piece.display_bounds)]


def _compatible(a: _Piece, b: _Piece, tolerance: float) -> bool:
    """True when two threshold pieces could share one threshold per axis."""
    for ta, tb in zip(_inner_thresholds(a), _inner_thresholds(b)):
        if ta is not None and tb is not None and abs(ta - tb) > tolerance:
            return False
    return True


def _group_siblings(pieces: list[_Piece], merge: bool,
                    tolerance: float) -> list[list[_Piece]]:
    """Sibling threshold pieces on the same axes, in distinct regions and
    with thresholds within *tolerance* of each other (display units), become
    one gate; everything else stays on its own."""
    groups: list[list[_Piece]] = []
    for p in pieces:
        if merge and p.kind == 'threshold':
            for grp in groups:
                q = grp[0]
                if q.kind == 'threshold' and q.channels == p.channels \
                        and p.region not in {r.region for r in grp} \
                        and all(_compatible(p, r, tolerance) for r in grp):
                    grp.append(p)
                    break
            else:
                groups.append([p])
        else:
            groups.append([p])
    return groups


def build_gate_definitions(source: FJSource, mapping: dict, exp: ExperimentChannels,
                           merge_siblings: bool = True,
                           merge_tolerance: float = 0.01,
                           existing_names: set | None = None,
                           workspace_name: str = '') -> tuple[list[dict], dict]:
    """Convert the importable populations of *source* into gate definitions.

    mapping : {FlowJo parameter: experiment channel or None}; populations
              using an unmapped parameter are dropped (see plan_import).

    Conversion rules:
      * polygon on the forward-scatter singlet pair → 'singlets';
      * other polygons, ellipses, and rectangles with a scatter axis →
        'free' polygons (rectangles keep all four sides);
      * 1-D range on a scatter channel → 'free' 1-D range;
      * rectangles on fluorescence channels → '1dsep'/'2dsep' threshold
        gates. Each axis keeps the bound nearer the channel's zero side
        boundary: the lower bound when the interval lies above raw zero,
        the upper bound when it contains or lies below zero. Sibling
        rectangles on the same axes, in distinct regions, whose kept
        bounds agree within *merge_tolerance* display units (FlowJo
        quadrants, split gates) become one gate with one population each.
        Siblings whose bounds differ stay separate gates, so each keeps
        its own thresholds.

    Every gate gets algorithm 'imported' and a ``template`` holding its
    boundary entries in Honeychrome display units, plus ``source`` with the
    FlowJo population paths and counts.

    Returns (gate_defs, report) where report has 'plan' (ImportPlan),
    'notes' ({gate name: [str]}) and 'populations' ({FlowJo population key:
    'gate/pop'}).
    """
    plan = plan_import(source, mapping)
    keep = set(plan.keep)
    used = set(existing_names or ())
    gate_defs: list[dict] = []
    notes: dict[str, list[str]] = {}
    pop_map: dict[str, tuple[str, str]] = {}

    def unique(name: str) -> str:
        base = _safe_name(name)
        cand, k = base, 2
        while cand in used:
            cand = f"{base} ({k})"
            k += 1
        used.add(cand)
        return cand

    def emit(parent: FJPopulation | None, children: list[FJPopulation]):
        kids = [c for c in children if c.key in keep]
        if not kids:
            return
        parent_gate, parent_pop = (pop_map[parent.key] if parent is not None
                                   else (None, 'root'))
        pieces = [_piece(c, source, mapping, exp) for c in kids]
        for grp in _group_siblings(pieces, merge_siblings, merge_tolerance):
            names = [p.pop.name for p in grp]
            gname = unique(names[0] if len(grp) == 1 else ' & '.join(names))
            gdef, gnotes, pops = _gate_def(grp, gname, parent_gate, parent_pop, exp)
            gdef['gate_number'] = len(gate_defs) + 1
            gdef['source'] = {
                'origin': 'FlowJo',
                'workspace': workspace_name,
                'gating_source': source.title,
                'populations': {pname: p.pop.key for pname, p in pops.items()},
                'counts': {pname: p.pop.count for pname, p in pops.items()},
                'parameters': {ch: fj for p in grp for fj, ch in zip(p.pop.gate.channels, p.channels)},
            }
            gate_defs.append(gdef)
            notes[gname] = gnotes
            for pname, p in pops.items():
                pop_map[p.pop.key] = (gname, pname)
        for c in kids:
            emit(c, c.children)

    emit(None, source.roots)
    report = {
        'plan': plan,
        'notes': notes,
        'populations': {k: ag_core.population_key(*v) for k, v in pop_map.items()},
    }
    return gate_defs, report


def _gate_def(grp: list[_Piece], gname: str, parent_gate, parent_pop,
              exp: ExperimentChannels) -> tuple[dict, list[str], dict]:
    first = grp[0]
    gnotes = [n for p in grp for n in p.notes]
    pops_by_name: dict[str, _Piece] = {}
    base = {
        'gate_name': gname,
        'gate_marker_x': first.channels[0],
        'gate_marker_y': first.channels[1] if len(first.channels) > 1 else None,
        'parent_gate': parent_gate,
        'parent_popul': parent_pop,
        'algorithm': ALGORITHM_IMPORTED,
        'gate_param': {},
        'stats_parent': {},
        'origin_gate': None,
    }

    if first.kind in ('polygon', 'range'):
        pname = _safe_name(first.pop.name)
        gtype = 'singlets' if first.singlets else 'free'
        populations = {pname: {'label': pname, 'label_pos': 1, 'region': ag_core.REGION_INSIDE}}
        if first.kind == 'range':
            lo, hi = first.display_bounds[0]
            entry = {'boundary': [[lo, 0.0], [hi, 0.0], [hi, 1.0], [lo, 1.0], [lo, 0.0]],
                     'range': [lo, hi]}
        else:
            entry = {'boundary': first.polygon}
        entry.update({'threshold_x': None, 'threshold_y': None,
                      'region': ag_core.REGION_INSIDE, 'label': pname})
        base.update({'gate_type': gtype, 'populations': populations,
                     'template': {pname: entry}})
        pops_by_name[pname] = first
        return base, gnotes, pops_by_name

    two_d = len(first.channels) == 2
    gtype = '2dsep' if two_d else '1dsep'
    defaults = ag_core.default_populations_for_type(gtype)
    region_to_default = {v['region']: k for k, v in defaults.items()}
    populations: dict[str, dict] = {}
    taken = set()
    for p in grp:
        pname = _safe_name(p.pop.name)
        populations[pname] = {'label': pname,
                              'label_pos': defaults[region_to_default[p.region]]['label_pos'],
                              'region': p.region}
        taken.add(p.region)
        pops_by_name[pname] = p
    for dname, d in defaults.items():
        if d['region'] not in taken:
            name = dname if dname not in populations else f"{dname} (other)"
            populations[name] = dict(d, label=name)
    tx, nx_ = _merge_threshold(grp, 0)
    ty, ny_ = (_merge_threshold(grp, 1) if two_d else (None, []))
    gnotes += nx_ + ny_
    lims = {}
    for ch in first.channels:
        lim = (exp.transforms.get(ch) or {}).get('limits')
        if lim:
            lims[ch] = (float(lim[0]), float(lim[1]))
    gdef = dict(base, gate_type=gtype, populations=populations)
    entries = ag_core.GateCalculator(lims).threshold_boundaries(gdef, tx, ty)
    for name, entry in entries.items():
        entry['label'] = populations[name]['label']
    gdef['template'] = entries
    return gdef, gnotes, pops_by_name
