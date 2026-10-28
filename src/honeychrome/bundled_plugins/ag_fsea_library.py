"""
ag_fsea_library.py — immune state library and feature matching for Flow Set
Enrichment Analysis (FSEA) in the Automated Gating plugin
==========================================================================
Pure computation (no Qt):

* ``load_cell_types`` — the canonical cell types (``ag_fsea_cell_types.csv``)
  with their lineage and the population names that denote them.
* ``load_library`` — the immune state library (``ag_fsea_library.csv`` or a
  user file): validation, species filter and SHA-256.
* ``population_info`` — every gated population classified as a positivity
  population (one-population 1dsep gate whose label is a marker with a sign,
  e.g. ``PD1+``, read as "fraction of the parent positive for the marker")
  or a phenotype population (everything else), with the marker named by the
  gate label.
* ``map_populations`` — phenotype populations to canonical cell types:
  a user override, else the population label against the cell-type names
  and synonyms, accepted only when the lineage agrees with the population's
  ancestors (so a B-cell "Memory" gate never becomes a memory T cell).
* ``resolve_channel_markers`` — the marker a channel measures in a given
  population. Panels that re-use one fluorochrome for lineage-exclusive
  markers (IMMPROVE: CD28/CD20, GATA3/CD23, …) have one antigen per channel
  in the experiment, so the marker is taken from the positivity gates drawn
  on that channel nearest to the population, and from the channel's antigen
  only when no gate label contradicts it.
* ``match_library`` / ``build_sets`` — library rows to features, and the
  sets tested.

Matching is at the exact phenotype only: a library cell type matches the
population mapped to it, never its descendants.

v1 simplifications: cell types are matched by name (with the lineage check)
rather than derived from the marker signs along the gate path; library
members are directed (UP/DOWN) and unweighted in scoring (weights order the
display only).

The file name does not end in ``_tab.py``, so the plugin loader does not
treat it as a tab.
"""

from __future__ import annotations

import hashlib
import io
import re
from collections import Counter
from pathlib import Path

import pandas as pd

import ag_core

log = ag_core.get_logger(__name__)

_HERE = Path(__file__).resolve().parent
LIBRARY_FILE = _HERE / 'ag_fsea_library.csv'
CELL_TYPES_FILE = _HERE / 'ag_fsea_cell_types.csv'

LIBRARY_COLUMNS = ('immune_state', 'description', 'cell_type', 'feature_kind', 'marker',
                   'direction', 'weight', 'gene_mapping', 'species', 'source_model',
                   'citation', 'curation_status', 'library_version')
REQUIRED_COLUMNS = ('immune_state', 'cell_type', 'feature_kind', 'marker', 'direction')
FEATURE_KINDS = ('Abund', 'Marker')
DIRECTIONS = ('UP', 'DOWN')
WEIGHTS = ('Critical', 'High', 'Medium')
SPECIES = ('human', 'mouse')
READOUTS = ('positivity', 'median')

# A positivity population's label: a marker followed by its sign.
POSITIVITY_LABEL = re.compile(r'^(?P<marker>\S+?)(?P<sign>\+|-|hi|lo|dim|bright)$')

__all__ = [
    'LIBRARY_FILE', 'CELL_TYPES_FILE', 'LIBRARY_COLUMNS', 'FEATURE_KINDS', 'DIRECTIONS',
    'WEIGHTS', 'SPECIES', 'READOUTS',
    'canonical_marker', 'load_cell_types', 'load_library', 'population_info',
    'map_populations', 'resolve_channel_markers', 'match_library', 'build_sets',
]


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

def _norm(text) -> str:
    """Case-, accent- and whitespace-insensitive form of a name."""
    s = str(text or '').replace('ï', 'i').replace('Ï', 'I')
    return re.sub(r'\s+', ' ', s).strip().lower()


def canonical_marker(name: str) -> str | None:
    """Canonical marker name from the marker database, or None."""
    from honeychrome.controller_components import label_matching
    text = str(name or '').strip()
    if not text:
        return None
    return label_matching.match_marker(text, label_matching.get_marker_db())


# ---------------------------------------------------------------------------
# Cell types
# ---------------------------------------------------------------------------

_cell_type_cache: dict[str, pd.DataFrame] = {}


def load_cell_types(path=None) -> pd.DataFrame:
    """Canonical cell types: columns cell_type, lineage, synonyms (list)."""
    p = Path(path) if path else CELL_TYPES_FILE
    key = str(p)
    if key in _cell_type_cache:
        return _cell_type_cache[key]
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    df['synonyms'] = [[s.strip() for s in str(v).split(';') if s.strip()] for v in df['synonyms']]
    df['lineage'] = df['lineage'].fillna('').str.strip()
    _cell_type_cache[key] = df
    return df


def _cell_type_lookup(cell_types: pd.DataFrame) -> dict[str, list[tuple[str, str]]]:
    """{normalised name: [(cell_type, lineage), …]} over names and synonyms."""
    out: dict[str, list[tuple[str, str]]] = {}
    for _i, r in cell_types.iterrows():
        for name in [r['cell_type']] + list(r['synonyms']):
            out.setdefault(_norm(name), []).append((r['cell_type'], r['lineage']))
    return out


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

def load_library(path=None, species: str = 'human',
                 cell_types: pd.DataFrame | None = None) -> tuple[pd.DataFrame, str, list[str]]:
    """Read and validate an immune state library.

    Rows are rejected (and listed in the messages) when a required field is
    missing, the feature kind or direction is not recognised, the cell type
    is not a canonical cell type (or a synonym of one), a Marker row's marker
    is not in the marker database, an Abund row names a marker, or the row
    repeats an earlier (state, cell type, kind, marker). Rows for another
    species are left out. Unknown columns are ignored.

    Returns (rows with added 'cell_type_canonical' and 'marker_canonical'
    columns, SHA-256 of the file, messages). Raises ValueError when a
    required column is missing or no row is usable.
    """
    p = Path(path) if path else LIBRARY_FILE
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"The library {p.name} lacks column(s): {', '.join(missing)}")
    messages: list[str] = []
    extra = [c for c in df.columns if c not in LIBRARY_COLUMNS]
    if extra:
        messages.append(f"Ignored column(s): {', '.join(extra)}")
        log.info("FSEA library %s: ignored columns %s", p.name, extra)
    for c in LIBRARY_COLUMNS:
        if c not in df.columns:
            df[c] = ''
    df = df[list(LIBRARY_COLUMNS)].copy()
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    ct = cell_types if cell_types is not None else load_cell_types()
    lookup = _cell_type_lookup(ct)
    keep, canon_ct, canon_mk = [], [], []
    seen = set()
    n_species = 0
    for i, r in df.iterrows():
        line = i + 2
        sp = r['species'].lower()
        if sp and sp not in (species, 'both'):
            n_species += 1
            continue
        problem = None
        if not r['immune_state']:
            problem = "no immune_state"
        elif r['feature_kind'] not in FEATURE_KINDS:
            problem = f"feature_kind {r['feature_kind']!r} is not one of {', '.join(FEATURE_KINDS)}"
        elif r['direction'] not in DIRECTIONS:
            problem = f"direction {r['direction']!r} is not one of {', '.join(DIRECTIONS)}"
        cell = None
        if problem is None:
            hits = lookup.get(_norm(r['cell_type'])) or []
            cell = hits[0][0] if hits else None
            if cell is None:
                problem = f"cell type {r['cell_type']!r} is not a known cell type"
        marker = ''
        if problem is None:
            if r['feature_kind'] == 'Marker':
                marker = canonical_marker(r['marker']) or ''
                if not marker:
                    problem = f"marker {r['marker']!r} is not in the marker database"
            elif r['marker']:
                problem = "an Abund row names a marker"
        if problem is None:
            key = (r['immune_state'], cell, r['feature_kind'], marker)
            if key in seen:
                problem = "repeats an earlier row"
            seen.add(key)
        if problem is not None:
            messages.append(f"Row {line} ({r['immune_state'] or '?'}): {problem}; left out.")
            continue
        if r['weight'] not in WEIGHTS:
            df.at[i, 'weight'] = 'Medium'
        keep.append(i)
        canon_ct.append(cell)
        canon_mk.append(marker)
    if n_species:
        messages.append(f"{n_species} row(s) for another species left out.")
    out = df.loc[keep].reset_index(drop=True)
    out['cell_type_canonical'] = canon_ct
    out['marker_canonical'] = canon_mk
    if out.empty:
        raise ValueError(f"The library {p.name} has no usable rows for {species}.")
    return out, sha, messages


# ---------------------------------------------------------------------------
# Populations
# ---------------------------------------------------------------------------

def population_info(gate_defs: list[dict]) -> dict[str, dict]:
    """{population key: record} for every population, in hierarchy order.

    Record: key, gate, population, label, parent_key ('root' at the top),
    depth, kind ('positivity' | 'phenotype'), and for positivity
    populations marker (canonical name, or the label's marker text when the
    database does not know it), sign and channel (the gate's separating
    channel).
    """
    out: dict[str, dict] = {}
    for g in ag_core.ordered_gate_defs(gate_defs):
        name = g['gate_name']
        parent = (ag_core.population_key(g['parent_gate'], g.get('parent_popul'))
                  if g.get('parent_gate') else 'root')
        depth = (out[parent]['depth'] + 1) if parent in out else 0
        pops = g.get('populations') or {}
        for pop, info in pops.items():
            label = (info or {}).get('label') or pop
            rec = {'key': ag_core.population_key(name, pop), 'gate': name, 'population': pop,
                   'label': label, 'parent_key': parent, 'depth': depth, 'kind': 'phenotype',
                   'marker': '', 'sign': '', 'channel': ''}
            m = POSITIVITY_LABEL.match(str(label).strip())
            if g.get('gate_type') == '1dsep' and len(pops) == 1 and m:
                stem = m.group('marker')
                rec.update(kind='positivity', marker=canonical_marker(stem) or stem,
                           sign=m.group('sign'), channel=g.get('gate_marker_x') or '')
            out[rec['key']] = rec
    return out


def _ancestors(pop_info: dict, key: str) -> list[str]:
    """Ancestor keys of *key*, nearest first (without 'root')."""
    out = []
    cur = (pop_info.get(key) or {}).get('parent_key')
    while cur and cur != 'root' and cur in pop_info and cur not in out:
        out.append(cur)
        cur = pop_info[cur]['parent_key']
    return out


def map_populations(pop_info: dict, cell_types: pd.DataFrame | None = None,
                    overrides: dict | None = None) -> dict[str, dict]:
    """Phenotype populations to canonical cell types.

    Returns {key: {'cell_type', 'lineage', 'method'}} for every phenotype
    population; method is 'override', 'name', 'name (lineage differs)' when
    the only name match has another lineage than the ancestors (left
    unmatched), or 'none'. An override of '' unmatches a population.
    """
    ct = cell_types if cell_types is not None else load_cell_types()
    lookup = _cell_type_lookup(ct)
    lineage_of = dict(zip(ct['cell_type'], ct['lineage']))
    overrides = overrides or {}
    out: dict[str, dict] = {}
    for key, rec in pop_info.items():
        if rec['kind'] != 'phenotype':
            continue
        inherited = ''
        for anc in _ancestors(pop_info, key):
            if out.get(anc, {}).get('lineage'):
                inherited = out[anc]['lineage']
                break
        if key in overrides:
            cell = overrides[key] or None
            out[key] = {'cell_type': cell, 'lineage': lineage_of.get(cell, '') or inherited,
                        'method': 'override'}
            continue
        hits = lookup.get(_norm(rec['label'])) or lookup.get(_norm(rec['population'])) or []
        ok = [h for h in hits if not inherited or not h[1] or h[1] == inherited]
        if ok:
            cell, lin = ok[0]
            out[key] = {'cell_type': cell, 'lineage': lin or inherited, 'method': 'name'}
        else:
            out[key] = {'cell_type': None, 'lineage': inherited,
                        'method': 'name (lineage differs)' if hits else 'none'}
    return out


def resolve_channel_markers(pop_info: dict, pop_keys, channels,
                            channel_antigens: dict | None = None) -> dict[tuple, tuple]:
    """The marker each channel measures in each population.

    For population P and channel c, in order: a positivity gate on c drawn
    on P; on the nearest ancestor of P; on P's descendants when they all
    name the same marker; else the channel's antigen, when no positivity
    gate anywhere names another marker on c. Otherwise unresolved.

    Returns {(pop_key, channel): (marker or '', source)} with source
    'gate', 'gate (ancestor)', 'gate (descendant)', 'channel' or
    'ambiguous'.
    """
    channel_antigens = channel_antigens or {}
    evidence: dict[tuple, Counter] = {}
    names_by_channel: dict[str, set] = {}
    for rec in pop_info.values():
        if rec['kind'] == 'positivity' and rec['channel']:
            evidence.setdefault((rec['parent_key'], rec['channel']), Counter())[rec['marker']] += 1
            names_by_channel.setdefault(rec['channel'], set()).add(rec['marker'])
    children: dict[str, list[str]] = {}
    for k, rec in pop_info.items():
        children.setdefault(rec['parent_key'], []).append(k)

    def _descendants(key):
        stack, out = list(children.get(key, [])), []
        while stack:
            k = stack.pop()
            out.append(k)
            stack.extend(children.get(k, []))
        return out

    antigen_canon: dict[str, str] = {}
    out: dict[tuple, tuple] = {}
    for key in pop_keys:
        desc = None
        for ch in channels:
            ev = evidence.get((key, ch))
            if ev:
                out[(key, ch)] = (ev.most_common(1)[0][0], 'gate')
                continue
            hit = None
            for anc in _ancestors(pop_info, key):
                ev = evidence.get((anc, ch))
                if ev:
                    hit = (ev.most_common(1)[0][0], 'gate (ancestor)')
                    break
            if hit is None:
                if desc is None:
                    desc = _descendants(key)
                names = {m for d in desc for m in (evidence.get((d, ch)) or {})}
                if len(names) == 1:
                    hit = (next(iter(names)), 'gate (descendant)')
            if hit is None:
                if ch not in antigen_canon:
                    ag = channel_antigens.get(ch, '')
                    antigen_canon[ch] = (canonical_marker(ag) or '') if ag else ''
                ag = antigen_canon[ch]
                others = names_by_channel.get(ch, set()) - {ag}
                hit = (ag, 'channel') if ag and not others else ('', 'ambiguous')
            out[(key, ch)] = hit
    return out


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

MATCH_COLUMNS = ('immune_state', 'cell_type', 'feature_kind', 'marker', 'direction', 'weight',
                 'citation', 'population', 'feature', 'readout', 'sign', 'status', 'detail')


def match_library(library: pd.DataFrame, features: pd.DataFrame, pop_map: dict,
                  marker_readout: str = 'positivity') -> pd.DataFrame:
    """One row per library row: the feature it matched, or why none did.

    features : feature metadata from ``ag_fsea.build_features`` (columns
               feature, type, pop_key, parent_key, marker, depth, kept,
               orientation).
    marker_readout : 'positivity' (fraction positive, the default) or
               'median'; the other readout is used when the preferred one
               is not available.

    sign is +1 (UP) or −1 (DOWN), times −1 when the matched feature was
    collapsed onto its mirror. status: 'matched', 'no population' (no
    population is mapped to the cell type), 'no feature' (no tested
    feature for it).
    """
    if marker_readout not in READOUTS:
        raise ValueError(f"marker_readout must be one of {READOUTS}")
    pops_by_cell: dict[str, list[str]] = {}
    for key, m in pop_map.items():
        if m.get('cell_type'):
            pops_by_cell.setdefault(m['cell_type'], []).append(key)
    feats = features.set_index('feature', drop=False)
    if 'kept' in feats.columns:
        feats = feats[feats['kept'].astype(str) != '']
    by_type = {t: feats[feats['type'] == t] for t in ('positivity', 'abundance', 'marker')}

    def _deepest(cands: pd.DataFrame, pop_col: str) -> tuple[pd.Series, str]:
        cands = cands.sort_values('depth', ascending=False, kind='stable')
        others = [str(x) for x in cands[pop_col].iloc[1:]]
        return cands.iloc[0], (f"also: {', '.join(others)}" if others else '')

    rows = []
    for _i, r in library.iterrows():
        cell, kind, marker = r['cell_type_canonical'], r['feature_kind'], r['marker_canonical']
        rec = {'immune_state': r['immune_state'], 'cell_type': cell, 'feature_kind': kind,
               'marker': marker, 'direction': r['direction'], 'weight': r['weight'],
               'citation': r['citation'], 'population': '', 'feature': '', 'readout': '',
               'sign': 0, 'status': '', 'detail': ''}
        keys = pops_by_cell.get(cell) or []
        if not keys:
            rec['status'] = 'no population'
            rows.append(rec)
            continue
        hit, readout, detail = None, '', ''
        if kind == 'Abund':
            c = by_type['abundance']
            c = c[c['pop_key'].isin(keys)]
            if len(c):
                hit, detail = _deepest(c, 'pop_key')
                readout = 'abundance'
        else:
            pos = by_type['positivity']
            pos = pos[pos['parent_key'].isin(keys) & (pos['marker'] == marker)]
            med = by_type['marker']
            med = med[med['pop_key'].isin(keys) & (med['marker'] == marker)]
            order = [('positivity', pos, 'parent_key'), ('median', med, 'pop_key')]
            if marker_readout == 'median':
                order.reverse()
            for name, c, col in order:
                if len(c):
                    hit, detail = _deepest(c, col)
                    readout = name
                    break
        if hit is None:
            rec['status'] = 'no feature'
            rec['detail'] = f"population(s): {', '.join(keys)}"
            rows.append(rec)
            continue
        kept = hit['kept'] or hit['feature']
        orient = int(hit['orientation']) if hit['orientation'] else 1
        rec.update(population=hit['parent_key'] if readout == 'positivity' else hit['pop_key'],
                   feature=kept, readout=readout,
                   sign=(1 if r['direction'] == 'UP' else -1) * orient,
                   status='matched', detail=detail)
        rows.append(rec)
    return pd.DataFrame(rows, columns=list(MATCH_COLUMNS))


def build_sets(matches: pd.DataFrame, library: pd.DataFrame, min_set_size: int) -> list[dict]:
    """The sets to test, one per immune state, in library order.

    A feature matched by several rows of one state counts once; when those
    rows ask for opposite signs it is left out of the state. A state with
    fewer than *min_set_size* distinct features is listed as not testable.

    Returns [{'name', 'description', 'n_library', 'n_matched', 'features',
    'signs', 'testable', 'note'}].
    """
    desc = dict(zip(library['immune_state'], library['description']))
    out = []
    for name in dict.fromkeys(library['immune_state']):
        rows = matches[matches['immune_state'] == name]
        hit = rows[rows['status'] == 'matched']
        signs: dict[str, int] = {}
        conflict = set()
        for f, s in zip(hit['feature'], hit['sign']):
            if f in signs and signs[f] != s:
                conflict.add(f)
            signs.setdefault(f, int(s))
        for f in conflict:
            signs.pop(f, None)
        n = len(signs)
        note = ''
        if conflict:
            note = f"{len(conflict)} feature(s) matched with opposite directions left out"
        testable = n >= min_set_size
        if not testable:
            note = (note + '; ' if note else '') + f"not testable: {n} matched / {min_set_size}"
        out.append({'name': name, 'description': desc.get(name, ''), 'n_library': int(len(rows)),
                    'n_matched': n, 'features': list(signs), 'signs': list(signs.values()),
                    'testable': testable, 'note': note})
    return out
