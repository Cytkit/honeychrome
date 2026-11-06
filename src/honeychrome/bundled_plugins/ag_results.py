"""
ag_results.py — channel alignment, per-sample summaries and feature tables
==========================================================================
Pure computation for the Automated Gating plugin (no Qt, no controller):

* ``align_channels`` — match a model's gated channels to an experiment's
  channels, antigen to antigen.
* ``summarise_sample`` — the compact per-sample record kept after gating:
  population counts and parent counts, per-population marker medians and
  quartiles (a ``MarkerSummary`` array), per-gate status and the
  boundaries applied; ``save_marker_summaries`` / ``load_marker_summaries``
  keep the marker summaries of a run in one .npz file.
* ``population_tables`` / ``anchor_tables`` / ``marker_table`` — samples ×
  features tables for the differential tests in
  ``controller_components.differential_stats``: counts with their parent
  counts, counts with the anchor population's counts, marker medians.
* ``sample_flags`` / ``population_export`` / ``marker_export`` — QC flags
  and long-format export tables; ``write_marker_export`` writes the marker
  table to CSV one sample at a time.
* ``read_sample_table`` / ``sample_table_frame`` — import and export of
  group assignments and covariates as a table with one row per sample.
* ``sample_pca`` / ``predictive_matrix`` — sample-level PCA and the feature
  matrix for the predictive analysis (``drc_predictive``), both from the
  per-sample values of the last statistics run;
  ``sample_analysis_to_jsonable`` / ``sample_analysis_from_jsonable`` save
  and restore their results.
* ``pack_display`` / ``display_population_mask`` / ``display_parent_mask`` /
  ``save_display`` / ``load_display`` — the display subsample kept per
  gated sample for plotting: its events and every population's membership
  as packed bits, small enough to keep for every sample and to save.
* ``statistics_tables`` — the statistics run as tables for CSV export;
  ``counts_only_hits`` — counts hits the frequency test does not share,
  when both asked the same question.
* ``group_population_values`` / ``population_result`` — one population's
  per-sample values by group and its test result, for the population
  comparison plots.

The file name does not end in ``_tab.py``, so the plugin loader does not
treat it as a tab.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Mapping

import numpy as np
import pandas as pd

import ag_core

ALIGN_EXACT = 'exact'        # same channel, same antigen (or both unlabelled)
ALIGN_ANTIGEN = 'antigen'    # the model's antigen is on a different channel here
ALIGN_CHANNEL = 'channel'    # same channel name, but the antigen differs
ALIGN_FUZZY = 'fuzzy'        # closest name only; needs checking
ALIGN_AMBIGUOUS = 'ambiguous'  # the antigen is on more than one channel here
CONFIDENT_ALIGNMENTS = frozenset({ALIGN_EXACT, ALIGN_ANTIGEN})

FEATURE_SEP = ' | '
DEFAULT_MIN_MARKER_EVENTS = 20
# A population is tested only when it can be measured (its parent has
# events) in at least this many samples.
MIN_MEASURED_SAMPLES = 3
# A population is also tested only when it has events in at least this many
# samples; below that the comparison reflects the parent counts, not the
# population.
MIN_NONZERO_SAMPLES = 3
# A frequency is measured in a sample only when its parent has at least
# this many events; a fraction of a handful of events mostly reflects
# gating noise.
DEFAULT_MIN_PARENT_EVENTS = 20
# What the counts test relates a population's events to.
COUNTS_REFERENCES = ('anchor', 'parent')
# How the counts test models them: the moderated linear model on log odds
# (the frequency test's model) or a negative-binomial GLM tested by the
# quasi-likelihood F test. The Wald test is not offered: it was far too
# liberal on human and on mouse data.
COUNTS_METHODS = ('moderated', 'ql')

__all__ = [
    'ALIGN_EXACT', 'ALIGN_ANTIGEN', 'ALIGN_CHANNEL', 'ALIGN_FUZZY', 'ALIGN_AMBIGUOUS',
    'CONFIDENT_ALIGNMENTS', 'FEATURE_SEP', 'DEFAULT_MIN_MARKER_EVENTS', 'MIN_MEASURED_SAMPLES',
    'MIN_NONZERO_SAMPLES', 'DEFAULT_MIN_PARENT_EVENTS', 'COUNTS_REFERENCES', 'COUNTS_METHODS',
    'align_channels', 'alignment_problems', 'apply_alignment',
    'summarise_sample', 'MarkerSummary', 'as_marker_summary', 'select_markers',
    'save_marker_summaries', 'load_marker_summaries', 'MARKER_FILE_FORMAT', 'population_tables', 'anchor_tables', 'marker_table', 'marker_feature',
    'sample_flags', 'population_export', 'marker_export', 'MARKER_EXPORT_COLUMNS',
    'write_marker_export', 'MIN_SAMPLES_PER_GROUP', 'qualifying_groups', 'run_statistics', 'reapply_thresholds',
    'read_sample_table', 'sample_table_frame', 'sample_pca', 'predictive_matrix',
    'stats_to_jsonable', 'stats_from_jsonable',
    'pack_display', 'display_population_mask', 'display_parent_mask', 'save_display',
    'load_display', 'display_file_name', 'statistics_tables', 'group_population_values',
    'population_result', 'counts_only_hits',
]


# ---------------------------------------------------------------------------
# Channel alignment
# ---------------------------------------------------------------------------

def _normalise(name: str) -> str:
    return re.sub(r'[\s_\-.]+', '', str(name or '')).lower()


def _labelled(channel: str, antigen: str | None) -> bool:
    antigen = (antigen or '').strip()
    return bool(antigen) and antigen != channel


def align_channels(model_channels: list[str], model_antigens: dict,
                   exp_pnn: list[str], exp_pns: list[str],
                   canonical=None) -> dict[str, dict]:
    """Match each of *model_channels* to an experiment channel.

    model_antigens : {model channel: antigen} (the model's ``channel_map``).
    exp_pnn/exp_pns: the experiment's channel names and antigen labels.
    canonical      : optional callable, antigen name → canonical marker name
                     or None (e.g. label_matching.match_marker), so synonyms
                     such as CD197 and CCR7 match.

    Labelled channels are matched on antigen first: the same channel
    carrying the same antigen is ``exact``; the antigen on another channel
    is ``antigen`` (a panel rearrangement). Failing that, the same channel
    name is ``channel`` (the antigen differs, which needs checking) and the
    closest antigen name is ``fuzzy``. Unlabelled channels (scatter, time)
    match on channel name.

    Returns {model channel: {'channel': experiment channel or None,
    'method': one of the ALIGN_* values or None, 'antigen': model antigen,
    'exp_antigen': antigen of the matched channel}}.
    """
    pns = list(exp_pns or []) + [''] * max(0, len(exp_pnn) - len(exp_pns or []))
    exp_antigen = {c: (a or '').strip() for c, a in zip(exp_pnn, pns)}

    def key(antigen: str) -> str:
        if not antigen:
            return ''
        canon = None
        if canonical is not None:
            try:
                canon = canonical(antigen)
            except Exception:
                canon = None
        return _normalise(canon or antigen)

    by_antigen: dict[str, list[str]] = {}
    for ch in exp_pnn:
        if _labelled(ch, exp_antigen[ch]):
            by_antigen.setdefault(key(exp_antigen[ch]), []).append(ch)
    labelled_keys = list(by_antigen)

    out: dict[str, dict] = {}
    for mc in model_channels:
        ma = (model_antigens or {}).get(mc) or ''
        match, method = None, None
        if _labelled(mc, ma):
            candidates = by_antigen.get(key(ma), [])
            if mc in candidates:
                match, method = mc, ALIGN_EXACT
            elif len(candidates) == 1:
                match, method = candidates[0], ALIGN_ANTIGEN
            elif len(candidates) > 1:
                match, method = candidates[0], ALIGN_AMBIGUOUS
            elif mc in exp_antigen:
                match, method = mc, ALIGN_CHANNEL
            else:
                close = difflib.get_close_matches(key(ma), labelled_keys, n=1, cutoff=0.8)
                if close:
                    match, method = by_antigen[close[0]][0], ALIGN_FUZZY
        else:
            if mc in exp_antigen:
                match, method = mc, ALIGN_EXACT
            else:
                close = difflib.get_close_matches(mc, list(exp_pnn), n=1, cutoff=0.8)
                if close:
                    match, method = close[0], ALIGN_FUZZY
        out[mc] = {'channel': match, 'method': method, 'antigen': ma,
                   'exp_antigen': exp_antigen.get(match, '') if match else ''}
    return out


def alignment_problems(alignment: dict[str, str | None]) -> list[str]:
    """Reasons an alignment {model channel: experiment channel} cannot be
    used: unassigned channels, or one experiment channel used twice."""
    problems = []
    missing = [mc for mc, ec in alignment.items() if not ec]
    if missing:
        problems.append("Not assigned: " + ", ".join(missing))
    used: dict[str, list[str]] = {}
    for mc, ec in alignment.items():
        if ec:
            used.setdefault(ec, []).append(mc)
    for ec, mcs in used.items():
        if len(mcs) > 1:
            problems.append(f"{ec} is assigned to more than one model channel: "
                            + ", ".join(mcs))
    return problems


def gated_channels(gate_defs: list[dict]) -> list[str]:
    """Channels used as an axis by any gate (including 1dsep view channels
    and imported shapes), in hierarchy order."""
    by_name = ag_core.gates_by_name(gate_defs)
    out: list[str] = []
    for g in ag_core.ordered_gate_defs(gate_defs):
        for ch in ag_core.referenced_channels(ag_core.effective_gate_def(g, by_name)):
            if ch and ch not in out:
                out.append(ch)
    return out


def apply_alignment(model: dict, alignment: dict[str, str]) -> dict:
    """Copy of a normalised model with its channels renamed to the
    experiment's (gate axes, imported shapes, channel map and transform
    keys)."""
    out = dict(model)
    out['gate_definitions'] = ag_core.rename_channels(model.get('gate_definitions') or [], alignment)
    out['trained_boundaries'] = ag_core.rename_boundary_channels(
        model.get('trained_boundaries') or {}, alignment)
    out['channel_map'] = ag_core.rekey(model.get('channel_map'), alignment)
    out['transforms'] = ag_core.rekey(model.get('transforms'), alignment)
    return out


# ---------------------------------------------------------------------------
# Per-sample summary
# ---------------------------------------------------------------------------

def _compact_values(values: np.ndarray) -> np.ndarray:
    """*values* as float32 when every value (NaN included) is exactly the
    same in float32, else as float64: half the memory, never a changed
    number."""
    values = np.asarray(values, dtype=np.float64)
    v32 = values.astype(np.float32)
    if np.array_equal(v32.astype(np.float64), values, equal_nan=True):
        return v32
    return values


# Name tuples shared between summaries with the same populations or
# channels (every sample of a run has the same ones), with their
# {name: position} lookup; cleared when it grows past the limit.
_SHARED_NAMES: dict[tuple, tuple] = {}
_SHARED_NAMES_LIMIT = 64


def _shared_names(names) -> tuple[tuple, dict]:
    names = tuple(str(n) for n in names)
    cached = _SHARED_NAMES.get(names)
    if cached is not None:
        return cached
    index = {n: i for i, n in enumerate(names)}
    if len(index) != len(names):
        raise ValueError("Marker summary names must be unique.")
    if len(_SHARED_NAMES) >= _SHARED_NAMES_LIMIT:
        _SHARED_NAMES.clear()
    _SHARED_NAMES[names] = (names, index)
    return names, index


class _MarkerRow(Mapping):
    """One population of a MarkerSummary: {channel: [median, q25, q75]}."""

    __slots__ = ('_owner', '_row')

    def __init__(self, owner: 'MarkerSummary', row: int):
        self._owner = owner
        self._row = row

    def _column(self, channel) -> int:
        owner = self._owner
        j = owner._ch_index[channel]
        if owner.present is not None and not owner.present[self._row, j]:
            raise KeyError(channel)
        return j

    def __getitem__(self, channel) -> list[float]:
        return self._owner.values[self._row, self._column(channel)].tolist()

    def __contains__(self, channel) -> bool:
        try:
            self._column(channel)
        except KeyError:
            return False
        return True

    def __iter__(self):
        owner = self._owner
        if owner.present is None:
            return iter(owner.channels)
        mask = owner.present[self._row]
        return (ch for ch, ok in zip(owner.channels, mask) if ok)

    def __len__(self) -> int:
        owner = self._owner
        if owner.present is None:
            return len(owner.channels)
        return int(owner.present[self._row].sum())

    def __repr__(self) -> str:
        return repr(dict(self.items()))


class MarkerSummary(Mapping):
    """One sample's marker medians and quartiles in every population, in
    transformed units.

    It reads like the nested dict ``{population key: {channel: [median,
    q25, q75]}}``, so ``summary['CD4 T/pos']['CD25'][0]`` is the median of
    CD25 in CD4 T cells, but the numbers are held as one array of
    populations × channels × 3 instead of one Python list per population and
    channel. For a large panel (thousands of populations, tens of channels,
    150 samples) that replaces tens of millions of small Python objects with
    a few hundred arrays: less memory, faster saving and loading, and
    nothing for Python's garbage collector to walk.

    populations : population keys, in order.
    channels    : channel names, in order.
    values      : array (populations × channels × 3) of [median, q25, q75],
                  float32 or float64.
    present     : optional bool array (populations × channels), False where
                  a population has no value for a channel; None when every
                  population has every channel.

    The summary is read-only, so copies share it.
    """

    __slots__ = ('populations', 'channels', 'values', 'present', '_pop_index', '_ch_index')

    def __init__(self, populations, channels, values, present=None):
        self.populations, self._pop_index = _shared_names(populations)
        self.channels, self._ch_index = _shared_names(channels)
        values = np.asarray(values)
        if values.dtype not in (np.float32, np.float64):
            values = values.astype(np.float64)
        shape = (len(self.populations), len(self.channels), 3)
        if values.size == 0:
            values = np.zeros(shape, dtype=values.dtype)
        if values.shape != shape:
            raise ValueError(f"Marker values have shape {values.shape}, expected {shape}.")
        values = values.view()
        values.flags.writeable = False
        if present is not None:
            present = np.asarray(present, dtype=bool)
            if present.shape != shape[:2]:
                raise ValueError(f"Marker presence has shape {present.shape}, "
                                 f"expected {shape[:2]}.")
            if present.all():
                present = None
            else:
                present = present.view()
                present.flags.writeable = False
        self.values = values
        self.present = present

    @classmethod
    def empty(cls) -> 'MarkerSummary':
        return cls((), (), np.zeros((0, 0, 3), dtype=np.float32))

    @classmethod
    def from_dict(cls, markers: dict | None) -> 'MarkerSummary':
        """The summary of a nested {population: {channel: [median, q25, q75]}}
        dict (missing values, saved as None, become NaN)."""
        markers = markers or {}
        populations = list(markers)
        rows = [markers[p] or {} for p in populations]
        first = list(rows[0]) if rows else []
        if all(list(r) == first for r in rows):
            values = np.array([[r[ch] for ch in first] for r in rows], dtype=np.float64)
            return cls(populations, first, _compact_values(values.reshape(len(rows), len(first), 3)))
        channels = list(dict.fromkeys(ch for r in rows for ch in r))
        col = {ch: j for j, ch in enumerate(channels)}
        values = np.full((len(rows), len(channels), 3), np.nan)
        present = np.zeros((len(rows), len(channels)), dtype=bool)
        for i, r in enumerate(rows):
            for ch, v in r.items():
                values[i, col[ch]] = np.array(v, dtype=np.float64)
                present[i, col[ch]] = True
        return cls(populations, channels, _compact_values(values), present)

    def median(self, population, channel) -> float:
        """The median of *channel* in *population* (KeyError when absent)."""
        i = self._pop_index[population]
        return float(self.values[i, _MarkerRow(self, i)._column(channel), 0])

    def subset(self, populations) -> 'MarkerSummary':
        """The summary of the listed populations that it has, in its order."""
        wanted = set(populations)
        rows = [i for i, p in enumerate(self.populations) if p in wanted]
        present = None if self.present is None else self.present[rows]
        return MarkerSummary([self.populations[i] for i in rows], self.channels,
                             self.values[rows], present)

    def __getitem__(self, population) -> _MarkerRow:
        return _MarkerRow(self, self._pop_index[population])

    def __contains__(self, population) -> bool:
        return population in self._pop_index

    def __iter__(self):
        return iter(self.populations)

    def __len__(self) -> int:
        return len(self.populations)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __repr__(self) -> str:
        return (f"MarkerSummary({len(self.populations)} populations × "
                f"{len(self.channels)} channels)")


def as_marker_summary(markers) -> MarkerSummary:
    """*markers* (a MarkerSummary, a nested dict or None) as a MarkerSummary."""
    if isinstance(markers, MarkerSummary):
        return markers
    return MarkerSummary.from_dict(markers)


def select_markers(markers, populations):
    """The marker summary of only the listed populations, of the same kind
    as *markers* (a MarkerSummary or a nested dict)."""
    if isinstance(markers, MarkerSummary):
        return markers.subset(populations)
    wanted = set(populations)
    return {k: v for k, v in (markers or {}).items() if k in wanted}


def _population_median(markers, population, channel) -> float:
    if isinstance(markers, MarkerSummary):
        return markers.median(population, channel)
    return float(markers[population][channel][0])


MARKER_FILE_FORMAT = 1


def save_marker_summaries(path, summaries: dict, fingerprint: str = '') -> None:
    """Write every sample's marker summary (``summaries[sample]['markers']``)
    to one .npz file, beside *path* and then moved into place.

    The file holds only numeric and text arrays plus a JSON header, no
    pickled Python objects, so it is read with ``allow_pickle=False``.
    Population and channel names are stored once; each sample stores
    positions into them and its values array.
    """
    import json
    import os
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pop_pos: dict[str, int] = {}
    ch_pos: dict[str, int] = {}
    samples = list(summaries)
    arrays: dict[str, np.ndarray] = {}
    for i, key in enumerate(samples):
        m = as_marker_summary((summaries[key] or {}).get('markers'))
        arrays[f'populations_{i}'] = np.array(
            [pop_pos.setdefault(p, len(pop_pos)) for p in m.populations], dtype=np.int32)
        arrays[f'channels_{i}'] = np.array(
            [ch_pos.setdefault(c, len(ch_pos)) for c in m.channels], dtype=np.int32)
        arrays[f'values_{i}'] = np.asarray(m.values)
        if m.present is not None:
            arrays[f'present_{i}'] = np.asarray(m.present)
    arrays['population_names'] = np.array(list(pop_pos), dtype=str)
    arrays['channel_names'] = np.array(list(ch_pos), dtype=str)
    arrays['meta'] = np.array(json.dumps({'format': MARKER_FILE_FORMAT,
                                          'fingerprint': str(fingerprint or ''),
                                          'samples': [str(s) for s in samples]}))
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def load_marker_summaries(path) -> tuple[str, dict]:
    """(fingerprint, {sample: MarkerSummary}) from save_marker_summaries.
    Raises ValueError for a file of another format."""
    import json
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['meta']))
        if meta.get('format') != MARKER_FILE_FORMAT:
            raise ValueError(f"Unsupported marker summary format {meta.get('format')!r}.")
        pop_names = z['population_names'].tolist()
        ch_names = z['channel_names'].tolist()
        files = set(z.files)
        out = {}
        for i, key in enumerate(meta.get('samples') or []):
            present = z[f'present_{i}'] if f'present_{i}' in files else None
            out[key] = MarkerSummary([pop_names[k] for k in z[f'populations_{i}'].tolist()],
                                     [ch_names[k] for k in z[f'channels_{i}'].tolist()],
                                     z[f'values_{i}'], present)
    return str(meta.get('fingerprint') or ''), out


def summarise_sample(gate_defs: list[dict], result: ag_core.GatingResult, data_t: np.ndarray,
                     channel_index: dict, marker_channels: list[str], n_total: int) -> dict:
    """Compact record of one sample's gating.

    data_t          : the sample's transformed events (rows as gated).
    marker_channels : channels summarised within every population.

    Returns {'populations': population_table rows, 'markers': MarkerSummary
    (reads as {population key: {channel: [median, q25, q75]}}), 'status':
    per-gate status, 'boundaries': the boundaries applied, 'n_total':
    events gated}. Everything but 'markers' is JSON-safe; marker quartiles
    are in transformed units.
    """
    rows = ag_core.population_table(gate_defs, result, n_total)
    channels = [ch for ch in dict.fromkeys(marker_channels) if ch in channel_index]
    col_idx = [channel_index[ch] for ch in channels]
    pop_row: dict[str, int] = {}
    quartiles: list[np.ndarray] = []
    if col_idx:
        for gate_def in gate_defs:
            name = gate_def['gate_name']
            for pop, m in (result.masks.get(name) or {}).items():
                if not m.any():
                    continue
                q = np.percentile(data_t[np.ix_(m, col_idx)], [50, 25, 75], axis=0)
                key = ag_core.population_key(name, pop)
                if key in pop_row:
                    quartiles[pop_row[key]] = q.T
                else:
                    pop_row[key] = len(quartiles)
                    quartiles.append(q.T)
    if quartiles:
        markers = MarkerSummary(list(pop_row), channels, _compact_values(np.stack(quartiles)))
    else:
        markers = MarkerSummary.empty()
    return {
        'populations': ag_core.to_jsonable(rows),
        'markers': markers,
        'status': ag_core.to_jsonable(result.status),
        'boundaries': ag_core.serialisable_boundaries(result.boundaries),
        'n_total': int(n_total),
    }


# ---------------------------------------------------------------------------
# Feature tables for statistics
# ---------------------------------------------------------------------------

def _name_list(names: list[str], limit: int = 8) -> str:
    """The first *limit* of *names*, comma-separated, with a count of the rest."""
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown} and {len(names) - limit} more"


def population_tables(summaries: dict, samples: list[str], use_stats_parent: bool = True,
                      min_parent_events: int = 0
                      ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Population counts and their denominators, samples × populations.

    The denominator is the parent population's count, or the population's
    ``stats_parent`` count when one is defined and *use_stats_parent* is
    set.

    A population with no events in a sample whose parent has events is a
    true zero and is kept as 0. Where the population cannot be measured
    (it is absent from the sample's record, or its parent has no events,
    or fewer than *min_parent_events*) both tables hold NaN, so the
    population is tested on the samples that do have it rather than being
    left out. Populations measurable in fewer than ``MIN_MEASURED_SAMPLES``
    samples, or with events in fewer than ``MIN_NONZERO_SAMPLES`` samples,
    are dropped.

    Returns (counts, denominators, dropped population keys).
    """
    counts: dict[str, dict[str, float]] = {}
    denoms: dict[str, dict[str, float]] = {}
    order: list[str] = []
    for s in samples:
        for row in (summaries.get(s) or {}).get('populations', []):
            key = row['key']
            if key not in order:
                order.append(key)
            use_sp = use_stats_parent and row.get('n_stats_parent') is not None
            counts.setdefault(key, {})[s] = float(row['n_events'])
            denoms.setdefault(key, {})[s] = float(row['n_stats_parent'] if use_sp
                                                 else row['n_parent'])
    counts_df = pd.DataFrame({k: counts[k] for k in order}, index=list(samples), dtype=float)
    denom_df = pd.DataFrame({k: denoms[k] for k in order}, index=list(samples), dtype=float)
    unmeasurable = (counts_df.isna() | denom_df.isna() | (denom_df <= 0)
                    | (denom_df < min_parent_events))
    counts_df = counts_df.mask(unmeasurable)
    denom_df = denom_df.mask(unmeasurable)
    from honeychrome.controller_components import differential_stats as ds
    measured = (~unmeasurable).sum(axis=0)
    sparse = set(ds.sparse_features(counts_df, MIN_NONZERO_SAMPLES))
    dropped = [k for k in order if measured[k] < MIN_MEASURED_SAMPLES or k in sparse]
    keep = [k for k in order if k not in dropped]
    return counts_df[keep], denom_df[keep], dropped


def anchor_tables(summaries: dict, samples: list[str], anchor_key: str | None
                  ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Population counts and the anchor population's counts, samples ×
    populations, for the populations below the anchor.

    *anchor_key* is a population key ('gate/pop'), or None or 'root' for
    all events of the sample (its ``n_total``). A population absent from a
    sample whose anchor has events is a true zero; samples whose anchor has
    no events are NaN throughout. Populations with events in fewer than
    ``MIN_NONZERO_SAMPLES`` samples are dropped.

    Returns (counts, anchor counts, dropped population keys).
    """
    from honeychrome.controller_components import differential_stats as ds

    anchor = anchor_key or 'root'
    events: dict[str, dict[str, float]] = {}
    children: dict[str, set] = {}
    order: list[str] = []
    for s in samples:
        rows = (summaries.get(s) or {}).get('populations', [])
        events[s] = {r['key']: float(r['n_events']) for r in rows}
        for r in rows:
            if r['key'] not in order:
                order.append(r['key'])
            children.setdefault(r.get('parent_key') or 'root', set()).add(r['key'])
    under, stack = set(), list(children.get(anchor, ()))
    while stack:
        k = stack.pop()
        if k not in under:
            under.add(k)
            stack.extend(children.get(k, ()))
    keys = [k for k in order if k in under]
    if anchor == 'root':
        n_anchor = pd.Series({s: float((summaries.get(s) or {}).get('n_total') or 0)
                              for s in samples})
    else:
        n_anchor = pd.Series({s: events[s].get(anchor, 0.0) for s in samples})
    n_anchor = n_anchor.reindex(list(samples)).where(lambda v: v > 0)
    counts_df = pd.DataFrame({k: {s: events[s].get(k, 0.0) for s in samples} for k in keys},
                             index=list(samples), columns=keys, dtype=float)
    counts_df.loc[n_anchor.isna().values] = np.nan
    denom_df = pd.DataFrame(np.repeat(n_anchor.values[:, None], len(keys), axis=1),
                            index=list(samples), columns=keys)
    sparse = set(ds.sparse_features(counts_df, MIN_NONZERO_SAMPLES))
    dropped = [k for k in keys if k in sparse]
    keep = [k for k in keys if k not in sparse]
    return counts_df[keep], denom_df[keep], dropped


def marker_feature(pop_key: str, channel: str) -> str:
    return f"{pop_key}{FEATURE_SEP}{channel}"


def marker_table(summaries: dict, samples: list[str], gate_defs: list[dict],
                 exclude_gating_markers: bool = True,
                 min_events: int = DEFAULT_MIN_MARKER_EVENTS,
                 population_labels: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Per-population marker medians, samples × (population, channel).

    exclude_gating_markers : leave out, for each population, the channels
        used by its own gate and every ancestor gate (its 'type' markers,
        which define the population and so differ by construction).
    min_events             : a population's medians are taken only from
        samples where it has at least this many events; in the others the
        value is NaN (not measured), and the feature is tested on the
        samples that have it. Populations measured in fewer than
        ``MIN_MEASURED_SAMPLES`` samples are dropped.

    Returns (values, feature metadata with 'feature', 'cluster' (display
    label), 'cluster_id' (population key) and 'channel' columns, dropped
    population keys).
    """
    labels = population_labels or {}
    type_markers = {g['gate_name']: set(ag_core.ancestor_channels(gate_defs, g['gate_name']))
                    for g in gate_defs}
    n_events: dict[str, dict[str, int]] = {}
    for s in samples:
        for row in (summaries.get(s) or {}).get('populations', []):
            n_events.setdefault(row['key'], {})[s] = int(row['n_events'])

    values: dict[str, dict[str, float]] = {}
    meta: list[dict] = []
    dropped: list[str] = []
    with_markers = {k for s in samples for k in ((summaries.get(s) or {}).get('markers') or {})}
    pop_keys = [k for k in n_events if k in with_markers]
    for pop_key in pop_keys:
        per_sample = n_events.get(pop_key, {})
        measured = [s for s in samples
                    if pop_key in ((summaries.get(s) or {}).get('markers') or {})
                    and per_sample.get(s, 0) >= min_events]
        if len(measured) < MIN_MEASURED_SAMPLES:
            dropped.append(pop_key)
            continue
        gate_name, _pop = ag_core.split_population_key(pop_key)
        skip = type_markers.get(gate_name, set()) if exclude_gating_markers else set()
        channels = None
        for s in measured:
            chans = set(((summaries[s].get('markers') or {}).get(pop_key) or {}).keys())
            channels = chans if channels is None else channels & chans
        for ch in sorted(channels or [], key=str):
            if ch in skip:
                continue
            feature = marker_feature(pop_key, ch)
            values[feature] = {s: _population_median(summaries[s]['markers'], pop_key, ch)
                               for s in measured}
            meta.append({'feature': feature, 'cluster': labels.get(pop_key, pop_key),
                         'cluster_id': pop_key, 'channel': ch})
    df = pd.DataFrame(values, index=list(samples), dtype=float)
    meta_df = pd.DataFrame(meta, columns=['feature', 'cluster', 'cluster_id', 'channel'])
    return df, meta_df, dropped


# ---------------------------------------------------------------------------
# QC flags and exports
# ---------------------------------------------------------------------------

def sample_flags(summary: dict, load_info: dict | None = None) -> list[str]:
    """Reasons to treat a sample with caution: gates that could not be
    calculated or applied, recalculated gates that drifted, and events
    removed by time QC."""
    flags = []
    for gate, st in (summary.get('status') or {}).items():
        flag = (st or {}).get('flag')
        if flag == 'drift':
            flags.append(f"{gate}: {st.get('message') or 'drifted'}")
        elif flag in ('failed', 'skipped'):
            flags.append(f"{gate}: {st.get('message') or flag}")
    info = load_info or {}
    n_file, n_kept = info.get('n_events_file'), info.get('n_events_kept')
    if n_file and n_kept is not None and n_kept < n_file:
        flags.append(f"time QC removed {100.0 * (n_file - n_kept) / n_file:.1f}% of events")
    return flags


def population_export(summaries: dict, samples: list[str], groups: dict | None = None) -> pd.DataFrame:
    """Long table: one row per sample × population."""
    rows = []
    for s in samples:
        for row in (summaries.get(s) or {}).get('populations', []):
            rec = {'sample': s, 'group': (groups or {}).get(s, '')}
            rec.update(row)
            rows.append(rec)
    return pd.DataFrame(rows)


MARKER_EXPORT_COLUMNS = ('sample', 'group', 'population', 'n_events', 'channel', 'antigen',
                         'median', 'q25', 'q75', 'iqr')


def _marker_columns(summaries: dict, samples: list[str], antigens: dict | None,
                    groups: dict | None) -> dict:
    """{column: list of values} of the marker export rows of *samples*."""
    cols = {c: [] for c in MARKER_EXPORT_COLUMNS}
    antigens = antigens or {}
    for s in samples:
        summ = summaries.get(s) or {}
        n_by_pop = {r['key']: r['n_events'] for r in summ.get('populations', [])}
        group = (groups or {}).get(s, '')
        markers = summ.get('markers') or {}
        if isinstance(markers, MarkerSummary) and markers.present is None:
            n_ch = len(markers.channels)
            cols['population'] += [p for p in markers.populations for _ in range(n_ch)]
            cols['n_events'] += [n_by_pop.get(p) for p in markers.populations for _ in range(n_ch)]
            cols['channel'] += list(markers.channels) * len(markers.populations)
            cols['median'] += markers.values[:, :, 0].ravel().tolist()
            cols['q25'] += markers.values[:, :, 1].ravel().tolist()
            cols['q75'] += markers.values[:, :, 2].ravel().tolist()
        else:
            for pop_key, chans in markers.items():
                n = n_by_pop.get(pop_key)
                for ch, (med, q25, q75) in chans.items():
                    cols['population'].append(pop_key)
                    cols['n_events'].append(n)
                    cols['channel'].append(ch)
                    cols['median'].append(med)
                    cols['q25'].append(q25)
                    cols['q75'].append(q75)
        added = len(cols['population']) - len(cols['sample'])
        cols['sample'] += [s] * added
        cols['group'] += [group] * added
    cols['antigen'] = [antigens.get(ch, ch) for ch in cols['channel']]
    cols['iqr'] = [q75 - q25 for q25, q75 in zip(cols['q25'], cols['q75'])]
    return cols


def marker_export(summaries: dict, samples: list[str], antigens: dict | None = None,
                  groups: dict | None = None) -> pd.DataFrame:
    """Long table: one row per sample × population × channel, with the
    median and quartiles in transformed units."""
    cols = _marker_columns(summaries, samples, antigens, groups)
    if not cols['sample']:
        return pd.DataFrame()
    return pd.DataFrame(cols)


def write_marker_export(path, summaries: dict, samples: list[str],
                        antigens: dict | None = None, groups: dict | None = None) -> int:
    """Write the marker export table to *path* as CSV (no index), one
    sample at a time so the whole table is never held in memory; the file
    is the same as ``marker_export(summaries, samples, antigens,
    groups).to_csv(path, index=False)``. Returns the number of rows
    written."""
    # Counts are written as floats in every chunk when any is missing, as
    # in the whole table.
    missing_counts = False
    for s in samples:
        summ = summaries.get(s) or {}
        n_by_pop = {r['key']: r['n_events'] for r in summ.get('populations', [])}
        if any(n_by_pop.get(k) is None for k in (summ.get('markers') or {})):
            missing_counts = True
            break
    n_rows = 0
    for s in samples:
        cols = _marker_columns(summaries, [s], antigens, groups)
        if not cols['sample']:
            continue
        frame = pd.DataFrame(cols)
        if missing_counts:
            frame['n_events'] = frame['n_events'].astype(float)
        frame.to_csv(path, index=False, mode='w' if n_rows == 0 else 'a',
                     header=n_rows == 0)
        n_rows += len(frame)
    if n_rows == 0:
        pd.DataFrame().to_csv(path, index=False)
    return n_rows


# ---------------------------------------------------------------------------
# Differential statistics
# ---------------------------------------------------------------------------

MIN_SAMPLES_PER_GROUP = 3


def qualifying_groups(samples: list[str], groups: dict, group_order: list[str]) -> list[str]:
    """Groups, in *group_order*, with at least MIN_SAMPLES_PER_GROUP of *samples*."""
    counts: dict[str, int] = {}
    for s in samples:
        g = groups.get(s) or ''
        if g:
            counts[g] = counts.get(g, 0) + 1
    order = list(group_order) + sorted(g for g in counts if g not in group_order)
    return [g for g in order if counts.get(g, 0) >= MIN_SAMPLES_PER_GROUP]


def run_statistics(summaries: dict, samples: list[str], groups: dict, gate_defs: list[dict], *,
                   group_order: list[str] | None = None, contrast_mode: str = 'reference',
                   reference: str = '', pairing: dict | None = None,
                   covariates: pd.DataFrame | None = None,
                   numeric_covariates: list[str] | None = None,
                   pval_threshold: float = 0.05, fc_threshold: float = 0.5,
                   marker_threshold: float = 0.03, use_treat: bool = True,
                   fdr_scope: str = 'global', run_freq: bool = True, run_counts: bool = False,
                   run_markers: bool = True, use_stats_parent: bool = True,
                   exclude_gating_markers: bool = True,
                   min_marker_events: int = DEFAULT_MIN_MARKER_EVENTS,
                   counts_method: str = 'moderated', counts_reference: str = 'anchor',
                   min_parent_events: int = DEFAULT_MIN_PARENT_EVENTS,
                   anchor_key: str | None = None) -> dict:
    """Compare gated populations between sample groups.

    * Frequencies: log2 odds of each population within its parent (or its
      stats parent), moderated linear model. A sample's frequency is
      measured only where the parent has at least *min_parent_events*
      events.
    * Counts: each population's events relative to *counts_reference*.
      'anchor': the anchor population (*anchor_key*; by default
      ``ag_core.anchor_population`` of *gate_defs*, else all events), for
      the populations below it, asking whether a population is over- or
      under-represented for the number of viable cells. 'parent': the
      frequency denominators, the frequency test's question.
      *counts_method* 'moderated' fits the moderated linear model to the
      log2 odds within the reference; 'ql' fits a negative-binomial GLM
      with the log reference count as offset, tested by the
      quasi-likelihood F test (``differential_stats.run_glm_counts``).
    * Marker medians: moderated linear model on per-population medians, one
      family per population (population-first testing).

    groups     : {sample: group name}; samples without a group are left out.
    pairing    : {sample: pairing level} (e.g. donor), or None.
    covariates : adjustment covariates, indexed by sample, or None.
    numeric_covariates : the covariates entered as continuous terms (age,
                 CRP); every other covariate is treated as categories.

    Returns {'samples', 'group_vec', 'comparisons', 'freq', 'freq_values'
    (% of denominator), 'counts', 'counts_values' (events), 'markers',
    'marker_values', 'notes', and with counts 'counts_reference',
    'counts_method' and 'counts_anchor' (population key, 'root' for all
    events, '' against the parent)}; a result is None when not requested.
    Raises
    RuntimeError when fewer than two groups qualify.
    """
    from honeychrome.controller_components import differential_stats as ds

    if counts_method not in COUNTS_METHODS:
        raise ValueError(f"counts_method must be one of {COUNTS_METHODS}, got {counts_method!r}")
    if counts_reference not in COUNTS_REFERENCES:
        raise ValueError(f"counts_reference must be one of {COUNTS_REFERENCES}, "
                         f"got {counts_reference!r}")
    order = list(group_order or [])
    names = qualifying_groups(samples, groups, order)
    if len(names) < 2:
        raise RuntimeError(
            f"At least two groups need {MIN_SAMPLES_PER_GROUP} or more gated samples each.")
    if contrast_mode == 'reference' and reference not in names:
        reference = names[0]
    contrasts = ds.build_contrasts(names, contrast_mode, reference)
    used = [s for s in samples if (groups.get(s) or '') in names]
    group_vec = [groups[s] for s in used]
    pairing_vec = [str((pairing or {}).get(s, '')) for s in used] if pairing else None
    cov = None
    if covariates is not None and len(covariates.columns):
        cov = ds.covariate_frame(covariates, used, list(covariates.columns),
                                 numeric=list(numeric_covariates or []))

    out = {'samples': used, 'group_vec': group_vec, 'comparisons': contrasts,
           'freq': None, 'freq_values': None, 'counts': None, 'counts_values': None,
           'markers': None, 'marker_values': None, 'notes': []}
    common = dict(pairing_vec=pairing_vec, fdr_scope=fdr_scope, covariates=cov,
                  use_treat=use_treat)

    if run_freq or (run_counts and counts_reference == 'parent'):
        counts, denom, dropped = population_tables(summaries, used, use_stats_parent,
                                                   min_parent_events)
        too_few = (f"fewer than {min_parent_events} events" if min_parent_events > 1
                   else "no events")
        if dropped:
            out['notes'].append(
                f"Populations measurable in fewer than {MIN_MEASURED_SAMPLES} samples (their "
                f"parent population had {too_few} in the others), or with events in fewer than "
                f"{MIN_NONZERO_SAMPLES} samples, were left out: "
                + _name_list(dropped))
        partial = [k for k in counts.columns if counts[k].isna().any()]
        if partial:
            out['notes'].append(
                f"{len(partial)} population(s) could not be measured in some samples (their "
                f"parent population had {too_few}) and were tested on the samples that could: "
                + _name_list(partial))
        if run_freq:
            data = ds.logit_proportions(counts, denom)
            out['freq'] = ds.run_moderated(data, group_vec, contrasts, contrast_mode,
                                           pval_threshold, fc_threshold, **common)
            out['freq_values'] = 100.0 * counts / denom
    if run_counts:
        if counts_reference == 'parent':
            c_counts, c_denom, anchor = counts, denom, ''
        else:
            anchor = (anchor_key or (ag_core.anchor_population(gate_defs) if gate_defs else None)
                      or 'root')
            c_counts, c_denom, c_dropped = anchor_tables(summaries, used, anchor)
            anchor_name = ('all events' if anchor == 'root'
                           else ag_core.gate_population_names(gate_defs).get(anchor, anchor))
            out['notes'].append(
                f"Counts are relative to the anchor population ({anchor_name}); the "
                f"{c_counts.shape[1]} population(s) below it were tested.")
            if c_dropped:
                out['notes'].append(
                    f"Populations with events in fewer than {MIN_NONZERO_SAMPLES} samples were "
                    "left out of the counts test: " + _name_list(c_dropped))
        if c_counts.shape[1] < 3:
            out['notes'].append("Fewer than three populations could be tested for counts, so "
                                "the counts test was not run.")
        else:
            if counts_method == 'moderated':
                out['counts'] = ds.run_moderated(ds.logit_proportions(c_counts, c_denom),
                                                 group_vec, contrasts, contrast_mode,
                                                 pval_threshold, fc_threshold, **common)
            else:
                out['counts'] = ds.run_glm_counts(c_counts, group_vec, contrasts, contrast_mode,
                                                  pval_threshold, fc_threshold,
                                                  offset_df=c_denom, method=counts_method,
                                                  **common)
            out['counts_values'] = c_counts
            out['counts_reference'] = counts_reference
            out['counts_method'] = counts_method
            out['counts_anchor'] = anchor
            empty = [k for k in c_counts.columns if not (c_counts[k].fillna(0) > 0).any()]
            if empty:
                out['notes'].append(
                    f"{len(empty)} population(s) have no events in any sample and could not be "
                    "tested for counts: " + ", ".join(empty))

    if run_markers:
        values, meta, dropped = marker_table(
            summaries, used, gate_defs, exclude_gating_markers=exclude_gating_markers,
            min_events=min_marker_events,
            population_labels=ag_core.gate_population_names(gate_defs))
        if dropped:
            out['notes'].append(
                f"Populations with at least {min_marker_events} events in fewer than "
                f"{MIN_MEASURED_SAMPLES} samples were left out of the marker tests: "
                + _name_list(dropped))
        partial = sorted({m['cluster_id'] for _i, m in meta.iterrows()
                          if values[m['feature']].isna().any()})
        if partial:
            out['notes'].append(
                f"{len(partial)} population(s) have fewer than {min_marker_events} events in "
                "some samples; their marker medians were tested on the samples that have "
                "enough: " + _name_list(partial))
        out['markers'] = ds.run_moderated(values, group_vec, contrasts, contrast_mode,
                                          pval_threshold, marker_threshold,
                                          feature_meta=meta, hierarchical=True, **common)
        out['marker_values'] = values
    return out


_STATS_FRAMES = ('freq', 'freq_values', 'counts', 'counts_values', 'markers', 'marker_values')


def _frame_to_jsonable(df: pd.DataFrame) -> dict:
    split = df.to_dict(orient='split')
    return ag_core.to_jsonable({'index': split['index'], 'columns': split['columns'],
                                'data': split['data']})


def _column_from_jsonable(values: list):
    """One saved column: booleans and text as they are; numbers as int64
    or, with missing values (None) or fractions, float64 with NaN."""
    if values and all(isinstance(v, bool) for v in values):
        return values
    numeric = all(v is None or (isinstance(v, (int, float)) and not isinstance(v, bool))
                  for v in values)
    if not numeric:
        return values
    if values and all(isinstance(v, int) for v in values):
        return np.asarray(values, dtype=np.int64)
    return np.asarray(values, dtype=float)


def _frame_from_jsonable(obj: dict) -> pd.DataFrame:
    """The DataFrame _frame_to_jsonable saved, built column by column in one
    step (wide tables, e.g. thousands of marker features, stay fast)."""
    columns = list(obj['columns'])
    rows = obj['data']
    by_column = list(zip(*rows)) if rows else [() for _c in columns]
    data = {i: _column_from_jsonable(list(values)) for i, values in enumerate(by_column)}
    df = pd.DataFrame(data, index=obj['index'])
    df.columns = columns
    return df


def stats_to_jsonable(stats: dict) -> dict:
    """*stats* (a ``run_statistics`` result) as JSON-safe values, for saving.
    Non-finite numbers become null."""
    out = {k: v for k, v in stats.items() if k not in _STATS_FRAMES}
    out['comparisons'] = [list(c) for c in stats.get('comparisons') or []]
    out = ag_core.to_jsonable(out)
    for key in _STATS_FRAMES:
        frame = stats.get(key)
        out[key] = _frame_to_jsonable(frame) if frame is not None else None
    return out


def stats_from_jsonable(payload: dict) -> dict:
    """The ``run_statistics`` result that ``stats_to_jsonable`` saved. The
    degrees of freedom of the count tests (infinite for a Wald test) are
    restored where they were saved as null."""
    out = {k: v for k, v in payload.items() if k not in _STATS_FRAMES}
    out['comparisons'] = [tuple(c) for c in payload.get('comparisons') or []]
    for key in _STATS_FRAMES:
        obj = payload.get(key)
        out[key] = _frame_from_jsonable(obj) if obj else None
    counts = out.get('counts')
    if counts is not None and 'df.total' in counts.columns:
        counts['df.total'] = counts['df.total'].fillna(np.inf)
    return out


_PCA_FRAMES = ('scores', 'loadings')
_PREDICTIVE_FRAMES = ('performance', 'roc', 'importance', 'panel')


def _result_to_jsonable(result: dict, frames: tuple) -> dict | None:
    if not result:
        return None
    out = ag_core.to_jsonable({k: v for k, v in result.items() if k not in frames})
    for key in frames:
        frame = result.get(key)
        out[key] = _frame_to_jsonable(frame) if frame is not None else None
    return out


def _result_from_jsonable(payload: dict | None, frames: tuple) -> dict:
    if not payload:
        return {}
    out = {k: v for k, v in payload.items() if k not in frames}
    for key in frames:
        obj = payload.get(key)
        out[key] = _frame_from_jsonable(obj) if obj else pd.DataFrame()
    return out


def sample_analysis_to_jsonable(pca: dict, predictive: dict) -> dict:
    """The sample PCA (``sample_pca``) and the predictive analysis
    (``drc_predictive.run_predictive``) as JSON-safe values, for saving;
    either may be empty."""
    return {'pca': _result_to_jsonable(pca, _PCA_FRAMES),
            'predictive': _result_to_jsonable(predictive, _PREDICTIVE_FRAMES)}


def sample_analysis_from_jsonable(payload: dict) -> tuple[dict, dict]:
    """(pca, predictive) that ``sample_analysis_to_jsonable`` saved, {} for
    either that was not."""
    pca = _result_from_jsonable(payload.get('pca'), _PCA_FRAMES)
    if pca:
        pca['explained_variance_ratio'] = tuple(pca['explained_variance_ratio'])
    return pca, _result_from_jsonable(payload.get('predictive'), _PREDICTIVE_FRAMES)


def reapply_thresholds(stats: dict, pval_threshold: float, fc_threshold: float,
                       marker_threshold: float, fdr_scope: str, use_treat: bool) -> dict:
    """*stats* with p-values, FDR and significance recomputed for new
    thresholds, without refitting."""
    from honeychrome.controller_components import differential_stats as ds

    out = dict(stats)
    for key, thr, hier in (('freq', fc_threshold, False), ('counts', fc_threshold, False),
                           ('markers', marker_threshold, True)):
        if out.get(key) is not None:
            out[key] = ds.apply_significance(out[key], pval_threshold, thr, fdr_scope=fdr_scope,
                                             use_treat=use_treat, hierarchical=hier)
    return out


def counts_only_hits(stats: dict) -> set[tuple[str, str]]:
    """(comparison, population key) of each significant counts result whose
    frequency test is not significant, when the counts were tested against
    the parent: both tests then ask the same question, so a counts-only hit
    deserves a second look. Empty when the counts were tested against the
    anchor, which asks a different question."""
    stats = stats or {}
    freq, counts = stats.get('freq'), stats.get('counts')
    if freq is None or counts is None or stats.get('counts_reference', 'parent') != 'parent':
        return set()

    def hits(df):
        sel = df[df['significant'].fillna(False).astype(bool)]
        return set(zip(sel['comparison'], sel['feature']))
    return hits(counts) - hits(freq)


# ---------------------------------------------------------------------------
# Sample tables (groups and covariates)
# ---------------------------------------------------------------------------

def read_sample_table(df: pd.DataFrame, sample_keys: list[str]) -> dict:
    """Group assignments and covariates from a table with one row per sample.

    The sample column is the first of 'sample', 'file', 'filename' or
    'name' (any case); it may hold the sample key, the file name, the file
    name without extension or a name found inside the file name (see
    ``controller_components.sample_table``). A 'group' (or 'condition')
    column assigns groups; every other column is a covariate. Empty cells
    are skipped.

    Returns {'groups': {key: group}, 'covariates': {name: {key: value}},
    'unmatched': [table rows' sample names with no sample], 'columns':
    covariate names}. Raises ValueError without a sample column.
    """
    from honeychrome.controller_components import sample_table

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    match = sample_table.match_table(df, sample_keys)
    group_col = sample_table.find_column(df, sample_table.GROUP_COLUMNS)
    cov_cols = sample_table.covariate_candidates(df, match, group_col)
    values = sample_table.table_values(df, match, group_col, cov_cols)
    return {'groups': values['groups'], 'covariates': values['covariates'],
            'unmatched': match.unmatched, 'columns': [str(c) for c in cov_cols]}


def sample_table_frame(samples: list[str], groups: dict, covariates: dict) -> pd.DataFrame:
    """One row per sample: 'sample' (key), 'group' and each covariate; the
    format read_sample_table reads back."""
    rows = []
    for key in samples:
        row = {'sample': key, 'group': groups.get(key, '')}
        for name, values in (covariates or {}).items():
            row[name] = (values or {}).get(key, '')
        rows.append(row)
    return pd.DataFrame(rows, columns=['sample', 'group'] + list(covariates or {}))


# ---------------------------------------------------------------------------
# Sample PCA and predictive features
# ---------------------------------------------------------------------------

def _source_frames(stats: dict, use_freq: bool, use_counts: bool, use_markers: bool,
                   labels: dict | None) -> tuple[list, list]:
    labels = labels or {}

    def named(df, prefix, label_of):
        return df.rename(columns={c: f"{prefix}: {label_of(c)}" for c in df.columns})

    def marker_label(feature):
        pop, _sep, ch = str(feature).partition(FEATURE_SEP)
        return f"{labels.get(pop, pop)} {labels.get(ch, ch)}"

    sources, frames = [], []
    for use, key, prefix, label_of in (
            (use_freq, 'freq_values', 'Freq', lambda c: labels.get(c, c)),
            (use_counts, 'counts_values', 'Counts', lambda c: labels.get(c, c)),
            (use_markers, 'marker_values', 'Median', marker_label)):
        df = stats.get(key) if use else None
        if df is not None and not df.empty:
            sources.append(prefix)
            frames.append(named(df.astype(float), prefix, label_of))
    return sources, frames


def sample_pca(stats: dict, use_freq: bool = True, use_counts: bool = False,
               use_markers: bool = False, n_loadings: int = 10,
               labels: dict | None = None) -> dict | None:
    """Two-component PCA of every sample of the last statistics run.

    The chosen per-sample tables (population % of parent, counts, marker
    medians) are z-scored column by column, so no table dominates on scale,
    and joined. Loadings are scaled by the square root of each component's
    variance (a correlation biplot) and reduced to the *n_loadings* longest.
    *labels* maps population keys and channels to display names.

    Returns None with fewer than 2 samples or 2 varying features, else
    {'scores' (samples × PC1, PC2), 'loadings' (features × PC1, PC2),
    'explained_variance_ratio' (PC1, PC2), 'groups' (per score row),
    'sources'}.
    """
    from sklearn.decomposition import PCA

    sources, frames = _source_frames(stats, use_freq, use_counts, use_markers, labels)
    if not frames:
        return None
    combined = pd.concat(frames, axis=1, join='inner')
    if combined.shape[0] < 2:
        return None
    std = combined.std(axis=0, ddof=0)
    combined = combined.loc[:, std > 1e-12]
    if combined.shape[1] < 2:
        return None
    z = ((combined - combined.mean(axis=0)) / combined.std(axis=0, ddof=0)).fillna(0.0)
    n_comp = min(2, z.shape[0], z.shape[1])
    pca = PCA(n_components=n_comp)
    pcs = pca.fit_transform(z.values)
    explained = list(pca.explained_variance_ratio_[:n_comp]) + [0.0] * (2 - n_comp)
    if n_comp < 2:
        pcs = np.hstack([pcs, np.zeros((pcs.shape[0], 2 - n_comp))])
    scores = pd.DataFrame(pcs[:, :2], index=combined.index, columns=['PC1', 'PC2'])
    load = np.zeros((z.shape[1], 2))
    for c in range(n_comp):
        load[:, c] = pca.components_[c] * np.sqrt(max(pca.explained_variance_[c], 0.0))
    loadings = pd.DataFrame(load, index=combined.columns, columns=['PC1', 'PC2'])
    length = np.hypot(loadings['PC1'], loadings['PC2'])
    loadings = loadings.loc[length.sort_values(ascending=False).head(max(1, int(n_loadings))).index]
    group_of = dict(zip(stats.get('samples') or [], stats.get('group_vec') or []))
    return {'scores': scores, 'loadings': loadings,
            'explained_variance_ratio': (float(explained[0]), float(explained[1])),
            'groups': [group_of.get(s, '') for s in scores.index], 'sources': sources}


def predictive_matrix(stats: dict, use_freq: bool = True, use_markers: bool = False,
                      labels: dict | None = None) -> pd.DataFrame:
    """Samples × features for the predictive analysis: log2(% of parent +
    0.01) and/or marker medians, from the last statistics run, named for
    display; missing or constant features are dropped (see
    drc_predictive.build_feature_matrix)."""
    import drc_predictive

    _sources, frames = _source_frames(stats, use_freq, False, False, labels)
    freq = frames[0] if frames else None
    if freq is not None:
        freq = freq.rename(columns=lambda c: c.split(': ', 1)[1])
    _sources, mframes = _source_frames(stats, False, False, use_markers, labels)
    med = mframes[0] if mframes else None
    if med is not None:
        med = med.rename(columns=lambda c: c.split(': ', 1)[1])
    X = drc_predictive.build_feature_matrix(freq, med, list(stats.get('samples') or []))
    return X.rename(columns=lambda c: c.replace('MFI: ', 'Median: '))


# ---------------------------------------------------------------------------
# Display subsamples
# ---------------------------------------------------------------------------

DISPLAY_FORMAT = 1


def pack_display(gate_defs: list[dict], result: ag_core.GatingResult, rows: np.ndarray,
                 events: np.ndarray, channels: list[str]) -> dict:
    """The display record of one gated sample.

    rows     : indices of the display events among the gated events.
    events   : the display events (untransformed), columns named by *channels*.

    Returns {'channels', 'events', 'populations' (population keys, hierarchy
    order), 'bits' (one packed membership row per population), 'n'}.
    """
    rows = np.asarray(rows)
    n = int(len(rows))
    keys, masks = [], []
    for g in ag_core.ordered_gate_defs(gate_defs):
        for pop, m in (result.masks.get(g['gate_name']) or {}).items():
            keys.append(ag_core.population_key(g['gate_name'], pop))
            masks.append(np.asarray(m)[rows])
    if masks:
        bits = np.packbits(np.vstack(masks).astype(bool), axis=1)
    else:
        bits = np.zeros((0, (n + 7) // 8), dtype=np.uint8)
    return {'channels': list(channels), 'events': np.asarray(events, dtype=np.float32),
            'populations': keys, 'bits': bits, 'n': n}


def _population_row(disp: dict, pop_key: str):
    index = disp.get('_index')
    if index is None:
        index = {k: i for i, k in enumerate(disp.get('populations') or [])}
        disp['_index'] = index
    return index.get(pop_key)


def display_population_mask(disp: dict, pop_key: str) -> np.ndarray | None:
    """Membership of a population ('gate/pop', or 'root') over the record's
    display events, or None when the population was not gated."""
    n = int(disp.get('n', len(disp.get('events') if disp.get('events') is not None else [])))
    if pop_key in ('root', ''):
        return np.ones(n, dtype=bool)
    if 'pop_masks' in disp:
        gate, pop = ag_core.split_population_key(pop_key)
        return (disp['pop_masks'].get(gate) or {}).get(pop)
    i = _population_row(disp, pop_key)
    if i is None:
        return None
    return np.unpackbits(disp['bits'][i], count=n).astype(bool)


def display_parent_mask(disp: dict, gate_name: str, gate_def: dict | None = None):
    """The events a gate was applied to (its parent population) over the
    record's display events, or None when unknown."""
    if 'parent_masks' in disp:
        return disp['parent_masks'].get(gate_name)
    if gate_def is None:
        return None
    if not gate_def.get('parent_gate'):
        return display_population_mask(disp, 'root')
    return display_population_mask(
        disp, ag_core.population_key(gate_def['parent_gate'], gate_def.get('parent_popul') or ''))


def display_file_name(sample_key: str) -> str:
    """File name of a sample's saved display record."""
    import hashlib
    return hashlib.sha1(str(sample_key).encode('utf-8')).hexdigest()[:20] + '.npz'


def save_display(path, disp: dict, sample_key: str, fingerprint: str):
    """Write a display record (compressed) next to its sample key and the
    run fingerprint it belongs to; replaced atomically."""
    import json
    import os
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.npz')
    meta = json.dumps({'format': DISPLAY_FORMAT, 'sample': str(sample_key),
                       'fingerprint': fingerprint, 'channels': list(disp['channels']),
                       'populations': list(disp['populations']), 'n': int(disp['n'])})
    np.savez_compressed(tmp, events=disp['events'], bits=disp['bits'], meta=np.array(meta))
    os.replace(tmp, path)


def load_display(path) -> tuple[dict, dict] | None:
    """(display record, {'sample', 'fingerprint'}) from save_display, or None."""
    import json
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['meta']))
            if meta.get('format') != DISPLAY_FORMAT:
                return None
            disp = {'channels': meta['channels'], 'events': z['events'],
                    'populations': meta['populations'], 'bits': z['bits'], 'n': int(meta['n'])}
    except Exception:
        return None
    return disp, {'sample': meta.get('sample'), 'fingerprint': meta.get('fingerprint')}


# ---------------------------------------------------------------------------
# Statistics export and population comparison
# ---------------------------------------------------------------------------

def statistics_tables(stats: dict, labels: dict | None = None,
                      antigens: dict | None = None) -> dict:
    """The statistics run as tables: {'frequencies', 'counts', 'markers'}
    for the tests that were run (every comparison, with a population label
    and, for markers, the antigen), plus 'values_frequencies' (samples ×
    populations, % of the denominator) and 'values_markers'. Counts tested
    against the parent carry a 'counts_only_hit' column
    (``counts_only_hits``)."""
    labels = labels or {}
    antigens = antigens or {}
    out = {}
    only = counts_only_hits(stats)
    flag_counts = (stats.get('counts_reference', 'parent') == 'parent'
                   and stats.get('freq') is not None)
    for key, name in (('freq', 'frequencies'), ('counts', 'counts'), ('markers', 'markers')):
        df = stats.get(key)
        if df is None:
            continue
        df = df.copy()
        pop_col = 'cluster_id' if key == 'markers' and 'cluster_id' in df.columns else 'feature'
        df.insert(0, 'population', [labels.get(k, k) for k in df[pop_col]])
        if key == 'markers' and 'channel' in df.columns:
            df.insert(1, 'antigen', [antigens.get(c, c) for c in df['channel']])
        if key == 'counts' and flag_counts:
            df['counts_only_hit'] = [(c, f) in only for c, f in zip(df['comparison'], df['feature'])]
        out[name] = df
    values = stats.get('freq_values')
    if values is not None:
        vf = values.rename(columns=lambda c: labels.get(c, c)).copy()
        vf.insert(0, 'group', list(stats.get('group_vec') or [''] * len(vf)))
        vf.index.name = 'sample'
        out['values_frequencies'] = vf
    mvalues = stats.get('marker_values')
    if mvalues is not None:
        def _label(feature):
            pop, _sep, ch = str(feature).partition(FEATURE_SEP)
            return f"{labels.get(pop, pop)} | {antigens.get(ch, ch)}"
        vm = mvalues.rename(columns=_label).copy()
        vm.insert(0, 'group', list(stats.get('group_vec') or [''] * len(vm)))
        vm.index.name = 'sample'
        out['values_markers'] = vm
    return out


def group_population_values(summaries: dict, samples: list[str], groups: dict, pop_key: str,
                            use_stats_parent: bool = True) -> dict:
    """{group: [(sample, % of parent)]} for one population, from the gated
    summaries; samples where the population cannot be measured are left out."""
    out: dict[str, list] = {}
    for s in samples:
        g = groups.get(s)
        if not g:
            continue
        row = next((r for r in (summaries.get(s) or {}).get('populations', [])
                    if r['key'] == pop_key), None)
        if row is None:
            continue
        use_sp = use_stats_parent and row.get('n_stats_parent') is not None
        denom = row['n_stats_parent'] if use_sp else row['n_parent']
        if not denom:
            continue
        out.setdefault(g, []).append((s, 100.0 * float(row['n_events']) / float(denom)))
    return out


def population_result(stats: dict, pop_key: str, comparison: str,
                      fdr_scope: str = 'global') -> dict | None:
    """The frequency test of one population in one comparison: {'logFC',
    'P', 'FDR', 'significant'}, or None when it was not tested."""
    df = (stats or {}).get('freq')
    if df is None or df.empty:
        return None
    sel = df[df['feature'] == pop_key]
    if 'comparison' in sel.columns:
        sel = sel[sel['comparison'] == comparison]
    if sel.empty:
        return None
    r = sel.iloc[0]
    fdr_col = 'adj.P.Val.global' if fdr_scope == 'global' and 'adj.P.Val.global' in sel.columns \
        else 'adj.P.Val'
    return {'logFC': float(r['logFC']), 'P': float(r['P.Value']),
            'FDR': float(r[fdr_col]) if fdr_col in sel.columns and pd.notna(r[fdr_col]) else None,
            'significant': bool(r.get('significant', False))}

