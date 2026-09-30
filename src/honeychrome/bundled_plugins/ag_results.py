"""
ag_results.py — channel alignment, per-sample summaries and feature tables
==========================================================================
Pure computation for the Automated Gating plugin (no Qt, no controller):

* ``align_channels`` — match a model's gated channels to an experiment's
  channels, antigen to antigen.
* ``summarise_sample`` — the compact per-sample record kept after gating:
  population counts and parent counts, per-population marker medians and
  quartiles, per-gate status and the boundaries applied.
* ``population_tables`` / ``marker_table`` — samples × features tables for
  the differential tests in ``controller_components.differential_stats``.
* ``sample_flags`` / ``population_export`` / ``marker_export`` — QC flags
  and long-format export tables.

The file name does not end in ``_tab.py``, so the plugin loader does not
treat it as a tab.
"""

from __future__ import annotations

import difflib
import re

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

__all__ = [
    'ALIGN_EXACT', 'ALIGN_ANTIGEN', 'ALIGN_CHANNEL', 'ALIGN_FUZZY', 'ALIGN_AMBIGUOUS',
    'CONFIDENT_ALIGNMENTS', 'FEATURE_SEP', 'DEFAULT_MIN_MARKER_EVENTS',
    'align_channels', 'alignment_problems', 'apply_alignment',
    'summarise_sample', 'population_tables', 'marker_table', 'marker_feature',
    'sample_flags', 'population_export', 'marker_export',
    'MIN_SAMPLES_PER_GROUP', 'qualifying_groups', 'run_statistics', 'reapply_thresholds',
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


def apply_alignment(model: dict, alignment: dict[str, str]) -> dict:
    """Copy of a normalised model with its channels renamed to the
    experiment's (gate axes, channel map and transform keys)."""
    out = dict(model)
    out['gate_definitions'] = ag_core.rename_channels(model.get('gate_definitions') or [], alignment)
    out['channel_map'] = ag_core.rekey(model.get('channel_map'), alignment)
    out['transforms'] = ag_core.rekey(model.get('transforms'), alignment)
    return out


# ---------------------------------------------------------------------------
# Per-sample summary
# ---------------------------------------------------------------------------

def summarise_sample(gate_defs: list[dict], result: ag_core.GatingResult, data_t: np.ndarray,
                     channel_index: dict, marker_channels: list[str], n_total: int) -> dict:
    """Compact, JSON-safe record of one sample's gating.

    data_t          : the sample's transformed events (rows as gated).
    marker_channels : channels summarised within every population.

    Returns {'populations': population_table rows, 'markers': {population
    key: {channel: [median, q25, q75]}}, 'status': per-gate status,
    'boundaries': the boundaries applied, 'n_total': events gated}.
    Marker quartiles are in transformed units.
    """
    rows = ag_core.population_table(gate_defs, result, n_total)
    cols = [(ch, channel_index[ch]) for ch in marker_channels if ch in channel_index]
    col_idx = [ci for _ch, ci in cols]
    markers: dict[str, dict[str, list[float]]] = {}
    if col_idx:
        for gate_def in gate_defs:
            name = gate_def['gate_name']
            for pop, m in (result.masks.get(name) or {}).items():
                if not m.any():
                    continue
                q = np.percentile(data_t[np.ix_(m, col_idx)], [50, 25, 75], axis=0)
                markers[ag_core.population_key(name, pop)] = {
                    ch: [float(q[0, j]), float(q[1, j]), float(q[2, j])]
                    for j, (ch, _ci) in enumerate(cols)
                }
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

def population_tables(summaries: dict, samples: list[str],
                      use_stats_parent: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Population counts and their denominators, samples × populations.

    The denominator is the parent population's count, or the population's
    ``stats_parent`` count when one is defined and *use_stats_parent* is
    set. Populations missing from any sample, or with a zero denominator in
    any sample, are dropped.

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
    bad = [k for k in order
           if counts_df[k].isna().any() or denom_df[k].isna().any() or (denom_df[k] <= 0).any()]
    keep = [k for k in order if k not in bad]
    return counts_df[keep], denom_df[keep], bad


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
    min_events             : a (population, channel) feature is kept only
        when the population has at least this many events in every sample.

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
        present = all(pop_key in ((summaries.get(s) or {}).get('markers') or {}) for s in samples)
        if not present or any(per_sample.get(s, 0) < min_events for s in samples):
            dropped.append(pop_key)
            continue
        gate_name, _pop = ag_core.split_population_key(pop_key)
        skip = type_markers.get(gate_name, set()) if exclude_gating_markers else set()
        channels = None
        for s in samples:
            chans = set(((summaries[s].get('markers') or {}).get(pop_key) or {}).keys())
            channels = chans if channels is None else channels & chans
        for ch in sorted(channels or [], key=str):
            if ch in skip:
                continue
            feature = marker_feature(pop_key, ch)
            values[feature] = {s: float(summaries[s]['markers'][pop_key][ch][0]) for s in samples}
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


def marker_export(summaries: dict, samples: list[str], antigens: dict | None = None,
                  groups: dict | None = None) -> pd.DataFrame:
    """Long table: one row per sample × population × channel, with the
    median and quartiles in transformed units."""
    rows = []
    for s in samples:
        summ = summaries.get(s) or {}
        n_by_pop = {r['key']: r['n_events'] for r in summ.get('populations', [])}
        for pop_key, chans in (summ.get('markers') or {}).items():
            for ch, (med, q25, q75) in chans.items():
                rows.append({'sample': s, 'group': (groups or {}).get(s, ''),
                             'population': pop_key, 'n_events': n_by_pop.get(pop_key),
                             'channel': ch, 'antigen': (antigens or {}).get(ch, ch),
                             'median': med, 'q25': q25, 'q75': q75, 'iqr': q75 - q25})
    return pd.DataFrame(rows)


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
                   pval_threshold: float = 0.05, fc_threshold: float = 0.5,
                   marker_threshold: float = 0.03, use_treat: bool = True,
                   fdr_scope: str = 'global', run_freq: bool = True, run_counts: bool = False,
                   run_markers: bool = True, use_stats_parent: bool = True,
                   exclude_gating_markers: bool = True,
                   min_marker_events: int = DEFAULT_MIN_MARKER_EVENTS) -> dict:
    """Compare gated populations between sample groups.

    * Frequencies: log2 odds of each population within its parent (or its
      stats parent), moderated linear model.
    * Counts: negative-binomial GLM with the log parent count as offset.
    * Marker medians: moderated linear model on per-population medians, one
      family per population (population-first testing).

    groups     : {sample: group name}; samples without a group are left out.
    pairing    : {sample: pairing level} (e.g. donor), or None.
    covariates : adjustment covariates, indexed by sample, or None.

    Returns {'samples', 'group_vec', 'comparisons', 'freq', 'freq_values'
    (% of denominator), 'counts', 'counts_values', 'markers',
    'marker_values', 'notes'}; a result is None when not requested. Raises
    RuntimeError when fewer than two groups qualify.
    """
    from honeychrome.controller_components import differential_stats as ds

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
        cov = ds.covariate_frame(covariates, used, list(covariates.columns))

    out = {'samples': used, 'group_vec': group_vec, 'comparisons': contrasts,
           'freq': None, 'freq_values': None, 'counts': None, 'counts_values': None,
           'markers': None, 'marker_values': None, 'notes': []}
    common = dict(pairing_vec=pairing_vec, fdr_scope=fdr_scope, covariates=cov,
                  use_treat=use_treat)

    if run_freq or run_counts:
        counts, denom, dropped = population_tables(summaries, used, use_stats_parent)
        if dropped:
            out['notes'].append("Populations missing or empty in some samples were left out: "
                                + ", ".join(dropped))
        if run_freq:
            data = ds.logit_proportions(counts, denom)
            out['freq'] = ds.run_moderated(data, group_vec, contrasts, contrast_mode,
                                           pval_threshold, fc_threshold, **common)
            out['freq_values'] = 100.0 * counts / denom
        if run_counts:
            out['counts'] = ds.run_glm_counts(counts, group_vec, contrasts, contrast_mode,
                                              pval_threshold, fc_threshold,
                                              offset_df=denom, **common)
            out['counts_values'] = counts

    if run_markers:
        values, meta, dropped = marker_table(
            summaries, used, gate_defs, exclude_gating_markers=exclude_gating_markers,
            min_events=min_marker_events,
            population_labels={r['key']: f"{r['gate']} {r['label']}"
                               for s in used for r in summaries[s].get('populations', [])})
        if dropped:
            out['notes'].append(
                f"Populations with fewer than {min_marker_events} events in some samples "
                "were left out of the marker tests: " + ", ".join(dropped))
        out['markers'] = ds.run_moderated(values, group_vec, contrasts, contrast_mode,
                                          pval_threshold, marker_threshold,
                                          feature_meta=meta, hierarchical=True, **common)
        out['marker_values'] = values
    return out


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
