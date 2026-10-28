"""
ag_fsea.py — Flow Set Enrichment Analysis (FSEA) for the Automated Gating plugin
===============================================================================
Pure computation (no Qt). Asks whether the features of a curated immune
state (``ag_fsea_library``) shift together between two sample groups, in
the manner of GSEA (Subramanian et al. 2005) with sample-label permutation.

Features (``build_features``), from the gated per-sample summaries:

* positivity — log2 odds of a positivity population ("PD1+ (CD8 TEM)")
  within its parent: the fraction of the parent positive for the marker;
* subset — log2 odds of a phenotype population within its parent;
* abundance — log2 odds of a phenotype population within the anchor
  population (viable cells by default), so samples with different event
  totals or debris are compared on the same footing;
* marker — per-population marker medians (transformed units) of phenotype
  populations only; medians inside positivity gates repeat their parent
  and are not used.

Exact duplicates and exact mirrors (the two halves of a two-way split of
one parent) are collapsed to one feature (``collapse_redundant``).

Statistics, per comparison:

* each feature's moderated linear model (Smyth 2004, the same model as the
  Results page) gives the ranking statistic r = sign(logFC) · −log10 p with
  the ordinary p-value (no TREAT threshold), or the moderated t;
* a state's enrichment score ES is the weighted Kolmogorov–Smirnov running
  sum (exponent 1) over the features ranked by r, with DOWN members' r
  negated, so ES > 0 means the state is gained and ES < 0 lost;
* the null distribution comes from permuting the sample labels and
  refitting every feature (GSEA's phenotype permutation), which keeps the
  correlation between features; with redundant cytometry features a null
  made by drawing random feature sets is far too liberal. With groups only,
  permutations use
  group sums over all features at once (identical to ``lm_fit`` +
  ``e_bayes``); with pairing or covariates, residuals of the model without
  the group terms are permuted (Freedman & Lane 1983), within pairs when
  pairing is set, and every feature is refitted with ``lm_fit``;
* NES = ES / mean of the null ES of the same sign; p from the same-sign
  null, (1 + exceedances) / (1 + permutations) so it is never zero
  (Phipson & Smyth 2010); BH FDR over the testable states of a
  comparison; FWER as the
  fraction of permutations whose largest same-sign null NES reaches the NES
  (GSEA's estimate; liberal in the 15-sample IMMPROVE calibration, 10% of
  null labellings below 0.05, so FDR is the figure to read).

A comparison is tested only with at least ``MIN_TOTAL_SAMPLES`` samples in
its two groups and ``MIN_GROUP_SAMPLES`` in each (``enforce_min_samples``
exists for validation on smaller data, not for the UI).

v1 simplifications: random permutations without enumeration; members tied
with non-members rank ahead of them; features not estimable under a
permutation take r = 0.

References
----------
Subramanian A et al. (2005) PNAS 102:15545–15550. doi:10.1073/pnas.0506580102
Smyth GK (2004) Stat Appl Genet Mol Biol 3:Article 3. doi:10.2202/1544-6115.1027
Freedman D, Lane D (1983) J Bus Econ Stat 1:292–298. doi:10.2307/1391660
Phipson B, Smyth GK (2010) Stat Appl Genet Mol Biol 9:Article 39.
    doi:10.2202/1544-6115.1585

The file name does not end in ``_tab.py``, so the plugin loader does not
treat it as a tab.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import special, stats

import ag_core
import ag_fsea_library as fl
import ag_results

log = ag_core.get_logger(__name__)

FORMAT_VERSION = 1
MIN_TOTAL_SAMPLES = 30
MIN_GROUP_SAMPLES = 10
MIN_SET_SIZE = 5
DEFAULT_N_PERM = 2000
MAX_N_PERM = 20000
DEFAULT_SEED = 42
PERM_CHUNK = 250
FEATURE_TYPES = ('positivity', 'subset', 'abundance', 'marker')
RANK_METRICS = ('signed_log10p', 'moderated_t')
# Features need this many observed samples in each group of a comparison.
MIN_OBSERVED_PER_GROUP = 3
_P_FLOOR_LOG10 = 300.0

__all__ = [
    'FORMAT_VERSION', 'MIN_TOTAL_SAMPLES', 'MIN_GROUP_SAMPLES', 'MIN_SET_SIZE',
    'DEFAULT_N_PERM', 'MAX_N_PERM', 'DEFAULT_SEED', 'FEATURE_TYPES', 'RANK_METRICS',
    'FseaBlocked', 'FseaCancelled',
    'default_anchor', 'anchor_choices', 'build_features', 'collapse_redundant',
    'comparison_plans', 'prepare', 'group_rank_stats', 'enrichment_scores', 'running_sum',
    'run_enrichment', 'result_to_jsonable', 'result_from_jsonable', 'result_tables',
    'state_curve',
]


class FseaBlocked(RuntimeError):
    """No comparison has enough samples (or nothing can be tested)."""


class FseaCancelled(RuntimeError):
    """The run was stopped by the caller."""


# ---------------------------------------------------------------------------
# Anchor
# ---------------------------------------------------------------------------

def anchor_choices(pop_info: dict) -> list[str]:
    """Phenotype populations that can serve as the abundance anchor, in
    hierarchy order, with 'root' (all gated events) first."""
    return ['root'] + [k for k, r in pop_info.items() if r['kind'] == 'phenotype']


def default_anchor(pop_info: dict, gate_defs: list[dict] | None = None) -> str:
    """The model's anchor population (ag_core.anchor_population: the one
    marked as anchor in the Gating Model Builder, else the viable population
    of the deepest gate using a viability dye channel, else the deepest
    singlets population, else the population labelled 'Cells') when it is a
    phenotype population, else 'root'."""
    key = ag_core.anchor_population(gate_defs) if gate_defs else None
    if key in pop_info and pop_info[key]['kind'] == 'phenotype':
        return key
    return 'root'


def _descendants(pop_info: dict, key: str) -> list[str]:
    children: dict[str, list[str]] = {}
    for k, r in pop_info.items():
        children.setdefault(r['parent_key'], []).append(k)
    out, stack = [], list(children.get(key, []))
    while stack:
        k = stack.pop()
        out.append(k)
        stack.extend(children.get(k, []))
    return out


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

META_COLUMNS = ('feature', 'type', 'pop_key', 'parent_key', 'label', 'marker', 'marker_source',
                'channel', 'depth')


def _feature_label(ftype: str, rec: dict, pop_info: dict, marker: str = '',
                   channel: str = '', anchor_label: str = '') -> str:
    parent = pop_info.get(rec.get('parent_key'), {}).get('label', 'root')
    if ftype == 'positivity':
        return f"{rec['label']} in {parent}"
    if ftype == 'subset':
        return f"{rec['label']} (% of {parent})"
    if ftype == 'abundance':
        return f"{rec['label']} (% of {anchor_label})"
    return f"{marker or channel} median in {rec['label']}"


def build_features(summaries: dict, samples: list[str], gate_defs: list[dict], *,
                   anchor_key: str = '', feature_types=FEATURE_TYPES,
                   use_stats_parent: bool = True,
                   min_marker_events: int = ag_results.DEFAULT_MIN_MARKER_EVENTS,
                   exclude_gating_markers: bool = True,
                   channel_antigens: dict | None = None,
                   pop_info: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Samples × features table and feature metadata.

    Returns (values, meta, notes). Feature ids are '<type>:<population
    key>' (marker features '<type>:<population key> | <channel>'); meta has
    META_COLUMNS.
    """
    from honeychrome.controller_components import differential_stats as ds

    info = pop_info if pop_info is not None else fl.population_info(gate_defs)
    types = set(feature_types)
    notes: list[str] = []
    frames: list[pd.DataFrame] = []
    metas: list[dict] = []

    counts, denom, dropped = ag_results.population_tables(summaries, samples, use_stats_parent)
    if dropped:
        notes.append(f"{len(dropped)} population(s) measurable or non-empty in fewer than "
                     f"{ag_results.MIN_MEASURED_SAMPLES} samples were left out.")
    for ftype, kind in (('positivity', 'positivity'), ('subset', 'phenotype')):
        if ftype not in types:
            continue
        keys = [k for k in counts.columns if info.get(k, {}).get('kind') == kind
                and info[k]['parent_key'] != 'root']
        if not keys:
            continue
        vals = ds.logit_proportions(counts[keys], denom[keys])
        vals.columns = [f"{ftype}:{k}" for k in keys]
        frames.append(vals)
        for k in keys:
            rec = info[k]
            metas.append({'feature': f"{ftype}:{k}", 'type': ftype, 'pop_key': k,
                          'parent_key': rec['parent_key'],
                          'label': _feature_label(ftype, rec, info),
                          'marker': rec['marker'] if ftype == 'positivity' else '',
                          'marker_source': 'gate' if ftype == 'positivity' else '',
                          'channel': rec['channel'], 'depth': rec['depth']})

    if 'abundance' in types:
        anchor = anchor_key or default_anchor(info, gate_defs)
        under = set(_descendants(info, anchor)) if anchor != 'root' else set(info)
        keys = [k for k, r in info.items() if r['kind'] == 'phenotype' and k in under]
        n_ev = {s: {row['key']: float(row['n_events'])
                    for row in (summaries.get(s) or {}).get('populations', [])} for s in samples}
        if anchor == 'root':
            anchor_n = pd.Series({s: float((summaries.get(s) or {}).get('n_total') or 0)
                                  for s in samples})
        else:
            anchor_n = pd.Series({s: n_ev[s].get(anchor, np.nan) for s in samples})
        anchor_n = anchor_n.where(anchor_n > 0)
        cnt = pd.DataFrame({k: {s: (n_ev[s].get(k, 0.0) if np.isfinite(anchor_n[s]) else np.nan)
                                for s in samples} for k in keys}, index=list(samples), dtype=float)
        sparse = set(ds.sparse_features(cnt, ag_results.MIN_NONZERO_SAMPLES))
        keys = [k for k in keys if k not in sparse]
        if keys:
            den = pd.DataFrame({k: anchor_n for k in keys}, index=list(samples), dtype=float)
            vals = ds.logit_proportions(cnt[keys], den)
            vals.columns = [f"abundance:{k}" for k in keys]
            frames.append(vals)
            anchor_label = info.get(anchor, {}).get('label', 'all events')
            for k in keys:
                rec = info[k]
                metas.append({'feature': f"abundance:{k}", 'type': 'abundance', 'pop_key': k,
                              'parent_key': rec['parent_key'],
                              'label': _feature_label('abundance', rec, info,
                                                      anchor_label=anchor_label),
                              'marker': '', 'marker_source': '', 'channel': '',
                              'depth': rec['depth']})
        if sparse:
            notes.append(f"{len(sparse)} population(s) with events in fewer than "
                         f"{ag_results.MIN_NONZERO_SAMPLES} samples were left out of the "
                         "abundance features.")

    if 'marker' in types:
        phen = {k for k, r in info.items() if r['kind'] == 'phenotype'}
        sub = {s: {'populations': (summaries.get(s) or {}).get('populations', []),
                   'markers': {k: v for k, v in ((summaries.get(s) or {}).get('markers') or {}).items()
                               if k in phen}}
               for s in samples}
        values, mmeta, _dropped = ag_results.marker_table(
            sub, samples, gate_defs, exclude_gating_markers=exclude_gating_markers,
            min_events=min_marker_events)
        if len(mmeta):
            resolved = fl.resolve_channel_markers(
                info, sorted(set(mmeta['cluster_id'])), sorted(set(mmeta['channel'])),
                channel_antigens)
            values = values[list(mmeta['feature'])]
            values.columns = [f"marker:{f}" for f in mmeta['feature']]
            frames.append(values)
            n_amb = 0
            for _i, m in mmeta.iterrows():
                rec = info.get(m['cluster_id'], {'label': m['cluster_id'], 'parent_key': '',
                                                  'depth': 0})
                marker, source = resolved.get((m['cluster_id'], m['channel']), ('', 'ambiguous'))
                n_amb += source == 'ambiguous'
                metas.append({'feature': f"marker:{m['feature']}", 'type': 'marker',
                              'pop_key': m['cluster_id'], 'parent_key': rec.get('parent_key', ''),
                              'label': _feature_label('marker', rec, info, marker=marker,
                                                      channel=m['channel']),
                              'marker': marker, 'marker_source': source, 'channel': m['channel'],
                              'depth': rec.get('depth', 0)})
            if n_amb:
                notes.append(f"{n_amb} marker feature(s) are on channels whose marker could not be "
                             "told from the gate names in that lineage; they are tested but match "
                             "no library row.")
    if not frames:
        return pd.DataFrame(index=list(samples)), pd.DataFrame(columns=list(META_COLUMNS)), notes
    values = pd.concat(frames, axis=1)
    meta = pd.DataFrame(metas, columns=list(META_COLUMNS))
    return values, meta, notes


def collapse_redundant(values: pd.DataFrame, meta: pd.DataFrame
                       ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collapse exact duplicates and exact mirrors (x = −y in every sample,
    with the same missing samples), and drop constant features.

    Returns (values of the kept features, meta with 'kept' (the feature
    each one is represented by; '' when dropped as constant) and
    'orientation' (+1, or −1 for a mirror), hygiene table: feature, kind
    ('duplicate' | 'mirror' | 'constant'), of).
    """
    arr = values.to_numpy(dtype=float)
    seen: dict[bytes, int] = {}
    kept_cols, kept_of, orient, hygiene = [], {}, {}, []
    names = list(values.columns)
    sentinel = 1e300
    for j, name in enumerate(names):
        col = arr[:, j]
        fin = col[np.isfinite(col)]
        if fin.size == 0 or np.ptp(fin) <= 1e-12:
            kept_of[name], orient[name] = '', 1
            hygiene.append({'feature': name, 'kind': 'constant', 'of': ''})
            continue
        r = np.round(col, 9) + 0.0
        key = np.where(np.isfinite(r), r, sentinel).tobytes()
        neg = np.where(np.isfinite(r), -r + 0.0, sentinel).tobytes()
        if key in seen:
            first = names[seen[key]]
            kept_of[name], orient[name] = first, 1
            hygiene.append({'feature': name, 'kind': 'duplicate', 'of': first})
        elif neg in seen:
            first = names[seen[neg]]
            kept_of[name], orient[name] = first, -1
            hygiene.append({'feature': name, 'kind': 'mirror', 'of': first})
        else:
            seen[key] = j
            kept_cols.append(name)
            kept_of[name], orient[name] = name, 1
    meta = meta.copy()
    meta['kept'] = [kept_of.get(f, f) for f in meta['feature']]
    meta['orientation'] = [orient.get(f, 1) for f in meta['feature']]
    return (values[kept_cols], meta,
            pd.DataFrame(hygiene, columns=['feature', 'kind', 'of']))


# ---------------------------------------------------------------------------
# Comparisons and designs
# ---------------------------------------------------------------------------

def comparison_plans(samples: list[str], groups: dict, *, group_order=None,
                     contrast_mode: str = 'reference', reference: str = '',
                     pairing: dict | None = None, covariates: pd.DataFrame | None = None,
                     numeric_covariates=None, enforce_min_samples: bool = True
                     ) -> tuple[list[dict], list[dict], list[str]]:
    """Designs to fit and the comparisons read from them, as the Results
    page builds them (``ag_results.run_statistics``).

    Returns (plans, blocked, used samples). Each plan: samples, X, columns
    ({other group: design column}), group_codes (base 0, others 1…),
    comparisons [(base, other)], pairing_vec, reduced (X without the group
    columns), fast (no pairing and no covariates). blocked: [{'comparison',
    'reason'}] for comparisons with too few samples.
    """
    from honeychrome.controller_components import differential_stats as ds

    names = ag_results.qualifying_groups(samples, groups, list(group_order or []))
    if len(names) < 2:
        raise FseaBlocked(f"At least two groups need {ag_results.MIN_SAMPLES_PER_GROUP} or more "
                          "gated samples each.")
    if contrast_mode == 'reference' and reference not in names:
        reference = names[0]
    contrasts = ds.build_contrasts(names, contrast_mode, reference)
    used = [s for s in samples if (groups.get(s) or '') in names]
    blocked, plans = [], []

    def _gate(base, other, gvec):
        n_b, n_o = gvec.count(base), gvec.count(other)
        if not enforce_min_samples:
            return None
        if n_b + n_o < MIN_TOTAL_SAMPLES or min(n_b, n_o) < MIN_GROUP_SAMPLES:
            return (f"{other} vs {base}: {n_o} + {n_b} samples; enrichment needs at least "
                    f"{MIN_TOTAL_SAMPLES} in the two groups and {MIN_GROUP_SAMPLES} in each.")
        return None

    if contrast_mode == 'reference':
        subsets = [(used, contrasts)]
    else:
        subsets = [([s for s in used if groups[s] in (b, o)], [(b, o)]) for b, o in contrasts]
    for sub, comps in subsets:
        gvec = [groups[s] for s in sub]
        ok = []
        for b, o in comps:
            reason = _gate(b, o, gvec)
            if reason:
                blocked.append({'comparison': f"{o} vs {b}", 'reason': reason})
            else:
                ok.append((b, o))
        if not ok:
            continue
        base = comps[0][0]
        pvec = [str((pairing or {}).get(s, '')) for s in sub] if pairing else None
        cov = None
        if covariates is not None and len(covariates.columns):
            cov = ds.covariate_frame(covariates, sub, list(covariates.columns),
                                     numeric=list(numeric_covariates or []))
        X, _cn, gcols = ds.build_design(gvec, base, pvec, cov)
        group_idx = sorted(gcols.values())
        reduced = np.delete(X, group_idx, axis=1)
        order = [base] + [g for g in dict.fromkeys(gvec) if g != base]
        code = {g: i for i, g in enumerate(order)}
        plans.append({'samples': sub, 'group_vec': gvec, 'X': X, 'columns': gcols,
                      'group_codes': np.array([code[g] for g in gvec]), 'groups': order,
                      'comparisons': ok, 'pairing_vec': pvec, 'reduced': reduced,
                      'fast': pvec is None and cov is None})
    return plans, blocked, used


# ---------------------------------------------------------------------------
# Ranking statistics
# ---------------------------------------------------------------------------

def _prior_columns(s2: np.ndarray, d: np.ndarray):
    """``moderated_stats.fit_prior_variance`` for every column of s2/d."""
    from honeychrome.controller_components import moderated_stats as ms

    use = np.isfinite(s2) & (s2 > 0) & (d > 0)
    dd = np.where(use, d, 2.0)
    e = np.where(use, np.log(np.where(use, s2, 1.0)) - special.digamma(dd / 2.0)
                 + np.log(dd / 2.0), np.nan)
    n = use.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        e_mean = np.nanmean(np.where(use, e, np.nan), axis=0)
        e_var = np.nansum((e - e_mean) ** 2, axis=0) / np.maximum(n - 1, 1)
        tri = np.nanmean(np.where(use, special.polygamma(1, dd / 2.0), np.nan), axis=0)
        s2_mean = np.nanmean(np.where(use, s2, np.nan), axis=0)
    excess = e_var - tri
    d0 = np.full(s2.shape[1], np.inf)
    s0 = s2_mean.copy()
    fit = (n >= 2) & (excess > 0)
    if fit.any():
        d0[fit] = 2.0 * ms.trigamma_inverse(excess[fit])
        s0[fit] = np.exp(e_mean[fit] + special.digamma(d0[fit] / 2.0) - np.log(d0[fit] / 2.0))
    return d0, s0


def _signed_stat(b, t, df_total, metric: str):
    if metric == 'moderated_t':
        return t
    at = np.abs(t)
    with np.errstate(invalid='ignore'):
        lsf = np.where(np.isinf(df_total), stats.norm.logsf(at),
                       stats.t.logsf(at, np.where(np.isfinite(df_total), df_total, 1.0)))
    nl = -(np.log(2.0) + lsf) / np.log(10.0)
    return np.sign(b) * np.minimum(np.maximum(nl, 0.0), _P_FLOOR_LOG10)


def group_rank_stats(Y: np.ndarray, codes: np.ndarray, metric: str = 'signed_log10p') -> dict:
    """Moderated statistics of every feature for many labellings at once,
    for a design of an intercept and group indicators (base code 0).

    Y     : (features, samples), NaN where not measured.
    codes : (labellings, samples) integer group codes.

    Identical to ``lm_fit`` + ``e_bayes`` read out per group coefficient:
    a group with no observed samples has no coefficient, a feature without
    observed baseline samples is not estimable. Returns {group code ≥ 1:
    {'logFC', 't', 'df', 'r'}} with (labellings, features) arrays.
    """
    codes = np.atleast_2d(codes)
    obs = np.isfinite(Y)
    with np.errstate(invalid='ignore'):
        centre = np.where(obs.any(axis=1), np.nanmean(np.where(obs, Y, np.nan), axis=1), 0.0)
    Y0 = np.where(obs, Y - centre[:, None], 0.0)
    O = obs.astype(float)
    Y2 = Y0 * Y0
    K = int(codes.max()) + 1
    sums, ns, ss = [], [], []
    for k in range(K):
        A = (codes == k).T.astype(float)              # (samples, labellings)
        n_k, s_k, q_k = O @ A, Y0 @ A, Y2 @ A          # (features, labellings)
        sums.append(s_k)
        ns.append(n_k)
        with np.errstate(invalid='ignore', divide='ignore'):
            ss.append(np.where(n_k > 0, q_k - s_k * s_k / np.where(n_k > 0, n_k, 1.0), 0.0))
    n_obs = sum(ns)
    present = sum((n > 0).astype(float) for n in ns)
    est = ns[0] > 0
    d = np.where(est, n_obs - present, 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        s2 = np.where(est & (d > 0), np.maximum(sum(ss), 0.0) / np.where(d > 0, d, 1.0), np.nan)
    d0, s0 = _prior_columns(s2, d)
    s2_obs = np.where(np.isfinite(s2), s2, 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        post = np.where(np.isinf(d0)[None, :], s0[None, :], (d0 * s0 + d * s2_obs) / (d0 + d))
    post = np.where(est, post, np.nan)
    df_pooled = np.where(est, d, 0.0).sum(axis=0)
    df_total = np.where(df_pooled > 0, np.minimum(d0 + d, df_pooled), d0)
    df_total = np.where(est, df_total, np.nan)
    out = {}
    with np.errstate(invalid='ignore', divide='ignore'):
        m0 = sums[0] / np.where(ns[0] > 0, ns[0], 1.0)
        for k in range(1, K):
            ok = est & (ns[k] > 0)
            b = np.where(ok, sums[k] / np.where(ns[k] > 0, ns[k], 1.0) - m0, np.nan)
            u = np.sqrt(1.0 / np.where(ns[k] > 0, ns[k], np.nan) + 1.0 / np.where(est, ns[0], np.nan))
            t = b / (u * np.sqrt(post))
            out[k] = {'logFC': b.T, 't': t.T, 'df': df_total.T,
                      'r': _signed_stat(b, t, df_total, metric).T}
    return out


def _model_rank_stats(Y: np.ndarray, X: np.ndarray, cols: dict, metric: str) -> dict:
    """``lm_fit`` + ``e_bayes`` for one design; {group: {'logFC','t','df','r'}}
    with (features,) arrays."""
    from honeychrome.controller_components import moderated_stats as ms

    fit = ms.e_bayes(ms.lm_fit(Y, X))
    out = {}
    for g, c in cols.items():
        b = fit.coefficients[:, c]
        t = fit.t[:, c]
        out[g] = {'logFC': b, 't': t, 'df': fit.df_total,
                  'r': _signed_stat(b, t, fit.df_total, metric)}
    return out


def _within_block_permutation(rng, blocks: np.ndarray | None, n: int) -> np.ndarray:
    if blocks is None:
        return rng.permutation(n)
    idx = np.arange(n)
    out = idx.copy()
    for b in np.unique(blocks):
        members = idx[blocks == b]
        out[members] = rng.permutation(members)
    return out


# ---------------------------------------------------------------------------
# Enrichment scores
# ---------------------------------------------------------------------------

def enrichment_scores(R: np.ndarray, sets: list[tuple[np.ndarray, np.ndarray]],
                      R_sorted: np.ndarray | None = None) -> np.ndarray:
    """Weighted KS enrichment scores (exponent 1) of every set in every row.

    R    : (rows, features) ranking statistics.
    sets : [(feature indices, signs ±1)]; members' statistics are multiplied
           by their sign before ranking.
    Returns (rows, sets).
    """
    R = np.atleast_2d(R)
    nrow, G = R.shape
    Rs = np.sort(R, axis=1) if R_sorted is None else R_sorted
    all_idx = np.concatenate([s[0] for s in sets]) if sets else np.zeros(0, int)
    all_sgn = np.concatenate([s[1] for s in sets]) if sets else np.zeros(0)
    Xall = R[:, all_idx] * all_sgn
    greater = np.empty(Xall.shape)
    for i in range(nrow):
        greater[i] = G - np.searchsorted(Rs[i], Xall[i], side='right')
    out = np.zeros((nrow, len(sets)))
    start = 0
    for j, (idx, _sgn) in enumerate(sets):
        m = len(idx)
        X = Xall[:, start:start + m]
        member_greater = (R[:, idx][:, None, :] > X[:, :, None]).sum(axis=2)
        misses = greater[:, start:start + m] - member_greater
        start += m
        out[:, j] = _es_from_hits(X, misses, G, m)[0]
    return out


def _es_from_hits(X: np.ndarray, misses: np.ndarray, G: int, m: int):
    """ES from members' (signed) statistics and the number of non-members
    ranked above each. Returns (ES, order, peak index, Ph, miss fraction)."""
    order = np.argsort(-X, axis=1, kind='stable')
    Xs = np.take_along_axis(X, order, axis=1)
    Ms = np.take_along_axis(misses, order, axis=1)
    w = np.abs(Xs)
    NR = w.sum(axis=1, keepdims=True)
    w = np.where(NR > 0, w, 1.0)
    NR = np.where(NR > 0, NR, m)
    Ph = np.cumsum(w, axis=1) / NR
    Pprev = Ph - w / NR
    miss = Ms / max(G - m, 1)
    hi = Ph - miss
    lo = Pprev - miss
    top = np.maximum(hi.max(axis=1), 0.0)
    bottom = np.minimum(lo.min(axis=1), 0.0)
    es = np.where(top >= -bottom, top, bottom)
    peak = np.where(top >= -bottom, hi.argmax(axis=1), lo.argmin(axis=1))
    return es, order, peak, Ph, miss


def running_sum(r: np.ndarray, idx: np.ndarray, signs: np.ndarray) -> dict:
    """The running enrichment curve of one set over one ranking, for plots.

    Returns {'curve' (G + 1 values from 0), 'positions' (members' ranks,
    0 = top, in member order), 'ordered' (set-specific ranking statistics,
    descending), 'es', 'leading' (bool per member)}.
    """
    r = np.asarray(r, dtype=float)
    G, m = r.size, len(idx)
    x = r[idx] * signs
    rs = np.sort(r)
    greater = G - np.searchsorted(rs, x, side='right')
    member_greater = (r[idx][None, :] > x[:, None]).sum(axis=1)
    misses = (greater - member_greater)[None, :]
    es, order, peak, _Ph, _miss = _es_from_hits(x[None, :], misses, G, m)
    order, peak, es = order[0], int(peak[0]), float(es[0])
    pos = np.empty(m, dtype=int)
    pos[order] = misses[0][order] + np.arange(m)
    w = np.abs(x)
    NR = w.sum() if w.sum() > 0 else float(m)
    w = w if w.sum() > 0 else np.ones(m)
    inc = np.full(G, -1.0 / max(G - m, 1))
    inc[pos] = w / NR
    curve = np.concatenate([[0.0], np.cumsum(inc)])
    leading = np.zeros(m, dtype=bool)
    if es >= 0:
        leading[order[:peak + 1]] = True
    else:
        leading[order[peak:]] = True
    rest = np.delete(r, idx)
    ordered = np.sort(np.concatenate([rest, x]))[::-1]
    return {'curve': curve, 'positions': pos, 'ordered': ordered, 'es': es, 'leading': leading}


def _null_summary(es_obs: np.ndarray, null: np.ndarray):
    """NES, nominal p and FWER from the null ES (permutations × sets)."""
    S = es_obs.size
    nes = np.full(S, np.nan)
    p = np.full(S, np.nan)
    norm_null = np.full(null.shape, np.nan)
    for j in range(S):
        col = null[:, j]
        pos, neg = col[col >= 0], col[col < 0]
        mp = pos.mean() if pos.size else np.nan
        mn = -neg.mean() if neg.size else np.nan
        with np.errstate(invalid='ignore', divide='ignore'):
            norm_null[:, j] = np.where(col >= 0, col / mp, col / mn)
        if es_obs[j] >= 0:
            nes[j] = es_obs[j] / mp if mp and np.isfinite(mp) else np.nan
            p[j] = (1 + np.sum(pos >= es_obs[j])) / (1 + pos.size)
        else:
            nes[j] = es_obs[j] / mn if mn and np.isfinite(mn) else np.nan
            p[j] = (1 + np.sum(neg <= es_obs[j])) / (1 + neg.size)
    B = null.shape[0]
    with np.errstate(invalid='ignore'):
        max_pos = np.nanmax(np.where(norm_null >= 0, norm_null, -np.inf), axis=1)
        min_neg = np.nanmin(np.where(norm_null < 0, norm_null, np.inf), axis=1)
    fwer = np.full(S, np.nan)
    for j in range(S):
        if not np.isfinite(nes[j]):
            continue
        hits = np.sum(max_pos >= nes[j]) if nes[j] >= 0 else np.sum(min_neg <= nes[j])
        fwer[j] = (1 + hits) / (1 + B)
    return nes, p, fwer


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def prepare(summaries: dict, samples: list[str], gate_defs: list[dict],
            library: pd.DataFrame, *, anchor_key: str = '', feature_types=FEATURE_TYPES,
            marker_readout: str = 'positivity', use_stats_parent: bool = True,
            min_marker_events: int = ag_results.DEFAULT_MIN_MARKER_EVENTS,
            exclude_gating_markers: bool = True, channel_antigens: dict | None = None,
            cell_type_overrides: dict | None = None, min_set_size: int = MIN_SET_SIZE) -> dict:
    """Features, cell types and library matching, without any test.

    Returns {'anchor', 'values' (kept features), 'features' (metadata of
    every feature, with kept, orientation and cell_type), 'hygiene',
    'cell_types' (population → cell type mapping), 'matches', 'sets',
    'notes', 'pop_info'}.
    """
    info = fl.population_info(gate_defs)
    anchor = (anchor_key if (anchor_key in info or anchor_key == 'root')
              else default_anchor(info, gate_defs))
    values, meta, notes = build_features(
        summaries, samples, gate_defs, anchor_key=anchor, feature_types=feature_types,
        use_stats_parent=use_stats_parent, min_marker_events=min_marker_events,
        exclude_gating_markers=exclude_gating_markers, channel_antigens=channel_antigens,
        pop_info=info)
    kept_values, meta, hygiene = collapse_redundant(values, meta)
    pop_map = fl.map_populations(info, overrides=cell_type_overrides)
    meta['cell_type'] = [(pop_map.get(k) or {}).get('cell_type') or '' for k in meta['pop_key']]
    matches = fl.match_library(library, meta, pop_map, marker_readout)
    sets = fl.build_sets(matches, library, min_set_size)
    n_dup = int((hygiene['kind'] == 'duplicate').sum())
    n_mir = int((hygiene['kind'] == 'mirror').sum())
    if n_dup or n_mir:
        notes.append(f"{n_dup} exact duplicate and {n_mir} mirror feature(s) collapsed.")
    return {'anchor': anchor, 'values': kept_values, 'features': meta, 'hygiene': hygiene,
            'cell_types': pop_map, 'matches': matches, 'sets': sets, 'notes': notes,
            'pop_info': info}


def run_enrichment(summaries: dict, samples: list[str], groups: dict, gate_defs: list[dict],
                   library: pd.DataFrame, *, library_info: dict | None = None,
                   group_order=None, contrast_mode: str = 'reference', reference: str = '',
                   pairing: dict | None = None, covariates: pd.DataFrame | None = None,
                   numeric_covariates=None, anchor_key: str = '',
                   feature_types=FEATURE_TYPES, marker_readout: str = 'positivity',
                   rank_metric: str = 'signed_log10p', use_stats_parent: bool = True,
                   min_marker_events: int = ag_results.DEFAULT_MIN_MARKER_EVENTS,
                   exclude_gating_markers: bool = True, channel_antigens: dict | None = None,
                   cell_type_overrides: dict | None = None, n_perm: int = DEFAULT_N_PERM,
                   seed: int = DEFAULT_SEED, min_set_size: int = MIN_SET_SIZE,
                   enforce_min_samples: bool = True, progress=None, stop_check=None) -> dict:
    """Flow Set Enrichment Analysis of gated results.

    groups, pairing, covariates, numeric_covariates, contrast_mode and
    reference are as for ``ag_results.run_statistics``. library is a
    ``ag_fsea_library.load_library`` table; library_info is stored with the
    result (path, sha256, version, messages).

    progress(fraction, message) and stop_check() (True to stop) are called
    between permutation chunks; stopping raises FseaCancelled.

    Raises FseaBlocked when no comparison has enough samples or no state is
    testable. Returns the result dict (see ``result_tables`` for its
    tables).
    """
    from honeychrome.controller_components import moderated_stats as ms

    t_start = time.perf_counter()
    if rank_metric not in RANK_METRICS:
        raise ValueError(f"rank_metric must be one of {RANK_METRICS}")
    n_perm = int(min(max(int(n_perm), 100), MAX_N_PERM))

    def _progress(frac, msg):
        if progress is not None:
            progress(float(frac), msg)
        if stop_check is not None and stop_check():
            raise FseaCancelled("Enrichment stopped.")

    plans, blocked, used = comparison_plans(
        samples, groups, group_order=group_order, contrast_mode=contrast_mode,
        reference=reference, pairing=pairing, covariates=covariates,
        numeric_covariates=numeric_covariates, enforce_min_samples=enforce_min_samples)
    if not plans:
        raise FseaBlocked("No comparison has enough samples for enrichment.\n"
                          + "\n".join(b['reason'] for b in blocked))

    _progress(0.0, "Building features …")
    prep = prepare(summaries, used, gate_defs, library, anchor_key=anchor_key,
                   feature_types=feature_types, marker_readout=marker_readout,
                   use_stats_parent=use_stats_parent, min_marker_events=min_marker_events,
                   exclude_gating_markers=exclude_gating_markers,
                   channel_antigens=channel_antigens, cell_type_overrides=cell_type_overrides,
                   min_set_size=min_set_size)
    if prep['values'].shape[1] < 3:
        raise FseaBlocked("Fewer than three features to rank.")
    kept_values, meta, hygiene = prep['values'], prep['features'], prep['hygiene']
    matches, sets, pop_map, notes = prep['matches'], prep['sets'], prep['cell_types'], prep['notes']
    anchor = prep['anchor']
    testable = [s for s in sets if s['testable']]
    if not testable:
        raise FseaBlocked(f"No immune state has {min_set_size} or more matched features.")

    rng = np.random.default_rng(int(seed))
    state_rows, member_rows, rank_frames = [], [], []
    total_steps = sum(len(p['comparisons']) for p in plans) * n_perm
    done = 0
    for plan in plans:
        sub = plan['samples']
        Y = kept_values.loc[sub].to_numpy(dtype=float).T          # (features, samples)
        codes = plan['group_codes']
        feats = np.array(kept_values.columns)
        # Observed fit with the Results page's model.
        obs_stats = _model_rank_stats(Y, plan['X'], plan['columns'], rank_metric)
        for base, other in plan['comparisons']:
            k = plan['groups'].index(other)
            ob = obs_stats[other]
            n_b = np.isfinite(Y[:, codes == 0]).sum(axis=1)
            n_o = np.isfinite(Y[:, codes == k]).sum(axis=1)
            usable = ((n_b >= MIN_OBSERVED_PER_GROUP) & (n_o >= MIN_OBSERVED_PER_GROUP)
                      & np.isfinite(ob['r']))
            col_of = {f: i for i, f in enumerate(feats[usable])}
            r_obs = ob['r'][usable]
            sets_idx = []
            for s in testable:
                pairs = [(col_of[f], sg) for f, sg in zip(s['features'], s['signs']) if f in col_of]
                sets_idx.append((np.array([p[0] for p in pairs], dtype=int),
                                 np.array([p[1] for p in pairs], dtype=float)))
            ok_sets = [i for i, (idx, _s) in enumerate(sets_idx) if len(idx) >= min_set_size]
            sets_eval = [sets_idx[i] for i in ok_sets]
            comp = f"{other} vs {base}"
            es_obs = enrichment_scores(r_obs[None, :], sets_eval)[0] if sets_eval else np.zeros(0)
            Yu = Y[usable]
            null = np.zeros((n_perm, len(sets_eval)))
            if plan['fast'] and sets_eval:
                for c0 in range(0, n_perm, PERM_CHUNK):
                    b = min(PERM_CHUNK, n_perm - c0)
                    perm_codes = np.array([rng.permutation(codes) for _ in range(b)])
                    R = np.nan_to_num(group_rank_stats(Yu, perm_codes, rank_metric)[k]['r'])
                    null[c0:c0 + b] = enrichment_scores(R, sets_eval)
                    done += b
                    _progress(done / total_steps, f"{comp}: {c0 + b} of {n_perm} permutations")
            elif sets_eval:
                Xr = plan['reduced']
                fit_r = ms.lm_fit(Yu, Xr)
                fitted = np.nan_to_num(fit_r.coefficients) @ Xr.T
                resid = Yu - fitted
                blocks = (np.array(plan['pairing_vec']) if plan['pairing_vec'] is not None
                          else None)
                col = plan['columns'][other]
                for i in range(n_perm):
                    pi = _within_block_permutation(rng, blocks, len(sub))
                    st = _model_rank_stats(fitted + resid[:, pi], plan['X'], {other: col},
                                           rank_metric)[other]
                    null[i] = enrichment_scores(np.nan_to_num(st['r'])[None, :], sets_eval)[0]
                    done += 1
                    if (i + 1) % 50 == 0 or i + 1 == n_perm:
                        _progress(done / total_steps, f"{comp}: {i + 1} of {n_perm} permutations")
            nes, p, fwer = _null_summary(es_obs, null)
            fdr = ms.bh_adjust(p)
            usable_feats = feats[usable]
            label_of = dict(zip(meta['feature'], meta['label']))
            for j_eval, j_set in enumerate(ok_sets):
                s = testable[j_set]
                idx, sg = sets_idx[j_set]
                rs = running_sum(r_obs, idx, sg)
                lead = [usable_feats[i] for i, flag in zip(idx, rs['leading']) if flag]
                state_rows.append({
                    'comparison': comp, 'immune_state': s['name'], 'n_library': s['n_library'],
                    'n_matched': s['n_matched'], 'n_tested': int(len(idx)),
                    'ES': float(es_obs[j_eval]), 'NES': float(nes[j_eval]), 'p': float(p[j_eval]),
                    'FDR': float(fdr[j_eval]), 'FWER': float(fwer[j_eval]),
                    'direction': 'gained' if es_obs[j_eval] >= 0 else 'lost',
                    'leading_edge': '; '.join(label_of.get(f, f) for f in lead),
                    'n_leading': len(lead), 'note': s['note']})
                for i, sign, flag, pos in zip(idx, sg, rs['leading'], rs['positions']):
                    f = usable_feats[i]
                    member_rows.append({
                        'comparison': comp, 'immune_state': s['name'], 'feature': f,
                        'label': label_of.get(f, f), 'sign': int(sign),
                        'logFC': float(ob['logFC'][usable][i]), 't': float(ob['t'][usable][i]),
                        'r': float(r_obs[i]), 'rank': int(pos) + 1, 'leading_edge': bool(flag)})
            for j_set, s in enumerate(testable):
                if j_set in ok_sets:
                    continue
                state_rows.append({
                    'comparison': comp, 'immune_state': s['name'], 'n_library': s['n_library'],
                    'n_matched': s['n_matched'], 'n_tested': int(len(sets_idx[j_set][0])),
                    'ES': np.nan, 'NES': np.nan, 'p': np.nan, 'FDR': np.nan, 'FWER': np.nan,
                    'direction': '', 'leading_edge': '', 'n_leading': 0,
                    'note': f"not testable in this comparison: {len(sets_idx[j_set][0])} "
                            f"features measured / {min_set_size}"})
            rank_frames.append(pd.DataFrame({
                'comparison': comp, 'feature': usable_feats, 'logFC': ob['logFC'][usable],
                't': ob['t'][usable], 'df': ob['df'][usable], 'r': r_obs}))
    for s in sets:
        if not s['testable']:
            for plan in plans:
                for base, other in plan['comparisons']:
                    state_rows.append({
                        'comparison': f"{other} vs {base}", 'immune_state': s['name'],
                        'n_library': s['n_library'], 'n_matched': s['n_matched'], 'n_tested': 0,
                        'ES': np.nan, 'NES': np.nan, 'p': np.nan, 'FDR': np.nan, 'FWER': np.nan,
                        'direction': '', 'leading_edge': '', 'n_leading': 0, 'note': s['note']})
    _progress(1.0, "Enrichment complete.")
    states = pd.DataFrame(state_rows).sort_values(['comparison', 'p'], kind='stable',
                                                  na_position='last').reset_index(drop=True)
    engine = ('group-label permutation' if all(p['fast'] for p in plans)
              else 'residual permutation (Freedman–Lane)')
    return {
        'format_version': FORMAT_VERSION,
        'settings': {'anchor': anchor, 'feature_types': list(feature_types),
                     'marker_readout': marker_readout, 'rank_metric': rank_metric,
                     'n_perm': n_perm, 'seed': int(seed), 'min_set_size': int(min_set_size),
                     'contrast_mode': contrast_mode, 'use_stats_parent': bool(use_stats_parent),
                     'min_marker_events': int(min_marker_events),
                     'exclude_gating_markers': bool(exclude_gating_markers),
                     'enforce_min_samples': bool(enforce_min_samples), 'engine': engine},
        'library': dict(library_info or {}),
        'samples': list(used), 'group_vec': [groups[s] for s in used],
        'comparisons': [list(c) for p in plans for c in p['comparisons']],
        'blocked': blocked, 'notes': notes,
        'features': meta, 'hygiene': hygiene, 'matches': matches, 'states': states,
        'members': pd.DataFrame(member_rows), 'ranks': pd.concat(rank_frames, ignore_index=True),
        'sets': [{k: v for k, v in s.items()} for s in sets],
        'cell_types': {k: v for k, v in pop_map.items()},
        'seconds': round(time.perf_counter() - t_start, 2),
    }


# ---------------------------------------------------------------------------
# Tables, curves and saving
# ---------------------------------------------------------------------------

_FRAMES = ('features', 'hygiene', 'matches', 'states', 'members', 'ranks')


def result_tables(result: dict) -> dict[str, pd.DataFrame]:
    """The result's tables for CSV export: states, leading_edge (members),
    matches, features, hygiene."""
    return {'states': result['states'], 'leading_edge': result['members'],
            'matches': result['matches'], 'features': result['features'],
            'hygiene': result['hygiene']}


def state_curve(result: dict, comparison: str, state: str) -> dict | None:
    """Running-sum curve of one state in one comparison, for plotting."""
    ranks = result['ranks']
    rk = ranks[ranks['comparison'] == comparison]
    members = result['members']
    mb = members[(members['comparison'] == comparison) & (members['immune_state'] == state)]
    if rk.empty or mb.empty:
        return None
    pos = {f: i for i, f in enumerate(rk['feature'])}
    idx = np.array([pos[f] for f in mb['feature'] if f in pos], dtype=int)
    sg = np.array([s for f, s in zip(mb['feature'], mb['sign']) if f in pos], dtype=float)
    out = running_sum(rk['r'].to_numpy(dtype=float), idx, sg)
    out['signs'] = sg
    out['labels'] = [lab for f, lab in zip(mb['feature'], mb['label']) if f in pos]
    return out


def result_to_jsonable(result: dict) -> dict:
    """*result* as JSON-safe values (tables in split orientation)."""
    out = {k: v for k, v in result.items() if k not in _FRAMES}
    out = ag_core.to_jsonable(out)
    for key in _FRAMES:
        frame = result.get(key)
        out[key] = ag_results._frame_to_jsonable(frame) if frame is not None else None
    return out


def result_from_jsonable(payload: dict) -> dict:
    """The result ``result_to_jsonable`` saved."""
    out = {k: v for k, v in payload.items() if k not in _FRAMES}
    for key in _FRAMES:
        obj = payload.get(key)
        out[key] = ag_results._frame_from_jsonable(obj) if obj else pd.DataFrame()
    return out
