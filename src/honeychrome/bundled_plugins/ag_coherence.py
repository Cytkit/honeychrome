"""
ag_coherence.py — immune-state coherence and the state network
==============================================================
Pure computation (no Qt). Companion to Flow Set Enrichment Analysis
(``ag_fsea``): FSEA asks whether an immune state *shifted* between two
groups; coherence asks whether the state's features *move together* from
donor to donor within a group, in the directions the library expects.

In plain terms
--------------
An immune state is a list of expected changes, for example "in cytotoxic
CD4 expansion, Granzyme B⁺ and perforin⁺ CD4 go up and CD27⁺ and CD28⁺ CD4
go down". If the programme is active to different degrees in different
donors, a donor with more Granzyme B⁺ CD4 cells should also have more
perforin⁺ and fewer CD27⁺ CD4 cells. Coherence measures how strongly that
holds across the donors of one group. Comparing it between two groups
asks whether the programme varies as one unit in one group and not in the
other.

The state network takes one score per donor and state (how far the donor
sits above or below the group for that state) and correlates the scores of
every pair of states: which programmes rise and fall together across
donors.

Method
------
For each group with at least ``MIN_GROUP_SAMPLES`` samples:

1. **Residuals.** Each feature (log2 odds, as in FSEA) is regressed on the
   adjustment covariates within the group (intercept, Results-page
   covariates such as plate, and the trend variable when one is set) and
   the residuals are standardised. Group differences and these covariates
   then cannot make features correlate. Pairing is not in this model: a
   donor term would make the two samples of a pair mirror each other.
2. **Marker background** (option). When a sample's whole distribution for
   a marker sits higher (staining, background, fix/perm, unmixing
   spread), more cells cross a fixed positivity gate in every population.
   A marker's background is its median in the sample's marker-negative
   populations (phenotype populations whose median share positive is
   below ``NEGATIVE_SHARE``), averaged over at least
   ``MIN_NEGATIVE_POPULATIONS`` populations; each positivity feature of
   that marker is regressed on it within the group.
3. **Reliability.** Counting noise of each sample's log2 odds is
   (1/(c + p) + 1/(d − c + p)) / ln(2)², with c the population's count, d
   its denominator and p the share prior of the features. A feature's
   reliability in a group is 1 − mean counting noise / residual variance:
   the share of its donor-to-donor variation that is not counting noise.
   Rare populations (a handful of events per sample) have reliability near
   0, and their noise would weaken every correlation they enter. Members
   below the minimum reliability in either group of a comparison are left
   out of that comparison.
4. **Correlation.** Rank correlation of the standardised residuals: each
   feature is ranked on its available samples and the correlation is taken
   over the samples two features share (at least ``MIN_JOINT``); without
   missing values this is Spearman's ρ.
5. **Coherence** of a state in a group, C = mean over pairs of members of
   sign_i × sign_j × ρ_ij, over pairs not linked by the gate tree
   (``structural_pairs``) and with at least ``MIN_PAIRS`` pairs. C > 0:
   the members move as the library expects; C ≈ 0: independently.
   95% intervals from resampling donors.
6. **Network.** A state's score per sample is the mean of sign ×
   standardised residual over its members; an edge is the rank
   correlation of two states' scores within a group (t-test p, BH over the
   pairs of states). States that share members are scored without the
   shared members for their own pair.
7. **Comparison of two groups.** ΔC = C(other) − C(base), and for each
   edge Δz = atanh ρ(other) − atanh ρ(base). Both are tested against one
   relabelling null: the standardised residual rows of the two groups are
   swapped within pairs when the design is paired (unpaired samples keep
   their group), otherwise pooled and re-split. p = (1 + exceedances) /
   (1 + relabellings); BH FDR over states (ΔC) and over pairs of states
   (edges) per comparison. The number of relabellings is raised when
   needed so that an edge can reach FDR 0.05 among all pairs of states
   (``relabellings_needed``): about 5,000 for 23 states.

Calibration (UCL IMMPROVE, Foetal 75 vs Maternal 68, 23 states): with the
defaults, random halves of one group gave no state below FDR 0.05 in 40
splits, and about 5% below p 0.05.

v1 simplifications: group comparisons only (no trend coherence); members
weigh equally (library weights unused); ranks are not recomputed per pair
when samples are missing (within 0.05 of exact Spearman on IMMPROVE); the
background is one number per marker and sample (a shift of the negative
populations), not a model of the whole distribution.

The file name does not end in ``_tab.py``, so the plugin loader does not
treat it as a tab.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

FORMAT_VERSION = 1
MIN_GROUP_SAMPLES = 20
WARN_GROUP_SAMPLES = 40
DEFAULT_MIN_RELIABILITY = 0.3
NEGATIVE_SHARE = 0.05
MIN_NEGATIVE_POPULATIONS = 3
MIN_JOINT = 20
MIN_PAIRS = 3
MIN_BACKGROUND_SAMPLES = 10
DEFAULT_N_BOOT = 1000
MEMBER_TYPES = ('positivity', 'subset', 'abundance')
EDGE_FDR = 0.05
MAX_N_PERM = 20000
_FRAMES = ('coherence', 'edges', 'members', 'residuals')


class CoherenceBlocked(RuntimeError):
    """No comparison has two groups large enough for coherence."""


class CoherenceCancelled(RuntimeError):
    """The user stopped the run."""


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def relabellings_needed(n_perm: int, n_states: int) -> int:
    """Relabellings for a run: at least *n_perm*, and enough that a network
    edge can reach FDR ``EDGE_FDR`` among the n(n − 1)/2 pairs of states
    (the smallest relabelling p is 1 / (relabellings + 1)); at most
    ``MAX_N_PERM``."""
    n_pairs = n_states * (n_states - 1) // 2
    return int(min(max(int(n_perm), int(np.ceil(n_pairs / EDGE_FDR))), MAX_N_PERM))


def rank_corr(X, min_joint: int = MIN_JOINT) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise-complete rank correlation of the columns of X (samples ×
    features, NaN allowed) and the number of samples each pair shares.

    Each column is ranked on its available samples and the ranks are
    standardised; a pair's correlation is taken over the samples both have.
    Pairs sharing fewer than *min_joint* samples are NaN.
    """
    X = np.asarray(X, dtype=float)
    M = ~np.isnan(X)
    cnt = M.sum(axis=0).astype(float)
    R = stats.rankdata(X, axis=0, nan_policy='omit')
    with np.errstate(invalid='ignore', divide='ignore'):
        R = (R - (cnt + 1) / 2.0) / np.sqrt((cnt ** 2 - 1) / 12.0)
    A = np.where(M, R, 0.0)
    Mf = M.astype(float)
    num = A.T @ A
    ss = (A ** 2).T @ Mf
    with np.errstate(invalid='ignore', divide='ignore'):
        C = num / np.sqrt(ss * ss.T)
    n = Mf.T @ Mf
    C[n < min_joint] = np.nan
    np.fill_diagonal(C, 1.0)
    return C, n


def residualise(Y: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Least-squares residuals of each column of Y on the design D, fitted
    on the rows where that column is observed (NaN elsewhere)."""
    Y = np.asarray(Y, dtype=float)
    out = np.full_like(Y, np.nan)
    for j in range(Y.shape[1]):
        ok = np.isfinite(Y[:, j])
        if ok.sum() > D.shape[1] + 1:
            b, *_ = np.linalg.lstsq(D[ok], Y[ok, j], rcond=None)
            out[ok, j] = Y[ok, j] - D[ok] @ b
    return out


def standardise(R: np.ndarray) -> np.ndarray:
    with np.errstate(invalid='ignore', divide='ignore'):
        return (R - np.nanmean(R, axis=0)) / np.nanstd(R, axis=0, ddof=1)


def log2_odds_noise(counts: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Binomial counting variance of each sample's log2 odds, samples ×
    features, for the samples of one group.

    1 / (d · π̄ · (1 − π̄)) / ln(2)², with d the sample's denominator and
    π̄ = Σ counts / Σ denominators the feature's pooled proportion in the
    group. Using the pooled rather than each sample's own proportion keeps
    the estimate finite and stable where a sample has no or all events.
    """
    c = np.asarray(counts, dtype=float)
    d = np.asarray(denom, dtype=float)
    ok = np.isfinite(c) & np.isfinite(d) & (d > 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        pi = np.where(ok, c, 0.0).sum(axis=0) / np.where(ok, d, 0.0).sum(axis=0)
        pi = np.clip(pi, 0.5 / np.maximum(np.where(ok, d, 0.0).sum(axis=0), 1.0), 1.0)
        pi = np.minimum(pi, 1.0 - 1e-12)
        return np.where(ok, 1.0 / (d * pi * (1.0 - pi)), np.nan) / np.log(2.0) ** 2


def reliability(residuals: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """1 − mean counting noise / residual variance per column, at least 0
    (0 where it cannot be computed)."""
    R = np.asarray(residuals, dtype=float)
    N = np.where(np.isfinite(R), np.asarray(noise, dtype=float), np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        rel = 1.0 - np.nanmean(N, axis=0) / np.nanvar(R, axis=0, ddof=1)
    return np.clip(np.nan_to_num(rel, nan=0.0), 0.0, None)


def structural_pairs(features: list[str], meta: pd.DataFrame, pop_info: dict) -> set:
    """Index pairs (i < j) of *features* that the gate tree makes correlate.

    * abundances where one population is an ancestor of the other;
    * the same population as two feature types (e.g. abundance and subset);
    * subsets or abundances of sibling populations of one gate with
      several populations (their shares are forced apart);
    * positivity of the same marker on parents that are equal or nested.
    meta is indexed by feature (columns type, pop_key, parent_key, marker).
    """
    def ancestors(key):
        out, cur = set(), (pop_info.get(key) or {}).get('parent_key')
        while cur and cur != 'root' and cur in pop_info and cur not in out:
            out.add(cur)
            cur = pop_info[cur]['parent_key']
        return out

    n_in_gate: dict[tuple, int] = {}
    for rec in pop_info.values():
        k = (rec.get('gate'), rec.get('parent_key'))
        n_in_gate[k] = n_in_gate.get(k, 0) + 1
    info = []
    for f in features:
        r = meta.loc[f]
        rec = pop_info.get(r['pop_key'], {})
        info.append({'type': r['type'], 'pop': r['pop_key'], 'parent': r['parent_key'],
                     'marker': r.get('marker', ''), 'gate': rec.get('gate'),
                     'anc': ancestors(r['pop_key']), 'parent_anc': ancestors(r['parent_key'])})
    out = set()
    for i, a in enumerate(info):
        for j in range(i + 1, len(info)):
            b = info[j]
            if a['pop'] == b['pop'] and a['type'] != b['type']:
                out.add((i, j))
            elif a['type'] in ('abundance', 'subset') and b['type'] in ('abundance', 'subset'):
                if a['type'] == b['type'] == 'abundance' and (a['pop'] in b['anc']
                                                             or b['pop'] in a['anc']):
                    out.add((i, j))
                elif (a['type'] == b['type'] and a['parent'] == b['parent']
                      and a['gate'] is not None and a['gate'] == b['gate']
                      and n_in_gate.get((a['gate'], a['parent']), 0) > 1):
                    out.add((i, j))
            elif (a['type'] == b['type'] == 'positivity' and a['marker']
                  and a['marker'] == b['marker']
                  and (a['parent'] == b['parent'] or a['parent'] in b['parent_anc']
                       or b['parent'] in a['parent_anc'])):
                out.add((i, j))
    return out


def coherence(C: np.ndarray, sets: list[tuple[np.ndarray, np.ndarray]],
              allowed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed mean within-state correlation per set and the number of pairs
    used; NaN for a set with fewer than ``MIN_PAIRS`` usable pairs."""
    out = np.full(len(sets), np.nan)
    npairs = np.zeros(len(sets), dtype=int)
    for k, (ix, sg) in enumerate(sets):
        if len(ix) < 2:
            continue
        sub = C[np.ix_(ix, ix)] * np.outer(sg, sg)
        use = np.triu(allowed[np.ix_(ix, ix)], 1) & np.isfinite(sub)
        npairs[k] = int(use.sum())
        if npairs[k] >= MIN_PAIRS:
            out[k] = float(sub[use].mean())
    return out, npairs


def state_scores(Z: np.ndarray, sets: list[tuple[np.ndarray, np.ndarray]],
                 drop: np.ndarray | None = None) -> np.ndarray:
    """Per-sample score of each set: mean of sign × standardised residual
    over its available members (members in *drop* left out)."""
    Z = np.asarray(Z, dtype=float)
    S = np.full((Z.shape[0], len(sets)), np.nan)
    for k, (ix, sg) in enumerate(sets):
        keep = np.ones(len(ix), bool) if drop is None else ~np.isin(ix, drop)
        if keep.any():
            with np.errstate(invalid='ignore'):
                S[:, k] = np.nanmean(Z[:, ix[keep]] * sg[keep], axis=1)
    return S


def _corr_p(r: np.ndarray, n: np.ndarray) -> np.ndarray:
    with np.errstate(invalid='ignore', divide='ignore'):
        t = r * np.sqrt((n - 2) / np.clip(1 - r ** 2, 1e-12, None))
        return 2 * stats.t.sf(np.abs(t), n - 2)


def _bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = stats.false_discovery_control(p[ok])
    return q


class _Comparison:
    """Coherence and network statistics for one assignment of rows to two
    groups, over a fixed set of members."""

    def __init__(self, sets, allowed, overlaps):
        self.sets = sets
        self.allowed = allowed
        self.overlaps = overlaps          # [(a, b, shared member indices)]

    def group(self, Z):
        C, _n = rank_corr(Z)
        coh, npairs = coherence(C, self.sets, self.allowed)
        S = state_scores(Z, self.sets)
        R, n = rank_corr(S, min_joint=3)
        for a, b, drop in self.overlaps:
            r2, n2 = rank_corr(state_scores(Z, [self.sets[a], self.sets[b]], drop=drop), min_joint=3)
            R[a, b] = R[b, a] = r2[0, 1]
            n[a, b] = n[b, a] = n2[0, 1]
        return coh, npairs, R, n

    def compute(self, Za, Zb):
        ca, pa, Ra, na = self.group(Za)
        cb, pb, Rb, nb = self.group(Zb)
        with np.errstate(invalid='ignore'):
            dz = np.arctanh(np.clip(Ra, -0.999, 0.999)) - np.arctanh(np.clip(Rb, -0.999, 0.999))
        return {'Ca': ca, 'Cb': cb, 'pairs_a': pa, 'pairs_b': pb, 'Ra': Ra, 'Rb': Rb,
                'na': na, 'nb': nb, 'dC': ca - cb, 'dz': dz}


def _perm_p(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    exc = (np.abs(null) >= np.abs(obs)[None] - 1e-12).sum(axis=0)
    p = (1 + exc) / (1 + null.shape[0])
    return np.where(np.isfinite(obs), p, np.nan)


def relabel_test(Z: np.ndarray, rows_a: np.ndarray, rows_b: np.ndarray,
                 pairs: list[tuple[int, int]], comp: _Comparison, n_perm: int,
                 rng: np.random.Generator, progress=None) -> dict:
    """Observed statistics and relabelling p-values for ΔC and every edge.

    With *pairs* (row of group a, row of group b) the two rows of a pair
    swap groups with probability 1/2 and other rows keep theirs; without,
    the rows of both groups are pooled and re-split at random.
    """
    obs = comp.compute(Z[rows_a], Z[rows_b])
    null_c = np.empty((n_perm, len(obs['dC'])))
    null_z = np.empty((n_perm,) + obs['dz'].shape)
    pos_a = {r: k for k, r in enumerate(rows_a)}
    pos_b = {r: k for k, r in enumerate(rows_b)}
    pool = np.concatenate([rows_a, rows_b])
    for i in range(n_perm):
        if pairs:
            A, B = rows_a.copy(), rows_b.copy()
            for (a, b), flip in zip(pairs, rng.random(len(pairs)) < 0.5):
                if flip:
                    A[pos_a[a]], B[pos_b[b]] = b, a
        else:
            perm = rng.permutation(pool)
            A, B = perm[:len(rows_a)], perm[len(rows_a):]
        r = comp.compute(Z[A], Z[B])
        null_c[i], null_z[i] = r['dC'], r['dz']
        if progress is not None and (i + 1) % 50 == 0:
            progress((i + 1) / n_perm)
    obs['p_dC'] = _perm_p(obs['dC'], null_c)
    iu = np.triu_indices(obs['dz'].shape[0], 1)
    p_edge = np.full(obs['dz'].shape, np.nan)
    p_edge[iu] = _perm_p(obs['dz'][iu], null_z[:, iu[0], iu[1]])
    obs['p_edge'] = p_edge
    return obs


def bootstrap_coherence(Z: np.ndarray, sets, allowed, n_boot: int,
                        rng: np.random.Generator) -> np.ndarray:
    """2.5 and 97.5 percentiles of coherence over donors resampled with
    replacement (2 × sets)."""
    out = np.full((n_boot, len(sets)), np.nan)
    n = Z.shape[0]
    for b in range(n_boot):
        C, _n = rank_corr(Z[rng.integers(0, n, n)])
        out[b], _p = coherence(C, sets, allowed)
    with np.errstate(invalid='ignore'):
        return np.nanpercentile(out, [2.5, 97.5], axis=0)


# ---------------------------------------------------------------------------
# Inputs from the gated results
# ---------------------------------------------------------------------------

def feature_counts(summaries: dict, samples: list[str], meta: pd.DataFrame, anchor: str, *,
                   use_stats_parent: bool = True, min_parent_events: int = 0
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Counts and denominators behind each positivity, subset and abundance
    feature of *meta* (indexed by feature), samples × features, as
    ``ag_fsea.build_features`` makes them."""
    import ag_results

    counts, denom, _dropped = ag_results.population_tables(summaries, samples, use_stats_parent,
                                                           min_parent_events)
    n_ev = {s: {row['key']: float(row['n_events'])
                for row in (summaries.get(s) or {}).get('populations', [])} for s in samples}
    if anchor == 'root':
        anchor_n = pd.Series({s: float((summaries.get(s) or {}).get('n_total') or 0)
                              for s in samples})
    else:
        anchor_n = pd.Series({s: n_ev[s].get(anchor, np.nan) for s in samples})
    anchor_n = anchor_n.where(anchor_n > 0)
    c_out, d_out = {}, {}
    for f, r in meta.iterrows():
        k = r['pop_key']
        if r['type'] in ('positivity', 'subset') and k in counts.columns:
            c_out[f], d_out[f] = counts[k], denom[k]
        elif r['type'] == 'abundance':
            c_out[f] = pd.Series({s: (n_ev[s].get(k, 0.0) if np.isfinite(anchor_n[s]) else np.nan)
                                  for s in samples})
            d_out[f] = anchor_n
    return (pd.DataFrame(c_out, index=list(samples), dtype=float),
            pd.DataFrame(d_out, index=list(samples), dtype=float))


def negative_backgrounds(meta: pd.DataFrame, share: pd.Series, markers: set[str],
                         cutoff: float = NEGATIVE_SHARE) -> dict[str, list[str]]:
    """{marker: marker-median features in its negative populations}.

    A phenotype population is negative for a marker when its positivity
    population for that marker has a median share below *cutoff*; markers
    with fewer than ``MIN_NEGATIVE_POPULATIONS`` such populations (with a
    median feature) are left out.
    """
    pos = meta[meta['type'] == 'positivity']
    med = meta[meta['type'] == 'marker']
    out = {}
    for m in sorted(markers):
        if not m:
            continue
        neg = {r['parent_key'] for f, r in pos[pos['marker'] == m].iterrows()
               if np.isfinite(share.get(f, np.nan)) and share[f] < cutoff}
        feats = list(med[(med['marker'] == m) & med['pop_key'].isin(neg)].index)
        if len(feats) >= MIN_NEGATIVE_POPULATIONS:
            out[m] = feats
    return out


def adjust_background(R: pd.DataFrame, B: pd.DataFrame, backgrounds: dict[str, list[str]],
                      meta: pd.DataFrame) -> tuple[pd.DataFrame, set]:
    """Positivity features of R adjusted for their marker's background.

    R: residuals (samples × features); B: standardised residuals of the
    marker-median features named in *backgrounds* ({marker: features in its
    negative populations}). A marker's background is the mean of its
    columns of B; each positivity feature of that marker is replaced by its
    residual from a straight-line fit on it (samples with both values, at
    least ``MIN_BACKGROUND_SAMPLES``). Returns (adjusted R, adjusted features).
    """
    R = R.copy()
    done = set()
    for m, feats in backgrounds.items():
        b = B[feats].mean(axis=1).to_numpy()
        for f in R.columns:
            if meta.loc[f, 'type'] != 'positivity' or meta.loc[f, 'marker'] != m:
                continue
            y = R[f].to_numpy(dtype=float).copy()
            ok = np.isfinite(y) & np.isfinite(b)
            if ok.sum() < MIN_BACKGROUND_SAMPLES:
                continue
            X = np.column_stack([np.ones(ok.sum()), b[ok]])
            coef, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
            y[ok] = y[ok] - X @ coef
            R[f] = y
            done.add(f)
    return R, done


def group_design(samples: list[str], covariates: pd.DataFrame | None, numeric,
                 time: dict | None) -> tuple[np.ndarray, list[str], list[str]]:
    """Within-group design (intercept, covariates, trend variable) and the
    samples it covers; samples without a covariate or trend value are left
    out. Returns (D, used samples, term names)."""
    from honeychrome.controller_components import differential_stats as ds

    used = list(samples)
    if time:
        used = [s for s in used if np.isfinite(_as_float(time.get(s)))]
    cov = None
    if covariates is not None and len(covariates.columns) and used:
        cov = ds.covariate_frame(covariates, used, list(covariates.columns),
                                 numeric=list(numeric or []), allow_missing=True)
        ok = ~cov.isna().any(axis=1).to_numpy()
        used = [s for s, keep in zip(used, ok) if keep]
        cov = cov.iloc[np.flatnonzero(ok)]
    names = ['(Intercept)']
    if cov is not None and len(used):
        X, cn, _g = ds.build_design(['g'] * len(used), 'g', None, cov)
        names = list(cn)
    else:
        X = np.ones((len(used), 1))
    if time and len(used):
        t = np.array([_as_float(time[s]) for s in used])
        X = np.column_stack([X, t - t.mean()])
        names.append('trend variable')
    return X, used, names


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def _state_lineage(features: list[str], meta: pd.DataFrame, cell_types: dict) -> str:
    lins = []
    for f in features:
        r = meta.loc[f]
        key = r['parent_key'] if r['type'] == 'positivity' else r['pop_key']
        lin = (cell_types.get(key) or {}).get('lineage') or ''
        if lin:
            lins.append(lin)
    return max(set(lins), key=lins.count) if lins else ''


def _tree_path(key: str, pop_info: dict) -> str:
    path, cur, seen = [], key, set()
    while cur and cur != 'root' and cur in pop_info and cur not in seen:
        seen.add(cur)
        path.append(cur)
        cur = pop_info[cur]['parent_key']
    return ' > '.join(reversed(path))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_coherence(*args, **kwargs) -> dict:
    """See ``_run_coherence``. Runtime warnings from features missing in
    some samples (means of empty slices) are expected and not shown."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return _run_coherence(*args, **kwargs)


def _run_coherence(summaries: dict, samples: list[str], groups: dict, gate_defs: list[dict],
                  fsea_result: dict, *, group_order=None, contrast_mode: str = 'reference',
                  reference: str = '', pairing: dict | None = None,
                  covariates: pd.DataFrame | None = None, numeric_covariates=None,
                  trend_time: dict | None = None, trend_variable: str = '',
                  channel_antigens: dict | None = None, background: bool = True,
                  min_reliability: float = DEFAULT_MIN_RELIABILITY,
                  negative_share: float = NEGATIVE_SHARE, n_perm: int = 2000,
                  n_boot: int = DEFAULT_N_BOOT, seed: int = 42,
                  min_group_samples: int = MIN_GROUP_SAMPLES, progress=None,
                  stop_check=None) -> dict:
    """Coherence of the immune states of *fsea_result* (an
    ``ag_fsea.run_enrichment`` result: its settings, features and sets) in
    every group with enough samples, and the comparisons of the design.

    groups, pairing, covariates, numeric_covariates, contrast_mode and
    reference are as for ``ag_fsea.run_enrichment``; trend_time
    ({sample: value}) adds the trend variable to the residual model.
    progress(fraction, message) and stop_check() as in ``ag_fsea``;
    stopping raises CoherenceCancelled.

    Returns the result block (see ``result_tables``); raises
    CoherenceBlocked when no comparison has two groups of at least
    *min_group_samples* samples.
    """
    import ag_fsea
    import ag_fsea_library as fl
    import ag_results
    from honeychrome.controller_components import differential_stats as ds

    t_start = time.perf_counter()

    def _progress(frac, msg):
        if progress is not None:
            progress(float(frac), msg)
        if stop_check is not None and stop_check():
            raise CoherenceCancelled("Coherence stopped.")

    settings = fsea_result['settings']
    if (trend_variable and covariates is not None
            and trend_variable in [str(c) for c in covariates.columns]):
        # Already an adjustment covariate: one term is enough.
        trend_time = None
    names = ag_results.qualifying_groups(samples, groups, list(group_order or []))
    if contrast_mode == 'reference' and reference not in names and names:
        reference = names[0]
    contrasts = ds.build_contrasts(names, contrast_mode, reference) if len(names) >= 2 else []
    by_group = {g: [s for s in samples if groups.get(s) == g] for g in names}
    notes, blocked = [], []
    ok_groups = {g for g, ss in by_group.items() if len(ss) >= min_group_samples}
    comparisons = []
    for base, other in contrasts:
        small = [g for g in (base, other) if g not in ok_groups]
        if small:
            blocked.append({'comparison': f"{other} vs {base}",
                            'reason': f"{other} vs {base}: coherence needs at least "
                                      f"{min_group_samples} samples in each group ("
                                      + ", ".join(f"{g} {len(by_group[g])}" for g in small) + ")."})
        else:
            comparisons.append((base, other))
    if not comparisons:
        raise CoherenceBlocked("No comparison has two groups with at least "
                               f"{min_group_samples} samples.\n"
                               + "\n".join(b['reason'] for b in blocked))
    for g in sorted(ok_groups):
        if len(by_group[g]) < WARN_GROUP_SAMPLES:
            notes.append(f"{g}: {len(by_group[g])} samples; below {WARN_GROUP_SAMPLES} only large "
                         "differences in coherence can be detected.")

    _progress(0.0, "Coherence: building features")
    used_samples = [s for g in sorted(ok_groups) for s in by_group[g]]
    pop_info = fl.population_info(gate_defs)
    values, meta, _bn = ag_fsea.build_features(
        summaries, used_samples, gate_defs, anchor_key=settings['anchor'],
        feature_types=('positivity', 'subset', 'abundance', 'marker'),
        use_stats_parent=settings['use_stats_parent'],
        min_parent_events=settings['min_parent_events'],
        min_marker_events=settings['min_marker_events'],
        exclude_gating_markers=settings['exclude_gating_markers'],
        channel_antigens=channel_antigens, pop_info=pop_info)
    values, meta, _hyg = ag_fsea.collapse_redundant(values, meta)
    meta = meta.drop_duplicates('feature').set_index('feature')

    sets_all = [s for s in fsea_result['sets'] if s.get('testable')]
    states, members = [], []
    for s in sets_all:
        fs = [(f, float(sg)) for f, sg in zip(s['features'], s['signs'])
              if f in values.columns and meta.loc[f, 'type'] in MEMBER_TYPES]
        dropped = [f for f in s['features'] if f in meta.index
                   and meta.loc[f, 'type'] not in MEMBER_TYPES]
        states.append({'name': s['name'], 'features': [f for f, _ in fs],
                       'signs': np.array([g for _, g in fs]), 'dropped_types': dropped})
        members += [f for f, _ in fs]
    members = list(dict.fromkeys(members))
    if len(members) < 3:
        raise CoherenceBlocked("Fewer than three state members are proportions.")
    pos_of = {f: i for i, f in enumerate(members)}

    counts, denom = feature_counts(summaries, used_samples, meta.loc[members],
                                   settings['anchor'],
                                   use_stats_parent=settings['use_stats_parent'],
                                   min_parent_events=settings['min_parent_events'])
    counts = counts.reindex(columns=members)
    denom = denom.reindex(columns=members)
    bg = {}
    if background:
        pos_feats = [f for f in meta.index if meta.loc[f, 'type'] == 'positivity']
        allc, alld = feature_counts(summaries, used_samples, meta.loc[pos_feats],
                                    settings['anchor'],
                                    use_stats_parent=settings['use_stats_parent'],
                                    min_parent_events=settings['min_parent_events'])
        with np.errstate(invalid='ignore', divide='ignore'):
            share = (allc / alld).median(axis=0)
        markers = {meta.loc[f, 'marker'] for f in members if meta.loc[f, 'type'] == 'positivity'}
        bg = negative_backgrounds(meta, share, markers, negative_share)
    bg_cols = sorted({f for fs in bg.values() for f in fs})

    # Residuals, background and reliability within each group.
    Z = pd.DataFrame(np.nan, index=used_samples, columns=members)
    rel, adjusted, design_terms, left_out = {}, set(), {}, {}
    for g in sorted(ok_groups):
        try:
            D, used, terms = group_design(by_group[g], covariates, numeric_covariates, trend_time)
        except RuntimeError as exc:
            raise CoherenceBlocked(f"{g}: the residual model cannot be fitted ({exc}).") from exc
        design_terms[g] = terms
        left_out[g] = len(by_group[g]) - len(used)
        R = pd.DataFrame(residualise(values.loc[used, members].to_numpy(), D),
                         index=used, columns=members)
        if bg:
            B = pd.DataFrame(standardise(residualise(values.loc[used, bg_cols].to_numpy(), D)),
                             index=used, columns=bg_cols)
            R, done = adjust_background(R, B, bg, meta)
            adjusted |= done
        noise = log2_odds_noise(counts.loc[used].to_numpy(), denom.loc[used].to_numpy())
        rel[g] = pd.Series(reliability(R.to_numpy(), noise), index=members)
        Z.loc[used] = standardise(R.to_numpy())
        if left_out[g]:
            notes.append(f"{g}: {left_out[g]} sample(s) without a covariate or trend value were "
                         "left out of coherence.")

    struct = structural_pairs(members, meta, pop_info)
    allowed_all = np.ones((len(members), len(members)), bool)
    for i, j in struct:
        allowed_all[i, j] = allowed_all[j, i] = False

    n_perm_used = relabellings_needed(n_perm, len(states))
    if n_perm_used > int(n_perm):
        notes.append(f"Coherence used {n_perm_used} relabellings (not {int(n_perm)}) so that a "
                     f"network edge can reach FDR {EDGE_FDR:g} among the "
                     f"{len(states) * (len(states) - 1) // 2} pairs of states.")
    rng = np.random.default_rng(int(seed))
    coh_rows, edge_rows, member_rows = [], [], []
    Zv = Z.to_numpy()
    row_of = {s: i for i, s in enumerate(used_samples)}
    n_steps = len(comparisons)
    for ci, (base, other) in enumerate(comparisons):
        comp_label = f"{other} vs {base}"
        keep = {f for f in members if rel[other][f] >= min_reliability
                and rel[base][f] >= min_reliability}
        sets = []
        for s in states:
            ix = [k for k, f in enumerate(s['features']) if f in keep]
            sets.append((np.array([pos_of[s['features'][k]] for k in ix], dtype=int),
                         s['signs'][ix] if len(ix) else np.zeros(0)))
        overlaps = []
        for a in range(len(sets)):
            for b in range(a + 1, len(sets)):
                sh = np.intersect1d(sets[a][0], sets[b][0])
                if len(sh):
                    overlaps.append((a, b, sh))
        comp = _Comparison(sets, allowed_all, overlaps)
        rows_a = np.array([row_of[s] for s in by_group[other] if s in row_of and
                           np.isfinite(Zv[row_of[s]]).any()])
        rows_b = np.array([row_of[s] for s in by_group[base] if s in row_of and
                           np.isfinite(Zv[row_of[s]]).any()])
        pairs = []
        if pairing:
            level_b = {}
            for r in rows_b:
                level_b.setdefault(str(pairing.get(used_samples[r], '')), []).append(r)
            for r in rows_a:
                lv = str(pairing.get(used_samples[r], ''))
                if lv and len(level_b.get(lv, [])) == 1:
                    pairs.append((r, level_b[lv][0]))

        def _p(frac, _ci=ci, _c=comp_label):
            _progress((_ci + 0.8 * frac) / n_steps,
                      f"Coherence {_c}: {int(frac * n_perm_used)} of {n_perm_used} relabellings")

        obs = relabel_test(Zv, rows_a, rows_b, pairs, comp, n_perm_used, rng, _p)
        _progress((ci + 0.8) / n_steps, f"Coherence {comp_label}: intervals")
        ci_a = bootstrap_coherence(Zv[rows_a], sets, allowed_all, int(n_boot), rng)
        ci_b = bootstrap_coherence(Zv[rows_b], sets, allowed_all, int(n_boot), rng)
        q_c = _bh(obs['p_dC'])
        for k, s in enumerate(states):
            coh_rows.append({
                'comparison': comp_label, 'immune_state': s['name'],
                'lineage': _state_lineage(s['features'], meta, fsea_result.get('cell_types') or {}),
                'group_a': other, 'group_b': base, 'n_a': len(rows_a), 'n_b': len(rows_b),
                'members_total': len(s['features']), 'members_used': int(len(sets[k][0])),
                'pairs_used': int(min(obs['pairs_a'][k], obs['pairs_b'][k])),
                'C_a': obs['Ca'][k], 'C_a_lo': ci_a[0, k], 'C_a_hi': ci_a[1, k],
                'C_b': obs['Cb'][k], 'C_b_lo': ci_b[0, k], 'C_b_hi': ci_b[1, k],
                'dC': obs['dC'][k], 'p': obs['p_dC'][k], 'FDR': q_c[k],
                'paired': bool(pairs)})
        iu = np.triu_indices(len(states), 1)
        pa = _corr_p(obs['Ra'][iu], obs['na'][iu])
        pb = _corr_p(obs['Rb'][iu], obs['nb'][iu])
        qa, qb, qd = _bh(pa), _bh(pb), _bh(obs['p_edge'][iu])
        shared = {(a, b): len(d) for a, b, d in overlaps}
        for k, (a, b) in enumerate(zip(*iu)):
            edge_rows.append({
                'comparison': comp_label, 'state_1': states[a]['name'], 'state_2': states[b]['name'],
                'shared_dropped': shared.get((a, b), 0),
                'r_a': obs['Ra'][a, b], 'FDR_a': qa[k], 'r_b': obs['Rb'][a, b], 'FDR_b': qb[k],
                'dz': obs['dz'][a, b], 'p': obs['p_edge'][a, b], 'FDR': qd[k]})
        for s in states:
            for f, sg in zip(s['features'], s['signs']):
                r = meta.loc[f]
                reason = '' if f in keep else (
                    f"reliability below {min_reliability:g} (median denominator "
                    f"{other} {np.nanmedian(denom.loc[by_group[other], f]):.0f}, "
                    f"{base} {np.nanmedian(denom.loc[by_group[base], f]):.0f})")
                member_rows.append({
                    'comparison': comp_label, 'immune_state': s['name'], 'feature': f,
                    'label': r['label'], 'type': r['type'], 'sign': int(sg),
                    'reliability_a': float(rel[other][f]), 'reliability_b': float(rel[base][f]),
                    'background_adjusted': f in adjusted, 'used': f in keep, 'reason': reason,
                    'tree_path': _tree_path(r['parent_key'] if r['type'] == 'positivity'
                                            else r['pop_key'], pop_info)})
            for f in s['dropped_types']:
                member_rows.append({
                    'comparison': comp_label, 'immune_state': s['name'], 'feature': f,
                    'label': meta.loc[f, 'label'] if f in meta.index else f,
                    'type': meta.loc[f, 'type'] if f in meta.index else '', 'sign': 0,
                    'reliability_a': np.nan, 'reliability_b': np.nan,
                    'background_adjusted': False, 'used': False,
                    'reason': 'marker medians are not used for coherence', 'tree_path': ''})
        _progress((ci + 1) / n_steps, f"Coherence {comp_label} done.")

    model = sorted({t for ts in design_terms.values() for t in ts if t != '(Intercept)'})
    model = [(trend_variable or 'trend variable') if t == 'trend variable' else t for t in model]
    return {
        'format_version': FORMAT_VERSION,
        'settings': {'background': bool(background), 'min_reliability': float(min_reliability),
                     'negative_share': float(negative_share), 'n_perm': n_perm_used,
                     'n_boot': int(n_boot), 'seed': int(seed),
                     'min_group_samples': int(min_group_samples), 'model': model,
                     'markers_with_background': len(bg), 'members_adjusted': len(adjusted)},
        'comparisons': [[b, o] for b, o in comparisons], 'blocked': blocked, 'notes': notes,
        'coherence': pd.DataFrame(coh_rows), 'edges': pd.DataFrame(edge_rows),
        'members': pd.DataFrame(member_rows),
        'residuals': Z.reset_index().rename(columns={'index': 'sample'}),
        'groups': {s: groups[s] for s in used_samples},
        'structural_pairs': [[members[i], members[j]] for i, j in sorted(struct)],
        'seconds': round(time.perf_counter() - t_start, 2),
    }


# ---------------------------------------------------------------------------
# Reading the result
# ---------------------------------------------------------------------------

def comparison_labels(block: dict) -> list[str]:
    return [f"{o} vs {b}" for b, o in (block or {}).get('comparisons') or []]


def pair_view(block: dict, comparison: str, state_1: str, state_2: str) -> dict | None:
    """What the state-pair figure needs: per group, each donor's two state
    scores and the rank correlation of every pair of member features
    (structural pairs NaN); member labels, signs, owning state and tree
    path, in gate-tree order."""
    mem = block['members']
    mem = mem[(mem['comparison'] == comparison) & mem['used']]
    m1 = mem[mem['immune_state'] == state_1]
    m2 = mem[mem['immune_state'] == state_2]
    if m1.empty or m2.empty:
        return None
    res = block['residuals'].set_index('sample')
    feats = list(dict.fromkeys(list(m1['feature']) + list(m2['feature'])))
    info = pd.concat([m1, m2]).drop_duplicates('feature').set_index('feature')
    owner = {f: ('both' if f in set(m1['feature']) and f in set(m2['feature'])
                 else (state_1 if f in set(m1['feature']) else state_2)) for f in feats}
    feats.sort(key=lambda f: (info.loc[f, 'tree_path'], info.loc[f, 'label']))
    struct = {tuple(p) for p in block.get('structural_pairs') or []}
    row = block['coherence']
    row = row[row['comparison'] == comparison].iloc[0]
    out = {'features': feats, 'labels': [info.loc[f, 'label'] for f in feats],
           'owner': [owner[f] for f in feats], 'paths': [info.loc[f, 'tree_path'] for f in feats],
           'groups': {}}
    for g in (row['group_b'], row['group_a']):
        samples = [s for s, gg in block['groups'].items() if gg == g and s in res.index]
        Zg = res.loc[samples, feats].to_numpy(dtype=float)
        C, _n = rank_corr(Zg)
        for i, a in enumerate(feats):
            for j, b in enumerate(feats):
                if (a, b) in struct or (b, a) in struct:
                    C[i, j] = np.nan
        s1 = np.nanmean(res.loc[samples, list(m1['feature'])].to_numpy(dtype=float)
                        * m1['sign'].to_numpy(dtype=float), axis=1)
        s2 = np.nanmean(res.loc[samples, list(m2['feature'])].to_numpy(dtype=float)
                        * m2['sign'].to_numpy(dtype=float), axis=1)
        out['groups'][g] = {'corr': C, 'score_1': s1, 'score_2': s2}
    return out


def result_tables(block: dict) -> dict[str, pd.DataFrame]:
    """Tables for CSV export: coherence, network edges, members."""
    if not block:
        return {}
    return {'coherence': block['coherence'], 'state_network': block['edges'],
            'coherence_members': block['members']}


def to_jsonable(block: dict) -> dict:
    import ag_core
    import ag_results
    out = ag_core.to_jsonable({k: v for k, v in block.items() if k not in _FRAMES})
    for key in _FRAMES:
        out[key] = ag_results._frame_to_jsonable(block[key]) if key in block else None
    return out


def from_jsonable(payload: dict) -> dict:
    import ag_results
    out = {k: v for k, v in payload.items() if k not in _FRAMES}
    for key in _FRAMES:
        obj = payload.get(key)
        out[key] = ag_results._frame_from_jsonable(obj) if obj else pd.DataFrame()
    return out
