"""
anova_stats.py — how much each variable matters, per feature
===========================================================

An ANOVA-style readout alongside the group comparisons of
``differential_stats``: for every feature (population or cluster
frequency, marker intensity), one test per variable in the design and the
share of the feature's variation that each variable explains. Used by the
Honeycluster and Automated Gating plugins.

In plain terms, for a design such as knockout vs wild-type mice, four
tissues per mouse:

* **Tests** (one p-value per variable): does knowing the genotype help
  predict this population's frequency once tissue and mouse are accounted
  for? Does knowing the tissue? Does the genotype difference depend on the
  tissue (the genotype × tissue interaction)? Is there a mouse-to-mouse
  effect beyond those?
* **Share of variation** (no p-value, a description): of the spread of
  this population across all samples, what percentage goes with genotype,
  with tissue, with the individual mouse, and what is left over.

Methods:

* Terms: the group; each adjustment covariate (numeric columns as
  centred continuous terms, others as categories); with ``interactions``,
  group × each covariate; and the individual (pairing variable), either as
  fixed indicators (individuals with samples in several groups) or as a
  random effect (individuals nested in the groups), as chosen by
  ``differential_stats.resolve_pairing``.
* Fixed terms are tested by moderated F tests (Smyth 2004; limma's
  ``topTableF``) of "Type II" form: each term is added to the model of
  every term that does not contain it (a main effect is tested after the
  other main effects, before its interactions; an interaction after all
  main effects), and every F uses the moderated residual variance of the
  full model. Type II keeps a term's test independent of the order in
  which terms are listed, which matters when groups are unbalanced
  (5 vs 3 mice).
* With the individual as a random effect, every fit is generalised least
  squares with the consensus correlation within individuals
  (``moderated_stats.block_correlation``, after limma's
  duplicateCorrelation), so the group is judged against the variation
  between individuals. The individual's own test is a REML likelihood
  ratio for a correlation above 0, against the 50:50 mixture of χ²₀ and
  χ²₁ (Stram & Lee 1994); it is not moderated.
* Partial R² of each fixed term: the share of the variation left after the
  other terms that the term explains (whitened values when the individual
  is random).
* Share of variation: ``moderated_stats.variance_components``, the model
  variancePartition fits (Hoffman & Schadt 2016): the group, categorical
  covariates, group × categorical covariate and the individual as random
  effects, numeric covariates as fixed terms (their share is the variance
  of their fitted contribution). Group × numeric covariate interactions are
  tested but get no share.
* Benjamini–Hochberg FDR within each term, over features.

Deliberate v1 simplifications: interactions are with the group only
(not between covariates); samples missing a covariate value are left out
of the whole readout rather than per term; the counts GLM has no ANOVA
form, so counts are not included.

References
----------
Smyth GK (2004). Stat Appl Genet Mol Biol 3(1), Article 3.
    doi:10.2202/1544-6115.1027
Smyth GK, Michaud J, Scott HS (2005). Bioinformatics 21(9):2067-2075.
    doi:10.1093/bioinformatics/bti270
Hoffman GE, Schadt EE (2016). BMC Bioinformatics 17:483.
    doi:10.1186/s12859-016-1323-z
Stram DO, Lee JW (1994). Biometrics 50(4):1171-1177.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from honeychrome.controller_components import moderated_stats as ms

log = logging.getLogger(__name__)

__all__ = ['run_anova', 'anova_wide', 'display_rows', 'TERM_COLUMNS']

#: Columns of the run_anova table.
TERM_COLUMNS = ('feature', 'term', 'kind', 'test', 'df1', 'df2', 'statistic', 'P.Value',
                'adj.P.Val', 'partial.R2', 'variance', 'variance.pct')

RESIDUAL = 'Residuals'


@dataclass
class _Term:
    name: str
    kind: str                     # 'fixed' or 'random'
    components: frozenset
    cols: np.ndarray              # (samples, q) design columns; empty for random


def _indicators(values: list, levels: list) -> np.ndarray:
    return np.column_stack([[1.0 if v == lvl else 0.0 for v in values] for lvl in levels])


def _full_rank(cols: list) -> bool:
    X = np.column_stack(cols)
    return np.linalg.matrix_rank(X) == X.shape[1]


def _covariate_columns(series: pd.Series) -> tuple[np.ndarray, bool]:
    """Design columns of one covariate and whether it is numeric."""
    if pd.api.types.is_numeric_dtype(series):
        x = series.to_numpy(dtype=float)
        return (x - x.mean())[:, None], True
    values = series.tolist()
    return _indicators(values, sorted(set(values))[1:]), False


def _design(terms: list) -> tuple[np.ndarray, dict]:
    """Intercept and the columns of ``terms``; {term name: column indices}."""
    n = terms[0].cols.shape[0] if terms else 0
    cols, index = [np.ones((n, 1))], {}
    pos = 1
    for t in terms:
        index[t.name] = list(range(pos, pos + t.cols.shape[1]))
        cols.append(t.cols)
        pos += t.cols.shape[1]
    return np.column_stack(cols), index


def run_anova(data_df: pd.DataFrame, group_vec: list[str], *,
              pairing_vec: list[str] | None = None,
              block_vec: list[str] | None = None,
              individual: str = 'Individual',
              covariates: pd.DataFrame | None = None,
              interactions: bool = True,
              group_label: str = 'Group',
              feature_meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Term-by-term tests and shares of variation for every feature.

    data_df    : rows = samples (aligned to group_vec), columns = features,
                 on the scale tested (log2 odds, log2 %, transformed MFI);
                 NaN = not measured.
    group_vec  : group of each sample; the first group in row order is the
                 baseline of the group columns (the test does not depend on it).
    pairing_vec, block_vec: the individual of each sample as a fixed term
                 or as a random effect (differential_stats.resolve_pairing);
                 at most one. ``individual`` names that term.
    covariates : adjustment covariates (differential_stats.covariate_frame),
                 rows aligned to data_df. Samples missing a value are left
                 out with a note.
    interactions: add group × each covariate.
    feature_meta: optional columns merged onto the table by 'feature'.

    Returns a long table, one row per feature and term (TERM_COLUMNS):
    kind 'fixed' (F test), 'random' (likelihood ratio test of the
    individual) or 'residual'; 'variance' and 'variance.pct' are the share
    of variation (NaN for a term without one). ``attrs['notes']`` holds
    plain-language notes and ``attrs['settings']`` the term order,
    the individual's mode and the consensus correlation.
    """
    if pairing_vec is not None and block_vec is not None:
        raise ValueError("pass pairing_vec or block_vec, not both")
    notes: list[str] = []
    keep = np.ones(len(group_vec), dtype=bool)
    if covariates is not None and len(covariates.columns):
        missing = covariates.isna().to_numpy().any(axis=1)
        if missing.any():
            notes.append(f"{int(missing.sum())} sample(s) without a value for every "
                         "adjustment covariate were left out of the variance table.")
            keep &= ~missing
    rows = np.flatnonzero(keep)
    groups = [group_vec[i] for i in rows]
    pairing = None if pairing_vec is None else [pairing_vec[i] for i in rows]
    block = None if block_vec is None else [block_vec[i] for i in rows]
    cov = None if covariates is None else covariates.iloc[rows]
    individual_vec = pairing if pairing is not None else block
    order = list(dict.fromkeys(groups))
    if len(order) < 2:
        raise RuntimeError("The variance table needs at least two groups.")
    n = len(rows)
    Y = data_df.iloc[rows].to_numpy(dtype=float).T
    if Y.shape[0] < 3:
        raise RuntimeError("The variance table needs at least three features.")

    terms = [_Term(group_label, 'fixed', frozenset({'group'}), _indicators(groups, order[1:]))]
    base_cols = [np.ones((n, 1)), terms[0].cols]
    absorbed = set()
    if cov is not None and pairing is not None:
        levels = np.asarray(pairing, dtype=object)
        for name in cov.columns:
            column = cov[name].reset_index(drop=True)
            if column.nunique() > 1 and (column.groupby(levels).nunique() <= 1).all():
                absorbed.add(name)
    cov_terms = []
    numeric = set()
    if cov is not None:
        for name in cov.columns:
            if cov[name].nunique() < 2:
                notes.append(f"'{name}' has one value over these samples and was left out.")
                continue
            if name in absorbed:
                notes.append(f"'{name}' is the same within every {individual}, so the fixed "
                             f"{individual} term already accounts for it; it was left out of "
                             "the tests.")
                continue
            cols, is_num = _covariate_columns(cov[name])
            if not _full_rank(base_cols + [cols]):
                raise RuntimeError(f"Covariate '{name}' is confounded with the groups (e.g. every "
                                   "sample of one group shares one level). Remove it from the "
                                   "adjustment list.")
            if is_num:
                numeric.add(name)
            base_cols.append(cols)
            term = _Term(name, 'fixed', frozenset({name}), cols)
            cov_terms.append(term)
    terms.extend(cov_terms)
    if pairing is not None:
        levels = sorted(set(pairing))
        cols = _indicators(pairing, levels[1:])
        if _full_rank(base_cols + [cols]):
            base_cols.append(cols)
            terms.append(_Term(individual, 'fixed', frozenset({'individual'}), cols))
        else:
            notes.append(f"The fixed {individual} term is confounded with the groups or "
                         "covariates and was left out.")
    if interactions:
        for ct in cov_terms:
            cols = np.column_stack([g[:, None] * ct.cols for g in terms[0].cols.T])
            name = f"{group_label} × {ct.name}"
            if _full_rank(base_cols + [cols]):
                base_cols.append(cols)
                terms.append(_Term(name, 'fixed', frozenset({'group', ct.name}), cols))
            else:
                notes.append(f"{name} could not be estimated (some combinations have no "
                             "samples, or it is confounded with the individual) and was left "
                             "out.")

    X_full, index = _design(terms)
    correlation = np.nan
    bc = None
    if block is not None:
        bc = ms.block_correlation(Y, X_full, block)
        correlation = bc.consensus
        rho = correlation if np.isfinite(correlation) else 0.0

        def fitter(y, x):
            return ms.gls_lm_fit(y, x, block, rho)
    else:
        fitter = ms.lm_fit
    try:
        full = ms.e_bayes(fitter(Y, X_full))
    except ValueError as exc:
        raise RuntimeError(f"{exc} Too few features have enough observed samples for the "
                           "variance table.") from exc

    def _rss(fit):
        return np.where(fit.df_residual > 0, fit.sigma2 * fit.df_residual, np.nan)

    rss_cache: dict[tuple, tuple] = {}

    def _rss_of(names: tuple) -> tuple:
        if names not in rss_cache:
            X, _idx = _design([t for t in terms if t.name in names])
            fit = full if X.shape[1] == X_full.shape[1] else fitter(Y, X)
            rss_cache[names] = (_rss(fit), fit.df_residual.astype(float))
        return rss_cache[names]

    G = Y.shape[0]
    tests: dict[str, dict] = {}
    for t in terms:
        ref = tuple(u.name for u in terms if not (u.components > t.components))
        red = tuple(name for name in ref if name != t.name)
        rss_ref, df_ref = _rss_of(ref)
        rss_red, df_red = _rss_of(red)
        df1 = df_red - df_ref
        with np.errstate(divide='ignore', invalid='ignore'):
            drop = np.maximum(rss_red - rss_ref, 0.0)
            F = np.where((df1 > 0) & (full.df_residual > 0), drop / df1 / full.s2_post, np.nan)
            r2 = np.where(rss_red > 0, drop / rss_red, np.nan)
        df1 = np.where(np.isfinite(F), df1, np.nan)
        tests[t.name] = {'kind': 'fixed', 'test': 'F', 'df1': df1, 'df2': full.df_total,
                         'statistic': F, 'P.Value': ms.f_pvalues(F, df1, full.df_total),
                         'partial.R2': np.where(np.isfinite(F), r2, np.nan)}
    if bc is not None:
        tests[individual] = {'kind': 'random', 'test': 'LR', 'df1': np.where(
            np.isfinite(bc.lr), 1.0, np.nan), 'df2': np.full(G, np.nan),
            'statistic': bc.lr, 'P.Value': bc.p_value, 'partial.R2': np.full(G, np.nan),
            'correlation': bc.rho}

    random_factors: dict[str, list] = {group_label: groups}
    fixed_terms: dict[str, np.ndarray] = {}
    if cov is not None:
        for ct in cov_terms:
            if ct.name in numeric:
                fixed_terms[ct.name] = ct.cols
            else:
                random_factors[ct.name] = cov[ct.name].tolist()
        for name in absorbed:
            if pd.api.types.is_numeric_dtype(cov[name]):
                x = cov[name].to_numpy(dtype=float)
                fixed_terms[name] = (x - x.mean())[:, None]
            else:
                random_factors[name] = cov[name].tolist()
    for t in terms:
        if len(t.components) == 2:
            other = next(iter(t.components - {'group'}))
            if other not in numeric:
                random_factors[t.name] = [f"{g}\x1f{v}" for g, v in
                                          zip(groups, cov[other].tolist())]
    if individual_vec is not None:
        random_factors[individual] = list(individual_vec)
    vc = ms.variance_components(Y, random_factors, fixed_terms)
    unconverged = int((~vc.converged & np.isfinite(vc.variance).all(axis=1)).sum())
    if unconverged:
        notes.append(f"The variance shares of {unconverged} feature(s) did not fully converge.")

    term_order = [t.name for t in terms if t.kind == 'fixed' and t.name != individual]
    absorbed_order = [name for name in vc.names if name not in term_order
                      and name not in (individual, RESIDUAL)]
    term_order += absorbed_order
    if individual_vec is not None:
        term_order.append(individual)
    term_order.append(RESIDUAL)

    features = list(data_df.columns)
    frames = []
    nan = np.full(G, np.nan)
    for name in term_order:
        t = tests.get(name, {})
        if name == RESIDUAL:
            kind = 'residual'
        else:
            kind = t.get('kind', 'random' if name in random_factors else 'fixed')
        if name in vc.names:
            j = vc.names.index(name)
            variance, pct = vc.variance[:, j], 100.0 * vc.fraction[:, j]
        else:
            variance, pct = nan, nan
        p = t.get('P.Value', nan)
        frame = pd.DataFrame({
            'feature': features, 'term': name, 'kind': kind, 'test': t.get('test', ''),
            'df1': t.get('df1', nan), 'df2': t.get('df2', nan),
            'statistic': t.get('statistic', nan), 'P.Value': p,
            'adj.P.Val': ms.bh_adjust(p), 'partial.R2': t.get('partial.R2', nan),
            'variance': variance, 'variance.pct': pct})
        if 'correlation' in t:
            frame['correlation'] = t['correlation']
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    if feature_meta is not None:
        table = table.merge(feature_meta, on='feature', how='left', sort=False)
    mode = 'random' if block is not None else ('fixed' if pairing is not None else '')
    table.attrs['notes'] = notes
    table.attrs['settings'] = {
        'terms': term_order, 'tested': [t.name for t in terms] + (
            [individual] if bc is not None else []),
        'group_label': group_label, 'individual': individual if individual_vec is not None else '',
        'individual_mode': mode,
        'correlation': None if not np.isfinite(correlation) else float(correlation),
        'n_samples': int(n), 'groups': order}
    log.info("variance table: %d features, %d samples, terms %s%s", G, n, term_order,
             f", correlation {correlation:.3g}" if np.isfinite(correlation) else '')
    return table


def anova_wide(table: pd.DataFrame, id_columns: list[str] | None = None) -> pd.DataFrame:
    """
    One row per feature: for each term its FDR ('<term> FDR') when it was
    tested and its share of variation ('<term> %'), in the term order of
    ``attrs['settings']``. ``id_columns`` (default ['feature']) are kept
    from the first row of each feature.
    """
    id_columns = list(id_columns or ['feature'])
    settings = table.attrs.get('settings') or {}
    order = settings.get('terms') or list(dict.fromkeys(table['term']))
    first = table.drop_duplicates('feature').set_index('feature')
    wide = pd.DataFrame(index=pd.Index(first.index, name='feature'))
    for col in id_columns:
        if col != 'feature' and col in first.columns:
            wide[col] = first[col]
    for name in order:
        part = table[table['term'] == name].set_index('feature')
        if part['test'].astype(str).str.len().gt(0).any():
            wide[f"{name} FDR"] = part['adj.P.Val']
        wide[f"{name} %"] = part['variance.pct']
    return wide.reset_index()


def display_rows(table: pd.DataFrame, id_columns: list[str] | None = None) -> tuple[list, list]:
    """
    (headers, rows) for showing anova_wide in a table: the id columns, then
    per term its FDR (3 significant figures) and share of variation (%,
    one decimal). Rows are ordered by the FDR of the first tested term
    (the group).
    """
    wide = anova_wide(table, id_columns)
    fdr_cols = [c for c in wide.columns if c.endswith(' FDR')]
    if fdr_cols:
        wide = wide.sort_values(fdr_cols[0], kind='stable', na_position='last')
    ids = [c for c in (id_columns or ['feature']) if c in wide.columns]
    headers = ids + [c for c in wide.columns if c not in ids]

    def _fmt(col, value):
        if value is None or pd.isna(value):
            return ''
        if col.endswith(' FDR'):
            return f"{value:.3g}"
        if col.endswith(' %'):
            return f"{value:.1f}"
        return str(value)

    rows = [{c: _fmt(c, r[c]) for c in headers} for _i, r in wide.iterrows()]
    return headers, rows
