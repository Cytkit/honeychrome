"""
differential_stats.py — differential testing between sample groups
===================================================================

Shared engine for group comparisons of per-sample features: cluster or
gated-population frequencies, raw counts, and per-population marker
intensities. Used by the Honeycluster and Automated Gating plugins.

* ``run_moderated`` — moderated linear model per feature (``moderated_stats``:
  least squares, empirical Bayes variance moderation after Smyth 2004).
* ``run_glm_counts`` — negative-binomial GLM per feature on raw counts, with
  a log library-size offset. The offset is the sample's total by default, or
  a per-feature denominator (e.g. each population's parent count). Two
  tests: ``'wald'`` (per-feature Cameron–Trivedi dispersion, Wald z) and
  ``'ql'`` (quasi-likelihood F test after Lund et al. 2012 and edgeR's
  glmQLFit: negative-binomial dispersion shrunk towards a common prior,
  then a per-feature quasi-dispersion moderated by empirical Bayes).
* ``library_sizes`` — effective library sizes for compositional data:
  the totals, a robust (trimmed) rescaling that a few dominant features
  cannot move, or the totals of chosen control features.
* ``apply_significance`` — TREAT p-values (McCarthy & Smyth 2009),
  Benjamini–Hochberg FDR pooled over every comparison or within each, and
  optional two-stage testing of markers within clusters (Benjamini &
  Bogomolov 2014).

Every test builds one design matrix: an intercept, one indicator per
non-baseline group, optional pairing (blocking) levels, and optional
adjustment covariates — columns the caller marks numeric (age, CRP) enter
as centred continuous terms, all others as treatment-coded factors.

Frequency scales:

* ``log2_frequencies`` — log2 percentage of a total; differences are log2
  fold changes. Suited to rare clusters.
* ``logit_proportions`` — log2 odds of a population within its parent, with
  half an event added to both the population and its complement. Suited to
  gated populations, whose fractions often sit near 0 or 1.

Results carry the estimate's standard error and degrees of freedom, so a
threshold change re-derives p-values and significance without refitting.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from honeychrome.controller_components import moderated_stats as ms

log = logging.getLogger(__name__)

__all__ = [
    'build_contrasts', 'covariate_kind', 'missing_covariate_values',
    'covariate_frame', 'build_design', 'log2_frequencies',
    'logit_proportions', 'sparse_features', 'run_moderated', 'run_glm_counts',
    'apply_significance', 'suggest_groupings_from_names', 'library_sizes',
    'COUNT_TESTS', 'NORMALISATIONS',
]

#: Counts tests offered by run_glm_counts(method=).
COUNT_TESTS = ('ql', 'wald')
#: Library-size methods offered by library_sizes(method=).
NORMALISATIONS = ('total', 'robust', 'control')


# ---------------------------------------------------------------------------
# Groupings suggested from sample names
# ---------------------------------------------------------------------------

def suggest_groupings_from_names(
    sample_rel_paths: list[str],
    display_names: dict[str, str] | None = None,
) -> dict:
    """
    Scan sample names for repeated, delimiter-separated tokens that could
    serve as a Group or pairing/covariate column -- e.g. 'Mouse1_Spleen.fcs' / 'Mouse1_LN.fcs' / 'Mouse2_Spleen.fcs'
    suggests a 2-level Spleen/LN grouping field and a per-mouse pairing
    field, since 'Mouse1' recurs across two different tissue values.

    Tokenizes both the raw filename (Path(rel_path).name) and, when
    given, the experiment's display name for that sample (already
    FCS-keyword-derived when Settings -> Sample name source is
    'tubename'/'fil' -- see experiment_model.py) -- so no separate FCS
    keyword read is needed here.

    Names are compared position by position over their leading fields:
    the longest field count that at least 90% of the names reach (names
    often end in a varying folder or panel suffix), or the most common
    field count when that covers more names. Names with fewer fields are
    reported under 'irregular_samples' and excluded from every suggestion.

    The group is the field with the fewest distinct values (2-8). A field is
    a pairing variable when each of its values occurs at most once per group
    value and recurs across group values (one sample per mouse per tissue);
    other repeated fields are covariates.

    Returns:
        {
            'suggestions': [
                {
                    'field_name':  str,   -- e.g. 'Name field 2 of 3 (filename)'
                    'source':      'filename' | 'display_name',
                    'position':    int,
                    'values':      dict[sample_rel_path, str],
                    'n_distinct':  int,
                    'examples':    list[str],   -- up to 5 distinct values
                    'role_guess':  'group' | 'pairing' | 'covariate',
                },
                ...
            ],
            'irregular_samples': list[str],  -- rel paths excluded from all suggestions
        }
    Returns {'suggestions': [], 'irregular_samples': [...]} if no usable
    pattern is found (e.g. fewer than 2 samples, or every position is
    either constant or fully unique).
    """
    import re
    from collections import Counter

    def _tokenize(name: str) -> list[str]:
        stem = re.sub(r'\.fcs$', '', Path(name).name, flags=re.IGNORECASE)
        return [t for t in re.split(r'[^A-Za-z0-9]+', stem) if t]

    sources: list[tuple[str, dict[str, list[str]]]] = []

    filename_tokens = {sp: _tokenize(Path(sp).name) for sp in sample_rel_paths}
    sources.append(('filename', filename_tokens))

    if display_names:
        display_tokens = {
            sp: _tokenize(display_names[sp])
            for sp in sample_rel_paths if sp in display_names
        }
        # Only worth treating as a separate source if it actually differs
        # from the filename tokenization for at least one sample --
        # otherwise it's the same information twice (plain-filename mode).
        if any(display_tokens.get(sp) != filename_tokens.get(sp) for sp in display_tokens):
            sources.append(('display_name', display_tokens))

    all_suggestions = []
    irregular_all: set[str] = set()

    for source_label, tokens_by_sample in sources:
        if len(tokens_by_sample) < 2:
            continue
        lengths = Counter(len(v) for v in tokens_by_sample.values())
        modal_length, modal_count = lengths.most_common(1)[0]
        n_names = len(tokens_by_sample)
        leading = max((n for n in lengths
                       if sum(c for m, c in lengths.items() if m >= n) >= 0.9 * n_names), default=0)
        n_leading = sum(c for m, c in lengths.items() if m >= leading)
        if leading and n_leading > modal_count:
            field_count = leading
            regular = {sp: toks[:leading] for sp, toks in tokens_by_sample.items() if len(toks) >= leading}
        else:
            field_count = modal_length
            regular = {sp: toks for sp, toks in tokens_by_sample.items() if len(toks) == modal_length}
        irregular = set(tokens_by_sample) - set(regular)
        irregular_all |= irregular
        n_samples = len(regular)
        if n_samples < 2 or field_count == 0:
            continue

        candidates = []
        for pos in range(field_count):
            values = {sp: toks[pos] for sp, toks in regular.items()}
            n_distinct = len(set(values.values()))
            if n_distinct <= 1 or n_distinct >= n_samples:
                continue  # constant, or no repeats at all -- not usable
            candidates.append((pos, values, n_distinct))

        if not candidates:
            continue

        # Primary "group" candidate: smallest distinct count within a
        # sane range for a comparison group (2-8 levels).
        group_candidates = [c for c in candidates if 2 <= c[2] <= 8]
        primary = min(group_candidates, key=lambda c: c[2]) if group_candidates else None

        for pos, values, n_distinct in sorted(candidates, key=lambda c: c[2]):
            if primary is not None and pos == primary[0]:
                role = 'group'
            elif primary is not None:
                # Against the group: a pairing value occurs at most once per
                # group value (one sample per mouse per tissue) and recurs
                # across group values; a value repeated within a group value
                # (a cohort or batch) or nested in one group value is a
                # covariate.
                _, group_values, _ = primary
                pairs = Counter((v, group_values[sp]) for sp, v in values.items())
                co_occurrence: dict[str, set[str]] = {}
                for sp, v in values.items():
                    co_occurrence.setdefault(v, set()).add(group_values[sp])
                crosses = sum(1 for s in co_occurrence.values() if len(s) >= 2)
                once = max(pairs.values()) == 1
                role = 'pairing' if once and crosses > len(co_occurrence) / 2 else 'covariate'
            else:
                role = 'covariate'

            distinct_vals = sorted(set(values.values()))
            all_suggestions.append({
                'field_name': f"Name field {pos + 1} of {field_count} ({source_label})",
                'source': source_label,
                'position': pos,
                'values': values,
                'n_distinct': n_distinct,
                'examples': distinct_vals[:5],
                'role_guess': role,
            })

    # Group suggestions first, then pairing, then covariate; smallest
    # n_distinct first within each role.
    role_order = {'group': 0, 'pairing': 1, 'covariate': 2}
    all_suggestions.sort(key=lambda s: (role_order[s['role_guess']], s['n_distinct']))

    return {'suggestions': all_suggestions, 'irregular_samples': sorted(irregular_all)}


# ---------------------------------------------------------------------------
# Contrasts
# ---------------------------------------------------------------------------

def build_contrasts(group_names: list[str], mode: str,
                    reference: str | None) -> list[tuple[str, str]]:
    """
    Returns the list of (baseline, other) pairs to test.

    mode='reference': (reference, other) for every OTHER qualifying group —
        one joint fit, multiple coefficients.
    mode='pairwise':  every unique pair among group_names, in group_names
        order — each pair gets its own independent fit.
    """
    if mode == 'reference':
        if reference not in group_names:
            raise RuntimeError(
                f"Reference group {reference!r} is not one of the groups "
                f"qualifying for this run ({group_names!r})."
            )
        return [(reference, g) for g in group_names if g != reference]
    elif mode == 'pairwise':
        pairs = []
        for i, g1 in enumerate(group_names):
            for g2 in group_names[i + 1:]:
                pairs.append((g1, g2))
        return pairs
    else:
        raise ValueError(f"contrast_mode must be 'reference' or 'pairwise', got {mode!r}")


# ---------------------------------------------------------------------------
# Design matrix and covariates
# ---------------------------------------------------------------------------

def _is_numeric_column(values) -> bool:
    """True when every non-blank value parses as a finite float."""
    seen = False
    for v in values:
        text = str(v).strip()
        if not text:
            continue
        try:
            if not np.isfinite(float(text)):
                return False
        except ValueError:
            return False
        seen = True
    return seen


def covariate_kind(values) -> str:
    """'numeric' when every filled value is a number (so the column can be
    used as a continuous covariate), else 'categorical'."""
    return 'numeric' if _is_numeric_column(values) else 'categorical'


def missing_covariate_values(covariates: pd.DataFrame | None, sample_ids: list[str],
                             names: list[str]) -> dict[str, list[str]]:
    """
    {covariate name: [sample ids with no value]} for each of ``names`` that
    is missing (or blank) for at least one sample in ``sample_ids``.
    ``covariates`` is indexed by sample id, one column per covariate.
    """
    missing: dict[str, list[str]] = {}
    cov = covariates
    for name in names:
        bad = [
            sid for sid in sample_ids
            if cov is None or name not in cov.columns or sid not in cov.index
            or not str(cov.loc[sid, name]).strip()
        ]
        if bad:
            missing[name] = bad
    return missing


def covariate_frame(covariates: pd.DataFrame | None, sample_ids: list[str],
                    names: list[str],
                    numeric: Iterable[str] | None = None) -> pd.DataFrame | None:
    """
    Adjustment covariates for ``sample_ids`` (rows in that order). Columns
    named in ``numeric`` are returned as floats and enter the design as
    continuous terms; every other column is returned as strings and enters
    as a factor. With ``numeric=None`` a column is numeric when all of its
    values are numbers.

    Returns None when ``names`` is empty. Raises RuntimeError if any sample
    lacks a value, or a column named in ``numeric`` holds a value that is
    not a number.
    """
    names = [n for n in names if n]
    if not names:
        return None
    missing = missing_covariate_values(covariates, sample_ids, names)
    if missing:
        detail = "; ".join(
            f"'{name}': " + ", ".join(Path(r).stem for r in rels[:5])
            + (" …" if len(rels) > 5 else "")
            for name, rels in missing.items()
        )
        raise RuntimeError(f"Adjustment covariates have missing values — {detail}")
    columns = {name: [str(covariates.loc[sid, name]).strip() for sid in sample_ids]
               for name in names}
    if numeric is None:
        as_numeric = {n for n in names if covariate_kind(columns[n]) == 'numeric'}
    else:
        as_numeric = set(names) & set(numeric)
    for name in sorted(as_numeric):
        bad = [sid for sid, v in zip(sample_ids, columns[name]) if not _is_numeric_column([v])]
        if bad:
            raise RuntimeError(
                f"Covariate '{name}' is set to numeric but has values that are not "
                "numbers for: " + ", ".join(Path(r).stem for r in bad[:5])
                + (" …" if len(bad) > 5 else "")
                + ". Untick 'Numeric' for this covariate or correct the values.")
        columns[name] = [float(v) for v in columns[name]]
    return pd.DataFrame(columns, index=list(sample_ids))


def build_design(group_vec: list[str], baseline: str,
                 pairing_vec: list[str] | None = None,
                 covariates: pd.DataFrame | None = None):
    """
    Treatment-coded design matrix.

    Columns, in order: intercept; one indicator per non-baseline group (in
    order of first appearance); one indicator per pairing level except the
    first in sorted order; then each covariate — numeric columns centred,
    categorical columns as indicators for every level except the first in
    sorted order. A column is numeric when it has a numeric dtype (see
    covariate_frame()); string columns are always categorical. Covariates
    that are constant over these samples are dropped.

    Returns (X, column_names, group_columns) where group_columns maps each
    non-baseline group to its column index. Raises RuntimeError naming the
    term that makes the design rank-deficient.
    """
    n = len(group_vec)
    cols = [np.ones(n)]
    names = ['(Intercept)']
    group_cols: dict[str, int] = {}
    for g in dict.fromkeys(group_vec):
        if g == baseline:
            continue
        group_cols[g] = len(cols)
        cols.append(np.array([1.0 if x == g else 0.0 for x in group_vec]))
        names.append(f"group[{g}]")
    if baseline not in group_vec:
        raise RuntimeError(f"Baseline group {baseline!r} has no samples in this comparison.")

    def _rank_ok() -> bool:
        return np.linalg.matrix_rank(np.column_stack(cols)) == len(cols)

    if pairing_vec is not None:
        levels = sorted(set(pairing_vec))
        for lvl in levels[1:]:
            cols.append(np.array([1.0 if x == lvl else 0.0 for x in pairing_vec]))
            names.append(f"pair[{lvl}]")
        if not _rank_ok():
            raise RuntimeError(
                "The design is rank-deficient with the pairing variable — check "
                "for missing values, or pairing levels confounded with group."
            )

    if covariates is not None:
        for name in covariates.columns:
            values = covariates[name].tolist()
            if len(set(values)) < 2:
                log.info("covariate %r is constant over these samples — dropped", name)
                continue
            added = []
            if pd.api.types.is_numeric_dtype(covariates[name]):
                x = np.array([float(v) for v in values])
                added.append((x - x.mean(), f"{name}"))
            else:
                for lvl in sorted(set(values))[1:]:
                    added.append((np.array([1.0 if v == lvl else 0.0 for v in values]),
                                  f"{name}[{lvl}]"))
            for col, label in added:
                cols.append(col)
                names.append(label)
            if not _rank_ok():
                raise RuntimeError(
                    f"Covariate '{name}' is confounded with group or pairing in this "
                    "comparison (e.g. every sample of one group shares one level). "
                    "Remove it from the adjustment list."
                )

    X = np.column_stack(cols)
    if not _rank_ok():
        raise RuntimeError("The design matrix is rank-deficient.")
    return X, names, group_cols


def _subset_rows(values, mask):
    return None if values is None else [v for v, m in zip(values, mask) if m]


# ---------------------------------------------------------------------------
# Frequency scales
# ---------------------------------------------------------------------------

def log2_frequencies(counts_df: pd.DataFrame, totals) -> pd.DataFrame:
    """
    log2 of each feature's percentage of the sample's events, with half an
    event added to every count (and one to every total) so empty features
    stay finite: log2((count + 0.5) / (total + 1) · 100). Differences on
    this scale are log2 fold changes of frequency.
    """
    totals = np.asarray(totals, dtype=float)[:, None]
    vals = np.log2((counts_df.values.astype(float) + 0.5) / (totals + 1.0) * 100.0)
    return pd.DataFrame(vals, index=counts_df.index, columns=counts_df.columns)


def logit_proportions(counts_df: pd.DataFrame, denominators_df: pd.DataFrame) -> pd.DataFrame:
    """
    Empirical log2 odds of each feature within its own denominator:
    log2((count + 0.5) / (denominator − count + 0.5)).

    ``denominators_df`` has the same shape, index and columns as
    ``counts_df`` (e.g. each population's parent count per sample). Adding
    half an event to both the population and its complement keeps 0% and
    100% finite. Differences on this scale are log2 odds ratios.

    A zero count maps to -log2(2 · denominator + 1), a function of the
    denominator alone. Where the groups' denominators differ, features that
    are empty in (nearly) every sample therefore show a spurious group
    difference; exclude them with sparse_features() before testing.
    """
    counts = counts_df.values.astype(float)
    denom = denominators_df.reindex(index=counts_df.index,
                                    columns=counts_df.columns).values.astype(float)
    if np.any(counts > denom):
        raise ValueError("A population count exceeds its denominator.")
    vals = np.log2((counts + 0.5) / (denom - counts + 0.5))
    return pd.DataFrame(vals, index=counts_df.index, columns=counts_df.columns)


def sparse_features(counts_df: pd.DataFrame, min_samples: int) -> list[str]:
    """
    Features whose count is above zero in fewer than ``min_samples`` samples,
    in column order. NaN counts (samples where the feature was not measured)
    are not above zero.

    With so few observations the fitted difference between groups reflects
    the pseudo-count and the denominators rather than the biology, and the
    features distort the variance prior shared by the rest. Drop them
    before run_moderated() or run_glm_counts().
    """
    nonzero = (counts_df.fillna(0.0) > 0).sum(axis=0)
    return [f for f in counts_df.columns if nonzero[f] < min_samples]


# ---------------------------------------------------------------------------
# Library sizes for compositional counts
# ---------------------------------------------------------------------------

ROBUST_TRIM = 0.2
_ROBUST_ITERATIONS = 4


def _geometric_unit(factors: np.ndarray) -> np.ndarray:
    """Factors rescaled to geometric mean 1; non-finite or non-positive → 1."""
    f = np.where(np.isfinite(factors) & (factors > 0), factors, 1.0)
    return f / np.exp(np.mean(np.log(f)))


def _robust_factors(counts: np.ndarray, totals: np.ndarray, trim: float) -> np.ndarray:
    """
    Trimmed ratio-of-sums scaling factor per sample. The reference is the
    per-feature median of the samples' scaled proportions. Per sample, the
    features whose log ratio to the reference lies in the [trim, 1 − trim]
    quantiles give the factor as (their observed events) / (their expected
    events). Zero counts take part, so small samples are not inflated (as a
    median of ratios over positive counts is). NaN counts are left out.
    """
    f = np.ones(counts.shape[0])
    for _ in range(_ROBUST_ITERATIONS):
        with np.errstate(divide='ignore', invalid='ignore'):
            scaled = counts / (totals * f)[:, None]
        ref = np.nanmedian(np.where(np.isfinite(scaled), scaled, np.nan), axis=0)
        ref = np.where(np.isfinite(ref), ref, 0.0)
        if ref.sum() <= 0:
            return np.ones(counts.shape[0])
        ref = ref / ref.sum()
        new = np.ones(counts.shape[0])
        for s in range(counts.shape[0]):
            use = (ref > 0) & np.isfinite(counts[s])
            if use.sum() < 3 or totals[s] <= 0:
                continue
            expected = totals[s] * f[s] * ref[use]
            ratio = np.log((counts[s, use] + 0.5) / (expected + 0.5))
            lo, hi = np.quantile(ratio, [trim, 1.0 - trim])
            keep = (ratio >= lo) & (ratio <= hi)
            denom = totals[s] * ref[use][keep].sum()
            if denom > 0 and counts[s, use][keep].sum() > 0:
                new[s] = counts[s, use][keep].sum() / denom
        f = _geometric_unit(new)
    return f


def library_sizes(counts_df: pd.DataFrame, totals, method: str = 'total',
                  control: Iterable[str] | None = None,
                  trim: float = ROBUST_TRIM) -> np.ndarray:
    """
    Effective library size per sample (rows of counts_df), for the offset of
    run_glm_counts() or the totals of log2_frequencies().

    method:
      'total'   — the totals as given.
      'robust'  — totals × a trimmed ratio-of-sums factor (geometric mean 1).
                  One feature that dominates a sample (an expanded clone)
                  changes the total but not the factor.
      'control' — proportional to the summed counts of the ``control``
                  features (e.g. non-targeting barcodes), rescaled so the
                  factors relative to the totals have geometric mean 1.

    NaN counts (features not measured in a sample) are left out.
    """
    totals = np.asarray(totals, dtype=float)
    if method == 'total':
        return totals.copy()
    counts = counts_df.values.astype(float)
    if method == 'robust':
        return totals * _robust_factors(counts, totals, trim)
    if method == 'control':
        cols = [c for c in (control or []) if c in counts_df.columns]
        if not cols:
            raise ValueError("Control normalisation needs at least one control feature.")
        ctrl = np.nansum(counts_df[cols].values.astype(float), axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            factors = (ctrl + 0.5) / (totals + 1.0)
        return totals * _geometric_unit(factors)
    raise ValueError(f"method must be one of {NORMALISATIONS}, got {method!r}")


# ---------------------------------------------------------------------------
# Moderated linear model
# ---------------------------------------------------------------------------

_TOO_FEW_FEATURES = (
    "Only {n} feature(s) to test (need >= 3) for {other!r} vs {base!r}."
)


def _moderated_fit_contrasts(data_df: pd.DataFrame, group_vec: list[str],
                             contrasts: list[tuple[str, str]],
                             pairing_vec: list[str] | None,
                             covariates: pd.DataFrame | None,
                             too_few_hint: str = '') -> pd.DataFrame:
    """
    One moderated fit over data_df's samples, read out for every contrast
    in ``contrasts`` (all sharing one baseline). Returns the stacked
    per-contrast tables with 'feature' and 'comparison' columns.
    """
    base = contrasts[0][0]
    X, _names, group_cols = build_design(group_vec, base, pairing_vec, covariates)
    expr = data_df.values.T.astype(float)            # (features, samples)
    n_features = expr.shape[0]
    if n_features < 3:
        msg = _TOO_FEW_FEATURES.format(n=n_features, other=contrasts[0][1], base=base)
        raise RuntimeError(f"{msg} {too_few_hint}".strip())
    try:
        fit = ms.e_bayes(ms.lm_fit(expr, X))
    except ValueError as exc:
        raise RuntimeError(
            f"{exc} ({contrasts[0][1]!r} vs {base!r}). Too few features "
            "have enough observed samples to fit."
        ) from exc
    log.info("moderated fit: %d features, %d samples, d0=%.3g, s0²=%.3g",
             n_features, X.shape[0], fit.df_prior, fit.s2_prior)

    frames = []
    for b, other in contrasts:
        tt = ms.top_table(fit, group_cols[other])
        tt.insert(0, 'feature', list(data_df.columns))
        tt['comparison'] = f"{other} vs {b}"
        frames.append(tt)
    return pd.concat(frames, ignore_index=True)


def _pair_subsets(data_df, group_vec, contrasts, pairing_vec, covariates):
    """Yield (sub_df, sub_groups, sub_pairing, sub_covariates, base, other) per pair."""
    for base, other in contrasts:
        mask = [g in (base, other) for g in group_vec]
        sub_df = data_df.loc[[sid for sid, m in zip(data_df.index, mask) if m]]
        sub_groups = _subset_rows(group_vec, mask)
        n_a, n_b = sub_groups.count(base), sub_groups.count(other)
        if n_a < 3 or n_b < 3:
            raise RuntimeError(
                f"Not enough samples for {other} vs {base}: {base}={n_a}, {other}={n_b}."
            )
        sub_cov = covariates.loc[sub_df.index] if covariates is not None else None
        yield sub_df, sub_groups, _subset_rows(pairing_vec, mask), sub_cov, base, other


def run_moderated(data_df: pd.DataFrame, group_vec: list[str],
                  contrasts: list[tuple[str, str]], mode: str,
                  pval_threshold: float, fc_threshold: float,
                  pairing_vec: list[str] | None = None,
                  fdr_scope: str = 'global',
                  covariates: pd.DataFrame | None = None,
                  use_treat: bool = True,
                  feature_meta: pd.DataFrame | None = None,
                  hierarchical: bool = False,
                  too_few_hint: str = '') -> pd.DataFrame:
    """
    Moderated linear model test of every feature, for N groups and
    multiple contrasts.

    data_df   : rows = samples (aligned to group_vec/pairing_vec/covariates),
                columns = features
    contrasts : (baseline, other) pairs from build_contrasts()
    mode      : 'reference' — one fit across all samples, one coefficient
                  read per contrast (the variance prior is shared).
                'pairwise'  — each contrast fitted on its two groups only.
    covariates: optional adjustment covariates (covariate_frame()).
    feature_meta: optional columns merged onto the results by 'feature'
                (e.g. cluster, cluster_id, channel for marker intensities).
    use_treat, hierarchical, fdr_scope: see apply_significance().
    too_few_hint: appended to the error raised when fewer than three
                features can be tested.

    Returns one row per (feature, comparison): feature, logFC, CI.L, CI.R,
    AveExpr, t, P.Value, adj.P.Val, SE, df.total, comparison,
    adj.P.Val.global, threshold, significant (plus the cluster-level
    columns when hierarchical).
    """
    if mode == 'reference':
        baseline = contrasts[0][0]
        if any(b != baseline for b, _o in contrasts):
            raise ValueError("reference mode expects a shared baseline")
        combined = _moderated_fit_contrasts(data_df, group_vec, contrasts, pairing_vec,
                                            covariates, too_few_hint)
    elif mode == 'pairwise':
        frames = [
            _moderated_fit_contrasts(sub_df, sub_groups, [(base, other)], sub_pair, sub_cov,
                                     too_few_hint)
            for sub_df, sub_groups, sub_pair, sub_cov, base, other
            in _pair_subsets(data_df, group_vec, contrasts, pairing_vec, covariates)
        ]
        combined = pd.concat(frames, ignore_index=True)
    else:
        raise ValueError(f"mode must be 'reference' or 'pairwise', got {mode!r}")

    if feature_meta is not None:
        combined = combined.merge(feature_meta, on='feature', how='left', sort=False)

    combined = apply_significance(combined, pval_threshold, fc_threshold,
                                  fdr_scope=fdr_scope, use_treat=use_treat,
                                  hierarchical=hierarchical)
    log.info("moderated test: %d rows across %d comparison(s), %d significant (%s FDR%s%s)",
             len(combined), combined['comparison'].nunique(),
             int(combined['significant'].sum()), fdr_scope,
             ", TREAT" if use_treat else "", ", cluster-first" if hierarchical else "")
    return combined


# ---------------------------------------------------------------------------
# Negative-binomial GLM on counts
# ---------------------------------------------------------------------------

def _glm_fit_one_feature(y: np.ndarray, design: np.ndarray, offset: np.ndarray):
    """Poisson-then-NB fit for one feature's raw counts, with the NB
    dispersion from the Cameron & Trivedi auxiliary-OLS estimate.
    ``offset`` is the log of the per-sample denominator (a library-size
    term), so the fit compares rates, not raw totals, across samples.
    Returns the fitted GLM result, or None when the feature cannot be
    tested: no events in any sample (there is nothing to fit, and the
    Poisson starting values are 0/0), or the Poisson fit fails."""
    import statsmodels.api as sm

    if not np.any(y):
        log.debug("GLM skipped: feature has no events in any sample")
        return None
    try:
        poisson_fit = sm.GLM(y, design, family=sm.families.Poisson(), offset=offset).fit()
    except Exception as e:
        log.warning("Poisson GLM failed (%s) — marking untestable", e)
        return None
    try:
        mu = poisson_fit.mu
        aux_y = ((y - mu) ** 2 - y) / mu
        alpha = float(sm.OLS(aux_y, mu).fit().params[0])
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("non-positive alpha estimate")
        return sm.GLM(y, design, family=sm.families.NegativeBinomial(alpha=alpha), offset=offset).fit()
    except Exception as e:
        log.warning("NB GLM failed (%s), falling back to Poisson", e)
        return poisson_fit


def _glm_fit_observed(y: np.ndarray, design: np.ndarray, offset: np.ndarray):
    """Fit one feature on the samples where it was measured (finite count
    and offset), on the full-rank basis of ``moderated_stats.estimable_columns``
    (columns all zero over those samples, e.g. a group with no measured
    sample, and columns aliased with earlier ones are left out). Returns
    (fit, columns of *design* the fit's parameters belong to, −1 for a
    parameter whose coefficient is not estimable), or (None, None) when
    the feature cannot be tested: too few measured samples for the basis,
    or no estimable coefficient."""
    observed = np.isfinite(y) & np.isfinite(offset)
    columns, estimable = ms.estimable_columns(design[observed])
    if observed.sum() <= columns.size or not estimable.any():
        return None, None
    sub_design = design[np.ix_(observed, columns)]
    fit = _glm_fit_one_feature(y[observed], sub_design, offset[observed])
    return (fit, np.where(estimable, columns, -1)) if fit is not None else (None, None)


class _GLMFit(NamedTuple):
    """The parts of a statsmodels GLM result the contrasts read."""
    params: np.ndarray
    bse: np.ndarray
    tvalues: np.ndarray
    mu: np.ndarray


_FLOAT_EPS = np.finfo(float).eps


def _glm_deviance(Y: np.ndarray, mu: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    """Row-wise deviance with statsmodels' Poisson / NegativeBinomial
    arithmetic, operation for operation."""
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        endog_mu = np.clip(Y / mu, _FLOAT_EPS, np.inf)
        if alpha is None:
            resid_dev = 2 * (Y * np.log(endog_mu) - (Y - mu))
        else:
            inv_alpha = 1 / alpha[:, None]
            endog_alpha = Y + inv_alpha
            mu_alpha = mu + inv_alpha
            resid_dev = Y * np.log(endog_mu)
            resid_dev -= endog_alpha * np.log(endog_alpha / mu_alpha)
            resid_dev = 2 * resid_dev
    return np.sum(resid_dev, axis=1)


def _glm_irls(Y: np.ndarray, X: np.ndarray, offset: np.ndarray,
              alpha: np.ndarray | None, maxiter: int = 100, tol: float = 1e-8) -> list:
    """
    statsmodels' GLM IRLS (GLM._fit_irls; log link, Poisson or negative
    binomial with a per-row alpha) for many features sharing one design.
    The element-wise steps run on every row at once; each row's weighted
    least squares (np.linalg.lstsq), linear predictor and final pinv fit
    run on that row alone, exactly as statsmodels runs them, so every row
    reaches statsmodels' numbers and stops at statsmodels' iteration.

    Y, offset: (features, samples); X: (samples, coefficients), full rank.
    Returns a _GLMFit per row, or None where statsmodels would raise or not
    converge within maxiter (the caller fits those rows with statsmodels).
    """
    from statsmodels.tools.tools import pinv_extended

    out: list = [None] * Y.shape[0]
    mu = (Y + Y.mean(axis=1, keepdims=True)) / 2.
    lin_pred = np.log(np.clip(mu, _FLOAT_EPS, np.inf))
    dev = _glm_deviance(Y, mu, alpha)
    rows = np.flatnonzero(~np.isnan(dev))
    weights = np.empty_like(Y)
    wlsendog = np.empty_like(Y)
    for iteration in range(maxiter):
        if rows.size == 0:
            break
        y, m = Y[rows], mu[rows]
        p = np.clip(m, _FLOAT_EPS, np.inf)
        deriv = 1. / p
        var = np.power(np.fabs(m), 1.) if alpha is None else p + alpha[rows, None] * p ** 2
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            w = 1. / (deriv ** 2 * var)
            z = lin_pred[rows] + deriv * (y - m) - offset[rows]
            w_half = np.sqrt(w)
        finite = np.all(np.isfinite(w_half), axis=1) & np.all(np.isfinite(z), axis=1)
        rows, w, z, w_half = rows[finite], w[finite], z[finite], w_half[finite]
        new_lin = np.empty_like(z)
        for j in range(rows.size):
            params = np.linalg.lstsq(w_half[j][:, None] * X, w_half[j] * z[j], rcond=-1)[0]
            new_lin[j] = np.dot(X, params)
        new_lin += offset[rows]
        with np.errstate(over='ignore'):
            new_mu = np.exp(new_lin)
        new_dev = _glm_deviance(Y[rows], new_mu, None if alpha is None else alpha[rows])
        lin_pred[rows], mu[rows] = new_lin, new_mu
        weights[rows], wlsendog[rows] = w, z
        with np.errstate(invalid='ignore'):
            converged = (np.abs(dev[rows] - new_dev) <= tol) | (dev[rows] == new_dev)
        dev[rows] = new_dev
        for i in rows[converged]:
            sw = np.sqrt(weights[i])
            pinv_wexog, _ = pinv_extended(sw[:, None] * X)
            params = np.dot(pinv_wexog, sw * wlsendog[i])
            bse = np.sqrt(np.diag(np.dot(pinv_wexog, np.transpose(pinv_wexog)) * 1.0))
            fitted = np.exp(np.dot(X, params) + offset[i] + 0.)
            out[i] = _GLMFit(params, bse, params / bse, fitted)
        rows = rows[~converged]
    return out


def _glm_fit_batch(Y: np.ndarray, X: np.ndarray, offset: np.ndarray) -> list:
    """
    _glm_fit_one_feature for every row of Y (features sharing one design):
    Poisson, the Cameron & Trivedi alpha from the Poisson fit, then the
    negative binomial; Poisson where alpha is not positive. Rows _glm_irls
    leaves unfitted go to _glm_fit_one_feature itself. Returns a _GLMFit,
    statsmodels result or None per row.
    """
    from statsmodels.tools.tools import pinv_extended

    poisson = _glm_irls(Y, X, offset, None)
    alpha = np.full(Y.shape[0], np.nan)
    for i, fit in enumerate(poisson):
        if fit is not None:
            aux_y = ((Y[i] - fit.mu) ** 2 - Y[i]) / fit.mu
            pinv_mu, _ = pinv_extended(fit.mu[:, None])
            alpha[i] = float(np.dot(pinv_mu, aux_y)[0])       # sm.OLS(aux_y, mu)
    use_nb = np.array([fit is not None for fit in poisson]) & np.isfinite(alpha) & (alpha > 0)
    nb_rows = np.flatnonzero(use_nb)
    nb = _glm_irls(Y[nb_rows], X, offset[nb_rows], alpha[nb_rows])
    n_poisson = sum(fit is not None for fit in poisson) - nb_rows.size
    if n_poisson:
        log.warning("NB GLM: %d feature(s) with a non-positive alpha estimate fitted as Poisson",
                    n_poisson)

    fits = list(poisson)
    for j, i in enumerate(nb_rows):
        fits[i] = nb[j]
    return [fit if fit is not None else _glm_fit_one_feature(Y[i], X, offset[i])
            for i, fit in enumerate(fits)]


def _glm_fit_features(counts: np.ndarray, design: np.ndarray, offsets: np.ndarray) -> list:
    """
    _glm_fit_observed for every feature (rows of counts and offsets), with
    the features that share a pattern of measured samples fitted together
    by _glm_fit_batch. Returns (fit, columns) or (None, None) per feature.
    """
    out = [(None, None)] * counts.shape[0]
    patterns: dict[bytes, list[int]] = {}
    for i in range(counts.shape[0]):
        observed = np.isfinite(counts[i]) & np.isfinite(offsets[i])
        if np.any(counts[i][observed]):        # no events: nothing to fit
            patterns.setdefault(observed.tobytes(), []).append(i)
    for key, rows in patterns.items():
        observed = np.frombuffer(key, dtype=bool)
        columns, estimable = ms.estimable_columns(design[observed])
        if observed.sum() <= columns.size or not estimable.any():
            continue
        sub_design = design[np.ix_(observed, columns)]
        readable = np.where(estimable, columns, -1)
        cols = np.flatnonzero(observed)
        fits = _glm_fit_batch(counts[np.ix_(rows, cols)], sub_design, offsets[np.ix_(rows, cols)])
        for i, fit in zip(rows, fits):
            if fit is not None:
                out[i] = (fit, readable)
    return out


# ---------------------------------------------------------------------------
# Quasi-likelihood F test
# ---------------------------------------------------------------------------

_LOG_ALPHA_GRID = np.linspace(np.log(1e-4), np.log(1e2), 141)
_MIN_PRIOR_VARIANCE = 0.25          # floor of the log-dispersion prior variance


class _Block(NamedTuple):
    """Features measured in the same samples, fitted on one sub-design."""
    rows: np.ndarray        # feature indices
    columns: np.ndarray     # design columns kept (−1: not estimable)
    Y: np.ndarray           # counts, features × observed samples
    O: np.ndarray           # log offsets
    X: np.ndarray           # sub-design


def _fit_rows(Y: np.ndarray, X: np.ndarray, O: np.ndarray, alpha: np.ndarray | None) -> list:
    """
    Fitted means and coefficients per row for a Poisson (alpha None) or
    negative-binomial GLM with a fixed alpha per row: the batched IRLS, with
    statsmodels for the rows it leaves unfitted. Returns (params, mu) or
    None per row.
    """
    import statsmodels.api as sm

    out = []
    for r, fit in enumerate(_glm_irls(Y, X, O, alpha)):
        if fit is not None:
            out.append((np.asarray(fit.params), np.asarray(fit.mu)))
            continue
        family = (sm.families.Poisson() if alpha is None
                  else sm.families.NegativeBinomial(alpha=float(alpha[r])))
        try:
            res = sm.GLM(Y[r], X, family=family, offset=O[r]).fit()
            out.append((np.asarray(res.params), np.asarray(res.mu)))
        except Exception as e:
            log.warning("count GLM failed (%s): feature left untested", e)
            out.append(None)
    return out


def _nb_profile_loglik(Y: np.ndarray, mu: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Cox–Reid adjusted negative-binomial log-likelihood of each row at every
    point of _LOG_ALPHA_GRID, for fixed means: features × grid. The
    adjustment, −½ log det(XᵀWX) with W = μ/(1 + αμ), accounts for the
    fitted coefficients.
    """
    from scipy import special

    out = np.empty((Y.shape[0], _LOG_ALPHA_GRID.size))
    lgy = special.gammaln(Y + 1.0)
    Xt = X.T[None, :, :]
    for g, la in enumerate(_LOG_ALPHA_GRID):
        a = np.exp(la)
        inv = 1.0 / a
        w = mu / (1.0 + a * mu)
        _sign, logdet = np.linalg.slogdet((Xt * w[:, None, :]) @ X)
        out[:, g] = np.sum(special.gammaln(Y + inv) - special.gammaln(inv) - lgy
                           + Y * np.log(a * mu / (1.0 + a * mu)) - inv * np.log1p(a * mu),
                           axis=1) - 0.5 * logdet
    return out


def _grid_argmax(values: np.ndarray) -> np.ndarray:
    """Location of each row's maximum on _LOG_ALPHA_GRID, refined by a
    parabola through the best point and its neighbours."""
    k = np.argmax(values, axis=1)
    x = _LOG_ALPHA_GRID[k].copy()
    h = _LOG_ALPHA_GRID[1] - _LOG_ALPHA_GRID[0]
    inner = (k > 0) & (k < _LOG_ALPHA_GRID.size - 1)
    i = np.flatnonzero(inner)
    y0, y1, y2 = values[i, k[i] - 1], values[i, k[i]], values[i, k[i] + 1]
    curve = y0 - 2.0 * y1 + y2
    with np.errstate(divide='ignore', invalid='ignore'):
        step = np.where(curve < 0, 0.5 * h * (y0 - y2) / curve, 0.0)
    x[i] += np.clip(np.nan_to_num(step), -h, h)
    return x


def _shrunk_dispersions(blocks: list, mus: list) -> tuple[list, dict]:
    """
    Negative-binomial dispersion per feature, shrunk towards a common value:
    the maximum of the Cox–Reid likelihood plus a normal prior on log α
    whose mean is the median of the features' log MLEs and whose variance
    is their robust (MAD) variance less the sampling variance ψ′(df/2),
    floored at 0.25 (as DESeq2). Returns one alpha array per block and the
    prior.
    """
    from scipy import special

    lls = [_nb_profile_loglik(b.Y, mu, b.X) for b, mu in zip(blocks, mus)]
    dfs = np.concatenate([np.full(b.Y.shape[0], b.X.shape[0] - b.X.shape[1]) for b in blocks])
    mle = np.concatenate([_LOG_ALPHA_GRID[np.argmax(ll, axis=1)] for ll in lls])
    usable = mle > _LOG_ALPHA_GRID[0] + 2.0          # off the Poisson boundary
    if usable.sum() < 3:
        usable = np.ones_like(usable)
    centre = float(np.median(mle[usable]))
    mad = 1.4826 * float(np.median(np.abs(mle[usable] - centre)))
    variance = max(mad ** 2 - float(special.polygamma(1, np.mean(dfs) / 2.0)), _MIN_PRIOR_VARIANCE)
    penalty = (_LOG_ALPHA_GRID[None, :] - centre) ** 2 / (2.0 * variance)
    alphas = [np.exp(_grid_argmax(ll - penalty)) for ll in lls]
    return alphas, {'log_alpha_mean': centre, 'log_alpha_variance': variance}


def _glm_ql_tests(counts: np.ndarray, design: np.ndarray, offsets: np.ndarray,
                  coefs: list[int]) -> dict:
    """
    Quasi-likelihood F test of each design column in ``coefs`` for every
    feature (rows of counts and offsets; NaN = not measured).

    Features sharing a pattern of measured samples are fitted together on
    that sub-design. Negative-binomial dispersions come from
    _shrunk_dispersions (two rounds of fit and estimate). The
    quasi-dispersion of each feature (residual deviance / residual df) is
    moderated with moderated_stats.fit_prior_variance, as limma moderates
    variances; F = (deviance without the column − deviance with it) /
    moderated quasi-dispersion, on 1 and df + d0 degrees of freedom.

    Returns {coef: (logFC (log2), SE, t, df_total)}; SE = |logFC| / √F and
    t = sign(logFC) √F, so a t test with df_total reproduces the F test and
    TREAT and threshold changes work from the stored columns.
    """
    n_features = counts.shape[0]
    patterns: dict[bytes, list[int]] = {}
    for i in range(n_features):
        observed = np.isfinite(counts[i]) & np.isfinite(offsets[i])
        if np.any(counts[i][observed]):
            patterns.setdefault(observed.tobytes(), []).append(i)

    blocks, mus = [], []
    for key, rows in patterns.items():
        observed = np.frombuffer(key, dtype=bool)
        columns, estimable = ms.estimable_columns(design[observed])
        if observed.sum() <= columns.size or not estimable.any():
            continue
        X = design[np.ix_(observed, columns)]
        columns = np.where(estimable, columns, -1)
        rows = np.asarray(rows)
        obs = np.flatnonzero(observed)
        Y, O = counts[np.ix_(rows, obs)], offsets[np.ix_(rows, obs)]
        fits = _fit_rows(Y, X, O, None)
        ok = np.array([f is not None for f in fits])
        if not ok.any():
            continue
        blocks.append(_Block(rows[ok], columns, Y[ok], O[ok], X))
        mus.append(np.array([f[1] for f in fits if f is not None]))

    out = {j: tuple(np.full(n_features, np.nan) for _ in range(4)) for j in coefs}
    if not blocks:
        return out

    full = None
    for _round in range(2):
        alphas, prior = _shrunk_dispersions(blocks, mus)
        full = [_fit_rows(b.Y, b.X, b.O, a) for b, a in zip(blocks, alphas)]
        mus = [np.array([f[1] if f is not None else m for f, m in zip(fb, mu)])
               for fb, mu in zip(full, mus)]

    dev_full = [_glm_deviance(b.Y, mu, a) for b, mu, a in zip(blocks, mus, alphas)]
    res_df = [np.full(b.Y.shape[0], float(b.X.shape[0] - b.X.shape[1])) for b in blocks]
    phi = np.concatenate([d / df for d, df in zip(dev_full, res_df)])
    df_all = np.concatenate(res_df)
    d0, s0 = ms.fit_prior_variance(phi, df_all)
    if np.isinf(d0):
        post = np.full(phi.shape, s0)
    else:
        post = (d0 * s0 + df_all * phi) / (d0 + df_all)
    log.info("QL counts test: %d features, log-dispersion prior %.3g (variance %.3g), "
             "quasi-dispersion prior d0=%.3g s0²=%.3g", phi.size, prior['log_alpha_mean'],
             prior['log_alpha_variance'], d0, s0)

    LN2 = np.log(2.0)
    for j in coefs:
        logfc, se, tvals, dft = (np.full(n_features, np.nan) for _ in range(4))
        start = 0
        for b, fb, mu, a, dev, rdf in zip(blocks, full, mus, alphas, dev_full, res_df):
            n_rows = b.Y.shape[0]
            block_post = post[start:start + n_rows]
            start += n_rows
            position = np.flatnonzero(b.columns == j)
            if position.size == 0:
                continue
            k = int(position[0])
            reduced = _fit_rows(b.Y, np.delete(b.X, k, axis=1), b.O, a)
            mu_red = np.array([f[1] if f is not None else m for f, m in zip(reduced, mu)])
            dev_red = _glm_deviance(b.Y, mu_red, a)
            with np.errstate(divide='ignore', invalid='ignore'):
                f_stat = np.maximum(dev_red - dev, 0.0) / block_post
            for r, i in enumerate(b.rows):
                if fb[r] is None or reduced[r] is None or not np.isfinite(f_stat[r]):
                    continue
                lfc = fb[r][0][k] / LN2
                root = np.sqrt(f_stat[r])
                logfc[i] = lfc
                tvals[i] = np.copysign(root, lfc)
                se[i] = abs(lfc) / root if root > 0 and lfc != 0 else np.inf
                dft[i] = rdf[r] + d0
        out[j] = (logfc, se, tvals, dft)
    return out


def _glm_counts_contrasts(counts_df: pd.DataFrame, group_vec: list[str],
                          contrasts: list[tuple[str, str]],
                          pairing_vec: list[str] | None,
                          covariates: pd.DataFrame | None,
                          offset_df: pd.DataFrame | None = None,
                          method: str = 'wald') -> pd.DataFrame:
    """
    One design + per-feature NB/Poisson fit, read out for every contrast in
    ``contrasts`` (shared baseline). The offset is log(row total of
    counts_df) unless ``offset_df`` gives a per-feature denominator. The
    coefficient is a rate difference between groups, reported on the log2
    scale. method 'wald': Wald standard errors (df.total = inf); 'ql':
    quasi-likelihood F tests (_glm_ql_tests).
    NaN counts or offsets mark samples where a feature was not measured:
    each feature is fitted on the samples that have it, and is untestable
    (NaN result) when a compared group has none.
    """
    if method not in COUNT_TESTS:
        raise ValueError(f"method must be one of {COUNT_TESTS}, got {method!r}")
    base = contrasts[0][0]
    design_mat, _names, group_cols = build_design(group_vec, base, pairing_vec, covariates)

    features = list(counts_df.columns)
    if offset_df is None:
        totals = counts_df.sum(axis=1).values.astype(float)
        row_offset = np.log(np.maximum(totals, 1.0))
        offsets = {f: row_offset for f in features}
    else:
        denom = offset_df.reindex(index=counts_df.index, columns=counts_df.columns)
        offsets = {f: np.log(np.maximum(denom[f].values.astype(float), 1.0)) for f in features}

    LN2 = np.log(2.0)
    Z95 = 1.959963984540054
    counts = counts_df[features].values.T.astype(float)
    log_offsets = (np.array([offsets[f] for f in features], dtype=float)
                   .reshape(len(features), len(counts_df)))

    if method == 'ql':
        tests = _glm_ql_tests(counts, design_mat, log_offsets,
                              [group_cols[other] for _b, other in contrasts])
        frames = []
        for b, other in contrasts:
            logfc, se, tvals, dft = tests[group_cols[other]]
            p = ms.treat_pvalues(logfc, se, dft, 0.0)
            with np.errstate(invalid='ignore'):
                q = ms._t_isf(np.full(logfc.shape, 0.025), np.where(np.isfinite(dft), dft, np.inf))
            frames.append(pd.DataFrame({
                'feature':   features,
                'logFC':     logfc,
                'CI.L':      logfc - q * se,
                'CI.R':      logfc + q * se,
                't':         tvals,
                'P.Value':   p,
                'adj.P.Val': ms.bh_adjust(p),
                'SE':        se,
                'df.total':  dft,
                'comparison': f"{other} vs {b}",
            }))
        return pd.concat(frames, ignore_index=True)

    fits = _glm_fit_features(counts, design_mat, log_offsets)

    frames = []
    for b, other in contrasts:
        j = group_cols[other]
        logfc = np.full(len(features), np.nan)
        se = np.full(len(features), np.nan)
        tvals = np.full(len(features), np.nan)
        for i, (fit, columns) in enumerate(fits):
            if fit is None:
                continue
            position = np.flatnonzero(columns == j)
            if position.size == 0:
                continue
            k = int(position[0])
            logfc[i] = fit.params[k] / LN2
            se[i] = fit.bse[k] / LN2
            tvals[i] = fit.tvalues[k]
        p = ms.treat_pvalues(logfc, se, np.inf, 0.0)
        frames.append(pd.DataFrame({
            'feature':   features,
            'logFC':     logfc,
            'CI.L':      logfc - Z95 * se,
            'CI.R':      logfc + Z95 * se,
            't':         tvals,
            'P.Value':   p,
            'adj.P.Val': ms.bh_adjust(p),
            'SE':        se,
            'df.total':  np.inf,
            'comparison': f"{other} vs {b}",
        }))
    return pd.concat(frames, ignore_index=True)


def run_glm_counts(counts_df: pd.DataFrame, group_vec: list[str],
                   contrasts: list[tuple[str, str]], mode: str,
                   pval_threshold: float, fc_threshold: float,
                   pairing_vec: list[str] | None = None,
                   fdr_scope: str = 'global',
                   covariates: pd.DataFrame | None = None,
                   use_treat: bool = True,
                   offset_df: pd.DataFrame | None = None,
                   method: str = 'wald') -> pd.DataFrame:
    """
    Per-feature negative-binomial GLM differential abundance test on raw
    event counts, with the same contrasts/mode/pairing/covariate semantics
    as run_moderated(). In 'reference' mode each feature is fitted once and
    every contrast is read from that fit; 'pairwise' fits each pair
    separately.

    offset_df: optional per-sample, per-feature denominators (same index and
        columns as counts_df), e.g. each gated population's parent count.
        Default: the sample's total over all features.
    method: 'wald' — per-feature Cameron–Trivedi dispersion (Poisson where
        it is not positive) and Wald z. Liberal with few residual degrees of
        freedom (paired designs, few samples per group). 'ql' —
        quasi-likelihood F test with shrunk dispersions (_glm_ql_tests).

    Returns the same schema as run_moderated() minus AveExpr.
    """
    if mode == 'reference':
        combined = _glm_counts_contrasts(counts_df, group_vec, contrasts, pairing_vec,
                                         covariates, offset_df, method)
    elif mode == 'pairwise':
        frames = [
            _glm_counts_contrasts(sub_df, sub_groups, [(base, other)], sub_pair, sub_cov,
                                  None if offset_df is None else offset_df.loc[sub_df.index],
                                  method)
            for sub_df, sub_groups, sub_pair, sub_cov, base, other
            in _pair_subsets(counts_df, group_vec, contrasts, pairing_vec, covariates)
        ]
        combined = pd.concat(frames, ignore_index=True)
    else:
        raise ValueError(f"mode must be 'reference' or 'pairwise', got {mode!r}")

    combined = apply_significance(combined, pval_threshold, fc_threshold,
                                  fdr_scope=fdr_scope, use_treat=use_treat)
    log.info("GLM counts (%s): %d rows across %d comparison(s), %d significant (%s FDR), "
             "%d untestable", method, len(combined), combined['comparison'].nunique(),
             int(combined['significant'].sum()), fdr_scope, int(combined['logFC'].isna().sum()))
    return combined


# ---------------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------------

def apply_significance(results: pd.DataFrame, pval_threshold: float,
                       fc_threshold: float, fdr_scope: str = 'global',
                       use_treat: bool = True,
                       hierarchical: bool = False) -> pd.DataFrame:
    """
    Recompute p-values, FDR and the 'significant' flag of a results table
    for the given thresholds, without refitting. Returns a new DataFrame.

    use_treat   : True — P.Value is the TREAT p-value for |logFC| >
                  fc_threshold (McCarthy & Smyth 2009). False — P.Value
                  tests logFC ≠ 0 and fc_threshold only filters estimates.
    fdr_scope   : 'global' — BH pooled over every comparison in the table
                  decides significance; 'per_comparison' — each
                  comparison's own BH. Both adj.P.Val (per comparison) and
                  adj.P.Val.global are always written.
    hierarchical: family-first testing for tables with a 'cluster_id'
                  column (marker intensities). Adds cluster.P.Value (Simes
                  over the family's markers), cluster.adj.P.Val (BH over
                  families) and stagewise.adj.P.Val; a marker is
                  significant when its stage-wise value ≤ pval_threshold.

    Tables without SE/df.total columns (results saved before these columns
    existed) keep their stored p-values and use fc_threshold as a filter.
    """
    df = results.copy()
    tau = float(fc_threshold) if use_treat else 0.0
    can_retest = {'SE', 'df.total', 'logFC'}.issubset(df.columns)
    if can_retest:
        df['P.Value'] = ms.treat_pvalues(df['logFC'].values, df['SE'].values,
                                         df['df.total'].values.astype(float), tau)
        df['threshold'] = tau
        adj = np.full(len(df), np.nan)
        for _comp, idx in df.groupby('comparison', sort=False).indices.items():
            adj[idx] = ms.bh_adjust(df['P.Value'].values[idx])
        df['adj.P.Val'] = adj
    df['adj.P.Val.global'] = ms.bh_adjust(df['P.Value'].values.astype(float))

    q_col = 'adj.P.Val.global' if fdr_scope == 'global' else 'adj.P.Val'
    passes_fc = df['logFC'].abs() >= fc_threshold

    if hierarchical and 'cluster_id' in df.columns:
        fam_p = np.full(len(df), np.nan)
        fam_adj = np.full(len(df), np.nan)
        stage = np.full(len(df), np.nan)
        groups = ({'all': np.arange(len(df))} if fdr_scope == 'global'
                  else df.groupby('comparison', sort=False).indices)
        for _key, idx in groups.items():
            family = [f"{c}\x1f{k}" for c, k in
                      zip(df['comparison'].values[idx], df['cluster_id'].values[idx])]
            res = ms.stagewise_adjust(df['P.Value'].values[idx].astype(float), family,
                                      pval_threshold)
            fam_p[idx] = res['family_p']
            fam_adj[idx] = res['family_adj']
            stage[idx] = res['adjusted']
        df['cluster.P.Value'] = fam_p
        df['cluster.adj.P.Val'] = fam_adj
        df['stagewise.adj.P.Val'] = stage
        df['significant'] = (df['stagewise.adj.P.Val'] <= pval_threshold) & passes_fc
    else:
        for col in ('cluster.P.Value', 'cluster.adj.P.Val', 'stagewise.adj.P.Val'):
            if col in df.columns:
                df = df.drop(columns=[col])
        df['significant'] = (df[q_col] <= pval_threshold) & passes_fc
    return df
