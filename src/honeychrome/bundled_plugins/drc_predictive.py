"""
drc_predictive.py — Predictive modelling of group membership from cluster features
==================================================================================
Companion to ``dr_clustering_tab.py`` (filename intentionally NOT ``*_tab.py``).

Answers "which small set of cluster features best separates two groups,
and how well?", complementing the per-feature differential tests in
``drc_stats.py``. The design follows the nested cross-validation approach
used in systems-immunology studies of flow cytometry cohorts (Humblet-Baron
et al. 2025, Alzheimer's Dement 21:e70952; Gemander et al. 2025, npj
Vaccines 10:140; Veiga et al. 2026, Nat Commun 17:4670):

* Models: L1-penalised (lasso) logistic regression and random forest, each
  with class weights balanced between groups. Hyperparameters are chosen
  on the training part of each outer split only: the L1 strength C by an
  inner stratified cross-validation, the forest's maximum depth by its
  out-of-bag AUC.
* Performance: repeated stratified K-fold outer cross-validation. Each
  sample is predicted once per repeat by a model that never saw it; ROC
  AUC and balanced accuracy are computed on these out-of-fold predictions
  and summarised as mean ± SD over repeats.
* Importance: how often the lasso keeps each feature across outer fits
  (selection frequency, with its mean standardised coefficient), and the
  forest's mean impurity importance.
* Panel size: AUC of an L2 logistic model using only the top-N features,
  ranked by |Welch t| within each training split, for increasing N. The
  smallest N reaching 95% of the best mean AUC is reported as the minimal
  panel.

All feature ranking and scaling happens inside the training splits, so
held-out samples never influence the model that predicts them.

Sample size: at least ``MIN_PER_GROUP`` samples in each group are required;
below ``RECOMMENDED_PER_GROUP`` the estimates are noisy and the result
carries a warning. Correlated features (e.g. a parent population and its
largest subset) share importance: the lasso tends to keep one of them, so
a feature it drops may still differ between groups.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from drc_logging import get_logger, log_stage

log = get_logger(__name__)

MIN_PER_GROUP = 10
RECOMMENDED_PER_GROUP = 30
MODEL_LABELS = {'l1_logistic': 'Lasso logistic regression',
                'random_forest': 'Random forest'}
_C_GRID = np.logspace(-2, 2, 15)
_RF_DEPTHS = [2, 4, None]
_RF_TREES = 300
_ROC_GRID = np.linspace(0.0, 1.0, 101)
_FREQ_OFFSET = 0.01     # % added before log2 so empty clusters stay finite


def sample_size_status(n_per_group: dict[str, int]) -> tuple[str, str]:
    """
    ('block' | 'warn' | 'ok', message) for the smallest group size.
    """
    if not n_per_group:
        return 'block', "No samples in the selected comparison."
    smallest = min(n_per_group.values())
    detail = ", ".join(f"{g}={n}" for g, n in n_per_group.items())
    if smallest < MIN_PER_GROUP:
        return 'block', (
            f"Predictive analysis needs at least {MIN_PER_GROUP} samples per group "
            f"({detail}). With fewer, cross-validated performance estimates are "
            "too unstable to interpret — use the differential tests instead."
        )
    if smallest < RECOMMENDED_PER_GROUP:
        return 'warn', (
            f"{detail}: results will be noisy. {RECOMMENDED_PER_GROUP} or more "
            "samples per group are recommended."
        )
    return 'ok', detail


def build_feature_matrix(freq_df: pd.DataFrame | None, mfi_df: pd.DataFrame | None,
                         rel_list: list[str]) -> pd.DataFrame:
    """
    Samples × features for ``rel_list``: log2(% + 0.01) cluster
    frequencies ('Freq: <cluster>') and/or cluster × marker MFIs
    ('MFI: <cluster>_<channel>'). Columns that are entirely missing or
    constant over these samples are dropped.
    """
    frames = []
    if freq_df is not None and not freq_df.empty:
        f = freq_df.reindex(rel_list)
        frames.append(np.log2(f.astype(float) + _FREQ_OFFSET).add_prefix('Freq: '))
    if mfi_df is not None and not mfi_df.empty:
        frames.append(mfi_df.reindex(rel_list).astype(float).add_prefix('MFI: '))
    if not frames:
        return pd.DataFrame(index=rel_list)
    X = pd.concat(frames, axis=1)
    keep = X.notna().any(axis=0) & (X.std(axis=0, skipna=True).fillna(0.0) > 1e-12)
    return X.loc[:, keep]


def _welch_abs_t(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|Welch t| per column, ignoring NaNs; 0 where undefined."""
    a, b = X[y == 1], X[y == 0]
    with np.errstate(invalid='ignore', divide='ignore'):
        ma, mb = np.nanmean(a, axis=0), np.nanmean(b, axis=0)
        va, vb = np.nanvar(a, axis=0, ddof=1), np.nanvar(b, axis=0, ddof=1)
        na, nb = np.sum(np.isfinite(a), axis=0), np.sum(np.isfinite(b), axis=0)
        t = np.abs(ma - mb) / np.sqrt(va / na + vb / nb)
    return np.where(np.isfinite(t), t, 0.0)


def _lasso_search(inner_cv, seed):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('model', LogisticRegression(l1_ratio=1.0, solver='liblinear',
                                     class_weight='balanced', max_iter=5000,
                                     random_state=seed)),
    ])
    return GridSearchCV(pipe, {'model__C': list(_C_GRID)}, scoring='roc_auc',
                        cv=inner_cv, n_jobs=1, refit=True)


def _fit_forest(X, y, seed):
    """
    Random forest on median-imputed features, with max_depth chosen by
    out-of-bag ROC AUC on these training samples (each tree is scored only
    on samples it did not see). Returns (imputer, forest).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score

    imputer = SimpleImputer(strategy='median').fit(X)
    Xi = imputer.transform(X)
    best, best_auc = None, -np.inf
    for depth in _RF_DEPTHS:
        rf = RandomForestClassifier(n_estimators=_RF_TREES, max_depth=depth,
                                    class_weight='balanced_subsample', max_features='sqrt',
                                    oob_score=True, n_jobs=-1, random_state=seed).fit(Xi, y)
        oob = rf.oob_decision_function_[:, 1]
        ok = np.isfinite(oob)
        auc = roc_auc_score(y[ok], oob[ok]) if len(np.unique(y[ok])) == 2 else -np.inf
        if auc > best_auc:
            best, best_auc = rf, auc
    return imputer, best


def _panel_sizes(n_features: int, max_panel: int) -> list[int]:
    base = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
    upper = min(n_features, max_panel)
    sizes = [n for n in base if n <= upper]
    if upper not in sizes:
        sizes.append(upper)
    return sizes


def run_predictive(X: pd.DataFrame, groups: list[str], positive: str, negative: str,
                   models=('l1_logistic', 'random_forest'), outer_folds: int = 5,
                   repeats: int = 5, inner_folds: int = 3, max_panel: int = 50,
                   seed: int = 0, progress=None) -> dict:
    """
    Nested cross-validated classification of ``positive`` vs ``negative``.

    X      : samples × features (rows aligned to ``groups``); NaN allowed.
    groups : group name per row; rows in other groups are ignored.
    models : any of 'l1_logistic', 'random_forest'.
    progress: optional callable(str) for status messages.

    Returns a dict:
      'performance': DataFrame — model, auc_mean, auc_sd,
                     balanced_accuracy_mean, balanced_accuracy_sd
      'roc':         DataFrame — model, fpr, tpr (mean over repeats)
      'importance':  DataFrame — feature, direction (+1 higher in positive),
                     lasso_selection, lasso_coef, forest_importance,
                     forest_importance_sd (columns for the models run)
      'panel':       DataFrame — n_features, auc_mean, auc_sd
      'summary':     dict — positive, negative, n_per_group, n_features,
                     minimal_panel, best_panel_auc, status, message,
                     outer_folds, repeats
    Raises ValueError when a group has fewer than MIN_PER_GROUP samples.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    log_stage(log, "PREDICTIVE ANALYSIS")

    def _say(msg):
        log.info(msg)
        if progress is not None:
            progress(msg)

    mask = np.array([g in (positive, negative) for g in groups])
    Xs = X.loc[mask]
    y = np.array([1 if g == positive else 0 for g, m in zip(groups, mask) if m])
    n_per_group = {positive: int(y.sum()), negative: int((1 - y).sum())}
    status, message = sample_size_status(n_per_group)
    if status == 'block':
        raise ValueError(message)
    if Xs.shape[1] == 0:
        raise ValueError("No usable features (all missing or constant).")

    features = list(Xs.columns)
    Xv = Xs.values.astype(float)
    n_min = int(min(n_per_group.values()))
    k_outer = int(min(outer_folds, n_min))
    k_inner = int(min(inner_folds, n_min - (n_min // k_outer) - 1)) if n_min > 3 else 2
    k_inner = max(2, k_inner)

    oof = {m: np.full((repeats, len(y)), np.nan) for m in models}
    lasso_nonzero = np.zeros(len(features))
    lasso_coef_sum = np.zeros(len(features))
    forest_imp = []
    n_lasso_fits = 0
    sizes = _panel_sizes(len(features), max_panel)
    panel_oof = np.full((repeats, len(sizes), len(y)), np.nan)

    for r in range(repeats):
        outer = StratifiedKFold(n_splits=k_outer, shuffle=True, random_state=seed + r)
        for f, (tr, te) in enumerate(outer.split(Xv, y)):
            _say(f"Predictive: repeat {r + 1}/{repeats}, fold {f + 1}/{k_outer}")
            inner = StratifiedKFold(n_splits=k_inner, shuffle=True, random_state=seed + 1000 + r)
            if 'l1_logistic' in models:
                search = _lasso_search(inner, seed).fit(Xv[tr], y[tr])
                oof['l1_logistic'][r, te] = search.predict_proba(Xv[te])[:, 1]
                coef = search.best_estimator_.named_steps['model'].coef_.ravel()
                lasso_nonzero += coef != 0
                lasso_coef_sum += coef
                n_lasso_fits += 1
            if 'random_forest' in models:
                imputer, forest = _fit_forest(Xv[tr], y[tr], seed + r)
                oof['random_forest'][r, te] = forest.predict_proba(imputer.transform(Xv[te]))[:, 1]
                forest_imp.append(forest.feature_importances_)

            ranking = np.argsort(-_welch_abs_t(Xv[tr], y[tr]), kind='mergesort')
            for si, n in enumerate(sizes):
                cols = ranking[:n]
                panel = Pipeline([
                    ('impute', SimpleImputer(strategy='median')),
                    ('scale', StandardScaler()),
                    ('model', LogisticRegression(C=1.0, class_weight='balanced', max_iter=5000)),
                ]).fit(Xv[np.ix_(tr, cols)], y[tr])
                panel_oof[r, si, te] = panel.predict_proba(Xv[np.ix_(te, cols)])[:, 1]

    perf_rows, roc_rows = [], []
    for m in models:
        aucs, baccs, tprs = [], [], []
        for r in range(repeats):
            score = oof[m][r]
            aucs.append(roc_auc_score(y, score))
            baccs.append(balanced_accuracy_score(y, (score >= 0.5).astype(int)))
            fpr, tpr, _ = roc_curve(y, score)
            tprs.append(np.interp(_ROC_GRID, fpr, tpr))
        perf_rows.append({'model': MODEL_LABELS[m],
                          'auc_mean': float(np.mean(aucs)), 'auc_sd': float(np.std(aucs, ddof=1)) if repeats > 1 else 0.0,
                          'balanced_accuracy_mean': float(np.mean(baccs)),
                          'balanced_accuracy_sd': float(np.std(baccs, ddof=1)) if repeats > 1 else 0.0})
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[0] = 0.0
        roc_rows.append(pd.DataFrame({'model': MODEL_LABELS[m], 'fpr': _ROC_GRID, 'tpr': mean_tpr}))

    with np.errstate(invalid='ignore'):
        diff = np.nanmean(Xv[y == 1], axis=0) - np.nanmean(Xv[y == 0], axis=0)
    importance = pd.DataFrame({'feature': features, 'direction': np.sign(np.nan_to_num(diff))})
    if 'l1_logistic' in models and n_lasso_fits:
        importance['lasso_selection'] = lasso_nonzero / n_lasso_fits
        importance['lasso_coef'] = lasso_coef_sum / n_lasso_fits
    if 'random_forest' in models and forest_imp:
        stack = np.vstack(forest_imp)
        importance['forest_importance'] = stack.mean(axis=0)
        importance['forest_importance_sd'] = stack.std(axis=0, ddof=1) if len(forest_imp) > 1 else 0.0
    sort_cols = [c for c in ('lasso_selection', 'forest_importance') if c in importance.columns]
    importance = importance.sort_values(sort_cols, ascending=False, kind='mergesort').reset_index(drop=True)

    panel_auc = np.array([[roc_auc_score(y, panel_oof[r, si]) for si in range(len(sizes))]
                          for r in range(repeats)])
    panel = pd.DataFrame({'n_features': sizes, 'auc_mean': panel_auc.mean(axis=0),
                          'auc_sd': panel_auc.std(axis=0, ddof=1) if repeats > 1 else 0.0})
    best = float(panel['auc_mean'].max())
    minimal = int(panel.loc[panel['auc_mean'] >= 0.95 * best, 'n_features'].min())

    summary = {'positive': positive, 'negative': negative, 'n_per_group': n_per_group,
               'n_features': len(features), 'minimal_panel': minimal, 'best_panel_auc': best,
               'status': status, 'message': message, 'outer_folds': k_outer, 'repeats': repeats}
    _say(f"Predictive: done ({len(features)} features, minimal panel {minimal}).")
    return {'performance': pd.DataFrame(perf_rows), 'roc': pd.concat(roc_rows, ignore_index=True),
            'importance': importance, 'panel': panel, 'summary': summary}
