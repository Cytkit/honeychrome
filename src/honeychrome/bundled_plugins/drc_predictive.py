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

``run_age_model`` is the same idea for a number instead of a group: an
"immune age" model that predicts a continuous variable (age, days after
vaccination, any number) from the features, judged only on samples it
was not trained on. See its docstring.
"""

from __future__ import annotations

import os

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
MIN_AGE_SAMPLES = 20
RECOMMENDED_AGE_SAMPLES = 40
AGE_L1_RATIOS = (0.2, 0.5, 0.8, 1.0)
AGE_N_ALPHAS = 30
AGE_MIN_OBSERVED = 0.8  # share of samples a feature must be measured in
DEFAULT_AGE_PERMUTATIONS = 100


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
                                    oob_score=True, n_jobs=max(1, (os.cpu_count() or 2) - 1),
                                    random_state=seed).fit(Xi, y)
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


# ---------------------------------------------------------------------------
# Immune age model (a numeric target)
# ---------------------------------------------------------------------------

def age_sample_size_status(n: int) -> tuple[str, str]:
    """('block' | 'warn' | 'ok', message) for the number of samples of an
    immune age model."""
    if n < MIN_AGE_SAMPLES:
        return 'block', (f"An immune age model needs at least {MIN_AGE_SAMPLES} samples with a "
                         f"value ({n} here): with fewer, its cross-validated error is too "
                         "unstable to interpret.")
    if n < RECOMMENDED_AGE_SAMPLES:
        return 'warn', (f"{n} samples: results will be noisy; {RECOMMENDED_AGE_SAMPLES} or more "
                        "are recommended.")
    return 'ok', f"{n} samples"


def _grouped_splits(blocks: np.ndarray, k: int, rng) -> list[tuple[np.ndarray, np.ndarray]]:
    """k folds keeping every block (donor, pair) whole, blocks shuffled."""
    uniq = np.unique(blocks)
    fold_of = {b: i % k for i, b in enumerate(rng.permutation(uniq))}
    fold = np.array([fold_of[b] for b in blocks])
    return [(np.flatnonzero(fold != i), np.flatnonzero(fold == i)) for i in range(k)]


def _age_pipeline(inner_cv):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNetCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    import sklearn

    major, minor = (int(v) for v in sklearn.__version__.split('.')[:2])
    # scikit-learn 1.7 takes the number of penalty strengths as 'alphas'
    grid = ({'alphas': AGE_N_ALPHAS} if (major, minor) >= (1, 7)
            else {'n_alphas': AGE_N_ALPHAS})
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         ElasticNetCV(l1_ratio=list(AGE_L1_RATIOS), cv=inner_cv, max_iter=5000,
                                      tol=1e-3, **grid))


def _age_fit(X: np.ndarray, y: np.ndarray, blocks: np.ndarray, inner_folds: int):
    """Elastic net with penalty strength and lasso/ridge mix chosen by
    grouped inner cross-validation, plus the linear bias of its predictions
    on age (slope, intercept), estimated from cross-validated predictions
    with that penalty. Returns (fitted pipeline, bias coefficients)."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    import warnings

    from sklearn.exceptions import ConvergenceWarning

    k = int(max(2, min(inner_folds, np.unique(blocks).size)))
    inner = list(GroupKFold(k).split(X, y, blocks))
    with warnings.catch_warnings():
        # weak penalties on many features stop at max_iter; the penalty
        # chosen by cross-validation is not among them
        warnings.simplefilter('ignore', ConvergenceWarning)
        model = _age_pipeline(inner).fit(X, y)
        en = model[-1]
        fixed = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                              ElasticNet(alpha=en.alpha_, l1_ratio=en.l1_ratio_, max_iter=5000,
                                         tol=1e-3))
        inner_pred = cross_val_predict(fixed, X, y, cv=inner)
    bias = np.polyfit(y, inner_pred - y, 1)
    return model, bias


def _age_scores(y: np.ndarray, pred: np.ndarray) -> dict:
    with np.errstate(invalid='ignore', divide='ignore'):
        r2 = 1.0 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2)
    return {'mae': float(np.mean(np.abs(pred - y))), 'r2': float(r2),
            'spearman': float(pd.Series(pred).corr(pd.Series(y), method='spearman'))}


def run_age_model(X: pd.DataFrame, y, *, blocks=None, X_other: pd.DataFrame | None = None,
                  y_other=None, outer_folds: int = 5, repeats: int = 5, inner_folds: int = 3,
                  n_permutations: int = DEFAULT_AGE_PERMUTATIONS, seed: int = 0,
                  progress=None, on_step=None) -> dict:
    """
    Immune age model: how well the features predict a continuous variable
    (age, days after vaccination, any number), which features carry the
    prediction, and each sample's predicted minus actual value.

    In plain terms, with age as the example: a model learns to guess each
    sample's age from its immune profile alone, and is always judged on
    samples it was not trained on. If it guesses well, the immune system
    changes with age in a consistent way; samples it places older or
    younger than they are may be ahead or behind for their age.

    Model: elastic net linear regression (Zou & Hastie 2005), a penalised
    regression mixing the lasso, which keeps few features, and ridge, which
    shares weight among correlated features such as a parent population and
    its largest subset. Features are median-imputed and standardised inside
    each training split.

    Nested cross-validation: the outer split (``outer_folds`` folds,
    ``repeats`` times) gives every sample a prediction from a model that
    never saw it; inside each training part an inner split chooses the
    penalty strength and the lasso/ridge mix (AGE_L1_RATIOS). Folds keep
    every *block* (donor, or pair of samples from one donor) whole,
    otherwise the same donor is in training and testing and the model
    looks better than it is.

    Gap: the out-of-fold prediction minus the actual value. The predictions
    of any such model are pulled towards the middle (the lowest values
    predicted too high, the highest too low), so the raw gap follows the
    variable. 'gap_corrected' removes that: the gap's linear trend on
    age is estimated inside each training split from its own
    cross-validated predictions and subtracted (de Lange & Cole 2020,
    NeuroImage Clin 26:102229). Only the corrected gap should be compared
    between samples or groups.

    Chance check: the whole procedure (one repeat) rerun with the variable
    shuffled among blocks ``n_permutations`` times; p = (1 + permutations
    with a mean absolute error at least as small) / (1 + permutations).

    X       : samples × features of the samples in the model (NaN allowed;
              features measured in fewer than AGE_MIN_OBSERVED of them, or
              constant, are dropped).
    y       : the variable for each row of X.
    blocks  : donor or pair of each row (None: each sample its own).
    X_other, y_other : samples outside the model (e.g. other groups),
              predicted by a final model fitted on all of X and marked as
              not cross-validated; y_other may hold NaN.
    progress: optional callable(str) for status messages.
    on_step : optional callable(done, total), called after each model fit
              (outer folds, chance shuffles and the final model).

    Returns {'performance' (one row: mae, r2, spearman as mean and SD over
    repeats, mae_baseline of predicting the mean, p_chance), 'predictions'
    (sample, actual, predicted, predicted_sd, gap, gap_corrected,
    cross_validated), 'importance' (feature, selection, coef, direction),
    'chance' (mae of each permutation), 'summary'}.
    Raises ValueError with too few samples or no usable feature.
    """
    log_stage(log, "IMMUNE AGE MODEL")

    def _say(msg):
        log.info(msg)
        if progress is not None:
            progress(msg)

    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    X = X.loc[ok]
    y = y[ok]
    blocks = (np.array([str(b) for b in np.asarray(blocks, dtype=object)[ok]])
              if blocks is not None else np.array([str(i) for i in X.index]))
    n = len(y)
    status, message = age_sample_size_status(n)
    if status == 'block':
        raise ValueError(message)
    keep = (X.notna().mean(axis=0) >= AGE_MIN_OBSERVED) & (X.std(axis=0, skipna=True).fillna(0) > 1e-12)
    X = X.loc[:, keep]
    if X.shape[1] == 0:
        raise ValueError("No usable features (each must be measured in at least "
                         f"{AGE_MIN_OBSERVED:.0%} of the samples and vary).")
    features = list(X.columns)
    Xv = X.to_numpy(float)
    k_outer = int(max(2, min(outer_folds, np.unique(blocks).size)))
    rng = np.random.default_rng(seed)
    total_steps = (repeats * k_outer + int(n_permutations) * k_outer
                   + (1 if X_other is not None and len(X_other) else 0))
    done_steps = [0]

    def _step():
        done_steps[0] += 1
        if on_step is not None:
            on_step(done_steps[0], total_steps)

    pred = np.full((repeats, n), np.nan)
    corrected = np.full((repeats, n), np.nan)
    coefs, chosen = [], []
    for r in range(repeats):
        for f, (tr, te) in enumerate(_grouped_splits(blocks, k_outer, rng)):
            _say(f"Immune age: repeat {r + 1}/{repeats}, fold {f + 1}/{k_outer}")
            model, bias = _age_fit(Xv[tr], y[tr], blocks[tr], inner_folds)
            pred[r, te] = model.predict(Xv[te])
            corrected[r, te] = (pred[r, te] - y[te]) - np.polyval(bias, y[te])
            en = model[-1]
            coefs.append(en.coef_)
            chosen.append((float(en.alpha_), float(en.l1_ratio_)))
            _step()
    scores = pd.DataFrame([_age_scores(y, p) for p in pred])

    # chance check: the variable shuffled among blocks
    chance = []
    shared = pd.Series(y).groupby(blocks).nunique().max() <= 1
    for i in range(int(n_permutations)):
        _say(f"Immune age: chance check {i + 1}/{n_permutations}")
        if shared:
            uniq = np.unique(blocks)
            value_of = dict(zip(uniq, rng.permutation([y[blocks == b][0] for b in uniq])))
            yp = np.array([value_of[b] for b in blocks])
        else:
            yp = rng.permutation(y)
        pp = np.full(n, np.nan)
        for tr, te in _grouped_splits(blocks, k_outer, rng):
            model, _bias = _age_fit(Xv[tr], yp[tr], blocks[tr], inner_folds)
            pp[te] = model.predict(Xv[te])
            _step()
        chance.append(_age_scores(yp, pp)['mae'])
    chance = np.asarray(chance, dtype=float)
    mae_mean = float(scores['mae'].mean())
    p_chance = (float((1 + np.sum(chance <= mae_mean)) / (1 + chance.size))
                if chance.size else np.nan)

    C = np.vstack(coefs)
    importance = pd.DataFrame({'feature': features, 'selection': (C != 0).mean(axis=0),
                               'coef': C.mean(axis=0)})
    importance['direction'] = np.sign(importance['coef'])
    importance = importance.sort_values(['selection', 'coef'], ascending=[False, False],
                                        key=lambda col: col.abs() if col.name == 'coef' else col,
                                        kind='mergesort').reset_index(drop=True)
    mean_pred = pred.mean(axis=0)
    predictions = pd.DataFrame({
        'sample': list(X.index), 'actual': y, 'predicted': mean_pred,
        'predicted_sd': pred.std(axis=0, ddof=1) if repeats > 1 else 0.0,
        'gap': mean_pred - y, 'gap_corrected': corrected.mean(axis=0), 'cross_validated': True})

    if X_other is not None and len(X_other):
        _say("Immune age: predicting the samples outside the model")
        model, bias = _age_fit(Xv, y, blocks, inner_folds)
        _step()
        Xo = X_other.reindex(columns=features).to_numpy(float)
        po = model.predict(Xo)
        yo = (np.asarray(y_other, dtype=float) if y_other is not None
              else np.full(len(X_other), np.nan))
        others = pd.DataFrame({'sample': list(X_other.index), 'actual': yo, 'predicted': po,
                               'predicted_sd': np.nan, 'gap': po - yo,
                               'gap_corrected': (po - yo) - np.polyval(bias, yo),
                               'cross_validated': False})
        predictions = pd.concat([predictions, others], ignore_index=True)

    pairs = pd.Series([f"{a:.3g} / {l1:g}" for a, l1 in chosen]).value_counts()
    performance = pd.DataFrame([{
        'mae_mean': mae_mean, 'mae_sd': float(scores['mae'].std(ddof=1)) if repeats > 1 else 0.0,
        'r2_mean': float(scores['r2'].mean()),
        'r2_sd': float(scores['r2'].std(ddof=1)) if repeats > 1 else 0.0,
        'spearman_mean': float(scores['spearman'].mean()),
        'spearman_sd': float(scores['spearman'].std(ddof=1)) if repeats > 1 else 0.0,
        'mae_baseline': float(np.mean(np.abs(y - y.mean())) * n / max(n - 1, 1)),
        'p_chance': p_chance}])
    summary = {'n': n, 'n_features': len(features), 'n_blocks': int(np.unique(blocks).size),
               'outer_folds': k_outer, 'inner_folds': int(inner_folds), 'repeats': int(repeats),
               'n_permutations': int(chance.size), 'status': status, 'message': message,
               'penalty': str(pairs.index[0]) if len(pairs) else ''}
    _say(f"Immune age: done (mean absolute error {mae_mean:.3g}).")
    return {'performance': performance, 'predictions': predictions, 'importance': importance,
            'chance': pd.DataFrame({'mae': chance}), 'summary': summary}


def age_predictions(result: dict, groups: dict | None = None) -> pd.DataFrame:
    """The immune age model's per-sample predictions with each sample's group
    (from *groups*, {sample: group}) as the second column."""
    pred = result['predictions'].copy()
    groups = groups or {}
    pred.insert(1, 'group', [str(groups.get(s, '')) for s in pred['sample']])
    return pred


def age_model_by_group(result: dict, groups: dict | None = None) -> pd.DataFrame:
    """
    How the immune age model sees each group: the model's own group
    (cross-validated predictions) and every group predicted by the final
    model, whether or not its samples have a value of the variable.
    Columns: group, in_model, n (samples predicted), predicted_mean,
    predicted_sd, n_actual (samples that also have a value), mae (mean
    absolute error), mean_gap (mean predicted − actual), mean_gap_corrected,
    spearman (predicted against actual; NaN with fewer than 3 samples). The
    columns that need a value are NaN for a group without one.
    """
    pred = age_predictions(result, groups)
    rows = []
    for (group, inside), part in pred.groupby(['group', 'cross_validated'], sort=False):
        part = part[part['predicted'].notna()]
        if part.empty:
            continue
        known = part[part['actual'].notna()]
        n_known = int(len(known))
        rho = (float(known['predicted'].corr(known['actual'], method='spearman'))
               if n_known >= 3 else np.nan)
        rows.append({
            'group': group, 'in_model': bool(inside), 'n': int(len(part)),
            'predicted_mean': float(part['predicted'].mean()),
            'predicted_sd': float(part['predicted'].std(ddof=1)) if len(part) > 1 else np.nan,
            'n_actual': n_known,
            'mae': (float(np.mean(np.abs(known['predicted'] - known['actual'])))
                    if n_known else np.nan),
            'mean_gap': float(known['gap'].mean()) if n_known else np.nan,
            'mean_gap_corrected': float(known['gap_corrected'].mean()) if n_known else np.nan,
            'spearman': rho})
    out = pd.DataFrame(rows, columns=['group', 'in_model', 'n', 'predicted_mean', 'predicted_sd',
                                      'n_actual', 'mae', 'mean_gap', 'mean_gap_corrected',
                                      'spearman'])
    return out.sort_values('in_model', ascending=False, kind='stable').reset_index(drop=True)


def age_group_rows(by_group: pd.DataFrame, unit: str = '') -> list[dict]:
    """Rows for showing age_model_by_group: each group's predicted values and,
    where its samples have a value of the variable, how close they are."""
    def _num(v, fmt):
        return format(v, fmt) + (f" {unit}" if unit else '') if np.isfinite(v) else ''

    rows = []
    for _i, r in by_group.iterrows():
        spread = (f"{r['predicted_mean']:.3g} ± {r['predicted_sd']:.2g}"
                  if np.isfinite(r['predicted_sd']) else f"{r['predicted_mean']:.3g}")
        rows.append({
            'Group': r['group'] + ('' if r['in_model'] else ' (predicted, not in the model)'),
            'Samples': int(r['n']),
            'Predicted (mean ± SD)': spread + (f" {unit}" if unit else ''),
            'With a value': int(r['n_actual']),
            'Mean absolute error': _num(r['mae'], '.3g'),
            'Mean gap': _num(r['mean_gap'], '+.3g'),
            'Mean gap, corrected': _num(r['mean_gap_corrected'], '+.3g'),
            'Spearman': f"{r['spearman']:.2f}" if np.isfinite(r['spearman']) else ''})
    return rows
