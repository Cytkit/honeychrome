"""
test_age_model.py
-----------------
The immune age model (bundled_plugins/drc_predictive.run_age_model) on
synthetic samples whose features follow age.

Usage:
    pytest tests/test_age_model.py -m numpy_only
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('sklearn')
_PLUGIN_DIR = Path(__file__).resolve().parents[1] / 'src' / 'honeychrome' / 'bundled_plugins'
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

import drc_predictive as dp  # noqa: E402


def _samples(n=40, seed=1, paired=False):
    rng = np.random.default_rng(seed)
    age = np.repeat(rng.uniform(18, 37, n // 2), 2) if paired else rng.uniform(18, 37, n)
    X = pd.DataFrame({f'noise{j}': rng.normal(size=n) for j in range(30)})
    X['rises'] = 0.3 * age + rng.normal(scale=0.5, size=n)
    X['falls'] = -0.2 * age + rng.normal(scale=0.5, size=n)
    X.index = [f's{i}' for i in range(n)]
    X.iloc[0, 0] = np.nan
    blocks = list(np.repeat([f'p{i}' for i in range(n // 2)], 2)) if paired else None
    return X, age, blocks


@pytest.mark.numpy_only
def test_age_model_predicts_age_and_names_the_features():
    X, age, _ = _samples()
    other = X.iloc[:3].copy()
    res = dp.run_age_model(X, age, X_other=other, y_other=age[:3], repeats=2,
                           n_permutations=10, seed=0)
    perf = res['performance'].iloc[0]
    assert perf['mae_mean'] < 0.5 * perf['mae_baseline']
    assert perf['p_chance'] == pytest.approx(1 / 11)
    top = set(res['importance']['feature'].head(2))
    assert top == {'rises', 'falls'}
    imp = res['importance'].set_index('feature')
    assert imp.loc['rises', 'direction'] == 1 and imp.loc['falls', 'direction'] == -1
    pred = res['predictions']
    assert pred['cross_validated'].sum() == 40 and (~pred['cross_validated']).sum() == 3
    cv = pred[pred['cross_validated']]
    assert abs(np.corrcoef(cv['gap_corrected'], cv['actual'])[0, 1]) < \
        abs(np.corrcoef(cv['gap'], cv['actual'])[0, 1]) + 0.1
    assert res['summary']['status'] == 'ok' and res['summary']['n_permutations'] == 10
    groups = {s: ('A' if i >= 3 else 'B') for i, s in enumerate(X.index)}
    table = dp.age_model_by_group(res, groups)
    assert list(table['group']) == ['B', 'A', 'B'] and list(table['in_model']) == [True, True, False]
    outside = table[~table['in_model']].iloc[0]
    assert outside['n'] == 3 and np.isfinite(outside['mae'])
    preds = dp.age_predictions(res, groups)
    assert list(preds.columns[:2]) == ['sample', 'group'] and len(preds) == 43


@pytest.mark.numpy_only
def test_age_model_keeps_pairs_together_and_refuses_too_few_samples():
    X, age, blocks = _samples(paired=True)
    rng = np.random.default_rng(0)
    for tr, te in dp._grouped_splits(np.array(blocks), 5, rng):
        assert not set(np.array(blocks)[tr]) & set(np.array(blocks)[te])
    res = dp.run_age_model(X, age, blocks=blocks, repeats=1, n_permutations=0)
    assert res['summary']['n_blocks'] == 20 and np.isnan(res['performance'].iloc[0]['p_chance'])
    with pytest.raises(ValueError, match='at least 20'):
        dp.run_age_model(X.iloc[:10], age[:10], n_permutations=0)
