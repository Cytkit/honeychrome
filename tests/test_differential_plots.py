"""
test_differential_plots.py
--------------------------
Layout of the shared statistics figures (view_components/differential_plots.py):
volcano labels kept apart, the heatmap colour bar's minimum height, and
hover labels hidden before an export. Built off screen with matplotlib only.

Usage:
    pytest tests/test_differential_plots.py -m numpy_only
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('matplotlib')
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from honeychrome.view_components import differential_plots as dp  # noqa: E402


def _close_volcano_results() -> pd.DataFrame:
    """Three significant points almost on top of each other, and three
    non-significant ones."""
    p = [1e-3, 1.1e-3, 1.2e-3, 0.5, 0.7, 0.9]
    return pd.DataFrame({
        'feature': ['Population number 0', 'Population number 1', 'Population number 2',
                    'a', 'b', 'c'],
        'logFC': [1.10, 1.15, 1.20, -0.2, 0.1, 0.3],
        'P.Value': p, 'adj.P.Val': p,
        'significant': [True, True, True, False, False, False]})


@pytest.mark.numpy_only
def test_volcano_labels_of_close_points_do_not_overlap():
    fig = dp.make_volcano_figure(_close_volcano_results(), 'Frequencies', 0.05, 0.5)
    FigureCanvasAgg(fig).draw()
    renderer = fig.canvas.get_renderer()
    notes = [t for t in fig.axes[0].texts if t.get_text().startswith('Population')]
    assert len(notes) == 3
    boxes = [Text.get_window_extent(n, renderer) for n in notes]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not boxes[i].overlaps(boxes[j])
    assert any(n.arrow_patch.get_visible() for n in notes)


@pytest.mark.numpy_only
def test_heatmap_colour_bar_is_tall_enough_with_one_significant_row():
    pytest.importorskip('scipy')
    rng = np.random.default_rng(0)
    samples = [f's{i}' for i in range(8)]
    data = pd.DataFrame(rng.normal(size=(8, 3)), index=samples, columns=['f1', 'f2', 'f3'])
    res = pd.DataFrame({'feature': ['f1', 'f2', 'f3'], 'logFC': [1.0, 0.1, 0.0],
                        'P.Value': [1e-4, 0.5, 0.9], 'adj.P.Val': [1e-3, 0.6, 0.9],
                        'significant': [True, False, False]})
    groups = dict(zip(samples, ['A'] * 4 + ['B'] * 4))
    fig = dp.make_heatmap_figure(res, data, 'Frequencies', groups, group_a='A', group_b='B')
    FigureCanvasAgg(fig).draw()
    colour_bar = [a for a in fig.axes if a.get_ylabel() == '% frequency'][0]
    assert colour_bar.get_position().height * fig.get_figheight() >= dp.MIN_COLOURBAR_INCHES


@pytest.mark.numpy_only
def test_hover_labels_can_be_hidden_before_an_export():
    fig = dp.make_volcano_figure(_close_volcano_results(), 'Frequencies', 0.05, 0.5)
    hover = fig._hover_annotations
    assert hover
    hover[0].set_visible(True)
    assert dp.hide_hover_annotations(fig) is True
    assert not any(a.get_visible() for a in hover)
    assert dp.hide_hover_annotations(fig) is False
