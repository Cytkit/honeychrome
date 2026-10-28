"""
test_plot_3d_af.py
------------------
AF Abundance / AF Index on plot axes: the 3D plot plugin's Reset Axes and
Fit Axes, the 2D plot's Fit Axes, and tick labels for the AF Index axis.

The plot methods are called on a light stand-in for the widget, so no
OpenGL context or QApplication is needed; only the transform logic runs.

Usage:
    pytest tests/test_plot_3d_af.py -m numpy_only
"""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import honeychrome.settings as hc_settings
from honeychrome.controller_components.functions import (
    assign_default_transforms,
    generate_transformations,
)

import af_synthetic as syn

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / 'src' / 'honeychrome' / 'bundled_plugins'
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

import plot_3d_tab as p3  # noqa: E402

AF_A = hc_settings.af_abundance_channel
AF_I = hc_settings.af_index_channel
CHANNELS = ['F1', AF_A, AF_I]


def _transformations(n_af_spectra):
    return generate_transformations(
        assign_default_transforms(syn.unmixed_settings(), n_af_spectra=n_af_spectra))


def _event_data(with_af=True, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    data = np.zeros((n, len(syn.UNMIXED_PNN)))
    data[:, syn.UNMIXED_PNN.index('F1')] = rng.gamma(1.0, 3000.0, n)
    if with_af:
        data[:, syn.UNMIXED_PNN.index(AF_A)] = rng.gamma(2.0, 400.0, n)
        data[:, syn.UNMIXED_PNN.index(AF_I)] = rng.integers(syn.N_OTHER_AF + 1, syn.N_AF_TOTAL + 1, n)
    return data


def _assert_usable(tr):
    """A transform the plots can bin, place and label."""
    interior = tr.scale[1:-1]
    assert np.all(np.isfinite(interior))
    assert np.all(np.diff(interior) > 0), 'histogram bin edges must increase'
    ticks = tr.ticks() if tr.ticks is not None else None
    if ticks and tr.id == 0:
        assert all(np.isfinite(pos) for level in ticks for pos, _label in level)


def _tile(tmp_path, with_af=True):
    """Stand-in Plot3DPlotWidget carrying only what the axis methods use."""
    controller = syn.FakeController(tmp_path)
    tile = SimpleNamespace(
        controller=controller,
        channels=list(CHANNELS),
        pnn=list(syn.UNMIXED_PNN),
        transformations=_transformations(controller.n_af_spectra()),
        event_data=_event_data(with_af),
        display_range=p3.DEFAULT_DISPLAY_RANGE,
        grid=SimpleNamespace(mark_channel_transformed=lambda ch: None,
                             mark_channels_reset=lambda chs: None),
    )
    tile.id_channels = [tile.pnn.index(c) for c in tile.channels]
    tile.displayed_indices = np.arange(len(tile.event_data))
    for name in ('refresh_transforms', 'rebuild_positions'):
        setattr(tile, name, types.MethodType(getattr(p3.Plot3DPlotWidget, name), tile))
    tile.rebuild_colors = lambda: None
    tile._refresh_scatter = lambda: None
    tile._update_axis_strip_labels = lambda: None
    tile._build_ticks = lambda: [p3.ticks_for_axis(tr, tile.display_range) for tr in tile.transforms]
    tile.refresh_transforms()
    return tile


# ---------------------------------------------------------------------------
# 3D plot plugin
# ---------------------------------------------------------------------------

@pytest.mark.numpy_only
def test_3d_reset_axes_sizes_af_index_to_the_experiment_library(tmp_path):
    """Reset Axes should give AF Index the same 0..n_spectra axis as the
    controller's own reset (controller.reset_axes_transforms)."""
    tile = _tile(tmp_path)
    tile.transformations[AF_I].scale_t = 999
    p3.Plot3DPlotWidget.reset_axes_transforms(tile)
    assert tile.transformations[AF_I].scale_t == syn.N_AF_TOTAL
    assert tile.transformations[AF_I].id == 0


@pytest.mark.numpy_only
def test_3d_fit_axes_on_af_channels_with_af_data(tmp_path):
    tile = _tile(tmp_path, with_af=True)
    p3.Plot3DPlotWidget.fit_axes_to_data(tile)
    for ch in CHANNELS:
        _assert_usable(tile.transformations[ch])
    assert np.isfinite(tile.positions).all()


@pytest.mark.numpy_only
def test_3d_fit_axes_leaves_all_zero_af_channels_usable(tmp_path):
    """A sample unmixed without AF correction has AF Abundance and AF Index
    at 0 for every event; Fit Axes must not collapse those axes."""
    tile = _tile(tmp_path, with_af=False)
    p3.Plot3DPlotWidget.fit_axes_to_data(tile)
    for ch in CHANNELS:
        _assert_usable(tile.transformations[ch])
    assert np.isfinite(tile.positions).all()


@pytest.mark.numpy_only
def test_af_index_axis_ticks_are_whole_numbers():
    tr = _transformations(12)[AF_I]
    labels = [label for _pos, label in p3.ticks_for_axis(tr)]
    assert labels[0] == '0' and labels[-1] == '12'
    assert all(label.isdigit() for label in labels)


@pytest.mark.numpy_only
def test_af_index_default_axis_without_af_profiles():
    tr = _transformations(None)[AF_I]
    assert tr.scale_t == hc_settings.default_af_index_range
    _assert_usable(tr)


# ---------------------------------------------------------------------------
# 2D plot
# ---------------------------------------------------------------------------

@pytest.mark.numpy_only
def test_2d_fit_axes_leaves_all_zero_af_channels_usable():
    from honeychrome.view_components.cytometry_plot_widget import CytometryPlotWidget

    plot = SimpleNamespace(
        plot={'type': 'hist2d', 'channel_x': AF_A, 'channel_y': AF_I},
        data_for_cytometry_plots={'event_data': _event_data(with_af=False)},
        pnn=list(syn.UNMIXED_PNN),
        transformations=_transformations(syn.N_AF),
        bus=None,
    )
    CytometryPlotWidget.fit_axes_to_data(plot)
    for ch in (AF_A, AF_I):
        _assert_usable(plot.transformations[ch])
