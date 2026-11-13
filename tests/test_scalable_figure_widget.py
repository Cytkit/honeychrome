"""
test_scalable_figure_widget.py
------------------------------
Exports from ScalableFigureWidget (view_components/scalable_figure_widget.py):
the figure is saved at its natural size whatever size the slider shows,
without hover labels, and is shown at the slider's size again afterwards;
Export Graphic writes into the results folder at EXPORT_DPI and reports the
file in the main window's status bar. Runs off screen.

Usage:
    QT_QPA_PLATFORM=offscreen pytest tests/test_scalable_figure_widget.py
"""

import os

import pytest

pytest.importorskip('PySide6')
pytest.importorskip('matplotlib')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import matplotlib.image as mpimg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtWidgets import QApplication, QMainWindow  # noqa: E402

from honeychrome.view_components.scalable_figure_widget import (  # noqa: E402
    EXPORT_DPI, ScalableFigureWidget, results_folder,
)


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def test_export_uses_the_natural_size_not_the_slider_size(app, tmp_path):
    fig = Figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1])
    hover = ax.annotate('hover label', xy=(0.5, 0.5))
    fig._hover_annotations = [hover]
    widget = ScalableFigureWidget(fig, title='test')
    widget.show()
    widget.size_slider.setValue(50)
    widget._apply_scale()
    app.processEvents()
    shown = tuple(fig.get_size_inches())
    assert shown[0] < 3.0    # the slider shrinks the figure itself

    path = tmp_path / 'figure.png'
    widget.save_full_size(path, dpi=100)

    height, width = mpimg.imread(path).shape[:2]
    assert width > 300       # about 4 in at 100 dpi, trimmed; 2 in if saved as shown
    assert not hover.get_visible()
    assert tuple(fig.get_size_inches()) == pytest.approx(shown)
    assert widget.make_toolbar() is not None
    widget.close()


def test_export_graphic_writes_to_the_results_folder_and_reports_it(app, tmp_path, monkeypatch):
    import honeychrome.settings as settings
    monkeypatch.setattr(settings, 'graphics_export_format_retrieved', 'png')
    fig = Figure(figsize=(2, 1))
    fig.add_subplot(111).plot([0, 1], [0, 1])
    window = QMainWindow()
    folder = results_folder(tmp_path, 'Plugin_Results')
    widget = ScalableFigureWidget(fig, title='my plot', experiment_dir=folder)
    window.setCentralWidget(widget)
    widget.export_graphic()

    path = folder / 'my_plot.png'
    assert path.exists()
    assert mpimg.imread(path).shape[1] > 1.5 * EXPORT_DPI   # about 2 in wide, trimmed
    assert window.statusBar().currentMessage() == f"Saved {path}"
    assert results_folder('', 'Plugin_Results') == ''
    window.close()


def test_options_panel_changes_the_figure_and_keeps_the_options(app):
    from honeychrome.view_components import figure_options as fo

    fig = Figure(figsize=(4, 3), layout='constrained')
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1])
    ax.grid(True)
    ax.set_title('Volcano')
    store = fo.FigureOptionsStore()
    ref = fo.FigureOptionsRef(store, 'test', 'volcano')
    widget = ScalableFigureWidget(fig, title='test', options=ref)
    widget.show()
    app.processEvents()
    assert not any(line.get_visible() for line in ax.get_xgridlines())
    panel = widget.options_panel
    panel.chk_grid.setChecked(True)
    panel.cb_title.setCurrentIndex(panel.cb_title.findData('custom'))
    panel.ed_title.setText('Figure 3')
    panel.ed_title.editingFinished.emit()
    panel.sizes['tick_size'].setValue(12.0)
    app.processEvents()
    assert any(line.get_visible() for line in ax.get_xgridlines())
    assert ax.title.get_text() == 'Figure 3'
    assert store.options('test', 'volcano').tick_size == 12.0
    panel.btn_reset.click()
    app.processEvents()
    assert ax.title.get_text() == 'Volcano' and not panel.chk_grid.isChecked()
    widget.close()
