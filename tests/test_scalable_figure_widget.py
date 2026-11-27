"""
test_scalable_figure_widget.py
------------------------------
Exports from ScalableFigureWidget (view_components/scalable_figure_widget.py):
the figure is saved at its natural size whatever size the slider shows,
without hover labels, and is shown at the slider's size again afterwards;
Export Graphic writes into the results folder at EXPORT_DPI and reports the
file in the main window's status bar; the Fit menu sizes the figure to the
panel's width, height or both, and is remembered by its key. Runs off screen.

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
    EXPORT_DPI, FIT_HEIGHT, FIT_OFF, FIT_WHOLE, FIT_WIDTH, ScalableFigureWidget, results_folder,
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


def test_fit_sizes_the_figure_to_the_panel_and_is_remembered(app, monkeypatch):
    from PySide6.QtCore import QSettings
    from honeychrome.view_components import scalable_figure_widget as sfw
    monkeypatch.setattr(sfw, '_FIT_SETTINGS', ('honeychrome_tests', 'figure_fit'))
    QSettings(*sfw._FIT_SETTINGS).clear()
    fig = Figure(figsize=(4, 8), dpi=100)   # 400 x 800 px, twice as tall as wide
    fig.add_subplot(111).plot([0, 1], [0, 1])
    widget = ScalableFigureWidget(fig, title='tall', fit=FIT_WHOLE, fit_key='test/tall')
    widget.resize(1200, 600)
    widget.show()
    app.processEvents()
    scroll = widget._scroll
    widget._apply_scale()
    whole = widget.scale_percent()
    assert int(800 * whole / 100) <= scroll.height() and int(400 * whole / 100) <= scroll.width()
    assert not widget.size_slider.isEnabled()

    widget.set_fit_mode(FIT_WIDTH)
    assert widget.scale_percent() > whole          # the full width needs a larger figure than the whole
    widget.set_fit_mode(FIT_HEIGHT)
    assert widget.scale_percent() == pytest.approx(whole, abs=5)
    widget.set_fit_mode(FIT_OFF)
    assert widget.size_slider.isEnabled()
    widget.set_fit_mode(FIT_WIDTH)
    widget.close()

    again = ScalableFigureWidget(Figure(figsize=(4, 8)), title='tall', fit=FIT_WHOLE, fit_key='test/tall')
    assert again.fit_mode() == FIT_WIDTH
    assert ScalableFigureWidget(Figure(), title='plain').fit_mode() == FIT_OFF
    again.close()
    QSettings(*sfw._FIT_SETTINGS).clear()


def test_extra_controls_and_the_note_while_popped_out(app):
    from PySide6.QtWidgets import QDialog, QPushButton
    fig = Figure(figsize=(3, 2))
    fig.add_subplot(111).plot([0, 1], [0, 1])
    widget = ScalableFigureWidget(fig, title='plot', pop_out=lambda: None)
    button = QPushButton('Export Values (CSV)')
    widget.add_control(button)
    widget.show()
    app.processEvents()
    assert button.isVisible()
    dialog = QDialog()
    widget.hold_while_open(dialog, 'open elsewhere')
    app.processEvents()
    assert not widget._scroll.isVisible() and widget._held_label.text() == 'open elsewhere'
    dialog.deleteLater()
    for _ in range(3):
        app.processEvents()
    from PySide6.QtCore import QCoreApplication, QEvent
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    assert widget._scroll.isVisible() and not widget._held_label.isVisible()
    widget.close()
