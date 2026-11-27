"""
test_focus_mode.py
------------------
Focus mode (view_components/focus_mode.py) hides the marked widgets and the
section headers of marked accordions that are showing, and shows exactly
those again. Runs off screen.
"""

import os

import pytest

pytest.importorskip('PySide6')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication, QLabel, QToolBox, QVBoxLayout, QWidget  # noqa: E402

from honeychrome.view_components import focus_mode  # noqa: E402
from honeychrome.view_components.help_toggle_widget import HelpToggleWidget  # noqa: E402


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def test_hides_marked_widgets_and_headers_and_restores_them(app):
    root = QWidget()
    lay = QVBoxLayout(root)
    help_widget = HelpToggleWidget(text='help')
    controls, already_hidden, figure = QLabel('controls'), QLabel('hidden by the page'), QLabel('figure')
    toolbox = QToolBox()
    toolbox.addItem(QLabel('one'), 'One')
    toolbox.addItem(QLabel('two'), 'Two')
    focus_mode.mark_focus_chrome(controls, already_hidden)
    focus_mode.mark_focus_headers(toolbox)
    for w in (help_widget, controls, already_hidden, figure, toolbox):
        lay.addWidget(w)
    root.show()
    already_hidden.hide()
    app.processEvents()
    headers = [b for b in toolbox.children() if b.objectName() == 'qt_toolbox_toolboxbutton']
    assert len(headers) == 2

    hidden = focus_mode.hide_chrome(root)
    assert help_widget.isHidden() and controls.isHidden() and all(h.isHidden() for h in headers)
    assert not figure.isHidden() and not toolbox.isHidden() and already_hidden not in hidden

    focus_mode.restore(hidden)
    assert not help_widget.isHidden() and not controls.isHidden() and not any(h.isHidden() for h in headers)
    assert already_hidden.isHidden()
    root.close()
