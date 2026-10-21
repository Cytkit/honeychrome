"""Workspace canvas: figures are rendered only when they are on screen."""
import sys
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / 'src' / 'honeychrome' / 'bundled_plugins'
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

import drc_workspace_canvas as wc  # noqa: E402


def _figure_item(scene, uid, page):
    fig = Figure(figsize=(60 / wc.MM_PER_IN, 60 / wc.MM_PER_IN))
    FigureCanvasAgg(fig)
    fig.add_subplot(111).plot([0, 1], [0, 1])
    source = wc.SourceFigure(uid, {'tab': 'test', 'key': uid, 'label': uid})
    source.set_figure(fig, False, 100.0)
    item = wc.SourceFigureItem(source)
    scene.addItem(item)
    item.layout_mm = {'page': page, 'x_mm': 10.0, 'y_mm': 10.0}
    scene.position_item(item)
    return item


def _workspace(qtbot):
    scene = wc.WorkspaceScene([dict(wc.DEFAULT_PAGE)] * 2)
    view = wc.WorkspaceView(scene)
    qtbot.addWidget(view)
    view.resize(800, 600)
    view.show()
    qtbot.waitExposed(view)
    on_screen = _figure_item(scene, 'page1', page=0)
    off_screen = _figure_item(scene, 'page2', page=1)
    return view, scene, on_screen, off_screen


def _settle(qtbot):
    qtbot.wait(wc.RENDER_DELAY_MS * 3)


def test_only_the_figure_on_screen_is_rendered(qtbot):
    view, _scene, on_screen, off_screen = _workspace(qtbot)
    view.fit_page(0)
    qtbot.waitUntil(lambda: on_screen._image is not None)
    _settle(qtbot)

    assert off_screen._image is None


def test_zoom_rerenders_only_the_figure_on_screen(qtbot):
    view, scene, on_screen, off_screen = _workspace(qtbot)
    view.fit_page(0)
    qtbot.waitUntil(lambda: on_screen._image is not None)
    width_before = on_screen._image.width()

    view.set_zoom(view.zoom() * 2)
    ox, oy = scene.page_origin(0)
    view.centerOn(ox + 40, oy + 40)
    qtbot.waitUntil(lambda: on_screen._image.width() > width_before)
    _settle(qtbot)

    assert off_screen._image is None


def test_figure_is_rendered_when_scrolled_into_view(qtbot):
    view, scene, on_screen, off_screen = _workspace(qtbot)
    view.fit_page(0)
    qtbot.waitUntil(lambda: on_screen._image is not None)
    _settle(qtbot)
    assert off_screen._image is None

    ox, oy = scene.page_origin(1)
    view.centerOn(ox + 40, oy + 40)
    qtbot.waitUntil(lambda: off_screen._image is not None)
