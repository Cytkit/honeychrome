"""Pure layout helpers of the Workspace page canvas (drc_workspace_canvas)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'honeychrome',
                                'bundled_plugins'))
pytest.importorskip('PySide6')
import drc_workspace_canvas as wc  # noqa: E402

A4 = {'preset': 'A4', 'w_mm': 210.0, 'h_mm': 297.0}


def test_snap():
    assert wc.snap(13.3, 1.0) == 13.0
    assert wc.snap(27.7, 5.0) == 30.0
    assert wc.snap(0.74, 0.5) == 0.5


def test_page_origins_and_index():
    pages = [A4, dict(A4, w_mm=297.0, h_mm=210.0)]
    assert wc.page_origins(pages) == [(0.0, 0.0), (0.0, 297.0 + wc.PAGE_GAP_MM)]
    assert wc.page_index_at(pages, 100.0) == 0
    assert wc.page_index_at(pages, 297.0 + 5.0) == 0          # in the gap below page 1
    assert wc.page_index_at(pages, 297.0 + wc.PAGE_GAP_MM + 1.0) == 1
    assert wc.page_index_at(pages, 5000.0) == 1


def test_place_items_two_columns_with_gaps():
    pages, lays = wc.place_items([A4], [], [(82.6, 60.0)] * 3)
    assert pages == [A4]
    assert [(l['x_mm'], l['y_mm']) for l in lays] == [(10.0, 10.0), (100.0, 10.0), (10.0, 75.0)]


def test_place_items_adds_page_and_respects_placed():
    placed = [{'page': 0, 'x_mm': 10.0, 'y_mm': 10.0, 'w_mm': 190.0, 'h_mm': 277.0}]
    pages, lays = wc.place_items([A4], placed, [(50.0, 50.0)])
    assert len(pages) == 2 and lays == [{'page': 1, 'x_mm': 10.0, 'y_mm': 10.0}]


def test_place_items_oversized_gets_own_page():
    pages, lays = wc.place_items([A4], [], [(300.0, 400.0), (50.0, 50.0)])
    assert lays[0] == {'page': 0, 'x_mm': 10.0, 'y_mm': 10.0}
    assert lays[1]['page'] == 1


def test_page_size_for():
    assert wc.page_size_for('A4', 'Landscape', (0, 0)) == (297.0, 210.0)
    assert wc.page_size_for('A4', 'Portrait', (0, 0)) == (210.0, 297.0)
    assert wc.page_size_for('Custom', 'Portrait', (300.0, 100.0)) == (100.0, 300.0)


def test_locked_size():
    assert wc.locked_size(60.0, 60.0, 90.0, 40.0, 'right') == (90.0, 90.0)
    assert wc.locked_size(80.0, 40.0, 70.0, 60.0, 'bottom') == (120.0, 60.0)
    # corner: the larger relative change wins
    assert wc.locked_size(60.0, 60.0, 66.0, 90.0, 'corner') == (90.0, 90.0)


def test_letters_and_reading_order():
    assert wc.letter_sequence(4, {'B'}) == ['A', 'C', 'D', 'E']
    assert wc.letter_sequence(28, set())[-2:] == ['AA', 'AB']
    # two rows (within 10 mm), each left to right
    positions = [(100.0, 12.0), (10.0, 80.0), (10.0, 10.0), (100.0, 82.0)]
    assert wc.reading_order(positions) == [2, 0, 1, 3]


def test_align_moves():
    rects = {'a': (10.0, 10.0, 50.0, 40.0), 'b': (80.0, 30.0, 30.0, 20.0)}
    assert wc.align_moves(rects, 'left') == {'a': (0.0, 0.0), 'b': (-70.0, 0.0)}
    assert wc.align_moves(rects, 'bottom') == {'a': (0.0, 0.0), 'b': (0.0, 0.0)}
    assert wc.align_moves(rects, 'top') == {'a': (0.0, 0.0), 'b': (0.0, -20.0)}
    three = {'a': (0.0, 0.0, 10.0, 5.0), 'b': (15.0, 0.0, 10.0, 5.0), 'c': (60.0, 0.0, 10.0, 5.0)}
    moves = wc.align_moves(three, 'hdistribute')
    assert moves['a'] == (0.0, 0.0) and moves['c'] == (0.0, 0.0)
    assert moves['b'] == pytest.approx((15.0, 0.0))     # gaps 20 mm each side


def test_detached_copy():
    from matplotlib.figure import Figure
    fig = Figure(figsize=(4.0, 3.0))
    fig.add_subplot(111).plot([0, 1], [0, 1])
    fig._hover_handler = lambda event: None          # left out of the copy, kept on the original
    copy, as_image = wc.detached_copy(fig)
    assert not as_image and copy is not fig and callable(fig._hover_handler)
    copy.set_size_inches(2.0, 1.0)
    assert tuple(fig.get_size_inches()) == (4.0, 3.0)
    fig._odd = [lambda: 1]                           # cannot be pickled: copied as an image
    copy, as_image = wc.detached_copy(fig)
    assert as_image and tuple(copy.get_size_inches()) == (4.0, 3.0) and copy.axes[0].images


def test_grid_display_step():
    # drawn lines are snap positions, >= 6 screen px apart, in step with the 10 mm lines
    assert wc.grid_display_step(5.0, 2.61) == 5.0
    assert wc.grid_display_step(1.0, 2.61) == 5.0
    assert wc.grid_display_step(2.0, 2.61) == 10.0
    assert wc.grid_display_step(0.5, 2.61) == 2.5
    assert wc.grid_display_step(1.0, 10.0) == 1.0
    assert wc.grid_display_step(2.0, 4.0) == 2.0
    assert wc.grid_display_step(1.0, 0.2) == 50.0
