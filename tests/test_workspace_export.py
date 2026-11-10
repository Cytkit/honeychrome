"""SVG nesting and file naming of the Workspace page export (drc_workspace_export)."""
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'honeychrome',
                                'bundled_plugins'))
pytest.importorskip('PySide6')
import drc_workspace_export as wx  # noqa: E402

SVG = ('<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg" '
       'xmlns:xlink="http://www.w3.org/1999/xlink" width="100pt" height="50pt" '
       'viewBox="0 0 100 50"><defs><clipPath id="c1"><rect width="10" height="10"/></clipPath>'
       '<path id="m1" d="M0 0"/></defs><g id="figure_1" clip-path="url(#c1)">'
       '<use xlink:href="#m1"/></g></svg>')


def test_nest_svg_prefixes_ids_and_places():
    nested = wx._nest_svg(SVG, 'i3_', 12.5, 20.0, 35.3, 17.6)
    root = ET.fromstring(nested)
    assert (root.get('x'), root.get('y'), root.get('width'), root.get('height')) == (
        '12.5000', '20.0000', '35.3000', '17.6000')
    assert root.get('viewBox') == '0 0 100 50'
    ids = {el.get('id') for el in root.iter() if el.get('id')}
    assert ids == {'i3_c1', 'i3_m1', 'i3_figure_1'}
    assert 'url(#i3_c1)' in nested and 'href="#i3_m1"' in nested


def test_page_path():
    assert wx._page_path(Path('/x/fig.svg'), 0, 1, '.svg') == Path('/x/fig.svg')
    assert wx._page_path(Path('/x/fig'), 1, 3, '.png') == Path('/x/fig_page2.png')
