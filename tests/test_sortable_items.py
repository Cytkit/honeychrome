"""
test_sortable_items.py
----------------------
The table cells with explicit sort keys (Honeycluster's run table and the
spectral reference library): cells sort by their key, and cells without a
key, or with keys that cannot be compared, sort by their text.

    pytest tests/test_sortable_items.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'honeychrome' / 'bundled_plugins'))


def _run_table_item(text, key=None):
    import dr_clustering_tab as drc

    item = drc._SortableItem(text)
    if key is not None:
        item.set_sort_key(key)
    return item


def _library_item(text, key=None):
    import honeychrome.view_components.reference_library_widget as rl

    item = rl._SortableItem(text)
    if key is not None:
        item.setData(rl._SORT_ROLE, key)
    return item


@pytest.mark.parametrize('make', [_run_table_item, _library_item])
def test_cells_sort_by_key_and_fall_back_on_text(qapp, make):
    assert make('2', 2) < make('10', 10) and not make('10', 10) < make('2', 2)
    assert make('alpha') < make('beta') and not make('beta') < make('alpha')
    assert make('alpha', 5) < make('beta') and not make('beta') < make('alpha', 5)


def test_run_table_keys_that_cannot_be_compared_sort_by_text(qapp):
    assert _run_table_item('alpha', 5) < _run_table_item('beta', 'x')
