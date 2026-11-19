"""
test_copyable_table_widget.py
-----------------------------
Sorting of ``CopyableTableWidget`` and ``SortableTableItem``: cells that
show numbers sort as numbers, whether they hold numbers or formatted text.

    pytest tests/test_copyable_table_widget.py
"""

import pytest
from PySide6.QtCore import Qt

from honeychrome.view_components.copyable_table_widget import CopyableTableWidget, SortableTableItem


def _column(table, column: int = 0) -> list:
    return [table.item(r, column).text() for r in range(table.rowCount())]


def _sorted(texts: list, order=Qt.AscendingOrder) -> list:
    table = CopyableTableWidget([], [])
    table.setSortingEnabled(False)
    table.setColumnCount(1)
    table.setRowCount(len(texts))
    for r, text in enumerate(texts):
        table.setItem(r, 0, SortableTableItem(text))
    table.sortItems(0, order)
    return _column(table)


@pytest.mark.parametrize('texts, expected', [
    (['10', '2', '1'], ['1', '2', '10']),
    (['1,234', '999', '12,000,000'], ['999', '1,234', '12,000,000']),
    (['+0.52', '-1.30', '−2.00', '0'], ['−2.00', '-1.30', '0', '+0.52']),
    (['3.1e-05', '0.002', '1e-10'], ['1e-10', '3.1e-05', '0.002']),
    (['12.5%', '9%', '100.0%'], ['9%', '12.5%', '100.0%']),
    (['45°', '9°'], ['9°', '45°']),
    (['<0.001', '0.5', '0.01'], ['<0.001', '0.01', '0.5']),
    (['inf', '3', '-inf'], ['-inf', '3', 'inf']),
])
def test_text_that_reads_as_numbers_sorts_as_numbers(qapp, texts, expected):
    assert _sorted(texts) == expected
    assert _sorted(texts, Qt.DescendingOrder) == expected[::-1]


def test_numbers_come_before_text_and_empty_cells_last(qapp):
    assert _sorted(['beta', '', '10', 'alpha', 'nan', '2']) == ['2', '10', 'alpha', 'beta', '', 'nan']
    assert _sorted(['C10', 'C2', '1-2', '1.2.3']) == ['1-2', '1.2.3', 'C10', 'C2']


def test_rows_given_to_the_table_sort_by_value_or_text(qapp):
    rows = [{'name': 'b', 'n': 10, 'p': '0.5', 'events': '1,200'}, {'name': 'a', 'n': 2, 'p': '3e-04', 'events': '900'},
            {'name': 'c', 'n': 100, 'p': '0.01', 'events': '15,000'}]
    table = CopyableTableWidget(rows, ['name', 'n', 'p', 'events'])
    for column, expected in ((1, ['a', 'b', 'c']), (2, ['a', 'c', 'b']), (3, ['a', 'b', 'c']), (0, ['a', 'b', 'c'])):
        table.sortItems(column, Qt.AscendingOrder)
        assert _column(table) == expected
