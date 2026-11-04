"""
test_drc_report_tables.py
-------------------------
Page titles and pagination of the DataFrame tables written to the report PDF
by drc_report.add_dataframe_pages.

Usage:
    pytest tests/test_drc_report_tables.py
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / 'src' / 'honeychrome' / 'bundled_plugins'
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

drc_report = pytest.importorskip('drc_report')


class _Pages:
    def __init__(self):
        self.figures = []

    def savefig(self, fig):
        self.figures.append(fig)


def test_table_page_title_drops_dataframe_suffix():
    assert drc_report.table_page_title('MFI Clusters', 'results_df') == 'MFI Clusters results'
    assert drc_report.table_page_title('Frequencies', 'pca_scores') == 'Frequencies pca scores'


def test_long_text_table_keeps_title_clear_of_table():
    rows = [{'feature': f'CD45-BV480 | Cluster {i} : Neutrophil-like CD11b+ Ly6G+ ({i})',
             'cluster': f'Cluster {i} : Neutrophil-like CD11b+ Ly6G+ ({i})',
             'logFC': 0.5 * i}
            for i in range(60)]
    pages = _Pages()
    drc_report.add_dataframe_pages(pages, pd.DataFrame(rows), 'MFI Clusters results')
    assert len(pages.figures) > 1
    fig = pages.figures[0]
    title_text = fig.texts[0]
    assert title_text.get_text().startswith('MFI Clusters results (page 1/')
    axes = fig.axes[0]
    # the title band lies wholly above the axes holding the table
    title_y = title_text.get_position()[1]
    assert title_y > axes.get_position().y1


def test_empty_table_writes_no_pages():
    pages = _Pages()
    drc_report.add_dataframe_pages(pages, pd.DataFrame(), 'empty')
    assert pages.figures == []


def test_export_report_item_leaves_large_tables_out_of_the_pdf(tmp_path):
    tables = {'long': pd.DataFrame({'a': range(12)}), 'wide': pd.DataFrame([list(range(5))]),
              'small': pd.DataFrame({'a': [1, 2]})}
    item = drc_report.ReportItem(key='k', tab='Samples', label='Item',
                                 get_tables=lambda: tables)
    pages = _Pages()
    drc_report.export_report_item(item, tmp_path, pages, max_pdf_rows=10, max_pdf_columns=4)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['Item_long.csv', 'Item_small.csv',
                                                          'Item_wide.csv']
    notes = [t.get_text() for fig in pages.figures for ax in fig.axes for t in ax.texts]
    assert any('12 rows' in t and 'Item_long.csv' in t for t in notes)
    assert any('5 columns' in t and 'Item_wide.csv' in t for t in notes)
    assert any(fig.texts and fig.texts[0].get_text().startswith('Item small')
               for fig in pages.figures)
