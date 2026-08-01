"""
drc_report.py — Report tab for the DR / Clustering / Statistics plugin
========================================================================
Companion module to ``dr_clustering_tab.py`` (filename intentionally does
NOT end in ``_tab.py``, so it is not picked up as a separate plugin tab —
same convention as drc_run_archive.py, drc_pipeline.py, etc.).

Provides:
  • ReportItem                — one tickable row (a plot and/or a table)
  • tables_from_maker_kwargs  — generic DataFrame extractor for
                                GroupsStatsTab's maker/maker_kwargs pattern
  • qtable_to_dataframe       — read a QTableWidget's displayed cells into
                                a DataFrame (used for the Cluster Label table)
  • sanitize_filename         — safe file/folder name component
  • build_settings_document   — plain-text "how to reproduce this analysis"
                                document, always generated
  • add_title_page / add_section_divider / add_dataframe_pages
                              — matplotlib PdfPages helpers (no new PDF
                                dependency -- matplotlib is already a
                                hard dependency of this plugin)
  • ReportTab(QWidget)        — the tab itself: three sections (Workspace,
                                Cluster Annotation, Stats), each populated
                                from that tab's own get_report_items().

Design notes
------------
Items reflect whatever is CURRENTLY rendered/computed in each source tab
(the currently open Workspace plot cards, the currently open Stats result
tabs, the currently selected Cluster Annotation run) -- there is no
separate "reporting run" selector here and no recomputation from scratch.
This keeps the Report tab a thin read-only consumer of the other tabs'
own state, with no duplicated pipeline logic and no new caches to keep in
sync. Folder/PDF naming and the settings document use whichever
clustering run is currently selected in Cluster Annotation, since that is
the one place in the plugin a clustering run selection is already
mandatory context.
"""

from __future__ import annotations

import math
import re
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QGroupBox,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QFrame,
    QAbstractItemView, QMessageBox, QApplication,
)

import drc_logging

log = drc_logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Report item
# ---------------------------------------------------------------------------

@dataclass
class ReportItem:
    """
    One tickable row in the Report tab.

    key         Stable identifier within its source tab (used to look the
                item back up after a refresh; the checkbox state itself
                is tracked by label -- see ReportTab._populate_section).
    tab         Source tab name -- 'Workspace' | 'Cluster Annotation' |
                'Stats' -- also the sub-folder these get exported into.
    label       Display text for the checkbox, and the exported file stem.
    get_figure  Optional zero-arg callable returning a fresh, canvas-less
                matplotlib Figure (or None on failure). Called once, at
                report-generation time, so it always reflects whatever is
                CURRENTLY computed in the source tab -- never a stale copy
                cached ahead of time.
    get_tables  Optional zero-arg callable returning {table_name: DataFrame}.
                An item can supply BOTH get_figure and get_tables (e.g. a
                results heatmap also exports its underlying results table).
    """
    key: str
    tab: str
    label: str
    get_figure: Callable[[], Any] | None = None
    get_tables: Callable[[], dict] | None = None


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def tables_from_maker_kwargs(maker_kwargs: dict) -> dict:
    """
    Pull every DataFrame value out of a GroupsStatsTab maker_kwargs dict,
    one level deep (covers state.pca_* results, nested one dict down as
    pca_result={'scores': df, 'loadings': df, ...}). Generic on purpose:
    a new result kind picks up CSV export for free as long as its
    maker_kwargs holds its DataFrame(s) directly or one dict deep, with
    no per-key special-casing needed here.
    """
    tables: dict = {}
    for k, v in maker_kwargs.items():
        if isinstance(v, pd.DataFrame):
            tables[k] = v
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, pd.DataFrame):
                    tables[f"{k}_{k2}"] = v2
    return tables


def qtable_to_dataframe(table_widget) -> pd.DataFrame:
    """
    Read a QTableWidget's current display text into a DataFrame -- used
    for the Cluster Annotation label table, whose values (name, colour
    swatch, counts, MEM label, suggested type) only exist as widget cell
    text, not as a standalone DataFrame anywhere in PipelineState.
    """
    n_cols = table_widget.columnCount()
    headers = []
    for c in range(n_cols):
        header_item = table_widget.horizontalHeaderItem(c)
        headers.append(header_item.text() if header_item is not None else f"col{c}")
    rows = []
    for r in range(table_widget.rowCount()):
        row = []
        for c in range(n_cols):
            item = table_widget.item(r, c)
            row.append(item.text() if item is not None else '')
        rows.append(row)
    return pd.DataFrame(rows, columns=headers)


_INVALID_FS_CHARS = re.compile(r'[<>:"/\\|?*]')


def sanitize_filename(text: str, fallback: str = 'item') -> str:
    """Make *text* safe to use as a file or folder name component."""
    text = (text or '').strip()
    text = _INVALID_FS_CHARS.sub('_', text)
    text = re.sub(r'\s+', '_', text)
    return text or fallback


# ---------------------------------------------------------------------------
# Settings document
# ---------------------------------------------------------------------------

def build_settings_document(
    state, controller,
    cl_run: dict | None,
    dr_run: dict | None,
    pval_threshold: float,
    fc_threshold: float,
) -> str:
    """
    Plain-text settings document -- everything needed to describe how to
    reproduce the analysis behind the report: samples, DR/clustering
    algorithm + hyperparameters, per-channel transforms, channel roles,
    stats settings, and group assignments. Always generated, regardless
    of which report items were ticked (see ReportTab.generate_report).
    """
    lines: list[str] = []

    def _section(title: str):
        lines.append('')
        lines.append(title)
        lines.append('=' * len(title))

    lines.append('Honeychrome DR/Clustering -- Analysis Settings')
    lines.append(f"Experiment:       {controller.experiment_dir}")
    lines.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    _section('Clustering Run')
    if cl_run:
        lines.append(f"Label:      {cl_run.get('label', '')}")
        lines.append(f"Algorithm:  {cl_run.get('algorithm', '')}")
        lines.append(f"Timestamp:  {cl_run.get('timestamp', '')}")
        lines.append(f"N clusters: {cl_run.get('n_clusters', '')}")
        gates = cl_run.get('gates', [])
        lines.append(f"Gate(s):    {', '.join(gates) if gates else '(none)'}")
        lines.append("Hyperparameters:")
        params = cl_run.get('params') or {}
        if params:
            for k, v in params.items():
                lines.append(f"    {k}: {v}")
        else:
            lines.append("    (none recorded)")
    else:
        lines.append('(no clustering run selected)')

    _section('DR Run (Cluster Map / embedding space)')
    if dr_run:
        lines.append(f"Label:      {dr_run.get('label', '')}")
        lines.append(f"Algorithm:  {dr_run.get('algorithm', '')}")
        lines.append(f"Timestamp:  {dr_run.get('timestamp', '')}")
        gates = dr_run.get('gates', [])
        lines.append(f"Gate(s):    {', '.join(gates) if gates else '(none)'}")
        lines.append("Hyperparameters:")
        params = dr_run.get('params') or {}
        if params:
            for k, v in params.items():
                lines.append(f"    {k}: {v}")
        else:
            lines.append("    (none recorded)")
    else:
        lines.append('(no DR run selected -- Cluster Map has no embedding, '
                      'or clustering ran directly on raw feature space)')

    _section('Samples')
    training = sorted(cl_run.get('training_sample_ids', [])) if cl_run else []
    lines.append(f"Samples used for training ({len(training)}):")
    if training:
        lines.extend(f"    {s}" for s in training)
    else:
        lines.append("    (none)")
    analysed = sorted((cl_run.get('labels') or {}).keys()) if cl_run else []
    lines.append(f"Samples used for analysis ({len(analysed)}):")
    if analysed:
        lines.extend(f"    {s}" for s in analysed)
    else:
        lines.append("    (none)")

    _section('Channels')
    cl_channels = cl_run.get('channels', []) if cl_run else []
    dr_channels = dr_run.get('channels', []) if dr_run else []
    lines.append(f"Channels used for clustering ({len(cl_channels)}):")
    if cl_channels:
        lines.extend(f"    {c}" for c in cl_channels)
    else:
        lines.append("    (none)")
    lines.append(f"Channels used for DR ({len(dr_channels)}):")
    if dr_channels:
        lines.extend(f"    {c}" for c in dr_channels)
    else:
        lines.append("    (none)")

    _section('Transform Settings per Channel')
    transform_params = state.channel_transform_params or {}
    if transform_params:
        lines.append(f"{'Channel':<24}{'W':>8}{'A':>8}{'T':>10}{'M':>8}")
        for ch in sorted(transform_params):
            p = transform_params[ch]
            lines.append(
                f"{ch:<24}{p.get('W', 0.0):>8.3f}{p.get('A', 0.0):>8.3f}"
                f"{p.get('T', 0.0):>10.1f}{p.get('M', 0.0):>8.3f}"
            )
    else:
        lines.append("    (no transform parameters recorded)")

    _section('Channels Used for Stats (Type / Activation)')
    stats_channels = sorted(set(cl_channels) | set(state.selected_channels))
    if stats_channels:
        lines.append(f"{'Channel':<24}{'Role':>28}")
        for ch in stats_channels:
            role = state.marker_roles.get(ch, 'state')
            role_label = 'Type (excluded)' if role == 'type' else 'Activation (included)'
            lines.append(f"{ch:<24}{role_label:>28}")
    else:
        lines.append("    (no channels recorded)")

    _section('Statistics Settings')
    lines.append(f"Contrast mode:     {state.contrast_mode}")
    if state.contrast_mode == 'reference':
        lines.append(f"Reference group:   {state.reference_group}")
    selection = state.testing_group_selection or state.group_names
    lines.append(f"Groups tested:     {', '.join(selection) if selection else '(none)'}")
    lines.append(f"Paired:            {state.paired}")
    if state.paired:
        lines.append(f"Pairing variable:  {state.pairing_variable}")
    lines.append(f"P-value threshold: {pval_threshold}")
    lines.append(f"Log2FC threshold:  {fc_threshold}")

    _section('Group Assignments')
    if state.sample_groups:
        lines.append(f"{'Sample':<50}{'Group':>16}")
        for sample in sorted(state.sample_groups):
            lines.append(f"{sample:<50}{state.sample_groups[sample]:>16}")
    else:
        lines.append("    (no samples assigned to a group)")

    if state.covariates is not None and not state.covariates.empty:
        _section('Covariates')
        lines.append(state.covariates.to_string())

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# PDF page helpers (matplotlib PdfPages -- no new dependency)
# ---------------------------------------------------------------------------

def add_title_page(pdf, title: str, subtitle_lines: list[str]) -> None:
    from matplotlib.figure import Figure
    fig = Figure(figsize=(8.27, 11.69))  # A4 portrait
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.text(0.5, 0.85, title, ha='center', va='top', fontsize=20, weight='bold',
             transform=ax.transAxes)
    y = 0.72
    for line in subtitle_lines:
        ax.text(0.5, y, line, ha='center', va='top', fontsize=11, transform=ax.transAxes)
        y -= 0.04
    pdf.savefig(fig)


def add_section_divider(pdf, title: str) -> None:
    from matplotlib.figure import Figure
    fig = Figure(figsize=(11.69, 2.0))
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.text(0.02, 0.5, title, ha='left', va='center', fontsize=18, weight='bold',
             transform=ax.transAxes)
    pdf.savefig(fig)


def _format_cell(value) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return '' if value is None else str(value)


def add_dataframe_pages(pdf, df, title: str, max_rows_per_page: int = 35) -> None:
    """Paginate *df* into one or more table pages of *pdf* (a
    matplotlib.backends.backend_pdf.PdfPages already open)."""
    if df is None or df.empty:
        return
    from matplotlib.figure import Figure

    df_display = df.reset_index() if df.index.name is not None else df.reset_index(drop=True)
    n_rows = len(df_display)
    n_pages = max(1, math.ceil(n_rows / max_rows_per_page))
    for page in range(n_pages):
        chunk = df_display.iloc[page * max_rows_per_page:(page + 1) * max_rows_per_page]
        formatted = chunk.map(_format_cell)
        page_title = title if n_pages == 1 else f"{title} (page {page + 1}/{n_pages})"

        fig = Figure(figsize=(11.69, 8.27))  # A4 landscape
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.set_title(page_title, fontsize=12, weight='bold', pad=12)
        table = ax.table(
            cellText=formatted.values.tolist(),
            colLabels=[str(c) for c in formatted.columns],
            loc='center', cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.3)
        pdf.savefig(fig)


# ---------------------------------------------------------------------------
# Report tab
# ---------------------------------------------------------------------------

_SECTIONS = ('Workspace', 'Cluster Annotation', 'Stats')


class ReportTab(QWidget):
    """
    Tab 6 -- Report
    -----------------------------------
    Three tick-list sections, one per source tab (Workspace, Cluster
    Annotation, Stats), each populated from that tab's own
    get_report_items(). "Generate Report" writes:
      • one PNG (figure items) and/or CSV (table items) per checked item,
        into a sub-folder named after its source tab
      • one combined PDF with a title page, a divider per section, every
        checked figure, and every checked table (paginated)
      • settings.txt -- always written, regardless of what's ticked
    into  experiment_dir / 'DR_Clustering_Reports' / '<run_label>_<timestamp>'.

    Reporting run: whichever clustering run is currently selected in the
    Cluster Annotation tab (the one place in the plugin a clustering run
    selection is already mandatory context) -- there is no separate run
    selector here.
    """

    def __init__(self, state, bus, controller, *, workspace_tab, groups_stats_tab,
                 cluster_annotation_tab, parent=None):
        super().__init__(parent)
        self.state = state
        self.bus = bus
        self.controller = controller
        self.workspace_tab = workspace_tab
        self.groups_stats_tab = groups_stats_tab
        self.cluster_annotation_tab = cluster_annotation_tab
        self._section_lists: dict[str, QListWidget] = {}
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        self.context_label = QLabel("Reporting on: (no clustering run selected)")
        self.context_label.setStyleSheet("font-weight: bold;")
        outer.addWidget(self.context_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll, stretch=1)

        content = QWidget()
        scroll.setWidget(content)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)

        for section_name in _SECTIONS:
            box = QGroupBox(section_name)
            box_layout = QVBoxLayout(box)

            btn_row = QHBoxLayout()
            all_btn = QPushButton("All")
            all_btn.setFixedWidth(50)
            none_btn = QPushButton("None")
            none_btn.setFixedWidth(50)
            btn_row.addWidget(all_btn)
            btn_row.addWidget(none_btn)
            btn_row.addStretch()
            box_layout.addLayout(btn_row)

            list_widget = QListWidget()
            list_widget.setSelectionMode(QAbstractItemView.NoSelection)
            box_layout.addWidget(list_widget)

            all_btn.clicked.connect(lambda checked=False, lw=list_widget: self._set_all_checked(lw, True))
            none_btn.clicked.connect(lambda checked=False, lw=list_widget: self._set_all_checked(lw, False))

            self._section_lists[section_name] = list_widget
            content_layout.addWidget(box)

        content_layout.addStretch()

        self.refresh_btn = QPushButton("⟳  Refresh Items")
        self.refresh_btn.clicked.connect(self.refresh)
        outer.addWidget(self.refresh_btn)

        self.output_label = QLabel("")
        self.output_label.setStyleSheet("color: grey;")
        self.output_label.setWordWrap(True)
        outer.addWidget(self.output_label)

        self.generate_btn = QPushButton("Generate Report")
        self.generate_btn.setFixedHeight(34)
        self.generate_btn.clicked.connect(self.generate_report)
        outer.addWidget(self.generate_btn)

    def _set_all_checked(self, list_widget: QListWidget, checked: bool):
        target_state = Qt.Checked if checked else Qt.Unchecked
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if item.flags() & Qt.ItemIsUserCheckable:
                item.setCheckState(target_state)

    # ------------------------------------------------------------------
    # Refresh (called on tab activation -- see PluginWidget._refresh_tab_at)
    # ------------------------------------------------------------------

    def refresh(self):
        cl_run = self.cluster_annotation_tab._selected_cluster_run()
        if cl_run:
            run_label = self.cluster_annotation_tab.run_combo.currentText()
            self.context_label.setText(f"Reporting on clustering run: {run_label}")
            safe_label = sanitize_filename(run_label)
            self.output_label.setText(
                f"Report folder: {self.controller.experiment_dir / 'DR_Clustering_Reports' / (safe_label + '_<timestamp>')}"
            )
        else:
            self.context_label.setText(
                "Reporting on: (no clustering run selected -- pick one in Cluster Annotation)"
            )
            self.output_label.setText("")
        self.generate_btn.setEnabled(cl_run is not None)

        self._populate_section('Workspace', self.workspace_tab.get_report_items())
        self._populate_section('Cluster Annotation', self.cluster_annotation_tab.get_report_items())
        self._populate_section('Stats', self.groups_stats_tab.get_report_items())

    def _populate_section(self, section_name: str, items: list):
        list_widget = self._section_lists[section_name]
        previously_checked = {
            list_widget.item(i).text()
            for i in range(list_widget.count())
            if list_widget.item(i).checkState() == Qt.Checked
        }
        list_widget.clear()
        if not items:
            placeholder = QListWidgetItem("(nothing to report yet)")
            placeholder.setFlags(Qt.ItemIsEnabled)
            list_widget.addItem(placeholder)
            return
        for item in items:
            list_item = QListWidgetItem(item.label)
            list_item.setFlags(list_item.flags() | Qt.ItemIsUserCheckable)
            checked = (item.label in previously_checked) if previously_checked else True
            list_item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            list_item.setData(Qt.UserRole, item)
            list_widget.addItem(list_item)

    def _checked_items(self) -> list:
        out = []
        for list_widget in self._section_lists.values():
            for i in range(list_widget.count()):
                list_item = list_widget.item(i)
                if list_item.checkState() == Qt.Checked and list_item.data(Qt.UserRole) is not None:
                    out.append(list_item.data(Qt.UserRole))
        return out

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate_report(self):
        cl_run = self.cluster_annotation_tab._selected_cluster_run()
        if cl_run is None:
            QMessageBox.warning(
                self, "Generate Report",
                "Select a clustering run in the Cluster Annotation tab first."
            )
            return

        items = self._checked_items()
        if not items:
            reply = QMessageBox.question(
                self, "Generate Report",
                "No items are checked -- the report will contain only the "
                "settings document. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        dr_run = self.cluster_annotation_tab._selected_dr_run()
        pval_threshold = self.groups_stats_tab.pval_spin.value()
        fc_threshold = self.groups_stats_tab.fc_spin.value()
        run_label = cl_run.get('label') or 'clustering_run'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        folder_name = f"{sanitize_filename(run_label)}_{timestamp}"
        report_dir = self.controller.experiment_dir / 'DR_Clustering_Reports' / folder_name

        log.info("generate_report: writing %d item(s) to %s", len(items), report_dir)

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            report_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = report_dir / f"{sanitize_filename(run_label)}_report.pdf"

            from matplotlib.backends.backend_pdf import PdfPages
            with PdfPages(str(pdf_path)) as pdf:
                add_title_page(
                    pdf, "Honeychrome DR/Clustering Report",
                    [
                        f"Experiment: {self.controller.experiment_dir}",
                        f"Clustering run: {run_label}",
                        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        f"Items included: {len(items)}",
                    ],
                )
                by_section: dict[str, list] = {name: [] for name in _SECTIONS}
                for item in items:
                    by_section.setdefault(item.tab, []).append(item)

                for section_name in _SECTIONS:
                    section_items = by_section.get(section_name, [])
                    if not section_items:
                        continue
                    self.bus.statusMessage.emit(f"ReportTab: adding {section_name} section")
                    add_section_divider(pdf, section_name)
                    subfolder = report_dir / sanitize_filename(section_name)
                    subfolder.mkdir(parents=True, exist_ok=True)
                    for item in section_items:
                        self._export_item(item, subfolder, pdf)

            settings_text = build_settings_document(
                self.state, self.controller, cl_run, dr_run, pval_threshold, fc_threshold,
            )
            (report_dir / 'settings.txt').write_text(settings_text)

            self.bus.statusMessage.emit(f"ReportTab: exported {report_dir}")
            self.bus.popupMessage.emit(f"Report generated: {report_dir}")

        except Exception as e:
            traceback.print_exc()
            self.bus.warningMessage.emit(f"Could not generate report: {e}")
        finally:
            QApplication.restoreOverrideCursor()

        self.refresh()

    def _export_item(self, item: ReportItem, subfolder, pdf):
        safe_label = sanitize_filename(item.label)
        if item.get_figure is not None:
            try:
                fig = item.get_figure()
            except Exception as e:
                fig = None
                log.warning("figure build failed for %r: %s", item.label, e)
            if fig is not None:
                fig.savefig(subfolder / f"{safe_label}.png", dpi=200, bbox_inches='tight')
                pdf.savefig(fig, bbox_inches='tight')
        if item.get_tables is not None:
            try:
                tables = item.get_tables() or {}
            except Exception as e:
                tables = {}
                log.warning("table build failed for %r: %s", item.label, e)
            for table_name, df in tables.items():
                if df is None:
                    continue
                csv_name = sanitize_filename(f"{item.label}_{table_name}")
                df.to_csv(subfolder / f"{csv_name}.csv")
                add_dataframe_pages(pdf, df, f"{item.label} -- {table_name}")