"""
sample_table_import_dialog.py — review a sample table before importing it
========================================================================

Shared by the plugins that read groups and covariates from a table
(Automated Gating, Honeycluster). ``import_sample_table`` reads the file,
matches its rows to the plugin's samples (``sample_table.match_table``),
and shows what matched before anything changes: how many rows and samples
matched and how, rows that matched nothing, names that fit several samples,
and samples left without a row. The user picks the group column (or keeps
the current groups) and the covariate columns to import. Nothing is applied
when no row matches or the dialog is cancelled; each plugin writes the
returned values into its own state.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHeaderView, QLabel,
    QListWidget, QListWidgetItem, QMessageBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from honeychrome.controller_components import sample_table

_NO_GROUP = "None (keep the current groups)"
_MULTIPLE_COLOUR = '#fff3cd'
_LIST_LIMIT = 8


def _listed(names: list[str]) -> str:
    shown = ', '.join(names[:_LIST_LIMIT])
    return shown + (f" and {len(names) - _LIST_LIMIT} more" if len(names) > _LIST_LIMIT else '')


def _summary(match: sample_table.SampleTableMatch, n_samples: int) -> str:
    rows = match.matched_rows
    n_exact = sum(1 for i in rows if match.how.get(i) == sample_table.EXACT)
    n_inside = len(rows) - n_exact
    how = []
    if n_exact:
        how.append(f"{n_exact} by exact name")
    if n_inside:
        how.append(f"{n_inside} by the name inside the file name")
    return (f"<b>{len(rows)} of {len(match.names)}</b> table rows match "
            f"<b>{len(match.matched_samples)} of {n_samples}</b> samples ({', '.join(how)}).")


def _problems(match: sample_table.SampleTableMatch) -> str:
    lines = []
    if match.multiple:
        lines.append(f"Fit more than one sample (applied to each, highlighted below): "
                     f"{_listed(match.multiple)}.")
    if match.unmatched:
        lines.append(f"{len(match.unmatched)} row(s) match no sample and are skipped: "
                     f"{_listed(match.unmatched)}.")
    if match.samples_without_row:
        labels = [sample_table.sample_label(k) for k in match.samples_without_row]
        lines.append(f"{len(labels)} sample(s) have no row and keep their current values: "
                     f"{_listed(labels)}.")
    return '<br>'.join(lines)


def import_sample_table(parent, path, sample_keys: list[str], current_groups: dict,
                        title: str = "Import table") -> dict | None:
    """Read the table at *path*, show the matches for review and return the
    values to apply, or None when the file cannot be read, nothing matches or
    the user cancels.

    Returns {'groups': {key: group}, 'covariates': {column: {key: value}},
    'group_column': column or None, 'file': file name}.
    """
    try:
        df = sample_table.read_table_file(path)
        match = sample_table.match_table(df, sample_keys)
    except Exception as exc:
        QMessageBox.warning(parent, title, f"Could not read the table: {exc}")
        return None
    name = Path(path).name
    if not match.matched_rows:
        example = sample_table.sample_label(sample_keys[0]) if sample_keys else ''
        QMessageBox.warning(
            parent, title,
            f"None of the {len(match.names)} rows in {name} matches a sample, so nothing was "
            f"imported.\n\nA row matches a sample when its '{match.sample_column}' value is the "
            f"file name (with or without extension), or appears inside the file name between "
            f"spaces, underscores or dashes"
            + (f" (for example '{example}')" if example else '') + ". Names used this way "
            "need at least three characters including a digit.")
        return None

    dlg = QDialog(parent)
    dlg.setWindowTitle(f"{title}: {name}")
    dlg.setMinimumWidth(760)
    layout = QVBoxLayout(dlg)

    summary = QLabel(_summary(match, len(sample_keys)))
    summary.setWordWrap(True)
    layout.addWidget(summary)
    problems = _problems(match)
    if problems:
        warn = QLabel(problems)
        warn.setWordWrap(True)
        warn.setStyleSheet("color: #8a6d00;")
        layout.addWidget(warn)

    form = QFormLayout()
    group_combo = QComboBox()
    group_combo.addItem(_NO_GROUP, None)
    for col in df.columns:
        if col != match.sample_column:
            group_combo.addItem(str(col), col)
    default_group = sample_table.find_column(df, sample_table.GROUP_COLUMNS)
    if default_group is not None:
        group_combo.setCurrentIndex(group_combo.findData(default_group))
    group_combo.setToolTip("The column whose values become the samples' groups. Choose None to "
                           "import covariates only and keep the groups you have.")
    form.addRow("Group column:", group_combo)
    layout.addLayout(form)

    cov_label = QLabel("Covariates to import (columns holding no value for any matched sample "
                       "are left unticked):")
    cov_label.setWordWrap(True)
    layout.addWidget(cov_label)
    cov_list = QListWidget()
    cov_list.setMaximumHeight(160)
    layout.addWidget(cov_list)

    def fill_covariates():
        ticked_before = {cov_list.item(r).data(Qt.UserRole): cov_list.item(r).checkState()
                         for r in range(cov_list.count())}
        cov_list.clear()
        candidates = sample_table.covariate_candidates(df, match, group_combo.currentData())
        filled = set(sample_table.filled_columns(df, match, candidates))
        for col in candidates:
            text = str(col) if col in filled else f"{col}  (empty)"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, col)
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(ticked_before.get(
                col, Qt.Checked if col in filled else Qt.Unchecked))
            cov_list.addItem(item)

    group_combo.currentIndexChanged.connect(lambda _i: fill_covariates())
    fill_covariates()

    pairs = [(i, key) for i in match.matched_rows for key in match.rows[i]]
    preview = QTableWidget(len(pairs), 3)
    preview.setHorizontalHeaderLabels(["Table name", "Sample", "Matched"])
    preview.verticalHeader().setVisible(False)
    preview.setEditTriggers(QAbstractItemView.NoEditTriggers)
    multiple = set(match.multiple)
    for r, (i, key) in enumerate(pairs):
        cells = (match.names[i], sample_table.sample_label(key), match.how.get(i, ''))
        for c, text in enumerate(cells):
            item = QTableWidgetItem(text)
            if match.names[i] in multiple:
                item.setBackground(QColor(_MULTIPLE_COLOUR))
            if c == 1:
                item.setToolTip(key)
            preview.setItem(r, c, item)
    header = preview.horizontalHeader()
    header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
    header.setSectionResizeMode(1, QHeaderView.Stretch)
    header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
    layout.addWidget(preview, stretch=1)

    result: dict = {}

    def on_apply():
        group_col = group_combo.currentData()
        columns = [cov_list.item(r).data(Qt.UserRole) for r in range(cov_list.count())
                   if cov_list.item(r).checkState() == Qt.Checked]
        if group_col is None and not columns:
            QMessageBox.information(dlg, title, "Choose a group column or at least one covariate.")
            return
        values = sample_table.table_values(df, match, group_col, columns)
        changed = [k for k, g in values['groups'].items()
                   if current_groups.get(k) and current_groups.get(k) != g]
        if changed:
            reply = QMessageBox.question(
                dlg, title,
                f"This replaces the group of {len(changed)} sample(s) that already have one. "
                "Continue?", QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        result.update(values, group_column=group_col, file=name)
        dlg.accept()

    buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
    buttons.rejected.connect(dlg.reject)
    buttons.button(QDialogButtonBox.Apply).clicked.connect(on_apply)
    layout.addWidget(buttons)

    if dlg.exec() != QDialog.Accepted:
        return None
    return result


def describe_import(result: dict) -> str:
    """One line describing an applied import, for the status bar."""
    n_groups = len(result['groups'])
    filled = {c: len(v) for c, v in result['covariates'].items()}
    parts = []
    if result.get('group_column') is not None:
        parts.append(f"groups for {n_groups} sample(s) from '{result['group_column']}'")
    if filled:
        parts.append(f"{len(filled)} covariate(s) ("
                     + ', '.join(f"{c}: {n}" for c, n in filled.items()) + " samples)")
    return f"Imported {' and '.join(parts)} from {result['file']}."
