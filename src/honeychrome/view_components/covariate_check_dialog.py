"""
covariate_check_dialog.py — the Check table window
==================================================

Shared by the plugins with a groups-and-covariates table (Automated Gating,
Honeycluster). Lists the problems ``covariate_check.check_covariates`` finds,
one line per problem and column (or per comparison), each with its suggested
fix chosen in a drop-down. Apply fixes carries out the chosen fixes through
the plugin's callback and checks the table again, so the list always shows
what is left.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from honeychrome.controller_components import covariate_check as cc

_KIND_LABELS = {
    'blank': "Blank cells",
    'missing': "NA spellings",
    'unknown': "Unknown",
    'duplicate': "Similar categories",
    'convert': "Numbers",
    'unreadable': "Not numbers",
    'confounded': "Confounded",
    'left_out': "Left out",
}


def check_table(parent, title: str, check: Callable[[], list],
                apply: Callable[[list], str]) -> None:
    """Show the Check table window. *check()* returns the current Issues;
    *apply(fixes)* carries out [(Issue, action)] and returns a status line."""
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(900, 520)
    layout = QVBoxLayout(dlg)

    summary = QLabel()
    summary.setWordWrap(True)
    layout.addWidget(summary)

    table = QTableWidget(0, 4)
    table.setHorizontalHeaderLabels(["Problem", "Column / comparison", "Details", "Fix"])
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setWordWrap(True)
    header = table.horizontalHeader()
    header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
    header.setSectionResizeMode(1, QHeaderView.Interactive)
    header.setSectionResizeMode(2, QHeaderView.Stretch)
    header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
    table.setColumnWidth(1, 200)
    layout.addWidget(table, stretch=1)

    status = QLabel()
    status.setWordWrap(True)
    status.setStyleSheet("color: grey; font-style: italic;")
    layout.addWidget(status)

    row = QHBoxLayout()
    btn_apply = QPushButton("Apply fixes")
    btn_close = QPushButton("Close")
    row.addStretch()
    row.addWidget(btn_apply)
    row.addWidget(btn_close)
    layout.addLayout(row)

    current: list = []
    combos: list = []

    def fill():
        current[:] = check()
        combos.clear()
        table.setRowCount(len(current))
        for r, issue in enumerate(current):
            table.setItem(r, 0, QTableWidgetItem(_KIND_LABELS.get(issue.kind, issue.kind)))
            where = QTableWidgetItem(issue.column or issue.comparison)
            where.setToolTip(issue.column or issue.comparison)
            table.setItem(r, 1, where)
            table.setItem(r, 2, QTableWidgetItem(issue.summary))
            if issue.actions == [cc.NO_ACTION]:
                text = "Nothing needed" if issue.kind == 'left_out' else "Edit by hand"
                table.setCellWidget(r, 3, None)
                table.setItem(r, 3, QTableWidgetItem(text))
                combos.append(None)
                continue
            combo = QComboBox()
            for action in issue.actions:
                combo.addItem(cc.ACTION_LABELS[action], action)
            table.setCellWidget(r, 3, combo)
            combos.append(combo)
        table.resizeRowsToContents()
        fixable = sum(1 for c in combos if c is not None)
        if not current:
            summary.setText("No problems found.")
        else:
            summary.setText(
                f"{len(current)} item(s) found; {fixable} can be fixed here. Choose a fix for "
                "each, then Apply fixes. Items marked 'Edit by hand' need the values changed "
                "in the table.")
        btn_apply.setEnabled(fixable > 0)

    def on_apply():
        fixes = [(issue, combo.currentData()) for issue, combo in zip(current, combos)
                 if combo is not None and combo.currentData() != cc.NO_ACTION]
        if not fixes:
            status.setText("No fixes chosen.")
            return
        status.setText(apply(fixes))
        fill()

    btn_apply.clicked.connect(on_apply)
    btn_close.clicked.connect(dlg.accept)
    fill()
    dlg.exec()
