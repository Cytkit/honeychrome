"""
suggest_groupings_dialog.py — review and choose groupings found in sample names
==============================================================================

Shared by the plugins that assign samples to comparison groups (Honeycluster,
Automated Gating, FlowCode Analysis). The detection itself is
``differential_stats.suggest_groupings_from_names``; this module turns its
suggestions into a list of choices, either automatically or through a dialog,
and each plugin writes the choices into its own state.

A choice is a dict:

* ``role``        — 'group', 'pairing' or 'covariate'
* ``name``        — column name for a pairing/covariate (``'Group'`` for the group)
* ``values``      — {sample id: value}
* ``field_name``  — the detected field, e.g. 'Name field 2 of 9 (filename)'
* ``examples``    — up to five distinct values
* ``n_distinct``  — number of distinct values
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QHeaderView, QLabel, QMessageBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout,
)

ROLE_LABELS = {'group': 'Group', 'pairing': 'Pairing/covariate', 'covariate': 'Covariate'}
_ROLE_KEYS = {label: key for key, label in ROLE_LABELS.items()}
_TITLE = "Suggest Groupings"


def default_column_name(suggestion: dict) -> str:
    """Default name for a suggestion: 'Group', or 'field_<position>'."""
    if suggestion['role_guess'] == 'group':
        return 'Group'
    return f"field_{suggestion['position'] + 1}"


def _choice(suggestion: dict, role: str, name: str) -> dict:
    return {'role': role, 'name': name, 'values': dict(suggestion['values']),
            'field_name': suggestion['field_name'], 'examples': list(suggestion['examples']),
            'n_distinct': int(suggestion['n_distinct'])}


def automatic_groupings(result: dict) -> list[dict]:
    """
    The choices applied without asking: the first suggestion guessed as the
    group and, when there is one, the first guessed as a pairing variable.
    Empty when no group was found. ``result`` is the return value of
    ``suggest_groupings_from_names``; its suggestions are already ordered
    group, pairing, covariate, fewest distinct values first.
    """
    suggestions = result.get('suggestions') or []
    group = next((s for s in suggestions if s['role_guess'] == 'group'), None)
    if group is None:
        return []
    chosen = [_choice(group, 'group', 'Group')]
    pairing = next((s for s in suggestions if s['role_guess'] == 'pairing'), None)
    if pairing is not None:
        chosen.append(_choice(pairing, 'pairing', default_column_name(pairing)))
    return chosen


def describe_groupings(chosen: list[dict]) -> str:
    """One line describing applied choices, for a status label."""
    parts = []
    for c in chosen:
        examples = ', '.join(c['examples'][:4]) + (', …' if c['n_distinct'] > 4 else '')
        field = c['field_name'].split(' (')[0].lower()
        if c['role'] == 'group':
            parts.append(f"Group = {field} ({examples})")
        elif c['role'] == 'pairing':
            parts.append(f"paired by '{c['name']}' = {field} ({c['n_distinct']} levels)")
        else:
            parts.append(f"covariate '{c['name']}' = {field}")
    return "Groups suggested from sample names: " + "; ".join(parts) + "."


def no_suggestion_message(result: dict) -> str:
    msg = "No repeated naming patterns were found across these sample names."
    if result.get('irregular_samples'):
        msg += (f"\n\n{len(result['irregular_samples'])} sample name(s) have an "
                "unusual structure and couldn't be compared.")
    return msg


def choose_groupings(parent, result: dict, confirm_group_overwrite: bool = True) -> list[dict] | None:
    """
    Show the suggestions for review: tick the ones to apply, change a role
    or a column name. Returns the choices on Apply, or None on Cancel.
    At most one row may be the group. With ``confirm_group_overwrite``, a
    group row asks before overwriting the samples' current groups.
    """
    suggestions = result['suggestions']

    dlg = QDialog(parent)
    dlg.setWindowTitle(_TITLE)
    dlg.setMinimumWidth(720)
    layout = QVBoxLayout(dlg)

    hint = QLabel("Repeated naming patterns found across the sample names. Tick the ones to "
                  "apply, adjust the role or column name if needed, then Apply.")
    hint.setWordWrap(True)
    layout.addWidget(hint)

    if result.get('irregular_samples'):
        warn = QLabel(f"{len(result['irregular_samples'])} sample(s) have a different naming "
                      "structure and are left out of every suggestion below.")
        warn.setWordWrap(True)
        warn.setStyleSheet("color: grey; font-style: italic;")
        layout.addWidget(warn)

    table = QTableWidget(len(suggestions), 5)
    table.setHorizontalHeaderLabels(['Use', 'Detected field', 'Role', 'Column / group name',
                                     'Example values'])
    header = table.horizontalHeader()
    for col in (1, 2, 3):
        header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
    header.setSectionResizeMode(4, QHeaderView.Stretch)
    table.verticalHeader().setVisible(False)

    for row, sug in enumerate(suggestions):
        chk = QTableWidgetItem()
        chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        chk.setCheckState(Qt.Checked if sug['role_guess'] in ('group', 'pairing') else Qt.Unchecked)
        table.setItem(row, 0, chk)
        field_item = QTableWidgetItem(sug['field_name'])
        field_item.setFlags(Qt.ItemIsEnabled)
        table.setItem(row, 1, field_item)
        role_combo = QComboBox()
        role_combo.addItems(list(ROLE_LABELS.values()))
        role_combo.setCurrentText(ROLE_LABELS[sug['role_guess']])
        table.setCellWidget(row, 2, role_combo)
        table.setItem(row, 3, QTableWidgetItem(default_column_name(sug)))
        examples_item = QTableWidgetItem(', '.join(sug['examples']))
        examples_item.setFlags(Qt.ItemIsEnabled)
        table.setItem(row, 4, examples_item)
    layout.addWidget(table)

    chosen: list[dict] = []

    def on_apply():
        picks = []
        for row, sug in enumerate(suggestions):
            if table.item(row, 0).checkState() != Qt.Checked:
                continue
            role = _ROLE_KEYS[table.cellWidget(row, 2).currentText()]
            name = table.item(row, 3).text().strip()
            if not name:
                QMessageBox.warning(dlg, _TITLE, "Column/group name cannot be blank.")
                return
            picks.append(_choice(sug, role, name))
        if sum(1 for c in picks if c['role'] == 'group') > 1:
            QMessageBox.warning(dlg, _TITLE,
                                "Only one detected field can be applied as the Group column at a time.")
            return
        if confirm_group_overwrite and any(c['role'] == 'group' for c in picks):
            reply = QMessageBox.question(
                dlg, _TITLE,
                "Applying a detected field as the Group column will overwrite every listed "
                "sample's current group, including manual selections. Continue?",
                QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        chosen.extend(picks)
        dlg.accept()

    buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
    buttons.rejected.connect(dlg.reject)
    buttons.button(QDialogButtonBox.Apply).clicked.connect(on_apply)
    layout.addWidget(buttons)

    if dlg.exec() != QDialog.Accepted:
        return None
    return chosen
