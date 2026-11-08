"""
figure_options_panel.py — the side panel that sets a figure's options
===================================================================

``FigureOptionsPanel`` edits the options of one figure
(``figure_options.FigureOptionsRef``): grid lines, run label, title (as
drawn, hidden or custom text), panel titles, font sizes of the title, axis
and panel labels, tick labels, legend and other text, the background and,
for figures that offer it (volcano plots), the x and y axis ranges.
Every change is kept with the experiment at once; whoever shows the figure
re-applies the options when the store reports a change.
"""

from __future__ import annotations

from PySide6.QtCore import QLocale
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QToolButton, QVBoxLayout, QWidget,
)

from honeychrome.view_components.figure_options import AXIS_RANGE, FigureOptions, FigureOptionsRef

__all__ = ['FigureOptionsPanel', 'options_toggle_button', 'PANEL_WIDTH']

PANEL_WIDTH = 250
_TITLE_MODES = (('As drawn', 'auto'), ('Hidden', 'hidden'), ('Custom text', 'custom'))
_BACKGROUNDS = (('Theme (black in dark mode, white in light)', 'theme'), ('White', 'white'),
                ('Black', 'black'), ('Transparent', 'transparent'))
_SIZES = (('title_size', 'Title'), ('label_size', 'Axis and panel labels'),
          ('tick_size', 'Tick labels'), ('legend_size', 'Legend'), ('text_size', 'Other text'))


def options_toggle_button(panel: 'FigureOptionsPanel') -> QToolButton:
    """A checkable "Options" button that shows and hides *panel*."""
    button = QToolButton()
    button.setText("⚙ Options")
    button.setCheckable(True)
    button.setToolTip("Show or hide this figure's options: grid lines, titles, run label, font "
                      "sizes and background. They are kept with the experiment and used for "
                      "Pop Out, Export and reports too.")
    button.toggled.connect(panel.setVisible)
    panel.setVisible(False)
    return button


class FigureOptionsPanel(QWidget):
    """Controls for the options of the figure *ref* names. With
    *axis_range* the panel also sets the x and y axis ranges
    (see figure_options.has_axis_range)."""

    def __init__(self, ref: FigureOptionsRef, parent=None, axis_range: bool = False):
        super().__init__(parent)
        self.ref = ref
        self._loading = False
        self.setFixedWidth(PANEL_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(QLabel("<b>Figure options</b>"))
        note = QLabel("This figure; kept with the experiment and used for Pop Out, Export and "
                      "reports.")
        note.setWordWrap(True)
        note.setStyleSheet("color: grey; font-size: 8pt;")
        layout.addWidget(note)

        self.chk_grid = QCheckBox("Grid lines")
        self.chk_grid.setToolTip("Grid lines behind the data. Threshold lines (e.g. on volcano "
                                 "plots) are part of the figure and always shown.")
        self.chk_run_label = QCheckBox("Run label (top left)")
        self.chk_run_label.setToolTip("The label naming the run the figure came from.")
        self.chk_panel_titles = QCheckBox("Panel titles")
        self.chk_panel_titles.setToolTip("Titles of the individual panels of a figure with "
                                         "several.")
        for w in (self.chk_grid, self.chk_run_label, self.chk_panel_titles):
            layout.addWidget(w)
            w.toggled.connect(self._changed)

        layout.addWidget(QLabel("Title:"))
        self.cb_title = QComboBox()
        for label, value in _TITLE_MODES:
            self.cb_title.addItem(label, value)
        self.cb_title.currentIndexChanged.connect(self._changed)
        layout.addWidget(self.cb_title)
        self.ed_title = QLineEdit()
        self.ed_title.setPlaceholderText("Title text")
        self.ed_title.editingFinished.connect(self._changed)
        layout.addWidget(self.ed_title)

        layout.addWidget(QLabel("Font sizes (pt):"))
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        self.sizes: dict[str, QDoubleSpinBox] = {}
        for name, label in _SIZES:
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 40.0)
            spin.setSingleStep(0.5)
            spin.setDecimals(1)
            spin.setSpecialValueText("As drawn")
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self._changed)
            form.addRow(label, spin)
            self.sizes[name] = spin
        layout.addLayout(form)

        layout.addWidget(QLabel("Background:"))
        self.cb_background = QComboBox()
        for label, value in _BACKGROUNDS:
            self.cb_background.addItem(label, value)
        self.cb_background.currentIndexChanged.connect(self._changed)
        layout.addWidget(self.cb_background)

        self.ranges: dict[str, QLineEdit] = {}
        if axis_range:
            range_label = QLabel("Axis range (blank = as drawn):")
            range_label.setToolTip("The range of the x and y axes. Leave an end blank to keep it as "
                                   "drawn (on FlowCode volcanos, the scale shared by every "
                                   "comparison). Points beyond the range are not shown. A minimum "
                                   "that is not below its maximum is ignored.")
            layout.addWidget(range_label)
            range_form = QFormLayout()
            range_form.setContentsMargins(0, 0, 0, 0)
            validator = QDoubleValidator(self)
            validator.setNotation(QDoubleValidator.Notation.StandardNotation)
            validator.setLocale(QLocale.c())
            for axis in ('x', 'y'):
                row = QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                for end in ('min', 'max'):
                    edit = QLineEdit()
                    edit.setPlaceholderText(end)
                    edit.setValidator(validator)
                    edit.editingFinished.connect(self._changed)
                    row.addWidget(edit)
                    self.ranges[f'{axis}_{end}'] = edit
                    if end == 'min':
                        row.addWidget(QLabel("to"))
                range_form.addRow(f"{axis.upper()}:", row)
            layout.addLayout(range_form)

        buttons = QHBoxLayout()
        self.btn_all = QPushButton("Use for all")
        self.btn_all.setToolTip("Make these options (all but the title and axis range) the "
                                "defaults of every figure of this plugin in this experiment.")
        self.btn_all.clicked.connect(self._use_for_all)
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setToolTip("Back to this plugin's defaults, title and axis range as drawn.")
        self.btn_reset.clicked.connect(self._reset)
        buttons.addWidget(self.btn_all)
        buttons.addWidget(self.btn_reset)
        layout.addLayout(buttons)
        layout.addStretch()

        self.load()
        self.ref.store.subscribe(self._on_store_changed)
        self.destroyed.connect(lambda _obj=None, store=ref.store, cb=self._on_store_changed:
                               store.unsubscribe(cb))

    def load(self):
        """Show the figure's current options."""
        opts = self.ref.options()
        self._loading = True
        try:
            self.chk_grid.setChecked(opts.grid)
            self.chk_run_label.setChecked(opts.run_label)
            self.chk_panel_titles.setChecked(opts.panel_titles)
            self.cb_title.setCurrentIndex(max(0, self.cb_title.findData(opts.title)))
            self.ed_title.setText(opts.title_text)
            self.ed_title.setEnabled(opts.title == 'custom')
            for name, spin in self.sizes.items():
                spin.setValue(float(getattr(opts, name)))
            self.cb_background.setCurrentIndex(max(0, self.cb_background.findData(opts.background)))
            for name, edit in self.ranges.items():
                edit.setText(getattr(opts, name))
        finally:
            self._loading = False

    def current(self) -> FigureOptions:
        """The options the controls show."""
        return FigureOptions(
            grid=self.chk_grid.isChecked(), run_label=self.chk_run_label.isChecked(),
            title=self.cb_title.currentData() or 'auto', title_text=self.ed_title.text(),
            panel_titles=self.chk_panel_titles.isChecked(),
            background=self.cb_background.currentData() or 'theme',
            **{name: float(spin.value()) for name, spin in self.sizes.items()},
            **{name: (self.ranges[name].text().strip() if name in self.ranges
                      else getattr(self.ref.options(), name)) for name in AXIS_RANGE})

    def _changed(self, *_args):
        if self._loading:
            return
        self.ed_title.setEnabled(self.cb_title.currentData() == 'custom')
        self.ref.store.set(self.ref.scope, self.ref.key, self.current())

    def _use_for_all(self):
        self.ref.store.set_defaults(self.ref.scope, self.current())

    def _reset(self):
        self.ref.store.reset(self.ref.scope, self.ref.key)

    def _on_store_changed(self, scope: str):
        if scope == self.ref.scope:
            self.load()
