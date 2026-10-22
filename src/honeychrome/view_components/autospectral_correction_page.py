"""
autospectral_correction_page.py
-------------------------------
AutoSpectral tab page: Spectral Correction (Fix My Unmix).

Sections
--------
1  Setup: what the other pages provide (AF profiles, spillover spread,
   spectral variants, the experiment's unmixing method).
2  Samples: the fully stained samples (default: every sample that is not a
   bead, unstained or single-stain control; each of those types offered with
   an Include tick box), run together or per raw
   subfolder, each group's unstained sample (Auto: scatter match), gate,
   events and seed; Advanced Settings; Fix Unmix; whether the experiment is
   unmixed with the corrected spectra.
3  Results: per group, the spectral change heatmap, before/after traces and
   the table of changed fluorophores with their Active ticks.
4  Comparison: the current sample with the base and the corrected spectra,
   both by the experiment's method.

The numerics live in controller_components/fix_my_unmix.py and the
experiment-level run and storage in spectral_correction.py.
"""

from __future__ import annotations

import logging

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QObject, QSettings, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QHeaderView, QAbstractItemView,
)

from honeychrome.controller_components import fix_my_unmix as fm
from honeychrome.controller_components import spectral_correction as sc
from honeychrome.controller_components import unmix_method
from honeychrome.controller_components.cytometer_whitelist import get_detector_laser_map, LASER_LABEL_COLORS
from honeychrome.controller_components.functions import resolve_base_gate
from honeychrome.controller_components.job_gate import acquire_job, guard_while_busy, wait_if_busy
from honeychrome.controller_components.gls_unmix import fluor_names_and_spectra, noise_in_use, spectra_fingerprint
from honeychrome.controller_components.particle_type import scatter_ceilings, scatter_channel_pair
from honeychrome.controller_components.sample_loader import af_profile_names, resolve_af_for_profiles
from honeychrome.controller_components.spread_model import raw_layout, spread_is_current
from honeychrome.view_components.autospectral_tab import (
    TAB_NAME, AfComparisonPlotWidget, pick_most_affected_channels,
)
from honeychrome.view_components.help_texts import autospectral_correction_help_text
from honeychrome.view_components.help_toggle_widget import HelpToggleWidget, WheelBlocker
from honeychrome.view_components.profiles_viewer import BottomAxisVerticalTickLabels, TransparentPlotWidget

logger = logging.getLogger(__name__)

COMPARISON_EVENTS = 30_000
SHOW_ADVANCED_SETTING = 'autospectral_correction_show_advanced'
TABLE_HEADERS = ['Active', 'Fluorophore', 'Change (deg)', 'Phase one (deg)', 'Phase two', 'Events']
COL_BASE = (80, 200, 80)          # green: base spectrum
COL_CORRECTED = (160, 50, 220)    # purple: corrected spectrum
LABEL_DENSITY_TARGET = 80
GROUP_TABLE_MAX_HEIGHT = 200


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class MatchUnstainedWorker(QObject):
    """Runs spectral_correction.match_unstained in a QThread."""
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, job: dict):
        super().__init__()
        self.job = job

    def run(self):
        try:
            j = self.job
            self.finished.emit(sc.match_unstained(j['experiment_dir'], j['groups'], j['references'], j['pair'],
                                                  j['ceilings'], progress=self.progress.emit))
        except Exception as e:
            logger.exception('AutoSpectral Spectral Correction: unstained match failed')
            self.error.emit(str(e) or type(e).__name__)


class FixUnmixWorker(QObject):
    """Runs spectral_correction.run_corrections in a QThread."""
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, snap: dict):
        super().__init__()
        self.snap = snap

    def run(self):
        try:
            self.finished.emit(sc.run_corrections(self.snap, self.progress.emit))
        except Exception as e:
            logger.exception('AutoSpectral Spectral Correction: Fix Unmix failed')
            self.error.emit(str(e) or type(e).__name__)


class CorrectionComparisonWorker(QObject):
    """The comparison pair: the experiment's method with the base and with the corrected spectra."""
    finished = Signal(object, object)
    error = Signal(str)

    def __init__(self, job: dict):
        super().__init__()
        self.job = job

    def run(self):
        try:
            j = self.job
            out = []
            for state in (j['state_base'], j['state_corrected']):
                res = unmix_method.unmix(j['raw'], j['transfer'], j['settings'], j['fl_ids'], j['spillover'],
                                         j['af'], state, sample_key=j['sample_key'])
                out.append(res['unmixed'])
            self.finished.emit(out[0], out[1])
        except Exception as e:
            logger.exception('AutoSpectral Spectral Correction: comparison failed')
            self.error.emit(str(e) or type(e).__name__)


def _spin(widget):
    """Wheel events and tab focus off, as for the tab's other spin boxes and combos."""
    widget.installEventFilter(WheelBlocker(widget))
    widget.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
    return widget


def _start_thread(worker, on_finished, on_error, on_thread_finished, progress=None):
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(on_finished)
    worker.error.connect(on_error)
    if progress is not None:
        worker.progress.connect(progress)
    worker.finished.connect(thread.quit)
    worker.error.connect(thread.quit)
    thread.finished.connect(thread.deleteLater)
    thread.finished.connect(on_thread_finished)
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

class SpectralCorrectionPage(QWidget):
    """AutoSpectral tab page: correct in-span spectral error from fully stained samples (Fix My Unmix)."""

    def __init__(self, bus=None, controller=None, active_fn=None, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.controller = controller
        # active_fn(page) -> bool: whether the page is on screen.
        self._active_fn = active_fn
        self._qsettings = QSettings('honeychrome', 'app_configuration')

        self._match_thread = None
        self._match_worker = None
        self._fix_thread = None
        self._job_token = None
        self._fix_worker = None
        self._cmp_thread = None
        self._cmp_worker = None
        self._cmp_running = False
        # Fix Unmix waits for the scatter match it asked for
        self._run_after_match = False
        self._fix_after_match = False
        # {(group, members): unstained path} from the last scatter match
        self._matched = {}
        self._left_data = None
        self._right_data = None
        self._last_cmp_state = None
        self._pending_cmp_state = None
        self._loading = False
        self._param_widgets = {}
        self._group_combos = {}

        self._build_ui()

        if self.bus:
            self.bus.modeChangeRequested.connect(self._on_mode_change)
            self.bus.loadSampleRequested.connect(self._on_sample_loaded)
            self.bus.spectralProcessRefreshed.connect(self._on_process_refreshed)
            self.bus.sampleTreeUpdated.connect(self._refresh_samples)
            self.bus.unmixingMethodChanged.connect(self._refresh_summary)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._label_disabled = QLabel(
            'Spectral Correction: calculate the unmixing matrix (Spectral Process tab) before using this page.'
        )
        self._label_disabled.setWordWrap(True)
        self._label_disabled.setContentsMargins(8, 8, 8, 8)
        outer.addWidget(self._label_disabled)

        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setAlignment(Qt.AlignTop)
        content_layout.setSpacing(10)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setWidget(self._content)
        outer.addWidget(self._scroll)

        content_layout.addWidget(HelpToggleWidget(text=autospectral_correction_help_text))
        if not fm.KERNEL_AVAILABLE:
            note = QLabel('<b>Compiled Fix My Unmix kernel not found.</b> The pair slopes run in NumPy, two to '
                          'three times slower. Run <tt>build_fix_my_unmix_kernel.py</tt> (in bundled_plugins/) '
                          'to build it.')
            note.setTextFormat(Qt.RichText)
            note.setWordWrap(True)
            note.setStyleSheet('color: #cc6600;')
            content_layout.addWidget(note)
        self._build_setup_section(content_layout)
        self._build_samples_section(content_layout)
        self._build_results_section(content_layout)
        self._build_compare_section(content_layout)
        content_layout.addStretch()

    def _build_setup_section(self, parent_layout):
        grp = QGroupBox('Setup')
        layout = QVBoxLayout(grp)
        self._summary = QLabel('')
        self._summary.setTextFormat(Qt.RichText)
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)
        parent_layout.addWidget(grp)

    def _build_samples_section(self, parent_layout):
        grp = QGroupBox('Fix Unmix')
        layout = QVBoxLayout(grp)

        include_row = QHBoxLayout()
        self._include_checks = {}
        for key, text in (('unstained', 'Include unstained samples'),
                          ('single_stain', 'Include single-stained controls'),
                          ('beads', 'Include bead samples')):
            chk = QCheckBox(text)
            chk.toggled.connect(self._on_include_toggled)
            include_row.addWidget(chk)
            self._include_checks[key] = chk
        include_row.addStretch()
        layout.addLayout(include_row)

        layout.addWidget(QLabel('Fully stained samples:'))
        self._sample_list = QListWidget()
        self._sample_list.setSelectionMode(QAbstractItemView.NoSelection)
        self._sample_list.setMinimumHeight(120)
        self._sample_list.setMaximumHeight(220)
        self._sample_list.itemChanged.connect(self._on_sample_ticked)
        layout.addWidget(self._sample_list)

        row = QHBoxLayout()
        self._default_btn = QPushButton('Default Samples')
        self._default_btn.setToolTip('Every sample that is not a bead sample, an unstained sample or a '
                                     'single-stain control; clears the Include tick boxes.')
        self._default_btn.clicked.connect(self._on_default_samples)
        row.addWidget(self._default_btn)
        self._per_group_check = QCheckBox('Concatenate per group')
        self._per_group_check.setToolTip('Run each raw subfolder on its own: each sample is unmixed with its own '
                                         'group\'s corrected spectra.')
        self._per_group_check.toggled.connect(self._on_per_group_toggled)
        row.addWidget(self._per_group_check)
        row.addStretch()
        layout.addLayout(row)

        layout.addWidget(QLabel('Unstained sample of each group (positivity thresholds):'))
        self._group_table = QTableWidget(0, 3)
        self._group_table.setHorizontalHeaderLabels(['Group', 'Samples', 'Unstained'])
        self._group_table.verticalHeader().setVisible(False)
        self._group_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._group_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._group_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._group_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        layout.addWidget(self._group_table)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.DontWrapRows)
        self._gate_combo = _spin(QComboBox())
        self._gate_combo.setToolTip('Raw gate for the fully stained and the unstained samples.')
        self._gate_combo.currentIndexChanged.connect(self._on_setup_changed)
        form.addRow('Gate:', self._gate_combo)
        self._events_spin = _spin(QSpinBox())
        self._events_spin.setRange(5_000, 5_000_000)
        self._events_spin.setSingleStep(50_000)
        self._events_spin.setValue(sc.DEFAULT_EVENTS)
        self._events_spin.setToolTip('Fully stained events per group, drawn equally from its samples; also the '
                                     'unstained sample\'s cap.')
        self._events_spin.valueChanged.connect(self._on_setup_changed)
        form.addRow('Events per group:', self._events_spin)
        self._seed_spin = _spin(QSpinBox())
        self._seed_spin.setRange(0, 1_000_000)
        self._seed_spin.setValue(sc.DEFAULT_SEED)
        self._seed_spin.valueChanged.connect(self._on_setup_changed)
        form.addRow('Seed:', self._seed_spin)
        layout.addLayout(form)

        layout.addWidget(QLabel('Fluorophores kept fixed (never corrected):'))
        self._fixed_list = QListWidget()
        self._fixed_list.setSelectionMode(QAbstractItemView.NoSelection)
        self._fixed_list.setMaximumHeight(120)
        self._fixed_list.setToolTip('Ticked fluorophores keep their spectra: Fix Unmix gathers no evidence with '
                                    'them as the source and does not re-measure them. Tick FlowCode tag '
                                    'fluorophores, whose FRET error the FlowCode plugin removes itself.')
        self._fixed_list.itemChanged.connect(self._on_setup_changed)
        layout.addWidget(self._fixed_list)

        self._build_advanced_settings(layout)

        row = QHBoxLayout()
        self._match_btn = QPushButton('Match Unstained')
        self._match_btn.setToolTip('Match each group to an unstained cell sample by scatter (as Auto-assign on '
                                   'the Autofluorescence page).')
        self._match_btn.clicked.connect(lambda: self._run_match(then_fix=False))
        row.addWidget(self._match_btn)
        self._fix_btn = QPushButton('Fix Unmix')
        self._fix_btn.clicked.connect(self._on_fix_clicked)
        row.addWidget(self._fix_btn)
        row.addStretch()
        layout.addLayout(row)

        self._fix_status = QLabel('')
        self._fix_status.setWordWrap(True)
        layout.addWidget(self._fix_status)

        self._use_check = QCheckBox('Use spectral corrections for the experiment\'s unmixing')
        self._use_check.setToolTip(
            'Each corrected sample is unmixed with its group\'s corrected spectra (Active rows) by the experiment\'s '
            'method, in the main window, exports, Honeycluster, the gating plugins and Statistics.')
        self._use_check.toggled.connect(self._on_use_toggled)
        layout.addWidget(self._use_check)
        # corrections decide the unmixing: not changed under a running job
        guard_while_busy(self.controller, self._match_btn, self._fix_btn, self._use_check)
        if getattr(self.controller, 'job_gate', None) is not None:
            self.controller.job_gate.changed.connect(self._on_job_gate_changed)

        parent_layout.addWidget(grp)

    def _build_advanced_settings(self, parent_layout):
        self._adv_toggle = QCheckBox('Show Advanced Settings')
        saved = self._qsettings.value(SHOW_ADVANCED_SETTING, 'false') == 'true'
        self._adv_toggle.setChecked(saved)
        parent_layout.addWidget(self._adv_toggle)

        self._adv_content = QWidget()
        adv_layout = QVBoxLayout(self._adv_content)
        adv_layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.DontWrapRows)
        defaults = fm.FixSettings()
        for key, label, kind, lo, hi, step, decimals in sc.EXPOSED_SETTINGS:
            value = getattr(defaults, key)
            if kind == 'bool':
                w = QCheckBox()
                w.setChecked(bool(value))
                w.toggled.connect(self._on_setup_changed)
            elif isinstance(kind, tuple):
                w = _spin(QComboBox())
                for data, text in kind:
                    w.addItem(text, data)
                w.setCurrentIndex(max(w.findData(value), 0))
                w.currentIndexChanged.connect(self._on_setup_changed)
            elif kind == 'int':
                w = _spin(QSpinBox())
                w.setRange(int(lo), int(hi))
                w.setSingleStep(int(step))
                w.setValue(int(value))
                w.valueChanged.connect(self._on_setup_changed)
            else:
                w = _spin(QDoubleSpinBox())
                w.setDecimals(int(decimals))
                w.setRange(float(lo), float(hi))
                w.setSingleStep(float(step))
                w.setValue(float(value))
                w.valueChanged.connect(self._on_setup_changed)
            form.addRow(label + ':', w)
            self._param_widgets[key] = w
        adv_layout.addLayout(form)
        reset = QPushButton('Restore Defaults')
        reset.clicked.connect(self._on_restore_defaults)
        adv_layout.addWidget(reset, alignment=Qt.AlignLeft)
        self._adv_content.setVisible(saved)
        self._adv_toggle.toggled.connect(self._adv_content.setVisible)
        self._adv_toggle.toggled.connect(
            lambda c: self._qsettings.setValue(SHOW_ADVANCED_SETTING, 'true' if c else 'false'))
        parent_layout.addWidget(self._adv_content)

    def _build_results_section(self, parent_layout):
        grp = QGroupBox('Results')
        layout = QVBoxLayout(grp)

        row = QHBoxLayout()
        row.addWidget(QLabel('Group:'))
        self._result_group_combo = _spin(QComboBox())
        self._result_group_combo.currentIndexChanged.connect(self._show_results)
        row.addWidget(self._result_group_combo, stretch=1)
        layout.addLayout(row)

        self._result_summary = QLabel('')
        self._result_summary.setTextFormat(Qt.RichText)
        self._result_summary.setWordWrap(True)
        layout.addWidget(self._result_summary)

        self._table = QTableWidget(0, len(TABLE_HEADERS))
        guard_while_busy(self.controller, self._table)   # Active ticks
        self._table.setHorizontalHeaderLabels(TABLE_HEADERS)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setMaximumHeight(260)
        self._table.itemChanged.connect(self._on_active_changed)
        layout.addWidget(self._table)

        self._heatmap_toggle = QCheckBox('Show spectral change heatmap')
        layout.addWidget(self._heatmap_toggle)
        self._heatmap_holder = QWidget()
        self._heatmap_layout = QVBoxLayout(self._heatmap_holder)
        self._heatmap_layout.setContentsMargins(0, 0, 0, 0)
        self._heatmap_canvas = None
        self._heatmap_holder.setVisible(False)
        self._heatmap_toggle.toggled.connect(self._on_heatmap_toggled)
        layout.addWidget(self._heatmap_holder)

        row = QHBoxLayout()
        row.addWidget(QLabel('Fluorophore:'))
        self._trace_combo = _spin(QComboBox())
        self._trace_combo.setMinimumWidth(200)
        self._trace_combo.currentIndexChanged.connect(self._draw_traces)
        row.addWidget(self._trace_combo)
        for color, text in ((COL_BASE, 'Base spectrum'), (COL_CORRECTED, 'Corrected spectrum')):
            r, g, b = color
            swatch = QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(f'background-color: rgb({r},{g},{b}); border: 1px solid #444;')
            row.addWidget(swatch)
            row.addWidget(QLabel(text))
        row.addStretch()
        layout.addLayout(row)
        self._trace_axis = BottomAxisVerticalTickLabels(angle=45)
        self._trace_plot = TransparentPlotWidget(axisItems={'bottom': self._trace_axis})
        self._trace_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._trace_plot.setMinimumHeight(220)
        self._trace_plot.getViewBox().setMenuEnabled(False)
        self._trace_plot.setLabel('left', 'Normalised intensity')
        self._trace_plot.showGrid(x=True, y=True, alpha=0.3)
        self._trace_plot.setMouseEnabled(x=False, y=False)
        layout.addWidget(self._trace_plot)

        parent_layout.addWidget(grp)

    def _build_compare_section(self, parent_layout):
        grp = QGroupBox('Comparison (Current Sample)')
        layout = QVBoxLayout(grp)

        row = QHBoxLayout()
        self._cmp_run_btn = QPushButton('Update Plots')
        self._cmp_run_btn.clicked.connect(self._run_comparison)
        row.addWidget(self._cmp_run_btn)
        row.addWidget(QLabel('Events:'))
        self._cmp_events_spin = _spin(QSpinBox())
        self._cmp_events_spin.setRange(1_000, 2_000_000)
        self._cmp_events_spin.setSingleStep(10_000)
        self._cmp_events_spin.setValue(COMPARISON_EVENTS)
        self._cmp_events_spin.setToolTip('Events of the current sample unmixed for the comparison (random subsample).')
        row.addWidget(self._cmp_events_spin)
        self._cmp_status = QLabel('')
        self._cmp_status.setWordWrap(True)
        row.addWidget(self._cmp_status, stretch=1)
        layout.addLayout(row)

        splitter = QSplitter(Qt.Horizontal)
        self._plot_left = AfComparisonPlotWidget('Uncorrected', self.controller, parent=self)
        self._plot_right = AfComparisonPlotWidget('Corrected', self.controller, parent=self)
        splitter.addWidget(self._plot_left)
        splitter.addWidget(self._plot_right)
        splitter.setSizes([500, 500])
        layout.addWidget(splitter)

        for plot, other in ((self._plot_left, self._plot_right), (self._plot_right, self._plot_left)):
            plot.sourceGateChanged.connect(other.set_source_gate)
            plot.channelChanged.connect(lambda x, y, o=other: self._mirror(o, 'set_channels', x, y))
            plot.scalingChanged.connect(lambda a, tr, o=other: self._mirror(o, 'set_scaling', a, tr))

        parent_layout.addWidget(grp)

    @staticmethod
    def _mirror(plot, method: str, *args):
        """Apply a sibling's channel or scaling change without echoing it back."""
        plot.blockSignals(True)
        getattr(plot, method)(*args)
        plot.blockSignals(False)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        """True while the page is on screen."""
        if self._active_fn is None:
            return self.controller is not None and self.controller.current_mode == TAB_NAME
        return self._active_fn(self)

    def _available(self) -> bool:
        return bool(self.controller and self.controller.experiment
                    and self.controller.experiment.process.get('unmixing_matrix') is not None)

    def _process(self) -> dict:
        return self.controller.experiment.process

    def _entry(self) -> dict | None:
        return self._process().get(sc.STORE_KEY)

    def _setup(self) -> sc.CorrectionSetup:
        return sc.load_setup(self._process())

    def _selected_samples(self) -> list:
        out = []
        for i in range(self._sample_list.count()):
            item = self._sample_list.item(i)
            if item.checkState() == Qt.Checked:
                out.append(item.data(Qt.UserRole))
        return out

    def _groups(self) -> dict:
        return sc.sample_groups(self._selected_samples(), self._per_group_check.isChecked())

    def _refresh_ui(self):
        available = self._available()
        self._label_disabled.setVisible(not available)
        self._content.setVisible(available)
        self._scroll.setVisible(available)
        if not available:
            return
        self._clear_if_stale()
        self._loading = True
        try:
            setup = self._setup()
            self._per_group_check.setChecked(bool(setup.per_group))
            self._events_spin.setValue(int(setup.n_events))
            self._seed_spin.setValue(int(setup.seed))
            self._set_param_values(setup.fix or {})
            self._refresh_fixed(setup.fixed)
            self._refresh_gates(setup.gate)
            self._refresh_samples()
        finally:
            self._loading = False
        self._refresh_summary()
        self._refresh_use_check()
        self._refresh_results()

    def _clear_if_stale(self):
        names, spectra = fluor_names_and_spectra(self.controller)
        if sc.clear_if_stale(self._process(), names, spectra):
            self._fix_status.setText('The base spectra have changed since Fix Unmix was run, so the spectral '
                                     'corrections were cleared. Run Fix Unmix again.')
            self._method_changed()

    def _refresh_gates(self, stored: str | None):
        c = self.controller
        gating = c.raw_gating
        names = ['root'] if gating is None else ['root'] + [g[0] for g in gating.get_gate_ids()]
        base_gate, _ = resolve_base_gate(gating, self._process().get('base_gate_priority_order', []))
        current = stored or self._gate_combo.currentData() or base_gate or 'root'
        self._gate_combo.blockSignals(True)
        self._gate_combo.clear()
        for name in names:
            self._gate_combo.addItem(name, name)
        i = self._gate_combo.findData(current)
        self._gate_combo.setCurrentIndex(max(i, 0))
        self._gate_combo.blockSignals(False)

    def _refresh_samples(self):
        if not self._available():
            return
        samples = self.controller.experiment.samples
        setup = self._setup()
        chosen = set(sc.default_samples(samples) if setup.samples is None else setup.samples)
        was_loading = self._loading
        self._loading = True
        for key, chk in self._include_checks.items():
            chk.setChecked(bool((setup.include or {}).get(key)))
        names = samples.get('all_samples') or {}
        self._sample_list.blockSignals(True)
        self._sample_list.clear()
        for path in sc.offered_samples(samples, setup.include):
            name = names[path]
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, path)
            item.setToolTip(path)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if path in chosen else Qt.Unchecked)
            self._sample_list.addItem(item)
        self._sample_list.blockSignals(False)
        self._loading = was_loading
        self._refresh_group_table()

    def _refresh_group_table(self):
        groups = self._groups()
        setup = self._setup()
        stored = setup.unstained or {}
        choices = sc.unstained_choices(self.controller)
        names = self.controller.experiment.samples.get('all_samples', {})
        self._group_combos = {}
        self._group_table.setRowCount(len(groups))
        for r, (group, members) in enumerate(groups.items()):
            self._group_table.setItem(r, 0, QTableWidgetItem(group))
            self._group_table.setItem(r, 1, QTableWidgetItem(str(len(members))))
            combo = _spin(QComboBox())
            matched = self._matched.get((group, tuple(members)))
            auto = f'Auto (scatter): {names.get(matched, matched)}' if matched else 'Auto (scatter match)'
            combo.addItem(auto, sc.AUTO)
            for path, name, has_af in choices:
                combo.addItem(name if has_af else f'{name} (no AF profile)', path)
            i = combo.findData(stored.get(group, sc.AUTO))
            combo.setCurrentIndex(max(i, 0))
            combo.currentIndexChanged.connect(self._on_setup_changed)
            self._group_table.setCellWidget(r, 2, combo)
            self._group_combos[group] = combo
        self._group_table.resizeRowsToContents()
        rows_height = sum(self._group_table.rowHeight(r) for r in range(self._group_table.rowCount()))
        self._group_table.setFixedHeight(min(GROUP_TABLE_MAX_HEIGHT, self._group_table.horizontalHeader().height()
                                             + rows_height + 2 * self._group_table.frameWidth() + 2))
        self._fix_btn.setEnabled(bool(groups) and self._fix_thread is None and self._match_thread is None)

    def _refresh_fixed(self, fixed):
        names, _spectra = fluor_names_and_spectra(self.controller)
        chosen = set(fixed or [])
        self._fixed_list.blockSignals(True)
        self._fixed_list.clear()
        for name in names or []:
            if name == fm.AF_NAME:
                continue
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in chosen else Qt.Unchecked)
            self._fixed_list.addItem(item)
        self._fixed_list.blockSignals(False)

    def _fixed_values(self) -> list:
        out = []
        for i in range(self._fixed_list.count()):
            item = self._fixed_list.item(i)
            if item.checkState() == Qt.Checked:
                out.append(item.data(Qt.UserRole))
        return out

    def _set_param_values(self, values: dict):
        defaults = fm.FixSettings()
        for key, w in self._param_widgets.items():
            value = values.get(key, getattr(defaults, key))
            w.blockSignals(True)
            if isinstance(w, QCheckBox):
                w.setChecked(bool(value))
            elif isinstance(w, QComboBox):
                w.setCurrentIndex(max(w.findData(value), 0))
            else:
                w.setValue(value)
            w.blockSignals(False)

    def _param_values(self) -> dict:
        """Exposed settings that differ from their defaults."""
        defaults = fm.FixSettings()
        out = {}
        for key, w in self._param_widgets.items():
            if isinstance(w, QCheckBox):
                value = w.isChecked()
            elif isinstance(w, QComboBox):
                value = w.currentData()
            else:
                value = w.value()
            if value != getattr(defaults, key):
                out[key] = value
        return out

    def _current_setup(self) -> sc.CorrectionSetup:
        samples = self.controller.experiment.samples
        include = {key: chk.isChecked() for key, chk in self._include_checks.items()}
        offered = set(sc.offered_samples(samples, include))
        selected = [p for p in self._selected_samples() if p in offered]
        stored = self._setup()
        sample_choice = None if (stored.samples is None
                                 and set(selected) == set(sc.default_samples(samples))) else selected
        unstained = {g: combo.currentData() for g, combo in self._group_combos.items()}
        return sc.CorrectionSetup(
            samples=sample_choice, per_group=self._per_group_check.isChecked(),
            gate=self._gate_combo.currentData(), n_events=int(self._events_spin.value()),
            seed=int(self._seed_spin.value()), unstained=unstained, include=include, fix=self._param_values(),
            fixed=self._fixed_values() or None)

    def _save_setup(self):
        sc.save_setup(self._process(), self._current_setup())
        if self.bus:
            self.bus.autoSaveRequested.emit()

    def _on_setup_changed(self, *_args):
        if self._loading or not self._available():
            return
        self._save_setup()

    def _on_per_group_toggled(self, _checked: bool):
        if self._loading or not self._available():
            return
        self._save_setup()
        self._refresh_group_table()
        self._save_setup()

    def _on_include_toggled(self, _checked: bool):
        """Offer or withdraw a sample type; ticked samples still offered stay ticked."""
        if self._loading or not self._available():
            return
        self._save_setup()
        self._refresh_samples()
        self._save_setup()

    def _on_sample_ticked(self, _item):
        if self._loading:
            return
        self._save_setup()
        self._refresh_group_table()

    def _on_default_samples(self):
        setup = self._setup()
        setup.samples = None
        setup.include = None
        sc.save_setup(self._process(), setup)
        if self.bus:
            self.bus.autoSaveRequested.emit()
        self._refresh_samples()

    def _on_restore_defaults(self):
        self._set_param_values({})
        self._save_setup()

    def _refresh_summary(self, *_args):
        if not self._available():
            return
        c = self.controller
        process = self._process()
        names, spectra = fluor_names_and_spectra(c)
        selected = self._selected_samples()
        with_af = sum(bool(af_profile_names(c, p)) for p in selected)
        n_profiles = len(process.get('af_profiles') or {})
        lines = [f'<b>AF profiles:</b> {n_profiles} stored; {with_af} of {len(selected)} chosen samples have '
                 f'AF profiles assigned (the others are unmixed without AF).']
        if spectra is not None and spread_is_current(process.get('af_spread'), names, spectra):
            lines.append('<b>Spillover spread:</b> current; it widens the positivity boundaries.')
        else:
            lines.append('<span style="color:#cc6600"><b>Spillover spread:</b> none for the current spectra, so the '
                         'positivity boundaries are flat. Extracting an AF profile estimates it.</span>')
        n_var = sum(d is not None for d in sc.variant_deltas(c, names))
        if n_var:
            lines.append(f'<b>Spectral variants:</b> {n_var} of {len(names)} fluorophores; they set the prior on '
                         'each coefficient.')
        else:
            lines.append('<b>Spectral variants:</b> none computed (Spectral Variants page), so coefficients are '
                         'trusted from their own fit only.')
        method = unmix_method.requested_method(process)
        lam = unmix_method.unmixing_settings(process)['prior_lambda']
        noise = ' (noise model in use)' if noise_in_use(process) else ''
        lines.append(f'<b>Experiment\'s unmixing:</b> {unmix_method.method_label(method, lam)}{noise}; corrected '
                     'spectra feed it. Fix Unmix itself reads per-cell AF without the positivity prior.')
        fixed = self._setup().fixed or []
        if fixed:
            lines.append(f'<b>Kept fixed:</b> {", ".join(fixed)}.')
        self._summary.setText('<br>'.join(lines))

    def _on_job_gate_changed(self):
        """Availability of the Use tick and the run buttons again once the job gate is free."""
        if not self.controller.job_gate.is_held():
            self._refresh_use_check()
            self._set_running(self._busy())

    def _refresh_use_check(self, *_args):
        if not self._available():
            return
        entry = self._entry()
        self._use_check.blockSignals(True)
        self._use_check.setEnabled(bool(entry))
        self._use_check.setChecked(bool(entry) and unmix_method.unmixing_settings(self._process())['corrections'])
        self._use_check.blockSignals(False)

    def _method_changed(self):
        """The experiment's unmixing changed: unmix the displayed sample again and tell the other pages."""
        if self.bus:
            self.bus.autoSaveRequested.emit()
        if self.controller is not None and hasattr(self.controller, 'refresh_unmixing'):
            self.controller.refresh_unmixing()
        if self.bus:
            self.bus.unmixingMethodChanged.emit()

    def _on_use_toggled(self, checked: bool):
        unmix_method.set_unmixing(self._process(), corrections=bool(checked))
        self._method_changed()
        self._refresh_comparison_if_active()

    # ------------------------------------------------------------------
    # Match Unstained and Fix Unmix
    # ------------------------------------------------------------------

    def _busy(self) -> bool:
        return self._match_thread is not None or self._fix_thread is not None

    def _take_job(self, label: str) -> bool:
        """Hold the job gate for *label*, or keep holding it under *label* (match, then Fix Unmix)."""
        if self._job_token is not None and self._job_token.held:
            self._job_token.hand_over(label)
            return True
        self._job_token = acquire_job(self.controller, label)
        if self._job_token is None:
            self._fix_status.setText(self.controller.job_gate.busy_text())
            return False
        return True

    def _release_job(self):
        token, self._job_token = self._job_token, None
        if token is not None:
            token.release()

    def _set_running(self, running: bool):
        self._fix_btn.setEnabled(not running and bool(self._groups()))
        self._match_btn.setEnabled(not running)

    def _run_match(self, then_fix: bool):
        if self._busy():
            self._fix_status.setText('Already running — please wait.')
            return
        c = self.controller
        groups = self._groups()
        if not groups:
            self._fix_status.setText('Choose at least one fully stained sample.')
            return
        raw_settings = c.experiment.settings['raw']
        pair = scatter_channel_pair(raw_settings, c.experiment.cytometry.get('raw_plots'))
        references = [p for p, _n, _af in sc.unstained_choices(c)]
        if pair is None or not references:
            self._fix_status.setText('No unstained cell sample or scatter channels to match by: choose each '
                                     'group\'s unstained sample.')
            return
        if not self._take_job('AutoSpectral: Fix Unmix' if then_fix else 'AutoSpectral: match unstained'):
            return
        self._run_after_match = then_fix
        job = {'experiment_dir': str(c.experiment_dir), 'groups': groups, 'references': references,
               'pair': pair, 'ceilings': scatter_ceilings(raw_settings, pair)}
        self._set_running(True)
        self._fix_status.setText('Matching unstained samples by scatter...')
        self._match_worker = MatchUnstainedWorker(job)
        self._match_thread = _start_thread(self._match_worker, self._on_match_finished, self._on_match_error,
                                           self._on_match_thread_finished, self._fix_status.setText)

    def _on_match_finished(self, found: dict):
        groups = self._groups()
        names = self.controller.experiment.samples.get('all_samples', {})
        unmatched = []
        for group, members in groups.items():
            path = (found.get(group) or {}).get('unstained')
            if path:
                self._matched[(group, tuple(members))] = path
            else:
                unmatched.append(group)
        self._refresh_group_table()
        text = '; '.join(f'{g}: {names.get(found[g]["unstained"], found[g]["unstained"])}'
                         for g in groups if (found.get(g) or {}).get('unstained'))
        if unmatched:
            text += ('; ' if text else '') + 'no scatter match for ' + ', '.join(unmatched) + ' (choose one)'
        self._fix_status.setText(f'Unstained by scatter: {text}.')
        self._fix_after_match = self._run_after_match and not unmatched

    def _on_match_error(self, msg: str):
        self._fix_after_match = False
        self._fix_status.setText(f'Error: {msg}')

    def _on_match_thread_finished(self):
        self._match_thread = None
        self._match_worker = None
        self._set_running(False)
        if self._fix_after_match:
            self._fix_after_match = False
            self._start_fix()
        else:
            self._release_job()

    def _resolved_unstained(self) -> dict | None:
        """{group: unstained path}, or None when a group on Auto has no scatter match yet."""
        out = {}
        for group, members in self._groups().items():
            choice = self._group_combos[group].currentData() if group in self._group_combos else sc.AUTO
            if choice == sc.AUTO:
                choice = self._matched.get((group, tuple(members)))
                if choice is None:
                    return None
            out[group] = choice
        return out

    def _on_fix_clicked(self):
        if self._busy():
            self._fix_status.setText('Already running — please wait.')
            return
        if self._resolved_unstained() is None:
            self._run_match(then_fix=True)
            return
        self._start_fix()

    def _start_fix(self):
        if self._busy():
            return
        unstained = self._resolved_unstained()
        if unstained is None:
            self._release_job()
            self._fix_status.setText('Choose an unstained sample for every group.')
            return
        if not self._take_job('AutoSpectral: Fix Unmix'):
            return
        self._save_setup()
        try:
            snap = sc.correction_snapshot(self.controller, self._current_setup(), unstained)
        except Exception as e:
            self._release_job()
            self._fix_status.setText(f'Error: {e}')
            return
        self._set_running(True)
        self._fix_status.setText('Running Fix My Unmix...')
        self._fix_worker = FixUnmixWorker(snap)
        self._fix_thread = _start_thread(self._fix_worker, self._on_fix_finished, self._on_fix_error,
                                         self._on_fix_thread_finished, self._fix_status.setText)

    def _on_fix_finished(self, run: dict):
        self._release_job()
        process = self._process()
        names, spectra = fluor_names_and_spectra(self.controller)
        if spectra is None or run['base_fingerprint'] != spectra_fingerprint(names, spectra):
            self._fix_status.setText('The spectra changed while Fix Unmix ran; the result was discarded. Run it again.')
            return
        sc.store_corrections(process, run)
        n_changed = sum(len(g['fluorophores']) for g in process[sc.STORE_KEY]['groups'].values())
        n_groups = len(run['groups'])
        where = f' in {n_groups} groups' if n_groups > 1 else ''
        self._fix_status.setText(f'Done. {n_changed} spectra corrected{where}; the experiment is now unmixed with '
                                 'them.' if n_changed else 'Done. No spectrum needed a correction.')
        self._refresh_use_check()
        self._refresh_results()
        self._method_changed()
        self._refresh_comparison_if_active()

    def _on_fix_error(self, msg: str):
        self._release_job()
        self._fix_status.setText(f'Error: {msg}')

    def _on_fix_thread_finished(self):
        self._release_job()
        self._fix_thread = None
        self._fix_worker = None
        self._set_running(False)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _refresh_results(self):
        entry = self._entry()
        current = self._result_group_combo.currentData()
        self._result_group_combo.blockSignals(True)
        self._result_group_combo.clear()
        for group in ((entry or {}).get('groups') or {}):
            self._result_group_combo.addItem(group, group)
        i = self._result_group_combo.findData(current)
        self._result_group_combo.setCurrentIndex(max(i, 0))
        self._result_group_combo.setEnabled(self._result_group_combo.count() > 1)
        self._result_group_combo.blockSignals(False)
        self._show_results()

    def _result_group(self) -> str | None:
        entry = self._entry()
        group = self._result_group_combo.currentData()
        if not entry or group not in (entry.get('groups') or {}):
            return None
        return group

    def _show_results(self, *_args):
        entry = self._entry()
        group = self._result_group()
        self._table.blockSignals(True)
        self._table.setRowCount(0)
        if group is None:
            self._result_summary.setText('No spectral corrections yet: choose the samples and press '
                                         '<b>Fix Unmix</b>.' if not entry else '')
            self._table.blockSignals(False)
            self._trace_combo.clear()
            self._trace_plot.clear()
            self._draw_heatmap()
            return
        g = entry['groups'][group]
        names = self.controller.experiment.samples.get('all_samples', {})
        n_events = sum((g.get('events') or {}).values())
        corrected = g.get('fluorophores') or {}
        lines = [f'<b>{len(corrected)}</b> spectra corrected from {n_events:,} events of {len(g["samples"])} '
                 f'sample(s), unstained <i>{names.get(g["unstained"], g["unstained"])}</i> '
                 f'({g.get("n_unstained", 0):,} events) in {entry.get("gate")}; '
                 + (f'smooth updates over {g["smooth_rounds"]} rounds, the last round\'s phase one: '
                    if g.get('smooth_rounds', 1) > 1 else 'phase one: ')
                 + f'{g.get("iterations", 0)} iteration(s), {g.get("stop_reason", "")}, '
                 f'{g.get("n_coefficients", 0)} coefficients trusted. Run {entry.get("fitted", "")}.']
        if corrected:
            spread = 'the spillover spread recomputed from the controls' if g.get('spread') else \
                'the base spillover spread (no controls could be read)'
            lines.append(f'The corrected spectra are unmixed with AF envelopes rebuilt for them and {spread}.')
        self._result_summary.setText('<br>'.join(lines))
        rows = g.get('log') or []
        self._table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            label = row['fluorophore']
            active = QTableWidgetItem('')
            if label in corrected:
                active.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                active.setCheckState(Qt.Checked if corrected[label].get('active', True) else Qt.Unchecked)
            else:
                active.setFlags(Qt.ItemIsEnabled)
            active.setData(Qt.UserRole, label)
            self._table.setItem(r, 0, active)
            self._table.setItem(r, 1, QTableWidgetItem(label))
            self._table.setItem(r, 2, QTableWidgetItem(f'{row["angle"]:.2f}'))
            self._table.setItem(r, 3, QTableWidgetItem(f'{row["phase_one_angle"]:.2f}'))
            verdict = 'accepted' if row['accepted'] else f'rejected: {row["result"]}'
            self._table.setItem(r, 4, QTableWidgetItem(verdict))
            self._table.setItem(r, 5, QTableWidgetItem(f'{row["n_events"]:,}'))
        self._table.blockSignals(False)

        current = self._trace_combo.currentText()
        self._trace_combo.blockSignals(True)
        self._trace_combo.clear()
        for label in corrected:
            self._trace_combo.addItem(label)
        i = self._trace_combo.findText(current)
        self._trace_combo.setCurrentIndex(max(i, 0))
        self._trace_combo.blockSignals(False)
        self._draw_traces()
        self._draw_heatmap()

    def _on_active_changed(self, item):
        if item.column() != 0:
            return
        group = self._result_group()
        label = item.data(Qt.UserRole)
        if group is None or not label:
            return
        if sc.set_active(self._process(), group, label, item.checkState() == Qt.Checked):
            self._draw_heatmap()
            if unmix_method.unmixing_settings(self._process())['corrections']:
                self._method_changed()
            elif self.bus:
                self.bus.autoSaveRequested.emit()
            self._refresh_comparison_if_active()

    def _detector_names(self) -> list:
        return list(raw_layout(self.controller)['detectors'])

    def _group_change(self, group: str) -> tuple:
        """(labels, change rows) of a group's corrected fluorophores: corrected minus base spectrum."""
        names, spectra = fluor_names_and_spectra(self.controller)
        entry = self._entry()
        corrected = sc.group_spectra(entry, group, names, spectra)
        labels = list((entry['groups'][group].get('fluorophores') or {}).keys())
        rows = np.array([corrected[names.index(k)] - spectra[names.index(k)] for k in labels if k in names])
        return [k for k in labels if k in names], rows

    def _on_heatmap_toggled(self, checked: bool):
        self._heatmap_holder.setVisible(checked)
        self._draw_heatmap()

    def _draw_heatmap(self):
        if not self._heatmap_toggle.isChecked():
            return
        if self._heatmap_canvas is not None:
            self._heatmap_layout.removeWidget(self._heatmap_canvas)
            self._heatmap_canvas.setParent(None)
            self._heatmap_canvas.deleteLater()
            self._heatmap_canvas = None
        group = self._result_group()
        if group is None:
            return
        labels, rows = self._group_change(group)
        if not labels:
            label = QLabel('No spectrum changed.')
            self._heatmap_layout.addWidget(label)
            self._heatmap_canvas = label
            return
        from matplotlib.figure import Figure

        from honeychrome.view_components.scalable_figure_widget import scrollable_canvas_class

        detectors = [d.removesuffix('-A') for d in self._detector_names()]
        fig = Figure(figsize=(9, 0.6 + 0.28 * len(labels)), layout='constrained')
        ax = fig.add_subplot(111)
        vmax = float(np.max(np.abs(rows))) or 1.0
        im = ax.imshow(rows, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        stride = max(1, round(len(detectors) / LABEL_DENSITY_TARGET))
        ticks = list(range(0, len(detectors), stride))
        ax.set_xticks(ticks, [detectors[i] for i in ticks], rotation=90, fontsize=6)
        fig.colorbar(im, ax=ax, label='Corrected − base (normalised)')
        canvas = scrollable_canvas_class()(fig)
        canvas.setMinimumHeight(int(120 + 22 * len(labels)))
        self._heatmap_layout.addWidget(canvas)
        self._heatmap_canvas = canvas

    def _draw_traces(self, *_args):
        self._trace_plot.clear()
        group = self._result_group()
        label = self._trace_combo.currentText()
        if group is None or not label:
            return
        names, spectra = fluor_names_and_spectra(self.controller)
        if label not in names:
            return
        corrected = sc.group_spectra(self._entry(), group, names, spectra)
        f = names.index(label)
        x = np.arange(spectra.shape[1]) + 0.5
        self._trace_plot.plot(x, spectra[f], pen=pg.mkPen(COL_BASE, width=2))
        self._trace_plot.plot(x, corrected[f], pen=pg.mkPen(COL_CORRECTED, width=2))
        detectors = self._detector_names()
        stride = max(1, round(len(detectors) / LABEL_DENSITY_TARGET))
        pairs = [(i + 0.5, d.removesuffix('-A')) for i, d in enumerate(detectors)]
        db_col = self.controller.experiment.settings['raw'].get('cytometer_db_col')
        colors = {}
        if db_col:
            laser_map = get_detector_laser_map(db_col)
            colors = {lab: LASER_LABEL_COLORS[laser] for _m, lab in pairs
                      if (laser := laser_map.get(lab + '-A')) in LASER_LABEL_COLORS}
        self._trace_axis.tick_colors = colors
        self._trace_axis.setTicks([[(m, lab) for m, lab in pairs if round(m - 0.5) % stride == 0], []])
        self._trace_plot.setXRange(-0.5, len(detectors) - 0.5, padding=0.02)
        self._trace_plot.setYRange(min(0.0, float(min(spectra[f].min(), corrected[f].min()))), 1.05, padding=0)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def _assigned_profiles(self) -> list:
        c = self.controller
        return list(c.experiment.samples.get('sample_af_profiles', {}).get(c.current_sample_path, []) or [])

    def _clear_comparison_plots(self):
        self._plot_left.clear()
        self._plot_right.clear()
        self._left_data = None
        self._right_data = None
        self._last_cmp_state = None
        self._pending_cmp_state = None

    def _refresh_comparison_if_active(self):
        if self.is_active() and self.controller is not None and self.controller.raw_event_data is not None:
            QTimer.singleShot(0, self._run_comparison)

    def _run_comparison(self):
        if self._cmp_running:
            self._cmp_status.setText('Comparison already running — please wait.')
            return
        if wait_if_busy(self.controller, ('AutoSpectral comparison', id(self)),
                        lambda: self.is_active() and self._run_comparison(), self._cmp_status.setText):
            return
        c = self.controller
        if c.raw_event_data is None:
            self._cmp_status.setText('No sample loaded.')
            return
        if c.transfer_matrix is None:
            self._cmp_status.setText('No unmixing matrix — run spectral process first.')
            return
        entry = self._entry()
        key = c.current_sample_path
        group = sc.group_for_sample(entry, key)
        if group is None:
            self._clear_comparison_plots()
            self._cmp_status.setText('The current sample is not among the corrected samples.' if entry
                                     else 'Run Fix Unmix first.')
            return
        names, spectra = fluor_names_and_spectra(c)
        corrected = sc.group_spectra(entry, group, names, spectra, active_only=True)
        profiles = self._assigned_profiles()
        af = resolve_af_for_profiles(c, profiles) if profiles else None
        if profiles and af is None:
            self._cmp_status.setText('AF matrices are not available yet for this sample. Wait for the spectral '
                                     'process to finish, then run the comparison again.')
            return
        try:
            state_base = unmix_method.method_state(c, corrections=({}, {}))
            state_corrected = unmix_method.method_state(c, corrections=({group: corrected}, {str(key): group}))
        except Exception as e:
            self._cmp_status.setText(f'Error: {e}')
            return
        n = len(c.raw_event_data)
        cap = int(self._cmp_events_spin.value())
        rows = np.arange(n) if n <= cap else np.sort(np.random.default_rng(0).choice(n, cap, replace=False))
        state_key = (key, tuple(profiles), unmix_method.method_token(state_base),
                     unmix_method.method_token(state_corrected), cap)
        if state_key == self._last_cmp_state and self._left_data is not None:
            self._cmp_status.setText(f'No change — redrawing from cached result ({self._left_data.shape[0]:,} events).')
            self._initialise_comparison_plots()
            return
        with_af = af is not None
        for plot in (self._plot_left, self._plot_right):
            plot._has_af_channels = with_af
        self._plot_left.set_status('Computing...')
        self._plot_right.set_status('Computing...')
        method = unmix_method.method_label(unmix_method.requested_method(self._process()),
                                           unmix_method.unmixing_settings(self._process())['prior_lambda'])
        self._cmp_status.setText(f'Unmixing the current sample ({method})...')
        job = {
            'raw': np.ascontiguousarray(c.raw_event_data[rows]), 'fl_ids': raw_layout(c)['fl_ids'],
            'transfer': np.array(c.transfer_matrix, dtype=np.float64, copy=True),
            'settings': c.experiment.settings, 'spillover': self._process().get('spillover'),
            'af': af, 'state_base': state_base, 'state_corrected': state_corrected, 'sample_key': key,
        }
        self._pending_cmp_state = state_key
        self._cmp_running = True
        self._cmp_worker = CorrectionComparisonWorker(job)
        self._cmp_thread = _start_thread(self._cmp_worker, self._on_comparison_finished,
                                         self._on_comparison_error, self._on_cmp_thread_finished)

    def _on_comparison_finished(self, left, right):
        self._left_data = left
        self._right_data = right
        self._last_cmp_state = self._pending_cmp_state
        self._cmp_status.setText(f'Done. {left.shape[0]:,} events.')
        self._plot_left.set_status('')
        self._plot_right.set_status('')
        self._initialise_comparison_plots()

    def _on_comparison_error(self, msg: str):
        self._plot_left.set_status('Error')
        self._plot_right.set_status('Error')
        self._cmp_status.setText(f'Error: {msg}')

    def _on_cmp_thread_finished(self):
        self._cmp_running = False
        self._cmp_thread = None
        self._cmp_worker = None

    def _initialise_comparison_plots(self):
        """Feed both plots; default channels are the pair that changes most.
        Existing channel, scaling and gate choices are kept."""
        if self._left_data is None or self._right_data is None:
            return
        c = self.controller
        pnn = c.experiment.settings['unmixed'].get('event_channels_pnn', [])
        fl_ids = c.experiment.settings['unmixed'].get('fluorescence_channel_ids', [])
        if not pnn or not fl_ids or len(fl_ids) < 2:
            return
        fl_names = [pnn[i] for i in fl_ids]
        ch_x, ch_y = fl_names[0], fl_names[1]
        try:
            picked = pick_most_affected_channels(self._left_data, self._right_data, pnn, fl_names)
            if picked is not None:
                ch_x, ch_y = picked
        except Exception as e:
            logger.debug(f'AutoSpectral Spectral Correction comparison: channel auto-pick failed: {e}')
        if self._plot_left._channel_x in fl_names and self._plot_left._channel_y in fl_names:
            ch_x, ch_y = self._plot_left._channel_x, self._plot_left._channel_y

        for plot, data in ((self._plot_left, self._left_data), (self._plot_right, self._right_data)):
            kept = {}
            for ch in (plot._channel_x, plot._channel_y):
                if ch and ch in plot._transformations:
                    tr = plot._transformations[ch]
                    kept[ch] = (tr.id, tr.limits, tr.logicle_w, tr.logicle_a)
            plot.set_event_data(data)
            plot.initialise_from_controller(ch_x, ch_y)
            for ch, (tid, limits, lw, la) in kept.items():
                if ch in plot._transformations:
                    tr = plot._transformations[ch]
                    tr.logicle_w = lw
                    tr.logicle_a = la
                    tr.set_transform(tid, limits)
            plot._configure_axes()
            plot.redraw()

    # ------------------------------------------------------------------
    # Bus slots
    # ------------------------------------------------------------------

    def on_activated(self):
        """The page came on screen: refresh it and bring the comparison up to date."""
        self._refresh_ui()
        if self._available() and self.controller.raw_event_data is not None and self._entry():
            QTimer.singleShot(0, self._run_comparison)

    def _on_mode_change(self, mode):
        if mode == TAB_NAME and self.is_active():
            self.on_activated()

    def _on_sample_loaded(self, _path):
        QTimer.singleShot(0, self._on_sample_loaded_deferred)

    def _on_sample_loaded_deferred(self):
        if not self._available():
            return
        if not self.is_active():
            self._clear_comparison_plots()
            return
        if self._entry():
            self._run_comparison()

    def _on_process_refreshed(self):
        if not self._available():
            return
        self._clear_if_stale()
        if self.is_active():
            self._refresh_ui()
