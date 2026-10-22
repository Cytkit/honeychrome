"""
autospectral_noise_page.py
--------------------------
AutoSpectral tab page: Noise (detector noise model and noise-weighted
unmixing, M2pg).

Sections
--------
1  Model Noise: the unstained cell sample, the bead negative, the positivity
   prior lambda, the estimator options (validation, default off), and the
   fit (gls_unmix.fit_experiment_noise_model; events selected automatically,
   user gates ignored), with its calibration summary and tables; whether
   the experiment is unmixed with the noise model (M2pg, everywhere through
   unmix_method), or, for clamped data, with the weights for clamped data.
2  Comparison: the current sample with per-cell AF | per-cell AF + noise
   model (or the clamped-data weights when those are in use), or OLS (no AF) | noise
   model (no AF) when it has no AF profile; the comparison plots keep their
   own gate menus.

The numerics live in controller_components/gls_unmix.py and noise_model.py.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
from PySide6.QtCore import Qt, QObject, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from honeychrome.controller_components import gls_unmix as gu
from honeychrome.controller_components import noise_model as nm
from honeychrome.controller_components import unmix_method
from honeychrome.controller_components.autospectral_functions import apply_af_transfer
from honeychrome.controller_components.functions import apply_transfer_matrix, resolve_base_gate
from honeychrome.controller_components.job_gate import acquire_job, guard_while_busy, wait_if_busy
from honeychrome.controller_components.particle_type import is_bead_sample
from honeychrome.controller_components.plugin_loaders import bundled_plugins_path
from honeychrome.controller_components.sample_loader import resolve_af_for_profiles
from honeychrome.controller_components.spread_model import raw_layout, spread_is_current
from honeychrome.view_components.autospectral_tab import (
    TAB_NAME, AfComparisonPlotWidget, pick_most_affected_channels,
)
from honeychrome.view_components.copyable_table_widget import CopyableTableWidget
from honeychrome.view_components.help_texts import autospectral_noise_help_text
from honeychrome.view_components.help_toggle_widget import HelpToggleWidget, WheelBlocker

logger = logging.getLogger(__name__)

if str(bundled_plugins_path) not in sys.path:
    sys.path.insert(0, str(bundled_plugins_path))

from autospectral_optimization_functions import af_envelope_for_profiles  # noqa: E402

NONE_KEY = '__none__'
COMPARISON_EVENTS = 30_000
# wait after the last lambda change before unmixing the displayed sample again
LAMBDA_REFRESH_MS = 600
SOURCE_HEADERS = ['Source', 'Kind', 'Events', 'Median obs/pred', 'Detectors in band', 'chi2 p < 0.001',
                  'Median chi2/df']
DETECTOR_HEADERS = ['Detector', 'sigma0', 'kappa', 'c', 'sigma0^2 kappa', 'Flags']
# (option key, label, value when ticked, value when not); keys of NoiseSetupSettings or its .noise
ESTIMATOR_OPTIONS = (
    ('bead_af_rows', 'Bead negatives and bead controls: the negative\'s own AF rows (3)', 3, 0),
    ('laser_split', 'One design row per laser', True, False),
    ('per_control_quadratic', 'Per-control quadratic term', True, False),
    ('sigma0_by_class', 'sigma0 per particle class (cells, beads)', True, False),
    ('sigma0_fallback', 'sigma0 of clamped detectors from the same laser (not the floor)', 'laser', 'floor'),
    ('clamp_mode', 'Detect data clamped at zero (unmix them with Poisson weights with a floor)', 'auto', 'off'),
)
_NOISE_OPTION_KEYS = ('per_control_quadratic', 'sigma0_by_class', 'sigma0_fallback')


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class NoiseFitWorker(QObject):
    """Runs gls_unmix.fit_experiment_noise_model in a QThread."""
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, snap: dict):
        super().__init__()
        self.snap = snap

    def run(self):
        try:
            self.finished.emit(gu.fit_experiment_noise_model(self.snap, self.progress.emit))
        except Exception as e:
            logger.exception('AutoSpectral Noise: Model Noise failed')
            self.error.emit(str(e) or type(e).__name__)


class NoiseComparisonWorker(QObject):
    """The comparison pair: per-cell AF (or OLS) and M2pg, as full unmixed arrays."""
    finished = Signal(object, object)
    error = Signal(str)

    def __init__(self, job: dict):
        super().__init__()
        self.job = job

    def run(self):
        try:
            j = self.job
            raw = j['raw']
            fl = np.ascontiguousarray(raw[:, j['fl_ids']], dtype=np.float64)
            af = j['af']
            if af is not None:
                left = apply_af_transfer(raw, j['transfer'], af.precomputed, af.spectra, j['settings'],
                                         filtered_fl_ids_raw=j['fl_ids'], spillover=j['spillover'],
                                         af_index_map=af.index_map)['unmixed']
            else:
                left = apply_transfer_matrix(j['transfer'], raw)
            res = gu.unmix_m2pg(fl, j['spectra'], j['noise'], None if af is None else af.spectra, j['envelope'],
                                null_var_no_af=j['null_var'], spread=j['spread'], prior_lambda=j['prior_lambda'],
                                ceiling=j['ceiling'], n_threads=j['n_threads'])
            right = gu.assemble_unmixed(raw, j['transfer'], j['settings'], res, spillover=j['spillover'],
                                        af_index_map=None if af is None else af.index_map)
            self.finished.emit(left, right)
        except Exception as e:
            logger.exception('AutoSpectral Noise: comparison failed')
            self.error.emit(str(e) or type(e).__name__)


def _selected_text(entry: dict | None) -> dict | None:
    """'selected / root (share)' per source of an automatic selection, keyed by the calibration set's name."""
    selection = (entry or {}).get('selection') or {}
    if selection.get('mode') != 'auto':
        return None
    counts = selection.get('counts') or {}
    out = {}
    for name, path in (selection.get('sources') or {}).items():
        n = counts.get(path)
        if n:
            out[name] = f'{n["n_singlet"]:,} / {n["n_root"]:,} ({n["n_singlet"] / max(n["n_root"], 1):.0%})'
    return out


def _spin(widget):
    """Wheel events and tab focus off, as for the tab's other spin boxes and combos."""
    widget.installEventFilter(WheelBlocker(widget))
    widget.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
    return widget


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

class NoisePage(QWidget):
    """AutoSpectral tab page: fit the detector noise model and compare noise-weighted unmixing."""

    def __init__(self, bus=None, controller=None, active_fn=None, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.controller = controller
        # active_fn(page) -> bool: whether the page is on screen.
        self._active_fn = active_fn

        self._fit_thread = None
        self._fit_token = None
        self._fit_worker = None
        self._cmp_thread = None
        self._cmp_worker = None
        self._cmp_running = False
        self._last_cmp_state = None
        self._pending_cmp_state = None
        self._left_data = None
        self._right_data = None
        self._cmp_note = ''
        # (unstained choice, bead choice) of the fit running now
        self._fit_choice = (gu.AUTO, 'auto')
        # (experiment dir, fingerprint, fit time) of the cached NoiseState
        self._state_key = None
        self._state = None

        self._lambda_timer = QTimer(self)
        self._lambda_timer.setSingleShot(True)
        self._lambda_timer.setInterval(LAMBDA_REFRESH_MS)
        self._lambda_timer.timeout.connect(self._on_lambda_settled)

        self._build_ui()

        if self.bus:
            self.bus.modeChangeRequested.connect(self._on_mode_change)
            self.bus.loadSampleRequested.connect(self._on_sample_loaded)
            self.bus.spectralProcessRefreshed.connect(self._on_process_refreshed)
            self.bus.sampleTreeUpdated.connect(self._refresh_selectors)
            self.bus.unmixingMethodChanged.connect(self._refresh_use_check)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._label_disabled = QLabel(
            'Noise: calculate the unmixing matrix (Spectral Process tab) before using this page.'
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

        content_layout.addWidget(HelpToggleWidget(text=autospectral_noise_help_text))
        if not gu.kernel_available():
            note = QLabel('<b>Compiled noise-model kernel not found.</b> Unmixing with the noise model runs in '
                          'NumPy, several times slower, here and in the experiment\'s unmixing (display, exports, '
                          'plugins). Run <tt>build_autospectral_opt_kernel.py</tt> (in bundled_plugins/) to '
                          'build it.')
            note.setTextFormat(Qt.RichText)
            note.setWordWrap(True)
            note.setStyleSheet('color: #cc6600;')
            content_layout.addWidget(note)
        self._build_fit_section(content_layout)
        self._build_compare_section(content_layout)
        content_layout.addStretch()

    def _build_fit_section(self, parent_layout):
        grp = QGroupBox('Model Noise')
        layout = QVBoxLayout(grp)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.DontWrapRows)

        self._unstained_combo = _spin(QComboBox())
        self._unstained_combo.setToolTip(
            'Unstained cells for the noise fit (and its AF basis). Auto takes, among the unstained cell '
            'samples with an AF profile, the one with the lowest mean signal. Every sample\'s events are '
            'selected automatically (scatter core and singlets); gates are not used.')
        form.addRow('Unstained cells:', self._unstained_combo)

        self._bead_combo = _spin(QComboBox())
        self._bead_combo.setToolTip(
            'Bead negatives anchor the noise floor (sigma0). Auto takes the unstained bead samples and the '
            'bead controls\' negatives, else unstained beads found inside the bead controls.')
        form.addRow('Bead negative:', self._bead_combo)

        self._lambda_spin = _spin(QDoubleSpinBox())
        self._lambda_spin.setRange(0.0, 5.0)
        self._lambda_spin.setSingleStep(0.25)
        self._lambda_spin.setDecimals(2)
        self._lambda_spin.setValue(gu.DEFAULT_PRIOR_LAMBDA)
        self._lambda_spin.setToolTip(
            'Strength of the positivity prior. Higher narrows the negatives further and over-estimates dim '
            'double positives of near-collinear dyes more (about +9% at 0.5 and +14% at 1).')
        self._lambda_spin.valueChanged.connect(self._on_lambda_changed)
        form.addRow('Positivity prior (lambda):', self._lambda_spin)
        layout.addLayout(form)

        self._options_toggle = QCheckBox('Show estimator options (validation)')
        layout.addWidget(self._options_toggle)
        self._options = QWidget()
        options_layout = QVBoxLayout(self._options)
        options_layout.setContentsMargins(16, 0, 0, 0)
        self._option_checks = {}
        for key, label, _on, _off in ESTIMATOR_OPTIONS:
            check = QCheckBox(label)
            self._option_checks[key] = check
            options_layout.addWidget(check)
        self._option_checks['clamp_mode'].setChecked(True)
        self._options.setVisible(False)
        self._options_toggle.toggled.connect(self._options.setVisible)
        layout.addWidget(self._options)

        row = QHBoxLayout()
        self._fit_btn = QPushButton('Model Noise')
        self._fit_btn.clicked.connect(self._run_fit)
        row.addWidget(self._fit_btn)
        row.addStretch()
        layout.addLayout(row)

        self._fit_status = QLabel('')
        self._fit_status.setWordWrap(True)
        layout.addWidget(self._fit_status)

        self._summary = QLabel('')
        self._summary.setTextFormat(Qt.RichText)
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        self._use_check = QCheckBox('Use the noise model for the experiment\'s unmixing')
        self._use_check.setToolTip(
            'Every sample is unmixed with the noise model and the positivity prior (M2pg): the main window, '
            'exports, Honeycluster, the gating plugins and Statistics. The main window shows per-cell AF first '
            'and swaps the noise-model result in when it is ready. Spectral variants are not used while this '
            'is on.')
        self._use_check.toggled.connect(self._on_use_toggled)
        layout.addWidget(self._use_check)

        self._clamped_check = QCheckBox('Use the weights for clamped data (with the positivity prior)')
        self._clamped_check.setToolTip(
            'The fit found data clamped at zero (no negative values on any detector), whose noise the model '
            'does not describe. The experiment is unmixed with Poisson weights from the fitted kappa with a '
            'floor (or flat weights), scaled to the unstained cells, and the positivity prior instead. The noise '
            'model, when ticked above, takes precedence.')
        self._clamped_check.toggled.connect(self._on_clamped_toggled)
        self._clamped_check.setVisible(False)
        layout.addWidget(self._clamped_check)
        # the noise model's use and lambda decide the unmixing: not changed under a running job
        guard_while_busy(self.controller, self._fit_btn, self._use_check, self._clamped_check, self._lambda_spin)
        if getattr(self.controller, 'job_gate', None) is not None:
            self.controller.job_gate.changed.connect(self._on_job_gate_changed)

        self._tables_toggle = QCheckBox('Show calibration and detector tables')
        layout.addWidget(self._tables_toggle)
        self._tables = QWidget()
        tables_layout = QVBoxLayout(self._tables)
        tables_layout.setContentsMargins(0, 0, 0, 0)
        tables_layout.addWidget(QLabel('Held-out events per source (observed / predicted residual variance):'))
        self._sources_table = CopyableTableWidget([], SOURCE_HEADERS)
        tables_layout.addWidget(self._sources_table)
        tables_layout.addWidget(QLabel('Detectors:'))
        self._detectors_table = CopyableTableWidget([], DETECTOR_HEADERS)
        tables_layout.addWidget(self._detectors_table)
        self._tables.setVisible(False)
        self._tables_toggle.toggled.connect(self._tables.setVisible)
        layout.addWidget(self._tables)

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
        self._plot_left = AfComparisonPlotWidget('Per-cell AF', self.controller, parent=self)
        self._plot_right = AfComparisonPlotWidget('Per-cell AF + noise model', self.controller, parent=self)
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

    def noise_state(self) -> gu.NoiseState | None:
        """The stored noise model for the current detectors, read once per fit."""
        c = self.controller
        entry = c.experiment.process.get(gu.PROCESS_KEY)
        if not entry:
            self._state_key, self._state = None, None
            return None
        key = (str(c.experiment_dir), entry.get('fingerprint'), entry.get('fitted'))
        if key != self._state_key:
            self._state_key = key
            self._state = gu.load_noise_state(c.experiment.process, c.experiment_dir,
                                              raw_layout(c)['detectors'])
        return self._state

    def _refresh_ui(self):
        available = self._available()
        self._label_disabled.setVisible(not available)
        self._content.setVisible(available)
        self._scroll.setVisible(available)
        if not available:
            return
        self._refresh_selectors()
        self._lambda_spin.blockSignals(True)
        self._lambda_spin.setValue(gu.prior_lambda(self.controller.experiment.process))
        self._lambda_spin.blockSignals(False)
        self._refresh_summary()
        self._refresh_use_check()
        self._refresh_comparison_controls()

    def _on_job_gate_changed(self):
        """Availability of the Use ticks again once the job gate is free."""
        if not self.controller.job_gate.is_held():
            self._refresh_use_check()

    def _refresh_use_check(self):
        """The Use tick: enabled with a readable noise model, ticked while it is the experiment's method."""
        if not self._available():
            return
        process = self.controller.experiment.process
        state = self.noise_state()
        has_model = state is not None
        self._use_check.blockSignals(True)
        self._use_check.setEnabled(has_model)
        self._use_check.setChecked(has_model and unmix_method.unmixing_settings(process)['noise_model'])
        self._use_check.blockSignals(False)
        clamped = has_model and bool(state.entry.get('clamped')) and state.clamped_model is not None
        self._clamped_check.blockSignals(True)
        self._clamped_check.setVisible(clamped)
        self._clamped_check.setChecked(clamped and gu.clamped_weights_requested(process))
        self._clamped_check.blockSignals(False)

    def _method_changed(self):
        """The experiment's unmixing changed: unmix the displayed sample again and tell the other pages."""
        if self.bus:
            self.bus.autoSaveRequested.emit()
        if self.controller is not None:
            self.controller.refresh_unmixing()
        if self.bus:
            self.bus.unmixingMethodChanged.emit()

    def _on_use_toggled(self, checked: bool):
        unmix_method.set_unmixing(self.controller.experiment.process, noise_model=bool(checked))
        self._method_changed()

    def _on_clamped_toggled(self, checked: bool):
        unmix_method.set_unmixing(self.controller.experiment.process,
                                  weights=gu.CLAMPED_WEIGHTS if checked else None)
        self._method_changed()

    def _on_lambda_settled(self):
        if self.controller is not None and gu.noise_in_use(self.controller.experiment.process):
            self._method_changed()

    def _refresh_selectors(self):
        if not self._available():
            return
        c = self.controller
        samples = c.experiment.samples
        entry = c.experiment.process.get(gu.PROCESS_KEY) or {}
        detectors = raw_layout(c)['detectors']

        current = self._unstained_combo.currentData() or entry.get('unstained_choice') or gu.AUTO
        self._unstained_combo.blockSignals(True)
        self._unstained_combo.clear()
        self._unstained_combo.addItem('Auto (lowest mean signal with an AF profile)', gu.AUTO)
        for path, name in gu.unstained_cell_samples(samples):
            spectra, _sidecar = gu.stored_af_for(c, path, detectors)
            self._unstained_combo.addItem(name if spectra is not None else f'{name} (no AF profile)', path)
        i = self._unstained_combo.findData(current)
        self._unstained_combo.setCurrentIndex(max(i, 0))
        self._unstained_combo.blockSignals(False)

        options = entry.get('options') or {}
        for key, _label, on, _off in ESTIMATOR_OPTIONS:
            if key in options:
                self._option_checks[key].setChecked(options[key] == on)

        current = self._bead_combo.currentData() or entry.get('bead_choice') or 'auto'
        beads = gu.unstained_bead_samples(samples)
        self._bead_combo.blockSignals(True)
        self._bead_combo.clear()
        if beads:
            auto_text = 'Auto: ' + ', '.join(n for _p, n in beads)
        else:
            auto_text = 'Auto: the bead controls\' negatives (none marked unstained)'
        self._bead_combo.addItem(auto_text, 'auto')
        for path, name in beads:
            self._bead_combo.addItem(name, path)
        self._bead_combo.addItem('None', NONE_KEY)
        i = self._bead_combo.findData(current)
        self._bead_combo.setCurrentIndex(max(i, 0))
        self._bead_combo.blockSignals(False)

    def _refresh_summary(self):
        c = self.controller
        entry = c.experiment.process.get(gu.PROCESS_KEY)
        if not entry:
            self._summary.setText('No noise model yet: choose the samples and press <b>Model Noise</b>.')
            self._fill_tables(None, None)
            return
        state = self.noise_state()
        names = c.experiment.samples.get('all_samples', {})
        cal = entry.get('calibration') or {}
        band = cal.get('band') or [0.8, 1.25]
        in_band = cal.get('in_band_controls')
        model = None if state is None else state.model
        beads = ', '.join(f'{names.get(b["name"], b["name"])} ({b["source"]})' for b in entry.get('beads') or [])
        selection = entry.get('selection') or {}
        if selection.get('mode') == 'auto':
            counts = list((selection.get('counts') or {}).values())
            n_sel = sum(x['n_singlet'] for x in counts)
            n_root = sum(x['n_root'] for x in counts)
            where = (f'events selected automatically (scatter core and singlets): {n_sel:,} of {n_root:,} '
                     f'({n_sel / max(n_root, 1):.0%}) over {len(counts)} samples')
        else:
            where = f'in {entry.get("gate")}'
        lines = [
            f'<b>Fitted</b> {entry.get("fitted", "")} from {entry.get("n_controls", 0)} controls, '
            f'unstained cells <i>{names.get(entry.get("unstained"), entry.get("unstained"))}</i>, '
            f'bead negatives: {beads or "none"}; {where}.',
        ]
        options = entry.get('options') or {}
        used = [label for key, label, on, _off in ESTIMATOR_OPTIONS
                if key != 'clamp_mode' and options.get(key) == on]
        if used:
            lines.append('Estimator options: ' + '; '.join(used) + '.')
        if in_band is not None:
            what = ' (calibration with per-control spread)' if cal.get('per_control_c') else ''
            lines.append(f'Held-out control cells within {band[0]:g}–{band[1]:g} of the predicted variance{what}: '
                         f'<b>{in_band:.1%}</b> (target 80% or more).')
        if model is not None:
            if model.sigma0_sq_by_class is not None:
                levels = ', '.join(f'{cls} {float(np.median(np.sqrt(row))):,.0f}'
                                   for cls, row in zip(model.classes, model.sigma0_sq_by_class))
                sigma = f'Median sigma0: {levels}'
            else:
                sigma = f'Median sigma0 {float(np.median(np.sqrt(model.sigma0_sq))):,.0f}'
            lines.append(f'{sigma}; {entry.get("flagged_detectors", 0)} of {len(model.detectors)} detectors '
                         'flagged.')
        else:
            lines.append('<span style="color:#cc6600">The stored noise model could not be read for the current '
                         'detectors: press Model Noise again.</span>')
        if entry.get('clamped'):
            kind = (entry.get('clamped_weights') or {}).get('kind')
            lines.append('<span style="color:#cc6600"><b>Clamped data:</b> the negatives have no values at or below '
                         'zero, so their noise is cut off at zero, which the noise model does not describe. Model '
                         f'Noise has left the noise model off and unmixes with {gu.clamped_weights_text(kind)} and '
                         'the positivity prior instead.</span>')
        names_f, spectra = gu.fluor_names_and_spectra(c)
        if spectra is not None and gu.spectra_changed(entry, names_f, spectra):
            lines.append('<span style="color:#cc6600">The spectra have changed since the noise model was fitted; '
                         'it still describes the detectors, but fitting it again is recommended.</span>')
        for note in entry.get('notes') or []:
            lines.append(f'Note: {note}')
        self._summary.setText('<br>'.join(lines))
        self._fill_tables(entry, model)

    def _replace_table(self, attr: str, rows: list, headers: list):
        old = getattr(self, attr)
        new = CopyableTableWidget(rows, headers)
        self._tables.layout().replaceWidget(old, new)
        old.setParent(None)
        old.deleteLater()
        setattr(self, attr, new)

    def _fill_tables(self, entry: dict | None, model: nm.NoiseModel | None):
        groups = [] if entry is None else (entry.get('calibration') or {}).get('groups') or []
        selected = _selected_text(entry)
        headers, rows = nm.groups_table({'groups': groups}, selected)
        self._replace_table('_sources_table', [dict(zip(headers, r)) for r in rows], headers)
        headers, rows = (DETECTOR_HEADERS, []) if model is None else nm.detector_table(model)
        self._replace_table('_detectors_table', [dict(zip(headers, r)) for r in rows], headers)

    def _on_lambda_changed(self, value: float):
        process = self.controller.experiment.process
        process.setdefault(gu.UNMIXING_KEY, {})['prior_lambda'] = float(value)
        if self.bus:
            self.bus.autoSaveRequested.emit()
        self._refresh_comparison_controls()
        self._lambda_timer.start()

    # ------------------------------------------------------------------
    # Model Noise
    # ------------------------------------------------------------------

    def _run_fit(self):
        if self._fit_thread is not None:
            self._fit_status.setText('Already fitting — please wait.')
            return
        bead = self._bead_combo.currentData() or 'auto'
        settings = gu.NoiseSetupSettings(bead_mode='none' if bead == NONE_KEY else bead)
        for key, _label, on, off in ESTIMATOR_OPTIONS:
            value = on if self._option_checks[key].isChecked() else off
            setattr(settings.noise if key in _NOISE_OPTION_KEYS else settings, key, value)
        base_gate, _ = resolve_base_gate(self.controller.raw_gating,
                                         self.controller.experiment.process.get('base_gate_priority_order', []))
        try:
            snap = gu.noise_setup_snapshot(self.controller, settings, self._unstained_combo.currentData() or gu.AUTO,
                                           base_gate or 'root')
        except Exception as e:
            self._fit_status.setText(f'Error: {e}')
            return
        token = acquire_job(self.controller, 'AutoSpectral: Model Noise')
        if token is None:
            self._fit_status.setText(self.controller.job_gate.busy_text())
            return
        self._fit_token = token
        self._fit_choice = (self._unstained_combo.currentData() or gu.AUTO, bead)
        self._fit_btn.setEnabled(False)
        self._fit_status.setText('Fitting the noise model...')
        self._fit_thread = QThread()
        self._fit_worker = NoiseFitWorker(snap)
        self._fit_worker.moveToThread(self._fit_thread)
        self._fit_thread.started.connect(self._fit_worker.run)
        self._fit_worker.progress.connect(self._fit_status.setText)
        self._fit_worker.finished.connect(self._on_fit_finished)
        self._fit_worker.error.connect(self._on_fit_error)
        self._fit_worker.finished.connect(self._fit_thread.quit)
        self._fit_worker.error.connect(self._fit_thread.quit)
        self._fit_thread.finished.connect(self._fit_thread.deleteLater)
        self._fit_thread.finished.connect(self._on_fit_thread_finished)
        self._fit_thread.start()

    def _on_fit_finished(self, fit: gu.NoiseFit):
        self._fit_token.release()
        c = self.controller
        try:
            gu.store_noise_fit(c.experiment.process, c.experiment_dir, fit, self._lambda_spin.value())
        except OSError as e:
            self._fit_status.setText(f'Error: the noise model could not be saved ({e}).')
            return
        entry = c.experiment.process[gu.PROCESS_KEY]
        entry['unstained_choice'], entry['bead_choice'] = self._fit_choice
        if self.bus:
            self.bus.autoSaveRequested.emit()
        name = c.experiment.samples.get('all_samples', {}).get(fit.unstained_path, fit.unstained_path)
        if fit.clamped:
            self._fit_status.setText(f'Done. Noise model fitted with unstained cells {name}. The data are clamped '
                                     'at zero, so the experiment is now unmixed with '
                                     f'{gu.clamped_weights_text(fit.clamped_weights.get("kind"))} and the '
                                     'positivity prior, not the noise model.')
        else:
            self._fit_status.setText(f'Done. Noise model fitted with unstained cells {name}; the experiment is '
                                     'now unmixed with it.')
        self._refresh_summary()
        self._refresh_use_check()
        self._method_changed()
        self._refresh_comparison_controls()
        if self.is_active() and self._cmp_run_btn.isEnabled():
            self._run_comparison()

    def _on_fit_error(self, msg: str):
        self._fit_token.release()
        self._fit_status.setText(f'Error: {msg}')

    def _on_fit_thread_finished(self):
        self._fit_token.release()
        self._fit_thread = None
        self._fit_worker = None
        self._fit_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def _assigned_profiles(self) -> list:
        c = self.controller
        return list(c.experiment.samples.get('sample_af_profiles', {}).get(c.current_sample_path, []) or [])

    def _refresh_comparison_controls(self):
        has_model = self._available() and self.noise_state() is not None
        self._cmp_run_btn.setEnabled(has_model)
        if not has_model:
            self._cmp_status.setText('Model the noise first.')
        else:
            self._cmp_status.setText('')
        with_af = bool(self._assigned_profiles())
        right = 'clamped-data weights' if self._clamped_in_use() else 'noise model'
        titles = (('Per-cell AF', f'Per-cell AF + {right}') if with_af
                  else ('OLS (no AF)', f'{right[0].upper()}{right[1:]} (no AF)'))
        for plot, title in zip((self._plot_left, self._plot_right), titles):
            plot._has_af_channels = with_af
            if plot._title_text != title:
                plot._title_text = title
                plot._configure_title()

    def _clamped_in_use(self) -> bool:
        """The clamped-data weights are the experiment's unmixing rather than the noise model."""
        if not self._available():
            return False
        process = self.controller.experiment.process
        state = self.noise_state()
        return (state is not None and state.clamped_model is not None
                and unmix_method.requested_method(process) == unmix_method.CLAMPED)

    def _clear_comparison_plots(self):
        self._plot_left.clear()
        self._plot_right.clear()
        self._left_data = None
        self._right_data = None
        self._last_cmp_state = None
        self._pending_cmp_state = None

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
        state = self.noise_state()
        if state is None:
            self._cmp_status.setText('Model the noise first.')
            return
        _names, spectra = gu.fluor_names_and_spectra(c)
        if spectra is None:
            self._cmp_status.setText('Fluorophore profiles unavailable.')
            return
        layout = raw_layout(c)
        profiles = self._assigned_profiles()
        af = resolve_af_for_profiles(c, profiles) if profiles else None
        if profiles and af is None:
            self._cmp_status.setText('AF matrices are not available yet for this sample. Wait for the spectral '
                                     'process to finish, then run the comparison again.')
            return
        note = ''
        envelope = null_var = spread = None
        bead = is_bead_sample(c.experiment.samples, c.current_sample_path)
        clamped = self._clamped_in_use()
        noise = state.clamped_model if clamped else state.model_for(bead)
        if af is not None:
            envelope = af_envelope_for_profiles(c, profiles)
            if envelope is None:
                note = (' The AF profiles have no negative envelope, so the noise model runs without the '
                        'positivity prior; extract the profiles again to add it.')
        else:
            null_var = state.bead_null_var if bead else state.cell_null_var
            if null_var is None:
                note = ' No bead negative was fitted, so beads run without the positivity prior.'
            names_f, _s = gu.fluor_names_and_spectra(c)
            entry = c.experiment.process.get('af_spread')
            if spread_is_current(entry, names_f, spectra):
                spread = np.asarray(entry['spread'], dtype=np.float64)
        lam = gu.prior_lambda(c.experiment.process)
        n = len(c.raw_event_data)
        cap = int(self._cmp_events_spin.value())
        rows = np.arange(n) if n <= cap else np.sort(np.random.default_rng(0).choice(n, cap, replace=False))
        state_key = (
            c.current_sample_path, tuple(profiles),
            None if af is None else bytes(np.ascontiguousarray(af.spectra).data),
            self._state_key, lam, cap,
            None if envelope is None else np.asarray(envelope['null_var']).tobytes(), bead, clamped,
        )
        if state_key == self._last_cmp_state and self._left_data is not None:
            self._cmp_status.setText(
                f'No change — redrawing from cached result ({self._left_data.shape[0]:,} events).{note}')
            self._initialise_comparison_plots()
            return

        self._plot_left.set_status('Computing...')
        self._plot_right.set_status('Computing...')
        self._cmp_status.setText('Running unmixing on current sample...')
        self._cmp_note = note
        job = {
            'raw': np.ascontiguousarray(c.raw_event_data[rows]), 'fl_ids': layout['fl_ids'],
            'transfer': np.array(c.transfer_matrix, dtype=np.float64, copy=True),
            'settings': c.experiment.settings, 'spillover': c.experiment.process.get('spillover'),
            'spectra': spectra, 'noise': noise, 'af': af, 'envelope': envelope,
            'null_var': null_var, 'spread': spread, 'prior_lambda': lam, 'ceiling': layout['ceiling'],
            'n_threads': max(1, (os.cpu_count() or 2) - 1),
        }
        self._pending_cmp_state = state_key
        self._cmp_running = True
        self._cmp_thread = QThread()
        self._cmp_worker = NoiseComparisonWorker(job)
        self._cmp_worker.moveToThread(self._cmp_thread)
        self._cmp_thread.started.connect(self._cmp_worker.run)
        self._cmp_worker.finished.connect(self._on_comparison_finished)
        self._cmp_worker.error.connect(self._on_comparison_error)
        self._cmp_worker.finished.connect(self._cmp_thread.quit)
        self._cmp_worker.error.connect(self._cmp_thread.quit)
        self._cmp_thread.finished.connect(self._cmp_thread.deleteLater)
        self._cmp_thread.finished.connect(self._on_cmp_thread_finished)
        self._cmp_thread.start()

    def _on_comparison_finished(self, left, right):
        self._left_data = left
        self._right_data = right
        self._last_cmp_state = self._pending_cmp_state
        self._cmp_status.setText(f'Done. {left.shape[0]:,} events.{self._cmp_note}')
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
            logger.debug(f'AutoSpectral Noise comparison: channel auto-pick failed: {e}')
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
        if self._available() and self._cmp_run_btn.isEnabled() and self.controller.raw_event_data is not None:
            QTimer.singleShot(0, self._run_comparison)

    def _on_mode_change(self, mode):
        if mode == TAB_NAME and self.is_active():
            self.on_activated()

    def _on_sample_loaded(self, _path):
        QTimer.singleShot(0, self._on_sample_loaded_deferred)

    def _on_sample_loaded_deferred(self):
        if not self._available():
            return
        self._refresh_comparison_controls()
        if not self.is_active():
            self._clear_comparison_plots()
            return
        if self._cmp_run_btn.isEnabled():
            self._run_comparison()

    def _on_process_refreshed(self):
        self._state_key = None
        if self.is_active():
            self._refresh_ui()
