"""
autospectral_variants_page.py
-----------------------------
AutoSpectral tab page: Spectral Variants (per-cell fluorophore + AF
optimisation with the compiled AutoSpectral kernel).

Sections
--------
1  Compute Spectral Variants: discover per-fluorophore variants from the
   single-stained controls (``discover_all_variants``); the necessity
   recommendation sets which fluorophores are optimised.
2  Plot Spectral Variants: a fluorophore's variants against its reference.
3  Comparison: the current sample with per-cell AF | per-cell AF + spectral
   variants.

The numerics live in ``bundled_plugins/autospectral_optimization_functions.py``
and the kernel wrapper beside it, imported from the bundled-plugins folder.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QObject, QSettings, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter,
    QVBoxLayout, QWidget,
)

from honeychrome.controller_components.functions import apply_transfer_matrix
from honeychrome.controller_components.autospectral_functions import (
    build_af_precomputed, apply_af_unmixing,
)
from honeychrome.controller_components.cytometer_whitelist import (
    get_detector_laser_map, LASER_LABEL_COLORS,
)
from honeychrome.controller_components.plugin_loaders import bundled_plugins_path
from honeychrome.view_components.autospectral_tab import (
    TAB_NAME,
    AfComparisonPlotWidget,
    pick_most_affected_channels,
)
from honeychrome.view_components.help_toggle_widget import HelpToggleWidget, WheelBlocker
from honeychrome.view_components.help_texts import autospectral_variants_help_text
from honeychrome.view_components.profiles_viewer import BottomAxisVerticalTickLabels
import honeychrome.settings as settings

logger = logging.getLogger(__name__)

if str(bundled_plugins_path) not in sys.path:
    sys.path.insert(0, str(bundled_plugins_path))

from autospectral_optimization_functions import (   # noqa: E402
    discover_all_variants,
    calculate_optimize_necessity,
    unmix_autospectral_optimization,
    fluorescence_channel_names,
    af_envelope_for_profiles,
)
from autospectral_opt_kernel_wrapper import AUTOSPECTRAL_OPT_KERNEL_AVAILABLE  # noqa: E402

SHOW_ADVANCED_SETTING = 'autospectral_variants_show_advanced'


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class VariantSetupWorker(QObject):
    """Runs discover_all_variants() in a QThread."""
    finished = Signal(dict, object, object)   # (variants, raw_pos_thresholds, unmixed_pos_thresholds)
    error = Signal(str)
    progress = Signal(int, int, str)

    def __init__(self, controller, n_cells, som_dim, k_neighbors, sim_threshold,
                 dedup_threshold=0.01, max_variants=10):
        super().__init__()
        self.controller = controller
        self.n_cells = n_cells
        self.som_dim = som_dim
        self.k_neighbors = k_neighbors
        self.sim_threshold = sim_threshold
        self.dedup_threshold = dedup_threshold
        self.max_variants = max_variants

    def run(self):
        try:
            def _progress(n, total, label):
                self.progress.emit(n, total, label)

            output = discover_all_variants(
                self.controller,
                n_cells=self.n_cells,
                som_dim=self.som_dim,
                k_neighbors=self.k_neighbors,
                sim_threshold=self.sim_threshold,
                dedup_threshold=self.dedup_threshold,
                max_variants=self.max_variants,
                progress_callback=_progress,
            )
            self.finished.emit(
                output['variants'], output['raw_pos_thresholds'], output['unmixed_pos_thresholds'],
            )
        except Exception as e:
            logger.exception('AutoSpectral Spectral Variants: computing variants failed')
            self.error.emit(str(e))


class JointComparisonWorker(QObject):
    """Computes OLS, AF-corrected, and AutoSpectral-Optimization arrays in a QThread."""
    finished = Signal(object, object, object)   # (ols_data, af_data, opt_data) full unmixed-shape arrays
    error = Signal(str)

    def __init__(self, raw_event_data, transfer_matrix, filtered_fl_ids_raw,
                 fl_ids_unmixed, reference_spectra, fluor_names, af_spectra,
                 variants_meta, active_labels, unmixed_pos_thresholds,
                 spillover, max_events, kernel_kwargs,
                 unmixed_pnn=None, af_index_map=None, envelope=None):
        super().__init__()
        self.envelope = envelope
        self.raw_event_data = raw_event_data
        self.transfer_matrix = transfer_matrix
        self.filtered_fl_ids_raw = filtered_fl_ids_raw
        self.fl_ids_unmixed = fl_ids_unmixed
        self.reference_spectra = reference_spectra
        self.fluor_names = fluor_names
        self.af_spectra = af_spectra
        self.variants_meta = variants_meta
        self.active_labels = active_labels
        self.unmixed_pos_thresholds = unmixed_pos_thresholds
        self.spillover = spillover
        self.max_events = max_events
        self.kernel_kwargs = kernel_kwargs
        self.unmixed_pnn = list(unmixed_pnn or [])
        self.af_index_map = af_index_map

    def _fill_af_channels(self, data, af_scale, af_idx):
        """Write per-cell AF abundance and AF Index into data's AF channels."""
        af_index = af_idx
        if self.af_index_map is not None:
            af_index = np.asarray(self.af_index_map)[np.asarray(af_idx, dtype=np.int64) - 1]
        for label, values in ((settings.af_abundance_channel, af_scale),
                              (settings.af_index_channel, af_index)):
            if label in self.unmixed_pnn:
                data[:, self.unmixed_pnn.index(label)] = values

    def run(self):
        try:
            n = self.raw_event_data.shape[0]
            if n > self.max_events:
                rng = np.random.default_rng(0)
                idx = np.sort(rng.choice(n, self.max_events, replace=False))
            else:
                idx = np.arange(n)
            raw_sub = self.raw_event_data[idx]
            raw_fl = raw_sub[:, self.filtered_fl_ids_raw]

            # Panel 1: plain OLS, no AF correction — baseline.
            plain_ols_data = apply_transfer_matrix(self.transfer_matrix, raw_sub)

            # Panel 2: AutoSpectral AF-corrected OLS (no variant optimization).
            af_precomputed = build_af_precomputed(self.reference_spectra, self.af_spectra, self.envelope)
            af_result = apply_af_unmixing(raw_fl, af_precomputed, self.af_spectra)
            af_unmixed_fl = af_result['unmixed']
            if self.spillover is not None:
                compensation = np.linalg.inv(np.array(self.spillover)).T
                af_unmixed_fl = (compensation @ af_unmixed_fl.T).T

            af_data = plain_ols_data.copy()
            af_data[:, self.fl_ids_unmixed] = af_unmixed_fl
            self._fill_af_channels(af_data, af_result['af_scale'], af_result['af_idx'])

            # Panel 3: AutoSpectral Optimization (per-fluorophore variant optimization).
            opt_result = unmix_autospectral_optimization(
                raw_fl_events=raw_fl,
                reference_spectra=self.reference_spectra,
                fluor_names=self.fluor_names,
                af_spectra=self.af_spectra,
                variants_meta=self.variants_meta,
                active_labels=self.active_labels,
                unmixed_pos_thresholds=self.unmixed_pos_thresholds,
                envelope=self.envelope,
                **self.kernel_kwargs,
            )
            opt_unmixed_fl = opt_result['unmixed']
            if self.spillover is not None:
                compensation = np.linalg.inv(np.array(self.spillover)).T
                opt_unmixed_fl = (compensation @ opt_unmixed_fl.T).T

            opt_data = plain_ols_data.copy()
            opt_data[:, self.fl_ids_unmixed] = opt_unmixed_fl
            self._fill_af_channels(opt_data, opt_result['af_scale'], opt_result['af_idx'])

            self.finished.emit(plain_ols_data, af_data, opt_data)
        except Exception as e:
            logger.exception('AutoSpectral Optimization: comparison failed')
            self.error.emit(str(e))


def _spin(widget):
    """Wheel events and tab focus off, as for the tab's other spin boxes and combos."""
    widget.installEventFilter(WheelBlocker(widget))
    widget.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
    return widget


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

class SpectralVariantsPage(QWidget):
    """AutoSpectral tab page: compute, plot and compare per-cell spectral variants."""

    def __init__(self, bus=None, controller=None, active_fn=None, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.controller = controller
        # active_fn(page) -> bool: whether the page is on screen.
        self._active_fn = active_fn
        self._qsettings = QSettings('honeychrome', 'app_configuration')

        self._setup_thread = None
        self._setup_worker = None

        self._cmp_thread = None
        self._cmp_worker = None
        self._cmp_running = False
        self._last_cmp_state = None
        self._pending_cmp_state = None
        self._af_data = None
        self._opt_data = None

        self._build_ui()

        if self.bus:
            self.bus.modeChangeRequested.connect(self._on_mode_change)
            self.bus.loadSampleRequested.connect(self._on_sample_loaded)
            self.bus.spectralProcessRefreshed.connect(self._on_process_refreshed)

        if self._setup_done():
            self._refresh_plot_combo()
        self._refresh_all_guards()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._label_disabled = QLabel(
            'Spectral Variants: calculate the unmixing matrix (Spectral Process tab) '
            'before using this page.'
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

        content_layout.addWidget(HelpToggleWidget(text=autospectral_variants_help_text))

        if not AUTOSPECTRAL_OPT_KERNEL_AVAILABLE:
            kernel_warning = QLabel(
                '<b>Compiled kernel not found.</b> Run '
                '<tt>build_autospectral_opt_kernel.py</tt> (in bundled_plugins/) '
                'before computing or comparing spectral variants.'
            )
            kernel_warning.setTextFormat(Qt.RichText)
            kernel_warning.setStyleSheet('color: #cc6600;')
            content_layout.addWidget(kernel_warning)

        self._build_advanced_settings(content_layout)
        self._build_setup_section(content_layout)
        self._build_plot_section(content_layout)
        self._build_compare_section(content_layout)
        content_layout.addStretch()

    # ------------------------------------------------------------------
    # Advanced settings
    # ------------------------------------------------------------------

    def _build_advanced_settings(self, parent_layout):
        self._adv_toggle = QCheckBox('Show Advanced Settings')
        saved = self._qsettings.value(SHOW_ADVANCED_SETTING, 'false') == 'true'
        self._adv_toggle.setChecked(saved)
        parent_layout.addWidget(self._adv_toggle)

        adv_content = QWidget()
        adv_layout = QHBoxLayout(adv_content)
        adv_content.setVisible(saved)
        self._adv_toggle.toggled.connect(adv_content.setVisible)
        self._adv_toggle.toggled.connect(
            lambda c: self._qsettings.setValue(SHOW_ADVANCED_SETTING, 'true' if c else 'false')
        )

        # --- Setup group ---
        setup_grp = QGroupBox('Setup')
        setup_form = QFormLayout(setup_grp)
        self._n_cells_spin = QSpinBox(); self._n_cells_spin.setRange(100, 1_000_000); self._n_cells_spin.setValue(10_000)
        self._som_dim_spin = QSpinBox(); self._som_dim_spin.setRange(2, 20); self._som_dim_spin.setValue(5)
        self._k_neighbors_spin = QSpinBox(); self._k_neighbors_spin.setRange(1, 50); self._k_neighbors_spin.setValue(3)
        self._sim_threshold_spin = QDoubleSpinBox(); self._sim_threshold_spin.setRange(0.0, 1.0)
        self._sim_threshold_spin.setDecimals(3); self._sim_threshold_spin.setSingleStep(0.005); self._sim_threshold_spin.setValue(0.985)
        self._dedup_threshold_spin = QDoubleSpinBox(); self._dedup_threshold_spin.setRange(0.0, 1.0)
        self._dedup_threshold_spin.setDecimals(3); self._dedup_threshold_spin.setSingleStep(0.005); self._dedup_threshold_spin.setValue(0.01)
        self._dedup_threshold_spin.setToolTip(
            'Variants whose largest single-detector difference from the reference, or from a '
            'better-populated variant, is below this fraction of the peak are dropped. '
            'The two variants furthest from the reference are always kept.'
        )
        self._max_variants_spin = QSpinBox(); self._max_variants_spin.setRange(1, 100); self._max_variants_spin.setValue(10)
        self._max_variants_spin.setToolTip(
            'Most variants kept per fluorophore (excluding the reference). The dedup threshold '
            'is raised for a fluorophore until it fits.'
        )
        for w in (self._n_cells_spin, self._som_dim_spin, self._k_neighbors_spin, self._sim_threshold_spin,
                  self._dedup_threshold_spin, self._max_variants_spin):
            _spin(w)
        setup_form.addRow('n.cells:', self._n_cells_spin)
        setup_form.addRow('som.dim:', self._som_dim_spin)
        setup_form.addRow('k.neighbors:', self._k_neighbors_spin)
        setup_form.addRow('sim.threshold:', self._sim_threshold_spin)
        setup_form.addRow('dedup.threshold:', self._dedup_threshold_spin)
        setup_form.addRow('max.variants:', self._max_variants_spin)
        adv_layout.addWidget(setup_grp)

        # --- Necessity group ---
        necessity_grp = QGroupBox('Necessity')
        necessity_form = QFormLayout(necessity_grp)
        self._necessity_threshold_spin = _spin(QDoubleSpinBox())
        self._necessity_threshold_spin.setRange(0.0, 1.0); self._necessity_threshold_spin.setDecimals(3)
        self._necessity_threshold_spin.setSingleStep(0.005); self._necessity_threshold_spin.setValue(0.01)
        self._necessity_threshold_spin.setToolTip(
            'Fluorophores whose normalised necessity score reaches this value are optimised '
            'per cell; the others keep their reference spectrum. Applied when variants are computed.'
        )
        necessity_form.addRow('optimize.necessity.threshold:', self._necessity_threshold_spin)
        adv_layout.addWidget(necessity_grp)

        # --- Unmixing group ---
        unmixing_grp = QGroupBox('Unmixing')
        unmixing_form = QFormLayout(unmixing_grp)
        self._n_passes_spin = QSpinBox(); self._n_passes_spin.setRange(1, 20); self._n_passes_spin.setValue(1)
        self._n_af_passes_spin = QSpinBox(); self._n_af_passes_spin.setRange(1, 20); self._n_af_passes_spin.setValue(1)
        self._refine_af_quantile_spin = QDoubleSpinBox(); self._refine_af_quantile_spin.setRange(0.0, 1.0)
        self._refine_af_quantile_spin.setDecimals(2); self._refine_af_quantile_spin.setSingleStep(0.05); self._refine_af_quantile_spin.setValue(0.5)
        self._cell_weight_check = QCheckBox('cell.weight')
        self._noise_floor_spin = QDoubleSpinBox(); self._noise_floor_spin.setRange(0.0, 1_000_000.0); self._noise_floor_spin.setValue(125.0)
        self._alpha_spin = QDoubleSpinBox(); self._alpha_spin.setRange(0.0, 1.0); self._alpha_spin.setDecimals(2); self._alpha_spin.setSingleStep(0.05); self._alpha_spin.setValue(0.5)
        self._collinear_threshold_spin = QDoubleSpinBox(); self._collinear_threshold_spin.setRange(0.0, 1.0); self._collinear_threshold_spin.setDecimals(2); self._collinear_threshold_spin.setValue(0.5)
        self._joint_pair_resolution_check = QCheckBox('joint.pair.resolution'); self._joint_pair_resolution_check.setChecked(True)
        _cpu_count = os.cpu_count() or 1
        self._threads_spin = QSpinBox(); self._threads_spin.setRange(1, _cpu_count); self._threads_spin.setValue(max(1, _cpu_count - 1))
        for w in (self._n_passes_spin, self._n_af_passes_spin, self._refine_af_quantile_spin,
                  self._noise_floor_spin, self._alpha_spin, self._collinear_threshold_spin, self._threads_spin):
            _spin(w)
        unmixing_form.addRow('n.passes:', self._n_passes_spin)
        unmixing_form.addRow('n.af.passes:', self._n_af_passes_spin)
        unmixing_form.addRow('refine.af.quantile:', self._refine_af_quantile_spin)
        unmixing_form.addRow(self._cell_weight_check)
        unmixing_form.addRow('noise.floor:', self._noise_floor_spin)
        unmixing_form.addRow('alpha:', self._alpha_spin)
        unmixing_form.addRow('collinear.threshold:', self._collinear_threshold_spin)
        unmixing_form.addRow(self._joint_pair_resolution_check)
        unmixing_form.addRow('threads:', self._threads_spin)
        adv_layout.addWidget(unmixing_grp)

        # --- Comparison group ---
        comparison_grp = QGroupBox('Comparison')
        comparison_form = QFormLayout(comparison_grp)
        self._comparison_max_events_spin = _spin(QSpinBox())
        self._comparison_max_events_spin.setRange(1_000, 2_000_000)
        self._comparison_max_events_spin.setSingleStep(5_000)
        self._comparison_max_events_spin.setValue(30_000)
        comparison_form.addRow('comparison.max_events:', self._comparison_max_events_spin)
        adv_layout.addWidget(comparison_grp)

        parent_layout.addWidget(adv_content)

        # Pre-set cell.weight from cytometer, still user-overridable.
        db_col = None
        if self.controller and self.controller.experiment:
            db_col = self.controller.experiment.settings.get('raw', {}).get('cytometer_db_col')
        self._cell_weight_check.setChecked(db_col == 'ID7000')

    def _kernel_kwargs(self) -> dict:
        """Advanced Settings values, packaged for unmix_autospectral_optimization()."""
        return dict(
            n_passes=self._n_passes_spin.value(),
            n_threads=self._threads_spin.value(),
            cell_weight=self._cell_weight_check.isChecked(),
            noise_floor=self._noise_floor_spin.value(),
            alpha=self._alpha_spin.value(),
            collinear_thresh=self._collinear_threshold_spin.value(),
            joint_pair_resolution=self._joint_pair_resolution_check.isChecked(),
            n_af_passes=self._n_af_passes_spin.value(),
            refine_af_quantile=self._refine_af_quantile_spin.value(),
        )

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        """True while the page is on screen."""
        if self._active_fn is None:
            return self.controller is not None and self.controller.current_mode == TAB_NAME
        return self._active_fn(self)

    def _gating_available(self) -> bool:
        return bool(self.controller and self.controller.experiment
                    and self.controller.experiment.process.get('unmixing_matrix') is not None)

    def _setup_done(self) -> bool:
        return bool(getattr(self.controller, 'autospectral_variants', None))

    def _sample_has_af_profile(self, sample_path) -> bool:
        if not sample_path:
            return False
        assigned = self.controller.experiment.samples.get('sample_af_profiles', {}).get(sample_path, [])
        af_profiles = self.controller.experiment.process.get('af_profiles', {})
        n_af = sum(len(af_profiles[name]['spectra']) for name in assigned if name in af_profiles)
        return n_af >= 2

    def _refresh_all_guards(self):
        available = self._gating_available()
        self._label_disabled.setVisible(not available)
        self._content.setVisible(available)
        self._scroll.setVisible(available)
        if not available:
            return
        setup_done = self._setup_done()
        self._plot_combo.setEnabled(setup_done)
        self._plot_status.setText('' if setup_done else 'Compute spectral variants first.')
        self._refresh_comparison_controls()

    # ------------------------------------------------------------------
    # Compute Spectral Variants
    # ------------------------------------------------------------------

    def _build_setup_section(self, parent_layout):
        grp = QGroupBox('Compute Spectral Variants')
        layout = QFormLayout(grp)

        self._setup_btn = QPushButton('Compute Spectral Variants')
        self._setup_btn.clicked.connect(self._run_setup)
        layout.addRow(self._setup_btn)

        self._setup_status = QLabel('')
        self._setup_status.setWordWrap(True)
        layout.addRow(self._setup_status)

        parent_layout.addWidget(grp)

    def _run_setup(self):
        if not AUTOSPECTRAL_OPT_KERNEL_AVAILABLE:
            self._setup_status.setText('Compiled kernel not found — see warning above.')
            return
        if self._setup_thread is not None:
            self._setup_status.setText('Already computing — please wait.')
            return

        self._setup_btn.setEnabled(False)
        self._setup_status.setText('Computing spectral variants...')

        self._setup_thread = QThread()
        self._setup_worker = VariantSetupWorker(
            self.controller,
            n_cells=self._n_cells_spin.value(),
            som_dim=self._som_dim_spin.value(),
            k_neighbors=self._k_neighbors_spin.value(),
            sim_threshold=self._sim_threshold_spin.value(),
            dedup_threshold=self._dedup_threshold_spin.value(),
            max_variants=self._max_variants_spin.value(),
        )
        self._setup_worker.moveToThread(self._setup_thread)
        self._setup_thread.started.connect(self._setup_worker.run)
        self._setup_worker.finished.connect(self._on_setup_finished)
        self._setup_worker.error.connect(self._on_setup_error)
        self._setup_worker.progress.connect(self._on_setup_progress)
        self._setup_worker.finished.connect(self._setup_thread.quit)
        self._setup_worker.error.connect(self._setup_thread.quit)
        self._setup_thread.finished.connect(self._setup_thread.deleteLater)
        self._setup_thread.finished.connect(self._on_setup_thread_finished)
        self._setup_thread.start()

    def _on_setup_progress(self, n, total, label):
        self._setup_status.setText(f'Computing spectral variants... ({n}/{total}: {label})')
        if self.bus:
            self.bus.progress.emit(n, total)

    def _on_setup_finished(self, results: dict, raw_pos_thresholds, unmixed_pos_thresholds):
        self.controller.autospectral_variants = results
        # Cached so the comparison and exports reuse these unstained-derived
        # thresholds instead of recomputing them from whichever stained sample
        # is loaded; set before the save so they are persisted with this run.
        self.controller.autospectral_raw_pos_thresholds = raw_pos_thresholds
        self.controller.autospectral_unmixed_pos_thresholds = unmixed_pos_thresholds
        self.controller._save_autospectral_variants()

        reference_spectra = self.controller._build_fluor_spectra()
        fluor_names = [c['label'] for c in self.controller.experiment.process.get('spectral_model', [])
                       if c['label'] in self.controller.experiment.process.get('profiles', {})]
        delta_dict = {label: v['delta'] for label, v in results.items()}
        necessity = calculate_optimize_necessity(
            reference_spectra, fluor_names, delta_dict,
            threshold=self._necessity_threshold_spin.value(),
        )

        meta = {}
        for label in fluor_names:
            if label not in results:
                continue
            meta[label] = {
                'optimize_score': necessity['scores_norm'].get(label, 0.0),
                'optimize_recommended': necessity['optimize_recommended'].get(label, False),
                'active': necessity['optimize_recommended'].get(label, False),
            }
        self.controller.experiment.process['autospectral_variants_meta'] = meta
        if self.bus:
            self.bus.autoSaveRequested.emit()

        n_active = sum(1 for m in meta.values() if m['active'])
        self._setup_status.setText(
            f'Done. {len(results)}/{len(fluor_names)} fluorophore(s) have computed variants '
            f'(others fall back to the reference spectrum); {n_active} optimised per cell.'
        )
        self._refresh_plot_combo()
        self._refresh_all_guards()

    def _on_setup_error(self, msg: str):
        self._setup_status.setText(f'Error: {msg}')

    def _on_setup_thread_finished(self):
        self._setup_thread = None
        self._setup_worker = None
        self._setup_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Plot Spectral Variants
    # ------------------------------------------------------------------

    def _build_plot_section(self, parent_layout):
        grp = QGroupBox('Plot Spectral Variants')
        layout = QVBoxLayout(grp)

        row = QHBoxLayout()
        row.addWidget(QLabel('Fluorophore:'))
        self._plot_combo = _spin(QComboBox())
        self._plot_combo.setMinimumWidth(220)
        self._plot_combo.currentTextChanged.connect(self._draw_selected_variant)
        row.addWidget(self._plot_combo)
        row.addStretch()
        self._plot_status = QLabel('')
        row.addWidget(self._plot_status)
        layout.addLayout(row)

        self._variant_plot_axis = BottomAxisVerticalTickLabels()
        self._variant_plot = pg.PlotWidget(axisItems={'bottom': self._variant_plot_axis})
        self._variant_plot.setLabel('left', 'Intensity')
        self._variant_plot.showGrid(x=True, y=True, alpha=0.3)
        self._variant_plot.setMaximumHeight(440)
        self._variant_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._variant_plot.setMouseEnabled(x=False, y=False)
        self._wheel_blocker = WheelBlocker(self)
        self._variant_plot.viewport().installEventFilter(self._wheel_blocker)
        layout.addWidget(self._variant_plot)

        parent_layout.addWidget(grp)

    def _refresh_plot_combo(self):
        self._plot_combo.blockSignals(True)
        current = self._plot_combo.currentText()
        self._plot_combo.clear()
        variants = getattr(self.controller, 'autospectral_variants', {}) or {}
        for label in self._labels_in_major_channel_order(variants.keys()):
            self._plot_combo.addItem(label)
        idx = self._plot_combo.findText(current)
        if idx >= 0:
            self._plot_combo.setCurrentIndex(idx)
        self._plot_combo.blockSignals(False)
        self._draw_selected_variant()

    def _labels_in_major_channel_order(self, labels) -> list:
        """
        Fluorophore labels ordered by their Major Channel's position in the
        detector list (the Spectral Process tab's order); labels with no
        resolvable gate_channel sort last.
        """
        channel_order = {ch: i for i, ch in enumerate(fluorescence_channel_names(self.controller))}
        gate_channel_by_label = {
            c['label']: c.get('gate_channel')
            for c in self.controller.experiment.process.get('spectral_model', [])
        }
        return sorted(
            labels,
            key=lambda lbl: channel_order.get(gate_channel_by_label.get(lbl) or '', len(channel_order))
        )

    def _draw_selected_variant(self):
        self._variant_plot.clear()
        label = self._plot_combo.currentText()
        variants = getattr(self.controller, 'autospectral_variants', {}) or {}
        entry = variants.get(label)
        if entry is None:
            return

        v_mats = entry['v_mats']
        profiles = self.controller.experiment.process.get('profiles', {})
        if profiles.get(label) is None:
            return
        reference_vec = np.array(profiles[label])

        n_variants, n_ch = v_mats.shape
        x = np.arange(n_ch)

        channel_names = fluorescence_channel_names(self.controller)
        if channel_names and len(channel_names) == n_ch:
            display_names = [n.removesuffix('-A') for n in channel_names]
            ticks = [[(i, dn) for i, dn in enumerate(display_names)], []]
            self._variant_plot_axis.setTicks(ticks)
            db_col = self.controller.experiment.settings['raw'].get('cytometer_db_col')
            if db_col:
                detector_laser_map = get_detector_laser_map(db_col)
                tick_colors = {
                    dn: LASER_LABEL_COLORS[laser]
                    for dn, name in zip(display_names, channel_names)
                    if (laser := detector_laser_map.get(name)) in LASER_LABEL_COLORS
                }
            else:
                tick_colors = {}
            self._variant_plot_axis.tick_colors = tick_colors
        else:
            self._variant_plot_axis.setTicks(None)
            self._variant_plot_axis.tick_colors = {}

        low_alpha_pen = pg.mkPen(color=(200, 80, 80, 110), width=4)
        for row in v_mats:
            self._variant_plot.plot(x, np.maximum(row, 0), pen=low_alpha_pen)
        # Reference drawn last, on top, in blue (reads on light and dark backgrounds).
        self._variant_plot.plot(x, np.maximum(reference_vec, 0), pen=pg.mkPen(color=(30, 144, 255), width=3))

        self._plot_status.setText(f'{n_variants} variant(s), {entry.get("n_events_used", "?")} event(s) used.')

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def _build_compare_section(self, parent_layout):
        grp = QGroupBox('Comparison (Current Sample)')
        layout = QVBoxLayout(grp)

        row = QHBoxLayout()
        self._cmp_run_btn = QPushButton('Update Plots')
        self._cmp_run_btn.clicked.connect(self._run_comparison)
        row.addWidget(self._cmp_run_btn)
        self._cmp_status = QLabel('')
        row.addWidget(self._cmp_status)
        row.addStretch()
        layout.addLayout(row)

        splitter = QSplitter(Qt.Horizontal)
        self._plot_af = AfComparisonPlotWidget('Per-cell AF', self.controller, parent=self)
        self._plot_opt = AfComparisonPlotWidget('Per-cell AF + spectral variants', self.controller,
                                                parent=self)
        splitter.addWidget(self._plot_af)
        splitter.addWidget(self._plot_opt)
        splitter.setSizes([500, 500])
        layout.addWidget(splitter)

        for plot, other in ((self._plot_af, self._plot_opt), (self._plot_opt, self._plot_af)):
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

    def _refresh_comparison_controls(self):
        sample_path = getattr(self.controller, 'current_sample_path', None)
        has_af = self._sample_has_af_profile(sample_path)
        setup_done = self._setup_done()
        self._cmp_run_btn.setEnabled(has_af and setup_done and AUTOSPECTRAL_OPT_KERNEL_AVAILABLE)
        if not setup_done:
            self._cmp_status.setText('Compute spectral variants first.')
        elif not has_af:
            self._cmp_status.setText(
                'Current sample has no AF profile assigned — see the Autofluorescence page.')
        else:
            self._cmp_status.setText('')

    def _clear_comparison_plots(self):
        self._plot_af.clear()
        self._plot_opt.clear()
        self._af_data = None
        self._opt_data = None
        self._last_cmp_state = None
        self._pending_cmp_state = None

    def _active_labels(self) -> set:
        meta = self.controller.experiment.process.get('autospectral_variants_meta', {})
        return {label for label, m in meta.items() if m.get('active')}

    def _run_comparison(self):
        if self._cmp_running:
            self._cmp_status.setText('Comparison already running — please wait.')
            return
        c = self.controller
        if c.raw_event_data is None:
            self._cmp_status.setText('No sample loaded.')
            return
        if c.transfer_matrix is None:
            self._cmp_status.setText('No unmixing matrix — run spectral process first.')
            return

        af_spectra = c.get_combined_af_spectra_for_sample(c.current_sample_path)
        if af_spectra is None or af_spectra.shape[0] < 2:
            self._cmp_status.setText(
                'No AF profile (or fewer than 2 spectra) assigned to this sample — '
                'see the Autofluorescence page.'
            )
            return

        reference_spectra = c._build_fluor_spectra()
        if reference_spectra is None:
            self._cmp_status.setText('Fluorophore profiles unavailable.')
            return
        fluor_names = [ctl['label'] for ctl in c.experiment.process.get('spectral_model', [])
                       if ctl['label'] in c.experiment.process.get('profiles', {})]

        active_labels = self._active_labels()
        variants_meta = getattr(c, 'autospectral_variants', {}) or {}
        envelope = af_envelope_for_profiles(
            c, c.experiment.samples.get('sample_af_profiles', {}).get(c.current_sample_path, []))

        raw_settings = c.experiment.settings['raw']
        pnn_raw_full = raw_settings['event_channels_pnn']
        pnn_raw = raw_settings.get('whitelisted_pnn') or pnn_raw_full
        fl_ids_raw = [pnn_raw.index(pnn_raw_full[i]) for i in c.filtered_raw_fluorescence_channel_ids]
        fl_ids_unmixed = np.array(c.experiment.settings['unmixed']['fluorescence_channel_ids'])

        cached_thresholds = getattr(c, 'autospectral_unmixed_pos_thresholds', None)
        if cached_thresholds is not None and len(cached_thresholds) == reference_spectra.shape[0]:
            unmixed_pos_thresholds = cached_thresholds
        else:
            logger.warning(
                'Spectral Variants comparison: no cached unstained-derived thresholds '
                '(compute spectral variants first) — using all-zero thresholds rather than '
                'recomputing from the loaded (stained) sample.'
            )
            unmixed_pos_thresholds = np.zeros(reference_spectra.shape[0])

        state_key = (
            c.current_sample_path,
            frozenset(active_labels),
            bytes(af_spectra.data),
            None if envelope is None else (np.asarray(envelope['null_var']).tobytes(),
                                           np.asarray(envelope['mean_rows']).tobytes(),
                                           None if envelope['spread'] is None else envelope['spread'].tobytes()),
            self._comparison_max_events_spin.value(),
            tuple(self._kernel_kwargs().items()),
        )
        if state_key == self._last_cmp_state and self._af_data is not None:
            self._cmp_status.setText(
                f'No change — redrawing from cached result ({self._af_data.shape[0]:,} events).'
            )
            self._initialise_comparison_plots()
            return

        self._plot_af.set_status('Computing...')
        self._plot_opt.set_status('Computing...')
        self._cmp_status.setText('Running unmixing on current sample...')

        self._pending_cmp_state = state_key
        self._cmp_running = True
        self._cmp_thread = QThread()
        self._cmp_worker = JointComparisonWorker(
            c.raw_event_data, c.transfer_matrix, fl_ids_raw, fl_ids_unmixed,
            reference_spectra, fluor_names, af_spectra, variants_meta, active_labels,
            unmixed_pos_thresholds, c.experiment.process.get('spillover'),
            self._comparison_max_events_spin.value(), self._kernel_kwargs(),
            unmixed_pnn=c.experiment.settings['unmixed']['event_channels_pnn'],
            af_index_map=c.get_af_index_map_for_sample(c.current_sample_path),
            envelope=envelope,
        )
        self._cmp_worker.moveToThread(self._cmp_thread)
        self._cmp_thread.started.connect(self._cmp_worker.run)
        self._cmp_worker.finished.connect(self._on_comparison_finished)
        self._cmp_worker.error.connect(self._on_comparison_error)
        self._cmp_worker.finished.connect(self._cmp_thread.quit)
        self._cmp_worker.error.connect(self._cmp_thread.quit)
        self._cmp_thread.finished.connect(self._cmp_thread.deleteLater)
        self._cmp_thread.finished.connect(self._on_cmp_thread_finished)
        self._cmp_thread.start()

    def _on_comparison_finished(self, _ols_data, af_data, opt_data):
        self._af_data = af_data
        self._opt_data = opt_data
        self._last_cmp_state = self._pending_cmp_state
        self._cmp_status.setText(f'Done. {af_data.shape[0]:,} events.')
        self._plot_af.set_status('')
        self._plot_opt.set_status('')
        self._initialise_comparison_plots()

    def _on_comparison_error(self, msg: str):
        self._plot_af.set_status('Error')
        self._plot_opt.set_status('Error')
        self._cmp_status.setText(f'Error: {msg}')
        logger.error(f'Spectral Variants comparison error: {msg}')

    def _on_cmp_thread_finished(self):
        self._cmp_running = False
        self._cmp_thread = None
        self._cmp_worker = None

    def _initialise_comparison_plots(self):
        """Feed both plots; default channels are the pair that changes most.
        Existing channel, scaling and gate choices are kept."""
        if self._af_data is None or self._opt_data is None:
            return
        c = self.controller
        pnn = c.experiment.settings['unmixed'].get('event_channels_pnn', [])
        fl_ids = c.experiment.settings['unmixed'].get('fluorescence_channel_ids', [])
        if not pnn or not fl_ids or len(fl_ids) < 2:
            return
        fl_names = [pnn[i] for i in fl_ids]
        ch_x, ch_y = fl_names[0], fl_names[1]
        try:
            picked = pick_most_affected_channels(self._af_data, self._opt_data, pnn, fl_names)
            if picked is not None:
                ch_x, ch_y = picked
        except Exception as e:
            logger.debug(f'Spectral Variants comparison: channel auto-pick failed: {e}')
        if self._plot_af._channel_x in fl_names and self._plot_af._channel_y in fl_names:
            ch_x, ch_y = self._plot_af._channel_x, self._plot_af._channel_y

        for plot, data in ((self._plot_af, self._af_data), (self._plot_opt, self._opt_data)):
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
        self._refresh_all_guards()
        self._refresh_plot_combo()
        if self._cmp_run_btn.isEnabled():
            QTimer.singleShot(0, self._run_comparison)

    def _on_mode_change(self, mode):
        if mode == TAB_NAME and self.is_active():
            self.on_activated()

    def _on_sample_loaded(self, _path):
        QTimer.singleShot(0, self._on_sample_loaded_deferred)

    def _on_sample_loaded_deferred(self):
        self._refresh_comparison_controls()
        if not self.is_active():
            self._clear_comparison_plots()
            return
        if self._cmp_run_btn.isEnabled():
            self._run_comparison()

    def _on_process_refreshed(self):
        self._refresh_all_guards()
