"""
autospectral_optimization_tab.py
-----------------------------------
Honeychrome bundled plugin: AutoSpectral Optimization (per-cell fluorophore +
AF joint unmixing).

Location: bundled_plugins/autospectral_optimization_tab.py (see
plugin_loaders.py — discovered automatically as *_tab.py, gated behind
'EnableBundledPlugin_autospectral_optimization_tab' in QSettings.

Sections
-------------------------------------------------
1. Setup    — discover per-fluorophore spectral variants
2. Plot     — per-fluorophore variant density plot
3. Table    — optimisation-necessity scores + Active override
4. Compare  — standard vs. AutoSpectral Optimization biplot
5. Unmix    — batch FCS export using the joint pipeline
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
    QLabel, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from honeychrome.controller_components.functions import apply_transfer_matrix
from honeychrome.controller_components.autospectral_functions import (
    precompute_af_matrices, precompute_joint_cov_extras, apply_af_unmixing,
)
from honeychrome.controller_components.cytometer_whitelist import (
    get_detector_laser_map, LASER_LABEL_COLORS,
)
from honeychrome.view_components.autospectral_tab import (
    AfComparisonPlotWidget,
    pick_most_affected_channels,
)
from honeychrome.view_components.busy_cursor import with_busy_cursor
from honeychrome.view_components.copyable_table_widget import CopyableTableWidget
from honeychrome.view_components.help_toggle_widget import WheelBlocker
from honeychrome.view_components.help_texts import autospectral_optimization_help_text
from honeychrome.view_components.ordered_multi_sample_picker import OrderedMultiSamplePicker
from honeychrome.view_components.profiles_viewer import BottomAxisVerticalTickLabels
import honeychrome.settings as settings

logger = logging.getLogger(__name__)

plugin_name = 'AutoSpectral Optimization'
TAB_NAME = plugin_name

# ---------------------------------------------------------------------------
# Sibling-module imports — this plugin is self-contained inside
# bundled_plugins/, so its own directory must be on sys.path before
# importing autospectral_optimization_functions / autospectral_opt_kernel_wrapper,
# regardless of how plugin_loaders.py loaded *this* file.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from autospectral_optimization_functions import (   # noqa: E402
    discover_all_variants,
    calculate_optimize_necessity,
    unmix_autospectral_optimization,
    fluorescence_channel_names,
    active_variant_labels,
    cached_unmixed_thresholds,
    has_active_variants,
    make_optimization_unmix_fn,
)
from autospectral_opt_kernel_wrapper import AUTOSPECTRAL_OPT_KERNEL_AVAILABLE  # noqa: E402
from joint_unmix_export import (   # noqa: E402
    JointUnmixExporter,
    JointUnmixRun,
    build_export_layout,
)


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class VariantSetupWorker(QObject):
    """Runs discover_all_variants() in a QThread."""
    finished = Signal(dict, object, object)   # (variants, raw_pos_thresholds, unmixed_pos_thresholds)
    error = Signal(str)
    progress = Signal(int, int, str)

    def __init__(self, controller, n_cells, som_dim, k_neighbors, sim_threshold):
        super().__init__()
        self.controller = controller
        self.n_cells = n_cells
        self.som_dim = som_dim
        self.k_neighbors = k_neighbors
        self.sim_threshold = sim_threshold

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
                progress_callback=_progress,
            )
            self.finished.emit(
                output['variants'], output['raw_pos_thresholds'], output['unmixed_pos_thresholds'],
            )
        except Exception as e:
            logger.exception('AutoSpectral Optimization: Setup failed')
            self.error.emit(str(e))


class JointComparisonWorker(QObject):
    """Computes OLS, AF-corrected, and AutoSpectral-Optimization arrays in a QThread."""
    finished = Signal(object, object, object)   # (ols_data, af_data, opt_data) full unmixed-shape arrays
    error = Signal(str)

    def __init__(self, raw_event_data, transfer_matrix, filtered_fl_ids_raw,
                 fl_ids_unmixed, reference_spectra, fluor_names, af_spectra,
                 variants_meta, active_labels, unmixed_pos_thresholds,
                 spillover, max_events, kernel_kwargs,
                 unmixed_pnn=None, af_index_map=None):
        super().__init__()
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
            af_precomputed = precompute_af_matrices(self.reference_spectra, self.af_spectra)
            af_precomputed.update(precompute_joint_cov_extras(af_precomputed, self.af_spectra))
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


class AutoSpectralOptExporter(JointUnmixExporter):
    """
    Batch FCS export with the joint AF + fluorophore-variant kernel for the
    samples picked in the Unmix section.

    Construct on the main thread: the export layout, the computed variants,
    the Active set and the cached positivity thresholds are snapshotted
    here. ``probe_fn`` is set when per-cell variant optimisation is on (at
    least one Active fluorophore has variants), so JointUnmixRun can time the
    run before it starts; AF-only unmixing is fast and is not timed.
    """

    def __init__(self, sample_paths, controller, bus, kernel_kwargs):
        layout = build_export_layout(controller)
        variants_meta = dict(getattr(controller, 'autospectral_variants', None) or {})
        active_labels = active_variant_labels(controller.experiment.process)
        unmix_fn = make_optimization_unmix_fn(
            layout.reference_spectra,
            list(layout.fluor_names),
            variants_meta,
            active_labels,
            cached_unmixed_thresholds(controller, layout.reference_spectra),
            kernel_kwargs,
        )
        super().__init__(sample_paths, layout, unmix_fn, bus=bus,
                         unmixing_method='AutoSpectral Optimization')
        if has_active_variants(variants_meta, active_labels):
            self.probe_fn = lambda raw_fl, af: unmix_fn(raw_fl, None, af)
        else:
            self.probe_fn = None


# ---------------------------------------------------------------------------
# PluginWidget
# ---------------------------------------------------------------------------

class PluginWidget(QWidget):
    def __init__(self, bus=None, controller=None, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.controller = controller
        self._qsettings = QSettings('honeychrome', f'plugin_{plugin_name}')

        # Setup worker/thread
        self._setup_thread = None
        self._setup_worker = None

        # Comparison state
        self._cmp_thread = None
        self._cmp_worker = None
        self._cmp_running = False
        self._last_cmp_state = None
        self._pending_cmp_state = None
        self._ols_data = None
        self._af_data = None
        self._opt_data = None

        # Unmix state
        self._unmix_run = None

        self._build_ui()

        if self.bus:
            self.bus.modeChangeRequested.connect(self._on_mode_change)
            self.bus.loadSampleRequested.connect(self._on_sample_loaded)
            self.bus.spectralProcessRefreshed.connect(self._on_process_refreshed)
            self.bus.sampleTreeUpdated.connect(self._on_sample_tree_updated)

        if self._setup_done():
            self._refresh_plot_combo()
            self._rebuild_table()

        self._refresh_all_guards()

    # ------------------------------------------------------------------
    # Top-level layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._label_disabled = QLabel(
            'AutoSpectral Optimization: calculate the unmixing matrix (Spectral '
            'Process tab) before using this plugin.'
        )
        self._label_disabled.setWordWrap(True)
        self._label_disabled.setContentsMargins(8, 8, 8, 8)
        outer.addWidget(self._label_disabled)

        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setAlignment(Qt.AlignTop)
        content_layout.setSpacing(10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self._content)
        outer.addWidget(scroll)

        from honeychrome.settings import heading_style
        title_label = QLabel(plugin_name)
        title_label.setStyleSheet(heading_style)
        content_layout.addWidget(title_label)

        self.help_label = QLabel(autospectral_optimization_help_text)
        self.help_label.setTextFormat(Qt.RichText)
        self.help_label.setWordWrap(True)
        font = self.help_label.font()
        font.setPointSize(14)
        self.help_label.setFont(font)
        content_layout.addWidget(self.help_label)

        if not AUTOSPECTRAL_OPT_KERNEL_AVAILABLE:
            kernel_warning = QLabel(
                '<b>Compiled kernel not found.</b> Run '
                '<tt>build_autospectral_opt_kernel.py</tt> (in bundled_plugins/) '
                'before using Setup, Compare, or Unmix.'
            )
            kernel_warning.setTextFormat(Qt.RichText)
            kernel_warning.setStyleSheet('color: #cc6600;')
            content_layout.addWidget(kernel_warning)

        self.toggle_content_button = QCheckBox(f'Show {plugin_name} process')
        saved_state = self._qsettings.value('show_content', 'false') == 'true'
        self.toggle_content_button.setCheckable(True)
        self.toggle_content_button.setChecked(saved_state)
        content_layout.addWidget(self.toggle_content_button)

        self.toggle_content = QWidget()
        toggle_layout = QVBoxLayout(self.toggle_content)
        self._build_advanced_settings(toggle_layout)
        self._build_setup_section(toggle_layout)
        self._build_plot_section(toggle_layout)
        self._build_table_section(toggle_layout)
        self._build_compare_section(toggle_layout)
        self._build_unmix_section(toggle_layout)
        content_layout.addWidget(self.toggle_content)
        self.toggle_content.setVisible(saved_state)

        self.toggle_content_button.toggled.connect(self.toggle_content.setVisible)
        self.toggle_content_button.toggled.connect(self._save_visibility)
        self.toggle_content_button.setStyleSheet(
            'QCheckBox { font-size: 14pt; spacing: 10px; padding: 10px; }'
        )

        content_layout.addStretch()

    def _save_visibility(self, checked: bool):
        self._qsettings.setValue('show_content', 'true' if checked else 'false')

    # ------------------------------------------------------------------
    # Advanced settings
    # ------------------------------------------------------------------

    def _build_advanced_settings(self, parent_layout):
        self._adv_toggle = QCheckBox('Show Advanced Settings')
        saved = self._qsettings.value('show_advanced', 'false') == 'true'
        self._adv_toggle.setChecked(saved)
        parent_layout.addWidget(self._adv_toggle)

        adv_content = QWidget()
        adv_layout = QHBoxLayout(adv_content)
        adv_content.setVisible(saved)
        self._adv_toggle.toggled.connect(adv_content.setVisible)
        self._adv_toggle.toggled.connect(
            lambda c: self._qsettings.setValue('show_advanced', 'true' if c else 'false')
        )

        # --- Setup group ---
        setup_grp = QGroupBox('Setup')
        setup_form = QFormLayout(setup_grp)
        self._n_cells_spin = QSpinBox(); self._n_cells_spin.setRange(100, 1_000_000); self._n_cells_spin.setValue(10_000)
        self._som_dim_spin = QSpinBox(); self._som_dim_spin.setRange(2, 20); self._som_dim_spin.setValue(5)
        self._k_neighbors_spin = QSpinBox(); self._k_neighbors_spin.setRange(1, 50); self._k_neighbors_spin.setValue(3)
        self._sim_threshold_spin = QDoubleSpinBox(); self._sim_threshold_spin.setRange(0.0, 1.0)
        self._sim_threshold_spin.setDecimals(3); self._sim_threshold_spin.setSingleStep(0.005); self._sim_threshold_spin.setValue(0.985)
        for w in (self._n_cells_spin, self._som_dim_spin, self._k_neighbors_spin, self._sim_threshold_spin):
            w.installEventFilter(WheelBlocker(w)); w.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        setup_form.addRow('n.cells:', self._n_cells_spin)
        setup_form.addRow('som.dim:', self._som_dim_spin)
        setup_form.addRow('k.neighbors:', self._k_neighbors_spin)
        setup_form.addRow('sim.threshold:', self._sim_threshold_spin)
        adv_layout.addWidget(setup_grp)

        # --- Necessity group ---
        necessity_grp = QGroupBox('Necessity')
        necessity_form = QFormLayout(necessity_grp)
        self._necessity_threshold_spin = QDoubleSpinBox()
        self._necessity_threshold_spin.setRange(0.0, 1.0); self._necessity_threshold_spin.setDecimals(3)
        self._necessity_threshold_spin.setSingleStep(0.005); self._necessity_threshold_spin.setValue(0.01)
        self._necessity_threshold_spin.installEventFilter(WheelBlocker(self._necessity_threshold_spin))
        self._necessity_threshold_spin.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
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
            w.installEventFilter(WheelBlocker(w)); w.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
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
        self._comparison_max_events_spin = QSpinBox()
        self._comparison_max_events_spin.setRange(1_000, 2_000_000)
        self._comparison_max_events_spin.setSingleStep(5_000)
        self._comparison_max_events_spin.setValue(30_000)
        self._comparison_max_events_spin.installEventFilter(WheelBlocker(self._comparison_max_events_spin))
        self._comparison_max_events_spin.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
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
        if not available:
            return

        setup_done = self._setup_done()
        for w in (self._plot_combo, self._table):
            w.setEnabled(setup_done)
        self._plot_status.setText('' if setup_done else 'Run Setup first.')
        self._table_status.setText('' if setup_done else 'Run Setup first.')

        self._refresh_comparison_controls()
        self._refresh_unmix_picker()

    # ------------------------------------------------------------------
    # Setup section
    # ------------------------------------------------------------------

    def _build_setup_section(self, parent_layout):
        grp = QGroupBox('1 — Setup: Compute Spectral Variants')
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
            self._setup_status.setText('Setup already running — please wait.')
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
        # Cached so Compare/Unmix/Export reuse these unstained-derived
        # thresholds instead of recomputing from whatever stained sample
        # happens to be loaded/exported. Must be set *before* the save call
        # below so they're actually persisted with this Setup run's cache.
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

        self._setup_status.setText(
            f'Done. {len(results)}/{len(fluor_names)} fluorophore(s) have '
            f'computed variants (others fall back to the reference spectrum).'
        )
        self._refresh_plot_combo()
        self._rebuild_table()
        self._refresh_all_guards()

    def _on_setup_error(self, msg: str):
        self._setup_status.setText(f'Error: {msg}')

    def _on_setup_thread_finished(self):
        self._setup_thread = None
        self._setup_worker = None
        self._setup_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Plot section
    # ------------------------------------------------------------------

    def _build_plot_section(self, parent_layout):
        grp = QGroupBox('2 — Plot: Spectral Variants')
        layout = QVBoxLayout(grp)

        row = QHBoxLayout()
        row.addWidget(QLabel('Fluorophore:'))
        self._plot_combo = QComboBox()
        self._plot_combo.setMinimumWidth(220)
        self._plot_combo.installEventFilter(WheelBlocker(self._plot_combo))
        self._plot_combo.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
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
        self._variant_plot.viewport().installEventFilter(WheelBlocker(self))
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
        Orders fluorophore labels by their Major Channel's position in the
        detector list — same convention as the Spectral Process tab
        (spectral_model_editor.py::_sort_model_by_gate_channel), rather than
        alphabetically. Labels with no resolvable gate_channel sort last.
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
        reference_vec = np.array(profiles.get(label))
        if reference_vec is None:
            return

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

        low_alpha_pen = pg.mkPen(color=(200, 80, 80, 110), width=2)
        for row in v_mats:
            self._variant_plot.plot(x, np.maximum(row, 0), pen=low_alpha_pen)
        # Reference spectrum drawn last, solid, on top — blue so it reads
        # against both light and dark backgrounds.
        self._variant_plot.plot(x, np.maximum(reference_vec, 0), pen=pg.mkPen(color=(30, 144, 255), width=3))

        self._plot_status.setText(f'{n_variants} variant(s), {entry.get("n_events_used", "?")} event(s) used.')

    # ------------------------------------------------------------------
    # Table section
    # ------------------------------------------------------------------

    def _build_table_section(self, parent_layout):
        grp = QGroupBox('3 — Table: Optimisation Necessity')
        layout = QVBoxLayout(grp)

        self._table_status = QLabel('')
        layout.addWidget(self._table_status)

        self._table = CopyableTableWidget([], ['Fluorophore', 'Optimize Score', 'Recommended', 'Active'])
        layout.addWidget(self._table)

        parent_layout.addWidget(grp)

    def _rebuild_table(self):
        fluor_names = [c['label'] for c in self.controller.experiment.process.get('spectral_model', [])
                       if c['label'] in self.controller.experiment.process.get('profiles', {})]
        meta = self.controller.experiment.process.get('autospectral_variants_meta', {})

        scored_labels = sorted(
            (label for label in fluor_names if label in meta),
            key=lambda lbl: meta[lbl]['optimize_score'],
            reverse=True,
        )
        rows = [
            {
                'Fluorophore': label,
                'Optimize Score': f'{meta[label]["optimize_score"]:.4f}',
                'Recommended': 'Yes' if meta[label]['optimize_recommended'] else 'No',
            }
            for label in scored_labels
        ]

        old = self._table
        self._table = CopyableTableWidget(rows, ['Fluorophore', 'Optimize Score', 'Recommended', 'Active'])
        # Replace the widget in its parent layout.
        layout = self.toggle_content.layout()
        # Find and swap within the Table group box's own layout instead:
        for i in range(layout.count()):
            item = layout.itemAt(i)
            w = item.widget() if item else None
            if isinstance(w, QGroupBox) and w.title().startswith('3 —'):
                grp_layout = w.layout()
                grp_layout.replaceWidget(old, self._table) if old else grp_layout.addWidget(self._table)
                if old:
                    old.setParent(None)
                break

        for row_idx, label in enumerate([r['Fluorophore'] for r in rows]):
            checkbox = QCheckBox()
            checkbox.setChecked(bool(meta[label].get('active')))
            checkbox.stateChanged.connect(
                lambda state, lbl=label: self._on_active_checkbox_changed(lbl, state)
            )
            self._table.setCellWidget(row_idx, 3, checkbox)

        self._table_status.setText(f'{len(rows)} fluorophore(s) with computed variants.')

    def _on_active_checkbox_changed(self, label: str, state: int):
        meta = self.controller.experiment.process.get('autospectral_variants_meta', {})
        if label in meta:
            meta[label]['active'] = bool(state)
            if self.bus:
                self.bus.autoSaveRequested.emit()

    def _active_labels(self) -> set:
        meta = self.controller.experiment.process.get('autospectral_variants_meta', {})
        return {label for label, m in meta.items() if m.get('active')}

    # ------------------------------------------------------------------
    # Compare section
    # ------------------------------------------------------------------

    def _build_compare_section(self, parent_layout):
        grp = QGroupBox('4 — Compare: Standard vs. AutoSpectral Optimization')
        layout = QVBoxLayout(grp)

        row = QHBoxLayout()
        self._cmp_run_btn = QPushButton('Update Plots')
        self._cmp_run_btn.clicked.connect(self._run_comparison)
        row.addWidget(self._cmp_run_btn)
        self._cmp_status = QLabel('')
        row.addWidget(self._cmp_status)
        row.addStretch()
        layout.addLayout(row)

        plots_row = QHBoxLayout()
        self._plot_ols = AfComparisonPlotWidget('Standard OLS (No AF Correction)', self.controller,
                                                has_af_channels=False)
        self._plot_af = AfComparisonPlotWidget('AutoSpectral AF-Corrected', self.controller)
        self._plot_opt = AfComparisonPlotWidget('AutoSpectral Optimization', self.controller)
        plots_row.addWidget(self._plot_ols)
        plots_row.addWidget(self._plot_af)
        plots_row.addWidget(self._plot_opt)
        layout.addLayout(plots_row)

        self._plot_ols.sourceGateChanged.connect(self._on_ols_gate_changed)
        self._plot_af.sourceGateChanged.connect(self._on_af_gate_changed)
        self._plot_opt.sourceGateChanged.connect(self._on_opt_gate_changed)
        self._plot_ols.channelChanged.connect(self._on_ols_channel_changed)
        self._plot_af.channelChanged.connect(self._on_af_channel_changed)
        self._plot_opt.channelChanged.connect(self._on_opt_channel_changed)
        self._plot_ols.scalingChanged.connect(self._on_ols_scaling_changed)
        self._plot_af.scalingChanged.connect(self._on_af_scaling_changed)
        self._plot_opt.scalingChanged.connect(self._on_opt_scaling_changed)

        parent_layout.addWidget(grp)

    def _refresh_comparison_controls(self):
        sample_path = getattr(self.controller, 'current_sample_path', None)
        has_af = self._sample_has_af_profile(sample_path)
        setup_done = self._setup_done()
        self._cmp_run_btn.setEnabled(has_af and setup_done and AUTOSPECTRAL_OPT_KERNEL_AVAILABLE)
        if not setup_done:
            self._cmp_status.setText('Run Setup first.')
        elif not has_af:
            self._cmp_status.setText('Current sample has no AF profile assigned — see AutoSpectral AF tab.')
        else:
            self._cmp_status.setText('')

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
                'see the AutoSpectral AF tab.'
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
                'AutoSpectral Optimization comparison: no cached unstained-derived '
                'thresholds (run Setup first) — falling back to all-zero thresholds '
                'rather than recomputing from the currently loaded (stained) sample.'
            )
            unmixed_pos_thresholds = np.zeros(reference_spectra.shape[0])

        state_key = (
            c.current_sample_path,
            frozenset(active_labels),
            bytes(af_spectra.data),
            self._comparison_max_events_spin.value(),
            tuple(self._kernel_kwargs().items()),
        )
        if state_key == self._last_cmp_state and self._ols_data is not None:
            self._cmp_status.setText(
                f'No change — redrawing from cached result ({self._ols_data.shape[0]:,} events).'
            )
            self._initialise_comparison_plots()
            return

        self._plot_ols.set_status('Computing...')
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

    def _on_comparison_finished(self, ols_data, af_data, opt_data):
        self._ols_data = ols_data
        self._af_data = af_data
        self._opt_data = opt_data
        self._last_cmp_state = self._pending_cmp_state
        self._cmp_status.setText(f'Done. {ols_data.shape[0]:,} events.')
        self._plot_ols.set_status('')
        self._plot_af.set_status('')
        self._plot_opt.set_status('')
        self._initialise_comparison_plots()

    def _on_comparison_error(self, msg: str):
        self._plot_ols.set_status('Error')
        self._plot_af.set_status('Error')
        self._plot_opt.set_status('Error')
        self._cmp_status.setText(f'Error: {msg}')
        logger.error(f'AutoSpectral Optimization comparison error: {msg}')

    def _on_cmp_thread_finished(self):
        self._cmp_running = False
        self._cmp_thread = None
        self._cmp_worker = None

    def _initialise_comparison_plots(self):
        if self._ols_data is None or self._af_data is None or self._opt_data is None:
            return
        c = self.controller
        pnn = c.experiment.settings['unmixed'].get('event_channels_pnn', [])
        fl_ids = c.experiment.settings['unmixed'].get('fluorescence_channel_ids', [])
        if not pnn or not fl_ids or len(fl_ids) < 2:
            return
        fl_names = [pnn[i] for i in fl_ids]
        ch_x, ch_y = fl_names[0], fl_names[1]

        # Better default: the two channels that differ most between
        # AF-corrected and Optimization results.
        try:
            picked = pick_most_affected_channels(
                self._af_data, self._opt_data, pnn, fl_names
            )
            if picked is not None:
                ch_x, ch_y = picked
        except Exception as e:
            logger.debug(
                f'AutoSpectral Optimization comparison: channel auto-pick failed: {e}'
            )

        if (self._plot_ols._channel_x in fl_names and self._plot_ols._channel_y in fl_names):
            ch_x, ch_y = self._plot_ols._channel_x, self._plot_ols._channel_y

        self._plot_ols.set_event_data(self._ols_data)
        self._plot_ols.initialise_from_controller(ch_x, ch_y)
        self._plot_ols.redraw()

        self._plot_af.set_event_data(self._af_data)
        self._plot_af.initialise_from_controller(ch_x, ch_y)
        self._plot_af.redraw()

        self._plot_opt.set_event_data(self._opt_data)
        self._plot_opt.initialise_from_controller(ch_x, ch_y)
        self._plot_opt.redraw()

    def _on_ols_gate_changed(self, gate_name: str):
        self._plot_af.set_source_gate(gate_name)
        self._plot_opt.set_source_gate(gate_name)

    def _on_af_gate_changed(self, gate_name: str):
        self._plot_ols.set_source_gate(gate_name)
        self._plot_opt.set_source_gate(gate_name)

    def _on_opt_gate_changed(self, gate_name: str):
        self._plot_ols.set_source_gate(gate_name)
        self._plot_af.set_source_gate(gate_name)

    def _on_ols_channel_changed(self, ch_x: str, ch_y: str):
        self._plot_af.blockSignals(True)
        self._plot_af.set_channels(ch_x, ch_y)
        self._plot_af.blockSignals(False)
        self._plot_opt.blockSignals(True)
        self._plot_opt.set_channels(ch_x, ch_y)
        self._plot_opt.blockSignals(False)

    def _on_af_channel_changed(self, ch_x: str, ch_y: str):
        self._plot_ols.blockSignals(True)
        self._plot_ols.set_channels(ch_x, ch_y)
        self._plot_ols.blockSignals(False)
        self._plot_opt.blockSignals(True)
        self._plot_opt.set_channels(ch_x, ch_y)
        self._plot_opt.blockSignals(False)

    def _on_opt_channel_changed(self, ch_x: str, ch_y: str):
        self._plot_ols.blockSignals(True)
        self._plot_ols.set_channels(ch_x, ch_y)
        self._plot_ols.blockSignals(False)
        self._plot_af.blockSignals(True)
        self._plot_af.set_channels(ch_x, ch_y)
        self._plot_af.blockSignals(False)

    def _on_ols_scaling_changed(self, axis_name: str, tr):
        self._plot_af.blockSignals(True)
        self._plot_af.set_scaling(axis_name, tr)
        self._plot_af.blockSignals(False)
        self._plot_opt.blockSignals(True)
        self._plot_opt.set_scaling(axis_name, tr)
        self._plot_opt.blockSignals(False)

    def _on_af_scaling_changed(self, axis_name: str, tr):
        self._plot_ols.blockSignals(True)
        self._plot_ols.set_scaling(axis_name, tr)
        self._plot_ols.blockSignals(False)
        self._plot_opt.blockSignals(True)
        self._plot_opt.set_scaling(axis_name, tr)
        self._plot_opt.blockSignals(False)

    def _on_opt_scaling_changed(self, axis_name: str, tr):
        self._plot_ols.blockSignals(True)
        self._plot_ols.set_scaling(axis_name, tr)
        self._plot_ols.blockSignals(False)
        self._plot_af.blockSignals(True)
        self._plot_af.set_scaling(axis_name, tr)
        self._plot_af.blockSignals(False)

    # ------------------------------------------------------------------
    # Unmix section
    # ------------------------------------------------------------------

    def _build_unmix_section(self, parent_layout):
        grp = QGroupBox('5 — Unmix: Batch Export')
        layout = QVBoxLayout(grp)

        self._unmix_picker = OrderedMultiSamplePicker(title='Samples with an AF profile assigned')
        layout.addWidget(self._unmix_picker)

        row = QHBoxLayout()
        self._unmix_btn = QPushButton('Export Selected Samples')
        self._unmix_btn.clicked.connect(self._run_unmix_export)
        row.addWidget(self._unmix_btn)
        self._unmix_status = QLabel('')
        row.addWidget(self._unmix_status)
        row.addStretch()
        layout.addLayout(row)

        parent_layout.addWidget(grp)

    def _refresh_unmix_picker(self):
        if not self.controller or not self.controller.experiment:
            return
        all_samples = self.controller.experiment.samples.get('all_samples', {})
        eligible = {path: name for path, name in all_samples.items() if self._sample_has_af_profile(path)}
        self._path_by_display_name = {v: k for k, v in eligible.items()}
        self._unmix_picker.set_items(sorted(eligible.values()))
        if not eligible:
            self._unmix_status.setText('No samples with an AF profile assigned — see the AutoSpectral AF tab.')
        else:
            self._unmix_status.setText('')
        self._unmix_btn.setEnabled(bool(eligible) and self._setup_done() and AUTOSPECTRAL_OPT_KERNEL_AVAILABLE)

    def _run_unmix_export(self):
        if self._unmix_run is not None:
            self._unmix_status.setText('Export already running — please wait.')
            return
        display_names = self._unmix_picker.get_ordered_list()
        sample_paths = [self._path_by_display_name[n] for n in display_names if n in self._path_by_display_name]
        if not sample_paths:
            self._unmix_status.setText('No samples selected.')
            return

        try:
            exporter = AutoSpectralOptExporter(
                sample_paths, self.controller, self.bus, self._kernel_kwargs()
            )
        except Exception as e:
            logger.exception('AutoSpectral Optimization: could not prepare the export')
            self._unmix_status.setText(f'Error: {e}')
            return

        self._unmix_btn.setEnabled(False)
        self._unmix_status.setText(f'Exporting {len(sample_paths)} sample(s)...')

        self._unmix_run = JointUnmixRun(
            exporter, parent_widget=self, probe_fn=exporter.probe_fn, bus=self.bus,
        )
        self._unmix_run.status.connect(self._unmix_status.setText)
        self._unmix_run.progress.connect(self._on_unmix_progress)
        self._unmix_run.finished.connect(self._on_unmix_finished)
        self._unmix_run.error.connect(self._on_unmix_error)
        self._unmix_run.done.connect(self._on_unmix_run_done)
        self._unmix_run.start()

    def _on_unmix_progress(self, n, total):
        self._unmix_status.setText(f'Exporting... ({n}/{total})')

    def _on_unmix_finished(self):
        self._unmix_status.setText('Export complete.')

    def _on_unmix_error(self, msg: str):
        self._unmix_status.setText(f'Error: {msg}')
        logger.error(f'AutoSpectral Optimization export error: {msg}')

    def _on_unmix_run_done(self):
        self._unmix_run = None
        self._unmix_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Bus slots
    # ------------------------------------------------------------------

    def _on_mode_change(self, _mode):
        self._refresh_all_guards()

    def _on_sample_loaded(self, _path):
        QTimer.singleShot(0, self._on_sample_loaded_deferred)

    def _on_sample_loaded_deferred(self):
        self._refresh_comparison_controls()
        if self._cmp_run_btn.isEnabled():
            self._run_comparison()

    def _on_process_refreshed(self):
        self._refresh_all_guards()

    def _on_sample_tree_updated(self):
        self._refresh_unmix_picker()
