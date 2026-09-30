"""
automated_gating_tab.py — Automated Gating plugin for Honeychrome
=================================================================
Applies a gating model (``.agmodel``, made with the Gating Model Builder
plugin) to experiment samples and compares the gated populations between
sample groups:

  • Setup   — load a model, align its channels to this experiment antigen by
              antigen, convert its boundaries to this experiment's axis
              transforms, and choose for each gate whether its stored
              boundary is applied as it is (fixed) or recalculated on each
              sample (recalculate, with a drift check against the stored
              boundary).
  • Run     — pick samples and gate them in a background worker. Each sample
              is unmixed with its own autofluorescence profile assignment
              and restricted to its time-QC keep-mask when present. Only
              counts, parent counts, per-population marker medians and a
              small display subsample are kept per sample.
  • Results — sample groups and covariates, moderated tests of population
              frequencies (log-odds within the parent population), optional
              negative-binomial tests of counts, population-first tests of
              marker medians, heatmaps and volcano plots, a per-sample
              population browser, and CSV export.

Gate application lives in ``ag_core.py`` (shared with the builder),
alignment and feature tables in ``ag_results.py``, and the statistics in
``controller_components/differential_stats.py`` (shared with the
DR/Clustering plugin). The two gating plugins have no runtime dependency on
each other.

Plugin contract (plugin_loaders.py):
  plugin_name  str        — main-window tab title
  PluginWidget(QWidget)   — instantiated with (bus, controller)
"""

import hashlib
import json
import os as _os
import sys
import warnings
from copy import deepcopy
from dataclasses import dataclass, field, fields as dataclass_fields
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QTimer, Signal, QSettings, QThread
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QPushButton, QLabel, QComboBox, QGroupBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QFormLayout,
    QSplitter, QFileDialog, QMessageBox, QInputDialog,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QTabWidget, QTreeWidget, QTreeWidgetItem,
    QAbstractItemView, QListWidget, QListWidgetItem,
    QProgressBar, QGridLayout,
)

import honeychrome
from honeychrome.settings import af_abundance_channel
from honeychrome.controller_components import differential_stats as ds
from honeychrome.controller_components.functions import (
    build_antigen_map,
    build_display_label_map,
    calc_hist1d,
    calc_hist2d,
)
from honeychrome.controller_components.sample_loader import (
    af_profile_names,
    analysis_sample_keys,
    control_sample_keys,
    load_unmixed,
    seeded_subsample_indices,
    snapshot_unmix_state,
)

# Sibling modules in this directory (not *_tab.py, so not loaded as tabs).
_PLUGIN_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

import ag_core  # noqa: E402
import ag_help_texts  # noqa: E402
import ag_report  # noqa: E402
import ag_results  # noqa: E402

log = ag_core.get_logger(__name__)


def _suppress_third_party_warnings():
    """Silence noisy third-party warnings from lazy scientific imports."""
    warnings.filterwarnings('ignore', category=ImportWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*SwigPy.*')
    warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*swigvarlink.*')


# ---------------------------------------------------------------------------
# Deferred Qt imports: pyqtgraph and the cytometry plot components create Qt
# objects on import, so they are imported on the main thread from
# PluginWidget.__init__ rather than at module scope (module import runs on
# the plugin loader's background thread).
# ---------------------------------------------------------------------------
pg = None
ZoomAxis = None
NoPanViewBox = None
TransparentGraphicsLayoutWidget = None
CopyableTableWidget = None
HelpToggleWidget = None
ExportablePlotWidget = None

_qt_imports_done = False


def _ensure_qt_imports():
    """Import pyqtgraph and Honeychrome view components (main thread only)."""
    global pg, ZoomAxis, NoPanViewBox, TransparentGraphicsLayoutWidget
    global CopyableTableWidget, HelpToggleWidget, ExportablePlotWidget, _qt_imports_done
    if _qt_imports_done:
        return
    import pyqtgraph as _pg
    pg = _pg
    from honeychrome.view_components.cytometry_plot_components import (
        ZoomAxis as _ZA, NoPanViewBox as _NPV,
        TransparentGraphicsLayoutWidget as _TGLW,
    )
    ZoomAxis = _ZA
    NoPanViewBox = _NPV
    TransparentGraphicsLayoutWidget = _TGLW
    from honeychrome.view_components.copyable_table_widget import (
        CopyableTableWidget as _CTW,
    )
    CopyableTableWidget = _CTW
    from honeychrome.view_components.help_toggle_widget import (
        HelpToggleWidget as _HTW,
    )
    HelpToggleWidget = _HTW
    from honeychrome.view_components.exportable_plot_widget import (
        ExportablePlotWidget as _EPW,
    )
    ExportablePlotWidget = _EPW
    _qt_imports_done = True


# ---------------------------------------------------------------------------
# Plugin identity and defaults
# ---------------------------------------------------------------------------
plugin_name = 'Automated Gating'

DEFAULT_DISPLAY_EVENTS = 20_000
DISPLAY_SEED = 42
RESULTS_FILE = Path('cache') / 'automated_gating' / 'results.json'

MODE_LABELS = {ag_core.MODE_FIXED: 'Fixed', ag_core.MODE_RECALCULATE: 'Recalculate'}
DRIFT_ACTIONS = {'flag': 'Flag only', 'reference': 'Use the stored boundary'}
FDR_SCOPES = {'global': 'Pooled over comparisons', 'per_comparison': 'Within each comparison'}
CONTRAST_MODES = {'reference': 'Each group against a reference', 'pairwise': 'All pairs of groups'}

_ALIGN_COLOURS = {
    ag_results.ALIGN_EXACT: '#d4edda',
    ag_results.ALIGN_ANTIGEN: '#d4edda',
    ag_results.ALIGN_CHANNEL: '#fff3cd',
    ag_results.ALIGN_FUZZY: '#fff3cd',
    ag_results.ALIGN_AMBIGUOUS: '#fff3cd',
    None: '#f8d7da',
}
_ALIGN_TEXT = {
    ag_results.ALIGN_EXACT: 'same channel and antigen',
    ag_results.ALIGN_ANTIGEN: 'antigen on another channel',
    ag_results.ALIGN_CHANNEL: 'same channel, different antigen — check',
    ag_results.ALIGN_FUZZY: 'closest name only — check',
    ag_results.ALIGN_AMBIGUOUS: 'antigen on several channels — check',
    None: 'no match — choose a channel',
    'manual': 'chosen by hand',
}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _experiment_channels(controller) -> list[str]:
    """Channel names (`$PnN`) of the experiment's unmixed event arrays."""
    try:
        return list(controller.experiment.settings.get('unmixed', {}).get('event_channels_pnn') or [])
    except Exception:
        return []


def _experiment_antigens(controller) -> list[str]:
    """Antigens from the spectral model, parallel to _experiment_channels();
    '' for channels without one."""
    pnn = _experiment_channels(controller)
    try:
        antigens = build_antigen_map(pnn, controller.experiment.process.get('spectral_model') or [])
    except Exception:
        antigens = {}
    return [antigens.get(ch, '') for ch in pnn]


def _antigen_map(controller) -> dict:
    """{channel: antigen, or the channel name when it has none}."""
    pnn = _experiment_channels(controller)
    return {ch: (ag or ch) for ch, ag in zip(pnn, _experiment_antigens(controller))}


def _fluorescence_channels(controller) -> list[str]:
    """Unmixed fluorescence channels (the markers summarised per population)."""
    pnn = _experiment_channels(controller)
    try:
        ids = controller.experiment.settings.get('unmixed', {}).get('fluorescence_channel_ids') or []
        return [pnn[i] for i in ids if 0 <= i < len(pnn)]
    except Exception:
        return []


def _marker_channels(controller) -> list[str]:
    """Channels summarised within every population: the fluorescence
    channels, plus AF Abundance when the experiment has it."""
    chans = _fluorescence_channels(controller)
    if af_abundance_channel in _experiment_channels(controller):
        chans.append(af_abundance_channel)
    return chans


def _axis_limits(transforms: dict) -> dict:
    """{channel: (lo, hi)} display limits from a transforms dict."""
    out = {}
    for ch, tr in (transforms or {}).items():
        lim = getattr(tr, 'limits', None)
        if lim is not None and len(lim) >= 2:
            out[ch] = (float(lim[0]), float(lim[1]))
    return out


def _live_transform_params(controller) -> dict:
    """{channel: transform parameter dict} of the current display transforms."""
    live = getattr(controller, 'unmixed_transformations', None) or {}
    return {ch: ag_core.transform_params(tr) for ch, tr in live.items()}


def _sample_label(key: str) -> str:
    """Short display name for a sample key."""
    return Path(str(key)).stem


def _gated_channels(gate_defs: list[dict]) -> list[str]:
    """Channels used as an axis by any gate, in hierarchy order."""
    by_name = ag_core.gates_by_name(gate_defs)
    out: list[str] = []
    for g in ag_core.ordered_gate_defs(gate_defs):
        for ch in ag_core.gate_channels(ag_core.effective_gate_def(g, by_name)):
            if ch and ch not in out:
                out.append(ch)
    return out


def _population_labels(gate_defs: list[dict]) -> dict:
    """{'gate/pop': 'gate label'} for display."""
    out = {}
    for g in gate_defs:
        for pop, info in (g.get('populations') or {}).items():
            label = (info or {}).get('label') or pop
            out[ag_core.population_key(g['gate_name'], pop)] = f"{g['gate_name']} {label}"
    return out


def _file_sha256(path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ''


def _time_qc_available() -> bool:
    try:
        from honeychrome.controller_components import time_qc  # noqa: F401
    except ImportError:
        return False
    return True


def _status(bus, msg: str):
    ts = datetime.now().strftime('%H:%M:%S')
    if bus is not None:
        bus.statusMessage.emit(f"[Automated Gating {ts}] {msg}")


def _clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.deleteLater()


def _coloured_item(text: str, colour: str | None = None) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
    if colour:
        item.setBackground(QColor(colour))
        item.setForeground(QColor('#000000'))
    return item


def _checkbox_item(checked: bool) -> QTableWidgetItem:
    item = QTableWidgetItem()
    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
    item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
    return item


# ---------------------------------------------------------------------------
# GatingRunState — owned by PluginWidget, shared by reference with every
# inner tab. Everything except `display` and `stats` is JSON-serialisable.
# ---------------------------------------------------------------------------

@dataclass
class GatingRunState:
    """Automated gating session state.

    model          : the normalised model as loaded (the model's channel names).
    alignment      : {model channel: experiment channel} for gated channels.
    applied_model  : the model renamed to this experiment's channels and
                     converted to its transforms; what the Run tab applies.
    modes          : {gate: 'fixed' | 'recalculate'}.
    summaries      : {sample key: ag_results.summarise_sample() record}.
    load_info      : {sample key: event counts and AF profiles used}.
    display        : {sample key: display subsample and masks} — this
                     session only.
    covariates     : {covariate name: {sample key: value}}.
    stats          : results of the last statistics run — this session only.
    """
    model_path: str = ''
    model: dict = field(default_factory=dict)
    model_messages: list = field(default_factory=list)
    alignment: dict = field(default_factory=dict)
    alignment_methods: dict = field(default_factory=dict)
    alignment_confirmed: bool = False
    applied_model: dict = field(default_factory=dict)
    modes: dict = field(default_factory=dict)
    drift_limit: float = ag_core.DEFAULT_DRIFT_LIMIT
    drift_action: str = 'flag'

    run_samples: list = field(default_factory=list)
    validation_samples: list = field(default_factory=list)
    include_controls: bool = False
    display_events: int = DEFAULT_DISPLAY_EVENTS
    summaries: dict = field(default_factory=dict)
    load_info: dict = field(default_factory=dict)
    marker_channels: list = field(default_factory=list)
    results_model_sha: str = ''
    display: dict = field(default_factory=dict)

    group_names: list = field(default_factory=list)
    group_patterns: dict = field(default_factory=dict)
    sample_groups: dict = field(default_factory=dict)
    covariates: dict = field(default_factory=dict)
    pairing: str = ''
    adjust_covariates: list = field(default_factory=list)
    contrast_mode: str = 'reference'
    reference_group: str = ''
    pval_threshold: float = 0.05
    fc_threshold: float = 0.5
    marker_threshold: float = 0.03
    use_treat: bool = True
    fdr_scope: str = 'global'
    run_freq: bool = True
    run_counts: bool = False
    run_markers: bool = True
    use_stats_parent: bool = True
    exclude_gating_markers: bool = True
    min_marker_events: int = ag_results.DEFAULT_MIN_MARKER_EVENTS
    exclude_flagged: bool = False
    stats: dict = field(default_factory=dict)

    def gate_defs(self) -> list[dict]:
        return list((self.applied_model or {}).get('gate_definitions') or [])

    def reference_boundaries(self) -> dict:
        return dict((self.applied_model or {}).get('trained_boundaries') or {})

    def gated_samples(self) -> list[str]:
        """Samples with results, in run order."""
        order = [k for k in self.run_samples if k in self.summaries]
        return order + [k for k in self.summaries if k not in order]

    def flags(self, key: str) -> list[str]:
        return ag_results.sample_flags(self.summaries.get(key) or {}, self.load_info.get(key))


_PERSISTED_FIELDS = (
    'model_path', 'alignment', 'alignment_methods', 'alignment_confirmed', 'modes',
    'drift_limit', 'drift_action', 'run_samples', 'validation_samples', 'include_controls',
    'display_events', 'group_names', 'group_patterns', 'sample_groups', 'covariates',
    'pairing', 'adjust_covariates', 'contrast_mode', 'reference_group', 'pval_threshold',
    'fc_threshold', 'marker_threshold', 'use_treat', 'fdr_scope', 'run_freq', 'run_counts',
    'run_markers', 'use_stats_parent', 'exclude_gating_markers', 'min_marker_events',
    'exclude_flagged',
)


def prepare_applied_model(model: dict, alignment: dict, live_params: dict) -> tuple[dict, list[str]]:
    """The model renamed to this experiment's channels and converted to its
    transforms, with messages describing any conversion."""
    aligned = ag_results.apply_alignment(model, alignment)
    return ag_core.remap_model(aligned, live_params)


# ---------------------------------------------------------------------------
# Setup tab
# ---------------------------------------------------------------------------

class SetupTab(QWidget):
    """Load a model, align its channels, and set how each gate is applied."""

    model_ready = Signal()

    def __init__(self, state: GatingRunState, bus, controller, parent=None):
        super().__init__(parent)
        self.state = state
        self.bus = bus
        self.controller = controller
        self._align_rows: list[str] = []
        self._read_warnings: list[str] = []
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)
        outer.addWidget(HelpToggleWidget(text=ag_help_texts.AG_SETUP))

        load_row = QHBoxLayout()
        self.btn_load = QPushButton("Load Model…")
        self.btn_load.clicked.connect(self._on_load_clicked)
        load_row.addWidget(self.btn_load)
        self.lbl_path = QLabel("No model loaded.")
        self.lbl_path.setWordWrap(True)
        load_row.addWidget(self.lbl_path, stretch=1)
        outer.addLayout(load_row)

        self.summary_group = QGroupBox("Model")
        sl = QVBoxLayout(self.summary_group)
        self.lbl_summary = QLabel("")
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        sl.addWidget(self.lbl_summary)
        outer.addWidget(self.summary_group)

        align_group = QGroupBox("Channel alignment (channels used by the gates)")
        al = QVBoxLayout(align_group)
        note = QLabel("Each model channel is matched to this experiment by antigen. Green rows "
                      "matched by antigen; amber rows need checking; red rows need a channel.")
        note.setWordWrap(True)
        al.addWidget(note)
        self.align_table = QTableWidget(0, 4)
        self.align_table.setHorizontalHeaderLabels(
            ["Model channel", "Model antigen", "Experiment channel", "Match"])
        self.align_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.align_table.verticalHeader().setVisible(False)
        self.align_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.align_table.setMinimumHeight(140)
        al.addWidget(self.align_table)
        acc_row = QHBoxLayout()
        self.lbl_align_state = QLabel("")
        self.lbl_align_state.setWordWrap(True)
        acc_row.addWidget(self.lbl_align_state, stretch=1)
        self.btn_accept = QPushButton("Accept Alignment")
        self.btn_accept.setEnabled(False)
        self.btn_accept.clicked.connect(self._on_accept_clicked)
        acc_row.addWidget(self.btn_accept)
        al.addLayout(acc_row)
        outer.addWidget(align_group)

        self.lbl_messages = QLabel("")
        self.lbl_messages.setWordWrap(True)
        self.lbl_messages.setStyleSheet("color: #856404;")
        outer.addWidget(self.lbl_messages)

        gates_group = QGroupBox("Gates")
        gl = QVBoxLayout(gates_group)
        opts = QHBoxLayout()
        opts.addWidget(QLabel("Drift limit:"))
        self.sb_drift = QDoubleSpinBox()
        self.sb_drift.setRange(0.001, 1.0)
        self.sb_drift.setDecimals(3)
        self.sb_drift.setSingleStep(0.01)
        self.sb_drift.setToolTip(
            "For recalculated gates: the largest shift from the stored boundary, in "
            "transformed axis units (the display axis spans about 0 to 1), before the "
            "sample is flagged.")
        self.sb_drift.valueChanged.connect(lambda v: setattr(self.state, 'drift_limit', float(v)))
        opts.addWidget(self.sb_drift)
        opts.addWidget(QLabel("  When exceeded:"))
        self.cb_drift_action = QComboBox()
        for key, text in DRIFT_ACTIONS.items():
            self.cb_drift_action.addItem(text, key)
        self.cb_drift_action.currentIndexChanged.connect(
            lambda _i: setattr(self.state, 'drift_action', self.cb_drift_action.currentData()))
        opts.addWidget(self.cb_drift_action)
        opts.addWidget(QLabel("  Set all:"))
        for mode in (ag_core.MODE_FIXED, ag_core.MODE_RECALCULATE):
            b = QPushButton(MODE_LABELS[mode])
            b.clicked.connect(lambda _c=False, m=mode: self._set_all_modes(m))
            opts.addWidget(b)
        opts.addStretch()
        gl.addLayout(opts)
        self.gate_tree = QTreeWidget()
        self.gate_tree.setColumnCount(6)
        self.gate_tree.setHeaderLabels(["Gate", "Type", "X axis", "Y axis", "Algorithm", "Application"])
        self.gate_tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.gate_tree.setMinimumHeight(160)
        gl.addWidget(self.gate_tree)
        outer.addWidget(gates_group, stretch=1)

    # ------------------------------------------------------------------

    def refresh(self):
        """Redraw from state (after a restore or an experiment change)."""
        self.sb_drift.blockSignals(True)
        self.sb_drift.setValue(float(self.state.drift_limit))
        self.sb_drift.blockSignals(False)
        idx = self.cb_drift_action.findData(self.state.drift_action)
        self.cb_drift_action.blockSignals(True)
        self.cb_drift_action.setCurrentIndex(max(idx, 0))
        self.cb_drift_action.blockSignals(False)
        if not self.state.model:
            self.lbl_path.setText("No model loaded.")
            self.lbl_summary.setText("")
            self.align_table.setRowCount(0)
            self.gate_tree.clear()
            self.lbl_messages.setText("")
            self.btn_accept.setEnabled(False)
            self.lbl_align_state.setText("")
            return
        self.lbl_path.setText(self.state.model_path)
        self._populate_summary()
        self._populate_alignment(keep_current=True)
        self._populate_gate_tree()
        self._show_messages()
        self._update_align_state()

    def _on_load_clicked(self):
        start = ''
        try:
            start = str(self.controller.experiment_dir)
        except Exception:
            pass
        path, _ = QFileDialog.getOpenFileName(self, "Load Gating Model", start,
                                              "Gating model (*.agmodel)")
        if path:
            self.load_model(path)

    def load_model(self, path: str, alignment: dict | None = None,
                   confirmed: bool = False) -> bool:
        """Read *path*; propose an alignment unless one is given."""
        try:
            model, warnings_list = ag_core.read_model(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", f"Could not read model:\n{exc}")
            return False
        st = self.state
        st.model_path = str(path)
        st.model = model
        self._read_warnings = list(warnings_list)
        st.model_messages = list(warnings_list)
        st.alignment_confirmed = False
        st.applied_model = {}
        app = model.get('application') or {}
        if not st.modes or set(st.modes) != {g['gate_name'] for g in model['gate_definitions']}:
            st.modes = dict(app.get('mode_per_gate') or {})
        if alignment is not None:
            st.alignment = dict(alignment)
        else:
            self._propose_alignment()
        self.lbl_path.setText(st.model_path)
        self._populate_summary()
        self._populate_alignment(keep_current=True)
        self._populate_gate_tree()
        if confirmed and not ag_results.alignment_problems(self._gated_alignment()):
            self._apply_alignment(quiet=True)
        self._show_messages()
        self._update_align_state()
        _status(self.bus, f"Model loaded: {Path(path).name}")
        return True

    def _model_gated_channels(self) -> list[str]:
        return _gated_channels(self.state.model.get('gate_definitions') or [])

    def _gated_alignment(self) -> dict:
        return {mc: self.state.alignment.get(mc) for mc in self._model_gated_channels()}

    def _propose_alignment(self):
        from honeychrome.controller_components.label_matching import get_marker_db, match_marker
        try:
            marker_db = get_marker_db()
            canonical = lambda name: match_marker(name, marker_db)  # noqa: E731
        except Exception:
            canonical = None
        proposal = ag_results.align_channels(
            self._model_gated_channels(), self.state.model.get('channel_map') or {},
            _experiment_channels(self.controller), _experiment_antigens(self.controller),
            canonical=canonical,
        )
        self.state.alignment = {mc: p['channel'] for mc, p in proposal.items()}
        self.state.alignment_methods = {mc: p['method'] for mc, p in proposal.items()}

    def _populate_summary(self):
        m = self.state.model
        meta = m.get('metadata') or {}
        ts = m.get('training_summary') or {}
        names = ts.get('training_sample_names') or []
        gates = m.get('gate_definitions') or []
        n_pops = sum(len(g.get('populations') or {}) for g in gates)
        af = (m.get('af') or {}).get('sample_af_profiles') or {}
        af_sets = sorted({', '.join(v) if v else 'none' for v in af.values()})
        lines = [
            f"<b>Experiment:</b> {meta.get('experiment_name', '—')} &nbsp; "
            f"<b>Created:</b> {meta.get('created', '—')} &nbsp; "
            f"<b>Honeychrome:</b> {meta.get('honeychrome_version', '—')}",
            f"<b>Cytometer:</b> {meta.get('cytometer_hint', '—') or '—'} &nbsp; "
            f"<b>Gates:</b> {len(gates)} &nbsp; <b>Populations:</b> {n_pops}",
            f"<b>Trained on:</b> {len(names)} sample(s)"
            + (f" — {', '.join(_sample_label(n) for n in names[:8])}"
               + (" …" if len(names) > 8 else "") if names else ""),
            f"<b>Training AF profiles:</b> {'; '.join(af_sets) if af_sets else '—'}",
        ]
        self.lbl_summary.setText("<br>".join(lines))

    def _populate_alignment(self, keep_current: bool = True):
        exp_pnn = _experiment_channels(self.controller)
        antigens = _antigen_map(self.controller)
        channel_map = self.state.model.get('channel_map') or {}
        rows = self._model_gated_channels()
        self._align_rows = rows
        self.align_table.setRowCount(0)
        for r, mc in enumerate(rows):
            self.align_table.insertRow(r)
            method = self.state.alignment_methods.get(mc)
            colour = _ALIGN_COLOURS.get(method, _ALIGN_COLOURS[None]) if method != 'manual' \
                else _ALIGN_COLOURS[ag_results.ALIGN_EXACT]
            self.align_table.setItem(r, 0, _coloured_item(mc, colour))
            self.align_table.setItem(r, 1, _coloured_item(channel_map.get(mc, ''), colour))
            combo = QComboBox()
            combo.addItem("— choose —", None)
            for ch in exp_pnn:
                if ch in ag_core.ALWAYS_EXCLUDED_CHANNELS:
                    continue
                label = antigens.get(ch, ch)
                combo.addItem(ch if label == ch else f"{ch} — {label}", ch)
            current = self.state.alignment.get(mc)
            i = combo.findData(current) if current else 0
            combo.setCurrentIndex(max(i, 0))
            combo.currentIndexChanged.connect(
                lambda _i, row=r, name=mc: self._on_align_changed(row, name))
            self.align_table.setCellWidget(r, 2, combo)
            self.align_table.setItem(r, 3, _coloured_item(_ALIGN_TEXT.get(method, ''), colour))

    def _on_align_changed(self, row: int, model_ch: str):
        combo = self.align_table.cellWidget(row, 2)
        if combo is None:
            return
        self.state.alignment[model_ch] = combo.currentData()
        self.state.alignment_methods[model_ch] = 'manual' if combo.currentData() else None
        colour = _ALIGN_COLOURS[ag_results.ALIGN_EXACT] if combo.currentData() else _ALIGN_COLOURS[None]
        for col in (0, 1, 3):
            item = self.align_table.item(row, col)
            if item is not None:
                item.setBackground(QColor(colour))
        self.align_table.item(row, 3).setText(_ALIGN_TEXT['manual' if combo.currentData() else None])
        if self.state.alignment_confirmed:
            self.state.alignment_confirmed = False
            self.state.applied_model = {}
            self._show_messages()
        self._update_align_state()

    def _update_align_state(self):
        problems = ag_results.alignment_problems(self._gated_alignment())
        needs_check = [mc for mc in self._align_rows
                       if self.state.alignment_methods.get(mc) not in
                       (ag_results.ALIGN_EXACT, ag_results.ALIGN_ANTIGEN, 'manual')
                       and self.state.alignment.get(mc)]
        if self.state.alignment_confirmed:
            text = "<b>Alignment accepted.</b> The Run tab uses it."
        elif problems:
            text = "<b>Before accepting:</b> " + "; ".join(problems)
        elif needs_check:
            text = "Check the amber rows, then accept: " + ", ".join(needs_check)
        else:
            text = "All gated channels matched by antigen."
        self.lbl_align_state.setText(text)
        self.btn_accept.setEnabled(bool(self.state.model) and not problems
                                   and not self.state.alignment_confirmed)

    def _on_accept_clicked(self):
        self._apply_alignment(quiet=False)

    def _apply_alignment(self, quiet: bool):
        st = self.state
        try:
            applied, messages = prepare_applied_model(st.model, self._gated_alignment(),
                                                      _live_transform_params(self.controller))
        except Exception as exc:
            if not quiet:
                QMessageBox.critical(self, "Cannot apply model", str(exc))
            log.exception("could not prepare the model")
            return
        st.applied_model = applied
        st.alignment_confirmed = True
        st.model_messages = list(self._read_warnings) + list(messages)
        self._show_messages()
        self._update_align_state()
        if not quiet:
            _status(self.bus, "Channel alignment accepted.")
            self.model_ready.emit()

    def _show_messages(self):
        msgs = list(self.state.model_messages)
        qc = (self.state.model.get('qc') or {}) if self.state.model else {}
        if qc.get('enabled') and not _time_qc_available():
            msgs.append("The model was trained on time-QC-filtered events, but time QC is "
                        "not available in this installation; samples are gated unfiltered.")
        self.lbl_messages.setText("<br>".join(msgs))

    def _populate_gate_tree(self):
        self.gate_tree.clear()
        gates = self.state.model.get('gate_definitions') or []
        items: dict[str, QTreeWidgetItem] = {}
        for g in ag_core.ordered_gate_defs(gates):
            name = g['gate_name']
            x = self.state.alignment.get(g.get('gate_marker_x')) or g.get('gate_marker_x') or ''
            y_model = g.get('gate_marker_y')
            y = (self.state.alignment.get(y_model) or y_model or '') if y_model else ''
            item = QTreeWidgetItem([name, g.get('gate_type', ''), x, y, g.get('algorithm', ''), ''])
            parent = items.get(g.get('parent_gate') or '')
            if parent is not None:
                parent.addChild(item)
            else:
                self.gate_tree.addTopLevelItem(item)
            items[name] = item
            combo = QComboBox()
            for mode, text in MODE_LABELS.items():
                combo.addItem(text, mode)
            if g.get('gate_type') == 'replicate':
                combo.setToolTip("A replicate gate copies its origin gate's boundary, "
                                 "whichever mode the origin uses.")
            mode = self.state.modes.get(name, ag_core.MODE_FIXED)
            combo.setCurrentIndex(max(combo.findData(mode), 0))
            combo.currentIndexChanged.connect(
                lambda _i, n=name, c=combo: self.state.modes.__setitem__(n, c.currentData()))
            self.gate_tree.setItemWidget(item, 5, combo)
        self.gate_tree.expandAll()

    def _set_all_modes(self, mode: str):
        for g in self.state.model.get('gate_definitions') or []:
            self.state.modes[g['gate_name']] = mode
        self._populate_gate_tree()


# ---------------------------------------------------------------------------
# Gate plot tiles (Run tab and population browser)
# ---------------------------------------------------------------------------

def _make_gate_tile(gate_def: dict, boundary: dict, events: np.ndarray | None,
                    channels: list[str], parent_mask: np.ndarray | None,
                    transforms: dict, axis_labels: dict, title: str,
                    fractions: dict | None = None, highlight: str | None = None,
                    footer: str = '') -> QWidget:
    """One gate plot: the parent population's events on the gate's axes with
    its boundary drawn over them.

    events       : untransformed unmixed events, columns named by *channels*.
    boundary     : {pop: entry} in transformed units.
    fractions    : {pop: fraction of parent} shown on the plot.
    highlight    : population drawn with a heavier outline.
    """
    import colorcet as cc
    import honeychrome.settings as hc_settings
    from PySide6.QtCore import QRectF

    gt = gate_def.get('gate_type', '')
    ch_x = gate_def.get('gate_marker_x') or ''
    ch_y = gate_def.get('gate_marker_y') or ''
    tr_x = transforms.get(ch_x)
    tr_y = transforms.get(ch_y) if ch_y else None
    tile_size = int(getattr(hc_settings, 'cytometry_plot_width_target_retrieved', 220) or 220)

    container = QWidget()
    container.setFixedSize(tile_size + 20, tile_size + 60)
    layout = QVBoxLayout(container)
    layout.setContentsMargins(2, 2, 2, 2)
    layout.setSpacing(2)
    lbl = QLabel(title)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setWordWrap(True)
    layout.addWidget(lbl)

    gw = TransparentGraphicsLayoutWidget()
    gw.setFixedSize(tile_size, tile_size)
    gl = gw.ci.layout
    gl.setHorizontalSpacing(0)
    gl.setVerticalSpacing(0)
    vb = NoPanViewBox()
    vb.setMouseEnabled(x=False, y=False)
    vb.raiseContextMenu = lambda ev: None
    gw.addItem(vb, row=1, col=2)
    axis_x = ZoomAxis('bottom', vb)
    axis_y = ZoomAxis('left', vb)
    gw.addItem(axis_y, row=1, col=1)
    gw.addItem(axis_x, row=2, col=2)
    axis_x.linkToView(vb)
    axis_y.linkToView(vb)
    axis_x.setLabel(axis_labels.get(ch_x, ch_x))
    axis_y.setLabel(axis_labels.get(ch_y, ch_y) if ch_y else 'Count')

    try:
        colours = cc.palette[hc_settings.colourmap_name_retrieved]
    except Exception:
        colours = cc.palette['rainbow4']
    cmap = pg.ColorMap(pos=0.9 * np.linspace(0, 1, len(colours)) ** 2
                       + 0.1 * np.linspace(0, 1, len(colours)), color=colours)
    lut = cmap.getLookupTable(alpha=True)
    lut[0, 3] = 0
    img = pg.ImageItem()
    img.setLookupTable(lut)
    vb.addItem(img)
    hist_curve = pg.PlotDataItem(stepMode='center', fillLevel=0, brush=(100, 100, 250, 150))
    vb.addItem(hist_curve)
    layout.addWidget(gw, stretch=1)

    def _set_axis(axis, tr):
        if tr.ticks:
            axis.setTicks(tr.ticks())
        axis.zoomZero = tr.zero
        axis.fullRange = (0, 1.1)
        axis.limits = tuple(tr.limits)

    if events is not None and tr_x is not None and ch_x in channels and len(events):
        ix = channels.index(ch_x)
        mask = parent_mask if (parent_mask is not None and len(parent_mask) == len(events)) \
            else np.ones(len(events), dtype=bool)
        if tr_y is not None and ch_y in channels:
            iy = channels.index(ch_y)
            heat = calc_hist2d(events, mask, ix, iy, tr_x, tr_y,
                               density_cutoff=hc_settings.density_cutoff_retrieved)
            img.setImage(heat)
            img.setRect(QRectF(tr_x.limits[0], tr_y.limits[0],
                               tr_x.limits[1] - tr_x.limits[0], tr_y.limits[1] - tr_y.limits[0]))
            _set_axis(axis_x, tr_x)
            _set_axis(axis_y, tr_y)
            vb.setXRange(tr_x.limits[0], tr_x.limits[1], padding=0)
            vb.setYRange(tr_y.limits[0], tr_y.limits[1], padding=0)
        else:
            count = calc_hist1d(events, mask, ix, tr_x)
            hist_curve.setData(tr_x.step_scale, count)
            _set_axis(axis_x, tr_x)
            vb.setXRange(tr_x.limits[0], tr_x.limits[1], padding=0)
            vb.enableAutoRange(axis=vb.YAxis, enable=True)

    colour = (30, 160, 50)
    fill = pg.mkBrush(255, 255, 255, 200)
    anchor_x = {'left': 0.0, 'center': 0.5, 'right': 1.0}
    anchor_y = {'top': 0.0, 'center': 0.5, 'bottom': 1.0}
    x_lim = tuple(tr_x.limits) if tr_x is not None else (0.0, 1.0)
    y_lim = tuple(tr_y.limits) if tr_y is not None else (0.0, 1.0)
    if gt == '1dsep':
        counts = hist_curve.yData
        y_lim = (0.0, float(np.max(counts)) if counts is not None and len(counts) else 1.0)
    regions = ag_core.population_regions(gate_def, boundary)
    fractions = fractions or {}
    for pop, entry in (boundary or {}).items():
        width = 2.5 if pop == highlight else 1.5
        label = (gate_def.get('populations') or {}).get(pop, {}).get('label', pop)
        frac = fractions.get(pop)
        text = f"{label}\n{frac * 100:.1f}%" if frac is not None else label
        if gt == '1dsep':
            tx = entry.get('threshold_x')
            if tx is None:
                continue
            vb.addItem(pg.InfiniteLine(pos=float(tx), angle=90, pen=pg.mkPen(color=colour, width=1.5)))
        else:
            coords = np.asarray(entry.get('boundary') or [], dtype=float)
            if coords.ndim != 2 or coords.shape[1] != 2 or not len(coords):
                continue
            xs_b, ys_b = coords[:, 0].tolist(), coords[:, 1].tolist()
            if xs_b[0] != xs_b[-1] or ys_b[0] != ys_b[-1]:
                xs_b.append(xs_b[0])
                ys_b.append(ys_b[0])
            vb.addItem(pg.PlotDataItem(xs_b, ys_b, pen=pg.mkPen(color=colour, width=width)))
        lx, ly, ha, va = ag_report.label_position(gt, regions.get(pop), entry, x_lim, y_lim)
        t = pg.TextItem(text, color=colour, fill=fill, anchor=(anchor_x[ha], anchor_y[va]))
        t.setPos(lx, ly)
        vb.addItem(t)

    if footer:
        f = QLabel(footer)
        f.setWordWrap(True)
        f.setStyleSheet("color: #856404; font-size: 8pt;")
        layout.addWidget(f)
    return container


# ---------------------------------------------------------------------------
# Gating worker
# ---------------------------------------------------------------------------

class _GatingWorker(QThread):
    """Gate each sample with the applied model, off the main thread.

    Receives only values snapshotted on the main thread. For each sample:
    load and unmix it with its own AF assignment (and time-QC keep-mask),
    transform, run the gate hierarchy, and emit sample_done(key, payload)
    with the compact summary, load counts and a display subsample. The full
    event array is released before the next sample.
    """

    progress = Signal(str)
    sample_done = Signal(str, object)
    finished = Signal(bool, str)

    def __init__(self, experiment_dir, keys, snapshot, gate_defs, reference, modes,
                 drift_limit, drift_action, channels, transforms, marker_channels,
                 display_events, parent=None):
        super().__init__(parent)
        self._experiment_dir = experiment_dir
        self._keys = list(keys)
        self._snap = snapshot
        self._gate_defs = gate_defs
        self._reference = reference
        self._modes = modes
        self._drift_limit = float(drift_limit)
        self._drift_action = drift_action
        self._channels = channels
        self._transforms = transforms
        self._markers = marker_channels
        self._display = int(display_events)
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            n_done = self._execute()
        except Exception as exc:
            log.exception("automated gating failed")
            self.finished.emit(False, str(exc))
            return
        if self._abort:
            self.finished.emit(False, f"Cancelled after {n_done} sample(s).")
        else:
            self.finished.emit(True, "")

    def _execute(self) -> int:
        index = {ch: i for i, ch in enumerate(self._channels)}
        display_cols = [ch for ch in _gated_channels(self._gate_defs) if ch in index]
        display_idx = [index[ch] for ch in display_cols]
        limits = _axis_limits(self._transforms)
        n_done = 0
        for i, key in enumerate(self._keys, 1):
            if self._abort:
                break
            self.progress.emit(f"Gating {_sample_label(key)} ({i}/{len(self._keys)}) …")
            try:
                out = load_unmixed(self._experiment_dir, key, self._snap)
            except Exception as exc:
                log.warning("could not load %s: %s", key, exc)
                self.progress.emit(f"Could not load {_sample_label(key)}: {exc}")
                continue
            unmixed = out['unmixed']
            data_t = ag_core.transform_columns(unmixed, self._channels, self._transforms)
            result = ag_core.run_gating(
                self._gate_defs, data_t, index, reference=self._reference,
                modes=self._modes, drift_limit=self._drift_limit,
                drift_action=self._drift_action, axis_limits=limits,
            )
            summary = ag_results.summarise_sample(self._gate_defs, result, data_t, index,
                                                  self._markers, len(data_t))
            rows = seeded_subsample_indices(len(unmixed), self._display, key, DISPLAY_SEED)
            display = {
                'channels': display_cols,
                'events': np.asarray(unmixed[np.ix_(rows, display_idx)], dtype=np.float32),
                'parent_masks': {g: m[rows] for g, m in result.parent_masks.items()},
                'pop_masks': {g: {p: m[rows] for p, m in pops.items()}
                              for g, pops in result.masks.items()},
            }
            load_info = {
                'n_events_file': int(out['n_events_file']),
                'n_events_kept': int(out['n_events_kept']),
                'af_profiles': list(out['af_profiles']),
            }
            del unmixed, data_t, result, out
            self.sample_done.emit(key, {'summary': summary, 'load_info': load_info,
                                        'display': display})
            n_done += 1
        return n_done


# ---------------------------------------------------------------------------
# Run tab
# ---------------------------------------------------------------------------

POOLED_VIEW = '__pooled__'


class RunTab(QWidget):
    """Pick samples, gate them, and inspect the result gate by gate."""

    gating_finished = Signal()

    COL_RUN, COL_VALIDATE, COL_SAMPLE, COL_AF = range(4)

    def __init__(self, state: GatingRunState, bus, controller, parent=None):
        super().__init__(parent)
        self.state = state
        self.bus = bus
        self.controller = controller
        self._worker: _GatingWorker | None = None
        self._run_keys: list[str] = []
        self._filling = False
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)
        outer.addWidget(HelpToggleWidget(text=ag_help_texts.AG_RUN))

        self.banner = QLabel("Load a model and accept its channel alignment on the Setup tab first.")
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet("QLabel { color: #856404; background: #fff3cd; "
                                  "border: 1px solid #ffc107; border-radius: 4px; padding: 6px; }")
        outer.addWidget(self.banner)

        self.content = QWidget()
        cl = QVBoxLayout(self.content)
        cl.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.content, stretch=1)

        splitter = QSplitter(Qt.Vertical)
        cl.addWidget(splitter)

        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("<b>Samples</b>"))
        b_all = QPushButton("Select all")
        b_all.clicked.connect(lambda: self._set_all(self.COL_RUN, True))
        b_none = QPushButton("Select none")
        b_none.clicked.connect(lambda: self._set_all(self.COL_RUN, False))
        sel_row.addWidget(b_all)
        sel_row.addWidget(b_none)
        self.chk_controls = QCheckBox("Show controls")
        self.chk_controls.setToolTip("Also offer single-stain controls and unstained samples.")
        self.chk_controls.toggled.connect(self._on_controls_toggled)
        sel_row.addWidget(self.chk_controls)
        sel_row.addStretch()
        sel_row.addWidget(QLabel("Display events per sample:"))
        self.sb_display = QSpinBox()
        self.sb_display.setRange(1_000, 1_000_000)
        self.sb_display.setSingleStep(5_000)
        self.sb_display.setToolTip(
            "Events kept per sample for plotting. Counts and statistics always use every event.")
        self.sb_display.valueChanged.connect(lambda v: setattr(self.state, 'display_events', int(v)))
        sel_row.addWidget(self.sb_display)
        tl.addLayout(sel_row)

        self.sample_table = QTableWidget(0, 4)
        self.sample_table.setHorizontalHeaderLabels(["Gate", "Validate", "Sample", "AF profiles"])
        hh = self.sample_table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(self.COL_SAMPLE, QHeaderView.Stretch)
        self.sample_table.verticalHeader().setVisible(False)
        self.sample_table.setToolTip("Tick samples to gate. Validation samples can be viewed "
                                     "one at a time below after gating.")
        self.sample_table.itemChanged.connect(self._on_sample_item_changed)
        tl.addWidget(self.sample_table)

        self.lbl_warnings = QLabel("")
        self.lbl_warnings.setWordWrap(True)
        self.lbl_warnings.setStyleSheet("color: #856404;")
        tl.addWidget(self.lbl_warnings)

        run_row = QHBoxLayout()
        self.btn_run = QPushButton("Run Gating")
        self.btn_run.clicked.connect(self._on_run_clicked)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        run_row.addWidget(self.btn_run)
        run_row.addWidget(self.btn_cancel)
        run_row.addWidget(self.progress_bar)
        run_row.addWidget(self.lbl_status, stretch=1)
        tl.addLayout(run_row)
        splitter.addWidget(top)

        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("<b>View:</b>"))
        self.cb_view = QComboBox()
        self.cb_view.currentIndexChanged.connect(lambda _i: self._draw_tiles())
        view_row.addWidget(self.cb_view, stretch=1)
        bl.addLayout(view_row)
        self.tile_scroll = QScrollArea()
        self.tile_scroll.setWidgetResizable(True)
        self.tile_canvas = QWidget()
        self.tile_grid = QGridLayout(self.tile_canvas)
        self.tile_grid.setSpacing(4)
        self.tile_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.tile_scroll.setWidget(self.tile_canvas)
        bl.addWidget(self.tile_scroll, stretch=3)
        bl.addWidget(QLabel("<b>Samples gated</b>"))
        self.summary_holder = QVBoxLayout()
        bl.addLayout(self.summary_holder, stretch=1)
        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

    # ------------------------------------------------------------------

    def refresh(self):
        ready = bool(self.state.alignment_confirmed and self.state.applied_model)
        self.banner.setVisible(not ready)
        self.content.setVisible(ready)
        self.chk_controls.blockSignals(True)
        self.chk_controls.setChecked(bool(self.state.include_controls))
        self.chk_controls.blockSignals(False)
        self.sb_display.blockSignals(True)
        self.sb_display.setValue(int(self.state.display_events))
        self.sb_display.blockSignals(False)
        if ready:
            self._populate_samples()
            self._update_warnings()
            self._populate_views()
            self._draw_summary_table()

    def _on_controls_toggled(self, on: bool):
        self.state.include_controls = bool(on)
        self._populate_samples()

    def _populate_samples(self):
        try:
            keys = analysis_sample_keys(self.controller, self.state.include_controls)
        except Exception:
            keys = []
        run = set(self.state.run_samples)
        val = set(self.state.validation_samples)
        self._filling = True
        try:
            self.sample_table.setRowCount(0)
            for r, key in enumerate(keys):
                self.sample_table.insertRow(r)
                self.sample_table.setItem(r, self.COL_RUN, _checkbox_item(key in run))
                self.sample_table.setItem(r, self.COL_VALIDATE, _checkbox_item(key in val))
                name = _coloured_item(key)
                name.setData(Qt.UserRole, key)
                self.sample_table.setItem(r, self.COL_SAMPLE, name)
                profiles = af_profile_names(self.controller, key)
                self.sample_table.setItem(r, self.COL_AF, _coloured_item(', '.join(profiles) or '—'))
        finally:
            self._filling = False

    def _row_key(self, row: int) -> str:
        return self.sample_table.item(row, self.COL_SAMPLE).data(Qt.UserRole)

    def _checked(self, col: int) -> list[str]:
        return [self._row_key(r) for r in range(self.sample_table.rowCount())
                if self.sample_table.item(r, col).checkState() == Qt.Checked]

    def _on_sample_item_changed(self, item: QTableWidgetItem):
        if self._filling or item.column() not in (self.COL_RUN, self.COL_VALIDATE):
            return
        self.state.run_samples = self._checked(self.COL_RUN)
        self.state.validation_samples = self._checked(self.COL_VALIDATE)
        self._update_warnings()

    def _set_all(self, col: int, on: bool):
        self._filling = True
        try:
            for r in range(self.sample_table.rowCount()):
                self.sample_table.item(r, col).setCheckState(Qt.Checked if on else Qt.Unchecked)
        finally:
            self._filling = False
        self.state.run_samples = self._checked(self.COL_RUN)
        self.state.validation_samples = self._checked(self.COL_VALIDATE)
        self._update_warnings()

    def _update_warnings(self):
        """Samples whose AF assignment was not represented in training."""
        training = ((self.state.model or {}).get('af') or {}).get('sample_af_profiles') or {}
        if not training:
            self.lbl_warnings.setText("")
            return
        seen = {tuple(v or []) for v in training.values()}
        odd = [k for k in self.state.run_samples
               if tuple(af_profile_names(self.controller, k)) not in seen]
        if odd:
            self.lbl_warnings.setText(
                "These samples use an AF profile assignment that none of the training "
                "samples had, so their unmixing differs from the training data: "
                + ", ".join(_sample_label(k) for k in odd[:10]) + (" …" if len(odd) > 10 else ""))
        else:
            self.lbl_warnings.setText("")

    # ------------------------------------------------------------------

    def _on_run_clicked(self):
        st = self.state
        keys = self._checked(self.COL_RUN)
        if not keys:
            QMessageBox.information(self, "No samples", "Tick at least one sample to gate.")
            return
        channels = _experiment_channels(self.controller)
        missing = [ch for ch in _gated_channels(st.gate_defs()) if ch not in channels]
        if missing:
            QMessageBox.warning(self, "Channels missing",
                                "Gated channels not in this experiment: " + ", ".join(missing))
            return
        try:
            snap = snapshot_unmix_state(self.controller, keys)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot gate", str(exc))
            return
        st.run_samples = keys
        st.summaries = {}
        st.load_info = {}
        st.display = {}
        st.stats = {}
        st.marker_channels = _marker_channels(self.controller)
        st.results_model_sha = _file_sha256(st.model_path)
        self._run_keys = keys

        self._worker = _GatingWorker(
            self.controller.experiment_dir, keys, snap,
            deepcopy(st.gate_defs()), deepcopy(st.reference_boundaries()), dict(st.modes),
            st.drift_limit, st.drift_action, channels,
            deepcopy(self.controller.unmixed_transformations or {}),
            list(st.marker_channels), st.display_events, parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.sample_done.connect(self._on_sample_done)
        self._worker.finished.connect(self._on_finished)
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setRange(0, len(keys))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self._worker.start()

    def _on_cancel_clicked(self):
        if self._worker is not None:
            self._worker.abort()

    def _on_progress(self, msg: str):
        self.lbl_status.setText(msg)
        _status(self.bus, msg)

    def _on_sample_done(self, key: str, payload: object):
        self.state.summaries[key] = payload['summary']
        self.state.load_info[key] = payload['load_info']
        self.state.display[key] = payload['display']
        self.progress_bar.setValue(self.progress_bar.value() + 1)

    def _on_finished(self, ok: bool, message: str):
        self._worker = None
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
        n = len(self.state.summaries)
        flagged = sum(1 for k in self.state.summaries if self.state.flags(k))
        if ok:
            text = f"Gated {n} sample(s)" + (f"; {flagged} flagged." if flagged else ".")
        else:
            text = f"{message} {n} sample(s) gated."
        self.lbl_status.setText(text)
        _status(self.bus, text)
        self._populate_views()
        self._draw_summary_table()
        if n:
            self.gating_finished.emit()

    # ------------------------------------------------------------------

    def _populate_views(self):
        current = self.cb_view.currentData()
        self.cb_view.blockSignals(True)
        self.cb_view.clear()
        if self.state.display:
            self.cb_view.addItem("All gated samples (pooled display events)", POOLED_VIEW)
            for key in self.state.gated_samples():
                if key in self.state.validation_samples and key in self.state.display:
                    self.cb_view.addItem(f"Validation: {_sample_label(key)}", key)
        i = self.cb_view.findData(current)
        self.cb_view.setCurrentIndex(max(i, 0))
        self.cb_view.blockSignals(False)
        self._draw_tiles()

    def _draw_tiles(self):
        _clear_layout(self.tile_grid)
        st = self.state
        view = self.cb_view.currentData()
        gates = ag_core.ordered_gate_defs(st.gate_defs())
        if not gates or not st.display or view is None:
            note = QLabel("Gate some samples to see their gates here."
                          if not st.summaries else
                          "Plots are available for samples gated in this session; "
                          "run gating again to see them.")
            note.setWordWrap(True)
            self.tile_grid.addWidget(note, 0, 0)
            return
        by_name = ag_core.gates_by_name(gates)
        transforms = self.controller.unmixed_transformations or {}
        labels = build_display_label_map(
            _experiment_channels(self.controller),
            self.controller.experiment.process.get('spectral_model') or [])
        cols = 4
        for i, g in enumerate(gates):
            name = g['gate_name']
            eff = ag_core.effective_gate_def(g, by_name)
            if view == POOLED_VIEW:
                events, parent, channels = ag_report.pooled_display(
                    st.display, st.gated_samples(), name)
                boundary = st.reference_boundaries().get(name, {})
                fractions = ag_report.boundary_fractions(eff, boundary, events, channels,
                                                         parent, transforms)
                mode = st.modes.get(name, ag_core.MODE_FIXED)
                n_flag = sum(1 for k in st.summaries
                             if ((st.summaries[k].get('status') or {}).get(name) or {}).get('flag'))
                title = (f"<b>{name}</b> <small>[{MODE_LABELS[mode]}; stored boundary, "
                         f"% of pooled display events]</small>")
                footer = f"{n_flag} sample(s) flagged at this gate" if n_flag else ''
            else:
                disp = st.display.get(view) or {}
                channels = disp.get('channels') or []
                events = disp.get('events')
                parent = (disp.get('parent_masks') or {}).get(name)
                summ = st.summaries.get(view) or {}
                boundary = (summ.get('boundaries') or {}).get(name, {})
                fractions = {r['population']: r['fraction_of_parent']
                             for r in summ.get('populations', []) if r['gate'] == name}
                status = (summ.get('status') or {}).get(name) or {}
                title = f"<b>{name}</b> <small>[{status.get('source') or '—'}]</small>"
                footer = status.get('message') if status.get('flag') else ''
            tile = _make_gate_tile(eff, boundary, events, channels, parent, transforms, labels,
                                   title, fractions=fractions, footer=footer)
            r, c = divmod(i, cols)
            self.tile_grid.addWidget(tile, r, c)

    def _draw_summary_table(self):
        _clear_layout(self.summary_holder)
        st = self.state
        rows = []
        for key in st.gated_samples():
            info = st.load_info.get(key) or {}
            rows.append({
                'Sample': _sample_label(key),
                'Events in file': info.get('n_events_file', ''),
                'Events gated': info.get('n_events_kept', ''),
                'AF profiles': ', '.join(info.get('af_profiles') or []) or '—',
                'Flags': '; '.join(st.flags(key)),
            })
        if not rows:
            return
        table = CopyableTableWidget(rows, ['Sample', 'Events in file', 'Events gated',
                                           'AF profiles', 'Flags'])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.summary_holder.addWidget(table)


# ---------------------------------------------------------------------------
# Results: background workers
# ---------------------------------------------------------------------------

class _CallWorker(QThread):
    """Run a pure function off the main thread; emits finished(ok, error, result)."""

    finished = Signal(bool, str, object)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            result = self._fn()
        except Exception as exc:
            log.exception("background task failed")
            self.finished.emit(False, str(exc), None)
            return
        self.finished.emit(True, '', result)


def _comparison_label(base: str, other: str) -> str:
    return f"{other} vs {base}"


def _comparison_view(results: pd.DataFrame | None, label: str) -> pd.DataFrame | None:
    if results is None or 'comparison' not in results.columns:
        return results
    return results[results['comparison'] == label].drop(columns=['comparison']).reset_index(drop=True)


def build_result_figures(stats: dict, base: str, other: str, pval_threshold: float,
                         fc_threshold: float, marker_threshold: float, fdr_scope: str,
                         is_dark: bool, antigens: dict, pop_labels: dict) -> dict:
    """Every results figure for one comparison. Pure matplotlib (no canvas),
    so it runs on a worker thread.

    Returns {key: {'fig', 'title', 'error', 'results'}}; 'results' is the
    table a volcano was drawn from (for click-through), in plotted order.
    """
    from honeychrome.view_components import differential_plots as dp

    label = _comparison_label(base, other)
    keep = [s for s, g in zip(stats['samples'], stats['group_vec']) if g in (base, other)]
    group_by_sample = dict(zip(stats['samples'], stats['group_vec']))
    run_label = f"{label}"
    out: dict = {}

    def _add(key, tab_title, maker, results=None, **kwargs):
        try:
            out[key] = {'fig': maker(**kwargs), 'title': tab_title, 'error': None,
                        'results': results}
        except Exception as exc:
            log.exception("figure %s failed", key)
            out[key] = {'fig': None, 'title': tab_title, 'error': str(exc), 'results': None}

    def _rows(df):
        return df.loc[[s for s in keep if s in df.index]] if df is not None else None

    specs = (
        ('freq', 'Frequency', stats.get('freq'), stats.get('freq_values'),
         '% of parent', 'log2 odds ratio', fc_threshold),
        ('counts', 'Counts', stats.get('counts'), stats.get('counts_values'),
         'Events', 'log2 fold change', fc_threshold),
    )
    for key, name, res, values, value_label, x_label, thr in specs:
        view = _comparison_view(res, label)
        if view is None:
            continue
        labels = [pop_labels.get(f, f) for f in view['feature']]
        _add(f'{key}_heatmap', f"{name} heatmap", dp.make_heatmap_figure,
             results_df=view, sample_df=_rows(values),
             title=f"{name}: significant populations, {label}",
             group_by_sample=group_by_sample, group_a=base, group_b=other, is_dark=is_dark,
             run_label=run_label, value_label=value_label,
             feature_labels={f: pop_labels.get(f, f) for f in view['feature']})
        _add(f'{key}_volcano', f"{name} volcano", dp.make_volcano_figure, results=view,
             results_df=view, title=f"{name} volcano: {label}", pval_threshold=pval_threshold,
             fc_threshold=thr, is_dark=is_dark, fdr_scope=fdr_scope, feature_labels=labels,
             x_label=x_label, run_label=run_label)

    mview = _comparison_view(stats.get('markers'), label)
    if mview is not None:
        names = {ch: antigens.get(ch, ch) for ch in mview['channel'].unique()}
        _add('marker_summary', "Marker summary", dp.make_marker_summary_figure,
             results_df=mview, title=f"Marker medians by population: {label}",
             channel_names=names, is_dark=is_dark, run_label=run_label,
             family_label='population', value_label='Δ median, transformed (● significant)')
        mlabels = [f"{r['cluster']} {names.get(r['channel'], r['channel'])}"
                   for _i, r in mview.iterrows()]
        _add('marker_volcano', "Marker volcano", dp.make_volcano_figure, results=mview,
             results_df=mview, title=f"Marker volcano: {label}", pval_threshold=pval_threshold,
             fc_threshold=marker_threshold, is_dark=is_dark, fdr_scope=fdr_scope,
             feature_labels=mlabels, x_label='Δ median (transformed units)', run_label=run_label)
    return out


def _is_dark_palette() -> bool:
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    palette = app.palette() if app is not None else QPalette()
    return palette.color(QPalette.ColorRole.Base).value() < 128


# ---------------------------------------------------------------------------
# Results tab
# ---------------------------------------------------------------------------

class ResultsTab(QWidget):
    """Groups and covariates, statistics, figures, population browser, export."""

    def __init__(self, state: GatingRunState, bus, controller, parent=None):
        super().__init__(parent)
        self.state = state
        self.bus = bus
        self.controller = controller
        self._stats_worker: _CallWorker | None = None
        self._figure_worker: _CallWorker | None = None
        self._figure_pending = False
        self._filling = False
        self._volcano_tables: dict = {}
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)
        outer.addWidget(HelpToggleWidget(text=ag_help_texts.AG_RESULTS))
        cmp_row = QHBoxLayout()
        cmp_row.addWidget(QLabel("<b>Comparison:</b>"))
        self.cb_comparison = QComboBox()
        self.cb_comparison.currentIndexChanged.connect(self._on_comparison_changed)
        cmp_row.addWidget(self.cb_comparison, stretch=1)
        outer.addLayout(cmp_row)
        self.sub_tabs = QTabWidget()
        self.sub_tabs.setDocumentMode(True)
        outer.addWidget(self.sub_tabs, stretch=1)

        self.groups_page = QWidget()
        self._build_groups_page()
        self.sub_tabs.addTab(self.groups_page, "Groups")
        self.stats_page = QWidget()
        self._build_stats_page()
        self.sub_tabs.addTab(self.stats_page, "Statistics")
        self.figures_tabs = QTabWidget()
        self.sub_tabs.addTab(self.figures_tabs, "Figures")
        self.browser_page = QWidget()
        self._build_browser_page()
        self.sub_tabs.addTab(self.browser_page, "Populations")
        self.export_page = QWidget()
        self._build_export_page()
        self.sub_tabs.addTab(self.export_page, "Export")

    # ------------------------------------------------------------------
    # Groups
    # ------------------------------------------------------------------

    def _build_groups_page(self):
        layout = QVBoxLayout(self.groups_page)
        top = QHBoxLayout()

        gbox = QGroupBox("Groups")
        gl = QVBoxLayout(gbox)
        self.groups_table = QTableWidget(0, 2)
        self.groups_table.setHorizontalHeaderLabels(["Group", "Sample-name pattern (regex)"])
        self.groups_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.groups_table.verticalHeader().setVisible(False)
        self.groups_table.itemChanged.connect(self._on_group_item_changed)
        gl.addWidget(self.groups_table)
        row = QHBoxLayout()
        for text, slot in (("Add group", self._on_add_group), ("Remove group", self._on_remove_group),
                           ("Assign by pattern", self._on_assign_by_pattern)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        gl.addLayout(row)
        top.addWidget(gbox, stretch=1)

        cbox = QGroupBox("Covariates")
        cl = QVBoxLayout(cbox)
        note = QLabel("Per-sample values, e.g. donor (for pairing) or age and sex (for "
                      "adjustment). Numbers are treated as continuous, anything else as "
                      "categories.")
        note.setWordWrap(True)
        cl.addWidget(note)
        crow = QHBoxLayout()
        b_add = QPushButton("Add covariate…")
        b_add.clicked.connect(self._on_add_covariate)
        b_rm = QPushButton("Remove covariate…")
        b_rm.clicked.connect(self._on_remove_covariate)
        crow.addWidget(b_add)
        crow.addWidget(b_rm)
        cl.addLayout(crow)
        cl.addStretch()
        top.addWidget(cbox, stretch=1)
        layout.addLayout(top)

        self.samples_table = QTableWidget(0, 3)
        self.samples_table.verticalHeader().setVisible(False)
        self.samples_table.itemChanged.connect(self._on_sample_cell_changed)
        layout.addWidget(self.samples_table, stretch=1)

    def _populate_groups(self):
        self._filling = True
        try:
            self.groups_table.setRowCount(0)
            for r, name in enumerate(self.state.group_names):
                self.groups_table.insertRow(r)
                item = QTableWidgetItem(name)
                item.setData(Qt.UserRole, name)
                self.groups_table.setItem(r, 0, item)
                self.groups_table.setItem(r, 1, QTableWidgetItem(self.state.group_patterns.get(name, '')))
        finally:
            self._filling = False
        self._populate_samples_table()

    def _populate_samples_table(self):
        st = self.state
        covs = list(st.covariates)
        headers = ["Sample", "Group", "Flags"] + covs
        self._filling = True
        try:
            self.samples_table.setRowCount(0)
            self.samples_table.setColumnCount(len(headers))
            self.samples_table.setHorizontalHeaderLabels(headers)
            for r, key in enumerate(st.gated_samples()):
                self.samples_table.insertRow(r)
                name = _coloured_item(_sample_label(key))
                name.setData(Qt.UserRole, key)
                name.setToolTip(key)
                self.samples_table.setItem(r, 0, name)
                combo = QComboBox()
                combo.addItem("—", '')
                for g in st.group_names:
                    combo.addItem(g, g)
                combo.setCurrentIndex(max(combo.findData(st.sample_groups.get(key, '')), 0))
                combo.currentIndexChanged.connect(
                    lambda _i, k=key, c=combo: self._set_sample_group(k, c.currentData()))
                self.samples_table.setCellWidget(r, 1, combo)
                flags = st.flags(key)
                self.samples_table.setItem(r, 2, _coloured_item(
                    '; '.join(flags), '#fff3cd' if flags else None))
                for j, cov in enumerate(covs, start=3):
                    self.samples_table.setItem(r, j, QTableWidgetItem(
                        str((st.covariates.get(cov) or {}).get(key, ''))))
            hh = self.samples_table.horizontalHeader()
            hh.setSectionResizeMode(QHeaderView.ResizeToContents)
            hh.setSectionResizeMode(0, QHeaderView.Stretch)
        finally:
            self._filling = False

    def _set_sample_group(self, key: str, group: str):
        if group:
            self.state.sample_groups[key] = group
        else:
            self.state.sample_groups.pop(key, None)

    def _on_group_item_changed(self, item: QTableWidgetItem):
        if self._filling:
            return
        st = self.state
        old = self.groups_table.item(item.row(), 0).data(Qt.UserRole)
        if item.column() == 0:
            new = item.text().strip()
            if not new or (new != old and new in st.group_names):
                self._filling = True
                item.setText(old)
                self._filling = False
                return
            st.group_names[st.group_names.index(old)] = new
            if old in st.group_patterns:
                st.group_patterns[new] = st.group_patterns.pop(old)
            for k, g in list(st.sample_groups.items()):
                if g == old:
                    st.sample_groups[k] = new
            if st.reference_group == old:
                st.reference_group = new
            item.setData(Qt.UserRole, new)
            self._populate_samples_table()
            self._populate_stats_choices()
        else:
            st.group_patterns[old] = item.text().strip()

    def _on_add_group(self):
        n = 1
        while f"Group {n}" in self.state.group_names:
            n += 1
        self.state.group_names.append(f"Group {n}")
        self._populate_groups()
        self._populate_stats_choices()

    def _on_remove_group(self):
        row = self.groups_table.currentRow()
        if row < 0:
            return
        name = self.groups_table.item(row, 0).data(Qt.UserRole)
        st = self.state
        st.group_names.remove(name)
        st.group_patterns.pop(name, None)
        st.sample_groups = {k: g for k, g in st.sample_groups.items() if g != name}
        self._populate_groups()
        self._populate_stats_choices()

    def _on_assign_by_pattern(self):
        import re
        st = self.state
        assigned, bad = 0, []
        for key in st.gated_samples():
            for name in st.group_names:
                pat = st.group_patterns.get(name, '').strip()
                if not pat:
                    continue
                try:
                    hit = re.search(pat, key)
                except re.error:
                    bad.append(name)
                    continue
                if hit:
                    st.sample_groups[key] = name
                    assigned += 1
                    break
        self._populate_samples_table()
        msg = f"Assigned {assigned} sample(s) by pattern."
        if bad:
            msg += " Invalid pattern for: " + ", ".join(sorted(set(bad)))
        _status(self.bus, msg)

    def _on_add_covariate(self):
        name, ok = QInputDialog.getText(self, "Add covariate", "Covariate name:")
        name = (name or '').strip()
        if not ok or not name:
            return
        if name in self.state.covariates:
            QMessageBox.information(self, "Covariate exists", f"'{name}' is already defined.")
            return
        self.state.covariates[name] = {}
        self._populate_samples_table()
        self._populate_stats_choices()

    def _on_remove_covariate(self):
        names = list(self.state.covariates)
        if not names:
            return
        name, ok = QInputDialog.getItem(self, "Remove covariate", "Covariate:", names, 0, False)
        if not ok:
            return
        self.state.covariates.pop(name, None)
        if self.state.pairing == name:
            self.state.pairing = ''
        self.state.adjust_covariates = [c for c in self.state.adjust_covariates if c != name]
        self._populate_samples_table()
        self._populate_stats_choices()

    def _on_sample_cell_changed(self, item: QTableWidgetItem):
        if self._filling or item.column() < 3:
            return
        cov = self.samples_table.horizontalHeaderItem(item.column()).text()
        key = self.samples_table.item(item.row(), 0).data(Qt.UserRole)
        self.state.covariates.setdefault(cov, {})[key] = item.text().strip()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _build_stats_page(self):
        layout = QHBoxLayout(self.stats_page)
        left = QWidget()
        left.setMaximumWidth(420)
        form = QFormLayout(left)

        self.chk_freq = QCheckBox("Population frequencies (log odds within parent)")
        self.chk_counts = QCheckBox("Population counts (negative binomial, parent offset)")
        self.chk_markers = QCheckBox("Marker medians (population first)")
        self.chk_stats_parent = QCheckBox("Use a population's stats parent where defined")
        self.chk_exclude_gating = QCheckBox("Leave out each population's gating markers")
        self.chk_exclude_gating.setToolTip(
            "A population's own gate and ancestor gates define it on those channels, "
            "so their medians differ between populations by construction.")
        self.sb_min_events = QSpinBox()
        self.sb_min_events.setRange(1, 1_000_000)
        self.sb_min_events.setToolTip(
            "A population's marker medians are tested only when it has at least this many "
            "events in every sample compared.")
        for w in (self.chk_freq, self.chk_counts, self.chk_markers, self.chk_stats_parent,
                  self.chk_exclude_gating):
            form.addRow(w)
        form.addRow("Min. events for medians:", self.sb_min_events)

        self.cb_mode = QComboBox()
        for key, text in CONTRAST_MODES.items():
            self.cb_mode.addItem(text, key)
        self.cb_reference = QComboBox()
        self.cb_pairing = QComboBox()
        self.list_adjust = QListWidget()
        self.list_adjust.setMaximumHeight(80)
        form.addRow("Comparisons:", self.cb_mode)
        form.addRow("Reference group:", self.cb_reference)
        form.addRow("Pair samples by:", self.cb_pairing)
        form.addRow("Adjust for:", self.list_adjust)

        self.sb_pval = QDoubleSpinBox()
        self.sb_pval.setRange(0.0001, 1.0)
        self.sb_pval.setDecimals(4)
        self.sb_pval.setSingleStep(0.01)
        self.sb_fc = QDoubleSpinBox()
        self.sb_fc.setRange(0.0, 10.0)
        self.sb_fc.setDecimals(2)
        self.sb_fc.setSingleStep(0.1)
        self.sb_fc.setToolTip("|log2 odds ratio| for frequencies, |log2 fold change| for counts.")
        self.sb_marker = QDoubleSpinBox()
        self.sb_marker.setRange(0.0, 1.0)
        self.sb_marker.setDecimals(3)
        self.sb_marker.setSingleStep(0.01)
        self.sb_marker.setToolTip("|difference in median| in transformed axis units.")
        self.chk_treat = QCheckBox("Test against the thresholds (TREAT)")
        self.chk_treat.setToolTip("On: p-values test |effect| > threshold. "
                                  "Off: p-values test effect ≠ 0 and thresholds only filter.")
        self.cb_fdr = QComboBox()
        for key, text in FDR_SCOPES.items():
            self.cb_fdr.addItem(text, key)
        self.chk_exclude_flagged = QCheckBox("Leave out flagged samples")
        form.addRow("FDR ≤", self.sb_pval)
        form.addRow("Frequency / count threshold:", self.sb_fc)
        form.addRow("Marker threshold:", self.sb_marker)
        form.addRow(self.chk_treat)
        form.addRow("FDR correction:", self.cb_fdr)
        form.addRow(self.chk_exclude_flagged)

        self.btn_stats = QPushButton("Run Statistics")
        self.btn_stats.clicked.connect(self._on_run_stats)
        form.addRow(self.btn_stats)
        self.lbl_stats = QLabel("")
        self.lbl_stats.setWordWrap(True)
        form.addRow(self.lbl_stats)
        layout.addWidget(left)

        self.results_tabs = QTabWidget()
        layout.addWidget(self.results_tabs, stretch=1)

        self.chk_freq.toggled.connect(lambda v: setattr(self.state, 'run_freq', bool(v)))
        self.chk_counts.toggled.connect(lambda v: setattr(self.state, 'run_counts', bool(v)))
        self.chk_markers.toggled.connect(lambda v: setattr(self.state, 'run_markers', bool(v)))
        self.chk_stats_parent.toggled.connect(
            lambda v: setattr(self.state, 'use_stats_parent', bool(v)))
        self.chk_exclude_gating.toggled.connect(
            lambda v: setattr(self.state, 'exclude_gating_markers', bool(v)))
        self.sb_min_events.valueChanged.connect(
            lambda v: setattr(self.state, 'min_marker_events', int(v)))
        self.chk_exclude_flagged.toggled.connect(
            lambda v: setattr(self.state, 'exclude_flagged', bool(v)))
        self.cb_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.cb_reference.currentIndexChanged.connect(
            lambda _i: self._filling or setattr(self.state, 'reference_group',
                                                self.cb_reference.currentData() or ''))
        self.cb_pairing.currentIndexChanged.connect(
            lambda _i: self._filling or setattr(self.state, 'pairing',
                                                self.cb_pairing.currentData() or ''))
        self.list_adjust.itemChanged.connect(self._on_adjust_changed)
        for w in (self.sb_pval, self.sb_fc, self.sb_marker):
            w.valueChanged.connect(self._on_thresholds_changed)
        self.chk_treat.toggled.connect(self._on_thresholds_changed)
        self.cb_fdr.currentIndexChanged.connect(self._on_thresholds_changed)

    def _load_stats_controls(self):
        st = self.state
        self._filling = True
        try:
            for w, v in ((self.chk_freq, st.run_freq), (self.chk_counts, st.run_counts),
                         (self.chk_markers, st.run_markers),
                         (self.chk_stats_parent, st.use_stats_parent),
                         (self.chk_exclude_gating, st.exclude_gating_markers),
                         (self.chk_treat, st.use_treat),
                         (self.chk_exclude_flagged, st.exclude_flagged)):
                w.blockSignals(True)
                w.setChecked(bool(v))
                w.blockSignals(False)
            for w, v in ((self.sb_pval, st.pval_threshold), (self.sb_fc, st.fc_threshold),
                         (self.sb_marker, st.marker_threshold)):
                w.blockSignals(True)
                w.setValue(float(v))
                w.blockSignals(False)
            self.sb_min_events.blockSignals(True)
            self.sb_min_events.setValue(int(st.min_marker_events))
            self.sb_min_events.blockSignals(False)
            for cb, v in ((self.cb_mode, st.contrast_mode), (self.cb_fdr, st.fdr_scope)):
                cb.blockSignals(True)
                cb.setCurrentIndex(max(cb.findData(v), 0))
                cb.blockSignals(False)
        finally:
            self._filling = False
        self._populate_stats_choices()

    def _populate_stats_choices(self):
        st = self.state
        self._filling = True
        try:
            self.cb_reference.clear()
            for g in st.group_names:
                self.cb_reference.addItem(g, g)
            if st.reference_group not in st.group_names:
                st.reference_group = st.group_names[0] if st.group_names else ''
            self.cb_reference.setCurrentIndex(max(self.cb_reference.findData(st.reference_group), 0))
            self.cb_reference.setEnabled(st.contrast_mode == 'reference')
            self.cb_pairing.clear()
            self.cb_pairing.addItem("Not paired", '')
            for c in st.covariates:
                self.cb_pairing.addItem(c, c)
            self.cb_pairing.setCurrentIndex(max(self.cb_pairing.findData(st.pairing), 0))
            self.list_adjust.clear()
            for c in st.covariates:
                it = QListWidgetItem(c)
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if c in st.adjust_covariates else Qt.Unchecked)
                self.list_adjust.addItem(it)
        finally:
            self._filling = False

    def _on_mode_changed(self, _i):
        if self._filling:
            return
        self.state.contrast_mode = self.cb_mode.currentData()
        self.cb_reference.setEnabled(self.state.contrast_mode == 'reference')

    def _on_adjust_changed(self, _item):
        if self._filling:
            return
        self.state.adjust_covariates = [
            self.list_adjust.item(i).text() for i in range(self.list_adjust.count())
            if self.list_adjust.item(i).checkState() == Qt.Checked]

    def _on_thresholds_changed(self, *_args):
        if self._filling:
            return
        st = self.state
        st.pval_threshold = float(self.sb_pval.value())
        st.fc_threshold = float(self.sb_fc.value())
        st.marker_threshold = float(self.sb_marker.value())
        st.use_treat = bool(self.chk_treat.isChecked())
        st.fdr_scope = self.cb_fdr.currentData()
        if st.stats:
            try:
                st.stats = ag_results.reapply_thresholds(
                    st.stats, st.pval_threshold, st.fc_threshold, st.marker_threshold,
                    st.fdr_scope, st.use_treat)
            except Exception as exc:
                self.lbl_stats.setText(f"Could not apply thresholds: {exc}")
                return
            self._show_results_tables()
            self._draw_figures()

    def _stats_samples(self) -> list[str]:
        st = self.state
        keys = [k for k in st.gated_samples() if st.sample_groups.get(k)]
        if st.exclude_flagged:
            keys = [k for k in keys if not st.flags(k)]
        return keys

    def _on_run_stats(self):
        st = self.state
        if self._stats_worker is not None:
            return
        if not (st.run_freq or st.run_counts or st.run_markers):
            QMessageBox.information(self, "Nothing to test", "Tick at least one kind of test.")
            return
        samples = self._stats_samples()
        groups = {k: st.sample_groups[k] for k in samples}
        pairing = None
        if st.pairing:
            vals = st.covariates.get(st.pairing) or {}
            missing = [_sample_label(k) for k in samples if not str(vals.get(k, '')).strip()]
            if missing:
                QMessageBox.warning(self, "Pairing incomplete",
                                    f"No '{st.pairing}' value for: " + ", ".join(missing[:10]))
                return
            pairing = {k: vals[k] for k in samples}
        covariates = None
        adjust = [c for c in st.adjust_covariates if c in st.covariates and c != st.pairing]
        if adjust:
            covariates = pd.DataFrame(
                {c: [str((st.covariates.get(c) or {}).get(k, '')) for k in samples] for c in adjust},
                index=samples)
        kwargs = dict(
            group_order=list(st.group_names), contrast_mode=st.contrast_mode,
            reference=st.reference_group, pairing=pairing, covariates=covariates,
            pval_threshold=st.pval_threshold, fc_threshold=st.fc_threshold,
            marker_threshold=st.marker_threshold, use_treat=st.use_treat,
            fdr_scope=st.fdr_scope, run_freq=st.run_freq, run_counts=st.run_counts,
            run_markers=st.run_markers, use_stats_parent=st.use_stats_parent,
            exclude_gating_markers=st.exclude_gating_markers,
            min_marker_events=st.min_marker_events,
        )
        summaries = deepcopy({k: st.summaries[k] for k in samples})
        gate_defs = deepcopy(st.gate_defs())

        def _run():
            return ag_results.run_statistics(summaries, samples, groups, gate_defs, **kwargs)

        self.btn_stats.setEnabled(False)
        self.lbl_stats.setText(f"Testing {len(samples)} sample(s) …")
        self._stats_worker = _CallWorker(_run, parent=self)
        self._stats_worker.finished.connect(self._on_stats_finished)
        self._stats_worker.start()

    def _on_stats_finished(self, ok: bool, error: str, result: object):
        self._stats_worker = None
        self.btn_stats.setEnabled(True)
        if not ok:
            self.lbl_stats.setText(f"<span style='color:#b00020'>{error}</span>")
            return
        self.state.stats = result
        n_sig = sum(int(result[k]['significant'].sum()) for k in ('freq', 'counts', 'markers')
                    if result.get(k) is not None)
        text = (f"{len(result['samples'])} samples, {len(result['comparisons'])} comparison(s), "
                f"{n_sig} significant result(s).")
        if result.get('notes'):
            text += "<br>" + "<br>".join(result['notes'])
        self.lbl_stats.setText(text)
        _status(self.bus, "Statistics complete.")
        self._populate_comparisons()
        self._show_results_tables()
        self._draw_figures()
        self._populate_report_items()

    def _populate_comparisons(self):
        comps = (self.state.stats or {}).get('comparisons') or []
        current = self.cb_comparison.currentData()
        self.cb_comparison.blockSignals(True)
        self.cb_comparison.clear()
        for base, other in comps:
            self.cb_comparison.addItem(_comparison_label(base, other), (base, other))
        i = self.cb_comparison.findData(current) if current else 0
        self.cb_comparison.setCurrentIndex(max(i, 0))
        self.cb_comparison.blockSignals(False)

    def _current_comparison(self):
        return self.cb_comparison.currentData()

    def _on_comparison_changed(self, _i):
        self._show_results_tables()
        self._draw_figures()

    def _show_results_tables(self):
        self.results_tabs.clear()
        comp = self._current_comparison()
        stats = self.state.stats or {}
        if comp is None:
            return
        label = _comparison_label(*comp)
        labels = _population_labels(self.state.gate_defs())
        antigens = _antigen_map(self.controller)
        fdr_col = 'adj.P.Val.global' if self.state.fdr_scope == 'global' else 'adj.P.Val'

        def _fmt(value) -> str:
            return f"{value:.3g}" if value is not None and pd.notna(value) else ''

        for key, title in (('freq', 'Frequencies'), ('counts', 'Counts'), ('markers', 'Markers')):
            view = _comparison_view(stats.get(key), label)
            if view is None:
                continue
            view = view.sort_values('P.Value', kind='stable')
            rows = []
            for _i, r in view.iterrows():
                pop_key = r['cluster_id'] if key == 'markers' else r['feature']
                row = {'Population': labels.get(pop_key, pop_key),
                       'Effect': _fmt(r['logFC']),
                       'P': _fmt(r['P.Value']),
                       'FDR': _fmt(r.get(fdr_col)),
                       'Significant': 'yes' if r['significant'] else ''}
                if key == 'markers':
                    row['Marker'] = antigens.get(r['channel'], r['channel'])
                    row['Stage-wise FDR'] = _fmt(r.get('stagewise.adj.P.Val'))
                rows.append(row)
            headers = (['Population', 'Marker', 'Effect', 'P', 'FDR', 'Stage-wise FDR', 'Significant']
                       if key == 'markers' else ['Population', 'Effect', 'P', 'FDR', 'Significant'])
            table = CopyableTableWidget(rows, headers)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            self.results_tabs.addTab(table, title)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def _draw_figures(self):
        comp = self._current_comparison()
        if not self.state.stats or comp is None:
            self.figures_tabs.clear()
            return
        if self._figure_worker is not None:
            self._figure_pending = True
            return
        st = self.state
        stats = st.stats
        args = (comp[0], comp[1], st.pval_threshold, st.fc_threshold, st.marker_threshold,
                st.fdr_scope, _is_dark_palette(), _antigen_map(self.controller),
                _population_labels(st.gate_defs()))

        def _build():
            return build_result_figures(stats, *args)

        self._figure_worker = _CallWorker(_build, parent=self)
        self._figure_worker.finished.connect(self._on_figures_built)
        self._figure_worker.start()

    def _on_figures_built(self, ok: bool, error: str, payload: object):
        self._figure_worker = None
        if self._figure_pending:
            self._figure_pending = False
            self._draw_figures()
            return
        self.figures_tabs.clear()
        self._volcano_tables = {}
        if not ok:
            self.figures_tabs.addTab(QLabel(f"Figures failed: {error}"), "Error")
            return
        exp_dir = getattr(self.controller, 'experiment_dir', '')
        for key, item in payload.items():
            if item['fig'] is None:
                lbl = QLabel(f"{item['title']}: {item['error']}")
                lbl.setStyleSheet("color: #b00020;")
                self.figures_tabs.addTab(lbl, item['title'])
                continue
            widget = ExportablePlotWidget(figure=item['fig'], title=item['title'].replace(' ', '_'),
                                          experiment_dir=exp_dir, parent=self)
            widget.delete_button.setVisible(False)
            fig = item['fig']
            handler = getattr(fig, '_hover_handler', None)
            if handler is not None:
                widget.canvas.mpl_connect('motion_notify_event', handler)
            if item.get('results') is not None:
                self._volcano_tables[id(widget.canvas)] = item['results']
                widget.canvas.mpl_connect('button_press_event', self._on_volcano_click)
                widget.canvas.setToolTip("Click a point to open that population in "
                                         "the Populations browser.")
            scroll = QScrollArea()
            scroll.setWidget(widget)
            self.figures_tabs.addTab(scroll, item['title'])

    def _on_volcano_click(self, event):
        """Open the clicked volcano point's population in the browser."""
        from honeychrome.view_components.differential_plots import significance_column
        ax = event.inaxes
        if ax is None or event.x is None:
            return
        table = self._volcano_tables.get(id(event.canvas))
        if table is None or table.empty:
            return
        col, _lbl = significance_column(table, self.state.fdr_scope)
        xy = np.column_stack([table['logFC'].values.astype(float),
                              -np.log10(np.maximum(table[col].values.astype(float), 1e-300))])
        ok = np.isfinite(xy).all(axis=1)
        if not ok.any():
            return
        px = ax.transData.transform(xy[ok])
        d = np.hypot(px[:, 0] - event.x, px[:, 1] - event.y)
        i = int(np.argmin(d))
        if d[i] > 12:
            return
        row = table[ok].iloc[i]
        pop_key = row.get('cluster_id') if 'cluster_id' in table.columns else row['feature']
        self.show_population(str(pop_key))

    # ------------------------------------------------------------------
    # Population browser
    # ------------------------------------------------------------------

    def _build_browser_page(self):
        layout = QHBoxLayout(self.browser_page)
        left = QWidget()
        left.setFixedWidth(220)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("<b>Populations</b>"))
        self.pop_list = QListWidget()
        self.pop_list.currentItemChanged.connect(lambda cur, _prev: self._draw_population(cur))
        ll.addWidget(self.pop_list)
        layout.addWidget(left)
        self.browser_scroll = QScrollArea()
        self.browser_scroll.setWidgetResizable(True)
        self.browser_canvas = QWidget()
        self.browser_grid = QGridLayout(self.browser_canvas)
        self.browser_grid.setSpacing(4)
        self.browser_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.browser_scroll.setWidget(self.browser_canvas)
        layout.addWidget(self.browser_scroll, stretch=1)

    def _populate_population_list(self):
        current = self.pop_list.currentItem().data(Qt.UserRole) if self.pop_list.currentItem() else None
        self.pop_list.blockSignals(True)
        self.pop_list.clear()
        labels = _population_labels(self.state.gate_defs())
        for g in ag_core.ordered_gate_defs(self.state.gate_defs()):
            for pop in (g.get('populations') or {}):
                key = ag_core.population_key(g['gate_name'], pop)
                it = QListWidgetItem(labels.get(key, key))
                it.setData(Qt.UserRole, key)
                self.pop_list.addItem(it)
                if key == current:
                    self.pop_list.setCurrentItem(it)
        self.pop_list.blockSignals(False)
        self._draw_population(self.pop_list.currentItem())

    def show_population(self, pop_key: str):
        for i in range(self.pop_list.count()):
            if self.pop_list.item(i).data(Qt.UserRole) == pop_key:
                self.pop_list.setCurrentRow(i)
                break
        self.sub_tabs.setCurrentWidget(self.browser_page)

    def _draw_population(self, item):
        _clear_layout(self.browser_grid)
        st = self.state
        if item is None:
            return
        pop_key = item.data(Qt.UserRole)
        gate_name, pop = ag_core.split_population_key(pop_key)
        by_name = ag_core.gates_by_name(st.gate_defs())
        gdef = by_name.get(gate_name)
        if gdef is None:
            return
        if not st.display:
            note = QLabel("Plots are available for samples gated in this session; "
                          "run gating again to see them.")
            note.setWordWrap(True)
            self.browser_grid.addWidget(note, 0, 0)
            return
        eff = ag_core.effective_gate_def(gdef, by_name)
        transforms = self.controller.unmixed_transformations or {}
        labels = build_display_label_map(
            _experiment_channels(self.controller),
            self.controller.experiment.process.get('spectral_model') or [])
        order = {g: i for i, g in enumerate(st.group_names)}
        keys = sorted([k for k in st.gated_samples() if k in st.display],
                      key=lambda k: (order.get(st.sample_groups.get(k, ''), len(order)),
                                     _sample_label(k)))
        cols = 4
        for i, key in enumerate(keys):
            disp = st.display[key]
            summ = st.summaries.get(key) or {}
            row = next((r for r in summ.get('populations', []) if r['key'] == pop_key), None)
            fractions = {r['population']: r['fraction_of_parent']
                         for r in summ.get('populations', []) if r['gate'] == gate_name}
            group = st.sample_groups.get(key, '')
            title = f"<b>{_sample_label(key)}</b>" + (f" <small>[{group}]</small>" if group else '')
            status = (summ.get('status') or {}).get(gate_name) or {}
            footer = ''
            if row is not None:
                footer = f"n = {row['n_events']:,} of {row['n_parent']:,}"
            if status.get('flag'):
                footer += f" — {status.get('message') or status['flag']}"
            tile = _make_gate_tile(
                eff, (summ.get('boundaries') or {}).get(gate_name, {}), disp['events'],
                disp['channels'], disp['parent_masks'].get(gate_name), transforms, labels,
                title, fractions=fractions, highlight=pop, footer=footer)
            r, c = divmod(i, cols)
            self.browser_grid.addWidget(tile, r, c)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _build_export_page(self):
        layout = QVBoxLayout(self.export_page)
        blurb = QLabel(
            "<b>Generate Report</b> writes a folder in the experiment's "
            "<i>Automated_Gating_Reports</i> directory with "
            "<i>population_statistics.csv</i> (one row per sample and population: counts, "
            "parent counts and fractions), <i>activation_statistics.csv</i> (one row per "
            "sample, population and marker: median and quartiles in transformed units) and "
            "<i>settings.txt</i> (model file and its SHA-256, gate application, AF "
            "assignments and statistics settings), plus a PNG and/or CSV for each ticked "
            "item below and one PDF containing all of them.")
        blurb.setWordWrap(True)
        layout.addWidget(blurb)
        row = QHBoxLayout()
        for text, on in (("All", True), ("None", False)):
            b = QPushButton(text)
            b.setFixedWidth(60)
            b.clicked.connect(lambda _c=False, v=on: self._set_report_checks(v))
            row.addWidget(b)
        b_refresh = QPushButton("Refresh Items")
        b_refresh.clicked.connect(self._populate_report_items)
        row.addWidget(b_refresh)
        row.addStretch()
        layout.addLayout(row)
        self.report_list = QListWidget()
        self.report_list.setSelectionMode(QAbstractItemView.NoSelection)
        layout.addWidget(self.report_list, stretch=1)
        self.btn_report = QPushButton("Generate Report")
        self.btn_report.clicked.connect(self._on_generate_report)
        layout.addWidget(self.btn_report)
        self.lbl_export = QLabel("")
        self.lbl_export.setWordWrap(True)
        layout.addWidget(self.lbl_export)

    def _report_items(self) -> list:
        """Report items built from the current results (fresh each call)."""
        st = self.state
        samples = st.gated_samples()
        return ag_report.build_report_items(
            gate_defs=st.gate_defs(), reference_boundaries=st.reference_boundaries(),
            modes=st.modes, summaries=st.summaries, samples=samples, load_info=st.load_info,
            display=st.display, groups=st.sample_groups,
            flags={k: st.flags(k) for k in samples}, antigens=_antigen_map(self.controller),
            transforms=self.controller.unmixed_transformations or {},
            axis_labels=build_display_label_map(
                _experiment_channels(self.controller),
                self.controller.experiment.process.get('spectral_model') or []),
            stats=st.stats, pval_threshold=st.pval_threshold, fc_threshold=st.fc_threshold,
            marker_threshold=st.marker_threshold, fdr_scope=st.fdr_scope,
            build_figures=build_result_figures,
        )

    def _populate_report_items(self):
        unchecked = {self.report_list.item(i).data(Qt.UserRole)
                     for i in range(self.report_list.count())
                     if self.report_list.item(i).checkState() == Qt.Unchecked}
        self.report_list.clear()
        items = self._report_items()
        if not items:
            placeholder = QListWidgetItem("(nothing to report yet — gate some samples)")
            placeholder.setFlags(Qt.ItemIsEnabled)
            self.report_list.addItem(placeholder)
        for item in items:
            it = QListWidgetItem(f"{item.tab} — {item.label}")
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked if item.key in unchecked else Qt.Checked)
            it.setData(Qt.UserRole, item.key)
            self.report_list.addItem(it)
        self.btn_report.setEnabled(bool(self.state.summaries))

    def _set_report_checks(self, on: bool):
        for i in range(self.report_list.count()):
            it = self.report_list.item(i)
            if it.flags() & Qt.ItemIsUserCheckable:
                it.setCheckState(Qt.Checked if on else Qt.Unchecked)

    def _on_generate_report(self):
        from PySide6.QtWidgets import QApplication, QProgressDialog
        from matplotlib.backends.backend_pdf import PdfPages
        import drc_report

        st = self.state
        if not st.summaries:
            QMessageBox.information(self, "Nothing to report", "Gate some samples first.")
            return
        checked = {self.report_list.item(i).data(Qt.UserRole)
                   for i in range(self.report_list.count())
                   if self.report_list.item(i).checkState() == Qt.Checked}
        items = [it for it in self._report_items() if it.key in checked]
        samples = st.gated_samples()
        stem = drc_report.sanitize_filename(Path(st.model_path).stem or 'automated_gating')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = Path(self.controller.experiment_dir) / 'Automated_Gating_Reports' / f"{stem}_{timestamp}"

        progress = QProgressDialog("Generating report…", None, 0, len(items) + 1, self)
        progress.setWindowTitle("Generate Report")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.setValue(0)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            report_dir.mkdir(parents=True, exist_ok=True)
            ag_results.population_export(st.summaries, samples, st.sample_groups).to_csv(
                report_dir / 'population_statistics.csv', index=False)
            ag_results.marker_export(st.summaries, samples, _antigen_map(self.controller),
                                     st.sample_groups).to_csv(
                report_dir / 'activation_statistics.csv', index=False)
            with PdfPages(str(report_dir / f"{stem}_report.pdf")) as pdf:
                drc_report.add_title_page(pdf, "Honeychrome Automated Gating Report", [
                    f"Experiment: {self.controller.experiment_dir}",
                    f"Model: {Path(st.model_path).name}",
                    f"Samples gated: {len(samples)}",
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"Items included: {len(items)}",
                ])
                done = 0
                for section in ag_report.SECTIONS:
                    section_items = [it for it in items if it.tab == section]
                    if not section_items:
                        continue
                    drc_report.add_section_divider(pdf, section)
                    subfolder = report_dir / drc_report.sanitize_filename(section)
                    subfolder.mkdir(parents=True, exist_ok=True)
                    for it in section_items:
                        progress.setLabelText(f"Adding {it.label} …")
                        drc_report.export_report_item(it, subfolder, pdf)
                        done += 1
                        progress.setValue(done)
                        QApplication.processEvents()
            progress.setLabelText("Writing settings document …")
            (report_dir / 'settings.txt').write_text(self._settings_text(samples), encoding='utf-8')
            progress.setValue(len(items) + 1)
        except Exception as exc:
            log.exception("report generation failed")
            QMessageBox.critical(self, "Report failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
            progress.close()
        self.lbl_export.setText(f"Report written to {report_dir}")
        _status(self.bus, f"Report written to {report_dir}")

    def _settings_text(self, samples: list[str]) -> str:
        st = self.state
        lines = [
            f"Exported: {datetime.now().isoformat(timespec='seconds')}",
            f"Honeychrome: {getattr(honeychrome, '__version__', '')}",
            f"Model: {st.model_path}",
            f"Model SHA-256: {_file_sha256(st.model_path)}",
            f"Model SHA-256 when gated: {st.results_model_sha}",
            "",
            "Gates:",
        ]
        for g in ag_core.ordered_gate_defs(st.gate_defs()):
            lines.append(f"  {g['gate_name']}: {MODE_LABELS.get(st.modes.get(g['gate_name'], 'fixed'))}")
        lines += [
            f"Drift limit: {st.drift_limit} (transformed units); when exceeded: "
            f"{DRIFT_ACTIONS.get(st.drift_action)}",
            "Channel alignment: " + ", ".join(f"{m} -> {e}" for m, e in st.alignment.items()),
            f"Time QC available: {_time_qc_available()}",
            "",
            "Samples (AF profiles; events in file; events gated; group; flags):",
        ]
        for k in samples:
            info = st.load_info.get(k) or {}
            lines.append(f"  {k}; {', '.join(info.get('af_profiles') or []) or 'none'}; "
                         f"{info.get('n_events_file')}; {info.get('n_events_kept')}; "
                         f"{st.sample_groups.get(k, '')}; {' | '.join(st.flags(k))}")
        if st.stats:
            lines += [
                "",
                "Statistics:",
                "  Comparisons: " + ", ".join(_comparison_label(b, o) for b, o in st.stats['comparisons']),
                f"  Contrast mode: {st.contrast_mode}; reference: {st.reference_group}",
                f"  Pairing: {st.pairing or 'none'}; adjusted for: {', '.join(st.adjust_covariates) or 'none'}",
                f"  FDR <= {st.pval_threshold} ({st.fdr_scope}); TREAT: {st.use_treat}",
                f"  Frequency/count threshold: {st.fc_threshold}; marker threshold: {st.marker_threshold}",
                f"  Frequencies relative to stats parent where defined: {st.use_stats_parent}",
                f"  Gating markers left out of marker tests: {st.exclude_gating_markers}; "
                f"min. events: {st.min_marker_events}",
                f"  Flagged samples left out: {st.exclude_flagged}",
            ]
            lines += [f"  Note: {n}" for n in st.stats.get('notes') or []]
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------

    def refresh(self):
        self._populate_groups()
        self._load_stats_controls()
        self._populate_comparisons()
        self._show_results_tables()
        if not self.state.stats:
            self.figures_tabs.clear()
            self.lbl_stats.setText("")
        self._populate_population_list()
        self._populate_report_items()


# ---------------------------------------------------------------------------
# PluginWidget
# ---------------------------------------------------------------------------

class PluginWidget(QWidget):
    """Top-level widget for the Automated Gating plugin.

    Owns GatingRunState and the Setup, Run and Results tabs.

    Persistence: settings (model file, alignment, gate modes, sample
    selection, groups, covariates and statistics options) are kept in
    QSettings per experiment as one JSON string. Gated results (counts,
    medians and per-gate status, not the display events) are kept in
    RESULTS_FILE in the experiment folder, so they survive a restart.
    """

    def __init__(self, bus=None, controller=None, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.controller = controller
        _ensure_qt_imports()
        _suppress_third_party_warnings()

        self.state = GatingRunState()
        self._qsettings = QSettings('honeychrome', 'plugin_automated_gating')
        self._plugin_was_active = False
        self._loaded_experiment_key = None
        self._loaded_experiment_dir = None

        self.label_disabled = QLabel(f"{plugin_name}: set up the spectral model first.")
        self.label_disabled.setWordWrap(True)
        self.label_disabled.setStyleSheet(
            "QLabel { color: #856404; background: #fff3cd; "
            "border: 1px solid #ffc107; border-radius: 4px; padding: 8px; }")

        self.content_widget = QWidget()
        content_layout = QVBoxLayout(self.content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self.content_widget)
        overall = QVBoxLayout(self)
        overall.setContentsMargins(6, 6, 6, 6)
        overall.addWidget(self.label_disabled)
        overall.addWidget(scroll)

        self.inner_tabs = QTabWidget()
        self.inner_tabs.setDocumentMode(True)
        self.setup_tab = SetupTab(self.state, bus, controller)
        self.run_tab = RunTab(self.state, bus, controller)
        self.results_tab = ResultsTab(self.state, bus, controller)
        self.inner_tabs.addTab(self.setup_tab, "Setup")
        self.inner_tabs.addTab(self.run_tab, "Run")
        self.inner_tabs.addTab(self.results_tab, "Results")
        self.inner_tabs.currentChanged.connect(self._on_inner_tab_changed)
        content_layout.addWidget(self.inner_tabs)

        self.setup_tab.model_ready.connect(self._on_model_ready)
        self.run_tab.gating_finished.connect(self._on_gating_finished)

        self.content_widget.setVisible(False)
        self.bus.modeChangeRequested.connect(self._on_mode_change)
        QTimer.singleShot(0, self._deferred_initial_refresh)

    # ------------------------------------------------------------------

    def _on_mode_change(self, mode: str):
        """Show the content when this tab is active and unmixing is set up.
        Leaving the tab saves; persisted state is read back only when the
        experiment has changed."""
        if mode != plugin_name:
            if self._plugin_was_active:
                self.save_state()
                self._plugin_was_active = False
            return
        self._plugin_was_active = True
        self._show_if_ready()

    def _deferred_initial_refresh(self):
        if getattr(self.controller, 'current_mode', None) == plugin_name:
            self._plugin_was_active = True
            self._show_if_ready()

    def _show_if_ready(self):
        if self.controller.experiment.process.get('unmixing_matrix') is not None:
            self.label_disabled.setVisible(False)
            self.content_widget.setVisible(True)
            self._ensure_state_loaded()
            self._refresh_active_tab()
        else:
            self.label_disabled.setVisible(True)
            self.content_widget.setVisible(False)

    def _ensure_state_loaded(self):
        current_key = self._settings_key()
        if current_key == self._loaded_experiment_key:
            return
        if self._loaded_experiment_key is not None:
            self.save_state()
        # The inner tabs share self.state by reference, so reset it in place.
        fresh = GatingRunState()
        for f in dataclass_fields(GatingRunState):
            setattr(self.state, f.name, getattr(fresh, f.name))
        self._loaded_experiment_key = current_key
        self._loaded_experiment_dir = Path(self.controller.experiment_dir)
        self.load_state()
        self.setup_tab.refresh()

    def _on_inner_tab_changed(self, index: int):
        self.save_state()
        self._refresh_tab_at(index)

    def _refresh_active_tab(self):
        self._refresh_tab_at(self.inner_tabs.currentIndex())

    def _refresh_tab_at(self, index: int):
        tabs = [self.setup_tab, self.run_tab, self.results_tab]
        if 0 <= index < len(tabs):
            tabs[index].refresh()

    def _on_model_ready(self):
        self.save_state()
        self.inner_tabs.setCurrentWidget(self.run_tab)

    def _on_gating_finished(self):
        self._write_results()
        self.save_state()

    def progress_message(self, msg: str):
        _status(self.bus, msg)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _settings_key(self) -> str:
        try:
            exp_dir = str(self.controller.experiment_dir)
        except Exception:
            exp_dir = 'unknown'
        return exp_dir.replace('\\', '/').replace(':', '_').replace(' ', '_')

    def _controller_shows_loaded_experiment(self) -> bool:
        if self._loaded_experiment_dir is None:
            return False
        try:
            return Path(self.controller.experiment_dir) == self._loaded_experiment_dir
        except Exception:
            return False

    def save_state(self):
        """Persist settings for the experiment the state was loaded from."""
        if self._loaded_experiment_key is None:
            return
        payload = {name: getattr(self.state, name) for name in _PERSISTED_FIELDS}
        s = self._qsettings
        s.beginGroup(self._loaded_experiment_key)
        try:
            s.setValue('state_json', json.dumps(ag_core.to_jsonable(payload)))
        finally:
            s.endGroup()

    def load_state(self):
        """Restore settings, the model and cached results for the loaded
        experiment."""
        s = self._qsettings
        s.beginGroup(self._loaded_experiment_key or self._settings_key())
        try:
            raw = s.value('state_json', '')
        finally:
            s.endGroup()
        try:
            payload = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            payload = {}
        defaults = GatingRunState()
        for name in _PERSISTED_FIELDS:
            if name not in payload:
                continue
            value = payload[name]
            expected = type(getattr(defaults, name))
            if expected is float and isinstance(value, (int, float)):
                value = float(value)
            if isinstance(value, expected):
                setattr(self.state, name, value)
        path = self.state.model_path
        if path:
            if Path(path).exists():
                self.setup_tab.load_model(path, alignment=dict(self.state.alignment),
                                          confirmed=bool(payload.get('alignment_confirmed')))
            else:
                self.state.model_path = ''
                self.state.alignment_confirmed = False
                self.progress_message(f"The model file {path} is no longer available.")
        self._load_results()

    def _results_path(self) -> Path | None:
        if self._loaded_experiment_dir is None:
            return None
        return self._loaded_experiment_dir / RESULTS_FILE

    def _write_results(self):
        """Write the gated results (not the display events) to RESULTS_FILE."""
        if not self._controller_shows_loaded_experiment():
            return
        path = self._results_path()
        st = self.state
        payload = {
            'format_version': 1,
            'model_path': st.model_path,
            'model_sha256': st.results_model_sha,
            'marker_channels': st.marker_channels,
            'run_samples': st.run_samples,
            'summaries': st.summaries,
            'load_info': st.load_info,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + '.tmp')
            tmp.write_text(json.dumps(ag_core.to_jsonable(payload)), encoding='utf-8')
            _os.replace(tmp, path)
        except Exception:
            log.exception("could not write gating results to %s", path)

    def _load_results(self):
        path = self._results_path()
        if path is None or not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            log.exception("could not read gating results %s", path)
            return
        st = self.state
        st.summaries = dict(payload.get('summaries') or {})
        st.load_info = dict(payload.get('load_info') or {})
        st.marker_channels = list(payload.get('marker_channels') or [])
        st.results_model_sha = str(payload.get('model_sha256') or '')
        if payload.get('run_samples'):
            st.run_samples = list(payload['run_samples'])
        if st.summaries and st.results_model_sha and st.model_path \
                and _file_sha256(st.model_path) != st.results_model_sha:
            self.progress_message("The saved gating results were made with a different version "
                                  "of the model file; run gating again to update them.")
