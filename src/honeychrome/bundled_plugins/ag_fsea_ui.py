"""
ag_fsea_ui.py — the Enrichment page of the Automated Gating plugin
=================================================================
``EnrichmentTab``: Flow Set Enrichment Analysis of the gated results
(``ag_fsea``) against an immune state library (``ag_fsea_library``).

Sections (one open at a time):
  • Library  — the bundled library or a user CSV, species, its states.
  • Features and cell types — the abundance anchor, the feature types, the
               marker readout; Check matching builds the features and shows
               the population → cell type table (editable) and how each
               library row matched.
  • Run      — ranking statistic, permutations, seed, minimum set size,
               and the coherence options; runs in a background worker with
               progress and Cancel.
  • Results  — states table, dot plot, and the running-sum and member
               figures of one state; with coherence on, the coherence
               table and the Coherence, State network and State pair
               figures (``ag_coherence``).
  • Export   — CSV files of the last run.

The page takes its samples, groups, pairing and covariates from the
Results page (the same samples and design as the statistics), and its
gated results from the shared GatingRunState.

Qt-dependent; pyqtgraph-free. The file name does not end in ``_tab.py``,
so the plugin loader does not treat it as a tab.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView,
    QLabel,
    QMessageBox, QProgressBar, QPushButton, QSpinBox, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

import ag_coherence
import ag_core
import ag_fsea
import ag_fsea_library as fl
import ag_help_texts
from ag_widgets import add_concertina_page, make_concertina

log = ag_core.get_logger(__name__)

FEATURE_TYPE_LABELS = {
    'positivity': "Positivity (fraction of a population positive for a marker)",
    'subset': "Subsets (population within its parent)",
    'abundance': "Abundance (population within the anchor)",
    'marker': "Marker medians (phenotype populations)",
}
READOUT_LABELS = {'positivity': 'Fraction positive, else median',
                  'median': 'Median, else fraction positive'}
METRIC_LABELS = {'signed_log10p': 'sign(logFC) · −log10 p', 'moderated_t': 'Moderated t'}
_METHOD_COLOURS = {'override': '#cfe2ff', 'name': '#d4edda', 'none': None,
                   'name (lineage differs)': '#fff3cd'}
_STATUS_COLOURS = {'matched': '#d4edda', 'no feature': '#fff3cd', 'no population': '#f8d7da'}
# Seconds per permutation and feature (two-core cloud machine; group-only designs).
_SECONDS_PER_PERM_FEATURE = 7e-7
_SECONDS_PER_PERM_ELEMENT = 3e-8


def _item(text, colour: str | None = None, data=None) -> QTableWidgetItem:
    it = QTableWidgetItem(str(text))
    it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
    if colour:
        it.setBackground(QColor(colour))
        it.setForeground(QColor('#000000'))
    if data is not None:
        it.setData(Qt.UserRole, data)
    return it


def _fmt(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return '' if f != f else f"{f:.3g}"


class _FseaWorker(QThread):
    """Run fn(progress, stop_check) off the main thread.

    progress(fraction, message); finished(ok, error, result); a cancelled
    run finishes with ok False and error 'cancelled'.
    """

    progress = Signal(float, str)
    finished = Signal(bool, str, object)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn
        self._stop = False

    def cancel(self):
        self._stop = True

    def run(self):
        try:
            result = self._fn(lambda f, m: self.progress.emit(f, m), lambda: self._stop)
        except ag_fsea.FseaCancelled:
            self.finished.emit(False, 'cancelled', None)
            return
        except Exception as exc:
            if not isinstance(exc, ag_fsea.FseaBlocked):
                log.exception("enrichment failed")
            self.finished.emit(False, str(exc), None)
            return
        self.finished.emit(True, '', result)


def input_fingerprint(state, samples: list[str], design: dict, library_sha: str,
                      trend: dict | None = None) -> str:
    """Hash of everything an enrichment run depends on (with *trend*, the
    trend inputs of ``EnrichmentTab``'s trend_fn)."""
    payload = {
        'trend': ({k: v for k, v in trend.items() if k not in ('design', 'patterns')}
                  if trend else None),
        'gating': state.run_fingerprint, 'samples': samples, 'groups': design['groups'],
        'pairing': design['pairing'],
        'covariates': (design['covariates'].to_dict() if design['covariates'] is not None
                       else None),
        'numeric': design['numeric'], 'library': library_sha,
        'settings': {k: getattr(state, k) for k in (
            'contrast_mode', 'reference_group', 'use_stats_parent', 'exclude_gating_markers',
            'min_marker_events', 'min_parent_events', 'fsea_species', 'fsea_anchor', 'fsea_feature_types',
            'fsea_marker_readout', 'fsea_rank_metric', 'fsea_n_perm', 'fsea_seed',
            'fsea_min_set_size', 'fsea_overrides')},
    }
    return hashlib.sha256(json.dumps(ag_core.to_jsonable(payload), sort_keys=True, default=str)
                          .encode('utf-8')).hexdigest()


class EnrichmentTab(QWidget):
    """Flow Set Enrichment Analysis page.

    samples_fn()        : the samples the statistics use (grouped, gated).
    design_fn(samples)  : {'groups', 'pairing', 'covariates', 'numeric',
                          'missing_pairing'} for those samples.
    antigens_fn()       : {channel: antigen} of the experiment.
    experiment_dir_fn() : the experiment folder (for exports), or ''.
    trend_fn()          : the inputs of the trend test set on the Results page
                          ({'samples', 'time', 'variable', 'unit', 'scale',
                          'shape', 'k', 'interaction', 'baseline', 'groups',
                          'design', 'patterns'}), or None without a trend
                          variable. The trend is then offered as a test beside
                          the group comparisons: which immune states change
                          with the variable.

    Group results are kept in state.fsea and the trend result in
    state.fsea_trend; the Test chooser at the top picks which is shown and
    which kind Run runs.
    """

    enrichment_finished = Signal()

    def __init__(self, state, bus, controller, samples_fn, design_fn, antigens_fn,
                 experiment_dir_fn, trend_fn=None, parent=None):
        super().__init__(parent)
        self._trend_fn = trend_fn
        self.state = state
        self.bus = bus
        self.controller = controller
        self._samples_fn = samples_fn
        self._design_fn = design_fn
        self._antigens_fn = antigens_fn
        self._experiment_dir_fn = experiment_dir_fn
        self._worker: _FseaWorker | None = None
        self._preview_worker: _FseaWorker | None = None
        self._preview: dict | None = None
        self._library = None
        self._library_sha = ''
        self._library_messages: list[str] = []
        self._filling = False
        self._build_ui()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        from honeychrome.view_components.help_toggle_widget import HelpToggleWidget

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)
        outer.addWidget(HelpToggleWidget(text=ag_help_texts.AG_ENRICHMENT, scrolling=True))
        row = QHBoxLayout()
        row.addWidget(QLabel("<b>Test:</b>"))
        self.cb_comparison = QComboBox()
        self.cb_comparison.setToolTip(
            "What the enrichment tests: a comparison between groups, or the trend with a "
            "numeric variable set on the Results page. Run runs the chosen kind (all group "
            "comparisons together, or the trend); group and trend results are kept side by "
            "side.")
        self.cb_comparison.currentIndexChanged.connect(self._on_test_changed)
        row.addWidget(self.cb_comparison, stretch=1)
        outer.addLayout(row)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        outer.addWidget(self.lbl_status)
        self.toolbox = make_concertina()
        outer.addWidget(self.toolbox, stretch=1)
        for title, builder in (("Library", self._build_library_page),
                               ("Features and cell types", self._build_features_page),
                               ("Run", self._build_run_page),
                               ("Results", self._build_results_page),
                               ("Export", self._build_export_page)):
            page = QWidget()
            builder(page)
            add_concertina_page(self.toolbox, page, title)

    def _build_library_page(self, page):
        layout = QVBoxLayout(page)
        row = QHBoxLayout()
        self.lbl_library = QLabel("")
        self.lbl_library.setWordWrap(True)
        row.addWidget(self.lbl_library, stretch=1)
        b_browse = QPushButton("Choose library…")
        b_browse.clicked.connect(self._on_choose_library)
        b_bundled = QPushButton("Use the bundled library")
        b_bundled.clicked.connect(self._on_bundled_library)
        row.addWidget(b_browse)
        row.addWidget(b_bundled)
        layout.addLayout(row)
        form = QFormLayout()
        self.cb_species = QComboBox()
        for sp in fl.SPECIES:
            self.cb_species.addItem(sp, sp)
        self.cb_species.currentIndexChanged.connect(self._on_species_changed)
        form.addRow("Species:", self.cb_species)
        layout.addLayout(form)
        self.lbl_library_messages = QLabel("")
        self.lbl_library_messages.setWordWrap(True)
        self.lbl_library_messages.setStyleSheet("color: #856404;")
        layout.addWidget(self.lbl_library_messages)
        self.tbl_states = QTableWidget(0, 4)
        self.tbl_states.setHorizontalHeaderLabels(["Immune state", "Rows", "Description",
                                                   "Citations"])
        self.tbl_states.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.tbl_states.horizontalHeader().setStretchLastSection(True)
        self.tbl_states.verticalHeader().setVisible(False)
        self.tbl_states.setMinimumHeight(260)
        layout.addWidget(self.tbl_states, stretch=1)

    def _build_features_page(self, page):
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.cb_anchor = QComboBox()
        self.cb_anchor.setToolTip(
            "Abundance features are each population's share of this population, so samples "
            "with different numbers of events, debris or dead cells compare on the same footing.")
        self.cb_anchor.currentIndexChanged.connect(self._on_anchor_changed)
        form.addRow("Abundance anchor:", self.cb_anchor)
        self.chk_types = {}
        for key, text in FEATURE_TYPE_LABELS.items():
            chk = QCheckBox(text)
            chk.toggled.connect(self._on_feature_types_changed)
            self.chk_types[key] = chk
            form.addRow(chk)
        self.cb_readout = QComboBox()
        for key, text in READOUT_LABELS.items():
            self.cb_readout.addItem(text, key)
        self.cb_readout.currentIndexChanged.connect(self._on_readout_changed)
        form.addRow("Marker rows read as:", self.cb_readout)
        layout.addLayout(form)
        row = QHBoxLayout()
        self.btn_match = QPushButton("Check matching")
        self.btn_match.setToolTip("Build the features and match the library, without testing.")
        self.btn_match.clicked.connect(self._on_check_matching)
        row.addWidget(self.btn_match)
        self.lbl_features = QLabel("")
        self.lbl_features.setWordWrap(True)
        row.addWidget(self.lbl_features, stretch=1)
        layout.addLayout(row)
        tabs = QTabWidget()
        tabs.setMinimumHeight(360)
        self.tbl_cells = QTableWidget(0, 4)
        self.tbl_cells.setHorizontalHeaderLabels(["Population", "Parent", "Cell type", "How"])
        self.tbl_cells.verticalHeader().setVisible(False)
        self.tbl_cells.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.tbl_cells.horizontalHeader().setStretchLastSection(True)
        tabs.addTab(self.tbl_cells, "Cell types")
        match_page = QWidget()
        ml = QVBoxLayout(match_page)
        frow = QHBoxLayout()
        frow.addWidget(QLabel("Show:"))
        self.cb_match_filter = QComboBox()
        self.cb_match_filter.addItem("All rows", 'all')
        self.cb_match_filter.addItem("Rows that did not match", 'unmatched')
        self.cb_match_filter.currentIndexChanged.connect(self._fill_matches)
        frow.addWidget(self.cb_match_filter)
        frow.addStretch()
        ml.addLayout(frow)
        self.tbl_matches = QTableWidget(0, 8)
        self.tbl_matches.setHorizontalHeaderLabels(
            ["Immune state", "Cell type", "Kind", "Marker", "Direction", "Status", "Feature",
             "Detail"])
        self.tbl_matches.verticalHeader().setVisible(False)
        self.tbl_matches.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.tbl_matches.horizontalHeader().setStretchLastSection(True)
        ml.addWidget(self.tbl_matches)
        tabs.addTab(match_page, "Library matching")
        self.tbl_sets = QTableWidget(0, 4)
        self.tbl_sets.setHorizontalHeaderLabels(["Immune state", "Library rows", "Matched",
                                                 "Note"])
        self.tbl_sets.verticalHeader().setVisible(False)
        self.tbl_sets.horizontalHeader().setStretchLastSection(True)
        tabs.addTab(self.tbl_sets, "States")
        layout.addWidget(tabs, stretch=1)

    def _build_run_page(self, page):
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.cb_metric = QComboBox()
        for key, text in METRIC_LABELS.items():
            self.cb_metric.addItem(text, key)
        self.cb_metric.currentIndexChanged.connect(self._on_run_settings_changed)
        self.sb_perm = QSpinBox()
        self.sb_perm.setRange(100, ag_fsea.MAX_N_PERM)
        self.sb_perm.setSingleStep(500)
        self.sb_perm.setToolTip("Sample-label permutations. The smallest p-value a state can "
                                "reach is about 2 / (permutations + 1).")
        self.sb_perm.valueChanged.connect(self._on_run_settings_changed)
        self.sb_seed = QSpinBox()
        self.sb_seed.setRange(0, 2_147_483_647)
        self.sb_seed.valueChanged.connect(self._on_run_settings_changed)
        self.sb_min_set = QSpinBox()
        self.sb_min_set.setRange(3, 100)
        self.sb_min_set.setToolTip("States with fewer distinct matched features are not tested.")
        self.sb_min_set.valueChanged.connect(self._on_run_settings_changed)
        form.addRow("Ranking statistic:", self.cb_metric)
        form.addRow("Permutations:", self.sb_perm)
        form.addRow("Random seed:", self.sb_seed)
        form.addRow("Minimum matched features:", self.sb_min_set)
        self.chk_coherence = QCheckBox("Coherence and state network")
        self.chk_coherence.setToolTip(
            "Also ask, within each group, whether each state's features rise and fall together "
            "from donor to donor, and which states move together. Group comparisons only; each "
            f"group needs at least {ag_coherence.MIN_GROUP_SAMPLES} samples.")
        self.chk_coherence.toggled.connect(self._on_run_settings_changed)
        self.chk_background = QCheckBox("Adjust positivity features for marker background")
        self.chk_background.setToolTip(
            "When a sample's whole distribution for a marker sits brighter (stain, background, "
            "fix/perm), more cells cross a fixed gate in every population. This removes each "
            "marker's own background, measured in the sample's populations that are negative "
            "for it, before correlating.")
        self.chk_background.toggled.connect(self._on_run_settings_changed)
        self.sb_reliability = QDoubleSpinBox()
        self.sb_reliability.setRange(0.0, 0.9)
        self.sb_reliability.setSingleStep(0.05)
        self.sb_reliability.setDecimals(2)
        self.sb_reliability.setToolTip(
            "Share of a feature's donor-to-donor variation that must be real rather than "
            "counting noise. Features of rare populations (a few events per sample) fall below "
            "it and are left out of coherence, because their noise weakens every correlation.")
        self.sb_reliability.valueChanged.connect(self._on_run_settings_changed)
        form.addRow("", self.chk_coherence)
        form.addRow("", self.chk_background)
        form.addRow("Minimum reliability:", self.sb_reliability)
        layout.addLayout(form)
        self.lbl_coherence_model = QLabel("")
        self.lbl_coherence_model.setWordWrap(True)
        self.lbl_coherence_model.setStyleSheet("color: grey;")
        layout.addWidget(self.lbl_coherence_model)
        self.lbl_gate = QLabel("")
        self.lbl_gate.setWordWrap(True)
        layout.addWidget(self.lbl_gate)
        self.lbl_estimate = QLabel("")
        self.lbl_estimate.setStyleSheet("color: grey;")
        layout.addWidget(self.lbl_estimate)
        row = QHBoxLayout()
        self.btn_run = QPushButton("Run enrichment")
        self.btn_run.clicked.connect(self._on_run)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_cancel)
        row.addStretch()
        layout.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.lbl_run = QLabel("")
        self.lbl_run.setWordWrap(True)
        layout.addWidget(self.lbl_run)
        layout.addStretch()

    def _build_results_page(self, page):
        layout = QVBoxLayout(page)
        self.lbl_results = QLabel("")
        self.lbl_results.setWordWrap(True)
        layout.addWidget(self.lbl_results)
        self.tbl_results = QTableWidget(0, 14)
        self.tbl_results.setHorizontalHeaderLabels(
            ["Immune state", "Direction", "NES", "p", "FDR", "FWER", "Mean t",
             "Self-contained FDR", "Competitive FDR", "Any-direction FDR", "Tested",
             "Leading edge", "Library rows", "Note"])
        self.tbl_results.setToolTip(
            "NES, p, FDR, FWER: the rank-based enrichment score. Mean t and Self-contained "
            "FDR: did the state's features change? Competitive FDR (CAMERA): did they change "
            "more than the other features? Any-direction FDR (trends only): did they change "
            "with the variable in any direction, a rise and fall included?")
        self.tbl_results.verticalHeader().setVisible(False)
        self.tbl_results.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.tbl_results.horizontalHeader().setStretchLastSection(True)
        self.tbl_results.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl_results.setMinimumHeight(240)
        self.tbl_results.itemSelectionChanged.connect(self._on_state_selected)
        layout.addWidget(self.tbl_results)
        self.lbl_coherence = QLabel("")
        self.lbl_coherence.setWordWrap(True)
        layout.addWidget(self.lbl_coherence)
        self.tbl_coherence = QTableWidget(0, 9)
        self.tbl_coherence.setToolTip(
            "Coherence: the signed mean correlation of a state's members across the donors of "
            "a group (1: they move exactly as the library expects; 0: independently), with its "
            "95% interval. ΔC, p, FDR: does it differ between the two groups?")
        self.tbl_coherence.verticalHeader().setVisible(False)
        self.tbl_coherence.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.tbl_coherence.horizontalHeader().setStretchLastSection(True)
        self.tbl_coherence.setMinimumHeight(160)
        layout.addWidget(self.tbl_coherence)
        row = QHBoxLayout()
        row.addWidget(QLabel("State pair:"))
        self.cb_pair = QComboBox()
        self.cb_pair.setToolTip("Two states of the network: each donor's two scores, and their "
                                "member features' correlations drawn along the gate tree.")
        self.cb_pair.currentIndexChanged.connect(self._on_pair_changed)
        row.addWidget(self.cb_pair, stretch=1)
        self.pair_row = QWidget()
        self.pair_row.setLayout(row)
        layout.addWidget(self.pair_row)
        self.fig_tabs = QTabWidget()
        self.fig_tabs.setMinimumHeight(420)
        layout.addWidget(self.fig_tabs, stretch=1)
        self._n_fixed_tabs = 0

    def _build_export_page(self, page):
        layout = QVBoxLayout(page)
        note = QLabel("Each run is written to Automated_Gating_Results as CSV files (states, "
                      "leading edge, library matching, features, collapsed features). Write "
                      "them to another folder here.")
        note.setWordWrap(True)
        layout.addWidget(note)
        b = QPushButton("Write CSV files…")
        b.clicked.connect(self._on_export)
        layout.addWidget(b)
        self.lbl_export = QLabel("")
        self.lbl_export.setWordWrap(True)
        layout.addWidget(self.lbl_export)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self):
        """Show the state's settings, library and last result."""
        st = self.state
        self._filling = True
        try:
            self.cb_species.setCurrentIndex(max(self.cb_species.findData(st.fsea_species), 0))
            for key, chk in self.chk_types.items():
                chk.setChecked(key in st.fsea_feature_types)
            self.cb_readout.setCurrentIndex(max(self.cb_readout.findData(st.fsea_marker_readout), 0))
            self.cb_metric.setCurrentIndex(max(self.cb_metric.findData(st.fsea_rank_metric), 0))
            self.sb_perm.setValue(int(st.fsea_n_perm))
            self.sb_seed.setValue(int(st.fsea_seed))
            self.sb_min_set.setValue(int(st.fsea_min_set_size))
            self.chk_coherence.setChecked(bool(st.coherence_on))
            self.chk_background.setChecked(bool(st.coherence_background))
            self.sb_reliability.setValue(float(st.coherence_min_reliability))
            self.chk_background.setEnabled(bool(st.coherence_on))
            self.sb_reliability.setEnabled(bool(st.coherence_on))
            self._fill_anchor()
        finally:
            self._filling = False
        self._load_library()
        self._fill_tests()
        self._update_gate()
        self._show_result()

    def _gate_defs(self) -> list[dict]:
        return self.state.gate_defs()

    def _fill_anchor(self):
        info = fl.population_info(self._gate_defs()) if self._gate_defs() else {}
        default = ag_fsea.default_anchor(info, self._gate_defs()) if info else 'root'
        self.cb_anchor.clear()
        for key in ag_fsea.anchor_choices(info) if info else ['root']:
            label = 'All gated events' if key == 'root' else info[key]['label']
            if key == default:
                label += '  (default)'
            self.cb_anchor.addItem(label, key)
        current = self.state.fsea_anchor or default
        self.cb_anchor.setCurrentIndex(max(self.cb_anchor.findData(current), 0))

    def _load_library(self):
        st = self.state
        path = st.fsea_library_path or None
        try:
            self._library, self._library_sha, self._library_messages = fl.load_library(
                path, species=st.fsea_species)
        except Exception as exc:
            self._library, self._library_sha, self._library_messages = None, '', [str(exc)]
        name = Path(path).name if path else 'bundled library'
        if self._library is None:
            self.lbl_library.setText(f"<b>{name}</b>: could not be read.")
        else:
            lib = self._library
            versions = ', '.join(sorted(set(lib['library_version']) - {''})) or '—'
            self.lbl_library.setText(
                f"<b>{name}</b> — version {versions}, {lib['immune_state'].nunique()} states, "
                f"{len(lib)} rows ({st.fsea_species}); SHA-256 {self._library_sha[:12]}…")
        self.lbl_library_messages.setText("<br>".join(self._library_messages))
        self.tbl_states.setRowCount(0)
        if self._library is None:
            return
        for name_, rows in self._library.groupby('immune_state', sort=False):
            r = self.tbl_states.rowCount()
            self.tbl_states.insertRow(r)
            cites = sorted({c.strip() for v in rows['citation'] for c in str(v).split(';')
                            if c.strip()})
            for j, v in enumerate((name_, len(rows), rows['description'].iloc[0], '; '.join(cites))):
                self.tbl_states.setItem(r, j, _item(v))
        self.tbl_states.resizeColumnToContents(0)

    def _design(self) -> tuple[list[str], dict]:
        samples = list(self._samples_fn())
        return samples, self._design_fn(samples)

    def _selected(self) -> tuple[str, str]:
        """(kind, label) of the test chosen at the top: ('groups', 'B vs A')
        or ('trend', 'Trend with age'); ('groups', '') when none."""
        data = self.cb_comparison.currentData()
        if not data:
            return 'groups', ''
        kind, label = json.loads(data)
        return kind, label

    def _result(self, kind: str | None = None) -> dict:
        """The stored result of *kind* (default: the chosen test's)."""
        kind = kind or self._selected()[0]
        return (self.state.fsea_trend if kind == 'trend' else self.state.fsea) or {}

    def _trend_inputs(self) -> dict | None:
        return self._trend_fn() if self._trend_fn is not None else None

    def _trend(self) -> dict | None:
        """The trend inputs when the chosen test is the trend, else None."""
        return self._trend_inputs() if self._selected()[0] == 'trend' else None

    def _fill_tests(self, select: tuple[str, str] | None = None):
        """The Test chooser: the comparisons of the current design and of the
        last group run, then the trend set on the Results page and that of
        the last trend run. Keeps the chosen test (or selects *select*)."""
        st = self.state
        keep = select or (self._selected() if self.cb_comparison.count() else None)
        labels: list[str] = []
        samples, design = self._design()
        try:
            plans, _blocked, _used = ag_fsea.comparison_plans(
                samples, design['groups'], group_order=list(st.group_names),
                contrast_mode=st.contrast_mode, reference=st.reference_group,
                enforce_min_samples=False)
            labels = [f"{o} vs {b}" for p in plans for b, o in p['comparisons']]
        except Exception:
            labels = []
        if (st.fsea or {}).get('settings', {}).get('kind') != 'trend':
            labels += [c for c in ag_fsea.comparison_labels(st.fsea or {}) if c not in labels]
        items = [('groups', c) for c in labels]
        trend = self._trend_inputs()
        if trend:
            items.append(('trend', ag_fsea.trend_label(
                trend['variable'], [trend['baseline']] + [g for g in trend['groups']
                                                          if g != trend['baseline']],
                trend['interaction'])))
        items += [('trend', c) for c in ag_fsea.comparison_labels(st.fsea_trend or {})
                  if ('trend', c) not in items]
        self.cb_comparison.blockSignals(True)
        self.cb_comparison.clear()
        for kind, label in items:
            self.cb_comparison.addItem(label, json.dumps([kind, label]))
        index = self.cb_comparison.findData(json.dumps(list(keep))) if keep else -1
        if index < 0 and keep:
            # The same kind of test, renamed (e.g. a new trend variable).
            index = next((i for i, (kind, _l) in enumerate(items) if kind == keep[0]), -1)
        self.cb_comparison.setCurrentIndex(max(index, 0))
        self.cb_comparison.blockSignals(False)

    def _update_gate(self):
        """Sample numbers of the chosen test against the minimum the
        enrichment needs; the Run button names what it runs."""
        st = self.state
        kind, label = self._selected()
        if kind == 'trend':
            self.btn_run.setText(f"Run enrichment: {label}")
            trend = self._trend()
            n = (sum(1 for s in trend['samples'] if trend['time'].get(s) == trend['time'].get(s))
                 if trend else 0)
            if not trend:
                self.lbl_gate.setText("Choose the trend variable and groups on the Results page.")
            elif n < ag_fsea.MIN_TOTAL_SAMPLES:
                self.lbl_gate.setText(
                    f"<span style='color:#b00020'>{n} sample(s) have a value for "
                    f"'{trend['variable']}'; enrichment needs at least "
                    f"{ag_fsea.MIN_TOTAL_SAMPLES}.</span>")
            else:
                self.lbl_gate.setText(
                    f"Trend with '{trend['variable']}': {n} samples ("
                    + ", ".join(trend['groups']) + ")"
                    + (", group × trend" if trend['interaction'] else "") + ".")
            self.btn_run.setEnabled(bool(trend) and n >= ag_fsea.MIN_TOTAL_SAMPLES
                                    and self._worker is None and bool(st.summaries))
            self._update_estimate()
            return
        self.btn_run.setText("Run enrichment: group comparisons")
        samples, design = self._design()
        counts: dict[str, int] = {}
        for s in samples:
            g = design['groups'].get(s)
            if g:
                counts[g] = counts.get(g, 0) + 1
        if len(counts) < 2:
            self.lbl_gate.setText("Assign samples to at least two groups on the Results page.")
            self.btn_run.setEnabled(False)
            return
        text = ", ".join(f"{g}: {n}" for g, n in counts.items())
        need = (f"Enrichment needs at least {ag_fsea.MIN_TOTAL_SAMPLES} samples in the two "
                f"groups compared and {ag_fsea.MIN_GROUP_SAMPLES} in each")
        try:
            plans, blocked, _used = ag_fsea.comparison_plans(
                samples, design['groups'], group_order=list(st.group_names),
                contrast_mode=st.contrast_mode, reference=st.reference_group)
        except Exception as exc:
            plans, blocked = [], [{'reason': str(exc)}]
        if plans:
            extra = "".join(f"<br>Not tested — {b['reason']}" for b in blocked)
            self.lbl_gate.setText(f"Samples per group — {text}. {need}.{extra}")
        else:
            self.lbl_gate.setText(
                f"<span style='color:#b00020'>Samples per group — {text}. {need}; with fewer, "
                "the permutation null is too coarse and set results are not reliable.</span>")
        self.btn_run.setEnabled(bool(plans) and self._worker is None and bool(st.summaries))
        self._update_estimate()
        self._update_coherence_model()

    def _update_estimate(self):
        if self._preview is None:
            self.lbl_estimate.setText("Check matching to see an estimate of the run time.")
            return
        samples, design = self._design()
        n_feat = self._preview['values'].shape[1]
        n_perm = int(self.sb_perm.value())
        trend = self._trend()
        fast = design['pairing'] is None and design['covariates'] is None and not trend
        per = (_SECONDS_PER_PERM_FEATURE * n_feat if fast
               else _SECONDS_PER_PERM_ELEMENT * n_feat * max(len(samples), 1) + 0.002)
        n_comp = max(1, len(self.state.group_names) - 1)
        if trend:
            # Each permutation fits the curve, its reduced model and the line.
            per, n_comp = 3.0 * per, 1
        secs = per * n_perm * n_comp
        how = ("residual permutation, trend" if trend else
               "group-label permutation" if fast else "residual permutation (pairing/covariates)")
        text = (f"About {secs:.0f} s for {n_feat} features × {n_perm} permutations × {n_comp} "
                f"comparison(s) ({how}; estimated from a two-core machine).")
        if self.state.coherence_on and not trend:
            # 23 ms per relabelling for 206 members and 143 samples, plus the intervals.
            testable = [s for s in self._preview['sets'] if s['testable']]
            n_mem = sum(len(s['features']) for s in testable)
            scale = (max(n_mem, 1) / 206) ** 2 * max(len(samples), 1) / 143
            n_rel = ag_coherence.relabellings_needed(n_perm, len(testable))
            coh = (0.023 * scale * (n_rel + 2 * ag_coherence.DEFAULT_N_BOOT)) * n_comp
            text += f" Coherence adds about {coh:.0f} s."
        self.lbl_estimate.setText(text)
        self._update_coherence_model()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _on_choose_library(self):
        start = self.state.fsea_library_path or str(fl.LIBRARY_FILE)
        path, _ = QFileDialog.getOpenFileName(self, "Immune state library", start,
                                              "CSV files (*.csv)")
        if not path:
            return
        self.state.fsea_library_path = path
        self._invalidate_preview()
        self._load_library()

    def _on_bundled_library(self):
        self.state.fsea_library_path = ''
        self._invalidate_preview()
        self._load_library()

    def _on_species_changed(self, _i):
        if self._filling:
            return
        self.state.fsea_species = self.cb_species.currentData()
        self._invalidate_preview()
        self._load_library()

    def _on_anchor_changed(self, _i):
        if self._filling:
            return
        self.state.fsea_anchor = self.cb_anchor.currentData() or ''
        self._invalidate_preview()

    def _on_feature_types_changed(self, _on):
        if self._filling:
            return
        self.state.fsea_feature_types = [k for k, c in self.chk_types.items() if c.isChecked()]
        self._invalidate_preview()

    def _on_readout_changed(self, _i):
        if self._filling:
            return
        self.state.fsea_marker_readout = self.cb_readout.currentData()
        self._invalidate_preview()

    def _on_run_settings_changed(self, *_a):
        if self._filling:
            return
        st = self.state
        st.fsea_rank_metric = self.cb_metric.currentData()
        st.fsea_n_perm = int(self.sb_perm.value())
        st.fsea_seed = int(self.sb_seed.value())
        st.fsea_min_set_size = int(self.sb_min_set.value())
        st.coherence_on = self.chk_coherence.isChecked()
        st.coherence_background = self.chk_background.isChecked()
        st.coherence_min_reliability = float(self.sb_reliability.value())
        self.chk_background.setEnabled(st.coherence_on)
        self.sb_reliability.setEnabled(st.coherence_on)
        self._update_estimate()

    def _invalidate_preview(self):
        self._preview = None
        self.lbl_features.setText("Settings changed — Check matching again.")
        self._update_estimate()

    # ------------------------------------------------------------------
    # Matching preview
    # ------------------------------------------------------------------

    def _prepare_kwargs(self) -> dict:
        st = self.state
        return dict(anchor_key=st.fsea_anchor, feature_types=list(st.fsea_feature_types),
                    marker_readout=st.fsea_marker_readout,
                    use_stats_parent=st.use_stats_parent, min_marker_events=st.min_marker_events,
                    min_parent_events=st.min_parent_events,
                    exclude_gating_markers=st.exclude_gating_markers,
                    channel_antigens=dict(self._antigens_fn()),
                    cell_type_overrides=dict(st.fsea_overrides),
                    min_set_size=int(st.fsea_min_set_size))

    def _on_check_matching(self):
        st = self.state
        if self._preview_worker is not None or self._library is None:
            return
        if not st.summaries:
            QMessageBox.information(self, "Nothing to match", "Gate some samples first.")
            return
        samples, _design = self._design()
        samples = samples or st.gated_samples()
        # Each sample's summary is replaced whole when it is gated again,
        # never changed in place, so the worker can read this snapshot.
        summaries = {k: st.summaries[k] for k in samples}
        gate_defs = deepcopy(self._gate_defs())
        library, kwargs = self._library.copy(), self._prepare_kwargs()

        def _job(_progress, _stop):
            return ag_fsea.prepare(summaries, samples, gate_defs, library, **kwargs)

        self.btn_match.setEnabled(False)
        self.lbl_features.setText(f"Building features for {len(samples)} sample(s) …")
        self._preview_worker = _FseaWorker(_job, parent=self)
        self._preview_worker.finished.connect(self._on_preview_done)
        self._preview_worker.start()

    def _on_preview_done(self, ok: bool, error: str, result: object):
        self._preview_worker = None
        self.btn_match.setEnabled(True)
        if not ok:
            self.lbl_features.setText(f"<span style='color:#b00020'>{error}</span>")
            return
        self._preview = result
        self._show_preview()
        self._update_estimate()

    def _show_preview(self):
        prev = self._preview
        if prev is None:
            return
        meta = prev['features']
        kept = meta[meta['kept'] == meta['feature']]
        counts = kept['type'].value_counts().to_dict()
        n_test = sum(1 for s in prev['sets'] if s['testable'])
        text = (", ".join(f"{counts.get(t, 0)} {t}" for t in ag_fsea.FEATURE_TYPES
                          if t in self.state.fsea_feature_types)
                + f" features; {n_test} of {len(prev['sets'])} states testable.")
        if prev['notes']:
            text += "<br>" + "<br>".join(prev['notes'])
        self.lbl_features.setText(text)
        self._fill_cells()
        self._fill_matches()
        self._fill_sets()

    def _fill_cells(self):
        prev = self._preview
        info = prev['pop_info']
        cell_types = list(fl.load_cell_types()['cell_type'])
        self._filling = True
        try:
            self.tbl_cells.setRowCount(0)
            for key, m in prev['cell_types'].items():
                r = self.tbl_cells.rowCount()
                self.tbl_cells.insertRow(r)
                rec = info.get(key, {})
                parent = info.get(rec.get('parent_key'), {}).get('label', '')
                self.tbl_cells.setItem(r, 0, _item(rec.get('label', key), data=key))
                self.tbl_cells.setItem(r, 1, _item(parent))
                combo = QComboBox()
                combo.addItem("—", '')
                for ct in cell_types:
                    combo.addItem(ct, ct)
                combo.setCurrentIndex(max(combo.findData(m.get('cell_type') or ''), 0))
                combo.currentIndexChanged.connect(
                    lambda _i, k=key, c=combo: self._on_override(k, c.currentData()))
                self.tbl_cells.setCellWidget(r, 2, combo)
                self.tbl_cells.setItem(r, 3, _item(m.get('method', ''),
                                                   _METHOD_COLOURS.get(m.get('method'))))
            self.tbl_cells.resizeColumnsToContents()
        finally:
            self._filling = False

    def _on_override(self, key: str, cell_type: str):
        if self._filling:
            return
        self.state.fsea_overrides[key] = cell_type or ''
        self.lbl_features.setText("Cell type changed — Check matching again to update the "
                                  "library matching.")

    def _fill_matches(self, *_a):
        prev = self._preview
        self.tbl_matches.setRowCount(0)
        if prev is None:
            return
        m = prev['matches']
        if self.cb_match_filter.currentData() == 'unmatched':
            m = m[m['status'] != 'matched']
        labels = dict(zip(prev['features']['feature'], prev['features']['label']))
        for _i, row in m.iterrows():
            r = self.tbl_matches.rowCount()
            self.tbl_matches.insertRow(r)
            vals = (row['immune_state'], row['cell_type'], row['feature_kind'], row['marker'],
                    row['direction'], row['status'], labels.get(row['feature'], row['feature']),
                    row['detail'])
            for j, v in enumerate(vals):
                self.tbl_matches.setItem(r, j, _item(v, _STATUS_COLOURS.get(row['status'])
                                                     if j == 5 else None))
        self.tbl_matches.resizeColumnsToContents()

    def _fill_sets(self):
        self.tbl_sets.setRowCount(0)
        for s in self._preview['sets']:
            r = self.tbl_sets.rowCount()
            self.tbl_sets.insertRow(r)
            for j, v in enumerate((s['name'], s['n_library'], s['n_matched'], s['note'])):
                self.tbl_sets.setItem(r, j, _item(v, None if s['testable'] or j != 2
                                                  else '#f8d7da'))
        self.tbl_sets.resizeColumnsToContents()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _on_run(self):
        st = self.state
        if self._worker is not None or self._library is None:
            return
        samples, design = self._design()
        # Each sample's summary is replaced whole when it is gated again,
        # never changed in place, so the worker can read this snapshot.
        summaries = {k: st.summaries[k] for k in samples}
        gate_defs = deepcopy(self._gate_defs())
        library = self._library.copy()
        kwargs = self._prepare_kwargs()
        kwargs.update(
            library_info={'path': st.fsea_library_path or 'bundled', 'sha256': self._library_sha,
                          'version': ', '.join(sorted(set(library['library_version']) - {''})),
                          'messages': list(self._library_messages)},
            group_order=list(st.group_names), contrast_mode=st.contrast_mode,
            reference=st.reference_group, pairing=design['pairing'],
            covariates=design['covariates'], numeric_covariates=design['numeric'],
            rank_metric=st.fsea_rank_metric, n_perm=int(st.fsea_n_perm), seed=int(st.fsea_seed))
        trend = self._trend()
        fingerprint = input_fingerprint(st, samples, design, self._library_sha, trend)
        groups = design['groups']
        if trend:
            samples, design = trend['samples'], trend['design']
            summaries = {k: st.summaries[k] for k in samples}
            groups = design['groups']
            for key in ('group_order', 'contrast_mode', 'reference'):
                kwargs.pop(key)
            kwargs.update(pairing=design['pairing'], covariates=design['covariates'],
                          numeric_covariates=design['numeric'])
            kwargs.update(time=trend['time'], variable=trend['variable'], unit=trend['unit'],
                          scale=trend['scale'], shape=trend['shape'], k=trend['k'],
                          interaction=trend['interaction'], baseline=trend['baseline'],
                          patterns=trend['patterns'])

        coherence = None
        if not trend and st.coherence_on:
            trend_in = self._trend_inputs()
            coherence = dict(
                group_order=list(st.group_names), contrast_mode=st.contrast_mode,
                reference=st.reference_group, pairing=design['pairing'],
                covariates=design['covariates'], numeric_covariates=design['numeric'],
                trend_time=dict(trend_in['time']) if trend_in else None,
                trend_variable=trend_in['variable'] if trend_in else '',
                channel_antigens=dict(self._antigens_fn()),
                background=bool(st.coherence_background),
                min_reliability=float(st.coherence_min_reliability),
                n_perm=int(st.fsea_n_perm), seed=int(st.fsea_seed))

        def _job(progress, stop):
            run = ag_fsea.run_trend_enrichment if trend else ag_fsea.run_enrichment
            share = 0.5 if coherence is not None else 1.0
            res = run(summaries, samples, groups, gate_defs, library,
                      progress=lambda f, m: progress(share * f, m), stop_check=stop, **kwargs)
            if coherence is not None:
                try:
                    res['coherence'] = ag_coherence.run_coherence(
                        summaries, samples, groups, gate_defs, res,
                        progress=lambda f, m: progress(share + (1 - share) * f, m),
                        stop_check=stop, **coherence)
                except ag_coherence.CoherenceCancelled as exc:
                    raise ag_fsea.FseaCancelled(str(exc)) from exc
                except ag_coherence.CoherenceBlocked as exc:
                    res['coherence'] = {'blocked_message': str(exc)}
            res['input_fingerprint'] = fingerprint
            return res

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        self.lbl_run.setText(f"Running on {len(samples)} sample(s) …")
        self._worker = _FseaWorker(_job, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_run_done)
        self._worker.start()

    def _on_cancel(self):
        if self._worker is not None:
            self._worker.cancel()
            self.lbl_run.setText("Cancelling …")

    def _on_progress(self, frac: float, msg: str):
        self.progress.setValue(int(1000 * max(0.0, min(1.0, frac))))
        self.lbl_run.setText(msg)

    def _on_run_done(self, ok: bool, error: str, result: object):
        self._worker = None
        self.btn_cancel.setEnabled(False)
        self._update_gate()
        if not ok:
            if error == 'cancelled':
                self.lbl_run.setText("Cancelled; the previous results are kept.")
            else:
                self.lbl_run.setText(f"<span style='color:#b00020'>{error}</span>")
            return
        result['run_fingerprint'] = self.state.run_fingerprint
        kind = 'trend' if result['settings'].get('kind') == 'trend' else 'groups'
        if kind == 'trend':
            self.state.fsea_trend = result
        else:
            self.state.fsea = result
        self.lbl_run.setText(f"Done in {result['seconds']:.0f} s ({result['settings']['engine']}).")
        labels = ag_fsea.comparison_labels(result)
        self._fill_tests(select=(kind, labels[0]) if labels else None)
        self._update_gate()
        self._show_result()
        self.toolbox.setCurrentIndex(3)
        self.enrichment_finished.emit()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _show_result(self):
        """The chosen test's stored result: its run summary, table and figures."""
        kind, label = self._selected()
        res = self._result(kind)
        if not res or label not in ag_fsea.comparison_labels(res):
            what = "the trend" if kind == 'trend' else "the group comparisons"
            self.lbl_status.setText(f"No enrichment run yet for {label or what}: choose it and "
                                    "press Run.")
            self.lbl_results.setText("")
            self.tbl_results.setRowCount(0)
            self.fig_tabs.clear()
            self._show_coherence_table('')
            return
        lines = [f"Last run: {len(res['samples'])} samples, {res['settings']['n_perm']} "
                 f"permutations, library {res['library'].get('version') or ''} "
                 f"({str(res['library'].get('sha256', ''))[:12]}…)."]
        samples, design = self._design()
        if res.get('input_fingerprint') != input_fingerprint(self.state, samples, design,
                                                             self._library_sha, self._trend()):
            lines.append("<span style='color:#856404'>Samples, groups, gating, library or "
                         "settings have changed since this run.</span>")
        for b in res.get('blocked') or []:
            lines.append(f"Not tested — {b['reason']}")
        lines += res.get('notes') or []
        self.lbl_status.setText("<br>".join(lines))
        self._on_comparison_changed(0)

    def _on_test_changed(self, _i):
        """A new test chosen at the top: its Run button and, when it has
        been run, its results."""
        self._update_gate()
        self._show_result()

    def _on_comparison_changed(self, _i):
        """The table and figures of the chosen test's result."""
        kind, comp = self._selected()
        res = self._result(kind)
        self.tbl_results.setRowCount(0)
        self.fig_tabs.clear()
        self._show_coherence_table(comp if res else '')
        if not res or not comp:
            return
        st = res['states']
        st = st[st['comparison'] == comp]
        thr = self.state.pval_threshold

        def _n_below(col):
            return int((st[col] < thr).sum()) if col in st.columns else 0

        text = (f"{comp}: of {int(st['NES'].notna().sum())} states tested, FDR < {thr:g} for "
                f"{_n_below('FDR')} by enrichment score")
        if 'FDR_self' in st.columns:
            text += (f", {_n_below('FDR_self')} self-contained and {_n_below('FDR_camera')} "
                     "competitive")
        if 'FDR_any' in st.columns:
            text += f", {_n_below('FDR_any')} in any direction"
        self.lbl_results.setText(text + ".")
        self._filling = True
        try:
            for _i2, row in st.iterrows():
                r = self.tbl_results.rowCount()
                self.tbl_results.insertRow(r)
                sig = row['FDR'] == row['FDR'] and row['FDR'] < self.state.pval_threshold
                colour = ('#f8d7da' if row['direction'] == 'gained' else '#cfe2ff') if sig else None
                vals = (row['immune_state'], row['direction'], _fmt(row['NES']), _fmt(row['p']),
                        _fmt(row['FDR']), _fmt(row['FWER']), _fmt(row.get('mean_t', float('nan'))),
                        _fmt(row.get('FDR_self', float('nan'))),
                        _fmt(row.get('FDR_camera', float('nan'))),
                        _fmt(row.get('FDR_any', float('nan'))),
                        int(row['n_tested']),
                        row['leading_edge'], int(row['n_library']), row['note'])
                for j, v in enumerate(vals):
                    self.tbl_results.setItem(r, j, _item(v, colour if j < 2 else None,
                                                         data=row['immune_state'] if j == 0 else None))
            self.tbl_results.resizeColumnToContents(0)
        finally:
            self._filling = False
        self._show_dot_plot()

    def _figure_widget(self, fig, title, options_key: str | None = None):
        from honeychrome.view_components import figure_options as fo
        from honeychrome.view_components.scalable_figure_widget import ScalableFigureWidget
        store = fo.store_for(self.controller) if options_key else None
        ref = fo.FigureOptionsRef(store, 'automated_gating', options_key) if store else None
        widget = ScalableFigureWidget(fig, title=title, experiment_dir=self._experiment_dir_fn(),
                                      parent=self, options=ref)
        handler = getattr(fig, '_hover_handler', None)
        if handler is not None:
            widget.canvas.mpl_connect('motion_notify_event', handler)
        return widget

    def _show_dot_plot(self):
        import ag_fsea_plots as fp
        res = self._result()
        comps = ag_fsea.comparison_labels(res)
        try:
            fig = fp.make_dot_plot(res['states'], comps, self.state.pval_threshold,
                                   _is_dark())
            self.fig_tabs.addTab(self._figure_widget(fig, "Enrichment dot plot", 'enrichment_dot'),
                                 "Dot plot")
        except Exception as exc:
            log.exception("dot plot failed")
            self.fig_tabs.addTab(QLabel(f"Dot plot failed: {exc}"), "Dot plot")
        pattern_states = res.get('pattern_states')
        if pattern_states is not None and len(pattern_states):
            try:
                fig = fp.make_pattern_dot_plot(pattern_states, self.state.pval_threshold,
                                               _is_dark())
                self.fig_tabs.addTab(self._figure_widget(fig, "Patterns and immune states",
                                                     'enrichment_patterns'),
                                     "Patterns")
            except Exception as exc:
                log.exception("pattern dot plot failed")
                self.fig_tabs.addTab(QLabel(f"Pattern plot failed: {exc}"), "Patterns")
        self._show_coherence_figures()
        self._n_fixed_tabs = self.fig_tabs.count()

    def _on_state_selected(self):
        if self._filling:
            return
        rows = self.tbl_results.selectionModel().selectedRows() if self.tbl_results.selectionModel() else []
        if not rows:
            return
        item = self.tbl_results.item(rows[0].row(), 0)
        state = item.data(Qt.UserRole) if item is not None else None
        if state:
            self.show_state(state)

    def show_state(self, state: str):
        """Running-sum and member figures of *state* in the current comparison."""
        import ag_fsea_plots as fp
        res = self._result()
        comp = self._selected()[1]
        if not res or not comp:
            return
        keep = self._n_fixed_tabs
        while self.fig_tabs.count() > keep:
            self.fig_tabs.removeTab(keep)
        st = res['states']
        row = st[(st['comparison'] == comp) & (st['immune_state'] == state)]
        if row.empty or row['NES'].isna().all():
            return
        curve = ag_fsea.state_curve(res, comp, state)
        mb = res['members']
        mb = mb[(mb['comparison'] == comp) & (mb['immune_state'] == state)]
        try:
            fig = fp.make_curve_figure(curve, state, comp, row.iloc[0].to_dict(), _is_dark())
            self.fig_tabs.addTab(self._figure_widget(fig, f"{state} enrichment",
                                                     'enrichment_running_sum'), "Running sum")
            fig2 = fp.make_members_figure(mb, state, comp, _is_dark())
            self.fig_tabs.addTab(self._figure_widget(fig2, f"{state} members",
                                                     'enrichment_members'), "Members")
            self.fig_tabs.setCurrentIndex(keep)
        except Exception as exc:
            log.exception("state figures failed")
            self.fig_tabs.addTab(QLabel(f"Figures failed: {exc}"), "Error")

    # ------------------------------------------------------------------
    # Coherence
    # ------------------------------------------------------------------

    def _coherence(self) -> dict:
        """The coherence block of the group result, or {}."""
        if self._selected()[0] == 'trend':
            return {}
        return (self.state.fsea or {}).get('coherence') or {}

    def _update_coherence_model(self):
        """The residual model coherence will use, in words."""
        if not self.state.coherence_on:
            self.lbl_coherence_model.setText("")
            return
        _samples, design = self._design()
        terms = ['group']
        cov = design.get('covariates')
        if cov is not None and len(cov.columns):
            terms += [str(c) for c in cov.columns]
        trend = self._trend_inputs()
        if trend:
            terms.append(f"{trend['variable']} (trend variable)")
        self.lbl_coherence_model.setText(
            "Coherence: features compared after removing " + ", ".join(terms)
            + f"; groups with fewer than {ag_coherence.MIN_GROUP_SAMPLES} samples are left out "
              f"(warning below {ag_coherence.WARN_GROUP_SAMPLES}).")

    def _show_coherence_table(self, comp: str):
        """The coherence table of *comp*, and the state pairs to choose from."""
        block = self._coherence()
        self.tbl_coherence.setRowCount(0)
        self.cb_pair.blockSignals(True)
        self.cb_pair.clear()
        self.cb_pair.blockSignals(False)
        coh = block.get('coherence') if block else None
        has = coh is not None and len(coh) and comp in set(coh['comparison'])
        self.tbl_coherence.setVisible(bool(has))
        self.pair_row.setVisible(bool(has))
        if block.get('blocked_message'):
            self.lbl_coherence.setText(f"Coherence not run: {block['blocked_message']}")
            return
        if not has:
            blocked = [b['reason'] for b in (block.get('blocked') or [])
                       if b.get('comparison') == comp]
            self.lbl_coherence.setText(("Coherence not tested — " + blocked[0]) if blocked else "")
            return
        thr = self.state.pval_threshold
        rows = coh[coh['comparison'] == comp]
        a, b = str(rows['group_a'].iloc[0]), str(rows['group_b'].iloc[0])
        sett = block.get('settings') or {}
        lines = [f"Coherence, {comp}: {int((rows['FDR'] < thr).sum())} of "
                 f"{int(rows['dC'].notna().sum())} states differ (FDR < {thr:g}); "
                 f"{int(rows['n_a'].iloc[0])} {a} and {int(rows['n_b'].iloc[0])} {b} samples"
                 + (", swapped within pairs" if bool(rows['paired'].iloc[0]) else "")
                 + ". Model: group" + "".join(f", {t}" for t in sett.get('model') or [])
                 + (f"; background removed for {sett.get('members_adjusted', 0)} features "
                    f"({sett.get('markers_with_background', 0)} markers)" if sett.get('background')
                    else "; no background adjustment")
                 + f"; minimum reliability {sett.get('min_reliability', 0):g}."]
        lines += block.get('notes') or []
        self.lbl_coherence.setText("<br>".join(lines))
        headers = ["Immune state", "Lineage", f"C {a}", f"C {b}", "ΔC", "p", "FDR",
                   "Members used", "Left out"]
        self.tbl_coherence.setHorizontalHeaderLabels(headers)
        mem = block['members']
        mem = mem[mem['comparison'] == comp]
        for _i, r in rows.iterrows():
            k = self.tbl_coherence.rowCount()
            self.tbl_coherence.insertRow(k)
            sig = r['FDR'] == r['FDR'] and r['FDR'] < thr
            colour = ('#f8d7da' if r['dC'] > 0 else '#cfe2ff') if sig else None
            out = mem[(mem['immune_state'] == r['immune_state']) & ~mem['used'].astype(bool)]
            vals = (r['immune_state'], r['lineage'] or '—',
                    f"{_fmt(r['C_a'])} ({_fmt(r['C_a_lo'])} to {_fmt(r['C_a_hi'])})",
                    f"{_fmt(r['C_b'])} ({_fmt(r['C_b_lo'])} to {_fmt(r['C_b_hi'])})",
                    _fmt(r['dC']), _fmt(r['p']), _fmt(r['FDR']),
                    f"{int(r['members_used'])} of {int(r['members_total'])}",
                    '; '.join(f"{lab}: {why}" for lab, why in zip(out['label'], out['reason'])))
            for j, v in enumerate(vals):
                self.tbl_coherence.setItem(k, j, _item(v, colour if j < 1 else None))
        self.tbl_coherence.resizeColumnToContents(0)
        edges = block['edges']
        edges = edges[edges['comparison'] == comp].copy()
        within = (edges['FDR_a'] < thr) | (edges['FDR_b'] < thr) | (edges['FDR'] < thr)
        edges = edges[within].sort_values(['FDR', 'p'], na_position='last')
        self.cb_pair.blockSignals(True)
        for _i, e in edges.iterrows():
            tag = " (differs)" if e['FDR'] < thr else ""
            self.cb_pair.addItem(f"{e['state_1']} — {e['state_2']}{tag}",
                                 json.dumps([e['state_1'], e['state_2']]))
        self.cb_pair.blockSignals(False)
        self.pair_row.setVisible(self.cb_pair.count() > 0)

    def _show_coherence_figures(self):
        import ag_fsea_plots as fp
        block = self._coherence()
        comp = self._selected()[1]
        coh = block.get('coherence') if block else None
        if coh is None or not len(coh) or comp not in set(coh['comparison']):
            return
        from honeychrome.view_components.differential_plots import GROUP_COLOURS
        st = self.state
        colours = {g: st.group_colors.get(g) or GROUP_COLOURS[i % len(GROUP_COLOURS)]
                   for i, g in enumerate(st.group_names)}
        for title, tab, key, make in (
                ("Coherence", "Coherence", 'enrichment_coherence',
                 lambda: fp.make_coherence_figure(block, comp, st.pval_threshold, _is_dark(),
                                                  colours)),
                ("State network", "State network", 'enrichment_state_network',
                 lambda: fp.make_state_network_figure(block, comp, st.pval_threshold,
                                                      _is_dark()))):
            try:
                self.fig_tabs.addTab(self._figure_widget(make(), f"{title} — {comp}", key), tab)
            except Exception as exc:
                log.exception("%s figure failed", title)
                self.fig_tabs.addTab(QLabel(f"{title} failed: {exc}"), tab)
        self._add_pair_tab()

    def _add_pair_tab(self):
        import ag_fsea_plots as fp
        block = self._coherence()
        comp = self._selected()[1]
        data = self.cb_pair.currentData()
        if not block or not data:
            return
        s1, s2 = json.loads(data)
        coh = block['coherence']
        row = coh[coh['comparison'] == comp].iloc[0]
        groups = (str(row['group_b']), str(row['group_a']))
        from honeychrome.view_components.differential_plots import GROUP_COLOURS
        st = self.state
        colours = {g: st.group_colors.get(g) or GROUP_COLOURS[i % len(GROUP_COLOURS)]
                   for i, g in enumerate(st.group_names)}
        try:
            view = ag_coherence.pair_view(block, comp, s1, s2)
            fig = fp.make_state_pair_figure(view, s1, s2, comp, groups, _is_dark(), colours)
            widget = self._figure_widget(fig, f"{s1} and {s2} — {comp}", 'enrichment_state_pair')
        except Exception as exc:
            log.exception("state pair figure failed")
            widget = QLabel(f"State pair failed: {exc}")
        self.fig_tabs.addTab(widget, "State pair")

    def _on_pair_changed(self, _i):
        if self._filling:
            return
        for k in range(self.fig_tabs.count()):
            if self.fig_tabs.tabText(k) == "State pair":
                self.fig_tabs.removeTab(k)
                self._add_pair_tab()
                self.fig_tabs.tabBar().moveTab(self.fig_tabs.count() - 1, k)
                self.fig_tabs.setCurrentIndex(k)
                return

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _on_export(self):
        if not (self.state.fsea or self.state.fsea_trend):
            QMessageBox.information(self, "Export", "Run the enrichment first.")
            return
        start = str(Path(self._experiment_dir_fn() or '.') / 'Automated_Gating_Results')
        folder = QFileDialog.getExistingDirectory(self, "Export to folder", start)
        if not folder:
            return
        from datetime import datetime
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        try:
            written = []
            for stem, result in (('enrichment', self.state.fsea),
                                 ('enrichment_trend', self.state.fsea_trend)):
                if result:
                    written += write_enrichment_csv(Path(folder), stem, stamp, result)
        except Exception as exc:
            log.exception("enrichment export failed")
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        self.lbl_export.setText("Written: " + ", ".join(p.name for p in written))


def write_enrichment_csv(folder: Path, stem: str, timestamp: str, result: dict) -> list[Path]:
    """Write the run's tables as ``<stem>_<timestamp>_enrichment_<table>.csv``.
    Worker-safe."""
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for name, frame in ag_fsea.result_tables(result).items():
        if frame is None:
            continue
        path = folder / f"{stem}_{timestamp}_enrichment_{name}.csv"
        pd.DataFrame(frame).to_csv(path, index=False)
        out.append(path)
    return out


def _is_dark() -> bool:
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    palette = app.palette() if app is not None else QPalette()
    return palette.color(QPalette.ColorRole.Base).value() < 128
