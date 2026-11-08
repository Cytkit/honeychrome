"""
automated_gating_tab.py — Automated Gating plugin for Honeychrome
=================================================================
Builds a gating model (``.agmodel``), applies it to experiment samples and
compares the gated populations between sample groups. Five pages build
the model (``ag_builder.BuilderPages``):

  • Transforms, Import/Setup, Hierarchy, Train — see ag_builder.py; Export,
              the last page, saves the model.
              A saved model is loaded on Import/Setup, where its channels
              are matched to this experiment antigen by antigen.

and three apply it, after Train:

  • Run     — choose for each gate whether its stored boundary is applied as
              it is (fixed) or recalculated on each sample (recalculate,
              with a drift check against the stored boundary); pick samples
              and gate them in a background worker. Each sample is unmixed
              with its own autofluorescence profile assignment and
              restricted to its time-QC keep-mask when present. Only counts,
              parent counts, per-population marker medians and a display
              subsample (events and population memberships) are kept per
              sample; the gated views show one population's gates pooled
              over every sample and for each validation sample.
  • Results — sample groups and covariates, moderated tests of population
              frequencies (log-odds within the parent population), optional
              negative-binomial tests of counts, population-first tests of
              marker medians, heatmaps and volcano plots, a population
              comparison (each group's pooled biplot and the per-sample
              values with the test's p-value), and CSV export.
  • Enrichment — Flow Set Enrichment Analysis: curated immune states
              tested for coordinated shifts between the groups of the
              Results page (``ag_fsea``, page in ``ag_fsea_ui``).

Run applies the model as it stands on the model pages, in their transforms,
once every gate has a trained boundary. Gated results record a hash of the
model they were made with (gates, boundaries, transforms and gate modes).
Gated results, display subsamples and statistics are saved in the
experiment's cache folder and restored when the experiment is opened again;
each gating and statistics run is also written as CSV files to
``Automated_Gating_Results`` (named after the model, with a timestamp).

Gate application lives in ``ag_core.py``, alignment and feature tables in
``ag_results.py``, and the statistics in
``controller_components/differential_stats.py`` (shared with the
Honeycluster plugin).

Plugin contract (plugin_loaders.py):
  plugin_name  str        — main-window tab title
  PluginWidget(QWidget)   — instantiated with (bus, controller)
"""

import hashlib
import json
import os as _os
import sys
import threading
import uuid
import warnings
from copy import deepcopy
from dataclasses import dataclass, field, fields as dataclass_fields
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QPushButton, QLabel, QComboBox, QGroupBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QFormLayout,
    QSplitter, QFileDialog, QMessageBox, QInputDialog,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QTabWidget, QTreeWidget, QTreeWidgetItem,
    QAbstractItemView, QListWidget, QListWidgetItem,
    QProgressBar, QGridLayout, QDialog, QColorDialog, QLineEdit,
)

import honeychrome
from honeychrome.settings import af_abundance_channel
from honeychrome.controller_components import differential_stats as ds
from honeychrome.controller_components import trend_stats as ts
from honeychrome.controller_components.job_gate import acquire_job, guard_while_busy
from honeychrome.controller_components import experiment_state
from honeychrome.controller_components import covariate_check, sample_table
from honeychrome.view_components import suggest_groupings_dialog as sgd
from honeychrome.view_components import covariate_check_dialog, sample_table_import_dialog
from honeychrome.view_components.checkable_header import CheckableHeaderView
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

import ag_builder  # noqa: E402
import ag_core  # noqa: E402
import ag_fsea  # noqa: E402
import ag_fsea_ui  # noqa: E402
import ag_help_texts  # noqa: E402
import ag_report  # noqa: E402
import ag_results  # noqa: E402
import transforms_panel as tp  # noqa: E402
from ag_widgets import (  # noqa: E402
    FilterBar, PopulationTree, add_concertina_page, block_wheel, make_concertina,
    text_matches,
)
from honeychrome.view_components.side_tab_widget import SideTabWidget  # noqa: E402

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
ScalableFigureWidget = None

_qt_imports_done = False


def _ensure_qt_imports():
    """Import pyqtgraph and Honeychrome view components (main thread only)."""
    global pg, ZoomAxis, NoPanViewBox, TransparentGraphicsLayoutWidget
    global CopyableTableWidget, HelpToggleWidget, ScalableFigureWidget, _qt_imports_done
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
    from honeychrome.view_components.scalable_figure_widget import (
        ScalableFigureWidget as _SFW,
    )
    ScalableFigureWidget = _SFW
    _qt_imports_done = True


# ---------------------------------------------------------------------------
# Plugin identity and defaults
# ---------------------------------------------------------------------------
plugin_name = 'Automated Gating'

DEFAULT_DISPLAY_EVENTS = 20_000
DISPLAY_SEED = 42
RESULTS_FILE = Path('cache') / 'automated_gating' / 'results.json'
# The marker medians and quartiles of RESULTS_FILE's samples, as arrays.
MARKERS_FILE = Path('cache') / 'automated_gating' / 'results_markers.npz'
RESULTS_FORMAT = 3
STATS_FILE = Path('cache') / 'automated_gating' / 'stats.json'
TREND_FILE = Path('cache') / 'automated_gating' / 'trend.json'
ENRICHMENT_FILE = Path('cache') / 'automated_gating' / 'enrichment.json'
SAMPLE_ANALYSIS_FILE = Path('cache') / 'automated_gating' / 'sample_analysis.json'
DISPLAY_DIR = Path('cache') / 'automated_gating' / 'display'
EXPORT_DIR = Path('Automated_Gating_Results')
# Events per group in the population comparison's pooled biplots.
GROUP_PLOT_EVENTS = 100_000
# Heatmaps and marker summaries open at this size (percent of natural size).
HEATMAP_SCALE_PCT = 50
# Samples-table column sizes (pixels): the sample name is capped so a long
# name cannot push the other columns out of view; covariates get room to type.
SAMPLE_COLUMN_MAX_WIDTH = 280
COVARIATE_COLUMN_MIN_WIDTH = 110
POP_FILTERS = (('significant', 'Significant populations'), ('all', 'All populations'))

MODE_LABELS = {ag_core.MODE_FIXED: 'Fixed', ag_core.MODE_RECALCULATE: 'Recalculate'}
DRIFT_ACTIONS = {'flag': 'Flag only', 'reference': 'Use the stored boundary'}
FDR_SCOPES = {'global': 'Pooled over comparisons', 'per_comparison': 'Within each comparison'}
CONTRAST_MODES = {'reference': 'Each group against a reference', 'pairwise': 'All pairs of groups'}
COUNTS_METHOD_LABELS = {'moderated': 'moderated log odds', 'ql': 'quasi-likelihood F',
                        'wald': 'Wald'}
STATS_TESTS = {'groups': 'Compare groups', 'trend': 'Trend with a numeric variable'}
TREND_SHAPE_LABELS = {'curve': 'Smooth curve', 'line': 'Straight line',
                      'points': 'Time points as groups'}
TREND_SCALE_LABELS = {'linear': 'Linear', 'log2': 'Log2 (value + 1): slope per doubling'}
TREND_CALL_LABELS = {'either': 'Curve or slope (each at FDR ÷ 2)', 'curve': 'Curve only',
                     'slope': 'Slope only'}
TREND_MAX_DF = 6
# Statistics settings put back by the Restore defaults button. The design
# (comparisons, reference group, pairing, adjustment covariates) is kept.
STATS_DEFAULT_KEYS = ('run_freq', 'run_counts', 'counts_method', 'counts_reference',
                      'min_parent_events', 'run_markers', 'use_stats_parent',
                      'exclude_gating_markers', 'min_marker_events', 'pval_threshold', 'fc_threshold',
                      'marker_threshold', 'use_treat', 'fdr_scope', 'exclude_flagged',
                      'trend_shape', 'trend_df', 'trend_scale', 'trend_call', 'trend_min_r2',
                      'trend_patterns')
# Statistics settings applied to stored results without running them again.
THRESHOLD_KEYS = ('pval_threshold', 'fc_threshold', 'marker_threshold', 'use_treat', 'fdr_scope',
                  'trend_call', 'trend_min_r2', 'trend_patterns')

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
    """{channel: transform parameter dict} of the experiment's display transforms."""
    live = getattr(controller, 'unmixed_transformations', None) or {}
    return {ch: ag_core.transform_params(tr) for ch, tr in live.items()}


def _export_stem(model_path: str) -> str:
    import drc_report
    return drc_report.sanitize_filename(Path(model_path).stem or 'automated_gating')


def _sample_label(key: str) -> str:
    """Short display name for a sample key."""
    return Path(str(key)).stem


def _group_colours(state) -> dict:
    """{group: colour} for every group: the colour chosen on the Groups page,
    else differential_plots.GROUP_COLOURS in group order, so a group has the
    same colour in every figure."""
    from honeychrome.view_components.differential_plots import GROUP_COLOURS
    return {g: state.group_colors.get(g) or GROUP_COLOURS[i % len(GROUP_COLOURS)]
            for i, g in enumerate(state.group_names)}


def _population_labels(gate_defs: list[dict]) -> dict:
    """{'gate/pop': display name}: the population's label, prefixed by its
    gate only where another population shares the label."""
    return ag_core.gate_population_names(gate_defs)


def model_sha(model: dict, converted: bool = False) -> str:
    """Hash of a model's gates, trained boundaries and transforms (not its
    metadata or gate modes, which run_fingerprint covers).

    converted : the model is already in JSON types (as ag_core.new_model
        returns it), so the conversion copy is skipped; the hash is the same.
    """
    payload = {k: (model or {}).get(k) for k in
               ('gate_definitions', 'trained_boundaries', 'transforms')}
    if not converted:
        payload = ag_core.to_jsonable(payload)
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     default=str).encode('utf-8')).hexdigest()


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
    """Automated gating session state (the Run, Results and Enrichment pages).

    model_path     : the model's file, when it was last saved or loaded
                     (names the CSV files and the report).
    applied_model  : the model built on the model pages, in this
                     experiment's channels and the plugin's transforms;
                     what the Run tab applies (set by PluginWidget).
    model_sha      : model_sha(applied_model) (gates, boundaries, transforms).
    modes          : {gate: 'fixed' | 'recalculate'}; saved with the model as
                     its application block, with the drift settings.
    summaries      : {sample key: ag_results.summarise_sample() record}.
    load_info      : {sample key: event counts and AF profiles used}.
    display        : {sample key: display subsample and masks} — this
                     session only.
    covariates     : {covariate name: {sample key: value}}.
    numeric_covariates: covariate names entered into the statistics as
                     continuous numbers (e.g. age); the rest are categories.
    stats          : results of the last statistics run (kept in STATS_FILE).
    stats_test     : 'groups' (compare groups) or 'trend' (trend with a
                     numeric covariate) — which test the Statistics page runs.
    trend_*        : trend settings: the numeric covariate and its unit, the
                     groups included ([] = all), group × trend interaction,
                     shape, curve df, scale, how a trend is called and the
                     minimum partial R² (``trend_stats``).
    trend          : results of the last trend run (kept in TREND_FILE).
    pca, predictive: the last sample PCA and predictive analysis (from the
                     statistics run's per-sample values; kept in
                     SAMPLE_ANALYSIS_FILE with the run_id of that run).
    pending_samples: samples of a paused (or interrupted) run still to be
                     gated; empty when the last run finished.
    run_fingerprint: what the gated results depend on (model file, channel
                     alignment, gate modes, drift settings, marker channels,
                     transforms), so a paused run is only resumed with the
                     same settings.
    run_unmixing   : digest of the experiment's unmixing when the results
                     were gated (``unmixing_digest``; '' for results gated
                     before it was kept).
    transform_params: {channel: transform parameters} gating uses (the
                     model pages' transforms).
    run_transforms : the transforms the gated results were made in (their
                     boundaries and display plots are in these units).
    report_populations: population keys ticked for the report.
    fsea_*         : Enrichment page settings (library file, '' for the
                     bundled one; species; abundance anchor, '' for the
                     default; feature types; marker readout; ranking
                     statistic; permutations; seed; minimum set size;
                     {population key: cell type} overrides).
    fsea           : the last enrichment run of the group comparisons, and
    fsea_trend     : of the trend (both kept in ENRICHMENT_FILE).

    results_revision (not a field) counts the times one of _RESULT_FIELDS
    was given another object, so the pages can tell whether the results
    they show are still current without comparing them.
    """
    model_path: str = ''
    applied_model: dict = field(default_factory=dict)
    model_sha: str = ''
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
    pending_samples: list = field(default_factory=list)
    run_fingerprint: str = ''
    run_unmixing: str = ''
    transform_params: dict = field(default_factory=dict)
    run_transforms: dict = field(default_factory=dict)
    report_populations: list = field(default_factory=list)
    pop_filter: str = 'significant'

    group_names: list = field(default_factory=list)
    group_patterns: dict = field(default_factory=dict)
    group_colors: dict = field(default_factory=dict)
    sample_groups: dict = field(default_factory=dict)
    covariates: dict = field(default_factory=dict)
    numeric_covariates: list = field(default_factory=list)
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
    counts_method: str = 'moderated'
    counts_reference: str = 'anchor'
    min_parent_events: int = ag_results.DEFAULT_MIN_PARENT_EVENTS
    run_markers: bool = True
    use_stats_parent: bool = True
    exclude_gating_markers: bool = True
    min_marker_events: int = ag_results.DEFAULT_MIN_MARKER_EVENTS
    exclude_flagged: bool = False
    stats: dict = field(default_factory=dict)
    stats_test: str = 'groups'
    trend_variable: str = ''
    trend_unit: str = ''
    trend_groups: list = field(default_factory=list)
    trend_interaction: bool = False
    trend_shape: str = 'curve'
    trend_df: int = ts.DEFAULT_CURVE_DF
    trend_scale: str = 'linear'
    trend_call: str = 'either'
    trend_min_r2: float = ts.DEFAULT_MIN_R2
    trend_patterns: int = 0          # shape patterns; 0 = chosen from stability
    trend: dict = field(default_factory=dict)
    pca_use_freq: bool = True
    pca_use_counts: bool = False
    pca_use_markers: bool = False
    pca_n_loadings: int = 10
    pred_use_freq: bool = True
    pred_use_markers: bool = False
    pred_lasso: bool = True
    pred_forest: bool = True
    pred_repeats: int = 5
    age_use_freq: bool = True
    age_use_markers: bool = False
    age_repeats: int = 5
    age_permutations: int = 100
    age_predict_others: bool = False
    pca: dict = field(default_factory=dict)
    predictive: dict = field(default_factory=dict)
    fsea_library_path: str = ''
    fsea_species: str = 'human'
    fsea_anchor: str = ''
    fsea_feature_types: list = field(default_factory=lambda: list(ag_fsea.FEATURE_TYPES))
    fsea_marker_readout: str = 'positivity'
    fsea_rank_metric: str = 'signed_log10p'
    fsea_n_perm: int = ag_fsea.DEFAULT_N_PERM
    fsea_seed: int = ag_fsea.DEFAULT_SEED
    fsea_min_set_size: int = ag_fsea.MIN_SET_SIZE
    fsea_overrides: dict = field(default_factory=dict)
    fsea: dict = field(default_factory=dict)
    fsea_trend: dict = field(default_factory=dict)

    def __setattr__(self, name, value):
        if name in _RESULT_FIELDS and value is not self.__dict__.get(name):
            object.__setattr__(self, 'results_revision',
                               self.__dict__.get('results_revision', 0) + 1)
        object.__setattr__(self, name, value)

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


# Large results, tracked by GatingRunState.results_revision.
_RESULT_FIELDS = ('applied_model', 'summaries', 'load_info', 'display', 'stats', 'pca',
                  'predictive', 'fsea', 'fsea_trend', 'trend')
# Further fields the Run, Results and Enrichment pages show (with
# _PERSISTED_FIELDS), compared by content.
_VIEW_FIELDS = ('model_path', 'model_sha', 'results_model_sha', 'marker_channels',
                'pending_samples', 'run_fingerprint', 'run_unmixing', 'transform_params',
                'run_transforms')

_PERSISTED_FIELDS = (
    'modes', 'drift_limit', 'drift_action', 'run_samples', 'validation_samples', 'include_controls',
    'display_events', 'group_names', 'group_patterns', 'group_colors', 'sample_groups', 'covariates',
    'numeric_covariates', 'pairing', 'adjust_covariates', 'contrast_mode', 'reference_group', 'pval_threshold',
    'fc_threshold', 'marker_threshold', 'use_treat', 'fdr_scope', 'run_freq', 'run_counts',
    'counts_method', 'counts_reference', 'min_parent_events', 'run_markers', 'use_stats_parent',
    'exclude_gating_markers', 'min_marker_events', 'exclude_flagged', 'pca_use_freq', 'pca_use_counts', 'pca_use_markers', 'pca_n_loadings',
    'stats_test', 'trend_variable', 'trend_unit', 'trend_groups', 'trend_interaction',
    'trend_shape', 'trend_df', 'trend_scale', 'trend_call', 'trend_min_r2', 'trend_patterns',
    'pred_use_freq', 'pred_use_markers', 'pred_lasso', 'pred_forest', 'pred_repeats',
    'age_use_freq', 'age_use_markers', 'age_repeats', 'age_permutations', 'age_predict_others',
    'report_populations', 'pop_filter',
    'fsea_library_path', 'fsea_species', 'fsea_anchor', 'fsea_feature_types',
    'fsea_marker_readout', 'fsea_rank_metric', 'fsea_n_perm', 'fsea_seed', 'fsea_min_set_size',
    'fsea_overrides',
)


def unmixing_digest(controller) -> str:
    """Digest of what decides every sample's unmixed values (transfer matrix,
    spillover, AF assignments and spectra, the AutoSpectral tab's method and
    spectral corrections). Main thread only."""
    from honeychrome.controller_components.sample_loader import unmixing_signature
    keys = list((controller.experiment.samples.get('all_samples') or {}).keys())
    payload = json.dumps(ag_core.to_jsonable(unmixing_signature(controller, keys)), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def run_fingerprint(state: 'GatingRunState', model_sha: str) -> str:
    """Hash of everything the gated results of a run depend on."""
    payload = {'model': model_sha, 'modes': state.modes,
               'drift_limit': state.drift_limit, 'drift_action': state.drift_action,
               'markers': state.marker_channels,
               'transforms': {ch: {k: v for k, v in (p or {}).items() if k != 'limits'}
                              for ch, p in (state.transform_params or {}).items()}}
    return hashlib.sha256(json.dumps(ag_core.to_jsonable(payload), sort_keys=True)
                          .encode('utf-8')).hexdigest()


# ---------------------------------------------------------------------------
# Gate plot tiles (Run tab and population browser)
# ---------------------------------------------------------------------------

def _gate_tile_size() -> tuple[int, int]:
    """(width, height) of a gate plot tile."""
    import honeychrome.settings as hc_settings
    tile_size = int(getattr(hc_settings, 'cytometry_plot_width_target_retrieved', 220) or 220)
    return tile_size + 20, tile_size + 60


# Approximate room the axes take in a gate tile, in pixels (tick labels and
# axis titles), so the plot area's size is known before Qt lays it out.
_TILE_AXIS_PX = (48, 40)


def _clamp_label(value: float, lim: tuple, extent_px: float, axis_px: float) -> float:
    """*value*, the centre of a centred label *extent_px* long, moved so the
    label stays within *lim*, an axis *axis_px* long."""
    span = float(lim[1]) - float(lim[0])
    if span <= 0 or axis_px <= 0:
        return value
    half = 0.5 * (extent_px + 6.0) / axis_px * span
    if 2 * half >= span:
        return 0.5 * (lim[0] + lim[1])
    return float(min(max(value, lim[0] + half), lim[1] - half))


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
    shapes = ag_core.is_shape_boundary(boundary)
    sep_y = ag_core.separation_axis(gate_def) == 'y' and not shapes
    ch_x, ch_y = ag_core.boundary_plot_channels(gate_def, boundary)
    ch_x, ch_y = ch_x or '', ch_y or ''
    tr_x = transforms.get(ch_x)
    tr_y = transforms.get(ch_y) if ch_y else None
    tile_size = int(getattr(hc_settings, 'cytometry_plot_width_target_retrieved', 220) or 220)

    container = QWidget()
    container.setFixedSize(*_gate_tile_size())
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
    if gt == '1dsep' and not ch_y:
        counts = hist_curve.yData
        y_lim = (0.0, float(np.max(counts)) if counts is not None and len(counts) else 1.0)
    regions = ag_core.population_regions(gate_def, boundary)
    fractions = fractions or {}
    plot_w, plot_h = tile_size - _TILE_AXIS_PX[0], tile_size - _TILE_AXIS_PX[1]
    for pop, entry in (boundary or {}).items():
        width = 2.5 if pop == highlight else 1.5
        label = (gate_def.get('populations') or {}).get(pop, {}).get('label', pop)
        frac = fractions.get(pop)
        text = f"{label}\n{frac * 100:.1f}%" if frac is not None else label
        if gt == '1dsep' and not shapes:
            tx = entry.get('threshold_x')
            if tx is None:
                continue
            vb.addItem(pg.InfiniteLine(pos=float(tx), angle=0 if sep_y else 90,
                                       pen=pg.mkPen(color=colour, width=1.5)))
            if sep_y:
                ly, lx, va_, ha_ = ag_report.label_position(gt, regions.get(pop), entry, y_lim, x_lim)
                t = pg.TextItem(text, color=colour, fill=fill, anchor=(0.5, 0.5))
                rect = t.boundingRect()
                t.setPos(_clamp_label(lx, x_lim, rect.width(), plot_w),
                         _clamp_label(ly, y_lim, rect.height(), plot_h))
                vb.addItem(t)
                continue
        else:
            coords = np.asarray(entry.get('boundary') or [], dtype=float)
            if coords.ndim != 2 or coords.shape[1] != 2 or not len(coords):
                continue
            xs_b, ys_b = coords[:, 0].tolist(), coords[:, 1].tolist()
            if xs_b[0] != xs_b[-1] or ys_b[0] != ys_b[-1]:
                xs_b.append(xs_b[0])
                ys_b.append(ys_b[0])
            vb.addItem(pg.PlotDataItem(xs_b, ys_b, pen=pg.mkPen(color=colour, width=width)))
        lx, ly, ha, va = ag_report.label_position('free' if shapes else gt, regions.get(pop),
                                                  entry, x_lim, y_lim)
        t = pg.TextItem(text, color=colour, fill=fill, anchor=(anchor_x[ha], anchor_y[va]))
        rect = t.boundingRect()
        if ha == 'center':
            lx = _clamp_label(lx, x_lim, rect.width(), plot_w)
        if va == 'center':
            ly = _clamp_label(ly, y_lim, rect.height(), plot_h)
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
    with the compact summary, load counts and a display record (see
    ag_results.pack_display), which is also saved in *display_dir* with the
    run fingerprint. The full event array is released before the next
    sample.

    pause() and abort() stop the run before the next gate; the sample being
    gated is dropped, so only complete samples are ever reported.
    finished(outcome, message) gives 'done', 'paused', 'cancelled' or
    'failed'.
    """

    progress = Signal(str)
    sample_done = Signal(str, object)
    finished = Signal(str, str)

    def __init__(self, experiment_dir, keys, snapshot, gate_defs, reference, modes,
                 drift_limit, drift_action, channels, transforms, marker_channels,
                 display_events, display_dir=None, fingerprint: str = '', parent=None):
        super().__init__(parent)
        self._display_dir = display_dir
        self._fingerprint = fingerprint
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
        self._stop = None

    def abort(self):
        self._stop = 'cancelled'

    def pause(self):
        if self._stop is None:
            self._stop = 'paused'

    def _stopping(self) -> bool:
        return self._stop is not None

    def run(self):
        try:
            n_done = self._execute()
        except Exception as exc:
            log.exception("automated gating failed")
            self.finished.emit('failed', str(exc))
            return
        self.finished.emit(self._stop or 'done', f"{n_done} sample(s) gated in this run.")

    def _execute(self) -> int:
        index = {ch: i for i, ch in enumerate(self._channels)}
        display_cols = [ch for ch in ag_results.gated_channels(self._gate_defs) if ch in index]
        display_idx = [index[ch] for ch in display_cols]
        limits = _axis_limits(self._transforms)
        n_done = 0
        for i, key in enumerate(self._keys, 1):
            if self._stopping():
                break
            self.progress.emit(f"Gating {_sample_label(key)} ({i}/{len(self._keys)}) …")
            try:
                out = load_unmixed(self._experiment_dir, key, self._snap, use_cache=True)
            except Exception as exc:
                log.warning("could not load %s: %s", key, exc)
                self.progress.emit(f"Could not load {_sample_label(key)}: {exc}")
                continue
            unmixed = out['unmixed']
            if self._stopping():
                break
            data_t = ag_core.transform_columns(unmixed, self._channels, self._transforms)
            try:
                result = ag_core.run_gating(
                    self._gate_defs, data_t, index, reference=self._reference,
                    modes=self._modes, drift_limit=self._drift_limit,
                    drift_action=self._drift_action, axis_limits=limits,
                    stop_check=self._stopping, transforms=self._transforms,
                )
            except ag_core.GatingStopped:
                break
            summary = ag_results.summarise_sample(self._gate_defs, result, data_t, index,
                                                  self._markers, len(data_t))
            rows = seeded_subsample_indices(len(unmixed), self._display, key, DISPLAY_SEED)
            display = ag_results.pack_display(
                self._gate_defs, result, rows,
                np.asarray(unmixed[np.ix_(rows, display_idx)], dtype=np.float32), display_cols)
            if self._display_dir is not None:
                try:
                    ag_results.save_display(Path(self._display_dir) / ag_results.display_file_name(key),
                                            display, key, self._fingerprint)
                except Exception:
                    log.exception("could not save the display events of %s", key)
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

def _gates_on(gate_defs: list[dict], node) -> list[dict]:
    """Gates drawn on population *node* ((gate, pop); the root is ('', 'root'))."""
    gate, pop = node
    out = []
    for g in ag_core.ordered_gate_defs(gate_defs):
        if (g.get('parent_gate') or '') != (gate or ''):
            continue
        if gate and (g.get('parent_popul') or 'root') != pop:
            continue
        out.append(g)
    return out


class RunTab(QWidget):
    """Set how each gate is applied, pick samples, gate them, and inspect the
    result population by population.

    The gated views show the gates drawn on the population picked in the
    hierarchy: pooled over every gated sample (the stored boundaries) and,
    in a row each, for the sample selected in the main window's sample
    panel and for every validation sample (their own boundaries).

    gating_transforms : callable -> {channel: Transform} gating uses.
    result_transforms : callable -> {channel: Transform} the gated results
                        are in (for drawing them).
    display_dir       : callable -> folder the display records are saved in.
    readiness         : callable -> '' when the model can be applied, else
                        why not (shown in place of the page).
    """

    gating_finished = Signal(str)

    COL_RUN, COL_VALIDATE, COL_SAMPLE, COL_AF = range(4)

    def __init__(self, state: GatingRunState, bus, controller, gating_transforms=None,
                 result_transforms=None, display_dir=None, readiness=None, parent=None):
        super().__init__(parent)
        self.state = state
        self.bus = bus
        self.controller = controller
        self._gating_transforms = gating_transforms or (
            lambda: self.controller.unmixed_transformations or {})
        self._result_transforms = result_transforms or self._gating_transforms
        self._display_dir = display_dir or (lambda: None)
        self._readiness = readiness or (lambda: '')
        self._worker: _GatingWorker | None = None
        self._run_keys: list[str] = []
        self._before_run: dict | None = None
        self._filling = False
        self._pool_cache = None
        self._tree_for = None
        self._gate_items: dict = {}
        self._gates_for = None
        # Gated-view tiles of the shown population ({'line', 'row', 'col',
        # 'kind', 'key', 'gate', 'holder', 'built'}), built when scrolled near.
        self._tiles: list = []
        self._tile_ctx: dict = {}
        self._tile_timer = QTimer(self)
        self._tile_timer.setSingleShot(True)
        self._tile_timer.setInterval(30)
        self._tile_timer.timeout.connect(self._build_visible_tiles)
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)
        outer.addWidget(HelpToggleWidget(text=ag_help_texts.AG_RUN, scrolling=True))

        self.banner = QLabel("")
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet("QLabel { color: #856404; background: #fff3cd; "
                                  "border: 1px solid #ffc107; border-radius: 4px; padding: 6px; }")
        outer.addWidget(self.banner)

        self.content = QWidget()
        cl = QVBoxLayout(self.content)
        cl.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.content, stretch=1)
        # Takes the space when the content is hidden, so the help and the
        # banner stay at the top.
        outer.addStretch()

        # One section open at a time; each page scrolls on its own.
        self.toolbox = make_concertina()
        cl.addWidget(self.toolbox)

        gates_page = self._build_gates_page()

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
        self.sb_display = block_wheel(QSpinBox())
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
        hh.setSectionResizeMode(QHeaderView.Interactive)
        hh.setStretchLastSection(False)
        self.sample_table.setTextElideMode(Qt.ElideMiddle)
        self.sample_table.setWordWrap(False)
        self.sample_table.verticalHeader().setVisible(False)
        self.sample_table.setToolTip(
            "Tick samples to gate. Validation samples are gated too, and their gates are "
            "plotted under Gated views beside the pooled view.")
        self.sample_table.itemChanged.connect(self._on_sample_item_changed)
        tl.addWidget(self.sample_table)

        self.lbl_warnings = QLabel("")
        self.lbl_warnings.setWordWrap(True)
        self.lbl_warnings.setStyleSheet("color: #856404;")
        tl.addWidget(self.lbl_warnings)

        run_row = QHBoxLayout()
        self.btn_run = QPushButton("Run Gating")
        self.btn_run.clicked.connect(self._on_run_clicked)
        guard_while_busy(self.controller, self.btn_run)
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.setToolTip("Stop after the sample being gated is dropped, keeping every "
                                  "sample gated so far; Run Gating offers to resume.")
        self.btn_pause.clicked.connect(self._on_pause_clicked)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setToolTip("Stop and discard this run, keeping the results from before it.")
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        run_row.addWidget(self.btn_run)
        run_row.addWidget(self.btn_pause)
        run_row.addWidget(self.btn_cancel)
        run_row.addWidget(self.progress_bar)
        run_row.addWidget(self.lbl_status, stretch=1)
        tl.addLayout(run_row)
        tl.addWidget(QLabel("<b>Samples gated</b>"))
        self.summary_holder = QVBoxLayout()
        tl.addLayout(self.summary_holder)
        tl.addStretch()

        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        view_row = QHBoxLayout()
        self.chk_pooled = QCheckBox("Pooled over all gated samples")
        self.chk_pooled.setChecked(True)
        self.chk_pooled.setToolTip("A row of the population's gates with the stored boundaries "
                                   "over every gated sample's display events.")
        self.chk_pooled.toggled.connect(lambda _on: self._draw_tiles())
        view_row.addWidget(self.chk_pooled)
        self.chk_selected = QCheckBox("Selected sample")
        self.chk_selected.setChecked(True)
        self.chk_selected.setToolTip("A row of the population's gates with the own boundaries "
                                     "of the sample selected in the sample panel, once it has "
                                     "been gated.")
        self.chk_selected.toggled.connect(lambda _on: self._draw_tiles())
        view_row.addWidget(self.chk_selected)
        self.lbl_views = QLabel("")
        self.lbl_views.setStyleSheet("color: #555; font-style: italic;")
        view_row.addWidget(self.lbl_views, stretch=1)
        bl.addLayout(view_row)

        tree_panel = QWidget()
        tree_layout = QVBoxLayout(tree_panel)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        self.tree_filter = FilterBar(placeholder="Search populations…")
        tree_layout.addWidget(self.tree_filter)
        self.pop_tree = PopulationTree()
        self.pop_tree.setToolTip("Pick a population to see the gates drawn on it.")
        self.pop_tree.populationSelected.connect(lambda _g, _p: self._draw_tiles())
        tree_layout.addWidget(self.pop_tree, stretch=1)
        self.tree_filter.changed.connect(self._filter_tree)

        self.tile_scroll = QScrollArea()
        self.tile_scroll.setWidgetResizable(True)
        self.tile_scroll.setMinimumHeight(260)
        self.tile_canvas = QWidget()
        self.tile_grid = QGridLayout(self.tile_canvas)
        self.tile_grid.setSpacing(4)
        self.tile_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.tile_scroll.setWidget(self.tile_canvas)
        for bar in (self.tile_scroll.verticalScrollBar(), self.tile_scroll.horizontalScrollBar()):
            bar.valueChanged.connect(lambda _v: self._tile_timer.start())
            bar.rangeChanged.connect(lambda _lo, _hi: self._tile_timer.start())

        split = QSplitter(Qt.Horizontal)
        split.addWidget(tree_panel)
        split.addWidget(self.tile_scroll)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 4)
        split.setSizes([240, 760])
        bl.addWidget(split, stretch=1)

        self._page_gates = add_concertina_page(self.toolbox, gates_page, "Gate application")
        self._page_samples = add_concertina_page(self.toolbox, top, "Samples and gating")
        self._page_views = self.toolbox.addItem(bottom, "Gated views")
        self.toolbox.setCurrentIndex(self._page_samples)

    def _build_gates_page(self) -> QWidget:
        """Per-gate fixed or recalculated application and the drift check."""
        gates_page = QWidget()
        gl = QVBoxLayout(gates_page)
        opts = QHBoxLayout()
        opts.addWidget(QLabel("Drift limit:"))
        self.sb_drift = block_wheel(QDoubleSpinBox())
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
        self.cb_drift_action = block_wheel(QComboBox())
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
        self.gate_filter = FilterBar(
            [('all', 'All gates'), (ag_core.MODE_RECALCULATE, 'Recalculated gates'),
             (ag_core.MODE_FIXED, 'Fixed gates')],
            placeholder="Search gates, channels or algorithms…")
        self.gate_filter.changed.connect(lambda _t, _m: self._filter_gates())
        gl.addWidget(self.gate_filter)
        self.gate_tree = QTreeWidget()
        self.gate_tree.setColumnCount(6)
        self.gate_tree.setHeaderLabels(["Gate", "Type", "X axis", "Y axis", "Algorithm", "Application"])
        self.gate_tree.header().setSectionResizeMode(QHeaderView.Interactive)
        self.gate_tree.setColumnWidth(0, 220)
        self.gate_tree.setUniformRowHeights(True)
        self.gate_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.gate_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.gate_tree.customContextMenuRequested.connect(self._on_gate_menu)
        self.gate_tree.itemDoubleClicked.connect(self._on_gate_double_clicked)
        self.gate_tree.setToolTip("Double-click a gate to switch between fixed and recalculated; "
                                  "right-click to set the selected gates.")
        self.gate_tree.setMinimumHeight(160)
        gl.addWidget(self.gate_tree, stretch=1)
        return gates_page

    # ------------------------------------------------------------------

    def refresh(self):
        reason = self._readiness()
        # A run in progress stays on show (with Pause and Cancel) whatever the model does.
        ready = not reason or self.is_running()
        self.banner.setText(reason)
        self.banner.setVisible(bool(reason))
        self.content.setVisible(ready)
        self.sb_drift.blockSignals(True)
        self.sb_drift.setValue(float(self.state.drift_limit))
        self.sb_drift.blockSignals(False)
        idx = self.cb_drift_action.findData(self.state.drift_action)
        self.cb_drift_action.blockSignals(True)
        self.cb_drift_action.setCurrentIndex(max(idx, 0))
        self.cb_drift_action.blockSignals(False)
        self.chk_controls.blockSignals(True)
        self.chk_controls.setChecked(bool(self.state.include_controls))
        self.chk_controls.blockSignals(False)
        self.sb_display.blockSignals(True)
        self.sb_display.setValue(int(self.state.display_events))
        self.sb_display.blockSignals(False)
        if ready:
            self._populate_gate_tree()
            self._populate_samples()
            self._update_warnings()
            self._populate_views()
            self._draw_summary_table()

    # -- gate application --------------------------------------------------

    def _populate_gate_tree(self):
        """One row per gate, nested as in the hierarchy. The Application
        column shows each gate's mode; double-click or the context menu
        changes it."""
        gates = self.state.gate_defs()
        key = (id(self.state.applied_model), len(gates))
        if key == self._gates_for:
            for name in self._gate_items:
                self._show_mode(name)
            return
        self._gates_for = key
        self.gate_tree.clear()
        self._gate_items = {}
        antigens = _antigen_map(self.controller)

        def axis_text(ch):
            return antigens.get(ch, ch) if ch else ''

        for g in ag_core.ordered_gate_defs(gates):
            name = g['gate_name']
            item = QTreeWidgetItem([name, g.get('gate_type', ''), axis_text(g.get('gate_marker_x')),
                                    axis_text(g.get('gate_marker_y')), g.get('algorithm', ''), ''])
            item.setData(0, Qt.UserRole, name)
            parent = self._gate_items.get(g.get('parent_gate') or '')
            if parent is not None:
                parent.addChild(item)
            else:
                self.gate_tree.addTopLevelItem(item)
            self._gate_items[name] = item
            self._show_mode(name)
        self.gate_tree.expandAll()
        self._filter_gates()

    def _show_mode(self, name: str):
        item = self._gate_items.get(name)
        if item is None:
            return
        mode = self.state.modes.get(name, ag_core.MODE_FIXED)
        item.setText(5, MODE_LABELS.get(mode, mode))
        item.setForeground(5, QColor('#a05a00') if mode == ag_core.MODE_RECALCULATE
                           else QColor('#1e7d32'))

    def _set_modes(self, names, mode: str):
        for name in names:
            self.state.modes[name] = mode
            self._show_mode(name)
        self._filter_gates()

    def _on_gate_double_clicked(self, item, _column):
        name = item.data(0, Qt.UserRole)
        current = self.state.modes.get(name, ag_core.MODE_FIXED)
        self._set_modes([name], ag_core.MODE_FIXED if current == ag_core.MODE_RECALCULATE
                        else ag_core.MODE_RECALCULATE)

    def _on_gate_menu(self, pos):
        from PySide6.QtWidgets import QMenu
        names = [it.data(0, Qt.UserRole) for it in self.gate_tree.selectedItems()]
        if not names:
            item = self.gate_tree.itemAt(pos)
            names = [item.data(0, Qt.UserRole)] if item is not None else []
        if not names:
            return
        menu = QMenu(self.gate_tree)
        for mode in (ag_core.MODE_FIXED, ag_core.MODE_RECALCULATE):
            act = menu.addAction(f"{MODE_LABELS[mode]} ({len(names)} gate"
                                 f"{'s' if len(names) != 1 else ''})")
            act.triggered.connect(lambda _c=False, m=mode, n=list(names): self._set_modes(n, m))
        menu.exec(self.gate_tree.viewport().mapToGlobal(pos))

    def _set_all_modes(self, mode: str):
        self._set_modes([g['gate_name'] for g in self.state.gate_defs()], mode)

    def _filter_gates(self):
        """Show the gates matching the search and the "show" choice, with
        the gates above them."""
        text, mode = self.gate_filter.text(), self.gate_filter.mode()
        if not text and mode in ('', 'all'):
            for item in self._gate_items.values():
                item.setHidden(False)
            return
        visible = set()
        for name, item in self._gate_items.items():
            hit = text_matches(text, *(item.text(c) for c in range(6)))
            if mode not in ('', 'all'):
                hit = hit and self.state.modes.get(name, ag_core.MODE_FIXED) == mode
            if hit:
                node = item
                while node is not None and id(node) not in visible:
                    visible.add(id(node))
                    node = node.parent()
        for item in self._gate_items.values():
            item.setHidden(id(item) not in visible)

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
                name.setToolTip(key)
                self.sample_table.setItem(r, self.COL_SAMPLE, name)
                profiles = af_profile_names(self.controller, key)
                self.sample_table.setItem(r, self.COL_AF, _coloured_item(', '.join(profiles) or '—'))
            # Columns fit their contents once; the sample name is capped (and
            # elided in the middle) so the AF profiles stay in view.
            self.sample_table.resizeColumnsToContents()
            self.sample_table.setColumnWidth(
                self.COL_SAMPLE,
                min(self.sample_table.columnWidth(self.COL_SAMPLE), SAMPLE_COLUMN_MAX_WIDTH))
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
        if item.column() == self.COL_VALIDATE:
            self._draw_tiles()

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
        """Samples whose AF assignment was not represented in training, and
        results gated with an unmixing that has changed since."""
        lines = []
        st = self.state
        if st.summaries and st.run_unmixing and not self.is_running():
            try:
                changed = st.run_unmixing != unmixing_digest(self.controller)
            except Exception as exc:
                log.warning("could not read the unmixing: %s", exc)
                changed = False
            if changed:
                lines.append("The unmixing has changed since these results were gated (AutoSpectral tab, "
                             "AF profiles or spectral process): gate the samples again to update them.")
        training = ((st.applied_model or {}).get('af') or {}).get('sample_af_profiles') or {}
        if training:
            seen = {tuple(v or []) for v in training.values()}
            odd = [k for k in self._to_gate()
                   if tuple(af_profile_names(self.controller, k)) not in seen]
            if odd:
                lines.append(
                    "These samples use an AF profile assignment that none of the training "
                    "samples had, so their unmixing differs from the training data: "
                    + ", ".join(_sample_label(k) for k in odd[:10]) + (" …" if len(odd) > 10 else ""))
        self.lbl_warnings.setText("\n".join(lines))

    def _to_gate(self) -> list[str]:
        """Ticked samples, gating or validation, in table order."""
        if self.sample_table.rowCount() == 0:
            run = list(self.state.run_samples)
            return run + [k for k in self.state.validation_samples if k not in run]
        wanted = set(self._checked(self.COL_RUN)) | set(self._checked(self.COL_VALIDATE))
        return [self._row_key(r) for r in range(self.sample_table.rowCount())
                if self._row_key(r) in wanted]

    # ------------------------------------------------------------------

    def _ask_resume(self, n_done: int, n_left: int, compatible: bool) -> str | None:
        """'resume', 'restart' or None (do nothing) for an unfinished run."""
        box = QMessageBox(self)
        box.setWindowTitle("Unfinished gating run")
        if compatible:
            box.setText(f"The last run stopped with {n_done} sample(s) gated and {n_left} still "
                        "to go.\n\nResume gates only the ticked samples without results; "
                        "Restart discards the results so far and gates every ticked sample.")
            resume = box.addButton("Resume", QMessageBox.AcceptRole)
        else:
            box.setText(f"The last run stopped with {n_done} sample(s) gated, but the model, "
                        "its settings or the unmixing have changed since, so it cannot be resumed.\n\n"
                        "Restart discards the results so far and gates every ticked sample.")
            resume = None
        restart = box.addButton("Restart", QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if resume is not None and clicked is resume:
            return 'resume'
        if clicked is restart:
            return 'restart'
        return None

    def _on_run_clicked(self):
        st = self.state
        reason = self._readiness()
        if reason:
            QMessageBox.information(self, "Cannot gate now", reason)
            return
        keys = self._to_gate()
        if not keys:
            QMessageBox.information(self, "No samples", "Tick at least one sample to gate.")
            return
        channels = _experiment_channels(self.controller)
        missing = [ch for ch in ag_results.gated_channels(st.gate_defs()) if ch not in channels]
        if missing:
            QMessageBox.warning(self, "Channels missing",
                                "Gated channels not in this experiment: " + ", ".join(missing))
            return
        token = acquire_job(self.controller, 'Automated Gating: gating')
        if token is None:
            self.lbl_status.setText(self.controller.job_gate.busy_text())
            return
        self._job_token = token
        markers = _marker_channels(self.controller)
        model_sha = st.model_sha
        saved_markers = st.marker_channels
        st.marker_channels = markers
        fingerprint = run_fingerprint(st, model_sha)
        st.marker_channels = saved_markers
        unmixing = unmixing_digest(self.controller)
        to_gate = list(keys)
        resume = False
        if st.pending_samples and st.summaries:
            same_unmixing = not st.run_unmixing or st.run_unmixing == unmixing
            choice = self._ask_resume(len(st.summaries), len(st.pending_samples),
                                      fingerprint == st.run_fingerprint and same_unmixing)
            if choice is None:
                token.release()
                return
            resume = choice == 'resume'
            if resume:
                to_gate = [k for k in keys if k not in st.summaries]
                if not to_gate:
                    token.release()
                    st.pending_samples = []
                    self.lbl_status.setText("Every ticked sample is already gated.")
                    self.gating_finished.emit('done')
                    return
        try:
            snap = snapshot_unmix_state(self.controller, to_gate)
        except Exception as exc:
            token.release()
            QMessageBox.warning(self, "Cannot gate", str(exc))
            return
        self._before_run = {name: (dict(getattr(st, name)) if isinstance(getattr(st, name), dict)
                                   else deepcopy(getattr(st, name))) for name in (
            'run_samples', 'summaries', 'load_info', 'display', 'stats', 'trend', 'fsea',
            'fsea_trend', 'marker_channels',
            'results_model_sha', 'pending_samples', 'run_fingerprint', 'run_unmixing', 'run_transforms')}
        st.run_samples = keys
        if resume:
            keep = set(keys)
            st.summaries = {k: v for k, v in st.summaries.items() if k in keep}
            st.load_info = {k: v for k, v in st.load_info.items() if k in keep}
            st.display = {k: v for k, v in st.display.items() if k in keep}
        else:
            st.summaries = {}
            st.load_info = {}
            st.display = {}
        st.stats = {}
        st.trend = {}
        st.fsea = {}
        st.fsea_trend = {}
        st.pca = {}
        st.predictive = {}
        st.marker_channels = markers
        st.results_model_sha = model_sha
        st.run_fingerprint = fingerprint
        st.run_unmixing = unmixing
        st.run_transforms = deepcopy(st.transform_params)
        st.pending_samples = list(to_gate)
        self._run_keys = to_gate
        self._pool_cache = None

        transforms = self._gating_transforms()
        worker = _GatingWorker(
            self.controller.experiment_dir, to_gate, snap,
            deepcopy(st.gate_defs()), deepcopy(st.reference_boundaries()), dict(st.modes),
            st.drift_limit, st.drift_action, channels, deepcopy(transforms),
            list(st.marker_channels), st.display_events,
            display_dir=self._display_dir(), fingerprint=fingerprint, parent=self,
        )
        # Slots ignore a worker that has been abandoned (see abandon_run).
        worker.progress.connect(lambda msg, w=worker: w is self._worker and self._on_progress(msg))
        worker.sample_done.connect(
            lambda key, payload, w=worker: w is self._worker and self._on_sample_done(key, payload))
        worker.finished.connect(
            lambda outcome, msg, w=worker: w is self._worker and self._on_finished(outcome, msg))
        self._worker = worker
        self._set_running(True)
        self.progress_bar.setRange(0, len(keys))
        self.progress_bar.setValue(len(keys) - len(to_gate))
        self.progress_bar.setVisible(True)
        worker.start()

    def _set_running(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.btn_pause.setEnabled(running)
        self.btn_cancel.setEnabled(running)

    def is_running(self) -> bool:
        return self._worker is not None

    def _on_pause_clicked(self):
        if self._worker is not None:
            self._worker.pause()
            self.btn_pause.setEnabled(False)
            self.lbl_status.setText("Pausing …")

    def _on_cancel_clicked(self):
        if self._worker is not None:
            self._worker.abort()
            self.btn_pause.setEnabled(False)
            self.btn_cancel.setEnabled(False)
            self.lbl_status.setText("Cancelling …")

    def abandon_run(self):
        """Stop a run whose results no longer belong to the state (the
        experiment is changing); its late signals are ignored."""
        if self._worker is None:
            return
        self._worker.abort()
        self._worker = None
        self._job_token.release()
        self._before_run = None
        self._set_running(False)
        self.progress_bar.setVisible(False)

    def _on_progress(self, msg: str):
        self.lbl_status.setText(msg)
        _status(self.bus, msg)

    def _on_sample_done(self, key: str, payload: object):
        st = self.state
        st.summaries[key] = payload['summary']
        st.load_info[key] = payload['load_info']
        st.display[key] = payload['display']
        if key in st.pending_samples:
            st.pending_samples.remove(key)
        self.progress_bar.setValue(self.progress_bar.value() + 1)

    def _on_finished(self, outcome: str, message: str):
        """Finish a run: keep it ('done'), keep what was gated and remember
        what is left ('paused' and 'failed'), or put the results from before
        the run back ('cancelled')."""
        self._worker = None
        self._job_token.release()
        self._set_running(False)
        self.progress_bar.setVisible(False)
        st = self.state
        if outcome == 'cancelled' and self._before_run is not None:
            for name, value in self._before_run.items():
                setattr(st, name, value)
            text = "Cancelled; the results from before this run are kept."
        else:
            n = len(st.summaries)
            flagged = sum(1 for k in st.summaries if st.flags(k))
            flag_text = f"; {flagged} flagged" if flagged else ''
            if outcome == 'done':
                st.pending_samples = []
                text = f"Gated {n} sample(s){flag_text}."
            elif outcome == 'paused':
                text = (f"Paused with {n} sample(s) gated{flag_text} and "
                        f"{len(st.pending_samples)} to go. Run Gating to resume or restart.")
            else:
                text = (f"Gating stopped: {message} {n} sample(s) gated, "
                        f"{len(st.pending_samples)} to go. Run Gating to resume or restart.")
        self._before_run = None
        self._pool_cache = None
        self.lbl_status.setText(text)
        _status(self.bus, text)
        self._populate_views()
        self._draw_summary_table()
        if st.summaries and outcome != 'cancelled':
            self.toolbox.setCurrentIndex(self._page_views)
        if outcome != 'cancelled':
            self.gating_finished.emit(outcome)

    # ------------------------------------------------------------------
    # Gated views
    # ------------------------------------------------------------------

    def _populate_views(self):
        """Fill the population tree (when the model changed) and draw the
        selected population's gates."""
        gates = self.state.gate_defs()
        key = (id(self.state.applied_model), len(gates))
        if key != self._tree_for:
            import ag_review
            self._tree_for = key
            self.pop_tree.set_nodes(ag_review.population_tree(gates) if gates else [],
                                    expand_depth=2)
            if self.pop_tree.current_population() is None and gates:
                self.pop_tree.select_population('', 'root')
            self._filter_tree(self.tree_filter.text(), '')
        self._draw_tiles()

    def _filter_tree(self, text: str, _mode: str):
        if not text:
            self.pop_tree.set_filter(None)
            return
        labels = {}
        for g in self.state.gate_defs():
            for pop, info in (g.get('populations') or {}).items():
                labels[(g['gate_name'], pop)] = f"{g['gate_name']} {(info or {}).get('label', pop)}"
        self.pop_tree.set_filter({k for k, lbl in labels.items() if text_matches(text, lbl)})

    def _validation_shown(self) -> list[str]:
        st = self.state
        return [k for k in st.validation_samples if k in st.display and k in st.summaries]

    def _selected_shown(self) -> str | None:
        """The gated sample selected in the main window's sample panel,
        when it has plots and is not already shown as a validation sample."""
        if not self.chk_selected.isChecked():
            return None
        current = getattr(self.controller, 'current_sample_path', None)
        if not current:
            return None
        current = str(current).replace('\\', '/')
        st = self.state
        for k in st.display:
            if str(k).replace('\\', '/') == current:
                if k in st.summaries and k not in st.validation_samples:
                    return k
                return None
        return None

    def selected_sample_changed(self):
        """The main window selected another sample: redraw the gated views
        when they show the selected sample."""
        if self.chk_selected.isChecked() and self.toolbox.currentIndex() == self._page_views:
            self._draw_tiles()

    def _pooled(self, samples: list[str], gate_def: dict):
        """Pooled display events of *samples* (concatenated once) with
        *gate_def*'s parent mask."""
        st = self.state
        have = tuple(k for k in samples if k in st.display)
        if not have:
            return None, None, []
        if self._pool_cache is None or self._pool_cache[0] != have:
            self._pool_cache = (have, np.concatenate([st.display[k]['events'] for k in have]),
                                st.display[have[0]]['channels'])
        _keys, events, channels = self._pool_cache
        parents = []
        for k in have:
            m = ag_results.display_parent_mask(st.display[k], gate_def['gate_name'], gate_def)
            parents.append(m if m is not None else np.zeros(len(st.display[k]['events']), dtype=bool))
        return events, np.concatenate(parents), channels

    def _draw_tiles(self):
        """Lay out the gated views of the selected population: a heading per
        row (pooled, the selected sample, validation samples) and a tile per
        gate, each a placeholder until it is scrolled near
        (_build_visible_tiles)."""
        self._tile_timer.stop()
        self._tiles = []
        _clear_layout(self.tile_grid)
        st = self.state
        gates = st.gate_defs()
        node = self.pop_tree.current_population() or ('', 'root')
        if not gates or not st.display:
            note = QLabel("Gate some samples to see their gates here."
                          if not st.summaries else
                          "The plots of these results are not available; run gating again "
                          "to see them.")
            note.setWordWrap(True)
            self.tile_grid.addWidget(note, 0, 0)
            self.lbl_views.setText("")
            return
        by_name = ag_core.gates_by_name(gates)
        shown = _gates_on(gates, node)
        highlight = None
        if not shown and node[0] in by_name:
            # A population without gates of its own: show the gate that makes it.
            shown = [by_name[node[0]]]
            highlight = node[1]
        transforms = self._result_transforms()
        labels = build_display_label_map(
            _experiment_channels(self.controller),
            self.controller.experiment.process.get('spectral_model') or [])
        validation = self._validation_shown()
        missing = [k for k in st.validation_samples if k not in validation]
        self.lbl_views.setText(
            f"{len(st.display)} sample(s) with plots; {len(validation)} validation sample(s) shown"
            + (f" ({len(missing)} ticked but not gated)" if missing else ''))
        selected = self._selected_shown()
        rows = ([('pooled', None)] if self.chk_pooled.isChecked() else []) + \
            ([('selected', selected)] if selected else []) + \
            [('validation', k) for k in validation]
        self._tile_ctx = {'transforms': transforms, 'labels': labels, 'highlight': highlight,
                          'by_name': by_name}
        tile_w, tile_h = _gate_tile_size()
        r = 0
        for line, (kind, key) in enumerate(rows):
            head = QLabel("<b>All gated samples</b> <small>(stored boundaries, % of pooled "
                          "display events)</small>" if kind == 'pooled' else
                          f"<b>{_sample_label(key)}</b> <small>[{kind}"
                          + (f"; {st.sample_groups[key]}" if st.sample_groups.get(key) else '')
                          + "]</small>")
            # Space above each heading after the first, so it sits with the
            # row of plots it names (below it), not the row above.
            head.setContentsMargins(0, 18 if r else 0, 0, 0)
            self.tile_grid.addWidget(head, r, 0, 1, max(1, len(shown)),
                                     Qt.AlignLeft | Qt.AlignBottom)
            r += 1
            for c, g in enumerate(shown):
                holder = QWidget()
                holder.setFixedSize(tile_w, tile_h)
                self.tile_grid.addWidget(holder, r, c)
                self._tiles.append({'line': line, 'row': r, 'col': c, 'kind': kind, 'key': key,
                                    'gate': g, 'holder': holder, 'built': False})
            r += 1
        if not rows:
            self.tile_grid.addWidget(QLabel("Tick 'Pooled', select a gated sample in the sample "
                                            "panel or tick some validation samples."), 0, 0)
        # The tiles on screen now; the others as the views are scrolled.
        self._build_visible_tiles()
        self._tile_timer.start()

    def _build_tile(self, tile: dict):
        """Build one gate tile in place of its placeholder."""
        st, ctx = self.state, self._tile_ctx
        g, kind, key = tile['gate'], tile['kind'], tile['key']
        name = g['gate_name']
        eff = ag_core.effective_gate_def(g, ctx['by_name'])
        if kind == 'pooled':
            events, parent, channels = self._pooled(st.gated_samples(), eff)
            boundary = st.reference_boundaries().get(name, {})
            fractions = ag_report.boundary_fractions(eff, boundary, events, channels,
                                                     parent, ctx['transforms'])
            mode = st.modes.get(name, ag_core.MODE_FIXED)
            n_flag = sum(1 for k in st.summaries
                         if ((st.summaries[k].get('status') or {}).get(name) or {}).get('flag'))
            title = f"<b>{name}</b> <small>[{MODE_LABELS[mode]}]</small>"
            footer = f"{n_flag} sample(s) flagged at this gate" if n_flag else ''
        else:
            disp = st.display.get(key) or {}
            channels = disp.get('channels') or []
            events = disp.get('events')
            parent = ag_results.display_parent_mask(disp, name, eff)
            summ = st.summaries.get(key) or {}
            boundary = (summ.get('boundaries') or {}).get(name, {})
            fractions = {row['population']: row['fraction_of_parent']
                         for row in summ.get('populations', []) if row['gate'] == name}
            status = (summ.get('status') or {}).get(name) or {}
            title = f"<b>{name}</b> <small>[{status.get('source') or '—'}]</small>"
            footer = status.get('message') if status.get('flag') else ''
        widget = _make_gate_tile(eff, boundary, events, channels, parent, ctx['transforms'],
                                 ctx['labels'], title, fractions=fractions,
                                 highlight=ctx['highlight'], footer=footer)
        holder = tile['holder']
        self.tile_grid.removeWidget(holder)
        holder.setParent(None)
        holder.deleteLater()
        tile['holder'] = None
        tile['built'] = True
        self.tile_grid.addWidget(widget, tile['row'], tile['col'])

    def _build_visible_tiles(self):
        """Build the tiles within a screen of the visible part of the gated
        views, in either direction; while the page is not on screen, those
        that would fill its first screens."""
        pending = [t for t in self._tiles if not t['built']]
        if not pending:
            return
        if self.tile_scroll.isVisible():
            self.tile_grid.activate()
            last = pending[-1]['holder']
            if len(pending) > 1 and last.x() <= 0 and last.y() <= 0:
                return    # not laid out yet; the scroll ranges change once it is
            vp = self.tile_scroll.viewport()
            w, h = max(vp.width(), 1), max(vp.height(), 1)
            x0 = self.tile_scroll.horizontalScrollBar().value()
            y0 = self.tile_scroll.verticalScrollBar().value()
            wanted = []
            for t in pending:
                box = t['holder'].geometry()
                if box.right() >= x0 - w and box.left() <= x0 + 2 * w \
                        and box.bottom() >= y0 - h and box.top() <= y0 + 2 * h:
                    wanted.append(t)
        else:
            tile_w, tile_h = _gate_tile_size()
            cols = 2 * max(1, 1400 // tile_w)
            lines = 2 * max(1, 800 // tile_h)
            wanted = [t for t in pending if t['col'] < cols and t['line'] < lines]
        for t in wanted:
            self._build_tile(t)

    def showEvent(self, event):
        super().showEvent(event)
        if self._tiles:
            self._tile_timer.start()

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


FIGURE_OPTIONS_SCOPE = 'automated_gating'
# Figure options key of each figure shown on the Figures page under a title
# rather than a key (the sample PCA and predictive analysis).
_EXTRA_FIGURE_KEYS = {"Sample PCA": 'sample_pca', "Predictive: ROC": 'pred_roc',
                      "Predictive: features": 'pred_features',
                      "Predictive: panel size": 'pred_panel'}


def figure_options_ref(controller, key: str | None):
    """The figure options of Automated Gating figure *key* in the experiment
    the controller shows (figure_options), or None."""
    from honeychrome.view_components import figure_options as fo
    store = fo.store_for(controller) if key else None
    return fo.FigureOptionsRef(store, FIGURE_OPTIONS_SCOPE, key) if store is not None else None


def report_figure_key(item_key: str) -> str:
    """The figure options key of a report item: its figure on the Figures
    page (e.g. 'freq_volcano' for 'freq_volcano:B vs A'), or one key for
    each kind of per-gate, per-population and enrichment figure."""
    parts = str(item_key).split(':')
    head = parts[0]
    if head == 'enrichment':
        if len(parts) == 1 or parts[1] == 'trend':
            return 'enrichment_dot'
        return 'enrichment_patterns' if parts[1] == 'patterns' else 'enrichment_running_sum'
    return head


class _ProgressCallWorker(QThread):
    """Run fn(report) off the main thread, where report(fraction, message)
    emits progress; emits finished(ok, error, result)."""

    progress = Signal(float, str)
    finished = Signal(bool, str, object)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            result = self._fn(lambda fraction, message: self.progress.emit(fraction, message))
        except Exception as exc:
            log.exception("background task failed")
            self.finished.emit(False, str(exc), None)
            return
        self.finished.emit(True, '', result)


def write_report(*, report_dir: Path, stem: str, items: list, title_lines: list[str],
                 settings_text: str, summaries: dict, samples: list[str], groups: dict,
                 antigens: dict, progress, cancelled) -> bool:
    """Write the report into *report_dir*: population_statistics.csv and
    activation_statistics.csv (CSV only, listed in the PDF), a PNG and/or
    CSV per item (items as from ag_report.gate_pages), one PDF of them all
    with tables over ag_report.PDF_MAX_ROWS rows or PDF_MAX_COLUMNS columns
    left as CSV, and settings.txt.

    progress(step, text) is called before each of the len(items) + 3 steps
    and cancelled() is checked between them. Returns False when cancelled.
    Worker-safe: plain data and canvas-less matplotlib figures only.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    import drc_report

    report_dir.mkdir(parents=True, exist_ok=True)
    progress(0, "Writing population_statistics.csv")
    pops = ag_results.population_export(summaries, samples, groups)
    pops.to_csv(report_dir / 'population_statistics.csv', index=False)
    if cancelled():
        return False
    progress(1, "Writing activation_statistics.csv")
    n_markers = ag_results.write_marker_export(report_dir / 'activation_statistics.csv',
                                               summaries, samples, antigens, groups)
    if cancelled():
        return False
    step = 2
    with PdfPages(str(report_dir / f"{stem}_report.pdf")) as pdf:
        drc_report.add_title_page(pdf, "Honeychrome Automated Gating Report", title_lines)
        drc_report.add_table_note(pdf, "Tables in CSV files only", [
            f"population_statistics.csv: {len(pops):,} rows, one per sample and population "
            "(counts, parent counts and fractions).",
            f"activation_statistics.csv: {n_markers:,} rows, one per sample, population and "
            "marker (median and quartiles in transformed units)."])
        for section in ag_report.SECTIONS:
            section_items = [it for it in items if it.tab == section]
            if not section_items:
                continue
            drc_report.add_section_divider(pdf, section)
            subfolder = report_dir / drc_report.sanitize_filename(section)
            subfolder.mkdir(parents=True, exist_ok=True)
            for it in section_items:
                if cancelled():
                    return False
                progress(step, f"Adding {it.label}")
                drc_report.export_report_item(it, subfolder, pdf,
                                              max_pdf_rows=ag_report.PDF_MAX_ROWS,
                                              max_pdf_columns=ag_report.PDF_MAX_COLUMNS)
                step += 1
    progress(step, "Writing settings document")
    (report_dir / 'settings.txt').write_text(settings_text, encoding='utf-8')
    progress(step + 1, "Done")
    return True


class _ReportWorker(QThread):
    """Run write_report off the main thread; emits progress(step, text) and
    finished(ok, error, cancelled). cancel() stops it between steps."""

    progress = Signal(int, str)
    finished = Signal(bool, str, bool)

    def __init__(self, job: dict, parent=None):
        super().__init__(parent)
        self._job = job
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        try:
            complete = write_report(progress=self.progress.emit,
                                    cancelled=self._cancel.is_set, **self._job)
        except Exception as exc:
            log.exception("report generation failed")
            self.finished.emit(False, str(exc), False)
            return
        self.finished.emit(True, '', not complete)


def _comparison_label(base: str, other: str) -> str:
    return f"{other} vs {base}"


def statistics_design(state: 'GatingRunState', samples: list[str]) -> dict:
    """The design the statistics (and the enrichment) use for *samples*:
    {'groups': {sample: group}, 'pairing': {sample: level} or None,
    'covariates': adjustment covariates (DataFrame) or None, 'numeric':
    covariates entered as numbers, 'missing_pairing': labels of samples
    without a pairing value}. A sample without a pairing value is unpaired:
    it gets a pairing level of its own."""
    st = state
    groups = {k: st.sample_groups[k] for k in samples}
    pairing, missing = None, []
    if st.pairing:
        vals = st.covariates.get(st.pairing) or {}
        missing = [_sample_label(k) for k in samples if ds.is_missing_value(vals.get(k))]
        pairing = {k: (f"__unpaired_{k}" if ds.is_missing_value(vals.get(k))
                       else str(vals[k]).strip()) for k in samples}
    covariates = None
    adjust = [c for c in st.adjust_covariates if c in st.covariates and c != st.pairing]
    if adjust:
        covariates = pd.DataFrame(
            {c: [str((st.covariates.get(c) or {}).get(k, '')) for k in samples] for c in adjust},
            index=samples)
    return {'groups': groups, 'pairing': pairing, 'covariates': covariates,
            'numeric': [c for c in st.numeric_covariates if c in adjust],
            'missing_pairing': missing}


def _comparison_view(results: pd.DataFrame | None, label: str) -> pd.DataFrame | None:
    if results is None or 'comparison' not in results.columns:
        return results
    return results[results['comparison'] == label].drop(columns=['comparison']).reset_index(drop=True)


def build_result_figures(stats: dict, base: str, other: str, pval_threshold: float,
                         fc_threshold: float, marker_threshold: float, fdr_scope: str,
                         is_dark: bool, antigens: dict, pop_labels: dict,
                         group_colours: dict | None = None) -> dict:
    """Every results figure for one comparison. Pure matplotlib (no canvas),
    so it runs on a worker thread. *group_colours* ({group: colour}) colours
    the group bars and direction ramps.

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
                        'results': results, 'make': lambda: maker(**kwargs)}
        except Exception as exc:
            log.exception("figure %s failed", key)
            out[key] = {'fig': None, 'title': tab_title, 'error': str(exc), 'results': None}

    def _rows(df):
        return df.loc[[s for s in keep if s in df.index]] if df is not None else None

    counts_x = ('log2 odds ratio' if stats.get('counts_method', 'wald') == 'moderated'
                else 'log2 fold change')
    specs = (
        ('freq', 'Frequency', stats.get('freq'), stats.get('freq_values'),
         '% of parent', 'log2 odds ratio', fc_threshold),
        ('counts', 'Counts', stats.get('counts'), stats.get('counts_values'),
         'Events', counts_x, fc_threshold),
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
             feature_labels={f: pop_labels.get(f, f) for f in view['feature']},
             group_colours=group_colours)
        _add(f'{key}_volcano', f"{name} volcano", dp.make_volcano_figure, results=view,
             results_df=view, title=f"{name} volcano: {label}", pval_threshold=pval_threshold,
             fc_threshold=thr, is_dark=is_dark, fdr_scope=fdr_scope, feature_labels=labels,
             x_label=x_label, run_label=run_label, group_a=base, group_b=other,
             group_colours=group_colours)

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
             feature_labels=mlabels, x_label='Δ median (transformed units)', run_label=run_label,
             group_a=base, group_b=other, group_colours=group_colours)
    return out


# Tested values behind each kind of trend result (keys of a trend run).
_TREND_DATA_KEYS = {'freq': 'freq_data', 'counts': 'counts_data', 'markers': 'marker_values'}
_TREND_KIND_NAMES = {'freq': 'Frequency', 'counts': 'Counts', 'markers': 'Marker'}


def _trend_value_label(trend: dict, kind: str) -> str:
    """Axis label of the values a trend of *kind* was tested on."""
    if kind == 'markers':
        return 'median (transformed units)'
    if kind == 'counts' and trend.get('counts_reference') != 'parent':
        return 'log2 odds within the anchor population'
    return 'log2 odds within the parent'


def _trend_feature_labels(table: pd.DataFrame, kind: str, pop_labels: dict,
                          antigens: dict) -> list[str]:
    """Display label of each row of a trend table."""
    if kind == 'markers' and {'cluster', 'channel'}.issubset(table.columns):
        return [f"{r['cluster']} {antigens.get(r['channel'], r['channel'])}"
                for _i, r in table.iterrows()]
    return [pop_labels.get(f, f) for f in table['feature'].astype(str)]


def build_trend_figures(trend: dict, is_dark: bool, antigens: dict, pop_labels: dict,
                        group_colours: dict | None = None) -> dict:
    """The trend volcano and trajectory heatmap of each kind of result of
    a trend run. Pure matplotlib (no canvas), so it runs on a worker
    thread. Each figure carries ``_trend_kind`` (freq, counts or markers)
    for click-through to the per-feature trend plot.

    Returns {key: {'fig', 'title', 'error', 'results', 'make'}} as
    build_result_figures does.
    """
    from honeychrome.view_components import trend_plots as tp

    settings = trend.get('settings') or {}
    variable = trend.get('variable') or settings.get('variable', 'time')
    unit = trend.get('unit') or settings.get('unit', '')
    scale = settings.get('scale', 'linear')
    groups = list(settings.get('groups') or [])
    what = (f" ({groups[1]} − {groups[0]})"
            if settings.get('interaction') and len(groups) == 2 else '')
    run_label = f"Trend with {variable}"
    out: dict = {}

    def _add(key, tab_title, kind, maker, **kwargs):
        def make():
            fig = maker(**kwargs)
            fig._trend_kind = kind
            return fig
        try:
            out[key] = {'fig': make(), 'title': tab_title, 'error': None, 'results': None,
                        'make': make}
        except Exception as exc:
            log.exception("figure %s failed", key)
            out[key] = {'fig': None, 'title': tab_title, 'error': str(exc), 'results': None}

    for kind, name in _TREND_KIND_NAMES.items():
        res = trend.get(kind)
        if res is None:
            continue
        labels = _trend_feature_labels(res.table, kind, pop_labels, antigens)
        per = tp.slope_label(variable, unit, scale)
        amount = 'median' if kind == 'markers' else 'log2 odds'
        _add(f'trend_{kind}_volcano', f"{name} trend volcano", kind,
             tp.make_trend_volcano_figure, table=res.table,
             title=f"{name}: trends with {variable}{what}", x_label=f"Slope ({amount} {per})",
             feature_labels=labels, pval_threshold=settings.get('pval_threshold', 0.05),
             call=settings.get('call', 'either'), is_dark=is_dark, run_label=run_label)
        _add(f'trend_{kind}_heatmap', f"{name} trajectories", kind,
             tp.make_trajectory_heatmap_figure, table=res.table, grid=res.grid,
             curves=res.curves, title=f"{name}: fitted curves of the called features{what}",
             variable=variable, unit=unit, scale=scale, feature_labels=labels,
             patterns=res.settings.get('patterns'), is_dark=is_dark, run_label=run_label)
        _add(f'trend_{kind}_patterns', f"{name} patterns", kind,
             tp.make_pattern_panel_figure, table=res.table, grid=res.grid, curves=res.curves,
             patterns=res.settings.get('patterns') or [],
             title=f"{name}: shape patterns of the called features{what}", variable=variable,
             unit=unit, scale=scale, is_dark=is_dark, run_label=run_label)
    age = trend.get('age_model')
    if age:
        groups = dict(zip(trend.get('samples') or [], trend.get('group_vec') or []))
        groups.update(age.get('groups') or {})
        model_group = (age.get('summary') or {}).get('group', '')
        _add('trend_age_model', "Immune age", None, tp.make_age_model_figure, result=age,
             variable=variable, unit=unit, groups=groups, group_colours=group_colours,
             title=f"Immune age model: {variable} from the immune profile ({model_group})",
             is_dark=is_dark, run_label=run_label)
        imp = age['importance']
        keys = ({f: k for f, k in zip(imp['feature'].astype(str), imp['key'].astype(str)) if k}
                if 'key' in imp.columns else None)
        _add('trend_age_features', "Immune age: features", None, tp.make_age_features_figure,
             result=age, variable=variable, is_dark=is_dark, run_label=run_label, keys=keys)
    return out


def _is_dark_palette() -> bool:
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    palette = app.palette() if app is not None else QPalette()
    return palette.color(QPalette.ColorRole.Base).value() < 128


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_results_csv(folder: Path, stem: str, timestamp: str, summaries: dict,
                      samples: list[str], groups: dict, labels: dict,
                      antigens: dict) -> list[Path]:
    """Write the gated samples' population counts and marker medians as
    ``<stem>_<timestamp>_populations.csv`` and ``…_marker_medians.csv``;
    returns the paths. Worker-safe (plain data only)."""
    folder.mkdir(parents=True, exist_ok=True)
    pops = ag_results.population_export(summaries, samples, groups)
    if not pops.empty:
        pops.insert(2, 'population_label', [labels.get(k, k) for k in pops['key']])
    markers = ag_results.marker_export(summaries, samples, antigens, groups)
    if not markers.empty:
        markers.insert(3, 'population_label', [labels.get(k, k) for k in markers['population']])
    out = []
    for frame, suffix in ((pops, 'populations'), (markers, 'marker_medians')):
        path = folder / f"{stem}_{timestamp}_{suffix}.csv"
        frame.to_csv(path, index=False)
        out.append(path)
    return out


def write_trend_csv(folder: Path, stem: str, timestamp: str, trend: dict, labels: dict,
                    antigens: dict) -> list[Path]:
    """Write the trend run's tables as ``<stem>_<timestamp>_<table>.csv``
    (trend_frequencies, trend_counts, trend_markers); returns the paths.
    Worker-safe (plain data only)."""
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for name, frame in ag_results.trend_tables(trend, labels, antigens).items():
        path = folder / f"{stem}_{timestamp}_{name}.csv"
        frame.to_csv(path, index=False)
        out.append(path)
    return out


def write_statistics_csv(folder: Path, stem: str, timestamp: str, stats: dict, labels: dict,
                         antigens: dict) -> list[Path]:
    """Write the statistics run's tables as ``<stem>_<timestamp>_statistics_
    <table>.csv``; returns the paths. Worker-safe (plain data only)."""
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for name, frame in ag_results.statistics_tables(stats, labels, antigens).items():
        path = folder / f"{stem}_{timestamp}_statistics_{name}.csv"
        frame.to_csv(path, index=name.startswith('values'))
        out.append(path)
    return out


# ---------------------------------------------------------------------------
# Results tab
# ---------------------------------------------------------------------------

class ResultsTab(QWidget):
    """Groups and covariates, statistics, figures, population browser, export."""

    stats_finished = Signal()
    trend_finished = Signal()
    report_populations_changed = Signal()
    sample_analysis_changed = Signal()

    def __init__(self, state: GatingRunState, bus, controller, result_transforms=None,
                 parent=None):
        super().__init__(parent)
        self.state = state
        self.bus = bus
        self.controller = controller
        self._result_transforms = result_transforms or (
            lambda: self.controller.unmixed_transformations or {})
        self._stats_worker: _CallWorker | None = None
        self._figure_worker: _CallWorker | None = None
        self._figure_pending = False
        self._predictive_worker: _CallWorker | None = None
        self._report_worker: _ReportWorker | None = None
        # Populations page: the background gathering of the shown
        # population, and pooled group panels kept for the current results.
        self._pop_worker: _CallWorker | None = None
        self._pop_cache: dict = {}
        self._pop_cache_lock = threading.Lock()
        self._popout_workers: list = []
        self._filling = False
        self._volcano_tables: dict = {}
        self._row_figures: dict = {}
        self._figures_for = None
        self._focus_figure = None
        self._shown_population = None
        # Automatic Suggest Groupings: the sample list it last ran for (it
        # runs once per list) and the state to restore on Undo.
        self._auto_groupings_key = None
        self._auto_groupings_undo = None
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)
        outer.addWidget(HelpToggleWidget(text=ag_help_texts.AG_RESULTS, scrolling=True))
        cmp_row = QHBoxLayout()
        cmp_row.addWidget(QLabel("<b>Comparison:</b>"))
        self.cb_comparison = QComboBox()
        self.cb_comparison.currentIndexChanged.connect(self._on_comparison_changed)
        cmp_row.addWidget(self.cb_comparison, stretch=1)
        outer.addLayout(cmp_row)
        # One section open at a time; each page scrolls on its own.
        self.toolbox = make_concertina()
        outer.addWidget(self.toolbox, stretch=1)

        self.groups_page = QWidget()
        self._build_groups_page()
        add_concertina_page(self.toolbox, self.groups_page, "Groups")
        self.stats_page = QWidget()
        self._build_stats_page()
        add_concertina_page(self.toolbox, self.stats_page, "Statistics")
        self.explore_page = QWidget()
        self._build_explore_page()
        add_concertina_page(self.toolbox, self.explore_page, "Sample PCA and prediction")
        self.figures_tabs = QTabWidget()
        self.figures_tabs.setMinimumHeight(400)
        self._page_figures = add_concertina_page(self.toolbox, self.figures_tabs, "Figures")
        self.browser_page = QWidget()
        self._build_browser_page()
        self._page_browser = add_concertina_page(self.toolbox, self.browser_page, "Populations")
        self.export_page = QWidget()
        self._build_export_page()
        add_concertina_page(self.toolbox, self.export_page, "Export")

    # ------------------------------------------------------------------
    # Groups
    # ------------------------------------------------------------------

    def _build_groups_page(self):
        layout = QVBoxLayout(self.groups_page)
        top = QHBoxLayout()

        gbox = QGroupBox("Groups")
        gl = QVBoxLayout(gbox)
        self.groups_table = QTableWidget(0, 3)
        self.groups_table.setHorizontalHeaderLabels(["Group", "Sample-name pattern (regex)", "Colour"])
        groups_header = self.groups_table.horizontalHeader()
        groups_header.setSectionResizeMode(QHeaderView.Interactive)
        groups_header.setSectionResizeMode(1, QHeaderView.Stretch)
        groups_header.setStretchLastSection(False)
        self.groups_table.setColumnWidth(0, 180)
        self.groups_table.setColumnWidth(2, 60)
        self.groups_table.verticalHeader().setVisible(False)
        self.groups_table.itemChanged.connect(self._on_group_item_changed)
        self.groups_table.cellDoubleClicked.connect(self._on_group_cell_double_clicked)
        gl.addWidget(self.groups_table)
        row = QHBoxLayout()
        for text, slot in (("Add group", self._on_add_group), ("Remove group", self._on_remove_group),
                           ("Assign by pattern", self._on_assign_by_pattern),
                           ("Suggest groupings…", self._on_suggest_groupings)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        for text, step, tip in (("▲", -1, "Move the selected group up."),
                                ("▼", 1, "Move the selected group down.")):
            b = QPushButton(text)
            b.setFixedWidth(32)
            b.setToolTip(tip + " The group order is the order used in the Statistics "
                         "choices, comparisons between every pair (the earlier group is "
                         "the baseline) and every figure.")
            b.clicked.connect(lambda _c=False, s=step: self._on_move_group(s))
            row.addWidget(b)
        gl.addLayout(row)
        top.addWidget(gbox, stretch=1)

        cbox = QGroupBox("Covariates")
        cl = QVBoxLayout(cbox)
        note = QLabel("Per-sample values, e.g. donor (for pairing) or age and sex (for "
                      "adjustment). Tick Numeric under a column's title to enter it as a "
                      "continuous number (age, CRP); unticked columns are categories.")
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
        trow = QHBoxLayout()
        b_import = QPushButton("Import table…")
        b_import.setToolTip(
            "Read groups and covariates from a CSV or tab-separated file with one row per "
            "sample: a 'sample' column (the file name, with or without extension, or a sample "
            "ID found inside the file name), an optional group column, and one column per "
            "covariate (e.g. donor, age, sex). The matches are shown for review before "
            "anything changes.")
        b_import.clicked.connect(self._on_import_table)
        b_export = QPushButton("Export table…")
        b_export.setToolTip("Save the samples' groups and covariates in the same format.")
        b_export.clicked.connect(self._on_export_table)
        b_check = QPushButton("Check table…")
        b_check.setToolTip(
            "Look for blank cells, NA and Unknown spellings, similar categories, numbers "
            "written as text and covariates confounded with the groups, and fix them.")
        b_check.clicked.connect(self._on_check_table)
        trow.addWidget(b_import)
        trow.addWidget(b_export)
        trow.addWidget(b_check)
        cl.addLayout(trow)
        cl.addStretch()
        top.addWidget(cbox, stretch=1)
        layout.addLayout(top)

        auto_row = QHBoxLayout()
        self.lbl_auto_groupings = QLabel()
        self.lbl_auto_groupings.setWordWrap(True)
        self.lbl_auto_groupings.setStyleSheet("color: grey; font-style: italic;")
        self.btn_undo_groupings = QPushButton("Undo")
        self.btn_undo_groupings.setToolTip("Remove the groups and pairing applied from the sample names.")
        self.btn_undo_groupings.clicked.connect(self._undo_auto_groupings)
        auto_row.addWidget(self.lbl_auto_groupings, stretch=1)
        auto_row.addWidget(self.btn_undo_groupings)
        self.lbl_auto_groupings.setVisible(False)
        self.btn_undo_groupings.setVisible(False)
        layout.addLayout(auto_row)

        self.samples_table = QTableWidget(0, 3)
        self._samples_header = CheckableHeaderView(
            self.samples_table, check_label="Numeric",
            tooltip="Enter this covariate as a continuous number (e.g. age, CRP) instead of "
                    "as categories. Every value must be a number.")
        self.samples_table.setHorizontalHeader(self._samples_header)
        self._samples_header.check_toggled.connect(self._on_covariate_numeric_toggled)
        self.samples_table.setTextElideMode(Qt.ElideMiddle)
        self.samples_table.verticalHeader().setVisible(False)
        self.samples_table.itemChanged.connect(self._on_sample_cell_changed)
        layout.addWidget(self.samples_table, stretch=1)

    def _populate_groups(self):
        self._filling = True
        try:
            self.groups_table.setRowCount(0)
            colours = _group_colours(self.state)
            for r, name in enumerate(self.state.group_names):
                self.groups_table.insertRow(r)
                item = QTableWidgetItem(name)
                item.setData(Qt.UserRole, name)
                self.groups_table.setItem(r, 0, item)
                self.groups_table.setItem(r, 1, QTableWidgetItem(self.state.group_patterns.get(name, '')))
                swatch = QTableWidgetItem('')
                swatch.setBackground(QColor(colours[name]))
                swatch.setFlags(swatch.flags() & ~Qt.ItemIsEditable)
                swatch.setToolTip("Double-click to choose this group's colour in every figure.")
                self.groups_table.setItem(r, 2, swatch)
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
            hh.setSectionResizeMode(QHeaderView.Interactive)
            hh.setStretchLastSection(False)
            self.samples_table.resizeColumnsToContents()
            self.samples_table.setColumnWidth(
                0, min(self.samples_table.columnWidth(0), SAMPLE_COLUMN_MAX_WIDTH))
            fm = self.samples_table.fontMetrics()
            group_w = max([fm.horizontalAdvance(g) for g in st.group_names] + [0]) + 48
            self.samples_table.setColumnWidth(1, max(group_w, 110))
            for j in range(3, len(headers)):
                self.samples_table.setColumnWidth(
                    j, max(self.samples_table.columnWidth(j), COVARIATE_COLUMN_MIN_WIDTH))
            self._samples_header.set_checks(
                {j: cov in st.numeric_covariates for j, cov in enumerate(covs, start=3)})
        finally:
            self._filling = False

    def _on_covariate_numeric_toggled(self, section: int, checked: bool):
        """Enter the covariate under *section* as a continuous number (or as
        categories). A column holding text cannot be numeric."""
        st = self.state
        name = self.samples_table.horizontalHeaderItem(section).text()
        if checked:
            filled = [str(v).strip() for v in (st.covariates.get(name) or {}).values()
                      if str(v).strip()]
            if filled and ds.covariate_kind(filled) != 'numeric':
                QMessageBox.warning(
                    self, "Numeric covariate",
                    f"'{name}' has values that are not numbers, so it cannot be numeric.")
                self._samples_header.set_checks(
                    {j: self.samples_table.horizontalHeaderItem(j).text() in st.numeric_covariates
                     for j in self._samples_header.checks()})
                return
        st.numeric_covariates = [c for c in st.numeric_covariates if c != name]
        if checked:
            st.numeric_covariates.append(name)
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
            if old in st.group_colors:
                st.group_colors[new] = st.group_colors.pop(old)
            for k, g in list(st.sample_groups.items()):
                if g == old:
                    st.sample_groups[k] = new
            if st.reference_group == old:
                st.reference_group = new
            item.setData(Qt.UserRole, new)
            self._populate_samples_table()
            self._populate_stats_choices()
        elif item.column() == 1:
            st.group_patterns[old] = item.text().strip()

    def _on_group_cell_double_clicked(self, row: int, col: int):
        """Choose a group's colour (the Colour column) and redraw the figures."""
        st = self.state
        if col != 2 or row >= len(st.group_names):
            return
        name = st.group_names[row]
        colour = QColorDialog.getColor(QColor(_group_colours(st)[name]), self,
                                       f"Colour for group '{name}'")
        if not colour.isValid():
            return
        st.group_colors[name] = colour.name()
        self._populate_groups()
        if st.stats:
            self._draw_figures()

    def _on_add_group(self):
        n = 1
        while f"Group {n}" in self.state.group_names:
            n += 1
        self.state.group_names.append(f"Group {n}")
        self._populate_groups()
        self._populate_stats_choices()

    def _on_move_group(self, step: int):
        """Move the selected group *step* rows up (-1) or down (+1). Each
        group keeps its colour."""
        st = self.state
        row = self.groups_table.currentRow()
        new = row + step
        if row < 0 or not 0 <= new < len(st.group_names):
            return
        st.group_colors = _group_colours(st)
        st.group_names.insert(new, st.group_names.pop(row))
        self._populate_groups()
        self._populate_stats_choices()
        self.groups_table.setCurrentCell(new, 0)
        if st.stats:
            self._draw_figures()

    def _on_remove_group(self):
        row = self.groups_table.currentRow()
        if row < 0:
            return
        name = self.groups_table.item(row, 0).data(Qt.UserRole)
        st = self.state
        st.group_names.remove(name)
        st.group_patterns.pop(name, None)
        st.group_colors.pop(name, None)
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

    def _groupings_result(self, keys: list[str], current_groups: dict | None = None) -> dict:
        display = self.controller.experiment.samples.get('all_samples', {}) if self.controller else {}
        return ds.suggest_groupings_from_names(keys, display, current_groups)

    def _on_suggest_groupings(self):
        """Offer groupings found in the gated samples' names (see
        ``differential_stats.suggest_groupings_from_names``)."""
        keys = self.state.gated_samples()
        if len(keys) < 2:
            QMessageBox.information(self, "Suggest Groupings",
                                    "Need at least two gated samples to detect a pattern.")
            return
        st = self.state
        current = {k: st.sample_groups[k] for k in keys
                   if st.sample_groups.get(k) in st.group_names}
        result = self._groupings_result(keys, current)
        if not result['suggestions']:
            QMessageBox.information(self, "Suggest Groupings", sgd.no_suggestion_message(result))
            return
        chosen = sgd.choose_groupings(self, result)
        if chosen:
            self._apply_groupings(chosen, keys)
            self._show_auto_groupings_note(None)

    def _maybe_auto_suggest_groupings(self):
        """Apply groupings found in the sample names when none of the gated
        samples has a group yet: the detected group and pairing variable.
        Runs once per sample list and never overwrites assignments; the note
        above the sample table says what was applied and offers Undo."""
        st = self.state
        keys = st.gated_samples()
        if len(keys) < 2 or any(st.sample_groups.get(k) in st.group_names for k in keys):
            return
        if tuple(keys) == self._auto_groupings_key:
            return
        self._auto_groupings_key = tuple(keys)
        chosen = sgd.automatic_groupings(self._groupings_result(keys))
        if not chosen:
            return
        self._auto_groupings_undo = (list(st.group_names), dict(st.group_patterns), dict(st.sample_groups),
                                     deepcopy(st.covariates), st.pairing)
        self._apply_groupings(chosen, keys)
        self._show_auto_groupings_note(sgd.describe_groupings(chosen))

    def _show_auto_groupings_note(self, text: str | None):
        if text is None:
            self._auto_groupings_undo = None
        self.lbl_auto_groupings.setText(text or '')
        self.lbl_auto_groupings.setVisible(text is not None)
        self.btn_undo_groupings.setVisible(text is not None and self._auto_groupings_undo is not None)

    def _undo_auto_groupings(self):
        if self._auto_groupings_undo is None:
            return
        st = self.state
        st.group_names, st.group_patterns, st.sample_groups, st.covariates, st.pairing = \
            self._auto_groupings_undo
        self._show_auto_groupings_note(None)
        self._populate_groups()
        self._load_stats_controls()

    def _apply_groupings(self, chosen: list[dict], keys: list[str]):
        """Write chosen groupings: a 'group' choice replaces the groups and
        the listed samples' assignments; other choices become covariates,
        and a 'pairing' choice is selected under Pair samples by."""
        st = self.state
        for choice in chosen:
            if choice['role'] == 'group':
                names = sorted(set(choice['values'].values()))
                st.group_names = names
                st.group_patterns = {n: st.group_patterns.get(n, '') for n in names}
                for k in keys:
                    if k in choice['values']:
                        st.sample_groups[k] = choice['values'][k]
                    else:
                        st.sample_groups.pop(k, None)
            else:
                name = choice['name']
                if name in st.covariates:
                    name = f"{name}_2"
                st.covariates[name] = dict(choice['values'])
                if choice['role'] == 'pairing':
                    st.pairing = name
        self._populate_groups()
        self._load_stats_controls()

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
        self._forget_covariate(name)
        self._populate_samples_table()
        self._populate_stats_choices()

    def _forget_covariate(self, name: str):
        """Clear a removed covariate from pairing, adjustment and Numeric."""
        if self.state.pairing == name:
            self.state.pairing = ''
        self.state.adjust_covariates = [c for c in self.state.adjust_covariates if c != name]
        self.state.numeric_covariates = [c for c in self.state.numeric_covariates if c != name]

    def _on_check_table(self):
        """Check the covariates of the samples the statistics would use."""
        st = self.state

        def comparisons(samples):
            names = ag_results.qualifying_groups(samples, st.sample_groups, list(st.group_names))
            if len(names) < 2:
                return []
            reference = st.reference_group if st.reference_group in names else names[0]
            return ds.build_contrasts(names, st.contrast_mode, reference)

        def check():
            samples = self._stats_samples()
            return covariate_check.check_covariates(
                st.covariates, samples, st.sample_groups, comparisons(samples),
                st.numeric_covariates)

        def apply(fixes):
            covs, numeric, removed = covariate_check.apply_fixes(
                st.covariates, st.numeric_covariates, self._stats_samples(), fixes)
            st.covariates = covs
            st.numeric_covariates = numeric
            for name in removed:
                self._forget_covariate(name)
            self._populate_samples_table()
            self._populate_stats_choices()
            text = covariate_check.describe_fixes(fixes, removed)
            _status(self.bus, text)
            return text

        covariate_check_dialog.check_table(self, "Check table", check, apply)

    def _on_import_table(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import sample table", str(getattr(self.controller, 'experiment_dir', '')),
            "Tables (*.csv *.tsv *.txt);;All files (*)")
        if not path:
            return
        st = self.state
        result = sample_table_import_dialog.import_sample_table(
            self, path, st.gated_samples(), st.sample_groups, title="Import table")
        if result is None:
            return
        if result['replace_covariates']:
            for name in list(st.covariates):
                if name not in result['covariates']:
                    self._forget_covariate(name)
            st.covariates = {}
        for group in result['groups'].values():
            if group not in st.group_names:
                st.group_names.append(group)
        st.sample_groups.update(result['groups'])
        for name, values in result['covariates'].items():
            st.covariates.setdefault(name, {}).update(values)
        self._populate_groups()
        self._populate_stats_choices()
        _status(self.bus, sample_table_import_dialog.describe_import(result))

    def _on_export_table(self):
        default = sample_table.default_export_path(
            getattr(self.controller, 'experiment_dir', ''), plugin_name)
        path, _ = QFileDialog.getSaveFileName(self, "Export sample table", default, "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith('.csv'):
            path += '.csv'
        st = self.state
        try:
            sample_table.write_table(
                sample_table.table_frame(st.gated_samples(), st.sample_groups, st.covariates), path)
        except Exception as exc:
            QMessageBox.warning(self, "Export table", str(exc))
            return
        _status(self.bus, f"Sample table written to {path}")

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

        self.btn_stats_defaults = QPushButton("Restore defaults")
        self.btn_stats_defaults.setToolTip(
            "Put the tests, minimum parent events, counts reference and model, marker options, "
            "thresholds, TREAT, FDR correction and 'Leave out flagged samples' back to their "
            "default values. Comparisons, reference group, pairing and adjustment covariates "
            "are kept.")
        self.btn_stats_defaults.clicked.connect(self._restore_stats_defaults)
        defaults_row = QHBoxLayout()
        defaults_row.addWidget(self.btn_stats_defaults)
        defaults_row.addStretch()
        form.addRow(defaults_row)

        self.cb_test = QComboBox()
        for key, text in STATS_TESTS.items():
            self.cb_test.addItem(text, key)
        self.cb_test.setToolTip(
            "Compare groups: which populations differ between sample groups. Trend: which "
            "populations change with a numeric covariate such as age or days after "
            "vaccination (mark it Numeric on the Groups page).")
        form.addRow("Test:", self.cb_test)

        self.chk_freq = QCheckBox("Population frequencies (log odds within parent)")
        self.chk_counts = QCheckBox("Population counts")
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
        self.sb_min_parent = QSpinBox()
        self.sb_min_parent.setRange(0, 1_000_000)
        self.sb_min_parent.setToolTip(
            "A population's frequency is measured in a sample only when its parent has at "
            "least this many events there; other samples are left out of that population's "
            "test. A fraction of a handful of events mostly reflects gating noise.")
        self.cb_counts_reference = QComboBox()
        self.cb_counts_reference.addItem("Anchor population", 'anchor')
        self.cb_counts_reference.addItem("Parent population", 'parent')
        self.cb_counts_reference.setToolTip(
            "Anchor: is the population over- or under-represented for the number of viable "
            "cells (the model's anchor population)? A large change in one abundant population "
            "moves every other population too. Parent: the frequency test's question, within "
            "the parent.")
        self.cb_counts_method = QComboBox()
        self.cb_counts_method.addItem("Moderated log odds (recommended)", 'moderated')
        self.cb_counts_method.addItem(
            "Quasi-likelihood F (negative binomial, homogeneous designs)", 'ql')
        self.cb_counts_method.setToolTip(
            "Moderated log odds: the frequency test's model, applied to the share of the "
            "reference population; it held its false-positive rate on human and mouse data. "
            "Quasi-likelihood F: a negative-binomial model of the event counts with the reference "
            "count as offset; suited to homogeneous designs such as inbred mice with plenty of "
            "events, too liberal on human cohorts.")
        self.lbl_counts_note = QLabel("")
        self.lbl_counts_note.setWordWrap(True)
        self.lbl_counts_note.setStyleSheet("color: #856404;")
        form.addRow(self.chk_freq)
        form.addRow("Min. parent events:", self.sb_min_parent)
        counts_box = QGroupBox("Counts (advanced)")
        counts_form = QFormLayout(counts_box)
        counts_form.addRow(self.chk_counts)
        counts_form.addRow("Relative to:", self.cb_counts_reference)
        counts_form.addRow("Model:", self.cb_counts_method)
        counts_form.addRow(self.lbl_counts_note)
        form.addRow(counts_box)
        for w in (self.chk_markers, self.chk_stats_parent, self.chk_exclude_gating):
            form.addRow(w)
        form.addRow("Min. events for medians:", self.sb_min_events)

        self.trend_box = QGroupBox("Trend")
        trend_form = QFormLayout(self.trend_box)
        self.cb_trend_var = QComboBox()
        self.cb_trend_var.setToolTip("Covariates ticked Numeric on the Groups page.")
        self.le_trend_unit = QLineEdit()
        self.le_trend_unit.setPlaceholderText("e.g. weeks, days")
        self.le_trend_unit.setToolTip("Unit of the variable, for the slope and the axes.")
        self.list_trend_groups = QListWidget()
        self.list_trend_groups.setMaximumHeight(80)
        self.list_trend_groups.setToolTip(
            "Groups whose samples are tested. Samples without a value for the variable are "
            "left out.")
        self.cb_trend_question = QComboBox()
        self.cb_trend_question.addItem("Change with the variable (groups ticked together)", False)
        self.cb_trend_question.addItem("Does the change differ between two groups?", True)
        self.cb_trend_question.setToolTip(
            "The second asks whether two ticked groups change differently (group × trend); "
            "the reference group is the first. With pairing (e.g. two tissues of one donor) "
            "it tests the within-pair difference against the variable.")
        self.cb_trend_shape = QComboBox()
        for key, text in TREND_SHAPE_LABELS.items():
            self.cb_trend_shape.addItem(text, key)
        self.cb_trend_shape.setToolTip(
            "Smooth curve: any shape, one p-value for the variable as a whole (an ANOVA-style F "
            "test). Straight line: a steady rise or fall. Time points as groups: each value is "
            "its own level, for designed time courses (day 0, 3, 7, 14).")
        self.sb_trend_df = QSpinBox()
        self.sb_trend_df.setRange(1, TREND_MAX_DF)
        self.sb_trend_df.setToolTip(
            "How many bends the curve may make (1 = a straight line, 2 = one bend). Capped at "
            "one per 10 samples.")
        self.cb_trend_scale = QComboBox()
        for key, text in TREND_SCALE_LABELS.items():
            self.cb_trend_scale.addItem(text, key)
        self.cb_trend_scale.setToolTip(
            "Log2 suits time courses sampled on a doubling schedule (day 0, 1, 3, 7, 14, 28).")
        self.cb_trend_call = QComboBox()
        for key, text in TREND_CALL_LABELS.items():
            self.cb_trend_call.addItem(text, key)
        self.cb_trend_call.setToolTip(
            "A population is called when the curve test or the slope test is significant, each "
            "at half the FDR so the combined error stays at the FDR set below; or by one test "
            "alone.")
        self.sb_trend_r2 = QDoubleSpinBox()
        self.sb_trend_r2.setRange(0.0, 1.0)
        self.sb_trend_r2.setDecimals(2)
        self.sb_trend_r2.setSingleStep(0.05)
        self.sb_trend_r2.setToolTip(
            "Minimum partial R²: the share of the sample-to-sample variation the variable "
            "explains, after groups and adjustment covariates. In large studies tiny trends "
            "become significant; this keeps them out.")
        self.sb_trend_patterns = QSpinBox()
        self.sb_trend_patterns.setRange(0, 10)
        self.sb_trend_patterns.setSpecialValueText("Auto")
        self.sb_trend_patterns.setToolTip(
            "Shape patterns: the called populations are grouped by the shape of their fitted curve (rises, falls, peaks mid-range). Auto uses the largest number of patterns that come back the same when the test is repeated on 80% of the samples; a number groups into exactly that many.")
        self.lbl_trend_check = QLabel("")
        self.lbl_trend_check.setWordWrap(True)
        trend_form.addRow("Variable:", self.cb_trend_var)
        trend_form.addRow("Unit:", self.le_trend_unit)
        trend_form.addRow("Groups:", self.list_trend_groups)
        trend_form.addRow("Question:", self.cb_trend_question)
        trend_form.addRow("Shape:", self.cb_trend_shape)
        trend_form.addRow("Curve df:", self.sb_trend_df)
        trend_form.addRow("Scale:", self.cb_trend_scale)
        trend_form.addRow("Call when:", self.cb_trend_call)
        trend_form.addRow("Min. partial R²:", self.sb_trend_r2)
        trend_form.addRow("Patterns:", self.sb_trend_patterns)
        trend_form.addRow(self.lbl_trend_check)
        form.addRow(self.trend_box)

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
        self.sb_fc.setToolTip("|log2 odds ratio| for frequencies and moderated counts, "
                              "|log2 fold change| for negative-binomial counts.")
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
        self.cb_counts_method.currentIndexChanged.connect(self._on_counts_settings_changed)
        self.cb_counts_reference.currentIndexChanged.connect(self._on_counts_settings_changed)
        self.sb_min_parent.valueChanged.connect(
            lambda v: setattr(self.state, 'min_parent_events', int(v)))
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
        for box in (self.chk_freq, self.chk_counts, self.chk_markers, self.chk_stats_parent,
                    self.chk_exclude_gating, self.chk_exclude_flagged, self.chk_treat):
            box.toggled.connect(self._update_stats_defaults_button)
        for spin in (self.sb_min_events, self.sb_min_parent, self.sb_pval, self.sb_fc,
                     self.sb_marker):
            spin.valueChanged.connect(self._update_stats_defaults_button)
        for combo in (self.cb_counts_method, self.cb_counts_reference, self.cb_fdr):
            combo.currentIndexChanged.connect(self._update_stats_defaults_button)
        self.cb_test.currentIndexChanged.connect(self._on_test_changed)
        self.cb_trend_var.currentIndexChanged.connect(self._on_trend_settings_changed)
        self.le_trend_unit.editingFinished.connect(self._on_trend_settings_changed)
        self.list_trend_groups.itemChanged.connect(self._on_trend_settings_changed)
        for combo in (self.cb_trend_question, self.cb_trend_shape, self.cb_trend_scale):
            combo.currentIndexChanged.connect(self._on_trend_settings_changed)
        self.sb_trend_df.valueChanged.connect(self._on_trend_settings_changed)
        self.cb_trend_call.currentIndexChanged.connect(self._on_trend_thresholds_changed)
        self.sb_trend_r2.valueChanged.connect(self._on_trend_thresholds_changed)
        self.sb_trend_patterns.valueChanged.connect(self._on_trend_thresholds_changed)
        for w in (self.cb_trend_shape, self.cb_trend_scale, self.cb_trend_call):
            w.currentIndexChanged.connect(self._update_stats_defaults_button)
        for w in (self.sb_trend_df, self.sb_trend_r2, self.sb_trend_patterns):
            w.valueChanged.connect(self._update_stats_defaults_button)

    # ------------------------------------------------------------------
    # Trend settings
    # ------------------------------------------------------------------

    def _trend_mode(self) -> bool:
        return self.state.stats_test == 'trend'

    def _on_test_changed(self, _i):
        if self._filling:
            return
        self.state.stats_test = self.cb_test.currentData() or 'groups'
        self._apply_test_mode()
        self._show_run_summary()
        self._show_results_tables()
        self._draw_figures()
        self._load_explore_controls()
        self._populate_population_list()

    def _show_run_summary(self):
        """The summary line of the shown test's last run."""
        st = self.state
        if self._trend_mode():
            self.lbl_stats.setText(self._trend_summary(st.trend) if st.trend else "")
        else:
            self.lbl_stats.setText(self._stats_summary(st.stats) if st.stats else "")

    def _apply_test_mode(self):
        """Show the trend settings in trend mode and grey out the settings
        that only group comparisons use."""
        st = self.state
        trend = self._trend_mode()
        self.trend_box.setVisible(trend)
        for w in (self.cb_mode, self.sb_fc, self.sb_marker, self.chk_treat, self.cb_fdr):
            w.setEnabled(not trend)
        self.cb_reference.setEnabled(st.trend_interaction if trend
                                     else st.contrast_mode == 'reference')
        self.cb_comparison.setEnabled(not trend)
        self.sb_trend_df.setEnabled(st.trend_shape == 'curve')
        self.btn_stats.setText("Run Trend Test" if trend else "Run Statistics")
        if trend:
            self._update_trend_check()

    def _load_trend_controls(self):
        st = self.state
        numeric = [c for c in st.covariates if c in st.numeric_covariates and c != st.pairing]
        if st.trend_variable not in numeric:
            st.trend_variable = numeric[0] if numeric else ''
        self._filling = True
        try:
            self.cb_test.setCurrentIndex(max(self.cb_test.findData(st.stats_test), 0))
            self.cb_trend_var.clear()
            for c in numeric:
                self.cb_trend_var.addItem(c, c)
            self.cb_trend_var.setCurrentIndex(max(self.cb_trend_var.findData(st.trend_variable), 0))
            self.le_trend_unit.setText(st.trend_unit)
            self.list_trend_groups.clear()
            chosen = [g for g in st.trend_groups if g in st.group_names]
            for g in st.group_names:
                it = QListWidgetItem(g)
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if (not chosen or g in chosen) else Qt.Unchecked)
                self.list_trend_groups.addItem(it)
            for cb, v in ((self.cb_trend_question, st.trend_interaction),
                          (self.cb_trend_shape, st.trend_shape),
                          (self.cb_trend_scale, st.trend_scale),
                          (self.cb_trend_call, st.trend_call)):
                cb.setCurrentIndex(max(cb.findData(v), 0))
            self.sb_trend_df.setValue(int(st.trend_df))
            self.sb_trend_r2.setValue(float(st.trend_min_r2))
            self.sb_trend_patterns.setValue(int(st.trend_patterns))
        finally:
            self._filling = False
        self._apply_test_mode()

    def _trend_group_names(self) -> list[str]:
        """The groups ticked for the trend, in the Groups table's order."""
        st = self.state
        return [g for g in st.group_names if not st.trend_groups or g in st.trend_groups]

    def _trend_samples(self) -> list[str]:
        groups = set(self._trend_group_names())
        return [k for k in self._stats_samples() if self.state.sample_groups.get(k) in groups]

    def _trend_values(self, samples: list[str]) -> dict:
        """{sample: value of the trend variable} (NaN where it has none)."""
        vals = self.state.covariates.get(self.state.trend_variable) or {}
        out = {}
        for k in samples:
            number = covariate_check.parse_number(vals.get(k))
            out[k] = float('nan') if number is None else float(number)
        return out

    def _on_trend_settings_changed(self, *_args):
        if self._filling:
            return
        st = self.state
        st.trend_variable = self.cb_trend_var.currentData() or ''
        st.trend_unit = self.le_trend_unit.text().strip()
        ticked = [self.list_trend_groups.item(i).text() for i in range(self.list_trend_groups.count())
                  if self.list_trend_groups.item(i).checkState() == Qt.Checked]
        st.trend_groups = [] if len(ticked) == self.list_trend_groups.count() else ticked
        st.trend_interaction = bool(self.cb_trend_question.currentData())
        st.trend_shape = self.cb_trend_shape.currentData() or 'curve'
        st.trend_df = int(self.sb_trend_df.value())
        st.trend_scale = self.cb_trend_scale.currentData() or 'linear'
        self._apply_test_mode()

    def _on_trend_thresholds_changed(self, *_args):
        if self._filling:
            return
        st = self.state
        st.trend_call = self.cb_trend_call.currentData() or 'either'
        st.trend_min_r2 = float(self.sb_trend_r2.value())
        st.trend_patterns = int(self.sb_trend_patterns.value())
        self._rethreshold_trend()

    def _rethreshold_trend(self):
        st = self.state
        if not st.trend:
            return
        st.trend = ag_results.reapply_trend_thresholds(st.trend, st.pval_threshold,
                                                       st.trend_min_r2, st.trend_call,
                                                       st.trend_patterns)
        if self._trend_mode():
            self.lbl_stats.setText(self._trend_summary(st.trend))
            self._show_results_tables()
            self._draw_figures()
            self._populate_population_list()

    def _update_trend_check(self):
        """The design check under the trend settings: samples with a value,
        the curve df they allow, and any covariate (or the events per
        sample) that follows the variable closely."""
        st = self.state
        var = st.trend_variable
        if not var:
            self.lbl_trend_check.setText(
                "Tick <b>Numeric</b> under a covariate on the Groups page to test a trend "
                "with it.")
            return
        samples = self._trend_samples()
        values = self._trend_values(samples)
        have = [k for k in samples if np.isfinite(values[k])]
        lines = []
        if st.trend_shape == 'curve':
            df = ts.curve_df(len(have), st.trend_df)
            lines.append(f"{len(have)} sample(s) with a value; the curve gets {df} df.")
        else:
            lines.append(f"{len(have)} sample(s) with a value.")
        if st.trend_interaction and len(self._trend_group_names()) != 2:
            lines.append("<span style='color:#b00020'>Tick exactly two groups to ask whether "
                         "they change differently.</span>")
        if have and st.trend_shape != 'points' and ts.suggest_shape([values[k] for k in have]) == 'points':
            lines.append(f"'{var}' has few distinct values with several samples each: "
                         "<i>Time points as groups</i> may suit a designed time course.")
        others = [c for c in st.covariates if c not in (var, st.pairing)]
        cov = pd.DataFrame({c: [(st.covariates.get(c) or {}).get(k, '') for k in have]
                            for c in others}, index=have) if others else None
        events = {'Events in the sample': [(st.summaries.get(k) or {}).get('n_total')
                                           for k in have]}
        checks = ts.design_check([values[k] for k in have], cov, events)
        warned = [c for c in checks if c['warn']]
        for c in warned:
            adjusted = c['name'] in st.adjust_covariates
            lines.append(f"<span style='color:#856404'>⚠ {c['name']} follows {var} "
                         f"({c['measure']} = {c['value']:.2f})"
                         + ("; adjusted for." if adjusted else
                            ": a trend could reflect it. Consider adjusting for it.")
                         + "</span>")
        if checks and not warned:
            lines.append(f"No other covariate, nor the events per sample, follows {var} "
                         f"closely (all below {ts.DESIGN_CHECK_LIMIT}).")
        self.lbl_trend_check.setText("<br>".join(lines))

    def _on_counts_settings_changed(self, *_args):
        if self._filling:
            return
        self.state.counts_method = self.cb_counts_method.currentData() or 'moderated'
        self.state.counts_reference = self.cb_counts_reference.currentData() or 'anchor'
        self._update_counts_note()

    def _update_counts_note(self):
        """Name the anchor population in the reference choice, and note when
        the chosen model is liberal or repeats the frequency test."""
        st = self.state
        gate_defs = st.gate_defs()
        anchor = ag_core.anchor_population(gate_defs) if gate_defs else None
        name = _population_labels(gate_defs).get(anchor, anchor) if anchor else 'all events'
        self.cb_counts_reference.setItemText(0, f"Anchor population ({name})")
        notes = []
        if st.counts_method == 'ql':
            notes.append("Quasi-likelihood F suits homogeneous designs such as inbred mice "
                         "(3–6% false positives at the 5% level, about 5% with 100 cells per "
                         "sample). On human cohort data it gave 7–10% for populations under 100 "
                         "events; use moderated log odds there.")
        if st.counts_reference == 'parent' and st.counts_method == 'moderated':
            notes.append("Against the parent with this model, the counts test repeats the "
                         "frequency test.")
        self.lbl_counts_note.setText(" ".join(notes))
        self.lbl_counts_note.setVisible(bool(notes))

    def _update_stats_defaults_button(self, *_args):
        """Grey the Restore defaults button while every statistics setting
        it puts back is at its default."""
        defaults = GatingRunState()
        self.btn_stats_defaults.setEnabled(
            any(getattr(self.state, key) != getattr(defaults, key) for key in STATS_DEFAULT_KEYS))

    def _restore_stats_defaults(self):
        """Put the statistics settings back to their defaults. Stored
        results are re-thresholded when only thresholds, TREAT or the FDR
        correction changed; the other settings apply at the next run."""
        st = self.state
        defaults = GatingRunState()
        changed = [key for key in STATS_DEFAULT_KEYS if getattr(st, key) != getattr(defaults, key)]
        if not changed:
            return
        for key in changed:
            setattr(st, key, getattr(defaults, key))
        self._load_stats_controls()
        if set(changed) & set(THRESHOLD_KEYS):
            self._on_thresholds_changed()
        _status(self.bus, "Statistics settings restored to their defaults.")

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
            for spin, v in ((self.sb_min_events, st.min_marker_events),
                            (self.sb_min_parent, st.min_parent_events)):
                spin.blockSignals(True)
                spin.setValue(int(v))
                spin.blockSignals(False)
            for cb, v in ((self.cb_mode, st.contrast_mode), (self.cb_fdr, st.fdr_scope),
                          (self.cb_counts_method, st.counts_method),
                          (self.cb_counts_reference, st.counts_reference)):
                cb.blockSignals(True)
                cb.setCurrentIndex(max(cb.findData(v), 0))
                cb.blockSignals(False)
        finally:
            self._filling = False
        self._populate_stats_choices()
        self._update_counts_note()
        self._load_trend_controls()
        self._update_stats_defaults_button()

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
        if self._trend_mode():
            self._update_trend_check()

    def _on_thresholds_changed(self, *_args):
        if self._filling:
            return
        st = self.state
        st.pval_threshold = float(self.sb_pval.value())
        st.fc_threshold = float(self.sb_fc.value())
        st.marker_threshold = float(self.sb_marker.value())
        st.use_treat = bool(self.chk_treat.isChecked())
        st.fdr_scope = self.cb_fdr.currentData()
        st.trend_call = self.cb_trend_call.currentData() or 'either'
        st.trend_min_r2 = float(self.sb_trend_r2.value())
        st.trend_patterns = int(self.sb_trend_patterns.value())
        self._rethreshold_trend()
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
            self._populate_population_list()

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
        if self._trend_mode():
            self._run_trend()
            return
        if not (st.run_freq or st.run_counts or st.run_markers):
            QMessageBox.information(self, "Nothing to test", "Tick at least one kind of test.")
            return
        samples = self._stats_samples()
        design = statistics_design(st, samples)
        groups = design['groups']
        kwargs = dict(
            group_order=list(st.group_names), contrast_mode=st.contrast_mode,
            reference=st.reference_group, pairing=design['pairing'],
            covariates=design['covariates'], numeric_covariates=design['numeric'],
            pval_threshold=st.pval_threshold, fc_threshold=st.fc_threshold,
            marker_threshold=st.marker_threshold, use_treat=st.use_treat,
            fdr_scope=st.fdr_scope, run_freq=st.run_freq, run_counts=st.run_counts,
            counts_method=st.counts_method, counts_reference=st.counts_reference,
            min_parent_events=st.min_parent_events,
            run_markers=st.run_markers, use_stats_parent=st.use_stats_parent,
            exclude_gating_markers=st.exclude_gating_markers,
            min_marker_events=st.min_marker_events,
        )
        # Each sample's summary is replaced whole when it is gated again,
        # never changed in place, so the worker can read this snapshot.
        summaries = {k: st.summaries[k] for k in samples}
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
        # Identifies the run, so a saved PCA or predictive analysis is only
        # restored with the statistics it was made from.
        result['run_id'] = uuid.uuid4().hex
        self.state.stats = result
        # PCA and predictive results describe the previous statistics run.
        self.state.pca = {}
        self.state.predictive = {}
        self._load_explore_controls()
        self.lbl_stats.setText(self._stats_summary(result))
        _status(self.bus, "Statistics complete.")
        self._populate_comparisons()
        self._show_results_tables()
        self._draw_figures()
        self._populate_population_list()
        self._populate_report_items()
        self.stats_finished.emit()

    def trend_enrichment_inputs(self) -> dict | None:
        """The trend settings for the Enrichment page: {'samples', 'time',
        'variable', 'unit', 'scale', 'shape', 'k', 'interaction', 'baseline',
        'groups', 'design', 'patterns'}; None without a trend variable or
        groups. 'patterns' is the shape pattern of each population of the
        last frequency trend run when it used the same variable."""
        st = self.state
        if not st.trend_variable or st.trend_variable not in st.covariates:
            return None
        group_names = self._trend_group_names()
        if not group_names or (st.trend_interaction and len(group_names) != 2):
            return None
        samples = self._trend_samples()
        patterns = None
        res = (st.trend or {}).get('freq')
        if res is not None and st.trend.get('variable') == st.trend_variable \
                and 'pattern' in res.table.columns:
            patterns = res.table.set_index('feature')['pattern']
        return {'samples': samples, 'time': self._trend_values(samples),
                'variable': st.trend_variable, 'unit': st.trend_unit, 'scale': st.trend_scale,
                'shape': st.trend_shape, 'k': st.trend_df, 'interaction': st.trend_interaction,
                'baseline': (st.reference_group if st.reference_group in group_names
                             else group_names[0]),
                'groups': group_names, 'design': statistics_design(st, samples),
                'patterns': patterns}

    def _run_trend(self):
        """Test the trend of every population with the chosen numeric
        covariate, on a worker thread."""
        st = self.state
        if not (st.run_freq or st.run_counts or st.run_markers):
            QMessageBox.information(self, "Nothing to test", "Tick at least one kind of test.")
            return
        if not st.trend_variable or st.trend_variable not in st.covariates:
            QMessageBox.information(
                self, "No numeric covariate",
                "Tick Numeric under a covariate on the Groups page (for example age) to test a "
                "trend with it.")
            return
        group_names = self._trend_group_names()
        if not group_names or (st.trend_interaction and len(group_names) != 2):
            QMessageBox.information(self, "Choose the groups",
                                    "Tick the groups to test (exactly two to ask whether they "
                                    "change differently).")
            return
        samples = self._trend_samples()
        design = statistics_design(st, samples)
        baseline = st.reference_group if st.reference_group in group_names else group_names[0]
        kwargs = dict(
            time=self._trend_values(samples), variable=st.trend_variable, unit=st.trend_unit,
            scale=st.trend_scale, shape=st.trend_shape, k=st.trend_df,
            interaction=st.trend_interaction, baseline=baseline, pairing=design['pairing'],
            covariates=design['covariates'], numeric_covariates=design['numeric'],
            pval_threshold=st.pval_threshold, min_r2=st.trend_min_r2, call=st.trend_call,
            patterns=st.trend_patterns, run_freq=st.run_freq, run_counts=st.run_counts,
            counts_reference=st.counts_reference, min_parent_events=st.min_parent_events,
            run_markers=st.run_markers, use_stats_parent=st.use_stats_parent,
            exclude_gating_markers=st.exclude_gating_markers,
            min_marker_events=st.min_marker_events,
        )
        groups = design['groups']
        # Each sample's summary is replaced whole when it is gated again,
        # never changed in place, so the worker can read this snapshot.
        summaries = {k: st.summaries[k] for k in samples}
        gate_defs = deepcopy(st.gate_defs())

        def _run():
            return ag_results.run_trend_statistics(summaries, samples, groups, gate_defs, **kwargs)

        self.btn_stats.setEnabled(False)
        self.lbl_stats.setText(f"Testing trends in {len(samples)} sample(s) …")
        self._stats_worker = _CallWorker(_run, parent=self)
        self._stats_worker.finished.connect(self._on_trend_finished)
        self._stats_worker.start()

    def _on_trend_finished(self, ok: bool, error: str, result: object):
        self._stats_worker = None
        self.btn_stats.setEnabled(True)
        if not ok:
            self.lbl_stats.setText(f"<span style='color:#b00020'>{error}</span>")
            return
        result['run_id'] = uuid.uuid4().hex
        self.state.trend = result
        self.lbl_stats.setText(self._trend_summary(result))
        _status(self.bus, "Trend test complete.")
        self._show_results_tables()
        self._draw_figures()
        self._load_explore_controls()
        self._populate_population_list()
        self._populate_report_items()
        self.trend_finished.emit()

    @staticmethod
    def _trend_summary(trend: dict) -> str:
        """One-line summary of a trend run, with its notes."""
        called = []
        for key, name in (('freq', 'frequencies'), ('counts', 'counts'), ('markers', 'markers')):
            res = trend.get(key)
            if res is not None:
                called.append(f"{int(res.table['significant'].sum())} {name}")
        settings = trend.get('settings') or {}
        shape = TREND_SHAPE_LABELS.get(settings.get('shape'), '').lower()
        text = (f"Trend with {trend.get('variable', '')}: {len(trend.get('samples') or [])} "
                f"samples, {shape} ({settings.get('df', '')} df); called: "
                + ", ".join(called) + ".")
        if trend.get('notes'):
            text += "<br>" + "<br>".join(trend['notes'])
        return text

    @staticmethod
    def _stats_summary(result: dict) -> str:
        """One-line summary of a statistics run, with its notes."""
        n_sig = sum(int(result[k]['significant'].sum()) for k in ('freq', 'counts', 'markers')
                    if result.get(k) is not None)
        text = (f"{len(result['samples'])} samples, {len(result['comparisons'])} comparison(s), "
                f"{n_sig} significant result(s).")
        if result.get('notes'):
            text += "<br>" + "<br>".join(result['notes'])
        return text

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
        self._populate_population_list()

    def _show_trend_tables(self):
        """The trend run's tables: slope, partial R², the two tests' FDR
        and the call, populations ordered by the stronger of the two
        tests."""
        trend = self.state.trend or {}
        labels = _population_labels(self.state.gate_defs())
        antigens = _antigen_map(self.controller)
        settings = trend.get('settings') or {}
        unit = trend.get('unit') or 'unit'
        slope_head = ("Slope (per doubling)" if settings.get('scale') == 'log2'
                      else f"Slope (per {unit})")

        def _fmt(value) -> str:
            return f"{value:.3g}" if value is not None and pd.notna(value) else ''

        for key, title in (('freq', 'Frequencies'), ('counts', 'Counts'), ('markers', 'Markers')):
            res = trend.get(key)
            if res is None:
                continue
            table_df = res.table
            view = table_df.assign(_order=np.fmin(table_df['adj.P.Val'], table_df['slope.adj.P.Val']))
            view = view.sort_values('_order', kind='stable')
            rows = []
            for _i, r in view.iterrows():
                pop_key = r['cluster_id'] if key == 'markers' else r['feature']
                row = {'Population': labels.get(pop_key, pop_key),
                       slope_head: _fmt(r['slope']),
                       'Partial R²': _fmt(r['R2']),
                       'Curve FDR': _fmt(r['adj.P.Val']),
                       'Slope FDR': _fmt(r['slope.adj.P.Val']),
                       'Called by': r['called.by'],
                       'Direction': r['direction'] if r['significant'] else '',
                       'Pattern': r.get('pattern', '') or '',
                       'Significant': 'yes' if r['significant'] else ''}
                if key == 'markers':
                    row['Marker'] = antigens.get(r['channel'], r['channel'])
                rows.append(row)
            headers = ['Population'] + (['Marker'] if key == 'markers' else []) + [
                slope_head, 'Partial R²', 'Curve FDR', 'Slope FDR', 'Called by', 'Direction',
                'Pattern', 'Significant']
            table = CopyableTableWidget(rows, headers)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            self.results_tabs.addTab(table, title)

    def _show_results_tables(self):
        self.results_tabs.clear()
        if self._trend_mode():
            self._show_trend_tables()
            return
        comp = self._current_comparison()
        stats = self.state.stats or {}
        if comp is None:
            return
        label = _comparison_label(*comp)
        labels = _population_labels(self.state.gate_defs())
        antigens = _antigen_map(self.controller)
        fdr_col = 'adj.P.Val.global' if self.state.fdr_scope == 'global' else 'adj.P.Val'
        counts_only = ag_results.counts_only_hits(stats)

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
                if key == 'counts' and (label, r['feature']) in counts_only:
                    row['Significant'] = 'yes ⚠ not a frequency hit'
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
        trend_mode = self._trend_mode()
        source = self.state.trend if trend_mode else self.state.stats
        if not source or (not trend_mode and comp is None):
            self.figures_tabs.clear()
            return
        self._figures_for = source
        if self._figure_worker is not None:
            self._figure_pending = True
            return
        st = self.state
        if trend_mode:
            trend = source
            args = (_is_dark_palette(), _antigen_map(self.controller),
                    _population_labels(st.gate_defs()), _group_colours(st))

            def _build():
                return build_trend_figures(trend, *args)
        else:
            stats = source
            args = (comp[0], comp[1], st.pval_threshold, st.fc_threshold, st.marker_threshold,
                    st.fdr_scope, _is_dark_palette(), _antigen_map(self.controller),
                    _population_labels(st.gate_defs()), _group_colours(st))

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
        self._row_figures = {}
        if not ok:
            self.figures_tabs.addTab(QLabel(f"Figures failed: {error}"), "Error")
            return
        exp_dir = getattr(self.controller, 'experiment_dir', '')
        from honeychrome.view_components.scalable_figure_widget import initial_scale_for
        for key, item in payload.items():
            if item['fig'] is None:
                lbl = QLabel(f"{item['title']}: {item['error']}")
                lbl.setStyleSheet("color: #b00020;")
                self.figures_tabs.addTab(lbl, item['title'])
                continue
            fig = item['fig']
            # Group comparison heatmaps open at half size; trend trajectories,
            # whose labels are already sized for the screen, at full size.
            large = ((key.endswith('heatmap') and not key.startswith('trend_'))
                     or key == 'marker_summary')
            widget = ScalableFigureWidget(
                fig, title=item['title'], experiment_dir=exp_dir,
                pop_out=lambda make=item['make'], title=item['title'], results=item.get('results'),
                large=large, k=key: self._pop_out_figure(make, title, results, large,
                                                         options_key=k),
                initial_scale=initial_scale_for(fig, HEATMAP_SCALE_PCT) if large else 100,
                parent=self, options=figure_options_ref(self.controller, key))
            self._connect_figure(widget.canvas, fig, item.get('results'))
            self.figures_tabs.addTab(widget, item['title'])
        self._add_extra_figures()
        if self._focus_figure:
            title, self._focus_figure = self._focus_figure, None
            self._open_figure(title)

    def _open_figure(self, title: str):
        """Open the Figures page on the figure called *title*; while the
        figures are being rebuilt, once they are shown."""
        self.toolbox.setCurrentIndex(self._page_figures)
        if self._figure_worker is not None:
            self._focus_figure = title
            return
        for i in range(self.figures_tabs.count()):
            if self.figures_tabs.tabText(i) == title:
                self.figures_tabs.setCurrentIndex(i)
                return

    def _connect_figure(self, canvas, fig, results=None):
        """Hover labels, and click-through to the Populations page: a
        double-click on a volcano point or a heatmap row shows that
        population (and ticks it for the report); a right-click offers the
        same and more."""
        handler = getattr(fig, '_hover_handler', None)
        if handler is not None:
            canvas.mpl_connect('motion_notify_event', handler)
        if results is not None:
            self._volcano_tables[id(canvas)] = results
        clickable = bool(getattr(fig, '_row_keys', None) or getattr(fig, '_point_keys', None))
        if clickable:
            self._row_figures[id(canvas)] = fig
        if results is not None or clickable:
            canvas.mpl_connect('button_press_event', self._on_figure_click)
            if getattr(fig, '_trend_kind', None):
                canvas.setToolTip("Double-click a point or row for its trend plot; "
                                  "right-click for more.")
            else:
                canvas.setToolTip("Double-click a point or row to show that population on the "
                                  "Populations page; right-click for more.")

    def _pop_out_figure(self, make, title: str, results=None, large: bool = False,
                        values_table=None, options_key: str | None = None):
        """Build a fresh copy of a figure in the background and show it in
        its own window (size slider, matplotlib toolbar, hover labels and
        the same click-through to the Populations page). The window stays
        open while the Populations page is used. *values_table*: optional
        function returning the values behind the figure, offered as CSV.
        *options_key*: the figure's options (figure_options), the same as
        where it was opened from."""
        from honeychrome.view_components import figure_options as fo
        ref = figure_options_ref(self.controller, options_key)
        worker = _CallWorker(fo.with_options(make, ref), parent=self)
        worker.finished.connect(
            lambda ok, err, fig, w=worker: self._show_pop_out(w, ok, err, fig, title, results,
                                                              large, values_table, ref))
        self._popout_workers.append(worker)
        _status(self.bus, f"Opening {title} …")
        worker.start()

    def _show_pop_out(self, worker, ok: bool, error: str, fig, title: str, results, large: bool,
                      values_table=None, options=None):
        from honeychrome.view_components.scalable_figure_widget import initial_scale_for
        if worker in self._popout_workers:
            self._popout_workers.remove(worker)
        if not ok or fig is None:
            QMessageBox.warning(self, "Pop out failed", error or "The figure could not be made.")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        widget = ScalableFigureWidget(
            fig, title=title, experiment_dir=getattr(self.controller, 'experiment_dir', ''),
            initial_scale=initial_scale_for(fig, HEATMAP_SCALE_PCT) if large else 100,
            parent=dialog, options=options)
        self._connect_figure(widget.canvas, fig, results)
        canvas_key = id(widget.canvas)
        dialog.finished.connect(lambda _code, k=canvas_key: (self._volcano_tables.pop(k, None),
                                                            self._row_figures.pop(k, None)))
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(widget.make_toolbar(dialog))
        layout.addWidget(widget, stretch=1)
        if values_table is not None:
            row = QHBoxLayout()
            row.addStretch()
            csv_button = QPushButton("Save values (CSV)…")
            csv_button.setToolTip("Write the values behind this figure as a CSV file.")
            csv_button.clicked.connect(
                lambda _c=False: self._save_values_csv(values_table, widget.export_stem))
            row.addWidget(csv_button)
            layout.addLayout(row)
        dpi = fig.get_dpi()
        scale = widget.scale_percent() / 100.0
        dialog.resize(min(int(fig.get_figwidth() * dpi * scale), 1400) + 40,
                      min(int(fig.get_figheight() * dpi * scale), 1000) + 110)
        dialog.show()

    def _figure_population(self, event) -> str | None:
        """Population key under a mouse event on a volcano or heatmap."""
        from honeychrome.view_components.differential_plots import significance_column
        ax = event.inaxes
        if ax is None or event.x is None:
            return None
        fig = self._row_figures.get(id(event.canvas))
        if fig is not None and ax is getattr(fig, '_row_axes', None) and event.ydata is not None:
            row = int(round(event.ydata))
            keys = fig._row_keys
            return str(keys[row]) if 0 <= row < len(keys) else None
        table = self._volcano_tables.get(id(event.canvas))
        if table is None or table.empty:
            return None
        col, _lbl = significance_column(table, self.state.fdr_scope)
        xy = np.column_stack([table['logFC'].values.astype(float),
                              -np.log10(np.maximum(table[col].values.astype(float), 1e-300))])
        ok = np.isfinite(xy).all(axis=1)
        if not ok.any():
            return None
        px = ax.transData.transform(xy[ok])
        d = np.hypot(px[:, 0] - event.x, px[:, 1] - event.y)
        i = int(np.argmin(d))
        if d[i] > 12:
            return None
        row = table[ok].iloc[i]
        return str(row.get('cluster_id') if 'cluster_id' in table.columns else row['feature'])

    def _save_values_csv(self, values_table, stem: str):
        """Ask for a file name (in the experiment folder by default) and
        write *values_table()* there as CSV."""
        start = str(Path(getattr(self.controller, 'experiment_dir', '') or '.') / f"{stem}.csv")
        path, _selected = QFileDialog.getSaveFileName(self, "Save values", start, "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith('.csv'):
            path += '.csv'
        try:
            values_table().to_csv(path, index=False)
        except Exception as exc:
            QMessageBox.warning(self, "Save values", str(exc))
            return
        _status(self.bus, f"Saved {Path(path).name}")

    def _on_figure_click(self, event):
        hit = self._figure_trend_feature(event)
        if hit is not None:
            kind, feature = hit
            if event.dblclick and event.button == 1:
                self._trend_plot_window(kind, feature)
            elif event.button == 3:
                self._trend_figure_menu(kind, feature)
            return
        pop_key = self._figure_population(event)
        if pop_key is None:
            return
        if event.dblclick and event.button == 1:
            self.show_population(pop_key, include=True)
        elif event.button == 3:
            self._figure_menu(pop_key)

    def _figure_menu(self, pop_key: str):
        from PySide6.QtGui import QCursor
        from PySide6.QtWidgets import QMenu
        label = _population_labels(self.state.gate_defs()).get(pop_key, pop_key)
        menu = QMenu(self)
        menu.addSection(label)
        act_show = menu.addAction("Show on the Populations page")
        act_window = menu.addAction("Open the comparison in a window")
        act_report = menu.addAction("Include in the report")
        act_report.setCheckable(True)
        act_report.setChecked(pop_key in self.state.report_populations)
        act_show.triggered.connect(lambda: self.show_population(pop_key))
        act_window.triggered.connect(lambda: self._population_window(pop_key))
        act_report.triggered.connect(lambda on: self._set_report_population(pop_key, on))
        self._add_composition_action(menu, pop_key)
        menu.exec(QCursor.pos())

    # ------------------------------------------------------------------
    # Trend figures: per-feature trend plot
    # ------------------------------------------------------------------

    def _figure_trend_feature(self, event) -> tuple[str, str] | None:
        """(kind, feature) under a mouse event on a trend volcano point or
        trajectory heatmap row."""
        from honeychrome.view_components import trend_plots as tp
        fig = self._row_figures.get(id(event.canvas))
        kind = getattr(fig, '_trend_kind', None) if fig is not None else None
        if kind is None or event.inaxes is None:
            return None
        feature = tp.point_under(fig, event)
        if feature is None and event.inaxes is getattr(fig, '_row_axes', None) \
                and event.ydata is not None:
            row = int(round(event.ydata))
            keys = getattr(fig, '_row_keys', None) or []
            feature = keys[row] if 0 <= row < len(keys) else None
        return (kind, str(feature)) if feature is not None else None

    def _trend_row(self, kind: str, feature: str) -> dict:
        res = (self.state.trend or {}).get(kind)
        if res is None:
            return {}
        rows = res.table[res.table['feature'].astype(str) == feature]
        return rows.iloc[0].to_dict() if len(rows) else {}

    def _trend_population(self, kind: str, feature: str) -> str:
        """Population key of a trend feature (a marker feature's population)."""
        if kind == 'markers':
            return str(self._trend_row(kind, feature).get('cluster_id') or feature)
        return feature

    def _trend_feature_label(self, kind: str, feature: str) -> str:
        labels = _population_labels(self.state.gate_defs())
        if kind == 'markers':
            row = self._trend_row(kind, feature)
            if row.get('channel'):
                channel = row['channel']
                return f"{row.get('cluster', feature)} {_antigen_map(self.controller).get(channel, channel)}"
        return labels.get(feature, feature)

    def _trend_figure_menu(self, kind: str, feature: str):
        from PySide6.QtGui import QCursor
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        menu.addSection(self._trend_feature_label(kind, feature))
        act_plot = menu.addAction("Show the trend plot in a window")
        act_line = menu.addAction("Show the trend plot with the straight line")
        res = (self.state.trend or {}).get(kind)
        act_percent = None
        if kind == 'freq' and res is not None and not res.settings.get('interaction'):
            act_percent = menu.addAction("Show the trend plot as % of parent")
            act_percent.triggered.connect(
                lambda: self._trend_plot_window(kind, feature, percent=True))
        act_show = menu.addAction("Show the population on the Populations page")
        pop_key = self._trend_population(kind, feature)
        act_report = menu.addAction("Include the population in the report")
        act_report.setCheckable(True)
        act_report.setChecked(pop_key in self.state.report_populations)
        act_plot.triggered.connect(lambda: self._trend_plot_window(kind, feature))
        act_line.triggered.connect(lambda: self._trend_plot_window(kind, feature, show_line=True))
        act_show.triggered.connect(lambda: self.show_population(pop_key))
        act_report.triggered.connect(lambda on: self._set_report_population(pop_key, on))
        if kind != 'markers':
            self._add_composition_action(menu, pop_key)
        menu.exec(QCursor.pos())

    # ------------------------------------------------------------------
    # Composition over the trend variable
    # ------------------------------------------------------------------

    def _composition_unit(self, pop_key: str) -> dict | None:
        """The gate *pop_key* belongs to when a trend run exists and the
        gate has two or more populations (ag_results.composition_gate)."""
        trend = self.state.trend or {}
        if not pop_key or not any(trend.get(k) is not None for k in ('freq', 'counts')):
            return None
        return ag_results.composition_gate(self.state.summaries,
                                           list(trend.get('samples') or []), pop_key)

    def _add_composition_action(self, menu, pop_key: str):
        unit = self._composition_unit(pop_key)
        if unit is None:
            return
        variable = (self.state.trend or {}).get('variable') or 'the variable'
        act = menu.addAction(f"Show the make-up of '{unit['gate']}' over {variable}")
        act.triggered.connect(lambda: self._composition_window(pop_key))

    def _composition_window(self, pop_key: str):
        """The make-up of *pop_key*'s gate against the trend variable in its
        own window, with the fitted and observed shares as CSV."""
        from honeychrome.view_components import trend_plots as tp
        unit = self._composition_unit(pop_key)
        if unit is None:
            return
        summaries = dict(self.state.summaries)
        trend = dict(self.state.trend or {})
        labels = _population_labels(self.state.gate_defs())
        parent = labels.get(unit['parent_key'], unit['parent_key'])
        title = f"{parent}: {unit['gate']} over {trend.get('variable') or 'the variable'}"
        is_dark = _is_dark_palette()
        cache: dict = {}

        def panels():
            if 'panels' not in cache:
                cache['panels'] = ag_results.trend_composition(summaries, trend, pop_key)
            return cache['panels']

        def make():
            return tp.make_composition_figure(panels(), title=title, is_dark=is_dark)

        self._pop_out_figure(make, f"Make-up of {unit['gate']}",
                             values_table=lambda: tp.composition_table(panels()),
                             options_key='composition')

    def _trend_pairing(self, samples: list[str]) -> dict | None:
        """{sample: pairing level} of the current pairing covariate."""
        st = self.state
        if not st.pairing:
            return None
        values = st.covariates.get(st.pairing) or {}
        return {k: str(values[k]).strip() for k in samples
                if not ds.is_missing_value(values.get(k))}

    def _trend_plot_kwargs(self, kind: str, feature: str, show_line: bool = False,
                           percent: bool = False) -> dict | None:
        """Arguments of trend_plots.make_trend_plot_figure (without the
        theme) for one feature of the trend run; None when it is not in it.
        *percent*: the gated frequencies (% of parent) with the fit drawn
        back on that scale (frequencies only, not for a group × trend test).
        Reads the state only, so it can run on a worker thread."""
        from honeychrome.view_components import trend_plots as tp
        trend = self.state.trend or {}
        res = trend.get(kind)
        if percent and (kind != 'freq' or (res is not None
                                           and res.settings.get('interaction'))):
            return None
        data = trend.get('freq_values' if percent else _TREND_DATA_KEYS[kind])
        if res is None or data is None:
            return None
        samples = list(trend.get('samples') or [])
        try:
            inputs = tp.feature_plot_inputs(res, data, samples, trend.get('time') or [],
                                            trend.get('group_vec') or [], feature,
                                            self._trend_pairing(samples))
        except KeyError:
            return None
        return dict(inputs, title=self._trend_feature_label(kind, feature),
                    value_label='% of parent' if percent else _trend_value_label(trend, kind),
                    group_colours=_group_colours(self.state), show_line=show_line,
                    back_transform='percent' if percent else '')

    def _trend_plot_window(self, kind: str, feature: str, show_line: bool = False,
                           percent: bool = False):
        """One feature's trend plot in its own window, with its values as CSV."""
        from honeychrome.view_components import trend_plots as tp
        kwargs = self._trend_plot_kwargs(kind, feature, show_line, percent)
        if kwargs is None:
            return
        is_dark = _is_dark_palette()

        def make():
            return tp.make_trend_plot_figure(**kwargs, is_dark=is_dark)

        def values():
            df2 = kwargs['row'].get('df2', np.inf)
            return tp.trend_plot_table(kwargs['time'], kwargs['values'], kwargs['groups'],
                                       kwargs['samples'], grid=kwargs['grid'],
                                       curve=kwargs['curve'], curve_se=kwargs['curve_se'],
                                       df2=float(df2) if df2 is not None else np.inf,
                                       variable=kwargs['variable'],
                                       back_transform=kwargs['back_transform'],
                                       group_curves=kwargs.get('group_curves'))
        self._pop_out_figure(make, f"Trend {kwargs['title']}", values_table=values,
                             options_key='trend_plot_percent' if percent else 'trend_plot')

    # ------------------------------------------------------------------
    # Population browser
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Sample PCA and predictive analysis
    # ------------------------------------------------------------------

    def _build_explore_page(self):
        layout = QVBoxLayout(self.explore_page)
        note = QLabel("Both use the per-sample values of the last statistics run (population "
                      "% of parent, counts and marker medians), so run Statistics first with "
                      "the matching tests ticked. Figures open on the Figures page.")
        note.setWordWrap(True)
        layout.addWidget(note)

        pbox = QGroupBox("Sample PCA")
        pl = QHBoxLayout(pbox)
        self.chk_pca_freq = QCheckBox("Frequencies")
        self.chk_pca_counts = QCheckBox("Counts")
        self.chk_pca_markers = QCheckBox("Marker medians")
        self.sb_pca_loadings = QSpinBox()
        self.sb_pca_loadings.setRange(1, 50)
        self.sb_pca_loadings.setToolTip("Number of loadings (arrows) shown: the longest.")
        self.btn_pca = QPushButton("Show Sample PCA")
        self.btn_pca.clicked.connect(self._on_show_pca)
        for w in (self.chk_pca_freq, self.chk_pca_counts, self.chk_pca_markers):
            pl.addWidget(w)
        pl.addWidget(QLabel("Loadings:"))
        pl.addWidget(self.sb_pca_loadings)
        pl.addWidget(self.btn_pca)
        pl.addStretch()
        layout.addWidget(pbox)

        dbox = QGroupBox("Predictive analysis")
        dl = QVBoxLayout(dbox)
        dnote = QLabel("Which features best separate two groups, and how well: lasso logistic "
                       "regression and a random forest in repeated, nested cross-validation, "
                       "with the cross-validated ROC curve and AUC, the most informative "
                       "features and the AUC by number of features. Needs at least 10 samples "
                       "per group; 30 or more is recommended.")
        dnote.setWordWrap(True)
        dl.addWidget(dnote)
        drow = QHBoxLayout()
        drow.addWidget(QLabel("Comparison:"))
        self.cb_pred_comparison = QComboBox()
        drow.addWidget(self.cb_pred_comparison, stretch=1)
        dl.addLayout(drow)
        orow = QHBoxLayout()
        self.chk_pred_freq = QCheckBox("Frequencies")
        self.chk_pred_markers = QCheckBox("Marker medians")
        self.chk_pred_lasso = QCheckBox("Lasso")
        self.chk_pred_forest = QCheckBox("Random forest")
        self.sb_pred_repeats = QSpinBox()
        self.sb_pred_repeats.setRange(1, 50)
        self.sb_pred_repeats.setToolTip("Repeats of the outer cross-validation.")
        for w in (self.chk_pred_freq, self.chk_pred_markers, self.chk_pred_lasso,
                  self.chk_pred_forest):
            orow.addWidget(w)
        orow.addWidget(QLabel("Repeats:"))
        orow.addWidget(self.sb_pred_repeats)
        self.btn_pred = QPushButton("Run Predictive Analysis")
        self.btn_pred.clicked.connect(self._on_run_predictive)
        orow.addWidget(self.btn_pred)
        orow.addStretch()
        dl.addLayout(orow)
        self.lbl_pred = QLabel("")
        self.lbl_pred.setWordWrap(True)
        dl.addWidget(self.lbl_pred)
        self.pred_table_holder = QVBoxLayout()
        dl.addLayout(self.pred_table_holder)
        layout.addWidget(dbox)

        abox = QGroupBox("Immune age model")
        al = QVBoxLayout(abox)
        anote = QLabel(
            "How well the immune profile predicts a continuous trend variable (for example age), "
            "which populations carry the prediction, and whether each sample's profile looks "
            "like a higher or lower value than its own (with age: older or younger). An elastic net (penalised regression) is trained and judged by "
            "nested cross-validation on the values of the last trend run, keeping each donor or "
            "pair whole; a chance check reruns it with the variable shuffled. Needs at least 20 "
            "samples in the model's group; 40 or more is recommended. Figures open on the "
            "Figures page in trend mode.")
        anote.setWordWrap(True)
        al.addWidget(anote)
        arow = QHBoxLayout()
        arow.addWidget(QLabel("Model on:"))
        self.cb_age_group = QComboBox()
        self.cb_age_group.setToolTip(
            "The group the model is trained and judged on. Other groups of the trend run can be "
            "predicted by the final model (not cross-validated); that is only meaningful for "
            "samples like the model's (e.g. the same tissue from another cohort), not for a "
            "different kind of sample (e.g. another tissue or species).")
        arow.addWidget(self.cb_age_group, stretch=1)
        al.addLayout(arow)
        aorow = QHBoxLayout()
        self.chk_age_freq = QCheckBox("Frequencies")
        self.chk_age_markers = QCheckBox("Marker medians")
        self.chk_age_others = QCheckBox("Predict the other groups")
        self.chk_age_others.setToolTip(
            "After fitting on the chosen group, predict every sample of the other groups, "
            "including groups left out of the trend run and samples without a value of the "
            "variable.")
        self.sb_age_repeats = QSpinBox()
        self.sb_age_repeats.setRange(1, 50)
        self.sb_age_repeats.setToolTip("Repeats of the outer cross-validation.")
        self.sb_age_perm = QSpinBox()
        self.sb_age_perm.setRange(0, 2000)
        self.sb_age_perm.setSingleStep(50)
        self.sb_age_perm.setToolTip(
            "Times the model is rerun with the variable shuffled among donors or pairs, to "
            "check its error is better than chance (0: no check). About as long as one repeat "
            "each.")
        for w in (self.chk_age_freq, self.chk_age_markers, self.chk_age_others):
            aorow.addWidget(w)
        aorow.addWidget(QLabel("Repeats:"))
        aorow.addWidget(self.sb_age_repeats)
        aorow.addWidget(QLabel("Chance shuffles:"))
        aorow.addWidget(self.sb_age_perm)
        self.btn_age = QPushButton("Run Immune Age Model")
        self.btn_age.clicked.connect(self._on_run_age_model)
        aorow.addWidget(self.btn_age)
        aorow.addStretch()
        al.addLayout(aorow)
        self.age_progress = QProgressBar()
        self.age_progress.setRange(0, 1000)
        self.age_progress.setVisible(False)
        al.addWidget(self.age_progress)
        self.lbl_age = QLabel("")
        self.lbl_age.setWordWrap(True)
        al.addWidget(self.lbl_age)
        self.age_table_holder = QVBoxLayout()
        al.addLayout(self.age_table_holder)
        layout.addWidget(abox)
        layout.addStretch()
        self._age_worker: _CallWorker | None = None

        for w, name in ((self.chk_pca_freq, 'pca_use_freq'), (self.chk_pca_counts, 'pca_use_counts'),
                        (self.chk_pca_markers, 'pca_use_markers'),
                        (self.chk_pred_freq, 'pred_use_freq'),
                        (self.chk_pred_markers, 'pred_use_markers'),
                        (self.chk_pred_lasso, 'pred_lasso'), (self.chk_pred_forest, 'pred_forest'),
                        (self.chk_age_freq, 'age_use_freq'),
                        (self.chk_age_markers, 'age_use_markers'),
                        (self.chk_age_others, 'age_predict_others')):
            w.toggled.connect(lambda v, n=name: self._filling or setattr(self.state, n, bool(v)))
        self.sb_age_repeats.valueChanged.connect(
            lambda v: self._filling or setattr(self.state, 'age_repeats', int(v)))
        self.sb_age_perm.valueChanged.connect(
            lambda v: self._filling or setattr(self.state, 'age_permutations', int(v)))
        self.sb_pca_loadings.valueChanged.connect(
            lambda v: self._filling or setattr(self.state, 'pca_n_loadings', int(v)))
        self.sb_pred_repeats.valueChanged.connect(
            lambda v: self._filling or setattr(self.state, 'pred_repeats', int(v)))

    def _load_explore_controls(self):
        st = self.state
        self._filling = True
        try:
            for w, v in ((self.chk_pca_freq, st.pca_use_freq), (self.chk_pca_counts, st.pca_use_counts),
                         (self.chk_pca_markers, st.pca_use_markers),
                         (self.chk_pred_freq, st.pred_use_freq),
                         (self.chk_pred_markers, st.pred_use_markers),
                         (self.chk_pred_lasso, st.pred_lasso), (self.chk_pred_forest, st.pred_forest)):
                w.setChecked(bool(v))
            self.sb_pca_loadings.setValue(int(st.pca_n_loadings))
            self.sb_pred_repeats.setValue(int(st.pred_repeats))
            for w, v in ((self.chk_age_freq, st.age_use_freq),
                         (self.chk_age_markers, st.age_use_markers),
                         (self.chk_age_others, st.age_predict_others)):
                w.setChecked(bool(v))
            self.sb_age_repeats.setValue(int(st.age_repeats))
            self.sb_age_perm.setValue(int(st.age_permutations))
            prev_group = self.cb_age_group.currentText()
            self.cb_age_group.clear()
            trend_groups = list(((st.trend or {}).get('settings') or {}).get('groups') or [])
            self.cb_age_group.addItems(trend_groups)
            if prev_group in trend_groups:
                self.cb_age_group.setCurrentText(prev_group)
            self.cb_pred_comparison.clear()
            for base, other in (st.stats or {}).get('comparisons') or []:
                self.cb_pred_comparison.addItem(_comparison_label(base, other),
                                                json.dumps([base, other]))
        finally:
            self._filling = False
        stats = st.stats or {}
        pca_source = (st.trend if self._trend_mode() else st.stats) or {}
        self.chk_pca_freq.setEnabled(pca_source.get('freq_values') is not None)
        self.chk_pca_counts.setEnabled(pca_source.get('counts_values') is not None)
        self.chk_pca_markers.setEnabled(pca_source.get('marker_values') is not None)
        self.chk_pred_freq.setEnabled(stats.get('freq_values') is not None)
        self.chk_pred_markers.setEnabled(stats.get('marker_values') is not None)
        self.btn_pca.setEnabled(bool(pca_source))
        self.btn_pred.setEnabled(bool(stats) and self._predictive_worker is None)
        self._show_predictive_table()
        trend = st.trend or {}
        self.chk_age_freq.setEnabled(trend.get('freq_values') is not None)
        self.chk_age_markers.setEnabled(trend.get('marker_values') is not None)
        self.btn_age.setEnabled(bool(trend) and self._age_worker is None)
        self.btn_age.setToolTip("" if trend else "Run the trend test first.")
        self._show_age_table()

    def _feature_labels(self) -> dict:
        """Display names for population keys and channels in feature names."""
        labels = dict(_population_labels(self.state.gate_defs()))
        labels.update(_antigen_map(self.controller))
        return labels

    def _on_show_pca(self):
        st = self.state
        trend_mode = self._trend_mode()
        source = st.trend if trend_mode else st.stats
        if not source:
            QMessageBox.information(self, "Sample PCA",
                                    "Run the trend test first." if trend_mode
                                    else "Run Statistics first.")
            return
        use = (self.chk_pca_freq.isChecked() and self.chk_pca_freq.isEnabled(),
               self.chk_pca_counts.isChecked() and self.chk_pca_counts.isEnabled(),
               self.chk_pca_markers.isChecked() and self.chk_pca_markers.isEnabled())
        if not any(use):
            QMessageBox.information(self, "Sample PCA",
                                    "Tick at least one table that the last statistics run made.")
            return
        try:
            pca = ag_results.sample_pca(source, *use, n_loadings=st.pca_n_loadings,
                                        labels=self._feature_labels())
        except Exception as exc:
            QMessageBox.warning(self, "Sample PCA", str(exc))
            return
        if pca is None:
            QMessageBox.information(self, "Sample PCA",
                                    "At least 2 samples and 2 features that vary are needed.")
            return
        if trend_mode:
            from honeychrome.view_components import trend_plots as tp
            time_of = dict(zip(source.get('samples') or [], source.get('time') or []))
            pca['colour_values'] = [time_of.get(s) for s in pca['scores'].index]
            pca['colour_label'] = tp.variable_label(source.get('variable') or '',
                                                    source.get('unit') or '')
        st.pca = pca
        self._add_extra_figures()
        self._populate_report_items()
        self._open_figure("Sample PCA")
        self.sample_analysis_changed.emit()
        _status(self.bus, f"Sample PCA of {len(pca['scores'])} samples.")

    def _on_run_predictive(self):
        import drc_predictive
        st = self.state
        if self._predictive_worker is not None or not st.stats:
            return
        data = self.cb_pred_comparison.currentData()
        if not data:
            return
        base, other = json.loads(data)
        models = [m for m, on in (('l1_logistic', self.chk_pred_lasso.isChecked()),
                                  ('random_forest', self.chk_pred_forest.isChecked())) if on]
        use_freq = self.chk_pred_freq.isChecked() and self.chk_pred_freq.isEnabled()
        use_markers = self.chk_pred_markers.isChecked() and self.chk_pred_markers.isEnabled()
        if not models or not (use_freq or use_markers):
            QMessageBox.information(self, "Predictive analysis",
                                    "Tick at least one model and one kind of feature.")
            return
        groups = list(st.stats['group_vec'])
        n_per_group = {other: groups.count(other), base: groups.count(base)}
        status, message = drc_predictive.sample_size_status(n_per_group)
        if status == 'block':
            QMessageBox.warning(self, "Predictive analysis", message)
            return
        X = ag_results.predictive_matrix(st.stats, use_freq, use_markers, self._feature_labels())
        repeats = int(st.pred_repeats)

        def _run():
            return drc_predictive.run_predictive(X, groups, other, base, models=tuple(models),
                                                 repeats=repeats)

        self.btn_pred.setEnabled(False)
        self.lbl_pred.setText(f"Running on {X.shape[1]} features, "
                              f"{sum(n_per_group.values())} samples …")
        worker = _CallWorker(_run, parent=self)
        worker.finished.connect(self._on_predictive_finished)
        self._predictive_worker = worker
        worker.start()

    def _on_predictive_finished(self, ok: bool, error: str, result: object):
        self._predictive_worker = None
        self.btn_pred.setEnabled(bool(self.state.stats))
        if not ok:
            self.lbl_pred.setText(f"<span style='color:#b00020'>{error}</span>")
            return
        self.state.predictive = result
        self._show_predictive_table()
        self._add_extra_figures()
        self._populate_report_items()
        self.sample_analysis_changed.emit()
        _status(self.bus, "Predictive analysis complete.")

    def _on_run_age_model(self):
        import drc_predictive
        st = self.state
        trend = st.trend or {}
        if self._age_worker is not None or not trend:
            return
        group = self.cb_age_group.currentText()
        use_freq = self.chk_age_freq.isChecked() and self.chk_age_freq.isEnabled()
        use_markers = self.chk_age_markers.isChecked() and self.chk_age_markers.isEnabled()
        if not (use_freq or use_markers):
            QMessageBox.information(self, "Immune age model",
                                    "Tick at least one kind of feature that the trend run made.")
            return
        samples = list(trend.get('samples') or [])
        group_of = dict(zip(samples, trend.get('group_vec') or []))
        time_of = dict(zip(samples, trend.get('time') or []))
        labels = self._feature_labels()
        X = ag_results.predictive_matrix(trend, use_freq, use_markers, labels)
        key_of = {}
        if use_freq and trend.get('freq_values') is not None:
            key_of.update({f"Freq: {labels.get(c, c)}": c for c in trend['freq_values'].columns})
        if use_markers and trend.get('marker_values') is not None:
            for c in trend['marker_values'].columns:
                pop, _sep, ch = str(c).partition(ag_results.FEATURE_SEP)
                key_of[f"Median: {labels.get(pop, pop)} {labels.get(ch, ch)}"] = pop
        value = pd.to_numeric(pd.Series({s: time_of.get(s) for s in X.index}), errors='coerce')
        in_model = [s for s in X.index if group_of.get(s) == group and pd.notna(value[s])]
        status, message = drc_predictive.age_sample_size_status(len(in_model))
        if status == 'block':
            QMessageBox.warning(self, "Immune age model", message)
            return
        # Samples of the other groups, in the trend run or not, with or
        # without a value of the variable: their features are worked out
        # from the gating results the same way as the trend run's.
        others = ([s for s in self._stats_samples()
                   if s not in in_model and st.sample_groups.get(s) != group]
                  if self.chk_age_others.isChecked() else [])
        other_value = pd.Series(self._trend_values(others), dtype=float).reindex(others)
        other_groups = {s: st.sample_groups.get(s, '') for s in others}
        summaries = {s: st.summaries[s] for s in others}
        gate_defs = deepcopy(st.gate_defs())
        feature_kw = dict(use_stats_parent=st.use_stats_parent,
                          min_parent_events=st.min_parent_events,
                          exclude_gating_markers=st.exclude_gating_markers,
                          min_marker_events=st.min_marker_events)
        pairing = self._trend_pairing(in_model)
        blocks = [pairing.get(s, s) for s in in_model] if pairing else None
        repeats, n_perm = int(st.age_repeats), int(st.age_permutations)
        variable = trend.get('variable') or ''

        def _run(report):
            message = {'text': ''}

            def _said(text):
                message['text'] = text

            X_other = None
            if others:
                source = ag_results.trend_feature_values(
                    summaries, others, gate_defs, use_freq=use_freq, use_markers=use_markers,
                    **feature_kw)
                X_other = ag_results.predictive_matrix(source, use_freq, use_markers, labels)
            res = drc_predictive.run_age_model(
                X.loc[in_model], value[in_model].to_numpy(float), blocks=blocks,
                X_other=X_other,
                y_other=other_value.reindex(X_other.index).to_numpy(float) if others else None,
                repeats=repeats, n_permutations=n_perm, progress=_said,
                on_step=lambda done, total: report(done / max(total, 1),
                                                   f"{message['text']} ({done} of {total} fits)"))
            res['importance']['key'] = [key_of.get(f, '') for f in res['importance']['feature']]
            res['groups'] = dict({s: group for s in in_model}, **other_groups)
            res['summary'].update({'group': group, 'variable': variable,
                                   'features': ', '.join(n for n, on in (
                                       ('frequencies', use_freq), ('marker medians', use_markers))
                                       if on)})
            return res

        self.btn_age.setEnabled(False)
        self.lbl_age.setText(f"Running on {X.shape[1]} features, {len(in_model)} samples "
                             f"({n_perm} chance shuffles) …")
        worker = _ProgressCallWorker(_run, parent=self)
        worker.progress.connect(self._on_age_progress)
        worker.finished.connect(lambda ok, err, res, run_id=trend.get('run_id'):
                                self._on_age_model_finished(ok, err, res, run_id))
        self._age_worker = worker
        self.age_progress.setValue(0)
        self.age_progress.setVisible(True)
        worker.start()

    def _on_age_progress(self, fraction: float, message: str):
        self.age_progress.setValue(int(1000 * min(max(fraction, 0.0), 1.0)))
        self.lbl_age.setText(message)

    def _on_age_model_finished(self, ok: bool, error: str, result: object, run_id):
        self._age_worker = None
        self.age_progress.setVisible(False)
        trend = self.state.trend or {}
        self.btn_age.setEnabled(bool(trend))
        if not ok:
            self.lbl_age.setText(f"<span style='color:#b00020'>{error}</span>")
            return
        if trend.get('run_id') != run_id:
            self.lbl_age.setText("The trend test was rerun meanwhile: run the model again.")
            return
        self.state.trend = dict(trend, age_model=result)
        self._show_age_table()
        self._draw_figures()
        self._populate_report_items()
        self.trend_finished.emit()
        _status(self.bus, "Immune age model complete.")

    def _show_age_table(self):
        """The last immune age model: its summary line and performance."""
        _clear_layout(self.age_table_holder)
        res = (self.state.trend or {}).get('age_model')
        if not res:
            if self._age_worker is None:
                self.lbl_age.setText("")
            return
        perf = res['performance'].iloc[0]
        summary = res['summary']
        if self._age_worker is None:
            text = (f"{summary.get('group', '')}: {summary['n']} samples, "
                    f"{summary['n_features']} features ({summary.get('features', '')}).")
            if summary.get('status') == 'warn':
                text += f" {summary.get('message', '')}"
            self.lbl_age.setText(text)
        unit = (self.state.trend or {}).get('unit') or ''
        p_chance = perf.get('p_chance')
        rows = [{'Mean absolute error': f"{perf['mae_mean']:.3g} ± {perf['mae_sd']:.2g} {unit}",
                 'Predicting the mean': f"{perf['mae_baseline']:.3g} {unit}",
                 'R²': f"{perf['r2_mean']:.2f} ± {perf['r2_sd']:.2f}",
                 'Spearman': f"{perf['spearman_mean']:.2f}",
                 'Chance p': (f"{p_chance:.3g}" if p_chance is not None and np.isfinite(p_chance)
                              else 'not run')}]
        table = CopyableTableWidget(rows, list(rows[0]))
        table.setMaximumHeight(80)
        self.age_table_holder.addWidget(table)
        self._add_age_group_table(res, unit)

    def _age_groups(self) -> dict:
        trend = self.state.trend or {}
        groups = dict(zip(trend.get('samples') or [], trend.get('group_vec') or []))
        groups.update((trend.get('age_model') or {}).get('groups') or {})
        return groups

    def _add_age_group_table(self, res: dict, unit: str):
        """How the model fits each group (its own, cross-validated, and any
        group predicted by the final model), and a button listing every
        sample's predicted value."""
        import drc_predictive
        from honeychrome.view_components.copyable_table_widget import show_frame_dialog
        by_group = drc_predictive.age_model_by_group(res, self._age_groups())
        rows = drc_predictive.age_group_rows(by_group, unit)
        if rows:
            table = CopyableTableWidget(rows, list(rows[0]))
            table.setMaximumHeight(36 + 26 * len(rows))
            self.age_table_holder.addWidget(table)
        btn = QPushButton("Show every sample's prediction…")
        btn.setToolTip("Each sample's actual and predicted value, the gap and the corrected gap, "
                       "with its group; copy or save as CSV.")
        variable = (self.state.trend or {}).get('variable') or 'variable'
        btn.clicked.connect(lambda: show_frame_dialog(
            self, f"Immune age model: predicted {variable}",
            drc_predictive.age_predictions(res, self._age_groups()),
            str(self._loaded_experiment_dir_or_home() / 'immune_age_predictions.csv')))
        self.age_table_holder.addWidget(btn)

    def _loaded_experiment_dir_or_home(self) -> Path:
        exp = getattr(self.controller, 'experiment_dir', '') or ''
        return Path(exp) if exp else Path.home()

    def _show_predictive_table(self):
        """The last predictive analysis: its summary line and performance
        table."""
        _clear_layout(self.pred_table_holder)
        res = self.state.predictive
        if not res:
            return
        summary = res['summary']
        if self._predictive_worker is None:
            text = (f"Minimal panel: {summary['minimal_panel']} feature(s) (best panel AUC "
                    f"{summary['best_panel_auc']:.2f}).")
            if summary.get('status') == 'warn':
                text += f" {summary.get('message', '')}"
            self.lbl_pred.setText(text)
        rows = [{'Model': r['model'],
                 'AUC': f"{r['auc_mean']:.3f} ± {r['auc_sd']:.3f}",
                 'Balanced accuracy': f"{r['balanced_accuracy_mean']:.3f} ± "
                                      f"{r['balanced_accuracy_sd']:.3f}"}
                for _i, r in res['performance'].iterrows()]
        table = CopyableTableWidget(rows, ['Model', 'AUC', 'Balanced accuracy'])
        table.setMaximumHeight(110)
        self.pred_table_holder.addWidget(table)

    def _extra_figures(self) -> list:
        """(title, figure, function that builds a fresh copy) of the sample
        PCA and predictive analysis."""
        st = self.state
        is_dark = _is_dark_palette()
        out = []
        try:
            makers = []
            if st.pca:
                pca, colours = st.pca, _group_colours(st)
                makers.append(("Sample PCA", lambda: ag_report.make_pca_figure(
                    pca, is_dark, group_colours=colours)))
            res = st.predictive
            if res:
                summary = res['summary']
                makers += [
                    ("Predictive: ROC", lambda: ag_report.make_roc_figure(
                        res['performance'], res['roc'], summary, is_dark)),
                    ("Predictive: features", lambda: ag_report.make_importance_figure(
                        res['importance'], summary, is_dark)),
                    ("Predictive: panel size", lambda: ag_report.make_panel_figure(
                        res['panel'], summary, is_dark)),
                ]
            for title, make in makers:
                out.append((title, make(), make))
        except Exception as exc:
            log.exception("sample PCA / predictive figure failed")
            _status(self.bus, f"Figure failed: {exc}")
        return out

    def _add_extra_figures(self):
        """Show the PCA and predictive figures on the Figures page, replacing
        earlier ones."""
        titles = {"Sample PCA", "Predictive: ROC", "Predictive: features", "Predictive: panel size"}
        for i in reversed(range(self.figures_tabs.count())):
            if self.figures_tabs.tabText(i) in titles:
                self.figures_tabs.removeTab(i)
        exp_dir = getattr(self.controller, 'experiment_dir', '')
        for title, fig, make in self._extra_figures():
            key = _EXTRA_FIGURE_KEYS.get(title, title)
            widget = ScalableFigureWidget(
                fig, title=title, experiment_dir=exp_dir,
                pop_out=lambda make=make, title=title, k=key: self._pop_out_figure(
                    make, title, options_key=k),
                parent=self, options=figure_options_ref(self.controller, key))
            handler = getattr(fig, '_hover_handler', None)
            if handler is not None:
                widget.canvas.mpl_connect('motion_notify_event', handler)
            self.figures_tabs.addTab(widget, title)

    def _build_browser_page(self):
        layout = QHBoxLayout(self.browser_page)
        left = QWidget()
        left.setFixedWidth(280)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("<b>Populations</b>"))
        self.pop_filter = FilterBar(POP_FILTERS, placeholder="Search populations…", vertical=True)
        self.pop_filter.setToolTip("Significant: populations whose frequency differs between the "
                                   "comparison's groups at the FDR threshold.")
        self.pop_filter.changed.connect(self._on_pop_filter_changed)
        ll.addWidget(self.pop_filter)
        self.pop_list = QListWidget()
        self.pop_list.currentItemChanged.connect(lambda cur, _prev: self._draw_population(cur))
        ll.addWidget(self.pop_list)
        self.lbl_pop_count = QLabel("")
        self.lbl_pop_count.setStyleSheet("color: #555; font-style: italic;")
        ll.addWidget(self.lbl_pop_count)
        layout.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        self.lbl_pop_title = QLabel("")
        self.lbl_pop_title.setWordWrap(True)
        head.addWidget(self.lbl_pop_title, stretch=1)
        self.chk_pop_report = QCheckBox("Include in report")
        self.chk_pop_report.setToolTip("Add this population's group comparison (biplots and the "
                                       "dot and violin plot) to the report.")
        self.chk_pop_report.toggled.connect(self._on_pop_report_toggled)
        head.addWidget(self.chk_pop_report)
        self.btn_pop_composition = QPushButton("Make-up over the variable")
        self.btn_pop_composition.setToolTip(
            "The share of each population of this gate (and the rest of the parent) against the "
            "trend variable: fitted shares as a stacked area, each sample's shares below.")
        self.btn_pop_composition.clicked.connect(
            lambda: self._shown_population and self._composition_window(self._shown_population))
        self.btn_pop_composition.setVisible(False)
        head.addWidget(self.btn_pop_composition)
        self.btn_pop_window = QPushButton("⤢ Open in window")
        self.btn_pop_window.clicked.connect(
            lambda: self._shown_population and self._population_window(self._shown_population))
        head.addWidget(self.btn_pop_window)
        self.btn_pop_save = QPushButton("Save image…")
        self.btn_pop_save.setToolTip("Save this population's group biplots and dot and violin "
                                     "plot as one image (PNG, PDF or SVG).")
        self.btn_pop_save.clicked.connect(
            lambda: self._shown_population and self._save_population_image(self._shown_population))
        head.addWidget(self.btn_pop_save)
        rl.addLayout(head)
        self.pop_scroll = QScrollArea()
        self.pop_scroll.setWidgetResizable(True)
        self.pop_host = QWidget()
        self.pop_view = QHBoxLayout(self.pop_host)
        self.pop_view.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.pop_scroll.setWidget(self.pop_host)
        rl.addWidget(self.pop_scroll, stretch=1)
        layout.addWidget(right, stretch=1)

    def _significant_populations(self) -> set | None:
        """Population keys significant in the current comparison's frequency
        test (in trend mode: called as changing with the variable), or None
        without one."""
        if self._trend_mode():
            res = (self.state.trend or {}).get('freq')
            if res is None:
                return None
            return set(res.table.loc[res.table['significant'].astype(bool), 'feature'].astype(str))
        comp = self._current_comparison()
        df = (self.state.stats or {}).get('freq')
        if comp is None or df is None:
            return None
        view = _comparison_view(df, _comparison_label(*comp))
        if view is None or 'significant' not in view.columns:
            return None
        return set(view.loc[view['significant'].astype(bool), 'feature'])

    def _populate_population_list(self):
        current = self.pop_list.currentItem().data(Qt.UserRole) if self.pop_list.currentItem() else None
        if self.pop_filter.mode() != self.state.pop_filter:
            self.pop_filter.blockSignals(True)
            self.pop_filter.set_mode(self.state.pop_filter)
            self.pop_filter.blockSignals(False)
        significant = self._significant_populations()
        only_sig = self.pop_filter.mode() == 'significant' and significant is not None
        text = self.pop_filter.text()
        self.pop_list.blockSignals(True)
        self.pop_list.clear()
        labels = _population_labels(self.state.gate_defs())
        n_all = 0
        for g in ag_core.ordered_gate_defs(self.state.gate_defs()):
            for pop in (g.get('populations') or {}):
                key = ag_core.population_key(g['gate_name'], pop)
                n_all += 1
                if only_sig and key not in significant:
                    continue
                label = labels.get(key, key)
                if not text_matches(text, label):
                    continue
                it = QListWidgetItem(("★ " if significant and key in significant else '') + label)
                it.setData(Qt.UserRole, key)
                if key in self.state.report_populations:
                    it.setToolTip("Included in the report")
                    font = it.font()
                    font.setBold(True)
                    it.setFont(font)
                self.pop_list.addItem(it)
                if key == current:
                    self.pop_list.setCurrentItem(it)
        self.pop_list.blockSignals(False)
        if significant is None:
            note = ("Run the trend test to see which populations change; all are listed."
                    if self._trend_mode() else
                    "Run statistics to see which populations differ; all are listed.")
        elif self._trend_mode():
            note = (f"{self.pop_list.count()} of {n_all} populations ({len(significant)} "
                    f"changing with {(self.state.trend or {}).get('variable', '')}).")
        else:
            note = f"{self.pop_list.count()} of {n_all} populations ({len(significant)} significant)."
        self.lbl_pop_count.setText(note)
        self._draw_population(self.pop_list.currentItem())

    def _on_pop_filter_changed(self, _text: str, mode: str):
        self.state.pop_filter = mode or 'significant'
        self._populate_population_list()

    def show_population(self, pop_key: str, include: bool = False):
        """Show a population on the Populations page (clearing the filter
        when it hides it); with *include*, tick it for the report."""
        if include:
            self._set_report_population(pop_key, True)
        found = self._select_population(pop_key)
        if not found:
            self.pop_filter.clear()
            self._select_population(pop_key)
        self.toolbox.setCurrentIndex(self._page_browser)
        win = self.window()
        if win is not None:
            win.raise_()
            win.activateWindow()

    def _select_population(self, pop_key: str) -> bool:
        for i in range(self.pop_list.count()):
            if self.pop_list.item(i).data(Qt.UserRole) == pop_key:
                self.pop_list.setCurrentRow(i)
                return True
        return False

    def _set_report_population(self, pop_key: str, on: bool):
        pops = list(self.state.report_populations)
        if on and pop_key not in pops:
            pops.append(pop_key)
        elif not on and pop_key in pops:
            pops.remove(pop_key)
        else:
            return
        self.state.report_populations = pops
        if self._shown_population == pop_key:
            self.chk_pop_report.blockSignals(True)
            self.chk_pop_report.setChecked(on)
            self.chk_pop_report.blockSignals(False)
        for i in range(self.pop_list.count()):
            it = self.pop_list.item(i)
            if it.data(Qt.UserRole) == pop_key:
                font = it.font()
                font.setBold(on)
                it.setFont(font)
        self._populate_report_items()
        self.report_populations_changed.emit()

    def _on_pop_report_toggled(self, on: bool):
        if self._shown_population:
            self._set_report_population(self._shown_population, bool(on))

    def _comparison_groups(self) -> list[str]:
        """Groups compared on the Populations page: the current comparison's,
        else every group with gated samples."""
        st = self.state
        if self._trend_mode() and st.trend:
            order = list(dict.fromkeys(st.trend.get('group_vec') or []))
            return [g for g in st.group_names if g in order] or order
        comp = self._current_comparison()
        if comp is not None:
            return [comp[0], comp[1]]
        have = {st.sample_groups.get(k) for k in st.gated_samples()}
        return [g for g in st.group_names if g in have]

    def _population_request(self) -> tuple:
        """(groups, comparison) the Populations page compares, read from the
        page's widgets (main thread); no comparison in trend mode."""
        comp = None if self._trend_mode() else self._current_comparison()
        return self._comparison_groups(), comp

    def _population_data(self, pop_key: str, request: tuple | None = None) -> dict | None:
        """Everything the comparison of one population shows. *request* is
        _population_request() taken on the main thread (read here when not
        given); with it, this only reads the state, so it can run in a
        worker thread."""
        st = self.state
        gate_name, pop = ag_core.split_population_key(pop_key)
        by_name = ag_core.gates_by_name(st.gate_defs())
        gdef = by_name.get(gate_name)
        if gdef is None:
            return None
        eff = ag_core.effective_gate_def(gdef, by_name)
        groups, comp = request if request is not None else self._population_request()
        panels = []
        for g in groups:
            samples = [k for k in st.gated_samples() if st.sample_groups.get(k) == g]
            panels.append(self._pooled_group(eff, pop, g, samples))
        values = ag_results.group_population_values(st.summaries, st.gated_samples(),
                                                    st.sample_groups, pop_key, st.use_stats_parent)
        values = {g: values.get(g, []) for g in groups}
        result = ag_results.population_result(st.stats, pop_key, _comparison_label(*comp),
                                               st.fdr_scope) if comp else None
        label = _population_labels(st.gate_defs()).get(pop_key, pop_key)
        title = f"{label}" + (f" — {_comparison_label(*comp)}" if comp else '')
        p_text = ag_report.p_value_text(result)
        trend = trend_percent = None
        if comp is None and self._trend_mode() and st.trend:
            from honeychrome.view_components import trend_plots as tp
            trend = self._trend_plot_kwargs('freq', pop_key)
            trend_percent = self._trend_plot_kwargs('freq', pop_key, percent=True)
            title = f"{label} — trend with {st.trend.get('variable', '')}"
            p_text = tp.trend_p_text(self._trend_row('freq', pop_key) or None)
        return {'gate_def': eff, 'pop': pop, 'panels': panels, 'values': values, 'groups': groups,
                'result': result, 'label': label, 'title': title, 'p_text': p_text,
                'trend': trend, 'trend_percent': trend_percent}

    _POP_CACHE_SIZE = 32

    def _pooled_group(self, gate_def: dict, pop: str, group: str, samples: list[str]) -> dict:
        """One group's pooled panel of a population (ag_report.pooled_population),
        its events cut to the channels the gate is plotted on, which is all
        its figures use. Kept for the current results."""
        st = self.state
        gate_name = gate_def['gate_name']
        key = (gate_name, pop, group, tuple(samples), st.results_revision)
        with self._pop_cache_lock:
            hit = self._pop_cache.get(key)
        if hit is not None:
            return hit
        events, parent, member, channels = ag_report.pooled_population(
            st.display, samples, gate_def, pop, max_events=GROUP_PLOT_EVENTS, seed=DISPLAY_SEED)
        boundary = st.reference_boundaries().get(gate_name, {})
        keep = [c for c in ag_core.boundary_plot_channels(gate_def, boundary) if c and c in channels]
        if events is not None and keep:
            events = np.ascontiguousarray(events[:, [channels.index(c) for c in keep]])
            channels = keep
        panel = {'group': group, 'events': events, 'parent': parent, 'member': member,
                 'channels': channels, 'n_samples': sum(1 for k in samples if k in st.display),
                 'boundary': boundary}
        with self._pop_cache_lock:
            self._pop_cache[key] = panel
            while len(self._pop_cache) > self._POP_CACHE_SIZE:
                del self._pop_cache[next(iter(self._pop_cache))]
        return panel

    def _draw_population(self, item):
        _clear_layout(self.pop_view)
        self._shown_population = item.data(Qt.UserRole) if item is not None else None
        self.chk_pop_report.blockSignals(True)
        self.chk_pop_report.setChecked(self._shown_population in self.state.report_populations)
        self.chk_pop_report.setEnabled(self._shown_population is not None)
        self.chk_pop_report.blockSignals(False)
        self.btn_pop_window.setEnabled(self._shown_population is not None)
        self.btn_pop_save.setEnabled(self._shown_population is not None)
        self.btn_pop_composition.setVisible(
            self._trend_mode() and self._composition_unit(self._shown_population) is not None)
        if self._shown_population is None:
            self.lbl_pop_title.setText("")
            return
        self._request_population(self._shown_population)

    def _request_population(self, pop_key: str):
        """Gather the shown population's data in the background, then draw it
        (_on_population_data). While a request runs, a newer selection waits
        for it to finish."""
        if self._pop_worker is not None:
            return
        request = self._population_request()
        self.lbl_pop_title.setText("<i>Pooling the groups' events</i>")
        worker = _CallWorker(lambda k=pop_key, r=request: self._population_data(k, r),
                             parent=self)
        worker.finished.connect(lambda ok, error, data, k=pop_key:
                                self._on_population_data(ok, error, data, k))
        self._pop_worker = worker
        worker.start()

    def _on_population_data(self, ok: bool, error: str, data, pop_key: str):
        self._pop_worker = None
        if self._shown_population != pop_key:
            # Another population was selected meanwhile.
            _clear_layout(self.pop_view)
            if self._shown_population is not None:
                self._request_population(self._shown_population)
            return
        _clear_layout(self.pop_view)
        if not ok:
            self.lbl_pop_title.setText(f"The population could not be shown: {error}")
            return
        self._fill_population_view(self.pop_view, pop_key, self.lbl_pop_title, data=data)

    def _fill_population_view(self, layout, pop_key: str, title_label: QLabel,
                              data: dict | None = None):
        """Each compared group's pooled biplot, then the dot and violin plot
        (*data*: _population_data, gathered here when not given)."""
        from honeychrome.view_components.scalable_figure_widget import scrollable_canvas_class
        if data is None:
            data = self._population_data(pop_key)
        if data is None:
            title_label.setText("")
            return
        title_label.setText(f"<b>{data['title']}</b><br>{data['p_text']}")
        st = self.state
        if not data['groups']:
            note = QLabel("Assign samples to groups on the Groups page to compare them.")
            note.setWordWrap(True)
            layout.addWidget(note)
            return
        transforms = self._result_transforms()
        labels = build_display_label_map(
            _experiment_channels(self.controller),
            self.controller.experiment.process.get('spectral_model') or [])
        if not st.display:
            note = QLabel("The plots of these results are not available; run gating again to "
                          "see the groups' biplots.")
            note.setWordWrap(True)
            layout.addWidget(note)
        else:
            for panel in data['panels']:
                member = panel['member']
                frac = float(member.mean()) if member is not None and len(member) else None
                vals = [v for _s, v in data['values'].get(panel['group'], [])]
                footer = (f"{len(panel['events']) if panel['events'] is not None else 0:,} events "
                          f"pooled from {panel['n_samples']} sample(s)")
                if vals:
                    footer += f"; mean {np.mean(vals):.2f}% of parent"
                tile = _make_gate_tile(
                    data['gate_def'], panel['boundary'], panel['events'], panel['channels'],
                    panel['parent'], transforms, labels, f"<b>{panel['group']}</b>",
                    fractions={data['pop']: frac} if frac is not None else None,
                    highlight=data['pop'], footer=footer)
                layout.addWidget(tile)
        import honeychrome.settings as hc_settings
        size = int(getattr(hc_settings, 'cytometry_plot_width_target_retrieved', 220) or 220)
        if data.get('trend'):
            for kwargs, name in ((data['trend'], "Trend"),
                                 (data.get('trend_percent'), "Trend, % of parent")):
                if kwargs:
                    layout.addWidget(self._trend_tile(kwargs, name))
            return
        else:
            fig = ag_report.make_group_values_figure(
                data['values'], data['groups'], f"{data['label']}: % of parent",
                data['p_text'], is_dark=_is_dark_palette(),
                group_colours=_group_colours(self.state))
            canvas = scrollable_canvas_class()(fig)
            canvas.setFixedSize(int(size * 1.3), size + 60)
        handler = getattr(fig, '_hover_handler', None)
        if handler is not None:
            canvas.mpl_connect('motion_notify_event', handler)
        layout.addWidget(canvas)

    def _trend_tile(self, kwargs: dict, name: str) -> QWidget:
        """A trend plot as a tile the size and look of the gate tiles: a
        title above a square, transparent plot."""
        import honeychrome.settings as hc_settings
        from honeychrome.view_components import trend_plots as tp
        from honeychrome.view_components.scalable_figure_widget import scrollable_canvas_class
        size = int(getattr(hc_settings, 'cytometry_plot_width_target_retrieved', 220) or 220)
        container = QWidget()
        container.setFixedSize(*_gate_tile_size())
        layout = QVBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        lbl = QLabel(f"<b>{name}</b>")
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)
        fig = tp.make_trend_plot_figure(**dict(kwargs, title=''), is_dark=_is_dark_palette(),
                                        figsize=(size / 100.0, size / 100.0), compact=True,
                                        transparent=True)
        canvas = scrollable_canvas_class()(fig)
        canvas.setFixedSize(size, size)
        canvas.setStyleSheet("background: transparent;")
        handler = getattr(fig, '_hover_handler', None)
        if handler is not None:
            canvas.mpl_connect('motion_notify_event', handler)
        layout.addWidget(canvas, stretch=1)
        return container

    def _population_window(self, pop_key: str):
        """The comparison of one population in its own window."""
        dialog = QDialog(self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setWindowTitle(_population_labels(self.state.gate_defs()).get(pop_key, pop_key))
        layout = QVBoxLayout(dialog)
        title = QLabel("")
        title.setWordWrap(True)
        layout.addWidget(title)
        host = QWidget()
        row = QHBoxLayout(host)
        row.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(host, stretch=1)
        chk = QCheckBox("Include in report")
        chk.setChecked(pop_key in self.state.report_populations)
        chk.toggled.connect(lambda on, k=pop_key: self._set_report_population(k, bool(on)))
        footer = QHBoxLayout()
        footer.addWidget(chk)
        footer.addStretch()
        save = QPushButton("Save image…")
        save.clicked.connect(lambda _c=False, k=pop_key: self._save_population_image(k))
        footer.addWidget(save)
        layout.addLayout(footer)
        self._fill_population_view(row, pop_key, title)
        dialog.show()

    def _population_report_figure(self, pop_key: str, request: tuple | None = None):
        """The report figure of one population's group comparison (*request*:
        _population_request() taken on the main thread)."""
        data = self._population_data(pop_key, request)
        if data is None:
            return None
        labels = build_display_label_map(
            _experiment_channels(self.controller),
            self.controller.experiment.process.get('spectral_model') or [])
        return ag_report.make_population_comparison_figure(
            data['gate_def'], data['pop'], data['panels'] if self.state.display else [],
            self._result_transforms(), labels, data['values'], data['groups'], data['title'],
            data['p_text'], group_colours=_group_colours(self.state), trend=data.get('trend'),
            trend_percent=data.get('trend_percent'))

    def _save_population_image(self, pop_key: str):
        """Save one population's group biplots and dot and violin plot as a
        single PNG, PDF or SVG image."""
        import re
        import honeychrome.settings as hc_settings
        fig = self._population_report_figure(pop_key)
        if fig is None:
            QMessageBox.warning(self, "Save image", "This population has no figure to save.")
            return
        fmt = str(getattr(hc_settings, 'graphics_export_format_retrieved', 'png') or 'png').lower()
        if fmt not in ('png', 'pdf', 'svg'):
            fmt = 'png'
        comp = self._population_request()[1]
        name = _population_labels(self.state.gate_defs()).get(pop_key, pop_key)
        if comp:
            name += f" {_comparison_label(*comp)}"
        elif self._trend_mode() and self.state.trend:
            name += f" trend {self.state.trend.get('variable', '')}"
        stem = re.sub(r'[^\w.\-]+', '_', name).strip('_')
        start = str(Path(getattr(self.controller, 'experiment_dir', '') or '.') / f"{stem or 'population'}.{fmt}")
        filters = "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        path, selected = QFileDialog.getSaveFileName(
            self, "Save population image", start, filters,
            {'png': "PNG (*.png)", 'pdf': "PDF (*.pdf)", 'svg': "SVG (*.svg)"}[fmt])
        if not path:
            return
        if Path(path).suffix.lower() not in ('.png', '.pdf', '.svg'):
            path += '.' + selected.split('*.')[-1].rstrip(')')
        try:
            fig.savefig(path, dpi=300, facecolor='white', bbox_inches='tight')
        except Exception as exc:
            QMessageBox.warning(self, "Save image", f"Could not save the image: {exc}")
            return
        _status(self.bus, f"Population image written to {path}")

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
            "item below and one PDF containing all of them. Tables longer than "
            f"{ag_report.PDF_MAX_ROWS} rows or wider than {ag_report.PDF_MAX_COLUMNS} columns "
            "are left out of the PDF, which names their CSV file instead. Gates are drawn "
            f"{ag_report.GATES_PER_PAGE} to a page; with more than "
            f"{ag_report.REPORT_GATES_DEFAULT_LIMIT} gates they start unticked. Populations "
            "ticked for the report on the Populations page are listed under Populations.")
        blurb.setWordWrap(True)
        layout.addWidget(blurb)
        csv_box = QGroupBox("CSV files")
        cb = QVBoxLayout(csv_box)
        csv_note = QLabel(
            "Each gating run and each statistics run is also written automatically to the "
            "experiment's <i>Automated_Gating_Results</i> folder, named after the model with a "
            "timestamp: the population counts and marker medians of every gated sample, and "
            "the statistics tables (results of every test, and the per-sample values tested).")
        csv_note.setWordWrap(True)
        cb.addWidget(csv_note)
        csv_row = QHBoxLayout()
        self.btn_export_results = QPushButton("Export Gating Results…")
        self.btn_export_results.setToolTip("Write the population counts and marker medians of "
                                           "every gated sample to CSV files.")
        self.btn_export_results.clicked.connect(lambda: self._export_csv('results'))
        self.btn_export_stats = QPushButton("Export Statistics…")
        self.btn_export_stats.setToolTip("Write the last statistics run's tables to CSV files.")
        self.btn_export_stats.clicked.connect(lambda: self._export_csv('stats'))
        csv_row.addWidget(self.btn_export_results)
        csv_row.addWidget(self.btn_export_stats)
        csv_row.addStretch()
        cb.addLayout(csv_row)
        layout.addWidget(csv_box)
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
            transforms=self._result_transforms(),
            axis_labels=build_display_label_map(
                _experiment_channels(self.controller),
                self.controller.experiment.process.get('spectral_model') or []),
            stats=st.stats, pval_threshold=st.pval_threshold, fc_threshold=st.fc_threshold,
            marker_threshold=st.marker_threshold, fdr_scope=st.fdr_scope,
            build_figures=build_result_figures, pca=st.pca or None,
            group_colours=_group_colours(st),
            predictive=st.predictive or None,
            populations=[(k, _population_labels(st.gate_defs()).get(k, k))
                         for k in st.report_populations],
            population_figure=lambda key, r=self._population_request():
                self._population_report_figure(key, r),
            enrichment=st.fsea or None, trend_enrichment=st.fsea_trend or None,
            trend=st.trend or None, build_trend_figures=build_trend_figures,
        )

    def _report_items_with_options(self) -> list:
        """Report items whose figures are drawn with the figure options of
        the figures they show (figure_options)."""
        from honeychrome.view_components import figure_options as fo
        items = self._report_items()
        for item in items:
            if getattr(item, 'get_figure', None) is not None:
                item.get_figure = fo.with_options(
                    item.get_figure, figure_options_ref(self.controller,
                                                        report_figure_key(item.key)))
        return items

    def _export_csv(self, what: str):
        """Ask for a folder and write the gating results or the statistics
        tables there as CSV files."""
        st = self.state
        if what == 'stats' and not st.stats:
            QMessageBox.information(self, "Export Statistics", "Run statistics first.")
            return
        if what == 'results' and not st.summaries:
            QMessageBox.information(self, "Export Gating Results", "Gate some samples first.")
            return
        start = str(Path(getattr(self.controller, 'experiment_dir', '')) / EXPORT_DIR)
        folder = QFileDialog.getExistingDirectory(self, "Export to folder", start)
        if not folder:
            return
        from PySide6.QtWidgets import QApplication
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        labels = _population_labels(st.gate_defs())
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if what == 'stats':
                written = write_statistics_csv(Path(folder), _export_stem(st.model_path), stamp,
                                               st.stats, labels, _antigen_map(self.controller))
            else:
                written = write_results_csv(Path(folder), _export_stem(st.model_path), stamp,
                                            st.summaries, st.gated_samples(), st.sample_groups,
                                            labels, _antigen_map(self.controller))
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            log.exception("CSV export failed")
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        QApplication.restoreOverrideCursor()
        self.lbl_export.setText("Written: " + ", ".join(p.name for p in written))
        _status(self.bus, f"CSV files written to {folder}")

    def _populate_report_items(self):
        """Fill the report's tick list. Items listed before keep their tick;
        new items start ticked, except the gates when there are more than
        ag_report.REPORT_GATES_DEFAULT_LIMIT of them."""
        known = {}
        for i in range(self.report_list.count()):
            it = self.report_list.item(i)
            if it.flags() & Qt.ItemIsUserCheckable:
                known[it.data(Qt.UserRole)] = it.checkState() == Qt.Checked
        self.report_list.clear()
        items = self._report_items()
        if not items:
            placeholder = QListWidgetItem("(nothing to report yet — gate some samples)")
            placeholder.setFlags(Qt.ItemIsEnabled)
            self.report_list.addItem(placeholder)
        gates_off = sum(1 for item in items if item.tab == 'Gates') > \
            ag_report.REPORT_GATES_DEFAULT_LIMIT
        for item in items:
            it = QListWidgetItem(f"{item.tab} — {item.label}")
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            on = known.get(item.key, not (gates_off and item.tab == 'Gates'))
            it.setCheckState(Qt.Checked if on else Qt.Unchecked)
            it.setData(Qt.UserRole, item.key)
            self.report_list.addItem(it)
        self.btn_report.setEnabled(bool(self.state.summaries))

    def _set_report_checks(self, on: bool):
        for i in range(self.report_list.count()):
            it = self.report_list.item(i)
            if it.flags() & Qt.ItemIsUserCheckable:
                it.setCheckState(Qt.Checked if on else Qt.Unchecked)

    def _on_generate_report(self):
        """Write the report in the background (write_report) under the
        heavy-job gate, with a progress dialog whose Cancel stops it."""
        from PySide6.QtWidgets import QProgressDialog
        import drc_report

        st = self.state
        if not st.summaries:
            QMessageBox.information(self, "Nothing to report", "Gate some samples first.")
            return
        if self._report_worker is not None:
            return
        checked = {self.report_list.item(i).data(Qt.UserRole)
                   for i in range(self.report_list.count())
                   if self.report_list.item(i).checkState() == Qt.Checked}
        items = [it for it in self._report_items_with_options() if it.key in checked]
        samples = st.gated_samples()
        stem = drc_report.sanitize_filename(Path(st.model_path).stem or 'automated_gating')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = Path(self.controller.experiment_dir) / 'Automated_Gating_Reports' / f"{stem}_{timestamp}"
        token = acquire_job(self.controller, 'Automated Gating: report')
        if token is None:
            self.lbl_export.setText(self.controller.job_gate.busy_text())
            return
        pages = ag_report.gate_pages(items)
        job = dict(
            report_dir=report_dir, stem=stem, items=pages,
            title_lines=[
                f"Experiment: {self.controller.experiment_dir}",
                f"Model: {Path(st.model_path).name}",
                f"Samples gated: {len(samples)}",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Items included: {len(items)}",
            ],
            settings_text=self._settings_text(samples), summaries=st.summaries,
            samples=samples, groups=dict(st.sample_groups),
            antigens=_antigen_map(self.controller))

        progress = QProgressDialog("Generating report…", "Cancel", 0, len(pages) + 3, self)
        progress.setWindowTitle("Generate Report")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)
        worker = _ReportWorker(job, parent=self)

        def _step(step: int, text: str):
            if progress.wasCanceled():
                worker.cancel()
                return
            progress.setValue(step)
            progress.setLabelText(f"{text} …")

        worker.progress.connect(_step)
        progress.canceled.connect(worker.cancel)
        worker.finished.connect(lambda ok, error, cancelled: self._on_report_finished(
            ok, error, cancelled, report_dir, progress, token))
        self._report_worker = worker
        self.btn_report.setEnabled(False)
        worker.start()

    def _on_report_finished(self, ok: bool, error: str, cancelled: bool, report_dir: Path,
                            progress, token):
        self._report_worker = None
        token.release()
        progress.close()
        self.btn_report.setEnabled(bool(self.state.summaries))
        if not ok:
            QMessageBox.critical(self, "Report failed", error)
            return
        if cancelled:
            # The folder was made by this report: leave nothing half written.
            import shutil
            shutil.rmtree(report_dir, ignore_errors=True)
            self.lbl_export.setText("Report cancelled; nothing was written.")
            _status(self.bus, "Report cancelled.")
            return
        self.lbl_export.setText(f"Report written to {report_dir}")
        _status(self.bus, f"Report written to {report_dir}")

    def _settings_text(self, samples: list[str]) -> str:
        st = self.state
        lines = [
            f"Exported: {datetime.now().isoformat(timespec='seconds')}",
            f"Honeychrome: {getattr(honeychrome, '__version__', '')}",
            f"Model: {st.model_path or '(not saved)'}",
            f"Model SHA-256 (gates, boundaries, transforms): {st.model_sha}",
            f"Model SHA-256 when gated: {st.results_model_sha}",
            "",
            "Gates:",
        ]
        for g in ag_core.ordered_gate_defs(st.gate_defs()):
            lines.append(f"  {g['gate_name']}: {MODE_LABELS.get(st.modes.get(g['gate_name'], 'fixed'))}")
        lines += [
            f"Drift limit: {st.drift_limit} (transformed units); when exceeded: "
            f"{DRIFT_ACTIONS.get(st.drift_action)}",
            f"Time QC available: {ag_builder.time_qc_available()}",
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
                f"  Pairing: {st.pairing or 'none'}; adjusted for: "
                + (', '.join(f"{c} (numeric)" if c in st.numeric_covariates else c
                             for c in st.adjust_covariates) or 'none'),
                f"  FDR <= {st.pval_threshold} ({st.fdr_scope}); TREAT: {st.use_treat}",
                f"  Counts: relative to the "
                f"{st.stats.get('counts_reference', 'parent')} population; model: "
                f"{COUNTS_METHOD_LABELS.get(st.stats.get('counts_method', st.counts_method), st.counts_method)}",
                f"  Min. parent events for frequencies: {st.min_parent_events}",
                f"  Frequency/count threshold: {st.fc_threshold}; marker threshold: {st.marker_threshold}",
                f"  Frequencies relative to stats parent where defined: {st.use_stats_parent}",
                f"  Gating markers left out of marker tests: {st.exclude_gating_markers}; "
                f"min. events: {st.min_marker_events}",
                f"  Flagged samples left out: {st.exclude_flagged}",
            ]
            lines += [f"  Note: {n}" for n in st.stats.get('notes') or []]
        if st.trend:
            ts_ = st.trend.get('settings') or {}
            unit = st.trend.get('unit') or ''
            lines += [
                "",
                "Trend with a numeric variable:",
                f"  Variable: {st.trend.get('variable', '')}" + (f" ({unit})" if unit else ''),
                f"  Samples: {ts_.get('n_samples', len(st.trend.get('samples') or []))}; "
                f"groups: {', '.join(ts_.get('groups') or [])}",
                f"  Shape: {ts_.get('shape', '')}, {ts_.get('df', '')} df; scale: "
                f"{ts_.get('scale', '')}; knots: "
                + (', '.join(f"{k:.3g}" for k in ts_.get('knots') or []) or 'none'),
                f"  Group × trend: {bool(ts_.get('interaction'))}; paired: {bool(ts_.get('paired'))}",
                f"  Call: {ts_.get('call', '')}; FDR <= {ts_.get('pval_threshold', '')}; "
                f"min. partial R²: {ts_.get('min_r2', '')}",
                f"  Counts: log odds within the {st.trend.get('counts_reference', 'anchor')} "
                "population (moderated)",
            ]
            lines += [f"  Note: {n}" for n in st.trend.get('notes') or []]
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------

    def refresh(self):
        self._populate_groups()
        self._maybe_auto_suggest_groupings()
        self._load_stats_controls()
        self._load_explore_controls()
        self._populate_comparisons()
        self._show_results_tables()
        if self._trend_mode():
            self._show_run_summary()
            if not self.state.trend:
                self.figures_tabs.clear()
                self._figures_for = None
            elif self._figures_for is not self.state.trend:
                self._draw_figures()
        elif not self.state.stats:
            self.figures_tabs.clear()
            self.lbl_stats.setText("")
            self._figures_for = None
        elif self._figures_for is not self.state.stats:
            self.lbl_stats.setText(self._stats_summary(self.state.stats))
            self._draw_figures()
        self._populate_population_list()
        self._populate_report_items()


# ---------------------------------------------------------------------------
# PluginWidget
# ---------------------------------------------------------------------------

class _StateLoader(QThread):
    """Read an experiment's gated results, display records and statistics
    (and a model file to carry over, see PluginWidget.load_state) off the
    main thread; finished(dict) carries what was read."""

    finished = Signal(object)

    def __init__(self, job: dict, parent=None):
        super().__init__(parent)
        self._job = job

    def run(self):
        job = self._job
        out = {'key': job['key'], 'errors': []}
        try:
            self._read_model(job, out)
        except Exception as exc:
            log.exception("could not read the model %s", job.get('model_path'))
            out['errors'].append(f"Could not read the model: {exc}")
        try:
            self._read_results(job, out)
        except Exception as exc:
            log.exception("could not read the saved results")
            out['errors'].append(f"Could not read the saved results: {exc}")
        try:
            self._read_trend(job, out)
        except Exception as exc:
            log.exception("could not read the saved trend test")
            out['errors'].append(f"Could not read the saved trend test: {exc}")
        try:
            self._read_enrichment(job, out)
        except Exception as exc:
            log.exception("could not read the saved enrichment")
            out['errors'].append(f"Could not read the saved enrichment: {exc}")
        try:
            self._read_sample_analysis(job, out)
        except Exception as exc:
            log.exception("could not read the saved sample PCA and predictive analysis")
            out['errors'].append(f"Could not read the saved sample PCA and predictive "
                                 f"analysis: {exc}")
        self.finished.emit(out)

    @staticmethod
    def _read_sample_analysis(job: dict, out: dict):
        """The last sample PCA and predictive analysis, while they belong to
        the statistics run that was read."""
        path = job.get('sample_analysis_path')
        run_id = (out.get('stats') or {}).get('run_id')
        if path is None or not run_id or not Path(path).exists():
            return
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        if payload.get('stats_run_id') != run_id:
            return
        out['pca'], out['predictive'] = ag_results.sample_analysis_from_jsonable(payload)

    @staticmethod
    def _read_enrichment(job: dict, out: dict):
        """The last enrichment run, while it matches the gated results."""
        path = job.get('enrichment_path')
        results = out.get('results')
        if path is None or not results or not Path(path).exists():
            return
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        if payload.get('run_fingerprint') != str(results.get('run_fingerprint') or ''):
            return
        if payload.get('result'):
            out['fsea'] = ag_fsea.result_from_jsonable(payload['result'])
        if payload.get('trend_result'):
            out['fsea_trend'] = ag_fsea.result_from_jsonable(payload['trend_result'])

    @staticmethod
    def _read_model(job: dict, out: dict):
        """The model file Automated Gating used before the model pages were
        added, when the model pages have no gates yet."""
        path = job.get('model_path')
        if not path:
            return
        if not Path(path).exists():
            out['model_missing'] = path
            return
        out['model'], out['model_warnings'] = ag_core.read_model(path)

    @staticmethod
    def _attach_markers(results_path: Path, payload: dict, fingerprint: str, out: dict):
        """Give each summary of *payload* its marker summary: from the
        marker file it names, or, in results saved before the marker file,
        converted from the nested lists in the JSON (out['upgrade_results']
        then asks for the results to be saved again in the current form)."""
        summaries = payload.get('summaries') or {}
        if int(payload.get('format_version') or 1) < RESULTS_FORMAT:
            for summ in summaries.values():
                summ['markers'] = ag_results.as_marker_summary(summ.get('markers'))
            out['upgrade_results'] = bool(summaries)
            return
        loaded: dict = {}
        name = Path(str(payload.get('marker_file') or MARKERS_FILE.name)).name
        try:
            saved_fingerprint, loaded = ag_results.load_marker_summaries(results_path.with_name(name))
            if saved_fingerprint != fingerprint:
                raise ValueError("it belongs to another gating run")
            missing = [k for k in summaries if k not in loaded]
            if missing:
                raise ValueError(f"{len(missing)} sample(s) are missing from it")
        except Exception as exc:
            log.warning("could not read the saved marker summaries: %s", exc)
            if summaries:
                out['errors'].append(
                    f"Could not read the saved marker medians ({exc}); run gating again to "
                    "restore them.")
        empty = ag_results.MarkerSummary.empty()
        for key, summ in summaries.items():
            summ['markers'] = loaded.get(key, empty)

    @staticmethod
    def _read_results(job: dict, out: dict):
        path = job.get('results_path')
        if path is None or not Path(path).exists():
            return
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        out['results'] = payload
        fingerprint = str(payload.get('run_fingerprint') or '')
        _StateLoader._attach_markers(Path(path), payload, fingerprint, out)
        display = {}
        folder = job.get('display_dir')
        if folder is not None:
            for key in (payload.get('summaries') or {}):
                loaded = ag_results.load_display(Path(folder) / ag_results.display_file_name(key))
                if loaded is None:
                    continue
                disp, meta = loaded
                if meta.get('sample') == key and meta.get('fingerprint') == fingerprint:
                    display[key] = disp
        out['display'] = display
        stats_path = job.get('stats_path')
        if stats_path is None or not Path(stats_path).exists():
            return
        sp = json.loads(Path(stats_path).read_text(encoding='utf-8'))
        if sp.get('run_fingerprint') != fingerprint:
            return
        stats = ag_results.stats_from_jsonable(sp['stats'])
        summaries = payload.get('summaries') or {}
        if not stats.get('samples') or any(k not in summaries for k in stats['samples']):
            return
        t = job['thresholds']
        out['stats'] = ag_results.reapply_thresholds(stats, t['pval'], t['fc'], t['marker'],
                                                     t['fdr_scope'], t['treat'])

    @staticmethod
    def _read_trend(job: dict, out: dict):
        """The last trend run, while it matches the gated results."""
        path = job.get('trend_path')
        results = out.get('results')
        if path is None or not results or not Path(path).exists():
            return
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        if payload.get('run_fingerprint') != str(results.get('run_fingerprint') or ''):
            return
        trend = ag_results.trend_from_jsonable(payload['trend'])
        summaries = results.get('summaries') or {}
        if not trend.get('samples') or any(k not in summaries for k in trend['samples']):
            return
        t = job['thresholds']
        out['trend'] = ag_results.reapply_trend_thresholds(trend, t['pval'], t['trend_min_r2'],
                                                           t['trend_call'],
                                                           t.get('trend_patterns', 0))


class _FileWriter(QThread):
    """Run one file-writing job (a callable) off the main thread."""

    done = Signal(bool, str, object)

    def __init__(self, job, label: str, parent=None):
        super().__init__(parent)
        self._job = job
        self.label = label

    def run(self):
        try:
            result = self._job()
        except Exception as exc:
            log.exception("%s failed", self.label)
            self.done.emit(False, str(exc), None)
            return
        self.done.emit(True, '', result)


def _write_json_atomic(path: Path, payload, jsonable: bool = False):
    """Write *payload* as JSON beside *path*, then move it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    data = ag_core.to_jsonable(payload) if jsonable else payload
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, separators=(',', ':'))
    _os.replace(tmp, path)


class PluginWidget(QWidget):
    """Top-level widget for the Automated Gating plugin.

    Pages, listed down the left-hand side: Transforms, Import/Setup,
    Hierarchy and Train build the model (ag_builder.BuilderPages, which owns
    the model state, the transforms and their persistence); Run, Results and
    Enrichment apply it (GatingRunState); Export, last, saves the model (a
    BuilderPages page). Run applies the model as it stands on the model
    pages, in their transforms: state.applied_model is rebuilt from them
    (_sync_applied_model) whenever Run, Results or Enrichment is shown.

    Persistence: everything is kept in the experiment folder, tied to the
    .kit it belongs to (see experiment_state). The model pages keep their
    own settings and session file (see BuilderPages). The settings of the
    last three pages (gate modes, drift settings, sample selection, groups,
    covariates and statistics options) are kept as one JSON string in
    cache/automated_gating/state.json. Gated results (counts and per-gate status) are kept in
    RESULTS_FILE in the experiment folder, with the samples a paused run has
    still to gate, their marker medians and quartiles in MARKERS_FILE, and
    each sample's display record in DISPLAY_DIR; the last
    statistics run is kept in STATS_FILE and the last enrichment run in
    ENRICHMENT_FILE, both restored while they still match the gated results;
    the last sample PCA and predictive analysis are kept in
    SAMPLE_ANALYSIS_FILE and restored with the statistics run they came from.
    These are read in the background when the experiment is opened and
    written in the background after each run, when the gating results and
    statistics are also written as CSV files to EXPORT_DIR.
    """

    def __init__(self, bus=None, controller=None, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.controller = controller
        _ensure_qt_imports()
        _suppress_third_party_warnings()

        self.state = GatingRunState()
        self._result_tf = (None, None)
        # The loaded experiment's settings store, opened by load_state.
        self._store = None
        self._plugin_was_active = False
        self._loaded_experiment_key = None
        self._loaded_experiment_dir = None
        self._loader = None
        self._loading = False
        self._writers: list = []
        self._legacy = {'path': '', 'alignment': {}}
        self._page_shown = None
        self._model_stale = True
        # The model pages' signature the applied model was last built from,
        # the run settings last written to QSettings, and, per Run/Results/
        # Enrichment page, the view key it last showed (see _view_key).
        self._synced_signature = None
        self._saved_run_state = None
        self._shown_keys: dict = {}

        self.label_disabled = QLabel(f"{plugin_name}: Loading...Set up the spectral model if you have not already done so.")
        self.label_disabled.setWordWrap(True)
        self.label_disabled.setStyleSheet(
            "QLabel { color: #856404; background: #fff3cd; "
            "border: 1px solid #ffc107; border-radius: 4px; padding: 8px; }")
        self.label_loading = QLabel("Loading the saved results …")
        self.label_loading.setStyleSheet("color: #555; font-style: italic;")
        self.label_loading.setVisible(False)

        self.content_widget = QWidget()
        content_layout = QVBoxLayout(self.content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self.label_loading)
        overall = QVBoxLayout(self)
        overall.setContentsMargins(6, 6, 6, 6)
        overall.addWidget(self.label_disabled)
        overall.addWidget(self.content_widget, stretch=1)
        overall.addStretch()

        # Each page scrolls on its own, so a wide page never sets a minimum
        # width for the main window's tabs (and so for the sample panel).
        self.inner_tabs = SideTabWidget(scroll_pages=True)
        self.builder = ag_builder.BuilderPages(
            bus, controller, self.inner_tabs, application=self._application,
            extra_samples=lambda: list(self.state.run_samples) or self.state.gated_samples(),
            train_blocked=lambda: ("Gating is running on the Run page; pause or cancel it "
                                   "first." if self.run_tab.is_running() else ''),
            parent=self)
        self.tf = self.builder.tf
        self.transforms_tab = self.builder.transforms_tab
        self.run_tab = RunTab(self.state, bus, controller,
                              gating_transforms=lambda: self.tf.objects,
                              result_transforms=self._result_transforms,
                              display_dir=self._display_dir,
                              readiness=self._run_readiness)
        self.results_tab = ResultsTab(self.state, bus, controller,
                                      result_transforms=self._result_transforms)
        self.enrichment_tab = ag_fsea_ui.EnrichmentTab(
            self.state, bus, controller, samples_fn=self.results_tab._stats_samples,
            design_fn=lambda samples: statistics_design(self.state, samples),
            antigens_fn=lambda: _antigen_map(self.controller),
            experiment_dir_fn=lambda: str(self._loaded_experiment_dir or ''),
            trend_fn=self.results_tab.trend_enrichment_inputs)
        self.inner_tabs.addTab(self.run_tab, "Run", 'player-play')
        self.inner_tabs.addTab(self.results_tab, "Results", 'table')
        self.inner_tabs.addTab(self.enrichment_tab, "Enrichment", 'sparkles')
        self.inner_tabs.addTab(self.builder.export_tab, "Export", 'file-export')
        self._page_shown = self.inner_tabs.currentWidget()
        self.inner_tabs.currentChanged.connect(self._on_inner_tab_changed)
        content_layout.addWidget(self.inner_tabs, stretch=1)

        self.builder.modelImported.connect(self._on_model_imported)
        self.builder.gateRenamed.connect(self._on_gate_renamed)
        self.builder.modelTrained.connect(self._on_model_trained)
        self.run_tab.gating_finished.connect(self._on_gating_finished)
        self.results_tab.stats_finished.connect(self._on_stats_finished)
        self.results_tab.trend_finished.connect(self._on_trend_finished)
        self.results_tab.report_populations_changed.connect(self._save_run_state)
        self.results_tab.sample_analysis_changed.connect(self._write_sample_analysis)
        self.enrichment_tab.enrichment_finished.connect(self._on_enrichment_finished)

        self.content_widget.setVisible(False)
        # modeChangeRequested fires on every main-window tab switch (guarded
        # on plugin_name); loadSampleRequested when a sample is picked in the
        # sample browser (deferred one tick so the sample has loaded).
        self.bus.modeChangeRequested.connect(self._on_mode_change)
        self.bus.loadSampleRequested.connect(self._on_sample_selected)
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

    def _on_sample_selected(self, _sample_path: str):
        if getattr(self.controller, 'current_mode', None) != plugin_name:
            return
        QTimer.singleShot(0, self.builder.on_sample_loaded)
        if self.inner_tabs.currentWidget() is self.run_tab:
            QTimer.singleShot(0, self.run_tab.selected_sample_changed)

    def _show_if_ready(self):
        if self.controller.experiment.process.get('unmixing_matrix') is not None:
            self.label_disabled.setVisible(False)
            self.content_widget.setVisible(True)
            self._model_stale = True
            if not self._ensure_state_loaded() and not self._loading:
                self._refresh_active_tab()
        else:
            self.label_disabled.setVisible(True)
            self.content_widget.setVisible(False)

    def _ensure_state_loaded(self) -> bool:
        """Load the persisted state when the experiment has changed (the
        model pages at once, the results in the background). Returns True
        when loading started."""
        current_key = self._experiment_key()
        if current_key == self._loaded_experiment_key:
            return False
        # A run's results belong to the experiment it started in.
        self.run_tab.abandon_run()
        if self._loaded_experiment_key is not None:
            self.save_state()
        # The inner tabs share self.state by reference, so reset it in place.
        fresh = GatingRunState()
        for f in dataclass_fields(GatingRunState):
            setattr(self.state, f.name, getattr(fresh, f.name))
        self._loaded_experiment_key = current_key
        self._loaded_experiment_dir = Path(self.controller.experiment_dir)
        self._result_tf = (None, None)
        self._synced_signature = None
        self._shown_keys = {}
        self.builder.switch_experiment(current_key, self._loaded_experiment_dir)
        self.load_state()
        return True

    def _on_inner_tab_changed(self, index: int):
        """The page being left shows the state as it stands, so it is noted
        as current and what changed is saved; then the page shown is
        refreshed if the state or its context changed since it was last
        shown."""
        left, self._page_shown = self._page_shown, self.inner_tabs.widget(index)
        signature = self.builder.signature()
        if self.builder.owns(left):
            self.builder.page_left(left, signature)
            self._model_stale = True
        elif left in self._shown_keys:
            # A page keeps its own settings up to date while it is shown; the
            # rest of its key (results, model, context) stays as it was when
            # it was last refreshed, since those can change in the background.
            key = list(self._shown_keys[left])
            key[1] = self._settings_digest()
            self._shown_keys[left] = tuple(key)
        self._save_run_state()
        if not self._loading:
            self._refresh_tab_at(index, signature)

    def _refresh_active_tab(self):
        self._refresh_tab_at(self.inner_tabs.currentIndex())

    def _run_pages(self) -> tuple:
        return (self.run_tab, self.results_tab, self.enrichment_tab)

    def _settings_digest(self) -> str:
        st = self.state
        return ag_core.content_digest(
            {name: getattr(st, name) for name in _PERSISTED_FIELDS + _VIEW_FIELDS})

    def _view_key(self, signature: str) -> tuple:
        """What the Run, Results and Enrichment pages show: the model pages'
        state, the run settings (second), the results and the experiment
        around them."""
        st = self.state
        return (signature, self._settings_digest(), st.results_revision,
                len(st.summaries), len(st.display), self.run_tab.is_running(),
                ag_builder.view_context(self.controller))

    def _refresh_tab_at(self, index: int, signature: str | None = None):
        widget = self.inner_tabs.widget(index)
        if self.builder.refresh_page(widget, signature):
            return
        if widget in self._run_pages():
            signature = signature or self.builder.signature()
            if self._model_stale and not self.run_tab.is_running():
                self._sync_applied_model(signature)
            if self._shown_keys.get(widget) == self._view_key(signature):
                if widget is self.run_tab:
                    # The unmixing is not part of the view key: check it.
                    self.run_tab._update_warnings()
                return
            widget.refresh()
            self._shown_keys[widget] = self._view_key(signature)

    def _on_gating_finished(self, outcome: str = 'done'):
        self._result_tf = (None, None)
        self._write_results()
        self._write_stats()
        self._write_trend()
        self._write_enrichment()
        self._write_sample_analysis()
        self._save_run_state()
        if self.state.summaries and outcome in ('done', 'paused', 'failed'):
            self._export_results_csv()

    def _on_stats_finished(self):
        """Keep the statistics, and the settings they were run with, as soon
        as a run completes, and write them as CSV files."""
        self._write_stats()
        self._write_sample_analysis()
        self._save_run_state()
        self._export_statistics_csv()

    def _on_trend_finished(self):
        """Keep the trend run, and the settings it was run with, as soon as
        it completes, and write it as CSV files."""
        self._write_trend()
        self._save_run_state()
        self._export_trend_csv()

    def _on_enrichment_finished(self):
        """Keep the enrichment run, write it as CSV files and list it in
        the report."""
        self._write_enrichment()
        self._save_run_state()
        self._export_enrichment_csv()
        self.results_tab._populate_report_items()

    def progress_message(self, msg: str):
        _status(self.bus, msg)

    # ------------------------------------------------------------------
    # The model the Run page applies
    # ------------------------------------------------------------------

    def _application(self) -> dict:
        """The model's application block: each gate's mode and the drift
        settings (set on the Run page)."""
        st = self.state
        return {'mode_per_gate': {g['gate_name']: st.modes.get(g['gate_name'], ag_core.MODE_FIXED)
                                  for g in self.builder.state.gate_definitions},
                'drift_limit': float(st.drift_limit), 'drift_action': st.drift_action}

    def _on_model_trained(self):
        """Training finished (it runs in the background) or a boundary was
        adjusted: the Run page's model is out of date."""
        self._model_stale = True
        if self.inner_tabs.currentWidget() in (self.run_tab, self.results_tab,
                                               self.enrichment_tab):
            self._refresh_active_tab()

    def _on_gate_renamed(self, old: str, new: str):
        if old in self.state.modes:
            self.state.modes[new] = self.state.modes.pop(old)

    def _on_model_imported(self, info: dict):
        """A saved model replaced the gates: take its gate modes and drift
        settings."""
        st = self.state
        app = info.get('application') or {}
        st.modes = {str(k): v for k, v in (app.get('mode_per_gate') or {}).items()
                    if v in MODE_LABELS}
        if isinstance(app.get('drift_limit'), (int, float)):
            st.drift_limit = float(app['drift_limit'])
        if app.get('drift_action') in DRIFT_ACTIONS:
            st.drift_action = app['drift_action']
        self._save_run_state()

    def _sync_applied_model(self, signature: str | None = None):
        """Point the Run, Results and Enrichment pages at the model as it
        stands on the model pages (rebuilt only when it has changed;
        *signature*: the model pages' current signature, when known)."""
        self._model_stale = False
        st = self.state
        st.modes = self._application()['mode_per_gate']
        signature = signature or self.builder.signature()
        if signature != self._synced_signature or not st.applied_model:
            # A new dict in JSON types, owned here: hashed without a converted
            # copy and completed in place.
            model = self.builder.model()
            sha = model_sha(model, converted=True)
            if sha != st.model_sha or not st.applied_model:
                try:
                    model = ag_core.normalize_model(model, copy=False)[0]
                except Exception:
                    log.exception("the model pages' gates do not form a valid hierarchy")
                st.applied_model = model
                st.model_sha = sha
            self._synced_signature = signature
        st.model_path = str(self.builder.state.model_metadata.get('last_model_path') or '')
        st.transform_params = self.tf.params()

    def _run_readiness(self) -> str:
        """'' when the model can be applied, else why not."""
        bs = self.builder.state
        if not bs.gate_definitions:
            return ("No gating model yet: import gates or load a saved model on the "
                    "Import/Setup page, then train it.")
        try:
            ag_core.ordered_gate_defs(bs.gate_definitions)
        except Exception as exc:
            return f"The gate hierarchy cannot be applied: {exc}"
        if self.builder.train_tab.is_running():
            return "Training is running on the Train page; gating can start when it finishes."
        if bs.training_progress:
            return ("Training is paused on the Train page; resume or overwrite it there "
                    "before gating.")
        untrained = [g['gate_name'] for g in ag_core.ordered_gate_defs(bs.gate_definitions)
                     if not bs.is_trained(g['gate_name'])]
        if untrained:
            return (f"{len(untrained)} gate(s) have no trained boundary yet: "
                    + ", ".join(untrained[:8]) + (" …" if len(untrained) > 8 else "")
                    + ". Train them on the Train page.")
        if self._model_stale and not self.run_tab.is_running():
            self._sync_applied_model()
        return ''

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def _result_transforms(self) -> dict:
        """Transforms the gated results are in (those of the run), as
        Transform objects."""
        st = self.state
        if not st.run_transforms:
            return self.tf.objects
        key = json.dumps(st.run_transforms, sort_keys=True, default=str)
        if self._result_tf[0] != key:
            built = tp.TransformSet()
            built.load(st.run_transforms, self.controller.unmixed_transformations or {})
            self._result_tf = (key, built.objects)
        return self._result_tf[1]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _experiment_key(self) -> tuple:
        """The experiment shown: its folder and the id of its .kit, so a new
        experiment created at the same path counts as a different one."""
        try:
            exp_dir = str(self.controller.experiment_dir)
        except Exception:
            exp_dir = 'unknown'
        return (experiment_state.legacy_group_key(exp_dir),
                str(getattr(self.controller.experiment, 'experiment_id', '') or ''))

    def _controller_shows_loaded_experiment(self) -> bool:
        if self._loaded_experiment_dir is None:
            return False
        try:
            return Path(self.controller.experiment_dir) == self._loaded_experiment_dir
        except Exception:
            return False

    def _display_dir(self):
        if self._loaded_experiment_dir is None:
            return None
        return self._loaded_experiment_dir / DISPLAY_DIR

    def save_state(self):
        """Persist every page's settings for the experiment the state was
        loaded from."""
        self.builder.save_state()
        self._save_run_state()

    def _save_run_state(self):
        """Persist the Run, Results and Enrichment settings."""
        if self._store is None or self._loading:
            return
        payload = {name: getattr(self.state, name) for name in _PERSISTED_FIELDS}
        text = json.dumps(ag_core.to_jsonable(payload))
        if self._saved_run_state == (self._loaded_experiment_key, text):
            return
        self._store.setValue('state_json', text)
        self._store.sync()
        self._saved_run_state = (self._loaded_experiment_key, text)

    def load_state(self):
        """Restore the settings, then read the gated results, display records
        and statistics in the background. Opening the settings store first
        sets aside results another experiment left in this folder.

        Automated Gating used to load a model file itself; when the model
        pages have no gates yet and such a file is remembered, it is read
        too and imported with the channel alignment that was accepted for it.
        """
        self._store = experiment_state.open_store(
            self.controller, 'automated_gating', legacy_app_key='plugin_automated_gating',
            experiment_dir=self._loaded_experiment_dir)
        raw = self._store.value('state_json', '')
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
        if self.state.counts_method not in ag_results.COUNTS_METHODS:
            self.state.counts_method = defaults.counts_method
        for name, allowed in (('stats_test', STATS_TESTS), ('trend_shape', TREND_SHAPE_LABELS),
                              ('trend_scale', TREND_SCALE_LABELS), ('trend_call', TREND_CALL_LABELS)):
            if getattr(self.state, name) not in allowed:
                setattr(self.state, name, getattr(defaults, name))
        st = self.state
        exp = self._loaded_experiment_dir
        legacy_model = ''
        if not self.builder.state.gate_definitions and payload.get('alignment_confirmed'):
            legacy_model = str(payload.get('model_path') or '')
        self._legacy = {'path': legacy_model, 'alignment': dict(payload.get('alignment') or {})}
        job = {
            'key': self._loaded_experiment_key,
            'model_path': legacy_model,
            'results_path': exp / RESULTS_FILE,
            'stats_path': exp / STATS_FILE,
            'trend_path': exp / TREND_FILE,
            'enrichment_path': exp / ENRICHMENT_FILE,
            'sample_analysis_path': exp / SAMPLE_ANALYSIS_FILE,
            'display_dir': exp / DISPLAY_DIR,
            'thresholds': {'pval': st.pval_threshold, 'fc': st.fc_threshold,
                           'marker': st.marker_threshold, 'fdr_scope': st.fdr_scope,
                           'treat': st.use_treat, 'trend_min_r2': st.trend_min_r2,
                           'trend_call': st.trend_call, 'trend_patterns': st.trend_patterns},
        }
        self._loading = True
        self.label_loading.setVisible(True)
        self.inner_tabs.setEnabled(False)
        loader = _StateLoader(job, parent=self)
        loader.finished.connect(self._on_state_loaded)
        self._loader = loader
        loader.start()

    def _on_state_loaded(self, out: dict):
        self._loader = None
        self._loading = False
        self.label_loading.setVisible(False)
        self.inner_tabs.setEnabled(True)
        if out.get('key') != self._loaded_experiment_key:
            return
        st = self.state
        if out.get('model') is not None:
            self._import_legacy_model(out['model'], out.get('model_warnings') or [])
        elif out.get('model_missing'):
            self.progress_message(f"The model file {out['model_missing']} is no longer available.")
        payload = out.get('results')
        if payload:
            st.summaries = dict(payload.get('summaries') or {})
            st.load_info = dict(payload.get('load_info') or {})
            st.marker_channels = list(payload.get('marker_channels') or [])
            st.results_model_sha = str(payload.get('model_sha256') or '')
            st.run_transforms = dict(payload.get('run_transforms') or {})
            if payload.get('run_samples'):
                st.run_samples = list(payload['run_samples'])
            st.pending_samples = [k for k in (payload.get('pending_samples') or [])
                                  if k not in st.summaries]
            st.run_fingerprint = str(payload.get('run_fingerprint') or '')
            st.run_unmixing = str(payload.get('run_unmixing') or '')
            st.display = dict(out.get('display') or {})
            if int(payload.get('format_version') or 1) < 2:
                # Hash of the model file, from before the model pages: not comparable.
                st.results_model_sha = ''
            if out.get('upgrade_results'):
                # Saved before the marker file: save them again, so the next
                # opening reads the arrays rather than the nested lists.
                self._write_results()
        if out.get('stats'):
            st.stats = out['stats']
        if out.get('trend'):
            st.trend = out['trend']
        if out.get('fsea'):
            st.fsea = out['fsea']
        if out.get('fsea_trend'):
            st.fsea_trend = out['fsea_trend']
        if out.get('pca'):
            st.pca = out['pca']
        if out.get('predictive'):
            st.predictive = out['predictive']
        for msg in out.get('errors') or []:
            self.progress_message(msg)
        self._sync_applied_model()
        if st.summaries and st.results_model_sha and st.results_model_sha != st.model_sha:
            self.progress_message("The saved gating results were made with a different "
                                  "version of the model; run gating again to update them.")
        self._refresh_active_tab()

    def _import_legacy_model(self, model: dict, warnings_list: list):
        """Import the model file Automated Gating used before the model
        pages were added, with the channel alignment accepted for it."""
        path, alignment = self._legacy['path'], self._legacy['alignment']
        gated = {mc: alignment.get(mc)
                 for mc in ag_results.gated_channels(model.get('gate_definitions') or [])}
        if ag_results.alignment_problems(gated):
            self.progress_message("The previously used model needs its channels matched: "
                                  "load it on the Import/Setup page.")
            return
        modes, drift = dict(self.state.modes), (self.state.drift_limit, self.state.drift_action)
        names = {g['gate_name'] for g in model.get('gate_definitions') or []}
        if self.builder.hierarchy_tab.import_saved_model(model, path, gated, warnings_list):
            # Gate modes and drift settings chosen for this model win over the file's.
            if modes and set(modes) == names:
                self.state.modes = modes
            self.state.drift_limit, self.state.drift_action = drift
            self.save_state()

    def _results_path(self) -> Path | None:
        if self._loaded_experiment_dir is None:
            return None
        return self._loaded_experiment_dir / RESULTS_FILE

    def _stats_path(self) -> Path | None:
        if self._loaded_experiment_dir is None:
            return None
        return self._loaded_experiment_dir / STATS_FILE

    def _start_writer(self, job, label: str):
        writer = _FileWriter(job, label, parent=self)
        writer.done.connect(lambda ok, err, _res, w=writer: self._on_writer_done(w, ok, err))
        self._writers.append(writer)
        writer.start()

    def _on_writer_done(self, writer, ok: bool, error: str):
        if writer in self._writers:
            self._writers.remove(writer)
        if not ok:
            self.progress_message(f"{writer.label} failed: {error}")

    def _write_results(self):
        """Write the gated results (summaries, not the display events, which
        the worker saved) to RESULTS_FILE, in the background."""
        if not self._controller_shows_loaded_experiment():
            return
        path = self._results_path()
        st = self.state
        summaries = dict(st.summaries)
        payload = {
            'format_version': RESULTS_FORMAT,
            'model_path': st.model_path,
            'model_sha256': st.results_model_sha,
            'marker_channels': list(st.marker_channels),
            'marker_file': MARKERS_FILE.name,
            'run_samples': list(st.run_samples),
            'summaries': {k: {f: v for f, v in (s or {}).items() if f != 'markers'}
                          for k, s in summaries.items()},
            'load_info': dict(st.load_info),
            'pending_samples': list(st.pending_samples),
            'run_fingerprint': st.run_fingerprint,
            'run_unmixing': st.run_unmixing,
            'run_transforms': deepcopy(st.run_transforms),
        }

        def _job():
            # The marker file is written first, so the results never name
            # samples it does not have.
            ag_results.save_marker_summaries(path.with_name(MARKERS_FILE.name), summaries,
                                             payload['run_fingerprint'])
            _write_json_atomic(path, payload)

        self._start_writer(_job, "Saving the gating results")

    def _write_stats(self):
        """Write the last statistics run to STATS_FILE (in the background),
        or remove the file when there is none (a new gating run clears the
        statistics)."""
        if not self._controller_shows_loaded_experiment():
            return
        path = self._stats_path()
        st = self.state
        if not st.stats:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                log.exception("could not remove %s", path)
            return
        stats, fingerprint = dict(st.stats), st.run_fingerprint

        def _job():
            _write_json_atomic(path, {'format_version': 1, 'run_fingerprint': fingerprint,
                                      'stats': ag_results.stats_to_jsonable(stats)})

        self._start_writer(_job, "Saving the statistics")

    def _write_trend(self):
        """Write the last trend run to TREND_FILE (in the background), or
        remove the file when there is none (a new gating run clears it)."""
        if not self._controller_shows_loaded_experiment():
            return
        path = self._loaded_experiment_dir / TREND_FILE
        st = self.state
        if not st.trend:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                log.exception("could not remove %s", path)
            return
        trend, fingerprint = dict(st.trend), st.run_fingerprint

        def _job():
            _write_json_atomic(path, {'format_version': 1, 'run_fingerprint': fingerprint,
                                      'trend': ag_results.trend_to_jsonable(trend)})

        self._start_writer(_job, "Saving the trend test")

    def _export_trend_csv(self):
        """Write the trend tables as CSV files to EXPORT_DIR (background)."""
        if not self._controller_shows_loaded_experiment() or not self.state.trend:
            return
        st = self.state
        folder = self._loaded_experiment_dir / EXPORT_DIR
        stem, stamp = _export_stem(st.model_path), datetime.now().strftime('%Y%m%d_%H%M%S')
        args = (dict(st.trend), _population_labels(st.gate_defs()), _antigen_map(self.controller))
        self._start_writer(lambda: write_trend_csv(folder, stem, stamp, *args),
                           "Writing the trend CSV")
        _status(self.bus, f"Trend results written to {folder}")

    def _write_enrichment(self):
        """Write the last enrichment runs (group comparisons and trend) to
        ENRICHMENT_FILE (in the background), or remove the file when there
        is neither."""
        if not self._controller_shows_loaded_experiment():
            return
        path = self._loaded_experiment_dir / ENRICHMENT_FILE
        st = self.state
        if not (st.fsea or st.fsea_trend):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                log.exception("could not remove %s", path)
            return
        result, trend_result = dict(st.fsea), dict(st.fsea_trend)
        fingerprint = st.run_fingerprint

        def _job():
            _write_json_atomic(path, {
                'format_version': 1, 'run_fingerprint': fingerprint,
                'result': ag_fsea.result_to_jsonable(result) if result else None,
                'trend_result': (ag_fsea.result_to_jsonable(trend_result) if trend_result
                                 else None)})

        self._start_writer(_job, "Saving the enrichment")

    def _write_sample_analysis(self):
        """Write the last sample PCA and predictive analysis to
        SAMPLE_ANALYSIS_FILE (in the background) with the run_id of the
        statistics run they came from, or remove the file when there is
        neither."""
        if not self._controller_shows_loaded_experiment():
            return
        path = self._loaded_experiment_dir / SAMPLE_ANALYSIS_FILE
        st = self.state
        run_id = (st.stats or {}).get('run_id')
        if not (st.pca or st.predictive) or not run_id:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                log.exception("could not remove %s", path)
            return
        pca, predictive = dict(st.pca), dict(st.predictive)

        def _job():
            payload = ag_results.sample_analysis_to_jsonable(pca, predictive)
            payload.update({'format_version': 1, 'stats_run_id': run_id})
            _write_json_atomic(path, payload)

        self._start_writer(_job, "Saving the sample PCA and predictive analysis")

    def _export_enrichment_csv(self):
        """Write the enrichment tables as CSV files to EXPORT_DIR (background)."""
        st = self.state
        if not self._controller_shows_loaded_experiment() or not (st.fsea or st.fsea_trend):
            return
        folder = self._loaded_experiment_dir / EXPORT_DIR
        stem, stamp = _export_stem(st.model_path), datetime.now().strftime('%Y%m%d_%H%M%S')
        runs = [(stem, dict(st.fsea)), (f"{stem}_trend", dict(st.fsea_trend))]

        def _job():
            for run_stem, result in runs:
                if result:
                    ag_fsea_ui.write_enrichment_csv(folder, run_stem, stamp, result)
        self._start_writer(_job, "Writing the enrichment CSV")
        _status(self.bus, f"Enrichment written to {folder}")

    def _export_results_csv(self):
        """Write the gating results as CSV files to EXPORT_DIR (background)."""
        if not self._controller_shows_loaded_experiment():
            return
        st = self.state
        folder = self._loaded_experiment_dir / EXPORT_DIR
        stem, stamp = _export_stem(st.model_path), datetime.now().strftime('%Y%m%d_%H%M%S')
        args = (dict(st.summaries), st.gated_samples(), dict(st.sample_groups),
                _population_labels(st.gate_defs()), _antigen_map(self.controller))
        self._start_writer(lambda: write_results_csv(folder, stem, stamp, *args),
                           "Writing the gating results CSV")
        _status(self.bus, f"Gating results written to {folder}")

    def _export_statistics_csv(self):
        """Write the statistics tables as CSV files to EXPORT_DIR (background)."""
        if not self._controller_shows_loaded_experiment() or not self.state.stats:
            return
        st = self.state
        folder = self._loaded_experiment_dir / EXPORT_DIR
        stem, stamp = _export_stem(st.model_path), datetime.now().strftime('%Y%m%d_%H%M%S')
        args = (dict(st.stats), _population_labels(st.gate_defs()), _antigen_map(self.controller))
        self._start_writer(lambda: write_statistics_csv(folder, stem, stamp, *args),
                           "Writing the statistics CSV")
        _status(self.bus, f"Statistics written to {folder}")
