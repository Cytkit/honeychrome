"""
automated_gating_tab.py — Automated Gating plugin for Honeychrome
=================================================================
Builds a gating model (``.agmodel``), applies it to experiment samples and
compares the gated populations between sample groups. The first five
pages build the model (``ag_builder.BuilderPages``):

  • Transforms, Import/Setup, Hierarchy, Train, Export — see ag_builder.py.
              A saved model is loaded on Import/Setup, where its channels
              are matched to this experiment antigen by antigen.

and the last three apply it:

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
DR/Clustering plugin).

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
    QProgressBar, QGridLayout, QDialog,
)

import honeychrome
from honeychrome.settings import af_abundance_channel
from honeychrome.controller_components import differential_stats as ds
from honeychrome.view_components import suggest_groupings_dialog as sgd
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
STATS_FILE = Path('cache') / 'automated_gating' / 'stats.json'
ENRICHMENT_FILE = Path('cache') / 'automated_gating' / 'enrichment.json'
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


def _population_labels(gate_defs: list[dict]) -> dict:
    """{'gate/pop': 'gate label'} for display."""
    out = {}
    for g in gate_defs:
        for pop, info in (g.get('populations') or {}).items():
            label = (info or {}).get('label') or pop
            out[ag_core.population_key(g['gate_name'], pop)] = f"{g['gate_name']} {label}"
    return out


def model_sha(model: dict) -> str:
    """Hash of a model's gates, trained boundaries and transforms (not its
    metadata or gate modes, which run_fingerprint covers)."""
    payload = {k: (model or {}).get(k) for k in
               ('gate_definitions', 'trained_boundaries', 'transforms')}
    return hashlib.sha256(json.dumps(ag_core.to_jsonable(payload), sort_keys=True,
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
    pca, predictive: the last sample PCA and predictive analysis (from the
                     statistics run's per-sample values) — this session only.
    pending_samples: samples of a paused (or interrupted) run still to be
                     gated; empty when the last run finished.
    run_fingerprint: what the gated results depend on (model file, channel
                     alignment, gate modes, drift settings, marker channels,
                     transforms), so a paused run is only resumed with the
                     same settings.
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
    fsea           : the last enrichment run (kept in ENRICHMENT_FILE).
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
    transform_params: dict = field(default_factory=dict)
    run_transforms: dict = field(default_factory=dict)
    report_populations: list = field(default_factory=list)
    pop_filter: str = 'significant'

    group_names: list = field(default_factory=list)
    group_patterns: dict = field(default_factory=dict)
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
    run_markers: bool = True
    use_stats_parent: bool = True
    exclude_gating_markers: bool = True
    min_marker_events: int = ag_results.DEFAULT_MIN_MARKER_EVENTS
    exclude_flagged: bool = False
    stats: dict = field(default_factory=dict)
    pca_use_freq: bool = True
    pca_use_counts: bool = False
    pca_use_markers: bool = False
    pca_n_loadings: int = 10
    pred_use_freq: bool = True
    pred_use_markers: bool = False
    pred_lasso: bool = True
    pred_forest: bool = True
    pred_repeats: int = 5
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
    'modes', 'drift_limit', 'drift_action', 'run_samples', 'validation_samples', 'include_controls',
    'display_events', 'group_names', 'group_patterns', 'sample_groups', 'covariates',
    'numeric_covariates', 'pairing', 'adjust_covariates', 'contrast_mode', 'reference_group', 'pval_threshold',
    'fc_threshold', 'marker_threshold', 'use_treat', 'fdr_scope', 'run_freq', 'run_counts',
    'run_markers', 'use_stats_parent', 'exclude_gating_markers', 'min_marker_events',
    'exclude_flagged', 'pca_use_freq', 'pca_use_counts', 'pca_use_markers', 'pca_n_loadings',
    'pred_use_freq', 'pred_use_markers', 'pred_lasso', 'pred_forest', 'pred_repeats',
    'report_populations', 'pop_filter',
    'fsea_library_path', 'fsea_species', 'fsea_anchor', 'fsea_feature_types',
    'fsea_marker_readout', 'fsea_rank_metric', 'fsea_n_perm', 'fsea_seed', 'fsea_min_set_size',
    'fsea_overrides',
)


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
    if gt == '1dsep' and not ch_y:
        counts = hist_curve.yData
        y_lim = (0.0, float(np.max(counts)) if counts is not None and len(counts) else 1.0)
    regions = ag_core.population_regions(gate_def, boundary)
    fractions = fractions or {}
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
                t.setPos(lx, ly)
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
                out = load_unmixed(self._experiment_dir, key, self._snap)
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
    in a row each, for every validation sample (its own boundaries).

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
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)
        outer.addWidget(HelpToggleWidget(text=ag_help_texts.AG_RUN))

        self.banner = QLabel("")
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet("QLabel { color: #856404; background: #fff3cd; "
                                  "border: 1px solid #ffc107; border-radius: 4px; padding: 6px; }")
        outer.addWidget(self.banner)

        self.content = QWidget()
        cl = QVBoxLayout(self.content)
        cl.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.content, stretch=1)

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
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(self.COL_SAMPLE, QHeaderView.Stretch)
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
        """Samples whose AF assignment was not represented in training."""
        training = ((self.state.applied_model or {}).get('af') or {}).get('sample_af_profiles') or {}
        if not training:
            self.lbl_warnings.setText("")
            return
        seen = {tuple(v or []) for v in training.values()}
        odd = [k for k in self._to_gate()
               if tuple(af_profile_names(self.controller, k)) not in seen]
        if odd:
            self.lbl_warnings.setText(
                "These samples use an AF profile assignment that none of the training "
                "samples had, so their unmixing differs from the training data: "
                + ", ".join(_sample_label(k) for k in odd[:10]) + (" …" if len(odd) > 10 else ""))
        else:
            self.lbl_warnings.setText("")

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
            box.setText(f"The last run stopped with {n_done} sample(s) gated, but the model or "
                        "its settings have changed since, so it cannot be resumed.\n\n"
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
        markers = _marker_channels(self.controller)
        model_sha = st.model_sha
        saved_markers = st.marker_channels
        st.marker_channels = markers
        fingerprint = run_fingerprint(st, model_sha)
        st.marker_channels = saved_markers
        to_gate = list(keys)
        resume = False
        if st.pending_samples and st.summaries:
            choice = self._ask_resume(len(st.summaries), len(st.pending_samples),
                                      fingerprint == st.run_fingerprint)
            if choice is None:
                return
            resume = choice == 'resume'
            if resume:
                to_gate = [k for k in keys if k not in st.summaries]
                if not to_gate:
                    st.pending_samples = []
                    self.lbl_status.setText("Every ticked sample is already gated.")
                    self.gating_finished.emit('done')
                    return
        try:
            snap = snapshot_unmix_state(self.controller, to_gate)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot gate", str(exc))
            return
        self._before_run = {name: (dict(getattr(st, name)) if isinstance(getattr(st, name), dict)
                                   else deepcopy(getattr(st, name))) for name in (
            'run_samples', 'summaries', 'load_info', 'display', 'stats', 'fsea', 'marker_channels',
            'results_model_sha', 'pending_samples', 'run_fingerprint', 'run_transforms')}
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
        st.fsea = {}
        st.marker_channels = markers
        st.results_model_sha = model_sha
        st.run_fingerprint = fingerprint
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
        rows = ([('pooled', None)] if self.chk_pooled.isChecked() else []) + \
            [('sample', k) for k in validation]
        r = 0
        for kind, key in rows:
            head = QLabel("<b>All gated samples</b> <small>(stored boundaries, % of pooled "
                          "display events)</small>" if kind == 'pooled' else
                          f"<b>{_sample_label(key)}</b> <small>[validation"
                          + (f"; {st.sample_groups[key]}" if st.sample_groups.get(key) else '')
                          + "]</small>")
            self.tile_grid.addWidget(head, r, 0, 1, max(1, len(shown)))
            r += 1
            for c, g in enumerate(shown):
                name = g['gate_name']
                eff = ag_core.effective_gate_def(g, by_name)
                if kind == 'pooled':
                    events, parent, channels = self._pooled(st.gated_samples(), eff)
                    boundary = st.reference_boundaries().get(name, {})
                    fractions = ag_report.boundary_fractions(eff, boundary, events, channels,
                                                             parent, transforms)
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
                tile = _make_gate_tile(eff, boundary, events, channels, parent, transforms, labels,
                                       title, fractions=fractions, highlight=highlight,
                                       footer=footer)
                self.tile_grid.addWidget(tile, r, c)
            r += 1
        if not rows:
            self.tile_grid.addWidget(QLabel("Tick 'Pooled' or some validation samples."), 0, 0)

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


def statistics_design(state: 'GatingRunState', samples: list[str]) -> dict:
    """The design the statistics (and the enrichment) use for *samples*:
    {'groups': {sample: group}, 'pairing': {sample: level} or None,
    'covariates': adjustment covariates (DataFrame) or None, 'numeric':
    covariates entered as numbers, 'missing_pairing': labels of samples
    without a pairing value (pairing is None when any is missing)}."""
    st = state
    groups = {k: st.sample_groups[k] for k in samples}
    pairing, missing = None, []
    if st.pairing:
        vals = st.covariates.get(st.pairing) or {}
        missing = [_sample_label(k) for k in samples if not str(vals.get(k, '')).strip()]
        if not missing:
            pairing = {k: vals[k] for k in samples}
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
                        'results': results, 'make': lambda: maker(**kwargs)}
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
             x_label=x_label, run_label=run_label, group_a=base, group_b=other)

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
             group_a=base, group_b=other)
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
    report_populations_changed = Signal()

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
        self._popout_workers: list = []
        self._filling = False
        self._volcano_tables: dict = {}
        self._row_figures: dict = {}
        self._figures_for = None
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
        outer.addWidget(HelpToggleWidget(text=ag_help_texts.AG_RESULTS))
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
        add_concertina_page(self.toolbox, self.figures_tabs, "Figures")
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
        self.groups_table = QTableWidget(0, 2)
        self.groups_table.setHorizontalHeaderLabels(["Group", "Sample-name pattern (regex)"])
        groups_header = self.groups_table.horizontalHeader()
        groups_header.setSectionResizeMode(QHeaderView.Interactive)
        groups_header.setStretchLastSection(True)
        self.groups_table.setColumnWidth(0, 180)
        self.groups_table.verticalHeader().setVisible(False)
        self.groups_table.itemChanged.connect(self._on_group_item_changed)
        gl.addWidget(self.groups_table)
        row = QHBoxLayout()
        for text, slot in (("Add group", self._on_add_group), ("Remove group", self._on_remove_group),
                           ("Assign by pattern", self._on_assign_by_pattern),
                           ("Suggest groupings…", self._on_suggest_groupings)):
            b = QPushButton(text)
            b.clicked.connect(slot)
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
            "sample: a 'sample' column (file name, with or without extension), an optional "
            "'group' column, and one column per covariate (e.g. donor, age, sex).")
        b_import.clicked.connect(self._on_import_table)
        b_export = QPushButton("Export table…")
        b_export.setToolTip("Save the samples' groups and covariates in the same format.")
        b_export.clicked.connect(self._on_export_table)
        trow.addWidget(b_import)
        trow.addWidget(b_export)
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

    def _groupings_result(self, keys: list[str]) -> dict:
        display = self.controller.experiment.samples.get('all_samples', {}) if self.controller else {}
        return ds.suggest_groupings_from_names(keys, display)

    def _on_suggest_groupings(self):
        """Offer groupings found in the gated samples' names (see
        ``differential_stats.suggest_groupings_from_names``)."""
        keys = self.state.gated_samples()
        if len(keys) < 2:
            QMessageBox.information(self, "Suggest Groupings",
                                    "Need at least two gated samples to detect a pattern.")
            return
        result = self._groupings_result(keys)
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
        if self.state.pairing == name:
            self.state.pairing = ''
        self.state.adjust_covariates = [c for c in self.state.adjust_covariates if c != name]
        self.state.numeric_covariates = [c for c in self.state.numeric_covariates if c != name]
        self._populate_samples_table()
        self._populate_stats_choices()

    def _on_import_table(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import sample table", str(getattr(self.controller, 'experiment_dir', '')),
            "Tables (*.csv *.tsv *.txt);;All files (*)")
        if not path:
            return
        try:
            df = pd.read_csv(path, sep=None, engine='python', dtype=str)
            table = ag_results.read_sample_table(df, self.state.gated_samples())
        except Exception as exc:
            QMessageBox.warning(self, "Import table", f"Could not read the table: {exc}")
            return
        st = self.state
        for group in table['groups'].values():
            if group not in st.group_names:
                st.group_names.append(group)
        st.sample_groups.update(table['groups'])
        for name, values in table['covariates'].items():
            st.covariates.setdefault(name, {}).update(values)
        self._populate_groups()
        self._populate_stats_choices()
        msg = (f"Imported {len(table['groups'])} group assignment(s) and "
               f"{len(table['columns'])} covariate(s) from {Path(path).name}.")
        if table['unmatched']:
            msg += (f" {len(table['unmatched'])} row(s) matched no gated sample: "
                    + ", ".join(table['unmatched'][:6]) + (" …" if len(table['unmatched']) > 6 else ''))
        _status(self.bus, msg)
        QMessageBox.information(self, "Import table", msg)

    def _on_export_table(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export sample table", str(getattr(self.controller, 'experiment_dir', '')),
            "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith('.csv'):
            path += '.csv'
        st = self.state
        try:
            ag_results.sample_table_frame(st.gated_samples(), st.sample_groups,
                                          st.covariates).to_csv(path, index=False)
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
        if not (st.run_freq or st.run_counts or st.run_markers):
            QMessageBox.information(self, "Nothing to test", "Tick at least one kind of test.")
            return
        samples = self._stats_samples()
        design = statistics_design(st, samples)
        if design['missing_pairing']:
            QMessageBox.warning(self, "Pairing incomplete",
                                f"No '{st.pairing}' value for: "
                                + ", ".join(design['missing_pairing'][:10]))
            return
        groups = design['groups']
        kwargs = dict(
            group_order=list(st.group_names), contrast_mode=st.contrast_mode,
            reference=st.reference_group, pairing=design['pairing'],
            covariates=design['covariates'], numeric_covariates=design['numeric'],
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
        self._figures_for = self.state.stats
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
            large = key.endswith('heatmap') or key == 'marker_summary'
            widget = ScalableFigureWidget(
                fig, title=item['title'], experiment_dir=exp_dir,
                pop_out=lambda make=item['make'], title=item['title'], results=item.get('results'),
                large=large: self._pop_out_figure(make, title, results, large),
                initial_scale=initial_scale_for(fig, HEATMAP_SCALE_PCT) if large else 100,
                parent=self)
            self._connect_figure(widget.canvas, fig, item.get('results'))
            self.figures_tabs.addTab(widget, item['title'])
        self._add_extra_figures()

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
        if getattr(fig, '_row_keys', None):
            self._row_figures[id(canvas)] = fig
        if results is not None or getattr(fig, '_row_keys', None):
            canvas.mpl_connect('button_press_event', self._on_figure_click)
            canvas.setToolTip("Double-click a point or row to show that population on the "
                              "Populations page; right-click for more.")

    def _pop_out_figure(self, make, title: str, results=None, large: bool = False):
        """Build a fresh copy of a figure in the background and show it in
        its own window (size slider, matplotlib toolbar, hover labels and
        the same click-through to the Populations page). The window stays
        open while the Populations page is used."""
        worker = _CallWorker(make, parent=self)
        worker.finished.connect(
            lambda ok, err, fig, w=worker: self._show_pop_out(w, ok, err, fig, title, results, large))
        self._popout_workers.append(worker)
        _status(self.bus, f"Opening {title} …")
        worker.start()

    def _show_pop_out(self, worker, ok: bool, error: str, fig, title: str, results, large: bool):
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
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
            parent=dialog)
        self._connect_figure(widget.canvas, fig, results)
        canvas_key = id(widget.canvas)
        dialog.finished.connect(lambda _code, k=canvas_key: (self._volcano_tables.pop(k, None),
                                                            self._row_figures.pop(k, None)))
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(NavigationToolbar2QT(widget.canvas, dialog))
        layout.addWidget(widget, stretch=1)
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

    def _on_figure_click(self, event):
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
        menu.exec(QCursor.pos())

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
        layout.addStretch()

        for w, name in ((self.chk_pca_freq, 'pca_use_freq'), (self.chk_pca_counts, 'pca_use_counts'),
                        (self.chk_pca_markers, 'pca_use_markers'),
                        (self.chk_pred_freq, 'pred_use_freq'),
                        (self.chk_pred_markers, 'pred_use_markers'),
                        (self.chk_pred_lasso, 'pred_lasso'), (self.chk_pred_forest, 'pred_forest')):
            w.toggled.connect(lambda v, n=name: self._filling or setattr(self.state, n, bool(v)))
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
            self.cb_pred_comparison.clear()
            for base, other in (st.stats or {}).get('comparisons') or []:
                self.cb_pred_comparison.addItem(_comparison_label(base, other),
                                                json.dumps([base, other]))
        finally:
            self._filling = False
        stats = st.stats or {}
        self.chk_pca_freq.setEnabled(stats.get('freq_values') is not None)
        self.chk_pca_counts.setEnabled(stats.get('counts_values') is not None)
        self.chk_pca_markers.setEnabled(stats.get('marker_values') is not None)
        self.chk_pred_freq.setEnabled(stats.get('freq_values') is not None)
        self.chk_pred_markers.setEnabled(stats.get('marker_values') is not None)
        self.btn_pca.setEnabled(bool(stats))
        self.btn_pred.setEnabled(bool(stats) and self._predictive_worker is None)
        self._show_predictive_table()

    def _feature_labels(self) -> dict:
        """Display names for population keys and channels in feature names."""
        labels = dict(_population_labels(self.state.gate_defs()))
        labels.update(_antigen_map(self.controller))
        return labels

    def _on_show_pca(self):
        st = self.state
        if not st.stats:
            QMessageBox.information(self, "Sample PCA", "Run Statistics first.")
            return
        use = (self.chk_pca_freq.isChecked() and self.chk_pca_freq.isEnabled(),
               self.chk_pca_counts.isChecked() and self.chk_pca_counts.isEnabled(),
               self.chk_pca_markers.isChecked() and self.chk_pca_markers.isEnabled())
        if not any(use):
            QMessageBox.information(self, "Sample PCA",
                                    "Tick at least one table that the last statistics run made.")
            return
        try:
            pca = ag_results.sample_pca(st.stats, *use, n_loadings=st.pca_n_loadings,
                                        labels=self._feature_labels())
        except Exception as exc:
            QMessageBox.warning(self, "Sample PCA", str(exc))
            return
        if pca is None:
            QMessageBox.information(self, "Sample PCA",
                                    "At least 2 samples and 2 features that vary are needed.")
            return
        st.pca = pca
        self._add_extra_figures()
        self._populate_report_items()
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
        worker.finished.connect(
            lambda ok, err, res, s=status, m=message: self._on_predictive_finished(ok, err, res, s, m))
        self._predictive_worker = worker
        worker.start()

    def _on_predictive_finished(self, ok: bool, error: str, result: object, status: str,
                                message: str):
        self._predictive_worker = None
        self.btn_pred.setEnabled(bool(self.state.stats))
        if not ok:
            self.lbl_pred.setText(f"<span style='color:#b00020'>{error}</span>")
            return
        self.state.predictive = result
        summary = result['summary']
        text = (f"Minimal panel: {summary['minimal_panel']} feature(s) (best panel AUC "
                f"{summary['best_panel_auc']:.2f}).")
        if status == 'warn':
            text += f" {message}"
        self.lbl_pred.setText(text)
        self._show_predictive_table()
        self._add_extra_figures()
        self._populate_report_items()
        _status(self.bus, "Predictive analysis complete.")

    def _show_predictive_table(self):
        _clear_layout(self.pred_table_holder)
        res = self.state.predictive
        if not res:
            return
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
                pca = st.pca
                makers.append(("Sample PCA", lambda: ag_report.make_pca_figure(pca, is_dark)))
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
            widget = ScalableFigureWidget(
                fig, title=title, experiment_dir=exp_dir,
                pop_out=lambda make=make, title=title: self._pop_out_figure(make, title),
                parent=self)
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
        test, or None without one."""
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
            note = "Run statistics to see which populations differ; all are listed."
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
        comp = self._current_comparison()
        if comp is not None:
            return [comp[0], comp[1]]
        st = self.state
        have = {st.sample_groups.get(k) for k in st.gated_samples()}
        return [g for g in st.group_names if g in have]

    def _population_data(self, pop_key: str) -> dict | None:
        """Everything the comparison of one population shows."""
        st = self.state
        gate_name, pop = ag_core.split_population_key(pop_key)
        by_name = ag_core.gates_by_name(st.gate_defs())
        gdef = by_name.get(gate_name)
        if gdef is None:
            return None
        eff = ag_core.effective_gate_def(gdef, by_name)
        groups = self._comparison_groups()
        panels = []
        for g in groups:
            samples = [k for k in st.gated_samples() if st.sample_groups.get(k) == g]
            events, parent, member, channels = ag_report.pooled_population(
                st.display, samples, eff, pop, max_events=GROUP_PLOT_EVENTS, seed=DISPLAY_SEED)
            panels.append({'group': g, 'events': events, 'parent': parent, 'member': member,
                           'channels': channels, 'n_samples': sum(1 for k in samples if k in st.display),
                           'boundary': st.reference_boundaries().get(gate_name, {})})
        values = ag_results.group_population_values(st.summaries, st.gated_samples(),
                                                    st.sample_groups, pop_key, st.use_stats_parent)
        values = {g: values.get(g, []) for g in groups}
        comp = self._current_comparison()
        result = ag_results.population_result(st.stats, pop_key, _comparison_label(*comp),
                                               st.fdr_scope) if comp else None
        label = _population_labels(st.gate_defs()).get(pop_key, pop_key)
        return {'gate_def': eff, 'pop': pop, 'panels': panels, 'values': values, 'groups': groups,
                'result': result, 'label': label,
                'title': f"{label}" + (f" — {_comparison_label(*comp)}" if comp else '')}

    def _draw_population(self, item):
        _clear_layout(self.pop_view)
        self._shown_population = item.data(Qt.UserRole) if item is not None else None
        self.chk_pop_report.blockSignals(True)
        self.chk_pop_report.setChecked(self._shown_population in self.state.report_populations)
        self.chk_pop_report.setEnabled(self._shown_population is not None)
        self.chk_pop_report.blockSignals(False)
        self.btn_pop_window.setEnabled(self._shown_population is not None)
        self.btn_pop_save.setEnabled(self._shown_population is not None)
        if self._shown_population is None:
            self.lbl_pop_title.setText("")
            return
        self._fill_population_view(self.pop_view, self._shown_population, self.lbl_pop_title)

    def _fill_population_view(self, layout, pop_key: str, title_label: QLabel):
        """Each compared group's pooled biplot, then the dot and violin plot."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        data = self._population_data(pop_key)
        if data is None:
            title_label.setText("")
            return
        title_label.setText(f"<b>{data['title']}</b><br>{ag_report.p_value_text(data['result'])}")
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
        fig = ag_report.make_group_values_figure(
            data['values'], data['groups'], f"{data['label']}: % of parent",
            ag_report.p_value_text(data['result']), is_dark=_is_dark_palette())
        import honeychrome.settings as hc_settings
        canvas = FigureCanvasQTAgg(fig)
        size = int(getattr(hc_settings, 'cytometry_plot_width_target_retrieved', 220) or 220)
        canvas.setFixedSize(int(size * 1.3), size + 60)
        handler = getattr(fig, '_hover_handler', None)
        if handler is not None:
            canvas.mpl_connect('motion_notify_event', handler)
        layout.addWidget(canvas)

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

    def _population_report_figure(self, pop_key: str):
        """The report figure of one population's group comparison."""
        data = self._population_data(pop_key)
        if data is None:
            return None
        labels = build_display_label_map(
            _experiment_channels(self.controller),
            self.controller.experiment.process.get('spectral_model') or [])
        return ag_report.make_population_comparison_figure(
            data['gate_def'], data['pop'], data['panels'] if self.state.display else [],
            self._result_transforms(), labels, data['values'], data['groups'], data['title'],
            ag_report.p_value_text(data['result']))

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
        comp = self._current_comparison()
        name = _population_labels(self.state.gate_defs()).get(pop_key, pop_key)
        if comp:
            name += f" {_comparison_label(*comp)}"
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
            "item below and one PDF containing all of them. Populations ticked for the "
            "report on the Populations page are listed under Populations.")
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
            predictive=st.predictive or None,
            populations=[(k, _population_labels(st.gate_defs()).get(k, k))
                         for k in st.report_populations],
            population_figure=self._population_report_figure,
            enrichment=st.fsea or None,
        )

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

        progress = QProgressDialog("Generating report…", "Cancel", 0, len(items) + 1, self)
        progress.setWindowTitle("Generate Report")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        cancelled = False
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
                    if not section_items or cancelled:
                        continue
                    drc_report.add_section_divider(pdf, section)
                    subfolder = report_dir / drc_report.sanitize_filename(section)
                    subfolder.mkdir(parents=True, exist_ok=True)
                    for it in section_items:
                        QApplication.processEvents()
                        if progress.wasCanceled():
                            cancelled = True
                            break
                        progress.setLabelText(f"Adding {it.label} …")
                        drc_report.export_report_item(it, subfolder, pdf)
                        done += 1
                        progress.setValue(done)
            if not cancelled:
                progress.setLabelText("Writing settings document …")
                (report_dir / 'settings.txt').write_text(self._settings_text(samples),
                                                         encoding='utf-8')
                progress.setValue(len(items) + 1)
        except Exception as exc:
            log.exception("report generation failed")
            QMessageBox.critical(self, "Report failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
            progress.close()
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
        self._maybe_auto_suggest_groupings()
        self._load_stats_controls()
        self._load_explore_controls()
        self._populate_comparisons()
        self._show_results_tables()
        if not self.state.stats:
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
            self._read_enrichment(job, out)
        except Exception as exc:
            log.exception("could not read the saved enrichment")
            out['errors'].append(f"Could not read the saved enrichment: {exc}")
        self.finished.emit(out)

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
        out['fsea'] = ag_fsea.result_from_jsonable(payload['result'])

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
    def _read_results(job: dict, out: dict):
        path = job.get('results_path')
        if path is None or not Path(path).exists():
            return
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        out['results'] = payload
        fingerprint = str(payload.get('run_fingerprint') or '')
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
    Hierarchy, Train and Export build the model (ag_builder.BuilderPages,
    which owns the model state, the transforms and their persistence); Run,
    Results and Enrichment apply it (GatingRunState). Run applies the model
    as it stands on the model pages, in their transforms: state.applied_model
    is rebuilt from them (_sync_applied_model) whenever one of the last three
    pages is shown.

    Persistence: the model pages keep their own QSettings group and session
    file (see BuilderPages). The settings of the last three pages (gate
    modes, drift settings, sample selection, groups, covariates and
    statistics options) are kept in QSettings per experiment as one JSON
    string. Gated results (counts, medians and per-gate status) are kept in
    RESULTS_FILE in the experiment folder, with the samples a paused run has
    still to gate, and each sample's display record in DISPLAY_DIR; the last
    statistics run is kept in STATS_FILE and the last enrichment run in
    ENRICHMENT_FILE, both restored while they still match the gated results.
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
        self._qsettings = QSettings('honeychrome', 'plugin_automated_gating')
        self._plugin_was_active = False
        self._loaded_experiment_key = None
        self._loaded_experiment_dir = None
        self._loader = None
        self._loading = False
        self._writers: list = []
        self._legacy = {'path': '', 'alignment': {}}
        self._page_shown = None
        self._model_stale = True

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

        self.inner_tabs = SideTabWidget()
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
            experiment_dir_fn=lambda: str(self._loaded_experiment_dir or ''))
        self.inner_tabs.addTab(self.run_tab, "Run", 'player-play')
        self.inner_tabs.addTab(self.results_tab, "Results", 'table')
        self.inner_tabs.addTab(self.enrichment_tab, "Enrichment", 'sparkles')
        self._page_shown = self.inner_tabs.currentWidget()
        self.inner_tabs.currentChanged.connect(self._on_inner_tab_changed)
        content_layout.addWidget(self.inner_tabs, stretch=1)

        self.builder.modelImported.connect(self._on_model_imported)
        self.builder.gateRenamed.connect(self._on_gate_renamed)
        self.builder.modelTrained.connect(self._on_model_trained)
        self.run_tab.gating_finished.connect(self._on_gating_finished)
        self.results_tab.stats_finished.connect(self._on_stats_finished)
        self.results_tab.report_populations_changed.connect(self._save_run_state)
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
        current_key = self._settings_key()
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
        self.builder.switch_experiment(current_key, self._loaded_experiment_dir)
        self.load_state()
        return True

    def _on_inner_tab_changed(self, index: int):
        """Save what the page being left may have changed, then refresh the
        page shown (the model pages' state is large, so it is saved only on
        leaving one of them)."""
        left, self._page_shown = self._page_shown, self.inner_tabs.widget(index)
        if self.builder.owns(left):
            self.builder.save_state()
            self._model_stale = True
        self._save_run_state()
        if not self._loading:
            self._refresh_tab_at(index)

    def _refresh_active_tab(self):
        self._refresh_tab_at(self.inner_tabs.currentIndex())

    def _refresh_tab_at(self, index: int):
        widget = self.inner_tabs.widget(index)
        if self.builder.refresh_page(widget):
            return
        if widget in (self.run_tab, self.results_tab, self.enrichment_tab):
            if self._model_stale and not self.run_tab.is_running():
                self._sync_applied_model()
            widget.refresh()

    def _on_gating_finished(self, outcome: str = 'done'):
        self._result_tf = (None, None)
        self._write_results()
        self._write_stats()
        self._write_enrichment()
        self._save_run_state()
        if self.state.summaries and outcome in ('done', 'paused', 'failed'):
            self._export_results_csv()

    def _on_stats_finished(self):
        """Keep the statistics, and the settings they were run with, as soon
        as a run completes, and write them as CSV files."""
        self._write_stats()
        self._save_run_state()
        self._export_statistics_csv()

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

    def _sync_applied_model(self):
        """Point the Run, Results and Enrichment pages at the model as it
        stands on the model pages (rebuilt only when it has changed)."""
        self._model_stale = False
        st = self.state
        st.modes = self._application()['mode_per_gate']
        model = self.builder.model()
        sha = model_sha(model)
        if sha != st.model_sha or not st.applied_model:
            try:
                model = ag_core.normalize_model(model)[0]
            except Exception:
                log.exception("the model pages' gates do not form a valid hierarchy")
            st.applied_model = model
            st.model_sha = sha
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
        if self._loaded_experiment_key is None or self._loading:
            return
        payload = {name: getattr(self.state, name) for name in _PERSISTED_FIELDS}
        s = self._qsettings
        s.beginGroup(self._loaded_experiment_key)
        try:
            s.setValue('state_json', json.dumps(ag_core.to_jsonable(payload)))
        finally:
            s.endGroup()

    def load_state(self):
        """Restore settings from QSettings, then read the gated results,
        display records and statistics in the background.

        Automated Gating used to load a model file itself; when the model
        pages have no gates yet and such a file is remembered, it is read
        too and imported with the channel alignment that was accepted for it.
        """
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
            'enrichment_path': exp / ENRICHMENT_FILE,
            'display_dir': exp / DISPLAY_DIR,
            'thresholds': {'pval': st.pval_threshold, 'fc': st.fc_threshold,
                           'marker': st.marker_threshold, 'fdr_scope': st.fdr_scope,
                           'treat': st.use_treat},
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
            st.display = dict(out.get('display') or {})
            if int(payload.get('format_version') or 1) < 2:
                # Hash of the model file, from before the model pages: not comparable.
                st.results_model_sha = ''
        if out.get('stats'):
            st.stats = out['stats']
        if out.get('fsea'):
            st.fsea = out['fsea']
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
        payload = {
            'format_version': 2,
            'model_path': st.model_path,
            'model_sha256': st.results_model_sha,
            'marker_channels': list(st.marker_channels),
            'run_samples': list(st.run_samples),
            'summaries': dict(st.summaries),
            'load_info': dict(st.load_info),
            'pending_samples': list(st.pending_samples),
            'run_fingerprint': st.run_fingerprint,
            'run_transforms': deepcopy(st.run_transforms),
        }
        self._start_writer(lambda: _write_json_atomic(path, payload), "Saving the gating results")

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

    def _write_enrichment(self):
        """Write the last enrichment run to ENRICHMENT_FILE (in the
        background), or remove the file when there is none."""
        if not self._controller_shows_loaded_experiment():
            return
        path = self._loaded_experiment_dir / ENRICHMENT_FILE
        st = self.state
        if not st.fsea:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                log.exception("could not remove %s", path)
            return
        result, fingerprint = dict(st.fsea), st.run_fingerprint

        def _job():
            _write_json_atomic(path, {'format_version': 1, 'run_fingerprint': fingerprint,
                                      'result': ag_fsea.result_to_jsonable(result)})

        self._start_writer(_job, "Saving the enrichment")

    def _export_enrichment_csv(self):
        """Write the enrichment tables as CSV files to EXPORT_DIR (background)."""
        if not self._controller_shows_loaded_experiment() or not self.state.fsea:
            return
        st = self.state
        folder = self._loaded_experiment_dir / EXPORT_DIR
        stem, stamp = _export_stem(st.model_path), datetime.now().strftime('%Y%m%d_%H%M%S')
        result = dict(st.fsea)
        self._start_writer(lambda: ag_fsea_ui.write_enrichment_csv(folder, stem, stamp, result),
                           "Writing the enrichment CSV")
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
