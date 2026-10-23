"""
dr_clustering_tab.py — Honeycluster Plugin for Honeychrome
===========================================================================
Honeychrome plugin providing:
  • Dimensionality reduction  — UMAP, PaCMAP, tSNE
  • Clustering               — FlowSOM, Leiden, HDBSCAN
  • Transform inspector      — read-only preview of Logicle parameters
  • Group & statistics       — sample grouping, differential statistics,
                               optional predictive modelling
  • Workspace                — customisable matplotlib scatter-plot canvas

Plugin contract (required by plugin_loaders.py):
  plugin_name  str           — displayed as the main-window tab title
  PluginWidget(QWidget)      — instantiated with (bus, controller)

Distribution:
  Lives in  honeychrome/bundled_plugins/dr_clustering_tab.py  (see
  plugin_loaders.py — discovered automatically as *_tab.py, gated behind
  'EnableBundledPlugin_dr_clustering_tab' in QSettings). Never copied into
  ~/Experiments/plugins/ — that mechanism is for the open, unreviewed
  user-plugin ecosystem only.
  Users enable it in App Configuration → restart → tab appears.

Dependencies:
  Heavy ML packages (umap-learn, openTSNE, pacmap, graspologic-native,
  networkx, hnswlib, statsmodels, hdbscan) are ordinary Honeychrome dependencies
  now (requirements.in) — installed as part of the app's own build, same
  as pandas or scikit-learn. All ML imports stay LAZY (inside method
  bodies), unchanged from before; this only removes the runtime
  pip-install path, not the lazy-import discipline.

Transforms:
  Transform preview, auto-transform, CSV import/export, QSettings persistence.

DR pipeline:
  ConfigTab DR section: algorithm radios, per-algorithm hyperparameter panels,
  Train / Apply to All Samples buttons, status labels.
  _run_umap(), _run_pacmap() with hnswlib kNN.
  _apply_dr_to_all_samples() projects every sample.

Clustering:
  ConfigTab clustering section: FlowSOM / Leiden / HDBSCAN radio buttons
  with per-algorithm hyperparameter panels.
  FlowSOM: SOM grid, manual metacluster count (2-200).
  Leiden: hnswlib kNN graph of the clustering's own training events;
  graspologic-native community detection.
  HDBSCAN: on a DR embedding only (PaCMAP, then UMAP, then tSNE by default).
  All algorithms assign labels to every sample via nearest-centroid.
  Cluster colours from cc.glasbey; noise label −1 → grey.

Groups & Stats:
  GroupsStatsTab: named groups with regex auto-population; moderated
  linear models (controller_components/moderated_stats.py) with TREAT
  thresholds, covariate adjustment and cluster-first MFI testing;
  negative-binomial GLM on counts; heatmap+dendrogram, volcano plot, MFI
  cluster summary, CSV result export; optional nested cross-validated
  predictive models (drc_predictive.py).

Workspace:
  Per-marker scatter colouring, per-cluster colour pickers (right-click), magic-wand
  display-config copy/paste, multi-page PDF export.

Report tab:
  ReportTab (drc_report.py): per-source-tab tick-lists (Workspace, Cluster
  Annotation, Stats), PNG/CSV export per item into a timestamped run
  folder, one combined PDF, and an always-generated settings.txt
  documenting how to reproduce the analysis.
"""

# ---------------------------------------------------------------------------
# 0.  Module-level setup
#     Bundled plugin: all heavy ML packages are ordinary Honeychrome
#     dependencies (requirements.in), installed as part of the app's own
#     build/environment — no runtime bootstrap/self-install needed. All ML
#     imports stay LAZY (inside method bodies) regardless, same as before.
# ---------------------------------------------------------------------------
import os as _os
import sys
import warnings


def _suppress_third_party_warnings():
    """Silence noisy third-party warnings triggered by the lazy ML imports
    below (umap/faiss/hnswlib Swig types, PaCMAP/UMAP notices). Called from
    PluginWidget.__init__ on the main thread — kept at the same call site
    the old _bootstrap() used, so warning-filter timing relative to the
    rest of __init__ doesn't change."""
    warnings.filterwarnings('ignore', category=ImportWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning,
                            message='.*SwigPy.*')
    warnings.filterwarnings('ignore', category=DeprecationWarning,
                            message='.*swigvarlink.*')
    # PaCMAP random_state notice
    warnings.filterwarnings('ignore', category=UserWarning,
                            message='.*random state is set.*')
    # UMAP n_jobs overridden by random_state — expected, not actionable.
    warnings.filterwarnings('ignore', category=UserWarning,
                            message='.*n_jobs value.*overridden.*random_state.*')


# ---------------------------------------------------------------------------
# 1.  Standard imports
#     Split into two groups:
#
#     Group A — safe to import on a background thread (pure Python / C
#     extensions that don't touch Qt).  These are needed for class
#     definitions so they must be at module scope.
#
#     Group B — deferred until _ensure_qt_imports() is called from
#     PluginWidget.__init__ (main thread).  pyqtgraph in particular
#     creates Qt objects on first import; doing so on a background thread
#     produces QObject::setParent cross-thread warnings.
#
#     All heavy ML imports (umap, pacmap, etc.) remain LAZY —
#     imported inside method bodies only.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Safe at module level (no Qt object creation on import)
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field
from copy import copy as _shallow_copy, deepcopy
import json
from datetime import datetime
from pathlib import Path
from typing import Any
import csv
import re
import time
import traceback

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QTimer, QSettings, QEvent, QThread, Signal, QSize
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QPushButton, QLabel, QComboBox, QGroupBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QRadioButton,
    QButtonGroup, QListWidget, QListWidgetItem,
    QAbstractItemView, QSplitter, QFrame, QGridLayout,
    QLineEdit, QFileDialog, QMessageBox, QSizePolicy,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QTabWidget, QColorDialog, QProgressBar, QSlider,
    QDialog, QTextEdit, QApplication, QToolBox, QStackedWidget,
)

from honeychrome.controller_components.functions import (
    apply_gates_in_place,
)
from honeychrome.controller_components import unmix_method
from honeychrome.controller_components.sample_loader import (
    sample_key_for_path, snapshot_unmix_state,
)
from honeychrome.controller_components.job_gate import acquire_job, guard_while_busy, wait_if_busy
from honeychrome.controller_components.spectral_correction import offered_samples
from honeychrome.controller_components.per_sample_gating_functions import SampleGating
from honeychrome.controller_components.transform import Transform
from honeychrome.view_components.ordered_multi_sample_picker import OrderedMultiSamplePicker
from honeychrome.view_components.copyable_table_widget import CopyableTableWidget
from honeychrome.view_components.help_toggle_widget import HelpToggleWidget
from honeychrome.view_components.side_tab_widget import SideTabWidget
from honeychrome.view_components import suggest_groupings_dialog as sgd
from honeychrome.view_components.checkable_header import CheckableHeaderView
from honeychrome.controller_components.differential_stats import suggest_groupings_from_names
from honeychrome.view_components.differential_plots import (
    make_heatmap_figure,
    make_marker_summary_figure,
    make_scatter_hover_handler as _make_scatter_hover_handler,
    make_volcano_figure,
    stamp_run_label,
    style_figure_theme as _style_figure_theme,
    text_width_inches,
    wrap_label,
)
import honeychrome.settings as hc_settings

# ---------------------------------------------------------------------------
# Split helper modules (siblings in the plugins directory).  These hold the
# data pipeline, clustering, statistics and logging logic that used to live in
# PluginWidget.  The plugins directory is not on sys.path under the file-based
# plugin loader, so add it before importing.  (None of these are named
# *_tab.py, so the loader will not pick them up as separate tabs.)
# ---------------------------------------------------------------------------
_PLUGIN_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

import drc_jit  # first: sets NUMBA_THREADING_LAYER before numba is imported
import drc_logging
import drc_pipeline
import drc_clustering
import drc_stats
import drc_persist
import drc_run_archive
import drc_gate_tree
import drc_scatter
import drc_cluster_id
import drc_report
import drc_help_texts
import drc_predictive
import drc_workspace_canvas
import drc_workspace_export
import drc_time_estimate
import drc_run_align
import drc_umap_index
from transforms_panel import (
    TransformInspector, pool_population_events, transform_from_params,
    transform_params as _transform_params,
)

_log = drc_logging.get_logger(__name__)

# ---------------------------------------------------------------------------
# Deferred: only pyqtgraph, colorcet, and cytometry_plot_components create
# Qt objects on first import.  Module exec runs on a background QThread
# (see _PluginLoaderWorker); importing these there triggers
# QObject::setParent cross-thread warnings.  _ensure_qt_imports() is called
# from PluginWidget.__init__ on the main thread instead.
# All class-body code that uses pg / cc / ZoomAxis etc. is inside methods,
# so None placeholders here are fine — they're filled before any widget runs.
# ---------------------------------------------------------------------------
cc = None          # colorcet — filled by _ensure_qt_imports()
pg = None          # pyqtgraph — filled by _ensure_qt_imports()
ZoomAxis = None
NoPanViewBox = None
TransparentGraphicsLayoutWidget = None
InteractiveLabel = None

_qt_imports_done = False


def _ensure_qt_imports():
    """Import pyqtgraph / colorcet on the main thread (called from PluginWidget.__init__)."""
    global cc, pg, ZoomAxis, NoPanViewBox, TransparentGraphicsLayoutWidget, InteractiveLabel
    global _qt_imports_done
    if _qt_imports_done:
        return
    import colorcet as _cc
    cc = _cc
    import pyqtgraph as _pg
    pg = _pg
    from honeychrome.view_components.cytometry_plot_components import (
        ZoomAxis as _ZA, NoPanViewBox as _NPV,
        TransparentGraphicsLayoutWidget as _TGLW, InteractiveLabel as _IL,
    )
    ZoomAxis = _ZA
    NoPanViewBox = _NPV
    TransparentGraphicsLayoutWidget = _TGLW
    InteractiveLabel = _IL
    _qt_imports_done = True

# ---------------------------------------------------------------------------
# 2.  Plugin identity
# ---------------------------------------------------------------------------
plugin_name = 'Honeycluster'

# Version of the DR parameter defaults saved per experiment; settings from an
# older version keep their values except the defaults that changed since.
DR_DEFAULTS_VERSION = 2

# QSettings 'application' identifier for this plugin's persisted
# gates/channels/DR-clustering state -- deliberately frozen to the
# plugin's ORIGINAL name, independent of plugin_name (the display
# name shown on the tab). QSettings('honeychrome', app) scopes an
# entirely separate store per distinct *app* string (a different INI
# file on Linux, a different plist domain on macOS) -- so renaming
# plugin_name for the tab label would silently orphan every
# experiment's already-saved settings under the old app string,
# exactly reproducing the "my data vanished on restart" symptom
# this plugin already fixed once. Never derive this from plugin_name;
# if it ever needs to change, add an explicit migration in
# load_state() that reads the old key when the new one is empty.
_SETTINGS_APP_KEY = 'plugin_DR / Clustering / Statistics'


# ---------------------------------------------------------------------------
# 3.  Module-level helper functions
# ---------------------------------------------------------------------------

def apply_gates_union_by_lookup_table(cytometry_data_dictionary, gate_names: list[str]):
    """
    Return the UNION of event_data belonging to ANY gate in *gate_names*,
    using the controller's gate lookup tables. Those belong to the sample
    on display, so this is
    only for that sample's events; other samples are gated with
    ``per_sample_gating_functions.SampleGating``.

    Returns
    -------
    np.ndarray  shape (n_gated_events, n_channels)
    """
    gate_membership = {
        'root': np.ones(len(cytometry_data_dictionary['event_data']), dtype=np.bool_)
    }
    cytometry_data_dictionary.update({'gate_membership': gate_membership})
    gates_to_calculate = [
        g[0] for g in cytometry_data_dictionary['gating'].get_gate_ids()
    ]
    apply_gates_in_place(cytometry_data_dictionary, gates_to_calculate=gates_to_calculate)
    masks = [cytometry_data_dictionary['gate_membership'][g] for g in gate_names]
    union_mask = np.logical_or.reduce(masks)
    return cytometry_data_dictionary['event_data'][union_mask]


def _read_transforms_from_experiment(controller):
    """
    Read transform parameters from the experiment model and return them in
    a plugin-friendly dict keyed by the Logicle parameter names W / A / T / M.

    The experiment stores transforms under controller.experiment.cytometry
    with keys 'transforms' (unmixed) and 'raw_transforms' (raw).  Each
    channel entry is a dict with keys:
        scale_t    → T (top of scale / ADC ceiling)
        logicle_w  → W (width basis)
        logicle_m  → M (decades)
        logicle_a  → A (negative decades)
        id         → transform type: 0=linear, 1=logicle, 2=log
        limits     → [min, max] of the TRANSFORMED (plotted) axis range, as
                     set in the Transforms tab -- see
                     functions.py::generate_transformations(), which passes
                     this straight to Transform.set_transform(limits=...).
                     Previously dropped here, so every Transform this
                     plugin built on its own (violin ticks/positioning)
                     silently fell back to the class default [0, 1]
                     instead of the channel's actual configured range.

    We prefer the unmixed transforms (which match what the plugin works on).
    Fall back to raw_transforms if unmixed are absent.

    Returns
    -------
    dict  {channel_name: {'W': float, 'A': float, 'T': float, 'M': float,
                          'id': int, 'limits': [float, float]}}
          Empty dict if transforms are not yet configured.
    """
    transforms = (
        controller.experiment.cytometry.get('transforms')
        or controller.experiment.cytometry.get('raw_transforms')
        or {}
    )
    return _channel_transform_params_from(transforms)


def _channel_transform_params_from(transforms: dict) -> dict:
    """
    Convert {channel: transform parameter dict} (the experiment's own keys:
    scale_t, logicle_w, logicle_m, logicle_a, id, limits) into the
    {channel: {'T', 'W', 'M', 'A', 'id', 'limits'}} form held in
    PipelineState.channel_transform_params.
    """
    result = {}
    for channel, params in transforms.items():
        if not isinstance(params, dict):
            continue
        result[channel] = {
            'T': float(params.get('scale_t') or 262144.0),
            'W': float(params.get('logicle_w') if params.get('logicle_w') is not None else 0.5),
            'M': float(params.get('logicle_m') or 4.5),
            'A': float(params.get('logicle_a') or 0.0),
            'id': params.get('id', 1),   # 0=linear, 1=logicle, 2=log
            'limits': list(params.get('limits') or [0, 1]),
        }
    return result


def _working_channel_transform_params(controller, saved: dict) -> dict:
    """
    channel_transform_params for the open experiment: the saved transforms
    where there are any, the experiment's own for the remaining channels.
    """
    params = _read_transforms_from_experiment(controller)
    params.update({ch: v for ch, v in _channel_transform_params_from(saved).items()
                   if ch in params})
    return params

def _qs_list(settings, key: str) -> list[str]:
    """
    Read a QSettings value that was written as a list of strings.

    QSettings' native backends do not round-trip Python lists faithfully.
    The INI backend (Linux, and any non-native format) stores an empty
    list as '@Invalid()' and returns None for it — the `default` argument
    does not apply, because the key exists. A one-element list comes back
    as a bare string, which list() would then split into characters.
    The macOS plist backend has neither problem, so both faults are
    invisible there.
    """
    raw = settings.value(key, None)
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw else []
    try:
        return [str(v) for v in raw]
    except TypeError:
        return [str(raw)]


def _qs_bool(settings, key: str, default=None):
    """Read a QSettings bool. INI backends return 'true'/'false' strings."""
    raw = settings.value(key, None)
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ('true', '1', 'yes', 'on')


def _qs_int(settings, key: str, default=None):
    """Read a QSettings int. INI backends return strings."""
    raw = settings.value(key, None)
    if raw is None or raw == '':
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _with_extension(path: str, ext: str) -> str:
    """Append *ext* ('.csv') unless *path* already ends in it."""
    return path if path.lower().endswith(ext.lower()) else path + ext


def _cluster_display_name(names: dict, label: int) -> str:
    """Display name for a cluster id, falling back to 'Cluster N'."""
    if label in names:
        return names[label]
    return 'Noise' if label < 0 else f'Cluster {label}'


# Muted italic styling for inline hint labels. Deliberately sets no
# font-size: an absolute px size ignores the user's font scaling. On a
# dark palette palette(mid) is grey on dark grey, so dark mode uses a
# light grey instead.
def _hint_style() -> str:
    app = QApplication.instance()
    is_dark = (app is not None
               and app.palette().color(QPalette.ColorRole.Base).value() < 128)
    colour = '#d0d0d0' if is_dark else 'palette(mid)'
    return f"color: {colour}; font-style: italic;"


class _HintLabel(QLabel):
    """Word-wrapped italic hint text that follows the light/dark palette."""

    def __init__(self, text: str = '', parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self._style = ''
        self._apply_hint_style()

    def _apply_hint_style(self):
        style = _hint_style()
        if style != self._style:
            self._style = style
            self.setStyleSheet(style)

    def changeEvent(self, event):
        if event.type() == QEvent.PaletteChange:
            self._apply_hint_style()
        super().changeEvent(event)


def _is_viability_antigen(antigen: str) -> bool:
    """
    True if *antigen* canonicalises to the 'Viability' marker in
    marker_database.csv (covers 'fixable', 'Livedead', 'viable',
    'viablity', 'live-dead', 'LD', and 'Viability' itself). Viability dyes
    carry a real antigen label -- they are not scatter or an empty
    detector -- but are not clustering markers, so they are excluded from
    the default channel selection alongside them.

    Falls back to False (never excludes) if label_matching is unavailable
    or antigen is blank, since a missing optional dependency should never
    make a channel silently disappear from the default selection.
    """
    if not antigen:
        return False
    try:
        from honeychrome.controller_components.label_matching import (
            match_marker, get_marker_db,
        )
    except ImportError:
        return False
    return match_marker(antigen, get_marker_db()) == 'Viability'


def _antigen_dash_labels(controller) -> dict[str, str]:
    """
    Map channel name -> 'Antigen - Channel' display label — this plugin's
    own variant of functions.py::build_display_label_map(), which instead
    joins with a space and is shared by other tabs we don't want to affect
    here. Channels with no antigen assigned map to themselves.
    """
    try:
        unmixed_pnn = controller.experiment.settings.get('unmixed', {}).get(
            'event_channels_pnn') or []
        spectral_model = controller.experiment.process.get('spectral_model') or []
    except (AttributeError, KeyError):
        return {}
    label_to_antigen = {
        c.get('label'): (c.get('antigen') or '') for c in spectral_model
    }
    result = {}
    for ch in unmixed_pnn:
        antigen = label_to_antigen.get(ch, '')
        result[ch] = f'{antigen} - {ch}' if antigen else ch
    return result

def _antigen_or_label_map(controller) -> dict[str, str]:
    """
    Map channel name -> antigen if assigned, else the channel/label itself.
    Used where space is tight (MFI Volcano/Heatmap tick labels) and the
    combined 'Antigen - Channel' format (_antigen_dash_labels) would be
    too verbose.
    """
    try:
        unmixed_pnn = controller.experiment.settings.get('unmixed', {}).get(
            'event_channels_pnn') or []
        spectral_model = controller.experiment.process.get('spectral_model') or []
    except (AttributeError, KeyError):
        return {}
    label_to_antigen = {
        c.get('label'): (c.get('antigen') or '') for c in spectral_model
    }
    result = {}
    for ch in unmixed_pnn:
        antigen = label_to_antigen.get(ch, '')
        result[ch] = antigen if antigen else ch
    return result

def _resolve_is_dark(state: 'PipelineState') -> bool:
    """
    Whether plots should render with dark styling, given the Workspace
    theme toggle. 'auto' mirrors the live app palette — the only
    behaviour that existed before; 'light'/'dark' force it regardless,
    replacing what used to be three separate ad hoc
    QApplication.palette() checks scattered across this file.
    """
    theme = getattr(state, 'plot_theme', 'auto')
    if theme == 'dark':
        return True
    if theme == 'light':
        return False
    app = QApplication.instance()
    palette = app.palette() if app is not None else QPalette()
    return palette.color(QPalette.ColorRole.Base).value() < 128


def _style_combo_popup(combo: 'QComboBox'):
    """
    Force readable dropdown-list colours on a QComboBox. Row
    labels already flip white/black automatically via QPalette, but the
    popup's item view doesn't reliably inherit that on every platform —
    left alone, items render in black text regardless of theme, which is
    unreadable against a dark popup. Keyed off the live app palette (same
    check as PlotCard's "Label colour" button), not state.plot_theme,
    since that flag only affects matplotlib figure colours.
    """
    app = QApplication.instance()
    is_dark = bool(app) and app.palette().color(QPalette.ColorRole.Base).value() < 128
    if is_dark:
        combo.setStyleSheet(
            "QComboBox QAbstractItemView { color: white; background-color: #2b2b2b; "
            "selection-background-color: #505050; selection-color: white; }"
        )
    else:
        combo.setStyleSheet(
            "QComboBox QAbstractItemView { color: black; background-color: white; "
            "selection-background-color: #d0d0d0; selection-color: black; }"
        )


def _contrasting_text_color(hex_color: str) -> str:
    """Return 'black' or 'white' -- whichever reads better against the
    given hex background. Used for swatch-style buttons whose
    background is a user-chosen colour, so it can't be assumed to already
    contrast with the app's own light/dark theme."""
    c = QColor(hex_color)
    luminance = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
    return 'black' if luminance > 140 else 'white'


def _resolve_group_colour(state: 'PipelineState', name: str) -> str:
    """Return this group's colour, assigning the next unused
    colorcet glasbey swatch and persisting it on first use. Shared between
    GroupsStatsTab (Comparison Groups table / Sample PCA) and PlotCard
    (Workspace 'Group' colour mode) so both always draw from the exact
    same state.group_colors dict and never disagree with each other."""
    colour = state.group_colors.get(name)
    if colour:
        return colour
    import colorcet as cc
    used = set(state.group_colors.values())
    for c in cc.glasbey:
        if c not in used:
            colour = c
            break
    else:
        colour = cc.glasbey[len(state.group_colors) % len(cc.glasbey)]
    state.group_colors[name] = colour
    return colour


def _sample_groups_by_rel(controller, state: 'PipelineState') -> dict[str, str]:
    """Return {rel_path: group_name}. state.sample_groups is
    keyed by each sample's FULL path (see
    GroupsStatsTab._populate_table), but Workspace plot data (DR
    embeddings, cluster labels) is keyed by path relative to
    raw_samples_subdirectory (see PluginWidget._archive_dr_run) -- this
    converts once so PlotCard's 'Group' colour mode can look samples up
    by the same rel-path keys its own 'origin' array already carries."""
    try:
        raw_subdir = controller.experiment.settings['raw']['raw_samples_subdirectory']
    except (KeyError, AttributeError):
        raw_subdir = None
    out: dict[str, str] = {}
    for sp, g in state.sample_groups.items():
        rel = sp
        if raw_subdir:
            try:
                rel = str(Path(sp).relative_to(raw_subdir))
            except ValueError:
                pass
        out[rel] = g
    return out


def _new_scrollable_canvas(fig):
    """
    FigureCanvasQTAgg subclass that ignores wheel events so
    mouse-scroll passes through to an enclosing QScrollArea instead of
    being swallowed by matplotlib's Qt backend, which unconditionally
    accept()s wheel events for its own (unused, in this plugin) scroll-
    to-zoom. Same 'ev.ignore() to bubble to the parent scroll area'
    convention TransparentGraphicsLayoutWidget already uses for
    pyqtgraph elsewhere in the app.

    FigureCanvasQTAgg is imported locally here (not hoisted to module
    level) to match this file's existing convention of importing it
    inside each function that needs it.
    """
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    class _ScrollableFigureCanvas(FigureCanvasQTAgg):
        def wheelEvent(self, ev):
            ev.ignore()

    return _ScrollableFigureCanvas(fig)


class _FrozenScrollArea(QScrollArea):
    """
    QScrollArea that never scrolls itself on mouse wheel. Used for the
    Marker Heatmap's frozen row/column header strips -- their own
    scrollbars are hidden and only ever moved in code (mirroring the
    main heatmap's scrollbars), so a wheel event over a header shouldn't
    scroll the header out of sync with the data it's labelling.
    """
    def wheelEvent(self, event):
        event.ignore()


_SideTabWidget = SideTabWidget


# Events kept per cluster for the Cluster Annotation plots (violins, Marker
# Heatmap, Ridgelines). Larger clusters are thinned by even systematic
# sampling across the run's samples (deterministic, the same events in
# every channel) before the transform is applied. The median, the violin
# density and the 200-bin histogram of 20,000 events agree with the full
# cluster to well within the resolution of the plots. None keeps every event.
CLUSTER_PLOT_MAX_EVENTS = 20_000
# Workspace legend: placements and spacing (in font-size units, as matplotlib's
# legend() takes them); LEGEND_GAP_IN is the gap between plot and legend.
LEGEND_POSITIONS = ('Separate', 'Right', 'Below', 'Off')
LEGEND_HANDLE_LENGTH = 1.0
LEGEND_HANDLE_PAD = 0.4
LEGEND_LABEL_SPACING = 0.3
LEGEND_COLUMN_SPACING = 1.0
LEGEND_GAP_IN = 0.15
# Workspace plot and page sizes (mm). LEGEND_ALLOWANCE_MM is reserved beside
# or below a new plot before its legend has been drawn.
MM_PER_INCH = 25.4
PLOT_MIN_MM, PLOT_MAX_MM, PLOT_DEFAULT_MM = 20.0, 280.0, 60.0
PAGE_MIN_MM, PAGE_MAX_MM = 50.0, 1000.0
LEGEND_ALLOWANCE_MM = 30.0
DUPLICATE_OFFSET_MM = 10.0
INSPECTOR_MIN_WIDTH_PX = 320
EXPORT_DPI_CHOICES = (300, 600)
# Sample Group Assignment table column sizes (pixels): the sample name is
# capped so a long name cannot push the other columns out of view;
# covariates get room to type.
SAMPLE_COLUMN_MAX_WIDTH = 280
COVARIATE_COLUMN_MIN_WIDTH = 110


def _channel_xform(state, ch: str):
    """
    The Transforms-tab transform for *ch* as an object with .apply(), or
    None if no transform is configured or the channel uses the
    'default'/time-gate scale. Build it once per channel and reuse it
    across clusters.
    """
    params = state.channel_transform_params.get(ch)
    if not params:
        return None
    tr = Transform(
        scale_t=params['T'], logicle_w=params['W'],
        logicle_m=params['M'], logicle_a=params['A'],
    )
    tr.set_transform(id=params['id'], limits=params.get('limits', [0, 1]))
    return tr.xform


def _apply_channel_transform(state, ch: str, values: np.ndarray) -> np.ndarray:
    """
    Apply the SAME transform configured in the Transforms tab to raw
    values, for display purposes (e.g. violin plots) -- uses the xform
    built by _channel_xform, the same conversion TransformTab itself uses
    to go from raw to displayed scale. Returns values unchanged if no
    transform is configured for this channel yet.
    """
    xform = _channel_xform(state, ch)
    if xform is None:
        return values
    return xform.apply(values)

def _channel_axis_ticks(state, ch: str):
    """
    (major_ticks, minor_ticks, limits) for channel *ch* in TRANSFORMED
    (plotted) coordinates -- ticks are (position, label) tuples, straight
    from the same Transform.ticks() method cytometry_plot_widget.py feeds
    to axis.setTicks() for the 2D histograms and Workspace marker overlays;
    limits is the [min, max] transformed-axis range from the Transforms tab.
    Returns None if no transform is configured, or it's a linear/time-gate
    'default' transform with no tick scheme of its own.
    """
    params = state.channel_transform_params.get(ch)
    if not params:
        return None
    tr = Transform(
        scale_t=params['T'], logicle_w=params['W'],
        logicle_m=params['M'], logicle_a=params['A'],
    )
    tr.set_transform(id=params['id'], limits=params.get('limits', [0, 1]))
    if tr.xform is None:
        return None
    result = tr.ticks()
    if result is None:
        return None
    minor_ticks, major_ticks = result
    return major_ticks, minor_ticks, tr.limits

def _log1p_powers_of_ten_ticks(vmax_raw: float):
    """
    Tick (position, label) pairs for a log1p-scaled colourbar: positions
    sit at log1p(10**n) so the mapping is log1p, but the LABEL shown is
    the raw value at that power of ten -- same '10' + unicode-superscript
    convention Honeychrome's 2D histogram axes use (see transform.py),
    just for log1p rather than the full logicle transform.
    """
    superscripts = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
                    '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'}
    def to_superscript(n):
        return "".join(superscripts.get(c, c) for c in str(n))

    if vmax_raw <= 0:
        return [0.0], ['0']
    max_power = int(np.floor(np.log10(max(vmax_raw, 1.0))))
    positions, labels = [0.0], ['0']
    for p in range(0, max_power + 1):
        raw = 10.0 ** p
        positions.append(float(np.log1p(raw)))
        labels.append(f"10{to_superscript(p)}")
    return positions, labels


def _non_control_sample_paths(controller) -> list[str]:
    """
    Every experiment sample path, excluding single-stain and unstained
    controls -- those are spectral/AutoSpectral reference samples, never
    biological samples to assign to test groups.
    """
    samples = controller.experiment.samples
    all_samples = list(samples.get('all_samples', {}).keys())
    excluded = set(samples.get('single_stain_controls', []) or []) \
             | set(samples.get('unstained_samples', []) or [])
    return [sp for sp in all_samples if sp not in excluded]


# ---------------------------------------------------------------------------
# 4.  PipelineState — all in-memory results for the plugin session
# ---------------------------------------------------------------------------

@dataclass
class PipelineState:
    """
    Single source of truth for all plugin computation results.

    Owned by PluginWidget and passed by reference to each inner tab so they
    all read from and write to the same object.  Nothing in this class
    computes — it only stores.

    Transforms
    ----------
    saved_transforms holds this plugin's transform of every channel
    (Transforms tab).  An experiment opened here for the first time starts
    from the transforms defined in Unmixed Data; after that the saved ones
    are kept, with every adjustment made in the Transforms tab, until the
    user reloads from the experiment.  channel_transform_params is the
    form the other tabs read.  Neither is ever written back to the
    experiment.
    """

    # --- Configuration ---
    selected_gates: list[str] = field(default_factory=list)
    # Checked gate names.  A run is built from the UNION of events
    # across all listed gates.  Empty list = no gate selected.
    selected_channels: list[str] = field(default_factory=list)
    training_sample_ids: list[str] = field(default_factory=list)
    n_training_events: int = 10_000

    # --- Transforms (read from experiment; never written back) ---
    channel_transform_params: dict = field(default_factory=dict)
    # {channel_name: {'W': float, 'A': float, 'T': float, 'M': float,
    #                 'id': int, 'limits': [float, float]}}
    saved_transforms: dict = field(default_factory=dict)
    # {channel_name: transforms_panel.transform_params() dict}

    # --- Dimensionality reduction ---
    trained_reducers: dict[str, Any] = field(default_factory=dict)
    # {algorithm_name: fitted reducer object} — current/most-recent model.
    # DR entries (UMAP/tSNE/PaCMAP) exist only in the session they
    # were trained in; FlowSOM/Leiden/HDBSCAN entries are plain tree-data
    # array dicts and are persisted with the sidecar.
    dr_status: dict[str, str] = field(default_factory=dict)
    # {algorithm_name: 'idle' | 'done' | 'error'}
    dr_timestamps: dict[str, str] = field(default_factory=dict)
    # {algorithm_name: ISO timestamp string}
    embeddings: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    # {algorithm_name: {sample_path: np.ndarray shape (n_events, 2)}} — current/most-recent only
    embedding_rows: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    # {algorithm_name: {sample_path: np.ndarray int32}} -- event keys: the
    # FCS-file row of every embedding row (drc_pipeline.load_gated_rows),
    # so an embedding can be matched to any later gating by event.
    dr_runs: list[dict] = field(default_factory=list)
    # Ordered archive of completed DR training runs, parallel to
    # clustering_runs below.  Each entry (manifest fields + heavy payload):
    # {
    #   'run_id': str, 'kind': 'dr', 'label': str, 'algorithm': str,
    #   'gates': list[str], 'training_sample_ids': list[str],
    #   'n_samples': int, 'n_events': int, 'channels': list[str],
    #   'params': dict, 'timestamp': str, 'n_clusters': None,
    #   'storage': str, 'embeddings': dict[str, np.ndarray],
    # }
    # Archived automatically each time a DR algorithm finishes training
    # (see PluginWidget._archive_dr_run); 'Apply to All Samples' afterwards
    # extends the SAME run's 'embeddings' in place rather than creating a
    # new entry (see PluginWidget._update_archived_dr_run).

    # --- Clustering ---
    cluster_labels: dict[str, np.ndarray] = field(default_factory=dict)
    # {sample_path: np.ndarray dtype int32}  — most-recent run (active)
    cluster_colors: dict[int, str] = field(default_factory=dict)
    # {cluster_id: hex colour string}
    cluster_names: dict[int, str] = field(default_factory=dict)
    # {cluster_id: display label — defaults to str(id) if absent}
    cluster_marker_values: dict[str, tuple] = field(default_factory=dict)
    # {sample_path: (raw_values: np.ndarray, channel_names: list[str],
    # rows: np.ndarray)} -- a per-cluster sample taken at the SAME instant
    # labels are assigned (see drc_clustering.py's
    # _snapshot_marker_values); rows index cluster_labels[sample_path].
    # Archived into the run's marker-values bundle.
    cluster_rows: dict[str, np.ndarray] = field(default_factory=dict)
    # {sample_path: np.ndarray int32} -- event keys of cluster_labels' rows
    # (the gated events, or the source embedding's rows for a DR-space run).
    n_clusters: int | None = None
    active_clustering_algorithm: str | None = None
    clustering_runs: list[dict] = field(default_factory=list)
    # Ordered archive of completed runs.  Each entry (manifest fields +
    # heavy payload — see drc_run_archive.archive_clustering_run):
    # {
    #   'run_id':    str,
    #   'kind':      'clustering',
    #   'label':     str,                      # display name shown in combo
    #   'algorithm': str,                      # 'FlowSOM' | 'Leiden' | 'HDBSCAN'
    #   'gates':     list[str],
    #   'training_sample_ids': list[str],
    #   'n_samples': int,
    #   'n_events':  int,
    #   'channels':  list[str],
    #   'params':    dict,
    #   'timestamp': str,
    #   'labels':    dict[str, np.ndarray],    # per-sample label arrays
    #   'colors':    dict[int, str],           # cluster → hex
    #   'names':     dict[int, str],           # cluster → display label
    #   'n_clusters': int | None,
    # }

    # --- Groups and covariates ---
    sample_groups: dict[str, str] = field(default_factory=dict)
    # {sample_path: group_name | 'Unassigned'}: group_name is a
    # free-form, user-defined string (there is no longer a fixed 'A'/'B'
    # slot; the group's NAME is the value directly). 'Unassigned' is a
    # reserved sentinel and can never itself be added as a group name.
    group_names: list[str] = field(default_factory=lambda: ['A', 'B'])
    # Ordered list of currently-defined group names. Display order in the UI == this order.
    group_patterns: dict[str, str] = field(default_factory=dict)
    # {group_name: filename regex} for "Auto-assign by pattern"
    covariates: pd.DataFrame | None = None
    # rows = samples (rel-path index), columns = covariate names, all
    # values str. now actually populated by CSV import
    # (see _import_csv) — previously defined but never written to.

    # --- N-group testing ---
    testing_group_selection: list[str] = field(default_factory=list)
    # Which defined groups participate in Frequency/Counts/MFI testing
    # and Composition-by-group. Empty means "not yet
    # chosen" -- the UI defaults every currently-defined group to checked.
    contrast_mode: str = 'reference'
    # 'reference' -- one joint fit; every other selected group vs
    #   reference_group.
    # 'pairwise'  -- every unique pair among selected groups, each its own
    #   independent fit.
    reference_group: str = ''
    # Baseline group for 'reference' mode.
    paired: bool = False
    pairing_variable: str = ''
    # Column name in state.covariates used as a fixed-effect blocking term
    # when paired=True (e.g. donor ID).
    stats_fdr_scope: str = 'global'
    # 'global'         -- 'significant' uses one BH-FDR correction pooled
    #   across every comparison currently displayed (default).
    # 'per_comparison' -- 'significant' uses each comparison's own
    #   correction instead, without pooling across comparisons.
    stats_adjust_covariates: list[str] = field(default_factory=list)
    # state.covariates columns added to every test's design as adjustment
    # terms: continuous when listed in numeric_covariates, otherwise
    # categorical.
    numeric_covariates: list[str] = field(default_factory=list)
    # state.covariates columns entered as continuous numbers (age, CRP);
    # ticked under the column title in the Sample Group Assignment table.
    stats_use_treat: bool = True
    # True: p-values test |effect| > threshold (TREAT). False: test
    #   effect != 0 and use the threshold only as a filter.
    stats_counts_method: str = 'wald'
    # Counts test: 'wald' (per-cluster dispersion, z) or 'ql'
    #   (quasi-likelihood F with shrunk dispersions).
    stats_normalisation: str = 'total'
    # Library size of the Counts and Frequency tests: 'total' (events over
    #   all clusters) or 'robust' (trimmed rescaling that a few expanding
    #   clusters cannot move).
    stats_mfi_threshold: float = 0.03
    # |difference| threshold for MFIs, in transformed (Transforms-tab) units.
    stats_mfi_hierarchical: bool = True
    # Cluster-first MFI testing: screen clusters, then markers within them.
    stats_pval_threshold: float = 0.05
    stats_fc_threshold: float = 0.5
    # Significance level and |log2 fold change| threshold of the last
    # Run Statistics call, so restored plots are drawn with them.
    stats_data_key: tuple | None = None
    # Fingerprint of everything the current stats results were computed
    # from (run, tests, groups, pairing, covariates, marker roles and
    # transform parameters). Saved with the results; a restored result
    # is only treated as current while GroupsStatsTab._current_data_key()
    # still returns it.
    stats_comparisons: list[tuple[str, str]] = field(default_factory=list)
    # (baseline, other) pairs actually tested by the last Run Statistics
    # call, aligned 1:1 with the unique values of freq_results/mfi_results/
    # counts_results' 'comparison' column, in the same order. Lets the tab
    # recover which two groups a comparison belongs to without parsing the
    # display string.

    # --- Statistics ---
    freq_results: pd.DataFrame | None = None
    # moderated linear model output for cluster frequencies (log2 scale)
    freq_df: pd.DataFrame | None = None
    # raw (samples × clusters) frequency matrix, % of events
    counts_results: pd.DataFrame | None = None
    # negative-binomial GLM output for cluster raw counts
    # a parallel abundance test alongside freq_results, not a replacement.
    counts_df: pd.DataFrame | None = None
    # raw (samples x clusters) count matrix
    mfi_results: pd.DataFrame | None = None
    # moderated linear model output for per-cluster marker MFIs
    mfi_df: pd.DataFrame | None = None
    # raw (samples x cluster·channel) MFI matrix, each channel on its
    # configured Transforms-tab scale (Logicle/biexponential/linear)
    mfi_sample_df: pd.DataFrame | None = None
    # raw (samples x channel) MFI matrix (same per-channel Transforms-tab
    # scale), whole-sample aggregate — what the MFI Heatmap actually draws
    # (no cluster breakdown, no significance filter). mfi_df/mfi_results
    # stay cluster-level, for the MFI Volcano.

    # Composition Barplot -- independent of Run Statistics, so persisted
    # separately so it survives reopen the same way freq/mfi results do.
    composition_df: pd.DataFrame | None = None
    composition_as_pct: bool = True
    composition_group_var: str = 'sample'
    composition_run_label: str = ''
    composition_run_id: str = ''
    # run_id of the clustering_runs entry composition_df was computed
    # against -- authoritative match key, same idea as stats_run_id above
    # (composition_run_label alone can't detect a deleted run, since
    # labels are just display text).
    composition_names: dict[int, str] = field(default_factory=dict)
    composition_colors: dict[int, str] = field(default_factory=dict)
    # cluster colours FROZEN from the run composition_df was computed
    # against, at compute time -- see _show_composition_barplot. Fixes a
    # bug where _make_composition_figure re-resolved colours from
    # whichever run happens to be selected in the combo at redraw time.
    marker_roles: dict[str, str] = field(default_factory=dict)
    # {channel_name: 'type' | 'state'} — diffcyt-style split.
    # 'type' channels are excluded from MFI significance testing, to avoid
    # the same channel driving both the cluster assignment and its own
    # significance call. A channel with no entry yet defaults to 'state'
    # (included) — there is no automatic categorisation, see
    # GroupsStatsTab._populate_marker_roles_list(). User-overridable (some
    # markers, e.g. HLA-DR, genuinely serve double duty). Checkbox polarity
    # in the UI: ticked = 'state'/included, unticked = 'type'/excluded.
    group_colors: dict[str, str] = field(default_factory=dict)
    # {group_name: hex colour}. Assigned from the colorcet
    # glasbey palette the first time a group is drawn (see
    # GroupsStatsTab._group_colour); user-overridable via the 'Colour'
    # column on the Comparison Groups table. Used by the Sample PCA plot
    # (and available for any future all-groups plot).
    pca_use_freq: bool = True
    pca_use_counts: bool = False
    pca_use_mfi: bool = False
    # which of state.freq_df/counts_df/mfi_df feed the Sample
    # PCA plot. Independent of the "Test:" checkboxes above (those gate
    # what Run Statistics computes in the first place; a source can only
    # be used here if it was also computed there).
    pca_show_loadings: bool = True
    pca_n_loadings: int = 10
    pca_point_size: float = 60.0
    pca_arrow_lw: float = 1.2
    pca_arrow_color: str = '#ADD8E6'
    pca_axis_fontsize: int = 9
    pca_show_grid: bool = True
    pca_label_loadings: bool = True
    pca_label_points: bool = False
    pca_scores_df: pd.DataFrame | None = None
    pca_loadings_df: pd.DataFrame | None = None
    pca_explained_variance: tuple = (0.0, 0.0)
    pca_run_label: str = ''
    pca_groups: list = field(default_factory=list)
    pca_sources: list = field(default_factory=list)
    predictive_result: dict | None = None
    # drc_predictive.run_predictive() output for the last Predictive
    # Analysis run (session only), plus 'run_label'.
    # groups/sources -- the last two pieces _make_pca_figure needs that
    # weren't already cached above -- so a fully-persisted PCA result can
    # be rebuilt into the dict shape it expects without recomputing (see
    # _pca_result_from_state).
    stats_all_rel: list = field(default_factory=list)
    # sample rel-paths in test-matrix row order
    stats_group_vec: list = field(default_factory=list)
    # 'A'/'B' per row, aligned to stats_all_rel
    stats_run_label: str = ''
    # display label of the clustering run used for the current stats
    # results — NOT the matching key (see stats_run_id); kept only so
    # status messages/tooltips have something human-readable to show.
    stats_run_id: str = ''
    # run_id of the clustering_runs entry used for the current stats
    # results — the authoritative match key for combo restore / change
    # detection, since labels are user-editable and not guaranteed unique.

    # --- Workspace ---
    plot_configs: list[dict] = field(default_factory=list)
    # One dict per PlotCard on the workspace canvas
    display_clipboard: dict | None = None

    # --- Data pipeline cache ---
    gated_data_cache: drc_pipeline.GatedCache = field(default_factory=drc_pipeline.GatedCache)
    # Loaded samples (drc_pipeline.load_unmixed_gated): float32 values of
    # the channels in use and each event's FCS-file row, keyed by file,
    # gates, unmixing and the sample's gating. Capped at a quarter of the
    # machine's memory, least recently used sample dropped first.
    # Holds a copied PlotCard display config for magic-wand paste
    workspace_pages: list[dict] = field(default_factory=list)
    # Workspace pages [{'preset', 'w_mm', 'h_mm'}]; empty = one A4 page
    workspace_grid_mm: float = drc_workspace_canvas.DEFAULT_GRID_MM
    workspace_snap: bool = True
    workspace_show_grid: bool = True
    workspace_texts: list[dict] = field(default_factory=list)
    workspace_figures: list[dict] = field(default_factory=list)
    # Workspace figures from other tabs: where they come from (tab, key,
    # label), size, aspect lock, uid, layout, group, z -- not the figure
    # Workspace text boxes and panel letters (spec, uid, layout, group, z)
    plot_theme: str = 'auto'
    # 'auto' | 'light' | 'dark' — Workspace toggle. 'auto' mirrors
    # the live app palette (the previous, only, behaviour); 'light'/'dark'
    # force every plot in this plugin regardless of the app's own setting.

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def n_per_group(self, sample_keys: set[str] | None = None) -> dict[str, int]:
        """Return {group_name: assigned_sample_count} for every defined
        group. Pass sample_keys (e.g. GroupsStatsTab._table_sample_paths())
        to restrict the count to a specific run's samples; omitting it
        counts every sample_groups entry ever tracked, which can include
        other runs and stale/removed samples."""
        counts = {name: 0 for name in self.group_names}
        items = (
            self.sample_groups.items() if sample_keys is None
            else ((sp, g) for sp, g in self.sample_groups.items() if sp in sample_keys)
        )
        for _, g in items:
            if g in counts:
                counts[g] += 1
        return counts

    def n_group_stats_runnable(self, sample_keys: set[str] | None = None) -> bool:
        """
        True if at least 2 of the checked 'Groups to Test'
        each have >= 3 assigned samples. Governs Run Statistics/
        Composition-by-group. Pass sample_keys to
        restrict the count to a specific run's samples (see n_per_group);
        omitting it counts across the whole experiment.
        """
        selection = self.testing_group_selection or self.group_names
        counts = self.n_per_group(sample_keys)
        qualifying = sum(1 for name in selection if counts.get(name, 0) >= 3)
        return qualifying >= 2

    def cluster_label(self, cluster_id: int) -> str:
        """Return display name for a cluster id, falling back to 'Cluster N'."""
        return _cluster_display_name(self.cluster_names, cluster_id)

    def initialise_sample_groups(self, sample_paths: list[str]):
        """
        Add 'Unassigned' entries for any sample not yet in sample_groups.
        Remove entries for samples no longer in the experiment.
        Safe to call repeatedly.
        """
        current = set(sample_paths)
        removed = [p for p in self.sample_groups if p not in current]
        if removed and len(removed) == len(self.sample_groups):
            # Wiping EVERY existing entry in one call is the failure mode
            # we're hunting -- legitimate sample-list shrinkage should be
            # partial, not total, except on a genuinely empty experiment.
            _log.warning(
                "initialise_sample_groups: about to delete ALL %d existing "
                "sample_groups entries (incoming sample_paths had %d entries)",
                len(removed), len(sample_paths),
            )
        for path in current:
            if path not in self.sample_groups:
                self.sample_groups[path] = 'Unassigned'
        for path in list(self.sample_groups.keys()):
            if path not in current:
                del self.sample_groups[path]


# ---------------------------------------------------------------------------
# 5.  Inner tab widgets
#     Each is a QWidget sub-class defined before PluginWidget.
#     They receive (state, bus, controller) and are parented to PluginWidget.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 5a.  Run management table — used by ConfigTab only.
# ---------------------------------------------------------------------------

class _SortableItem(QTableWidgetItem):
    """
    QTableWidgetItem whose sort order is driven by an explicit key
    (Qt.UserRole) rather than its displayed text.  Needed for the
    Samples/Events columns (numeric, but displayed with thousands
    separators) and Timestamp (displayed human-readably, sorted by the
    raw ISO string) — both cases where display text and sort order
    diverge, so relying on QTableWidgetItem's own EditRole-type
    comparison isn't safe.
    """

    def set_sort_key(self, key):
        self.setData(Qt.UserRole, key)

    def __lt__(self, other):
        if isinstance(other, _SortableItem):
            a, b = self.data(Qt.UserRole), other.data(Qt.UserRole)
            if a is not None and b is not None:
                try:
                    return a < b
                except TypeError:
                    pass
        return super().__lt__(other)


class RunDetailDialog(QDialog):
    """
    Pop-out 'View' dialog for one archived run — full gate list,
    sample list with per-sample event counts, channel list, and the
    algorithm's hyperparameter dict, plus a 'Save config as CSV' export.

    Hydrates the run's payload on open (via drc_run_archive.hydrate_run)
    purely to compute per-sample event counts from the embeddings/labels
    dict — these are never stored as a separate manifest field since
    they're already implicit in that payload.  A no-op if already
    hydrated from an earlier View or run-selector use this session.
    """

    def __init__(self, controller, entry: dict, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.entry = drc_run_archive.hydrate_run(controller, entry)
        self.setWindowTitle(f"Run details — {self.entry.get('label', '')}")
        self.resize(560, 480)
        self._build_ui()

    def _per_sample_counts(self) -> dict[str, int]:
        kind = self.entry.get('kind')
        payload = (self.entry.get('embeddings') if kind == 'dr'
                   else self.entry.get('labels')) or {}
        counts = {rel: len(arr) for rel, arr in payload.items()}
        # A training sample that contributed no events (or whose payload
        # is only partially intact) still gets listed, at 0, rather than
        # silently dropped from the view.
        for rel in self.entry.get('training_sample_ids', []):
            counts.setdefault(rel, 0)
        return counts

    def _build_ui(self):
        from PySide6.QtGui import QFontDatabase

        layout = QVBoxLayout(self)

        text = QTextEdit()
        text.setReadOnly(True)
        text.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        text.setPlainText(self._format_config_text())
        layout.addWidget(text, stretch=1)

        btn_row = QHBoxLayout()
        save_csv_btn = QPushButton("Save config as CSV")
        save_csv_btn.clicked.connect(self._save_csv)
        btn_row.addWidget(save_csv_btn)
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _format_config_text(self) -> str:
        e = self.entry
        counts = self._per_sample_counts()
        lines = [
            f"Label:      {e.get('label', '')}",
            f"Kind:       {'DR' if e.get('kind') == 'dr' else 'Clustering'}",
            f"Algorithm:  {e.get('algorithm', '')}",
            f"Timestamp:  {e.get('timestamp', '')}",
        ]
        if e.get('kind') == 'clustering':
            lines.append(f"Clusters:   {e.get('n_clusters', '')}")
        lines.append("")
        gates = e.get('gates', [])
        lines.append(f"Gate(s) ({len(gates)}):")
        lines.extend(f"  - {g}" for g in gates)
        lines.append("")
        channels = e.get('channels', [])
        lines.append(f"Channels ({len(channels)}):")
        lines.extend(f"  - {ch}" for ch in channels)
        lines.append("")
        lines.append(f"Samples ({len(counts)}), {sum(counts.values()):,} events total:")
        lines.extend(f"  {counts[rel]:>10,}   {rel}" for rel in sorted(counts))
        lines.append("")
        lines.append("Hyperparameters:")
        lines.extend(f"  {k}: {v}" for k, v in e.get('params', {}).items())
        return "\n".join(lines)

    def _save_csv(self):
        e = self.entry
        default_name = f"{e.get('label', 'run')}_config.csv".replace(' ', '_')
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Run Config CSV", default_name, "CSV files (*.csv)"
        )
        if not path:
            return
        path = _with_extension(path, '.csv')
        counts = self._per_sample_counts()
        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['sample', 'n_events'])
                for rel in sorted(counts):
                    writer.writerow([rel, counts[rel]])
                writer.writerow([])
                writer.writerow(['key', 'value'])
                writer.writerow(['label', e.get('label', '')])
                writer.writerow(['kind', e.get('kind', '')])
                writer.writerow(['algorithm', e.get('algorithm', '')])
                writer.writerow(['timestamp', e.get('timestamp', '')])
                writer.writerow(['gates', '; '.join(e.get('gates', []))])
                writer.writerow(['channels', '; '.join(e.get('channels', []))])
                if e.get('kind') == 'clustering':
                    writer.writerow(['n_clusters', e.get('n_clusters', '')])
                for k, v in e.get('params', {}).items():
                    writer.writerow([f"param:{k}", v])
            QMessageBox.information(self, "Saved", f"Saved to {path}")
        except Exception as ex:
            QMessageBox.warning(self, "Save Error", str(ex))


class RunManagementTable(CopyableTableWidget):
    """
    Run management table.  Lives at the bottom of ConfigTab.

    Columns: Label (editable, double-click) · Kind · Algorithm · Gate(s)
    (count + tooltip) · Samples (count + tooltip) · Events · Channels
    (count + tooltip) · Timestamp.  Double-clicking the Label cell renames
    the run (writes back to the manifest); double-clicking any other cell
    opens the View/expand dialog.  Deletion is a separate button below the
    table (see ConfigTab._build_ui) since it's destructive and always
    needs confirmation regardless of how it's triggered.

    Unlike CopyableTableWidget's usual one-shot construction from a fixed
    list_of_dicts, this table's rows change over the plugin's lifetime, so
    __init__ bypasses that constructor's population loop entirely and
    calls QTableWidget.__init__ directly, with refresh() as the repeatable
    populate step.
    """

    runsChanged = Signal()  # emitted after a rename or delete, so
                            # PluginWidget can refresh every other
                            # run-selector elsewhere in the plugin.

    _HEADERS = ['Label', 'Kind', 'Algorithm', 'Gate(s)', 'Samples',
                'Events', 'Channels', 'Timestamp']

    # Text colour of a run whose events no longer match the current gating
    # and unmixing (see drc_run_archive.sync_status).
    _OUT_OF_SYNC_COLOUR = QColor('#ff9800')

    # Item-data role holding a label's own tooltip, so the sync note can be
    # appended to it (and replaced) without losing it.
    _BASE_TOOLTIP_ROLE = Qt.UserRole + 1

    def __init__(self, controller, state, parent=None):
        QTableWidget.__init__(self, 0, len(self._HEADERS), parent)
        self.controller = controller
        self.state = state

        self.setHorizontalHeaderLabels(self._HEADERS)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSortingEnabled(True)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)

        self.refresh()

    # ------------------------------------------------------------------
    # Populate
    # ------------------------------------------------------------------

    def refresh(self):
        """Rebuild every row from state.dr_runs + state.clustering_runs.
        Metadata only — never hydrates a run just to populate the table."""
        self.setSortingEnabled(False)  # avoid resort-during-populate churn
        entries = list(self.state.dr_runs) + list(self.state.clustering_runs)
        self.setRowCount(len(entries))
        fingerprints: dict = {}
        for row, entry in enumerate(entries):
            self._populate_row(row, entry)
            self._apply_sync_style(row, entry, fingerprints)
        self.setSortingEnabled(True)
        self.resizeColumnsToContents()
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)

    def _populate_row(self, row: int, entry: dict):
        run_id = entry.get('run_id')
        label_text = entry.get('label', '')

        label_item = QTableWidgetItem(label_text)
        label_item.setData(Qt.UserRole, run_id)
        if not drc_run_archive.run_payload_exists(self.controller, run_id):
            label_item.setText(f"⚠ {label_text}")
            missing_note = ("Payload file missing from disk — this run's data can no "
                            "longer be loaded.")
            label_item.setToolTip(missing_note)
            label_item.setData(self._BASE_TOOLTIP_ROLE, missing_note)
        self.setItem(row, 0, label_item)

        self.setItem(row, 1, QTableWidgetItem(
            'DR' if entry.get('kind') == 'dr' else 'Clustering'
        ))
        self.setItem(row, 2, QTableWidgetItem(entry.get('algorithm') or ''))

        gates = entry.get('gates', [])
        gate_item = QTableWidgetItem(
            gates[0] if len(gates) == 1 else f"{len(gates)} gates"
        )
        gate_item.setToolTip(', '.join(gates))
        self.setItem(row, 3, gate_item)

        samples = entry.get('training_sample_ids', [])
        n_samples = entry.get('n_samples', len(samples))
        samples_item = _SortableItem(str(n_samples))
        samples_item.set_sort_key(n_samples)
        samples_item.setToolTip('\n'.join(Path(s).name for s in samples))
        self.setItem(row, 4, samples_item)

        n_events = entry.get('n_events', 0)
        events_item = _SortableItem(f"{n_events:,}")
        events_item.set_sort_key(n_events)
        self.setItem(row, 5, events_item)

        channels = entry.get('channels', [])
        ch_item = QTableWidgetItem(str(len(channels)))
        ch_item.setToolTip(', '.join(channels))
        self.setItem(row, 6, ch_item)

        ts_raw = entry.get('timestamp', '') or ''
        ts_item = _SortableItem(ts_raw.replace('T', '  '))
        ts_item.set_sort_key(ts_raw)  # ISO 8601 — lexicographic == chronological
        self.setItem(row, 7, ts_item)

    # ------------------------------------------------------------------
    # Sync with the current gating / unmixing
    # ------------------------------------------------------------------

    def _current_fingerprint(self, entry: dict, cache: dict):
        """(drc_pipeline.sync_fingerprint(), drc_pipeline.split_fingerprints())
        for the run's own gates and samples, computed once per distinct
        (gates, samples) pair per pass."""
        key = (tuple(entry.get('gates', [])), tuple(entry.get('training_sample_ids', [])))
        if key not in cache:
            cache[key] = (drc_pipeline.sync_fingerprint(self.controller, *key),
                          drc_pipeline.split_fingerprints(self.controller, *key))
        return cache[key]

    def _apply_sync_style(self, row: int, entry: dict, cache: dict):
        """Colour the row's text orange when the run is out of sync with the
        current gating and unmixing, and say why in the label's tooltip."""
        current, split = self._current_fingerprint(entry, cache)
        status = drc_run_archive.sync_status(entry, current)
        out_of_sync = status == drc_run_archive.SYNC_OUT_OF_SYNC
        keyed = drc_run_align.has_event_keys(entry)
        label_item = self.item(row, 0)
        for col in range(self.columnCount()):
            item = self.item(row, col)
            if item is not None:
                item.setData(Qt.ForegroundRole,
                             self._OUT_OF_SYNC_COLOUR if out_of_sync else None)
        if label_item is None:
            return
        base = label_item.data(self._BASE_TOOLTIP_ROLE) or ''
        if out_of_sync and keyed:
            detail = drc_run_archive.sync_detail(entry, split)
            changed = [part for part in ('gating', 'unmixing')
                       if detail[part] == drc_run_archive.SYNC_CHANGED]
            note = (f"Changed since this run was made: {' and '.join(changed) or 'gating or unmixing'}. "
                    "The run records which events it holds, so it is matched to the "
                    "current gating event by event: cells that left its gates drop "
                    "out, and new cells are labelled from its SOM codebook or "
                    "cluster centroids where it has them (otherwise left unlabelled).")
            if 'unmixing' in changed:
                note += (" Its embedding or clusters were computed from the earlier "
                         "unmixed values; re-run it to use the current ones.")
        elif out_of_sync:
            note = ("Out of sync, and made before event keys were recorded: the "
                    "unmixing, AF profile assignment or gating has changed since, so "
                    "its events can no longer be matched to the current data. It can "
                    "still be viewed on its own; use \u2018Use These Settings\u2019 to "
                    "re-run it.")
        elif status == drc_run_archive.SYNC_UNKNOWN:
            note = ("Sync cannot be verified: this run was made before sync tracking "
                    "was added, or the current state could not be fingerprinted. It "
                    "is used on its own (its stored embedding, labels and marker values) "
                    "and not matched to events loaded now.")
        else:
            note = ''
        label_item.setToolTip('\n\n'.join(t for t in (base, note) if t))

    def refresh_sync_status(self):
        """Re-evaluate every row's sync colouring without rebuilding the table
        (keeps the selection). Called when the table is shown, since the
        unmixing or gating may have been edited elsewhere in the app."""
        fingerprints: dict = {}
        for row in range(self.rowCount()):
            label_item = self.item(row, 0)
            if label_item is None:
                continue
            entry = self._find_entry(label_item.data(Qt.UserRole))
            if entry is not None:
                self._apply_sync_style(row, entry, fingerprints)

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_sync_status()

    # ------------------------------------------------------------------
    # Row lookup
    # ------------------------------------------------------------------

    def _find_entry(self, run_id: str) -> dict | None:
        for entry in list(self.state.dr_runs) + list(self.state.clustering_runs):
            if entry.get('run_id') == run_id:
                return entry
        return None

    def selected_run_id(self) -> str | None:
        row = self.currentRow()
        if row < 0:
            return None
        item = self.item(row, 0)
        return item.data(Qt.UserRole) if item else None

    # ------------------------------------------------------------------
    # Rename / View (double-click)
    # ------------------------------------------------------------------

    def _on_item_double_clicked(self, item: QTableWidgetItem):
        label_item = self.item(item.row(), 0)
        run_id = label_item.data(Qt.UserRole)
        if run_id is None:
            return
        if item.column() == 0:
            self._rename_run(run_id, label_item)
        else:
            self._view_run(run_id)

    def _rename_run(self, run_id: str, label_item: QTableWidgetItem):
        from PySide6.QtWidgets import QInputDialog
        entry = self._find_entry(run_id)
        if entry is None:
            return
        new_label, ok = QInputDialog.getText(
            self, "Rename Run", "New label:", text=entry.get('label', '')
        )
        if not ok or not new_label.strip():
            return
        new_label = new_label.strip()
        entry['label'] = new_label          # same dict object state holds
        drc_run_archive.rename_run(self.controller, run_id, new_label)
        label_item.setText(new_label)
        self.runsChanged.emit()

    def _view_run(self, run_id: str):
        entry = self._find_entry(run_id)
        if entry is None:
            return
        if not drc_run_archive.run_payload_exists(self.controller, run_id):
            QMessageBox.warning(
                self, "Run Unavailable",
                f"The data file for \"{entry.get('label', '')}\" is missing "
                "from disk and can no longer be viewed."
            )
            return
        dlg = RunDetailDialog(self.controller, entry, self)
        dlg.exec()

    # ------------------------------------------------------------------
    # Delete (external button — see ConfigTab._build_ui)
    # ------------------------------------------------------------------

    def delete_selected(self) -> bool:
        """
        Confirm-then-delete the currently selected run: removes the
        manifest entry, deletes its payload bundle, and drops it from
        state.dr_runs/clustering_runs (all via drc_run_archive.delete_run),
        then refreshes this table and emits runsChanged.  Returns True if
        a run was actually deleted.
        """
        run_id = self.selected_run_id()
        if run_id is None:
            return False
        entry = self._find_entry(run_id)
        label = entry.get('label', run_id) if entry else run_id
        reply = QMessageBox.question(
            self, "Delete Run",
            f"Delete run \"{label}\"?  This removes its cached data from "
            "disk and cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return False
        drc_run_archive.delete_run(self.controller, self.state, run_id)
        self.refresh()
        self.runsChanged.emit()
        return True


def _make_scroll_guard(widget):
    """
    Return a ``wheelEvent`` handler that only scrolls when the widget
    already has keyboard focus.

    Assigning this to ``widget.wheelEvent`` prevents accidental value
    changes when the user scrolls the panel without having clicked a field.
    The widget must be clicked (or tabbed into) before scroll-wheel input
    takes effect.
    """
    def _handler(event):
        if widget.hasFocus():
            # Delegate to the class-level handler so normal behaviour applies
            type(widget).wheelEvent(widget, event)
        else:
            event.ignore()
    return _handler


class ConfigTab(QWidget):
    """
    Tab — Analysis Configuration
    --------------------------------
    Gate selector, channel checkboxes, training-sample picker,
    training-event count, DR algorithm + hyperparameter panels,
    clustering algorithm + hyperparameter panels.
    """

    _PAGE_DATA, _PAGE_DR, _PAGE_CL, _PAGE_RUNS = range(4)

    def __init__(self, state: PipelineState, bus, controller, parent=None):
        super().__init__(parent)
        self.state = state
        self.bus = bus
        self.controller = controller
        # {'dr' | 'cl': True while that kind of run is in progress}
        self._run_locked = {'dr': False, 'cl': False}
        self._page_titles: list[str] = []
        self._build_ui()

    def _add_page(self, page: QWidget, title: str) -> int:
        """Add *page* to the accordion inside its own scroll area."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(page)
        self._page_titles.append(title)
        return self.toolbox.addItem(scroll, title)

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        self.help_widget = HelpToggleWidget(
            text=drc_help_texts.configuration_tab_help_text
        )
        outer.addWidget(self.help_widget)

        # One section open at a time; each page scrolls on its own.
        self.toolbox = QToolBox()
        outer.addWidget(self.toolbox, stretch=1)

        # ------------------------------------------------------------------
        # Data Selection
        # ------------------------------------------------------------------
        self._data_page = QWidget()
        shared_layout = QVBoxLayout(self._data_page)

        # Gate selector — checkable multi-select tree.  Same tree
        # class as TransformTab's; this one is documented as the override —
        # checking different gates here updates the same shared
        # state.selected_gates list (kept in sync by
        # PluginWidget._on_gate_tree_changed).
        shared_layout.addWidget(QLabel("Gate(s):"))
        self.gate_tree = drc_gate_tree.GateTreeWidget()
        self.gate_tree.setToolTip(
            "Check one or more gates.  Their events are UNIONED before "
            "loading into the pipeline.  Checking a parent gate checks its "
            "whole subtree; you can still hand-pick individual leaves."
        )
        self.gate_tree.setMinimumHeight(140)
        self.gate_tree.setMaximumHeight(220)
        self.gate_tree.selectionChanged.connect(self._on_gate_tree_changed)
        shared_layout.addWidget(self.gate_tree)

        # Channel checkboxes — populated in refresh()
        shared_layout.addWidget(QLabel("Parameters to use for DR and clustering:"))
        self.channel_widget = QWidget()
        self.channel_layout = QGridLayout(self.channel_widget)
        self.channel_layout.setSpacing(4)
        self.channel_checkboxes: dict[str, QCheckBox] = {}
        shared_layout.addWidget(self.channel_widget)

        # Training-event count
        events_row = QHBoxLayout()
        events_row.addWidget(QLabel("Training events per sample:"))
        self.events_spinbox = QSpinBox()
        self.events_spinbox.setRange(1_000, 500_000)
        self.events_spinbox.setSingleStep(1_000)
        self.events_spinbox.setValue(self.state.n_training_events)
        self.events_spinbox.valueChanged.connect(
            lambda v: setattr(self.state, 'n_training_events', v)
        )
        events_row.addWidget(self.events_spinbox)
        events_row.addStretch()
        shared_layout.addLayout(events_row)

        # Sample types offered as training samples (off by default)
        self.include_unstained_chk = QCheckBox("Include unstained samples")
        self.include_single_stain_chk = QCheckBox("Include single-stained controls")
        self.include_beads_chk = QCheckBox("Include bead samples")
        include_row = QHBoxLayout()
        for chk in (self.include_unstained_chk, self.include_single_stain_chk,
                    self.include_beads_chk):
            chk.toggled.connect(self._on_training_pool_filter_changed)
            include_row.addWidget(chk)
        include_row.addStretch()
        shared_layout.addLayout(include_row)

        # Training sample picker
        shared_layout.addWidget(QLabel("Training samples:"))
        self.picker = OrderedMultiSamplePicker(title="Choose Training Samples")
        self.picker.changed.connect(self._on_training_samples_changed)
        shared_layout.addWidget(self.picker)
        shared_layout.addStretch()

        self._add_page(self._data_page, "Data Selection")

        # ------------------------------------------------------------------
        # Dimensionality Reduction
        # ------------------------------------------------------------------
        dr_page = QWidget()
        dr_page_layout = QVBoxLayout(dr_page)
        # Everything a DR run reads; locked while one is in progress.
        self._dr_controls = QWidget()
        dr_layout = QVBoxLayout(self._dr_controls)
        dr_layout.setContentsMargins(0, 0, 0, 0)
        dr_page_layout.addWidget(self._dr_controls)

        # Algorithm radio buttons
        algo_row = QHBoxLayout()
        algo_row.addWidget(QLabel("Algorithm:"))
        self._dr_algo_group = QButtonGroup(self)
        for algo in ('UMAP', 'tSNE', 'PaCMAP'):
            rb = QRadioButton(algo)
            if algo == 'UMAP':
                rb.setChecked(True)
            self._dr_algo_group.addButton(rb)
            algo_row.addWidget(rb)
        algo_row.addStretch()
        dr_layout.addLayout(algo_row)

        # Algorithm note (updates when selection changes)
        self._dr_algo_note = QLabel()
        self._dr_algo_note.setWordWrap(True)
        self._dr_algo_note.setStyleSheet("color: grey; font-style: italic;")
        dr_layout.addWidget(self._dr_algo_note)

        # ---- UMAP params ----
        self._umap_params = QWidget()
        umap_grid = QGridLayout(self._umap_params)
        umap_grid.setContentsMargins(0, 0, 0, 0)
        umap_grid.setSpacing(4)

        umap_grid.addWidget(QLabel("n_neighbors:"), 0, 0)
        self.umap_n_neighbors = QSpinBox()
        self.umap_n_neighbors.setRange(2, 500)
        self.umap_n_neighbors.setValue(15)
        self.umap_n_neighbors.setToolTip(
            "Number of nearest neighbours used to build the manifold graph.\n"
            "Larger = more global structure; smaller = more local detail.\n"
            "Default: 15"
        )
        umap_grid.addWidget(self.umap_n_neighbors, 0, 1)

        umap_grid.addWidget(QLabel("min_dist:"), 0, 2)
        self.umap_min_dist = QDoubleSpinBox()
        self.umap_min_dist.setRange(0.001, 0.99)
        self.umap_min_dist.setSingleStep(0.05)
        self.umap_min_dist.setDecimals(3)
        self.umap_min_dist.setValue(0.1)
        self.umap_min_dist.setToolTip(
            "Minimum distance between points in the 2-D embedding.\n"
            "Smaller = tighter clusters; larger = more spread-out layout.\n"
            "Default: 0.1"
        )
        umap_grid.addWidget(self.umap_min_dist, 0, 3)

        umap_grid.addWidget(QLabel("metric:"), 1, 0)
        self.umap_metric = QComboBox()
        self.umap_metric.addItems(['euclidean', 'cosine', 'manhattan', 'correlation'])
        self.umap_metric.setToolTip("Distance metric used for the neighbour graph.")
        umap_grid.addWidget(self.umap_metric, 1, 1)

        umap_grid.addWidget(QLabel("n_epochs:"), 1, 2)
        self.umap_n_epochs = QSpinBox()
        self.umap_n_epochs.setRange(50, 2000)
        self.umap_n_epochs.setValue(200)
        self.umap_n_epochs.setToolTip(
            "Number of optimisation epochs.\n"
            "More epochs = finer embedding; also slower (layout time\n"
            "grows in proportion).\n"
            "Default: 200 (umap-learn's default above 10,000 events)"
        )
        umap_grid.addWidget(self.umap_n_epochs, 1, 3)

        umap_grid.addWidget(QLabel("n_jobs:"), 2, 0)
        self.umap_n_jobs = QSpinBox()
        self.umap_n_jobs.setRange(-1, _os.cpu_count() or 1)
        self.umap_n_jobs.setValue(max(1, (_os.cpu_count() or 2) - 1))
        self.umap_n_jobs.setSpecialValueText('all cores')
        self.umap_n_jobs.setToolTip(
            "-1 = use all available cores.  Default: all cores but one.\n"
            "UMAP disables real multi-threading whenever a fixed random seed\n"
            "is set (needed for reproducible runs).\n"
            "Setting this above 1 drops the fixed seed for THIS run, so the\n"
            "embedding is not bit-for-bit reproducible run-to-run. Set to 1\n"
            "if exact reproducibility matters more than training speed."
        )
        umap_grid.addWidget(self.umap_n_jobs, 2, 1)

        dr_layout.addWidget(self._umap_params)

        # ---- tSNE params ----
        self._tsne_params = QWidget()
        tsne_grid = QGridLayout(self._tsne_params)
        tsne_grid.setContentsMargins(0, 0, 0, 0)
        tsne_grid.setSpacing(4)

        tsne_grid.addWidget(QLabel('Perplexity:'), 0, 0)
        self.tsne_perplexity = QSpinBox()
        self.tsne_perplexity.setRange(5, 500)
        self.tsne_perplexity.setValue(30)
        self.tsne_perplexity.setToolTip(
            'Balances attention between local and global structure.\n'
            'Rule of thumb: sqrt(N).  Default: 30'
        )
        tsne_grid.addWidget(self.tsne_perplexity, 0, 1)

        tsne_grid.addWidget(QLabel('n_iter:'), 0, 2)
        self.tsne_n_iter = QSpinBox()
        self.tsne_n_iter.setRange(250, 5000)
        self.tsne_n_iter.setValue(500)
        self.tsne_n_iter.setToolTip(
            'Optimisation iterations after the 250 early-exaggeration\n'
            'iterations.  Default: 500 (openTSNE default)')
        tsne_grid.addWidget(self.tsne_n_iter, 0, 3)

        tsne_grid.addWidget(QLabel('n_jobs:'), 1, 0)
        self.tsne_n_jobs = QSpinBox()
        self.tsne_n_jobs.setRange(-1, 64)
        self.tsne_n_jobs.setValue(max(1, (_os.cpu_count() or 2) - 1))
        self.tsne_n_jobs.setSpecialValueText('all cores')
        self.tsne_n_jobs.setToolTip('-1 = use all available cores.  Default: all cores but one')
        tsne_grid.addWidget(self.tsne_n_jobs, 1, 1)

        dr_layout.addWidget(self._tsne_params)
        self._tsne_params.setVisible(False)

        # ---- PaCMAP params ----
        self._pacmap_params = QWidget()
        pacmap_grid = QGridLayout(self._pacmap_params)
        pacmap_grid.setContentsMargins(0, 0, 0, 0)
        pacmap_grid.setSpacing(4)

        pacmap_grid.addWidget(QLabel("n_neighbors:"), 0, 0)
        self.pacmap_n_neighbors = QSpinBox()
        self.pacmap_n_neighbors.setRange(5, 500)
        self.pacmap_n_neighbors.setValue(10)
        self.pacmap_n_neighbors.setToolTip("Nearest neighbours for near pairs. Default: 10")
        pacmap_grid.addWidget(self.pacmap_n_neighbors, 0, 1)

        pacmap_grid.addWidget(QLabel("MN_ratio:"), 0, 2)
        self.pacmap_mn_ratio = QDoubleSpinBox()
        self.pacmap_mn_ratio.setRange(0.1, 5.0)
        self.pacmap_mn_ratio.setSingleStep(0.1)
        self.pacmap_mn_ratio.setDecimals(1)
        self.pacmap_mn_ratio.setValue(0.5)
        self.pacmap_mn_ratio.setToolTip(
            "Mid-near pair ratio relative to n_neighbors. Default: 0.5"
        )
        pacmap_grid.addWidget(self.pacmap_mn_ratio, 0, 3)

        pacmap_grid.addWidget(QLabel("FP_ratio:"), 1, 0)
        self.pacmap_fp_ratio = QDoubleSpinBox()
        self.pacmap_fp_ratio.setRange(0.5, 10.0)
        self.pacmap_fp_ratio.setSingleStep(0.5)
        self.pacmap_fp_ratio.setDecimals(1)
        self.pacmap_fp_ratio.setValue(2.0)
        self.pacmap_fp_ratio.setToolTip(
            "Further pair ratio relative to n_neighbors. Default: 2.0"
        )
        pacmap_grid.addWidget(self.pacmap_fp_ratio, 1, 1)

        dr_layout.addWidget(self._pacmap_params)
        self._pacmap_params.setVisible(False)

        # ---- Run button + status ----
        dr_run_row = QHBoxLayout()
        self.dr_run_btn = QPushButton("▶  Train DR Model")
        self.dr_run_btn.setFixedHeight(30)
        self.dr_run_btn.clicked.connect(self._on_run_dr_clicked)
        dr_run_row.addWidget(self.dr_run_btn)

        self.dr_apply_btn = QPushButton("Apply to All Samples")
        self.dr_apply_btn.setFixedHeight(30)
        self.dr_apply_btn.setToolTip(
            "Project every sample through the trained model to produce embeddings\n"
            "(including samples not in the training set)."
        )
        self.dr_apply_btn.clicked.connect(self._on_apply_dr_clicked)
        self.dr_apply_btn.setEnabled(False)
        dr_run_row.addWidget(self.dr_apply_btn)

        dr_run_row.addStretch()
        dr_layout.addLayout(dr_run_row)

        # Status, Cancel and progress sit outside _dr_controls so they stay
        # live during a run.
        dr_status_row = QHBoxLayout()
        self.dr_status_label = QLabel("No model trained.")
        self.dr_status_label.setStyleSheet("color: grey;")
        dr_status_row.addWidget(self.dr_status_label, stretch=1)

        self.dr_cancel_btn = QPushButton("✕  Cancel")
        self.dr_cancel_btn.setFixedHeight(30)
        self.dr_cancel_btn.setToolTip("Cancel the running DR job.")
        self.dr_cancel_btn.clicked.connect(self._on_cancel_dr_clicked)
        self.dr_cancel_btn.setEnabled(False)
        self.dr_cancel_btn.setStyleSheet("color: #c0392b;")
        dr_status_row.addWidget(self.dr_cancel_btn)
        dr_page_layout.addLayout(dr_status_row)

        self.dr_progress_bar = QProgressBar()
        self.dr_progress_bar.setRange(0, 100)
        self.dr_progress_bar.setTextVisible(True)
        self.dr_progress_bar.setFixedHeight(14)
        self.dr_progress_bar.setVisible(False)
        dr_page_layout.addWidget(self.dr_progress_bar)
        dr_page_layout.addStretch()

        self._add_page(dr_page, "Dimensionality Reduction")

        # Connect algo radio buttons → show/hide param panels + update note
        self._dr_algo_group.buttonClicked.connect(self._on_dr_algo_changed)
        self._on_dr_algo_changed()   # set initial state

        # ------------------------------------------------------------------
        # Clustering
        # ------------------------------------------------------------------
        cl_page = QWidget()
        cl_page_layout = QVBoxLayout(cl_page)
        # Everything a clustering run reads; locked while one is in progress.
        self._cl_controls = QWidget()
        cl_layout = QVBoxLayout(self._cl_controls)
        cl_layout.setContentsMargins(0, 0, 0, 0)
        cl_page_layout.addWidget(self._cl_controls)

        # Algorithm radio buttons
        cl_algo_row = QHBoxLayout()
        cl_algo_row.addWidget(QLabel("Algorithm:"))
        self._cl_algo_group = QButtonGroup(self)
        for algo in ('FlowSOM', 'Leiden', 'HDBSCAN'):
            rb = QRadioButton(algo)
            if algo == 'FlowSOM':
                rb.setChecked(True)
            self._cl_algo_group.addButton(rb)
            cl_algo_row.addWidget(rb)
        cl_algo_row.addStretch()
        cl_layout.addLayout(cl_algo_row)

        self._cl_algo_note = QLabel()
        self._cl_algo_note.setWordWrap(True)
        self._cl_algo_note.setStyleSheet("color: grey; font-style: italic;")
        cl_layout.addWidget(self._cl_algo_note)

        # ---- FlowSOM params ----
        self._flowsom_params = QWidget()
        fs_grid = QGridLayout(self._flowsom_params)
        fs_grid.setContentsMargins(0, 0, 0, 0)
        fs_grid.setSpacing(4)

        fs_grid.addWidget(QLabel("Grid size (x):"), 0, 0)
        self.flowsom_xdim = QSpinBox()
        self.flowsom_xdim.setRange(3, 30)
        self.flowsom_xdim.setValue(10)
        self.flowsom_xdim.setToolTip("SOM grid width. Total nodes = x × y. Default: 10")
        fs_grid.addWidget(self.flowsom_xdim, 0, 1)

        fs_grid.addWidget(QLabel("Grid size (y):"), 0, 2)
        self.flowsom_ydim = QSpinBox()
        self.flowsom_ydim.setRange(3, 30)
        self.flowsom_ydim.setValue(10)
        self.flowsom_ydim.setToolTip("SOM grid height. Total nodes = x × y. Default: 10")
        fs_grid.addWidget(self.flowsom_ydim, 0, 3)

        fs_grid.addWidget(QLabel("Metaclusters:"), 1, 0)
        self.flowsom_metaclusters = QSpinBox()
        self.flowsom_metaclusters.setRange(2, 200)
        self.flowsom_metaclusters.setValue(10)
        self.flowsom_metaclusters.setToolTip("Number of metaclusters.")
        fs_grid.addWidget(self.flowsom_metaclusters, 1, 1)

        fs_grid.addWidget(QLabel("Iterations:"), 2, 0)
        self.flowsom_n_iter = QSpinBox()
        self.flowsom_n_iter.setRange(1, 50)
        self.flowsom_n_iter.setValue(10)
        self.flowsom_n_iter.setToolTip("Training iterations for the SOM. Default: 10")
        fs_grid.addWidget(self.flowsom_n_iter, 2, 1)

        cl_layout.addWidget(self._flowsom_params)

        # ---- Leiden params ----
        self._leiden_params = QWidget()
        leiden_grid = QGridLayout(self._leiden_params)
        leiden_grid.setContentsMargins(0, 0, 0, 0)
        leiden_grid.setSpacing(4)

        leiden_grid.addWidget(QLabel("Resolution:"), 0, 0)
        self.leiden_resolution = QDoubleSpinBox()
        self.leiden_resolution.setRange(0.01, 10.0)
        self.leiden_resolution.setSingleStep(0.1)
        self.leiden_resolution.setDecimals(2)
        self.leiden_resolution.setValue(1.0)
        self.leiden_resolution.setToolTip(
            "Community-detection resolution.\n"
            "Higher = more clusters; lower = fewer. Default: 1.0"
        )
        leiden_grid.addWidget(self.leiden_resolution, 0, 1)

        leiden_grid.addWidget(QLabel("n_neighbors (kNN):"), 0, 2)
        self.leiden_n_neighbors = QSpinBox()
        self.leiden_n_neighbors.setRange(2, 500)
        self.leiden_n_neighbors.setValue(15)
        self.leiden_n_neighbors.setToolTip(
            "Neighbours per event in the kNN graph. Default: 15"
        )
        leiden_grid.addWidget(self.leiden_n_neighbors, 0, 3)

        cl_layout.addWidget(self._leiden_params)
        self._leiden_params.setVisible(False)

        # ---- HDBSCAN params ----
        self._hdbscan_params = QWidget()
        hdbscan_grid = QGridLayout(self._hdbscan_params)
        hdbscan_grid.setContentsMargins(0, 0, 0, 0)
        hdbscan_grid.setSpacing(4)

        hdbscan_grid.addWidget(QLabel("min_cluster_size:"), 0, 0)
        self.hdbscan_min_cluster_size = QSpinBox()
        self.hdbscan_min_cluster_size.setRange(2, 5000)
        self.hdbscan_min_cluster_size.setValue(25)
        self.hdbscan_min_cluster_size.setToolTip(
            "Smallest grouping of events that counts as its own cluster.\n"
            "Applied to UMAP embedding (if available) else raw feature space.\n"
            "Default: 25"
        )
        hdbscan_grid.addWidget(self.hdbscan_min_cluster_size, 0, 1)

        hdbscan_grid.addWidget(QLabel("min_samples:"), 0, 2)
        self.hdbscan_min_samples = QSpinBox()
        self.hdbscan_min_samples.setRange(0, 500)
        self.hdbscan_min_samples.setValue(0)
        self.hdbscan_min_samples.setToolTip(
            "How conservative density estimation is.  Higher = more events\n"
            "labelled −1 (noise).  0 = use min_cluster_size (sklearn default)."
        )
        hdbscan_grid.addWidget(self.hdbscan_min_samples, 0, 3)

        hdbscan_grid.addWidget(QLabel("cluster_selection_epsilon:"), 1, 0)
        self.hdbscan_cluster_selection_epsilon = QDoubleSpinBox()
        self.hdbscan_cluster_selection_epsilon.setRange(0.0, 100.0)
        self.hdbscan_cluster_selection_epsilon.setSingleStep(0.05)
        self.hdbscan_cluster_selection_epsilon.setDecimals(3)
        self.hdbscan_cluster_selection_epsilon.setValue(0.0)
        self.hdbscan_cluster_selection_epsilon.setToolTip(
            "Merge sub-clusters closer than this distance into one.\n"
            "0.0 = pure hierarchical selection, no merging. Default: 0.0"
        )
        hdbscan_grid.addWidget(self.hdbscan_cluster_selection_epsilon, 1, 1)

        cl_layout.addWidget(self._hdbscan_params)
        self._hdbscan_params.setVisible(False)

        # ---- Clustering space selector ----
        space_row = QHBoxLayout()
        space_row.addWidget(QLabel('Cluster on:'))
        self._cl_space_group = QButtonGroup(self)
        self._rb_space_raw = QRadioButton('Original features')
        self._rb_space_raw.setChecked(True)
        self._rb_space_raw.setToolTip(
            'Cluster directly on the logicle-transformed channel values.'
        )
        self._rb_space_dr = QRadioButton('DR embedding:')
        self._rb_space_dr.setToolTip(
            'Cluster on a dimensionality-reduction embedding.'
        )
        self._cl_space_group.addButton(self._rb_space_raw)
        self._cl_space_group.addButton(self._rb_space_dr)
        space_row.addWidget(self._rb_space_raw)
        space_row.addWidget(self._rb_space_dr)
        self._cl_dr_combo = QComboBox()
        self._cl_dr_combo.setToolTip('Choose which DR embedding to cluster on.')
        self._cl_dr_combo.setEnabled(False)
        space_row.addWidget(self._cl_dr_combo)
        space_row.addStretch()
        cl_layout.addLayout(space_row)
        self._rb_space_dr.toggled.connect(
            lambda on: self._cl_dr_combo.setEnabled(on)
        )

        # ---- Assignment scope ----
        # Unchecked by default: every training sample's own
        # events get their EXACT label straight out of the fit itself
        # (Leiden/HDBSCAN) or the native SOM node mapping (FlowSOM) -- no
        # downsampling, no approximate assignment needed for them at all.
        self.cl_downsample_chk = QCheckBox("Downsample training data (faster, less accurate)")
        self.cl_downsample_chk.setChecked(False)
        self.cl_downsample_chk.setToolTip(
            "Unchecked (default): every gated event from every Training\n"
            "Sample is used to fit the model -- slower, but every training\n"
            "sample's own events get an exact label with no approximation.\n"
            "Checked: caps each training sample to 'Training events per\n"
            "sample' above (like the previous default) -- faster, but\n"
            "events beyond the cap are labelled via an approximate method\n"
            "(k-NN majority vote for Leiden, approximate_predict for\n"
            "HDBSCAN) instead of the fit itself."
        )
        cl_layout.addWidget(self.cl_downsample_chk)

        # Opt-in, not default (matches DR's own "Apply to All
        # Samples" being a manual second step, not automatic): every
        # sample in the experiment gets cluster labels, not just Training
        # Samples. Training Samples still only decides what fits the
        # model either way.
        self.cl_assign_all_chk = QCheckBox("Assign clusters to all samples")
        self.cl_assign_all_chk.setChecked(False)
        self.cl_assign_all_chk.setToolTip(
            "Unchecked (default): only Training Samples get cluster labels.\n"
            "Checked: every other sample in the experiment gets labels too\n"
            "(via the same approximate method used for any downsampled\n"
            "events), not just the ones picked as Training Samples."
        )
        cl_layout.addWidget(self.cl_assign_all_chk)

        # ---- Run button + status ----
        cl_run_row = QHBoxLayout()
        self.cl_run_btn = QPushButton("▶  Run Clustering")
        self.cl_run_btn.setFixedHeight(30)
        self.cl_run_btn.clicked.connect(self._on_run_clustering_clicked)
        cl_run_row.addWidget(self.cl_run_btn)
        cl_run_row.addStretch()
        cl_layout.addLayout(cl_run_row)

        # Status, Cancel and progress sit outside _cl_controls so they stay
        # live while a run locks the controls.
        cl_status_row = QHBoxLayout()
        self.cl_status_label = QLabel("No clustering run.")
        self.cl_status_label.setStyleSheet("color: grey;")
        cl_status_row.addWidget(self.cl_status_label, stretch=1)

        self.cl_cancel_btn = QPushButton("✕  Cancel")
        self.cl_cancel_btn.setFixedHeight(30)
        self.cl_cancel_btn.setToolTip("Cancel the running clustering job.")
        self.cl_cancel_btn.clicked.connect(self._on_cancel_clustering_clicked)
        self.cl_cancel_btn.setEnabled(False)
        self.cl_cancel_btn.setStyleSheet("color: #c0392b;")
        cl_status_row.addWidget(self.cl_cancel_btn)
        cl_page_layout.addLayout(cl_status_row)

        self.cl_progress_bar = QProgressBar()
        self.cl_progress_bar.setRange(0, 0)   # indeterminate by default
        self.cl_progress_bar.setTextVisible(False)
        self.cl_progress_bar.setFixedHeight(14)
        self.cl_progress_bar.setVisible(False)
        cl_page_layout.addWidget(self.cl_progress_bar)
        cl_page_layout.addStretch()

        self._add_page(cl_page, "Clustering")

        # ---- Archived Runs ----
        runs_page = QWidget()
        runs_box_layout = QVBoxLayout(runs_page)

        runs_hint = _HintLabel(
            "Double-click a run's label to rename it, or any other cell to "
            "view its full configuration. Renaming or deleting here updates "
            "every run selector elsewhere in the plugin immediately."
        )
        runs_hint.setWordWrap(True)
        runs_box_layout.addWidget(runs_hint)

        self.run_table = RunManagementTable(self.controller, self.state)
        runs_box_layout.addWidget(self.run_table, stretch=1)

        runs_btn_row = QHBoxLayout()
        runs_btn_row.addStretch()
        self.use_run_settings_btn = QPushButton("Use These Settings")
        self.use_run_settings_btn.setEnabled(False)
        self.use_run_settings_btn.setToolTip(
            "Load the selected run's gates, training samples, channels, algorithm "
            "and parameters into the Configuration pages, ready to run again.")
        self.use_run_settings_btn.clicked.connect(self._on_use_run_settings_clicked)
        runs_btn_row.addWidget(self.use_run_settings_btn)
        self.delete_run_btn = QPushButton("Delete Selected Run")
        self.delete_run_btn.setEnabled(False)
        self.delete_run_btn.setToolTip("Select a run above to delete it.")
        self.delete_run_btn.clicked.connect(self._on_delete_run_clicked)
        runs_btn_row.addWidget(self.delete_run_btn)
        runs_box_layout.addLayout(runs_btn_row)

        self.run_table.itemSelectionChanged.connect(self._on_run_table_selection_changed)

        self._add_page(runs_page, "Archived Runs")

        # Connect clustering algo buttons
        self._cl_algo_group.buttonClicked.connect(self._on_cl_algo_changed)
        self._on_cl_algo_changed()   # set initial state

        # Block scroll-wheel on all spinboxes and comboboxes so that
        # accidental scrolling over a field doesn't silently change values.
        # The widget must have explicit keyboard focus before wheel events act.
        # PySide6 findChildren() accepts only a single type, not a tuple.
        _scroll_guarded = (
            self.findChildren(QSpinBox)
            + self.findChildren(QDoubleSpinBox)
            + self.findChildren(QComboBox)
        )
        for widget in _scroll_guarded:
            widget.setFocusPolicy(Qt.StrongFocus)
            widget.wheelEvent = _make_scroll_guard(widget)

    def set_run_lock(self, kind: str, running: bool):
        """
        Lock the controls a run depends on while it is in progress.

        *kind* is 'dr' or 'cl'. That section's controls are disabled (its
        status line, progress bar and Cancel stay live), and Data Selection
        is disabled while either kind is running, because a finished run is
        archived with the gates, channels and training samples current at
        that moment.
        """
        self._run_locked[kind] = running
        controls, page = ((self._dr_controls, self._PAGE_DR) if kind == 'dr'
                          else (self._cl_controls, self._PAGE_CL))
        controls.setEnabled(not running)
        title = self._page_titles[page]
        self.toolbox.setItemText(page, f"{title}  —  ⏳ running" if running else title)
        locked = self.is_run_in_progress()
        self._data_page.setEnabled(not locked)
        self.toolbox.setItemToolTip(
            self._PAGE_DATA, "Locked while a run is in progress." if locked else ""
        )

    def is_run_in_progress(self) -> bool:
        return any(self._run_locked.values())

    def _on_training_samples_changed(self, ordered_list: list):
        """Keep state.training_sample_ids in sync with the picker."""
        self.state.training_sample_ids = list(ordered_list)

    def training_pool_paths(self) -> list[str]:
        """Samples offered for training, relative to the raw samples subdirectory.
        A sample of an excluded type is offered only if one of its types is ticked."""
        experiment = self.controller.experiment
        raw_subdir = experiment.settings['raw']['raw_samples_subdirectory']
        samples = experiment.samples
        include = {'unstained': self.include_unstained_chk.isChecked(),
                   'single_stain': self.include_single_stain_chk.isChecked(),
                   'beads': self.include_beads_chk.isChecked()}
        pool = []
        for s in offered_samples(samples, include):
            try:
                pool.append(str(Path(s).relative_to(raw_subdir)))
            except ValueError:
                pool.append(s)
        return pool

    def _on_training_pool_filter_changed(self, *_):
        """Rebuild the picker pool, keeping selected samples that are still offered."""
        pool = self.training_pool_paths()
        selected = [p for p in self.picker.get_ordered_list() if p in pool]
        self.picker.set_items(pool, selected=selected)

    def _on_run_table_selection_changed(self):
        selected = self.run_table.selected_run_id() is not None
        self.delete_run_btn.setEnabled(selected)
        self.use_run_settings_btn.setEnabled(selected)

    def _on_use_run_settings_clicked(self):
        plugin = self._plugin_widget()
        entry = self.run_table._find_entry(self.run_table.selected_run_id())
        if plugin is not None and entry is not None:
            plugin.load_run_settings(entry)

    def _on_delete_run_clicked(self):
        self.run_table.delete_selected()
        # delete_selected() already refreshed this table and emitted
        # runsChanged (PluginWidget refreshes every other run selector off
        # that signal) — nothing further to do here.

    def _on_gate_tree_changed(self, gates: list[str]):
        """Checked-set changed in this tab's gate tree."""
        self.state.selected_gates = list(gates)

    def _update_selected_channels(self, *_):
        """Sync state.selected_channels from the current checkbox state."""
        self.state.selected_channels = [
            ch for ch, cb in self.channel_checkboxes.items() if cb.isChecked()
        ]

    # ------------------------------------------------------------------
    # DR algorithm selection
    # ------------------------------------------------------------------

    def _selected_dr_algo(self) -> str:
        btn = self._dr_algo_group.checkedButton()
        return btn.text() if btn else 'UMAP'

    def _on_dr_algo_changed(self, *_):
        algo = self._selected_dr_algo()
        self._umap_params.setVisible(algo == 'UMAP')
        self._tsne_params.setVisible(algo == 'tSNE')
        self._pacmap_params.setVisible(algo == 'PaCMAP')

        notes = {
            'UMAP': (
                "UMAP (umap-learn): preserves both local and global structure.\n"
                "kNN graph stored for Leiden reuse."
            ),
            'tSNE': (
                "tSNE (openTSNE): probabilistic neighbour embedding.\n"
                "Good for visualising tight clusters; slower than UMAP."
            ),
            'PaCMAP': (
                "PaCMAP: pair-centric MA embedding; balances local and global "
                "structure without perplexity tuning."
            ),
        }
        self._dr_algo_note.setText(notes.get(algo, ''))

        # Reflect any existing training status
        self._refresh_dr_status()

    def _refresh_dr_status(self):
        """Update DR status label from PipelineState."""
        algo = self._selected_dr_algo()
        status = self.state.dr_status.get(algo, 'idle')
        if status == 'running' and not self._run_locked['dr']:
            # A run exists only while its lock is held; anything else is a
            # stale status (e.g. restored from a session that ended mid-run).
            status = self.state.dr_status[algo] = 'idle'
        ts     = self.state.dr_timestamps.get(algo, '')
        running = (status == 'running')
        # Fitted DR models are not saved, so after a restart only the
        # embeddings of a 'done' algorithm are available.
        model_in_memory = self.state.trained_reducers.get(algo) is not None
        if self._run_locked['dr']:
            # Covers 'Apply to All Samples' too, where status stays 'done'.
            running = True
            self.dr_status_label.setText("⏳ Running …")
            self.dr_status_label.setStyleSheet("color: orange;")
            self.dr_apply_btn.setEnabled(False)
        elif status == 'done':
            n_emb = len(self.state.embeddings.get(algo, {}))
            model_note = '' if model_in_memory else "  —  model not kept after restart"
            self.dr_status_label.setText(
                f"✓ Trained  ({ts})  —  embeddings: {n_emb} sample(s){model_note}"
            )
            self.dr_status_label.setStyleSheet("color: green;")
            self.dr_apply_btn.setEnabled(model_in_memory)
        elif status == 'error':
            self.dr_status_label.setText(f"✗ Error during last run  ({ts})")
            self.dr_status_label.setStyleSheet("color: red;")
            self.dr_apply_btn.setEnabled(False)
        elif status == 'running':
            self.dr_status_label.setText("⏳ Running …")
            self.dr_status_label.setStyleSheet("color: orange;")
            self.dr_apply_btn.setEnabled(False)
        else:
            self.dr_status_label.setText("No model trained.")
            self.dr_status_label.setStyleSheet("color: grey;")
            self.dr_apply_btn.setEnabled(False)
        self.dr_run_btn.setEnabled(not running)
        self.dr_cancel_btn.setEnabled(running)

        if status == 'done' and not model_in_memory:
            self.dr_apply_btn.setToolTip(
                f"The fitted {algo} model is only kept for the session it was\n"
                "trained in. Its embeddings were restored, but projecting more\n"
                f"samples needs the model: re-train {algo} to use this."
            )
        else:
            self.dr_apply_btn.setToolTip(
                "Project every sample through the trained model to produce embeddings\n"
                "(including samples not in the training set)."
            )

    def _on_run_dr_clicked(self):
        """Collect params and delegate to PluginWidget._run_dr()."""
        algo = self._selected_dr_algo()
        params = self._collect_dr_params(algo)
        plugin = self._plugin_widget()
        if plugin:
            plugin._run_dr(algo, params)

    def _on_apply_dr_clicked(self):
        """Apply trained model to all samples."""
        algo = self._selected_dr_algo()
        plugin = self._plugin_widget()
        if plugin:
            plugin._apply_dr_to_all_samples(algo)

    def _collect_dr_params(self, algo: str) -> dict:
        if algo == 'UMAP':
            return {
                'n_neighbors': self.umap_n_neighbors.value(),
                'min_dist':    self.umap_min_dist.value(),
                'metric':      self.umap_metric.currentText(),
                'n_epochs':    self.umap_n_epochs.value(),
                'n_jobs':      self.umap_n_jobs.value(),
            }
        elif algo == 'tSNE':
            return {
                'perplexity': self.tsne_perplexity.value(),
                'n_iter':     self.tsne_n_iter.value(),
                'n_jobs':     self.tsne_n_jobs.value(),
            }
        elif algo == 'PaCMAP':
            return {
                'n_neighbors': self.pacmap_n_neighbors.value(),
                'MN_ratio':    self.pacmap_mn_ratio.value(),
                'FP_ratio':    self.pacmap_fp_ratio.value(),
            }
        return {}

    # ------------------------------------------------------------------
    # Clustering algorithm selection
    # ------------------------------------------------------------------

    def _selected_cl_algo(self) -> str:
        btn = self._cl_algo_group.checkedButton()
        return btn.text() if btn else 'FlowSOM'

    def _on_cl_algo_changed(self, *_):
        algo = self._selected_cl_algo()
        self._flowsom_params.setVisible(algo == 'FlowSOM')
        self._leiden_params.setVisible(algo == 'Leiden')
        self._hdbscan_params.setVisible(algo == 'HDBSCAN')

        # HDBSCAN doesn't behave well directly on the full
        # multichannel feature space (confirmed: hundreds of
        # near-meaningless clusters on a real run). Force DR space and
        # disable the raw-feature-space option while it's selected.
        # FlowSOM trains its SOM on the channel values only, so the DR
        # option is disabled for it.  Leiden is fine in either space.
        is_hdbscan = (algo == 'HDBSCAN')
        is_flowsom = (algo == 'FlowSOM')
        self._rb_space_raw.setEnabled(not is_hdbscan)
        self._rb_space_dr.setEnabled(not is_flowsom)
        if is_hdbscan and not self._rb_space_dr.isChecked():
            self._rb_space_dr.setChecked(True)
        if is_flowsom and not self._rb_space_raw.isChecked():
            self._rb_space_raw.setChecked(True)
        self._rb_space_raw.setToolTip(
            "Not available for HDBSCAN -- see the note below."
            if is_hdbscan else
            "Cluster directly on the logicle-transformed channel values."
        )
        self._rb_space_dr.setToolTip(
            "Not available for FlowSOM: its SOM trains on the channel values."
            if is_flowsom else
            "Cluster on a dimensionality-reduction embedding."
        )

        notes = {
            'FlowSOM': (
                "FlowSOM: self-organising map → metaclustering, on the channel values."
            ),
            'Leiden': (
                "Leiden: graph community detection (graspologic-native)\n"
                "on its own kNN graph of the training events.\n"
                "Higher resolution = more fine-grained clusters."
            ),
            'HDBSCAN': (
                "HDBSCAN: density-based clustering, no fixed distance threshold.\n"
                "Restricted to a DR embedding -- high-dimensional density estimates\n"
                "degenerate badly directly on the full feature space (far too many\n"
                "small, near-meaningless clusters). Events labelled −1 (noise) are\n"
                "shown in grey."
            ),
        }
        self._cl_algo_note.setText(notes.get(algo, ''))
        self._refresh_cl_space_combo()
        self._refresh_cl_status()

    def _refresh_cl_status(self):
        algo = self.state.active_clustering_algorithm
        n_cl = self.state.n_clusters
        self.cl_cancel_btn.setEnabled(self._run_locked['cl'])
        if self._run_locked['cl']:
            self.cl_status_label.setText("⏳ Running …")
            self.cl_status_label.setStyleSheet("color: orange;")
        elif algo and n_cl is not None:
            self.cl_status_label.setText(
                f"✓ Last run: {algo}  —  {n_cl} cluster(s)"
            )
            self.cl_status_label.setStyleSheet("color: green;")
        else:
            self.cl_status_label.setText("No clustering run.")
            self.cl_status_label.setStyleSheet("color: grey;")

    # Default DR embedding for clustering: first usable one wins
    _CL_DR_PREFERENCE = ('PaCMAP', 'UMAP', 'tSNE')

    def _refresh_cl_space_combo(self):
        """
        Populate the DR embedding combo with the embeddings that can be
        clustered on, in _CL_DR_PREFERENCE order. The current choice is kept
        while it stays usable; otherwise the first (preferred) one is selected.
        """
        if not hasattr(self, '_cl_dr_combo'):
            return
        plugin = self._plugin_widget()
        if plugin is None:
            return
        available = [a for a in self._CL_DR_PREFERENCE
                     if plugin._dr_embedding_unusable(a) is None]
        current = self._cl_dr_combo.currentText()
        self._cl_dr_combo.blockSignals(True)
        self._cl_dr_combo.clear()
        self._cl_dr_combo.addItems(available)
        if current in available:
            self._cl_dr_combo.setCurrentText(current)
        self._cl_dr_combo.blockSignals(False)

    def clustering_space(self) -> tuple[str, str | None]:
        """Return ('raw', None) or ('dr', algo_name) based on UI selection."""
        if hasattr(self, '_rb_space_dr') and self._rb_space_dr.isChecked():
            return 'dr', self._cl_dr_combo.currentText() or None
        return 'raw', None

    def _on_run_clustering_clicked(self):
        algo = self._selected_cl_algo()
        params = self._collect_cl_params(algo)
        space, dr_algo = self.clustering_space()
        params['_space'] = space
        params['_dr_algo'] = dr_algo
        params['_assign_all_samples'] = self.cl_assign_all_chk.isChecked()
        params['_event_cap'] = (
            self.state.n_training_events if self.cl_downsample_chk.isChecked() else None
        )
        plugin = self._plugin_widget()
        if plugin:
            plugin._run_clustering(algo, params)

    def _collect_cl_params(self, algo: str) -> dict:
        if algo == 'FlowSOM':
            return {
                'xdim':          self.flowsom_xdim.value(),
                'ydim':          self.flowsom_ydim.value(),
                'n_metaclusters': self.flowsom_metaclusters.value(),
                'n_iter':        self.flowsom_n_iter.value(),
            }
        elif algo == 'Leiden':
            return {
                'resolution':   self.leiden_resolution.value(),
                'n_neighbors':  self.leiden_n_neighbors.value(),
            }
        elif algo == 'HDBSCAN':
            return {
                'min_cluster_size':          self.hdbscan_min_cluster_size.value(),
                'min_samples':               self.hdbscan_min_samples.value(),
                'cluster_selection_epsilon': self.hdbscan_cluster_selection_epsilon.value(),
            }
        return {}

    def _on_cancel_clustering_clicked(self):
        plugin = self._plugin_widget()
        if plugin:
            plugin._cancel_clustering()

    def _on_cancel_dr_clicked(self):
        plugin = self._plugin_widget()
        if plugin:
            plugin._cancel_dr()

    def _plugin_widget(self):
        """Walk the parent chain to find the PluginWidget."""
        w = self.parent()
        while w is not None:
            if isinstance(w, PluginWidget):
                return w
            w = w.parent()
        return None

    def refresh(self):
        """
        Repopulate gate combo and channel checkboxes from the current
        experiment.  Called when the tab becomes active.
        """
        if self.controller.experiment.process.get('unmixing_matrix') is None:
            return

        # Gate tree — always rebuilt so renamed/added gates appear.  The
        # checked set comes from state.selected_gates — the single shared
        # source of truth kept in sync with TransformTab's tree — so the
        # tree can never drift out of sync with the model.
        hierarchy = self.controller.unmixed_gating.get_gate_hierarchy(output='dict')
        self.gate_tree.set_hierarchy(hierarchy)
        self.gate_tree.set_checked_names(self.state.selected_gates)

        # Channel checkboxes — rebuild from experiment settings
        channels = self.controller.experiment.settings.get(
            'unmixed', {}
        ).get('event_channels_pnn') or []

        # Channels to always exclude from DR (non-data columns)
        _always_exclude = {'event_id', 'Time', 'ribbon'}

        if channels and not self.channel_checkboxes:
            labels = _antigen_dash_labels(self.controller)
            antigens = drc_cluster_id._channel_to_antigen_map(self.controller)
            grid_idx = 0
            for ch in channels:
                if ch in _always_exclude:
                    continue
                cb = QCheckBox(labels.get(ch, ch))
                # Pre-check only channels with an antibody assigned. A
                # channel with no antigen (scatter, Time, AF, an unused
                # detector) is excluded by construction rather than by
                # matching against a scatter-channel name list, and
                # viability dyes -- which do carry an antigen-like label
                # but are not clustering markers -- are excluded
                # explicitly via _is_viability_antigen().
                antigen = antigens.get(ch, '')
                default_on = bool(antigen) and not _is_viability_antigen(antigen)
                cb.setChecked(default_on)
                self.channel_checkboxes[ch] = cb
                row, col = divmod(grid_idx, 4)
                self.channel_layout.addWidget(cb, row, col)
                grid_idx += 1
            # Connect all checkboxes to update state
            for cb in self.channel_checkboxes.values():
                cb.stateChanged.connect(self._update_selected_channels)
            self._update_selected_channels()

        # Training sample picker — only if empty
        if not self.picker.get_ordered_list():
            self.picker.set_items(self.training_pool_paths())

        # Refresh DR and clustering status labels
        self._refresh_dr_status()
        self._refresh_cl_status()
        self._refresh_cl_space_combo()

        # Run management table — cheap metadata-only rebuild, safe
        # to call every time this tab becomes active.
        self.run_table.refresh()


class TransformTab(TransformInspector):
    """
    Tab 1 — Transform Inspector (Read-Only Preview)
    ------------------------------------------------
    Shows the Logicle / biexponential transform parameters from the experiment.
    Lets the user preview local W-adjustments (and auto-transform suggestions)
    without writing back to the experiment.

    The histogram, biplots, axis drags and CSV import/export are those of
    ``transforms_panel.TransformInspector`` (shared with the gating
    plugins); this tab adds the gate tree whose events are previewed, the
    training samples used by Auto-transform, and the reload of the
    transforms defined in Unmixed Data.  The transforms as adjusted are
    kept in ``state.saved_transforms``, which the plugin saves with the
    experiment's other settings.
    """

    transformsPersistRequested = Signal()

    def __init__(self, state: PipelineState, bus, controller, parent=None):
        super().__init__(controller, help_text=drc_help_texts.transforms_tab_help_text,
                         parent=parent)
        self.state = state
        self.bus = bus
        self._transforms_source = None
        self.auto_btn.setToolTip(
            "Set the Logicle W of every Logicle channel from the training "
            "samples (Parks, Roederer & Moore 2006).  The result is kept "
            "with this analysis and is not written to the experiment.  Use "
            "Reload from experiment to undo."
        )
        self.save_csv_btn.setToolTip(
            "Export current (local preview) transform parameters to a CSV file.\n"
            "Columns: channel, id, T, W, M, A"
        )

    # ------------------------------------------------------------------
    # Inspector hooks
    # ------------------------------------------------------------------

    def _build_selection(self, layout):
        """Gate selection — primary tree."""
        gate_box = QGroupBox("Gate(s)")
        gate_box_layout = QVBoxLayout(gate_box)
        self.gate_tree = drc_gate_tree.GateTreeWidget()
        self.gate_tree.setToolTip(
            "Check one or more gates.  Their events are UNIONED for the "
            "preview below and for Honeycluster training pools.  Checking "
            "a parent gate checks its whole subtree."
        )
        self.gate_tree.setMinimumHeight(140)
        self.gate_tree.setMaximumHeight(220)
        self.gate_tree.selectionChanged.connect(self._on_gate_tree_changed)
        gate_box_layout.addWidget(self.gate_tree)
        layout.addWidget(gate_box)

    def _extend_toolbar(self, toolbar):
        self.reload_btn = QPushButton("↺ Reload from experiment")
        self.reload_btn.setToolTip(
            "Restore every transform to the one defined in Unmixed Data, "
            "discarding Auto-transform results and axis changes made here."
        )
        self.reload_btn.clicked.connect(self.reload_from_experiment)
        toolbar.addWidget(self.reload_btn)

    def _default_channels(self) -> list[str]:
        """Fluorescence channels in the cytometer's channel order."""
        settings = self.controller.experiment.settings.get('unmixed', {})
        pnn = settings.get('event_channels_pnn') or []
        ids = sorted(set(settings.get('fluorescence_channel_ids') or []))
        named = [pnn[i] for i in ids if 0 <= i < len(pnn)]
        return named or list(settings.get('fluorescence_channels') or [])

    def channel_labels(self) -> dict:
        return _antigen_dash_labels(self.controller)

    def _on_transform_edited(self, channel: str, tr, before: dict):
        self._store_transform(channel, tr)
        self.transformsPersistRequested.emit()

    def _available_channels(self) -> list[str]:
        skip = {'ribbon', 'Time', 'event_id'}
        all_ch = [ch for ch in self._transforms if ch not in skip]
        sel = self.state.selected_channels
        return [ch for ch in all_ch if ch in sel] if sel else all_ch

    # ------------------------------------------------------------------
    # Refresh — called on tab activation
    # ------------------------------------------------------------------

    def refresh(self):
        """
        Bring the tab up to date: rebuild the gate tree, repopulate the
        channel list and biplot channels, and load the preview events.

        The transforms shown are the saved ones (state.saved_transforms),
        which start as the transforms defined in Unmixed Data and keep every
        adjustment made here (axis drags, Auto-transform, CSV);
        reload_from_experiment() discards those adjustments.
        """
        self._refresh(force_reload=False)

    def reload_from_experiment(self):
        """Replace every transform with the one defined in Unmixed Data,
        discarding Auto-transform results and axis changes made here."""
        self._refresh(force_reload=True)

    def _refresh(self, force_reload: bool):
        if self.controller.experiment.process.get('unmixing_matrix') is None:
            return

        # Gate tree — always rebuilt so renamed/added gates appear; checked
        # set comes from the shared state.selected_gates (kept in sync with
        # ConfigTab's tree by PluginWidget._on_gate_tree_changed).
        hierarchy = self.controller.unmixed_gating.get_gate_hierarchy(output='dict')
        self.gate_tree.set_hierarchy(hierarchy)
        self.gate_tree.set_checked_names(self.state.selected_gates)

        # Local copies of the controller's Transform objects, set to the saved
        # parameters.  Rebuilt on request or for a different experiment only,
        # so that adjustments survive tab switches.
        src = self.controller.unmixed_transformations or {}
        source_key = (str(getattr(self.controller, 'experiment_dir', '')), tuple(src))
        if force_reload or not self._transforms or source_key != self._transforms_source:
            saved = {} if force_reload else self.state.saved_transforms
            self._transforms = self._build_transforms(src, saved)
            self._transforms_source = source_key
            self._snapshot_transforms()

        self._pnn = self.controller.experiment.settings.get(
            'unmixed', {}
        ).get('event_channels_pnn', [])

        if not self._transforms:
            return

        # The Transforms tab always shows every channel (in PNN order);
        # channel selection for Honeycluster happens in Configuration.
        self.show_channels()
        self.redraw_all()

        # Load data (deferred so load_sample has finished)
        self._load_preview_data()

        QTimer.singleShot(50, self._redraw_plots)

    # ------------------------------------------------------------------
    # Event data loading
    # ------------------------------------------------------------------

    def _load_preview_data(self):
        """Schedule deferred data load (same pattern as AfComparisonPlotWidget)."""
        QTimer.singleShot(0, self._do_load_preview_data)

    def _do_load_preview_data(self):
        """
        Populate self._raw_data from controller.unmixed_event_data,
        optionally gating on state.selected_gates.
        """
        self._raw_data = {}
        try:
            event_data = self.controller.unmixed_event_data
            if event_data is None or len(event_data) == 0:
                print("[DR Plugin] TransformTab: no unmixed event data on controller")
                return

            channel_names = self.controller.experiment.settings.get(
                'unmixed', {}
            ).get('event_channels_pnn', [])
            if not channel_names:
                return

            selected_gates = self.state.selected_gates
            if selected_gates and selected_gates != ['root']:
                try:
                    cytometry_dict = dict(self.controller.data_for_cytometry_plots_unmixed)
                    cytometry_dict['event_data'] = event_data.copy()
                    event_data = apply_gates_union_by_lookup_table(cytometry_dict, selected_gates)
                except Exception as e:
                    print(f"[DR Plugin] TransformTab: gating failed ({selected_gates}): {e}")

            for i, ch in enumerate(channel_names):
                if i < event_data.shape[1]:
                    self._raw_data[ch] = event_data[:, i]

            print(f"[DR Plugin] TransformTab: {len(event_data):,} events "
                  f"(gates={selected_gates!r})")

            # Immediately draw tiles — don't wait for the debounce timer
            QTimer.singleShot(0, self._redraw_plots)

        except Exception as e:
            print(f"[DR Plugin] TransformTab: could not load preview data: {e}")

    def _on_gate_tree_changed(self, gates: list[str]):
        """Checked-set changed in this tab's gate tree — reload the preview."""
        self.state.selected_gates = list(gates)
        self._load_preview_data()

    @staticmethod
    def _build_transforms(src: dict, saved: dict) -> dict:
        """Copies of the Transform objects in *src*, set to the parameters in
        *saved* where there are any."""
        out = {}
        for ch, tr in src.items():
            try:
                out[ch] = (transform_from_params(saved[ch], template=tr)
                           if saved.get(ch) else deepcopy(tr))
            except Exception:
                out[ch] = deepcopy(tr)
        return out

    def _store_transform(self, channel: str, tr: 'Transform'):
        """Record one channel's transform in the shared state, where the
        other tabs read it and from where it is saved."""
        params = _transform_params(tr)
        self.state.saved_transforms[channel] = params
        self.state.channel_transform_params.update(
            _channel_transform_params_from({channel: params}))

    def _snapshot_transforms(self):
        """Record every channel's transform in the shared state."""
        self.state.saved_transforms = {
            ch: _transform_params(tr) for ch, tr in self._transforms.items()}
        self.state.channel_transform_params = _working_channel_transform_params(
            self.controller, self.state.saved_transforms)
        self.transformsPersistRequested.emit()

    # ------------------------------------------------------------------
    # Auto-transform (Parks & Moore) — over all training samples
    # ------------------------------------------------------------------

    def auto_transform_data(self):
        """
        Events for Auto-transform, pooled across ALL selected training
        samples (a more robust estimate than the currently loaded sample
        alone, and appropriate for the whole dataset): about
        TRANSFORM_TARGET_EVENTS gated events in all, an equal share from
        each sample. The currently loaded sample when no training samples
        are selected.
        """
        training_paths = self._get_training_sample_full_paths()
        if not training_paths:
            if self._raw_data:
                return ({ch: [arr] for ch, arr in self._raw_data.items()},
                        "active sample only (no training samples selected)")
            QMessageBox.information(
                self, "Auto-Transform",
                "No training samples selected and no sample is currently loaded.\n"
                "Select training samples in the Configuration tab first."
            )
            return None, ''
        print(f"\n[DR TransformTab] Auto-transform: loading {len(training_paths)} training sample(s)…")
        QApplication.setOverrideCursor(Qt.BusyCursor)
        QApplication.processEvents()
        try:
            pooled = self._load_all_training_samples(training_paths)
        finally:
            QApplication.restoreOverrideCursor()
        return pooled, f"{len(training_paths)} training sample(s)"

    def _after_auto_transform(self, n_updated: int, source_note: str):
        """Report the computed W values."""
        QMessageBox.information(
            self, "Auto-Transform",
            f"W computed for {n_updated} Logicle channel(s) from {source_note},\n"
            f"using the Parks & Moore method (lower-quarter quantile of negatives).\n\n"
            f"These transforms are kept with this analysis and are not written "
            f"to the experiment.  Reload from experiment restores the "
            f"transforms defined in Unmixed Data."
        )

    def _get_training_sample_full_paths(self) -> list[Path]:
        """
        Resolve training sample relative paths from the picker to absolute paths.
        The picker stores relative paths (relative to raw_samples_subdirectory).
        """
        try:
            raw_subdir = Path(
                self.controller.experiment.settings['raw']['raw_samples_subdirectory']
            )
            rel_paths = self.state.training_sample_ids
            if not rel_paths:
                return []
            full_paths = []
            for rel in rel_paths:
                # Try as relative to experiment dir, then relative to raw_subdir
                candidate = self.controller.experiment_dir / rel
                if not candidate.exists():
                    candidate = self.controller.experiment_dir / raw_subdir / rel
                if candidate.exists():
                    full_paths.append(candidate)
                else:
                    print(f"[DR TransformTab] Training sample not found: {rel}")
            return full_paths
        except Exception as e:
            print(f"[DR TransformTab] Could not resolve training sample paths: {e}")
            return []

    def _load_all_training_samples(
        self, full_paths: list[Path]
    ) -> dict[str, list[np.ndarray]]:
        """
        Gated, unmixed events pooled from the training samples, as
        {channel: [pooled array]}.

        Each sample contributes an equal share of about
        TRANSFORM_TARGET_EVENTS gated events (see
        transforms_panel.pool_population_events): only that many events
        are read from each file, unmixed with the sample's own AF profile
        assignment and gated with the sample's own gates (per-sample
        overrides and its own Time range). Does NOT call
        controller.load_sample (which has side effects).
        """
        channel_names = self.controller.experiment.settings.get(
            'unmixed', {}
        ).get('event_channels_pnn', [])

        result: dict[str, list[np.ndarray]] = {ch: [] for ch in channel_names}

        if self.controller.transfer_matrix is None:
            print("[DR TransformTab] transfer_matrix not initialised on controller — "
                  "cannot unmix training samples.")
            return result

        selected_gates = [g for g in (self.state.selected_gates or []) if g != 'root']
        keys = [sample_key_for_path(self.controller.experiment_dir, path) for path in full_paths]
        try:
            snap = snapshot_unmix_state(self.controller, keys, include_qc=False)
            sample_gating = SampleGating.from_controller(self.controller) if selected_gates else None
            gating_pnn = list(self.controller.data_for_cytometry_plots_unmixed['pnn'])
            pooled, counts = pool_population_events(
                self.controller.experiment_dir, keys, snap, sample_gating, gating_pnn,
                selected_gates)
        except Exception as e:
            print(f"[DR TransformTab] Could not load the training samples: {e}")
            return result

        for i, ch in enumerate(channel_names):
            if i < pooled.shape[1]:
                result[ch].append(pooled[:, i])
        print(f"[DR TransformTab] Auto-transform: {len(pooled):,} gated events from "
              f"{len(counts)} training sample(s)")
        return result


class _ResultsDrawWorker(QThread):
    """
    Builds every Groups & Stats results Figure (heatmap/volcano for
    freq/counts/MFI) off the main thread -- same reasoning as
    _MarkerSummaryWorker elsewhere in this file: Figure objects are plain
    matplotlib, no Qt, so building them off-thread is safe; only tab/
    canvas creation (GroupsStatsTab._on_results_figures_built, on the
    main thread) touches Qt.
    """
    finished = Signal(dict)

    def __init__(self, build_fn, parent=None):
        super().__init__(parent)
        self._build_fn = build_fn

    def run(self):
        try:
            payload = self._build_fn()
        except Exception:
            traceback.print_exc()
            payload = {}
        self.finished.emit(payload)


class GroupsStatsTab(QWidget):
    """
    Tab 2 — Groups, Factors & Statistics
    --------------------------------------
      • Named comparison groups with regex auto-population
      • Sample-to-group assignment table with combo boxes
      • Covariate entry (arbitrary named columns)
      • CSV import / export
      • Statistics controls: cluster frequencies (moderated linear
        model), cluster counts (GLM) and/or MFIs; FDR level, effect
        thresholds, TREAT, covariate adjustment, cluster-first MFIs
      • Background statistics worker
      • Results: heatmap+dendrogram, volcano plot, MFI cluster summary,
        CSV export
      • Optional predictive analysis (nested cross-validation)
    """

    # Emitted when new statistics or predictive results should be written
    # to the experiment's saved state.
    results_computed = Signal()
    # Emitted once the result figures have been built and shown as tabs.
    results_drawn = Signal()

    def __init__(self, state: PipelineState, bus, controller, parent=None):
        super().__init__(parent)
        self.state = state
        self.bus = bus
        self.controller = controller
        # group names and patterns live on state (state.group_names /
        # state.group_patterns) — no local mirror to keep in sync.
        self._stats_worker = None
        self._predictive_worker = None
        # Names of the long-running jobs that currently hold a busy cursor.
        self._busy_cursor_owners: set[str] = set()
        # _current_data_key() of the last computed (or verified restored)
        # Run Statistics — lets _run_statistics replot instead of
        # refitting when only thresholds changed.
        self._last_stats_data_key = None
        # the _ResultsDrawWorker currently building figures, if
        # any, and whether another _draw_results() call arrived while it
        # was busy (coalesced into one more redraw once it finishes,
        # rather than a second worker starting against a moving target).
        self._results_draw_worker = None
        self._results_draw_pending = False
        # {key: {'title', 'maker', 'maker_kwargs'}} -- mirrors whatever's
        # currently open in self._results_tabs, kept in sync by
        # _add_results_tab / _remove_results_tab_by_key /
        # _on_results_tab_close_requested. Lets the Report tab regenerate
        # a fresh Figure/DataFrame for any open result without duplicating
        # the maker logic -- see get_report_items().
        self._results_registry: dict[str, dict] = {}
        # Automatic Suggest Groupings: the sample list it last ran for (it
        # runs once per list) and the state to restore on Undo.
        self._auto_groupings_key = None
        self._auto_groupings_undo = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # The plugin's outer QScrollArea (PluginWidget) uses
        # setWidgetResizable(True), which clamps its content to the
        # viewport size rather than growing with it — so once several
        # result plots are stacked in the bottom panel, this tab needs its
        # own internal scrollbar instead of relying on that outer area.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer_scroll = QScrollArea()
        outer_scroll.setWidgetResizable(True)
        outer_scroll.setFrameShape(QFrame.NoFrame)
        outer_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(outer_scroll)

        content = QWidget()
        outer_scroll.setWidget(content)

        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(8)

        self.help_widget = HelpToggleWidget(
            text=drc_help_texts.stats_tab_help_text
        )
        content_layout.addWidget(self.help_widget)

        # ============================================================
        # Top panel: group definitions + sample assignment table
        # ============================================================
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)

        # ---- Group management (arbitrary N named groups) ----
        group_box = QGroupBox("Comparison Groups")
        group_box_layout = QVBoxLayout(group_box)

        name_hint = _HintLabel(
            "Define one or more comparison groups. Double-click a group's "
            "name or match pattern below to edit it in place. Assign "
            "samples to a group in the table beneath, or set a match "
            "pattern per group and use 'Auto-assign by pattern'."
        )
        name_hint.setWordWrap(True)
        group_box_layout.addWidget(name_hint)

        self.groups_table = QTableWidget(0, 3)
        self.groups_table.setHorizontalHeaderLabels(
            ['Group Name', 'Match Pattern (regex)', 'Colour']
        )
        self.groups_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.groups_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.groups_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.groups_table.verticalHeader().setVisible(False)
        self.groups_table.setEditTriggers(QAbstractItemView.DoubleClicked)
        self.groups_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.groups_table.setMaximumHeight(120)
        self.groups_table.itemChanged.connect(self._on_groups_table_item_changed)
        self.groups_table.cellDoubleClicked.connect(self._on_groups_table_cell_double_clicked)
        self.groups_table.setToolTip(
            "Double-click a group's Colour cell to change it — used by the "
            "Sample PCA plot (and any future all-groups plot)."
        )
        group_box_layout.addWidget(self.groups_table)

        groups_btn_row = QHBoxLayout()
        self.add_group_btn = QPushButton("Add Group")
        self.add_group_btn.clicked.connect(self._add_group)
        groups_btn_row.addWidget(self.add_group_btn)
        self.remove_group_btn = QPushButton("Remove Selected Group")
        self.remove_group_btn.clicked.connect(self._remove_selected_group)
        groups_btn_row.addWidget(self.remove_group_btn)
        groups_btn_row.addStretch()

        self.auto_assign_btn = QPushButton("Auto-assign by pattern")
        self.auto_assign_btn.setToolTip(
            "Regex-match each group's pattern (above) against sample "
            "filenames and assign groups automatically. First matching "
            "pattern wins (in the order shown above).\n"
            "This overwrites every on-screen sample's current group, "
            "including manual selections -- re-run any time patterns change."
        )
        self.auto_assign_btn.clicked.connect(self._auto_assign_by_name)
        groups_btn_row.addWidget(self.auto_assign_btn)
        group_box_layout.addLayout(groups_btn_row)
        top_layout.addWidget(group_box)

        # ---- Sample assignment table ----
        table_box = QGroupBox("Sample Group Assignment")
        table_box_layout = QVBoxLayout(table_box)

        csv_row = QHBoxLayout()
        self.import_csv_btn = QPushButton("Import CSV")
        self.import_csv_btn.setToolTip(
            "Import group and covariate assignments from a CSV file.\n"
            "Required columns: 'sample', 'group'.  Optional covariate columns follow."
        )
        self.import_csv_btn.clicked.connect(self._import_csv)
        self.export_csv_btn = QPushButton("Export CSV")
        self.export_csv_btn.clicked.connect(self._export_csv)
        self.add_column_btn = QPushButton("+ Add Column")
        self.add_column_btn.setToolTip(
            "Add a new covariate column to the table (e.g. 'donor', "
            "'batch') -- fill it in per sample below, then pick it as "
            "the 'Pairing variable' further down to use it for paired "
            "testing."
        )
        self.add_column_btn.clicked.connect(self._add_covariate_column)
        self.suggest_groupings_btn = QPushButton("Suggest Groupings…")
        self.suggest_groupings_btn.setToolTip(
            "Scan the sample names below for repeated, delimiter-separated "
            "tokens (e.g. tissue, mouse/donor ID) and suggest a Group "
            "column and/or covariate/pairing columns automatically."
        )
        self.suggest_groupings_btn.clicked.connect(self._suggest_groupings)
        self.remove_column_btn = QPushButton("Remove Column")
        self.remove_column_btn.setToolTip(
            "Remove a covariate column (e.g. one added by mistake, or left "
            "empty) from the table below."
        )
        self.remove_column_btn.clicked.connect(self._remove_covariate_column)
        csv_row.addWidget(self.import_csv_btn)
        csv_row.addWidget(self.export_csv_btn)
        csv_row.addWidget(self.add_column_btn)
        csv_row.addWidget(self.remove_column_btn)
        csv_row.addWidget(self.suggest_groupings_btn)
        csv_row.addStretch()

        self._group_count_label = QLabel()
        self._group_count_label.setStyleSheet("color: grey; font-style: italic;")
        csv_row.addWidget(self._group_count_label)
        table_box_layout.addLayout(csv_row)

        auto_row = QHBoxLayout()
        self._auto_groupings_label = QLabel()
        self._auto_groupings_label.setWordWrap(True)
        self._auto_groupings_label.setStyleSheet("color: grey; font-style: italic;")
        self._auto_groupings_undo_btn = QPushButton("Undo")
        self._auto_groupings_undo_btn.setToolTip("Remove the groups and pairing applied from the sample names.")
        self._auto_groupings_undo_btn.clicked.connect(self._undo_auto_groupings)
        auto_row.addWidget(self._auto_groupings_label, stretch=1)
        auto_row.addWidget(self._auto_groupings_undo_btn)
        self._auto_groupings_label.setVisible(False)
        self._auto_groupings_undo_btn.setVisible(False)
        table_box_layout.addLayout(auto_row)

        # Table: Sample | Group | [covariate columns...] -- column count is
        # dynamic (one per state.covariates column, added via '+ Add
        # Column' or CSV import). Never includes single-stain/unstained
        # controls (see _non_control_sample_paths).
        self.table = QTableWidget(0, 2)
        self._table_header = CheckableHeaderView(
            self.table, check_label="Numeric",
            tooltip="Enter this covariate as a continuous number (e.g. age, CRP) instead of "
                    "as categories. Every value must be a number.")
        self.table.setHorizontalHeader(self._table_header)
        self._table_header.check_toggled.connect(self._on_covariate_numeric_toggled)
        self.table.setHorizontalHeaderLabels(['Sample', 'Group'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setTextElideMode(Qt.ElideMiddle)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        table_box_layout.addWidget(self.table)

        top_layout.addWidget(table_box, stretch=1)
        content_layout.addWidget(top_widget)

        # ============================================================
        # Bottom panel: statistics controls + results
        # ============================================================
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(6)

        stats_box = QGroupBox("Differential Statistics")
        stats_layout = QVBoxLayout(stats_box)

        # Run selector row
        run_sel_row = QHBoxLayout()
        run_sel_row.addWidget(QLabel("Run:"))
        self._run_combo = QComboBox()
        self._run_combo.setMinimumWidth(320)
        self._run_combo.setToolTip(
            "Select which run to use for statistics (DR runs are shown but "
            "have no cluster labels — 'Run Statistics' stays disabled for "
            "those; select or run a clustering run instead).\n"
            "Only runs whose training samples overlap the currently-assigned "
            "samples are shown."
        )
        run_sel_row.addWidget(self._run_combo)
        run_sel_row.addStretch()
        stats_layout.addLayout(run_sel_row)
        self._run_combo.currentIndexChanged.connect(self._on_run_combo_changed)

        # ---- Groups to Test + contrast mode + pairing ----
        test_groups_box = QGroupBox("Groups to Test")
        test_groups_layout = QVBoxLayout(test_groups_box)

        test_groups_hint = _HintLabel(
            "Check every group to include in Frequency/Counts/MFI testing "
            "and Composition-by-group."
        )
        test_groups_hint.setWordWrap(True)
        test_groups_layout.addWidget(test_groups_hint)

        self.test_groups_list = QListWidget()
        self.test_groups_list.setFixedHeight(90)
        self.test_groups_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.test_groups_list.itemChanged.connect(self._on_test_group_checked_changed)
        test_groups_layout.addWidget(self.test_groups_list)

        contrast_row = QHBoxLayout()
        self.contrast_mode_group = QButtonGroup(self)
        self.radio_reference = QRadioButton("Reference group")
        self.radio_pairwise = QRadioButton("All pairwise")
        self.radio_reference.setChecked(True)
        self.contrast_mode_group.addButton(self.radio_reference)
        self.contrast_mode_group.addButton(self.radio_pairwise)
        self.radio_reference.toggled.connect(self._on_contrast_mode_changed)
        contrast_row.addWidget(self.radio_reference)
        contrast_row.addWidget(self.radio_pairwise)
        contrast_row.addSpacing(12)
        contrast_row.addWidget(QLabel("Reference:"))
        self.reference_group_combo = QComboBox()
        self.reference_group_combo.setMinimumWidth(140)
        self.reference_group_combo.currentTextChanged.connect(self._on_reference_group_changed)
        contrast_row.addWidget(self.reference_group_combo)
        contrast_row.addStretch()
        test_groups_layout.addLayout(contrast_row)

        paired_row = QHBoxLayout()
        self.chk_paired = QCheckBox("Paired design")
        self.chk_paired.setToolTip(
            "Add the pairing variable below as a fixed-effect blocking "
            "term (e.g. donor ID). Each pairing level should appear in "
            "more than one group, or the design cannot be fitted."
        )
        self.chk_paired.toggled.connect(self._on_paired_toggled)
        paired_row.addWidget(self.chk_paired)
        paired_row.addWidget(QLabel("Pairing variable:"))
        self.pairing_variable_combo = QComboBox()
        self.pairing_variable_combo.setMinimumWidth(140)
        self.pairing_variable_combo.setEnabled(False)
        self.pairing_variable_combo.setToolTip(
            "Which covariate column (added via '+ Add Column' on the table "
            "above, or imported via CSV) to use as the pairing/blocking "
            "variable."
        )
        self.pairing_variable_combo.currentTextChanged.connect(self._on_pairing_variable_changed)
        paired_row.addWidget(self.pairing_variable_combo)
        paired_row.addStretch()
        test_groups_layout.addLayout(paired_row)

        adjust_row = QHBoxLayout()
        adjust_label = QLabel("Adjust for:")
        adjust_label.setAlignment(Qt.AlignTop)
        adjust_row.addWidget(adjust_label)
        self.adjust_covariates_list = QListWidget()
        self.adjust_covariates_list.setFixedHeight(70)
        self.adjust_covariates_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.adjust_covariates_list.setToolTip(
            "Covariate columns from the table above to include in every "
            "test's model (e.g. age, sex, batch). Columns ticked Numeric "
            "in the table above are fitted as continuous terms; anything "
            "else as categories. Every tested sample needs a value. The pairing "
            "variable is already in the model when 'Paired design' is on."
        )
        self.adjust_covariates_list.itemChanged.connect(self._on_adjust_covariate_changed)
        adjust_row.addWidget(self.adjust_covariates_list, stretch=1)
        test_groups_layout.addLayout(adjust_row)

        stats_layout.addWidget(test_groups_box)

        # Config row: what to test
        config_row = QHBoxLayout()
        config_row.addWidget(QLabel("Test:"))
        self.chk_freq = QCheckBox("Cluster Frequencies")
        self.chk_freq.setChecked(True)
        self.chk_freq.setToolTip(
            "log2 % of events per cluster per sample → moderated linear "
            "model (empirical Bayes variance moderation)"
        )
        self.chk_counts = QCheckBox("Cluster Counts (GLM)")
        self.chk_counts.setChecked(False)
        self.chk_counts.setToolTip(
            "Raw event counts per cluster per sample → negative-"
            "binomial GLM.\nParallel option to the Frequency test, not a "
            "replacement — run both to compare on the same data."
        )
        self.chk_mfi = QCheckBox("Cluster MFIs")
        self.chk_mfi.setChecked(True)
        self.chk_mfi.setToolTip(
            "Mean transformed intensity of each marker within each cluster, "
            "per sample → moderated linear model. Clusters with no events "
            "in a sample are left out for that sample."
        )
        config_row.addWidget(self.chk_freq)
        config_row.addWidget(self.chk_counts)
        config_row.addWidget(self.chk_mfi)
        config_row.addSpacing(20)

        config_row.addWidget(QLabel("FDR ≤"))
        self.pval_spin = QDoubleSpinBox()
        self.pval_spin.setRange(0.001, 1.0)
        self.pval_spin.setSingleStep(0.01)
        self.pval_spin.setDecimals(3)
        self.pval_spin.setValue(0.05)
        self.pval_spin.setFixedWidth(70)
        self.pval_spin.setToolTip(
            "Benjamini–Hochberg adjusted p-value cut-off. For cluster-first "
            "MFI testing this is the level used at both stages."
        )
        config_row.addWidget(self.pval_spin)

        config_row.addWidget(QLabel("Freq/Counts |log₂FC| ≥"))
        self.fc_spin = QDoubleSpinBox()
        self.fc_spin.setRange(0.0, 10.0)
        self.fc_spin.setSingleStep(0.1)
        self.fc_spin.setDecimals(2)
        self.fc_spin.setValue(0.5)
        self.fc_spin.setFixedWidth(70)
        self.fc_spin.setToolTip(
            "Smallest log2 fold change in cluster frequency (or count rate) "
            "worth reporting. 0.5 is a 1.4-fold change, 1 a 2-fold change."
        )
        config_row.addWidget(self.fc_spin)

        config_row.addWidget(QLabel("MFI |Δ| ≥"))
        self.mfi_fc_spin = QDoubleSpinBox()
        self.mfi_fc_spin.setRange(0.0, 1.0e6)
        self.mfi_fc_spin.setSingleStep(0.01)
        self.mfi_fc_spin.setDecimals(3)
        self.mfi_fc_spin.setValue(self.state.stats_mfi_threshold)
        self.mfi_fc_spin.setFixedWidth(80)
        self.mfi_fc_spin.setToolTip(
            "Smallest difference in mean transformed intensity worth "
            "reporting, in Transforms-tab units. Logicle/arcsinh scales run "
            "0–1, so on a 4.5-decade Logicle 0.067 is about a 2-fold "
            "change and 0.03 about 1.4-fold. Linear channels use raw units."
        )
        config_row.addWidget(self.mfi_fc_spin)

        config_row.addSpacing(20)
        config_row.addWidget(QLabel("FDR:"))
        self.fdr_scope_combo = QComboBox()
        self.fdr_scope_combo.addItems(["Pooled (all comparisons)", "Per comparison"])
        self.fdr_scope_combo.setToolTip(
            "Which BH-FDR correction feeds the 'significant' flag on the "
            "volcano plot and heatmap. Pooled: one correction across every "
            "comparison shown. Per comparison: each comparison corrected "
            "on its own clusters/channels only."
        )
        config_row.addWidget(self.fdr_scope_combo)

        config_row.addStretch()
        stats_layout.addLayout(config_row)

        method_row = QHBoxLayout()
        self.chk_treat = QCheckBox("Test against thresholds (TREAT)")
        self.chk_treat.setChecked(self.state.stats_use_treat)
        self.chk_treat.setToolTip(
            "On: each p-value tests whether the effect is LARGER than the "
            "threshold (McCarthy & Smyth 2009), so small but consistent "
            "shifts are not reported. Off: p-values test for any non-zero "
            "effect and the threshold only filters the estimates."
        )
        method_row.addWidget(self.chk_treat)
        method_row.addSpacing(20)
        self.chk_mfi_hierarchical = QCheckBox("Cluster-first MFI testing")
        self.chk_mfi_hierarchical.setChecked(self.state.stats_mfi_hierarchical)
        self.chk_mfi_hierarchical.setToolTip(
            "On: clusters are tested first (all their markers combined); "
            "markers are then tested only inside clusters that pass, with "
            "the FDR level adjusted for how many clusters were selected "
            "(Benjamini & Bogomolov 2014). Reports which clusters change "
            "and the markers that drive them. Off: every cluster × marker "
            "is corrected as one flat list."
        )
        method_row.addWidget(self.chk_mfi_hierarchical)
        method_row.addSpacing(20)
        method_row.addWidget(QLabel("Counts test:"))
        self.counts_method_combo = QComboBox()
        self.counts_method_combo.addItem("Wald", 'wald')
        self.counts_method_combo.addItem("Quasi-likelihood F", 'ql')
        self.counts_method_combo.setToolTip(
            "Wald: each cluster's own negative-binomial dispersion and a z "
            "test; finds too many hits with few samples per group or a "
            "paired design. Quasi-likelihood F: dispersions shrunk towards a "
            "common value, then a moderated quasi-dispersion per cluster; "
            "holds its false-positive rate with few samples."
        )
        method_row.addWidget(self.counts_method_combo)
        method_row.addSpacing(10)
        method_row.addWidget(QLabel("Library size:"))
        self.normalisation_combo = QComboBox()
        self.normalisation_combo.addItem("Total events", 'total')
        self.normalisation_combo.addItem("Robust (trimmed)", 'robust')
        self.normalisation_combo.setToolTip(
            "What the Frequency and Counts tests compare each cluster "
            "against. Total: the sample's events over all clusters, so one "
            "expanding cluster pushes every other cluster down. Robust: the "
            "total rescaled by a trimmed ratio of sums that a few dominant "
            "clusters cannot move."
        )
        method_row.addWidget(self.normalisation_combo)
        method_row.addStretch()
        stats_layout.addLayout(method_row)

        # ---- Marker roles: type (clustering) vs state (tested) ----
        roles_box = QGroupBox("Marker Roles — MFI Testing")
        roles_layout = QVBoxLayout(roles_box)
        roles_hint = _HintLabel(
            "Every channel defaults to 'state' (tested) and is included in "
            "MFI significance testing. Check a channel to keep it included; "
            "uncheck a channel to mark it 'type' (a clustering marker) and "
            "exclude it -- useful for a channel that drove the selected "
            "clustering run, to avoid it driving both the cluster "
            "assignment and its own significance call."
        )
        roles_hint.setWordWrap(True)
        roles_layout.addWidget(roles_hint)

        roles_btn_row = QHBoxLayout()
        self.reset_roles_btn = QPushButton("Reset all to Activation")
        self.reset_roles_btn.setToolTip(
            "Set every selected channel back to 'state' (activation) -- "
            "overwrites any per-channel overrides made above."
        )
        self.reset_roles_btn.clicked.connect(self._reset_marker_roles_to_defaults)
        roles_btn_row.addWidget(self.reset_roles_btn)
        roles_btn_row.addStretch()
        roles_layout.addLayout(roles_btn_row)

        # Channel checkboxes laid out in a grid (same pattern as
        # ConfigTab.channel_layout) instead of a single fixed-height
        # column -- grows to fit however many channels are selected;
        # this tab's own outer QScrollArea (see top of _build_ui) scrolls
        # the whole page if that makes it taller than the viewport.
        self.marker_roles_widget = QWidget()
        self.marker_roles_grid = QGridLayout(self.marker_roles_widget)
        self.marker_roles_grid.setSpacing(4)
        self.marker_roles_checkboxes: dict[str, QCheckBox] = {}
        roles_layout.addWidget(self.marker_roles_widget)

        self.chk_include_type_markers = QCheckBox(
            "Include clustering (type) markers in MFI testing"
        )
        # default is now "include everything", matching every
        # channel checkbox below defaulting to checked/included.
        self.chk_include_type_markers.setChecked(True)
        self.chk_include_type_markers.setToolTip(
            "On (default): every selected channel is tested for MFI "
            "significance, regardless of the roles below -- ticking this "
            "also ticks every channel checkbox below. Off: only channels "
            "checked below are tested; unchecked ('type') channels are "
            "excluded."
        )
        self.chk_include_type_markers.toggled.connect(self._on_include_type_markers_toggled)
        roles_layout.addWidget(self.chk_include_type_markers)

        stats_layout.addWidget(roles_box)

        # Run / status row
        run_row = QHBoxLayout()
        self.run_stats_btn = QPushButton("▶  Run Statistics")
        self.run_stats_btn.setEnabled(False)
        self.run_stats_btn.setToolTip("Requires ≥ 3 samples in each group and clustering complete.")
        self.run_stats_btn.clicked.connect(self._run_statistics)
        run_row.addWidget(self.run_stats_btn)

        self.composition_btn = QPushButton("Composition Barplot")
        self.composition_btn.setEnabled(False)
        self.composition_btn.setToolTip(
            "Stacked cluster-composition barplot (counts or %).\n"
            "Available as soon as groups are assigned and a clustering run "
            "is selected — no Frequency/MFI stats required first."
        )
        self.composition_btn.clicked.connect(self._show_composition_barplot)
        run_row.addWidget(self.composition_btn)

        self.composition_pct_chk = QCheckBox("%")
        self.composition_pct_chk.setChecked(True)
        self.composition_pct_chk.setToolTip(
            "Show composition as percentage rather than raw event counts."
        )
        run_row.addWidget(self.composition_pct_chk)

        self.composition_by_group_chk = QCheckBox("By group")
        self.composition_by_group_chk.setToolTip(
            "Sum across each group's samples instead of one bar per sample."
        )
        run_row.addWidget(self.composition_by_group_chk)

        self.stats_status_label = QLabel("")
        self.stats_status_label.setStyleSheet("color: grey;")
        self.stats_status_label.setWordWrap(True)
        run_row.addWidget(self.stats_status_label, 1)
        run_row.addStretch()

        self.export_results_btn = QPushButton("Export Results CSV")
        self.export_results_btn.setEnabled(False)
        self.export_results_btn.clicked.connect(self._export_results_csv)
        run_row.addWidget(self.export_results_btn)

        self.export_tables_btn = QPushButton("Export Tables…")
        self.export_tables_btn.setToolTip(
            "Write the last statistics run into a folder as one CSV file per table: "
            "per-sample cluster frequencies, counts and MFIs (with group and covariates), "
            "the cluster list, and the differential results of each test."
        )
        self.export_tables_btn.setEnabled(False)
        self.export_tables_btn.clicked.connect(self._export_tables)
        run_row.addWidget(self.export_tables_btn)
        stats_layout.addLayout(run_row)

        # ---- Viewing comparison ----
        viewing_row = QHBoxLayout()
        viewing_row.addWidget(QLabel("Viewing comparison:"))
        self.viewing_comparison_combo = QComboBox()
        self.viewing_comparison_combo.setMinimumWidth(200)
        self.viewing_comparison_combo.setToolTip(
            "A single Run Statistics click can test several group "
            "comparisons at once. Freq/Counts/MFI Heatmap and Volcano "
            "each show one comparison at a time -- pick which here."
        )
        self.viewing_comparison_combo.currentIndexChanged.connect(
            self._on_viewing_comparison_changed
        )
        viewing_row.addWidget(self.viewing_comparison_combo)
        viewing_row.addStretch()
        stats_layout.addLayout(viewing_row)

        bottom_layout.addWidget(stats_box)

        # ============================================================
        # Sample PCA
        # ============================================================
        pca_box = QGroupBox("Sample PCA")
        pca_layout = QVBoxLayout(pca_box)

        pca_hint = _HintLabel(
            "PCA over every sample in the checked 'Groups to Test' "
            "(not one comparison at a time) — pick which computed source(s) "
            "to build it from below. Requires Run Statistics to have been "
            "run first with the matching 'Test:' box(es) checked."
        )
        pca_hint.setWordWrap(True)
        pca_layout.addWidget(pca_hint)

        pca_source_row = QHBoxLayout()
        pca_source_row.addWidget(QLabel("Build from:"))
        self.pca_chk_freq = QCheckBox("Frequencies")
        self.pca_chk_freq.setChecked(self.state.pca_use_freq)
        self.pca_chk_counts = QCheckBox("Counts")
        self.pca_chk_counts.setChecked(self.state.pca_use_counts)
        self.pca_chk_mfi = QCheckBox("MFIs")
        self.pca_chk_mfi.setChecked(self.state.pca_use_mfi)
        pca_source_row.addWidget(self.pca_chk_freq)
        pca_source_row.addWidget(self.pca_chk_counts)
        pca_source_row.addWidget(self.pca_chk_mfi)
        pca_source_row.addSpacing(20)

        self.pca_chk_loadings = QCheckBox("Show loadings")
        self.pca_chk_loadings.setChecked(self.state.pca_show_loadings)
        pca_source_row.addWidget(self.pca_chk_loadings)

        pca_source_row.addWidget(QLabel("Top N"))
        self.pca_n_loadings_spin = QSpinBox()
        self.pca_n_loadings_spin.setRange(1, 50)
        self.pca_n_loadings_spin.setValue(self.state.pca_n_loadings)
        self.pca_n_loadings_spin.setFixedWidth(60)
        pca_source_row.addWidget(self.pca_n_loadings_spin)

        self.pca_chk_label_loadings = QCheckBox("Label loadings")
        self.pca_chk_label_loadings.setChecked(self.state.pca_label_loadings)
        self.pca_chk_label_loadings.setToolTip(
            "Show each loading's feature name as a permanent label at its "
            "arrow tip. Mouse-over identifies every loading regardless of "
            "this toggle -- this only controls the always-visible ones."
        )
        pca_source_row.addWidget(self.pca_chk_label_loadings)

        self.pca_chk_label_points = QCheckBox("Label points")
        self.pca_chk_label_points.setChecked(self.state.pca_label_points)
        self.pca_chk_label_points.setToolTip(
            "Show each sample's name as a permanent label next to its "
            "point. Mouse-over identifies every point regardless of this "
            "toggle -- this only controls the always-visible ones."
        )
        pca_source_row.addWidget(self.pca_chk_label_points)

        pca_source_row.addStretch()
        pca_layout.addLayout(pca_source_row)

        pca_appearance_row = QHBoxLayout()
        pca_appearance_row.addWidget(QLabel("Point size"))
        self.pca_point_size_spin = QSpinBox()
        self.pca_point_size_spin.setRange(5, 500)
        self.pca_point_size_spin.setValue(int(self.state.pca_point_size))
        self.pca_point_size_spin.setFixedWidth(60)
        pca_appearance_row.addWidget(self.pca_point_size_spin)

        pca_appearance_row.addWidget(QLabel("Arrow width"))
        self.pca_arrow_lw_spin = QDoubleSpinBox()
        self.pca_arrow_lw_spin.setRange(0.1, 10.0)
        self.pca_arrow_lw_spin.setSingleStep(0.1)
        self.pca_arrow_lw_spin.setDecimals(1)
        self.pca_arrow_lw_spin.setValue(self.state.pca_arrow_lw)
        self.pca_arrow_lw_spin.setFixedWidth(60)
        pca_appearance_row.addWidget(self.pca_arrow_lw_spin)

        pca_appearance_row.addWidget(QLabel("Arrow colour"))
        self.pca_arrow_color_btn = QPushButton()
        self.pca_arrow_color_btn.setFixedWidth(50)
        self.pca_arrow_color_btn.clicked.connect(self._pick_pca_arrow_color)
        pca_appearance_row.addWidget(self.pca_arrow_color_btn)
        self._update_pca_arrow_color_btn()

        pca_appearance_row.addWidget(QLabel("Axis font size"))
        self.pca_axis_fontsize_spin = QSpinBox()
        self.pca_axis_fontsize_spin.setRange(6, 24)
        self.pca_axis_fontsize_spin.setValue(self.state.pca_axis_fontsize)
        self.pca_axis_fontsize_spin.setFixedWidth(50)
        pca_appearance_row.addWidget(self.pca_axis_fontsize_spin)

        self.pca_chk_grid = QCheckBox("Grid lines")
        self.pca_chk_grid.setChecked(self.state.pca_show_grid)
        pca_appearance_row.addWidget(self.pca_chk_grid)
        pca_appearance_row.addStretch()
        pca_layout.addLayout(pca_appearance_row)

        pca_btn_row = QHBoxLayout()
        self.pca_btn = QPushButton("Show PCA")
        self.pca_btn.setEnabled(False)
        self.pca_btn.setToolTip(
            "Requires ≥ 3 samples in at least two checked 'Groups to Test', "
            "and Run Statistics already computed for at least one checked "
            "'Build from' source."
        )
        self.pca_btn.clicked.connect(self._show_pca)
        pca_btn_row.addWidget(self.pca_btn)
        pca_btn_row.addStretch()
        pca_layout.addLayout(pca_btn_row)

        # Live-refresh -- once the PCA has been shown once,
        # changing any of these regenerates it without needing another
        # 'Show PCA' click. See _on_pca_option_changed.
        self.pca_chk_freq.stateChanged.connect(self._on_pca_option_changed)
        self.pca_chk_counts.stateChanged.connect(self._on_pca_option_changed)
        self.pca_chk_mfi.stateChanged.connect(self._on_pca_option_changed)
        self.pca_chk_loadings.stateChanged.connect(self._on_pca_option_changed)
        self.pca_n_loadings_spin.valueChanged.connect(self._on_pca_option_changed)
        self.pca_chk_label_loadings.stateChanged.connect(self._on_pca_option_changed)
        self.pca_chk_label_points.stateChanged.connect(self._on_pca_option_changed)
        self.pca_point_size_spin.valueChanged.connect(self._on_pca_option_changed)
        self.pca_arrow_lw_spin.valueChanged.connect(self._on_pca_option_changed)
        self.pca_axis_fontsize_spin.valueChanged.connect(self._on_pca_option_changed)
        self.pca_chk_grid.stateChanged.connect(self._on_pca_option_changed)

        bottom_layout.addWidget(pca_box)

        # ============================================================
        # Predictive analysis (optional)
        # ============================================================
        pred_box = QGroupBox("Predictive Analysis (optional)")
        pred_layout = QVBoxLayout(pred_box)
        pred_hint = _HintLabel(
            "Which small set of cluster features best tells two groups apart, "
            "and how well? Lasso logistic regression and random forest models "
            "are trained and scored by nested cross-validation on the "
            "matrices from Run Statistics. Needs at least "
            f"{drc_predictive.MIN_PER_GROUP} samples per group; "
            f"{drc_predictive.RECOMMENDED_PER_GROUP}+ per group is recommended. "
            "This complements the differential tests: a feature can differ "
            "between groups without being picked, e.g. when a correlated "
            "feature carries the same information."
        )
        pred_hint.setWordWrap(True)
        pred_layout.addWidget(pred_hint)

        pred_row = QHBoxLayout()
        pred_row.addWidget(QLabel("Comparison:"))
        self.pred_comparison_combo = QComboBox()
        self.pred_comparison_combo.setMinimumWidth(200)
        pred_row.addWidget(self.pred_comparison_combo)
        pred_row.addSpacing(12)
        pred_row.addWidget(QLabel("Features:"))
        self.pred_chk_freq = QCheckBox("Frequencies")
        self.pred_chk_freq.setChecked(True)
        self.pred_chk_mfi = QCheckBox("MFIs")
        self.pred_chk_mfi.setToolTip(
            "Adds every cluster × marker MFI. Many more features: slower, and "
            "needs more samples to learn from."
        )
        pred_row.addWidget(self.pred_chk_freq)
        pred_row.addWidget(self.pred_chk_mfi)
        pred_row.addSpacing(12)
        pred_row.addWidget(QLabel("Models:"))
        self.pred_chk_lasso = QCheckBox("Lasso logistic")
        self.pred_chk_lasso.setChecked(True)
        self.pred_chk_forest = QCheckBox("Random forest")
        self.pred_chk_forest.setChecked(True)
        pred_row.addWidget(self.pred_chk_lasso)
        pred_row.addWidget(self.pred_chk_forest)
        pred_row.addSpacing(12)
        pred_row.addWidget(QLabel("Repeats"))
        self.pred_repeats_spin = QSpinBox()
        self.pred_repeats_spin.setRange(1, 50)
        self.pred_repeats_spin.setValue(5)
        self.pred_repeats_spin.setFixedWidth(55)
        self.pred_repeats_spin.setToolTip(
            "How many times the outer 5-fold cross-validation is repeated with "
            "a different split. More repeats give steadier estimates."
        )
        pred_row.addWidget(self.pred_repeats_spin)
        pred_row.addStretch()
        pred_layout.addLayout(pred_row)

        pred_btn_row = QHBoxLayout()
        self.pred_run_btn = QPushButton("▶  Run Predictive Analysis")
        self.pred_run_btn.setEnabled(False)
        self.pred_run_btn.setToolTip("Run Statistics first.")
        self.pred_run_btn.clicked.connect(self._run_predictive)
        pred_btn_row.addWidget(self.pred_run_btn)
        self.pred_status_label = QLabel("")
        self.pred_status_label.setWordWrap(True)
        self.pred_status_label.setStyleSheet("color: grey;")
        pred_btn_row.addWidget(self.pred_status_label, stretch=1)
        pred_layout.addLayout(pred_btn_row)
        self._predictive_worker = None

        bottom_layout.addWidget(pred_box)

        # Results area: one tab per plot
        self._results_tabs = QTabWidget()
        self._results_tabs.setTabsClosable(True)
        self._results_tabs.tabCloseRequested.connect(self._on_results_tab_close_requested)
        self._results_tabs.setMinimumHeight(1400)
        bottom_layout.addWidget(self._results_tabs, stretch=1)

        content_layout.addWidget(bottom_widget, stretch=1)

        # Sentinel: cluster names snapshot at last _draw_results() call.
        self._last_drawn_cluster_names: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Group management (arbitrary N named groups)
    # ------------------------------------------------------------------

    def _group_options(self) -> list[str]:
        """Return the full list of group options for the per-sample combos."""
        return ['Unassigned'] + list(self.state.group_names)

    def _populate_groups_table(self):
        """Rebuild the Group Name / Match Pattern / Colour management table
        from state."""
        self.groups_table.blockSignals(True)
        self.groups_table.setRowCount(0)
        for name in self.state.group_names:
            row = self.groups_table.rowCount()
            self.groups_table.insertRow(row)
            self.groups_table.setItem(row, 0, QTableWidgetItem(name))
            self.groups_table.setItem(
                row, 1, QTableWidgetItem(self.state.group_patterns.get(name, ''))
            )
            colour_item = QTableWidgetItem('')
            colour_item.setBackground(QColor(self._group_colour(name)))
            colour_item.setFlags(colour_item.flags() & ~Qt.ItemIsEditable)
            self.groups_table.setItem(row, 2, colour_item)
        self.groups_table.blockSignals(False)

    def _group_colour(self, name: str) -> str:
        """Return this group's colour -- thin per-tab wrapper
        around the shared _resolve_group_colour() so PlotCard's
        'Group' colour mode and this tab's Comparison Groups table /
        Sample PCA plot always agree."""
        return _resolve_group_colour(self.state, name)

    def _on_groups_table_cell_double_clicked(self, row: int, col: int):
        """Colour column only (0/1 are text-edited via itemChanged above)."""
        if col != 2 or row >= len(self.state.group_names):
            return
        name = self.state.group_names[row]
        current = self._group_colour(name)
        colour = QColorDialog.getColor(QColor(current), self, f"Colour for group '{name}'")
        if colour.isValid():
            self.state.group_colors[name] = colour.name()
            self._populate_groups_table()

    def _on_groups_table_item_changed(self, item: QTableWidgetItem):
        """
        Handle an in-place edit of a group's name (col 0) or match pattern
        (col 1). A name edit is a REAL rename now — sample_groups values
        ARE group names (no slot indirection left) — so every
        sample currently assigned to the old name is rewritten to the new
        one in the same operation.
        """
        row, col = item.row(), item.column()
        if row >= len(self.state.group_names):
            return
        old_name = self.state.group_names[row]

        if col == 0:
            new_name = item.text().strip()
            if not new_name or new_name == 'Unassigned':
                QMessageBox.warning(self, "Invalid Group Name",
                                    "Group name cannot be blank or 'Unassigned'.")
                self._populate_groups_table()
                return
            if new_name != old_name and new_name in self.state.group_names:
                QMessageBox.warning(self, "Duplicate Group Name",
                                    f"A group named '{new_name}' already exists.")
                self._populate_groups_table()
                return
            if new_name != old_name:
                for sp, g in self.state.sample_groups.items():
                    if g == old_name:
                        self.state.sample_groups[sp] = new_name
                if old_name in self.state.group_patterns:
                    self.state.group_patterns[new_name] = self.state.group_patterns.pop(old_name)
                if old_name in self.state.group_colors:
                    self.state.group_colors[new_name] = self.state.group_colors.pop(old_name)
                # 'Groups to Test' is a separate checked-name
                # list (state.testing_group_selection)
                if self.state.testing_group_selection and old_name in self.state.testing_group_selection:
                    self.state.testing_group_selection = [
                        new_name if g == old_name else g
                        for g in self.state.testing_group_selection
                    ]
                if self.state.reference_group == old_name:
                    self.state.reference_group = new_name
                self.state.group_names[row] = new_name
                self._populate_table()
                self._populate_test_groups_list()
                self._update_group_count_label()
        else:
            self.state.group_patterns[old_name] = item.text().strip()

    def _add_group(self):
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Add Group", "Group name:")
        if not ok:
            return
        name = name.strip()
        if not name or name == 'Unassigned':
            QMessageBox.warning(self, "Invalid Group Name",
                                "Group name cannot be blank or 'Unassigned'.")
            return
        if name in self.state.group_names:
            QMessageBox.warning(self, "Duplicate Group Name",
                                f"A group named '{name}' already exists.")
            return
        self.state.group_names.append(name)
        self._group_colour(name)   # seed a default colour immediately
        # A new group is ticked in 'Groups to Test' straight away. An empty
        # selection already means "every group", so only extend a real one.
        if self.state.testing_group_selection:
            self.state.testing_group_selection.append(name)
        self._populate_groups_table()
        self._refresh_group_combos()
        self._populate_test_groups_list()
        self._last_stats_data_key = None
        self._update_group_count_label()
        self._update_run_button()

    def _remove_selected_group(self):
        row = self.groups_table.currentRow()
        if row < 0 or row >= len(self.state.group_names):
            return
        name = self.state.group_names[row]
        n_assigned = sum(1 for g in self.state.sample_groups.values() if g == name)
        reply = QMessageBox.question(
            self, "Remove Group",
            f"Remove group '{name}'? {n_assigned} assigned sample(s) will "
            "become Unassigned.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        for sp, g in list(self.state.sample_groups.items()):
            if g == name:
                self.state.sample_groups[sp] = 'Unassigned'
        self.state.group_names.pop(row)
        self.state.group_patterns.pop(name, None)
        self.state.group_colors.pop(name, None)
        self.state.testing_group_selection = [
            g for g in self.state.testing_group_selection if g != name
        ]
        if self.state.reference_group == name:
            self.state.reference_group = ''
        self._populate_groups_table()
        self._populate_table()
        self._populate_test_groups_list()
        self._last_stats_data_key = None
        self._update_group_count_label()
        self._update_run_button()

    def _refresh_group_combos(self):
        """Update every per-sample combo box in the assignment table (e.g.
        after a group is renamed/added/removed) without rebuilding rows."""
        opts = self._group_options()
        for row in range(self.table.rowCount()):
            combo = self.table.cellWidget(row, 1)
            if combo is None:
                continue
            sp_item = self.table.item(row, 0)
            if sp_item is None:
                continue
            sp = sp_item.data(Qt.UserRole)
            current = self.state.sample_groups.get(sp, 'Unassigned')
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(opts)
            idx = combo.findText(current)
            combo.setCurrentIndex(max(0, idx))
            combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Groups to Test / contrast mode / pairing
    # ------------------------------------------------------------------

    def _populate_test_groups_list(self):
        """Rebuild the Groups-to-Test checklist from state.group_names,
        defaulting every group to checked the first time (empty selection)."""
        default_all = not self.state.testing_group_selection
        selection = (set(self.state.group_names) if default_all
                    else set(self.state.testing_group_selection))

        self.test_groups_list.blockSignals(True)
        self.test_groups_list.clear()
        for name in self.state.group_names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in selection else Qt.Unchecked)
            self.test_groups_list.addItem(item)
        self.test_groups_list.blockSignals(False)

        if default_all:
            self.state.testing_group_selection = list(self.state.group_names)

        self._refresh_reference_group_combo()

    def _on_test_group_checked_changed(self, _item: QListWidgetItem):
        self.state.testing_group_selection = [
            self.test_groups_list.item(i).text()
            for i in range(self.test_groups_list.count())
            if self.test_groups_list.item(i).checkState() == Qt.Checked
        ]
        self._last_stats_data_key = None
        self._refresh_reference_group_combo()
        self._update_run_button()

    def _refresh_reference_group_combo(self):
        """Populate the Reference combo from the currently checked test
        groups only -- a reference must itself be one of the groups in play."""
        checked = self.state.testing_group_selection or list(self.state.group_names)
        self.reference_group_combo.blockSignals(True)
        prev = self.state.reference_group
        self.reference_group_combo.clear()
        self.reference_group_combo.addItems(checked)
        idx = self.reference_group_combo.findText(prev)
        self.reference_group_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.reference_group_combo.blockSignals(False)
        self.state.reference_group = self.reference_group_combo.currentText()
        self.reference_group_combo.setEnabled(self.radio_reference.isChecked())

    def _on_contrast_mode_changed(self, _checked: bool):
        self.state.contrast_mode = 'reference' if self.radio_reference.isChecked() else 'pairwise'
        self.reference_group_combo.setEnabled(self.radio_reference.isChecked())
        self._last_stats_data_key = None
        self._update_run_button()

    def _on_reference_group_changed(self, text: str):
        self.state.reference_group = text
        self._last_stats_data_key = None

    def _on_paired_toggled(self, checked: bool):
        self.state.paired = checked
        self.pairing_variable_combo.setEnabled(checked)
        self._last_stats_data_key = None
        self._update_run_button()

    def _populate_pairing_variable_combo(self):
        cols = list(self.state.covariates.columns) if self.state.covariates is not None else []
        self.pairing_variable_combo.blockSignals(True)
        prev = self.state.pairing_variable
        self.pairing_variable_combo.clear()
        self.pairing_variable_combo.addItems(cols)
        idx = self.pairing_variable_combo.findText(prev)
        self.pairing_variable_combo.setCurrentIndex(max(0, idx))
        self.pairing_variable_combo.blockSignals(False)
        self.state.pairing_variable = self.pairing_variable_combo.currentText()
        self._populate_adjust_covariates_list()

    def _on_pairing_variable_changed(self, text: str):
        self.state.pairing_variable = text
        self._last_stats_data_key = None

    def _populate_adjust_covariates_list(self):
        """One checkable row per covariate column, ticked if it is in
        state.stats_adjust_covariates. Drops selections whose column no
        longer exists."""
        cols = list(self.state.covariates.columns) if self.state.covariates is not None else []
        self.state.stats_adjust_covariates = [
            c for c in self.state.stats_adjust_covariates if c in cols
        ]
        self.adjust_covariates_list.blockSignals(True)
        self.adjust_covariates_list.clear()
        for col in cols:
            kind = 'numeric' if col in self.state.numeric_covariates else 'categorical'
            item = QListWidgetItem(f"{col}  ({kind})")
            item.setData(Qt.UserRole, col)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(
                Qt.Checked if col in self.state.stats_adjust_covariates else Qt.Unchecked
            )
            self.adjust_covariates_list.addItem(item)
        self.adjust_covariates_list.blockSignals(False)

    def _on_adjust_covariate_changed(self, _item):
        self.state.stats_adjust_covariates = [
            self.adjust_covariates_list.item(i).data(Qt.UserRole)
            for i in range(self.adjust_covariates_list.count())
            if self.adjust_covariates_list.item(i).checkState() == Qt.Checked
        ]
        self._last_stats_data_key = None

    def _refresh_viewing_comparison_combo(self):
        """Populate 'Viewing comparison:' from state.stats_comparisons,
        set by the last Run Statistics call."""
        self.viewing_comparison_combo.blockSignals(True)
        prev = self.viewing_comparison_combo.currentText()
        self.viewing_comparison_combo.clear()
        labels = [f"{other} vs {base}" for base, other in self.state.stats_comparisons]
        self.viewing_comparison_combo.addItems(labels)
        idx = self.viewing_comparison_combo.findText(prev)
        self.viewing_comparison_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.viewing_comparison_combo.blockSignals(False)

    def _on_viewing_comparison_changed(self, _index: int):
        if (self.state.freq_results is not None or self.state.mfi_results is not None
                or self.state.counts_results is not None):
            self._draw_results()

    # ------------------------------------------------------------------
    # Auto-assign by name (regex matching)
    # ------------------------------------------------------------------

    def _auto_assign_by_name(self):
        """
        Regex-match each defined group's pattern against the filenames of
        the samples currently shown in the table. First matching pattern
        wins, in state.group_names order. This OVERWRITES every on-screen
        sample's current group (including manual picks and prior
        auto-assign results) so a re-run is always a clean, deterministic
        pass over the current patterns -- nothing about a sample's history
        can leave it stuck on a stale value. Patterns are separate from
        group names so a group named 'A' can't match every file,
        and only on-screen samples are touched.
        """
        patterns = {}
        for name in self.state.group_names:
            pat = self.state.group_patterns.get(name, '').strip()
            if not pat:
                continue
            try:
                patterns[name] = re.compile(pat, re.IGNORECASE)
            except re.error as e:
                QMessageBox.warning(self, "Invalid Regex", f"{name}: {e}")
                return
        _log.info("auto-assign: patterns=%r", {k: p.pattern for k, p in patterns.items()})
        if not patterns:
            QMessageBox.information(self, "Auto-assign",
                "Enter a filename match pattern for at least one group.")
            return

        # Only the samples currently in the table (the run-scoped view).
        shown = self._table_sample_paths()

        _log.info("auto-assign: table rows=%d, shown=%d", self.table.rowCount(), len(shown))
        for sp in shown:
            _log.info("  sample key=%r  filename=%r  current=%r",
                      sp, Path(sp).name, self.state.sample_groups.get(sp, 'Unassigned'))

        assigned_counts = {name: 0 for name in patterns}
        for sp in shown:
            fn = Path(sp).name
            self.state.sample_groups[sp] = 'Unassigned'
            for name, rx in patterns.items():
                if rx.search(fn):
                    self.state.sample_groups[sp] = name
                    assigned_counts[name] += 1
                    _log.info("  → %s: %r", name, fn)
                    break
            else:
                _log.info("  no match: %r", fn)

        self.refresh()                  # repopulate the SAME (training) view
        self._update_run_button()
        summary = ", ".join(f"{n} → {name}" for name, n in assigned_counts.items())
        self.stats_status_label.setText(f"Auto-assigned {summary}.")

    def _scoped_sample_groups(self) -> dict:
        """
        Return the {sample_path: group} mapping restricted to the samples
        belonging to the currently-selected run in the Stats run combo
        (its own archived 'training_sample_ids'), falling back to
        Configuration's live training-sample selection if no run is
        selected yet, and to all samples if that's empty too. This is the
        single source of truth for "which samples are in scope right now"
        -- refresh() uses it to build the table, and _populate_table()
        defaults to it so every no-argument caller (rename, remove group,
        Add Column, CSV import) stays scoped the same way instead of
        silently falling back to the whole experiment.
        """
        selected_run_id = self._run_combo.currentData() or self.state.stats_run_id
        run_training_ids = None
        if selected_run_id:
            for entry in list(self.state.dr_runs) + list(self.state.clustering_runs):
                if entry.get('run_id') == selected_run_id:
                    run_training_ids = entry.get('training_sample_ids')
                    break

        training = run_training_ids if run_training_ids else self.state.training_sample_ids
        if training:
            # training_sample_ids are relative paths; sample_groups keys may be
            # absolute paths.  Normalise by comparing Path.name and suffix matches.
            training_set = set(training)
            training_names = {Path(t).name for t in training}

            try:
                raw_subdir = self.controller.experiment.settings['raw'][
                    'raw_samples_subdirectory'
                ]
            except (KeyError, AttributeError):
                raw_subdir = None

            matched = {}
            for sp, grp in self.state.sample_groups.items():
                # Try: exact rel-path match
                if sp in training_set:
                    matched[sp] = grp
                    continue
                # Try: convert abs → rel, then match
                if raw_subdir:
                    try:
                        rel = str(Path(sp).relative_to(raw_subdir))
                        if rel in training_set:
                            matched[sp] = grp
                            continue
                    except ValueError:
                        pass
                # Try: filename match (last resort)
                if Path(sp).name in training_names:
                    matched[sp] = grp

            if not matched:
                # No training samples matched — show all
                all_samples = _non_control_sample_paths(self.controller)
                self.state.initialise_sample_groups(all_samples)
                matched = self.state.sample_groups
        else:
            all_samples = _non_control_sample_paths(self.controller)
            self.state.initialise_sample_groups(all_samples)
            matched = self.state.sample_groups

        return matched
    
    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self):
        """
        Rebuild the sample table showing only the samples belonging to the
        currently-selected run in the Stats run combo (its own archived
        'training_sample_ids' -- fixed at the time that run was computed),
        so the table always matches what Run Statistics/Auto-assign will
        actually operate on. Falls back to Configuration's live
        training-sample selection if no run is selected yet, and to all
        samples if that's empty too.
        """
        self._populate_groups_table()

        self._populate_test_groups_list()
        self._populate_pairing_variable_combo()
        self.chk_paired.blockSignals(True)
        self.chk_paired.setChecked(self.state.paired)
        self.chk_paired.blockSignals(False)
        self.pairing_variable_combo.setEnabled(self.state.paired)
        self.chk_treat.setChecked(self.state.stats_use_treat)
        self.chk_mfi_hierarchical.setChecked(self.state.stats_mfi_hierarchical)
        self.counts_method_combo.setCurrentIndex(
            max(self.counts_method_combo.findData(self.state.stats_counts_method), 0))
        self.normalisation_combo.setCurrentIndex(
            max(self.normalisation_combo.findData(self.state.stats_normalisation), 0))
        self.mfi_fc_spin.setValue(self.state.stats_mfi_threshold)
        self.radio_reference.blockSignals(True)
        self.radio_pairwise.blockSignals(True)
        if self.state.contrast_mode == 'pairwise':
            self.radio_pairwise.setChecked(True)
        else:
            self.radio_reference.setChecked(True)
        self.radio_reference.blockSignals(False)
        self.radio_pairwise.blockSignals(False)
        self.reference_group_combo.setEnabled(self.radio_reference.isChecked())

        matched = self._scoped_sample_groups()
        self._populate_table(matched)
        self._maybe_auto_suggest_groupings()
        self._populate_run_combo()
        self._sync_composition_to_run()
        self._populate_marker_roles_list()
        self._update_run_button()
        self._refresh_predictive_controls()
        self._update_stats_freshness()

        # Redraw plots if results are present and haven't been rendered yet
        # (_last_drawn_cluster_names == {} after load_state resets it) or if
        # cluster names changed since last draw (e.g. after a rename).
        _log.info("GroupsStatsTab.refresh: freq_results=%s mfi_results=%s counts_results=%s",
                  self.state.freq_results is not None,
                  self.state.mfi_results is not None,
                  self.state.counts_results is not None)
        if (self.state.freq_results is not None or self.state.mfi_results is not None
                or self.state.counts_results is not None):
            if (not self._last_drawn_cluster_names or
                    dict(self.state.cluster_names) != self._last_drawn_cluster_names):
                _log.info("GroupsStatsTab.refresh: calling _draw_results() "
                          "(last_drawn=%r stats_comparisons=%d)",
                          self._last_drawn_cluster_names, len(self.state.stats_comparisons))
                self._draw_results()
            else:
                _log.info("GroupsStatsTab.refresh: SKIPPING _draw_results() -- "
                          "_last_drawn_cluster_names (%r) already matches current state",
                          self._last_drawn_cluster_names)

        # Composition Barplot is independent of Run Statistics -- redraw it
        # once per load if its persisted data is present and it isn't
        # already showing.
        if self.state.composition_df is not None and not self._has_results_tab('composition_barplot'):
            self.composition_pct_chk.setChecked(self.state.composition_as_pct)
            self.composition_by_group_chk.setChecked(self.state.composition_group_var == 'group')
            fig = self._make_composition_figure(
                self.state.composition_df, as_pct=self.state.composition_as_pct,
                run_label=self.state.composition_run_label, names=self.state.composition_names,
                colors=self.state.composition_colors,
                group_var=self.state.composition_group_var,
            )
            self._add_results_tab(
                fig, "Composition", "composition_barplot",
                maker=self._make_composition_figure,
                maker_kwargs=dict(comp_df=self.state.composition_df,
                                  as_pct=self.state.composition_as_pct,
                                  run_label=self.state.composition_run_label,
                                  names=self.state.composition_names,
                                  colors=self.state.composition_colors,
                                  group_var=self.state.composition_group_var),
                key="composition_barplot",
            )

        # PCA controls -- sync widget state from self.state
        # now, since load_state() restores QSettings into self.state but
        # runs after these widgets were already built (with construction-
        # time defaults), so nothing else re-syncs them. Signals blocked
        # so this doesn't itself fire _on_pca_option_changed.
        for w, val, setter in (
            (self.pca_chk_freq, self.state.pca_use_freq, 'setChecked'),
            (self.pca_chk_counts, self.state.pca_use_counts, 'setChecked'),
            (self.pca_chk_mfi, self.state.pca_use_mfi, 'setChecked'),
            (self.pca_chk_loadings, self.state.pca_show_loadings, 'setChecked'),
            (self.pca_n_loadings_spin, self.state.pca_n_loadings, 'setValue'),
            (self.pca_chk_label_loadings, self.state.pca_label_loadings, 'setChecked'),
            (self.pca_chk_label_points, self.state.pca_label_points, 'setChecked'),
            (self.pca_point_size_spin, int(self.state.pca_point_size), 'setValue'),
            (self.pca_arrow_lw_spin, self.state.pca_arrow_lw, 'setValue'),
            (self.pca_axis_fontsize_spin, self.state.pca_axis_fontsize, 'setValue'),
            (self.pca_chk_grid, self.state.pca_show_grid, 'setChecked'),
        ):
            w.blockSignals(True)
            getattr(w, setter)(val)
            w.blockSignals(False)
        self._update_pca_arrow_color_btn()
        self._update_pca_source_availability()

        # Redraw the PCA tab too, same "persisted + not already showing"
        # rule as the Composition Barplot above.
        pca_result = self._pca_result_from_state()
        if pca_result is not None and not self._has_results_tab('sample_pca'):
            fig_pca = self._make_pca_figure(pca_result, run_label=self.state.pca_run_label)
            self._add_results_tab(
                fig_pca, "Sample PCA", "sample_pca",
                maker=self._make_pca_figure,
                maker_kwargs=dict(pca_result=pca_result, run_label=self.state.pca_run_label),
                key="sample_pca",
            )

    # ------------------------------------------------------------------
    # Table population
    # ------------------------------------------------------------------

    def _populate_table(self, sample_groups: dict | None = None):
        """Rebuild table rows AND covariate columns from sample_groups.
        Column count is dynamic -- one per state.covariates column, so
        this must rebuild the header every call, not just the rows.
        Defaults to the run-scoped set (_scoped_sample_groups()), NOT the
        raw state.sample_groups, so every no-argument caller (rename,
        remove group, Add Column, CSV import) stays consistent with what
        refresh() itself shows."""
        if sample_groups is None:
            sample_groups = self._scoped_sample_groups()

        try:
            raw_subdir = self.controller.experiment.settings['raw'][
                'raw_samples_subdirectory'
            ]
        except (KeyError, AttributeError):
            raw_subdir = None

        def _to_rel(sp):
            if raw_subdir:
                try:
                    return str(Path(sp).relative_to(raw_subdir))
                except ValueError:
                    pass
            return sp

        covariate_cols = (
            list(self.state.covariates.columns) if self.state.covariates is not None else []
        )
        self.table.setColumnCount(2 + len(covariate_cols))
        self.table.setHorizontalHeaderLabels(['Sample', 'Group'] + covariate_cols)
        # Every column is Interactive so it can be dragged to any width, and
        # the table scrolls sideways when the columns are wider than the
        # view. Initial widths come from font metrics rather than from the
        # cell widgets' size hints, which are only measured on first show.
        opts = self._group_options()
        fm = self.table.fontMetrics()
        longest_opt_width = max((fm.horizontalAdvance(o) for o in opts), default=0)
        longest_name_width = max(
            (fm.horizontalAdvance(Path(sp).name) for sp in sample_groups), default=0)
        self.table.setColumnWidth(0, min(longest_name_width + 24, SAMPLE_COLUMN_MAX_WIDTH))
        self.table.setColumnWidth(1, longest_opt_width + 40)  # padding for combo chrome/arrow
        for c, col_name in enumerate(covariate_cols, start=2):
            self.table.setColumnWidth(
                c, max(COVARIATE_COLUMN_MIN_WIDTH, fm.horizontalAdvance(col_name) + 32))
        self._table_header.set_checks(
            {c: col_name in self.state.numeric_covariates
             for c, col_name in enumerate(covariate_cols, start=2)})

        self.table.setRowCount(0)
        for sample_path, group in sorted(
            sample_groups.items(), key=lambda kv: Path(kv[0]).name.lower()
        ):
            row = self.table.rowCount()
            self.table.insertRow(row)

            name_item = QTableWidgetItem(Path(sample_path).name)
            name_item.setData(Qt.UserRole, sample_path)
            name_item.setToolTip(sample_path)
            self.table.setItem(row, 0, name_item)

            combo = QComboBox()
            combo.addItems(opts)
            idx = combo.findText(group)
            combo.setCurrentIndex(max(0, idx))
            combo.currentTextChanged.connect(
                lambda text, sp=sample_path: self._on_group_changed(sp, text)
            )
            self.table.setCellWidget(row, 1, combo)

            rel = _to_rel(sample_path)
            for c, col_name in enumerate(covariate_cols):
                edit = QLineEdit()
                if self.state.covariates is not None and rel in self.state.covariates.index:
                    edit.setText(str(self.state.covariates.loc[rel, col_name]))
                edit.editingFinished.connect(
                    lambda le=edit, r=rel, cn=col_name: self._on_covariate_edited(r, cn, le.text())
                )
                self.table.setCellWidget(row, 2 + c, edit)

        # Size to fit every row -- no splitter/drag-handle in this tab, so
        # the table just grows to hold its full content; this tab's own
        # outer QScrollArea (see _build_ui) scrolls the whole page if that
        # makes it taller than the viewport, instead of nesting a second
        # scrollbar inside this one.
        n_rows = self.table.rowCount()
        self.table.setMinimumHeight(28 * n_rows + 40)
        self._update_group_count_label()

    def _on_covariate_numeric_toggled(self, section: int, checked: bool):
        """Enter the covariate under *section* as a continuous number (or as
        categories). A column holding text cannot be numeric."""
        name = self.table.horizontalHeaderItem(section).text()
        if checked and self.state.covariates is not None and name in self.state.covariates.columns:
            filled = [v for v in self.state.covariates[name].astype(str).str.strip() if v]
            if filled and drc_stats.covariate_kind(filled) != 'numeric':
                QMessageBox.warning(
                    self, "Numeric covariate",
                    f"'{name}' has values that are not numbers, so it cannot be numeric.")
                self._table_header.set_checks(
                    {c: self.table.horizontalHeaderItem(c).text() in self.state.numeric_covariates
                     for c in self._table_header.checks()})
                return
        self.state.numeric_covariates = [c for c in self.state.numeric_covariates if c != name]
        if checked:
            self.state.numeric_covariates.append(name)
        self._populate_adjust_covariates_list()
        self._last_stats_data_key = None

    def _add_covariate_column(self):
        """Add a new covariate/pairing column directly in-app -- fill it
        in per sample below, then select it as 'Pairing variable'."""
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Add Column", "Column name (e.g. 'donor'):")
        name = name.strip()
        if not ok or not name:
            return
        if self.state.covariates is not None and name in self.state.covariates.columns:
            QMessageBox.warning(self, "Add Column", f"A column named '{name}' already exists.")
            return
        if self.state.covariates is None:
            self.state.covariates = pd.DataFrame(index=pd.Index([], dtype=str))
        self.state.covariates[name] = ''
        self._populate_table()
        self._populate_pairing_variable_combo()

    def _remove_covariate_column(self):
        """Delete one covariate column entirely (e.g. an empty leftover
        from a mistaken '+ Add Column', or one superseded by a Suggest
        Groupings result). Clears it from the Pairing variable selector
        too if it was the one selected."""
        from PySide6.QtWidgets import QInputDialog
        cols = list(self.state.covariates.columns) if self.state.covariates is not None else []
        if not cols:
            QMessageBox.information(self, "Remove Column", "There are no covariate columns to remove.")
            return
        name, ok = QInputDialog.getItem(
            self, "Remove Column", "Column to remove:", cols, editable=False
        )
        if not ok or not name:
            return
        reply = QMessageBox.question(
            self, "Remove Column",
            f"Remove covariate column '{name}'? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.state.covariates = self.state.covariates.drop(columns=[name])
        self.state.numeric_covariates = [c for c in self.state.numeric_covariates if c != name]
        if self.state.pairing_variable == name:
            self.state.pairing_variable = ''
            if self.state.paired:
                self.state.paired = False
                self.chk_paired.setChecked(False)
        self._last_stats_data_key = None
        self._populate_table()
        self._populate_pairing_variable_combo()

    def _suggest_groupings(self):
        """
        Scan the on-screen sample names (and, when the experiment's Sample
        name source is not the filename, the display names computed at
        import) for repeated, delimiter-separated tokens, and let the user
        choose which to apply as the Group column or as covariate/pairing
        columns.
        """
        sample_paths = self._table_sample_paths()
        if len(sample_paths) < 2:
            QMessageBox.information(
                self, "Suggest Groupings",
                "Need at least two samples on screen to detect a pattern."
            )
            return
        display_names = self.controller.experiment.samples.get('all_samples', {})
        result = suggest_groupings_from_names(sample_paths, display_names)
        if not result['suggestions']:
            QMessageBox.information(self, "Suggest Groupings", sgd.no_suggestion_message(result))
            return
        chosen = sgd.choose_groupings(self, result)
        if chosen:
            self._apply_groupings(chosen, sample_paths)
            self._show_auto_groupings_note(None)

    def _maybe_auto_suggest_groupings(self):
        """
        Apply groupings suggested from the sample names when none of the
        on-screen samples has a group yet: the detected group, and the
        detected pairing variable (selected for a paired design). Runs once
        per sample list and never overwrites existing assignments; the note
        above the table says what was applied and offers Undo.
        """
        sample_paths = self._table_sample_paths()
        if len(sample_paths) < 2:
            return
        if any(self.state.sample_groups.get(sp) in self.state.group_names for sp in sample_paths):
            return
        key = tuple(sample_paths)
        if key == self._auto_groupings_key:
            return
        self._auto_groupings_key = key
        display_names = self.controller.experiment.samples.get('all_samples', {})
        chosen = sgd.automatic_groupings(suggest_groupings_from_names(sample_paths, display_names))
        if not chosen:
            return
        st = self.state
        self._auto_groupings_undo = (
            list(st.group_names), dict(st.sample_groups),
            None if st.covariates is None else st.covariates.copy(),
            st.pairing_variable, st.paired,
        )
        self._apply_groupings(chosen, sample_paths, select_pairing=True)
        self._show_auto_groupings_note(sgd.describe_groupings(chosen))

    def _show_auto_groupings_note(self, text: str | None):
        """Show (text) or hide (None) the automatic-groupings note and Undo."""
        if text is None:
            self._auto_groupings_undo = None
        self._auto_groupings_label.setText(text or '')
        self._auto_groupings_label.setVisible(text is not None)
        self._auto_groupings_undo_btn.setVisible(text is not None and self._auto_groupings_undo is not None)

    def _undo_auto_groupings(self):
        """Restore the groups, covariates and pairing from before the
        automatic groupings were applied."""
        if self._auto_groupings_undo is None:
            return
        groups, sample_groups, covariates, pairing_variable, paired = self._auto_groupings_undo
        st = self.state
        st.group_names = groups
        st.sample_groups = sample_groups
        st.covariates = covariates
        st.pairing_variable = pairing_variable
        st.paired = paired
        self.chk_paired.blockSignals(True)
        self.chk_paired.setChecked(paired)
        self.chk_paired.blockSignals(False)
        self.pairing_variable_combo.setEnabled(paired)
        self._show_auto_groupings_note(None)
        self._refresh_after_groupings()

    def _apply_groupings(self, chosen: list[dict], sample_paths: list[str],
                         select_pairing: bool = False):
        """
        Write chosen groupings into state: a 'group' choice replaces
        state.group_names and the on-screen samples' groups; every other
        choice becomes a covariate column (state.covariates is keyed by
        path relative to the raw samples folder). A 'pairing' choice is
        made the pairing variable; with ``select_pairing`` the paired
        design is also switched on.
        """
        if self.state.covariates is None:
            self.state.covariates = pd.DataFrame(index=pd.Index([], dtype=str))

        raw_subdir = self.controller.experiment.settings['raw'].get('raw_samples_subdirectory')

        def _to_rel(sp):
            if raw_subdir:
                try:
                    return str(Path(sp).relative_to(raw_subdir))
                except ValueError:
                    pass
            return sp

        for choice in chosen:
            role, name = choice['role'], choice['name']
            if role == 'group':
                distinct_vals = sorted(set(choice['values'].values()))
                self.state.group_names = distinct_vals
                for sp in sample_paths:
                    self.state.sample_groups[sp] = choice['values'].get(sp, 'Unassigned')
                for val in distinct_vals:
                    self._group_colour(val)  # seed default colours
            else:
                if name in self.state.covariates.columns:
                    name = f"{name}_2"
                self.state.covariates[name] = ''
                for sp, val in choice['values'].items():
                    rel = _to_rel(sp)
                    if rel not in self.state.covariates.index:
                        self.state.covariates.loc[rel] = ''
                    self.state.covariates.loc[rel, name] = val
                if role == 'pairing':
                    self.state.pairing_variable = name
                    if select_pairing:
                        self.state.paired = True
                        self.chk_paired.blockSignals(True)
                        self.chk_paired.setChecked(True)
                        self.chk_paired.blockSignals(False)
                        self.pairing_variable_combo.setEnabled(True)
        self._refresh_after_groupings()

    def _refresh_after_groupings(self):
        self._last_stats_data_key = None
        self._populate_groups_table()
        self._refresh_group_combos()
        self._populate_table()
        self._populate_pairing_variable_combo()
        self._populate_test_groups_list()
        self._update_group_count_label()
        self._update_run_button()

    def _on_covariate_edited(self, rel: str, col_name: str, value: str):
        """Write one sample's value for one covariate column directly."""
        if self.state.covariates is None:
            self.state.covariates = pd.DataFrame(index=pd.Index([], dtype=str))
        if col_name not in self.state.covariates.columns:
            self.state.covariates[col_name] = ''
        if rel not in self.state.covariates.index:
            self.state.covariates.loc[rel] = ''
        self.state.covariates.loc[rel, col_name] = value
        self._last_stats_data_key = None

    def _on_group_changed(self, sample_path: str, group_name: str):
        self.state.sample_groups[sample_path] = group_name
        self._update_run_button()
        self._update_group_count_label()

    def _table_sample_paths(self) -> list[str]:
        """Return the sample_groups keys for every row currently shown in
        the Sample Group Assignment table -- i.e. the samples belonging to
        whichever run the table is presently scoped to (see refresh()).
        This is deliberately NOT all of state.sample_groups, which can
        carry entries for other runs and stale/removed samples."""
        return [self.table.item(r, 0).data(Qt.UserRole)
                for r in range(self.table.rowCount())
                if self.table.item(r, 0) is not None]

    def _update_group_count_label(self):
        shown = self._table_sample_paths()
        counts = {name: 0 for name in self.state.group_names}
        unassigned = 0
        for sp in shown:
            g = self.state.sample_groups.get(sp, 'Unassigned')
            if g in counts:
                counts[g] += 1
            else:
                unassigned += 1
        parts = [f"{name}: {counts.get(name, 0)}" for name in self.state.group_names]
        parts.append(f"Unassigned: {unassigned}")
        self._group_count_label.setText("  |  ".join(parts))

    def _selected_run_kind(self) -> str | None:
        """Return 'dr' / 'clustering' for the current combo selection, or
        None if nothing valid is selected.  Deliberately does NOT hydrate —
        'kind' is a manifest field present on every entry regardless of
        hydration state, so this is safe to call on every combo change."""
        run_id = self._run_combo.currentData()
        if run_id is None:
            return None
        for entry in self.state.dr_runs:
            if entry.get('run_id') == run_id:
                return 'dr'
        for entry in self.state.clustering_runs:
            if entry.get('run_id') == run_id:
                return 'clustering'
        return None

    def _update_run_button(self):
        shown = set(self._table_sample_paths())
        any_clustering = bool(self.state.clustering_runs) or bool(self.state.cluster_labels)
        dr_only_selected = self._selected_run_kind() == 'dr'
        n_group_ok = self.state.n_group_stats_runnable(shown)
        runnable = n_group_ok and any_clustering and not dr_only_selected
        stats_running = self._stats_worker is not None and self._stats_worker.isRunning()
        self.run_stats_btn.setEnabled(runnable and not self._analysis_busy())
        self.composition_btn.setEnabled(runnable and not stats_running)
        if hasattr(self, 'pca_btn'):
            self.pca_btn.setEnabled(runnable and not stats_running)
            self._update_pca_source_availability()
        if dr_only_selected:
            self.run_stats_btn.setToolTip(
                "This is a DR run — it has no cluster labels.  Select or "
                "run a clustering run to compute statistics."
            )
        elif not any_clustering:
            self.run_stats_btn.setToolTip("Run clustering first.")
        elif not n_group_ok:
            counts = self.state.n_per_group(shown)
            selection = self.state.testing_group_selection or self.state.group_names
            detail = ", ".join(f"{name}={counts.get(name, 0)}" for name in selection)
            self.run_stats_btn.setToolTip(
                f"Requires ≥ 3 samples in at least two checked 'Groups to "
                f"Test'.  Currently: {detail}."
            )
        else:
            self.run_stats_btn.setToolTip("")

    def _update_pca_source_availability(self):
        """Grey out whichever 'Build from' checkbox has no computed data
        yet -- Frequencies/Counts/MFIs only become usable once Run
        Statistics has actually computed that source."""
        self.pca_chk_freq.setEnabled(
            self.state.freq_df is not None and not self.state.freq_df.empty
        )
        self.pca_chk_counts.setEnabled(
            self.state.counts_df is not None and not self.state.counts_df.empty
        )
        self.pca_chk_mfi.setEnabled(
            self.state.mfi_df is not None and not self.state.mfi_df.empty
        )

    # ------------------------------------------------------------------
    # Run combo management
    # ------------------------------------------------------------------

    def _populate_run_combo(self):
        """
        Rebuild the Differential Statistics run combo -- clustering runs
        only. DR runs have no cluster labels, so Run Statistics /
        Composition / PCA can never operate on one. A run is included
        only if its recorded training sample set overlaps the samples
        currently assigned to a group.

        Uses each entry's 'training_sample_ids' manifest field, NEVER
        'labels'/'embeddings' — this works from metadata alone (every
        entry has it, hydrated or not), so populating this combo never
        forces a run to be loaded from disk.  Only actually selecting one
        does that (see _selected_run_entry).

        Combo items are keyed by run_id (userData), not label text —
        labels are user-editable and not guaranteed
        unique, so text-matching would be fragile.
        """
        assigned = {sp for sp, g in self.state.sample_groups.items()
                    if g != 'Unassigned'}

        try:
            raw_subdir = self.controller.experiment.settings['raw'][
                'raw_samples_subdirectory'
            ]
        except (KeyError, AttributeError):
            raw_subdir = None

        def _to_rel(sp):
            if raw_subdir:
                try:
                    return str(Path(sp).relative_to(raw_subdir))
                except ValueError:
                    pass
            return sp

        assigned_rel = {_to_rel(sp) for sp in assigned}

        self._run_combo.blockSignals(True)
        prev_run_id = self._run_combo.currentData()
        self._run_combo.clear()

        all_runs = sorted(
            list(self.state.clustering_runs),
            key=lambda e: e.get('timestamp', ''),
        )
        for entry in all_runs:
            run_keys = set(entry.get('training_sample_ids', []))
            if not run_keys.isdisjoint(assigned_rel) or not assigned_rel:
                self._run_combo.addItem(entry['label'], entry.get('run_id'))

        # If no archived runs, show a placeholder (userData=None; every
        # other method treats currentData() is None as "nothing selected").
        if self._run_combo.count() == 0:
            self._run_combo.addItem("(no runs yet)", None)

        # On first populate after load, prefer the run stats were computed
        # for (by run_id).  On subsequent repopulations (group changes
        # etc.) keep whatever was already selected in this session.
        preferred_id = self.state.stats_run_id or prev_run_id
        idx = self._run_combo.findData(preferred_id) if preferred_id else -1
        if idx >= 0:
            self._run_combo.setCurrentIndex(idx)
        else:
            self._run_combo.setCurrentIndex(self._run_combo.count() - 1)

        self._run_combo.blockSignals(False)

    def _clear_stale_run_results(self):
        """
        drop Freq/Counts/MFI/PCA Stats results (tied to
        state.stats_run_id) if their source clustering run has been
        deleted from the archive. Called from PluginWidget._on_runs_
        changed on every rename/delete; rename leaves run_id untouched
        (only the label changes) so this is a no-op then -- only a
        genuine delete removes a run_id from state.clustering_runs,
        which is what actually triggers a clear here. '' (the "Active
        (unsaved)" cluster_labels case -- see _resolve_stats_source) is
        never archived and therefore never stale by this check.

        The Composition Barplot is handled separately, by
        _sync_composition_to_run() -- it is also meant to go stale on a
        plain combo-selection change, not just a delete.
        """
        valid_ids = {e.get('run_id') for e in self.state.clustering_runs}

        if self.state.stats_run_id and self.state.stats_run_id not in valid_ids:
            for key in ('freq_heatmap', 'freq_volcano', 'counts_heatmap',
                       'counts_volcano', 'mfi_heatmap', 'mfi_volcano', 'mfi_clusters',
                       'sample_pca', 'pred_performance', 'pred_importance', 'pred_panel'):
                self._remove_results_tab_by_key(key)
            self.state.predictive_result = None
            self.state.freq_results = None
            self.state.counts_results = None
            self.state.mfi_results = None
            self.state.freq_df = None
            self.state.counts_df = None
            self.state.mfi_df = None
            self.state.mfi_sample_df = None
            self.state.stats_all_rel = []
            self.state.stats_group_vec = []
            self.state.stats_comparisons = []
            self.state.stats_run_label = ''
            self.state.stats_run_id = ''
            # Sample PCA is built from freq_df/counts_df/mfi_df above --
            # stale along with them; it has no run_id of its own.
            self.state.pca_scores_df = None
            self.state.pca_loadings_df = None
            self.state.pca_explained_variance = (0.0, 0.0)
            self.state.pca_run_label = ''
            self.state.pca_groups = []
            self.state.pca_sources = []
            self._last_stats_data_key = None
            self.stats_status_label.setText(
                "Previous results' clustering run was deleted -- run "
                "statistics again."
            )
            self.stats_status_label.setStyleSheet("color: #d9822b;")

    def _sync_composition_to_run(self):
        """
        Unlike Freq/MFI/Counts/PCA (see
        _clear_stale_run_results), the Composition Barplot is meant to go
        stale on a plain combo-selection change too, not just a run
        deletion -- a deletion just happens to also change what's
        selected here, once the deleted entry falls out of the combo.
        Called from _on_run_combo_changed (user picks a different run),
        refresh() (tab reactivation), and PluginWidget._on_runs_changed
        (AFTER _populate_run_combo() has re-settled on whatever's still
        there post-delete/rename).
        """
        current_run_id = self._run_combo.currentData()

        if self.state.composition_run_id and self.state.composition_run_id != current_run_id:
            _log.info("_sync_composition_to_run: clearing composition_barplot "
                      "(was %r)", self.state.composition_run_id)
            self._remove_results_tab_by_key('composition_barplot')
            self.state.composition_df = None
            self.state.composition_run_label = ''
            self.state.composition_run_id = ''
            self.state.composition_names = {}
            self.state.composition_colors = {}

    def _selected_run_entry(self) -> dict | None:
        """
        Return the (hydrated) run entry for the current combo selection —
        may be a DR or a clustering run.  Searched by run_id, not label
        text.  This is the one place a combo selection actually triggers
        a disk read: hydrate_run() is a no-op if the entry was already
        hydrated earlier in this session.
        """
        run_id = self._run_combo.currentData()
        if run_id is None:
            return None
        for entry in self.state.clustering_runs:
            if entry.get('run_id') == run_id:
                return drc_run_archive.hydrate_run(self.controller, entry)
        for entry in self.state.dr_runs:
            if entry.get('run_id') == run_id:
                return drc_run_archive.hydrate_run(self.controller, entry)
        return None

    def _on_run_combo_changed(self, _index: int):
        """
        Called when the user selects a different run in the unified combo.

        Compares by run_id, not label text (labels are user-editable and
        not guaranteed unique).  Also re-evaluates the Run Statistics
        button on every change, since selecting a DR-only run must grey
        it out immediately even if some other clustering run exists
        elsewhere in the list.

        Results are linked to the run they were computed against via
        state.stats_run_id, not deleted just because a different run is
        now showing in this combo -- selecting a different run only means
        "Run Statistics" will (re)compute against that run when clicked;
        it doesn't retroactively invalidate a previous run's results,
        which are still valid for that run and still exportable.

        The Composition Barplot is the exception -- it persists without a
        "Run Statistics" click, so a stale one left showing for a run no
        longer selected here is easy to miss. _sync_composition_to_run()
        clears it the moment this combo's selection stops matching the
        run it came from.
        """
        run_id = self._run_combo.currentData()
        self._update_run_button()
        self._sync_composition_to_run()
        if run_id is None:
            return
        have_results = (self.state.freq_results is not None
                        or self.state.mfi_results is not None
                        or self.state.counts_results is not None)
        if run_id == self.state.stats_run_id:
            if have_results:
                self._draw_results()
            return
        # Different run selected -- existing results (if any) still belong
        # to state.stats_run_id and are left alone; just tell the user
        # this combo's selection no longer matches what's currently shown.
        if have_results:
            self.stats_status_label.setText(
                f"Showing results for a different run ({self.state.stats_run_label or 'previous run'}) "
                "— click 'Run Statistics' to compute for this run."
            )
            self.stats_status_label.setStyleSheet("color: #d9822b;")
        else:
            self.stats_status_label.setText(
                "No results yet for this run — click 'Run Statistics' to compute."
            )
            self.stats_status_label.setStyleSheet("color: orange;")

    # ------------------------------------------------------------------
    # Marker roles — type (clustering) vs state (tested)
    # ------------------------------------------------------------------

    def _populate_marker_roles_list(self):
        """
        Rebuild the marker-role checkboxes from state.selected_channels.
        A channel with no entry yet in state.marker_roles defaults to
        'state' ("activation") for every channel — there is no automatic
        categorisation yet (a marker_database.csv-style table is planned
        for that later), so nothing is assumed to be a clustering ("type")
        marker by default. An existing explicit entry (including one the
        user overrode) is never overwritten just by refreshing.
        """
        channels = [c for c in self.state.selected_channels
                   if c not in drc_pipeline.META_CHANNELS]

        # Rebuild from scratch each time -- channel selection can change
        # between refreshes (Configuration tab), so stale checkboxes for
        # channels no longer selected need to be dropped, not just added to.
        while self.marker_roles_grid.count():
            grid_item = self.marker_roles_grid.takeAt(0)
            w = grid_item.widget()
            if w is not None:
                w.deleteLater()
        self.marker_roles_checkboxes.clear()

        labels = _antigen_dash_labels(self.controller)
        for grid_idx, ch in enumerate(channels):
            role = self.state.marker_roles.get(ch)
            if role is None:
                role = 'state'
                self.state.marker_roles[ch] = role
            cb = QCheckBox(labels.get(ch, ch))
            # hecked now means "included" ('state'), not
            # "type"/excluded, so ticked boxes visually match what's
            # actually tested.
            cb.setChecked(role == 'state')
            cb.toggled.connect(lambda checked, c=ch: self._on_marker_role_changed(c, checked))
            self.marker_roles_checkboxes[ch] = cb
            row, col = divmod(grid_idx, 4)
            self.marker_roles_grid.addWidget(cb, row, col)

    def _on_marker_role_changed(self, ch: str, checked: bool):
        # checked = included/'state', unchecked = excluded/'type'.
        self.state.marker_roles[ch] = 'state' if checked else 'type'

    def _on_include_type_markers_toggled(self, checked: bool):
        """Ticking 'Include clustering (type) markers' also
        ticks every channel checkbox above, so the two controls can't
        disagree about which channels are actually included."""
        if checked:
            for cb in self.marker_roles_checkboxes.values():
                cb.setChecked(True)

    def _reset_marker_roles_to_defaults(self):
        """
        Reset every selected channel's role back to 'state' ("activation")
        — the only default until automatic categorisation exists (see
        _populate_marker_roles_list). Overwrites any per-channel overrides,
        unlike _populate_marker_roles_list()'s lazy fill-in-the-gaps
        behaviour.
        """
        channels = [c for c in self.state.selected_channels
                   if c not in drc_pipeline.META_CHANNELS]
        self.state.marker_roles = {ch: 'state' for ch in channels}
        self._populate_marker_roles_list()

    # ------------------------------------------------------------------
    # CSV import / export (assignments)
    # ------------------------------------------------------------------

    def _import_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Group Assignments", "", "CSV files (*.csv)"
        )
        if not path:
            return
        try:
            df = pd.read_csv(path)
            if 'sample' not in df.columns or 'group' not in df.columns:
                QMessageBox.warning(
                    self, "Import Error",
                    "CSV must contain 'sample' and 'group' columns."
                )
                return
            valid = set(self.state.group_names) | {'Unassigned'}
            unknown_names = set()
            # Any column beyond sample/group/full_path is a
            # covariate (e.g. donor ID for paired testing)
            covariate_cols = [c for c in df.columns if c not in ('sample', 'group', 'full_path')]
            try:
                raw_subdir = self.controller.experiment.settings['raw'][
                    'raw_samples_subdirectory'
                ]
            except (KeyError, AttributeError):
                raw_subdir = None
            cov_rows = {}
            for _, row in df.iterrows():
                sample = str(row['sample'])
                group = str(row['group']).strip()
                match = next(
                    (sp for sp in self.state.sample_groups
                     if Path(sp).name == sample or sp == sample),
                    None
                )
                if match is None:
                    continue
                if group not in valid:
                    unknown_names.add(group)
                else:
                    self.state.sample_groups[match] = group
                if covariate_cols:
                    rel = match
                    if raw_subdir:
                        try:
                            rel = str(Path(match).relative_to(raw_subdir))
                        except ValueError:
                            pass
                    cov_rows[rel] = {c: str(row[c]) for c in covariate_cols}
            if cov_rows:
                new_cov = pd.DataFrame.from_dict(cov_rows, orient='index')
                self.state.covariates = (
                    new_cov if self.state.covariates is None
                    else self.state.covariates.combine_first(new_cov).astype(str)
                )
                self._populate_pairing_variable_combo()
            self._populate_table()
            self._update_run_button()
            if unknown_names:
                QMessageBox.warning(
                    self, "Unrecognised Groups",
                    "These group names in the CSV don't match a defined "
                    "group and were skipped (add the group first, then "
                    "re-import): " + ", ".join(sorted(unknown_names))
                )
        except Exception as e:
            QMessageBox.warning(self, "Import Error", str(e))

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Group Assignments", "", "CSV files (*.csv)"
        )
        if not path:
            return
        path = _with_extension(path, '.csv')
        try:
            try:
                raw_subdir = self.controller.experiment.settings['raw'][
                    'raw_samples_subdirectory'
                ]
            except (KeyError, AttributeError):
                raw_subdir = None
            rows = []
            for sp, grp in self.state.sample_groups.items():
                rel = sp
                if raw_subdir:
                    try:
                        rel = str(Path(sp).relative_to(raw_subdir))
                    except ValueError:
                        pass
                row = {'sample': Path(sp).name, 'full_path': sp, 'group': grp}
                if self.state.covariates is not None and rel in self.state.covariates.index:
                    row.update(self.state.covariates.loc[rel].to_dict())
                rows.append(row)
            pd.DataFrame(rows).to_csv(path, index=False)
            QMessageBox.information(self, "Exported", f"Saved to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _run_statistics(self):
        """
        Launch background differential statistics — or, if the same run and
        group assignment already produced results and only the
        significance thresholds changed, just re-flag significance and
        replot without refitting.
        """
        if self._analysis_busy():
            return

        resolved = self._resolve_stats_source()
        if resolved is None:
            return
        labels_for_stats, run_label, run_id, names_for_stats = resolved

        run_freq   = self.chk_freq.isChecked()
        run_counts = self.chk_counts.isChecked()
        run_mfi    = self.chk_mfi.isChecked()
        if not run_freq and not run_counts and not run_mfi:
            QMessageBox.warning(self, "Nothing Selected",
                                "Select at least one statistic to compute.")
            return

        if self.state.paired and self.state.pairing_variable:
            pv = self.state.pairing_variable
            try:
                group_rel = drc_stats.resolve_test_groups(
                    self.controller, self.state, cluster_labels_override=labels_for_stats
                )
            except Exception:
                group_rel = {}
            qualifying = [g for g in (self.state.testing_group_selection or self.state.group_names)
                         if g in group_rel]
            all_rel = [rel for g in qualifying for rel in group_rel[g]]
            missing = [
                rel for rel in all_rel
                if self.state.covariates is None
                or pv not in self.state.covariates.columns
                or rel not in self.state.covariates.index
                or not str(self.state.covariates.loc[rel, pv]).strip()
            ]
            if missing:
                names = "\n".join(Path(r).stem for r in missing[:15])
                more = "\n…" if len(missing) > 15 else ""
                QMessageBox.warning(
                    self, "Missing Pairing Values",
                    f"'Paired design' is checked, but {len(missing)} sample(s) "
                    f"have no value for pairing variable '{pv}':\n\n{names}{more}\n\n"
                    "Fill these in on the Sample Group Assignment table, or "
                    "uncheck 'Paired design'."
                )
                return

        adjust_covariates = [
            c for c in self.state.stats_adjust_covariates
            if not (self.state.paired and c == self.state.pairing_variable)
        ]
        if adjust_covariates:
            try:
                group_rel = drc_stats.resolve_test_groups(
                    self.controller, self.state, cluster_labels_override=labels_for_stats
                )
            except Exception:
                group_rel = {}
            qualifying = [g for g in (self.state.testing_group_selection or self.state.group_names)
                         if g in group_rel]
            all_rel = [rel for g in qualifying for rel in group_rel[g]]
            missing = drc_stats.missing_covariate_values(self.state, all_rel, adjust_covariates)
            if missing:
                lines = []
                for name, rels in missing.items():
                    shown = ", ".join(Path(r).stem for r in rels[:8])
                    more = " …" if len(rels) > 8 else ""
                    lines.append(f"'{name}' ({len(rels)}): {shown}{more}")
                QMessageBox.warning(
                    self, "Missing Covariate Values",
                    "These adjustment covariates have no value for some tested "
                    "samples:\n\n" + "\n".join(lines) + "\n\nFill them in on the "
                    "Sample Group Assignment table, or untick them under 'Adjust for'."
                )
                return

        pval_threshold = self.pval_spin.value()
        fc_threshold   = self.fc_spin.value()
        mfi_threshold  = self.mfi_fc_spin.value()
        fdr_scope = 'per_comparison' if self.fdr_scope_combo.currentIndex() == 1 else 'global'
        self.state.stats_fdr_scope = fdr_scope
        self.state.stats_use_treat = self.chk_treat.isChecked()
        self.state.stats_mfi_threshold = mfi_threshold
        self.state.stats_mfi_hierarchical = self.chk_mfi_hierarchical.isChecked()
        self.state.stats_counts_method = self.counts_method_combo.currentData() or 'wald'
        self.state.stats_normalisation = self.normalisation_combo.currentData() or 'total'
        use_treat = self.state.stats_use_treat
        mfi_hierarchical = self.state.stats_mfi_hierarchical

        include_type_markers = self.chk_include_type_markers.isChecked()
        self.state.stats_pval_threshold = pval_threshold
        self.state.stats_fc_threshold = fc_threshold
        data_key = self._current_data_key(run_id, run_freq, run_counts, run_mfi)
        have_results = (not run_freq or self.state.freq_results is not None) and \
                       (not run_counts or self.state.counts_results is not None) and \
                       (not run_mfi or self.state.mfi_results is not None)

        if data_key == self._last_stats_data_key and have_results:
            self.state.stats_run_label = run_label
            self.state.stats_run_id = run_id
            # Same run, same groups, same tests already computed — only the
            # thresholds may have changed. Recompute p-values/significance
            # from the stored estimates and replot without refitting.
            self.state.stats_data_key = data_key
            self._apply_significance_thresholds(pval_threshold, fc_threshold)
            self.stats_status_label.setText("✓ Statistics complete (replotted — run unchanged).")
            self.stats_status_label.setStyleSheet("color: green;")
            self.export_results_btn.setEnabled(True)
            self.export_tables_btn.setEnabled(True)
            self._draw_results()
            self.results_computed.emit()
            return

        token = acquire_job(self.controller, 'Honeycluster: statistics')
        if token is None:
            self.stats_status_label.setText(self.controller.job_gate.busy_text())
            self.stats_status_label.setStyleSheet("color: orange;")
            return
        self._set_stats_busy(True)
        self.stats_status_label.setText("⏳ Computing statistics …")
        self.stats_status_label.setStyleSheet("color: orange;")

        # The worker reads and writes its own copy of the state, so edits
        # made in the UI while it runs cannot change the design under it
        # and half-written results are never visible. The results are
        # copied back in _on_stats_finished.
        stats_state = self._snapshot_stats_state()
        controller_ref = self.controller
        # Snapshot AF/transfer-matrix state HERE, on the main thread, before
        # the worker starts. compute_mfis() reads these via
        # drc_pipeline.apply_unmixing_af_aware(); the live controller
        # attributes are reassigned in place by controller.load_sample() /
        # initialise_af_matrices() whenever the user loads a different
        # sample in the main window. Without this snapshot the background
        # worker and a main-window sample load race on the same mutable
        # numpy arrays, and a reload mid-run could pair one sample's
        # transfer matrix with another sample's AF library.
        af_state = drc_pipeline.snapshot_af_state(self.controller)
        align_run, align_mode = self._stats_alignment(run_id)

        class _StatsWorker(QThread):
            finished = Signal(bool, str)
            progress = Signal(str)

            def __init__(self_, run_freq, run_mfi, labels_override,
                        include_type_markers, names_override, run_counts, af_state,
                        fdr_scope, adjust_covariates, use_treat, mfi_threshold,
                        mfi_hierarchical, state):
                super().__init__()
                self_.state = state
                self_.ran = (run_freq, run_counts, run_mfi)
                self_._run_freq = run_freq
                self_._run_mfi  = run_mfi
                self_._run_counts = run_counts
                self_._labels_override = labels_override
                self_._include_type_markers = include_type_markers
                self_._names_override = names_override
                self_._af_state = af_state
                self_._fdr_scope = fdr_scope
                self_._adjust_covariates = adjust_covariates
                self_._use_treat = use_treat
                self_._mfi_threshold = mfi_threshold
                self_._mfi_hierarchical = mfi_hierarchical
                self_.controller = controller_ref

            def run(self_):
                try:
                    self_._do_stats()
                    self_.finished.emit(True, '')
                except Exception as exc:
                    traceback.print_exc()
                    self_.finished.emit(False, str(exc))

            def _do_stats(self_):
                freq, mfi, counts = drc_stats.run_statistics(
                    self_.controller, self_.state,
                    self_._run_freq, self_._run_mfi,
                    pval_threshold, fc_threshold,
                    cluster_labels_override=self_._labels_override,
                    include_type_markers=self_._include_type_markers,
                    names_override=self_._names_override,
                    run_counts=self_._run_counts,
                    af_state=self_._af_state,
                    fdr_scope=self_._fdr_scope,
                    covariate_names=self_._adjust_covariates,
                    use_treat=self_._use_treat,
                    mfi_threshold=self_._mfi_threshold,
                    mfi_hierarchical=self_._mfi_hierarchical,
                    align_run=align_run, align_mode=align_mode,
                )
                if freq is not None:
                    self_.progress.emit(f"Frequencies: {len(freq)} rows tested.")
                if counts is not None:
                    self_.progress.emit(f"Counts GLM: {len(counts)} rows tested.")
                if mfi is not None:
                    self_.progress.emit(f"MFIs: {len(mfi)} rows tested.")

        worker = _StatsWorker(run_freq, run_mfi, labels_for_stats,
                             include_type_markers, names_for_stats, run_counts, af_state,
                             fdr_scope, adjust_covariates, use_treat, mfi_threshold,
                             mfi_hierarchical, stats_state)
        worker.progress.connect(lambda msg: print(f"[DR Stats] {msg}"))
        worker.job_token = token
        worker.finished.connect(
            lambda success, err, key=data_key, w=worker:
                self._on_stats_finished(success, err, key, w)
        )
        worker.run_meta = (run_label, run_id)
        self._stats_worker = worker
        worker.start()

    def _snapshot_stats_state(self) -> 'PipelineState':
        """
        Copy of the state with its own copies of every input the
        statistics read, for the worker thread. Large caches (gated data)
        stay shared.
        """
        snap = _shallow_copy(self.state)
        for attr in ('sample_groups', 'group_names', 'group_patterns',
                     'testing_group_selection', 'stats_adjust_covariates',
                     'numeric_covariates', 'marker_roles', 'channel_transform_params', 'selected_channels',
                     'cluster_names'):
            setattr(snap, attr, deepcopy(getattr(self.state, attr)))
        if self.state.covariates is not None:
            snap.covariates = self.state.covariates.copy()
        return snap

    def _adopt_stats_results(self, ran: tuple, snap: 'PipelineState'):
        """
        Copy a finished job's results from its private state onto the live
        state. Results of tests the job did not run are cleared, so the
        stored tables always come from one run.
        """
        ran_freq, ran_counts, ran_mfi = ran
        self.state.stats_all_rel = snap.stats_all_rel
        self.state.stats_group_vec = snap.stats_group_vec
        self.state.stats_comparisons = snap.stats_comparisons
        self.state.freq_results = snap.freq_results if ran_freq else None
        self.state.freq_df = snap.freq_df if ran_freq else None
        self.state.counts_results = snap.counts_results if ran_counts else None
        self.state.counts_df = snap.counts_df if ran_counts else None
        self.state.mfi_results = snap.mfi_results if ran_mfi else None
        self.state.mfi_df = snap.mfi_df if ran_mfi else None
        self.state.mfi_sample_df = snap.mfi_sample_df if ran_mfi else None

    def _set_stats_busy(self, busy: bool):
        """
        Busy cursor, and every control that starts or reads a statistics
        job disabled while one runs; re-enabled from the usual
        availability rules when it ends.
        """
        self._set_cursor_busy('stats', busy)
        controls = [self.run_stats_btn, self.composition_btn,
                    self.export_results_btn, self.export_tables_btn]
        if hasattr(self, 'pca_btn'):
            controls.append(self.pca_btn)
        if busy:
            for control in controls:
                control.setEnabled(False)
            self.pred_run_btn.setEnabled(False)
        else:
            self._update_run_button()
            self._refresh_predictive_controls()

    def restore_stats_controls(self):
        """Show the thresholds and FDR scope the saved results were computed with."""
        self.pval_spin.setValue(self.state.stats_pval_threshold)
        self.fc_spin.setValue(self.state.stats_fc_threshold)
        self.fdr_scope_combo.setCurrentIndex(
            1 if self.state.stats_fdr_scope == 'per_comparison' else 0
        )

    def _current_data_key(self, run_id: str, run_freq: bool, run_counts: bool,
                          run_mfi: bool) -> tuple:
        """
        Fingerprint of every input the statistics depend on: the clustering
        run, which tests are on, sample-to-group assignment, the test
        design (groups tested, contrast mode, reference, pairing,
        adjustment covariates and their values), the type-marker choice,
        marker roles and, for MFIs, the transform parameters of every
        channel. Two equal keys mean the stored results answer the
        current settings.
        """
        paired = bool(self.state.paired and self.state.pairing_variable)
        adjust_covariates = [
            c for c in self.state.stats_adjust_covariates
            if not (self.state.paired and c == self.state.pairing_variable)
        ]
        design_columns = list(adjust_covariates)
        if paired:
            design_columns.append(self.state.pairing_variable)
        covariate_values = ()
        if self.state.covariates is not None:
            present = [c for c in design_columns if c in self.state.covariates.columns]
            covariate_values = tuple(
                (rel, tuple(str(v) for v in row))
                for rel, row in self.state.covariates[present].iterrows()
            )
        groups_fingerprint = tuple(sorted(self.state.sample_groups.items()))
        test_fingerprint = (
            tuple(self.state.testing_group_selection), self.state.contrast_mode,
            self.state.reference_group, self.state.paired, self.state.pairing_variable,
            tuple(adjust_covariates), covariate_values,
            tuple(c for c in adjust_covariates if c in self.state.numeric_covariates),
            self.state.stats_counts_method if run_counts else None,
            self.state.stats_normalisation if (run_freq or run_counts) else None,
        )
        roles_fingerprint = tuple(sorted(self.state.marker_roles.items()))
        transform_fingerprint = ()
        if run_mfi:
            transform_fingerprint = tuple(
                (ch, round(float(v.get('T', 0.0)), 4), round(float(v.get('W', 0.0)), 4),
                 round(float(v.get('M', 0.0)), 4), round(float(v.get('A', 0.0)), 4),
                 str(v.get('id', '')))
                for ch, v in sorted(self.state.channel_transform_params.items())
            )
        return (run_id, run_freq, run_counts, run_mfi, groups_fingerprint,
                test_fingerprint, self.chk_include_type_markers.isChecked(),
                roles_fingerprint, transform_fingerprint, self._run_sync_key(run_id))

    def _run_sync_key(self, run_id: str) -> str:
        """Current fingerprint of a run's gates and samples: results computed
        under other gating or unmixing are not current."""
        for entry in self.state.clustering_runs:
            if entry.get('run_id') == run_id:
                return drc_pipeline.sync_fingerprint(
                    self.controller, entry.get('gates', []),
                    entry.get('training_sample_ids', [])) or ''
        return ''

    def _stats_alignment(self, run_id: str):
        """(run, mode) when the selected archived run no longer matches the
        current gating and unmixing and its labels have to be matched to
        the current events first, else (None, None)."""
        if not run_id:
            return None, None
        run = self._selected_run_entry()
        if run is None or run.get('run_id') != run_id:
            return None, None
        current = drc_pipeline.sync_fingerprint(
            self.controller, run.get('gates', []), run.get('training_sample_ids', []))
        if drc_run_archive.sync_status(run, current) == drc_run_archive.SYNC_IN_SYNC:
            return None, None
        return run, drc_run_align.live_join_mode(run, current)

    def _update_stats_freshness(self):
        """
        After results are restored from a saved experiment, check that the
        groups, pairing, covariates, marker roles and transforms they were
        computed from are unchanged. Unchanged: the results count as
        current, so Run Statistics only replots. Changed: they stay
        visible, flagged as out of date, and Run Statistics refits.
        """
        if self._analysis_busy():
            return
        if (self.state.freq_results is None and self.state.counts_results is None
                and self.state.mfi_results is None):
            return
        saved = self.state.stats_data_key
        if not saved:
            self._last_stats_data_key = None
            self.stats_status_label.setText(
                "Restored results have no recorded settings — Run Statistics to confirm them."
            )
            self.stats_status_label.setStyleSheet("color: orange;")
            return
        try:
            current = self._current_data_key(saved[0], saved[1], saved[2], saved[3])
        except Exception:
            _log.exception("_update_stats_freshness: could not build the current key")
            return
        if current == saved:
            if self._last_stats_data_key != saved:
                self._last_stats_data_key = saved
                for box, flag in ((self.chk_freq, saved[1]), (self.chk_counts, saved[2]),
                                  (self.chk_mfi, saved[3])):
                    box.setChecked(bool(flag))
                self.stats_status_label.setText(
                    "✓ Restored statistics — groups, pairing and transforms unchanged."
                )
                self.stats_status_label.setStyleSheet("color: green;")
        else:
            self._last_stats_data_key = None
            self.stats_status_label.setText(
                "⚠ Results are out of date: groups, pairing, covariates, marker "
                "roles or transforms changed since they were computed. "
                "Run Statistics to update."
            )
            self.stats_status_label.setStyleSheet("color: orange;")

    def on_job_gate_changed(self):
        """Re-apply the availability rules of the run buttons once the job gate is free."""
        gate = self.controller.job_gate
        if not gate.is_held():
            self._update_run_button()
            self._refresh_predictive_controls()

    def _analysis_busy(self) -> bool:
        """True while a statistics or predictive-analysis job is running."""
        return any(w is not None and w.isRunning()
                   for w in (self._stats_worker, self._predictive_worker))

    def _set_cursor_busy(self, owner: str, busy: bool):
        """Hold or release the busy cursor for one named job, at most once each."""
        if busy == (owner in self._busy_cursor_owners):
            return
        if busy:
            self._busy_cursor_owners.add(owner)
            QApplication.setOverrideCursor(Qt.BusyCursor)
        else:
            self._busy_cursor_owners.discard(owner)
            QApplication.restoreOverrideCursor()

    def _apply_significance_thresholds(self, pval_threshold: float, fc_threshold: float):
        """
        Recompute p-values, FDR and 'significant' on the stored freq/counts/
        mfi results for the current thresholds, TREAT and cluster-first
        settings, without refitting (drc_stats.apply_significance works from
        each row's estimate, SE and df).
        """
        for attr in ('freq_results', 'counts_results', 'mfi_results'):
            df = getattr(self.state, attr)
            if df is None or 'logFC' not in df.columns:
                continue
            is_mfi = attr == 'mfi_results'
            setattr(self.state, attr, drc_stats.apply_significance(
                df, pval_threshold,
                self.state.stats_mfi_threshold if is_mfi else fc_threshold,
                fdr_scope=self.state.stats_fdr_scope,
                use_treat=self.state.stats_use_treat,
                hierarchical=is_mfi and self.state.stats_mfi_hierarchical,
            ))

    def _resolve_stats_source(self):
        """
        Resolve which run's cluster labels to use for Run Statistics /
        Composition Barplot, plus a short label to stamp on the resulting
        plot(s), a stable run id for cache/run-change checks, and that
        SAME run's own names dict (never the ambient state.cluster_names).
        Both views share this one validation path.

        Returns (labels_dict, run_label, run_id, names_dict) or None (after
        showing the appropriate warning) if nothing usable is selected.
        """
        run_entry = self._selected_run_entry()
        if run_entry is not None and run_entry.get('kind') != 'clustering':
            run_entry = None
        if run_entry is None and not self.state.cluster_labels:
            QMessageBox.warning(self, "No Clustering", "Run clustering first.")
            return None
        shown = set(self._table_sample_paths())
        if not self.state.n_group_stats_runnable(shown):
            counts = self.state.n_per_group(shown)
            selection = self.state.testing_group_selection or self.state.group_names
            detail = ", ".join(f"{name}: {counts.get(name, 0)}" for name in selection)
            QMessageBox.warning(
                self, "Insufficient Groups",
                f"Need ≥ 3 samples in at least two checked 'Groups to Test'.\n{detail}"
            )
            return None

        if run_entry is not None:
            return (run_entry['labels'], run_entry['label'], run_entry['run_id'],
                    run_entry.get('names', {}))
        return self.state.cluster_labels, 'Active (unsaved)', '', dict(self.state.cluster_names)

    def _pick_pca_arrow_color(self):
        colour = QColorDialog.getColor(QColor(self.state.pca_arrow_color), self, "Arrow colour")
        if colour.isValid():
            self.state.pca_arrow_color = colour.name()
            self._update_pca_arrow_color_btn()
            self._on_pca_option_changed()

    def _update_pca_arrow_color_btn(self):
        text_color = _contrasting_text_color(self.state.pca_arrow_color)
        self.pca_arrow_color_btn.setStyleSheet(
            f"background-color: {self.state.pca_arrow_color}; color: {text_color};"
        )

    def _on_pca_option_changed(self, *_args):
        """
        Any PCA control changing after the plot has already been shown
        once regenerates it live -- point size, arrow colour/width, grid,
        and the two label toggles now take effect immediately instead of
        needing another 'Show PCA' click. The very first render still
        requires the button (Run Statistics has to have run first, and
        _show_pca's own missing-source checks still apply on every
        re-render -- e.g. unchecking the only computed 'Build from'
        source will still warn).
        """
        if self.state.pca_scores_df is not None and self.pca_btn.isEnabled():
            self._show_pca()

    def _show_pca(self):
        """
        Compute and display the Sample PCA plot. Independent of
        Run Statistics' per-comparison "Viewing comparison" — always uses
        every sample across every checked 'Groups to Test'. Re-clicking
        replaces the existing PCA tab in place, same as the Composition
        Barplot.
        """
        resolved = self._resolve_stats_source()
        if resolved is None:
            return
        _labels_for_stats, run_label, _run_id, _names_for_stats = resolved

        use_freq = self.pca_chk_freq.isChecked()
        use_counts = self.pca_chk_counts.isChecked()
        use_mfi = self.pca_chk_mfi.isChecked()
        self.state.pca_use_freq = use_freq
        self.state.pca_use_counts = use_counts
        self.state.pca_use_mfi = use_mfi
        self.state.pca_show_loadings = self.pca_chk_loadings.isChecked()
        self.state.pca_n_loadings = self.pca_n_loadings_spin.value()
        self.state.pca_point_size = float(self.pca_point_size_spin.value())
        self.state.pca_arrow_lw = self.pca_arrow_lw_spin.value()
        self.state.pca_axis_fontsize = self.pca_axis_fontsize_spin.value()
        self.state.pca_show_grid = self.pca_chk_grid.isChecked()
        self.state.pca_label_loadings = self.pca_chk_label_loadings.isChecked()
        self.state.pca_label_points = self.pca_chk_label_points.isChecked()

        if not (use_freq or use_counts or use_mfi):
            QMessageBox.warning(self, "Sample PCA",
                                "Check at least one of Frequencies/Counts/MFIs "
                                "under 'Build from' first.")
            return

        # Warn (rather than silently drop) whenever a checked
        # 'Build from' source hasn't actually been computed by Run
        # Statistics yet (e.g. Counts checked here but 'Cluster Counts
        # (GLM)' wasn't checked under 'Test:' the last time Run Statistics
        # ran).
        missing = []
        if use_freq and (self.state.freq_df is None or self.state.freq_df.empty):
            missing.append('Frequencies')
        if use_counts and (self.state.counts_df is None or self.state.counts_df.empty):
            missing.append('Counts')
        if use_mfi and (self.state.mfi_df is None or self.state.mfi_df.empty):
            missing.append('MFIs')

        if missing:
            n_checked = sum([use_freq, use_counts, use_mfi])
            if len(missing) >= n_checked:
                QMessageBox.warning(
                    self, "Sample PCA",
                    f"{', '.join(missing)} checked under 'Build from', but not "
                    "yet computed. Run Statistics first with the matching "
                    "'Test:' box(es) checked (Frequencies → 'Cluster "
                    "Frequencies', Counts → 'Cluster Counts (GLM)', "
                    "MFIs → 'Cluster MFIs')."
                )
                return
            reply = QMessageBox.warning(
                self, "Sample PCA",
                f"{', '.join(missing)} checked under 'Build from', but not yet "
                "computed by Run Statistics (check the matching 'Test:' "
                "box(es) and re-run it first) -- continuing with the "
                "remaining checked source(s) only.",
                QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Ok,
            )
            if reply != QMessageBox.Ok:
                return

        try:
            pca_result = drc_stats.compute_sample_pca(
                self.state, use_freq, use_counts, use_mfi,
                n_loadings=self.state.pca_n_loadings,
            )
        except Exception as e:
            QMessageBox.critical(self, "Sample PCA Error", str(e))
            return

        if pca_result is None:
            QMessageBox.warning(
                self, "Sample PCA",
                "No PCA could be computed — make sure Run Statistics has "
                "already been run with the matching 'Test:' box(es) checked, "
                "and that at least 2 samples / 2 non-degenerate features "
                "are available."
            )
            return

        self.state.pca_scores_df = pca_result['scores']
        self.state.pca_loadings_df = pca_result['loadings']
        self.state.pca_explained_variance = pca_result['explained_variance_ratio']
        self.state.pca_run_label = run_label
        self.state.pca_groups = pca_result['groups']
        self.state.pca_sources = pca_result['sources']

        fig = self._make_pca_figure(pca_result, run_label=run_label)
        self._add_results_tab(
            fig, "Sample PCA", "sample_pca",
            maker=self._make_pca_figure,
            maker_kwargs=dict(pca_result=pca_result, run_label=run_label),
            key="sample_pca",
        )

    def _pca_result_from_state(self) -> dict | None:
        """
        Rebuild the dict _make_pca_figure() expects from persisted
        state.pca_* fields, so the PCA tab gets the same "already
        computed -- redraw it without recomputing" treatment refresh()/
        refresh_theme_dependent_result_tabs() already give the
        Composition Barplot. Returns None if nothing's been
        computed yet this session/experiment.
        """
        if self.state.pca_scores_df is None or self.state.pca_loadings_df is None:
            return None
        return {
            'scores': self.state.pca_scores_df,
            'loadings': self.state.pca_loadings_df,
            'explained_variance_ratio': self.state.pca_explained_variance,
            'groups': self.state.pca_groups,
            'sources': self.state.pca_sources,
        }

    def _make_pca_figure(self, pca_result: dict | None, run_label: str = ''):
        """
        Sample-level PCA scatter, coloured by Group, with an optional
        top-N loadings biplot (arrows). Mouse-over (_make_scatter_hover_
        handler -- same convention as _make_volcano_figure) identifies
        every point and every loading UNCONDITIONALLY, regardless of the
        two label toggles -- those instead control a separate, always-
        on-screen static label (no adjustText; a plain fixed-offset
        annotation, same as the volcano plots' static top labels). Axis
        labels carry the % variance explained by that component.
        """
        from matplotlib.figure import Figure

        is_dark = _resolve_is_dark(self.state)
        fig = Figure(figsize=(6, 6), constrained_layout=True)
        ax = fig.add_subplot(111)
        fg = 'white' if is_dark else 'black'

        if pca_result is None:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No data to display',
                    ha='center', va='center', fontsize=10, transform=ax.transAxes)
            _style_figure_theme(fig, is_dark)
            self._stamp_run_label(fig, run_label)
            return fig

        scores = pca_result['scores']
        loadings = pca_result['loadings']
        groups = pca_result['groups']
        ev = pca_result['explained_variance_ratio']

        point_size = self.state.pca_point_size
        show_grid = self.state.pca_show_grid
        axis_fontsize = self.state.pca_axis_fontsize
        arrow_color = self.state.pca_arrow_color
        arrow_lw = self.state.pca_arrow_lw
        show_loadings = self.state.pca_show_loadings
        label_loadings = self.state.pca_label_loadings
        label_points = self.state.pca_label_points

        # Mouse-over tooltip targets (shared with the
        # volcano plots) -- collected as loadings/points are drawn below,
        # then handed to a single invisible overlay scatter at the end.
        # Always populated, independent of the two label toggles below.
        hover_xy: list[tuple[float, float]] = []
        hover_labels: list[str] = []

        unique_groups = list(dict.fromkeys(groups))   # first-seen order
        for grp in unique_groups:
            colour = self._group_colour(grp) if grp != 'Unassigned' else '#7f7f7f'
            mask = [g == grp for g in groups]
            ax.scatter(
                scores['PC1'].values[mask], scores['PC2'].values[mask],
                s=point_size, color=colour, edgecolors=fg, linewidths=0.4,
                label=grp, zorder=3,
            )

        sample_names = [Path(s).stem for s in scores.index]
        hover_xy.extend(zip(scores['PC1'].values, scores['PC2'].values))
        hover_labels.extend(sample_names)
        if label_points:
            # Static, always-on-screen label -- separate from the
            # mouse-over above, which covers every point either way.
            for x_val, y_val, name in zip(
                scores['PC1'].values, scores['PC2'].values, sample_names
            ):
                ax.annotate(name, xy=(x_val, y_val), xytext=(4, 4),
                           textcoords='offset points', fontsize=6,
                           color=fg, zorder=5)

        if show_loadings and not loadings.empty:
            score_max = max(
                float(scores['PC1'].abs().max()) if len(scores) else 1.0,
                float(scores['PC2'].abs().max()) if len(scores) else 1.0,
                1e-6,
            )
            load_max = max(
                float(loadings['PC1'].abs().max()),
                float(loadings['PC2'].abs().max()),
                1e-12,
            )
            scale = (score_max * 0.8) / load_max
            arrow_ends = loadings[['PC1', 'PC2']].values * scale

            # ax.text() does NOT contribute to matplotlib's
            # autoscale, so without this the axes were only ever sized to
            # the sample points (score_max), never the loading arrows
            # themselves. Widen the limits to include every arrow
            # endpoint so nothing lands outside the visible axes.
            all_x = np.concatenate([scores['PC1'].values, arrow_ends[:, 0], [0.0]])
            all_y = np.concatenate([scores['PC2'].values, arrow_ends[:, 1], [0.0]])
            x_span = max(float(np.ptp(all_x)), 1e-6)
            y_span = max(float(np.ptp(all_y)), 1e-6)
            ax.set_xlim(all_x.min() - x_span * 0.15, all_x.max() + x_span * 0.15)
            ax.set_ylim(all_y.min() - y_span * 0.15, all_y.max() + y_span * 0.15)

            for (feat, _row), (x_end, y_end) in zip(loadings.iterrows(), arrow_ends):
                ax.annotate(
                    '', xy=(x_end, y_end), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color=arrow_color, lw=arrow_lw),
                    zorder=4,
                )
                # Sample points along the whole shaft (not just the tip)
                # so hovering anywhere on the arrow -- not only its head
                # -- shows the label, unconditionally. Still just more
                # points fed into the same overlay scatter/handler below,
                # no new hover machinery. Starts at t=0.2 rather than 0
                # so hover targets from several arrows converging near
                # the origin don't all pile on top of each other.
                for t in np.linspace(0.2, 1.0, 8):
                    hover_xy.append((x_end * t, y_end * t))
                    hover_labels.append(feat)
                if label_loadings:
                    # Static, always-on-screen label at the tip --
                    # separate from the mouse-over above.
                    ax.annotate(feat, xy=(x_end, y_end), xytext=(4, 4),
                               textcoords='offset points', fontsize=7,
                               color=fg, zorder=5)

        # single invisible overlay scatter covers every point
        # and loading for mouse-over, unconditionally; handed to
        # _make_scatter_hover_handler exactly like the volcano plots. No
        # canvas exists yet here, so the handler is stashed on the Figure
        # for _add_results_tab/_pop_out to connect once one does.
        if hover_xy:
            hover_arr = np.array(hover_xy)
            hover_sc = ax.scatter(
                hover_arr[:, 0], hover_arr[:, 1],
                s=point_size * 2.5, alpha=0, zorder=6,
            )
            fig._hover_handler = _make_scatter_hover_handler(
                fig, ax, hover_sc, hover_labels, is_dark,
            )

        if show_grid:
            ax.axhline(0, color='grey', linestyle='--', linewidth=0.6, zorder=1)
            ax.axvline(0, color='grey', linestyle='--', linewidth=0.6, zorder=1)
        else:
            for spine in ax.spines.values():
                spine.set_visible(False)

        ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}% variance)", fontsize=axis_fontsize)
        ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}% variance)", fontsize=axis_fontsize)
        ax.tick_params(labelsize=axis_fontsize)
        ax.grid(show_grid, which='both', linestyle=':', linewidth=0.5, alpha=0.6)
        ax.set_title(f"Sample PCA ({' + '.join(pca_result['sources'])})", fontsize=10)
        ax.legend(fontsize=7, loc='best', frameon=True)

        _style_figure_theme(fig, is_dark)
        self._stamp_run_label(fig, run_label)
        return fig

    def _show_composition_barplot(self):
        """
        Compute and display the stacked composition barplot.
        Available once groups are assigned and a clustering run is
        selected, independent of Run Statistics. Re-clicking
        (or toggling %/By-group) replaces the existing Composition tab in
        place rather than adding another.
        """
        resolved = self._resolve_stats_source()
        if resolved is None:
            return
        labels_for_stats, run_label, run_id, names_for_stats = resolved

        # freeze this run's OWN colours now, same as
        # names_for_stats above. _selected_run_entry() is what
        # _resolve_stats_source() itself just called internally
        # (hydrate_run is a no-op on an already-hydrated entry, so this
        # second call is cheap), NOT a fresh/different lookup.
        run_entry_for_colors = self._selected_run_entry()
        if run_entry_for_colors is not None and run_entry_for_colors.get('kind') == 'clustering':
            colors_for_stats = dict(run_entry_for_colors.get('colors', {}))
        else:
            colors_for_stats = dict(self.state.cluster_colors)

        group_var = 'group' if self.composition_by_group_chk.isChecked() else 'sample'
        as_pct = self.composition_pct_chk.isChecked()
        table_fn = drc_stats.get_frequency_table if as_pct else drc_stats.get_counts_table

        try:
            comp_df = table_fn(
                self.controller, self.state, group_var=group_var,
                cluster_labels_override=labels_for_stats,
                names_override=names_for_stats,
            )
        except Exception as e:
            QMessageBox.critical(self, "Composition Barplot Error", str(e))
            return

        self.state.composition_df = comp_df
        self.state.composition_as_pct = as_pct
        self.state.composition_group_var = group_var
        self.state.composition_run_label = run_label
        self.state.composition_run_id = run_id
        self.state.composition_names = dict(names_for_stats)
        self.state.composition_colors = colors_for_stats

        fig = self._make_composition_figure(comp_df, as_pct=as_pct, run_label=run_label,
                                            names=names_for_stats, colors=colors_for_stats,
                                            group_var=group_var)
        self._add_results_tab(
            fig, "Composition", "composition_barplot",
            maker=self._make_composition_figure,
            maker_kwargs=dict(comp_df=comp_df, as_pct=as_pct, run_label=run_label,
                              names=names_for_stats, colors=colors_for_stats,
                              group_var=group_var),
            key="composition_barplot",
        )

    def _make_composition_figure(self, comp_df: 'pd.DataFrame', as_pct: bool,
                                 run_label: str = '', names: dict | None = None,
                                 colors: dict | None = None, group_var: str = 'sample'):
        """
        Stacked bar: x-axis = sample or group (comp_df.index), segments =
        clusters, coloured via the colours FROZEN at compute time (the
        colors param -- see _show_composition_barplot), never a live
        re-lookup of "whichever run is selected right now" -- that broke
        the moment the combo moved to a different run (including via a
        deletion), miscolouring or greying out this run's own clusters.
        Falls back to a neutral grey if a cluster's colour can't be
        resolved even from the frozen dict.
        """
        from matplotlib.figure import Figure

        is_dark = _resolve_is_dark(self.state)

        if comp_df.empty:
            fig = Figure(figsize=(5, 2), constrained_layout=True)
            ax = fig.add_subplot(111)
            ax.axis('off')
            ax.text(0.5, 0.5, 'No data to display',
                    ha='center', va='center', fontsize=10, transform=ax.transAxes)
            _style_figure_theme(fig, is_dark)
            return fig

        n_rows = len(comp_df)
        fig_w = max(4.0, 0.6 * n_rows + 2.5)
        fig = Figure(figsize=(fig_w, 5.0), layout='constrained')
        ax = fig.add_subplot(111)

        x = np.arange(n_rows)
        bottom = np.zeros(n_rows)
        colors = colors or {}
        for cl_id, col in enumerate(comp_df.columns):
            color = colors.get(cl_id, '#888888')
            vals = comp_df[col].values.astype(float)
            ax.bar(x, vals, bottom=bottom, width=0.7, color=color, label=col)
            bottom += vals

        ax.set_xticks(x)
        if group_var == 'sample':
            xtick_labels = [Path(s).stem for s in comp_df.index]
        else:
            xtick_labels = list(comp_df.index)
        ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('% of events' if as_pct else 'Event count')
        ax.set_title('Cluster Composition', fontsize=10)
        legend = ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8,
                           title='Cluster', frameon=False)
        fg = _style_figure_theme(fig, is_dark)
        # _style_figure_theme only touches tick/axis-label/title/spine
        # colours -- legend text is a separate artist it doesn't walk,
        # so it stayed black (invisible in dark mode) without this.
        for text in legend.get_texts():
            text.set_color(fg)
        legend.get_title().set_color(fg)
        self._stamp_run_label(fig, run_label)
        return fig

    def _on_stats_finished(self, success: bool, error_msg: str, data_key=None, worker=None):
        # The signal is emitted from inside run(); wait() lets the thread
        # finish before the last reference to it is dropped.
        if worker is not None:
            worker.wait()
            worker.job_token.release()
        if worker is None or self._stats_worker is worker:
            self._stats_worker = None
        self._set_stats_busy(False)
        if success and worker is not None:
            self._adopt_stats_results(worker.ran, worker.state)
            self.state.stats_run_label, self.state.stats_run_id = worker.run_meta
        if not success:
            self.stats_status_label.setText(f"Error: {error_msg}")
            self.stats_status_label.setStyleSheet("color: red;")
            QMessageBox.critical(self, "Statistics Error", error_msg)
            return

        if data_key is not None:
            self._last_stats_data_key = data_key
            self.state.stats_data_key = data_key
        note = getattr(worker.state, 'stats_alignment_note', '') if worker is not None else ''
        self.stats_status_label.setText(
            "✓ Statistics complete." + (f"\n{note}" if note else ''))
        self.stats_status_label.setStyleSheet("color: green;")
        self.export_results_btn.setEnabled(True)
        self.export_tables_btn.setEnabled(True)
        self._update_pca_source_availability()
        self._refresh_predictive_controls()
        self._draw_results()
        self.results_computed.emit()

    # ------------------------------------------------------------------
    # Predictive analysis
    # ------------------------------------------------------------------

    def _refresh_predictive_controls(self):
        """Comparison list from the last Run Statistics; feature boxes
        enabled only for matrices that run computed."""
        prev = self.pred_comparison_combo.currentText()
        self.pred_comparison_combo.blockSignals(True)
        self.pred_comparison_combo.clear()
        labels = [f"{other} vs {base}" for base, other in self.state.stats_comparisons]
        self.pred_comparison_combo.addItems(labels)
        idx = self.pred_comparison_combo.findText(prev)
        self.pred_comparison_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.pred_comparison_combo.blockSignals(False)

        have_freq = self.state.freq_df is not None and not self.state.freq_df.empty
        have_mfi = self.state.mfi_df is not None and not self.state.mfi_df.empty
        self.pred_chk_freq.setEnabled(have_freq)
        self.pred_chk_mfi.setEnabled(have_mfi)
        runnable = bool(labels) and (have_freq or have_mfi)
        self.pred_run_btn.setEnabled(runnable and not self._analysis_busy())
        self.pred_run_btn.setToolTip(
            "" if runnable else
            "Run Statistics first, with Cluster Frequencies and/or Cluster MFIs ticked."
        )

    def _run_predictive(self):
        if self._analysis_busy():
            return
        idx = self.pred_comparison_combo.currentIndex()
        if idx < 0 or idx >= len(self.state.stats_comparisons):
            return
        negative, positive = self.state.stats_comparisons[idx]
        models = []
        if self.pred_chk_lasso.isChecked():
            models.append('l1_logistic')
        if self.pred_chk_forest.isChecked():
            models.append('random_forest')
        if not models:
            QMessageBox.warning(self, "Predictive Analysis", "Tick at least one model.")
            return
        use_freq = self.pred_chk_freq.isChecked() and self.pred_chk_freq.isEnabled()
        use_mfi = self.pred_chk_mfi.isChecked() and self.pred_chk_mfi.isEnabled()
        if not (use_freq or use_mfi):
            QMessageBox.warning(self, "Predictive Analysis",
                                "Tick Frequencies and/or MFIs (computed by Run Statistics).")
            return

        rel_list = [rel for rel, g in zip(self.state.stats_all_rel, self.state.stats_group_vec)
                    if g in (negative, positive)]
        groups = [g for g in self.state.stats_group_vec if g in (negative, positive)]
        n_per_group = {positive: groups.count(positive), negative: groups.count(negative)}
        status, message = drc_predictive.sample_size_status(n_per_group)
        if status == 'block':
            QMessageBox.warning(self, "Predictive Analysis", message)
            return

        X = drc_predictive.build_feature_matrix(
            self.state.freq_df if use_freq else None,
            self.state.mfi_df if use_mfi else None,
            rel_list,
        )
        repeats = self.pred_repeats_spin.value()
        run_label = self.state.stats_run_label or 'Active (unsaved)'
        token = acquire_job(self.controller, 'Honeycluster: predictive analysis')
        if token is None:
            self.pred_status_label.setText(self.controller.job_gate.busy_text())
            self.pred_status_label.setStyleSheet("color: orange;")
            return

        class _PredictiveWorker(QThread):
            finished = Signal(bool, str)
            progress = Signal(str)

            def __init__(self_):
                super().__init__()
                self_.result = None

            def run(self_):
                try:
                    self_.result = drc_predictive.run_predictive(
                        X, groups, positive, negative, models=tuple(models),
                        repeats=repeats, progress=self_.progress.emit,
                    )
                    self_.result['run_label'] = run_label
                    self_.finished.emit(True, '')
                except Exception as exc:
                    traceback.print_exc()
                    self_.finished.emit(False, str(exc))

        worker = _PredictiveWorker()
        worker.job_token = token
        worker.progress.connect(self.pred_status_label.setText)
        worker.finished.connect(
            lambda ok, err, w=worker, st=status, msg=message:
            self._on_predictive_finished(ok, err, w, st, msg))
        self._predictive_worker = worker
        self._set_cursor_busy('predictive', True)
        self.pred_run_btn.setEnabled(False)
        self.run_stats_btn.setEnabled(False)
        self.pred_status_label.setStyleSheet("color: orange;")
        self.pred_status_label.setText(
            f"⏳ Running on {X.shape[1]} features, {len(rel_list)} samples …")
        worker.start()

    def _on_predictive_finished(self, success: bool, error_msg: str, worker,
                                status: str, message: str):
        # The signal is emitted from inside run(); wait() lets the thread
        # finish before the last reference to it is dropped.
        worker.wait()
        worker.job_token.release()
        if self._predictive_worker is worker:
            self._predictive_worker = None
        self._set_cursor_busy('predictive', False)
        self._update_run_button()
        self._refresh_predictive_controls()
        if not success:
            self.pred_status_label.setStyleSheet("color: red;")
            self.pred_status_label.setText(f"Error: {error_msg}")
            QMessageBox.critical(self, "Predictive Analysis Error", error_msg)
            return
        self.state.predictive_result = worker.result
        summary = worker.result['summary']
        text = (f"✓ Done. Minimal panel: {summary['minimal_panel']} feature(s) "
                f"(best panel AUC {summary['best_panel_auc']:.2f}).")
        if status == 'warn':
            text += f"  ⚠ {message}"
            self.pred_status_label.setStyleSheet("color: orange;")
        else:
            self.pred_status_label.setStyleSheet("color: green;")
        self.pred_status_label.setText(text)
        self._draw_predictive_results()

    def _draw_predictive_results(self):
        res = self.state.predictive_result
        if not res:
            return
        run_label = res.get('run_label', '')
        summary = res['summary']
        for key, title, maker, kwargs in (
            ('pred_performance', "Predictive: ROC", self._make_predictive_roc_figure,
             dict(performance=res['performance'], roc=res['roc'], summary=summary,
                  run_label=run_label)),
            ('pred_importance', "Predictive: Features", self._make_predictive_importance_figure,
             dict(importance=res['importance'], summary=summary, run_label=run_label)),
            ('pred_panel', "Predictive: Panel Size", self._make_predictive_panel_figure,
             dict(panel=res['panel'], summary=summary, run_label=run_label)),
        ):
            try:
                fig = maker(**kwargs)
            except Exception as e:
                traceback.print_exc()
                QMessageBox.warning(self, "Predictive Analysis", f"{title}: {e}")
                continue
            self._add_results_tab(fig, title, key, maker=maker, maker_kwargs=kwargs, key=key)

    def _predictive_subtitle(self, summary: dict) -> str:
        n = ", ".join(f"{g} n={k}" for g, k in summary['n_per_group'].items())
        text = (f"{summary['positive']} vs {summary['negative']} ({n}); "
                f"{summary['repeats']}× repeated {summary['outer_folds']}-fold CV")
        if summary.get('status') == 'warn':
            text += (f"\nfewer than {drc_predictive.RECOMMENDED_PER_GROUP} per group: "
                     "estimates are noisy")
        return text

    def _make_predictive_roc_figure(self, performance, roc, summary: dict, run_label: str = ''):
        """Mean out-of-fold ROC curve per model, AUC ± SD over repeats in the legend."""
        from matplotlib.figure import Figure

        is_dark = _resolve_is_dark(self.state)
        fig = Figure(figsize=(5.5, 5), constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=0.8)
        for _, row in performance.iterrows():
            curve = roc[roc['model'] == row['model']]
            ax.plot(curve['fpr'], curve['tpr'], linewidth=1.6,
                    label=(f"{row['model']}: AUC {row['auc_mean']:.2f} ± {row['auc_sd']:.2f}, "
                           f"bal. acc. {row['balanced_accuracy_mean']:.2f}"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc='lower right', fontsize=7)
        ax.set_title(f"Cross-validated ROC\n{self._predictive_subtitle(summary)}", fontsize=9)
        _style_figure_theme(fig, is_dark)
        self._stamp_run_label(fig, run_label)
        return fig

    def _make_predictive_importance_figure(self, importance, summary: dict,
                                           run_label: str = '', top_n: int = 20):
        """Top features: lasso selection frequency and/or forest importance,
        coloured by which group has the higher value."""
        from matplotlib.figure import Figure

        is_dark = _resolve_is_dark(self.state)
        metrics = [(c, lbl) for c, lbl in (
            ('lasso_selection', 'Lasso: fraction of fits selecting'),
            ('forest_importance', 'Random forest: mean importance'),
        ) if c in importance.columns]
        top = importance.head(top_n).iloc[::-1]
        colours = ['#EE6677' if d > 0 else '#4477AA' for d in top['direction']]
        fig = Figure(figsize=(4.5 * len(metrics) + 2.5, max(3.5, 0.28 * len(top) + 1.8)),
                     constrained_layout=True)
        for i, (col, label) in enumerate(metrics):
            ax = fig.add_subplot(1, len(metrics), i + 1)
            xerr = top['forest_importance_sd'] if col == 'forest_importance' else None
            ax.barh(range(len(top)), top[col], color=colours, xerr=xerr,
                    error_kw={'elinewidth': 0.6})
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top['feature'] if i == 0 else [''] * len(top), fontsize=7)
            ax.set_xlabel(label, fontsize=8)
        import matplotlib.patches as mpatches
        fig.legend(handles=[
            mpatches.Patch(color='#EE6677', label=f"higher in {summary['positive']}"),
            mpatches.Patch(color='#4477AA', label=f"higher in {summary['negative']}"),
        ], loc='outside lower center', ncol=2, fontsize=7)
        fig.suptitle(f"Most informative features\n{self._predictive_subtitle(summary)}", fontsize=9)
        _style_figure_theme(fig, is_dark)
        self._stamp_run_label(fig, run_label)
        return fig

    def _make_predictive_panel_figure(self, panel, summary: dict, run_label: str = ''):
        """Cross-validated AUC against the number of top-ranked features."""
        from matplotlib.figure import Figure

        is_dark = _resolve_is_dark(self.state)
        fig = Figure(figsize=(5.5, 4.5), constrained_layout=True)
        ax = fig.add_subplot(111)
        x = panel['n_features'].values
        m = panel['auc_mean'].values
        sd = panel['auc_sd'].values
        ax.fill_between(x, m - sd, m + sd, alpha=0.25)
        ax.plot(x, m, marker='o', markersize=3, linewidth=1.4)
        ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8)
        ax.axvline(summary['minimal_panel'], color='grey', linestyle=':', linewidth=1.0)
        ax.annotate(f"minimal panel: {summary['minimal_panel']}",
                    xy=(summary['minimal_panel'], 0.5), xytext=(4, 4),
                    textcoords='offset points', fontsize=7)
        ax.set_xscale('log')
        ax.set_xlabel("Number of top-ranked features (log scale)")
        ax.set_ylabel("Cross-validated AUC (mean ± SD)")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"How many features are needed?\n{self._predictive_subtitle(summary)}",
                     fontsize=9)
        _style_figure_theme(fig, is_dark)
        self._stamp_run_label(fig, run_label)
        return fig

    # ------------------------------------------------------------------
    # Results visualisation
    # ------------------------------------------------------------------

    def _add_results_tab(self, fig, tab_title: str, export_stem: str, maker, maker_kwargs,
                        key: str | None = None):
        """
        Add a plot tab (with pop-out + export) to self._results_tabs.

        maker       : callable that returns a fresh Figure (e.g. _make_heatmap_figure)
        maker_kwargs: dict of keyword args to pass to maker (excluding positional args
                      already captured in the lambda at call site)
        key         : stable identifier for "this same plot, regenerated" (e.g.
                      'composition_barplot'). If a tab with the same key already
                      exists it's replaced in place (same position, refocused)
                      instead of appended — so re-clicking a button doesn't pile
                      up duplicate tabs. Every tab (keyed or not) can still be
                      closed manually via its close button.

        Extracted from the old _draw_results-local `_add_tab` closure so
        it can add standalone result tabs (Composition Barplot,
        Sample PCA) outside of _draw_results, using the same pop-out/
        export machinery as the heatmap/volcano tabs.
        """
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
        from PySide6.QtWidgets import QDialog

        insert_at = None
        if key is not None:
            for i in range(self._results_tabs.count()):
                if self._results_tabs.widget(i).property('_tab_key') == key:
                    insert_at = i
                    self._results_tabs.removeTab(i)
                    break

        dpi  = fig.get_dpi()
        w_px = int(fig.get_figwidth()  * dpi)
        h_px = int(fig.get_figheight() * dpi)

        # these plots routinely overflowed the viewport at full
        # size. The inline canvas now starts at roughly a third of the
        # figure's natural size (DEFAULT_SCALE_PCT) and the slider added
        # to btn_row below lets the user scale it back up. Fixed (not
        # Expanding) so the canvas actually shrinks instead of the
        # QScrollArea stretching it back out to fill the viewport. Pop
        # Out / Export both call maker(**maker_kwargs) for a fresh
        # full-size figure of their own, so neither is affected by this.
        DEFAULT_SCALE_PCT = 100
        canvas = _new_scrollable_canvas(fig)
        canvas.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        # Hover tooltips -- figure-makers that want them stash a
        # handler on the Figure itself (fig._hover_handler), since no
        # canvas exists yet when they run; connect it now that one does.
        if hasattr(fig, '_hover_handler'):
            canvas.mpl_connect('motion_notify_event', fig._hover_handler)

        scroll = QScrollArea()
        scroll.setWidget(canvas)
        scroll.setWidgetResizable(True)

        def _apply_canvas_scale(pct: int):
            """Resize the inline canvas to pct% of the figure's natural
            size; keeps this tab's own QScrollArea tall enough to show it
            without a nested scrollbar."""
            new_w = max(80, int(w_px * pct / 100))
            new_h = max(80, int(h_px * pct / 100))
            canvas.setFixedSize(new_w, new_h)
            # Grows/shrinks with the slider -- no cap otherwise. This
            # tab's own outer QScrollArea (GroupsStatsTab._build_ui)
            # scrolls the whole page if that makes it taller than the
            # viewport, rather than nesting a second scrollbar in here.
            scroll.setMinimumHeight(new_h + 40)

        _apply_canvas_scale(DEFAULT_SCALE_PCT)

        # Placeholder shown while plot is popped out
        placeholder = QLabel(f'"{tab_title}" is open in a separate window.\nClose that window to restore it here.')
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: grey; font-style: italic;")
        placeholder.hide()

        def _pop_out(checked=False):
            # Generate a fresh figure for the dialog (avoids shared-canvas DPI issues)
            try:
                dlg_fig = maker(**maker_kwargs)
            except Exception as exc:
                QMessageBox.warning(self, "Pop Out Error", str(exc))
                return

            dlg_dpi  = dlg_fig.get_dpi()
            dlg_w_px = int(dlg_fig.get_figwidth()  * dlg_dpi)
            dlg_h_px = int(dlg_fig.get_figheight() * dlg_dpi)

            dlg_canvas  = _new_scrollable_canvas(dlg_fig)
            dlg_canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            # Hover tooltips -- dlg_fig is a fresh figure (from
            # maker(**maker_kwargs) above), so its own handler needs
            # connecting too.
            if hasattr(dlg_fig, '_hover_handler'):
                dlg_canvas.mpl_connect('motion_notify_event', dlg_fig._hover_handler)
            dlg_toolbar = NavigationToolbar2QT(dlg_canvas, None)

            class _PlotDialog(QDialog):
                def closeEvent(self_, event):
                    # Disconnect toolbar before C++ objects are destroyed
                    try:
                        dlg_toolbar.canvas = None
                    except Exception:
                        pass
                    # Restore tab view
                    scroll.show()
                    placeholder.hide()
                    popout_btn.setEnabled(True)
                    super().closeEvent(event)

            dlg = _PlotDialog(self)
            dlg.setWindowTitle(tab_title)
            dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            dlg.resize(dlg_w_px + 24, dlg_h_px + 80)

            vb = QVBoxLayout(dlg)
            vb.setContentsMargins(4, 4, 4, 4)
            vb.setSpacing(4)
            vb.addWidget(dlg_toolbar)
            vb.addWidget(dlg_canvas, stretch=1)
            dlg.show()

            # Hide tab canvas, show placeholder
            scroll.hide()
            placeholder.show()
            popout_btn.setEnabled(False)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        popout_btn = QPushButton("⤢ Pop Out")
        popout_btn.setFixedWidth(80)
        popout_btn.clicked.connect(_pop_out)
        export_btn = QPushButton(f"Export {tab_title}")
        export_btn.clicked.connect(lambda checked=False, f=fig, s=export_stem:
                                   self._export_figure(f, s))
        btn_row.addWidget(popout_btn)
        btn_row.addWidget(export_btn)
        btn_row.addSpacing(16)

        # manual resize slider for the inline canvas.
        btn_row.addWidget(QLabel("Size:"))
        size_slider = QSlider(Qt.Horizontal)
        size_slider.setRange(20, 200)
        size_slider.setValue(DEFAULT_SCALE_PCT)
        size_slider.setFixedWidth(120)
        size_slider.setToolTip(
            "Resize this plot. Pop Out and Export always use full size."
        )
        size_pct_label = QLabel(f"{DEFAULT_SCALE_PCT}%")
        size_pct_label.setFixedWidth(36)

        def _on_size_slider_changed(pct: int):
            _apply_canvas_scale(pct)
            size_pct_label.setText(f"{pct}%")

        size_slider.valueChanged.connect(_on_size_slider_changed)
        btn_row.addWidget(size_slider)
        btn_row.addWidget(size_pct_label)
        btn_row.addStretch()

        container = QWidget()
        if key is not None:
            container.setProperty('_tab_key', key)
            self._results_registry[key] = dict(
                title=tab_title, maker=maker, maker_kwargs=maker_kwargs,
            )
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)
        # Pop Out/Export/Size row above the canvas so it's always visible
        # without scrolling, whatever the canvas's current scale.
        vbox.addLayout(btn_row)
        vbox.addWidget(scroll, stretch=1)
        vbox.addWidget(placeholder, stretch=1)

        if insert_at is not None:
            self._results_tabs.insertTab(insert_at, container, tab_title)
            self._results_tabs.setCurrentIndex(insert_at)
        else:
            idx = self._results_tabs.addTab(container, tab_title)
            self._results_tabs.setCurrentIndex(idx)

    def _on_results_tab_close_requested(self, index: int):
        """Manually close any results tab (Freq/MFI heatmap/volcano,
        Composition Barplot, or Sample PCA)."""
        widget = self._results_tabs.widget(index)
        key = widget.property('_tab_key') if widget is not None else None
        self._results_tabs.removeTab(index)
        if widget is not None:
            widget.deleteLater()
        if key is not None:
            self._results_registry.pop(key, None)

    def get_report_items(self) -> list:
        """One drc_report.ReportItem per currently-open results tab
        (Freq/Counts/MFI heatmap+volcano, Composition,
        Sample PCA) -- covers every result kind automatically since they
        all go through _add_results_tab's maker/maker_kwargs registry."""
        items = []
        for i in range(self._results_tabs.count()):
            widget = self._results_tabs.widget(i)
            key = widget.property('_tab_key') if widget is not None else None
            if key is None:
                continue
            entry = self._results_registry.get(key)
            if entry is None:
                continue
            maker = entry['maker']
            maker_kwargs = entry['maker_kwargs']
            items.append(drc_report.ReportItem(
                key=key,
                tab='Stats',
                label=entry['title'],
                get_figure=lambda maker=maker, kwargs=maker_kwargs: maker(**kwargs),
                get_tables=lambda kwargs=maker_kwargs: drc_report.tables_from_maker_kwargs(kwargs),
            ))
        return items

    def _has_results_tab(self, key: str) -> bool:
        """True if a results tab carrying this key is already showing —
        avoids adding a duplicate Composition tab on
        every refresh() once it's been drawn once this session."""
        for i in range(self._results_tabs.count()):
            widget = self._results_tabs.widget(i)
            if widget is not None and widget.property('_tab_key') == key:
                return True
        return False

    def refresh_theme_dependent_result_tabs(self):
        """
        The Composition Barplot isn't tied to a single cached 'last drawn
        from' call the way Stats' Freq/MFI/Counts results are
        (_draw_results), so a plain theme change wouldn't otherwise touch
        it. Re-render it in place, using exactly the same persisted state
        (state.composition_df, etc.) the startup tab-restore logic
        already uses, only if that tab is actually currently open.
        """
        if self.state.composition_df is not None and self._has_results_tab('composition_barplot'):
            fig_comp = self._make_composition_figure(
                self.state.composition_df, as_pct=self.state.composition_as_pct,
                run_label=self.state.composition_run_label, names=self.state.composition_names,
                colors=self.state.composition_colors,
                group_var=self.state.composition_group_var,
            )
            self._add_results_tab(
                fig_comp, "Composition", "composition_barplot",
                maker=self._make_composition_figure,
                maker_kwargs=dict(comp_df=self.state.composition_df,
                                  as_pct=self.state.composition_as_pct,
                                  run_label=self.state.composition_run_label,
                                  names=self.state.composition_names,
                                  colors=self.state.composition_colors,
                                  group_var=self.state.composition_group_var),
                key="composition_barplot",
            )
        pca_result = self._pca_result_from_state()
        if pca_result is not None and self._has_results_tab('sample_pca'):
            fig_pca = self._make_pca_figure(pca_result, run_label=self.state.pca_run_label)
            self._add_results_tab(
                fig_pca, "Sample PCA", "sample_pca",
                maker=self._make_pca_figure,
                maker_kwargs=dict(pca_result=pca_result, run_label=self.state.pca_run_label),
                key="sample_pca",
            )
        if self.state.predictive_result and self._has_results_tab('pred_performance'):
            self._draw_predictive_results()

    def _remove_results_tab_by_key(self, key: str):
        """Remove the results tab whose container carries this key, if
        present. Used by _draw_results to clear only its own four tabs
        (Freq/MFI heatmap/volcano) rather than every tab — the
        Composition Barplot is independent and must survive a Run
        Statistics (re-)run."""
        for i in range(self._results_tabs.count()):
            widget = self._results_tabs.widget(i)
            if widget is not None and widget.property('_tab_key') == key:
                self._results_tabs.removeTab(i)
                widget.deleteLater()
                self._results_registry.pop(key, None)
                return

    def _draw_results(self):
        """
        Render heatmap and volcano into per-plot tabs, for the currently
        selected 'Viewing comparison' (a single Run
        Statistics call can now produce several comparisons; each plot
        still shows exactly one at a time ).

        Figure-building (the expensive part -- heatmap dendrograms,
        volcano scatter/labelling) happens on a background QThread
        (_ResultsDrawWorker). Only tab add/remove (Qt) happens here and in
        _on_results_figures_built, on the main thread.
        """
        if self._stats_worker is not None and self._stats_worker.isRunning():
            # The result tables are still being written; the stats job
            # redraws when it finishes.
            return
        if self._results_draw_worker is not None:
            # A build is already in flight -- don't start a second one
            # against a moving target; remember to redraw once more as
            # soon as this one finishes instead.
            _log.info("_draw_results: a build is already in flight -- "
                      "deferring (setting _results_draw_pending=True)")
            self._results_draw_pending = True
            return

        self._last_drawn_cluster_names = dict(self.state.cluster_names)
        # Remove only this method's own tabs (by key) — leaves the
        # Composition Barplot (and anything else) untouched.
        # Done immediately (not after the background build) so a stale
        # previous comparison's plots don't linger while the new ones
        # are being built.
        for key in ('freq_heatmap', 'freq_volcano', 'counts_heatmap', 'counts_volcano',
                   'mfi_heatmap', 'mfi_volcano', 'mfi_clusters'):
            self._remove_results_tab_by_key(key)

        self._refresh_viewing_comparison_combo()
        if not self.state.stats_comparisons:
            _log.info("_draw_results: state.stats_comparisons is empty -- nothing to draw")
            return
        view_idx = max(0, self.viewing_comparison_combo.currentIndex())
        name_a, name_b = self.state.stats_comparisons[view_idx]
        run_label = self.state.stats_run_label or 'Active (unsaved)'
        _log.info("_draw_results: building comparison %r vs %r, run_label=%r "
                  "(freq_results=%s mfi_results=%s counts_results=%s)",
                  name_b, name_a, run_label,
                  self.state.freq_results is not None,
                  self.state.mfi_results is not None,
                  self.state.counts_results is not None)
        pval_threshold = self.pval_spin.value()
        fc_threshold = self.fc_spin.value()
        mfi_threshold = self.state.stats_mfi_threshold
        channel_names = _antigen_or_label_map(self.controller)

        # Snapshot the current result DataFrames NOW, on the main thread.
        # The background worker must never read self.state live --
        # _on_stats_finished re-enables the Run Statistics button as soon
        # as computation finishes, so a fresh click could otherwise
        # reassign state.freq_results/mfi_df/etc. WHILE this build is
        # still reading them.
        freq_results_snap    = self.state.freq_results
        freq_df_snap         = self.state.freq_df
        counts_results_snap  = self.state.counts_results
        counts_df_snap       = self.state.counts_df
        mfi_results_snap     = self.state.mfi_results
        mfi_sample_df_snap   = self.state.mfi_sample_df
        stats_all_rel_snap   = list(self.state.stats_all_rel)
        stats_group_vec_snap = list(self.state.stats_group_vec)

        def _build():
            return self._build_results_figures(
                name_a, name_b, run_label, pval_threshold, fc_threshold,
                freq_results_snap, freq_df_snap,
                counts_results_snap, counts_df_snap,
                mfi_results_snap, mfi_sample_df_snap,
                stats_all_rel_snap, stats_group_vec_snap,
                mfi_threshold=mfi_threshold, channel_names=channel_names,
            )

        worker = _ResultsDrawWorker(_build)
        worker.finished.connect(self._on_results_figures_built)
        self._results_draw_worker = worker
        worker.start()

    def _build_results_figures(self, name_a, name_b, run_label,
                                pval_threshold, fc_threshold,
                                freq_results, freq_df,
                                counts_results, counts_df,
                                mfi_results, mfi_sample_df,
                                stats_all_rel, stats_group_vec,
                                mfi_threshold: float | None = None,
                                channel_names: dict | None = None) -> dict:
        """
        Pure compute: build every results Figure from the snapshotted
        DataFrames passed in (no self.state reads, no Qt) -- runs on
        _ResultsDrawWorker's background thread. Returns an ordered dict
        of {key: {'fig', 'error', 'title', 'maker', 'maker_kwargs'}} for
        _on_results_figures_built to turn into tabs on the main thread.
        """
        comparison_label = f"{name_b} vs {name_a}"
        relevant_rel = {
            rel for rel, g in zip(stats_all_rel, stats_group_vec)
            if g in (name_a, name_b)
        }

        def _view_results(results_df):
            if results_df is None or 'comparison' not in results_df.columns:
                return results_df
            return (results_df[results_df['comparison'] == comparison_label]
                    .drop(columns=['comparison']).reset_index(drop=True))

        def _view_samples(sample_df):
            if sample_df is None:
                return sample_df
            keep = [r for r in sample_df.index if r in relevant_rel]
            return sample_df.loc[keep]

        freq_results_view    = _view_results(freq_results)
        freq_df_view         = _view_samples(freq_df)
        counts_results_view  = _view_results(counts_results)
        counts_df_view       = _view_samples(counts_df)
        mfi_results_view     = _view_results(mfi_results)
        mfi_sample_df_view   = _view_samples(mfi_sample_df)

        results: dict = {}

        def _build(key, title, maker, maker_kwargs):
            try:
                fig = maker(**maker_kwargs)
                results[key] = dict(fig=fig, error=None, title=title,
                                     maker=maker, maker_kwargs=maker_kwargs)
            except Exception as e:
                traceback.print_exc()
                results[key] = dict(fig=None, error=str(e), title=title,
                                     maker=maker, maker_kwargs=maker_kwargs)

        if freq_results_view is not None:
            _build('freq_heatmap', "Freq Heatmap", self._make_heatmap_figure, dict(
                results_df=freq_results_view, sample_df=freq_df_view,
                title=f"Significantly Different Cluster Frequencies: {name_b} vs {name_a}",
                group_a=name_a, group_b=name_b, run_label=run_label,
            ))
            _build('freq_volcano', "Freq Volcano", self._make_volcano_figure, dict(
                results_df=freq_results_view,
                title=f"Cluster Frequency Volcano: {name_b} vs {name_a}",
                pval_threshold=pval_threshold, fc_threshold=fc_threshold,
                run_label=run_label, group_a=name_a, group_b=name_b,
            ))

        if counts_results_view is not None:
            _build('counts_heatmap', "Counts Heatmap", self._make_heatmap_figure, dict(
                results_df=counts_results_view, sample_df=counts_df_view,
                title=f"Cluster Counts: {name_b} vs {name_a}",
                group_a=name_a, group_b=name_b, run_label=run_label,
            ))
            _build('counts_volcano', "Counts Volcano", self._make_volcano_figure, dict(
                results_df=counts_results_view,
                title=f"Cluster Counts Volcano: {name_b} vs {name_a}",
                pval_threshold=pval_threshold, fc_threshold=fc_threshold,
                run_label=run_label, group_a=name_a, group_b=name_b,
            ))

        if mfi_sample_df_view is not None:
            _build('mfi_heatmap', "MFI Heatmap", self._make_sample_mfi_heatmap_figure, dict(
                mfi_sample_df=mfi_sample_df_view,
                group_a=name_a, group_b=name_b, run_label=run_label,
            ))
        if mfi_results_view is not None:
            mfi_fc = fc_threshold if mfi_threshold is None else mfi_threshold
            _build('mfi_volcano', "MFI Volcano", self._make_volcano_figure, dict(
                results_df=mfi_results_view,
                title=f"Cluster MFI Volcano: {name_b} vs {name_a}",
                pval_threshold=pval_threshold, fc_threshold=mfi_fc,
                run_label=run_label, group_a=name_a, group_b=name_b,
            ))
            _build('mfi_clusters', "MFI Clusters", self._make_mfi_cluster_summary_figure, dict(
                results_df=mfi_results_view,
                title=f"Marker changes in significant clusters: {name_b} vs {name_a}",
                channel_names=channel_names or {},
                run_label=run_label,
            ))

        return results

    def _on_results_figures_built(self, payload: dict):
        """
        Main-thread slot: turn _build_results_figures' payload into tabs
        -- the only Qt-touching part of the whole redraw. Connected to
        _ResultsDrawWorker.finished.
        """
        self._results_draw_worker = None
        _log.info("_on_results_figures_built: received %d entry/entries: %r",
                  len(payload), {k: ('ok' if v['error'] is None else v['error'])
                                 for k, v in payload.items()})
        for key, item in payload.items():
            if item['error'] is not None:
                tab = QLabel(f"{item['title']} error: {item['error']}")
                tab.setStyleSheet("color: red;")
                tab.setProperty('_tab_key', key)
                self._results_tabs.addTab(tab, item['title'])
            else:
                self._add_results_tab(
                    item['fig'], item['title'], key,
                    maker=item['maker'], maker_kwargs=item['maker_kwargs'],
                    key=key,
                )
        if self._results_draw_pending:
            self._results_draw_pending = False
            self._draw_results()
        else:
            self.results_drawn.emit()

    @staticmethod
    def _stamp_run_label(fig, run_label: str):
        """Stamp the source run's label in the upper-left corner of a
        results figure, so a popped-out or exported plot stays traceable
        to the run it came from."""
        stamp_run_label(fig, run_label)

    def _make_sample_mfi_heatmap_figure(self, mfi_sample_df, group_a: str = '',
                                        group_b: str = '', run_label: str = ''):
        """
        Sample-level MFI heatmap: rows = channels, columns = samples. No
        cluster breakdown, no significance filter (shows every requested
        channel) — see compute_sample_mfis()'s docstring for why the
        per-cluster version doesn't work for this view.
        """
        from matplotlib.figure import Figure
        from matplotlib.gridspec import GridSpec
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import pdist
        import matplotlib.patches as mpatches

        name_a, name_b = group_a, group_b
        COLOR_A = '#4477AA'
        COLOR_B = '#EE6677'

        is_dark = _resolve_is_dark(self.state)

        if mfi_sample_df is None or mfi_sample_df.empty:
            fig = Figure(figsize=(5, 2), constrained_layout=True)
            ax = fig.add_subplot(111)
            ax.axis('off')
            ax.text(0.5, 0.5, 'No data to display',
                    ha='center', va='center', fontsize=10, transform=ax.transAxes)
            _style_figure_theme(fig, is_dark)
            self._stamp_run_label(fig, run_label)
            return fig

        mat = mfi_sample_df.values.astype(float)      # (n_samples, n_channels)
        sample_labels = list(mfi_sample_df.index)
        feat_labels_raw = list(mfi_sample_df.columns)
        n_samples, n_features = mat.shape

        rel_to_group: dict[str, str] = {
            rel: grp
            for rel, grp in zip(self.state.stats_all_rel, self.state.stats_group_vec)
        }

        def _hclust(data):
            """Returns (leaf_order, linkage_matrix_or_None) -- Z is reused
            for the dendrogram plots below instead of being recomputed
            (see the same fix in _make_heatmap_figure)."""
            filled = np.nan_to_num(data, nan=0.0)
            if filled.shape[0] < 2:
                return list(range(filled.shape[0])), None
            try:
                Z = linkage(pdist(filled, metric='euclidean'), method='ward')
                return dendrogram(Z, no_plot=True)['leaves'], Z
            except Exception:
                return list(range(filled.shape[0])), None

        row_order, Zr = _hclust(mat.T)
        col_order, Zc = _hclust(mat)

        mat_ord = mat[:, row_order][col_order, :].T   # (n_features, n_samples)
        feat_labels = [feat_labels_raw[i] for i in row_order]
        samp_labels = [sample_labels[i] for i in col_order]
        grp_ord = [rel_to_group.get(sample_labels[i], name_a) for i in col_order]

        # Antigen-only remap, falling back to Label/channel name.
        disp_map = _antigen_or_label_map(self.controller)
        feat_labels = [disp_map.get(f, f) for f in feat_labels]

        col_w, row_h, dend_h, grp_h, xlabel_h = 0.55, 0.30, 1.2, 0.18, 0.8
        label_w = max(len(f) for f in feat_labels) * 0.07 + 0.3
        cbar_w = 0.5
        fig_w = max(5.0, n_samples * col_w + label_w + cbar_w + 1.2)
        fig_h = max(4.0, n_features * row_h + dend_h + grp_h + xlabel_h + 1.5)

        fig = Figure(figsize=(fig_w, fig_h), layout='constrained')
        gs = GridSpec(
            4, 3, figure=fig,
            height_ratios=[dend_h, grp_h, n_features * row_h, xlabel_h],
            width_ratios=[label_w, n_samples * col_w, cbar_w],
            hspace=0.02, wspace=0.02,
        )

        ax_cdend = fig.add_subplot(gs[0, 1])
        ax_cdend.axis('off')
        if n_samples > 1 and Zc is not None:
            try:
                dendrogram(Zc, ax=ax_cdend, color_threshold=0,
                           above_threshold_color='#555555',
                           link_color_func=lambda _: '#555555', no_labels=True)
                ax_cdend.set_xlim(-0.5, n_samples * 10 - 0.5)
            except Exception:
                pass
        ax_cdend.set_title('Sample MFI', fontsize=10, pad=4)

        ax_grp = fig.add_subplot(gs[1, 1])
        ax_grp.set_xlim(0, n_samples)
        ax_grp.set_ylim(0, 1)
        ax_grp.axis('off')
        for xi, grp in enumerate(grp_ord):
            color = COLOR_A if grp == name_a else COLOR_B
            ax_grp.add_patch(mpatches.Rectangle((xi, 0), 1, 1, color=color, transform=ax_grp.transData))
        handles = [mpatches.Patch(color=COLOR_A, label=name_a),
                   mpatches.Patch(color=COLOR_B, label=name_b)]
        ax_cdend.legend(handles=handles, loc='lower right', fontsize=7, frameon=True, ncol=2)

        ax_rdend = fig.add_subplot(gs[2, 0])
        ax_rdend.axis('off')
        if n_features > 1 and Zr is not None:
            try:
                dendrogram(Zr, ax=ax_rdend, orientation='left', color_threshold=0,
                           above_threshold_color='#555555',
                           link_color_func=lambda _: '#555555', no_labels=True)
            except Exception:
                pass

        ax_hm = fig.add_subplot(gs[2, 1])
        finite_vals = mat_ord[np.isfinite(mat_ord)]
        if finite_vals.size:
            vmin = float(np.percentile(finite_vals, 5))
            vmax = max(float(np.percentile(finite_vals, 95)), vmin + 0.01)
        else:
            vmin, vmax = 0.0, 1.0
        im = ax_hm.imshow(
            np.nan_to_num(mat_ord, nan=vmin),
            aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax,
            interpolation='nearest',
        )
        ax_hm.set_xticks(range(n_samples))
        ax_hm.set_xticklabels([Path(s).stem for s in samp_labels], rotation=45, ha='right', fontsize=7)
        ax_hm.set_yticks(range(n_features))
        ax_hm.set_yticklabels(feat_labels, fontsize=7)
        ax_hm.yaxis.set_label_position('right')
        ax_hm.yaxis.tick_right()
        ax_hm.grid(False)   # suppress inherited seaborn 'whitegrid' (draws through tick/cell centres)
        ax_hm.set_xticks(np.arange(-0.5, n_samples, 1), minor=True)
        ax_hm.set_yticks(np.arange(-0.5, n_features, 1), minor=True)
        ax_hm.grid(which='minor', color='white', linestyle='-', linewidth=0.6)
        ax_hm.tick_params(which='minor', bottom=False, left=False, right=False)

        ax_cb = fig.add_subplot(gs[2, 2])
        cb = fig.colorbar(im, cax=ax_cb)
        cb.ax.tick_params(labelsize=7)
        cb.set_label('log1p-MFI', fontsize=7)

        _style_figure_theme(fig, is_dark)
        self._stamp_run_label(fig, run_label)
        return fig

    def _make_heatmap_figure(self, results_df, sample_df, title: str,
                             group_a: str = '', group_b: str = '', run_label: str = ''):
        """
        Per-sample heatmap with row + column dendrograms for ONE comparison.

        results_df : that comparison's test output (feature, logFC,
                     significant, …).
        sample_df  : raw feature matrix for the same comparison's samples
                     (rows=samples, cols=features), index=rel-paths.
        group_a/group_b : the two group names this comparison represents.
        run_label  : stamped in the upper-left corner (plot provenance)
        """
        group_by_sample = dict(zip(self.state.stats_all_rel, self.state.stats_group_vec))
        return make_heatmap_figure(
            results_df, sample_df, title, group_by_sample,
            group_a=group_a, group_b=group_b,
            is_dark=_resolve_is_dark(self.state), run_label=run_label,
            group_colours={name: _resolve_group_colour(self.state, name)
                           for name in (group_a, group_b) if name},
        )

    def _make_volcano_figure(self, results_df, title: str,
                              pval_threshold: float, fc_threshold: float,
                              run_label: str = '', group_a: str = '',
                              group_b: str = ''):
        """Volcano plot of one comparison's results (see
        differential_plots.make_volcano_figure). MFI features are
        "{cluster_label}_{channel}"; their channel suffix is shown as the
        Antigen (falling back to Label). group_a/group_b are the
        comparison's baseline and other group, named on the plot in their
        group colours."""
        features = results_df['feature'].tolist()
        if 'MFI' in title:
            disp_map = _antigen_or_label_map(self.controller)
            known_channels = set(disp_map.keys())

            def _remap_feat(feat: str) -> str:
                for ch in known_channels:
                    suffix = f'_{ch}'
                    if feat.endswith(suffix):
                        cluster_part = feat[: -len(suffix)]
                        return f'{cluster_part}_{disp_map[ch]}'
                return feat
            features = [_remap_feat(f) for f in features]
        return make_volcano_figure(
            results_df, title, pval_threshold, fc_threshold,
            is_dark=_resolve_is_dark(self.state),
            fdr_scope=self.state.stats_fdr_scope,
            feature_labels=features, run_label=run_label,
            group_a=group_a, group_b=group_b,
            group_colours={name: _resolve_group_colour(self.state, name)
                           for name in (group_a, group_b) if name},
        )

    def _make_mfi_cluster_summary_figure(self, results_df, title: str,
                                         channel_names: dict, run_label: str = ''):
        """
        Cluster × marker grid for ONE comparison's MFI results, limited to
        clusters with at least one significant marker (see
        differential_plots.make_marker_summary_figure).
        """
        return make_marker_summary_figure(
            results_df, title, channel_names,
            is_dark=_resolve_is_dark(self.state), run_label=run_label,
            family_label='cluster',
        )

    def _export_figure(self, fig, stem: str):
        """Save a matplotlib Figure to disk."""
        fmt = getattr(hc_settings, 'graphics_export_format_retrieved', 'png')
        exp_dir = self.controller.experiment_dir
        out_path = exp_dir / f"{stem}.{fmt}"
        try:
            fig.savefig(out_path, bbox_inches='tight')
            QMessageBox.information(self, "Exported", f"Saved to {out_path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    def _export_results_csv(self):
        """Export all statistics results to a combined CSV."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Statistics Results", "", "CSV files (*.csv)"
        )
        if not path:
            return
        path = _with_extension(path, '.csv')
        try:
            frames = {}
            if self.state.freq_results is not None:
                frames['Frequencies'] = self.state.freq_results
            if self.state.counts_results is not None:
                frames['Counts (GLM)'] = self.state.counts_results
            if self.state.mfi_results is not None:
                frames['MFIs'] = self.state.mfi_results

            if not frames:
                QMessageBox.warning(self, "No Results", "No statistics results to export.")
                return

            with open(path, 'w', newline='') as f:
                for sheet_name, df in frames.items():
                    f.write(f"# {sheet_name}\n")
                    df.to_csv(f, index=False)
                    f.write("\n")
            QMessageBox.information(self, "Exported", f"Saved to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    def _export_tables(self):
        """Write the last statistics run's tables, one CSV file each, into a
        folder named by the user (see drc_stats.export_tables)."""
        run = next((entry for entry in self.state.clustering_runs
                    if entry.get('run_id') == self.state.stats_run_id), None)
        if run is not None:
            run = drc_run_archive.hydrate_run(self.controller, run)
        run = run or {}
        tables = drc_stats.export_tables(
            self.state, run.get('mem_labels'), run.get('cell_type_suggestions'))
        if not tables:
            QMessageBox.warning(self, "No Results", "No statistics results to export.")
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Export Tables to Folder", str(self.controller.experiment_dir))
        if not folder:
            return
        stem = drc_report.sanitize_filename(self.state.stats_run_label or 'honeycluster')
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        try:
            for name, table in tables.items():
                table.to_csv(Path(folder) / f"{stem}_{stamp}_{name}.csv", index=False)
            QMessageBox.information(
                self, "Exported", f"Saved {len(tables)} files to {folder}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))


class WorkspaceTab(QWidget):
    """
    Tab — Workspace
    -------------------------------------------
    Figure composer: pages (A4 by default) on a zoomable canvas
    (drc_workspace_canvas) holding plots, their separate legends, text
    boxes, panel letters and figures copied from the Cluster Annotation and
    Stats tabs, which can be moved, resized, aligned, grouped and stacked,
    with undo. The inspector on the right shows the settings
    of the selected item (a plot's PlotCard, a text box, a group) or, with
    nothing selected, the current page. Pages export as PDF, SVG, PNG or
    TIFF (drc_workspace_export) and are what the Report's Workspace section
    offers.

    Persisted: each plot's layout ({'page', 'x_mm', 'y_mm'}), group and
    stacking order in its entry of state.plot_configs; text boxes and
    letters in state.workspace_texts; figures from other tabs (where they
    come from, not the figure) in state.workspace_figures; pages in
    state.workspace_pages.
    """

    plots_changed = Signal()  # an item was added or removed, or the layout changed

    def __init__(self, state: PipelineState, bus, controller, parent=None):
        super().__init__(parent)
        self.state = state
        self.bus = bus
        self.controller = controller
        self._plot_cards: list['PlotCard'] = []
        self._items: dict[str, drc_workspace_canvas.PlotItem] = {}
        self._texts: dict[str, drc_workspace_canvas.TextItem] = {}
        self._figures: dict[str, drc_workspace_canvas.SourceFigureItem] = {}
        self._figure_sources: dict = {}   # tab name -> callable returning its ReportItems
        self._restoring = False
        self._build_ui()

    def _build_ui(self):
        from PySide6.QtGui import QKeySequence, QShortcut
        from PySide6.QtWidgets import QMenu, QToolButton
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.help_widget = HelpToggleWidget(
            text=drc_help_texts.workspace_tab_help_text
        )
        outer.addWidget(self.help_widget)

        self.scene = drc_workspace_canvas.WorkspaceScene(
            self._pages_from_state(), grid_mm=self.state.workspace_grid_mm,
            snap_enabled=self.state.workspace_snap, show_grid=self.state.workspace_show_grid,
            parent=self)
        self.undo_stack = self.scene.undo_stack

        # --- Toolbar row 1: insert, undo, arrange, export ---
        row1 = QFrame()
        row1.setFrameShape(QFrame.StyledPanel)
        tb1 = QHBoxLayout(row1)
        tb1.setContentsMargins(8, 4, 8, 2)
        tb1.setSpacing(6)
        self.add_plot_btn = QPushButton("＋ Add Plot")
        self.add_plot_btn.clicked.connect(lambda _checked=False: self.add_plot())
        tb1.addWidget(self.add_plot_btn)
        self.add_text_btn = QPushButton("＋ Text")
        self.add_text_btn.clicked.connect(lambda _checked=False: self.add_text('text'))
        tb1.addWidget(self.add_text_btn)
        self.add_letter_btn = QPushButton("＋ Panel Letter")
        self.add_letter_btn.setToolTip(
            "Add a panel letter (A, B, C … in reading order on the page) at the "
            "top-left of the selected plot or group")
        self.add_letter_btn.clicked.connect(lambda _checked=False: self.add_text('letter'))
        tb1.addWidget(self.add_letter_btn)
        self.add_figure_btn = QToolButton()
        self.add_figure_btn.setText("＋ From Other Tabs ▾")
        self.add_figure_btn.setToolTip(
            "Add a figure from the Cluster Annotation or Stats tab (as it is shown there now)")
        self.add_figure_btn.setPopupMode(QToolButton.InstantPopup)
        self._figure_menu = QMenu(self.add_figure_btn)
        self._figure_menu.aboutToShow.connect(
            lambda: self._fill_figure_menu(self._figure_menu, None))
        self.add_figure_btn.setMenu(self._figure_menu)
        tb1.addWidget(self.add_figure_btn)

        tb1.addSpacing(12)
        self.undo_btn = QPushButton("↶ Undo")
        self.undo_btn.clicked.connect(self.undo_stack.undo)
        self.undo_stack.canUndoChanged.connect(self.undo_btn.setEnabled)
        self.undo_btn.setEnabled(False)
        tb1.addWidget(self.undo_btn)
        self.redo_btn = QPushButton("↷ Redo")
        self.redo_btn.clicked.connect(self.undo_stack.redo)
        self.undo_stack.canRedoChanged.connect(self.redo_btn.setEnabled)
        self.redo_btn.setEnabled(False)
        tb1.addWidget(self.redo_btn)

        tb1.addSpacing(12)
        self.arrange_btn = QToolButton()
        self.arrange_btn.setText("Arrange ▾")
        self.arrange_btn.setPopupMode(QToolButton.InstantPopup)
        arrange = QMenu(self.arrange_btn)
        for mode, text in (('left', "Align left edges"), ('hcentre', "Align centres (horizontal)"),
                           ('right', "Align right edges"), (None, None),
                           ('top', "Align top edges"), ('vcentre', "Align centres (vertical)"),
                           ('bottom', "Align bottom edges"), (None, None),
                           ('hdistribute', "Distribute horizontally"),
                           ('vdistribute', "Distribute vertically")):
            if mode is None:
                arrange.addSeparator()
            else:
                arrange.addAction(text, lambda m=mode: self.align_selected(m))
        arrange.addSeparator()
        arrange.addAction("Group", self.group_selected)
        arrange.addAction("Ungroup", self.ungroup_selected)
        arrange.addSeparator()
        arrange.addAction("Bring to front", lambda: self.restack_selected(True))
        arrange.addAction("Send to back", lambda: self.restack_selected(False))
        self.arrange_btn.setMenu(arrange)
        tb1.addWidget(self.arrange_btn)

        tb1.addStretch()
        self.export_btn = QPushButton("Export…")
        self.export_btn.setToolTip("Export pages as PDF, SVG, PNG or TIFF")
        self.export_btn.clicked.connect(self._export_pages)
        tb1.addWidget(self.export_btn)
        outer.addWidget(row1)

        # --- Toolbar row 2: pages, zoom, grid, theme ---
        row2 = QFrame()
        row2.setFrameShape(QFrame.StyledPanel)
        tb_layout = QHBoxLayout(row2)
        tb_layout.setContentsMargins(8, 2, 8, 4)
        tb_layout.setSpacing(6)
        self.add_page_btn = QPushButton("＋ Page")
        self.add_page_btn.setToolTip("Add a page after the current page")
        self.add_page_btn.clicked.connect(self._add_page)
        tb_layout.addWidget(self.add_page_btn)
        self.remove_page_btn = QPushButton("− Page")
        self.remove_page_btn.setToolTip("Remove the current page and everything on it")
        self.remove_page_btn.clicked.connect(self._remove_page)
        tb_layout.addWidget(self.remove_page_btn)

        tb_layout.addSpacing(12)
        zoom_out = QPushButton("−")
        zoom_out.setFixedWidth(28)
        zoom_out.setToolTip("Zoom out (Ctrl/⌘ + wheel, or pinch)")
        zoom_out.clicked.connect(
            lambda: self.view.set_zoom(self.view.zoom() / drc_workspace_canvas.ZOOM_STEP))
        tb_layout.addWidget(zoom_out)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(44)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        tb_layout.addWidget(self.zoom_label)
        zoom_in = QPushButton("+")
        zoom_in.setFixedWidth(28)
        zoom_in.setToolTip("Zoom in (Ctrl/⌘ + wheel, or pinch)")
        zoom_in.clicked.connect(
            lambda: self.view.set_zoom(self.view.zoom() * drc_workspace_canvas.ZOOM_STEP))
        tb_layout.addWidget(zoom_in)
        actual = QPushButton("100%")
        actual.setToolTip("Show pages at their printed size")
        actual.clicked.connect(lambda: self.view.set_zoom(1.0))
        tb_layout.addWidget(actual)
        fit = QPushButton("Fit Page")
        fit.clicked.connect(lambda: self.view.fit_page(self.scene.current_page))
        tb_layout.addWidget(fit)

        tb_layout.addSpacing(12)
        self.snap_check = QCheckBox("Snap")
        self.snap_check.setToolTip("Snap to the grid (hold Alt/⌥ to place freely)")
        self.snap_check.setChecked(self.state.workspace_snap)
        self.snap_check.toggled.connect(self._on_grid_changed)
        tb_layout.addWidget(self.snap_check)
        self.grid_combo = QComboBox()
        for step in drc_workspace_canvas.GRID_STEPS_MM:
            self.grid_combo.addItem(f"{step:g} mm", step)
        idx = self.grid_combo.findData(self.state.workspace_grid_mm)
        self.grid_combo.setCurrentIndex(
            idx if idx >= 0 else self.grid_combo.findData(drc_workspace_canvas.DEFAULT_GRID_MM))
        self.grid_combo.setToolTip(
            "Snap step (also the arrow-key nudge). Show grid draws these steps; "
            "when zoomed out, every 2nd, 3rd … step so lines stay apart.")
        self.grid_combo.currentIndexChanged.connect(self._on_grid_changed)
        _style_combo_popup(self.grid_combo)
        tb_layout.addWidget(self.grid_combo)
        self.show_grid_check = QCheckBox("Show grid")
        self.show_grid_check.setChecked(self.state.workspace_show_grid)
        self.show_grid_check.toggled.connect(self._on_grid_changed)
        tb_layout.addWidget(self.show_grid_check)

        tb_layout.addSpacing(12)
        tb_layout.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Auto", "Light", "Dark"])
        self.theme_combo.setToolTip(
            "Plot background for every plot in this plugin. Auto mirrors "
            "the app's own light/dark setting; Light/Dark forces it."
        )
        self.theme_combo.setCurrentText(self.state.plot_theme.capitalize())
        self.theme_combo.setFixedWidth(80)
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        _style_combo_popup(self.theme_combo)
        tb_layout.addWidget(self.theme_combo)
        tb_layout.addStretch()
        outer.addWidget(row2)

        # --- Canvas | inspector ---
        self.view = drc_workspace_canvas.WorkspaceView(self.scene)
        self.view.zoom_changed.connect(lambda z: self.zoom_label.setText(f"{z * 100:.0f}%"))
        self.zoom_label.setText(f"{self.view.zoom() * 100:.0f}%")
        self.scene.selectionChanged.connect(self._on_selection_changed)
        self.scene.layout_changed.connect(self._on_layout_changed)
        self.scene.current_page_changed.connect(self._on_current_page_changed)
        self.scene.group_scale_requested.connect(self.scale_group)
        self.scene.page_menu_requested.connect(self._show_page_menu)

        self.inspector_title = QLabel()
        self.inspector_title.setStyleSheet("font-weight: bold; padding: 4px;")
        self.inspector_stack = QStackedWidget()
        self.inspector_stack.addWidget(self._build_page_panel())    # 0
        self.inspector_stack.addWidget(self._build_text_panel())    # 1
        self.inspector_stack.addWidget(self._build_group_panel())   # 2
        self.inspector_stack.addWidget(self._build_figure_panel())  # 3
        inspector_scroll = QScrollArea()
        inspector_scroll.setWidgetResizable(True)
        inspector_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inspector_scroll.setWidget(self.inspector_stack)
        inspector = QWidget()
        inspector_layout = QVBoxLayout(inspector)
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        inspector_layout.addWidget(self.inspector_title)
        inspector_layout.addWidget(inspector_scroll, stretch=1)
        inspector.setMinimumWidth(INSPECTOR_MIN_WIDTH_PX)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.view)
        splitter.addWidget(inspector)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        outer.addWidget(splitter, stretch=1)

        # --- Shortcuts (active while the canvas has focus) ---
        for keys, slot in ((QKeySequence.StandardKey.Undo, self.undo_stack.undo),
                           (QKeySequence.StandardKey.Redo, self.undo_stack.redo),
                           (QKeySequence("Ctrl+G"), self.group_selected),
                           (QKeySequence("Ctrl+Shift+G"), self.ungroup_selected)):
            shortcut = QShortcut(keys, self.view)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)
        for key, (dx, dy) in ((Qt.Key_Left, (-1, 0)), (Qt.Key_Right, (1, 0)),
                              (Qt.Key_Up, (0, -1)), (Qt.Key_Down, (0, 1))):
            shortcut = QShortcut(QKeySequence(key), self.view)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda dx=dx, dy=dy: self.nudge_selected(dx, dy))

        self._show_page_panel()

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------

    def _pages_from_state(self) -> list[dict]:
        return [dict(p) for p in self.state.workspace_pages] or [
            dict(drc_workspace_canvas.DEFAULT_PAGE)]

    def _build_page_panel(self) -> QWidget:
        from PySide6.QtWidgets import QFormLayout
        panel = QWidget()
        form = QFormLayout(panel)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.page_spin = QSpinBox()
        self.page_spin.setMinimum(1)
        self.page_spin.valueChanged.connect(lambda v: self._on_current_page_changed(v - 1))
        form.addRow("Page:", self.page_spin)
        self.page_preset_combo = QComboBox()
        self.page_preset_combo.addItems(list(drc_workspace_canvas.PAGE_PRESETS))
        _style_combo_popup(self.page_preset_combo)
        form.addRow("Size:", self.page_preset_combo)
        self.page_orient_combo = QComboBox()
        self.page_orient_combo.addItems(['Portrait', 'Landscape'])
        _style_combo_popup(self.page_orient_combo)
        form.addRow("Orientation:", self.page_orient_combo)
        self.page_w_spin = QDoubleSpinBox()
        self.page_h_spin = QDoubleSpinBox()
        for spin in (self.page_w_spin, self.page_h_spin):
            spin.setRange(PAGE_MIN_MM, PAGE_MAX_MM)
            spin.setDecimals(1)
            spin.setSuffix(" mm")
        form.addRow("Width:", self.page_w_spin)
        form.addRow("Height:", self.page_h_spin)
        hint = QLabel("Select an item to edit it; right-click it for more actions. "
                      "Ctrl/⌘ + wheel or pinch zooms; arrow keys nudge by the grid step; "
                      "Ctrl/⌘ + G groups.")
        hint.setWordWrap(True)
        hint.setStyleSheet(_hint_style())
        form.addRow(hint)
        self.page_preset_combo.currentTextChanged.connect(self._on_page_settings_changed)
        self.page_orient_combo.currentTextChanged.connect(self._on_page_settings_changed)
        self.page_w_spin.valueChanged.connect(self._on_page_settings_changed)
        self.page_h_spin.valueChanged.connect(self._on_page_settings_changed)
        self._page_panel_loading = False
        return panel

    def _show_page_panel(self, title: str | None = None):
        index = self.scene.current_page
        page = self.scene.pages[index]
        self._page_panel_loading = True
        try:
            self.page_spin.setMaximum(len(self.scene.pages))
            self.page_spin.setValue(index + 1)
            self.page_preset_combo.setCurrentText(page.get('preset', 'Custom'))
            self.page_orient_combo.setCurrentText(
                'Landscape' if page['w_mm'] > page['h_mm'] else 'Portrait')
            self.page_w_spin.setValue(page['w_mm'])
            self.page_h_spin.setValue(page['h_mm'])
            custom = self.page_preset_combo.currentText() == 'Custom'
            self.page_w_spin.setEnabled(custom)
            self.page_h_spin.setEnabled(custom)
        finally:
            self._page_panel_loading = False
        self.inspector_stack.setCurrentIndex(0)
        self.inspector_title.setText(title or f"Page {index + 1} of {len(self.scene.pages)}")
        self.remove_page_btn.setEnabled(len(self.scene.pages) > 1)

    def _on_page_settings_changed(self, *_):
        if self._page_panel_loading:
            return
        preset = self.page_preset_combo.currentText()
        w, h = drc_workspace_canvas.page_size_for(
            preset, self.page_orient_combo.currentText(),
            (self.page_w_spin.value(), self.page_h_spin.value()))
        self.scene.set_page(self.scene.current_page, {'preset': preset, 'w_mm': w, 'h_mm': h})
        self._show_page_panel()

    def _on_current_page_changed(self, index: int):
        if self._page_panel_loading or not 0 <= index < len(self.scene.pages):
            return
        self.scene.current_page = index
        if self.inspector_stack.currentIndex() == 0:
            self._show_page_panel()

    def _add_page(self):
        self.scene.add_page(self.scene.current_page)
        self.scene.current_page += 1
        self._show_page_panel()
        self.view.fit_page(self.scene.current_page)

    def _remove_page(self):
        """Remove the current page and what is on it (asks first if it is
        not empty; not undoable)."""
        index = self.scene.current_page
        if len(self.scene.pages) <= 1:
            return
        on_page = [it for it in self.scene.top_items() if it.layout_mm['page'] == index]
        if on_page:
            answer = QMessageBox.question(
                self, "Remove page",
                f"Page {index + 1} holds {len(on_page)} item(s). Remove the page and them? "
                "This cannot be undone.")
            if answer != QMessageBox.Yes:
                return
        for item in on_page:
            self._discard_item(item)
        self.scene.remove_page(index)
        self.undo_stack.clear()
        self._show_page_panel()
        self.plots_changed.emit()

    def _on_grid_changed(self, *_):
        self.state.workspace_snap = self.snap_check.isChecked()
        self.state.workspace_grid_mm = float(self.grid_combo.currentData())
        self.state.workspace_show_grid = self.show_grid_check.isChecked()
        self.scene.set_grid(self.state.workspace_grid_mm, self.state.workspace_snap,
                            self.state.workspace_show_grid)
        self.plots_changed.emit()

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _next_id(self, prefix: str, used) -> str:
        n = 1
        while f"{prefix}_{n}" in used:
            n += 1
        return f"{prefix}_{n}"

    def _place(self, size_mm: tuple[float, float]) -> dict:
        pages, (layout,) = drc_workspace_canvas.place_items(
            self.scene.pages, self.scene.placed_layouts(), [size_mm],
            start_page=self.scene.current_page)
        if len(pages) != len(self.scene.pages):
            self.scene.set_pages(pages)
        return layout

    def add_plot(self, refresh: bool = True, config: dict | None = None,
                 layout: dict | None = None, plot_id: str | None = None,
                 record: bool = True) -> 'PlotCard':
        """
        Add a plot: a PlotCard (its settings, shown in the inspector) and a
        PlotItem on the page. *config* is a saved display config to apply;
        *layout* ({'page', 'x_mm', 'y_mm'}) places it, otherwise it goes in
        the first free spot from the current page on (a page is added if
        none is free). record=True makes it one undoable step.
        """
        card = PlotCard(
            plot_id=plot_id or self._next_id('plot', {c.plot_id for c in self._plot_cards}),
            state=self.state,
            bus=self.bus,
            controller=self.controller,
            workspace=self,
        )
        if config:
            card.apply_size_config(config)
            card.group_id = config.get('group')
            card.z_value = float(config.get('z') or 0.0)
        item = drc_workspace_canvas.PlotItem(card)
        item.action_requested.connect(self._on_item_action)
        if layout is None:
            layout = self._place(card.estimated_size_mm())
        item.layout_mm = {'page': int(layout['page']), 'x_mm': float(layout['x_mm']),
                          'y_mm': float(layout['y_mm'])}
        item.group_id = card.group_id
        item.setZValue(card.z_value)
        card.layout_mm = item.layout_mm
        self.scene.addItem(item)
        self.scene.position_item(item)
        self._plot_cards.append(card)
        self._items[card.plot_id] = item
        self.inspector_stack.addWidget(card)
        self.state.plot_configs.append({'plot_id': card.plot_id, 'layout': dict(item.layout_mm)})
        if config:
            card.apply_config(config)
        elif refresh:
            card.refresh()
        item.schedule_render()
        if record and not self._restoring:
            self._record_added([card.plot_id], "Add plot")
        self.plots_changed.emit()
        return card

    def remove_card(self, card: 'PlotCard'):
        """Remove a plot (undoable)."""
        item = self._items.get(card.plot_id)
        if item is not None:
            self.remove_items([item])

    def _discard_item(self, item):
        """Remove a plot or text item for good (no undo record)."""
        if isinstance(item, drc_workspace_canvas.PlotItem):
            card = item.card
            if card in self._plot_cards:
                self._plot_cards.remove(card)
            self._items.pop(card.plot_id, None)
            self.state.plot_configs = [
                c for c in self.state.plot_configs if c.get('plot_id') != card.plot_id]
            self.inspector_stack.removeWidget(card)
            card.deleteLater()
        elif isinstance(item, drc_workspace_canvas.TextItem):
            self._texts.pop(item.uid, None)
        elif isinstance(item, drc_workspace_canvas.SourceFigureItem):
            self._figures.pop(item.uid, None)
        if item.scene() is not None:
            self.scene.removeItem(item)
        item.deleteLater()

    def clear_plots(self):
        for item in list(self.scene.top_items()):
            self._discard_item(item)
        self.undo_stack.clear()

    # ------------------------------------------------------------------
    # Text boxes and panel letters
    # ------------------------------------------------------------------

    def add_text(self, kind: str = 'text', spec: dict | None = None,
                 layout: dict | None = None, uid: str | None = None,
                 record: bool = True) -> 'drc_workspace_canvas.TextItem':
        """Add a text box or panel letter. A new letter goes at the top-left
        of the selected plot or group (else in a free spot) and takes the
        next letter in reading order on its page."""
        spec = dict(spec or {}, kind=kind)
        box = drc_workspace_canvas.TextBox(uid or self._next_id('text', set(self._texts)), spec)
        item = drc_workspace_canvas.TextItem(box)
        item.action_requested.connect(self._on_item_action)
        if layout is None:
            layout = self._new_text_layout(item, near_selection=(kind == 'letter'))
            if kind == 'letter' and box.spec['auto']:
                box.set_spec(text=self.scene.next_letter(
                    layout['page'], layout['x_mm'], layout['y_mm']))
        item.layout_mm = {'page': int(layout['page']), 'x_mm': float(layout['x_mm']),
                          'y_mm': float(layout['y_mm'])}
        item.group_id = spec.get('group')
        item.setZValue(float(spec.get('z') or 0.0))
        self.scene.addItem(item)
        self.scene.position_item(item)
        self._texts[box.uid] = item
        box.changed.connect(self._on_layout_changed)
        if record and not self._restoring:
            self.scene.relabel_letters()
            self._record_added([box.uid], "Add text" if kind == 'text' else "Add panel letter")
            self.scene.clearSelection()
            item.setSelected(True)
        self._on_layout_changed()
        return item

    def _new_text_layout(self, item, near_selection: bool) -> dict:
        """Where a new text item goes: a panel letter at the top-left of the
        selected plot or group; otherwise the last point clicked on a page
        (the next one goes just below it); otherwise the first free spot."""
        anchor = [it for it in self.scene.selected_figure_items() if it.parentItem() is None]
        if anchor and near_selection:
            gid = self.scene.selected_group()
            rect = (self.scene.group_rect(gid) if gid else anchor[0].scene_rect_mm())
            page = self.scene.page_index_at(rect.center().y())
            ox, oy = self.scene.page_origin(page)
            off = drc_workspace_canvas.LETTER_OFFSET_MM
            return {'page': page, 'x_mm': max(0.0, rect.x() - ox - off),
                    'y_mm': max(0.0, rect.y() - oy - off)}
        return self._layout_at_last_click(item.size_mm())

    def _layout_at_last_click(self, size_mm: tuple[float, float]) -> dict:
        """At the last point clicked on a page if the item fits on the page
        there; otherwise the first free spot."""
        point = self.scene.last_click
        if point is not None:
            page, x, y = point
            p = self.scene.pages[min(page, len(self.scene.pages) - 1)]
            if x + size_mm[0] <= p['w_mm'] and y + size_mm[1] <= p['h_mm']:
                self.scene.take_last_click(size_mm[1])
                return {'page': page, 'x_mm': x, 'y_mm': y}
        return self._place(size_mm)

    def _build_text_panel(self) -> QWidget:
        from PySide6.QtWidgets import QFormLayout, QPlainTextEdit
        panel = QWidget()
        form = QFormLayout(panel)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.text_edit = QPlainTextEdit()
        self.text_edit.setFixedHeight(80)
        self.text_edit.textChanged.connect(self._on_text_settings_changed)
        form.addRow("Text:", self.text_edit)
        self.text_auto_check = QCheckBox("Automatic letter (reading order)")
        self.text_auto_check.toggled.connect(self._on_text_settings_changed)
        form.addRow(self.text_auto_check)
        self.text_size_spin = QDoubleSpinBox()
        self.text_size_spin.setRange(4.0, 72.0)
        self.text_size_spin.setDecimals(1)
        self.text_size_spin.setSuffix(" pt")
        self.text_size_spin.valueChanged.connect(self._on_text_settings_changed)
        form.addRow("Size:", self.text_size_spin)
        style_row = QHBoxLayout()
        self.text_bold_check = QCheckBox("Bold")
        self.text_bold_check.toggled.connect(self._on_text_settings_changed)
        style_row.addWidget(self.text_bold_check)
        self.text_italic_check = QCheckBox("Italic")
        self.text_italic_check.toggled.connect(self._on_text_settings_changed)
        style_row.addWidget(self.text_italic_check)
        style_row.addStretch()
        form.addRow(style_row)
        self.text_align_combo = QComboBox()
        self.text_align_combo.addItems(['left', 'centre', 'right'])
        self.text_align_combo.currentTextChanged.connect(self._on_text_settings_changed)
        _style_combo_popup(self.text_align_combo)
        form.addRow("Lines:", self.text_align_combo)
        self.text_colour_btn = QPushButton()
        self.text_colour_btn.clicked.connect(self._pick_text_colour)
        form.addRow("Colour:", self.text_colour_btn)
        self._text_panel_item = None
        self._text_panel_loading = False
        return panel

    def _show_text_panel(self, item):
        spec = item.box.spec
        self._text_panel_item = None
        self._text_panel_loading = True
        try:
            self.text_edit.setPlainText(spec['text'])
            is_letter = spec['kind'] == 'letter'
            self.text_auto_check.setVisible(is_letter)
            self.text_auto_check.setChecked(bool(spec['auto']))
            self.text_size_spin.setValue(spec['size_pt'])
            self.text_bold_check.setChecked(spec['bold'])
            self.text_italic_check.setChecked(spec['italic'])
            self.text_align_combo.setCurrentText(spec['align'])
            self.text_colour_btn.setText(spec['colour'])
            self.text_colour_btn.setStyleSheet(
                f"background-color: {spec['colour']}; color: {_contrasting_text_color(spec['colour'])};")
        finally:
            self._text_panel_loading = False
        self._text_panel_item = item
        self.inspector_stack.setCurrentIndex(1)
        self.inspector_title.setText("Panel letter" if is_letter else "Text")

    def _on_text_settings_changed(self, *_):
        item = self._text_panel_item
        if self._text_panel_loading or item is None:
            return
        spec = item.box.spec
        text = self.text_edit.toPlainText()
        auto = self.text_auto_check.isChecked()
        if spec['kind'] == 'letter' and text != spec['text'] and auto == spec['auto']:
            auto = False   # a letter typed by hand is kept
            self._text_panel_loading = True
            self.text_auto_check.setChecked(False)
            self._text_panel_loading = False
        item.box.set_spec(text=text, auto=auto, size_pt=self.text_size_spin.value(),
                          bold=self.text_bold_check.isChecked(),
                          italic=self.text_italic_check.isChecked(),
                          align=self.text_align_combo.currentText())
        if spec['kind'] == 'letter':
            self.scene.relabel_letters()
            if auto:
                self._text_panel_loading = True
                self.text_edit.setPlainText(item.box.spec['text'])
                self._text_panel_loading = False

    def _pick_text_colour(self):
        item = self._text_panel_item
        if item is None:
            return
        colour = QColorDialog.getColor(QColor(item.box.spec['colour']), self, "Text colour")
        if colour.isValid():
            item.box.set_spec(colour=colour.name())
            self._show_text_panel(item)

    # ------------------------------------------------------------------
    # Groups
    # ------------------------------------------------------------------

    def _build_group_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.group_label = QLabel()
        self.group_label.setWordWrap(True)
        layout.addWidget(self.group_label)
        ungroup = QPushButton("Ungroup")
        ungroup.clicked.connect(self.ungroup_selected)
        layout.addWidget(ungroup)
        hint = QLabel("Drag the group to move it; drag its orange corner to scale it "
                      "(plots are redrawn at the new size, text keeps its size). "
                      "Double-click an item to select it alone and edit it.")
        hint.setWordWrap(True)
        hint.setStyleSheet(_hint_style())
        layout.addWidget(hint)
        layout.addStretch()
        return panel

    def _selected_top_items(self) -> list:
        return [it for it in self.scene.selected_figure_items() if it.parentItem() is None]

    def group_selected(self):
        items = self._selected_top_items()
        if len(items) < 2:
            return
        used = {it.group_id for it in self.scene.top_items() if it.group_id}
        gid = self._next_id('group', used)
        self._set_groups_undoable({it.uid: gid for it in items}, "Group")

    def ungroup_selected(self):
        items = [it for it in self._selected_top_items() if it.group_id]
        gids = {it.group_id for it in items}
        members = [m for g in gids for m in self.scene.members(g)]
        if members:
            self._set_groups_undoable({m.uid: None for m in members}, "Ungroup")

    def _set_groups_undoable(self, after: dict, text: str):
        before = {uid: self.scene.item_by_uid(uid).group_id for uid in after}
        self.undo_stack.push(drc_workspace_canvas.CallbackCommand(
            text, lambda: self.scene.set_groups(after), lambda: self.scene.set_groups(before)))
        self._on_selection_changed()

    def scale_group(self, gid: str, scale: float):
        """Scale a group about its top-left corner: positions scale, plots
        are resized (fonts stay in pt), text keeps its size."""
        members = self.scene.members(gid)
        origin = self.scene.group_rect(gid).topLeft()

        def snapshot():
            return {m.uid: {'pos': (m.pos().x(), m.pos().y()),
                            'size': m.card.plot_size_mm() if isinstance(
                                m, drc_workspace_canvas.PlotItem) else None,
                            'legend': m.card.legend_offset_mm if isinstance(
                                m, drc_workspace_canvas.PlotItem) else None}
                    for m in members}

        before = snapshot()
        after = {}
        for uid, s in before.items():
            x, y = s['pos']
            size = s['size']
            legend = s['legend']
            after[uid] = {
                'pos': (origin.x() + (x - origin.x()) * scale,
                        origin.y() + (y - origin.y()) * scale),
                'size': (size[0] * scale, size[1] * scale) if size else None,
                'legend': (legend[0] * scale, legend[1] * scale) if legend else None,
            }

        def apply(states):
            for uid, s in states.items():
                item = self.scene.item_by_uid(uid)
                if item is None:
                    continue
                if isinstance(item, drc_workspace_canvas.PlotItem):
                    item.card.legend_offset_mm = s['legend']
                    if s['size']:
                        item.card.set_plot_size_mm(*s['size'])
            self.scene.set_positions({uid: s['pos'] for uid, s in states.items()})

        self.undo_stack.push(drc_workspace_canvas.CallbackCommand(
            "Scale group", lambda: apply(after), lambda: apply(before)))

    # ------------------------------------------------------------------
    # Arrange
    # ------------------------------------------------------------------

    def _units(self) -> dict:
        """Selected items as units to arrange: {key: (rect, [uids])}; a
        fully selected group is one unit."""
        units = {}
        items = self._selected_top_items()
        for it in items:
            gid = it.group_id
            if gid and set(self.scene.members(gid)) <= set(items):
                units[f'group:{gid}'] = (self.scene.group_rect(gid),
                                         [m.uid for m in self.scene.members(gid)])
            else:
                units[it.uid] = (it.scene_rect_mm(), [it.uid])
        return units

    def align_selected(self, mode: str):
        units = self._units()
        rects = {k: (r.x(), r.y(), r.width(), r.height()) for k, (r, _uids) in units.items()}
        moves = drc_workspace_canvas.align_moves(rects, mode)
        deltas = {uid: moves[k] for k, (_r, uids) in units.items() for uid in uids}
        self.scene.move_items(deltas, "Align" if 'distribute' not in mode else "Distribute")

    def restack_selected(self, to_front: bool):
        items = self._selected_top_items()
        if not items:
            return
        zs = [it.zValue() for it in self.scene.top_items()]
        base = (max(zs) + 1) if to_front else (min(zs) - len(items))
        ordered = sorted(items, key=lambda it: it.zValue())
        before = {it.uid: it.zValue() for it in ordered}
        after = {it.uid: base + i for i, it in enumerate(ordered)}

        def apply(zmap):
            for uid, z in zmap.items():
                item = self.scene.item_by_uid(uid)
                if item is not None:
                    item.setZValue(z)
            self._on_layout_changed()

        self.undo_stack.push(drc_workspace_canvas.CallbackCommand(
            "Bring to front" if to_front else "Send to back",
            lambda: apply(after), lambda: apply(before)))

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            items = self._selected_top_items()
            legends = [it for it in self.scene.selected_figure_items()
                       if isinstance(it, drc_workspace_canvas.LegendItem)]
            for legend in legends:
                legend.card._legend_pos_combo.setCurrentText('Off')
            if items:
                self.remove_items(items)
            if items or legends:
                event.accept()
                return
        super().keyPressEvent(event)

    def nudge_selected(self, dx: int, dy: int):
        """Move the selection by one grid step (undoable)."""
        step = self.state.workspace_grid_mm
        self.scene.move_items({it.uid: (dx * step, dy * step)
                               for it in self.scene.selected_figure_items()}, "Nudge")

    # ------------------------------------------------------------------
    # Adding / removing as undoable steps
    # ------------------------------------------------------------------

    def _snapshot(self, uid: str) -> dict | None:
        item = self.scene.item_by_uid(uid)
        if isinstance(item, drc_workspace_canvas.PlotItem):
            item.card.layout_mm = dict(item.layout_mm)
            item.card.group_id, item.card.z_value = item.group_id, item.zValue()
            return {'kind': 'plot', 'config': item.card.get_display_config()}
        if isinstance(item, drc_workspace_canvas.TextItem):
            return {'kind': 'text', 'spec': dict(item.box.spec, group=item.group_id,
                                                 z=item.zValue()),
                    'layout': dict(item.layout_mm)}
        if isinstance(item, drc_workspace_canvas.SourceFigureItem):
            return {'kind': 'figure', 'spec': dict(item.figure_source.spec, group=item.group_id,
                                                   z=item.zValue()),
                    'layout': dict(item.layout_mm)}
        return None

    def _restore_snapshot(self, uid: str, snap: dict):
        if snap['kind'] == 'plot':
            cfg = snap['config']
            self.add_plot(config=cfg, layout=cfg.get('layout'), plot_id=uid, record=False)
        elif snap['kind'] == 'figure':
            self.add_source_figure(snap['spec'], layout=snap['layout'], uid=uid, record=False)
        else:
            spec = snap['spec']
            self.add_text(spec['kind'], spec=spec, layout=snap['layout'], uid=uid, record=False)

    def _record_added(self, uids: list[str], text: str):
        snaps = {}

        def undo():
            for uid in uids:
                snaps[uid] = self._snapshot(uid)
                item = self.scene.item_by_uid(uid)
                if item is not None:
                    self._discard_item(item)
            self.scene.relabel_letters()
            self.plots_changed.emit()

        def redo():
            for uid in uids:
                if snaps.get(uid):
                    self._restore_snapshot(uid, snaps[uid])
            self.scene.relabel_letters()

        self.undo_stack.push(drc_workspace_canvas.CallbackCommand(text, redo, undo, done=True))

    def remove_items(self, items: list):
        """Remove plots / text items as one undoable step."""
        uids = [it.uid for it in items]
        snaps = {uid: self._snapshot(uid) for uid in uids}

        def redo():
            for uid in uids:
                item = self.scene.item_by_uid(uid)
                if item is not None:
                    snaps[uid] = self._snapshot(uid)
                    self._discard_item(item)
            self.scene.relabel_letters()
            self._show_page_panel()
            self.plots_changed.emit()

        def undo():
            for uid in uids:
                if snaps.get(uid):
                    self._restore_snapshot(uid, snaps[uid])
            self.scene.relabel_letters()

        self.undo_stack.push(drc_workspace_canvas.CallbackCommand(
            f"Remove {len(uids)} item(s)", redo, undo))

    # ------------------------------------------------------------------
    # Restore / persist
    # ------------------------------------------------------------------

    def restore_layout(self, configs: list[dict], texts: list[dict] | None = None,
                       figures: list[dict] | None = None):
        """Rebuild the canvas from state.workspace_pages, saved plot
        *configs*, saved *texts* and saved *figures* from other tabs (shown
        as placeholders until their source can be found). Configs saved before the page canvas (no
        'layout') are placed in order in the first free spots, adding pages
        as needed."""
        self._restoring = True
        try:
            self.clear_plots()
            self.scene.current_page = 0
            self.scene.set_pages(self._pages_from_state())
            self._sync_grid_controls()
            for cfg in configs:
                self.add_plot(refresh=False, config=cfg, layout=cfg.get('layout'),
                              plot_id=cfg.get('plot_id') or None)
            for t in texts or []:
                self.add_text(t.get('kind', 'text'), spec=t, layout=t.get('layout'),
                              uid=t.get('uid'))
            for f in figures or []:
                if f.get('layout'):
                    self.add_source_figure(f, layout=f['layout'], uid=f.get('uid'), record=False)
        finally:
            self._restoring = False
        self.undo_stack.clear()
        self.scene.relabel_letters()
        self._on_layout_changed()
        self._show_page_panel()
        self.view.fit_page(0)

    def _sync_grid_controls(self):
        for widget in (self.snap_check, self.grid_combo, self.show_grid_check):
            widget.blockSignals(True)
        self.snap_check.setChecked(self.state.workspace_snap)
        idx = self.grid_combo.findData(self.state.workspace_grid_mm)
        self.grid_combo.setCurrentIndex(
            idx if idx >= 0 else self.grid_combo.findData(drc_workspace_canvas.DEFAULT_GRID_MM))
        self.show_grid_check.setChecked(self.state.workspace_show_grid)
        for widget in (self.snap_check, self.grid_combo, self.show_grid_check):
            widget.blockSignals(False)
        self.scene.set_grid(self.state.workspace_grid_mm, self.state.workspace_snap,
                            self.state.workspace_show_grid)

    def _on_layout_changed(self):
        """Pages, positions, groups, stacking or text changed: keep state
        in step."""
        if self._restoring:
            return
        self.state.workspace_pages = [dict(p) for p in self.scene.pages]
        for card in self._plot_cards:
            item = self._items[card.plot_id]
            card.layout_mm = dict(item.layout_mm)
            card.group_id, card.z_value = item.group_id, item.zValue()
            card._sync_config_to_state()
        self.state.workspace_texts = [
            dict(it.box.spec, uid=it.uid, layout=dict(it.layout_mm), group=it.group_id,
                 z=it.zValue())
            for it in sorted(self._texts.values(), key=lambda t: t.uid)]
        self.state.workspace_figures = [
            dict(it.figure_source.spec, uid=it.uid, layout=dict(it.layout_mm),
                 group=it.group_id, z=it.zValue())
            for it in sorted(self._figures.values(), key=lambda f: f.uid)]
        self.plots_changed.emit()

    # ------------------------------------------------------------------
    # Selection and item actions
    # ------------------------------------------------------------------

    def _on_selection_changed(self):
        from shiboken6 import isValid
        if not isValid(self.scene):
            return   # the scene is being destroyed
        gid = self.scene.selected_group()
        selected = self.scene.selected_figure_items()
        if gid:
            self.group_label.setText(f"Group of {len(self.scene.members(gid))} items")
            self.inspector_stack.setCurrentIndex(2)
            self.inspector_title.setText("Group")
            return
        if len(selected) == 1:
            item = selected[0]
            if isinstance(item, drc_workspace_canvas.TextItem):
                self._show_text_panel(item)
                return
            if isinstance(item, drc_workspace_canvas.SourceFigureItem):
                self._show_figure_panel(item)
                return
            card = item.card
            self.inspector_stack.setCurrentWidget(card)
            suffix = " legend" if isinstance(item, drc_workspace_canvas.LegendItem) else ""
            self.inspector_title.setText(f"Plot {card.plot_id.split('_')[-1]}{suffix}")
            top = item.parentItem() or item
            self.scene.current_page = top.layout_mm['page']
            return
        self._show_page_panel(f"{len(selected)} items selected" if selected else None)

    def _on_item_action(self, action: str, item):
        card = getattr(item, 'card', None)
        if action == 'duplicate':
            layout = dict(item.layout_mm)
            layout['x_mm'] += drc_workspace_canvas.DUPLICATE_OFFSET_MM
            layout['y_mm'] += drc_workspace_canvas.DUPLICATE_OFFSET_MM
            if isinstance(item, drc_workspace_canvas.TextItem):
                spec = dict(item.box.spec, group=None)
                new = self.add_text(spec['kind'], spec=spec, layout=layout)
            elif isinstance(item, drc_workspace_canvas.SourceFigureItem):
                spec = dict(item.figure_source.spec, group=None)
                new = self.add_source_figure(spec, layout=layout)
            else:
                config = dict(card.get_display_config(), group=None)
                new = self._items[self.add_plot(config=config, layout=layout).plot_id]
            self.scene.clearSelection()
            new.setSelected(True)
        elif action == 'copy_style':
            card._copy_display_config()
        elif action == 'paste_style':
            card._paste_display_config()
        elif action == 'export_png':
            self.export_item_png(item.parentItem() or item)
        elif action == 'update_source':
            self.update_source_figure(item)
        elif action == 'original_size':
            self._resize_source_undoable(item, item.figure_source.natural_mm)
        elif action == 'letter':
            self.scene.clearSelection()
            item.setSelected(True)
            self.add_text('letter')
        elif action == 'edit_text':
            self.scene.clearSelection()
            item.setSelected(True)
            self.text_edit.setFocus()
            self.text_edit.selectAll()
        elif action == 'dock_legend':
            card.legend_offset_mm = None
            self._items[card.plot_id].on_legend_changed()
            card._sync_config_to_state()
        elif action == 'hide_legend':
            card._legend_pos_combo.setCurrentText('Off')
        elif action == 'group':
            self.group_selected()
        elif action == 'ungroup':
            self.ungroup_selected()
        elif action in ('front', 'back'):
            self.restack_selected(action == 'front')
        elif action == 'delete':
            self.remove_items(self._selected_top_items() or [item])

    def _on_theme_changed(self, text: str):
        """Walk up to PluginWidget so a Workspace-toggle change immediately
        redraws every currently-visible plot in the other tabs too, rather
        than only taking effect the next time each one happens to redraw
        on its own."""
        self.state.plot_theme = text.lower()
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, 'refresh_plot_theme'):
                parent.refresh_plot_theme()
                return
            parent = parent.parent()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_pages(self):
        """Ask for format, pages, resolution and file, then export."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout
        if not self.scene.top_items():
            QMessageBox.information(self, "Export", "There is nothing on the pages to export.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Export pages")
        form = QFormLayout(dlg)
        fmt_combo = QComboBox()
        fmt_combo.addItems(list(drc_workspace_export.FORMATS))
        form.addRow("Format:", fmt_combo)
        pages_combo = QComboBox()
        pages_combo.addItems([f"All pages ({len(self.scene.pages)})",
                              f"Current page ({self.scene.current_page + 1})"])
        form.addRow("Pages:", pages_combo)
        dpi_combo = QComboBox()
        for dpi in EXPORT_DPI_CHOICES:
            dpi_combo.addItem(f"{dpi} dpi", dpi)
        dpi_combo.setToolTip("PNG/TIFF resolution; also used for the point layers of "
                             "scatter plots in PDF and SVG")
        form.addRow("Resolution:", dpi_combo)
        editable = QCheckBox("Editable text (SVG)")
        editable.setChecked(True)
        editable.setToolTip("Keep text as text in SVG files (fonts must be installed where "
                            "the file is opened); otherwise text is drawn as outlines")
        form.addRow(editable)
        fmt_combo.currentTextChanged.connect(lambda f: editable.setEnabled(f == 'SVG'))
        editable.setEnabled(False)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec() != QDialog.Accepted:
            return
        fmt = fmt_combo.currentText()
        pages = (list(range(len(self.scene.pages))) if pages_combo.currentIndex() == 0
                 else [self.scene.current_page])
        ext = {'PDF': '.pdf', 'SVG': '.svg', 'PNG': '.png', 'TIFF': '.tif'}[fmt]
        path, _ = QFileDialog.getSaveFileName(
            self, f"Export {fmt}", f"Workspace{ext}", f"{fmt} files (*{ext})")
        if not path:
            return
        self.export_pages(path, fmt, pages, int(dpi_combo.currentData()),
                          editable.isChecked())

    def export_pages(self, path: str, fmt: str, pages: list[int], dpi: int,
                     editable_text: bool = True) -> list:
        """Export *pages* as *fmt*; returns the files written."""
        from pathlib import Path as _Path
        target = _Path(path)
        try:
            if fmt == 'PDF':
                written = [drc_workspace_export.export_pdf(
                    target.with_suffix('.pdf'), self.scene, pages, dpi)]
            elif fmt == 'SVG':
                written = drc_workspace_export.export_svg(
                    target, self.scene, pages, editable_text, dpi)
            else:
                written = drc_workspace_export.export_raster(target, self.scene, pages, fmt, dpi)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.warning(self, "Export Error", str(e))
            return []
        names = "\n".join(p.name for p in written)
        QMessageBox.information(self, "Exported", f"Saved to {target.parent}:\n{names}")
        return written

    def export_plot_png(self, card: 'PlotCard'):
        """Export one plot, with its separate legend, as a 300 dpi PNG."""
        item = self._items.get(card.plot_id)
        if item is not None:
            self.export_item_png(item)

    def export_item_png(self, item):
        """Export one item (a plot with its separate legend) as a 300 dpi PNG."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PNG", f"{item.uid}.png", "PNG files (*.png)")
        if not path:
            return
        path = _with_extension(path, '.png')
        legend = getattr(item, 'legend_item', None)
        parts = [item] + ([legend] if legend is not None else [])
        entries, w, h = drc_workspace_export.entries_for(parts)
        try:
            array = drc_workspace_export.render_entries(
                entries, w, h, drc_workspace_export.DEFAULT_EXPORT_DPI)
            drc_workspace_export.save_raster(array, Path(path), 'PNG',
                                             drc_workspace_export.DEFAULT_EXPORT_DPI)
            QMessageBox.information(self, "Exported", f"Saved to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    def refresh(self):
        """Re-render all plots from current PipelineState; figures from
        other tabs are fetched again once those tabs have redrawn."""
        for card in self._plot_cards:
            card.refresh()
        if self._figures:
            QTimer.singleShot(0, self.update_all_source_figures)

    # ------------------------------------------------------------------
    # Figures from other tabs
    # ------------------------------------------------------------------

    def set_figure_sources(self, sources: dict):
        """{tab name: zero-arg callable returning that tab's ReportItems};
        the items with a figure can be added to the pages."""
        self._figure_sources = dict(sources)

    def _source_items(self, tab: str) -> list:
        getter = self._figure_sources.get(tab)
        if getter is None:
            return []
        try:
            return [it for it in getter() if it.get_figure is not None]
        except Exception:
            traceback.print_exc()
            return []

    def _fill_figure_menu(self, menu, point):
        """One submenu per source tab listing its figures; *point* (page,
        x, y) is where they go (None: last click / free spot)."""
        menu.clear()
        for tab in self._figure_sources:
            sub = menu.addMenu(tab)
            items = self._source_items(tab)
            if not items:
                action = sub.addAction(f"(nothing to add yet — show it in the {tab} tab)")
                action.setEnabled(False)
            for report_item in items:
                sub.addAction(report_item.label,
                              lambda tab=tab, it=report_item: self._add_figure_at(tab, it, point))

    def _add_figure_at(self, tab: str, report_item, point):
        if point is not None:
            self.scene.last_click = point
        self.add_source_figure({'tab': tab, 'key': report_item.key, 'label': report_item.label})

    def _fetch_source_figure(self, source):
        """Find the source's ReportItem by (tab, key) and copy its figure in;
        a placeholder when it is not available."""
        spec = source.spec
        match = next((it for it in self._source_items(spec['tab']) if it.key == spec['key']), None)
        fig = None
        if match is not None:
            try:
                fig = match.get_figure()
            except Exception:
                traceback.print_exc()
        if fig is None:
            source.set_unavailable()
            return
        copy, as_image = drc_workspace_canvas.detached_copy(fig)
        page = self.scene.pages[self.scene.current_page]
        source.set_figure(copy, as_image,
                          page['w_mm'] * drc_workspace_canvas.SOURCE_FIGURE_MAX_FRACTION)

    def add_source_figure(self, spec: dict, layout: dict | None = None,
                          uid: str | None = None, record: bool = True):
        """Add a figure from another tab (spec: 'tab', 'key', 'label', and
        optionally 'w_mm', 'h_mm', 'aspect_locked', 'group', 'z'). A new one
        goes at the last point clicked on a page, else in a free spot."""
        source = drc_workspace_canvas.SourceFigure(
            uid or self._next_id('figure', set(self._figures)),
            {k: spec.get(k) for k in ('tab', 'key', 'label', 'w_mm', 'h_mm', 'aspect_locked')
             if spec.get(k) is not None})
        self._fetch_source_figure(source)
        item = drc_workspace_canvas.SourceFigureItem(source)
        item.action_requested.connect(self._on_item_action)
        if layout is None:
            layout = self._layout_at_last_click(item.size_mm())
        item.layout_mm = {'page': int(layout['page']), 'x_mm': float(layout['x_mm']),
                          'y_mm': float(layout['y_mm'])}
        item.group_id = spec.get('group')
        item.setZValue(float(spec.get('z') or 0.0))
        self.scene.addItem(item)
        self.scene.position_item(item)
        self._figures[source.uid] = item
        source.changed.connect(self._on_layout_changed)
        if record and not self._restoring:
            self._record_added([source.uid], f"Add {spec.get('label', 'figure')}")
            self.scene.clearSelection()
            item.setSelected(True)
        self._on_layout_changed()
        return item

    def update_source_figure(self, item):
        """Fetch the item's figure again from its tab (size kept)."""
        self._fetch_source_figure(item.figure_source)
        if self._figure_panel_item is item:
            self._show_figure_panel(item)

    def update_all_source_figures(self):
        for item in list(self._figures.values()):
            self._fetch_source_figure(item.figure_source)

    def _resize_source_undoable(self, item, size):
        if not size:
            return
        before = item.size_state()
        item.apply_size_state(size)
        self.scene.record_resize(item, before, item.size_state())
        if self._figure_panel_item is item:
            self._show_figure_panel(item)

    def _build_figure_panel(self) -> QWidget:
        from PySide6.QtWidgets import QFormLayout
        panel = QWidget()
        form = QFormLayout(panel)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.figure_source_label = QLabel()
        self.figure_source_label.setWordWrap(True)
        form.addRow("From:", self.figure_source_label)
        self.figure_w_spin = QDoubleSpinBox()
        self.figure_h_spin = QDoubleSpinBox()
        for spin in (self.figure_w_spin, self.figure_h_spin):
            spin.setRange(PLOT_MIN_MM, PAGE_MAX_MM)
            spin.setDecimals(1)
            spin.setSuffix(" mm")
        self.figure_w_spin.valueChanged.connect(lambda v: self._on_figure_size_spin('w', v))
        self.figure_h_spin.valueChanged.connect(lambda v: self._on_figure_size_spin('h', v))
        form.addRow("Width:", self.figure_w_spin)
        form.addRow("Height:", self.figure_h_spin)
        self.figure_aspect_check = QCheckBox("Lock aspect ratio")
        self.figure_aspect_check.toggled.connect(self._on_figure_aspect_toggled)
        form.addRow(self.figure_aspect_check)
        buttons = QHBoxLayout()
        update = QPushButton("Update from source")
        update.setToolTip("Copy the figure again as it is shown in its tab now")
        update.clicked.connect(lambda: self._figure_panel_item and self.update_source_figure(
            self._figure_panel_item))
        buttons.addWidget(update)
        original = QPushButton("Original size")
        original.clicked.connect(lambda: self._figure_panel_item and self._resize_source_undoable(
            self._figure_panel_item, self._figure_panel_item.figure_source.natural_mm))
        buttons.addWidget(original)
        form.addRow(buttons)
        self.figure_status = QLabel()
        self.figure_status.setWordWrap(True)
        self.figure_status.setStyleSheet(_hint_style())
        form.addRow(self.figure_status)
        self._figure_panel_item = None
        self._figure_panel_loading = False
        return panel

    def _show_figure_panel(self, item):
        source = item.figure_source
        spec = source.spec
        self._figure_panel_item = None
        self._figure_panel_loading = True
        try:
            self.figure_source_label.setText(f"{spec.get('tab', '')} — {spec.get('label', '')}")
            self.figure_w_spin.setValue(spec.get('w_mm') or 0.0)
            self.figure_h_spin.setValue(spec.get('h_mm') or 0.0)
            self.figure_aspect_check.setChecked(bool(spec.get('aspect_locked', True)))
            if not source.available:
                status = (f"Not available: show it in the {spec.get('tab', '')} tab, then "
                          "Update from source.")
            elif source.as_image:
                status = ("Copied as an image (the figure could not be copied as a figure), "
                          "so it scales like a picture.")
            else:
                status = ("A copy of the figure as it was when added or last updated; fonts "
                          "keep their size when it is resized.")
            self.figure_status.setText(status)
        finally:
            self._figure_panel_loading = False
        self._figure_panel_item = item
        self.inspector_stack.setCurrentIndex(3)
        self.inspector_title.setText("Figure")

    def _on_figure_size_spin(self, which: str, value: float):
        item = self._figure_panel_item
        if self._figure_panel_loading or item is None:
            return
        w, h = item.size_state()
        if not (w and h):
            return
        if which == 'w':
            new = (value, value * h / w if self.figure_aspect_check.isChecked() else h)
        else:
            new = (value * w / h if self.figure_aspect_check.isChecked() else w, value)
        self._resize_source_undoable(item, new)

    def _on_figure_aspect_toggled(self, locked: bool):
        item = self._figure_panel_item
        if self._figure_panel_loading or item is None:
            return
        item.figure_source.spec['aspect_locked'] = bool(locked)
        self._on_layout_changed()

    def _show_page_menu(self, page: int, x_mm: float, y_mm: float, screen_pos):
        """Right-click on an empty part of a page: add things there."""
        from PySide6.QtWidgets import QMenu
        point = (page, x_mm, y_mm)
        menu = QMenu(self)
        menu.addAction("Add text here", lambda: self._add_text_at('text', point))
        menu.addAction("Add panel letter here", lambda: self._add_text_at('letter', point))
        menu.addAction("Add plot here", lambda: self.add_plot(
            layout={'page': page, 'x_mm': x_mm, 'y_mm': y_mm}))
        figures = menu.addMenu("Add from other tabs")
        self._fill_figure_menu(figures, point)
        menu.exec(screen_pos)

    def _add_text_at(self, kind: str, point):
        self.scene.clearSelection()
        self.scene.last_click = point
        self.add_text(kind)

    def get_report_items(self) -> list:
        """One drc_report.ReportItem per page with something on it: the page
        as a figure (rendered at 300 dpi) for the combined PDF and PNG, plus
        a vector PDF of the page written beside them."""
        items = []
        for page in range(len(self.scene.pages)):
            if not drc_workspace_export.page_entries(self.scene, page):
                continue
            items.append(drc_report.ReportItem(
                key=f'page_{page + 1}',
                tab='Workspace',
                label=f"Page {page + 1}",
                get_figure=lambda p=page: drc_workspace_export.page_figure(self.scene, p),
                write_files=lambda folder, p=page: drc_workspace_export.export_pdf(
                    Path(folder) / f"Page {p + 1} (vector).pdf", self.scene, [p]),
            ))
        return items


class _MarkerValuesWorker(QThread):
    """
    Loads per-sample marker values (disk I/O + AF-aware unmixing, via
    drc_pipeline.load_sample_marker_values) for a PlotCard's Marker colour
    mode on a background thread, so a cold redraw never blocks the UI.
    Same pattern as _MarkerSummaryWorker/_StatsWorker/_DrWorker elsewhere
    in this tab. Loaded arrays flow through the SAME state.gated_data_cache
    every other caller of load_unmixed_gated shares, so this only actually
    touches disk on a genuine cache miss (first Marker-mode draw for a
    given sample set / gate selection / unmixing matrix) -- PlotCard's own
    cache (see _marker_cache_key) skips even starting this worker once
    that's warm.
    """
    finished = Signal(bool, str, dict)  # success, error, {rel_path: (values, names, rows)}

    def __init__(self, controller, state, samples: list[str], af_state, parent=None):
        super().__init__(parent)
        self._controller = controller
        self._state = state
        self._samples = samples
        self._af_state = af_state

    def run(self):
        try:
            results: dict[str, tuple] = {}
            for rel in self._samples:
                mv = drc_pipeline.load_sample_marker_values(
                    self._controller, self._state, rel, af_state=self._af_state)
                rows = drc_pipeline.load_sample_rows(
                    self._controller, self._state, rel, af_state=self._af_state)
                if mv is not None and rows is not None:
                    results[rel] = (mv[0], mv[1], rows)
            self.finished.emit(True, '', results)
        except Exception as exc:
            traceback.print_exc()
            self.finished.emit(False, str(exc), {})


class _ClusterTreeLayoutWorker(QThread):
    """
    Computes the MST + Kamada-Kawai layout for a Cluster Tree view on a
    background thread. HDBSCAN in particular can produce far more nodes
    (final clusters) than FlowSOM's usual handful of metaclusters, and
    the Kamada-Kawai layout's cost grows fast with node count --
    easily long enough to freeze the UI if called straight from
    refresh() on the main thread. Pure computation only (numpy/scipy/
    networkx, via drc_clustering.build_flowsom_tree) -- no Qt widget
    access, so unlike most workers in this file it needs no AF/
    transfer-matrix snapshot.
    """
    finished = Signal(bool, str, dict)  # success, error, layout dict

    def __init__(self, node_weights, node_to_meta, node_counts, parent=None):
        super().__init__(parent)
        self._node_weights = node_weights
        self._node_to_meta = node_to_meta
        self._node_counts = node_counts

    def run(self):
        try:
            layout = drc_clustering.build_flowsom_tree(
                self._node_weights, self._node_to_meta, self._node_counts)
            self.finished.emit(True, '', layout)
        except Exception as exc:
            traceback.print_exc()
            self.finished.emit(False, str(exc), {})


class PlotCard(QFrame):
    """
    One plot of the Workspace: its settings (shown in the Workspace
    inspector) and its matplotlib figure, drawn offscreen on an Agg canvas
    and shown on the page by a drc_workspace_canvas.PlotItem, which listens
    to figure_changed.

    Settings: plot type, DR run, sample (all pooled or one), colour mode
    (Clusters | Marker | Group), marker channel (Marker mode), clustering
    run (Clusters mode and Cluster Tree), magic-wand copy/paste, PNG export,
    remove, and appearance (plot size in mm, legend, fonts, grid, labels).

    The cluster/group legend is drawn in the figure: right-click an entry
    → colour picker, double-click a cluster name → rename.

    Parameters
    ----------
    plot_id    Unique string identifier stored in state.plot_configs.
    state      Shared PipelineState.
    bus        Honeychrome EventBus.
    controller Main Honeychrome Controller.
    workspace  Parent WorkspaceTab (for remove_card / magic-wand).
    """

    _COLOUR_MODES = ['Clusters', 'Marker', 'Group']

    figure_changed = Signal()   # the figure was redrawn or resized
    legend_changed = Signal()   # the separate legend was redrawn, shown or hidden

    def __init__(self, plot_id: str, state: PipelineState, bus, controller,
                 workspace: 'WorkspaceTab', parent=None):
        super().__init__(parent)
        self.plot_id    = plot_id
        self.state      = state
        self.bus        = bus
        self.controller = controller
        self.workspace  = workspace
        self._figure    = None   # current matplotlib Figure
        self._marker_worker = None            # in-flight _MarkerValuesWorker, if any
        self._marker_values_cache: dict = {}  # single-entry -- see _on_marker_values_finished
        self._tree_layout_worker = None       # in-flight _ClusterTreeLayoutWorker, if any
        self._tree_layout_cache: dict = {}    # {run_id: layout} -- a run's tree_data is
                                               # frozen at archive time, so its MST/layout
                                               # never changes; cached permanently per run_id.

        self.layout_mm: dict | None = None   # {'page', 'x_mm', 'y_mm'}, set by WorkspaceTab
        self.group_id: str | None = None     # Workspace group, set by WorkspaceTab
        self.z_value: float = 0.0            # stacking order on the page
        self.setFrameShape(QFrame.NoFrame)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from PySide6.QtWidgets import QFormLayout

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ---- What is plotted ----
        self._form = QFormLayout()
        self._form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(['Scatter', 'Cluster Tree'])
        self.plot_type_combo.setToolTip(
            "Scatter: per-event DR embedding.\n"
            "Cluster Tree: minimum-spanning-tree view of a clustering run's "
            "nodes (pick it under Clustering run) -- SOM nodes for "
            "FlowSOM, per-cluster centroids for Leiden/HDBSCAN."
        )
        self.plot_type_combo.currentTextChanged.connect(self._on_plot_type_changed)
        _style_combo_popup(self.plot_type_combo)
        self._form.addRow("Plot type:", self.plot_type_combo)

        self.dr_combo = QComboBox()
        self.dr_combo.setToolTip("Archived DR run to display")
        self.dr_combo.currentIndexChanged.connect(self._on_dr_run_changed)
        _style_combo_popup(self.dr_combo)
        self._form.addRow("DR run:", self.dr_combo)

        self.sample_combo = QComboBox()
        self.sample_combo.setToolTip("Sample to display (or all pooled)")
        self.sample_combo.currentTextChanged.connect(self._schedule_refresh)
        _style_combo_popup(self.sample_combo)
        self._form.addRow("Sample:", self.sample_combo)

        self.colour_mode_combo = QComboBox()
        self.colour_mode_combo.addItems(self._COLOUR_MODES)
        self.colour_mode_combo.currentTextChanged.connect(self._on_colour_mode_changed)
        _style_combo_popup(self.colour_mode_combo)
        self._form.addRow("Colour:", self.colour_mode_combo)

        self.marker_combo = QComboBox()
        self.marker_combo.setToolTip("Channel to colour by (Marker mode)")
        self.marker_combo.currentTextChanged.connect(self._schedule_refresh)
        _style_combo_popup(self.marker_combo)
        self._form.addRow("Marker:", self.marker_combo)

        # Clustering run shown in Clusters colour mode and in Cluster Tree
        self.cluster_run_combo = QComboBox()
        self.cluster_run_combo.setToolTip("Archived clustering run whose clusters colour the plot")
        self.cluster_run_combo.currentIndexChanged.connect(self._on_cluster_run_changed)
        _style_combo_popup(self.cluster_run_combo)
        # ⚠ when the DR run's and clustering run's gate/sample sets don't line up
        self._compat_warning = QLabel("⚠")
        self._compat_warning.setStyleSheet("color: #d9822b; font-weight: bold;")
        self._compat_warning.setVisible(False)
        cluster_row = QHBoxLayout()
        cluster_row.setContentsMargins(0, 0, 0, 0)
        cluster_row.addWidget(self.cluster_run_combo, stretch=1)
        cluster_row.addWidget(self._compat_warning)
        self._cluster_row_widget = QWidget()
        self._cluster_row_widget.setLayout(cluster_row)
        self._form.addRow("Clustering run:", self._cluster_row_widget)

        layout.addLayout(self._form)

        # ---- Actions ----
        actions = QHBoxLayout()
        actions.setSpacing(4)
        self.wand_btn = QPushButton("🪄 Copy style")
        self.wand_btn.setToolTip("Copy this plot's display settings (magic wand)")
        self.wand_btn.clicked.connect(self._copy_display_config)
        actions.addWidget(self.wand_btn)
        self.paste_btn = QPushButton("📋 Paste style")
        self.paste_btn.setToolTip("Apply display settings copied from another plot")
        self.paste_btn.clicked.connect(self._paste_display_config)
        actions.addWidget(self.paste_btn)
        self.png_btn = QPushButton("PNG")
        self.png_btn.setToolTip("Export this plot as PNG")
        self.png_btn.clicked.connect(self._export_png)
        actions.addWidget(self.png_btn)
        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedWidth(30)
        self.close_btn.setToolTip("Remove this plot")
        self.close_btn.clicked.connect(lambda: self.workspace.remove_card(self))
        actions.addWidget(self.close_btn)
        layout.addLayout(actions)

        # ---- Appearance ----
        appearance = QGroupBox("Appearance")
        app_form = QFormLayout(appearance)
        app_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        # Plot size in mm; the legend adds to the figure's width (Right) or
        # height (Below).
        self._plot_w_spin = QDoubleSpinBox()
        self._plot_w_spin.setRange(PLOT_MIN_MM, PLOT_MAX_MM)
        self._plot_w_spin.setDecimals(1)
        self._plot_w_spin.setSingleStep(1.0)
        self._plot_w_spin.setSuffix(" mm")
        self._plot_w_spin.setValue(PLOT_DEFAULT_MM)
        self._plot_w_spin.setToolTip("Plot width, not counting a legend at the right")
        self._plot_w_spin.valueChanged.connect(self._on_plot_w_changed)
        app_form.addRow("Plot width:", self._plot_w_spin)

        self._plot_h_spin = QDoubleSpinBox()
        self._plot_h_spin.setRange(PLOT_MIN_MM, PLOT_MAX_MM)
        self._plot_h_spin.setDecimals(1)
        self._plot_h_spin.setSingleStep(1.0)
        self._plot_h_spin.setSuffix(" mm")
        self._plot_h_spin.setValue(PLOT_DEFAULT_MM)
        self._plot_h_spin.setToolTip("Plot height, not counting a legend below")
        self._plot_h_spin.valueChanged.connect(self._on_plot_h_changed)
        app_form.addRow("Plot height:", self._plot_h_spin)

        self._aspect_check = QCheckBox("Lock aspect ratio")
        self._aspect_check.setChecked(True)
        self._aspect_check.setToolTip(
            "Keep the plot's width:height ratio when it is resized here or on the page")
        self._aspect_check.toggled.connect(self._on_aspect_toggled)
        app_form.addRow(self._aspect_check)
        self._aspect_ratio = 1.0          # width / height kept while locked
        self._applying_size = False

        self._legend_pos_combo = QComboBox()
        self._legend_pos_combo.addItems(LEGEND_POSITIONS)
        self._legend_pos_combo.setToolTip(
            "Where the cluster/group legend is drawn.\n"
            "Separate: its own box on the page, beside the plot until you move it; "
            "drag its edges to change its columns.\n"
            "Right / Below: part of the plot figure, added to its width / height.\n"
            "Right-click an entry to change its colour or rename it; double-click a "
            "cluster name to rename it.")
        self._legend_pos_combo.currentTextChanged.connect(self._schedule_refresh)
        _style_combo_popup(self._legend_pos_combo)
        app_form.addRow("Legend:", self._legend_pos_combo)

        self._legend_cols_spin = QSpinBox()
        self._legend_cols_spin.setRange(0, 8)
        self._legend_cols_spin.setValue(0)
        self._legend_cols_spin.setSpecialValueText("Auto")
        self._legend_cols_spin.setToolTip(
            "Legend columns. Auto fits the entries to the plot's height (Separate, "
            "Right) or width (Below).")
        self._legend_cols_spin.valueChanged.connect(self._on_legend_cols_changed)
        app_form.addRow("Legend columns:", self._legend_cols_spin)

        self._legend_font_spin = QSpinBox()
        self._legend_font_spin.setRange(6, 24)
        self._legend_font_spin.setValue(9)
        self._legend_font_spin.setSuffix(" pt")
        self._legend_font_spin.setToolTip("Font size of the legend and colourbar labels")
        self._legend_font_spin.valueChanged.connect(self._schedule_refresh)
        app_form.addRow("Legend font:", self._legend_font_spin)

        self._axis_font_spin = QSpinBox()
        self._axis_font_spin.setRange(5, 20)
        self._axis_font_spin.setValue(7)
        self._axis_font_spin.setSuffix(" pt")
        self._axis_font_spin.valueChanged.connect(self._schedule_refresh)
        app_form.addRow("Axis font:", self._axis_font_spin)

        checks = QHBoxLayout()
        self._show_grid = QCheckBox("Grid")
        self._show_grid.setChecked(False)
        self._show_grid.stateChanged.connect(self._schedule_refresh)
        checks.addWidget(self._show_grid)
        self._show_ticks = QCheckBox("Tick labels")
        self._show_ticks.setChecked(True)
        self._show_ticks.stateChanged.connect(self._schedule_refresh)
        checks.addWidget(self._show_ticks)
        self._show_axis_labels = QCheckBox("Axis labels")
        self._show_axis_labels.setChecked(True)
        self._show_axis_labels.stateChanged.connect(self._schedule_refresh)
        checks.addWidget(self._show_axis_labels)
        checks.addStretch()
        app_form.addRow(checks)

        self._label_color: str | None = None   # None = Auto (follows the plot theme)
        self._label_color_btn = QPushButton()
        self._label_color_btn.setToolTip(
            "Colour of the axis titles, plot title, colourbar and legend text")
        self._label_color_btn.clicked.connect(self._pick_label_color)
        self._label_auto_btn = QPushButton("Auto")
        self._label_auto_btn.setToolTip(
            "Follow the plot theme: black on Light, white on Dark")
        self._label_auto_btn.clicked.connect(self._reset_label_color)
        label_row = QHBoxLayout()
        label_row.setContentsMargins(0, 0, 0, 0)
        label_row.addWidget(self._label_color_btn, stretch=1)
        label_row.addWidget(self._label_auto_btn)
        label_row_widget = QWidget()
        label_row_widget.setLayout(label_row)
        app_form.addRow("Label colour:", label_row_widget)
        self._update_label_color_btn()

        layout.addWidget(appearance)
        layout.addStretch()

        # ---- Figure (drawn offscreen; the Workspace page shows it) ----
        self._plot_w_in = PLOT_DEFAULT_MM / MM_PER_INCH
        self._plot_h_in = PLOT_DEFAULT_MM / MM_PER_INCH
        self._figure = Figure(figsize=(self._plot_w_in, self._plot_h_in), layout='compressed')
        FigureCanvasAgg(self._figure)
        self._legend_extra_in = (0.0, 0.0)   # legend's addition to (width, height)
        self._legend_pick_targets: dict = {}  # id(artist) -> ('cluster'|'group', key)
        self.legend_pick_handled = False
        self.legend_probe = False        # True: a pick only reports the entry (legend_probe_hit)
        self.legend_probe_hit = None     # (kind, key, name)
        self._figure.canvas.mpl_connect('pick_event', self._on_legend_pick)
        # Separate legend: its own figure, shown by the page's LegendItem
        self._legend_figure = Figure(figsize=(1.0, 1.0))
        FigureCanvasAgg(self._legend_figure)
        self._legend_figure.canvas.mpl_connect('pick_event', self._on_legend_pick)
        self.legend_shown = False
        self.legend_offset_mm: tuple[float, float] | None = None   # None = beside the plot

        # Populate selectors
        self._populate_selectors()
        self._on_plot_type_changed(self.plot_type_combo.currentText())

    def _populate_selectors(self):
        """
        Fill DR-run, clustering-run-overlay, and marker-channel combos from
        current state.  Called at card construction, and again whenever
        the run archive changes elsewhere (PluginWidget._on_runs_changed)
        so a rename/delete in ConfigTab's management table is reflected
        here without the user needing to touch this card.
        """
        self._populate_dr_run_combo()
        self._populate_cluster_run_combo()
        self._populate_marker_combo()

        self._update_compatibility_warning()
        self._schedule_refresh()

    def _populate_marker_combo(self):
        """
        Rebuild the marker-overlay combo from the CURRENTLY SELECTED DR
        run's own archived channel list, not the live session's
        state.selected_channels -- a run trained under an earlier channel
        selection must still offer every marker it was actually trained
        on, and must not silently offer ones it wasn't. Falls back to
        state.selected_channels only when no run is selected yet (a fresh,
        empty card) or an old run has nothing archived under 'channels'.
        """
        run = self._selected_dr_run()
        channels = (run.get('channels') if run else None) or self.state.selected_channels

        self.marker_combo.blockSignals(True)
        current_ch = self.marker_combo.currentData()
        self.marker_combo.clear()
        labels = _antigen_dash_labels(self.controller)
        for ch in channels:
            self.marker_combo.addItem(labels.get(ch, ch), ch)
        if current_ch:
            idx = self.marker_combo.findData(current_ch)
            if idx >= 0:
                self.marker_combo.setCurrentIndex(idx)
        self.marker_combo.blockSignals(False)

    def _populate_dr_run_combo(self):
        """Rebuild the DR-run combo from state.dr_runs, keyed by run_id —
        never label text, since labels are user-editable."""
        self.dr_combo.blockSignals(True)
        prev_run_id = self.dr_combo.currentData()
        self.dr_combo.clear()
        runs = sorted(self.state.dr_runs, key=lambda e: e.get('timestamp', ''))
        for entry in runs:
            self.dr_combo.addItem(entry.get('label', ''), entry.get('run_id'))
        if self.dr_combo.count() == 0:
            self.dr_combo.addItem('—', None)
        idx = self.dr_combo.findData(prev_run_id) if prev_run_id else -1
        if idx >= 0:
            self.dr_combo.setCurrentIndex(idx)
        else:
            self.dr_combo.setCurrentIndex(self.dr_combo.count() - 1)
        self.dr_combo.blockSignals(False)
        self._refresh_sample_combo()

    def _populate_cluster_run_combo(self):
        """Rebuild the clustering-run overlay combo from
        state.clustering_runs, keyed by run_id. Used for both Scatter-mode
        overlay and Cluster Tree mode -- Tree mode now works for any
        clustering algorithm (FlowSOM/Leiden/HDBSCAN all populate
        tree_data), so no algorithm filtering is needed here; a run
        archived before this feature existed just falls back to the
        "no tree data" placeholder in _draw_cluster_tree_view()."""
        self.cluster_run_combo.blockSignals(True)
        prev_run_id = self.cluster_run_combo.currentData()
        self.cluster_run_combo.clear()
        runs = sorted(self.state.clustering_runs, key=lambda e: e.get('timestamp', ''))
        for entry in runs:
            self.cluster_run_combo.addItem(entry.get('label', ''), entry.get('run_id'))
        if self.cluster_run_combo.count() == 0:
            self.cluster_run_combo.addItem('(none)', None)
        idx = self.cluster_run_combo.findData(prev_run_id) if prev_run_id else -1
        if idx >= 0:
            self.cluster_run_combo.setCurrentIndex(idx)
        else:
            self.cluster_run_combo.setCurrentIndex(self.cluster_run_combo.count() - 1)
        self.cluster_run_combo.blockSignals(False)

    def _selected_dr_run(self) -> dict | None:
        """Return the (hydrated) DR run entry for the current dr_combo
        selection, or None.  hydrate_run() is a no-op if this run was
        already hydrated earlier this session (by this card, another
        card, or GroupsStatsTab)."""
        run_id = self.dr_combo.currentData()
        if run_id is None:
            return None
        for entry in self.state.dr_runs:
            if entry.get('run_id') == run_id:
                return drc_run_archive.hydrate_run(self.controller, entry)
        return None

    def _selected_cluster_run(self) -> dict | None:
        """Return the (hydrated) clustering run entry for the current
        overlay selection, or None."""
        run_id = self.cluster_run_combo.currentData()
        if run_id is None:
            return None
        for entry in self.state.clustering_runs:
            if entry.get('run_id') == run_id:
                return drc_run_archive.hydrate_run(self.controller, entry)
        return None

    def _refresh_sample_combo(self):
        """Rebuild the sample combo from the currently-selected DR run's
        embeddings."""
        run = self._selected_dr_run()
        emb_dict = run.get('embeddings', {}) if run else {}

        self.sample_combo.blockSignals(True)
        current_sample = self.sample_combo.currentData()
        self.sample_combo.clear()
        self.sample_combo.addItem('All Samples')
        for rel in sorted(emb_dict.keys()):
            self.sample_combo.addItem(Path(rel).name, userData=rel)
        if current_sample:
            for i in range(self.sample_combo.count()):
                if self.sample_combo.itemData(i) == current_sample:
                    self.sample_combo.setCurrentIndex(i)
                    break
        self.sample_combo.blockSignals(False)

    def _on_dr_run_changed(self, _index: int):
        self._refresh_sample_combo()
        self._populate_marker_combo()
        self._update_compatibility_warning()
        self._schedule_refresh()

    def _on_cluster_run_changed(self, _index: int):
        self._update_compatibility_warning()
        self._schedule_refresh()

    def _update_compatibility_warning(self):
        """
        Show the ⚠ indicator when the selected clustering run's gate-set
        or sample-set isn't a subset of the selected DR run's. Delegates
        the set comparison to drc_scatter (shared with the
        Cluster Annotation tab's cluster-map panel).
        """
        if self.colour_mode_combo.currentText() != 'Clusters':
            self._compat_warning.setVisible(False)
            self._compat_warning.setToolTip('')
            return
        dr_run = self._selected_dr_run()
        cl_run = self._selected_cluster_run()
        warning = drc_scatter.compatibility_warning(dr_run, cl_run)
        self._compat_warning.setVisible(warning is not None)
        self._compat_warning.setToolTip(warning or '')

    def _on_colour_mode_changed(self, mode: str):
        """Show the Marker row in Marker mode and the Clustering run row in
        Clusters mode (and always in Cluster Tree)."""
        is_tree = self.plot_type_combo.currentText() == 'Cluster Tree'
        self._form.setRowVisible(self.marker_combo, mode == 'Marker' and not is_tree)
        self._form.setRowVisible(self._cluster_row_widget, mode == 'Clusters' or is_tree)
        self._update_compatibility_warning()
        self._schedule_refresh()

    def _on_plot_type_changed(self, mode: str):
        """Toggle between the per-event Scatter view and the Cluster Tree
        view. Tree mode has no DR run / sample / marker concept — it only
        needs a clustering run (any algorithm)."""
        is_tree = (mode == 'Cluster Tree')
        for widget in (self.dr_combo, self.sample_combo, self.colour_mode_combo):
            self._form.setRowVisible(widget, not is_tree)
        self._populate_cluster_run_combo()
        self._on_colour_mode_changed(self.colour_mode_combo.currentText())

    def _schedule_refresh(self, *_):
        self._sync_config_to_state()
        # Debounce: apply_config() can trigger this from several combo
        # signals in the same tick (dr_combo, colour_mode_combo,
        # marker_combo, plus its own trailing call). Without this guard,
        # each one queues an independent QTimer.singleShot(0, self.refresh),
        # and QTimer does not de-duplicate — the expensive refresh() (FCS
        # reload + redraw) would run once per combo touched.
        if getattr(self, '_refresh_pending', False):
            return
        self._refresh_pending = True
        QTimer.singleShot(0, self._run_scheduled_refresh)

    def _run_scheduled_refresh(self):
        self._refresh_pending = False
        self.refresh()

    def _sync_config_to_state(self):
        """Write current display config back to state.plot_configs entry."""
        cfg = self.get_display_config()
        cfg['plot_id'] = self.plot_id
        for i, pc in enumerate(self.state.plot_configs):
            if pc.get('plot_id') == self.plot_id:
                self.state.plot_configs[i] = cfg
                break

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def refresh(self):
        """Re-render the scatter plot from the currently-selected DR run
        (and, in Clusters mode, the currently-selected clustering run).
        In 'Cluster Tree' plot-type mode, delegates entirely to
        _draw_cluster_tree_view() instead — a tree has no DR-run concept."""
        if self.plot_type_combo.currentText() == 'Cluster Tree':
            self._draw_cluster_tree_view()
            return

        run = self._selected_dr_run()
        if run is None:
            self._show_placeholder("No DR run selected.\nTrain a DR algorithm first.")
            return
        algo = run.get('algorithm', '') or ''
        emb_dict = run.get('embeddings', {}) or {}
        if not emb_dict:
            self._show_placeholder(
                f"No embeddings for \"{run.get('label', '')}\".\n"
                "Run 'Apply to All Samples' for this run."
            )
            return

        cl_run = self._selected_cluster_run()
        labels_dict = cl_run.get('labels', {}) if cl_run else {}
        join_mode = drc_run_align.runs_join_mode(run, cl_run) if cl_run else None
        unaligned_samples: list[str] = []

        def _labels_on(rel, n):
            return drc_scatter.align_labels_to_embedding(
                labels_dict.get(rel), n,
                label_rows=drc_run_align.run_rows(cl_run, rel) if cl_run else None,
                emb_rows=drc_run_align.run_rows(run, rel), mode=join_mode)

        # Gather data (plus the sample-of-origin per row, needed for Marker mode)
        sample_data_item = self.sample_combo.currentData()
        if sample_data_item:
            # Single sample
            rel_path = sample_data_item
            emb = emb_dict.get(rel_path)
            if emb is None:
                self._show_placeholder(f"No embedding for {Path(rel_path).name}.")
                return
            xy = emb
            lbl_raw = labels_dict.get(rel_path)
            lab, aligned_ok = _labels_on(rel_path, len(emb))
            if lbl_raw is not None and not aligned_ok:
                unaligned_samples.append(rel_path)
            origin = np.array([rel_path] * len(emb), dtype=object)
            sample_row_offsets = {rel_path: (0, len(emb))}
        else:
            # All samples pooled — track row offsets per sample for marker mode
            xys, labs, origins = [], [], []
            sample_row_offsets: dict[str, tuple[int, int]] = {}
            offset = 0
            for rel, emb in emb_dict.items():
                n = len(emb)
                sample_row_offsets[rel] = (offset, offset + n)
                offset += n
                xys.append(emb)
                origins.append(np.array([rel] * n, dtype=object))
                lbl_raw = labels_dict.get(rel)
                lbl, aligned_ok = _labels_on(rel, n)
                if lbl_raw is None:
                    _log.debug("workspace: no cluster labels for %s — greyed", rel)
                elif not aligned_ok:
                    _log.warning("workspace: labels of %s do not cover every embedded "
                                 "event (%d labels, %d points) — unmatched points greyed",
                                 rel, len(lbl_raw), n)
                    unaligned_samples.append(rel)
                labs.append(lbl)
            if not xys:
                self._show_placeholder("No embeddings found.")
                return
            xy = np.concatenate(xys, axis=0)
            lab = np.concatenate(labs, axis=0)
            origin = np.concatenate(origins, axis=0)

        self._last_unaligned_samples = unaligned_samples

        colour_mode = self.colour_mode_combo.currentText()

        # Marker mode needs untransformed per-sample values -- disk I/O +
        # AF-aware unmixing for every sample shown, the single most
        # expensive step in this whole method (see
        # drc_pipeline.load_unmixed_gated's docstring). Load them on a
        # background thread and re-enter refresh() once they land, rather
        # than blocking the UI here. A per-card cache (_marker_cache_key)
        # means this only actually happens on a cold cache -- first
        # Marker-mode draw for this sample set / gate selection / unmixing
        # matrix -- repeat draws (theme, columns, even switching marker
        # channel) hit the cache and fall straight through.
        marker_values_by_sample = None
        if colour_mode == 'Marker':
            needed_samples = sorted({str(r) for r in np.unique(origin)})
            cache_key = self._marker_cache_key(needed_samples)
            marker_values_by_sample = self._marker_values_cache.get(cache_key)
            if marker_values_by_sample is None:
                self._show_placeholder("Loading marker values …")
                self._start_marker_values_worker(needed_samples, cache_key)
                return

        is_dark = _resolve_is_dark(self.state)

        self._figure.clear()
        if colour_mode == 'Marker':
            gs = self._figure.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.35)
            ax = self._figure.add_subplot(gs[0, 0])
            self._cbar_ax = self._figure.add_subplot(gs[0, 1])
            self._cbar_ax.set_visible(False)
        else:
            ax = self._figure.add_subplot(111)
            self._cbar_ax = None
        _style_figure_theme(self._figure, is_dark, axes=[ax])
        # box_aspect gives the axes box the plot's own height:width,
        # whatever the colourbar column or the embedding's x/y range;
        # adjustable='datalim' pads xlim/ylim to match instead of
        # reshaping the box.
        ax.set_box_aspect(self._plot_h_in / self._plot_w_in)
        ax.set_aspect('equal', adjustable='datalim')
        _af = self._axis_font_spin.value()
        if self._show_axis_labels.isChecked():
            ax.set_xlabel(f"{algo} 1", fontsize=_af)
            ax.set_ylabel(f"{algo} 2", fontsize=_af)
        else:
            ax.set_xlabel('')
            ax.set_ylabel('')
        if self._show_ticks.isChecked():
            ax.tick_params(labelsize=_af)
        else:
            ax.tick_params(labelbottom=False, labelleft=False)
        if self._show_grid.isChecked():
            ax.grid(True, linewidth=0.4, alpha=0.5)
        else:
            ax.grid(False)

        # Downsample for display speed (max 30k points)
        if len(xy) > 30_000:
            rng = np.random.default_rng(0)
            disp_idx = rng.choice(len(xy), 30_000, replace=False)
            xy_disp     = xy[disp_idx]
            lab_disp    = lab[disp_idx] if lab is not None else None
            origin_disp = origin[disp_idx]
        else:
            disp_idx    = None
            xy_disp     = xy
            lab_disp    = lab
            origin_disp = origin

        # cached so _legend_entries() (called right after
        # this method, and only from here) knows which samples/groups are
        # actually present in the CURRENT view without recomputing origin.
        self._last_origin_disp = origin_disp

        if colour_mode == 'Clusters':
            self._draw_cluster_scatter(ax, xy_disp, lab_disp, cl_run)
            if self._last_unaligned_samples:
                ax.text(0.02, 0.02,
                       f"⚠ {len(self._last_unaligned_samples)} sample(s) shown "
                       "without cluster colour — DR run and clustering run "
                       "don't share indexed events",
                       transform=ax.transAxes, fontsize=max(_af - 1, 6),
                       color='#d9822b', va='bottom', ha='left', zorder=10)
        elif colour_mode == 'Marker':
            self._draw_marker_scatter(ax, xy_disp, origin_disp,
                                      disp_idx, sample_row_offsets, run,
                                      marker_values_by_sample)
        elif colour_mode == 'Group':
            self._draw_group_scatter(ax, xy_disp, origin_disp)

        # Apply label colour to colourbar (Marker), title, and axis labels
        lc = self._label_fg()
        ax.title.set_color(lc)
        ax.xaxis.label.set_color(lc)
        ax.yaxis.label.set_color(lc)
        for cb_ax in self._figure.axes:
            if cb_ax is not ax:
                cb_ax.tick_params(colors=lc)
                cb_ax.yaxis.label.set_color(lc)

        self._legend_ax = ax
        self._rebuild_legend()

    # ------------------------------------------------------------------
    # Marker-mode background loading
    # ------------------------------------------------------------------

    def _marker_cache_key(self, samples: list[str]):
        """Cache key for _marker_values_cache. Deliberately excludes the
        selected marker channel -- load_sample_marker_values returns ALL
        selected channels for a sample in one call, so switching which
        channel is displayed never needs a reload, only a re-draw."""
        return (
            tuple(sorted(samples)),
            tuple(sorted(self.state.selected_gates)),
            id(self.controller.transfer_matrix),
            unmix_method.method_token(unmix_method.method_state(self.controller)),
        )

    def _start_marker_values_worker(self, samples: list[str], cache_key):
        """Load per-sample marker values on a background thread. AF/
        transfer-matrix state is snapshotted HERE, on the main thread,
        before the worker starts -- same reasoning as _MarkerSummaryWorker
        (reading it live off the controller from a background thread is a
        memory-corruption hazard, not just a stale-data one)."""
        if self._marker_worker is not None:
            return   # already loading; its finish will re-trigger refresh()
        if wait_if_busy(self.controller, ('Honeycluster marker values', id(self)), self.refresh,
                        self._show_placeholder):
            return
        af_state = drc_pipeline.snapshot_af_state(self.controller)
        worker = _MarkerValuesWorker(self.controller, self.state, samples, af_state)
        worker.finished.connect(
            lambda ok, err, payload, key=cache_key:
                self._on_marker_values_finished(ok, err, payload, key)
        )
        self._marker_worker = worker
        worker.start()

    def _on_marker_values_finished(self, success: bool, error: str,
                                   payload: dict, cache_key):
        self._marker_worker = None
        if not success:
            self._show_placeholder(f"Failed to load marker values: {error}")
            return
        # Single-entry cache -- only one gate/matrix combination is ever
        # "current" in a session, and each entry holds full per-sample
        # gated arrays, so keeping stale combinations around would just be
        # wasted memory for data nothing will ask for again.
        self._marker_values_cache = {cache_key: payload}
        self.refresh()

    def _start_tree_layout_worker(self, run_id: str, tree_data: dict):
        """Compute the Cluster Tree's MST + Kamada-Kawai layout on a
        background thread (see _ClusterTreeLayoutWorker) -- pure
        computation over already-archived tree_data, so unlike the
        marker-values worker it needs no AF/transfer-matrix snapshot."""
        worker = _ClusterTreeLayoutWorker(
            tree_data['node_weights'], tree_data['node_to_meta'], tree_data['node_counts'])
        worker.finished.connect(
            lambda ok, err, layout, rid=run_id:
                self._on_tree_layout_finished(ok, err, layout, rid)
        )
        self._tree_layout_worker = worker
        worker.start()

    def _on_tree_layout_finished(self, success: bool, error: str,
                                 layout: dict, run_id: str):
        self._tree_layout_worker = None
        if not success:
            self._show_placeholder(f"Failed to build tree layout: {error}")
            return
        self._tree_layout_cache[run_id] = layout
        self.refresh()

    def _draw_cluster_scatter(self, ax, xy, labels, cl_run: dict | None):
        """
        Colour points by cluster label. Delegates to drc_scatter
        (extracted so the Cluster Annotation tab's cluster-map panel renders
        identically instead of duplicating this logic).
        """
        drc_scatter.draw_cluster_scatter(ax, xy, labels, cl_run, self.controller)

    def _draw_cluster_tree_view(self):
        """
        Render a cluster tree: node bubbles sized by cell count, coloured
        by cluster/metacluster, connected by minimum-spanning-tree edges
        over the node weight vectors (classic FlowSOM tree layout, see
        drc_clustering.build_flowsom_tree -- a pure MST+layout function
        that doesn't care what the "nodes" represent). Works for any
        clustering algorithm: FlowSOM's nodes are its SOM codebook
        vectors (many nodes roll up into few metaclusters); Leiden/
        HDBSCAN have no such sub-structure, so each node IS a final
        cluster (see drc_clustering.build_centroid_tree_data). Wired to
        _rebuild_legend(), same as the Scatter path -- without this the
        legend kept showing whatever run/colour-mode was last rendered in
        Scatter view (plain 0..n numbers) instead of the selected run's
        own adopted names (e.g. MEM/cell-type labels from "Compute
        Cluster ID Suggestions") whenever the run shown in the Tree
        changed.
        """
        cl_run = self._selected_cluster_run()
        if cl_run is None:
            self._show_placeholder(
                "Select a clustering run in the Overlay dropdown."
            )
            return
        tree_data = cl_run.get('tree_data')
        if not tree_data:
            self._show_placeholder(
                f"This {cl_run.get('algorithm', '')} run has no tree data.\n"
                "Re-run clustering to enable the tree view."
            )
            return

        # The MST/layout computation (the Kamada-Kawai layout, in
        # particular) can be slow for runs with many nodes -- HDBSCAN
        # especially -- so it's computed on a background thread and
        # cached per run_id (tree_data is frozen at archive time, so a
        # run's layout never needs recomputing once it's built).
        run_id = cl_run.get('run_id')
        layout = self._tree_layout_cache.get(run_id)
        if layout is None:
            self._show_placeholder("Computing tree layout …")
            if self._tree_layout_worker is None:
                self._start_tree_layout_worker(run_id, tree_data)
            return

        is_dark = _resolve_is_dark(self.state)
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        _style_figure_theme(self._figure, is_dark, axes=[ax])
        ax.set_box_aspect(self._plot_h_in / self._plot_w_in)
        ax.set_aspect('equal', adjustable='datalim')
        ax.axis('off')

        positions    = layout['positions']
        edges        = layout['edges']
        node_to_meta = tree_data['node_to_meta']
        node_counts  = tree_data['node_counts']

        colors_dict = cl_run.get('colors', {})
        node_colors = [colors_dict.get(int(m), '#7f7f7f') for m in node_to_meta]

        edge_color = '#888888' if is_dark else '#bbbbbb'
        for i, j in edges:
            ax.plot([positions[i, 0], positions[j, 0]],
                    [positions[i, 1], positions[j, 1]],
                    color=edge_color, linewidth=0.8, zorder=1)

        max_count = max(int(node_counts.max()), 1)
        sizes = 20 + 480 * (node_counts / max_count)
        ax.scatter(positions[:, 0], positions[:, 1], s=sizes, c=node_colors,
                   edgecolors=('#dddddd' if is_dark else '#222222'),
                   linewidths=0.4, zorder=2)

        lc = self._label_fg()
        ax.set_title(f"{cl_run.get('algorithm', '')} Cluster Tree — "
                    f"{cl_run.get('label', '')}", color=lc,
                    fontsize=self._axis_font_spin.value())
        ax.title.set_color(lc)

        self._legend_ax = ax
        self._rebuild_legend()

    def _draw_marker_scatter(self, ax, xy, origin, disp_idx, sample_row_offsets,
                             run: dict, marker_values_by_sample: dict):
        """Colour by UNTRANSFORMED marker intensity on the full data scale.
        marker_values_by_sample is {rel_path: (values, names, rows)} for every
        sample in *origin*, already loaded by refresh() (synchronously from
        cache, or via _MarkerValuesWorker on a cold cache) -- this method
        does no disk I/O of its own."""
        ch = self.marker_combo.currentData()
        if not ch or not self.state.selected_channels:
            ax.scatter(xy[:, 0], xy[:, 1], s=1, c='#aaaaaa', alpha=0.4, rasterized=True)
            ax.set_title("Marker: no channel", fontsize=8)
            return

        # Marker values come from the CURRENT gating and unmixing.  A run
        # with event keys is matched to them event by event (points whose
        # event is not gated now stay grey); a legacy run only row for row,
        # and only when it is in sync and was made with the selected gates.
        mode = drc_run_align.live_join_mode(run, drc_pipeline.sync_fingerprint(
            self.controller, run.get('gates', []), run.get('training_sample_ids', [])))
        run_gates = run.get('gates')
        if mode != drc_run_align.JOIN_KEYS and (
                mode == drc_run_align.JOIN_NONE or run_gates is None
                or sorted(run_gates) != sorted(self.state.selected_gates)):
            ax.scatter(xy[:, 0], xy[:, 1], s=1, c='#cccccc', alpha=0.3, rasterized=True)
            ax.set_title(f"{run.get('label', '')}: made before event keys and no longer "
                         "matches the current gating — retrain DR", fontsize=8)
            _log.warning("marker scatter: run %r cannot be matched to the current events",
                         run.get('label'))
            return

        values = np.full(len(xy), np.nan, dtype=np.float32)
        n_unmatched = 0
        for rel in np.unique(origin):
            mv = marker_values_by_sample.get(str(rel))
            if mv is None:
                continue
            vals, names, cur_rows = mv
            if ch not in names:
                continue
            start, end = sample_row_offsets.get(rel, (0, int((origin == rel).sum())))
            col, matched = drc_run_align.pair_arrays(
                vals[:, names.index(ch)].astype(np.float32), end - start, mode,
                src_rows=cur_rows, dst_rows=drc_run_align.run_rows(run, rel), fill=np.nan)
            if col is None:
                _log.warning("marker %s: %s embedding has %d points but current "
                             "gating yields %d events — skipped", ch, rel, end - start,
                             len(vals))
                continue
            n_unmatched += int((~matched).sum())
            mask = origin == rel   # which rows in the display array belong to this sample
            if disp_idx is not None:
                values[mask] = col[disp_idx[mask] - start]
            else:
                values[mask] = col
        if n_unmatched:
            _log.info("marker %s: %d embedded event(s) are not in the current gating "
                      "and are not drawn", ch, n_unmatched)

        finite = np.isfinite(values)
        display_label = _antigen_dash_labels(self.controller).get(ch, ch)
        if not finite.any():
            _log.warning("marker %s: no values could be loaded", ch)
            ax.scatter(xy[:, 0], xy[:, 1], s=1, c='#cccccc', alpha=0.4, rasterized=True)
            ax.set_title(f"Marker {display_label}: no data", fontsize=8)
            return

        # Plot on log1p(raw) so the display isn't dominated by the extreme
        # top-end fuzz, but label the colourbar in raw units at powers of
        # ten -- same convention as Honeychrome's 2D histogram axes
        # (transform.py), just log1p instead of the full logicle transform.
        log_values = np.log1p(np.maximum(values, 0.0))
        vmin = float(np.nanpercentile(log_values[finite], 1))
        vmax = float(np.nanpercentile(log_values[finite], 99))
        _log.debug("marker %s: vmin=%.4g vmax=%.4g (log1p) over %d points",
                   ch, vmin, vmax, int(finite.sum()))
        sc = ax.scatter(xy[finite, 0], xy[finite, 1], s=1,
                        c=log_values[finite], cmap='viridis',
                        vmin=vmin, vmax=vmax, alpha=0.6, linewidths=0, rasterized=True)
        self._cbar_ax.set_visible(True)
        cb = self._figure.colorbar(sc, cax=self._cbar_ax)
        tick_pos, tick_lab = _log1p_powers_of_ten_ticks(float(np.expm1(vmax)))
        keep = [(p, l) for p, l in zip(tick_pos, tick_lab) if vmin <= p <= vmax]
        if keep:
            cb.set_ticks([p for p, l in keep])
            cb.set_ticklabels([l for p, l in keep])
        # "Legend font" now sizes the Marker colourbar's label and
        # tick text too, not just the Clusters-mode swatch legend.
        _lf = self._legend_font_spin.value()
        cb.set_label(display_label, fontsize=_lf)
        cb.ax.tick_params(labelsize=_lf)
        ax.set_title(f"Marker: {display_label}", fontsize=8)

    def _draw_group_scatter(self, ax, xy, origin):
        """
        Colour points by their sample's assigned comparison Group,
        sharing state.group_colors with the Comparison Groups
        table's Colour column and the Sample PCA plot via
        _resolve_group_colour(). 'Unassigned' samples are grey.
        """
        rel_to_group = _sample_groups_by_rel(self.controller, self.state)
        groups = np.array(
            [rel_to_group.get(str(rel), 'Unassigned') for rel in origin],
            dtype=object,
        )
        present = set(np.unique(groups))
        ordered = [g for g in self.state.group_names if g in present]
        if 'Unassigned' in present:
            ordered.append('Unassigned')
        if not ordered:
            ax.scatter(xy[:, 0], xy[:, 1], s=1, c='#7f7f7f', alpha=0.4, rasterized=True)
            ax.set_title("Group: no samples assigned", fontsize=8)
            return
        for grp in ordered:
            colour = _resolve_group_colour(self.state, grp) if grp != 'Unassigned' else '#7f7f7f'
            mask = groups == grp
            ax.scatter(xy[mask, 0], xy[mask, 1], s=1, c=colour, alpha=0.6, linewidths=0,
                       rasterized=True)
        ax.set_title("Group", fontsize=8)

    def _show_placeholder(self, text: str):
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.text(0.5, 0.5, text, ha='center', va='center',
                transform=ax.transAxes, color='grey', fontsize=9,
                wrap=True)
        ax.axis('off')
        self.figure_changed.emit()

    # ------------------------------------------------------------------
    # Legend (drawn in the figure; right-click colour, double-click rename)
    # ------------------------------------------------------------------

    def _legend_entries(self) -> tuple[str, list[tuple[object, str, str]]]:
        """('cluster'|'group', [(key, name, colour), ...]) for the current
        view; an empty list when it has no legend. Clusters come from the
        selected clustering run's own 'colors'/'names'; groups are those
        present in the current plot. Cluster Tree mode always uses the
        cluster legend (its colour-mode combo is hidden and may be stale)."""
        is_tree = self.plot_type_combo.currentText() == 'Cluster Tree'
        mode = self.colour_mode_combo.currentText()
        if mode == 'Group' and not is_tree:
            origin_disp = getattr(self, '_last_origin_disp', None)
            if origin_disp is None:
                return 'group', []
            rel_to_group = _sample_groups_by_rel(self.controller, self.state)
            present = {rel_to_group.get(str(rel), 'Unassigned') for rel in origin_disp}
            ordered = [g for g in self.state.group_names if g in present]
            if 'Unassigned' in present:
                ordered.append('Unassigned')
            return 'group', [
                (g, g, _resolve_group_colour(self.state, g) if g != 'Unassigned' else '#7f7f7f')
                for g in ordered]
        if mode != 'Clusters' and not is_tree:
            return 'cluster', []
        cl_run = self._selected_cluster_run()
        colors = cl_run.get('colors', {}) if cl_run else {}
        names = cl_run.get('names', {}) if cl_run else {}
        return 'cluster', [(lbl, _cluster_display_name(names, lbl), colors[lbl])
                           for lbl in sorted(colors.keys())]

    def _legend_layout_for(self, names: list[str], font_pt: float,
                           position: str) -> tuple[int, float, float]:
        """(n_columns, extra width in, extra height in) of a legend with
        these entries: the Cols spin, or Auto (as many rows as fit the plot
        height at the right; as many columns as fit the plot width below)."""
        em = font_pt / 72.0
        row_h = em * (1.0 + LEGEND_LABEL_SPACING)
        col_w = max(text_width_inches(n, font_pt) for n in names) + em * (
            LEGEND_HANDLE_LENGTH + LEGEND_HANDLE_PAD)
        col_gap = em * LEGEND_COLUMN_SPACING
        n = len(names)
        ncol = self._legend_cols_spin.value()
        if position == 'Right':
            if ncol <= 0:
                rows_fit = max(1, int(self._plot_h_in * 0.9 / row_h))
                ncol = -(-n // rows_fit)
            ncol = min(ncol, n)
            return ncol, LEGEND_GAP_IN + ncol * col_w + (ncol - 1) * col_gap, 0.0
        if ncol <= 0:
            ncol = max(1, int((self._plot_w_in + col_gap) / (col_w + col_gap)))
        ncol = min(ncol, n)
        rows = -(-n // ncol)
        return ncol, 0.0, LEGEND_GAP_IN + rows * row_h

    def _apply_figure_size(self):
        """Figure = plot W x H plus the legend's extra width/height."""
        w = self._plot_w_in + self._legend_extra_in[0]
        h = self._plot_h_in + self._legend_extra_in[1]
        self._figure.set_size_inches(w, h, forward=False)

    def estimated_size_mm(self) -> tuple[float, float]:
        """Figure size in mm; before the legend has been drawn, the plot
        plus LEGEND_ALLOWANCE_MM on the side the legend goes."""
        if self._legend_extra_in != (0.0, 0.0):
            w_in, h_in = self._figure.get_size_inches()
            return float(w_in) * MM_PER_INCH, float(h_in) * MM_PER_INCH
        w, h = self._plot_w_spin.value(), self._plot_h_spin.value()
        position = self._legend_pos_combo.currentText()
        return (w + (LEGEND_ALLOWANCE_MM if position in ('Right', 'Separate') else 0.0),
                h + (LEGEND_ALLOWANCE_MM if position == 'Below' else 0.0))

    def _rebuild_legend(self):
        """Draw the cluster/group legend into the figure beside (axes legend)
        or below (figure legend) self._legend_ax, resize the figure to make
        room for it and register
        its entries for right-click recolour / double-click rename."""
        ax = getattr(self, '_legend_ax', None)
        self._legend_pick_targets = {}
        if ax is not None and ax in self._figure.axes and ax.get_legend() is not None:
            ax.get_legend().remove()
        for old in list(self._figure.legends):
            old.remove()
        kind, entries = self._legend_entries()
        position = self._legend_pos_combo.currentText()
        valid = ax is not None and ax in self._figure.axes and bool(entries)
        self.legend_shown = valid and position == 'Separate'
        if self.legend_shown:
            self._draw_separate_legend(kind, entries)
        if not valid or position in ('Off', 'Separate'):
            self._legend_extra_in = (0.0, 0.0)
            self._apply_figure_size()
            self.figure_changed.emit()
            self.legend_changed.emit()
            return

        from matplotlib.lines import Line2D
        font_pt = float(self._legend_font_spin.value())
        names = [name for _key, name, _colour in entries]
        ncol, extra_w, extra_h = self._legend_layout_for(names, font_pt, position)
        handles = [Line2D([], [], linestyle='', marker='o', markersize=font_pt * 0.7,
                          markerfacecolor=colour, markeredgewidth=0, label=name)
                   for _key, name, colour in entries]
        common = dict(handles=handles, ncol=ncol, fontsize=font_pt, frameon=False,
                      borderaxespad=0.0, handlelength=LEGEND_HANDLE_LENGTH,
                      handletextpad=LEGEND_HANDLE_PAD, labelspacing=LEGEND_LABEL_SPACING,
                      columnspacing=LEGEND_COLUMN_SPACING, labelcolor=self._label_fg())
        if position == 'Right':
            leg = ax.legend(loc='upper left', bbox_to_anchor=(1.03, 1.0), **common)
        else:
            # Figure legend: the layout engine keeps it clear of the tick
            # labels and axis title at any font size or plot size.
            leg = self._figure.legend(loc='outside lower center', **common)
        self._register_legend_picks(leg, kind, entries)

        self._legend_extra_in = (extra_w, extra_h)
        self._apply_figure_size()
        self.figure_changed.emit()
        self.legend_changed.emit()

    def _register_legend_picks(self, leg, kind, entries):
        for (key, _name, _colour), handle, text in zip(entries, leg.legend_handles,
                                                      leg.get_texts()):
            handle.set_picker(5)
            text.set_picker(True)
            self._legend_pick_targets[id(handle)] = (kind, key, 'swatch', _name)
            self._legend_pick_targets[id(text)] = (kind, key, 'name', _name)

    def _legend_metrics(self, names: list[str], font_pt: float) -> tuple[float, float, float]:
        """(row height, column width, column gap) in inches of a legend with
        these entries at font_pt, from matplotlib's font metrics."""
        em = font_pt / 72.0
        row_h = em * (1.0 + LEGEND_LABEL_SPACING)
        col_w = max(text_width_inches(n, font_pt) for n in names) + em * (
            LEGEND_HANDLE_LENGTH + LEGEND_HANDLE_PAD)
        return row_h, col_w, em * LEGEND_COLUMN_SPACING

    def _separate_legend_ncol(self, names: list[str], font_pt: float) -> int:
        """Legend columns: the Columns setting, or Auto (as many rows as fit
        the plot's height)."""
        ncol = self._legend_cols_spin.value()
        if ncol <= 0:
            row_h, _col_w, _gap = self._legend_metrics(names, font_pt)
            rows_fit = max(1, int(self._plot_h_in * 0.95 / row_h))
            ncol = -(-len(names) // rows_fit)
        return max(1, min(ncol, len(names)))

    def _draw_separate_legend(self, kind, entries):
        """Draw the legend alone into self._legend_figure, sized to fit it
        (font size fixed in pt; the column count sets its shape)."""
        from matplotlib.lines import Line2D
        fig = self._legend_figure
        fig.clear()
        fig.patch.set_facecolor('#1e1e1e' if _resolve_is_dark(self.state) else 'white')
        font_pt = float(self._legend_font_spin.value())
        names = [name for _key, name, _colour in entries]
        ncol = self._separate_legend_ncol(names, font_pt)
        row_h, col_w, gap = self._legend_metrics(names, font_pt)
        rows = -(-len(names) // ncol)
        # Large enough to hold the legend while it is measured
        fig.set_size_inches(ncol * (col_w + gap) + 1.0, rows * row_h + 1.0)
        handles = [Line2D([], [], linestyle='', marker='o', markersize=font_pt * 0.7,
                          markerfacecolor=colour, markeredgewidth=0, label=name)
                   for _key, name, colour in entries]
        leg = fig.legend(handles=handles, ncol=ncol, fontsize=font_pt, frameon=False,
                         loc='upper left', bbox_to_anchor=(0.0, 1.0), borderaxespad=0.0,
                         handlelength=LEGEND_HANDLE_LENGTH, handletextpad=LEGEND_HANDLE_PAD,
                         labelspacing=LEGEND_LABEL_SPACING, columnspacing=LEGEND_COLUMN_SPACING,
                         labelcolor=self._label_fg())
        bb = leg.get_window_extent(fig.canvas.get_renderer())
        fig.set_size_inches(max(bb.width / fig.dpi, 0.05), max(bb.height / fig.dpi, 0.05))
        self._register_legend_picks(leg, kind, entries)

    def legend_figure(self):
        return self._legend_figure

    def legend_columns(self) -> int:
        return self._legend_cols_spin.value()

    def set_legend_columns(self, ncol: int):
        self._legend_cols_spin.setValue(int(ncol))

    def set_legend_box_mm(self, w_mm: float, h_mm: float, edge: str):
        """Reflow the separate legend from a box dragged to w_mm x h_mm: the
        right edge sets how many columns fit the width; the bottom edge or
        corner sets how many rows fit the height."""
        kind, entries = self._legend_entries()
        if not entries:
            return
        import matplotlib
        names = [name for _key, name, _colour in entries]
        font_pt = float(self._legend_font_spin.value())
        row_h, col_w, gap = self._legend_metrics(names, font_pt)
        border = 2 * matplotlib.rcParams['legend.borderpad'] * font_pt / 72.0
        n = len(names)
        if edge == 'right':
            ncol = int((w_mm / MM_PER_INCH - border + gap) / (col_w + gap))
        else:
            rows = max(1, int((h_mm / MM_PER_INCH - border) / row_h))
            ncol = -(-n // rows)
        self.set_legend_columns(max(1, min(ncol, n)))

    def _on_legend_cols_changed(self, *_):
        if self._legend_pos_combo.currentText() == 'Separate':
            self._sync_config_to_state()
            self._rebuild_legend()
        else:
            self._schedule_refresh()

    def _on_legend_pick(self, event):
        """Double-click a cluster's name: rename it. While probing (the page
        is building a right-click menu) only record which entry is there."""
        target = self._legend_pick_targets.get(id(event.artist))
        if target is None:
            return
        kind, key, part, name = target
        if self.legend_probe:
            if self.legend_probe_hit is None:
                self.legend_probe_hit = (kind, key, name)
            return
        if self.legend_pick_handled:
            return
        if event.mouseevent.dblclick and part == 'name' and kind == 'cluster':
            self.legend_pick_handled = True
            self._rename_cluster(key)

    def legend_entry_action(self, action: str, kind: str, key):
        """A legend entry's menu action: 'entry_colour' or 'entry_rename'."""
        if action == 'entry_colour':
            if kind == 'cluster':
                self._pick_cluster_colour(key)
            elif key != 'Unassigned':
                self._pick_group_colour(key)
        elif action == 'entry_rename' and kind == 'cluster':
            self._rename_cluster(key)

    def _pick_group_colour(self, name: str):
        """Open a colour dialog to change a group's colour--
        writes to the same state.group_colors dict as the Comparison
        Groups table's Colour column and the Sample PCA plot, so the
        change shows up in all three immediately."""
        current = _resolve_group_colour(self.state, name)
        colour = QColorDialog.getColor(QColor(current), self, f"Colour for group '{name}'")
        if colour.isValid():
            self.state.group_colors[name] = colour.name()
            self.refresh()

    def _pick_cluster_colour(self, label: int):
        """Open a colour dialog to change a cluster's colour. Persistence
        delegates to drc_scatter (— shared with the Cluster
        Annotation tab)."""
        cl_run = self._selected_cluster_run()
        if cl_run is None:
            return
        current = cl_run.get('colors', {}).get(label, '#aaaaaa')
        colour = QColorDialog.getColor(QColor(current), self, f"Colour for cluster {label}")
        if colour.isValid():
            drc_scatter.recolor_cluster(self.controller, cl_run, label, colour.name())
            self.refresh()

    def _rename_cluster(self, label: int):
        """Inline rename for a cluster via an input dialog. Duplicate-check
        and persistence delegate to drc_scatter (shared with the
        Cluster Annotation tab's rename paths)."""
        from PySide6.QtWidgets import QInputDialog
        cl_run = self._selected_cluster_run()
        if cl_run is None:
            return
        names = cl_run.get('names', {})
        current = _cluster_display_name(names, label)
        new_name, ok = QInputDialog.getText(
            self, f"Rename cluster {label}", "New name:", text=current
        )
        if not ok or not new_name.strip():
            return
        if drc_scatter.rename_cluster(self.controller, cl_run, label, new_name.strip(), self):
            self._rebuild_legend()
    
    def _on_plot_dims_changed(self):
        """Plot W/H (mm) changed — resize the figure (plus any legend)."""
        self._plot_w_in = self._plot_w_spin.value() / MM_PER_INCH
        self._plot_h_in = self._plot_h_spin.value() / MM_PER_INCH
        self._apply_figure_size()
        self.figure_changed.emit()
        self._schedule_refresh()

    def _set_spin_quietly(self, spin, value: float):
        spin.blockSignals(True)
        spin.setValue(min(max(value, PLOT_MIN_MM), PLOT_MAX_MM))
        spin.blockSignals(False)

    def _on_plot_w_changed(self, value: float):
        if self._aspect_check.isChecked() and not self._applying_size:
            self._set_spin_quietly(self._plot_h_spin, value / self._aspect_ratio)
        self._on_plot_dims_changed()

    def _on_plot_h_changed(self, value: float):
        if self._aspect_check.isChecked() and not self._applying_size:
            self._set_spin_quietly(self._plot_w_spin, value * self._aspect_ratio)
        self._on_plot_dims_changed()

    def _on_aspect_toggled(self, locked: bool):
        if locked:
            self._aspect_ratio = self._plot_w_spin.value() / self._plot_h_spin.value()
        self._sync_config_to_state()

    def aspect_locked(self) -> bool:
        return self._aspect_check.isChecked()

    def plot_size_mm(self) -> tuple[float, float]:
        return self._plot_w_spin.value(), self._plot_h_spin.value()

    def set_plot_size_mm(self, w_mm: float, h_mm: float):
        """Set the plot size exactly (the locked ratio follows it)."""
        self._set_spin_quietly(self._plot_w_spin, w_mm)
        self._set_spin_quietly(self._plot_h_spin, h_mm)
        if self._aspect_check.isChecked():
            self._aspect_ratio = self._plot_w_spin.value() / self._plot_h_spin.value()
        self._on_plot_dims_changed()

    def set_plot_size_from_item(self, w_mm: float, h_mm: float, edge: str):
        """Resize from the page: the figure was dragged to w_mm x h_mm. The
        plot takes what an attached legend leaves; with the aspect ratio
        locked, the dragged edge decides and the other side follows."""
        extra_w, extra_h = (v * MM_PER_INCH for v in self._legend_extra_in)
        w, h = w_mm - extra_w, h_mm - extra_h
        if self._aspect_check.isChecked():
            w0, h0 = self.plot_size_mm()
            w, h = drc_workspace_canvas.locked_size(w0, h0, w, h, edge)
        self.set_plot_size_mm(w, h)

    def apply_size_config(self, config: dict):
        """Plot size, aspect lock, legend placement and legend offset from a
        saved config, without redrawing, so a new plot can be placed before
        it is drawn."""
        widgets = (self._plot_w_spin, self._plot_h_spin, self._legend_pos_combo,
                   self._aspect_check)
        for widget in widgets:
            widget.blockSignals(True)
        if 'plot_w' in config:
            self._plot_w_spin.setValue(float(config['plot_w']) * MM_PER_INCH)
        if 'plot_h' in config:
            self._plot_h_spin.setValue(float(config['plot_h']) * MM_PER_INCH)
        if config.get('legend_pos') in LEGEND_POSITIONS:
            self._legend_pos_combo.setCurrentText(config['legend_pos'])
        if 'aspect_locked' in config:
            self._aspect_check.setChecked(bool(config['aspect_locked']))
        for widget in widgets:
            widget.blockSignals(False)
        offset = config.get('legend_offset')
        self.legend_offset_mm = (float(offset[0]), float(offset[1])) if offset else None
        self._aspect_ratio = self._plot_w_spin.value() / self._plot_h_spin.value()
        self._plot_w_in = self._plot_w_spin.value() / MM_PER_INCH
        self._plot_h_in = self._plot_h_spin.value() / MM_PER_INCH
        self._apply_figure_size()

    def _label_fg(self) -> str:
        """Label colour in use: the one picked, or Auto (black on a light
        plot theme, white on a dark one)."""
        if self._label_color:
            return self._label_color
        return 'white' if _resolve_is_dark(self.state) else 'black'

    def _pick_label_color(self):
        """Open colour dialog to set colourbar / title text colour."""
        colour = QColorDialog.getColor(QColor(self._label_fg()), self, "Label colour")
        if colour.isValid():
            self._label_color = colour.name()
            self._update_label_color_btn()
            self._schedule_refresh()

    def _reset_label_color(self):
        self._label_color = None
        self._update_label_color_btn()
        self._schedule_refresh()

    def _update_label_color_btn(self):
        """Show the chosen colour on the button (with contrasting text), or
        'Auto' when the colour follows the plot theme."""
        if self._label_color is None:
            self._label_color_btn.setText("Auto (follows theme)")
            self._label_color_btn.setStyleSheet("")
            return
        self._label_color_btn.setText(self._label_color)
        text_color = _contrasting_text_color(self._label_color)
        self._label_color_btn.setStyleSheet(
            f"background-color: {self._label_color}; color: {text_color};"
        )

    # ------------------------------------------------------------------
    # Magic wand copy / paste
    # ------------------------------------------------------------------

    def get_display_config(self) -> dict:
        return {
            'dr_run_id':      self.dr_combo.currentData(),
            'cluster_run_id': self.cluster_run_combo.currentData(),
            'sample':         self.sample_combo.currentData(),
            'colour_mode':    self.colour_mode_combo.currentText(),
            'marker_ch':      self.marker_combo.currentData(),
            'show_grid':      self._show_grid.isChecked(),
            'show_ticks':     self._show_ticks.isChecked(),
            'show_axis_labels': self._show_axis_labels.isChecked(),
            'title_font':     self._legend_font_spin.value(),  # key kept for backward-compat with saved layouts
            'axis_font':      self._axis_font_spin.value(),
            'label_color':    self._label_color,   # None = Auto
            'plot_w':         round(self._plot_w_spin.value() / MM_PER_INCH, 4),  # inches
            'plot_h':         round(self._plot_h_spin.value() / MM_PER_INCH, 4),
            'aspect_locked':  self._aspect_check.isChecked(),
            'legend_pos':     self._legend_pos_combo.currentText(),
            'legend_cols':    self._legend_cols_spin.value(),
            'legend_offset':  list(self.legend_offset_mm) if self.legend_offset_mm else None,
            'layout':         dict(self.layout_mm) if self.layout_mm else None,
            'group':          self.group_id,
            'z':              self.z_value,
        }

    def apply_config(self, config: dict):
        if not config:
            return
        dr_run_id = config.get('dr_run_id')
        if dr_run_id:
            idx = self.dr_combo.findData(dr_run_id)
            if idx >= 0:
                self.dr_combo.setCurrentIndex(idx)   # → _on_dr_run_changed rebuilds sample_combo
        cluster_run_id = config.get('cluster_run_id')
        if cluster_run_id:
            idx = self.cluster_run_combo.findData(cluster_run_id)
            if idx >= 0:
                self.cluster_run_combo.setCurrentIndex(idx)
        # Restore sample selection. Must happen AFTER dr_run_id is set above —
        # changing the DR run rebuilds sample_combo's contents via
        # _on_dr_run_changed, which only preserves the in-session selection,
        # not a persisted one.
        sample_rel = config.get('sample')
        if sample_rel:
            for i in range(self.sample_combo.count()):
                if self.sample_combo.itemData(i) == sample_rel:
                    self.sample_combo.setCurrentIndex(i)
                    break
        if config.get('colour_mode') and self.colour_mode_combo.findText(config['colour_mode']) >= 0:
            self.colour_mode_combo.setCurrentText(config['colour_mode'])
        if config.get('marker_ch'):
            idx = self.marker_combo.findData(config['marker_ch'])
            if idx >= 0:
                self.marker_combo.setCurrentIndex(idx)
        if 'show_grid' in config:
            self._show_grid.setChecked(config['show_grid'])
        if 'show_ticks' in config:
            self._show_ticks.setChecked(config['show_ticks'])
        if 'show_axis_labels' in config:
            self._show_axis_labels.setChecked(config['show_axis_labels'])
        if 'title_font' in config:
            self._legend_font_spin.setValue(int(config['title_font']))
        if 'axis_font' in config:
            self._axis_font_spin.setValue(int(config['axis_font']))
        if 'label_color' in config:
            # 'white' / 'black' were the automatic defaults before Auto existed
            colour = config['label_color']
            self._label_color = None if colour in (None, 'auto', 'white', 'black') else colour
            self._update_label_color_btn()
        if config.get('legend_pos') in LEGEND_POSITIONS:
            self._legend_pos_combo.setCurrentText(config['legend_pos'])
        if 'legend_cols' in config:
            self._legend_cols_spin.setValue(int(config['legend_cols']))
        if 'plot_w' in config or 'plot_h' in config:
            self._applying_size = True
            try:
                self._plot_w_spin.setValue(float(config.get(
                    'plot_w', self._plot_w_spin.value() / MM_PER_INCH)) * MM_PER_INCH)
                self._plot_h_spin.setValue(float(config.get(
                    'plot_h', self._plot_h_spin.value() / MM_PER_INCH)) * MM_PER_INCH)
            finally:
                self._applying_size = False
        if 'aspect_locked' in config:
            self._aspect_check.setChecked(bool(config['aspect_locked']))
        self._aspect_ratio = self._plot_w_spin.value() / self._plot_h_spin.value()
            # setValue's valueChanged already triggers _on_plot_dims_changed,
            # which calls _schedule_refresh() itself -- no need to also
            # fall through to the refresh call below for this one.
        self._schedule_refresh()

    def _copy_display_config(self):
        self.state.display_clipboard = self.get_display_config()
        self.wand_btn.setToolTip("Copied! Click 📋 on another plot to paste.")

    def _paste_display_config(self):
        if self.state.display_clipboard:
            self.apply_config(self.state.display_clipboard)
        else:
            QMessageBox.information(self, "Paste", "Nothing copied yet. Use 🪄 on another plot first.")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def get_figure(self):
        """Return the current matplotlib Figure (for PDF export)."""
        return self._figure

    def _export_png(self):
        """Export this plot (with its separate legend) as PNG."""
        self.workspace.export_plot_png(self)



# ---------------------------------------------------------------------------

_annoy_checked = False


def _check_annoy_build():
    """
    Raise a clear error if the installed Annoy returns no neighbours.

    PaCMAP finds neighbours with Annoy, and an Annoy compiled with
    -ffast-math (its default on non-Windows platforms) can return only the
    queried item, which PaCMAP then reports as an array-shape error.  The
    check builds a tiny index once per session.
    """
    global _annoy_checked
    if _annoy_checked:
        return
    from annoy import AnnoyIndex
    points = np.random.default_rng(0).normal(size=(200, 5)).astype(np.float32)
    index = AnnoyIndex(5, 'euclidean')
    for i, point in enumerate(points):
        index.add_item(i, point)
    index.build(10)
    if len(index.get_nns_by_item(0, 20)) < 20:
        raise RuntimeError(
            "The installed Annoy library returns no nearest neighbours, so PaCMAP "
            "cannot run.  Rebuild it without -ffast-math: run 'python "
            "install_annoy.py' from the Honeychrome source folder.")
    _annoy_checked = True


def _array_summary(data) -> str:
    """One-line description of an event array for diagnostic logging."""
    arr = np.asarray(data)
    if arr.size == 0:
        return f"shape={arr.shape} (empty)"
    finite = np.isfinite(arr)
    if not finite.any():
        return f"shape={arr.shape} dtype={arr.dtype} all values non-finite"
    unique_rows = len(np.unique(arr, axis=0)) if arr.ndim == 2 else -1
    return (f"shape={arr.shape} dtype={arr.dtype} non-finite={int(arr.size - finite.sum())} "
            f"unique rows={unique_rows} min={float(arr[finite].min()):.6g} "
            f"max={float(arr[finite].max()):.6g}")


# Largest Annoy tree PaCMAP keeps in memory; above this it is rebuilt for
# every transform (slower, no extra memory).
PACMAP_TREE_MAX_BYTES = 5e9


def _pacmap_tree_bytes(n_events: int, n_channels: int) -> float:
    """Approximate size of PaCMAP's Annoy tree (20 trees, Euclidean): about
    7.2 nodes per event of (16 + 4 x channels) bytes, measured at 480,000
    events x 7 channels (153 MB)."""
    return 7.2 * n_events * (16 + 4 * n_channels)


class _PaCMAPWrapper:
    """
    Thin wrapper around a fitted PaCMAP reducer.

    With the Annoy tree kept (save_tree=True), transform() needs nothing
    else.  A tree too large to keep (_pacmap_tree_bytes) is rebuilt from
    the training data on every call, which PaCMAP takes as `basis`.

    transform() returns a plain ndarray — not a tuple.
    """
    def __init__(self, reducer, training_data: np.ndarray | None):
        self._reducer = reducer
        self._training_data = training_data

    @property
    def embedding_(self) -> np.ndarray:
        """Embedding of the training data, in training-pool row order."""
        return self._reducer.embedding_

    def transform(self, new_data: np.ndarray) -> np.ndarray:
        try:
            with drc_jit.numba_section():
                if self._training_data is None:
                    return self._reducer.transform(new_data)
                return self._reducer.transform(new_data, basis=self._training_data)
        except ValueError as exc:
            _log.error("PaCMAP transform failed (%s): %s", exc, _array_summary(new_data))
            raise


# ---------------------------------------------------------------------------
# DR background worker
# ---------------------------------------------------------------------------


class _RunCancelled(Exception):
    """Raised inside a DR fit, at a point where it can stop, after Cancel."""


class _UMAPTqdmHook:
    """
    Drop-in tqdm-compatible object for UMAP's ``tqdm_kwds`` interface.

    UMAP calls ``tqdm.tqdm(range(n_epochs))`` internally and iterates over it.
    By passing an instance of this class as ``tqdm_kwds={'tqdm_class': hook}``
    we intercept each epoch update and forward it as a ``(current, total)``
    pair to a caller-supplied callback — without any tqdm dependency.

    total_override: the user-configured n_epochs value (from the UI
    spinbox). UMAP's own tqdm call doesn't reliably hand back a total that
    matches this -- previously n_epochs was left as an unused
    n_epochs_total parameter on _run_umap instead of actually being wired
    through, so the bar's denominator came only from whatever UMAP itself
    reported. This class now always reports the real configured epoch
    count instead.

    Usage::

        hook = _UMAPTqdmHook(callback=lambda cur, tot: ..., total_override=500)
        umap_lib.UMAP(..., tqdm_kwds={'tqdm_class': hook}).fit(data)
    """

    def __init__(self, callback, total_override: int | None = None,
                 should_cancel=None):
        self._cb = callback          # callable(current: int, total: int)
        self._total_override = total_override
        self._should_cancel = should_cancel   # callable() -> bool; raises _RunCancelled

    # tqdm is instantiated as tqdm_class(iterable, **kw); we capture total.
    def __call__(self, iterable=None, total=None, **_kw):
        eff_total = self._total_override if self._total_override else (total or 0)
        return self._Iter(iterable, eff_total, self._cb, self._should_cancel)

    class _Iter:
        def __init__(self, iterable, total, cb, should_cancel=None):
            self._it   = iter(iterable) if iterable is not None else iter([])
            self._total = total or 0
            self._n    = 0
            self._cb   = cb
            self._should_cancel = should_cancel

        def _check_cancel(self):
            if self._should_cancel is not None and self._should_cancel():
                raise _RunCancelled()

        def __iter__(self):
            return self

        def __next__(self):
            self._check_cancel()
            val = next(self._it)      # raises StopIteration when done
            self._n += 1
            self._cb(self._n, self._total)
            return val

        # tqdm interface stubs so UMAP doesn't fail on attribute access
        def update(self, n=1):
            self._check_cancel()
            self._n += n
            self._cb(self._n, self._total)

        def set_postfix(self, *_, **__): pass
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass


class _DrWorker(QThread):
    """
    Runs DR training and embedding in a background thread so the UI stays
    responsive.  Progress messages arrive via the *progress* signal;
    completion (success or error) via *finished*.

    Parameters
    ----------
    task : str
        'train' or 'apply'
    plugin : PluginWidget
        The owning widget; worker calls its computation methods directly.
    algo : str
        Algorithm name ('UMAP', 'tSNE', 'PaCMAP').
    params : dict
        Hyperparameters for the chosen algorithm.
    training_only : bool
        If True (default for 'apply' triggered automatically after training),
        only embed the training samples.  The user can later click
        "Apply to All Samples" to project the full sample set.
    """

    progress       = Signal(str)        # status message
    finished       = Signal(bool, str)  # (success, error_message)
    progress_value = Signal(int, int)   # (current, total) — 0,0 = indeterminate

    def __init__(self, task: str, plugin, algo: str, params: dict,
                 training_only: bool = True, parent=None, af_state=None):
        super().__init__(parent)
        self._task = task
        self._plugin = plugin
        self._algo = algo
        self._params = params
        self._training_only = training_only
        self._cancelled = False
        # Snapshot captured on the main thread before start() — see
        # drc_pipeline.apply_unmixing_af_aware() docstring for why this
        # must not be read live off the controller from this thread.
        self._af_state = af_state
        # Measured stage times and event counts, for drc_time_estimate's
        # per-machine calibration: 'io' (read, unmix and gate files not yet
        # cached; events = file events read), 'fit' (pool events), 'embed'
        # (events transformed).  jit_included: the run compiled numba code.
        self.stage_seconds = {'io': 0.0, 'fit': 0.0, 'index': 0.0, 'embed': 0.0}
        self.stage_events = {'io': 0, 'fit': 0, 'index': 0, 'embed': 0}
        self.embed_calls = 0
        self.jit_included = False

    @property
    def algo(self) -> str:
        return self._algo

    @property
    def task(self) -> str:
        return self._task

    def detach(self):
        """Cancel and disconnect every signal, so a step that cannot be
        interrupted may finish in the background without touching the UI."""
        self._cancelled = True
        for signal in (self.progress, self.finished, self.progress_value):
            try:
                signal.disconnect()
            except (RuntimeError, TypeError):
                pass

    def run(self):
        try:
            if self._cancelled:
                return
            if self._task == 'train':
                self._do_train()
            elif self._task == 'apply':
                self.jit_included = self._algo == 'UMAP' and not (
                    drc_jit.is_warm('UMAP', True) or drc_jit.is_warm('UMAP', False))
                self._do_apply(training_only=self._training_only)
        except _RunCancelled:
            self.finished.emit(False, "Cancelled.")
        except Exception as exc:
            traceback.print_exc()
            self.finished.emit(False, str(exc))

    def _emit(self, msg: str):
        self.progress.emit(msg)

    def _emit_progress(self, current: int, total: int):
        """Emit a deterministic progress update (0,0 = indeterminate)."""
        self.progress_value.emit(current, total)

    def _timed_load(self, rel_paths) -> None:
        """Load (read, unmix, gate) the samples not yet cached, timing it as 'io'."""
        seconds, events = drc_pipeline.timed_preload(
            self._plugin.controller, self._plugin.state, rel_paths,
            af_state=self._af_state, should_stop=lambda: self._cancelled)
        self.stage_seconds['io'] += seconds
        self.stage_events['io'] += events

    def _do_train(self):
        plugin = self._plugin
        algo = self._algo

        self._timed_load(plugin.state.training_sample_ids)
        training_data, selections = plugin._load_training_data(af_state=self._af_state)
        if training_data is None:
            self.finished.emit(False, "No training data could be loaded.")
            return

        if self._cancelled:
            self.finished.emit(False, "Cancelled.")
            return

        self._emit(f"Training {algo} on {len(training_data):,} events …")
        # Signal indeterminate progress while setting up
        self._emit_progress(0, 0)

        self.jit_included = (algo == 'UMAP' and not drc_jit.is_warm(
            'UMAP', parallel=_umap_parallel(self._params)))
        fit_start = time.perf_counter()
        if algo == 'UMAP':
            # UMAP supports a tqdm hook — gives us per-epoch progress.
            # total_override pins the bar's denominator to the actual
            # configured n_epochs, instead of whatever total UMAP's own
            # tqdm call happens to report.
            n_epochs = self._params.get('n_epochs', 200)
            hook = _UMAPTqdmHook(callback=self._emit_progress, total_override=n_epochs,
                                 should_cancel=lambda: self._cancelled)
            reducer = plugin._run_umap(self._params, training_data,
                                       progress_hook=hook,
                                       should_cancel=lambda: self._cancelled)
        elif algo == 'tSNE':
            reducer = plugin._run_opentsne(self._params, training_data,
                                           should_cancel=lambda: self._cancelled)
        elif algo == 'PaCMAP':
            reducer = plugin._run_pacmap(self._params, training_data)
        else:
            self.finished.emit(False, f"Unknown algorithm: {algo}")
            return
        self.stage_seconds['fit'] = time.perf_counter() - fit_start
        self.stage_events['fit'] = len(training_data)

        if self._cancelled:
            self.finished.emit(False, "Cancelled.")
            return

        plugin.state.trained_reducers[algo] = reducer
        plugin.state.dr_status[algo] = 'done'
        plugin.state.dr_timestamps[algo] = datetime.now().isoformat(timespec='seconds')
        # Fresh model → any previously embedded samples for this algo are
        # from a different training set and no longer meaningful. Clear
        # before repopulating, or leftover keys from an earlier (possibly
        # larger) sample set inflate the "embeddings: N sample(s)" count.
        plugin.state.embeddings[algo] = {}
        plugin.state.embedding_rows[algo] = {}

        self._emit(f"{algo} training complete.  Embedding training samples …")
        self._do_apply(training_only=True,
                       pool=(selections, _fit_embedding(reducer, len(training_data))))

    def _do_apply(self, training_only: bool, pool=None):
        """
        Embed samples with the trained model.

        training_only: embed the training samples (after training), else
        every sample not yet embedded by this model ('Apply to All Samples').
        pool: (selections, fit_embedding) from training -- the events of the
        training pool take their position from the fit itself, and only the
        rest of each sample's events are transformed.
        """
        plugin = self._plugin
        algo = self._algo
        reducer = plugin.state.trained_reducers.get(algo)
        if reducer is None:
            self.finished.emit(False, f"No trained {algo} model found.")
            return

        raw_subdir = plugin.controller.experiment.settings['raw']['raw_samples_subdirectory']
        all_sample_keys = []
        for abs_key in plugin.controller.experiment.samples.get('all_samples', {}):
            try:
                rel = str(Path(abs_key).relative_to(raw_subdir))
            except ValueError:
                rel = abs_key
            if training_only and rel not in plugin.state.training_sample_ids:
                continue
            all_sample_keys.append((abs_key, rel))

        for store in (plugin.state.embeddings, plugin.state.embedding_rows):
            store.setdefault(algo, {})
        embedded = plugin.state.embeddings[algo]
        if not training_only:
            # Samples this model has already embedded keep their positions
            all_sample_keys = [(a, r) for a, r in all_sample_keys if r not in embedded]

        from_fit = {}
        if pool is not None:
            selections, fit_embedding = pool
            if fit_embedding is not None:
                offset = 0
                for rel, idx in selections:
                    from_fit[rel] = (idx, fit_embedding[offset:offset + len(idx)])
                    offset += len(idx)

        self._timed_load([rel for _abs, rel in all_sample_keys])

        n_total = len(all_sample_keys)
        first_error = ''
        for i, (abs_key, rel_path) in enumerate(all_sample_keys):
            if self._cancelled:
                self.finished.emit(False, "Cancelled.")
                return
            self._emit(f"  Embedding {Path(rel_path).name} …")
            self._emit_progress(i, n_total)
            sample_data = plugin._get_sample_data(rel_path, algo, af_state=self._af_state)
            rows = drc_pipeline.load_sample_rows(plugin.controller, plugin.state, rel_path,
                                                 af_state=self._af_state)
            if sample_data is None or rows is None:
                continue
            try:
                emb = np.empty((len(sample_data), 2), dtype=np.float32)
                todo = np.ones(len(sample_data), dtype=bool)
                if rel_path in from_fit:
                    idx, fit_rows = from_fit[rel_path]
                    emb[idx] = fit_rows
                    todo[idx] = False
                if todo.any():
                    if algo == 'UMAP' and drc_umap_index.needs_index(reducer):
                        self._emit("  Building the hnswlib search index for embedding …")
                        start = time.perf_counter()
                        if drc_umap_index.attach_search_index(reducer, _umap_threads(reducer)):
                            self.stage_seconds['index'] += time.perf_counter() - start
                            self.stage_events['index'] += len(reducer._raw_data)
                    start = time.perf_counter()
                    emb[todo] = plugin._dr_transform(algo, reducer, sample_data[todo])
                    self.stage_seconds['embed'] += time.perf_counter() - start
                    self.stage_events['embed'] += int(todo.sum())
                    self.embed_calls += 1
                if self._cancelled:
                    self.finished.emit(False, "Cancelled.")
                    return
                embedded[rel_path] = emb
                plugin.state.embedding_rows[algo][rel_path] = rows
            except Exception as e:
                self._emit(f"    Could not embed {rel_path}: {e}")
                first_error = first_error or f"{rel_path}: {e}"

        n = len(embedded)
        if n == 0 and n_total > 0:
            self.finished.emit(
                False, f"No samples could be embedded with {algo}.  {first_error}".strip())
            return
        label = "training " if training_only else ""
        self._emit(f"Embeddings complete: {n} {label}sample(s) projected.")
        self.finished.emit(True, "")


def _umap_parallel(params: dict) -> bool:
    """True when UMAP runs without a fixed seed (multi-threaded numba path)."""
    return params.get('n_jobs', 1) != 1


def _umap_threads(reducer) -> int:
    """Threads a fitted UMAP was set to use (1 keeps its search index
    deterministic, as its fixed seed keeps the fit)."""
    n_jobs = getattr(reducer, 'n_jobs', 1)
    return (_os.cpu_count() or 1) if n_jobs in (-1, None) else max(int(n_jobs), 1)


def _complete_rows(labels: dict, rows: dict) -> dict | None:
    """Event keys of every labelled sample, or None (archived as a run without
    keys) when any sample has none, e.g. clustering an embedding made before
    event keys were recorded."""
    out = {rel: rows.get(rel) for rel in labels}
    if any(r is None for r in out.values()):
        return None
    return out


def _fit_embedding(reducer, n_pool: int) -> np.ndarray | None:
    """The fit's own 2-D layout of the training pool, in pool row order, or
    None when the reducer does not expose one of the expected shape."""
    emb = getattr(reducer, 'embedding_', None)
    if emb is None and isinstance(reducer, np.ndarray):
        emb = reducer          # openTSNE's TSNEEmbedding is the embedding
    if emb is None:
        return None
    emb = np.asarray(emb, dtype=np.float32)
    if emb.shape != (n_pool, 2):
        _log.warning("fit embedding has shape %s, expected (%d, 2); transforming "
                     "every event instead", emb.shape, n_pool)
        return None
    return emb


class _RunEstimateWorker(QThread):
    """
    Counts the events of a DR or clustering run and predicts its run time
    (drc_time_estimate.estimate_run), timing a small probe only when this
    machine has no calibration for the algorithm yet, so the UI stays
    responsive.  Emits *estimated* with a RunTimeEstimate; if counting
    itself fails, the estimate is empty and the run goes ahead to report
    its own error.
    """

    estimated = Signal(object)   # RunTimeEstimate

    @property
    def kind(self) -> str:
        return self._kind

    def detach(self):
        """Stop delivering the estimate; the probe finishes in the background."""
        try:
            self.estimated.disconnect()
        except (RuntimeError, TypeError):
            pass

    def __init__(self, controller, state, kind: str, algo: str, params: dict,
                 calibration: dict, af_state=None, dr_probe_fn=None,
                 jit_warm: bool = True, parent=None):
        super().__init__(parent)
        self._controller = controller
        self._state = state
        self._kind = kind
        self._algo = algo
        self._params = params
        self._calibration = calibration
        self._dr_probe_fn = dr_probe_fn
        self._jit_warm = jit_warm
        # Snapshot captured on the main thread — see
        # drc_pipeline.apply_unmixing_af_aware() docstring.
        self._af_state = af_state

    def run(self):
        try:
            estimate = drc_time_estimate.estimate_run(
                self._controller, self._state, self._kind, self._algo,
                self._params, self._calibration, af_state=self._af_state,
                dr_probe_fn=self._dr_probe_fn, jit_warm=self._jit_warm)
        except Exception:
            traceback.print_exc()
            estimate = drc_time_estimate.RunTimeEstimate(kind=self._kind, algo=self._algo)
        self.estimated.emit(estimate)


class _ClusterWorker(QThread):
    """
    Runs a clustering fit off the UI thread.  *fn* returns the run's
    staged result (drc_clustering.StagedResult), kept in *result*; the plugin
    commits it to the state only if the run was not cancelled.  Cancelling
    takes effect at the next progress report (fn's progress callback raises
    _RunCancelled); a step that cannot be interrupted finishes in the
    background with its result discarded.
    """

    finished = Signal(bool, str)   # (success, error_message)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn
        self._cancelled = False
        self.result = None

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def detach(self):
        """Cancel and disconnect, so the UI is free while this winds down."""
        self._cancelled = True
        try:
            self.finished.disconnect()
        except (RuntimeError, TypeError):
            pass

    def run(self):
        try:
            self.result = self._fn()
            if self._cancelled:
                self.result = None      # cancelled too late; drop the arrays
            self.finished.emit(True, '')
        except _RunCancelled:
            self.finished.emit(False, 'Cancelled.')
        except Exception as e:
            traceback.print_exc()
            self.finished.emit(False, str(e))


# ---------------------------------------------------------------------------
# Cluster Annotation tab
# ---------------------------------------------------------------------------

class _SquareContainer(QWidget):
    """
    Keeps a single child widget square and centred, resizing it manually
    on every container resize rather than fighting a layout (Qt layouts
    can't express "always square" directly).
    """
    def __init__(self, child: QWidget, parent=None):
        super().__init__(parent)
        self._child = child
        child.setParent(self)

    def resizeEvent(self, event):
        # Skip the very first resize(s) before this widget has real
        # on-screen geometry (both 0 during initial construction, prior
        # to being shown/laid out) -- handing the canvas a literal 0x0
        # geometry here is what made matplotlib's constrained-layout
        # engine warn about axes collapsing to zero.
        if self.width() <= 0 or self.height() <= 0:
            super().resizeEvent(event)
            return
        side = max(0, min(self.width(), self.height()))
        x = (self.width() - side) // 2
        y = (self.height() - side) // 2
        self._child.setGeometry(x, y, side, side)
        super().resizeEvent(event)


class _WrappingLegendWidget(QWidget):
    """
    Cluster-swatch legend for the Cluster Map panel. Entries fill
    top-to-bottom then wrap into a new column, capped at MAX_ROWS_PER_COL
    rows per column -- so long cluster lists (>20) always spread across
    columns rather than relying on the widget's on-screen height, which
    is 0/unreliable the first time set_entries() runs (e.g. inside a
    not-yet-shown pop-out QDialog, before dlg.show()), previously
    collapsing every entry into its own column instead of wrapping.

    sizeHint()/minimumSizeHint() reflect the ACTUAL flowed size (computed
    in _reflow, cached in _cached_size) rather than a fixed placeholder --
    the containing QScrollArea has setWidgetResizable(True), so a
    hardcoded small hint was making it shrink this widget down to the
    viewport width, squeezing every column's swatch+label into that
    width and clipping the name labels.
    """
    ROW_HEIGHT = 20
    MAX_ROWS_PER_COL = 20

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries: list[QWidget] = []
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(2, 2, 2, 2)
        self._grid.setSpacing(4)
        self._grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._cached_size = QSize(90, self.ROW_HEIGHT)

    def set_entries(self, entries: list[QWidget]):
        """Replace the displayed rows and re-flow into columns."""
        for w in self._entries:
            self._grid.removeWidget(w)
            w.deleteLater()
        self._entries = entries
        for w in entries:
            w.setParent(self)
        self._reflow()

    def sizeHint(self):
        return self._cached_size

    def minimumSizeHint(self):
        return self._cached_size

    def resizeEvent(self, event):
        self._reflow()
        super().resizeEvent(event)

    def _reflow(self):
        if not self._entries:
            return
        for w in self._entries:
            self._grid.removeWidget(w)
        n = len(self._entries)
        # Column count now derives from entry count alone (capped at
        # MAX_ROWS_PER_COL rows/column), not from self.height() -- makes
        # the layout deterministic regardless of whether the widget has
        # been shown/resized yet.
        rows_per_col = max(1, min(self.MAX_ROWS_PER_COL, n))
        ncols = (n + rows_per_col - 1) // rows_per_col
        col_widths = [0] * ncols
        for i, w in enumerate(self._entries):
            col, row = divmod(i, rows_per_col)
            self._grid.addWidget(w, row, col)
            # sizeHint here needs the label's natural (unclipped) width --
            # ensure the row widget hasn't been squashed by a previous
            # reflow before we measure it.
            w.setMinimumWidth(0)
            col_widths[col] = max(col_widths[col], w.sizeHint().width())

        margins = self._grid.contentsMargins()
        spacing = self._grid.spacing()
        total_w = (margins.left() + margins.right()
                   + sum(col_widths) + spacing * max(0, ncols - 1))
        total_h = (margins.top() + margins.bottom()
                   + rows_per_col * self.ROW_HEIGHT)
        new_size = QSize(max(60, total_w), max(self.ROW_HEIGHT, total_h))
        if new_size != self._cached_size:
            self._cached_size = new_size
            self.updateGeometry()

class _MarkerSummaryWorker(QThread):
    """
    Pools per-cluster marker values (disk I/O + unmixing, via
    the owning ClusterAnnotationTab's own _pool_violin_data) and builds
    the heatmap/ridgeline Figures on a background thread, so recomputing
    -- or just switching back to an already-pooled run -- never blocks
    the UI. Figure objects themselves are plain matplotlib (no Qt), so
    building them off the main thread is safe; only the canvas
    construction/draw() in _apply_marker_summary_figures touches Qt and
    must stay on the main thread (see the finished signal's connection).

    If *pooled* is already known (switching back to a cached run), pass
    it in directly and the pooling step is skipped.
    """
    finished = Signal(bool, str, dict)

    def __init__(self, plugin, cl_run: dict, channels: list[str],
                cluster_order: list[int], af_state, pooled, live_mode=None, parent=None):
        super().__init__(parent)
        self._plugin = plugin
        self._live_mode = live_mode
        self._cl_run = cl_run
        self._channels = channels
        self._cluster_order = cluster_order
        self._af_state = af_state
        self._pooled = pooled

    def run(self):
        import time
        t_start = time.perf_counter()

        def _lap(prev: float, label: str) -> float:
            now = time.perf_counter()
            _log.info("marker summary timing: %-22s %6.2fs", label, now - prev)
            return now

        try:
            plugin = self._plugin
            pooled = self._pooled
            t = t_start
            if pooled is None:
                pooled = plugin._pool_violin_data(
                    self._cl_run, self._channels, af_state=self._af_state,
                    live_mode=self._live_mode,
                )
                t = _lap(t, "pool_violin_data")
            else:
                _log.info("marker summary timing: %-22s %6s  (reused, run switch)",
                          "pool_violin_data", "--")
            names_map = self._cl_run.get('names', {})
            colors_map = self._cl_run.get('colors', {})
            transformed = plugin._transform_and_pool_matrix_values(
                pooled, self._channels, self._cluster_order,
            )
            t = _lap(t, "transform_and_pool")
            mat = plugin._compute_marker_heatmap_matrix(
                transformed, self._channels, self._cluster_order,
            )
            t = _lap(t, "compute_matrix")
            main_fig, col_fig, row_fig = plugin._make_marker_cluster_heatmap_figures(
                mat, self._channels, self._cluster_order, names_map,
            )
            t = _lap(t, "heatmap_figures")
            cbar_fig = plugin._make_marker_cluster_colorbar_figure(mat)
            t = _lap(t, "colorbar_figure")
            ridge_fig = plugin._make_marker_ridgeline_figure(
                transformed, self._channels, self._cluster_order, names_map, colors_map,
            )
            t = _lap(t, "ridgeline_figure")
            _log.info("marker summary timing: %-22s %6.2fs", "TOTAL", t - t_start)
            self.finished.emit(True, '', {
                'pooled': pooled,
                'channels': self._channels,
                'cluster_order': self._cluster_order,
                'names_map': names_map,
                'colors_map': colors_map,
                'main_fig': main_fig, 'col_fig': col_fig, 'row_fig': row_fig,
                'cbar_fig': cbar_fig, 'ridge_fig': ridge_fig,
            })
        except Exception as exc:
            traceback.print_exc()
            self.finished.emit(False, str(exc), {})


class _RunHydrateWorker(QThread):
    """
    Reads a clustering/DR run's heavy payload (see
    drc_run_archive.hydrate_run) on a background thread. The first time a
    run is selected in a session, its manifest entry only carries
    metadata (drc_run_archive.load_manifest_entries loads lazily) --
    hydrate_run()'s disk read used to happen inline from refresh() on the
    main thread, which is what froze the UI the first time the Cluster
    Annotation tab opened in an experiment. hydrate_run() mutates each
    entry dict in place and does no Qt widget access, so it's safe to
    run here.
    """
    finished = Signal()

    def __init__(self, controller, entries: list, parent=None):
        super().__init__(parent)
        self._controller = controller
        self._entries = entries

    def run(self):
        for entry in self._entries:
            try:
                drc_run_archive.hydrate_run(self._controller, entry)
            except Exception:
                traceback.print_exc()
        self.finished.emit()


class ClusterAnnotationTab(QWidget):
    """
    Tab — Cluster Annotation
    -------------------------------------
    Four coordinated elements for a SELECTED clustering run:
      1. Per-marker violin plots, one per selected channel, cluster on the
         x-axis. Values come from drc_pipeline.load_sample_marker_values(),
         pooled across the run's own training samples and split by that
         run's own per-sample label arrays — never the "currently active"
         globals, so switching runs here never touches Workspace state.
      2. Cluster map — a DR scatter plot coloured by cluster, via its own
         DR-run selector. Rendering, legend editing, and the
         Honeycluster compatibility warning all delegate to drc_scatter,
         shared with Workspace's PlotCard so both render identically.
      3. Cluster label table — one row per cluster: editable name, colour
         swatch (double-click to change), event count, % of total. Both
         this table and the map legend rename/recolour through the same
         drc_scatter helpers PlotCard uses, so edits made from either UI
         (or from Workspace) are always the SAME per-run 'names'/'colors'
         payload — there is no separate copy to fall out of sync.
      4. Cluster ID Suggestions (drc_cluster_id.py) — two more
         label-table columns computed from the SAME checked channels as
         Panel 1: an auto-generated MEM label (safe one-click adoption,
         double-click the cell or "Adopt All MEM Labels") and a suggested
         cell type scored against a bundled marker-signature database
         (display-only — a biological claim, not a statistic, so adopting
         one means retyping it into Name yourself).
    """

    def __init__(self, state: PipelineState, bus, controller, parent=None):
        super().__init__(parent)
        self.state = state
        self.bus = bus
        self.controller = controller
        # {run_id: {'pooled': {ch: {cl_id: [arrays]}}, 'channels': [...],
        #  'names_map': {...}}} — cached per clustering run so
        # switching FlowSOM -> Leiden -> FlowSOM redraws the FlowSOM
        # violin instantly instead of reverting to the placeholder.
        self._violin_cache: dict[str, dict] = {}
        # Debounced auto-recompute — avoids
        # recomputing on every single checkbox click while the user is
        # still adjusting the checked set.
        self._violin_recompute_timer = QTimer(self)
        self._violin_recompute_timer.setSingleShot(True)
        self._violin_recompute_timer.setInterval(400)
        self._violin_recompute_timer.timeout.connect(self._recompute_violins)
        # Cluster ID suggestion cache, mirrored into the run
        # archive (drc_run_archive.archive_clustering_run's 'mem_labels'/
        # 'cell_type_suggestions' payload fields -- see
        # update_cluster_id_suggestions). These three attributes are still
        # the tab's live working copy; _on_run_changed restores them from
        # the newly-selected run's persisted values (if any) instead of
        # always clearing to empty, and _on_cluster_id_finished writes
        # back through to the archive after a fresh computation.
        self._mem_labels: dict[int, str] = {}
        self._cell_type_df = None
        self._suggestions_run_id: str | None = None
        self._cluster_id_worker = None
        # In-flight _RunHydrateWorker, if any -- see
        # _start_hydrate_worker_if_needed.
        self._hydrate_worker = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Same reasoning as GroupsStatsTab: the outer plugin QScrollArea
        # uses setWidgetResizable(True), which clamps this tab to the
        # viewport instead of letting it grow — so the violin/map/table
        # splitter below needs its own scroll area rather than fighting
        # that clamp for space.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer_scroll = QScrollArea()
        outer_scroll.setWidgetResizable(True)
        outer_scroll.setFrameShape(QFrame.NoFrame)
        outer_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(outer_scroll)

        content = QWidget()
        outer_scroll.setWidget(content)

        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(8)

        self.help_widget = HelpToggleWidget(
            text=drc_help_texts.cluster_annotation_tab_help_text
        )
        content_layout.addWidget(self.help_widget)

        # ---- Clustering run selector (shared by both sub-tabs below) ----
        run_row = QHBoxLayout()
        run_row.addWidget(QLabel("Clustering run:"))
        self.run_combo = QComboBox()
        self.run_combo.setMinimumWidth(280)
        self.run_combo.setToolTip("Archived clustering run to annotate.")
        self.run_combo.currentIndexChanged.connect(self._on_run_changed)
        run_row.addWidget(self.run_combo)
        run_row.addStretch()
        content_layout.addLayout(run_row)

        self.annotation_sub_tabs = QTabWidget()
        self.annotation_sub_tabs.setDocumentMode(True)
        self.annotation_sub_tabs.currentChanged.connect(self._on_annotation_sub_tab_changed)
        content_layout.addWidget(self.annotation_sub_tabs, stretch=1)

        annotation_page = QWidget()
        annotation_page_layout = QVBoxLayout(annotation_page)
        annotation_page_layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Vertical)
        annotation_page_layout.addWidget(splitter)
        self.annotation_sub_tabs.addTab(annotation_page, "Annotation")

        # ============================================================
        # Panel 1 — Per-marker violin plots
        # ============================================================
        violin_box = QGroupBox("Per-Marker Violin Plots")
        violin_layout = QVBoxLayout(violin_box)

        ch_row = QHBoxLayout()
        ch_row.addWidget(QLabel("Channels:"))

        # Grid of individually-labelled checkboxes -- same pattern as
        # GroupsStatsTab's "Marker Roles -- MFI Testing" list (see
        # _populate_marker_roles_list) -- instead of a QListWidget, so full
        # channel labels are always readable rather than wrapped/elided
        # list items.
        self._channel_scroll = QScrollArea()
        self._channel_scroll.setWidgetResizable(True)
        self._channel_scroll.setMinimumHeight(70)
        self._channel_scroll.setMaximumHeight(140)
        self.channel_list_widget = QWidget()
        self.channel_grid = QGridLayout(self.channel_list_widget)
        self.channel_grid.setSpacing(4)
        self.channel_checkboxes: dict[str, QCheckBox] = {}
        self._channel_scroll.setWidget(self.channel_list_widget)
        ch_row.addWidget(self._channel_scroll, stretch=1)

        ch_btn_col = QVBoxLayout()
        self.select_all_channels_btn = QPushButton("All")
        self.select_all_channels_btn.setFixedWidth(50)
        self.select_all_channels_btn.clicked.connect(lambda: self._set_all_channels(True))
        self.select_none_channels_btn = QPushButton("None")
        self.select_none_channels_btn.setFixedWidth(50)
        self.select_none_channels_btn.clicked.connect(lambda: self._set_all_channels(False))
        ch_btn_col.addWidget(self.select_all_channels_btn)
        ch_btn_col.addWidget(self.select_none_channels_btn)
        ch_btn_col.addStretch()
        ch_row.addLayout(ch_btn_col)
        violin_layout.addLayout(ch_row)

        viewing_row = QHBoxLayout()
        viewing_row.addWidget(QLabel("Viewing:"))
        self.violin_channel_combo = QComboBox()
        self.violin_channel_combo.setMinimumWidth(200)
        self.violin_channel_combo.setToolTip(
            "Which checked channel's violin plot is currently shown."
        )
        self.violin_channel_combo.currentIndexChanged.connect(self._draw_current_violin)
        viewing_row.addWidget(self.violin_channel_combo)
        viewing_row.addStretch()

        self.plot_violins_btn = QPushButton("⟳  Recompute Violins")
        self.plot_violins_btn.setToolTip(
            "Checked channels plot automatically. Use this if underlying "
            "gate/transform data changed without touching the checkboxes."
        )
        self.plot_violins_btn.clicked.connect(self._recompute_violins)
        viewing_row.addWidget(self.plot_violins_btn)
        violin_layout.addLayout(viewing_row)

        self._violin_scroll = QScrollArea()
        self._violin_scroll.setWidgetResizable(True)
        self._violin_placeholder = QLabel(
            "Select a clustering run and channel(s), then click 'Plot Violins'."
        )
        self._violin_placeholder.setStyleSheet("color: grey; font-style: italic;")
        self._violin_placeholder.setAlignment(Qt.AlignCenter)
        self._violin_scroll.setWidget(self._violin_placeholder)
        violin_layout.addWidget(self._violin_scroll, stretch=1)

        splitter.addWidget(violin_box)

        # ============================================================
        # Panel 2 — Cluster map
        # ============================================================
        map_box = QGroupBox("Cluster Map")
        map_layout = QVBoxLayout(map_box)

        map_row = QHBoxLayout()
        map_row.addWidget(QLabel("DR run:"))
        self.dr_run_combo = QComboBox()
        self.dr_run_combo.setFixedWidth(180)
        self.dr_run_combo.setToolTip(
            "Archived DR run to map this clustering run's labels onto."
        )
        self.dr_run_combo.currentIndexChanged.connect(self._on_dr_run_changed)
        map_row.addWidget(self.dr_run_combo)
        self._compat_warning = QLabel("⚠")
        self._compat_warning.setStyleSheet("color: #d9822b; font-weight: bold;")
        self._compat_warning.setVisible(False)
        map_row.addWidget(self._compat_warning)
        map_row.addStretch()
        self.map_popout_btn = QPushButton("⤢ Pop Out")
        self.map_popout_btn.setToolTip("Open the cluster map in a larger window.")
        self.map_popout_btn.clicked.connect(self._pop_out_map)
        map_row.addWidget(self.map_popout_btn)
        map_layout.addLayout(map_row)

        map_plot_row = QSplitter(Qt.Horizontal)
        map_plot_row.setChildrenCollapsible(False)
        from matplotlib.figure import Figure
        self._map_figure = Figure(figsize=(4, 4), constrained_layout=True)
        self._map_canvas = _new_scrollable_canvas(self._map_figure)
        self._map_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._map_square = _SquareContainer(self._map_canvas)
        self._map_square.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        map_plot_row.addWidget(self._map_square)

        legend_scroll = QScrollArea()
        legend_scroll.setWidgetResizable(True)
        legend_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        legend_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        legend_scroll.setMinimumWidth(90)
        legend_scroll.setFrameShape(QFrame.NoFrame)
        self._legend_widget = _WrappingLegendWidget()
        legend_scroll.setWidget(self._legend_widget)
        self._legend_scroll = legend_scroll
        map_plot_row.addWidget(legend_scroll)

        # Plot pane absorbs any extra space; legend
        # pane starts at a rough default and gets fitted to its real
        # content the first time _rebuild_map_legend runs (see
        # _sync_legend_column_width), then stays user-controlled once
        # the handle below has been dragged.
        map_plot_row.setStretchFactor(0, 1)
        map_plot_row.setStretchFactor(1, 0)
        map_plot_row.setSizes([600, 110])
        self._legend_width_user_set = False
        map_plot_row.splitterMoved.connect(self._on_map_legend_splitter_moved)
        self._map_plot_splitter = map_plot_row

        map_layout.addWidget(map_plot_row, stretch=1)
        splitter.addWidget(map_box)

        # ============================================================
        # Panel 3 — Cluster label table
        # ============================================================
        table_box = QGroupBox("Cluster Labels")
        table_layout = QVBoxLayout(table_box)

        suggest_row = QHBoxLayout()
        suggest_row.addWidget(QLabel("MEM threshold:"))
        self.mem_threshold_spin = QDoubleSpinBox()
        self.mem_threshold_spin.setRange(0.5, 10.0)
        self.mem_threshold_spin.setSingleStep(0.5)
        self.mem_threshold_spin.setValue(2.0)
        self.mem_threshold_spin.setFixedWidth(70)
        self.mem_threshold_spin.setToolTip(
            "Minimum |MEM score| (0-10 scale) for a marker to appear in the\n"
            "generated MEM label. Lower = more markers included per cluster."
        )
        suggest_row.addWidget(self.mem_threshold_spin)

        suggest_row.addWidget(QLabel("Species:"))
        self.species_combo = QComboBox()
        self.species_combo.addItem("Human", "human")
        self.species_combo.addItem("Mouse", "mouse")
        self.species_combo.setToolTip(
            "Restricts Suggested Type to drc_cell_type_database.csv entries\n"
            "defined for this species. Human and mouse entries can define the\n"
            "same cell-type name with different markers (e.g. CD90 vs CD90.2),\n"
            "so mixing both would let wrong-species definitions compete for\n"
            "the same cluster. Has no effect on MEM Label (species-agnostic)."
        )
        suggest_row.addWidget(self.species_combo)

        self.compute_suggestions_btn = QPushButton("Compute Cluster ID Suggestions")
        self.compute_suggestions_btn.setToolTip(
            "Uses the channel set recorded for THIS clustering run at the "
            "time it was created (its Configuration tab selection back "
            "then) as the marker set for both the MEM label and the "
            "cell-type score -- not whatever is currently checked above, "
            "and not Configuration's current selection either."
        )
        self.compute_suggestions_btn.clicked.connect(self._compute_cluster_id_suggestions)
        suggest_row.addWidget(self.compute_suggestions_btn)

        self.adopt_all_mem_btn = QPushButton("Adopt All MEM Labels")
        self.adopt_all_mem_btn.setToolTip(
            "One-click: set every cluster's Name in this run to its generated\n"
            "MEM label. Duplicate labels are numbered to stay unique."
        )
        self.adopt_all_mem_btn.clicked.connect(self._adopt_all_mem_labels)
        suggest_row.addWidget(self.adopt_all_mem_btn)
        suggest_row.addStretch()
        table_layout.addLayout(suggest_row)

        self._suggestions_status = QLabel(
            "Cluster ID suggestions not yet computed for this run."
        )
        self._suggestions_status.setStyleSheet("color: grey; font-style: italic;")
        self._suggestions_status.setWordWrap(True)
        table_layout.addWidget(self._suggestions_status)

        # Compute loads + unmixes every training
        # sample from disk, which can take a while; this is the only
        # visible feedback besides the status label above.
        self._suggestions_progress = QProgressBar()
        self._suggestions_progress.setVisible(False)
        self._suggestions_progress.setTextVisible(True)
        table_layout.addWidget(self._suggestions_progress)

        self.label_table = QTableWidget(0, 6)
        self.label_table.setHorizontalHeaderLabels(
            ['Name', 'Colour', 'Events', '% of total', 'MEM Label', 'Suggested Type']
        )
        self.label_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.label_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.label_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.label_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.label_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.label_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.label_table.verticalHeader().setVisible(False)
        self.label_table.setToolTip(
            "Double-click Name to rename; double-click Colour to recolour;\n"
            "double-click MEM Label to adopt it as this cluster's Name.\n"
            "Suggested Type is informational only -- retype it into Name\n"
            "yourself if you want to adopt it."
        )
        self.label_table.itemChanged.connect(self._on_table_item_changed)
        self.label_table.cellDoubleClicked.connect(self._on_table_cell_double_clicked)
        table_layout.addWidget(self.label_table)
        splitter.addWidget(table_box)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([780, 520, 340])
        splitter.setChildrenCollapsible(False)
        splitter.setMinimumHeight(2080)

        # ============================================================
        # Sub-tab — Marker Heatmap & Ridgelines
        # ============================================================
        self._build_marker_summary_ui()

    def _build_marker_summary_ui(self):
        """
        Sub-tab 2: mean-MFI heatmap and marker ridgeline grid,
        both keyed off the SAME clustering run selected at the
        top of this tab, both cached per run_id
        so switching runs doesn't discard an already-computed summary.
        """
        self._marker_summary_cache: dict[str, dict] = {}
        # (run_id, is_dark) last actually rendered -- lets _draw_marker_summary
        # skip rebuilding both figures from scratch on every sub-tab visit
        # when nothing has changed since the last render.
        self._marker_summary_last_drawn: tuple | None = None
        # At most one _MarkerSummaryWorker exists at a time; a request that
        # arrives while it runs sets the pending flag and is replayed from
        # the finished handler.
        self._marker_summary_worker = None
        self._marker_summary_rerun_pending = False
        self._marker_summary_busy = False

        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(4, 4, 4, 4)

        btn_row = QHBoxLayout()
        self.marker_summary_recompute_btn = QPushButton("⟳  Recompute Marker Summary")
        self.marker_summary_recompute_btn.setToolTip(
            "Uses every channel currently selected in the Configuration tab."
        )
        self.marker_summary_recompute_btn.clicked.connect(self._recompute_marker_summary)
        btn_row.addWidget(self.marker_summary_recompute_btn)
        btn_row.addStretch()
        page_layout.addLayout(btn_row)

        self._summary_splitter = QSplitter(Qt.Vertical)
        page_layout.addWidget(self._summary_splitter, stretch=1)

        heatmap_box = QGroupBox("Median MFI per Cluster (Transformed)")
        self._heatmap_box = heatmap_box
        heatmap_outer_layout = QVBoxLayout(heatmap_box)

        # Frozen-header grid: (0,0) corner spacer, (0,1) marker-name
        # header (scrolls horizontally in lock-step), (1,0) cluster-name
        # header (scrolls vertically in lock-step), (1,1) the heatmap
        # itself (scrolls both ways -- the only one with real scrollbars).
        heatmap_grid = QGridLayout()
        heatmap_grid.setContentsMargins(0, 0, 0, 0)
        heatmap_grid.setSpacing(0)

        self._heatmap_corner = QWidget()
        heatmap_grid.addWidget(self._heatmap_corner, 0, 0)

        self._heatmap_col_header_scroll = _FrozenScrollArea()
        self._heatmap_col_header_scroll.setWidgetResizable(True)
        self._heatmap_col_header_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._heatmap_col_header_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._heatmap_col_header_scroll.setFrameShape(QFrame.NoFrame)
        heatmap_grid.addWidget(self._heatmap_col_header_scroll, 0, 1)

        self._heatmap_row_header_scroll = _FrozenScrollArea()
        self._heatmap_row_header_scroll.setWidgetResizable(True)
        self._heatmap_row_header_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._heatmap_row_header_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._heatmap_row_header_scroll.setFrameShape(QFrame.NoFrame)
        heatmap_grid.addWidget(self._heatmap_row_header_scroll, 1, 0)

        self._heatmap_scroll = QScrollArea()
        self._heatmap_scroll.setWidgetResizable(True)
        self._heatmap_placeholder = QLabel(
            "Select a clustering run, then click 'Recompute Marker Summary'."
        )
        self._heatmap_placeholder.setStyleSheet("color: grey; font-style: italic;")
        self._heatmap_placeholder.setAlignment(Qt.AlignCenter)
        self._heatmap_scroll.setWidget(self._heatmap_placeholder)
        heatmap_grid.addWidget(self._heatmap_scroll, 1, 1)

        heatmap_grid.setColumnStretch(1, 1)
        heatmap_grid.setRowStretch(1, 1)

        # The headers never scroll themselves (see _FrozenScrollArea) --
        # they're driven entirely by the main heatmap's own scrollbars.
        self._heatmap_scroll.horizontalScrollBar().valueChanged.connect(
            self._heatmap_col_header_scroll.horizontalScrollBar().setValue
        )
        self._heatmap_scroll.verticalScrollBar().valueChanged.connect(
            self._heatmap_row_header_scroll.verticalScrollBar().setValue
        )

        heatmap_outer_layout.addLayout(heatmap_grid)

        # Colour scale -- kept separate from the (now full-bleed, label-
        # free) heatmap figure so it doesn't need to scroll with anything.
        self._heatmap_colorbar_scroll = QScrollArea()
        self._heatmap_colorbar_scroll.setWidgetResizable(True)
        self._heatmap_colorbar_scroll.setFrameShape(QFrame.NoFrame)
        self._heatmap_colorbar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._heatmap_colorbar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._heatmap_colorbar_scroll.setFixedHeight(70)
        heatmap_outer_layout.addWidget(self._heatmap_colorbar_scroll)
        # Any leftover height in this splitter panel collapses to the
        # bottom instead of stretching the (fixed-size) heatmap's scroll
        # viewport -- that stretch was the source of the blank gap that
        # used to appear between the heatmap and the colour scale.
        heatmap_outer_layout.addStretch(1)

        self._summary_splitter.addWidget(heatmap_box)

        ridge_box = QGroupBox("Marker Ridgeline Grid")
        ridge_layout = QVBoxLayout(ridge_box)
        self._ridge_scroll = QScrollArea()
        self._ridge_scroll.setWidgetResizable(True)
        self._ridge_placeholder = QLabel(
            "Select a clustering run, then click 'Recompute Marker Summary'."
        )
        self._ridge_placeholder.setStyleSheet("color: grey; font-style: italic;")
        self._ridge_placeholder.setAlignment(Qt.AlignCenter)
        self._ridge_scroll.setWidget(self._ridge_placeholder)
        ridge_layout.addWidget(self._ridge_scroll, stretch=1)
        self._summary_splitter.addWidget(ridge_box)

        # Heatmap pane stays pinned to its own content height (set in
        # _apply_marker_summary_figures via setMaximumHeight) instead of
        # sharing growth with the ridge pane -- factor 0 means it never
        # claims extra splitter height, so the ridge grid (factor 1) is
        # the only pane that grows into newly available space.
        self._summary_splitter.setStretchFactor(0, 0)
        self._summary_splitter.setStretchFactor(1, 1)
        self._summary_splitter.setChildrenCollapsible(False)
        self._summary_splitter.setMinimumHeight(1800)

        self._marker_summary_tab_index = self.annotation_sub_tabs.addTab(
            page, "Marker Heatmap / Ridgelines"
        )

    # ------------------------------------------------------------------
    # Refresh (called on tab activation)
    # ------------------------------------------------------------------

    def refresh(self):
        _log.info("ClusterAnnotationTab.refresh: selected_channels=%d clustering_runs=%d",
                  len(self.state.selected_channels), len(self.state.clustering_runs))
        self._populate_run_combo()
        self._populate_dr_run_combo()
        self._populate_channel_list()
        # The rest of refresh() below (via _selected_cluster_run() /
        # _selected_dr_run()) needs the selected run(s)' heavy payload
        # read from disk. Do that on a background thread first if
        # it hasn't happened yet this session; _on_hydrate_finished
        # re-enters refresh() once it's ready, at which point
        # hydrate_run() is a cheap no-op and this proceeds normally.
        if self._start_hydrate_worker_if_needed():
            return
        # _populate_run_combo() above re-selects whatever run_id
        # was already active, so it never fires currentIndexChanged /
        # _on_run_changed on its own; without this call, a freshly
        # (re)activated tab (including right after opening the experiment)
        # never restores a previously-computed, already-persisted
        # suggestion for the still-selected run. See
        # _sync_suggestions_from_run's docstring.
        self._sync_suggestions_from_run(self._selected_cluster_run())
        self._update_compat_warning()
        self._redraw_map()
        self._populate_label_table()
        run_id = self.run_combo.currentData()
        cache = self._violin_cache.get(run_id)
        if cache:
            self._populate_violin_channel_combo(cache['channels'])
            self._draw_current_violin()
        if self.annotation_sub_tabs.currentIndex() == getattr(self, '_marker_summary_tab_index', -1):
            ms_cache = self._marker_summary_cache.get(run_id)
            ms_worker = getattr(self, '_marker_summary_worker', None)
            if ms_cache:
                self._draw_marker_summary()
            elif run_id is not None and not (ms_worker is not None and ms_worker.isRunning()):
                # Figures are not persisted: first view after reopening
                # the experiment rebuilds them.
                self._recompute_marker_summary()

    def _start_hydrate_worker_if_needed(self) -> bool:
        """
        Kick a background _RunHydrateWorker for the currently-selected
        clustering/DR run entries if either still needs its heavy
        payload read from disk (see drc_run_archive.hydrate_run).
        Returns True if a worker was started or one is already in
        flight -- callers should stop the rest of refresh() in that
        case and let _on_hydrate_finished re-enter it once ready.
        """
        if self._hydrate_worker is not None:
            return True
        run_id = self.run_combo.currentData()
        dr_run_id = self.dr_run_combo.currentData()
        to_hydrate = [
            entry for entry in self.state.clustering_runs
            if entry.get('run_id') == run_id and 'labels' not in entry
        ]
        to_hydrate += [
            entry for entry in self.state.dr_runs
            if entry.get('run_id') == dr_run_id and 'embeddings' not in entry
        ]
        if not to_hydrate:
            return False
        self._map_figure.clear()
        ax = self._map_figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Loading clustering run …', ha='center', va='center',
                transform=ax.transAxes, fontsize=9)
        self._map_canvas.draw_idle()
        worker = _RunHydrateWorker(self.controller, to_hydrate)
        worker.finished.connect(self._on_hydrate_finished)
        self._hydrate_worker = worker
        worker.start()
        return True

    def _on_hydrate_finished(self):
        self._hydrate_worker = None
        self.refresh()

    # ------------------------------------------------------------------
    # Run selectors
    # ------------------------------------------------------------------

    def _populate_run_combo(self):
        """Rebuild the clustering-run combo from state.clustering_runs,
        keyed by run_id (labels are user-editable)."""
        self.run_combo.blockSignals(True)
        prev_run_id = self.run_combo.currentData()
        self.run_combo.clear()
        runs = sorted(self.state.clustering_runs, key=lambda e: e.get('timestamp', ''))
        for entry in runs:
            self.run_combo.addItem(entry.get('label', ''), entry.get('run_id'))
        if self.run_combo.count() == 0:
            self.run_combo.addItem('(no clustering runs yet)', None)
        idx = self.run_combo.findData(prev_run_id) if prev_run_id else -1
        if idx >= 0:
            self.run_combo.setCurrentIndex(idx)
        else:
            self.run_combo.setCurrentIndex(self.run_combo.count() - 1)
        self.run_combo.blockSignals(False)

    def _populate_dr_run_combo(self):
        """Rebuild the DR-run combo from state.dr_runs, keyed by run_id —
        same convention as PlotCard's own dr_combo."""
        self.dr_run_combo.blockSignals(True)
        prev_run_id = self.dr_run_combo.currentData()
        self.dr_run_combo.clear()
        runs = sorted(self.state.dr_runs, key=lambda e: e.get('timestamp', ''))
        for entry in runs:
            self.dr_run_combo.addItem(entry.get('label', ''), entry.get('run_id'))
        if self.dr_run_combo.count() == 0:
            self.dr_run_combo.addItem('—', None)
        idx = self.dr_run_combo.findData(prev_run_id) if prev_run_id else -1
        if idx >= 0:
            self.dr_run_combo.setCurrentIndex(idx)
        else:
            self.dr_run_combo.setCurrentIndex(self.dr_run_combo.count() - 1)
        self.dr_run_combo.blockSignals(False)

    def _selected_cluster_run(self) -> dict | None:
        run_id = self.run_combo.currentData()
        if run_id is None:
            return None
        for entry in self.state.clustering_runs:
            if entry.get('run_id') == run_id:
                return drc_run_archive.hydrate_run(self.controller, entry)
        return None

    def _selected_dr_run(self) -> dict | None:
        run_id = self.dr_run_combo.currentData()
        if run_id is None:
            return None
        for entry in self.state.dr_runs:
            if entry.get('run_id') == run_id:
                return drc_run_archive.hydrate_run(self.controller, entry)
        return None

    def _sync_suggestions_from_run(self, cl_run):
        """
        Restore self._mem_labels / self._cell_type_df /
        self._suggestions_run_id from whatever cl_run has persisted (see
        drc_run_archive.update_cluster_id_suggestions), or clear them if
        this run has nothing saved.

        Shared by _on_run_changed (an actual combo selection change) AND
        refresh() (tab (re)activation, called on EVERY switch back into
        this plugin, including right after an experiment loads).
        refresh()'s own _populate_run_combo() re-selects the SAME run_id
        that was already selected, so Qt's currentIndexChanged never
        fires and _on_run_changed never runs on its own -- without this
        also being called from refresh(), a freshly reopened experiment
        (or simply switching back to this plugin tab) never restores
        suggestions for the still-selected run, even though they were
        safely on disk the whole time.
        """
        self._mem_labels = (cl_run.get('mem_labels') or {}) if cl_run else {}
        self._cell_type_df = cl_run.get('cell_type_suggestions') if cl_run else None
        has_saved = bool(self._mem_labels) or self._cell_type_df is not None
        self._suggestions_run_id = cl_run.get('run_id') if (cl_run and has_saved) else None
        if self._suggestions_run_id:
            self._suggestions_status.setText(
                f"Suggestions restored for {len(self._mem_labels)} cluster(s), "
                f"{len(cl_run.get('channels', []))} channel(s), "
                f"{self.species_combo.currentText()} cell types."
            )
        else:
            self._suggestions_status.setText(
                "Clustering run changed — click 'Compute Cluster ID Suggestions' to refresh."
            )

    def _on_run_changed(self, _index: int):
        # suggestions are per-run and now persisted (see
        # update_cluster_id_suggestions); restore whatever the
        # newly-selected run already has saved rather than always
        # clearing to empty, so switching runs doesn't throw away a
        # suggestion computed earlier in a previous session.
        cl_run = self._selected_cluster_run()
        self._sync_suggestions_from_run(cl_run)
        self._update_compat_warning()
        self._redraw_map()
        self._populate_label_table()
        run_id = self.run_combo.currentData()
        cache = self._violin_cache.get(run_id)
        if cache:
            # this run's violins were already computed earlier in
            # the session — restore instantly instead of discarding them.
            self._populate_violin_channel_combo(cache['channels'])
            self._draw_current_violin()
        elif run_id is not None and self._checked_channels():
            self._recompute_violins()
        else:
            self._show_violin_placeholder(
                "Select a clustering run and channel(s) to plot violins."
            )
        # Marker Summary sub-tab: only touch it if it's the
        # currently visible sub-tab -- same lazy-when-visible rule
        # _on_annotation_sub_tab_changed uses.
        if self.annotation_sub_tabs.currentIndex() == getattr(self, '_marker_summary_tab_index', -1):
            ms_cache = self._marker_summary_cache.get(run_id)
            if ms_cache:
                self._draw_marker_summary()
            elif run_id is not None:
                self._recompute_marker_summary()

    def _on_dr_run_changed(self, _index: int):
        self._update_compat_warning()
        self._redraw_map()

    def _update_compat_warning(self):
        warning = drc_scatter.compatibility_warning(
            self._selected_dr_run(), self._selected_cluster_run()
        )
        self._compat_warning.setVisible(warning is not None)
        self._compat_warning.setToolTip(warning or '')

    # ------------------------------------------------------------------
    # Panel 1 — violin plots
    # ------------------------------------------------------------------

    def _populate_channel_list(self):
        """Rebuild the checkable channel grid from state.selected_channels.
        Preserves the current check state for channels still present."""
        previously_checked = set(self._checked_channels())
        channels = [c for c in self.state.selected_channels
                   if c not in drc_pipeline.META_CHANNELS]
        labels = _antigen_dash_labels(self.controller)

        while self.channel_grid.count():
            grid_item = self.channel_grid.takeAt(0)
            w = grid_item.widget()
            if w is not None:
                w.deleteLater()
        self.channel_checkboxes.clear()

        n_cols = 4
        for grid_idx, ch in enumerate(channels):
            cb = QCheckBox(labels.get(ch, ch))
            cb.setChecked(ch in previously_checked)
            cb.toggled.connect(self._on_violin_channel_checkbox_toggled)
            self.channel_checkboxes[ch] = cb
            row, col = divmod(grid_idx, n_cols)
            self.channel_grid.addWidget(cb, row, col)

        # Grow to show every marker at once instead of clipping to the old
        # fixed 70-140px box -- still caps out and falls back to the
        # scroll area's own scrollbar for large panels.
        n_rows = -(-len(channels) // n_cols) if channels else 1  # ceil div
        content_h = min(n_rows * 22 + 16, 260)
        self._channel_scroll.setMinimumHeight(content_h)
        self._channel_scroll.setMaximumHeight(content_h)

    def _set_all_channels(self, checked: bool):
        for cb in self.channel_checkboxes.values():
            cb.setChecked(checked)

    def _checked_channels(self) -> list[str]:
        return [ch for ch, cb in self.channel_checkboxes.items() if cb.isChecked()]

    def _show_violin_placeholder(self, text: str):
        self._violin_placeholder = QLabel(text)
        self._violin_placeholder.setStyleSheet("color: grey; font-style: italic;")
        self._violin_placeholder.setAlignment(Qt.AlignCenter)
        self._violin_scroll.setWidget(self._violin_placeholder)

    def _on_violin_channel_checkbox_toggled(self, _checked: bool):
        self._violin_recompute_timer.start()

    def _live_join_mode(self, cl_run: dict) -> str:
        """drc_run_align.live_join_mode() of a run (main thread only)."""
        return drc_run_align.live_join_mode(cl_run, drc_pipeline.sync_fingerprint(
            self.controller, cl_run.get('gates', []), cl_run.get('training_sample_ids', [])))

    def _pool_violin_data(self, cl_run: dict, channels: list[str],
                          af_state=None,
                          max_events_per_cluster: int | None = CLUSTER_PLOT_MAX_EVENTS,
                          live_mode: str | None = None,
                          ) -> dict[str, dict[int, list]]:
        """
        Pool per-cluster raw values for each channel from the run's own
        per-sample label arrays. Prefers the run's own stored marker
        values (drc_run_archive.run_marker_values: taken at classification
        time, a per-cluster sample whose rows index 'labels', or every
        event for older runs -- unaffected by gate/channel edits made
        since). For a sample without them, values are loaded from the
        current events and matched to the labels by event key, or row for row
        for a legacy run still in sync (live_mode, see
        drc_run_align.live_join_mode; computed here when not given, which
        must then be on the main thread).

        A cluster with more than *max_events_per_cluster* events (over all
        samples, in sample order) is thinned by even systematic sampling to
        that many, the same events in every channel, so later steps (the
        transform, violin densities, medians, histograms) only see a fixed
        number of events per cluster. None keeps every event.

        Returns {channel: {cluster_id: [per-sample arrays]}}.
        """
        labels_dict = cl_run.get('labels', {}) or {}
        snapshot_dict = drc_run_archive.run_marker_values(self.controller, cl_run)
        training_samples = cl_run.get('training_sample_ids', [])
        pooled: dict[str, dict[int, list]] = {ch: {} for ch in channels}
        self._violin_skipped_samples = []

        loaded = []
        for rel in training_samples:
            labels = labels_dict.get(rel)
            if labels is None:
                continue
            labels = np.asarray(labels)

            snap = snapshot_dict.get(rel)
            if snap is not None:
                values, names = snap[0], snap[1]
                values = np.asarray(values)
                if len(snap) == 3:
                    labels = labels[np.asarray(snap[2])]
                if len(values) != len(labels):
                    _log.warning(
                        "_pool_violin_data: %s -- snapshot (%d) vs labels (%d) "
                        "length mismatch, skipping sample.",
                        rel, len(values), len(labels),
                    )
                    self._violin_skipped_samples.append(rel)
                    continue
            else:
                # af_state, when given, is a main-thread unmixing
                # snapshot so a background worker never reads the
                # controller's matrices while the main window could be
                # reassigning them (see apply_unmixing_af_aware()).
                if live_mode is None:
                    live_mode = self._live_join_mode(cl_run)
                mv = drc_pipeline.load_sample_marker_values(
                    self.controller, self.state, rel, af_state=af_state,
                )
                rows = drc_pipeline.load_sample_rows(self.controller, self.state, rel,
                                                     af_state=af_state)
                if mv is None or rows is None:
                    continue
                values, names = mv
                values, matched = drc_run_align.pair_arrays(
                    np.asarray(values, dtype=np.float64), len(labels), live_mode,
                    src_rows=rows, dst_rows=drc_run_align.run_rows(cl_run, rel), fill=np.nan)
                if values is None:
                    _log.warning(
                        "_pool_violin_data: %s -- the run has no stored marker values "
                        "and cannot be matched to the current events, skipping sample.",
                        rel)
                    self._violin_skipped_samples.append(rel)
                    continue
                labels, values = labels[matched], values[matched]
            loaded.append((labels, values, list(names)))

        # Events of each cluster per sample, and each cluster's total.
        rows_by_sample = []
        totals: dict[int, int] = {}
        for labels, _values, _names in loaded:
            order = np.argsort(labels, kind='stable')
            ids, starts, counts = np.unique(labels[order], return_index=True, return_counts=True)
            rows = {int(cl_id): order[start:start + n]
                    for cl_id, start, n in zip(ids, starts, counts) if cl_id >= 0}
            rows_by_sample.append(rows)
            for cl_id, r in rows.items():
                totals[cl_id] = totals.get(cl_id, 0) + len(r)

        # Positions kept in each cluster's sample-ordered event sequence.
        kept = {}
        for cl_id, total in totals.items():
            if max_events_per_cluster is not None and total > max_events_per_cluster:
                kept[cl_id] = (np.arange(max_events_per_cluster, dtype=np.int64)
                               * total) // max_events_per_cluster
            else:
                kept[cl_id] = None

        offsets = {cl_id: 0 for cl_id in totals}
        for (labels, values, names), rows in zip(loaded, rows_by_sample):
            columns = [(ch, names.index(ch)) for ch in channels if ch in names]
            for cl_id, r in rows.items():
                keep = kept[cl_id]
                if keep is not None:
                    lo, hi = np.searchsorted(keep, [offsets[cl_id], offsets[cl_id] + len(r)])
                    r = r[keep[lo:hi] - offsets[cl_id]]
                offsets[cl_id] += len(rows[cl_id])
                if not len(r):
                    continue
                block = values[r]
                for ch, col in columns:
                    pooled[ch].setdefault(cl_id, []).append(block[:, col])

        return pooled

    def _recompute_violins(self):
        """
        Pool data for every checked channel and cache it against the
        current run, then show whichever channel is selected in
        violin_channel_combo (one big plot at a time instead of
        a multi-panel grid).
        """
        cl_run = self._selected_cluster_run()
        run_id = self.run_combo.currentData()
        if cl_run is None or run_id is None:
            self._show_violin_placeholder("Select a clustering run to plot violins.")
            return
        channels = self._checked_channels()
        if not channels:
            self._show_violin_placeholder("Check at least one channel to plot.")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            pooled = self._pool_violin_data(cl_run, channels)
        finally:
            QApplication.restoreOverrideCursor()
        self._warn_violin_samples_skipped(run_id, cl_run)
        self._violin_cache[run_id] = {
            'pooled': pooled,
            'channels': list(channels),
            'names_map': cl_run.get('names', {}),
            'colors_map': cl_run.get('colors', {}),
        }
        self._populate_violin_channel_combo(channels)
        self._draw_current_violin()

    def _warn_violin_samples_skipped(self, run_id, cl_run: dict):
        """Tell the user, once per run, when samples were left out of the
        violins because their marker values do not line up with the run's
        cluster labels."""
        skipped = list(getattr(self, '_violin_skipped_samples', []))
        warned = getattr(self, '_violin_warned_runs', None)
        if warned is None:
            warned = self._violin_warned_runs = set()
        if not skipped or run_id in warned:
            return
        warned.add(run_id)
        QMessageBox.warning(
            self, "Samples Left Out of the Violins",
            f"{len(skipped)} of {len(cl_run.get('training_sample_ids', []))} sample(s) "
            "were left out: this run has no stored marker values for them, and "
            "the current gating and unmixing give a different number of events "
            f"than were clustered (for example {skipped[0]}).\n\n"
            "Re-run the clustering on the current gating and unmixing to bring "
            "them back in sync.")

    def _populate_violin_channel_combo(self, channels: list[str]):
        prev = self.violin_channel_combo.currentData()
        self.violin_channel_combo.blockSignals(True)
        self.violin_channel_combo.clear()
        labels = _antigen_dash_labels(self.controller)
        for ch in channels:
            self.violin_channel_combo.addItem(labels.get(ch, ch), ch)
        idx = self.violin_channel_combo.findData(prev)
        self.violin_channel_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.violin_channel_combo.blockSignals(False)

    def _draw_current_violin(self):
        """Redraw from cached pooled data -- no recomputation, so
        switching the viewed channel (or switching back to a previously-
        viewed clustering run) is instant."""
        run_id = self.run_combo.currentData()
        cache = self._violin_cache.get(run_id)
        if not cache or not cache['pooled']:
            self._show_violin_placeholder("Check at least one channel to plot.")
            return
        ch = self.violin_channel_combo.currentData()
        if ch is None:
            return
        fig = self._make_single_violin_figure(
            ch, cache['pooled'].get(ch, {}), cache['names_map'], cache.get('colors_map', {}),
        )
        canvas = _new_scrollable_canvas(fig)
        # fill the Splitter panel instead of forcing a fixed
        # pixel size (which made the scroll area scroll rather than
        # shrink/grow the plot with the panel). setWidgetResizable(True)
        # on _violin_scroll already resizes whatever widget it holds to
        # the viewport size; Expanding just lets it grow past the
        # figure's own inches-based default too.
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._violin_scroll.setWidget(canvas)
        canvas.draw()

    def _make_single_violin_figure(self, ch: str, by_cluster: dict, names_map: dict,
                                   colors_map: dict | None = None):
        """One big violin plot for a single channel, filling the space.
        Each violin is coloured by its own cluster's colour (same colours
        as the Cluster Map / legend / label table) instead of one flat
        colour for all of them."""
        from matplotlib.figure import Figure

        is_dark = _resolve_is_dark(self.state)
        fg = 'white' if is_dark else 'black'
        fallback_color = '#5dade2' if is_dark else '#2e6da4'
        mean_color = '#ffd54f' if is_dark else '#c0392b'
        colors_map = colors_map or {}
        title = _antigen_dash_labels(self.controller).get(ch, ch)

        fig = Figure(figsize=(9, 6), constrained_layout=True)
        ax = fig.add_subplot(111)
        _style_figure_theme(fig, is_dark, axes=[ax])
        # Explicit grid, drawn UNDER the data and deliberately faint --
        # otherwise this inherits whatever global style is active (see
        # the "suppress inherited seaborn 'whitegrid'" comments elsewhere
        # in this file), which draws full-strength gridlines on TOP of
        # the violin.
        ax.set_axisbelow(True)
        ax.grid(True, axis='y', linewidth=0.4, alpha=0.15, color=fg)

        cl_ids = sorted(by_cluster.keys())
        if not cl_ids:
            ax.set_title(f"{title} (no data)", fontsize=11)
            ax.axis('off')
            return fig

        data = [_apply_channel_transform(self.state, ch, np.concatenate(by_cluster[cl_id]))
               for cl_id in cl_ids]
        parts = ax.violinplot(data, showmeans=True, showextrema=False)
        for cl_id, body in zip(cl_ids, parts['bodies']):
            color = colors_map.get(cl_id, fallback_color)
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.75)
        if 'cmeans' in parts:
            parts['cmeans'].set_color(mean_color)
            parts['cmeans'].set_linewidth(1.5)
        ax.set_xticks(range(1, len(cl_ids) + 1))
        ax.set_xticklabels(
            [names_map.get(cl, str(cl)) for cl in cl_ids],
            rotation=45, ha='right', fontsize=10,
        )
        ax.set_title(title, fontsize=13)
        tick_spec = _channel_axis_ticks(self.state, ch)
        if tick_spec is not None:
            major_ticks, minor_ticks, limits = tick_spec
            ax.set_yticks([pos for pos, _label in major_ticks])
            ax.set_yticklabels([label for _pos, label in major_ticks])
            ax.set_yticks([pos for pos, _label in minor_ticks], minor=True)
            ax.tick_params(axis='y', which='minor', length=2, labelsize=0)
            ax.set_ylim(limits[0], limits[1])
            ax.set_ylabel('Intensity', fontsize=10)
        else:
            ax.set_ylabel('Transformed intensity', fontsize=10)
        ax.tick_params(labelsize=9)
        return fig

    # ------------------------------------------------------------------
    # Panel 2 — cluster map
    # ------------------------------------------------------------------

    def _redraw_map(self):
        cl_run = self._selected_cluster_run()
        dr_run = self._selected_dr_run()

        self._map_figure.clear()
        ax = self._map_figure.add_subplot(111)
        self._draw_map_axes(ax, cl_run, dr_run, fontsize=7)
        _style_figure_theme(self._map_figure, _resolve_is_dark(self.state), axes=[ax])
        self._map_canvas.draw_idle()
        self._rebuild_map_legend(cl_run)

    def _draw_map_axes(self, ax, cl_run: dict | None, dr_run: dict | None, fontsize: int = 7):
        """
        Draw the cluster map onto *ax* -- shared by the inline panel
        (_redraw_map) and the bigger pop-out dialog (_pop_out_map)
        so both render identically, the same way PlotCard and this
        tab already share drc_scatter.draw_cluster_scatter itself.
        """
        ax.set_box_aspect(1)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(False)   # suppress inherited seaborn 'whitegrid'

        if dr_run is None:
            ax.text(0.5, 0.5, 'No DR run selected.', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9)
            return

        emb_dict = dr_run.get('embeddings', {}) or {}
        if not emb_dict:
            ax.text(0.5, 0.5, f"No embeddings for \"{dr_run.get('label', '')}\".",
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)
            return

        labels_dict = cl_run.get('labels', {}) if cl_run else {}
        own_positions = None
        cl_params = (cl_run.get('params', {}) if cl_run else {}) or {}
        # Only trust the run's own frozen positions as a substitute when
        # they're nominally the SAME embedding space as what's on screen --
        # mixing two different algorithms' coordinates would be meaningless.
        same_algo = bool(cl_params.get('_dr_algo')) and \
            cl_params.get('_dr_algo') == (dr_run.get('algorithm') or '')
        unaligned_samples: list[str] = []
        join_mode = drc_run_align.runs_join_mode(dr_run, cl_run) if cl_run else None
        xys, labs = [], []
        for rel, emb in emb_dict.items():
            n = len(emb)
            lbl_raw = labels_dict.get(rel)
            lbl, aligned_ok = drc_scatter.align_labels_to_embedding(
                lbl_raw, n,
                label_rows=drc_run_align.run_rows(cl_run, rel) if cl_run else None,
                emb_rows=drc_run_align.run_rows(dr_run, rel), mode=join_mode)
            if not aligned_ok and lbl_raw is not None and same_algo and own_positions is None:
                own_positions = self._run_dr_positions(cl_run)
            if aligned_ok:
                xys.append(emb)
                labs.append(lbl)
            elif lbl_raw is not None and same_algo and rel in own_positions \
                    and len(own_positions[rel]) == len(lbl_raw):
                # The selected DR run's live embeddings for this sample no
                # longer match this clustering run's labels (e.g. DR was
                # retrained for this algorithm since archiving) -- fall
                # back to the exact positions this run actually classified
                # against instead of greying the sample out.
                xys.append(own_positions[rel])
                labs.append(lbl_raw)
            else:
                # Keyed runs: labels where the events match, grey elsewhere
                if lbl_raw is not None:
                    unaligned_samples.append(rel)
                xys.append(emb)
                labs.append(lbl)
        xy = np.concatenate(xys, axis=0)
        lab = np.concatenate(labs, axis=0)

        if len(xy) > 30_000:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(xy), 30_000, replace=False)
            xy, lab = xy[idx], lab[idx]

        algo = dr_run.get('algorithm', '') or ''
        ax.set_xlabel(f"{algo} 1", fontsize=fontsize)
        ax.set_ylabel(f"{algo} 2", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)

        drc_scatter.draw_cluster_scatter(ax, xy, lab, cl_run, self.controller)
        if unaligned_samples:
            ax.text(0.02, 0.02,
                   f"⚠ {len(unaligned_samples)} sample(s) shown without cluster "
                   "colour — DR run and clustering run don't share indexed events",
                   transform=ax.transAxes, fontsize=max(fontsize - 1, 6),
                   color='#d9822b', va='bottom', ha='left', zorder=10)

    def _run_dr_positions(self, cl_run: dict) -> dict:
        """The DR positions a DR-space clustering run classified: stored with
        runs archived before 'params'['_dr_run_id'] existed, else its source
        DR run's embeddings ({} if that run has been deleted)."""
        stored = cl_run.get('dr_positions') or {}
        if stored:
            return stored
        source_id = (cl_run.get('params') or {}).get('_dr_run_id')
        for entry in self.state.dr_runs:
            if source_id and entry.get('run_id') == source_id:
                return drc_run_archive.hydrate_run(self.controller, entry).get('embeddings') or {}
        return {}

    def _pop_out_map(self):
        """Regenerate the cluster map at a larger size in its own window,
        with the same pan/zoom toolbar the Stats results tabs already
        have, and its own copy of the swatch legend."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

        cl_run = self._selected_cluster_run()
        dr_run = self._selected_dr_run()

        dlg_fig = Figure(figsize=(8, 8), constrained_layout=True)
        ax = dlg_fig.add_subplot(111)
        self._draw_map_axes(ax, cl_run, dr_run, fontsize=10)
        _style_figure_theme(dlg_fig, _resolve_is_dark(self.state), axes=[ax])

        dlg_canvas = _new_scrollable_canvas(dlg_fig)
        dlg_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        dlg_toolbar = NavigationToolbar2QT(dlg_canvas, None)

        dlg_legend_scroll = QScrollArea()
        dlg_legend_scroll.setWidgetResizable(True)
        dlg_legend_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        dlg_legend_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        dlg_legend_scroll.setMinimumWidth(140)
        dlg_legend_scroll.setFrameShape(QFrame.NoFrame)
        dlg_legend_widget = _WrappingLegendWidget()
        dlg_legend_scroll.setWidget(dlg_legend_widget)

        def _refresh_popout():
            ax.clear()
            self._draw_map_axes(ax, self._selected_cluster_run(), dr_run, fontsize=10)
            _style_figure_theme(dlg_fig, _resolve_is_dark(self.state), axes=[ax])
            dlg_canvas.draw_idle()
            dlg_legend_widget.set_entries(
                self._build_legend_entries(self._selected_cluster_run(), extra_refresh=_refresh_popout)
            )

        dlg_legend_widget.set_entries(
            self._build_legend_entries(cl_run, extra_refresh=_refresh_popout)
        )

        dlg = QDialog(self)
        dlg.setWindowTitle("Cluster Map")
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.resize(900, 900)
        layout = QVBoxLayout(dlg)
        layout.addWidget(dlg_toolbar)
        plot_row = QHBoxLayout()
        plot_row.addWidget(dlg_canvas, stretch=1)
        plot_row.addWidget(dlg_legend_scroll)
        layout.addLayout(plot_row)
        dlg_canvas.draw()
        dlg.show()

    def _build_legend_entries(self, cl_run: dict | None, extra_refresh=None) -> list[QWidget]:
        """
        Build one swatch+name row per cluster. Shared by the inline
        legend (_rebuild_map_legend) and the pop-out dialog's own legend
        (_pop_out_map) so both look and behave identically.

        extra_refresh, if given, is called after a rename/recolour is
        applied through THIS set of entries -- lets the pop-out dialog
        redraw its own (separate) figure and legend after an edit made
        from within the dialog, in addition to the normal
        self._rebuild_map_legend()/self._redraw_map() the underlying
        _pick_colour()/_prompt_rename() already trigger for the main panel.
        """
        colors = cl_run.get('colors', {}) if cl_run else {}
        names = cl_run.get('names', {}) if cl_run else {}
        entries: list[QWidget] = []
        for lbl in sorted(colors.keys()):
            color = colors[lbl]
            name  = _cluster_display_name(names, lbl)

            row = QHBoxLayout()
            row.setSpacing(4)

            swatch = QPushButton()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(
                f"background-color: {color}; border: 1px solid #555; border-radius: 2px;"
            )
            swatch.setToolTip("Right-click to change colour")
            swatch.setContextMenuPolicy(Qt.CustomContextMenu)

            def _on_recolor(pos, l=lbl):
                self._pick_colour(l)
                if extra_refresh:
                    extra_refresh()
            swatch.customContextMenuRequested.connect(_on_recolor)
            row.addWidget(swatch)

            name_lbl = QLabel(name)
            name_lbl.setStyleSheet("font-size: 9px;")
            name_lbl.setToolTip("Double-click to rename")

            def _on_rename(e, l=lbl):
                self._prompt_rename(l)
                if extra_refresh:
                    extra_refresh()
            name_lbl.mouseDoubleClickEvent = _on_rename
            row.addWidget(name_lbl, stretch=1)

            row_w = QWidget()
            row_w.setLayout(row)
            entries.append(row_w)

        return entries

    def _rebuild_map_legend(self, cl_run: dict | None):
        self._legend_widget.set_entries(self._build_legend_entries(cl_run))
        self._sync_legend_column_width()

    def _on_map_legend_splitter_moved(self, *_args):
        """User dragged the Cluster Map / legend splitter handle -- stop
        auto-fitting the legend column from here on, so a later rename/
        recolour rebuild doesn't undo their choice."""
        self._legend_width_user_set = True

    def _sync_legend_column_width(self):
        """
        Size the legend column to fit its current
        content (longest name across however many columns
        _WrappingLegendWidget wrapped into), so the default is never
        narrower than the names actually need. No-ops once the user has
        manually dragged the splitter handle (_on_map_legend_splitter_
        moved) -- their choice then sticks across future rebuilds.
        """
        if self._legend_width_user_set:
            return
        splitter = self._map_plot_splitter
        total = sum(splitter.sizes()) or splitter.width()
        if total <= 0:
            return
        # Content width + scrollbar/frame allowance, clamped so a
        # handful of long names doesn't eat the whole panel and a
        # single short name doesn't collapse it.
        content_w = self._legend_widget.sizeHint().width()
        legend_w = max(90, min(content_w + 24, 320, total - 150))
        splitter.setSizes([max(150, total - legend_w), legend_w])

    def _prompt_rename(self, label: int):
        from PySide6.QtWidgets import QInputDialog
        cl_run = self._selected_cluster_run()
        if cl_run is None:
            return
        names = cl_run.get('names', {})
        current = _cluster_display_name(names, label)
        new_name, ok = QInputDialog.getText(
            self, f"Rename cluster {label}", "New name:", text=current
        )
        if not ok or not new_name.strip():
            return
        if drc_scatter.rename_cluster(self.controller, cl_run, label, new_name.strip(), self):
            self._rebuild_map_legend(cl_run)
            self._populate_label_table()

    def _pick_colour(self, label: int):
        cl_run = self._selected_cluster_run()
        if cl_run is None:
            return
        current = cl_run.get('colors', {}).get(label, '#aaaaaa')
        colour = QColorDialog.getColor(QColor(current), self, f"Colour for cluster {label}")
        if colour.isValid():
            drc_scatter.recolor_cluster(self.controller, cl_run, label, colour.name())
            self._rebuild_map_legend(cl_run)
            self._populate_label_table()
            self._redraw_map()

    # ------------------------------------------------------------------
    # Panel 3 — cluster label table
    # ------------------------------------------------------------------

    def _populate_label_table(self):
        cl_run = self._selected_cluster_run()
        self.label_table.blockSignals(True)
        self.label_table.setRowCount(0)
        if cl_run is None:
            self.label_table.blockSignals(False)
            return

        labels_dict = cl_run.get('labels', {}) or {}
        colors = cl_run.get('colors', {}) or {}
        names = cl_run.get('names', {}) or {}
        all_labels = (np.concatenate([np.asarray(v) for v in labels_dict.values()])
                     if labels_dict else np.array([], dtype=int))
        total = len(all_labels)
        unique = (sorted(int(l) for l in np.unique(all_labels)) if total
                 else sorted(colors.keys()))

        # only show suggestions if they were computed for THIS
        # run (not stale leftovers from a previously-selected run).
        suggestions_current = bool(cl_run.get('run_id')) and \
            self._suggestions_run_id == cl_run.get('run_id')
        cell_type_df = self._cell_type_df if suggestions_current else None

        self.label_table.setRowCount(len(unique))
        for row, cl_id in enumerate(unique):
            name = _cluster_display_name(names, cl_id)
            name_item = QTableWidgetItem(name)
            name_item.setData(Qt.UserRole, cl_id)
            self.label_table.setItem(row, 0, name_item)

            colour_item = QTableWidgetItem('')
            colour_item.setBackground(QColor(colors.get(cl_id, '#aaaaaa')))
            colour_item.setFlags(colour_item.flags() & ~Qt.ItemIsEditable)
            colour_item.setData(Qt.UserRole, cl_id)
            self.label_table.setItem(row, 1, colour_item)

            n_events = int(np.sum(all_labels == cl_id)) if total else 0
            pct = (n_events / total * 100.0) if total else 0.0
            events_item = QTableWidgetItem(str(n_events))
            events_item.setFlags(events_item.flags() & ~Qt.ItemIsEditable)
            self.label_table.setItem(row, 2, events_item)
            pct_item = QTableWidgetItem(f"{pct:.1f}%")
            pct_item.setFlags(pct_item.flags() & ~Qt.ItemIsEditable)
            self.label_table.setItem(row, 3, pct_item)

            mem_text = '—'
            if suggestions_current and cl_id >= 0:
                mem_text = self._mem_labels.get(cl_id, '—')
            mem_item = QTableWidgetItem(mem_text)
            mem_item.setFlags(mem_item.flags() & ~Qt.ItemIsEditable)
            mem_item.setData(Qt.UserRole, cl_id)
            mem_item.setToolTip(
                "Double-click to adopt as this cluster's Name." if mem_text != '—' else ''
            )
            self.label_table.setItem(row, 4, mem_item)

            type_text, type_tip = '—', ''
            if cell_type_df is not None and cl_id >= 0 and cl_id in cell_type_df.index:
                rec = cell_type_df.loc[cl_id]
                suggested = rec.get('suggested_type') or ''
                if suggested:
                    low_conf = bool(rec.get('low_confidence'))
                    type_text = f"{suggested} ⚠" if low_conf else suggested
                    type_tip = (
                        "Driven mainly by absent negative markers, not positive "
                        "evidence -- treat with caution." if low_conf
                        else f"Score: {rec.get('score')}"
                    )
            type_item = QTableWidgetItem(type_text)
            type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)
            type_item.setToolTip(type_tip)
            self.label_table.setItem(row, 5, type_item)
        self.label_table.blockSignals(False)

        # Small fixed floor only. This used to grow with cluster count,
        # but _populate_label_table() runs on every run switch, rename,
        # and recolour -- not just once at startup -- so a minimum that
        # regrew here kept re-asserting itself mid-drag and fighting the
        # splitter. The table has its own scrollbar for extra rows.
        row_h = self.label_table.verticalHeader().defaultSectionSize()
        header_h = self.label_table.horizontalHeader().height()
        self.label_table.setMinimumHeight(header_h + row_h * 3 + 8)

    def _on_table_item_changed(self, item: QTableWidgetItem):
        """Inline rename via the Name column. Same duplicate check as the
        map legend and PlotCard (drc_scatter.rename_cluster) — reverts the
        cell text if rejected."""
        if item.column() != 0:
            return
        cl_run = self._selected_cluster_run()
        if cl_run is None:
            return
        cl_id = item.data(Qt.UserRole)
        new_name = item.text().strip()
        if not new_name or not drc_scatter.rename_cluster(
            self.controller, cl_run, cl_id, new_name, self
        ):
            self._populate_label_table()
            return
        self._rebuild_map_legend(cl_run)

    def _on_table_cell_double_clicked(self, row: int, col: int):
        if col == 1:
            item = self.label_table.item(row, col)
            cl_id = item.data(Qt.UserRole)
            cl_run = self._selected_cluster_run()
            if cl_run is None:
                return
            current = cl_run.get('colors', {}).get(cl_id, '#aaaaaa')
            colour = QColorDialog.getColor(QColor(current), self, f"Colour for cluster {cl_id}")
            if colour.isValid():
                drc_scatter.recolor_cluster(self.controller, cl_run, cl_id, colour.name())
                self._populate_label_table()
                self._rebuild_map_legend(cl_run)
                self._redraw_map()
        elif col == 4:
            #one-click MEM label adoption for a single cluster.
            item = self.label_table.item(row, col)
            cl_id = item.data(Qt.UserRole)
            cl_run = self._selected_cluster_run()
            if cl_run is None or cl_id not in self._mem_labels:
                return
            if drc_scatter.rename_cluster(
                self.controller, cl_run, cl_id, self._mem_labels[cl_id], self
            ):
                self._rebuild_map_legend(cl_run)
                self._populate_label_table()

    # ------------------------------------------------------------------
    # Panel 4 — Cluster ID suggestions
    # ------------------------------------------------------------------

    def _compute_cluster_id_suggestions(self):
        """
        Runs both suggestion mechanisms for the selected run over the
        CHANNEL SET RECORDED FOR THIS RUN -- cl_run['channels'], the
        Configuration tab's selection at the time THIS SPECIFIC run was
        created (archived once by drc_run_archive.archive_clustering_run() /
        archive_dr_run(), never touched again). Deliberately NOT Panel 1's
        Per-Marker Violin Plots checkboxes (those only ever drove the violin
        plots) and NOT state.selected_channels (today's live Configuration
        selection, which may belong to an entirely different run by the
        time this one is selected in the combo):
          - MEM Label: a descriptive statistic of the data itself, safe to
            one-click adopt (see _adopt_all_mem_labels / the MEM Label
            double-click handler above).
          - Suggested Type: a database lookup against drc_cluster_id's
            bundled cell-type marker signatures -- a biological CLAIM, not
            just a computed statistic, so it is display-only here;
            adopting one is a manual retype into Name.

        Marker naming is enforced in two tiers before/after computing --
        see drc_cluster_id's module docstring:
          1. HARD gate (blocks): any channel in this run's recorded set
             with no Antigen typed in at all, in the Spectral Process tab.
             Refuses to proceed -- this used to silently fall back to the
             fluorophore Label, mixing marker names and fluorophore names
             in the same MEM Label output with no indication why.
          2. SOFT warning (proceeds anyway): an Antigen that's filled in
             but doesn't match anything in marker_database.csv. Shown
             after computing -- these channels contribute nothing to
             Suggested Type, but still count toward MEM Label as typed.

        Shows a determinate progress bar while computing -- loading +
        unmixing every training sample from disk is the dominant cost, so
        the bar advances one step per sample rather than per channel.
        """
        if self._cluster_id_worker is not None and self._cluster_id_worker.isRunning():
            return

        cl_run = self._selected_cluster_run()
        if cl_run is None:
            QMessageBox.warning(self, "No Run Selected", "Select a clustering run first.")
            return
        channels = cl_run.get('channels', [])
        if not channels:
            QMessageBox.warning(
                self, "No Channels Recorded",
                "This clustering run has no channel set recorded (the "
                "Configuration tab's selection at the time it was created) "
                "-- Cluster ID cannot run without it."
            )
            return

        # Every channel in
        # this run's recorded set must have an Antigen typed in before
        # we'll compute anything -- otherwise MEM Label falls back to the
        # fluorophore Label for that channel and mixes marker/fluorophore
        # names in the same MEM Label output with no indication why.
        missing_antigen = drc_cluster_id.channels_missing_antigen(self.controller, channels)
        if missing_antigen:
            display_labels = _antigen_dash_labels(self.controller)
            names = '\n  '.join(display_labels.get(ch, ch) for ch in missing_antigen)
            QMessageBox.warning(
                self, "Antigen Not Assigned",
                "Cluster ID requires every channel in this run's recorded "
                "set to have an Antigen assigned in the Spectral Process "
                "tab first.\n\nMissing Antigen for:\n  " + names +
                "\n\nAssign an Antigen to these channels and try again."
            )
            return

        token = acquire_job(self.controller, 'Honeycluster: Cluster ID')
        if token is None:
            self._suggestions_status.setText(self.controller.job_gate.busy_text())
            return
        total_steps = drc_cluster_id.total_progress_steps(cl_run)
        self._suggestions_progress.setRange(0, total_steps)
        self._suggestions_progress.setValue(0)
        self._suggestions_progress.setVisible(True)
        self._suggestions_status.setText("Computing MEM scores and cell-type matches …")
        self.compute_suggestions_btn.setEnabled(False)

        mem_threshold = self.mem_threshold_spin.value()
        species = self.species_combo.currentData()

        # Snapshot AF/transfer-matrix state HERE, on the main thread, before
        # the worker starts -- same reasoning as _StatsWorker/_ClWorker/
        # _DrWorker: controller.load_sample() reassigns these arrays in
        # place whenever the user loads a different sample in the main
        # window, and the AF kernel touches them via raw C pointers, so a
        # concurrent reassignment is a memory-corruption hazard, not just
        # stale data.
        af_state = drc_pipeline.snapshot_af_state(self.controller)
        # The unstained sample(s) used to derive positivity
        # thresholds have no AF profile of their own; resolve a stand-in
        # per unstained sample (name match, or the run's most common
        # assignment) and snapshot its AF state too, same main-thread
        # requirement as af_state above -- see
        # drc_cluster_id.resolve_unstained_af_states's docstring.
        unstained_af_states = drc_cluster_id.resolve_unstained_af_states(self.controller, cl_run)
        live_mode = self._live_join_mode(cl_run)

        plugin_ref = self

        class _ClusterIdWorker(QThread):
            progress = Signal(int)
            finished = Signal(bool, str, object)

            def __init__(self_, cl_run, channels, mem_threshold, species, af_state,
                         unstained_af_states):
                super().__init__()
                self_._cl_run = cl_run
                self_._channels = channels
                self_._mem_threshold = mem_threshold
                self_._species = species
                self_._af_state = af_state
                self_._unstained_af_states = unstained_af_states

            def run(self_):
                try:
                    result = drc_cluster_id.compute_cluster_id_suggestions(
                        plugin_ref.controller, plugin_ref.state,
                        self_._cl_run, self_._channels,
                        mem_threshold=self_._mem_threshold,
                        species=self_._species,
                        progress_callback=lambda n: self_.progress.emit(n),
                        af_state=self_._af_state,
                        unstained_af_states=self_._unstained_af_states,
                        live_mode=live_mode,
                    )
                    self_.finished.emit(True, '', result)
                except Exception as exc:
                    traceback.print_exc()
                    self_.finished.emit(False, str(exc), None)

        worker = _ClusterIdWorker(cl_run, channels, mem_threshold, species, af_state,
                                   unstained_af_states)
        worker.finished.connect(lambda *_: token.release())
        worker.progress.connect(self._suggestions_progress.setValue)
        worker.finished.connect(
            lambda success, err, result, run_id=cl_run.get('run_id'), n_channels=len(channels):
                self._on_cluster_id_finished(success, err, result, run_id, n_channels)
        )
        self._cluster_id_worker = worker
        worker.start()

    def _on_cluster_id_finished(self, success: bool, error_msg: str, result, run_id, n_channels):
        self._cluster_id_worker = None
        self.compute_suggestions_btn.setEnabled(True)
        self._suggestions_progress.setVisible(False)

        if not success:
            _log.error("Cluster ID suggestion computation failed: %s", error_msg)
            QMessageBox.warning(
                self, "Computation Failed", f"Could not compute suggestions:\n{error_msg}"
            )
            self._suggestions_status.setText(
                "Cluster ID suggestions not yet computed for this run."
            )
            return

        mem_labels, _mem_scores, cell_type_df, unmatched_markers = result
        self._mem_labels = mem_labels
        self._cell_type_df = cell_type_df
        self._suggestions_run_id = run_id
        drc_run_archive.update_cluster_id_suggestions(
            self.controller, self.state, run_id, mem_labels, cell_type_df,
        )

        # Antigen text
        # that's present but unrecognised by marker_database.csv doesn't
        # block anything -- just flag it, since it silently contributes
        # nothing to Suggested Type otherwise.
        if unmatched_markers:
            display_labels = _antigen_dash_labels(self.controller)
            lines = '\n  '.join(
                f'{display_labels.get(ch, ch)} — Antigen "{antigen}"'
                for ch, antigen in unmatched_markers
            )
            QMessageBox.warning(
                self, "Unmatched Antigens",
                "The following Antigen entries didn't match anything in "
                "marker_database.csv, so they will NOT contribute to "
                "Suggested Type (cell-type scoring). They still count "
                "toward MEM Label as typed.\n\n  " + lines
            )

        self._suggestions_status.setText(
            f"Suggestions ready for {len(mem_labels)} cluster(s), "
            f"{n_channels} channel(s), {self.species_combo.currentText()} "
            f"cell types."
        )
        self._populate_label_table()

    def _adopt_all_mem_labels(self):
        """
        One-click adoption of every cluster's MEM label as its Name (the
        MEM-label path -- see class docstring). Duplicate labels (most
        commonly two clusters both scoring 'Uncharacterized') are numbered
        rather than rejected one-by-one, so this doesn't pop a Duplicate
        Name dialog per collision.
        """
        cl_run = self._selected_cluster_run()
        if cl_run is None or not self._mem_labels or \
           self._suggestions_run_id != cl_run.get('run_id'):
            QMessageBox.warning(
                self, "Nothing to Adopt",
                "Compute Cluster ID suggestions for this run first."
            )
            return
        reply = QMessageBox.question(
            self, "Adopt All MEM Labels",
            f"Set the Name of all {len(self._mem_labels)} cluster(s) in this "
            "run to their generated MEM label? This overwrites any existing "
            "names.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        # All names are assigned in memory, then persisted in one write:
        # one rename_cluster() call per cluster wrote once per cluster, and
        # checking each new name against not-yet-replaced old names
        # rejected valid ones. The numbering below makes the new set unique.
        names = cl_run.setdefault('names', {})
        seen: dict[str, int] = {}
        for cl_id in sorted(self._mem_labels.keys()):
            base = self._mem_labels[cl_id].strip()
            if not base:
                continue
            seen[base] = seen.get(base, 0) + 1
            names[cl_id] = base if seen[base] == 1 else f"{base} ({seen[base]})"
        drc_run_archive.update_run_fields(self.controller, cl_run, 'names')
        self._rebuild_map_legend(cl_run)
        self._populate_label_table()

    def _on_annotation_sub_tab_changed(self, index: int):
        """Lazily compute the Marker Summary sub-tab only when it becomes
        visible, and only if this run hasn't been computed yet -- same
        lazy/cached pattern as the violin panel."""
        if index != getattr(self, '_marker_summary_tab_index', -1):
            return
        run_id = self.run_combo.currentData()
        if run_id in self._marker_summary_cache:
            self._draw_marker_summary()
        elif run_id is not None:
            self._recompute_marker_summary()

    def _recompute_marker_summary(self):
        cl_run = self._selected_cluster_run()
        run_id = self.run_combo.currentData()
        if cl_run is None or run_id is None:
            self._show_marker_summary_placeholder("Select a clustering run first.")
            return
        channels = [c for c in self.state.selected_channels
                   if c not in drc_pipeline.META_CHANNELS]
        if not channels:
            self._show_marker_summary_placeholder(
                "No channels selected (see Configuration tab)."
            )
            return
        cluster_order = sorted(cl for cl in cl_run.get('colors', {}) if cl >= 0)
        if wait_if_busy(self.controller, ('Honeycluster marker summary', id(self)),
                        self._recompute_marker_summary, self._show_marker_summary_placeholder):
            return

        # Pooling (disk I/O + unmixing) and figure construction
        # can take a while for large runs; both now happen on a
        # background QThread instead of blocking the UI, the same way
        # _StatsWorker/_DrWorker/_ClusterIdWorker already do elsewhere in
        # this tab. AF/transfer-matrix state is snapshotted HERE, on the
        # main thread, before the worker starts -- see _StatsWorker's own
        # comment for why reading it live off the controller from a
        # background thread is a memory-corruption hazard, not just a
        # stale-data one.
        af_state = drc_pipeline.snapshot_af_state(self.controller)
        self._marker_summary_last_drawn = None
        self._start_marker_summary_worker(
            cl_run, channels, cluster_order, af_state, pooled=None, run_id=run_id,
        )

    def _start_marker_summary_worker(self, cl_run: dict, channels: list[str],
                                     cluster_order: list[int], af_state, pooled, run_id):
        running = self._marker_summary_worker
        if running is not None and running.isRunning():
            _log.info("marker summary: worker already running, request deferred")
            self._marker_summary_rerun_pending = True
            return
        self.marker_summary_recompute_btn.setEnabled(False)
        self._set_marker_summary_busy(True)
        self._show_marker_summary_placeholder("⏳ Computing marker summary …")
        worker = _MarkerSummaryWorker(self, cl_run, channels, cluster_order, af_state, pooled,
                                      live_mode=self._live_join_mode(cl_run))
        worker.finished.connect(
            lambda ok, err, payload, rid=run_id, w=worker:
                self._on_marker_summary_finished(ok, err, payload, rid, w)
        )
        self._marker_summary_worker = worker
        worker.start()

    def _set_marker_summary_busy(self, busy: bool):
        """Busy cursor while the summary computes and draws."""
        if busy == self._marker_summary_busy:
            return
        self._marker_summary_busy = busy
        if busy:
            QApplication.setOverrideCursor(Qt.BusyCursor)
        else:
            QApplication.restoreOverrideCursor()

    def _on_marker_summary_finished(self, success: bool, error: str, payload: dict,
                                    run_id, worker=None):
        # The thread has returned from run(); wait() lets it finish before
        # the last reference to the QThread is dropped.
        if worker is not None:
            worker.wait()
            if self._marker_summary_worker is worker:
                self._marker_summary_worker = None
        try:
            self._store_and_draw_marker_summary(success, error, payload, run_id)
        finally:
            self._set_marker_summary_busy(False)
            self.marker_summary_recompute_btn.setEnabled(True)
            if self._marker_summary_rerun_pending:
                self._marker_summary_rerun_pending = False
                QTimer.singleShot(0, self._resume_marker_summary)

    def _resume_marker_summary(self):
        """Handle a request that arrived while a worker was running."""
        if self.annotation_sub_tabs.currentIndex() != getattr(self, '_marker_summary_tab_index', -1):
            return
        run_id = self.run_combo.currentData()
        if run_id is None:
            return
        if run_id in self._marker_summary_cache:
            self._draw_marker_summary()
        else:
            self._recompute_marker_summary()

    def _store_and_draw_marker_summary(self, success: bool, error: str, payload: dict, run_id):
        if not success:
            self._show_marker_summary_placeholder(f"Failed to compute marker summary: {error}")
            return
        is_dark = _resolve_is_dark(self.state)
        self._marker_summary_cache[run_id] = {
            'channels': payload['channels'],
            'cluster_order': payload['cluster_order'],
            'pooled': payload['pooled'],
            'names_map': payload['names_map'],
            'colors_map': payload['colors_map'],
            'is_dark': is_dark,
            'main_fig': payload['main_fig'],
            'col_fig': payload['col_fig'],
            'row_fig': payload['row_fig'],
            'cbar_fig': payload['cbar_fig'],
            'ridge_fig': payload['ridge_fig'],
        }
        if self.run_combo.currentData() != run_id:
            # The user switched to a different run while this was
            # computing -- the result is cached for later, but don't draw
            # it over whatever run is now actually selected.
            return
        self._marker_summary_last_drawn = (run_id, is_dark)
        self._apply_marker_summary_figures(
            payload['main_fig'], payload['col_fig'], payload['row_fig'],
            payload['cbar_fig'], payload['ridge_fig'],
        )

    def _show_marker_summary_placeholder(self, text: str):
        for scroll in (self._heatmap_scroll, self._ridge_scroll):
            placeholder = QLabel(text)
            placeholder.setStyleSheet("color: grey; font-style: italic;")
            placeholder.setAlignment(Qt.AlignCenter)
            scroll.setWidget(placeholder)
        for empty_scroll in (self._heatmap_col_header_scroll, self._heatmap_row_header_scroll,
                             self._heatmap_colorbar_scroll):
            empty_scroll.setWidget(QWidget())

    def _draw_marker_summary(self):
        run_id = self.run_combo.currentData()
        cache = self._marker_summary_cache.get(run_id)
        if not cache or not cache['cluster_order']:
            self._marker_summary_last_drawn = None
            self._show_marker_summary_placeholder(
                "No clustered data to summarise for this run."
            )
            return

        is_dark = _resolve_is_dark(self.state)
        draw_key = (run_id, is_dark)
        if draw_key == self._marker_summary_last_drawn:
            # Same run, same theme, pooled data hasn't changed since the
            # last render -- switching back to this sub-tab shouldn't
            # rebuild two matplotlib figures from scratch every time.
            return

        if cache.get('main_fig') is not None and cache.get('is_dark') == is_dark:
            # Figures already built this session at the CURRENT theme --
            # reuse them directly instead of rebuilding via the worker.
            self._marker_summary_last_drawn = draw_key
            self._apply_marker_summary_figures(
                cache['main_fig'], cache['col_fig'], cache['row_fig'],
                cache['cbar_fig'], cache['ridge_fig'],
            )
            return

        if cache.get('pooled') is not None:
            # figure construction from already-pooled data still
            # goes through the background worker (skipping the pooling
            # step, since 'pooled' is passed straight through) so a big
            # ridge grid can't block the UI just from a theme switch.
            cl_run = self._selected_cluster_run() or {}
            self._start_marker_summary_worker(
                cl_run, cache['channels'], cache['cluster_order'], None,
                pooled=cache['pooled'], run_id=run_id,
            )
        else:
            # No pooled arrays cached -- needs a full repool, same cost
            # as the very first view of this run.
            self._recompute_marker_summary()

    def _apply_marker_summary_figures(self, main_fig, col_fig, row_fig, cbar_fig, ridge_fig):
        # No draw() here: each canvas draws itself once when shown
        dpi = main_fig.get_dpi()

        main_canvas = _new_scrollable_canvas(main_fig)
        main_canvas.setFixedSize(int(main_fig.get_figwidth() * dpi), int(main_fig.get_figheight() * dpi))
        self._heatmap_scroll.setWidget(main_canvas)

        col_canvas = _new_scrollable_canvas(col_fig)
        col_canvas.setFixedSize(int(col_fig.get_figwidth() * dpi), int(col_fig.get_figheight() * dpi))
        self._heatmap_col_header_scroll.setWidget(col_canvas)
        self._heatmap_col_header_scroll.setFixedHeight(int(col_fig.get_figheight() * dpi))

        row_canvas = _new_scrollable_canvas(row_fig)
        row_canvas.setFixedSize(int(row_fig.get_figwidth() * dpi), int(row_fig.get_figheight() * dpi))
        self._heatmap_row_header_scroll.setWidget(row_canvas)
        self._heatmap_row_header_scroll.setFixedWidth(int(row_fig.get_figwidth() * dpi))

        self._heatmap_corner.setFixedSize(
            int(row_fig.get_figwidth() * dpi), int(col_fig.get_figheight() * dpi)
        )

        cbar_dpi = cbar_fig.get_dpi()
        cbar_canvas = _new_scrollable_canvas(cbar_fig)
        cbar_canvas.setFixedSize(int(cbar_fig.get_figwidth() * cbar_dpi), int(cbar_fig.get_figheight() * cbar_dpi))
        self._heatmap_colorbar_scroll.setWidget(cbar_canvas)

        ridge_canvas = _new_scrollable_canvas(ridge_fig)
        dpi2 = ridge_fig.get_dpi()
        # Fixed: a resize would redraw the whole grid
        ridge_canvas.setFixedSize(int(ridge_fig.get_figwidth() * dpi2), int(ridge_fig.get_figheight() * dpi2))
        self._ridge_scroll.setWidget(ridge_canvas)

        # Default the splitter to the heatmap's own natural height, read
        # from Qt's own layout metrics rather than a guessed "chrome"
        # constant -- pin the one flexible piece (the main
        # heatmap viewport, normally stretched by heatmap_grid) to its
        # exact content height just long enough to ask heatmap_box for its
        # real sizeHint (title bar + margins included, computed by the
        # actual style rather than estimated), then release the pin so
        # the viewport goes back to its normal flexible behaviour.
        main_h_px = int(main_fig.get_figheight() * dpi)
        self._heatmap_scroll.setFixedHeight(main_h_px)
        self._heatmap_box.adjustSize()
        ideal_top_h = self._heatmap_box.sizeHint().height()
        self._heatmap_scroll.setMinimumHeight(0)
        self._heatmap_scroll.setMaximumHeight(16_777_215)  # QWIDGETSIZE_MAX

        # Cap the heatmap pane at exactly its content height instead of
        # reading the splitter's CURRENT height (which is 0 the first
        # time this runs, before the tab has ever been shown, and which
        # otherwise grows the pane past its content on every later
        # window/splitter resize since both panes used to share stretch
        # factor 1). With the cap in place, requesting more than the pane
        # can take below just clamps to the cap -- stretch factor 0 (set
        # at splitter construction) keeps it pinned there afterward, and
        # ridge_box, the splitter's only stretch=1 pane, absorbs whatever
        # height the heatmap pane doesn't need.
        self._heatmap_box.setMaximumHeight(ideal_top_h)
        self._summary_splitter.setSizes([ideal_top_h, 1_000_000])

    def _transform_and_pool_matrix_values(self, pooled_by_channel: dict, channels: list[str],
                                          cluster_order: list[int]) -> dict[str, dict[int, np.ndarray]]:
        """
        Concatenate + transform each (channel, cluster) cell's pooled raw
        values exactly ONCE, shared between the heatmap matrix (median per
        cell) and the ridgeline histogram grid. The channel's transform is
        built once and applied to every cluster; cells larger than
        CLUSTER_PLOT_MAX_EVENTS are thinned first.
        """
        cap = CLUSTER_PLOT_MAX_EVENTS
        out: dict[str, dict[int, np.ndarray]] = {}
        for ch in channels:
            xform = _channel_xform(self.state, ch)
            by_cluster = pooled_by_channel.get(ch, {})
            out[ch] = {}
            for cl_id in cluster_order:
                vals = by_cluster.get(cl_id)
                if vals:
                    arr = np.concatenate(vals)
                    if cap is not None and len(arr) > cap:
                        keep = (np.arange(cap, dtype=np.int64) * len(arr)) // cap
                        arr = arr[keep]
                    out[ch][cl_id] = arr if xform is None else xform.apply(arr)
        return out
    
    def _compute_marker_heatmap_matrix(self, transformed: dict, channels: list[str],
                                       cluster_order: list[int]) -> np.ndarray:
        """
        median MFI per cluster x marker, on the TRANSFORMED
        (Transforms-tab) scale -- same convention the ridgeline grid
        uses, so both panels describe the same axis. Rows = clusters
        (same fixed order as the ridgeline grid), columns = markers.
        Median rather than mean so a handful of extreme events in a
        cluster can't swing a whole cell's colour. NaN where a
        cluster/marker combination has no pooled data.

        transformed: precomputed {channel: {cluster_id: transformed_array}}
        from _transform_and_pool_matrix_values -- shared with the
        ridgeline grid so the transform + concatenation only happens once.
        """
        mat = np.full((len(cluster_order), len(channels)), np.nan)
        for col, ch in enumerate(channels):
            by_cluster = transformed.get(ch, {})
            for row, cl_id in enumerate(cluster_order):
                arr = by_cluster.get(cl_id)
                if arr is not None:
                    mat[row, col] = float(np.median(arr))
        return mat

    def _make_marker_cluster_heatmap_figures(self, mat: np.ndarray, channels: list[str],
                                             cluster_order: list[int], names_map: dict):
        """
        Three figures sharing one DPI, built for frozen-header scrolling:
        main_fig (the heatmap image, no labels), col_fig (marker-name
        strip, same total WIDTH as main_fig), row_fig (cluster-name
        strip, same total HEIGHT as main_fig).

        main_fig and col_fig both reserve a fixed blank margin on the
        left and right of their data region (label_margin_in, identical
        in both figures) -- without it, a marker label anchored at the
        first or last column has nowhere to go but off the edge of the
        canvas. Adding the same margin to both figures keeps the
        pixels-per-column scale identical between them, so column i
        still lands at the same pixel offset in both (which is what
        lets one QScrollArea's scrollbars drive the other) -- it just
        means there's now a thin blank strip down each side of the
        heatmap image too.
        """
        from matplotlib.figure import Figure

        is_dark = _resolve_is_dark(self.state)
        labels = _antigen_dash_labels(self.controller)
        fg = 'white' if is_dark else 'black'

        n_channels = len(channels)
        n_clusters = len(cluster_order)
        dpi = 100
        core_w = max(5.0, 0.55 * n_channels)
        label_margin_in = 1.1
        main_w = core_w + 2 * label_margin_in

        # Long cluster names wrap onto two lines; the row strip is then
        # sized to the widest wrapped line, so no name is cut off.
        row_labels = [wrap_label(names_map.get(cl, str(cl)), 30, max_lines=2)
                      for cl in cluster_order]
        row_font = 8.0
        row_label_w = max((text_width_inches(t, row_font) for t in row_labels), default=0.0)
        max_row_header_w = 4.5
        if row_label_w + 0.3 > max_row_header_w:
            row_font = max(5.0, row_font * (max_row_header_w - 0.3) / row_label_w)
            row_label_w = max(text_width_inches(t, row_font) for t in row_labels)
        row_header_w = max(1.6, row_label_w + 0.3)
        row_h = 0.42 if any('\n' in t for t in row_labels) else 0.35
        main_h = max(4.0, row_h * n_clusters)
        col_header_h = 1.8
        # Horizontal placement shared by main_fig and col_fig -- same
        # left offset and same width fraction, so column i is at the
        # identical pixel x in both.
        data_rect_x = label_margin_in / main_w
        data_rect_w = core_w / main_w

        mat_filled = np.nan_to_num(mat, nan=0.0)

        main_fig = Figure(figsize=(main_w, main_h), dpi=dpi)
        ax = main_fig.add_axes([data_rect_x, 0, data_rect_w, 1])
        ax.imshow(mat_filled, aspect='auto', cmap='viridis')
        ax.set_xlim(-0.5, n_channels - 0.5)
        ax.set_ylim(n_clusters - 0.5, -0.5)
        ax.axis('off')
        ax.set_xticks(np.arange(-0.5, n_channels, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_clusters, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.6)
        ax.tick_params(which='minor', bottom=False, left=False)
        _style_figure_theme(main_fig, is_dark, axes=[ax])

        col_fig = Figure(figsize=(main_w, col_header_h), dpi=dpi)
        cax = col_fig.add_axes([data_rect_x, 0, data_rect_w, 1])
        cax.set_xlim(-0.5, n_channels - 0.5)
        cax.set_ylim(0, 1)
        cax.axis('off')
        for i, ch in enumerate(channels):
            # Anchor at the start of the label, at the column's centre
            cax.text(i, 0.05, labels.get(ch, ch), rotation=45, ha='left', va='bottom',
                     rotation_mode='anchor', fontsize=8, color=fg)
        _style_figure_theme(col_fig, is_dark)

        row_fig = Figure(figsize=(row_header_w, main_h), dpi=dpi)
        rax = row_fig.add_axes([0, 0, 1, 1])
        rax.set_ylim(n_clusters - 0.5, -0.5)
        rax.set_xlim(0, 1)
        rax.axis('off')
        label_x = 1.0 - 0.12 / row_header_w
        for row, text in enumerate(row_labels):
            rax.text(label_x, row, text, ha='right', va='center',
                     fontsize=row_font, color=fg, linespacing=1.1)
        _style_figure_theme(row_fig, is_dark)

        return main_fig, col_fig, row_fig

    def _make_marker_cluster_colorbar_figure(self, mat: np.ndarray):
        """Standalone colour-scale strip, separate from the (now
        label-free) heatmap figure so it never needs to scroll."""
        from matplotlib.figure import Figure
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        is_dark = _resolve_is_dark(self.state)
        mat_filled = np.nan_to_num(mat, nan=0.0)
        vmin = float(mat_filled.min()) if mat_filled.size else 0.0
        vmax = float(mat_filled.max()) if mat_filled.size else 1.0
        if vmin == vmax:
            vmax = vmin + 1.0
        sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap='viridis')
        sm.set_array([])

        fig = Figure(figsize=(4.0, 0.7), dpi=100, constrained_layout=True)
        ax = fig.add_subplot(111)
        fig.colorbar(sm, cax=ax, orientation='horizontal', label='Median transformed MFI')
        _style_figure_theme(fig, is_dark, axes=[ax])
        return fig

    def _make_marker_ridgeline_figure(self, transformed: dict, channels: list[str],
                                      cluster_order: list[int], names_map: dict,
                                      colors_map: dict):
        """
        CATALYST/diffcyt-style marker ridge grid — one small
        subplot per marker (channels, as columns), each containing every
        cluster's histogram stacked as a row, in the SAME order in every
        panel (cluster_order — see _recompute_marker_summary), coloured
        with the run's own per-cluster colours so identity matches the
        Cluster Map / legend elsewhere in this tab. First cluster in
        cluster_order is drawn at the TOP row.

        Uses a plain np.histogram per cluster instead of gaussian_kde --
        same approach as functions.calc_hist1d (the fast 1D histogram
        used elsewhere in Honeychrome): a single O(n) binning pass per
        cluster with a fixed bin count, no O(n x eval_points) kernel
        evaluation. Bin edges are shared across every cluster within a
        channel's panel (computed once from that channel's fixed
        transformed-scale limits, or from the pooled data's own range as
        a fallback) so the curves stay directly comparable and bins don't
        need recomputing per cluster.

        transformed: precomputed {channel: {cluster_id: transformed_array}}
        from _transform_and_pool_matrix_values -- shared with the heatmap
        matrix so the transform + concatenation only happens once.

        Each panel's ridges are one polygon collection and one line
        collection, and the ticks without labels (minor intensity ticks,
        cluster rows beyond the first column) are drawn as one marker line
        and one grid-line collection per panel rather than one Tick object
        each, so the grid stays quick to build and draw with many markers
        and clusters.
        """
        import matplotlib as mpl
        from matplotlib import markers as mmarkers
        from matplotlib.collections import LineCollection, PolyCollection
        from matplotlib.figure import Figure

        is_dark = _resolve_is_dark(self.state)
        fg = 'white' if is_dark else 'black'
        labels = _antigen_dash_labels(self.controller)
        rc = mpl.rcParams

        def _unlabelled_ticks(ax, positions, axis: str, which: str, length=None):
            """Tick marks, and grid lines where the style draws them, at
            *positions* on the 'x' or 'y' axis, styled as the axis' own
            *which* ('major'/'minor') ticks."""
            positions = np.asarray(positions, dtype=float)
            if not len(positions):
                return
            key = f'{axis}tick.{which}'
            size = rc[f'{key}.size'] if length is None else length
            inward = rc[f'{axis}tick.direction'] == 'in'
            if axis == 'x':
                transform = ax.get_xaxis_transform()
                xy = (positions, np.zeros(len(positions)))
                marker = mmarkers.TICKUP if inward else mmarkers.TICKDOWN
                segments = [[(p, 0.0), (p, 1.0)] for p in positions]
            else:
                transform = ax.get_yaxis_transform()
                xy = (np.zeros(len(positions)), positions)
                marker = mmarkers.TICKRIGHT if inward else mmarkers.TICKLEFT
                segments = [[(0.0, p), (1.0, p)] for p in positions]
            shown = rc['xtick.bottom'] if axis == 'x' else rc['ytick.left']
            if shown and size > 0:
                line, = ax.plot(*xy, transform=transform, linestyle='none', marker=marker,
                                markersize=size, markeredgewidth=rc[f'{key}.width'],
                                color=fg, clip_on=False)
                line.set_in_layout(False)
            if (rc['axes.grid'] and rc['axes.grid.axis'] in ('both', axis)
                    and rc['axes.grid.which'] in ('both', which)):
                below = rc['axes.axisbelow']
                zorder = 0.5 if below is True else (1.5 if below == 'line' else 2.5)
                ax.add_collection(LineCollection(
                    segments, transform=transform, colors=rc['grid.color'],
                    linewidths=rc['grid.linewidth'], linestyles=rc['grid.linestyle'],
                    alpha=rc['grid.alpha'], zorder=zorder), autolim=False)

        n = len(channels)
        n_cols = min(4, max(1, n))
        n_rows = -(-n // n_cols) if n else 1
        row_h = 0.55
        n_clusters = len(cluster_order)
        panel_h = max(3.0, n_clusters * row_h + 1.0)
        fig = Figure(figsize=(3.2 * n_cols, panel_h * n_rows), constrained_layout=True)

        N_BINS = 200

        for i, ch in enumerate(channels):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)

            by_cluster = transformed.get(ch, {})

            # Shared bin edges for every cluster in this panel -- prefer
            # the channel's fixed transformed-scale limits (same ones
            # used for the x-axis below) so bins line up with the axis
            # exactly; fall back to the pooled data's own range if no
            # transform tick spec is available for this channel.
            tick_spec = _channel_axis_ticks(self.state, ch)
            if tick_spec is not None:
                bin_lo, bin_hi = tick_spec[2][0], tick_spec[2][1]
            else:
                all_vals = [v[np.isfinite(v)] for v in by_cluster.values() if v is not None]
                all_vals = [v for v in all_vals if v.size]
                if all_vals:
                    combined = np.concatenate(all_vals)
                    bin_lo, bin_hi = float(combined.min()), float(combined.max())
                else:
                    bin_lo, bin_hi = 0.0, 1.0
            if bin_hi <= bin_lo:
                bin_hi = bin_lo + 1.0
            bin_edges = np.linspace(bin_lo, bin_hi, N_BINS + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            # Step outline through the bin centres, as drawstyle 'steps-mid'
            step_x = np.concatenate((
                [bin_centers[0]],
                np.repeat(0.5 * (bin_centers[:-1] + bin_centers[1:]), 2),
                [bin_centers[-1]],
            ))

            polygons, outlines, colours = [], [], []
            for row_idx, cl_id in enumerate(cluster_order):
                arr = by_cluster.get(cl_id)
                if arr is None:
                    continue
                arr = arr[np.isfinite(arr)]
                if len(arr) < 5 or np.ptp(arr) == 0:
                    continue
                counts, _ = np.histogram(arr, bins=bin_edges)
                counts = counts.astype(float)
                if counts.max() <= 0:
                    continue
                density = counts / counts.max() * 1.4   # ridge height, allows slight overlap
                # First cluster in cluster_order at the TOP -- baseline
                # counts down from n_clusters-1, not up from 0.
                baseline = n_clusters - 1 - row_idx
                step_y = baseline + np.repeat(density, 2)
                top = np.column_stack((step_x, step_y))
                polygons.append(np.vstack((
                    top, [[step_x[-1], baseline], [step_x[0], baseline]])))
                outlines.append(top)
                colours.append(colors_map.get(cl_id, '#888888'))
            if polygons:
                ax.add_collection(PolyCollection(polygons, facecolors=colours, alpha=0.75,
                                                 linewidths=0))
                ax.add_collection(LineCollection(outlines, colors=colours, linewidths=0.8))

            ax.set_title(labels.get(ch, ch), fontsize=9)
            ax.set_ylim(-0.3, n_clusters + 0.3)
            if i % n_cols == 0:
                ax.set_yticks(range(n_clusters))
                ax.set_yticklabels(
                    [names_map.get(cl, str(cl)) for cl in reversed(cluster_order)],
                    fontsize=7,
                )
            else:
                ax.set_yticks([])
                _unlabelled_ticks(ax, np.arange(n_clusters), 'y', 'major')
            ax.tick_params(labelsize=6)

            # Data stays on the transformed scale (arr, above) -- only the
            # tick labels switch to the original raw-scale numbers, same
            # convention as the Transforms tab's 2D histograms and the
            # violin y-axis (_channel_axis_ticks).
            if tick_spec is not None:
                major_ticks, minor_ticks, limits = tick_spec
                ax.set_xticks([pos for pos, _label in major_ticks])
                ax.set_xticklabels([label for _pos, label in major_ticks], fontsize=6)
                minor_x = np.array([pos for pos, _label in minor_ticks], dtype=float)
                lo, hi = min(limits), max(limits)
                _unlabelled_ticks(ax, minor_x[(minor_x >= lo) & (minor_x <= hi)], 'x', 'minor',
                                  length=2)
                ax.set_xlim(limits[0], limits[1])
                ax.set_xlabel('Intensity', fontsize=6)
            else:
                ax.set_xlabel('Transformed intensity', fontsize=6)

        _style_figure_theme(fig, is_dark)
        return fig

    def _make_marker_heatmap_export_figure(self, run_id: str):
        """
        Single self-contained heatmap (axis-labelled, with an integrated
        colour bar) for Report/PDF export -- unlike the on-screen frozen-
        header split (main/col/row/cbar figs) used by
        _apply_marker_summary_figures, which only makes sense inside the
        live scrolling GUI.
        """
        from matplotlib.figure import Figure

        cache = self._marker_summary_cache.get(run_id)
        if not cache or not cache.get('pooled') or not cache.get('cluster_order'):
            return None

        channels = cache['channels']
        cluster_order = cache['cluster_order']
        names_map = cache['names_map']
        transformed = self._transform_and_pool_matrix_values(cache['pooled'], channels, cluster_order)
        mat = self._compute_marker_heatmap_matrix(transformed, channels, cluster_order)

        is_dark = _resolve_is_dark(self.state)
        labels = _antigen_dash_labels(self.controller)
        n_channels = len(channels)
        n_clusters = len(cluster_order)

        fig = Figure(
            figsize=(max(6.0, 0.5 * n_channels + 2), max(4.0, 0.35 * n_clusters + 1)),
            constrained_layout=True,
        )
        gs = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.05)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])

        mat_filled = np.nan_to_num(mat, nan=0.0)
        im = ax.imshow(mat_filled, aspect='auto', cmap='viridis')
        ax.set_xticks(range(n_channels))
        ax.set_xticklabels([labels.get(ch, ch) for ch in channels], rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(n_clusters))
        ax.set_yticklabels([names_map.get(cl, str(cl)) for cl in cluster_order], fontsize=8)
        ax.set_title('Median MFI per Cluster (Transformed)', fontsize=11)
        fig.colorbar(im, cax=cax, label='Median transformed MFI')
        _style_figure_theme(fig, is_dark, axes=[ax])
        return fig

    def get_report_items(self) -> list:
        """
        drc_report.ReportItems for the currently selected clustering run:
        the Cluster Map, one violin plot per checked channel (from the
        already-pooled violin cache), the Marker Heatmap + Ridgeline (if
        Marker Summary has been computed for this run), and the Cluster
        Label table.
        """
        items = []
        cl_run = self._selected_cluster_run()
        if cl_run is None:
            return items
        run_id = self.run_combo.currentData()
        run_label = self.run_combo.currentText()

        items.append(drc_report.ReportItem(
            key='cluster_map',
            tab='Cluster Annotation',
            label=f"Cluster Map ({run_label})",
            get_figure=lambda: self._map_figure,
        ))

        violin_cache = self._violin_cache.get(run_id)
        if violin_cache and violin_cache.get('pooled'):
            labels_map = _antigen_dash_labels(self.controller)
            for ch in violin_cache['channels']:
                items.append(drc_report.ReportItem(
                    key=f'violin_{ch}',
                    tab='Cluster Annotation',
                    label=f"Violin: {labels_map.get(ch, ch)}",
                    get_figure=lambda ch=ch, cache=violin_cache: self._make_single_violin_figure(
                        ch, cache['pooled'].get(ch, {}), cache['names_map'], cache.get('colors_map', {}),
                    ),
                ))

        summary_cache = self._marker_summary_cache.get(run_id)
        if summary_cache and summary_cache.get('pooled'):
            items.append(drc_report.ReportItem(
                key='marker_heatmap',
                tab='Cluster Annotation',
                label=f"Marker Heatmap ({run_label})",
                get_figure=lambda rid=run_id: self._make_marker_heatmap_export_figure(rid),
            ))
            items.append(drc_report.ReportItem(
                key='marker_ridgeline',
                tab='Cluster Annotation',
                label=f"Marker Ridgeline ({run_label})",
                get_figure=lambda cache=summary_cache: cache['ridge_fig'],
            ))

        items.append(drc_report.ReportItem(
            key='cluster_label_table',
            tab='Cluster Annotation',
            label=f"Cluster Labels ({run_label})",
            get_tables=lambda: {'cluster_labels': drc_report.qtable_to_dataframe(self.label_table)},
        ))

        return items


class PluginWidget(QWidget):
    """
    Top-level plugin widget.

    Inserted as a tab in the Honeychrome main window by plugin_loaders.py.
    Owns the PipelineState and all six inner tabs (Transforms,
    Configuration, Cluster Annotation, Stats, Workspace, Report).

    The outer layer follows the exact pattern of the Honeychrome example
    plugin: a disabled-label shown when unmixing is not yet available,
    and a scrollable content widget shown when it is.

    Parameters
    ----------
    bus        Honeychrome EventBus (shared with the rest of the app).
    controller Main Honeychrome Controller.
    parent     Qt parent (the main window tab widget).
    """

    def __init__(self, bus=None, controller=None, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.controller = controller

        # Perform Qt-dependent imports now that we're on the main thread.
        # pyqtgraph and colorcet create Qt internals on first import; doing
        # so on the background loader thread produces QObject::setParent warnings.
        _ensure_qt_imports()

        # Silence noisy third-party warnings (deferred to main thread).
        _suppress_third_party_warnings()

        # Shared state — passed by reference to all inner tabs
        self.state = PipelineState()

        # {algorithm: run_id} — the archived DR run behind each live embedding
        # (state.embeddings); 'Apply to All Samples' extends this run. Saved
        # in the current_state bundle.
        self._active_dr_run_id: dict[str, str] = {}
        # Training settings and pool size of each session-trained DR model, for
        # the run-time calibration of later 'Apply to All Samples' runs
        self._dr_train_params: dict[str, dict] = {}
        self._dr_pool_size: dict[str, int] = {}

        # Cancelled workers (DR, clustering, run-time estimates) still
        # finishing an uninterruptible step; kept referenced until they end.
        self._detached_workers: list = []
        # Holds on the application's job gate for a DR or clustering run,
        # from before its time estimate until it ends
        self._run_tokens: dict = {}

        # QSettings instance — keyed per experiment in save_state/load_state
        self._qsettings = QSettings('honeychrome', _SETTINGS_APP_KEY)

        # Guard: suppress save_state during the load sequence (content_widget
        # becoming visible fires currentChanged → save before load completes).
        self._loading = False

        # Tracks whether this plugin's mode was active on the previous
        # modeChangeRequested — lets initialise_gui() save state when the
        # user leaves the plugin via the main app's tab bar, not just when
        # switching between the plugin's own inner tabs (see initialise_gui).
        self._plugin_was_active = False

        # Tracks which experiment's persisted state is currently loaded into
        # self.state. initialise_gui() only re-reads QSettings and rebuilds
        # the workspace when this changes — see initialise_gui() for why.
        # _loaded_experiment_dir is the matching experiment_dir, cached so
        # save_state() can still correctly flush self.state to where it
        # actually belongs even after self.controller has already switched
        # to a different experiment (see save_state()'s key-mismatch
        # handling).
        self._loaded_experiment_key = None
        self._loaded_experiment_dir = None

        self._last_tab_refresh_key: dict[int, tuple] = {}

        # ------------------------------------------------------------------
        # Outer layout: disabled label + scrollable content
        # ------------------------------------------------------------------
        self.label_disabled = QLabel(
            f"{plugin_name}: loading...  "
            "Loading...Set up the spectral model first if you have not done so."
        )
        self.label_disabled.setWordWrap(True)
        self.label_disabled.setStyleSheet(
            "QLabel { color: #856404; background: #fff3cd; "
            "border: 1px solid #ffc107; border-radius: 4px; padding: 8px; }"
        )

        # The content widget holds the inner tab widget inside a scroll area
        self.content_widget = QWidget()
        content_layout = QVBoxLayout(self.content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self.content_widget)

        overall_layout = QVBoxLayout(self)
        overall_layout.setContentsMargins(6, 6, 6, 6)
        overall_layout.addWidget(self.label_disabled)
        overall_layout.addWidget(scroll)

        # ------------------------------------------------------------------
        # Inner sections — vertical navigation beside a page stack
        # ------------------------------------------------------------------
        self.inner_tabs = _SideTabWidget()

        self.config_tab = ConfigTab(self.state, bus, controller)
        self.transform_tab = TransformTab(self.state, bus, controller)
        self.groups_stats_tab = GroupsStatsTab(self.state, bus, controller)
        self.groups_stats_tab.results_computed.connect(self.save_state)
        self.workspace_tab = WorkspaceTab(self.state, bus, controller)
        self.cluster_annotation_tab = ClusterAnnotationTab(self.state, bus, controller)
        # Built last -- holds references to the four tabs above so it can
        # enumerate their currently-rendered plots/tables (see
        # drc_report.ReportTab.get_report_items callers).
        self.report_tab = drc_report.ReportTab(
            self.state, bus, controller,
            workspace_tab=self.workspace_tab,
            groups_stats_tab=self.groups_stats_tab,
            cluster_annotation_tab=self.cluster_annotation_tab,
        )

        # Figures the Workspace can copy onto its pages
        self.workspace_tab.set_figure_sources({
            'Cluster Annotation': self.cluster_annotation_tab.get_report_items,
            'Stats': self.groups_stats_tab.get_report_items,
        })

        # Result figures are built on a worker thread; list them in the
        # report once they exist if it is the tab being shown.
        self.groups_stats_tab.results_drawn.connect(self._refresh_report_if_shown)

        # Run buttons greyed out while another heavy job runs in the application
        guard_while_busy(controller, self.config_tab.dr_run_btn, self.config_tab.dr_apply_btn,
                         self.config_tab.cl_run_btn, self.groups_stats_tab.run_stats_btn,
                         self.groups_stats_tab.pred_run_btn,
                         self.cluster_annotation_tab.compute_suggestions_btn)
        if getattr(controller, 'job_gate', None) is not None:
            controller.job_gate.changed.connect(self.groups_stats_tab.on_job_gate_changed)

        self.inner_tabs.addTab(self.transform_tab,    "Transforms",    'wave-sine')
        self.inner_tabs.addTab(self.config_tab,       "Configuration", 'settings')
        self.inner_tabs.addTab(self.cluster_annotation_tab, "Cluster Annotation", 'shape')
        self.inner_tabs.addTab(self.groups_stats_tab, "Stats",         'table')
        self.inner_tabs.addTab(self.workspace_tab,    "Workspace",     'border-all')
        self.inner_tabs.addTab(self.report_tab,       "Report",        'file-description')

        self.inner_tabs.currentChanged.connect(self._on_inner_tab_changed)

        # a rename/delete in the run management table must be
        # reflected immediately in every other run selector, not just on
        # next tab switch.
        self.config_tab.run_table.runsChanged.connect(self._on_runs_changed)

        # Keep the two gate trees (TransformTab's primary, ConfigTab's
        # override) in sync — same idea as the old combo pair's
        # blockSignals(), applied to tree checkbox state instead.
        self.transform_tab.gate_tree.selectionChanged.connect(self._on_gate_tree_changed)
        self.config_tab.gate_tree.selectionChanged.connect(self._on_gate_tree_changed)

        content_layout.addWidget(self.inner_tabs)

        # Start with content hidden until unmixing is confirmed available
        self.content_widget.setVisible(False)

        # Persist the workspace layout immediately on every add/remove, not
        # just when the user switches inner tabs — otherwise a deletion is
        # lost if the plugin tab (or the app) closes before that happens.
        self.workspace_tab.plots_changed.connect(self._save_plot_configs_only)

        # Same for the transforms: written shortly after the last change, so
        # a drag does not write on every mouse move.
        self._transform_save_timer = QTimer(self)
        self._transform_save_timer.setSingleShot(True)
        self._transform_save_timer.setInterval(500)
        self._transform_save_timer.timeout.connect(self._save_transforms_only)
        self.transform_tab.transformsPersistRequested.connect(
            lambda: self._transform_save_timer.start())

        # ------------------------------------------------------------------
        # Connect bus signals
        # ------------------------------------------------------------------
        self.bus.modeChangeRequested.connect(self.initialise_gui)
        self.bus.loadSampleRequested.connect(self.on_sample_selected)

    # ------------------------------------------------------------------
    # Bus signal handlers
    # ------------------------------------------------------------------

    def _on_runs_changed(self):
        """
        A run was renamed or deleted in ConfigTab's management table.
        Refresh every other place that lists runs so the change shows up immediately.
        """
        self._drop_orphaned_dr_embeddings()
        if hasattr(self, 'config_tab'):
            self.config_tab._refresh_dr_status()
            self.config_tab._refresh_cl_space_combo()
        if hasattr(self, 'groups_stats_tab'):
            self.groups_stats_tab._clear_stale_run_results()
            self.groups_stats_tab._populate_run_combo()
            self.groups_stats_tab._sync_composition_to_run()
            self.groups_stats_tab._update_run_button()
        if hasattr(self, 'workspace_tab'):
            for card in self.workspace_tab._plot_cards:
                card._populate_selectors()
        if hasattr(self, 'cluster_annotation_tab'):
            # Hidden: refresh on the next visit instead (the report reads its selection)
            current = self.inner_tabs.currentWidget() if hasattr(self, 'inner_tabs') else None
            if current is not None and current in (self.cluster_annotation_tab,
                                                   getattr(self, 'report_tab', None)):
                self.cluster_annotation_tab.refresh()
            else:
                self._last_tab_refresh_key.pop(2, None)
        if hasattr(self, 'report_tab'):
            self.report_tab.refresh()

    def initialise_gui(self, mode: str):
        """
        Called whenever any tab in Honeychrome is activated.
        Only acts when mode matches plugin_name.
        """
        if mode != plugin_name:
            # Leaving the plugin entirely (switching to a different
            # top-level Honeychrome tab).  Previously, settings were only
            # persisted on an inner-tab switch (_on_inner_tab_changed), so
            # configuring samples/channels/numbers and then switching
            # straight to another main-app tab silently dropped them.
            if self._plugin_was_active:
                self.save_state()
                self._plugin_was_active = False
            return

        self._plugin_was_active = True

        if self.controller.experiment.process.get('unmixing_matrix') is not None:
            self.label_disabled.setVisible(False)
            self.content_widget.setVisible(True)
            self._update_sample_browser_visibility()

            # Sync state with current experiment
            self._loading = True
            try:
                all_samples = _non_control_sample_paths(self.controller)
                self.state.initialise_sample_groups(all_samples)

                current_key = self._settings_key()
                if current_key != self._loaded_experiment_key:
                    # Only reload from QSettings / rebuild the workspace when
                    # we're switching to a different experiment.
                    self._last_tab_refresh_key = {}
                    # Claim the experiment BEFORE the restore work below.
                    # If any of it raises, save_state() must still know
                    # which experiment self.state belongs to -- otherwise
                    # it silently skips every subsequent save and the
                    # session becomes unsaveable as well as unrestorable.
                    self._loaded_experiment_key = current_key
                    self._loaded_experiment_dir = self.controller.experiment_dir
                    self.load_state()
                    if hasattr(self, 'workspace_tab'):
                        wt = self.workspace_tab
                        wt.theme_combo.setCurrentText(self.state.plot_theme.capitalize())
                        wt.restore_layout(getattr(self, '_pending_plot_configs', []),
                                          self.state.workspace_texts,
                                          self.state.workspace_figures)
                    if hasattr(self, 'groups_stats_tab'):
                        gst = self.groups_stats_tab
                        if hasattr(gst, 'chk_include_type_markers'):
                            gst.chk_include_type_markers.setChecked(
                                bool(getattr(self, '_pending_include_type_markers', False))
                            )
                self._sync_transform_state()
            finally:
                self._loading = False

            # Refresh the active inner tab
            self._refresh_active_tab()
            QTimer.singleShot(0, self._start_jit_warmup)
        else:
            self.label_disabled.setVisible(True)
            self.content_widget.setVisible(False)

    def _start_jit_warmup(self):
        """Compile UMAP's numba code in the background, once per session, on
        the code path the current n_jobs setting will use."""
        if not hasattr(self, 'config_tab'):
            return
        parallel = _umap_parallel({'n_jobs': self.config_tab.umap_n_jobs.value()})
        if drc_jit.start_warmup('UMAP', parallel):
            _log.info("started UMAP JIT warm-up (%s path)", 'parallel' if parallel else 'serial')

    # ------------------------------------------------------------------
    # State persistence (QSettings — keyed by experiment directory)
    # ------------------------------------------------------------------

    def _settings_key(self) -> str:
        """Return a QSettings group key unique to this experiment."""
        try:
            exp_dir = str(self.controller.experiment_dir)
        except Exception:
            exp_dir = 'unknown'
        # Sanitise to a safe settings key
        safe = exp_dir.replace('\\', '/').replace(':', '_').replace(' ', '_')
        return safe

    def _save_plot_configs_only(self):
        """
        Lightweight persistence for just the Workspace layout, fired
        immediately on every plot add/remove (see plots_changed connection
        in __init__). Deliberately skips the model sidecar bundle that the
        full save_state() writes (trained reducers, embeddings, cluster
        labels) — that would be far too expensive to redo on every click.
        """
        if self._loading:
            return
        key = self._settings_key()
        s = self._qsettings
        s.beginGroup(key)
        try:
            import json
            s.setValue('plot_configs', json.dumps(self.state.plot_configs))
            self._save_workspace_settings(s)
        finally:
            s.endGroup()

    def _save_workspace_settings(self, s):
        """Workspace pages and grid, into the open QSettings group *s*."""
        import json
        s.setValue('workspace_pages', json.dumps(self.state.workspace_pages))
        s.setValue('workspace_grid_mm', self.state.workspace_grid_mm)
        s.setValue('workspace_snap', self.state.workspace_snap)
        s.setValue('workspace_show_grid', self.state.workspace_show_grid)
        s.setValue('workspace_texts', json.dumps(self.state.workspace_texts))
        s.setValue('workspace_figures', json.dumps(self.state.workspace_figures))

    def _save_transforms_only(self):
        """
        Persist just the saved transforms, to the experiment the state was
        loaded for (see save_state).
        """
        if self._loading or self._loaded_experiment_key is None:
            return
        if not self.state.saved_transforms:
            return
        import json
        s = self._qsettings
        s.beginGroup(self._loaded_experiment_key)
        try:
            s.setValue('saved_transforms', json.dumps(self.state.saved_transforms))
        finally:
            s.endGroup()

    def _sync_transform_state(self):
        """
        Make the saved transforms and channel_transform_params consistent
        for the open experiment.  An experiment with no saved transforms
        starts from the ones defined in Unmixed Data and keeps its own from
        then on.
        """
        if not self.state.saved_transforms:
            src = self.controller.unmixed_transformations or {}
            self.state.saved_transforms = {
                ch: _transform_params(tr) for ch, tr in src.items()}
            if self.state.saved_transforms:
                self._transform_save_timer.start()
        self.state.channel_transform_params = _working_channel_transform_params(
            self.controller, self.state.saved_transforms)

    def save_state(self):
        """
        Persist the plugin's PipelineState to QSettings so the user can
        pick up where they left off next time this experiment is opened.

        Lightweight scalar/list values go to QSettings.
        Embeddings, cluster labels and stats results are written to a
        sidecar bundle in the experiment's cache folder (avoids QSettings
        size limits); see _save_model_sidecar.

        self.controller.experiment can already point at a DIFFERENT
        experiment than the one self.state actually belongs to (e.g.
        opening a new experiment fires a mode-change that reaches
        initialise_gui()'s "leaving the plugin" branch, which calls this,
        before this plugin's own load_state() has run for the new
        experiment). Recomputing the destination live in that situation
        would silently write the OLD experiment's results into the NEW
        one's QSettings group and sidecar file. Instead, flush to the
        cached key/dir this state was actually loaded under -- this is
        still synchronous and completes before the caller moves on, so no
        delay or user-facing message is needed, just the right target.
        """
        if self._loading:
            return
        if self._loaded_experiment_key is None:
            # This instance has never successfully run load_state() for ANY
            # experiment yet (e.g. a freshly (re)constructed PluginWidget,
            # right after an experiment reopen, before the user has even
            # navigated into this tab). self.state is still just
            # constructor defaults at this point -- saving now would
            # overwrite a real, previously-saved file with blanks, which is
            # exactly what was happening. Nothing meaningful to save yet.
            _log.info(
                "save_state: skipping -- no experiment has been loaded into "
                "this instance yet (self.state is still default-constructed)."
            )
            return
        current_key = self._settings_key()
        if current_key != self._loaded_experiment_key:
            _log.info(
                "save_state: controller has switched to %r, but state.* still "
                "belongs to %r -- flushing there instead of the live (wrong) target.",
                current_key, self._loaded_experiment_key,
            )
            key = self._loaded_experiment_key
            sidecar_dir = self._loaded_experiment_dir
        else:
            key = current_key
            sidecar_dir = self.controller.experiment_dir
        s = self._qsettings
        s.beginGroup(key)
        try:
            # Guard against clobbering a good saved selection with a
            # transiently empty one -- same reasoning as
            # config_channels_checked below.
            if self.state.selected_gates:
                s.setValue('selected_gates', list(self.state.selected_gates))
            if self.state.selected_channels:
                s.setValue('selected_channels', self.state.selected_channels)
            s.setValue('n_training_events', self.state.n_training_events)
            s.setValue('training_samples',  self.state.training_sample_ids)
            s.setValue('sample_groups',     repr(self.state.sample_groups))
            s.setValue('group_names',       list(self.state.group_names))
            s.setValue('group_patterns',    repr(self.state.group_patterns))
            s.setValue('group_colors',      repr(self.state.group_colors))
            # Sample PCA appearance/config
            s.setValue('pca_use_freq',      self.state.pca_use_freq)
            s.setValue('pca_use_counts',    self.state.pca_use_counts)
            s.setValue('pca_use_mfi',       self.state.pca_use_mfi)
            s.setValue('pca_show_loadings', self.state.pca_show_loadings)
            s.setValue('pca_n_loadings',    self.state.pca_n_loadings)
            s.setValue('pca_point_size',    self.state.pca_point_size)
            s.setValue('pca_arrow_lw',      self.state.pca_arrow_lw)
            s.setValue('pca_arrow_color',   self.state.pca_arrow_color)
            s.setValue('pca_axis_fontsize', self.state.pca_axis_fontsize)
            s.setValue('pca_show_grid',     self.state.pca_show_grid)
            s.setValue('pca_label_loadings', self.state.pca_label_loadings)
            s.setValue('pca_label_points',   self.state.pca_label_points)
            # stats and covariates
            s.setValue('testing_group_selection', list(self.state.testing_group_selection))
            s.setValue('contrast_mode',     self.state.contrast_mode)
            s.setValue('reference_group',   self.state.reference_group)
            s.setValue('paired',            self.state.paired)
            s.setValue('pairing_variable',  self.state.pairing_variable)
            s.setValue('stats_adjust_covariates', list(self.state.stats_adjust_covariates))
            s.setValue('numeric_covariates', list(self.state.numeric_covariates))
            s.setValue('stats_use_treat',   self.state.stats_use_treat)
            s.setValue('stats_counts_method', self.state.stats_counts_method)
            s.setValue('stats_normalisation', self.state.stats_normalisation)
            s.setValue('stats_mfi_threshold', self.state.stats_mfi_threshold)
            s.setValue('stats_mfi_hierarchical', self.state.stats_mfi_hierarchical)
            s.setValue('stats_fdr_scope', self.state.stats_fdr_scope)
            s.setValue('stats_pval_threshold', self.state.stats_pval_threshold)
            s.setValue('stats_fc_threshold', self.state.stats_fc_threshold)
            s.setValue('covariate_columns',
                      list(self.state.covariates.columns)
                      if self.state.covariates is not None else [])
            s.setValue('covariates',
                      repr(self.state.covariates.to_dict(orient='index'))
                      if self.state.covariates is not None else '')
            # Honeycluster status (lightweight — no arrays)
            s.setValue('dr_status',         repr({
                algo: ('idle' if status == 'running' else status)
                for algo, status in self.state.dr_status.items()}))
            s.setValue('dr_timestamps',     repr(self.state.dr_timestamps))
            s.setValue('n_clusters',        self.state.n_clusters if self.state.n_clusters is not None else '')
            s.setValue('active_cl_algo',    self.state.active_clustering_algorithm or '')
            s.setValue('cluster_colors',    repr(self.state.cluster_colors))
            self._save_workspace_settings(s)
            s.setValue('plot_theme', self.state.plot_theme)
            # plot_configs: only save serialisable display settings (no widgets)
            try:
                import json
                s.setValue('plot_configs', json.dumps(self.state.plot_configs))
            except Exception:
                pass
            s.setValue('cluster_names', repr(self.state.cluster_names))
            s.setValue('marker_roles', repr(self.state.marker_roles))
            if hasattr(self, 'cluster_annotation_tab'):
                cat = self.cluster_annotation_tab
                # Guard against clobbering a good saved selection with the
                # tab's un-populated defaults. run_combo/dr_run_combo are
                # only filled in by _populate_run_combo(), which only runs
                # inside ClusterAnnotationTab.refresh() -- i.e. only once
                # the user has actually visited this tab this session (see
                # _refresh_tab_at). Until then count() is 0, so these
                # fields must not be overwritten.
                if cat.run_combo.count() > 0:
                    s.setValue('annotation_run_id', cat.run_combo.currentData() or '')
                    s.setValue('annotation_dr_run_id', cat.dr_run_combo.currentData() or '')
                    s.setValue('annotation_channels_checked', cat._checked_channels())
                    if hasattr(cat, 'species_combo'):
                        s.setValue('cluster_id_species', cat.species_combo.currentData() or 'human')
            # Groups tab: include-type-markers checkbox
            if hasattr(self, 'groups_stats_tab'):
                gst = self.groups_stats_tab
                if hasattr(gst, 'chk_include_type_markers'):
                    s.setValue('include_type_markers',
                              gst.chk_include_type_markers.isChecked())
            # Config tab: channel checkboxes and training sample picker.
            # Gate selection itself is already covered by the top-level
            # 'selected_gates' key above — both trees read from
            # state.selected_gates directly on refresh(), so there is no
            # separate per-tab gate key to save.
            # Guard against clobbering a good saved selection with an empty
            # one. ConfigTab.refresh() builds these widgets lazily (only
            # once the tab becomes active this session), but
            # _on_inner_tab_changed() calls save_state() before refreshing
            # the newly-activated tab — so on the first switch into
            # Configuration each session, channel_checkboxes/picker can
            # still be empty here even though a real selection already
            # exists on disk. Only overwrite when there's something to save.
            if getattr(self.config_tab, 'channel_checkboxes', None):
                checked = {ch: cb.isChecked()
                           for ch, cb in self.config_tab.channel_checkboxes.items()}
                s.setValue('config_channels_checked', repr(checked))
            if hasattr(self.config_tab, 'include_unstained_chk'):
                s.setValue('config_training_include_unstained',
                           self.config_tab.include_unstained_chk.isChecked())
                s.setValue('config_training_include_single_stain',
                           self.config_tab.include_single_stain_chk.isChecked())
                s.setValue('config_training_include_beads',
                           self.config_tab.include_beads_chk.isChecked())
            if hasattr(self.config_tab, 'picker') and self.config_tab.picker.get_ordered_list():
                s.setValue('config_training_samples',
                           self.config_tab.picker.get_ordered_list())
            # Transform tab: biplot tile y-channels
            if hasattr(self.transform_tab, '_biplot_tiles'):
                tile_y = [t.y_channel() for t in self.transform_tab._biplot_tiles]
                s.setValue('transform_biplot_y_channels', tile_y)
                s.setValue('transform_n_tiles', len(tile_y))
            if self.state.saved_transforms:
                import json
                s.setValue('saved_transforms', json.dumps(self.state.saved_transforms))
            # Config tab: DR algorithm and hyperparameters
            ct = self.config_tab
            if hasattr(ct, '_dr_algo_group'):
                btn = ct._dr_algo_group.checkedButton()
                if btn:
                    s.setValue('dr_algo', btn.text())
            s.setValue('dr_defaults_version', DR_DEFAULTS_VERSION)
            if hasattr(ct, 'umap_n_neighbors'):
                s.setValue('umap_n_neighbors', ct.umap_n_neighbors.value())
                s.setValue('umap_min_dist',    ct.umap_min_dist.value())
                s.setValue('umap_metric',      ct.umap_metric.currentText())
                s.setValue('umap_n_epochs',    ct.umap_n_epochs.value())
                s.setValue('umap_n_jobs',      ct.umap_n_jobs.value())
            if hasattr(ct, 'tsne_perplexity'):
                s.setValue('tsne_perplexity', ct.tsne_perplexity.value())
                s.setValue('tsne_n_iter',     ct.tsne_n_iter.value())
                s.setValue('tsne_n_jobs',     ct.tsne_n_jobs.value())
            if hasattr(ct, 'pacmap_n_neighbors'):
                s.setValue('pacmap_n_neighbors', ct.pacmap_n_neighbors.value())
                s.setValue('pacmap_mn_ratio',    ct.pacmap_mn_ratio.value())
                s.setValue('pacmap_fp_ratio',    ct.pacmap_fp_ratio.value())
            # Config tab: Clustering algorithm and hyperparameters
            if hasattr(ct, '_cl_algo_group'):
                btn = ct._cl_algo_group.checkedButton()
                if btn:
                    s.setValue('cl_algo', btn.text())
            if hasattr(ct, 'flowsom_xdim'):
                s.setValue('flowsom_xdim',         ct.flowsom_xdim.value())
                s.setValue('flowsom_ydim',         ct.flowsom_ydim.value())
                s.setValue('flowsom_metaclusters', ct.flowsom_metaclusters.value())
                s.setValue('flowsom_n_iter',       ct.flowsom_n_iter.value())
            if hasattr(ct, 'leiden_resolution'):
                s.setValue('leiden_resolution',  ct.leiden_resolution.value())
                s.setValue('leiden_n_neighbors', ct.leiden_n_neighbors.value())
            if hasattr(ct, 'hdbscan_min_cluster_size'):
                s.setValue('hdbscan_min_cluster_size',
                          ct.hdbscan_min_cluster_size.value())
                s.setValue('hdbscan_min_samples',
                          ct.hdbscan_min_samples.value())
                s.setValue('hdbscan_cluster_selection_epsilon',
                          ct.hdbscan_cluster_selection_epsilon.value())
            if hasattr(ct, 'cl_assign_all_chk'):
                s.setValue('cl_assign_all_samples', ct.cl_assign_all_chk.isChecked())
            if hasattr(ct, 'cl_downsample_chk'):
                s.setValue('cl_downsample_training', ct.cl_downsample_chk.isChecked())

            # temporary diagnostic
            _log.info(
                "save_state[%s] SUMMARY: gates=%r channels=%d training=%r "
                "n_clusters=%r algo=%r",
                key, list(self.state.selected_gates),
                len(self.state.selected_channels),
                list(self.state.training_sample_ids),
                self.state.n_clusters, self.state.active_clustering_algorithm,
            )
        finally:
            s.endGroup()

        # Heavy state (embeddings, cluster labels, stats) to the sidecar bundle
        self._save_model_sidecar(sidecar_dir)
        print(f"[DR Plugin] State saved for experiment: {key}")

    def load_state(self):
        """
        Restore persisted PipelineState from QSettings.
        Called once per experiment load, before tabs are refreshed.
        """
        key = self._settings_key()
        s = self._qsettings
        s.beginGroup(key)
        try:
            gates = _qs_list(s, 'selected_gates')
            if gates:
                self.state.selected_gates = gates
            else:
                # Pre-multi-gate settings, saved under the old singular key.
                legacy_gate = s.value('selected_gate', '')
                if legacy_gate:
                    self.state.selected_gates = [legacy_gate]

            channels = _qs_list(s, 'selected_channels')
            if channels:
                self.state.selected_channels = channels
            _log.info("load_state[%s]: selected_gates restored=%r, "
                      "selected_channels restored=%d (raw value had %d)",
                      key, self.state.selected_gates,
                      len(self.state.selected_channels), len(channels))

            n_ev = s.value('n_training_events', None)
            if n_ev is not None:
                try:
                    self.state.n_training_events = int(n_ev)
                except (ValueError, TypeError):
                    pass

            training = _qs_list(s, 'training_samples')
            if training:
                self.state.training_sample_ids = training
            _log.info("load_state[%s]: training_samples restored=%r",
                      key, self.state.training_sample_ids)

            groups_repr = s.value('sample_groups', '')
            group_names_stored = _qs_list(s, 'group_names')
            loaded_groups = None
            if groups_repr:
                try:
                    loaded_groups = eval(groups_repr)  # noqa: S307
                except Exception:
                    loaded_groups = None

            # one-time migration for development.
            is_legacy = (
                loaded_groups is not None
                and len(group_names_stored) == 2
                and all(g in ('A', 'B', 'Unassigned') for g in loaded_groups.values())
            )

            if is_legacy:
                name_a, name_b = group_names_stored
                slot_to_name = {'A': name_a, 'B': name_b}
                self.state.sample_groups = {
                    sp: slot_to_name.get(g, 'Unassigned') for sp, g in loaded_groups.items()
                }
                self.state.group_names = [name_a, name_b]
                legacy_pat_a = s.value('group_a_pattern', '')
                legacy_pat_b = s.value('group_b_pattern', '')
                self.state.group_patterns = {
                    k: v for k, v in ((name_a, legacy_pat_a), (name_b, legacy_pat_b)) if v
                }
                _log.info(
                    "sample_groups: migrated %d entries from legacy A/B slot "
                    "format to named groups %r", len(loaded_groups), self.state.group_names
                )
            elif loaded_groups is not None:
                if group_names_stored:
                    self.state.group_names = group_names_stored
                valid = set(self.state.group_names) | {'Unassigned'}
                self.state.sample_groups = {
                    sp: (g if g in valid else 'Unassigned') for sp, g in loaded_groups.items()
                }
                bad = [sp for sp, g in loaded_groups.items() if g not in valid]
                if bad:
                    _log.warning(
                        "sample_groups: reset %d entries with an unrecognised "
                        "group name to 'Unassigned': %s", len(bad), bad
                    )
                patterns_repr = s.value('group_patterns', '')
                if patterns_repr:
                    try:
                        self.state.group_patterns = eval(patterns_repr)  # noqa: S307
                    except Exception:
                        pass
                colors_repr = s.value('group_colors', '')
                if colors_repr:
                    try:
                        self.state.group_colors = eval(colors_repr)  # noqa: S307
                    except Exception:
                        pass

            self.state.testing_group_selection = _qs_list(s, 'testing_group_selection')
            self.state.contrast_mode = s.value('contrast_mode', 'reference') or 'reference'
            self.state.reference_group = s.value('reference_group', '')
            paired_val = s.value('paired', False)
            self.state.paired = paired_val in (True, 'true', 'True', 1, '1')
            self.state.pairing_variable = s.value('pairing_variable', '')
            self.state.stats_adjust_covariates = _qs_list(s, 'stats_adjust_covariates')
            self.state.numeric_covariates = _qs_list(s, 'numeric_covariates')
            use_treat = s.value('stats_use_treat', None)
            if use_treat is not None:
                self.state.stats_use_treat = use_treat in (True, 'true', 'True', 1, '1')
            mfi_threshold = s.value('stats_mfi_threshold', None)
            if mfi_threshold is not None:
                try:
                    self.state.stats_mfi_threshold = float(mfi_threshold)
                except (ValueError, TypeError):
                    pass
            mfi_hier = s.value('stats_mfi_hierarchical', None)
            if mfi_hier is not None:
                self.state.stats_mfi_hierarchical = mfi_hier in (True, 'true', 'True', 1, '1')
            fdr_scope = s.value('stats_fdr_scope', None)
            if fdr_scope in ('global', 'per_comparison'):
                self.state.stats_fdr_scope = fdr_scope
            counts_method = s.value('stats_counts_method', None)
            if counts_method in ('wald', 'ql'):
                self.state.stats_counts_method = counts_method
            normalisation = s.value('stats_normalisation', None)
            if normalisation in ('total', 'robust'):
                self.state.stats_normalisation = normalisation
            for qs_key, attr in (('stats_pval_threshold', 'stats_pval_threshold'),
                                 ('stats_fc_threshold', 'stats_fc_threshold')):
                stored = s.value(qs_key, None)
                if stored is not None:
                    try:
                        setattr(self.state, attr, float(stored))
                    except (ValueError, TypeError):
                        pass
            pca_use_freq = s.value('pca_use_freq', None)
            if pca_use_freq is not None:
                self.state.pca_use_freq = pca_use_freq in (True, 'true', 'True', 1, '1')
            pca_use_counts = s.value('pca_use_counts', None)
            if pca_use_counts is not None:
                self.state.pca_use_counts = pca_use_counts in (True, 'true', 'True', 1, '1')
            pca_use_mfi = s.value('pca_use_mfi', None)
            if pca_use_mfi is not None:
                self.state.pca_use_mfi = pca_use_mfi in (True, 'true', 'True', 1, '1')
            pca_show_loadings = s.value('pca_show_loadings', None)
            if pca_show_loadings is not None:
                self.state.pca_show_loadings = pca_show_loadings in (True, 'true', 'True', 1, '1')
            pca_n_loadings = s.value('pca_n_loadings', None)
            if pca_n_loadings is not None:
                try:
                    self.state.pca_n_loadings = int(pca_n_loadings)
                except (ValueError, TypeError):
                    pass
            pca_point_size = s.value('pca_point_size', None)
            if pca_point_size is not None:
                try:
                    self.state.pca_point_size = float(pca_point_size)
                except (ValueError, TypeError):
                    pass
            pca_arrow_lw = s.value('pca_arrow_lw', None)
            if pca_arrow_lw is not None:
                try:
                    self.state.pca_arrow_lw = float(pca_arrow_lw)
                except (ValueError, TypeError):
                    pass
            self.state.pca_arrow_color = s.value('pca_arrow_color', self.state.pca_arrow_color)
            pca_axis_fontsize = s.value('pca_axis_fontsize', None)
            if pca_axis_fontsize is not None:
                try:
                    self.state.pca_axis_fontsize = int(pca_axis_fontsize)
                except (ValueError, TypeError):
                    pass
            pca_show_grid = s.value('pca_show_grid', None)
            if pca_show_grid is not None:
                self.state.pca_show_grid = pca_show_grid in (True, 'true', 'True', 1, '1')
            pca_label_loadings = s.value('pca_label_loadings', None)
            if pca_label_loadings is not None:
                self.state.pca_label_loadings = pca_label_loadings in (True, 'true', 'True', 1, '1')
            pca_label_points = s.value('pca_label_points', None)
            if pca_label_points is not None:
                self.state.pca_label_points = pca_label_points in (True, 'true', 'True', 1, '1')
            cov_cols = _qs_list(s, 'covariate_columns')
            cov_repr = s.value('covariates', '')
            if cov_cols:
                try:
                    loaded_cov = eval(cov_repr) if cov_repr else {}  # noqa: S307
                except Exception:
                    loaded_cov = {}
                cov_df = pd.DataFrame.from_dict(loaded_cov, orient='index')
                for c in cov_cols:
                    if c not in cov_df.columns:
                        cov_df[c] = ''
                # Restore the original column order (to_dict/from_dict does
                # not guarantee it) and drop anything not in cov_cols --
                # e.g. a column whose group was since removed elsewhere.
                self.state.covariates = cov_df[cov_cols]
            elif cov_repr:
                try:
                    loaded_cov = eval(cov_repr)  # noqa: S307
                    self.state.covariates = (
                        pd.DataFrame.from_dict(loaded_cov, orient='index') if loaded_cov else None
                    )
                except Exception:
                    self.state.covariates = None
            else:
                self.state.covariates = None

            # Honeycluster status metadata
            dr_status_repr = s.value('dr_status', '')
            if dr_status_repr:
                try:
                    self.state.dr_status = eval(dr_status_repr)  # noqa: S307
                except Exception:
                    pass
                # A run cannot outlive the session it was started in
                self.state.dr_status = {
                    algo: ('idle' if status == 'running' else status)
                    for algo, status in self.state.dr_status.items()
                }

            dr_ts_repr = s.value('dr_timestamps', '')
            if dr_ts_repr:
                try:
                    self.state.dr_timestamps = eval(dr_ts_repr)  # noqa: S307
                except Exception:
                    pass

            n_cl = s.value('n_clusters', '')
            if n_cl != '':
                try:
                    self.state.n_clusters = int(n_cl)
                except (ValueError, TypeError):
                    pass

            active_cl = s.value('active_cl_algo', '')
            if active_cl:
                self.state.active_clustering_algorithm = active_cl

            colors_repr = s.value('cluster_colors', '')
            if colors_repr:
                try:
                    loaded = eval(colors_repr)  # noqa: S307
                    self.state.cluster_colors = {int(k): v for k, v in loaded.items()}
                except Exception:
                    pass

            try:
                import json
                pages = json.loads(s.value('workspace_pages', '') or '[]')
                self.state.workspace_pages = [
                    {'preset': str(p.get('preset', 'Custom')),
                     'w_mm': float(p['w_mm']), 'h_mm': float(p['h_mm'])} for p in pages]
            except (ValueError, TypeError, KeyError, AttributeError):
                self.state.workspace_pages = []
            try:
                self.state.workspace_grid_mm = float(s.value(
                    'workspace_grid_mm', drc_workspace_canvas.DEFAULT_GRID_MM))
            except (ValueError, TypeError):
                self.state.workspace_grid_mm = drc_workspace_canvas.DEFAULT_GRID_MM
            self.state.workspace_snap = _qs_bool(s, 'workspace_snap', True)
            self.state.workspace_show_grid = _qs_bool(s, 'workspace_show_grid', True)
            try:
                texts = json.loads(s.value('workspace_texts', '') or '[]')
                self.state.workspace_texts = [t for t in texts if isinstance(t, dict)]
            except (ValueError, TypeError):
                self.state.workspace_texts = []
            try:
                figures = json.loads(s.value('workspace_figures', '') or '[]')
                self.state.workspace_figures = [f for f in figures if isinstance(f, dict)]
            except (ValueError, TypeError):
                self.state.workspace_figures = []

            theme = s.value('plot_theme', None)
            if theme in ('auto', 'light', 'dark'):
                self.state.plot_theme = theme

            plot_configs_json = s.value('plot_configs', '')
            if plot_configs_json:
                try:
                    import json
                    self._pending_plot_configs = json.loads(plot_configs_json)
                except Exception:
                    self._pending_plot_configs = []
            else:
                self._pending_plot_configs = []

            names_repr = s.value('cluster_names', '')
            if names_repr:
                try:
                    loaded = eval(names_repr)  # noqa: S307
                    self.state.cluster_names = {int(k): v for k, v in loaded.items()}
                except Exception:
                    pass

            roles_repr = s.value('marker_roles', '')
            if roles_repr:
                try:
                    loaded = eval(roles_repr)  # noqa: S307
                    self.state.marker_roles = {str(k): v for k, v in loaded.items()}
                except Exception:
                    pass

            self._pending_include_type_markers = s.value('include_type_markers', True)
            self._pending_annotation_run_id = s.value('annotation_run_id', '')
            self._pending_annotation_dr_run_id = s.value('annotation_dr_run_id', '')
            self._pending_annotation_channels = _qs_list(
                s, 'annotation_channels_checked')
            self._pending_cluster_id_species = s.value('cluster_id_species', 'human')

            # Config tab: restore channel and training-sample selections
            # (will be applied when ConfigTab.refresh() runs).  Gate
            # selection needs no pending-state variable — ConfigTab.refresh()
            # and TransformTab.refresh() both read state.selected_gates
            # (restored above) directly.
            self._pending_channels_checked = s.value('config_channels_checked', '')
            self._pending_training_samples = _qs_list(s, 'config_training_samples')
            self._pending_training_include = {
                'unstained': _qs_bool(s, 'config_training_include_unstained', False),
                'single_stain': _qs_bool(s, 'config_training_include_single_stain', False),
                'beads': _qs_bool(s, 'config_training_include_beads', False),
            }
            # Transform tab: restore biplot tile count and y-channels
            self._pending_tile_y_channels = _qs_list(
                s, 'transform_biplot_y_channels')
            self._pending_n_tiles = _qs_int(s, 'transform_n_tiles', 0) or 0
            # Transform tab: the transforms as last adjusted (empty for an
            # experiment not seen before, which then starts from Unmixed Data)
            try:
                import json
                loaded_transforms = json.loads(s.value('saved_transforms', '') or '{}')
            except Exception:
                loaded_transforms = {}
            self.state.saved_transforms = (
                loaded_transforms if isinstance(loaded_transforms, dict) else {})
            # DR algorithm and hyperparameters
            self._pending_dr_algo          = s.value('dr_algo', '')
            self._pending_umap_n_neighbors = s.value('umap_n_neighbors', None)
            self._pending_umap_min_dist    = s.value('umap_min_dist', None)
            self._pending_umap_metric      = s.value('umap_metric', '')
            self._pending_umap_n_epochs    = s.value('umap_n_epochs', None)
            # Settings saved before the defaults changed hold the old
            # defaults (UMAP 500 epochs, tSNE 1000 iterations): use the new ones
            old_defaults = _qs_int(s, 'dr_defaults_version', 1) < DR_DEFAULTS_VERSION
            self._pending_umap_n_jobs      = s.value('umap_n_jobs', None)
            self._pending_tsne_perplexity  = s.value('tsne_perplexity', None)
            self._pending_tsne_n_iter      = s.value('tsne_n_iter', None)
            if old_defaults:
                if str(self._pending_umap_n_epochs) == '500':
                    self._pending_umap_n_epochs = None
                if str(self._pending_tsne_n_iter) == '1000':
                    self._pending_tsne_n_iter = None
            self._pending_tsne_n_jobs      = s.value('tsne_n_jobs', None)
            self._pending_pacmap_n_neighbors = s.value('pacmap_n_neighbors', None)
            self._pending_pacmap_mn_ratio  = s.value('pacmap_mn_ratio', None)
            self._pending_pacmap_fp_ratio  = s.value('pacmap_fp_ratio', None)
            # Clustering algorithm and hyperparameters
            self._pending_cl_algo             = s.value('cl_algo', '')
            self._pending_flowsom_xdim        = s.value('flowsom_xdim', None)
            self._pending_flowsom_ydim        = s.value('flowsom_ydim', None)
            self._pending_flowsom_metaclusters = s.value('flowsom_metaclusters', None)
            self._pending_flowsom_n_iter      = s.value('flowsom_n_iter', None)
            self._pending_leiden_resolution   = s.value('leiden_resolution', None)
            self._pending_leiden_n_neighbors  = s.value('leiden_n_neighbors', None)
            self._pending_hdbscan_min_cluster_size = s.value('hdbscan_min_cluster_size', None)
            self._pending_hdbscan_min_samples      = s.value('hdbscan_min_samples', None)
            self._pending_hdbscan_cluster_selection_epsilon = s.value(
                'hdbscan_cluster_selection_epsilon', None)
            self._pending_cl_assign_all_samples = _qs_bool(
                s, 'cl_assign_all_samples', None)
            self._pending_cl_downsample_training = _qs_bool(
                s, 'cl_downsample_training', None)

        except Exception:
            # A single unreadable settings key must not cost the user the
            # entire restore. The sidecar below holds the expensive state
            # (embeddings, cluster labels, stats results) and the run
            # manifest, so it runs regardless.
            _log.exception(
                "load_state[%s]: settings restore failed part-way; "
                "continuing to the model sidecar with whatever was read.",
                key,
            )
        finally:
            s.endGroup()

        # temporary diagnostic
        _log.info(
            "load_state[%s] SUMMARY: gates=%r channels=%d training=%r "
            "groups=%d group_names=%r cov_cols=%r runs_pending",
            key, self.state.selected_gates, len(self.state.selected_channels),
            self.state.training_sample_ids, len(self.state.sample_groups),
            self.state.group_names,
            list(self.state.covariates.columns)
            if self.state.covariates is not None else None,
        )
        # Restore heavy state (models, embeddings, cluster labels) from sidecar
        self._load_model_sidecar()
        self._drop_orphaned_dr_embeddings()
        print(f"[DR Plugin] State loaded for experiment: {key}")

        # Re-evaluate the Run Stats button now that cluster_labels are restored.
        # GroupsStatsTab.refresh() is lazy (only fires on tab switch), so the
        # button would otherwise stay greyed out until the user switches away
        # and back.
        if hasattr(self, 'groups_stats_tab'):
            self.groups_stats_tab.restore_stats_controls()
            self.groups_stats_tab._update_run_button()
            if (self.state.freq_results is not None or self.state.mfi_results is not None
                    or self.state.counts_results is not None):
                self.groups_stats_tab.export_results_btn.setEnabled(True)
                self.groups_stats_tab.export_tables_btn.setEnabled(True)
                # Reset the guard so refresh() redraws on first tab switch.
                self.groups_stats_tab._last_drawn_cluster_names = {}

    # ------------------------------------------------------------------
    # Sidecar persistence — JSON + npz + CSV bundle for heavy state
    # ------------------------------------------------------------------

    def _sidecar_path(self, experiment_dir: Path | None = None) -> Path | None:
        """
        Return the bundle directory for the live-state sidecar, or None if
        unavailable.

        experiment_dir: explicit override, used by save_state() to flush a
        PREVIOUS experiment's state after self.controller has already
        switched to a different one -- drc_run_archive.current_state_path()
        always reads the CURRENT (live) experiment, which would be wrong
        in that situation. Falls back to that live lookup otherwise.
        """
        if experiment_dir is not None:
            return drc_run_archive.current_state_path_for(experiment_dir)
        try:
            return drc_run_archive.current_state_path(self.controller)
        except Exception:
            return None

    def _save_model_sidecar(self, experiment_dir: Path | None = None):
        """
        Write clustering tree data, the active DR run ids and stats results
        to the experiment's current_state bundle (see drc_persist). A field
        that cannot be stored is skipped with a log line rather than failing
        the save. Embeddings and cluster labels are not written here: every
        run is archived when it completes, and the active DR runs'
        embeddings are restored from the archive (_load_model_sidecar).

        Fitted UMAP/openTSNE/PaCMAP model objects are not written;
        only trained_reducers entries that are plain array dicts (the
        FlowSOM/Leiden/HDBSCAN tree data) are kept.

        clustering_runs / dr_runs are not included here — those are
        archived individually (their own bundle + manifest entry) at the
        moment each run completes; see drc_run_archive.py.

        experiment_dir: see _sidecar_path() -- passed through from
        save_state() when flushing a previous experiment's state.
        """
        path = self._sidecar_path(experiment_dir)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        _log.info(
            "_save_model_sidecar: writing %s -- freq_results=%s mfi_results=%s "
            "counts_results=%s stats_comparisons=%d stats_run_id=%r "
            "sample_groups=%d group_names=%r",
            path,
            self.state.freq_results is not None,
            self.state.mfi_results is not None,
            self.state.counts_results is not None,
            len(self.state.stats_comparisons),
            self.state.stats_run_id,
            len(self.state.sample_groups),
            list(self.state.group_names),
        )
        tree_data = {algo: model for algo, model in self.state.trained_reducers.items()
                     if isinstance(model, dict)}
        payload = dict((
            ('trained_reducers',  tree_data),
            ('active_dr_run_ids', dict(self._active_dr_run_id)),
            ('freq_results',      self.state.freq_results),
            ('counts_results',    self.state.counts_results),
            ('mfi_results',       self.state.mfi_results),
            ('freq_df',           self.state.freq_df),
            ('counts_df',         self.state.counts_df),
            ('mfi_df',            self.state.mfi_df),
            ('mfi_sample_df',     self.state.mfi_sample_df),
            ('stats_all_rel',     self.state.stats_all_rel),
            ('stats_group_vec',   self.state.stats_group_vec),
            ('stats_comparisons', self.state.stats_comparisons),
            ('stats_run_label',   self.state.stats_run_label),
            ('stats_run_id',      self.state.stats_run_id),
            ('stats_data_key',    self.state.stats_data_key),
            ('composition_df',        self.state.composition_df),
            ('composition_as_pct',    self.state.composition_as_pct),
            ('composition_group_var', self.state.composition_group_var),
            ('composition_run_label', self.state.composition_run_label),
            ('composition_run_id',    self.state.composition_run_id),
            ('composition_names',     self.state.composition_names),
            ('composition_colors',    self.state.composition_colors),
            ('pca_scores_df',           self.state.pca_scores_df),
            ('pca_loadings_df',         self.state.pca_loadings_df),
            ('pca_explained_variance',  self.state.pca_explained_variance),
            ('pca_run_label',           self.state.pca_run_label),
            ('pca_groups',              self.state.pca_groups),
            ('pca_sources',             self.state.pca_sources),
        ))
        try:
            skipped = drc_persist.save_bundle(path, payload, skip_unstorable=True)
            for key in skipped:
                print(f"[DR Plugin] Sidecar: skipped '{key}' (type cannot be stored)")
            print(f"[DR Plugin] Model sidecar saved → {path.name} "
                  f"(keys: {[k for k in payload if k not in skipped]})")
        except Exception as e:
            print(f"[DR Plugin] Could not save model sidecar: {e}")

    def _restore_active_embeddings(self):
        """state.embeddings / embedding_rows of each algorithm from its
        active DR run in the archive (shared arrays, not copies)."""
        for algo, run_id in self._active_dr_run_id.items():
            entry = next((e for e in self.state.dr_runs if e.get('run_id') == run_id), None)
            if entry is None:
                continue
            drc_run_archive.hydrate_run(self.controller, entry)
            if entry.get('embeddings'):
                self.state.embeddings[algo] = dict(entry['embeddings'])
                self.state.embedding_rows[algo] = dict(entry.get('event_rows') or {})

    def _load_model_sidecar(self):
        """
        Restore embeddings, cluster labels, clustering tree data and stats
        results from the current_state bundle, and rebuild state.dr_runs /
        state.clustering_runs from the run archive's manifest.  Missing or
        corrupt files are logged and otherwise ignored.

        Any pickle-era files and manifest entries are deleted first (see
        drc_run_archive.purge_legacy_pickles).
        """
        try:
            n_files, n_runs = drc_run_archive.purge_legacy_pickles(self.controller)
            if n_files or n_runs:
                msg = (f"Removed {n_files} file(s) and {n_runs} archived run(s) "
                       "saved in the old pickle format.")
                print(f"[DR Plugin] {msg}")
                self.progress_message(msg)
        except Exception as e:
            print(f"[DR Plugin] Could not remove pickle-era files: {e}")

        path = self._sidecar_path()

        # temporary diagnostic
        _log.info("_load_model_sidecar: path=%s exists=%s cache_root=%s",
                path, path.exists() if path else None,
                drc_run_archive.cache_root(self.controller))
                

        # Rebuild the run archive regardless of whether the current-state
        # sidecar exists — manifest.json/runs/ are independent of it.
        # Metadata only  — a run's actual payload is
        # hydrated on demand the first time something selects it (see
        # GroupsStatsTab._selected_run_entry / PlotCard's run selector).
        try:
            dr_entries, cl_entries = drc_run_archive.load_manifest_entries(self.controller)
            self.state.dr_runs = dr_entries
            self.state.clustering_runs = cl_entries
            if hasattr(self, 'config_tab'):
                self.config_tab.run_table.refresh()
            # temporary diagnostic
            _log.info("_load_model_sidecar: manifest gave %d DR run(s), "
                      "%d clustering run(s)", len(dr_entries), len(cl_entries))
        except Exception as e:
            print(f"[DR Plugin] Could not load run archive: {e}")

        # Source runs belong to the experiment being opened
        self._active_dr_run_id = {}

        if path is None or not drc_persist.bundle_exists(path):
            return
        try:
            payload = drc_persist.load_bundle(path) or {}
            if isinstance(payload.get('trained_reducers'), dict):
                self.state.trained_reducers = payload['trained_reducers']
            if isinstance(payload.get('embeddings'), dict):
                self.state.embeddings = payload['embeddings']
            if isinstance(payload.get('embedding_rows'), dict):
                self.state.embedding_rows = payload['embedding_rows']
            if isinstance(payload.get('active_dr_run_ids'), dict):
                self._active_dr_run_id = {
                    str(a): r for a, r in payload['active_dr_run_ids'].items()
                    if isinstance(r, str) and r}
            self._restore_active_embeddings()
            if isinstance(payload.get('cluster_labels'), dict):
                self.state.cluster_labels = payload['cluster_labels']
            if isinstance(payload.get('cluster_rows'), dict):
                self.state.cluster_rows = payload['cluster_rows']
            if isinstance(payload.get('freq_results'), pd.DataFrame):
                self.state.freq_results = payload['freq_results']
            if isinstance(payload.get('counts_results'), pd.DataFrame):
                self.state.counts_results = payload['counts_results']
            if isinstance(payload.get('mfi_results'), pd.DataFrame):
                self.state.mfi_results = payload['mfi_results']
            if isinstance(payload.get('freq_df'), pd.DataFrame):
                self.state.freq_df = payload['freq_df']
            if isinstance(payload.get('counts_df'), pd.DataFrame):
                self.state.counts_df = payload['counts_df']
            if isinstance(payload.get('mfi_df'), pd.DataFrame):
                self.state.mfi_df = payload['mfi_df']
            if isinstance(payload.get('mfi_sample_df'), pd.DataFrame):
                self.state.mfi_sample_df = payload['mfi_sample_df']
            if isinstance(payload.get('stats_all_rel'), list):
                self.state.stats_all_rel = payload['stats_all_rel']
            if isinstance(payload.get('stats_group_vec'), list):
                self.state.stats_group_vec = payload['stats_group_vec']
            if isinstance(payload.get('stats_comparisons'), list):
                self.state.stats_comparisons = payload['stats_comparisons']
            if isinstance(payload.get('composition_df'), pd.DataFrame):
                self.state.composition_df = payload['composition_df']
            if isinstance(payload.get('composition_as_pct'), bool):
                self.state.composition_as_pct = payload['composition_as_pct']
            if isinstance(payload.get('composition_group_var'), str):
                self.state.composition_group_var = payload['composition_group_var']
            if isinstance(payload.get('composition_run_label'), str):
                self.state.composition_run_label = payload['composition_run_label']
            if isinstance(payload.get('composition_run_id'), str):
                self.state.composition_run_id = payload['composition_run_id']
            if isinstance(payload.get('composition_names'), dict):
                self.state.composition_names = payload['composition_names']
            if isinstance(payload.get('composition_colors'), dict):
                self.state.composition_colors = payload['composition_colors']

            # migration for sidecars saved before composition_run_id
            # existed: composition_df loaded fine above, but the id came
            # back '' even though composition_run_label is a real
            # (non-"Active (unsaved)") run name. Best-effort backfill by
            # matching that label against a still-archived run; if none
            # matches (renamed or deleted since), the id is unverifiable,
            # so drop the data rather than risk showing it forever with
            # no way to detect it's gone stale.
            if (self.state.composition_df is not None and not self.state.composition_run_id
                    and self.state.composition_run_label not in ('', 'Active (unsaved)')):
                match = next((e for e in self.state.clustering_runs
                             if e.get('label') == self.state.composition_run_label), None)
                if match is not None:
                    self.state.composition_run_id = match.get('run_id', '')
                    _log.info("_load_model_sidecar: backfilled composition_run_id=%r "
                              "from label %r", self.state.composition_run_id,
                              self.state.composition_run_label)
                else:
                    _log.info("_load_model_sidecar: dropping unverifiable composition_df "
                              "(label %r, no matching run)", self.state.composition_run_label)
                    self.state.composition_df = None
                    self.state.composition_run_label = ''
                    self.state.composition_names = {}
                    self.state.composition_colors = {}

            if isinstance(payload.get('pca_scores_df'), pd.DataFrame):
                self.state.pca_scores_df = payload['pca_scores_df']
            if isinstance(payload.get('pca_loadings_df'), pd.DataFrame):
                self.state.pca_loadings_df = payload['pca_loadings_df']
            if isinstance(payload.get('pca_explained_variance'), tuple):
                self.state.pca_explained_variance = payload['pca_explained_variance']
            if isinstance(payload.get('pca_run_label'), str):
                self.state.pca_run_label = payload['pca_run_label']
            if isinstance(payload.get('pca_groups'), list):
                self.state.pca_groups = payload['pca_groups']
            if isinstance(payload.get('pca_sources'), list):
                self.state.pca_sources = payload['pca_sources']
            if isinstance(payload.get('stats_run_label'), str):
                self.state.stats_run_label = payload['stats_run_label']
            if isinstance(payload.get('stats_run_id'), str):
                self.state.stats_run_id = payload['stats_run_id']
            if isinstance(payload.get('stats_data_key'), tuple):
                self.state.stats_data_key = payload['stats_data_key']
            print(f"[DR Plugin] Model sidecar loaded ← {path.name} "
                  f"({list(self.state.trained_reducers.keys())} trained, "
                  f"{sum(len(v) for v in self.state.embeddings.values())} embeddings, "
                  f"{len(self.state.cluster_labels)} labelled samples, "
                  f"{len(self.state.dr_runs)} DR run(s), "
                  f"{len(self.state.clustering_runs)} clustering run(s) archived)")
            _log.info(
                "sidecar restore check: freq_results=%s counts_results=%s mfi_results=%s "
                "stats_run_id=%r selected_channels=%d marker_roles=%d",
                self.state.freq_results is not None,
                self.state.counts_results is not None,
                self.state.mfi_results is not None,
                self.state.stats_run_id,
                len(self.state.selected_channels),
                len(self.state.marker_roles),
            )
        except Exception as e:
            print(f"[DR Plugin] Could not load model sidecar: {e}")

    # Run parameter -> pending-state attribute applied by
    # _apply_pending_state_to_config_tab
    _RUN_PARAM_PENDING = {
        'UMAP': {'n_neighbors': '_pending_umap_n_neighbors', 'min_dist': '_pending_umap_min_dist',
                 'metric': '_pending_umap_metric', 'n_epochs': '_pending_umap_n_epochs',
                 'n_jobs': '_pending_umap_n_jobs'},
        'tSNE': {'perplexity': '_pending_tsne_perplexity', 'n_iter': '_pending_tsne_n_iter',
                 'n_jobs': '_pending_tsne_n_jobs'},
        'PaCMAP': {'n_neighbors': '_pending_pacmap_n_neighbors',
                   'MN_ratio': '_pending_pacmap_mn_ratio', 'FP_ratio': '_pending_pacmap_fp_ratio'},
        'FlowSOM': {'xdim': '_pending_flowsom_xdim', 'ydim': '_pending_flowsom_ydim',
                    'n_metaclusters': '_pending_flowsom_metaclusters',
                    'n_iter': '_pending_flowsom_n_iter'},
        'Leiden': {'resolution': '_pending_leiden_resolution',
                   'n_neighbors': '_pending_leiden_n_neighbors'},
        'HDBSCAN': {'min_cluster_size': '_pending_hdbscan_min_cluster_size',
                    'min_samples': '_pending_hdbscan_min_samples',
                    'cluster_selection_epsilon': '_pending_hdbscan_cluster_selection_epsilon'},
    }

    def load_run_settings(self, entry: dict):
        """Put an archived run's gates, training samples, channels, algorithm
        and parameters into the Configuration pages, so it can be run again
        on the current data (the user starts the run)."""
        if self.config_tab.is_run_in_progress():
            QMessageBox.information(self, "Run In Progress",
                                    "Wait for the current run to finish first.")
            return
        algo = entry.get('algorithm') or ''
        params = dict(entry.get('params') or {})
        self.state.selected_gates = list(entry.get('gates', []))
        channels = set(entry.get('channels', []))
        self._pending_channels_checked = repr(
            {ch: ch in channels for ch in getattr(self.config_tab, 'channel_checkboxes', {})})
        self._pending_training_samples = list(entry.get('training_sample_ids', []))
        for name, attr in self._RUN_PARAM_PENDING.get(algo, {}).items():
            if name in params:
                setattr(self, attr, params[name])
        if entry.get('kind') == 'dr':
            self._pending_dr_algo = algo
            if params.get('_n_training_events'):
                self.state.n_training_events = int(params['_n_training_events'])
        else:
            self._pending_cl_algo = algo
            self._pending_cl_assign_all_samples = bool(params.get('_assign_all_samples'))
            self._pending_cl_downsample_training = params.get('_event_cap') is not None
        self._loading = True
        try:
            self.config_tab.refresh()
            self._apply_pending_state_to_config_tab()
        finally:
            self._loading = False
        ct = self.config_tab
        if entry.get('kind') == 'clustering' and hasattr(ct, '_rb_space_dr'):
            if drc_clustering.clusters_in_dr_space(algo, params):
                ct._rb_space_dr.setChecked(True)
                idx = ct._cl_dr_combo.findText(params['_dr_algo'])
                if idx >= 0:
                    ct._cl_dr_combo.setCurrentIndex(idx)
            elif ct._rb_space_raw.isEnabled():
                ct._rb_space_raw.setChecked(True)
        self.progress_message(
            f"Settings of \"{entry.get('label', '')}\" loaded into Configuration; "
            f"press {'Train' if entry.get('kind') == 'dr' else 'Run'} to re-run it.")

    def _apply_pending_state_to_config_tab(self):
        """Apply persisted config-tab state after ConfigTab.refresh() has run."""
        channels_repr = getattr(self, '_pending_channels_checked', '')
        if channels_repr and hasattr(self.config_tab, 'channel_checkboxes'):
            try:
                checked_map = eval(channels_repr)  # noqa: S307
                for ch, cb in self.config_tab.channel_checkboxes.items():
                    if ch in checked_map:
                        cb.setChecked(checked_map[ch])
            except Exception:
                pass

        include_pending = getattr(self, '_pending_training_include', None) or {}
        if hasattr(self.config_tab, 'include_unstained_chk'):
            for chk, key in ((self.config_tab.include_unstained_chk, 'unstained'),
                             (self.config_tab.include_single_stain_chk, 'single_stain'),
                             (self.config_tab.include_beads_chk, 'beads')):
                if key in include_pending:
                    chk.blockSignals(True)
                    chk.setChecked(bool(include_pending[key]))
                    chk.blockSignals(False)

        training = list(getattr(self, '_pending_training_samples', []) or [])
        if ((training or any(include_pending.values()))
                and hasattr(self.config_tab, 'picker')):
            # Re-populate the picker with the saved selection order
            try:
                rel_paths = self.config_tab.training_pool_paths()
                # Only restore samples that are still offered
                valid_selected = [p for p in training if p in rel_paths]
                self.config_tab.picker.set_items(rel_paths, selected=valid_selected)
            except Exception as e:
                print(f"[DR Plugin] Could not restore training samples: {e}")

        ct = self.config_tab

        # ---- Training events spinbox ----
        if hasattr(ct, 'events_spinbox'):
            try:
                ct.events_spinbox.setValue(int(self.state.n_training_events))
            except Exception:
                pass

        # ---- DR algorithm radio ----
        dr_algo = getattr(self, '_pending_dr_algo', '')
        if dr_algo and hasattr(ct, '_dr_algo_group'):
            for btn in ct._dr_algo_group.buttons():
                if btn.text() == dr_algo:
                    btn.setChecked(True)
                    ct._on_dr_algo_changed()
                    break

        # ---- DR hyperparameters ----
        def _set_spin(widget, val, cast=int):
            if val is not None:
                try:
                    widget.setValue(cast(val))
                except Exception:
                    pass

        def _set_combo(widget, val):
            if val:
                idx = widget.findText(str(val))
                if idx >= 0:
                    widget.setCurrentIndex(idx)

        if hasattr(ct, 'umap_n_neighbors'):
            _set_spin(ct.umap_n_neighbors, getattr(self, '_pending_umap_n_neighbors', None))
            _set_spin(ct.umap_min_dist,    getattr(self, '_pending_umap_min_dist', None), float)
            _set_combo(ct.umap_metric,     getattr(self, '_pending_umap_metric', ''))
            _set_spin(ct.umap_n_epochs,    getattr(self, '_pending_umap_n_epochs', None))
            _set_spin(ct.umap_n_jobs,      getattr(self, '_pending_umap_n_jobs', None))
        if hasattr(ct, 'tsne_perplexity'):
            _set_spin(ct.tsne_perplexity, getattr(self, '_pending_tsne_perplexity', None))
            _set_spin(ct.tsne_n_iter,     getattr(self, '_pending_tsne_n_iter', None))
            _set_spin(ct.tsne_n_jobs,     getattr(self, '_pending_tsne_n_jobs', None))
        if hasattr(ct, 'pacmap_n_neighbors'):
            _set_spin(ct.pacmap_n_neighbors, getattr(self, '_pending_pacmap_n_neighbors', None))
            _set_spin(ct.pacmap_mn_ratio,    getattr(self, '_pending_pacmap_mn_ratio', None), float)
            _set_spin(ct.pacmap_fp_ratio,    getattr(self, '_pending_pacmap_fp_ratio', None), float)

        # ---- Clustering algorithm radio ----
        cl_algo = getattr(self, '_pending_cl_algo', '')
        if cl_algo and hasattr(ct, '_cl_algo_group'):
            for btn in ct._cl_algo_group.buttons():
                if btn.text() == cl_algo:
                    btn.setChecked(True)
                    ct._on_cl_algo_changed()
                    break

        # ---- Clustering hyperparameters ----
        if hasattr(ct, 'flowsom_xdim'):
            _set_spin(ct.flowsom_xdim,        getattr(self, '_pending_flowsom_xdim', None))
            _set_spin(ct.flowsom_ydim,        getattr(self, '_pending_flowsom_ydim', None))
            _set_spin(ct.flowsom_metaclusters, getattr(self, '_pending_flowsom_metaclusters', None))
            _set_spin(ct.flowsom_n_iter,      getattr(self, '_pending_flowsom_n_iter', None))
        if hasattr(ct, 'leiden_resolution'):
            _set_spin(ct.leiden_resolution,  getattr(self, '_pending_leiden_resolution', None), float)
            _set_spin(ct.leiden_n_neighbors, getattr(self, '_pending_leiden_n_neighbors', None))
        if hasattr(ct, 'hdbscan_min_cluster_size'):
            _set_spin(ct.hdbscan_min_cluster_size,
                     getattr(self, '_pending_hdbscan_min_cluster_size', None))
            _set_spin(ct.hdbscan_min_samples,
                     getattr(self, '_pending_hdbscan_min_samples', None))
            _set_spin(ct.hdbscan_cluster_selection_epsilon,
                     getattr(self, '_pending_hdbscan_cluster_selection_epsilon', None), float)
        if hasattr(ct, 'cl_assign_all_chk'):
            _pending_assign_all = getattr(self, '_pending_cl_assign_all_samples', None)
            if _pending_assign_all is not None:
                ct.cl_assign_all_chk.setChecked(_pending_assign_all)
        if hasattr(ct, 'cl_downsample_chk'):
            _pending_downsample = getattr(self, '_pending_cl_downsample_training', None)
            if _pending_downsample is not None:
                ct.cl_downsample_chk.setChecked(_pending_downsample)

    def _apply_pending_state_to_annotation_tab(self):
        """Restore the Cluster Annotation tab's run selections and checked
        channels after ClusterAnnotationTab.refresh() has run, then redraw
        the violin panel so it survives a tab switch or experiment reopen
        instead of reverting to the placeholder -- mirrors _redraw_map()
        and _populate_label_table(), which already redraw unconditionally
        on every refresh(); the violin panel was the one panel of the three
        that didn't."""
        cat = self.cluster_annotation_tab
        run_id = getattr(self, '_pending_annotation_run_id', '')
        if run_id:
            idx = cat.run_combo.findData(run_id)
            if idx >= 0:
                cat.run_combo.setCurrentIndex(idx)
        dr_run_id = getattr(self, '_pending_annotation_dr_run_id', '')
        if dr_run_id:
            idx = cat.dr_run_combo.findData(dr_run_id)
            if idx >= 0:
                cat.dr_run_combo.setCurrentIndex(idx)
        checked = getattr(self, '_pending_annotation_channels', [])
        if checked:
            for ch, cb in cat.channel_checkboxes.items():
                cb.setChecked(ch in checked)
        if hasattr(cat, 'species_combo'):
            idx = cat.species_combo.findData(getattr(self, '_pending_cluster_id_species', 'human'))
            if idx >= 0:
                cat.species_combo.setCurrentIndex(idx)
        run_id = cat.run_combo.currentData()
        checked_now = cat._checked_channels()
        cached = cat._violin_cache.get(run_id)
        if run_id is not None and checked_now and (
                cached is None or cached['channels'] != checked_now):
            cat._recompute_violins()

    def _apply_pending_state_to_transform_tab(self):
        """Add and configure biplot tiles after TransformTab.refresh() has run."""
        tile_y = getattr(self, '_pending_tile_y_channels', [])
        n_tiles = getattr(self, '_pending_n_tiles', 0)
        if not tile_y and not n_tiles:
            return
        tab = self.transform_tab
        # Remove any default tile added by refresh
        while tab._biplot_tiles:
            tab._remove_biplot()
        for y_ch in tile_y:
            tab._add_biplot()
            if tab._biplot_tiles:
                tab._biplot_tiles[-1].set_channels(
                    tab._available_channels(), y_channel=y_ch
                )

    def on_sample_selected(self, sample_path: str):
        """
        Called whenever the user clicks a sample in the sample browser.
        Only acts when this plugin's tab is active.

        Uses QTimer.singleShot(0) — the same pattern as AfComparisonPlotWidget —
        so that controller.load_sample() has fully completed before we read data.
        """
        if self.controller.current_mode != plugin_name:
            return

        # Defer by one event-loop tick so load_sample() finishes first
        QTimer.singleShot(0, self._load_and_redraw_transform_tab)

    def _load_and_redraw_transform_tab(self):
        """Load preview data then redraw if the Transforms tab is visible."""
        self.transform_tab._do_load_preview_data()
        if self.inner_tabs.currentIndex() == 0:
            self.transform_tab._schedule_redraw()

    # ------------------------------------------------------------------
    # Inner tab management
    # ------------------------------------------------------------------

    def _on_gate_tree_changed(self, gates: list[str]):
        """
        Fired when either tab's gate tree is toggled by the user.  Updates
        the shared state and mirrors the checked set onto the OTHER tree so
        both stay visually in sync.
        """
        self.state.selected_gates = list(gates)
        sender_tree = self.sender()
        for tree in (self.transform_tab.gate_tree, self.config_tab.gate_tree):
            if tree is not sender_tree:
                tree.set_checked_names(gates)

    def _refresh_report_if_shown(self):
        if self.inner_tabs.currentWidget() is self.report_tab:
            self.report_tab.refresh()

    def _on_inner_tab_changed(self, index: int):
        """Save state when leaving a tab, then refresh the newly activated one."""
        self.save_state()
        self._update_sample_browser_visibility()
        self._refresh_tab_at(index)

    def _update_sample_browser_visibility(self):
        """
        Show the main window's sample browser only on the Transforms tab,
        the one section that previews the clicked sample. The main window
        re-shows it whenever the user leaves this plugin.
        """
        signal = getattr(self.bus, 'sampleBrowserVisibilityRequested', None)
        if signal is not None:
            signal.emit(self.inner_tabs.currentWidget() is self.transform_tab)

    def _refresh_active_tab(self):
        self._refresh_tab_at(self.inner_tabs.currentIndex())

    def refresh_plot_theme(self):
        """
        Re-render every currently-visible plot under the new theme.
        """
        self.workspace_tab.refresh()
        self.cluster_annotation_tab._redraw_map()
        cat = self.cluster_annotation_tab
        run_id = cat.run_combo.currentData()
        if run_id in cat._violin_cache:
            cat._draw_current_violin()
        if cat.annotation_sub_tabs.currentIndex() == getattr(cat, '_marker_summary_tab_index', -1) \
                and run_id in cat._marker_summary_cache:
            cat._draw_marker_summary()
        if self.state.stats_comparisons:
            self.groups_stats_tab._draw_results()
        self.groups_stats_tab.refresh_theme_dependent_result_tabs()

    def _tab_refresh_key(self, index: int):
        """
        Cheap fingerprint of the state that determines what tab *index*
        would show if refreshed right now (avoid redoing a full
        tab refresh when nothing relevant changed, e.g. plain tab-hopping).

        Deliberately coarse and conservative: every field a tab's refresh()
        reads to decide WHAT to display is included, so a false "unchanged"
        read (which would show stale data) shouldn't happen; missing a field
        here just costs one redundant refresh, which is the status quo
        today. Dicts mutated in place (cluster_names, sample_groups, ...)
        are snapshotted BY VALUE (sorted tuple of items), not id(), since
        id() wouldn't change on an in-place edit. Objects that are always
        wholesale-replaced when they change (result DataFrames, run
        archives) use id()/len() instead of hashing their content.
        """
        s = self.state
        if index == 0:      # TransformTab
            return (
                tuple(sorted(s.selected_gates)),
                tuple(
                    (ch, round(v.get('W', 0.0), 4), round(v.get('A', 0.0), 4),
                     round(v.get('T', 0.0), 4), round(v.get('M', 0.0), 4))
                    for ch, v in sorted(s.channel_transform_params.items())
                ),
            )
        elif index == 1:    # ConfigTab
            return (
                tuple(sorted(s.selected_gates)),
                tuple(s.selected_channels),
                tuple(sorted(s.training_sample_ids)),
                s.n_training_events,
                tuple(sorted(s.dr_status.items())),
                s.active_clustering_algorithm,
                s.n_clusters,
                len(s.dr_runs), len(s.clustering_runs),
            )
        elif index == 2:    # ClusterAnnotationTab
            return (
                len(s.dr_runs), len(s.clustering_runs),
                tuple(sorted(s.cluster_labels)),
                tuple(sorted(s.cluster_names.items())),
                tuple(sorted(s.cluster_colors.items())),
            )
        elif index == 3:    # GroupsStatsTab
            return (
                tuple(sorted(s.sample_groups.items())),
                tuple(s.group_names),
                tuple(sorted(s.group_patterns.items())),
                tuple(s.testing_group_selection),
                s.contrast_mode, s.reference_group, s.paired, s.pairing_variable,
                len(s.dr_runs), len(s.clustering_runs),
                tuple(sorted(s.cluster_labels)),
                s.stats_run_id,
                tuple(sorted(s.cluster_names.items())),
                id(s.freq_results), id(s.counts_results), id(s.mfi_results),
                id(s.composition_df),
                s.composition_as_pct, s.composition_group_var,
                tuple(sorted(s.marker_roles.items())),
                tuple(
                    (ch, round(v.get('W', 0.0), 4), round(v.get('A', 0.0), 4),
                     round(v.get('T', 0.0), 4), round(v.get('M', 0.0), 4))
                    for ch, v in sorted(s.channel_transform_params.items())
                ),
            )
        elif index == 4:    # WorkspaceTab
            return (
                len(s.dr_runs), len(s.clustering_runs),
                tuple(sorted(s.cluster_names.items())),
                tuple(sorted(s.cluster_colors.items())),
                len(s.plot_configs), s.plot_theme,
            )
        elif index == 5:    # ReportTab -- cheap to rebuild; always refresh
            return None
        return None

    def _refresh_tab_at(self, index: int):
        """
        Call refresh() on the tab at *index* if it has that method --
        skipping the call entirely if this tab's own fingerprint
        (_tab_refresh_key) hasn't changed since the last time it ran.
        This sits ABOVE refresh(); it doesn't replace the
        finer-grained guards individual refresh() methods already have
        (e.g. GroupsStatsTab's _last_drawn_cluster_names / _has_results_tab
        checks), which still apply whenever refresh() does run.
        """
        if index == 5:
            # The report lists what these tabs hold, and they build it
            # from the restored state on refresh.
            for source_index in (2, 3):
                self._refresh_tab_at(source_index)
        key = self._tab_refresh_key(index)
        if key is not None and self._last_tab_refresh_key.get(index) == key:
            return
        self._last_tab_refresh_key[index] = key

        tabs = [
            self.transform_tab,
            self.config_tab,
            self.cluster_annotation_tab,
            self.groups_stats_tab,
            self.workspace_tab,
            self.report_tab,
        ]
        if 0 <= index < len(tabs):
            tab = tabs[index]
            if hasattr(tab, 'refresh'):
                tab.refresh()
            # After refresh, apply any persisted state for that tab
            if index == 0:
                self._apply_pending_state_to_transform_tab()
            elif index == 1:
                self._apply_pending_state_to_config_tab()
            elif index == 2:
                self._apply_pending_state_to_annotation_tab()

    # ------------------------------------------------------------------
    # Progress reporting (convenience — tabs can call this via parent)
    # ------------------------------------------------------------------

    def progress_message(self, text: str):
        """Emit to status bar and print to stdout."""
        print(f"[DR Plugin] {text}")
        if self.bus:
            self.bus.statusMessage.emit(text)

    # ==================================================================
    # Run-time estimate for large runs
    # ==================================================================

    # _estimate_worker holds the active _RunEstimateWorker (or None).
    _estimate_worker = None

    def _dr_probe_fn(self, algo: str, params: dict):
        """Function timing a DR fit on one event array and the embedding of
        another; returns {'fit': seconds, 'embed': seconds}, and for UMAP
        'index' (its hnswlib search index).  UMAP's numba code is compiled
        first, so no time includes compilation.
        single_thread: time tSNE on one thread (calibrates its FFT fit)."""
        def probe(fit_data, embed_data, single_thread=False):
            if algo == 'UMAP':
                drc_jit.ensure_warm('UMAP', _umap_parallel(params))
            start = time.perf_counter()
            if algo == 'UMAP':
                reducer = self._run_umap(params, fit_data)
            elif algo == 'tSNE':
                reducer = self._run_opentsne({**params, 'n_jobs': 1} if single_thread else params,
                                             fit_data, negative_gradient_method='bh')
            elif algo == 'PaCMAP':
                reducer = self._run_pacmap(params, fit_data)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")
            fitted = time.perf_counter()
            timings = {'fit': fitted - start}
            if algo == 'UMAP' and drc_umap_index.attach_search_index(
                    reducer, _umap_threads(reducer)):
                timings['index'] = time.perf_counter() - fitted
                fitted = time.perf_counter()
            self._dr_transform(algo, reducer, embed_data)
            timings['embed'] = time.perf_counter() - fitted
            return timings
        return probe

    # ------------------------------------------------------------------
    # Per-machine run-time calibration (drc_time_estimate), kept in the
    # plugin's QSettings outside any experiment group
    # ------------------------------------------------------------------

    _CALIBRATION_KEY = 'run_time_calibration'

    def _calibration(self) -> dict:
        raw = QSettings('honeychrome', _SETTINGS_APP_KEY).value(self._CALIBRATION_KEY, '')
        try:
            data = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            data = {}
        return data if isinstance(data, dict) else {}

    def _store_calibration(self, updates: dict):
        if not updates:
            return
        data = self._calibration()
        data.update(updates)
        settings = QSettings('honeychrome', _SETTINGS_APP_KEY)
        settings.setValue(self._CALIBRATION_KEY, json.dumps(data))
        settings.sync()

    def _learn_run_time(self, algo: str, params: dict, seconds: dict, events: dict,
                        pool: int = 0, calls: int = 0, jit_included: bool = False):
        """Update this machine's calibration from a completed run's stage times."""
        try:
            updates = drc_time_estimate.observe_run(
                algo, params, self._calibration(), seconds, events,
                pool=pool, calls=calls, jit_included=jit_included)
            self._store_calibration(updates)
        except Exception:
            _log.exception("could not update the run-time calibration")

    def _confirm_large_run(self, kind: str, algo: str, params: dict, proceed):
        """
        Call proceed() to start a 'dr' or 'cl' run.  The run's time is
        predicted first (drc_time_estimate); a run predicted to take
        drc_time_estimate.CONFIRM_SECONDS or longer starts only if the user
        confirms, shorter ones start immediately.
        """
        if self._estimate_worker is not None:
            self._release_run_token(kind)
            QMessageBox.information(self, "Estimating Run Time",
                                    "A run time estimate is already in progress.")
            return
        af_state = drc_pipeline.snapshot_af_state(self.controller)
        self._set_estimating(kind, True)
        jit_warm = (algo != 'UMAP' or kind != 'dr'
                    or drc_jit.is_warm('UMAP', _umap_parallel(params)))
        worker = _RunEstimateWorker(
            self.controller, self.state, kind, algo, dict(params), self._calibration(),
            af_state=af_state,
            dr_probe_fn=self._dr_probe_fn(algo, params) if kind == 'dr' else None,
            jit_warm=jit_warm)
        worker.estimated.connect(
            lambda estimate: self._on_run_estimated(estimate, proceed, worker))
        worker.finished.connect(lambda: self._on_estimate_thread_finished(worker))
        self._estimate_worker = worker
        worker.start()

    def _on_estimate_thread_finished(self, worker):
        if self._estimate_worker is worker:
            self._estimate_worker = None

    def _cancel_estimate(self, kind: str) -> bool:
        """Abandon a run-time estimate in progress for *kind*; True if there was one."""
        worker = self._estimate_worker
        if worker is None or worker.kind != kind:
            return False
        worker.detach()
        self._detached_workers = [w for w in self._detached_workers if w.isRunning()]
        self._detached_workers.append(worker)
        self._estimate_worker = None
        self._set_estimating(kind, False)
        self._release_run_token(kind)
        self.progress_message("Run cancelled before it started.")
        return True

    def _acquire_run_token(self, kind: str, label: str) -> bool:
        """Take the job gate for a 'dr' or 'cl' run; False (busy shown) when another job holds it."""
        token = acquire_job(self.controller, label)
        if token is None:
            self.progress_message(self.controller.job_gate.busy_text())
            return False
        self._run_tokens[kind] = token
        return True

    def _release_run_token(self, kind: str):
        token = self._run_tokens.pop(kind, None)
        if token is not None:
            token.release()

    def _on_run_estimated(self, estimate, proceed, worker=None):
        """Main thread: start the run, or ask first when it was timed."""
        if worker is not None and worker is not self._estimate_worker:
            return
        self._store_calibration(estimate.calibration_updates)
        if not estimate.needs_confirmation:
            proceed()
            return
        label = drc_time_estimate.KIND_NOUNS[estimate.kind]
        if estimate.error:
            text = (f"The {label} time could not be measured on this computer for "
                    f"{estimate.total_events:,} events; assuming a slow computer it "
                    f"could take about "
                    f"{drc_time_estimate.format_duration(estimate.seconds)}.\n\n"
                    f"Start {label} anyway?")
        else:
            self.progress_message(
                f"Estimated {label} time: about "
                f"{drc_time_estimate.format_duration(estimate.seconds)}")
            text = drc_time_estimate.estimate_message(estimate)
            if any(w.isRunning() for w in self._detached_workers):
                text += ("\n\nA cancelled DR job is still finishing in the "
                         "background and slowed the timing, so the estimate "
                         "is probably too high.")
        reply = QMessageBox.question(
            self, "Run time estimate", text,
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            proceed()
        else:
            self._set_estimating(estimate.kind, False)
            self._release_run_token(estimate.kind)
            self.progress_message(
                f"{drc_time_estimate.KIND_LABELS[estimate.kind]} cancelled.")

    def _set_estimating(self, kind: str, estimating: bool):
        """Lock the inputs of a 'dr' or 'cl' run and show an indeterminate
        progress bar while its time is being estimated; undo both when not."""
        if not hasattr(self, 'config_tab'):
            return
        ct = self.config_tab
        self._set_run_lock(kind, estimating)
        if kind == 'dr':
            status_label = ct.dr_status_label
            bar = getattr(ct, 'dr_progress_bar', None)
        else:
            status_label = ct.cl_status_label
            bar = getattr(ct, 'cl_progress_bar', None)
        cancel_btn = ct.dr_cancel_btn if kind == 'dr' else ct.cl_cancel_btn
        cancel_btn.setEnabled(estimating)
        if estimating:
            status_label.setText("⏳ Estimating run time …")
            status_label.setStyleSheet("color: orange;")
            if bar is not None:
                bar.setRange(0, 0)
                bar.setTextVisible(False)
                bar.setVisible(True)
        else:
            if bar is not None:
                bar.setVisible(False)
            if kind == 'dr':
                ct._refresh_dr_status()
            else:
                ct._refresh_cl_status()

    # ==================================================================
    # Dimensionality Reduction
    # ==================================================================

    def _load_training_data(self, af_state=None):
        """
        Pool transformed, gated, downsampled events from all training samples:
        ``(data, selections)`` from drc_pipeline.load_training_pool_with_rows
        (correct channel alignment, correct FlowKit transform, no silent
        arcsinh fallback, full logging).

        Must be callable from a background thread — contains no Qt GUI calls.
        Pre-flight validation (empty samples / channels / gate) is performed
        by the caller (_run_dr) on the main thread before the worker starts.

        af_state: optional AF snapshot captured on the main thread before the
        worker started — see drc_pipeline.apply_unmixing_af_aware() docstring.
        """
        return drc_pipeline.load_training_pool_with_rows(
            self.controller, self.state, af_state=af_state)

    def _dr_transform(self, algo: str, reducer, data: np.ndarray) -> np.ndarray:
        """Embed *data* with a fitted reducer (numba work in a safe section)."""
        if algo == 'UMAP':
            with drc_jit.numba_section(self.progress_message):
                return reducer.transform(data)
        return reducer.transform(data)

    def _get_sample_data(self, rel_path: str, algo: str, af_state=None) -> np.ndarray | None:
        """
        Load + unmix + gate + transform one sample → (n_events, n_selected).
        Used during 'Apply to All Samples' and per-sample cluster assignment.

        af_state: optional AF snapshot captured on the main thread before the
        worker started — see drc_pipeline.apply_unmixing_af_aware() docstring.
        """
        return drc_pipeline.load_sample_features(self.controller, self.state, rel_path, af_state=af_state)

    # ------------------------------------------------------------------
    # UMAP
    # ------------------------------------------------------------------

    def _run_umap(self, params: dict, training_data: np.ndarray,
                  progress_hook=None, should_cancel=None):
        """Train a UMAP reducer.

        should_cancel (callable() -> bool) is checked before the fit starts;
        the epoch loop is stopped through progress_hook.
        """
        drc_jit.ensure_threading_layer()
        import umap as umap_lib
        import os

        n_neighbors = params['n_neighbors']

        # n_jobs > 1 requires giving up the fixed seed -- umap-learn forces
        # n_jobs back to 1 internally whenever random_state is set, to keep
        # training deterministic.
        n_jobs = params.get('n_jobs', 1)
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1
        random_state = 42 if n_jobs == 1 else None

        if should_cancel is not None and should_cancel():
            raise _RunCancelled()

        self.progress_message(
            f"Training UMAP  (n_neighbors={n_neighbors}, "
            f"min_dist={params['min_dist']}, metric={params['metric']}, "
            f"n_epochs={params['n_epochs']}, n_jobs={n_jobs}) …"
        )

        # Wire the tqdm hook for per-epoch progress if one was supplied.
        tqdm_kwds = {'tqdm_class': progress_hook} if progress_hook is not None else {}

        reducer = umap_lib.UMAP(
            n_neighbors=n_neighbors,
            min_dist=params['min_dist'],
            metric=params['metric'],
            n_epochs=params['n_epochs'],
            n_components=2,
            random_state=random_state,
            n_jobs=n_jobs,
            low_memory=False,
            verbose=False,
            tqdm_kwds=tqdm_kwds,
        )
        with drc_jit.numba_section(self.progress_message):
            reducer.fit(training_data)

        # Drop the tqdm hook now training is done. UMAP (sklearn
        # BaseEstimator) stores constructor kwargs as attributes, so
        # reducer.tqdm_kwds would otherwise keep a live reference to the
        # _UMAPTqdmHook and, through it, the finished training worker.
        # The hook has no purpose after fit() returns.
        reducer.tqdm_kwds = {}

        return reducer

    # ------------------------------------------------------------------
    # openTSNE
    # ------------------------------------------------------------------

    def _run_opentsne(self, params: dict, training_data: np.ndarray,
                      should_cancel=None, negative_gradient_method: str = 'auto'):
        """
        Train tSNE using openTSNE.  should_cancel (callable() -> bool) is
        polled during the optimisation and stops the fit by raising
        _RunCancelled.

        openTSNE.TSNE.fit() returns a TSNEEmbedding object which has a
        native .transform(new_data) method for out-of-sample projection.
        """
        import openTSNE
        import os

        perplexity = params['perplexity']
        n_iter     = params['n_iter']
        n_jobs     = params['n_jobs']
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        self.progress_message(
            f"Training tSNE  (perplexity={perplexity}, "
            f"n_iter={n_iter}, n_jobs={n_jobs}) …"
        )

        def _poll_cancel(iteration, error, embedding):
            if should_cancel():
                raise _RunCancelled()

        cancel_kwds = ({'callbacks': _poll_cancel, 'callbacks_every_iters': 10}
                       if should_cancel is not None else {})

        if negative_gradient_method == 'auto':
            # Barnes-Hut up to 50,000 events where openTSNE has OpenMP,
            # openTSNE's own 10,000 otherwise
            negative_gradient_method = drc_time_estimate.tsne_gradient_method(len(training_data))
            self.progress_message(f"tSNE gradient: {negative_gradient_method.upper()}")
        tsne = openTSNE.TSNE(
            n_components=2,
            perplexity=perplexity,
            n_jobs=n_jobs,
            n_iter=n_iter,
            random_state=42,
            verbose=True,
            negative_gradient_method=negative_gradient_method,
            **cancel_kwds,
        )
        # fit() returns a TSNEEmbedding that supports .transform()
        embedding = tsne.fit(training_data)
        # transform() inherits the fit's gradient method; 'auto' keeps
        # openTSNE's choice by the size of each batch of new events
        embedding.gradient_descent_params['negative_gradient_method'] = 'auto'
        # transform() first centres the fitted map in place (on its bounding
        # box); centre it now, so the pool's own fit coordinates (P2) and
        # transformed events share one frame
        embedding -= (np.max(embedding, axis=0) + np.min(embedding, axis=0)) / 2
        self.progress_message("tSNE training complete.")
        return embedding

    # ------------------------------------------------------------------
    # PaCMAP
    # ------------------------------------------------------------------

    def _run_pacmap(self, params: dict, training_data: np.ndarray):
        drc_jit.ensure_threading_layer()
        _check_annoy_build()
        import pacmap
        self.progress_message(
            f"Training PaCMAP  (n_neighbors={params['n_neighbors']}, "
            f"MN_ratio={params['MN_ratio']}, FP_ratio={params['FP_ratio']}) …"
        )
        # Keeping the Annoy tree lets transform() reuse it instead of
        # rebuilding it from the whole training pool on every call.
        tree_bytes = _pacmap_tree_bytes(*training_data.shape)
        save_tree = tree_bytes <= PACMAP_TREE_MAX_BYTES
        if not save_tree:
            _log.warning("PaCMAP: the Annoy tree for %d events x %d channels would take "
                         "about %.1f GB, so it is rebuilt for every transform instead",
                         training_data.shape[0], training_data.shape[1], tree_bytes / 1e9)
        reducer = pacmap.PaCMAP(
            n_components=2,
            n_neighbors=params['n_neighbors'],
            MN_ratio=params['MN_ratio'],
            FP_ratio=params['FP_ratio'],
            random_state=42,
            save_tree=save_tree,
        )
        try:
            with drc_jit.numba_section(self.progress_message):
                reducer.fit_transform(training_data)
        except ValueError as exc:
            _log.error("PaCMAP fit failed (%s): %s", exc, _array_summary(training_data))
            raise
        return _PaCMAPWrapper(reducer, None if save_tree else training_data)

    # ------------------------------------------------------------------
    # Public DR entry points
    # ------------------------------------------------------------------

    # _dr_worker holds the active _DrWorker QThread (or None).
    _dr_worker = None

    def _run_dr(self, algo: str, params: dict):
        """Start background DR training.  Returns immediately; UI stays live."""
        if self._dr_worker is not None and self._dr_worker.isRunning():
            QMessageBox.information(self, "DR Running",
                                    "A DR job is already running.  "
                                    "Click Cancel to stop it first.")
            return

        # Pre-flight validation — must run on the main thread before the
        # worker starts, because _load_training_data is called from the
        # background thread and must not touch Qt GUI objects.
        if not self.state.training_sample_ids:
            QMessageBox.warning(self, "No Training Samples",
                                "Select at least one training sample in the picker.")
            return
        if not self.state.selected_channels:
            QMessageBox.warning(self, "No Channels",
                                "Select at least one channel in the picker.")
            return
        if not self.state.selected_gates:
            QMessageBox.warning(self, "No Gate",
                                "Select a gate in the Configuration tab.")
            return
        if not self._acquire_run_token('dr', f'Honeycluster: {algo} training'):
            return

        self._confirm_large_run(
            'dr', algo, params, lambda: self._start_dr_training(algo, params))

    def _start_dr_training(self, algo: str, params: dict):
        """Start the background DR training worker (inputs already validated)."""
        self.state.dr_status[algo] = 'running'
        self._set_dr_buttons_running(True)

        # Snapshot AF/transfer-matrix state on the main thread before the
        # worker starts — see apply_unmixing_af_aware() docstring.
        af_state = drc_pipeline.snapshot_af_state(self.controller)
        # The unmixing and gating the run is made with, for its archive entry
        fingerprints = (
            drc_pipeline.sync_fingerprint(self.controller, self.state.selected_gates,
                                          self.state.training_sample_ids),
            drc_pipeline.split_fingerprints(self.controller, self.state.selected_gates,
                                            self.state.training_sample_ids))

        worker = _DrWorker('train', self, algo, params, training_only=True, af_state=af_state)
        worker.progress.connect(self.progress_message)
        worker.progress_value.connect(
            lambda cur, tot: self._on_dr_progress(cur, tot))
        worker.finished.connect(
            lambda ok, err: self._on_dr_finished(algo, ok, err, task='train', params=params,
                                                 fingerprints=fingerprints))
        self._dr_worker = worker
        worker.start()

    def _apply_dr_to_all_samples(self, algo: str):
        """Start background embedding of ALL samples.  Returns immediately."""
        if self.state.trained_reducers.get(algo) is None:
            if self.state.embeddings.get(algo):
                QMessageBox.information(
                    self, "Model Not Available",
                    f"The fitted {algo} model is only kept for the session it "
                    f"was trained in. Its embeddings were restored, but "
                    f"projecting more samples needs the model: re-train {algo} "
                    f"first."
                )
            else:
                QMessageBox.warning(self, "No Trained Model",
                                    f"Train a {algo} model first.")
            return
        if self._dr_worker is not None and self._dr_worker.isRunning():
            QMessageBox.information(self, "DR Running",
                                    "A DR job is already running.  "
                                    "Click Cancel to stop it first.")
            return
        if not self._acquire_run_token('dr', f'Honeycluster: {algo} on all samples'):
            return

        self._set_dr_buttons_running(True)
        # Snapshot AF/transfer-matrix state on the main thread before the
        # worker starts — see apply_unmixing_af_aware() docstring.
        af_state = drc_pipeline.snapshot_af_state(self.controller)
        worker = _DrWorker('apply', self, algo, {}, training_only=False, af_state=af_state)
        worker.progress.connect(self.progress_message)
        worker.progress_value.connect(
            lambda cur, tot: self._on_dr_progress(cur, tot))
        worker.finished.connect(
            lambda ok, err: self._on_dr_finished(algo, ok, err, task='apply'))
        self._dr_worker = worker
        worker.start()

    def _cancel_dr(self):
        """
        Cancel the running DR worker.  UMAP and tSNE stop at their next
        epoch/iteration check; a step that cannot be interrupted (graph
        building, PaCMAP) finishes in the background with its result
        discarded.  Either way the UI is released immediately.
        """
        if self._cancel_estimate('dr'):
            return
        worker = self._dr_worker
        if worker is None or not worker.isRunning():
            return
        worker.detach()
        self._detached_workers = [w for w in self._detached_workers if w.isRunning()]
        self._detached_workers.append(worker)
        self.progress_message(
            "DR job cancelled.  A step that cannot be interrupted may keep "
            "running in the background; its result is discarded.")
        self._on_dr_finished(worker.algo, False, 'Cancelled.', task=worker.task)

    def _on_dr_progress(self, current: int, total: int):
        """Update the DR progress bar from the worker thread (via signal)."""
        if not hasattr(self, 'config_tab'):
            return
        bar = self.config_tab.dr_progress_bar
        if total > 0:
            bar.setRange(0, total)
            bar.setValue(current)
            bar.setFormat(f"embedded {current}/{total}")
            bar.setTextVisible(True)
        else:
            # Indeterminate — e.g. kNN build phase or non-UMAP algorithms
            bar.setRange(0, 0)
            bar.setTextVisible(False)
        bar.setVisible(True)

    def _set_dr_buttons_running(self, running: bool):
        """Toggle Train/Apply/Cancel button states and the run lock during a DR run."""
        if not hasattr(self, 'config_tab'):
            return
        ct = self.config_tab
        ct.dr_run_btn.setEnabled(not running)
        ct.dr_apply_btn.setEnabled(not running)
        ct.dr_cancel_btn.setEnabled(running)
        self._set_run_lock('dr', running)
        if not running and hasattr(ct, 'dr_progress_bar'):
            ct.dr_progress_bar.setVisible(False)
        elif running and hasattr(ct, 'dr_progress_bar'):
            ct._refresh_dr_status()
            # Show indeterminate bar immediately; UMAP will switch to determinate
            ct.dr_progress_bar.setRange(0, 0)
            ct.dr_progress_bar.setTextVisible(False)
            ct.dr_progress_bar.setVisible(True)

    def _set_run_lock(self, kind: str, running: bool):
        """
        Lock the inputs of a DR ('dr') or clustering ('cl') run while it is
        in progress: the Configuration controls, and the Transforms gate
        tree, which edits the same shared gate selection.
        """
        self.config_tab.set_run_lock(kind, running)
        self.transform_tab.gate_tree.setEnabled(not self.config_tab.is_run_in_progress())

    def _on_dr_finished(self, algo: str, success: bool, error_msg: str,
                        task: str = 'train', params: dict | None = None,
                        fingerprints: tuple | None = None):
        """Called on the main thread when the DR worker thread finishes."""
        self._release_run_token('dr')
        self._set_dr_buttons_running(False)
        if success:
            if self.state.dr_status.get(algo) != 'done':
                # apply-only run: status was already 'done'
                self.state.dr_status[algo] = 'done'
            if task == 'train':
                self._dr_train_params[algo] = dict(params or {})
                self._archive_dr_run(algo, params or {}, fingerprints)
            else:
                self._update_archived_dr_run(algo)
            worker = self._dr_worker
            if worker is not None:
                seconds = dict(worker.stage_seconds)
                if task == 'train':
                    pool = worker.stage_events['fit']
                    self._dr_pool_size[algo] = pool
                else:
                    pool = self._dr_pool_size.get(algo, 0)
                    if not pool:
                        # Model from an earlier session: its settings and
                        # pool size are unknown, so only file reading is learnt
                        seconds.pop('embed', None)
                self._learn_run_time(
                    algo, self._dr_train_params.get(algo, {}), seconds,
                    worker.stage_events, pool=pool,
                    calls=worker.embed_calls, jit_included=worker.jit_included)
        else:
            if error_msg and error_msg != 'Cancelled.':
                self.state.dr_status[algo] = 'error'
                self.state.dr_timestamps[algo] = datetime.now().isoformat(timespec='seconds')
                QMessageBox.critical(self, "DR Error", error_msg)
            else:
                # A cancelled apply leaves the trained model as it was
                self.state.dr_status[algo] = 'done' if task == 'apply' else 'idle'
        if hasattr(self, 'config_tab'):
            self.config_tab._refresh_dr_status()
            self.config_tab._refresh_cl_space_combo()
        self._dr_worker = None

    def _archive_dr_run(self, algo: str, params: dict, fingerprints: tuple):
        """Archive a freshly completed DR training run (§0.2), with the
        (sync, split) fingerprints taken when it started."""
        reducer = self.state.trained_reducers.get(algo)
        embeddings = self.state.embeddings.get(algo, {})
        if reducer is None:
            return
        channels = [c for c in self.state.selected_channels
                    if c not in drc_pipeline.META_CHANNELS]
        n_events = sum(len(e) for e in embeddings.values())
        try:
            entry = drc_run_archive.archive_dr_run(
                self.controller, self.state,
                algorithm=algo,
                embeddings=dict(embeddings),
                gates=list(self.state.selected_gates),
                training_sample_ids=list(self.state.training_sample_ids),
                channels=channels,
                params={**params, '_n_training_events': int(self.state.n_training_events)},
                n_events=n_events,
                sync_fingerprint=fingerprints[0],
                event_rows=dict(self.state.embedding_rows.get(algo, {})),
                split_fingerprints=fingerprints[1],
                transform_params=self._run_transform_params(channels),
            )
        except Exception as e:
            # Archiving must never take down _on_dr_finished — a failure
            # here left status/UI refresh and worker cleanup un-run,
            # which is why runs went missing from the table/Workspace.
            print(f"[DR Plugin] Could not archive DR run for {algo}: {e}")
            self.progress_message(f"DR run completed but could not be archived: {e}")
            return
        self._active_dr_run_id[algo] = entry['run_id']
        self.progress_message(f"DR run archived as \"{entry['label']}\"")
        if hasattr(self, 'config_tab'):
            self.config_tab.run_table.refresh()
        self._on_runs_changed()

    def _run_transform_params(self, channels: list[str]) -> dict:
        """The transforms a run is fitted in, recorded with it so events that
        enter its gates later can be placed in the same feature space."""
        return {ch: dict(self.state.channel_transform_params.get(ch, {}))
                for ch in channels}

    def _update_archived_dr_run(self, algo: str):
        """
        Refresh an already-archived DR run's payload after 'Apply to All
        Samples' embeds additional samples under the same trained model.
        Does not create a new manifest entry — see PipelineState.dr_runs.
        """
        run_id = self._active_dr_run_id.get(algo)
        if not run_id:
            return
        embeddings = dict(self.state.embeddings.get(algo, {}))
        drc_run_archive.update_dr_run_embeddings(
            self.controller, self.state, run_id, embeddings,
            event_rows=dict(self.state.embedding_rows.get(algo, {})),
        )
        if hasattr(self, 'config_tab'):
            self.config_tab.run_table.refresh()
        self._on_runs_changed()

    # ==================================================================
    # Clustering
    # ==================================================================

    # ------------------------------------------------------------------
    # Public clustering entry point
    # ------------------------------------------------------------------

    def _run_clustering(self, algo: str, params: dict):
        """Run the selected clustering algorithm in a background thread."""
        if hasattr(self, '_cl_worker') and self._cl_worker is not None \
                and self._cl_worker.isRunning():
            QMessageBox.information(self, "Clustering Running",
                                    "A clustering job is already running.")
            return

        # Pre-flight validation — mirrors _run_dr's guards.  Without this,
        # an empty training_sample_ids list runs through to completion with
        # empty per-sample label arrays instead of failing loudly.
        if not self.state.training_sample_ids:
            QMessageBox.warning(self, "No Training Samples",
                                "Select at least one training sample in the picker.")
            return
        if not self.state.selected_channels:
            QMessageBox.warning(self, "No Channels",
                                "Select at least one channel in the picker.")
            return
        if not self.state.selected_gates:
            QMessageBox.warning(self, "No Gate",
                                "Select a gate in the Configuration tab.")
            return
        if not self._acquire_run_token('cl', f'Honeycluster: {algo} clustering'):
            return

        # Clustering in a DR space needs a usable embedding of every training
        # sample from a DR run that still exists.
        if params.get('_space') == 'dr':
            dr_algo = params.get('_dr_algo') or ''
            if not dr_algo:
                self._release_run_token('cl')
                QMessageBox.warning(
                    self, "No DR Embedding",
                    f"{algo} in DR space needs a PaCMAP, UMAP or tSNE embedding of "
                    "every training sample, and none is available.\n\n"
                    "Train a DR first"
                    + ("." if algo == 'HDBSCAN' else ", or cluster on the original features."))
                return
            problem = self._dr_embedding_unusable(dr_algo)
            if problem:
                self._release_run_token('cl')
                QMessageBox.warning(
                    self, "DR Embedding Not Usable",
                    f"Cannot cluster on the {dr_algo} embedding:\n\n{problem}\n\n"
                    f"Train {dr_algo} again or choose another embedding"
                    + ("." if algo == 'HDBSCAN' else ", or cluster on the original features."))
                return
            problems = self._dr_embedding_sync_problems(dr_algo)
            if problems:
                bullet_list = '\n'.join(f"  \u2022 {p}" for p in problems)
                reply = QMessageBox.question(
                    self, "DR Embedding Out of Sync",
                    f"The {dr_algo} embedding may not match the current data:\n\n"
                    f"{bullet_list}\n\n"
                    "Clustering will label the embedding's own cells correctly, but "
                    "the marker values behind the violins, heatmaps and Cluster ID "
                    "come from the current gating and unmixing. Samples whose "
                    "embedded cells are not exactly the cells gated now will have no "
                    "marker values stored.\n\n"
                    "Re-running the DR brings it back in sync.\n\n"
                    "Cluster on this embedding anyway?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    self._release_run_token('cl')
                    return

        self._confirm_large_run(
            'cl', algo, params, lambda: self._start_clustering(algo, params))

    def _source_dr_run_entry(self, dr_algo: str) -> dict | None:
        """
        The archived DR run behind state.embeddings[dr_algo]: the run trained
        in this session if there is one, otherwise the newest archived run of
        that algorithm. None when no archived run of it exists.
        """
        candidates = [e for e in self.state.dr_runs if e.get('algorithm') == dr_algo]
        active_id = self._active_dr_run_id.get(dr_algo)
        if active_id:
            # None once this session's run has been deleted
            return next((e for e in candidates if e.get('run_id') == active_id), None)
        return max(candidates, key=lambda e: e.get('timestamp', ''), default=None)

    def _dr_embedding_unusable(self, dr_algo: str) -> str | None:
        """
        Why the live *dr_algo* embedding cannot be clustered on, or None when
        it can: it must be trained, cover every training sample, and come from
        a DR run still in the archive (a deleted run's embedding cannot be
        checked against the current gating and unmixing).
        """
        if self.state.dr_status.get(dr_algo) != 'done':
            return f"No {dr_algo} embedding has been trained."
        embedded = self.state.embeddings.get(dr_algo) or {}
        missing = [rel for rel in self.state.training_sample_ids if rel not in embedded]
        if missing:
            return (f"{len(missing)} of {len(self.state.training_sample_ids)} training "
                    f"samples have no {dr_algo} embedding.")
        if self._source_dr_run_entry(dr_algo) is None:
            return (f"The {dr_algo} run that made this embedding has been deleted, so "
                    "its cells cannot be matched to the current gating and unmixing.")
        return None

    def _drop_orphaned_dr_embeddings(self):
        """
        Clear the live model, embedding and status of each DR algorithm whose
        source run is no longer archived (deleted), so it is neither offered
        for clustering nor shown as trained. Skipped while a DR run is active.
        """
        if self._dr_worker is not None:
            return
        for algo in ConfigTab._CL_DR_PREFERENCE:
            live = (algo in self.state.embeddings or algo in self.state.trained_reducers
                    or self.state.dr_status.get(algo) == 'done')
            if not live or self._source_dr_run_entry(algo) is not None:
                continue
            for store in (self.state.embeddings, self.state.embedding_rows,
                          self.state.trained_reducers,
                          self.state.dr_timestamps, self._active_dr_run_id,
                          self._dr_train_params, self._dr_pool_size):
                store.pop(algo, None)
            self.state.dr_status[algo] = 'idle'
            _log.info("Dropped the live %s embedding: its DR run is no longer archived",
                      algo)

    def _dr_embedding_sync_problems(self, dr_algo: str) -> list[str]:
        """
        Reasons the live *dr_algo* embedding may not line up with the current
        gating and unmixing; empty when it does (or no archived run exists to
        compare against).
        """
        entry = self._source_dr_run_entry(dr_algo)
        if entry is None:
            return []
        problems = []
        run_gates = list(entry.get('gates', []))
        if sorted(run_gates) != sorted(self.state.selected_gates):
            problems.append(
                f"The DR run used gate(s) {', '.join(run_gates) or 'none'}; the "
                f"Configuration tab now selects "
                f"{', '.join(self.state.selected_gates) or 'none'}.")
        status = drc_run_archive.sync_status(
            entry, drc_pipeline.sync_fingerprint(
                self.controller, run_gates, entry.get('training_sample_ids', [])))
        if status == drc_run_archive.SYNC_OUT_OF_SYNC:
            problems.append(
                "The unmixing, AF profile assignment or gating has changed since "
                f"\"{entry.get('label', '')}\" was made.")
        elif status == drc_run_archive.SYNC_UNKNOWN:
            problems.append(
                f"\"{entry.get('label', '')}\" was made before sync tracking was "
                "added, so it cannot be checked against the current data.")
        return problems

    def _start_clustering(self, algo: str, params: dict):
        """Start the background clustering worker (inputs already validated)."""
        if hasattr(self, 'config_tab'):
            self.config_tab.cl_run_btn.setEnabled(False)
            self._set_run_lock('cl', True)
            self.config_tab.cl_status_label.setText("⏳ Running …")
            self.config_tab.cl_status_label.setStyleSheet("color: orange;")
            if hasattr(self.config_tab, 'cl_progress_bar'):
                bar = self.config_tab.cl_progress_bar
                bar.setRange(0, 0)          # indeterminate
                bar.setTextVisible(False)
                bar.setVisible(True)

        # Snapshot AF/transfer-matrix state on the main thread before the
        # worker starts — see drc_pipeline.apply_unmixing_af_aware() docstring.
        af_state = drc_pipeline.snapshot_af_state(self.controller)

        # Which cells the labels will belong to: the embedding's own for a
        # DR-space run (so it inherits the DR run's fingerprints), the
        # current gating's otherwise.
        if drc_clustering.clusters_in_dr_space(algo, params):
            source_dr = self._source_dr_run_entry(params.get('_dr_algo') or '') or {}
            # The run's DR positions are this run's embeddings
            params['_dr_run_id'] = source_dr.get('run_id') or ''
            run_fingerprint = source_dr.get('sync_fingerprint')
            run_split = {'unmixing': source_dr.get('unmixing_fingerprint'),
                         'gating': source_dr.get('gating_fingerprint')}
        else:
            run_fingerprint = drc_pipeline.sync_fingerprint(
                self.controller, self.state.selected_gates,
                self.state.training_sample_ids)
            run_split = drc_pipeline.split_fingerprints(
                self.controller, self.state.selected_gates,
                self.state.training_sample_ids)
        channels_now = [c for c in self.state.selected_channels
                        if c not in drc_pipeline.META_CHANNELS]
        transform_params = self._run_transform_params(channels_now)
        unaligned_samples: list[str] = []

        def _checked_progress(msg):
            # Raises in the worker thread once the run is cancelled
            if worker.cancelled:
                raise _RunCancelled()
            self.progress_message(msg)

        def _do_cluster():
            return drc_clustering.run_clustering_staged(
                self.controller, self.state, algo, params,
                progress=_checked_progress,
                af_state=af_state,
            )

        def _on_cl_done(ok, err):
            if worker is not self._cl_worker or worker.cancelled:
                return
            if not ok:
                if err != 'Cancelled.':
                    self.progress_message(f"Clustering error: {err}")
                    QMessageBox.critical(self, "Clustering Error", err)
            else:
                staged = worker.result
                drc_clustering.commit_clustering_result(self.state, staged, algo)
                self._learn_run_time(algo, params, staged.stage_seconds, staged.stage_events)
                # Archive this run so GroupsStatsTab can select it later.
                if self.state.cluster_labels:
                    channels = [c for c in self.state.selected_channels
                                if c not in drc_pipeline.META_CHANNELS]
                    n_events = sum(len(a) for a in self.state.cluster_labels.values())
                    # Cluster Tree view (Workspace) needs the node/centroid
                    # weights, node-to-cluster mapping and per-node counts
                    # archived alongside the run — not just the live
                    # in-memory copy, which a later run of the same
                    # algorithm would overwrite. Works for any clustering
                    # algorithm now: FlowSOM stores its SOM codebook here,
                    # Leiden/HDBSCAN store per-cluster centroids
                    # (build_centroid_tree_data) -- same dict shape either way.
                    tree_data = None
                    reducer_state = self.state.trained_reducers.get(algo)
                    if isinstance(reducer_state, dict) and 'node_weights' in reducer_state:
                        tree_data = dict(reducer_state)
                    entry = drc_run_archive.archive_clustering_run(
                        self.controller, self.state,
                        algorithm=algo,
                        cluster_labels=dict(self.state.cluster_labels),
                        colors=dict(self.state.cluster_colors),
                        # Every new run starts with its own empty names dict
                        names={},
                        n_clusters=self.state.n_clusters,
                        gates=list(self.state.selected_gates),
                        training_sample_ids=list(self.state.training_sample_ids),
                        channels=channels,
                        params=dict(params),
                        n_events=n_events,
                        marker_values=dict(self.state.cluster_marker_values),
                        tree_data=tree_data,
                        sync_fingerprint=run_fingerprint,
                        event_rows=_complete_rows(self.state.cluster_labels,
                                                  self.state.cluster_rows),
                        split_fingerprints=run_split,
                        transform_params=transform_params,
                    )
                    self.progress_message(
                        f"Run archived as \"{entry['label']}\""
                    )
                    unaligned_samples.extend(
                        rel for rel in self.state.cluster_labels
                        if rel not in self.state.cluster_marker_values)
            self._release_clustering_ui()
            if hasattr(self, 'config_tab'):
                self.config_tab.run_table.refresh()
            self._on_runs_changed()
            self._cl_worker = None
            if unaligned_samples:
                QMessageBox.warning(
                    self, "Marker Values Not Stored",
                    f"For {len(unaligned_samples)} of {len(self.state.cluster_labels)} "
                    "sample(s), the events under the current gating and unmixing do "
                    "not line up with the events that were clustered (for example "
                    f"{unaligned_samples[0]}). The cluster labels are saved, but no "
                    "marker values could be stored for those samples, so violin "
                    "plots, heatmaps and Cluster ID will leave them out.\n\n"
                    "Re-run the DR (and then the clustering) on the current "
                    "gating and unmixing to bring them back in sync.")

        worker = _ClusterWorker(_do_cluster)
        worker.finished.connect(_on_cl_done)
        token = self._run_tokens.get('cl')
        if token is not None:
            # after _on_cl_done, and also if it raised
            worker.finished.connect(lambda _ok, _err: self._release_cl_token(token))
        self._cl_worker = worker
        if hasattr(self, 'config_tab'):
            self.config_tab.cl_cancel_btn.setEnabled(True)
        worker.start()

    def _release_cl_token(self, token):
        """Release *token* if it is still the clustering run's hold."""
        if self._run_tokens.get('cl') is token:
            self._release_run_token('cl')

    def _release_clustering_ui(self):
        """Unlock the clustering controls and hide its progress after a run ends."""
        self._release_run_token('cl')
        if not hasattr(self, 'config_tab'):
            return
        ct = self.config_tab
        ct.cl_run_btn.setEnabled(True)
        self._set_run_lock('cl', False)
        ct._refresh_cl_status()
        if hasattr(ct, 'cl_progress_bar'):
            ct.cl_progress_bar.setVisible(False)

    def _cancel_clustering(self):
        """
        Cancel the running clustering job.  The fit stops at its next progress
        report; a step that cannot be interrupted (SOM training, kNN build,
        Leiden, HDBSCAN) finishes in the background.  A run's results are
        committed to the state only when it completes uncancelled, so a
        cancelled run changes nothing.  The UI is released immediately.
        """
        if self._cancel_estimate('cl'):
            return
        worker = self._cl_worker
        if worker is None or not worker.isRunning():
            return
        worker.detach()
        self._detached_workers = [w for w in self._detached_workers if w.isRunning()]
        self._detached_workers.append(worker)
        self._cl_worker = None
        self._release_clustering_ui()
        self.progress_message(
            "Clustering cancelled.  A step that cannot be interrupted may keep "
            "running in the background; its result is discarded.")
