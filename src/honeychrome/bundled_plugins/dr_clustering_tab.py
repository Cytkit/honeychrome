"""
dr_clustering_tab.py — DR / Clustering / Statistics Plugin for Honeychrome
===========================================================================
Honeychrome plugin providing:
  • Dimensionality reduction  — UMAP, PaCMAP
  • Clustering               — FlowSOM, Leiden, HDBSCAN
  • Transform inspector      — read-only preview of Logicle parameters
  • Group & statistics       — sample grouping, limma differential analysis
  • Workspace                — customisable matplotlib scatter-plot canvas

Plugin contract (required by plugin_loaders.py):
  plugin_name  str           — displayed as the main-window tab title
  PluginWidget(QWidget)      — instantiated with (bus, controller)

Distribution:
  Place in  honeychrome/plugin_templates/dr_clustering_tab.py
  main.py copies it to  ~/Experiments/plugins/  on first launch.
  Users enable it in App Configuration → restart → tab appears.

Dependencies:
  Heavy ML packages are installed automatically on first load via
  _bootstrap() below.  No user Python interaction is required.

Stage 1 — skeleton (complete):
  PipelineState, PluginWidget, 4-tab QTabWidget, bootstrap, bus wiring.

Stage 2 — Transforms tab (complete):
  Transform preview, auto-transform, CSV import/export, QSettings persistence.

Stage 3 — DR pipeline (complete):
  ConfigTab DR section: algorithm radios, per-algorithm hyperparameter panels,
  Train / Apply to All Samples buttons, status labels.
  _run_umap(), _run_pacmap() with hnswlib kNN.
  _apply_dr_to_all_samples() projects every sample.

Stage 4 — Clustering (complete):
  ConfigTab clustering section: FlowSOM / Leiden / HDBSCAN radio buttons
  with per-algorithm hyperparameter panels.
  FlowSOM: SOM grid, manual metacluster count (2-200).
  Leiden: reuses UMAP hnswlib kNN; igraph + leidenalg community detection.
  HDBSCAN: applied to UMAP embedding if available, else raw feature space.
  All algorithms assign labels to every sample via nearest-centroid.
  Cluster colours from cc.glasbey; noise label −1 → grey.

Stage 5 — Workspace basics (complete):
  WorkspaceTab with scrollable PlotCard grid, matplotlib scatter, cluster
  colouring, DR / sample selectors, PNG export, PDF export.

Stage 6 — Groups & Stats (complete):
  GroupsStatsTab: named groups with regex auto-population, limma statistics
  (inmoose), heatmap+dendrogram, volcano plot, CSV result export.

Stage 7 — Advanced workspace (complete):
  T-REX scatter colouring, per-cluster colour pickers (right-click), magic-wand
  display-config copy/paste, multi-page PDF export.
"""

# ---------------------------------------------------------------------------
# 0.  Dependency bootstrap — runs once at module load
#     Installs any missing ML packages into the same Python environment
#     that Honeychrome is using.  Silent on subsequent launches.
# ---------------------------------------------------------------------------
import os as _os
import sys
import subprocess
import warnings
from importlib import import_module

# Set Numba threading layer at module-import time, before any JIT compilation.
# 'omp' ships with llvmlite (a hard Numba dep) and is always available.
# Unlike the default 'workqueue' layer, 'omp' is safe when two Numba-compiled
# functions (e.g. UMAP and FlowSOM) are running concurrently on separate
# QThreads.  Must be set before the first numba import, which is why it lives
# here at module scope rather than inside _bootstrap().
if 'NUMBA_THREADING_LAYER' not in _os.environ:
    _os.environ['NUMBA_THREADING_LAYER'] = 'omp'

_REQUIRED_PACKAGES = {
    'umap':       'umap-learn',
    'openTSNE':   'openTSNE',
    'pacmap':     'pacmap',
    'leidenalg':  'leidenalg',
    'igraph':     'python-igraph',
    'hnswlib':    'hnswlib',
    'inmoose':    'inmoose',
}



def _can_import(module_name: str) -> bool:
    """Return True if *module_name* can be imported without error."""
    try:
        # Suppress warnings during import check so noisy packages
        # (umap TensorFlow check, faiss/hnswlib Swig types) don't pollute the log.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            import_module(module_name)
        return True
    except ImportError:
        return False


def _bootstrap():
    """Install any missing plugin dependencies into the current environment."""
    missing = [
        pip_name
        for mod_name, pip_name in _REQUIRED_PACKAGES.items()
        if not _can_import(mod_name)
    ]
    if missing:
        print(f"[DR/Clustering Plugin] First-time setup: installing {missing} …")
        print("[DR/Clustering Plugin] This may take up to a minute.")
        try:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '--quiet'] + missing
            )
            print("[DR/Clustering Plugin] Dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[DR/Clustering Plugin] WARNING: could not install some packages: {e}")
            print("[DR/Clustering Plugin] Some features may be unavailable.")


    # Suppress noisy third-party warnings.
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


# _bootstrap() is intentionally NOT called here at module scope.
# Module import runs on a background QThread (see _PluginLoaderWorker in
# main_window.py).  Calling subprocess.check_call or importing heavy packages
# (pyqtgraph, colorcet) on a non-main thread triggers QObject::setParent
# warnings.  _bootstrap() is called instead from PluginWidget.__init__,
# which runs on the main thread via instantiate_plugin_widgets().


# Third-party warning filters are applied inside _bootstrap() so they
# take effect on the main thread after PluginWidget is instantiated.


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
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
import csv
import math
import re
import time
import traceback

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QTimer, QSettings, QEvent, QRectF, QThread, Signal
from PySide6.QtGui import QPen, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QPushButton, QLabel, QComboBox, QGroupBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QRadioButton,
    QButtonGroup, QListWidget, QListWidgetItem, QListView,
    QAbstractItemView, QSplitter, QFrame, QGridLayout,
    QLineEdit, QFileDialog, QMessageBox, QSizePolicy,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QTabWidget, QColorDialog, QProgressBar,
    QDialog, QTextEdit,
)

from honeychrome.controller_components.functions import (
    sample_from_fcs,
    apply_gates_in_place,
)
from honeychrome.controller_components.transform import Transform
from honeychrome.view_components.busy_cursor import with_busy_cursor
from honeychrome.view_components.clear_layout import clear_layout
from honeychrome.view_components.exportable_plot_widget import ExportablePlotWidget
from honeychrome.view_components.ordered_multi_sample_picker import OrderedMultiSamplePicker
from honeychrome.view_components.copyable_table_widget import CopyableTableWidget
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

import drc_logging
import drc_pipeline
import drc_clustering
import drc_stats
import drc_run_archive
import drc_gate_tree
import drc_scatter

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
plugin_name = 'DR / Clustering / Statistics'


# ---------------------------------------------------------------------------
# 3.  Module-level helper functions
# ---------------------------------------------------------------------------

def apply_gate_by_lookup_table(cytometry_data_dictionary, gate_name):
    """
    Return the subset of event_data belonging to *gate_name*.

    Uses the fast lookup-table approach from the Honeychrome example plugin.
    Always operate on a copy of the cytometry data dictionary to avoid
    mutating the main application state.

    Parameters
    ----------
    cytometry_data_dictionary : dict
        Copy of controller.data_for_cytometry_plots_unmixed with
        'event_data' replaced by the current sample's unmixed data.
    gate_name : str
        Name of the gate to apply (e.g. 'Singlets', 'root').

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
    mask = cytometry_data_dictionary['gate_membership'][gate_name]
    return cytometry_data_dictionary['event_data'][mask]


def apply_gates_union_by_lookup_table(cytometry_data_dictionary, gate_names: list[str]):
    """
    Return the UNION of event_data belonging to ANY gate in *gate_names*.
    Same lookup-table approach as apply_gate_by_lookup_table() above; see
    drc_pipeline.apply_gates_union_by_lookup_table() for the identical
    sibling used by the data-pipeline module.

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
    result = {}
    for channel, params in transforms.items():
        if not isinstance(params, dict):
            continue
        result[channel] = {
            'T': float(params.get('scale_t',   262144.0)),
            'W': float(params.get('logicle_w', 0.5)),
            'M': float(params.get('logicle_m', 4.5)),
            'A': float(params.get('logicle_a', 0.0)),
            'id': params.get('id', 1),   # 0=linear, 1=logicle, 2=log
            'limits': list(params.get('limits', [0, 1])),
        }
    return result

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

def _apply_channel_transform(state, ch: str, values: np.ndarray) -> np.ndarray:
    """
    Apply the SAME transform configured in the Transforms tab to raw
    values, for display purposes (e.g. violin plots) -- builds a Transform
    from state.channel_transform_params[ch] and uses its xform.apply(),
    the same conversion TransformTab itself uses to go from raw to
    displayed scale. Returns values unchanged if no transform is configured
    for this channel yet.
    """
    params = state.channel_transform_params.get(ch)
    if not params:
        return values
    tr = Transform(
        scale_t=params['T'], logicle_w=params['W'],
        logicle_m=params['M'], logicle_a=params['A'],
    )
    tr.set_transform(id=params['id'], limits=params.get('limits', [0, 1]))
    if tr.xform is None:      # 'default'/time-gate case -- no transform to apply
        return values
    return tr.xform.apply(values)

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


def _downsample(data: np.ndarray, n: int, rng=None) -> np.ndarray:
    """
    Return a random subset of *n* rows from *data*.
    If data has fewer than *n* rows, return data unchanged.

    Parameters
    ----------
    data : np.ndarray  shape (n_events, n_channels)
    n    : int         target number of events
    rng  : np.random.Generator | None
    """
    if len(data) <= n:
        return data
    rng = rng or np.random.default_rng()
    idx = rng.choice(len(data), size=n, replace=False)
    return data[idx]


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
    channel_transform_params is populated from the experiment model at
    initialisation and when the tab is activated.  It is NEVER written back
    to the experiment.  Any local modifications (e.g. from the auto-transform
    preview in the Transform tab) stay here only.
    """

    # --- Configuration ---
    selected_gates: list[str] = field(default_factory=list)
    # Checked gate names (§0.3).  A run is built from the UNION of events
    # across all listed gates.  Empty list = no gate selected.  Until Item 4
    # replaces the gate combo with the checkable gate tree, the UI only ever
    # sets this to a 0- or 1-element list.
    selected_channels: list[str] = field(default_factory=list)
    training_sample_ids: list[str] = field(default_factory=list)
    n_training_events: int = 10_000

    # --- Transforms (read from experiment; never written back) ---
    channel_transform_params: dict = field(default_factory=dict)
    # {channel_name: {'W': float, 'A': float, 'T': float, 'M': float}}

    # --- Dimensionality reduction ---
    trained_reducers: dict[str, Any] = field(default_factory=dict)
    # {algorithm_name: fitted reducer object} — current/most-recent model
    dr_status: dict[str, str] = field(default_factory=dict)
    # {algorithm_name: 'idle' | 'done' | 'error'}
    dr_timestamps: dict[str, str] = field(default_factory=dict)
    # {algorithm_name: ISO timestamp string}
    embeddings: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    # {algorithm_name: {sample_path: np.ndarray shape (n_events, 2)}} — current/most-recent only
    embedding_features: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    # {algorithm_name: {sample_path: np.ndarray shape (n_events, n_channels)}} —
    # the ORIGINAL high-dimensional feature vectors each embedding row came
    # from (same feature space the reducer was fit on). Cached alongside
    # embeddings precisely so T-REX (or anything else needing true
    # marker-space neighbours) never has to re-derive them from live data.
    umap_knn_index: Any | None = None
    # hnswlib index built during UMAP training; reused by Leiden and T-REX
    dr_runs: list[dict] = field(default_factory=list)
    # Ordered archive of completed DR training runs (§0.2), parallel to
    # clustering_runs below.  Each entry (manifest fields + heavy payload):
    # {
    #   'run_id': str, 'kind': 'dr', 'label': str, 'algorithm': str,
    #   'gates': list[str], 'training_sample_ids': list[str],
    #   'n_samples': int, 'n_events': int, 'channels': list[str],
    #   'params': dict, 'timestamp': str, 'n_clusters': None,
    #   'reducer': Any, 'embeddings': dict[str, np.ndarray],
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
    # {sample_path: (raw_values: np.ndarray, channel_names: list[str])} --
    # snapshotted at the SAME instant labels are assigned (see
    # drc_clustering.py's _snapshot_marker_values), so guaranteed
    # row-for-row aligned to cluster_labels even if gates/channels are
    # edited later. Archived into clustering_runs['marker_values'].
    cluster_dr_positions: dict[str, np.ndarray] = field(default_factory=dict)
    # {sample_path: np.ndarray} -- the exact DR-embedding rows a DR-space
    # clustering run assigned labels against. Only populated when
    # clustering ran in DR-embedding space. Archived into
    # clustering_runs['dr_positions'] so the Cluster Annotation map can
    # still plot correct positions even if the live state.embeddings for
    # that algorithm have since been overwritten by a later DR run.
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
    # {sample_path: group_name | 'Unassigned'} — Item 13: group_name is a
    # free-form, user-defined string (there is no longer a fixed 'A'/'B'
    # slot; the group's NAME is the value directly). 'Unassigned' is a
    # reserved sentinel and can never itself be added as a group name.
    group_names: list[str] = field(default_factory=lambda: ['A', 'B'])
    # Ordered list of currently-defined group names (Item 13 — replaces the
    # old fixed two-slot model). Display order in the UI == this order.
    group_patterns: dict[str, str] = field(default_factory=dict)
    # {group_name: filename regex} for "Auto-assign by pattern" (Item 13 —
    # replaces the old group_a_pattern/group_b_pattern QLineEdit pair).
    compare_group_a: str = ''
    compare_group_b: str = ''
    # Item 13 — T-REX's own two-group pair (its neighbour-fraction score is
    # only defined for two conditions, so it never grew past Phase 1 — see
    # §0.2 point 2). Run Statistics/Confusion Matrix/Composition-by-group
    # use testing_group_selection below instead, as of Phase 2.
    covariates: pd.DataFrame | None = None
    # rows = samples (rel-path index), columns = covariate names, all
    # values str. Item 13 phase 2: now actually populated by CSV import
    # (see _import_csv) — previously defined but never written to.

    # --- Item 13 phase 2: N-group testing ---
    testing_group_selection: list[str] = field(default_factory=list)
    # Which defined groups participate in Frequency/Counts/MFI testing,
    # Confusion Matrix, and Composition-by-group. Empty means "not yet
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
    stats_comparisons: list[tuple[str, str]] = field(default_factory=list)
    # (baseline, other) pairs actually tested by the last Run Statistics
    # call, aligned 1:1 with the unique values of freq_results/mfi_results/
    # counts_results' 'comparison' column, in the same order. Lets the tab
    # recover which two groups a comparison belongs to without parsing the
    # display string.

    # --- Statistics ---
    freq_results: pd.DataFrame | None = None
    # limma output for cluster frequencies
    freq_df: pd.DataFrame | None = None
    # raw (samples × clusters) frequency matrix
    counts_results: pd.DataFrame | None = None
    # negative-binomial GLM output for cluster raw counts (Item 14) —
    # a parallel abundance test alongside freq_results, not a replacement.
    counts_df: pd.DataFrame | None = None
    # raw (samples × clusters) count matrix
    mfi_results: pd.DataFrame | None = None
    # limma output for per-cluster marker MFIs
    mfi_df: pd.DataFrame | None = None
    # raw (samples × cluster·channel) log1p-MFI matrix
    mfi_sample_df: pd.DataFrame | None = None
    # raw (samples × channel) log1p-MFI matrix, whole-sample aggregate —
    # what the MFI Heatmap actually draws (no cluster breakdown, no
    # significance filter). mfi_df/mfi_results stay cluster-level, for
    # the MFI Volcano.

    # Confusion Matrix / Composition Barplot (Items 10 & 12) — independent
    # of Run Statistics, so persisted separately so they survive reopen
    # the same way freq/mfi results already do.
    confusion_df: pd.DataFrame | None = None
    confusion_run_label: str = ''
    confusion_names: dict[int, str] = field(default_factory=dict)
    composition_df: pd.DataFrame | None = None
    composition_as_pct: bool = True
    composition_group_var: str = 'sample'
    composition_run_label: str = ''
    composition_names: dict[int, str] = field(default_factory=dict)
    marker_roles: dict[str, str] = field(default_factory=dict)
    # {channel_name: 'type' | 'state'} — Item 11 (diffcyt-style split).
    # 'type' channels drove the active/selected clustering run and are
    # excluded from MFI significance testing by default, to avoid the same
    # channel driving both the cluster assignment and its own significance
    # call. A channel with no entry yet defaults to 'type' if it was in the
    # selected run's training channel set, else 'state' — see
    # GroupsStatsTab._populate_marker_roles_list(). User-overridable (some
    # markers, e.g. HLA-DR, genuinely serve double duty).
    stats_all_rel: list = field(default_factory=list)
    # sample rel-paths in limma row order
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

    # --- T-REX ---
    trex_knn_index: Any | None = None
    # hnswlib index over pooled Group A + B events
    trex_knn_group_labels: np.ndarray | None = None
    # group label ('A' or 'B') per row in trex_knn_index
    trex_scores: dict[str, np.ndarray] = field(default_factory=dict)
    # {sample_path: np.ndarray float32 in [-1, +1]}
    trex_k: int = 30
    trex_dr_run_id: str = ''
    # Which archived DR run's embeddings trex_scores was computed against --
    # T-REX can only ever be plotted against THIS run's events.

    # --- Workspace ---
    plot_configs: list[dict] = field(default_factory=list)
    # One dict per PlotCard on the workspace canvas
    display_clipboard: dict | None = None

    # --- Data pipeline cache (perf) ---
    gated_data_cache: dict[tuple, np.ndarray] = field(default_factory=dict)
    # {(abs_path_str, sorted_gates_tuple, id(transfer_matrix)): gated array}
    # Populated by drc_pipeline.load_unmixed_gated(). Avoids re-reading and
    # re-unmixing an FCS file from disk on every PlotCard.refresh() — the
    # single biggest cost in Workspace "Marker" colouring. Intentionally
    # unbounded for the life of the session (same pattern as the existing
    # threshold_cache in autospectral_optimization_functions.py); a stale
    # entry is simply never looked up again once its key's inputs change.
    # Holds a copied PlotCard display config for magic-wand paste
    workspace_n_columns: int = 3

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def stats_runnable(self) -> bool:
        """
        Return True if the two selected "Compare" groups (Item 13) are
        distinct and each has >= 3 assigned samples.
        """
        if not self.compare_group_a or not self.compare_group_b or \
                self.compare_group_a == self.compare_group_b:
            return False
        counts = self.n_per_group()
        return (counts.get(self.compare_group_a, 0) >= 3 and
                counts.get(self.compare_group_b, 0) >= 3)

    def n_per_group(self) -> dict[str, int]:
        """Return {group_name: assigned_sample_count} for every defined group."""
        counts = {name: 0 for name in self.group_names}
        for g in self.sample_groups.values():
            if g in counts:
                counts[g] += 1
        return counts

    def n_group_stats_runnable(self) -> bool:
        """
        Item 13 phase 2. True if at least 2 of the checked 'Groups to Test'
        each have >= 3 assigned samples. Governs Run Statistics/Confusion
        Matrix/Composition-by-group -- separate from stats_runnable(),
        which still gates T-REX's fixed Compare pair.
        """
        selection = self.testing_group_selection or self.group_names
        counts = self.n_per_group()
        qualifying = sum(1 for name in selection if counts.get(name, 0) >= 3)
        return qualifying >= 2

    def available_algorithms(self) -> list[str]:
        """Return list of DR algorithms that have completed training."""
        return [a for a, s in self.dr_status.items() if s == 'done']

    def invalidate_trex(self):
        """Call whenever group assignments change — clears cached T-REX data."""
        self.trex_knn_index = None
        self.trex_knn_group_labels = None
        self.trex_scores = {}

    def cluster_label(self, cluster_id: int) -> str:
        """Return display name for a cluster id, falling back to str(id)."""
        if cluster_id < 0:
            return self.cluster_names.get(cluster_id, 'Noise')
        return self.cluster_names.get(cluster_id, str(cluster_id))

    def initialise_sample_groups(self, sample_paths: list[str]):
        """
        Add 'Unassigned' entries for any sample not yet in sample_groups.
        Remove entries for samples no longer in the experiment.
        Safe to call repeatedly.
        """
        current = set(sample_paths)
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
# 5a.  Run management table (Item 6) — used by ConfigTab only.
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
    Item 6 pop-out 'View' dialog for one archived run — full gate list,
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
    Item 6 run management table.  Lives at the bottom of ConfigTab.

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
        for row, entry in enumerate(entries):
            self._populate_row(row, entry)
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
            label_item.setToolTip(
                "Payload file missing from disk — this run's data can no "
                "longer be loaded."
            )
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
        manifest entry, unlinks its pickle, and drops it from
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
    Tab 0 — Analysis Configuration
    --------------------------------
    Gate selector, channel checkboxes, training-sample picker,
    training-event count, DR algorithm + hyperparameter panels,
    clustering algorithm + hyperparameter panels.

    Stage 1: placeholder layout.
    Stage 3: DR controls implemented.
    Stage 4: Clustering controls implemented.
    """

    def __init__(self, state: PipelineState, bus, controller, parent=None):
        super().__init__(parent)
        self.state = state
        self.bus = bus
        self.controller = controller
        self._build_ui()

    def _build_ui(self):
        # Outer scrollable container — config panels can be tall
        self.content_widget = QWidget()
        self.main_layout = QVBoxLayout(self.content_widget)
        self.main_layout.setAlignment(Qt.AlignTop)
        self.main_layout.setSpacing(10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self.content_widget)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # ------------------------------------------------------------------
        # Shared controls
        # ------------------------------------------------------------------
        shared_box = QGroupBox("Data Selection")
        shared_layout = QVBoxLayout(shared_box)

        # Gate selector — checkable multi-select tree (§0.3).  Same tree
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

        # Training sample picker
        shared_layout.addWidget(QLabel("Training samples:"))
        self.picker = OrderedMultiSamplePicker(title="Choose Training Samples")
        self.picker.changed.connect(self._on_training_samples_changed)
        shared_layout.addWidget(self.picker)

        self.main_layout.addWidget(shared_box)

        # ------------------------------------------------------------------
        # Dimensionality Reduction — Stage 3
        # ------------------------------------------------------------------
        dr_box = QGroupBox("Dimensionality Reduction")
        dr_box.setCheckable(True)
        dr_box.setChecked(True)
        dr_layout = QVBoxLayout(dr_box)

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
        self.umap_n_epochs.setValue(500)
        self.umap_n_epochs.setToolTip(
            "Number of optimisation epochs.\n"
            "More epochs = finer embedding; also slower.\n"
            "Default: 500"
        )
        umap_grid.addWidget(self.umap_n_epochs, 1, 3)

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

        tsne_grid.addWidget(QLabel('n_iter (max):'), 0, 2)
        self.tsne_n_iter = QSpinBox()
        self.tsne_n_iter.setRange(250, 5000)
        self.tsne_n_iter.setValue(1000)
        self.tsne_n_iter.setToolTip('Maximum iterations.  Default: 1000')
        tsne_grid.addWidget(self.tsne_n_iter, 0, 3)

        tsne_grid.addWidget(QLabel('n_jobs:'), 1, 0)
        self.tsne_n_jobs = QSpinBox()
        self.tsne_n_jobs.setRange(-1, 64)
        self.tsne_n_jobs.setValue(-1)
        self.tsne_n_jobs.setSpecialValueText('all cores')
        self.tsne_n_jobs.setToolTip('-1 = use all available cores.  Default: -1')
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

        self.dr_cancel_btn = QPushButton("✕  Cancel")
        self.dr_cancel_btn.setFixedHeight(30)
        self.dr_cancel_btn.setToolTip("Cancel the running DR job.")
        self.dr_cancel_btn.clicked.connect(self._on_cancel_dr_clicked)
        self.dr_cancel_btn.setEnabled(False)
        self.dr_cancel_btn.setStyleSheet("color: #c0392b;")
        dr_run_row.addWidget(self.dr_cancel_btn)

        dr_run_row.addStretch()
        dr_layout.addLayout(dr_run_row)

        self.dr_status_label = QLabel("No model trained.")
        self.dr_status_label.setStyleSheet("color: grey;")
        dr_layout.addWidget(self.dr_status_label)

        self.dr_progress_bar = QProgressBar()
        self.dr_progress_bar.setRange(0, 100)
        self.dr_progress_bar.setTextVisible(True)
        self.dr_progress_bar.setFixedHeight(14)
        self.dr_progress_bar.setVisible(False)
        dr_layout.addWidget(self.dr_progress_bar)

        self.main_layout.addWidget(dr_box)

        # Connect algo radio buttons → show/hide param panels + update note
        self._dr_algo_group.buttonClicked.connect(self._on_dr_algo_changed)
        self._on_dr_algo_changed()   # set initial state

        # ------------------------------------------------------------------
        # Clustering — Stage 4
        # ------------------------------------------------------------------
        cl_box = QGroupBox("Clustering")
        cl_box.setCheckable(True)
        cl_box.setChecked(True)
        cl_layout = QVBoxLayout(cl_box)

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
            "Neighbours for the kNN graph.\n"
            "If UMAP was run, its hnswlib index is reused and this value is ignored."
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

        # ---- Run button + status ----
        cl_run_row = QHBoxLayout()
        self.cl_run_btn = QPushButton("▶  Run Clustering")
        self.cl_run_btn.setFixedHeight(30)
        self.cl_run_btn.clicked.connect(self._on_run_clustering_clicked)
        cl_run_row.addWidget(self.cl_run_btn)
        cl_run_row.addStretch()
        cl_layout.addLayout(cl_run_row)

        self.cl_status_label = QLabel("No clustering run.")
        self.cl_status_label.setStyleSheet("color: grey;")
        cl_layout.addWidget(self.cl_status_label)

        self.cl_progress_bar = QProgressBar()
        self.cl_progress_bar.setRange(0, 0)   # indeterminate by default
        self.cl_progress_bar.setTextVisible(False)
        self.cl_progress_bar.setFixedHeight(14)
        self.cl_progress_bar.setVisible(False)
        cl_layout.addWidget(self.cl_progress_bar)

        self.main_layout.addWidget(cl_box)

        # ---- Archived Runs (Item 6) ----
        runs_box = QGroupBox("Archived Runs")
        runs_box_layout = QVBoxLayout(runs_box)

        runs_hint = QLabel(
            "Double-click a run's label to rename it, or any other cell to "
            "view its full configuration. Renaming or deleting here updates "
            "every run selector elsewhere in the plugin immediately."
        )
        runs_hint.setWordWrap(True)
        runs_hint.setStyleSheet("color: grey; font-style: italic; font-size: 10px;")
        runs_box_layout.addWidget(runs_hint)

        self.run_table = RunManagementTable(self.controller, self.state)
        runs_box_layout.addWidget(self.run_table)

        runs_btn_row = QHBoxLayout()
        runs_btn_row.addStretch()
        self.delete_run_btn = QPushButton("Delete Selected Run")
        self.delete_run_btn.setEnabled(False)
        self.delete_run_btn.setToolTip("Select a run above to delete it.")
        self.delete_run_btn.clicked.connect(self._on_delete_run_clicked)
        runs_btn_row.addWidget(self.delete_run_btn)
        runs_box_layout.addLayout(runs_btn_row)

        self.run_table.itemSelectionChanged.connect(self._on_run_table_selection_changed)

        self.main_layout.addWidget(runs_box)
        self.main_layout.addStretch()

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

    def _on_training_samples_changed(self, ordered_list: list):
        """Keep state.training_sample_ids in sync with the picker."""
        self.state.training_sample_ids = list(ordered_list)

    def _on_run_table_selection_changed(self):
        self.delete_run_btn.setEnabled(self.run_table.selected_run_id() is not None)

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
        ts     = self.state.dr_timestamps.get(algo, '')
        running = (status == 'running')
        if status == 'done':
            n_emb = len(self.state.embeddings.get(algo, {}))
            self.dr_status_label.setText(
                f"✓ Trained  ({ts})  —  embeddings: {n_emb} sample(s)"
            )
            self.dr_status_label.setStyleSheet("color: green;")
            self.dr_apply_btn.setEnabled(True)
        elif status == 'error':
            self.dr_status_label.setText(f"✗ Error during last run  ({ts})")
            self.dr_status_label.setStyleSheet("color: red;")
            self.dr_apply_btn.setEnabled(False)
        elif status == 'running':
            self.dr_status_label.setText("⏳ Running — Honeychrome remains usable …")
            self.dr_status_label.setStyleSheet("color: orange;")
            self.dr_apply_btn.setEnabled(False)
        else:
            self.dr_status_label.setText("No model trained.")
            self.dr_status_label.setStyleSheet("color: grey;")
            self.dr_apply_btn.setEnabled(False)
        self.dr_run_btn.setEnabled(not running)
        self.dr_cancel_btn.setEnabled(running)

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

        notes = {
            'FlowSOM': (
                "FlowSOM: self-organising map → metaclustering."
            ),
            'Leiden': (
                "Leiden: graph community detection.  Reuses UMAP kNN if available.\n"
                "Higher resolution = more fine-grained clusters."
            ),
            'HDBSCAN': (
                "HDBSCAN: density-based clustering, no fixed distance threshold.\n"
                "Works on both the raw feature space and a DR embedding without\n"
                "retuning.  Events labelled −1 (noise) are shown in grey."
            ),
        }
        self._cl_algo_note.setText(notes.get(algo, ''))
        self._refresh_cl_status()

    def _refresh_cl_status(self):
        algo = self.state.active_clustering_algorithm
        n_cl = self.state.n_clusters
        if algo and n_cl is not None:
            self.cl_status_label.setText(
                f"✓ Last run: {algo}  —  {n_cl} cluster(s)"
            )
            self.cl_status_label.setStyleSheet("color: green;")
        else:
            self.cl_status_label.setText("No clustering run.")
            self.cl_status_label.setStyleSheet("color: grey;")

    def _refresh_cl_space_combo(self):
        """Populate the DR embedding combo with any trained DR algorithms."""
        if not hasattr(self, '_cl_dr_combo'):
            return
        available = self.state.available_algorithms()
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

        # Identify scatter channels to pre-uncheck them
        scatter = set(
            self.controller.experiment.settings.get(
                'unmixed', {}
            ).get('scatter_channels') or []
        )

        # Channels to always exclude from DR (non-data columns)
        _always_exclude = {'event_id', 'Time', 'ribbon'}

        if channels and not self.channel_checkboxes:
            labels = _antigen_dash_labels(self.controller)
            grid_idx = 0
            for ch in channels:
                if ch in _always_exclude:
                    continue
                cb = QCheckBox(labels.get(ch, ch))
                # Pre-check fluorescence channels; uncheck scatter
                is_scatter = any(s in ch for s in scatter)
                cb.setChecked(not is_scatter)
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
            raw_subdir = self.controller.experiment.settings['raw'][
                'raw_samples_subdirectory'
            ]
            all_samples = self.controller.experiment.samples.get('all_samples', {})
            # Single stain controls should not be used as training samples
            single_stain_controls = set(
                self.controller.experiment.samples.get('single_stain_controls', [])
            )
            rel_paths = []
            for s in all_samples:
                if s in single_stain_controls:
                    continue
                try:
                    rel_paths.append(str(Path(s).relative_to(raw_subdir)))
                except ValueError:
                    rel_paths.append(s)
            self.picker.set_items(rel_paths)

        # Refresh DR and clustering status labels
        self._refresh_dr_status()
        self._refresh_cl_status()
        self._refresh_cl_space_combo()

        # Run management table (Item 6) — cheap metadata-only rebuild, safe
        # to call every time this tab becomes active.
        self.run_table.refresh()


class BiplotTile(QWidget):
    """
    A single square biplot tile — heatmap of x_channel vs y_channel.
    x_channel is always the TransformTab's active channel (set externally).
    y_channel is chosen by clicking the y-axis InteractiveLabel (same
    mechanism as CytometryPlotWidget) or via the combo at the top.
    Drag the axes to adjust W / zoom, exactly as in CytometryPlotWidget.
    """

    # InteractiveLabel.mousePressEvent calls parent_plot.select_plot_on_parent_grid()
    # BiplotTile is the parent_plot — provide a no-op stub so the call doesn't crash.
    def select_plot_on_parent_grid(self):
        pass

    def __init__(self, rgba_lut, parent_tab, y_channel: str = '', parent=None):
        super().__init__(parent)
        self._parent_tab = parent_tab
        self._y_channel  = y_channel
        self._rgba_lut   = rgba_lut

        size = hc_settings.cytometry_plot_width_target_retrieved
        self.setMinimumSize(size, size)
        self.setMaximumSize(size, size)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        # y-channel selector combo (hidden; driven by InteractiveLabel click)
        self._y_combo = QComboBox()
        self._y_combo.setMaximumHeight(22)
        self._y_combo.currentTextChanged.connect(self._on_y_changed)
        outer.addWidget(self._y_combo)

        # pyqtgraph graphics layout
        self._gw = TransparentGraphicsLayoutWidget()
        self._gw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gl = self._gw.ci.layout
        gl.setHorizontalSpacing(0)
        gl.setVerticalSpacing(0)

        # Y-axis InteractiveLabel — click to change y-channel, same as Honeychrome
        self._label_y = InteractiveLabel('', parent_plot=self, angle=-90, size='9pt')
        self._label_y.leftClickMenuFunction = self._set_y_from_label
        self._gw.addItem(self._label_y, row=0, col=0)

        self._vb = NoPanViewBox()
        self._gw.addItem(self._vb, row=0, col=2)

        self._axis_x = ZoomAxis('bottom', self._vb)
        self._axis_y = ZoomAxis('left',   self._vb)
        self._gw.addItem(self._axis_x, row=1, col=2)
        self._gw.addItem(self._axis_y, row=0, col=1)
        self._axis_x.linkToView(self._vb)
        self._axis_y.linkToView(self._vb)

        # X-axis label (non-interactive — pinned to active channel)
        self._label_x = pg.LabelItem('', size='9pt')
        self._gw.addItem(self._label_x, row=2, col=2)

        self._vb.setMouseEnabled(x=False, y=False)
        self._vb.raiseContextMenu = lambda ev: None

        self._img = pg.ImageItem()
        self._img.setLookupTable(rgba_lut)
        self._vb.addItem(self._img)

        self._axis_x.zoom_timer.timeout.connect(
            lambda: self._parent_tab._apply_tile_zoom(self, 'x')
        )
        self._axis_y.zoom_timer.timeout.connect(
            lambda: self._parent_tab._apply_tile_zoom(self, 'y')
        )

        outer.addWidget(self._gw)

    # ------------------------------------------------------------------

    def _set_y_from_label(self, item_index, _parent):
        """Called by InteractiveLabel when user picks a channel from the menu."""
        channels = [self._y_combo.itemText(i) for i in range(self._y_combo.count())]
        if 0 <= item_index < len(channels):
            self._label_y.leftItemSelected = item_index
            self._y_combo.setCurrentText(channels[item_index])

    def set_channels(self, available: list[str], y_channel: str = ''):
        self._y_combo.blockSignals(True)
        current = y_channel or self._y_combo.currentText()
        self._y_combo.clear()
        self._y_combo.addItems(available)
        if current in available:
            self._y_combo.setCurrentText(current)
        self._y_combo.blockSignals(False)
        self._y_channel = self._y_combo.currentText()

        # Sync InteractiveLabel menu items
        self._label_y.leftClickMenuItems = available
        ch = self._y_combo.currentText()
        self._label_y.leftItemSelected = available.index(ch) if ch in available else 0

    def y_channel(self) -> str:
        return self._y_combo.currentText()

    def _on_y_changed(self, ch: str):
        self._y_channel = ch
        self._label_y.setText(ch)
        # Keep leftItemSelected in sync
        channels = [self._y_combo.itemText(i) for i in range(self._y_combo.count())]
        if ch in channels:
            self._label_y.leftItemSelected = channels.index(ch)
        self._parent_tab._configure_tile_axes(self)
        self._parent_tab._draw_tile(self)

    def configure_axes(self, tr_x: 'Transform', tr_y: 'Transform',
                       x_label: str = '', y_label: str = ''):
        """Apply transform ticks/limits and axis labels.
        Calls setTicks() BEFORE setXRange/setYRange so real-value labels appear immediately."""
        if x_label:
            self._label_x.setText(x_label)
        if y_label:
            self._label_y.setText(y_label)
            self._y_channel = y_label

        if tr_x.ticks:
            self._axis_x.setTicks(tr_x.ticks())
        self._axis_x.zoomZero = tr_x.zero
        self._axis_x.fullRange = (0, 1.1)
        self._axis_x.limits = tuple(tr_x.limits)
        self._vb.setXRange(tr_x.limits[0], tr_x.limits[1], padding=0)

        if tr_y.ticks:
            self._axis_y.setTicks(tr_y.ticks())
        self._axis_y.zoomZero = tr_y.zero
        self._axis_y.fullRange = (0, 1.1)
        self._axis_y.limits = tuple(tr_y.limits)
        self._vb.setYRange(tr_y.limits[0], tr_y.limits[1], padding=0)

    def draw(self, heatmap, tr_x: 'Transform', tr_y: 'Transform'):
        self._img.setImage(heatmap)
        self._img.setRect(QRectF(
            tr_x.limits[0], tr_y.limits[0],
            tr_x.limits[1] - tr_x.limits[0],
            tr_y.limits[1] - tr_y.limits[0],
        ))

    def clear(self):
        self._img.clear()


class _SquareWidget(QWidget):
    """A widget that always maintains a square geometry."""
    def resizeEvent(self, event):
        if self.height() != self.width():
            self.setFixedHeight(self.width())
        super().resizeEvent(event)


class TransformTab(QWidget):
    """
    Tab 1 — Transform Inspector (Read-Only Preview)
    ------------------------------------------------
    Shows the Logicle / biexponential transform parameters from the experiment.
    Lets the user preview local W-adjustments (and auto-transform suggestions)
    without writing back to the experiment.

    Visual style matches CytometryPlotWidget exactly:
      • 1-D histogram: pg.PlotDataItem stepMode='center', colorcet fill, ZoomAxis
        with the Honeychrome axis-drag W-adjustment (lower half = W, upper = zoom)
      • Biplot: pg.ImageItem heatmap with the same colourmap and ZoomAxis on both
        axes; dragging either axis adjusts W for that channel the same way
        CytometryPlotWidget does.
      • Axes tick labels show real (raw) values — 10², 10³ etc. — via
        Transform.ticks(), not logicle floating-point numbers.
      • Biplot is square (resizeEvent enforces width == height).
    """

    _XFORM_LABELS = {0: 'Linear', 1: 'Logicle', 2: 'Log', 'default': 'Default'}

    def __init__(self, state: PipelineState, bus, controller, parent=None):
        super().__init__(parent)
        self.state      = state
        self.bus        = bus
        self.controller = controller

        self._active_channel: str | None = None
        # {channel: Transform}  — local copies, never written back
        self._transforms: dict[str, Transform] = {}
        # {channel: np.ndarray}  — raw unmixed event data for current sample
        self._raw_data: dict[str, np.ndarray] = {}
        # channel-name list in pnn order
        self._pnn: list[str] = []

        # Debounce timer — redraws plots 200 ms after last change
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(200)
        self._redraw_timer.timeout.connect(self._redraw_plots)

        # Build LUT from Honeychrome colourmap settings (same as CytometryPlotWidget)
        try:
            colors = cc.palette[hc_settings.colourmap_name_retrieved]
        except Exception:
            colors = cc.palette['bjy']
        cmap = pg.ColorMap(
            pos=0.9 * np.linspace(0, 1, len(colors)) ** 2
                + 0.1 * np.linspace(0, 1, len(colors)),
            color=colors,
        )
        rgba_lut = cmap.getLookupTable(alpha=True)
        rgba_lut[0, 3] = 0   # fully transparent for zero bins
        self._rgba_lut = rgba_lut

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # --- Read-only notice ---
        notice = QLabel(
            "ℹ  Parameters are read from the current experiment.  "
            "Edits here are local previews only — they do not affect the "
            "main Honeychrome cytometry plots.  To save changes, use the "
            "Transforms panel in the main Honeychrome interface."
        )
        notice.setWordWrap(True)
        notice.setStyleSheet(
            "QLabel { background:#fff3cd; color:#5a4a00; border:1px solid #ffc107; "
            "border-radius:4px; padding:6px; }"
        )
        outer.addWidget(notice)

        # --- Gate selection — primary tree (§0.3, Item 4) ---
        gate_box = QGroupBox("Gate(s)")
        gate_box_layout = QVBoxLayout(gate_box)
        self.gate_tree = drc_gate_tree.GateTreeWidget()
        self.gate_tree.setToolTip(
            "Check one or more gates.  Their events are UNIONED for the "
            "preview below and for DR/clustering training pools.  Checking "
            "a parent gate checks its whole subtree."
        )
        self.gate_tree.setMinimumHeight(140)
        self.gate_tree.setMaximumHeight(220)
        self.gate_tree.selectionChanged.connect(self._on_gate_tree_changed)
        gate_box_layout.addWidget(self.gate_tree)
        outer.addWidget(gate_box)

        # --- Top toolbar ---
        toolbar = QHBoxLayout()

        self.reload_btn = QPushButton("↺ Reload from experiment")
        self.reload_btn.setToolTip(
            "Re-read transform parameters from the experiment model, "
            "discarding any local preview edits."
        )
        self.reload_btn.clicked.connect(self.refresh)
        toolbar.addWidget(self.reload_btn)

        self.auto_btn = QPushButton("✦ Auto-transform preview (all channels)")
        self.auto_btn.setToolTip(
            "Compute suggested Logicle W/A for every fluorescence channel "
            "(Parks, Roederer & Moore 2006).  Preview only — not saved."
        )
        self.auto_btn.clicked.connect(self._auto_transform_all)
        toolbar.addWidget(self.auto_btn)

        self.save_csv_btn = QPushButton("⬇ Save CSV")
        self.save_csv_btn.setToolTip(
            "Export current (local preview) transform parameters to a CSV file.\n"
            "Columns: channel, id, T, W, M, A"
        )
        self.save_csv_btn.clicked.connect(self._save_transforms_csv)
        toolbar.addWidget(self.save_csv_btn)

        self.load_csv_btn = QPushButton("⬆ Load CSV")
        self.load_csv_btn.setToolTip(
            "Import transform parameters from a CSV file.\n"
            "Channel names are validated against the current experiment.\n"
            "Unrecognised channels are skipped; missing channels keep their\n"
            "current parameters."
        )
        self.load_csv_btn.clicked.connect(self._load_transforms_csv)
        toolbar.addWidget(self.load_csv_btn)

        toolbar.addStretch()
        outer.addLayout(toolbar)

        # --- Splitter: channel list | plots ---
        splitter = QSplitter(Qt.Horizontal)

        # Left: channel list + transform type label
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Channels"))
        self.channel_list = QListWidget()
        self.channel_list.setMaximumWidth(180)
        self.channel_list.currentItemChanged.connect(self._on_channel_item_changed)
        left_layout.addWidget(self.channel_list)
        self.xform_type_label = QLabel("—")
        self.xform_type_label.setStyleSheet("font-style: italic; color: grey; padding: 4px;")
        left_layout.addWidget(self.xform_type_label)
        splitter.addWidget(left)

        # Right: scrollable area holding histogram + biplot
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout(self.right_widget)
        self.right_layout.setContentsMargins(6, 0, 0, 0)
        self.right_layout.setSpacing(8)
        right_scroll.setWidget(self.right_widget)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter, stretch=1)

        # ------------------------------------------------------------------
        # 1-D Histogram — built once, updated in-place each redraw
        # ------------------------------------------------------------------
        self.right_layout.addWidget(QLabel("1-D Histogram  (drag axis to adjust transform)"))

        self._hist_gw = TransparentGraphicsLayoutWidget()
        self._hist_gw.setMinimumHeight(180)
        self._hist_gw.setMaximumHeight(280)
        self._hist_gw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        gl = self._hist_gw.ci.layout
        gl.setHorizontalSpacing(0)
        gl.setVerticalSpacing(0)

        self._hist_vb = NoPanViewBox()
        self._hist_vb.enableAutoRange(axis=self._hist_vb.YAxis, enable=True)
        self._hist_gw.addItem(self._hist_vb, row=0, col=1)

        self._hist_axis_x = ZoomAxis('bottom', self._hist_vb)
        self._hist_axis_y = ZoomAxis('left',   self._hist_vb)
        self._hist_gw.addItem(self._hist_axis_x, row=1, col=1)
        self._hist_gw.addItem(self._hist_axis_y, row=0, col=0)
        self._hist_axis_x.linkToView(self._hist_vb)
        self._hist_axis_y.linkToView(self._hist_vb)

        # Disable panning in the viewbox; zoom done via axis drag
        self._hist_vb.setMouseEnabled(x=False, y=False)
        self._hist_vb.raiseContextMenu = lambda ev: None

        # Filled step-curve — same as CytometryPlotWidget.hist
        self._hist_curve = pg.PlotDataItem(
            stepMode='center',
            fillLevel=0,
            brush=(100, 100, 250, 150),
        )
        self._hist_vb.addItem(self._hist_curve)

        self._hist_axis_x.zoom_timer.timeout.connect(self._apply_hist_zoom)
        self._hist_axis_y.zoom_timer.timeout.connect(lambda: None)  # y auto-ranges

        self.right_layout.addWidget(self._hist_gw)

        # ------------------------------------------------------------------
        # Biplot grid — multiple square tiles, Add/Remove buttons
        # x-axis of every tile = active channel; y-axis chosen per tile
        # ------------------------------------------------------------------
        biplot_header = QHBoxLayout()
        biplot_header.addWidget(QLabel("Biplots  (drag axes to adjust transforms)"))
        biplot_header.addStretch()
        add_btn = QPushButton("＋ Add biplot")
        add_btn.setFixedHeight(24)
        add_btn.clicked.connect(self._add_biplot)
        biplot_header.addWidget(add_btn)
        remove_btn = QPushButton("－ Remove last")
        remove_btn.setFixedHeight(24)
        remove_btn.clicked.connect(self._remove_biplot)
        biplot_header.addWidget(remove_btn)
        self.right_layout.addLayout(biplot_header)

        # Scrollable area for biplot tiles — tiles wrap based on available width
        self._biplot_scroll = QScrollArea()
        self._biplot_scroll.setWidgetResizable(True)
        self._biplot_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._biplot_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._biplot_scroll.setMinimumHeight(
            hc_settings.cytometry_plot_width_target_retrieved + 40
        )
        self._biplot_scroll.setMaximumHeight(
            2 * hc_settings.cytometry_plot_width_target_retrieved + 50
        )

        self._biplot_container = QWidget()
        self._biplot_grid = QGridLayout(self._biplot_container)
        self._biplot_grid.setContentsMargins(0, 0, 0, 0)
        self._biplot_grid.setSpacing(4)
        self._biplot_scroll.setWidget(self._biplot_container)
        self.right_layout.addWidget(self._biplot_scroll)

        # Debounce timer for tile reflow on resize (mirrors CytometryGridWidget)
        self._tile_relayout_timer = QTimer(self)
        self._tile_relayout_timer.setSingleShot(True)
        self._tile_relayout_timer.setInterval(150)
        self._tile_relayout_timer.timeout.connect(self._relayout_tiles)
        self._biplot_scroll.viewport().installEventFilter(self)

        self._biplot_tiles: list[BiplotTile] = []
        self.right_layout.addStretch()

    # ------------------------------------------------------------------
    # Refresh — called on tab activation
    # ------------------------------------------------------------------

    def refresh(self):
        """
        Reload Transform objects from controller.unmixed_transformations,
        repopulate the channel list and biplot x-combo, then load event data.
        """
        _t_refresh0 = time.perf_counter()
        if self.controller.experiment.process.get('unmixing_matrix') is None:
            return

        # Gate tree — always rebuilt so renamed/added gates appear; checked
        # set comes from the shared state.selected_gates (kept in sync with
        # ConfigTab's tree by PluginWidget._on_gate_tree_changed).
        _t0 = time.perf_counter()
        hierarchy = self.controller.unmixed_gating.get_gate_hierarchy(output='dict')
        _log.info("TIMING TransformTab.refresh: get_gate_hierarchy took %.3fs",
                   time.perf_counter() - _t0)
        _t0 = time.perf_counter()
        self.gate_tree.set_hierarchy(hierarchy)
        self.gate_tree.set_checked_names(self.state.selected_gates)
        _log.info("TIMING TransformTab.refresh: gate_tree rebuild+sync took %.3fs",
                   time.perf_counter() - _t0)

        # Deep-copy Transform objects from controller so we can tweak locally
        src = self.controller.unmixed_transformations or {}
        self._transforms = {ch: deepcopy(tr) for ch, tr in src.items()}

        # Also sync state.channel_transform_params for downstream use
        self.state.channel_transform_params = _read_transforms_from_experiment(
            self.controller
        )

        # pnn list
        self._pnn = self.controller.experiment.settings.get(
            'unmixed', {}
        ).get('event_channels_pnn', [])

        if not self._transforms:
            return

        # Restore any previously computed auto-transform W values
        self._load_computed_transforms()
        skip = {'ribbon', 'Time', 'event_id'}
        all_channels = [ch for ch in self._transforms if ch not in skip]
        # Transforms tab always shows every channel — channel selection
        # for DR/clustering happens later, in Configuration.
        channels = all_channels

        # Repopulate channel list
        self.channel_list.blockSignals(True)
        prev = self.channel_list.currentItem()
        prev_channel = prev.data(Qt.UserRole) if prev else None
        self.channel_list.clear()
        labels = _antigen_dash_labels(self.controller)
        for ch in channels:
            item = QListWidgetItem(labels.get(ch, ch))
            item.setData(Qt.UserRole, ch)
            self.channel_list.addItem(item)
        self.channel_list.blockSignals(False)

        # Update all biplot tiles with the new channel list
        # Add a default first tile if none exist yet
        if not self._biplot_tiles:
            self._add_biplot()
        for tile in self._biplot_tiles:
            tile.set_channels(channels)

        # Load data (deferred so load_sample has finished)
        self._load_preview_data()

        # Restore / default channel selection
        restored = False
        if prev_channel and prev_channel in channels:
            for i in range(self.channel_list.count()):
                if self.channel_list.item(i).data(Qt.UserRole) == prev_channel:
                    self.channel_list.setCurrentRow(i)
                    restored = True
                    break
        if not restored and channels:
            self.channel_list.setCurrentRow(0)

        _log.info("TIMING TransformTab.refresh: total (excluding deferred preview load) took %.3fs",
                   time.perf_counter() - _t_refresh0)
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
        _t_total0 = time.perf_counter()
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
                _t0 = time.perf_counter()
                _n_in = len(event_data)
                try:
                    cytometry_dict = dict(self.controller.data_for_cytometry_plots_unmixed)
                    cytometry_dict['event_data'] = event_data.copy()
                    event_data = apply_gates_union_by_lookup_table(cytometry_dict, selected_gates)
                except Exception as e:
                    print(f"[DR Plugin] TransformTab: gating failed ({selected_gates}): {e}")
                _log.info("TIMING TransformTab._do_load_preview_data: "
                          "apply_gates_union_by_lookup_table (%d events in -> %d out, gates=%r) took %.3fs",
                          _n_in, len(event_data), selected_gates, time.perf_counter() - _t0)

            for i, ch in enumerate(channel_names):
                if i < event_data.shape[1]:
                    self._raw_data[ch] = event_data[:, i]

            print(f"[DR Plugin] TransformTab: {len(event_data):,} events "
                  f"(gates={selected_gates!r})")
            _log.info("TIMING TransformTab._do_load_preview_data: total took %.3fs",
                       time.perf_counter() - _t_total0)

            # Immediately draw tiles — don't wait for the debounce timer
            QTimer.singleShot(0, self._redraw_plots)

        except Exception as e:
            print(f"[DR Plugin] TransformTab: could not load preview data: {e}")

    def _on_gate_tree_changed(self, gates: list[str]):
        """Checked-set changed in this tab's gate tree — reload the preview."""
        self.state.selected_gates = list(gates)
        self._load_preview_data()

    # ------------------------------------------------------------------
    # Channel selection
    # ------------------------------------------------------------------

    def _on_channel_item_changed(self, current, previous):
        """Bridge from the QListWidgetItem (display label) to the raw
        channel name stored in its UserRole data."""
        channel = current.data(Qt.UserRole) if current else None
        self._on_channel_selected(channel)

    def _on_channel_selected(self, channel: str):
        if not channel or channel not in self._transforms:
            return
        self._active_channel = channel
        tr = self._transforms[channel]
        self.xform_type_label.setText(self._XFORM_LABELS.get(tr.id, str(tr.id)))

        # Debug: print current transform parameters for this channel
        print(f"\n[DR TransformTab] Active channel: {channel}")
        print(f"  id={tr.id}  ({self._XFORM_LABELS.get(tr.id, '?')})")
        print(f"  T (scale_t)   = {tr.scale_t}")
        print(f"  W (logicle_w) = {tr.logicle_w}")
        print(f"  M (logicle_m) = {tr.logicle_m}  [stored in Transform]")
        print(f"  M (from PnR)  = {self._compute_M():.4f}  [log10(magnitude_ceiling)]")
        print(f"  A (logicle_a) = {tr.logicle_a}")
        print(f"  limits        = {tr.limits}")
        print(f"  zero          = {tr.zero}")

        self._configure_hist_axes()
        self._configure_tile_axes_all()
        self._schedule_redraw()

    # ------------------------------------------------------------------
    # M parameter — cytometer-specific, derived from $PnR
    # ------------------------------------------------------------------

    def _compute_M(self) -> float:
        """
        Compute M (logicle decades) from the cytometer's ADC range.
        M = log10(magnitude_ceiling) − 1.

        Per the Gating-ML spec the top-of-scale decade is excluded from M
        (e.g. a 1 × 10⁶ instrument → M = 5, not 6).  magnitude_ceiling is
        the max $PnR value across fluorescence channels stored at import time.
        """
        try:
            ceiling = self.controller.experiment.settings['raw']['magnitude_ceiling']
            if ceiling and ceiling > 0:
                return math.log10(float(ceiling)) - 1.0
        except (KeyError, TypeError, ValueError):
            pass
        return 4.5   # fall back to Honeychrome default

    # ------------------------------------------------------------------
    # Biplot tile management
    # ------------------------------------------------------------------

    def _available_channels(self) -> list[str]:
        skip = {'ribbon', 'Time', 'event_id'}
        all_ch = [ch for ch in self._transforms if ch not in skip]
        sel = self.state.selected_channels
        return [ch for ch in all_ch if ch in sel] if sel else all_ch

    def eventFilter(self, obj, event):
        """Debounce tile reflow when the biplot scroll viewport is resized."""
        if obj is self._biplot_scroll.viewport() and event.type() == QEvent.Resize:
            if event.oldSize().width() != event.size().width():
                self._tile_relayout_timer.start()
        return super().eventFilter(obj, event)

    def _relayout_tiles(self):
        """Place biplot tiles into the grid, wrapping based on available width."""
        # Remove all from grid without deleting
        for i in range(self._biplot_grid.count()):
            item = self._biplot_grid.itemAt(i)
            if item and item.widget():
                self._biplot_grid.removeWidget(item.widget())

        tile_size = hc_settings.cytometry_plot_width_target_retrieved
        available_w = self._biplot_scroll.viewport().width() or (tile_size + 4)
        cols = max(1, available_w // (tile_size + 4))

        for n, tile in enumerate(self._biplot_tiles):
            row, col = divmod(n, cols)
            self._biplot_grid.addWidget(tile, row, col)

        # Update scroll area height to fit all rows
        n_rows = max(1, (len(self._biplot_tiles) + cols - 1) // cols) if self._biplot_tiles else 1
        self._biplot_scroll.setMinimumHeight(
            min(n_rows, 2) * (tile_size + 30) + 10
        )

    def _add_biplot(self):
        """Add a new BiplotTile to the grid."""
        channels = self._available_channels()
        used_y = {t.y_channel() for t in self._biplot_tiles}
        x_ch = self._active_channel or (channels[0] if channels else '')
        candidates = [ch for ch in channels if ch != x_ch and ch not in used_y]
        default_y = candidates[0] if candidates else (
            channels[1] if len(channels) > 1 else (channels[0] if channels else '')
        )

        tile = BiplotTile(self._rgba_lut, parent_tab=self)
        tile.set_channels(channels, y_channel=default_y)
        self._biplot_tiles.append(tile)
        self._relayout_tiles()
        self._configure_tile_axes(tile)
        self._draw_tile(tile)

    def _remove_biplot(self):
        """Remove the last BiplotTile."""
        if not self._biplot_tiles:
            return
        tile = self._biplot_tiles.pop()
        self._biplot_grid.removeWidget(tile)
        tile.deleteLater()
        self._relayout_tiles()

    def _configure_tile_axes(self, tile: 'BiplotTile'):
        """Configure axes and labels on a single tile."""
        x_ch = self._active_channel
        y_ch = tile.y_channel()
        if not x_ch or not y_ch:
            return
        if x_ch not in self._transforms or y_ch not in self._transforms:
            return
        labels = _antigen_dash_labels(self.controller)
        tile.configure_axes(self._transforms[x_ch], self._transforms[y_ch],
                            x_label=labels.get(x_ch, x_ch), y_label=y_ch)

    def _configure_tile_axes_all(self):
        for tile in self._biplot_tiles:
            self._configure_tile_axes(tile)

    def _configure_hist_axes(self):
        """Set ZoomAxis ticks/limits for the 1-D histogram x-axis.
        Must call setTicks() before setXRange() so real-value labels
        are shown immediately, not pyqtgraph auto-generated floats."""
        ch = self._active_channel
        if not ch or ch not in self._transforms:
            return
        tr = self._transforms[ch]
        if tr.ticks is None:
            return
        ticks = tr.ticks()
        if ticks:
            self._hist_axis_x.setTicks(ticks)
        self._hist_axis_x.zoomZero = tr.zero
        self._hist_axis_x.fullRange = (0, 1.1)
        self._hist_axis_x.limits = tuple(tr.limits)
        self._hist_vb.setXRange(tr.limits[0], tr.limits[1], padding=0)

    def _draw_tile(self, tile: 'BiplotTile'):
        """Compute and render heatmap for a single tile."""
        x_ch = self._active_channel
        y_ch = tile.y_channel()
        if (not x_ch or not y_ch or x_ch == y_ch
                or x_ch not in self._transforms or y_ch not in self._transforms
                or x_ch not in self._raw_data or y_ch not in self._raw_data):
            tile.clear()
            return
        tr_x = self._transforms[x_ch]
        tr_y = self._transforms[y_ch]
        raw_x = self._raw_data[x_ch]
        raw_y = self._raw_data[y_ch]
        try:
            heatmap, _, _ = np.histogram2d(raw_x, raw_y,
                                            bins=[tr_x.scale, tr_y.scale])
        except Exception as e:
            print(f"[DR Plugin] biplot histogram2d error ({x_ch} vs {y_ch}): {e}")
            tile.clear()
            return
        try:
            cutoff = hc_settings.density_cutoff_retrieved
        except Exception:
            cutoff = 1
        heatmap[heatmap < cutoff] = 0
        tile.draw(heatmap, tr_x, tr_y)

    def _apply_tile_zoom(self, tile: 'BiplotTile', axis_name: str):
        """Axis-drag zoom for a tile — mirrors CytometryPlotWidget.apply_zoom."""
        if axis_name == 'x':
            axis   = tile._axis_x
            channel = self._active_channel
            vb_set  = tile._vb.setXRange
            vb_range_idx = 0
            factor_flip  = False
        else:
            axis   = tile._axis_y
            channel = tile.y_channel()
            vb_set  = tile._vb.setYRange
            vb_range_idx = 1
            factor_flip  = True

        if not channel or channel not in self._transforms:
            return
        if axis._pending_delta == 0:
            return
        step = axis._pending_delta
        axis._pending_delta = 0
        if abs(step) < 1:
            return

        zoom_rate = 1.04
        factor = (1 / zoom_rate) if step > 0 else zoom_rate
        if factor_flip:
            factor = 1 / factor

        tr = self._transforms[channel]
        vmin, vmax = tile._vb.viewRange()[vb_range_idx]
        map_pos = (tile._vb.mapToView(axis.initial_pos).x()
                   if axis_name == 'x' and axis.initial_pos else
                   tile._vb.mapToView(axis.initial_pos).y()
                   if axis.initial_pos else 0.5)

        if tr.id == 1:   # logicle: lower half → W, upper half → zoom
            if map_pos < 0.5 * vmax:
                tr.logicle_w = max(0.01, tr.logicle_w / factor)
                tr.set_transform()
                print(f"[DR TransformTab] {channel}: W adjusted → {tr.logicle_w:.4f}")
            else:
                new_max = (vmax - axis.zoomZero) * factor + axis.zoomZero
                new_min = (vmin - axis.zoomZero) * factor + axis.zoomZero
                if new_max < axis.fullRange[1] * 1.01:
                    vb_set(new_min, new_max, padding=0)
                axis.limits = (new_min, new_max)
                tr.set_transform(limits=list(axis.limits))
        else:
            new_max = (vmax - axis.zoomZero) * factor + axis.zoomZero
            new_min = (vmin - axis.zoomZero) * factor + axis.zoomZero
            if new_max < axis.fullRange[1] * 1.01:
                vb_set(new_min, new_max, padding=0)
            axis.limits = (new_min, new_max)
            tr.set_transform(limits=list(axis.limits))

        axis.zoomZero = tr.zero
        if tr.ticks:
            axis.setTicks(tr.ticks())
        # Keep all tiles that share this channel consistent
        for other in self._biplot_tiles:
            if other is not tile:
                if axis_name == 'x' and self._active_channel == channel:
                    if other._transforms if hasattr(other, '_transforms') else True:
                        self._configure_tile_axes(other)
        # Also sync hist x-axis if the active channel was changed
        if channel == self._active_channel:
            self._configure_hist_axes()
        self._sync_transform_to_state(channel, tr)
        self._schedule_redraw()

    def _apply_hist_zoom(self):
        """Called by hist x-axis ZoomAxis timer — adjusts W (lower half) or limits (upper)."""
        ch = self._active_channel
        if not ch or ch not in self._transforms:
            return
        axis = self._hist_axis_x
        if axis._pending_delta == 0:
            return
        step = axis._pending_delta
        axis._pending_delta = 0
        if abs(step) < 1:
            return

        zoom_rate = 1.04
        factor = (1 / zoom_rate) if step > 0 else zoom_rate

        tr = self._transforms[ch]
        vmin, vmax = self._hist_vb.viewRange()[0]
        map_pos = self._hist_vb.mapToView(axis.initial_pos).x() if axis.initial_pos else 0.5

        if tr.id == 1:
            if map_pos < 0.5 * vmax:
                tr.logicle_w = max(0.01, tr.logicle_w / factor)
                tr.set_transform()
                print(f"[DR TransformTab] {ch}: W adjusted via hist axis → {tr.logicle_w:.4f}")
            else:
                new_max = (vmax - axis.zoomZero) * factor + axis.zoomZero
                new_min = (vmin - axis.zoomZero) * factor + axis.zoomZero
                if new_max < axis.fullRange[1] * 1.01:
                    self._hist_vb.setXRange(new_min, new_max, padding=0)
                axis.limits = (new_min, new_max)
                tr.set_transform(limits=list(axis.limits))
        else:
            new_max = (vmax - axis.zoomZero) * factor + axis.zoomZero
            new_min = (vmin - axis.zoomZero) * factor + axis.zoomZero
            if new_max < axis.fullRange[1] * 1.01:
                self._hist_vb.setXRange(new_min, new_max, padding=0)
            axis.limits = (new_min, new_max)
            tr.set_transform(limits=list(axis.limits))

        axis.zoomZero = tr.zero
        if tr.ticks:
            axis.setTicks(tr.ticks())
        self._sync_transform_to_state(ch, tr)
        # Keep all biplot tiles that use this channel on x-axis in sync
        self._configure_tile_axes_all()
        self._schedule_redraw()

    def _sync_transform_to_state(self, channel: str, tr: 'Transform'):
        """Push updated W back into state.channel_transform_params."""
        if channel in self.state.channel_transform_params:
            self.state.channel_transform_params[channel]['W'] = tr.logicle_w

    def _schedule_redraw(self):
        self._redraw_timer.start()

    def _redraw_plots(self):
        self._draw_histogram()
        for tile in self._biplot_tiles:
            self._draw_tile(tile)

    def _draw_histogram(self):
        """Update 1-D histogram in-place using Transform.step_scale."""
        ch = self._active_channel
        if not ch or ch not in self._transforms or ch not in self._raw_data:
            self._hist_curve.setData([], [])
            return

        tr = self._transforms[ch]
        raw = self._raw_data[ch]

        # Bin raw events into the transform's scale bins (same as CytometryPlotWidget)
        try:
            count, _ = np.histogram(raw, bins=tr.scale)
        except Exception:
            self._hist_curve.setData([], [])
            return

        # step_scale has len = scale_bins + 2; count has len = scale_bins + 1
        # PlotDataItem with stepMode='center' needs x and y the same length
        self._hist_curve.setData(tr.step_scale, count)
        self._configure_hist_axes()

    # ------------------------------------------------------------------
    # CSV import / export
    # ------------------------------------------------------------------

    def _save_transforms_csv(self):
        """
        Export current local transform parameters to a CSV file.
        Columns: channel, id, T, W, M, A
        id codes: 0=linear, 1=logicle, 2=log
        """
        if not self._transforms:
            QMessageBox.information(self, "Save Transforms",
                                    "No transforms loaded yet.")
            return

        default_dir = str(getattr(self.controller, 'experiment_dir', ''))
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Transforms CSV", default_dir,
            "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return

        skip = {'ribbon', 'Time', 'event_id'}
        rows = []
        for ch, tr in self._transforms.items():
            if ch in skip:
                continue
            rows.append({
                'channel': ch,
                'id':      tr.id,
                'T':       tr.scale_t,
                'W':       round(tr.logicle_w, 6),
                'M':       round(tr.logicle_m, 6),
                'A':       round(tr.logicle_a, 6),
            })

        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['channel', 'id', 'T', 'W', 'M', 'A'])
                writer.writeheader()
                writer.writerows(rows)
            print(f"[DR TransformTab] Saved {len(rows)} channel transforms → {path}")
            QMessageBox.information(
                self, "Save Transforms",
                f"Saved {len(rows)} channels to:\n{path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _load_transforms_csv(self):
        """
        Import transform parameters from a CSV file.

        Validation:
          • Required columns: channel, W  (T, M, A, id are optional)
          • Only channels present in the current experiment are updated;
            unrecognised channel names are reported and skipped.
          • W must satisfy 0 ≤ W ≤ M/2; values outside this range are clamped.
          • T must be > 0 if provided.

        After loading, the local Transform objects are updated and plots redrawn.
        Nothing is written back to the experiment.
        """
        default_dir = str(getattr(self.controller, 'experiment_dir', ''))
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Transforms CSV", default_dir,
            "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return

        try:
            with open(path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Could not read CSV:\n{e}")
            return

        if not rows:
            QMessageBox.warning(self, "Load Transforms", "CSV file is empty.")
            return

        required_cols = {'channel', 'W'}
        if not required_cols.issubset(set(rows[0].keys())):
            QMessageBox.critical(
                self, "Load Transforms",
                f"CSV must contain at least columns: {sorted(required_cols)}\n"
                f"Found: {sorted(rows[0].keys())}"
            )
            return

        known_channels = set(self._transforms.keys())
        updated, skipped_unknown, skipped_invalid = [], [], []

        for row in rows:
            ch = (row.get('channel') or '').strip()
            if not ch:
                continue
            if ch not in known_channels:
                skipped_unknown.append(ch)
                continue

            tr = self._transforms[ch]
            try:
                W = float(row['W'])
            except (ValueError, KeyError):
                skipped_invalid.append(f"{ch} (W invalid)")
                continue

            # Apply optional columns
            if 'T' in row and row['T'].strip():
                try:
                    T = float(row['T'])
                    if T > 0:
                        tr.scale_t = T
                except ValueError:
                    pass

            if 'M' in row and row['M'].strip():
                try:
                    M = float(row['M'])
                    if M > 0:
                        tr.logicle_m = M
                except ValueError:
                    pass

            if 'A' in row and row['A'].strip():
                try:
                    tr.logicle_a = float(row['A'])
                except ValueError:
                    pass

            if 'id' in row and row['id'].strip():
                try:
                    tr.id = int(row['id'])
                except ValueError:
                    pass

            # Clamp W to valid range: 0 ≤ W ≤ M/2
            M_eff = tr.logicle_m
            W = max(0.0, min(M_eff / 2.0, W))
            tr.logicle_w = round(W, 6)

            try:
                tr.set_transform()
            except Exception as e:
                skipped_invalid.append(f"{ch} (set_transform failed: {e})")
                continue

            self._sync_transform_to_state(ch, tr)
            updated.append(ch)

        # Report
        msg_parts = [f"Updated {len(updated)} channel(s)."]
        if skipped_unknown:
            msg_parts.append(
                f"\nSkipped {len(skipped_unknown)} unrecognised channel(s):\n"
                + ", ".join(skipped_unknown[:10])
                + (" …" if len(skipped_unknown) > 10 else "")
            )
        if skipped_invalid:
            msg_parts.append(
                f"\nSkipped {len(skipped_invalid)} invalid row(s):\n"
                + ", ".join(skipped_invalid[:10])
            )
        print(f"[DR TransformTab] CSV load: {'  '.join(msg_parts)}")
        QMessageBox.information(self, "Load Transforms", "".join(msg_parts))

        if updated:
            self._configure_hist_axes()
            self._configure_tile_axes_all()
            self._schedule_redraw()

    # ------------------------------------------------------------------
    # Auto-transform (Parks & Moore) — over all training samples
    # ------------------------------------------------------------------

    def _auto_transform_all(self):
        """
        Compute optimal Logicle W for every Logicle channel using Parks & Moore,
        pooling data across ALL selected training samples.

        Formula: W = (M − log₁₀(T / |q|)) / 2
        where q = 25th-percentile of the negative-event population across all
        training samples combined.  This gives a more robust estimate than using
        only the currently loaded sample, and ensures the transform is appropriate
        for the whole dataset.

        After computing W, the result is persisted to QSettings so it survives
        experiment close/re-open.
        """
        training_paths = self._get_training_sample_full_paths()
        if not training_paths:
            # Fall back to the currently loaded sample
            if self._raw_data:
                training_data = {ch: [arr] for ch, arr in self._raw_data.items()}
                source_note = "active sample only (no training samples selected)"
            else:
                QMessageBox.information(
                    self, "Auto-Transform",
                    "No training samples selected and no sample is currently loaded.\n"
                    "Select training samples in the Configuration tab first."
                )
                return
        else:
            print(f"\n[DR TransformTab] Auto-transform: loading {len(training_paths)} training sample(s)…")
            training_data = self._load_all_training_samples(training_paths)
            source_note = f"{len(training_paths)} training sample(s)"

        M = self._compute_M()
        print(f"[DR TransformTab] Auto-transform: M = {M:.4f},  source = {source_note}")
        n_updated = 0

        for ch, tr in self._transforms.items():
            if tr.id != 1:
                continue
            arrays = training_data.get(ch, [])
            if not arrays:
                continue

            # Pool all negative events across all training samples
            negatives = np.concatenate([a[a < 0] for a in arrays if len(a[a < 0]) > 0]) \
                if any(len(a[a < 0]) > 0 for a in arrays) else np.array([])

            T = tr.scale_t

            if len(negatives) == 0:
                W = 0.5
                print(f"  {ch}: no negatives in any training sample → W = {W}")
            else:
                q = np.percentile(negatives, 25)   # q < 0
                try:
                    W = (M - math.log10(T / abs(q))) / 2.0
                except (ValueError, ZeroDivisionError):
                    W = 0.5
                W = max(0.0, min(M / 2.0, W))
                print(f"  {ch}: n_neg={len(negatives):,}, q(25th)={q:.1f}, T={T} → W = {W:.4f}")

            tr.logicle_w = round(W, 3)
            tr.set_transform()
            self._sync_transform_to_state(ch, tr)
            n_updated += 1

        # Persist the computed W values
        self._save_computed_transforms()

        if self._active_channel and self._active_channel in self._transforms:
            self._configure_hist_axes()
            self._configure_tile_axes_all()
        self._schedule_redraw()

        QMessageBox.information(
            self, "Auto-Transform Preview",
            f"W computed for {n_updated} Logicle channel(s) from {source_note},\n"
            f"using the Parks & Moore method (lower-quarter quantile of negatives).\n\n"
            f"Parameters saved and will be restored on next open.\n"
            f"To make permanent, use the Transforms panel in the main interface."
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
        Load and unmix each training sample FCS file, returning a dict of
        {channel: [array_per_sample, …]}.

        Does NOT call controller.load_sample (which has side effects).
        Instead replicates its unmixing step using
        drc_pipeline.apply_unmixing_af_aware() — AF-corrected if an AF
        profile is active, otherwise plain apply_transfer_matrix() — the
        same side-effect-free helper load_unmixed_gated() uses, matching
        what controller.load_sample / controller._apply_unmixing do.
        """
        channel_names = self.controller.experiment.settings.get(
            'unmixed', {}
        ).get('event_channels_pnn', [])

        result: dict[str, list[np.ndarray]] = {ch: [] for ch in channel_names}

        if self.controller.transfer_matrix is None:
            print("[DR TransformTab] transfer_matrix not initialised on controller — "
                  "cannot unmix training samples.")
            return result

        selected_gates = self.state.selected_gates

        whitelisted_pnn = self.controller.experiment.settings['raw'].get('whitelisted_pnn')

        for path in full_paths:
            try:
                sample = sample_from_fcs(path)
                raw = sample.get_events(source='raw', col_order=whitelisted_pnn)

                # Replicate controller.load_sample unmixing (AF-corrected if
                # an AF profile is active — see drc_pipeline.apply_unmixing_af_aware).
                unmixed = drc_pipeline.apply_unmixing_af_aware(self.controller, raw)

                # Apply gate(s) if selected
                if selected_gates and selected_gates != ['root']:
                    try:
                        cytometry_dict = dict(
                            self.controller.data_for_cytometry_plots_unmixed
                        )
                        cytometry_dict['event_data'] = unmixed.copy()
                        unmixed = apply_gates_union_by_lookup_table(
                            cytometry_dict, selected_gates
                        )
                    except Exception as e:
                        print(f"[DR TransformTab] Gating failed for {path.name}: {e}")

                for i, ch in enumerate(channel_names):
                    if i < unmixed.shape[1]:
                        result[ch].append(unmixed[:, i])

                print(f"  Loaded {path.name}: {len(unmixed):,} events")
            except Exception as e:
                print(f"[DR TransformTab] Failed to load {path}: {e}")

        return result

    def _save_computed_transforms(self):
        """
        Persist the currently computed local Transform W values to QSettings,
        keyed by experiment directory.  Called after auto-transform completes.
        """
        try:
            exp_dir = str(self.controller.experiment_dir)
            safe_key = exp_dir.replace('\\', '/').replace(':', '_').replace(' ', '_')
            s = QSettings('honeychrome', f'plugin_{plugin_name}')
            s.beginGroup(safe_key)
            computed = {
                ch: round(tr.logicle_w, 6)
                for ch, tr in self._transforms.items()
                if tr.id == 1
            }
            s.setValue('computed_transform_W', repr(computed))
            s.endGroup()
            print(f"[DR TransformTab] Saved computed W for {len(computed)} channels")
        except Exception as e:
            print(f"[DR TransformTab] Could not save computed transforms: {e}")

    def _load_computed_transforms(self):
        """
        Restore previously computed W values from QSettings and apply them
        to the local Transform objects.  Called from refresh() after transforms
        are loaded from the experiment.
        """
        try:
            exp_dir = str(self.controller.experiment_dir)
            safe_key = exp_dir.replace('\\', '/').replace(':', '_').replace(' ', '_')
            s = QSettings('honeychrome', f'plugin_{plugin_name}')
            s.beginGroup(safe_key)
            computed_repr = s.value('computed_transform_W', '')
            s.endGroup()
            if not computed_repr:
                return
            computed: dict[str, float] = eval(computed_repr)  # noqa: S307
            M = self._compute_M()
            n_restored = 0
            for ch, W in computed.items():
                if ch not in self._transforms:
                    continue
                tr = self._transforms[ch]
                if tr.id != 1:
                    continue
                W = max(0.0, min(M / 2.0, float(W)))
                tr.logicle_w = W
                tr.set_transform()
                self._sync_transform_to_state(ch, tr)
                n_restored += 1
            if n_restored:
                print(f"[DR TransformTab] Restored computed W for {n_restored} channels")
        except Exception as e:
            print(f"[DR TransformTab] Could not restore computed transforms: {e}")



class GroupsStatsTab(QWidget):
    """
    Tab 2 — Groups, Factors & Statistics
    --------------------------------------
    Stage 6 (complete):
      • Named comparison groups with regex auto-population
      • Sample-to-group assignment table with combo boxes
      • Covariate entry (arbitrary named columns)
      • CSV import / export
      • Statistics controls: cluster frequencies (limma), cluster counts
        (GLM, Item 14), and/or MFIs, p-value and log2FC thresholds
      • Background limma statistics via inmoose
      • Results: heatmap+dendrogram and volcano plot, CSV export
    """

    def __init__(self, state: PipelineState, bus, controller, parent=None):
        super().__init__(parent)
        self.state = state
        self.bus = bus
        self.controller = controller
        # Item 13: group names/patterns/compare-selection all live directly
        # on state now (state.group_names / state.group_patterns /
        # state.compare_group_a / state.compare_group_b) — no local mirror
        # to keep in sync.
        self._stats_worker = None
        # (run_id, run_freq, run_mfi, groups_fingerprint) for the last
        # successfully computed Run Statistics — lets _run_statistics
        # replot instead of re-running limma when only thresholds changed.
        self._last_stats_data_key = None
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

        # ============================================================
        # Top panel: group definitions + sample assignment table
        # ============================================================
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)

        # ---- Group management (Item 13 — arbitrary N named groups) ----
        group_box = QGroupBox("Comparison Groups")
        group_box_layout = QVBoxLayout(group_box)

        name_hint = QLabel(
            "Define one or more comparison groups. Double-click a group's "
            "name or match pattern below to edit it in place. Assign "
            "samples to a group in the table beneath, or set a match "
            "pattern per group and use 'Auto-assign by pattern'."
        )
        name_hint.setWordWrap(True)
        name_hint.setStyleSheet("color: grey; font-style: italic; font-size: 10px;")
        group_box_layout.addWidget(name_hint)

        self.groups_table = QTableWidget(0, 2)
        self.groups_table.setHorizontalHeaderLabels(['Group Name', 'Match Pattern (regex)'])
        self.groups_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.groups_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.groups_table.verticalHeader().setVisible(False)
        self.groups_table.setEditTriggers(QAbstractItemView.DoubleClicked)
        self.groups_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.groups_table.setMaximumHeight(120)
        self.groups_table.itemChanged.connect(self._on_groups_table_item_changed)
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
            "Only currently-Unassigned, on-screen samples are affected."
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
        csv_row.addWidget(self.import_csv_btn)
        csv_row.addWidget(self.export_csv_btn)
        csv_row.addWidget(self.add_column_btn)
        csv_row.addStretch()

        self._group_count_label = QLabel()
        self._group_count_label.setStyleSheet("color: grey; font-style: italic;")
        csv_row.addWidget(self._group_count_label)
        table_box_layout.addLayout(csv_row)

        # Table: Sample | Group | [covariate columns...] -- column count is
        # dynamic (one per state.covariates column, added via '+ Add
        # Column' or CSV import). Never includes single-stain/unstained
        # controls (see _non_control_sample_paths).
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(['Sample', 'Group'])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
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

        # ---- Groups to Test + contrast mode + pairing (Item 13 phase 2) ----
        test_groups_box = QGroupBox("Groups to Test")
        test_groups_layout = QVBoxLayout(test_groups_box)

        test_groups_hint = QLabel(
            "Check every group to include in Frequency/Counts/MFI testing, "
            "Confusion Matrix, and Composition-by-group."
        )
        test_groups_hint.setWordWrap(True)
        test_groups_hint.setStyleSheet("color: grey; font-style: italic; font-size: 10px;")
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
            "term (e.g. donor ID). InMoose has no duplicateCorrelation-"
            "style random-effect blocking documented -- this is a fixed-"
            "effect term, which needs a full-rank design (each pairing "
            "level shouldn't be unique to one group)."
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

        stats_layout.addWidget(test_groups_box)

        # ---- T-REX Compare selector (Item 13 — pairwise only, see §0.2) ----
        compare_row = QHBoxLayout()
        compare_row.addWidget(QLabel("T-REX Compare:"))
        self.compare_group_a_combo = QComboBox()
        self.compare_group_a_combo.setMinimumWidth(140)
        compare_row.addWidget(self.compare_group_a_combo)
        compare_row.addWidget(QLabel("vs"))
        self.compare_group_b_combo = QComboBox()
        self.compare_group_b_combo.setMinimumWidth(140)
        compare_row.addWidget(self.compare_group_b_combo)
        compare_row.addStretch()
        stats_layout.addLayout(compare_row)
        self.compare_group_a_combo.currentTextChanged.connect(self._on_compare_group_changed)
        self.compare_group_b_combo.currentTextChanged.connect(self._on_compare_group_changed)

        compare_hint = QLabel(
            "T-REX's neighbour-fraction score only works for exactly two "
            "conditions, so it uses this dedicated pair regardless of how "
            "many groups are checked above."
        )
        compare_hint.setWordWrap(True)
        compare_hint.setStyleSheet("color: grey; font-style: italic; font-size: 10px;")
        stats_layout.addWidget(compare_hint)

        # ---- T-REX DR run selector ----
        # T-REX can only ever be plotted against the events a specific DR
        # run actually embedded -- it must be scored against exactly that
        # run's event set, not freshly re-gated data that may have since
        # drifted in size.
        trex_dr_row = QHBoxLayout()
        trex_dr_row.addWidget(QLabel("T-REX DR run:"))
        self.trex_dr_run_combo = QComboBox()
        self.trex_dr_run_combo.setMinimumWidth(160)
        self.trex_dr_run_combo.setToolTip(
            "The archived DR run T-REX will score against. Only events "
            "that run actually embedded can be plotted with a T-REX score."
        )
        self.trex_dr_run_combo.currentIndexChanged.connect(self._on_trex_dr_run_changed)
        trex_dr_row.addWidget(self.trex_dr_run_combo)
        trex_dr_row.addStretch()
        stats_layout.addLayout(trex_dr_row)

        # Config row: what to test
        config_row = QHBoxLayout()
        config_row.addWidget(QLabel("Test:"))
        self.chk_freq = QCheckBox("Cluster Frequencies (limma)")
        self.chk_freq.setChecked(True)
        self.chk_freq.setToolTip("% events per cluster per sample → limma lmFit + eBayes")
        self.chk_counts = QCheckBox("Cluster Counts (GLM)")
        self.chk_counts.setChecked(False)
        self.chk_counts.setToolTip(
            "Item 14 — raw event counts per cluster per sample → negative-"
            "binomial GLM.\nParallel option to the Frequency test, not a "
            "replacement — run both to compare on the same data."
        )
        self.chk_mfi = QCheckBox("Cluster MFIs")
        self.chk_mfi.setChecked(True)
        self.chk_mfi.setToolTip("Mean channel intensity per cluster per sample → limma")
        config_row.addWidget(self.chk_freq)
        config_row.addWidget(self.chk_counts)
        config_row.addWidget(self.chk_mfi)
        config_row.addSpacing(20)

        config_row.addWidget(QLabel("p-value ≤"))
        self.pval_spin = QDoubleSpinBox()
        self.pval_spin.setRange(0.001, 1.0)
        self.pval_spin.setSingleStep(0.01)
        self.pval_spin.setDecimals(3)
        self.pval_spin.setValue(0.05)
        self.pval_spin.setFixedWidth(70)
        config_row.addWidget(self.pval_spin)

        config_row.addWidget(QLabel("|log₂FC| ≥"))
        self.fc_spin = QDoubleSpinBox()
        self.fc_spin.setRange(0.0, 10.0)
        self.fc_spin.setSingleStep(0.1)
        self.fc_spin.setDecimals(2)
        self.fc_spin.setValue(0.5)
        self.fc_spin.setFixedWidth(70)
        config_row.addWidget(self.fc_spin)
        config_row.addStretch()
        stats_layout.addLayout(config_row)

        # ---- Marker roles (Item 11): type (clustering) vs state (tested) ----
        roles_box = QGroupBox("Marker Roles — MFI Testing")
        roles_layout = QVBoxLayout(roles_box)
        roles_hint = QLabel(
            "Channels used to build the selected clustering run default to "
            "'type' and are excluded from MFI significance testing, to avoid "
            "the same channel driving both the cluster assignment and its "
            "own significance call. Check a channel to mark it 'type' "
            "(excluded); uncheck to test it ('state')."
        )
        roles_hint.setWordWrap(True)
        roles_hint.setStyleSheet("color: grey; font-style: italic; font-size: 10px;")
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
        self.chk_include_type_markers.setChecked(False)
        self.chk_include_type_markers.setToolTip(
            "Off (default): MFI significance testing uses 'state'-role "
            "channels only. On: restores testing every selected channel, "
            "regardless of role."
        )
        roles_layout.addWidget(self.chk_include_type_markers)

        stats_layout.addWidget(roles_box)

        # Run / status row
        run_row = QHBoxLayout()
        self.run_stats_btn = QPushButton("▶  Run Statistics")
        self.run_stats_btn.setEnabled(False)
        self.run_stats_btn.setToolTip("Requires ≥ 3 samples in each group and clustering complete.")
        self.run_stats_btn.clicked.connect(self._run_statistics)
        run_row.addWidget(self.run_stats_btn)

        self.run_trex_btn = QPushButton("▶  Run T-REX")
        self.run_trex_btn.setEnabled(False)
        self.run_trex_btn.setToolTip(
            "Build T-REX kNN index and score all samples.\n"
            "Results appear in Workspace plots (Colour: T-REX)."
        )
        self.run_trex_btn.clicked.connect(self._run_trex)
        run_row.addWidget(self.run_trex_btn)

        self.confusion_btn = QPushButton("Confusion Matrix")
        self.confusion_btn.setEnabled(False)
        self.confusion_btn.setToolTip(
            "Per-group-normalized cluster composition heatmap.\n"
            "Available as soon as groups are assigned and a clustering run "
            "is selected — no Frequency/MFI stats required first."
        )
        self.confusion_btn.clicked.connect(self._show_confusion_matrix)
        run_row.addWidget(self.confusion_btn)

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
        run_row.addWidget(self.stats_status_label)
        run_row.addStretch()

        self.export_results_btn = QPushButton("Export Results CSV")
        self.export_results_btn.setEnabled(False)
        self.export_results_btn.clicked.connect(self._export_results_csv)
        run_row.addWidget(self.export_results_btn)
        stats_layout.addLayout(run_row)

        # ---- Viewing comparison (Item 13 phase 2) ----
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

        # Results area: one tab per plot
        self._results_tabs = QTabWidget()
        self._results_tabs.setTabsClosable(True)
        self._results_tabs.tabCloseRequested.connect(self._on_results_tab_close_requested)
        self._results_tabs.setMinimumHeight(700)
        bottom_layout.addWidget(self._results_tabs, stretch=1)

        content_layout.addWidget(bottom_widget, stretch=1)

        # Sentinel: cluster names snapshot at last _draw_results() call.
        self._last_drawn_cluster_names: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Group management (Item 13 — arbitrary N named groups)
    # ------------------------------------------------------------------

    def _group_options(self) -> list[str]:
        """Return the full list of group options for the per-sample combos."""
        return ['Unassigned'] + list(self.state.group_names)

    def _populate_groups_table(self):
        """Rebuild the Group Name / Match Pattern management table from state."""
        self.groups_table.blockSignals(True)
        self.groups_table.setRowCount(0)
        for name in self.state.group_names:
            row = self.groups_table.rowCount()
            self.groups_table.insertRow(row)
            self.groups_table.setItem(row, 0, QTableWidgetItem(name))
            self.groups_table.setItem(
                row, 1, QTableWidgetItem(self.state.group_patterns.get(name, ''))
            )
        self.groups_table.blockSignals(False)

    def _on_groups_table_item_changed(self, item: QTableWidgetItem):
        """
        Handle an in-place edit of a group's name (col 0) or match pattern
        (col 1). A name edit is a REAL rename now — sample_groups values
        ARE group names (no slot indirection left, see §0.2) — so every
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
                if self.state.compare_group_a == old_name:
                    self.state.compare_group_a = new_name
                if self.state.compare_group_b == old_name:
                    self.state.compare_group_b = new_name
                self.state.group_names[row] = new_name
                self.state.invalidate_trex()
                self._populate_table()
                self._refresh_compare_combos()
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
        self._populate_groups_table()
        self._refresh_group_combos()
        self._refresh_compare_combos()
        self._update_group_count_label()

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
        if self.state.compare_group_a == name:
            self.state.compare_group_a = ''
        if self.state.compare_group_b == name:
            self.state.compare_group_b = ''
        self.state.invalidate_trex()
        self._populate_groups_table()
        self._populate_table()
        self._refresh_compare_combos()
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

    def _refresh_compare_combos(self):
        """Repopulate the Compare: selector from state.group_names, trying
        to preserve the current selection; falls back to the first two
        defined groups."""
        names = list(self.state.group_names)
        for combo in (self.compare_group_a_combo, self.compare_group_b_combo):
            combo.blockSignals(True)
        prev_a, prev_b = self.state.compare_group_a, self.state.compare_group_b
        self.compare_group_a_combo.clear()
        self.compare_group_a_combo.addItems(names)
        self.compare_group_b_combo.clear()
        self.compare_group_b_combo.addItems(names)
        idx_a = self.compare_group_a_combo.findText(prev_a)
        idx_b = self.compare_group_b_combo.findText(prev_b)
        self.compare_group_a_combo.setCurrentIndex(idx_a if idx_a >= 0 else 0)
        default_b = 1 if len(names) > 1 else 0
        self.compare_group_b_combo.setCurrentIndex(idx_b if idx_b >= 0 else default_b)
        for combo in (self.compare_group_a_combo, self.compare_group_b_combo):
            combo.blockSignals(False)
        self.state.compare_group_a = self.compare_group_a_combo.currentText()
        self.state.compare_group_b = self.compare_group_b_combo.currentText()

    def _on_compare_group_changed(self, _text: str):
        self.state.compare_group_a = self.compare_group_a_combo.currentText()
        self.state.compare_group_b = self.compare_group_b_combo.currentText()
        self.state.invalidate_trex()
        self._update_run_button()
        self._update_group_count_label()

    # ------------------------------------------------------------------
    # Groups to Test / contrast mode / pairing (Item 13 phase 2)
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

    def _on_pairing_variable_changed(self, text: str):
        self.state.pairing_variable = text
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
        the samples currently shown in the table, assigning only Unassigned
        ones. First matching pattern wins, in state.group_names order.
        Patterns are separate from group names so a group named 'A' can't
        match every file (bug R1), and only on-screen samples are touched
        (bug R2) — Item 13 generalizes this from a fixed A/B pair to
        however many groups are defined.
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

        # Only the samples currently in the table (the training-filtered view).
        shown = [self.table.item(r, 0).data(Qt.UserRole)
                 for r in range(self.table.rowCount())
                 if self.table.item(r, 0) is not None]

        _log.info("auto-assign: table rows=%d, shown=%d", self.table.rowCount(), len(shown))
        for sp in shown:
            _log.info("  sample key=%r  filename=%r  current=%r",
                      sp, Path(sp).name, self.state.sample_groups.get(sp, 'Unassigned'))

        assigned_counts = {name: 0 for name in patterns}
        for sp in shown:
            current = self.state.sample_groups.get(sp, 'Unassigned')
            if current != 'Unassigned':
                _log.info("  skip %r (already %r)", Path(sp).name, current)
                continue
            fn = Path(sp).name
            for name, rx in patterns.items():
                if rx.search(fn):
                    self.state.sample_groups[sp] = name
                    assigned_counts[name] += 1
                    _log.info("  → %s: %r", name, fn)
                    break
            else:
                _log.info("  no match: %r", fn)

        self.state.invalidate_trex()
        self.refresh()                  # repopulate the SAME (training) view
        self._update_run_button()
        summary = ", ".join(f"{n} → {name}" for name, n in assigned_counts.items())
        self.stats_status_label.setText(f"Auto-assigned {summary}.")

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self):
        """
        Rebuild the sample table showing only the training samples selected
        in Configuration.  Falls back to all samples if none are selected.
        """
        # Item 13: group names/patterns/compare-selection live on state
        # directly now — just repopulate the management widgets from it.
        self._populate_groups_table()
        self._refresh_compare_combos()

        # Item 13 phase 2: Groups-to-Test / contrast mode / pairing widgets.
        self._populate_test_groups_list()
        self._populate_pairing_variable_combo()
        self.chk_paired.blockSignals(True)
        self.chk_paired.setChecked(self.state.paired)
        self.chk_paired.blockSignals(False)
        self.pairing_variable_combo.setEnabled(self.state.paired)
        self.radio_reference.blockSignals(True)
        self.radio_pairwise.blockSignals(True)
        if self.state.contrast_mode == 'pairwise':
            self.radio_pairwise.setChecked(True)
        else:
            self.radio_reference.setChecked(True)
        self.radio_reference.blockSignals(False)
        self.radio_pairwise.blockSignals(False)
        self.reference_group_combo.setEnabled(self.radio_reference.isChecked())

        training = self.state.training_sample_ids
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

        self._populate_table(matched)
        self._populate_run_combo()
        self._populate_trex_dr_combo()
        self._populate_marker_roles_list()
        self._update_run_button()

        # Redraw plots if results are present and haven't been rendered yet
        # (_last_drawn_cluster_names == {} after load_state resets it) or if
        # cluster names changed since last draw (e.g. after a rename).
        if (self.state.freq_results is not None or self.state.mfi_results is not None
                or self.state.counts_results is not None):
            if (not self._last_drawn_cluster_names or
                    dict(self.state.cluster_names) != self._last_drawn_cluster_names):
                self._draw_results()

        # Confusion Matrix / Composition Barplot are independent of Run
        # Statistics and of each other — redraw each once per load if its
        # own persisted data is present and it isn't already showing.
        if self.state.confusion_df is not None and not self._has_results_tab('confusion_matrix'):
            fig = self._make_confusion_matrix_figure(
                self.state.confusion_df, run_label=self.state.confusion_run_label,
            )
            self._add_results_tab(
                fig, "Confusion Matrix", "confusion_matrix",
                maker=self._make_confusion_matrix_figure,
                maker_kwargs=dict(conf_df=self.state.confusion_df,
                                  run_label=self.state.confusion_run_label),
                key="confusion_matrix",
            )
        if self.state.composition_df is not None and not self._has_results_tab('composition_barplot'):
            self.composition_pct_chk.setChecked(self.state.composition_as_pct)
            self.composition_by_group_chk.setChecked(self.state.composition_group_var == 'group')
            fig = self._make_composition_figure(
                self.state.composition_df, as_pct=self.state.composition_as_pct,
                run_label=self.state.composition_run_label, names=self.state.composition_names,
                group_var=self.state.composition_group_var,
            )
            self._add_results_tab(
                fig, "Composition", "composition_barplot",
                maker=self._make_composition_figure,
                maker_kwargs=dict(comp_df=self.state.composition_df,
                                  as_pct=self.state.composition_as_pct,
                                  run_label=self.state.composition_run_label,
                                  names=self.state.composition_names,
                                  group_var=self.state.composition_group_var),
                key="composition_barplot",
            )

    # ------------------------------------------------------------------
    # Table population
    # ------------------------------------------------------------------

    def _populate_table(self, sample_groups: dict | None = None):
        """Rebuild table rows AND covariate columns from sample_groups.
        Column count is dynamic -- one per state.covariates column, so
        this must rebuild the header every call, not just the rows."""
        if sample_groups is None:
            sample_groups = self.state.sample_groups

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
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        for c in range(2, 2 + len(covariate_cols)):
            self.table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)

        opts = self._group_options()
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
        self.state.invalidate_trex()
        self._update_run_button()
        self._update_group_count_label()

    def _update_group_count_label(self):
        counts = self.state.n_per_group()
        parts = [f"{name}: {counts.get(name, 0)}" for name in self.state.group_names]
        unassigned = sum(1 for g in self.state.sample_groups.values() if g == 'Unassigned')
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
        any_clustering = bool(self.state.clustering_runs) or bool(self.state.cluster_labels)
        dr_only_selected = self._selected_run_kind() == 'dr'
        n_group_ok = self.state.n_group_stats_runnable()
        runnable = n_group_ok and any_clustering and not dr_only_selected
        self.run_stats_btn.setEnabled(runnable)
        self.run_trex_btn.setEnabled(self.state.stats_runnable())
        self.confusion_btn.setEnabled(runnable)
        self.composition_btn.setEnabled(runnable)
        if dr_only_selected:
            self.run_stats_btn.setToolTip(
                "This is a DR run — it has no cluster labels.  Select or "
                "run a clustering run to compute statistics."
            )
        elif not any_clustering:
            self.run_stats_btn.setToolTip("Run clustering first.")
        elif not n_group_ok:
            counts = self.state.n_per_group()
            selection = self.state.testing_group_selection or self.state.group_names
            detail = ", ".join(f"{name}={counts.get(name, 0)}" for name in selection)
            self.run_stats_btn.setToolTip(
                f"Requires ≥ 3 samples in at least two checked 'Groups to "
                f"Test'.  Currently: {detail}."
            )
        else:
            self.run_stats_btn.setToolTip("")

    # ------------------------------------------------------------------
    # Run combo management
    # ------------------------------------------------------------------

    def _populate_run_combo(self):
        """
        Rebuild the unified run combo — DR + clustering runs together
        (Item 6 Decision #2).  A run is included only if its recorded
        training sample set overlaps the samples currently assigned to
        group A or B.

        Uses each entry's 'training_sample_ids' manifest field, NEVER
        'labels'/'embeddings' — this works from metadata alone (every
        entry has it, hydrated or not), so populating this combo never
        forces a run to be loaded from disk.  Only actually selecting one
        does that (see _selected_run_entry).

        Combo items are keyed by run_id (userData), not label text —
        labels are user-editable (Item 6's rename) and not guaranteed
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
            list(self.state.dr_runs) + list(self.state.clustering_runs),
            key=lambda e: e.get('timestamp', ''),
        )
        for entry in all_runs:
            run_keys = set(entry.get('training_sample_ids', []))
            if not run_keys.isdisjoint(assigned_rel) or not assigned_rel:
                suffix = ' · DR' if entry.get('kind') == 'dr' else ''
                self._run_combo.addItem(entry['label'] + suffix, entry.get('run_id'))

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

    def _populate_trex_dr_combo(self):
        """T-REX needs its own DR-run selector (distinct from the Stats
        run combo, which is clustering-only, and from Workspace's own
        dr_combo, a different widget) -- see state.trex_dr_run_id."""
        self.trex_dr_run_combo.blockSignals(True)
        prev_run_id = self.trex_dr_run_combo.currentData()
        self.trex_dr_run_combo.clear()
        for entry in self.state.dr_runs:
            self.trex_dr_run_combo.addItem(entry.get('label', ''), entry.get('run_id'))
        if self.trex_dr_run_combo.count() == 0:
            self.trex_dr_run_combo.addItem("(no DR runs yet)", None)
        preferred_id = self.state.trex_dr_run_id or prev_run_id
        idx = self.trex_dr_run_combo.findData(preferred_id) if preferred_id else -1
        if idx >= 0:
            self.trex_dr_run_combo.setCurrentIndex(idx)
        else:
            self.trex_dr_run_combo.setCurrentIndex(self.trex_dr_run_combo.count() - 1)
        self.trex_dr_run_combo.blockSignals(False)
        self.state.trex_dr_run_id = self.trex_dr_run_combo.currentData()

    def _on_trex_dr_run_changed(self, _index=None):
        self.state.trex_dr_run_id = self.trex_dr_run_combo.currentData()

    def _selected_trex_dr_run(self) -> dict | None:
        run_id = self.trex_dr_run_combo.currentData()
        if run_id is None:
            return None
        for entry in self.state.dr_runs:
            if entry.get('run_id') == run_id:
                return drc_run_archive.hydrate_run(self.controller, entry)
        return None

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
        """
        run_id = self._run_combo.currentData()
        self._update_run_button()
        if run_id is None:
            return
        if run_id == self.state.stats_run_id:
            # Matches stored results — redraw if needed.
            if (self.state.freq_results is not None or self.state.mfi_results is not None
                    or self.state.counts_results is not None):
                self._draw_results()
            return
        # Different run — clear stored results and prompt re-run.
        self.state.freq_results = None
        self.state.mfi_results  = None
        self.state.freq_df      = None
        self.state.mfi_df       = None
        self.state.mfi_sample_df = None
        self.state.counts_results = None
        self.state.counts_df      = None
        self.state.stats_all_rel  = []
        self.state.stats_group_vec = []
        self._last_stats_data_key = None
        self._results_tabs.clear()
        self.export_results_btn.setEnabled(False)
        self.stats_status_label.setText(
            "Run changed — click 'Run Statistics' to recompute."
        )
        self.stats_status_label.setStyleSheet("color: orange;")

    # ------------------------------------------------------------------
    # Marker roles (Item 11) — type (clustering) vs state (tested)
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

        for grid_idx, ch in enumerate(channels):
            role = self.state.marker_roles.get(ch)
            if role is None:
                role = 'state'
                self.state.marker_roles[ch] = role
            cb = QCheckBox(ch)
            cb.setChecked(role == 'type')
            cb.toggled.connect(lambda checked, c=ch: self._on_marker_role_changed(c, checked))
            self.marker_roles_checkboxes[ch] = cb
            row, col = divmod(grid_idx, 4)
            self.marker_roles_grid.addWidget(cb, row, col)

    def _on_marker_role_changed(self, ch: str, checked: bool):
        self.state.marker_roles[ch] = 'type' if checked else 'state'

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
            # Item 13 phase 2: any column beyond sample/group/full_path is a
            # covariate (e.g. donor ID for paired testing) -- the CSV
            # import tooltip already promised this; it just wasn't wired
            # up until now.
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
            self.state.invalidate_trex()
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
    # Statistics — Stage 6
    # ------------------------------------------------------------------

    def _run_statistics(self):
        """
        Launch background limma/GLM statistics — or, if the same run and
        group assignment already produced results and only the
        significance thresholds changed, just re-flag significance and
        replot without re-running the (intensive) limma/GLM calls.
        """
        if self._stats_worker is not None and self._stats_worker.isRunning():
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

        pval_threshold = self.pval_spin.value()
        fc_threshold   = self.fc_spin.value()

        include_type_markers = self.chk_include_type_markers.isChecked()
        groups_fingerprint = tuple(sorted(self.state.sample_groups.items()))
        test_fingerprint = (
            tuple(self.state.testing_group_selection), self.state.contrast_mode,
            self.state.reference_group, self.state.paired, self.state.pairing_variable,
        )
        roles_fingerprint  = tuple(sorted(self.state.marker_roles.items()))
        data_key = (run_id, run_freq, run_counts, run_mfi, groups_fingerprint,
                   test_fingerprint, include_type_markers, roles_fingerprint)
        have_results = (not run_freq or self.state.freq_results is not None) and \
                       (not run_counts or self.state.counts_results is not None) and \
                       (not run_mfi or self.state.mfi_results is not None)

        self.state.stats_run_label = run_label
        self.state.stats_run_id = run_id

        if data_key == self._last_stats_data_key and have_results:
            # Same run, same groups, same tests already computed — only the
            # thresholds may have changed. Re-flag significance in place
            # and replot instead of re-running limma/GLM.
            self._apply_significance_thresholds(pval_threshold, fc_threshold)
            self.stats_status_label.setText("✓ Statistics complete (replotted — run unchanged).")
            self.stats_status_label.setStyleSheet("color: green;")
            self.export_results_btn.setEnabled(True)
            self._draw_results()
            return

        self.run_stats_btn.setEnabled(False)
        self.stats_status_label.setText("⏳ Computing statistics …")
        self.stats_status_label.setStyleSheet("color: orange;")

        plugin_ref = self
        # Snapshot AF/transfer-matrix state HERE, on the main thread, before
        # the worker starts. compute_mfis() reads these via
        # drc_pipeline.apply_unmixing_af_aware(); the live controller
        # attributes are reassigned in place by controller.load_sample() /
        # initialise_af_matrices() whenever the user loads a different
        # sample in the main window. Without this snapshot the background
        # worker and a main-window sample load race on the same mutable
        # numpy arrays — and the AF kernel touches them via raw C pointers
        # (af_kernel_wrapper.py), so a concurrent reassignment is a
        # memory-corruption/crash hazard, not just stale data.
        af_state = (
            self.controller.transfer_matrix,
            self.controller.af_precomputed,
            self.controller.af_spectra,
        )

        class _StatsWorker(QThread):
            finished = Signal(bool, str)
            progress = Signal(str)

            def __init__(self_, run_freq, run_mfi, group_names, labels_override,
                        include_type_markers, names_override, run_counts, af_state):
                super().__init__()
                self_._run_freq = run_freq
                self_._run_mfi  = run_mfi
                self_._run_counts = run_counts
                self_._group_names = group_names
                self_._labels_override = labels_override
                self_._include_type_markers = include_type_markers
                self_._names_override = names_override
                self_._af_state = af_state

            def run(self_):
                try:
                    self_._do_stats()
                    self_.finished.emit(True, '')
                except Exception as exc:
                    traceback.print_exc()
                    self_.finished.emit(False, str(exc))

            def _do_stats(self_):
                freq, mfi, counts = drc_stats.run_statistics(
                    plugin_ref.controller, plugin_ref.state,
                    self_._run_freq, self_._run_mfi,
                    pval_threshold, fc_threshold,
                    cluster_labels_override=self_._labels_override,
                    include_type_markers=self_._include_type_markers,
                    names_override=self_._names_override,
                    run_counts=self_._run_counts,
                    af_state=self_._af_state,
                )
                if freq is not None:
                    self_.progress.emit(f"Frequency limma: {len(freq)} clusters tested.")
                if counts is not None:
                    self_.progress.emit(f"Counts GLM: {len(counts)} clusters tested.")
                if mfi is not None:
                    self_.progress.emit(f"MFI limma: {len(mfi)} features tested.")

        worker = _StatsWorker(run_freq, run_mfi, list(self.state.group_names), labels_for_stats,
                             include_type_markers, names_for_stats, run_counts, af_state)
        worker.progress.connect(lambda msg: print(f"[DR Stats] {msg}"))
        worker.finished.connect(lambda success, err, key=data_key: self._on_stats_finished(success, err, key))
        self._stats_worker = worker
        worker.start()

    def _apply_significance_thresholds(self, pval_threshold: float, fc_threshold: float):
        """
        Recompute the 'significant' column on already-computed freq/counts/
        mfi results in place, without re-running limma/GLM. logFC/p-values
        are threshold-independent, so this is all that's needed when only
        the thresholds changed since the last run.
        """
        for attr in ('freq_results', 'counts_results', 'mfi_results'):
            df = getattr(self.state, attr)
            if df is None or 'logFC' not in df.columns:
                continue
            # Item 13 phase 2 addendum (§2.7): prefer the global (all-
            # contrasts-pooled) correction, matching run_limma()/
            # run_glm_counts()'s own default. The plain-adj.P.Val/P.Value
            # fallbacks only matter for results computed before this fix
            # landed (results aren't persisted across sessions, so this is
            # just a same-session safety net, not a migration path).
            if 'adj.P.Val.global' in df.columns:
                pval_col = 'adj.P.Val.global'
            elif 'adj.P.Val' in df.columns:
                pval_col = 'adj.P.Val'
            else:
                pval_col = 'P.Value'
            df['significant'] = (
                (df[pval_col] <= pval_threshold) &
                (df['logFC'].abs() >= fc_threshold)
            )

    def _resolve_stats_source(self):
        """
        Resolve which run's cluster labels to use for Run Statistics /
        Confusion Matrix / Composition Barplot, plus a short label to stamp
        on the resulting plot(s), a stable run id for cache/run-change
        checks, and that SAME run's own names dict (never the ambient
        state.cluster_names — see the "bleeding across runs" fix). All
        three views share this one validation path.

        Returns (labels_dict, run_label, run_id, names_dict) or None (after
        showing the appropriate warning) if nothing usable is selected.
        """
        run_entry = self._selected_run_entry()
        if run_entry is not None and run_entry.get('kind') != 'clustering':
            run_entry = None
        if run_entry is None and not self.state.cluster_labels:
            QMessageBox.warning(self, "No Clustering", "Run clustering first.")
            return None
        if not self.state.n_group_stats_runnable():
            counts = self.state.n_per_group()
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

    def _show_confusion_matrix(self):
        """
        Compute and display the confusion-matrix heatmap (Item 10).
        Independent of Run Statistics — usable as soon as groups are
        assigned and a clustering run is selected. Re-clicking replaces
        the existing Confusion Matrix tab in place rather than adding
        another.
        """
        resolved = self._resolve_stats_source()
        if resolved is None:
            return
        labels_for_stats, run_label, _run_id, names_for_stats = resolved

        try:
            conf_df = drc_stats.compute_confusion_matrix(
                self.controller, self.state, cluster_labels_override=labels_for_stats,
                names_override=names_for_stats,
            )
        except Exception as e:
            QMessageBox.critical(self, "Confusion Matrix Error", str(e))
            return

        self.state.confusion_df = conf_df
        self.state.confusion_run_label = run_label
        self.state.confusion_names = dict(names_for_stats)

        fig = self._make_confusion_matrix_figure(conf_df, run_label=run_label)
        self._add_results_tab(
            fig, "Confusion Matrix", "confusion_matrix",
            maker=self._make_confusion_matrix_figure,
            maker_kwargs=dict(conf_df=conf_df, run_label=run_label),
            key="confusion_matrix",
        )

    def _make_confusion_matrix_figure(self, conf_df: 'pd.DataFrame', run_label: str = ''):
        """
        Per-group-normalized cluster composition heatmap (CyCONDOR's
        plot_confusion_HM). Each cell = that group's normalized share of a
        cluster.
        """
        from matplotlib.figure import Figure

        if conf_df.empty:
            fig = Figure(figsize=(5, 2), constrained_layout=True)
            ax = fig.add_subplot(111)
            ax.axis('off')
            ax.text(0.5, 0.5, 'No clusters to display',
                    ha='center', va='center', fontsize=10, transform=ax.transAxes)
            self._stamp_run_label(fig, run_label)
            return fig

        # Item 13: drc_stats.compute_confusion_matrix() now returns the
        # real selected group names as columns directly — no rename needed.
        disp_df = conf_df
        n_clusters = len(disp_df)

        fig_w = max(4.0, 1.2 + len(disp_df.columns) * 1.2)
        fig_h = max(3.0, 0.35 * n_clusters + 1.2)
        fig = Figure(figsize=(fig_w, fig_h), layout='constrained')
        ax = fig.add_subplot(111)

        im = ax.imshow(disp_df.values, aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(disp_df.columns)))
        ax.set_xticklabels(disp_df.columns, rotation=0)
        ax.set_yticks(range(n_clusters))
        ax.set_yticklabels(disp_df.index)
        ax.grid(False)   # suppress inherited seaborn 'whitegrid' (draws through tick/cell centres)
        ax.set_xticks(np.arange(-0.5, len(disp_df.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_clusters, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.6)
        ax.tick_params(which='minor', bottom=False, left=False)
        ax.set_title("Cluster Composition by Group\n(normalized per-group event count)",
                    fontsize=10)

        vmax = disp_df.values.max() if disp_df.values.size else 0.0
        for i in range(n_clusters):
            for j in range(len(disp_df.columns)):
                val = disp_df.values[i, j]
                ax.text(j, i, f"{val:.0f}", ha='center', va='center',
                        color='white' if val < vmax * 0.6 else 'black',
                        fontsize=8)

        fig.colorbar(im, ax=ax, shrink=0.7, label='Normalized events')
        self._stamp_run_label(fig, run_label)
        return fig

    def _show_composition_barplot(self):
        """
        Compute and display the stacked composition barplot (Item 12).
        Same availability gate as Item 10's confusion matrix. Re-clicking
        (or toggling %/By-group) replaces the existing Composition tab in
        place rather than adding another.
        """
        resolved = self._resolve_stats_source()
        if resolved is None:
            return
        labels_for_stats, run_label, _run_id, names_for_stats = resolved

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
        self.state.composition_names = dict(names_for_stats)

        fig = self._make_composition_figure(comp_df, as_pct=as_pct, run_label=run_label,
                                            names=names_for_stats, group_var=group_var)
        self._add_results_tab(
            fig, "Composition", "composition_barplot",
            maker=self._make_composition_figure,
            maker_kwargs=dict(comp_df=comp_df, as_pct=as_pct, run_label=run_label,
                              names=names_for_stats, group_var=group_var),
            key="composition_barplot",
        )

    def _make_composition_figure(self, comp_df: 'pd.DataFrame', as_pct: bool,
                                 run_label: str = '', names: dict | None = None,
                                 group_var: str = 'sample'):
        """
        Stacked bar: x-axis = sample or group (comp_df.index), segments =
        clusters, coloured via the SELECTED run's own colours (not the
        ambient state.cluster_colors, for the same "no bleeding across
        runs" reason names are now passed in explicitly) — falls back to a
        neutral grey if a cluster's colour can't be resolved.
        """
        from matplotlib.figure import Figure

        if comp_df.empty:
            fig = Figure(figsize=(5, 2), constrained_layout=True)
            ax = fig.add_subplot(111)
            ax.axis('off')
            ax.text(0.5, 0.5, 'No data to display',
                    ha='center', va='center', fontsize=10, transform=ax.transAxes)
            return fig

        n_rows = len(comp_df)
        fig_w = max(4.0, 0.6 * n_rows + 2.5)
        fig = Figure(figsize=(fig_w, 5.0), layout='constrained')
        ax = fig.add_subplot(111)

        x = np.arange(n_rows)
        bottom = np.zeros(n_rows)
        cl_run = self._selected_run_entry()
        if cl_run is not None and cl_run.get('kind') != 'clustering':
            cl_run = None
        colors = cl_run.get('colors', {}) if cl_run else self.state.cluster_colors
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
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8,
                  title='Cluster', frameon=False)
        self._stamp_run_label(fig, run_label)
        return fig

    def _limma_test(
        self,
        data_df: 'pd.DataFrame',
        group_vec: list[str],
        pval_threshold: float,
        fc_threshold: float,
    ) -> 'pd.DataFrame':
        """
        Run limma lmFit + eBayes + topTable on *data_df*.

        data_df  : rows = samples, columns = features (clusters / channel-clusters)
        group_vec: list of 'A' or 'B' per row

        Returns a DataFrame with columns:
            feature, logFC, AveExpr, t, P.Value, adj.P.Val, B, significant
        """
        from inmoose.limma import lmFit, eBayes, topTable

        n = len(group_vec)
        design = np.zeros((n, 2), dtype=float)
        for i, g in enumerate(group_vec):
            design[i, 0] = 1.0          # intercept (group A baseline)
            design[i, 1] = 1.0 if g == 'B' else 0.0

        # inmoose expects (features × samples) as a numpy array
        expr = data_df.values.T.astype(float)   # (n_features, n_samples)

        fit = lmFit(expr, design)
        fit = eBayes(fit)
        n_features = expr.shape[0]
        tt = topTable(fit, coef=1, number=n_features, sort_by='p')

        tt = tt.reset_index(drop=True)
        tt.insert(0, 'feature', data_df.columns.tolist()[:n_features])
        tt['significant'] = (
            (tt['adj.P.Val'] <= pval_threshold) &
            (tt['logFC'].abs() >= fc_threshold)
        )
        return tt

    def _on_stats_finished(self, success: bool, error_msg: str, data_key=None):
        self._stats_worker = None
        self.run_stats_btn.setEnabled(True)
        if not success:
            self.stats_status_label.setText(f"Error: {error_msg}")
            self.stats_status_label.setStyleSheet("color: red;")
            QMessageBox.critical(self, "Statistics Error", error_msg)
            return

        if data_key is not None:
            self._last_stats_data_key = data_key
        self.stats_status_label.setText("✓ Statistics complete.")
        self.stats_status_label.setStyleSheet("color: green;")
        self.export_results_btn.setEnabled(True)
        self._draw_results()

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
                      'confusion_matrix'). If a tab with the same key already
                      exists it's replaced in place (same position, refocused)
                      instead of appended — so re-clicking a button doesn't pile
                      up duplicate tabs. Every tab (keyed or not) can still be
                      closed manually via its close button.

        Extracted from the old _draw_results-local `_add_tab` closure so
        Items 10/12 can add standalone result tabs (Confusion Matrix,
        Composition) outside of _draw_results, using the same pop-out/
        export machinery as the heatmap/volcano tabs.
        """
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
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

        canvas = FigureCanvasQTAgg(fig)
        canvas.setMinimumSize(w_px, h_px)
        canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        toolbar = NavigationToolbar2QT(canvas, None)

        scroll = QScrollArea()
        scroll.setWidget(canvas)
        scroll.setWidgetResizable(True)
        # Grows to the actual figure height -- no cap. This tab's own
        # outer QScrollArea (GroupsStatsTab._build_ui) scrolls the whole
        # page if that makes it taller than the viewport, rather than
        # nesting a second scrollbar inside this one.
        scroll.setMinimumHeight(h_px + 40)

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

            dlg_canvas  = FigureCanvasQTAgg(dlg_fig)
            dlg_canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
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
        btn_row.addStretch()

        container = QWidget()
        if key is not None:
            container.setProperty('_tab_key', key)
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)
        vbox.addWidget(toolbar)
        vbox.addWidget(scroll, stretch=1)
        vbox.addWidget(placeholder, stretch=1)
        vbox.addLayout(btn_row)

        if insert_at is not None:
            self._results_tabs.insertTab(insert_at, container, tab_title)
            self._results_tabs.setCurrentIndex(insert_at)
        else:
            idx = self._results_tabs.addTab(container, tab_title)
            self._results_tabs.setCurrentIndex(idx)

    def _on_results_tab_close_requested(self, index: int):
        """Manually close any results tab (Freq/MFI heatmap/volcano,
        Confusion Matrix, or Composition Barplot)."""
        widget = self._results_tabs.widget(index)
        self._results_tabs.removeTab(index)
        if widget is not None:
            widget.deleteLater()

    def _has_results_tab(self, key: str) -> bool:
        """True if a results tab carrying this key is already showing —
        avoids adding a duplicate Confusion Matrix / Composition tab on
        every refresh() once it's been drawn once this session."""
        for i in range(self._results_tabs.count()):
            widget = self._results_tabs.widget(i)
            if widget is not None and widget.property('_tab_key') == key:
                return True
        return False

    def _remove_results_tab_by_key(self, key: str):
        """Remove the results tab whose container carries this key, if
        present. Used by _draw_results to clear only its own four tabs
        (Freq/MFI heatmap/volcano) rather than every tab — Confusion
        Matrix and Composition Barplot are independent and must survive a
        Run Statistics (re-)run."""
        for i in range(self._results_tabs.count()):
            widget = self._results_tabs.widget(i)
            if widget is not None and widget.property('_tab_key') == key:
                self._results_tabs.removeTab(i)
                widget.deleteLater()
                return

    def _draw_results(self):
        """
        Render heatmap and volcano into per-plot tabs, for the currently
        selected 'Viewing comparison' (Item 13 phase 2 — a single Run
        Statistics call can now produce several comparisons; each plot
        still shows exactly one at a time — see §2.4 of the change doc).
        """
        self._last_drawn_cluster_names = dict(self.state.cluster_names)
        # Remove only this method's own tabs (by key) — leaves Confusion
        # Matrix / Composition Barplot (and anything else) untouched.
        for key in ('freq_heatmap', 'freq_volcano', 'counts_heatmap', 'counts_volcano',
                   'mfi_heatmap', 'mfi_volcano'):
            self._remove_results_tab_by_key(key)

        self._refresh_viewing_comparison_combo()
        if not self.state.stats_comparisons:
            return
        view_idx = max(0, self.viewing_comparison_combo.currentIndex())
        name_a, name_b = self.state.stats_comparisons[view_idx]
        comparison_label = f"{name_b} vs {name_a}"
        run_label = self.state.stats_run_label or 'Active (unsaved)'

        # Samples outside this specific comparison aren't part of what it
        # tested (pairwise mode fits each pair on just its own samples;
        # reference mode's joint fit still only makes sense to *display*
        # two groups at a time here).
        relevant_rel = {
            rel for rel, g in zip(self.state.stats_all_rel, self.state.stats_group_vec)
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

        freq_results_view    = _view_results(self.state.freq_results)
        freq_df_view         = _view_samples(self.state.freq_df)
        counts_results_view  = _view_results(self.state.counts_results)
        counts_df_view       = _view_samples(self.state.counts_df)
        mfi_results_view     = _view_results(self.state.mfi_results)
        mfi_df_view          = _view_samples(self.state.mfi_df)
        mfi_sample_df_view   = _view_samples(self.state.mfi_sample_df)

        if freq_results_view is not None:
            try:
                fig = self._make_heatmap_figure(
                    freq_results_view,
                    freq_df_view,
                    title=f"Significantly Different Cluster Frequencies: {name_b} vs {name_a}",
                    group_a=name_a, group_b=name_b,
                    run_label=run_label,
                )
                self._add_results_tab(fig, "Freq Heatmap", "freq_heatmap",
                         maker=self._make_heatmap_figure,
                         maker_kwargs=dict(
                             results_df=freq_results_view,
                             sample_df=freq_df_view,
                             title=f"Significantly Different Cluster Frequencies: {name_b} vs {name_a}",
                             group_a=name_a, group_b=name_b,
                             run_label=run_label,
                         ),
                         key="freq_heatmap")
            except Exception as e:
                tab = QLabel(f"Heatmap error: {e}")
                tab.setStyleSheet("color: red;")
                tab.setProperty('_tab_key', 'freq_heatmap')
                self._results_tabs.addTab(tab, "Freq Heatmap")

            try:
                fig_v = self._make_volcano_figure(
                    freq_results_view,
                    title=f"Cluster Frequency Volcano: {name_b} vs {name_a}",
                    pval_threshold=self.pval_spin.value(),
                    fc_threshold=self.fc_spin.value(),
                    run_label=run_label,
                )
                self._add_results_tab(fig_v, "Freq Volcano", "freq_volcano",
                         maker=self._make_volcano_figure,
                         maker_kwargs=dict(
                             results_df=freq_results_view,
                             title=f"Cluster Frequency Volcano: {name_b} vs {name_a}",
                             pval_threshold=self.pval_spin.value(),
                             fc_threshold=self.fc_spin.value(),
                             run_label=run_label,
                         ),
                         key="freq_volcano")
            except Exception as e:
                tab = QLabel(f"Volcano error: {e}")
                tab.setStyleSheet("color: red;")
                tab.setProperty('_tab_key', 'freq_volcano')
                self._results_tabs.addTab(tab, "Freq Volcano")

        if counts_results_view is not None:
            try:
                fig_c = self._make_heatmap_figure(
                    counts_results_view,
                    counts_df_view,
                    title=f"Cluster Counts: {name_b} vs {name_a}",
                    group_a=name_a, group_b=name_b,
                    run_label=run_label,
                )
                self._add_results_tab(fig_c, "Counts Heatmap", "counts_heatmap",
                         maker=self._make_heatmap_figure,
                         maker_kwargs=dict(
                             results_df=counts_results_view,
                             sample_df=counts_df_view,
                             title=f"Cluster Counts: {name_b} vs {name_a}",
                             group_a=name_a, group_b=name_b,
                             run_label=run_label,
                         ),
                         key="counts_heatmap")
            except Exception as e:
                tab = QLabel(f"Counts heatmap error: {e}")
                tab.setStyleSheet("color: red;")
                tab.setProperty('_tab_key', 'counts_heatmap')
                self._results_tabs.addTab(tab, "Counts Heatmap")

            try:
                fig_cv = self._make_volcano_figure(
                    counts_results_view,
                    title=f"Cluster Counts Volcano: {name_b} vs {name_a}",
                    pval_threshold=self.pval_spin.value(),
                    fc_threshold=self.fc_spin.value(),
                    run_label=run_label,
                )
                self._add_results_tab(fig_cv, "Counts Volcano", "counts_volcano",
                         maker=self._make_volcano_figure,
                         maker_kwargs=dict(
                             results_df=counts_results_view,
                             title=f"Cluster Counts Volcano: {name_b} vs {name_a}",
                             pval_threshold=self.pval_spin.value(),
                             fc_threshold=self.fc_spin.value(),
                             run_label=run_label,
                         ),
                         key="counts_volcano")
            except Exception as e:
                tab = QLabel(f"Counts volcano error: {e}")
                tab.setStyleSheet("color: red;")
                tab.setProperty('_tab_key', 'counts_volcano')
                self._results_tabs.addTab(tab, "Counts Volcano")

        if mfi_sample_df_view is not None:
            try:
                fig_m = self._make_sample_mfi_heatmap_figure(
                    mfi_sample_df_view,
                    group_a=name_a, group_b=name_b,
                    run_label=run_label,
                )
                self._add_results_tab(fig_m, "MFI Heatmap", "mfi_heatmap",
                         maker=self._make_sample_mfi_heatmap_figure,
                         maker_kwargs=dict(
                             mfi_sample_df=mfi_sample_df_view,
                             group_a=name_a, group_b=name_b,
                             run_label=run_label,
                         ),
                         key="mfi_heatmap")
            except Exception as e:
                tab = QLabel(f"MFI heatmap error: {e}")
                tab.setStyleSheet("color: red;")
                tab.setProperty('_tab_key', 'mfi_heatmap')
                self._results_tabs.addTab(tab, "MFI Heatmap")

            try:
                fig_mv = self._make_volcano_figure(
                    mfi_results_view,
                    title=f"Cluster MFI Volcano: {name_b} vs {name_a}",
                    pval_threshold=self.pval_spin.value(),
                    fc_threshold=self.fc_spin.value(),
                    run_label=run_label,
                )
                self._add_results_tab(fig_mv, "MFI Volcano", "mfi_volcano",
                         maker=self._make_volcano_figure,
                         maker_kwargs=dict(
                             results_df=mfi_results_view,
                             title=f"Cluster MFI Volcano: {name_b} vs {name_a}",
                             pval_threshold=self.pval_spin.value(),
                             fc_threshold=self.fc_spin.value(),
                             run_label=run_label,
                         ),
                         key="mfi_volcano")
            except Exception as e:
                tab = QLabel(f"MFI volcano error: {e}")
                tab.setStyleSheet("color: red;")
                tab.setProperty('_tab_key', 'mfi_volcano')
                self._results_tabs.addTab(tab, "MFI Volcano")

    @staticmethod
    def _stamp_run_label(fig, run_label: str):
        """Stamp the source run's label in the upper-left corner of a
        results figure, so a popped-out or exported plot stays traceable
        to the run it came from."""
        if not run_label:
            return
        fig.text(
            0.01, 0.99, run_label,
            ha='left', va='top', fontsize=8, color='#555555',
            transform=fig.transFigure,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='#cccccc', alpha=0.85),
        )

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

        if mfi_sample_df is None or mfi_sample_df.empty:
            fig = Figure(figsize=(5, 2), constrained_layout=True)
            ax = fig.add_subplot(111)
            ax.axis('off')
            ax.text(0.5, 0.5, 'No data to display',
                    ha='center', va='center', fontsize=10, transform=ax.transAxes)
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
            filled = np.nan_to_num(data, nan=0.0)
            if filled.shape[0] < 2:
                return list(range(filled.shape[0]))
            try:
                Z = linkage(pdist(filled, metric='euclidean'), method='ward')
                return dendrogram(Z, no_plot=True)['leaves']
            except Exception:
                return list(range(filled.shape[0]))

        row_order = _hclust(mat.T)
        col_order = _hclust(mat)

        mat_ord = mat[:, row_order][col_order, :].T   # (n_features, n_samples)
        feat_labels = [feat_labels_raw[i] for i in row_order]
        samp_labels = [sample_labels[i] for i in col_order]
        grp_ord = [rel_to_group.get(sample_labels[i], name_a) for i in col_order]

        # Same antigen display-label remap the per-cluster MFI heatmap uses.
        from honeychrome.controller_components.functions import build_display_label_map
        unmixed_pnn = self.controller.experiment.settings.get('unmixed', {}).get(
            'event_channels_pnn') or []
        spectral_model = self.controller.experiment.process.get('spectral_model') or []
        disp_map = build_display_label_map(unmixed_pnn, spectral_model)
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
        if n_samples > 1:
            try:
                Zc = linkage(pdist(np.nan_to_num(mat, nan=0.0), metric='euclidean'), method='ward')
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
        if n_features > 1:
            try:
                Zr = linkage(pdist(np.nan_to_num(mat.T, nan=0.0), metric='euclidean'), method='ward')
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

        self._stamp_run_label(fig, run_label)
        return fig

    def _make_heatmap_figure(self, results_df, sample_df, title: str,
                             group_a: str = '', group_b: str = '', run_label: str = ''):
        """
        Per-sample heatmap with row + column dendrograms.

        results_df : limma/GLM output for ONE comparison (feature, logFC,
                     significant, …) — Item 13 phase 2: the caller has
                     already filtered a multi-comparison result down to a
                     single comparison's rows before calling this.
        sample_df  : raw feature matrix for that SAME comparison's samples
                     only (rows=samples, cols=features), index=rel-paths.
        group_a/group_b : the two group names this comparison represents
                     (Item 13 phase 2 — passed explicitly rather than read
                     from state.compare_group_a/b, since that pair now
                     belongs to T-REX and may be completely unrelated to
                     whichever comparison is being drawn here).
        run_label  : stamped in the upper-left corner (plot provenance)
        """
        from matplotlib.figure import Figure
        from matplotlib.gridspec import GridSpec
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import pdist

        name_a = group_a
        name_b = group_b
        COLOR_A = '#4477AA'
        COLOR_B = '#EE6677'

        # ---- Select significant features ----
        sig = results_df[results_df['significant']].copy() if 'significant' in results_df.columns else pd.DataFrame()
        if sig.empty:
            fig = Figure(figsize=(5, 2), constrained_layout=True)
            ax  = fig.add_subplot(111)
            ax.axis('off')
            ax.text(0.5, 0.5, 'No significant features at current thresholds',
                    ha='center', va='center', fontsize=10, transform=ax.transAxes)
            ax.set_title(title, fontsize=10)
            self._stamp_run_label(fig, run_label)
            return fig

        sig_features = sig['feature'].tolist()

        # Subset sample_df to significant feature columns only
        cols_present = [c for c in sig_features if c in sample_df.columns]
        if not cols_present:
            fig = Figure(figsize=(5, 2), constrained_layout=True)
            ax  = fig.add_subplot(111)
            ax.axis('off')
            ax.text(0.5, 0.5, 'Feature columns not found in sample matrix',
                    ha='center', va='center', fontsize=10, transform=ax.transAxes)
            ax.set_title(title, fontsize=10)
            self._stamp_run_label(fig, run_label)
            return fig

        mat = sample_df[cols_present].values.astype(float)   # (n_samples, n_sig_features)
        sample_labels = list(sample_df.index)

        # Build rel-path → group-slot lookup so row order doesn't matter.
        rel_to_group: dict[str, str] = {
            rel: grp
            for rel, grp in zip(self.state.stats_all_rel, self.state.stats_group_vec)
        }

        n_samples, n_features = mat.shape

        # ---- Hierarchical clustering ----
        def _hclust(data, metric='euclidean', method='ward'):
            if data.shape[0] < 2:
                return list(range(data.shape[0]))
            try:
                Z    = linkage(pdist(data, metric=metric), method=method)
                dend = dendrogram(Z, no_plot=True)
                return dend['leaves']
            except Exception:
                return list(range(data.shape[0]))

        row_order = _hclust(mat.T)              # cluster features (rows of heatmap)
        col_order = _hclust(mat)                # cluster samples  (cols of heatmap)

        mat_ord     = mat[:, row_order][col_order, :].T   # (n_features, n_samples)
        feat_labels = [cols_present[i] for i in row_order]
        samp_labels = [sample_labels[i] for i in col_order]
        grp_ord     = [rel_to_group.get(sample_labels[i], name_a) for i in col_order]

        # Remap MFI feature labels: replace channel suffix with antigen display label.
        if 'MFI' in title:
            from honeychrome.controller_components.functions import build_display_label_map
            unmixed_pnn = self.controller.experiment.settings.get('unmixed', {}).get(
                'event_channels_pnn') or []
            spectral_model = self.controller.experiment.process.get('spectral_model') or []
            disp_map = build_display_label_map(unmixed_pnn, spectral_model)
            # Each feature is "{cluster_label}_{channel_name}"; channel_name may itself
            # contain underscores, so match against the known channel set.
            known_channels = set(disp_map.keys())
            def _remap_feat(feat: str) -> str:
                for ch in known_channels:
                    suffix = f'_{ch}'
                    if feat.endswith(suffix):
                        cluster_part = feat[: -len(suffix)]
                        return f'{cluster_part}_{disp_map[ch]}'
                return feat
            feat_labels = [_remap_feat(f) for f in feat_labels]

        # ---- Figure sizing ----
        col_w   = 0.55          # inches per sample column
        row_h   = 0.30          # inches per feature row
        dend_h  = 1.2           # column dendrogram height
        grp_h   = 0.18          # group colour bar height
        xlabel_h = 0.8
        label_w = max(len(f) for f in feat_labels) * 0.07 + 0.3   # y-tick space
        cbar_w  = 0.5
        fig_w   = max(5.0, n_samples * col_w + label_w + cbar_w + 1.2)
        fig_h   = max(4.0, n_features * row_h + dend_h + grp_h + xlabel_h + 1.5)

        xlabel_h = 0.8          # extra room for rotated x-tick labels

        fig = Figure(figsize=(fig_w, fig_h), layout='constrained')

        gs = GridSpec(
            4, 3,
            figure=fig,
            height_ratios=[dend_h, grp_h, n_features * row_h, xlabel_h],
            width_ratios=[label_w, n_samples * col_w, cbar_w],
            hspace=0.02,
            wspace=0.02,
        )

        # ---- Column dendrogram (top-centre) ----
        ax_cdend = fig.add_subplot(gs[0, 1])
        ax_cdend.axis('off')
        if n_samples > 1:
            try:
                Zc = linkage(pdist(mat, metric='euclidean'), method='ward')
                dendrogram(Zc, ax=ax_cdend, color_threshold=0,
                           above_threshold_color='#555555',
                           link_color_func=lambda _: '#555555',
                           no_labels=True)
                ax_cdend.set_xlim(-0.5, n_samples * 10 - 0.5)
            except Exception:
                pass
        ax_cdend.set_title(title, fontsize=10, pad=4)

        # ---- Group colour bar (middle-centre) ----
        ax_grp = fig.add_subplot(gs[1, 1])
        ax_grp.set_xlim(0, n_samples)
        ax_grp.set_ylim(0, 1)
        ax_grp.axis('off')
        for xi, grp in enumerate(grp_ord):
            color = COLOR_A if grp == name_a else COLOR_B
            ax_grp.add_patch(
                __import__('matplotlib.patches', fromlist=['Rectangle']).Rectangle(
                    (xi, 0), 1, 1, color=color, transform=ax_grp.transData
                )
            )
        # Group legend patches
        import matplotlib.patches as mpatches
        handles = [
            mpatches.Patch(color=COLOR_A, label=name_a),
            mpatches.Patch(color=COLOR_B, label=name_b),
        ]
        ax_cdend.legend(handles=handles, loc='lower right', fontsize=7,
                        frameon=True, ncol=2)

        # ---- Row dendrogram (main-left) ----
        ax_rdend = fig.add_subplot(gs[2, 0])
        ax_rdend.axis('off')
        if n_features > 1:
            try:
                Zr = linkage(pdist(mat.T, metric='euclidean'), method='ward')
                dendrogram(Zr, ax=ax_rdend, orientation='left',
                           color_threshold=0,
                           above_threshold_color='#555555',
                           link_color_func=lambda _: '#555555',
                           no_labels=True)
            except Exception:
                pass

        # ---- Heatmap (main-centre) ----
        ax_hm = fig.add_subplot(gs[2, 1])
        # Raw per-sample values (log1p-MFI / % / counts) — always non-negative,
        # never a delta, so normalize from the data's own range rather than
        # forcing a zero-centered diverging scale (that's what was washing
        # out low-value samples like the reference group).
        vmin = float(np.percentile(mat_ord, 5))
        vmax = max(float(np.percentile(mat_ord, 95)), vmin + 0.01)
        im = ax_hm.imshow(
            mat_ord,
            aspect='auto',
            cmap='viridis',
            vmin=vmin, vmax=vmax,
            interpolation='nearest',
        )
        ax_hm.set_xticks(range(n_samples))
        ax_hm.set_xticklabels(
            [Path(s).stem for s in samp_labels],
            rotation=45, ha='right', fontsize=7,
        )
        ax_hm.set_yticks(range(n_features))
        ax_hm.set_yticklabels(feat_labels, fontsize=7)
        ax_hm.yaxis.set_label_position('right')
        ax_hm.yaxis.tick_right()
        ax_hm.grid(False)   # suppress inherited seaborn 'whitegrid' (draws through tick/cell centres)
        ax_hm.set_xticks(np.arange(-0.5, n_samples, 1), minor=True)
        ax_hm.set_yticks(np.arange(-0.5, n_features, 1), minor=True)
        ax_hm.grid(which='minor', color='white', linestyle='-', linewidth=0.6)
        ax_hm.tick_params(which='minor', bottom=False, left=False, right=False)

        # ---- Colorbar (main-right) ----
        ax_cb = fig.add_subplot(gs[2, 2])
        cb = fig.colorbar(im, cax=ax_cb)
        cb.ax.tick_params(labelsize=7)
        if 'MFI' in title:
            cb_label = 'log1p-MFI'
        elif 'Counts' in title:
            cb_label = 'Event count'
        else:
            cb_label = '% frequency'
        cb.set_label(cb_label, fontsize=7)

        self._stamp_run_label(fig, run_label)
        return fig

    def _make_volcano_figure(self, results_df, title: str,
                              pval_threshold: float, fc_threshold: float,
                              run_label: str = ''):
        """Volcano plot: x=logFC, y=-log10(adj.P.Val), coloured by significance.
        run_label is stamped in the upper-left corner (plot provenance)."""
        from matplotlib.figure import Figure

        pval_col = 'adj.P.Val' if 'adj.P.Val' in results_df.columns else 'P.Value'
        logfc  = results_df['logFC'].values.astype(float)
        neg_lp = -np.log10(np.maximum(results_df[pval_col].values.astype(float), 1e-300))
        sig    = results_df['significant'].values if 'significant' in results_df.columns else (
            (results_df['P.Value'] <= pval_threshold) & (results_df['logFC'].abs() >= fc_threshold)
        ).values
        features = results_df['feature'].tolist()

        fig = Figure(figsize=(5, 5), constrained_layout=True)
        ax  = fig.add_subplot(111)

        colors = ['#e74c3c' if s else '#aaaaaa' for s in sig]
        ax.scatter(logfc, neg_lp, c=colors, s=25, alpha=0.8, linewidths=0)

        ax.axhline(-np.log10(pval_threshold), color='grey', linestyle='--', linewidth=0.8)
        ax.axvline( fc_threshold,              color='grey', linestyle='--', linewidth=0.8)
        ax.axvline(-fc_threshold,              color='grey', linestyle='--', linewidth=0.8)

        # Label significant features
        for fc_val, nlp_val, label, is_sig in zip(logfc, neg_lp, features, sig):
            if is_sig:
                ax.annotate(label, (fc_val, nlp_val), fontsize=6,
                            textcoords='offset points', xytext=(3, 3))

        x_lim = max(np.abs(logfc).max() * 1.05, fc_threshold * 1.5)
        ax.set_xlim(-x_lim, x_lim)

        ax.set_xlabel("log2 Fold Change")
        ax.set_ylabel("-log10(adj. P-value)")
        ax.set_title(title, fontsize=10)
        self._stamp_run_label(fig, run_label)
        return fig

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

    def _run_trex(self):
        """Delegate T-REX computation to PluginWidget which owns the data loaders."""
        # Walk up to PluginWidget
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, 'run_trex'):
                parent.run_trex()
                return
            parent = parent.parent()
        QMessageBox.warning(self, "T-REX", "Could not find parent PluginWidget.")

    def _export_results_csv(self):
        """Export all statistics results to a combined CSV."""
        """Export all statistics results to a combined CSV."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Statistics Results", "", "CSV files (*.csv)"
        )
        if not path:
            return
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


class WorkspaceTab(QWidget):
    """
    Tab 3 — Workspace  (Stage 5 + 7 complete)
    -------------------------------------------
    Scrollable canvas of PlotCard widgets showing DR scatter plots.
    Each PlotCard supports:
      • DR algorithm selector
      • Sample selector (single sample or all pooled)
      • Colour mode: Clusters | Marker intensity | T-REX
      • Per-cluster colour pickers (right-click)
      • Magic-wand display-config copy/paste
      • PNG/PDF export

    Stage 7: T-REX colouring, marker intensity, magic-wand, PDF export.
    """

    plots_changed = Signal()  # emitted whenever a card is added or removed

    def __init__(self, state: PipelineState, bus, controller, parent=None):
        super().__init__(parent)
        self.state = state
        self.bus = bus
        self.controller = controller
        self._plot_cards: list['PlotCard'] = []
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # --- Toolbar ---
        toolbar = QFrame()
        toolbar.setFrameShape(QFrame.StyledPanel)
        toolbar.setMaximumHeight(44)
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(8, 4, 8, 4)
        tb_layout.setSpacing(10)

        tb_layout.addWidget(QLabel("Columns:"))
        self.col_combo = QComboBox()
        self.col_combo.addItems(["2", "3", "4"])
        self.col_combo.setCurrentText(str(self.state.workspace_n_columns))
        self.col_combo.setFixedWidth(55)
        self.col_combo.currentTextChanged.connect(self._on_column_count_changed)
        tb_layout.addWidget(self.col_combo)

        tb_layout.addStretch()

        self.add_plot_btn = QPushButton("＋ Add Plot")
        self.add_plot_btn.setFixedHeight(30)
        self.add_plot_btn.clicked.connect(self.add_plot)
        tb_layout.addWidget(self.add_plot_btn)

        self.export_pdf_btn = QPushButton("Export PDF")
        self.export_pdf_btn.setFixedHeight(30)
        self.export_pdf_btn.setToolTip("Export all plots as a multi-page PDF.")
        self.export_pdf_btn.clicked.connect(self._export_pdf)
        tb_layout.addWidget(self.export_pdf_btn)

        outer.addWidget(toolbar)

        # --- Scrollable canvas ---
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.canvas = QWidget()
        self.canvas_layout = QGridLayout(self.canvas)
        self.canvas_layout.setSpacing(6)
        self.canvas_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll.setWidget(self.canvas)
        outer.addWidget(self.scroll)

        # Empty-canvas hint
        self._placeholder = QLabel(
            "No plots yet.\n\nClick  ＋ Add Plot  to create a DR scatter plot."
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet("color: grey; font-style: italic;")
        self.canvas_layout.addWidget(self._placeholder, 0, 0, 1, 4)

    def add_plot(self, refresh: bool = True):
        """
        Add a new PlotCard to the canvas.

        refresh=False skips the initial draw — used when the caller is about
        to call apply_config() immediately afterwards (e.g. restoring a
        persisted layout), so the card doesn't render once with blank/default
        combo values and then immediately again with the real ones.
        """
        if self._placeholder is not None:
            self._placeholder.setParent(None)
            self._placeholder = None

        plot_id = f"plot_{len(self._plot_cards) + 1}"
        card = PlotCard(
            plot_id=plot_id,
            state=self.state,
            bus=self.bus,
            controller=self.controller,
            workspace=self,
            parent=self.canvas,
        )
        self._plot_cards.append(card)
        self.state.plot_configs.append({'plot_id': plot_id})
        # Save display config whenever a card changes (via each card's _schedule_refresh)

        n_cols = self.state.workspace_n_columns
        idx = len(self._plot_cards) - 1
        row, col = divmod(idx, n_cols)
        self.canvas_layout.addWidget(card, row, col)
        if refresh:
            card.refresh()
        self.plots_changed.emit()

    def remove_card(self, card: 'PlotCard'):
        """Remove a PlotCard from the canvas and state."""
        if card in self._plot_cards:
            self._plot_cards.remove(card)
        self.state.plot_configs = [
            c for c in self.state.plot_configs if c.get('plot_id') != card.plot_id
        ]
        self.canvas_layout.removeWidget(card)
        card.deleteLater()
        self._relayout()
        if not self._plot_cards:
            self._placeholder = QLabel(
                "No plots yet.\n\nClick  ＋ Add Plot  to create a DR scatter plot."
            )
            self._placeholder.setAlignment(Qt.AlignCenter)
            self._placeholder.setStyleSheet("color: grey; font-style: italic;")
            self.canvas_layout.addWidget(self._placeholder, 0, 0, 1, 4)
        self.plots_changed.emit()

    def _on_column_count_changed(self, text: str):
        try:
            self.state.workspace_n_columns = int(text)
        except ValueError:
            pass
        self._relayout()

    def _relayout(self):
        """Re-arrange all PlotCards after a column-count change."""
        for card in self._plot_cards:
            self.canvas_layout.removeWidget(card)
        n_cols = self.state.workspace_n_columns
        for idx, card in enumerate(self._plot_cards):
            row, col = divmod(idx, n_cols)
            self.canvas_layout.addWidget(card, row, col)

    def _export_pdf(self):
        """Export all PlotCards to a multi-page PDF."""
        if not self._plot_cards:
            QMessageBox.information(self, "Export PDF", "No plots to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Workspace PDF", "", "PDF files (*.pdf)"
        )
        if not path:
            return
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            with PdfPages(path) as pdf:
                for card in self._plot_cards:
                    fig = card.get_figure()
                    if fig is not None:
                        pdf.savefig(fig, bbox_inches='tight')
            QMessageBox.information(self, "Exported", f"Saved to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    def refresh(self):
        """Re-render all PlotCards from current PipelineState."""
        for card in self._plot_cards:
            card.refresh()


class PlotCard(QFrame):
    """
    A single DR scatter plot on the workspace canvas.  (Stage 5 + 7)

    Controls (top toolbar):
      • DR algorithm selector
      • Sample selector (All Samples pooled, or individual)
      • Colour mode: Clusters | Marker | T-REX
      • Marker channel selector (visible in Marker mode)
      • Magic wand (copy) and paste buttons
      • Close button

    Right-click on a cluster legend colour swatch → colour picker.

    Parameters
    ----------
    plot_id    Unique string identifier stored in state.plot_configs.
    state      Shared PipelineState.
    bus        Honeychrome EventBus.
    controller Main Honeychrome Controller.
    workspace  Parent WorkspaceTab (for remove_card / magic-wand).
    """

    _COLOUR_MODES = ['Clusters', 'Marker', 'T-REX']

    def __init__(self, plot_id: str, state: PipelineState, bus, controller,
                 workspace: 'WorkspaceTab', parent=None):
        super().__init__(parent)
        self.plot_id    = plot_id
        self.state      = state
        self.bus        = bus
        self.controller = controller
        self.workspace  = workspace
        self._figure    = None   # current matplotlib Figure

        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setMinimumSize(300, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # ---- Toolbar row 1: DR run | sample | colour mode | overlay ----
        row1 = QHBoxLayout()
        row1.setSpacing(4)

        row1.addWidget(QLabel("DR run:"))
        self.dr_combo = QComboBox()
        self.dr_combo.setFixedWidth(120)
        self.dr_combo.setToolTip("Archived DR run to display")
        self.dr_combo.currentIndexChanged.connect(self._on_dr_run_changed)
        row1.addWidget(self.dr_combo)

        row1.addWidget(QLabel("Sample:"))
        self.sample_combo = QComboBox()
        self.sample_combo.setFixedWidth(130)
        self.sample_combo.setToolTip("Sample to display (or all pooled)")
        self.sample_combo.currentTextChanged.connect(self._schedule_refresh)
        row1.addWidget(self.sample_combo)

        row1.addWidget(QLabel("Colour:"))
        self.colour_mode_combo = QComboBox()
        self.colour_mode_combo.addItems(self._COLOUR_MODES)
        self.colour_mode_combo.setFixedWidth(85)
        self.colour_mode_combo.currentTextChanged.connect(self._on_colour_mode_changed)
        row1.addWidget(self.colour_mode_combo)

        self.marker_combo = QComboBox()
        self.marker_combo.setFixedWidth(110)
        self.marker_combo.setToolTip("Channel to colour by (Marker mode)")
        self.marker_combo.currentTextChanged.connect(self._schedule_refresh)
        self.marker_combo.setVisible(False)
        row1.addWidget(self.marker_combo)

        # Clustering-run overlay selector — only meaningful in Clusters mode
        # (Item 6: each card independently picks which archived clustering
        # run's labels to overlay on the chosen DR run's embedding).
        self.cluster_run_label = QLabel("Overlay:")
        self.cluster_run_label.setVisible(False)
        row1.addWidget(self.cluster_run_label)
        self.cluster_run_combo = QComboBox()
        self.cluster_run_combo.setFixedWidth(120)
        self.cluster_run_combo.setToolTip("Archived clustering run to overlay")
        self.cluster_run_combo.currentIndexChanged.connect(self._on_cluster_run_changed)
        self.cluster_run_combo.setVisible(False)
        row1.addWidget(self.cluster_run_combo)

        # Compatibility warning — shown when the selected DR run's and
        # clustering run's gate/sample sets don't line up.
        self._compat_warning = QLabel("⚠")
        self._compat_warning.setStyleSheet("color: #d9822b; font-weight: bold;")
        self._compat_warning.setVisible(False)
        row1.addWidget(self._compat_warning)

        row1.addStretch()

        # Magic wand (copy) button
        self.wand_btn = QPushButton("🪄")
        self.wand_btn.setFixedSize(26, 26)
        self.wand_btn.setToolTip("Copy display config (magic wand)")
        self.wand_btn.clicked.connect(self._copy_display_config)
        row1.addWidget(self.wand_btn)

        # Paste button
        self.paste_btn = QPushButton("📋")
        self.paste_btn.setFixedSize(26, 26)
        self.paste_btn.setToolTip("Paste display config from magic wand")
        self.paste_btn.clicked.connect(self._paste_display_config)
        row1.addWidget(self.paste_btn)

        # Export PNG
        self.png_btn = QPushButton("PNG")
        self.png_btn.setFixedSize(36, 26)
        self.png_btn.setToolTip("Export this plot as PNG")
        self.png_btn.clicked.connect(self._export_png)
        row1.addWidget(self.png_btn)

        # Close
        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(26, 26)
        self.close_btn.setToolTip("Remove this plot")
        self.close_btn.clicked.connect(lambda: self.workspace.remove_card(self))
        row1.addWidget(self.close_btn)

        layout.addLayout(row1)

        # ---- Appearance controls row ----
        self._appearance_box = QGroupBox("Appearance")
        self._appearance_box.setCheckable(True)
        self._appearance_box.setChecked(False)   # collapsed by default
        self._appearance_box.toggled.connect(
            lambda on: self._appearance_inner.setVisible(on))
        app_outer = QVBoxLayout(self._appearance_box)
        app_outer.setContentsMargins(4, 2, 4, 4)
        app_outer.setSpacing(2)

        self._appearance_inner = QWidget()
        app_inner_layout = QHBoxLayout(self._appearance_inner)
        app_inner_layout.setContentsMargins(0, 0, 0, 0)
        app_inner_layout.setSpacing(8)

        # Gridlines
        self._show_grid = QCheckBox("Grid")
        self._show_grid.setChecked(False)
        self._show_grid.stateChanged.connect(self._schedule_refresh)
        app_inner_layout.addWidget(self._show_grid)

        # Axis tick labels
        self._show_ticks = QCheckBox("Tick labels")
        self._show_ticks.setChecked(True)
        self._show_ticks.stateChanged.connect(self._schedule_refresh)
        app_inner_layout.addWidget(self._show_ticks)

        # Axis title labels
        self._show_axis_labels = QCheckBox("Axis labels")
        self._show_axis_labels.setChecked(True)
        self._show_axis_labels.stateChanged.connect(self._schedule_refresh)
        app_inner_layout.addWidget(self._show_axis_labels)

        app_inner_layout.addWidget(QLabel("Title font:"))
        self._title_font_spin = QSpinBox()
        self._title_font_spin.setRange(6, 24)
        self._title_font_spin.setValue(8)
        self._title_font_spin.setFixedWidth(48)
        self._title_font_spin.valueChanged.connect(self._schedule_refresh)
        app_inner_layout.addWidget(self._title_font_spin)

        app_inner_layout.addWidget(QLabel("Axis font:"))
        self._axis_font_spin = QSpinBox()
        self._axis_font_spin.setRange(5, 20)
        self._axis_font_spin.setValue(7)
        self._axis_font_spin.setFixedWidth(48)
        self._axis_font_spin.valueChanged.connect(self._schedule_refresh)
        app_inner_layout.addWidget(self._axis_font_spin)

        from PySide6.QtGui import QPalette
        from PySide6.QtWidgets import QApplication
        _palette = QApplication.instance().palette()
        _default_label_color = (
            'white' if _palette.color(QPalette.ColorRole.Base).value() < 128 else 'black'
        )
        self._label_color = _default_label_color
        self._label_color_btn = QPushButton("Label colour")
        self._label_color_btn.setFixedHeight(22)
        self._label_color_btn.setToolTip("Colour for colourbar and title text")
        self._label_color_btn.clicked.connect(self._pick_label_color)
        app_inner_layout.addWidget(self._label_color_btn)
        self._update_label_color_btn()

        app_inner_layout.addStretch()
        app_outer_layout_inner = QVBoxLayout()
        app_outer_layout_inner.addWidget(self._appearance_inner)
        self._appearance_inner.setVisible(False)
        app_outer.addWidget(self._appearance_inner)

        layout.addWidget(self._appearance_box)

        # ---- Matplotlib canvas (square container enforces equal width/height) ----
        self._figure = Figure(figsize=(4, 4), constrained_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas_container = _SquareWidget()
        sq_layout = QVBoxLayout(self._canvas_container)
        sq_layout.setContentsMargins(0, 0, 0, 0)
        sq_layout.addWidget(self._canvas)
        self._canvas_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(self._canvas_container, stretch=1)

        # ---- Plot area: square canvas + right legend panel ----
        plot_row = QHBoxLayout()
        plot_row.setSpacing(4)
        plot_row.addWidget(self._canvas_container, stretch=1)

        # Right-side legend: scrollable list of cluster swatches
        legend_scroll = QScrollArea()
        legend_scroll.setWidgetResizable(True)
        legend_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        legend_scroll.setFixedWidth(120)
        legend_scroll.setFrameShape(QFrame.NoFrame)
        self._legend_widget = QWidget()
        self._legend_layout = QVBoxLayout(self._legend_widget)
        self._legend_layout.setContentsMargins(2, 2, 2, 2)
        self._legend_layout.setSpacing(2)
        self._legend_layout.setAlignment(Qt.AlignTop)
        legend_scroll.setWidget(self._legend_widget)
        self._legend_scroll = legend_scroll
        plot_row.addWidget(legend_scroll)

        layout.addLayout(plot_row, stretch=1)

        # Populate selectors
        self._populate_selectors()

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

        self.marker_combo.blockSignals(True)
        self.marker_combo.clear()
        labels = _antigen_dash_labels(self.controller)
        for ch in self.state.selected_channels:
            self.marker_combo.addItem(labels.get(ch, ch), ch)
        self.marker_combo.blockSignals(False)

        self._update_compatibility_warning()
        self._schedule_refresh()

    def _populate_dr_run_combo(self):
        """Rebuild the DR-run combo from state.dr_runs, keyed by run_id —
        never label text, since labels are user-editable (Item 6 rename)."""
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
        state.clustering_runs, keyed by run_id."""
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
        self._update_compatibility_warning()
        self._schedule_refresh()

    def _on_cluster_run_changed(self, _index: int):
        self._update_compatibility_warning()
        self._schedule_refresh()

    def _update_compatibility_warning(self):
        """
        Show the ⚠ indicator when the selected clustering run's gate-set
        or sample-set isn't a subset of the selected DR run's. Delegates
        the set comparison to drc_scatter (Item 8 — shared with the
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
        self.marker_combo.setVisible(mode == 'Marker')
        show_overlay = mode == 'Clusters'
        self.cluster_run_label.setVisible(show_overlay)
        self.cluster_run_combo.setVisible(show_overlay)
        self._update_compatibility_warning()
        self._schedule_refresh()

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
        (and, in Clusters mode, the currently-selected clustering run)."""
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
            lab = labels_dict.get(rel_path)
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
                lbl = labels_dict.get(rel)
                if lbl is None:
                    _log.debug("workspace: no cluster labels for %s — greyed", rel)
                    labs.append(np.full(n, -1, dtype=np.int32))
                elif len(lbl) != n:
                    _log.warning("workspace: label/embedding length mismatch for %s "
                                 "(%d labels vs %d points) — greyed",
                                 rel, len(lbl), n)
                    labs.append(np.full(n, -1, dtype=np.int32))
                else:
                    labs.append(lbl)
            if not xys:
                self._show_placeholder("No embeddings found.")
                return
            xy = np.concatenate(xys, axis=0)
            lab = np.concatenate(labs, axis=0)
            origin = np.concatenate(origins, axis=0)

        colour_mode = self.colour_mode_combo.currentText()

        from PySide6.QtGui import QPalette
        from PySide6.QtWidgets import QApplication
        _palette = QApplication.instance().palette()
        is_dark = _palette.color(QPalette.ColorRole.Base).value() < 128

        self._figure.clear()
        if is_dark:
            self._figure.patch.set_facecolor('#1e1e1e')
            ax = self._figure.add_subplot(111)
            ax.set_facecolor('#2b2b2b')
            _fg = 'white'
            ax.tick_params(colors=_fg)
            ax.xaxis.label.set_color(_fg)
            ax.yaxis.label.set_color(_fg)
            for spine in ax.spines.values():
                spine.set_edgecolor(_fg)
        else:
            ax = self._figure.add_subplot(111)
        ax.set_aspect('equal', adjustable='datalim')
        _tf = self._title_font_spin.value()
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

        if colour_mode == 'Clusters':
            self._draw_cluster_scatter(ax, xy_disp, lab_disp, cl_run)
        elif colour_mode == 'Marker':
            self._draw_marker_scatter(ax, xy_disp, origin_disp,
                                      disp_idx, sample_row_offsets, run)
        elif colour_mode == 'T-REX':
            self._draw_trex_scatter(ax, xy_disp, lab_disp, emb_dict, run, origin_disp)

        # Apply label colour to colourbar (Marker / T-REX) and title.
        lc = self._label_color
        ax.title.set_color(lc)
        for cb_ax in self._figure.axes:
            if cb_ax is not ax:
                cb_ax.tick_params(colors=lc)
                cb_ax.yaxis.label.set_color(lc)

        self._canvas.draw_idle()
        self._rebuild_legend()

    def _draw_cluster_scatter(self, ax, xy, labels, cl_run: dict | None):
        """
        Colour points by cluster label. Delegates to drc_scatter (Item 8 —
        extracted so the Cluster Annotation tab's cluster-map panel renders
        identically instead of duplicating this logic).
        """
        drc_scatter.draw_cluster_scatter(ax, xy, labels, cl_run, self.controller)

    def _draw_marker_scatter(self, ax, xy, origin, disp_idx, sample_row_offsets,
                             run: dict):
        """Colour by UNTRANSFORMED marker intensity on the full data scale."""
        ch = self.marker_combo.currentData()
        if not ch or not self.state.selected_channels:
            ax.scatter(xy[:, 0], xy[:, 1], s=1, c='#aaaaaa', alpha=0.4)
            ax.set_title("Marker: no channel", fontsize=8)
            return

        # Marker values are re-gated against the CURRENT gate selection, but
        # the embedding itself was built under whatever gates were active
        # when this run was archived (run['gates']). If those have since
        # diverged, per-sample row counts won't line up with the embedding
        # and points would be silently miscoloured rather than just greyed.
        run_gates = run.get('gates')
        if run_gates is not None and sorted(run_gates) != sorted(self.state.selected_gates):
            ax.scatter(xy[:, 0], xy[:, 1], s=1, c='#cccccc', alpha=0.3)
            ax.set_title(f"{run.get('label', '')}: gate changed since training — retrain DR",
                        fontsize=8)
            _log.warning(
                "marker scatter: gate selection changed since run %r was trained "
                "(trained on %r, now %r) — embeddings are stale",
                run.get('label'), run_gates, self.state.selected_gates,
            )
            return

        values = np.full(len(xy), np.nan, dtype=np.float32)
        for rel in np.unique(origin):
            mv = drc_pipeline.load_sample_marker_values(
                self.controller, self.state, str(rel))
            if mv is None:
                continue
            vals, names = mv
            if ch not in names:
                continue
            full_col = vals[:, names.index(ch)]  # all gated events for this sample

            mask = origin == rel   # which rows in the display array belong to this sample
            if disp_idx is not None and rel in sample_row_offsets:
                # Recover which rows within this sample survived global downsampling.
                start, end = sample_row_offsets[rel]
                # The embedding's row range for this sample must match its
                # CURRENT re-gated event count exactly, or positional
                # indexing below is meaningless (not just out-of-range —
                # in-range-but-wrong is possible too, and silent). Check the
                # count up front rather than only catching an overflow
                # after the fact.
                if (end - start) != len(full_col):
                    _log.warning(
                        "marker %s: %s embedding has %d points but current "
                        "gating yields %d events — stale embedding for this "
                        "sample, skipped (retrain DR to refresh)",
                        ch, rel, end - start, len(full_col))
                    continue
                kept_global = disp_idx[mask]          # global indices of kept rows
                within_sample = kept_global - start   # row indices inside full_col
                values[mask] = full_col[within_sample]
            else:
                # No downsampling or no offset info — direct assignment
                if int(mask.sum()) == len(full_col):
                    values[mask] = full_col
                else:
                    _log.warning("marker %s: %s has %d events but %d displayed points",
                                 ch, rel, len(full_col), int(mask.sum()))

        finite = np.isfinite(values)
        if not finite.any():
            _log.warning("marker %s: no values could be loaded", ch)
            ax.scatter(xy[:, 0], xy[:, 1], s=1, c='#cccccc', alpha=0.4)
            ax.set_title(f"Marker {ch}: no data", fontsize=8)
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
                        vmin=vmin, vmax=vmax, alpha=0.6, linewidths=0)
        cb = self._figure.colorbar(sc, ax=ax, shrink=0.6)
        tick_pos, tick_lab = _log1p_powers_of_ten_ticks(float(np.expm1(vmax)))
        keep = [(p, l) for p, l in zip(tick_pos, tick_lab) if vmin <= p <= vmax]
        if keep:
            cb.set_ticks([p for p, l in keep])
            cb.set_ticklabels([l for p, l in keep])
        cb.set_label(ch, fontsize=8)
        ax.set_title(f"Marker: {ch}", fontsize=8)

    def _draw_trex_scatter(self, ax, xy, labels, emb_dict: dict, run: dict, origin):
        """Colour by T-REX score using a red-blue diverging colourmap.
        Aligned per-sample via origin (like _draw_marker_scatter) — T-REX
        only scores samples in its Compare pair, which is often a SUBSET
        of the samples in the current DR embedding, so a blanket total-
        length comparison against all pooled samples was always going to
        fail whenever other groups were also in the embedding."""
        if not self.state.trex_scores:
            ax.scatter(xy[:, 0], xy[:, 1], s=1, c='#cccccc', alpha=0.4)
            ax.set_title("T-REX: assign groups and run T-REX first", fontsize=7)
            return

        if self.state.trex_dr_run_id and run.get('run_id') != self.state.trex_dr_run_id:
            ax.scatter(xy[:, 0], xy[:, 1], s=1, c='#cccccc', alpha=0.3)
            ax.set_title("T-REX was scored against a different DR run — "
                        "switch to that run, or re-run T-REX for this one",
                        fontsize=7)
            return

        run_gates = run.get('gates')
        if run_gates is not None and sorted(run_gates) != sorted(self.state.selected_gates):
            ax.scatter(xy[:, 0], xy[:, 1], s=1, c='#cccccc', alpha=0.3)
            ax.set_title(f"{run.get('label', '')}: gate changed since training — retrain DR",
                        fontsize=7)
            _log.warning(
                "trex scatter: gate selection changed since run %r was trained "
                "(trained on %r, now %r) — embeddings are stale",
                run.get('label'), run_gates, self.state.selected_gates,
            )
            return

        scores = np.full(len(xy), np.nan, dtype=np.float32)
        for rel in np.unique(origin):
            s = self.state.trex_scores.get(str(rel))
            if s is None:
                continue          # sample wasn't part of T-REX's Compare pair
            mask = origin == rel
            if int(mask.sum()) == len(s):
                scores[mask] = s
            else:
                _log.warning(
                    "trex scatter: %s has %d T-REX scores but %d displayed "
                    "points — skipped", rel, len(s), int(mask.sum()))

        finite = np.isfinite(scores)
        if not finite.any():
            ax.scatter(xy[:, 0], xy[:, 1], s=1, c='#cccccc', alpha=0.4)
            ax.set_title("T-REX: no scored samples in current view "
                         "(check the Compare pair)", fontsize=7)
            return

        if (~finite).any():
            ax.scatter(xy[~finite, 0], xy[~finite, 1], s=1, c='#cccccc',
                      alpha=0.2, linewidths=0)
        sc = ax.scatter(
            xy[finite, 0], xy[finite, 1],
            s=1, c=scores[finite], cmap='RdBu_r', vmin=-1, vmax=1, alpha=0.5, linewidths=0
        )
        self._figure.colorbar(sc, ax=ax, shrink=0.6, label='T-REX score')
        ax.set_title("T-REX enrichment (red=A, blue=B) — grey = not in Compare pair",
                    fontsize=8)

    def _show_placeholder(self, text: str):
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.text(0.5, 0.5, text, ha='center', va='center',
                transform=ax.transAxes, color='grey', fontsize=9,
                wrap=True)
        ax.axis('off')
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Legend (cluster swatches with right-click colour picker)
    # ------------------------------------------------------------------

    def _rebuild_legend(self):
        """Rebuild the right-side cluster legend (swatch + name label per
        cluster), reading from the currently-selected clustering run's own
        'colors'/'names' (Item 6 — these live per-run, not on a single
        ambient dict)."""
        while self._legend_layout.count():
            item = self._legend_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if self.colour_mode_combo.currentText() != 'Clusters':
            self._legend_scroll.setVisible(False)
            return
        cl_run = self._selected_cluster_run()
        colors = cl_run.get('colors', {}) if cl_run else {}
        names = cl_run.get('names', {}) if cl_run else {}
        if not colors:
            self._legend_scroll.setVisible(False)
            return

        self._legend_scroll.setVisible(True)
        for lbl in sorted(colors.keys()):
            color = colors[lbl]
            name  = names.get(lbl, 'Noise' if lbl < 0 else str(lbl))

            row = QHBoxLayout()
            row.setSpacing(4)

            swatch = QPushButton()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(
                f"background-color: {color}; border: 1px solid #555; border-radius: 2px;"
            )
            swatch.setToolTip("Right-click to change colour")
            swatch.setContextMenuPolicy(Qt.CustomContextMenu)
            swatch.customContextMenuRequested.connect(
                lambda pos, l=lbl, s=swatch: self._pick_cluster_colour(l, s)
            )
            row.addWidget(swatch)

            name_lbl = QLabel(name)
            name_lbl.setStyleSheet("font-size: 9px;")
            name_lbl.setToolTip("Double-click to rename")
            name_lbl.mouseDoubleClickEvent = lambda e, l=lbl: self._rename_cluster(l)
            row.addWidget(name_lbl, stretch=1)

            row_w = QWidget()
            row_w.setLayout(row)
            self._legend_layout.addWidget(row_w)

        self._legend_layout.addStretch()

    def _pick_cluster_colour(self, label: int, swatch: QPushButton):
        """Open a colour dialog to change a cluster's colour. Persistence
        delegates to drc_scatter (Item 8 — shared with the Cluster
        Annotation tab)."""
        cl_run = self._selected_cluster_run()
        if cl_run is None:
            return
        current = cl_run.get('colors', {}).get(label, '#aaaaaa')
        colour = QColorDialog.getColor(QColor(current), self, f"Colour for cluster {label}")
        if colour.isValid():
            hex_col = colour.name()
            drc_scatter.recolor_cluster(self.controller, cl_run, label, hex_col)
            swatch.setStyleSheet(
                f"background-color: {hex_col}; border: 1px solid #555; border-radius: 2px;"
            )
            self.refresh()

    def _rename_cluster(self, label: int):
        """Inline rename for a cluster via an input dialog. Duplicate-check
        and persistence delegate to drc_scatter (Item 8 — shared with the
        Cluster Annotation tab's rename paths)."""
        from PySide6.QtWidgets import QInputDialog
        cl_run = self._selected_cluster_run()
        if cl_run is None:
            return
        names = cl_run.get('names', {})
        current = names.get(label, 'Noise' if label < 0 else str(label))
        new_name, ok = QInputDialog.getText(
            self, f"Rename cluster {label}", "New name:", text=current
        )
        if not ok or not new_name.strip():
            return
        if drc_scatter.rename_cluster(self.controller, cl_run, label, new_name.strip(), self):
            self._rebuild_legend()
    
    def _pick_label_color(self):
        """Open colour dialog to set colourbar / title text colour."""
        colour = QColorDialog.getColor(QColor(self._label_color), self, "Label colour")
        if colour.isValid():
            self._label_color = colour.name()
            self._update_label_color_btn()
            self._schedule_refresh()

    def _update_label_color_btn(self):
        """Update the button background to preview the chosen colour."""
        self._label_color_btn.setStyleSheet(
            f"background-color: {self._label_color};"
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
            'title_font':     self._title_font_spin.value(),
            'axis_font':      self._axis_font_spin.value(),
            'label_color':    self._label_color,
        }

    def apply_config(self, config: dict):
        if not config:
            return
        # Pre-Item-6 saved layouts have 'dr_algo' instead of 'dr_run_id' —
        # there's no run_id to recover from a bare algorithm name, so those
        # fall through to whatever _populate_dr_run_combo() already
        # defaulted to (the newest run) rather than raising.
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
            self._title_font_spin.setValue(int(config['title_font']))
        if 'axis_font' in config:
            self._axis_font_spin.setValue(int(config['axis_font']))
        if 'label_color' in config:
            self._label_color = config['label_color']
            self._update_label_color_btn()
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
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Plot", f"{self.plot_id}.png", "PNG files (*.png)"
        )
        if not path:
            return
        try:
            self._figure.savefig(path, dpi=150, bbox_inches='tight')
            QMessageBox.information(self, "Exported", f"Saved to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))



# ---------------------------------------------------------------------------

class _PaCMAPWrapper:
    """
    Thin wrapper around a fitted PaCMAP reducer that stores the training
    data so out-of-sample transform() calls can pass it as `basis`.

    PaCMAP.transform(X_new, basis=X_train) is required when the Annoy
    index is not saved to disk; without basis the call raises:
      'If the index is not cached, the original dataset must be provided.'

    transform() returns a plain ndarray — not a tuple.
    """
    def __init__(self, reducer, training_data: np.ndarray):
        self._reducer = reducer
        self._training_data = training_data

    def transform(self, new_data: np.ndarray) -> np.ndarray:
        return self._reducer.transform(new_data, basis=self._training_data)


# ---------------------------------------------------------------------------
# DR background worker
# ---------------------------------------------------------------------------


class _UMAPTqdmHook:
    """
    Drop-in tqdm-compatible object for UMAP's ``tqdm_kwds`` interface.

    UMAP calls ``tqdm.tqdm(range(n_epochs))`` internally and iterates over it.
    By passing an instance of this class as ``tqdm_kwds={'tqdm_class': hook}``
    we intercept each epoch update and forward it as a ``(current, total)``
    pair to a caller-supplied callback — without any tqdm dependency.

    Usage::

        hook = _UMAPTqdmHook(callback=lambda cur, tot: ...)
        umap_lib.UMAP(..., tqdm_kwds={'tqdm_class': hook}).fit(data)
    """

    def __init__(self, callback):
        self._cb = callback          # callable(current: int, total: int)

    # tqdm is instantiated as tqdm_class(iterable, **kw); we capture total.
    def __call__(self, iterable=None, total=None, **_kw):
        return self._Iter(iterable, total, self._cb)

    class _Iter:
        def __init__(self, iterable, total, cb):
            self._it   = iter(iterable) if iterable is not None else iter([])
            self._total = total or 0
            self._n    = 0
            self._cb   = cb

        def __iter__(self):
            return self

        def __next__(self):
            val = next(self._it)      # raises StopIteration when done
            self._n += 1
            self._cb(self._n, self._total)
            return val

        # tqdm interface stubs so UMAP doesn't fail on attribute access
        def update(self, n=1):
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

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            if self._cancelled:
                return
            if self._task == 'train':
                self._do_train()
            elif self._task == 'apply':
                self._do_apply(training_only=self._training_only)
        except Exception as exc:
            traceback.print_exc()
            self.finished.emit(False, str(exc))

    def _emit(self, msg: str):
        self.progress.emit(msg)

    def _emit_progress(self, current: int, total: int):
        """Emit a deterministic progress update (0,0 = indeterminate)."""
        self.progress_value.emit(current, total)

    def _do_train(self):
        plugin = self._plugin
        algo = self._algo

        training_data = plugin._load_training_data(af_state=self._af_state)
        if training_data is None:
            self.finished.emit(False, "No training data could be loaded.")
            return

        if self._cancelled:
            self.finished.emit(False, "Cancelled.")
            return

        self._emit(f"Training {algo} on {len(training_data):,} events …")
        # Signal indeterminate progress while setting up
        self._emit_progress(0, 0)

        if algo == 'UMAP':
            # UMAP supports a tqdm hook — gives us per-epoch progress.
            n_epochs = self._params.get('n_epochs', 500)
            hook = _UMAPTqdmHook(callback=self._emit_progress)
            reducer = plugin._run_umap(self._params, training_data,
                                       progress_hook=hook,
                                       n_epochs_total=n_epochs)
        elif algo == 'tSNE':
            reducer = plugin._run_opentsne(self._params, training_data)
        elif algo == 'PaCMAP':
            reducer = plugin._run_pacmap(self._params, training_data)
        else:
            self.finished.emit(False, f"Unknown algorithm: {algo}")
            return

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
        plugin.state.embedding_features[algo] = {}

        self._emit(f"{algo} training complete.  Embedding training samples …")
        self._do_apply(training_only=True)

    def _do_apply(self, training_only: bool):
        plugin = self._plugin
        algo = self._algo
        reducer = plugin.state.trained_reducers.get(algo)
        if reducer is None:
            self.finished.emit(False, f"No trained {algo} model found.")
            return

        raw_subdir = plugin.controller.experiment.settings['raw']['raw_samples_subdirectory']

        if training_only:
            # Embed only the training samples
            all_sample_keys = []
            all_samples_dict = plugin.controller.experiment.samples.get('all_samples', {})
            for abs_key in all_samples_dict:
                try:
                    rel = str(__import__('pathlib').Path(abs_key).relative_to(raw_subdir))
                except ValueError:
                    rel = abs_key
                if rel in plugin.state.training_sample_ids:
                    all_sample_keys.append((abs_key, rel))
        else:
            # Embed every sample in the experiment
            all_sample_keys = []
            for abs_key in plugin.controller.experiment.samples.get('all_samples', {}).keys():
                try:
                    rel = str(__import__('pathlib').Path(abs_key).relative_to(raw_subdir))
                except ValueError:
                    rel = abs_key
                all_sample_keys.append((abs_key, rel))

        if algo not in plugin.state.embeddings:
            plugin.state.embeddings[algo] = {}
        if algo not in plugin.state.embedding_features:
            plugin.state.embedding_features[algo] = {}

        n_total = len(all_sample_keys)
        for i, (abs_key, rel_path) in enumerate(all_sample_keys):
            if self._cancelled:
                self.finished.emit(False, "Cancelled.")
                return
            self._emit(f"  Embedding {__import__('pathlib').Path(rel_path).name} …")
            self._emit_progress(i, n_total)
            sample_data = plugin._get_sample_data(rel_path, algo, af_state=self._af_state)
            if sample_data is None:
                continue
            try:
                if algo == 'UMAP':
                    emb = reducer.transform(sample_data)
                elif algo == 'tSNE':
                    emb = reducer.transform(sample_data)
                elif algo == 'PaCMAP':
                    # transform(new_data, basis=train_data) is required;
                    # returns a plain ndarray (not a tuple).
                    emb = reducer.transform(sample_data)
                else:
                    continue
                plugin.state.embeddings[algo][rel_path] = emb.astype(np.float32)
                # Same feature vectors that produced this embedding --
                # cached so T-REX can align to them row-for-row later,
                # with no live re-gating step in between.
                plugin.state.embedding_features[algo][rel_path] = sample_data.astype(np.float32)
            except Exception as e:
                self._emit(f"    Could not embed {rel_path}: {e}")

        n = len(plugin.state.embeddings[algo])
        label = "training " if training_only else ""
        self._emit(f"Embeddings complete: {n} {label}sample(s) projected.")
        self.finished.emit(True, "")


# ---------------------------------------------------------------------------
# 5g.  Cluster Annotation tab (Item 8)
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
        side = max(0, min(self.width(), self.height()))
        x = (self.width() - side) // 2
        y = (self.height() - side) // 2
        self._child.setGeometry(x, y, side, side)
        super().resizeEvent(event)

class ClusterAnnotationTab(QWidget):
    """
    Tab 5 — Cluster Annotation (Item 8)
    -------------------------------------
    Three coordinated panels for a SELECTED clustering run:
      1. Per-marker violin plots, one per selected channel, cluster on the
         x-axis. Values come from drc_pipeline.load_sample_marker_values(),
         pooled across the run's own training samples and split by that
         run's own per-sample label arrays — never the "currently active"
         globals, so switching runs here never touches Workspace state.
      2. Cluster map — a DR scatter plot coloured by cluster, via its own
         DR-run selector (Item 6's "other places that now need to know
         about multiple runs"). Rendering, legend editing, and the
         DR/clustering compatibility warning all delegate to drc_scatter,
         shared with Workspace's PlotCard so both render identically.
      3. Cluster label table — one row per cluster: editable name, colour
         swatch (double-click to change), event count, % of total. Both
         this table and the map legend rename/recolour through the same
         drc_scatter helpers PlotCard uses, so edits made from either UI
         (or from Workspace) are always the SAME per-run 'names'/'colors'
         payload — there is no separate copy to fall out of sync.
    """

    def __init__(self, state: PipelineState, bus, controller, parent=None):
        super().__init__(parent)
        self.state = state
        self.bus = bus
        self.controller = controller
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

        # ---- Clustering run selector ----
        run_row = QHBoxLayout()
        run_row.addWidget(QLabel("Clustering run:"))
        self.run_combo = QComboBox()
        self.run_combo.setMinimumWidth(280)
        self.run_combo.setToolTip("Archived clustering run to annotate.")
        self.run_combo.currentIndexChanged.connect(self._on_run_changed)
        run_row.addWidget(self.run_combo)
        run_row.addStretch()
        content_layout.addLayout(run_row)

        splitter = QSplitter(Qt.Vertical)
        content_layout.addWidget(splitter, stretch=1)

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

        self.plot_violins_btn = QPushButton("▶  Plot Violins")
        self.plot_violins_btn.clicked.connect(self._plot_violins)
        violin_layout.addWidget(self.plot_violins_btn)

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
        map_layout.addLayout(map_row)

        map_plot_row = QHBoxLayout()
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        self._map_figure = Figure(figsize=(4, 4), constrained_layout=True)
        self._map_canvas = FigureCanvasQTAgg(self._map_figure)
        self._map_canvas.setMinimumSize(200, 200)
        self._map_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._map_square = _SquareContainer(self._map_canvas)
        self._map_square.setMinimumSize(200, 200)
        self._map_square.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        map_plot_row.addWidget(self._map_square, stretch=1)

        legend_scroll = QScrollArea()
        legend_scroll.setWidgetResizable(True)
        legend_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        legend_scroll.setFixedWidth(140)
        legend_scroll.setFrameShape(QFrame.NoFrame)
        self._legend_widget = QWidget()
        self._legend_layout = QVBoxLayout(self._legend_widget)
        self._legend_layout.setContentsMargins(2, 2, 2, 2)
        self._legend_layout.setSpacing(2)
        self._legend_layout.setAlignment(Qt.AlignTop)
        legend_scroll.setWidget(self._legend_widget)
        map_plot_row.addWidget(legend_scroll)

        map_layout.addLayout(map_plot_row, stretch=1)
        splitter.addWidget(map_box)

        # ============================================================
        # Panel 3 — Cluster label table
        # ============================================================
        table_box = QGroupBox("Cluster Labels")
        table_layout = QVBoxLayout(table_box)

        self.label_table = QTableWidget(0, 4)
        self.label_table.setHorizontalHeaderLabels(['Name', 'Colour', 'Events', '% of total'])
        self.label_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.label_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.label_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.label_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.label_table.verticalHeader().setVisible(False)
        self.label_table.setToolTip(
            "Double-click Name to rename; double-click Colour to recolour."
        )
        self.label_table.itemChanged.connect(self._on_table_item_changed)
        self.label_table.cellDoubleClicked.connect(self._on_table_cell_double_clicked)
        table_layout.addWidget(self.label_table)
        splitter.addWidget(table_box)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([520, 520, 340])
        splitter.setChildrenCollapsible(False)
        self._violin_scroll.setMinimumHeight(400)

    # ------------------------------------------------------------------
    # Refresh (called on tab activation)
    # ------------------------------------------------------------------

    def refresh(self):
        self._populate_run_combo()
        self._populate_dr_run_combo()
        self._populate_channel_list()
        self._update_compat_warning()
        self._redraw_map()
        self._populate_label_table()

    # ------------------------------------------------------------------
    # Run selectors
    # ------------------------------------------------------------------

    def _populate_run_combo(self):
        """Rebuild the clustering-run combo from state.clustering_runs,
        keyed by run_id (labels are user-editable — Item 6's rename)."""
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

    def _on_run_changed(self, _index: int):
        self._update_compat_warning()
        self._redraw_map()
        self._populate_label_table()
        self._show_violin_placeholder(
            "Clustering run changed — click 'Plot Violins' to refresh."
        )

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

        for grid_idx, ch in enumerate(channels):
            cb = QCheckBox(labels.get(ch, ch))
            cb.setChecked(ch in previously_checked)
            self.channel_checkboxes[ch] = cb
            row, col = divmod(grid_idx, 4)
            self.channel_grid.addWidget(cb, row, col)

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

    def _plot_violins(self):
        """
        Split by the SELECTED run's own per-sample label arrays and draw
        one violin subplot per checked channel. Prefers the run's own
        'marker_values' snapshot (frozen at classification time -- see
        drc_clustering.py's _snapshot_marker_values -- guaranteed
        row-for-row aligned to 'labels' regardless of gate/channel edits
        made since). Falls back to live-reloading + truncating for runs
        archived before that snapshot existed.
        """
        cl_run = self._selected_cluster_run()
        if cl_run is None:
            QMessageBox.warning(self, "No Run Selected", "Select a clustering run first.")
            return
        channels = self._checked_channels()
        if not channels:
            QMessageBox.warning(self, "No Channels", "Check at least one channel to plot.")
            return

        labels_dict = cl_run.get('labels', {}) or {}
        snapshot_dict = cl_run.get('marker_values', {}) or {}
        training_samples = cl_run.get('training_sample_ids', [])
        pooled: dict[str, dict[int, list]] = {ch: {} for ch in channels}

        for rel in training_samples:
            labels = labels_dict.get(rel)
            if labels is None:
                continue
            labels = np.asarray(labels)

            snap = snapshot_dict.get(rel)
            if snap is not None:
                values, names = snap
                values = np.asarray(values)
                if len(values) != len(labels):
                    _log.warning(
                        "_plot_violins: %s -- snapshot (%d) vs labels (%d) "
                        "length mismatch, skipping sample.",
                        rel, len(values), len(labels),
                    )
                    continue
            else:
                mv = drc_pipeline.load_sample_marker_values(self.controller, self.state, rel)
                if mv is None:
                    continue
                values, names = mv
                if len(values) != len(labels):
                    _log.warning(
                        "_plot_violins: %s -- values (%d) vs labels (%d) length "
                        "mismatch, truncating to %d. This run predates the "
                        "marker-value snapshot fix -- re-run clustering for "
                        "guaranteed alignment.",
                        rel, len(values), len(labels), min(len(values), len(labels)),
                    )
                m = min(len(values), len(labels))
                values, labels = values[:m], labels[:m]

            for ch in channels:
                if ch not in names:
                    continue
                col = values[:, names.index(ch)]
                for cl_id in np.unique(labels):
                    if cl_id < 0:
                        continue
                    pooled[ch].setdefault(int(cl_id), []).append(col[labels == cl_id])

        for ch in channels:
            means = {cl: float(np.mean(np.concatenate(vals)))
                    for cl, vals in pooled.get(ch, {}).items()}
            _log.info("_plot_violins: %s per-cluster raw mean: %s", ch, means)

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from PySide6.QtGui import QPalette
        from PySide6.QtWidgets import QApplication

        n = len(channels)
        n_cols = min(3, n)
        n_rows = -(-n // n_cols)

        _palette = QApplication.instance().palette()
        is_dark = _palette.color(QPalette.ColorRole.Base).value() < 128
        _fg = 'white' if is_dark else 'black'
        _violin_color = '#5dade2' if is_dark else '#2e6da4'
        _mean_color = '#ffd54f' if is_dark else '#c0392b'
        display_labels = _antigen_dash_labels(self.controller)

        fig = Figure(figsize=(4.2 * n_cols, 3.2 * n_rows), constrained_layout=True)
        if is_dark:
            fig.patch.set_facecolor('#1e1e1e')
        names_map = cl_run.get('names', {})

        for i, ch in enumerate(channels):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            if is_dark:
                ax.set_facecolor('#2b2b2b')
                ax.tick_params(colors=_fg)
                ax.xaxis.label.set_color(_fg)
                ax.yaxis.label.set_color(_fg)
                ax.title.set_color(_fg)
                for spine in ax.spines.values():
                    spine.set_edgecolor(_fg)
            # Explicit grid, drawn UNDER the data and deliberately faint --
            # otherwise this inherits whatever global style is active (see
            # the "suppress inherited seaborn 'whitegrid'" comments
            # elsewhere in this file), which draws full-strength gridlines
            # on TOP of the violins.
            ax.set_axisbelow(True)
            ax.grid(True, axis='y', linewidth=0.4, alpha=0.15, color=_fg)
            title = display_labels.get(ch, ch)
            by_cluster = pooled.get(ch, {})
            cl_ids = sorted(by_cluster.keys())
            if not cl_ids:
                ax.set_title(f"{title} (no data)", fontsize=8)
                ax.axis('off')
                continue
            data = [_apply_channel_transform(self.state, ch, np.concatenate(by_cluster[cl_id]))
                   for cl_id in cl_ids]
            parts = ax.violinplot(data, showmeans=True, showextrema=False)
            for body in parts['bodies']:
                body.set_facecolor(_violin_color)
                body.set_edgecolor(_violin_color)
                body.set_alpha(0.75)
            if 'cmeans' in parts:
                parts['cmeans'].set_color(_mean_color)
                parts['cmeans'].set_linewidth(1.5)
            ax.set_xticks(range(1, len(cl_ids) + 1))
            ax.set_xticklabels(
                [names_map.get(cl, str(cl)) for cl in cl_ids],
                rotation=45, ha='right', fontsize=7,
            )
            ax.set_title(title, fontsize=9)
            tick_spec = _channel_axis_ticks(self.state, ch)
            if tick_spec is not None:
                major_ticks, minor_ticks, limits = tick_spec
                ax.set_yticks([pos for pos, _label in major_ticks])
                ax.set_yticklabels([label for _pos, label in major_ticks])
                ax.set_yticks([pos for pos, _label in minor_ticks], minor=True)
                ax.tick_params(axis='y', which='minor', length=2, labelsize=0)
                ax.set_ylim(limits[0], limits[1])
                ax.set_ylabel('Intensity', fontsize=7)
            else:
                ax.set_ylabel('Transformed intensity', fontsize=7)
            ax.tick_params(labelsize=7)

        canvas = FigureCanvasQTAgg(fig)
        dpi = fig.get_dpi()
        canvas.setMinimumSize(int(fig.get_figwidth() * dpi), int(fig.get_figheight() * dpi))
        self._violin_scroll.setWidget(canvas)
        canvas.draw()

    # ------------------------------------------------------------------
    # Panel 2 — cluster map
    # ------------------------------------------------------------------

    def _redraw_map(self):
        cl_run = self._selected_cluster_run()
        dr_run = self._selected_dr_run()

        self._map_figure.clear()
        ax = self._map_figure.add_subplot(111)
        ax.set_aspect('equal', adjustable='datalim')

        if dr_run is None:
            ax.text(0.5, 0.5, 'No DR run selected.', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9)
            self._map_canvas.draw_idle()
            self._rebuild_map_legend(cl_run)
            return

        emb_dict = dr_run.get('embeddings', {}) or {}
        if not emb_dict:
            ax.text(0.5, 0.5, f"No embeddings for \"{dr_run.get('label', '')}\".",
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)
            self._map_canvas.draw_idle()
            self._rebuild_map_legend(cl_run)
            return

        labels_dict = cl_run.get('labels', {}) if cl_run else {}
        own_positions = (cl_run.get('dr_positions', {}) if cl_run else {}) or {}
        cl_params = (cl_run.get('params', {}) if cl_run else {}) or {}
        # Only trust the run's own frozen positions as a substitute when
        # they're nominally the SAME embedding space as what's on screen --
        # mixing two different algorithms' coordinates would be meaningless.
        same_algo = bool(cl_params.get('_dr_algo')) and \
            cl_params.get('_dr_algo') == (dr_run.get('algorithm') or '')
        xys, labs = [], []
        for rel, emb in emb_dict.items():
            n = len(emb)
            lbl = labels_dict.get(rel)
            if lbl is not None and len(lbl) == n:
                xys.append(emb)
                labs.append(lbl)
            elif lbl is not None and same_algo and rel in own_positions \
                    and len(own_positions[rel]) == len(lbl):
                # The selected DR run's live embeddings for this sample no
                # longer match this clustering run's labels (e.g. DR was
                # retrained for this algorithm since archiving) -- fall
                # back to the exact positions this run actually classified
                # against instead of greying the sample out.
                xys.append(own_positions[rel])
                labs.append(lbl)
            else:
                xys.append(emb)
                labs.append(np.full(n, -1, dtype=np.int32))
        xy = np.concatenate(xys, axis=0)
        lab = np.concatenate(labs, axis=0)

        if len(xy) > 30_000:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(xy), 30_000, replace=False)
            xy, lab = xy[idx], lab[idx]

        algo = dr_run.get('algorithm', '') or ''
        ax.set_xlabel(f"{algo} 1", fontsize=7)
        ax.set_ylabel(f"{algo} 2", fontsize=7)
        ax.tick_params(labelsize=7)

        drc_scatter.draw_cluster_scatter(ax, xy, lab, cl_run, self.controller)
        self._map_canvas.draw_idle()
        self._rebuild_map_legend(cl_run)

    def _rebuild_map_legend(self, cl_run: dict | None):
        while self._legend_layout.count():
            item = self._legend_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        colors = cl_run.get('colors', {}) if cl_run else {}
        names = cl_run.get('names', {}) if cl_run else {}
        for lbl in sorted(colors.keys()):
            color = colors[lbl]
            name = names.get(lbl, 'Noise' if lbl < 0 else str(lbl))

            row = QHBoxLayout()
            row.setSpacing(4)

            swatch = QPushButton()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(
                f"background-color: {color}; border: 1px solid #555; border-radius: 2px;"
            )
            swatch.setToolTip("Right-click to change colour")
            swatch.setContextMenuPolicy(Qt.CustomContextMenu)
            swatch.customContextMenuRequested.connect(
                lambda pos, l=lbl: self._pick_colour(l)
            )
            row.addWidget(swatch)

            name_lbl = QLabel(name)
            name_lbl.setStyleSheet("font-size: 9px;")
            name_lbl.setToolTip("Double-click to rename")
            name_lbl.mouseDoubleClickEvent = lambda e, l=lbl: self._prompt_rename(l)
            row.addWidget(name_lbl, stretch=1)

            row_w = QWidget()
            row_w.setLayout(row)
            self._legend_layout.addWidget(row_w)

        self._legend_layout.addStretch()

    def _prompt_rename(self, label: int):
        from PySide6.QtWidgets import QInputDialog
        cl_run = self._selected_cluster_run()
        if cl_run is None:
            return
        names = cl_run.get('names', {})
        current = names.get(label, 'Noise' if label < 0 else str(label))
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

        self.label_table.setRowCount(len(unique))
        for row, cl_id in enumerate(unique):
            name = names.get(cl_id, 'Noise' if cl_id < 0 else str(cl_id))
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
        self.label_table.blockSignals(False)

        # Grow to show every cluster, capped at ~half when there are a lot
        # (matches the violin/map panels rather than fighting them for space).
        row_h = self.label_table.verticalHeader().defaultSectionSize()
        header_h = self.label_table.horizontalHeader().height()
        n_rows = len(unique)
        visible_rows = n_rows if n_rows <= 20 else max(10, n_rows // 2)
        self.label_table.setMinimumHeight(header_h + row_h * visible_rows + 8)

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
        if col != 1:
            return
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


class PluginWidget(QWidget):
    """
    Top-level plugin widget.

    Inserted as a tab in the Honeychrome main window by plugin_loaders.py.
    Owns the PipelineState and all four inner tabs.

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
        _t0 = time.perf_counter()
        _ensure_qt_imports()
        _log.info("TIMING PluginWidget.__init__: _ensure_qt_imports took %.3fs",
                   time.perf_counter() - _t0)

        # Install missing ML dependencies (also deferred to main thread).
        _t0 = time.perf_counter()
        _bootstrap()
        _log.info("TIMING PluginWidget.__init__: _bootstrap took %.3fs",
                   time.perf_counter() - _t0)

        # Shared state — passed by reference to all inner tabs
        self.state = PipelineState()

        # {algorithm: run_id} — tracks which archived DR run 'Apply to All
        # Samples' should extend, for the algorithm most recently trained
        # in THIS session (see _archive_dr_run / _update_archived_dr_run).
        self._active_dr_run_id: dict[str, str] = {}

        # QSettings instance — keyed per experiment in save_state/load_state
        self._qsettings = QSettings('honeychrome', f'plugin_{plugin_name}')

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
        self._loaded_experiment_key = None

        # ------------------------------------------------------------------
        # Outer layout: disabled label + scrollable content
        # ------------------------------------------------------------------
        self.label_disabled = QLabel(
            f"{plugin_name}: loading...  "
            "Set up the spectral model first if you have not done so."
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
        # Inner tab widget — four tabs
        # ------------------------------------------------------------------
        self.inner_tabs = QTabWidget()
        self.inner_tabs.setDocumentMode(True)

        self.config_tab = ConfigTab(self.state, bus, controller)
        self.transform_tab = TransformTab(self.state, bus, controller)
        self.groups_stats_tab = GroupsStatsTab(self.state, bus, controller)
        self.workspace_tab = WorkspaceTab(self.state, bus, controller)
        self.cluster_annotation_tab = ClusterAnnotationTab(self.state, bus, controller)

        self.inner_tabs.addTab(self.transform_tab,    "Transforms")
        self.inner_tabs.addTab(self.config_tab,       "Configuration")
        self.inner_tabs.addTab(self.cluster_annotation_tab, "Cluster Annotation")
        self.inner_tabs.addTab(self.groups_stats_tab, "Stats")
        self.inner_tabs.addTab(self.workspace_tab,    "Workspace")

        self.inner_tabs.currentChanged.connect(self._on_inner_tab_changed)

        # Item 6: a rename/delete in the run management table must be
        # reflected immediately in every other run selector, not just on
        # next tab switch.
        self.config_tab.run_table.runsChanged.connect(self._on_runs_changed)

        # Keep the two gate trees (TransformTab's primary, ConfigTab's
        # override) in sync — same idea as the old combo pair's
        # blockSignals(), applied to tree checkbox state instead (§0.3).
        self.transform_tab.gate_tree.selectionChanged.connect(self._on_gate_tree_changed)
        self.config_tab.gate_tree.selectionChanged.connect(self._on_gate_tree_changed)

        content_layout.addWidget(self.inner_tabs)

        # Start with content hidden until unmixing is confirmed available
        self.content_widget.setVisible(False)

        # Persist the workspace layout immediately on every add/remove, not
        # just when the user switches inner tabs — otherwise a deletion is
        # lost if the plugin tab (or the app) closes before that happens.
        self.workspace_tab.plots_changed.connect(self._save_plot_configs_only)

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
        A run was renamed or deleted in ConfigTab's management table
        (Item 6).  Refresh every other place that lists runs so the
        change shows up immediately.
        """
        if hasattr(self, 'groups_stats_tab'):
            self.groups_stats_tab._populate_run_combo()
            self.groups_stats_tab._populate_trex_dr_combo()
            self.groups_stats_tab._update_run_button()
        if hasattr(self, 'workspace_tab'):
            for card in self.workspace_tab._plot_cards:
                card._populate_selectors()
        if hasattr(self, 'cluster_annotation_tab'):
            self.cluster_annotation_tab.refresh()

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

            # Sync state with current experiment
            self._loading = True
            try:
                all_samples = _non_control_sample_paths(self.controller)
                self.state.initialise_sample_groups(all_samples)
                self.state.channel_transform_params = _read_transforms_from_experiment(
                    self.controller
                )

                current_key = self._settings_key()
                if current_key != self._loaded_experiment_key:
                    # Only reload from QSettings / rebuild the workspace when
                    # we're switching to a DIFFERENT experiment. On every
                    # other activation, self.state.plot_configs is already
                    # authoritative (kept live by add_plot/remove_card) —
                    # QSettings is only written on inner-tab switch, so
                    # re-pulling it here on every reactivation was clobbering
                    # live deletions with a stale on-disk snapshot.
                    self.load_state()
                    if hasattr(self, 'workspace_tab'):
                        wt = self.workspace_tab
                        wt.col_combo.setCurrentText(str(self.state.workspace_n_columns))
                        for card in list(wt._plot_cards):
                            wt.remove_card(card)
                        pending = getattr(self, '_pending_plot_configs', [])
                        for cfg in pending:
                            wt.add_plot(refresh=False)
                            if wt._plot_cards:
                                wt._plot_cards[-1].apply_config(cfg)
                    if hasattr(self, 'groups_stats_tab'):
                        gst = self.groups_stats_tab
                        if hasattr(gst, 'chk_include_type_markers'):
                            gst.chk_include_type_markers.setChecked(
                                bool(getattr(self, '_pending_include_type_markers', False))
                            )
                    self._loaded_experiment_key = current_key
            finally:
                self._loading = False

            # Refresh the active inner tab
            _t0 = time.perf_counter()
            self._refresh_active_tab()
            _log.info("TIMING initialise_gui: _refresh_active_tab (tab index %d) took %.3fs",
                       self.inner_tabs.currentIndex(), time.perf_counter() - _t0)
        else:
            self.label_disabled.setVisible(True)
            self.content_widget.setVisible(False)

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
        in __init__). Deliberately skips the pickled model sidecar that the
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
        finally:
            s.endGroup()

    def save_state(self):
        """
        Persist the plugin's PipelineState to QSettings so the user can
        pick up where they left off next time this experiment is opened.

        Lightweight scalar/list values go to QSettings.
        Fitted models, embeddings, and cluster labels are pickled to a
        sidecar file alongside the experiment (avoids QSettings size limits).
        """
        if self._loading:
            return
        key = self._settings_key()
        s = self._qsettings
        s.beginGroup(key)
        try:
            s.setValue('selected_gates',    list(self.state.selected_gates))
            s.setValue('selected_channels', self.state.selected_channels)
            s.setValue('n_training_events', self.state.n_training_events)
            s.setValue('training_samples',  self.state.training_sample_ids)
            s.setValue('sample_groups',     repr(self.state.sample_groups))
            s.setValue('group_names',       list(self.state.group_names))
            s.setValue('group_patterns',    repr(self.state.group_patterns))
            s.setValue('compare_group_a',   self.state.compare_group_a)
            s.setValue('compare_group_b',   self.state.compare_group_b)
            # Item 13 phase 2
            s.setValue('testing_group_selection', list(self.state.testing_group_selection))
            s.setValue('contrast_mode',     self.state.contrast_mode)
            s.setValue('reference_group',   self.state.reference_group)
            s.setValue('paired',            self.state.paired)
            s.setValue('pairing_variable',  self.state.pairing_variable)
            s.setValue('covariates',
                      repr(self.state.covariates.to_dict(orient='index'))
                      if self.state.covariates is not None else '')
            # DR / clustering status (lightweight — no arrays)
            s.setValue('dr_status',         repr(self.state.dr_status))
            s.setValue('dr_timestamps',     repr(self.state.dr_timestamps))
            s.setValue('n_clusters',        self.state.n_clusters if self.state.n_clusters is not None else '')
            s.setValue('active_cl_algo',    self.state.active_clustering_algorithm or '')
            s.setValue('cluster_colors',    repr(self.state.cluster_colors))
            s.setValue('workspace_n_columns', self.state.workspace_n_columns)
            # plot_configs: only save serialisable display settings (no widgets)
            try:
                import json
                s.setValue('plot_configs', json.dumps(self.state.plot_configs))
            except Exception:
                pass
            s.setValue('cluster_names', repr(self.state.cluster_names))
            # marker_roles is intentionally NOT persisted across sessions --
            # there's no automatic categorisation yet, so every experiment
            # open should re-default every channel to 'state' rather than
            # carry forward a per-channel dict that may predate this
            # default (see _populate_marker_roles_list). Within a session,
            # checkbox choices already live on self.state.marker_roles in
            # memory and need no QSettings round-trip to survive a tab switch.
            if hasattr(self, 'cluster_annotation_tab'):
                cat = self.cluster_annotation_tab
                s.setValue('annotation_run_id', cat.run_combo.currentData() or '')
                s.setValue('annotation_dr_run_id', cat.dr_run_combo.currentData() or '')
                s.setValue('annotation_channels_checked', cat._checked_channels())
            # Groups tab: include-type-markers checkbox (patterns are
            # already saved directly from state.group_patterns above —
            # Item 13 removed the group_a_pattern/group_b_pattern QLineEdits
            # this block used to read from).
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
            # _on_inner_tab_changed() calls save_state() BEFORE refreshing
            # the newly-activated tab — so on the first switch into
            # Configuration each session, channel_checkboxes/picker can
            # still be empty here even though a real selection already
            # exists on disk. Only overwrite when there's something to save.
            if getattr(self.config_tab, 'channel_checkboxes', None):
                checked = {ch: cb.isChecked()
                           for ch, cb in self.config_tab.channel_checkboxes.items()}
                s.setValue('config_channels_checked', repr(checked))
            if hasattr(self.config_tab, 'picker') and self.config_tab.picker.get_ordered_list():
                s.setValue('config_training_samples',
                           self.config_tab.picker.get_ordered_list())
            # Transform tab: biplot tile y-channels
            if hasattr(self.transform_tab, '_biplot_tiles'):
                tile_y = [t.y_channel() for t in self.transform_tab._biplot_tiles]
                s.setValue('transform_biplot_y_channels', tile_y)
                s.setValue('transform_n_tiles', len(tile_y))
            # Config tab: DR algorithm and hyperparameters
            ct = self.config_tab
            if hasattr(ct, '_dr_algo_group'):
                btn = ct._dr_algo_group.checkedButton()
                if btn:
                    s.setValue('dr_algo', btn.text())
            if hasattr(ct, 'umap_n_neighbors'):
                s.setValue('umap_n_neighbors', ct.umap_n_neighbors.value())
                s.setValue('umap_min_dist',    ct.umap_min_dist.value())
                s.setValue('umap_metric',      ct.umap_metric.currentText())
                s.setValue('umap_n_epochs',    ct.umap_n_epochs.value())
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
        finally:
            s.endGroup()

        # Pickle heavy state (models, embeddings, cluster labels) to sidecar
        self._save_model_sidecar()
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
            gates = s.value('selected_gates', [])
            if gates:
                self.state.selected_gates = list(gates)
            else:
                # Pre-multi-gate settings, saved under the old singular key.
                legacy_gate = s.value('selected_gate', '')
                if legacy_gate:
                    self.state.selected_gates = [legacy_gate]

            channels = s.value('selected_channels', [])
            if channels:
                self.state.selected_channels = list(channels)

            n_ev = s.value('n_training_events', None)
            if n_ev is not None:
                try:
                    self.state.n_training_events = int(n_ev)
                except (ValueError, TypeError):
                    pass

            training = s.value('training_samples', [])
            if training:
                self.state.training_sample_ids = list(training)

            groups_repr = s.value('sample_groups', '')
            group_names_stored = list(s.value('group_names', []))
            loaded_groups = None
            if groups_repr:
                try:
                    loaded_groups = eval(groups_repr)  # noqa: S307
                except Exception:
                    loaded_groups = None

            # Item 13: detect the pre-Item-13 session shape (exactly 2
            # stored names, every sample_groups value restricted to the old
            # 'A'/'B'/'Unassigned' slots) and migrate transparently — 'A' /
            # 'B' become that session's own custom names, one-time, with no
            # user action needed.
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
                self.state.compare_group_a, self.state.compare_group_b = name_a, name_b
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
                self.state.compare_group_a = s.value('compare_group_a', '') or (
                    self.state.group_names[0] if self.state.group_names else '')
                self.state.compare_group_b = s.value('compare_group_b', '') or (
                    self.state.group_names[1] if len(self.state.group_names) > 1 else '')

            # Item 13 phase 2 (restored regardless of the legacy/normal
            # branch above -- these are new fields with no pre-Phase-2
            # equivalent to migrate from).
            selection = s.value('testing_group_selection', [])
            self.state.testing_group_selection = list(selection) if selection else []
            self.state.contrast_mode = s.value('contrast_mode', 'reference') or 'reference'
            self.state.reference_group = s.value('reference_group', '')
            paired_val = s.value('paired', False)
            self.state.paired = paired_val in (True, 'true', 'True', 1, '1')
            self.state.pairing_variable = s.value('pairing_variable', '')
            cov_repr = s.value('covariates', '')
            if cov_repr:
                try:
                    loaded_cov = eval(cov_repr)  # noqa: S307
                    self.state.covariates = (
                        pd.DataFrame.from_dict(loaded_cov, orient='index') if loaded_cov else None
                    )
                except Exception:
                    self.state.covariates = None

            # DR / clustering status metadata
            dr_status_repr = s.value('dr_status', '')
            if dr_status_repr:
                try:
                    self.state.dr_status = eval(dr_status_repr)  # noqa: S307
                except Exception:
                    pass

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

            n_cols = s.value('workspace_n_columns', None)
            if n_cols is not None:
                try:
                    self.state.workspace_n_columns = int(n_cols)
                except (ValueError, TypeError):
                    pass

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

            # marker_roles: deliberately not restored -- see the matching
            # comment in save_state(). state.marker_roles starts empty each
            # session, so _populate_marker_roles_list() always applies the
            # current 'state'-by-default policy.

            self._pending_include_type_markers = s.value('include_type_markers', False)
            self._pending_annotation_run_id = s.value('annotation_run_id', '')
            self._pending_annotation_dr_run_id = s.value('annotation_dr_run_id', '')
            annotation_channels = s.value('annotation_channels_checked', [])
            self._pending_annotation_channels = (
                list(annotation_channels) if annotation_channels else []
            )

            # Config tab: restore channel and training-sample selections
            # (will be applied when ConfigTab.refresh() runs).  Gate
            # selection needs no pending-state variable — ConfigTab.refresh()
            # and TransformTab.refresh() both read state.selected_gates
            # (restored above) directly.
            self._pending_channels_checked = s.value('config_channels_checked', '')
            self._pending_training_samples = s.value('config_training_samples', [])
            # Transform tab: restore biplot tile count and y-channels
            n_tiles = s.value('transform_n_tiles', None)
            tile_y  = s.value('transform_biplot_y_channels', [])
            self._pending_tile_y_channels = list(tile_y) if tile_y else []
            self._pending_n_tiles         = int(n_tiles) if n_tiles else 0
            # DR algorithm and hyperparameters
            self._pending_dr_algo          = s.value('dr_algo', '')
            self._pending_umap_n_neighbors = s.value('umap_n_neighbors', None)
            self._pending_umap_min_dist    = s.value('umap_min_dist', None)
            self._pending_umap_metric      = s.value('umap_metric', '')
            self._pending_umap_n_epochs    = s.value('umap_n_epochs', None)
            self._pending_tsne_perplexity  = s.value('tsne_perplexity', None)
            self._pending_tsne_n_iter      = s.value('tsne_n_iter', None)
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

        finally:
            s.endGroup()

        # Restore heavy state (models, embeddings, cluster labels) from sidecar
        self._load_model_sidecar()
        print(f"[DR Plugin] State loaded for experiment: {key}")

        # Re-evaluate the Run Stats button now that cluster_labels are restored.
        # GroupsStatsTab.refresh() is lazy (only fires on tab switch), so the
        # button would otherwise stay greyed out until the user switches away
        # and back.
        if hasattr(self, 'groups_stats_tab'):
            self.groups_stats_tab._update_run_button()
            if (self.state.freq_results is not None or self.state.mfi_results is not None
                    or self.state.counts_results is not None):
                self.groups_stats_tab.export_results_btn.setEnabled(True)
                # Reset the guard so refresh() redraws on first tab switch.
                self.groups_stats_tab._last_drawn_cluster_names = {}

    # ------------------------------------------------------------------
    # Sidecar persistence — pickle for heavy objects
    # ------------------------------------------------------------------

    def _sidecar_path(self) -> Path | None:
        """
        Return the path for the pickle sidecar file, or None if unavailable.
        Moved under experiment_dir/cache/dr_clustering/ (§0.2) — one-time
        migration from the old loose file happens in _load_model_sidecar().
        """
        try:
            return drc_run_archive.current_state_path(self.controller)
        except Exception:
            return None

    def _save_model_sidecar(self):
        """
        Pickle trained reducers, embeddings, and cluster labels alongside
        the experiment file.  Silently skips on failure (e.g. a reducer that
        is not picklable — the user will need to retrain).

        clustering_runs / dr_runs are NOT included here — those are
        archived individually (their own pickle + manifest entry) at the
        moment each run completes; see drc_run_archive.py.
        """
        path = self._sidecar_path()
        if path is None:
            return
        import pickle
        payload = {}
        for key, obj in (
            ('trained_reducers',  self.state.trained_reducers),
            ('embeddings',        self.state.embeddings),
            ('cluster_labels',    self.state.cluster_labels),
            ('trex_scores',       self.state.trex_scores),
            ('freq_results',      self.state.freq_results),
            ('counts_results',    self.state.counts_results),
            ('mfi_results',       self.state.mfi_results),
            ('freq_df',           self.state.freq_df),
            ('counts_df',         self.state.counts_df),
            ('mfi_df',            self.state.mfi_df),
            ('stats_all_rel',     self.state.stats_all_rel),
            ('stats_group_vec',   self.state.stats_group_vec),
            ('stats_run_label',   self.state.stats_run_label),
            ('stats_run_id',      self.state.stats_run_id),
            ('confusion_df',          self.state.confusion_df),
            ('confusion_run_label',   self.state.confusion_run_label),
            ('confusion_names',       self.state.confusion_names),
            ('composition_df',        self.state.composition_df),
            ('composition_as_pct',    self.state.composition_as_pct),
            ('composition_group_var', self.state.composition_group_var),
            ('composition_run_label', self.state.composition_run_label),
            ('composition_names',     self.state.composition_names),
        ):
            try:
                pickle.dumps(obj)   # probe before writing
                payload[key] = obj
            except Exception as e:
                print(f"[DR Plugin] Sidecar: skipping '{key}' (not picklable: {e})")
        try:
            with open(path, 'wb') as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[DR Plugin] Model sidecar saved → {path.name} "
                  f"(keys: {list(payload.keys())})")
        except Exception as e:
            print(f"[DR Plugin] Could not save model sidecar: {e}")

    def _load_model_sidecar(self):
        """
        Restore trained reducers, embeddings, and cluster labels from the
        pickle sidecar, and rebuild state.dr_runs / state.clustering_runs
        from the run archive's manifest.  Missing or corrupt files are
        silently ignored.
        """
        path = self._sidecar_path()
        if path is not None and not path.exists():
            legacy = drc_run_archive.legacy_current_state_path(self.controller)
            if legacy.exists():
                try:
                    legacy.replace(path)
                    print(f"[DR Plugin] Migrated legacy sidecar → {path}")
                except OSError as e:
                    print(f"[DR Plugin] Could not migrate legacy sidecar: {e}")

        # Rebuild the run archive regardless of whether the current-state
        # sidecar exists — manifest.json/runs/ are independent of it.
        # Metadata only (Item 6 §0.2 lazy-load) — a run's actual payload is
        # hydrated on demand the first time something selects it (see
        # GroupsStatsTab._selected_run_entry / PlotCard's run selector).
        try:
            dr_entries, cl_entries = drc_run_archive.load_manifest_entries(self.controller)
            self.state.dr_runs = dr_entries
            self.state.clustering_runs = cl_entries
            if hasattr(self, 'config_tab'):
                self.config_tab.run_table.refresh()
        except Exception as e:
            print(f"[DR Plugin] Could not load run archive: {e}")

        if path is None or not path.exists():
            return
        import pickle
        try:
            with open(path, 'rb') as f:
                payload = pickle.load(f)
            if isinstance(payload.get('trained_reducers'), dict):
                self.state.trained_reducers = payload['trained_reducers']
            if isinstance(payload.get('embeddings'), dict):
                self.state.embeddings = payload['embeddings']
            if isinstance(payload.get('cluster_labels'), dict):
                self.state.cluster_labels = payload['cluster_labels']
            if isinstance(payload.get('trex_scores'), dict):
                self.state.trex_scores = payload['trex_scores']
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
            if isinstance(payload.get('stats_all_rel'), list):
                self.state.stats_all_rel = payload['stats_all_rel']
            if isinstance(payload.get('stats_group_vec'), list):
                self.state.stats_group_vec = payload['stats_group_vec']
            if isinstance(payload.get('confusion_df'), pd.DataFrame):
                self.state.confusion_df = payload['confusion_df']
            if isinstance(payload.get('confusion_run_label'), str):
                self.state.confusion_run_label = payload['confusion_run_label']
            if isinstance(payload.get('confusion_names'), dict):
                self.state.confusion_names = payload['confusion_names']
            if isinstance(payload.get('composition_df'), pd.DataFrame):
                self.state.composition_df = payload['composition_df']
            if isinstance(payload.get('composition_as_pct'), bool):
                self.state.composition_as_pct = payload['composition_as_pct']
            if isinstance(payload.get('composition_group_var'), str):
                self.state.composition_group_var = payload['composition_group_var']
            if isinstance(payload.get('composition_run_label'), str):
                self.state.composition_run_label = payload['composition_run_label']
            if isinstance(payload.get('composition_names'), dict):
                self.state.composition_names = payload['composition_names']
            if isinstance(payload.get('stats_run_label'), str):
                self.state.stats_run_label = payload['stats_run_label']
            if isinstance(payload.get('stats_run_id'), str):
                self.state.stats_run_id = payload['stats_run_id']
            print(f"[DR Plugin] Model sidecar loaded ← {path.name} "
                  f"({list(self.state.trained_reducers.keys())} trained, "
                  f"{sum(len(v) for v in self.state.embeddings.values())} embeddings, "
                  f"{len(self.state.cluster_labels)} labelled samples, "
                  f"{len(self.state.dr_runs)} DR run(s), "
                  f"{len(self.state.clustering_runs)} clustering run(s) archived)")
        except Exception as e:
            print(f"[DR Plugin] Could not load model sidecar: {e}")

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

        training = list(getattr(self, '_pending_training_samples', []) or [])
        if training and hasattr(self.config_tab, 'picker'):
            # Re-populate the picker with the saved selection order.
            # We need the full pool first; re-derive it the same way ConfigTab.refresh does.
            try:
                raw_subdir = self.controller.experiment.settings['raw'][
                    'raw_samples_subdirectory'
                ]
                all_samples = self.controller.experiment.samples.get('all_samples', {})
                single_stain = set(
                    self.controller.experiment.samples.get('single_stain_controls', [])
                )
                rel_paths = []
                for s in all_samples:
                    if s in single_stain:
                        continue
                    try:
                        rel_paths.append(str(Path(s).relative_to(raw_subdir)))
                    except ValueError:
                        rel_paths.append(s)
                # Only restore samples that are still present in the experiment
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
        if cat.run_combo.currentData() is not None and cat._checked_channels():
            cat._plot_violins()

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

    def _on_inner_tab_changed(self, index: int):
        """Save state when leaving a tab, then refresh the newly activated one."""
        self.save_state()
        self._refresh_tab_at(index)

    def _refresh_active_tab(self):
        self._refresh_tab_at(self.inner_tabs.currentIndex())

    def _refresh_tab_at(self, index: int):
        """Call refresh() on the tab at *index* if it has that method."""
        tabs = [
            self.transform_tab,
            self.config_tab,
            self.cluster_annotation_tab,
            self.groups_stats_tab,
            self.workspace_tab,
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
    # Stage 3 — Dimensionality Reduction
    # ==================================================================

    def _load_training_data(self, af_state=None) -> np.ndarray | None:
        """
        Pool transformed, gated, downsampled events from all training samples.
        Delegates to drc_pipeline.load_training_pool (correct channel alignment,
        correct FlowKit transform, no silent arcsinh fallback, full logging).

        Must be callable from a background thread — contains no Qt GUI calls.
        Pre-flight validation (empty samples / channels / gate) is performed
        by the caller (_run_dr) on the main thread before the worker starts.

        af_state: optional AF snapshot captured on the main thread before the
        worker started — see drc_pipeline.apply_unmixing_af_aware() docstring.
        """
        result = drc_pipeline.load_training_pool(self.controller, self.state, af_state=af_state)
        return result

    def _logicle_transform_array(self, data: np.ndarray, sample=None) -> np.ndarray:
        """
        Apply each selected channel's configured transform and return a
        float32 (n_events, n_selected) feature matrix.

        Delegates to drc_pipeline.transform_selected_channels, which builds the
        channel→column map from the FULL unmixed channel list (no index shift)
        and uses the FlowKit 1.3.0 transform signature with no arcsinh fallback.
        """
        return drc_pipeline.transform_selected_channels(
            self.controller, self.state, data
        )

    def _get_sample_data(self, rel_path: str, algo: str, af_state=None) -> np.ndarray | None:
        """
        Load + unmix + gate + transform ONE sample → (n_events, n_selected).
        Used during 'Apply to All Samples' and per-sample cluster assignment.

        af_state: optional AF snapshot captured on the main thread before the
        worker started — see drc_pipeline.apply_unmixing_af_aware() docstring.
        """
        return drc_pipeline.load_sample_features(self.controller, self.state, rel_path, af_state=af_state)

    # ------------------------------------------------------------------
    # UMAP
    # ------------------------------------------------------------------

    def _run_umap(self, params: dict, training_data: np.ndarray,
                  progress_hook=None, n_epochs_total: int = 500):
        """Train a UMAP reducer and store the kNN index in state."""
        import umap as umap_lib
        import hnswlib

        n_neighbors = params['n_neighbors']
        self.progress_message(f"Building hnswlib kNN index (k={n_neighbors}) …")

        # Build hnswlib index first — reuse for Leiden later
        dim = training_data.shape[1]
        index = hnswlib.Index(space='l2', dim=dim)
        index.init_index(max_elements=len(training_data),
                         ef_construction=200, M=16)
        index.add_items(training_data)
        index.set_ef(50)
        self.state.umap_knn_index = index

        self.progress_message(
            f"Training UMAP  (n_neighbors={n_neighbors}, "
            f"min_dist={params['min_dist']}, metric={params['metric']}) …"
        )

        # Wire the tqdm hook for per-epoch progress if one was supplied.
        tqdm_kwds = {'tqdm_class': progress_hook} if progress_hook is not None else {}

        reducer = umap_lib.UMAP(
            n_neighbors=n_neighbors,
            min_dist=params['min_dist'],
            metric=params['metric'],
            n_epochs=params['n_epochs'],
            n_components=2,
            random_state=42,
            low_memory=False,
            verbose=False,
            tqdm_kwds=tqdm_kwds,
        )
        reducer.fit(training_data)

        # Drop the tqdm hook now training is done. UMAP (sklearn
        # BaseEstimator) stores constructor kwargs as attributes, so
        # reducer.tqdm_kwds otherwise keeps a live reference to the
        # _UMAPTqdmHook instance — a class that lives in the synthetic
        # "bundled_plugins.*" module namespace, which isn't a real
        # importable package. pickle can't re-import it to save a
        # reference, so any object holding this hook fails to pickle.
        # The hook has no purpose after fit() returns.
        reducer.tqdm_kwds = {}

        return reducer

    # ------------------------------------------------------------------
    # openTSNE
    # ------------------------------------------------------------------

    def _run_opentsne(self, params: dict, training_data: np.ndarray):
        """
        Train tSNE using openTSNE.

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

        tsne = openTSNE.TSNE(
            n_components=2,
            perplexity=perplexity,
            n_jobs=n_jobs,
            n_iter=n_iter,
            random_state=42,
            verbose=True,
        )
        # fit() returns a TSNEEmbedding that supports .transform()
        embedding = tsne.fit(training_data)
        self.progress_message("tSNE training complete.")
        return embedding

    # ------------------------------------------------------------------
    # PaCMAP
    # ------------------------------------------------------------------

    def _run_pacmap(self, params: dict, training_data: np.ndarray):
        import pacmap
        self.progress_message(
            f"Training PaCMAP  (n_neighbors={params['n_neighbors']}, "
            f"MN_ratio={params['MN_ratio']}, FP_ratio={params['FP_ratio']}) …"
        )
        reducer = pacmap.PaCMAP(
            n_components=2,
            n_neighbors=params['n_neighbors'],
            MN_ratio=params['MN_ratio'],
            FP_ratio=params['FP_ratio'],
            random_state=42,
        )
        # fit_transform trains and returns the embedding of the training data.
        # We don't use the training embedding directly here — it's reproduced
        # per-sample in _do_apply via reducer.transform(new, basis=train_data).
        reducer.fit_transform(training_data)

        # Wrap reducer + training_data together so the worker can pass 'basis'
        # to transform().  PaCMAP requires the original data for out-of-sample
        # projection when the Annoy index is not cached to disk.
        return _PaCMAPWrapper(reducer, training_data)

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

        self.state.dr_status[algo] = 'running'
        self._set_dr_buttons_running(True)

        # Snapshot AF/transfer-matrix state on the main thread before the
        # worker starts — see apply_unmixing_af_aware() docstring.
        af_state = (
            self.controller.transfer_matrix,
            self.controller.af_precomputed,
            self.controller.af_spectra,
        )

        worker = _DrWorker('train', self, algo, params, training_only=True, af_state=af_state)
        worker.progress.connect(self.progress_message)
        worker.progress_value.connect(
            lambda cur, tot: self._on_dr_progress(cur, tot))
        worker.finished.connect(
            lambda ok, err: self._on_dr_finished(algo, ok, err, task='train', params=params))
        self._dr_worker = worker
        worker.start()

    def _apply_dr_to_all_samples(self, algo: str):
        """Start background embedding of ALL samples.  Returns immediately."""
        if self.state.trained_reducers.get(algo) is None:
            QMessageBox.warning(self, "No Trained Model",
                                f"Train a {algo} model first.")
            return
        if self._dr_worker is not None and self._dr_worker.isRunning():
            QMessageBox.information(self, "DR Running",
                                    "A DR job is already running.  "
                                    "Click Cancel to stop it first.")
            return

        self._set_dr_buttons_running(True)
        # Snapshot AF/transfer-matrix state on the main thread before the
        # worker starts — see apply_unmixing_af_aware() docstring.
        af_state = (
            self.controller.transfer_matrix,
            self.controller.af_precomputed,
            self.controller.af_spectra,
        )
        worker = _DrWorker('apply', self, algo, {}, training_only=False, af_state=af_state)
        worker.progress.connect(self.progress_message)
        worker.progress_value.connect(
            lambda cur, tot: self._on_dr_progress(cur, tot))
        worker.finished.connect(
            lambda ok, err: self._on_dr_finished(algo, ok, err, task='apply'))
        self._dr_worker = worker
        worker.start()

    def _cancel_dr(self):
        """Request cancellation of the running DR worker."""
        if self._dr_worker and self._dr_worker.isRunning():
            self._dr_worker.cancel()
            self.progress_message("DR job cancellation requested …")

    def _on_dr_progress(self, current: int, total: int):
        """Update the DR progress bar from the worker thread (via signal)."""
        if not hasattr(self, 'config_tab'):
            return
        bar = self.config_tab.dr_progress_bar
        if total > 0:
            bar.setRange(0, total)
            bar.setValue(current)
            bar.setFormat(f"epoch {current}/{total}")
            bar.setTextVisible(True)
        else:
            # Indeterminate — e.g. kNN build phase or non-UMAP algorithms
            bar.setRange(0, 0)
            bar.setTextVisible(False)
        bar.setVisible(True)

    def _set_dr_buttons_running(self, running: bool):
        """Toggle Train/Apply/Cancel button states during a DR run."""
        if not hasattr(self, 'config_tab'):
            return
        ct = self.config_tab
        ct.dr_run_btn.setEnabled(not running)
        ct.dr_apply_btn.setEnabled(not running)
        ct.dr_cancel_btn.setEnabled(running)
        if not running and hasattr(ct, 'dr_progress_bar'):
            ct.dr_progress_bar.setVisible(False)
        elif running and hasattr(ct, 'dr_progress_bar'):
            # Show indeterminate bar immediately; UMAP will switch to determinate
            ct.dr_progress_bar.setRange(0, 0)
            ct.dr_progress_bar.setTextVisible(False)
            ct.dr_progress_bar.setVisible(True)

    def _on_dr_finished(self, algo: str, success: bool, error_msg: str,
                        task: str = 'train', params: dict | None = None):
        """Called on the main thread when the DR worker thread finishes."""
        self._set_dr_buttons_running(False)
        if success:
            if self.state.dr_status.get(algo) != 'done':
                # apply-only run: status was already 'done'
                self.state.dr_status[algo] = 'done'
            if task == 'train':
                self._archive_dr_run(algo, params or {})
            else:
                self._update_archived_dr_run(algo)
        else:
            if error_msg and error_msg != 'Cancelled.':
                self.state.dr_status[algo] = 'error'
                self.state.dr_timestamps[algo] = datetime.now().isoformat(timespec='seconds')
                QMessageBox.critical(self, "DR Error", error_msg)
            else:
                self.state.dr_status[algo] = 'idle'
        if hasattr(self, 'config_tab'):
            self.config_tab._refresh_dr_status()
        self._dr_worker = None

    def _archive_dr_run(self, algo: str, params: dict):
        """Archive a freshly completed DR training run (§0.2)."""
        reducer = self.state.trained_reducers.get(algo)
        embeddings = self.state.embeddings.get(algo, {})
        embedding_features = self.state.embedding_features.get(algo, {})
        if reducer is None:
            return
        channels = [c for c in self.state.selected_channels
                    if c not in drc_pipeline.META_CHANNELS]
        n_events = sum(len(e) for e in embeddings.values())
        try:
            entry = drc_run_archive.archive_dr_run(
                self.controller, self.state,
                algorithm=algo,
                reducer=reducer,
                embeddings=dict(embeddings),
                embedding_features=dict(embedding_features),
                gates=list(self.state.selected_gates),
                training_sample_ids=list(self.state.training_sample_ids),
                channels=channels,
                params=dict(params),
                n_events=n_events,
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
        embedding_features = dict(self.state.embedding_features.get(algo, {}))
        drc_run_archive.update_dr_run_embeddings(
            self.controller, self.state, run_id, embeddings, embedding_features
        )
        if hasattr(self, 'config_tab'):
            self.config_tab.run_table.refresh()
        self._on_runs_changed()

    # ==================================================================
    # Stage 4 — Clustering
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

        if hasattr(self, 'config_tab'):
            self.config_tab.cl_run_btn.setEnabled(False)
            self.config_tab.cl_status_label.setText("⏳ Running …")
            self.config_tab.cl_status_label.setStyleSheet("color: orange;")
            if hasattr(self.config_tab, 'cl_progress_bar'):
                bar = self.config_tab.cl_progress_bar
                bar.setRange(0, 0)          # indeterminate
                bar.setTextVisible(False)
                bar.setVisible(True)

        # Snapshot AF/transfer-matrix state on the main thread before the
        # worker starts — see drc_pipeline.apply_unmixing_af_aware() docstring.
        af_state = (
            self.controller.transfer_matrix,
            self.controller.af_precomputed,
            self.controller.af_spectra,
        )

        class _ClWorker(QThread):
            finished = Signal(bool, str)
            progress = Signal(str)
            def __init__(self_, fn):
                super().__init__()
                self_._fn = fn
            def run(self_):
                try:
                    self_._fn()
                    self_.finished.emit(True, '')
                except Exception as e:
                    traceback.print_exc()
                    self_.finished.emit(False, str(e))

        def _do_cluster():
            drc_clustering.run_clustering(
                self.controller, self.state, algo, params,
                progress=self.progress_message,
                af_state=af_state,
            )

        def _on_cl_done(ok, err):
            if not ok:
                self.progress_message(f"Clustering error: {err}")
                QMessageBox.critical(self, "Clustering Error", err)
            else:
                # Archive this run so GroupsStatsTab can select it later.
                if self.state.cluster_labels:
                    channels = [c for c in self.state.selected_channels
                                if c not in drc_pipeline.META_CHANNELS]
                    n_events = sum(len(a) for a in self.state.cluster_labels.values())
                    entry = drc_run_archive.archive_clustering_run(
                        self.controller, self.state,
                        algorithm=algo,
                        cluster_labels=dict(self.state.cluster_labels),
                        colors=dict(self.state.cluster_colors),
                        # Every new run starts with its own EMPTY names dict
                        # (same as colors already does via
                        # assign_cluster_colors resetting state.cluster_colors
                        # each run) — never seeded from state.cluster_names,
                        # which is a legacy global that isn't kept in sync
                        # with per-run renames and was the actual source of
                        # names "bleeding" between unrelated runs.
                        names={},
                        n_clusters=self.state.n_clusters,
                        gates=list(self.state.selected_gates),
                        training_sample_ids=list(self.state.training_sample_ids),
                        channels=channels,
                        params=dict(params),
                        n_events=n_events,
                        marker_values=dict(self.state.cluster_marker_values),
                        dr_positions=dict(self.state.cluster_dr_positions),
                    )
                    self.progress_message(
                        f"Run archived as \"{entry['label']}\""
                    )
            if hasattr(self, 'config_tab'):
                self.config_tab.cl_run_btn.setEnabled(True)
                self.config_tab._refresh_cl_status()
                self.config_tab.run_table.refresh()
                if hasattr(self.config_tab, 'cl_progress_bar'):
                    self.config_tab.cl_progress_bar.setVisible(False)
            self._on_runs_changed()
            self._cl_worker = None

        worker = _ClWorker(_do_cluster)
        worker.finished.connect(_on_cl_done)
        worker.progress.connect(self.progress_message)
        self._cl_worker = worker
        worker.start()

    # ==================================================================
    # Stage 7 — T-REX (on-demand, called from WorkspaceTab PlotCard)
    # ==================================================================

    def run_trex(self):
        """
        Build a T-REX kNN index over pooled Group A + B events and score all
        samples.  Results stored in state.trex_scores.

        Only events belonging to the selected T-REX DR run's OWN embedding
        can ever be plotted with a score, so scoring is restricted to that
        run's event set from the start -- rather than freshly re-gating
        live data and hoping it still lines up with whatever's on screen.

        Algorithm (Irish lab, DOI 10.1016/j.cels.2020.11.009):
          1. Pool all gated events from Group A samples.
          2. Pool all gated events from Group B samples.
          3. Build hnswlib index over A ∪ B.
          4. Per event: query k neighbours; score =
             (n_A_neighbours / n_A_total) − (n_B_neighbours / n_B_total),
             normalised so |score| ≤ 1.
          5. Store per-sample arrays in state.trex_scores.
        """
        if not self.state.stats_runnable():
            QMessageBox.warning(
                self, "T-REX",
                "Assign ≥ 3 samples to each group before running T-REX."
            )
            return

        dr_run = self.groups_stats_tab._selected_trex_dr_run()
        if dr_run is None:
            QMessageBox.warning(
                self, "T-REX",
                "Select a DR run for T-REX to score against first "
                "('T-REX DR run' above the Run T-REX button)."
            )
            return
        emb_dict = dr_run.get('embeddings', {}) or {}
        if not emb_dict:
            QMessageBox.warning(
                self, "T-REX",
                f"\"{dr_run.get('label', '')}\" has no embeddings yet -- "
                "run 'Apply to All Samples' for it first."
            )
            return
        if not (dr_run.get('embedding_features') or {}):
            QMessageBox.warning(
                self, "T-REX",
                f"\"{dr_run.get('label', '')}\" was archived before T-REX's "
                "marker-space feature cache was added -- re-run 'Apply to "
                "All Samples' for it (or retrain) to enable T-REX."
            )
            return

        self.progress_message("Building T-REX kNN index …")

        plugin_ref = self

        class _TrexWorker(QThread):
            finished = Signal(bool, str)
            progress = Signal(str)

            def __init__(self_, dr_run):
                super().__init__()
                self_._dr_run = dr_run

            def run(self_):
                try:
                    self_._do_trex()
                    self_.finished.emit(True, '')
                except Exception as exc:
                    traceback.print_exc()
                    self_.finished.emit(False, str(exc))

            def _do_trex(self_):
                import hnswlib

                state = plugin_ref.state
                raw_subdir = plugin_ref.controller.experiment.settings['raw']['raw_samples_subdirectory']
                emb_dict = self_._dr_run.get('embeddings', {}) or {}
                feat_dict = self_._dr_run.get('embedding_features', {}) or {}

                def _to_rel(sp):
                    try:
                        return str(Path(sp).relative_to(raw_subdir))
                    except ValueError:
                        return sp

                # kNN runs in the ORIGINAL marker-feature space (feat_dict),
                # not the 2D embedding -- true T-REX neighbours, per the
                # published algorithm. feat_dict and emb_dict were cached
                # together, row-for-row, at embedding time (_DrWorker), so
                # scoring here still lines up exactly with what's plotted,
                # with no live re-gating step involved anywhere.
                cmp_a, cmp_b = state.compare_group_a, state.compare_group_b
                a_rels = [_to_rel(sp) for sp, g in state.sample_groups.items()
                          if g == cmp_a and _to_rel(sp) in feat_dict]
                b_rels = [_to_rel(sp) for sp, g in state.sample_groups.items()
                          if g == cmp_b and _to_rel(sp) in feat_dict]

                self_.progress.emit("Pooling Group A features for T-REX …")
                a_data = np.concatenate([feat_dict[rel] for rel in a_rels], axis=0) if a_rels else None
                self_.progress.emit("Pooling Group B features for T-REX …")
                b_data = np.concatenate([feat_dict[rel] for rel in b_rels], axis=0) if b_rels else None

                if a_data is None or b_data is None:
                    raise RuntimeError(
                        "Neither group has any samples with cached "
                        "marker-space features in the selected DR run — "
                        "check the T-REX DR run selection and the Compare "
                        "pair."
                    )

                n_a, n_b = len(a_data), len(b_data)
                pooled = np.concatenate([a_data, b_data], axis=0).astype(np.float32)
                group_labels = np.array([cmp_a] * n_a + [cmp_b] * n_b)

                self_.progress.emit(f"Building hnswlib T-REX index ({len(pooled):,} events) …")
                dim = pooled.shape[1]
                idx_h = hnswlib.Index(space='l2', dim=dim)
                idx_h.init_index(max_elements=len(pooled), ef_construction=200, M=16)
                idx_h.add_items(pooled)
                idx_h.set_ef(50)
                state.trex_knn_index = idx_h
                state.trex_knn_group_labels = group_labels

                k = state.trex_k
                all_rels = a_rels + b_rels
                state.trex_scores = {}
                for rel in all_rels:
                    sample_data = feat_dict[rel].astype(np.float32)
                    nn_labels, _ = idx_h.knn_query(sample_data, k=min(k, len(pooled)))
                    scores = np.zeros(len(sample_data), dtype=np.float32)
                    for i, neighbours in enumerate(nn_labels):
                        n_a_nb = np.sum(group_labels[neighbours] == cmp_a)
                        n_b_nb = np.sum(group_labels[neighbours] == cmp_b)
                        score = (n_a_nb / max(n_a, 1)) - (n_b_nb / max(n_b, 1))
                        scores[i] = score
                    mx = max(abs(scores).max(), 1e-9)
                    scores = scores / mx
                    state.trex_scores[rel] = scores
                    self_.progress.emit(f"  T-REX scored {rel}: {len(scores):,} events")

                state.trex_dr_run_id = self_._dr_run.get('run_id')
                self_.progress.emit(f"T-REX complete: scored {len(state.trex_scores)} samples.")

        self._trex_worker = _TrexWorker(dr_run)
        self._trex_worker.progress.connect(self.progress_message)
        self._trex_worker.finished.connect(self._on_trex_finished)
        self._trex_worker.start()

    def _on_trex_finished(self, success: bool, error_msg: str):
        if success:
            self.progress_message("T-REX scores ready.  Switch to Workspace to view.")
            # Refresh workspace if visible
            if self.inner_tabs.currentIndex() == 3:
                self.workspace_tab.refresh()
        else:
            QMessageBox.critical(self, "T-REX Error", error_msg)
        self._trex_worker = None
