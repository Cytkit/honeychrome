"""
gating_model_builder_tab.py — Gating Model Builder plugin for Honeychrome
=========================================================================
Authors automated gating models (``.agmodel`` files) for the Automated
Gating plugin:

  • Hierarchy — import the gate hierarchy from the unmixed gating strategy,
                choose each gate's type, algorithm and parameters.
  • Train     — pick training samples; each gate's boundary is calculated
                from its parent population in a pooled, equal-sized sample
                of every training file (background worker), then inspected
                and adjusted gate by gate.
  • Export    — review the model and save or load ``.agmodel`` files.

Samples are unmixed with their own autofluorescence profile assignments,
exactly as the main window does, and time-QC keep-masks are applied when
that module is installed. Gate calculation and application live in
``ag_core.py`` (shared with the Automated Gating plugin); the plugins have
no runtime dependency on each other.

Plugin contract (plugin_loaders.py):
  plugin_name  str        — main-window tab title
  PluginWidget(QWidget)   — instantiated with (bus, controller)
"""

import json
import os as _os
import sys
import warnings
from copy import deepcopy
from dataclasses import dataclass, field, fields as dataclass_fields
from datetime import datetime
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal, QSettings, QThread
from PySide6.QtGui import QBrush, QColor, QAction
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QPushButton, QLabel, QComboBox, QGroupBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QSizePolicy,
    QSplitter, QLineEdit, QFileDialog, QMessageBox, QFormLayout,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QTabWidget, QTreeWidget, QTreeWidgetItem,
    QAbstractItemView, QMenu,
    QDialog, QDialogButtonBox, QGridLayout,
)

import honeychrome
from honeychrome.controller_components.functions import (
    build_antigen_map,
    build_display_label_map,
    calc_hist1d,
    calc_hist2d,
)
from honeychrome.controller_components.cytometer_whitelist import singlet_channels
from honeychrome.controller_components.sample_loader import (
    analysis_sample_keys,
    control_sample_keys,
    load_unmixed,
    snapshot_unmix_state,
)

# Sibling modules in this directory (not *_tab.py, so not loaded as tabs).
_PLUGIN_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

import ag_core  # noqa: E402
from ag_core import (  # noqa: E402
    ALGORITHMS_BY_GATE_TYPE,
    ALWAYS_EXCLUDED_CHANNELS,
    GATE_TYPES,
    default_algorithm_for_type as _default_algorithm_for_type,
    default_populations_for_type as _default_populations_for_type,
)
import ag_help_texts  # noqa: E402

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
InteractiveLabel = None
OrderedMultiSamplePicker = None
CopyableTableWidget = None
HelpToggleWidget = None

_qt_imports_done = False


def _ensure_qt_imports():
    """Import pyqtgraph and Honeychrome view components (main thread only)."""
    global pg, ZoomAxis, NoPanViewBox, TransparentGraphicsLayoutWidget
    global InteractiveLabel, OrderedMultiSamplePicker, CopyableTableWidget
    global HelpToggleWidget, _qt_imports_done
    if _qt_imports_done:
        return
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
    from honeychrome.view_components.ordered_multi_sample_picker import (
        OrderedMultiSamplePicker as _OMSP,
    )
    OrderedMultiSamplePicker = _OMSP
    from honeychrome.view_components.copyable_table_widget import (
        CopyableTableWidget as _CTW,
    )
    CopyableTableWidget = _CTW
    from honeychrome.view_components.help_toggle_widget import (
        HelpToggleWidget as _HTW,
    )
    HelpToggleWidget = _HTW
    _qt_imports_done = True


# ---------------------------------------------------------------------------
# Plugin identity
# ---------------------------------------------------------------------------
plugin_name = 'Gating Model Builder'

DEFAULT_EVENTS_PER_SAMPLE = 50_000
TRAINING_SEED = 42
SESSION_FILE = Path('cache') / 'gating_model_builder' / 'session.agmodel'


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _antigen_map(controller) -> dict:
    """{channel: antigen} for the experiment's unmixed channels, from the
    spectral model; channels without an antigen map to ''."""
    try:
        return build_antigen_map(
            controller.experiment.settings.get('unmixed', {}).get('event_channels_pnn') or [],
            controller.experiment.process.get('spectral_model') or [],
        )
    except Exception:
        return {}


def _channel_to_antigen(controller, channel: str) -> str:
    """The antigen of *channel* from the spectral model, or the channel name
    itself when it has none, so the caller always has something printable."""
    return _antigen_map(controller).get(channel) or channel


def _experiment_channels(controller) -> list[str]:
    """Return the list of channel names (`$PnN`) for the current experiment.

    Returns an empty list if the experiment is not configured or if the
    settings dict lacks the expected key.  Order is preserved — it
    matches the column order in unmixed event arrays.
    """
    try:
        return list(
            controller.experiment.settings.get('unmixed', {})
            .get('event_channels_pnn') or []
        )
    except Exception:
        return []


def _fluorescence_channels(controller) -> list[str]:
    """Return the channel names that are flagged as fluorescence channels.

    Uses ``settings['unmixed']['fluorescence_channel_ids']`` (a list of
    integer indices into ``event_channels_pnn``) to filter the full
    channel list.  Scatter, time, ribbon, and event_id channels are
    excluded.  Used to pre-populate axis combo boxes in
    `GateEditDialog`.
    """
    pnn = _experiment_channels(controller)
    try:
        ids = controller.experiment.settings.get('unmixed', {}).get(
            'fluorescence_channel_ids'
        ) or []
        return [pnn[i] for i in ids if 0 <= i < len(pnn)]
    except Exception:
        return []


def _gateable_channels(controller, include_scatter: bool = True) -> list[str]:
    """Return channels that may be offered as a gate axis.

    - Always excludes `Time`, `ribbon`, `event_id`.
    - With ``include_scatter=True`` (default) returns all remaining
      channels — fluorescence and scatter — so the user can build
      FSC/SSC gates as well as marker gates.  With ``include_scatter=False``
      returns only fluorescence channels.

    Order is preserved.
    """
    pnn = _experiment_channels(controller)
    if include_scatter:
        return [ch for ch in pnn if ch not in ALWAYS_EXCLUDED_CHANNELS]
    fluoro = set(_fluorescence_channels(controller))
    return [ch for ch in pnn if ch in fluoro]


def _axis_limits(transforms: dict) -> dict:
    """{channel: (lo, hi)} display limits from a transforms dict."""
    out = {}
    for ch, tr in (transforms or {}).items():
        lim = getattr(tr, 'limits', None)
        if lim is not None and len(lim) >= 2:
            out[ch] = (float(lim[0]), float(lim[1]))
    return out


def _is_singlet_pair(controller, channels: list[str]) -> bool:
    """True when two channels are the cytometer's forward-scatter singlet
    pair (area vs height or width).

    Uses the cytometer's canonical names from cytometer_whitelist; for an
    unrecognised cytometer, falls back to two scatter channels sharing a
    base name (e.g. 'FSC-A' / 'FSC-H').
    """
    if len(channels) < 2:
        return False
    pair = set(channels[:2])
    try:
        raw = controller.experiment.settings.get('raw', {})
        canonical = singlet_channels(raw.get('cytometer', ''))
    except Exception:
        canonical = None
    if canonical is not None:
        area, singlet_y = canonical
        stem = area.rsplit('-', 1)[0]
        alternatives = {singlet_y, f'{stem}-H', f'{stem}-W'} - {area}
        return area in pair and bool(pair & alternatives)
    unmixed = controller.experiment.settings.get('unmixed', {})
    pnn = unmixed.get('event_channels_pnn') or []
    scatter = {pnn[i] for i in unmixed.get('scatter_channel_ids') or [] if 0 <= i < len(pnn)}
    a, b = channels[:2]
    return (a in scatter and b in scatter and a != b
            and a.rsplit('-', 1)[0] == b.rsplit('-', 1)[0])


def _infer_gate_type_from_flowkit(gate, controller) -> str:
    """Infer the gate type ('singlets', '1dsep', '2dsep', 'free') of a
    FlowKit gate.

        PolygonGate           → 'singlets' on the forward-scatter singlet
                                pair, otherwise 'free'
        RectangleGate, 1 dim  → '1dsep'
        RectangleGate, 2 dims → '1dsep' if one dimension is open-ended,
                                otherwise '2dsep'
        QuadrantGate          → '2dsep'

    Other gate types (ellipsoid, Boolean) become 'free'; the user can
    change the type afterwards.
    """
    gtype = getattr(gate, 'gate_type', '')
    channels = list(gate.get_dimension_ids()) if hasattr(gate, 'get_dimension_ids') else []

    if gtype == 'PolygonGate':
        return 'singlets' if _is_singlet_pair(controller, channels) else 'free'
    if gtype == 'QuadrantGate':
        return '2dsep'
    if gtype == 'RectangleGate':
        dims = list(getattr(gate, 'dimensions', []) or [])
        n_unbounded = sum(1 for d in dims if _is_unbounded(d))
        if len(dims) == 1 or n_unbounded == 1:
            return '1dsep'
        return '2dsep'
    return 'free'


def _quadrant_divider_range(quadrant, divider_id):
    """(lo, hi) range of one divider in a FlowKit Quadrant, or None."""
    getter = getattr(quadrant, 'get_divider_range', None)
    if callable(getter):
        try:
            return getter(divider_id)
        except Exception:
            pass
    ranges = getattr(quadrant, '_divider_ranges', None)
    if isinstance(ranges, dict):
        return ranges.get(divider_id)
    return None


def _quadrant_populations(gate) -> tuple[dict, list[str]]:
    """Populations for an imported QuadrantGate, one per quadrant.

    Each quadrant's region (x-y-, x+y-, x-y+, x+y+) comes from its divider
    ranges: a quadrant bounded below on a divider lies on that divider's
    positive side. Returns (populations, [x channel, y channel]).
    """
    dividers = list(getattr(gate, 'dimensions', []) or [])[:2]
    channels = [getattr(d, 'dimension_ref', None) for d in dividers]
    positions = {'x-y-': 3, 'x+y-': 4, 'x-y+': 1, 'x+y+': 2}
    order = {'x-y-': 0, 'x+y-': 1, 'x-y+': 2, 'x+y+': 3}
    pops = {}
    for qid, quadrant in (getattr(gate, 'quadrants', {}) or {}).items():
        signs = []
        for div in dividers:
            rng = _quadrant_divider_range(quadrant, getattr(div, 'id', None))
            lo = rng[0] if isinstance(rng, (list, tuple)) and rng else None
            signs.append('+' if lo is not None else '-')
        if len(signs) != 2:
            continue
        region = f'x{signs[0]}y{signs[1]}'
        pops[str(qid)] = {'label': str(qid), 'label_pos': positions[region], 'region': region}
    pops = dict(sorted(pops.items(), key=lambda kv: order[kv[1]['region']]))
    return pops, [c for c in channels if c]


def _range_gate_population(gate, transforms: dict) -> str | None:
    """Which default population ('neg'/'pos', or a quadrant region) a
    Honeychrome range or rectangle gate encloses, judged from where its
    bounds sit on the display axis. Used to point child gates at the
    right parent population."""
    dims = list(getattr(gate, 'dimensions', []) or [])
    regions = []
    for d in dims:
        lo = getattr(d, 'min', None)
        ch = getattr(d, 'id', None)
        lim = getattr(transforms.get(ch), 'limits', None) if ch else None
        if lo is None or (isinstance(lo, float) and not np.isfinite(lo)):
            regions.append('-')
            continue
        if lim is not None and len(lim) >= 2:
            span = float(lim[1]) - float(lim[0])
            regions.append('-' if float(lo) <= float(lim[0]) + 0.02 * span else '+')
        else:
            regions.append('+')
    if len(regions) == 1:
        return 'pos' if regions[0] == '+' else 'neg'
    if len(regions) == 2:
        return {'--': 'DN', '+-': 'X+', '-+': 'Y+', '++': 'DP'}[''.join(regions)]
    return None


def _is_unbounded(dim) -> bool:
    """Return True if a flowkit Dimension is open on at least one side.

    A dimension is considered open if either ``min`` or ``max`` is
    ``None``, NaN, or ±infinity.  flowkit's exact representation varies
    by version (some serialise unbounded sides as None; some as ±inf),
    so all three spellings are accepted.
    """
    lo = getattr(dim, 'min', None)
    hi = getattr(dim, 'max', None)

    def _open(v) -> bool:
        if v is None:
            return True
        try:
            vf = float(v)
        except (TypeError, ValueError):
            return False
        return (vf != vf) or (vf == float('inf')) or (vf == float('-inf'))

    return _open(lo) or _open(hi)


def _polygon_vertices(gate) -> list[tuple[float, float]] | None:
    """Return the vertex list for a PolygonGate, or None for non-polygon gates."""
    if getattr(gate, 'gate_type', '') != 'PolygonGate':
        return None
    v = getattr(gate, 'vertices', None) or []
    return [(float(p[0]), float(p[1])) for p in v]


def _rectangle_corners(gate) -> list[tuple[float, float]] | None:
    """Return the four corners of a RectangleGate as a closed-style list.

    Returns ``None`` for non-rectangle gates, single-dim gates (which
    have no rectangle to draw), or rectangles where one or both
    dimensions are unbounded.  Used by the Hierarchy tab's preview plot
    to overlay the original gate geometry on a density heatmap.
    """
    if getattr(gate, 'gate_type', '') != 'RectangleGate':
        return None
    dims = list(getattr(gate, 'dimensions', []) or [])
    if len(dims) != 2:
        return None
    if _is_unbounded(dims[0]) or _is_unbounded(dims[1]):
        return None
    x0, x1 = float(dims[0].min), float(dims[0].max)
    y0, y1 = float(dims[1].min), float(dims[1].max)
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def _sample_label(key: str) -> str:
    """Short display name for a sample key."""
    return Path(str(key)).stem


# ---------------------------------------------------------------------------
# GatingModelState — single source of truth for the plugin session.
# Owned by PluginWidget and passed by reference to each inner tab; it only
# stores. Everything on it is JSON-serialisable.
# ---------------------------------------------------------------------------

@dataclass
class GatingModelState:
    """Builder session state.

    Attributes
    ----------
    gate_definitions
        Gate definition dicts (see ag_core's module docstring), edited on
        the Hierarchy tab and consumed by training.
    trained_boundaries
        {gate_name: {population_name: entry}} from the last training run
        (plus any manual adjustments); written to the .agmodel.
    training_samples
        Training sample keys (``all_samples`` keys, experiment-relative).
    events_per_sample
        Events drawn from each training sample (seeded random subsample),
        so every sample contributes equally; 0 uses every event.
    include_controls
        Offer single-stain controls and unstained samples in the picker.
    training_summary
        Per-run record: sample keys, events used per sample, cap, seed,
        and the AF/QC state of the training samples.
    model_metadata
        Free-form dict written into the .agmodel ``metadata`` block.
    algorithm_recommendations
        {gate_name: algorithm} from the bimodality check.
    """

    gate_definitions: list = field(default_factory=list)
    trained_boundaries: dict = field(default_factory=dict)
    training_samples: list = field(default_factory=list)
    events_per_sample: int = DEFAULT_EVENTS_PER_SAMPLE
    include_controls: bool = False
    training_summary: dict = field(default_factory=dict)
    model_metadata: dict = field(default_factory=dict)
    algorithm_recommendations: dict = field(default_factory=dict)

    def is_trained(self, gate_name: str) -> bool:
        """Return True if a non-empty boundary exists for *gate_name*."""
        entry = self.trained_boundaries.get(gate_name)
        return bool(entry) and any(
            pop.get('boundary') or pop.get('threshold_x') is not None
            for pop in entry.values()
        )

    def n_gates(self) -> int:
        return len(self.gate_definitions)

    def n_populations(self) -> int:
        return sum(len(g.get('populations', {})) for g in self.gate_definitions)

    def gate_by_name(self, name: str) -> dict | None:
        for g in self.gate_definitions:
            if g.get('gate_name') == name:
                return g
        return None


# ---------------------------------------------------------------------------
# Inner tab widgets. Each exposes refresh(), called by PluginWidget when the
# tab becomes active or after a bus event.
# ---------------------------------------------------------------------------


class GateParamDialog(QDialog):
    """Per-gate "Advanced…" parameter editor.

    Edits the ``gate_param`` sub-dict of a GateDefinition.  These values
    are read by ag_core.GateCalculator and override the global
    defaults that come from `get_autogating_param_aurora` /
    `get_autogating_param_fortessa` style cytometer profiles in the R
    package.  Only a handful of parameters are exposed here — the rest
    are sensible enough as defaults that surfacing them would clutter
    the dialog.

    Parameters surfaced
    -------------------
    quantile_trim
        Symmetric trim applied before any boundary calculation.  Default
        ``0.001`` (= 0.1% from each tail) matches R's
        ``trim.quantiles()``.  Critical for singlet and scatter gates.
    axis_lo / axis_hi
        Crop window for the boundary *calculation*, as fractions
        (0.0-1.0) of that axis's own [min, max] data range -- NOT
        percentiles of the distribution.  Mirrors R's
        ``region.min``/``region.max`` (1dsep) and
        ``region.xmin``/``region.xmax``/``region.ymin``/``region.ymax``
        (2dsep) in ``calculate_gate.r``, default ``0.01``/``0.99`` there.
        Before the threshold algorithm runs, the (quantile-trimmed) data
        is restricted to
        ``[(1-axis_lo)*min + axis_lo*max, (1-axis_hi)*min + axis_hi*max]``;
        the algorithm never sees data outside that window, so the
        resulting threshold is always inside it too.
    tail_fraction
        Symmetric-tail probability for the ``tail`` algorithm.
        Default ``0.01`` (1%).
    mixture_components
        Component count for the ``mixture`` algorithm.  Default ``2``.
    width_sd
        Standard-deviation half-width for the singlet parallelogram.
        Default ``3``.
    """

    def __init__(self, gate_def: dict, parent=None):
        super().__init__(parent)
        self.gate_def = gate_def
        self._is_2dsep = gate_def.get('gate_type') == '2dsep'
        self.setWindowTitle(
            f"Advanced parameters — {gate_def.get('gate_name', '(unnamed)')}"
        )
        self.setModal(True)

        params = gate_def.setdefault('gate_param', {})
        outer = QVBoxLayout(self)

        if self._is_2dsep:
            self._build_2dsep_ui(params, outer, gate_def)
        else:
            self._build_standard_ui(params, outer)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    # ------------------------------------------------------------------
    # UI builders
    # ------------------------------------------------------------------

    def _build_standard_ui(self, params: dict, outer: QVBoxLayout):
        """Single-axis form for non-2dsep gates (unchanged layout)."""
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.sb_quantile_trim = QDoubleSpinBox()
        self.sb_quantile_trim.setRange(0.0, 0.1)
        self.sb_quantile_trim.setDecimals(4)
        self.sb_quantile_trim.setSingleStep(0.0005)
        self.sb_quantile_trim.setValue(float(params.get('quantile_trim', 0.001)))
        self.sb_quantile_trim.setToolTip(
            "Symmetric tail trim applied before boundary calculation."
        )
        form.addRow("Quantile trim", self.sb_quantile_trim)

        self.sb_axis_lo = QDoubleSpinBox()
        self.sb_axis_lo.setRange(0.0, 1.0)
        self.sb_axis_lo.setDecimals(3)
        self.sb_axis_lo.setSingleStep(0.01)
        self.sb_axis_lo.setValue(float(params.get('axis_lo', 0.01)))
        self.sb_axis_lo.setToolTip(
            "Crops the data fed into the threshold calculation to this "
            "fraction of the axis's own min-max range (not a percentile). "
            "Matches R's region.min. Default 0.01."
        )
        form.addRow("Axis lo (calc crop)", self.sb_axis_lo)

        self.sb_axis_hi = QDoubleSpinBox()
        self.sb_axis_hi.setRange(0.0, 1.0)
        self.sb_axis_hi.setDecimals(3)
        self.sb_axis_hi.setSingleStep(0.01)
        self.sb_axis_hi.setValue(float(params.get('axis_hi', 0.99)))
        self.sb_axis_hi.setToolTip(
            "Crops the data fed into the threshold calculation to this "
            "fraction of the axis's own min-max range (not a percentile). "
            "Matches R's region.max. Default 0.99."
        )
        form.addRow("Axis hi (calc crop)", self.sb_axis_hi)

        self.sb_tail_fraction = QDoubleSpinBox()
        self.sb_tail_fraction.setRange(0.0001, 0.5)
        self.sb_tail_fraction.setDecimals(4)
        self.sb_tail_fraction.setSingleStep(0.001)
        self.sb_tail_fraction.setValue(float(params.get('tail_fraction', 0.01)))
        self.sb_tail_fraction.setToolTip(
            "Tail probability for the 'tail' algorithm. Smaller = stricter."
        )
        form.addRow("Tail fraction", self.sb_tail_fraction)

        self.sb_mixture_components = QSpinBox()
        self.sb_mixture_components.setRange(2, 6)
        self.sb_mixture_components.setValue(int(params.get('mixture_components', 2)))
        self.sb_mixture_components.setToolTip(
            "Gaussian components for the 'mixture' algorithm."
        )
        form.addRow("Mixture components", self.sb_mixture_components)

        self.sb_width_sd = QDoubleSpinBox()
        self.sb_width_sd.setRange(0.5, 10.0)
        self.sb_width_sd.setDecimals(2)
        self.sb_width_sd.setSingleStep(0.1)
        self.sb_width_sd.setValue(float(params.get('width_sd', 3.0)))
        self.sb_width_sd.setToolTip(
            "Half-width of the singlet parallelogram, in residual SDs."
        )
        form.addRow("Singlet width (SDs)", self.sb_width_sd)

        outer.addLayout(form)

    def _build_2dsep_ui(self, params: dict, outer: QVBoxLayout, gate_def: dict):
        """Two-column form for 2dsep gates (X axis | Y axis)."""
        default_algo = gate_def.get('algorithm', 'tail')

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(6)

        ch_x = gate_def.get('gate_marker_x', 'X')
        ch_y = gate_def.get('gate_marker_y', 'Y') or 'Y'

        # Header row
        grid.addWidget(QLabel(f"<b>{ch_x} axis</b>"), 0, 1)
        grid.addWidget(QLabel(f"<b>{ch_y} axis</b>"), 0, 2)

        # Algorithm combos
        grid.addWidget(QLabel("Algorithm"), 1, 0)
        self.cb_algo_x = QComboBox()
        self.cb_algo_y = QComboBox()
        for cb, key in ((self.cb_algo_x, 'algorithm_x'),
                        (self.cb_algo_y, 'algorithm_y')):
            cb.addItems(ALGORITHMS_BY_GATE_TYPE['2dsep'])
            val = params.get(key, '') or default_algo
            if val in ALGORITHMS_BY_GATE_TYPE['2dsep']:
                cb.setCurrentText(val)
        grid.addWidget(self.cb_algo_x, 1, 1)
        grid.addWidget(self.cb_algo_y, 1, 2)

        # Tail fraction
        grid.addWidget(QLabel("Tail fraction"), 2, 0)
        self.sb_tail_x = QDoubleSpinBox()
        self.sb_tail_y = QDoubleSpinBox()
        for sb, key in ((self.sb_tail_x, 'tail_fraction_x'),
                        (self.sb_tail_y, 'tail_fraction_y')):
            sb.setRange(0.0001, 0.5)
            sb.setDecimals(4)
            sb.setSingleStep(0.001)
            sb.setValue(float(params.get(key, 0.01)))
        grid.addWidget(self.sb_tail_x, 2, 1)
        grid.addWidget(self.sb_tail_y, 2, 2)

        # Mixture components
        grid.addWidget(QLabel("Mixture components"), 3, 0)
        self.sb_mix_x = QSpinBox()
        self.sb_mix_y = QSpinBox()
        for sb, key in ((self.sb_mix_x, 'mixture_components_x'),
                        (self.sb_mix_y, 'mixture_components_y')):
            sb.setRange(2, 6)
            sb.setValue(int(params.get(key, 2)))
        grid.addWidget(self.sb_mix_x, 3, 1)
        grid.addWidget(self.sb_mix_y, 3, 2)

        # Quantile trim (shared — one row)
        grid.addWidget(QLabel("Quantile trim"), 4, 0)
        self.sb_quantile_trim = QDoubleSpinBox()
        self.sb_quantile_trim.setRange(0.0, 0.1)
        self.sb_quantile_trim.setDecimals(4)
        self.sb_quantile_trim.setSingleStep(0.0005)
        self.sb_quantile_trim.setValue(float(params.get('quantile_trim', 0.001)))
        self.sb_quantile_trim.setToolTip("Applied to both axes before threshold calculation.")
        grid.addWidget(self.sb_quantile_trim, 4, 1, 1, 2)  # span both columns

        # Axis crop window for the calculation (R region.xmin/xmax/ymin/ymax)
        grid.addWidget(QLabel("Axis lo (calc crop)"), 5, 0)
        self.sb_axis_lo_x = QDoubleSpinBox()
        self.sb_axis_lo_y = QDoubleSpinBox()
        for sb, key in ((self.sb_axis_lo_x, 'axis_lo_x'),
                        (self.sb_axis_lo_y, 'axis_lo_y')):
            sb.setRange(0.0, 1.0)
            sb.setDecimals(3)
            sb.setSingleStep(0.01)
            sb.setValue(float(params.get(key, 0.01)))
            sb.setToolTip(
                "Crops the data fed into the threshold calculation to "
                "this fraction of the axis's own min-max range. Default 0.01."
            )
        grid.addWidget(self.sb_axis_lo_x, 5, 1)
        grid.addWidget(self.sb_axis_lo_y, 5, 2)

        grid.addWidget(QLabel("Axis hi (calc crop)"), 6, 0)
        self.sb_axis_hi_x = QDoubleSpinBox()
        self.sb_axis_hi_y = QDoubleSpinBox()
        for sb, key in ((self.sb_axis_hi_x, 'axis_hi_x'),
                        (self.sb_axis_hi_y, 'axis_hi_y')):
            sb.setRange(0.0, 1.0)
            sb.setDecimals(3)
            sb.setSingleStep(0.01)
            sb.setValue(float(params.get(key, 0.99)))
            sb.setToolTip(
                "Crops the data fed into the threshold calculation to "
                "this fraction of the axis's own min-max range. Default 0.99."
            )
        grid.addWidget(self.sb_axis_hi_x, 6, 1)
        grid.addWidget(self.sb_axis_hi_y, 6, 2)

        outer.addLayout(grid)

    def accept(self):
        params = self.gate_def.setdefault('gate_param', {})
        params['quantile_trim'] = float(self.sb_quantile_trim.value())
        if self._is_2dsep:
            params['algorithm_x']          = self.cb_algo_x.currentText()
            params['algorithm_y']          = self.cb_algo_y.currentText()
            params['tail_fraction_x']      = float(self.sb_tail_x.value())
            params['tail_fraction_y']      = float(self.sb_tail_y.value())
            params['mixture_components_x'] = int(self.sb_mix_x.value())
            params['mixture_components_y'] = int(self.sb_mix_y.value())
            params['axis_lo_x']            = float(self.sb_axis_lo_x.value())
            params['axis_hi_x']            = float(self.sb_axis_hi_x.value())
            params['axis_lo_y']            = float(self.sb_axis_lo_y.value())
            params['axis_hi_y']            = float(self.sb_axis_hi_y.value())
        else:
            params['axis_lo']            = float(self.sb_axis_lo.value())
            params['axis_hi']            = float(self.sb_axis_hi.value())
            params['tail_fraction']      = float(self.sb_tail_fraction.value())
            params['mixture_components'] = int(self.sb_mixture_components.value())
            params['width_sd']           = float(self.sb_width_sd.value())
        super().accept()


class GateEditDialog(QDialog):
    """Create-or-edit dialog for a single GateDefinition.

    The dialog operates on a deep copy of the supplied gate dict and
    writes back into the original on Accept; this lets the caller skip
    rollback handling on Cancel.  All combos pre-populate from the
    current experiment's channel and gate lists; Time, ribbon and
    event_id are never offered as axes.

    Parameters
    ----------
    gate_def
        The dict to edit.  Must at minimum carry the GateDefinition
        keys; missing keys are filled with defaults.  Modified in
        place on Accept.
    state
        The shared GatingModelState — used to populate the parent-gate
        combo from existing gates.
    controller
        The Honeychrome controller — used to populate axis channel
        combos from the current experiment.
    is_new
        ``True`` if this gate has not yet been added to
        ``state.gate_definitions`` — affects the dialog title only.
    """

    def __init__(self, gate_def: dict, state: GatingModelState,
                 controller, is_new: bool = False, parent=None):
        super().__init__(parent)
        self.gate_def = gate_def
        self.state = state
        self.controller = controller
        self.is_new = is_new

        self.setWindowTitle(
            ("Add gate" if is_new else "Edit gate") +
            f" — {gate_def.get('gate_name', '(unnamed)')}"
        )
        self.setModal(True)
        self.resize(560, 520)

        # Working copies of mutable substructures.  These get written
        # back into self.gate_def on accept().
        self._working_populations: dict = deepcopy(
            gate_def.get('populations') or {}
        )
        # Tracks the gate type as of the last _on_type_changed call so we
        # only reset populations on an actual change, not on the initial
        # combo population during _build_ui().
        self._last_seen_type: str = gate_def.get('gate_type', 'free')

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        # --- Basic identity ---
        self.le_name = QLineEdit(self.gate_def.get('gate_name', ''))
        self.le_name.setToolTip("Unique gate name (used as a key in trained_boundaries).")
        form.addRow("Gate name", self.le_name)

        # --- Gate type ---
        self.cb_type = QComboBox()
        self.cb_type.addItems(list(GATE_TYPES))
        current_type = self.gate_def.get('gate_type', 'free')
        idx = self.cb_type.findText(current_type)
        if idx >= 0:
            self.cb_type.setCurrentIndex(idx)
        self.cb_type.currentTextChanged.connect(self._on_type_changed)
        form.addRow("Gate type", self.cb_type)

        self.lbl_type_change_notice = QLabel("")
        self.lbl_type_change_notice.setStyleSheet("color: #b06000; font-style: italic;")
        self.lbl_type_change_notice.setWordWrap(True)
        form.addRow("", self.lbl_type_change_notice)

        # --- Axis channels ---
        # Channel pool: by default fluorescence + scatter (so FSC/SSC
        # gates work), but the user can include all channels via the
        # checkbox below.
        self.cb_include_all_channels = QCheckBox("Show all channels (including scatter)")
        self.cb_include_all_channels.setChecked(True)
        self.cb_include_all_channels.toggled.connect(self._populate_axis_combos)

        self.cb_marker_x = QComboBox()
        self.cb_marker_y = QComboBox()
        self._populate_axis_combos()

        form.addRow("X-axis channel", self.cb_marker_x)
        form.addRow("Y-axis channel", self.cb_marker_y)
        form.addRow("", self.cb_include_all_channels)

        # --- Parent gate and population ---
        self.cb_parent_gate = QComboBox()
        self.cb_parent_gate.addItem("(root)")
        for g in self.state.gate_definitions:
            name = g.get('gate_name')
            if name and name != self.gate_def.get('gate_name'):
                self.cb_parent_gate.addItem(name)
        parent_name = self.gate_def.get('parent_gate')
        if parent_name:
            idx = self.cb_parent_gate.findText(parent_name)
            if idx >= 0:
                self.cb_parent_gate.setCurrentIndex(idx)
        self.cb_parent_gate.currentTextChanged.connect(self._populate_parent_popul_combo)
        form.addRow("Parent gate", self.cb_parent_gate)

        self.cb_parent_popul = QComboBox()
        self._populate_parent_popul_combo()
        form.addRow("Parent population", self.cb_parent_popul)

        # --- Algorithm ---
        self.cb_algorithm = QComboBox()
        self._populate_algorithm_combo()
        self.cb_type.currentTextChanged.connect(self._populate_algorithm_combo)
        form.addRow("Algorithm", self.cb_algorithm)

        # --- Origin gate (only meaningful for replicate type) ---
        self.cb_origin_gate = QComboBox()
        self.cb_origin_gate.addItem("(none)")
        for g in self.state.gate_definitions:
            name = g.get('gate_name')
            if name and name != self.gate_def.get('gate_name'):
                self.cb_origin_gate.addItem(name)
        origin = self.gate_def.get('origin_gate')
        if origin:
            idx = self.cb_origin_gate.findText(origin)
            if idx >= 0:
                self.cb_origin_gate.setCurrentIndex(idx)
        self.cb_origin_gate.setEnabled(self.cb_type.currentText() == 'replicate')
        self.cb_type.currentTextChanged.connect(
            lambda t: self.cb_origin_gate.setEnabled(t == 'replicate')
        )
        form.addRow("Origin gate (replicate)", self.cb_origin_gate)

        outer.addLayout(form)

        # --- Populations editor ---
        pop_box = QGroupBox("Populations")
        pop_layout = QVBoxLayout(pop_box)

        self.tbl_populations = QTableWidget(0, 3)
        self.tbl_populations.setHorizontalHeaderLabels(["Name", "Label", "Position"])
        self.tbl_populations.horizontalHeader().setStretchLastSection(False)
        self.tbl_populations.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.tbl_populations.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.tbl_populations.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        pop_layout.addWidget(self.tbl_populations)

        pop_btn_row = QHBoxLayout()
        self.btn_pop_add    = QPushButton("Add row")
        self.btn_pop_remove = QPushButton("Remove selected")
        self.btn_pop_reset  = QPushButton("Reset to type default")
        self.btn_pop_add.clicked.connect(self._add_population_row)
        self.btn_pop_remove.clicked.connect(self._remove_selected_population_row)
        self.btn_pop_reset.clicked.connect(self._reset_populations_to_default)
        pop_btn_row.addWidget(self.btn_pop_add)
        pop_btn_row.addWidget(self.btn_pop_remove)
        pop_btn_row.addWidget(self.btn_pop_reset)
        pop_btn_row.addStretch()
        pop_layout.addLayout(pop_btn_row)

        outer.addWidget(pop_box)

        # --- OK / Cancel ---
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        # Initial population of the table.
        self._refresh_population_table()
        # Apply enable/disable rules that depend on the initial gate type
        # (e.g. greying out the Y combo for 1dsep / replicate).
        self._on_type_changed(self.cb_type.currentText())

    # ------------------------------------------------------------------
    # Helpers for populating combos / the populations table
    # ------------------------------------------------------------------

    def _populate_axis_combos(self):
        """Re-populate X and Y axis combos based on the current filter.

        Each combo stores the bare channel name as ``itemData`` and
        shows "channel — antigen" as the visible text.  When repopulating
        we must compare against ``currentData`` (the channel) — not
        ``currentText`` (the display string) — or selections are lost
        whenever the scatter-channel filter is toggled.
        """
        include_scatter = self.cb_include_all_channels.isChecked()
        channels = _gateable_channels(self.controller, include_scatter=include_scatter)

        prev_x = self.cb_marker_x.currentData() or self.gate_def.get('gate_marker_x', '')
        prev_y = self.cb_marker_y.currentData() or self.gate_def.get('gate_marker_y') or ''

        self.cb_marker_x.blockSignals(True)
        self.cb_marker_y.blockSignals(True)
        try:
            self.cb_marker_x.clear()
            self.cb_marker_y.clear()
            # First Y-axis entry is the "(none)" sentinel for 1D gates.
            self.cb_marker_y.addItem("(none)", None)
            for ch in channels:
                antigen = _channel_to_antigen(self.controller, ch)
                display = ch if antigen == ch else f"{ch} — {antigen}"
                self.cb_marker_x.addItem(display, ch)
                self.cb_marker_y.addItem(display, ch)
            self._select_combo_by_data(self.cb_marker_x, prev_x)
            if prev_y:
                self._select_combo_by_data(self.cb_marker_y, prev_y)
        finally:
            self.cb_marker_x.blockSignals(False)
            self.cb_marker_y.blockSignals(False)

    @staticmethod
    def _select_combo_by_data(combo: QComboBox, value: str):
        """Set a combo's current index to the row whose itemData equals *value*."""
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                combo.setCurrentIndex(i)
                return

    def _populate_parent_popul_combo(self):
        """Populate the parent-population combo from the selected parent gate."""
        self.cb_parent_popul.clear()
        parent_name = self.cb_parent_gate.currentText()
        if parent_name == "(root)":
            self.cb_parent_popul.addItem("root")
            return
        parent = self.state.gate_by_name(parent_name)
        if parent is None:
            self.cb_parent_popul.addItem("(none)")
            return
        pops = list((parent.get('populations') or {}).keys())
        if not pops:
            self.cb_parent_popul.addItem("(none)")
            return
        self.cb_parent_popul.addItems(pops)
        # Restore the persisted parent_popul if it still exists.
        existing = self.gate_def.get('parent_popul')
        if existing:
            idx = self.cb_parent_popul.findText(existing)
            if idx >= 0:
                self.cb_parent_popul.setCurrentIndex(idx)

    def _populate_algorithm_combo(self):
        """Populate the algorithm combo from the current gate type."""
        gate_type = self.cb_type.currentText()
        algos = ALGORITHMS_BY_GATE_TYPE.get(gate_type, [])
        prev = self.cb_algorithm.currentText() or self.gate_def.get('algorithm', '')
        self.cb_algorithm.blockSignals(True)
        try:
            self.cb_algorithm.clear()
            self.cb_algorithm.addItems(algos)
            if prev in algos:
                self.cb_algorithm.setCurrentText(prev)
            else:
                # Pick the type's default if the previous choice isn't valid.
                default = _default_algorithm_for_type(gate_type)
                if default in algos:
                    self.cb_algorithm.setCurrentText(default)
        finally:
            self.cb_algorithm.blockSignals(False)

    def _on_type_changed(self, new_type: str):
        """Adjust dependent fields when the gate type changes.

        Population *names* are algorithm output — 1dsep always yields
        neg/pos, 2dsep always yields DN/X+/Y+/DP, etc. (see
        `_default_populations_for_type`).  They are not free-form user
        data, so when the gate type changes we must reset the
        populations table to the new type's default set.  Leaving stale
        population keys in place (e.g. 'DN'/'X+'/'Y+'/'DP' surviving a
        2dsep -> 1dsep change) causes child gates whose `parent_popul`
        references one of those keys to silently fail lookup and fall
        back to the unfiltered parent array — exactly the bug that
        produced bogus CD45- counts in 'live T cells'.

        We only reset when the *previous* type's default population set
        no longer matches what's in the table (so we don't clobber a
        user's custom label/position edits on a no-op type "change",
        e.g. re-selecting the same type from the algorithm-recommendation
        flow). On every actual gate_type change the table is reset to
        the canonical default for the new type; if the user had manually
        renamed populations they must redo that after switching types.
        """
        # Y-axis combo is disabled only for replicate gates (no display axes).
        # 1dsep gates can carry a display Y channel so the tile shows a biplot
        # background rather than a histogram-only view.
        is_2d_like = new_type in ('1dsep', '2dsep', 'free', 'singlets')
        self.cb_marker_y.setEnabled(is_2d_like)

        if new_type == self._last_seen_type:
            return
        self._last_seen_type = new_type

        default_pops = _default_populations_for_type(new_type)
        current_pops = self._read_populations_from_table()
        if set(current_pops.keys()) != set(default_pops.keys()):
            self._working_populations = default_pops
            self._refresh_population_table()
            pop_list = ', '.join(default_pops.keys()) or '(none)'
            self.lbl_type_change_notice.setText(
                f"Populations reset to default for '{new_type}': {pop_list}. "
                f"Re-apply any custom labels if needed."
            )
        else:
            self.lbl_type_change_notice.setText("")

    # ------------------------------------------------------------------
    # Populations table helpers
    # ------------------------------------------------------------------

    def _refresh_population_table(self):
        """Repopulate the populations table from `self._working_populations`."""
        self.tbl_populations.setRowCount(0)
        for name, meta in self._working_populations.items():
            self._add_population_row(name=name,
                                     label=meta.get('label', name),
                                     pos=meta.get('label_pos', 1),
                                     region=meta.get('region'))

    def _add_population_row(self, name: str | None = None,
                            label: str | None = None,
                            pos: int = 1, region: str | None = None):
        """Append one row to the populations table.

        When called from the button the arguments default to a blank
        row that the user fills in.  When called from
        `_refresh_population_table` they carry the existing values.
        Qt connects the QPushButton clicked signal with a bool argument,
        which we deliberately ignore by accepting `name=None`.

        *region* (which side of the thresholds the population is on) is
        kept on the row, so renaming a population keeps its region.
        """
        # Treat the bool-from-clicked case as "no name supplied".
        if isinstance(name, bool) or name is None:
            name = ""
        if label is None:
            label = ""

        row = self.tbl_populations.rowCount()
        self.tbl_populations.insertRow(row)
        name_item = QTableWidgetItem(str(name))
        name_item.setData(Qt.UserRole, region)
        self.tbl_populations.setItem(row, 0, name_item)
        self.tbl_populations.setItem(row, 1, QTableWidgetItem(str(label)))
        sb = QSpinBox()
        sb.setRange(1, 8)
        sb.setValue(int(pos))
        self.tbl_populations.setCellWidget(row, 2, sb)

    def _remove_selected_population_row(self):
        """Remove the highlighted row(s) from the populations table."""
        rows = sorted(
            {idx.row() for idx in self.tbl_populations.selectedIndexes()},
            reverse=True,
        )
        for r in rows:
            self.tbl_populations.removeRow(r)

    def _reset_populations_to_default(self):
        """Reset the table to the default populations for the current type."""
        gate_type = self.cb_type.currentText()
        self._working_populations = _default_populations_for_type(gate_type)
        self._refresh_population_table()

    def _read_populations_from_table(self) -> dict:
        """Read the populations table back into a dict (rejecting empty names)."""
        out: dict = {}
        for row in range(self.tbl_populations.rowCount()):
            name_item  = self.tbl_populations.item(row, 0)
            label_item = self.tbl_populations.item(row, 1)
            pos_widget = self.tbl_populations.cellWidget(row, 2)
            name  = (name_item.text() if name_item else "").strip()
            label = (label_item.text() if label_item else "").strip() or name
            pos   = int(pos_widget.value()) if isinstance(pos_widget, QSpinBox) else 1
            if not name:
                continue
            out[name] = {'label': label, 'label_pos': pos}
            region = name_item.data(Qt.UserRole) if name_item else None
            if region:
                out[name]['region'] = region
        return out

    # ------------------------------------------------------------------
    # Accept — validate, then write everything back into self.gate_def
    # ------------------------------------------------------------------

    def accept(self):
        """Validate the form, write back into ``self.gate_def``, then close."""
        name = self.le_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation", "Gate name is required.")
            return

        # No name collisions with other gates.
        for g in self.state.gate_definitions:
            if g is self.gate_def:
                continue
            if g.get('gate_name') == name:
                QMessageBox.warning(
                    self, "Validation",
                    f"Another gate is already named '{name}'."
                )
                return

        gate_type = self.cb_type.currentText()
        marker_x = self.cb_marker_x.currentData() or ''
        # The "(none)" sentinel in the Y combo has itemData=None, so
        # currentData() returns None directly when no Y channel is set.
        marker_y = self.cb_marker_y.currentData() or None

        if gate_type != 'replicate' and not marker_x:
            QMessageBox.warning(self, "Validation", "X-axis channel is required.")
            return
        if gate_type in ('2dsep', 'free', 'singlets') and not marker_y:
            QMessageBox.warning(
                self, "Validation",
                f"{gate_type} gates require a Y-axis channel."
            )
            return

        parent_text = self.cb_parent_gate.currentText()
        parent_gate = None if parent_text == "(root)" else parent_text

        parent_popul = self.cb_parent_popul.currentText() or 'root'
        if parent_popul in ("(none)",):
            parent_popul = 'root'

        # Validate parent_popul against the parent's *current* populations.
        # The combo is repopulated whenever parent_gate changes
        # (_populate_parent_popul_combo), but if the parent's populations
        # were edited or reset (e.g. via the type-change reset in
        # _on_type_changed) after this combo was last populated, a stale
        # selection can persist. A silent fallback to the full parent
        # array — rather than an explicit validation error — is what
        # produced the bogus CD45- counts in 'live T cells' against
        # 'CD3+', which no longer existed once CD3's populations were
        # named 'X+'/'X-'. Re-check here, at save time, against the
        # live parent definition rather than trusting the combo.
        if parent_gate is not None:
            parent_def = self.state.gate_by_name(parent_gate)
            valid_pops = set((parent_def.get('populations') or {}).keys()) if parent_def else set()
            if valid_pops and parent_popul not in valid_pops:
                QMessageBox.warning(
                    self, "Validation",
                    f"Parent population '{parent_popul}' does not exist on "
                    f"gate '{parent_gate}'. Available populations: "
                    f"{', '.join(sorted(valid_pops))}. Please reselect."
                )
                return

        algorithm = self.cb_algorithm.currentText() or _default_algorithm_for_type(gate_type)

        populations = self._read_populations_from_table()
        if gate_type != 'replicate' and not populations:
            QMessageBox.warning(
                self, "Validation",
                "Define at least one population (use 'Reset to type default' if unsure)."
            )
            return

        origin_text = self.cb_origin_gate.currentText()
        origin_gate = None if origin_text in ("(none)", "") else origin_text
        if gate_type == 'replicate' and not origin_gate:
            QMessageBox.warning(
                self, "Validation",
                "Replicate gates require an origin gate."
            )
            return

        # Write back into the supplied dict.  Caller takes ownership.
        self.gate_def['gate_name']     = name
        self.gate_def['gate_type']     = gate_type
        self.gate_def['gate_marker_x'] = marker_x
        self.gate_def['gate_marker_y'] = marker_y
        self.gate_def['parent_gate']   = parent_gate
        self.gate_def['parent_popul']  = parent_popul
        self.gate_def['algorithm']     = algorithm
        self.gate_def['populations']   = populations
        self.gate_def['origin_gate']   = origin_gate
        # Initialise the substructure dicts if missing (they may be
        # filled later by `GateParamDialog` and the train worker).
        self.gate_def.setdefault('gate_param', {})
        self.gate_def.setdefault('stats_parent', {})

        super().accept()

# ---------------------------------------------------------------------------
# Hierarchy tab
# ---------------------------------------------------------------------------

class _RecommendWorker(QThread):
    """Load one sample and recommend an algorithm per gate (see
    HierarchyTab._recommend_algorithms)."""

    finished = Signal(bool, str, object)   # ok, message, {gate_name: algorithm}

    def __init__(self, experiment_dir, sample_key, snapshot, gate_defs, channels,
                 transforms, parent=None):
        super().__init__(parent)
        self._experiment_dir = experiment_dir
        self._key = sample_key
        self._snap = snapshot
        self._gate_defs = gate_defs
        self._channels = channels
        self._transforms = transforms

    def run(self):
        try:
            out = load_unmixed(self._experiment_dir, self._key, self._snap,
                               max_events=DEFAULT_EVENTS_PER_SAMPLE, seed=TRAINING_SEED)
            data = ag_core.transform_columns(out['unmixed'], self._channels, self._transforms)
            idx = {ch: i for i, ch in enumerate(self._channels)}
            trained = ag_core.train_boundaries(self._gate_defs, data, idx)
            recs = {}
            for g in self._gate_defs:
                parent = trained.parent_masks.get(g['gate_name'])
                sub = data[parent] if parent is not None else data
                recs[g['gate_name']] = ag_core.recommend_algorithm(g, sub, idx)
            self.finished.emit(True, '', recs)
        except Exception as exc:
            log.exception("algorithm recommendation failed")
            self.finished.emit(False, str(exc), {})


class HierarchyTab(QWidget):
    """Tab 0 — Gate Hierarchy.

    Layout
    ------
    Horizontal QSplitter:
      • Left  — control bar (import / refresh / add / advanced / delete /
                recommend) above a QTreeWidget of every gate definition.
      • Right — preview of the selected gate: the displayed sample's
                density with the source gate or trained boundary on top.

    Tree columns:
        #, Gate name, Type, X axis, Y axis, Parent gate, Parent population,
        Algorithm, Status

    Status badges
    -------------
        - "Trained"   — green   — a trained boundary exists
        - "Untrained" — amber   — the gate has not been trained
        - "Stale"     — red     — the gate uses a channel that is not in
                                  the current experiment
    """

    # Tree column indices.  Defining them as class constants keeps the
    # methods that read/write the tree readable.
    COL_NUMBER     = 0
    COL_NAME       = 1
    COL_TYPE       = 2
    COL_X          = 3
    COL_Y          = 4
    COL_PARENT     = 5
    COL_PARENT_POP = 6
    COL_ALGORITHM  = 7
    COL_STATUS     = 8

    _COLUMN_LABELS = [
        "#", "Gate name", "Type", "X axis", "Y axis",
        "Parent gate", "Parent population", "Algorithm", "Status",
    ]

    def __init__(self, state: GatingModelState, bus, controller, parent=None):
        super().__init__(parent)
        self.state = state
        self.bus = bus
        self.controller = controller

        self._recommend_worker: _RecommendWorker | None = None

        # Whether the tree's itemChanged signal should be acted on.
        # We block it during programmatic repopulation so editing
        # state doesn't trigger a feedback loop.
        self._suppress_item_changed = False

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Horizontal splitter: tree+controls on left, preview on right.
        splitter = QSplitter(Qt.Horizontal)

        # ----- Left side -----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(HelpToggleWidget(text=ag_help_texts.BUILDER_HIERARCHY))

        # Control bar.
        ctrl_row = QHBoxLayout()
        self.btn_import   = QPushButton("Import from Hierarchy")
        self.btn_refresh  = QPushButton("Refresh from Hierarchy")
        self.btn_add      = QPushButton("Add gate manually")
        self.btn_advanced = QPushButton("Advanced…")
        self.btn_delete   = QPushButton("Delete")

        self.btn_import.setToolTip(
            "Snapshot the current controller.unmixed_gating into the model. "
            "Replaces any existing gates in the model."
        )
        self.btn_refresh.setToolTip(
            "Merge new gates from controller.unmixed_gating into the model. "
            "Existing gate_param overrides are preserved. Gates removed "
            "from the source hierarchy are marked Stale."
        )

        self.btn_import.clicked.connect(self._on_import_clicked)
        self.btn_refresh.clicked.connect(self._on_refresh_clicked)
        self.btn_add.clicked.connect(self._on_add_clicked)
        self.btn_advanced.clicked.connect(self._on_advanced_clicked)
        self.btn_delete.clicked.connect(self._on_delete_clicked)

        for b in (self.btn_import, self.btn_refresh, self.btn_add,
                  self.btn_advanced, self.btn_delete):
            ctrl_row.addWidget(b)
        ctrl_row.addStretch()
        left_layout.addLayout(ctrl_row)

        # Tree widget.
        self.tree = QTreeWidget()
        self.tree.setColumnCount(len(self._COLUMN_LABELS))
        self.tree.setHeaderLabels(self._COLUMN_LABELS)
        self.tree.setRootIsDecorated(False)            # flat list, not nested
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tree.setAlternatingRowColors(True)
        # Inline editing is opt-in: F2 starts an inline editor on the
        # current cell, double-click is reserved for opening the rich
        # GateEditDialog instead.  Without this, Qt would auto-edit on
        # double-click and our itemDoubleClicked signal would race
        # against the inline editor.
        self.tree.setEditTriggers(QAbstractItemView.EditKeyPressed)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree.itemChanged.connect(self._on_item_changed)

        # Sensible default column widths.
        self.tree.header().setSectionResizeMode(QHeaderView.Interactive)
        self.tree.setColumnWidth(self.COL_NUMBER,     32)
        self.tree.setColumnWidth(self.COL_NAME,       140)
        self.tree.setColumnWidth(self.COL_TYPE,       80)
        self.tree.setColumnWidth(self.COL_X,          110)
        self.tree.setColumnWidth(self.COL_Y,          110)
        self.tree.setColumnWidth(self.COL_PARENT,     110)
        self.tree.setColumnWidth(self.COL_PARENT_POP, 110)
        self.tree.setColumnWidth(self.COL_ALGORITHM,  100)
        self.tree.setColumnWidth(self.COL_STATUS,     90)
        left_layout.addWidget(self.tree)

        # Help / status text just under the tree.
        self.lbl_summary = QLabel("No gates loaded.")
        self.lbl_summary.setStyleSheet("color: #555; font-style: italic;")
        left_layout.addWidget(self.lbl_summary)

        splitter.addWidget(left)

        # ----- Right side: preview plot -----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)

        self.lbl_preview_title = QLabel("<i>Select a gate to preview its geometry.</i>")
        self.lbl_preview_title.setWordWrap(True)
        right_layout.addWidget(self.lbl_preview_title)

        # Proper Honeychrome plot widget: TransparentGraphicsLayoutWidget +
        # NoPanViewBox + ZoomAxis — matches CytometryPlotWidget and BiplotTile.
        self._preview_gw = TransparentGraphicsLayoutWidget()
        self._preview_gw.setMinimumSize(280, 280)
        self._preview_gw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gl = self._preview_gw.ci.layout
        gl.setHorizontalSpacing(0)
        gl.setVerticalSpacing(0)

        self._preview_vb = NoPanViewBox()
        self._preview_vb.setMouseEnabled(x=False, y=False)
        self._preview_vb.raiseContextMenu = lambda ev: None
        # CytometryPlotWidget layout: vb at (1,2), axis_left at (1,1), axis_bottom at (2,2)
        self._preview_gw.addItem(self._preview_vb,     row=1, col=2)

        self._preview_axis_x = ZoomAxis('bottom', self._preview_vb)
        self._preview_axis_y = ZoomAxis('left',   self._preview_vb)
        self._preview_gw.addItem(self._preview_axis_y, row=1, col=1)
        self._preview_gw.addItem(self._preview_axis_x, row=2, col=2)
        self._preview_axis_x.linkToView(self._preview_vb)
        self._preview_axis_y.linkToView(self._preview_vb)
        # No separate LabelItems — labels set via axis.setLabel() in _update_preview.

        # ImageItem for 2D density heatmap (colourmap set in _update_preview)
        self._preview_img = pg.ImageItem()
        self._preview_vb.addItem(self._preview_img)

        # PlotDataItem for 1D histogram
        self._preview_hist = pg.PlotDataItem(
            stepMode='center', fillLevel=0,
            brush=(100, 100, 250, 150),
        )
        self._preview_vb.addItem(self._preview_hist)

        right_layout.addWidget(self._preview_gw, stretch=1)

        # Per-gate "Advanced…" shortcut also lives on the right pane.
        right_btn_row = QHBoxLayout()
        self.btn_edit_selected = QPushButton("Edit selected gate…")
        self.btn_edit_selected.clicked.connect(self._on_edit_selected_clicked)
        right_btn_row.addWidget(self.btn_edit_selected)
        right_btn_row.addStretch()
        right_layout.addLayout(right_btn_row)

        splitter.addWidget(right)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

        # Initial population (if state already has gates from QSettings).
        self._repopulate_tree()

    # ------------------------------------------------------------------
    # Public refresh hook — called by PluginWidget on tab activation.
    # ------------------------------------------------------------------

    def refresh(self):
        """Re-evaluate channel alignment and repopulate the tree.

        Called by the outer PluginWidget whenever the Hierarchy tab
        becomes active.  This is a cheap operation; it does not load
        any sample data.
        """
        self._repopulate_tree()

    # ------------------------------------------------------------------
    # Tree population and serialisation
    # ------------------------------------------------------------------

    def _repopulate_tree(self):
        """Rebuild every row of the tree from `state.gate_definitions`."""
        self._suppress_item_changed = True
        try:
            self.tree.clear()
            experiment_channels = set(_experiment_channels(self.controller))

            for g in self._sorted_gate_defs():
                item = self._build_tree_item(g, experiment_channels)
                self.tree.addTopLevelItem(item)
        finally:
            self._suppress_item_changed = False

        n = self.state.n_gates()
        if n == 0:
            self.lbl_summary.setText("No gates loaded — click 'Import from Hierarchy'.")
        else:
            trained = sum(1 for g in self.state.gate_definitions
                          if self.state.is_trained(g.get('gate_name', '')))
            self.lbl_summary.setText(
                f"{n} gate{'s' if n != 1 else ''} in model; "
                f"{trained} trained, {n - trained} untrained."
            )

        self._update_preview()

    def _sorted_gate_defs(self) -> list[dict]:
        """Return `gate_definitions` sorted by `gate_number`, then by name."""
        return sorted(
            self.state.gate_definitions,
            key=lambda g: (int(g.get('gate_number', 999)), g.get('gate_name', '')),
        )

    def _build_tree_item(self, g: dict, experiment_channels: set) -> QTreeWidgetItem:
        """Construct one QTreeWidgetItem from a GateDefinition."""
        gate_name_for_item = g.get('gate_name', '')
        rec = self.state.algorithm_recommendations.get(gate_name_for_item, '')
        algo_display = g.get('algorithm', '') or ''
        if rec and rec != algo_display:
            algo_display = f"{algo_display} [rec: {rec}]" if algo_display else f"[rec: {rec}]"

        item = QTreeWidgetItem([
            str(g.get('gate_number', '')),
            str(gate_name_for_item),
            str(g.get('gate_type', '')),
            str(g.get('gate_marker_x', '') or ''),
            str(g.get('gate_marker_y') or ''),
            str(g.get('parent_gate') or 'root'),
            str(g.get('parent_popul', '') or ''),
            algo_display,
            '',  # Status — filled by _status_for()
        ])

        # Make the name / number columns editable inline.  Other
        # columns require richer pickers and are routed through the
        # dialog on double-click.
        item.setFlags(item.flags() | Qt.ItemIsEditable)

        # Determine status — Stale if any axis channel is missing
        # from the current experiment.
        x = g.get('gate_marker_x') or ''
        y = g.get('gate_marker_y') or ''
        is_stale = bool(experiment_channels) and (
            (x and x not in experiment_channels) or
            (y and y not in experiment_channels)
        )
        status_text, status_brush, row_tint = self._status_for(g, is_stale)
        item.setText(self.COL_STATUS, status_text)
        item.setForeground(self.COL_STATUS, status_brush)
        if row_tint is not None:
            for col in range(self.tree.columnCount()):
                item.setBackground(col, row_tint)

        # Stash the gate_name on the item so context-menu / double-click
        # handlers can find the matching dict without a string compare
        # against a mutable column.
        item.setData(0, Qt.UserRole, g.get('gate_name', ''))
        return item

    def _status_for(self, g: dict, is_stale: bool):
        """Return (text, foreground brush, optional background tint) for status column."""
        if is_stale:
            return ("Stale", QBrush(QColor('#a00000')), QColor(255, 235, 235))
        if self.state.is_trained(g.get('gate_name', '')):
            return ("Trained", QBrush(QColor('#0a7a23')), None)
        return ("Untrained", QBrush(QColor('#946100')), None)

    # ------------------------------------------------------------------
    # Selection / inline edit / double-click handlers
    # ------------------------------------------------------------------

    def _selected_gate_name(self) -> str | None:
        items = self.tree.selectedItems()
        if not items:
            return None
        return items[0].data(0, Qt.UserRole)

    def _selected_gate(self) -> dict | None:
        name = self._selected_gate_name()
        if not name:
            return None
        return self.state.gate_by_name(name)

    def _on_selection_changed(self):
        """Update the preview pane to reflect the new selection."""
        self._update_preview()

    def _on_item_changed(self, item: QTreeWidgetItem, column: int):
        """Apply an inline edit to the underlying GateDefinition.

        Only the # (gate_number) and Gate name columns accept inline
        edits; everything else requires the dialog because the valid
        values are constrained (combos).
        """
        if self._suppress_item_changed:
            return

        gate_name_key = item.data(0, Qt.UserRole)
        g = self.state.gate_by_name(gate_name_key) if gate_name_key else None
        if g is None:
            return

        new_value = item.text(column).strip()

        if column == self.COL_NUMBER:
            try:
                g['gate_number'] = int(new_value)
            except ValueError:
                # Revert the cell text.
                self._suppress_item_changed = True
                try:
                    item.setText(self.COL_NUMBER, str(g.get('gate_number', 0)))
                finally:
                    self._suppress_item_changed = False
            else:
                # Re-sort the tree by gate_number.
                self._repopulate_tree()
        elif column == self.COL_NAME:
            if not new_value:
                self._suppress_item_changed = True
                try:
                    item.setText(self.COL_NAME, str(g.get('gate_name', '')))
                finally:
                    self._suppress_item_changed = False
                return
            # Reject duplicates.
            if any(other is not g and other.get('gate_name') == new_value
                   for other in self.state.gate_definitions):
                QMessageBox.warning(
                    self, "Rename gate",
                    f"Another gate is already named '{new_value}'."
                )
                self._suppress_item_changed = True
                try:
                    item.setText(self.COL_NAME, str(g.get('gate_name', '')))
                finally:
                    self._suppress_item_changed = False
                return
            old_name = g.get('gate_name', '')
            g['gate_name'] = new_value
            # Update any references in other gates and in trained_boundaries.
            self._rename_gate_references(old_name, new_value)
            item.setData(0, Qt.UserRole, new_value)
        else:
            # Other columns are not editable inline — revert.  The
            # double-click handler routes the user to GateEditDialog
            # instead, where every column can be edited under combo
            # control.
            self._suppress_item_changed = True
            try:
                item.setText(column, self._cell_value_for(g, column))
            finally:
                self._suppress_item_changed = False

    def _cell_value_for(self, g: dict, column: int) -> str:
        if column == self.COL_TYPE:        return str(g.get('gate_type', ''))
        if column == self.COL_X:           return str(g.get('gate_marker_x', '') or '')
        if column == self.COL_Y:           return str(g.get('gate_marker_y') or '')
        if column == self.COL_PARENT:      return str(g.get('parent_gate') or 'root')
        if column == self.COL_PARENT_POP:  return str(g.get('parent_popul', '') or '')
        if column == self.COL_ALGORITHM:   return str(g.get('algorithm', '') or '')
        return ""

    def _rename_gate_references(self, old: str, new: str):
        """Update parent_gate / origin_gate / trained_boundaries on rename."""
        for other in self.state.gate_definitions:
            if other.get('parent_gate') == old:
                other['parent_gate'] = new
            if other.get('origin_gate') == old:
                other['origin_gate'] = new
        if old in self.state.trained_boundaries:
            self.state.trained_boundaries[new] = self.state.trained_boundaries.pop(old)
        if old in self.state.algorithm_recommendations:
            self.state.algorithm_recommendations[new] = (
                self.state.algorithm_recommendations.pop(old)
            )

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Open the full edit dialog when any cell is double-clicked.

        Inline editing of the # and Gate name columns is reachable via
        F2 (see `setEditTriggers` in `_build_ui`).  Double-click always
        opens the rich dialog so the user can change combos and the
        populations table in one place.
        """
        self._edit_gate(item.data(0, Qt.UserRole))

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------

    def _on_context_menu(self, pos):
        """Show the right-click menu over the tree."""
        item = self.tree.itemAt(pos)
        menu = QMenu(self.tree)

        act_edit  = QAction("Edit…", menu)
        act_dup   = QAction("Duplicate", menu)
        act_del   = QAction("Delete", menu)
        act_child = QAction("Add child gate…", menu)
        act_param = QAction("Advanced parameters…", menu)
        act_split = QAction("Split into 1D gates…", menu)

        for a in (act_edit, act_dup, act_del, act_child, act_param, act_split):
            a.setEnabled(item is not None)

        if item is not None:
            gate_name = item.data(0, Qt.UserRole)
            g = self.state.gate_by_name(gate_name)
            act_split.setEnabled(g is not None and g.get('gate_type') == '2dsep')

            act_edit.triggered.connect(lambda: self._edit_gate(gate_name))
            act_dup.triggered.connect(lambda: self._duplicate_gate(gate_name))
            act_del.triggered.connect(lambda: self._delete_gate(gate_name))
            act_child.triggered.connect(lambda: self._add_child_gate(gate_name))
            act_param.triggered.connect(lambda: self._open_param_dialog(gate_name))
            act_split.triggered.connect(lambda: self._split_2dsep_gate(gate_name))

        for a in (act_edit, act_dup, act_del, act_child, act_param):
            menu.addAction(a)
        menu.addSeparator()
        menu.addAction(act_split)

        menu.exec(self.tree.viewport().mapToGlobal(pos))

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_import_clicked(self):
        """Snapshot `controller.unmixed_gating` into the model.

        Confirms with the user if there are existing gates to avoid
        destroying user edits accidentally.
        """
        if self.state.gate_definitions:
            ans = QMessageBox.question(
                self, "Import from Hierarchy",
                "This will replace the current set of gates with a fresh "
                "snapshot of controller.unmixed_gating.\n\n"
                "Use 'Refresh from Hierarchy' instead to merge new gates "
                "without losing your edits.\n\n"
                "Continue with full replacement?",
            )
            if ans != QMessageBox.Yes:
                return
        n = self._import_from_unmixed_gating(replace=True)
        self._repopulate_tree()
        if n == 0:
            self.bus.statusMessage.emit(
                "[Gating Model Builder] No gates found in controller.unmixed_gating."
            )
        else:
            self.bus.statusMessage.emit(
                f"[Gating Model Builder] Imported {n} gate{'s' if n != 1 else ''} "
                "from hierarchy."
            )
        self._recommend_algorithms()

    def _on_refresh_clicked(self):
        added, stale = self._merge_from_unmixed_gating()
        self._repopulate_tree()
        msg = (
            f"[Gating Model Builder] Refresh: added {added}, "
            f"stale {stale}."
        )
        self.bus.statusMessage.emit(msg)
        self._recommend_algorithms()

    def _on_add_clicked(self):
        """Create a fresh GateDefinition and open the edit dialog on it."""
        # Default new gate placeholder.
        new_def = self._make_blank_gate_def()
        dlg = GateEditDialog(new_def, self.state, self.controller,
                             is_new=True, parent=self)
        if dlg.exec() == QDialog.Accepted:
            self.state.gate_definitions.append(new_def)
            self._repopulate_tree()
            self._select_gate_by_name(new_def['gate_name'])

    def _on_advanced_clicked(self):
        name = self._selected_gate_name()
        if not name:
            QMessageBox.information(self, "Advanced parameters",
                                    "Select a gate first.")
            return
        self._open_param_dialog(name)

    def _on_delete_clicked(self):
        name = self._selected_gate_name()
        if not name:
            return
        self._delete_gate(name)

    def _on_edit_selected_clicked(self):
        name = self._selected_gate_name()
        if not name:
            QMessageBox.information(self, "Edit gate", "Select a gate first.")
            return
        self._edit_gate(name)

    # ------------------------------------------------------------------
    # Mutating helpers — edit / duplicate / delete / add child
    # ------------------------------------------------------------------

    def _edit_gate(self, gate_name: str):
        g = self.state.gate_by_name(gate_name)
        if g is None:
            return
        old_name = g.get('gate_name', '')
        dlg = GateEditDialog(g, self.state, self.controller,
                             is_new=False, parent=self)
        if dlg.exec() == QDialog.Accepted:
            new_name = g.get('gate_name', '')
            if new_name and new_name != old_name:
                # The dialog only writes self.gate_def['gate_name']; it has
                # no visibility into sibling gates' parent_gate/origin_gate
                # references or trained_boundaries/algorithm_recommendations
                # keys, so the rename must be cascaded here exactly as the
                # tree's inline F2-rename path already does.
                self._rename_gate_references(old_name, new_name)
            self._repopulate_tree()
            self._select_gate_by_name(g['gate_name'])

    def _duplicate_gate(self, gate_name: str):
        g = self.state.gate_by_name(gate_name)
        if g is None:
            return
        copy = deepcopy(g)
        # Choose a non-clashing new name.
        base = (g.get('gate_name') or 'gate') + '_copy'
        candidate = base
        i = 2
        while any(o.get('gate_name') == candidate
                  for o in self.state.gate_definitions):
            candidate = f"{base}{i}"
            i += 1
        copy['gate_name'] = candidate
        copy['gate_number'] = max(
            (int(o.get('gate_number', 0)) for o in self.state.gate_definitions),
            default=0,
        ) + 1
        # Don't carry a stale trained_boundary reference into the copy;
        # the new gate is untrained until the user runs Train.
        self.state.gate_definitions.append(copy)
        self._repopulate_tree()
        self._select_gate_by_name(candidate)

    def _delete_gate(self, gate_name: str):
        g = self.state.gate_by_name(gate_name)
        if g is None:
            return
        # Refuse to delete if other gates list this one as parent / origin.
        dependents = [
            o.get('gate_name')
            for o in self.state.gate_definitions
            if o is not g and (
                o.get('parent_gate') == gate_name
                or o.get('origin_gate') == gate_name
            )
        ]
        if dependents:
            QMessageBox.warning(
                self, "Cannot delete",
                f"Other gates depend on '{gate_name}':\n  "
                + ', '.join(dependents)
                + "\n\nRetarget or delete them first."
            )
            return
        ans = QMessageBox.question(
            self, "Delete gate",
            f"Delete '{gate_name}'? This cannot be undone.",
        )
        if ans != QMessageBox.Yes:
            return
        self.state.gate_definitions.remove(g)
        self.state.trained_boundaries.pop(gate_name, None)
        self.state.algorithm_recommendations.pop(gate_name, None)
        self._repopulate_tree()

    def _add_child_gate(self, parent_gate_name: str):
        """Add a new gate whose parent is *parent_gate_name*."""
        parent = self.state.gate_by_name(parent_gate_name)
        new_def = self._make_blank_gate_def()
        new_def['parent_gate'] = parent_gate_name
        # Default the parent population to the first one in the parent gate.
        if parent and parent.get('populations'):
            new_def['parent_popul'] = next(iter(parent['populations']))
        dlg = GateEditDialog(new_def, self.state, self.controller,
                             is_new=True, parent=self)
        if dlg.exec() == QDialog.Accepted:
            self.state.gate_definitions.append(new_def)
            self._repopulate_tree()
            self._select_gate_by_name(new_def['gate_name'])

    def _open_param_dialog(self, gate_name: str):
        g = self.state.gate_by_name(gate_name)
        if g is None:
            return
        dlg = GateParamDialog(g, parent=self)
        dlg.exec()

    def _make_blank_gate_def(self) -> dict:
        """Construct an empty GateDefinition with sensible default fields."""
        next_num = max(
            (int(o.get('gate_number', 0)) for o in self.state.gate_definitions),
            default=0,
        ) + 1
        return {
            'gate_number':   next_num,
            'gate_name':     f"gate_{next_num}",
            'gate_type':     'free',
            'gate_marker_x': '',
            'gate_marker_y': None,
            'parent_gate':   None,
            'parent_popul':  'root',
            'populations':   _default_populations_for_type('free'),
            'algorithm':     _default_algorithm_for_type('free'),
            'gate_param':    {},
            'stats_parent':  {},
            'origin_gate':   None,
        }

    def _select_gate_by_name(self, name: str):
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            if it.data(0, Qt.UserRole) == name:
                self.tree.setCurrentItem(it)
                return

    def _split_2dsep_gate(self, gate_name: str):
        """Replace a 2dsep gate with two chained 1dsep gates.

        Creates:
          • ``{name}_x`` — 1dsep on gate_marker_x, same parent/parent_popul
            as the original gate.
          • ``{name}_y`` — 1dsep on gate_marker_y, parent = ``{name}_x``,
            parent_popul = 'pos_x'.

        Child gates of the original are re-targeted by quadrant: children
        of the X+Y+ / X+Y− quadrants move to ``{name}_y`` 'pos_y' / 'neg_y';
        children of the X− quadrants move to ``{name}_x`` 'neg_x', which
        contains both.  Per-axis Advanced parameters (algorithm_x/y,
        tail_fraction_x/y, etc.) are promoted to the respective 1dsep
        gate_param dicts.  The original gate and its trained boundary are
        removed; the two new gates must be re-trained.
        """
        g = self.state.gate_by_name(gate_name)
        if g is None or g.get('gate_type') != '2dsep':
            return

        name_x = f"{gate_name}_x"
        name_y = f"{gate_name}_y"

        # Collision check.
        existing = {d.get('gate_name') for d in self.state.gate_definitions}
        collisions = [n for n in (name_x, name_y) if n in existing]
        if collisions:
            QMessageBox.warning(
                self, "Split gate",
                "Cannot split: gate name(s) already exist: "
                + ', '.join(f"'{c}'" for c in collisions)
                + "\n\nRename them first or rename the 2dsep gate."
            )
            return

        ans = QMessageBox.question(
            self, "Split into 1D gates",
            f"Split '{gate_name}' into two sequential 1dsep gates?\n\n"
            f"  • {name_x}  — 1dsep on {g.get('gate_marker_x', '?')}\n"
            f"  • {name_y}  — 1dsep on {g.get('gate_marker_y', '?')}, "
            f"child of {name_x}\n\n"
            "The original gate and its trained boundary will be removed.\n"
            f"Child gates of the X+ quadrants move under {name_y}; children "
            f"of the X− quadrants move under {name_x} (neg).",
        )
        if ans != QMessageBox.Yes:
            return

        orig_param = deepcopy(g.get('gate_param') or {})
        orig_num   = int(g.get('gate_number', 0))
        default_algo = g.get('algorithm', 'tail')

        def _param_for_axis(axis: str) -> tuple[str, dict]:
            """Return (algorithm, gate_param) for one axis of the split."""
            algo = orig_param.get(f'algorithm_{axis}') or default_algo
            p = {}
            p['quantile_trim'] = orig_param.get('quantile_trim', 0.001)
            p['tail_fraction'] = float(orig_param.get(f'tail_fraction_{axis}',
                                                       orig_param.get('tail_fraction', 0.01)))
            p['mixture_components'] = int(orig_param.get(f'mixture_components_{axis}',
                                                          orig_param.get('mixture_components', 2)))
            lo = orig_param.get(f'axis_lo_{axis}', 0.0)
            hi = orig_param.get(f'axis_hi_{axis}', 0.0)
            if lo != 0.0:
                p['axis_lo'] = float(lo)
            if hi != 0.0:
                p['axis_hi'] = float(hi)
            return algo, p

        algo_x, param_x = _param_for_axis('x')
        algo_y, param_y = _param_for_axis('y')

        gate_x = {
            'gate_number':   orig_num,
            'gate_name':     name_x,
            'gate_type':     '1dsep',
            'gate_marker_x': g.get('gate_marker_x', ''),
            'gate_marker_y': None,
            'parent_gate':   g.get('parent_gate'),
            'parent_popul':  g.get('parent_popul', 'root'),
            'populations':   {
                'neg_x': {'label': 'neg', 'label_pos': 1, 'region': 'below'},
                'pos_x': {'label': 'pos', 'label_pos': 2, 'region': 'above'},
            },
            'algorithm':     algo_x,
            'gate_param':    param_x,
            'stats_parent':  {},
            'origin_gate':   None,
        }
        gate_y = {
            'gate_number':   orig_num + 1,
            'gate_name':     name_y,
            'gate_type':     '1dsep',
            'gate_marker_x': g.get('gate_marker_y') or '',
            'gate_marker_y': None,
            'parent_gate':   name_x,
            'parent_popul':  'pos_x',
            'populations':   {
                'neg_y': {'label': 'neg', 'label_pos': 1, 'region': 'below'},
                'pos_y': {'label': 'pos', 'label_pos': 2, 'region': 'above'},
            },
            'algorithm':     algo_y,
            'gate_param':    param_y,
            'stats_parent':  {},
            'origin_gate':   None,
        }

        # Bump gate numbers for everything that followed the original.
        for other in self.state.gate_definitions:
            if int(other.get('gate_number', 0)) > orig_num:
                other['gate_number'] = int(other['gate_number']) + 1

        # Re-parent children by the quadrant they were in: x+ quadrants map
        # to the y gate (under pos_x); x- quadrants map to neg_x.
        regions = ag_core.population_regions(g, self.state.trained_boundaries.get(gate_name))
        for other in self.state.gate_definitions:
            if other.get('parent_gate') != gate_name:
                continue
            region = regions.get(other.get('parent_popul'), 'x+y+')
            if region.startswith('x+'):
                other['parent_gate'] = name_y
                other['parent_popul'] = 'pos_y' if region.endswith('y+') else 'neg_y'
            else:
                other['parent_gate'] = name_x
                other['parent_popul'] = 'neg_x'

        # Splice: remove original, insert gate_x then gate_y at same position.
        idx = self.state.gate_definitions.index(g)
        self.state.gate_definitions.pop(idx)
        self.state.gate_definitions.insert(idx, gate_y)
        self.state.gate_definitions.insert(idx, gate_x)

        # Clean up orphaned state.
        self.state.trained_boundaries.pop(gate_name, None)
        self.state.algorithm_recommendations.pop(gate_name, None)

        self._repopulate_tree()
        self._select_gate_by_name(name_x)
        self.bus.statusMessage.emit(
            f"[Gating Model Builder] Replaced '{gate_name}' with "
            f"'{name_x}' + '{name_y}'."
        )
        
    # ------------------------------------------------------------------
    # Importing the live GatingStrategy
    # ------------------------------------------------------------------

    def _import_from_unmixed_gating(self, replace: bool = True) -> int:
        """Snapshot ``controller.unmixed_gating`` into ``state.gate_definitions``.

        Every gate becomes a gate definition with its type inferred from the
        FlowKit gate (see _infer_gate_type_from_flowkit). A QuadrantGate
        becomes one 2dsep gate whose populations are its quadrants; the
        quadrant nodes themselves are not imported, and gates under a
        quadrant get that quadrant as their parent population.

        Parameters
        ----------
        replace
            If True, clear the existing list first.  If False, append
            only those gates that do not yet exist by name (used by
            `_merge_from_unmixed_gating`).

        Returns
        -------
        int
            Number of gates imported.
        """
        gating = getattr(self.controller, 'unmixed_gating', None)
        if gating is None:
            return 0

        try:
            gate_ids = list(gating.get_gate_ids())
        except Exception as exc:
            self.bus.statusMessage.emit(
                f"[Gating Model Builder] Could not read unmixed_gating: {exc}"
            )
            return 0

        node_types = {}
        for gate_name, gate_path in gate_ids:
            try:
                node_types[gate_name] = gating._get_gate_node(gate_name, gate_path).gate_type
            except Exception:
                node_types[gate_name] = None
        quadrant_names = {n for n, t in node_types.items() if t == 'Quadrant'}

        if replace:
            self.state.gate_definitions = []
            # Trained boundaries belong to the previous set of gates.
            self.state.trained_boundaries = {}
            self.state.algorithm_recommendations = {}

        existing_names = {g.get('gate_name') for g in self.state.gate_definitions}
        next_number = max(
            (int(g.get('gate_number', 0)) for g in self.state.gate_definitions),
            default=0,
        )
        imported = 0

        # get_gate_ids() lists gates top-down, so parents are imported first.
        for gate_name, gate_path in gate_ids:
            if gate_name in existing_names or gate_name in quadrant_names:
                continue
            try:
                gate = gating.get_gate(gate_name)
            except Exception:
                continue

            try:
                gate_def = self._build_gate_def_from_flowkit(
                    gate, gate_name, tuple(gate_path), next_number + 1,
                    gating, quadrant_names,
                )
            except Exception as exc:
                self.bus.statusMessage.emit(
                    f"[Gating Model Builder] Could not import '{gate_name}': {exc}"
                )
                continue
            self.state.gate_definitions.append(gate_def)
            existing_names.add(gate_name)
            next_number += 1
            imported += 1

        return imported

    def _build_gate_def_from_flowkit(self, gate, gate_name: str, gate_path: tuple,
                                     gate_number: int, gating=None,
                                     quadrant_names: set | None = None) -> dict:
        """Construct a gate definition from one FlowKit gate.

        The parent population of a child gate is the region its source
        parent gate covered: the enclosing quadrant, the 'pos' or 'neg'
        side of a range gate, or the single population of a polygon.
        """
        quadrant_names = quadrant_names or set()
        transforms = getattr(self.controller, 'unmixed_transformations', None) or {}
        gate_type = _infer_gate_type_from_flowkit(gate, self.controller)
        populations = _default_populations_for_type(gate_type)

        if getattr(gate, 'gate_type', '') == 'QuadrantGate':
            populations, channels = _quadrant_populations(gate)
            if len(populations) != 4:
                raise ValueError('quadrant gate without four quadrants')
        else:
            channels = list(gate.get_dimension_ids()) if hasattr(gate, 'get_dimension_ids') else []
        marker_x = channels[0] if len(channels) > 0 else ''
        marker_y = channels[1] if len(channels) > 1 else None
        if gate_type == '1dsep':
            # Keep only the bounded axis of a half-plane rectangle.
            dims = list(getattr(gate, 'dimensions', []) or [])
            bounded_idx = [i for i, d in enumerate(dims) if not _is_unbounded(d)]
            if bounded_idx and len(channels) > bounded_idx[0]:
                marker_x = channels[bounded_idx[0]]
            marker_y = None

        parent_gate: str | None = None
        parent_popul = 'root'
        tail = gate_path[-1] if gate_path else 'root'
        if tail in quadrant_names and len(gate_path) >= 2:
            parent_gate, parent_popul = str(gate_path[-2]), str(tail)
        elif tail != 'root':
            parent_gate = str(tail)
            parent_def = self.state.gate_by_name(parent_gate)
            pops = list((parent_def or {}).get('populations') or {})
            parent_popul = pops[0] if pops else 'root'
            if parent_def is not None and gating is not None \
                    and parent_def.get('gate_type') in ag_core.THRESHOLD_TYPES:
                try:
                    source = _range_gate_population(gating.get_gate(parent_gate), transforms)
                except Exception:
                    source = None
                if source in pops:
                    parent_popul = source

        return {
            'gate_number':   gate_number,
            'gate_name':     gate_name,
            'gate_type':     gate_type,
            'gate_marker_x': marker_x,
            'gate_marker_y': marker_y,
            'parent_gate':   parent_gate,
            'parent_popul':  parent_popul,
            'populations':   populations,
            'algorithm':     _default_algorithm_for_type(gate_type),
            'gate_param':    {},
            'stats_parent':  {},
            'origin_gate':   None,
        }

    def _merge_from_unmixed_gating(self) -> tuple[int, int]:
        """Add new gates from the live hierarchy; mark removed ones Stale.

        Returns ``(added, marked_stale)``.  Existing ``gate_param``
        overrides on retained gates are preserved.  Gates whose names
        no longer appear in the live hierarchy are not deleted — the
        user may have authored them manually.  They simply pick up a
        Stale badge during repopulate because their axis channels
        won't match the current experiment.  Their actual `gate_param`
        is untouched.
        """
        added = self._import_from_unmixed_gating(replace=False)

        # Count gates that are about to render Stale because their axis
        # channels are not in the current experiment.  This is the same
        # check the tree uses; we expose the count here so the status
        # message is honest about what happened.
        experiment_channels = set(_experiment_channels(self.controller))
        stale = 0
        for g in self.state.gate_definitions:
            x = g.get('gate_marker_x') or ''
            y = g.get('gate_marker_y') or ''
            if experiment_channels and (
                (x and x not in experiment_channels) or
                (y and y not in experiment_channels)
            ):
                stale += 1
        return added, stale

    # ------------------------------------------------------------------
    # Algorithm recommendation
    # ------------------------------------------------------------------

    def _recommend_algorithms(self):
        """Suggest an algorithm for each gate from one representative sample.

        Uses the first training sample, or the first non-control sample.
        A background worker loads it (with its AF assignment), runs the
        hierarchy with the current algorithms to find each gate's parent
        population, and checks each threshold gate's axes for bimodality
        (GMM ΔBIC): bimodal → 'mixture', otherwise 'tail'.
        """
        if self._recommend_worker is not None:
            return
        keys = list(self.state.training_samples) or analysis_sample_keys(self.controller)
        if not keys or not self.state.gate_definitions:
            self.bus.statusMessage.emit(
                "[Gating Model Builder] Cannot recommend algorithms: no samples "
                "or gates available."
            )
            return
        key = keys[0]
        try:
            snap = snapshot_unmix_state(self.controller, [key])
        except Exception as exc:
            self.bus.statusMessage.emit(f"[Gating Model Builder] {exc}")
            return
        worker = _RecommendWorker(
            self.controller.experiment_dir, key, snap,
            deepcopy(self.state.gate_definitions),
            _experiment_channels(self.controller),
            deepcopy(self.controller.unmixed_transformations or {}),
            parent=self,
        )
        worker.finished.connect(self._on_recommendations_ready)
        self._recommend_worker = worker
        self.bus.statusMessage.emit(
            f"[Gating Model Builder] Checking bimodality on {_sample_label(key)} …"
        )
        worker.start()

    def _on_recommendations_ready(self, ok: bool, message: str, recommendations: object):
        self._recommend_worker = None
        if not ok:
            self.bus.statusMessage.emit(
                f"[Gating Model Builder] Algorithm recommendation failed: {message}"
            )
            return
        self.state.algorithm_recommendations.update(recommendations or {})
        self._repopulate_tree()
        self.bus.statusMessage.emit(
            f"[Gating Model Builder] Algorithm recommendations computed for "
            f"{len(recommendations or {})} gate(s)."
        )

    # ------------------------------------------------------------------
    # Right-hand preview pane
    # ------------------------------------------------------------------

    def _update_preview(self):
        """Refresh the right-pane preview plot for the selected gate."""
        import honeychrome.settings as hc_settings
        import colorcet as cc

        # Clear previous overlays (geometry lines, text items).
        # ImageItem and PlotDataItem are reused in-place.
        for item in list(self._preview_vb.addedItems):
            if item not in (self._preview_img, self._preview_hist):
                self._preview_vb.removeItem(item)
        self._preview_img.clear()
        self._preview_hist.setData([], [])

        g = self._selected_gate()
        if g is None:
            self.lbl_preview_title.setText(
                "<i>Select a gate to preview its geometry.</i>"
            )
            return

        ch_x = g.get('gate_marker_x', '')
        ch_y = g.get('gate_marker_y') or ''

        # Display labels via build_display_label_map
        pnn = _experiment_channels(self.controller)
        spectral_model = self.controller.experiment.process.get('spectral_model') or []
        pnn_labels = build_display_label_map(pnn, spectral_model)
        x_label = pnn_labels.get(ch_x, ch_x)
        y_label = pnn_labels.get(ch_y, ch_y) if ch_y else 'Count'

        self._preview_axis_x.setLabel(x_label)
        self._preview_axis_y.setLabel(y_label)

        title_bits = [f"<b>{g.get('gate_name', '')}</b>",
                      f"type={g.get('gate_type', '')}",
                      f"algo={g.get('algorithm', '')}"]
        self.lbl_preview_title.setText(" — ".join(title_bits))

        transforms = self.controller.unmixed_transformations or {}
        tr_x = transforms.get(ch_x)
        tr_y = transforms.get(ch_y) if ch_y else None

        # --- Colourmap (matches CytometryPlotWidget) ---
        try:
            colors = cc.palette[hc_settings.colourmap_name_retrieved]
        except Exception:
            colors = cc.palette['rainbow4']
        cmap = pg.ColorMap(
            pos=0.9 * np.linspace(0, 1, len(colors)) ** 2
                + 0.1 * np.linspace(0, 1, len(colors)),
            color=colors,
        )
        rgba_lut = cmap.getLookupTable(alpha=True)
        rgba_lut[0, 3] = 0
        self._preview_img.setLookupTable(rgba_lut)

        # --- Load raw unmixed events ---
        raw_data = self._load_preview_events_raw()

        # --- Histogram / heatmap background ---
        if raw_data is not None and tr_x is not None and ch_x in pnn:
            ix = pnn.index(ch_x)
            mask = np.ones(len(raw_data), dtype=bool)
            if tr_y is not None and ch_y in pnn:
                iy = pnn.index(ch_y)
                heatmap = calc_hist2d(
                    raw_data, mask, ix, iy, tr_x, tr_y,
                    density_cutoff=hc_settings.density_cutoff_retrieved,
                )
                self._preview_img.setImage(heatmap)
                from PySide6.QtCore import QRectF
                self._preview_img.setRect(QRectF(
                    tr_x.limits[0], tr_y.limits[0],
                    tr_x.limits[1] - tr_x.limits[0],
                    tr_y.limits[1] - tr_y.limits[0],
                ))
                # Configure axes — ticks BEFORE range
                if tr_x.ticks:
                    self._preview_axis_x.setTicks(tr_x.ticks())
                self._preview_axis_x.zoomZero  = tr_x.zero
                self._preview_axis_x.fullRange = (0, 1.1)
                self._preview_axis_x.limits    = tuple(tr_x.limits)
                self._preview_vb.setXRange(tr_x.limits[0], tr_x.limits[1], padding=0)

                if tr_y.ticks:
                    self._preview_axis_y.setTicks(tr_y.ticks())
                self._preview_axis_y.zoomZero  = tr_y.zero
                self._preview_axis_y.fullRange = (0, 1.1)
                self._preview_axis_y.limits    = tuple(tr_y.limits)
                self._preview_vb.setYRange(tr_y.limits[0], tr_y.limits[1], padding=0)
            else:
                # 1-D histogram
                count = calc_hist1d(raw_data, mask, ix, tr_x)
                self._preview_hist.setData(tr_x.step_scale, count)
                if tr_x.ticks:
                    self._preview_axis_x.setTicks(tr_x.ticks())
                self._preview_axis_x.zoomZero  = tr_x.zero
                self._preview_axis_x.fullRange = (0, 1.1)
                self._preview_axis_x.limits    = tuple(tr_x.limits)
                self._preview_vb.setXRange(tr_x.limits[0], tr_x.limits[1], padding=0)
                self._preview_vb.enableAutoRange(axis=self._preview_vb.YAxis, enable=True)

        # --- Gate geometry overlay ---
        flowkit_gate = self._lookup_flowkit_gate(g.get('gate_name', ''))
        drew_geometry = False
        if flowkit_gate is not None:
            drew_geometry = self._draw_flowkit_geometry(flowkit_gate)
        if not drew_geometry:
            drew_geometry = self._draw_trained_boundary(g)

        if not drew_geometry and raw_data is None:
            txt = pg.TextItem(
                "No geometry to preview yet.\n"
                "Run Train to compute boundaries.",
                color=(120, 120, 120),
                anchor=(0.5, 0.5),
            )
            self._preview_vb.addItem(txt)

    def _load_preview_events_raw(self) -> 'np.ndarray | None':
        """Return the controller's currently-loaded unmixed event array.

        Uses controller.unmixed_event_data directly — no file I/O, always
        reflects the sample the user is actively looking at in the main window.
        Returns float64 (n_events, n_channels) raw unmixed, no logicle applied.
        Subsamples to settings.max_display_events if needed.
        """
        import honeychrome.settings as hc_settings

        data = getattr(self.controller, 'unmixed_event_data', None)
        if data is None or len(data) == 0:
            return None

        cap = int(getattr(hc_settings, 'max_display_events', 500_000))
        if len(data) > cap:
            rng = np.random.default_rng(0)
            data = data[rng.choice(len(data), cap, replace=False)]
        return data

    def _lookup_flowkit_gate(self, gate_name: str):
        """Return the flowkit gate object for *gate_name*, or None."""
        if not gate_name:
            return None
        gating = getattr(self.controller, 'unmixed_gating', None)
        if gating is None:
            return None
        try:
            return gating.get_gate(gate_name)
        except Exception:
            return None

    def _draw_flowkit_geometry(self, gate) -> bool:
        """Render a flowkit gate's geometry into the preview plot.

        Returns True if anything was drawn, False if the gate type is
        unrecognised or has nothing useful to show.
        """
        gtype = getattr(gate, 'gate_type', '')

        if gtype == 'PolygonGate':
            verts = _polygon_vertices(gate)
            if not verts:
                return False
            # Close the polygon for a clean outline.
            closed = list(verts) + [verts[0]]
            xs = [p[0] for p in closed]
            ys = [p[1] for p in closed]
            line_item = pg.PlotDataItem(xs, ys, pen=pg.mkPen(color=(30, 160, 50), width=2))
            self._preview_vb.addItem(line_item)
            return True

        if gtype == 'RectangleGate':
            corners = _rectangle_corners(gate)
            if corners is not None:
                xs = [p[0] for p in corners]
                ys = [p[1] for p in corners]
                line_item = pg.PlotDataItem(xs, ys,
                                            pen=pg.mkPen(color=(30, 160, 50), width=2))
                self._preview_vb.addItem(line_item)
                return True
            dims = list(getattr(gate, 'dimensions', []) or [])
            for d in dims:
                if _is_unbounded(d):
                    continue
                if d.min is not None:
                    line = pg.InfiniteLine(pos=float(d.min), angle=90,
                                           pen=pg.mkPen(color=(30, 160, 50), width=2))
                    self._preview_vb.addItem(line)
                if d.max is not None:
                    line = pg.InfiniteLine(pos=float(d.max), angle=90,
                                           pen=pg.mkPen(color=(30, 160, 50), width=2,
                                                        style=Qt.DashLine))
                    self._preview_vb.addItem(line)
            return True

        if gtype == 'QuadrantGate':
            dividers = list(getattr(gate, 'dimensions', []) or [])[:2]
            for angle, div in zip((90, 0), dividers):
                values = getattr(div, 'values', None) or []
                for v in values:
                    self._preview_vb.addItem(pg.InfiniteLine(
                        pos=float(v), angle=angle,
                        pen=pg.mkPen(color=(30, 160, 50), width=2)))
            return bool(dividers)

        return False

    def _draw_trained_boundary(self, gate_def: dict) -> bool:
        """Draw any TrainedBoundary entries for *gate_def*.

        Returns True if anything was drawn.
        """
        gate_name = gate_def.get('gate_name', '')
        entry = self.state.trained_boundaries.get(gate_name)
        if not entry:
            return False
        drew = False
        for pop_name, pop_data in entry.items():
            boundary = pop_data.get('boundary')
            if not boundary:
                continue
            coords = np.asarray(boundary)
            if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 2:
                continue
            xs = coords[:, 0].tolist()
            ys = coords[:, 1].tolist()
            if xs[0] != xs[-1] or ys[0] != ys[-1]:
                xs.append(xs[0])
                ys.append(ys[0])
            line_item = pg.PlotDataItem(xs, ys,
                                        pen=pg.mkPen(color=(30, 160, 50), width=2))
            self._preview_vb.addItem(line_item)
            # Percentage label at centroid of boundary
            frac = pop_data.get('fraction')
            if frac is not None:
                cx = float(np.mean(xs))
                cy = float(np.mean(ys))
                pct_txt = pg.TextItem(
                    f"{pop_name}\n{frac * 100:.1f}%",
                    color=(30, 160, 50),
                    anchor=(0.5, 0.5),
                )
                pct_txt.setPos(cx, cy)
                self._preview_vb.addItem(pct_txt)
            drew = True
        return drew


class GateDetailWidget(QDialog):
    """Full-size gate plot with interactive threshold adjustment.

    Opened when the user clicks a tile in the TrainTab gate-plot grid.
    For 1D gates a QDoubleSpinBox lets the user nudge the threshold and
    see the boundary move in real time.  For 2D gates two movable
    pg.InfiniteLines serve the same purpose.  Accept stores the
    adjusted boundary back into state.trained_boundaries.

    Parameters
    ----------
    gate_def
        The GateDefinition dict for this gate.
    state
        Shared GatingModelState — read and written on Accept.
    channel_list
        Ordered channel name list (event_channels_pnn).
    event_data
        Untransformed unmixed display events (may be None — the plot is
        then empty but the threshold controls still work).
    parent_mask
        Rows of event_data in the gate's parent population.
    """

    def __init__(self, gate_def: dict, state: GatingModelState,
                 channel_list: list[str], event_data,
                 parent=None, parent_mask: 'np.ndarray | None' = None):
        super().__init__(parent)
        self.gate_def = gate_def
        self.state = state
        self.channel_list = channel_list
        self.event_data = event_data
        self._parent_mask = parent_mask

        gate_name = gate_def.get('gate_name', '')
        self.setWindowTitle(f"Gate detail — {gate_name}")
        self.setModal(False)
        self.resize(640, 540)

        outer = QVBoxLayout(self)

        # Proper Honeychrome plot widget
        import honeychrome.settings as hc_settings
        import colorcet as cc

        self._gw = TransparentGraphicsLayoutWidget()
        self._gw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gl = self._gw.ci.layout
        gl.setHorizontalSpacing(0)
        gl.setVerticalSpacing(0)

        self._vb = NoPanViewBox()
        self._vb.setMouseEnabled(x=False, y=False)
        self._vb.raiseContextMenu = lambda ev: None
        self._gw.addItem(self._vb, row=0, col=1)

        self._axis_x = ZoomAxis('bottom', self._vb)
        self._axis_y = ZoomAxis('left',   self._vb)
        self._gw.addItem(self._axis_x, row=1, col=1)
        self._gw.addItem(self._axis_y, row=0, col=0)
        self._axis_x.linkToView(self._vb)
        self._axis_y.linkToView(self._vb)

        try:
            colors = cc.palette[hc_settings.colourmap_name_retrieved]
        except Exception:
            colors = cc.palette['rainbow4']
        cmap = pg.ColorMap(
            pos=0.9 * np.linspace(0, 1, len(colors)) ** 2
                + 0.1 * np.linspace(0, 1, len(colors)),
            color=colors,
        )
        rgba_lut = cmap.getLookupTable(alpha=True)
        rgba_lut[0, 3] = 0

        self._img = pg.ImageItem()
        self._img.setLookupTable(rgba_lut)
        self._vb.addItem(self._img)

        self._hist_curve = pg.PlotDataItem(
            stepMode='center', fillLevel=0,
            brush=(100, 100, 250, 150),
        )
        self._vb.addItem(self._hist_curve)

        outer.addWidget(self._gw, stretch=1)

        # Threshold controls (1D only)
        self._x_line: object = None
        self._y_line: object = None
        self._ctrl_row = QHBoxLayout()
        self._build_controls()
        outer.addLayout(self._ctrl_row)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

        self._draw_plot()

    def _build_controls(self):
        gt = self.gate_def.get('gate_type', '')
        if gt == '1dsep':
            entry = self._current_boundary_entry()
            tx = (entry.get('threshold_x') if entry else None) or 0.0
            lbl = QLabel("Threshold:")
            self._sb_threshold = QDoubleSpinBox()
            self._sb_threshold.setRange(-1e6, 1e6)
            self._sb_threshold.setDecimals(4)
            self._sb_threshold.setSingleStep(0.01)
            self._sb_threshold.setValue(float(tx))
            self._sb_threshold.valueChanged.connect(self._on_threshold_changed)
            self._ctrl_row.addWidget(lbl)
            self._ctrl_row.addWidget(self._sb_threshold)
            self._ctrl_row.addStretch()

    def _current_boundary_entry(self):
        """Return the first population entry for this gate, or None."""
        gate_name = self.gate_def.get('gate_name', '')
        tb = self.state.trained_boundaries.get(gate_name, {})
        if not tb:
            return None
        return next(iter(tb.values()))

    def _draw_plot(self):
        import honeychrome.settings as hc_settings
        from PySide6.QtCore import QRectF

        # Clear overlays (keep ImageItem and PlotDataItem in the vb)
        for item in list(self._vb.addedItems):
            if item not in (self._img, self._hist_curve):
                self._vb.removeItem(item)
        self._img.clear()
        self._hist_curve.setData([], [])

        gate_name = self.gate_def.get('gate_name', '')
        gt = self.gate_def.get('gate_type', '')
        ch_x = self.gate_def.get('gate_marker_x', '')
        ch_y = self.gate_def.get('gate_marker_y') or ''

        transforms = getattr(self, '_transforms', {})
        # Try to get live transforms from parent TrainTab
        try:
            transforms = self.parent().controller.unmixed_transformations or {}
        except Exception:
            pass
        tr_x = transforms.get(ch_x)
        tr_y = transforms.get(ch_y) if ch_y else None

        raw_data = self.event_data  # raw unmixed, no logicle applied
        if raw_data is not None and tr_x is not None and ch_x in self.channel_list:
            ix = self.channel_list.index(ch_x)
            mask = (self._parent_mask
                    if self._parent_mask is not None
                    else np.ones(len(raw_data), dtype=bool))
            if tr_y is not None and ch_y in self.channel_list:
                iy = self.channel_list.index(ch_y)
                heatmap = calc_hist2d(
                    raw_data, mask, ix, iy, tr_x, tr_y,
                    density_cutoff=hc_settings.density_cutoff_retrieved,
                )
                self._img.setImage(heatmap)
                self._img.setRect(QRectF(
                    tr_x.limits[0], tr_y.limits[0],
                    tr_x.limits[1] - tr_x.limits[0],
                    tr_y.limits[1] - tr_y.limits[0],
                ))
                if tr_x.ticks:
                    self._axis_x.setTicks(tr_x.ticks())
                self._axis_x.zoomZero  = tr_x.zero
                self._axis_x.fullRange = (0, 1.1)
                self._axis_x.limits    = tuple(tr_x.limits)
                self._vb.setXRange(tr_x.limits[0], tr_x.limits[1], padding=0)

                if tr_y.ticks:
                    self._axis_y.setTicks(tr_y.ticks())
                self._axis_y.zoomZero  = tr_y.zero
                self._axis_y.fullRange = (0, 1.1)
                self._axis_y.limits    = tuple(tr_y.limits)
                self._vb.setYRange(tr_y.limits[0], tr_y.limits[1], padding=0)
            else:
                count = calc_hist1d(raw_data, mask, ix, tr_x)
                self._hist_curve.setData(tr_x.step_scale, count)
                if tr_x.ticks:
                    self._axis_x.setTicks(tr_x.ticks())
                self._axis_x.zoomZero  = tr_x.zero
                self._axis_x.fullRange = (0, 1.1)
                self._axis_x.limits    = tuple(tr_x.limits)
                self._vb.setXRange(tr_x.limits[0], tr_x.limits[1], padding=0)
                self._vb.enableAutoRange(axis=self._vb.YAxis, enable=True)

        # Overlay boundary
        tb = self.state.trained_boundaries.get(gate_name, {})
        if gt == '1dsep':
            entry = self._current_boundary_entry()
            tx = (entry.get('threshold_x') if entry else None) or 0.0
            self._x_line = pg.InfiniteLine(
                pos=float(tx), angle=90, movable=True,
                pen=pg.mkPen(color=(30, 160, 50), width=2)
            )
            self._x_line.sigPositionChanged.connect(
                lambda ln: self._sb_threshold.setValue(ln.value())
                if hasattr(self, '_sb_threshold') else None
            )
            self._vb.addItem(self._x_line)
        elif gt == '2dsep':
            entry = self._current_boundary_entry()
            tx = (entry.get('threshold_x') if entry else None) or 0.0
            ty = (entry.get('threshold_y') if entry else None) or 0.0
            self._x_line = pg.InfiniteLine(
                pos=float(tx), angle=90, movable=True,
                pen=pg.mkPen(color=(30, 160, 50), width=2)
            )
            self._y_line = pg.InfiniteLine(
                pos=float(ty), angle=0, movable=True,
                pen=pg.mkPen(color=(30, 160, 50), width=2)
            )
            self._vb.addItem(self._x_line)
            self._vb.addItem(self._y_line)
        else:
            for pop_name, pop_data in tb.items():
                boundary = pop_data.get('boundary')
                if not boundary:
                    continue
                coords = np.asarray(boundary)
                if coords.ndim == 2 and coords.shape[1] == 2:
                    line_item = pg.PlotDataItem(
                        coords[:, 0].tolist(), coords[:, 1].tolist(),
                        pen=pg.mkPen(color=(30, 160, 50), width=2)
                    )
                    self._vb.addItem(line_item)

    def _on_threshold_changed(self, value: float):
        """Move the InfiniteLine to match the spinbox."""
        if self._x_line is not None:
            self._x_line.setPos(float(value))

    def _on_accept(self):
        """Write the adjusted thresholds back into state.trained_boundaries.

        The population rectangles are redrawn from the new thresholds; event
        counts are refreshed by the Train tab afterwards.
        """
        gate_name = self.gate_def.get('gate_name', '')
        gt = self.gate_def.get('gate_type', '')
        tb = self.state.trained_boundaries.get(gate_name, {})
        try:
            transforms = self.parent().controller.unmixed_transformations or {}
        except Exception:
            transforms = {}
        calc = ag_core.GateCalculator(_axis_limits(transforms))

        new_entries = None
        if gt == '1dsep' and hasattr(self, '_sb_threshold') and tb:
            new_entries = calc.threshold_boundaries(self.gate_def, float(self._sb_threshold.value()), None)
        elif gt == '2dsep' and self._x_line and self._y_line and tb:
            new_entries = calc.threshold_boundaries(self.gate_def, float(self._x_line.value()),
                                                    float(self._y_line.value()))
        if new_entries:
            for pop_name, entry in new_entries.items():
                entry['label'] = (tb.get(pop_name) or {}).get('label', pop_name)
                entry['adjusted'] = True
            self.state.trained_boundaries[gate_name] = new_entries

        self.accept()


class _TrainingWorker(QThread):
    """Load training samples, pool them and calculate every gate.

    Each sample contributes a seeded random subsample of at most
    ``events_per_sample`` events (0 = all), unmixed with its own AF
    assignment and restricted to its time-QC keep-mask when present. The
    pooled events are transformed and passed to ag_core.train_boundaries.

    Emits finished(ok, error, payload); payload holds the boundaries, the
    per-gate status, per-sample event counts, and a display subsample of
    the pooled (untransformed) events with each gate's parent mask.
    """

    progress = Signal(str)
    finished = Signal(bool, str, object)

    def __init__(self, experiment_dir, sample_keys, snapshot, gate_defs, channels,
                 transforms, events_per_sample: int, display_cap: int, seed: int,
                 parent=None):
        super().__init__(parent)
        self._experiment_dir = experiment_dir
        self._keys = list(sample_keys)
        self._snap = snapshot
        self._gate_defs = gate_defs
        self._channels = channels
        self._transforms = transforms
        self._cap = int(events_per_sample) or None
        self._display_cap = int(display_cap)
        self._seed = int(seed)
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            payload = self._execute()
        except Exception as exc:
            log.exception("training failed")
            self.finished.emit(False, str(exc), None)
            return
        if payload is None:
            self.finished.emit(False, "Cancelled.", None)
        else:
            self.finished.emit(True, "", payload)

    def _execute(self):
        chunks = []
        per_sample = {}
        for i, key in enumerate(self._keys, 1):
            if self._abort:
                return None
            self.progress.emit(f"Loading {_sample_label(key)} ({i}/{len(self._keys)}) …")
            try:
                out = load_unmixed(self._experiment_dir, key, self._snap,
                                   max_events=self._cap, seed=self._seed)
            except Exception as exc:
                log.warning("could not load %s: %s", key, exc)
                self.progress.emit(f"Could not load {_sample_label(key)}: {exc}")
                continue
            chunks.append(np.asarray(out['unmixed'], dtype=np.float32))
            per_sample[key] = {
                'n_events_used': int(len(out['unmixed'])),
                'n_events_file': int(out['n_events_file']),
                'n_events_kept': int(out['n_events_kept']),
                'af_profiles': list(out['af_profiles']),
            }
        if not chunks:
            raise RuntimeError("No training samples could be loaded.")
        if self._abort:
            return None

        pooled = np.concatenate(chunks, axis=0)
        del chunks
        self.progress.emit(f"Calculating gates on {len(pooled):,} pooled events …")
        data = ag_core.transform_columns(pooled, self._channels, self._transforms)
        index = {ch: i for i, ch in enumerate(self._channels)}
        result = ag_core.train_boundaries(self._gate_defs, data, index,
                                          axis_limits=_axis_limits(self._transforms))
        del data

        n = len(pooled)
        if self._display_cap and n > self._display_cap:
            rng = np.random.default_rng(self._seed)
            rows = np.sort(rng.choice(n, self._display_cap, replace=False))
        else:
            rows = np.arange(n)
        return {
            'boundaries': result.boundaries,
            'status': result.status,
            'per_sample': per_sample,
            'n_pooled': int(n),
            'display_events': pooled[rows],
            'display_parent_masks': {g: m[rows] for g, m in result.parent_masks.items()},
        }


class TrainTab(QWidget):
    """Tab 1 — Train.

    Top: training-sample picker, events per sample, and Calculate/Abort.
    Bottom: one tile per gate showing the trained boundary over a display
    subsample of the pooled training events (parent population only).
    "Detail…" on a tile opens GateDetailWidget to adjust thresholds.
    """

    trained = Signal()              # training finished successfully
    boundaries_changed = Signal()   # a boundary was adjusted by hand

    def __init__(self, state: GatingModelState, bus, controller, parent=None):
        super().__init__(parent)
        self.state = state
        self.bus = bus
        self.controller = controller

        self._worker: _TrainingWorker | None = None
        self._tile_widgets: dict[str, QWidget] = {}
        self._last_event_data: np.ndarray | None = None      # untransformed display events
        self._last_gate_masks: dict[str, np.ndarray] = {}    # gate -> parent mask over display rows

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        splitter = QSplitter(Qt.Vertical)

        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(4, 4, 4, 4)
        top_layout.addWidget(HelpToggleWidget(text=ag_help_texts.BUILDER_TRAIN))
        top_layout.addWidget(QLabel("<b>Training samples</b>"))

        self.picker = OrderedMultiSamplePicker()
        top_layout.addWidget(self.picker)

        opts_row = QHBoxLayout()
        opts_row.addWidget(QLabel("Events per sample:"))
        self.sb_events = QSpinBox()
        self.sb_events.setRange(0, 10_000_000)
        self.sb_events.setSingleStep(10_000)
        self.sb_events.setSpecialValueText("All")
        self.sb_events.setValue(int(self.state.events_per_sample))
        self.sb_events.setToolTip(
            "Seeded random subsample drawn from each training sample, so every "
            "sample contributes equally to the pooled thresholds. 0 = all events."
        )
        self.sb_events.valueChanged.connect(
            lambda v: setattr(self.state, 'events_per_sample', int(v)))
        opts_row.addWidget(self.sb_events)
        self.chk_controls = QCheckBox("Show controls")
        self.chk_controls.setToolTip("Also offer single-stain controls and unstained samples.")
        self.chk_controls.setChecked(bool(self.state.include_controls))
        self.chk_controls.toggled.connect(self._on_controls_toggled)
        opts_row.addWidget(self.chk_controls)
        opts_row.addStretch()
        top_layout.addLayout(opts_row)

        run_row = QHBoxLayout()
        self.btn_run = QPushButton("Calculate Gates")
        self.btn_run.setToolTip("Calculate every gate's boundary from the selected training samples.")
        self.btn_abort = QPushButton("Abort")
        self.btn_abort.setEnabled(False)
        self.btn_run.clicked.connect(self._on_run_clicked)
        self.btn_abort.clicked.connect(self._on_abort_clicked)
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #555;")
        self.lbl_status.setWordWrap(True)
        run_row.addWidget(self.btn_run)
        run_row.addWidget(self.btn_abort)
        run_row.addWidget(self.lbl_status, stretch=1)
        top_layout.addLayout(run_row)

        splitter.addWidget(top)

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        self.grid_scroll = QScrollArea()
        self.grid_scroll.setWidgetResizable(True)
        self.grid_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(4)
        self.grid_layout.setContentsMargins(4, 4, 4, 4)

        self.grid_scroll.setWidget(self.grid_container)
        bottom_layout.addWidget(self.grid_scroll)
        splitter.addWidget(bottom)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

    # ------------------------------------------------------------------
    # Public refresh hook
    # ------------------------------------------------------------------

    def refresh(self):
        """Repopulate the sample picker from the current experiment."""
        self.sb_events.blockSignals(True)
        self.sb_events.setValue(int(self.state.events_per_sample))
        self.sb_events.blockSignals(False)
        self.chk_controls.blockSignals(True)
        self.chk_controls.setChecked(bool(self.state.include_controls))
        self.chk_controls.blockSignals(False)
        self._populate_picker()
        self._rebuild_grid()

    def _populate_picker(self):
        try:
            available = analysis_sample_keys(self.controller, self.state.include_controls)
        except Exception:
            available = []
        current = self.picker.get_ordered_list() or list(self.state.training_samples)
        selected = [s for s in current if s in available]
        self.picker.set_items(available, selected=selected)

    def _on_controls_toggled(self, checked: bool):
        self.state.include_controls = bool(checked)
        self._populate_picker()

    # ------------------------------------------------------------------
    # Tile grid
    # ------------------------------------------------------------------

    def _rebuild_grid(self):
        """Redraw all tiles from state.trained_boundaries."""
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._tile_widgets.clear()

        cols = 4
        for i, g in enumerate(ag_core.ordered_gate_defs(self.state.gate_definitions)):
            tile = self._make_tile(g)
            row, col = divmod(i, cols)
            self.grid_layout.addWidget(tile, row, col)
            self._tile_widgets[g.get('gate_name', '')] = tile

    def _make_tile(self, gate_def: dict) -> QWidget:
        """Create one plot tile for a gate."""
        import honeychrome.settings as hc_settings
        import colorcet as cc
        from PySide6.QtCore import QRectF

        gate_name = gate_def.get('gate_name', '')
        eff = ag_core.effective_gate_def(
            gate_def, ag_core.gates_by_name(self.state.gate_definitions))
        gt = eff.get('gate_type', '')
        ch_x = eff.get('gate_marker_x', '')
        ch_y = eff.get('gate_marker_y') or ''
        tb = self.state.trained_boundaries.get(gate_name, {})

        pnn = list(
            self.controller.experiment.settings.get('unmixed', {})
            .get('event_channels_pnn') or []
        )
        spectral_model = self.controller.experiment.process.get('spectral_model') or []
        pnn_labels = build_display_label_map(pnn, spectral_model)
        x_label = pnn_labels.get(ch_x, ch_x)
        y_label = pnn_labels.get(ch_y, ch_y) if ch_y else 'Count'
        transforms = self.controller.unmixed_transformations or {}
        tr_x = transforms.get(ch_x)
        tr_y = transforms.get(ch_y) if ch_y else None

        # Tile size from settings
        tile_size = hc_settings.cytometry_plot_width_target_retrieved

        container = QWidget()
        container.setFixedSize(tile_size + 20, tile_size + 40)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Title label
        status = "Trained" if tb else "Untrained"
        lbl = QLabel(f"<b>{gate_name}</b> <small>[{status}]</small>")
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)

        # Honeychrome plot widget
        gw = TransparentGraphicsLayoutWidget()
        gw.setFixedSize(tile_size, tile_size)
        gl = gw.ci.layout
        gl.setHorizontalSpacing(0)
        gl.setVerticalSpacing(0)

        vb = NoPanViewBox()
        vb.setMouseEnabled(x=False, y=False)
        vb.raiseContextMenu = lambda ev: None
        gw.addItem(vb,     row=1, col=2)

        axis_x = ZoomAxis('bottom', vb)
        axis_y = ZoomAxis('left',   vb)
        gw.addItem(axis_y, row=1, col=1)
        gw.addItem(axis_x, row=2, col=2)
        axis_x.linkToView(vb)
        axis_y.linkToView(vb)
        axis_x.setLabel(x_label)
        axis_y.setLabel(y_label)

        # Colourmap
        try:
            colors = cc.palette[hc_settings.colourmap_name_retrieved]
        except Exception:
            colors = cc.palette['rainbow4']
        cmap = pg.ColorMap(
            pos=0.9 * np.linspace(0, 1, len(colors)) ** 2
                + 0.1 * np.linspace(0, 1, len(colors)),
            color=colors,
        )
        rgba_lut = cmap.getLookupTable(alpha=True)
        rgba_lut[0, 3] = 0

        img = pg.ImageItem()
        img.setLookupTable(rgba_lut)
        vb.addItem(img)

        hist_curve = pg.PlotDataItem(
            stepMode='center', fillLevel=0,
            brush=(100, 100, 250, 150),
        )
        vb.addItem(hist_curve)

        layout.addWidget(gw, stretch=1)

        # --- Histogram / heatmap background from pooled raw event data ---
        raw_data = self._last_event_data
        if raw_data is not None and tr_x is not None and ch_x in pnn:
            ix = pnn.index(ch_x)
            stored_mask = self._last_gate_masks.get(gate_name)
            if stored_mask is not None and len(stored_mask) == len(raw_data):
                mask = stored_mask
            else:
                mask = np.ones(len(raw_data), dtype=bool)
            if tr_y is not None and ch_y in pnn:
                iy = pnn.index(ch_y)
                heatmap = calc_hist2d(
                    raw_data, mask, ix, iy, tr_x, tr_y,
                    density_cutoff=hc_settings.density_cutoff_retrieved,
                )
                img.setImage(heatmap)
                img.setRect(QRectF(
                    tr_x.limits[0], tr_y.limits[0],
                    tr_x.limits[1] - tr_x.limits[0],
                    tr_y.limits[1] - tr_y.limits[0],
                ))
                # Axes — ticks BEFORE range
                if tr_x.ticks:
                    axis_x.setTicks(tr_x.ticks())
                axis_x.zoomZero  = tr_x.zero
                axis_x.fullRange = (0, 1.1)
                axis_x.limits    = tuple(tr_x.limits)
                vb.setXRange(tr_x.limits[0], tr_x.limits[1], padding=0)

                if tr_y.ticks:
                    axis_y.setTicks(tr_y.ticks())
                axis_y.zoomZero  = tr_y.zero
                axis_y.fullRange = (0, 1.1)
                axis_y.limits    = tuple(tr_y.limits)
                vb.setYRange(tr_y.limits[0], tr_y.limits[1], padding=0)
            else:
                count = calc_hist1d(raw_data, mask, ix, tr_x)
                hist_curve.setData(tr_x.step_scale, count)
                if tr_x.ticks:
                    axis_x.setTicks(tr_x.ticks())
                axis_x.zoomZero  = tr_x.zero
                axis_x.fullRange = (0, 1.1)
                axis_x.limits    = tuple(tr_x.limits)
                vb.setXRange(tr_x.limits[0], tr_x.limits[1], padding=0)
                vb.enableAutoRange(axis=vb.YAxis, enable=True)

        # --- Gate boundary overlay (green) + percentage labels ---
        if tb:
            if gt == '1dsep':
                entry = next(iter(tb.values()))
                tx = entry.get('threshold_x') or 0.0
                line = pg.InfiniteLine(
                    pos=float(tx), angle=90,
                    pen=pg.mkPen(color=(30, 160, 50), width=1.5)
                )
                vb.addItem(line)
                for pop_name, pop_data in tb.items():
                    frac = pop_data.get('fraction')
                    if frac is not None:
                        bd = pop_data.get('boundary', [])
                        xs_bd = [p[0] for p in bd]
                        cx = float(np.mean(xs_bd)) if xs_bd else float(tx)
                        tr_lim = tr_x.limits if tr_x else [0, 1]
                        cy = (tr_lim[0] + tr_lim[1]) * 0.5
                        pop_label = pop_data.get('label', pop_name)
                        pct_txt = pg.TextItem(
                            f"{pop_label}\n{frac * 100:.1f}%",
                            color=(30, 160, 50), anchor=(0.5, 0.5),
                        )
                        pct_txt.setPos(cx, cy)
                        vb.addItem(pct_txt)
            else:
                for pop_name, pop_data in tb.items():
                    boundary = pop_data.get('boundary')
                    if not boundary:
                        continue
                    coords = np.asarray(boundary)
                    if coords.ndim != 2 or coords.shape[1] != 2:
                        continue
                    xs_b = coords[:, 0].tolist()
                    ys_b = coords[:, 1].tolist()
                    if xs_b[0] != xs_b[-1] or ys_b[0] != ys_b[-1]:
                        xs_b.append(xs_b[0])
                        ys_b.append(ys_b[0])
                    line_item = pg.PlotDataItem(
                        xs_b, ys_b,
                        pen=pg.mkPen(color=(30, 160, 50), width=1.5)
                    )
                    vb.addItem(line_item)
                    frac = pop_data.get('fraction')
                    if frac is not None:
                        cx = float(np.mean(xs_b))
                        cy = float(np.mean(ys_b))
                        pop_label = pop_data.get('label', pop_name)
                        pct_txt = pg.TextItem(
                            f"{pop_label}\n{frac * 100:.1f}%",
                            color=(30, 160, 50), anchor=(0.5, 0.5),
                        )
                        pct_txt.setPos(cx, cy)
                        vb.addItem(pct_txt)

        # Click → GateDetailWidget
        def _open_detail(_checked=False, _gd=eff, _gn=gate_name):
            dlg = GateDetailWidget(
                _gd, self.state, pnn,
                self._last_event_data, parent=self,
                parent_mask=self._last_gate_masks.get(_gn),
            )
            if dlg.exec() == QDialog.Accepted:
                self._refresh_display_counts()
                self.boundaries_changed.emit()
            self._rebuild_grid()

        btn_detail = QPushButton("Detail…")
        btn_detail.setFixedHeight(22)
        btn_detail.clicked.connect(_open_detail)
        layout.addWidget(btn_detail)

        return container

    def _refresh_display_counts(self):
        """Re-apply the current boundaries to the display events, updating
        each gate's parent masks and population fractions (after a manual
        adjustment, children of the adjusted gate change too)."""
        if self._last_event_data is None or not self.state.trained_boundaries:
            return
        channels = _experiment_channels(self.controller)
        transforms = self.controller.unmixed_transformations or {}
        data = ag_core.transform_columns(self._last_event_data, channels, transforms)
        index = {ch: i for i, ch in enumerate(channels)}
        result = ag_core.run_gating(self.state.gate_definitions, data, index,
                                    reference=self.state.trained_boundaries,
                                    default_mode=ag_core.MODE_FIXED)
        for gate_name, entries in result.boundaries.items():
            stored = self.state.trained_boundaries.get(gate_name, {})
            for pop, entry in entries.items():
                if pop in stored:
                    stored[pop]['fraction'] = entry.get('fraction')
        self._last_gate_masks = dict(result.parent_masks)

    # ------------------------------------------------------------------
    # Run / abort handlers
    # ------------------------------------------------------------------

    def _on_run_clicked(self):
        selected = self.picker.get_ordered_list()
        if not selected:
            QMessageBox.information(self, "No samples", "Select at least one training sample.")
            return
        if not self.state.gate_definitions:
            QMessageBox.information(self, "No gates",
                                    "Import or add gates on the Hierarchy tab first.")
            return
        try:
            ag_core.ordered_gate_defs(self.state.gate_definitions)
            snap = snapshot_unmix_state(self.controller, selected)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot train", str(exc))
            return
        self.state.training_samples = list(selected)

        import honeychrome.settings as hc_settings
        display_cap = int(getattr(hc_settings, 'max_display_events', 500_000) or 500_000)

        self.btn_run.setEnabled(False)
        self.btn_abort.setEnabled(True)
        self.lbl_status.setText("Running …")

        self._worker = _TrainingWorker(
            self.controller.experiment_dir, selected, snap,
            deepcopy(self.state.gate_definitions),
            _experiment_channels(self.controller),
            deepcopy(self.controller.unmixed_transformations or {}),
            events_per_sample=self.state.events_per_sample,
            display_cap=display_cap, seed=TRAINING_SEED, parent=self,
        )
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_abort_clicked(self):
        if self._worker is not None:
            self._worker.abort()

    def _on_worker_progress(self, msg: str):
        self.lbl_status.setText(msg)
        ts = datetime.now().strftime('%H:%M:%S')
        self.bus.statusMessage.emit(f"[GatingModelBuilder {ts}] {msg}")

    def _on_worker_finished(self, success: bool, error: str, payload: object):
        self.btn_run.setEnabled(True)
        self.btn_abort.setEnabled(False)
        self._worker = None
        if not success:
            self.lbl_status.setText(f"Error: {error}")
            ts = datetime.now().strftime('%H:%M:%S')
            self.bus.statusMessage.emit(f"[GatingModelBuilder {ts}] Training error: {error}")
            return

        self.state.trained_boundaries = payload['boundaries']
        self.state.training_summary = {
            'training_sample_names': list(payload['per_sample'].keys()),
            'n_events_per_sample': {k: v['n_events_used'] for k, v in payload['per_sample'].items()},
            'events_cap_per_sample': int(self.state.events_per_sample),
            'seed': TRAINING_SEED,
            'n_events_pooled': payload['n_pooled'],
            'per_sample': payload['per_sample'],
            'gate_status': payload['status'],
        }
        self._last_event_data = payload['display_events']
        self._last_gate_masks = payload['display_parent_masks']
        self._rebuild_grid()

        n_trained = len(payload['boundaries'])
        problems = [
            f"{name}: {st.get('message') or st.get('flag')}"
            for name, st in payload['status'].items() if st.get('flag')
        ]
        self.lbl_status.setText(f"Done — {n_trained} gate(s) trained.")
        self.trained.emit()
        text = (f"{n_trained} gate(s) trained on {payload['n_pooled']:,} events from "
                f"{len(payload['per_sample'])} sample(s).")
        if problems:
            text += "\n\nNot trained:\n" + "\n".join(problems[:15])
        QMessageBox.information(self, "Training Complete", text)


class ExportTab(QWidget):
    """Tab 2 — Export.

    Gate summary, model metadata, and Save/Load of ``.agmodel`` files
    (format 2: gates, trained boundaries, the gated channels' transforms,
    training samples with their AF assignments and QC state).
    """

    def __init__(self, state: GatingModelState, bus, controller, parent=None):
        super().__init__(parent)
        self.state = state
        self.bus = bus
        self.controller = controller
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # Heading
        heading = QLabel("<b>Export Model</b>")
        heading.setStyleSheet("font-size: 13pt;")
        outer.addWidget(heading)

        blurb = QLabel(
            "Review the gate summary below, then save the model as a "
            "portable <b>.agmodel</b> file.  Load an existing model file "
            "to inspect it or resume training."
        )
        blurb.setWordWrap(True)
        outer.addWidget(blurb)
        outer.addWidget(HelpToggleWidget(text=ag_help_texts.BUILDER_EXPORT))

        # Summary table — CopyableTableWidget takes (list_of_dicts, headers) at
        # construction and cannot be mutated in place; refresh() replaces the
        # widget by removing the old one and inserting a new one.
        summary_lbl = QLabel("<b>Gate summary</b>")
        outer.addWidget(summary_lbl)

        self._summary_headers = ["Gate name", "Type", "Algorithm", "Populations", "Status"]
        self.summary_table = CopyableTableWidget([], self._summary_headers)
        # Wrap in a container widget so we can swap the table on refresh.
        self._table_container = QWidget()
        self._table_layout = QVBoxLayout(self._table_container)
        self._table_layout.setContentsMargins(0, 0, 0, 0)
        self._table_layout.addWidget(self.summary_table)
        outer.addWidget(self._table_container, stretch=1)

        # Metadata card
        self.lbl_meta = QLabel("")
        self.lbl_meta.setWordWrap(True)
        self.lbl_meta.setStyleSheet(
            "QLabel { border: 1px solid palette(mid); border-radius: 4px; padding: 6px; }"
        )
        outer.addWidget(self.lbl_meta)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_save = QPushButton("Save Model…")
        self.btn_load = QPushButton("Load Model…")
        self.btn_save.setToolTip("Save the current model to a .agmodel file.")
        self.btn_load.setToolTip(
            "Load a .agmodel file for inspection or resuming training."
        )
        self.btn_save.clicked.connect(self._on_save_clicked)
        self.btn_load.clicked.connect(self._on_load_clicked)
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_load)
        btn_row.addStretch()
        outer.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Public refresh hook
    # ------------------------------------------------------------------

    def refresh(self):
        """Repopulate the summary table and metadata card."""
        self._repopulate_table()
        self._update_meta_card()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _repopulate_table(self):
        """Rebuild the CopyableTableWidget from state.gate_definitions.

        CopyableTableWidget is immutable after construction, so we remove
        the old widget and insert a fresh one into _table_layout.
        """
        rows = []
        for g in sorted(
            self.state.gate_definitions,
            key=lambda g: (int(g.get('gate_number', 999)), g.get('gate_name', ''))
        ):
            gate_name = g.get('gate_name', '')
            pops = ', '.join(g.get('populations', {}).keys())
            trained = self.state.is_trained(gate_name)
            rows.append({
                "Gate name":   gate_name,
                "Type":        g.get('gate_type', ''),
                "Algorithm":   g.get('algorithm', ''),
                "Populations": pops,
                "Status":      "Trained" if trained else "Untrained",
            })

        # Remove the old table widget.
        old = self._table_layout.takeAt(0)
        if old and old.widget():
            old.widget().deleteLater()

        # Build a fresh one.
        new_table = CopyableTableWidget(rows, self._summary_headers)
        self._table_layout.addWidget(new_table)
        self.summary_table = new_table

    def _update_meta_card(self):
        n_gates = self.state.n_gates()
        n_pop   = self.state.n_populations()
        n_trained = sum(
            1 for g in self.state.gate_definitions
            if self.state.is_trained(g.get('gate_name', ''))
        )
        n_samples = len(self.state.training_samples)
        try:
            exp_name = str(self.controller.experiment_dir.name)
        except Exception:
            exp_name = '—'
        last_path = self.state.model_metadata.get('last_model_path', '') or '—'
        self.lbl_meta.setText(
            f"<b>Experiment:</b> {exp_name}&nbsp;&nbsp; "
            f"<b>Gates:</b> {n_gates} ({n_trained} trained)&nbsp;&nbsp; "
            f"<b>Populations:</b> {n_pop}&nbsp;&nbsp; "
            f"<b>Training samples:</b> {n_samples}<br>"
            f"<b>Last saved/loaded:</b> {last_path}"
        )

    def _on_save_clicked(self):
        default_dir = ''
        try:
            default_dir = str(self.controller.experiment_dir)
        except Exception:
            pass
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Gating Model", default_dir, "Gating model (*.agmodel)"
        )
        if not path:
            return
        if not path.endswith('.agmodel'):
            path += '.agmodel'
        try:
            ag_core.write_model(path, build_model_from_state(self.state, self.controller))
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.state.model_metadata['last_model_path'] = path
        self._update_meta_card()
        ts = datetime.now().strftime('%H:%M:%S')
        self.bus.statusMessage.emit(f"[GatingModelBuilder {ts}] Model saved: {path}")

    def _on_load_clicked(self):
        default_dir = ''
        try:
            default_dir = str(self.controller.experiment_dir)
        except Exception:
            pass
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Gating Model", default_dir, "Gating model (*.agmodel)"
        )
        if not path:
            return
        try:
            model, warnings_list = ag_core.read_model(path)
            model, remap_messages = ag_core.remap_model(
                model, _live_transform_params(self.controller))
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", f"Could not read model:\n{exc}")
            return
        warnings_list = list(warnings_list) + list(remap_messages)
        apply_model_to_state(model, self.state)
        self.state.model_metadata['last_model_path'] = path
        if warnings_list:
            QMessageBox.information(self, "Model loaded", "\n\n".join(warnings_list))
        self.refresh()
        ts = datetime.now().strftime('%H:%M:%S')
        self.bus.statusMessage.emit(
            f"[GatingModelBuilder {ts}] Model loaded: {path} ({self.state.n_gates()} gates)"
        )


def build_model_from_state(state: GatingModelState, controller) -> dict:
    """Assemble a format-2 .agmodel dict from the builder state.

    Main thread: reads the experiment's channel names, the live display
    transforms of every gated channel, and the training samples' AF
    assignments.
    """
    antigens = _antigen_map(controller)
    channel_map = {ch: (antigen or ch) for ch, antigen in antigens.items()}

    by_name = ag_core.gates_by_name(state.gate_definitions)
    gated = []
    for g in state.gate_definitions:
        for ch in ag_core.gate_channels(ag_core.effective_gate_def(g, by_name)):
            if ch not in gated:
                gated.append(ch)
    live = controller.unmixed_transformations or {}
    transforms = {ch: ag_core.transform_params(live[ch]) for ch in gated if ch in live}

    samples = controller.experiment.samples
    af_assign = samples.get('sample_af_profiles', {}) or {}
    training = list(state.training_summary.get('training_sample_names') or state.training_samples)
    summary = dict(state.training_summary)
    summary.setdefault('training_sample_names', training)
    per_sample = summary.get('per_sample') or {}
    qc_fingerprints = {
        k: {'n_events_file': v.get('n_events_file'), 'n_events_kept': v.get('n_events_kept')}
        for k, v in per_sample.items()
        if v.get('n_events_kept') is not None and v.get('n_events_kept') != v.get('n_events_file')
    }
    try:
        exp_name = str(controller.experiment_dir.name)
    except Exception:
        exp_name = ''
    metadata = {
        'created': datetime.now().isoformat(),
        'honeychrome_version': getattr(honeychrome, '__version__', ''),
        'experiment_name': exp_name,
        'cytometer_hint': controller.experiment.settings.get('raw', {}).get('cytometer', ''),
    }
    return ag_core.new_model(
        state.gate_definitions,
        ag_core.serialisable_boundaries(state.trained_boundaries),
        channel_map=channel_map,
        transforms=transforms,
        training_summary=summary,
        metadata=metadata,
        af={'sample_af_profiles': {k: list(af_assign.get(k, [])) for k in training}},
        qc={'enabled': bool(qc_fingerprints), 'settings': {}, 'fingerprints': qc_fingerprints},
    )


def apply_model_to_state(model: dict, state: GatingModelState):
    """Load a normalised model's gates, boundaries and training record."""
    state.gate_definitions = list(model.get('gate_definitions', []))
    state.trained_boundaries = dict(model.get('trained_boundaries', {}))
    summary = dict(model.get('training_summary', {}))
    state.training_summary = summary
    state.training_samples = list(summary.get('training_sample_names', []))
    cap = summary.get('events_cap_per_sample')
    if isinstance(cap, int):
        state.events_per_sample = cap
    state.model_metadata.update(model.get('metadata', {}))


# ---------------------------------------------------------------------------
# 5.  PluginWidget — outer container, fulfils the plugin contract
#
# Inserted as a tab in the Honeychrome main window by plugin_loaders.py.
# Owns the GatingModelState and all three inner tabs.
#
# The outer layer follows the standard Honeychrome plugin pattern:
#   • a disabled-label shown when an unmixing matrix is not yet available;
#   • a scrollable content widget shown when it is.
#
# Both visibilities are toggled by `_on_mode_change()` in response to the
# bus's `modeChangeRequested` signal.  No other widget in the application
# changes the visibility of these two.
# ---------------------------------------------------------------------------


class PluginWidget(QWidget):
    """Top-level widget for the Gating Model Builder plugin.

    Parameters
    ----------
    bus
        Honeychrome EventBus — shared with the rest of the application.
    controller
        Main Honeychrome Controller — owns the experiment, sample
        registry, transfer matrix, transforms, and live GatingStrategy.
    parent
        Qt parent (the main window's QTabWidget).
    """

    def __init__(self, bus=None, controller=None, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.controller = controller

        # ------------------------------------------------------------------
        # Main-thread initialisation.
        # _ensure_qt_imports() must precede any widget construction that
        # depends on pyqtgraph or the cytometry_plot_components.
        # ------------------------------------------------------------------
        _ensure_qt_imports()
        _suppress_third_party_warnings()

        # Shared state — passed by reference to all inner tabs.
        self.state = GatingModelState()

        # QSettings — keyed per experiment in save_state/load_state.
        # Two-part key per Qt convention: ('honeychrome', 'plugin_<name>').
        self._qsettings = QSettings('honeychrome', 'plugin_gating_model_builder')

        # Pending state placeholders — populated by load_state() and
        # consumed by each tab's refresh() once it has been built.
        # Used to restore selections that can only be applied after the
        # tab's UI has been refreshed against the current experiment.
        self._pending_training_samples: list = []
        self._pending_last_model_path: str = ''

        # The experiment whose persisted state is held in self.state.
        # save_state() writes there even after the controller has moved to
        # another experiment; load_state() runs only when it changes.
        self._plugin_was_active = False
        self._loaded_experiment_key = None
        self._loaded_experiment_dir = None

        # ------------------------------------------------------------------
        # Outer layout: disabled label + scrollable content area.
        # The disabled label is shown until the experiment has an
        # unmixing matrix; the content widget is hidden in that state.
        # ------------------------------------------------------------------
        self.label_disabled = QLabel(
            f"{plugin_name}: unmixed data not available.  "
            "Set up the spectral model first."
        )
        self.label_disabled.setWordWrap(True)
        self.label_disabled.setStyleSheet(
            "QLabel { color: #856404; background: #fff3cd; "
            "border: 1px solid #ffc107; border-radius: 4px; padding: 8px; }"
        )

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
        # Inner tab widget — three tabs per CONTEXT §4.1.
        # ------------------------------------------------------------------
        self.inner_tabs = QTabWidget()
        self.inner_tabs.setDocumentMode(True)

        self.hierarchy_tab = HierarchyTab(self.state, bus, controller)
        self.train_tab     = TrainTab(self.state, bus, controller)
        self.export_tab    = ExportTab(self.state, bus, controller)

        self.inner_tabs.addTab(self.hierarchy_tab, "Hierarchy")
        self.inner_tabs.addTab(self.train_tab,     "Train")
        self.inner_tabs.addTab(self.export_tab,    "Export")

        self.inner_tabs.currentChanged.connect(self._on_inner_tab_changed)
        self.train_tab.trained.connect(self._write_session)
        self.train_tab.boundaries_changed.connect(self._write_session)
        content_layout.addWidget(self.inner_tabs)

        # Start with the content area hidden until the bus tells us
        # unmixing is available.
        self.content_widget.setVisible(False)

        # ------------------------------------------------------------------
        # Bus wiring.
        # modeChangeRequested fires on every Honeychrome tab switch;
        # the handler must guard on `mode == plugin_name` before acting.
        # loadSampleRequested fires when the user picks a sample in the
        # sample browser; the handler defers reading by one event-loop
        # tick to allow controller.load_sample() to complete first.
        # ------------------------------------------------------------------
        self.bus.modeChangeRequested.connect(self._on_mode_change)
        self.bus.loadSampleRequested.connect(self._on_sample_selected)
        # Populate pickers on first show in case modeChangeRequested already
        # fired before this widget was constructed.
        QTimer.singleShot(0, self._deferred_initial_refresh)

    # ----------------------------------------------------------------------
    # Bus signal handlers
    # ----------------------------------------------------------------------

    def _on_mode_change(self, mode: str):
        """Show/hide the plugin content based on the active main-window tab.

        Honeychrome fires `modeChangeRequested(str)` for every tab switch
        in the main window.  Each plugin guards on `mode == plugin_name`
        and only responds when its own tab is now active.

        Even when the tab is active, we additionally gate on the
        existence of an unmixing matrix — without one, the plugin cannot
        meaningfully load events, so the disabled label is shown instead
        of the content widget.

        Leaving the plugin saves its state. Persisted state is read back
        only when the experiment has changed, so returning to the tab
        keeps whatever was trained or edited in the meantime.
        """
        if mode != plugin_name:
            if self._plugin_was_active:
                self.save_state()
                self._plugin_was_active = False
            return

        self._plugin_was_active = True
        if self.controller.experiment.process.get('unmixing_matrix') is not None:
            self.label_disabled.setVisible(False)
            self.content_widget.setVisible(True)
            self._ensure_state_loaded()
            self._refresh_active_tab()
        else:
            self.label_disabled.setVisible(True)
            self.content_widget.setVisible(False)

    def _ensure_state_loaded(self):
        """Load persisted state when the controller's experiment is not the
        one currently held in self.state."""
        current_key = self._settings_key()
        if current_key == self._loaded_experiment_key:
            return
        if self._loaded_experiment_key is not None:
            self.save_state()
        # The inner tabs share self.state by reference, so it is reset in
        # place rather than replaced.
        fresh = GatingModelState()
        for f in dataclass_fields(GatingModelState):
            setattr(self.state, f.name, getattr(fresh, f.name))
        # Claim the experiment before restoring, so a failed restore still
        # saves to the right place afterwards.
        self._loaded_experiment_key = current_key
        self._loaded_experiment_dir = Path(self.controller.experiment_dir)
        self.load_state()

    def _on_sample_selected(self, sample_path: str):
        """Defer sample-driven refresh by one event-loop tick.

        controller.load_sample() schedules its own work asynchronously;
        if we read controller state synchronously here we will see the
        previous sample.  QTimer.singleShot(0, ...) yields to the event
        loop so load_sample completes first.  This is the established
        pattern in dr_clustering_tab.py and is mandatory.
        """
        # Only act when this plugin's tab is the active one — otherwise
        # we would re-trigger expensive work on every sample click in
        # the rest of the app.
        if getattr(self.controller, 'current_mode', None) != plugin_name:
            return
        QTimer.singleShot(0, self._deferred_load)

    def _deferred_load(self):
        """Run after the event loop has processed the load-sample event.

        The builder's views load their own training samples, so nothing
        here depends on which sample the main window shows.
        """
        pass

    def _deferred_initial_refresh(self):
        """One-shot refresh called after construction to populate pickers
        if the plugin tab is already active when the plugin loads."""
        log.debug("initial refresh: unmixing matrix present = %s",
                  self.controller.experiment.process.get('unmixing_matrix') is not None)
        if self.controller.experiment.process.get('unmixing_matrix') is not None:
            self.label_disabled.setVisible(False)
            self.content_widget.setVisible(True)
            self._ensure_state_loaded()
            self._refresh_active_tab()

    # ----------------------------------------------------------------------
    # Inner-tab management
    # ----------------------------------------------------------------------

    def _on_inner_tab_changed(self, index: int):
        """Save state when leaving a tab, then refresh the newly active one."""
        self.save_state()
        self._refresh_tab_at(index)

    def _refresh_active_tab(self):
        self._refresh_tab_at(self.inner_tabs.currentIndex())

    def _refresh_tab_at(self, index: int):
        """Call refresh() on the tab at *index* if it exposes one."""
        tabs = [self.hierarchy_tab, self.train_tab, self.export_tab]
        if 0 <= index < len(tabs):
            tab = tabs[index]
            if hasattr(tab, 'refresh'):
                tab.refresh()

    # ----------------------------------------------------------------------
    # Convenience: progress / status messages routed through the bus.
    # Tabs and the future _GatingWorker call this via `self.parent()` /
    # the worker's plugin reference.
    # ----------------------------------------------------------------------

    def progress_message(self, msg: str):
        """Emit a timestamped status message on the bus."""
        ts = datetime.now().strftime('%H:%M:%S')
        self.bus.statusMessage.emit(f"[GatingModelBuilder {ts}] {msg}")

    # ----------------------------------------------------------------------
    # State persistence
    #
    # QSettings, keyed by experiment directory, holds the training-sample
    # selection, events per sample, the controls toggle, the last model path
    # and the gate hierarchy, all as JSON strings (the Linux INI backend
    # does not round-trip lists). The trained boundaries are kept in a
    # session model in the experiment's cache folder (SESSION_FILE), in the
    # same .agmodel format as an export, together with the transforms they
    # were calculated in, so training survives a restart and is converted if
    # a transform changes in the meantime.
    # ----------------------------------------------------------------------

    def _settings_key(self) -> str:
        """Return a QSettings group key unique to this experiment.

        Sanitised so it is safe as a settings group name across
        Windows / macOS / Linux backends.
        """
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

    def _write_session(self):
        """Write the gate hierarchy and trained boundaries to the session file.

        Skipped while the controller shows a different experiment, because
        the session records the transforms of the experiment it belongs to.
        """
        if not self._controller_shows_loaded_experiment():
            return
        path = self._loaded_experiment_dir / SESSION_FILE
        if not self.state.gate_definitions and not path.exists():
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            ag_core.write_model(path, build_model_from_state(self.state, self.controller))
        except Exception:
            log.exception("could not write the session model %s", path)

    def save_state(self):
        """Persist the plugin's session state for the experiment it was
        loaded from (see _ensure_state_loaded).

        Does nothing before any experiment has been loaded, so default
        state never overwrites a saved session.
        """
        if self._loaded_experiment_key is None:
            return
        s = self._qsettings
        s.beginGroup(self._loaded_experiment_key)
        try:
            s.setValue('training_samples_json', json.dumps(list(self.state.training_samples)))
            s.setValue('events_per_sample', int(self.state.events_per_sample))
            s.setValue('include_controls', bool(self.state.include_controls))
            s.setValue('last_model_path',
                       str(self.state.model_metadata.get('last_model_path', '') or ''))
            s.setValue('gate_definitions_json',
                       json.dumps(ag_core.to_jsonable(self.state.gate_definitions)))
            s.remove('training_samples')
        finally:
            s.endGroup()
        self._write_session()

    def load_state(self):
        """Restore persisted state for the loaded experiment.

        Values that depend on tab UI being built first are stashed on
        `self._pending_*` attributes and consumed by the relevant tab's
        refresh() implementation.
        """
        key = self._loaded_experiment_key or self._settings_key()
        gate_defs = None
        s = self._qsettings
        s.beginGroup(key)
        try:
            training = _settings_json(s.value('training_samples_json', ''), None)
            if not isinstance(training, list):
                training = _settings_str_list(s.value('training_samples', []))
            training = [str(t) for t in training]
            self._pending_training_samples = list(training)
            self.state.training_samples = list(training)

            events = _settings_int(s.value('events_per_sample', None))
            if events is not None and events >= 0:
                self.state.events_per_sample = events
            self.state.include_controls = _settings_bool(
                s.value('include_controls', None), self.state.include_controls)

            last_path = s.value('last_model_path', '')
            if last_path:
                self._pending_last_model_path = str(last_path)
                self.state.model_metadata['last_model_path'] = str(last_path)

            gate_defs = _settings_json(s.value('gate_definitions_json', ''), None)
            if isinstance(gate_defs, list):
                self.state.gate_definitions = gate_defs
        finally:
            s.endGroup()
        self._load_session(have_gates=isinstance(gate_defs, list))

    def _load_session(self, have_gates: bool):
        """Restore trained boundaries (and, when QSettings has none, the
        gate hierarchy) from the session file, converted to the current
        transforms."""
        if self._loaded_experiment_dir is None:
            return
        path = self._loaded_experiment_dir / SESSION_FILE
        if not path.exists():
            return
        try:
            model, _ = ag_core.read_model(path)
            model, messages = ag_core.remap_model(model, _live_transform_params(self.controller))
        except Exception:
            log.exception("could not read the session model %s", path)
            return
        if not have_gates:
            self.state.gate_definitions = list(model.get('gate_definitions', []))
        names = {g['gate_name'] for g in self.state.gate_definitions}
        self.state.trained_boundaries = {
            k: v for k, v in (model.get('trained_boundaries') or {}).items() if k in names
        }
        self.state.training_summary = dict(model.get('training_summary') or {})
        for msg in messages:
            self.progress_message(msg)


# ---------------------------------------------------------------------------
# QSettings value helpers
#
# The INI backend (Linux) returns every value as a string, an empty list as
# None and a single-element list as a bare string, so values are stored as
# JSON or scalars and read back through these helpers.
# ---------------------------------------------------------------------------

def _settings_json(value, default):
    """Decode a JSON string stored in QSettings, or return *default*."""
    if value is None or value == '':
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _settings_str_list(value) -> list:
    """A list stored directly in QSettings, as a list of strings."""
    if value is None or value == '':
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _settings_int(value):
    """An integer stored in QSettings, or None."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _settings_bool(value, default: bool) -> bool:
    """A boolean stored in QSettings (the INI backend returns 'true'/'false')."""
    if value is None or value == '':
        return default
    if isinstance(value, str):
        return value.strip().lower() in ('true', '1', 'yes')
    return bool(value)


def _live_transform_params(controller) -> dict:
    """{channel: transform parameter dict} for the experiment's current
    display transforms. Main thread only."""
    live = getattr(controller, 'unmixed_transformations', None) or {}
    return {ch: ag_core.transform_params(tr) for ch, tr in live.items()}
