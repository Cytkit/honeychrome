"""
ag_builder.py — model-building pages of the Automated Gating plugin
===================================================================
Builds the gating model that the Automated Gating plugin
(``automated_gating_tab.py``) applies:

  • Transforms — the axis transforms the model is trained and applied in
                (they start as the experiment's and are saved with the
                model; a loaded model's own transforms can be applied).
  • Import/Setup — import the gate hierarchy from the unmixed gating
                strategy or a FlowJo workspace, or load a saved model and
                match its channels to this experiment; choose each gate's
                type, algorithm and parameters.
  • Hierarchy — compare imported gates with their trainable form on review
                samples, convert, edit and flag them.
  • Train     — pick training samples; each gate's boundary is calculated
                from its parent population in a pooled, equal-sized sample
                of every training file (background worker, which can be
                paused and resumed), then inspected and adjusted gate by gate.
  • Export    — review the model and save it as an ``.agmodel`` file.

``BuilderPages`` owns these pages, their shared ``GatingModelState`` and
transforms, and their persistence; the plugin widget adds them before its
Run, Results and Enrichment pages. Samples are unmixed with their own
autofluorescence profile assignments, exactly as the main window does, and
time-QC keep-masks are applied when that module is installed. Gate
calculation and application live in ``ag_core.py``.

Not a ``*_tab.py`` file, so plugin_loaders.py does not load it as a tab.
"""

import json
import os as _os
import sys
from copy import deepcopy
from dataclasses import dataclass, field, fields as dataclass_fields
from datetime import datetime
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, QObject, QTimer, Signal, QSettings, QThread
from PySide6.QtGui import QBrush, QColor, QAction, QPalette
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QPushButton, QLabel, QComboBox, QGroupBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QSizePolicy,
    QSplitter, QLineEdit, QFileDialog, QMessageBox, QFormLayout,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QTreeWidget, QTreeWidgetItem,
    QAbstractItemView, QMenu, QToolButton,
    QDialog, QDialogButtonBox, QGridLayout, QTextEdit, QGroupBox,
    QRadioButton, QButtonGroup,
)

import honeychrome
from honeychrome.controller_components.job_gate import acquire_job, guard_while_busy, wait_if_busy
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
import ag_flowjo  # noqa: E402
import ag_results  # noqa: E402
import ag_review  # noqa: E402
from ag_core import (  # noqa: E402
    ALGORITHMS_BY_GATE_TYPE,
    ALWAYS_EXCLUDED_CHANNELS,
    GATE_TYPES,
    default_algorithm_for_type as _default_algorithm_for_type,
    default_populations_for_type as _default_populations_for_type,
)
import ag_help_texts  # noqa: E402
import transforms_panel as tp  # noqa: E402
from ag_widgets import (  # noqa: E402
    DecadeSpinBox,
    FilterBar,
    PopulationTree,
    SquareHolder,
    add_concertina_page,
    algorithm_combo,
    block_wheel,
    colour_algorithm_items,
    make_concertina,
    text_matches,
    zoom_axis_transform,
)

log = ag_core.get_logger(__name__)


# ---------------------------------------------------------------------------
# Deferred Qt imports: pyqtgraph and the cytometry plot components create Qt
# objects on import, so they are imported on the main thread from
# BuilderPages.__init__ rather than at module scope (module import runs on
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


THRESHOLD_FACTOR_TIP = (
    "Threshold factor: the calculated threshold is multiplied by this in raw "
    "(untransformed) intensity units. 2 places it at twice the calculated "
    "intensity (stricter), 0.5 at half. 1 leaves it as calculated. Not "
    "applied to a threshold at or below zero in raw units.")


def _threshold_factor_spin(value: float) -> QDoubleSpinBox:
    """Spin box for a threshold factor (ag_core.scale_threshold)."""
    spin = QDoubleSpinBox()
    spin.setRange(*ag_core.THRESHOLD_FACTOR_RANGE)
    spin.setDecimals(2)
    spin.setSingleStep(0.1)
    spin.setSuffix(" ×")
    spin.setValue(float(value))
    spin.setToolTip(THRESHOLD_FACTOR_TIP)
    return spin


def _set_threshold_factor(params: dict, key: str, value: float):
    """Store a threshold factor; the default is left out of gate_param."""
    if float(value) == ag_core.DEFAULT_THRESHOLD_FACTOR:
        params.pop(key, None)
    else:
        params[key] = float(value)


def _axis_cells(g: dict, antigens: dict | None = None) -> tuple[str, str]:
    """Text of a gate's X and Y axis columns: each channel's marker (its
    antigen), or the channel name when it has none. A 1dsep gate's channels
    are marked as the separation (sep) and view parameters, on the plot
    axes they are drawn on; a view of None is a histogram."""
    antigens = antigens or {}

    def name(ch):
        return (antigens.get(ch) or ch) if ch else ''

    if g.get('gate_type') != '1dsep':
        return name(g.get('gate_marker_x')), name(g.get('gate_marker_y'))
    sep = f"{name(g.get('gate_marker_x'))} (sep)"
    view = f"{name(g['gate_marker_y'])} (view)" if g.get('gate_marker_y') else "Histogram (view)"
    return (view, sep) if ag_core.separation_axis(g) == 'y' else (sep, view)


def _boundary_axes(gate_def: dict, boundary: dict | None) -> tuple[str, str]:
    """(x, y) channels to draw *boundary* of *gate_def* on ('' for a
    histogram's y); see ag_core.boundary_plot_channels."""
    x, y = ag_core.boundary_plot_channels(gate_def, boundary)
    return x or '', y or ''


def _preview_axes(g: dict) -> tuple[str, str]:
    """(x, y) channels a gate is previewed on: the plot its shapes were
    drawn on while it applies them, else the plot its type is drawn on
    ('' for a histogram's y)."""
    template = g.get('template') if g.get('algorithm') == ag_core.ALGORITHM_IMPORTED else None
    return _boundary_axes(g, template)


def _sample_label(key: str) -> str:
    """Short display name for a sample key."""
    return Path(str(key)).stem


# Item data role holding a populations-table row's name as loaded.
_ORIGINAL_NAME_ROLE = int(Qt.UserRole) + 1

# Filters of the gate lists (Import/Setup table, Hierarchy tree).
GATE_FILTERS = (('all', 'All gates'), ('unconverted', 'Unconverted'),
                ('conflicting', 'Conflicting algorithms'), ('flagged', 'Flagged conversions'),
                ('overlapping', 'Overlapping gates'))


def filtered_gate_names(state, mode: str) -> set | None:
    """Names of the gates a filter keeps (None: every gate)."""
    defs = state.gate_definitions
    if mode == 'unconverted':
        return {g['gate_name'] for g in defs if g.get('algorithm') == ag_core.ALGORITHM_IMPORTED}
    if mode == 'conflicting':
        return ag_review.conflicting_gate_names(defs)
    if mode == 'flagged':
        names = {g['gate_name'] for g in defs}
        return {n for n in state.conversion_flags if n in names}
    if mode == 'overlapping':
        names = {g['gate_name'] for g in defs}
        return {n for n in state.overlap_flags if n in names}
    return None


def filter_counts(state) -> dict:
    """{filter key: number of gates it keeps}."""
    return {key: len(filtered_gate_names(state, key) or ()) for key, _l in GATE_FILTERS[1:]}


# Gate status colours (Import/Setup table, Hierarchy tree).
STATUS_COLOURS = {'Unconverted': (192, 57, 43), 'Untrained': (148, 97, 0),
                  'Trained': (10, 122, 35)}


def gate_status(state, g: dict) -> str:
    """'Unconverted' (applies its shapes as drawn), 'Untrained' or 'Trained'."""
    if g.get('algorithm') == ag_core.ALGORITHM_IMPORTED:
        return 'Unconverted'
    return 'Trained' if state.is_trained(g.get('gate_name', '')) else 'Untrained'


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
        the Import/Setup tab and consumed by training.
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
    default_sep_axis
        Plot axis ('x' or 'y') given to the separation channel of 1dsep
        gates that are added or split here.
    conversion_flags
        {gate_name: {'shift', 'fraction', 'message'}} for imported gates
        whose trainable form lands far from the imported boundary on the
        review samples (see ReviewTab's Flag conversions).
    overlap_flags
        {gate_name: {'pairs', 'message'}} for gates with a population that
        holds largely the same events as a population elsewhere in the
        hierarchy (see ReviewTab's Flag overlaps, ag_review.overlap_flags).
    training_progress
        Set while a training run is paused: {'fingerprint', 'done'} with
        the gates trained so far (their boundaries are kept).
    transform_population
        Population of the experiment's gating hierarchy shown on the
        Transforms tab.
    transform_view
        Channels shown on the Transforms tab (PopulationTransformsPanel.view).
    model_transforms
        {channel: transform parameters} of the last saved model loaded, on
        this experiment's channels (the transforms it was trained in);
        empty when the gates came from elsewhere.
    transforms_custom
        True once the transforms were changed by hand (Auto-transform, axis
        drag, CSV load) since they were last reset or set to a model's.
    """

    gate_definitions: list = field(default_factory=list)
    trained_boundaries: dict = field(default_factory=dict)
    training_samples: list = field(default_factory=list)
    events_per_sample: int = DEFAULT_EVENTS_PER_SAMPLE
    include_controls: bool = False
    training_summary: dict = field(default_factory=dict)
    model_metadata: dict = field(default_factory=dict)
    algorithm_recommendations: dict = field(default_factory=dict)
    default_sep_axis: str = 'x'
    conversion_flags: dict = field(default_factory=dict)
    overlap_flags: dict = field(default_factory=dict)
    training_progress: dict = field(default_factory=dict)
    transform_population: str = ''
    transform_view: dict = field(default_factory=dict)
    model_transforms: dict = field(default_factory=dict)
    transforms_custom: bool = False

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
    mixture_cut
        Which two components, ordered by mean, cross at the threshold:
        ``outer`` (lowest and highest, default), ``lower`` or ``upper``.
    threshold_factor
        Multiplies the calculated threshold in raw units (default ``1``).
    width_sd
        Standard-deviation half-width for the singlet parallelogram.
        Default ``3``.
    edge_density_x, edge_density_y
        Density fall-off of a ``cells`` gate's sides on each scatter
        channel, as a fraction of the peak.  Default ``0.002``.
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
        algo = self.gate_def.get('algorithm')
        self.sb_axis_lo.setValue(float(params.get('axis_lo', ag_core.default_axis_lo(algo))))
        self.sb_axis_lo.setToolTip(
            "Crops the data fed into the threshold calculation to this "
            "fraction of the axis's own min-max range (not a percentile). "
            "Matches R's region.min. Default 0 for 'tail' (it measures the "
            "low side of the negative population), else 0.01."
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

        self.sb_tail_fraction = DecadeSpinBox()
        self.sb_tail_fraction.setValue(float(params.get('tail_fraction',
                                                        ag_core.DEFAULT_TAIL_FRACTION)))
        self.sb_tail_fraction.setToolTip(
            "Tail probability for the 'tail' algorithm. Smaller = stricter. "
            "The arrows step by factors of ten; type any other value."
        )
        form.addRow("Tail fraction", self.sb_tail_fraction)

        self.sb_mixture_components = QSpinBox()
        self.sb_mixture_components.setRange(2, 6)
        self.sb_mixture_components.setValue(int(params.get('mixture_components', 2)))
        self.sb_mixture_components.setToolTip(
            "Gaussian components for the 'mixture' algorithm."
        )
        form.addRow("Mixture components", self.sb_mixture_components)

        self.cb_mixture_cut = self._cut_combo(params.get('mixture_cut', 'outer'))
        form.addRow("Mixture cut", self.cb_mixture_cut)

        self.sb_threshold_factor = _threshold_factor_spin(ag_core.threshold_factor(params))
        form.addRow("Threshold factor", self.sb_threshold_factor)

        self.sb_width_sd = QDoubleSpinBox()
        self.sb_width_sd.setRange(0.5, 10.0)
        self.sb_width_sd.setDecimals(2)
        self.sb_width_sd.setSingleStep(0.1)
        self.sb_width_sd.setValue(float(params.get('width_sd', 3.0)))
        self.sb_width_sd.setToolTip(
            "Half-width of the singlet parallelogram, in residual SDs."
        )
        form.addRow("Singlet width (SDs)", self.sb_width_sd)

        self._is_free = self.gate_def.get('gate_type') == 'free'
        if self._is_free:
            self.sb_edge = {}
            for axis in ('x', 'y'):
                ch = self.gate_def.get(f'gate_marker_{axis}') or axis.upper()
                sb = DecadeSpinBox()
                sb.setValue(float(params.get(f'edge_density_{axis}',
                                             params.get('edge_density',
                                                        ag_core.DEFAULT_EDGE_DENSITY))))
                sb.setToolTip(
                    "Cells gate: its sides on this channel (other than the debris cut) are "
                    "the outermost points where the density is still this fraction of its "
                    f"peak (default {ag_core.DEFAULT_EDGE_DENSITY:g}). Smaller reaches "
                    "further out."
                )
                self.sb_edge[axis] = sb
                form.addRow(f"Cells fall-off, {ch}", sb)

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
        axis_algos = [a for a in ag_core.ALGORITHMS_BY_GATE_TYPE['2dsep']
                      if a != ag_core.ALGORITHM_IMPORTED]
        for cb, key in ((self.cb_algo_x, 'algorithm_x'),
                        (self.cb_algo_y, 'algorithm_y')):
            cb.addItems(axis_algos)
            cb.setToolTip("'replicate' takes this axis's threshold from another gate or a "
                          "reference population; choose the source on the Hierarchy tab.")
            val = params.get(key, '') or default_algo
            if val in axis_algos:
                cb.setCurrentText(val)
            block_wheel(cb)
            colour_algorithm_items(cb, ag_core.algorithm_colour)
        grid.addWidget(self.cb_algo_x, 1, 1)
        grid.addWidget(self.cb_algo_y, 1, 2)

        # Tail fraction
        grid.addWidget(QLabel("Tail fraction"), 2, 0)
        self.sb_tail_x = DecadeSpinBox()
        self.sb_tail_y = DecadeSpinBox()
        for sb, key in ((self.sb_tail_x, 'tail_fraction_x'),
                        (self.sb_tail_y, 'tail_fraction_y')):
            sb.setValue(float(params.get(key, params.get('tail_fraction',
                                                          ag_core.DEFAULT_TAIL_FRACTION))))
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

        self.cb_cut_x = self._cut_combo(params.get('mixture_cut_x', params.get('mixture_cut', 'outer')))
        self.cb_cut_y = self._cut_combo(params.get('mixture_cut_y', params.get('mixture_cut', 'outer')))
        grid.addWidget(self.cb_cut_x, 7, 1)
        grid.addWidget(self.cb_cut_y, 7, 2)
        grid.addWidget(QLabel("Mixture cut"), 7, 0)

        grid.addWidget(QLabel("Threshold factor"), 8, 0)
        self.sb_factor_x = _threshold_factor_spin(ag_core.threshold_factor(params, '_x'))
        self.sb_factor_y = _threshold_factor_spin(ag_core.threshold_factor(params, '_y'))
        grid.addWidget(self.sb_factor_x, 8, 1)
        grid.addWidget(self.sb_factor_y, 8, 2)

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
        for sb, key, axis in ((self.sb_axis_lo_x, 'axis_lo_x', 'x'),
                              (self.sb_axis_lo_y, 'axis_lo_y', 'y')):
            sb.setRange(0.0, 1.0)
            sb.setDecimals(3)
            sb.setSingleStep(0.01)
            axis_algo = params.get(f'algorithm_{axis}') or default_algo
            sb.setValue(float(params.get(key, ag_core.default_axis_lo(axis_algo))))
            sb.setToolTip(
                "Crops the data fed into the threshold calculation to "
                "this fraction of the axis's own min-max range. Default 0 for "
                "'tail', else 0.01."
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

    @staticmethod
    def _cut_combo(current: str) -> QComboBox:
        """Which two mixture components, ordered by mean, cross at the
        threshold (only matters with three or more)."""
        cb = QComboBox()
        for value, text in (('outer', 'lowest | highest'), ('lower', 'lowest two'),
                            ('upper', 'highest two')):
            cb.addItem(text, value)
        cb.setCurrentIndex(max(cb.findData(current), 0))
        cb.setToolTip("Which two Gaussian components, ordered by mean, cross at the threshold "
                      "of the 'mixture' algorithm. Matters with three or more components.")
        block_wheel(cb)
        return cb

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
            params['mixture_cut_x']        = self.cb_cut_x.currentData()
            params['mixture_cut_y']        = self.cb_cut_y.currentData()
            params['axis_lo_x']            = float(self.sb_axis_lo_x.value())
            params['axis_hi_x']            = float(self.sb_axis_hi_x.value())
            params['axis_lo_y']            = float(self.sb_axis_lo_y.value())
            params['axis_hi_y']            = float(self.sb_axis_hi_y.value())
            params.pop('threshold_factor', None)
            _set_threshold_factor(params, 'threshold_factor_x', self.sb_factor_x.value())
            _set_threshold_factor(params, 'threshold_factor_y', self.sb_factor_y.value())
        else:
            params['axis_lo']            = float(self.sb_axis_lo.value())
            params['axis_hi']            = float(self.sb_axis_hi.value())
            params['tail_fraction']      = float(self.sb_tail_fraction.value())
            params['mixture_components'] = int(self.sb_mixture_components.value())
            params['mixture_cut']        = self.cb_mixture_cut.currentData()
            params['width_sd']           = float(self.sb_width_sd.value())
            _set_threshold_factor(params, 'threshold_factor', self.sb_threshold_factor.value())
            if self._is_free:
                for axis, sb in self.sb_edge.items():
                    params[f'edge_density_{axis}'] = float(sb.value())
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

        self.lbl_marker_x = QLabel("X-axis channel")
        self.lbl_marker_y = QLabel("Y-axis channel")
        form.addRow(self.lbl_marker_x, self.cb_marker_x)
        form.addRow(self.lbl_marker_y, self.cb_marker_y)

        # 1dsep only: which plot axis carries the separation parameter.
        self.lbl_sep_axis = QLabel("Separation drawn on")
        self.cb_sep_axis = QComboBox()
        self.cb_sep_axis.addItem("X axis", 'x')
        self.cb_sep_axis.addItem("Y axis", 'y')
        sep_axis = self.gate_def.get('sep_axis') or (
            self.state.default_sep_axis if self.is_new else 'x')
        self._select_combo_by_data(self.cb_sep_axis, sep_axis)
        self.cb_sep_axis.setToolTip(
            "Plot axis the separation parameter is drawn on; the view parameter "
            "takes the other axis. A histogram view always draws the separation "
            "parameter on the x axis.")
        form.addRow(self.lbl_sep_axis, self.cb_sep_axis)
        form.addRow("", self.cb_include_all_channels)
        self.cb_marker_y.currentIndexChanged.connect(self._update_axis_labels)

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
        self._update_axis_labels()

    def _update_axis_labels(self, *_args):
        """Name the axis fields after their role: a 1dsep gate has a
        separation parameter and a view parameter (or a histogram)."""
        if not hasattr(self, 'lbl_sep_axis'):
            return
        one_d = self.cb_type.currentText() == '1dsep'
        self.lbl_marker_x.setText("Separation parameter (sep)" if one_d else "X-axis channel")
        self.lbl_marker_y.setText("View parameter (view)" if one_d else "Y-axis channel")
        self.cb_marker_y.setItemText(0, "Histogram" if one_d else "(none)")
        has_view = self.cb_marker_y.currentData() is not None
        self.cb_sep_axis.setEnabled(one_d and has_view)
        self.lbl_sep_axis.setEnabled(one_d and has_view)

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
        algos = list(ALGORITHMS_BY_GATE_TYPE.get(gate_type, []))
        # 'imported' applies the stored boundary, so it needs a boundary of this gate type.
        if not self.gate_def.get('template') or self.gate_def.get('gate_type') != gate_type:
            algos = [a for a in algos if a != ag_core.ALGORITHM_IMPORTED]
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
        colour_algorithm_items(self.cb_algorithm, ag_core.algorithm_colour)

    def _on_type_changed(self, new_type: str):
        """Adjust dependent fields when the gate type changes.

        Between 1dsep and 2dsep the population names are kept, so gates
        drawn on them stay attached; their regions are converted on accept
        (ag_review.convert_threshold_regions). Any other change resets the
        populations table to the new type's defaults; accept refuses a
        change that leaves gates below without their parent population.
        """
        # Y-axis combo is disabled only for replicate gates (no display axes).
        # 1dsep gates can carry a display Y channel so the tile shows a biplot
        # background rather than a histogram-only view.
        is_2d_like = new_type in ('1dsep', '2dsep', 'free', 'singlets')
        self.cb_marker_y.setEnabled(is_2d_like)
        self._update_axis_labels()

        if new_type == self._last_seen_type:
            return
        previous, self._last_seen_type = self._last_seen_type, new_type

        if {previous, new_type} <= set(ag_core.THRESHOLD_TYPES):
            self.lbl_type_change_notice.setText(
                "Population names are kept, so the gates below stay attached; each takes its "
                "side of the separation parameter when you press OK.")
            return
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
        name_item.setData(_ORIGINAL_NAME_ROLE, str(name))
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

    def _population_renames(self) -> dict:
        """{name the row was loaded with: name now} for renamed rows."""
        out = {}
        for row in range(self.tbl_populations.rowCount()):
            item = self.tbl_populations.item(row, 0)
            if item is None:
                continue
            original = item.data(_ORIGINAL_NAME_ROLE)
            name = item.text().strip()
            if original and name and original != name:
                out[str(original)] = name
        return out

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
            QMessageBox.warning(self, "Validation",
                                "Separation parameter is required." if gate_type == '1dsep'
                                else "X-axis channel is required.")
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
        renames = {}
        if not self.is_new:
            populations, problem = ag_review.convert_threshold_regions(
                populations, self.gate_def.get('gate_type'), self.gate_def.get('gate_marker_x'),
                self.gate_def.get('gate_marker_y'), gate_type, marker_x, marker_y)
            if populations is None:
                QMessageBox.warning(self, "Validation", problem)
                return
            renames = {o: n for o, n in self._population_renames().items() if n in populations}
            removed = (set(self.gate_def.get('populations') or {}) - set(populations)
                       - set(renames))
            orphans = ag_review.orphaned_by_populations(
                self.state.gate_definitions, self.gate_def.get('gate_name'), removed)
            if orphans:
                QMessageBox.warning(
                    self, "Validation",
                    "These gates are drawn on populations this change removes: "
                    + ", ".join(f"'{o}'" for o in orphans)
                    + ". Keep those population names, or move or delete those gates first.")
                return

        origin_text = self.cb_origin_gate.currentText()
        origin_gate = None if origin_text in ("(none)", "") else origin_text
        if gate_type == 'replicate' and not origin_gate:
            QMessageBox.warning(
                self, "Validation",
                "Replicate gates require an origin gate."
            )
            return

        if renames:
            ag_review.rename_populations(self.state.gate_definitions,
                                         self.gate_def.get('gate_name'), renames,
                                         self.state.trained_boundaries)
        # Write back into the supplied dict.  Caller takes ownership.
        self.gate_def['gate_name']     = name
        self.gate_def['gate_type']     = gate_type
        self.gate_def['gate_marker_x'] = marker_x
        self.gate_def['gate_marker_y'] = marker_y
        self.gate_def['sep_axis']      = (self.cb_sep_axis.currentData()
                                          if gate_type == '1dsep' and marker_y else 'x')
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

class SplitGateDialog(QDialog):
    """Choose how a 2dsep gate is split into sequential 1dsep gates.

    One row per step, first step first: the separation parameter, its
    algorithm, its view parameter (the other channel, or a histogram) and
    the plot axis the separation is drawn on. "Swap order" separates on
    the other channel first. ``steps()`` gives the choices in order for
    ``ag_review.split_2dsep``.
    """

    _COLUMNS = ["Step", "Separation parameter", "Algorithm", "View parameter", "Separation drawn on"]

    def __init__(self, gate_def: dict, controller, default_algorithm: str, parent=None):
        super().__init__(parent)
        self.gate_def = gate_def
        self.controller = controller
        self.setWindowTitle(f"Split into 1D gates — {gate_def.get('gate_name', '')}")
        self.setModal(True)
        self.resize(720, 280)
        x, y = gate_def.get('gate_marker_x'), gate_def.get('gate_marker_y')
        params = gate_def.get('gate_param') or {}
        algos = list(ag_core.THRESHOLD_ALGORITHMS)
        self._algorithms = algos
        self._order = [x, y]
        self._choices = {}
        for ch, axis in ((x, 'x'), (y, 'y')):
            algo = params.get(f'algorithm_{axis}') or default_algorithm
            self._choices[ch] = {
                'algorithm': algo if algo in algos else 'tail',
                'view': y if ch == x else x,
                'sep_axis': axis,
            }

        outer = QVBoxLayout(self)
        intro = QLabel(
            "The first step separates the parent population on its parameter; on each "
            "side that holds a population of this gate, the second step separates on the "
            "other parameter. Populations keep their names, so the gates below them stay "
            "attached. The new gates use the algorithms chosen here and must be trained.")
        intro.setWordWrap(True)
        outer.addWidget(intro)
        self.tbl = QTableWidget(2, len(self._COLUMNS))
        self.tbl.setHorizontalHeaderLabels(self._COLUMNS)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setSelectionMode(QAbstractItemView.NoSelection)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        for col, width in enumerate((40, 190, 110, 190)):
            self.tbl.setColumnWidth(col, width)
        outer.addWidget(self.tbl)
        row = QHBoxLayout()
        self.btn_swap = QPushButton("Swap order")
        self.btn_swap.setToolTip("Separate on the other parameter first.")
        self.btn_swap.clicked.connect(self._swap)
        row.addWidget(self.btn_swap)
        row.addStretch()
        outer.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self._fill()

    def _label(self, ch: str) -> str:
        antigen = _channel_to_antigen(self.controller, ch)
        return ch if antigen == ch else f"{ch} — {antigen}"

    def _read(self):
        for r, ch in enumerate(self._order):
            algo = self.tbl.cellWidget(r, 2)
            view = self.tbl.cellWidget(r, 3)
            axis = self.tbl.cellWidget(r, 4)
            if algo is None:
                continue
            self._choices[ch] = {'algorithm': algo.currentText(), 'view': view.currentData(),
                                 'sep_axis': axis.currentData()}

    def _swap(self):
        self._read()
        self._order.reverse()
        self._fill()

    def _fill(self):
        for r, ch in enumerate(self._order):
            choice = self._choices[ch]
            other = self._order[1 - r]
            self.tbl.setItem(r, 0, QTableWidgetItem(str(r + 1)))
            self.tbl.setItem(r, 1, QTableWidgetItem(f"{self._label(ch)} (sep)"))
            algo = algorithm_combo(self._algorithms, choice['algorithm'], ag_core.algorithm_colour)
            view = QComboBox()
            view.addItem(f"{self._label(other)} (view)", other)
            view.addItem("Histogram", None)
            view.setCurrentIndex(0 if choice['view'] else 1)
            axis = QComboBox()
            axis.addItem("X axis", 'x')
            axis.addItem("Y axis", 'y')
            axis.setCurrentIndex(1 if choice['sep_axis'] == 'y' else 0)
            view.currentIndexChanged.connect(
                lambda _i, v=view, a=axis: a.setEnabled(v.currentData() is not None))
            axis.setEnabled(view.currentData() is not None)
            self.tbl.setCellWidget(r, 2, algo)
            self.tbl.setCellWidget(r, 3, view)
            self.tbl.setCellWidget(r, 4, axis)

    def steps(self) -> list[dict]:
        """[{channel, algorithm, view, sep_axis}] in gating order."""
        self._read()
        out = []
        for ch in self._order:
            c = self._choices[ch]
            out.append({'channel': ch, 'algorithm': c['algorithm'], 'view': c['view'],
                        'sep_axis': c['sep_axis'] if c['view'] else 'x'})
        return out


# ---------------------------------------------------------------------------
# Import/Setup tab
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
            trained = ag_core.train_boundaries(self._gate_defs, data, idx,
                                               transforms=self._transforms)
            recs = {}
            for g in self._gate_defs:
                parent = trained.parent_masks.get(g['gate_name'])
                sub = data[parent] if parent is not None else data
                recs[g['gate_name']] = ag_core.recommend_algorithm(g, sub, idx)
            self.finished.emit(True, '', recs)
        except Exception as exc:
            log.exception("algorithm recommendation failed")
            self.finished.emit(False, str(exc), {})


def _flowjo_experiment_channels(controller, transforms: dict | None = None) -> 'ag_flowjo.ExperimentChannels':
    """The experiment's unmixed channels in the form ag_flowjo aligns against;
    gate coordinates are converted to *transforms* ({channel: Transform},
    the experiment's display transforms when None)."""
    import honeychrome.settings as hc_settings
    channels = _experiment_channels(controller)
    fluorescence = _fluorescence_channels(controller)
    excluded = set(ALWAYS_EXCLUDED_CHANNELS) | set(getattr(hc_settings, 'af_channels', ()) or ())
    scatter = [c for c in channels if c not in fluorescence and c not in excluded]
    cytometer = ''
    try:
        cytometer = controller.experiment.settings.get('raw', {}).get('cytometer', '')
    except Exception:
        pass
    params = {}
    source = transforms if transforms is not None else (controller.unmixed_transformations or {})
    for ch, tr in source.items():
        try:
            params[ch] = ag_core.transform_params(tr)
        except Exception:
            continue
    return ag_flowjo.ExperimentChannels(
        channels=channels,
        fluorescence=fluorescence,
        scatter=scatter,
        antigens=_antigen_map(controller),
        transforms=params,
        singlet_pair=singlet_channels(cytometer),
    )


def _flowjo_matchers():
    """(fluorophore matcher, marker matcher) from the label databases."""
    from honeychrome.controller_components.label_matching import (
        get_fluorophore_db, get_marker_db, match_fluorophore, match_marker,
    )
    try:
        fluor_db, marker_db = get_fluorophore_db(), get_marker_db()
    except Exception:
        return None, None
    return (lambda n: match_fluorophore(n, fluor_db),
            lambda n: match_marker(n, marker_db))


class FlowJoImportPanel(QWidget):
    """Page of the Import/Setup tab for importing a FlowJo workspace.

    Opened when a workspace is chosen. It shows the gating source, the
    proposed channel for every FlowJo parameter the gates use (row text in
    the default colour: confident, amber: check, bold red: unmatched and
    not yet resolved, grey italic: dropped), and inline messages about
    populations that cannot be imported and conversions that change a
    gate. Unmatched parameters must be assigned a channel or dropped before
    the import buttons enable. Populations using a dropped parameter, and
    those below them, are not imported. The model changes only when an import button is pressed,
    which emits ``importRequested(gate_defs, report, replace, name)``.
    ``titleChanged(text)`` gives the heading for the page, and
    ``closeRequested`` asks the host to leave the page. The parameter table
    and the messages can be searched, and the table narrowed to the rows
    that need a decision (amber and red).
    """

    importRequested = Signal(object, object, bool, str)
    titleChanged = Signal(str)
    closeRequested = Signal()

    _DROP = "(drop)"
    _DROPPED = "__dropped__"
    _UNSET = "__unset__"
    _COLUMNS = ["FlowJo parameter", "Label", "Experiment channel", "Match", "Note"]
    _RED = {False: QColor('#c00000'), True: QColor('#ff6b6b')}    # by is_dark
    _AMBER = {False: QColor('#b06000'), True: QColor('#f0a030')}
    _GREY = QColor(128, 128, 128)

    def __init__(self, controller, existing_names, transforms=None, parent=None):
        super().__init__(parent)
        self.controller = controller
        self._existing_names = existing_names
        self._transforms = transforms
        self._notes: list = []
        self.workspace = None
        self.exp = None
        self._fluor_match = self._marker_match = None
        self._combos: dict = {}
        self._matches: dict = {}
        self._params: list = []
        self._name = ''

        body = QVBoxLayout(self)
        body.setContentsMargins(6, 6, 6, 6)

        self.lbl_status = QLabel()
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setTextFormat(Qt.RichText)
        body.addWidget(self.lbl_status)

        form = QFormLayout()
        self.combo_source = block_wheel(QComboBox())
        form.addRow("Gating source:", self.combo_source)
        body.addLayout(form)

        self.filter_bar = FilterBar([('all', 'All parameters'), ('check', 'Flagged or unmatched')],
                                    placeholder="Search parameters, labels, channels and messages…")
        self.filter_bar.changed.connect(lambda _t, _m: self._apply_filter())
        body.addWidget(self.filter_bar)

        self.tbl = QTableWidget(0, len(self._COLUMNS))
        self.tbl.setHorizontalHeaderLabels(self._COLUMNS)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setSelectionMode(QAbstractItemView.NoSelection)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        for col, width in enumerate((160, 110, 190, 90)):
            self.tbl.setColumnWidth(col, width)
        self.tbl.setMinimumHeight(480)
        body.addWidget(self.tbl, stretch=3)

        opts = QHBoxLayout()
        self.btn_drop_unmatched = QPushButton("Drop all unmatched")
        self.btn_drop_unmatched.setToolTip(
            "Drop every unmatched parameter, and with it the populations that use it.")
        self.chk_merge = QCheckBox("Combine sibling gates drawn on the same parameters")
        self.chk_merge.setChecked(True)
        self.chk_merge.setToolTip(
            "Sibling gates drawn on the same pair of parameters (CD4 and CD8, "
            "quadrants) become one gate with one population each, trained "
            "as one 2dsep gate. Each gate keeps its shape as drawn until it "
            "is converted to trainable.")
        opts.addWidget(self.btn_drop_unmatched)
        opts.addWidget(self.chk_merge)
        opts.addStretch()
        body.addLayout(opts)

        self.txt_notes = QTextEdit()
        self.txt_notes.setReadOnly(True)
        self.txt_notes.setMinimumHeight(260)
        self.txt_notes.setPlaceholderText("Populations that cannot be imported and what each gate converts to are listed here.")
        body.addWidget(self.txt_notes, stretch=2)

        buttons = QHBoxLayout()
        self.btn_add = QPushButton("Import all — add")
        self.btn_replace = QPushButton("Import all — replace")
        self.btn_cancel = QPushButton("Close")
        self.btn_add.setToolTip("Add the imported gates to the model's existing gates.")
        self.btn_replace.setToolTip("Replace the model's gates with the imported ones.")
        buttons.addStretch()
        for b in (self.btn_add, self.btn_replace, self.btn_cancel):
            buttons.addWidget(b)
        body.addLayout(buttons)

        self.combo_source.currentIndexChanged.connect(self._load_source)
        self.chk_merge.toggled.connect(self._refresh)
        self.btn_drop_unmatched.clicked.connect(self._drop_unmatched)
        self.btn_add.clicked.connect(lambda: self._import(replace=False))
        self.btn_replace.clicked.connect(lambda: self._import(replace=True))
        self.btn_cancel.clicked.connect(self.close_section)

    # -- opening and closing ------------------------------------------------

    def open_workspace(self, path: str):
        """Read *path* and show its alignment for review."""
        self._name = Path(path).name
        self.titleChanged.emit(f"FlowJo import — {self._name}")
        self.workspace = None
        self.combo_source.blockSignals(True)
        self.combo_source.clear()
        self.combo_source.blockSignals(False)
        self.tbl.setRowCount(0)
        self.txt_notes.clear()
        try:
            workspace = ag_flowjo.read_workspace(path)
        except Exception as exc:
            self._fail(f"Could not read the workspace: {exc}")
            return
        default = workspace.default_source()
        if default is None:
            self._fail("The workspace contains no gated populations.")
            return
        self.workspace = workspace
        self.exp = _flowjo_experiment_channels(
            self.controller, self._transforms() if self._transforms is not None else None)
        self._fluor_match, self._marker_match = _flowjo_matchers()
        self.combo_source.blockSignals(True)
        for s in workspace.sources:
            if s.n_populations:
                self.combo_source.addItem(s.title, s)
        for i in range(self.combo_source.count()):
            if self.combo_source.itemData(i) is default:
                self.combo_source.setCurrentIndex(i)
        self.combo_source.blockSignals(False)
        self._load_source()

    def close_section(self):
        self.closeRequested.emit()

    def collapse(self, summary: str):
        """Leave the page with a one-line summary in its heading after an import."""
        self.titleChanged.emit(f"FlowJo import — {self._name}: {summary}")
        self.closeRequested.emit()

    def _fail(self, message: str):
        self.lbl_status.setText(f"<span style='color:#a00000'>{message}</span>")
        for w in (self.btn_add, self.btn_replace, self.btn_drop_unmatched):
            w.setEnabled(False)

    # -- mapping table ------------------------------------------------------

    def _source(self):
        return self.combo_source.currentData()

    def _load_source(self):
        """Propose a mapping for the selected source and fill the table."""
        source = self._source()
        self.tbl.setRowCount(0)
        self._combos, self._matches, self._params = {}, {}, []
        if source is None or self.workspace is None:
            return
        self._params = ag_flowjo.gated_parameters(source)
        proposal = ag_flowjo.align_parameters(
            self._params, source.labels, self.exp, affixes=self.workspace.affixes,
            fluor_match=self._fluor_match, marker_match=self._marker_match)
        self.tbl.setRowCount(len(self._params))
        for row, param in enumerate(self._params):
            hit = proposal.get(param, {})
            channel = hit.get('channel')
            self._matches[param] = hit.get('method') if channel else None
            self.tbl.setItem(row, 0, QTableWidgetItem(param))
            self.tbl.setItem(row, 1, QTableWidgetItem(hit.get('label') or source.labels.get(param, '') or ''))
            combo = block_wheel(QComboBox())
            if not channel:
                combo.addItem("— choose a channel or drop —", self._UNSET)
            combo.addItem(self._DROP, self._DROPPED)
            for ch in self.exp.channels:
                if ch not in ALWAYS_EXCLUDED_CHANNELS or ch == channel:
                    combo.addItem(ch, ch)
            combo.setCurrentIndex(max(combo.findData(channel), 0) if channel else 0)
            combo.currentIndexChanged.connect(lambda _i, p=param: self._on_edited(p))
            self.tbl.setCellWidget(row, 2, combo)
            self._combos[param] = combo
            self.tbl.setItem(row, 3, QTableWidgetItem(hit.get('method') or 'unmatched'))
            self.tbl.setItem(row, 4, QTableWidgetItem(hit.get('note') or ''))
        for param in self._params:
            self._colour_row(param)
        self._refresh()
        self._apply_filter()

    def _needs_check(self, param: str) -> bool:
        """Amber or red: unresolved, or matched without confidence."""
        value = self._combos[param].currentData()
        if value == self._UNSET:
            return True
        if value == self._DROPPED:
            return False
        return self._matches.get(param) not in ag_flowjo.CONFIDENT_MATCHES

    def _apply_filter(self):
        """Show the rows and messages matching the search and the "show"
        choice."""
        text, mode = self.filter_bar.text(), self.filter_bar.mode()
        n_check = 0
        for row, param in enumerate(self._params):
            check = self._needs_check(param)
            n_check += int(check)
            cells = [self.tbl.item(row, c).text() for c in (0, 1, 3, 4) if self.tbl.item(row, c)]
            combo = self._combos.get(param)
            shown = text_matches(text, param, combo.currentText() if combo else '', *cells)
            if mode == 'check':
                shown = shown and check
            self.tbl.setRowHidden(row, not shown)
        self.filter_bar.set_mode_label('check', f"Flagged or unmatched ({n_check})")
        lines = [m for m in self._notes if text_matches(text, m)] if text else list(self._notes)
        self.txt_notes.setPlainText("\n".join(lines))

    def _on_edited(self, param: str):
        row = self._params.index(param)
        value = self._combos[param].currentData()
        if value not in (self._UNSET, self._DROPPED):
            self._matches[param] = ag_flowjo.MATCH_MANUAL
            self.tbl.item(row, 3).setText(ag_flowjo.MATCH_MANUAL)
        self._colour_row(param)
        self._refresh()
        self._apply_filter()

    def _colour_row(self, param: str):
        row = self._params.index(param)
        value = self._combos[param].currentData()
        is_dark = self.palette().color(QPalette.ColorRole.Base).value() < 128
        bold = italic = False
        if value == self._UNSET:
            colour, bold = self._RED[is_dark], True
        elif value == self._DROPPED:
            colour, italic = self._GREY, True
        elif self._matches.get(param) in ag_flowjo.CONFIDENT_MATCHES:
            colour = None
        else:
            colour = self._AMBER[is_dark]
        for col in (0, 1, 3, 4):
            item = self.tbl.item(row, col)
            item.setData(Qt.ForegroundRole, None if colour is None else QBrush(colour))
            font = item.font()
            font.setBold(bold)
            font.setItalic(italic)
            item.setFont(font)

    def _drop_unmatched(self):
        for param, combo in self._combos.items():
            if combo.currentData() == self._UNSET:
                combo.setCurrentIndex(combo.findData(self._DROPPED))

    def mapping(self) -> dict:
        """{FlowJo parameter: experiment channel, or None when dropped or unresolved}."""
        return {p: (None if c.currentData() in (self._UNSET, self._DROPPED) else c.currentData())
                for p, c in self._combos.items()}

    def _unresolved(self) -> list:
        return [p for p, c in self._combos.items() if c.currentData() == self._UNSET]

    # -- summary and messages ----------------------------------------------

    def _build(self, replace: bool = True):
        source = self._source()
        return ag_flowjo.build_gate_definitions(
            source, self.mapping(), self.exp,
            merge_siblings=self.chk_merge.isChecked(),
            existing_names=set() if replace else set(self._existing_names()),
            workspace_name=self._name)

    def _refresh(self):
        source = self._source()
        if source is None or self.workspace is None:
            return
        unresolved = self._unresolved()
        problems = ag_flowjo.mapping_problems(self.mapping())
        plan = ag_flowjo.plan_import(source, self.mapping())
        lines = [f"{len(plan.keep)} of {source.n_populations} populations will be imported."]
        if unresolved:
            lines.append(f"<span style='color:#a00000'>{len(unresolved)} parameter(s) unmatched: "
                         "choose a channel or drop them to continue.</span>")
        lines += [f"<span style='color:#a00000'>{p}</span>" for p in problems]
        self.lbl_status.setText("<br>".join(lines))
        messages = list(plan.messages())
        if not unresolved and not problems and plan.keep:
            try:
                _gate_defs, report = self._build()
                for gate, notes in report['notes'].items():
                    messages += [f"{gate}: {note}" for note in notes]
            except Exception as exc:
                messages.append(f"Conversion failed: {exc}")
        self._notes = messages
        self.txt_notes.setPlainText("\n".join(messages))
        ready = bool(plan.keep) and not unresolved and not problems
        self.btn_add.setEnabled(ready)
        self.btn_replace.setEnabled(ready)
        self.btn_drop_unmatched.setEnabled(bool(unresolved))

    def _import(self, replace: bool):
        try:
            gate_defs, report = self._build(replace)
        except Exception as exc:
            self._fail(f"Import failed: {exc}")
            return
        if not gate_defs:
            self.lbl_status.setText("No populations could be converted to gates.")
            return
        self.importRequested.emit(gate_defs, report, replace, self._name)


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


def time_qc_available() -> bool:
    """True when the time-QC module is installed."""
    try:
        from honeychrome.controller_components import time_qc  # noqa: F401
    except ImportError:
        return False
    return True


class ModelImportPanel(QWidget):
    """Page of the Import/Setup tab for loading a saved model (``.agmodel``).

    Opened when a model file is chosen. It shows where and how the model was
    trained, and matches every channel its gates use to this experiment by
    antigen (green rows: matched by antigen; amber: check; red: choose a
    channel). The model changes only when Import is pressed, which emits
    ``importRequested(model, path, alignment, messages)`` with the
    alignment of the gated channels ({model channel: experiment channel})
    and the messages from reading the file. ``titleChanged(text)`` gives the
    heading for the page, and ``closeRequested`` asks the host to leave it.
    """

    importRequested = Signal(object, str, object, object)
    titleChanged = Signal(str)
    closeRequested = Signal()

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.model: dict = {}
        self.path = ''
        self.alignment: dict = {}
        self.methods: dict = {}
        self._rows: list[str] = []
        self._read_warnings: list[str] = []

        body = QVBoxLayout(self)
        body.setContentsMargins(6, 6, 6, 6)

        self.lbl_messages = QLabel("")
        self.lbl_messages.setWordWrap(True)
        self.lbl_messages.setStyleSheet("color: #856404;")
        body.addWidget(self.lbl_messages)
        self.lbl_summary = QLabel("")
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        body.addWidget(self.lbl_summary)

        note = QLabel("Each channel the model's gates use is matched to this experiment by "
                      "antigen. Green rows matched by antigen; amber rows need checking; red "
                      "rows need a channel.")
        note.setWordWrap(True)
        body.addWidget(note)
        self.filter_bar = FilterBar([('all', 'All channels'), ('check', 'Rows to check')],
                                    placeholder="Search channels or antigens…")
        self.filter_bar.changed.connect(lambda _t, _m: self._apply_filter())
        body.addWidget(self.filter_bar)
        self.tbl = QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(
            ["Model channel", "Model antigen", "Experiment channel", "Match"])
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl.setMinimumHeight(240)
        body.addWidget(self.tbl, stretch=1)

        row = QHBoxLayout()
        self.lbl_state = QLabel("")
        self.lbl_state.setWordWrap(True)
        row.addWidget(self.lbl_state, stretch=1)
        self.btn_import = QPushButton("Import model — replace")
        self.btn_import.setToolTip(
            "Replace the gates with the saved model's (renamed to the channels chosen "
            "here), keeping its trained boundaries.")
        self.btn_import.setEnabled(False)
        self.btn_import.clicked.connect(self._import)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.closeRequested.emit)
        row.addWidget(self.btn_import)
        row.addWidget(self.btn_close)
        body.addLayout(row)

    # -- opening and closing ------------------------------------------------

    def open_model(self, path: str) -> bool:
        """Read *path* and show its channel alignment for review."""
        from PySide6.QtWidgets import QApplication
        self.path = str(path)
        self.titleChanged.emit(f"Saved model import — {Path(path).name}")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            model, warnings_list = ag_core.read_model(path)
        except Exception as exc:
            self.model = {}
            self.tbl.setRowCount(0)
            self.lbl_summary.setText("")
            self.lbl_messages.setText(f"<span style='color:#a00000'>Could not read the "
                                      f"model: {exc}</span>")
            self.btn_import.setEnabled(False)
            return False
        finally:
            QApplication.restoreOverrideCursor()
        self.model = model
        self._read_warnings = list(warnings_list)
        self.alignment, self.methods = propose_alignment(self.controller, model)
        self._populate_summary()
        self._populate_table()
        self._show_messages()
        self._update_state()
        return True

    def collapse(self, summary: str):
        """Leave the page with a one-line summary in its heading after an import."""
        self.titleChanged.emit(f"Saved model import — {Path(self.path).name}: {summary}")
        self.closeRequested.emit()

    def _import(self):
        if self.model and not ag_results.alignment_problems(self.gated_alignment()):
            self.importRequested.emit(self.model, self.path, self.gated_alignment(),
                                      list(self._read_warnings))

    # -- summary and alignment ----------------------------------------------

    def gated_alignment(self) -> dict:
        return {mc: self.alignment.get(mc)
                for mc in ag_results.gated_channels(self.model.get('gate_definitions') or [])}

    def _populate_summary(self):
        m = self.model
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

    @staticmethod
    def _item(text: str, colour: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        item.setBackground(QColor(colour))
        item.setForeground(QColor('#000000'))
        return item

    def _populate_table(self):
        exp_pnn = _experiment_channels(self.controller)
        antigens = _antigen_map(self.controller)
        channel_map = self.model.get('channel_map') or {}
        self._rows = list(self.gated_alignment())
        self.tbl.setRowCount(0)
        for r, mc in enumerate(self._rows):
            self.tbl.insertRow(r)
            method = self.methods.get(mc)
            colour = _ALIGN_COLOURS.get(method, _ALIGN_COLOURS[None]) if method != 'manual' \
                else _ALIGN_COLOURS[ag_results.ALIGN_EXACT]
            self.tbl.setItem(r, 0, self._item(mc, colour))
            self.tbl.setItem(r, 1, self._item(channel_map.get(mc, ''), colour))
            combo = block_wheel(QComboBox())
            combo.addItem("— choose —", None)
            for ch in exp_pnn:
                if ch in ALWAYS_EXCLUDED_CHANNELS:
                    continue
                label = antigens.get(ch) or ch
                combo.addItem(ch if label == ch else f"{ch} — {label}", ch)
            current = self.alignment.get(mc)
            i = combo.findData(current) if current else 0
            combo.setCurrentIndex(max(i, 0))
            combo.currentIndexChanged.connect(
                lambda _i, row=r, name=mc: self._on_edited(row, name))
            self.tbl.setCellWidget(r, 2, combo)
            self.tbl.setItem(r, 3, self._item(_ALIGN_TEXT.get(method, ''), colour))
        self._apply_filter()

    def _needs_check(self, mc: str) -> bool:
        return (not self.alignment.get(mc)
                or self.methods.get(mc) not in
                (ag_results.ALIGN_EXACT, ag_results.ALIGN_ANTIGEN, 'manual'))

    def _apply_filter(self):
        """Hide the rows outside the search and the "show" choice."""
        text, mode = self.filter_bar.text(), self.filter_bar.mode()
        channel_map = self.model.get('channel_map') or {}
        n_check = 0
        for r, mc in enumerate(self._rows):
            check = self._needs_check(mc)
            n_check += int(check)
            shown = text_matches(text, mc, channel_map.get(mc, ''), self.alignment.get(mc) or '')
            if mode == 'check':
                shown = shown and check
            self.tbl.setRowHidden(r, not shown)
        self.filter_bar.set_mode_label('check', f"Rows to check ({n_check})")

    def _on_edited(self, row: int, model_ch: str):
        combo = self.tbl.cellWidget(row, 2)
        if combo is None:
            return
        self.alignment[model_ch] = combo.currentData()
        self.methods[model_ch] = 'manual' if combo.currentData() else None
        colour = _ALIGN_COLOURS[ag_results.ALIGN_EXACT] if combo.currentData() else _ALIGN_COLOURS[None]
        for col in (0, 1, 3):
            item = self.tbl.item(row, col)
            if item is not None:
                item.setBackground(QColor(colour))
        self.tbl.item(row, 3).setText(_ALIGN_TEXT['manual' if combo.currentData() else None])
        self._update_state()
        self._apply_filter()

    def _update_state(self):
        problems = ag_results.alignment_problems(self.gated_alignment())
        needs_check = [mc for mc in self._rows if self._needs_check(mc) and self.alignment.get(mc)]
        if problems:
            text = "<b>Before importing:</b> " + "; ".join(problems)
        elif needs_check:
            text = "Check the amber rows, then import: " + ", ".join(needs_check)
        else:
            text = "All gated channels matched by antigen."
        self.lbl_state.setText(text)
        self.btn_import.setEnabled(bool(self.model) and not problems)

    def _show_messages(self):
        msgs = list(self._read_warnings)
        qc = self.model.get('qc') or {}
        if qc.get('enabled') and not time_qc_available():
            msgs.append("The model was trained on time-QC-filtered events, but time QC is "
                        "not available in this installation; samples are gated unfiltered.")
        self.lbl_messages.setText("<br>".join(msgs))


def propose_alignment(controller, model: dict) -> tuple[dict, dict]:
    """({model channel: experiment channel or None}, {model channel: match
    method}) for the channels a saved model's gates use, matched to this
    experiment by antigen (synonyms recognised when the marker database
    loads)."""
    from honeychrome.controller_components.label_matching import get_marker_db, match_marker
    try:
        marker_db = get_marker_db()
        canonical = lambda name: match_marker(name, marker_db)  # noqa: E731
    except Exception:
        canonical = None
    pnn = _experiment_channels(controller)
    antigens = _antigen_map(controller)
    proposal = ag_results.align_channels(
        ag_results.gated_channels(model.get('gate_definitions') or []),
        model.get('channel_map') or {}, pnn, [antigens.get(ch, '') for ch in pnn],
        canonical=canonical,
    )
    return ({mc: p['channel'] for mc, p in proposal.items()},
            {mc: p['method'] for mc, p in proposal.items()})


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
        - "Unconverted" — red   — the gate applies its shapes as drawn
        - "Trained"   — green   — a trained boundary exists
        - "Untrained" — amber   — converted, not trained yet
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
    COL_CONVERT    = 9

    _COLUMN_LABELS = [
        "#", "Gate name", "Type", "X axis", "Y axis",
        "Parent gate", "Parent population", "Algorithm", "Status", "Conversion",
    ]

    # Emitted after gates are imported, so the plugin can open the Hierarchy tab.
    gatesImported = Signal()
    # Emitted after a 2dsep gate was split into 1dsep gates.
    gatesSplit = Signal()
    # Emitted when an axis drag changed a channel's transform: (channel, parameters before).
    transformChanged = Signal(str, object)
    # Emitted after a saved model replaced the gates: {'application', 'transforms_applied', 'messages'}.
    modelImported = Signal(object)
    # Emitted after a gate was renamed: (old name, new name).
    gateRenamed = Signal(str, str)

    def __init__(self, state: GatingModelState, bus, controller, tf=None, parent=None):
        super().__init__(parent)
        self.state = state
        self.bus = bus
        self.controller = controller
        self.tf = tf

        self._recommend_worker: _RecommendWorker | None = None

        # Whether the tree's itemChanged signal should be acted on.
        # We block it during programmatic repopulation so editing
        # state doesn't trigger a feedback loop.
        self._suppress_item_changed = False

        # Parent-population events of the previewed gate: (gate name, events, note).
        # Kept while an axis is dragged so the data is not gated again for every step.
        self._preview_cache = None
        self._preview_channels = ('', '')

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Horizontal splitter: concertina on left, preview on right.
        splitter = QSplitter(Qt.Horizontal)

        # ----- Left side -----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(HelpToggleWidget(text=ag_help_texts.BUILDER_HIERARCHY))

        # One section open at a time; each page scrolls on its own.
        self.toolbox = make_concertina()
        left_layout.addWidget(self.toolbox, stretch=1)

        # Gate hierarchy page: control bar above the gate list.
        gates_page = QWidget()
        page_layout = QVBoxLayout(gates_page)
        page_layout.setContentsMargins(0, 0, 0, 0)

        # Default plot axis of the separation parameter of 1dsep gates.
        axis_row = QHBoxLayout()
        axis_label = QLabel("1dsep separation axis:")
        self.rb_sep_x = QRadioButton("X axis")
        self.rb_sep_y = QRadioButton("Y axis")
        self._sep_group = QButtonGroup(self)
        self._sep_group.addButton(self.rb_sep_x)
        self._sep_group.addButton(self.rb_sep_y)
        (self.rb_sep_y if self.state.default_sep_axis == 'y' else self.rb_sep_x).setChecked(True)
        tip = ("Plot axis that carries the separation parameter of 1dsep gates you "
               "add or split; the other axis is the view parameter (or a "
               "histogram). Imported gates take it from the shape as drawn, and "
               "each gate can be changed in its Edit dialog.")
        for w in (axis_label, self.rb_sep_x, self.rb_sep_y):
            w.setToolTip(tip)
            axis_row.addWidget(w)
        axis_row.addStretch()
        page_layout.addLayout(axis_row)
        self.rb_sep_y.toggled.connect(self._on_default_sep_axis_toggled)

        # Control bar: import buttons, then gate editing.
        import_row = QHBoxLayout()
        ctrl_row = QHBoxLayout()
        self.btn_import   = QPushButton("Import from Hierarchy")
        self.btn_flowjo   = QPushButton("Import FlowJo…")
        self.btn_load_model = QPushButton("Load saved model…")
        self.btn_load_model.setToolTip(
            "Load a .agmodel file, match its channels to this experiment and use it as "
            "the model, with its trained boundaries. Replaces any existing gates.")
        self.btn_convert_all = QPushButton("Convert all to trainable")
        self.btn_convert_all.setToolTip(
            "Switch every gate that uses its imported boundary to its proposed learning "
            "algorithm, so Train calculates it from the training samples. Gates sharing "
            "a name with a gate you converted yourself follow your choice."
        )
        self.btn_refresh  = QPushButton("Refresh from Hierarchy")
        self.btn_add      = QPushButton("Add gate manually")
        self.btn_advanced = QPushButton("Advanced…")
        self.btn_delete   = QPushButton("Delete")

        self.btn_import.setToolTip(
            "Copy the gates drawn on the main plots into the model, as drawn "
            "(algorithm 'imported'). Replaces any existing gates in the model."
        )
        self.btn_refresh.setToolTip(
            "Merge new gates from controller.unmixed_gating into the model. "
            "Existing gate_param overrides are preserved. Gates removed "
            "from the source hierarchy are marked Stale."
        )

        self.btn_flowjo.setToolTip(
            "Import the gating strategy of a FlowJo workspace (.wsp) or "
            "template (.wspt), matching its parameters to this experiment's "
            "channels. Gates keep their FlowJo boundaries (algorithm 'imported')."
        )
        self.btn_import.clicked.connect(self._on_import_clicked)
        self.btn_flowjo.clicked.connect(self._on_import_flowjo_clicked)
        self.btn_load_model.clicked.connect(self._on_load_model_clicked)
        self.btn_convert_all.clicked.connect(self._convert_all_gates)
        self.btn_refresh.clicked.connect(self._on_refresh_clicked)
        self.btn_add.clicked.connect(self._on_add_clicked)
        self.btn_advanced.clicked.connect(self._on_advanced_clicked)
        self.btn_delete.clicked.connect(self._on_delete_clicked)

        for b in (self.btn_import, self.btn_flowjo, self.btn_load_model, self.btn_refresh):
            import_row.addWidget(b)
        import_row.addStretch()
        page_layout.addLayout(import_row)
        for b in (self.btn_convert_all, self.btn_add, self.btn_advanced, self.btn_delete):
            ctrl_row.addWidget(b)
        ctrl_row.addStretch()
        page_layout.addLayout(ctrl_row)

        self.flowjo_panel = FlowJoImportPanel(
            self.controller,
            lambda: {g.get('gate_name') for g in self.state.gate_definitions},
            transforms=lambda: self.tf.objects,
            parent=self)
        self.flowjo_panel.importRequested.connect(self._on_flowjo_import_requested)
        self.model_panel = ModelImportPanel(self.controller, parent=self)
        self.model_panel.importRequested.connect(self._on_model_import_requested)

        # Search and filter.
        self.filter_bar = FilterBar(GATE_FILTERS, placeholder="Search gates, markers or algorithms…")
        self.filter_bar.setToolTip(
            "Unconverted: gates still applying their imported boundary. Conflicting: gates "
            "sharing a name whose algorithms were set differently by hand. Flagged: gates whose "
            "trainable form lands far from the imported boundary (Hierarchy → Flag conversions).")
        self.filter_bar.changed.connect(lambda _t, _m: self._apply_filter())
        page_layout.addWidget(self.filter_bar)

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
        self.tree.setColumnWidth(self.COL_CONVERT,    130)
        self.tree.setMinimumHeight(240)
        page_layout.addWidget(self.tree)

        # Help / status text just under the tree.
        self.lbl_summary = QLabel("No gates loaded.")
        self.lbl_summary.setStyleSheet("color: #555; font-style: italic;")
        page_layout.addWidget(self.lbl_summary)

        # A loaded model's own transforms.
        self.lbl_model_transforms = QLabel("")
        self.lbl_model_transforms.setWordWrap(True)
        self.lbl_model_transforms.setStyleSheet(
            "QLabel { color: #0c5460; background: #d1ecf1; "
            "border: 1px solid #bee5eb; border-radius: 4px; padding: 6px; }")
        self.lbl_model_transforms.setVisible(False)
        page_layout.addWidget(self.lbl_model_transforms)

        self._page_gates = add_concertina_page(self.toolbox, gates_page, "Gate hierarchy")
        self._page_flowjo = add_concertina_page(self.toolbox, self.flowjo_panel, "FlowJo import")
        self.toolbox.setItemEnabled(self._page_flowjo, False)
        self.flowjo_panel.titleChanged.connect(
            lambda text: self.toolbox.setItemText(self._page_flowjo, text))
        self.flowjo_panel.closeRequested.connect(
            lambda: self.toolbox.setCurrentIndex(self._page_gates))
        self._page_model = add_concertina_page(self.toolbox, self.model_panel, "Saved model import")
        self.toolbox.setItemEnabled(self._page_model, False)
        self.model_panel.titleChanged.connect(
            lambda text: self.toolbox.setItemText(self._page_model, text))
        self.model_panel.closeRequested.connect(
            lambda: self.toolbox.setCurrentIndex(self._page_gates))
        self.toolbox.currentChanged.connect(self._on_page_changed)

        splitter.addWidget(left)

        # ----- Right side: preview plot -----
        right = QWidget()
        self._preview_panel = right
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)

        self.lbl_preview_title = QLabel("<i>Select a gate to preview its geometry.</i>")
        self.lbl_preview_title.setWordWrap(True)
        right_layout.addWidget(self.lbl_preview_title)

        # Proper Honeychrome plot widget: TransparentGraphicsLayoutWidget +
        # NoPanViewBox + ZoomAxis — matches CytometryPlotWidget and BiplotTile.
        self._preview_gw = TransparentGraphicsLayoutWidget()
        self._preview_gw.setMinimumSize(280, 280)
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
        self._preview_axis_x.zoom_timer.timeout.connect(lambda: self._on_preview_zoom('x'))
        self._preview_axis_y.zoom_timer.timeout.connect(lambda: self._on_preview_zoom('y'))
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

        right_layout.addWidget(SquareHolder(self._preview_gw, minimum=280), stretch=1)

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

    def refresh_preview(self):
        """Redraw the preview for the selected gate."""
        self._update_preview()

    def _on_page_changed(self, index: int):
        """The gate preview follows the gate list, so it is hidden while
        an import page is open."""
        self._preview_panel.setVisible(index not in (self._page_flowjo, self._page_model))

    def _on_default_sep_axis_toggled(self, _checked: bool = False):
        self.state.default_sep_axis = 'y' if self.rb_sep_y.isChecked() else 'x'

    def sync_default_sep_axis(self):
        """Show the state's default separation axis (after loading state)."""
        self.rb_sep_x.blockSignals(True)
        self.rb_sep_y.blockSignals(True)
        try:
            (self.rb_sep_y if self.state.default_sep_axis == 'y' else self.rb_sep_x).setChecked(True)
        finally:
            self.rb_sep_x.blockSignals(False)
            self.rb_sep_y.blockSignals(False)

    def refresh(self):
        """Re-evaluate channel alignment and repopulate the tree.

        Called by BuilderPages whenever the Import/Setup tab becomes
        active.  This is a cheap operation; it does not load any sample
        data.
        """
        self._repopulate_tree()
        self.update_model_transforms_note()

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
                actions = self._conversion_buttons(g)
                if actions is not None:
                    self.tree.setItemWidget(item, self.COL_CONVERT, actions)
        finally:
            self._suppress_item_changed = False

        self.btn_convert_all.setEnabled(any(
            g.get('algorithm') == ag_core.ALGORITHM_IMPORTED for g in self.state.gate_definitions))
        self._apply_filter()
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

    def _apply_filter(self):
        """Hide the rows outside the search and the "show" choice."""
        text, mode = self.filter_bar.text(), self.filter_bar.mode()
        wanted = filtered_gate_names(self.state, mode)
        counts = filter_counts(self.state)
        for key, label in GATE_FILTERS[1:]:
            self.filter_bar.set_mode_label(key, f"{label} ({counts.get(key, 0)})")
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            name = item.data(0, Qt.UserRole)
            shown = (wanted is None or name in wanted) and text_matches(
                text, *(item.text(c) for c in range(self.tree.columnCount())))
            item.setHidden(not shown)

    def _conversion_buttons(self, g: dict):
        """'Convert to trainable' for a gate using its imported shapes and
        'Split…' for a 2dsep gate, or None when neither applies."""
        name = g.get('gate_name', '')
        buttons = []
        if g.get('algorithm') == ag_core.ALGORITHM_IMPORTED:
            target, _note = ag_review.convert_target(g, self.state.algorithm_recommendations.get(name))
            button = QPushButton("Convert to trainable")
            button.setToolTip(
                f"Train this gate with '{target}' instead of applying the shapes "
                "it was imported with. Compare the two on the Hierarchy tab first.")
            button.clicked.connect(lambda _checked=False, n=name: self._convert_gate(n))
            buttons.append(button)
        if g.get('gate_type') == '2dsep':
            button = QPushButton("Split…")
            button.setToolTip("Replace this 2dsep gate with sequential 1dsep gates.")
            button.clicked.connect(lambda _checked=False, n=name: self._split_2dsep_gate(n))
            buttons.append(button)
        if not buttons:
            return None
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        for button in buttons:
            row.addWidget(button)
        return holder

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
        if algo_display == ag_core.ALGORITHM_IMPORTED:
            target, _note = ag_review.convert_target(g, rec)
            if target:
                algo_display = f"imported → {target}"
                rec = ''
        if rec and rec != algo_display:
            algo_display = f"{algo_display} [rec: {rec}]" if algo_display else f"[rec: {rec}]"

        x_text, y_text = _axis_cells(g, _antigen_map(self.controller))
        item = QTreeWidgetItem([
            str(g.get('gate_number', '')),
            str(gate_name_for_item),
            str(g.get('gate_type', '')),
            x_text,
            y_text,
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
        flag = self.state.conversion_flags.get(gate_name_for_item)
        overlap = self.state.overlap_flags.get(gate_name_for_item)
        tips = [f.get('message', '') for f in (flag, overlap) if f]
        if tips:
            item.setToolTip(self.COL_STATUS, "\n".join(tips))
        item.setForeground(self.COL_STATUS, status_brush)
        if row_tint is not None:
            for col in range(self.tree.columnCount()):
                item.setBackground(col, row_tint)
                if col != self.COL_STATUS:
                    item.setForeground(col, QBrush(QColor('#000000')))

        # Stash the gate_name on the item so context-menu / double-click
        # handlers can find the matching dict without a string compare
        # against a mutable column.
        item.setData(0, Qt.UserRole, g.get('gate_name', ''))
        return item

    def _status_for(self, g: dict, is_stale: bool):
        """Return (text, foreground brush, optional background tint) for status column."""
        if is_stale:
            return ("Stale", QBrush(QColor('#a00000')), QColor(255, 235, 235))
        flag = self.state.conversion_flags.get(g.get('gate_name', ''))
        if flag and g.get('algorithm') == ag_core.ALGORITHM_IMPORTED:
            return ("⚠ Flagged", QBrush(QColor('#c0392b')), None)
        if self.state.overlap_flags.get(g.get('gate_name', '')):
            return ("⚠ Overlap", QBrush(QColor('#d35400')), None)
        status = gate_status(self.state, g)
        return (status, QBrush(QColor(*STATUS_COLOURS[status])), None)

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
        if column == self.COL_X:           return _axis_cells(g, _antigen_map(self.controller))[0]
        if column == self.COL_Y:           return _axis_cells(g, _antigen_map(self.controller))[1]
        if column == self.COL_PARENT:      return str(g.get('parent_gate') or 'root')
        if column == self.COL_PARENT_POP:  return str(g.get('parent_popul', '') or '')
        if column == self.COL_ALGORITHM:   return str(g.get('algorithm', '') or '')
        return ""

    def _rename_gate_references(self, old: str, new: str):
        """Update every reference to a renamed gate, its trained boundaries
        and its recommendation."""
        ag_review.rename_references(self.state.gate_definitions, old, new)
        if old in self.state.trained_boundaries:
            self.state.trained_boundaries[new] = self.state.trained_boundaries.pop(old)
        if old in self.state.algorithm_recommendations:
            self.state.algorithm_recommendations[new] = (
                self.state.algorithm_recommendations.pop(old)
            )
        self.gateRenamed.emit(old, new)

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
        act_convert = QAction("Convert to trainable", menu)
        act_keep = QAction("Keep imported boundary", menu)

        for a in (act_edit, act_dup, act_del, act_child, act_param, act_split,
                  act_convert, act_keep):
            a.setEnabled(item is not None)

        if item is not None:
            gate_name = item.data(0, Qt.UserRole)
            g = self.state.gate_by_name(gate_name)
            act_split.setEnabled(g is not None and g.get('gate_type') == '2dsep')
            is_imported = g is not None and g.get('algorithm') == ag_core.ALGORITHM_IMPORTED
            act_convert.setEnabled(is_imported)
            act_keep.setEnabled(g is not None and not is_imported and bool(g.get('template'))
                                and g.get('gate_type') in ag_core.ALGORITHMS_BY_GATE_TYPE
                                and ag_core.ALGORITHM_IMPORTED in ag_core.ALGORITHMS_BY_GATE_TYPE[g['gate_type']])
            act_convert.triggered.connect(lambda: self._convert_gate(gate_name))
            act_keep.triggered.connect(lambda: self._keep_imported(gate_name))

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
        menu.addSeparator()
        menu.addAction(act_convert)
        menu.addAction(act_keep)

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
                "[Automated Gating] No gates found in controller.unmixed_gating."
            )
        else:
            self.bus.statusMessage.emit(
                f"[Automated Gating] Imported {n} gate{'s' if n != 1 else ''} "
                "from hierarchy."
            )
        self._recommend_algorithms()

    def _on_import_flowjo_clicked(self):
        """Choose a FlowJo workspace and expand the import section."""
        start_dir = ''
        try:
            start_dir = str(self.controller.experiment_dir)
        except Exception:
            pass
        path, _ = QFileDialog.getOpenFileName(
            self, "Import FlowJo workspace", start_dir,
            "FlowJo workspaces (*.wsp *.wspt);;All files (*)",
        )
        if path:
            self.toolbox.setItemEnabled(self._page_flowjo, True)
            self.flowjo_panel.open_workspace(path)
            self.toolbox.setCurrentIndex(self._page_flowjo)

    def _on_flowjo_import_requested(self, gate_defs, report, replace: bool, name: str):
        """Add or replace the model's gates with the imported ones."""
        if replace:
            self.state.gate_definitions = []
            self.state.trained_boundaries = {}
            self.state.algorithm_recommendations = {}
            self.state.model_transforms = {}
            self.update_model_transforms_note()
        offset = max((int(g.get('gate_number', 0)) for g in self.state.gate_definitions), default=0)
        for i, gate_def in enumerate(gate_defs, start=1):
            gate_def['gate_number'] = offset + i
            self.state.gate_definitions.append(gate_def)
        self._repopulate_tree()
        n_dropped = len(report['plan'].dropped)
        tail = f", {n_dropped} population(s) not imported" if n_dropped else ''
        self.flowjo_panel.collapse(f"{len(gate_defs)} gate(s) {'replaced' if replace else 'added'}{tail}")
        self.bus.statusMessage.emit(
            f"[Automated Gating] Imported {len(gate_defs)} gate(s) from {name}{tail}. "
            "Review them on the Hierarchy tab before training."
        )
        self.gatesImported.emit()

    def _on_load_model_clicked(self):
        """Choose a saved model and open its import section."""
        start_dir = ''
        try:
            start_dir = str(self.controller.experiment_dir)
        except Exception:
            pass
        path, _ = QFileDialog.getOpenFileName(self, "Load saved model", start_dir,
                                              "Gating model (*.agmodel)")
        if path:
            self.toolbox.setItemEnabled(self._page_model, True)
            self.model_panel.open_model(path)
            self.toolbox.setCurrentIndex(self._page_model)

    def _on_model_import_requested(self, model, path: str, alignment, messages):
        """Replace the model's gates with the saved model's, after confirming."""
        if self.state.gate_definitions:
            ans = QMessageBox.question(
                self, "Load saved model",
                f"This will replace the current {self.state.n_gates()} gate(s), and their "
                "training, with the saved model's.\n\nContinue?")
            if ans != QMessageBox.Yes:
                return
        if self.import_saved_model(model, path, alignment, messages):
            self.model_panel.collapse(f"{self.state.n_gates()} gate(s) imported")

    def import_saved_model(self, model: dict, path: str, alignment: dict, messages=()) -> bool:
        """Replace the gates with a saved model's, renamed to this experiment's
        channels by *alignment* and converted to the builder's transforms.

        The model's own transforms (on this experiment's channels) are kept
        in ``state.model_transforms`` for the Transforms page. They become
        the builder's transforms straight away when the builder's are still
        untouched (no Auto-transform, axis drag or CSV load since the last
        reset). Emits modelImported.
        """
        st = self.state
        live = self.controller.unmixed_transformations or {}
        try:
            aligned = ag_results.apply_alignment(model, alignment)
            model_tf = {ch: dict(p) for ch, p in (aligned.get('transforms') or {}).items()
                        if p and ch in self.tf.objects}
            applied = bool(model_tf) and not st.transforms_custom
            target = self.tf.params()
            if applied:
                target.update(model_tf)
            converted, notes = ag_core.remap_model(aligned, target)
        except Exception as exc:
            log.exception("could not import the model %s", path)
            QMessageBox.critical(self, "Load saved model", f"Could not use the model:\n{exc}")
            return False
        if applied:
            self.tf.load(target, live)
        apply_model_to_state(converted, st)
        try:
            available = set(analysis_sample_keys(self.controller, True))
        except Exception:
            available = set()
        st.training_samples = [k for k in st.training_samples if k in available]
        st.algorithm_recommendations = {}
        st.conversion_flags = {}
        st.overlap_flags = {}
        st.training_progress = {}
        st.model_transforms = model_tf
        st.model_metadata['last_model_path'] = str(path)
        self._repopulate_tree()
        self.update_model_transforms_note()
        self.bus.statusMessage.emit(
            f"[Automated Gating] Loaded {Path(path).name}: {st.n_gates()} gate(s), "
            f"{sum(1 for g in st.gate_definitions if st.is_trained(g['gate_name']))} trained.")
        self.modelImported.emit({'application': dict(converted.get('application') or {}),
                                 'transforms_applied': applied,
                                 'messages': list(messages) + list(notes)})
        return True

    def update_model_transforms_note(self):
        """Show whether a loaded model's own transforms are in use or available."""
        model_tf = self.state.model_transforms
        if not model_tf or self.tf is None:
            self.lbl_model_transforms.setVisible(False)
            return
        current = self.tf.params()
        in_use = all(ag_core.same_transform(current.get(ch), p) for ch, p in model_tf.items())
        if in_use:
            text = ("The loaded model's own transforms are in use (Transforms page).")
        else:
            text = ("The loaded model's own transforms are available: <b>Apply the model's "
                    "transforms</b> on the Transforms page. Its boundaries were converted to "
                    "the current transforms.")
        self.lbl_model_transforms.setText(text)
        self.lbl_model_transforms.setVisible(True)

    def _descendant_names(self, gate_name: str) -> set:
        out, frontier = set(), {gate_name}
        while frontier:
            frontier = {g['gate_name'] for g in self.state.gate_definitions
                        if g.get('parent_gate') in frontier} - out
            out |= frontier
        return out

    def _set_algorithm(self, gate_def: dict, algorithm: str):
        """Change a gate's algorithm; its training and that of the gates
        below it no longer applies."""
        gate_def['algorithm'] = algorithm
        for name in {gate_def['gate_name']} | self._descendant_names(gate_def['gate_name']):
            self.state.trained_boundaries.pop(name, None)

    def _sync(self, gate_name: str) -> list[str]:
        """Bring the gates sharing *gate_name*'s imported name in line (see
        ag_review.apply_name_sync); returns the names changed."""
        changed, conflict = ag_review.apply_name_sync(self.state.gate_definitions, gate_name)
        for name in changed:
            for n in {name} | self._descendant_names(name):
                self.state.trained_boundaries.pop(n, None)
        if conflict:
            self.bus.statusMessage.emit(
                "[Automated Gating] Conflicting algorithms chosen — identically named gates "
                f"will not be auto-updated ('{conflict}').")
        return changed

    def _convert_gate(self, gate_name: str):
        """Switch one imported gate to a learning algorithm; same-named gates
        still as drawn follow it."""
        gate_def = self.state.gate_by_name(gate_name)
        if gate_def is None:
            return
        algorithm, note = ag_review.convert_target(
            gate_def, self.state.algorithm_recommendations.get(gate_name))
        if algorithm is None:
            return
        if gate_def.get('gate_type') == '2dsep':
            gate_def.setdefault('gate_param', {}).setdefault('algorithm_x', algorithm)
        self._set_algorithm(gate_def, algorithm)
        gate_def['sync_origin'] = ag_review.SYNC_USER
        changed = self._sync(gate_name)
        self._repopulate_tree()
        tail = f" {len(changed)} gate(s) with the same name follow it." if changed else ''
        self.bus.statusMessage.emit(
            f"[Automated Gating] '{gate_name}' now uses '{algorithm}'; train to calculate "
            f"it.{tail} {note}".strip())

    def _convert_all_gates(self):
        """Switch every imported gate to its proposed learning algorithm.
        Same-named gates you converted yourself bring theirs in line."""
        converted = 0
        for gate_def in self.state.gate_definitions:
            if gate_def.get('algorithm') != ag_core.ALGORITHM_IMPORTED:
                continue
            algorithm, _note = ag_review.convert_target(
                gate_def, self.state.algorithm_recommendations.get(gate_def['gate_name']))
            if algorithm is not None:
                if gate_def.get('gate_type') == '2dsep':
                    gate_def.setdefault('gate_param', {}).setdefault('algorithm_x', algorithm)
                self._set_algorithm(gate_def, algorithm)
                gate_def['sync_origin'] = ag_review.SYNC_AUTO
                converted += 1
        followed = 0
        for gate_def in self.state.gate_definitions:
            if gate_def.get('sync_origin') not in (ag_review.SYNC_AUTO, ag_review.SYNC_DRAWN) \
                    and gate_def.get('algorithm') != ag_core.ALGORITHM_IMPORTED:
                followed += len(self._sync(gate_def['gate_name']))
        self._repopulate_tree()
        tail = f" {followed} follow a same-named gate you set." if followed else ''
        self.bus.statusMessage.emit(
            f"[Automated Gating] Converted {converted} gate(s) to learning algorithms; "
            f"train to calculate them.{tail}")

    def _keep_imported(self, gate_name: str):
        """Return a converted gate to its imported boundary."""
        gate_def = self.state.gate_by_name(gate_name)
        if gate_def is None or not gate_def.get('template'):
            return
        self._set_algorithm(gate_def, ag_core.ALGORITHM_IMPORTED)
        gate_def['sync_origin'] = ag_review.SYNC_DRAWN
        self._repopulate_tree()

    def _on_refresh_clicked(self):
        added, stale = self._merge_from_unmixed_gating()
        self._repopulate_tree()
        msg = (
            f"[Automated Gating] Refresh: added {added}, "
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

    def _edit_gate(self, gate_name: str) -> str | None:
        """Edit a gate in GateEditDialog; returns its name when accepted."""
        g = self.state.gate_by_name(gate_name)
        if g is None:
            return None
        old_name = g.get('gate_name', '')
        old_layout = (g.get('gate_type'), g.get('gate_marker_x'), g.get('gate_marker_y'),
                      {p: m.get('region') for p, m in (g.get('populations') or {}).items()})
        dlg = GateEditDialog(g, self.state, self.controller,
                             is_new=False, parent=self)
        if dlg.exec() == QDialog.Accepted:
            new_layout = (g.get('gate_type'), g.get('gate_marker_x'), g.get('gate_marker_y'),
                          {p: m.get('region') for p, m in (g.get('populations') or {}).items()})
            if new_layout != old_layout:
                self.state.trained_boundaries.pop(old_name, None)
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
            return g['gate_name']
        return None

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
        defs, _removed, orphaned = ag_review.delete_gates(self.state.gate_definitions, [gate_name])
        self.state.gate_definitions = defs
        self.state.trained_boundaries.pop(gate_name, None)
        self.state.algorithm_recommendations.pop(gate_name, None)
        if orphaned:
            self.bus.statusMessage.emit(
                "[Automated Gating] These gates replicated the deleted gate and need a new "
                "source: " + ", ".join(f"'{n}'" for n in orphaned) + ".")
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
            'sep_axis':      self.state.default_sep_axis,
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
        """Replace a 2dsep gate with sequential 1dsep gates chosen in
        SplitGateDialog (see ag_review.split_2dsep). The gate's training,
        and that of the gates below it, no longer applies."""
        g = self.state.gate_by_name(gate_name)
        if g is None or g.get('gate_type') != '2dsep':
            return
        if g.get('algorithm') == ag_core.ALGORITHM_IMPORTED:
            default_algo, _note = ag_review.convert_target(
                g, self.state.algorithm_recommendations.get(gate_name))
        else:
            default_algo = g.get('algorithm') or 'tail'
        dlg = SplitGateDialog(g, self.controller, default_algo or 'tail', parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        stale = {gate_name} | self._descendant_names(gate_name)
        try:
            new_defs, new_names = ag_review.split_2dsep(
                self.state.gate_definitions, gate_name, dlg.steps())
        except ValueError as exc:
            QMessageBox.warning(self, "Split gate", str(exc))
            return
        self.state.gate_definitions = new_defs
        for name in stale:
            self.state.trained_boundaries.pop(name, None)
        self.state.algorithm_recommendations.pop(gate_name, None)
        self._repopulate_tree()
        self._select_gate_by_name(new_names[0])
        self.gatesSplit.emit()
        self.bus.statusMessage.emit(
            f"[Automated Gating] Replaced '{gate_name}' with "
            + ", ".join(f"'{n}'" for n in new_names) + ". Train to calculate them.")

    # ------------------------------------------------------------------
    # Importing the live GatingStrategy
    # ------------------------------------------------------------------

    def _import_from_unmixed_gating(self, replace: bool = True) -> int:
        """Copy ``controller.unmixed_gating`` into ``state.gate_definitions``.

        The hierarchy is converted the same way as a FlowJo workspace (see
        ag_flowjo): each drawn gate keeps its shape as an ``imported``
        boundary and its population names. Rectangles on fluorescence
        channels become threshold gates, quadrants become one threshold gate
        with a population per quadrant, other shapes stay as drawn, and a
        gate's parent is the population it was drawn on.

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
            source = ag_flowjo.source_from_gating(gating)
            exp = _flowjo_experiment_channels(self.controller)
            mapping = {p: (p if p in exp.channels else None)
                       for p in ag_flowjo.gated_parameters(source)}
            gate_defs, report = ag_flowjo.build_gate_definitions(
                source, mapping, exp, native_display=True, origin='Experiment hierarchy')
            # The hierarchy's gates are drawn in the experiment's transforms.
            live = _live_transform_params(self.controller)
            own = self.tf.params()
            for ch, params in own.items():
                if ch in live and not ag_core.same_transform(live[ch], params):
                    ag_core.remap_channel_transform(gate_defs, {}, ch, live[ch], params,
                                                    points_per_edge=16)
        except Exception as exc:
            self.bus.statusMessage.emit(
                f"[Automated Gating] Could not read unmixed_gating: {exc}"
            )
            return 0

        if replace:
            self.state.gate_definitions = []
            # Trained boundaries belong to the previous set of gates.
            self.state.trained_boundaries = {}
            self.state.algorithm_recommendations = {}
            self.state.model_transforms = {}
            self.update_model_transforms_note()

        existing = {g.get('gate_name') for g in self.state.gate_definitions}
        next_number = max(
            (int(g.get('gate_number', 0)) for g in self.state.gate_definitions),
            default=0,
        )
        imported = 0
        for gate_def in gate_defs:
            if gate_def['gate_name'] in existing:
                continue
            next_number += 1
            gate_def['gate_number'] = next_number
            self.state.gate_definitions.append(gate_def)
            imported += 1

        dropped = report['plan'].messages()
        if dropped:
            self.bus.statusMessage.emit(
                f"[Automated Gating] {len(dropped)} branch(es) of the hierarchy not "
                f"imported. First: {dropped[0]}"
            )
        return imported

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
        if wait_if_busy(self.controller, ('Automated Gating recommendation', id(self)),
                        self._recommend_algorithms):
            return
        keys = list(self.state.training_samples) or analysis_sample_keys(self.controller)
        if not keys or not self.state.gate_definitions:
            self.bus.statusMessage.emit(
                "[Automated Gating] Cannot recommend algorithms: no samples "
                "or gates available."
            )
            return
        key = keys[0]
        try:
            snap = snapshot_unmix_state(self.controller, [key])
        except Exception as exc:
            self.bus.statusMessage.emit(f"[Automated Gating] {exc}")
            return
        worker = _RecommendWorker(
            self.controller.experiment_dir, key, snap,
            deepcopy(self.state.gate_definitions),
            _experiment_channels(self.controller),
            deepcopy(self.tf.objects),
            parent=self,
        )
        worker.finished.connect(self._on_recommendations_ready)
        self._recommend_worker = worker
        self.bus.statusMessage.emit(
            f"[Automated Gating] Checking bimodality on {_sample_label(key)} …"
        )
        worker.start()

    def _on_recommendations_ready(self, ok: bool, message: str, recommendations: object):
        self._recommend_worker = None
        if not ok:
            self.bus.statusMessage.emit(
                f"[Automated Gating] Algorithm recommendation failed: {message}"
            )
            return
        self.state.algorithm_recommendations.update(recommendations or {})
        self._repopulate_tree()
        self.bus.statusMessage.emit(
            f"[Automated Gating] Algorithm recommendations computed for "
            f"{len(recommendations or {})} gate(s)."
        )

    # ------------------------------------------------------------------
    # Right-hand preview pane
    # ------------------------------------------------------------------

    def _update_preview(self, keep_events: bool = False):
        """Refresh the right-pane preview plot for the selected gate.

        With *keep_events* the parent-population events found by the last
        full refresh are drawn again (used while an axis is being dragged).
        """
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
            self._preview_cache = None
            self._preview_channels = ('', '')
            self.lbl_preview_title.setText(
                "<i>Select a gate to preview its geometry.</i>"
            )
            return

        ch_x, ch_y = _preview_axes(g)
        self._preview_channels = (ch_x, ch_y)

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

        transforms = self.tf.objects
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
        cache = self._preview_cache
        if keep_events and cache is not None and cache[0] == g.get('gate_name'):
            _name, raw_data, parent_note = cache
        else:
            raw_data, parent_note = self._preview_events(g)
            self._preview_cache = (g.get('gate_name'), raw_data, parent_note)
        if parent_note:
            title_bits.append(parent_note)
        self.lbl_preview_title.setText(" — ".join(title_bits))

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
        drew_geometry = self._draw_trained_boundary(g)

        if not drew_geometry and raw_data is None:
            txt = pg.TextItem(
                "No geometry to preview yet.\n"
                "Run Train to compute boundaries.",
                color=(120, 120, 120),
                anchor=(0.5, 0.5),
            )
            self._preview_vb.addItem(txt)

    def _on_preview_zoom(self, axis_name: str):
        """Apply an axis drag on the preview to that channel's transform.

        A change of the logicle width is announced with ``transformChanged``
        so the model's boundaries are converted before the plot is redrawn.
        """
        channel = self._preview_channels[0 if axis_name == 'x' else 1]
        axis = self._preview_axis_x if axis_name == 'x' else self._preview_axis_y
        tr = self.tf.objects.get(channel) if channel else None
        if tr is None:
            axis._pending_delta = 0
            return
        before = ag_core.transform_params(tr)
        changed = zoom_axis_transform(axis, self._preview_vb, axis_name, tr)
        if changed is None:
            return
        if changed:
            self.transformChanged.emit(channel, before)
        self._update_preview(keep_events=True)

    def _preview_events(self, gate_def: dict) -> tuple:
        """Events of the gate's parent population, and a note for the title.

        Returns ``(events, note)``. *events* is the displayed sample's
        unmixed array restricted to the parent population (subsampled to
        the display cap), or None when there is nothing to plot. The note
        gives the parent and its event count, or says why the parent could
        not be evaluated and all events are shown instead.
        """
        import honeychrome.settings as hc_settings

        data = getattr(self.controller, 'unmixed_event_data', None)
        if data is None or len(data) == 0:
            return None, ''

        cap = int(getattr(hc_settings, 'max_display_events', 500_000))
        if len(data) > cap:
            rng = np.random.default_rng(0)
            data = data[np.sort(rng.choice(len(data), cap, replace=False))]

        parent_gate = gate_def.get('parent_gate')
        parent_popul = gate_def.get('parent_popul') or 'root'
        label = f"{parent_gate}/{parent_popul}" if parent_gate else 'root'
        mask = self._parent_event_mask(gate_def, data)
        if mask is None:
            note = f"parent {label} could not be evaluated; showing all {len(data):,} events"
        else:
            data = data[mask]
            note = f"parent {label}: {len(data):,} events"
            if len(data) == 0:
                return None, note + " (empty)"
        return data, note

    def _parent_event_mask(self, gate_def: dict, data: np.ndarray):
        """Boolean mask over *data* of the events in the gate's parent
        population, or None when the parent cannot be evaluated."""
        if not gate_def.get('parent_gate'):
            return np.ones(len(data), dtype=bool)
        try:
            return self._model_parent_mask(gate_def, data)
        except Exception as exc:
            log.warning("preview: parent population of %s not evaluated: %s",
                        gate_def.get('gate_name'), exc)
            return None

    def _model_parent_mask(self, gate_def: dict, data: np.ndarray):
        """Parent membership from the model's own boundaries: trained ones,
        else the shapes imported with a gate that still applies them; other
        untrained gates are calculated on the displayed events."""
        chain = []
        name, seen = gate_def.get('parent_gate'), set()
        while name and name not in seen:
            seen.add(name)
            ancestor = self.state.gate_by_name(name)
            if ancestor is None:
                return None
            chain.append(ancestor)
            name = ancestor.get('parent_gate')
        chain.reverse()
        reference, modes = {}, {}
        for ancestor in chain:
            name = ancestor['gate_name']
            boundary = self.state.trained_boundaries.get(name)
            if not boundary and ancestor.get('algorithm') == ag_core.ALGORITHM_IMPORTED:
                boundary = ancestor.get('template')
            if boundary:
                reference[name] = boundary
            else:
                modes[name] = ag_core.MODE_RECALCULATE
        pnn = _experiment_channels(self.controller)
        transforms = self.tf.objects
        transformed = ag_core.transform_columns(data, pnn, transforms)
        result = ag_core.run_gating(
            chain, transformed, {ch: i for i, ch in enumerate(pnn)},
            reference=reference, modes=modes, default_mode=ag_core.MODE_FIXED,
            axis_limits=_axis_limits(transforms), transforms=transforms,
        )
        return result.population_mask(gate_def['parent_gate'], gate_def.get('parent_popul'))

    def _draw_trained_boundary(self, gate_def: dict) -> bool:
        """Draw any TrainedBoundary entries for *gate_def*.

        Returns True if anything was drawn.
        """
        gate_name = gate_def.get('gate_name', '')
        trained = self.state.trained_boundaries.get(gate_name)
        if trained and gate_def.get('algorithm') != ag_core.ALGORITHM_IMPORTED:
            entry = trained
        else:
            entry = gate_def.get('template') or trained
        if not entry:
            return False
        entry = ag_core.display_entries(gate_def, entry)
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
        # Shapes as drawn are shown, not adjusted here.
        self._shapes = ag_core.is_shape_boundary(state.trained_boundaries.get(gate_name))
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
        if gt == '1dsep' and not self._shapes:
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

    def _builder_transforms(self) -> dict:
        """The builder's transforms (from the TrainTab that opened this)."""
        tf = getattr(self.parent(), 'tf', None)
        return tf.objects if tf is not None else {}

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
        ch_x, ch_y = _boundary_axes(self.gate_def, self.state.trained_boundaries.get(gate_name))
        sep_y = ag_core.separation_axis(self.gate_def) == 'y'

        transforms = self._builder_transforms()
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
        if gt == '1dsep' and not self._shapes:
            entry = self._current_boundary_entry()
            tx = (entry.get('threshold_x') if entry else None) or 0.0
            self._x_line = pg.InfiniteLine(
                pos=float(tx), angle=0 if sep_y else 90, movable=True,
                pen=pg.mkPen(color=(30, 160, 50), width=2)
            )
            self._x_line.sigPositionChanged.connect(
                lambda ln: self._sb_threshold.setValue(ln.value())
                if hasattr(self, '_sb_threshold') else None
            )
            self._vb.addItem(self._x_line)
        elif gt == '2dsep' and not self._shapes:
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
        calc = ag_core.GateCalculator(_axis_limits(self._builder_transforms()))

        new_entries = None
        if self._shapes:
            pass
        elif gt == '1dsep' and hasattr(self, '_sb_threshold') and tb:
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

    pause() stops before the next gate and reports the gates trained so far
    (payload['stopped'] True, payload['done'] their names); abort() stops
    and reports nothing. *done* ({gate: boundary}) resumes a paused run:
    those gates are applied as they are and only the others calculated.
    """

    progress = Signal(str)
    finished = Signal(bool, str, object)

    def __init__(self, experiment_dir, sample_keys, snapshot, gate_defs, channels,
                 transforms, events_per_sample: int, display_cap: int, seed: int,
                 done: dict | None = None, parent=None):
        super().__init__(parent)
        self._done = dict(done or {})
        self._pause = False
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

    def pause(self):
        self._pause = True

    def _stopping(self) -> bool:
        return self._abort or self._pause

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
            if self._stopping():
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
        if self._stopping():
            return None

        pooled = np.concatenate(chunks, axis=0)
        del chunks
        n_left = len(self._gate_defs) - len(self._done)
        self.progress.emit(f"Calculating {n_left} gate(s) on {len(pooled):,} pooled events …")
        data = ag_core.transform_columns(pooled, self._channels, self._transforms)
        index = {ch: i for i, ch in enumerate(self._channels)}
        result = ag_core.train_boundaries(self._gate_defs, data, index,
                                          axis_limits=_axis_limits(self._transforms),
                                          done=self._done, stop_check=self._stopping,
                                          transforms=self._transforms)
        del data
        if result.stopped and self._abort:
            return None

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
            'stopped': bool(result.stopped),
            'done': list(result.status),
        }


class TrainTab(QWidget):
    """Tab — Train.

    The gating hierarchy runs down the left beside two sections: the
    training samples, events per sample and Calculate / Pause / Abort, and
    the gate boundaries — one tile per gate showing the trained boundary
    over a display subsample of the pooled training events (parent
    population only). Hierarchy nodes are coloured by training status.
    Tiles are made only for the gates drawn on populations open in the
    tree: the first three levels to start with, and more as nodes are
    expanded, so large hierarchies open quickly. "Detail…" on a tile opens
    GateDetailWidget to adjust thresholds.

    Pause keeps every gate trained so far; Calculate Gates then offers to
    resume (train only the remaining gates on the same pooled events) or to
    overwrite (train every gate again).
    """

    _EXPAND_DEPTH = 3
    _STATUS_COLOURS = {'trained': (10, 122, 35), 'adjusted': (30, 100, 220),
                       'failed': (192, 57, 43), 'untrained': (148, 97, 0)}

    trained = Signal()              # training finished successfully (or paused)
    boundaries_changed = Signal()   # a boundary was adjusted by hand

    def __init__(self, state: GatingModelState, bus, controller, tf=None, blocked_reason=None,
                 parent=None):
        super().__init__(parent)
        self.state = state
        self.bus = bus
        self.controller = controller
        self.tf = tf
        # callable -> '' or why training cannot start now (e.g. gating is running)
        self._blocked_reason = blocked_reason or (lambda: '')

        self._worker: _TrainingWorker | None = None
        self._tile_widgets: dict[str, QWidget] = {}
        self._layout_pending = False
        self._last_event_data: np.ndarray | None = None      # untransformed display events
        self._last_gate_masks: dict[str, np.ndarray] = {}    # gate -> parent mask over display rows
        self._run_fingerprint = ''
        self._previous_boundaries: dict = {}

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(HelpToggleWidget(text=ag_help_texts.BUILDER_TRAIN))

        # Hierarchy (always shown) beside the training and boundary sections.
        tree_panel = QWidget()
        tree_layout = QVBoxLayout(tree_panel)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        tree_header = QHBoxLayout()
        tree_header.addWidget(QLabel("<b>Gating hierarchy</b>"))
        tree_header.addStretch()
        self.btn_expand = QPushButton("Expand")
        self.btn_collapse = QPushButton("Collapse")
        self.btn_expand.setToolTip("Expand every node: shows a tile for every gate, which can "
                                   "be slow for large hierarchies.")
        self.btn_collapse.setToolTip("Collapse the hierarchy to the root.")
        tree_header.addWidget(self.btn_expand)
        tree_header.addWidget(self.btn_collapse)
        tree_layout.addLayout(tree_header)
        self.tree_filter = FilterBar(placeholder="Search populations…")
        self.tree_filter.changed.connect(lambda _t, _m: self._filter_tree())
        tree_layout.addWidget(self.tree_filter)
        self.pop_tree = PopulationTree()
        self.pop_tree.setToolTip("Select a population to show the gates drawn on it. Colours: "
                                 "green trained, blue adjusted by hand, amber untrained, red "
                                 "failed to train.")
        tree_layout.addWidget(self.pop_tree, stretch=1)
        self.lbl_tree_status = QLabel("")
        self.lbl_tree_status.setWordWrap(True)
        self.lbl_tree_status.setStyleSheet("color: #555; font-style: italic;")
        tree_layout.addWidget(self.lbl_tree_status)
        self.btn_expand.clicked.connect(self.pop_tree.expandAll)
        self.btn_collapse.clicked.connect(self.pop_tree.collapseAll)
        self.pop_tree.expansionChanged.connect(self._schedule_layout)
        self.pop_tree.populationSelected.connect(self._on_tree_selected)

        # One section open at a time; each page scrolls on its own.
        self.toolbox = make_concertina()

        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(4, 4, 4, 4)
        top_layout.addWidget(QLabel("<b>Training samples</b>"))

        self.picker = OrderedMultiSamplePicker()
        top_layout.addWidget(self.picker)

        opts_row = QHBoxLayout()
        opts_row.addWidget(QLabel("Events per sample:"))
        self.sb_events = block_wheel(QSpinBox())
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
        guard_while_busy(self.controller, self.btn_run)
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.setToolTip("Stop before the next gate, keeping every gate trained so far; "
                                  "Calculate Gates then offers to resume or overwrite.")
        self.btn_abort = QPushButton("Abort")
        self.btn_abort.setEnabled(False)
        self.btn_abort.setToolTip("Stop and discard this run, keeping the boundaries from before it.")
        self.btn_run.clicked.connect(self._on_run_clicked)
        self.btn_pause.clicked.connect(self._on_pause_clicked)
        self.btn_abort.clicked.connect(self._on_abort_clicked)
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #555;")
        self.lbl_status.setWordWrap(True)
        run_row.addWidget(self.btn_run)
        run_row.addWidget(self.btn_pause)
        run_row.addWidget(self.btn_abort)
        run_row.addWidget(self.lbl_status, stretch=1)
        top_layout.addLayout(run_row)
        top_layout.addStretch()

        self.grid_scroll = QScrollArea()
        self.grid_scroll.setWidgetResizable(True)
        self.grid_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(4)
        self.grid_layout.setContentsMargins(4, 4, 4, 4)
        self.grid_scroll.setWidget(self.grid_container)

        tiles_page = QWidget()
        tiles_layout = QVBoxLayout(tiles_page)
        tiles_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_tiles = QLabel("")
        self.lbl_tiles.setStyleSheet("color: #555; font-style: italic;")
        tiles_layout.addWidget(self.lbl_tiles)
        tiles_layout.addWidget(self.grid_scroll, stretch=1)

        add_concertina_page(self.toolbox, top, "Training")
        self._page_tiles = self.toolbox.addItem(tiles_page, "Gate boundaries")

        split = QSplitter(Qt.Horizontal)
        split.addWidget(tree_panel)
        split.addWidget(self.toolbox)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 4)
        split.setSizes([260, 760])
        outer.addWidget(split, stretch=1)

    # ------------------------------------------------------------------
    # Public refresh hook
    # ------------------------------------------------------------------

    def refresh(self):
        """Repopulate the sample picker and the hierarchy from the state."""
        self.sb_events.blockSignals(True)
        self.sb_events.setValue(int(self.state.events_per_sample))
        self.sb_events.blockSignals(False)
        self.chk_controls.blockSignals(True)
        self.chk_controls.setChecked(bool(self.state.include_controls))
        self.chk_controls.blockSignals(False)
        self._populate_picker()
        self._rebuild_grid()
        if self.state.training_progress and self._worker is None:
            done = len(self.state.training_progress.get('done') or [])
            self.lbl_status.setText(f"Paused: {done} of {self.state.n_gates()} gate(s) trained. "
                                    "Calculate Gates to resume or overwrite.")

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
    # Hierarchy status
    # ------------------------------------------------------------------

    def _gate_status(self, gate_name: str) -> str:
        """'trained', 'adjusted', 'failed' or 'untrained'."""
        entries = self.state.trained_boundaries.get(gate_name)
        if self.state.is_trained(gate_name):
            if any((e or {}).get('adjusted') for e in (entries or {}).values()):
                return 'adjusted'
            return 'trained'
        status = (self.state.training_summary.get('gate_status') or {}).get(gate_name) or {}
        return 'failed' if status.get('flag') in ('failed', 'skipped') else 'untrained'

    def _mark_tree(self):
        """Colour each population by the training status of its gate."""
        marks, counts = {}, {}
        for g in self.state.gate_definitions:
            status = self._gate_status(g['gate_name'])
            counts[status] = counts.get(status, 0) + 1
            for pop in (g.get('populations') or {}):
                marks[(g['gate_name'], pop)] = (self._STATUS_COLOURS[status], status)
        self.pop_tree.mark(marks)
        parts = [f"{counts[k]} {k}" for k in ('trained', 'adjusted', 'untrained', 'failed')
                 if counts.get(k)]
        self.lbl_tree_status.setText(", ".join(parts))

    def _filter_tree(self):
        text = self.tree_filter.text()
        if not text:
            self.pop_tree.set_filter(None)
            return
        hits = set()
        for g in self.state.gate_definitions:
            for pop, info in (g.get('populations') or {}).items():
                if text_matches(text, g['gate_name'], (info or {}).get('label', pop), pop):
                    hits.add((g['gate_name'], pop))
        self.pop_tree.set_filter(hits)

    # ------------------------------------------------------------------
    # Tile grid
    # ------------------------------------------------------------------

    def _rebuild_grid(self):
        """Refill the hierarchy tree and redraw the tiles of the gates in
        view from state.trained_boundaries."""
        while self.grid_layout.count():
            self.grid_layout.takeAt(0)
        for tile in self._tile_widgets.values():
            tile.setParent(None)
            tile.deleteLater()
        self._tile_widgets.clear()
        self.pop_tree.set_nodes(ag_review.population_tree(self.state.gate_definitions),
                                expand_depth=self._EXPAND_DEPTH)
        self._mark_tree()
        self._filter_tree()
        self._layout_tiles()

    def _schedule_layout(self):
        """Lay the tiles out once the tree has finished expanding or
        collapsing (Expand all sends one signal per node)."""
        if self._layout_pending:
            return
        self._layout_pending = True
        QTimer.singleShot(0, self._layout_tiles)

    def _shown_gates(self) -> list[dict]:
        """Gates drawn on populations open in the tree, in hierarchy order."""
        out = []
        for g in ag_core.ordered_gate_defs(self.state.gate_definitions):
            parent = (g.get('parent_gate') or '',
                      (g.get('parent_popul') or 'root') if g.get('parent_gate') else 'root')
            if self.pop_tree.is_open(parent):
                out.append(g)
        return out

    def _layout_tiles(self):
        """Show a tile for each gate in view, making tiles only when first
        needed; tiles of gates out of view are hidden and kept."""
        self._layout_pending = False
        while self.grid_layout.count():
            self.grid_layout.takeAt(0)
        shown = self._shown_gates()
        names = {g.get('gate_name', '') for g in shown}
        for name, tile in self._tile_widgets.items():
            if name not in names:
                tile.hide()
        cols = 4
        for i, g in enumerate(shown):
            name = g.get('gate_name', '')
            tile = self._tile_widgets.get(name)
            if tile is None:
                tile = self._make_tile(g)
                self._tile_widgets[name] = tile
            row, col = divmod(i, cols)
            self.grid_layout.addWidget(tile, row, col)
            tile.show()
        total = len(self.state.gate_definitions)
        if total and len(shown) < total:
            self.lbl_tiles.setText(f"Showing {len(shown)} of {total} gates. Expand populations "
                                   "in the hierarchy to show the gates drawn on them.")
        else:
            self.lbl_tiles.setText(f"Showing all {total} gates." if total else "")

    def _on_tree_selected(self, gate: str, pop: str):
        """Open the selected population and scroll to the first of its gates."""
        key = (gate or '', pop)
        if self.toolbox.currentIndex() != self._page_tiles:
            self.toolbox.setCurrentIndex(self._page_tiles)
        if not self.pop_tree.is_open(key):
            self.pop_tree.expand_population(gate, pop)
            self._layout_tiles()
        for g in self._shown_gates():
            parent = (g.get('parent_gate') or '',
                      (g.get('parent_popul') or 'root') if g.get('parent_gate') else 'root')
            if parent == key and g.get('gate_name') in self._tile_widgets:
                self.grid_scroll.ensureWidgetVisible(self._tile_widgets[g['gate_name']])
                break

    def _make_tile(self, gate_def: dict) -> QWidget:
        """Create one plot tile for a gate."""
        import honeychrome.settings as hc_settings
        import colorcet as cc
        from PySide6.QtCore import QRectF

        gate_name = gate_def.get('gate_name', '')
        eff = ag_core.effective_gate_def(
            gate_def, ag_core.gates_by_name(self.state.gate_definitions))
        gt = eff.get('gate_type', '')
        tb = self.state.trained_boundaries.get(gate_name, {})
        shapes = ag_core.is_shape_boundary(tb)
        ch_x, ch_y = _boundary_axes(eff, tb)

        pnn = list(
            self.controller.experiment.settings.get('unmixed', {})
            .get('event_channels_pnn') or []
        )
        spectral_model = self.controller.experiment.process.get('spectral_model') or []
        pnn_labels = build_display_label_map(pnn, spectral_model)
        x_label = pnn_labels.get(ch_x, ch_x)
        y_label = pnn_labels.get(ch_y, ch_y) if ch_y else 'Count'
        transforms = self.tf.objects
        tr_x = transforms.get(ch_x)
        tr_y = transforms.get(ch_y) if ch_y else None
        line_colour = ag_core.algorithm_colour(gate_def.get('algorithm'))

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
            if gt == '1dsep' and not shapes:
                sep_y = ag_core.separation_axis(eff) == 'y'
                entry = next(iter(tb.values()))
                tx = entry.get('threshold_x') or 0.0
                line = pg.InfiniteLine(
                    pos=float(tx), angle=0 if sep_y else 90,
                    pen=pg.mkPen(color=line_colour, width=1.5)
                )
                vb.addItem(line)
                view_tr = tr_x if sep_y else tr_y
                for pop_name, pop_data in tb.items():
                    frac = pop_data.get('fraction')
                    if frac is not None:
                        bd = pop_data.get('boundary', [])
                        along = [p[0] for p in bd]
                        at = float(np.mean(along)) if along else float(tx)
                        lim = view_tr.limits if view_tr is not None else (
                            tr_x.limits if tr_x is not None else [0, 1])
                        across = (lim[0] + lim[1]) * 0.5
                        pop_label = pop_data.get('label', pop_name)
                        pct_txt = pg.TextItem(
                            f"{pop_label}\n{frac * 100:.1f}%",
                            color=line_colour, anchor=(0.5, 0.5),
                        )
                        if sep_y:
                            pct_txt.setPos(across, at)
                        else:
                            pct_txt.setPos(at, across)
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
                        pen=pg.mkPen(color=line_colour, width=1.5)
                    )
                    vb.addItem(line_item)
                    frac = pop_data.get('fraction')
                    if frac is not None:
                        cx = float(np.mean(xs_b))
                        cy = float(np.mean(ys_b))
                        pop_label = pop_data.get('label', pop_name)
                        pct_txt = pg.TextItem(
                            f"{pop_label}\n{frac * 100:.1f}%",
                            color=line_colour, anchor=(0.5, 0.5),
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
        transforms = self.tf.objects
        data = ag_core.transform_columns(self._last_event_data, channels, transforms)
        index = {ch: i for i, ch in enumerate(channels)}
        result = ag_core.run_gating(self.state.gate_definitions, data, index,
                                    reference=self.state.trained_boundaries,
                                    default_mode=ag_core.MODE_FIXED,
                                    axis_limits=_axis_limits(transforms), transforms=transforms)
        for gate_name, entries in result.boundaries.items():
            stored = self.state.trained_boundaries.get(gate_name, {})
            for pop, entry in entries.items():
                if pop in stored:
                    stored[pop]['fraction'] = entry.get('fraction')
        self._last_gate_masks = dict(result.parent_masks)

    # ------------------------------------------------------------------
    # Run / abort handlers
    # ------------------------------------------------------------------

    def _fingerprint(self, selected: list[str]) -> str:
        """What a training run's result depends on: samples, events per
        sample, seed, the gate definitions and the transforms."""
        import hashlib
        payload = {'samples': list(selected), 'events': int(self.state.events_per_sample),
                   'seed': TRAINING_SEED, 'gates': self.state.gate_definitions,
                   'transforms': {ch: {k: v for k, v in p.items() if k != 'limits'}
                                  for ch, p in self.tf.params().items()}}
        return hashlib.sha256(json.dumps(ag_core.to_jsonable(payload), sort_keys=True,
                                         default=str).encode('utf-8')).hexdigest()

    def _ask_resume(self, n_done: int, compatible: bool) -> str | None:
        """'resume', 'overwrite' or None for a paused training run."""
        box = QMessageBox(self)
        box.setWindowTitle("Paused training")
        total = self.state.n_gates()
        if compatible:
            box.setText(f"Training was paused with {n_done} of {total} gate(s) trained.\n\n"
                        "Resume keeps them and trains the remaining gates; Overwrite trains "
                        "every gate again.")
            resume = box.addButton("Resume", QMessageBox.AcceptRole)
        else:
            box.setText(f"Training was paused with {n_done} of {total} gate(s) trained, but the "
                        "training samples, events per sample, gates or transforms have changed "
                        "since, so it cannot be resumed.\n\nOverwrite trains every gate again.")
            resume = None
        overwrite = box.addButton("Overwrite", QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if resume is not None and clicked is resume:
            return 'resume'
        if clicked is overwrite:
            return 'overwrite'
        return None

    def is_running(self) -> bool:
        return self._worker is not None

    def _on_run_clicked(self):
        blocked = self._blocked_reason()
        if blocked:
            QMessageBox.information(self, "Cannot train now", blocked)
            return
        selected = self.picker.get_ordered_list()
        if not selected:
            QMessageBox.information(self, "No samples", "Select at least one training sample.")
            return
        if not self.state.gate_definitions:
            QMessageBox.information(self, "No gates",
                                    "Import or add gates on the Import/Setup tab first.")
            return
        token = acquire_job(self.controller, 'Automated Gating: training')
        if token is None:
            self.lbl_status.setText(self.controller.job_gate.busy_text())
            return
        try:
            ag_core.ordered_gate_defs(self.state.gate_definitions)
            snap = snapshot_unmix_state(self.controller, selected)
        except Exception as exc:
            token.release()
            QMessageBox.warning(self, "Cannot train", str(exc))
            return
        fingerprint = self._fingerprint(selected)
        done = {}
        progress = self.state.training_progress
        if progress:
            names = [n for n in progress.get('done') or [] if self.state.is_trained(n)]
            choice = self._ask_resume(len(names), progress.get('fingerprint') == fingerprint)
            if choice is None:
                token.release()
                return
            if choice == 'resume':
                done = {n: deepcopy(self.state.trained_boundaries[n]) for n in names}
        self.state.training_samples = list(selected)
        self._run_fingerprint = fingerprint
        self._previous_boundaries = dict(self.state.trained_boundaries)

        import honeychrome.settings as hc_settings
        display_cap = int(getattr(hc_settings, 'max_display_events', 500_000) or 500_000)

        self.btn_run.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_abort.setEnabled(True)
        self.lbl_status.setText("Resuming …" if done else "Running …")

        self._worker = _TrainingWorker(
            self.controller.experiment_dir, selected, snap,
            deepcopy(self.state.gate_definitions),
            _experiment_channels(self.controller),
            deepcopy(self.tf.objects),
            events_per_sample=self.state.events_per_sample,
            display_cap=display_cap, seed=TRAINING_SEED, done=done, parent=self,
        )
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(lambda *_: token.release())
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_pause_clicked(self):
        if self._worker is not None:
            self._worker.pause()
            self.btn_pause.setEnabled(False)
            self.lbl_status.setText("Pausing …")

    def _on_abort_clicked(self):
        if self._worker is not None:
            self._worker.abort()
            self.btn_pause.setEnabled(False)
            self.btn_abort.setEnabled(False)
            self.lbl_status.setText("Aborting …")

    def _on_worker_progress(self, msg: str):
        self.lbl_status.setText(msg)
        ts = datetime.now().strftime('%H:%M:%S')
        self.bus.statusMessage.emit(f"[Automated Gating {ts}] {msg}")

    def _on_worker_finished(self, success: bool, error: str, payload: object):
        self.btn_run.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_abort.setEnabled(False)
        self._worker = None
        if not success:
            text = ("Stopped before any gate was trained; the boundaries from before are kept."
                    if error == "Cancelled." else f"Error: {error}")
            self.lbl_status.setText(text)
            ts = datetime.now().strftime('%H:%M:%S')
            self.bus.statusMessage.emit(f"[Automated Gating {ts}] Training: {text}")
            return

        stopped = bool(payload.get('stopped'))
        self.state.trained_boundaries = payload['boundaries']
        self.state.training_summary = {
            'training_sample_names': list(payload['per_sample'].keys()),
            'n_events_per_sample': {k: v['n_events_used'] for k, v in payload['per_sample'].items()},
            'events_cap_per_sample': int(self.state.events_per_sample),
            'seed': TRAINING_SEED,
            'n_events_pooled': payload['n_pooled'],
            'per_sample': payload['per_sample'],
            'gate_status': payload['status'],
            'complete': not stopped,
        }
        self.state.training_progress = ({'fingerprint': self._run_fingerprint,
                                         'done': list(payload.get('done') or [])}
                                        if stopped else {})
        self._last_event_data = payload['display_events']
        self._last_gate_masks = payload['display_parent_masks']
        self._rebuild_grid()
        self.toolbox.setCurrentIndex(self._page_tiles)

        n_trained = len(payload['boundaries'])
        self.trained.emit()
        if stopped:
            left = self.state.n_gates() - n_trained
            self.lbl_status.setText(f"Paused — {n_trained} gate(s) trained, {left} to go. "
                                    "Calculate Gates to resume or overwrite.")
            return
        problems = [
            f"{name}: {st.get('message') or st.get('flag')}"
            for name, st in payload['status'].items() if st.get('flag')
        ]
        self.lbl_status.setText(f"Done — {n_trained} gate(s) trained.")
        text = (f"{n_trained} gate(s) trained on {payload['n_pooled']:,} events from "
                f"{len(payload['per_sample'])} sample(s).")
        if problems:
            text += "\n\nNot trained:\n" + "\n".join(problems[:15])
        QMessageBox.information(self, "Training Complete", text)


class _ReviewWorker(QThread):
    """Load and unmix the review samples for the Hierarchy tab, each capped at
    the same number of events, and pool them."""

    finished = Signal(bool, str, object)   # ok, message, unmixed events

    def __init__(self, experiment_dir, sample_keys, snapshot, max_events, parent=None):
        super().__init__(parent)
        self._experiment_dir = experiment_dir
        self._keys = list(sample_keys)
        self._snap = snapshot
        self._max_events = max_events

    def run(self):
        try:
            chunks = []
            for key in self._keys:
                out = load_unmixed(self._experiment_dir, key, self._snap,
                                   max_events=self._max_events, seed=TRAINING_SEED)
                chunks.append(np.asarray(out['unmixed'], dtype=np.float32))
            self.finished.emit(True, '', np.concatenate(chunks, axis=0))
        except Exception as exc:
            log.exception("review sample load failed")
            self.finished.emit(False, str(exc), None)


_REVIEW_COLOURS = [(30, 160, 50), (210, 100, 20), (40, 100, 210), (170, 50, 170), (0, 150, 150)]
_REVIEW_ALERT = (200, 30, 30)
_TITLE_NAMES_CHARS = 70


def _pop_key_text(gate, pop) -> str:
    """A population as combo-box data (PySide6 returns stored tuples as lists)."""
    return f"{gate or ''}\x1f{pop}"


def _pop_key_tuple(text):
    if text is None:
        return None, 'root'
    gate, _, pop = str(text).partition('\x1f')
    return (gate or None), (pop or 'root')


class _ReviewPlot(QWidget):
    """One biplot (or histogram) of a parent population with the daughter
    gates that use its parameters drawn on it.

    ``gateEdited(gate_name, entries)`` is emitted after a threshold line,
    range limit or polygon is moved in edit mode. Right-clicking a gated
    population offers to open it (``populationRequested(gate, pop)``);
    ``can_open(gate, pop)``, when set, says whether it has gates below.
    """

    gateEdited = Signal(str, object)
    populationRequested = Signal(str, str)
    # (channel, transform parameters before the drag, or None when only the range moved)
    axisZoomed = Signal(str, object)

    def __init__(self, channels, gates, controller, tile_size, parent=None, caption: str = '',
                 transforms: dict | None = None):
        super().__init__(parent)
        self.channels = tuple(channels)
        self.gates = list(gates)
        self.controller = controller
        self.caption = caption
        self._overlay: list = []
        self._lines: dict = {}
        self._rois: dict = {}
        self._shown: list = []
        self.can_open = None
        self._transforms = transforms if transforms is not None else (
            controller.unmixed_transformations or {})
        self._pnn = _experiment_channels(controller)
        labels = build_display_label_map(
            self._pnn, controller.experiment.process.get('spectral_model') or [])
        self._labels = [labels.get(c, c) for c in self.channels]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self.lbl_title = QLabel()
        self.lbl_title.setAlignment(Qt.AlignCenter)
        # Two lines, always: plots side by side stay level.
        self.lbl_title.setFixedHeight(2 * self.lbl_title.fontMetrics().lineSpacing() + 6)
        layout.addWidget(self.lbl_title)

        self.gw = TransparentGraphicsLayoutWidget()
        self.gw.setFixedSize(tile_size, tile_size)
        gl = self.gw.ci.layout
        gl.setHorizontalSpacing(0)
        gl.setVerticalSpacing(0)
        self.vb = NoPanViewBox()
        self.vb.setMouseEnabled(x=False, y=False)
        self.vb.raiseContextMenu = self._context_menu
        self.gw.addItem(self.vb, row=1, col=2)
        self.axis_x = ZoomAxis('bottom', self.vb)
        self.axis_y = ZoomAxis('left', self.vb)
        self.gw.addItem(self.axis_y, row=1, col=1)
        self.gw.addItem(self.axis_x, row=2, col=2)
        self.axis_x.linkToView(self.vb)
        self.axis_y.linkToView(self.vb)
        self.axis_x.zoom_timer.timeout.connect(lambda: self._on_zoom('x'))
        self.axis_y.zoom_timer.timeout.connect(lambda: self._on_zoom('y'))
        self.axis_x.setLabel(self._labels[0])
        self.axis_y.setLabel(self._labels[1] if len(self._labels) > 1 else 'Count')
        self.img = pg.ImageItem()
        self.vb.addItem(self.img)
        self.hist = pg.PlotDataItem(stepMode='center', fillLevel=0, brush=(100, 100, 250, 150))
        self.vb.addItem(self.hist)
        layout.addWidget(self.gw, alignment=Qt.AlignCenter)

    def _tr(self, i):
        return self._transforms.get(self.channels[i]) if i < len(self.channels) else None

    def _on_zoom(self, axis_name: str):
        """Apply an axis drag to that channel's transform and announce it."""
        i = 0 if axis_name == 'x' else 1
        axis = self.axis_x if axis_name == 'x' else self.axis_y
        tr = self._tr(i)
        if tr is None:
            axis._pending_delta = 0
            return
        before = ag_core.transform_params(tr)
        changed = zoom_axis_transform(axis, self.vb, axis_name, tr)
        if changed is not None:
            self.axisZoomed.emit(self.channels[i], before if changed else None)

    def set_events(self, raw, mask):
        """Draw the density of the parent population (rows of *raw* in *mask*)."""
        import honeychrome.settings as hc_settings
        import colorcet as cc
        from PySide6.QtCore import QRectF

        self.img.clear()
        self.hist.setData([], [])
        tr_x, tr_y = self._tr(0), self._tr(1)
        if raw is None or tr_x is None or self.channels[0] not in self._pnn:
            return
        ix = self._pnn.index(self.channels[0])
        if len(self.channels) > 1 and tr_y is not None and self.channels[1] in self._pnn:
            try:
                colors = cc.palette[hc_settings.colourmap_name_retrieved]
            except Exception:
                colors = cc.palette['rainbow4']
            cmap = pg.ColorMap(pos=0.9 * np.linspace(0, 1, len(colors)) ** 2
                               + 0.1 * np.linspace(0, 1, len(colors)), color=colors)
            lut = cmap.getLookupTable(alpha=True)
            lut[0, 3] = 0
            self.img.setLookupTable(lut)
            iy = self._pnn.index(self.channels[1])
            self.img.setImage(calc_hist2d(raw, mask, ix, iy, tr_x, tr_y,
                                          density_cutoff=hc_settings.density_cutoff_retrieved))
            self.img.setRect(QRectF(tr_x.limits[0], tr_y.limits[0],
                                    tr_x.limits[1] - tr_x.limits[0],
                                    tr_y.limits[1] - tr_y.limits[0]))
            self._set_axis(self.axis_x, tr_x, x=True)
            self._set_axis(self.axis_y, tr_y, x=False)
        else:
            self.hist.setData(tr_x.step_scale, calc_hist1d(raw, mask, ix, tr_x))
            self._set_axis(self.axis_x, tr_x, x=True)
            self.vb.enableAutoRange(axis=self.vb.YAxis, enable=True)

    def _set_axis(self, axis, tr, x: bool):
        if tr.ticks:
            axis.setTicks(tr.ticks())
        axis.zoomZero = tr.zero
        axis.fullRange = (0, 1.1)
        axis.limits = tuple(tr.limits)
        (self.vb.setXRange if x else self.vb.setYRange)(tr.limits[0], tr.limits[1], padding=0)

    # -- overlays -----------------------------------------------------------

    def clear_overlays(self):
        for item in self._overlay:
            try:
                self.vb.removeItem(item)
            except Exception:
                pass
        self._overlay, self._lines, self._rois, self._shown = [], {}, {}, []

    def _add(self, item):
        self.vb.addItem(item)
        self._overlay.append(item)
        return item

    def draw_overlays(self, boundaries: dict, texts: dict, editable: set,
                      colours: dict | None = None):
        """Draw every daughter gate.

        boundaries : {gate: {pop: entry}} to draw.
        texts      : {(gate, pop): (text, alert)} label for each population.
        editable   : names of the gates whose boundaries can be moved.
        colours    : optional {gate: RGB, or {'x': RGB, 'y': RGB} for a
                     threshold gate's two lines}; default one colour per
                     gate in turn.
        """
        self.clear_overlays()
        names = ", ".join(g['gate_name'] for g in self.gates)
        short = names if len(names) <= _TITLE_NAMES_CHARS else names[:_TITLE_NAMES_CHARS - 1] + '…'
        axes = f"{self._labels[0]} vs {self._labels[1]}" if len(self._labels) > 1 else self._labels[0]
        head = f"{self.caption} — {axes}" if self.caption else axes
        self.lbl_title.setText(f"<b>{head}</b><br><small>{short}</small>")
        self.lbl_title.setToolTip(names)
        for k, gate in enumerate(self.gates):
            colour = (colours or {}).get(gate['gate_name']) or _REVIEW_COLOURS[k % len(_REVIEW_COLOURS)]
            entries = boundaries.get(gate['gate_name']) or {}
            if not entries:
                continue
            movable = gate['gate_name'] in editable
            gtype = gate.get('gate_type')
            if gtype in ag_core.THRESHOLD_TYPES and not ag_core.is_shape_boundary(entries):
                entries = ag_core.display_entries(gate, entries)
                self._draw_thresholds(gate, entries, colour, movable)
            else:
                for pop, entry in entries.items():
                    if entry.get('range') is not None and len(self.channels) == 1:
                        self._draw_range(gate, pop, entry, colour, movable)
                    else:
                        self._draw_polygon(gate, pop, entry, colour, movable)
            self._shown.append((gate, entries))
            for pop, entry in entries.items():
                text = texts.get((gate['gate_name'], pop))
                if text is None:
                    continue
                pos = self._label_position(entry)
                if pos is None:
                    continue
                text_colour = colour.get('x') if isinstance(colour, dict) else colour
                label = pg.TextItem(text[0], color=_REVIEW_ALERT if text[1] else text_colour,
                                    anchor=(0.5, 0.5))
                label.setPos(*pos)
                self._add(label)

    def populations_at(self, x: float, y: float) -> list[tuple[str, str]]:
        """(gate, population) of every drawn population containing the plot
        point (x, y); on a histogram only x counts."""
        from matplotlib.path import Path as MplPath
        hits = []
        for gate, entries in self._shown:
            for pop, entry in entries.items():
                rng = entry.get('range')
                coords = np.asarray(entry.get('boundary') or [], dtype=float)
                if len(self.channels) == 1:
                    tx = entry.get('threshold_x')
                    if tx is not None and entry.get('region') in (ag_core.REGION_ABOVE,
                                                                  ag_core.REGION_BELOW):
                        inside = (x >= float(tx)) == (entry['region'] == ag_core.REGION_ABOVE)
                    elif rng is not None:
                        inside = float(rng[0]) <= x <= float(rng[1])
                    elif coords.ndim == 2 and len(coords):
                        inside = float(coords[:, 0].min()) <= x <= float(coords[:, 0].max())
                    else:
                        inside = False
                elif coords.ndim == 2 and coords.shape[0] >= 3:
                    inside = bool(MplPath(coords).contains_point((x, y)))
                else:
                    inside = False
                if inside:
                    hits.append((gate['gate_name'], pop))
        return hits

    def _context_menu(self, ev):
        """Right-click on a gated population: offer to open it."""
        try:
            point = self.vb.mapSceneToView(ev.scenePos())
            hits = self.populations_at(float(point.x()), float(point.y()))
        except Exception:
            log.exception("population lookup failed")
            return
        if not hits:
            return
        menu = QMenu(self)
        for gate_name, pop in hits:
            if self.can_open is not None and not self.can_open(gate_name, pop):
                action = menu.addAction(f"'{pop}' has no gates below it")
                action.setEnabled(False)
                continue
            menu.addAction(f"Open '{pop}'",
                           lambda g=gate_name, p=pop: self.populationRequested.emit(g, p))
        menu.exec(ev.screenPos().toPoint())

    def _label_position(self, entry):
        coords = np.asarray(entry.get('boundary') or [], dtype=float)
        if coords.ndim != 2 or coords.shape[0] < 2:
            return None
        x = float(np.mean(coords[:, 0]))
        if len(self.channels) == 1:
            return x, self._y_top()
        return x, float(np.mean(coords[:, 1]))

    def _y_top(self) -> float:
        yr = self.vb.viewRange()[1]
        return float(yr[0] + 0.9 * (yr[1] - yr[0]))

    def _pen(self, colour, width=2):
        return pg.mkPen(color=colour, width=width)

    def _draw_thresholds(self, gate, entries, colour, movable):
        first = next(iter(entries.values()))
        tx, ty = first.get('threshold_x'), first.get('threshold_y')
        cx = colour.get('x') if isinstance(colour, dict) else colour
        cy = colour.get('y', cx) if isinstance(colour, dict) else colour
        lines = {}
        if not movable and tx is not None and ty is not None and len(self.channels) > 1 \
                and gate.get('gate_type') == '2dsep':
            # A joined region: draw each threshold only where it separates.
            sides = {a: ag_core.threshold_sides(gate, a) for a in ('x', 'y')}
            if any(len(v) == 1 for v in sides.values()):
                xr, yr = self._channel_limits(0), self._channel_limits(1)
                for axis, colour_ in (('x', cx), ('y', cy)):
                    for side in sides[axis]:
                        if axis == 'x':
                            ys = (yr[0], float(ty)) if side == '-' else (float(ty), yr[1])
                            item = pg.PlotDataItem([float(tx)] * 2, list(ys), pen=self._pen(colour_))
                        else:
                            xs = (xr[0], float(tx)) if side == '-' else (float(tx), xr[1])
                            item = pg.PlotDataItem(list(xs), [float(ty)] * 2, pen=self._pen(colour_))
                        self._add(item)
                self._lines[gate['gate_name']] = {}
                return
        if tx is not None:
            lines['x'] = self._add(pg.InfiniteLine(pos=float(tx), angle=90, movable=movable,
                                                   pen=self._pen(cx, 3 if movable else 2)))
        if ty is not None and len(self.channels) > 1:
            lines['y'] = self._add(pg.InfiniteLine(pos=float(ty), angle=0, movable=movable,
                                                   pen=self._pen(cy, 3 if movable else 2)))
        self._lines[gate['gate_name']] = lines
        if movable:
            for line in lines.values():
                line.setHoverPen(self._pen(_REVIEW_ALERT, 3))
                line.sigPositionChangeFinished.connect(
                    lambda _l=None, g=gate: self._on_thresholds_moved(g))

    def _channel_limits(self, k: int) -> tuple[float, float]:
        """Display limits of the plot's channel *k* (0: x, 1: y): its
        transform's limits, else the view range."""
        ch = self.channels[k] if k < len(self.channels) else None
        lim = getattr(self._transforms.get(ch), 'limits', None) if ch else None
        if lim is not None and len(lim) >= 2:
            return float(lim[0]), float(lim[1])
        rng = self.vb.viewRange()[k]
        return float(rng[0]), float(rng[1])

    def _on_thresholds_moved(self, gate):
        try:
            lines = self._lines.get(gate['gate_name']) or {}
            tx = lines['x'].value() if 'x' in lines else None
            ty = lines['y'].value() if 'y' in lines else None
            if ag_core.separation_axis(gate) == 'y':
                tx, ty = ty, None
            limits = {c: tuple(self._transforms[c].limits) for c in self.channels
                      if c in self._transforms}
            self.gateEdited.emit(gate['gate_name'], ag_review.with_thresholds(gate, tx, ty, limits))
        except Exception:
            log.exception("threshold edit failed")

    def _draw_range(self, gate, pop, entry, colour, movable):
        colour = colour.get('x') if isinstance(colour, dict) else colour
        lo, hi = entry['range']
        pair = [self._add(pg.InfiniteLine(pos=float(v), angle=90, movable=movable,
                                          pen=self._pen(colour, 3 if movable else 2)))
                for v in (lo, hi)]
        self._lines[(gate['gate_name'], pop)] = pair
        if movable:
            for line in pair:
                line.setHoverPen(self._pen(_REVIEW_ALERT, 3))
                line.sigPositionChangeFinished.connect(
                    lambda _l=None, g=gate, p=pop, pr=pair: self.gateEdited.emit(
                        g['gate_name'],
                        ag_review.with_range(g, p, pr[0].value(), pr[1].value())))

    def _draw_polygon(self, gate, pop, entry, colour, movable):
        colour = colour.get('x') if isinstance(colour, dict) else colour
        coords = np.asarray(entry.get('boundary') or [], dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 3:
            return
        if not movable:
            closed = coords if np.allclose(coords[0], coords[-1]) else np.vstack([coords, coords[:1]])
            self._add(pg.PlotDataItem(closed[:, 0], closed[:, 1], pen=self._pen(colour)))
            return
        roi = pg.PolyLineROI(ag_review.polygon_vertices(coords), closed=True,
                             pen=self._pen(colour, 3), movable=True)
        self._add(roi)
        self._rois[(gate['gate_name'], pop)] = roi
        roi.sigRegionChangeFinished.connect(
            lambda _r=None, g=gate, p=pop, r=roi: self._on_polygon_moved(g, p, r))

    def _on_polygon_moved(self, gate, pop, roi):
        try:
            pts = []
            for _name, local in roi.getLocalHandlePositions():
                mapped = roi.mapToParent(local)
                pts.append((float(mapped.x()), float(mapped.y())))
            self.gateEdited.emit(gate['gate_name'], ag_review.with_polygon(gate, pop, pts))
        except Exception:
            log.exception("polygon edit failed")


def _review_gating(defs, trained, data, index, limits, stop_check, transforms=None):
    """Gate the review events with the hierarchy as reviewed: trained
    boundaries, shapes as drawn for gates still using them, and the
    trainable form of converted gates not yet trained."""
    reference = ag_review.reference_boundaries(defs, trained)
    modes = {}
    for g in defs:
        name = g['gate_name']
        if name in trained or g.get('algorithm') == ag_core.ALGORITHM_IMPORTED:
            continue
        if not g.get('template'):
            reference.pop(name, None)
        modes[name] = ag_core.MODE_RECALCULATE
    return ag_core.run_gating(defs, data, index, reference=reference, modes=modes,
                              default_mode=ag_core.MODE_FIXED, axis_limits=limits,
                              stop_check=stop_check, keep_partial=True, transforms=transforms)


class _OverlapWorker(QThread):
    """Find populations defined twice in the hierarchy on the review
    events (see ag_review.overlapping_populations); finished(ok, message,
    {gate: flag})."""

    finished = Signal(bool, str, object)

    MAX_EVENTS = 50_000

    def __init__(self, gate_defs, trained, data, pnn, limits, limit, transforms=None,
                 parent=None):
        super().__init__(parent)
        self._defs = gate_defs
        self._trained = trained
        self._data = data
        self._pnn = list(pnn)
        self._limits = limits
        self._transforms = transforms
        self._limit = float(limit)
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            index = {c: i for i, c in enumerate(self._pnn)}
            result = _review_gating(self._defs, self._trained, self._data, index, self._limits,
                                    lambda: self._abort, transforms=self._transforms)
            if self._abort:
                self.finished.emit(False, "Cancelled.", {})
                return
            n = len(self._data)
            keep = slice(None)
            if n > self.MAX_EVENTS:
                rng = np.random.default_rng(TRAINING_SEED)
                keep = np.sort(rng.choice(n, self.MAX_EVENTS, replace=False))
            masks = {(gate, pop): m[keep] for gate, pops in result.masks.items()
                     for pop, m in pops.items()}
            pairs = ag_review.overlapping_populations(self._defs, masks, limit=self._limit)
            self.finished.emit(True, '', ag_review.overlap_flags(pairs))
        except Exception as exc:
            log.exception("overlap check failed")
            self.finished.emit(False, str(exc), {})


_NO_ANCHOR = object()     # ReviewTab._anchor not looked up yet


class _EvidenceWorker(QThread):
    """What the review events say about converting each imported gate
    (ag_review.conversion_evidence); finished(ok, message, {key: evidence})
    with the keys given. progress(done, total) reports as it goes."""

    progress = Signal(int, int)
    finished = Signal(bool, str, object)

    def __init__(self, jobs, gate_defs, data, index, result, limits, zeros, parent=None):
        super().__init__(parent)
        self._jobs = jobs            # [(key, gate definition)]
        self._defs = gate_defs
        self._data = data
        self._index = index
        self._result = result
        self._limits = limits
        self._zeros = zeros
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        out = {}
        try:
            anchor = ag_core.anchor_population(self._defs)
            for k, (key, g) in enumerate(self._jobs, 1):
                if self._abort:
                    self.finished.emit(False, "Cancelled.", {})
                    return
                if k % 25 == 0:
                    self.progress.emit(k, len(self._jobs))
                try:
                    out[key] = ag_review.conversion_evidence(g, self._defs, self._data, self._index,
                                                             self._result, self._limits, self._zeros,
                                                             anchor)
                except Exception as exc:
                    log.info("conversion evidence of %s skipped: %s", g.get('gate_name'), exc)
                    out[key] = {}
        except Exception as exc:
            log.exception("conversion evidence failed")
            self.finished.emit(False, str(exc), {})
            return
        self.finished.emit(True, '', out)


class _FlagWorker(QThread):
    """Compare every imported gate's boundary with its trainable form on the
    review events (see ag_review.conversion_check); finished(ok, message,
    {gate: flag}). progress(done, total) reports as it goes."""

    progress = Signal(int, int)
    finished = Signal(bool, str, object)

    MAX_EVENTS = 10_000

    def __init__(self, gate_defs, trained, data, pnn, limits, recommendations,
                 shift_limit: float, fraction_limit: float, zeros: dict | None = None,
                 transforms=None, parent=None):
        super().__init__(parent)
        self._zeros = dict(zeros or {})
        self._transforms = transforms
        self._defs = gate_defs
        self._trained = trained
        self._data = data
        self._pnn = list(pnn)
        self._limits = limits
        self._recs = recommendations
        self._shift = float(shift_limit)
        self._fraction = float(fraction_limit)
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            flags = self._execute()
        except Exception as exc:
            log.exception("conversion check failed")
            self.finished.emit(False, str(exc), {})
            return
        if flags is None:
            self.finished.emit(False, "Cancelled.", {})
        else:
            self.finished.emit(True, '', flags)

    def _execute(self):
        defs = self._defs
        index = {c: i for i, c in enumerate(self._pnn)}
        by_name = ag_core.gates_by_name(defs)
        result = _review_gating(defs, self._trained, self._data, index, self._limits,
                                lambda: self._abort, transforms=self._transforms)
        if self._abort:
            return None
        anchor = ag_core.anchor_population(defs)
        candidates = [g for g in ag_core.ordered_gate_defs(defs)
                      if g.get('template') and g['gate_name'] not in self._trained]
        calc = ag_core.GateCalculator(self._limits, self._transforms)
        rng = np.random.default_rng(TRAINING_SEED)
        flags = {}
        for k, g in enumerate(candidates, 1):
            if self._abort:
                return None
            if k % 25 == 0:
                self.progress.emit(k, len(candidates))
            name = g['gate_name']
            parent = result.parent_masks.get(name)
            if parent is None or not parent.any():
                continue
            events = self._data[parent]
            if len(events) > self.MAX_EVENTS:
                events = events[np.sort(rng.choice(len(events), self.MAX_EVENTS, replace=False))]
            drawn = g['template']
            try:
                drawn_masks = ag_core.population_masks(g, drawn, events, index,
                                                       axis_limits=self._limits)
            except Exception as exc:
                log.info("conversion check of %s skipped: %s", name, exc)
                continue
            fractions = ag_review.positive_fractions(
                g, {p: float(m.mean()) for p, m in drawn_masks.items()})
            imported = g.get('algorithm') == ag_core.ALGORITHM_IMPORTED
            evidence = {}
            if imported:
                try:
                    evidence = ag_review.conversion_evidence(g, defs, self._data, index, result,
                                                             self._limits, self._zeros, anchor)
                except Exception as exc:
                    log.info("conversion evidence of %s skipped: %s", name, exc)
            algo = (ag_review.convert_target(g, self._recs.get(name), fractions, evidence)[0]
                    if imported else g.get('algorithm'))
            if not algo:
                continue
            proposed_def = ag_review.proposed_definition(g, self._recs.get(name), fractions,
                                                         evidence)
            eff = ag_core.effective_gate_def(proposed_def, by_name)
            try:
                resolved, _reason = ag_core.resolve_replicates(eff, self._data, index, result,
                                                               by_name, transforms=self._transforms)
                if resolved is None:
                    continue
                proposed = calc.calculate(events, resolved, index, trained=result.boundaries,
                                          by_name=by_name)
                if not proposed:
                    continue
                proposed_masks = ag_core.population_masks(eff, proposed, events, index,
                                                          axis_limits=self._limits)
            except Exception as exc:
                log.info("conversion check of %s skipped: %s", name, exc)
                continue
            flag = ag_review.conversion_check(
                g, drawn, proposed, {p: float(m.mean()) for p, m in drawn_masks.items()},
                {p: float(m.mean()) for p, m in proposed_masks.items()},
                shift_limit=self._shift, fraction_limit=self._fraction)
            if flag is not None:
                flag['algorithm'] = algo
                flags[name] = flag
        self.progress.emit(len(candidates), len(candidates))
        return flags


class ReviewTab(QWidget):
    """Review imported gates against one or more samples, convert them to
    trainable and edit them.

    Pick a population in the gating hierarchy; for each distinct plot among
    the population's daughter gates the tab shows the gates as drawn (left)
    and, beside them, each gate's trainable form (right), calculated on the
    review samples' parent population with the algorithm chosen above it
    (or the trained boundary once the gate is trained). Each population is
    labelled with its fraction of parent, next to the fraction recorded at
    import when one review sample is the sample the counts came from; a
    difference is shown in red.

    Per gate: the algorithm of each separating axis (a 2dsep gate has two),
    with a source to copy from when an axis replicates a threshold, Convert
    to trainable (or Use shapes as drawn), Split… for 2dsep gates and a red
    − to delete the gate and the gates below it. Converting a gate brings
    the gates that share its imported name into line (see
    ag_review.apply_name_sync); ⇊ gives a converted gate's configuration to
    the same-named gates below its population, ⇉ to every same-named gate.
    A gate with replicated thresholds has a row below it showing each
    population a threshold is calculated on, with the gate drawn on it.
    Convert all converts every gate still as drawn; Update calculates the
    shown gates again on the review samples.
    With Edit gates on, the shapes as drawn of gates that apply them, and
    the boundaries of converted gates, can be moved; Reset restores the
    shapes as imported.

    The hierarchy tree colours each population by its gate's status (see
    gate_status) and takes several selected populations (Ctrl/Shift-click)
    for its right-click menu: edit a gate or its advanced parameters, copy
    gates or whole branches and paste them on other populations, delete
    gates or populations, and mark a population as a reference for
    replicated thresholds.

    Only the gates the shown plots depend on are gated, so a large
    hierarchy stays responsive.

    The hierarchy can be searched and narrowed to unconverted gates, gates
    whose same-named partners were set to different algorithms, or flagged
    gates. Flag conversions compares, for every gate with an imported
    boundary, that boundary with its trainable form on the review samples
    and flags the gates where a threshold moves, or a population changes,
    by more than the limits set beside it. Each gate's arrow button gives
    the gates of the same name below its population its configuration.
    """

    templatesChanged = Signal()
    # A gate's algorithm or conversion changed: the hierarchy must be redrawn.
    gatesChanged = Signal()
    # Split of a 2dsep gate requested: (gate name).
    splitRequested = Signal(str)
    # Edit of a gate in the gate editor requested: (gate name).
    editRequested = Signal(str)
    # Emitted when an axis drag changed a channel's transform: (channel, parameters before).
    transformChanged = Signal(str, object)
    _MAX_EVENTS = 150_000
    _MIN_EVENTS_PER_SAMPLE = 20_000
    _PROPOSAL_EVENTS = 50_000
    _REGATE_DELAY_MS = 400

    def __init__(self, state: GatingModelState, bus, controller, tf=None, parent=None):
        super().__init__(parent)
        self.state = state
        self.bus = bus
        self.controller = controller
        self.tf = tf
        self._flag_worker = None
        self._overlap_worker = None
        self._evidence_worker = None
        self._fallbacks: set = set()
        self._raw = None
        self._data_t = None
        self._loaded_keys = None
        self._loaded_dir = None
        self._wanted_keys = None
        self._sample_keys: list = []
        self._result = None
        self._gated: set = set()
        self._worker = None
        self._plots: list = []
        self._trainable: list = []
        self._references: list = []
        self._building = False
        self._mask = None
        self._data_signature = None
        self._tree_views: dict = {}
        self._tree_labels: dict = {}
        self._proposals: dict = {}
        self._evidence: dict = {}
        self._anchor = _NO_ANCHOR       # anchor population, found when first needed
        self._clipboard: list = []
        self._clipboard_branch = False
        self._regate_timer = QTimer(self)
        self._regate_timer.setSingleShot(True)
        self._regate_timer.setInterval(self._REGATE_DELAY_MS)
        self._regate_timer.timeout.connect(self._on_regate_timer)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(HelpToggleWidget(text=ag_help_texts.BUILDER_REVIEW))
        row = QHBoxLayout()
        self.combo_pop = QComboBox()
        self.combo_pop.setMinimumWidth(220)
        self.btn_samples = QToolButton()
        self.btn_samples.setPopupMode(QToolButton.InstantPopup)
        self.btn_samples.setMinimumWidth(200)
        self.btn_samples.setToolTip("Samples the gates are reviewed on. Tick several to pool "
                                    "them, each contributing the same number of events.")
        self.menu_samples = QMenu(self.btn_samples)
        self.btn_samples.setMenu(self.menu_samples)
        self.chk_edit = QCheckBox("Edit gates")
        self.chk_edit.setToolTip(
            "Drag the boundaries: on the left, the shapes as drawn of gates that still "
            "apply them; on the right, the boundaries of converted gates (kept as a "
            "manual adjustment until the gate is trained again).")
        self.btn_reset = QPushButton("Reset to import")
        self.btn_reset.setToolTip("Restore the shapes shown to their shapes as imported.")
        self.btn_update = QPushButton("Update")
        self.btn_update.setToolTip("Calculate the shown gates again on the review samples "
                                   "with their current algorithms.")
        self.btn_convert_all = QPushButton("Convert all to trainable")
        self.btn_convert_all.setToolTip(
            "Convert every gate that still applies its shapes as drawn to its proposed "
            "algorithm. Gates sharing a name with a gate you converted yourself follow "
            "your choice.")
        row.addWidget(QLabel("Population:"))
        row.addWidget(self.combo_pop)
        row.addWidget(QLabel("Review samples:"))
        row.addWidget(self.btn_samples)
        row.addWidget(self.chk_edit)
        row.addWidget(self.btn_reset)
        row.addWidget(self.btn_update)
        row.addWidget(self.btn_convert_all)
        row.addStretch()
        layout.addLayout(row)

        flag_row = QHBoxLayout()
        self.btn_flag = QPushButton("Flag conversions")
        self.btn_flag.setToolTip(
            "Compare every gate's imported boundary with its trainable form on the review "
            "samples, and flag the gates whose trainable form lands far from it. Flagged "
            "gates are red in the hierarchy and listed by Show: Flagged conversions.")
        self.btn_flag.clicked.connect(self._on_flag_clicked)
        flag_row.addWidget(self.btn_flag)
        flag_row.addWidget(QLabel("when a threshold moves more than"))
        self.sb_flag_shift = block_wheel(QDoubleSpinBox())
        self.sb_flag_shift.setRange(0.01, 1.0)
        self.sb_flag_shift.setDecimals(2)
        self.sb_flag_shift.setSingleStep(0.01)
        self.sb_flag_shift.setValue(0.10)
        self.sb_flag_shift.setToolTip("In transformed axis units (the display axis spans about 0 to 1).")
        flag_row.addWidget(self.sb_flag_shift)
        flag_row.addWidget(QLabel("or a population changes by more than"))
        self.sb_flag_fraction = block_wheel(QDoubleSpinBox())
        self.sb_flag_fraction.setRange(1.0, 100.0)
        self.sb_flag_fraction.setDecimals(0)
        self.sb_flag_fraction.setSuffix(" pp")
        self.sb_flag_fraction.setValue(10.0)
        self.sb_flag_fraction.setToolTip("Percentage points of its parent.")
        flag_row.addWidget(self.sb_flag_fraction)
        self.lbl_flag = QLabel("")
        self.lbl_flag.setStyleSheet("color: #555; font-style: italic;")
        flag_row.addWidget(self.lbl_flag, stretch=1)
        layout.addLayout(flag_row)

        overlap_row = QHBoxLayout()
        self.btn_overlap = QPushButton("Flag overlaps")
        self.btn_overlap.setToolTip(
            "Find populations defined twice: two populations, neither below the other, that "
            "hold largely the same events on the review samples. Only populations with gates "
            "drawn on them are compared, and one that is nearly the same as a population above "
            "it counts as that population. Flagged gates are orange in the hierarchy and listed "
            "by Show: Overlapping gates.")
        self.btn_overlap.clicked.connect(self._on_overlap_clicked)
        overlap_row.addWidget(self.btn_overlap)
        overlap_row.addWidget(QLabel("when two populations share at least"))
        self.sb_overlap = block_wheel(QDoubleSpinBox())
        self.sb_overlap.setRange(10.0, 100.0)
        self.sb_overlap.setDecimals(0)
        self.sb_overlap.setSuffix(" %")
        self.sb_overlap.setValue(100.0 * ag_review.OVERLAP_LIMIT)
        self.sb_overlap.setToolTip("Events in both, as a share of the events in either "
                                   "(intersection over union).")
        overlap_row.addWidget(self.sb_overlap)
        overlap_row.addWidget(QLabel("of their events"))
        self.lbl_overlap = QLabel("")
        self.lbl_overlap.setStyleSheet("color: #555; font-style: italic;")
        overlap_row.addWidget(self.lbl_overlap, stretch=1)
        layout.addLayout(overlap_row)
        self.lbl_status = QLabel("Import a gating strategy to review it here.")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #555; font-style: italic;")
        layout.addWidget(self.lbl_status)

        self._host = QWidget()
        self._grid = QGridLayout(self._host)
        self._grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        # Spare column: wide controls overflow into it, not the plot column.
        self._grid.setColumnStretch(5, 1)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._host)

        # Gating hierarchy on the left, inspection plots on the right.
        tree_panel = QWidget()
        tree_layout = QVBoxLayout(tree_panel)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        tree_header = QHBoxLayout()
        tree_header.addWidget(QLabel("<b>Gating hierarchy</b>"))
        tree_header.addStretch()
        self.btn_expand = QPushButton("Expand")
        self.btn_collapse = QPushButton("Collapse")
        self.btn_expand.setToolTip("Expand every node of the hierarchy.")
        self.btn_collapse.setToolTip("Collapse the hierarchy to the root.")
        tree_header.addWidget(self.btn_expand)
        tree_header.addWidget(self.btn_collapse)
        tree_layout.addLayout(tree_header)
        self.tree_filter = FilterBar(GATE_FILTERS, placeholder="Search populations and gates…",
                                     vertical=True)
        self.tree_filter.changed.connect(lambda _t, _m: self._apply_tree_filter())
        tree_layout.addWidget(self.tree_filter)
        self.pop_tree = PopulationTree(multi_select=True)
        self.pop_tree.setToolTip(
            "Select a population to inspect the gates drawn on it. Populations with no "
            "gates below them show the plot they are gated on. Right-click to edit, copy, "
            "paste or delete gates, or to mark a reference population; Ctrl/Shift-click "
            "selects several. Red: unconverted; amber: untrained; green: trained; "
            "⚠: flagged.")
        self.pop_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.pop_tree.customContextMenuRequested.connect(self._on_tree_menu)
        tree_layout.addWidget(self.pop_tree, stretch=1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(tree_panel)
        splitter.addWidget(self._scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        splitter.setSizes([240, 760])
        layout.addWidget(splitter, stretch=1)

        self.lbl_sync = QLabel()
        self.lbl_sync.setWordWrap(True)
        self.lbl_sync.setStyleSheet("color: #b06000; font-weight: bold;")
        self.lbl_sync.setVisible(False)
        layout.addWidget(self.lbl_sync)

        self.pop_tree.populationSelected.connect(self._on_tree_selected)
        self.gatesChanged.connect(self._mark_flags)
        self.btn_expand.clicked.connect(self.pop_tree.expandAll)
        self.btn_collapse.clicked.connect(self.pop_tree.collapseAll)
        self.combo_pop.currentIndexChanged.connect(self._on_population_changed)
        self.chk_edit.toggled.connect(self._on_edit_toggled)
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_update.clicked.connect(self._on_update)
        self.btn_convert_all.clicked.connect(self._on_convert_all)

    # -- population and sample pickers --------------------------------------

    def refresh(self):
        """Rebuild the pickers from the model; reload the samples if needed."""
        self._anchor = _NO_ANCHOR
        self._building = True
        try:
            previous = self.combo_pop.currentData()
            self.combo_pop.clear()
            for choice in ag_review.population_choices(self.state.gate_definitions):
                self.combo_pop.addItem(f"{choice['label']}  ({choice['n_gates']} gate"
                                       f"{'s' if choice['n_gates'] != 1 else ''})",
                                       _pop_key_text(choice['gate'], choice['pop']))
            if previous is not None:
                i = self.combo_pop.findData(previous)
                if i >= 0:
                    self.combo_pop.setCurrentIndex(i)
            self._fill_tree()
            self._fill_samples()
        finally:
            self._building = False
        self._update_sync_banner()
        self.btn_convert_all.setEnabled(any(
            g.get('algorithm') == ag_core.ALGORITHM_IMPORTED for g in self.state.gate_definitions))
        if self.combo_pop.count() == 0:
            self._clear_plots()
            self.lbl_status.setText("The model has no gates to review.")
            return
        self._ensure_sample()

    def _fill_tree(self):
        """Rebuild the hierarchy tree and select the population shown."""
        nodes = ag_review.population_tree(self.state.gate_definitions)
        self._tree_views = {(n['gate'], n['pop']): n['view'] for n in nodes}
        self._tree_labels = {(n['gate'], n['pop']): n['label'] for n in nodes}
        self.pop_tree.set_nodes(nodes)
        self._mark_flags()
        self._apply_tree_filter()
        self._select_in_tree()

    def _mark_flags(self):
        """Colour each population by its gate's status (see gate_status);
        populations of flagged conversions or overlapping gates start with
        ⚠ and have the flag as tooltip."""
        flags = {}
        for name, flag in self.state.overlap_flags.items():
            for pop, _other_gate, _other_pop, _share in flag.get('pairs') or []:
                flags.setdefault((name, pop), []).append(flag.get('message', ''))
        for name, flag in self.state.conversion_flags.items():
            g = self.state.gate_by_name(name)
            if g is None:
                continue
            for pop in (g.get('populations') or {}):
                flags.setdefault((name, pop), []).append(flag.get('message', ''))
        marks = {}
        for g in self.state.gate_definitions:
            colour = STATUS_COLOURS[gate_status(self.state, g)]
            for pop in (g.get('populations') or {}):
                key = (g['gate_name'], pop)
                tips = flags.get(key)
                marks[key] = (colour, "\n".join(t for t in tips if t), True) if tips else (colour, '')
        self.pop_tree.mark(marks)

    def _apply_tree_filter(self):
        """Show the populations whose gates match the search and the
        "show" choice (with the populations above them)."""
        text, mode = self.tree_filter.text(), self.tree_filter.mode()
        counts = filter_counts(self.state)
        for key, label in GATE_FILTERS[1:]:
            self.tree_filter.set_mode_label(key, f"{label} ({counts.get(key, 0)})")
        wanted = filtered_gate_names(self.state, mode)
        if wanted is None and not text:
            self.pop_tree.set_filter(None)
            return
        shown = set()
        for g in self.state.gate_definitions:
            name = g['gate_name']
            if wanted is not None and name not in wanted:
                continue
            pops = g.get('populations') or {}
            labels = [(pops.get(p) or {}).get('label', p) for p in pops]
            if not text_matches(text, name, *labels):
                continue
            parent = (g.get('parent_gate') or '',
                      (g.get('parent_popul') or 'root') if g.get('parent_gate') else 'root')
            shown.add(parent)
            shown |= {(name, p) for p in pops}
        self.pop_tree.set_filter(shown)

    def _select_in_tree(self):
        gate, pop = self._selected()
        self.pop_tree.select_population(gate, pop)

    def _on_tree_selected(self, gate: str, pop: str):
        """A node was picked in the tree: show its plots through the combobox."""
        view = self._tree_views.get((gate, pop))
        if view is None:
            return
        i = self.combo_pop.findData(_pop_key_text(view[0], view[1]))
        if i < 0:
            return
        self._building = True
        try:
            self.combo_pop.setCurrentIndex(i)
        finally:
            self._building = False
        self._rebuild_plots()

    def _import_sample_key(self):
        """The experiment sample the imported counts were recorded on, if any."""
        keys = analysis_sample_keys(self.controller, include_controls=True)
        for g in self.state.gate_definitions:
            for key in keys:
                if ag_review.matches_import_sample(g, key):
                    return key
        return None

    def _fill_samples(self):
        """Offer every analysis sample; keep the chosen ones, else start with
        the sample the import's counts came from, or the one on show."""
        keys = list(analysis_sample_keys(self.controller))
        chosen = [k for k in self._sample_keys if k in keys]
        if not chosen:
            wanted = self._import_sample_key()
            if wanted is None:
                current = getattr(self.controller, 'current_sample_path', None)
                wanted = next((k for k in keys if current and str(current).endswith(str(k))), None)
            if wanted is None and keys:
                wanted = keys[0]
            if wanted is not None and wanted not in keys:
                keys.insert(0, wanted)
            chosen = [wanted] if wanted is not None else []
        self._sample_keys = chosen
        self.menu_samples.clear()
        for key in keys:
            action = QAction(_sample_label(key), self.menu_samples)
            action.setCheckable(True)
            action.setChecked(key in chosen)
            action.toggled.connect(lambda on, k=key: self._on_sample_toggled(k, on))
            self.menu_samples.addAction(action)
        self._update_samples_button()

    def _update_samples_button(self):
        keys = self._sample_keys
        if not keys:
            self.btn_samples.setText("(no samples)")
        elif len(keys) == 1:
            self.btn_samples.setText(_sample_label(keys[0]))
        else:
            self.btn_samples.setText(f"{len(keys)} samples pooled")

    def _on_sample_toggled(self, key, on: bool):
        if on and key not in self._sample_keys:
            self._sample_keys.append(key)
        elif not on and key in self._sample_keys:
            if len(self._sample_keys) == 1:
                # At least one sample stays chosen.
                QTimer.singleShot(0, self._fill_samples)
                return
            self._sample_keys.remove(key)
        self._update_samples_button()
        if not self._building:
            self._ensure_sample()

    def _on_population_changed(self, _index=None):
        if not self._building:
            self._select_in_tree()
            self._rebuild_plots()

    def _samples_text(self) -> str:
        keys = self._loaded_keys or ()
        if len(keys) == 1:
            return _sample_label(keys[0])
        return f"{len(keys)} pooled samples"

    def _single_review_key(self):
        """The review sample when exactly one is loaded, else None."""
        keys = self._loaded_keys or ()
        return keys[0] if len(keys) == 1 else None

    # -- loading and gating -------------------------------------------------

    def _ensure_sample(self):
        keys = tuple(self._sample_keys)
        if not keys:
            self.lbl_status.setText("No sample available for review.")
            return
        experiment_dir = getattr(self.controller, 'experiment_dir', None)
        if experiment_dir != self._loaded_dir:
            self._raw = self._data_t = self._result = self._loaded_keys = None
            self._gated = set()
            self._loaded_dir = experiment_dir
        if keys == self._loaded_keys and self._raw is not None:
            if self._data_signature != self._transform_signature():
                self._transform_events()
            self._regate()
            self._rebuild_plots()
            return
        if self._worker is not None:
            self._wanted_keys = keys
            return
        if wait_if_busy(self.controller, ('Automated Gating review samples', id(self)), self._ensure_sample,
                        self.lbl_status.setText):
            return
        try:
            snap = snapshot_unmix_state(self.controller, list(keys))
        except Exception as exc:
            self.lbl_status.setText(f"Cannot load the review samples: {exc}")
            return
        label = _sample_label(keys[0]) if len(keys) == 1 else f"{len(keys)} samples"
        self.lbl_status.setText(f"Loading {label} …")
        self._wanted_keys = keys
        per_sample = max(self._MAX_EVENTS // len(keys), self._MIN_EVENTS_PER_SAMPLE)
        worker = _ReviewWorker(self.controller.experiment_dir, keys, snap, per_sample, parent=self)
        worker.finished.connect(lambda ok, msg, data, k=keys: self._on_loaded(ok, msg, data, k))
        self._worker = worker
        worker.start()

    def _on_loaded(self, ok, message, data, keys):
        self._worker = None
        if not ok:
            self.lbl_status.setText(f"Could not load the review samples: {message}")
            return
        self._raw = data
        self._loaded_keys = tuple(keys)
        self._transform_events()
        if self._wanted_keys not in (None, tuple(keys)):
            self._ensure_sample()
            return
        self._regate()
        self._rebuild_plots()

    def _transform_signature(self):
        """The transforms the events are shown in, as comparable values."""
        live = self.tf.objects
        return tuple(sorted(
            (ch, tuple(sorted((k, str(v)) for k, v in ag_core.transform_params(tr).items()
                              if k != 'limits')))
            for ch, tr in live.items()))

    def _transform_events(self):
        """Transform every column of the loaded events with the live transforms."""
        pnn = _experiment_channels(self.controller)
        self._data_t = ag_core.transform_columns(self._raw, pnn, self.tf.objects)
        self._data_signature = self._transform_signature()

    def _retransform_channel(self, channel: str):
        """Transform one column again after its transform changed."""
        pnn = _experiment_channels(self.controller)
        tr = self.tf.objects.get(channel)
        xform = getattr(tr, 'xform', None)
        if self._raw is None or self._data_t is None or xform is None or channel not in pnn:
            return
        col = pnn.index(channel)
        self._data_t[:, col] = np.asarray(
            xform.apply(self._raw[:, col].astype(np.float64)), dtype=np.float32)
        self._data_signature = self._transform_signature()

    def _axis_limits(self) -> dict:
        return _axis_limits(self.tf.objects)

    def _needed_gates(self) -> set:
        """The gates the shown population's plots depend on: its ancestors,
        its daughter gates and everything their calculation needs (replicate
        sources and reference populations, with their own ancestors)."""
        by_name = ag_core.gates_by_name(self.state.gate_definitions)
        gate, pop = self._selected()
        start = {g['gate_name'] for g in self.state.gate_definitions
                 if (g.get('parent_gate') or None) == (gate or None)
                 and (gate is None or (g.get('parent_popul') or 'root') == pop)}
        if gate:
            start.add(gate)
        needed, stack = set(), list(start)
        while stack:
            name = stack.pop()
            if name in needed or name not in by_name:
                continue
            needed.add(name)
            g = by_name[name]
            deps = ag_core.gate_dependencies(g)
            if g.get('algorithm') == ag_core.ALGORITHM_IMPORTED:
                deps |= ag_core.gate_dependencies(ag_review.trainable_definition(g))
            stack.extend(d for d in deps if d not in needed)
        return needed

    def _regate(self, names: set | None = None):
        """Gate the review events for the gates in *names* (default: those
        the shown plots need): trained gates and gates applying their shapes
        as drawn use those boundaries; gates converted to a learning
        algorithm but not yet trained are calculated on these events."""
        self._proposals = {}
        self._evidence = {}
        self._anchor = _NO_ANCHOR
        if self._data_t is None:
            return
        names = self._needed_gates() if names is None else set(names)
        subset = [g for g in self.state.gate_definitions if g['gate_name'] in names]
        pnn = _experiment_channels(self.controller)
        reference = ag_review.reference_boundaries(subset, self.state.trained_boundaries)
        modes = {}
        for g in subset:
            name = g['gate_name']
            if self.state.is_trained(name) or g.get('algorithm') == ag_core.ALGORITHM_IMPORTED:
                continue
            # The imported boundary stands in while the gate cannot be
            # calculated (e.g. a replicate source not yet chosen), so the
            # gates below it keep their events.
            if not g.get('template'):
                reference.pop(name, None)
            modes[name] = ag_core.MODE_RECALCULATE
        self._result = ag_core.run_gating(
            subset, self._data_t, {c: i for i, c in enumerate(pnn)},
            reference=reference, modes=modes, default_mode=ag_core.MODE_FIXED,
            axis_limits=self._axis_limits(), transforms=self.tf.objects)
        self._gated = names

    def _ensure_gated(self):
        if self._data_t is not None and not self._needed_gates() <= self._gated:
            self._regate()

    def _on_regate_timer(self):
        """Gate again once an axis drag has stopped."""
        self._regate()
        self._redraw_overlays()

    # -- plots --------------------------------------------------------------

    def _clear_plots(self):
        for plot in self._plots:
            plot.setParent(None)
            plot.deleteLater()
        self._plots = []
        self._trainable = []
        self._references = []

    def _selected(self):
        return _pop_key_tuple(self.combo_pop.currentData())

    def _rebuild_plots(self):
        """Lay out the shown population's plots: per parameter set, the gates
        as drawn on the left and each gate's controls above its trainable
        form on the right, so the two plots of a gate sit level. The scroll
        position is kept."""
        import honeychrome.settings as hc_settings
        from PySide6.QtWidgets import QApplication
        bar = self._scroll.verticalScrollBar()
        scroll_to = bar.value()
        # Hiding the focused widget (the button just clicked) passes focus to
        # the next one, and the scroll area scrolls to show it: clear the
        # focus first so the view stays where it is.
        focused = QApplication.focusWidget()
        if focused is not None and self._host.isAncestorOf(focused):
            focused.clearFocus()
        self._clear_plots()
        if self._raw is None:
            return
        self._ensure_gated()
        if self._result is None:
            return
        gate, pop = self._selected()
        groups = ag_review.daughter_groups(self.state.gate_definitions, gate, pop)
        if gate is None:
            mask = np.ones(len(self._raw), dtype=bool)
        else:
            mask = self._result.population_mask(gate, pop)
        if mask is None:
            self.lbl_status.setText(
                f"The parent population {gate}/{pop} could not be gated on these samples "
                "(its gate has no boundary or its channels are missing).")
            return
        self._mask = mask
        tile = hc_settings.cytometry_plot_width_target_retrieved
        self._trainable = []
        r = 0
        for group in groups:
            drawn = any(ag_review.has_drawn_shape(g) for g in group['gates'])
            left = _ReviewPlot(group['channels'], group['gates'], self.controller, tile, self,
                               caption="As drawn" if drawn else "Current",
                               transforms=self.tf.objects)
            left.gateEdited.connect(self._on_gate_edited)
            left.axisZoomed.connect(self._on_axis_zoomed)
            self._connect_open(left)
            left.set_events(self._raw, mask)
            for k, g in enumerate(group['gates']):
                controls = self._controls(g)
                self._grid.addWidget(controls, r, 1, 1, 5, Qt.AlignLeft | Qt.AlignBottom)
                self._plots.append(controls)
                plot = self._trainable_plot(g, tile)
                self._grid.addWidget(plot, r + 1, 1, Qt.AlignTop)
                self._plots.append(plot)
                minus = self._delete_button(g)
                self._grid.addWidget(minus, r + 1, 2, Qt.AlignTop)
                self._plots.append(minus)
                forward = self._feed_forward_button(g)
                self._grid.addWidget(forward, r + 1, 3, Qt.AlignTop)
                self._plots.append(forward)
                same = self._apply_named_button(g)
                self._grid.addWidget(same, r + 1, 4, Qt.AlignTop)
                self._plots.append(same)
                if k == 0:
                    self._grid.addWidget(left, r + 1, 0, Qt.AlignTop)
                    self._plots.append(left)
                r += 2
                refs = self._reference_plots(g, tile)
                if refs is not None:
                    self._grid.addWidget(refs, r, 1, 1, 5, Qt.AlignLeft | Qt.AlignTop)
                    self._plots.append(refs)
                    r += 1
        self._redraw_overlays()
        label = 'All events' if gate is None else f"{pop}"
        text = (f"{label}: {int(mask.sum()):,} of {len(self._raw):,} events on "
                f"{self._samples_text()}; {len(groups)} parameter set"
                f"{'s' if len(groups) != 1 else ''}. Left: gates as drawn; right: their "
                "trainable form.")
        node = self.pop_tree.current_population()
        if node is not None and self._tree_views.get(node) not in (None, node):
            text = f"{self._tree_labels.get(node, node[1])} is gated on this plot. " + text
        if self.chk_edit.isChecked():
            text += " " + self._edit_hint(groups)
        self.lbl_status.setText(text)
        QTimer.singleShot(0, lambda v=scroll_to: bar.setValue(v))

    def _connect_open(self, plot):
        """Let a plot's right-click open the population clicked."""
        plot.can_open = lambda gate, pop: self._tree_views.get((gate, pop)) == (gate, pop)
        plot.populationRequested.connect(self._open_population)

    def _open_population(self, gate: str, pop: str):
        """Show a population's daughter gates, as picking it in the tree does."""
        if self._tree_views.get((gate, pop)) != (gate, pop):
            return
        self.pop_tree.select_population(gate, pop)
        self._on_tree_selected(gate, pop)

    def _edit_hint(self, groups) -> str:
        left = sum(1 for grp in groups for g in grp['gates']
                   if g.get('template') and g.get('algorithm') == ag_core.ALGORITHM_IMPORTED)
        right = sum(1 for grp in groups for g in grp['gates']
                    if g.get('algorithm') != ag_core.ALGORITHM_IMPORTED)
        return (f"Editing: {left} gate(s) movable as drawn (left), {right} converted gate(s) "
                "movable on the right.")

    def _trainable_algorithm(self, g: dict) -> str | None:
        """The learning algorithm a gate uses, or converts to while imported."""
        if g.get('algorithm') != ag_core.ALGORITHM_IMPORTED:
            return g.get('algorithm')
        return ag_review.convert_target(
            g, self.state.algorithm_recommendations.get(g['gate_name']),
            positive_fraction=self._positive_fractions(g),
            evidence=self._gate_evidence(g))[0]

    def _trainable_axes(self, g: dict) -> dict:
        """{axis: algorithm} of a threshold gate, or what it converts to
        while imported (see ag_review.proposed_axis_algorithms)."""
        return ag_review.proposed_axis_algorithms(
            g, self.state.algorithm_recommendations.get(g['gate_name']),
            positive_fraction=self._positive_fractions(g), evidence=self._gate_evidence(g))

    def _proposed_def(self, g: dict) -> dict:
        """The trainable gate *g* is, or converts to while imported (see
        ag_review.proposed_definition)."""
        if g.get('algorithm') != ag_core.ALGORITHM_IMPORTED:
            return g
        return ag_review.proposed_definition(
            g, self.state.algorithm_recommendations.get(g['gate_name']),
            positive_fraction=self._positive_fractions(g), evidence=self._gate_evidence(g))

    def _zeros(self) -> dict:
        """{channel: raw zero in the builder's display units}."""
        out = {}
        for ch, tr in (self.tf.objects if self.tf is not None else {}).items():
            xform = getattr(tr, 'xform', None)
            if xform is None:
                continue
            try:
                out[ch] = float(np.asarray(xform.apply(np.array([0.0])))[0])
            except Exception:
                continue
        return out

    @staticmethod
    def _evidence_key(g: dict) -> tuple:
        return (g['gate_name'], ag_review.sync_fingerprint(g), g.get('gate_type'),
                json.dumps(g.get('populations'), sort_keys=True, default=str))

    def _gate_evidence(self, g: dict) -> dict:
        """What the review events say about converting an imported gate
        (ag_review.conversion_evidence), cached until the events are gated
        again; {} without review events or for converted gates."""
        if g.get('algorithm') != ag_core.ALGORITHM_IMPORTED or self._result is None \
                or self._data_t is None or g['gate_name'] not in self._result.parent_masks:
            return {}
        key = self._evidence_key(g)
        if key not in self._evidence:
            pnn = _experiment_channels(self.controller)
            try:
                if self._anchor is _NO_ANCHOR:
                    self._anchor = ag_core.anchor_population(self.state.gate_definitions)
                self._evidence[key] = ag_review.conversion_evidence(
                    g, self.state.gate_definitions, self._data_t,
                    {c: i for i, c in enumerate(pnn)}, self._result, self._axis_limits(),
                    self._zeros(), self._anchor)
            except Exception:
                log.exception("conversion evidence for %s failed", g['gate_name'])
                self._evidence[key] = {}
        return self._evidence[key]

    def _positive_fractions(self, g: dict) -> dict | None:
        """{axis: fraction of the parent on the positive side} of a gate's
        shapes as drawn on the review events, for a gate whose proposal
        follows the import's rule; None otherwise."""
        if g.get('proposal_rule') not in (ag_review.RULE_HISTOGRAM, ag_review.RULE_FRACTION) \
                or self._result is None or self._data_t is None:
            return None
        if g['gate_name'] not in self._result.parent_masks:
            return None
        boundary = self._drawn_boundary(g)
        fractions = {p: e.get('fraction') for p, e in boundary.items()}
        return ag_review.positive_fractions(g, fractions) or None

    def _axis_label(self, channel) -> str:
        return _channel_to_antigen(self.controller, channel) if channel else ''

    def _controls(self, g: dict) -> QWidget:
        """The gate's name and type, then its controls, above one gate's
        trainable form."""
        name = g['gate_name']
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        imported = g.get('algorithm') == ag_core.ALGORITHM_IMPORTED
        pd = self._proposed_def(g)
        evidence = self._gate_evidence(g) if imported else {}
        column.addWidget(QLabel(f"<b>{name}</b> → {pd.get('gate_type', '')}"))
        for note in self._conversion_notes(g, pd, evidence):
            lbl = QLabel(f"<small>{note}</small>")
            lbl.setWordWrap(True)
            column.addWidget(lbl)
        warning = self._reference_warning(g, pd, evidence)
        if warning is not None:
            column.addWidget(warning)
        row = QHBoxLayout()
        column.addLayout(row)
        algos = ag_review.trainable_algorithms(pd)
        trainable = self._trainable_algorithm(g)
        axes = ag_core.threshold_axes(pd) if pd.get('gate_type') in ag_core.THRESHOLD_TYPES else ('x',)
        per_axis = self._trainable_axes(g)
        reasons = ag_review.evidence_axes(evidence) if evidence and g.get('proposal_rule') else {}
        for axis in axes:
            if len(axes) > 1:
                row.addWidget(QLabel(f"{self._axis_label(ag_core.axis_channel(pd, axis))}:"))
            current = per_axis.get(axis) or trainable
            combo = algorithm_combo(algos, current, ag_core.algorithm_colour)
            combo.setToolTip("Algorithm that calculates this axis from the training samples; "
                             "'replicate' takes the threshold from another gate or a reference "
                             "population. The plot below shows it on the review samples, its "
                             "threshold drawn in the algorithm's colour.")
            if axis in reasons and reasons[axis][0] == current:
                combo.setToolTip(f"Proposed from the review samples: {reasons[axis][1]}.\n\n"
                                 + combo.toolTip())
            combo.currentTextChanged.connect(
                lambda algo, n=name, a=axis: self._on_axis_algorithm(n, a, algo))
            row.addWidget(combo)
            if current == 'tail':
                self._add_tail_fraction(row, pd, axis)
            if current == 'mixture':
                self._add_mixture_controls(row, pd, axis)
            if current == ag_core.ALGORITHM_REPLICATE:
                self._add_source_controls(row, pd, axis)
            if current == 'cells':
                self._add_edge_density(row, pd)
            if current in ag_core.THRESHOLD_ALGORITHMS:
                self._add_threshold_factor(row, pd, axis)
        if imported:
            btn = QPushButton("Convert to trainable")
            btn.setToolTip("Train this gate with the chosen algorithm instead of applying the "
                           "shapes as drawn. Gates with the same imported name that are still "
                           "as drawn are converted the same way.")
            btn.clicked.connect(lambda _c=False, n=name: self._convert(n))
            row.addWidget(btn)
        elif ag_review.has_drawn_shape(g):
            btn = QPushButton("Use shapes as drawn")
            btn.setToolTip("Apply the imported shapes again instead of training this gate.")
            btn.clicked.connect(lambda _c=False, n=name: self._use_drawn(n))
            row.addWidget(btn)
        if g.get('gate_type') == '2dsep':
            btn = QPushButton("Split…")
            btn.setToolTip("Replace this gate with sequential 1dsep gates.")
            btn.clicked.connect(lambda _c=False, n=name: self.splitRequested.emit(n))
            row.addWidget(btn)
        members = ag_review.sync_members(self.state.gate_definitions, name)
        if members:
            tag = QLabel(f"<small>{len(members)} gates named "
                         f"'{ag_review.name_group(g)}'</small>")
            tag.setToolTip("Gates sharing this imported name follow the algorithm you choose "
                           "for any of them, unless you set them differently.")
            row.addWidget(tag)
        row.addStretch()
        return holder

    def _conversion_notes(self, g: dict, pd: dict, evidence: dict) -> list[str]:
        """Lines under a gate's title saying how the review samples changed
        what an imported gate converts to."""
        notes = []
        if g.get('algorithm') != ag_core.ALGORITHM_IMPORTED or not evidence:
            return notes
        drop = evidence.get('drop_axis')
        if drop:
            ch = self._axis_label(ag_core.axis_channel(g, drop))
            keep = self._axis_label(pd.get('gate_marker_x'))
            notes.append(f"Converts to 1dsep on {keep}: the shape's {ch} edges leave out at most "
                         f"{100 * ag_review.SEPARATES_MIN:.0f}% of the events in its {keep} band.")
        if evidence.get('split'):
            first, second = evidence['split']
            notes.append(f"Convert splits this gate into sequential 1dsep gates: "
                         f"{self._axis_label(first['channel'])}, then "
                         f"{self._axis_label(second['channel'])} on one side.")
        side = (pd.get('gate_param') or {}).get('cell_side')
        if side == ag_core.CELL_SIDE_DEBRIS:
            notes.append("Keeps the events below the debris cut: the shape as drawn lies there.")
        return notes

    def _reference_warning(self, g: dict, pd: dict, evidence: dict):
        """Red line asking for a reference population when a replicated
        threshold uses the default one (the parent population's parent), with
        a button to keep it; None otherwise."""
        axes = ag_review.default_source_axes(pd)
        if not axes:
            return None
        src = (pd.get('gate_param') or {}).get(f'replicate_{axes[0]}') or {}
        ref_gate, ref_pop = ag_core.split_population_key(src.get('population') or '')
        parent = g.get('parent_popul') or 'root'
        why = ''
        b0 = (evidence.get('below_zero') or {}).get('x')
        pos = (evidence.get('positive') or {}).get('x')
        if b0 is not None and pos is not None:
            why = (f" {100 * pos:.0f}% of '{parent}' is drawn positive and {100 * b0:.1f}% lies "
                   "below zero, so its own events cannot place the threshold.")
        box = QWidget()
        line = QHBoxLayout(box)
        line.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(f"<span style='color:#c00000'>Set the reference population: the threshold "
                     f"is calculated on '{ref_pop}', the anchor population.{why}</span>")
        lbl.setWordWrap(True)
        line.addWidget(lbl, 1)
        btn = QPushButton(f"Keep '{ref_pop}'")
        btn.setToolTip("Use this population and remove the warning.")
        btn.clicked.connect(lambda _c=False, n=g['gate_name']: self._confirm_reference(n))
        line.addWidget(btn)
        return box

    def _confirm_reference(self, gate_name: str):
        g = self.state.gate_by_name(gate_name)
        if g is None:
            return
        self._adopt_proposal_sources(g)
        for axis in ('x', 'y'):
            src = (g.get('gate_param') or {}).get(f'replicate_{axis}')
            if isinstance(src, dict):
                src.pop('default', None)
        self._after_config_change(g)

    def _adopt_layout(self, g: dict):
        """Give an imported gate the layout it converts to (a 1dsep gate when
        the review samples drop an axis), so per-axis settings land on the
        right axis. Its shapes as drawn still apply."""
        if g.get('algorithm') != ag_core.ALGORITHM_IMPORTED:
            return
        base = self._gate_evidence(g).get('definition')
        if not base or base.get('gate_type') == g.get('gate_type'):
            return
        for key in ('gate_type', 'gate_marker_x', 'gate_marker_y', 'sep_axis', 'populations',
                    'gate_param'):
            g[key] = deepcopy(base.get(key))
        self._evidence = {}
        self._proposals = {}

    def _adopt_proposal_sources(self, g: dict):
        """Store the default reference population an imported gate's
        proposal uses, so it can be confirmed or changed."""
        self._adopt_layout(g)
        pd = self._proposed_def(g)
        gp = g.setdefault('gate_param', {})
        for axis in ('x', 'y'):
            src = (pd.get('gate_param') or {}).get(f'replicate_{axis}')
            if isinstance(src, dict) and not isinstance(gp.get(f'replicate_{axis}'), dict):
                gp[f'replicate_{axis}'] = deepcopy(src)

    def _add_edge_density(self, row, g: dict):
        """Density fall-off of a 'cells' gate on each axis: a side is placed
        where the density falls below this fraction of the peak."""
        gp = g.get('gate_param') or {}
        for axis in ('x', 'y'):
            ch = g.get('gate_marker_x' if axis == 'x' else 'gate_marker_y')
            spin = block_wheel(DecadeSpinBox())
            spin.setKeyboardTracking(False)
            spin.setValue(float(gp.get(f'edge_density_{axis}',
                                       gp.get('edge_density', ag_core.DEFAULT_EDGE_DENSITY))))
            spin.setToolTip(f"Density fall-off on {ch}: the rectangle's {ch} sides (other than the "
                            "debris cut) are the outermost points where the density is still this "
                            f"fraction of its peak (default {ag_core.DEFAULT_EDGE_DENSITY:g} = "
                            f"{100 * ag_core.DEFAULT_EDGE_DENSITY:g}%). Smaller reaches further "
                            "out. The arrows step by factors of ten; type any other value.")
            spin.valueChanged.connect(
                lambda v, n=g['gate_name'], a=axis: self._on_edge_density(n, a, v))
            row.addWidget(QLabel(f"Fall-off {self._axis_label(ch)}:"))
            row.addWidget(spin)

    def _on_edge_density(self, gate_name: str, axis: str, value: float):
        g = self.state.gate_by_name(gate_name)
        if g is None:
            return
        g.setdefault('gate_param', {})[f'edge_density_{axis}'] = float(value)
        self._after_config_change(g)

    def _add_source_controls(self, row, g: dict, axis: str):
        """Source combo (and, for a reference population, its algorithm) of a
        replicated axis."""
        name = g['gate_name']
        gp = g.get('gate_param') or {}
        current = gp.get(f'replicate_{axis}') if isinstance(gp.get(f'replicate_{axis}'), dict) else None
        if self._anchor is _NO_ANCHOR:
            self._anchor = ag_core.anchor_population(self.state.gate_definitions)
        choices = ag_review.replicate_choices(self.state.gate_definitions, name, axis, self._anchor)
        source = block_wheel(QComboBox())
        source.setMinimumWidth(200)
        if not choices:
            source.addItem("(no gate on this channel or reference population)", None)
        else:
            source.addItem("— choose a source —", None)
        selected = 0
        for k, (label, src) in enumerate(choices, start=1):
            source.addItem(label, json.dumps(src, sort_keys=True))
            if current and (src.get('gate') == current.get('gate')
                            and src.get('population') == current.get('population')):
                selected = k
        source.setCurrentIndex(selected)
        source.setToolTip("Copy the threshold another gate has on this channel, or calculate "
                          "it on a reference population (mark one in the tree's right-click "
                          "menu).")
        source.currentIndexChanged.connect(
            lambda _i, n=name, a=axis, c=source: self._on_source(n, a, c.currentData()))
        row.addWidget(source)
        if current and current.get('population'):
            algo = algorithm_combo(list(ag_core.THRESHOLD_ALGORITHMS),
                                   current.get('algorithm') or 'tail', ag_core.algorithm_colour)
            algo.setToolTip("Algorithm that calculates the threshold on the reference population.")
            algo.currentTextChanged.connect(
                lambda a, n=name, ax=axis: self._on_source_algorithm(n, ax, a))
            row.addWidget(algo)
            if (current.get('algorithm') or 'tail') == 'tail':
                self._add_tail_fraction(row, g, axis)
            self._add_threshold_factor(row, g, axis)

    def _add_mixture_controls(self, row, g: dict, axis: str):
        """Components and cut of a 'mixture' axis: the threshold is where two
        of the Gaussian components cross, ordered by mean: the lowest and
        highest, the lowest two, or the highest two."""
        gp = g.get('gate_param') or {}
        suffix = f'_{axis}' if g.get('gate_type') == '2dsep' else ''
        spin = block_wheel(QSpinBox())
        spin.setRange(2, 6)
        spin.setKeyboardTracking(False)
        spin.setValue(int(gp.get(f'mixture_components{suffix}', gp.get('mixture_components', 2))))
        spin.setToolTip("Gaussian components fitted on this axis. With three, the cut chooses "
                        "which two cross at the threshold (e.g. a dim and a bright positive "
                        "population above the negative one).")
        cut = block_wheel(QComboBox())
        for value, text in (('outer', 'lowest | highest'), ('lower', 'lowest two'),
                            ('upper', 'highest two')):
            cut.addItem(text, value)
        cut.setCurrentIndex(max(cut.findData(gp.get(f'mixture_cut{suffix}',
                                                     gp.get('mixture_cut', 'outer'))), 0))
        cut.setEnabled(spin.value() > 2)
        cut.setToolTip("Which two components, ordered by mean, cross at the threshold.")
        name = g['gate_name']
        spin.valueChanged.connect(
            lambda v, n=name, k=f'mixture_components{suffix}': self._on_mixture_param(n, k, int(v)))
        cut.currentIndexChanged.connect(
            lambda _i, n=name, k=f'mixture_cut{suffix}', c=cut: self._on_mixture_param(
                n, k, c.currentData()))
        row.addWidget(QLabel("Components:"))
        row.addWidget(spin)
        row.addWidget(QLabel("Cut:"))
        row.addWidget(cut)

    def _on_mixture_param(self, gate_name: str, key: str, value):
        g = self.state.gate_by_name(gate_name)
        if g is None:
            return
        self._adopt_layout(g)
        gp = g.setdefault('gate_param', {})
        gp[key] = value
        # Components and cut are set together once either is chosen.
        pd = self._proposed_def(g)
        partner = key.replace('mixture_components', 'mixture_cut') if 'components' in key \
            else key.replace('mixture_cut', 'mixture_components')
        if partner not in gp and partner in (pd.get('gate_param') or {}):
            gp[partner] = pd['gate_param'][partner]
        self._after_config_change(g)

    def _add_tail_fraction(self, row, g: dict, axis: str):
        """Tail fraction of a 'tail' axis: the share of the mirrored negative
        population left above the threshold (smaller is stricter)."""
        gp = g.get('gate_param') or {}
        key = f'tail_fraction_{axis}' if g.get('gate_type') == '2dsep' else 'tail_fraction'
        spin = block_wheel(DecadeSpinBox())
        spin.setKeyboardTracking(False)
        spin.setValue(float(gp.get(key, gp.get('tail_fraction', ag_core.DEFAULT_TAIL_FRACTION))))
        spin.setToolTip("Tail fraction: the negative population's events below the mode are "
                        "mirrored about it, and the threshold is placed where this fraction of "
                        "that symmetric population lies above it (0.0001 = 0.01%). Smaller is "
                        "stricter. The arrows step by factors of ten; type any other value.")
        spin.valueChanged.connect(lambda v, n=g['gate_name'], k=key: self._on_tail_fraction(n, k, v))
        row.addWidget(QLabel("Tail fraction:"))
        row.addWidget(spin)

    def _on_tail_fraction(self, gate_name: str, key: str, value: float):
        g = self.state.gate_by_name(gate_name)
        if g is None:
            return
        self._adopt_layout(g)
        g.setdefault('gate_param', {})[key] = float(value)
        self._after_config_change(g)

    def _add_threshold_factor(self, row, g: dict, axis: str):
        """Factor multiplying an axis's calculated threshold in raw units."""
        suffix = f'_{axis}' if g.get('gate_type') == '2dsep' else ''
        spin = block_wheel(_threshold_factor_spin(
            ag_core.threshold_factor(g.get('gate_param'), suffix)))
        spin.setKeyboardTracking(False)
        spin.valueChanged.connect(
            lambda v, n=g['gate_name'], k=f'threshold_factor{suffix}':
            self._on_threshold_factor(n, k, v))
        row.addWidget(QLabel("Factor:"))
        row.addWidget(spin)

    def _on_threshold_factor(self, gate_name: str, key: str, value: float):
        g = self.state.gate_by_name(gate_name)
        if g is None:
            return
        self._adopt_layout(g)
        gp = g.setdefault('gate_param', {})
        if key != 'threshold_factor' and 'threshold_factor' in gp:
            other = 'threshold_factor_y' if key.endswith('_x') else 'threshold_factor_x'
            gp.setdefault(other, gp.pop('threshold_factor'))
        _set_threshold_factor(gp, key, value)
        self._after_config_change(g)

    def _feed_forward_button(self, g: dict) -> QWidget:
        """Give the same-named gates below this gate's population its
        configuration (see ag_review.feed_forward)."""
        btn = QPushButton("⇊")
        btn.setFixedWidth(28)
        btn.setStyleSheet("QPushButton { font-weight: bold; font-size: 14pt; }")
        btn.setFocusPolicy(Qt.NoFocus)
        converted = g.get('algorithm') != ag_core.ALGORITHM_IMPORTED
        btn.setEnabled(converted)
        btn.setToolTip(
            f"Apply '{g['gate_name']}''s algorithm and settings to every gate of the same name "
            "below the population it is drawn on." if converted else
            "Convert this gate to trainable first; its settings can then be applied to the gates "
            "of the same name below its population.")
        btn.clicked.connect(lambda _c=False, n=g['gate_name']: self._feed_forward(n))
        return btn

    def _apply_named_button(self, g: dict) -> QWidget:
        """Give every gate of the same name, type and channels in the model
        this gate's configuration (see ag_review.feed_forward)."""
        btn = QPushButton("⇉")
        btn.setFixedWidth(28)
        btn.setStyleSheet("QPushButton { font-weight: bold; font-size: 14pt; }")
        btn.setFocusPolicy(Qt.NoFocus)
        converted = g.get('algorithm') != ag_core.ALGORITHM_IMPORTED
        others = len(ag_review.sync_members(self.state.gate_definitions, g['gate_name'])) - 1
        btn.setEnabled(converted and others > 0)
        if not converted:
            tip = ("Convert this gate to trainable first; its settings can then be applied to "
                   "every gate of the same name.")
        elif others <= 0:
            tip = f"No other gate is named '{ag_review.name_group(g)}' on the same channels."
        else:
            tip = (f"Apply '{g['gate_name']}''s algorithm and settings to the {others} other "
                   f"gate(s) named '{ag_review.name_group(g)}' on the same channels, anywhere "
                   "in the model.")
        btn.setToolTip(tip)
        btn.clicked.connect(
            lambda _c=False, n=g['gate_name']: self._feed_forward(n, everywhere=True))
        return btn

    def _feed_forward(self, gate_name: str, everywhere: bool = False):
        g = self.state.gate_by_name(gate_name)
        if g is None:
            return
        if everywhere:
            scope = {o['gate_name'] for o in self.state.gate_definitions}
            title, where = "Apply to all of this name", "in the model"
        else:
            scope = ag_review.gates_below_population(
                self.state.gate_definitions, g.get('parent_gate'), g.get('parent_popul') or 'root')
            title, where = "Apply downstream", f"below '{g.get('parent_popul') or 'root'}'"
        key = (ag_review.name_group(g), g.get('gate_type'), g.get('gate_marker_x'),
               g.get('gate_marker_y'))
        targets = [o['gate_name'] for o in self.state.gate_definitions
                   if o['gate_name'] in scope and o['gate_name'] != gate_name
                   and (ag_review.name_group(o), o.get('gate_type'), o.get('gate_marker_x'),
                        o.get('gate_marker_y')) == key]
        if not targets:
            QMessageBox.information(self, title,
                                    f"No other gate named '{ag_review.name_group(g)}' lies {where}.")
            return
        if not self._confirm(title,
                             f"Give {len(targets)} gate(s) named '{ag_review.name_group(g)}' {where} "
                             f"the algorithm and settings of '{gate_name}'? Their training is "
                             "cleared."):
            return
        saved = deepcopy(self.state.gate_definitions)
        changed, skipped = ag_review.feed_forward(self.state.gate_definitions, gate_name,
                                                  everywhere=everywhere)
        try:
            ag_core.ordered_gate_defs(self.state.gate_definitions)
        except ValueError:
            self.state.gate_definitions[:] = saved
            QMessageBox.warning(self, title, "These settings would make the gates' replicate "
                                             "sources depend on each other, so nothing was "
                                             "changed.")
            return
        self._untrain(changed)
        msg = f"[Automated Gating] Applied '{gate_name}''s settings to {len(changed)} gate(s) {where}."
        if skipped:
            msg += (f" {len(skipped)} were left alone because they would replicate from "
                    "themselves or a gate below them.")
        self.bus.statusMessage.emit(msg)
        self._update_sync_banner()
        self._after_change()

    def _delete_button(self, g: dict) -> QWidget:
        btn = QPushButton("−")
        btn.setFixedWidth(28)
        btn.setStyleSheet("QPushButton { color: #c00000; font-weight: bold; font-size: 16pt; }")
        btn.setToolTip(f"Delete '{g['gate_name']}' and the gates below it.")
        btn.setFocusPolicy(Qt.NoFocus)
        btn.clicked.connect(lambda _c=False, n=g['gate_name']: self._delete_gates([n]))
        return btn

    def _trainable_plot(self, g: dict, tile: int) -> QWidget:
        """Plot of one gate's trainable form."""
        name = g['gate_name']
        imported = g.get('algorithm') == ag_core.ALGORITHM_IMPORTED
        chans = [c for c in ag_core.plot_channels(g) if c]
        trained = (not imported) and self.state.is_trained(name)
        adjusted = trained and any((e or {}).get('adjusted') for e in
                                   (self.state.trained_boundaries.get(name) or {}).values())
        caption = ("Adjusted" if adjusted else "Trained") if trained else (
            f"Calculated on {self._samples_text()}")
        plot = _ReviewPlot(chans, [g], self.controller, tile, self, caption=caption,
                           transforms=self.tf.objects)
        plot._base_caption = caption
        plot.axisZoomed.connect(self._on_axis_zoomed)
        plot.gateEdited.connect(self._on_trainable_edited)
        self._connect_open(plot)
        plot.set_events(self._raw, self._mask)
        self._trainable.append((name, plot))
        return plot

    def _reference_sources(self, g: dict) -> list[tuple]:
        """[(source key, label)] of the populations a gate's replicated
        thresholds are calculated on: a reference population, or the parent
        of the gate whose threshold is copied; one per distinct source."""
        out = []
        for src in ag_core.replicate_sources(self._proposed_def(g)).values():
            if not src:
                continue
            if src.get('gate'):
                key = ('gate', src['gate'])
                origin = self.state.gate_by_name(src['gate']) or {}
                label = f"{origin.get('parent_popul') or 'root'} (threshold of '{src['gate']}')"
            else:
                key = ('population', src['population'])
                gate, pop = ag_core.split_population_key(src['population'])
                meta = ((self.state.gate_by_name(gate) or {}).get('populations') or {}).get(pop)
                label = (meta or {}).get('label') or pop
            if key not in [k for k, _l in out]:
                out.append((key, label))
        return out

    def _reference_mask(self, key):
        """Rows of the review events in a replicate source's population, or None."""
        if self._result is None:
            return None
        kind, name = key
        if kind == 'gate':
            return self._result.parent_masks.get(name)
        return self._result.population_mask(*ag_core.split_population_key(name))

    def _reference_plots(self, g: dict, tile: int) -> QWidget | None:
        """Below a replicating gate's trainable form: each population its
        replicated thresholds are calculated on, with the gate drawn on it.
        None when the gate replicates nothing or no source is gated."""
        refs = [(key, label, self._reference_mask(key))
                for key, label in self._reference_sources(g)]
        refs = [r for r in refs if r[2] is not None]
        if not refs:
            return None
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        chans = [c for c in ag_core.plot_channels(g) if c]
        for key, label, mask in refs:
            plot = _ReviewPlot(chans, [g], self.controller, tile, self,
                               caption=f"Reference: {label}", transforms=self.tf.objects)
            plot.axisZoomed.connect(self._on_axis_zoomed)
            self._connect_open(plot)
            plot.set_events(self._raw, mask)
            row.addWidget(plot)
            self._references.append((g['gate_name'], key, plot))
        return holder

    def _proposal(self, g: dict) -> dict:
        """The gate's trainable boundary on the review events' parent
        population: the trained boundary once trained, else calculated with
        its algorithm (cached until the events are gated again)."""
        name = g['gate_name']
        algo = self._trainable_algorithm(g)
        pd = self._proposed_def(g)
        if g.get('algorithm') != ag_core.ALGORITHM_IMPORTED:
            # Trained, or calculated on these events when they were gated.
            boundary = deepcopy(self.state.trained_boundaries.get(name)
                                or self._result.boundaries.get(name)) or None
            key = (name, 'applied')
        else:
            key = (name, pd.get('gate_type'), pd.get('gate_marker_x'),
                   ag_review.sync_fingerprint(pd))
            boundary = None
        if key in self._proposals:
            return self._proposals[key]
        pnn = _experiment_channels(self.controller)
        index = {c: i for i, c in enumerate(pnn)}
        by_name = ag_core.gates_by_name(self.state.gate_definitions)
        eff = ag_core.effective_gate_def(pd, by_name)
        parent = self._data_t[self._mask] if self._mask is not None else self._data_t
        limits = self._axis_limits()
        if boundary is None:
            sample = parent
            if len(sample) > self._PROPOSAL_EVENTS:
                rng = np.random.default_rng(TRAINING_SEED)
                sample = sample[np.sort(rng.choice(len(sample), self._PROPOSAL_EVENTS, replace=False))]
            calc = ag_core.GateCalculator(limits, self.tf.objects)
            try:
                resolved, reason = ag_core.resolve_replicates(eff, self._data_t, index,
                                                              self._result, by_name,
                                                              transforms=self.tf.objects)
                if resolved is None:
                    log.info("review: %s not calculated: %s", name, reason)
                    boundary = {}
                else:
                    boundary = calc.calculate(sample, resolved, index,
                                              trained=self._result.boundaries, by_name=by_name)
                    if calc.notes:
                        log.info("review: %s: %s", name, '; '.join(calc.notes))
            except Exception as exc:
                log.warning("review: %s could not be calculated with %s: %s", name, algo, exc)
                boundary = {}
        self._fallbacks.discard(name)
        status = (self._result.status.get(name) or {}) if self._result is not None else {}
        if (not boundary or status.get('source') == 'reference') and g.get('template'):
            # Not calculable yet (e.g. no replicate source): show the imported boundary.
            boundary = deepcopy(g['template'])
            eff = g
            self._fallbacks.add(name)
        if boundary and len(parent):
            masks = ag_core.population_masks(eff, boundary, parent, index, axis_limits=limits)
            for pop, m in masks.items():
                if pop in boundary:
                    boundary[pop]['fraction'] = float(m.mean())
        self._proposals[key] = boundary
        return boundary

    # -- changing gates ---------------------------------------------------------

    def _descendants(self, gate_name: str) -> set:
        return ag_review.descendants(self.state.gate_definitions, gate_name)

    def _untrain(self, names):
        """Training of the named gates, and of the gates below them, no
        longer applies."""
        for name in list(names):
            for n in {name} | self._descendants(name):
                self.state.trained_boundaries.pop(n, None)

    def _set_algorithm(self, g: dict, algorithm: str):
        """Change a gate's algorithm; its training and that of the gates
        below it no longer applies."""
        g['algorithm'] = algorithm
        self._untrain([g['gate_name']])

    def _sync(self, gate_name: str, quiet: bool = False) -> list[str]:
        """Bring the gates sharing *gate_name*'s imported name in line (see
        ag_review.apply_name_sync); returns the names changed."""
        changed, conflict = ag_review.apply_name_sync(self.state.gate_definitions, gate_name)
        self._untrain(changed)
        if changed and not quiet:
            self.bus.statusMessage.emit(
                f"[Automated Gating] {len(changed)} gate(s) named "
                f"'{ag_review.name_group(self.state.gate_by_name(gate_name))}' now use the same "
                "algorithm.")
        if conflict:
            self.bus.statusMessage.emit(
                "[Automated Gating] Conflicting algorithms chosen — identically named gates "
                f"will not be auto-updated ('{conflict}').")
        self._update_sync_banner()
        return changed

    def _update_sync_banner(self):
        conflicts = ag_review.sync_conflicts(self.state.gate_definitions)
        if conflicts:
            shown = ", ".join(f"'{c}'" for c in conflicts[:8])
            more = f" and {len(conflicts) - 8} more" if len(conflicts) > 8 else ''
            self.lbl_sync.setText("Conflicting algorithms chosen — identically named gates will "
                                  f"not be auto-updated: {shown}{more}.")
        self.lbl_sync.setVisible(bool(conflicts))

    def _on_axis_algorithm(self, gate_name: str, axis: str, algorithm: str):
        """An axis algorithm was chosen in a gate's controls."""
        g = self.state.gate_by_name(gate_name)
        if g is None or not algorithm:
            return
        self._adopt_layout(g)
        gp = g.setdefault('gate_param', {})
        imported = g.get('algorithm') == ag_core.ALGORITHM_IMPORTED
        two_d = g.get('gate_type') == '2dsep'
        if two_d:
            # Each axis keeps its own algorithm when the other one changes;
            # while imported, choosing y leaves x to the proposal (fitted
            # with the y chosen).
            current = self._trainable_axes(g)
            for a in ('x', 'y') if not imported or axis == 'x' else ():
                gp.setdefault(f'algorithm_{a}', current.get(a))
        if axis == 'x':
            if imported:
                g['proposed_algorithm'] = algorithm
                g.pop('proposal_rule', None)
            else:
                g['algorithm'] = algorithm
            if two_d:
                gp['algorithm_x'] = algorithm
        else:
            gp['algorithm_y'] = algorithm
        if algorithm == ag_core.ALGORITHM_REPLICATE and not isinstance(gp.get(f'replicate_{axis}'), dict):
            gp[f'replicate_{axis}'] = None
        self._after_config_change(g)

    def _on_source(self, gate_name: str, axis: str, data):
        g = self.state.gate_by_name(gate_name)
        if g is None:
            return
        src = json.loads(data) if data else None
        self._adopt_proposal_sources(g)
        gp = g.setdefault('gate_param', {})
        old = gp.get(f'replicate_{axis}')
        if isinstance(old, dict) and src and src.get('population') and old.get('population') == src.get('population'):
            src['algorithm'] = old.get('algorithm', 'tail')
        gp[f'replicate_{axis}'] = src
        try:
            ag_core.ordered_gate_defs(self.state.gate_definitions)
        except ValueError:
            gp[f'replicate_{axis}'] = old
            QMessageBox.warning(self, "Replicate source",
                                "That source depends on this gate, so it cannot be used.")
            QTimer.singleShot(0, self._rebuild_plots)
            return
        self._after_config_change(g)

    def _on_source_algorithm(self, gate_name: str, axis: str, algorithm: str):
        g = self.state.gate_by_name(gate_name)
        if g is not None:
            self._adopt_proposal_sources(g)
        src = ((g or {}).get('gate_param') or {}).get(f'replicate_{axis}')
        if g is None or not isinstance(src, dict):
            return
        src['algorithm'] = algorithm
        self._after_config_change(g)

    def _after_config_change(self, g: dict):
        """A gate's trainable configuration changed. A converted gate becomes
        the user's choice, is calculated again and brings its same-named
        gates along; an imported gate only shows the new proposal."""
        if g.get('algorithm') != ag_core.ALGORITHM_IMPORTED:
            g['sync_origin'] = ag_review.SYNC_USER
            self._untrain([g['gate_name']])
            self._sync(g['gate_name'])
            self._regate()
        # The combo that sent this is rebuilt, so wait until it has returned.
        QTimer.singleShot(0, self._rebuild_plots)
        self.gatesChanged.emit()
        self.templatesChanged.emit()

    def _convert(self, gate_name: str):
        g = self.state.gate_by_name(gate_name)
        if g is None:
            return
        if self._gate_evidence(g).get('split'):
            made = self._split_imported([gate_name])
            if not made:
                return
            self._regate(self._needed_gates() | set(made))
            for name in made:
                new = self.state.gate_by_name(name)
                if new is not None and self._convert_one(new):
                    new['sync_origin'] = ag_review.SYNC_USER
            self.bus.statusMessage.emit(
                f"[Automated Gating] Split '{gate_name}' into "
                + ", ".join(f"'{n}'" for n in made) + "; train to calculate them.")
            self._structure_changed()
            return
        algorithm = self._convert_one(g)
        if not algorithm:
            return
        g['sync_origin'] = ag_review.SYNC_USER
        changed = self._sync(gate_name, quiet=True)
        tail = f" {len(changed)} gate(s) with the same name follow it." if changed else ''
        self.bus.statusMessage.emit(
            f"[Automated Gating] '{gate_name}' now uses '{algorithm}'; train to calculate "
            f"it.{tail}")
        self._after_change()

    def _convert_one(self, g: dict) -> str | None:
        """Make an imported gate the trainable gate it converts to (see
        ag_review.proposed_definition); returns its algorithm, or None when
        it has none."""
        if not self._trainable_algorithm(g):
            return None
        ag_review.apply_conversion(g, self._proposed_def(g))
        self._untrain([g['gate_name']])
        return g.get('algorithm')

    def _split_imported(self, names) -> list[str]:
        """Split the named imported 2dsep gates into sequential 1dsep gates
        as the review samples propose (ag_review.split_steps), keeping their
        shapes as drawn; returns the names of the new gates."""
        made = []
        for name in names:
            g = self.state.gate_by_name(name)
            steps = self._gate_evidence(g).get('split') if g is not None else None
            if not steps:
                continue
            labels = {st['channel']: self._axis_label(st['channel']) for st in steps}
            stale = {name} | self._descendants(name)
            try:
                defs, new = ag_review.split_2dsep(self.state.gate_definitions, name, steps,
                                                  names=labels, keep_drawn=True)
            except ValueError as exc:
                self.bus.statusMessage.emit(f"[Automated Gating] '{name}' was not split: {exc}")
                continue
            self.state.gate_definitions = defs
            for n in stale:
                self.state.trained_boundaries.pop(n, None)
            self.state.algorithm_recommendations.pop(name, None)
            made += new
        return made

    def _on_convert_all(self):
        """Convert every gate still as drawn to its proposed form (splitting
        the 2dsep gates the review samples propose to split first). The
        review events' evidence for every gate is gathered in a worker
        first. Bulk conversions are not the user's choice for a gate, so
        same-named gates the user did set bring them in line."""
        if self._evidence_worker is not None:
            self._evidence_worker.abort()
            return
        pending = [g for g in self.state.gate_definitions
                   if g.get('algorithm') == ag_core.ALGORITHM_IMPORTED]
        if not pending:
            return
        ans = QMessageBox.question(
            self, "Convert all to trainable",
            f"Convert {len(pending)} gate(s) that still apply their shapes as drawn to their "
            "proposed algorithms?")
        if ans != QMessageBox.Yes:
            return
        if self._data_t is None:
            self._finish_convert_all()
            return
        everything = {g['gate_name'] for g in self.state.gate_definitions}
        if not everything <= self._gated:
            # The proposals need every gate gated on the review events.
            from PySide6.QtWidgets import QApplication
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self._regate(everything)
            finally:
                QApplication.restoreOverrideCursor()
        jobs = [(self._evidence_key(g), deepcopy(g)) for g in pending
                if g['gate_name'] in self._result.parent_masks
                and self._evidence_key(g) not in self._evidence]
        if not jobs:
            self._finish_convert_all()
            return
        pnn = _experiment_channels(self.controller)
        worker = _EvidenceWorker(jobs, deepcopy(self.state.gate_definitions), self._data_t,
                                 {c: i for i, c in enumerate(pnn)}, self._result,
                                 self._axis_limits(), self._zeros(), parent=self)
        worker.progress.connect(lambda done, total: self.lbl_status.setText(
            f"Convert all: checking {done} of {total} gate(s) on {self._samples_text()} … "
            "(click Convert all again to stop)"))
        worker.finished.connect(self._on_evidence_ready)
        self._evidence_worker = worker
        self.lbl_status.setText(f"Convert all: checking {len(jobs)} gate(s) on "
                                f"{self._samples_text()} …")
        worker.start()

    def _on_evidence_ready(self, ok: bool, message: str, evidence: object):
        self._evidence_worker = None
        if not ok:
            self.lbl_status.setText(f"Convert all stopped: {message}")
            return
        self._evidence.update(evidence or {})
        self._finish_convert_all()

    def _finish_convert_all(self):
        pending = [g for g in self.state.gate_definitions
                   if g.get('algorithm') == ag_core.ALGORITHM_IMPORTED]
        made = self._split_imported([g['gate_name'] for g in pending
                                     if self._gate_evidence(g).get('split')])
        if made and self._data_t is not None:
            # Gating again only adds the new gates; the others' evidence holds.
            kept = dict(self._evidence)
            self._regate({g['gate_name'] for g in self.state.gate_definitions})
            self._evidence.update(kept)
        converted = 0
        for g in [g for g in self.state.gate_definitions
                  if g.get('algorithm') == ag_core.ALGORITHM_IMPORTED]:
            if not self._convert_one(g):
                continue
            g['sync_origin'] = ag_review.SYNC_AUTO
            converted += 1
        followed = 0
        for g in self.state.gate_definitions:
            if g.get('sync_origin') not in (ag_review.SYNC_AUTO, ag_review.SYNC_DRAWN) \
                    and g.get('algorithm') != ag_core.ALGORITHM_IMPORTED:
                followed += len(self._sync(g['gate_name'], quiet=True))
        tail = f" {followed} follow a same-named gate you set." if followed else ''
        split = f" Split {len(made)} new gate(s) from 2dsep gates." if made else ''
        self.bus.statusMessage.emit(
            f"[Automated Gating] Converted {converted} gate(s) to their proposed "
            f"algorithms; train to calculate them.{split}{tail}")
        self.btn_convert_all.setEnabled(False)
        if made:
            self._structure_changed()
        else:
            self._after_change()

    def _use_drawn(self, gate_name: str):
        g = self.state.gate_by_name(gate_name)
        if g is None or not ag_review.has_drawn_shape(g):
            return
        if g.get('algorithm') in ag_review.trainable_algorithms(g):
            g['proposed_algorithm'] = g['algorithm']
        self._set_algorithm(g, ag_core.ALGORITHM_IMPORTED)
        g['sync_origin'] = ag_review.SYNC_DRAWN
        self._update_sync_banner()
        self._after_change()

    def _after_change(self):
        self._regate()
        # The button that sent this is rebuilt, so wait until it has returned.
        QTimer.singleShot(0, self._rebuild_plots)
        self.gatesChanged.emit()
        self.templatesChanged.emit()

    def _on_update(self):
        """Calculate the shown gates again on the review samples."""
        if self._data_t is None:
            self._ensure_sample()
            return
        if self._data_signature != self._transform_signature():
            self._transform_events()
        self._regate()
        self._rebuild_plots()

    def _on_axis_zoomed(self, channel: str, before):
        """An axis of a review plot was dragged.

        A change of the transform itself (*before* holds its earlier
        parameters) converts the model's boundaries (through
        ``transformChanged``) and the boundaries on show, transforms that
        channel's events again and gates again once the drag has stopped.
        The plots using the channel are redrawn at once.
        """
        if before is not None:
            tr = self.tf.objects.get(channel)
            after = ag_core.transform_params(tr) if tr is not None else None
            self.transformChanged.emit(channel, before)
            self._remap_shown(channel, before, after)
            self._retransform_channel(channel)
            self._regate_timer.start()
        for plot in self._plots + [p for _n, p in self._trainable]:
            if isinstance(plot, _ReviewPlot) and channel in plot.channels:
                plot.set_events(self._raw, self._mask)
        for _name, key, plot in self._references:
            mask = self._reference_mask(key)
            if channel in plot.channels and mask is not None:
                plot.set_events(self._raw, mask)
        self._redraw_overlays(channel)

    def _remap_shown(self, channel: str, before: dict, after: dict | None):
        """Convert the applied and proposed boundaries on *channel* to its new
        transform, so they are drawn in place until the events are gated again."""
        if not after or self._result is None:
            return
        defs = self.state.gate_definitions
        names = ag_core.gates_on_channel(defs, self._result.boundaries, channel)
        src, dst = {channel: before}, {channel: after}
        applied = {n: self._result.boundaries[n] for n in names if n in self._result.boundaries}
        if applied:
            converted, _ = ag_core.remap_boundaries(defs, applied, src, dst, points_per_edge=1)
            self._result.boundaries.update(converted)
        for key, boundary in list(self._proposals.items()):
            if key[0] in names and boundary:
                converted, _ = ag_core.remap_boundaries(defs, {key[0]: boundary}, src, dst,
                                                        points_per_edge=1)
                self._proposals[key] = converted[key[0]]

    def _label_texts(self, gate_def: dict, boundary: dict, compare: bool = True) -> dict:
        """{(gate, pop): (text, alert)} for the populations worth labelling;
        *compare* False leaves out the fractions recorded at import."""
        src = gate_def.get('source') or {}
        shown = set((src.get('populations') or {}).keys()) or set(boundary)
        single = self._single_review_key()
        compare = compare and single is not None and ag_review.matches_import_sample(gate_def, single)
        out = {}
        for pop, entry in boundary.items():
            if pop not in shown:
                continue
            frac = entry.get('fraction')
            if frac is None:
                continue
            label = entry.get('label', pop)
            text = f"{label}\n{frac * 100:.1f}%"
            alert = False
            ref = ag_review.imported_fraction(gate_def, pop) if compare else None
            if ref is not None:
                alert = ag_review.fraction_discrepancy(frac, ref)
                text = f"{label}\n{frac * 100:.1f}% (import {ref * 100:.1f}%)"
            out[(gate_def['gate_name'], pop)] = (text, alert)
        return out

    def _drawn_boundary(self, g: dict) -> dict:
        """A gate's boundary for the left-hand plot: its shapes as drawn, with
        their fractions of the parent on these events, else its boundary as
        applied."""
        name = g['gate_name']
        applied = self._result.boundaries.get(name) or {}
        template = g.get('template') or {}
        if not ag_core.is_shape_boundary(template):
            return applied or template
        if ag_core.is_shape_boundary(applied):
            return applied
        out = deepcopy(template)
        pnn = _experiment_channels(self.controller)
        mask = self._result.parent_masks.get(name)
        if mask is None:
            mask = self._mask
        parent = self._data_t[mask] if mask is not None else self._data_t
        if len(parent):
            masks = ag_core.population_masks(g, out, parent, {c: i for i, c in enumerate(pnn)},
                                             axis_limits=self._axis_limits())
            for pop, m in masks.items():
                out[pop]['fraction'] = float(m.mean())
        return out

    def _redraw_overlays(self, channel: str | None = None):
        """Draw the boundaries on the plots (only those using *channel* when
        one is given)."""
        if self._result is None:
            return
        editing = self.chk_edit.isChecked()
        by_plot = {id(plot) for _name, plot in self._trainable}
        for plot in self._plots:
            if not isinstance(plot, _ReviewPlot) or id(plot) in by_plot:
                continue
            if channel is not None and channel not in plot.channels:
                continue
            boundaries, texts, editable = {}, {}, set()
            for g in plot.gates:
                name = g['gate_name']
                boundary = self._drawn_boundary(g)
                boundaries[name] = boundary
                texts.update(self._label_texts(g, boundary))
                if editing and g.get('template') and g.get('algorithm') == ag_core.ALGORITHM_IMPORTED:
                    editable.add(name)
            plot.draw_overlays(boundaries, texts, editable)
        for name, plot in self._trainable:
            if channel is not None and channel not in plot.channels:
                continue
            g = self.state.gate_by_name(name)
            if g is None:
                continue
            boundary = self._proposal(g)
            shown = g if name in self._fallbacks else self._proposed_def(g)
            # The plot edits a copy whose shapes are the boundary shown.
            plot.gates = [dict(shown, template=deepcopy(boundary or {}))]
            fallback = name in self._fallbacks
            editable = {name} if editing and boundary and not fallback and \
                g.get('algorithm') != ag_core.ALGORITHM_IMPORTED else set()
            plot.caption = ("Not calculated (choose a replicate source): imported boundary"
                            if fallback else getattr(plot, '_base_caption', plot.caption))
            colour = _REVIEW_ALERT if fallback else self._algorithm_colours(g)
            plot.draw_overlays({name: boundary}, self._label_texts(g, boundary), editable,
                               colours={name: colour})
        self._redraw_references(channel)

    def _redraw_references(self, channel: str | None = None):
        """Draw each replicating gate on its reference populations, labelled
        with their fractions there."""
        if not self._references:
            return
        pnn = _experiment_channels(self.controller)
        index = {c: i for i, c in enumerate(pnn)}
        by_name = ag_core.gates_by_name(self.state.gate_definitions)
        for name, key, plot in self._references:
            if channel is not None and channel not in plot.channels:
                continue
            g = by_name.get(name)
            mask = self._reference_mask(key)
            boundary = deepcopy(self._proposal(g) or {}) if g is not None else {}
            if g is None or mask is None or name in self._fallbacks or not boundary:
                plot.draw_overlays({}, {}, set())
                continue
            shown = self._proposed_def(g)
            events = self._data_t[mask]
            for entry in boundary.values():
                entry.pop('fraction', None)
            if len(events):
                eff = ag_core.effective_gate_def(shown, by_name)
                masks = ag_core.population_masks(eff, boundary, events, index,
                                                 axis_limits=self._axis_limits())
                for pop, m in masks.items():
                    if pop in boundary:
                        boundary[pop]['fraction'] = float(m.mean())
            plot.gates = [dict(shown, template=deepcopy(boundary))]
            plot.draw_overlays({name: boundary}, self._label_texts(g, boundary, compare=False),
                               set(), colours={name: self._algorithm_colours(g)})

    def _algorithm_colours(self, g: dict):
        """Colour of a gate's trainable boundary: its algorithm's, per axis
        for a 2dsep gate."""
        algo = self._trainable_algorithm(g)
        if self._proposed_def(g).get('gate_type') == '2dsep':
            axes = self._trainable_axes(g)
            return {'x': ag_core.algorithm_colour(axes.get('x') or algo),
                    'y': ag_core.algorithm_colour(axes.get('y') or algo)}
        return ag_core.algorithm_colour(algo)

    # -- conversion flags ----------------------------------------------------

    def _on_flag_clicked(self):
        if self._flag_worker is not None:
            self._flag_worker.abort()
            return
        if self._data_t is None:
            self.lbl_flag.setText("Load the review samples first.")
            return
        worker = _FlagWorker(deepcopy(self.state.gate_definitions),
                             deepcopy(self.state.trained_boundaries), np.array(self._data_t),
                             _experiment_channels(self.controller), self._axis_limits(),
                             dict(self.state.algorithm_recommendations),
                             self.sb_flag_shift.value(), self.sb_flag_fraction.value() / 100.0,
                             zeros=self._zeros(), transforms=dict(self.tf.objects),
                             parent=self)
        worker.progress.connect(lambda done, total: self.lbl_flag.setText(
            f"Checking {done} of {total} gate(s) … (click again to stop)"))
        worker.finished.connect(self._on_flags_ready)
        self._flag_worker = worker
        self.btn_flag.setText("Stop checking")
        self.lbl_flag.setText("Checking …")
        worker.start()

    def _on_flags_ready(self, ok: bool, message: str, flags: object):
        self._flag_worker = None
        self.btn_flag.setText("Flag conversions")
        if not ok:
            self.lbl_flag.setText(message)
            return
        self.state.conversion_flags = dict(flags or {})
        n = len(self.state.conversion_flags)
        self.lbl_flag.setText(f"{n} gate(s) flagged on {self._samples_text()}."
                              + (" Show: Flagged conversions lists them." if n else ''))
        self._mark_flags()
        self._apply_tree_filter()
        self.gatesChanged.emit()
        self.templatesChanged.emit()

    def _on_overlap_clicked(self):
        if self._overlap_worker is not None:
            self._overlap_worker.abort()
            return
        if self._data_t is None:
            self.lbl_overlap.setText("Load the review samples first.")
            return
        worker = _OverlapWorker(deepcopy(self.state.gate_definitions),
                                deepcopy(self.state.trained_boundaries), np.array(self._data_t),
                                _experiment_channels(self.controller), self._axis_limits(),
                                self.sb_overlap.value() / 100.0,
                                transforms=dict(self.tf.objects), parent=self)
        worker.finished.connect(self._on_overlaps_ready)
        self._overlap_worker = worker
        self.btn_overlap.setText("Stop checking")
        self.lbl_overlap.setText("Checking …")
        worker.start()

    def _on_overlaps_ready(self, ok: bool, message: str, flags: object):
        self._overlap_worker = None
        self.btn_overlap.setText("Flag overlaps")
        if not ok:
            self.lbl_overlap.setText(message)
            return
        self.state.overlap_flags = dict(flags or {})
        n = len(self.state.overlap_flags)
        self.lbl_overlap.setText(f"{n} gate(s) flagged on {self._samples_text()}."
                                 + (" Show: Overlapping gates lists them." if n else ''))
        self._mark_flags()
        self._apply_tree_filter()
        self.gatesChanged.emit()
        self.templatesChanged.emit()

    def _on_edit_toggled(self, on: bool):
        self._redraw_overlays()
        if on and self._result is not None:
            gate, pop = self._selected()
            groups = ag_review.daughter_groups(self.state.gate_definitions, gate, pop)
            self.bus.statusMessage.emit(f"[Automated Gating] {self._edit_hint(groups)}")

    # -- editing ------------------------------------------------------------

    def _apply_template(self, gate_name: str, entries: dict):
        gate = self.state.gate_by_name(gate_name)
        if gate is None:
            return
        gate['template'] = entries
        self._untrain([gate_name])

    def _on_gate_edited(self, gate_name: str, entries: dict):
        self._apply_template(gate_name, entries)
        QTimer.singleShot(0, self._after_edit)
        self.bus.statusMessage.emit(
            f"[Automated Gating] Adjusted '{gate_name}'. Train again to apply it to the training samples.")

    def _on_trainable_edited(self, gate_name: str, entries: dict):
        """A converted gate's boundary was moved: keep it as the gate's
        boundary, marked as adjusted, until the gate is trained again."""
        g = self.state.gate_by_name(gate_name)
        if g is None or not entries:
            return
        clean = {}
        for pop, entry in entries.items():
            e = {k: v for k, v in entry.items() if k not in ('fraction', 'n_events')}
            e['adjusted'] = True
            clean[pop] = e
        for name in self._descendants(gate_name):
            self.state.trained_boundaries.pop(name, None)
        self.state.trained_boundaries[gate_name] = clean
        QTimer.singleShot(0, self._after_edit)
        self.bus.statusMessage.emit(
            f"[Automated Gating] Adjusted '{gate_name}' by hand; training it again "
            "replaces the adjustment.")

    def _after_edit(self):
        self._regate()
        self._redraw_overlays()
        self._mark_flags()
        self.templatesChanged.emit()

    def _on_reset(self):
        gate, pop = self._selected()
        changed = False
        for group in ag_review.daughter_groups(self.state.gate_definitions, gate, pop):
            for g in group['gates']:
                original = ag_review.reset_template(g)
                if original:
                    self._apply_template(g['gate_name'], original)
                    changed = True
        if changed:
            self._after_edit()

    # -- hierarchy tree menu ---------------------------------------------------

    def _node_gates(self, nodes) -> list[str]:
        """The gates that make the given populations (the root has none)."""
        out = []
        for gate, _pop in nodes:
            if gate and gate not in out and self.state.gate_by_name(gate) is not None:
                out.append(gate)
        return out

    def _on_tree_menu(self, pos):
        clicked = self.pop_tree.population_at(pos)
        nodes = self.pop_tree.selected_populations()
        if clicked is not None and clicked not in nodes:
            self.pop_tree.select_population(*clicked)
            nodes = [clicked]
        gates = self._node_gates(nodes)
        menu = QMenu(self.pop_tree)
        one = gates[0] if len(gates) == 1 else None
        act_edit = menu.addAction(f"Edit gate '{one}'…" if one else "Edit gate…")
        act_params = menu.addAction(f"Advanced parameters of '{one}'…" if one
                                    else "Advanced parameters…")
        act_edit.setEnabled(one is not None)
        act_params.setEnabled(one is not None)
        menu.addSeparator()
        act_copy = menu.addAction("Copy gate")
        act_branch = menu.addAction("Copy gating branch")
        act_paste = menu.addAction(
            f"Paste ({len(self._clipboard)} gate{'s' if len(self._clipboard) != 1 else ''})"
            if self._clipboard else "Paste")
        menu.addSeparator()
        act_del_gate = menu.addAction("Delete gate…")
        act_del_pop = menu.addAction("Delete population…")
        menu.addSeparator()
        act_ref = menu.addAction("Use as reference population")
        act_ref.setCheckable(True)
        act_anchor = menu.addAction("Use as anchor population")
        act_anchor.setCheckable(True)
        for a in (act_copy, act_branch, act_del_gate):
            a.setEnabled(bool(gates))
        act_paste.setEnabled(bool(self._clipboard) and bool(nodes))
        single = nodes[0] if len(nodes) == 1 and nodes[0][0] else None
        act_del_pop.setEnabled(bool(nodes) and all(n[0] for n in nodes))
        act_ref.setEnabled(single is not None)
        act_anchor.setEnabled(single is not None)
        if single is not None:
            meta = ((self.state.gate_by_name(single[0]) or {}).get('populations') or {}).get(single[1]) or {}
            act_ref.setChecked(bool(meta.get('reference')))
            act_anchor.setChecked(bool(meta.get(ag_core.ANCHOR_FLAG)))
        if one is not None:
            act_edit.triggered.connect(lambda: self.editRequested.emit(one))
            act_params.triggered.connect(lambda: self._edit_params(one))
        act_copy.triggered.connect(lambda: self._copy(gates, branch=False))
        act_branch.triggered.connect(lambda: self._copy(gates, branch=True))
        act_paste.triggered.connect(lambda: self._paste(nodes))
        act_del_gate.triggered.connect(lambda: self._delete_gates(gates))
        act_del_pop.triggered.connect(lambda: self._delete_populations(nodes))
        if single is not None:
            act_ref.triggered.connect(lambda on: self._set_reference(single, on))
            act_anchor.triggered.connect(lambda on: self._set_anchor(single, on))
        menu.exec(self.pop_tree.viewport().mapToGlobal(pos))

    def _edit_params(self, gate_name: str):
        """Edit a gate's advanced parameters; a change counts as a change of
        its configuration."""
        g = self.state.gate_by_name(gate_name)
        if g is None:
            return
        # What OK writes with nothing changed (the dialog fills in defaults).
        probe = deepcopy(g)
        GateParamDialog(probe).accept()
        before = json.dumps(probe.get('gate_param') or {}, sort_keys=True, default=str)
        if GateParamDialog(g, parent=self).exec() != QDialog.Accepted:
            return
        if json.dumps(g.get('gate_param') or {}, sort_keys=True, default=str) == before:
            return
        self._proposals = {}
        self._evidence = {}
        self._after_config_change(g)

    def gate_edited(self, old_name: str, new_name: str):
        """A gate was edited in the gate editor from the tree's menu: keep
        the population on show (under the gate's new name) and refresh."""
        gate, pop = self._selected()
        if old_name != new_name and gate == old_name:
            self.combo_pop.setItemData(self.combo_pop.currentIndex(),
                                       _pop_key_text(new_name, pop))
        self._structure_changed()

    def _copy(self, gates: list[str], branch: bool):
        self._clipboard = ag_review.copy_gates(self.state.gate_definitions, gates, branch=branch)
        self._clipboard_branch = branch
        what = "branch(es)" if branch else "gate(s)"
        self.bus.statusMessage.emit(
            f"[Automated Gating] Copied {len(gates)} {what} ({len(self._clipboard)} gates). "
            "Right-click a population and Paste to add them there.")

    def _paste(self, nodes):
        if not self._clipboard:
            return
        defs = self.state.gate_definitions
        added = []
        try:
            for gate, pop in nodes:
                defs, names = ag_review.paste_gates(defs, self._clipboard, gate or None, pop)
                added += names
        except ValueError as exc:
            QMessageBox.warning(self, "Paste", f"The gates could not be pasted: {exc}")
            return
        self.state.gate_definitions = defs
        for name in added:
            g = self.state.gate_by_name(name)
            if g is not None and g.get('algorithm') != ag_core.ALGORITHM_IMPORTED:
                g['sync_origin'] = ag_review.SYNC_AUTO
            self._sync(name, quiet=True)
        self.bus.statusMessage.emit(
            f"[Automated Gating] Pasted {len(added)} gate(s) onto {len(nodes)} "
            "population(s); train to calculate them.")
        self._structure_changed()

    def _confirm(self, title: str, text: str) -> bool:
        return QMessageBox.question(self, title, text) == QMessageBox.Yes

    def _delete_gates(self, gates: list[str]):
        if not gates:
            return
        below = set()
        for name in gates:
            below |= self._descendants(name)
        below -= set(gates)
        names = ", ".join(f"'{n}'" for n in gates[:5]) + (" …" if len(gates) > 5 else '')
        extra = f" and the {len(below)} gate(s) below" if below else ''
        if not self._confirm("Delete gate", f"Delete {names}{extra}? This cannot be undone."):
            return
        defs, removed, orphaned = ag_review.delete_gates(self.state.gate_definitions, gates)
        self._finish_delete(defs, removed, orphaned)

    def _delete_populations(self, nodes):
        nodes = [n for n in nodes if n[0]]
        if not nodes:
            return
        labels = ", ".join(f"'{p}' of '{g}'" for g, p in nodes[:4]) + (" …" if len(nodes) > 4 else '')
        if not self._confirm("Delete population",
                             f"Delete {labels} and the gates drawn on "
                             f"{'it' if len(nodes) == 1 else 'them'}? This cannot be undone."):
            return
        defs, removed, orphaned = self.state.gate_definitions, [], []
        for gate, pop in nodes:
            defs, gone, lost = ag_review.delete_population(defs, gate, pop)
            removed += gone
            orphaned += [n for n in lost if n not in orphaned]
        self._finish_delete(defs, removed, orphaned)

    def _finish_delete(self, defs, removed, orphaned):
        self.state.gate_definitions = defs
        for name in removed:
            self.state.trained_boundaries.pop(name, None)
            self.state.algorithm_recommendations.pop(name, None)
        msg = f"[Automated Gating] Deleted {len(removed)} gate(s)."
        if orphaned:
            msg += (" These replicated a deleted gate and need a new source: "
                    + ", ".join(f"'{n}'" for n in orphaned[:6]) + ".")
        self.bus.statusMessage.emit(msg)
        self._structure_changed()

    def _set_reference(self, node, on: bool):
        g = self.state.gate_by_name(node[0])
        if g is None or node[1] not in (g.get('populations') or {}):
            return
        meta = g['populations'][node[1]]
        if on:
            meta['reference'] = True
        else:
            meta.pop('reference', None)
        self.bus.statusMessage.emit(
            f"[Automated Gating] '{node[1]}' is {'now' if on else 'no longer'} a reference "
            "population for replicated thresholds.")
        self._structure_changed()

    def _set_anchor(self, node, on: bool):
        """Mark a population as the anchor (the default reference of
        replicated thresholds; one at a time), or return to the automatic
        anchor (ag_core.anchor_population)."""
        g = self.state.gate_by_name(node[0])
        if g is None or node[1] not in (g.get('populations') or {}):
            return
        for other in self.state.gate_definitions:
            for meta in (other.get('populations') or {}).values():
                if isinstance(meta, dict):
                    meta.pop(ag_core.ANCHOR_FLAG, None)
        if on:
            g['populations'][node[1]][ag_core.ANCHOR_FLAG] = True
        anchor = ag_core.anchor_population(self.state.gate_definitions)
        shown = ag_core.split_population_key(anchor)[1] if anchor else 'none'
        self.bus.statusMessage.emit(
            f"[Automated Gating] Anchor population: '{shown}'"
            + ("" if on else " (automatic)") + ".")
        self._evidence = {}
        self._proposals = {}
        self._anchor = _NO_ANCHOR
        self._structure_changed()

    def _structure_changed(self):
        """The hierarchy changed: renumber, refresh the pickers and plots."""
        for k, d in enumerate(ag_core.ordered_gate_defs(self.state.gate_definitions), start=1):
            d['gate_number'] = k
        self._gated = set()
        self._anchor = _NO_ANCHOR
        self.refresh()
        self.gatesChanged.emit()
        self.templatesChanged.emit()


class ExportTab(QWidget):
    """Export tab.

    Gate summary, model metadata, and Save of ``.agmodel`` files (format 2:
    gates, trained boundaries, the builder's transforms, training samples
    with their AF assignments and QC state, and how each gate is applied).
    Saved models are loaded on the Import/Setup tab.

    application : callable -> the model's ``application`` block (per-gate
                  fixed/recalculate modes and drift settings, set on the Run
                  page).
    """

    def __init__(self, state: GatingModelState, bus, controller, tf=None, application=None,
                 parent=None):
        super().__init__(parent)
        self.state = state
        self.bus = bus
        self.controller = controller
        self.tf = tf if tf is not None else tp.TransformSet()
        self._application = application or (lambda: None)
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
            "portable <b>.agmodel</b> file.  Saved models are loaded on "
            "the Import/Setup page."
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
        self.btn_save.setToolTip("Save the current model to a .agmodel file.")
        self.btn_save.clicked.connect(self._on_save_clicked)
        btn_row.addWidget(self.btn_save)
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
            ag_core.write_model(path, build_model_from_state(self.state, self.controller,
                                                             self.tf.params(),
                                                             application=self._application()))
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.state.model_metadata['last_model_path'] = path
        self._update_meta_card()
        ts = datetime.now().strftime('%H:%M:%S')
        self.bus.statusMessage.emit(f"[Automated Gating {ts}] Model saved: {path}")


def build_model_from_state(state: GatingModelState, controller,
                           transform_params: dict | None = None,
                           application: dict | None = None) -> dict:
    """Assemble a format-2 .agmodel dict from the builder state.

    *transform_params* ({channel: parameter dict}) are the transforms the
    boundaries are in: the builder's own, stored for every channel so that
    the model is applied in the transforms it was trained in. Channels
    without one take the experiment's display transform. *application* is
    the model's ``application`` block (per-gate modes and drift settings).

    Main thread: reads the experiment's channel names, its display
    transforms and the training samples' AF assignments.
    """
    antigens = _antigen_map(controller)
    channel_map = {ch: (antigen or ch) for ch, antigen in antigens.items()}

    transforms = _live_transform_params(controller)
    transforms.update({ch: dict(p) for ch, p in (transform_params or {}).items() if p})

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
        application=application,
    )


def apply_model_to_state(model: dict, state: GatingModelState):
    """Load a normalised model's gates, boundaries and training record."""
    state.gate_definitions = list(model.get('gate_definitions', []))
    state.trained_boundaries = dict(model.get('trained_boundaries', {}))
    ag_flowjo.upgrade_gate_definitions(state.gate_definitions)
    summary = dict(model.get('training_summary', {}))
    state.training_summary = summary
    state.training_samples = list(summary.get('training_sample_names', []))
    cap = summary.get('events_cap_per_sample')
    if isinstance(cap, int):
        state.events_per_sample = cap
    state.model_metadata.update(model.get('metadata', {}))


# ---------------------------------------------------------------------------
# BuilderPages — the model-building pages of the Automated Gating plugin.
# Owns the GatingModelState, the transforms and the five pages; the plugin
# widget (automated_gating_tab.PluginWidget) adds its own pages after them.
# ---------------------------------------------------------------------------


class BuilderPages(QObject):
    """The Transforms, Import/Setup, Hierarchy, Train and Export pages, with
    their shared GatingModelState, transforms and persistence.

    The pages are added to *inner_tabs* (the plugin's side navigation) in
    that order. The transforms (``tf``) start as the experiment's, are saved
    with the model and are the transforms the plugin gates samples in; axis
    drags on any page change them, never the main window's plots.

    application   : callable -> the model's ``application`` block (per-gate
                    modes and drift settings, kept by the Run page).
    extra_samples : callable -> samples pooled on the Transforms page when
                    no training samples are chosen.
    train_blocked : callable -> '' or why training cannot start now.

    modelImported(info) is emitted after a saved model replaced the gates
    (see HierarchyTab.import_saved_model), gateRenamed(old, new) after a
    gate was renamed, and modelTrained() after training finished, was
    paused or a boundary was adjusted.
    """

    modelImported = Signal(object)
    gateRenamed = Signal(str, str)
    modelTrained = Signal()

    def __init__(self, bus, controller, inner_tabs, application=None, extra_samples=None,
                 train_blocked=None, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.controller = controller
        _ensure_qt_imports()
        self._application = application or (lambda: None)
        self._extra_samples = extra_samples or (lambda: [])

        # Shared state and transforms — passed by reference to every page.
        self.state = GatingModelState()
        self.tf = tp.TransformSet()

        # QSettings — keyed per experiment in save_state/load_state.
        self._qsettings = QSettings('honeychrome', 'plugin_gating_model_builder')

        # The experiment whose persisted state is held in self.state; set by
        # switch_experiment(). save_state() writes there even after the
        # controller has moved to another experiment.
        self.experiment_key = None
        self.experiment_dir = None

        self.transforms_tab = tp.PopulationTransformsPanel(
            controller, self.tf, self._transform_samples,
            help_text=ag_help_texts.BUILDER_TRANSFORMS,
            reset_label="Reset to the experiment's transforms", reset=self._reset_transforms,
            apply_label="Apply the model's transforms", apply=self._apply_model_transforms)
        self.hierarchy_tab = HierarchyTab(self.state, bus, controller, tf=self.tf)
        self.review_tab    = ReviewTab(self.state, bus, controller, tf=self.tf)
        self.train_tab     = TrainTab(self.state, bus, controller, tf=self.tf,
                                      blocked_reason=train_blocked)
        self.export_tab    = ExportTab(self.state, bus, controller, tf=self.tf,
                                       application=self._application)

        self.inner_tabs = inner_tabs
        inner_tabs.addTab(self.transforms_tab, "Transforms", 'wave-sine')
        inner_tabs.addTab(self.hierarchy_tab, "Import/Setup", 'binary-tree')
        inner_tabs.addTab(self.review_tab,    "Hierarchy", 'adjustments')
        inner_tabs.addTab(self.train_tab,     "Train", 'repeat')
        inner_tabs.addTab(self.export_tab,    "Export", 'file-export')

        self.hierarchy_tab.gatesImported.connect(
            lambda: self.inner_tabs.setCurrentWidget(self.review_tab))
        self.hierarchy_tab.modelImported.connect(self._on_model_imported)
        self.hierarchy_tab.gateRenamed.connect(self.gateRenamed)
        self.review_tab.templatesChanged.connect(self._write_session)
        self.review_tab.splitRequested.connect(self.hierarchy_tab._split_2dsep_gate)
        self.review_tab.editRequested.connect(self._edit_from_review)
        self.hierarchy_tab.gatesSplit.connect(self._write_session)
        self.hierarchy_tab.gatesSplit.connect(
            lambda: self.refresh_page(self.inner_tabs.currentWidget()))
        self.hierarchy_tab.transformChanged.connect(self._on_axis_transformed)
        self.review_tab.transformChanged.connect(self._on_axis_transformed)
        self.transforms_tab.transformsChanged.connect(self._on_axis_transformed)
        self.transforms_tab.populationChanged.connect(
            lambda name: setattr(self.state, 'transform_population', name))
        self.train_tab.trained.connect(self._write_session)
        self.train_tab.boundaries_changed.connect(self._write_session)
        self.train_tab.trained.connect(self.modelTrained)
        self.train_tab.boundaries_changed.connect(self.modelTrained)

    # ----------------------------------------------------------------------
    # Pages
    # ----------------------------------------------------------------------

    def owns(self, widget) -> bool:
        """True when *widget* is one of these pages."""
        return any(widget is page for page in (self.transforms_tab, self.hierarchy_tab,
                                               self.review_tab, self.train_tab, self.export_tab))

    def refresh_page(self, widget) -> bool:
        """Refresh *widget* if it is one of these pages; False otherwise."""
        if widget is self.transforms_tab:
            self.transforms_tab.refresh(self.state.transform_population or None)
            self._update_apply_button()
            return True
        if self.owns(widget):
            widget.refresh()
            return True
        return False

    def on_sample_loaded(self):
        """The main window loaded another sample: only the Import/Setup
        page's preview shows it."""
        if self.inner_tabs.currentWidget() is self.hierarchy_tab:
            self.hierarchy_tab.refresh_preview()

    def _edit_from_review(self, gate_name: str):
        """Edit a gate from the Hierarchy page's tree with the Import/Setup
        gate editor (which carries a rename through the model)."""
        new_name = self.hierarchy_tab._edit_gate(gate_name)
        if new_name is not None:
            self.review_tab.gate_edited(gate_name, new_name)

    def model(self) -> dict:
        """The current model, as it would be saved."""
        return build_model_from_state(self.state, self.controller, self.tf.params(),
                                      application=self._application())

    def progress_message(self, msg: str):
        """Emit a timestamped status message on the bus."""
        ts = datetime.now().strftime('%H:%M:%S')
        self.bus.statusMessage.emit(f"[Automated Gating {ts}] {msg}")

    # ----------------------------------------------------------------------
    # Transforms
    # ----------------------------------------------------------------------

    def _on_axis_transformed(self, channel: str, before: dict):
        """A page changed one of the transforms: keep the model's boundaries
        in the same place in the data."""
        tr = self.tf.objects.get(channel)
        if tr is None:
            return
        ag_core.remap_channel_transform(
            self.state.gate_definitions, self.state.trained_boundaries,
            channel, before, ag_core.transform_params(tr))
        self.state.transforms_custom = True
        self.hierarchy_tab.update_model_transforms_note()

    def _transform_samples(self) -> list[str]:
        """Samples pooled on the Transforms page: the training samples, else
        the plugin's (the samples ticked to gate)."""
        return (self.train_tab.picker.get_ordered_list() or list(self.state.training_samples)
                or list(self._extra_samples()))

    def _reset_transforms(self):
        """Back to the experiment's display transforms, converting the
        model's boundaries so they stay in place in the data."""
        old = self.tf.params()
        self.tf.load(None, self.controller.unmixed_transformations or {})
        for ch, params in self.tf.params().items():
            if ch in old:
                ag_core.remap_channel_transform(
                    self.state.gate_definitions, self.state.trained_boundaries,
                    ch, old[ch], params, points_per_edge=16)
        self.state.transforms_custom = False
        self.transforms_tab.set_transforms()
        self.hierarchy_tab.update_model_transforms_note()
        self._write_session()
        self.progress_message("Transforms reset to the experiment's.")

    def _apply_model_transforms(self):
        """Use the loaded model's own transforms, converting the boundaries
        so they stay in place in the data."""
        st = self.state
        if not st.model_transforms:
            return
        old = self.tf.params()
        new = dict(old)
        new.update(st.model_transforms)
        self.tf.load(new, self.controller.unmixed_transformations or {})
        for ch, params in self.tf.params().items():
            if ch in old and not ag_core.same_transform(old[ch], params):
                ag_core.remap_channel_transform(
                    st.gate_definitions, st.trained_boundaries,
                    ch, old[ch], params, points_per_edge=16)
        st.transforms_custom = False
        self.transforms_tab.set_transforms()
        self.hierarchy_tab.update_model_transforms_note()
        self._write_session()
        self.progress_message("Transforms set to the loaded model's.")

    def _update_apply_button(self):
        """'Apply the model's transforms' only when a loaded model has them."""
        self.transforms_tab.set_apply_available(
            bool(self.state.model_transforms),
            "Use the transforms the loaded model was trained in." if self.state.model_transforms
            else "Available after loading a saved model that stores its transforms "
                 "(Import/Setup → Load saved model…).")

    def _on_model_imported(self, info: dict):
        """A saved model replaced the gates on the Import/Setup page."""
        self.transforms_tab.set_transforms()
        self._update_apply_button()
        self.modelImported.emit(info)
        self._write_session()
        for msg in info.get('messages') or []:
            self.progress_message(msg)
        if info.get('transforms_applied'):
            self.progress_message("The model's own transforms were applied (Transforms page).")
        elif self.state.model_transforms:
            self.progress_message("The model's own transforms are available on the Transforms "
                                  "page (Apply the model's transforms).")

    # ----------------------------------------------------------------------
    # State persistence
    #
    # QSettings, keyed by experiment directory, holds the training-sample
    # selection, events per sample, the controls toggle, the last model path,
    # the gate hierarchy, the conversion flags, a paused run's progress, the
    # population shown on the Transforms page and a loaded model's own
    # transforms, as JSON strings or scalars (the Linux INI backend does not
    # round-trip lists). The trained boundaries are kept in a session model
    # in the experiment's cache folder (SESSION_FILE), in the same .agmodel
    # format as an export, together with the transforms they are in, so
    # training and the transforms survive a restart.
    # ----------------------------------------------------------------------

    def switch_experiment(self, key: str, experiment_dir):
        """Save the held experiment's state, then load *key*'s."""
        if self.experiment_key is not None:
            self.save_state()
        # The pages share self.state by reference, so it is reset in place.
        fresh = GatingModelState()
        for f in dataclass_fields(GatingModelState):
            setattr(self.state, f.name, getattr(fresh, f.name))
        # Claim the experiment before restoring, so a failed restore still
        # saves to the right place afterwards.
        self.experiment_key = key
        self.experiment_dir = Path(experiment_dir)
        self.load_state()

    def _controller_shows_loaded_experiment(self) -> bool:
        if self.experiment_dir is None:
            return False
        try:
            return Path(self.controller.experiment_dir) == self.experiment_dir
        except Exception:
            return False

    def _write_session(self):
        """Write the gate hierarchy and trained boundaries to the session file.

        Skipped while the controller shows a different experiment, because
        the session records the transforms of the experiment it belongs to.
        """
        if not self._controller_shows_loaded_experiment():
            return
        path = self.experiment_dir / SESSION_FILE
        if not self.state.gate_definitions and not path.exists():
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            ag_core.write_model(path, self.model())
        except Exception:
            log.exception("could not write the session model %s", path)

    def save_state(self):
        """Persist the pages' state for the experiment it was loaded from.

        Does nothing before any experiment has been loaded, so default
        state never overwrites a saved session.
        """
        if self.experiment_key is None:
            return
        s = self._qsettings
        s.setValue('default_sep_axis', self.state.default_sep_axis)
        s.beginGroup(self.experiment_key)
        try:
            s.setValue('training_samples_json', json.dumps(list(self.state.training_samples)))
            s.setValue('events_per_sample', int(self.state.events_per_sample))
            s.setValue('include_controls', bool(self.state.include_controls))
            s.setValue('last_model_path',
                       str(self.state.model_metadata.get('last_model_path', '') or ''))
            s.setValue('gate_definitions_json',
                       json.dumps(ag_core.to_jsonable(self.state.gate_definitions)))
            s.setValue('conversion_flags_json',
                       json.dumps(ag_core.to_jsonable(self.state.conversion_flags)))
            s.setValue('overlap_flags_json',
                       json.dumps(ag_core.to_jsonable(self.state.overlap_flags)))
            s.setValue('training_progress_json',
                       json.dumps(ag_core.to_jsonable(self.state.training_progress)))
            s.setValue('transform_population', str(self.state.transform_population or ''))
            view = self.transforms_tab.view()
            if view is not None:
                self.state.transform_view = view
            s.setValue('transform_view_json', json.dumps(self.state.transform_view))
            s.setValue('model_transforms_json',
                       json.dumps(ag_core.to_jsonable(self.state.model_transforms)))
            s.setValue('transforms_custom', bool(self.state.transforms_custom))
            s.remove('training_samples')
        finally:
            s.endGroup()
        self._write_session()

    def load_state(self):
        """Restore persisted state for the held experiment."""
        key = self.experiment_key
        gate_defs = None
        custom = None
        s = self._qsettings
        if str(s.value('default_sep_axis', 'x')) in ('x', 'y'):
            self.state.default_sep_axis = str(s.value('default_sep_axis', 'x'))
        self.hierarchy_tab.sync_default_sep_axis()
        s.beginGroup(key)
        try:
            training = _settings_json(s.value('training_samples_json', ''), None)
            if not isinstance(training, list):
                training = _settings_str_list(s.value('training_samples', []))
            self.state.training_samples = [str(t) for t in training]

            events = _settings_int(s.value('events_per_sample', None))
            if events is not None and events >= 0:
                self.state.events_per_sample = events
            self.state.include_controls = _settings_bool(
                s.value('include_controls', None), self.state.include_controls)

            last_path = s.value('last_model_path', '')
            if last_path:
                self.state.model_metadata['last_model_path'] = str(last_path)

            gate_defs = _settings_json(s.value('gate_definitions_json', ''), None)
            if isinstance(gate_defs, list):
                self.state.gate_definitions = gate_defs

            flags = _settings_json(s.value('conversion_flags_json', ''), None)
            if isinstance(flags, dict):
                self.state.conversion_flags = flags
            overlaps = _settings_json(s.value('overlap_flags_json', ''), None)
            if isinstance(overlaps, dict):
                self.state.overlap_flags = overlaps
            progress = _settings_json(s.value('training_progress_json', ''), None)
            if isinstance(progress, dict):
                self.state.training_progress = progress
            self.state.transform_population = str(s.value('transform_population', '') or '')
            view = _settings_json(s.value('transform_view_json', ''), None)
            self.state.transform_view = view if isinstance(view, dict) else {}
            model_tf = _settings_json(s.value('model_transforms_json', ''), None)
            self.state.model_transforms = model_tf if isinstance(model_tf, dict) else {}
            if s.value('transforms_custom', None) not in (None, ''):
                custom = _settings_bool(s.value('transforms_custom', None), False)
        finally:
            s.endGroup()
        self.transforms_tab.set_view(self.state.transform_view)
        self._load_session(have_gates=isinstance(gate_defs, list))
        ag_flowjo.upgrade_gate_definitions(self.state.gate_definitions)
        ag_core.migrate_retired_algorithms(self.state.gate_definitions,
                                           self.state.trained_boundaries)
        if custom is None:
            # Saved before this was recorded: custom when they differ from the experiment's.
            live = _live_transform_params(self.controller)
            custom = any(ch in live and not ag_core.same_transform(live[ch], p)
                         for ch, p in self.tf.params().items())
        self.state.transforms_custom = custom
        self._update_apply_button()

    def _load_session(self, have_gates: bool):
        """Restore the transforms, the trained boundaries (and, when
        QSettings has none, the gate hierarchy) from the session file.

        The boundaries are kept in the transforms they were saved in, which
        become the builder's; channels the session has no transform for
        take the experiment's.
        """
        live = self.controller.unmixed_transformations or {}
        self.tf.load(None, live)
        if self.experiment_dir is None:
            return
        path = self.experiment_dir / SESSION_FILE
        if not path.exists():
            return
        try:
            model, _ = ag_core.read_model(path)
        except Exception:
            log.exception("could not read the session model %s", path)
            return
        self.tf.load(model.get('transforms'), live)
        if not have_gates:
            self.state.gate_definitions = list(model.get('gate_definitions', []))
        names = {g['gate_name'] for g in self.state.gate_definitions}
        self.state.trained_boundaries = {
            k: v for k, v in (model.get('trained_boundaries') or {}).items() if k in names
        }
        self.state.training_summary = dict(model.get('training_summary') or {})


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
