"""
drc_gating_confirmation.py — Gating confirmation page of the Honeycluster plugin
=================================================================================
Companion module to ``dr_clustering_tab.py`` (filename intentionally does
NOT end in ``_tab.py``, so it is not picked up as a separate plugin tab).

For the clusters of a clustering run, finds the fewest ordinary gates that
recover each cluster (drc_gate_finder) and shows the result as a gating
hierarchy with its verdict: confirmed, not gateable or continuum. Meant to
be used after Stats, on the clusters that differ between groups.

Events come from the run's stored per-cluster sample (the same events the
Cluster Annotation violins and Cluster ID use, at most
drc_clustering.MARKER_SAMPLE_PER_CLUSTER per cluster), on the run's own
transforms; each event is weighted by how many of its cluster's cells it
stands for, so recall and purity are those of all the clustered cells.

v1 simplifications:
  * Runs archived before clusters were sampled (no stored events with rows)
    use their stored values row for row; runs with no stored values at all
    are not offered (cluster again).
  * Noise events (HDBSCAN label -1) are not sampled, so they are not part of
    the background a gate must exclude.

Results are saved as each cluster finishes, one top-level field per target
in the bundle runs/<run_id>.gating_confirmation (drc_run_archive), with the
settings they were found with, plus the last settings and the clusters
ticked for the report.

Thresholds are shown on the run's transform scale (the plots' axes) and in
raw intensity (the unmixed value before the transform), so the gates can be
redrawn elsewhere.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QScrollArea,
    QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget,
)

from honeychrome.controller_components.job_gate import acquire_job
from honeychrome.view_components.differential_plots import significance_column
from honeychrome.view_components.help_toggle_widget import HelpToggleWidget
from honeychrome.view_components.scalable_figure_widget import (
    ask_figure_path, figure_toolbar, report_saved, results_folder, start_path,
)

import drc_gate_finder as gf
import drc_help_texts
import drc_persist
import drc_pipeline
import drc_report
import drc_run_align
import drc_run_archive
from drc_logging import get_logger

log = get_logger(__name__)

JOB_LABEL = 'Honeycluster: gating confirmation'
DISPLAY_EVENTS = 20_000
SETTINGS_FIELD = 'settings'
REPORT_FIELD = 'report'
REPORT_SECTION = 'Gating confirmation'
IMAGE_DPI = 300

FILTER_ALL = 'all'
FILTER_SIGNIFICANT = 'significant'
FILTER_NOT_RUN = 'not_run'

VERDICT_TEXT = {
    gf.CONFIRMED: 'Confirmed',
    gf.NOT_GATEABLE: 'Not gateable',
    gf.CONTINUUM: 'Continuum',
    gf.NOT_CONFIRMED: 'Not confirmed',
}
VERDICT_COLOUR = {
    gf.CONFIRMED: '#1b9e77',
    gf.NOT_GATEABLE: '#d95f02',
    gf.CONTINUUM: '#7570b3',
    gf.NOT_CONFIRMED: '#888888',
}

DEFAULT_SETTINGS = {
    'recall': gf.DEFAULT_RECALL,
    'purity': gf.DEFAULT_PURITY,
    'max_markers': gf.DEFAULT_MAX_MARKERS,
    'holdout': gf.HOLDOUT_EVENTS,
    'snap': False,
    'channels': None,
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class RunEvents:
    """A run's stored events on its own transforms: values (columns =
    channels), cluster labels, sample index, and the number of clustered
    cells each event stands for."""
    X: np.ndarray
    labels: np.ndarray
    sample: np.ndarray
    weights: np.ndarray
    channels: list[str]
    samples: list[str]
    totals: dict


def run_events(controller, cl_run: dict) -> tuple[RunEvents | None, str]:
    """The stored events of a hydrated clustering run, or (None, reason)."""
    channels = list(cl_run.get('channels') or [])
    params = cl_run.get('transform_params')
    labels_of = cl_run.get('labels') or {}
    stored = drc_run_archive.run_marker_values(controller, cl_run)
    if not channels or not params:
        return None, ('This run did not record its channels and transforms; '
                      'cluster again to use this page.')
    if not stored:
        return None, ('This run has no stored events; cluster again to use this page.')
    totals: dict = {}
    for labels in labels_of.values():
        ids, counts = np.unique(np.asarray(labels), return_counts=True)
        for c, n in zip(ids.tolist(), counts.tolist()):
            if c >= 0:
                totals[c] = totals.get(c, 0) + n
    xs, labs, samp, names = [], [], [], []
    for rel in sorted(stored):
        labels = labels_of.get(rel)
        if labels is None:
            continue
        entry = stored[rel]
        values, value_names = np.asarray(entry[0]), list(entry[1])
        lab = np.asarray(labels)
        if len(entry) == 3:
            lab = lab[np.asarray(entry[2], dtype=np.int64)]
        elif len(lab) != len(values):
            continue
        keep = lab >= 0
        X, kept = drc_run_align.transform_named_values(values[keep], value_names, params, channels)
        if list(kept) != channels:
            missing = sorted(set(channels) - set(kept))
            return None, f'The stored events lack channel(s) {", ".join(missing)}.'
        xs.append(X)
        labs.append(lab[keep].astype(np.int32))
        samp.append(np.full(int(keep.sum()), len(names), dtype=np.int32))
        names.append(rel)
    if not xs:
        return None, 'No stored events match the run\'s samples.'
    labels = np.concatenate(labs)
    ids, stored_counts = np.unique(labels, return_counts=True)
    per_event = {int(c): totals.get(int(c), n) / n for c, n in zip(ids, stored_counts)}
    weights = np.array([per_event[int(c)] for c in labels], dtype=np.float64)
    return RunEvents(np.concatenate(xs), labels, np.concatenate(samp), weights,
                     channels, names, totals), ''


def settings_key(settings: dict) -> str:
    return json.dumps(settings, sort_keys=True)


def target_key(clusters) -> str:
    return 'cluster:' + '+'.join(str(int(c)) for c in sorted(np.atleast_1d(clusters)))


def load_results(controller, run_id: str) -> dict:
    """{target key: saved entry} plus the last settings under SETTINGS_FIELD."""
    try:
        return drc_persist.load_bundle(drc_run_archive.gating_confirmation_path(
            controller, run_id)) or {}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.warning('could not load gating confirmation results of run %s (%s)', run_id, exc)
        return {}


def save_fields(controller, run_id: str, fields: dict) -> None:
    """Write *fields* (target keys or SETTINGS_FIELD) without rewriting the
    others."""
    path = drc_run_archive.gating_confirmation_path(controller, run_id)
    if drc_persist.bundle_exists(path):
        drc_persist.update_bundle_fields(path, fields)
    else:
        drc_persist.save_bundle(path, fields)


def cluster_fdr(state, run_id: str, names: dict) -> dict:
    """{cluster id: smallest adjusted p over the frequency and counts tests}
    for the run whose statistics are loaded; {} for any other run."""
    if getattr(state, 'stats_run_id', None) != run_id:
        return {}
    by_name = {}
    for table in (getattr(state, 'freq_results', None), getattr(state, 'counts_results', None)):
        if table is None or table.empty or 'feature' not in table.columns:
            continue
        col, _label = significance_column(table, getattr(state, 'stats_fdr_scope', 'global'))
        for feature, p in table.groupby('feature')[col].min().items():
            by_name[str(feature)] = min(float(p), by_name.get(str(feature), 1.0))
    out = {}
    for c in names:
        p = by_name.get(cluster_name(names, int(c)))
        if p is not None:
            out[int(c)] = p
    return out


def cluster_name(names: dict, c: int) -> str:
    if c in names:
        return str(names[c])
    if str(c) in names:
        return str(names[str(c)])
    return 'Noise' if c < 0 else f'Cluster {c}'


# ---------------------------------------------------------------------------
# Figures and tables
# ---------------------------------------------------------------------------

def raw_value(cl_run: dict, channel: str, value):
    """A threshold on the run's transform scale as raw intensity (the
    unmixed value before the run's transform); None stays None."""
    if value is None:
        return None
    params = (cl_run.get('transform_params') or {}).get(channel) or {}
    xform = drc_pipeline._build_flowkit_transform(int(params.get('id', 1)) if params else 1,
                                                   params)
    if xform is None:
        return float(value)
    return float(np.asarray(xform.inverse(np.array([float(value)])))[0])


def finding_tables(cl_run: dict, f: dict, compromise: bool, labels: dict, names: dict,
                   samples: list) -> dict:
    """The tables behind a result: gates (one row per marker, thresholds on
    the plot scale and in raw intensity, metrics after its step), per sample,
    impurity, fewer markers and trade-off. Per sample and impurity describe
    the gating strategy."""
    cuts = f['compromise'] if compromise else f['strategy']
    if compromise:
        m = f.get('compromise_metrics') or {}
        groups = [(None, list(cuts), m.get('recall'), m.get('purity'))]
    else:
        groups = [(k, [s['x']] + ([s['y']] if s['y'] else []), s['recall'], s['purity'])
                  for k, s in enumerate(f['steps'], 1)]
    gates = []
    for step, chans, r, p in groups:
        for ch in chans:
            lo, hi = cuts.get(ch, [None, None])
            gates.append({'step': step, 'marker': labels.get(ch, ch), 'channel': ch,
                          'low': lo, 'high': hi,
                          'low_raw': raw_value(cl_run, ch, lo),
                          'high_raw': raw_value(cl_run, ch, hi),
                          'recall_after_step': r, 'purity_after_step': p})
    per_sample = [{'sample': samples[r['sample']] if r['sample'] < len(samples)
                   else str(r['sample']), 'cluster_events': r['n_target'],
                   'recall': r['recall'], 'purity': r['purity']} for r in f['per_sample']]
    impurity = [{'other_cluster': cluster_name(names, c), 'share_of_gate': v}
                for c, v in f['impurity']]
    return {
        'gates': pd.DataFrame(gates),
        'per_sample': pd.DataFrame(per_sample),
        'impurity': pd.DataFrame(impurity),
        'fewer_markers': pd.DataFrame(f['path']).rename(columns={'n_markers': 'markers'}),
        'tradeoff': pd.DataFrame(f['tradeoff']),
    }


def make_hierarchy_figure(events: 'RunEvents', f: dict, clusters: list, colour: str,
                          labels: dict, compromise: bool, title: str = ''):
    """One biaxial plot per step (or the compromise's markers in pairs):
    the events passing the earlier steps, the cluster's in its colour, the
    gate as a rectangle; a single-marker step as a histogram. Events are a
    weighted draw of the stored sample, so cluster sizes look as in the
    data. Returns a canvas-less matplotlib Figure."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    cuts = f['compromise'] if compromise else f['strategy']
    chans = list(cuts)
    if not compromise and f['steps']:
        pairs = [(s['x'], s['y']) for s in f['steps']]
    else:
        pairs = [(chans[i], chans[i + 1] if i + 1 < len(chans) else None)
                 for i in range(0, len(chans), 2)]
    rng = np.random.default_rng(0)
    n = len(events.labels)
    pick = rng.choice(n, size=min(DISPLAY_EVENTS, n), replace=False,
                      p=events.weights / events.weights.sum())
    X = events.X[pick]
    in_target = np.isin(events.labels[pick], clusters)
    col = {ch: i for i, ch in enumerate(events.channels)}
    alive = np.ones(len(pick), dtype=bool)
    ncol = min(3, len(pairs))
    nrow = int(np.ceil(len(pairs) / ncol))
    fig = Figure(figsize=(3.4 * ncol, 3.3 * nrow + (0.4 if title else 0)), dpi=100)
    for k, (cx, cy) in enumerate(pairs, 1):
        ax = fig.add_subplot(nrow, ncol, k)
        x = X[:, col[cx]]
        lo_x, hi_x = cuts[cx]
        if cy is None:
            bins = np.linspace(np.quantile(x[alive], 0.002), np.quantile(x[alive], 0.998), 80)
            ax.hist(x[alive & ~in_target], bins=bins, color='#bbbbbb')
            ax.hist(x[alive & in_target], bins=bins, color=colour, alpha=0.8)
            for v in (lo_x, hi_x):
                if v is not None:
                    ax.axvline(v, color='black', lw=1.2)
            ax.set_xlabel(labels.get(cx, cx))
            ax.set_ylabel('Cells')
            inside = (x >= (lo_x if lo_x is not None else -np.inf)) & \
                     (x <= (hi_x if hi_x is not None else np.inf))
        else:
            y = X[:, col[cy]]
            ax.scatter(x[alive & ~in_target], y[alive & ~in_target], s=1, c='#bbbbbb',
                       linewidths=0, rasterized=True)
            ax.scatter(x[alive & in_target], y[alive & in_target], s=1, c=colour,
                       linewidths=0, rasterized=True)
            xl = np.quantile(x[alive], [0.002, 0.998])
            yl = np.quantile(y[alive], [0.002, 0.998])
            lo_y, hi_y = cuts[cy]
            x0 = lo_x if lo_x is not None else min(xl[0], x[alive].min())
            x1 = hi_x if hi_x is not None else max(xl[1], x[alive].max())
            y0 = lo_y if lo_y is not None else min(yl[0], y[alive].min())
            y1 = hi_y if hi_y is not None else max(yl[1], y[alive].max())
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor='black', lw=1.2))
            ax.set_xlim(min(xl[0], x0) - 0.02, max(xl[1], x1) + 0.02)
            ax.set_ylim(min(yl[0], y0) - 0.02, max(yl[1], y1) + 0.02)
            ax.set_xlabel(labels.get(cx, cx))
            ax.set_ylabel(labels.get(cy, cy))
            inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        name = f'Step {k}' if len(pairs) > 1 else 'Gate'
        step = f['steps'][k - 1] if not compromise and k <= len(f['steps']) else None
        if step is not None:
            name += f': {_pct(step["recall"])} kept, {_pct(step["purity"])} pure'
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
        alive &= inside
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class _GateWorker(QThread):
    """Finds the gating strategy of each target in turn."""
    progress = Signal(int, int, str)
    result = Signal(str, str, object)
    done = Signal(bool, str)

    def __init__(self, run_id: str, events: RunEvents, targets: list, settings: dict,
                 names: dict):
        super().__init__()
        self._run_id = run_id
        self._events = events
        self._targets = targets
        self._settings = dict(settings)
        self._names = names

    def run(self):
        try:
            ev, st = self._events, self._settings
            channels = st['channels'] or ev.channels
            cols = [ev.channels.index(ch) for ch in channels]
            X = ev.X[:, cols]
            targets = gf.Targets(recall=st['recall'], purity=st['purity'],
                                 max_markers=st['max_markers'])
            for i, clusters in enumerate(self._targets):
                if self.isInterruptionRequested():
                    break
                label = ' + '.join(cluster_name(self._names, c) for c in clusters)
                self.progress.emit(i, len(self._targets), label)
                entry = {'settings': st, 'clusters': [int(c) for c in clusters],
                         'saved': datetime.now().isoformat(timespec='seconds')}
                try:
                    es = gf.select_events(X, ev.labels, ev.sample, clusters, weights=ev.weights,
                                          holdout=st['holdout'], seed=int(min(clusters)))
                    finding = gf.find_gating_strategy(es, targets, snap=st['snap'])
                    entry['finding'] = gf.finding_to_dict(finding, channels)
                except ValueError as exc:
                    entry['error'] = str(exc)
                self.result.emit(self._run_id, target_key(clusters), entry)
            self.done.emit(True, '')
        except Exception as exc:
            traceback.print_exc()
            self.done.emit(False, str(exc))


# ---------------------------------------------------------------------------
# Dialogs and small widgets
# ---------------------------------------------------------------------------

class _ChannelDialog(QDialog):
    """Tick the markers the gates may use."""

    def __init__(self, channels: list[str], labels: dict, chosen: list[str] | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Markers the gates may use')
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel('Untick markers the gates should not use (for example markers '
                             'a sorter does not have).'))
        self.list = QListWidget()
        for ch in channels:
            item = QListWidgetItem(labels.get(ch, ch))
            item.setData(Qt.UserRole, ch)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if chosen is None or ch in chosen else Qt.Unchecked)
            self.list.addItem(item)
        lay.addWidget(self.list)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def chosen(self) -> list[str]:
        return [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.Checked]


class _NumberItem(QTableWidgetItem):
    """A table cell that sorts by its number (missing values last)."""

    def __init__(self, value: float | None, text: str):
        super().__init__(text)
        self._value = value

    def __lt__(self, other):
        a = self._value if self._value is not None else float('inf')
        b = getattr(other, '_value', None)
        b = b if b is not None else float('inf')
        return a < b


def _table(headers: list[str]) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    t.horizontalHeader().setStretchLastSection(True)
    return t


def _pct(v) -> str:
    return '' if v is None else f'{100 * v:.1f}%'


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

class GatingConfirmationTab(QWidget):
    """
    Tab — Gating confirmation. Cluster list on the left (filter, verdicts,
    run buttons), the selected cluster's gating hierarchy and tables on the
    right.

    channel_labels: callable returning {channel: display label}.
    fdr_threshold: callable returning the Stats tab's FDR threshold.
    """

    def __init__(self, state, bus, controller, *, channel_labels, fdr_threshold, parent=None):
        super().__init__(parent)
        self.state = state
        self.bus = bus
        self.controller = controller
        self._channel_labels = channel_labels
        self._fdr_threshold = fdr_threshold
        self._run = None
        self._events: dict[str, RunEvents] = {}
        self._results: dict[str, dict] = {}
        self._worker: _GateWorker | None = None
        self._token = None
        self._shown = None
        self._loading = False
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.addWidget(HelpToggleWidget(text=drc_help_texts.gating_confirmation_tab_help_text,
                                         scrolling=True))

        top = QHBoxLayout()
        top.addWidget(QLabel('<b>Clustering run:</b>'))
        self.run_combo = QComboBox()
        self.run_combo.currentIndexChanged.connect(lambda _i: self._on_run_changed())
        top.addWidget(self.run_combo, stretch=1)
        outer.addLayout(top)

        settings = QHBoxLayout()
        settings.addWidget(QLabel('Keep at least'))
        self.recall_spin = QDoubleSpinBox()
        self.recall_spin.setRange(50.0, 100.0)
        self.recall_spin.setSuffix(' %')
        self.recall_spin.setDecimals(0)
        self.recall_spin.setToolTip('Recall target: the share of the cluster\'s cells the '
                                    'gates must keep.')
        settings.addWidget(self.recall_spin)
        settings.addWidget(QLabel('of the cluster, with at least'))
        self.purity_spin = QDoubleSpinBox()
        self.purity_spin.setRange(10.0, 100.0)
        self.purity_spin.setSuffix(' %')
        self.purity_spin.setDecimals(0)
        self.purity_spin.setToolTip('Purity target: the share of the cells in the last gate '
                                    'that must belong to the cluster.')
        settings.addWidget(self.purity_spin)
        settings.addWidget(QLabel('of the gate from it, using at most'))
        self.markers_spin = QSpinBox()
        self.markers_spin.setRange(1, 40)
        self.markers_spin.setToolTip('The most markers the gates may use (two per biaxial plot).')
        settings.addWidget(self.markers_spin)
        settings.addWidget(QLabel('markers.'))
        settings.addStretch(1)
        outer.addLayout(settings)

        options = QHBoxLayout()
        options.addWidget(QLabel('Validate on:'))
        self.holdout_combo = QComboBox()
        self.holdout_combo.addItem('Half of each sample\'s cells', gf.HOLDOUT_EVENTS)
        self.holdout_combo.addItem('Half of the samples', gf.HOLDOUT_SAMPLES)
        self.holdout_combo.setToolTip('The gates are found on one half and every number shown '
                                      'is measured on the other half. Half of the samples is '
                                      'stricter: do the gates work on samples they have not '
                                      'seen?')
        options.addWidget(self.holdout_combo)
        self.snap_check = QCheckBox('Move thresholds to density valleys')
        self.snap_check.setToolTip('Move each threshold to the nearest dip in the cell density '
                                   'when the targets still hold, as a person would draw it.')
        options.addWidget(self.snap_check)
        self.channels_btn = QPushButton('Markers…')
        self.channels_btn.clicked.connect(self._choose_channels)
        options.addWidget(self.channels_btn)
        self.lbl_channels = QLabel('')
        options.addWidget(self.lbl_channels)
        self.defaults_btn = QPushButton('Restore defaults')
        self.defaults_btn.clicked.connect(lambda: self._apply_settings(DEFAULT_SETTINGS, save=True))
        options.addWidget(self.defaults_btn)
        options.addStretch(1)
        outer.addLayout(options)
        self._channels: list[str] | None = None
        for w in (self.recall_spin, self.purity_spin, self.markers_spin):
            w.valueChanged.connect(lambda _v: self._on_settings_changed())
        self.holdout_combo.currentIndexChanged.connect(lambda _i: self._on_settings_changed())
        self.snap_check.toggled.connect(lambda _b: self._on_settings_changed())

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        frow = QHBoxLayout()
        frow.addWidget(QLabel('Show:'))
        self.filter_combo = QComboBox()
        self.filter_combo.addItem('All clusters', FILTER_ALL)
        self.filter_combo.addItem('Significant in Stats', FILTER_SIGNIFICANT)
        for status in (gf.CONFIRMED, gf.NOT_GATEABLE, gf.CONTINUUM):
            self.filter_combo.addItem(VERDICT_TEXT[status], status)
        self.filter_combo.addItem('Not run', FILTER_NOT_RUN)
        self.filter_combo.setToolTip('Significant: clusters whose frequency or counts differ at '
                                     'the Stats tab\'s FDR threshold, in the statistics run '
                                     'of this clustering run.')
        self.filter_combo.currentIndexChanged.connect(lambda _i: self._populate_table())
        frow.addWidget(self.filter_combo, stretch=1)
        ll.addLayout(frow)
        self.table = _table(['Cluster', '% of cells', 'FDR', 'Verdict', 'Recall', 'Purity',
                             'Markers'])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setSortingEnabled(True)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        ll.addWidget(self.table, stretch=1)
        brow = QHBoxLayout()
        self.find_btn = QPushButton('Find gating strategy')
        self.find_btn.setToolTip('Find the gates of each selected cluster.')
        self.find_btn.clicked.connect(self._find_selected)
        brow.addWidget(self.find_btn)
        self.find_all_btn = QPushButton('Run for all shown')
        self.find_all_btn.setToolTip('Find the gates of every cluster in the list.')
        self.find_all_btn.clicked.connect(self._find_shown)
        brow.addWidget(self.find_all_btn)
        self.stop_btn = QPushButton('Stop')
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        brow.addWidget(self.stop_btn)
        ll.addLayout(brow)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        ll.addWidget(self.progress)
        self.lbl_status = QLabel('')
        self.lbl_status.setWordWrap(True)
        ll.addWidget(self.lbl_status)
        splitter.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        self.lbl_title = QLabel('')
        self.lbl_title.setWordWrap(True)
        head.addWidget(self.lbl_title, stretch=1)
        self.view_combo = QComboBox()
        self.view_combo.addItem('Gating strategy', 'strategy')
        self.view_combo.addItem('Best compromise', 'compromise')
        self.view_combo.setToolTip('Best compromise: the gates that best balance recall and '
                                   'purity, when the targets cannot be met.')
        self.view_combo.currentIndexChanged.connect(lambda _i: self._show_current())
        head.addWidget(self.view_combo)
        rl.addLayout(head)
        actions = QHBoxLayout()
        self.chk_report = QCheckBox('Include in report')
        self.chk_report.setToolTip('List this cluster\'s gating hierarchy and tables on the '
                                   'Report tab.')
        self.chk_report.toggled.connect(self._on_report_toggled)
        actions.addWidget(self.chk_report)
        actions.addStretch(1)
        self.window_btn = QPushButton('⤢ Open in window')
        self.window_btn.clicked.connect(self._open_window)
        actions.addWidget(self.window_btn)
        self.image_btn = QPushButton('Save image…')
        self.image_btn.setToolTip('Save the gating hierarchy as PNG, PDF or SVG.')
        self.image_btn.clicked.connect(self._save_image_dialog)
        actions.addWidget(self.image_btn)
        self.csv_btn = QPushButton('Export CSV…')
        self.csv_btn.setToolTip('Write the summary of every cluster of this run and the '
                                'gates, per-sample, impurity, fewer-markers and trade-off '
                                'tables of every cluster with a result, to a new folder.')
        self.csv_btn.clicked.connect(self._export_csv_dialog)
        actions.addWidget(self.csv_btn)
        rl.addLayout(actions)
        self.lbl_verdict = QLabel('')
        self.lbl_verdict.setWordWrap(True)
        self.lbl_verdict.setTextFormat(Qt.RichText)
        rl.addWidget(self.lbl_verdict)
        self.figure_scroll = QScrollArea()
        self.figure_scroll.setWidgetResizable(True)
        self.figure_scroll.setMinimumHeight(360)
        rl.addWidget(self.figure_scroll, stretch=3)
        self.detail_tabs = QTabWidget()
        self.steps_table = _table(['Step', 'Marker', 'Gate (plot scale)', 'Gate (raw intensity)',
                                   'Recall', 'Purity'])
        self.samples_table = _table(['Sample', 'Cluster events', 'Recall', 'Purity'])
        self.samples_table.setSortingEnabled(True)
        self.impurity_table = _table(['Cluster', 'Share of the gate'])
        self.path_table = _table(['Markers', 'Recall', 'Purity'])
        self.tradeoff_table = _table(['Balance (beta)', 'Markers', 'Recall', 'Purity'])
        self.detail_tabs.addTab(self.steps_table, 'Steps')
        self.detail_tabs.addTab(self.samples_table, 'Per sample')
        self.detail_tabs.addTab(self.impurity_table, 'Impurity')
        self.detail_tabs.addTab(self.path_table, 'Fewer markers')
        self.detail_tabs.addTab(self.tradeoff_table, 'Trade-off')
        rl.addWidget(self.detail_tabs, stretch=2)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([640, 1060])
        outer.addWidget(splitter, stretch=1)
        self._apply_settings(DEFAULT_SETTINGS)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def settings(self) -> dict:
        return {
            'recall': round(self.recall_spin.value() / 100.0, 4),
            'purity': round(self.purity_spin.value() / 100.0, 4),
            'max_markers': int(self.markers_spin.value()),
            'holdout': self.holdout_combo.currentData(),
            'snap': bool(self.snap_check.isChecked()),
            'channels': None if self._channels is None else list(self._channels),
        }

    def _apply_settings(self, s: dict, save: bool = False):
        self._loading = True
        try:
            self.recall_spin.setValue(100.0 * s.get('recall', gf.DEFAULT_RECALL))
            self.purity_spin.setValue(100.0 * s.get('purity', gf.DEFAULT_PURITY))
            self.markers_spin.setValue(int(s.get('max_markers') or gf.DEFAULT_MAX_MARKERS))
            i = self.holdout_combo.findData(s.get('holdout', gf.HOLDOUT_EVENTS))
            self.holdout_combo.setCurrentIndex(max(i, 0))
            self.snap_check.setChecked(bool(s.get('snap', False)))
            self._channels = None if s.get('channels') is None else list(s['channels'])
        finally:
            self._loading = False
        self._on_settings_changed(save=save)

    def _on_settings_changed(self, save: bool = True):
        n_all = len(self._run.get('channels') or []) if self._run else 0
        n = n_all if self._channels is None else len(self._channels)
        self.lbl_channels.setText(f'{n} of {n_all} markers' if self._run else '')
        if self._loading:
            return
        if save and self._run is not None:
            save_fields(self.controller, self._run['run_id'], {SETTINGS_FIELD: self.settings()})
        self._populate_table()
        self._show_current()

    def _choose_channels(self):
        if self._run is None:
            return
        channels = list(self._run.get('channels') or [])
        dlg = _ChannelDialog(channels, self._channel_labels(), self._channels, self)
        if dlg.exec() == QDialog.Accepted:
            chosen = dlg.chosen()
            if not chosen:
                QMessageBox.warning(self, 'Markers', 'Tick at least one marker.')
                return
            self._channels = None if len(chosen) == len(channels) else chosen
            self._on_settings_changed()

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def refresh(self):
        """Fill the run selector, keeping the selection (else the run of the
        statistics, else the newest run)."""
        previous = self.run_combo.currentData()
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        for entry in self.state.clustering_runs:
            self.run_combo.addItem(entry.get('label') or entry.get('run_id'), entry.get('run_id'))
        for wanted in (previous, getattr(self.state, 'stats_run_id', None)):
            i = self.run_combo.findData(wanted)
            if wanted and i >= 0:
                self.run_combo.setCurrentIndex(i)
                break
        else:
            self.run_combo.setCurrentIndex(self.run_combo.count() - 1)
        self.run_combo.blockSignals(False)
        self._on_run_changed()

    def _on_run_changed(self):
        run_id = self.run_combo.currentData()
        entry = next((e for e in self.state.clustering_runs if e.get('run_id') == run_id), None)
        self._run = None if entry is None else drc_run_archive.hydrate_run(self.controller, entry)
        if self._run is not None and run_id not in self._results:
            self._results[run_id] = load_results(self.controller, run_id)
        saved = (self._results.get(run_id) or {}).get(SETTINGS_FIELD)
        if saved and self._run is not None:
            chans = saved.get('channels')
            if chans is not None and not set(chans) <= set(self._run.get('channels') or []):
                saved = {**saved, 'channels': None}
        self._apply_settings({**DEFAULT_SETTINGS, **(saved or {})})
        self._shown = None
        self._clear_detail()

    def _ensure_events(self) -> RunEvents | None:
        return self._events_for(self._run)

    def _events_for(self, run: dict) -> RunEvents | None:
        """The stored events of *run*, loaded once."""
        run_id = run['run_id']
        if run_id not in self._events:
            events, reason = run_events(self.controller, run)
            if events is None:
                self.lbl_status.setText(reason)
                return None
            self._events[run_id] = events
        return self._events[run_id]

    def _names(self) -> dict:
        return (self._run or {}).get('names') or {}

    def _clusters(self) -> list[int]:
        if self._run is None:
            return []
        seen = set()
        for labels in (self._run.get('labels') or {}).values():
            seen.update(int(c) for c in np.unique(np.asarray(labels)) if c >= 0)
        return sorted(seen)

    # ------------------------------------------------------------------
    # Cluster list
    # ------------------------------------------------------------------

    def _entry(self, clusters) -> dict | None:
        if self._run is None:
            return None
        return (self._results.get(self._run['run_id']) or {}).get(target_key(clusters))

    def _populate_table(self):
        if self._loading:
            return
        selected = {self.table.item(i.row(), 0).data(Qt.UserRole)
                    for i in self.table.selectionModel().selectedRows()} \
            if self.table.selectionModel() else set()
        self.table.blockSignals(True)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        if self._run is None:
            self.table.blockSignals(False)
            return
        names = self._names()
        colours = self._run.get('colors') or {}
        fdr = cluster_fdr(self.state, self._run['run_id'], names)
        threshold = float(self._fdr_threshold())
        totals = {}
        for labels in (self._run.get('labels') or {}).values():
            ids, counts = np.unique(np.asarray(labels), return_counts=True)
            for c, n in zip(ids.tolist(), counts.tolist()):
                totals[c] = totals.get(c, 0) + n
        all_cells = sum(n for c, n in totals.items() if c >= 0) or 1
        current = settings_key(self.settings())
        mode = self.filter_combo.currentData()
        for c in self._clusters():
            entry = self._entry([c])
            finding = (entry or {}).get('finding')
            status = finding['status'] if finding else None
            if mode == FILTER_SIGNIFICANT and not fdr.get(c, 1.0) <= threshold:
                continue
            if mode == FILTER_NOT_RUN and entry is not None:
                continue
            if mode in VERDICT_TEXT and status != mode:
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)
            name_item = QTableWidgetItem(cluster_name(names, c))
            name_item.setData(Qt.UserRole, c)
            colour = colours.get(c, colours.get(str(c)))
            if colour:
                name_item.setData(Qt.DecorationRole, QColor(colour))
            self.table.setItem(row, 0, name_item)
            frac = totals.get(c, 0) / all_cells
            self.table.setItem(row, 1, _NumberItem(frac, _pct(frac)))
            p = fdr.get(c)
            self.table.setItem(row, 2, _NumberItem(p, '' if p is None else f'{p:.3g}'))
            stale = entry is not None and settings_key(entry.get('settings') or {}) != current
            if entry is None:
                verdict = _NumberItem(None, '')
            elif 'error' in entry:
                verdict = _NumberItem(None, 'Too few events')
                verdict.setToolTip(entry['error'])
            else:
                verdict = _NumberItem(None, VERDICT_TEXT.get(status, status)
                                      + (' (other settings)' if stale else ''))
                verdict.setForeground(QColor('#999999' if stale else VERDICT_COLOUR[status]))
            self.table.setItem(row, 3, verdict)
            val = (finding or {}).get('validation') or {}
            self.table.setItem(row, 4, _NumberItem(val.get('recall'), _pct(val.get('recall'))))
            self.table.setItem(row, 5, _NumberItem(val.get('purity'), _pct(val.get('purity'))))
            n_markers = len((finding or {}).get('strategy') or {}) if finding else None
            self.table.setItem(row, 6, _NumberItem(n_markers, '' if n_markers is None
                                                   else str(n_markers)))
            if c in selected:
                self.table.selectRow(row)
        self.table.setSortingEnabled(True)
        self.table.blockSignals(False)

    def _selected_clusters(self) -> list[int]:
        rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
        return [self.table.item(r, 0).data(Qt.UserRole) for r in rows]

    def select_cluster(self, run_id: str, cluster) -> None:
        """Show *cluster* (an id, or a name as the Stats tab shows it) of run
        *run_id*, selecting the run first."""
        i = self.run_combo.findData(run_id)
        if i < 0:
            return
        if self.run_combo.currentIndex() != i:
            self.run_combo.setCurrentIndex(i)
        if isinstance(cluster, str):
            names = self._names()
            match = [c for c in self._clusters() if cluster_name(names, c) == cluster]
            if not match:
                return
            cluster = match[0]
        self.filter_combo.setCurrentIndex(self.filter_combo.findData(FILTER_ALL))
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).data(Qt.UserRole) == int(cluster):
                self.table.clearSelection()
                self.table.selectRow(row)
                self.table.scrollToItem(self.table.item(row, 0))
                break

    def _on_selection_changed(self):
        clusters = self._selected_clusters()
        self._shown = clusters[:1] or None
        self._show_current()

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def _find_selected(self):
        clusters = self._selected_clusters()
        if not clusters:
            QMessageBox.information(self, 'Gating confirmation', 'Select one or more clusters.')
            return
        self._start([[c] for c in clusters])

    def _find_shown(self):
        clusters = [self.table.item(r, 0).data(Qt.UserRole) for r in range(self.table.rowCount())]
        if clusters:
            self._start([[c] for c in clusters])

    def _start(self, targets: list):
        if self._run is None or self._worker is not None:
            return
        events = self._ensure_events()
        if events is None:
            return
        token = acquire_job(self.controller, JOB_LABEL)
        if token is None:
            self.lbl_status.setText(self.controller.job_gate.busy_text())
            return
        self._token = token
        worker = _GateWorker(self._run['run_id'], events, targets, self.settings(), self._names())
        worker.progress.connect(self._on_progress)
        worker.result.connect(self._on_result)
        worker.done.connect(self._on_done)
        self._worker = worker
        self.progress.setRange(0, len(targets))
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.stop_btn.setEnabled(True)
        self.find_btn.setEnabled(False)
        self.find_all_btn.setEnabled(False)
        worker.start()

    def _stop(self):
        if self._worker is not None:
            self._worker.requestInterruption()
            self.lbl_status.setText('Stopping after the current cluster …')

    def _on_progress(self, i: int, n: int, label: str):
        self.progress.setValue(i)
        self.lbl_status.setText(f'Finding the gates of {label} ({i + 1} of {n}) …')

    def _on_result(self, run_id: str, key: str, entry: dict):
        self._results.setdefault(run_id, {})[key] = entry
        try:
            save_fields(self.controller, run_id, {key: entry})
        except OSError as exc:
            log.warning('could not save gating confirmation result %s (%s)', key, exc)
        if self._run is not None and self._run['run_id'] == run_id:
            self._populate_table()
            if self._shown is not None and target_key(self._shown) == key:
                self._show_current()

    def _on_done(self, ok: bool, error: str):
        if self._token is not None:
            self._token.release()
            self._token = None
        self._worker = None
        self.progress.setVisible(False)
        self.stop_btn.setEnabled(False)
        self.find_btn.setEnabled(True)
        self.find_all_btn.setEnabled(True)
        self.lbl_status.setText('' if ok else f'Failed: {error}')

    # ------------------------------------------------------------------
    # Result display
    # ------------------------------------------------------------------

    def _clear_detail(self):
        self.lbl_title.setText('')
        self.lbl_verdict.setText('')
        self.figure_scroll.takeWidget()
        for t in (self.steps_table, self.samples_table, self.impurity_table,
                  self.path_table, self.tradeoff_table):
            t.setRowCount(0)

    def _verdict_text(self, entry: dict) -> str:
        if 'error' in entry:
            return f'Not run: {entry["error"]}'
        f = entry['finding']
        st = entry.get('settings') or {}
        val = f['validation']
        names = self._names()
        impurity = ', '.join(f'{cluster_name(names, c)} ({100 * s:.0f}%)'
                             for c, s in f['impurity'][:2])
        status = f['status']
        colour = VERDICT_COLOUR[status]
        head = f'<b style="color:{colour}">{VERDICT_TEXT[status]}.</b> '
        if status == gf.CONFIRMED:
            body = (f'These gates keep {_pct(val["recall"])} of the cluster, and '
                    f'{_pct(val["purity"])} of the cells in the last gate belong to it.')
        else:
            body = (f'No gates on {st.get("max_markers")} markers or fewer keep '
                    f'{_pct(st.get("recall"))} of the cluster with {_pct(st.get("purity"))} '
                    f'purity (best: {_pct(val["recall"])} kept, {_pct(val["purity"])} pure).')
            sep = f.get('separability')
            if status == gf.NOT_GATEABLE and sep:
                body += (f' A weighted combination of all markers does separate it '
                         f'({_pct(sep["purity"])} pure): a real population whose border with '
                         f'its neighbours runs through many markers at once.')
            elif status == gf.CONTINUUM and sep:
                body += (f' Not even a weighted combination of all markers separates it '
                         f'({_pct(sep["purity"])} pure): a slice of a continuous population.')
            if (self._run or {}).get('algorithm') == 'FlowSOM':
                body += (' If many clusters of a FlowSOM run are not confirmed, try a Leiden '
                         'run: its clusters follow the gaps in the data more closely.')
        if impurity:
            body += f' The other cells in the gate are mostly {impurity}.'
        if settings_key(st) != settings_key(self.settings()):
            body += ' <i>Found with other settings; click Find gating strategy to update.</i>'
        return head + body

    def _show_current(self):
        if self._loading or self._run is None or not self._shown:
            return
        entry = self._entry(self._shown)
        names = self._names()
        self._clear_detail()
        self.lbl_title.setText('<b>' + ' + '.join(cluster_name(names, c) for c in self._shown)
                               + '</b>')
        self._loading = True
        self.chk_report.setChecked(target_key(self._shown) in self._report_keys(self._run['run_id']))
        self._loading = False
        has_result = entry is not None and 'finding' in entry
        for w in (self.chk_report, self.window_btn, self.image_btn):
            w.setEnabled(has_result)
        if entry is None:
            self.lbl_verdict.setText('Not run yet: click <b>Find gating strategy</b>.')
            return
        self.lbl_verdict.setText(self._verdict_text(entry))
        f = entry.get('finding')
        if f is None:
            return
        has_compromise = f.get('compromise') is not None
        if not has_compromise and self.view_combo.currentIndex() != 0:
            self.view_combo.blockSignals(True)
            self.view_combo.setCurrentIndex(0)
            self.view_combo.blockSignals(False)
        self.view_combo.setEnabled(has_compromise)
        use_compromise = has_compromise and self.view_combo.currentData() == 'compromise'
        cuts = f['compromise'] if use_compromise else f['strategy']
        self._ensure_events()
        self._fill_tables(f, use_compromise)
        self._draw(f, cuts, self._shown, use_compromise)

    def _fill_tables(self, f: dict, compromise: bool):
        tables = finding_tables(self._run, f, compromise, self._channel_labels(), self._names(),
                                self._samples())

        def set_rows(table, rows):
            sorting = table.isSortingEnabled()
            table.setSortingEnabled(False)
            table.setRowCount(0)
            for values in rows:
                r = table.rowCount()
                table.insertRow(r)
                for col, (value, text) in enumerate(values):
                    table.setItem(r, col, _NumberItem(value, text))
            table.setSortingEnabled(sorting)

        def cut(lo, hi, fmt):
            parts = []
            if lo is not None and not pd.isna(lo):
                parts.append(f'> {fmt(lo)}')
            if hi is not None and not pd.isna(hi):
                parts.append(f'< {fmt(hi)}')
            return ' and '.join(parts)

        set_rows(self.steps_table, [
            [(None if pd.isna(g['step']) else int(g['step']),
              '' if pd.isna(g['step']) else str(int(g['step']))),
             (None, g['marker']),
             (None, cut(g['low'], g['high'], lambda v: f'{v:.3g}')),
             (None, cut(g['low_raw'], g['high_raw'], lambda v: f'{v:,.0f}')),
             (g['recall_after_step'], _pct(g['recall_after_step'])),
             (g['purity_after_step'], _pct(g['purity_after_step']))]
            for g in tables['gates'].to_dict('records')])
        set_rows(self.samples_table, [
            [(None, r['sample']), (r['cluster_events'], str(r['cluster_events'])),
             (r['recall'], _pct(r['recall'])), (r['purity'], _pct(r['purity']))]
            for r in tables['per_sample'].to_dict('records')])
        set_rows(self.impurity_table, [[(None, r['other_cluster']), (r['share_of_gate'],
                                                               _pct(r['share_of_gate']))]
                                       for r in tables['impurity'].to_dict('records')])
        set_rows(self.path_table, [[(p['markers'], str(p['markers'])),
                                    (p['recall'], _pct(p['recall'])),
                                    (p['purity'], _pct(p['purity']))]
                                   for p in tables['fewer_markers'].to_dict('records')])
        set_rows(self.tradeoff_table, [[(t['beta'], f'{t["beta"]:g}'),
                                        (t['n_markers'], str(t['n_markers'])),
                                        (t['recall'], _pct(t['recall'])),
                                        (t['purity'], _pct(t['purity']))]
                                       for t in tables['tradeoff'].to_dict('records')])

    def _samples(self, run: dict | None = None) -> list:
        events = self._events.get((run or self._run)['run_id'])
        return events.samples if events is not None else []

    def _colour(self, run: dict, clusters: list) -> str:
        colours = run.get('colors') or {}
        return colours.get(clusters[0], colours.get(str(clusters[0]), '#d62728'))

    def _figure(self, run: dict, entry: dict, compromise: bool = False, title: str = ''):
        """The hierarchy figure of a saved result, or None."""
        events = self._events_for(run)
        f = entry.get('finding')
        if events is None or f is None:
            return None
        cuts = f['compromise'] if compromise else f['strategy']
        if not cuts:
            return None
        return make_hierarchy_figure(events, f, entry['clusters'],
                                     self._colour(run, entry['clusters']),
                                     self._channel_labels(), compromise, title)

    def _draw(self, f: dict, cuts: dict, clusters: list[int], compromise: bool):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        entry = self._entry(clusters)
        fig = None if entry is None else self._figure(self._run, entry, compromise)
        if fig is None:
            self.figure_scroll.takeWidget()
            return
        canvas = FigureCanvasQTAgg(fig)
        canvas.setMinimumHeight(int(fig.get_figheight() * 100))
        self.figure_scroll.setWidget(canvas)

    # ------------------------------------------------------------------
    # Report, image and CSV export
    # ------------------------------------------------------------------

    def _report_keys(self, run_id: str) -> list:
        return list((self._results.get(run_id) or {}).get(REPORT_FIELD) or [])

    def _on_report_toggled(self, checked: bool):
        if self._run is None or not self._shown or self._loading:
            return
        run_id = self._run['run_id']
        key = target_key(self._shown)
        keys = [k for k in self._report_keys(run_id) if k != key] + ([key] if checked else [])
        self._results.setdefault(run_id, {})[REPORT_FIELD] = keys
        save_fields(self.controller, run_id, {REPORT_FIELD: keys})

    def _title(self, run: dict, clusters: list) -> str:
        names = run.get('names') or {}
        return (f'{" + ".join(cluster_name(names, c) for c in clusters)} '
                f'({run.get("label") or run["run_id"]})')

    def _current_figure(self):
        if self._run is None or not self._shown:
            return None
        entry = self._entry(self._shown)
        if entry is None:
            return None
        compromise = self.view_combo.isEnabled() and self.view_combo.currentData() == 'compromise'
        return self._figure(self._run, entry, compromise, self._title(self._run, self._shown))

    def _open_window(self):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        fig = self._current_figure()
        if fig is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(self._title(self._run, self._shown))
        lay = QVBoxLayout(dialog)
        canvas = FigureCanvasQTAgg(fig)
        lay.addWidget(figure_toolbar(canvas, dialog, self._export_folder(), self._image_stem(),
                                     save=self.save_image))
        lay.addWidget(canvas)
        dialog.resize(int(fig.get_figwidth() * 110), int(fig.get_figheight() * 110) + 60)
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        dialog.show()

    def save_image(self, path) -> bool:
        """Write the shown cluster's hierarchy to *path* (format from its
        suffix) at IMAGE_DPI."""
        fig = self._current_figure()
        if fig is None:
            return False
        fig.savefig(str(path), dpi=IMAGE_DPI, bbox_inches='tight')
        return True

    def _export_folder(self):
        """Where gating hierarchy images are saved by default."""
        return results_folder(getattr(self.controller, 'experiment_dir', ''),
                              drc_report.REPORTS_SUBDIR)

    def _image_stem(self) -> str:
        names = self._names()
        return drc_report.sanitize_filename(
            'gating_' + '_'.join(cluster_name(names, c) for c in self._shown))

    def _save_image_dialog(self):
        if self._current_figure() is None:
            return
        path = ask_figure_path(self, self._export_folder(), self._image_stem(),
                               title='Save gating hierarchy')
        if path:
            self.save_image(path)
            self.lbl_status.setText(f'Saved {path}')
            report_saved(self, path)

    def summary_table(self, run: dict | None = None) -> pd.DataFrame:
        """One row per cluster of *run* (default: the selected run): size,
        FDR, verdict, metrics, gates and the settings the result was found
        with."""
        run = run or self._run
        if run is None:
            return pd.DataFrame()
        names = run.get('names') or {}
        labels = self._channel_labels()
        results = self._results.get(run['run_id']) or {}
        fdr = cluster_fdr(self.state, run['run_id'], names)
        totals = {}
        for lab in (run.get('labels') or {}).values():
            ids, counts = np.unique(np.asarray(lab), return_counts=True)
            for c, n in zip(ids.tolist(), counts.tolist()):
                if c >= 0:
                    totals[c] = totals.get(c, 0) + n
        all_cells = sum(totals.values()) or 1
        current = settings_key(self.settings())
        rows = []
        for c in sorted(totals):
            entry = results.get(target_key([c])) or {}
            f = entry.get('finding') or {}
            st = entry.get('settings') or {}
            val = f.get('validation') or {}
            comp = f.get('compromise_metrics') or {}
            gates = '; '.join(
                f'{labels.get(ch, ch)}'
                + (f' > {lo:.3g}' if lo is not None else '')
                + (f' < {hi:.3g}' if hi is not None else '')
                for ch, (lo, hi) in (f.get('strategy') or {}).items())
            rows.append({
                'cluster': cluster_name(names, c), 'cluster_id': c,
                'percent_of_cells': 100.0 * totals[c] / all_cells,
                'fdr': fdr.get(c),
                'verdict': VERDICT_TEXT.get(f.get('status'), entry.get('error', 'not run')),
                'recall': val.get('recall'), 'purity': val.get('purity'),
                'markers': len(f.get('strategy') or {}) if f else None,
                'gates': gates,
                'separability_purity': (f.get('separability') or {}).get('purity'),
                'compromise_recall': comp.get('recall'), 'compromise_purity': comp.get('purity'),
                'recall_target': st.get('recall'), 'purity_target': st.get('purity'),
                'max_markers': st.get('max_markers'), 'validation': st.get('holdout'),
                'other_settings': bool(entry) and settings_key(st) != current,
                'found': entry.get('saved'),
            })
        return pd.DataFrame(rows)

    def _detail_tables(self, run: dict, entry: dict) -> dict:
        f = entry.get('finding')
        if f is None:
            return {}
        self._events_for(run)
        return finding_tables(run, f, False, self._channel_labels(), run.get('names') or {},
                              self._samples(run))

    def export_csv(self, folder) -> list:
        """Write summary.csv and, over every cluster with a result, gates,
        per_sample, impurity, fewer_markers and tradeoff CSVs (a cluster
        column first) into *folder*. Returns the files written."""
        from pathlib import Path
        if self._run is None:
            return []
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        written = []
        summary = self.summary_table()
        summary.to_csv(folder / 'summary.csv', index=False)
        written.append(folder / 'summary.csv')
        names = self._names()
        parts: dict = {}
        for key, entry in (self._results.get(self._run['run_id']) or {}).items():
            if not key.startswith('cluster:') or 'finding' not in (entry or {}):
                continue
            name = ' + '.join(cluster_name(names, c) for c in entry['clusters'])
            for table_name, df in self._detail_tables(self._run, entry).items():
                if df.empty:
                    continue
                df = df.copy()
                df.insert(0, 'cluster', name)
                parts.setdefault(table_name, []).append(df)
        for table_name, dfs in parts.items():
            path = folder / f'{table_name}.csv'
            pd.concat(dfs, ignore_index=True).to_csv(path, index=False)
            written.append(path)
        return written

    def _export_csv_dialog(self):
        if self._run is None:
            return
        base = QFileDialog.getExistingDirectory(self, 'Folder for the gating confirmation CSVs',
                                                start_path(self._export_folder(), ''))
        if not base:
            return
        from pathlib import Path
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = drc_report.sanitize_filename(
            f'gating_confirmation_{self._run.get("label") or self._run["run_id"]}_{stamp}')
        written = self.export_csv(Path(base) / name)
        self.lbl_status.setText(f'Wrote {len(written)} CSV file(s) to {Path(base) / name}')

    def get_report_items(self) -> list:
        """For the selected run: the summary table, and each cluster ticked
        'Include in report' (hierarchy figure and its tables)."""
        run = self._run
        if run is None:
            return []
        results = self._results.get(run['run_id']) or {}
        if not any(k.startswith('cluster:') for k in results):
            return []
        label = run.get('label') or run['run_id']
        items = [drc_report.ReportItem(
            key=f'gating_confirmation:{run["run_id"]}:summary', tab=REPORT_SECTION,
            label=f'{label} gating confirmation',
            get_tables=lambda run=run: {'summary': self.summary_table(run)})]
        names = run.get('names') or {}
        for key in self._report_keys(run['run_id']):
            entry = results.get(key)
            if not entry or 'finding' not in entry:
                continue
            name = ' + '.join(cluster_name(names, c) for c in entry['clusters'])
            items.append(drc_report.ReportItem(
                key=f'gating_confirmation:{run["run_id"]}:{key}', tab=REPORT_SECTION,
                label=f'{label} {name}',
                get_figure=lambda run=run, entry=entry: self._figure(
                    run, entry, False, self._title(run, entry['clusters'])),
                get_tables=lambda run=run, entry=entry: self._detail_tables(run, entry)))
        return items
