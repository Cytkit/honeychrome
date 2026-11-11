"""
tools.py
--------
The tools an AI agent calls to drive Honeychrome (design/agent_control.md).

Every tool runs on the Qt main thread (``bridge.AgentBridge``) and changes
the experiment the way the matching click does, so the open window follows:
it drives the widget the user would use, emits the EventBus signal the click
emits, or edits the gating and emits the signals the plot widgets listen to
(``updateSourceChildGates``, ``updateRois``, ``changedGatingHierarchy``).

Tools take and return plain JSON values. ``ToolError`` is an expected
failure, reported to the agent as a tool error it can act on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QBuffer, QIODevice
from PySide6.QtWidgets import QMessageBox
from shiboken6 import isValid

from honeychrome.agent_control import coordinates as geo

logger = logging.getLogger(__name__)

SCOPES = ('raw', 'unmixed')
TAB_FOR_SCOPE = {'raw': 'Raw Data', 'unmixed': 'Unmixed Data'}
# controller.current_mode for each tab (Controller.set_mode); plugin tabs use their name
MODE_FOR_TAB = {'Raw Data': 'raw', 'Spectral Process': 'process', 'Unmixed Data': 'unmixed',
                'Simple Stats': 'statistics'}
TRANSFORM_IDS = {'linear': 0, 'logicle': 1, 'log': 2}
MAX_PEAKS = 6


class ToolError(Exception):
    """An expected failure, reported to the agent as a tool error."""


@dataclass
class Image:
    png: bytes
    info: dict


@dataclass
class ToolSpec:
    name: str
    title: str
    description: str
    schema: dict
    fn: Callable
    writes: bool
    needs_experiment: bool
    destructive: bool
    timeout: float


TOOLS: dict[str, ToolSpec] = {}


def tool(title, description, properties=None, required=(), writes=False, needs_experiment=True,
         destructive=False, timeout=120.0):
    def register(fn):
        TOOLS[fn.__name__] = ToolSpec(
            fn.__name__, title, description,
            {'type': 'object', 'properties': properties or {}, 'required': list(required),
             'additionalProperties': False},
            fn, writes, needs_experiment, destructive, timeout)
        return fn
    return register


SCOPE = {'type': 'string', 'enum': list(SCOPES),
         'description': "'raw' is the Raw Data tab (data as acquired); 'unmixed' is the Unmixed Data tab "
                        "(after spectral unmixing, needs a spectral model)."}
PLOT_INDEX = {'type': 'integer', 'minimum': 0, 'description': 'Index of the plot in its tab, from list_plots.'}
UNITS = {'type': 'string', 'enum': ['data', 'plot'], 'default': 'data',
         'description': "'data' (default): channel values as on the axis tick labels. 'plot': transformed "
                        "axis coordinates, as returned by summarise_plot's plot_coordinates."}
COORDINATES = {
    'type': 'object',
    'description': 'Gate geometry. rectangle and ellipse: x_min, x_max, y_min, y_max (an ellipse fills that '
                   'box; optional angle in degrees). polygon: vertices, a list of [x, y]. range (1D plots): '
                   'x_min, x_max. quadrant: x, y (the crossing point).',
}


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

class ToolContext:
    """What the tools reach: the View, its controller and bus, looked up afresh on every call."""

    def __init__(self, view):
        self.view = view
        self.confirm_hook = None   # replaces the confirmation dialog (headless runs)

    @property
    def controller(self):
        return self.view.controller

    @property
    def bus(self):
        return self.view.bus

    def main_window(self):
        window = self.view.main_window
        if window is None or not isValid(window):
            raise ToolError('No experiment is open in Honeychrome. Use list_experiments and open_experiment.')
        return window

    def experiment_open(self) -> bool:
        window = self.view.main_window
        return window is not None and isValid(window) and self.controller.experiment_dir is not None

    def check_scope(self, scope: str) -> None:
        if scope not in SCOPES:
            raise ToolError(f'Unknown scope "{scope}". Use "raw" or "unmixed".')
        if scope == 'unmixed' and self.controller.experiment.process.get('unmixing_matrix') is None:
            raise ToolError('There is no unmixed data yet: the experiment has no spectral model and unmixing matrix. '
                            'Use the raw scope, or build the spectral model in the Spectral Process tab.')

    def data(self, scope: str) -> dict:
        self.check_scope(scope)
        return getattr(self.controller, f'data_for_cytometry_plots_{scope}')

    def ensure_scope(self, scope: str) -> dict:
        """Show the scope's tab, so that its plots and statistics are current and the user sees the change."""
        self.check_scope(scope)
        if self.controller.current_mode != scope:
            self.switch_tab(TAB_FOR_SCOPE[scope])
        return self.data(scope)

    def tab_names(self) -> list:
        tabs = self.main_window().tabs
        return [tabs.tabText(i) for i in range(tabs.count())]

    def switch_tab(self, name: str) -> str:
        tabs = self.main_window().tabs
        names = self.tab_names()
        matches = [i for i, n in enumerate(names) if n.lower() == name.strip().lower()]
        if not matches:
            raise ToolError(f'No tab "{name}". Tabs: {", ".join(names)}.')
        index = matches[0]
        if tabs.currentIndex() != index:
            tabs.setCurrentIndex(index)   # runs on_tab_changed, which emits modeChangeRequested
        elif self.controller.current_mode != MODE_FOR_TAB.get(names[index], names[index]):
            self.bus.modeChangeRequested.emit(names[index])
        return names[index]

    def grid(self, scope: str):
        return getattr(self.main_window(), f'cytometry_grid_{scope}')

    def plot(self, data: dict, index: int) -> dict:
        plots = data['plots']
        if not 0 <= index < len(plots):
            raise ToolError(f'No plot {index}: the tab has {len(plots)} plots (indices 0 to {len(plots) - 1}).')
        return plots[index]

    def plot_widget(self, scope: str, index: int):
        widgets = self.grid(scope).plot_widgets
        if not 0 <= index < len(widgets) or not isValid(widgets[index]):
            raise ToolError(f'Plot {index} is not shown in the {TAB_FOR_SCOPE[scope]} tab.')
        return widgets[index]

    def require_sample(self) -> str:
        path = self.controller.current_sample_path
        if not path or self.controller.current_sample is None:
            raise ToolError('No sample is loaded. Use list_samples and load_sample first.')
        return path

    def resolve_channel(self, data: dict, name: str) -> str:
        pnn = data['pnn'] or []
        labels = data.get('pnn_labels') or {}
        if name in pnn:
            return name
        wanted = name.strip().lower()
        for channel in pnn:
            if channel.lower() == wanted or str(labels.get(channel, '')).lower() == wanted:
                return channel
        close = [c for c in pnn if wanted in c.lower() or wanted in str(labels.get(c, '')).lower()]
        hint = f' Did you mean: {", ".join(close[:8])}?' if close else ' Use list_channels.'
        raise ToolError(f'No channel "{name}".{hint}')

    def gate_names(self, gating) -> list:
        return [g[0] for g in gating.get_gate_ids()]

    def require_gate(self, gating, name: str) -> tuple:
        paths = gating.find_matching_gate_paths(name)
        if not paths:
            names = self.gate_names(gating)
            raise ToolError(f'No gate "{name}". Gates: {", ".join(names) if names else "none"}.')
        return paths[0]

    def plot_with_gate(self, data: dict, name: str) -> int | None:
        for i, plot in enumerate(data['plots']):
            if name in plot.get('child_gates', []):
                return i
        return None

    def gate_statistics(self, data: dict, name: str) -> dict | None:
        stats = (data.get('statistics') or {}).get(name)
        return _stats_entry(stats) if stats else None

    def confirm(self, text: str) -> None:
        if self.confirm_hook is not None:
            accepted = self.confirm_hook(text)
        else:
            answer = QMessageBox.question(self.main_window(), 'Request from AI agent', text,
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            accepted = answer == QMessageBox.Yes
        if not accepted:
            raise ToolError('The user declined this change in Honeychrome.')


def _stats_entry(stats: dict) -> dict:
    return {'events': int(stats.get('n_events_gate', 0)),
            'percent_of_parent': round(100 * float(stats.get('p_gate_parent', 0) or 0), 3),
            'percent_of_total': round(100 * float(stats.get('p_gate_total', 0) or 0), 3)}


def _transform(data: dict, channel: str):
    transformations = data.get('transformations') or {}
    if channel not in transformations:
        raise ToolError(f'Channel "{channel}" has no axis transform.')
    return transformations[channel]


def _axis(data: dict, channel: str) -> dict:
    t = _transform(data, channel)
    label = (data.get('pnn_labels') or {}).get(channel)
    out = {'channel': channel, 'transform': geo.transform_name(t), 'visible_range': geo.axis_range(t)}
    if label and label != channel:
        out['label'] = label
    return out


def _describe_plot(data: dict, index: int) -> dict:
    plot = data['plots'][index]
    out = {'index': index, 'type': plot['type'], 'source_gate': plot['source_gate'],
           'child_gates': list(plot.get('child_gates', []))}
    if plot.get('channel_x'):
        out['x'] = plot['channel_x']
    if plot.get('channel_y'):
        out['y'] = plot['channel_y']
    return out


def _gate_path_text(path: tuple) -> str:
    return '/'.join(path)


# ---------------------------------------------------------------------------
# Experiment and samples
# ---------------------------------------------------------------------------

@tool('Get Honeychrome state',
      'What is open in Honeychrome now: experiment, tabs and current tab, current sample, whether unmixed data '
      'exists and whether a long job is running. Call this first.',
      needs_experiment=False)
def get_state(ctx: ToolContext) -> dict:
    c = ctx.controller
    if not ctx.experiment_open():
        return {'experiment_open': False,
                'hint': 'Use list_experiments and open_experiment to open an experiment.'}
    window = ctx.main_window()
    samples = c.experiment.samples
    sample = None
    if c.current_sample_path and c.current_sample is not None:
        sample = {'path': c.current_sample_path, 'name': samples['all_samples'].get(c.current_sample_path),
                  'events_in_file': int(c.current_sample.event_count),
                  'events_displayed': None if c.raw_event_data is None else int(len(c.raw_event_data))}
    return {
        'experiment_open': True,
        'experiment': str(c.experiment.experiment_path),
        'tabs': ctx.tab_names(),
        'current_tab': window.tabs.tabText(window.tabs.currentIndex()),
        'current_scope': c.current_mode,
        'current_sample': sample,
        'n_samples': len(samples['all_samples']),
        'unmixed_data_available': c.experiment.process.get('unmixing_matrix') is not None,
        'busy': c.job_gate.busy_text() or None,
        'live_acquisition': not c.stop_live_data_processing.is_set(),
    }


def _experiments_folder() -> Path:
    from honeychrome.settings import experiments_folder
    return (Path.home() / experiments_folder).resolve()


def _recent_experiments() -> list:
    from honeychrome.controller_components.functions import q_settings
    recent = q_settings.value('recent_files', [])
    if isinstance(recent, str):
        recent = [recent]
    return [str(p) for p in recent or []]


@tool('List experiments',
      'Experiment (.kit) files the agent may open: recently opened ones and those in the Experiments folder.',
      needs_experiment=False)
def list_experiments(ctx: ToolContext) -> dict:
    seen, out = set(), []
    candidates = _recent_experiments() + [str(p) for p in sorted(_experiments_folder().glob('*.kit'))]
    candidates += [str(p) for p in sorted(_experiments_folder().glob('*/*.kit'))]
    for p in candidates:
        path = Path(p)
        key = str(path.resolve()).lower()
        if key in seen or not path.exists() or path.suffix != '.kit':
            continue
        seen.add(key)
        out.append({'path': str(path), 'name': path.stem})
        if len(out) >= 50:
            break
    return {'experiments': out}


def _allowed_experiment(path: Path) -> bool:
    resolved = path.resolve()
    if resolved.is_relative_to(_experiments_folder()):
        return True
    return any(Path(p).resolve() == resolved for p in _recent_experiments())


@tool('Open experiment',
      'Open an experiment (.kit) in Honeychrome, replacing the one that is open. The path must come from '
      'list_experiments.',
      properties={'path': {'type': 'string', 'description': 'Path of the .kit file.'}},
      required=('path',), writes=True, needs_experiment=False, timeout=300.0)
def open_experiment(ctx: ToolContext, path: str) -> dict:
    kit = Path(path).expanduser()
    if kit.suffix != '.kit' or not kit.is_file():
        raise ToolError(f'"{path}" is not an experiment file. Use list_experiments.')
    if not _allowed_experiment(kit):
        raise ToolError('Only experiments in the Experiments folder or recently opened ones can be opened.')
    ctx.bus.loadExpRequested.emit(str(kit))
    if not ctx.experiment_open() or Path(ctx.controller.experiment.experiment_path).resolve() != kit.resolve():
        raise ToolError(f'Honeychrome did not open "{kit.name}".')
    return get_state(ctx)


@tool('List samples',
      'Samples in the experiment: path (the key other tools take), name, event count, and whether each is a '
      'single-stain control, unstained or beads.',
      properties={'filter': {'type': 'string', 'description': 'Only samples whose name or path contains this text.'}})
def list_samples(ctx: ToolContext, filter: str = '') -> dict:
    s = ctx.controller.experiment.samples
    controls = set(s.get('single_stain_controls', []))
    unstained = set(s.get('unstained_samples', []))
    beads = set(s.get('bead_samples', []))
    n_events = s.get('all_sample_nevents', {})
    wanted = filter.lower()
    out = []
    for path, name in s['all_samples'].items():
        if wanted and wanted not in path.lower() and wanted not in str(name).lower():
            continue
        out.append({'path': path, 'name': name, 'events': n_events.get(path), 'control': path in controls,
                    'unstained': path in unstained, 'beads': path in beads,
                    'loaded': path == ctx.controller.current_sample_path})
    return {'samples': out, 'count': len(out)}


def _resolve_sample(ctx: ToolContext, sample: str) -> str:
    all_samples = ctx.controller.experiment.samples['all_samples']
    if sample in all_samples:
        return sample
    wanted = sample.strip().lower()
    exact = [p for p, n in all_samples.items() if str(n).lower() == wanted or Path(p).stem.lower() == wanted]
    if len(exact) == 1:
        return exact[0]
    partial = exact or [p for p, n in all_samples.items() if wanted in str(n).lower() or wanted in p.lower()]
    if len(partial) == 1:
        return partial[0]
    if partial:
        raise ToolError(f'"{sample}" matches {len(partial)} samples: '
                        f'{", ".join(all_samples[p] for p in partial[:10])}. Give the path from list_samples.')
    raise ToolError(f'No sample "{sample}". Use list_samples.')


@tool('Load sample',
      'Select and load a sample, as clicking it in the sample panel does. Plots and statistics then show it.',
      properties={'sample': {'type': 'string', 'description': 'Sample path or name, from list_samples.'}},
      required=('sample',), writes=True)
def load_sample(ctx: ToolContext, sample: str) -> dict:
    path = _resolve_sample(ctx, sample)
    c = ctx.controller
    if c.current_sample_path != path or c.current_sample is None:
        ctx.bus.selectSample.emit(path)   # the sample panel selects it and requests the load
        if c.current_sample_path != path:
            ctx.bus.loadSampleRequested.emit(path)
    if c.current_sample_path != path or c.current_sample is None:
        raise ToolError(f'Honeychrome could not load "{path}".')
    return {'loaded': path, 'name': c.experiment.samples['all_samples'].get(path),
            'events_in_file': int(c.current_sample.event_count),
            'events_displayed': None if c.raw_event_data is None else int(len(c.raw_event_data))}


@tool('Switch tab',
      'Show a tab of the main window, e.g. "Raw Data", "Spectral Process", "Unmixed Data", "Simple Stats".',
      properties={'name': {'type': 'string', 'description': 'Tab name, from get_state.'}},
      required=('name',))
def switch_tab(ctx: ToolContext, name: str) -> dict:
    shown = ctx.switch_tab(name)
    return {'current_tab': shown, 'current_scope': ctx.controller.current_mode}


@tool('Save experiment', 'Save the experiment (.kit). Changes are also autosaved a few seconds after each edit.',
      writes=True)
def save_experiment(ctx: ToolContext) -> dict:
    ctx.bus.saveExpRequested.emit('')
    return {'saved': str(ctx.controller.experiment.experiment_path)}


# ---------------------------------------------------------------------------
# Channels, plots, gating, statistics
# ---------------------------------------------------------------------------

@tool('List channels',
      'Channels of a scope with their display labels, whether they are fluorescence channels, and their '
      'axis transform and visible range.',
      properties={'scope': SCOPE}, required=('scope',))
def list_channels(ctx: ToolContext, scope: str) -> dict:
    data = ctx.data(scope)
    if not data['pnn']:
        raise ToolError('The channels are not set up yet: load a sample first.')
    fluoro = {data['pnn'][i] for i in (data.get('fluoro_indices') or [])}
    out = []
    for channel in data['pnn']:
        entry = {'name': channel, 'fluorescence': channel in fluoro}
        if channel in (data.get('transformations') or {}):
            entry.update({k: v for k, v in _axis(data, channel).items() if k != 'channel'})
        out.append(entry)
    return {'scope': scope, 'channels': out}


@tool('List plots', 'The plots in a scope\'s tab: index, type, channels, source gate and the gates drawn on each.',
      properties={'scope': SCOPE}, required=('scope',))
def list_plots(ctx: ToolContext, scope: str) -> dict:
    data = ctx.data(scope)
    return {'scope': scope, 'plots': [_describe_plot(data, i) for i in range(len(data['plots']))]}


@tool('Get gating tree',
      'The gating hierarchy of a scope: each gate with its type, parent, channels and geometry in data units, '
      'the plots it is drawn on, and whether the loaded sample has its own custom version of it.',
      properties={'scope': SCOPE}, required=('scope',))
def get_gating_tree(ctx: ToolContext, scope: str) -> dict:
    data = ctx.data(scope)
    gating = data['gating']
    sample_id = data.get('sample_id')
    out = []
    for name, path in gating.get_gate_ids():
        node = gating._get_gate_node(name, path)
        entry = {'name': name, 'parent': path[-1], 'path': _gate_path_text(path + (name,)), 'kind': node.gate_type}
        if node.gate_type != 'Quadrant':
            try:
                gate = gating.get_gate(name, gate_path=path, sample_id=sample_id)
                entry.update(geo.describe_gate(gate, data.get('transformations') or {}))
            except Exception as exc:
                logger.warning('Agent tools: gate %s not described (%s)', name, exc)
            try:
                entry['custom_for_loaded_sample'] = bool(sample_id and gating.is_custom_gate(sample_id, name))
            except Exception as exc:
                logger.warning('Agent tools: custom state of gate %s not read (%s)', name, exc)
        entry['on_plots'] = [i for i, p in enumerate(data['plots']) if name in p.get('child_gates', [])]
        out.append(entry)
    return {'scope': scope, 'gates': out}


@tool('Get statistics',
      'Event counts and percentages (of parent and of total) for every gate on the loaded sample. Shows the '
      'scope\'s tab. Computed on the displayed events, which may be a subsample of a large file.',
      properties={'scope': SCOPE}, required=('scope',))
def get_statistics(ctx: ToolContext, scope: str) -> dict:
    data = ctx.ensure_scope(scope)
    path = ctx.require_sample()
    gating = data['gating']
    stats = data.get('statistics') or {}
    rows = []
    if 'root' in stats:
        rows.append({'gate': 'root', **_stats_entry(stats['root'])})
    for name, gate_path in gating.get_gate_ids():
        if name in stats:
            parent = gate_path[-1]
            rows.append({'gate': name, 'parent': parent, **_stats_entry(stats[name])})
    c = ctx.controller
    return {'scope': scope, 'sample': path, 'name': c.experiment.samples['all_samples'].get(path),
            'events_in_file': int(c.current_sample.event_count),
            'events_displayed': None if c.raw_event_data is None else int(len(c.raw_event_data)),
            'statistics': rows}


def _png(pixmap) -> bytes:
    buffer = QBuffer()
    buffer.open(QIODevice.WriteOnly)
    pixmap.toImage().save(buffer, 'PNG')
    return bytes(buffer.data())


@tool('Get plot image',
      'A PNG image of one plot as shown in Honeychrome, with its axes (channel, transform, visible range in '
      'data units) and gates. Shows the scope\'s tab.',
      properties={'scope': SCOPE, 'plot_index': PLOT_INDEX}, required=('scope', 'plot_index'))
def get_plot_image(ctx: ToolContext, scope: str, plot_index: int) -> Image:
    from honeychrome.view_components.cytometry_plot_widget import get_widget_pixmap

    data = ctx.ensure_scope(scope)
    plot = ctx.plot(data, plot_index)
    widget = ctx.plot_widget(scope, plot_index)
    pixmap = get_widget_pixmap(widget, scale_factor=1)
    if pixmap.isNull() or pixmap.width() < 10:
        raise ToolError('The plot has not been laid out yet; try again.')
    info = _describe_plot(data, plot_index)
    for axis in ('x', 'y'):
        channel = plot.get(f'channel_{axis}')
        if channel:
            info[f'{axis}_axis'] = _axis(data, channel)
    return Image(_png(pixmap), info)


def _peaks_2d(px, py, tx, ty, bins=64) -> tuple:
    from scipy import ndimage

    counts, ex, ey = np.histogram2d(px, py, bins=bins, range=[tx.limits, ty.limits])
    total = counts.sum()
    if total == 0:
        return [], 0
    smooth = ndimage.gaussian_filter(counts, 1.0)
    is_peak = (smooth == ndimage.maximum_filter(smooth, size=5)) & (smooth > 0.05 * smooth.max())
    peaks = []
    claimed = np.zeros_like(counts, dtype=bool)
    for i, j in sorted(zip(*np.nonzero(is_peak)), key=lambda ij: -smooth[ij]):
        if claimed[i, j]:
            continue
        labels, _ = ndimage.label(smooth >= 0.25 * smooth[i, j])
        region = (labels == labels[i, j]) & ~claimed
        claimed |= region
        rows, cols = np.nonzero(region)
        x_box = geo.to_data(tx, [ex[rows.min()], ex[rows.max() + 1]])
        y_box = geo.to_data(ty, [ey[cols.min()], ey[cols.max() + 1]])
        centre = [float(geo.to_data(tx, (ex[i] + ex[i + 1]) / 2)[0]), float(geo.to_data(ty, (ey[j] + ey[j + 1]) / 2)[0])]
        peaks.append({'centre': centre, 'x_range': [float(v) for v in x_box], 'y_range': [float(v) for v in y_box],
                      'percent_of_events': round(100 * counts[region].sum() / total, 2),
                      'plot_coordinates': {'centre': [float((ex[i] + ex[i + 1]) / 2), float((ey[j] + ey[j + 1]) / 2)],
                                           'x_range': [float(ex[rows.min()]), float(ex[rows.max() + 1])],
                                           'y_range': [float(ey[cols.min()]), float(ey[cols.max() + 1])]}})
        if len(peaks) >= MAX_PEAKS:
            break
    return peaks, int(total)


def _peaks_1d(px, tx, bins=128) -> tuple:
    from scipy import ndimage

    counts, ex = np.histogram(px, bins=bins, range=tx.limits)
    total = counts.sum()
    if total == 0:
        return [], 0
    smooth = ndimage.gaussian_filter1d(counts.astype(float), 1.5)
    is_peak = (smooth == ndimage.maximum_filter1d(smooth, size=7)) & (smooth > 0.05 * smooth.max())
    peaks = []
    claimed = np.zeros(len(counts), dtype=bool)
    for i in sorted(np.nonzero(is_peak)[0], key=lambda k: -smooth[k]):
        if claimed[i]:
            continue
        labels, _ = ndimage.label(smooth >= 0.25 * smooth[i])
        region = (labels == labels[i]) & ~claimed
        claimed |= region
        idx = np.nonzero(region)[0]
        box = geo.to_data(tx, [ex[idx.min()], ex[idx.max() + 1]])
        peaks.append({'centre': float(geo.to_data(tx, (ex[i] + ex[i + 1]) / 2)[0]),
                      'x_range': [float(v) for v in box],
                      'percent_of_events': round(100 * counts[region].sum() / total, 2),
                      'plot_coordinates': {'centre': float((ex[i] + ex[i + 1]) / 2),
                                           'x_range': [float(ex[idx.min()]), float(ex[idx.max() + 1])]}})
        if len(peaks) >= MAX_PEAKS:
            break
    return peaks, int(total)


@tool('Summarise plot',
      'A numerical summary of the events shown on a plot (those in its source gate): percentiles per axis and '
      'the main populations (density peaks) with their approximate extent and share of events, in data units '
      'and plot coordinates. Use it with get_plot_image to decide where to draw gates. Shows the scope\'s tab.',
      properties={'scope': SCOPE, 'plot_index': PLOT_INDEX}, required=('scope', 'plot_index'))
def summarise_plot(ctx: ToolContext, scope: str, plot_index: int) -> dict:
    data = ctx.ensure_scope(scope)
    ctx.require_sample()
    plot = ctx.plot(data, plot_index)
    if plot['type'] not in ('hist1d', 'hist2d'):
        raise ToolError('Only 1D and 2D histogram plots can be summarised.')
    mask = (data.get('gate_membership') or {}).get(plot['source_gate'])
    events = data.get('event_data')
    if mask is None or events is None:
        raise ToolError('The plot\'s events are not computed yet; try again.')
    pnn = data['pnn']
    out = _describe_plot(data, plot_index)
    axes = [('x', plot['channel_x'])] + ([('y', plot['channel_y'])] if plot['type'] == 'hist2d' else [])
    values = {}
    for axis, channel in axes:
        v = np.asarray(events[mask, pnn.index(channel)], dtype=np.float64)
        values[axis] = v
        t = _transform(data, channel)
        info = _axis(data, channel)
        if len(v):
            info['percentiles'] = {str(q): float(p) for q, p in zip((1, 5, 25, 50, 75, 95, 99),
                                                                     np.percentile(v, (1, 5, 25, 50, 75, 95, 99)))}
            pv = geo.to_plot(t, v)
            lo, hi = t.limits
            info['percent_off_scale'] = round(100 * float(np.mean((pv < lo) | (pv > hi))), 2)
        out[f'{axis}_axis'] = info
    out['events'] = int(mask.sum())
    tx = _transform(data, plot['channel_x'])
    if plot['type'] == 'hist2d':
        ty = _transform(data, plot['channel_y'])
        out['populations'], _ = _peaks_2d(geo.to_plot(tx, values['x']), geo.to_plot(ty, values['y']), tx, ty)
    else:
        out['populations'], _ = _peaks_1d(geo.to_plot(tx, values['x']), tx)
    return out


# ---------------------------------------------------------------------------
# Plot editing
# ---------------------------------------------------------------------------

@tool('Add plot',
      'Add a plot to a scope\'s tab: a 2D histogram (x and y channels) or a 1D histogram (x only), showing the '
      'events in source_gate. Shows the scope\'s tab.',
      properties={'scope': SCOPE, 'x_channel': {'type': 'string'},
                  'y_channel': {'type': 'string', 'description': 'Omit for a 1D histogram.'},
                  'source_gate': {'type': 'string', 'default': 'root',
                                  'description': 'Gate whose events the plot shows; "root" for all events.'}},
      required=('scope', 'x_channel'), writes=True)
def add_plot(ctx: ToolContext, scope: str, x_channel: str, y_channel: str | None = None,
             source_gate: str = 'root') -> dict:
    data = ctx.ensure_scope(scope)
    x = ctx.resolve_channel(data, x_channel)
    y = ctx.resolve_channel(data, y_channel) if y_channel else None
    if y == x:
        raise ToolError('x and y must be different channels.')
    if source_gate != 'root':
        ctx.require_gate(data['gating'], source_gate)
    ctx.bus.newPlotRequested.emit(x, y or 'Count')
    index = len(data['plots']) - 1
    if source_gate != 'root':
        data['plots'][index]['source_gate'] = source_gate
        ctx.bus.plotChangeRequested.emit(scope, index)
    ctx.grid(scope).init_grid()   # places the new plot now rather than after the grid's debounce
    return _describe_plot(data, index)


@tool('Delete plot',
      'Delete a plot. Gates drawn on it are deleted too (their child gates move up to the parent). '
      'Honeychrome asks the user to confirm when the plot has gates.',
      properties={'scope': SCOPE, 'plot_index': PLOT_INDEX}, required=('scope', 'plot_index'),
      writes=True, destructive=True, timeout=600.0)
def delete_plot(ctx: ToolContext, scope: str, plot_index: int) -> dict:
    data = ctx.ensure_scope(scope)
    plot = ctx.plot(data, plot_index)
    widget = ctx.plot_widget(scope, plot_index)
    if plot.get('child_gates'):
        ctx.confirm(f'The AI agent wants to delete plot {plot_index} and its gates: '
                    f'{", ".join(plot["child_gates"])}. Allow?')
    grid = ctx.grid(scope)
    grid.select_plot(widget)
    grid.proceed_to_delete_plot()
    grid.init_grid()   # renumbers the remaining plots now rather than after the grid's debounce
    return {'deleted_plot': plot_index, 'plots': [_describe_plot(data, i) for i in range(len(data['plots']))]}


@tool('Set plot source gate',
      'Change which gate\'s events a plot shows. Gates drawn on the plot move under the new source gate, as '
      'when the user picks it from the plot title. Shows the scope\'s tab.',
      properties={'scope': SCOPE, 'plot_index': PLOT_INDEX, 'source_gate': {'type': 'string'}},
      required=('scope', 'plot_index', 'source_gate'), writes=True)
def set_plot_source_gate(ctx: ToolContext, scope: str, plot_index: int, source_gate: str) -> dict:
    data = ctx.ensure_scope(scope)
    ctx.plot(data, plot_index)
    widget = ctx.plot_widget(scope, plot_index)
    widget.configure_title()
    choices = list(widget.plot_title.leftClickMenuItems)
    if source_gate not in choices:
        raise ToolError(f'"{source_gate}" cannot be this plot\'s source gate. Choices: {", ".join(choices)}.')
    widget.set_source_gate(choices.index(source_gate), None)
    if data['plots'][plot_index]['source_gate'] != source_gate:
        raise ToolError(f'"{source_gate}" is inside a gate drawn on this plot, so it cannot be its source gate.')
    return _describe_plot(data, plot_index)


@tool('Set plot channels',
      'Change the channels of a plot without gates (y omitted: 1D histogram). Shows the scope\'s tab.',
      properties={'scope': SCOPE, 'plot_index': PLOT_INDEX, 'x_channel': {'type': 'string'},
                  'y_channel': {'type': 'string'}},
      required=('scope', 'plot_index', 'x_channel'), writes=True)
def set_plot_channels(ctx: ToolContext, scope: str, plot_index: int, x_channel: str,
                      y_channel: str | None = None) -> dict:
    data = ctx.ensure_scope(scope)
    plot = ctx.plot(data, plot_index)
    if plot.get('child_gates'):
        raise ToolError('The plot has gates drawn on it; its channels cannot change. Add a new plot instead.')
    x = ctx.resolve_channel(data, x_channel)
    y = ctx.resolve_channel(data, y_channel) if y_channel else None
    if y == x:
        raise ToolError('x and y must be different channels.')
    plot['channel_x'] = x
    if y:
        plot['channel_y'] = y
        plot['type'] = 'hist2d'
    else:
        plot.pop('channel_y', None)
        plot['type'] = 'hist1d'
    ctx.bus.plotChangeRequested.emit(scope, plot_index)
    return _describe_plot(data, plot_index)


@tool('Set axis transform',
      'Set a channel\'s axis transform in a scope (every plot using the channel changes). Gates on the channel '
      'keep their position on the axis, so they may cover different values afterwards.',
      properties={'scope': SCOPE, 'channel': {'type': 'string'},
                  'transform': {'type': 'string', 'enum': list(TRANSFORM_IDS)}},
      required=('scope', 'channel', 'transform'), writes=True)
def set_axis_transform(ctx: ToolContext, scope: str, channel: str, transform: str) -> dict:
    data = ctx.ensure_scope(scope)
    channel = ctx.resolve_channel(data, channel)
    t = _transform(data, channel)
    if t.id == 'default':
        raise ToolError(f'"{channel}" has a fixed linear axis (e.g. Time) and cannot be transformed.')
    t.set_transform(id=TRANSFORM_IDS[transform])
    ctx.bus.axisTransformed.emit(channel)
    return _axis(data, channel)


@tool('Fit axes to data', 'Rescale a plot\'s axes to the 1st-99th percentile of its events, as the plot\'s '
      '"Fit Axes to Data" menu does.',
      properties={'scope': SCOPE, 'plot_index': PLOT_INDEX}, required=('scope', 'plot_index'), writes=True)
def fit_axes_to_data(ctx: ToolContext, scope: str, plot_index: int) -> dict:
    data = ctx.ensure_scope(scope)
    plot = ctx.plot(data, plot_index)
    ctx.plot_widget(scope, plot_index).fit_axes_to_data()
    out = _describe_plot(data, plot_index)
    for axis in ('x', 'y'):
        if plot.get(f'channel_{axis}'):
            out[f'{axis}_axis'] = _axis(data, plot[f'channel_{axis}'])
    return out


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def _plot_channels(plot: dict, gate_type: str) -> tuple:
    if gate_type == 'range':
        if plot['type'] != 'hist1d':
            raise ToolError('A range gate is drawn on a 1D histogram plot.')
        return plot['channel_x'], None
    if plot['type'] != 'hist2d':
        raise ToolError(f'A {gate_type} gate is drawn on a 2D histogram plot.')
    return plot['channel_x'], plot['channel_y']


def _geometry(data, gate_type, coordinates, cx, cy, units) -> tuple:
    tx = _transform(data, cx)
    ty = _transform(data, cy) if cy else None
    try:
        geom = geo.plot_geometry(gate_type, coordinates, tx, ty, units)
    except geo.GeometryError as exc:
        raise ToolError(str(exc)) from None
    notes = []
    if geo.outside_axes(gate_type, geom, tx, ty):
        notes.append('Part of the gate lies outside the visible axis range.')
    return geom, notes


def _gate_result(ctx: ToolContext, data: dict, name: str, plot_index: int | None, notes: list) -> dict:
    gating = data['gating']
    out = {'gate': name}
    try:
        gate = gating.get_gate(name, sample_id=data.get('sample_id'))
        out.update(geo.describe_gate(gate, data.get('transformations') or {}))
    except Exception as exc:
        logger.warning('Agent tools: gate %s not described (%s)', name, exc)
    out['plot_index'] = plot_index
    stats = ctx.gate_statistics(data, name)
    if stats is not None:
        out['statistics'] = stats
    elif out.get('type') == 'quadrant':
        out['statistics'] = {q: ctx.gate_statistics(data, q) for q in out.get('quadrants', [])}
    if notes:
        out['notes'] = notes
    return out


@tool('Create gate',
      'Draw a new gate on a plot; it gates the events of the plot\'s source gate. Gate types: rectangle, polygon, '
      'ellipse and quadrant on 2D plots, range on 1D plots. Coordinates are in data units unless units is '
      '"plot". Returns the gate and its statistics on the loaded sample. Shows the scope\'s tab.',
      properties={'scope': SCOPE, 'plot_index': PLOT_INDEX,
                  'gate_type': {'type': 'string', 'enum': list(geo.GATE_TYPES)},
                  'coordinates': COORDINATES,
                  'name': {'type': 'string', 'description': 'Gate name, unique in the scope (default R1, P1, ...).'},
                  'units': UNITS},
      required=('scope', 'plot_index', 'gate_type', 'coordinates'), writes=True)
def create_gate(ctx: ToolContext, scope: str, plot_index: int, gate_type: str, coordinates: dict,
                name: str | None = None, units: str = 'data') -> dict:
    data = ctx.ensure_scope(scope)
    plot = ctx.plot(data, plot_index)
    widget = ctx.plot_widget(scope, plot_index)
    gating = data['gating']
    cx, cy = _plot_channels(plot, gate_type)
    existing = set(ctx.gate_names(gating))
    if name:
        name = name.strip()
        if not name or name == 'root' or '/' in name:
            raise ToolError('Gate names must be non-empty, not "root", and contain no "/".')
        if name in existing:
            raise ToolError(f'A gate named "{name}" already exists in {scope}. Choose another name or use update_gate.')
    else:
        n = 1
        while f'{geo.NAME_PREFIX[gate_type]}{n}' in existing:
            n += 1
        name = f'{geo.NAME_PREFIX[gate_type]}{n}'
    geom, notes = _geometry(data, gate_type, coordinates, cx, cy, units)
    gate = geo.build_gate(name, gate_type, geom, cx, cy, data['transformations'])
    gating.add_gate(gate, gate_path=widget.get_gate_path())
    plot['child_gates'].append(name)
    ctx.bus.updateSourceChildGates.emit(scope, name)
    ctx.bus.updateRois.emit(scope, plot_index)
    ctx.bus.changedGatingHierarchy.emit(scope, name)
    return _gate_result(ctx, data, name, plot_index, notes)


@tool('Update gate',
      'Move or reshape an existing gate, as dragging it does: if the loaded sample has a custom version of the '
      'gate, that version changes, otherwise the gate shared by all samples. Same coordinates as create_gate. '
      'Shows the scope\'s tab.',
      properties={'scope': SCOPE, 'gate_name': {'type': 'string'}, 'coordinates': COORDINATES, 'units': UNITS},
      required=('scope', 'gate_name', 'coordinates'), writes=True)
def update_gate(ctx: ToolContext, scope: str, gate_name: str, coordinates: dict, units: str = 'data') -> dict:
    data = ctx.ensure_scope(scope)
    ctx.require_gate(data['gating'], gate_name)
    plot_index = ctx.plot_with_gate(data, gate_name)
    if plot_index is None:
        raise ToolError(f'Gate "{gate_name}" is not drawn on any plot, so it cannot be reshaped.')
    plot = data['plots'][plot_index]
    widget = ctx.plot_widget(scope, plot_index)
    gate = widget._effective_gate(gate_name)
    gate_type = geo.gate_kind(gate) if gate is not None else None
    if gate_type is None:
        raise ToolError(f'Gate "{gate_name}" cannot be reshaped.')
    cx, cy = _plot_channels(plot, gate_type)
    geom, notes = _geometry(data, gate_type, coordinates, cx, cy, units)
    geo.reshape_gate(gate, gate_type, geom, cx, cy, data['transformations'])
    ctx.bus.updateRois.emit(scope, plot_index)
    ctx.bus.changedGatingHierarchy.emit(scope, gate_name)
    return _gate_result(ctx, data, gate_name, plot_index, notes)


@tool('Delete gate',
      'Delete a gate (for all samples). Its child gates are kept and move up to its parent, as in the plot\'s '
      'right-click menu. Honeychrome asks the user to confirm.',
      properties={'scope': SCOPE, 'gate_name': {'type': 'string'}}, required=('scope', 'gate_name'),
      writes=True, destructive=True, timeout=600.0)
def delete_gate(ctx: ToolContext, scope: str, gate_name: str) -> dict:
    data = ctx.ensure_scope(scope)
    gating = data['gating']
    ctx.require_gate(gating, gate_name)
    ctx.confirm(f'The AI agent wants to delete the gate "{gate_name}" in {TAB_FOR_SCOPE[scope]}. Allow?')
    parent = gating.get_parent_gate_id(gate_name)
    parent_name = 'root' if parent is None else parent[0]
    gating.remove_gate(gate_name, keep_children=True)
    plot_index = ctx.plot_with_gate(data, gate_name)
    if plot_index is not None:
        data['plots'][plot_index]['child_gates'].remove(gate_name)
    ctx.bus.updateSourceChildGates.emit(scope, parent_name)
    ctx.bus.changedGatingHierarchy.emit(scope, parent_name)
    if plot_index is not None:
        ctx.bus.updateRois.emit(scope, plot_index)
    return {'deleted': gate_name, 'children_moved_to': parent_name}


@tool('Customise gate for this sample',
      'Give the loaded sample its own copy of a gate (drawn orange), so update_gate then changes it for this '
      'sample only.',
      properties={'scope': SCOPE, 'gate_name': {'type': 'string'}}, required=('scope', 'gate_name'), writes=True)
def customise_gate(ctx: ToolContext, scope: str, gate_name: str) -> dict:
    data = ctx.ensure_scope(scope)
    path = ctx.require_sample()
    ctx.require_gate(data['gating'], gate_name)
    ctx.bus.customiseGateRequested.emit(scope, gate_name)
    return {'gate': gate_name, 'sample': path,
            'custom_for_loaded_sample': bool(data['gating'].is_custom_gate(path, gate_name))}


@tool('Revert gate to template',
      'Remove the loaded sample\'s custom copy of a gate, so the sample uses the shared gate again.',
      properties={'scope': SCOPE, 'gate_name': {'type': 'string'}}, required=('scope', 'gate_name'), writes=True)
def revert_gate(ctx: ToolContext, scope: str, gate_name: str) -> dict:
    data = ctx.ensure_scope(scope)
    path = ctx.require_sample()
    ctx.require_gate(data['gating'], gate_name)
    ctx.bus.revertGateRequested.emit(scope, gate_name)
    return {'gate': gate_name, 'sample': path,
            'custom_for_loaded_sample': bool(data['gating'].is_custom_gate(path, gate_name))}


# ---------------------------------------------------------------------------
# Spectral process
# ---------------------------------------------------------------------------

@tool('Get spectral model',
      'The spectral model (one row per fluorophore or AF control) and the state of the unmixing: method, '
      'condition number, whether an unmixing matrix exists.')
def get_spectral_model(ctx: ToolContext) -> dict:
    process = ctx.controller.experiment.process
    profiles = process.get('profiles') or {}
    rows = [{'label': c.get('label'), 'control_type': c.get('control_type'), 'particle_type': c.get('particle_type'),
             'sample': c.get('sample_name'), 'positive_gate': c.get('gate_label'),
             'major_channel': c.get('gate_channel'), 'has_spectrum': c.get('label') in profiles}
            for c in process.get('spectral_model', [])]
    return {'controls': rows, 'negative_type': process.get('negative_type'),
            'unmixing_method': ctx.controller.experiment.settings.get('unmixing_method', 'OLS'),
            'condition_number': process.get('condition_number'),
            'unmixing_matrix_available': process.get('unmixing_matrix') is not None}


@tool('Refresh unmixing',
      'Recalculate the unmixing from the current spectral model, as the Spectral Process tab does after an edit. '
      'Unmixed plots and statistics are rebuilt.',
      writes=True, timeout=900.0)
def refresh_unmixing(ctx: ToolContext) -> dict:
    if not ctx.controller.experiment.process.get('spectral_model'):
        raise ToolError('The spectral model is empty. Define the spectral controls in the Spectral Process tab first.')
    ctx.bus.spectralModelUpdated.emit()
    return get_spectral_model(ctx)
