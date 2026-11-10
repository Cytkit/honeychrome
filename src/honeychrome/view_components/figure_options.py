"""
figure_options.py — appearance options for the plugins' figures
==============================================================

Every figure a plugin shows (Automated Gating, Honeycluster, FlowCode) can
be adjusted without changing what it shows: grid lines, the run label in
the corner, the title (as drawn, hidden or replaced), panel titles, the
font size of the title, the axis and panel labels, the tick labels, the
legend and other text, the background (the theme's, white, black or
transparent) and, on figures that offer it (volcano plots), the range of
the x and y axes.

Options belong to the experiment: they are kept in the experiment's folder
(``cache/figure_options/state.json``, see ``experiment_state``), never in
QSettings. Each plugin has its own *scope* (e.g. ``'automated_gating'``)
with defaults for all its figures and, per figure (a stable *key* such as
``'trend_freq_volcano'``), the options that figure overrides. The title
text and the axis ranges are always per figure.

``apply_figure_options`` changes a figure in place and can be called again
with other options: what the figure was drawn with is remembered on the
figure the first time, so "as drawn" restores it. It is applied whenever a
figure is shown, popped out, exported or put into a report, so all of them
match. Matplotlib only; no Qt.
"""

from __future__ import annotations

import json
import logging

import numpy as np
from dataclasses import asdict, dataclass, fields
from pathlib import Path

log = logging.getLogger(__name__)

__all__ = ['FigureOptions', 'FigureOptionsStore', 'FigureOptionsRef', 'apply_figure_options',
           'store_for', 'with_options', 'has_axis_range', 'TITLE_MODES', 'BACKGROUNDS',
           'PER_FIGURE_ONLY', 'AXIS_RANGE']

TITLE_MODES = ('auto', 'hidden', 'custom')
BACKGROUNDS = ('theme', 'white', 'black', 'transparent')
# Axis range options: text, blank for the end as drawn.
AXIS_RANGE = ('x_min', 'x_max', 'y_min', 'y_max')
# Options that only ever apply to one figure (never copied to the defaults).
PER_FIGURE_ONLY = ('title', 'title_text') + AXIS_RANGE
STORE_FOLDER = 'figure_options'
STORE_KEY = 'options'


@dataclass
class FigureOptions:
    """How one figure is drawn. A size of 0 keeps the size it was drawn
    with; a blank axis range end keeps that end as drawn."""
    grid: bool = False
    run_label: bool = True
    title: str = 'auto'
    title_text: str = ''
    panel_titles: bool = True
    title_size: float = 0.0
    label_size: float = 0.0
    tick_size: float = 0.0
    legend_size: float = 0.0
    text_size: float = 0.0
    background: str = 'theme'
    x_min: str = ''
    x_max: str = ''
    y_min: str = ''
    y_max: str = ''

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict | None) -> 'FigureOptions':
        out = cls()
        for f in fields(cls):
            if values and f.name in values:
                value = values[f.name]
                kind = type(getattr(out, f.name))
                if kind is bool:
                    value = (value if isinstance(value, bool)
                             else str(value).strip().lower() in ('true', '1', 'yes'))
                else:
                    try:
                        value = kind(value)
                    except (TypeError, ValueError):
                        continue
                setattr(out, f.name, value)
        if out.title not in TITLE_MODES:
            out.title = 'auto'
        if out.background not in BACKGROUNDS:
            out.background = 'theme'
        return out


# ---------------------------------------------------------------------------
# Storage with the experiment
# ---------------------------------------------------------------------------

class FigureOptionsStore:
    """
    The figure options of one experiment. *backing* is an
    ``experiment_state.ExperimentStateStore`` (or any object with value,
    setValue and sync); None keeps them in memory only.
    """

    def __init__(self, backing=None):
        self._backing = backing
        raw = backing.value(STORE_KEY, '') if backing is not None else ''
        try:
            data = json.loads(raw) if isinstance(raw, str) and raw else (raw or {})
        except (TypeError, ValueError):
            data = {}
        self._data: dict = data if isinstance(data, dict) else {}
        self._listeners: list = []

    def _scope(self, scope: str) -> dict:
        return self._data.setdefault(str(scope), {'defaults': {}, 'figures': {}})

    def defaults(self, scope: str) -> FigureOptions:
        """The options of a figure of *scope* that sets none of its own."""
        return FigureOptions.from_dict(self._scope(scope).get('defaults'))

    def options(self, scope: str, key: str) -> FigureOptions:
        """The options of figure *key*: the scope's defaults with the
        figure's own values over them."""
        values = dict(self._scope(scope).get('defaults') or {})
        values.update(self._scope(scope).get('figures', {}).get(str(key)) or {})
        return FigureOptions.from_dict(values)

    def has_own(self, scope: str, key: str) -> bool:
        return bool(self._scope(scope).get('figures', {}).get(str(key)))

    def set(self, scope: str, key: str, options: FigureOptions):
        """Keep *options* for figure *key*: the values that differ from the
        scope's defaults (and its title, always its own)."""
        base = self.defaults(scope).to_dict()
        own = {k: v for k, v in options.to_dict().items()
               if k in PER_FIGURE_ONLY or v != base.get(k)}
        if own.get('title') == 'auto' and not own.get('title_text'):
            own.pop('title', None)
            own.pop('title_text', None)
        for name in AXIS_RANGE:
            if not str(own.get(name, '')).strip():
                own.pop(name, None)
        figures = self._scope(scope).setdefault('figures', {})
        if own:
            figures[str(key)] = own
        else:
            figures.pop(str(key), None)
        self._changed(scope)

    def set_defaults(self, scope: str, options: FigureOptions):
        """Make *options* (all but the title and axis ranges) the defaults
        of every figure of *scope*; figures keep only their own titles and
        axis ranges."""
        values = {k: v for k, v in options.to_dict().items() if k not in PER_FIGURE_ONLY}
        entry = self._scope(scope)
        entry['defaults'] = values
        for key, own in list(entry.get('figures', {}).items()):
            kept = {k: v for k, v in own.items() if k in PER_FIGURE_ONLY}
            if kept:
                entry['figures'][key] = kept
            else:
                del entry['figures'][key]
        self._changed(scope)

    def reset(self, scope: str, key: str):
        """Figure *key* back to the scope's defaults (its title as drawn)."""
        self._scope(scope).get('figures', {}).pop(str(key), None)
        self._changed(scope)

    def subscribe(self, callback):
        """Call ``callback(scope)`` whenever options of a scope change."""
        if callback not in self._listeners:
            self._listeners.append(callback)

    def unsubscribe(self, callback):
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _changed(self, scope: str):
        if self._backing is not None:
            self._backing.setValue(STORE_KEY, json.dumps(self._data))
            self._backing.sync()
        for callback in list(self._listeners):
            try:
                callback(scope)
            except RuntimeError:
                # a widget that subscribed has been deleted
                self.unsubscribe(callback)
            except Exception:
                log.exception("figure options listener failed")


_STORES: dict = {}


def store_for(controller) -> FigureOptionsStore | None:
    """The figure options of the experiment *controller* shows (opened once
    per experiment and shared by the plugins), or None without one."""
    exp_dir = getattr(controller, 'experiment_dir', None) if controller is not None else None
    experiment = getattr(controller, 'experiment', None) if controller is not None else None
    if not exp_dir or experiment is None:
        return None
    key = (str(Path(exp_dir)), str(getattr(experiment, 'experiment_id', '') or ''))
    store = _STORES.get(key)
    if store is None:
        from honeychrome.controller_components import experiment_state
        try:
            backing = experiment_state.open_store(controller, STORE_FOLDER, experiment_dir=exp_dir)
        except Exception:
            log.exception("could not open the figure options of %s", exp_dir)
            backing = None
        store = FigureOptionsStore(backing)
        _STORES.clear()
        _STORES[key] = store
    return store


@dataclass
class FigureOptionsRef:
    """Where one figure's options live: a store, the plugin's scope and the
    figure's key."""
    store: FigureOptionsStore
    scope: str
    key: str

    def options(self) -> FigureOptions:
        return self.store.options(self.scope, self.key)

    def apply(self, fig):
        """*fig* with this figure's options applied (returned for chaining)."""
        if fig is not None:
            apply_figure_options(fig, self.options())
        return fig


def with_options(make, ref: 'FigureOptionsRef | None'):
    """A figure maker whose figures get *ref*'s options (the maker itself
    when *ref* is None). Keyword arguments are passed through."""
    if ref is None:
        return make

    def _make(*args, **kwargs):
        return ref.apply(make(*args, **kwargs))
    return _make


# ---------------------------------------------------------------------------
# Applying options to a figure
# ---------------------------------------------------------------------------

def has_axis_range(fig) -> bool:
    """Whether *fig* offers the axis range options: its maker names the
    plot area they apply to as ``fig._fo_range_axes``."""
    return getattr(fig, '_fo_range_axes', None) is not None


def _range_value(text) -> float | None:
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _axis_range(fig, options: FigureOptions):
    """Set the range of the figure's range axes: each end given in
    *options*, the others as drawn. A range whose low end is not below its
    high end is shown as drawn."""
    ax = getattr(fig, '_fo_range_axes', None)
    if ax is None or not hasattr(ax, '_fo_xlim0'):
        return
    for drawn, low, high, setter in ((ax._fo_xlim0, options.x_min, options.x_max, ax.set_xlim),
                                     (ax._fo_ylim0, options.y_min, options.y_max, ax.set_ylim)):
        lo, hi = _range_value(low), _range_value(high)
        lo = drawn[0] if lo is None else lo
        hi = drawn[1] if hi is None else hi
        if not lo < hi:
            lo, hi = drawn
        setter(lo, hi)


def _is_colorbar(ax) -> bool:
    return hasattr(ax, '_colorbar') or getattr(ax, '_colorbar_info', None) is not None


def _main_title(fig):
    """The figure's title: its suptitle, else the title of its largest plot
    area with one."""
    suptitle = getattr(fig, '_suptitle', None)
    if suptitle is not None and (getattr(suptitle, '_fo_text0', None) or suptitle.get_text()):
        return suptitle
    best, area = None, -1.0
    for ax in fig.axes:
        if _is_colorbar(ax):
            continue
        title = ax.title
        if not (getattr(title, '_fo_text0', None) or title.get_text()):
            continue
        pos = ax.get_position()
        if pos.width * pos.height > area:
            best, area = title, pos.width * pos.height
    return best


def _remember(fig):
    """What the figure was drawn with, kept on its artists the first time."""
    if getattr(fig, '_fo_drawn', False):
        return
    from matplotlib.collections import Collection
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    fig._fo_facecolor0 = fig.get_facecolor()
    fig._fo_dark0 = sum(fig._fo_facecolor0[:3]) / 3.0 < 0.5
    for ax in fig.axes:
        ax._fo_facecolor0 = ax.get_facecolor()
        ticks = ax.get_xticklabels() + ax.get_yticklabels()
        ax._fo_tick0 = ticks[0].get_fontsize() if ticks else None
        ax._fo_grid0 = any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines())
    for text in fig.findobj(Text):
        text._fo_size0 = text.get_fontsize()
        text._fo_color0 = text.get_color()
        text._fo_text0 = text.get_text()
        text._fo_visible0 = text.get_visible()
    for line in fig.findobj(Line2D):
        line._fo_colors0 = (line.get_color(), line.get_markeredgecolor(),
                            line.get_markerfacecolor())
    for coll in fig.findobj(Collection):
        coll._fo_colors0 = (np.array(coll.get_edgecolor(), copy=True),
                            np.array(coll.get_facecolor(), copy=True))
    range_axes = getattr(fig, '_fo_range_axes', None)
    if range_axes is not None:
        range_axes._fo_xlim0 = tuple(range_axes.get_xlim())
        range_axes._fo_ylim0 = tuple(range_axes.get_ylim())
    fig._fo_main_title = _main_title(fig)
    fig._fo_drawn = True


def _text_groups(fig) -> dict:
    """The figure's text by role: title, labels (axis and panel), legend,
    other."""
    main = getattr(fig, '_fo_main_title', None)
    run_label = getattr(fig, '_run_label', None)
    labels, legend, other, panel_titles = [], [], [], []
    for name in ('_supxlabel', '_supylabel'):
        artist = getattr(fig, name, None)
        if artist is not None:
            labels.append(artist)
    legends = list(fig.legends)
    for ax in fig.axes:
        labels.extend([ax.xaxis.label, ax.yaxis.label])
        for title in (ax.title, ax._left_title, ax._right_title):
            if title is not main and getattr(title, '_fo_text0', title.get_text()):
                panel_titles.append(title)
        other.extend(t for t in ax.texts)
        if ax.get_legend() is not None:
            legends.append(ax.get_legend())
    for lg in legends:
        legend.extend(lg.get_texts())
        if lg.get_title() is not None:
            legend.append(lg.get_title())
    figure_level = {id(a) for a in (run_label, main, getattr(fig, '_suptitle', None),
                                    getattr(fig, '_supxlabel', None),
                                    getattr(fig, '_supylabel', None)) if a is not None}
    other.extend(t for t in fig.texts if id(t) not in figure_level)
    return {'title': [main] if main is not None else [], 'labels': labels + panel_titles,
            'panel_titles': panel_titles, 'legend': legend, 'other': other}


def _size(text, size: float):
    size0 = getattr(text, '_fo_size0', None)
    if size and size > 0:
        text.set_fontsize(size)
    elif size0 is not None:
        text.set_fontsize(size0)


def _background(fig, choice: str):
    """Background and foreground: 'theme' restores what the figure was drawn
    with; 'white' and 'black' restyle it (text of the old foreground colour
    follows); 'transparent' clears the backgrounds and keeps the theme's
    text colours."""
    from matplotlib.colors import to_rgba
    from matplotlib.text import Text
    from honeychrome.view_components import differential_plots as dp

    dark0 = bool(getattr(fig, '_fo_dark0', False))
    last = getattr(fig, '_fo_background', 'theme')
    fig._fo_background = choice
    if choice == 'theme' and last == 'theme':
        return
    if choice in ('white', 'black'):
        dark = choice == 'black'
        fg0 = to_rgba('white' if dark0 else 'black')
        fg = 'white' if dark else 'black'
        dp.style_figure_theme(fig, dark)
        for text in fig.findobj(Text):
            if text is getattr(fig, '_run_label', None):
                continue
            color0 = getattr(text, '_fo_color0', None)
            if color0 is not None and to_rgba(color0) == fg0:
                text.set_color(fg)
        _swap_marks(fig, fg0, to_rgba(fg))
        bg = dp.figure_background(dark)
        for lg in list(fig.legends) + [ax.get_legend() for ax in fig.axes if ax.get_legend()]:
            lg.get_frame().set_facecolor(bg)
        return
    if last in ('white', 'black'):
        dp.style_figure_theme(fig, dark0)
        for text in fig.findobj(Text):
            color0 = getattr(text, '_fo_color0', None)
            if color0 is not None:
                text.set_color(color0)
        _restore_marks(fig)
    fig.patch.set_facecolor(getattr(fig, '_fo_facecolor0', fig.get_facecolor()))
    for ax in fig.axes:
        ax.set_facecolor(getattr(ax, '_fo_facecolor0', ax.get_facecolor()))
    if choice == 'transparent':
        fig.patch.set_facecolor('none')
        for ax in fig.axes:
            ax.set_facecolor('none')


def _swap_marks(fig, old, new):
    """Lines, markers and edges drawn in the old foreground colour (zero
    lines, open markers, legend keys) take the new one."""
    from matplotlib.collections import Collection
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D

    def same(colour):
        try:
            return to_rgba(colour) == old
        except (TypeError, ValueError):
            return False

    for line in fig.findobj(Line2D):
        colours = getattr(line, '_fo_colors0', None)
        if colours is None:
            continue
        line.set_color(new if same(colours[0]) else colours[0])
        line.set_markeredgecolor(new if same(colours[1]) else colours[1])
        line.set_markerfacecolor(new if same(colours[2]) else colours[2])
    for coll in fig.findobj(Collection):
        colours = getattr(coll, '_fo_colors0', None)
        if colours is None:
            continue
        edges, faces = colours
        if len(edges):
            coll.set_edgecolor([new if same(tuple(c)) else tuple(c) for c in edges])
        if len(faces) and all(same(tuple(c)) for c in faces):
            coll.set_facecolor(new)


def _restore_marks(fig):
    from matplotlib.collections import Collection
    from matplotlib.lines import Line2D

    for line in fig.findobj(Line2D):
        colours = getattr(line, '_fo_colors0', None)
        if colours is not None:
            line.set_color(colours[0])
            line.set_markeredgecolor(colours[1])
            line.set_markerfacecolor(colours[2])
    for coll in fig.findobj(Collection):
        colours = getattr(coll, '_fo_colors0', None)
        if colours is not None:
            coll.set_edgecolor(colours[0])
            coll.set_facecolor(colours[1])


def apply_figure_options(fig, options: FigureOptions):
    """
    Change *fig* in place to *options* (see FigureOptions); can be called
    again with other options. Figure-level text (axis labels, ticks,
    legends) keeps its own colours unless the background is changed to
    white or black. The run label band follows dp.place_run_label.
    """
    from honeychrome.view_components import differential_plots as dp

    _remember(fig)
    fig._fo_options = options
    for ax in fig.axes:
        if ax.name == 'polar' or _is_colorbar(ax):
            continue
        ax.grid(bool(options.grid), which='major')

    groups = _text_groups(fig)
    main = groups['title'][0] if groups['title'] else None
    if main is not None:
        text0 = getattr(main, '_fo_text0', main.get_text())
        if options.title == 'hidden':
            main.set_text('')
        elif options.title == 'custom' and options.title_text.strip():
            main.set_text(options.title_text)
        else:
            main.set_text(text0)
    for title in groups['panel_titles']:
        title.set_text(getattr(title, '_fo_text0', title.get_text())
                       if options.panel_titles else '')

    for text in groups['title']:
        _size(text, options.title_size)
    for text in groups['labels']:
        _size(text, options.label_size)
    for text in groups['legend']:
        _size(text, options.legend_size)
    for text in groups['other']:
        _size(text, options.text_size)
    for ax in fig.axes:
        size0 = getattr(ax, '_fo_tick0', None)
        size = options.tick_size if options.tick_size and options.tick_size > 0 else size0
        if size:
            ax.tick_params(labelsize=size)

    _background(fig, options.background)
    _axis_range(fig, options)
    dp.place_run_label(fig, bool(options.run_label))
    return fig
