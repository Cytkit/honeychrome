"""
ag_report.py — report items for the Automated Gating plugin
===========================================================
Builds the tickable items of the Automated Gating report as
``drc_report.ReportItem`` objects, so the report is written with the same
PNG / CSV / PDF machinery as the Honeycluster report
(``drc_report.export_report_item``). Pure matplotlib and pandas, no Qt
widgets: every figure is built without a canvas.

Sections:
  • Samples    — per-sample events, AF profiles and flags; population
                 fractions. The population counts and the marker medians
                 and quartiles are the report's own CSV files
                 (population_statistics.csv, activation_statistics.csv).
  • Gates      — one plot per gate: the parent population's events pooled
                 over the gated samples' display subsamples, with the stored
                 boundary and pooled fractions; ``gate_pages`` puts the
                 ticked gates GATES_PER_PAGE to a page.
  • Statistics — for each comparison of the last statistics run: heatmaps,
                 volcano plots (with their results tables) and the marker
                 summary; the sample PCA and the predictive analysis (ROC
                 with AUC, most informative features, panel size) when
                 they have been run.
  • Populations — for each population ticked for the report: the two
                 groups' pooled biplots side by side with their gated
                 frequencies, and a dot and violin plot of the population's
                 frequency per sample with the test's p-value.
  • Enrichment — the last Flow Set Enrichment Analysis run: the dot plot of
                 immune states with the states table, and for each
                 comparison the running-sum figure of every state below
                 the FDR threshold, with its members.

The file name does not end in ``_tab.py``, so the plugin loader does not
treat it as a tab.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

import ag_core
import ag_results
from drc_report import ReportItem
from honeychrome.view_components.differential_plots import (
    GROUP_COLOURS as _GROUP_COLOURS, make_group_values_figure,
)

SECTIONS = ('Samples', 'Gates', 'Statistics', 'Populations', 'Enrichment')

# Gate plots per PDF page (A4 landscape), as columns × rows.
GATE_PAGE_COLUMNS = 3
GATE_PAGE_ROWS = 2
GATES_PER_PAGE = GATE_PAGE_COLUMNS * GATE_PAGE_ROWS
# Above this many gates the Gates section starts unticked.
REPORT_GATES_DEFAULT_LIMIT = 50
# Larger tables are written to CSV only, with a note in the PDF.
PDF_MAX_ROWS = 500
PDF_MAX_COLUMNS = 30

__all__ = ['SECTIONS', 'GATE_PAGE_COLUMNS', 'GATE_PAGE_ROWS', 'GATES_PER_PAGE',
           'REPORT_GATES_DEFAULT_LIMIT', 'PDF_MAX_ROWS', 'PDF_MAX_COLUMNS', 'GateReportItem',
           'pooled_display', 'pooled_population', 'boundary_fractions',
           'label_position', 'make_gate_figure', 'population_frequency_table', 'sample_table',
           'build_report_items', 'gate_pages', 'make_pca_figure', 'make_roc_figure',
           'make_importance_figure', 'make_panel_figure', 'predictive_subtitle',
           'make_group_values_figure', 'make_population_comparison_figure', 'p_value_text',
           'enrichment_report_items']


@dataclass
class GateReportItem(ReportItem):
    """A Gates section item: *get_figure* draws the gate alone; *draw(ax)*
    draws it into the given axes of a page of several gates, under the
    page's *page_title*."""
    draw: Callable[[Any], Any] | None = None
    page_title: str = ''


def pooled_display(display: dict, samples: list[str], gate_name: str,
                   gate_def: dict | None = None):
    """Display events of *samples* pooled, with *gate_name*'s parent mask.

    *gate_def* (the gate's definition) is needed for display records that
    keep population memberships (see ag_results.pack_display).

    Returns (events, parent mask, channels); events is None when no sample
    has display data.
    """
    chunks, parents = [], []
    channels = None
    for key in samples:
        disp = display.get(key)
        if not disp:
            continue
        if channels is None:
            channels = disp['channels']
        chunks.append(disp['events'])
        parent = ag_results.display_parent_mask(disp, gate_name, gate_def)
        parents.append(parent if parent is not None
                       else np.zeros(len(disp['events']), dtype=bool))
    if not chunks:
        return None, None, []
    return np.concatenate(chunks), np.concatenate(parents), channels or []


def pooled_population(display: dict, samples: list[str], gate_def: dict, pop: str,
                      max_events: int | None = None, seed: int = 0):
    """Parent-population events of one gate pooled over *samples*, with the
    membership of population *pop* (as each sample was gated).

    With *max_events*, each sample contributes an equal share of at most
    that many parent events in all (a seeded subsample).

    Returns (events, parent mask, population mask, channels) over the pooled
    events; events is None when no sample has display data.
    """
    key = ag_core.population_key(gate_def['gate_name'], pop)
    have = [s for s in samples if display.get(s)]
    if not have:
        return None, None, None, []
    share = None if not max_events else max(1, int(max_events) // len(have))
    rng = np.random.default_rng(seed)
    chunks, parents, members = [], [], []
    channels = None
    for s in have:
        disp = display[s]
        if channels is None:
            channels = disp['channels']
        parent = ag_results.display_parent_mask(disp, gate_def['gate_name'], gate_def)
        member = ag_results.display_population_mask(disp, key)
        if parent is None:
            continue
        idx = np.flatnonzero(parent)
        if share is not None and len(idx) > share:
            idx = np.sort(rng.choice(idx, share, replace=False))
        chunks.append(disp['events'][idx])
        parents.append(np.ones(len(idx), dtype=bool))
        members.append(member[idx] if member is not None else np.zeros(len(idx), dtype=bool))
    if not chunks:
        return None, None, None, channels or []
    return (np.concatenate(chunks), np.concatenate(parents), np.concatenate(members),
            channels or [])


def boundary_fractions(gate_def: dict, boundary: dict, events: np.ndarray | None,
                       channels: list[str], parent_mask: np.ndarray | None,
                       transforms: dict) -> dict:
    """{pop: fraction of the parent events} for *boundary* applied to
    untransformed *events* (e.g. the stored boundary on pooled display
    events), so the fractions describe the boundary that is drawn.

    *gate_def* must be the effective definition (ag_core.effective_gate_def).
    """
    if events is None or not len(events) or not boundary:
        return {}
    mask = parent_mask if (parent_mask is not None and len(parent_mask) == len(events)) \
        else np.ones(len(events), dtype=bool)
    if not mask.any():
        return {}
    needed = ag_core.boundary_channels(gate_def, boundary)
    gate_chans = [ch for ch in needed if ch in channels]
    if len(gate_chans) != len(needed):
        return {}
    data = np.column_stack([_apply(transforms.get(ch), events[mask, channels.index(ch)])
                            for ch in gate_chans])
    limits = {ch: tuple(transforms[ch].limits) for ch in gate_chans
              if getattr(transforms.get(ch), 'limits', None) is not None}
    masks = ag_core.population_masks(gate_def, boundary, data,
                                     {ch: i for i, ch in enumerate(gate_chans)},
                                     axis_limits=limits)
    return {pop: float(m.mean()) for pop, m in masks.items()}


def label_position(gate_type: str, region: str | None, entry: dict,
                   x_lim: tuple, y_lim: tuple) -> tuple[float, float, str, str]:
    """(x, y, horizontal alignment, vertical alignment) for a population
    label: quadrant labels in their quadrant's outer corner, clear of the
    events near the thresholds (a joined region: its last quadrant in
    QUADRANT_REGIONS order; an L: the corner opposite the missing quadrant),
    1D labels mid-height in their region, polygon labels at the vertex
    mean."""
    quads = ag_core.region_quadrants(region) if gate_type == '2dsep' else ()
    if quads:
        dx = 0.03 * (x_lim[1] - x_lim[0])
        dy = 0.03 * (y_lim[1] - y_lim[0])
        corner = quads[-1]
        if len(quads) == 3:
            missing = next(q for q in ag_core.QUADRANT_REGIONS if q not in quads)
            corner = f"x{'+' if missing[1] == '-' else '-'}y{'+' if missing[3] == '-' else '-'}"
        right, top = corner[1] == '+', corner[3] == '+'
        return ((x_lim[1] - dx) if right else (x_lim[0] + dx),
                (y_lim[1] - dy) if top else (y_lim[0] + dy),
                'right' if right else 'left', 'top' if top else 'bottom')
    coords = np.asarray(entry.get('boundary') or [], dtype=float)
    if gate_type == '1dsep':
        tx = float(entry.get('threshold_x') or 0.0)
        xs = coords[:, 0] if coords.ndim == 2 and len(coords) else np.array([tx])
        return float(np.mean(xs)), 0.5 * (y_lim[0] + y_lim[1]), 'center', 'center'
    if coords.ndim == 2 and len(coords):
        return float(coords[:, 0].mean()), float(coords[:, 1].mean()), 'center', 'center'
    return 0.5 * (x_lim[0] + x_lim[1]), 0.5 * (y_lim[0] + y_lim[1]), 'center', 'center'


GATE_LABEL_FONT = 8


def _inside(value: float, lim: tuple, text: str, axis_inches: float, horizontal: bool) -> float:
    """*value* (the centre of a centred label) moved so the label's text
    stays within *lim*, the axis it sits on, which is *axis_inches* long."""
    from honeychrome.view_components import differential_plots as dp

    lines = str(text).split('\n')
    if horizontal:
        extent = max(dp.text_width_inches(line, GATE_LABEL_FONT) for line in lines) + 0.1
    else:
        extent = len(lines) * GATE_LABEL_FONT * 1.3 / 72.0 + 0.1
    span = float(lim[1]) - float(lim[0])
    half = 0.5 * extent / max(axis_inches, 0.5) * span
    if 2 * half >= span:
        return 0.5 * (lim[0] + lim[1])
    return float(min(max(value, lim[0] + half), lim[1] - half))


def _apply(tr, values: np.ndarray) -> np.ndarray:
    xform = getattr(tr, 'xform', None)
    values = np.asarray(values, dtype=float)
    return np.asarray(xform.apply(values), dtype=float) if xform is not None else values


def _limits(tr, values: np.ndarray) -> tuple[float, float]:
    lim = getattr(tr, 'limits', None)
    if lim is not None and len(lim) >= 2 and float(lim[1]) > float(lim[0]):
        return float(lim[0]), float(lim[1])
    if len(values):
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        if hi > lo:
            return lo, hi
    return 0.0, 1.0


def _set_ticks(ax, axis: str, tr):
    ticks_fn = getattr(tr, 'ticks', None)
    if not callable(ticks_fn):
        return
    try:
        ticks = ticks_fn()
    except Exception:
        return
    if not ticks:
        return
    major = [(float(p), str(lbl)) for p, lbl in ticks[1]]
    if axis == 'x':
        ax.set_xticks([p for p, _l in major], [lbl for _p, lbl in major])
    else:
        ax.set_yticks([p for p, _l in major], [lbl for _p, lbl in major])


def make_gate_figure(gate_def: dict, boundary: dict, events: np.ndarray | None,
                     channels: list[str], parent_mask: np.ndarray | None, transforms: dict,
                     axis_labels: dict, title: str, fractions: dict | None = None, ax=None):
    """Matplotlib figure of one gate: the parent population's events on the
    gate's axes, in transformed units, with its boundary.

    events     : untransformed events, columns named by *channels*.
    transforms : {channel: Transform} (display transforms, with ``xform``,
                 ``limits`` and ``ticks``).
    boundary   : {pop: entry} in transformed units.
    ax         : draw into these axes instead of a new figure.
    """
    from matplotlib.colors import LogNorm
    from matplotlib.figure import Figure

    if ax is None:
        fig = Figure(figsize=(4.6, 4.4), constrained_layout=True)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    gt = gate_def.get('gate_type', '')
    shapes = ag_core.is_shape_boundary(boundary)
    sep_y = ag_core.separation_axis(gate_def) == 'y' and not shapes
    ch_x, ch_y = ag_core.boundary_plot_channels(gate_def, boundary)
    ch_x, ch_y = ch_x or '', ch_y or ''
    tr_x = transforms.get(ch_x)
    tr_y = transforms.get(ch_y) if ch_y else None
    colour = '#1e9e32'

    x_lim = (0.0, 1.0)
    y_lim = (0.0, 1.0)
    if events is not None and len(events) and ch_x in channels:
        mask = parent_mask if (parent_mask is not None and len(parent_mask) == len(events)) \
            else np.ones(len(events), dtype=bool)
        x = _apply(tr_x, events[mask, channels.index(ch_x)])
        x_lim = _limits(tr_x, x)
        if ch_y and ch_y in channels:
            y = _apply(tr_y, events[mask, channels.index(ch_y)])
            y_lim = _limits(tr_y, y)
            h, xe, ye = np.histogram2d(x, y, bins=200, range=[x_lim, y_lim])
            if h.max() >= 1:
                ax.pcolormesh(xe, ye, np.ma.masked_equal(h.T, 0), cmap='viridis',
                              norm=LogNorm(vmin=1, vmax=max(float(h.max()), 1.0)))
            ax.set_ylim(*y_lim)
            _set_ticks(ax, 'y', tr_y)
        else:
            counts, edges = np.histogram(x, bins=200, range=x_lim)
            ax.stairs(counts, edges, fill=True, color='#6464fa', alpha=0.6)
        ax.set_xlim(*x_lim)
        _set_ticks(ax, 'x', tr_x)

    fractions = fractions or {}
    pops = gate_def.get('populations') or {}
    regions = ag_core.population_regions(gate_def, boundary)
    bbox = dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none', alpha=0.8)
    # Size of the plot area, so centred labels can be kept inside it.
    pos = ax.get_position()
    ax_w = fig.get_figwidth() * pos.width
    ax_h = fig.get_figheight() * pos.height
    for pop, entry in (boundary or {}).items():
        label = (pops.get(pop) or {}).get('label', pop)
        frac = fractions.get(pop)
        text = f"{label}\n{frac * 100:.1f}%" if frac is not None else label
        if gt == '1dsep' and not shapes:
            tx = entry.get('threshold_x')
            if tx is None:
                continue
            if sep_y:
                ax.axhline(float(tx), color=colour, linewidth=1.5)
                ly, lx, va, ha = label_position(gt, regions.get(pop), entry, y_lim, (0.0, 1.0))
                ly = _inside(ly, y_lim, text, ax_h, horizontal=False)
                ax.text(lx, ly, text, color=colour, ha=ha, va=va, fontsize=GATE_LABEL_FONT,
                        bbox=bbox, transform=ax.get_yaxis_transform())
                continue
            ax.axvline(float(tx), color=colour, linewidth=1.5)
            lx, ly, ha, va = label_position(gt, regions.get(pop), entry, x_lim, (0.0, 1.0))
            lx = _inside(lx, x_lim, text, ax_w, horizontal=True)
            ax.text(lx, ly, text, color=colour, ha=ha, va=va, fontsize=GATE_LABEL_FONT,
                    bbox=bbox, transform=ax.get_xaxis_transform())
            continue
        coords = np.asarray(entry.get('boundary') or [], dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2 or not len(coords):
            continue
        closed = np.vstack([coords, coords[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color=colour, linewidth=1.5)
        lx, ly, ha, va = label_position('free' if shapes else gt, regions.get(pop), entry,
                                        x_lim, y_lim)
        if ha == 'center':
            lx = _inside(lx, x_lim, text, ax_w, horizontal=True)
        if va == 'center':
            ly = _inside(ly, y_lim, text, ax_h, horizontal=False)
        ax.text(lx, ly, text, color=colour, ha=ha, va=va, fontsize=GATE_LABEL_FONT, bbox=bbox)

    ax.set_xlabel(axis_labels.get(ch_x, ch_x))
    ax.set_ylabel(axis_labels.get(ch_y, ch_y) if ch_y else 'Count')
    ax.set_title(title, fontsize=10)
    return fig


def population_frequency_table(summaries: dict, samples: list[str], labels: dict) -> pd.DataFrame:
    """Samples × populations, % of each population's parent."""
    rows = {}
    for s in samples:
        # An empty parent's fraction is stored as None (JSON has no NaN).
        rows[s] = {labels.get(r['key'], r['key']):
                   100.0 * r['fraction_of_parent'] if r.get('fraction_of_parent') is not None
                   else float('nan')
                   for r in (summaries.get(s) or {}).get('populations', [])}
    df = pd.DataFrame.from_dict(rows, orient='index')
    df.index.name = 'sample'
    return df


def sample_table(samples: list[str], load_info: dict, groups: dict, flags: dict) -> pd.DataFrame:
    """One row per gated sample: group, events, AF profiles and flags."""
    rows = []
    for s in samples:
        info = load_info.get(s) or {}
        rows.append({
            'sample': s,
            'group': groups.get(s, ''),
            'events_in_file': info.get('n_events_file'),
            'events_gated': info.get('n_events_kept'),
            'af_profiles': ', '.join(info.get('af_profiles') or []),
            'flags': '; '.join(flags.get(s) or []),
        })
    return pd.DataFrame(rows).set_index('sample') if rows else pd.DataFrame()


def build_report_items(*, gate_defs: list[dict], reference_boundaries: dict, modes: dict,
                       summaries: dict, samples: list[str], load_info: dict, display: dict,
                       groups: dict, flags: dict, antigens: dict, transforms: dict,
                       axis_labels: dict, stats: dict | None, pval_threshold: float,
                       fc_threshold: float, marker_threshold: float, fdr_scope: str,
                       build_figures, pca: dict | None = None,
                       predictive: dict | None = None, include_gates: bool = True,
                       populations: list | None = None,
                       population_figure=None,
                       enrichment: dict | None = None,
                       group_colours: dict | None = None,
                       trend: dict | None = None,
                       build_trend_figures=None,
                       trend_enrichment: dict | None = None) -> list[ReportItem]:
    """Every available report item, in section order.

    pca               : ag_results.sample_pca() result, or None.
    predictive        : drc_predictive.run_predictive() result, or None.
    include_gates     : one item per gate (pooled gate plots).
    populations       : [(population key, label)] ticked for the report.
    population_figure : callable(population key) -> Figure of the
                        population's group comparison.
    group_colours     : {group: colour} for the statistics figures and the
                        sample PCA (default palette when None).
    enrichment        : ag_fsea.run_enrichment() result, or None.
    trend_enrichment  : ag_fsea.run_trend_enrichment() result, or None.
    trend             : ag_results.run_trend_statistics() result, or None.
    build_trend_figures : callable(trend, is_dark, antigens, pop_labels) ->
                        {key: {'fig', …}}, e.g. the tab's build_trend_figures;
                        called once, when the first trend figure is exported.

    build_figures : callable(stats, base, other, pval, fc, marker, fdr_scope,
                    is_dark, antigens, pop_labels) -> {key: {'fig', 'title',
                    'error', 'results'}}, e.g. the tab's build_result_figures.
                    It is called at most once per comparison, when the first
                    of that comparison's figures is exported.
    """
    items: list[ReportItem] = []
    pop_labels = ag_core.gate_population_names(gate_defs)

    if summaries:
        items.append(ReportItem(
            key='samples', tab='Samples', label='Samples',
            get_tables=lambda: {'summary': sample_table(samples, load_info, groups, flags)}))
        items.append(ReportItem(
            key='frequencies', tab='Samples', label='Population frequencies',
            get_tables=lambda: {'percent_of_parent': population_frequency_table(
                summaries, samples, pop_labels)}))

    if display and include_gates:
        by_name = ag_core.gates_by_name(gate_defs)
        page_title = f"Stored boundaries; % of pooled display events, {len(samples)} samples"
        for g in ag_core.ordered_gate_defs(gate_defs):
            name = g['gate_name']
            eff = ag_core.effective_gate_def(g, by_name)
            mode = modes.get(name, ag_core.MODE_FIXED)

            def _gate_plot(title, ax=None, _eff=eff, _name=name):
                events, parent, channels = pooled_display(display, samples, _name, _eff)
                boundary = reference_boundaries.get(_name, {})
                fractions = boundary_fractions(_eff, boundary, events, channels, parent,
                                               transforms)
                return make_gate_figure(_eff, boundary, events, channels, parent, transforms,
                                        axis_labels, title, fractions=fractions, ax=ax)

            def _gate_fig(_plot=_gate_plot, _name=name, _mode=mode):
                return _plot(f"{_name} ({_mode}; stored boundary)\n"
                             f"% of pooled display events, {len(samples)} samples")

            def _gate_draw(ax, _plot=_gate_plot, _name=name, _mode=mode):
                return _plot(f"{_name} ({_mode})", ax=ax)

            items.append(GateReportItem(key=f'gate:{name}', tab='Gates', label=f"Gate {name}",
                                        get_figure=_gate_fig, draw=_gate_draw,
                                        page_title=page_title))

    if stats:
        cache: dict = {}

        def _figures(comp):
            if comp not in cache:
                colours = {'group_colours': group_colours} if group_colours else {}
                cache[comp] = build_figures(stats, comp[0], comp[1], pval_threshold,
                                            fc_threshold, marker_threshold, fdr_scope, False,
                                            antigens, pop_labels, **colours)
            return cache[comp]

        for base, other in stats.get('comparisons') or []:
            comp = (base, other)
            label = f"{other} vs {base}"
            for key, title in (('freq_heatmap', 'Frequency heatmap'),
                               ('freq_volcano', 'Frequency volcano'),
                               ('counts_heatmap', 'Counts heatmap'),
                               ('counts_volcano', 'Counts volcano'),
                               ('marker_summary', 'Marker summary'),
                               ('marker_volcano', 'Marker volcano')):
                source = {'freq_heatmap': 'freq', 'freq_volcano': 'freq',
                          'counts_heatmap': 'counts', 'counts_volcano': 'counts',
                          'marker_summary': 'markers', 'marker_volcano': 'markers'}[key]
                if stats.get(source) is None:
                    continue

                def _fig(_comp=comp, _key=key):
                    entry = _figures(_comp).get(_key) or {}
                    return entry.get('fig')

                get_tables = None
                if key.endswith('volcano'):
                    def get_tables(_comp=comp, _key=key):
                        res = (_figures(_comp).get(_key) or {}).get('results')
                        return {'results': res} if res is not None else {}

                items.append(ReportItem(key=f'{key}:{label}', tab='Statistics',
                                        label=f"{title}: {label}", get_figure=_fig,
                                        get_tables=get_tables))
    if trend and build_trend_figures is not None:
        trend_cache: dict = {}

        def _trend_figures():
            if 'figures' not in trend_cache:
                trend_cache['figures'] = build_trend_figures(trend, False, antigens, pop_labels)
            return trend_cache['figures']

        variable = trend.get('variable') or ''
        for kind, name, table_name in (('freq', 'Frequency', 'trend_frequencies'),
                                       ('counts', 'Counts', 'trend_counts'),
                                       ('markers', 'Marker', 'trend_markers')):
            if trend.get(kind) is None:
                continue
            for figure, title in (('volcano', 'trend volcano'), ('heatmap', 'trajectories'),
                                  ('patterns', 'shape patterns')):
                key = f'trend_{kind}_{figure}'

                def _fig(_key=key):
                    return (_trend_figures().get(_key) or {}).get('fig')

                def _table(_name=table_name):
                    tables = ag_results.trend_tables(trend, pop_labels, antigens)
                    return {_name: tables[_name]} if _name in tables else {}

                items.append(ReportItem(key=f'{key}:{variable}', tab='Statistics',
                                        label=f"{name} {title}: {variable}", get_figure=_fig,
                                        get_tables=_table if figure == 'volcano' else None))
        if trend.get('age_model'):
            for key, title, table_name in (
                    ('trend_age_model', 'Immune age model', 'immune_age_predictions'),
                    ('trend_age_features', 'Immune age features', 'immune_age_features')):

                def _age_fig(_key=key):
                    return (_trend_figures().get(_key) or {}).get('fig')

                def _age_table(_name=table_name):
                    tables = ag_results.trend_tables(trend, pop_labels, antigens)
                    return {_name: tables[_name]} if _name in tables else {}

                items.append(ReportItem(key=f'{key}:{variable}', tab='Statistics',
                                        label=f"{title}: {variable}", get_figure=_age_fig,
                                        get_tables=_age_table))
    if pca:
        def _pca_scores():
            scores = pca['scores'].assign(group=pca['groups'])
            if pca.get('colour_values') is not None:
                scores[pca.get('colour_label') or 'value'] = pca['colour_values']
            return scores

        items.append(ReportItem(
            key='sample_pca', tab='Statistics', label='Sample PCA',
            get_figure=lambda: make_pca_figure(pca, False, group_colours=group_colours),
            get_tables=lambda: {'scores': _pca_scores(), 'loadings': pca['loadings']}))
    if predictive:
        summary = predictive['summary']
        what = f"{summary['positive']} vs {summary['negative']}"
        items.append(ReportItem(
            key='pred_roc', tab='Statistics', label=f"Predictive ROC: {what}",
            get_figure=lambda: make_roc_figure(predictive['performance'], predictive['roc'],
                                               summary, False),
            get_tables=lambda: {'performance': predictive['performance']}))
        items.append(ReportItem(
            key='pred_features', tab='Statistics', label=f"Predictive features: {what}",
            get_figure=lambda: make_importance_figure(predictive['importance'], summary, False),
            get_tables=lambda: {'importance': predictive['importance']}))
        items.append(ReportItem(
            key='pred_panel', tab='Statistics', label=f"Predictive panel size: {what}",
            get_figure=lambda: make_panel_figure(predictive['panel'], summary, False),
            get_tables=lambda: {'panel': predictive['panel']}))
    if populations and population_figure is not None:
        for pop_key, label in populations:
            items.append(ReportItem(
                key=f'population:{pop_key}', tab='Populations', label=f"Population {label}",
                get_figure=lambda _k=pop_key: population_figure(_k)))
    if enrichment:
        items.extend(enrichment_report_items(enrichment, pval_threshold))
    if trend_enrichment:
        items.extend(enrichment_report_items(trend_enrichment, pval_threshold))
    return items


def gate_pages(items: list, per_page: int = GATES_PER_PAGE) -> list:
    """*items* with the gate items (GateReportItem) put *per_page* to a
    page: each page is one item, labelled with the range of the gates on
    it in report order (e.g. 'Gates 007-012 of 150'), whose figure has the
    gates in GATE_PAGE_COLUMNS columns. Other items are kept in place."""
    gates = [it for it in items if isinstance(it, GateReportItem)]
    if not gates:
        return list(items)
    width = len(str(len(gates)))
    pages = []
    for start in range(0, len(gates), per_page):
        chunk = gates[start:start + per_page]
        first, last = start + 1, start + len(chunk)
        pages.append(ReportItem(
            key=f"gates:{first}-{last}", tab=chunk[0].tab,
            label=f"Gates {first:0{width}d}-{last:0{width}d} of {len(gates)}",
            get_figure=lambda _chunk=chunk: _gate_page_figure(_chunk, per_page)))
    out, placed = [], False
    for it in items:
        if isinstance(it, GateReportItem):
            if not placed:
                out.extend(pages)
                placed = True
            continue
        out.append(it)
    return out


def _gate_page_figure(chunk: list, per_page: int):
    """One A4 landscape page of gate plots; a gate that cannot be drawn
    says so in its panel."""
    from matplotlib.figure import Figure

    rows = max(1, -(-per_page // GATE_PAGE_COLUMNS))
    fig = Figure(figsize=(11.69, 8.27), constrained_layout=True)
    axes = fig.subplots(rows, GATE_PAGE_COLUMNS, squeeze=False).ravel()
    for ax, item in zip(axes, chunk):
        try:
            item.draw(ax)
        except Exception as exc:
            ax.clear()
            ax.axis('off')
            ax.text(0.5, 0.5, f"{item.label}: could not be drawn ({exc})", ha='center',
                    va='center', wrap=True, transform=ax.transAxes)
    for ax in axes[len(chunk):]:
        ax.axis('off')
    if chunk and chunk[0].page_title:
        fig.suptitle(chunk[0].page_title, fontsize=10)
    return fig


def enrichment_report_items(result: dict, fdr_threshold: float) -> list[ReportItem]:
    """Report items of an enrichment run: the dot plot with every table,
    and the running-sum and member figures of each state below
    *fdr_threshold* in each comparison."""
    import ag_fsea
    import ag_fsea_plots as fp

    comps = ag_fsea.comparison_labels(result)
    trend = (result.get('settings') or {}).get('kind') == 'trend'
    items = [ReportItem(
        key='enrichment:trend' if trend else 'enrichment', tab='Enrichment',
        label=f"Immune state enrichment: {comps[0]}" if trend and comps
        else 'Immune state enrichment',
        get_figure=lambda: fp.make_dot_plot(result['states'], comps, fdr_threshold, False),
        get_tables=lambda: ag_fsea.result_tables(result))]
    states = result['states']
    sig = states[states['FDR'] < fdr_threshold]
    for _i, row in sig.iterrows():
        comp, name = row['comparison'], row['immune_state']
        members = result['members']
        mb = members[(members['comparison'] == comp) & (members['immune_state'] == name)]

        def _curve(_c=comp, _n=name, _r=row.to_dict()):
            return fp.make_curve_figure(ag_fsea.state_curve(result, _c, _n), _n, _c, _r, False)

        items.append(ReportItem(
            key=f'enrichment:{comp}:{name}', tab='Enrichment', label=f"{name}: {comp}",
            get_figure=_curve, get_tables=lambda _m=mb: {'members': _m}))
    pattern_states = result.get('pattern_states')
    if pattern_states is not None and len(pattern_states):
        items.append(ReportItem(
            key='enrichment:patterns', tab='Enrichment', label='Immune states in shape patterns',
            get_figure=lambda: fp.make_pattern_dot_plot(pattern_states, fdr_threshold, False),
            get_tables=lambda: {'pattern_states': pattern_states}))
    return items


# ---------------------------------------------------------------------------
# Population group comparison
# ---------------------------------------------------------------------------

def p_value_text(result: dict | None) -> str:
    """'p = …, FDR = …' for a population's test result ('' when untested)."""
    if not result:
        return 'not tested'
    text = f"p = {result['P']:.3g}"
    if result.get('FDR') is not None:
        text += f", FDR = {result['FDR']:.3g}"
    return text


def make_population_comparison_figure(gate_def: dict, pop: str, panels: list, transforms: dict,
                                      axis_labels: dict, values: dict, group_order: list[str],
                                      title: str, p_text: str = '',
                                      group_colours: dict | None = None,
                                      trend: dict | None = None,
                                      trend_percent: dict | None = None):
    """The report figure of one population: each group's pooled parent
    events on the gate's axes with the reference boundary and the group's
    gated frequency, then the dot and violin plot of the per-sample values
    (coloured by *group_colours*, {group: colour}, when given).

    panels : [{'group', 'events', 'parent', 'member', 'channels',
               'boundary', 'n_samples'}], one per group.
    trend  : keyword arguments of trend_plots.make_trend_plot_figure; the
             population's trend plot then takes the place of the dot and
             violin plot (two panels for a paired group × trend test).
    trend_percent : the same trend on the % of parent scale, drawn after it.
    """
    from matplotlib.figure import Figure

    two_panels = bool(trend and trend.get('interaction') and not trend.get('group_curves'))
    n_last = (2 if two_panels else 1) + (1 if trend_percent else 0) if trend else 1
    n = len(panels) + n_last
    last_w = 4.6 if trend else 3.9
    fig = Figure(figsize=(3.9 * len(panels) + last_w * n_last, 4.4), constrained_layout=True)
    axes = fig.subplots(1, n, squeeze=False,
                        width_ratios=[3.9] * len(panels) + [last_w] * n_last)[0]
    label = ((gate_def.get('populations') or {}).get(pop) or {}).get('label', pop)
    for ax, panel in zip(axes, panels):
        member = panel.get('member')
        frac = float(member.mean()) if member is not None and len(member) else None
        make_gate_figure(gate_def, panel.get('boundary') or {}, panel.get('events'),
                         panel.get('channels') or [], panel.get('parent'), transforms,
                         axis_labels,
                         f"{panel['group']} ({panel.get('n_samples', 0)} samples pooled)",
                         fractions={pop: frac} if frac is not None else None, ax=ax)
    if trend:
        from honeychrome.view_components import trend_plots as tp
        n_trend = 2 if two_panels else 1
        tp.make_trend_plot_figure(**trend, axes=list(axes[len(panels):len(panels) + n_trend]))
        if trend_percent:
            tp.make_trend_plot_figure(**trend_percent, axes=[axes[len(panels) + n_trend]])
    else:
        make_group_values_figure(values, group_order, f"{label}: % of parent", p_text,
                                 ax=axes[-1], group_colours=group_colours)
    fig.suptitle(title, fontsize=10)
    return fig


# ---------------------------------------------------------------------------
# Sample PCA and predictive analysis figures
# ---------------------------------------------------------------------------

def _group_colours(groups: list[str]) -> dict:
    order = list(dict.fromkeys(groups))
    return {g: _GROUP_COLOURS[i % len(_GROUP_COLOURS)] for i, g in enumerate(order)}


def make_pca_figure(pca: dict, is_dark: bool, sample_labels: dict | None = None,
                    show_loadings: bool = True, label_points: bool = False,
                    group_colours: dict | None = None):
    """Samples on PC1 × PC2 coloured by group, with the top loadings as
    arrows; hovering names a sample or loading (fig._hover_handler).
    *group_colours* ({group: colour}) overrides the default palette. When
    *pca* has 'colour_values' (one number per sample, e.g. age), samples
    are coloured by that value instead, with a colour bar named
    'colour_label', and shaped by group."""
    from matplotlib.figure import Figure
    from honeychrome.view_components import differential_plots as dp

    with_bar = pca.get('colour_values') is not None
    fig = Figure(figsize=(6.9 if with_bar else 6, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.set_box_aspect(1)  # the plot area itself is square, whatever surrounds it
    fg = 'white' if is_dark else 'black'
    scores, loadings, groups = pca['scores'], pca['loadings'], pca['groups']
    colours = {g: (group_colours or {}).get(g) or c for g, c in _group_colours(groups).items()}
    names = [(sample_labels or {}).get(s, str(s).replace('\\', '/').rsplit('/', 1)[-1])
             for s in scores.index]
    if pca.get('colour_values') is not None:
        from honeychrome.view_components import trend_plots as tp
        tp.scatter_by_value(fig, ax, scores['PC1'].values, scores['PC2'].values,
                            pca['colour_values'], groups, label=pca.get('colour_label') or 'value',
                            is_dark=is_dark)
    else:
        for grp, colour in colours.items():
            mask = np.array([g == grp for g in groups])
            ax.scatter(scores['PC1'].values[mask], scores['PC2'].values[mask], s=60, color=colour,
                       edgecolors=fg, linewidths=0.4, label=grp or 'no group', zorder=3)
    hover_xy = list(zip(scores['PC1'].values, scores['PC2'].values))
    hover_labels = list(names)
    if label_points:
        for x, y, name in zip(scores['PC1'].values, scores['PC2'].values, names):
            ax.annotate(name, xy=(x, y), xytext=(4, 4), textcoords='offset points',
                        fontsize=6, color=fg, zorder=5)
    if show_loadings and len(loadings):
        score_max = max(float(scores[['PC1', 'PC2']].abs().values.max()), 1e-6)
        load_max = max(float(loadings[['PC1', 'PC2']].abs().values.max()), 1e-12)
        ends = loadings[['PC1', 'PC2']].values * (0.8 * score_max / load_max)
        for feat, (x, y) in zip(loadings.index, ends):
            ax.annotate('', xy=(x, y), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='-|>', color='#7fa7c9', lw=1.1), zorder=4)
            ax.annotate(str(feat), xy=(x, y), xytext=(3, 3), textcoords='offset points',
                        fontsize=6, color=fg, zorder=5)
            for t in np.linspace(0.2, 1.0, 6):
                hover_xy.append((x * t, y * t))
                hover_labels.append(str(feat))
        all_x = np.concatenate([scores['PC1'].values, ends[:, 0], [0.0]])
        all_y = np.concatenate([scores['PC2'].values, ends[:, 1], [0.0]])
        pad_x = max(float(np.ptp(all_x)), 1e-6) * 0.15
        pad_y = max(float(np.ptp(all_y)), 1e-6) * 0.15
        ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
        ax.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)
    hover = np.asarray(hover_xy, dtype=float)
    overlay = ax.scatter(hover[:, 0], hover[:, 1], s=120, alpha=0, zorder=6)
    fig._hover_handler = dp.make_scatter_hover_handler(fig, ax, overlay, hover_labels, is_dark)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.6, zorder=1)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.6, zorder=1)
    ev = pca['explained_variance_ratio']
    ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}% variance)")
    ax.set_title(f"Sample PCA ({' + '.join(pca['sources'])})", fontsize=10)
    ax.legend(fontsize=7, loc='best')
    dp.style_figure_theme(fig, is_dark)
    return fig


def predictive_subtitle(summary: dict) -> str:
    import drc_predictive
    n = ", ".join(f"{g} n={k}" for g, k in summary['n_per_group'].items())
    text = (f"{summary['positive']} vs {summary['negative']} ({n}); "
            f"{summary['repeats']}× repeated {summary['outer_folds']}-fold CV")
    if summary.get('status') == 'warn':
        text += (f"\nfewer than {drc_predictive.RECOMMENDED_PER_GROUP} per group: "
                 "estimates are noisy")
    return text


def make_roc_figure(performance: pd.DataFrame, roc: pd.DataFrame, summary: dict, is_dark: bool):
    """Mean cross-validated ROC curve per model; AUC ± SD and balanced
    accuracy over the repeats in the legend."""
    from matplotlib.figure import Figure
    from honeychrome.view_components import differential_plots as dp

    fig = Figure(figsize=(5.5, 5), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=0.8)
    for _i, row in performance.iterrows():
        curve = roc[roc['model'] == row['model']]
        ax.plot(curve['fpr'], curve['tpr'], linewidth=1.6,
                label=(f"{row['model']}: AUC {row['auc_mean']:.2f} ± {row['auc_sd']:.2f}, "
                       f"bal. acc. {row['balanced_accuracy_mean']:.2f}"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc='lower right', fontsize=7)
    ax.set_title(dp.wrap_to_width(f"Cross-validated ROC\n{predictive_subtitle(summary)}", 9, 5.0),
                 fontsize=9)
    dp.style_figure_theme(fig, is_dark)
    return fig


def make_importance_figure(importance: pd.DataFrame, summary: dict, is_dark: bool,
                           top_n: int = 20):
    """The most informative features: lasso selection frequency and/or
    forest importance, coloured by the group with the higher value."""
    from matplotlib.figure import Figure
    import matplotlib.patches as mpatches
    from honeychrome.view_components import differential_plots as dp

    metrics = [(c, lbl) for c, lbl in (
        ('lasso_selection', 'Lasso: fraction of fits selecting'),
        ('forest_importance', 'Random forest: mean importance'),
    ) if c in importance.columns]
    top = importance.head(top_n).iloc[::-1]
    colours = ['#EE6677' if d > 0 else '#4477AA' for d in top['direction']]
    fig = Figure(figsize=(4.5 * max(1, len(metrics)) + 2.5, max(3.5, 0.28 * len(top) + 1.8)),
                 constrained_layout=True)
    for i, (col, label) in enumerate(metrics):
        ax = fig.add_subplot(1, len(metrics), i + 1)
        xerr = top['forest_importance_sd'] if col == 'forest_importance' else None
        ax.barh(range(len(top)), top[col], color=colours, xerr=xerr, error_kw={'elinewidth': 0.6})
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top['feature'] if i == 0 else [''] * len(top), fontsize=7)
        ax.set_xlabel(label, fontsize=8)
    fig.legend(handles=[
        mpatches.Patch(color='#EE6677', label=f"higher in {summary['positive']}"),
        mpatches.Patch(color='#4477AA', label=f"higher in {summary['negative']}"),
    ], loc='outside lower center', ncol=2, fontsize=7)
    fig.suptitle(f"Most informative features\n{predictive_subtitle(summary)}", fontsize=9)
    dp.style_figure_theme(fig, is_dark)
    return fig


def make_panel_figure(panel: pd.DataFrame, summary: dict, is_dark: bool):
    """Cross-validated AUC against the number of top-ranked features."""
    from matplotlib.figure import Figure
    from honeychrome.view_components import differential_plots as dp

    fig = Figure(figsize=(5.5, 4.5), constrained_layout=True)
    ax = fig.add_subplot(111)
    x, m, sd = panel['n_features'].values, panel['auc_mean'].values, panel['auc_sd'].values
    ax.fill_between(x, m - sd, m + sd, alpha=0.25)
    ax.plot(x, m, marker='o', markersize=3, linewidth=1.4)
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(summary['minimal_panel'], color='grey', linestyle=':', linewidth=1.0)
    ax.annotate(f"minimal panel: {summary['minimal_panel']}", xy=(summary['minimal_panel'], 0.5),
                xytext=(4, 4), textcoords='offset points', fontsize=7)
    if len(x) > 1:
        ax.set_xscale('log')
    ax.set_xlabel("Number of top-ranked features (log scale)")
    ax.set_ylabel("Cross-validated AUC (mean ± SD)")
    ax.set_ylim(0, 1.02)
    ax.set_title(dp.wrap_to_width(f"How many features are needed?\n{predictive_subtitle(summary)}",
                                  9, 5.0), fontsize=9)
    dp.style_figure_theme(fig, is_dark)
    return fig
