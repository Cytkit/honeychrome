"""
ag_report.py — report items for the Automated Gating plugin
===========================================================
Builds the tickable items of the Automated Gating report as
``drc_report.ReportItem`` objects, so the report is written with the same
PNG / CSV / PDF machinery as the DR/Clustering report
(``drc_report.export_report_item``). Pure matplotlib and pandas, no Qt
widgets: every figure is built without a canvas.

Sections:
  • Samples    — per-sample events, AF profiles and flags; population
                 counts and fractions; marker medians and quartiles.
  • Gates      — one plot per gate: the parent population's events pooled
                 over the gated samples' display subsamples, with the stored
                 boundary and pooled fractions.
  • Statistics — for each comparison of the last statistics run: heatmaps,
                 volcano plots (with their results tables) and the marker
                 summary.

The file name does not end in ``_tab.py``, so the plugin loader does not
treat it as a tab.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import ag_core
import ag_results
from drc_report import ReportItem

SECTIONS = ('Samples', 'Gates', 'Statistics')

__all__ = ['SECTIONS', 'pooled_display', 'boundary_fractions', 'label_position',
           'make_gate_figure', 'population_frequency_table', 'sample_table', 'build_report_items']


def pooled_display(display: dict, samples: list[str], gate_name: str):
    """Display events of *samples* pooled, with *gate_name*'s parent mask.

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
        parents.append(disp['parent_masks'].get(gate_name, np.zeros(len(disp['events']), dtype=bool)))
    if not chunks:
        return None, None, []
    return np.concatenate(chunks), np.concatenate(parents), channels or []


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
    gate_chans = [ch for ch in ag_core.gate_channels(gate_def) if ch in channels]
    if len(gate_chans) != len(ag_core.gate_channels(gate_def)):
        return {}
    data = np.column_stack([_apply(transforms.get(ch), events[mask, channels.index(ch)])
                            for ch in gate_chans])
    masks = ag_core.population_masks(gate_def, boundary, data,
                                     {ch: i for i, ch in enumerate(gate_chans)})
    return {pop: float(m.mean()) for pop, m in masks.items()}


def label_position(gate_type: str, region: str | None, entry: dict,
                   x_lim: tuple, y_lim: tuple) -> tuple[float, float, str, str]:
    """(x, y, horizontal alignment, vertical alignment) for a population
    label: quadrant labels in their quadrant's outer corner (clear of the
    events near the thresholds), 1D labels mid-height in their region,
    polygon labels at the vertex mean."""
    if gate_type == '2dsep' and region in ag_core.QUADRANT_REGIONS:
        dx = 0.03 * (x_lim[1] - x_lim[0])
        dy = 0.03 * (y_lim[1] - y_lim[0])
        right, top = region.startswith('x+'), region.endswith('y+')
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
                     axis_labels: dict, title: str, fractions: dict | None = None):
    """Matplotlib figure of one gate: the parent population's events on the
    gate's axes, in transformed units, with its boundary.

    events     : untransformed events, columns named by *channels*.
    transforms : {channel: Transform} (display transforms, with ``xform``,
                 ``limits`` and ``ticks``).
    boundary   : {pop: entry} in transformed units.
    """
    from matplotlib.colors import LogNorm
    from matplotlib.figure import Figure

    fig = Figure(figsize=(4.6, 4.4), constrained_layout=True)
    ax = fig.add_subplot(111)
    gt = gate_def.get('gate_type', '')
    ch_x = gate_def.get('gate_marker_x') or ''
    ch_y = gate_def.get('gate_marker_y') or ''
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
    for pop, entry in (boundary or {}).items():
        label = (pops.get(pop) or {}).get('label', pop)
        frac = fractions.get(pop)
        text = f"{label}\n{frac * 100:.1f}%" if frac is not None else label
        if gt == '1dsep':
            tx = entry.get('threshold_x')
            if tx is None:
                continue
            ax.axvline(float(tx), color=colour, linewidth=1.5)
            lx, ly, ha, va = label_position(gt, regions.get(pop), entry, x_lim, (0.0, 1.0))
            ax.text(lx, ly, text, color=colour, ha=ha, va=va, fontsize=8, bbox=bbox,
                    transform=ax.get_xaxis_transform())
            continue
        coords = np.asarray(entry.get('boundary') or [], dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2 or not len(coords):
            continue
        closed = np.vstack([coords, coords[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color=colour, linewidth=1.5)
        lx, ly, ha, va = label_position(gt, regions.get(pop), entry, x_lim, y_lim)
        ax.text(lx, ly, text, color=colour, ha=ha, va=va, fontsize=8, bbox=bbox)

    ax.set_xlabel(axis_labels.get(ch_x, ch_x))
    ax.set_ylabel(axis_labels.get(ch_y, ch_y) if ch_y else 'Count')
    ax.set_title(title, fontsize=10)
    return fig


def population_frequency_table(summaries: dict, samples: list[str], labels: dict) -> pd.DataFrame:
    """Samples × populations, % of each population's parent."""
    rows = {}
    for s in samples:
        rows[s] = {labels.get(r['key'], r['key']): 100.0 * r['fraction_of_parent']
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
                       build_figures) -> list[ReportItem]:
    """Every available report item, in section order.

    build_figures : callable(stats, base, other, pval, fc, marker, fdr_scope,
                    is_dark, antigens, pop_labels) -> {key: {'fig', 'title',
                    'error', 'results'}}, e.g. the tab's build_result_figures.
                    It is called at most once per comparison, when the first
                    of that comparison's figures is exported.
    """
    items: list[ReportItem] = []
    pop_labels = {}
    for g in gate_defs:
        for pop, info in (g.get('populations') or {}).items():
            pop_labels[ag_core.population_key(g['gate_name'], pop)] = \
                f"{g['gate_name']} {(info or {}).get('label') or pop}"

    if summaries:
        items.append(ReportItem(
            key='samples', tab='Samples', label='Samples',
            get_tables=lambda: {'summary': sample_table(samples, load_info, groups, flags)}))
        items.append(ReportItem(
            key='frequencies', tab='Samples', label='Population frequencies',
            get_tables=lambda: {'percent_of_parent': population_frequency_table(
                summaries, samples, pop_labels)}))
        items.append(ReportItem(
            key='populations', tab='Samples', label='Population counts',
            get_tables=lambda: {'counts': ag_results.population_export(summaries, samples, groups)}))
        items.append(ReportItem(
            key='markers', tab='Samples', label='Marker medians',
            get_tables=lambda: {'medians': ag_results.marker_export(summaries, samples, antigens,
                                                                    groups)}))

    if display:
        by_name = ag_core.gates_by_name(gate_defs)
        for g in ag_core.ordered_gate_defs(gate_defs):
            name = g['gate_name']
            eff = ag_core.effective_gate_def(g, by_name)
            mode = modes.get(name, ag_core.MODE_FIXED)

            def _gate_fig(_eff=eff, _name=name, _mode=mode):
                events, parent, channels = pooled_display(display, samples, _name)
                boundary = reference_boundaries.get(_name, {})
                fractions = boundary_fractions(_eff, boundary, events, channels, parent,
                                               transforms)
                return make_gate_figure(
                    _eff, boundary, events, channels, parent, transforms, axis_labels,
                    f"{_name} ({_mode}; stored boundary)\n"
                    f"% of pooled display events, {len(samples)} samples",
                    fractions=fractions)

            items.append(ReportItem(key=f'gate:{name}', tab='Gates', label=f"Gate {name}",
                                    get_figure=_gate_fig))

    if stats:
        cache: dict = {}

        def _figures(comp):
            if comp not in cache:
                cache[comp] = build_figures(stats, comp[0], comp[1], pval_threshold,
                                            fc_threshold, marker_threshold, fdr_scope, False,
                                            antigens, pop_labels)
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
    return items
