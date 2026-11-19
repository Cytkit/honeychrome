"""
trend_plots.py — figures for trends with a numeric variable
==========================================================

Matplotlib figure builders for the results of
``controller_components.trend_stats`` (age, days after vaccination, any
time course), shared by the Honeycluster and Automated
Gating plugins. Like ``differential_plots`` each returns a
``matplotlib.figure.Figure`` without a canvas, so it can be built on a
worker thread.

* ``make_trend_volcano_figure`` — slope against the curve test's FDR.
  Features called as rising or falling are coloured by direction; those
  called by the curve but not the slope (a rise and fall, or a change that
  levels off) are drawn as open diamonds.
* ``make_trend_plot_figure`` — one feature: each sample at its value of
  the variable, coloured by group, with the fitted curve, its 95%
  confidence band, a rug of the variable's values and, on request, the
  straight line. For a group × trend test a second panel shows the
  fitted difference between the groups (and the within-pair differences
  when samples are paired).
* ``make_trajectory_heatmap_figure`` — the called features' fitted
  curves, each scaled to mean 0 and SD 1 across the variable's range, in
  rows ordered by where the curve peaks.
* ``make_pattern_panel_figure`` — one small plot per shape pattern: the
  members' z-scored fitted curves and their mean.
* ``scatter_by_value`` — samples coloured by a numeric variable (with a
  colour bar) and shaped by group, for the sample PCA.
* ``make_composition_figure`` — the make-up of a set of populations
  against the variable (``trend_stats.fit_composition``): fitted shares as
  a stacked area, with each sample's observed shares as a strip of bars
  below, samples ordered by the variable; one column per group.
* ``embedding_densities`` / ``make_embedding_by_variable_figure`` — where
  cells sit on a shared embedding (UMAP and the like) in bins of the
  variable, as small multiples, plus the difference between the last and
  the first bin.
* ``make_age_model_figure`` / ``make_age_features_figure`` — the immune
  age model (``drc_predictive.run_age_model``): predicted against actual,
  the age-corrected gap, the chance check, and the features carrying the
  prediction.

Figures that can be clicked carry ``fig._point_keys`` / ``fig._point_xy``
/ ``fig._point_axes`` (volcano points, see ``point_under``) or
``fig._row_keys`` / ``fig._row_axes`` (heatmap rows), and a hover handler
in ``fig._hover_handler`` as in ``differential_plots``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from honeychrome.view_components import differential_plots as dp

__all__ = [
    'variable_label', 'slope_label', 'make_trend_volcano_figure', 'point_under',
    'make_trend_plot_figure', 'trend_plot_table', 'feature_plot_inputs', 'trend_p_text',
    'make_trajectory_heatmap_figure',
    'scatter_by_value', 'UP_COLOUR', 'DOWN_COLOUR', 'make_pattern_panel_figure',
    'PATTERN_COLOURS', 'pattern_colours', 'make_composition_figure', 'composition_table',
    'part_colours', 'OTHER_COLOUR', 'embedding_densities', 'make_embedding_by_variable_figure',
    'embedding_bins_table', 'EMBEDDING_BINS', 'make_age_model_figure',
    'make_age_features_figure', 'AGE_FEATURES_SHOWN',
]

UP_COLOUR = '#CC3311'
DOWN_COLOUR = '#0077BB'
NOT_CALLED_COLOUR = '#aaaaaa'
GROUP_MARKERS = ('o', 's', '^', 'D', 'v', 'P', 'X')
HEATMAP_LABEL_ROWS = 100       # up to this many rows are labelled
HEATMAP_Z_LIMIT = 2.5
OTHER_COLOUR = '#bbbbbb'
EMBEDDING_BINS = 4             # default number of bins of the variable
EMBEDDING_GRID = 200           # pixels per side of the density images
EMBEDDING_SMOOTH = 2.0         # Gaussian smoothing, in pixels
AGE_FEATURES_SHOWN = 20
# Colours of composition parts without a colour of their own (Paul Tol's
# bright and muted schemes, without grey).
PART_COLOURS = ('#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377',
                '#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77',
                '#CC6677', '#882255')
# Colours of shape patterns, in pattern order (Paul Tol's muted scheme).
PATTERN_COLOURS = ('#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77',
                   '#CC6677', '#882255', '#AA4499', '#DDDDDD')


def pattern_colours(names) -> dict:
    """{pattern name: colour} in the order given."""
    return {n: PATTERN_COLOURS[i % len(PATTERN_COLOURS)] for i, n in enumerate(names)}


def variable_label(variable: str, unit: str = '') -> str:
    """Axis label of the variable, e.g. 'Age (years)'."""
    return f"{variable} ({unit})" if unit else str(variable)


def slope_label(variable: str, unit: str = '', scale: str = 'linear') -> str:
    """What the slope is per: 'per week' (unit 'weeks'), 'per doubling of
    Day', 'per unit'."""
    if scale == 'log2':
        return f"per doubling of {variable}"
    unit = str(unit or '').strip()
    if len(unit) > 2 and unit.endswith('s') and not unit.endswith('ss'):
        unit = unit[:-1]
    return f"per {unit}" if unit else "per unit"


def _log2_axis(ax, axis: str = 'x'):
    """Show an axis on the log2(t + 1) scale the trend was fitted on, with
    ticks in the variable's own units."""
    def forward(t):
        return np.log2(np.maximum(np.asarray(t, dtype=float), -0.999) + 1.0)

    def inverse(v):
        return np.power(2.0, np.asarray(v, dtype=float)) - 1.0
    (ax.set_xscale if axis == 'x' else ax.set_yscale)('function', functions=(forward, inverse))


def _group_colour_map(groups, group_colours: dict | None) -> dict:
    order = list(dict.fromkeys(groups))
    return {g: (group_colours or {}).get(g) or dp.GROUP_COLOURS[i % len(dp.GROUP_COLOURS)]
            for i, g in enumerate(order)}


# ---------------------------------------------------------------------------
# Trend volcano
# ---------------------------------------------------------------------------

def make_trend_volcano_figure(table: pd.DataFrame, title: str, *, x_label: str,
                              feature_labels: list[str] | None = None,
                              keys: list[str] | None = None,
                              pval_threshold: float = 0.05, call: str = 'either',
                              is_dark: bool = False, run_label: str = ''):
    """
    Trend volcano: x = slope (signed change per unit of the variable), y =
    -log10 of the curve test's FDR (adj.P.Val). Called features rise (red)
    or fall (blue) by the sign of their net change; called features whose
    curve but not slope is significant (``curve.only``) are open diamonds.
    The dashed line is the FDR the curve test is called at (half the
    threshold when either test may call). The ten strongest called
    features are labelled; every called point has a hover label.

    table          : a TrendResult table (after trend_significance).
    feature_labels : display label per row (default 'feature').
    keys           : click key per row (default 'feature'), stored with
                     the plotted positions for point_under().
    """
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    slope = table['slope'].to_numpy(float)
    neg_lp = -np.log10(np.maximum(table['adj.P.Val'].to_numpy(float), 1e-300))
    called = table['significant'].to_numpy(bool)
    curve_only = table['curve.only'].to_numpy(bool) & called
    up = table['direction'].to_numpy(str) == 'up'
    labels = list(feature_labels) if feature_labels is not None else table['feature'].astype(str).tolist()
    point_keys = list(keys) if keys is not None else table['feature'].astype(str).tolist()
    ok = np.isfinite(slope) & np.isfinite(neg_lp)

    fig = Figure(figsize=(5.4, 5), layout='constrained')
    ax = fig.add_subplot(111)
    fg = 'white' if is_dark else 'black'
    rest = ok & ~called
    ax.scatter(slope[rest], neg_lp[rest], s=20, c=NOT_CALLED_COLOUR, alpha=0.7, linewidths=0)
    for mask, colour in ((up, UP_COLOUR), (~up, DOWN_COLOUR)):
        solid = ok & called & ~curve_only & mask
        ax.scatter(slope[solid], neg_lp[solid], s=26, c=colour, alpha=0.9, linewidths=0, zorder=3)
        hollow = ok & curve_only & mask
        ax.scatter(slope[hollow], neg_lp[hollow], s=40, marker='D', facecolors='none',
                   edgecolors=colour, linewidths=1.1, zorder=4)
    if call != 'slope':
        alpha = pval_threshold / 2.0 if call == 'either' else pval_threshold
        ax.axhline(-np.log10(alpha), color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(0.0, color='grey', linewidth=0.6)

    hits = np.flatnonzero(ok & called)
    if hits.size:
        overlay = ax.scatter(slope[hits], neg_lp[hits], s=60, alpha=0, zorder=6)
        fig._hover_handler = dp.make_scatter_hover_handler(
            fig, ax, overlay, [labels[i] for i in hits], is_dark)
    notes = []
    for i in hits[np.argsort(neg_lp[hits])[::-1][:dp.MAX_STATIC_LABELS]]:
        on_right = slope[i] > 0
        note = ax.annotate(dp.wrap_label(labels[i], dp.VOLCANO_LABEL_WRAP),
                           xy=(slope[i], neg_lp[i]), xytext=(-4 if on_right else 4, 4),
                           textcoords='offset points', ha='right' if on_right else 'left',
                           va='bottom', fontsize=6, color=fg,
                           arrowprops=dict(arrowstyle='-', color=fg, lw=0.5, alpha=0.6,
                                           shrinkA=0, shrinkB=2,
                                           relpos=(1.0, 1.0) if on_right else (0.0, 1.0)))
        note.arrow_patch.set_visible(False)
        note.set_in_layout(False)
        notes.append(note)

    finite = np.abs(slope[ok])
    x_lim = max(float(finite.max()) * 1.08 if finite.size else 0.0, 1e-6)
    ax.set_xlim(-x_lim, x_lim)
    ax.margins(y=0.15)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(x_label)
    ax.set_ylabel("-log10(curve FDR)")
    ax.set_title(dp.wrap_to_width(title, 10, 4.8), fontsize=10)
    handles = [
        Line2D([], [], marker='o', linestyle='', color=UP_COLOUR, label=f"rises ({int((called & up & ok).sum())})"),
        Line2D([], [], marker='o', linestyle='', color=DOWN_COLOUR, label=f"falls ({int((called & ~up & ok).sum())})"),
        Line2D([], [], marker='D', linestyle='', markerfacecolor='none', color=fg,
               label=f"curve only ({int((curve_only & ok).sum())})"),
        Line2D([], [], marker='o', linestyle='', color=NOT_CALLED_COLOUR, label="not called"),
    ]
    ax.legend(handles=handles, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.13),
              ncol=4, frameon=False, handletextpad=0.2, columnspacing=1.0)
    dp.style_figure_theme(fig, is_dark)
    dp.stamp_run_label(fig, run_label)
    dp._separate_labels(fig, notes)
    fig._point_keys = [point_keys[i] for i in np.flatnonzero(ok)]
    fig._point_xy = np.column_stack([slope[ok], neg_lp[ok]])
    fig._point_axes = ax
    return fig


def point_under(fig, event, max_px: float = 12.0):
    """Key of the volcano point (``fig._point_keys``) nearest a mouse
    event, within *max_px* pixels; None otherwise."""
    ax = getattr(fig, '_point_axes', None)
    xy = getattr(fig, '_point_xy', None)
    if ax is None or event.inaxes is not ax or event.x is None or xy is None or not len(xy):
        return None
    px = ax.transData.transform(np.asarray(xy, dtype=float))
    d = np.hypot(px[:, 0] - event.x, px[:, 1] - event.y)
    i = int(np.argmin(d))
    return fig._point_keys[i] if d[i] <= max_px else None


# ---------------------------------------------------------------------------
# Per-feature trend plot
# ---------------------------------------------------------------------------

def _band(curve, curve_se, df2):
    from scipy import stats
    if curve_se is None:
        return None
    se = np.asarray(curve_se, dtype=float)
    if not np.isfinite(se).any():
        return None
    q = stats.norm.ppf(0.975) if not np.isfinite(df2) else stats.t.ppf(0.975, df2)
    return curve - q * se, curve + q * se


def _stats_text(row: dict, slope_unit: str) -> str:
    def g(key):
        v = row.get(key)
        return float(v) if v is not None and pd.notna(v) else np.nan
    parts = [f"curve p = {g('P.Value'):.2g} (FDR {g('adj.P.Val'):.2g})",
             f"slope {g('slope'):+.3g} {slope_unit} (FDR {g('slope.adj.P.Val'):.2g})",
             f"partial R² {g('R2'):.2f}"]
    return "; ".join(parts)


def trend_p_text(row: dict | None) -> str:
    """'curve FDR = 0.001, slope FDR = 0.02, partial R² = 0.35' from a trend table row,
    with 'called' when the feature is called ('not tested' without a row)."""
    if not row:
        return 'not tested'

    def g(key):
        v = row.get(key)
        return float(v) if v is not None and pd.notna(v) else np.nan
    text = (f"curve FDR = {g('adj.P.Val'):.3g}, slope FDR = {g('slope.adj.P.Val'):.3g}, "
            f"partial R² = {g('R2'):.2f}")
    return text + ("; called" if bool(row.get('significant')) else '')


def _group_span(grid, time, groups, group) -> np.ndarray:
    """Grid points within the values of the variable a group was sampled
    at, so a group's own curve is never drawn beyond its samples."""
    t = np.asarray(time, dtype=float)
    m = np.isfinite(t) & np.array([str(g) == str(group) for g in groups])
    grid = np.asarray(grid, dtype=float)
    if not m.any():
        return np.zeros(grid.size, dtype=bool)
    tol = 1e-9 * max(1.0, float(np.abs(grid).max()) if grid.size else 1.0)
    return (grid >= t[m].min() - tol) & (grid <= t[m].max() + tol)


def trend_plot_table(time, values, groups, samples, *, grid, curve, curve_se=None,
                     df2: float = np.inf, variable: str = 'time',
                     back_transform: str = '', group_curves: dict | None = None) -> pd.DataFrame:
    """The values behind a trend plot, for CSV: one row per sample
    (kind 'sample': sample, group, variable, value) and one per grid point
    (kind 'fit': variable, fitted value and the 95% band), with the fit on
    the percent scale for back_transform 'percent' (see
    make_trend_plot_figure). With *group_curves* ({group: {'curve', 'se'}},
    an unpaired group × trend test) the fit rows are each group's own curve
    (group filled in) followed by the difference between the groups (kind
    'difference')."""
    rows = [{'kind': 'sample', 'sample': s, 'group': g, variable: t, 'value': v}
            for s, g, t, v in zip(samples, groups, time, values)]
    if group_curves:
        for g, gc in group_curves.items():
            g_curve = np.asarray(gc['curve'], dtype=float)
            g_band = _band(g_curve, gc.get('se'), df2)
            span = _group_span(grid, time, groups, g)
            for i, t in enumerate(grid):
                if not span[i]:
                    continue
                rows.append({'kind': 'fit', 'group': g, variable: float(t),
                             'fitted': float(g_curve[i]),
                             'lower_95': float(g_band[0][i]) if g_band else np.nan,
                             'upper_95': float(g_band[1][i]) if g_band else np.nan})
    curve = np.asarray(curve, dtype=float)
    band = _band(curve, curve_se, df2)
    if back_transform == 'percent':
        def to_percent(y):
            return 100.0 / (1.0 + np.power(2.0, -np.asarray(y, dtype=float)))
        curve = to_percent(curve)
        band = (to_percent(band[0]), to_percent(band[1])) if band is not None else None
    for i, t in enumerate(grid):
        rows.append({'kind': 'difference' if group_curves else 'fit', variable: float(t),
                     'fitted': float(curve[i]),
                     'lower_95': float(band[0][i]) if band else np.nan,
                     'upper_95': float(band[1][i]) if band else np.nan})
    return pd.DataFrame(rows, columns=['kind', 'sample', 'group', variable, 'value',
                                       'fitted', 'lower_95', 'upper_95'])


def feature_plot_inputs(result, data: pd.DataFrame, samples, time, groups, feature: str,
                        pairing: dict | None = None) -> dict:
    """
    Keyword arguments of make_trend_plot_figure for one feature of a
    TrendResult (all but the labels, colours and theme).

    result  : the TrendResult.
    data    : the values tested, samples × features.
    samples, time, groups: the trend run's samples with their value of the
              variable and their group; only the samples the result was
              fitted on (``settings['samples']``) are drawn.
    pairing : {sample: pairing level}; used for the within-pair
              differences of a paired group × trend test.

    Raises KeyError when *feature* is not in the result.
    """
    table = result.table
    hit = np.flatnonzero(table['feature'].astype(str).to_numpy() == str(feature))
    if not hit.size:
        raise KeyError(feature)
    i = int(hit[0])
    settings = result.settings or {}
    time_of = dict(zip(samples, time))
    group_of = dict(zip(samples, groups))
    used = [s for s in (settings.get('samples') or list(samples)) if s in data.index]
    column = (data[feature] if feature in data.columns
              else pd.Series(np.nan, index=data.index)).reindex(used)
    interaction = bool(settings.get('interaction'))
    pairs = None
    if interaction and settings.get('paired') and pairing:
        pairs = [pairing.get(s, f"unpaired {s}") for s in used]
    group_curves = None
    if interaction and getattr(result, 'group_curves', None) is not None:
        se = getattr(result, 'group_curve_se', None)
        group_curves = {str(g): {'curve': result.group_curves[j][i],
                                 'se': None if se is None else se[j][i]}
                        for j, g in enumerate(list(settings.get('groups') or [])[:2])}
    return dict(time=pd.to_numeric(pd.Series([time_of.get(s) for s in used]),
                                   errors='coerce').to_numpy(float),
                values=column.to_numpy(float), groups=[group_of.get(s, '') for s in used],
                samples=used, grid=result.grid, curve=result.curves[i],
                curve_se=None if result.curve_se is None else result.curve_se[i],
                row=table.iloc[i].to_dict(), variable=settings.get('variable', 'time'),
                unit=settings.get('unit', ''), scale=settings.get('scale', 'linear'),
                interaction=interaction, group_order=list(settings.get('groups') or []),
                pairs=pairs, group_curves=group_curves)


def make_trend_plot_figure(*, time, values, groups, samples, grid, curve, row: dict,
                           title: str, variable: str, unit: str = '', scale: str = 'linear',
                           value_label: str = 'value', curve_se=None,
                           group_colours: dict | None = None, show_line: bool = False,
                           interaction: bool = False, group_order: list[str] | None = None,
                           pairs=None, is_dark: bool = False, run_label: str = '',
                           axes=None, back_transform: str = '', figsize=None,
                           compact: bool = False, transparent: bool = False,
                           group_curves: dict | None = None):
    """
    One feature against the variable.

    time, values, groups, samples : per sample (values on the tested
                     scale; NaN values are not drawn).
    grid, curve, curve_se : the fitted curve on the grid (original units of
                     the variable) and its pointwise SE; a 95% band is drawn
                     when the SE is given.
    row            : the feature's row of the trend table (p, FDR, slope,
                     partial R² for the subtitle; df2 for the band).
    scale          : 'log2' draws the variable's axis on log2(t + 1).
    show_line      : add the straight line with the fitted slope, through
                     the mean of the samples.
    interaction    : the curve is the difference between the two groups of
                     *group_order* (second minus first); the samples are
                     drawn in the upper panel, the difference in the lower
                     one, with the within-pair differences when *pairs*
                     (pairing level per sample) is given.
    group_curves   : {group: {'curve', 'se'}} for an unpaired interaction:
                     one panel, each group's own curve and 95% band in its
                     group colour over the values that group was sampled at
                     (*curve* is then only used for the CSV).
    axes           : optional axes of a larger figure to draw into (two for
                     an interaction without *group_curves*, side by side); the figure is then
                     returned without theme styling, hover labels or run
                     label, as make_group_values_figure does with *ax*.
    back_transform : 'percent' when *values* are percentages (e.g. % of
                     parent) and the fit was made on log2 odds: the curve
                     and its band are drawn back on the percent scale,
                     100 / (1 + 2^−y). Not for an interaction; the straight
                     line is not drawn.
    figsize, compact, transparent : size in inches; small type with the
                     test results left out of the title (for a plot tile);
                     no background, so the plot sits on the window like the
                     gating plots.
    """
    from matplotlib.figure import Figure

    t = np.asarray(time, dtype=float)
    v = np.asarray(values, dtype=float)
    groups = [str(g) for g in groups]
    ok = np.isfinite(t) & np.isfinite(v)
    grid = np.asarray(grid, dtype=float)
    curve = np.asarray(curve, dtype=float)
    df2 = float(row.get('df2', np.inf)) if pd.notna(row.get('df2', np.nan)) else np.inf
    band = _band(curve, curve_se, df2)
    if back_transform == 'percent':
        def to_percent(y):
            return 100.0 / (1.0 + np.power(2.0, -np.asarray(y, dtype=float)))
        curve = to_percent(curve)
        band = (to_percent(band[0]), to_percent(band[1])) if band is not None else None
        show_line = False
    order = list(group_order or dict.fromkeys(groups))
    colours = _group_colour_map(order + groups, group_colours)
    fg = 'white' if is_dark else 'black'
    fill = '#888888'
    slope_unit = slope_label(variable, unit, scale)
    group_curves = group_curves if interaction and group_curves else None
    two_panels = interaction and group_curves is None

    own = axes is None
    if not own:
        ax = axes[0]
        ax_diff = axes[1] if two_panels else None
        fig = ax.figure
    else:
        fig = Figure(figsize=figsize or (5.2, 6.6 if two_panels else 4.4), layout='constrained')
        if two_panels:
            ax, ax_diff = fig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 1]})
        else:
            ax, ax_diff = fig.add_subplot(111), None

    hover_xy, hover_labels = [], []
    for g in dict.fromkeys(order + groups):
        m = ok & np.array([x == g for x in groups])
        if not m.any():
            continue
        ax.scatter(t[m], v[m], s=26, color=colours[g], edgecolors=fg, linewidths=0.4,
                   label=f"{g} (n={int(m.sum())})", zorder=3)
        for s, x, y in zip(np.asarray(samples, dtype=object)[m], t[m], v[m]):
            hover_xy.append((x, y))
            hover_labels.append(f"{str(s).replace(chr(92), '/').rsplit('/', 1)[-1]}: {y:.3g}")

    target = ax_diff if two_panels else ax
    if group_curves is not None:
        for g, gc in group_curves.items():
            g_curve = np.asarray(gc['curve'], dtype=float)
            g_band = _band(g_curve, gc.get('se'), df2)
            colour = colours.get(g, fg)
            span = _group_span(grid, t, groups, g)
            if g_band is not None:
                ax.fill_between(grid[span], g_band[0][span], g_band[1][span], color=colour,
                                alpha=0.2, linewidth=0, zorder=1)
            ax.plot(grid[span], g_curve[span], color=colour, linewidth=1.8, zorder=4,
                    label='_nolegend_')
    else:
        if band is not None:
            target.fill_between(grid, band[0], band[1], color=fill, alpha=0.25, linewidth=0,
                                zorder=1)
        target.plot(grid, curve, color=fg, linewidth=1.8, zorder=4, label='fitted curve')
    if show_line and not interaction and ok.sum() >= 2:
        x = np.log2(t[ok] + 1.0) if scale == 'log2' else t[ok]
        xg = np.log2(grid + 1.0) if scale == 'log2' else grid
        slope = float(row.get('slope', np.nan))
        if np.isfinite(slope):
            target.plot(grid, v[ok].mean() + slope * (xg - x.mean()), color=fg, linestyle='--',
                        linewidth=1.0, zorder=4, label='straight line')

    if two_panels:
        a, b = (order + ['', ''])[:2]
        ax_diff.axhline(0.0, color='grey', linewidth=0.7)
        if pairs is not None:
            frame = pd.DataFrame({'pair': [str(p) for p in pairs], 'group': groups,
                                  't': t, 'v': v})[ok]
            wide = frame.pivot_table(index='pair', columns='group', values='v', aggfunc='mean')
            tp = frame.groupby('pair')['t'].mean()
            if a in wide.columns and b in wide.columns:
                diff = (wide[b] - wide[a]).dropna()
                ax_diff.scatter(tp.reindex(diff.index), diff.to_numpy(float), s=22, color=fill,
                                edgecolors=fg, linewidths=0.4, zorder=3,
                                label=f"pairs (n={len(diff)})")
        ax_diff.set_ylabel(dp.wrap_to_width(f"{b} − {a}", 9, 2.6))
        ax_diff.set_xlabel(variable_label(variable, unit))
        ax_diff.legend(fontsize=7, loc='best')
    if not two_panels or not own:
        ax.set_xlabel(variable_label(variable, unit))
    ax.set_ylabel(dp.wrap_to_width(value_label, 9, 3.0))
    rug_ax = ax_diff if two_panels else ax
    rug_ax.plot(t[np.isfinite(t)], np.zeros(int(np.isfinite(t).sum())), '|', color=fg,
                markersize=8, alpha=0.5, transform=rug_ax.get_xaxis_transform(), zorder=2)
    for axis in ([ax, ax_diff] if two_panels else [ax]):
        if scale == 'log2':
            _log2_axis(axis)
        if grid.size:
            span = float(grid.max() - grid.min()) or 1.0
            axis.set_xlim(grid.min() - 0.03 * span, grid.max() + 0.03 * span)
    if not compact:
        ax.legend(fontsize=7, loc='best')
    elif len({g for g, keep in zip(groups, ok) if keep}) > 1:
        ax.legend(handles=[h for h in ax.get_legend_handles_labels()[0]
                           if h.get_label() != 'fitted curve'],
                  fontsize=6, loc='best', frameon=False)
    title_w = 4.8 if own else max(2.0, ax.get_position().width * fig.get_figwidth())
    if compact:
        ax.set_title(dp.wrap_to_width(title, 8, fig.get_figwidth() - 0.2), fontsize=8)
        for axis in ([ax, ax_diff] if two_panels else [ax]):
            axis.tick_params(labelsize=7)
            axis.xaxis.label.set_size(8)
            axis.yaxis.label.set_size(8)
    else:
        ax.set_title(dp.wrap_to_width(f"{title}\n{_stats_text(row, slope_unit)}", 9, title_w),
                     fontsize=9)
    if not own:
        return fig
    if hover_xy:
        hover = np.asarray(hover_xy, dtype=float)
        overlay = ax.scatter(hover[:, 0], hover[:, 1], s=60, alpha=0, zorder=6)
        fig._hover_handler = dp.make_scatter_hover_handler(fig, ax, overlay, hover_labels, is_dark)
    dp.style_figure_theme(fig, is_dark)
    if transparent:
        fig.patch.set_alpha(0.0)
        for axis in fig.axes:
            axis.set_facecolor('none')
    dp.stamp_run_label(fig, run_label)
    return fig


# ---------------------------------------------------------------------------
# Trajectory heatmap
# ---------------------------------------------------------------------------

def make_trajectory_heatmap_figure(table: pd.DataFrame, grid, curves, *, title: str,
                                   variable: str, unit: str = '', scale: str = 'linear',
                                   feature_labels: list[str] | None = None,
                                   keys: list[str] | None = None,
                                   patterns: list[dict] | None = None,
                                   is_dark: bool = False, run_label: str = ''):
    """
    Fitted curves of the called features, one row each, scaled to mean 0
    and SD 1 across the grid (so the shape, not the size, is compared),
    ordered by the value of the variable where the curve peaks, then where
    it is lowest. Columns are the grid points (the variable's observed
    range). Rows carry ``fig._row_keys`` for click-through.

    patterns : the result's shape patterns (settings['patterns']); with a
               'pattern' column in *table*, rows are grouped by pattern in
               this order, with a coloured strip and a key.
    """
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec

    called = table['significant'].to_numpy(bool)
    curves = np.asarray(curves, dtype=float)
    grid = np.asarray(grid, dtype=float)
    labels = list(feature_labels) if feature_labels is not None else table['feature'].astype(str).tolist()
    row_keys = list(keys) if keys is not None else table['feature'].astype(str).tolist()
    idx = np.flatnonzero(called & np.isfinite(curves).all(axis=1)) if curves.size else np.array([], int)
    if idx.size == 0:
        return dp.message_figure("No feature is called as changing with "
                                 f"{variable}.", title, is_dark, run_label)
    sub = curves[idx]
    sd = sub.std(axis=1, keepdims=True)
    z = np.where(sd > 0, (sub - sub.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1.0), 0.0)
    names = [p['name'] for p in (patterns or [])]
    rank = {n: i for i, n in enumerate(names)}
    if names and 'pattern' in table.columns:
        pat = np.array([rank.get(str(v), len(names)) for v in table['pattern'].to_numpy()[idx]])
    else:
        pat = np.zeros(idx.size, dtype=int)
        names = []
    order = np.lexsort((table['at.min'].to_numpy(float)[idx], table['at.max'].to_numpy(float)[idx],
                        pat))
    z = z[order]
    idx = idx[order]
    pat = pat[order]
    n = idx.size
    labelled = n <= HEATMAP_LABEL_ROWS
    longest = max((len(labels[i]) for i in idx), default=0) if labelled else 0
    label_in = min(0.055 * longest, 3.5) if labelled else 0.3
    height = 1.8 + (0.13 * n if labelled else min(0.03 * n, 10.0))
    fig = Figure(figsize=(4.2 + label_in, height))
    bottom = 0.65 + (0.18 * ((len(names) + 1) // 2) + 0.1 if names else 0.0)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.05, 0.035, 1], wspace=0.12)
    fig.set_layout_engine(dp._fixed_margin_layout(gs, left=0.8, right=0.3 + label_in,
                                                  top=0.75, bottom=bottom))
    cax = fig.add_subplot(gs[0, 0])
    sax = fig.add_subplot(gs[0, 1])
    ax = fig.add_subplot(gs[0, 2])
    sax.axis('off')
    if names:
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
        colours = list(pattern_colours(names).values()) + ['#bbbbbb']
        sax.imshow(pat[:, None], aspect='auto', cmap=ListedColormap(colours), vmin=-0.5,
                   vmax=len(colours) - 0.5, interpolation='nearest',
                   extent=(-0.5, 0.5, n - 0.5, -0.5))
        sax.set_ylim(n - 0.5, -0.5)
        for b in np.flatnonzero(np.diff(pat)) + 0.5:
            ax.axhline(b, color=dp.figure_background(is_dark), linewidth=1.2)
        counts = [int((pat == i).sum()) for i in range(len(names))]
        fig.legend(handles=[Patch(color=c, label=f"{nm} ({k})")
                            for nm, c, k in zip(names, colours, counts) if k],
                   loc='lower center', ncol=2, fontsize=7, frameon=False,
                   bbox_to_anchor=(0.5, 0.0))
    image = ax.imshow(z, aspect='auto', cmap='RdBu_r', vmin=-HEATMAP_Z_LIMIT,
                      vmax=HEATMAP_Z_LIMIT, interpolation='nearest',
                      extent=(-0.5, grid.size - 0.5, n - 0.5, -0.5))
    ticks = np.unique(np.linspace(0, grid.size - 1, min(6, grid.size)).round().astype(int))
    ax.set_xticks(ticks, [f"{grid[i]:.3g}" for i in ticks], fontsize=7)
    ax.set_xlabel(variable_label(variable, unit) + (" (log2 spaced)" if scale == 'log2' else ''))
    ax.yaxis.tick_right()
    if labelled:
        size = 7 if n <= 40 else 5
        ax.set_yticks(range(n), [labels[i] for i in idx], fontsize=size)
    else:
        ax.set_yticks([])
        ax.yaxis.set_label_position('right')
        ax.set_ylabel(f"{n} features")
    ax.set_title(dp.wrap_to_width(title, 10, 3.6 + label_in), fontsize=10)
    fig.colorbar(image, cax=cax, label="fitted curve, z-score")
    cax.yaxis.set_ticks_position('left')
    cax.yaxis.set_label_position('left')
    fig._row_keys = [row_keys[i] for i in idx]
    fig._row_axes = ax
    fig._row_labels = [labels[i] for i in idx]
    dp.style_figure_theme(fig, is_dark)
    dp.stamp_run_label(fig, run_label)
    return fig


# ---------------------------------------------------------------------------
# Pattern panel
# ---------------------------------------------------------------------------

MAX_PATTERN_LINES = 150


def make_pattern_panel_figure(table: pd.DataFrame, grid, curves, patterns: list[dict], *,
                              title: str, variable: str, unit: str = '',
                              scale: str = 'linear', is_dark: bool = False,
                              run_label: str = ''):
    """
    One small plot per shape pattern: the member features' fitted curves
    scaled to mean 0 and SD 1 (thin lines, at most MAX_PATTERN_LINES drawn)
    and the pattern's mean curve (bold), titled with the pattern's name and
    member count.

    table, curves : a TrendResult's table (with 'pattern') and curves.
    patterns      : settings['patterns'] of the result.
    """
    from matplotlib.figure import Figure

    from honeychrome.controller_components.trend_stats import zscore_curves

    if not patterns or 'pattern' not in table.columns:
        return dp.message_figure("No shape patterns: fewer than two features are called.",
                                 title, is_dark, run_label)
    grid = np.asarray(grid, dtype=float)
    curves = np.asarray(curves, dtype=float)
    labels = table['pattern'].astype(str).to_numpy()
    n = len(patterns)
    ncol = min(n, 4)
    nrow = int(np.ceil(n / ncol))
    fig = Figure(figsize=(2.6 * ncol + 0.4, 2.3 * nrow + 0.8), layout='constrained')
    axes = np.atleast_1d(fig.subplots(nrow, ncol, sharex=True, sharey=True, squeeze=False)).ravel()
    colours = pattern_colours([p['name'] for p in patterns])
    rng = np.random.default_rng(0)
    for ax, p in zip(axes, patterns):
        members = np.flatnonzero(labels == p['name'])
        members = members[np.isfinite(curves[members]).all(axis=1)]
        if members.size > MAX_PATTERN_LINES:
            members = np.sort(rng.choice(members, MAX_PATTERN_LINES, replace=False))
        z = zscore_curves(curves[members]) if members.size else np.zeros((0, grid.size))
        colour = colours[p['name']]
        for row in z:
            ax.plot(grid, row, color=colour, alpha=0.18, linewidth=0.6)
        ax.plot(grid, np.asarray(p['mean'], dtype=float), color=colour, linewidth=2.4)
        ax.axhline(0.0, color='grey', linewidth=0.5)
        ax.set_title(dp.wrap_to_width(f"{p['name']} ({p['n']})", 8, 2.4), fontsize=8)
        if scale == 'log2':
            _log2_axis(ax)
    for ax in axes[n:]:
        ax.axis('off')
    fig.supxlabel(variable_label(variable, unit), fontsize=9)
    fig.supylabel("fitted curve, z-score", fontsize=9)
    fig.suptitle(dp.wrap_to_width(title, 10, 2.6 * ncol), fontsize=10)
    dp.style_figure_theme(fig, is_dark)
    dp.stamp_run_label(fig, run_label)
    return fig


# ---------------------------------------------------------------------------
# Composition over the variable
# ---------------------------------------------------------------------------

def part_colours(parts, given: dict | None = None, other: str = 'Other') -> dict:
    """{part: colour}: the colour given for a part (e.g. a cluster's), else
    the next of PART_COLOURS; the remainder part *other* is grey."""
    given = given or {}
    out, i = {}, 0
    for part in parts:
        if part == other:
            out[part] = OTHER_COLOUR
        elif given.get(part):
            out[part] = given[part]
        else:
            out[part] = PART_COLOURS[i % len(PART_COLOURS)]
            i += 1
    return out


def make_composition_figure(panels: list[dict], *, title: str, colours: dict | None = None,
                            is_dark: bool = False, run_label: str = ''):
    """
    Fitted composition against the variable as a stacked area (% of the
    parent), with each sample's observed composition as a strip of stacked
    bars underneath (one bar per sample, ordered by the variable, ticks at
    the variable's value), so the fit can be checked against the data.

    panels  : [{'label': panel title (e.g. the group) or '', **fit_composition
              result}], drawn side by side with a shared legend.
    colours : {part: colour}; see part_colours.
    """
    from matplotlib.figure import Figure

    panels = [p for p in panels if p is not None]
    if not panels:
        return dp.message_figure("No composition to show.", title, is_dark, run_label)
    parts = list(dict.fromkeys(part for p in panels for part in p['parts']))
    colours = part_colours(parts, colours)
    legend_w = min(max(dp.text_width_inches(str(x), 8) for x in parts) + 0.5, 3.0)
    n = len(panels)
    fig = Figure(figsize=(4.4 * n + legend_w + 0.4, 5.4), layout='constrained')
    axes = fig.subplots(2, n, squeeze=False, gridspec_kw={'height_ratios': [3, 1]})
    for col, panel in enumerate(panels):
        st = panel.get('settings') or {}
        variable, unit, scale = st.get('variable', 'time'), st.get('unit', ''), st.get('scale', 'linear')
        grid = np.asarray(panel['grid'], dtype=float)
        shares = 100.0 * np.asarray(panel['shares'], dtype=float)
        ax = axes[0, col]
        ax.stackplot(grid, shares, colors=[colours[x] for x in panel['parts']],
                     labels=panel['parts'], linewidth=0.3,
                     edgecolor=dp.figure_background(is_dark))
        ax.set_ylim(0, 100)
        if grid.size:
            ax.set_xlim(grid.min(), grid.max())
        if scale == 'log2':
            _log2_axis(ax)
        ax.set_xlabel(variable_label(variable, unit))
        ax.set_ylabel('% (fitted)' if col == 0 else '')
        sub_title = panel.get('label') or ''
        ax.set_title(dp.wrap_to_width(f"{sub_title} (n = {len(panel['samples'])})".strip(), 9, 4.2),
                     fontsize=9)
        bars = axes[1, col]
        order = np.argsort(np.asarray(panel['time'], dtype=float), kind='stable')
        obs = 100.0 * np.asarray(panel['observed'], dtype=float)[order]
        x = np.arange(order.size)
        bottom = np.zeros(order.size)
        for j, part in enumerate(panel['parts']):
            bars.bar(x, obs[:, j], bottom=bottom, width=1.0, color=colours[part], linewidth=0)
            bottom += obs[:, j]
        bars.set_xlim(-0.5, order.size - 0.5)
        bars.set_ylim(0, 100)
        step = max(1, int(np.ceil(order.size / 6)))
        ticks = x[::step]
        t_sorted = np.asarray(panel['time'], dtype=float)[order]
        bars.set_xticks(ticks, [f"{t_sorted[i]:.3g}" for i in ticks], fontsize=7)
        bars.set_xlabel(f"samples ordered by {variable}", fontsize=8)
        bars.set_ylabel('% (each sample)' if col == 0 else '', fontsize=8)
        bars.tick_params(labelsize=7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    for col in range(1, n):
        for h, lbl in zip(*axes[0, col].get_legend_handles_labels()):
            seen.setdefault(lbl, h)
    legend = fig.legend([seen[p] for p in parts if p in seen], [p for p in parts if p in seen],
                        loc='outside right center', fontsize=8, frameon=False)
    fig.suptitle(dp.wrap_to_width(title, 10, 4.4 * n), fontsize=10)
    fg = dp.style_figure_theme(fig, is_dark)
    for text in legend.get_texts():
        text.set_color(fg)
    dp.stamp_run_label(fig, run_label)
    return fig


def composition_table(panels: list[dict]) -> pd.DataFrame:
    """The values behind make_composition_figure, for CSV: fitted shares
    on the grid (kind 'fit') and each sample's observed shares (kind
    'sample'), in % of the parent, one row per part."""
    rows = []
    for panel in panels:
        if panel is None:
            continue
        variable = (panel.get('settings') or {}).get('variable', 'time')
        label = panel.get('label') or ''
        for i, t in enumerate(panel['grid']):
            for j, part in enumerate(panel['parts']):
                rows.append({'panel': label, 'kind': 'fit', 'sample': '', variable: float(t),
                             'population': part, 'percent': 100.0 * float(panel['shares'][j][i])})
        for i, (sample, t) in enumerate(zip(panel['samples'], panel['time'])):
            for j, part in enumerate(panel['parts']):
                rows.append({'panel': label, 'kind': 'sample', 'sample': sample,
                             variable: float(t), 'population': part,
                             'percent': 100.0 * float(panel['observed'][i][j])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Embedding by the variable
# ---------------------------------------------------------------------------

def _variable_bins(values: np.ndarray, n_bins: int, levels) -> list[tuple[float, float]]:
    """(low, high) of each bin: the distinct *levels* when given (a
    designed time course), otherwise quantile bins holding about the same
    number of samples (bins that would be empty are merged away)."""
    if levels is not None:
        return [(float(v), float(v)) for v in sorted(set(float(x) for x in levels))]
    edges = np.unique(np.quantile(values, np.linspace(0, 1, int(n_bins) + 1)))
    if edges.size < 2:
        return [(float(values.min()), float(values.max()))]
    return [(float(lo), float(hi)) for lo, hi in zip(edges[:-1], edges[1:])]


def embedding_densities(embeddings: dict, time: dict, *, n_bins: int = EMBEDDING_BINS,
                        levels=None, size: int = EMBEDDING_GRID,
                        smooth: float = EMBEDDING_SMOOTH) -> dict:
    """
    Density of events on a shared 2-D embedding in bins of the variable.

    In plain terms: the samples are split into groups by their value of
    the variable (e.g. four age bins with about the same number of samples
    each), and for each bin the map shows where its cells sit. Every
    sample counts equally within its bin, whatever its number of events,
    so one large sample cannot paint the whole bin. The difference map
    (last bin minus first) shows where cells gather late (positive) or
    early (negative). It is a picture; which clusters change is the trend
    test's answer.

    Maths: each sample's events binned on a size × size grid over the
    0.5–99.5th percentiles of all events (5% margin), normalised to sum 1;
    a bin's density is the mean of its samples' grids, smoothed with a
    Gaussian of *smooth* pixels.

    embeddings : {sample: (events × 2) coordinates}.
    time       : {sample: value of the variable}; samples without a value
                 or without an embedding are left out.
    levels     : the distinct values to use as bins (time points as groups).
    Returns {'extent', 'bins': [{'low', 'high', 'samples', 'density'}],
    'difference', 'n_left_out'}.
    """
    from scipy.ndimage import gaussian_filter

    pairs = []
    for sample, xy in embeddings.items():
        t = time.get(sample)
        try:
            t = float(t)
        except (TypeError, ValueError):
            continue
        xy = np.asarray(xy, dtype=float)
        if not np.isfinite(t) or xy.ndim != 2 or xy.shape[1] < 2 or not len(xy):
            continue
        pairs.append((sample, t, xy[:, :2]))
    left_out = len(set(time) | set(embeddings)) - len(pairs)
    if len(pairs) < 2:
        raise RuntimeError("Fewer than two samples have both an embedding and a value of the "
                           "variable.")
    values = np.array([t for _s, t, _xy in pairs])
    bins = _variable_bins(values, n_bins, levels)
    pooled = np.concatenate([xy[np.isfinite(xy).all(axis=1)] for _s, _t, xy in pairs])
    lo = np.percentile(pooled, 0.5, axis=0)
    hi = np.percentile(pooled, 99.5, axis=0)
    pad = 0.05 * np.maximum(hi - lo, 1e-9)
    lo, hi = lo - pad, hi + pad
    edges_x = np.linspace(lo[0], hi[0], size + 1)
    edges_y = np.linspace(lo[1], hi[1], size + 1)
    grids = {}
    for sample, _t, xy in pairs:
        h, _ex, _ey = np.histogram2d(xy[:, 0], xy[:, 1], bins=(edges_x, edges_y))
        total = h.sum()
        if total > 0:
            grids[sample] = h.T / total
    out_bins = []
    for i, (b_lo, b_hi) in enumerate(bins):
        last = i == len(bins) - 1
        members = [s for s, t, _xy in pairs if s in grids
                   and (np.isclose(t, b_lo) if b_lo == b_hi
                        else (b_lo <= t < b_hi or (last and t == b_hi)))]
        if not members:
            continue
        dens = gaussian_filter(np.mean([grids[s] for s in members], axis=0), smooth)
        out_bins.append({'low': b_lo, 'high': b_hi, 'samples': members, 'density': dens})
    diff = (out_bins[-1]['density'] - out_bins[0]['density']) if len(out_bins) >= 2 else None
    return {'extent': (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])),
            'bins': out_bins, 'difference': diff, 'n_left_out': int(max(left_out, 0))}


def _bin_label(b: dict, unit: str) -> str:
    span = (f"{b['low']:.3g}" if b['low'] == b['high'] else f"{b['low']:.3g}–{b['high']:.3g}")
    return f"{span}{(' ' + unit) if unit else ''} (n = {len(b['samples'])})"


def make_embedding_by_variable_figure(dens: dict, *, title: str, variable: str, unit: str = '',
                                      axis_labels: tuple[str, str] = ('', ''),
                                      is_dark: bool = False, run_label: str = ''):
    """
    Small multiples of embedding_densities: one density map per bin of the
    variable on a shared colour scale, then the last − first difference
    map (red: more common in the last bin, blue: more common in the first).
    """
    from matplotlib.figure import Figure

    bins = dens.get('bins') or []
    if not bins:
        return dp.message_figure("No sample has both an embedding and a value of the variable.",
                                 title, is_dark, run_label)
    panels = len(bins) + (1 if dens.get('difference') is not None else 0)
    ncol = min(panels, 5)
    nrow = int(np.ceil(panels / ncol))
    fig = Figure(figsize=(2.7 * ncol + 0.6, 2.8 * nrow + 0.8), layout='constrained')
    axes = np.atleast_1d(fig.subplots(nrow, ncol, squeeze=False)).ravel()
    extent = dens['extent']
    vmax = max(float(np.percentile(b['density'], 99.9)) for b in bins) or 1.0
    image = None
    for ax, b in zip(axes, bins):
        image = ax.imshow(b['density'], origin='lower', extent=extent, aspect='auto',
                          cmap='magma' if is_dark else 'viridis', vmin=0.0, vmax=vmax,
                          interpolation='nearest')
        ax.set_title(dp.wrap_to_width(_bin_label(b, unit), 8, 2.6), fontsize=8)
    if image is not None:
        cb = fig.colorbar(image, ax=list(axes[:len(bins)]), shrink=0.7, pad=0.01)
        cb.set_label('share of events per pixel', fontsize=7)
        cb.ax.tick_params(labelsize=6)
    if dens.get('difference') is not None:
        ax = axes[len(bins)]
        diff = dens['difference']
        lim = float(np.percentile(np.abs(diff), 99.9)) or 1.0
        im = ax.imshow(diff, origin='lower', extent=extent, aspect='auto', cmap='RdBu_r',
                       vmin=-lim, vmax=lim, interpolation='nearest')
        ax.set_title(dp.wrap_to_width(f"last − first bin of {variable}", 8, 2.6), fontsize=8)
        cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.01)
        cb.set_label('difference', fontsize=7)
        cb.ax.tick_params(labelsize=6)
    for ax in axes[:panels]:
        ax.set_xlabel(axis_labels[0], fontsize=7)
        ax.set_ylabel(axis_labels[1], fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes[panels:]:
        ax.axis('off')
    note = f"; {dens['n_left_out']} sample(s) without an embedding or value left out" \
        if dens.get('n_left_out') else ''
    fig.suptitle(dp.wrap_to_width(f"{title}{note}", 10, 2.7 * ncol), fontsize=10)
    dp.style_figure_theme(fig, is_dark)
    dp.stamp_run_label(fig, run_label)
    return fig


def embedding_bins_table(dens: dict, time: dict, variable: str = 'time') -> pd.DataFrame:
    """Which bin each sample of embedding_densities fell in, for CSV."""
    rows = []
    for i, b in enumerate(dens.get('bins') or []):
        for sample in b['samples']:
            rows.append({'sample': sample, variable: time.get(sample), 'bin': i + 1,
                         'bin_low': b['low'], 'bin_high': b['high']})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Immune age model
# ---------------------------------------------------------------------------

def age_model_subtitle(result: dict) -> str:
    perf = result['performance'].iloc[0]
    summary = result['summary']
    text = (f"n = {summary['n']}, {summary['n_features']} features; mean absolute error "
            f"{perf['mae_mean']:.3g} ± {perf['mae_sd']:.2g} (predicting the mean: "
            f"{perf['mae_baseline']:.3g}); R² {perf['r2_mean']:.2f}")
    if np.isfinite(perf.get('p_chance', np.nan)):
        text += f"; chance p = {perf['p_chance']:.3g} ({summary['n_permutations']} shuffles)"
    return text


def make_age_model_figure(result: dict, *, variable: str, unit: str = '',
                          groups: dict | None = None, group_colours: dict | None = None,
                          title: str = 'Immune age model', is_dark: bool = False,
                          run_label: str = ''):
    """
    The immune age model in three panels: each sample's cross-validated
    prediction against its actual value (identity line; samples outside the
    model, predicted by the final model, as open markers), the corrected
    gap against the actual value, and the chance check (mean absolute error
    with the variable shuffled, the real model's as a line). With more than
    one group, a fourth panel shows each group's predicted values against
    the range of the variable the model was trained on, so groups without a
    value of the variable are shown too.

    result : drc_predictive.run_age_model() output.
    groups : {sample: group} for colours.
    """
    from matplotlib.figure import Figure

    pred = result['predictions']
    groups = groups or {}
    labels = [str(groups.get(s, '')) for s in pred['sample']]
    colours = _group_colour_map(list(dict.fromkeys(labels)), group_colours)
    fg = 'white' if is_dark else 'black'
    var_label = variable_label(variable, unit)
    by_group = len(set(labels)) > 1
    fig = Figure(figsize=(17.0 if by_group else 13.0, 4.6), layout='constrained')
    axes = fig.subplots(1, 4 if by_group else 3)
    ax_pred, ax_gap, ax_null = axes[0], axes[1], axes[2]
    cv = pred['cross_validated'].to_numpy(bool)
    hover_xy, hover_labels = [], []
    for g in dict.fromkeys(labels):
        m = np.array([x == g for x in labels])
        known = pred['actual'].notna().to_numpy(bool)
        for inside, marker_face in ((True, None), (False, 'none')):
            sel = m & (cv == inside) & known
            if not sel.any():
                continue
            colour = colours[g]
            name = (g or 'samples') + ('' if inside else ' (not in the model)')
            kw = dict(s=26, edgecolors=colour if marker_face == 'none' else fg, linewidths=0.6,
                      facecolors=marker_face or colour, label=f"{name} (n={int(sel.sum())})",
                      zorder=3)
            ax_pred.scatter(pred['actual'][sel], pred['predicted'][sel], **kw)
            ax_gap.scatter(pred['actual'][sel], pred['gap_corrected'][sel], **{**kw, 'label': None})
            for s_, a, b in zip(pred['sample'][sel], pred['actual'][sel], pred['predicted'][sel]):
                hover_xy.append((a, b))
                hover_labels.append(f"{str(s_).replace(chr(92), '/').rsplit('/', 1)[-1]}: "
                                    f"{b:.3g} (actual {a:.3g})")
    finite = pred[['actual', 'predicted']].to_numpy(float)
    finite = finite[np.isfinite(finite).all(axis=1)]
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        pad = 0.05 * ((hi - lo) or 1.0)
        ax_pred.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color='grey', linewidth=0.8)
        ax_pred.set_xlim(lo - pad, hi + pad)
        ax_pred.set_ylim(lo - pad, hi + pad)
    ax_pred.set_xlabel(dp.wrap_to_width(var_label, 10, 3.2))
    predicted_label = dp.wrap_to_width(
        f"Predicted {variable} ({unit})" if unit else f"Predicted {variable}", 10, 3.2)
    ax_pred.set_ylabel(predicted_label)
    ax_pred.set_title("Predicted (out-of-fold) against actual", fontsize=9)
    ax_pred.legend(fontsize=7, loc='upper left', frameon=False)
    ax_gap.axhline(0.0, color='grey', linewidth=0.8)
    ax_gap.set_xlabel(dp.wrap_to_width(var_label, 10, 3.2))
    ax_gap.set_ylabel("Predicted − actual, corrected")
    ax_gap.set_title("Profile like a higher (+) or lower (−) value", fontsize=9)
    chance = result['chance']['mae'].to_numpy(float) if len(result.get('chance', [])) else np.array([])
    mae = float(result['performance'].iloc[0]['mae_mean'])
    if chance.size:
        ax_null.hist(chance, bins=min(30, max(5, chance.size // 4)), color='#bbbbbb')
        ax_null.set_title("Chance check: the variable shuffled", fontsize=9)
    else:
        ax_null.set_title("No chance check run", fontsize=9)
    ax_null.axvline(mae, color=UP_COLOUR, linewidth=2, label=f"real: {mae:.3g}")
    ax_null.set_xlabel(f"Mean absolute error ({unit})" if unit else "Mean absolute error")
    ax_null.set_ylabel("Shuffles")
    ax_null.legend(fontsize=7, frameon=False)
    if by_group:
        ax_grp = axes[3]
        rng = np.random.default_rng(0)
        order = list(dict.fromkeys(labels))
        predicted = pred['predicted'].to_numpy(float)
        trained = pred.loc[cv, 'actual'].to_numpy(float)
        trained = trained[np.isfinite(trained)]
        if trained.size:
            ax_grp.axhspan(float(trained.min()), float(trained.max()), color='grey', alpha=0.15,
                           linewidth=0, label="Range trained on")
        for i, g in enumerate(order):
            m = np.array([x == g for x in labels]) & np.isfinite(predicted)
            if not m.any():
                continue
            inside = bool(cv[m].all())
            x = i + rng.uniform(-0.18, 0.18, int(m.sum()))
            ax_grp.scatter(x, predicted[m], s=20, facecolors=colours[g] if inside else 'none',
                           edgecolors=colours[g], linewidths=0.8, zorder=3)
            ax_grp.plot([i - 0.3, i + 0.3], [predicted[m].mean()] * 2, color=fg, linewidth=2,
                        zorder=4)
        ax_grp.set_xticks(range(len(order)),
                          [dp.wrap_to_width(g + ('' if cv[[x == g for x in labels]].all()
                                                 else ' (predicted)'), 8, 1.4) for g in order],
                          fontsize=8)
        ax_grp.set_ylabel(predicted_label)
        ax_grp.set_title("Predicted by group (bar: mean)", fontsize=9)
        if trained.size:
            ax_grp.legend(fontsize=7, frameon=False, loc='best')
    fig.suptitle(dp.wrap_to_width(f"{title}\n{age_model_subtitle(result)}", 9,
                                  16.5 if by_group else 12.5), fontsize=9)
    if hover_xy:
        hover = np.asarray(hover_xy, dtype=float)
        overlay = ax_pred.scatter(hover[:, 0], hover[:, 1], s=60, alpha=0, zorder=6)
        fig._hover_handler = dp.make_scatter_hover_handler(fig, ax_pred, overlay, hover_labels,
                                                           is_dark)
    dp.style_figure_theme(fig, is_dark)
    dp.stamp_run_label(fig, run_label)
    return fig


def make_age_features_figure(result: dict, *, variable: str, top: int = AGE_FEATURES_SHOWN,
                             title: str = 'Immune age model: features', is_dark: bool = False,
                             run_label: str = '', keys: dict | None = None,
                             kinds: dict | None = None):
    """
    The features carrying the prediction: the *top* most often kept by the
    elastic net across the outer fits, as bars of their mean standardised
    coefficient (red: higher at larger values of the variable, blue:
    lower), with how often each was kept.

    keys, kinds : {feature: key} and {feature: kind} for click-through; the
                  figure then carries ``_row_keys`` (and ``_row_kinds``) by
                  bar position, bottom to top, and ``_row_axes``.
    """
    from matplotlib.figure import Figure

    imp = result['importance']
    imp = imp[imp['selection'] > 0].head(int(top))
    if imp.empty:
        return dp.message_figure("The model kept no feature.", title, is_dark, run_label)
    names = [dp.wrap_to_width(n, 7, 4.5) for n in imp['feature'].astype(str)]
    label_w = max(dp.text_width_inches(line, 7) for n in names for line in n.split('\n')) + 0.3
    lines = sum(n.count('\n') + 1 for n in names)
    width = max(6.0, 4.2 + label_w)
    title_text = dp.wrap_to_width(f"{title} ({variable})", 9, width - 0.6)
    xlabel = dp.wrap_to_width(
        "Mean standardised coefficient (per SD of the feature). +: higher at larger values "
        "of the variable; % = kept in the outer fits", 8, width - label_w - 0.6)
    # Room for the bars plus the title, the x label, tick labels and the run
    # label band, so that nothing is cut off at 100% or when exported.
    text_h = (title_text.count('\n') + 1) * 9 * 1.35 / 72 + (xlabel.count('\n') + 1) * 8 * 1.35 / 72
    fig = Figure(figsize=(width, 0.2 * lines + 0.06 * len(names) + text_h + 1.2),
                 layout='constrained')
    ax = fig.add_subplot(111)
    y = np.arange(len(imp))[::-1]
    coef = imp['coef'].to_numpy(float)
    ax.barh(y, coef, color=[UP_COLOUR if c > 0 else DOWN_COLOUR for c in coef])
    ax.grid(False)
    ax.set_yticks(y, names, fontsize=7)
    ax.axvline(0.0, color='grey', linewidth=0.8)
    for yi, c, sel in zip(y, coef, imp['selection'].to_numpy(float)):
        ax.text(c, yi, f" {sel:.0%} ", va='center', ha='left' if c >= 0 else 'right', fontsize=6)
    span = max(float(np.abs(coef).max()), 1e-12)
    ax.set_xlim(min(0.0, float(coef.min())) - 0.18 * span, max(0.0, float(coef.max())) + 0.18 * span)
    ax.set_xlabel(xlabel, fontsize=8)
    features = imp['feature'].astype(str).tolist()[::-1]
    if keys:
        fig._row_keys = [keys.get(f, f) for f in features]
        fig._row_axes = ax
        if kinds:
            fig._row_kinds = [kinds.get(f) for f in features]
    ax.set_title(title_text, fontsize=9)
    dp.style_figure_theme(fig, is_dark)
    dp.stamp_run_label(fig, run_label)
    return fig


# ---------------------------------------------------------------------------
# Samples coloured by a numeric variable
# ---------------------------------------------------------------------------

def scatter_by_value(fig, ax, x, y, values, groups, *, label: str, point_size: float = 60,
                     is_dark: bool = False):
    """
    Draw samples at (x, y) coloured by a numeric *values* (viridis, with a
    colour bar named *label*) and shaped by group. Samples without a value
    are grey crosses. Returns the scatter of the coloured points (None when
    no sample has a value).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    vals = pd.to_numeric(pd.Series(list(values)), errors='coerce').to_numpy(float)
    groups = [str(g) for g in groups]
    fg = 'white' if is_dark else 'black'
    has = np.isfinite(vals)
    vmin, vmax = (float(np.nanmin(vals)), float(np.nanmax(vals))) if has.any() else (0.0, 1.0)
    mappable = None
    for i, g in enumerate(dict.fromkeys(groups)):
        m = np.array([h == g for h in groups])
        marker = GROUP_MARKERS[i % len(GROUP_MARKERS)]
        if (m & has).any():
            mappable = ax.scatter(x[m & has], y[m & has], c=vals[m & has], cmap='viridis',
                                  vmin=vmin, vmax=vmax, s=point_size, marker=marker,
                                  edgecolors=fg, linewidths=0.4, zorder=3)
        ax.scatter([], [], marker=marker, color='none', edgecolors=fg, label=g or 'no group')
        if (m & ~has).any():
            ax.scatter(x[m & ~has], y[m & ~has], s=point_size * 0.7, marker='x',
                       color=NOT_CALLED_COLOUR, zorder=3)
    if (~has).any():
        ax.scatter([], [], marker='x', color=NOT_CALLED_COLOUR, label='no value')
    if mappable is not None:
        fig.colorbar(mappable, ax=ax, label=label, shrink=0.8)
    return mappable
