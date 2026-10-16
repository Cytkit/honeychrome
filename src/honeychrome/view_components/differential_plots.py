"""
differential_plots.py — figures for differential statistics results
===================================================================

Matplotlib figure builders shared by the DR/Clustering and Automated Gating
plugins. Each returns a ``matplotlib.figure.Figure`` without a canvas, so it
can be built on a worker thread and embedded later (e.g. in
``ExportablePlotWidget``).

* ``make_heatmap_figure`` — per-sample values of the significant features,
  with row and column dendrograms and a group colour bar.
* ``make_volcano_figure`` — effect size against adjusted p-value, 95% CI
  bars and a hover tooltip on significant points. Given the two groups, a
  ramp below the x axis shows which side is higher in which group.
* ``make_marker_summary_figure`` — population (or cluster) × marker grid of
  intensity differences, significant markers marked with a dot.
* ``make_comparison_heatmap_figure`` — comparison × feature grid of effect
  sizes, coloured only where significant: every comparison of a run at once.

Hover tooltips need a canvas: a figure builder stores its handler as
``fig._hover_handler``; connect it with
``fig.canvas.mpl_connect('motion_notify_event', fig._hover_handler)`` once
the figure is on a canvas.
"""

from __future__ import annotations

import functools
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    'style_figure_theme', 'make_scatter_hover_handler', 'stamp_run_label',
    'message_figure', 'make_heatmap_figure', 'make_volcano_figure',
    'group_pair_colours', 'draw_direction_ramp',
    'make_marker_summary_figure', 'make_comparison_heatmap_figure', 'significance_column',
    'wrap_label', 'text_width_inches', 'cell_grid_lines',
]

GROUP_COLOUR_A = '#4477AA'
GROUP_COLOUR_B = '#EE6677'
MAX_STATIC_LABELS = 10
VOLCANO_LABEL_WRAP = 24     # characters per line for volcano point labels


def wrap_label(text: str, width: int, max_lines: int | None = None) -> str:
    """
    Break *text* onto lines of at most *width* characters, splitting at
    spaces (long cluster labels such as MEM labels are space-separated).
    A single word longer than *width* stays whole. With *max_lines*, the
    words that do not fit are folded into the last line instead of
    adding further lines.
    """
    text = str(text)
    lines = textwrap.wrap(text, width=max(1, width), break_long_words=False,
                          break_on_hyphens=False) or [text]
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines - 1] + [' '.join(lines[max_lines - 1:])]
    return '\n'.join(lines)


def text_width_inches(text: str, fontsize: float) -> float:
    """
    Rendered width in inches of the widest line of *text* at *fontsize*
    points, from matplotlib's default font's advance widths. Needs no
    canvas, so it is safe on a worker thread.
    """
    widest = 0.0
    for line in str(text).split('\n'):
        if line:
            widest = max(widest, _line_width_points(line, float(fontsize)))
    return widest / 72.0


@functools.lru_cache(maxsize=8192)
def _line_width_points(line: str, fontsize: float) -> float:
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import text_to_path
    width, _height, _descent = text_to_path.get_text_width_height_descent(
        line, FontProperties(size=fontsize), ismath=False)
    return float(width)


def cell_grid_lines(ax, n_cols: int, n_rows: int, color: str, linewidth: float):
    """Lines between the cells of an image drawn with imshow on *ax*
    (cell centres at integer positions), as two line collections rather
    than one minor tick per boundary, which is far quicker to build and
    draw for large grids. The axis limits are left as they were."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x_edges = np.arange(-0.5, n_cols, 1)
    y_edges = np.arange(-0.5, n_rows, 1)
    ax.vlines(x_edges, -0.5, n_rows - 0.5, colors=color, linewidth=linewidth)
    ax.hlines(y_edges, -0.5, n_cols - 0.5, colors=color, linewidth=linewidth)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def _is_default_text_colour(colour) -> bool:
    """True when *colour* is what text gets when nothing sets it: opaque
    black, or the active style's ``text.color`` (the seaborn theme the
    plot widgets apply makes this a dark grey)."""
    import matplotlib
    from matplotlib.colors import to_rgba
    try:
        rgba = tuple(to_rgba(colour))
        defaults = {(0.0, 0.0, 0.0, 1.0), tuple(to_rgba(matplotlib.rcParams['text.color']))}
    except ValueError:
        return False
    return rgba in defaults


def _recolour_default_text(fig, axes, fg: str, bg: str):
    """
    Make text readable on a dark background. Axis labels, ticks and titles
    are coloured explicitly by style_figure_theme; this covers the other
    text artists, which matplotlib leaves black unless told otherwise: text
    and annotations placed on an axis, figure-level text and suptitles, and
    legends. Only text still at the default black is changed, and text with
    its own background box is left alone, so colours chosen on purpose
    survive.
    """
    texts = list(fig.texts)
    suptitle = getattr(fig, '_suptitle', None)
    if suptitle is not None:
        texts.append(suptitle)
    legends = list(fig.legends)
    for ax in axes:
        texts.extend(ax.texts)
        legend = ax.get_legend()
        if legend is not None:
            legends.append(legend)
    for text in texts:
        if text.get_bbox_patch() is None and _is_default_text_colour(text.get_color()):
            text.set_color(fg)
    for legend in legends:
        if legend.get_frame_on():
            frame = legend.get_frame()
            frame.set_facecolor(bg)
            frame.set_edgecolor('#888888')
        for text in list(legend.get_texts()) + [legend.get_title()]:
            if _is_default_text_colour(text.get_color()):
                text.set_color(fg)


def style_figure_theme(fig, is_dark: bool, axes=None) -> str:
    """
    Apply dark or light background styling to every axis of *fig* (or to
    *axes*), including figure-level text, annotations and legends in dark
    mode. Returns the foreground colour so callers can reuse it.
    """
    if axes is None:
        axes = fig.axes
    if is_dark:
        fig.patch.set_facecolor('#1e1e1e')
        fg, bg = 'white', '#2b2b2b'
    else:
        fig.patch.set_facecolor('white')
        fg, bg = 'black', 'white'
    for ax in axes:
        ax.set_facecolor(bg)
        ax.tick_params(colors=fg)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)
        for spine in ax.spines.values():
            spine.set_edgecolor(fg)
    if is_dark:
        _recolour_default_text(fig, axes, fg, bg)
    return fg


def make_scatter_hover_handler(fig, ax, scatter, labels: list[str], is_dark: bool):
    """
    Build a matplotlib 'motion_notify_event' handler that shows labels[i]
    in a small annotation box while the mouse is over point i of
    *scatter*. The caller connects it once the figure has a canvas.
    """
    annot = ax.annotate(
        '', xy=(0, 0), xytext=(12, 12), textcoords='offset points',
        fontsize=7,
        bbox=dict(boxstyle='round', fc='#333333' if is_dark else '#ffffe0',
                  ec='#888888', alpha=0.95),
        color='white' if is_dark else 'black',
        arrowprops=dict(arrowstyle='-', color='#888888'),
    )
    annot.set_visible(False)
    annot.set_in_layout(False)

    def _on_hover(event):
        if event.inaxes != ax:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
            return
        cont, ind = scatter.contains(event)
        if cont:
            idx = ind['ind'][0]
            annot.xy = scatter.get_offsets()[idx]
            annot.set_text(wrap_label(labels[idx], VOLCANO_LABEL_WRAP * 2))
            annot.set_visible(True)
            fig.canvas.draw_idle()
        elif annot.get_visible():
            annot.set_visible(False)
            fig.canvas.draw_idle()

    return _on_hover


def group_pair_colours(group_a: str, group_b: str,
                       group_colours: dict | None = None) -> tuple[str, str]:
    """Colours of the two groups of a comparison: those in *group_colours*
    ({group: colour}), otherwise the module's pair of defaults."""
    colours = group_colours or {}
    return (colours.get(group_a) or GROUP_COLOUR_A, colours.get(group_b) or GROUP_COLOUR_B)


def _fit_text(text: str, fontsize: float, max_inches: float) -> str:
    """*text* shortened with a trailing ellipsis character until it is at
    most *max_inches* wide at *fontsize* points."""
    text = str(text)
    if text_width_inches(text, fontsize) <= max_inches:
        return text
    while len(text) > 1 and text_width_inches(text + '\u2026', fontsize) > max_inches:
        text = text[:-1]
    return text + '\u2026'


def _text_colour_on(fill: str) -> str:
    """Black or white, whichever reads better on the colour *fill*."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(fill)
    return 'black' if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else 'white'


def draw_direction_ramp(ax, x_limit: float, group_a: str, group_b: str,
                        colour_a: str, colour_b: str, max_label_inches: float = 1.5):
    """
    Group key for an effect-size axis: a flat rectangle across *ax* split
    along its diagonal into two ramps. Group A's ramp (the left side,
    where the effect is lower) is fat at the left end, group B's (the
    right side, where it is higher) fat at the right end, each in its
    group's colour and named in its fat end. *ax* spans the effect axis
    from -x_limit to x_limit.
    """
    from matplotlib.patches import Polygon

    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(0.0, 1.0)
    ax.axis('off')
    ax.add_patch(Polygon([(-x_limit, 0.0), (-x_limit, 1.0), (x_limit, 1.0)], closed=True,
                         facecolor=colour_a, edgecolor='none'))
    ax.add_patch(Polygon([(-x_limit, 0.0), (x_limit, 0.0), (x_limit, 1.0)], closed=True,
                         facecolor=colour_b, edgecolor='none'))
    pad = 0.02 * x_limit
    ax.text(-x_limit + pad, 0.68, _fit_text(f"higher in {group_a}", 7, max_label_inches),
            ha='left', va='center', fontsize=7, color=_text_colour_on(colour_a))
    ax.text(x_limit - pad, 0.32, _fit_text(f"higher in {group_b}", 7, max_label_inches),
            ha='right', va='center', fontsize=7, color=_text_colour_on(colour_b))


RUN_LABEL_BAND_INCHES = 0.3


def stamp_run_label(fig, run_label: str):
    """Write the source run's label in the figure's upper-left corner. A
    figure laid out by constrained layout gives the label a band of its
    own along the top, so it never covers a title or an axis."""
    if not run_label:
        return
    engine = fig.get_layout_engine()
    if engine is not None and type(engine).__name__ == 'ConstrainedLayoutEngine':
        engine.set(rect=(0.0, 0.0, 1.0, 1.0 - RUN_LABEL_BAND_INCHES / fig.get_figheight()))
    fig.text(
        0.01, 0.99, run_label,
        ha='left', va='top', fontsize=8, color='#555555',
        transform=fig.transFigure,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                  edgecolor='#cccccc', alpha=0.85),
    )


def message_figure(text: str, title: str, is_dark: bool, run_label: str = ''):
    """Small figure showing *text* in place of a plot."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5, 2), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=10,
            transform=ax.transAxes)
    ax.set_title(title, fontsize=10)
    style_figure_theme(fig, is_dark)
    stamp_run_label(fig, run_label)
    return fig


def significance_column(results_df: pd.DataFrame, fdr_scope: str = 'global') -> tuple[str, str]:
    """
    ``(column, axis label)`` of the adjusted p-value that decides
    significance: the stage-wise value for family-first results, otherwise
    the pooled or per-comparison BH value matching *fdr_scope*.
    """
    if 'stagewise.adj.P.Val' in results_df.columns:
        return 'stagewise.adj.P.Val', "-log10(stage-wise adj. P)"
    if fdr_scope == 'global' and 'adj.P.Val.global' in results_df.columns:
        return 'adj.P.Val.global', "-log10(adj. P, pooled)"
    if 'adj.P.Val' in results_df.columns:
        return 'adj.P.Val', "-log10(adj. P-value)"
    return 'P.Value', "-log10(P-value)"


def _fixed_margin_layout(gridspec, left: float, right: float, top: float, bottom: float):
    """
    Layout engine that keeps a figure's outer margins at fixed sizes in
    inches (*left*, *right*, *top*, *bottom*) and lets *gridspec* fill the
    rest. It places the axes directly from the current figure size on every
    draw, so it stays valid at any canvas size and, unlike constrained
    layout, never has to solve for a figure whose rows span a very wide
    range of heights (a heatmap of several hundred rows).
    """
    from matplotlib.layout_engine import LayoutEngine

    class _FixedMarginEngine(LayoutEngine):
        _adjust_compatible = False
        _colorbar_gridspec = False

        def execute(self, fig):
            width, height = fig.get_size_inches()
            lo_x = min(left / width, 0.3)
            hi_x = min(max(1.0 - right / width, lo_x + 0.1), 1.0)
            lo_y = min(bottom / height, 0.3)
            hi_y = min(max(1.0 - top / height, lo_y + 0.1), 1.0)
            gridspec.update(left=lo_x, right=hi_x, bottom=lo_y, top=hi_y)
            # GridSpec.update only moves the axes of pyplot-managed figures.
            for ax in fig.axes:
                spec = ax.get_subplotspec()
                if spec is not None and spec.get_topmost_subplotspec().get_gridspec() is gridspec:
                    ax._set_position(spec.get_position(fig))

    return _FixedMarginEngine()


def make_heatmap_figure(results_df: pd.DataFrame, sample_df: pd.DataFrame, title: str,
                        group_by_sample: dict, group_a: str = '', group_b: str = '',
                        is_dark: bool = False, run_label: str = '',
                        value_label: str | None = None,
                        feature_labels: dict | None = None,
                        group_colours: dict | None = None):
    """
    Per-sample heatmap of the significant features, with row and column
    dendrograms and a two-group colour bar. The group key sits below the
    plot, clear of the run label and the title.
    Values that could not be measured in a sample (NaN) are drawn grey.

    results_df      : test output for ONE comparison (feature, significant, …).
    sample_df       : values for that comparison's samples (rows = samples,
                      columns = features), indexed by sample id.
    group_by_sample : {sample id: group name}.
    group_a/group_b : the two groups this comparison represents.
    value_label     : colour-bar label. Default: inferred from *title*
                      ('MFI', 'Counts', otherwise '% frequency').
    feature_labels  : optional {feature: display label} for the row labels.
    group_colours   : optional {group: colour}; default two fixed colours.
    """
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec
    import matplotlib.patches as mpatches
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist

    sig = (results_df[results_df['significant'].astype(bool)].copy()
           if 'significant' in results_df.columns else pd.DataFrame())
    if sig.empty:
        return message_figure('No significant features at current thresholds',
                              title, is_dark, run_label)

    sig_features = sig['feature'].tolist()
    cols_present = [c for c in sig_features if c in sample_df.columns]
    if not cols_present:
        return message_figure('Feature columns not found in sample matrix',
                              title, is_dark, run_label)

    mat = sample_df[cols_present].values.astype(float)   # (n_samples, n_sig_features)
    # Samples where a feature could not be measured are NaN: they are
    # drawn grey, and filled with the feature's mean only to order the
    # dendrograms.
    with np.errstate(invalid='ignore'):
        feature_means = np.where(np.isfinite(mat).any(axis=0), np.nanmean(mat, axis=0), 0.0)
    mat_for_clustering = np.where(np.isfinite(mat), mat, feature_means[None, :])
    sample_labels = list(sample_df.index)
    n_samples, n_features = mat.shape

    def _hclust(data, metric='euclidean', method='ward'):
        """(leaf_order, linkage matrix or None)."""
        if data.shape[0] < 2:
            return list(range(data.shape[0])), None
        try:
            Z = linkage(pdist(data, metric=metric), method=method)
            dend = dendrogram(Z, no_plot=True)
            return dend['leaves'], Z
        except Exception:
            return list(range(data.shape[0])), None

    row_order, Zr = _hclust(mat_for_clustering.T)   # features
    col_order, Zc = _hclust(mat_for_clustering)     # samples

    mat_ord = mat[:, row_order][col_order, :].T   # (n_features, n_samples)
    labels = feature_labels or {}
    feat_labels = [labels.get(cols_present[i], cols_present[i]) for i in row_order]
    samp_labels = [sample_labels[i] for i in col_order]
    grp_ord = [group_by_sample.get(sample_labels[i], group_a) for i in col_order]

    # Sizes in inches: the plot's rows and columns, then the margins
    # around them (title and run label above; colour bar labels on the
    # right; the group key below).
    col_w = 0.55
    row_h = 0.30
    dend_h = 1.2
    grp_h = 0.18
    xlabel_h = 0.8
    dend_w = 1.0
    label_w = max(text_width_inches(str(f), 7) for f in feat_labels) * 1.2 + 0.25
    cbar_w = 0.3
    title_text = wrap_label(title, max(20, int((5.0 + n_samples * col_w) / 0.09)))
    left_m = 0.2
    right_m = 1.0
    top_m = (RUN_LABEL_BAND_INCHES if run_label else 0.1) + 0.22 * (title_text.count('\n') + 1) + 0.15
    two_columns = text_width_inches(f"{group_a}        {group_b}", 7) + 1.0 < (
        n_samples * col_w + dend_w + label_w + cbar_w)
    bottom_m = 0.4 if two_columns else 0.65
    fig_w = max(5.0, left_m + dend_w + n_samples * col_w + label_w + cbar_w + right_m)
    fig_h = max(4.0, top_m + dend_h + grp_h + n_features * row_h + xlabel_h + bottom_m)

    fig = Figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        4, 4,
        figure=fig,
        height_ratios=[dend_h, grp_h, n_features * row_h, xlabel_h],
        width_ratios=[dend_w, n_samples * col_w, label_w, cbar_w],
        hspace=0.02,
        wspace=0.0,
    )
    fig.set_layout_engine(_fixed_margin_layout(gs, left_m, right_m, top_m, bottom_m))

    ax_cdend = fig.add_subplot(gs[0, 1])
    ax_cdend.axis('off')
    if n_samples > 1 and Zc is not None:
        try:
            dendrogram(Zc, ax=ax_cdend, color_threshold=0,
                       above_threshold_color='#555555',
                       link_color_func=lambda _: '#555555',
                       no_labels=True)
            ax_cdend.set_xlim(-0.5, n_samples * 10 - 0.5)
        except Exception:
            pass
    ax_cdend.set_title(title_text, fontsize=10, pad=4)

    ax_grp = fig.add_subplot(gs[1, 1])
    ax_grp.set_xlim(0, n_samples)
    ax_grp.set_ylim(0, 1)
    ax_grp.axis('off')
    colour_a, colour_b = group_pair_colours(group_a, group_b, group_colours)
    for xi, grp in enumerate(grp_ord):
        colour = colour_a if grp == group_a else colour_b
        ax_grp.add_patch(mpatches.Rectangle((xi, 0), 1, 1, color=colour,
                                            transform=ax_grp.transData))
    handles = [
        mpatches.Patch(color=colour_a, label=group_a),
        mpatches.Patch(color=colour_b, label=group_b),
    ]
    fig.legend(handles=handles, loc='lower center', fontsize=7,
               frameon=True, ncol=2 if two_columns else 1)

    ax_rdend = fig.add_subplot(gs[2, 0])
    ax_rdend.axis('off')
    if n_features > 1 and Zr is not None:
        try:
            dendrogram(Zr, ax=ax_rdend, orientation='left',
                       color_threshold=0,
                       above_threshold_color='#555555',
                       link_color_func=lambda _: '#555555',
                       no_labels=True)
        except Exception:
            pass

    ax_hm = fig.add_subplot(gs[2, 1])
    # Per-sample values, not differences: scale to the data's own range.
    finite_values = mat_ord[np.isfinite(mat_ord)]
    vmin = float(np.percentile(finite_values, 5)) if finite_values.size else 0.0
    vmax = max(float(np.percentile(finite_values, 95)) if finite_values.size else 1.0,
               vmin + 0.01)
    import matplotlib
    cmap = matplotlib.colormaps['viridis'].copy()
    cmap.set_bad('#bdbdbd')
    # One quad per cell: unlike an image, draw time does not grow with the
    # canvas size, which matters for heatmaps of several hundred rows.
    im = ax_hm.pcolormesh(np.arange(-0.5, n_samples, 1), np.arange(-0.5, n_features, 1),
                          np.ma.masked_invalid(mat_ord), cmap=cmap, vmin=vmin, vmax=vmax)
    ax_hm.set_xlim(-0.5, n_samples - 0.5)
    ax_hm.set_ylim(n_features - 0.5, -0.5)
    ax_hm.set_xticks(range(n_samples))
    ax_hm.set_xticklabels([Path(str(s)).stem for s in samp_labels],
                          rotation=45, ha='right', fontsize=7)
    ax_hm.set_yticks(range(n_features))
    ax_hm.set_yticklabels(feat_labels, fontsize=7)
    ax_hm.yaxis.set_label_position('right')
    ax_hm.yaxis.tick_right()
    ax_hm.grid(False)
    cell_grid_lines(ax_hm, n_samples, n_features, 'white', 0.6)

    ax_cb = fig.add_subplot(gs[2, 3])
    cb = fig.colorbar(im, cax=ax_cb)
    cb.ax.tick_params(labelsize=7)
    if value_label is None:
        if 'MFI' in title:
            value_label = 'MFI (Transforms-tab scale)'
        elif 'Counts' in title:
            value_label = 'Event count'
        else:
            value_label = '% frequency'
    cb.set_label(value_label, fontsize=7)

    # Row identities, for clicking through to a feature.
    fig._row_axes = ax_hm
    fig._row_keys = [cols_present[i] for i in row_order]

    style_figure_theme(fig, is_dark)
    stamp_run_label(fig, run_label)
    return fig


def make_volcano_figure(results_df: pd.DataFrame, title: str,
                        pval_threshold: float, fc_threshold: float,
                        is_dark: bool = False, fdr_scope: str = 'global',
                        feature_labels: list[str] | None = None,
                        x_label: str | None = None, run_label: str = '',
                        group_a: str = '', group_b: str = '',
                        group_colours: dict | None = None):
    """
    Volcano plot: x = effect (log2 fold change, log2 odds ratio or intensity
    difference), y = -log10 of the adjusted p-value that decides
    significance. Significant points are coloured by that value and carry
    a 95% CI bar; the ten most significant are labelled, and every
    significant point has a hover tooltip. Non-significant points are grey.

    The x axis is symmetric about zero and sized to the plotted points and
    to the CI bars that are drawn (a bar far beyond the points is cut off
    at the axis edge); the CIs of non-significant points are not drawn and
    do not count.

    feature_labels : display label per row of results_df (default: 'feature').
    x_label        : default 'log2 Fold Change', or an intensity label
                     when 'MFI' is in the title.
    group_a/group_b: the comparison's baseline and other group; the effect
                     is higher in group_b on the right and in group_a on
                     the left. When both are given, a ramp below the plot
                     names them (see draw_direction_ramp).
    group_colours  : optional {group: colour} for the ramp.
    """
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec

    pval_col, y_label = significance_column(results_df, fdr_scope)
    logfc = results_df['logFC'].values.astype(float)
    neg_lp = -np.log10(np.maximum(results_df[pval_col].values.astype(float), 1e-300))
    sig = results_df['significant'].values.astype(bool) if 'significant' in results_df.columns else (
        (results_df['P.Value'] <= pval_threshold) & (results_df['logFC'].abs() >= fc_threshold)
    ).values
    features = list(feature_labels) if feature_labels is not None else results_df['feature'].tolist()

    show_ramp = bool(group_a and group_b)
    fig = Figure(figsize=(5, 5.4 if show_ramp else 5), layout='constrained')
    grid = GridSpec(2 if show_ramp else 1, 2 if sig.any() else 1, figure=fig,
                    height_ratios=[14, 1] if show_ramp else [1],
                    width_ratios=[1, 0.04] if sig.any() else [1])
    ax = fig.add_subplot(grid[0, 0])
    ax_ramp = fig.add_subplot(grid[1, 0]) if show_ramp else None
    fg = 'white' if is_dark else 'black'

    non_sig = ~sig
    ax.scatter(logfc[non_sig], neg_lp[non_sig], c='#aaaaaa', s=25, alpha=0.8, linewidths=0)

    if 'CI.L' in results_df.columns and 'CI.R' in results_df.columns and sig.any():
        ci_lo = results_df['CI.L'].values.astype(float)[sig]
        ci_hi = results_df['CI.R'].values.astype(float)[sig]
        valid_ci = np.isfinite(ci_lo) & np.isfinite(ci_hi)
        if valid_ci.any():
            xerr = np.vstack([
                np.maximum(logfc[sig][valid_ci] - ci_lo[valid_ci], 0.0),
                np.maximum(ci_hi[valid_ci] - logfc[sig][valid_ci], 0.0),
            ])
            ax.errorbar(
                logfc[sig][valid_ci], neg_lp[sig][valid_ci],
                xerr=xerr, fmt='none', ecolor=fg, elinewidth=0.6,
                alpha=0.35, capsize=0, zorder=1,
            )

    sig_scatter = None
    if sig.any():
        sig_scatter = ax.scatter(logfc[sig], neg_lp[sig], c=neg_lp[sig], cmap='viridis',
                                 s=25, alpha=0.9, linewidths=0)
        fig.colorbar(sig_scatter, cax=fig.add_subplot(grid[0, 1]), label=y_label)

    ax.axhline(-np.log10(pval_threshold), color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(fc_threshold, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(-fc_threshold, color='grey', linestyle='--', linewidth=0.8)

    ax.margins(y=0.15)     # room above the top points for wrapped labels
    sig_x = logfc[sig]
    sig_y = neg_lp[sig]
    sig_labels = [lbl for lbl, is_pt_sig in zip(features, sig) if is_pt_sig]
    if len(sig_labels) > MAX_STATIC_LABELS:
        top_idx = np.argsort(sig_y)[::-1][:MAX_STATIC_LABELS]
    else:
        top_idx = np.arange(len(sig_labels))
    for i in top_idx:
        # Labels on the right half extend leftwards so they stay inside the axes.
        on_right = sig_x[i] > 0
        note = ax.annotate(wrap_label(sig_labels[i], VOLCANO_LABEL_WRAP),
                           xy=(sig_x[i], sig_y[i]),
                           xytext=(-4 if on_right else 4, 4),
                           textcoords='offset points',
                           ha='right' if on_right else 'left', va='bottom',
                           fontsize=6, color=fg)
        note.set_in_layout(False)

    if sig_scatter is not None:
        fig._hover_handler = make_scatter_hover_handler(fig, ax, sig_scatter, sig_labels, is_dark)
        fig._hover_labels = sig_labels

    finite_fc = np.abs(logfc[np.isfinite(logfc)])
    max_fc = float(finite_fc.max()) if finite_fc.size else 0.0
    ci_extent = 0.0
    if 'CI.L' in results_df.columns and 'CI.R' in results_df.columns and sig.any():
        drawn_ci = np.concatenate([
            results_df['CI.L'].values.astype(float)[sig],
            results_df['CI.R'].values.astype(float)[sig],
        ])
        drawn_ci = np.abs(drawn_ci[np.isfinite(drawn_ci)])
        if drawn_ci.size:
            ci_extent = float(drawn_ci.max())
    ci_extent = min(ci_extent, 2.0 * max(max_fc, fc_threshold))
    x_lim = max(max_fc * 1.05, fc_threshold * 1.5, ci_extent * 1.05, 1e-6)
    ax.set_xlim(-x_lim, x_lim)
    if ax_ramp is not None:
        colour_a, colour_b = group_pair_colours(group_a, group_b, group_colours)
        draw_direction_ramp(ax_ramp, x_lim, group_a, group_b, colour_a, colour_b)

    if x_label is None:
        x_label = "Δ mean intensity (transformed units)" if 'MFI' in title else "log2 Fold Change"
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=10)
    style_figure_theme(fig, is_dark)
    stamp_run_label(fig, run_label)
    return fig


def make_marker_summary_figure(results_df: pd.DataFrame, title: str, channel_names: dict,
                               is_dark: bool = False, run_label: str = '',
                               family_label: str = 'cluster',
                               value_label: str = 'Δ mean intensity (● significant)'):
    """
    Family (cluster or population) × marker grid for ONE comparison's
    marker-intensity results, limited to families with at least one
    significant marker. Cell colour is the difference in transformed
    intensity (diverging, centred on 0); a dot marks significant markers.
    Row labels carry each family's FDR from the family-level screen when
    family-first testing was used.

    results_df needs 'cluster', 'channel', 'logFC' and 'significant'.
    """
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.colors import TwoSlopeNorm

    fg = 'white' if is_dark else 'black'

    needed = {'cluster', 'channel', 'logFC', 'significant'}
    if results_df is None or not needed.issubset(results_df.columns):
        return message_figure(f'Re-run statistics to see the {family_label} summary',
                              title, is_dark, run_label)
    sig = results_df[results_df['significant'].astype(bool)]
    if sig.empty:
        return message_figure('No significant markers at current thresholds',
                              title, is_dark, run_label)

    fam_df = results_df[results_df['cluster'].isin(sig['cluster'].unique())]
    order_key = ('cluster.adj.P.Val' if 'cluster.adj.P.Val' in results_df.columns
                 else 'P.Value')
    fam_order = (fam_df.groupby('cluster', sort=False)[order_key].min()
                 .sort_values(kind='mergesort').index.tolist())
    channels = list(dict.fromkeys(results_df['channel'].tolist()))

    grid = fam_df.pivot_table(index='cluster', columns='channel',
                              values='logFC', aggfunc='first')
    grid = grid.reindex(index=fam_order, columns=channels)
    sig_grid = fam_df.pivot_table(index='cluster', columns='channel',
                                  values='significant', aggfunc='first')
    sig_grid = sig_grid.reindex(index=fam_order, columns=channels)
    sig_mask = sig_grid.eq(True).values

    row_labels = []
    for fam in fam_order:
        if 'cluster.adj.P.Val' in fam_df.columns:
            q = fam_df.loc[fam_df['cluster'] == fam, 'cluster.adj.P.Val'].min()
            row_labels.append(f"{fam}  ({family_label} FDR {q:.2g})")
        else:
            row_labels.append(str(fam))
    col_labels = [channel_names.get(ch, ch) for ch in channels]

    values = grid.values.astype(float)
    vmax = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    vmax = vmax if vmax > 0 else 1.0

    n_rows, n_cols = values.shape
    label_w = max(len(r) for r in row_labels) * 0.07 + 0.4
    fig_w = max(5.0, n_cols * 0.45 + label_w + 1.5)
    fig_h = max(3.0, n_rows * 0.35 + 2.0)
    fig = Figure(figsize=(fig_w, fig_h), constrained_layout=True)
    ax = fig.add_subplot(111)
    cmap = matplotlib.colormaps['RdBu_r'].copy()
    cmap.set_bad('#bbbbbb')
    im = ax.imshow(np.ma.masked_invalid(values), aspect='auto', cmap=cmap,
                   norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
                   interpolation='nearest')
    ys, xs = np.nonzero(sig_mask)
    ax.scatter(xs, ys, s=14, c=fg, marker='o', linewidths=0)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.grid(False)
    cell_grid_lines(ax, n_cols, n_rows, 'white', 0.6)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(value_label, fontsize=7)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=10)
    # Row identities, for clicking through to a family.
    fig._row_axes = ax
    if 'cluster_id' in fam_df.columns:
        ids = fam_df.drop_duplicates('cluster').set_index('cluster')['cluster_id']
        fig._row_keys = [ids.get(f, f) for f in fam_order]
    else:
        fig._row_keys = list(fam_order)
    style_figure_theme(fig, is_dark)
    stamp_run_label(fig, run_label)
    return fig


def make_comparison_heatmap_figure(results_df: pd.DataFrame, title: str, is_dark: bool = False,
                                   run_label: str = '', order: str = 'effect', n_rows: int = 1,
                                   value_label: str = 'log2 fold change',
                                   feature_labels: dict | None = None,
                                   comparison_order: list[str] | None = None,
                                   show_all: bool = False):
    """
    Comparison × feature heatmap of effect sizes across every comparison
    of a results table. A cell is coloured by ``logFC`` where the feature
    is significant in that comparison and left white (0) otherwise, so the
    grid shows the same hits as the volcano plots. Diverging colours
    centred on 0, limits ± the largest |logFC| shown.

    results_df       : stacked results with 'feature', 'comparison', 'logFC'
                       and 'significant' (``apply_significance`` output).
    order            : 'effect' — features ordered by mean coloured logFC
                       across comparisons, most negative first;
                       'alphabetical' — by display label.
    n_rows           : split the features over this many stacked panels
                       (1–3), for long feature lists.
    feature_labels   : optional {feature: display label}.
    comparison_order : optional row order; default first appearance.
    show_all         : colour every cell by its logFC, significant or not.
    """
    import matplotlib
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.figure import Figure

    needed = {'feature', 'comparison', 'logFC', 'significant'}
    if results_df is None or results_df.empty or not needed.issubset(results_df.columns):
        return message_figure('Run statistics to see the comparison heatmap', title, is_dark, run_label)

    df = results_df.copy()
    keep = df['significant'].astype(bool) | bool(show_all)
    df['value'] = np.where(keep & np.isfinite(df['logFC'].astype(float)), df['logFC'].astype(float), 0.0)
    grid = df.pivot_table(index='comparison', columns='feature', values='value', aggfunc='first')
    comparisons = list(comparison_order or dict.fromkeys(df['comparison'].tolist()))
    comparisons = [c for c in comparisons if c in grid.index]
    labels = feature_labels or {}
    features = list(dict.fromkeys(df['feature'].tolist()))
    if order == 'alphabetical':
        features.sort(key=lambda f: str(labels.get(f, f)).lower())
    else:
        mean_effect = grid.mean(axis=0)
        features.sort(key=lambda f: (float(mean_effect.get(f, 0.0)), str(labels.get(f, f)).lower()))
    grid = grid.reindex(index=comparisons, columns=features).fillna(0.0)

    if not (grid.values != 0).any():
        return message_figure('No significant features at current thresholds', title, is_dark, run_label)

    vmax = float(np.nanmax(np.abs(grid.values)))
    vmax = vmax if vmax > 0 else 1.0
    n_rows = int(min(max(1, n_rows), 3, len(features)))
    per_row = int(np.ceil(len(features) / n_rows))
    chunks = [features[i * per_row:(i + 1) * per_row] for i in range(n_rows)]

    row_label_w = max(text_width_inches(c, 7) for c in comparisons) + 0.3
    fig_w = min(max(5.0, per_row * 0.28 + row_label_w + 1.2), 28.0)
    longest = max(len(str(labels.get(f, f))) for f in features)
    panel_h = len(comparisons) * 0.32 + min(longest, 30) * 0.055 + 0.3
    fig = Figure(figsize=(fig_w, max(2.5, panel_h * n_rows + 0.8)), constrained_layout=True)
    axes = fig.subplots(n_rows, 1, squeeze=False)[:, 0]
    cmap = matplotlib.colormaps['RdBu_r']
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    image = None
    for ax, chunk in zip(axes, chunks):
        values = grid[chunk].values
        image = ax.imshow(values, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
        ax.set_xticks(range(len(chunk)))
        ax.set_xticklabels([str(labels.get(f, f)) for f in chunk], rotation=90, fontsize=7)
        ax.set_yticks(range(len(comparisons)))
        ax.set_yticklabels(comparisons, fontsize=7)
        cell_grid_lines(ax, len(chunk), len(comparisons), '#898989', 0.4)
        if len(chunk) < per_row:
            ax.set_xlim(-0.5, per_row - 0.5)
    cb = fig.colorbar(image, ax=list(axes), shrink=0.8)
    cb.set_label(value_label if show_all else f'{value_label} (significant only)', fontsize=7)
    cb.ax.tick_params(labelsize=7)
    axes[0].set_title(title, fontsize=10)
    style_figure_theme(fig, is_dark)
    stamp_run_label(fig, run_label)
    return fig
