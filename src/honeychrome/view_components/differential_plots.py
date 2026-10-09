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
  bars and a hover tooltip on significant points.
* ``make_marker_summary_figure`` — population (or cluster) × marker grid of
  intensity differences, significant markers marked with a dot.

Hover tooltips need a canvas: a figure builder stores its handler as
``fig._hover_handler``; connect it with
``fig.canvas.mpl_connect('motion_notify_event', fig._hover_handler)`` once
the figure is on a canvas.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    'style_figure_theme', 'make_scatter_hover_handler', 'stamp_run_label',
    'message_figure', 'make_heatmap_figure', 'make_volcano_figure',
    'make_marker_summary_figure', 'significance_column',
    'wrap_label', 'text_width_inches',
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
    points, measured with matplotlib's default font. Needs no canvas, so
    it is safe on a worker thread.
    """
    from matplotlib.textpath import TextPath

    widest = 0.0
    for line in str(text).split('\n'):
        if not line:
            continue
        extent = TextPath((0, 0), line, size=fontsize).get_extents()
        widest = max(widest, float(extent.width))
    return widest / 72.0


def style_figure_theme(fig, is_dark: bool, axes=None) -> str:
    """
    Apply dark or light background styling to every axis of *fig* (or to
    *axes*). Returns the foreground colour so callers can reuse it.
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


def stamp_run_label(fig, run_label: str):
    """Write the source run's label in the figure's upper-left corner."""
    if not run_label:
        return
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


def make_heatmap_figure(results_df: pd.DataFrame, sample_df: pd.DataFrame, title: str,
                        group_by_sample: dict, group_a: str = '', group_b: str = '',
                        is_dark: bool = False, run_label: str = '',
                        value_label: str | None = None,
                        feature_labels: dict | None = None):
    """
    Per-sample heatmap of the significant features, with row and column
    dendrograms and a two-group colour bar.

    results_df      : test output for ONE comparison (feature, significant, …).
    sample_df       : values for that comparison's samples (rows = samples,
                      columns = features), indexed by sample id.
    group_by_sample : {sample id: group name}.
    group_a/group_b : the two groups this comparison represents.
    value_label     : colour-bar label. Default: inferred from *title*
                      ('MFI', 'Counts', otherwise '% frequency').
    feature_labels  : optional {feature: display label} for the row labels.
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

    row_order, Zr = _hclust(mat.T)              # features
    col_order, Zc = _hclust(mat)                # samples

    mat_ord = mat[:, row_order][col_order, :].T   # (n_features, n_samples)
    labels = feature_labels or {}
    feat_labels = [labels.get(cols_present[i], cols_present[i]) for i in row_order]
    samp_labels = [sample_labels[i] for i in col_order]
    grp_ord = [group_by_sample.get(sample_labels[i], group_a) for i in col_order]

    col_w = 0.55
    row_h = 0.30
    dend_h = 1.2
    grp_h = 0.18
    xlabel_h = 0.8
    label_w = max(len(str(f)) for f in feat_labels) * 0.07 + 0.3
    cbar_w = 0.5
    fig_w = max(5.0, n_samples * col_w + label_w + cbar_w + 1.2)
    fig_h = max(4.0, n_features * row_h + dend_h + grp_h + xlabel_h + 1.5)

    fig = Figure(figsize=(fig_w, fig_h), layout='constrained')
    gs = GridSpec(
        4, 3,
        figure=fig,
        height_ratios=[dend_h, grp_h, n_features * row_h, xlabel_h],
        width_ratios=[label_w, n_samples * col_w, cbar_w],
        hspace=0.02,
        wspace=0.02,
    )

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
    ax_cdend.set_title(title, fontsize=10, pad=4)

    ax_grp = fig.add_subplot(gs[1, 1])
    ax_grp.set_xlim(0, n_samples)
    ax_grp.set_ylim(0, 1)
    ax_grp.axis('off')
    for xi, grp in enumerate(grp_ord):
        colour = GROUP_COLOUR_A if grp == group_a else GROUP_COLOUR_B
        ax_grp.add_patch(mpatches.Rectangle((xi, 0), 1, 1, color=colour,
                                            transform=ax_grp.transData))
    handles = [
        mpatches.Patch(color=GROUP_COLOUR_A, label=group_a),
        mpatches.Patch(color=GROUP_COLOUR_B, label=group_b),
    ]
    ax_cdend.legend(handles=handles, loc='lower right', fontsize=7,
                    frameon=True, ncol=2)

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
    vmin = float(np.percentile(mat_ord, 5))
    vmax = max(float(np.percentile(mat_ord, 95)), vmin + 0.01)
    im = ax_hm.imshow(mat_ord, aspect='auto', cmap='viridis',
                      vmin=vmin, vmax=vmax, interpolation='nearest')
    ax_hm.set_xticks(range(n_samples))
    ax_hm.set_xticklabels([Path(str(s)).stem for s in samp_labels],
                          rotation=45, ha='right', fontsize=7)
    ax_hm.set_yticks(range(n_features))
    ax_hm.set_yticklabels(feat_labels, fontsize=7)
    ax_hm.yaxis.set_label_position('right')
    ax_hm.yaxis.tick_right()
    ax_hm.grid(False)
    ax_hm.set_xticks(np.arange(-0.5, n_samples, 1), minor=True)
    ax_hm.set_yticks(np.arange(-0.5, n_features, 1), minor=True)
    ax_hm.grid(which='minor', color='white', linestyle='-', linewidth=0.6)
    ax_hm.tick_params(which='minor', bottom=False, left=False, right=False)

    ax_cb = fig.add_subplot(gs[2, 2])
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

    style_figure_theme(fig, is_dark)
    stamp_run_label(fig, run_label)
    return fig


def make_volcano_figure(results_df: pd.DataFrame, title: str,
                        pval_threshold: float, fc_threshold: float,
                        is_dark: bool = False, fdr_scope: str = 'global',
                        feature_labels: list[str] | None = None,
                        x_label: str | None = None, run_label: str = ''):
    """
    Volcano plot: x = effect (log2 fold change, log2 odds ratio or intensity
    difference), y = -log10 of the adjusted p-value that decides
    significance. Significant points are coloured by that value and carry
    a 95% CI bar; the ten most significant are labelled, and every
    significant point has a hover tooltip. Non-significant points are grey.

    feature_labels : display label per row of results_df (default: 'feature').
    x_label        : default 'log2 Fold Change', or an intensity label
                     when 'MFI' is in the title.
    """
    from matplotlib.figure import Figure

    pval_col, y_label = significance_column(results_df, fdr_scope)
    logfc = results_df['logFC'].values.astype(float)
    neg_lp = -np.log10(np.maximum(results_df[pval_col].values.astype(float), 1e-300))
    sig = results_df['significant'].values.astype(bool) if 'significant' in results_df.columns else (
        (results_df['P.Value'] <= pval_threshold) & (results_df['logFC'].abs() >= fc_threshold)
    ).values
    features = list(feature_labels) if feature_labels is not None else results_df['feature'].tolist()

    fig = Figure(figsize=(5, 5), constrained_layout=True)
    ax = fig.add_subplot(111)
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
        fig.colorbar(sig_scatter, ax=ax, shrink=0.7, label=y_label)

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

    ci_extent = 0.0
    if 'CI.L' in results_df.columns and 'CI.R' in results_df.columns:
        finite_ci = np.concatenate([
            results_df['CI.L'].values.astype(float),
            results_df['CI.R'].values.astype(float),
        ])
        finite_ci = np.abs(finite_ci[np.isfinite(finite_ci)])
        if finite_ci.size:
            ci_extent = float(finite_ci.max())
    finite_fc = np.abs(logfc[np.isfinite(logfc)])
    max_fc = float(finite_fc.max()) if finite_fc.size else 0.0
    x_lim = max(max_fc * 1.05, fc_threshold * 1.5, ci_extent * 1.05, 1e-6)
    ax.set_xlim(-x_lim, x_lim)

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
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.6)
    ax.tick_params(which='minor', bottom=False, left=False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(value_label, fontsize=7)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=10)
    style_figure_theme(fig, is_dark)
    stamp_run_label(fig, run_label)
    return fig
