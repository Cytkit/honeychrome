"""
ag_fsea_plots.py — figures for Flow Set Enrichment Analysis
===========================================================
Pure matplotlib (no canvas), so every figure can be built on a worker
thread, styled like the Results page's figures
(``view_components.differential_plots``).

* ``make_dot_plot`` — immune states × comparisons: colour = NES (blue lost,
  red gained), size = leading-edge share of the tested members, a black
  ring where FDR is below the threshold.
* ``make_curve_figure`` — one state in one comparison: the running
  enrichment score, the members' positions in the ranking (coloured by
  library direction) and the ranking statistic.
* ``make_members_figure`` — the members' ranking statistics (times their
  library sign), leading edge highlighted.
* ``make_coherence_figure`` — each state's coherence in the two groups of a
  comparison, with 95% intervals (``ag_coherence``).
* ``make_state_network_figure`` — the state network of each group, lines
  that differ between the groups drawn solid.
* ``make_state_pair_figure`` — two states: each donor's two scores, and
  their member features' correlations drawn along the gate tree.

Every figure sets its own font sizes (the app's plot theme would
otherwise enlarge them) and is sized from its longest labels, with titles
and long axis labels wrapped to the space they have, so no text runs past
the figure's edge on screen or in an export.

The file name does not end in ``_tab.py``, so the plugin loader does not
treat it as a tab.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ['make_dot_plot', 'make_curve_figure', 'make_members_figure', 'make_coherence_figure',
           'make_state_network_figure', 'make_state_pair_figure']

_UP, _DOWN = '#B2182B', '#2166AC'


def make_dot_plot(states: pd.DataFrame, comparisons: list[str], fdr_threshold: float,
                  is_dark: bool = False, run_label: str = ''):
    """Dot plot of the tested states in every comparison."""
    from matplotlib.figure import Figure
    from honeychrome.view_components import differential_plots as dp

    tested = states[np.isfinite(states['NES'].astype(float))]
    if tested.empty:
        return dp.message_figure("No state could be tested", "Enrichment", is_dark, run_label)
    best = tested.groupby('immune_state')['NES'].apply(lambda v: float(np.nanmax(np.abs(v))))
    order = list(best.sort_values().index)
    comps = [c for c in comparisons if c in set(tested['comparison'])]
    state_w = max(dp.text_width_inches(s, 9) for s in order)
    comp_w = max((dp.text_width_inches(c, 8) for c in comps), default=0.0)
    base_h = max(3.0, 0.32 * len(order) + 1.6)
    # Comparison labels are rotated 30 degrees: half their width is height.
    fig_h = base_h + 0.5 * comp_w
    fig_w = max(4.5, state_w + 1.5 * len(comps) + 2.0)
    fig = Figure(figsize=(fig_w, fig_h), layout='constrained')
    ax = fig.add_subplot(111)
    lim = max(2.0, float(np.nanmax(np.abs(tested['NES']))))
    xs, ys, cs, ss, edge, labels = [], [], [], [], [], []
    for _i, r in tested.iterrows():
        if r['comparison'] not in comps:
            continue
        xs.append(comps.index(r['comparison']))
        ys.append(order.index(r['immune_state']))
        cs.append(float(r['NES']))
        share = float(r['n_leading']) / max(float(r['n_tested']), 1.0)
        ss.append(30 + 270 * share)
        sig = float(r['FDR']) < fdr_threshold
        edge.append('white' if (sig and is_dark) else ('black' if sig else 'none'))
        labels.append(f"{r['immune_state']} — {r['comparison']}\nNES {r['NES']:.2f}, "
                      f"p {r['p']:.3g}, FDR {r['FDR']:.3g}\nleading edge {int(r['n_leading'])}"
                      f" of {int(r['n_tested'])}")
    sc = ax.scatter(xs, ys, c=cs, s=ss, cmap='RdBu_r', vmin=-lim, vmax=lim,
                    edgecolors=edge, linewidths=1.4)
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels(comps, rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlim(-0.6, len(comps) - 0.4)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.grid(True, alpha=0.25)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6)
    cb.set_label(dp.wrap_to_width('NES (red gained, blue lost)', 8, 0.6 * (base_h - 1.3)),
                 fontsize=8)
    cb.ax.tick_params(labelsize=8)
    # A figure-level title: the plot sits right of the state names, so a
    # title centred over the plot alone would run off the right edge.
    fig.suptitle(dp.wrap_to_width(f"Immune state enrichment (ringed: FDR < {fdr_threshold:g}; "
                                  "size: leading-edge share)", 10, fig_w - 0.4), fontsize=10)
    fg = dp.style_figure_theme(fig, is_dark)
    cb.ax.yaxis.label.set_color(fg)
    cb.ax.tick_params(colors=fg)
    fig._hover_handler = dp.make_scatter_hover_handler(fig, ax, sc, labels, is_dark)
    dp.stamp_run_label(fig, run_label)
    return fig


def make_pattern_dot_plot(pattern_states: pd.DataFrame, fdr_threshold: float,
                          is_dark: bool = False, run_label: str = ''):
    """
    Immune states × shape patterns of a trend: dot colour = fold
    enrichment of the state's populations in the pattern (log2), size =
    share of the state's populations in the pattern, ringed when the
    hypergeometric FDR is below *fdr_threshold*.
    """
    from matplotlib.figure import Figure
    from honeychrome.view_components import differential_plots as dp

    df = pattern_states
    if df is None or df.empty:
        return dp.message_figure("No immune state overlaps a pattern", "Patterns", is_dark,
                                 run_label)
    patterns = list(dict.fromkeys(df['pattern']))
    best = df.groupby('immune_state')['p'].min().sort_values(ascending=False)
    order = list(best.index)
    state_w = max(dp.text_width_inches(s, 9) for s in order)
    fig_h = max(3.0, 0.32 * len(order) + 1.8)
    fig_w = max(4.5, state_w + 1.2 * len(patterns) + 2.2)
    fig = Figure(figsize=(fig_w, fig_h), layout='constrained')
    ax = fig.add_subplot(111)
    fold = np.log2(df['fold'].astype(float).clip(lower=1e-3))
    lim = max(1.0, float(np.nanmax(np.abs(fold))))
    edge = ['white' if (f < fdr_threshold and is_dark) else ('black' if f < fdr_threshold
                                                             else 'none') for f in df['FDR']]
    sc = ax.scatter([patterns.index(p) for p in df['pattern']],
                    [order.index(s) for s in df['immune_state']], c=fold, cmap='RdBu_r',
                    vmin=-lim, vmax=lim, s=30 + 270 * df['overlap'] / df['n_state'].clip(lower=1),
                    edgecolors=edge, linewidths=1.4)
    labels = [f"{r['immune_state']} — {r['pattern']}\n{int(r['overlap'])} of {int(r['n_state'])} "
              f"populations, fold {r['fold']:.2f}\np {r['p']:.3g}, FDR {r['FDR']:.3g}"
              for _i, r in df.iterrows()]
    ax.set_xticks(range(len(patterns)))
    ax.set_xticklabels(patterns, rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlim(-0.6, len(patterns) - 0.4)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.grid(True, alpha=0.25)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6)
    cb.set_label(dp.wrap_to_width('log2 fold enrichment in the pattern', 8, 0.6 * (fig_h - 1.3)),
                 fontsize=8)
    cb.ax.tick_params(labelsize=8)
    fig.suptitle(dp.wrap_to_width(f"Immune states in shape patterns (ringed: FDR < "
                                  f"{fdr_threshold:g}; size: share of the state's populations)",
                                  10, fig_w - 0.4), fontsize=10)
    fg = dp.style_figure_theme(fig, is_dark)
    cb.ax.yaxis.label.set_color(fg)
    cb.ax.tick_params(colors=fg)
    fig._hover_handler = dp.make_scatter_hover_handler(fig, ax, sc, labels, is_dark)
    dp.stamp_run_label(fig, run_label)
    return fig


def make_curve_figure(curve: dict, state: str, comparison: str, row: dict | None = None,
                      is_dark: bool = False, run_label: str = ''):
    """GSEA-style running-sum figure for one state.

    curve : ``ag_fsea.state_curve`` output (curve, positions, ordered, es,
            leading); row : the state's result row (NES, p, FDR) or None.
    """
    from matplotlib.figure import Figure
    from honeychrome.view_components import differential_plots as dp

    if curve is None:
        return dp.message_figure("No curve for this state", state, is_dark, run_label)
    ordered = np.asarray(curve['ordered'])
    G = ordered.size
    fig = Figure(figsize=(7, 5), layout='constrained')
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 0.6, 1.6])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    fg = 'white' if is_dark else 'black'
    x = np.arange(G + 1)
    ax1.plot(x, curve['curve'], color='#228833', lw=1.6)
    ax1.axhline(0, color=fg, lw=0.6)
    peak = int(np.argmax(np.abs(curve['curve'])))
    ax1.axvline(peak, color='#888888', ls='--', lw=0.8)
    ax1.set_ylabel('Running ES', fontsize=9)
    txt = f"ES {curve['es']:.2f}"
    if row:
        txt += f", NES {row['NES']:.2f}, p {row['p']:.3g}, FDR {row['FDR']:.3g}"
    ax1.set_title(dp.wrap_to_width(f"{state} — {comparison}\n{txt}", 10, 6.6), fontsize=10)
    pos = np.asarray(curve['positions'])
    signs = np.asarray(curve.get('signs', np.ones(pos.size)))
    lead = np.asarray(curve['leading'])
    for p_, s_, l_ in zip(pos, signs, lead):
        ax2.axvline(p_ + 0.5, color=_UP if s_ > 0 else _DOWN, lw=2.0 if l_ else 0.8,
                    alpha=1.0 if l_ else 0.5)
    ax2.set_yticks([])
    ax2.set_ylabel('Members', rotation=0, ha='right', va='center', fontsize=9)
    ax3.fill_between(np.arange(G), ordered, 0, color='#888888', step='mid')
    ax3.axhline(0, color=fg, lw=0.6)
    ax3.set_ylabel('Rank statistic', fontsize=9)
    ax3.set_xlabel(dp.wrap_to_width('Features ranked (DOWN members negated); '
                                    'red UP, blue DOWN members', 9, 5.8), fontsize=9)
    ax3.set_xlim(0, G)
    for ax in (ax1, ax2, ax3):
        ax.tick_params(labelsize=8)
    for ax in (ax1, ax2):
        ax.tick_params(labelbottom=False)
    dp.style_figure_theme(fig, is_dark)
    dp.stamp_run_label(fig, run_label)
    return fig


def make_members_figure(members: pd.DataFrame, state: str, comparison: str,
                        is_dark: bool = False, run_label: str = ''):
    """Horizontal bars of each member's ranking statistic times its sign."""
    from matplotlib.figure import Figure
    from honeychrome.view_components import differential_plots as dp

    if members is None or members.empty:
        return dp.message_figure("No members", state, is_dark, run_label)
    m = members.assign(signed=members['r'] * members['sign']).sort_values('signed')
    tick_labels = [f"{lab} ({'UP' if s > 0 else 'DOWN'})" for lab, s in zip(m['label'], m['sign'])]
    # The figure widens with the member names so the bars keep at least
    # four inches; the axis label and title wrap to the space they have.
    label_w = max(dp.text_width_inches(t, 8) for t in tick_labels)
    fig_w = max(7.0, label_w + 4.6)
    bars_w = fig_w - label_w - 0.6
    fig = Figure(figsize=(fig_w, max(2.5, 0.28 * len(m) + 1.2)), layout='constrained')
    ax = fig.add_subplot(111)
    colours = [(_UP if s > 0 else _DOWN) if lead else '#BBBBBB'
               for s, lead in zip(m['sign'], m['leading_edge'])]
    ax.barh(range(len(m)), m['signed'], color=colours)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.axvline(0, color='white' if is_dark else 'black', lw=0.6)
    ax.set_xlabel(dp.wrap_to_width('sign(logFC)·−log10 p × library direction '
                                   '(coloured: leading edge)', 9, bars_w), fontsize=9)
    # A figure-level title, centred over the member names and bars together.
    fig.suptitle(dp.wrap_to_width(f"{state} — {comparison}: members", 10, fig_w - 0.4),
                 fontsize=10)
    dp.style_figure_theme(fig, is_dark)
    dp.stamp_run_label(fig, run_label)
    return fig


# ---------------------------------------------------------------------------
# Coherence and the state network (``ag_coherence``)
# ---------------------------------------------------------------------------

_LINEAGE_ORDER = ('T', 'NK', 'ILC', 'B', 'myeloid')
# Dark2 without its yellow, checked for colour-blind separation; every node
# also carries a text label.
_LINEAGE_COLOURS = {'T': '#1b9e77', 'NK': '#a6761d', 'ILC': '#7570b3', 'B': '#e7298a',
                    'myeloid': '#66a61e'}
_OTHER_LINEAGE = '#737373'
_PAIR_COLOURS = ('#4d9221', '#c51b7d')


def _lineage_colour(lineage: str) -> str:
    return _LINEAGE_COLOURS.get(lineage, _OTHER_LINEAGE)


def _lineage_key(lineage: str) -> int:
    return _LINEAGE_ORDER.index(lineage) if lineage in _LINEAGE_ORDER else len(_LINEAGE_ORDER)


def _comparison_groups(rows: pd.DataFrame) -> tuple[str, str]:
    """(base, other) of a comparison's coherence rows."""
    return str(rows['group_b'].iloc[0]), str(rows['group_a'].iloc[0])


def _two_colours(groups: tuple[str, str], group_colours: dict | None) -> list[str]:
    cols = dict(group_colours or {})
    return [cols.get(g) or _PAIR_COLOURS[i] for i, g in enumerate(groups)]


def make_coherence_figure(block: dict, comparison: str, fdr_threshold: float,
                          is_dark: bool = False, group_colours: dict | None = None,
                          run_label: str = ''):
    """Coherence of each state in the two groups of *comparison*, with 95%
    intervals; states whose coherence differs (FDR below the threshold)
    are starred and in bold."""
    from matplotlib.figure import Figure
    from honeychrome.view_components import differential_plots as dp

    rows = block['coherence']
    rows = rows[(rows['comparison'] == comparison)
                & np.isfinite(rows['C_a'].astype(float)) & np.isfinite(rows['C_b'].astype(float))]
    if rows.empty:
        return dp.message_figure("No state has enough members for coherence", comparison,
                                 is_dark, run_label)
    rows = rows.assign(_lk=rows['lineage'].map(_lineage_key)).sort_values(
        ['_lk', 'immune_state'], ascending=[False, False])
    base, other = _comparison_groups(rows)
    cols = _two_colours((base, other), group_colours)
    labels = [f"{s}  *" if q < fdr_threshold else s
              for s, q in zip(rows['immune_state'], rows['FDR'].astype(float))]
    label_w = max(dp.text_width_inches(t, 8.5) for t in labels)
    fig_w = max(6.5, label_w + 4.5)
    fig = Figure(figsize=(fig_w, max(3.0, 0.32 * len(rows) + 1.6)), layout='constrained')
    ax = fig.add_subplot(111)
    fg = 'white' if is_dark else 'black'
    xs, ys, hover = [], [], []
    for y, (_i, r) in enumerate(rows.iterrows()):
        ax.plot([r['C_b'], r['C_a']], [y, y], color='#bbbbbb', lw=2, zorder=1)
        for (g, c, lo, hi, dy, col) in ((base, r['C_b'], r['C_b_lo'], r['C_b_hi'], 0.13, cols[0]),
                                        (other, r['C_a'], r['C_a_lo'], r['C_a_hi'], -0.13, cols[1])):
            ax.plot([lo, hi], [y + dy, y + dy], color=col, lw=1.2, alpha=0.7, zorder=2)
            xs.append(c)
            ys.append(y + dy)
            hover.append(f"{r['immune_state']} — {g}\nC {c:.2f} (95% {lo:.2f} to {hi:.2f})\n"
                         f"ΔC {r['dC']:+.2f}, p {r['p']:.3g}, FDR {r['FDR']:.3g}\n"
                         f"{int(r['members_used'])} of {int(r['members_total'])} members")
    sc = ax.scatter(xs, ys, s=34, c=[cols[0], cols[1]] * len(rows), edgecolors='white',
                    linewidths=0.8, zorder=3)
    ax.axvline(0, color=fg, lw=0.6, alpha=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8.5)
    for tick, q in zip(ax.get_yticklabels(), rows['FDR'].astype(float)):
        if q < fdr_threshold:
            tick.set_fontweight('bold')
    for tick, lin in zip(ax.get_yticklabels(), rows['lineage']):
        tick.set_color(_lineage_colour(lin))
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(True, axis='x', alpha=0.25)
    ax.set_xlabel(dp.wrap_to_width("Coherence: signed mean correlation of the state's members "
                                   "across donors (lines: 95% intervals)", 9, fig_w - label_w - 0.6),
                  fontsize=9)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker='o', ls='', color=cols[0], label=base),
                       Line2D([], [], marker='o', ls='', color=cols[1], label=other)],
              fontsize=8, frameon=False, loc='lower right')
    fig.suptitle(dp.wrap_to_width(f"Coherence — {comparison} (*: differs, FDR < "
                                  f"{fdr_threshold:g}; names coloured by lineage)", 10, fig_w - 0.4),
                 fontsize=10)
    dp.style_figure_theme(fig, is_dark)
    for tick, lin in zip(ax.get_yticklabels(), rows['lineage']):
        tick.set_color(_lineage_colour(lin))
    fig._hover_handler = dp.make_scatter_hover_handler(fig, ax, sc, hover, is_dark)
    dp.stamp_run_label(fig, run_label)
    return fig


def _circle_layout(states: list[str], lineages: dict) -> dict:
    """Angle of each state on a circle, grouped by lineage with a gap
    between lineages."""
    order = sorted(states, key=lambda s: (_lineage_key(lineages.get(s, '')), s))
    slots, prev = [], None
    for s in order:
        if prev is not None and lineages.get(s, '') != lineages.get(prev, ''):
            slots.append(None)
        slots.append(s)
        prev = s
    slots.append(None)
    n = len(slots)
    return {s: np.pi / 2 - np.pi / n - 2 * np.pi * k / n for k, s in enumerate(slots) if s}


def make_state_network_figure(block: dict, comparison: str, fdr_threshold: float,
                              is_dark: bool = False, run_label: str = ''):
    """The state network of each group of *comparison*, side by side.

    Node: an immune state, coloured by lineage, sized by its coherence in
    that group. Faint line: the two states' scores correlate across the
    group's donors (FDR below the threshold). Solid line: the correlation
    differs between the groups (relabelling FDR below the threshold;
    exploratory). Red positive, blue negative.
    """
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from honeychrome.view_components import differential_plots as dp

    coh = block['coherence']
    coh = coh[coh['comparison'] == comparison]
    edges = block['edges']
    edges = edges[edges['comparison'] == comparison]
    if coh.empty:
        return dp.message_figure("No coherence result", comparison, is_dark, run_label)
    base, other = _comparison_groups(coh)
    lineages = dict(zip(coh['immune_state'], coh['lineage']))
    ang = _circle_layout(list(coh['immune_state']), lineages)
    xy = {s: np.array([np.cos(a), np.sin(a)]) for s, a in ang.items()}
    label_w = max(dp.text_width_inches(s, 7.5) for s in ang)
    panel = 4.2 + 2 * label_w * 0.8
    fig = Figure(figsize=(2 * panel, panel + 1.0), layout='constrained')
    axes = fig.subplots(1, 2)
    lim = 1.15 + 1.25 * label_w / (panel / 2.6)
    for ax, (g, rcol, qcol, ccol) in zip(axes, ((base, 'r_b', 'FDR_b', 'C_b'),
                                                (other, 'r_a', 'FDR_a', 'C_a'))):
        n_within = 0
        for _i, e in edges.iterrows():
            r = float(e[rcol])
            within = float(e[qcol]) < fdr_threshold
            diff = float(e['FDR']) < fdr_threshold
            if not (within or diff) or not np.isfinite(r):
                continue
            n_within += within
            p1, p2 = xy[e['state_1']], xy[e['state_2']]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=_UP if r > 0 else _DOWN,
                    lw=0.6 + 4.0 * abs(r), alpha=0.95 if diff else 0.18,
                    solid_capstyle='round', zorder=3 if diff else 1)
        sizes = {s: max(float(c), 0.0) if np.isfinite(float(c)) else 0.0
                 for s, c in zip(coh['immune_state'], coh[ccol])}
        for s, a in ang.items():
            ax.scatter(*xy[s], s=30 + 700 * sizes[s], color=_lineage_colour(lineages.get(s, '')),
                       edgecolors='white', linewidths=1.2, zorder=4)
            rot = np.degrees(a) + (180 if np.cos(a) < 0 else 0)
            ax.text(*(1.14 * xy[s]), s, rotation=rot, rotation_mode='anchor',
                    ha='left' if np.cos(a) >= 0 else 'right', va='center', fontsize=7.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"{g}: {n_within} correlations within the group", fontsize=9.5, y=-0.04)
    n_diff = int((edges['FDR'].astype(float) < fdr_threshold).sum())
    present = [lin for lin in _LINEAGE_ORDER if lin in set(lineages.values())]
    handles = [Line2D([], [], color=_UP, lw=3, label='positive'),
               Line2D([], [], color=_DOWN, lw=3, label='negative'),
               Line2D([], [], color='#777777', lw=3, alpha=0.95,
                      label=f'differs between the groups ({n_diff}; exploratory)'),
               Line2D([], [], color='#777777', lw=3, alpha=0.18, label='within the group only')]
    handles += [Line2D([], [], marker='o', ls='', color=_lineage_colour(lin), label=lin)
                for lin in present]
    if any(lin not in _LINEAGE_COLOURS for lin in lineages.values()):
        handles.append(Line2D([], [], marker='o', ls='', color=_OTHER_LINEAGE, label='other'))
    fig.legend(handles=handles, loc='outside lower center', ncol=min(len(handles), 5),
               fontsize=8, frameon=False)
    fig.suptitle(dp.wrap_to_width(
        f"State network — {comparison}. Lines: state scores correlated across donors "
        f"(FDR < {fdr_threshold:g}); node size: coherence", 10, 2 * panel - 0.4), fontsize=10)
    dp.style_figure_theme(fig, is_dark)
    dp.stamp_run_label(fig, run_label)
    return fig


def _bundle(nodes: list[np.ndarray], beta: float = 0.85) -> np.ndarray:
    """A smooth curve through control points pulled towards their straight
    line by 1 − beta (hierarchical edge bundling, Holten 2006)."""
    P = np.array(nodes, dtype=float)
    t = np.linspace(0, 1, len(P))[:, None]
    P = beta * P + (1 - beta) * (P[0] + t * (P[-1] - P[0]))
    for _ in range(5):
        Q = [P[0]]
        for i in range(len(P) - 1):
            Q += [0.75 * P[i] + 0.25 * P[i + 1], 0.25 * P[i] + 0.75 * P[i + 1]]
        Q.append(P[-1])
        P = np.array(Q)
    return P


def make_state_pair_figure(view: dict | None, state_1: str, state_2: str, comparison: str,
                           groups: tuple[str, str], is_dark: bool = False,
                           group_colours: dict | None = None, min_abs_rho: float = 0.3,
                           run_label: str = ''):
    """Two states in the two groups of a comparison.

    Top: each donor's two state scores (one dot per donor), with the rank
    correlation. Bottom: the states' member features around a circle in
    gate-tree order, with every pair of features correlated at |ρ| ≥
    *min_abs_rho* (pairs linked by the gate tree left out) drawn along the
    tree. *view* is ``ag_coherence.pair_view``.
    """
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from honeychrome.view_components import differential_plots as dp

    if view is None:
        return dp.message_figure("These states have no members in use", comparison, is_dark,
                                 run_label)
    feats, labels, owner, paths = view['features'], view['labels'], view['owner'], view['paths']
    n = len(feats)
    theta = np.pi / 2 - 2 * np.pi * np.arange(n) / max(n, 1)
    leaf = {f: np.array([np.cos(t), np.sin(t)]) for f, t in zip(feats, theta)}
    chains = {f: [p for p in str(paths[i]).split(' > ') if p] for i, f in enumerate(feats)}
    depth = max((len(c) for c in chains.values()), default=1)
    inner: dict[str, list] = {}
    for f, c in chains.items():
        for node in c:
            inner.setdefault(node, []).append(leaf[f])
    pos = {}
    for node, vecs in inner.items():
        v = np.mean(vecs, axis=0)
        level = max(c.index(node) for c in chains.values() if node in c)
        norm = np.linalg.norm(v)
        pos[node] = v / norm * (0.9 * (level + 1) / (depth + 1)) if norm > 1e-9 else np.zeros(2)
    own_col = {state_1: '#1b9e77', state_2: '#d95f02', 'both': '#555555'}
    cols = _two_colours(groups, group_colours)
    label_w = max(dp.text_width_inches(lab, 6.5) for lab in labels)
    panel = 3.4 + 2 * label_w
    fig = Figure(figsize=(2 * panel, 3.6 + panel), layout='constrained')
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, panel])
    for k, g in enumerate(groups):
        d = view['groups'].get(g)
        ax = fig.add_subplot(gs[0, k])
        if d is not None:
            x, y = d['score_1'], d['score_2']
            ok = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[ok], y[ok], s=14, color=cols[k], alpha=0.8, edgecolors='white',
                       linewidths=0.4)
            if ok.sum() > 2:
                b = np.polyfit(x[ok], y[ok], 1)
                xx = np.linspace(x[ok].min(), x[ok].max(), 10)
                ax.plot(xx, np.polyval(b, xx), color='#333333' if not is_dark else '#dddddd', lw=1.2)
                from scipy import stats as _st
                rho = _st.spearmanr(x[ok], y[ok]).statistic
                ax.set_title(f"{g}: ρ = {rho:+.2f} ({int(ok.sum())} donors)", fontsize=9.5)
        ax.set_xlabel(dp.wrap_to_width(f"{state_1} score", 8.5, panel - 0.6), fontsize=8.5)
        ax.set_ylabel(dp.wrap_to_width(f"{state_2} score", 8.5, 2.6), fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        axc = fig.add_subplot(gs[1, k])
        n_shown = 0
        if d is not None:
            C = d['corr']
            for i in range(n):
                for j in range(i + 1, n):
                    r = C[i, j]
                    if not np.isfinite(r) or abs(r) < min_abs_rho:
                        continue
                    ci, cj = chains[feats[i]], chains[feats[j]]
                    common = next((node for node in ci[::-1] if node in cj), None)
                    # Up from the first feature to the deepest shared population,
                    # then down to the second (paths run from the root down).
                    up = ci[::-1][:ci[::-1].index(common) + 1] if common else ci[::-1]
                    down = cj[cj.index(common) + 1:] if common else cj
                    pts = [leaf[feats[i]]] + [pos[nd] for nd in up] + [pos[nd] for nd in down]
                    pts.append(leaf[feats[j]])
                    P = _bundle(pts)
                    axc.plot(P[:, 0], P[:, 1], color=_UP if r > 0 else _DOWN,
                             lw=0.4 + 3 * r ** 2, alpha=0.25 + 0.6 * abs(r), zorder=1)
                    n_shown += 1
        for f, lab, o, t in zip(feats, labels, owner, theta):
            axc.scatter(*leaf[f], s=26, color=own_col[o], edgecolors='white', linewidths=0.8,
                        zorder=3)
            rot = np.degrees(t) + (180 if np.cos(t) < 0 else 0)
            axc.text(*(1.06 * leaf[f]), lab, rotation=rot, rotation_mode='anchor',
                     ha='left' if np.cos(t) >= 0 else 'right', va='center', fontsize=6.5,
                     color=own_col[o])
        lim = 1.1 + label_w * 0.95
        axc.set_xlim(-lim, lim)
        axc.set_ylim(-lim, lim)
        axc.set_aspect('equal')
        axc.axis('off')
        axc.set_title(f"{g}: {n_shown} feature pairs with |ρ| ≥ {min_abs_rho:g}", fontsize=9,
                      y=-0.03)
    fig.legend(handles=[Line2D([], [], marker='o', ls='', color=own_col[state_1], label=state_1),
                        Line2D([], [], marker='o', ls='', color=own_col[state_2], label=state_2),
                        Line2D([], [], marker='o', ls='', color=own_col['both'], label='in both'),
                        Line2D([], [], color=_UP, lw=2, label='positive'),
                        Line2D([], [], color=_DOWN, lw=2, label='negative')],
               loc='outside lower center', ncol=5, fontsize=8, frameon=False)
    fig.suptitle(dp.wrap_to_width(f"{state_1} and {state_2} — {comparison}. Top: one dot per "
                                  "donor. Bottom: member features in gate-tree order, "
                                  "correlations drawn along the tree", 10, 2 * panel - 0.4),
                 fontsize=10)
    dp.style_figure_theme(fig, is_dark)
    dp.stamp_run_label(fig, run_label)
    return fig
