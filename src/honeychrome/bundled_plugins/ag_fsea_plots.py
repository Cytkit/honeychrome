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

The file name does not end in ``_tab.py``, so the plugin loader does not
treat it as a tab.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ['make_dot_plot', 'make_curve_figure', 'make_members_figure']

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
    fig_h = max(3.0, 0.32 * len(order) + 1.6)
    fig_w = max(4.5, 1.5 * len(comps) + 4.0)
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
    ax.set_xticklabels(comps, rotation=30, ha='right')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlim(-0.6, len(comps) - 0.4)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.grid(True, alpha=0.25)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6)
    cb.set_label('NES (red gained, blue lost)')
    ax.set_title(f"Immune state enrichment (ringed: FDR < {fdr_threshold:g}; "
                 "size: leading-edge share)", fontsize=10)
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
    ax1.set_ylabel('Running ES')
    txt = f"ES {curve['es']:.2f}"
    if row:
        txt += f", NES {row['NES']:.2f}, p {row['p']:.3g}, FDR {row['FDR']:.3g}"
    ax1.set_title(f"{state} — {comparison}\n{txt}", fontsize=10)
    pos = np.asarray(curve['positions'])
    signs = np.asarray(curve.get('signs', np.ones(pos.size)))
    lead = np.asarray(curve['leading'])
    for p_, s_, l_ in zip(pos, signs, lead):
        ax2.axvline(p_ + 0.5, color=_UP if s_ > 0 else _DOWN, lw=2.0 if l_ else 0.8,
                    alpha=1.0 if l_ else 0.5)
    ax2.set_yticks([])
    ax2.set_ylabel('Members', rotation=0, ha='right', va='center')
    ax3.fill_between(np.arange(G), ordered, 0, color='#888888', step='mid')
    ax3.axhline(0, color=fg, lw=0.6)
    ax3.set_ylabel('Rank statistic')
    ax3.set_xlabel('Features ranked (DOWN members negated); red UP, blue DOWN members')
    ax3.set_xlim(0, G)
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
    fig = Figure(figsize=(7, max(2.5, 0.28 * len(m) + 1.2)), layout='constrained')
    ax = fig.add_subplot(111)
    colours = [(_UP if s > 0 else _DOWN) if lead else '#BBBBBB'
               for s, lead in zip(m['sign'], m['leading_edge'])]
    ax.barh(range(len(m)), m['signed'], color=colours)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels([f"{lab} ({'UP' if s > 0 else 'DOWN'})"
                        for lab, s in zip(m['label'], m['sign'])], fontsize=8)
    ax.axvline(0, color='white' if is_dark else 'black', lw=0.6)
    ax.set_xlabel('sign(logFC)·−log10 p × library direction (coloured: leading edge)')
    ax.set_title(f"{state} — {comparison}: members", fontsize=10)
    dp.style_figure_theme(fig, is_dark)
    dp.stamp_run_label(fig, run_label)
    return fig
