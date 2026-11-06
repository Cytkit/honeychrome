"""
cell_type_panel.py
------------------
Which cell types of a definition database a panel can tell apart.

In plain terms. A cell-type definition lists markers that count for the
cell type (positive) and markers that count against it (negative). It
describes the population in every way it might be stained, so on a given
panel only the stained markers have a say: on a panel without CD90.2 or
the TCRs a T cell is told by CD3 alone, and CD19 or CD11b, when stained,
add evidence against other lineages. Restricted to the panel's markers,
two definitions are

* indistinguishable when their signs are identical (without the TCRs,
  alpha-beta and gamma-delta T cells both read CD3+);
* nested when the signs of one are a subset of the other's (CD4 T cells
  and CD4 Tregs);
* separable when they have opposite signs on a stained marker (CD4 and CD8
  T cells);
* overlapping otherwise.

Definitions with identical signs on the panel form a group: the groups are
what the panel can tell apart. A group is labelled by its one member's
name, or by its signs followed by all its members' names, and the markers
off the panel on which its members differ say what staining would separate
them.

Method. Each definition is a row of a positive indicator matrix P and a
negative indicator matrix N over every marker. Restricting to the panel is
a column mask; identical rows are grouped by hashing them. For the groups,
the shared signs S = P P' + N N' and the conflicting signs
C = P N' + N P' give the relation of every pair: group a contains group b
when C_ab = 0 and S_ab equals a's number of signs, b having more.

A root (an analysis restricted to CD3+ cells, say) keeps only the
definitions that themselves include the root's signs, and removes those
signs from them: a definition that says nothing about CD3 is not a T-cell
type, and every T-cell type reads CD3+ under that root.

The definition database is ``drc_cell_type_database.csv`` in the bundled
plugins: one row per cell type and species, positive and negative markers
in canonical marker names (``marker_database.csv``), separated by
semicolons.

No Qt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATABASE_PATH = Path(__file__).resolve().parent.parent / 'bundled_plugins' / 'drc_cell_type_database.csv'


@dataclass(frozen=True)
class PanelGroup:
    """
    Definitions with identical signs on the panel.

    names        : member names, in database order.
    positive     : the panel markers that count for the group, in the
                   first member's order.
    negative     : the panel markers that count against it.
    separated_by : markers off the panel (and outside the root) on which
                   the members' definitions differ; empty for one member.
    """
    names: tuple
    positive: tuple
    negative: tuple
    separated_by: tuple = ()

    @property
    def signs(self) -> str:
        """``CD4+ CD44+ CD8-``."""
        return ' '.join([f'{m}+' for m in self.positive] + [f'{m}-' for m in self.negative])

    @property
    def label(self) -> str:
        """The name of a single member, else the signs and every member:
        ``CD4+ CD44+ CD8-: Tfh / Th1 / Th2``."""
        if len(self.names) == 1:
            return self.names[0]
        return f'{self.signs}: {" / ".join(self.names)}'


@dataclass
class PanelGroups:
    """
    groups       : the groups, ordered by their first member's position.
    contains     : (groups, groups) bool, [a, b] when group a contains b.
    no_marker    : names with no sign on the panel beyond the root's (the
                   root itself, or nothing stained).
    outside_root : names whose definitions lack one of the root's signs.
    """
    groups: list
    contains: np.ndarray
    no_marker: tuple = ()
    outside_root: tuple = ()
    _index: dict = field(default_factory=dict, repr=False)

    def group_of(self, name: str) -> int | None:
        """Index of the group holding the definition *name*, or None."""
        if not self._index:
            self._index.update({n: g for g, grp in enumerate(self.groups) for n in grp.names})
        return self._index.get(name)

    def parent(self, g: int) -> int | None:
        """The nearest group containing group *g* (the one with the most
        signs), or None."""
        above = np.flatnonzero(self.contains[:, g])
        if above.size == 0:
            return None
        sizes = np.array([len(self.groups[a].positive) + len(self.groups[a].negative) for a in above])
        return int(above[np.argmax(sizes)])


def panel_groups(definitions, panel_markers, root: dict | None = None) -> PanelGroups:
    """
    Group *definitions* by their signs on *panel_markers*.

    definitions   : iterable of (name, positive markers, negative markers),
                    canonical marker names. A marker repeated on one side
                    counts once.
    panel_markers : the canonical markers stained on the panel.
    root          : {marker: +1 or -1} the analysed cells all have, or None
                    for all cells.
    """
    defs = [(str(name), tuple(dict.fromkeys(pos)), tuple(dict.fromkeys(neg))) for name, pos, neg in definitions]
    root = {str(m): int(s) for m, s in (root or {}).items()}
    panel = {str(m) for m in panel_markers}
    markers = sorted({m for _n, p, q in defs for m in (*p, *q)} | panel | set(root))
    col = {m: j for j, m in enumerate(markers)}
    P = np.zeros((len(defs), len(markers)))
    N = np.zeros((len(defs), len(markers)))
    for i, (_name, pos, neg) in enumerate(defs):
        P[i, [col[m] for m in pos]] = 1.0
        N[i, [col[m] for m in neg]] = 1.0
    root_pos = np.array([root.get(m) == 1 for m in markers], dtype=float)
    root_neg = np.array([root.get(m) == -1 for m in markers], dtype=float)
    on = np.array([m in panel for m in markers], dtype=float) * (1.0 - root_pos - root_neg)
    under_root = (P @ root_pos + N @ root_neg) == root_pos.sum() + root_neg.sum()
    Pp, Np = P * on, N * on
    has_sign = (Pp + Np).sum(axis=1) > 0

    members: dict = {}
    for i in np.flatnonzero(under_root & has_sign):
        members.setdefault((Pp[i].tobytes(), Np[i].tobytes()), []).append(int(i))
    keys = list(members)
    groups = []
    off_panel = [j for j, m in enumerate(markers) if not on[j] and m not in root]
    for key in keys:
        rows = members[key]
        first = defs[rows[0]]
        positive = tuple(m for m in first[1] if m in panel and m not in root)
        negative = tuple(m for m in first[2] if m in panel and m not in root)
        signed = P[rows][:, off_panel] - N[rows][:, off_panel]
        differ = (signed != signed[0]).any(axis=0) if len(rows) > 1 else np.zeros(len(off_panel), dtype=bool)
        groups.append(PanelGroup(names=tuple(defs[i][0] for i in rows), positive=positive, negative=negative,
                                 separated_by=tuple(markers[off_panel[j]] for j in np.flatnonzero(differ))))
    if keys:
        GP = np.array([np.frombuffer(k[0]) for k in keys])
        GN = np.array([np.frombuffer(k[1]) for k in keys])
        size = (GP + GN).sum(axis=1)
        shared = GP @ GP.T + GN @ GN.T
        conflict = GP @ GN.T + GN @ GP.T
        contains = (conflict == 0) & (shared == size[:, None]) & (size[:, None] < size[None, :])
    else:
        contains = np.zeros((0, 0), dtype=bool)
    return PanelGroups(groups=groups, contains=contains,
                       no_marker=tuple(defs[i][0] for i in np.flatnonzero(under_root & ~has_sign)),
                       outside_root=tuple(defs[i][0] for i in np.flatnonzero(~under_root)))
