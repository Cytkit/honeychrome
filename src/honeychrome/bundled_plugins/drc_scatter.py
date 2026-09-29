"""
drc_scatter.py — Shared cluster-scatter rendering + legend edit helpers
========================================================================
Companion module to ``dr_clustering_tab.py`` (filename intentionally does
NOT end in ``_tab.py``, so it is not picked up as a separate plugin tab).

Extracted from PlotCard's cluster-colouring code so the Workspace's PlotCard
and the Cluster Annotation tab render and edit clusters through the same
code path instead of duplicating it — and so a future gating plugin can
import this directly instead of copy-pasting (population map instead of
cluster map, same rendering).

Everything here operates on a clustering-run dict (an entry from
state.clustering_runs, hydrated via drc_run_archive.hydrate_run) rather
than any ambient state — colours/names live per-run, not on a
single shared dict.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QMessageBox

import drc_run_archive
from drc_logging import get_logger

log = get_logger(__name__)


def assign_run_cluster_colors(labels: np.ndarray) -> dict[int, str]:
    """
    Fallback colour assignment for an archived clustering run whose
    'colors' payload is empty — same colorcet glasbey palette convention as
    drc_clustering.assign_cluster_colors, but returns a plain dict rather
    than writing into ambient state, since colours live per-run.
    """
    import colorcet as cc
    palette = cc.glasbey
    colors: dict[int, str] = {}
    color_idx = 0
    for lbl in sorted(int(l) for l in np.unique(labels)):
        if lbl == -1:
            colors[-1] = '#7f7f7f'
        else:
            colors[lbl] = palette[color_idx % len(palette)]
            color_idx += 1
    return colors


def draw_cluster_scatter(ax, xy, labels, cl_run: dict | None, controller) -> None:
    """
    Colour points by cluster label using cl_run's own 'colors'/'names'
    (cluster colours/names live per clustering-run — see
    state.clustering_runs / archive_clustering_run — not a single ambient
    dict). Shared by PlotCard (Workspace) and the Cluster Annotation tab's
    cluster-map panel so both render identically.
    """
    if cl_run is None:
        ax.scatter(xy[:, 0], xy[:, 1], s=1, c='#aaaaaa', alpha=0.4)
        ax.set_title("No clustering run selected", fontsize=8)
        return
    if labels is None or len(labels) == 0:
        log.warning("cluster scatter: no labels for this selection")
        ax.scatter(xy[:, 0], xy[:, 1], s=1, c='#aaaaaa', alpha=0.4)
        return

    # If colours were never archived for this run (shouldn't normally
    # happen post-Item-6, but a defensive fallback for a very old
    # pre-migration run), assign them now rather than silently greying
    # every point, and persist so this doesn't recompute every refresh.
    colors = cl_run.setdefault('colors', {})
    if not colors:
        log.warning("cluster_colors empty for run %s — assigning palette on the fly",
                     cl_run.get('run_id'))
        colors.update(assign_run_cluster_colors(np.asarray(labels)))
        drc_run_archive.update_run_payload(controller, cl_run)
    names = cl_run.get('names', {})

    unique_labels = sorted(int(l) for l in np.unique(labels))
    n_grey = 0
    for lbl in unique_labels:
        mask = labels == lbl
        color = colors.get(lbl)
        if color is None:
            color = '#7f7f7f' if lbl < 0 else '#aaaaaa'
            if lbl >= 0:
                n_grey += int(mask.sum())
        label_text = names.get(lbl, 'Noise' if lbl < 0 else str(lbl))
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            s=1, c=color, alpha=0.5, linewidths=0,
            label=label_text
        )
    log.debug("cluster scatter: %d points, %d clusters, %d uncoloured",
              len(labels), len(unique_labels), n_grey)


def rename_cluster(controller, cl_run: dict | None, label: int, new_name: str,
                   parent=None) -> bool:
    """
    Rename a cluster within cl_run, rejecting a name already used by
    another cluster in the SAME run. Persists immediately via
    drc_run_archive.save_run_payload. Returns True on success, False if
    rejected (duplicate, blank, or cl_run is None).

    Shared by PlotCard's legend rename (Workspace) and the Cluster
    Annotation tab's cluster-label table / map legend so both
    apply the identical duplicate check rather than two copies of it.
    """
    if cl_run is None:
        return False
    new_name = new_name.strip()
    if not new_name:
        return False
    names = cl_run.setdefault('names', {})
    for other_id, other_name in names.items():
        if other_id != label and other_name == new_name:
            QMessageBox.warning(
                parent, "Duplicate Name",
                f"Cluster name '{new_name}' is already used by cluster {other_id}.\n"
                "Please choose a unique name."
            )
            return False
    names[label] = new_name
    drc_run_archive.update_run_payload(controller, cl_run)
    return True


def recolor_cluster(controller, cl_run: dict | None, label: int, new_hex: str) -> None:
    """Set a cluster's colour within cl_run and persist immediately. No-op
    if cl_run is None."""
    if cl_run is None:
        return
    colors = cl_run.setdefault('colors', {})
    colors[label] = new_hex
    drc_run_archive.update_run_payload(controller, cl_run)


def compatibility_warning(dr_run: dict | None, cl_run: dict | None) -> str | None:
    """
    Return a warning string if cl_run's gate-set/sample-set isn't a subset
    of dr_run's (overlaying labels computed on a different/broader
    population isn't guaranteed to line up point-for-point), else None.

    Plain set comparison against manifest fields already on both entries. 
    Shared by PlotCard (Workspace) and the Cluster Annotation tab's cluster-map panel.
    """
    if dr_run is None or cl_run is None:
        return None
    dr_gates = set(dr_run.get('gates', []))
    cl_gates = set(cl_run.get('gates', []))
    dr_samples = set(dr_run.get('training_sample_ids', []))
    cl_samples = set(cl_run.get('training_sample_ids', []))
    if cl_gates.issubset(dr_gates) and cl_samples.issubset(dr_samples):
        return None
    return (
        f"Labels computed on gate(s) {sorted(cl_gates)} / "
        f"{len(cl_samples)} sample(s); this DR run uses gate(s) "
        f"{sorted(dr_gates)} / {len(dr_samples)} sample(s) — "
        "results may not align."
    )

def align_labels_to_embedding(labels: np.ndarray | None,
                              n: int) -> tuple[np.ndarray, bool]:
    """
    Align one sample's cluster-label array to one of its DR embeddings.

    UMAP/tSNE/PaCMAP embeddings and cluster labels both cover every gated
    event of a sample in the same order, so they correspond row for row
    exactly when their lengths match. A DR run and a clustering run can
    share the same gate and sample set and still differ in length (e.g.
    one was made before the gate was edited), which
    compatibility_warning's gate/sample-set check does not catch.

    labels: the sample's cluster-label array from a clustering run, or
        None if this sample has no labels under the selected run.

    Returns (aligned_labels, ok): aligned_labels always has length *n*
    (-1 = unlabelled). ok is False when the sample had to be greyed out;
    callers should surface a warning in that case.
    """
    if labels is None or len(labels) != n:
        return np.full(n, -1, dtype=np.int32), False
    return labels.astype(np.int32), True