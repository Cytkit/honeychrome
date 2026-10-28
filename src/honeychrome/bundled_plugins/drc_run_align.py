"""
drc_run_align.py — Matching archived runs to events by event key
================================================================
Companion module to ``dr_clustering_tab.py`` (not a plugin: the filename
does not end in ``_tab.py``).

Every DR and clustering run archived since event keys were introduced
stores, for each sample, the FCS-file row of each of its rows
(``event_rows``, from drc_pipeline.load_gated_rows).  A file's rows never
change, so a run's labels or embedding can be matched to the events the
current gating selects, or to another run's rows, event by event:

* events in both keep the run's value;
* events the run has but the current gating no longer selects drop out;
* events the current gating selects that the run never saw are new.  A
  clustering run in marker space labels them with its stored SOM codebook
  or cluster centroids (in the transforms the run used); otherwise they are
  left unlabelled (``UNLABELLED``) and reported.

Runs archived before event keys ("legacy" runs) carry no rows.  Their
arrays are only paired with other data row for row, and only where the two
sides are known to come from the same gating and unmixing: a legacy run
against live data only when its fingerprint matches the current one, two
legacy runs only when their fingerprints match (or neither has one).  Equal
row counts alone are not enough: under a later gating a sample can have the
same number of events as before but not the same events.

Pure numpy; no Qt.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

import drc_clustering
import drc_pipeline
import drc_run_archive
from drc_logging import get_logger

log = get_logger(__name__)

# Label of an event in the current gating that a run cannot label
UNLABELLED = -2

JOIN_KEYS = 'keys'
JOIN_POSITIONAL = 'positional'
JOIN_NONE = 'none'


def has_event_keys(entry: dict | None) -> bool:
    """True for a run archived with event keys (manifest flag, no hydration)."""
    return bool(entry) and bool(entry.get('event_keys'))


def run_rows(entry: dict, rel_path: str) -> np.ndarray | None:
    """A hydrated run's event keys for one sample, or None."""
    rows = (entry.get('event_rows') or {}).get(rel_path)
    return None if rows is None else np.asarray(rows, dtype=np.int64)


def join_rows(a_rows: np.ndarray, b_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positions (ia, ib) with a_rows[ia] == b_rows[ib]; both inputs hold
    unique file rows."""
    _common, ia, ib = np.intersect1d(np.asarray(a_rows), np.asarray(b_rows),
                                     assume_unique=True, return_indices=True)
    return ia, ib


def align_values(values: np.ndarray, src_rows: np.ndarray, dst_rows: np.ndarray,
                 fill) -> tuple[np.ndarray, np.ndarray]:
    """values (one per src row) placed on dst_rows; rows without a value get
    *fill*.  Returns (aligned, matched_mask over dst_rows)."""
    values = np.asarray(values)
    out = np.full((len(dst_rows),) + values.shape[1:], fill,
                  dtype=np.result_type(values.dtype, fill))
    matched = np.zeros(len(dst_rows), dtype=bool)
    ia, ib = join_rows(src_rows, dst_rows)
    out[ib] = values[ia]
    matched[ib] = True
    return out, matched


def live_join_mode(entry: dict, current_fingerprint: str | None) -> str:
    """How a run's arrays may be paired with events loaded now."""
    if has_event_keys(entry):
        return JOIN_KEYS
    if drc_run_archive.sync_status(entry, current_fingerprint) == drc_run_archive.SYNC_IN_SYNC:
        return JOIN_POSITIONAL
    return JOIN_NONE


def runs_join_mode(a: dict, b: dict) -> str:
    """How two runs' arrays may be paired with each other."""
    if has_event_keys(a) and has_event_keys(b):
        return JOIN_KEYS
    fa, fb = a.get('sync_fingerprint'), b.get('sync_fingerprint')
    if has_event_keys(a) != has_event_keys(b):
        # One keyed run: same cells only if both were made under one fingerprint
        return JOIN_POSITIONAL if fa and fb and fa == fb else JOIN_NONE
    if fa and fb:
        return JOIN_POSITIONAL if fa == fb else JOIN_NONE
    return JOIN_POSITIONAL


def pair_arrays(values, n_target: int, mode: str, src_rows=None, dst_rows=None, fill=-1):
    """
    values (from one run or array) placed on a target of n_target rows.

    mode JOIN_KEYS uses src_rows/dst_rows; JOIN_POSITIONAL needs equal
    lengths.  Returns (aligned, matched_mask) or (None, None) when the
    arrays cannot be paired.
    """
    if values is None:
        return None, None
    values = np.asarray(values)
    if mode == JOIN_KEYS and src_rows is not None and dst_rows is not None:
        return align_values(values, src_rows, dst_rows, fill)
    if mode == JOIN_POSITIONAL and len(values) == n_target:
        return values, np.ones(n_target, dtype=bool)
    return None, None


class Coverage(NamedTuple):
    """How a run's events of one sample relate to the current gating."""
    kept: int            # events in both
    dropped: int         # in the run, no longer gated
    new: int             # gated now, not in the run
    labelled_new: int    # new events labelled from the run's codebook / centroids

    @property
    def unlabelled(self) -> int:
        return self.new - self.labelled_new


def coverage_total(parts) -> Coverage:
    parts = list(parts)
    return Coverage(*(int(sum(getattr(p, f) for p in parts)) for f in Coverage._fields))


def coverage_text(cov: Coverage) -> str:
    """'99.1% of the run's events still gated; 0.6% new events (0.2% unlabelled)'."""
    run_total = cov.kept + cov.dropped
    now_total = cov.kept + cov.new
    kept = 100.0 * cov.kept / run_total if run_total else 0.0
    new = 100.0 * cov.new / now_total if now_total else 0.0
    text = f"{kept:.1f}% of the run's events still gated; {new:.1f}% new events"
    if cov.new:
        unl = 100.0 * cov.unlabelled / now_total if now_total else 0.0
        text += f" ({unl:.1f}% unlabelled)"
    return text


def _nearest_node(features: np.ndarray, node_weights: np.ndarray,
                  node_to_meta: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    """Label of each event's nearest SOM node or cluster centroid."""
    weights = np.asarray(node_weights, dtype=np.float64)
    meta = np.asarray(node_to_meta)
    w2 = (weights ** 2).sum(axis=1)
    out = np.empty(len(features), dtype=np.int32)
    for start in range(0, len(features), chunk):
        block = np.asarray(features[start:start + chunk], dtype=np.float64)
        d2 = w2[None, :] - 2.0 * block @ weights.T
        out[start:start + chunk] = meta[np.argmin(d2, axis=1)]
    return out


def can_label_new_events(run: dict) -> bool:
    """A marker-space clustering run with a stored codebook or centroids and
    the transforms it was fitted in."""
    params = run.get('params') or {}
    tree = run.get('tree_data')
    return (run.get('kind') == 'clustering'
            and not drc_clustering.clusters_in_dr_space(run.get('algorithm') or '', params)
            and isinstance(tree, dict) and 'node_weights' in tree
            and bool(run.get('transform_params')))


def labels_for_current_events(controller, state, run: dict, rel_path: str,
                              mode: str, af_state=None):
    """
    A hydrated clustering run's labels for the events the current gating
    selects in one sample, in their gated order.

    mode: live_join_mode() of the run.  Returns (labels, coverage) or
    (None, None) when the run cannot be matched to the current events
    (legacy run out of sync, or no labels for this sample).  Labels of new
    events that cannot be derived are UNLABELLED.
    """
    labels = (run.get('labels') or {}).get(rel_path)
    if labels is None:
        return None, None
    labels = np.asarray(labels, dtype=np.int32)
    cur_rows = drc_pipeline.load_sample_rows(controller, state, rel_path, af_state=af_state)
    if cur_rows is None:
        return None, None
    if mode == JOIN_POSITIONAL:
        if len(labels) != len(cur_rows):
            return None, None
        return labels, Coverage(len(labels), 0, 0, 0)
    if mode != JOIN_KEYS:
        return None, None
    src_rows = run_rows(run, rel_path)
    if src_rows is None:
        return None, None
    aligned, matched = align_values(labels, src_rows, cur_rows, UNLABELLED)
    aligned = aligned.astype(np.int32)
    kept = int(matched.sum())
    new_mask = ~matched
    labelled_new = 0
    if new_mask.any() and can_label_new_events(run):
        gated = drc_pipeline.load_unmixed_gated(
            controller, state, drc_pipeline.sample_abs_path(controller, rel_path),
            af_state=af_state, channels=run['channels'])
        features = transform_with_params(controller, gated[new_mask], run['channels'],
                                         run['transform_params'])
        if features is not None:
            tree = run['tree_data']
            aligned[new_mask] = _nearest_node(features, tree['node_weights'],
                                              tree['node_to_meta'])
            labelled_new = int(new_mask.sum())
    return aligned, Coverage(kept, len(src_rows) - kept, int(new_mask.sum()), labelled_new)


def transform_with_params(controller, gated: np.ndarray, channels: list[str],
                          transform_params: dict) -> np.ndarray | None:
    """Gated unmixed events as the features a run was fitted on: its channels,
    in its order, under the transforms recorded with it."""
    import types
    shim = types.SimpleNamespace(channel_transform_params=dict(transform_params))
    try:
        features, names = drc_pipeline.transform_channels(controller, shim, gated, channels)
    except Exception:
        log.exception("could not transform new events with the run's transforms")
        return None
    if list(names) != list(channels):
        log.warning("new events: channels %s of the run are no longer available",
                    sorted(set(channels) - set(names)))
        return None
    return features


def transform_named_values(values: np.ndarray, names: list[str], channel_transform_params: dict,
                           channels: list[str] | None = None):
    """Untransformed values (columns *names*) under the given transforms:
    (transformed values, channel names), keeping *channels* when given."""
    keep = [ch for ch in (channels or names) if ch in names]
    cols = []
    for ch in keep:
        col = np.asarray(values[:, names.index(ch)], dtype=np.float64)
        params = channel_transform_params.get(ch, {})
        xform = drc_pipeline._build_flowkit_transform(int(params.get('id', 1)) if params else 1,
                                                      params)
        cols.append(xform.apply(col) if xform is not None else col)
    if not cols:
        return np.empty((len(values), 0), dtype=np.float32), []
    return np.column_stack(cols).astype(np.float32), keep


def statistics_inputs(controller, state, run: dict, mode: str, af_state=None):
    """
    Labels (and, for a legacy run, values) to test a hydrated clustering
    run that is not in sync with the current gating and unmixing.

    JOIN_KEYS: every sample's labels matched to its currently gated events
    (new events labelled from the run's codebook or centroids where
    possible, else UNLABELLED and left out of the tests); values are
    loaded from the current events as usual.
    Otherwise (legacy run): the run's own labels and its stored marker
    values, under the current transforms, so the tests describe the cells
    the run clustered rather than mismatched current events.

    Returns (labels_dict, values_override or None, note).
    """
    run_labels = run.get('labels') or {}
    if mode == JOIN_KEYS:
        labels, coverages = {}, []
        for rel in run_labels:
            aligned, cov = labels_for_current_events(controller, state, run, rel, mode,
                                                     af_state=af_state)
            if aligned is None:
                log.warning("statistics: %s could not be matched to the current events "
                            "and is left out", rel)
                continue
            labels[rel] = aligned
            coverages.append(cov)
        note = f"Matched to the current gating by event: {coverage_text(coverage_total(coverages))}."
        log.info("statistics on %s: %s", run.get('label'), note)
        return labels, None, note
    values = {}
    for rel, mv in (run.get('marker_values') or {}).items():
        try:
            raw, names = mv
            values[rel] = transform_named_values(np.asarray(raw), list(names),
                                                 state.channel_transform_params)
        except Exception:
            log.exception("statistics: stored values of %s could not be transformed", rel)
    note = ("This run predates event keys and no longer matches the current gating and "
            "unmixing: tests use the cells it clustered, with its stored marker values.")
    return dict(run_labels), values, note
