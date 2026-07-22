"""
drc_run_archive.py — Per-run cache/archive for the DR/Clustering plugin
========================================================================
Companion module to ``dr_clustering_tab.py`` (filename intentionally does
NOT end in ``_tab.py``, so it is not picked up as a separate plugin tab).

Implements the §0.2 cache/run-archive redesign from
DR_CLUSTERING_REVAMP_PLAN.md: replaces the single monolithic
``dr_clustering_state.pkl`` blob (which overwrote DR results on every rerun
and gave clustering only a partial, un-queryable history) with:

  cache/dr_clustering/
      current_state.pkl       — "live" in-progress state (current
                                 trained_reducers, embeddings, cluster_labels,
                                 stats results, …) — the same kind of thing
                                 the old sidecar held, just relocated under
                                 the cache/ convention already used by
                                 Controller.cleaned_npz_path.
      manifest.json            — lightweight, human-readable list of every
                                 archived DR / clustering run's metadata.
      runs/<run_id>.pkl         — ONE file per archived run, holding only
                                 that run's heavy payload (fitted reducer +
                                 embeddings for a DR run; per-sample label
                                 arrays for a clustering run).

state.dr_runs / state.clustering_runs hold the manifest fields PLUS the
heavy payload merged in — the same shape existing consumers
(GroupsStatsTab's run combo) already expect from clustering_runs entries.
Runs are loaded eagerly at experiment-open (load_all_runs()) so those
existing consumers keep working unchanged; Item 6's management table can
switch to lazy loading (only the manifest fields, hydrating a run's payload
on first selection) once it exists, since a table only needs metadata to
populate its rows.
"""

from __future__ import annotations

import json
import pickle
import uuid
from datetime import datetime
from pathlib import Path

from drc_logging import get_logger, log_stage

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def cache_root(controller) -> Path:
    """``experiment_dir / 'cache' / 'dr_clustering'`` — created if absent."""
    root = Path(controller.experiment_dir) / 'cache' / 'dr_clustering'
    root.mkdir(parents=True, exist_ok=True)
    return root


def runs_dir(controller) -> Path:
    d = cache_root(controller) / 'runs'
    d.mkdir(parents=True, exist_ok=True)
    return d


def manifest_path(controller) -> Path:
    return cache_root(controller) / 'manifest.json'


def current_state_path(controller) -> Path:
    """
    Path for the 'live' in-progress state pickle — replaces the old loose
    ``dr_clustering_state.pkl`` that used to sit directly in experiment_dir.
    """
    return cache_root(controller) / 'current_state.pkl'


def legacy_current_state_path(controller) -> Path:
    """Pre-migration location, alongside the .kit file (loose at experiment root)."""
    return Path(controller.experiment_dir) / 'dr_clustering_state.pkl'


def run_pickle_path(controller, run_id: str) -> Path:
    return runs_dir(controller) / f"{run_id}.pkl"


# ---------------------------------------------------------------------------
# Manifest (JSON — lightweight, diffable)
# ---------------------------------------------------------------------------

def read_manifest(controller) -> list[dict]:
    path = manifest_path(controller)
    if not path.exists():
        return []
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("manifest at %s could not be read (%s) — starting fresh", path, exc)
    return []


def write_manifest(controller, entries: list[dict]) -> None:
    path = manifest_path(controller)
    with open(path, 'w') as f:
        json.dump(entries, f, indent=2, default=str)


def _new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def make_run_label(kind: str, algorithm: str, gates: list[str],
                    timestamp: datetime | None = None) -> str:
    """
    Auto-generate a run label: 'UMAP_Live_20260719-1210' for one gate named
    'Live', or 'UMAP_3gates_20260719-1210' for several.  ``kind`` is only
    used as a fallback prefix when ``algorithm`` is empty.
    """
    ts = (timestamp or datetime.now()).strftime('%Y%m%d-%H%M')
    algo_part = algorithm or kind
    if len(gates) == 1:
        gate_part = gates[0]
    elif gates:
        gate_part = f"{len(gates)}gates"
    else:
        gate_part = "nogate"
    return f"{algo_part}_{gate_part}_{ts}"


def _manifest_entry(run_id, kind, label, algorithm, gates, training_sample_ids,
                     n_events, channels, params, timestamp, n_clusters=None) -> dict:
    return {
        'run_id': run_id,
        'kind': kind,
        'label': label,
        'algorithm': algorithm,
        'gates': list(gates),
        'training_sample_ids': list(training_sample_ids),
        'n_samples': len(training_sample_ids),
        'n_events': int(n_events),
        'channels': list(channels),
        'params': dict(params),
        'timestamp': timestamp,
        'n_clusters': n_clusters,
    }


# ---------------------------------------------------------------------------
# Per-run pickle payload
# ---------------------------------------------------------------------------

def save_run_payload(controller, run_id: str, payload: dict) -> None:
    """Pickle *payload* (the heavy, non-JSON part of one run) to its own file."""
    path = run_pickle_path(controller, run_id)
    with open(path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_run_payload(controller, run_id: str) -> dict | None:
    """Unpickle a run's heavy payload.  Returns None if missing/corrupt."""
    path = run_pickle_path(controller, run_id)
    if not path.exists():
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except (pickle.PickleError, EOFError, OSError, AttributeError) as exc:
        log.warning("could not load run payload %s (%s)", run_id, exc)
        return None


def delete_run_payload(controller, run_id: str) -> None:
    path = run_pickle_path(controller, run_id)
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# Archiving
# ---------------------------------------------------------------------------

def archive_dr_run(controller, state, *, algorithm, reducer, embeddings,
                    gates, training_sample_ids, channels, params,
                    n_events, label=None) -> dict:
    """
    Archive a completed DR training run: pickle the reducer + embeddings to
    ``cache/dr_clustering/runs/<run_id>.pkl``, append a lightweight entry to
    ``manifest.json``, and append the full (manifest fields + heavy payload)
    entry to ``state.dr_runs``.  Returns the in-memory entry.
    """
    log_stage(log, "ARCHIVE DR RUN")
    run_id = _new_run_id()
    timestamp = datetime.now().isoformat(timespec='seconds')
    run_label = label or make_run_label('dr', algorithm, gates)

    save_run_payload(controller, run_id, {
        'reducer': reducer,
        'embeddings': embeddings,
    })

    entry = _manifest_entry(
        run_id, 'dr', run_label, algorithm, gates, training_sample_ids,
        n_events, channels, params, timestamp,
    )
    manifest = read_manifest(controller)
    manifest.append(entry)
    write_manifest(controller, manifest)

    full_entry = dict(entry)
    full_entry['reducer'] = reducer
    full_entry['embeddings'] = embeddings
    state.dr_runs.append(full_entry)
    log.info("archived DR run %r (run_id=%s, %d embedded sample(s))",
             run_label, run_id, len(embeddings))
    return full_entry


def update_dr_run_embeddings(controller, state, run_id: str, embeddings: dict) -> None:
    """
    Refresh the pickled payload and in-memory entry for an already-archived
    DR run after 'Apply to All Samples' embeds additional samples under the
    same trained model.  Does NOT create a new manifest entry or run_id —
    it is still the same run, just covering more samples.
    """
    payload = load_run_payload(controller, run_id) or {}
    payload['embeddings'] = embeddings
    save_run_payload(controller, run_id, payload)
    for entry in state.dr_runs:
        if entry.get('run_id') == run_id:
            entry['embeddings'] = embeddings
            break


def archive_clustering_run(controller, state, *, algorithm, cluster_labels,
                            colors, names, n_clusters, gates,
                            training_sample_ids, channels, params,
                            n_events, label=None) -> dict:
    """
    Archive a completed clustering run.  Same file/manifest layout as
    archive_dr_run(), with 'labels' (per-sample label arrays), 'colors' and
    'names' as the heavy payload and n_clusters recorded in the manifest.
    """
    log_stage(log, "ARCHIVE CLUSTERING RUN")
    run_id = _new_run_id()
    timestamp = datetime.now().isoformat(timespec='seconds')
    run_label = label or make_run_label('clustering', algorithm, gates)

    save_run_payload(controller, run_id, {
        'labels': cluster_labels,
        'colors': colors,
        'names': names,
    })

    entry = _manifest_entry(
        run_id, 'clustering', run_label, algorithm, gates, training_sample_ids,
        n_events, channels, params, timestamp, n_clusters=n_clusters,
    )
    manifest = read_manifest(controller)
    manifest.append(entry)
    write_manifest(controller, manifest)

    full_entry = dict(entry)
    full_entry['labels'] = cluster_labels
    full_entry['colors'] = colors
    full_entry['names'] = names
    state.clustering_runs.append(full_entry)
    log.info("archived clustering run %r (run_id=%s, %s cluster(s))",
             run_label, run_id, n_clusters)
    return full_entry


def rename_run(controller, run_id: str, new_label: str) -> None:
    """
    Update a run's label in manifest.json.  The in-memory entry (the same
    dict object held in state.dr_runs / state.clustering_runs) is the
    caller's responsibility to update in place — this only persists the
    change to disk, mirroring how archive_* only ever writes what it's
    given rather than reaching into state itself.
    """
    manifest = read_manifest(controller)
    for entry in manifest:
        if entry.get('run_id') == run_id:
            entry['label'] = new_label
            break
    write_manifest(controller, manifest)


def delete_run(controller, state, run_id: str) -> None:
    """
    Remove a run everywhere: manifest entry, pickle file, and the matching
    in-memory entry in state.dr_runs / state.clustering_runs.  Self-contained
    so Item 6's management table only has to call this one function.
    """
    manifest = [e for e in read_manifest(controller) if e.get('run_id') != run_id]
    write_manifest(controller, manifest)
    delete_run_payload(controller, run_id)
    state.dr_runs = [e for e in state.dr_runs if e.get('run_id') != run_id]
    state.clustering_runs = [e for e in state.clustering_runs if e.get('run_id') != run_id]
    log.info("deleted run %s", run_id)


def load_manifest_entries(controller) -> tuple[list[dict], list[dict]]:
    """
    Rebuild (dr_entries, clustering_entries) from manifest.json ONLY —
    metadata fields, no heavy payload.  This is what experiment-open calls
    (replaces the old load_all_runs(), which eagerly unpickled every run's
    payload — fine for a couple of runs, doesn't scale once Item 6 lets
    these accumulate across a session).

    Every consumer that just needs to populate a combo/table row (run
    label, kind, algorithm, gates, sample/event/channel counts, timestamp)
    can work from these entries directly.  Anything that needs the actual
    reducer/embeddings/labels/colors/names must call hydrate_run() on the
    specific entry it cares about, the moment it's actually selected.
    """
    manifest = read_manifest(controller)
    dr_entries = [dict(e) for e in manifest if e.get('kind') == 'dr']
    cl_entries = [dict(e) for e in manifest if e.get('kind') == 'clustering']
    log.info("loaded %d DR run(s), %d clustering run(s) from manifest (metadata only)",
             len(dr_entries), len(cl_entries))
    return dr_entries, cl_entries


def run_payload_exists(controller, run_id: str) -> bool:
    """Cheap existence check (no unpickling) — a run whose pickle has gone
    missing from disk (e.g. manually deleted, or a corrupted experiment
    folder) can be flagged as invalid before anything tries to hydrate it."""
    return run_pickle_path(controller, run_id).exists()


def hydrate_run(controller, entry: dict) -> dict:
    """
    Ensure *entry* carries its heavy payload, mutating it in place and
    returning it.  No-op if already hydrated.

    'Already hydrated' is detected by the presence of the payload key
    itself ('embeddings' for a DR run, 'labels' for a clustering run) —
    manifest-only entries from load_manifest_entries() never have these
    keys at all, so their absence is an unambiguous signal, and mutating
    the SAME dict object held in state.dr_runs / state.clustering_runs is
    what makes this double as the in-session cache: once hydrated, a run
    reselected later in the same session doesn't touch disk again.

    Missing/corrupt pickles (see load_run_payload's own handling) resolve
    to empty dicts rather than raising — callers see a hydrated-but-empty
    entry (e.g. embeddings == {}) and should treat that as "nothing to
    plot" rather than crash.
    """
    kind = entry.get('kind')
    if kind == 'dr' and 'embeddings' not in entry:
        payload = load_run_payload(controller, entry['run_id']) or {}
        entry['reducer'] = payload.get('reducer')
        entry['embeddings'] = payload.get('embeddings', {})
    elif kind == 'clustering' and 'labels' not in entry:
        payload = load_run_payload(controller, entry['run_id']) or {}
        entry['labels'] = payload.get('labels', {})
        entry['colors'] = payload.get('colors', {})
        entry['names'] = payload.get('names', {})
    return entry