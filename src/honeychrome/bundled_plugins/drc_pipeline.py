"""
drc_pipeline.py — Data loading & transform pipeline for the Honeycluster plugin
================================================================================
Companion module to ``dr_clustering_tab.py`` (filename intentionally does NOT
end in ``_tab.py``, so it is not picked up as a separate plugin tab).

This module isolates the *data pipeline* — the part that loads FCS files,
unmixes them, applies gates, and transforms the selected channels into the
feature matrix fed to dimensionality reduction and clustering.  It was split
out of ``PluginWidget`` because this is exactly the code that was producing
wrong results and was the hardest to inspect in the 6 000-line monolith.

Every stage logs the structure of the data passing through it (channels,
event counts, files, transform parameters, value ranges) via ``drc_logging``.

Corrections relative to the original in-line implementation
-----------------------------------------------------------
1.  Transform construction used the FlowKit 0.x signature
    ``LogicleTransform('tr', param_t=...)`` which raises ``TypeError`` on
    FlowKit 1.3.0, so EVERY channel silently fell back to
    ``np.arcsinh(col / (5*W))``.  Here the FlowKit 1.3.0 signature is used and
    there is **no silent fallback** — a transform failure is logged and raised.
2.  The channel→column index map was built from a *filtered* channel list
    (Time/event_id/ribbon removed) while the data array still contained those
    columns, shifting every channel by the number of leading meta-columns.
    Here the index map is built from the FULL unmixed channel list, so column
    indices line up with the actual unmixed event matrix.
3.  Adds an untransformed channel reader so the Workspace "Marker" colour mode
    can display real, full-scale intensities (not embedding-scale values).
4.  load_unmixed_gated() unmixes each sample with its own AF profile
    assignment through the shared ``sample_loader`` module, so background
    loads match what the main window shows for that sample.
5.  Loaded samples are cached as float32 values of the channels in use
    only (GatedEvents in a GatedCache), up to a quarter of the machine's
    memory, the least recently used sample dropped first.

All functions take ``controller`` and ``state`` explicitly so they can be
unit-tested without a live ``PluginWidget``.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict

import numpy as np

from honeychrome.controller_components.functions import (
    sample_from_fcs,
    apply_transfer_matrix,
)
from honeychrome.settings import af_index_channel
from honeychrome.controller_components.autospectral_functions import (
    apply_af_transfer,
)
from honeychrome.controller_components.sample_loader import (
    UnmixSnapshot,
    af_profile_names,
    load_unmixed,
    resolve_af_for_profiles,
    sample_key_for_path,
    snapshot_unmix_state,
    unmix_events,
    unmixing_signature,
)

from drc_logging import (
    get_logger, log_stage, log_array, log_channel_map,
    log_transform_params, log_files,
)

log = get_logger(__name__)

# Meta channels that are never used as Honeycluster features. They are still
# PRESENT as columns in the unmixed event matrix, so they must be excluded by
# *name selection*, never by deleting them from the index map. AF Index is a
# library position, not an intensity, so it is a label rather than a feature.
META_CHANNELS = ("event_id", "Time", "ribbon", af_index_channel)


# ---------------------------------------------------------------------------
# Channel bookkeeping
# ---------------------------------------------------------------------------

def get_unmixed_channels(controller) -> list[str]:
    """
    Return the FULL ordered list of unmixed channel PnN labels.  Column *i* of
    the unmixed event matrix corresponds to entry *i* of this list (see
    ``Controller.initialise_transfer_matrix``: the transfer matrix has one
    output column per ``settings['unmixed']['event_channels_pnn']`` entry,
    including the Time and event_id columns).
    """
    return list(
        controller.experiment.settings.get('unmixed', {}).get('event_channels_pnn', [])
    )


def build_channel_index(controller) -> dict[str, int]:
    """
    Map each unmixed channel name to its column index in the unmixed event
    matrix.  Built from the FULL channel list (meta columns included) so the
    indices stay aligned with the data array.
    """
    return {ch: i for i, ch in enumerate(get_unmixed_channels(controller))}


def resolve_selected_columns(controller, state) -> list[tuple[str, int]]:
    """
    Return ``[(channel_name, column_index), ...]`` for the user-selected
    channels, in selection order, excluding meta channels and any channel that
    cannot be resolved to a column.  Emits a warning for unresolved channels.
    """
    ch_idx = build_channel_index(controller)
    selected = [c for c in state.selected_channels if c not in META_CHANNELS]
    log_channel_map(log, get_unmixed_channels(controller), selected, ch_idx)
    resolved = []
    for ch in selected:
        col = ch_idx.get(ch)
        if col is None:
            log.warning("selected channel %r has no unmixed column — skipped", ch)
            continue
        resolved.append((ch, col))
    return resolved


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _build_flowkit_transform(transform_id: int, params: dict):
    """
    Build the correct FlowKit 1.3.0 transform object for the given params.

    Mirrors exactly what ``honeychrome...transform.Transform.set_*`` constructs
    internally (LogicleTransform/LogTransform/LinearTransform) — but without
    needing axis ``limits``, since we only call ``.apply()``.

    Returns the transform instance, or ``None`` for linear (id 0), which is
    passed through unchanged to preserve historical behaviour for scatter.
    """
    from flowkit import transforms as fk_transforms

    T = float(params.get('T', 262144.0))
    W = float(params.get('W', 0.5))
    M = float(params.get('M', 4.5))
    A = float(params.get('A', 0.0))

    if transform_id == 1:        # logicle / biexponential
        # FlowKit 1.3.0 signature: LogicleTransform(param_t, param_w, param_m, param_a)
        return fk_transforms.LogicleTransform(param_t=T, param_w=W, param_m=M, param_a=A)
    if transform_id == 2:        # log
        return fk_transforms.LogTransform(param_t=T, param_m=M)
    # transform_id == 0 (linear) → pass through (no transform object)
    return None


def transform_channels(controller, state, gated_data: np.ndarray,
                        channels: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Slice the GIVEN channels out of the FULL unmixed gated matrix and apply
    each channel's configured transform (state.channel_transform_params) --
    same per-channel logic transform_selected_channels always used, just
    for an EXPLICIT channel list rather than state.selected_channels. Lets
    a caller transform a channel set that isn't necessarily today's live
    Configuration selection (e.g. Cluster ID's per-run recorded channels,
    which may differ from what's currently selected there).

    Parameters
    ----------
    gated_data : (n_events, n_unmixed_full_channels) untransformed unmixed
                 data, or a GatedEvents (its own named columns)
    channels   : channel names to slice + transform, in the order wanted

    Returns
    -------
    (feature_matrix, used_names) -- used_names may be SHORTER than
    `channels` if some don't resolve to an unmixed column (skip-and-warn,
    same as resolve_selected_columns) -- callers must zip against
    used_names, never assume channels[i] lines up with column i.

    Raises
    ------
    RuntimeError if a transform cannot be applied (no silent arcsinh fallback).
    """
    if isinstance(gated_data, GatedEvents):
        ch_idx = gated_data.index
        gated_data = gated_data.values
    else:
        ch_idx = build_channel_index(controller)
    resolved = []
    for ch in channels:
        col = ch_idx.get(ch)
        if col is None:
            log.warning("transform_channels: channel %r has no unmixed column — skipped", ch)
            continue
        resolved.append((ch, col))

    out_cols = []
    used_names = []
    for ch, col_i in resolved:
        if col_i >= gated_data.shape[1]:
            log.warning(
                "channel %r maps to column %d but data has only %d columns — skipped",
                ch, col_i, gated_data.shape[1],
            )
            continue
        col = gated_data[:, col_i].astype(np.float64, copy=True)

        params = state.channel_transform_params.get(ch, {})
        transform_id = int(params.get('id', 1)) if params else 1
        log_transform_params(log, ch, transform_id, params)

        xform = _build_flowkit_transform(transform_id, params)
        if xform is not None:
            try:
                col = xform.apply(col)
            except Exception as exc:
                # No silent fallback: surface the real failure.
                raise RuntimeError(
                    f"Transform failed for channel {ch!r} "
                    f"(id={transform_id}, params={params}): {exc}"
                ) from exc
        else:
            log.info("  %-22s transform=linear (passed through, untransformed)", ch)

        out_cols.append(col)
        used_names.append(ch)

    if not out_cols:
        return np.empty((len(gated_data), 0), dtype=np.float32), []
    feature = np.column_stack(out_cols).astype(np.float32)
    log_array(log, "transformed_features", feature, used_names)
    return feature, used_names


def transform_selected_channels(controller, state, gated_data: np.ndarray) -> np.ndarray:
    """
    Slice the user-selected channels out of the FULL unmixed gated matrix and
    apply each channel's configured transform. Thin wrapper around
    transform_channels() for state.selected_channels -- unchanged signature
    and behaviour for every existing caller.

    Raises
    ------
    RuntimeError if no selected channel resolves to an unmixed column, or a
    transform cannot be applied (no silent arcsinh fallback).
    """
    log_stage(log, "TRANSFORM SELECTED CHANNELS")
    channels = [c for c in state.selected_channels if c not in META_CHANNELS]
    feature, used_names = transform_channels(controller, state, gated_data, channels)
    if not used_names:
        raise RuntimeError(
            "No selected channels could be resolved to unmixed columns — "
            "check the channel selection in the Configuration tab."
        )
    return feature


def select_untransformed_channels(controller, state, gated_data: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Slice the user-selected channels out of the unmixed gated matrix WITHOUT
    transforming them — used by the Workspace "Marker" colour mode so points
    are coloured on the real (untransformed) data scale.

    Returns ``(values, channel_names)`` where ``values`` is
    ``(n_events, n_selected)`` and ``channel_names`` matches the column order.
    gated_data may be a GatedEvents (its own named columns).
    """
    if isinstance(gated_data, GatedEvents):
        index = gated_data.index
        resolved = [(ch, index[ch]) for ch in state.selected_channels
                    if ch not in META_CHANNELS and ch in index]
        gated_data = gated_data.values
    else:
        resolved = resolve_selected_columns(controller, state)
    cols = []
    names = []
    for ch, col_i in resolved:
        if col_i < gated_data.shape[1]:
            cols.append(gated_data[:, col_i])
            names.append(ch)
    if not cols:
        return np.empty((len(gated_data), 0), dtype=np.float32), []
    return np.column_stack(cols).astype(np.float32), names


# ---------------------------------------------------------------------------
# Per-sample loading
# ---------------------------------------------------------------------------

def apply_unmixing_af_aware(controller, raw_event_data: np.ndarray, af_state=None,
                            sample_key=None, sample_file=None) -> np.ndarray:
    """
    Unmix *raw_event_data* with the AF library assigned to *sample_key*,
    without the side effects of Controller._apply_unmixing() (which would
    overwrite the AF sidecar data of the sample shown in the main window).

    af_state selects where the unmixing state comes from:

    * ``UnmixSnapshot`` (from ``snapshot_unmix_state`` on the main thread) —
      the normal case for background workers. The sample's own AF
      assignment is looked up in the snapshot by *sample_key*.
    * ``(UnmixSnapshot, AFState or None)`` — the experiment's method with
      an explicit AF library, used for unstained samples that borrow a
      stand-in library (see drc_cluster_id.resolve_unstained_af_states).
    * ``(transfer_matrix, af_precomputed, af_spectra)`` — an explicit AF
      library unmixed with per-cell AF, whatever the experiment's method.
    * ``None`` — main-thread callers only. The sample's assignment is
      resolved from the live controller, so *sample_key* is required: the
      AF matrices of the sample shown in the main window are never used
      for another sample.

    Background workers must pass a snapshot: the main thread reassigns the
    controller's transfer and AF matrices whenever a sample is (re)loaded.

    sample_file: the sample's FCS file when *raw_event_data* is all of its
    events; the experiment's noise-model or variants result is then read
    from or written to the on-disk unmixing cache (unmix_method.unmix).
    """
    if isinstance(af_state, UnmixSnapshot):
        return unmix_events(raw_event_data, af_state, af_state.af_for(sample_key),
                            sample_key=sample_key, cache_file=sample_file)['unmixed']
    if isinstance(af_state, tuple) and len(af_state) == 2 and isinstance(af_state[0], UnmixSnapshot):
        snap, af = af_state
        return unmix_events(raw_event_data, snap, af, sample_key=sample_key)['unmixed']

    if af_state is None and sample_key is not None:
        snap = snapshot_unmix_state(controller, [sample_key], include_qc=False)
        return unmix_events(raw_event_data, snap, snap.af_for(sample_key), sample_key=sample_key,
                            cache_file=sample_file)['unmixed']

    if af_state is None:
        raise ValueError('apply_unmixing_af_aware: pass sample_key or af_state; the AF matrices '
                         'of the displayed sample are not used for other samples.')
    transfer_matrix, af_precomputed, af_spectra = af_state

    if af_precomputed is not None and af_spectra is not None:
        raw_settings = controller.experiment.settings['raw']
        pnn_raw = raw_settings.get('whitelisted_pnn') or raw_settings['event_channels_pnn']
        full_pnn_raw = raw_settings['event_channels_pnn']
        fl_ids_remapped = [
            pnn_raw.index(full_pnn_raw[i])
            for i in controller.filtered_raw_fluorescence_channel_ids
        ]
        result = apply_af_transfer(
            raw_event_data,
            transfer_matrix,
            af_precomputed,
            af_spectra,
            controller.experiment.settings,
            filtered_fl_ids_raw=fl_ids_remapped,
            spillover=controller.experiment.process.get('spillover'),
        )
        return result['unmixed']
    return apply_transfer_matrix(transfer_matrix, raw_event_data)


def resolve_af_state_for_profiles(controller, profile_names: list[str]):
    """
    ``(transfer_matrix, af_precomputed, af_spectra)`` for an explicit list of
    AF profile names, or None when none of them can be used. Main thread
    only — see sample_loader.resolve_af_for_profiles.
    """
    af = resolve_af_for_profiles(controller, profile_names)
    if af is None:
        return None
    return (controller.transfer_matrix, af.precomputed, af.spectra)


def snapshot_af_state(controller, sample_keys=None) -> UnmixSnapshot:
    """
    Main-thread snapshot of per-sample unmixing state for a background
    worker. Pass the result as ``af_state`` to the loading functions below;
    each sample is then unmixed with its own AF profile assignment.
    """
    return snapshot_unmix_state(controller, sample_keys, include_qc=False)


def _af_cache_token(controller, af_state, sample_key) -> tuple:
    """Cache-key component describing how a sample is unmixed."""
    if isinstance(af_state, UnmixSnapshot):
        return (id(af_state.transfer_matrix), af_state.af_profile_key(sample_key),
                af_state.method_key(sample_key))
    if af_state is not None:
        return (id(af_state[0]), id(af_state[1]))
    from honeychrome.controller_components import unmix_method
    state = unmix_method.method_state(controller)
    return (id(controller.transfer_matrix), tuple(af_profile_names(controller, sample_key)),
            unmix_method.method_token(state), '' if state is None else state.set_for(sample_key))


def _gated_cache_key(controller, state, abs_path, af_state, sample_key) -> tuple:
    from honeychrome.controller_components.per_sample_gating_functions import SampleGating
    return (
        str(abs_path),
        tuple(sorted(state.selected_gates)),
        _af_cache_token(controller, af_state, sample_key),
        SampleGating.from_controller(controller, copy=False).signature([sample_key]),
    )


class GatedEvents:
    """
    One sample's gated events as cached: untransformed unmixed values
    (float32) of some channels, by name, and the FCS-file row of each event
    (its event key, int32, ascending).  Indexing with rows selects events.
    """
    __slots__ = ('values', 'names', 'rows', 'index')

    def __init__(self, values: np.ndarray, names: list[str], rows: np.ndarray):
        self.values = values
        self.names = list(names)
        self.rows = rows
        self.index = {ch: i for i, ch in enumerate(self.names)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx) -> 'GatedEvents':
        return GatedEvents(self.values[idx], self.names, self.rows[idx])

    @property
    def nbytes(self) -> int:
        return int(self.values.nbytes + self.rows.nbytes)


def default_cache_bytes() -> int:
    """A quarter of the machine's physical memory (4 GB if unknown)."""
    try:
        import psutil
        return int(psutil.virtual_memory().total // 4)
    except Exception:
        return 4 * 1024 ** 3


class GatedCache:
    """
    Thread-safe least-recently-used cache of GatedEvents, keyed like
    _gated_cache_key and capped at *max_bytes* (default_cache_bytes()).
    The newest entry is always kept, even alone above the cap; *evictions*
    counts entries dropped.
    """

    def __init__(self, max_bytes: int | None = None):
        self.max_bytes = default_cache_bytes() if max_bytes is None else int(max_bytes)
        self.evictions = 0
        self._entries: OrderedDict = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
            return entry

    def put(self, key, entry: GatedEvents) -> None:
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._bytes -= old.nbytes
            self._entries[key] = entry
            self._bytes += entry.nbytes
            while self._bytes > self.max_bytes and len(self._entries) > 1:
                _key, dropped = self._entries.popitem(last=False)
                self._bytes -= dropped.nbytes
                self.evictions += 1

    def __contains__(self, key) -> bool:
        with self._lock:
            return key in self._entries

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def nbytes(self) -> int:
        return self._bytes


def _gated_cache(state) -> GatedCache:
    """state.gated_data_cache, replaced by a GatedCache on states lacking one."""
    cache = getattr(state, 'gated_data_cache', None)
    if not isinstance(cache, GatedCache):
        cache = GatedCache()
        state.gated_data_cache = cache
    return cache


def _cached_channels(controller, state, channels) -> list[str]:
    """Unmixed channels to keep for a sample: the selected ones (meta
    channels excluded) plus *channels*, in unmixed column order."""
    wanted = {c for c in state.selected_channels if c not in META_CHANNELS} \
        if getattr(state, 'selected_channels', None) else set()
    wanted |= set(channels or [])
    return [ch for ch in get_unmixed_channels(controller) if ch in wanted]


def load_unmixed_gated(controller, state, abs_path, af_state=None,
                       channels: list[str] | None = None) -> GatedEvents:
    """
    Load one FCS file, unmix it with the transfer matrix, and apply the
    selected gate(s).  Returns the gated events as a GatedEvents holding the
    UNTRANSFORMED unmixed values (float32) of the selected channels plus
    *channels*, and the file row of every gated event (its event key, see
    ``load_gated_rows``).

    The gates are this sample's own: its per-sample gate overrides and its
    own range on per-sample (Time) axes, not those of the sample shown in
    the main window (see ``per_sample_gating_functions.SampleGating``).

    Cached in state.gated_data_cache (GatedCache), keyed by (file, gates,
    unmixing matrix identity, the sample's gating).  A cached sample lacking
    one of the channels asked for is loaded again and cached with the
    channels it had plus the new ones.

    af_state: optional unmixing snapshot — see apply_unmixing_af_aware()
        docstring. Pass this from any background worker thread; leave as
        None only for main-thread callers. Each sample is unmixed with its
        own AF profile assignment.
    """
    from honeychrome.controller_components.per_sample_gating_functions import SampleGating

    sample_key = sample_key_for_path(controller.experiment_dir, abs_path)

    cache = _gated_cache(state)
    cache_key = _gated_cache_key(controller, state, abs_path, af_state, sample_key)
    wanted = _cached_channels(controller, state, channels)
    cached = cache.get(cache_key)
    if cached is not None and all(ch in cached.index for ch in wanted):
        log.debug("load_unmixed_gated: cache hit for %s",
                   getattr(abs_path, 'name', abs_path))
        return cached
    if cached is not None:
        wanted = _cached_channels(controller, state, list(cached.names) + wanted)

    _raw_settings = controller.experiment.settings['raw']
    whitelisted_pnn = _raw_settings.get('whitelisted_pnn') or None

    sample = sample_from_fcs(abs_path)
    if whitelisted_pnn is not None:
        try:
            raw = sample.get_events(source='raw', col_order=whitelisted_pnn)
        except (KeyError, ValueError) as e:
            log.warning(
                "load_unmixed_gated: col_order get_events failed (%s) — reading all channels", e
            )
            raw = sample.get_events(source='raw')
    else:
        raw = sample.get_events(source='raw')
    log.debug(
        "loaded %s: %d raw events × %d raw channels",
        getattr(abs_path, 'name', abs_path), raw.shape[0], raw.shape[1],
    )
    unmixed = apply_unmixing_af_aware(controller, raw, af_state=af_state,
                                      sample_key=sample_key, sample_file=abs_path)
    log.debug(
        "  unmixed → %d events × %d channels (gates=%r)",
        unmixed.shape[0], unmixed.shape[1], state.selected_gates,
    )
    pnn = list(controller.data_for_cytometry_plots_unmixed['pnn'])
    sample_gating = SampleGating.from_controller(controller)
    mask = np.asarray(sample_gating.population_mask(
        sample_key, unmixed, pnn, list(state.selected_gates)), dtype=bool)
    rows = np.flatnonzero(mask).astype(np.int32)
    ch_idx = build_channel_index(controller)
    names = [ch for ch in wanted if ch_idx.get(ch, unmixed.shape[1]) < unmixed.shape[1]]
    values = unmixed[np.ix_(rows, [ch_idx[ch] for ch in names])].astype(np.float32)
    log.debug("  gated → %d events retained (%d removed), %d channel(s) cached",
              len(rows), unmixed.shape[0] - len(rows), len(names))
    entry = GatedEvents(values, names, rows)
    cache.put(cache_key, entry)
    return entry


def load_gated_rows(controller, state, abs_path, af_state=None) -> np.ndarray:
    """
    Event keys of the events load_unmixed_gated() returns for this file:
    the row of each gated event in the FCS file (int32, ascending, one per
    gated event, in the same order).  A file's rows never change, so these
    identify the same cells whatever the gating or unmixing later becomes.
    """
    return load_unmixed_gated(controller, state, abs_path, af_state=af_state).rows


def load_sample_rows(controller, state, rel_path, af_state=None) -> np.ndarray | None:
    """load_gated_rows() for a picker-relative sample path; None on error."""
    try:
        return load_gated_rows(controller, state, sample_abs_path(controller, rel_path),
                               af_state=af_state)
    except Exception as exc:
        log.exception("could not load event rows for %s: %s", rel_path, exc)
        return None


def is_gated_cached(controller, state, rel_path, af_state=None) -> bool:
    """True when load_unmixed_gated() would return this sample from the cache."""
    abs_path = sample_abs_path(controller, rel_path)
    sample_key = sample_key_for_path(controller.experiment_dir, abs_path)
    key = _gated_cache_key(controller, state, abs_path, af_state, sample_key)
    return key in _gated_cache(state)


def cached_first(controller, state, rel_paths, af_state=None) -> list[str]:
    """*rel_paths* with the samples already cached first (each group in its
    given order), so a loop over them uses what is cached before loading
    the rest drops it."""
    rel_paths = list(rel_paths)
    cached = [rel for rel in rel_paths if is_gated_cached(controller, state, rel, af_state)]
    return cached + [rel for rel in rel_paths if rel not in set(cached)]


def sample_file_events(controller, rel_path) -> int:
    """Events in a sample's FCS file, from the experiment (0 if unknown)."""
    abs_path = sample_abs_path(controller, rel_path)
    sample_key = sample_key_for_path(controller.experiment_dir, abs_path)
    counts = controller.experiment.samples.get('all_sample_nevents', {}) or {}
    return int(counts.get(sample_key, 0) or 0)


def timed_preload(controller, state, rel_paths, af_state=None,
                  should_stop=None) -> tuple[float, int]:
    """Read, unmix and gate the samples not cached yet, while the cache has
    room for them (once it drops a sample, the rest are left for the run's
    own loads).  Returns (seconds taken, FCS-file events read), the 'io'
    stage of drc_time_estimate."""
    import time
    seconds, events = 0.0, 0
    cache = _gated_cache(state)
    evictions = cache.evictions
    for rel in rel_paths:
        if should_stop is not None and should_stop():
            break
        if cache.evictions != evictions:
            break
        if is_gated_cached(controller, state, rel, af_state=af_state):
            continue
        start = time.perf_counter()
        if load_sample_rows(controller, state, rel, af_state=af_state) is None:
            continue
        seconds += time.perf_counter() - start
        events += sample_file_events(controller, rel)
    return seconds, events


# Events read to estimate a sample's gated count without loading it whole
COUNT_ESTIMATE_EVENTS = 10_000


def _preview(controller, state, rel_path, af_state):
    """(gated count, exact, gated unmixed preview) of one sample, cached on
    state.count_preview_cache under the gated-data cache key."""
    from honeychrome.controller_components.per_sample_gating_functions import SampleGating

    abs_path = sample_abs_path(controller, rel_path)
    sample_key = sample_key_for_path(controller.experiment_dir, abs_path)
    key = _gated_cache_key(controller, state, abs_path, af_state, sample_key)
    cached = _gated_cache(state).get(key)
    if cached is not None:
        return len(cached), True, cached
    previews = getattr(state, 'count_preview_cache', None)
    if previews is None:
        previews = {}
        state.count_preview_cache = previews
    if key in previews:
        return previews[key]
    res = load_unmixed(controller.experiment_dir, sample_key, af_state,
                       max_events=COUNT_ESTIMATE_EVENTS)
    unmixed = res['unmixed']
    pnn = list(controller.data_for_cytometry_plots_unmixed['pnn'])
    mask = np.asarray(SampleGating.from_controller(controller).population_mask(
        sample_key, unmixed, pnn, list(state.selected_gates)), dtype=bool)
    n_file = int(res['n_events_file'])
    if len(unmixed) >= n_file:
        result = (int(mask.sum()), True, unmixed[mask])
    else:
        result = ((int(round(mask.mean() * n_file)) if len(mask) else 0), False, unmixed[mask])
    previews[key] = result
    return result


def estimate_gated_count(controller, state, rel_path, af_state) -> tuple[int, bool]:
    """
    (gated event count, exact) for one sample under the current gates.

    Exact from the gated-data cache when the sample is already loaded, or
    when the file holds no more than COUNT_ESTIMATE_EVENTS events.
    Otherwise the file's event count times the gated fraction of a seeded
    subsample of COUNT_ESTIMATE_EVENTS events, of which only the selected
    rows are read (relative standard error about 3% for a gate holding 10%
    of the events).  af_state must be an UnmixSnapshot.
    """
    count, exact, _gated = _preview(controller, state, rel_path, af_state)
    return count, exact


def preview_features(controller, state, rel_path, af_state) -> np.ndarray:
    """Transformed selected-channel values of the gated events in the
    sample's count preview (all gated events when the sample is loaded)."""
    _count, _exact, gated = _preview(controller, state, rel_path, af_state)
    if len(gated) > COUNT_ESTIMATE_EVENTS:
        rng = np.random.default_rng(0)
        gated = gated[np.sort(rng.choice(len(gated), COUNT_ESTIMATE_EVENTS, replace=False))]
    return transform_selected_channels(controller, state, gated)


def sample_abs_path(controller, rel_path) -> "object":
    """Resolve a picker-relative sample path to an absolute Path."""
    raw_subdir = controller.experiment.settings['raw']['raw_samples_subdirectory']
    return controller.experiment_dir / raw_subdir / rel_path


def load_sample_features(controller, state, rel_path, af_state=None) -> np.ndarray | None:
    """
    Full pipeline for ONE sample → transformed feature matrix for the selected
    channels (all gated events, no downsampling).  Used during 'Apply to All
    Samples' and per-sample cluster assignment.

    af_state: optional AF snapshot — see apply_unmixing_af_aware() docstring.
    """
    log.debug("LOAD SAMPLE FEATURES — %s", rel_path)
    try:
        gated = load_unmixed_gated(controller, state, sample_abs_path(controller, rel_path), af_state=af_state)
        return transform_selected_channels(controller, state, gated)
    except Exception as exc:
        log.exception("could not load features for %s: %s", rel_path, exc)
        return None


def load_sample_marker_values(controller, state, rel_path, af_state=None) -> tuple[np.ndarray, list[str]] | None:
    """
    Full pipeline for ONE sample → UNTRANSFORMED selected-channel values, for
    Workspace Marker colouring.  Returns ``(values, channel_names)`` or None.

    af_state: optional AF snapshot — see apply_unmixing_af_aware() docstring.
    """
    try:
        gated = load_unmixed_gated(controller, state, sample_abs_path(controller, rel_path), af_state=af_state)
        return select_untransformed_channels(controller, state, gated)
    except Exception as exc:
        log.exception("could not load marker values for %s: %s", rel_path, exc)
        return None


def load_sample_transformed_values(controller, state, rel_path, channels: list[str],
                                    af_state=None) -> tuple[np.ndarray, list[str]] | None:
    """
    Full pipeline for ONE sample -> TRANSFORMED values (state's configured
    per-channel transform, same as load_sample_features) for an EXPLICIT
    channel list rather than state.selected_channels -- e.g. Cluster ID's
    per-run recorded channel set, which may differ from today's live
    Configuration selection.

    Returns (values, channel_names) or None on error. channel_names may be
    SHORTER than `channels` -- see transform_channels().

    af_state: optional AF snapshot — see apply_unmixing_af_aware() docstring.
    """
    try:
        gated = load_unmixed_gated(controller, state, sample_abs_path(controller, rel_path),
                                   af_state=af_state, channels=channels)
        return transform_channels(controller, state, gated, channels)
    except Exception as exc:
        log.exception("could not load transformed values for %s: %s", rel_path, exc)
        return None


def _fingerprint_parts(controller, gates, sample_ids) -> dict:
    """The two halves of a run's fingerprint: 'unmixing' (transfer matrix,
    spillover, each sample's AF profile assignment and the spectra of the
    profiles used, and the unmixing method when it is not per-cell AF with
    the base spectra) and 'gating' (gates, those samples' per-sample
    overrides, axis transforms)."""
    from honeychrome.controller_components.per_sample_gating_functions import SampleGating

    keys = sorted(
        sample_key_for_path(controller.experiment_dir, sample_abs_path(controller, rel))
        for rel in sample_ids
    )
    unmixing = unmixing_signature(controller, keys)
    gating = {
        'gates': sorted(str(g) for g in gates),
        'gating': SampleGating.from_controller(controller, copy=False).signature(keys),
    }
    return {'unmixing': unmixing, 'gating': gating}


def _digest(parts) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str).encode('utf-8')
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def sync_fingerprint(controller, gates, sample_ids) -> str | None:
    """
    Digest of everything that decides which events *gates* select in
    *sample_ids* and what their unmixed values are: the transfer matrix and
    spillover, each sample's AF profile assignment and the spectra of the
    profiles it uses, and the gating (template gates, those samples'
    per-sample overrides, axis transforms).

    A DR or clustering run stores the digest taken when it was made. Run
    again later, a different digest means cells may have moved in or out of
    the gates since, or their values changed. Main thread only, because it
    reads the controller's live matrices and gating.

    Returns None when the digest cannot be computed (callers treat that as
    "cannot verify").
    """
    try:
        parts = _fingerprint_parts(controller, gates, sample_ids)
        return _digest({**parts['unmixing'], **parts['gating']})
    except Exception as exc:
        log.warning("sync_fingerprint: could not compute (%s)", exc)
        return None


def split_fingerprints(controller, gates, sample_ids) -> dict:
    """{'unmixing': digest, 'gating': digest} (None values when they cannot
    be computed), so a run can tell whether its cells' values or their gate
    membership changed.  Main thread only."""
    try:
        parts = _fingerprint_parts(controller, gates, sample_ids)
        return {name: _digest(part) for name, part in parts.items()}
    except Exception as exc:
        log.warning("split_fingerprints: could not compute (%s)", exc)
        return {'unmixing': None, 'gating': None}


def _downsample(data: np.ndarray, n: int, rng) -> np.ndarray:
    if len(data) <= n:
        return data
    idx = rng.choice(len(data), size=n, replace=False)
    return data[idx]


def load_training_pool_with_rows(controller, state, seed: int = 42, af_state=None):
    """
    Pool gated, transformed events from every training sample, each
    downsampled to state.n_training_events, into one float32 array, plus
    which events each sample contributed:
    ``(data, selections)`` with selections ``[(rel_path, idx), ...]`` in
    pool order, where ``idx`` indexes the sample's gated events (the rows
    of load_sample_features) and ``data`` holds those events in that order.
    ``(None, [])`` when nothing could be loaded.
    """
    log_stage(log, "LOAD TRAINING POOL")
    if not state.training_sample_ids:
        log.warning("no training samples selected")
        return None, []
    if not state.selected_channels:
        log.warning("no channels selected")
        return None, []
    if not state.selected_gates:
        log.warning("no gate selected")
        return None, []

    log_files(log, "training samples", state.training_sample_ids)
    log.info("per-sample event cap (n_training_events) = %d", state.n_training_events)

    rng = np.random.default_rng(seed)
    chunks = []
    selections = []
    for rel_path in state.training_sample_ids:
        feats = load_sample_features(controller, state, rel_path, af_state=af_state)
        if feats is None or feats.shape[0] == 0:
            log.warning("  %s contributed no events", rel_path)
            continue
        if len(feats) > state.n_training_events:
            idx = rng.choice(len(feats), size=state.n_training_events, replace=False)
        else:
            idx = np.arange(len(feats))
        chunk = feats[idx]
        log.info("  %s → %d events after downsample", rel_path, chunk.shape[0])
        chunks.append(chunk)
        selections.append((rel_path, idx))

    if not chunks:
        log.error("no training data could be loaded")
        return None, []

    data = np.concatenate(chunks, axis=0).astype(np.float32)
    log_array(log, "training_pool",
              data,
              [c for c in state.selected_channels if c not in META_CHANNELS])
    return data, selections
