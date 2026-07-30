"""
drc_pipeline.py — Data loading & transform pipeline for the DR/Clustering plugin
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
4.  load_unmixed_gated() now goes through the same AF-corrected unmixing
    path as the main app's active sample (Controller._apply_unmixing) instead
    of always calling apply_transfer_matrix() directly — previously every
    background-loaded sample (training pools, per-sample features) bypassed
    AF correction entirely, even when an AF profile was active.

All functions take ``controller`` and ``state`` explicitly so they can be
unit-tested without a live ``PluginWidget``.
"""

from __future__ import annotations

import numpy as np

from honeychrome.controller_components.functions import (
    sample_from_fcs,
    apply_transfer_matrix,
    apply_gates_in_place,
)
from honeychrome.controller_components.autospectral_functions import (
    apply_af_transfer,
    combine_af_precomputed,
    precompute_joint_cov_extras,
)

from drc_logging import (
    get_logger, log_stage, log_array, log_channel_map,
    log_transform_params, log_files,
)

log = get_logger(__name__)

# Meta channels that are never used as DR/clustering features. They are still
# PRESENT as columns in the unmixed event matrix, so they must be excluded by
# *name selection*, never by deleting them from the index map.
META_CHANNELS = ("event_id", "Time", "ribbon")


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

def apply_gate_by_lookup_table(cytometry_data_dictionary: dict, gate_name: str) -> np.ndarray:
    """
    Return the subset of ``event_data`` belonging to *gate_name* using the
    Honeychrome lookup-table gating approach.  Operates on the supplied dict
    (caller is responsible for passing a copy).
    """
    event_data = cytometry_data_dictionary['event_data']
    gate_membership = {'root': np.ones(len(event_data), dtype=np.bool_)}
    cytometry_data_dictionary.update({'gate_membership': gate_membership})
    gates_to_calculate = [
        g[0] for g in cytometry_data_dictionary['gating'].get_gate_ids()
    ]
    apply_gates_in_place(cytometry_data_dictionary, gates_to_calculate=gates_to_calculate)
    mask = cytometry_data_dictionary['gate_membership'][gate_name]
    return event_data[mask]


def apply_gates_union_by_lookup_table(cytometry_data_dictionary: dict,
                                      gate_names: list[str]) -> np.ndarray:
    """
    Return the UNION of events belonging to ANY gate in *gate_names*, using
    the same lookup-table approach as apply_gate_by_lookup_table().  Backs
    the multi-select gate tree (§0.3): a run can be built from gates picked
    across different branches, not just one.  ``gates_to_calculate`` already
    computes every gate's mask regardless of which is used, so the union is
    just a logical OR reduce — no extra FlowKit calls.
    """
    event_data = cytometry_data_dictionary['event_data']
    gate_membership = {'root': np.ones(len(event_data), dtype=np.bool_)}
    cytometry_data_dictionary.update({'gate_membership': gate_membership})
    gates_to_calculate = [
        g[0] for g in cytometry_data_dictionary['gating'].get_gate_ids()
    ]
    apply_gates_in_place(cytometry_data_dictionary, gates_to_calculate=gates_to_calculate)
    masks = [cytometry_data_dictionary['gate_membership'][g] for g in gate_names]
    union_mask = np.logical_or.reduce(masks)
    return event_data[union_mask]


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
    gated_data : (n_events, n_unmixed_full_channels) untransformed unmixed data
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
    """
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

def apply_unmixing_af_aware(controller, raw_event_data: np.ndarray, af_state=None) -> np.ndarray:
    """
    Unmix *raw_event_data*, using AF correction if the controller currently
    has AF matrices set for the active sample — mirroring the branching in
    Controller._apply_unmixing() — but WITHOUT that method's side effect of
    overwriting controller.af_sidecar_data.

    Public (no leading underscore): used both by load_unmixed_gated() below
    and by TransformTab._load_all_training_samples() in dr_clustering_tab.py
    (see that file's changes), so every place the DR plugin background-loads
    and unmixes an FCS file goes through the same AF-aware path.

    The DR plugin loads OTHER samples in the background (training pools,
    per-sample features) while the main app may have a different sample
    loaded; calling controller._apply_unmixing() directly would stomp on the
    AF sidecar data for whatever sample the user is actually looking at in
    the main window.  This is a local, side-effect-free copy of the same
    branching logic instead — keeps the plugin's "no side effects on main
    app" design principle intact.

    af_state: optional (transfer_matrix, af_precomputed, af_spectra) snapshot
        captured on the main thread BEFORE a background worker starts.
        Background callers (DR/clustering/stats worker threads) MUST pass
        this — reading controller.transfer_matrix/af_precomputed/af_spectra
        live races against the main thread's controller.load_sample()/
        initialise_af_matrices(), which reassign these same attributes
        whenever the user loads/reloads a sample in the main window while
        the worker is still running. The AF kernel operates on raw pointers
        into these arrays (af_kernel_wrapper.py), so a concurrent
        reassignment/GC of an array the worker is mid-read on is a memory-
        corruption hazard, not just a stale-data one.
    """
    if af_state is not None:
        transfer_matrix, af_precomputed, af_spectra = af_state
    else:
        transfer_matrix = controller.transfer_matrix
        af_precomputed = controller.af_precomputed
        af_spectra = controller.af_spectra

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
    Build an (transfer_matrix, af_precomputed, af_spectra) AF snapshot for
    an EXPLICIT list of AF profile names, mirroring
    Controller.initialise_af_matrices()'s cache-combining logic (dict
    lookups + hstack only, no linalg -- see that method's own docstring)
    for a profile list that isn't necessarily
    controller.current_sample_path's own assignment (e.g. an unstained
    sample, which never has one -- see
    drc_cluster_id.resolve_unstained_af_states, the caller this exists
    for).

    MAIN-THREAD ONLY. Reads controller.af_precomputed_cache and
    controller.experiment.process['af_profiles'], both of which the main
    thread can reassign (editing/deleting an AF profile in the
    AutoSpectral AF tab, or cache_all_af_profiles() after a spectral
    process refresh) -- a caller needing this from a background worker
    MUST call this HERE, on the main thread, before the worker starts, and
    pass the returned tuple through as a snapshot, exactly like any other
    af_state (see apply_unmixing_af_aware's docstring above for why a
    concurrent reassignment while a worker reads these is a
    memory-corruption hazard, not just stale data).

    Returns (transfer_matrix, af_precomputed, af_spectra), or None if
    profile_names is empty or none of them have a cache hit (mirrors
    initialise_af_matrices's own "no cache hit" fallback) -- callers
    should treat None as "AF-unaware" and pass
    (transfer_matrix, None, None) to apply_unmixing_af_aware instead.
    """
    if not profile_names:
        return None
    cached = [
        controller.af_precomputed_cache[name]
        for name in profile_names
        if name in controller.af_precomputed_cache
    ]
    if not cached:
        log.warning(
            "resolve_af_state_for_profiles: no cache hit for profile(s) %s "
            "(cache has: %s)", profile_names, list(controller.af_precomputed_cache),
        )
        return None

    af_profiles = controller.experiment.process.get('af_profiles', {})
    spectra_mats = [
        np.array(af_profiles[name]['spectra'])
        for name in profile_names
        if name in af_profiles and name in controller.af_precomputed_cache
    ]
    if not spectra_mats:
        return None
    af_spectra = np.vstack(spectra_mats)

    if len(cached) == 1:
        combined = cached[0]
    else:
        combined = combine_af_precomputed(cached)
        combined.update(precompute_joint_cov_extras(combined, af_spectra))

    return (controller.transfer_matrix, combined, af_spectra)


def load_unmixed_gated(controller, state, abs_path, af_state=None) -> np.ndarray:
    """
    Load one FCS file, unmix it with the transfer matrix, and apply the
    selected gate(s).  Returns the UNTRANSFORMED unmixed gated matrix with
    all unmixed columns (shape ``(n_gated, n_unmixed_full)``).

    Cached on state.gated_data_cache, keyed by (file, gates, unmixing
    matrix identity).  This is the single most expensive step in the
    Workspace Marker-colour path — without caching it re-reads and
    re-unmixes the FCS file from disk on every PlotCard refresh.

    af_state: optional (transfer_matrix, af_precomputed, af_spectra) snapshot —
        see apply_unmixing_af_aware() docstring. Pass this from any
        background worker thread; leave as None only for main-thread callers.
    """
    from copy import deepcopy

    transfer_matrix = af_state[0] if af_state is not None else controller.transfer_matrix

    cache_key = (
        str(abs_path),
        tuple(sorted(state.selected_gates)),
        id(transfer_matrix),
    )
    cached = state.gated_data_cache.get(cache_key)
    if cached is not None:
        log.debug("load_unmixed_gated: cache hit for %s",
                   getattr(abs_path, 'name', abs_path))
        return cached

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
    unmixed = apply_unmixing_af_aware(controller, raw, af_state=af_state)
    log.debug(
        "  unmixed → %d events × %d channels (gates=%r)",
        unmixed.shape[0], unmixed.shape[1], state.selected_gates,
    )
    cdd = deepcopy(controller.data_for_cytometry_plots_unmixed)
    cdd['event_data'] = unmixed
    gated = apply_gates_union_by_lookup_table(cdd, state.selected_gates)
    log.debug("  gated → %d events retained (%d removed)",
              gated.shape[0], unmixed.shape[0] - gated.shape[0])
    state.gated_data_cache[cache_key] = gated
    return gated


def sample_abs_path(controller, rel_path) -> "object":
    """Resolve a picker-relative sample path to an absolute Path."""
    from pathlib import Path
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
        gated = load_unmixed_gated(controller, state, sample_abs_path(controller, rel_path), af_state=af_state)
        return transform_channels(controller, state, gated, channels)
    except Exception as exc:
        log.exception("could not load transformed values for %s: %s", rel_path, exc)
        return None


def _downsample(data: np.ndarray, n: int, rng) -> np.ndarray:
    if len(data) <= n:
        return data
    idx = rng.choice(len(data), size=n, replace=False)
    return data[idx]


def _downsample_with_indices(data: np.ndarray, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """
    Like _downsample, but also returns the indices into *data* that were
    kept, in the same row order as the returned chunk. Needed wherever a
    downsampled embedding must later be aligned back to a full-length
    per-sample array (e.g. cluster labels, which always cover every
    gated event) by real event identity rather than by coincidental row
    count -- see drc_scatter.align_labels_to_embedding.
    """
    if len(data) <= n:
        return data, np.arange(len(data))
    idx = rng.choice(len(data), size=n, replace=False)
    return data[idx], idx


def load_training_pool_with_sample_bounds(
        controller, state, seed: int = 42, af_state=None
) -> tuple[np.ndarray, list[tuple[str, int, np.ndarray]]] | None:
    """
    Same pooling as load_training_pool, but also returns, per sample in
    concatenation order, (rel_path, n_events, event_indices) --
    event_indices are the row indices into that sample's FULL gated/
    transformed feature array (load_sample_features's own output) that
    were kept, in the same order as the pooled rows.

    Used by PHATE, which has no out-of-sample transform: it must be fit on
    the whole pool in one call, so the caller needs to know which rows of
    the returned array belong to which sample in order to split the single
    resulting embedding back out per-sample afterwards. event_indices then
    lets that per-sample embedding be aligned back to cluster labels (which
    always cover every gated event) by real event identity, even though
    PHATE only ever saw a downsampled subset of them.

    af_state: optional (transfer_matrix, af_precomputed, af_spectra) snapshot —
        see apply_unmixing_af_aware() docstring. Pass this from any
        background worker thread; leave as None only for main-thread callers.
    """
    log_stage(log, "LOAD TRAINING POOL (WITH BOUNDS)")
    if not state.training_sample_ids:
        log.warning("no training samples selected")
        return None
    if not state.selected_channels:
        log.warning("no channels selected")
        return None
    if not state.selected_gates:
        log.warning("no gate selected")
        return None

    log_files(log, "training samples", state.training_sample_ids)
    log.info("per-sample event cap (n_training_events) = %d", state.n_training_events)

    rng = np.random.default_rng(seed)
    chunks = []
    bounds: list[tuple[str, int, np.ndarray]] = []
    for rel_path in state.training_sample_ids:
        feats = load_sample_features(controller, state, rel_path, af_state=af_state)
        if feats is None or feats.shape[0] == 0:
            log.warning("  %s contributed no events", rel_path)
            continue
        chunk, idx = _downsample_with_indices(feats, state.n_training_events, rng)
        log.info("  %s → %d events after downsample", rel_path, chunk.shape[0])
        chunks.append(chunk)
        bounds.append((rel_path, chunk.shape[0], idx))

    if not chunks:
        log.error("no training data could be loaded")
        return None

    data = np.concatenate(chunks, axis=0).astype(np.float32)
    log_array(log, "training_pool",
              data,
              [c for c in state.selected_channels if c not in META_CHANNELS])
    return data, bounds
