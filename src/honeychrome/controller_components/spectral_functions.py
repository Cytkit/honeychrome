import warnings
import logging

import numpy as np
from flowkit import GatingStrategy


import honeychrome.settings as settings

logger = logging.getLogger(__name__)

PROFILE_COSINE_WARNING_THRESHOLD = 0.95  # cross-profile cosine similarity QC threshold

# otb: get_best_channel can now be removed entirely
def get_best_channel(sample, gating_strategy, base_gate_label, fluorescence_channel_ids):
    from sklearn.decomposition import PCA

    # gate first by cells
    if base_gate_label != 'root':
        base_event_mask = gating_strategy.gate_sample(sample).get_gate_membership(base_gate_label) # Note this is slow
    else:
        base_event_mask = np.ones(sample.event_count, dtype=bool)
    # base_event_mask_indices = np.where(base_event_mask)[0]

    if np.sum(base_event_mask) < 2:
        return None
    else:
        pca = PCA(n_components=1)
        event_data_base_gate = sample.get_events('raw')[base_event_mask]
        event_data_fluorescence = event_data_base_gate[:, fluorescence_channel_ids]
        pca.fit(event_data_fluorescence)
        explained_variance = pca.explained_variance_ratio_[0]

        # # get top n events in principle axis
        # events_transformed_to_pca = pca.transform(event_data_fluorescence).flatten()
        # n_to_gate = 100
        # # More efficient for large arrays - doesn't fully sort array
        # indices_top_n_events = np.argpartition(events_transformed_to_pca, -n_to_gate)[-n_to_gate:]
        # # Sort the top 100 indices by value (descending)
        # indices_top_n_events = base_event_mask_indices[indices_top_n_events[np.argsort(events_transformed_to_pca[indices_top_n_events])][::-1]]

        # define gates
        # n_matches_per_fluorescence_channel = {}
        # matches_per_fluorescence_channel = {}
        # fl_raw = sample.get_events('raw')
        # for channel_id in fluorescence_channel_ids:
        #     fl = fl_raw[base_event_mask, channel_id]
        #     indices_top_n_events_fluorescence_channels = base_event_mask_indices[np.argpartition(fl, -n_to_gate)[-n_to_gate:]]
        #     matches_per_fluorescence_channel[channel_id] = list(set(indices_top_n_events) & set(indices_top_n_events_fluorescence_channels))
        #     n_matches_per_fluorescence_channel[channel_id] = len(matches_per_fluorescence_channel[channel_id])
        # # in which single channel are most of those top n events at the top?
        # channel_id_best_match = np.argmax(n_matches_per_fluorescence_channel)
        # or just peak of spectral profile (PCA component)
        channel_id_best_match = fluorescence_channel_ids[np.argmax(pca.components_[0])]
        # best_match = n_matches_per_fluorescence_channel[channel_id_best_match]
        # fl_top = fl_raw[matches_per_fluorescence_channel[channel_id_best_match], channel_id_best_match]

        fluorescence_on_best_channel = event_data_base_gate[:, channel_id_best_match]
        if len(fluorescence_on_best_channel) > 100:
            pos_percentile = settings.spectral_positive_gate_percent_retrieved
            neg_percentile = settings.spectral_negative_gate_percent_retrieved
        else:
            pos_percentile = 50
            neg_percentile = 50

        # define top gate
        fl_top = np.percentile(fluorescence_on_best_channel, [100 - pos_percentile, 100])

        # define bottom gate
        fl_bottom = np.percentile(fluorescence_on_best_channel, [0, neg_percentile])

        # print([len(fluorescence_on_best_channel), pos_percentile, neg_percentile, fl_top, fl_bottom])
        return channel_id_best_match, fl_top, fl_bottom, explained_variance

def _gate_events_via_lookup(
    all_events: np.ndarray,
    pnn: list,
    gate_label: str,
    gating_strategy,
) -> np.ndarray:
    """
    Return a boolean mask (n_events,) for events inside gate_label and all its
    ancestors, without calling gate_sample() on the full event array.

    Uses the lookup-table pattern from apply_gates_in_place() / controller.py:
    gate a small coordinate grid (transform bin edges), then map events via
    np.searchsorted.  Handles RectangleGate, PolygonGate, and EllipsoidGate
    uniformly — no per-gate-type special-casing required.

    Parameters
    ----------
    all_events : (n_events, n_channels) float64 — columns in pnn order.
    pnn        : channel name list matching all_events column order.
                 For whitelisted cytometers this is whitelisted_pnn (e.g. 78
                 channels); for others it is sample.pnn_labels (full list).
    gate_label : name of the target gate; ancestors are resolved automatically.
    gating_strategy : GatingStrategy with transforms already attached.
    """
    from flowkit import Sample as FKSample, GatingStrategy as FKGatingStrategy

    gate_ids = list(gating_strategy.find_matching_gate_paths(gate_label)[0]) + [gate_label]
    gate_ids = [g for g in gate_ids if g != 'root']

    mask = np.ones(len(all_events), dtype=np.bool_)

    for gate_id in gate_ids:
        gate = gating_strategy.get_gate(gate_id)
        channels = gate.get_dimension_ids()

        xforms = {ch: gating_strategy.transformations[ch] for ch in channels
                  if ch in gating_strategy.transformations}

        if len(channels) == 1:
            ch = channels[0]
            col_idx = pnn.index(ch)
            col = all_events[:, col_idx]
            xform = xforms.get(ch)

            if xform is not None and hasattr(xform, 'scale'):
                scale = xform.scale
            else:
                lo, hi = float(col.min()), float(col.max())
                scale = np.linspace(lo, hi, 512)

            grid = scale[1:].reshape(-1, 1)
            grid_sample = FKSample(grid, channel_labels=[ch], sample_id='_lut_1d')
            temp_gs = FKGatingStrategy()
            temp_gs.add_gate(gate, gate_path=('root',))
            if ch in xforms:
                temp_gs.transformations[ch] = xforms[ch]
            lut = temp_gs.gate_sample(grid_sample).get_gate_membership(gate_id)

            indices = np.searchsorted(scale, col) - 2
            indices = np.clip(indices, 0, len(lut) - 1)
            mask &= lut[indices]

        else:  # 2D gate (scatter, singlets, etc.)
            ch_x, ch_y = channels[0], channels[1]
            ix = pnn.index(ch_x)
            iy = pnn.index(ch_y)

            xform_x = xforms.get(ch_x)
            xform_y = xforms.get(ch_y)

            if xform_x is not None and hasattr(xform_x, 'scale'):
                scale_x = xform_x.scale
            else:
                lo, hi = float(all_events[:, ix].min()), float(all_events[:, ix].max())
                scale_x = np.linspace(lo, hi, 256)
            if xform_y is not None and hasattr(xform_y, 'scale'):
                scale_y = xform_y.scale
            else:
                lo, hi = float(all_events[:, iy].min()), float(all_events[:, iy].max())
                scale_y = np.linspace(lo, hi, 256)

            sx = scale_x[1:]
            sy = scale_y[1:]
            grid_x, grid_y = np.meshgrid(sx, sy, indexing='ij')
            coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))

            grid_sample = FKSample(coords, channel_labels=[ch_x, ch_y], sample_id='_lut_2d')
            temp_gs = FKGatingStrategy()
            temp_gs.add_gate(gate, gate_path=('root',))
            for ch in [ch_x, ch_y]:
                if ch in xforms:
                    temp_gs.transformations[ch] = xforms[ch]
            lut_flat = temp_gs.gate_sample(grid_sample).get_gate_membership(gate_id)
            lut = lut_flat.reshape(len(sx), len(sy))

            ix_ev = np.searchsorted(scale_x, all_events[:, ix]) - 2
            iy_ev = np.searchsorted(scale_y, all_events[:, iy]) - 2
            ix_ev = np.clip(ix_ev, 0, lut.shape[0] - 1)
            iy_ev = np.clip(iy_ev, 0, lut.shape[1] - 1)
            mask &= lut[ix_ev, iy_ev]

    return mask

def get_profile(
    sample,
    gate_label: str,
    raw_gating,
    fluorescence_channel_ids: list,
    preloaded_events: np.ndarray | None = None,
    preloaded_pnn: list | None = None,
    event_channels_pnn: list | None = None,
):
    """
    Return mean fluorescence profile for events within gate_label.

    preloaded_events    : (n_events, n_channels) float64, already loaded by the
                          caller.  If None, loaded here from sample.
    preloaded_pnn       : channel name list matching preloaded_events column
                          order.  For whitelisted cytometers this is
                          whitelisted_pnn; for others, sample.pnn_labels.
                          Must be supplied when preloaded_events is supplied.
    event_channels_pnn  : the full settings['raw']['event_channels_pnn'] list.
                          Required when preloaded_pnn is supplied, so that
                          fluorescence_channel_ids (which index into this full
                          list) can be translated to positions in the
                          preloaded subset.
    """
    if preloaded_events is not None:
        all_events = preloaded_events
        pnn = preloaded_pnn
    else:
        all_events = sample.get_events('raw')
        pnn = sample.pnn_labels

    if gate_label != 'root':
        event_mask = _gate_events_via_lookup(all_events, pnn, gate_label, raw_gating)
    else:
        event_mask = np.ones(len(all_events), dtype=np.bool_)

    if event_mask.sum() > 0:
        profile = all_events[event_mask].mean(axis=0)
        if preloaded_pnn is not None and event_channels_pnn is not None:
            # fluorescence_channel_ids index into event_channels_pnn (full list).
            # Translate to column positions within the preloaded subset by name.
            # Mirrors the pattern in controller.py:initialise_ephemeral_data().
            fl_cols = [
                preloaded_pnn.index(event_channels_pnn[i])
                for i in fluorescence_channel_ids
                if event_channels_pnn[i] in preloaded_pnn
            ]
        else:
            fl_cols = fluorescence_channel_ids
        profile = profile[fl_cols]
    else:
        profile = np.zeros(len(fluorescence_channel_ids))
        warnings.warn('No events in gate')

    return profile

def compute_sample_means_for_wls(
    experiment_dir,
    experiment_samples: dict,
    fluorescence_channel_ids: list,
    raw_settings: dict | None = None,
) -> 'np.ndarray | None':
    """
    Compute per-detector mean raw fluorescence across real experimental samples,
    for use as WLS (Poisson) weights.

    Selection priority:
      1. Samples in all_samples that are NOT in single_stain_controls and NOT
         tagged or named as unstained.
      2. If none qualify, fall back to all samples in all_samples.

    Returns a (n_channels,) float64 array, or None if no files could be read.
    """
    import re
    from pathlib import Path
    from honeychrome.controller_components.functions import sample_from_fcs

    all_samples = experiment_samples.get('all_samples', {})
    controls = set(experiment_samples.get('single_stain_controls', []))
    unstained = set(experiment_samples.get('unstained_samples', []))

    def _is_unstained(path, name):
        return (path in unstained
                or re.search(r'unstained', name, re.IGNORECASE)
                or re.search(r'unstained', path, re.IGNORECASE))

    preferred = [
        p for p, name in all_samples.items()
        if p not in controls and not _is_unstained(p, name)
    ]
    candidates = preferred if preferred else list(all_samples.keys())

    if not candidates:
        return None

    # For whitelisted cytometers (e.g. FACSDiscover) the full FCS array has
    # heterogeneous channel counts across files; load via col_order so that
    # fluorescence_channel_ids (full-PNN indices) map consistently.
    whitelisted_pnn = (raw_settings or {}).get('whitelisted_pnn')
    full_pnn = (raw_settings or {}).get('event_channels_pnn')
    if whitelisted_pnn and full_pnn:
        fl_ids_local = [whitelisted_pnn.index(full_pnn[i]) for i in fluorescence_channel_ids]
        col_order = whitelisted_pnn
    else:
        fl_ids_local = fluorescence_channel_ids
        col_order = None

    channel_sums = None
    n_events_total = 0
    for path in candidates:
        try:
            sample = sample_from_fcs(Path(experiment_dir) / path)
            try:
                all_events = sample.get_events('raw', col_order=col_order)
            except (KeyError, ValueError):
                all_events = sample.get_events('raw')
            fl = all_events[:, fl_ids_local].astype(np.float64)
            channel_sums = fl.sum(axis=0) if channel_sums is None else channel_sums + fl.sum(axis=0)
            n_events_total += fl.shape[0]
        except Exception:
            continue

    if channel_sums is None or n_events_total == 0:
        return None

    return channel_sums / n_events_total


def _build_omega_inv(
    M: np.ndarray,
    method: str = 'OLS',
    sample_means: 'np.ndarray | None' = None,
) -> np.ndarray:
    """
    Build diagonal weight matrix Ω⁻¹ for spectral unmixing.
    M shape: (n_fluors, n_channels). Returns (n_channels, n_channels).

    For WLS (Poisson), sample_means must be provided: a (n_channels,) array of
    mean raw fluorescence across the experiment's real samples.
    Falls back to OLS if sample_means is None.
    """
    n_channels = M.shape[1]
    if method == 'WLS' and sample_means is not None:
        weights = np.where(sample_means > 1e-9, sample_means, 1.0)
    else:
        weights = np.ones(n_channels)
    return np.diag(1.0 / weights)

def calculate_spectral_process(raw_settings, spectral_model, profiles,
                                existing_spillover=None, unmixing_method='OLS',
                                experiment_dir=None, experiment_samples=None,
                                filtered_fluorescence_channel_ids=None):
    from sklearn.metrics.pairwise import cosine_similarity
    from pandas import DataFrame

    fluorescence_channels = [control['label'] for control in spectral_model]
    
    # Build DataFrame with columns in spectral_model order, not dict insertion order.
    # profiles dict key order may differ from spectral_model order after JSON round-trip.
    profiles_df = DataFrame({label: profiles[label] for label in fluorescence_channels})
    M = np.array(profiles_df).T
    raw_length = np.shape(M)[1]
    unmixed_length = np.shape(M)[0]

    Mnorm = M / np.tile(np.sqrt(np.sum(M**2, axis=0)), (unmixed_length,1))
    Mnorm[np.isnan(Mnorm)] = 0
    similarity_matrix = cosine_similarity(Mnorm)
    hotspot_matrix = np.sqrt(np.abs(np.linalg.inv(similarity_matrix)))

    # --- Spectral profile QC: condition number + inter-profile cosine similarity ---
    # M is (n_fluorophores, n_channels); a high condition number here means the
    # unmixing system (M @ Omega_inv @ M.T) is close to singular.
    conditioning_warnings = []
    n_profiles = unmixed_length
    cond_number = np.linalg.cond(M)
    if cond_number > n_profiles:
        msg = (
            f'Spectral profile matrix is ill-conditioned: condition number '
            f'{cond_number:.1f} exceeds the number of fluorophores ({n_profiles}). '
            f'Unmixing may be unstable — check for redundant or highly similar profiles.'
        )
        conditioning_warnings.append(msg)
        logger.warning(f'calculate_spectral_process: {msg}')

    high_similarity_pairs = [
        (fluorescence_channels[i], fluorescence_channels[j], similarity_matrix[i, j])
        for i in range(n_profiles)
        for j in range(i + 1, n_profiles)
        if similarity_matrix[i, j] > PROFILE_COSINE_WARNING_THRESHOLD
    ]
    if high_similarity_pairs:
        pairs_text = ', '.join(
            f'"{a}" / "{b}" ({sim:.3f})' for a, b, sim in high_similarity_pairs
        )
        msg = (
            f'The following profile pair(s) have cosine similarity above '
            f'{PROFILE_COSINE_WARNING_THRESHOLD}: {pairs_text}. '
            f'These fluorophores may unmix poorly.'
        )
        conditioning_warnings.append(msg)
        logger.warning(f'calculate_spectral_process: {msg}')

    # calculate unmixing matrix
    variance_per_detector = np.ones(raw_length)  # trivial example... try something better like cv on each detector for brightest fluorophores?
    # otb: pure variance is too noisy and varies between samples. could be done per experiment
    # sd is a bit better, but mean (Poisson-like) works well empirically
    # we probably will want to measure the noise in the detectors on the CytKit
    sample_means = None
    if unmixing_method == 'WLS' and experiment_dir is not None and experiment_samples is not None:
        # fl_ids must index event_channels_pnn at the detector-channel granularity
        # of the profile vectors (M.shape[1]), not at the per-control/label
        # granularity of fluorescence_channels (M.shape[0]). The caller passes
        # the same filtered channel ids used to build the profiles.
        fl_ids = filtered_fluorescence_channel_ids
        if fl_ids is None:
            fl_ids = raw_settings.get('fluorescence_channel_ids', [])
        sample_means = compute_sample_means_for_wls(experiment_dir, experiment_samples, fl_ids, raw_settings)
    Omega_inv = _build_omega_inv(M, method=unmixing_method, sample_means=sample_means)
    unmixing_matrix = np.linalg.inv(M @ Omega_inv @ M.T) @ M @ Omega_inv  # "W" matrix in Novo paper

    # define unmixed channels
    pnn_raw = raw_settings['event_channels_pnn']
    scatter_channel_ids_raw = raw_settings['scatter_channel_ids']
    scatter_channels_pnn = [pnn_raw[n] for n in scatter_channel_ids_raw]
    fluorescence_channels_pnn = fluorescence_channels
    # synthetic placeholder if raw data has no Time channel (mirrors 'event_id' below)
    time_channel_name_raw = (
        pnn_raw[raw_settings['time_channel_id']]
        if raw_settings['time_channel_id'] is not None else 'Time'
    )
    event_channels_pnn = ([time_channel_name_raw, 'event_id'] + scatter_channels_pnn
                          + fluorescence_channels_pnn + list(settings.af_channels))
    area_channels = [s.removesuffix('-A') for s in event_channels_pnn if s.endswith("-A")]
    height_channels = [s.removesuffix('-H') for s in event_channels_pnn if s.endswith("-H")]
    width_channels = [s.removesuffix('-W') for s in event_channels_pnn if s.endswith("-W")]

    time_channel_id = 0
    event_id_channel_id = 1

    scatter_channel_ids = [event_channels_pnn.index(c) for c in scatter_channels_pnn]
    n_scatter_channels = len(scatter_channel_ids)
    fluorescence_channel_ids = [event_channels_pnn.index(c) for c in fluorescence_channels_pnn]
    n_fluorophore_channels = len(fluorescence_channel_ids)
    af_channel_ids = [event_channels_pnn.index(c) for c in settings.af_channels]

    unmixed_settings = {
        'unmixed_samples_subdirectory': 'Unmixed',
        'area_channels': area_channels,
        'height_channels': height_channels,
        'width_channels': width_channels,
        'scatter_channels': scatter_channels_pnn, # consider stripping the -A, -H -W from this, but not sure if it matters
        'fluorescence_channels': fluorescence_channels,
        'event_channels_pnn': event_channels_pnn,
        'width_ceiling': raw_settings['width_ceiling'],
        'magnitude_ceiling': raw_settings['magnitude_ceiling'],
        'default_ceiling': raw_settings['default_ceiling'],
        'time_channel_id': time_channel_id,
        'event_id_channel_id': event_id_channel_id,
        'scatter_channel_ids': scatter_channel_ids,
        'n_scatter_channels': n_scatter_channels,
        'fluorescence_channel_ids': fluorescence_channel_ids,
        'n_fluorophore_channels': n_fluorophore_channels,
        'af_channel_ids': af_channel_ids,
    }

    # Preserve existing spillover if it is the right size; otherwise reset to identity.
    if (existing_spillover is not None
            and np.array(existing_spillover).shape == (n_fluorophore_channels, n_fluorophore_channels)):
        spillover = np.array(existing_spillover)
    else:
        spillover = np.eye(n_fluorophore_channels)


    # Extract the diagonal weights for FCS export (Ω diagonal = 1/weight)
    omega_diag = np.diag(Omega_inv)
    weights_vector = np.where(omega_diag > 0, 1.0 / omega_diag, 1.0)

    # populate process variables
    spectral_process = {
        'similarity_matrix': similarity_matrix.tolist(),
        'hotspot_matrix': hotspot_matrix.tolist(),
        'unmixing_matrix': unmixing_matrix.tolist(),
        'spillover': spillover.tolist(),
        'unmixing_method': unmixing_method,
        'unmixing_weights': weights_vector.tolist(),
    }

    # # set up unmixed channels with default transforms
    # unmixed_cytometry = {
    #     'transforms': assign_default_transforms(unmixed_settings),
    #     'plots': []
    # }
    # transformations = generate_transformations(unmixed_cytometry['transforms'])
    #
    # # initialise unmixed gating and plots
    # gating = fk.GatingStrategy()
    # for label in event_channels_pnn:
    #     gating.transformations[label] = transformations[label].xform

    return unmixed_settings, spectral_process, conditioning_warnings

def sanitise_control_in_place(control):
    if control['control_type'] == 'Single Stained Spectral Control':
        pass  # gate_channel is set by the auto-generator and must not be cleared on re-generation
    elif control['control_type'] == 'Single Stained Spectral Control from Library':
        control['particle_type'] = ''
        control['gate_channel'] = ''
        control['gate_label'] = ''
    elif control['control_type'] == 'Channel Assignment':
        control['particle_type'] = ''
        control['sample_name'] = ''
        control['gate_label'] = ''
    else:
        control['particle_type'] = ''
        control['gate_channel'] = ''
        control['sample_name'] = ''
        control['gate_label'] = ''

    # print(json.dumps(control, indent=2))

def _find_default_unstained(samples: dict) -> str | None:
    """Return the tube name of the first sample whose path or name matches 'unstained', or None."""
    import re
    for path, name in samples.items():
        if re.search(r'unstained', path, re.IGNORECASE) or re.search(r'unstained', name, re.IGNORECASE):
            return name
    return None


def get_raw_events(
    sample,
    fluorescence_channel_ids: list,
    gate_label: str | None = None,
    gating_strategy=None,
    extra_channel_ids: list | None = None,
    col_order: list | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Return a (n_events, n_channels) float64 array of raw fluorescence values.

    If gate_label is supplied (and gating_strategy is not None), only events
    inside that gate are returned. Otherwise all events are returned.

    If extra_channel_ids is supplied (e.g. scatter channel indices), a second
    array of shape (n_events, len(extra_channel_ids)) is returned as a tuple:
        (fluorescence_array, extra_array)
    This is the foundation that every subsequent cleaning stage builds on.

    col_order: if supplied, passed to get_events() to select/reorder columns by
    PnN name (used for FACSDiscover files with inconsistent derived-parameter sets).
    """
    try:
        all_events = sample.get_events('raw', col_order=col_order)
    except (KeyError, ValueError) as e:
        logger.warning('get_raw_events: col_order failed (%s) — reading all channels', e)
        all_events = sample.get_events('raw')

    if np.any(np.isnan(all_events)):
        n_nan = int(np.isnan(all_events).sum())
        logger.warning('get_raw_events: %d NaN values — replacing with 0', n_nan)
        all_events = np.where(np.isnan(all_events), 0.0, all_events)

    if gate_label and gating_strategy:
        gate_paths = gating_strategy.find_matching_gate_paths(gate_label)
        if gate_paths:
            # col_order is the whitelisted_pnn list when set, otherwise None.
            # all_events was loaded with col_order, so its columns match
            # col_order (or sample.pnn_labels when col_order is None).
            pnn = col_order if col_order is not None else sample.pnn_labels
            event_mask = _gate_events_via_lookup(all_events, pnn, gate_label, gating_strategy)
        else:
            warnings.warn(f'get_raw_events: gate "{gate_label}" not found — returning all events.')
            event_mask = np.ones(len(all_events), dtype=np.bool_)
    else:
        event_mask = np.ones(len(all_events), dtype=np.bool_)

    gated = all_events[event_mask]
    fluor = gated[:, fluorescence_channel_ids].astype(np.float64)

    if extra_channel_ids is not None:
        extra = gated[:, extra_channel_ids].astype(np.float64)
        return fluor, extra

    return fluor
