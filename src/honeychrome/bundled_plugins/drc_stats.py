"""
drc_stats.py — Differential statistics for the Honeycluster plugin
===================================================================
Companion to ``dr_clustering_tab.py`` (filename intentionally NOT ``*_tab.py``).

The test engine (design matrices, moderated fits, GLMs, significance) is
shared with other plugins in ``controller_components/differential_stats.py``;
this module assembles the Honeycluster feature tables and runs it.

Tests cluster abundance and per-cluster marker expression between sample
groups:

* Frequencies — log2 of each cluster's share of a sample's events, tested
  with a moderated linear model (``moderated_stats``: least squares per
  cluster, empirical Bayes variance moderation after Smyth 2004).
* Counts — raw event counts per cluster, tested with a negative-binomial
  GLM (statsmodels) with a log(total events) offset.
* MFIs — mean transformed intensity of each marker within each cluster,
  tested with the same moderated linear model. A cluster with no events in
  a sample is missing (NaN) for that sample, not 0.

Every test builds one design matrix: an intercept, one indicator per
non-baseline group, optional pairing (blocking) levels, and optional
adjustment covariates — numeric columns enter as centred continuous
terms, anything else as treatment-coded factors.

Significance is set by ``apply_significance``:

* TREAT (McCarthy & Smyth 2009): with ``use_treat`` the p-value tests
  whether |effect| exceeds the threshold, instead of filtering on the
  estimate after testing for a non-zero effect.
* FDR: Benjamini–Hochberg, pooled over every comparison ('global') or
  within each comparison ('per_comparison').
* Cluster-first MFI testing (``hierarchical=True``): clusters are screened
  first (Simes-combined p-value of their markers, BH across clusters);
  markers are then tested only inside selected clusters, at the level
  adjusted for the number of clusters selected (Benjamini & Bogomolov
  2014).

Results carry the estimate's standard error and degrees of freedom, so a
threshold change re-derives p-values and significance without refitting.

Contrasts: ``state.contrast_mode`` 'reference' fits once across every
tested group and reads one coefficient per non-reference group;
'pairwise' fits each pair of groups separately. Results carry a
``comparison`` column.

Trends with a numeric covariate (``run_trend_statistics``): instead of
comparing groups, each cluster's frequency, counts (as log odds) and MFIs
are tested for a change with a number such as age or days after
vaccination (``controller_components.trend_stats``): a curve F test and a
slope test, partial R², and a call by either test. The samples are those
of Groups to Test; ``reapply_trend_thresholds``, ``relabel_trend_results``
and ``trend_to_bundle`` / ``trend_from_bundle`` re-threshold, rename and
store the results.

Populations within samples (``state.population_split``): instead of
comparing groups of samples, each sample is split into the gated
populations in ``state.population_gates`` (for example CD45.1+ and
CD45.1- cells of a mixed bone-marrow chimera) and the populations are the
groups. Every population of a sample becomes one row of the feature tables
(``population_key``), its frequencies are shares of that population's own
events, and the design is paired on the sample, so each mouse's
populations are compared with each other and mouse-to-mouse differences
drop out. Sample groups ticked under Groups to Test choose which samples
are split. Events inside more than one chosen population are left out of
all of them; a sample where they are more than
``POPULATION_OVERLAP_LIMIT`` of its events in the populations stops the
run, since the populations would then share cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import drc_pipeline
import drc_run_align
from drc_logging import get_logger, log_stage
from honeychrome.controller_components import differential_stats as _ds
from honeychrome.controller_components.differential_stats import (
    apply_significance,
    build_contrasts,
    covariate_kind,
    log2_frequencies,
    run_glm_counts,
    run_moderated,
    sparse_features,
)

log = get_logger(__name__)

# A cluster is tested for differential abundance only when it has events in
# at least this many samples; below that the comparison reflects the
# samples' event totals, not the cluster.
MIN_NONZERO_SAMPLES = 3

# Largest share of a sample's events in the chosen populations that may lie
# in more than one of them (those events are left out of every population).
POPULATION_OVERLAP_LIMIT = 0.01


# ---------------------------------------------------------------------------
# Group resolution
# ---------------------------------------------------------------------------

def resolve_test_groups(controller, state, cluster_labels_override=None):
    """
    Resolve every group in ``state.testing_group_selection`` 
    (falls back to every defined group if the selection is empty) to
    rel-path keys present in ``cluster_labels_override`` (or
    ``state.cluster_labels`` if not supplied).

    Groups with < 3 labelled samples are silently DROPPED from the
    returned dict rather than raising — a user testing 5 groups where one
    only has 2 samples should still get results for the other 4;
    ``build_contrasts``/the caller decide whether what's left is enough.

    Returns dict[group_name, list[rel_path]] for qualifying groups only.
    Raises RuntimeError if fewer than 2 groups qualify.
    """
    log_stage(log, "RESOLVE TEST GROUPS")
    cluster_labels = cluster_labels_override if cluster_labels_override is not None \
        else state.cluster_labels
    raw_subdir = controller.experiment.settings['raw']['raw_samples_subdirectory']

    selection = state.testing_group_selection or list(state.group_names)
    result = {}
    for name in selection:
        rels = []
        for sp, g in state.sample_groups.items():
            if g != name:
                continue
            rel = _rel_path(sp, raw_subdir)
            if rel in cluster_labels:
                rels.append(rel)
        log.info("group %r: %d with cluster labels", name, len(rels))
        if len(rels) >= 3:
            result[name] = rels
        else:
            log.info("group %r dropped from testing (only %d qualifying samples)",
                     name, len(rels))

    if len(result) < 2:
        raise RuntimeError(
            "Need at least 2 groups with ≥3 labelled samples each in "
            "'Groups to Test'. Qualifying: "
            + (", ".join(f"{k}={len(v)}" for k, v in result.items()) or "none")
        )
    return result


def _rel_path(sample_path: str, raw_subdir) -> str:
    """A sample's path relative to the raw samples folder (unchanged when
    it is not inside it), the key of cluster labels and covariates."""
    try:
        return str(Path(sample_path).relative_to(raw_subdir))
    except ValueError:
        return sample_path


# ---------------------------------------------------------------------------
# Populations within samples
# ---------------------------------------------------------------------------

def population_key(rel: str, gate: str) -> str:
    """
    Row name of one population of one sample in the statistics tables: the
    sample's path with ' · <gate>' added to its file name, so every plot
    that labels samples by file name shows e.g. 'Mouse1 · CD45.1+'.
    """
    p = Path(rel)
    label = str(gate).replace('/', '\u2215').replace('\\', '\u2215')
    return str(p.with_name(f"{p.stem} \u00b7 {label}{p.suffix}"))


def population_samples(controller, state, cluster_labels: dict) -> list[str]:
    """
    Samples to split into populations: labelled samples whose group is
    ticked under Groups to Test (every defined group when none is ticked).
    Unassigned samples are left out, as in a comparison of sample groups.
    """
    raw_subdir = controller.experiment.settings['raw']['raw_samples_subdirectory']
    selection = set(state.testing_group_selection or state.group_names)
    out = []
    for sp, g in state.sample_groups.items():
        if g not in selection:
            continue
        rel = _rel_path(sp, raw_subdir)
        if rel in cluster_labels and rel not in out:
            out.append(rel)
    return out


@dataclass
class PopulationSplit:
    """Each chosen sample split into the chosen populations.

    labels        : {key: cluster labels of that population's events}
    groups        : {gate: [keys]}, in the order of state.population_gates
    population_of : {key: (sample rel path, gate)}
    masks         : {key: bool mask over the sample's labelled events}
    note          : what was left out, for the status line
    """
    labels: dict = field(default_factory=dict)
    groups: dict = field(default_factory=dict)
    population_of: dict = field(default_factory=dict)
    masks: dict = field(default_factory=dict)
    note: str = ''


def split_by_population(controller, state, cluster_labels: dict,
                        af_state=None) -> PopulationSplit:
    """
    Split every sample of population_samples() into the gates of
    state.population_gates. *cluster_labels* must be row for row with each
    sample's currently gated events (an in-sync run, or one matched to the
    current events by event key).

    A population with no labelled events in a sample gives no row for that
    sample. Raises RuntimeError when fewer than two populations are chosen,
    no sample qualifies, or the populations overlap by more than
    POPULATION_OVERLAP_LIMIT in any sample.
    """
    log_stage(log, "SPLIT SAMPLES BY POPULATION")
    gates = list(dict.fromkeys(state.population_gates))
    if len(gates) < 2:
        raise RuntimeError("Choose at least two populations to compare under "
                           "'Compare populations within samples'.")
    rels = population_samples(controller, state, cluster_labels)
    if not rels:
        raise RuntimeError("No clustered samples belong to the groups ticked under "
                           "Groups to Test.")

    split = PopulationSplit(groups={g: [] for g in gates})
    n_overlap, n_empty, unmatched = 0, 0, []
    for rel in rels:
        lab = np.asarray(cluster_labels[rel])
        loaded = drc_pipeline.load_sample_populations(controller, state, rel, gates,
                                                      af_state=af_state)
        if loaded is None or len(loaded[0]) != len(lab):
            log.warning("population split: %s -- membership does not match its %d labels, "
                        "sample left out", rel, len(lab))
            unmatched.append(rel)
            continue
        membership = loaded[0]
        inside = membership.sum(axis=1)
        overlap = inside > 1
        n_in = int(np.count_nonzero(inside))
        n_shared = int(np.count_nonzero(overlap))
        if n_in and n_shared > POPULATION_OVERLAP_LIMIT * n_in:
            raise RuntimeError(
                f"The chosen populations share cells: {n_shared:,} of the {n_in:,} events of "
                f"{Path(rel).stem} in them are in more than one. Choose populations that do "
                "not overlap (for example CD45.1+ and CD45.1-).")
        n_overlap += n_shared
        labelled = lab != drc_run_align.UNLABELLED
        for j, g in enumerate(gates):
            mask = membership[:, j] & ~overlap
            if not np.any(mask & labelled):
                n_empty += 1
                continue
            key = population_key(rel, g)
            split.labels[key] = lab[mask]
            split.masks[key] = mask
            split.population_of[key] = (rel, g)
            split.groups[g].append(key)
        log.info("population split: %s -- %s", rel,
                 ", ".join(f"{g}={int(membership[:, j].sum())}" for j, g in enumerate(gates)))

    if len(unmatched) == len(rels):
        raise RuntimeError(
            "No sample's clustered cells could be matched to the populations. The run "
            "must be in step with the gates checked on the Configuration tab: select the "
            "gates it was made with, or re-run the clustering.")
    parts = [f"{len(rels) - len(unmatched)} sample(s) split into {len(gates)} populations."]
    if n_overlap:
        parts.append(f"{n_overlap:,} event(s) inside more than one population were left out.")
    if n_empty:
        parts.append(f"{n_empty} sample population(s) with no events were left out.")
    if unmatched:
        parts.append(f"{len(unmatched)} sample(s) could not be matched to their populations.")
    split.note = " ".join(parts)
    log.info("population split: %s", split.note)
    return split


def population_values(controller, state, split: PopulationSplit, channels: list[str],
                      af_state=None) -> dict:
    """
    {key: (transformed values, channel names)} of each population's events
    for *channels*, row for row with split.labels[key]; each sample is
    loaded once.
    """
    by_rel: dict[str, list[str]] = {}
    for key, (rel, _gate) in split.population_of.items():
        by_rel.setdefault(rel, []).append(key)
    out = {}
    for rel, keys in by_rel.items():
        mv = drc_pipeline.load_sample_transformed_values(controller, state, rel, channels,
                                                         af_state=af_state)
        if mv is None:
            continue
        values, names = mv
        for key in keys:
            mask = split.masks[key]
            if len(mask) == len(values):
                out[key] = (values[mask], names)
    return out


def missing_covariate_values(state, all_rel: list[str], names: list[str]) -> dict[str, list[str]]:
    """
    {covariate name: [rel paths with no value]} for each of ``names``
    that is missing (or blank) for at least one sample in ``all_rel``.
    """
    return _ds.missing_covariate_values(state.covariates, all_rel, names)


def covariate_frame(state, all_rel: list[str], names: list[str]) -> pd.DataFrame | None:
    """
    Adjustment covariates for ``all_rel`` (rows in that order): the columns
    in ``state.numeric_covariates`` as floats, the others as strings.
    A sample without a value gets None (NaN in a numeric column); the tests
    then leave out that covariate or those samples comparison by comparison
    (differential_stats.comparison_parts). Returns None when ``names`` is
    empty. Raises RuntimeError if a numeric covariate has a value that is
    not a number.
    """
    return _ds.covariate_frame(state.covariates, all_rel, names,
                               numeric=list(state.numeric_covariates), allow_missing=True)


def n_clusters_from_labels(state, all_rel, cluster_labels_override=None) -> int:
    """Number of (non-noise) clusters across the given samples. 0-based labels,
    so result = max_label + 1. Guards the all-noise case."""
    cluster_labels = cluster_labels_override if cluster_labels_override is not None \
        else state.cluster_labels
    positive = [int(lbl)
                for rel in all_rel
                for lbl in np.unique(cluster_labels[rel]) if lbl >= 0]
    if not positive:
        raise RuntimeError("No non-noise clusters to test.")
    n = int(max(positive) + 1)
    log.info("n_clusters across %d samples: %d", len(all_rel), n)
    return n


# ---------------------------------------------------------------------------
# Feature matrices
# ---------------------------------------------------------------------------

def _label_for(state, cl: int, names_override: dict | None) -> str:
    """
    Resolve a cluster's display name. When names_override is given (the
    SELECTED run's own 'names' dict — see drc_scatter.py), it is
    authoritative and state.cluster_names is never consulted, so labels
    can't bleed in from an unrelated run. Falls back to the legacy global
    dict only for the "Active (unsaved)" pseudo-run, which has no run
    entry of its own yet.
    """
    if names_override is not None:
        if cl in names_override:
            return names_override[cl]
        return 'Noise' if cl < 0 else str(cl)
    return state.cluster_label(cl)


def compute_frequencies(state, all_rel, n_clusters, cluster_labels_override=None,
                        names_override: dict | None = None) -> pd.DataFrame:
    """Per-sample % of events in each cluster → (n_samples × n_clusters)."""
    cluster_labels = cluster_labels_override if cluster_labels_override is not None \
        else state.cluster_labels
    freq_mat = np.zeros((len(all_rel), n_clusters), dtype=float)
    for i, rel in enumerate(all_rel):
        labels = np.asarray(cluster_labels[rel])
        total = max(int(np.sum(labels != drc_run_align.UNLABELLED)), 1)
        in_range = (labels >= 0) & (labels < n_clusters)
        freq_mat[i] = np.bincount(labels[in_range].astype(np.intp),
                                  minlength=n_clusters) / total * 100.0
    df = pd.DataFrame(freq_mat, index=all_rel,
                      columns=[_label_for(state, cl, names_override) for cl in range(n_clusters)])
    log.info("frequency matrix: %s", df.shape)
    return df


def compute_counts(state, all_rel, n_clusters, cluster_labels_override=None,
                   names_override: dict | None = None) -> pd.DataFrame:
    """
    Per-sample RAW event count in each cluster → (n_samples by n_clusters).

    Same shape/index/columns as compute_frequencies() — just unnormalized.
    This is the counterpart the GLM-on-counts path needs alongside the
    existing percentage matrix; run_statistics() computes both from the same
    already-resolved all_rel/n_clusters rather than each re-deriving them.
    """
    cluster_labels = cluster_labels_override if cluster_labels_override is not None \
        else state.cluster_labels
    count_mat = np.zeros((len(all_rel), n_clusters), dtype=float)
    for i, rel in enumerate(all_rel):
        labels = np.asarray(cluster_labels[rel])
        in_range = (labels >= 0) & (labels < n_clusters)
        count_mat[i] = np.bincount(labels[in_range].astype(np.intp), minlength=n_clusters)
    df = pd.DataFrame(count_mat, index=all_rel,
                      columns=[_label_for(state, cl, names_override) for cl in range(n_clusters)])
    log.info("counts matrix: %s", df.shape)
    return df


def sample_event_totals(state, all_rel, cluster_labels_override=None) -> np.ndarray:
    """Events per sample, including noise (-1) labels but not events the run
    could not label (drc_run_align.UNLABELLED) — the frequency denominator."""
    cluster_labels = cluster_labels_override if cluster_labels_override is not None \
        else state.cluster_labels
    return np.array([np.sum(np.asarray(cluster_labels[rel]) != drc_run_align.UNLABELLED)
                     for rel in all_rel], dtype=float)


def export_tables(state, mem_labels: dict | None = None,
                  cell_types: pd.DataFrame | None = None) -> dict:
    """
    The last statistics run as one tidy table per file, for graphing and
    analysis in other software: {name: DataFrame}.

    ``cluster_frequencies`` (% of a sample's labelled events),
    ``cluster_counts`` and ``cluster_mfis`` have one row per sample and one
    column per cluster (per cluster and channel for the MFIs). Each row
    starts with the sample name and path, its group and its covariates.
    Frequencies and counts then give ``n_events``, the sample's labelled
    events: the frequency denominator, noise included, recovered from the
    largest cluster's count and share.

    ``clusters`` lists each cluster's id, name and event count over the
    samples tested, with the MEM label and suggested cell type when
    ``mem_labels`` ({cluster id: label}) and ``cell_types`` (the Cluster
    Annotation suggestions, indexed by cluster id) are given.

    ``differential_frequencies``, ``differential_counts`` and
    ``differential_mfis`` are the test results as computed, one row per
    feature and comparison. A test that was not run gives no table.
    """
    out = {}
    matrices = {name: df for name, df in (('cluster_frequencies', state.freq_df),
                                          ('cluster_counts', state.counts_df),
                                          ('cluster_mfis', state.mfi_df))
                if df is not None and not df.empty}
    freq_df, counts_df = matrices.get('cluster_frequencies'), matrices.get('cluster_counts')

    per_cluster = counts_df if counts_df is not None else freq_df
    if per_cluster is not None:
        cluster_ids = list(range(per_cluster.shape[1]))
        clusters = pd.DataFrame({'cluster_id': cluster_ids,
                                 'cluster_name': list(per_cluster.columns)})
        if counts_df is not None:
            clusters['n_events'] = counts_df.sum(axis=0).astype(int).to_numpy()
        if mem_labels:
            clusters['mem_label'] = [mem_labels.get(cl, '') for cl in cluster_ids]
        if cell_types is not None and 'suggested_type' in cell_types.columns:
            clusters['suggested_type'] = (cell_types['suggested_type'].reindex(cluster_ids)
                                          .fillna('').to_numpy())
        out['clusters'] = clusters

    if matrices:
        index = next(iter(matrices.values())).index
        group_of = dict(zip(state.stats_all_rel, state.stats_group_vec))
        population_of = getattr(state, 'stats_population_of', None) or {}
        sample_of = {key: population_of[key][0] if key in population_of else key
                     for key in index}
        meta = pd.DataFrame({'sample': [Path(sample_of[key]).name for key in index],
                             'sample_path': [sample_of[key] for key in index]},
                            index=index)
        if population_of:
            meta['population'] = [population_of[key][1] if key in population_of else ''
                                  for key in index]
        meta['group'] = [group_of.get(key, '') for key in index]
        if state.covariates is not None:
            covariates = state.covariates.reindex([sample_of[key] for key in index])
            covariates.index = index
            meta = meta.join(covariates, rsuffix='_covariate')
        n_events = None
        if freq_df is not None and counts_df is not None:
            rows = np.arange(len(counts_df))
            top = counts_df.to_numpy().argmax(axis=1)
            top_count = counts_df.to_numpy()[rows, top]
            share = freq_df.to_numpy()[rows, top] / 100.0
            n_events = pd.Series(np.rint(top_count / np.where(share > 0, share, np.nan)),
                                 index=index, name='n_events')
        for name, df in matrices.items():
            parts = [meta] if n_events is None or name == 'cluster_mfis' else [meta, n_events]
            out[name] = pd.concat(parts + [df], axis=1)

    for name, results in (('differential_frequencies', state.freq_results),
                          ('differential_counts', state.counts_results),
                          ('differential_mfis', state.mfi_results)):
        if results is not None and not results.empty:
            out[name] = results
    return out


def mfi_feature_table(state, n_clusters: int, channels: list[str],
                      names_override: dict | None = None) -> pd.DataFrame:
    """
    One row per compute_mfis() column, in the same order: feature name,
    cluster display label, cluster index and channel.
    """
    rows = []
    for ch in channels:
        for cl in range(n_clusters):
            label = _label_for(state, cl, names_override)
            rows.append({'feature': f'{label}_{ch}', 'cluster': label,
                         'cluster_id': cl, 'channel': ch})
    return pd.DataFrame(rows)


def resolve_mfi_channels(state, include_type_markers: bool = False) -> list[str]:
    """
    Return the channel list MFI significance testing should use, filtered
    by marker role (diffcyt's type/state split). 'type'
    channels (those that drove the clustering assignment) can be excluded by
    default, to avoid the same channel driving both the cluster call and
    its own significance test. include_type_markers=True restores the
    previous all-selected-channels behaviour.
    """
    channels = [c for c in state.selected_channels if c not in drc_pipeline.META_CHANNELS]
    if include_type_markers:
        return channels
    return [ch for ch in channels if state.marker_roles.get(ch, 'state') != 'type']


def compute_mfis(controller, state, all_rel, n_clusters,
                 cluster_labels_override=None, channels=None,
                 names_override: dict | None = None,
                 af_state=None, values_override: dict | None = None,
                 loaded: dict | None = None) -> pd.DataFrame | None:
    """
    Per-sample mean intensity of each selected channel within each cluster.

    Returns (n_samples by (n_clusters · n_channels)), columns ordered
    channel-major (every cluster for the first channel, then the next).
    Loads each sample's selected-channel values ONCE via drc_pipeline, each
    channel on its own configured Transforms-tab scale (Logicle/
    biexponential/linear — the same scale the main cytometry plots and
    Transforms tab use), then iterates channels in memory. A cluster with
    no events in a sample is NaN there: it has no intensity to compare.

    channels: explicit channel list to test. Defaults to every
              selected channel when not supplied,
              so any other caller keeps working unchanged.

    af_state: optional unmixing snapshot, captured on the main thread
        before this (background-thread) call —
        see drc_pipeline.apply_unmixing_af_aware() docstring for why this
        must not be read live off ``controller`` from a worker thread.
    values_override: {rel: (transformed values, names)} row for row with
        the labels, used instead of the current events (a legacy run's own
        stored values, see drc_run_align.statistics_inputs).

    loaded: optional dict, filled with each sample's loaded (values, names)
        so compute_sample_mfis() can reuse them (not with values_override).

    A sample whose values and labels differ in length cannot be paired
    and is left out (NaN), never truncated.
    """
    log_stage(log, "MFI MATRIX")
    cluster_labels = cluster_labels_override if cluster_labels_override is not None \
        else state.cluster_labels
    if channels is None:
        channels = [c for c in state.selected_channels if c not in drc_pipeline.META_CHANNELS]

    sample_vals = {}
    for rel in all_rel:
        if values_override is not None:
            mv = values_override.get(rel)
        else:
            mv = drc_pipeline.load_sample_transformed_values(controller, state, rel, channels,
                                                             af_state=af_state)
        if mv is None:
            continue
        if loaded is not None and values_override is None:
            loaded[rel] = mv
        if len(mv[0]) != len(np.asarray(cluster_labels[rel])):
            log.warning("compute_mfis: %s -- %d values vs %d labels; the run's events do not "
                        "match these values, sample left out", rel, len(mv[0]),
                        len(np.asarray(cluster_labels[rel])))
            continue
        sample_vals[rel] = mv          # (values (n,n_sel), names)

    # One pass per sample and channel: per-cluster sums / counts (float64)
    mfi_mats = {ch: np.full((len(all_rel), n_clusters), np.nan, dtype=float) for ch in channels}
    for i, rel in enumerate(all_rel):
        mv = sample_vals.get(rel)
        if mv is None:
            continue
        values, names = mv
        lab = np.asarray(cluster_labels[rel])
        in_range = (lab >= 0) & (lab < n_clusters)
        idx = lab[in_range].astype(np.intp)
        n = np.bincount(idx, minlength=n_clusters)
        present = n > 0
        for ch in channels:
            if ch not in names:
                continue
            sums = np.bincount(idx, weights=values[in_range, names.index(ch)],
                               minlength=n_clusters)
            mfi_mats[ch][i, present] = sums[present] / n[present]

    frames = []
    for ch in channels:
        frames.append(pd.DataFrame(
            mfi_mats[ch], index=all_rel,
            columns=[f'{_label_for(state, cl, names_override)}_{ch}' for cl in range(n_clusters)]))

    if not frames:
        log.warning("no MFI features could be built")
        return None
    full = pd.concat(frames, axis=1)
    log.info("MFI matrix: %s", full.shape)
    return full


def compute_sample_mfis(controller, state, all_rel, channels=None,
                        af_state=None, loaded: dict | None = None) -> pd.DataFrame | None:
    """
    Per-sample mean intensity of each selected channel across the WHOLE
    sample (every gated event), independent of cluster assignment.

    Companion to compute_mfis() (cluster x channel, for the per-cluster
    differential MFI test / Volcano) — that granularity isn't right for a
    heatmap: with dozens of clusters it either needs one heatmap per
    cluster or collapses to whichever single cluster happens to pass
    significance, and a cluster with zero cells in one group produces a
    fabricated near-zero MFI there rather than a real biological zero.
    This is the plain sample-level view for the MFI Heatmap instead:
    no cluster breakdown, no significance filtering.

    Each channel is on its own configured Transforms-tab scale (Logicle/
    biexponential/linear), same as compute_mfis().

    Returns (n_samples x n_channels), NaN where a sample has no events for
    a channel (rather than a fabricated 0).

    loaded: samples already loaded by compute_mfis() (same channels and
    af_state); the rest are loaded here.
    """
    log_stage(log, "SAMPLE MFI MATRIX")
    if channels is None:
        channels = [c for c in state.selected_channels if c not in drc_pipeline.META_CHANNELS]
    if not channels:
        log.warning("no sample-level MFI features could be built")
        return None

    sample_vals = {}
    for rel in all_rel:
        if loaded is not None and rel in loaded:
            mv = loaded[rel]
        else:
            mv = drc_pipeline.load_sample_transformed_values(controller, state, rel, channels,
                                                             af_state=af_state)
        if mv is not None:
            sample_vals[rel] = mv

    mfi_mat = np.full((len(all_rel), len(channels)), np.nan, dtype=float)
    for j, ch in enumerate(channels):
        for i, rel in enumerate(all_rel):
            mv = sample_vals.get(rel)
            if mv is None:
                continue
            values, names = mv
            if ch not in names:
                continue
            col = values[:, names.index(ch)]
            if len(col):
                mfi_mat[i, j] = float(np.mean(col))

    df = pd.DataFrame(mfi_mat, index=all_rel, columns=list(channels))
    log.info("sample MFI matrix: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Sample PCA
# ---------------------------------------------------------------------------

def compute_sample_pca(state, use_freq: bool, use_counts: bool, use_mfi: bool,
                       n_loadings: int = 10, tables: dict | None = None,
                       group_of: dict | None = None) -> dict | None:
    """
    Sample-level PCA over any combination of the per-sample feature
    matrices Run Statistics already computed (``state.freq_df`` /
    ``state.counts_df`` / ``state.mfi_df`` — samples x cluster-features,
    raw scale). Independent of "Viewing comparison": uses every sample
    across every group in ``state.stats_group_vec``, not one pairwise
    comparison at a time.

    Each requested matrix's columns are z-scored independently before
    concatenation, so frequency (%), count (raw events), and MFI
    (log1p-intensity) don't dominate one another on scale alone. A single
    joint 2-component PCA is then fit on the combined, standardized
    matrix.

    Loadings are scaled by sqrt(explained_variance) per axis (a
    correlation-biplot convention) and reduced to the top ``n_loadings``
    by 2-D vector length, so the arrows shown are the most differentiating
    variables rather than every feature in the (potentially huge)
    cluster x channel matrix.

    tables / group_of: optional {'freq', 'counts', 'mfi': samples × features}
    and {sample: group} used instead of the state's tables and groups (the
    trend run's tested values, in trend mode).

    Returns None if no requested source is available/populated, or if
    fewer than 2 samples or 2 non-degenerate (non-zero-variance) features
    remain. Otherwise a dict:
      'scores':   DataFrame (n_samples x ['PC1','PC2']), index=rel-path,
                  in state.stats_all_rel order.
      'loadings': DataFrame (<=n_loadings x ['PC1','PC2']), index=feature
                  label, already reduced to the top-N.
      'explained_variance_ratio': (float, float) — PC1, PC2.
      'groups':   list[str] aligned 1:1 to scores.index.
      'sources':  list[str] subset of ['Freq','Counts','MFI'] — which
                  matrices actually contributed (for the plot title).
    """
    log_stage(log, "SAMPLE PCA")

    if tables is None:
        tables = {'freq': state.freq_df, 'counts': state.counts_df, 'mfi': state.mfi_df}
    sources, frames = [], []
    for use, key, name in ((use_freq, 'freq', 'Freq'), (use_counts, 'counts', 'Counts'),
                           (use_mfi, 'mfi', 'MFI')):
        frame = tables.get(key)
        if use and frame is not None and not frame.empty:
            sources.append(name)
            frames.append(frame.add_prefix(f'{name}: '))

    if not frames:
        log.warning("Sample PCA: no requested source is available -- "
                   "run Statistics with Frequencies/Counts/MFIs checked first.")
        return None

    combined = pd.concat(frames, axis=1, join='inner')
    if combined.shape[0] < 2:
        log.warning("Sample PCA: fewer than 2 samples after aligning sources.")
        return None

    # Drop zero-variance columns -- guards the z-score division and keeps
    # a cluster with an identical value in every sample from contributing
    # a meaningless (but numerically NaN-producing) loading.
    stds = combined.std(axis=0, ddof=0)
    combined = combined[stds[stds > 1e-12].index]
    if combined.shape[1] < 2:
        log.warning("Sample PCA: fewer than 2 non-degenerate features to run PCA on.")
        return None

    means = combined.mean(axis=0)
    stds = combined.std(axis=0, ddof=0)
    # Missing MFIs (cluster absent from a sample) sit at the column mean.
    z = ((combined - means) / stds).fillna(0.0)

    n_comp = min(2, z.shape[0], z.shape[1])
    pca = PCA(n_components=n_comp)
    pcs = pca.fit_transform(z.values)

    if n_comp < 2:
        pcs = np.hstack([pcs, np.zeros((pcs.shape[0], 2 - n_comp))])
        explained = list(pca.explained_variance_ratio_) + [0.0] * (2 - n_comp)
    else:
        explained = list(pca.explained_variance_ratio_[:2])

    scores = pd.DataFrame(pcs[:, :2], index=combined.index, columns=['PC1', 'PC2'])

    # Correlation-biplot scaling: component loadings * sqrt(eigenvalue),
    # so arrow length reflects how much variance that feature explains on
    # each axis (not just the raw eigenvector direction).
    load_mat = np.zeros((z.shape[1], 2))
    for c in range(n_comp):
        load_mat[:, c] = pca.components_[c] * np.sqrt(max(pca.explained_variance_[c], 0.0))
    loadings = pd.DataFrame(load_mat, index=combined.columns, columns=['PC1', 'PC2'])

    magnitude = np.sqrt(loadings['PC1'] ** 2 + loadings['PC2'] ** 2)
    top_idx = magnitude.sort_values(ascending=False).head(max(1, int(n_loadings))).index
    loadings_top = loadings.loc[top_idx]

    group_by_rel = (group_of if group_of is not None
                    else dict(zip(state.stats_all_rel, state.stats_group_vec)))
    groups = [group_by_rel.get(rel, 'Unassigned') for rel in scores.index]

    log.info("Sample PCA: %s, %d samples, %d features (%d shown as loadings), "
             "PC1=%.1f%% PC2=%.1f%%",
             '+'.join(sources), scores.shape[0], combined.shape[1], len(loadings_top),
             explained[0] * 100, explained[1] * 100)

    return {
        'scores': scores,
        'loadings': loadings_top,
        'explained_variance_ratio': tuple(explained[:2]),
        'groups': groups,
        'sources': sources,
    }


# ---------------------------------------------------------------------------
# Composition views
# ---------------------------------------------------------------------------

def get_counts_table(controller, state, group_var: str = 'sample',
                     cluster_labels_override=None,
                     names_override: dict | None = None) -> pd.DataFrame:
    """
    Raw event counts per cluster (CyCONDOR's getTable(), counts variant).

    group_var: 'sample' → one row per sample (rel-path index).
               'group'  → one row per qualifying tested group (display
                          name), summed across that group's samples.
    Returns (n_rows × n_clusters) DataFrame, columns = cluster display
    labels. Also the natural building block for CSV export alongside the
    existing "Save Statistics CSVs" pattern.
    """
    log_stage(log, "COUNTS TABLE")
    group_rel = resolve_test_groups(controller, state, cluster_labels_override=cluster_labels_override)
    qualifying = [g for g in (state.testing_group_selection or state.group_names) if g in group_rel]
    all_rel = [rel for g in qualifying for rel in group_rel[g]]
    group_vec = [g for g in qualifying for _rel in group_rel[g]]
    n_clusters = n_clusters_from_labels(
        state, all_rel, cluster_labels_override=cluster_labels_override
    )
    cluster_labels = cluster_labels_override if cluster_labels_override is not None \
        else state.cluster_labels
    cols = [_label_for(state, cl, names_override) for cl in range(n_clusters)]

    per_sample = np.zeros((len(all_rel), n_clusters), dtype=int)
    for i, rel in enumerate(all_rel):
        labels = np.asarray(cluster_labels[rel])
        for cl in range(n_clusters):
            per_sample[i, cl] = int(np.sum(labels == cl))

    if group_var == 'sample':
        df = pd.DataFrame(per_sample, index=all_rel, columns=cols)
    elif group_var == 'group':
        masks = [np.array([g == name for g in group_vec]) for name in qualifying]
        summed = np.stack([
            per_sample[m].sum(axis=0) if m.any() else np.zeros(n_clusters)
            for m in masks
        ])
        df = pd.DataFrame(summed, index=qualifying, columns=cols)
    else:
        raise ValueError(f"group_var must be 'sample' or 'group', got {group_var!r}")

    log.info("counts table (%s): %s", group_var, df.shape)
    return df


def get_frequency_table(controller, state, group_var: str = 'sample',
                        cluster_labels_override=None,
                        names_override: dict | None = None) -> pd.DataFrame:
    """
    Per-row % of events per cluster (CyCONDOR's getTable(), frequency
    variant). Same shape/grouping as get_counts_table, row-normalized to
    100%.
    """
    counts_df = get_counts_table(
        controller, state, group_var=group_var,
        cluster_labels_override=cluster_labels_override,
        names_override=names_override,
    )
    totals = counts_df.sum(axis=1).replace(0, 1)   # guard empty rows
    freq_df = counts_df.div(totals, axis=0) * 100.0
    log.info("frequency table (%s): %s", group_var, freq_df.shape)
    return freq_df


_TOO_FEW_CLUSTERS_HINT = (
    "This usually means the current clustering run produced too few clusters "
    "for differential testing. Increase clustering granularity (e.g. lower "
    "HDBSCAN's min_cluster_size) and re-run, or test a space/run with more "
    "clusters."
)


# ---------------------------------------------------------------------------
# Cluster renames
# ---------------------------------------------------------------------------

def stored_cluster_names(state) -> dict[int, str]:
    """
    {cluster id: name} used by the stored statistics results: the names
    recorded when they were computed (state.stats_cluster_names), else, for
    results saved before those were recorded, read from the tables
    themselves (column i of the frequency and counts tables is cluster i;
    MFI results carry 'cluster_id' and 'cluster').
    """
    recorded = getattr(state, 'stats_cluster_names', None)
    if recorded:
        return dict(recorded)
    for df in (state.freq_df, state.counts_df):
        if df is not None and not df.empty:
            return {cl: str(name) for cl, name in enumerate(df.columns)}
    res = state.mfi_results
    if res is not None and not res.empty and {'cluster_id', 'cluster'} <= set(res.columns):
        pairs = res[['cluster_id', 'cluster']].drop_duplicates()
        return {int(cl): str(name) for cl, name in pairs.itertuples(index=False)}
    return {}


def cluster_renames(state, stored: dict[int, str], names: dict) -> dict[str, str]:
    """
    {old name: new name} for the clusters in *stored* ({cluster id: name in
    a stored table}) whose display name under *names* (a run's own names
    dict) differs. Raises ValueError when the new names are not unique.
    """
    current = {cl: _label_for(state, cl, names) for cl in stored}
    renames = {stored[cl]: current[cl] for cl in stored if stored[cl] != current[cl]}
    if renames and len(set(current.values())) != len(current):
        raise ValueError("Two clusters of this run share a name.")
    return renames


def _renamed_values(df: pd.DataFrame | None, column: str,
                    renames: dict[str, str]) -> pd.DataFrame | None:
    """A copy of *df* with the values of *column* renamed; *df* itself when
    it is None or has no such column."""
    if df is None or column not in df.columns:
        return df
    df = df.copy()
    df[column] = df[column].map(lambda value: renames.get(value, value))
    return df


def _stored_mfi_channels(state) -> list[str]:
    """Channels of the stored MFI tables, in order."""
    channels = []
    if state.mfi_sample_df is not None:
        channels.extend(str(ch) for ch in state.mfi_sample_df.columns)
    res = state.mfi_results
    if res is not None and 'channel' in res.columns:
        channels.extend(str(ch) for ch in res['channel'].dropna().unique())
    return list(dict.fromkeys(channels))


def relabel_stats_results(state, names: dict) -> dict[str, str]:
    """
    Give the stored statistics results the current cluster names of their
    run (*names*, the run's own names dict) without refitting: the
    frequency, counts and MFI tables, the differential results, the Sample
    PCA loadings and the predictive-analysis features.

    Returns {old name: new name}, empty when nothing changed. Raises
    ValueError, changing nothing, when the new names are not unique.
    """
    stored = stored_cluster_names(state)
    renames = cluster_renames(state, stored, names)
    if not renames:
        return {}
    mfi_renames = {f'{old}_{ch}': f'{new}_{ch}'
                   for old, new in renames.items() for ch in _stored_mfi_channels(state)}
    # Features of the Sample PCA and the predictive analysis carry a source prefix.
    prefixed = {f'{prefix}{old}': f'{prefix}{new}'
                for prefix in ('Freq: ', 'Counts: ') for old, new in renames.items()}
    prefixed.update({f'MFI: {old}': f'MFI: {new}' for old, new in mfi_renames.items()})

    if state.freq_df is not None:
        state.freq_df = state.freq_df.rename(columns=renames)
    if state.counts_df is not None:
        state.counts_df = state.counts_df.rename(columns=renames)
    if state.mfi_df is not None:
        state.mfi_df = state.mfi_df.rename(columns=mfi_renames)
    state.freq_results = _renamed_values(state.freq_results, 'feature', renames)
    state.counts_results = _renamed_values(state.counts_results, 'feature', renames)
    state.mfi_results = _renamed_values(
        _renamed_values(state.mfi_results, 'feature', mfi_renames), 'cluster', renames)
    if state.pca_loadings_df is not None:
        state.pca_loadings_df = state.pca_loadings_df.rename(index=prefixed)
    predictive = getattr(state, 'predictive_result', None)
    if predictive and predictive.get('importance') is not None:
        state.predictive_result = dict(
            predictive, importance=_renamed_values(predictive['importance'], 'feature', prefixed))
    state.stats_cluster_names = {cl: _label_for(state, cl, names) for cl in stored}
    log.info("statistics results relabelled: %s", renames)
    return renames


def relabel_composition(state, names: dict) -> dict[str, str]:
    """
    Give the stored Composition table the current cluster names of its run
    (*names*); its column i is cluster i. Returns
    {old name: new name}; raises ValueError, changing nothing, when the new
    names are not unique.
    """
    df = state.composition_df
    if df is None or df.empty:
        return {}
    stored = {cl: str(name) for cl, name in enumerate(df.columns)}
    renames = cluster_renames(state, stored, names)
    if renames:
        state.composition_df = df.rename(columns=renames)
        state.composition_names = dict(names)
    return renames


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _normalisation(state) -> str:
    """Library-size method of the abundance tests ('total' or 'robust')."""
    method = getattr(state, 'stats_normalisation', 'total')
    return method if method in ('total', 'robust') else 'total'


def run_statistics(controller, state, run_freq: bool, run_mfi: bool,
                   pval_threshold: float, fc_threshold: float,
                   cluster_labels_override=None,
                   include_type_markers: bool = False,
                   names_override: dict | None = None,
                   run_counts: bool = False,
                   af_state=None,
                   fdr_scope: str = 'global',
                   covariate_names: list[str] | None = None,
                   use_treat: bool = True,
                   mfi_threshold: float | None = None,
                   mfi_hierarchical: bool = True,
                   align_run: dict | None = None,
                   align_mode: str | None = None):
    """
    Run the requested differential tests. Writes results onto ``state`` and
    returns ``(freq_results, mfi_results, counts_results)`` (any may be
    None).

    align_run / align_mode: a hydrated archived clustering run that is not
    in sync with the current gating and unmixing, and how it can be matched
    to the current events (drc_run_align.live_join_mode).  Its labels are
    then matched to the current gated events by event key before testing
    (see drc_run_align.statistics_inputs); state.stats_alignment_note says
    how well they matched.

    Resolves state.testing_group_selection (falling back to every defined
    group), builds the contrast list from state.contrast_mode/
    reference_group, the pairing vector (if state.paired and
    state.pairing_variable names a covariates column) and the adjustment
    covariates, all shared across freq/counts/mfi so the three tests
    report the same comparisons from the same design.

    fc_threshold  : |log2 fold change| threshold for Frequencies and Counts.
    mfi_threshold : |difference| threshold for MFIs, in transformed units
                    (defaults to fc_threshold).
    covariate_names: state.covariates columns to adjust for.
    use_treat     : test against the thresholds (TREAT) rather than zero.
    mfi_hierarchical: cluster-first testing of MFIs.
    include_type_markers: when False, the MFI branch tests 'state'-role
        channels only. When True, tests every selected channel.
    run_counts: also run the negative-binomial GLM on raw counts, tested by
        the quasi-likelihood F test (state.stats_counts_method, 'ql').

    The Frequency and Counts library sizes follow state.stats_normalisation
    ('total' or 'robust', differential_stats.library_sizes).
    """
    log_stage(log, "DIFFERENTIAL STATISTICS")
    if mfi_threshold is None:
        mfi_threshold = fc_threshold

    values_override = None
    state.stats_alignment_note = ''
    state.stats_population_of = {}
    if align_run is not None and align_mode is not None:
        cluster_labels_override, values_override, note = drc_run_align.statistics_inputs(
            controller, state, align_run, align_mode, af_state=af_state)
        state.stats_alignment_note = note

    split = None
    if getattr(state, 'population_split', False):
        if values_override is not None:
            raise RuntimeError(
                "This clustering run was made before event keys and no longer matches the "
                "current gating, so its cells cannot be assigned to populations. Re-run the "
                "clustering to compare populations within samples.")
        split = split_by_population(
            controller, state,
            cluster_labels_override if cluster_labels_override is not None
            else state.cluster_labels,
            af_state=af_state)
        cluster_labels_override = split.labels
        state.stats_population_of = {key: list(v) for key, v in split.population_of.items()}
        state.stats_alignment_note = " ".join(
            t for t in (state.stats_alignment_note, split.note) if t)
        group_rel = {g: keys for g, keys in split.groups.items() if len(keys) >= 3}
        for g, keys in split.groups.items():
            if len(keys) < 3:
                log.info("population %r dropped from testing (only %d samples with events)",
                         g, len(keys))
        if len(group_rel) < 2:
            raise RuntimeError(
                "Need at least 2 populations with events in 3 or more samples each. "
                "Qualifying: " + (", ".join(f"{g}={len(k)}" for g, k in split.groups.items())
                                  or "none"))
        qualifying = [g for g in state.population_gates if g in group_rel]
        reference = state.population_reference \
            if state.population_reference in qualifying else qualifying[0]
        contrasts = build_contrasts(qualifying, state.contrast_mode, reference)
    else:
        group_rel = resolve_test_groups(controller, state,
                                        cluster_labels_override=cluster_labels_override)
        qualifying = [g for g in (state.testing_group_selection or state.group_names)
                      if g in group_rel]
        contrasts = build_contrasts(qualifying, state.contrast_mode, state.reference_group)

    all_rel = [rel for g in qualifying for rel in group_rel[g]]
    group_vec = [g for g in qualifying for _rel in group_rel[g]]

    pairing_vec = None
    if split is not None:
        # Each sample is its own block: its populations are compared with
        # each other. Sample-level covariates are constant within a block,
        # so the pairing already accounts for them.
        pairing_vec = [split.population_of[key][0] for key in all_rel]
    elif state.paired and state.pairing_variable and state.covariates is not None \
            and state.pairing_variable in state.covariates.columns:
        cov = state.covariates[state.pairing_variable]
        # A sample without a pairing value (blank, NA) is unpaired: it gets
        # a blocking level of its own rather than failing the whole run.
        pairing_vec = [str(cov[rel]).strip()
                       if rel in cov.index and not _ds.is_missing_value(cov[rel])
                       else f"__unpaired_{rel}" for rel in all_rel]

    adjust = [] if split is not None else [
        c for c in (covariate_names or [])
        if not (pairing_vec is not None and c == state.pairing_variable)]
    covariates = covariate_frame(state, all_rel, adjust)
    if covariates is not None:
        log.info("adjusting for covariates: %s", list(covariates.columns))

    n_clusters = n_clusters_from_labels(
        state, all_rel, cluster_labels_override=cluster_labels_override
    )
    # The names the result tables use, so a later rename can be applied to them.
    state.stats_cluster_names = {cl: _label_for(state, cl, names_override)
                                 for cl in range(n_clusters)}

    freq_results = mfi_results = counts_results = None

    # Group metadata for the heatmap and the 'Viewing comparison' selector.
    state.stats_all_rel = all_rel
    state.stats_group_vec = group_vec
    state.stats_comparisons = contrasts

    if run_freq or run_counts:
        counts_df = compute_counts(
            state, all_rel, n_clusters,
            cluster_labels_override=cluster_labels_override,
            names_override=names_override,
        )
        sparse = set(sparse_features(counts_df, MIN_NONZERO_SAMPLES))
        if sparse:
            log.info("abundance tests skip %d cluster(s) with events in fewer than %d samples: %s",
                     len(sparse), MIN_NONZERO_SAMPLES, ", ".join(sorted(sparse)))
        tested = [c for c in counts_df.columns if c not in sparse]

    if run_freq:
        freq_df = compute_frequencies(
            state, all_rel, n_clusters,
            cluster_labels_override=cluster_labels_override,
            names_override=names_override,
        )
        totals = sample_event_totals(state, all_rel, cluster_labels_override)
        totals = _ds.library_sizes(counts_df[tested], totals, _normalisation(state))
        freq_results = run_moderated(
            log2_frequencies(counts_df[tested], totals, share_prior=True), group_vec, contrasts,
            state.contrast_mode,
            pval_threshold, fc_threshold, pairing_vec=pairing_vec, fdr_scope=fdr_scope,
            covariates=covariates, use_treat=use_treat,
            too_few_hint=_TOO_FEW_CLUSTERS_HINT,
        )
        state.freq_results = freq_results
        state.freq_df = freq_df          # % per cluster (display)

    if run_counts:
        # The library-size offset is each sample's total over all clusters,
        # including those left out of the test (robustly rescaled on request).
        row_totals = pd.Series(
            _ds.library_sizes(counts_df[tested], counts_df.sum(axis=1).values, _normalisation(state)),
            index=counts_df.index)
        offset_df = pd.DataFrame({c: row_totals for c in tested}, index=counts_df.index)
        counts_results = run_glm_counts(counts_df[tested], group_vec, contrasts,
                                        state.contrast_mode, pval_threshold, fc_threshold,
                                        pairing_vec=pairing_vec, fdr_scope=fdr_scope,
                                        covariates=covariates, use_treat=use_treat,
                                        offset_df=offset_df,
                                        method=getattr(state, 'stats_counts_method', 'ql'))
        state.counts_results = counts_results
        state.counts_df = counts_df      # raw (samples × clusters) counts

    if run_mfi:
        mfi_channels = resolve_mfi_channels(state, include_type_markers=include_type_markers)
        if not mfi_channels:
            log.warning(
                "MFI testing: no 'state'-role channels selected — check "
                "'Include clustering (type) markers' or assign roles."
            )
        loaded = {}       # each sample read once for both MFI matrices
        if split is not None:
            values_override = population_values(controller, state, split, mfi_channels,
                                                af_state=af_state)
            loaded = dict(values_override)
        mfi_df = compute_mfis(
            controller, state, all_rel, n_clusters,
            cluster_labels_override=cluster_labels_override,
            channels=mfi_channels,
            names_override=names_override,
            af_state=af_state,
            values_override=values_override,
            loaded=loaded,
        )
        if mfi_df is not None:
            meta = mfi_feature_table(state, n_clusters, mfi_channels, names_override)
            meta = meta[meta['feature'].isin(mfi_df.columns)].drop_duplicates('feature')
            mfi_results = run_moderated(
                mfi_df, group_vec, contrasts, state.contrast_mode,
                pval_threshold, mfi_threshold, pairing_vec=pairing_vec, fdr_scope=fdr_scope,
                covariates=covariates, use_treat=use_treat,
                feature_meta=meta, hierarchical=mfi_hierarchical,
                too_few_hint=_TOO_FEW_CLUSTERS_HINT,
            )
            state.mfi_results = mfi_results
            state.mfi_df = mfi_df        # raw (samples × cluster·channel) MFIs

        state.mfi_sample_df = compute_sample_mfis(
            controller, state, all_rel, channels=mfi_channels, af_state=af_state,
            loaded=loaded,
        )

    notes = []
    for frame in (freq_results, mfi_results, counts_results):
        for note in (frame.attrs.get('notes', []) if frame is not None else []):
            if note not in notes:
                notes.append(note)
    if notes:
        state.stats_alignment_note = " ".join(
            t for t in [state.stats_alignment_note] + notes if t)
    return freq_results, mfi_results, counts_results


# ---------------------------------------------------------------------------
# Trends with a numeric covariate
# ---------------------------------------------------------------------------

_TREND_KEYS = ('freq', 'counts', 'mfi')


def trend_samples(controller, state, cluster_labels=None) -> tuple[list[str], list[str]]:
    """Rel paths and groups of the samples a trend test uses: the samples of
    the groups ticked in Groups to Test (every group when none is ticked)
    that have cluster labels, group by group."""
    labels = cluster_labels if cluster_labels is not None else state.cluster_labels
    raw_subdir = controller.experiment.settings['raw']['raw_samples_subdirectory']
    all_rel, group_vec = [], []
    for name in (state.testing_group_selection or list(state.group_names)):
        for sp, g in state.sample_groups.items():
            rel = _rel_path(sp, raw_subdir)
            if g == name and rel in labels and rel not in all_rel:
                all_rel.append(rel)
                group_vec.append(name)
    return all_rel, group_vec


def samples_outside_trend(controller, state, trend_rel: list[str],
                          cluster_labels=None) -> tuple[list[str], list[str]]:
    """Rel paths and groups of the labelled samples of any group that are not
    among *trend_rel*: the samples an immune age model may predict besides
    those of the trend run."""
    labels = cluster_labels if cluster_labels is not None else state.cluster_labels
    raw_subdir = controller.experiment.settings['raw']['raw_samples_subdirectory']
    seen = set(trend_rel)
    rels, groups = [], []
    for sp, g in state.sample_groups.items():
        rel = _rel_path(sp, raw_subdir)
        if g and g in state.group_names and rel in labels and rel not in seen:
            seen.add(rel)
            rels.append(rel)
            groups.append(g)
    return rels, groups


def trend_values(state, all_rel: list[str]) -> list[float]:
    """The trend variable (state.trend_variable) of each sample; NaN where
    it has no number."""
    cov = state.covariates
    var = state.trend_variable
    out = []
    for rel in all_rel:
        value = cov[var].get(rel) if cov is not None and var in cov.columns else None
        try:
            out.append(float('nan') if value is None or _ds.is_missing_value(value)
                       else float(str(value).strip()))
        except ValueError:
            out.append(float('nan'))
    return out


def _outside_frequencies(controller, state, all_rel, counts_df, tested, n_clusters, labels,
                         names_override, mean_total: float) -> dict:
    """
    The tested log2 cluster frequencies of the labelled samples outside the
    trend run (other groups, or samples without a value of the variable),
    for the immune age model to predict: {'other_freq_data', 'other_groups',
    'other_time'}. Library sizes are worked out over the trend's and these
    samples together, so they are on the same footing as the trend's, and
    the prior count is set by the trend's mean library size *mean_total*.
    """
    rels, groups = samples_outside_trend(controller, state, all_rel, labels)
    if not rels:
        return {'other_freq_data': None, 'other_groups': {}, 'other_time': {}}
    other = compute_counts(state, rels, n_clusters, cluster_labels_override=labels,
                           names_override=names_override)
    both = pd.concat([counts_df[tested], other[tested]])
    totals = sample_event_totals(state, list(all_rel) + rels, labels)
    library = np.asarray(_ds.library_sizes(both, totals, _normalisation(state)), dtype=float)
    data = log2_frequencies(other[tested], library[len(all_rel):], share_prior=True,
                            mean_total=mean_total)
    return {'other_freq_data': data, 'other_groups': dict(zip(rels, groups)),
            'other_time': dict(zip(rels, trend_values(state, rels)))}


def run_trend_statistics(controller, state, run_freq: bool, run_mfi: bool,
                         pval_threshold: float, cluster_labels_override=None,
                         include_type_markers: bool = False, names_override: dict | None = None,
                         run_counts: bool = False, af_state=None,
                         covariate_names: list[str] | None = None,
                         align_run: dict | None = None, align_mode: str | None = None) -> dict:
    """
    Trends of cluster frequencies, counts and MFIs with the numeric
    covariate state.trend_variable (``trend_stats.run_trend``: a curve F
    test and a slope test on the moderated linear model).

    The feature tables are those of run_statistics: log2 % of each
    sample's events (library size from state.stats_normalisation),
    counts as log2 odds within that library size (moderated; the
    quasi-likelihood counts test gave far too many false trends), and
    per-cluster MFIs. The samples are those of trend_samples; settings come
    from the state's trend_* fields. With state.trend_interaction exactly
    two groups are ticked and the reference group comes first.

    Returns {'samples', 'group_vec', 'time', 'variable', 'unit', 'freq' /
    'counts' / 'mfi' (TrendResult or None), 'freq_data' / 'counts_data' /
    'mfi_data' (the tested values, samples × features), 'cluster_names',
    'settings', 'notes'}.
    """
    from honeychrome.controller_components import trend_stats as ts

    log_stage(log, "TREND STATISTICS")
    values_override = None
    notes = []
    if align_run is not None and align_mode is not None:
        cluster_labels_override, values_override, note = drc_run_align.statistics_inputs(
            controller, state, align_run, align_mode, af_state=af_state)
        if note:
            notes.append(note)
    if getattr(state, 'population_split', False):
        raise RuntimeError("A trend test compares samples: untick 'Compare populations within "
                           "samples' to test one.")
    labels = cluster_labels_override if cluster_labels_override is not None \
        else state.cluster_labels
    variable = state.trend_variable
    if not variable or state.covariates is None or variable not in state.covariates.columns:
        raise RuntimeError("Choose a numeric covariate for the trend (tick Numeric under it in "
                           "the samples table).")
    all_rel, group_vec = trend_samples(controller, state, labels)
    order = list(dict.fromkeys(group_vec))
    if state.trend_interaction and len(order) != 2:
        raise RuntimeError("Tick exactly two groups in Groups to Test to ask whether they "
                           "change differently.")
    baseline = state.reference_group if state.reference_group in order else (order or [''])[0]
    time = trend_values(state, all_rel)

    pairing_vec = None
    if state.paired and state.pairing_variable and state.pairing_variable in state.covariates.columns:
        cov = state.covariates[state.pairing_variable]
        pairing_vec = [str(cov[rel]).strip()
                       if rel in cov.index and not _ds.is_missing_value(cov[rel])
                       else f"__unpaired_{rel}" for rel in all_rel]
    adjust = [c for c in (covariate_names or []) if c != variable
              and not (pairing_vec is not None and c == state.pairing_variable)]
    covariates = covariate_frame(state, all_rel, adjust)
    n_clusters = n_clusters_from_labels(state, all_rel, cluster_labels_override=labels)
    common = dict(variable=variable, unit=state.trend_unit, scale=state.trend_scale,
                  shape=state.trend_shape, k=state.trend_df, group_vec=group_vec,
                  interaction=state.trend_interaction, baseline=baseline, pairing_vec=pairing_vec,
                  covariates=covariates, pval_threshold=pval_threshold,
                  min_r2=state.trend_min_r2, call=state.trend_call)
    out = {'samples': all_rel, 'group_vec': group_vec, 'time': time, 'variable': variable,
           'unit': state.trend_unit, 'freq': None, 'counts': None, 'mfi': None,
           'freq_data': None, 'counts_data': None, 'mfi_data': None, 'cluster_counts': None,
           'cluster_names': {cl: _label_for(state, cl, names_override) for cl in range(n_clusters)},
           'settings': {}, 'notes': notes}

    if run_freq or run_counts:
        counts_df = compute_counts(state, all_rel, n_clusters, cluster_labels_override=labels,
                                   names_override=names_override)
        out['cluster_counts'] = counts_df
        sparse = set(sparse_features(counts_df, MIN_NONZERO_SAMPLES))
        tested = [c for c in counts_df.columns if c not in sparse]
        if sparse:
            notes.append(f"{len(sparse)} cluster(s) with events in fewer than "
                         f"{MIN_NONZERO_SAMPLES} samples were left out.")
        library = _ds.library_sizes(counts_df[tested],
                                    sample_event_totals(state, all_rel, labels), _normalisation(state))
        if run_freq:
            data = log2_frequencies(counts_df[tested], library, share_prior=True)
            out['freq'] = ts.run_trend(data, time, **common)
            out['freq_data'] = data
            out.update(_outside_frequencies(controller, state, all_rel, counts_df, tested,
                                            n_clusters, labels, names_override,
                                            float(np.mean(library))))
        if run_counts:
            denominators = pd.DataFrame({c: library for c in tested}, index=counts_df.index)
            data = _ds.logit_proportions(counts_df[tested], denominators, share_prior=True)
            out['counts'] = ts.run_trend(data, time, **common)
            out['counts_data'] = data
            notes.append("Counts are tested as log odds within each sample's library size "
                         "(moderated linear model).")
    if run_mfi:
        channels = resolve_mfi_channels(state, include_type_markers=include_type_markers)
        mfi_df = compute_mfis(controller, state, all_rel, n_clusters,
                              cluster_labels_override=labels, channels=channels,
                              names_override=names_override, af_state=af_state,
                              values_override=values_override, loaded={})
        if mfi_df is not None and mfi_df.shape[1] >= 3:
            meta = mfi_feature_table(state, n_clusters, channels, names_override)
            meta = meta[meta['feature'].isin(mfi_df.columns)].drop_duplicates('feature')
            out['mfi'] = ts.run_trend(mfi_df, time, feature_meta=meta, **common)
            out['mfi_data'] = mfi_df
    for key in _TREND_KEYS:
        if out[key] is not None:
            out[key] = ts.add_patterns(out[key], out[f'{key}_data'], time,
                                       n_patterns=int(getattr(state, 'trend_patterns', 0)),
                                       **common)
    for key in _TREND_KEYS:
        res = out[key]
        if res is None:
            continue
        if not out['settings']:
            out['settings'] = dict(res.settings)
        for note in res.notes:
            if note not in notes:
                notes.append(note)
    if not any(out[k] is not None for k in _TREND_KEYS):
        raise RuntimeError("Nothing could be tested for a trend: tick at least one test.")
    return out


def trend_to_bundle(trend: dict) -> dict:
    """*trend* (a run_trend_statistics result) with each TrendResult as a
    plain dict of tables and arrays, for drc_persist."""
    out = dict(trend)
    for key in _TREND_KEYS:
        res = trend.get(key)
        if res is None:
            out[key] = None
            continue
        out[key] = {'table': res.table, 'grid': np.asarray(res.grid),
                    'curves': np.asarray(res.curves),
                    'notes': list(res.notes), 'settings': dict(res.settings)}
        for name in ('curve_se', 'group_curves', 'group_curve_se'):
            value = getattr(res, name)
            if value is not None:
                out[key][name] = np.asarray(value, dtype=float)
    return out


def trend_from_bundle(payload: dict) -> dict:
    """The run_trend_statistics result that trend_to_bundle stored."""
    from honeychrome.controller_components import trend_stats as ts

    out = dict(payload)
    for key in _TREND_KEYS:
        obj = payload.get(key)

        def optional(name):
            value = (obj or {}).get(name)
            return None if value is None else np.asarray(value, dtype=float)

        out[key] = None if not obj else ts.TrendResult(
            table=obj['table'], grid=np.asarray(obj['grid'], dtype=float),
            curves=np.asarray(obj['curves'], dtype=float), notes=list(obj.get('notes') or []),
            settings=dict(obj.get('settings') or {}), curve_se=optional('curve_se'),
            group_curves=optional('group_curves'), group_curve_se=optional('group_curve_se'))
    out['cluster_names'] = {int(cl): str(name)
                            for cl, name in (payload.get('cluster_names') or {}).items()}
    return out


COMPOSITION_TOP_CLUSTERS = 12


def age_model_matrix(trend: dict, use_freq: bool = True, use_mfi: bool = False) -> pd.DataFrame:
    """Samples × features of the trend run for the immune age model: the
    tested log2 cluster frequencies ('Freq: <cluster>') and/or cluster ×
    marker MFIs ('MFI: <cluster>_<channel>')."""
    frames = []
    if use_freq and trend.get('freq_data') is not None:
        frames.append(trend['freq_data'].astype(float).add_prefix('Freq: '))
    if use_mfi and trend.get('mfi_data') is not None:
        frames.append(trend['mfi_data'].astype(float).add_prefix('MFI: '))
    if not frames:
        return pd.DataFrame(index=list(trend.get('samples') or []))
    return pd.concat(frames, axis=1)


def trend_composition(trend: dict, keep: int = COMPOSITION_TOP_CLUSTERS) -> list[dict]:
    """
    The cluster make-up of the trend run's samples against its variable
    (``trend_stats.composition_panels``): the *keep* clusters with the
    largest mean share, the rest added together as 'Other clusters'.
    Raises RuntimeError when the run kept no cluster counts (frequencies
    and counts both unticked).
    """
    from honeychrome.controller_components import trend_stats as ts

    counts = trend.get('cluster_counts')
    res = next((trend.get(k) for k in _TREND_KEYS if trend.get(k) is not None), None)
    if counts is None or res is None:
        raise RuntimeError("Run the trend test with frequencies or counts ticked first.")
    counts = ts.merge_small_parts(counts.astype(float), keep, other='Other clusters')
    return ts.composition_panels(counts, dict(zip(trend.get('samples') or [], trend.get('time') or [])),
                                 dict(zip(trend.get('samples') or [], trend.get('group_vec') or [])),
                                 res.settings or {}, variable=trend.get('variable') or '',
                                 unit=trend.get('unit') or '')


def reapply_trend_thresholds(trend: dict, pval_threshold: float, min_r2: float, call: str,
                             patterns: int = 0) -> dict:
    """*trend* with its calls redone for new thresholds and its clusters
    regrouped into *patterns* shape patterns (0: the number chosen from
    stability at the run), without refitting."""
    from honeychrome.controller_components import trend_stats as ts

    out = dict(trend)
    for key in _TREND_KEYS:
        res = out.get(key)
        if res is not None:
            out[key] = ts.recall(res, pval_threshold, min_r2, call, n_patterns=patterns)
    out['settings'] = dict(out.get('settings') or {}, pval_threshold=pval_threshold,
                           min_r2=min_r2, call=call)
    return out


def relabel_trend_results(state, names: dict) -> dict[str, str]:
    """
    Give the stored trend results (state.trend_results) the current cluster
    names of their run (*names*) without refitting. Returns {old name: new
    name}; raises ValueError, changing nothing, when the new names are not
    unique.
    """
    from dataclasses import replace

    trend = state.trend_results
    if not trend:
        return {}
    stored = {int(cl): str(name) for cl, name in (trend.get('cluster_names') or {}).items()}
    renames = cluster_renames(state, stored, names)
    if not renames:
        return {}
    out = dict(trend)
    mfi = trend.get('mfi')
    channels = ([str(ch) for ch in mfi.table['channel'].dropna().unique()]
                if mfi is not None and 'channel' in mfi.table.columns else [])
    mfi_renames = {f'{old}_{ch}': f'{new}_{ch}' for old, new in renames.items() for ch in channels}
    for key, mapping in (('freq', renames), ('counts', renames), ('mfi', mfi_renames)):
        res = trend.get(key)
        if res is not None:
            table = _renamed_values(res.table, 'feature', mapping)
            if key == 'mfi':
                table = _renamed_values(table, 'cluster', renames)
            out[key] = replace(res, table=table)
        data = trend.get(f'{key}_data')
        if data is not None:
            out[f'{key}_data'] = data.rename(columns=mapping)
    if trend.get('other_freq_data') is not None:
        out['other_freq_data'] = trend['other_freq_data'].rename(columns=renames)
    if trend.get('cluster_counts') is not None:
        out['cluster_counts'] = trend['cluster_counts'].rename(columns=renames)
    age = trend.get('age_model')
    if age and age.get('importance') is not None:
        feature_renames = {f'Freq: {old}': f'Freq: {new}' for old, new in renames.items()}
        feature_renames.update({f'MFI: {old}': f'MFI: {new}' for old, new in mfi_renames.items()})
        out['age_model'] = dict(age, importance=_renamed_values(age['importance'], 'feature',
                                                                feature_renames))
    out['cluster_names'] = {cl: _label_for(state, cl, names) for cl in stored}
    state.trend_results = out
    log.info("trend results relabelled: %s", renames)
    return renames
